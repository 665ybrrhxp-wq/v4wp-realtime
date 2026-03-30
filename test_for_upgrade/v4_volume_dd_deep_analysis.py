"""
Volume-Dynamic DD Gate — 심층 분석
===================================
이전 실험 결과를 더 깊이 파고들어:
1. 종목별 성과 차이 (어떤 종목에서 Dynamic이 유리한가?)
2. 거래량 증가 기간 vs 수익률 상관관계 (패턴 존재?)
3. 시기별 분석 (최근 vs 과거)
4. DD lookback 구간별 성과 (짧은 surge vs 긴 surge)
5. 차단 신호 상세 분석 (어떤 특성의 신호가 잘못 차단되는가?)
6. Dynamic-C(VMA)에 집중 분석 (유일한 개선 후보)
"""
import sys, os, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import yfinance as yf
from real_market_backtest import (
    calc_v4_score, calc_v4_subindicators,
    detect_signal_events, build_price_filter, smooth_earnings_volume,
    calc_pv_divergence,
)

# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════
TICKERS = {
    'TSLA': 'Tech', 'NVDA': 'Tech', 'AVGO': 'Tech',
    'AMZN': 'Tech', 'GOOGL': 'Tech', 'PLTR': 'Tech',
    'COIN': 'Fintech', 'HOOD': 'Fintech', 'SOFI': 'Fintech',
    'HIMS': 'Growth', 'RKLB': 'Growth', 'JOBY': 'Growth',
    'PGY': 'Growth',
    'IONQ': 'Quantum', 'PL': 'Space', 'ASTS': 'Space',
    'QQQ': 'Benchmark', 'VOO': 'Benchmark',
}

SIGNAL_TH = 0.05
COOLDOWN = 5
ER_Q = 80
ATR_Q = 40
LOOKBACK = 252
DIVGATE = 3

BASELINE_DD_LB = 20
BASELINE_DD_TH = 0.03
MIN_LOOKBACK = 5
MAX_LOOKBACK = 60
DEFAULT_LOOKBACK = 20

HORIZONS = [10, 20, 30, 60, 90]


# ═══════════════════════════════════════════════════════════
# Helper Functions (이전 스크립트에서 가져옴)
# ═══════════════════════════════════════════════════════════
def download_ticker(ticker, years=5):
    t = yf.Ticker(ticker)
    df = t.history(period=f'{years}y', auto_adjust=True)
    if df is None or len(df) == 0:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.dropna(subset=['Close', 'Volume'])
    df = df[df['Close'] > 0]
    return df


def calc_vol_surge_days_vnorm(df):
    v_norm = df['Volume'] / df['Volume'].rolling(20).mean()
    v_norm = v_norm.fillna(0)
    n = len(df)
    streak = np.zeros(n)
    for i in range(n):
        if v_norm.iloc[i] > 1.0:
            streak[i] = (streak[i-1] + 1) if i > 0 else 1
    return streak


def calc_vol_surge_days_sdiv(df, w=20):
    pv_div = calc_pv_divergence(df, w)
    raw_div = np.array([np.clip(pv_div.iloc[i] / 3, -1, 1) for i in range(len(df))])
    n = len(df)
    streak = np.zeros(n)
    for i in range(n):
        if raw_div[i] > 0:
            streak[i] = (streak[i-1] + 1) if i > 0 else 1
    return streak


def calc_vol_surge_days_vma(df):
    vma = df['Volume'].rolling(20).mean()
    vma_diff = vma.diff().fillna(0)
    n = len(df)
    streak = np.zeros(n)
    for i in range(n):
        if vma_diff.iloc[i] > 0:
            streak[i] = (streak[i-1] + 1) if i > 0 else 1
    return streak


def get_dynamic_lookback(surge_days_at_idx):
    if surge_days_at_idx < MIN_LOOKBACK:
        return DEFAULT_LOOKBACK
    return int(min(surge_days_at_idx, MAX_LOOKBACK))


# ═══════════════════════════════════════════════════════════
# 전체 신호 데이터 수집 (모든 정보 포함)
# ═══════════════════════════════════════════════════════════
def collect_all_signals(ticker, sector):
    df = download_ticker(ticker, years=5)
    if df is None or len(df) < 252:
        return None

    df = smooth_earnings_volume(df, ticker)
    score = calc_v4_score(df, w=20, divgate_days=DIVGATE)
    sub = calc_v4_subindicators(df, w=20, divgate_days=DIVGATE)

    events = detect_signal_events(score, th=SIGNAL_TH, cooldown=COOLDOWN)
    pf = build_price_filter(df, er_q=ER_Q, atr_q=ATR_Q, lookback=LOOKBACK)
    buy_events = [e for e in events if e['type'] == 'bottom' and pf(e['peak_idx'])]

    if not buy_events:
        return None

    # 거래량 지표 계산
    surge_vnorm = calc_vol_surge_days_vnorm(df)
    surge_sdiv = calc_vol_surge_days_sdiv(df)
    surge_vma = calc_vol_surge_days_vma(df)
    v_norm = (df['Volume'] / df['Volume'].rolling(20).mean()).fillna(1).values

    close = df['Close'].values
    n = len(close)

    # Baseline rolling high
    rolling_high_base = pd.Series(close).rolling(BASELINE_DD_LB, min_periods=1).max().values

    rows = []
    for ev in buy_events:
        pidx = ev['peak_idx']
        if pidx >= n:
            continue
        price = close[pidx]
        date = df.index[pidx]

        # Baseline DD
        rh_base = rolling_high_base[pidx]
        dd_base = (rh_base - price) / rh_base if rh_base > 0 else 0
        base_pass = dd_base >= BASELINE_DD_TH

        # Dynamic lookbacks
        lb_vnorm = get_dynamic_lookback(surge_vnorm[pidx])
        lb_sdiv = get_dynamic_lookback(surge_sdiv[pidx])
        lb_vma = get_dynamic_lookback(surge_vma[pidx])

        # Dynamic DD 계산
        def calc_dd_for_lb(lb):
            start = max(0, pidx - lb + 1)
            rh = max(close[start:pidx+1])
            return (rh - price) / rh if rh > 0 else 0

        dd_vnorm = calc_dd_for_lb(lb_vnorm)
        dd_sdiv = calc_dd_for_lb(lb_sdiv)
        dd_vma = calc_dd_for_lb(lb_vma)

        # Forward returns
        fwd = {}
        hit = {}
        for h in HORIZONS:
            end_idx = min(pidx + h, n - 1)
            if end_idx > pidx:
                ret = (close[end_idx] - price) / price * 100
                fwd[h] = ret
                hit[h] = 1 if ret > 0 else 0
            else:
                fwd[h] = np.nan
                hit[h] = np.nan

        # MFE/MAE (60d)
        mfe_60 = np.nan
        mae_60 = np.nan
        end_60 = min(pidx + 60, n - 1)
        if end_60 > pidx:
            window_high = df['High'].values[pidx+1:end_60+1]
            window_low = df['Low'].values[pidx+1:end_60+1]
            if len(window_high) > 0:
                mfe_60 = (max(window_high) - price) / price * 100
                mae_60 = (price - min(window_low)) / price * 100

        row = {
            'ticker': ticker, 'sector': sector,
            'date': date, 'year': date.year,
            'price': price,
            'score': ev['peak_val'],
            's_force': sub['s_force'].iloc[pidx] if pidx < len(sub) else 0,
            's_div': sub['s_div'].iloc[pidx] if pidx < len(sub) else 0,
            'duration': ev['duration'],
            'v_norm': v_norm[pidx],

            # Surge days
            'surge_vnorm': surge_vnorm[pidx],
            'surge_sdiv': surge_sdiv[pidx],
            'surge_vma': surge_vma[pidx],

            # DD values
            'dd_base': dd_base,
            'dd_vnorm': dd_vnorm,
            'dd_sdiv': dd_sdiv,
            'dd_vma': dd_vma,

            # Lookbacks
            'lb_base': BASELINE_DD_LB,
            'lb_vnorm': lb_vnorm,
            'lb_sdiv': lb_sdiv,
            'lb_vma': lb_vma,

            # Pass/Block
            'pass_base': base_pass,
            'pass_vnorm': dd_vnorm >= BASELINE_DD_TH,
            'pass_sdiv': dd_sdiv >= BASELINE_DD_TH,
            'pass_vma': dd_vma >= BASELINE_DD_TH,

            # Forward returns
            'mfe_60': mfe_60,
            'mae_60': mae_60,
        }

        for h in HORIZONS:
            row[f'fwd_{h}d'] = fwd[h]
            row[f'hit_{h}d'] = hit[h]

        rows.append(row)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════
# 분석 함수들
# ═══════════════════════════════════════════════════════════
def analyze_per_ticker(data):
    """종목별 baseline vs dynamic-C 비교"""
    print("\n" + "=" * 100)
    print("1. 종목별 성과 비교 — Baseline vs Dynamic-C (VMA)")
    print("=" * 100)

    tickers = data['ticker'].unique()

    print(f"\n{'Ticker':<7} {'Sector':<10} | {'Base':>4} {'DynC':>4} | "
          f"{'Base 90d':>9} {'DynC 90d':>9} {'Delta':>7} | "
          f"{'Base Hit':>8} {'DynC Hit':>8} {'Delta':>7} | Winner")
    print("-" * 100)

    base_wins = 0
    dync_wins = 0

    for tk in sorted(tickers):
        tk_data = data[data['ticker'] == tk]
        sector = tk_data['sector'].iloc[0]

        base = tk_data[tk_data['pass_base']]
        dync = tk_data[tk_data['pass_vma']]

        b_n = len(base)
        d_n = len(dync)

        b_ret = base['fwd_90d'].dropna().mean() if len(base) > 0 else 0
        d_ret = dync['fwd_90d'].dropna().mean() if len(dync) > 0 else 0

        b_hit = base['hit_90d'].dropna().mean() * 100 if len(base) > 0 else 0
        d_hit = dync['hit_90d'].dropna().mean() * 100 if len(dync) > 0 else 0

        delta_ret = d_ret - b_ret
        delta_hit = d_hit - b_hit

        winner = ""
        if abs(delta_ret) > 2 or abs(delta_hit) > 5:
            if delta_ret > 0 and delta_hit >= -2:
                winner = "← DynC"
                dync_wins += 1
            elif delta_ret < 0 and delta_hit <= 2:
                winner = "← Base"
                base_wins += 1

        print(f"{tk:<7} {sector:<10} | {b_n:>4} {d_n:>4} | "
              f"{b_ret:>+8.1f}% {d_ret:>+8.1f}% {delta_ret:>+6.1f}% | "
              f"{b_hit:>7.1f}% {d_hit:>7.1f}% {delta_hit:>+6.1f}% | {winner}")

    print("-" * 100)
    print(f"  Baseline 우세: {base_wins}종목 / Dynamic-C 우세: {dync_wins}종목")


def analyze_surge_vs_return(data):
    """거래량 증가 기간 vs 수익률 상관관계"""
    print("\n\n" + "=" * 100)
    print("2. 거래량 증가 기간 vs 수익률 상관관계")
    print("=" * 100)

    # 모든 신호 (DD gate 적용 전)
    for surge_col, label in [('surge_vnorm', 'V_norm>1 연속'),
                              ('surge_sdiv', 'S_Div>0 연속'),
                              ('surge_vma', 'VMA 상승 연속')]:

        print(f"\n  ── {label} ──")

        # 구간별 분석
        bins = [(0, 0, '0일 (없음)'),
                (1, 4, '1~4일 (짧은)'),
                (5, 9, '5~9일'),
                (10, 19, '10~19일'),
                (20, 99, '20일+')]

        print(f"  {'구간':<16} {'신호수':>6} | {'30d':>8} {'60d':>8} {'90d':>8} | "
              f"{'Hit30':>6} {'Hit60':>6} {'Hit90':>6} | {'MFE60':>7} {'MAE60':>7}")
        print(f"  {'-'*96}")

        for lo, hi, label_str in bins:
            subset = data[(data[surge_col] >= lo) & (data[surge_col] <= hi)]
            if len(subset) == 0:
                continue

            n = len(subset)
            r30 = subset['fwd_30d'].dropna().mean()
            r60 = subset['fwd_60d'].dropna().mean()
            r90 = subset['fwd_90d'].dropna().mean()
            h30 = subset['hit_30d'].dropna().mean() * 100
            h60 = subset['hit_60d'].dropna().mean() * 100
            h90 = subset['hit_90d'].dropna().mean() * 100
            mfe = subset['mfe_60'].dropna().mean()
            mae = subset['mae_60'].dropna().mean()

            print(f"  {label_str:<16} {n:>6} | {r30:>+7.1f}% {r60:>+7.1f}% {r90:>+7.1f}% | "
                  f"{h30:>5.1f}% {h60:>5.1f}% {h90:>5.1f}% | {mfe:>+6.1f}% {mae:>6.1f}%")

        # 상관계수
        valid = data[[surge_col, 'fwd_90d']].dropna()
        if len(valid) > 5:
            corr = valid[surge_col].corr(valid['fwd_90d'])
            print(f"  상관계수 (surge vs 90d return): {corr:+.3f}")


def analyze_by_period(data):
    """시기별 분석"""
    print("\n\n" + "=" * 100)
    print("3. 시기별 분석 — Baseline vs Dynamic-C")
    print("=" * 100)

    years = sorted(data['year'].unique())
    print(f"\n  {'연도':<6} | {'Base 수':>6} {'DynC 수':>7} | "
          f"{'Base 90d':>9} {'DynC 90d':>9} {'Delta':>7} | "
          f"{'Base Hit':>8} {'DynC Hit':>8}")
    print(f"  {'-'*85}")

    for yr in years:
        yr_data = data[data['year'] == yr]
        base = yr_data[yr_data['pass_base']]
        dync = yr_data[yr_data['pass_vma']]

        b_n = len(base)
        d_n = len(dync)

        b_ret = base['fwd_90d'].dropna().mean() if b_n > 0 else np.nan
        d_ret = dync['fwd_90d'].dropna().mean() if d_n > 0 else np.nan

        b_hit = base['hit_90d'].dropna().mean() * 100 if b_n > 0 else np.nan
        d_hit = dync['hit_90d'].dropna().mean() * 100 if d_n > 0 else np.nan

        delta = (d_ret - b_ret) if not np.isnan(d_ret) and not np.isnan(b_ret) else np.nan

        b_ret_s = f"{b_ret:>+8.1f}%" if not np.isnan(b_ret) else f"{'N/A':>9}"
        d_ret_s = f"{d_ret:>+8.1f}%" if not np.isnan(d_ret) else f"{'N/A':>9}"
        delta_s = f"{delta:>+6.1f}%" if not np.isnan(delta) else f"{'N/A':>7}"
        b_hit_s = f"{b_hit:>7.1f}%" if not np.isnan(b_hit) else f"{'N/A':>8}"
        d_hit_s = f"{d_hit:>7.1f}%" if not np.isnan(d_hit) else f"{'N/A':>8}"

        print(f"  {yr:<6} | {b_n:>6} {d_n:>7} | {b_ret_s} {d_ret_s} {delta_s} | {b_hit_s} {d_hit_s}")


def analyze_blocked_signals(data):
    """차단 신호 심층 분석"""
    print("\n\n" + "=" * 100)
    print("4. 차단 신호 심층 분석 — Dynamic-C vs Baseline")
    print("=" * 100)

    # Case A: Baseline 통과 + Dynamic-C 차단
    case_a = data[(data['pass_base']) & (~data['pass_vma'])]
    # Case B: Baseline 차단 + Dynamic-C 통과
    case_b = data[(~data['pass_base']) & (data['pass_vma'])]
    # Case C: 둘 다 통과
    case_c = data[(data['pass_base']) & (data['pass_vma'])]
    # Case D: 둘 다 차단
    case_d = data[(~data['pass_base']) & (~data['pass_vma'])]

    print(f"\n  양쪽 다 통과: {len(case_c)}")
    print(f"  양쪽 다 차단: {len(case_d)}")
    print(f"  Baseline만 통과 (DynC가 추가 차단): {len(case_a)}")
    print(f"  DynC만 통과 (Baseline이 차단했던 것): {len(case_b)}")

    if len(case_a) > 0:
        print(f"\n  ── Baseline만 통과한 신호 (DynC가 잘못 차단?) ──")
        print(f"  {'Ticker':<7} {'Date':>12} {'Score':>6} {'DD%':>6} {'Surge':>6} | "
              f"{'30d':>7} {'60d':>7} {'90d':>7} | 판정")
        print(f"  {'-'*80}")
        for _, r in case_a.sort_values('date').iterrows():
            r30 = f"{r['fwd_30d']:>+6.1f}%" if not np.isnan(r['fwd_30d']) else "  N/A "
            r60 = f"{r['fwd_60d']:>+6.1f}%" if not np.isnan(r['fwd_60d']) else "  N/A "
            r90 = f"{r['fwd_90d']:>+6.1f}%" if not np.isnan(r['fwd_90d']) else "  N/A "
            verdict = "좋은차단" if (not np.isnan(r['fwd_90d']) and r['fwd_90d'] < 0) else "잘못된차단"
            print(f"  {r['ticker']:<7} {str(r['date'])[:10]:>12} {r['score']:>5.3f} "
                  f"{r['dd_vma']*100:>5.1f}% {r['surge_vma']:>5.0f}d | "
                  f"{r30} {r60} {r90} | {verdict}")

        avg_90 = case_a['fwd_90d'].dropna().mean()
        hit_90 = case_a['hit_90d'].dropna().mean() * 100
        print(f"\n  차단된 신호 평균: 90d {avg_90:+.1f}%, 적중률 {hit_90:.1f}%")

    if len(case_b) > 0:
        print(f"\n  ── DynC만 통과한 신호 (추가 기회?) ──")
        for _, r in case_b.sort_values('date').iterrows():
            r90 = f"{r['fwd_90d']:>+6.1f}%" if not np.isnan(r['fwd_90d']) else "  N/A "
            print(f"  {r['ticker']:<7} {str(r['date'])[:10]:>12} "
                  f"score={r['score']:.3f} dd_base={r['dd_base']*100:.1f}% "
                  f"dd_vma={r['dd_vma']*100:.1f}% surge_vma={r['surge_vma']:.0f}d "
                  f"90d={r90}")


def analyze_score_vs_dd(data):
    """스코어 강도 x DD 기간 교차 분석"""
    print("\n\n" + "=" * 100)
    print("5. 스코어 강도 x 거래량 증가 기간 교차 분석")
    print("=" * 100)

    # 스코어 구간
    score_bins = [(0, 0.05, '약 (0~0.05)'),
                  (0.05, 0.15, '중 (0.05~0.15)'),
                  (0.15, 1.0, '강 (0.15+)')]

    # VMA surge 구간
    surge_bins = [(0, 2, '짧은 (0~2d)'),
                  (3, 9, '중간 (3~9d)'),
                  (10, 99, '긴 (10d+)')]

    print(f"\n  {'Score \\ Surge':<16}", end="")
    for _, _, sl in surge_bins:
        print(f" | {sl:>16}", end="")
    print()
    print(f"  {'-'*70}")

    for slo, shi, slab in score_bins:
        print(f"  {slab:<16}", end="")
        for vlo, vhi, _ in surge_bins:
            subset = data[(data['score'] >= slo) & (data['score'] < shi) &
                         (data['surge_vma'] >= vlo) & (data['surge_vma'] <= vhi)]
            if len(subset) >= 3:
                avg = subset['fwd_90d'].dropna().mean()
                hit = subset['hit_90d'].dropna().mean() * 100
                n = len(subset)
                print(f" | {avg:>+5.1f}% h{hit:.0f}% n{n}", end="")
            else:
                print(f" | {'(n<3)':>16}", end="")
        print()


def analyze_dd_amount_vs_return(data):
    """DD 낙폭 크기 vs 수익률"""
    print("\n\n" + "=" * 100)
    print("6. DD 낙폭 크기별 성과 (모든 신호, DD gate 적용 전)")
    print("=" * 100)

    dd_bins = [(0, 0.03, '0~3% (현재 차단)'),
               (0.03, 0.05, '3~5%'),
               (0.05, 0.10, '5~10%'),
               (0.10, 0.20, '10~20%'),
               (0.20, 1.0, '20%+')]

    print(f"\n  {'DD 구간':<20} {'신호수':>6} | {'30d':>8} {'60d':>8} {'90d':>8} | "
          f"{'Hit30':>6} {'Hit60':>6} {'Hit90':>6} | {'MFE60':>7}")
    print(f"  {'-'*95}")

    for lo, hi, label in dd_bins:
        subset = data[(data['dd_base'] >= lo) & (data['dd_base'] < hi)]
        if len(subset) == 0:
            continue

        n = len(subset)
        print(f"  {label:<20} {n:>6} | "
              f"{subset['fwd_30d'].dropna().mean():>+7.1f}% "
              f"{subset['fwd_60d'].dropna().mean():>+7.1f}% "
              f"{subset['fwd_90d'].dropna().mean():>+7.1f}% | "
              f"{subset['hit_30d'].dropna().mean()*100:>5.1f}% "
              f"{subset['hit_60d'].dropna().mean()*100:>5.1f}% "
              f"{subset['hit_90d'].dropna().mean()*100:>5.1f}% | "
              f"{subset['mfe_60'].dropna().mean():>+6.1f}%")

    # 상관계수
    valid = data[['dd_base', 'fwd_90d']].dropna()
    if len(valid) > 5:
        corr = valid['dd_base'].corr(valid['fwd_90d'])
        print(f"\n  상관계수 (DD낙폭 vs 90d return): {corr:+.3f}")
        print(f"  해석: {'낙폭 클수록 수익 높음' if corr > 0.1 else '낙폭과 수익 무관' if abs(corr) < 0.1 else '낙폭 클수록 수익 낮음'}")


def analyze_sector(data):
    """섹터별 분석"""
    print("\n\n" + "=" * 100)
    print("7. 섹터별 분석")
    print("=" * 100)

    sectors = data['sector'].unique()
    print(f"\n  {'Sector':<12} {'신호':>4} | {'90d ret':>8} {'Hit90':>6} | "
          f"{'avg surge_vma':>13} {'avg DD%':>7} | {'DynC 유리?':>10}")
    print(f"  {'-'*80}")

    for sec in sorted(sectors):
        sec_data = data[data['sector'] == sec]
        n = len(sec_data)

        # Baseline
        base = sec_data[sec_data['pass_base']]
        dync = sec_data[sec_data['pass_vma']]

        b_ret = base['fwd_90d'].dropna().mean() if len(base) > 0 else np.nan
        d_ret = dync['fwd_90d'].dropna().mean() if len(dync) > 0 else np.nan

        b_hit = base['hit_90d'].dropna().mean() * 100 if len(base) > 0 else np.nan

        avg_surge = sec_data['surge_vma'].mean()
        avg_dd = sec_data['dd_base'].mean() * 100

        better = ""
        if not np.isnan(b_ret) and not np.isnan(d_ret):
            delta = d_ret - b_ret
            if delta > 3:
                better = f"DynC +{delta:.0f}%"
            elif delta < -3:
                better = f"Base +{-delta:.0f}%"
            else:
                better = "비슷"

        b_ret_s = f"{b_ret:>+7.1f}%" if not np.isnan(b_ret) else "N/A"
        b_hit_s = f"{b_hit:>5.1f}%" if not np.isnan(b_hit) else "N/A"

        print(f"  {sec:<12} {n:>4} | {b_ret_s} {b_hit_s} | "
              f"{avg_surge:>12.1f}d {avg_dd:>6.1f}% | {better:>10}")


def analyze_vnorm_at_signal(data):
    """신호 시점의 거래량 수준 분석"""
    print("\n\n" + "=" * 100)
    print("8. 신호 시점 거래량 수준 (V_norm) vs 수익률")
    print("=" * 100)

    vn_bins = [(0, 0.7, '매우 낮은 (V<0.7)'),
               (0.7, 1.0, '평균 이하 (0.7~1.0)'),
               (1.0, 1.5, '평균 이상 (1.0~1.5)'),
               (1.5, 2.5, '높음 (1.5~2.5)'),
               (2.5, 99, '매우 높음 (2.5+)')]

    print(f"\n  {'V_norm 구간':<22} {'신호수':>6} | {'30d':>8} {'60d':>8} {'90d':>8} | "
          f"{'Hit90':>6} | {'Score':>6}")
    print(f"  {'-'*80}")

    for lo, hi, label in vn_bins:
        subset = data[(data['v_norm'] >= lo) & (data['v_norm'] < hi)]
        if len(subset) < 2:
            continue

        n = len(subset)
        print(f"  {label:<22} {n:>6} | "
              f"{subset['fwd_30d'].dropna().mean():>+7.1f}% "
              f"{subset['fwd_60d'].dropna().mean():>+7.1f}% "
              f"{subset['fwd_90d'].dropna().mean():>+7.1f}% | "
              f"{subset['hit_90d'].dropna().mean()*100:>5.1f}% | "
              f"{subset['score'].mean():>.3f}")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Volume-Dynamic DD Gate — 심층 분석")
    print("=" * 60)

    all_data = []
    for ticker, sector in TICKERS.items():
        print(f"  {ticker}...", end=" ", flush=True)
        try:
            df = collect_all_signals(ticker, sector)
            if df is not None and len(df) > 0:
                all_data.append(df)
                print(f"OK ({len(df)} signals)")
            else:
                print("SKIP")
        except Exception as e:
            print(f"ERROR: {e}")

    if not all_data:
        print("No data collected.")
        sys.exit(1)

    data = pd.concat(all_data, ignore_index=True)
    print(f"\n총 {len(data)} 신호 수집 완료")

    # 분석 실행
    analyze_per_ticker(data)
    analyze_surge_vs_return(data)
    analyze_by_period(data)
    analyze_blocked_signals(data)
    analyze_score_vs_dd(data)
    analyze_dd_amount_vs_return(data)
    analyze_sector(data)
    analyze_vnorm_at_signal(data)

    print("\n\n" + "=" * 100)
    print("분석 완료")
    print("=" * 100)
