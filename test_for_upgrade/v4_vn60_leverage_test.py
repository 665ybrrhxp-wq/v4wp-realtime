"""
VN60 Leverage Buy Backtest: 레버리지 매수 효과 상세 분석
========================================================
메커니즘:
  - 월초 $500 입금
  - VN60 시그널 발생 시 → 가용자금의 50%로 2x 매수
  - 월말 잔여자금 → 1x 매수

비교:
  A) VN60+2x:   시그널에 2x 매수 (현재 시스템)
  B) VN60+1x:   같은 시그널 타이밍에 1x만 매수 (레버리지 효과 분리)
  C) Pure DCA:  시그널 무시, 매월 1x만

분석:
  - 레버리지가 더한 수익 = A - B
  - 타이밍이 더한 수익 = B - C
  - 총 추가 수익 = A - C
  - 개별 2x 매수의 90d/180d 수익률
  - 포트폴리오 내 2x vs 1x 비중 추이
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
    calc_pv_divergence,
    detect_signal_events, build_price_filter, smooth_earnings_volume,
)

TICKERS = {
    'TSLA': 'Tech', 'NVDA': 'Tech', 'AVGO': 'Tech',
    'AMZN': 'Tech', 'GOOGL': 'Tech', 'PLTR': 'Tech',
    'COIN': 'Fintech', 'HOOD': 'Fintech',
    'HIMS': 'Growth', 'RKLB': 'Growth', 'JOBY': 'Growth',
    'IONQ': 'Quantum',
    'QQQ': 'Benchmark', 'VOO': 'Benchmark',
}

MONTHLY_DEPOSIT = 500.0
SIGNAL_BUY_PCT = 0.50
LEVERAGE = 2.0
EXPENSE_RATIO_DAILY = 0.0095 / 252

V4_W = 20; SIGNAL_TH = 0.15; COOLDOWN = 5
ER_Q = 66; ATR_Q = 55; LOOKBACK = 252
DIVGATE = 3; CONFIRM = 3
BUY_DD_LOOKBACK = 20; BUY_DD_THRESHOLD = 0.05


def download_max(ticker):
    t = yf.Ticker(ticker)
    df = t.history(period='max', auto_adjust=True)
    if df is None or len(df) == 0:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.dropna(subset=['Close', 'Volume'])
    df = df[df['Close'] > 0]
    return df


def calc_force_macd_vel(df, fast=12, slow=26, signal=9):
    p_vel = df['Close'].pct_change().fillna(0)
    v_norm = df['Volume'] / df['Volume'].rolling(20).mean().replace(0, np.nan).fillna(df['Volume'])
    force = v_norm * p_vel
    fm = force.ewm(span=fast, adjust=False).mean() - force.ewm(span=slow, adjust=False).mean()
    fs = fm.ewm(span=signal, adjust=False).mean()
    return (fm - fs).rename('PV_Force_Hist')


def build_synthetic_2x(close):
    daily_ret = np.diff(close) / close[:-1]
    lev_price = np.zeros(len(close))
    lev_price[0] = close[0]
    for i in range(1, len(close)):
        lev_ret = LEVERAGE * daily_ret[i - 1] - EXPENSE_RATIO_DAILY
        lev_price[i] = lev_price[i - 1] * (1 + lev_ret)
        if lev_price[i] < 0.001:
            lev_price[i] = 0.001
    return lev_price


def precompute_vn60(df, w=20, divgate_days=3):
    n = len(df)
    pv_div = calc_pv_divergence(df, w)
    pv_fh_vel = calc_force_macd_vel(df)
    raw_div = pv_div.values if hasattr(pv_div, 'values') else np.array(pv_div)
    consec = np.zeros(n, dtype=int)
    for i in range(1, n):
        if raw_div[i] != 0 and np.sign(raw_div[i]) == np.sign(raw_div[i - 1]):
            consec[i] = consec[i - 1] + 1
        elif raw_div[i] != 0:
            consec[i] = 1
    return raw_div, consec, pv_fh_vel


def calc_score_vn60(df, raw_div, consec, pv_fh_vel, w=20, divgate_days=3):
    n = len(df)
    scores = np.zeros(n)
    for i in range(max(60, w), n):
        s_div = raw_div[i] if consec[i] >= divgate_days else 0.0
        fhr_std = pv_fh_vel.iloc[max(0, i - w):i].std() + 1e-10
        s_force = np.clip(pv_fh_vel.iloc[i] / (2 * fhr_std), -1, 1)
        scores[i] = 0.60 * s_force + 0.40 * s_div
    return pd.Series(scores, index=df.index)


def get_buy_signals(df, ticker):
    """VN60 매수 시그널 인덱스 리스트 반환 (BUY_DD_GATE 포함)"""
    try:
        df_s = smooth_earnings_volume(df, ticker=ticker)
    except Exception:
        df_s = df.copy()

    raw_div, consec, pv_fh_vel = precompute_vn60(df_s)
    score_series = calc_score_vn60(df_s, raw_div, consec, pv_fh_vel)
    events = detect_signal_events(score_series, th=SIGNAL_TH, cooldown=COOLDOWN)
    pf = build_price_filter(df_s, er_q=ER_Q, atr_q=ATR_Q, lookback=LOOKBACK)

    close_vals = df_s['Close'].values
    rolling_high = pd.Series(close_vals).rolling(BUY_DD_LOOKBACK, min_periods=1).max().values
    n = len(df_s)

    buy_indices = []
    buy_details = []

    for ev in events:
        if ev['type'] != 'bottom':
            continue
        if not pf(ev['peak_idx']):
            continue
        dur = ev['end_idx'] - ev['start_idx'] + 1
        ci = ev['start_idx'] + CONFIRM - 1
        if ci > ev['end_idx'] or dur < CONFIRM or ci >= n:
            continue

        pidx = ev['peak_idx']
        rh = rolling_high[pidx]
        dd = (rh - close_vals[pidx]) / rh if rh > 0 else 0
        if dd < BUY_DD_THRESHOLD:
            continue

        buy_indices.append(ci)
        buy_details.append({
            'idx': ci,
            'score': ev['peak_val'],
            'duration': dur,
            'dd_from_high': dd,
            'price': close_vals[ci],
        })

    return buy_indices, buy_details


def simulate_three_modes(close, close_2x, buy_indices, dates):
    """
    3가지 모드 동시 시뮬레이션:
      mode_a: VN60+2x (시그널에 2x 매수)
      mode_b: VN60+1x (시그널에 1x 매수, 같은 타이밍)
      mode_c: Pure DCA (시그널 무시, 월말 1x 매수)
    """
    n = len(close)
    month_periods = pd.Series(dates).dt.to_period('M')
    month_map = {}
    for i, mp in enumerate(month_periods):
        key = str(mp)
        if key not in month_map:
            month_map[key] = {'first': i, 'last': i}
        else:
            month_map[key]['last'] = i
    sorted_months = sorted(month_map.keys())

    buy_set = set(buy_indices)

    # Mode A: VN60+2x
    cash_a = 0.0; sh_1x_a = 0.0; sh_2x_a = 0.0
    # Mode B: VN60+1x (same timing, but buy 1x instead of 2x)
    cash_b = 0.0; sh_1x_b = 0.0
    # Mode C: Pure DCA
    cash_c = 0.0; sh_1x_c = 0.0

    total_dep = 0.0
    sig_count = 0
    buy_2x_log = []  # log each 2x purchase

    yr_data = {}; prev_yr = None

    def pf_a(idx):
        return sh_1x_a * close[idx] + sh_2x_a * close_2x[idx] + cash_a
    def pf_b(idx):
        return sh_1x_b * close[idx] + cash_b
    def pf_c(idx):
        return sh_1x_c * close[idx] + cash_c

    # Track portfolio composition over time
    pf_snapshots = []

    for mk in sorted_months:
        mm = month_map[mk]
        fi, li = mm['first'], mm['last']
        yr = int(mk[:4])

        if yr != prev_yr:
            if prev_yr is not None:
                ref = fi - 1 if fi > 0 else fi
                yr_data[prev_yr]['end_a'] = pf_a(ref)
                yr_data[prev_yr]['end_b'] = pf_b(ref)
                yr_data[prev_yr]['end_c'] = pf_c(ref)
            yr_data[yr] = {
                'start_a': pf_a(fi), 'start_b': pf_b(fi), 'start_c': pf_c(fi),
                'deposits': 0.0, 'sigs': 0,
            }
            prev_yr = yr

        # Monthly deposit
        cash_a += MONTHLY_DEPOSIT
        cash_b += MONTHLY_DEPOSIT
        cash_c += MONTHLY_DEPOSIT
        total_dep += MONTHLY_DEPOSIT
        yr_data[yr]['deposits'] += MONTHLY_DEPOSIT

        for day_idx in range(fi, li + 1):
            if day_idx in buy_set:
                # Mode A: 2x leveraged buy
                if cash_a > 1.0:
                    amt_a = cash_a * SIGNAL_BUY_PCT
                    sh_2x_a += amt_a / close_2x[day_idx]
                    cash_a -= amt_a
                    sig_count += 1
                    yr_data[yr]['sigs'] += 1
                    buy_2x_log.append({
                        'idx': day_idx,
                        'date': dates[day_idx],
                        'price_1x': close[day_idx],
                        'price_2x': close_2x[day_idx],
                        'amount': amt_a,
                        'shares_2x': amt_a / close_2x[day_idx],
                    })

                # Mode B: 1x buy at same timing
                if cash_b > 1.0:
                    amt_b = cash_b * SIGNAL_BUY_PCT
                    sh_1x_b += amt_b / close[day_idx]
                    cash_b -= amt_b

        # Month-end: remaining cash → 1x
        if cash_a > 1.0:
            sh_1x_a += cash_a / close[li]; cash_a = 0.0
        if cash_b > 1.0:
            sh_1x_b += cash_b / close[li]; cash_b = 0.0
        if cash_c > 1.0:
            sh_1x_c += cash_c / close[li]; cash_c = 0.0

        # Snapshot at month end
        val_1x_a = sh_1x_a * close[li]
        val_2x_a = sh_2x_a * close_2x[li]
        total_a = val_1x_a + val_2x_a + cash_a
        pf_snapshots.append({
            'date': dates[li],
            'total_a': total_a,
            'val_2x': val_2x_a,
            'pct_2x': val_2x_a / total_a * 100 if total_a > 1 else 0,
            'total_b': pf_b(li),
            'total_c': pf_c(li),
        })

    if prev_yr is not None:
        yr_data[prev_yr]['end_a'] = pf_a(n - 1)
        yr_data[prev_yr]['end_b'] = pf_b(n - 1)
        yr_data[prev_yr]['end_c'] = pf_c(n - 1)

    final_a = pf_a(n - 1); final_b = pf_b(n - 1); final_c = pf_c(n - 1)

    # 2x portion value at end
    val_2x_end = sh_2x_a * close_2x[n - 1]
    val_1x_end = sh_1x_a * close[n - 1]

    # Year-by-year results
    yr_results = []
    for yr in sorted(yr_data.keys()):
        yd = yr_data[yr]
        if 'end_a' not in yd:
            continue
        for mode in ['a', 'b', 'c']:
            d = yd[f'start_{mode}'] + yd['deposits'] * 0.5
            if d > 10:
                val = (yd[f'end_{mode}'] - yd[f'start_{mode}'] - yd['deposits']) / d * 100
                yd[f'ret_{mode}'] = val if np.isfinite(val) else 0.0
            else:
                yd[f'ret_{mode}'] = 0
        yr_results.append({
            'yr': yr,
            'ret_a': yd['ret_a'], 'ret_b': yd['ret_b'], 'ret_c': yd['ret_c'],
            'edge_a': yd['ret_a'] - yd['ret_c'],  # VN60+2x vs DCA
            'edge_b': yd['ret_b'] - yd['ret_c'],  # VN60+1x vs DCA
            'lev_add': yd['ret_a'] - yd['ret_b'],  # 레버리지가 추가한 수익
            'sigs': yd['sigs'],
        })

    return {
        'yr_results': yr_results,
        'final_a': final_a, 'final_b': final_b, 'final_c': final_c,
        'total_dep': total_dep,
        'sig_count': sig_count,
        'val_2x_end': val_2x_end, 'val_1x_end': val_1x_end,
        'buy_2x_log': buy_2x_log,
        'pf_snapshots': pf_snapshots,
    }


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
print("=" * 120)
print("  VN60 LEVERAGE BUY BACKTEST")
print("  월초 $500 입금 → 시그널 시 2x 매수(50%) → 월말 잔여자금 1x 매수")
print("=" * 120)

all_results = {}

for tk, sector in TICKERS.items():
    print(f"  {tk}...", end=" ", flush=True)
    df = download_max(tk)
    if df is None or len(df) < 300:
        print("SKIP")
        continue

    close = df['Close'].values
    dates = df.index
    close_2x = build_synthetic_2x(close)
    buy_indices, buy_details = get_buy_signals(df, tk)

    res = simulate_three_modes(close, close_2x, buy_indices, dates)
    res['buy_details'] = buy_details
    res['close'] = close
    res['close_2x'] = close_2x
    res['dates'] = dates
    res['n'] = len(close)
    all_results[tk] = res

    avg_ea = np.nanmean([r['edge_a'] for r in res['yr_results']]) if res['yr_results'] else 0
    avg_eb = np.nanmean([r['edge_b'] for r in res['yr_results']]) if res['yr_results'] else 0
    avg_lev = np.nanmean([r['lev_add'] for r in res['yr_results']]) if res['yr_results'] else 0
    print(f"sigs={res['sig_count']:3d}  2x+1x edge={avg_ea:+.1f}%  1x-only edge={avg_eb:+.1f}%  레버리지 추가={avg_lev:+.1f}%")

tks = list(all_results.keys())


# ═══════════════════════════════════════════════════════════
# SECTION 1: Overall Comparison
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*120}")
print(f"  SECTION 1: 3가지 모드 비교 — 연평균 수익률 & Edge")
print(f"  A=VN60+2x | B=VN60+1x(같은 타이밍) | C=Pure DCA(1x)")
print(f"{'='*120}")

print(f"  {'Ticker':<7s} {'Sigs':>5s} {'A(2x)ret':>10s} {'B(1x)ret':>10s} {'C(DCA)ret':>10s} │"
      f" {'Total Edge':>11s} {'Timing':>9s} {'Leverage':>9s} │ {'Final A':>12s} {'Final C':>12s}")
print(f"  {'-'*115}")

sum_ea = []; sum_eb = []; sum_lev = []

for tk in tks:
    res = all_results[tk]
    yr = res['yr_results']
    avg_a = np.nanmean([r['ret_a'] for r in yr])
    avg_b = np.nanmean([r['ret_b'] for r in yr])
    avg_c = np.nanmean([r['ret_c'] for r in yr])
    avg_ea = np.nanmean([r['edge_a'] for r in yr])  # total edge (2x vs DCA)
    avg_eb = np.nanmean([r['edge_b'] for r in yr])  # timing edge (1x vs DCA)
    avg_lev = np.nanmean([r['lev_add'] for r in yr])  # leverage edge (2x vs 1x)

    sum_ea.append(avg_ea); sum_eb.append(avg_eb); sum_lev.append(avg_lev)

    fa = res['final_a']; fc = res['final_c']

    print(f"  {tk:<7s} {res['sig_count']:>5d} {avg_a:>+8.1f}% {avg_b:>+8.1f}% {avg_c:>+8.1f}% │"
          f" {avg_ea:>+9.2f}%p {avg_eb:>+7.2f}%p {avg_lev:>+7.2f}%p │"
          f" ${fa:>11,.0f} ${fc:>11,.0f}")

print(f"  {'-'*115}")
print(f"  {'AVG':<7s} {'':>5s} {'':>10s} {'':>10s} {'':>10s} │"
      f" {np.nanmean(sum_ea):>+9.2f}%p {np.nanmean(sum_eb):>+7.2f}%p {np.nanmean(sum_lev):>+7.2f}%p │")

print(f"\n  * Total Edge = 전체 추가수익 (VN60+2x vs DCA)")
print(f"  * Timing     = 시그널 타이밍만의 효과 (VN60+1x vs DCA)")
print(f"  * Leverage   = 2x 레버리지만의 효과 (VN60+2x vs VN60+1x)")


# ═══════════════════════════════════════════════════════════
# SECTION 2: Year-by-Year Detail (상위 6 티커)
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*120}")
print(f"  SECTION 2: Year-by-Year 상세 (주요 티커)")
print(f"{'='*120}")

# Pick tickers with most data
top_tks = sorted(tks, key=lambda t: len(all_results[t]['yr_results']), reverse=True)[:6]

for tk in top_tks:
    res = all_results[tk]
    yr = res['yr_results']
    print(f"\n  {tk}:")
    print(f"    {'Year':>6s} {'A(2x)':>8s} {'B(1x)':>8s} {'C(DCA)':>8s} │ {'Total':>8s} {'Timing':>8s} {'Lev':>8s} │ {'Sigs':>4s}")
    print(f"    {'-'*75}")

    for r in yr:
        print(f"    {r['yr']:>6d} {r['ret_a']:>+7.1f}% {r['ret_b']:>+7.1f}% {r['ret_c']:>+7.1f}% │"
              f" {r['edge_a']:>+7.2f}% {r['edge_b']:>+7.2f}% {r['lev_add']:>+7.2f}% │ {r['sigs']:>4d}")

    avg_ea = np.nanmean([r['edge_a'] for r in yr])
    avg_eb = np.nanmean([r['edge_b'] for r in yr])
    avg_lev = np.nanmean([r['lev_add'] for r in yr])
    print(f"    {'-'*75}")
    print(f"    {'AVG':>6s} {'':>8s} {'':>8s} {'':>8s} │ {avg_ea:>+7.2f}% {avg_eb:>+7.2f}% {avg_lev:>+7.2f}% │")

    # Worst year for leverage
    worst_lev = min(yr, key=lambda r: r['lev_add'])
    best_lev = max(yr, key=lambda r: r['lev_add'])
    print(f"    Best leverage year: {best_lev['yr']} ({best_lev['lev_add']:+.2f}%)")
    print(f"    Worst leverage year: {worst_lev['yr']} ({worst_lev['lev_add']:+.2f}%)")


# ═══════════════════════════════════════════════════════════
# SECTION 3: Portfolio Composition — 2x 비중
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*120}")
print(f"  SECTION 3: 최종 포트폴리오 — 2x 비중")
print(f"{'='*120}")

print(f"  {'Ticker':<7s} {'Final A':>12s} {'2x Value':>12s} {'1x Value':>12s} {'2x Pct':>8s} {'Dep Total':>10s} {'ROI(A)':>8s} {'ROI(C)':>8s}")
print(f"  {'-'*80}")

for tk in tks:
    res = all_results[tk]
    fa = res['final_a']
    v2x = res['val_2x_end']; v1x = res['val_1x_end']
    pct2x = v2x / fa * 100 if fa > 1 else 0
    roi_a = (fa / res['total_dep'] - 1) * 100
    roi_c = (res['final_c'] / res['total_dep'] - 1) * 100
    print(f"  {tk:<7s} ${fa:>11,.0f} ${v2x:>11,.0f} ${v1x:>11,.0f} {pct2x:>6.1f}% ${res['total_dep']:>9,.0f} {roi_a:>+7.1f}% {roi_c:>+7.1f}%")


# ═══════════════════════════════════════════════════════════
# SECTION 4: 개별 2x 매수의 90d/180d Forward Return
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*120}")
print(f"  SECTION 4: 개별 2x 매수의 Forward Return (90d / 180d)")
print(f"{'='*120}")

print(f"  {'Ticker':<7s} {'2xBuys':>6s} │ {'1x 90d':>9s} {'2x 90d':>9s} {'Lev Win':>8s} │ {'1x 180d':>9s} {'2x 180d':>9s} {'Lev Win':>8s}")
print(f"  {'-'*80}")

all_1x_90 = []; all_2x_90 = []; all_1x_180 = []; all_2x_180 = []

for tk in tks:
    res = all_results[tk]
    close = res['close']
    close_2x = res['close_2x']
    n = res['n']
    logs = res['buy_2x_log']

    ret_1x_90 = []; ret_2x_90 = []; ret_1x_180 = []; ret_2x_180 = []
    lev_win_90 = 0; lev_win_180 = 0
    cnt_90 = 0; cnt_180 = 0

    for log in logs:
        idx = log['idx']

        # 90d
        end90 = min(idx + 90, n - 1)
        if end90 > idx + 10:
            r1x = (close[end90] - close[idx]) / close[idx] * 100
            r2x = (close_2x[end90] - close_2x[idx]) / close_2x[idx] * 100
            ret_1x_90.append(r1x); ret_2x_90.append(r2x)
            if r2x > r1x:
                lev_win_90 += 1
            cnt_90 += 1

        # 180d
        end180 = min(idx + 180, n - 1)
        if end180 > idx + 30:
            r1x = (close[end180] - close[idx]) / close[idx] * 100
            r2x = (close_2x[end180] - close_2x[idx]) / close_2x[idx] * 100
            ret_1x_180.append(r1x); ret_2x_180.append(r2x)
            if r2x > r1x:
                lev_win_180 += 1
            cnt_180 += 1

    a1x90 = np.nanmean(ret_1x_90) if ret_1x_90 else 0
    a2x90 = np.nanmean(ret_2x_90) if ret_2x_90 else 0
    a1x180 = np.nanmean(ret_1x_180) if ret_1x_180 else 0
    a2x180 = np.nanmean(ret_2x_180) if ret_2x_180 else 0
    lw90 = lev_win_90 / cnt_90 * 100 if cnt_90 > 0 else 0
    lw180 = lev_win_180 / cnt_180 * 100 if cnt_180 > 0 else 0

    all_1x_90.append(a1x90); all_2x_90.append(a2x90)
    all_1x_180.append(a1x180); all_2x_180.append(a2x180)

    print(f"  {tk:<7s} {len(logs):>6d} │ {a1x90:>+7.1f}% {a2x90:>+7.1f}% {lw90:>6.0f}% │ {a1x180:>+7.1f}% {a2x180:>+7.1f}% {lw180:>6.0f}%")

print(f"  {'-'*80}")
print(f"  {'AVG':<7s} {'':>6s} │ {np.nanmean(all_1x_90):>+7.1f}% {np.nanmean(all_2x_90):>+7.1f}% {'':>8s} │ "
      f"{np.nanmean(all_1x_180):>+7.1f}% {np.nanmean(all_2x_180):>+7.1f}% {'':>8s}")
print(f"\n  * Lev Win = 2x 수익률 > 1x 수익률인 비율 (2x 레버리지가 이긴 %)")


# ═══════════════════════════════════════════════════════════
# SECTION 5: Bear Year 상세 — 레버리지 손실
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*120}")
print(f"  SECTION 5: BEAR YEAR 상세 — 하락장에서 레버리지 손실")
print(f"{'='*120}")

print(f"  {'Ticker':<7s} {'Year':>6s} {'A(2x)':>8s} {'B(1x)':>8s} {'C(DCA)':>8s} │ {'Lev Cost':>9s}  (레버리지가 추가한 손실)")
print(f"  {'-'*70}")

bear_lev_costs = []
for tk in tks:
    yr = all_results[tk]['yr_results']
    for r in yr:
        if r['ret_c'] < -5:  # bear year
            print(f"  {tk:<7s} {r['yr']:>6d} {r['ret_a']:>+7.1f}% {r['ret_b']:>+7.1f}% {r['ret_c']:>+7.1f}% │ {r['lev_add']:>+7.2f}%")
            bear_lev_costs.append(r['lev_add'])

if bear_lev_costs:
    print(f"  {'-'*70}")
    print(f"  Bear year 평균 레버리지 비용: {np.nanmean(bear_lev_costs):+.2f}%p")
    print(f"  Bear year 최악 레버리지 비용: {min(bear_lev_costs):+.2f}%p")


# ═══════════════════════════════════════════════════════════
# SECTION 6: Bull Year 상세 — 레버리지 이득
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*120}")
print(f"  SECTION 6: BULL YEAR 상세 — 상승장에서 레버리지 이득 (DCA > +15%)")
print(f"{'='*120}")

print(f"  {'Ticker':<7s} {'Year':>6s} {'A(2x)':>8s} {'B(1x)':>8s} {'C(DCA)':>8s} │ {'Lev Gain':>9s}  (레버리지가 추가한 수익)")
print(f"  {'-'*70}")

bull_lev_gains = []
for tk in tks:
    yr = all_results[tk]['yr_results']
    for r in yr:
        if r['ret_c'] > 15:  # strong bull year
            print(f"  {tk:<7s} {r['yr']:>6d} {r['ret_a']:>+7.1f}% {r['ret_b']:>+7.1f}% {r['ret_c']:>+7.1f}% │ {r['lev_add']:>+7.2f}%")
            bull_lev_gains.append(r['lev_add'])

if bull_lev_gains:
    print(f"  {'-'*70}")
    print(f"  Bull year 평균 레버리지 이득: {np.nanmean(bull_lev_gains):+.2f}%p")
    print(f"  Bull year 최고 레버리지 이득: {max(bull_lev_gains):+.2f}%p")


# ═══════════════════════════════════════════════════════════
# SECTION 7: Efficiency 비교
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*120}")
print(f"  SECTION 7: EFFICIENCY 비교 — Edge / |Worst Year Loss|")
print(f"{'='*120}")

print(f"  {'Ticker':<7s} {'A(2x) Edge':>12s} {'A Worst':>9s} {'A Eff':>8s} │ {'B(1x) Edge':>12s} {'B Worst':>9s} {'B Eff':>8s} │ {'Winner':>8s}")
print(f"  {'-'*85}")

eff_a_list = []; eff_b_list = []

for tk in tks:
    yr = all_results[tk]['yr_results']
    edge_a = np.nanmean([r['edge_a'] for r in yr])
    edge_b = np.nanmean([r['edge_b'] for r in yr])
    worst_a = min(r['edge_a'] for r in yr)
    worst_b = min(r['edge_b'] for r in yr)
    eff_a = edge_a / abs(worst_a) if abs(worst_a) > 0.1 else 0
    eff_b = edge_b / abs(worst_b) if abs(worst_b) > 0.1 else 0
    eff_a_list.append(eff_a); eff_b_list.append(eff_b)
    winner = "2x" if eff_a > eff_b else "1x" if eff_b > eff_a else "TIE"

    print(f"  {tk:<7s} {edge_a:>+10.2f}%p {worst_a:>+7.2f}% {eff_a:>+7.3f} │ {edge_b:>+10.2f}%p {worst_b:>+7.2f}% {eff_b:>+7.3f} │ {winner:>8s}")

print(f"  {'-'*85}")
print(f"  {'AVG':<7s} {np.nanmean([np.nanmean([r['edge_a'] for r in all_results[t]['yr_results']]) for t in tks]):>+10.2f}%p "
      f"{'':>9s} {np.nanmean(eff_a_list):>+7.3f} │"
      f" {np.nanmean([np.nanmean([r['edge_b'] for r in all_results[t]['yr_results']]) for t in tks]):>+10.2f}%p "
      f"{'':>9s} {np.nanmean(eff_b_list):>+7.3f} │")

wins_2x = sum(1 for a, b in zip(eff_a_list, eff_b_list) if a > b)
wins_1x = sum(1 for a, b in zip(eff_a_list, eff_b_list) if b > a)
print(f"\n  2x wins: {wins_2x}/{len(tks)} tickers  |  1x wins: {wins_1x}/{len(tks)} tickers")


# ═══════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*120}")
print(f"  FINAL SUMMARY")
print(f"{'='*120}")

avg_total_edge = np.nanmean(sum_ea)
avg_timing_edge = np.nanmean(sum_eb)
avg_lev_edge = np.nanmean(sum_lev)

# Bear/Bull breakdown
all_bear_lev = []
all_bull_lev = []
for tk in tks:
    for r in all_results[tk]['yr_results']:
        if r['ret_c'] < -5:
            all_bear_lev.append(r['lev_add'])
        elif r['ret_c'] > 5:
            all_bull_lev.append(r['lev_add'])

print(f"""
  +-------------------------------------------------------+
  |  VN60 레버리지 매수 효과 분해                         |
  +-------------------------------------------------------+
  |  Total Edge (VN60+2x vs DCA):  {avg_total_edge:>+7.2f}%p / year      |
  |    = Timing  (VN60+1x vs DCA): {avg_timing_edge:>+7.2f}%p / year      |
  |    + Leverage (2x vs 1x):      {avg_lev_edge:>+7.2f}%p / year      |
  +-------------------------------------------------------+
  |  Leverage in Bear years:        {np.nanmean(all_bear_lev):>+7.2f}%p             |
  |  Leverage in Bull years:        {np.nanmean(all_bull_lev):>+7.2f}%p             |
  +-------------------------------------------------------+
  |  Avg Efficiency (2x):           {np.nanmean(eff_a_list):>+7.3f}              |
  |  Avg Efficiency (1x):           {np.nanmean(eff_b_list):>+7.3f}              |
  |  2x wins {wins_2x}/{len(tks)} tickers on efficiency               |
  +-------------------------------------------------------+
""")

print("=" * 120)
print("  Done.")
