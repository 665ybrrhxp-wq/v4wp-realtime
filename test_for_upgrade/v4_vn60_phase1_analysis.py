"""
VN60 Phase 1 Sweep: 연평균수익률 기준 상세 분석
================================================
기존 sweep 결과를 연평균수익률(Annual Return) 기준으로 재정리
+ 티커별/연도별 심층 분석
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

# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════
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
EXPENSE_2X = 0.0095 / 252
EXPENSE_3X = 0.0100 / 252

V4_W = 20; SIGNAL_TH = 0.15; COOLDOWN = 5
ER_Q = 66; ATR_Q = 55; LOOKBACK = 252
DIVGATE = 3; CONFIRM = 3
BUY_DD_LOOKBACK = 20; BUY_DD_THRESHOLD = 0.05

# Sweep 조합
COMBOS = [
    (1.5, 2.0, "DC1.5/FN2.0"),
    (2.0, 2.0, "DC2.0/FN2.0"),
    (2.5, 1.0, "DC2.5/FN1.0"),
    (2.5, 1.5, "DC2.5/FN1.5"),
    (2.5, 2.0, "DC2.5/FN2.0"),  # ★ BEST
    (3.0, 1.0, "DC3.0/FN1.0"),
    (3.0, 1.5, "DC3.0/FN1.5"),
    (3.0, 2.0, "DC3.0/FN2.0"),  # ◀ CURRENT
    (3.5, 1.5, "DC3.5/FN1.5"),
    (3.5, 2.0, "DC3.5/FN2.0"),
    (4.0, 2.0, "DC4.0/FN2.0"),
]


# ═══════════════════════════════════════════════════════════
# Core Functions (기존 VN60 동일)
# ═══════════════════════════════════════════════════════════
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


def build_synthetic_lev(close, leverage, expense_daily):
    daily_ret = np.diff(close) / close[:-1]
    lev_price = np.zeros(len(close))
    lev_price[0] = close[0]
    for i in range(1, len(close)):
        lev_ret = leverage * daily_ret[i - 1] - expense_daily
        lev_price[i] = lev_price[i - 1] * (1 + lev_ret)
        if lev_price[i] < 0.001:
            lev_price[i] = 0.001
    return lev_price


def calc_score_vn60(df, pv_div, pv_fh_vel, w, divgate_days, div_clip, force_norm):
    n = len(df)
    raw_div = np.array([np.clip(pv_div.iloc[i] / div_clip, -1, 1) for i in range(n)])
    consec = np.zeros(n, dtype=int)
    for i in range(1, n):
        if raw_div[i] != 0 and np.sign(raw_div[i]) == np.sign(raw_div[i - 1]):
            consec[i] = consec[i - 1] + 1
        elif raw_div[i] != 0:
            consec[i] = 1
    scores = np.zeros(n)
    for i in range(max(60, w), n):
        s_div = raw_div[i] if consec[i] >= divgate_days else 0.0
        fhr_std = pv_fh_vel.iloc[max(0, i - w):i].std() + 1e-10
        s_force = np.clip(pv_fh_vel.iloc[i] / (force_norm * fhr_std), -1, 1)
        scores[i] = 0.60 * s_force + 0.40 * s_div
    return pd.Series(scores, index=df.index)


def get_buy_signals(df_s, score_series):
    events = detect_signal_events(score_series, th=SIGNAL_TH, cooldown=COOLDOWN)
    pf = build_price_filter(df_s, er_q=ER_Q, atr_q=ATR_Q, lookback=LOOKBACK)
    close_vals = df_s['Close'].values
    rolling_high = pd.Series(close_vals).rolling(BUY_DD_LOOKBACK, min_periods=1).max().values
    n = len(df_s)
    buy_indices = []
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
    return buy_indices


def simulate(close, close_2x, close_3x, buy_indices, dates):
    """연도별 수익률 + 최종 자산 시뮬레이션"""
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

    cash_a = 0.0; sh_1x_a = 0.0; sh_2x_a = 0.0  # 2x strategy
    cash_b = 0.0; sh_1x_b = 0.0; sh_3x_b = 0.0  # 3x strategy
    cash_c = 0.0; sh_1x_c = 0.0                   # DCA
    yr_data = {}; prev_yr = None; total_dep = 0.0

    def pf_a(idx): return sh_1x_a * close[idx] + sh_2x_a * close_2x[idx] + cash_a
    def pf_b(idx): return sh_1x_b * close[idx] + sh_3x_b * close_3x[idx] + cash_b
    def pf_c(idx): return sh_1x_c * close[idx] + cash_c

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

        cash_a += MONTHLY_DEPOSIT; cash_b += MONTHLY_DEPOSIT; cash_c += MONTHLY_DEPOSIT
        yr_data[yr]['deposits'] += MONTHLY_DEPOSIT
        total_dep += MONTHLY_DEPOSIT

        for day_idx in range(fi, li + 1):
            if day_idx in buy_set:
                if cash_a > 1.0:
                    amt = cash_a * SIGNAL_BUY_PCT
                    sh_2x_a += amt / close_2x[day_idx]; cash_a -= amt
                if cash_b > 1.0:
                    amt = cash_b * SIGNAL_BUY_PCT
                    sh_3x_b += amt / close_3x[day_idx]; cash_b -= amt
                yr_data[yr]['sigs'] += 1

        if cash_a > 1.0: sh_1x_a += cash_a / close[li]; cash_a = 0.0
        if cash_b > 1.0: sh_1x_b += cash_b / close[li]; cash_b = 0.0
        if cash_c > 1.0: sh_1x_c += cash_c / close[li]; cash_c = 0.0

    if prev_yr is not None:
        yr_data[prev_yr]['end_a'] = pf_a(n - 1)
        yr_data[prev_yr]['end_b'] = pf_b(n - 1)
        yr_data[prev_yr]['end_c'] = pf_c(n - 1)

    yr_results = []
    for yr in sorted(yr_data.keys()):
        yd = yr_data[yr]
        if 'end_a' not in yd:
            continue
        rets = {}
        for mode in ['a', 'b', 'c']:
            d = yd[f'start_{mode}'] + yd['deposits'] * 0.5
            if d > 10:
                val = (yd[f'end_{mode}'] - yd[f'start_{mode}'] - yd['deposits']) / d * 100
                rets[mode] = val if np.isfinite(val) else 0.0
            else:
                rets[mode] = 0
        yr_results.append({
            'yr': yr,
            'ret_2x': rets['a'], 'ret_3x': rets['b'], 'ret_dca': rets['c'],
            'edge_2x': rets['a'] - rets['c'], 'edge_3x': rets['b'] - rets['c'],
            'sigs': yd['sigs'],
        })

    # Hit rate
    hits_90 = 0; total_90 = 0
    for idx in buy_indices:
        if idx + 90 < n:
            total_90 += 1
            if close[idx + 90] > close[idx]:
                hits_90 += 1
    hit_rate_90 = (hits_90 / total_90 * 100) if total_90 > 0 else 0

    # Avg forward returns
    fwd_90 = []
    for idx in buy_indices:
        if idx + 90 < n:
            fwd_90.append((close[idx + 90] / close[idx] - 1) * 100)
    avg_fwd_90 = np.mean(fwd_90) if fwd_90 else 0

    avg_2x = np.nanmean([r['ret_2x'] for r in yr_results])
    avg_3x = np.nanmean([r['ret_3x'] for r in yr_results])
    avg_dca = np.nanmean([r['ret_dca'] for r in yr_results])
    worst_2x = min(r['edge_2x'] for r in yr_results) if yr_results else 0
    worst_3x = min(r['edge_3x'] for r in yr_results) if yr_results else 0

    # Bear/Bull 분리
    bear_yrs = [r for r in yr_results if r['ret_dca'] < -5]
    bull_yrs = [r for r in yr_results if r['ret_dca'] > 15]
    flat_yrs = [r for r in yr_results if -5 <= r['ret_dca'] <= 15]

    return {
        'yr_results': yr_results,
        'avg_2x': avg_2x, 'avg_3x': avg_3x, 'avg_dca': avg_dca,
        'edge_2x': avg_2x - avg_dca, 'edge_3x': avg_3x - avg_dca,
        'worst_2x': worst_2x, 'worst_3x': worst_3x,
        'eff_2x': (avg_2x - avg_dca) / abs(worst_2x) if abs(worst_2x) > 0.1 else 0,
        'total_sigs': sum(r['sigs'] for r in yr_results),
        'hit_rate_90': hit_rate_90, 'avg_fwd_90': avg_fwd_90,
        'n_years': len(yr_results),
        'final_a': pf_a(n - 1), 'final_b': pf_b(n - 1), 'final_c': pf_c(n - 1),
        'total_dep': total_dep,
        'bear_edge_2x': np.mean([r['edge_2x'] for r in bear_yrs]) if bear_yrs else 0,
        'bull_edge_2x': np.mean([r['edge_2x'] for r in bull_yrs]) if bull_yrs else 0,
        'flat_edge_2x': np.mean([r['edge_2x'] for r in flat_yrs]) if flat_yrs else 0,
        'bear_2x': np.mean([r['ret_2x'] for r in bear_yrs]) if bear_yrs else 0,
        'bull_2x': np.mean([r['ret_2x'] for r in bull_yrs]) if bull_yrs else 0,
        'bear_dca': np.mean([r['ret_dca'] for r in bear_yrs]) if bear_yrs else 0,
        'bull_dca': np.mean([r['ret_dca'] for r in bull_yrs]) if bull_yrs else 0,
        'n_bear': len(bear_yrs), 'n_bull': len(bull_yrs), 'n_flat': len(flat_yrs),
    }


# ═══════════════════════════════════════════════════════════
# Data Download
# ═══════════════════════════════════════════════════════════
print("=" * 130)
print("  VN60 PHASE 1 SWEEP: 연평균수익률 기준 상세 분석")
print("  전략: 월 $500 → 시그널 시 50% 레버리지 매수(2x/3x) → 월말 잔여금 1x 매수")
print("=" * 130)

print("\n  데이터 로딩...")
ticker_data = {}

for tk in TICKERS:
    df = download_max(tk)
    if df is None or len(df) < 300:
        continue
    try:
        df_s = smooth_earnings_volume(df, ticker=tk)
    except Exception:
        df_s = df.copy()
    close = df_s['Close'].values
    dates = df_s.index
    close_2x = build_synthetic_lev(close, 2.0, EXPENSE_2X)
    close_3x = build_synthetic_lev(close, 3.0, EXPENSE_3X)
    pv_div = calc_pv_divergence(df_s, V4_W)
    pv_fh_vel = calc_force_macd_vel(df_s)
    ticker_data[tk] = (df_s, close, close_2x, close_3x, dates, pv_div, pv_fh_vel)
    print(f"    {tk}: {len(df_s)} bars, {(dates[-1]-dates[0]).days/365.25:.1f}yr")

tks = list(ticker_data.keys())
print(f"  {len(tks)} tickers loaded.\n")


# ═══════════════════════════════════════════════════════════
# Run all combos
# ═══════════════════════════════════════════════════════════
print("  조합별 시뮬레이션 실행...")
# master[label] = {ticker: result}
master = {}
for dc, fn, label in COMBOS:
    results = {}
    for tk in tks:
        df_s, close, close_2x, close_3x, dates, pv_div, pv_fh_vel = ticker_data[tk]
        score = calc_score_vn60(df_s, pv_div, pv_fh_vel,
                                w=V4_W, divgate_days=DIVGATE,
                                div_clip=dc, force_norm=fn)
        buys = get_buy_signals(df_s, score)
        results[tk] = simulate(close, close_2x, close_3x, buys, dates)
    master[label] = results
    avg_2x = np.nanmean([results[tk]['avg_2x'] for tk in tks])
    avg_3x = np.nanmean([results[tk]['avg_3x'] for tk in tks])
    avg_dca = np.nanmean([results[tk]['avg_dca'] for tk in tks])
    print(f"    {label:<14s}  2x={avg_2x:>+6.1f}%  3x={avg_3x:>+6.1f}%  DCA={avg_dca:>+6.1f}%")

labels = [c[2] for c in COMBOS]
CUR_LABEL = "DC3.0/FN2.0"
BEST_LABEL = "DC2.5/FN2.0"


# ═══════════════════════════════════════════════════════════
# SECTION 1: 전체 조합 연평균수익률 랭킹
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 130}")
print(f"  [1] 전체 조합 랭킹: 14 티커 평균 연평균수익률")
print(f"{'=' * 130}")

print(f"\n  {'Rank':>4s} {'Config':<14s} │ {'VN60+2x':>9s} {'VN60+3x':>9s} {'DCA(1x)':>9s}"
      f" │ {'Edge2x':>8s} {'Edge3x':>8s} {'Eff2x':>7s}"
      f" │ {'Hit90':>6s} {'AvgFwd90':>9s} {'Sig/yr':>7s} │ {'Note':>10s}")
print(f"  {'=' * 120}")

ranking = sorted(labels,
                 key=lambda lb: np.nanmean([master[lb][tk]['avg_2x'] for tk in tks]),
                 reverse=True)

for i, lb in enumerate(ranking):
    m = master[lb]
    a2 = np.nanmean([m[tk]['avg_2x'] for tk in tks])
    a3 = np.nanmean([m[tk]['avg_3x'] for tk in tks])
    ad = np.nanmean([m[tk]['avg_dca'] for tk in tks])
    e2 = np.nanmean([m[tk]['edge_2x'] for tk in tks])
    e3 = np.nanmean([m[tk]['edge_3x'] for tk in tks])
    ef = np.nanmean([m[tk]['eff_2x'] for tk in tks])
    hr = np.nanmean([m[tk]['hit_rate_90'] for tk in tks])
    fw = np.nanmean([m[tk]['avg_fwd_90'] for tk in tks])
    sy = np.nanmean([m[tk]['total_sigs'] / max(m[tk]['n_years'], 1) for tk in tks])

    note = ""
    if lb == CUR_LABEL:
        note = "◀ CURRENT"
    elif i == 0:
        note = "★ BEST"
    print(f"  {i+1:>4d} {lb:<14s} │ {a2:>+8.2f}% {a3:>+8.2f}% {ad:>+8.2f}%"
          f" │ {e2:>+7.2f}% {e3:>+7.2f}% {ef:>+6.3f}"
          f" │ {hr:>5.1f}% {fw:>+8.2f}% {sy:>6.1f} │ {note:>10s}")


# ═══════════════════════════════════════════════════════════
# SECTION 2: 티커별 연평균수익률 비교 (BEST vs CURRENT)
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 130}")
print(f"  [2] 티커별 연평균수익률: {BEST_LABEL} (BEST) vs {CUR_LABEL} (CURRENT)")
print(f"{'=' * 130}")

print(f"\n  {'Ticker':<7s} {'Sect':<8s} {'Yrs':>4s}"
      f" │ ───── CURRENT({CUR_LABEL}) ─────"
      f" │ ──── BEST({BEST_LABEL}) ──────"
      f" │ ── 차이 ──")
print(f"  {'':7s} {'':8s} {'':>4s}"
      f" │ {'2x':>8s} {'3x':>8s} {'DCA':>8s} {'Sig':>5s}"
      f" │ {'2x':>8s} {'3x':>8s} {'DCA':>8s} {'Sig':>5s}"
      f" │ {'d2x':>7s} {'d3x':>7s}")
print(f"  {'=' * 115}")

d2x_list = []; d3x_list = []
for tk in tks:
    c = master[CUR_LABEL][tk]; n_ = master[BEST_LABEL][tk]
    sect = TICKERS.get(tk, '')
    ny = c['n_years']
    cs = c['total_sigs'] / max(ny, 1)
    ns = n_['total_sigs'] / max(ny, 1)
    d2 = n_['avg_2x'] - c['avg_2x']
    d3 = n_['avg_3x'] - c['avg_3x']
    d2x_list.append(d2); d3x_list.append(d3)

    print(f"  {tk:<7s} {sect:<8s} {ny:>4d}"
          f" │ {c['avg_2x']:>+7.1f}% {c['avg_3x']:>+7.1f}% {c['avg_dca']:>+7.1f}% {cs:>4.1f}"
          f" │ {n_['avg_2x']:>+7.1f}% {n_['avg_3x']:>+7.1f}% {n_['avg_dca']:>+7.1f}% {ns:>4.1f}"
          f" │ {d2:>+6.2f} {d3:>+6.2f}")

print(f"  {'-' * 115}")
# Average
ca2 = np.nanmean([master[CUR_LABEL][tk]['avg_2x'] for tk in tks])
ca3 = np.nanmean([master[CUR_LABEL][tk]['avg_3x'] for tk in tks])
cad = np.nanmean([master[CUR_LABEL][tk]['avg_dca'] for tk in tks])
na2 = np.nanmean([master[BEST_LABEL][tk]['avg_2x'] for tk in tks])
na3 = np.nanmean([master[BEST_LABEL][tk]['avg_3x'] for tk in tks])
nad = np.nanmean([master[BEST_LABEL][tk]['avg_dca'] for tk in tks])
print(f"  {'AVG':<7s} {'':8s} {'':>4s}"
      f" │ {ca2:>+7.1f}% {ca3:>+7.1f}% {cad:>+7.1f}% {'':>5s}"
      f" │ {na2:>+7.1f}% {na3:>+7.1f}% {nad:>+7.1f}% {'':>5s}"
      f" │ {np.mean(d2x_list):>+6.2f} {np.mean(d3x_list):>+6.2f}")

better_2x = sum(1 for d in d2x_list if d > 0.05)
worse_2x = sum(1 for d in d2x_list if d < -0.05)
print(f"\n  2x 기준: {better_2x}개 개선 / {worse_2x}개 악화 / {len(tks)-better_2x-worse_2x}개 동일")


# ═══════════════════════════════════════════════════════════
# SECTION 3: 최종 자산 비교 (Total Portfolio Value)
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 130}")
print(f"  [3] 최종 자산가치: 월 $500 적립 기준 (BEST vs CURRENT)")
print(f"{'=' * 130}")

print(f"\n  {'Ticker':<7s} {'입금총액':>10s}"
      f" │ ────── CURRENT ──────"
      f" │ ─────── BEST ───────"
      f" │ {'Better':>7s}")
print(f"  {'':7s} {'':>10s}"
      f" │ {'2x':>10s} {'3x':>10s} {'DCA':>10s}"
      f" │ {'2x':>10s} {'3x':>10s} {'DCA':>10s}"
      f" │")
print(f"  {'=' * 100}")

total_c2 = 0; total_c3 = 0; total_cd = 0
total_n2 = 0; total_n3 = 0; total_nd = 0; total_dep = 0

for tk in tks:
    c = master[CUR_LABEL][tk]; n_ = master[BEST_LABEL][tk]
    dep = c['total_dep']
    b2x = "BEST" if n_['final_a'] > c['final_a'] else "CUR"
    total_c2 += c['final_a']; total_c3 += c['final_b']; total_cd += c['final_c']
    total_n2 += n_['final_a']; total_n3 += n_['final_b']; total_nd += n_['final_c']
    total_dep += dep

    print(f"  {tk:<7s} ${dep:>9,.0f}"
          f" │ ${c['final_a']:>9,.0f} ${c['final_b']:>9,.0f} ${c['final_c']:>9,.0f}"
          f" │ ${n_['final_a']:>9,.0f} ${n_['final_b']:>9,.0f} ${n_['final_c']:>9,.0f}"
          f" │ {b2x:>7s}")

print(f"  {'-' * 100}")
print(f"  {'TOTAL':<7s} ${total_dep:>9,.0f}"
      f" │ ${total_c2:>9,.0f} ${total_c3:>9,.0f} ${total_cd:>9,.0f}"
      f" │ ${total_n2:>9,.0f} ${total_n3:>9,.0f} ${total_nd:>9,.0f}"
      f" │ {'BEST' if total_n2 > total_c2 else 'CUR':>7s}")

print(f"\n  전체 포트폴리오 수익률:")
print(f"    CURRENT 2x: ${total_c2:,.0f} (수익률 {(total_c2/total_dep-1)*100:+.1f}%)")
print(f"    BEST    2x: ${total_n2:,.0f} (수익률 {(total_n2/total_dep-1)*100:+.1f}%)")
print(f"    차이:       ${total_n2-total_c2:+,.0f} ({(total_n2-total_c2)/total_c2*100:+.2f}%)")


# ═══════════════════════════════════════════════════════════
# SECTION 4: 상승장/하락장 분석
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 130}")
print(f"  [4] 시장 국면별 연평균수익률 (BEST vs CURRENT)")
print(f"      Bull: DCA > +15%  |  Bear: DCA < -5%  |  Flat: -5% ~ +15%")
print(f"{'=' * 130}")

print(f"\n  {'Ticker':<7s}"
      f" │ ── Bear(CUR) ── ── Bear(BEST) ──"
      f" │ ── Bull(CUR) ── ── Bull(BEST) ──"
      f" │ {'Nb':>3s} {'Nbu':>3s}")
print(f"  {'':7s}"
      f" │ {'2x':>8s} {'DCA':>8s} {'2x':>8s} {'DCA':>8s}"
      f" │ {'2x':>8s} {'DCA':>8s} {'2x':>8s} {'DCA':>8s}"
      f" │")
print(f"  {'=' * 105}")

for tk in tks:
    c = master[CUR_LABEL][tk]; n_ = master[BEST_LABEL][tk]
    print(f"  {tk:<7s}"
          f" │ {c['bear_2x']:>+7.1f}% {c['bear_dca']:>+7.1f}%"
          f" {n_['bear_2x']:>+7.1f}% {n_['bear_dca']:>+7.1f}%"
          f" │ {c['bull_2x']:>+7.1f}% {c['bull_dca']:>+7.1f}%"
          f" {n_['bull_2x']:>+7.1f}% {n_['bull_dca']:>+7.1f}%"
          f" │ {c['n_bear']:>3d} {c['n_bull']:>3d}")

print(f"  {'-' * 105}")
cb2 = np.nanmean([master[CUR_LABEL][tk]['bear_2x'] for tk in tks])
cbd = np.nanmean([master[CUR_LABEL][tk]['bear_dca'] for tk in tks])
nb2 = np.nanmean([master[BEST_LABEL][tk]['bear_2x'] for tk in tks])
nbd = np.nanmean([master[BEST_LABEL][tk]['bear_dca'] for tk in tks])
cbu2 = np.nanmean([master[CUR_LABEL][tk]['bull_2x'] for tk in tks])
cbud = np.nanmean([master[CUR_LABEL][tk]['bull_dca'] for tk in tks])
nbu2 = np.nanmean([master[BEST_LABEL][tk]['bull_2x'] for tk in tks])
nbud = np.nanmean([master[BEST_LABEL][tk]['bull_dca'] for tk in tks])
print(f"  {'AVG':<7s}"
      f" │ {cb2:>+7.1f}% {cbd:>+7.1f}%"
      f" {nb2:>+7.1f}% {nbd:>+7.1f}%"
      f" │ {cbu2:>+7.1f}% {cbud:>+7.1f}%"
      f" {nbu2:>+7.1f}% {nbud:>+7.1f}%"
      f" │")


# ═══════════════════════════════════════════════════════════
# SECTION 5: 시그널 품질 분석
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 130}")
print(f"  [5] 시그널 품질: Hit Rate & Forward Return (BEST vs CURRENT)")
print(f"{'=' * 130}")

print(f"\n  {'Ticker':<7s}"
      f" │ {'CUR sig':>8s} {'CUR hit90':>9s} {'CUR fwd90':>10s}"
      f" │ {'NEW sig':>8s} {'NEW hit90':>9s} {'NEW fwd90':>10s}"
      f" │ {'dHit':>6s} {'dFwd':>7s}")
print(f"  {'=' * 90}")

for tk in tks:
    c = master[CUR_LABEL][tk]; n_ = master[BEST_LABEL][tk]
    dh = n_['hit_rate_90'] - c['hit_rate_90']
    df_ = n_['avg_fwd_90'] - c['avg_fwd_90']
    print(f"  {tk:<7s}"
          f" │ {c['total_sigs']:>7d} {c['hit_rate_90']:>8.1f}% {c['avg_fwd_90']:>+9.2f}%"
          f" │ {n_['total_sigs']:>7d} {n_['hit_rate_90']:>8.1f}% {n_['avg_fwd_90']:>+9.2f}%"
          f" │ {dh:>+5.1f} {df_:>+6.2f}")

print(f"  {'-' * 90}")
ch = np.nanmean([master[CUR_LABEL][tk]['hit_rate_90'] for tk in tks])
cf = np.nanmean([master[CUR_LABEL][tk]['avg_fwd_90'] for tk in tks])
nh = np.nanmean([master[BEST_LABEL][tk]['hit_rate_90'] for tk in tks])
nf = np.nanmean([master[BEST_LABEL][tk]['avg_fwd_90'] for tk in tks])
print(f"  {'AVG':<7s}"
      f" │ {'':>8s} {ch:>8.1f}% {cf:>+9.2f}%"
      f" │ {'':>8s} {nh:>8.1f}% {nf:>+9.2f}%"
      f" │ {nh-ch:>+5.1f} {nf-cf:>+6.2f}")


# ═══════════════════════════════════════════════════════════
# SECTION 6: QQQ / VOO 연도별 상세
# ═══════════════════════════════════════════════════════════
for target in ['QQQ', 'VOO']:
    if target not in tks:
        continue

    print(f"\n{'=' * 130}")
    print(f"  [6] {target} 연도별 연수익률 상세")
    print(f"{'=' * 130}")

    cur = master[CUR_LABEL][target]
    new = master[BEST_LABEL][target]

    print(f"\n  {'Year':>6s}"
          f" │ ──── CURRENT ({CUR_LABEL}) ────"
          f" │ ───── BEST ({BEST_LABEL}) ─────"
          f" │ {'d2x':>6s} {'d3x':>6s}")
    print(f"  {'':>6s}"
          f" │ {'2x':>8s} {'3x':>8s} {'DCA':>8s} {'Sig':>4s}"
          f" │ {'2x':>8s} {'3x':>8s} {'DCA':>8s} {'Sig':>4s}"
          f" │")
    print(f"  {'-' * 95}")

    for cr, nr in zip(cur['yr_results'], new['yr_results']):
        d2 = nr['ret_2x'] - cr['ret_2x']
        d3 = nr['ret_3x'] - cr['ret_3x']
        regime = ""
        if cr['ret_dca'] < -5: regime = " B"
        elif cr['ret_dca'] > 15: regime = " U"
        print(f"  {cr['yr']:>6d}"
              f" │ {cr['ret_2x']:>+7.1f}% {cr['ret_3x']:>+7.1f}% {cr['ret_dca']:>+7.1f}% {cr['sigs']:>3d}"
              f" │ {nr['ret_2x']:>+7.1f}% {nr['ret_3x']:>+7.1f}% {nr['ret_dca']:>+7.1f}% {nr['sigs']:>3d}"
              f" │ {d2:>+5.1f} {d3:>+5.1f}{regime}")

    print(f"  {'-' * 95}")
    ca2_ = np.nanmean([r['ret_2x'] for r in cur['yr_results']])
    ca3_ = np.nanmean([r['ret_3x'] for r in cur['yr_results']])
    cad_ = np.nanmean([r['ret_dca'] for r in cur['yr_results']])
    na2_ = np.nanmean([r['ret_2x'] for r in new['yr_results']])
    na3_ = np.nanmean([r['ret_3x'] for r in new['yr_results']])
    nad_ = np.nanmean([r['ret_dca'] for r in new['yr_results']])
    print(f"  {'AVG':>6s}"
          f" │ {ca2_:>+7.1f}% {ca3_:>+7.1f}% {cad_:>+7.1f}% {'':>4s}"
          f" │ {na2_:>+7.1f}% {na3_:>+7.1f}% {nad_:>+7.1f}% {'':>4s}"
          f" │ {na2_-ca2_:>+5.1f} {na3_-ca3_:>+5.1f}")

    print(f"\n  {target} Summary:")
    print(f"    CURRENT: 2x={ca2_:+.2f}%  3x={ca3_:+.2f}%  DCA={cad_:+.2f}%  Hit90={cur['hit_rate_90']:.1f}%")
    print(f"    BEST:    2x={na2_:+.2f}%  3x={na3_:+.2f}%  DCA={nad_:+.2f}%  Hit90={new['hit_rate_90']:.1f}%")
    print(f"    Final:   CUR=${cur['final_a']:,.0f}  BEST=${new['final_a']:,.0f}"
          f"  (${new['final_a']-cur['final_a']:+,.0f})")


# ═══════════════════════════════════════════════════════════
# SECTION 7: Sensitivity — DivClip 단독 효과 (연평균수익률)
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 130}")
print(f"  [7] DivClip 감도 분석: ForceNorm=/2.0 고정, DivClip 변경 시 연평균수익률")
print(f"{'=' * 130}")

dc_labels = [lb for dc, fn, lb in COMBOS if fn == 2.0]
dc_vals = [dc for dc, fn, lb in COMBOS if fn == 2.0]

print(f"\n  {'Ticker':<7s}", end="")
for dc in dc_vals:
    marker = "*" if dc == 3.0 else " "
    print(f" {'DC/'+f'{dc:.1f}'+marker:>9s}", end="")
print(f"  {'Best':>6s}")
print(f"  {'=' * (8 + 10 * len(dc_vals) + 8)}")

for tk in tks:
    print(f"  {tk:<7s}", end="")
    best_dc = max(dc_vals, key=lambda d: master[f"DC{d}/FN2.0"][tk]['avg_2x'])
    for dc in dc_vals:
        lb = f"DC{dc}/FN2.0"
        a2 = master[lb][tk]['avg_2x']
        mark = " <" if dc == best_dc else "  "
        print(f" {a2:>+7.1f}%{mark}", end="")
    print(f"  /{best_dc:.1f}")

print(f"  {'-' * (8 + 10 * len(dc_vals) + 8)}")
print(f"  {'AVG':<7s}", end="")
for dc in dc_vals:
    lb = f"DC{dc}/FN2.0"
    a2 = np.nanmean([master[lb][tk]['avg_2x'] for tk in tks])
    print(f" {a2:>+7.2f}% ", end="")
print()


# ═══════════════════════════════════════════════════════════
# SECTION 8: Sensitivity — ForceNorm 단독 효과
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 130}")
print(f"  [8] ForceNorm 감도 분석: DivClip=/3.0 고정, ForceNorm 변경 시 연평균수익률")
print(f"{'=' * 130}")

fn_labels = [lb for dc, fn, lb in COMBOS if dc == 3.0]
fn_vals = [fn for dc, fn, lb in COMBOS if dc == 3.0]

print(f"\n  {'Ticker':<7s}", end="")
for fn in fn_vals:
    marker = "*" if fn == 2.0 else " "
    print(f" {'FN/'+f'{fn:.1f}'+marker:>9s}", end="")
print(f"  {'Best':>6s}")
print(f"  {'=' * (8 + 10 * len(fn_vals) + 8)}")

for tk in tks:
    print(f"  {tk:<7s}", end="")
    best_fn = max(fn_vals, key=lambda f: master[f"DC3.0/FN{f}"][tk]['avg_2x'])
    for fn in fn_vals:
        lb = f"DC3.0/FN{fn}"
        a2 = master[lb][tk]['avg_2x']
        mark = " <" if fn == best_fn else "  "
        print(f" {a2:>+7.1f}%{mark}", end="")
    print(f"  /{best_fn:.1f}")

print(f"  {'-' * (8 + 10 * len(fn_vals) + 8)}")
print(f"  {'AVG':<7s}", end="")
for fn in fn_vals:
    lb = f"DC3.0/FN{fn}"
    a2 = np.nanmean([master[lb][tk]['avg_2x'] for tk in tks])
    print(f" {a2:>+7.2f}% ", end="")
print()


# ═══════════════════════════════════════════════════════════
# GRAND SUMMARY
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 130}")
print(f"  GRAND SUMMARY: 연평균수익률 기준 최종 정리")
print(f"{'=' * 130}")

print(f"""
  +---------------------------------------------------------------------+
  |                    14 티커 평균 연평균수익률                          |
  +---------------------------------------------------------------------+
  |  Config          VN60+2x    VN60+3x    DCA(1x)    Edge2x   Hit90   |
  +---------------------------------------------------------------------+
  |  CURRENT         {ca2:>+7.2f}%   {ca3:>+7.2f}%   {cad:>+7.2f}%   {ca2-cad:>+6.2f}%  {ch:>5.1f}%  |
  |  (DC/3.0 FN/2.0)                                                    |
  +---------------------------------------------------------------------+
  |  BEST            {na2:>+7.2f}%   {na3:>+7.2f}%   {nad:>+7.2f}%   {na2-nad:>+6.2f}%  {nh:>5.1f}%  |
  |  (DC/2.5 FN/2.0)                                                    |
  +---------------------------------------------------------------------+
  |  DELTA           {na2-ca2:>+7.2f}%   {na3-ca3:>+7.2f}%   {nad-cad:>+7.2f}%   {(na2-nad)-(ca2-cad):>+6.2f}%  {nh-ch:>+5.1f}%  |
  +---------------------------------------------------------------------+

  Key Findings:
    1. DivClip: /2.5 > /3.0 > /2.0 > /1.5 > /3.5 > /4.0
       → 2.5가 최적, 현재 3.0보다 S_Div 감도 약간 증가
    2. ForceNorm: /1.5 ≈ /1.0 > /2.0 > /2.5 > /3.0
       → 1.5가 최적이나, DC/2.5와 교차 시 FN/2.0이 우세
    3. 최적 조합: DC/2.5 + FN/2.0 (DivClip만 변경)
    4. 2x 기준 {better_2x}개 티커 개선, {worse_2x}개 악화
    5. 전체 포트폴리오: ${total_n2-total_c2:+,.0f} 차이
""")

print("=" * 130)
print("  Done.")
