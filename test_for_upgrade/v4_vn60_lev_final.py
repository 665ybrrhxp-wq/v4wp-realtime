"""
VN60 vs DCA: 전체 티커 매년 연수익률 비교 (1x DCA / 2x / 3x)
DCA  = 월 $500 1x 매수
VN60+2x = 시그널 시 2x(50%) + 월말 1x
VN60+3x = 시그널 시 3x(50%) + 월말 1x
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
EXPENSE_2X = 0.0095 / 252   # ~0.95% annual
EXPENSE_3X = 0.0100 / 252   # ~1.00% annual (TQQQ 등 3x ETF)
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


def build_synthetic_lev(close, leverage, expense_daily):
    """합성 레버리지 가격 시리즈 (일간 리밸런싱)"""
    daily_ret = np.diff(close) / close[:-1]
    lev_price = np.zeros(len(close))
    lev_price[0] = close[0]
    for i in range(1, len(close)):
        lev_ret = leverage * daily_ret[i - 1] - expense_daily
        lev_price[i] = lev_price[i - 1] * (1 + lev_ret)
        if lev_price[i] < 0.001:
            lev_price[i] = 0.001
    return lev_price


def get_buy_signals(df, ticker):
    try:
        df_s = smooth_earnings_volume(df, ticker=ticker)
    except Exception:
        df_s = df.copy()
    n = len(df_s)
    pv_div = calc_pv_divergence(df_s, V4_W)
    pv_fh_vel = calc_force_macd_vel(df_s)
    raw_div = pv_div.values if hasattr(pv_div, 'values') else np.array(pv_div)
    consec = np.zeros(n, dtype=int)
    for i in range(1, n):
        if raw_div[i] != 0 and np.sign(raw_div[i]) == np.sign(raw_div[i - 1]):
            consec[i] = consec[i - 1] + 1
        elif raw_div[i] != 0:
            consec[i] = 1
    scores = np.zeros(n)
    for i in range(max(60, V4_W), n):
        s_div = raw_div[i] if consec[i] >= DIVGATE else 0.0
        fhr_std = pv_fh_vel.iloc[max(0, i - V4_W):i].std() + 1e-10
        s_force = np.clip(pv_fh_vel.iloc[i] / (2 * fhr_std), -1, 1)
        scores[i] = 0.60 * s_force + 0.40 * s_div
    score_series = pd.Series(scores, index=df_s.index)
    events = detect_signal_events(score_series, th=SIGNAL_TH, cooldown=COOLDOWN)
    pf = build_price_filter(df_s, er_q=ER_Q, atr_q=ATR_Q, lookback=LOOKBACK)
    close_vals = df_s['Close'].values
    rolling_high = pd.Series(close_vals).rolling(BUY_DD_LOOKBACK, min_periods=1).max().values

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
    """DCA / VN60+2x / VN60+3x 동시 시뮬레이션"""
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

    # DCA (c), VN60+2x (a), VN60+3x (b)
    cash_a = 0.0; sh_1x_a = 0.0; sh_2x_a = 0.0
    cash_b = 0.0; sh_1x_b = 0.0; sh_3x_b = 0.0
    cash_c = 0.0; sh_1x_c = 0.0
    yr_data = {}; prev_yr = None

    def pf_a(idx):
        return sh_1x_a * close[idx] + sh_2x_a * close_2x[idx] + cash_a
    def pf_b(idx):
        return sh_1x_b * close[idx] + sh_3x_b * close_3x[idx] + cash_b
    def pf_c(idx):
        return sh_1x_c * close[idx] + cash_c

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

        for day_idx in range(fi, li + 1):
            if day_idx in buy_set:
                if cash_a > 1.0:
                    amt = cash_a * SIGNAL_BUY_PCT
                    sh_2x_a += amt / close_2x[day_idx]
                    cash_a -= amt
                if cash_b > 1.0:
                    amt = cash_b * SIGNAL_BUY_PCT
                    sh_3x_b += amt / close_3x[day_idx]
                    cash_b -= amt
                yr_data[yr]['sigs'] += 1

        if cash_a > 1.0:
            sh_1x_a += cash_a / close[li]; cash_a = 0.0
        if cash_b > 1.0:
            sh_1x_b += cash_b / close[li]; cash_b = 0.0
        if cash_c > 1.0:
            sh_1x_c += cash_c / close[li]; cash_c = 0.0

    if prev_yr is not None:
        yr_data[prev_yr]['end_a'] = pf_a(n - 1)
        yr_data[prev_yr]['end_b'] = pf_b(n - 1)
        yr_data[prev_yr]['end_c'] = pf_c(n - 1)

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
            'dca': yd['ret_c'],
            'v2x': yd['ret_a'],
            'v3x': yd['ret_b'],
            'd2x': yd['ret_a'] - yd['ret_c'],
            'd3x': yd['ret_b'] - yd['ret_c'],
            'sigs': yd['sigs'],
        })

    return yr_results


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
print("=" * 85)
print("  VN60 Leverage Compare: DCA vs 2x vs 3x")
print("  DCA  = 월 $500 → 매월 1x 매수")
print("  2x   = 월 $500 → 시그널 시 2x 매수(50%) + 월말 1x")
print("  3x   = 월 $500 → 시그널 시 3x 매수(50%) + 월말 1x")
print("=" * 85)

all_results = {}

for tk in TICKERS:
    df = download_max(tk)
    if df is None or len(df) < 300:
        continue
    close = df['Close'].values
    dates = df.index
    close_2x = build_synthetic_lev(close, 2.0, EXPENSE_2X)
    close_3x = build_synthetic_lev(close, 3.0, EXPENSE_3X)
    buy_indices = get_buy_signals(df, tk)
    yr = simulate(close, close_2x, close_3x, buy_indices, dates)
    all_results[tk] = yr
    print(f"  {tk} ({len(yr)}yr, {sum(r['sigs'] for r in yr)} sigs)")

tks = list(all_results.keys())


# ═══════════════════════════════════════════════════════════
# Per-Ticker Year-by-Year
# ═══════════════════════════════════════════════════════════
for tk in tks:
    yr = all_results[tk]
    avg_dca = np.nanmean([r['dca'] for r in yr])
    avg_2x = np.nanmean([r['v2x'] for r in yr])
    avg_3x = np.nanmean([r['v3x'] for r in yr])
    avg_d2x = np.nanmean([r['d2x'] for r in yr])
    avg_d3x = np.nanmean([r['d3x'] for r in yr])
    total_sigs = sum(r['sigs'] for r in yr)

    print(f"\n{'='*85}")
    print(f"  {tk}  ({len(yr)}년, {total_sigs} 시그널)")
    print(f"{'='*85}")
    print(f"  {'Year':>6s}  {'DCA':>9s}  {'VN60+2x':>9s}  {'VN60+3x':>9s} │ {'2x-DCA':>9s}  {'3x-DCA':>9s}  {'Sigs':>4s}")
    print(f"  {'-'*68}")

    for r in yr:
        print(f"  {r['yr']:>6d}  {r['dca']:>+7.1f}%  {r['v2x']:>+7.1f}%  {r['v3x']:>+7.1f}% │"
              f" {r['d2x']:>+7.2f}%p {r['d3x']:>+7.2f}%p  {r['sigs']:>4d}")

    print(f"  {'-'*68}")
    print(f"  {'AVG':>6s}  {avg_dca:>+7.1f}%  {avg_2x:>+7.1f}%  {avg_3x:>+7.1f}% │"
          f" {avg_d2x:>+7.2f}%p {avg_d3x:>+7.2f}%p")

    worst_2x = min(r['d2x'] for r in yr)
    worst_3x = min(r['d3x'] for r in yr)
    eff_2x = avg_d2x / abs(worst_2x) if abs(worst_2x) > 0.1 else 0
    eff_3x = avg_d3x / abs(worst_3x) if abs(worst_3x) > 0.1 else 0
    win_2x = sum(1 for r in yr if r['d2x'] > 0.1)
    win_3x = sum(1 for r in yr if r['d3x'] > 0.1)
    lose_2x = sum(1 for r in yr if r['d2x'] < -0.1)
    lose_3x = sum(1 for r in yr if r['d3x'] < -0.1)
    print(f"  2x: Eff={eff_2x:+.3f}  W/L={win_2x}/{lose_2x}  Worst={worst_2x:+.2f}%p")
    print(f"  3x: Eff={eff_3x:+.3f}  W/L={win_3x}/{lose_3x}  Worst={worst_3x:+.2f}%p")


# ═══════════════════════════════════════════════════════════
# GRAND SUMMARY
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*85}")
print(f"  GRAND SUMMARY: 전체 티커 연평균수익률")
print(f"{'='*85}")

print(f"\n  {'Ticker':<7s} {'Yrs':>4s} {'Sig':>4s} │ {'DCA':>8s} {'VN60+2x':>8s} {'VN60+3x':>8s} │"
      f" {'2x Edge':>8s} {'3x Edge':>8s} │ {'2x Eff':>7s} {'3x Eff':>7s} │ {'Best':>4s}")
print(f"  {'-'*90}")

g_dca=[]; g_2x=[]; g_3x=[]; g_d2=[]; g_d3=[]; g_e2=[]; g_e3=[]
wins_2x = 0; wins_3x = 0

for tk in tks:
    yr = all_results[tk]
    avg_dca = np.nanmean([r['dca'] for r in yr])
    avg_2x = np.nanmean([r['v2x'] for r in yr])
    avg_3x = np.nanmean([r['v3x'] for r in yr])
    avg_d2 = np.nanmean([r['d2x'] for r in yr])
    avg_d3 = np.nanmean([r['d3x'] for r in yr])
    worst_2 = min(r['d2x'] for r in yr)
    worst_3 = min(r['d3x'] for r in yr)
    eff_2 = avg_d2 / abs(worst_2) if abs(worst_2) > 0.1 else 0
    eff_3 = avg_d3 / abs(worst_3) if abs(worst_3) > 0.1 else 0
    total_sigs = sum(r['sigs'] for r in yr)

    best = "3x" if eff_3 > eff_2 else "2x" if eff_2 > eff_3 else "TIE"
    if eff_3 > eff_2:
        wins_3x += 1
    else:
        wins_2x += 1

    g_dca.append(avg_dca); g_2x.append(avg_2x); g_3x.append(avg_3x)
    g_d2.append(avg_d2); g_d3.append(avg_d3); g_e2.append(eff_2); g_e3.append(eff_3)

    print(f"  {tk:<7s} {len(yr):>4d} {total_sigs:>4d} │ {avg_dca:>+6.1f}% {avg_2x:>+6.1f}% {avg_3x:>+6.1f}% │"
          f" {avg_d2:>+6.2f}%p {avg_d3:>+6.2f}%p │ {eff_2:>+6.3f} {eff_3:>+6.3f} │ {best:>4s}")

print(f"  {'-'*90}")
print(f"  {'AVG':<7s} {'':>4s} {'':>4s} │ {np.nanmean(g_dca):>+6.1f}% {np.nanmean(g_2x):>+6.1f}% {np.nanmean(g_3x):>+6.1f}% │"
      f" {np.nanmean(g_d2):>+6.2f}%p {np.nanmean(g_d3):>+6.2f}%p │ {np.nanmean(g_e2):>+6.3f} {np.nanmean(g_e3):>+6.3f} │")

print(f"""
  +-------------------------------------------------------+
  |                  연평균수익률 요약                      |
  +-------------------------------------------------------+
  |  DCA (1x):          {np.nanmean(g_dca):>+7.1f}% / year                |
  |  VN60 + 2x:         {np.nanmean(g_2x):>+7.1f}% / year                |
  |  VN60 + 3x:         {np.nanmean(g_3x):>+7.1f}% / year                |
  +-------------------------------------------------------+
  |  2x Edge vs DCA:    {np.nanmean(g_d2):>+7.2f}%p   Eff: {np.nanmean(g_e2):>+6.3f}         |
  |  3x Edge vs DCA:    {np.nanmean(g_d3):>+7.2f}%p   Eff: {np.nanmean(g_e3):>+6.3f}         |
  +-------------------------------------------------------+
  |  2x wins: {wins_2x}/14     3x wins: {wins_3x}/14                  |
  +-------------------------------------------------------+
""")
print("=" * 85)
print("  Done.")
