"""
VN60 Leverage Summary: QQQ/VOO 상세 + 전체 연평균수익률 요약
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


def simulate(close, close_2x, buy_indices, dates):
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

    # A: VN60+2x, B: VN60+1x, C: Pure DCA
    cash_a = 0.0; sh_1x_a = 0.0; sh_2x_a = 0.0
    cash_b = 0.0; sh_1x_b = 0.0
    cash_c = 0.0; sh_1x_c = 0.0
    sig_count = 0

    yr_data = {}; prev_yr = None

    def pf_a(idx):
        return sh_1x_a * close[idx] + sh_2x_a * close_2x[idx] + cash_a
    def pf_b(idx):
        return sh_1x_b * close[idx] + cash_b
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
                    sig_count += 1
                    yr_data[yr]['sigs'] += 1
                if cash_b > 1.0:
                    amt = cash_b * SIGNAL_BUY_PCT
                    sh_1x_b += amt / close[day_idx]
                    cash_b -= amt

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

    final_a = pf_a(n - 1); final_b = pf_b(n - 1); final_c = pf_c(n - 1)
    val_2x_end = sh_2x_a * close_2x[n - 1]

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
            'ret_a': rets['a'], 'ret_b': rets['b'], 'ret_c': rets['c'],
            'edge_a': rets['a'] - rets['c'],
            'edge_b': rets['b'] - rets['c'],
            'lev_add': rets['a'] - rets['b'],
            'sigs': yd['sigs'],
        })

    return {
        'yr_results': yr_results,
        'final_a': final_a, 'final_b': final_b, 'final_c': final_c,
        'sig_count': sig_count,
        'val_2x_end': val_2x_end,
    }


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
print("=" * 100)
print("  VN60 LEVERAGE SUMMARY: 전체 티커 연평균수익률 + QQQ/VOO 상세")
print("  월초 $500 입금 → 시그널 시 2x 매수(50%) → 월말 잔여자금 1x 매수")
print("=" * 100)

all_results = {}

for tk, sector in TICKERS.items():
    print(f"  {tk}...", end=" ", flush=True)
    df = download_max(tk)
    if df is None or len(df) < 300:
        print("SKIP"); continue

    close = df['Close'].values
    dates = df.index
    close_2x = build_synthetic_2x(close)
    buy_indices = get_buy_signals(df, tk)
    res = simulate(close, close_2x, buy_indices, dates)
    all_results[tk] = res
    print(f"OK ({res['sig_count']} sigs, {len(res['yr_results'])} years)")

tks = list(all_results.keys())


# ═══════════════════════════════════════════════════════════
# SECTION 1: 전체 티커 연평균수익률
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*100}")
print(f"  SECTION 1: 전체 티커 연평균수익률 (Arithmetic Mean of Annual Returns)")
print(f"{'='*100}")

print(f"\n  {'Ticker':<7s} {'Years':>5s} {'Sigs':>5s} │ {'A(2x)':>9s} {'B(1x)':>9s} {'C(DCA)':>9s} │"
      f" {'Total':>8s} {'Timing':>8s} {'Lev':>8s} │ {'A Worst':>8s} {'A Eff':>7s}")
print(f"  {'-'*100}")

sum_a = []; sum_b = []; sum_c = []; sum_ea = []; sum_eb = []; sum_lev = []

for tk in tks:
    res = all_results[tk]
    yr = res['yr_results']
    n_yr = len(yr)

    avg_a = np.nanmean([r['ret_a'] for r in yr])
    avg_b = np.nanmean([r['ret_b'] for r in yr])
    avg_c = np.nanmean([r['ret_c'] for r in yr])
    avg_ea = np.nanmean([r['edge_a'] for r in yr])
    avg_eb = np.nanmean([r['edge_b'] for r in yr])
    avg_lev = np.nanmean([r['lev_add'] for r in yr])
    worst_ea = min(r['edge_a'] for r in yr)
    eff = avg_ea / abs(worst_ea) if abs(worst_ea) > 0.1 else 0

    sum_a.append(avg_a); sum_b.append(avg_b); sum_c.append(avg_c)
    sum_ea.append(avg_ea); sum_eb.append(avg_eb); sum_lev.append(avg_lev)

    print(f"  {tk:<7s} {n_yr:>5d} {res['sig_count']:>5d} │"
          f" {avg_a:>+7.1f}% {avg_b:>+7.1f}% {avg_c:>+7.1f}% │"
          f" {avg_ea:>+6.2f}%p {avg_eb:>+6.2f}%p {avg_lev:>+6.2f}%p │"
          f" {worst_ea:>+6.2f}% {eff:>+6.3f}")

print(f"  {'-'*100}")
print(f"  {'AVG':<7s} {'':>5s} {'':>5s} │"
      f" {np.nanmean(sum_a):>+7.1f}% {np.nanmean(sum_b):>+7.1f}% {np.nanmean(sum_c):>+7.1f}% │"
      f" {np.nanmean(sum_ea):>+6.2f}%p {np.nanmean(sum_eb):>+6.2f}%p {np.nanmean(sum_lev):>+6.2f}%p │")


# ═══════════════════════════════════════════════════════════
# SECTION 2: QQQ Year-by-Year
# ═══════════════════════════════════════════════════════════
for target_tk in ['QQQ', 'VOO']:
    if target_tk not in all_results:
        continue
    res = all_results[target_tk]
    yr = res['yr_results']

    print(f"\n{'='*100}")
    print(f"  SECTION: {target_tk} Year-by-Year 상세 (연평균수익률 기준)")
    print(f"{'='*100}")

    print(f"\n    {'Year':>6s} │ {'A(VN60+2x)':>12s} {'B(VN60+1x)':>12s} {'C(DCA)':>10s} │"
          f" {'Total Edge':>11s} {'Timing':>9s} {'Leverage':>9s} │ {'Sigs':>4s}")
    print(f"    {'-'*90}")

    bear_lev = []; bull_lev = []

    for r in yr:
        marker = ""
        if r['ret_c'] < -5:
            marker = " BEAR"
        elif r['ret_c'] > 15:
            marker = " BULL"

        print(f"    {r['yr']:>6d} │ {r['ret_a']:>+10.1f}% {r['ret_b']:>+10.1f}% {r['ret_c']:>+8.1f}% │"
              f" {r['edge_a']:>+9.2f}%p {r['edge_b']:>+7.2f}%p {r['lev_add']:>+7.2f}%p │ {r['sigs']:>4d}{marker}")

        if r['ret_c'] < -5:
            bear_lev.append(r['lev_add'])
        elif r['ret_c'] > 5:
            bull_lev.append(r['lev_add'])

    avg_a = np.nanmean([r['ret_a'] for r in yr])
    avg_b = np.nanmean([r['ret_b'] for r in yr])
    avg_c = np.nanmean([r['ret_c'] for r in yr])
    avg_ea = np.nanmean([r['edge_a'] for r in yr])
    avg_eb = np.nanmean([r['edge_b'] for r in yr])
    avg_lev = np.nanmean([r['lev_add'] for r in yr])
    worst_ea = min(r['edge_a'] for r in yr)
    best_lev = max(yr, key=lambda r: r['lev_add'])
    worst_lev_r = min(yr, key=lambda r: r['lev_add'])

    print(f"    {'-'*90}")
    print(f"    {'AVG':>6s} │ {avg_a:>+10.1f}% {avg_b:>+10.1f}% {avg_c:>+8.1f}% │"
          f" {avg_ea:>+9.2f}%p {avg_eb:>+7.2f}%p {avg_lev:>+7.2f}%p │")

    print(f"\n    Summary:")
    print(f"      연평균 수익률: A(VN60+2x) {avg_a:+.1f}%  B(VN60+1x) {avg_b:+.1f}%  C(DCA) {avg_c:+.1f}%")
    print(f"      Total Edge:  {avg_ea:+.2f}%p/yr  (= Timing {avg_eb:+.2f}%p + Leverage {avg_lev:+.2f}%p)")
    print(f"      Worst Year:  {worst_ea:+.2f}%p  ({worst_lev_r['yr']})")
    print(f"      Best Leverage: {best_lev['yr']} ({best_lev['lev_add']:+.2f}%p)")
    print(f"      Worst Leverage: {worst_lev_r['yr']} ({worst_lev_r['lev_add']:+.2f}%p)")
    if bear_lev:
        print(f"      Bear year avg leverage cost: {np.nanmean(bear_lev):+.2f}%p")
    if bull_lev:
        print(f"      Bull year avg leverage gain: {np.nanmean(bull_lev):+.2f}%p")

    # Win/Loss years for leverage
    lev_win = sum(1 for r in yr if r['lev_add'] > 0.01)
    lev_lose = sum(1 for r in yr if r['lev_add'] < -0.01)
    lev_tie = len(yr) - lev_win - lev_lose
    print(f"      Leverage 승/패/무: {lev_win}W / {lev_lose}L / {lev_tie}T")


# ═══════════════════════════════════════════════════════════
# SECTION 3: FINAL SUMMARY
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*100}")
print(f"  FINAL SUMMARY: 연평균수익률 기준")
print(f"{'='*100}")

print(f"""
  +-----------------------------------------------------------+
  |  14 티커 평균 연평균수익률                                 |
  +-----------------------------------------------------------+
  |  A (VN60 + 2x 레버리지):     {np.nanmean(sum_a):>+7.1f}% / year          |
  |  B (VN60 + 1x, 같은 타이밍): {np.nanmean(sum_b):>+7.1f}% / year          |
  |  C (Pure DCA, 1x):           {np.nanmean(sum_c):>+7.1f}% / year          |
  +-----------------------------------------------------------+
  |  Total Edge  (A vs C):        {np.nanmean(sum_ea):>+6.2f}%p / year       |
  |   = Timing   (B vs C):        {np.nanmean(sum_eb):>+6.2f}%p / year       |
  |   + Leverage  (A vs B):        {np.nanmean(sum_lev):>+6.2f}%p / year       |
  +-----------------------------------------------------------+
""")

print("=" * 100)
print("  Done.")
