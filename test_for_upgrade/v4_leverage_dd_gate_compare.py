"""
BUY_DD_GATE ON vs OFF Comparison — V4 + 2x Leverage
=====================================================
BUY_DD_GATE: 20일 고점 대비 5% 이상 하락했을 때만 매수 허용
- OFF: 현재 백테스트 (V4 신호만으로 매수)
- ON:  V4 신호 + 20일 고점 대비 5%+ 하락 조건 추가

두 경우의 시그널 수, 연평균 수익률, Edge 비교
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
    calc_v4_score, detect_signal_events, build_price_filter,
    smooth_earnings_volume,
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
LEVERAGE = 2.0
EXPENSE_RATIO_DAILY = 0.0095 / 252

# V4 production parameters
V4_W = 20; SIGNAL_TH = 0.15; COOLDOWN = 5
ER_Q = 66; ATR_Q = 55; LOOKBACK = 252
DIVGATE = 3; CONFIRM = 3

# BUY_DD_GATE parameters
BUY_DD_LOOKBACK = 20
BUY_DD_THRESHOLD = 0.05


def download_max(ticker):
    t = yf.Ticker(ticker)
    df = t.history(period='max', auto_adjust=True)
    if df is None or len(df) == 0:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def build_synthetic_2x(close):
    daily_ret = np.diff(close) / close[:-1]
    lev_price = np.zeros(len(close))
    lev_price[0] = close[0]
    for i in range(1, len(close)):
        lev_ret = LEVERAGE * daily_ret[i - 1] - EXPENSE_RATIO_DAILY
        lev_price[i] = lev_price[i - 1] * (1 + lev_ret)
    return lev_price


def get_buy_signal_indices(df, ticker, use_dd_gate=False):
    """V4 buy signal indices. If use_dd_gate=True, also require 20d high -5% drawdown."""
    try:
        df_s = smooth_earnings_volume(df, ticker=ticker)
    except Exception:
        df_s = df.copy()
    score = calc_v4_score(df_s, w=V4_W, divgate_days=DIVGATE)
    events = detect_signal_events(score, th=SIGNAL_TH, cooldown=COOLDOWN)
    pf = build_price_filter(df_s, er_q=ER_Q, atr_q=ATR_Q, lookback=LOOKBACK)

    close = df_s['Close'].values
    n = len(df_s)

    # For DD gate
    rolling_high = df_s['Close'].rolling(BUY_DD_LOOKBACK, min_periods=1).max().values

    buys = set()
    for ev in events:
        if ev['type'] != 'bottom':
            continue
        if not pf(ev['peak_idx']):
            continue

        # BUY_DD_GATE check
        if use_dd_gate:
            pidx = ev['peak_idx']
            rh = rolling_high[pidx]
            dd = (rh - close[pidx]) / rh if rh > 0 else 0
            if dd < BUY_DD_THRESHOLD:
                continue

        dur = ev['end_idx'] - ev['start_idx'] + 1
        ci = ev['start_idx'] + CONFIRM - 1
        if ci <= ev['end_idx'] and dur >= CONFIRM and ci < n:
            buys.add(ci)
    return buys


def simulate(close, close_2x, buy_signals, dates):
    """Run DCA simulation, return yearly results."""
    n = len(close)
    month_periods = pd.Series(dates).dt.to_period('M')
    month_map = {}
    for i, mp in enumerate(month_periods):
        key = str(mp)
        if key not in month_map:
            month_map[key] = {'first': i, 'last': i, 'period': mp}
        else:
            month_map[key]['last'] = i
    sorted_months = sorted(month_map.keys())

    cash_a = 0.0; shares_1x_a = 0.0; shares_2x_a = 0.0; signal_count = 0
    cash_b = 0.0; shares_1x_b = 0.0
    total_dep = 0.0
    yr_data = {}; prev_yr = None

    def pf_a(idx):
        return shares_1x_a * close[idx] + shares_2x_a * close_2x[idx] + cash_a
    def pf_b(idx):
        return shares_1x_b * close[idx] + cash_b

    for mk in sorted_months:
        mm = month_map[mk]
        fi, li = mm['first'], mm['last']
        yr = mm['period'].year

        if yr != prev_yr:
            if prev_yr is not None:
                ref = fi - 1 if fi > 0 else fi
                yr_data[prev_yr]['end_a'] = pf_a(ref)
                yr_data[prev_yr]['end_b'] = pf_b(ref)
            yr_data[yr] = {
                'start_a': pf_a(fi), 'start_b': pf_b(fi),
                'deposits': 0.0, 'end_a': 0, 'end_b': 0, 'sigs': 0,
            }
            prev_yr = yr

        cash_a += MONTHLY_DEPOSIT; cash_b += MONTHLY_DEPOSIT
        total_dep += MONTHLY_DEPOSIT
        yr_data[yr]['deposits'] += MONTHLY_DEPOSIT

        for day_idx in range(fi, li + 1):
            if day_idx in buy_signals and cash_a > 1.0:
                amt = cash_a * SIGNAL_BUY_PCT
                shares_2x_a += amt / close_2x[day_idx]
                cash_a -= amt
                signal_count += 1
                yr_data[yr]['sigs'] += 1

        if cash_a > 1.0:
            shares_1x_a += cash_a / close[li]; cash_a = 0.0
        if cash_b > 1.0:
            shares_1x_b += cash_b / close[li]; cash_b = 0.0

    yr_data[prev_yr]['end_a'] = pf_a(n - 1)
    yr_data[prev_yr]['end_b'] = pf_b(n - 1)

    final_a = pf_a(n - 1); final_b = pf_b(n - 1)

    yr_results = []
    for yr in sorted(yr_data.keys()):
        yd = yr_data[yr]
        da = yd['start_a'] + yd['deposits'] * 0.5
        db = yd['start_b'] + yd['deposits'] * 0.5
        ra = ((yd['end_a'] - yd['start_a'] - yd['deposits']) / da * 100) if da > 10 else 0
        rb = ((yd['end_b'] - yd['start_b'] - yd['deposits']) / db * 100) if db > 10 else 0
        yr_results.append((yr, ra, rb, ra - rb, yd['sigs']))

    avg_a = np.mean([r[1] for r in yr_results])
    avg_b = np.mean([r[2] for r in yr_results])

    return {
        'final_a': final_a, 'final_b': final_b,
        'avg_ann_a': avg_a, 'avg_ann_b': avg_b,
        'avg_diff': avg_a - avg_b,
        'signals': signal_count,
        'total_dep': total_dep,
        'yr_results': yr_results,
        'wins': sum(1 for r in yr_results if r[3] > 0.5),
        'losses': sum(1 for r in yr_results if r[3] < -0.5),
    }


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
print("=" * 120)
print("  BUY_DD_GATE Comparison: V4 + 2x Leverage")
print(f"  BUY_DD_GATE = 20-day high -{BUY_DD_THRESHOLD*100:.0f}% drawdown filter")
print(f"  OFF: V4 signal only  |  ON: V4 signal + drawdown filter")
print("=" * 120)

results = []

for tk, sector in TICKERS.items():
    df = download_max(tk)
    if df is None or len(df) < 300:
        print(f"  {tk}: SKIP")
        continue

    close = df['Close'].values
    dates = df.index
    close_2x = build_synthetic_2x(close)

    # Annual volatility
    daily_rets = np.diff(close) / close[:-1]
    ann_vol = np.std(daily_rets) * np.sqrt(252) * 100

    # OFF: no DD gate
    sigs_off = get_buy_signal_indices(df, tk, use_dd_gate=False)
    res_off = simulate(close, close_2x, sigs_off, dates)

    # ON: with DD gate
    sigs_on = get_buy_signal_indices(df, tk, use_dd_gate=True)
    res_on = simulate(close, close_2x, sigs_on, dates)

    results.append({
        'ticker': tk, 'sector': sector, 'ann_vol': ann_vol,
        'sigs_off': res_off['signals'], 'sigs_on': res_on['signals'],
        'sigs_filtered': res_off['signals'] - res_on['signals'],
        'diff_off': res_off['avg_diff'], 'diff_on': res_on['avg_diff'],
        'avg_a_off': res_off['avg_ann_a'], 'avg_a_on': res_on['avg_ann_a'],
        'avg_b': res_off['avg_ann_b'],
        'final_a_off': res_off['final_a'], 'final_a_on': res_on['final_a'],
        'final_b': res_off['final_b'],
        'wins_off': res_off['wins'], 'losses_off': res_off['losses'],
        'wins_on': res_on['wins'], 'losses_on': res_on['losses'],
        'yr_off': res_off['yr_results'], 'yr_on': res_on['yr_results'],
    })

    sig_reduction = (1 - res_on['signals'] / res_off['signals']) * 100 if res_off['signals'] > 0 else 0
    print(f"  {tk:<6s}  OFF: {res_off['signals']:>3d}sig edge={res_off['avg_diff']:>+.2f}%p  "
          f"ON: {res_on['signals']:>3d}sig edge={res_on['avg_diff']:>+.2f}%p  "
          f"(filtered {res_off['signals']-res_on['signals']:>2d}sig, {sig_reduction:.0f}%)")


# ═══════════════════════════════════════════════════════════
# Comparison Table
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 120}")
print(f"  COMPARISON: DD_GATE OFF vs ON (sorted by edge difference)")
print(f"{'=' * 120}")
print(f"  {'Ticker':<7s} {'Vol':>5s} "
      f"{'--- DD_GATE OFF ---':>30s}  {'--- DD_GATE ON ---':>30s}  "
      f"{'Delta':>8s} {'Better':>7s}")
print(f"  {'':>12s} "
      f"{'Sigs':>5s} {'V4+2x':>9s} {'Edge':>9s} {'W/L':>6s}  "
      f"{'Sigs':>5s} {'V4+2x':>9s} {'Edge':>9s} {'W/L':>6s}  "
      f"{'':>8s} {'':>7s}")
print(f"  {'=' * 118}")

results.sort(key=lambda x: -(x['diff_on'] - x['diff_off']))

for r in results:
    delta = r['diff_on'] - r['diff_off']
    better = "ON" if delta > 0.5 else ("OFF" if delta < -0.5 else "SAME")
    print(f"  {r['ticker']:<7s} {r['ann_vol']:>4.0f}% "
          f"{r['sigs_off']:>5d} {r['avg_a_off']:>+8.2f}% {r['diff_off']:>+8.2f}%p {r['wins_off']:>2d}/{r['losses_off']:<2d}  "
          f"{r['sigs_on']:>5d} {r['avg_a_on']:>+8.2f}% {r['diff_on']:>+8.2f}%p {r['wins_on']:>2d}/{r['losses_on']:<2d}  "
          f"{delta:>+7.2f}%p {better:>7s}")


# ═══════════════════════════════════════════════════════════
# Signal filtering analysis
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 120}")
print(f"  SIGNAL FILTERING: How many signals does DD_GATE remove?")
print(f"{'=' * 120}")
print(f"  {'Ticker':<7s} {'OFF':>5s} {'ON':>5s} {'Filtered':>9s} {'% Removed':>10s}")
print(f"  {'=' * 45}")

for r in sorted(results, key=lambda x: -(x['sigs_off'] - x['sigs_on'])):
    filt = r['sigs_off'] - r['sigs_on']
    pct = filt / r['sigs_off'] * 100 if r['sigs_off'] > 0 else 0
    print(f"  {r['ticker']:<7s} {r['sigs_off']:>5d} {r['sigs_on']:>5d} {filt:>+8d} {pct:>9.0f}%")

total_off = sum(r['sigs_off'] for r in results)
total_on = sum(r['sigs_on'] for r in results)
total_filt = total_off - total_on
pct_filt = total_filt / total_off * 100 if total_off > 0 else 0
print(f"  {'=' * 45}")
print(f"  {'TOTAL':<7s} {total_off:>5d} {total_on:>5d} {total_filt:>+8d} {pct_filt:>9.0f}%")


# ═══════════════════════════════════════════════════════════
# Dollar impact
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 120}")
print(f"  DOLLAR IMPACT: Final Portfolio Value")
print(f"{'=' * 120}")
print(f"  {'Ticker':<7s} {'V4+2x OFF':>14s} {'V4+2x ON':>14s} {'Pure DCA':>14s} {'OFF vs DCA':>12s} {'ON vs DCA':>12s}")
print(f"  {'=' * 80}")

for r in sorted(results, key=lambda x: -x['final_a_off']):
    diff_off = r['final_a_off'] - r['final_b']
    diff_on = r['final_a_on'] - r['final_b']
    print(f"  {r['ticker']:<7s} ${r['final_a_off']:>12,.0f} ${r['final_a_on']:>12,.0f} ${r['final_b']:>12,.0f} "
          f"${diff_off:>+10,.0f} ${diff_on:>+10,.0f}")


# ═══════════════════════════════════════════════════════════
# Grand Summary
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 120}")
print(f"  GRAND SUMMARY")
print(f"{'=' * 120}")

avg_edge_off = np.mean([r['diff_off'] for r in results])
avg_edge_on = np.mean([r['diff_on'] for r in results])
med_edge_off = np.median([r['diff_off'] for r in results])
med_edge_on = np.median([r['diff_on'] for r in results])

n_pos_off = sum(1 for r in results if r['diff_off'] > 0)
n_pos_on = sum(1 for r in results if r['diff_on'] > 0)

total_wins_off = sum(r['wins_off'] for r in results)
total_losses_off = sum(r['losses_off'] for r in results)
total_wins_on = sum(r['wins_on'] for r in results)
total_losses_on = sum(r['losses_on'] for r in results)

n_better_on = sum(1 for r in results if r['diff_on'] > r['diff_off'] + 0.5)
n_better_off = sum(1 for r in results if r['diff_off'] > r['diff_on'] + 0.5)
n_same = len(results) - n_better_on - n_better_off

print(f"""
  {'':>20s} {'DD_GATE OFF':>15s} {'DD_GATE ON':>15s}
  {'':>20s} {'=' * 15:>15s} {'=' * 15:>15s}
  Total signals:      {total_off:>15d} {total_on:>15d}  ({total_filt:+d}, {pct_filt:.0f}% filtered)
  Avg annual edge:    {avg_edge_off:>+14.2f}%p {avg_edge_on:>+14.2f}%p
  Median annual edge: {med_edge_off:>+14.2f}%p {med_edge_on:>+14.2f}%p
  Edge positive:      {n_pos_off:>11d}/{len(results)} {n_pos_on:>11d}/{len(results)}
  Win/Loss:           {total_wins_off}W/{total_losses_off}L{'':<8s} {total_wins_on}W/{total_losses_on}L

  DD_GATE ON better:  {n_better_on}/{len(results)} tickers
  DD_GATE OFF better: {n_better_off}/{len(results)} tickers
  Similar:            {n_same}/{len(results)} tickers

  Conclusion: DD_GATE {'ON' if avg_edge_on > avg_edge_off else 'OFF'} produces better average edge
              ({avg_edge_on - avg_edge_off:+.2f}%p difference)
""")

print("=" * 120)
print("  Done.")
