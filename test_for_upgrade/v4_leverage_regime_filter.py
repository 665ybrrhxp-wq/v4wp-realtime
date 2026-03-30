"""
V4 Leverage + Market Regime Filter
====================================
문제: V4+2x는 상승장에서만 +13%p, 하락/횡보장에서 -3%p.
해결: 시장 레짐을 감지해서, 상승장에서만 레버리지 적용.

테스트할 레짐 필터:
  A) ALWAYS:    항상 2x (기존 전략, baseline)
  B) MA200:     종가 > 200일 SMA일 때만 2x, 아니면 1x
  C) MA50/200:  50일 SMA > 200일 SMA (Golden Cross)일 때만 2x
  D) MOM6M:     최근 126거래일 수익률 > 0일 때만 2x
  E) VOLREG:    최근 20일 변동성 < 60일 변동성 중앙값일 때만 2x
  F) COMBO:     MA200 AND MOM6M 둘 다 충족할 때만 2x

각 필터별로: edge, W/L, 상승장 edge, 하락장 edge 비교
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


def download_max(ticker):
    t = yf.Ticker(ticker)
    df = t.history(period='max', auto_adjust=True)
    if df is None or len(df) == 0:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def build_synthetic_lev(close, leverage):
    daily_ret = np.diff(close) / close[:-1]
    lev_price = np.zeros(len(close))
    lev_price[0] = close[0]
    for i in range(1, len(close)):
        lev_ret = leverage * daily_ret[i - 1] - EXPENSE_RATIO_DAILY
        lev_price[i] = lev_price[i - 1] * (1 + lev_ret)
        if lev_price[i] < 0.001:
            lev_price[i] = 0.001
    return lev_price


def get_buy_signal_indices(df, ticker):
    try:
        df_s = smooth_earnings_volume(df, ticker=ticker)
    except Exception:
        df_s = df.copy()
    score = calc_v4_score(df_s, w=V4_W, divgate_days=DIVGATE)
    events = detect_signal_events(score, th=SIGNAL_TH, cooldown=COOLDOWN)
    pf = build_price_filter(df_s, er_q=ER_Q, atr_q=ATR_Q, lookback=LOOKBACK)

    n = len(df_s)
    buys = set()
    for ev in events:
        if ev['type'] != 'bottom':
            continue
        if not pf(ev['peak_idx']):
            continue
        dur = ev['end_idx'] - ev['start_idx'] + 1
        ci = ev['start_idx'] + CONFIRM - 1
        if ci <= ev['end_idx'] and dur >= CONFIRM and ci < n:
            buys.add(ci)
    return buys


# ═══════════════════════════════════════════════════════════
# Regime Filters: return boolean array [True=bull, False=bear]
# ═══════════════════════════════════════════════════════════
def regime_always(close, dates):
    """Always bull (baseline)"""
    return np.ones(len(close), dtype=bool)


def regime_ma200(close, dates):
    """Bull when price > 200-day SMA"""
    sma200 = pd.Series(close).rolling(200, min_periods=200).mean().values
    bull = np.zeros(len(close), dtype=bool)
    for i in range(len(close)):
        if not np.isnan(sma200[i]):
            bull[i] = close[i] > sma200[i]
    return bull


def regime_golden_cross(close, dates):
    """Bull when 50-day SMA > 200-day SMA"""
    sma50 = pd.Series(close).rolling(50, min_periods=50).mean().values
    sma200 = pd.Series(close).rolling(200, min_periods=200).mean().values
    bull = np.zeros(len(close), dtype=bool)
    for i in range(len(close)):
        if not np.isnan(sma50[i]) and not np.isnan(sma200[i]):
            bull[i] = sma50[i] > sma200[i]
    return bull


def regime_momentum_6m(close, dates):
    """Bull when 126-day (6-month) return > 0"""
    bull = np.zeros(len(close), dtype=bool)
    for i in range(126, len(close)):
        bull[i] = close[i] > close[i - 126]
    return bull


def regime_vol_low(close, dates):
    """Bull when recent 20d vol < rolling 60d median vol"""
    rets = np.zeros(len(close))
    rets[1:] = np.diff(close) / close[:-1]
    vol20 = pd.Series(rets).rolling(20, min_periods=20).std().values
    vol60_med = pd.Series(vol20).rolling(60, min_periods=60).median().values
    bull = np.zeros(len(close), dtype=bool)
    for i in range(len(close)):
        if not np.isnan(vol20[i]) and not np.isnan(vol60_med[i]):
            bull[i] = vol20[i] < vol60_med[i]
    return bull


def regime_combo(close, dates):
    """Bull when BOTH MA200 AND Momentum 6M are true"""
    ma = regime_ma200(close, dates)
    mom = regime_momentum_6m(close, dates)
    return ma & mom


def regime_ma200_or_mom(close, dates):
    """Bull when MA200 OR Momentum 6M is true (more permissive)"""
    ma = regime_ma200(close, dates)
    mom = regime_momentum_6m(close, dates)
    return ma | mom


REGIME_FILTERS = {
    'ALWAYS':     regime_always,
    'MA200':      regime_ma200,
    'GoldenX':    regime_golden_cross,
    'MOM6M':      regime_momentum_6m,
    'LOW_VOL':    regime_vol_low,
    'MA+MOM':     regime_combo,
    'MA|MOM':     regime_ma200_or_mom,
}


def simulate_with_regime(close, close_2x, buy_signals, dates, regime_bull):
    """
    V4 signal + regime filter:
    - If regime is bull at signal day: buy 2x leveraged
    - If regime is bear at signal day: buy 1x (normal)
    - Month-end: always buy 1x
    """
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

    cash_a = 0.0; sh_1x_a = 0.0; sh_2x_a = 0.0
    cash_b = 0.0; sh_1x_b = 0.0
    total_dep = 0.0
    sig_2x = 0; sig_1x = 0

    yr_data = {}; prev_yr = None

    def pf_a(idx):
        return sh_1x_a * close[idx] + sh_2x_a * close_2x[idx] + cash_a
    def pf_b(idx):
        return sh_1x_b * close[idx] + cash_b

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
                'deposits': 0.0, 'end_a': 0, 'end_b': 0,
            }
            prev_yr = yr

        cash_a += MONTHLY_DEPOSIT; cash_b += MONTHLY_DEPOSIT
        total_dep += MONTHLY_DEPOSIT
        yr_data[yr]['deposits'] += MONTHLY_DEPOSIT

        for day_idx in range(fi, li + 1):
            if day_idx in buy_signals and cash_a > 1.0:
                amt = cash_a * SIGNAL_BUY_PCT
                if regime_bull[day_idx]:
                    # Bull: buy 2x
                    sh_2x_a += amt / close_2x[day_idx]
                    sig_2x += 1
                else:
                    # Bear: buy 1x instead
                    sh_1x_a += amt / close[day_idx]
                    sig_1x += 1
                cash_a -= amt

        if cash_a > 1.0:
            sh_1x_a += cash_a / close[li]; cash_a = 0.0
        if cash_b > 1.0:
            sh_1x_b += cash_b / close[li]; cash_b = 0.0

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
        yr_results.append({'yr': yr, 'ret_a': ra, 'ret_b': rb, 'diff': ra - rb})

    avg_a = np.mean([r['ret_a'] for r in yr_results])
    avg_b = np.mean([r['ret_b'] for r in yr_results])
    worst_diff = min(r['diff'] for r in yr_results)

    bull_yrs = [r for r in yr_results if r['ret_b'] > 5]
    bear_yrs = [r for r in yr_results if r['ret_b'] < -5]
    flat_yrs = [r for r in yr_results if -5 <= r['ret_b'] <= 5]

    return {
        'edge': avg_a - avg_b,
        'final_a': final_a, 'final_b': final_b,
        'wins': sum(1 for r in yr_results if r['diff'] > 0.5),
        'losses': sum(1 for r in yr_results if r['diff'] < -0.5),
        'worst_diff': worst_diff,
        'sig_2x': sig_2x, 'sig_1x': sig_1x,
        'bull_edge': np.mean([r['diff'] for r in bull_yrs]) if bull_yrs else 0,
        'bear_edge': np.mean([r['diff'] for r in bear_yrs]) if bear_yrs else 0,
        'flat_edge': np.mean([r['diff'] for r in flat_yrs]) if flat_yrs else 0,
        'n_bull': len(bull_yrs), 'n_bear': len(bear_yrs), 'n_flat': len(flat_yrs),
        'yr_results': yr_results,
        'efficiency': (avg_a - avg_b) / abs(worst_diff) if abs(worst_diff) > 0.1 else 0,
    }


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
print("=" * 130)
print("  V4 LEVERAGE + MARKET REGIME FILTER: Solving the Bear Market Problem")
print("  Strategy: V4 signal + bull regime -> 2x leverage | V4 signal + bear regime -> 1x (no leverage)")
print("=" * 130)

# master[ticker][filter_name] = result
master = {}
filter_names = list(REGIME_FILTERS.keys())

for tk, sector in TICKERS.items():
    print(f"  {tk}...", end=" ", flush=True)
    df = download_max(tk)
    if df is None or len(df) < 300:
        print("SKIP")
        continue

    close = df['Close'].values
    dates = df.index
    close_2x = build_synthetic_lev(close, LEVERAGE)
    buy_signals = get_buy_signal_indices(df, tk)

    master[tk] = {}
    edges = []
    for fname, ffunc in REGIME_FILTERS.items():
        regime = ffunc(close, dates)
        res = simulate_with_regime(close, close_2x, buy_signals, dates, regime)
        master[tk][fname] = res
        edges.append(f"{fname}={res['edge']:+.1f}")

    print("  ".join(edges))


# ═══════════════════════════════════════════════════════════
# [1] Overview: Edge by Regime Filter
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 130}")
print(f"  [1] EDGE BY REGIME FILTER (all tickers)")
print(f"{'=' * 130}")

header = f"  {'Ticker':<7s}"
for fn in filter_names:
    header += f" {fn:>10s}"
header += f" {'Best':>10s}"
print(header)
print(f"  {'=' * 128}")

for tk in master:
    line = f"  {tk:<7s}"
    best_fn = ''; best_edge = -999
    for fn in filter_names:
        e = master[tk][fn]['edge']
        line += f" {e:>+9.2f}%p"  # Changed from 10s to fit
        if fn != 'ALWAYS' and e > best_edge:
            best_edge = e; best_fn = fn
    line += f" {best_fn:>10s}"
    print(line)

# Average row
line = f"  {'AVG':<7s}"
for fn in filter_names:
    avg = np.mean([master[tk][fn]['edge'] for tk in master])
    line += f" {avg:>+9.2f}%p"
print(f"  {'=' * 128}")
print(line)


# ═══════════════════════════════════════════════════════════
# [2] Bear Market Edge improvement
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 130}")
print(f"  [2] BEAR MARKET EDGE: Does the filter fix the -3%p bear market problem?")
print(f"{'=' * 130}")

header = f"  {'Ticker':<7s}"
for fn in filter_names:
    header += f" {fn:>9s}"
print(header)
print(f"  {'=' * 80}")

for tk in master:
    line = f"  {tk:<7s}"
    for fn in filter_names:
        be = master[tk][fn]['bear_edge']
        line += f" {be:>+8.2f}%p"
    print(line)

line = f"  {'AVG':<7s}"
for fn in filter_names:
    avg_be = np.mean([master[tk][fn]['bear_edge'] for tk in master])
    line += f" {avg_be:>+8.2f}%p"
print(f"  {'=' * 80}")
print(line)


# ═══════════════════════════════════════════════════════════
# [3] Bull Market Edge (are we sacrificing upside?)
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 130}")
print(f"  [3] BULL MARKET EDGE: How much upside do we sacrifice?")
print(f"{'=' * 130}")

header = f"  {'Ticker':<7s}"
for fn in filter_names:
    header += f" {fn:>9s}"
print(header)
print(f"  {'=' * 80}")

for tk in master:
    line = f"  {tk:<7s}"
    for fn in filter_names:
        bue = master[tk][fn]['bull_edge']
        line += f" {bue:>+8.2f}%p"
    print(line)

line = f"  {'AVG':<7s}"
for fn in filter_names:
    avg_bue = np.mean([master[tk][fn]['bull_edge'] for tk in master])
    line += f" {avg_bue:>+8.2f}%p"
print(f"  {'=' * 80}")
print(line)


# ═══════════════════════════════════════════════════════════
# [4] Signal routing: how many signals go to 2x vs 1x?
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 130}")
print(f"  [4] SIGNAL ROUTING: What % of signals get 2x leverage?")
print(f"{'=' * 130}")

header = f"  {'Ticker':<7s} {'Total':>5s}"
for fn in filter_names[1:]:  # skip ALWAYS
    header += f" {fn:>10s}"
print(header)
print(f"  {'=' * 85}")

for tk in master:
    total = master[tk]['ALWAYS']['sig_2x']
    line = f"  {tk:<7s} {total:>5d}"
    for fn in filter_names[1:]:
        s2x = master[tk][fn]['sig_2x']
        pct = s2x / total * 100 if total > 0 else 0
        line += f" {pct:>8.0f}%  "
    print(line)

# Average
line = f"  {'AVG':<7s} {'':>5s}"
for fn in filter_names[1:]:
    pcts = []
    for tk in master:
        total = master[tk]['ALWAYS']['sig_2x']
        s2x = master[tk][fn]['sig_2x']
        if total > 0:
            pcts.append(s2x / total * 100)
    line += f" {np.mean(pcts):>8.0f}%  "
print(f"  {'=' * 85}")
print(line)


# ═══════════════════════════════════════════════════════════
# [5] Risk: Worst Year Loss
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 130}")
print(f"  [5] RISK: Worst Year Extra Loss")
print(f"{'=' * 130}")

header = f"  {'Ticker':<7s}"
for fn in filter_names:
    header += f" {fn:>9s}"
print(header)
print(f"  {'=' * 80}")

for tk in master:
    line = f"  {tk:<7s}"
    for fn in filter_names:
        w = master[tk][fn]['worst_diff']
        line += f" {w:>+8.2f}%p"
    print(line)

line = f"  {'AVG':<7s}"
for fn in filter_names:
    avg_w = np.mean([master[tk][fn]['worst_diff'] for tk in master])
    line += f" {avg_w:>+8.2f}%p"
print(f"  {'=' * 80}")
print(line)


# ═══════════════════════════════════════════════════════════
# [6] Efficiency: Edge / |Worst Loss|
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 130}")
print(f"  [6] EFFICIENCY: Edge / |Worst Year Loss| (higher = better)")
print(f"{'=' * 130}")

header = f"  {'Ticker':<7s}"
for fn in filter_names:
    header += f" {fn:>9s}"
print(header)
print(f"  {'=' * 80}")

for tk in master:
    line = f"  {tk:<7s}"
    for fn in filter_names:
        eff = master[tk][fn]['efficiency']
        line += f" {eff:>+8.3f} "
    print(line)

line = f"  {'AVG':<7s}"
for fn in filter_names:
    avg_eff = np.mean([master[tk][fn]['efficiency'] for tk in master])
    line += f" {avg_eff:>+8.3f} "
print(f"  {'=' * 80}")
print(line)


# ═══════════════════════════════════════════════════════════
# GRAND SUMMARY
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 130}")
print(f"  GRAND SUMMARY: REGIME FILTER COMPARISON")
print(f"{'=' * 130}")

print(f"\n  {'Filter':<10s} {'Avg Edge':>10s} {'Bear Edge':>10s} {'Bull Edge':>10s} "
      f"{'Worst Loss':>11s} {'Efficiency':>11s} {'2x Signal%':>11s}")
print(f"  {'=' * 75}")

summary_rows = []
for fn in filter_names:
    avg_edge = np.mean([master[tk][fn]['edge'] for tk in master])
    avg_bear = np.mean([master[tk][fn]['bear_edge'] for tk in master])
    avg_bull = np.mean([master[tk][fn]['bull_edge'] for tk in master])
    avg_worst = np.mean([master[tk][fn]['worst_diff'] for tk in master])
    avg_eff = np.mean([master[tk][fn]['efficiency'] for tk in master])

    if fn == 'ALWAYS':
        sig_pct = 100.0
    else:
        pcts = []
        for tk in master:
            total = master[tk]['ALWAYS']['sig_2x']
            s2x = master[tk][fn]['sig_2x']
            if total > 0:
                pcts.append(s2x / total * 100)
        sig_pct = np.mean(pcts)

    summary_rows.append({
        'fn': fn, 'edge': avg_edge, 'bear': avg_bear, 'bull': avg_bull,
        'worst': avg_worst, 'eff': avg_eff, 'sig_pct': sig_pct,
    })

    print(f"  {fn:<10s} {avg_edge:>+9.2f}%p {avg_bear:>+9.2f}%p {avg_bull:>+9.2f}%p "
          f"{avg_worst:>+10.2f}%p {avg_eff:>+10.3f}  {sig_pct:>9.0f}%")

# Best filter analysis
print(f"\n  Key Comparisons vs ALWAYS (baseline):")
baseline = summary_rows[0]
for sr in summary_rows[1:]:
    edge_delta = sr['edge'] - baseline['edge']
    bear_delta = sr['bear'] - baseline['bear']
    worst_delta = sr['worst'] - baseline['worst']
    eff_delta = sr['eff'] - baseline['eff']
    print(f"    {sr['fn']:<10s}: edge {edge_delta:>+.2f}%p | bear {bear_delta:>+.2f}%p | "
          f"worst loss {worst_delta:>+.2f}%p | efficiency {eff_delta:>+.3f}")

# Winner
best_eff = max(summary_rows, key=lambda x: x['eff'])
best_bear = max(summary_rows, key=lambda x: x['bear'])
best_edge = max(summary_rows, key=lambda x: x['edge'])

print(f"\n  Best overall edge:       {best_edge['fn']} ({best_edge['edge']:+.2f}%p)")
print(f"  Best bear market fix:    {best_bear['fn']} ({best_bear['bear']:+.2f}%p)")
print(f"  Best risk efficiency:    {best_eff['fn']} ({best_eff['eff']:+.3f})")

# Per-ticker best filter
print(f"\n  Per-ticker best efficiency filter:")
from collections import Counter
best_per_ticker = {}
for tk in master:
    best_fn = max(filter_names, key=lambda fn: master[tk][fn]['efficiency'])
    best_per_ticker[tk] = best_fn
    print(f"    {tk:<7s}: {best_fn}")

dist = Counter(best_per_ticker.values())
print(f"\n  Distribution:")
for fn, cnt in sorted(dist.items(), key=lambda x: -x[1]):
    print(f"    {fn:<10s}: {cnt}/{len(master)} tickers")

print()
print("=" * 130)
print("  Done.")
