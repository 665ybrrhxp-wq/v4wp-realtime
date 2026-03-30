"""
V4 Signal + 3x Leverage vs Pure DCA — Individual Stocks
========================================================
개별 종목에 V4 신호 시 3배 레버리지 효과를 시뮬레이션.

전략 A (V4+3x):
  - 매월 초 $500 입금
  - V4 매수 신호 시: 가용자금의 50%로 합성 3x 레버리지 매수
  - 월 말: 남은 자금으로 일반(1x) 매수

전략 B (순수 DCA):
  - 매월 초 $500 입금
  - 월 말: 전부 일반(1x) 매수

합성 3x 레버리지 계산:
  매일: leveraged_price[t] = leveraged_price[t-1] * (1 + 3 * daily_return[t])
  3x ETF 기준 expense ratio 0.95%/yr 적용.
  변동성 감쇠가 2x보다 더 심하므로 리스크도 증가.
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
LEVERAGE = 3.0
EXPENSE_RATIO_DAILY = 0.0095 / 252  # ~0.95% annual expense for leveraged ETFs

# V4 production parameters
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


def build_synthetic_3x(close):
    """Build synthetic 3x daily-rebalanced leveraged price series.
    Includes volatility decay and expense ratio, mimicking real 3x leveraged ETFs (TQQQ, UPRO, etc.)."""
    daily_ret = np.diff(close) / close[:-1]
    lev_price = np.zeros(len(close))
    lev_price[0] = close[0]
    for i in range(1, len(close)):
        lev_ret = LEVERAGE * daily_ret[i - 1] - EXPENSE_RATIO_DAILY
        lev_price[i] = lev_price[i - 1] * (1 + lev_ret)
        # 3x leverage can go to zero if daily loss > 33.3%
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

    n = len(df)
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
# Main
# ═══════════════════════════════════════════════════════════
print("=" * 115)
print("  V4 Signal + Synthetic 3x Leverage vs Pure DCA — Individual Stocks")
print(f"  Monthly: ${MONTHLY_DEPOSIT:.0f} | Signal: {SIGNAL_BUY_PCT:.0%} cash -> 3x | Month-end: rest -> 1x")
print(f"  Synthetic 3x: daily rebalanced, expense ratio {EXPENSE_RATIO_DAILY*252*100:.2f}%/yr")
print("=" * 115)

# Collect results for summary table
all_results = []

for tk, sector in TICKERS.items():
    df = download_max(tk)
    if df is None or len(df) < 300:
        print(f"  {tk}: SKIP (insufficient data)")
        continue

    close = df['Close'].values
    dates = df.index
    n = len(close)

    # Build synthetic 3x price
    close_3x = build_synthetic_3x(close)

    # V4 signals
    buy_signals = get_buy_signal_indices(df, tk)

    # Annual volatility
    daily_rets = np.diff(close) / close[:-1]
    ann_vol = np.std(daily_rets) * np.sqrt(252) * 100

    # Month map
    month_periods = pd.Series(dates).dt.to_period('M')
    month_map = {}
    for i, mp in enumerate(month_periods):
        key = str(mp)
        if key not in month_map:
            month_map[key] = {'first': i, 'last': i, 'period': mp}
        else:
            month_map[key]['last'] = i
    sorted_months = sorted(month_map.keys())

    # ── Simulate ──
    # A: V4 signal -> 3x, month-end -> 1x
    cash_a = 0.0
    shares_1x_a = 0.0
    shares_3x_a = 0.0
    signal_count = 0

    # B: pure DCA (1x only)
    cash_b = 0.0
    shares_1x_b = 0.0

    total_dep = 0.0

    # Year tracking
    yr_data = {}
    prev_yr = None

    def pf_a(idx):
        return shares_1x_a * close[idx] + shares_3x_a * close_3x[idx] + cash_a
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

        cash_a += MONTHLY_DEPOSIT
        cash_b += MONTHLY_DEPOSIT
        total_dep += MONTHLY_DEPOSIT
        yr_data[yr]['deposits'] += MONTHLY_DEPOSIT

        # A: V4 signals -> buy synthetic 3x
        for day_idx in range(fi, li + 1):
            if day_idx in buy_signals and cash_a > 1.0:
                amt = cash_a * SIGNAL_BUY_PCT
                shares_3x_a += amt / close_3x[day_idx]
                cash_a -= amt
                signal_count += 1
                yr_data[yr]['sigs'] += 1

        # A: month-end -> buy 1x
        if cash_a > 1.0:
            shares_1x_a += cash_a / close[li]
            cash_a = 0.0

        # B: month-end -> buy 1x
        if cash_b > 1.0:
            shares_1x_b += cash_b / close[li]
            cash_b = 0.0

    yr_data[prev_yr]['end_a'] = pf_a(n - 1)
    yr_data[prev_yr]['end_b'] = pf_b(n - 1)

    final_a = pf_a(n - 1)
    final_b = pf_b(n - 1)
    ret_a = (final_a / total_dep - 1) * 100
    ret_b = (final_b / total_dep - 1) * 100

    val_1x = shares_1x_a * close[-1]
    val_3x = shares_3x_a * close_3x[-1]
    pct_3x = val_3x / final_a * 100 if final_a > 0 else 0

    # Yearly Modified Dietz
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
    avg_diff = avg_a - avg_b
    wins = sum(1 for r in yr_results if r[3] > 0.5)
    losses = sum(1 for r in yr_results if r[3] < -0.5)
    draws = len(yr_results) - wins - losses

    # Period
    start_yr = dates[0].year
    end_yr = dates[-1].year
    n_years = len(yr_results)

    # 1x price change
    price_change_1x = (close[-1] / close[0] - 1) * 100
    price_change_3x = (close_3x[-1] / close_3x[0] - 1) * 100

    all_results.append({
        'ticker': tk, 'sector': sector,
        'period': f"{start_yr}-{end_yr}", 'n_years': n_years,
        'ann_vol': ann_vol,
        'price_1x': price_change_1x, 'price_3x': price_change_3x,
        'signals': signal_count,
        'final_a': final_a, 'final_b': final_b,
        'ret_a': ret_a, 'ret_b': ret_b,
        'avg_ann_a': avg_a, 'avg_ann_b': avg_b,
        'avg_diff': avg_diff,
        'pct_3x': pct_3x,
        'wins': wins, 'losses': losses, 'draws': draws,
        'total_dep': total_dep,
        'yr_results': yr_results,
    })

    print(f"  {tk:<6s} ({sector:<8s}) {start_yr}-{end_yr}  "
          f"vol={ann_vol:.0f}%  sigs={signal_count:>3d}  "
          f"V4+3x={avg_a:>+.1f}%  DCA={avg_b:>+.1f}%  "
          f"diff={avg_diff:>+.2f}%p  W/L={wins}/{losses}")

# ═══════════════════════════════════════════════════════════
# Summary Tables
# ═══════════════════════════════════════════════════════════
all_results.sort(key=lambda x: -x['avg_diff'])

print(f"\n{'=' * 115}")
print(f"  RANKING: V4+3x Edge by Ticker (sorted by annual avg diff)")
print(f"{'=' * 115}")
print(f"  {'Rank':>4s} {'Ticker':<7s} {'Sector':<9s} {'Period':>10s} {'Vol':>5s} "
      f"{'Sigs':>5s} {'V4+3x Avg':>10s} {'DCA Avg':>10s} {'Diff':>9s} "
      f"{'Final V4+3x':>13s} {'Final DCA':>13s} {'$Diff':>12s} {'W/L/D':>7s}")
print(f"  {'=' * 113}")

for i, r in enumerate(all_results, 1):
    fdiff = r['final_a'] - r['final_b']
    print(f"  {i:>4d} {r['ticker']:<7s} {r['sector']:<9s} {r['period']:>10s} {r['ann_vol']:>4.0f}% "
          f"{r['signals']:>5d} {r['avg_ann_a']:>+9.2f}% {r['avg_ann_b']:>+9.2f}% {r['avg_diff']:>+8.2f}%p "
          f"${r['final_a']:>11,.0f} ${r['final_b']:>11,.0f} ${fdiff:>+10,.0f} "
          f"{r['wins']}/{r['losses']}/{r['draws']}")

# ═══════════════════════════════════════════════════════════
# Volatility vs Edge correlation
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 115}")
print(f"  ANALYSIS: Volatility vs V4+3x Edge")
print(f"{'=' * 115}")

vols = np.array([r['ann_vol'] for r in all_results])
edges = np.array([r['avg_diff'] for r in all_results])

corr = 0.0
if len(vols) > 2:
    corr = np.corrcoef(vols, edges)[0, 1]
    print(f"\n  Correlation (volatility vs edge): {corr:.3f}")

# Group by volatility
low_vol = [r for r in all_results if r['ann_vol'] < 30]
mid_vol = [r for r in all_results if 30 <= r['ann_vol'] < 50]
high_vol = [r for r in all_results if r['ann_vol'] >= 50]

for label, group in [("Low vol (<30%)", low_vol), ("Mid vol (30-50%)", mid_vol), ("High vol (50%+)", high_vol)]:
    if group:
        avg_v = np.mean([r['ann_vol'] for r in group])
        avg_e = np.mean([r['avg_diff'] for r in group])
        tickers = ', '.join(r['ticker'] for r in group)
        total_wins = sum(r['wins'] for r in group)
        total_losses = sum(r['losses'] for r in group)
        print(f"\n  {label}:  avg vol={avg_v:.0f}%  avg edge={avg_e:>+.2f}%p  W/L={total_wins}/{total_losses}")
        print(f"    Tickers: {tickers}")

# ═══════════════════════════════════════════════════════════
# Sector analysis
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 115}")
print(f"  ANALYSIS: By Sector")
print(f"{'=' * 115}")

sectors = {}
for r in all_results:
    s = r['sector']
    if s not in sectors:
        sectors[s] = []
    sectors[s].append(r)

for s, group in sorted(sectors.items(), key=lambda x: -np.mean([r['avg_diff'] for r in x[1]])):
    avg_e = np.mean([r['avg_diff'] for r in group])
    avg_v = np.mean([r['ann_vol'] for r in group])
    tickers = ', '.join(r['ticker'] for r in group)
    total_fdiff = sum(r['final_a'] - r['final_b'] for r in group)
    print(f"  {s:<10s}: edge={avg_e:>+.2f}%p  vol={avg_v:.0f}%  "
          f"total $diff=${total_fdiff:>+,.0f}  [{tickers}]")

# ═══════════════════════════════════════════════════════════
# Year-by-year detail for top 5
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 115}")
print(f"  TOP 5 Year-by-year Detail")
print(f"{'=' * 115}")

for r in all_results[:5]:
    tk = r['ticker']
    print(f"\n  {tk} ({r['sector']}, vol={r['ann_vol']:.0f}%)")
    print(f"  {'Year':>6s} {'V4+3x':>10s} {'DCA':>10s} {'Diff':>9s} {'Sigs':>5s}")
    print(f"  {'=' * 48}")
    for yr, ra, rb, diff, sig in r['yr_results']:
        v = "WIN" if diff > 0.5 else ("LOSE" if diff < -0.5 else "")
        print(f"  {yr:>6d} {ra:>+9.2f}% {rb:>+9.2f}% {diff:>+8.2f}%p {sig:>4d}  {v}")
    print(f"  {'=' * 48}")
    print(f"  {'AVG':>6s} {r['avg_ann_a']:>+9.2f}% {r['avg_ann_b']:>+9.2f}% {r['avg_diff']:>+8.2f}%p")


# ═══════════════════════════════════════════════════════════
# Risk analysis: worst years
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 115}")
print(f"  RISK: Worst Year Drawdown (V4+3x vs DCA)")
print(f"{'=' * 115}")
print(f"  {'Ticker':<7s} {'Worst Year':>10s} {'V4+3x':>10s} {'DCA':>10s} {'Extra Loss':>12s}")
print(f"  {'=' * 55}")

for r in all_results:
    worst = min(r['yr_results'], key=lambda x: x[3])
    yr, ra, rb, diff, sig = worst
    print(f"  {r['ticker']:<7s} {yr:>10d} {ra:>+9.2f}% {rb:>+9.2f}% {diff:>+10.2f}%p")


# ═══════════════════════════════════════════════════════════
# 2x vs 3x comparison reference
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 115}")
print(f"  GRAND SUMMARY")
print(f"{'=' * 115}")

total_final_a = sum(r['final_a'] for r in all_results)
total_final_b = sum(r['final_b'] for r in all_results)
total_dep_all = sum(r['total_dep'] for r in all_results)
grand_diff = total_final_a - total_final_b

avg_edge_all = np.mean([r['avg_diff'] for r in all_results])
med_edge_all = np.median([r['avg_diff'] for r in all_results])
total_wins = sum(r['wins'] for r in all_results)
total_losses = sum(r['losses'] for r in all_results)

n_positive = sum(1 for r in all_results if r['avg_diff'] > 0)

print(f"""
  Total tickers tested: {len(all_results)}
  Edge positive tickers: {n_positive}/{len(all_results)} ({n_positive/len(all_results)*100:.0f}%)

  Avg annual edge:    {avg_edge_all:>+.2f}%p
  Median annual edge: {med_edge_all:>+.2f}%p

  Total final portfolio (all tickers):
    V4+3x:    ${total_final_a:>15,.0f}
    Pure DCA:  ${total_final_b:>15,.0f}
    Difference: ${grand_diff:>+14,.0f} ({grand_diff/total_final_b*100:+.1f}%)

  Total invested: ${total_dep_all:>,.0f}
  Win/Loss (all ticker-years): {total_wins}W / {total_losses}L

  Correlation (volatility vs edge): {corr:.3f}
""")

# ═══════════════════════════════════════════════════════════
# Portfolio Allocation: 3x vs 1x weight
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 115}")
print(f"  PORTFOLIO ALLOCATION: 3x Leveraged Position Weight in Final Portfolio")
print(f"{'=' * 115}")
print(f"  {'Ticker':<7s} {'Sector':<9s} {'Signals':>7s} {'3x Weight':>10s} {'1x Weight':>10s} {'Final Value':>13s}")
print(f"  {'=' * 65}")

pct_3x_list = []
for r in sorted(all_results, key=lambda x: -x['pct_3x']):
    pct_1x = 100 - r['pct_3x']
    pct_3x_list.append(r['pct_3x'])
    print(f"  {r['ticker']:<7s} {r['sector']:<9s} {r['signals']:>7d} {r['pct_3x']:>9.1f}% {pct_1x:>9.1f}% ${r['final_a']:>11,.0f}")

avg_3x_pct = np.mean(pct_3x_list)
med_3x_pct = np.median(pct_3x_list)
print(f"  {'=' * 65}")
print(f"  Average 3x weight:  {avg_3x_pct:.1f}%")
print(f"  Median 3x weight:   {med_3x_pct:.1f}%")

print("\n" + "=" * 115)
print("  Done.")
