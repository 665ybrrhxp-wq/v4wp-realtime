"""
V4 Signal + 2x Leverage  vs  Pure DCA
======================================
전략 A (V4+2x):
  - 매월 초 $500 입금
  - V4 매수 신호 시: 가용자금의 50%로 2x 레버리지 ETF 매수 (SSO/QLD)
  - 월 말: 남은 자금으로 일반 ETF 매수 (VOO/QQQ)

전략 B (순수 DCA):
  - 매월 초 $500 입금
  - 월 말: 전부 일반 ETF 매수 (VOO/QQQ)

2x Leverage Pairs:
  VOO (S&P 500) → SSO (ProShares Ultra S&P500 2x)
  QQQ (Nasdaq 100) → QLD (ProShares Ultra QQQ 2x)
"""
import sys, os, warnings
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
PAIRS = [
    {'regular': 'VOO', 'leveraged': 'SSO', 'label': 'S&P 500'},
    {'regular': 'QQQ', 'leveraged': 'QLD', 'label': 'Nasdaq 100'},
]
MONTHLY_DEPOSIT = 500.0
SIGNAL_BUY_PCT = 0.50

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


def get_buy_signal_dates(df, ticker):
    """Return set of dates with confirmed V4 buy signals"""
    try:
        df_s = smooth_earnings_volume(df, ticker=ticker)
    except Exception:
        df_s = df.copy()
    score = calc_v4_score(df_s, w=V4_W, divgate_days=DIVGATE)
    events = detect_signal_events(score, th=SIGNAL_TH, cooldown=COOLDOWN)
    pf = build_price_filter(df_s, er_q=ER_Q, atr_q=ATR_Q, lookback=LOOKBACK)

    n = len(df)
    buy_dates = set()
    for ev in events:
        if ev['type'] != 'bottom':
            continue
        if not pf(ev['peak_idx']):
            continue
        dur = ev['end_idx'] - ev['start_idx'] + 1
        ci = ev['start_idx'] + CONFIRM - 1
        if ci <= ev['end_idx'] and dur >= CONFIRM and ci < n:
            buy_dates.add(df.index[ci])
    return buy_dates


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
print("=" * 105)
print("  V4 Signal + 2x Leverage  vs  Pure Month-end DCA")
print(f"  Monthly: ${MONTHLY_DEPOSIT:.0f} | Signal: {SIGNAL_BUY_PCT:.0%} cash -> 2x ETF | Month-end: rest -> 1x ETF")
print("=" * 105)

for pair in PAIRS:
    tk_reg = pair['regular']
    tk_lev = pair['leveraged']
    label = pair['label']

    print(f"\n{'━' * 105}")
    print(f"  {label}:  {tk_reg} (1x)  +  {tk_lev} (2x leverage)")
    print(f"{'━' * 105}")

    # Download both
    df_reg = download_max(tk_reg)
    df_lev = download_max(tk_lev)

    if df_reg is None or df_lev is None:
        print("  Data unavailable")
        continue

    # Align dates (inner join)
    common_dates = df_reg.index.intersection(df_lev.index)
    common_dates = common_dates.sort_values()

    if len(common_dates) < 200:
        print("  Insufficient overlapping data")
        continue

    # Create aligned price series
    close_reg = df_reg.loc[common_dates, 'Close'].values
    close_lev = df_lev.loc[common_dates, 'Close'].values
    dates = common_dates
    n = len(dates)

    print(f"  Period: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')} ({n} bars)")
    print(f"  {tk_reg} start=${close_reg[0]:.2f} end=${close_reg[-1]:.2f}  "
          f"({(close_reg[-1]/close_reg[0]-1)*100:+.1f}%)")
    print(f"  {tk_lev} start=${close_lev[0]:.2f} end=${close_lev[-1]:.2f}  "
          f"({(close_lev[-1]/close_lev[0]-1)*100:+.1f}%)")

    # V4 signals (computed on regular ETF's full data, then filtered to common dates)
    buy_signal_dates = get_buy_signal_dates(df_reg, tk_reg)
    # Map to common date indices
    date_to_idx = {d: i for i, d in enumerate(dates)}
    signal_indices = set()
    for sd in buy_signal_dates:
        if sd in date_to_idx:
            signal_indices.add(date_to_idx[sd])

    print(f"  V4 buy signals in period: {len(signal_indices)}")

    # Build month map
    month_periods = pd.Series(dates).dt.to_period('M')
    month_map = {}
    for i, mp in enumerate(month_periods):
        key = str(mp)
        if key not in month_map:
            month_map[key] = {'first': i, 'last': i, 'period': mp}
        else:
            month_map[key]['last'] = i
    sorted_month_keys = sorted(month_map.keys())

    # ══════════════════════════════════════════════════════
    # Simulate
    # ══════════════════════════════════════════════════════
    # Strategy A: V4 signal -> 2x leveraged, month-end -> 1x regular
    cash_a = 0.0
    shares_reg_a = 0.0   # regular ETF shares
    shares_lev_a = 0.0   # leveraged ETF shares
    signal_buy_count = 0
    signal_buy_total = 0.0

    # Strategy B: pure month-end DCA (1x regular only)
    cash_b = 0.0
    shares_reg_b = 0.0

    total_deposited = 0.0

    # Year tracking
    year_data = {}
    prev_yr = None

    def portfolio_a(idx):
        return shares_reg_a * close_reg[idx] + shares_lev_a * close_lev[idx] + cash_a

    def portfolio_b(idx):
        return shares_reg_b * close_reg[idx] + cash_b

    for mk in sorted_month_keys:
        mm = month_map[mk]
        fi, li = mm['first'], mm['last']
        yr = mm['period'].year

        # Year transition
        if yr != prev_yr:
            if prev_yr is not None:
                ref_idx = fi - 1 if fi > 0 else fi
                year_data[prev_yr]['end_a'] = portfolio_a(ref_idx)
                year_data[prev_yr]['end_b'] = portfolio_b(ref_idx)
            ref_idx = fi
            year_data[yr] = {
                'start_a': portfolio_a(ref_idx),
                'start_b': portfolio_b(ref_idx),
                'deposits': 0.0, 'end_a': 0, 'end_b': 0,
                'signal_buys': 0,
            }
            prev_yr = yr

        # Monthly deposit
        cash_a += MONTHLY_DEPOSIT
        cash_b += MONTHLY_DEPOSIT
        total_deposited += MONTHLY_DEPOSIT
        year_data[yr]['deposits'] += MONTHLY_DEPOSIT

        # Strategy A: V4 signal -> buy LEVERAGED ETF
        for day_idx in range(fi, li + 1):
            if day_idx in signal_indices and cash_a > 1.0:
                buy_amt = cash_a * SIGNAL_BUY_PCT
                shares_lev_a += buy_amt / close_lev[day_idx]
                cash_a -= buy_amt
                signal_buy_count += 1
                signal_buy_total += buy_amt
                year_data[yr]['signal_buys'] += 1

        # Strategy A: month-end buy REGULAR ETF with remaining
        if cash_a > 1.0:
            shares_reg_a += cash_a / close_reg[li]
            cash_a = 0.0

        # Strategy B: month-end buy regular ETF with all
        if cash_b > 1.0:
            shares_reg_b += cash_b / close_reg[li]
            cash_b = 0.0

    # Close final year
    year_data[prev_yr]['end_a'] = portfolio_a(n - 1)
    year_data[prev_yr]['end_b'] = portfolio_b(n - 1)

    final_a = portfolio_a(n - 1)
    final_b = portfolio_b(n - 1)
    total_ret_a = (final_a / total_deposited - 1) * 100
    total_ret_b = (final_b / total_deposited - 1) * 100

    # ══════════════════════════════════════════════════════
    # Portfolio composition (Strategy A)
    # ══════════════════════════════════════════════════════
    val_reg_a = shares_reg_a * close_reg[-1]
    val_lev_a = shares_lev_a * close_lev[-1]
    pct_lev = val_lev_a / final_a * 100 if final_a > 0 else 0

    # ══════════════════════════════════════════════════════
    # Summary table
    # ══════════════════════════════════════════════════════
    print(f"\n  {'─' * 100}")
    print(f"  {'':34s} {'V4+2x Leverage':>20s} {'Pure DCA (1x)':>20s} {'Diff':>15s}")
    print(f"  {'─' * 100}")
    print(f"  {'Total deposited':<32s}   ${total_deposited:>17,.0f} ${total_deposited:>17,.0f}")
    print(f"  {'Final portfolio':<32s}   ${final_a:>17,.2f} ${final_b:>17,.2f} ${final_a-final_b:>+12,.2f}")
    print(f"  {'Total return':<32s}   {total_ret_a:>17.2f}% {total_ret_b:>17.2f}% {total_ret_a-total_ret_b:>+12.2f}%p")
    print(f"  {'─' * 100}")
    print(f"  {'Portfolio breakdown (A):'}")
    print(f"    {tk_reg} (1x):  {shares_reg_a:>12.4f} shares  ${val_reg_a:>14,.2f}  ({val_reg_a/final_a*100:.1f}%)")
    print(f"    {tk_lev} (2x):  {shares_lev_a:>12.4f} shares  ${val_lev_a:>14,.2f}  ({pct_lev:.1f}%)")
    print(f"    Cash:           {'':>12s}  ${cash_a:>14,.2f}")
    print(f"  {'─' * 100}")
    print(f"  V4 signal buys: {signal_buy_count}x  total ${signal_buy_total:,.2f} ({signal_buy_total/total_deposited*100:.1f}% of deposits)")
    print(f"  Months: {len(sorted_month_keys)}")

    # ══════════════════════════════════════════════════════
    # Yearly breakdown
    # ══════════════════════════════════════════════════════
    print(f"\n  {'─' * 100}")
    print(f"  Year-by-year (Modified Dietz)")
    print(f"  {'─' * 100}")
    print(f"  {'Year':>6s} {'V4+2x':>12s} {'Pure DCA':>12s} {'Diff':>10s} {'Signals':>8s} {'Verdict':>12s}")
    print(f"  {'─' * 72}")

    yr_results = []
    for yr in sorted(year_data.keys()):
        yd = year_data[yr]
        sa, sb = yd['start_a'], yd['start_b']
        ea, eb = yd['end_a'], yd['end_b']
        dep = yd['deposits']
        sig = yd['signal_buys']

        denom_a = sa + dep * 0.5
        denom_b = sb + dep * 0.5
        ret_a = ((ea - sa - dep) / denom_a * 100) if denom_a > 10 else 0
        ret_b = ((eb - sb - dep) / denom_b * 100) if denom_b > 10 else 0
        diff = ret_a - ret_b

        yr_results.append((yr, ret_a, ret_b, diff, sig))

        if diff > 0.5:
            verdict = "V4+2x WIN"
        elif diff < -0.5:
            verdict = "DCA WIN"
        else:
            verdict = "DRAW"
        print(f"  {yr:>6d} {ret_a:>+10.2f}% {ret_b:>+10.2f}% {diff:>+8.2f}%p {sig:>6d}   {verdict}")

    if yr_results:
        avg_a = np.mean([r[1] for r in yr_results])
        avg_b = np.mean([r[2] for r in yr_results])
        avg_diff = avg_a - avg_b
        med_diff = np.median([r[3] for r in yr_results])
        wins_v4 = sum(1 for r in yr_results if r[3] > 0.5)
        wins_dca = sum(1 for r in yr_results if r[3] < -0.5)
        draws = len(yr_results) - wins_v4 - wins_dca

        print(f"  {'─' * 72}")
        print(f"  {'AVG':>6s} {avg_a:>+10.2f}% {avg_b:>+10.2f}% {avg_diff:>+8.2f}%p")
        print(f"  {'MED':>6s} {np.median([r[1] for r in yr_results]):>+10.2f}% "
              f"{np.median([r[2] for r in yr_results]):>+10.2f}% {med_diff:>+8.2f}%p")
        print(f"\n  V4+2x wins: {wins_v4}  |  DCA wins: {wins_dca}  |  Draw: {draws}  (total {len(yr_results)} yrs)")

        # ══════════════════════════════════════════════════════
        # Market condition analysis
        # ══════════════════════════════════════════════════════
        market_labels = {
            2006: 'Pre-crisis', 2007: 'Subprime', 2008: 'Financial Crisis',
            2009: 'Recovery', 2010: 'QE Recovery', 2011: 'EU Crisis',
            2012: 'Stable', 2013: 'Bull', 2014: 'Stable', 2015: 'China Shock',
            2016: 'Trump Rally', 2017: 'Low Vol', 2018: 'Rate Shock',
            2019: 'Recovery', 2020: 'COVID+V', 2021: 'Overheat',
            2022: 'Bear Market', 2023: 'AI Rally', 2024: 'AI Expansion',
            2025: 'Tariff Shock', 2026: 'Current',
        }

        print(f"\n  {'─' * 100}")
        print(f"  Market condition analysis")
        print(f"  {'─' * 100}")

        sorted_by_diff = sorted(yr_results, key=lambda x: -x[3])

        print(f"  [V4+2x BEST years]")
        for yr, ra, rb, diff, sig in sorted_by_diff[:5]:
            lb = market_labels.get(yr, '')
            print(f"    {yr} ({lb}): V4+2x={ra:>+.2f}%  DCA={rb:>+.2f}%  diff={diff:>+.2f}%p  (signals: {sig})")

        print(f"\n  [V4+2x WORST years]")
        for yr, ra, rb, diff, sig in sorted_by_diff[-5:]:
            lb = market_labels.get(yr, '')
            print(f"    {yr} ({lb}): V4+2x={ra:>+.2f}%  DCA={rb:>+.2f}%  diff={diff:>+.2f}%p  (signals: {sig})")

        # Bear vs Bull breakdown
        bear = [2008, 2022, 2018, 2011, 2015, 2000, 2001, 2002]
        bull = [2013, 2017, 2019, 2021, 2023, 2024, 2009]
        crisis = [2008, 2020, 2022]

        for lbl, target_yrs in [("Bear markets", bear), ("Bull markets", bull), ("Crisis years", crisis)]:
            matches = [r for r in yr_results if r[0] in target_yrs]
            if matches:
                avg_d = np.mean([r[3] for r in matches])
                w = sum(1 for r in matches if r[3] > 0.5)
                print(f"\n  {lbl} ({len(matches)} yrs): avg diff = {avg_d:>+.2f}%p  "
                      f"(V4+2x wins {w}/{len(matches)})")

    # ══════════════════════════════════════════════════════
    # Signal detail with leverage performance
    # ══════════════════════════════════════════════════════
    if signal_indices:
        signal_list = sorted(signal_indices)
        print(f"\n  {'─' * 100}")
        print(f"  V4 signal detail: {tk_lev} (2x) vs {tk_reg} (1x) forward returns")
        print(f"  {'─' * 100}")
        print(f"  {'Date':<14s} {tk_reg+' price':>10s} {tk_lev+' price':>10s} "
              f"{'30d 1x':>8s} {'30d 2x':>8s} {'90d 1x':>8s} {'90d 2x':>8s} {'2x Edge 90d':>12s}")
        print(f"  {'─' * 90}")

        edge_30 = []
        edge_90 = []
        for idx in signal_list:
            d = dates[idx].strftime('%Y-%m-%d')
            pr = close_reg[idx]
            pl = close_lev[idx]

            # Forward returns
            i30 = min(idx + 30, n - 1)
            i90 = min(idx + 90, n - 1)
            has_30 = (idx + 30) < n
            has_90 = (idx + 90) < n

            r30_reg = ((close_reg[i30] / pr) - 1) * 100 if has_30 else None
            r30_lev = ((close_lev[i30] / pl) - 1) * 100 if has_30 else None
            r90_reg = ((close_reg[i90] / pr) - 1) * 100 if has_90 else None
            r90_lev = ((close_lev[i90] / pl) - 1) * 100 if has_90 else None

            s30r = f"{r30_reg:>+7.1f}%" if r30_reg is not None else f"{'N/A':>8s}"
            s30l = f"{r30_lev:>+7.1f}%" if r30_lev is not None else f"{'N/A':>8s}"
            s90r = f"{r90_reg:>+7.1f}%" if r90_reg is not None else f"{'N/A':>8s}"
            s90l = f"{r90_lev:>+7.1f}%" if r90_lev is not None else f"{'N/A':>8s}"

            if r90_lev is not None and r90_reg is not None:
                e90 = r90_lev - r90_reg
                s_e90 = f"{e90:>+10.1f}%p"
                edge_90.append(e90)
            else:
                s_e90 = f"{'':>12s}"

            if r30_lev is not None and r30_reg is not None:
                edge_30.append(r30_lev - r30_reg)

            print(f"  {d:<14s} ${pr:>9.2f} ${pl:>9.2f} {s30r} {s30l} {s90r} {s90l} {s_e90}")

        if edge_90:
            print(f"  {'─' * 90}")
            avg_e90 = np.mean(edge_90)
            pos_e90 = sum(1 for e in edge_90 if e > 0)
            print(f"  2x Edge (90d): avg={avg_e90:>+.2f}%p  positive={pos_e90}/{len(edge_90)}")
        if edge_30:
            avg_e30 = np.mean(edge_30)
            pos_e30 = sum(1 for e in edge_30 if e > 0)
            print(f"  2x Edge (30d): avg={avg_e30:>+.2f}%p  positive={pos_e30}/{len(edge_30)}")


# ══════════════════════════════════════════════════════
# Grand Summary
# ══════════════════════════════════════════════════════
print(f"\n{'━' * 105}")
print(f"  CONCLUSION")
print(f"{'━' * 105}")
print(f"""
  Strategy A: V4 signal -> 2x leveraged ETF (50% of cash) + month-end -> 1x regular ETF
  Strategy B: Month-end -> 1x regular ETF only (pure DCA)

  Key question: Does V4's timing ability amplify returns when combined with 2x leverage?
  If V4 catches bottoms, 2x leverage doubles the rebound gain.
  If V4 misses, 2x leverage doubles the loss + suffers volatility decay.
""")
print("=" * 105)
print("  Done.")
