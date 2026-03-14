"""
V4 Indicator Improvements — Comprehensive Backtest
====================================================
Tests 5 improvements individually and in combinations across multiple market periods.

Improvements:
  F1: PRICE_FILTER_RELAX  — ATR quantile 66 -> 55
  F2: ACT_MUL_CAP         — Activity multiplier cap 2.2 -> 1.8
  F3: DIV_REBALANCE       — Divergence 40%->30%, Force 30%->35%, Concordance 30%->35%
  F4: LATE_SELL_BLOCK     — Suppress sell when price already >5% below 20d high
  F5: BUY_TH_RAISE        — Buy threshold th*0.5 -> th*0.7
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import json
import os
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
import time

# ============================================================
# PATHS
# ============================================================
_project_root = Path(__file__).resolve().parents[2]  # ...\거래량 백테스트
_config_path = _project_root / 'v4wp_realtime' / 'config' / 'watchlist.json'
_cache_dir = str(_project_root / 'cache')

sys.path.insert(0, str(_project_root))

# Import base functions from real_market_backtest
from real_market_backtest import (
    download_data,
    calc_pv_divergence,
    calc_pv_concordance,
    calc_pv_force_macd,
    calc_efficiency_ratio,
    calc_atr_percentile,
)

# ============================================================
# MARKET PERIODS
# ============================================================
MARKET_PERIODS = [
    {'name': '1.Lehman',      'start': '2007-01-01', 'end': '2009-12-31'},
    {'name': '2.Recovery',    'start': '2009-03-01', 'end': '2015-12-31'},
    {'name': '3.RateShock',   'start': '2018-01-01', 'end': '2019-12-31'},
    {'name': '4.Covid',       'start': '2020-01-01', 'end': '2021-12-31'},
    {'name': '5.InflDown',    'start': '2022-01-01', 'end': '2023-06-30'},
    {'name': '6.AI+Tariff',   'start': '2023-07-01', 'end': '2026-03-31'},
    {'name': '7.Full',        'start': '2007-01-01', 'end': '2026-03-31'},
]

# ============================================================
# TRADING PARAMETERS
# ============================================================
INITIAL_CASH = 1000.0
MONTHLY_ADD = 100.0
STRONG_BUY_TH = 0.15
STRONG_SELL_TH = -0.25
BUY_STRONG_PCT = 0.50
BUY_NORMAL_PCT = 0.30
SELL_STRONG_PCT = 0.10
SELL_NORMAL_PCT = 0.05

# ============================================================
# IMPROVEMENT FLAGS
# ============================================================
IMPROVEMENTS = {
    'PRICE_FILTER_RELAX': False,   # F1
    'ACT_MUL_CAP': False,          # F2
    'DIV_REBALANCE': False,        # F3
    'LATE_SELL_BLOCK': False,      # F4
    'BUY_TH_RAISE': False,        # F5
}

# ============================================================
# CONFIGURATIONS TO TEST
# ============================================================
CONFIGS = {
    'BASE': {},
    'F1':   {'PRICE_FILTER_RELAX': True},
    'F2':   {'ACT_MUL_CAP': True},
    'F3':   {'DIV_REBALANCE': True},
    'F4':   {'LATE_SELL_BLOCK': True},
    'F5':   {'BUY_TH_RAISE': True},
    'C12':  {'PRICE_FILTER_RELAX': True, 'ACT_MUL_CAP': True},
    'C13':  {'PRICE_FILTER_RELAX': True, 'DIV_REBALANCE': True},
    'C14':  {'PRICE_FILTER_RELAX': True, 'LATE_SELL_BLOCK': True},
    'C15':  {'PRICE_FILTER_RELAX': True, 'BUY_TH_RAISE': True},
    'C23':  {'ACT_MUL_CAP': True, 'DIV_REBALANCE': True},
    'C34':  {'DIV_REBALANCE': True, 'LATE_SELL_BLOCK': True},
    'C45':  {'LATE_SELL_BLOCK': True, 'BUY_TH_RAISE': True},
    'C123': {'PRICE_FILTER_RELAX': True, 'ACT_MUL_CAP': True, 'DIV_REBALANCE': True},
    'C134': {'PRICE_FILTER_RELAX': True, 'DIV_REBALANCE': True, 'LATE_SELL_BLOCK': True},
    'C1345':{'PRICE_FILTER_RELAX': True, 'DIV_REBALANCE': True, 'LATE_SELL_BLOCK': True, 'BUY_TH_RAISE': True},
    'CALL': {'PRICE_FILTER_RELAX': True, 'ACT_MUL_CAP': True, 'DIV_REBALANCE': True,
             'LATE_SELL_BLOCK': True, 'BUY_TH_RAISE': True},
}


# ============================================================
# MODIFIED CORE FUNCTIONS
# ============================================================

def calc_v4_score_mod(df, w=20, flags=None):
    """
    Modified V4 score calculation.
    Supports:
      - ACT_MUL_CAP: cap 3-active multiplier at 1.8 instead of 2.2
      - DIV_REBALANCE: weights Force=0.35, Div=0.30, Conc=0.35 instead of 0.30/0.40/0.30
    """
    if flags is None:
        flags = {}

    # Weights
    if flags.get('DIV_REBALANCE', False):
        w_force, w_div, w_conc = 0.35, 0.30, 0.35
    else:
        w_force, w_div, w_conc = 0.30, 0.40, 0.30

    # Activity multiplier map
    if flags.get('ACT_MUL_CAP', False):
        mm = {0: 0.5, 1: 1.0, 2: 1.5, 3: 1.8}
    else:
        mm = {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.2}

    n = len(df)
    pv_div = calc_pv_divergence(df, w)
    pv_conc = calc_pv_concordance(df, w)
    pv_fh = calc_pv_force_macd(df)
    scores = np.zeros(n)

    for i in range(max(60, w), n):
        s_div = np.clip(pv_div.iloc[i] / 3, -1, 1)
        s_conc = pv_conc.iloc[i]
        fhr = abs(pv_fh.iloc[max(0, i - w):i]).mean() + 1e-10
        s_force = np.clip(pv_fh.iloc[i] / fhr, -1, 1)

        dire = w_force * s_force + w_div * s_div + w_conc * s_conc
        act = sum([abs(s_div) > 0.1, abs(s_conc) > 0.1, abs(s_force) > 0.1])
        scores[i] = dire * mm.get(act, 1.0)

    return pd.Series(scores, index=df.index, name='V4_mod')


def build_price_filter_mod(df, er_q=66, atr_q=66, lookback=252):
    """
    Modified price filter.
    F1 (PRICE_FILTER_RELAX): pass atr_q=55 instead of 66
    """
    er = calc_efficiency_ratio(df, 20).values
    atr = calc_atr_percentile(df, 14, 252).values
    n = len(df)

    er_thresh = np.full(n, np.nan)
    atr_thresh = np.full(n, np.nan)
    for i in range(lookback, n):
        lb = max(0, i - lookback)
        er_window = er[lb:i]
        atr_window = atr[lb:i]
        if len(er_window) >= 60:
            er_thresh[i] = np.percentile(er_window, er_q)
            atr_thresh[i] = np.percentile(atr_window, atr_q)

    def price_filter(peak_idx):
        if peak_idx >= n or np.isnan(er_thresh[peak_idx]) or np.isnan(atr_thresh[peak_idx]):
            return False
        return er[peak_idx] < er_thresh[peak_idx] and atr[peak_idx] > atr_thresh[peak_idx]

    return price_filter


def detect_signal_events_mod(score, th=0.15, cooldown=5, flags=None):
    """
    Modified signal detection.
    F5 (BUY_TH_RAISE): bottom threshold th*0.7 instead of th*0.5
    """
    if flags is None:
        flags = {}

    arr = score.values
    n = len(arr)
    events = []

    # --- Top signals (score < -th) ---
    i = 0
    while i < n:
        if not np.isnan(arr[i]) and arr[i] < -th:
            start = i
            peak_val = arr[i]
            peak_idx = i
            end = i
            while i < n:
                if not np.isnan(arr[i]) and arr[i] < -th:
                    if arr[i] < peak_val:
                        peak_val = arr[i]
                        peak_idx = i
                    end = i
                    i += 1
                else:
                    gap = 0
                    while gap < cooldown and i + gap < n:
                        if not np.isnan(arr[i + gap]) and arr[i + gap] < -th:
                            break
                        gap += 1
                    if gap < cooldown and i + gap < n:
                        i += gap
                    else:
                        break
            events.append({
                'type': 'top', 'start_idx': start, 'end_idx': end,
                'peak_idx': peak_idx, 'peak_val': peak_val
            })
        else:
            i += 1

    # --- Bottom signals ---
    if flags.get('BUY_TH_RAISE', False):
        bot_th = th * 0.7   # raised from 0.5
    else:
        bot_th = th * 0.5   # original

    i = 0
    while i < n:
        if not np.isnan(arr[i]) and arr[i] > bot_th:
            start = i
            peak_val = arr[i]
            peak_idx = i
            end = i
            while i < n:
                if not np.isnan(arr[i]) and arr[i] > bot_th:
                    if arr[i] > peak_val:
                        peak_val = arr[i]
                        peak_idx = i
                    end = i
                    i += 1
                else:
                    gap = 0
                    while gap < cooldown and i + gap < n:
                        if not np.isnan(arr[i + gap]) and arr[i + gap] > bot_th:
                            break
                        gap += 1
                    if gap < cooldown and i + gap < n:
                        i += gap
                    else:
                        break
            events.append({
                'type': 'bottom', 'start_idx': start, 'end_idx': end,
                'peak_idx': peak_idx, 'peak_val': peak_val
            })
        else:
            i += 1

    return sorted(events, key=lambda e: e['start_idx'])


def apply_late_sell_block(events, df):
    """
    F4 (LATE_SELL_BLOCK): Suppress sell signals (type='top') where price
    at signal time is already >5% below the 20-day rolling high.
    """
    close = df['Close'].values
    rolling_high_20 = df['Close'].rolling(20, min_periods=1).max().values

    filtered = []
    for ev in events:
        if ev['type'] == 'top':
            pidx = ev['peak_idx']
            if pidx < len(close):
                price_at_signal = close[pidx]
                rh = rolling_high_20[pidx]
                drop_pct = (rh - price_at_signal) / rh if rh > 0 else 0
                if drop_pct > 0.05:
                    # Price already dropped >5% from 20d high -> suppress
                    continue
        filtered.append(ev)
    return filtered


# ============================================================
# PIPELINE: score -> events -> trade simulation
# ============================================================

def run_pipeline(df, flags, th=0.15, cooldown=5):
    """
    Run the full pipeline for a given ticker/period DataFrame with specified flags.
    Returns: (events, score_series)
    """
    # Step 1: Calculate V4 score (potentially modified)
    score = calc_v4_score_mod(df, w=20, flags=flags)

    # Step 2: Detect signal events (potentially modified threshold)
    events = detect_signal_events_mod(score, th=th, cooldown=cooldown, flags=flags)

    # Step 3: Apply price filter
    if flags.get('PRICE_FILTER_RELAX', False):
        pf = build_price_filter_mod(df, er_q=66, atr_q=55, lookback=252)
    else:
        pf = build_price_filter_mod(df, er_q=66, atr_q=66, lookback=252)

    events = [e for e in events if pf(e['peak_idx'])]

    # Step 4: Apply late sell block
    if flags.get('LATE_SELL_BLOCK', False):
        events = apply_late_sell_block(events, df)

    return events, score


def simulate_trading(events, df, initial_cash=INITIAL_CASH, monthly_add=MONTHLY_ADD):
    """
    Simulate trading based on signal events.
    Returns dict with metrics.
    """
    close = df['Close'].values
    dates = df.index
    n = len(df)

    if n < 2:
        return {
            'final_portfolio': initial_cash,
            'total_invested': initial_cash,
            'ann_return': 0.0,
            'mdd': 0.0,
            'n_buys': 0,
            'n_sells': 0,
            'n_signals': 0,
            'win_rate': 0.0,
            'portfolio_history': [],
        }

    cash = initial_cash
    shares = 0.0
    total_invested = initial_cash

    # Track monthly additions
    last_month = (dates[0].year, dates[0].month)

    # Build event lookup: peak_idx -> event
    event_map = {}
    for ev in events:
        pidx = ev['peak_idx']
        if 0 <= pidx < n:
            event_map[pidx] = ev

    portfolio_values = []
    n_buys = 0
    n_sells = 0
    trades = []  # list of (buy_price, sell_price) for win rate

    # Track buy prices for win rate calculation
    buy_log = []  # list of (price, shares_bought)

    peak_portfolio = 0.0
    max_drawdown = 0.0

    for i in range(n):
        # Monthly cash injection
        cur_month = (dates[i].year, dates[i].month)
        if cur_month != last_month:
            cash += monthly_add
            total_invested += monthly_add
            last_month = cur_month

        # Check for signal event at this index
        if i in event_map:
            ev = event_map[i]
            price = close[i]

            if ev['type'] == 'bottom':
                # BUY signal
                if abs(ev['peak_val']) >= STRONG_BUY_TH:
                    buy_pct = BUY_STRONG_PCT
                else:
                    buy_pct = BUY_NORMAL_PCT

                buy_amount = cash * buy_pct
                if buy_amount > 1.0 and price > 0:
                    shares_bought = buy_amount / price
                    shares += shares_bought
                    cash -= buy_amount
                    n_buys += 1
                    buy_log.append((price, shares_bought))

            elif ev['type'] == 'top':
                # SELL signal
                if ev['peak_val'] <= STRONG_SELL_TH:
                    sell_pct = SELL_STRONG_PCT
                else:
                    sell_pct = SELL_NORMAL_PCT

                shares_to_sell = shares * sell_pct
                if shares_to_sell > 0 and price > 0:
                    proceeds = shares_to_sell * price
                    shares -= shares_to_sell
                    cash += proceeds
                    n_sells += 1

                    # Calculate win/loss vs average buy price
                    if buy_log:
                        total_cost = sum(p * s for p, s in buy_log)
                        total_shares = sum(s for _, s in buy_log)
                        if total_shares > 0:
                            avg_buy = total_cost / total_shares
                            trades.append((avg_buy, price))

        # Portfolio value
        pv = cash + shares * close[i]
        portfolio_values.append(pv)

        # MDD
        if pv > peak_portfolio:
            peak_portfolio = pv
        if peak_portfolio > 0:
            dd = (peak_portfolio - pv) / peak_portfolio
            if dd > max_drawdown:
                max_drawdown = dd

    final_portfolio = portfolio_values[-1] if portfolio_values else initial_cash

    # Annualized return
    years = len(df) / 252.0
    if years > 0 and total_invested > 0:
        ann_return = ((final_portfolio / total_invested) ** (1.0 / years) - 1) * 100
    else:
        ann_return = 0.0

    # Win rate
    wins = sum(1 for b, s in trades if s > b)
    win_rate = (wins / len(trades) * 100) if trades else 0.0

    return {
        'final_portfolio': final_portfolio,
        'total_invested': total_invested,
        'ann_return': ann_return,
        'mdd': max_drawdown * 100,
        'n_buys': n_buys,
        'n_sells': n_sells,
        'n_signals': len(events),
        'win_rate': win_rate,
        'portfolio_history': portfolio_values,
    }


# ============================================================
# MAIN BACKTEST
# ============================================================

def load_tickers():
    """Load tickers from watchlist.json"""
    with open(str(_config_path), 'r', encoding='utf-8') as f:
        config = json.load(f)
    tickers = list(config['tickers'].keys())
    benchmarks = config.get('benchmarks', [])
    all_tickers = tickers + [b for b in benchmarks if b not in tickers]
    return all_tickers


def download_all_data(tickers, cache_dir):
    """Download data for all tickers."""
    data = {}
    for ticker in tickers:
        try:
            df = download_data(ticker, start='2000-01-01', end='2026-12-31', cache_dir=cache_dir)
            if len(df) >= 100:
                data[ticker] = df
            else:
                print(f"  SKIP {ticker}: only {len(df)} days")
        except Exception as e:
            print(f"  ERROR downloading {ticker}: {e}")
    return data


def slice_period(df, start, end):
    """Slice DataFrame to a date range. Returns None if insufficient data."""
    mask = (df.index >= start) & (df.index <= end)
    sub = df.loc[mask]
    if len(sub) < 100:
        return None
    return sub


def run_all_backtests(all_data, tickers):
    """
    Run backtests for all configs x periods x tickers.
    Returns: results[config_name][period_name] = list of per-ticker dicts
    """
    results = {}
    total_configs = len(CONFIGS)
    total_periods = len(MARKET_PERIODS)

    for ci, (cfg_name, cfg_flags) in enumerate(CONFIGS.items()):
        results[cfg_name] = {}
        print(f"\n{'='*60}")
        print(f"  Config [{ci+1}/{total_configs}]: {cfg_name}")
        desc_parts = [k for k, v in cfg_flags.items() if v]
        if desc_parts:
            print(f"    Flags: {', '.join(desc_parts)}")
        else:
            print(f"    Flags: (none - baseline)")
        print(f"{'='*60}")

        for pi, period in enumerate(MARKET_PERIODS):
            pname = period['name']
            pstart = period['start']
            pend = period['end']
            results[cfg_name][pname] = []

            ticker_results = []
            for ticker in tickers:
                if ticker not in all_data:
                    continue
                df_full = all_data[ticker]
                df_sub = slice_period(df_full, pstart, pend)
                if df_sub is None:
                    continue

                try:
                    events, score = run_pipeline(df_sub, flags=cfg_flags)
                    metrics = simulate_trading(events, df_sub)
                    metrics['ticker'] = ticker
                    ticker_results.append(metrics)
                except Exception as e:
                    pass  # silently skip errors for individual tickers

            results[cfg_name][pname] = ticker_results
            n_tickers = len(ticker_results)
            if n_tickers > 0:
                avg_ret = np.mean([r['ann_return'] for r in ticker_results])
                avg_final = np.mean([r['final_portfolio'] for r in ticker_results])
                print(f"    {pname}: {n_tickers} tickers, avg_ann_ret={avg_ret:.2f}%, avg_final=${avg_final:.0f}")

    return results


# ============================================================
# AGGREGATION & OUTPUT
# ============================================================

def aggregate_period(ticker_results):
    """Aggregate ticker results for a period."""
    if not ticker_results:
        return {
            'avg_ann_return': np.nan,
            'avg_final_portfolio': np.nan,
            'avg_mdd': np.nan,
            'avg_win_rate': np.nan,
            'total_signals': 0,
            'total_buys': 0,
            'total_sells': 0,
            'n_tickers': 0,
            'median_ann_return': np.nan,
        }

    returns = [r['ann_return'] for r in ticker_results]
    finals = [r['final_portfolio'] for r in ticker_results]
    mdds = [r['mdd'] for r in ticker_results]
    win_rates = [r['win_rate'] for r in ticker_results]
    signals = sum(r['n_signals'] for r in ticker_results)
    buys = sum(r['n_buys'] for r in ticker_results)
    sells = sum(r['n_sells'] for r in ticker_results)

    return {
        'avg_ann_return': np.mean(returns),
        'avg_final_portfolio': np.mean(finals),
        'avg_mdd': np.mean(mdds),
        'avg_win_rate': np.mean(win_rates),
        'total_signals': signals,
        'total_buys': buys,
        'total_sells': sells,
        'n_tickers': len(ticker_results),
        'median_ann_return': np.median(returns),
    }


def print_results(results):
    """Print comprehensive results tables."""
    config_names = list(CONFIGS.keys())
    period_names = [p['name'] for p in MARKET_PERIODS]

    # Aggregate all data
    agg = {}
    for cfg in config_names:
        agg[cfg] = {}
        for pname in period_names:
            agg[cfg][pname] = aggregate_period(results[cfg].get(pname, []))

    # ================================================================
    # TABLE 1: Per-period summary for each config
    # ================================================================
    print("\n" + "=" * 120)
    print("  TABLE 1: PER-PERIOD SUMMARY FOR EACH CONFIGURATION")
    print("=" * 120)

    for cfg in config_names:
        desc_parts = [k for k, v in CONFIGS[cfg].items() if v]
        desc = ', '.join(desc_parts) if desc_parts else '(baseline)'
        print(f"\n--- {cfg} [{desc}] ---")
        header = f"{'Period':<15} {'AvgAnnRet%':>10} {'MedAnnRet%':>11} {'AvgFinalP':>10} {'AvgMDD%':>8} {'WinRate%':>8} {'Signals':>8} {'Buys':>6} {'Sells':>6} {'Tickers':>8}"
        print(header)
        print("-" * len(header))
        for pname in period_names:
            a = agg[cfg][pname]
            print(f"{pname:<15} {a['avg_ann_return']:>10.2f} {a['median_ann_return']:>11.2f} {a['avg_final_portfolio']:>10.0f} {a['avg_mdd']:>8.2f} {a['avg_win_rate']:>8.1f} {a['total_signals']:>8} {a['total_buys']:>6} {a['total_sells']:>6} {a['n_tickers']:>8}")

    # ================================================================
    # TABLE 2: CAGR cross-table
    # ================================================================
    print("\n\n" + "=" * 120)
    print("  TABLE 2: CAGR (Avg Annualized Return %) — Rows=Periods, Cols=Configs")
    print("=" * 120)

    # Header
    col_width = 8
    header = f"{'Period':<15}"
    for cfg in config_names:
        header += f" {cfg:>{col_width}}"
    print(header)
    print("-" * len(header))

    for pname in period_names:
        row = f"{pname:<15}"
        for cfg in config_names:
            val = agg[cfg][pname]['avg_ann_return']
            if np.isnan(val):
                row += f" {'N/A':>{col_width}}"
            else:
                row += f" {val:>{col_width}.2f}"
        print(row)

    # Average across periods (excluding Full)
    print("-" * len(header))
    row = f"{'Avg(1-6)':<15}"
    for cfg in config_names:
        vals = [agg[cfg][pname]['avg_ann_return'] for pname in period_names[:-1]
                if not np.isnan(agg[cfg][pname]['avg_ann_return'])]
        avg = np.mean(vals) if vals else np.nan
        if np.isnan(avg):
            row += f" {'N/A':>{col_width}}"
        else:
            row += f" {avg:>{col_width}.2f}"
    print(row)

    # ================================================================
    # TABLE 3: Final Portfolio cross-table
    # ================================================================
    print("\n\n" + "=" * 120)
    print("  TABLE 3: AVG FINAL PORTFOLIO ($) — Rows=Periods, Cols=Configs")
    print("=" * 120)

    header = f"{'Period':<15}"
    for cfg in config_names:
        header += f" {cfg:>{col_width}}"
    print(header)
    print("-" * len(header))

    for pname in period_names:
        row = f"{pname:<15}"
        for cfg in config_names:
            val = agg[cfg][pname]['avg_final_portfolio']
            if np.isnan(val):
                row += f" {'N/A':>{col_width}}"
            else:
                row += f" {val:>{col_width}.0f}"
        print(row)

    # ================================================================
    # TABLE 4: MDD cross-table
    # ================================================================
    print("\n\n" + "=" * 120)
    print("  TABLE 4: AVG MAX DRAWDOWN (%) — Rows=Periods, Cols=Configs (lower is better)")
    print("=" * 120)

    header = f"{'Period':<15}"
    for cfg in config_names:
        header += f" {cfg:>{col_width}}"
    print(header)
    print("-" * len(header))

    for pname in period_names:
        row = f"{pname:<15}"
        for cfg in config_names:
            val = agg[cfg][pname]['avg_mdd']
            if np.isnan(val):
                row += f" {'N/A':>{col_width}}"
            else:
                row += f" {val:>{col_width}.2f}"
        print(row)

    # ================================================================
    # TABLE 5: Signal count comparison
    # ================================================================
    print("\n\n" + "=" * 120)
    print("  TABLE 5: TOTAL SIGNAL COUNT — Rows=Periods, Cols=Configs")
    print("=" * 120)

    header = f"{'Period':<15}"
    for cfg in config_names:
        header += f" {cfg:>{col_width}}"
    print(header)
    print("-" * len(header))

    for pname in period_names:
        row = f"{pname:<15}"
        for cfg in config_names:
            val = agg[cfg][pname]['total_signals']
            row += f" {val:>{col_width}}"
        print(row)

    # Buy/Sell breakdown for 7.Full period
    print("\n  Signal Breakdown (7.Full period):")
    header2 = f"{'Config':<10} {'TotalSig':>10} {'Buys':>8} {'Sells':>8} {'B/S Ratio':>10}"
    print(f"  {header2}")
    print(f"  {'-' * len(header2)}")
    for cfg in config_names:
        a = agg[cfg].get('7.Full', {})
        ts = a.get('total_signals', 0)
        tb = a.get('total_buys', 0)
        tsl = a.get('total_sells', 0)
        ratio = f"{tb/tsl:.2f}" if tsl > 0 else "inf"
        print(f"  {cfg:<10} {ts:>10} {tb:>8} {tsl:>8} {ratio:>10}")

    # ================================================================
    # TABLE 6: Ranking by avg annualized return (7.Full period)
    # ================================================================
    print("\n\n" + "=" * 120)
    print("  TABLE 6: RANKING BY FULL-PERIOD AVG ANNUALIZED RETURN")
    print("=" * 120)

    ranking = []
    for cfg in config_names:
        a_full = agg[cfg].get('7.Full', {})
        avg_ret = a_full.get('avg_ann_return', np.nan)
        avg_mdd = a_full.get('avg_mdd', np.nan)
        avg_final = a_full.get('avg_final_portfolio', np.nan)
        avg_wr = a_full.get('avg_win_rate', np.nan)
        total_sig = a_full.get('total_signals', 0)

        # Also compute average across periods 1-6
        period_rets = [agg[cfg][pname]['avg_ann_return'] for pname in period_names[:-1]
                       if not np.isnan(agg[cfg][pname]['avg_ann_return'])]
        avg_per_ret = np.mean(period_rets) if period_rets else np.nan

        ranking.append({
            'config': cfg,
            'full_ann_ret': avg_ret,
            'avg_period_ret': avg_per_ret,
            'full_mdd': avg_mdd,
            'full_final': avg_final,
            'full_wr': avg_wr,
            'full_signals': total_sig,
        })

    ranking.sort(key=lambda x: x['full_ann_ret'] if not np.isnan(x['full_ann_ret']) else -999, reverse=True)

    header = f"{'Rank':>4} {'Config':<10} {'FullAnnRet%':>12} {'AvgPerRet%':>11} {'FullMDD%':>9} {'FinalPort$':>11} {'WinRate%':>9} {'Signals':>8}"
    print(header)
    print("-" * len(header))
    for rank, r in enumerate(ranking, 1):
        print(f"{rank:>4} {r['config']:<10} {r['full_ann_ret']:>12.3f} {r['avg_period_ret']:>11.3f} {r['full_mdd']:>9.2f} {r['full_final']:>11.0f} {r['full_wr']:>9.1f} {r['full_signals']:>8}")

    # ================================================================
    # TABLE 7: Improvement vs BASE (%p difference)
    # ================================================================
    print("\n\n" + "=" * 120)
    print("  TABLE 7: IMPROVEMENT vs BASE (%p difference in Avg Annualized Return)")
    print("=" * 120)

    base_rets = {}
    for pname in period_names:
        base_rets[pname] = agg['BASE'][pname]['avg_ann_return']

    header = f"{'Period':<15}"
    non_base = [c for c in config_names if c != 'BASE']
    for cfg in non_base:
        header += f" {cfg:>{col_width}}"
    print(header)
    print("-" * len(header))

    for pname in period_names:
        row = f"{pname:<15}"
        base_val = base_rets[pname]
        for cfg in non_base:
            val = agg[cfg][pname]['avg_ann_return']
            if np.isnan(val) or np.isnan(base_val):
                row += f" {'N/A':>{col_width}}"
            else:
                diff = val - base_val
                sign = '+' if diff >= 0 else ''
                row += f" {sign}{diff:>{col_width-1}.2f}"
        print(row)

    # Average improvement
    print("-" * len(header))
    row = f"{'Avg(1-6)':<15}"
    for cfg in non_base:
        diffs = []
        for pname in period_names[:-1]:
            bv = base_rets[pname]
            cv = agg[cfg][pname]['avg_ann_return']
            if not np.isnan(bv) and not np.isnan(cv):
                diffs.append(cv - bv)
        avg_diff = np.mean(diffs) if diffs else np.nan
        if np.isnan(avg_diff):
            row += f" {'N/A':>{col_width}}"
        else:
            sign = '+' if avg_diff >= 0 else ''
            row += f" {sign}{avg_diff:>{col_width-1}.2f}"
    print(row)

    # Also show MDD difference
    print(f"\n  MDD Improvement vs BASE (%p, negative means less drawdown = better):")
    header_m = f"{'Period':<15}"
    for cfg in non_base:
        header_m += f" {cfg:>{col_width}}"
    print(header_m)
    print("-" * len(header_m))

    for pname in period_names:
        row = f"{pname:<15}"
        base_mdd = agg['BASE'][pname]['avg_mdd']
        for cfg in non_base:
            val = agg[cfg][pname]['avg_mdd']
            if np.isnan(val) or np.isnan(base_mdd):
                row += f" {'N/A':>{col_width}}"
            else:
                diff = val - base_mdd
                sign = '+' if diff >= 0 else ''
                row += f" {sign}{diff:>{col_width-1}.2f}"
        print(row)

    # ================================================================
    # TABLE 8: Top 5 Configurations Analysis
    # ================================================================
    print("\n\n" + "=" * 120)
    print("  TABLE 8: TOP 5 CONFIGURATIONS — DETAILED ANALYSIS")
    print("=" * 120)

    top5 = ranking[:5]
    for rank, r in enumerate(top5, 1):
        cfg = r['config']
        desc_parts = [k for k, v in CONFIGS[cfg].items() if v]
        desc = ', '.join(desc_parts) if desc_parts else '(baseline)'

        print(f"\n  #{rank}: {cfg} [{desc}]")
        print(f"  {'':->70}")
        print(f"  Full Period: AnnRet={r['full_ann_ret']:.3f}%, MDD={r['full_mdd']:.2f}%, "
              f"FinalPortfolio=${r['full_final']:.0f}, WinRate={r['full_wr']:.1f}%, "
              f"Signals={r['full_signals']}")

        base_full_ret = agg['BASE']['7.Full']['avg_ann_return']
        if not np.isnan(base_full_ret) and not np.isnan(r['full_ann_ret']):
            diff = r['full_ann_ret'] - base_full_ret
            sign = '+' if diff >= 0 else ''
            print(f"  vs BASE: {sign}{diff:.3f}%p annualized return difference")

        print(f"\n  Per-Period Breakdown:")
        print(f"  {'Period':<15} {'AnnRet%':>10} {'MDD%':>8} {'FinalP$':>10} {'Signals':>8} {'vsBase%p':>10}")
        print(f"  {'-'*65}")
        for pname in period_names:
            a = agg[cfg][pname]
            base_a = agg['BASE'][pname]
            diff = a['avg_ann_return'] - base_a['avg_ann_return'] if not np.isnan(a['avg_ann_return']) and not np.isnan(base_a['avg_ann_return']) else np.nan
            diff_str = f"{diff:+.2f}" if not np.isnan(diff) else "N/A"
            print(f"  {pname:<15} {a['avg_ann_return']:>10.2f} {a['avg_mdd']:>8.2f} {a['avg_final_portfolio']:>10.0f} {a['total_signals']:>8} {diff_str:>10}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n\n" + "=" * 120)
    print("  EXECUTIVE SUMMARY")
    print("=" * 120)

    best = ranking[0]
    worst = ranking[-1]
    base_rank = next(i+1 for i, r in enumerate(ranking) if r['config'] == 'BASE')

    print(f"\n  Total configurations tested: {len(config_names)}")
    print(f"  Total market periods: {len(period_names)}")
    print(f"  Tickers tested: {agg['BASE']['7.Full'].get('n_tickers', 'N/A')}")
    print(f"\n  BEST  config: {best['config']} (AnnRet={best['full_ann_ret']:.3f}%, MDD={best['full_mdd']:.2f}%)")
    print(f"  WORST config: {worst['config']} (AnnRet={worst['full_ann_ret']:.3f}%, MDD={worst['full_mdd']:.2f}%)")
    print(f"  BASE  rank: #{base_rank} of {len(ranking)}")

    if best['config'] != 'BASE':
        base_ret = agg['BASE']['7.Full']['avg_ann_return']
        improvement = best['full_ann_ret'] - base_ret if not np.isnan(base_ret) else np.nan
        if not np.isnan(improvement):
            print(f"\n  Best improvement over BASE: {improvement:+.3f}%p annualized")
            desc_parts = [k for k, v in CONFIGS[best['config']].items() if v]
            print(f"  Improvements used: {', '.join(desc_parts)}")

    # Individual improvements impact
    print(f"\n  Individual Improvement Impact (Full Period AnnRet vs BASE):")
    indiv = ['F1', 'F2', 'F3', 'F4', 'F5']
    names = ['PRICE_FILTER_RELAX', 'ACT_MUL_CAP', 'DIV_REBALANCE', 'LATE_SELL_BLOCK', 'BUY_TH_RAISE']
    base_ret = agg['BASE']['7.Full']['avg_ann_return']
    for fi, fname in zip(indiv, names):
        fi_ret = agg[fi]['7.Full']['avg_ann_return']
        if not np.isnan(fi_ret) and not np.isnan(base_ret):
            diff = fi_ret - base_ret
            sign = '+' if diff >= 0 else ''
            verdict = "BETTER" if diff > 0.01 else ("WORSE" if diff < -0.01 else "NEUTRAL")
            print(f"    {fi} ({fname}): {sign}{diff:.3f}%p [{verdict}]")
        else:
            print(f"    {fi} ({fname}): N/A")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    t_start = time.time()

    print("=" * 80)
    print("  V4 INDICATOR IMPROVEMENTS - COMPREHENSIVE BACKTEST")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Load tickers
    tickers = load_tickers()
    print(f"\nTickers ({len(tickers)}): {', '.join(tickers)}")
    print(f"Configs ({len(CONFIGS)}): {', '.join(CONFIGS.keys())}")
    print(f"Periods ({len(MARKET_PERIODS)}): {', '.join(p['name'] for p in MARKET_PERIODS)}")

    # Download all data
    print(f"\n{'='*60}")
    print("  DOWNLOADING / LOADING DATA")
    print(f"{'='*60}")
    all_data = download_all_data(tickers, _cache_dir)
    print(f"\nLoaded data for {len(all_data)} tickers")

    # Run backtests
    print(f"\n{'='*60}")
    print("  RUNNING BACKTESTS")
    print(f"  {len(CONFIGS)} configs x {len(MARKET_PERIODS)} periods x {len(all_data)} tickers")
    print(f"{'='*60}")

    results = run_all_backtests(all_data, tickers)

    # Print results
    print_results(results)

    elapsed = time.time() - t_start
    print(f"\n\nTotal runtime: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
