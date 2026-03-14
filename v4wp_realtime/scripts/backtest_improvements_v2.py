"""
V4 C14 기반 개선안 종합 백테스트
=================================
C14(PRICE_FILTER_RELAX + LATE_SELL_BLOCK)를 베이스라인으로,
분석에서 도출된 8가지 개선안을 개별 + 조합 테스트.

개별 개선안 (8개):
  I1: STRONG_BUY_TH 상향 (0.15 -> 0.25)
  I2: STRONG_BUY_TH 더 상향 (0.15 -> 0.30)
  I3: 워치리스트 정리 (PGY, BA, INTC 제거)
  I4: 매도 비율 상향 (5%/10% -> 8%/15%)
  I5: 매수 비율 상향 (30%/50% -> 40%/60%)
  I6: LATE_SELL_BLOCK 완화 (5% -> 7%)
  I7: LATE_SELL_BLOCK 더 완화 (5% -> 10%)
  I8: 3단계 신호 체계 (20%/35%/50%)

조합 (누적 적용):
  C15: I1 + I4         (STRONG_BUY 0.25 + 매도상향)
  C16: I1 + I3 + I4    (+ 워치리스트 정리)
  C17: I1 + I3 + I4 + I6   (+ LATE_SELL 0.07)
  C18: I1 + I3 + I4 + I5   (+ 매수상향)
  C19: I1 + I3 + I4 + I5 + I6  (전부)
  C20: I2 + I3 + I4    (STRONG_BUY 0.30 변형)
  C21: I2 + I3 + I4 + I5 + I6  (0.30 변형 전부)
  C22: I8 + I3 + I4    (3단계 + 정리 + 매도상향)
  C23: I8 + I3 + I4 + I5 + I6  (3단계 전부)
  C24: I1 + I3 + I4 + I7   (LATE_SELL 0.10 변형)
  C25: I1 + I3 + I5    (매수상향만)
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
_project_root = Path(__file__).resolve().parents[2]
_config_path = _project_root / 'v4wp_realtime' / 'config' / 'watchlist.json'
_cache_dir = str(_project_root / 'cache')

sys.path.insert(0, str(_project_root))

from real_market_backtest import (
    download_data,
    calc_v4_score,
    detect_signal_events,
    build_price_filter,
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
# REMOVED TICKERS FOR WATCHLIST CLEANUP
# ============================================================
REMOVED_TICKERS = {'PGY', 'BA', 'INTC'}

# ============================================================
# CONFIGURATION SYSTEM
# ============================================================
# Each config is a dict of parameter overrides over C14 baseline

C14_BASELINE = {
    'strong_buy_th': 0.15,
    'strong_sell_th': -0.25,
    'buy_strong_pct': 0.50,
    'buy_normal_pct': 0.30,
    'sell_strong_pct': 0.10,
    'sell_normal_pct': 0.05,
    'late_sell_drop_th': 0.05,
    'atr_quantile': 55,
    'cleanup_watchlist': False,
    'three_tier': False,         # 3-tier signal system
    'tier_weak_pct': 0.20,       # weak signal buy %
    'tier_mid_pct': 0.35,        # mid signal buy %
    'tier_strong_pct': 0.50,     # strong signal buy %
    'tier_weak_th': 0.20,        # weak/mid boundary
    'tier_strong_th': 0.30,      # mid/strong boundary
}

CONFIGS = {
    # Baseline = C14 as-is
    'C14':  {},

    # Individual improvements
    'I1':   {'strong_buy_th': 0.25},
    'I2':   {'strong_buy_th': 0.30},
    'I3':   {'cleanup_watchlist': True},
    'I4':   {'sell_normal_pct': 0.08, 'sell_strong_pct': 0.15},
    'I5':   {'buy_normal_pct': 0.40, 'buy_strong_pct': 0.60},
    'I6':   {'late_sell_drop_th': 0.07},
    'I7':   {'late_sell_drop_th': 0.10},
    'I8':   {'three_tier': True},

    # Combinations
    'C15':  {'strong_buy_th': 0.25, 'sell_normal_pct': 0.08, 'sell_strong_pct': 0.15},
    'C16':  {'strong_buy_th': 0.25, 'sell_normal_pct': 0.08, 'sell_strong_pct': 0.15,
             'cleanup_watchlist': True},
    'C17':  {'strong_buy_th': 0.25, 'sell_normal_pct': 0.08, 'sell_strong_pct': 0.15,
             'cleanup_watchlist': True, 'late_sell_drop_th': 0.07},
    'C18':  {'strong_buy_th': 0.25, 'sell_normal_pct': 0.08, 'sell_strong_pct': 0.15,
             'cleanup_watchlist': True, 'buy_normal_pct': 0.40, 'buy_strong_pct': 0.60},
    'C19':  {'strong_buy_th': 0.25, 'sell_normal_pct': 0.08, 'sell_strong_pct': 0.15,
             'cleanup_watchlist': True, 'buy_normal_pct': 0.40, 'buy_strong_pct': 0.60,
             'late_sell_drop_th': 0.07},
    'C20':  {'strong_buy_th': 0.30, 'sell_normal_pct': 0.08, 'sell_strong_pct': 0.15,
             'cleanup_watchlist': True},
    'C21':  {'strong_buy_th': 0.30, 'sell_normal_pct': 0.08, 'sell_strong_pct': 0.15,
             'cleanup_watchlist': True, 'buy_normal_pct': 0.40, 'buy_strong_pct': 0.60,
             'late_sell_drop_th': 0.07},
    'C22':  {'three_tier': True, 'sell_normal_pct': 0.08, 'sell_strong_pct': 0.15,
             'cleanup_watchlist': True},
    'C23':  {'three_tier': True, 'sell_normal_pct': 0.08, 'sell_strong_pct': 0.15,
             'cleanup_watchlist': True, 'buy_normal_pct': 0.40, 'buy_strong_pct': 0.60,
             'late_sell_drop_th': 0.07},
    'C24':  {'strong_buy_th': 0.25, 'sell_normal_pct': 0.08, 'sell_strong_pct': 0.15,
             'cleanup_watchlist': True, 'late_sell_drop_th': 0.10},
    'C25':  {'strong_buy_th': 0.25, 'cleanup_watchlist': True,
             'buy_normal_pct': 0.40, 'buy_strong_pct': 0.60},
}


def get_params(cfg_overrides):
    """Merge config overrides with C14 baseline."""
    params = dict(C14_BASELINE)
    params.update(cfg_overrides)
    return params


# ============================================================
# CORE PIPELINE
# ============================================================

def run_pipeline(df_full, period_start, period_end, params):
    """
    Full C14-based pipeline for a single ticker.
    Returns (events_in_period, df_period, n_blocked_sells) or None.
    """
    if len(df_full) < 100:
        return None

    # V4 score (unmodified from C14)
    v4_score = calc_v4_score(df_full, w=20)

    # Signal detection
    events_all = detect_signal_events(v4_score, th=0.15, cooldown=5)

    # Price filter with C14 relaxed ATR
    pf = build_price_filter(df_full, er_q=66, atr_q=params['atr_quantile'], lookback=252)
    events_filtered = [e for e in events_all if pf(e['peak_idx'])]

    # LATE_SELL_BLOCK
    close = df_full['Close'].values
    rolling_high_20 = df_full['Close'].rolling(20, min_periods=1).max().values
    drop_th = params['late_sell_drop_th']

    events_final = []
    n_blocked = 0
    for ev in events_filtered:
        if ev['type'] == 'top':
            pidx = ev['peak_idx']
            if pidx < len(close):
                price_at_signal = close[pidx]
                rh = rolling_high_20[pidx]
                drop_pct = (rh - price_at_signal) / rh if rh > 0 else 0
                if drop_pct > drop_th:
                    n_blocked += 1
                    continue
        events_final.append(ev)

    # Slice to period
    mask = (df_full.index >= period_start) & (df_full.index <= period_end)
    df_period = df_full.loc[mask]
    if len(df_period) < 10:
        return None

    period_start_ts = df_period.index[0]
    period_end_ts = df_period.index[-1]
    full_start_loc = df_full.index.get_loc(period_start_ts)
    full_end_loc = df_full.index.get_loc(period_end_ts)

    # Filter events within period, remap to period indices
    period_events = []
    period_blocked = 0
    for e in events_final:
        if full_start_loc <= e['peak_idx'] <= full_end_loc:
            period_events.append(e)

    # Count blocked sells that would have been in period
    for ev in events_filtered:
        if ev['type'] == 'top' and full_start_loc <= ev['peak_idx'] <= full_end_loc:
            pidx = ev['peak_idx']
            if pidx < len(close):
                price_at_signal = close[pidx]
                rh = rolling_high_20[pidx]
                drop_pct = (rh - price_at_signal) / rh if rh > 0 else 0
                if drop_pct > drop_th:
                    period_blocked += 1

    return period_events, df_period, df_full, period_blocked


def simulate_trading(period_events, df_period, df_full, params):
    """
    Simulate trading with given params.
    """
    INITIAL_CASH = 1000.0
    MONTHLY_ADD = 100.0

    close_full = df_full['Close'].values
    dates = df_period.index
    n = len(df_period)

    cash = INITIAL_CASH
    shares = 0.0
    total_deposited = INITIAL_CASH
    cost_basis = 0.0

    n_buys = 0
    n_sells = 0
    n_strong_buys = 0
    n_normal_buys = 0

    months_added = set()
    portfolio_values = []
    peak_portfolio = 0.0
    max_drawdown = 0.0

    # Build event lookup: full_idx -> events
    event_dict = {}
    for e in period_events:
        idx = e['peak_idx']
        if idx not in event_dict:
            event_dict[idx] = []
        event_dict[idx].append(e)

    for i in range(n):
        date = dates[i]
        price = df_period['Close'].iloc[i]
        full_idx = df_full.index.get_loc(date)

        # Monthly DCA
        month_key = (date.year, date.month)
        if month_key not in months_added:
            if i > 0:
                cash += MONTHLY_ADD
                total_deposited += MONTHLY_ADD
            months_added.add(month_key)

        # Check signals
        if full_idx in event_dict:
            for ev in event_dict[full_idx]:
                if ev['type'] == 'bottom':
                    # Determine buy percentage
                    pv = abs(ev['peak_val'])

                    if params['three_tier']:
                        # 3-tier system
                        if pv >= params['tier_strong_th']:
                            buy_pct = params['tier_strong_pct']
                        elif pv >= params['tier_weak_th']:
                            buy_pct = params['tier_mid_pct']
                        else:
                            buy_pct = params['tier_weak_pct']
                    else:
                        # 2-tier system
                        if pv >= params['strong_buy_th']:
                            buy_pct = params['buy_strong_pct']
                            n_strong_buys += 1
                        else:
                            buy_pct = params['buy_normal_pct']
                            n_normal_buys += 1

                    buy_amount = cash * buy_pct
                    if buy_amount > 1.0 and price > 0:
                        new_shares = buy_amount / price
                        total_cost = cost_basis * shares + buy_amount
                        shares += new_shares
                        cost_basis = total_cost / shares if shares > 0 else 0
                        cash -= buy_amount
                        n_buys += 1

                elif ev['type'] == 'top':
                    is_strong = ev['peak_val'] <= params['strong_sell_th']
                    sell_pct = params['sell_strong_pct'] if is_strong else params['sell_normal_pct']
                    sell_shares = shares * sell_pct
                    if sell_shares > 0.0001 and price > 0:
                        proceeds = sell_shares * price
                        shares -= sell_shares
                        cash += proceeds
                        n_sells += 1

        # Portfolio value
        pv = cash + shares * price
        portfolio_values.append(pv)

        # MDD
        if pv > peak_portfolio:
            peak_portfolio = pv
        if peak_portfolio > 0:
            dd = (peak_portfolio - pv) / peak_portfolio
            if dd > max_drawdown:
                max_drawdown = dd

    final_portfolio = portfolio_values[-1] if portfolio_values else INITIAL_CASH

    # Annualized return
    period_start_ts = dates[0]
    period_end_ts = dates[-1]
    n_days = (period_end_ts - period_start_ts).days
    n_years = n_days / 365.25
    if n_years > 0.5 and total_deposited > 0:
        ann_return = ((final_portfolio / total_deposited) ** (1.0 / n_years) - 1) * 100
    else:
        ann_return = ((final_portfolio / total_deposited) - 1) * 100

    # Open position value
    open_val = shares * df_period['Close'].iloc[-1] if len(df_period) > 0 else 0

    return {
        'final_portfolio': final_portfolio,
        'total_deposited': total_deposited,
        'ann_return': ann_return,
        'mdd': max_drawdown * 100,
        'n_buys': n_buys,
        'n_sells': n_sells,
        'n_strong_buys': n_strong_buys,
        'n_normal_buys': n_normal_buys,
        'cash_remaining': cash,
        'open_position_val': open_val,
        'profit_rate': (final_portfolio / total_deposited - 1) * 100,
    }


# ============================================================
# MAIN
# ============================================================

def load_tickers():
    with open(str(_config_path), 'r', encoding='utf-8') as f:
        config = json.load(f)
    tickers = list(config['tickers'].keys())
    benchmarks = config.get('benchmarks', [])
    all_tickers = tickers + [b for b in benchmarks if b not in tickers]
    return all_tickers


def download_all_data(tickers, cache_dir):
    data = {}
    for ticker in tickers:
        try:
            df = download_data(ticker, start='2000-01-01', end='2026-12-31', cache_dir=cache_dir)
            if len(df) >= 100:
                data[ticker] = df
        except Exception as e:
            print(f"  ERROR {ticker}: {e}")
    return data


def run_all_backtests(all_data, all_tickers):
    """Run all configs x periods x tickers."""
    results = {}
    total_configs = len(CONFIGS)

    for ci, (cfg_name, cfg_overrides) in enumerate(CONFIGS.items()):
        params = get_params(cfg_overrides)
        results[cfg_name] = {}

        # Determine which tickers to use
        if params['cleanup_watchlist']:
            use_tickers = [t for t in all_tickers if t not in REMOVED_TICKERS]
        else:
            use_tickers = all_tickers

        print(f"\n  [{ci+1}/{total_configs}] {cfg_name}: ", end='', flush=True)

        for pi, period in enumerate(MARKET_PERIODS):
            pname = period['name']
            pstart = period['start']
            pend = period['end']
            ticker_results = []

            for ticker in use_tickers:
                if ticker not in all_data:
                    continue
                df_full = all_data[ticker]

                pipeline_result = run_pipeline(df_full, pstart, pend, params)
                if pipeline_result is None:
                    continue

                period_events, df_period, df_full_ref, n_blocked = pipeline_result
                metrics = simulate_trading(period_events, df_period, df_full_ref, params)
                metrics['ticker'] = ticker
                metrics['n_blocked'] = n_blocked
                ticker_results.append(metrics)

            results[cfg_name][pname] = ticker_results

        # Summary for this config
        full_results = results[cfg_name].get('7.Full', [])
        if full_results:
            avg_ret = np.mean([r['ann_return'] for r in full_results])
            avg_mdd = np.mean([r['mdd'] for r in full_results])
            n_t = len(full_results)
            print(f"{n_t}종목, avg_ann={avg_ret:.2f}%, avg_mdd=-{avg_mdd:.1f}%")
        else:
            print("no data")

    return results


def aggregate_period(ticker_results):
    if not ticker_results:
        return {k: np.nan for k in ['avg_ann', 'med_ann', 'std_ann', 'avg_mdd',
                                     'avg_final', 'n_tickers', 'total_buys',
                                     'total_sells', 'total_blocked',
                                     'total_strong_buys', 'total_normal_buys']}
    rets = [r['ann_return'] for r in ticker_results]
    return {
        'avg_ann': np.mean(rets),
        'med_ann': np.median(rets),
        'std_ann': np.std(rets),
        'avg_mdd': np.mean([r['mdd'] for r in ticker_results]),
        'avg_final': np.mean([r['final_portfolio'] for r in ticker_results]),
        'n_tickers': len(ticker_results),
        'total_buys': sum(r['n_buys'] for r in ticker_results),
        'total_sells': sum(r['n_sells'] for r in ticker_results),
        'total_blocked': sum(r.get('n_blocked', 0) for r in ticker_results),
        'total_strong_buys': sum(r.get('n_strong_buys', 0) for r in ticker_results),
        'total_normal_buys': sum(r.get('n_normal_buys', 0) for r in ticker_results),
    }


def print_results(results):
    config_names = list(CONFIGS.keys())
    period_names = [p['name'] for p in MARKET_PERIODS]

    # Aggregate
    agg = {}
    for cfg in config_names:
        agg[cfg] = {}
        for pname in period_names:
            agg[cfg][pname] = aggregate_period(results[cfg].get(pname, []))

    # ================================================================
    # TABLE 1: CAGR cross-table
    # ================================================================
    print("\n\n" + "=" * 160)
    print("  TABLE 1: 평균 연환산 수익률(%) — 행=기간, 열=구성")
    print("=" * 160)

    cw = 7
    header = f"{'기간':<15}"
    for cfg in config_names:
        header += f" {cfg:>{cw}}"
    print(header)
    print("-" * len(header))

    for pname in period_names:
        row = f"{pname:<15}"
        for cfg in config_names:
            val = agg[cfg][pname]['avg_ann']
            if np.isnan(val):
                row += f" {'N/A':>{cw}}"
            else:
                row += f" {val:>{cw}.2f}"
        print(row)

    # Average 1-6
    print("-" * len(header))
    row = f"{'Avg(1-6)':<15}"
    for cfg in config_names:
        vals = [agg[cfg][pn]['avg_ann'] for pn in period_names[:-1]
                if not np.isnan(agg[cfg][pn]['avg_ann'])]
        avg = np.mean(vals) if vals else np.nan
        row += f" {avg:>{cw}.2f}" if not np.isnan(avg) else f" {'N/A':>{cw}}"
    print(row)

    # ================================================================
    # TABLE 2: MDD cross-table
    # ================================================================
    print("\n\n" + "=" * 160)
    print("  TABLE 2: 평균 MDD(%) — 행=기간, 열=구성 (낮을수록 좋음)")
    print("=" * 160)

    header = f"{'기간':<15}"
    for cfg in config_names:
        header += f" {cfg:>{cw}}"
    print(header)
    print("-" * len(header))

    for pname in period_names:
        row = f"{pname:<15}"
        for cfg in config_names:
            val = agg[cfg][pname]['avg_mdd']
            if np.isnan(val):
                row += f" {'N/A':>{cw}}"
            else:
                row += f" {val:>{cw}.1f}"
        print(row)

    # ================================================================
    # TABLE 3: C14 대비 개선(%p)
    # ================================================================
    print("\n\n" + "=" * 160)
    print("  TABLE 3: C14 대비 수익률 변화(%p) — 양수=개선")
    print("=" * 160)

    non_base = [c for c in config_names if c != 'C14']
    header = f"{'기간':<15}"
    for cfg in non_base:
        header += f" {cfg:>{cw}}"
    print(header)
    print("-" * len(header))

    for pname in period_names:
        row = f"{pname:<15}"
        base_val = agg['C14'][pname]['avg_ann']
        for cfg in non_base:
            val = agg[cfg][pname]['avg_ann']
            if np.isnan(val) or np.isnan(base_val):
                row += f" {'N/A':>{cw}}"
            else:
                diff = val - base_val
                sign = '+' if diff >= 0 else ''
                row += f" {sign}{diff:>{cw-1}.2f}"
        print(row)

    # Avg 1-6
    print("-" * len(header))
    row = f"{'Avg(1-6)':<15}"
    for cfg in non_base:
        diffs = []
        for pn in period_names[:-1]:
            bv = agg['C14'][pn]['avg_ann']
            cv = agg[cfg][pn]['avg_ann']
            if not np.isnan(bv) and not np.isnan(cv):
                diffs.append(cv - bv)
        avg_d = np.mean(diffs) if diffs else np.nan
        if np.isnan(avg_d):
            row += f" {'N/A':>{cw}}"
        else:
            sign = '+' if avg_d >= 0 else ''
            row += f" {sign}{avg_d:>{cw-1}.2f}"
    print(row)

    # ================================================================
    # TABLE 4: MDD 변화 (%p)
    # ================================================================
    print("\n\n" + "=" * 160)
    print("  TABLE 4: C14 대비 MDD 변화(%p) — 음수=MDD 개선(감소)")
    print("=" * 160)

    header = f"{'기간':<15}"
    for cfg in non_base:
        header += f" {cfg:>{cw}}"
    print(header)
    print("-" * len(header))

    for pname in period_names:
        row = f"{pname:<15}"
        base_mdd = agg['C14'][pname]['avg_mdd']
        for cfg in non_base:
            val = agg[cfg][pname]['avg_mdd']
            if np.isnan(val) or np.isnan(base_mdd):
                row += f" {'N/A':>{cw}}"
            else:
                diff = val - base_mdd
                sign = '+' if diff >= 0 else ''
                row += f" {sign}{diff:>{cw-1}.1f}"
        print(row)

    # ================================================================
    # TABLE 5: Signal breakdown (7.Full only)
    # ================================================================
    print("\n\n" + "=" * 160)
    print("  TABLE 5: 신호 분해 (7.Full 기간)")
    print("=" * 160)

    print(f"  {'Config':<8} {'종목수':>6} {'매수':>6} {'강매수':>6} {'일매수':>6} {'매도':>6} {'차단':>6} {'차단률':>7} {'강매수%':>8}")
    print("  " + "-" * 75)

    for cfg in config_names:
        a = agg[cfg].get('7.Full', {})
        nt = a.get('n_tickers', 0)
        tb = a.get('total_buys', 0)
        ts = a.get('total_sells', 0)
        tbl = a.get('total_blocked', 0)
        tsb = a.get('total_strong_buys', 0)
        tnb = a.get('total_normal_buys', 0)
        block_rate = f"{tbl/(ts+tbl)*100:.1f}%" if (ts+tbl) > 0 else "N/A"
        strong_pct = f"{tsb/tb*100:.1f}%" if tb > 0 else "N/A"
        print(f"  {cfg:<8} {nt:>6} {tb:>6} {tsb:>6} {tnb:>6} {ts:>6} {tbl:>6} {block_rate:>7} {strong_pct:>8}")

    # ================================================================
    # TABLE 6: Ranking
    # ================================================================
    print("\n\n" + "=" * 160)
    print("  TABLE 6: 종합 순위 (7.Full 기간 기준)")
    print("=" * 160)

    ranking = []
    for cfg in config_names:
        a_full = agg[cfg].get('7.Full', {})
        avg_ret = a_full.get('avg_ann', np.nan)
        avg_mdd = a_full.get('avg_mdd', np.nan)
        avg_final = a_full.get('avg_final', np.nan)
        n_t = a_full.get('n_tickers', 0)

        # Risk-adjusted: return / MDD ratio
        risk_adj = abs(avg_ret / avg_mdd) if not np.isnan(avg_mdd) and avg_mdd != 0 else 0

        # Avg across 1-6 periods
        p16_rets = [agg[cfg][pn]['avg_ann'] for pn in period_names[:-1]
                    if not np.isnan(agg[cfg][pn]['avg_ann'])]
        avg_p16 = np.mean(p16_rets) if p16_rets else np.nan

        ranking.append({
            'config': cfg,
            'avg_ann': avg_ret,
            'avg_p16': avg_p16,
            'avg_mdd': avg_mdd,
            'avg_final': avg_final,
            'risk_adj': risk_adj,
            'n_tickers': n_t,
        })

    ranking.sort(key=lambda x: x['avg_ann'] if not np.isnan(x['avg_ann']) else -999, reverse=True)

    c14_ret = agg['C14']['7.Full']['avg_ann']
    print(f"\n  {'순위':>4} {'Config':<8} {'연환산%':>8} {'Avg1-6%':>8} {'MDD%':>7} {'수익/MDD':>9} {'최종포트$':>10} {'종목수':>6} {'vs C14':>8}")
    print("  " + "-" * 78)
    for rank, r in enumerate(ranking, 1):
        diff = r['avg_ann'] - c14_ret if not np.isnan(r['avg_ann']) and not np.isnan(c14_ret) else np.nan
        diff_str = f"{diff:+.2f}" if not np.isnan(diff) else "N/A"
        marker = " <-- BEST" if rank == 1 else (" <-- C14" if r['config'] == 'C14' else "")
        print(f"  {rank:>4} {r['config']:<8} {r['avg_ann']:>8.2f} {r['avg_p16']:>8.2f} {r['avg_mdd']:>7.1f} {r['risk_adj']:>9.4f} {r['avg_final']:>10.0f} {r['n_tickers']:>6} {diff_str:>8}{marker}")

    # ================================================================
    # TABLE 7: TOP 5 detailed
    # ================================================================
    print("\n\n" + "=" * 160)
    print("  TABLE 7: TOP 5 구성 상세 분석")
    print("=" * 160)

    top5 = ranking[:5]
    for rank, r in enumerate(top5, 1):
        cfg = r['config']
        overrides = CONFIGS[cfg]
        params = get_params(overrides)

        # Describe changes
        changes = []
        if params['strong_buy_th'] != 0.15:
            changes.append(f"강매수TH={params['strong_buy_th']}")
        if params['cleanup_watchlist']:
            changes.append("워치정리(-PGY/BA/INTC)")
        if params['sell_normal_pct'] != 0.05:
            changes.append(f"매도={params['sell_normal_pct']*100:.0f}%/{params['sell_strong_pct']*100:.0f}%")
        if params['buy_normal_pct'] != 0.30:
            changes.append(f"매수={params['buy_normal_pct']*100:.0f}%/{params['buy_strong_pct']*100:.0f}%")
        if params['late_sell_drop_th'] != 0.05:
            changes.append(f"차단TH={params['late_sell_drop_th']*100:.0f}%")
        if params['three_tier']:
            changes.append("3단계신호")
        if not changes:
            changes.append("C14 베이스라인")

        desc = ", ".join(changes)
        diff = r['avg_ann'] - c14_ret if not np.isnan(r['avg_ann']) and not np.isnan(c14_ret) else 0

        print(f"\n  #{rank}: {cfg} [{desc}]")
        print(f"  전체기간: 연환산={r['avg_ann']:.2f}%, MDD=-{r['avg_mdd']:.1f}%, "
              f"수익/MDD={r['risk_adj']:.4f}, vs C14: {diff:+.2f}%p")
        print(f"  {'기간':<15} {'연환산%':>8} {'MDD%':>7} {'최종포트$':>10} {'매수':>5} {'매도':>5} {'차단':>5} {'vs C14':>8}")
        print(f"  {'-'*65}")
        for pname in period_names:
            a = agg[cfg][pname]
            base_a = agg['C14'][pname]
            d = a['avg_ann'] - base_a['avg_ann'] if not np.isnan(a['avg_ann']) and not np.isnan(base_a['avg_ann']) else np.nan
            d_str = f"{d:+.2f}" if not np.isnan(d) else "N/A"
            print(f"  {pname:<15} {a['avg_ann']:>8.2f} {a['avg_mdd']:>7.1f} {a['avg_final']:>10.0f} {a['total_buys']:>5} {a['total_sells']:>5} {a['total_blocked']:>5} {d_str:>8}")

    # ================================================================
    # EXECUTIVE SUMMARY
    # ================================================================
    print("\n\n" + "=" * 160)
    print("  종합 분석 (EXECUTIVE SUMMARY)")
    print("=" * 160)

    best = ranking[0]
    c14_rank = next(i+1 for i, r in enumerate(ranking) if r['config'] == 'C14')

    print(f"\n  테스트 구성: {len(config_names)}개 (개별 {len([c for c in config_names if c.startswith('I')])}개 + 조합 {len([c for c in config_names if c.startswith('C') and c != 'C14'])}개 + C14 베이스)")
    print(f"\n  BEST : {best['config']} (연환산 {best['avg_ann']:.2f}%, MDD -{best['avg_mdd']:.1f}%, 수익/MDD {best['risk_adj']:.4f})")
    print(f"  C14  : 순위 #{c14_rank}/{len(ranking)} (연환산 {c14_ret:.2f}%)")

    if best['config'] != 'C14':
        imp = best['avg_ann'] - c14_ret
        print(f"  개선폭: {imp:+.2f}%p 연환산")

    print(f"\n  개별 개선안 효과 (7.Full 기간):")
    indiv_names = {
        'I1': 'STRONG_BUY_TH 0.25',
        'I2': 'STRONG_BUY_TH 0.30',
        'I3': '워치리스트 정리',
        'I4': '매도 8%/15%',
        'I5': '매수 40%/60%',
        'I6': 'LATE_SELL 7%',
        'I7': 'LATE_SELL 10%',
        'I8': '3단계 신호',
    }
    indiv_results = []
    for fi, fname in indiv_names.items():
        fi_ret = agg[fi]['7.Full']['avg_ann']
        fi_mdd = agg[fi]['7.Full']['avg_mdd']
        if not np.isnan(fi_ret) and not np.isnan(c14_ret):
            diff = fi_ret - c14_ret
            mdd_diff = fi_mdd - agg['C14']['7.Full']['avg_mdd']
            verdict = "BETTER" if diff > 0.05 else ("WORSE" if diff < -0.05 else "NEUTRAL")
            indiv_results.append((fi, fname, diff, mdd_diff, verdict))
            sign = '+' if diff >= 0 else ''
            mdd_sign = '+' if mdd_diff >= 0 else ''
            print(f"    {fi} ({fname}): 수익 {sign}{diff:.2f}%p, MDD {mdd_sign}{mdd_diff:.1f}%p [{verdict}]")

    # Best by risk-adjusted ratio
    print(f"\n  위험조정 수익(수익/MDD) TOP 3:")
    by_risk = sorted(ranking, key=lambda x: x['risk_adj'], reverse=True)
    for i, r in enumerate(by_risk[:3], 1):
        print(f"    #{i}: {r['config']} (수익/MDD={r['risk_adj']:.4f}, 연환산={r['avg_ann']:.2f}%, MDD=-{r['avg_mdd']:.1f}%)")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    t_start = time.time()

    print("=" * 100)
    print("  V4 C14 기반 개선안 종합 백테스트")
    print(f"  시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  구성: {len(CONFIGS)}개 (C14 + 개별 8개 + 조합 11개)")
    print("=" * 100)

    # Load & download
    tickers = load_tickers()
    print(f"\n종목: {len(tickers)}개")
    print(f"구성: {', '.join(CONFIGS.keys())}")

    print(f"\n데이터 로딩...")
    all_data = download_all_data(tickers, _cache_dir)
    print(f"로딩 완료: {len(all_data)}개 종목")

    # Run
    print(f"\n백테스트 시작 ({len(CONFIGS)} x {len(MARKET_PERIODS)} x {len(all_data)})")
    results = run_all_backtests(all_data, tickers)

    # Results
    print_results(results)

    elapsed = time.time() - t_start
    print(f"\n\n총 소요시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
    print(f"완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
