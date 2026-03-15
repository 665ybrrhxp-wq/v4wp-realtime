"""
V4 C25 기반 업그레이드 백테스트 v3
===================================
v2 대비 개선사항:
  1. 병렬 처리: V4 스코어를 종목당 1회만 계산 후 재사용 (140x 중복 제거)
     - ProcessPoolExecutor로 V4 precomputation 병렬화
  2. B3 익절 전략: 매수가 대비 +100% 도달 시 해당 로트의 50% 매도
     - 로트 기반 포지션 관리 (각 매수 = 독립 로트)
     - C25_B3: 1회성 익절
     - C25_B3r: 반복 익절 (트리거 후 새 기준가 설정)

구성:
  C25:       베이스라인 (I1+I3+I5)
  C25_B3:    +100% 익절 (1회성)
  C25_B3r:   +100% 익절 (반복)
  C25_A1:    매도 비율 상향 (8%/15%)
  C25_A1_B3: 매도상향 + 익절
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import multiprocessing

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

REMOVED_TICKERS = {'PGY', 'BA', 'INTC'}

# ============================================================
# CONFIGURATION
# ============================================================
C25_BASELINE = {
    'strong_buy_th': 0.25,
    'strong_sell_th': -0.25,
    'buy_strong_pct': 0.60,
    'buy_normal_pct': 0.40,
    'sell_strong_pct': 0.10,
    'sell_normal_pct': 0.05,
    'late_sell_drop_th': 0.05,
    'atr_quantile': 55,
    'cleanup_watchlist': True,
    # B3 profit-taking params
    'profit_take_enabled': False,
    'profit_take_threshold': 1.0,   # +100%
    'profit_take_sell_pct': 0.50,   # 50% of lot
    'profit_take_repeatable': False,
}

CONFIGS = {
    'C25':       {},
    'C25_B3':    {'profit_take_enabled': True},
    'C25_B3r':   {'profit_take_enabled': True, 'profit_take_repeatable': True},
    'C25_A1':    {'sell_normal_pct': 0.08, 'sell_strong_pct': 0.15},
    'C25_A1_B3': {'sell_normal_pct': 0.08, 'sell_strong_pct': 0.15,
                  'profit_take_enabled': True},
}


def get_params(cfg_overrides):
    """Merge config overrides with C25 baseline."""
    params = dict(C25_BASELINE)
    params.update(cfg_overrides)
    return params


# ============================================================
# PHASE 1: V4 PRECOMPUTATION (per ticker, done once)
# ============================================================

def precompute_ticker(ticker, df_full, atr_quantile=55):
    """
    V4 스코어 + 신호 + 필터를 종목당 1회만 계산.
    모든 config/period에서 재사용.

    Returns dict with:
        events_all: all raw signal events
        events_filtered: price-filtered events (before LATE_SELL_BLOCK)
        close: np.ndarray of close prices
        rolling_high_20: np.ndarray of 20-day rolling high
        index: DatetimeIndex
    """
    if len(df_full) < 100:
        return None

    v4_score = calc_v4_score(df_full, w=20)
    events_all = detect_signal_events(v4_score, th=0.15, cooldown=5)
    pf = build_price_filter(df_full, er_q=66, atr_q=atr_quantile, lookback=252)
    events_filtered = [e for e in events_all if pf(e['peak_idx'])]

    close = df_full['Close'].values.copy()
    rolling_high_20 = df_full['Close'].rolling(20, min_periods=1).max().values.copy()

    return {
        'events_filtered': events_filtered,
        'close': close,
        'rolling_high_20': rolling_high_20,
        'index': df_full.index,
    }


def _precompute_worker(args):
    """Worker function for ProcessPoolExecutor."""
    ticker, df_full, atr_q = args
    try:
        result = precompute_ticker(ticker, df_full, atr_q)
        return ticker, result
    except Exception as e:
        print(f"  PRECOMPUTE ERROR {ticker}: {e}")
        return ticker, None


# ============================================================
# PHASE 2: TRADING SIMULATION (uses precomputed data)
# ============================================================

def apply_late_sell_block(events_filtered, close, rolling_high_20, drop_th):
    """LATE_SELL_BLOCK 적용. drop_th는 config별로 다를 수 있음."""
    events_final = []
    n_blocked = 0
    for ev in events_filtered:
        if ev['type'] == 'top':
            pidx = ev['peak_idx']
            if pidx < len(close):
                price = close[pidx]
                rh = rolling_high_20[pidx]
                drop_pct = (rh - price) / rh if rh > 0 else 0
                if drop_pct > drop_th:
                    n_blocked += 1
                    continue
        events_final.append(ev)
    return events_final, n_blocked


def get_period_events(events_final, df_index, period_start, period_end):
    """기간 내 이벤트 필터링."""
    mask = (df_index >= period_start) & (df_index <= period_end)
    period_idx = np.where(mask)[0]
    if len(period_idx) < 10:
        return None, None, None

    start_loc = period_idx[0]
    end_loc = period_idx[-1]

    period_events = [
        e for e in events_final
        if start_loc <= e['peak_idx'] <= end_loc
    ]

    return period_events, start_loc, end_loc


def simulate_trading(period_events, close_arr, date_index, start_loc, end_loc, params):
    """
    로트 기반 트레이딩 시뮬레이션.
    B3 profit-taking: 각 매수를 독립 로트로 추적.
    """
    INITIAL_CASH = 1000.0
    MONTHLY_ADD = 100.0

    pt_enabled = params['profit_take_enabled']
    pt_threshold = params['profit_take_threshold']
    pt_sell_pct = params['profit_take_sell_pct']
    pt_repeatable = params['profit_take_repeatable']

    n = end_loc - start_loc + 1
    dates = date_index[start_loc:end_loc + 1]
    close_period = close_arr[start_loc:end_loc + 1]

    cash = INITIAL_CASH
    total_deposited = INITIAL_CASH

    # 로트 기반 포지션: [{buy_price, shares, triggered}]
    lots = []

    n_buys = 0
    n_sells = 0
    n_strong_buys = 0
    n_normal_buys = 0
    n_profit_takes = 0
    profit_take_proceeds = 0.0

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
        full_idx = start_loc + i
        date = dates[i]
        price = close_period[i]

        # Monthly DCA
        month_key = (date.year, date.month)
        if month_key not in months_added:
            if i > 0:
                cash += MONTHLY_ADD
                total_deposited += MONTHLY_ADD
            months_added.add(month_key)

        # B3 profit-taking check (before signal processing)
        if pt_enabled and price > 0:
            for lot in lots:
                if lot['shares'] <= 0.0001:
                    continue
                trigger_price = lot['buy_price'] * (1 + pt_threshold)
                if not pt_repeatable and lot['triggered']:
                    continue
                if price >= trigger_price:
                    sell_shares = lot['shares'] * pt_sell_pct
                    if sell_shares > 0.0001:
                        proceeds = sell_shares * price
                        lot['shares'] -= sell_shares
                        cash += proceeds
                        n_profit_takes += 1
                        profit_take_proceeds += proceeds
                        lot['triggered'] = True
                        if pt_repeatable:
                            # 반복 익절: 새 기준가 = 현재가 기준 +threshold
                            lot['buy_price'] = price

        # Check signals
        if full_idx in event_dict:
            for ev in event_dict[full_idx]:
                if ev['type'] == 'bottom':
                    pv = abs(ev['peak_val'])
                    if pv >= params['strong_buy_th']:
                        buy_pct = params['buy_strong_pct']
                        n_strong_buys += 1
                    else:
                        buy_pct = params['buy_normal_pct']
                        n_normal_buys += 1

                    buy_amount = cash * buy_pct
                    if buy_amount > 1.0 and price > 0:
                        new_shares = buy_amount / price
                        lots.append({
                            'buy_price': price,
                            'shares': new_shares,
                            'triggered': False,
                        })
                        cash -= buy_amount
                        n_buys += 1

                elif ev['type'] == 'top':
                    is_strong = ev['peak_val'] <= params['strong_sell_th']
                    sell_pct = params['sell_strong_pct'] if is_strong else params['sell_normal_pct']

                    # 전체 보유량의 sell_pct를 균등 매도
                    total_shares = sum(l['shares'] for l in lots)
                    sell_shares = total_shares * sell_pct
                    if sell_shares > 0.0001 and price > 0:
                        # 각 로트에서 비례 매도
                        remaining_sell = sell_shares
                        for lot in lots:
                            if remaining_sell <= 0.0001:
                                break
                            lot_sell = min(lot['shares'], lot['shares'] / total_shares * sell_shares)
                            lot['shares'] -= lot_sell
                            remaining_sell -= lot_sell
                        proceeds = sell_shares * price
                        cash += proceeds
                        n_sells += 1

        # 빈 로트 정리 (메모리 관리)
        lots = [l for l in lots if l['shares'] > 0.0001]

        # Portfolio value
        total_shares = sum(l['shares'] for l in lots)
        pv = cash + total_shares * price
        portfolio_values.append(pv)

        # MDD
        if pv > peak_portfolio:
            peak_portfolio = pv
        if peak_portfolio > 0:
            dd = (peak_portfolio - pv) / peak_portfolio
            if dd > max_drawdown:
                max_drawdown = dd

    final_portfolio = portfolio_values[-1] if portfolio_values else INITIAL_CASH
    total_shares = sum(l['shares'] for l in lots)

    # Annualized return
    n_days = (dates[-1] - dates[0]).days
    n_years = n_days / 365.25
    if n_years > 0.5 and total_deposited > 0:
        ann_return = ((final_portfolio / total_deposited) ** (1.0 / n_years) - 1) * 100
    else:
        ann_return = ((final_portfolio / total_deposited) - 1) * 100

    open_val = total_shares * close_period[-1] if len(close_period) > 0 else 0

    return {
        'final_portfolio': final_portfolio,
        'total_deposited': total_deposited,
        'ann_return': ann_return,
        'mdd': max_drawdown * 100,
        'n_buys': n_buys,
        'n_sells': n_sells,
        'n_strong_buys': n_strong_buys,
        'n_normal_buys': n_normal_buys,
        'n_profit_takes': n_profit_takes,
        'profit_take_proceeds': profit_take_proceeds,
        'cash_remaining': cash,
        'open_position_val': open_val,
        'profit_rate': (final_portfolio / total_deposited - 1) * 100,
    }


# ============================================================
# MAIN ORCHESTRATION
# ============================================================

def load_tickers():
    with open(str(_config_path), 'r', encoding='utf-8') as f:
        config = json.load(f)
    tickers = list(config['tickers'].keys())
    benchmarks = config.get('benchmarks', [])
    return tickers + [b for b in benchmarks if b not in tickers]


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


def precompute_all_parallel(all_data, n_workers=None):
    """모든 종목 V4 precomputation 병렬 실행."""
    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), len(all_data), 8)

    precomputed = {}
    args_list = [(ticker, df, 55) for ticker, df in all_data.items()]

    print(f"  V4 병렬 계산 시작 ({len(args_list)}종목, {n_workers}워커)...")
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_precompute_worker, args): args[0]
                   for args in args_list}
        for future in as_completed(futures):
            ticker, result = future.result()
            if result is not None:
                precomputed[ticker] = result

    elapsed = time.time() - t0
    print(f"  V4 계산 완료: {len(precomputed)}종목, {elapsed:.1f}초")
    return precomputed


def run_all_backtests(precomputed, all_tickers):
    """모든 config × period × ticker 시뮬레이션 실행."""
    results = {}
    total_configs = len(CONFIGS)

    for ci, (cfg_name, cfg_overrides) in enumerate(CONFIGS.items()):
        params = get_params(cfg_overrides)
        results[cfg_name] = {}

        use_tickers = [t for t in all_tickers if t not in REMOVED_TICKERS] \
            if params['cleanup_watchlist'] else all_tickers

        print(f"\n  [{ci+1}/{total_configs}] {cfg_name}: ", end='', flush=True)

        for period in MARKET_PERIODS:
            pname = period['name']
            pstart = period['start']
            pend = period['end']
            ticker_results = []

            for ticker in use_tickers:
                if ticker not in precomputed:
                    continue
                pc = precomputed[ticker]

                # Apply LATE_SELL_BLOCK (config-specific drop_th)
                events_final, n_blocked = apply_late_sell_block(
                    pc['events_filtered'], pc['close'],
                    pc['rolling_high_20'], params['late_sell_drop_th']
                )

                # Get period events
                result = get_period_events(events_final, pc['index'], pstart, pend)
                if result[0] is None:
                    continue
                period_events, start_loc, end_loc = result

                # Simulate
                metrics = simulate_trading(
                    period_events, pc['close'], pc['index'],
                    start_loc, end_loc, params
                )
                metrics['ticker'] = ticker
                metrics['n_blocked'] = n_blocked
                ticker_results.append(metrics)

            results[cfg_name][pname] = ticker_results

        # Summary
        full_results = results[cfg_name].get('7.Full', [])
        if full_results:
            avg_ret = np.mean([r['ann_return'] for r in full_results])
            avg_mdd = np.mean([r['mdd'] for r in full_results])
            n_pt = sum(r['n_profit_takes'] for r in full_results)
            pt_str = f", 익절={n_pt}회" if n_pt > 0 else ""
            print(f"{len(full_results)}종목, avg_ann={avg_ret:.2f}%, "
                  f"avg_mdd=-{avg_mdd:.1f}%{pt_str}")
        else:
            print("no data")

    return results


# ============================================================
# RESULTS OUTPUT
# ============================================================

def aggregate_period(ticker_results):
    if not ticker_results:
        return {k: np.nan for k in ['avg_ann', 'med_ann', 'std_ann', 'avg_mdd',
                                     'avg_final', 'n_tickers', 'total_buys',
                                     'total_sells', 'total_blocked',
                                     'total_strong_buys', 'total_normal_buys',
                                     'total_profit_takes', 'total_pt_proceeds']}
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
        'total_profit_takes': sum(r.get('n_profit_takes', 0) for r in ticker_results),
        'total_pt_proceeds': sum(r.get('profit_take_proceeds', 0) for r in ticker_results),
    }


def print_results(results):
    config_names = list(CONFIGS.keys())
    period_names = [p['name'] for p in MARKET_PERIODS]

    agg = {}
    for cfg in config_names:
        agg[cfg] = {}
        for pname in period_names:
            agg[cfg][pname] = aggregate_period(results[cfg].get(pname, []))

    cw = 10

    # ================================================================
    # TABLE 1: CAGR
    # ================================================================
    print("\n\n" + "=" * 120)
    print("  TABLE 1: 평균 연환산 수익률(%) — 행=기간, 열=구성")
    print("=" * 120)

    header = f"{'기간':<15}"
    for cfg in config_names:
        header += f" {cfg:>{cw}}"
    print(header)
    print("-" * len(header))

    for pname in period_names:
        row = f"{pname:<15}"
        for cfg in config_names:
            val = agg[cfg][pname]['avg_ann']
            row += f" {val:>{cw}.2f}" if not np.isnan(val) else f" {'N/A':>{cw}}"
        print(row)

    print("-" * len(header))
    row = f"{'Avg(1-6)':<15}"
    for cfg in config_names:
        vals = [agg[cfg][pn]['avg_ann'] for pn in period_names[:-1]
                if not np.isnan(agg[cfg][pn]['avg_ann'])]
        avg = np.mean(vals) if vals else np.nan
        row += f" {avg:>{cw}.2f}" if not np.isnan(avg) else f" {'N/A':>{cw}}"
    print(row)

    # ================================================================
    # TABLE 2: MDD
    # ================================================================
    print("\n\n" + "=" * 120)
    print("  TABLE 2: 평균 MDD(%) — 행=기간, 열=구성 (낮을수록 좋음)")
    print("=" * 120)

    header = f"{'기간':<15}"
    for cfg in config_names:
        header += f" {cfg:>{cw}}"
    print(header)
    print("-" * len(header))

    for pname in period_names:
        row = f"{pname:<15}"
        for cfg in config_names:
            val = agg[cfg][pname]['avg_mdd']
            row += f" {val:>{cw}.1f}" if not np.isnan(val) else f" {'N/A':>{cw}}"
        print(row)

    # ================================================================
    # TABLE 3: C25 대비 변화
    # ================================================================
    print("\n\n" + "=" * 120)
    print("  TABLE 3: C25 대비 수익률 변화(%p) — 양수=개선")
    print("=" * 120)

    non_base = [c for c in config_names if c != 'C25']
    header = f"{'기간':<15}"
    for cfg in non_base:
        header += f" {cfg:>{cw}}"
    print(header)
    print("-" * len(header))

    for pname in period_names:
        row = f"{pname:<15}"
        base_val = agg['C25'][pname]['avg_ann']
        for cfg in non_base:
            val = agg[cfg][pname]['avg_ann']
            if np.isnan(val) or np.isnan(base_val):
                row += f" {'N/A':>{cw}}"
            else:
                diff = val - base_val
                sign = '+' if diff >= 0 else ''
                row += f" {sign}{diff:>{cw-1}.2f}"
        print(row)

    print("-" * len(header))
    row = f"{'Avg(1-6)':<15}"
    for cfg in non_base:
        diffs = []
        for pn in period_names[:-1]:
            bv = agg['C25'][pn]['avg_ann']
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
    # TABLE 4: MDD 변화
    # ================================================================
    print("\n\n" + "=" * 120)
    print("  TABLE 4: C25 대비 MDD 변화(%p) — 음수=MDD 개선(감소)")
    print("=" * 120)

    header = f"{'기간':<15}"
    for cfg in non_base:
        header += f" {cfg:>{cw}}"
    print(header)
    print("-" * len(header))

    for pname in period_names:
        row = f"{pname:<15}"
        base_mdd = agg['C25'][pname]['avg_mdd']
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
    # TABLE 5: 신호 + 익절 분해
    # ================================================================
    print("\n\n" + "=" * 120)
    print("  TABLE 5: 신호 + 익절 분해 (7.Full 기간)")
    print("=" * 120)

    print(f"  {'Config':<11} {'종목':>4} {'매수':>5} {'강매수':>5} {'일매수':>5} "
          f"{'매도':>5} {'차단':>5} {'차단률':>6} {'익절':>5} {'익절수익$':>9}")
    print("  " + "-" * 80)

    for cfg in config_names:
        a = agg[cfg].get('7.Full', {})
        nt = a.get('n_tickers', 0)
        tb = a.get('total_buys', 0)
        ts = a.get('total_sells', 0)
        tbl = a.get('total_blocked', 0)
        tsb = a.get('total_strong_buys', 0)
        tnb = a.get('total_normal_buys', 0)
        tpt = a.get('total_profit_takes', 0)
        tpp = a.get('total_pt_proceeds', 0)
        block_rate = f"{tbl/(ts+tbl)*100:.1f}%" if (ts+tbl) > 0 else "N/A"
        print(f"  {cfg:<11} {nt:>4} {tb:>5} {tsb:>5} {tnb:>5} "
              f"{ts:>5} {tbl:>5} {block_rate:>6} {tpt:>5} {tpp:>9,.0f}")

    # ================================================================
    # TABLE 6: 종합 순위
    # ================================================================
    print("\n\n" + "=" * 120)
    print("  TABLE 6: 종합 순위 (7.Full 기간 기준)")
    print("=" * 120)

    ranking = []
    for cfg in config_names:
        a_full = agg[cfg].get('7.Full', {})
        avg_ret = a_full.get('avg_ann', np.nan)
        avg_mdd = a_full.get('avg_mdd', np.nan)
        avg_final = a_full.get('avg_final', np.nan)
        n_t = a_full.get('n_tickers', 0)
        n_pt = a_full.get('total_profit_takes', 0)
        risk_adj = abs(avg_ret / avg_mdd) if not np.isnan(avg_mdd) and avg_mdd != 0 else 0

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
            'n_profit_takes': n_pt,
        })

    ranking.sort(key=lambda x: x['avg_ann'] if not np.isnan(x['avg_ann']) else -999,
                 reverse=True)

    c25_ret = agg['C25']['7.Full']['avg_ann']
    print(f"\n  {'순위':>4} {'Config':<11} {'연환산%':>8} {'Avg1-6%':>8} {'MDD%':>7} "
          f"{'수익/MDD':>9} {'최종포트$':>10} {'익절':>5} {'vs C25':>8}")
    print("  " + "-" * 85)
    for rank, r in enumerate(ranking, 1):
        diff = r['avg_ann'] - c25_ret if not np.isnan(r['avg_ann']) and not np.isnan(c25_ret) else np.nan
        diff_str = f"{diff:+.2f}" if not np.isnan(diff) else "N/A"
        marker = " <-- BEST" if rank == 1 else (" <-- BASE" if r['config'] == 'C25' else "")
        print(f"  {rank:>4} {r['config']:<11} {r['avg_ann']:>8.2f} {r['avg_p16']:>8.2f} "
              f"{r['avg_mdd']:>7.1f} {r['risk_adj']:>9.4f} {r['avg_final']:>10.0f} "
              f"{r['n_profit_takes']:>5} {diff_str:>8}{marker}")

    # ================================================================
    # TABLE 7: 각 구성 상세
    # ================================================================
    print("\n\n" + "=" * 120)
    print("  TABLE 7: 각 구성 상세 분석")
    print("=" * 120)

    for rank, r in enumerate(ranking, 1):
        cfg = r['config']
        overrides = CONFIGS[cfg]
        params = get_params(overrides)

        changes = []
        if params['sell_normal_pct'] != 0.05:
            changes.append(f"매도={params['sell_normal_pct']*100:.0f}%/{params['sell_strong_pct']*100:.0f}%")
        if params['profit_take_enabled']:
            pt_type = "반복" if params['profit_take_repeatable'] else "1회"
            changes.append(f"익절+{params['profit_take_threshold']*100:.0f}%→"
                          f"{params['profit_take_sell_pct']*100:.0f}%매도({pt_type})")
        if not changes:
            changes.append("C25 베이스라인")

        desc = ", ".join(changes)
        diff = r['avg_ann'] - c25_ret if not np.isnan(r['avg_ann']) else 0

        print(f"\n  #{rank}: {cfg} [{desc}]")
        print(f"  전체기간: 연환산={r['avg_ann']:.2f}%, MDD=-{r['avg_mdd']:.1f}%, "
              f"수익/MDD={r['risk_adj']:.4f}, vs C25: {diff:+.2f}%p")
        print(f"  {'기간':<15} {'연환산%':>8} {'MDD%':>7} {'최종포트$':>10} "
              f"{'매수':>5} {'매도':>5} {'익절':>5} {'vs C25':>8}")
        print(f"  {'-'*70}")
        for pname in period_names:
            a = agg[cfg][pname]
            base_a = agg['C25'][pname]
            d = a['avg_ann'] - base_a['avg_ann'] \
                if not np.isnan(a['avg_ann']) and not np.isnan(base_a['avg_ann']) else np.nan
            d_str = f"{d:+.2f}" if not np.isnan(d) else "N/A"
            pt = a.get('total_profit_takes', 0)
            print(f"  {pname:<15} {a['avg_ann']:>8.2f} {a['avg_mdd']:>7.1f} "
                  f"{a['avg_final']:>10.0f} {a['total_buys']:>5} {a['total_sells']:>5} "
                  f"{pt:>5} {d_str:>8}")

    # ================================================================
    # B3 익절 상세 분석
    # ================================================================
    b3_configs = [c for c in config_names if 'B3' in c]
    if b3_configs:
        print("\n\n" + "=" * 120)
        print("  TABLE 8: B3 익절 전략 상세 분석 (7.Full 기간)")
        print("=" * 120)

        for cfg in b3_configs:
            full_results = results[cfg].get('7.Full', [])
            if not full_results:
                continue

            pt_tickers = [r for r in full_results if r['n_profit_takes'] > 0]
            no_pt_tickers = [r for r in full_results if r['n_profit_takes'] == 0]

            print(f"\n  {cfg}:")
            print(f"    익절 발생 종목: {len(pt_tickers)}/{len(full_results)}")
            if pt_tickers:
                total_pt = sum(r['n_profit_takes'] for r in pt_tickers)
                total_pp = sum(r['profit_take_proceeds'] for r in pt_tickers)
                avg_pt_ret = np.mean([r['ann_return'] for r in pt_tickers])
                avg_npt_ret = np.mean([r['ann_return'] for r in no_pt_tickers]) if no_pt_tickers else np.nan

                print(f"    총 익절 횟수: {total_pt}회")
                print(f"    총 익절 수익: ${total_pp:,.0f}")
                print(f"    익절종목 평균수익률: {avg_pt_ret:.2f}%")
                if not np.isnan(avg_npt_ret):
                    print(f"    비익절종목 평균수익률: {avg_npt_ret:.2f}%")

                # Top profit-taking tickers
                pt_sorted = sorted(pt_tickers, key=lambda x: x['n_profit_takes'], reverse=True)
                print(f"\n    {'종목':<8} {'익절':>5} {'익절수익$':>10} {'연환산%':>8} {'최종포트$':>10}")
                print(f"    {'-'*50}")
                for r in pt_sorted[:10]:
                    print(f"    {r['ticker']:<8} {r['n_profit_takes']:>5} "
                          f"{r['profit_take_proceeds']:>10,.0f} "
                          f"{r['ann_return']:>8.2f} {r['final_portfolio']:>10,.0f}")

    # ================================================================
    # TABLE 9: 종목별 전체 데이터 (7.Full)
    # ================================================================
    print("\n\n" + "=" * 180)
    print("  TABLE 9: 종목별 전체 데이터 (7.Full 기간)")
    print("=" * 180)

    # 9-A: 종목별 연환산 수익률
    print("\n  [9-A] 종목별 연환산 수익률(%)")
    print("  " + "-" * 140)
    hdr = f"  {'종목':<8}"
    for cfg in config_names:
        hdr += f" {cfg:>10}"
    hdr += f" {'B3효과':>8} {'B3r효과':>8} {'A1효과':>8}"
    print(hdr)
    print("  " + "-" * 140)

    # Build per-ticker lookup: cfg -> ticker -> metrics
    ticker_data = {}
    for cfg in config_names:
        for r in results[cfg].get('7.Full', []):
            ticker_data.setdefault(r['ticker'], {})[cfg] = r

    # Collect all tickers that appear, preserving order
    all_tickers_in_full = []
    for r in results['C25'].get('7.Full', []):
        if r['ticker'] not in all_tickers_in_full:
            all_tickers_in_full.append(r['ticker'])

    for ticker in all_tickers_in_full:
        td = ticker_data.get(ticker, {})
        row = f"  {ticker:<8}"
        c25_val = td.get('C25', {}).get('ann_return', np.nan)
        for cfg in config_names:
            val = td.get(cfg, {}).get('ann_return', np.nan)
            row += f" {val:>10.2f}" if not np.isnan(val) else f" {'N/A':>10}"
        # 효과 계산
        b3_val = td.get('C25_B3', {}).get('ann_return', np.nan)
        b3r_val = td.get('C25_B3r', {}).get('ann_return', np.nan)
        a1_val = td.get('C25_A1', {}).get('ann_return', np.nan)
        b3_eff = b3_val - c25_val if not np.isnan(b3_val) and not np.isnan(c25_val) else np.nan
        b3r_eff = b3r_val - c25_val if not np.isnan(b3r_val) and not np.isnan(c25_val) else np.nan
        a1_eff = a1_val - c25_val if not np.isnan(a1_val) and not np.isnan(c25_val) else np.nan
        row += f" {b3_eff:>+8.2f}" if not np.isnan(b3_eff) else f" {'N/A':>8}"
        row += f" {b3r_eff:>+8.2f}" if not np.isnan(b3r_eff) else f" {'N/A':>8}"
        row += f" {a1_eff:>+8.2f}" if not np.isnan(a1_eff) else f" {'N/A':>8}"
        print(row)

    # 평균 행
    print("  " + "-" * 140)
    row = f"  {'평균':<8}"
    for cfg in config_names:
        vals = [td.get(cfg, {}).get('ann_return', np.nan)
                for td in ticker_data.values()]
        vals = [v for v in vals if not np.isnan(v)]
        avg = np.mean(vals) if vals else np.nan
        row += f" {avg:>10.2f}" if not np.isnan(avg) else f" {'N/A':>10}"
    print(row)

    # 9-B: 종목별 MDD
    print(f"\n\n  [9-B] 종목별 MDD(%)")
    print("  " + "-" * 140)
    hdr = f"  {'종목':<8}"
    for cfg in config_names:
        hdr += f" {cfg:>10}"
    hdr += f" {'B3효과':>8} {'B3r효과':>8}"
    print(hdr)
    print("  " + "-" * 140)

    for ticker in all_tickers_in_full:
        td = ticker_data.get(ticker, {})
        row = f"  {ticker:<8}"
        c25_mdd = td.get('C25', {}).get('mdd', np.nan)
        for cfg in config_names:
            val = td.get(cfg, {}).get('mdd', np.nan)
            row += f" {val:>10.1f}" if not np.isnan(val) else f" {'N/A':>10}"
        b3_mdd = td.get('C25_B3', {}).get('mdd', np.nan)
        b3r_mdd = td.get('C25_B3r', {}).get('mdd', np.nan)
        b3_eff = b3_mdd - c25_mdd if not np.isnan(b3_mdd) and not np.isnan(c25_mdd) else np.nan
        b3r_eff = b3r_mdd - c25_mdd if not np.isnan(b3r_mdd) and not np.isnan(c25_mdd) else np.nan
        row += f" {b3_eff:>+8.1f}" if not np.isnan(b3_eff) else f" {'N/A':>8}"
        row += f" {b3r_eff:>+8.1f}" if not np.isnan(b3r_eff) else f" {'N/A':>8}"
        print(row)

    print("  " + "-" * 140)
    row = f"  {'평균':<8}"
    for cfg in config_names:
        vals = [td.get(cfg, {}).get('mdd', np.nan) for td in ticker_data.values()]
        vals = [v for v in vals if not np.isnan(v)]
        avg = np.mean(vals) if vals else np.nan
        row += f" {avg:>10.1f}" if not np.isnan(avg) else f" {'N/A':>10}"
    print(row)

    # 9-C: 종목별 종합 (연환산 수익률 + 포트폴리오 + 매수/매도/익절)
    print(f"\n\n  [9-C] 종목별 종합 데이터 (7.Full)")
    print("  " + "-" * 200)
    hdr = (f"  {'종목':<8}"
           f" {'C25%':>7} {'B3%':>7} {'B3r%':>7} {'A1%':>7} {'A1B3%':>7}"
           f" {'C25_포트$':>10} {'B3_포트$':>10} {'B3r_포트$':>10}"
           f" {'A1_포트$':>10} {'A1B3_포트$':>10}"
           f" {'매수':>5} {'매도':>5} {'차단':>5}"
           f" {'B3익절':>6} {'B3r익절':>7} {'B3수익$':>10} {'B3r수익$':>10}"
           f" {'수익/MDD':>8}")
    print(hdr)
    print("  " + "-" * 200)

    for ticker in all_tickers_in_full:
        td = ticker_data.get(ticker, {})
        c25 = td.get('C25', {})
        b3 = td.get('C25_B3', {})
        b3r = td.get('C25_B3r', {})
        a1 = td.get('C25_A1', {})
        a1b3 = td.get('C25_A1_B3', {})

        c25_ret = c25.get('ann_return', 0)
        b3_ret = b3.get('ann_return', 0)
        b3r_ret = b3r.get('ann_return', 0)
        a1_ret = a1.get('ann_return', 0)
        a1b3_ret = a1b3.get('ann_return', 0)

        c25_fp = c25.get('final_portfolio', 0)
        b3_fp = b3.get('final_portfolio', 0)
        b3r_fp = b3r.get('final_portfolio', 0)
        a1_fp = a1.get('final_portfolio', 0)
        a1b3_fp = a1b3.get('final_portfolio', 0)

        n_buys = c25.get('n_buys', 0)
        n_sells = c25.get('n_sells', 0)
        n_blocked = c25.get('n_blocked', 0)
        b3_pt = b3.get('n_profit_takes', 0)
        b3r_pt = b3r.get('n_profit_takes', 0)
        b3_pp = b3.get('profit_take_proceeds', 0)
        b3r_pp = b3r.get('profit_take_proceeds', 0)

        c25_mdd = c25.get('mdd', 1)
        risk_adj = abs(c25_ret / c25_mdd) if c25_mdd > 0 else 0

        print(f"  {ticker:<8}"
              f" {c25_ret:>7.2f} {b3_ret:>7.2f} {b3r_ret:>7.2f} {a1_ret:>7.2f} {a1b3_ret:>7.2f}"
              f" {c25_fp:>10,.0f} {b3_fp:>10,.0f} {b3r_fp:>10,.0f}"
              f" {a1_fp:>10,.0f} {a1b3_fp:>10,.0f}"
              f" {n_buys:>5} {n_sells:>5} {n_blocked:>5}"
              f" {b3_pt:>6} {b3r_pt:>7} {b3_pp:>10,.0f} {b3r_pp:>10,.0f}"
              f" {risk_adj:>8.4f}")

    # 평균/합계
    print("  " + "-" * 200)
    avg_row = f"  {'평균':<8}"
    for cfg in config_names:
        vals = [td.get(cfg, {}).get('ann_return', np.nan) for td in ticker_data.values()]
        vals = [v for v in vals if not np.isnan(v)]
        avg = np.mean(vals) if vals else 0
        avg_row += f" {avg:>7.2f}"
    totals = {cfg: {} for cfg in config_names}
    for cfg in config_names:
        full_res = results[cfg].get('7.Full', [])
        totals[cfg] = {
            'fp': sum(r['final_portfolio'] for r in full_res),
            'buys': sum(r['n_buys'] for r in full_res),
            'sells': sum(r['n_sells'] for r in full_res),
            'blocked': sum(r.get('n_blocked', 0) for r in full_res),
            'pt': sum(r.get('n_profit_takes', 0) for r in full_res),
            'pp': sum(r.get('profit_take_proceeds', 0) for r in full_res),
        }
    for cfg in config_names:
        avg_row += f" {totals[cfg]['fp']:>10,.0f}"
    avg_row += (f" {totals['C25']['buys']:>5} {totals['C25']['sells']:>5}"
                f" {totals['C25']['blocked']:>5}"
                f" {totals['C25_B3']['pt']:>6} {totals['C25_B3r']['pt']:>7}"
                f" {totals['C25_B3']['pp']:>10,.0f} {totals['C25_B3r']['pp']:>10,.0f}")
    print(avg_row)

    # 9-D: 종목별 수익/MDD 순위
    print(f"\n\n  [9-D] 종목별 C25 수익/MDD 순위 (위험조정 수익 기준)")
    print("  " + "-" * 100)
    print(f"  {'순위':>4} {'종목':<8} {'연환산%':>8} {'MDD%':>7} {'수익/MDD':>9} "
          f"{'최종포트$':>10} {'B3효과%p':>8} {'B3r효과%p':>9}")
    print("  " + "-" * 100)

    ticker_ranking = []
    for ticker in all_tickers_in_full:
        td = ticker_data.get(ticker, {})
        c25 = td.get('C25', {})
        ret = c25.get('ann_return', np.nan)
        mdd = c25.get('mdd', np.nan)
        fp = c25.get('final_portfolio', 0)
        ra = abs(ret / mdd) if not np.isnan(mdd) and mdd > 0 else 0
        b3_eff = td.get('C25_B3', {}).get('ann_return', np.nan) - ret \
            if not np.isnan(td.get('C25_B3', {}).get('ann_return', np.nan)) else np.nan
        b3r_eff = td.get('C25_B3r', {}).get('ann_return', np.nan) - ret \
            if not np.isnan(td.get('C25_B3r', {}).get('ann_return', np.nan)) else np.nan
        ticker_ranking.append({
            'ticker': ticker, 'ret': ret, 'mdd': mdd,
            'risk_adj': ra, 'fp': fp, 'b3_eff': b3_eff, 'b3r_eff': b3r_eff,
        })

    ticker_ranking.sort(key=lambda x: x['risk_adj'], reverse=True)
    for rank, r in enumerate(ticker_ranking, 1):
        b3_str = f"{r['b3_eff']:>+8.2f}" if not np.isnan(r['b3_eff']) else f"{'N/A':>8}"
        b3r_str = f"{r['b3r_eff']:>+9.2f}" if not np.isnan(r['b3r_eff']) else f"{'N/A':>9}"
        print(f"  {rank:>4} {r['ticker']:<8} {r['ret']:>8.2f} {r['mdd']:>7.1f} "
              f"{r['risk_adj']:>9.4f} {r['fp']:>10,.0f} {b3_str} {b3r_str}")

    # ================================================================
    # EXECUTIVE SUMMARY
    # ================================================================
    print("\n\n" + "=" * 120)
    print("  종합 분석 (EXECUTIVE SUMMARY)")
    print("=" * 120)

    best = ranking[0]
    c25_rank = next(i+1 for i, r in enumerate(ranking) if r['config'] == 'C25')

    print(f"\n  테스트 구성: {len(config_names)}개")
    print(f"  BEST : {best['config']} (연환산 {best['avg_ann']:.2f}%, "
          f"MDD -{best['avg_mdd']:.1f}%, 수익/MDD {best['risk_adj']:.4f})")
    print(f"  C25  : 순위 #{c25_rank}/{len(ranking)} "
          f"(연환산 {c25_ret:.2f}%)")

    if best['config'] != 'C25':
        imp = best['avg_ann'] - c25_ret
        print(f"  개선폭: {imp:+.2f}%p 연환산")

    # B3 효과 요약
    if 'C25_B3' in config_names:
        b3_ret = agg['C25_B3']['7.Full']['avg_ann']
        b3_mdd = agg['C25_B3']['7.Full']['avg_mdd']
        b3_diff = b3_ret - c25_ret
        print(f"\n  B3 익절(1회) 효과: 수익 {b3_diff:+.2f}%p, "
              f"MDD {b3_mdd - agg['C25']['7.Full']['avg_mdd']:+.1f}%p")

    if 'C25_B3r' in config_names:
        b3r_ret = agg['C25_B3r']['7.Full']['avg_ann']
        b3r_mdd = agg['C25_B3r']['7.Full']['avg_mdd']
        b3r_diff = b3r_ret - c25_ret
        print(f"  B3r익절(반복) 효과: 수익 {b3r_diff:+.2f}%p, "
              f"MDD {b3r_mdd - agg['C25']['7.Full']['avg_mdd']:+.1f}%p")

    # Risk-adjusted ranking
    by_risk = sorted(ranking, key=lambda x: x['risk_adj'], reverse=True)
    print(f"\n  위험조정 수익(수익/MDD) 순위:")
    for i, r in enumerate(by_risk, 1):
        marker = " *" if r['config'] == best['config'] else ""
        print(f"    #{i}: {r['config']} (수익/MDD={r['risk_adj']:.4f}, "
              f"연환산={r['avg_ann']:.2f}%, MDD=-{r['avg_mdd']:.1f}%){marker}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Windows 호환
    t_start = time.time()

    print("=" * 100)
    print("  V4 C25 기반 업그레이드 백테스트 v3")
    print(f"  시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  구성: {len(CONFIGS)}개 — {', '.join(CONFIGS.keys())}")
    print(f"  주요 변경: 병렬 V4 계산 + B3 익절 전략")
    print("=" * 100)

    # 1. Load tickers
    tickers = load_tickers()
    print(f"\n종목: {len(tickers)}개")

    # 2. Download data (sequential, cached)
    print(f"\n데이터 로딩...")
    t_dl = time.time()
    all_data = download_all_data(tickers, _cache_dir)
    print(f"로딩 완료: {len(all_data)}종목 ({time.time()-t_dl:.1f}초)")

    # 3. Parallel V4 precomputation
    print(f"\nV4 Precomputation (병렬)...")
    precomputed = precompute_all_parallel(all_data)

    # 4. Run all simulations (fast after precomputation)
    print(f"\n트레이딩 시뮬레이션 ({len(CONFIGS)} configs x "
          f"{len(MARKET_PERIODS)} periods x {len(precomputed)} tickers)")
    t_sim = time.time()
    results = run_all_backtests(precomputed, tickers)
    print(f"\n시뮬레이션 완료 ({time.time()-t_sim:.1f}초)")

    # 5. Results
    print_results(results)

    elapsed = time.time() - t_start
    print(f"\n\n총 소요시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
    print(f"완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
