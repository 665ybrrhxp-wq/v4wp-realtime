"""
V4 F/D/C 가중치 + ActivityMultiplier Sweep
==========================================
calc_v4_score의 dire = wF*Force + wD*Div + wC*Conc 가중치를 변경하며
C25 전략 성능을 비교.

ActivityMultiplier 변형도 함께 테스트.
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
    detect_signal_events,
    build_price_filter,
    calc_pv_force_macd,
    calc_pv_divergence,
    calc_pv_concordance,
)

# ============================================================
# CUSTOM V4 SCORE WITH CONFIGURABLE WEIGHTS
# ============================================================

def calc_v4_score_weighted(df, w=20, wF=0.30, wD=0.40, wC=0.30,
                           act_map=None):
    """가중치 파라미터화된 V4 스코어 계산.

    Args:
        wF, wD, wC: Force, Divergence, Concordance 가중치
        act_map: ActivityMultiplier dict {0: x, 1: x, 2: x, 3: x}
    """
    if act_map is None:
        act_map = {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.2}

    n = len(df)
    pv_div = calc_pv_divergence(df, w)
    pv_conc = calc_pv_concordance(df, w)
    pv_fh = calc_pv_force_macd(df)
    scores = np.zeros(n)

    for i in range(max(60, w), n):
        s_div = np.clip(pv_div.iloc[i] / 3, -1, 1)
        s_conc = pv_conc.iloc[i]
        fhr_std = pv_fh.iloc[max(0, i - w):i].std() + 1e-10
        s_force = np.clip(pv_fh.iloc[i] / (2 * fhr_std), -1, 1)

        dire = wF * s_force + wD * s_div + wC * s_conc
        act = sum([abs(s_div) > 0.1, abs(s_conc) > 0.1, abs(s_force) > 0.1])
        scores[i] = dire * act_map.get(act, 1.0)

    return pd.Series(scores, index=df.index, name='V4')


# ============================================================
# WEIGHT CONFIGS
# ============================================================

WEIGHT_CONFIGS = {}

# 기준점
WEIGHT_CONFIGS['BASE'] = {'wF': 0.30, 'wD': 0.40, 'wC': 0.30, 'desc': '현재 F30/D40/C30'}

# Force 강화 세밀 sweep: F=0.35~0.55, D/C 다양 조합 (합=1.0)
_sweep = [
    # F=0.35 계열
    (0.35, 0.40, 0.25), (0.35, 0.35, 0.30), (0.35, 0.30, 0.35),
    # F=0.40 계열
    (0.40, 0.40, 0.20), (0.40, 0.35, 0.25), (0.40, 0.30, 0.30),
    (0.40, 0.25, 0.35),
    # F=0.45 계열
    (0.45, 0.35, 0.20), (0.45, 0.30, 0.25), (0.45, 0.25, 0.30),
    (0.45, 0.20, 0.35),
    # F=0.50 계열
    (0.50, 0.30, 0.20), (0.50, 0.25, 0.25), (0.50, 0.20, 0.30),
    # F=0.55 계열
    (0.55, 0.25, 0.20), (0.55, 0.20, 0.25),
]

for wF, wD, wC in _sweep:
    name = f'F{int(wF*100)}D{int(wD*100)}C{int(wC*100)}'
    WEIGHT_CONFIGS[name] = {
        'wF': wF, 'wD': wD, 'wC': wC,
        'desc': f'F{int(wF*100)}/D{int(wD*100)}/C{int(wC*100)}',
    }

# C25 trading params (fixed)
C25_PARAMS = {
    'strong_buy_th': 0.25,
    'strong_sell_th': -0.25,
    'buy_strong_pct': 0.60,
    'buy_normal_pct': 0.40,
    'sell_strong_pct': 0.10,
    'sell_normal_pct': 0.05,
    'late_sell_drop_th': 0.05,
}

REMOVED_TICKERS = {'PGY', 'BA', 'INTC'}

MARKET_PERIODS = [
    {'name': '1.Lehman',    'start': '2007-01-01', 'end': '2009-12-31'},
    {'name': '2.Recovery',  'start': '2009-03-01', 'end': '2015-12-31'},
    {'name': '3.RateShock', 'start': '2018-01-01', 'end': '2019-12-31'},
    {'name': '4.Covid',     'start': '2020-01-01', 'end': '2021-12-31'},
    {'name': '5.InflDown',  'start': '2022-01-01', 'end': '2023-06-30'},
    {'name': '6.AI+Tariff', 'start': '2023-07-01', 'end': '2026-03-31'},
    {'name': '7.Full',      'start': '2007-01-01', 'end': '2026-03-31'},
]


# ============================================================
# PRECOMPUTE PER WEIGHT CONFIG
# ============================================================

def precompute_ticker_weighted(df_full, wF, wD, wC, act_map=None):
    """종목 1개에 대해 특정 가중치로 V4 계산."""
    if len(df_full) < 100:
        return None

    v4_score = calc_v4_score_weighted(df_full, w=20, wF=wF, wD=wD, wC=wC,
                                       act_map=act_map)
    events_all = detect_signal_events(v4_score, th=0.15, cooldown=5)
    pf = build_price_filter(df_full, er_q=66, atr_q=55, lookback=252)
    events_filtered = [e for e in events_all if pf(e['peak_idx'])]

    close = df_full['Close'].values.copy()
    rolling_high_20 = df_full['Close'].rolling(20, min_periods=1).max().values.copy()

    return {
        'events_filtered': events_filtered,
        'close': close,
        'rolling_high_20': rolling_high_20,
        'index': df_full.index,
    }


def _worker(args):
    """Parallel worker."""
    ticker, df, wF, wD, wC, act_map, cfg_name = args
    try:
        result = precompute_ticker_weighted(df, wF, wD, wC, act_map)
        return cfg_name, ticker, result
    except Exception as e:
        return cfg_name, ticker, None


# ============================================================
# TRADING SIMULATION (same as v3 but without profit-taking)
# ============================================================

def simulate_trading(period_events, close_arr, date_index, start_loc, end_loc, params):
    INITIAL_CASH = 1000.0
    MONTHLY_ADD = 100.0

    n = end_loc - start_loc + 1
    dates = date_index[start_loc:end_loc + 1]
    close_period = close_arr[start_loc:end_loc + 1]

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

        month_key = (date.year, date.month)
        if month_key not in months_added:
            if i > 0:
                cash += MONTHLY_ADD
                total_deposited += MONTHLY_ADD
            months_added.add(month_key)

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

        pv = cash + shares * price
        portfolio_values.append(pv)
        if pv > peak_portfolio:
            peak_portfolio = pv
        if peak_portfolio > 0:
            dd = (peak_portfolio - pv) / peak_portfolio
            if dd > max_drawdown:
                max_drawdown = dd

    final_portfolio = portfolio_values[-1] if portfolio_values else INITIAL_CASH

    n_days = (dates[-1] - dates[0]).days
    n_years = n_days / 365.25
    if n_years > 0.5 and total_deposited > 0:
        ann_return = ((final_portfolio / total_deposited) ** (1.0 / n_years) - 1) * 100
    else:
        ann_return = ((final_portfolio / total_deposited) - 1) * 100

    return {
        'final_portfolio': final_portfolio,
        'total_deposited': total_deposited,
        'ann_return': ann_return,
        'mdd': max_drawdown * 100,
        'n_buys': n_buys,
        'n_sells': n_sells,
        'n_strong_buys': n_strong_buys,
        'n_normal_buys': n_normal_buys,
    }


# ============================================================
# MAIN
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
            df = download_data(ticker, start='2000-01-01', end='2026-12-31',
                               cache_dir=cache_dir)
            if len(df) >= 100:
                data[ticker] = df
        except Exception as e:
            print(f"  ERROR {ticker}: {e}")
    return data


def run_sweep(all_data, all_tickers):
    """모든 가중치 config에 대해 병렬 precompute + 시뮬레이션."""
    n_workers = min(multiprocessing.cpu_count(), 8)
    use_tickers = [t for t in all_tickers if t not in REMOVED_TICKERS]

    # Phase 1: 모든 (config × ticker) 조합을 병렬 precompute
    print(f"\n  Phase 1: V4 병렬 계산 ({len(WEIGHT_CONFIGS)} configs × "
          f"{len(all_data)} tickers, {n_workers} workers)...")
    t0 = time.time()

    args_list = []
    for cfg_name, cfg in WEIGHT_CONFIGS.items():
        act_map = cfg.get('act_map', None)
        for ticker, df in all_data.items():
            args_list.append((ticker, df, cfg['wF'], cfg['wD'], cfg['wC'],
                              act_map, cfg_name))

    precomputed = {}  # {cfg_name: {ticker: result}}
    for cfg_name in WEIGHT_CONFIGS:
        precomputed[cfg_name] = {}

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_worker, args) for args in args_list]
        for future in as_completed(futures):
            cfg_name, ticker, result = future.result()
            if result is not None:
                precomputed[cfg_name][ticker] = result

    print(f"  V4 계산 완료: {time.time()-t0:.1f}초")

    # Phase 2: 시뮬레이션
    print(f"\n  Phase 2: 트레이딩 시뮬레이션...")
    t1 = time.time()

    results = {}
    for cfg_name in WEIGHT_CONFIGS:
        results[cfg_name] = {}
        pc = precomputed[cfg_name]

        for period in MARKET_PERIODS:
            pname = period['name']
            ticker_results = []

            for ticker in use_tickers:
                if ticker not in pc:
                    continue
                tc = pc[ticker]

                # LATE_SELL_BLOCK
                events_final = []
                n_blocked = 0
                drop_th = C25_PARAMS['late_sell_drop_th']
                for ev in tc['events_filtered']:
                    if ev['type'] == 'top':
                        pidx = ev['peak_idx']
                        if pidx < len(tc['close']):
                            price = tc['close'][pidx]
                            rh = tc['rolling_high_20'][pidx]
                            drop_pct = (rh - price) / rh if rh > 0 else 0
                            if drop_pct > drop_th:
                                n_blocked += 1
                                continue
                    events_final.append(ev)

                # Period filter
                mask = (tc['index'] >= period['start']) & (tc['index'] <= period['end'])
                period_idx = np.where(mask)[0]
                if len(period_idx) < 10:
                    continue
                start_loc = period_idx[0]
                end_loc = period_idx[-1]

                period_events = [e for e in events_final
                                 if start_loc <= e['peak_idx'] <= end_loc]

                metrics = simulate_trading(
                    period_events, tc['close'], tc['index'],
                    start_loc, end_loc, C25_PARAMS
                )
                metrics['ticker'] = ticker
                metrics['n_blocked'] = n_blocked
                ticker_results.append(metrics)

            results[cfg_name][pname] = ticker_results

    print(f"  시뮬레이션 완료: {time.time()-t1:.1f}초")
    return results


def print_results(results):
    cfg_names = list(WEIGHT_CONFIGS.keys())
    period_names = [p['name'] for p in MARKET_PERIODS]

    # Aggregate
    def agg(ticker_results):
        if not ticker_results:
            return {'avg_ann': np.nan, 'med_ann': np.nan, 'std_ann': np.nan,
                    'avg_mdd': np.nan, 'avg_final': np.nan, 'n_tickers': 0,
                    'total_buys': 0, 'total_sells': 0, 'total_blocked': 0,
                    'total_strong_buys': 0, 'total_normal_buys': 0}
        rets = [r['ann_return'] for r in ticker_results]
        return {
            'avg_ann': np.mean(rets), 'med_ann': np.median(rets),
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

    ag = {}
    for cfg in cfg_names:
        ag[cfg] = {}
        for pname in period_names:
            ag[cfg][pname] = agg(results[cfg].get(pname, []))

    cw = 10

    # ================================================================
    # TABLE 1: CAGR
    # ================================================================
    print("\n\n" + "=" * 180)
    print("  TABLE 1: 평균 연환산 수익률(%) — 가중치별 비교")
    print("=" * 180)

    # Split into weight configs and actmul configs
    w_cfgs = [c for c in cfg_names if c.startswith('W')]
    a_cfgs = [c for c in cfg_names if c.startswith('A') or (c.startswith('W') and 'A' in c[2:])]
    # Actually just show all together
    header = f"{'기간':<15}"
    for cfg in cfg_names:
        label = cfg[:cw]
        header += f" {label:>{cw}}"
    print(header)
    print("-" * len(header))

    for pname in period_names:
        row = f"{pname:<15}"
        for cfg in cfg_names:
            val = ag[cfg][pname]['avg_ann']
            row += f" {val:>{cw}.2f}" if not np.isnan(val) else f" {'N/A':>{cw}}"
        print(row)

    print("-" * len(header))
    row = f"{'Avg(1-6)':<15}"
    for cfg in cfg_names:
        vals = [ag[cfg][pn]['avg_ann'] for pn in period_names[:-1]
                if not np.isnan(ag[cfg][pn]['avg_ann'])]
        avg_val = np.mean(vals) if vals else np.nan
        row += f" {avg_val:>{cw}.2f}" if not np.isnan(avg_val) else f" {'N/A':>{cw}}"
    print(row)

    # ================================================================
    # TABLE 2: MDD
    # ================================================================
    print("\n\n" + "=" * 180)
    print("  TABLE 2: 평균 MDD(%) — 가중치별 비교")
    print("=" * 180)

    header = f"{'기간':<15}"
    for cfg in cfg_names:
        header += f" {cfg[:cw]:>{cw}}"
    print(header)
    print("-" * len(header))

    for pname in period_names:
        row = f"{pname:<15}"
        for cfg in cfg_names:
            val = ag[cfg][pname]['avg_mdd']
            row += f" {val:>{cw}.1f}" if not np.isnan(val) else f" {'N/A':>{cw}}"
        print(row)

    # ================================================================
    # TABLE 3: W0 대비 변화
    # ================================================================
    base = 'BASE'
    print("\n\n" + "=" * 180)
    print(f"  TABLE 3: {base} 대비 수익률 변화(%p)")
    print("=" * 180)

    non_base = [c for c in cfg_names if c != base]
    header = f"{'기간':<15}"
    for cfg in non_base:
        header += f" {cfg[:cw]:>{cw}}"
    print(header)
    print("-" * len(header))

    for pname in period_names:
        row = f"{pname:<15}"
        bv = ag[base][pname]['avg_ann']
        for cfg in non_base:
            val = ag[cfg][pname]['avg_ann']
            if np.isnan(val) or np.isnan(bv):
                row += f" {'N/A':>{cw}}"
            else:
                diff = val - bv
                sign = '+' if diff >= 0 else ''
                row += f" {sign}{diff:>{cw-1}.2f}"
        print(row)

    print("-" * len(header))
    row = f"{'Avg(1-6)':<15}"
    for cfg in non_base:
        diffs = []
        for pn in period_names[:-1]:
            bv2 = ag[base][pn]['avg_ann']
            cv2 = ag[cfg][pn]['avg_ann']
            if not np.isnan(bv2) and not np.isnan(cv2):
                diffs.append(cv2 - bv2)
        avg_d = np.mean(diffs) if diffs else np.nan
        if np.isnan(avg_d):
            row += f" {'N/A':>{cw}}"
        else:
            sign = '+' if avg_d >= 0 else ''
            row += f" {sign}{avg_d:>{cw-1}.2f}"
    print(row)

    # ================================================================
    # TABLE 4: 신호 분해
    # ================================================================
    print("\n\n" + "=" * 180)
    print("  TABLE 4: 신호 분해 (7.Full)")
    print("=" * 180)

    print(f"  {'Config':<14} {'설명':<25} {'종목':>4} {'매수':>5} {'강매수':>5} "
          f"{'일매수':>5} {'매도':>5} {'차단':>5} {'강매수%':>7}")
    print("  " + "-" * 90)

    for cfg in cfg_names:
        a = ag[cfg].get('7.Full', {})
        nt = a.get('n_tickers', 0)
        tb = a.get('total_buys', 0)
        ts = a.get('total_sells', 0)
        tbl = a.get('total_blocked', 0)
        tsb = a.get('total_strong_buys', 0)
        tnb = a.get('total_normal_buys', 0)
        sp = f"{tsb/tb*100:.1f}%" if tb > 0 else "N/A"
        desc = WEIGHT_CONFIGS[cfg]['desc']
        print(f"  {cfg:<14} {desc:<25} {nt:>4} {tb:>5} {tsb:>5} "
              f"{tnb:>5} {ts:>5} {tbl:>5} {sp:>7}")

    # ================================================================
    # TABLE 5: 종합 순위
    # ================================================================
    print("\n\n" + "=" * 180)
    print("  TABLE 5: 종합 순위 (7.Full 기준)")
    print("=" * 180)

    ranking = []
    for cfg in cfg_names:
        a_full = ag[cfg].get('7.Full', {})
        avg_ret = a_full.get('avg_ann', np.nan)
        avg_mdd = a_full.get('avg_mdd', np.nan)
        avg_final = a_full.get('avg_final', np.nan)
        n_t = a_full.get('n_tickers', 0)
        risk_adj = abs(avg_ret / avg_mdd) if not np.isnan(avg_mdd) and avg_mdd != 0 else 0

        p16_rets = [ag[cfg][pn]['avg_ann'] for pn in period_names[:-1]
                    if not np.isnan(ag[cfg][pn]['avg_ann'])]
        avg_p16 = np.mean(p16_rets) if p16_rets else np.nan

        ranking.append({
            'config': cfg, 'desc': WEIGHT_CONFIGS[cfg]['desc'],
            'avg_ann': avg_ret, 'avg_p16': avg_p16,
            'avg_mdd': avg_mdd, 'avg_final': avg_final,
            'risk_adj': risk_adj, 'n_tickers': n_t,
        })

    # Sort by ann_return
    ranking.sort(key=lambda x: x['avg_ann'] if not np.isnan(x['avg_ann']) else -999,
                 reverse=True)

    base_ret = ag[base]['7.Full']['avg_ann']

    print(f"\n  {'순위':>4} {'Config':<14} {'설명':<22} {'연환산%':>8} {'Avg1-6%':>8} "
          f"{'MDD%':>7} {'수익/MDD':>9} {'최종포트$':>10} {'vs W0':>8}")
    print("  " + "-" * 105)

    for rank, r in enumerate(ranking, 1):
        diff = r['avg_ann'] - base_ret if not np.isnan(r['avg_ann']) else np.nan
        diff_str = f"{diff:+.2f}" if not np.isnan(diff) else "N/A"
        marker = " <-- BEST" if rank == 1 else (" <-- 현재" if r['config'] == base else "")
        print(f"  {rank:>4} {r['config']:<14} {r['desc']:<22} {r['avg_ann']:>8.2f} "
              f"{r['avg_p16']:>8.2f} {r['avg_mdd']:>7.1f} {r['risk_adj']:>9.4f} "
              f"{r['avg_final']:>10.0f} {diff_str:>8}{marker}")

    # ================================================================
    # TABLE 6: 위험조정 수익 순위
    # ================================================================
    by_risk = sorted(ranking, key=lambda x: x['risk_adj'], reverse=True)
    print(f"\n\n  위험조정 수익(수익/MDD) 순위:")
    print("  " + "-" * 90)
    for i, r in enumerate(by_risk, 1):
        diff = r['avg_ann'] - base_ret if not np.isnan(r['avg_ann']) else 0
        marker = " *" if r['config'] == base else ""
        print(f"  #{i:>2} {r['config']:<14} {r['desc']:<22} "
              f"수익/MDD={r['risk_adj']:.4f} 연환산={r['avg_ann']:.2f}% "
              f"MDD=-{r['avg_mdd']:.1f}% vs W0={diff:+.2f}%p{marker}")

    # ================================================================
    # TABLE 7: 종목별 수익률 (7.Full, 상위/하위 비교)
    # ================================================================
    print("\n\n" + "=" * 180)
    print("  TABLE 7: 종목별 연환산 수익률 비교 (7.Full)")
    print("=" * 180)

    # Build per-ticker data
    ticker_data = {}
    for cfg in cfg_names:
        for r in results[cfg].get('7.Full', []):
            ticker_data.setdefault(r['ticker'], {})[cfg] = r

    all_tickers_sorted = sorted(ticker_data.keys(),
                                 key=lambda t: ticker_data[t].get(base, {}).get('ann_return', 0),
                                 reverse=True)

    header = f"  {'종목':<8}"
    for cfg in cfg_names:
        header += f" {cfg[:8]:>8}"
    header += f" {'최적Config':>14}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for ticker in all_tickers_sorted:
        td = ticker_data.get(ticker, {})
        row = f"  {ticker:<8}"
        best_cfg = base
        best_ret = -999
        for cfg in cfg_names:
            val = td.get(cfg, {}).get('ann_return', np.nan)
            row += f" {val:>8.2f}" if not np.isnan(val) else f" {'N/A':>8}"
            if not np.isnan(val) and val > best_ret:
                best_ret = val
                best_cfg = cfg
        row += f" {best_cfg:>14}"
        print(row)

    # ================================================================
    # EXECUTIVE SUMMARY
    # ================================================================
    print("\n\n" + "=" * 120)
    print("  EXECUTIVE SUMMARY")
    print("=" * 120)

    best = ranking[0]
    best_risk = by_risk[0]
    w0_rank = next(i+1 for i, r in enumerate(ranking) if r['config'] == base)
    w0_risk_rank = next(i+1 for i, r in enumerate(by_risk) if r['config'] == base)

    print(f"\n  테스트: {len(WEIGHT_CONFIGS)}개 가중치 조합")
    print(f"\n  [수익률 기준]")
    print(f"    BEST: {best['config']} ({best['desc']}) — "
          f"연환산 {best['avg_ann']:.2f}%, vs W0: {best['avg_ann']-base_ret:+.2f}%p")
    print(f"    현재(W0): 순위 #{w0_rank}/{len(ranking)} — 연환산 {base_ret:.2f}%")

    print(f"\n  [위험조정 기준]")
    print(f"    BEST: {best_risk['config']} ({best_risk['desc']}) — "
          f"수익/MDD {best_risk['risk_adj']:.4f}")
    print(f"    현재(W0): 순위 #{w0_risk_rank}/{len(ranking)} — "
          f"수익/MDD {ag[base]['7.Full']['avg_mdd']:.1f}")

    # 가중치 vs ActMul 효과 비교
    w_only = [r for r in ranking if r['config'].startswith('W') and 'A' not in r['config'][2:]]
    a_only = [r for r in ranking if r['config'].startswith('A')]
    combo = [r for r in ranking if 'A' in r['config'] and r['config'][0] == 'W' and len(r['config']) > 3]

    if w_only:
        best_w = max(w_only, key=lambda x: x['avg_ann'])
        print(f"\n  [가중치만 변경] 최적: {best_w['config']} ({best_w['desc']}) "
              f"→ {best_w['avg_ann']:.2f}% ({best_w['avg_ann']-base_ret:+.2f}%p)")
    if a_only:
        best_a = max(a_only, key=lambda x: x['avg_ann'])
        print(f"  [ActMul만 변경] 최적: {best_a['config']} ({best_a['desc']}) "
              f"→ {best_a['avg_ann']:.2f}% ({best_a['avg_ann']-base_ret:+.2f}%p)")
    if combo:
        best_c = max(combo, key=lambda x: x['avg_ann'])
        print(f"  [가중치+ActMul] 최적: {best_c['config']} ({best_c['desc']}) "
              f"→ {best_c['avg_ann']:.2f}% ({best_c['avg_ann']-base_ret:+.2f}%p)")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    multiprocessing.freeze_support()
    t_start = time.time()

    print("=" * 100)
    print("  V4 F/D/C 가중치 + ActivityMultiplier Sweep")
    print(f"  시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  테스트: {len(WEIGHT_CONFIGS)}개 조합")
    print("=" * 100)

    tickers = load_tickers()
    print(f"\n종목: {len(tickers)}개")

    print(f"\n데이터 로딩...")
    all_data = download_all_data(tickers, _cache_dir)
    print(f"로딩 완료: {len(all_data)}종목")

    results = run_sweep(all_data, tickers)
    print_results(results)

    elapsed = time.time() - t_start
    print(f"\n\n총 소요시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
    print(f"완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
