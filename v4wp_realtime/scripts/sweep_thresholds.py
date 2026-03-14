"""
Strong Buy / Strong Sell Threshold Sweep
=========================================
다양한 강매수/강매도 임계값 조합을 스윕하여 CAGR(연환산수익률)을 측정.

실행:
  cd "거래량 백테스트"
  python -m v4wp_realtime.scripts.sweep_thresholds
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_project_root = str(_script_dir.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd
import json, warnings, time
from itertools import product
warnings.filterwarnings('ignore')

from real_market_backtest import (
    download_data, calc_v4_score, detect_signal_events, build_price_filter,
)

# ============================================================
# 설정
# ============================================================
V4_WINDOW = 20
SIGNAL_THRESHOLD = 0.15
COOLDOWN = 5
ER_QUANTILE = 66
ATR_QUANTILE = 66
LOOKBACK = 252
INITIAL_CASH = 1000.0
MONTHLY_DEPOSIT = 100.0

# 매수/매도 비율 (고정)
BUY_PCT_NORMAL = 0.30
BUY_PCT_STRONG = 0.50
SELL_PCT_NORMAL = 0.05
SELL_PCT_STRONG = 0.10

# 스윕 범위
STRONG_BUY_THS  = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40]
STRONG_SELL_THS = [-0.15, -0.18, -0.20, -0.25, -0.30, -0.35, -0.40, -0.50]

# 백테스트 기간
START_DATE = '2007-01-01'
END_DATE   = '2026-03-31'


# ============================================================
# 백테스트 엔진 (단일 티커)
# ============================================================

def run_backtest_single(df, filtered_events, strong_buy_th, strong_sell_th):
    """가용자금 비율 기반 백테스트 - 단일 티커, 특정 임계값 조합"""

    sorted_events = sorted(filtered_events, key=lambda e: e['peak_idx'])
    event_by_idx = {}
    for ev in sorted_events:
        event_by_idx.setdefault(ev['peak_idx'], []).append(ev)

    cash = INITIAL_CASH
    total_deposited = INITIAL_CASH
    positions = []
    n_strong_buys = 0
    n_weak_buys = 0
    n_strong_sells = 0
    n_weak_sells = 0
    peak_portfolio = INITIAL_CASH
    max_drawdown = 0.0
    last_deposit_month = None

    for i in range(len(df)):
        price = df['Close'].iloc[i]
        date = df.index[i]

        # 매월 첫 거래일: 가용자금에 $100 추가
        cur_month = (date.year, date.month)
        if last_deposit_month is None:
            last_deposit_month = cur_month
        elif cur_month != last_deposit_month:
            cash += MONTHLY_DEPOSIT
            total_deposited += MONTHLY_DEPOSIT
            last_deposit_month = cur_month

        if i in event_by_idx:
            for ev in event_by_idx[i]:
                pv = ev['peak_val']

                if ev['type'] == 'bottom':
                    is_strong = abs(pv) >= strong_buy_th
                    buy_pct = BUY_PCT_STRONG if is_strong else BUY_PCT_NORMAL

                    if buy_pct <= 0 or cash <= 1:
                        continue

                    amt = cash * buy_pct
                    cash -= amt
                    shares = amt / price
                    positions.append({
                        'buy_price': price, 'shares': shares,
                        'buy_date': date, 'buy_idx': i,
                    })
                    if is_strong:
                        n_strong_buys += 1
                    else:
                        n_weak_buys += 1

                elif ev['type'] == 'top' and positions:
                    is_strong = pv <= strong_sell_th
                    sell_pct = SELL_PCT_STRONG if is_strong else SELL_PCT_NORMAL

                    if sell_pct <= 0:
                        continue

                    remaining = []
                    for pos in positions:
                        ss = pos['shares'] * sell_pct
                        ks = pos['shares'] * (1 - sell_pct)
                        if ss > 1e-10:
                            revenue = ss * price
                            cash += revenue
                        if ks > 1e-10:
                            remaining.append({**pos, 'shares': ks})
                    positions = remaining

                    if is_strong:
                        n_strong_sells += 1
                    else:
                        n_weak_sells += 1

        # MDD 추적
        holdings_val = sum(p['shares'] * price for p in positions)
        portfolio_val = cash + holdings_val
        if portfolio_val > peak_portfolio:
            peak_portfolio = portfolio_val
        dd = (peak_portfolio - portfolio_val) / peak_portfolio * 100 if peak_portfolio > 0 else 0
        if dd > max_drawdown:
            max_drawdown = dd

    # 최종 평가
    last_price = df['Close'].iloc[-1]
    holdings_val = sum(p['shares'] * last_price for p in positions)
    final_portfolio = cash + holdings_val
    total_pnl = final_portfolio - total_deposited

    ad = (df.index[-1] - df.index[0]).days
    ay = ad / 365.25 if ad > 0 else 1.0
    ann_return = ((final_portfolio / total_deposited) ** (1 / ay) - 1) if (
        ay > 0 and total_deposited > 0 and final_portfolio > 0
    ) else 0

    n_total_buys = n_strong_buys + n_weak_buys
    n_total_sells = n_strong_sells + n_weak_sells

    return {
        'total_deposited': total_deposited,
        'final_portfolio': final_portfolio,
        'total_pnl': total_pnl,
        'ann_return': ann_return * 100,  # percent
        'max_drawdown': max_drawdown,
        'actual_years': ay,
        'n_strong_buys': n_strong_buys,
        'n_weak_buys': n_weak_buys,
        'n_total_buys': n_total_buys,
        'n_strong_sells': n_strong_sells,
        'n_weak_sells': n_weak_sells,
        'n_total_sells': n_total_sells,
    }


# ============================================================
# 데이터 준비 (한 번만 로드/계산)
# ============================================================

def prepare_ticker_data(tickers, cache_dir):
    """모든 티커의 데이터, 스코어, 필터링된 이벤트를 미리 계산"""
    ticker_data = {}
    failed = []
    skipped = []

    for ticker in tickers:
        try:
            df = download_data(ticker, start=START_DATE, end=END_DATE, cache_dir=cache_dir)
            if len(df) < 100:
                skipped.append(ticker)
                continue
            score = calc_v4_score(df, w=V4_WINDOW)
            events = detect_signal_events(score, th=SIGNAL_THRESHOLD, cooldown=COOLDOWN)
            pf = build_price_filter(df, er_q=ER_QUANTILE, atr_q=ATR_QUANTILE, lookback=LOOKBACK)
            filtered = [e for e in events if pf(e['peak_idx'])]

            n_buy_signals = sum(1 for e in filtered if e['type'] == 'bottom')
            if n_buy_signals == 0:
                skipped.append(ticker)
                continue

            ticker_data[ticker] = {'df': df, 'events': filtered}
        except Exception as ex:
            failed.append(ticker)
            print(f"  [ERROR] {ticker}: {ex}")

    return ticker_data, failed, skipped


# ============================================================
# 스윕 엔진
# ============================================================

def run_sweep(ticker_data, buy_ths, sell_ths):
    """모든 (강매수, 강매도) 조합에 대해 백테스트 실행"""
    results = {}
    total_combos = len(buy_ths) * len(sell_ths)
    combo_idx = 0

    for sb_th in buy_ths:
        for ss_th in sell_ths:
            combo_idx += 1
            key = (sb_th, ss_th)
            combo_results = []

            for ticker, data in ticker_data.items():
                r = run_backtest_single(
                    data['df'], data['events'],
                    strong_buy_th=sb_th,
                    strong_sell_th=ss_th,
                )
                combo_results.append(r)

            if combo_results:
                n = len(combo_results)
                agg = {
                    'avg_ann_return': np.mean([r['ann_return'] for r in combo_results]),
                    'avg_final_portfolio': np.mean([r['final_portfolio'] for r in combo_results]),
                    'avg_mdd': np.mean([r['max_drawdown'] for r in combo_results]),
                    'avg_deposited': np.mean([r['total_deposited'] for r in combo_results]),
                    'total_strong_buys': sum(r['n_strong_buys'] for r in combo_results),
                    'total_weak_buys': sum(r['n_weak_buys'] for r in combo_results),
                    'total_buys': sum(r['n_total_buys'] for r in combo_results),
                    'total_strong_sells': sum(r['n_strong_sells'] for r in combo_results),
                    'total_weak_sells': sum(r['n_weak_sells'] for r in combo_results),
                    'total_sells': sum(r['n_total_sells'] for r in combo_results),
                    'n_tickers': n,
                }
                # strong/total ratios
                agg['strong_buy_ratio'] = (
                    agg['total_strong_buys'] / agg['total_buys'] * 100
                    if agg['total_buys'] > 0 else 0
                )
                agg['strong_sell_ratio'] = (
                    agg['total_strong_sells'] / agg['total_sells'] * 100
                    if agg['total_sells'] > 0 else 0
                )
                results[key] = agg

            if combo_idx % 10 == 0 or combo_idx == total_combos:
                print(f"  Progress: {combo_idx}/{total_combos} combinations done", flush=True)

    return results


# ============================================================
# 출력 유틸리티
# ============================================================

def print_heatmap_table(title, results, buy_ths, sell_ths, value_key, fmt='.2f',
                        suffix='%', highlight_max=True):
    """2D 테이블 출력 (행=강매도, 열=강매수)"""
    W = 12  # 셀 폭

    col_headers = [f'{th:.2f}' for th in buy_ths]
    row_headers = [f'{th:.2f}' for th in sell_ths]

    # 최대/최소값 찾기
    best_val = -1e18 if highlight_max else 1e18
    best_key = None
    for key, agg in results.items():
        v = agg[value_key]
        if (highlight_max and v > best_val) or (not highlight_max and v < best_val):
            best_val = v
            best_key = key

    header_w = 10
    line_w = header_w + 2 + len(col_headers) * (W + 1) + 1

    print(f"\n{'='*line_w}")
    print(f"  {title}")
    print(f"  (rows = Strong Sell Threshold, cols = Strong Buy Threshold)")
    print(f"{'='*line_w}")

    # 헤더 행
    header_line = f"{'Sell\\Buy':>{header_w}} |"
    for ch in col_headers:
        header_line += f"{ch:>{W}} "
    print(header_line)
    print(f"{'-'*header_w}-+-{'-'*(len(col_headers)*(W+1))}")

    for si, ss_th in enumerate(sell_ths):
        row_line = f"{row_headers[si]:>{header_w}} |"
        for bi, sb_th in enumerate(buy_ths):
            key = (sb_th, ss_th)
            if key in results:
                v = results[key][value_key]
                cell = f"{v:{fmt}}{suffix}"
                if key == best_key:
                    cell = f"*{cell}*"
                row_line += f"{cell:>{W}} "
            else:
                row_line += f"{'N/A':>{W}} "
        print(row_line)

    if best_key:
        print(f"\n  >>> BEST: Strong Buy={best_key[0]:.2f}, Strong Sell={best_key[1]:.2f}"
              f"  =>  {best_val:{fmt}}{suffix}")
    print()


def print_ranking(results, buy_ths, sell_ths, top_n=10):
    """Top N 조합 랭킹 출력"""
    ranked = sorted(results.items(), key=lambda x: x[1]['avg_ann_return'], reverse=True)

    print(f"\n{'='*120}")
    print(f"  TOP {top_n} COMBINATIONS (by Avg Annualized Return)")
    print(f"{'='*120}")
    print(f"  {'Rank':>4}  {'StrongBuy':>10}  {'StrongSell':>10}  "
          f"{'AvgAnnRet':>10}  {'AvgFinal$':>10}  {'AvgMDD':>8}  "
          f"{'SBuyRatio':>10}  {'SSellRatio':>10}  "
          f"{'TotalBuys':>10}  {'TotalSells':>10}")
    print(f"  {'-'*114}")

    for rank, (key, agg) in enumerate(ranked[:top_n], 1):
        sb_th, ss_th = key
        is_current = (abs(sb_th - 0.15) < 0.001 and abs(ss_th - (-0.25)) < 0.001)
        marker = " <-- CURRENT" if is_current else ""
        print(f"  {rank:>4}  {sb_th:>10.2f}  {ss_th:>10.2f}  "
              f"{agg['avg_ann_return']:>9.2f}%  "
              f"${agg['avg_final_portfolio']:>9.0f}  "
              f"{agg['avg_mdd']:>7.1f}%  "
              f"{agg['strong_buy_ratio']:>9.1f}%  "
              f"{agg['strong_sell_ratio']:>9.1f}%  "
              f"{agg['total_buys']:>10}  "
              f"{agg['total_sells']:>10}{marker}")

    # 현재 조합 찾기
    current_key = None
    for key in results:
        if abs(key[0] - 0.15) < 0.001 and abs(key[1] - (-0.25)) < 0.001:
            current_key = key
            break

    if current_key:
        current_rank = None
        for rank, (key, _) in enumerate(ranked, 1):
            if key == current_key:
                current_rank = rank
                break
        agg = results[current_key]
        print(f"\n  CURRENT SETTING (StrongBuy=0.15, StrongSell=-0.25):")
        print(f"    Rank: #{current_rank} / {len(ranked)}")
        print(f"    Avg Annualized Return: {agg['avg_ann_return']:.2f}%")
        print(f"    Avg Final Portfolio: ${agg['avg_final_portfolio']:.0f}")
        print(f"    Avg MDD: {agg['avg_mdd']:.1f}%")

    print()


# ============================================================
# 메인
# ============================================================

def main():
    t0 = time.time()

    config_path = _script_dir.parent / 'config' / 'watchlist.json'
    with open(config_path, encoding='utf-8') as f:
        config = json.load(f)

    tickers = list(config['tickers'].keys()) + config.get('benchmarks', [])
    cache_dir = str(Path(_project_root) / 'cache')

    W = 120
    print(f"\n{'='*W}")
    print(f"  Strong Buy / Strong Sell Threshold Sweep")
    print(f"  Period: {START_DATE} ~ {END_DATE}")
    print(f"  Tickers: {len(tickers)} ({', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''})")
    print(f"  Initial Cash: ${INITIAL_CASH:.0f} + Monthly ${MONTHLY_DEPOSIT:.0f}")
    print(f"  Buy: Normal {BUY_PCT_NORMAL*100:.0f}% / Strong {BUY_PCT_STRONG*100:.0f}%")
    print(f"  Sell: Normal {SELL_PCT_NORMAL*100:.0f}% / Strong {SELL_PCT_STRONG*100:.0f}%")
    print(f"  Strong Buy Thresholds:  {STRONG_BUY_THS}")
    print(f"  Strong Sell Thresholds: {STRONG_SELL_THS}")
    print(f"  Total Combinations: {len(STRONG_BUY_THS) * len(STRONG_SELL_THS)}")
    print(f"{'='*W}")

    # 1) 데이터 준비 (한 번만)
    print(f"\n--- Phase 1: Loading & computing indicators for {len(tickers)} tickers ---")
    ticker_data, failed, skipped = prepare_ticker_data(tickers, cache_dir)
    print(f"\n  Loaded: {len(ticker_data)} tickers")
    if skipped:
        print(f"  Skipped (insufficient data/signals): {', '.join(skipped)}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")

    # 2) 스윕 실행
    print(f"\n--- Phase 2: Running {len(STRONG_BUY_THS)*len(STRONG_SELL_THS)} threshold combinations ---")
    results = run_sweep(ticker_data, STRONG_BUY_THS, STRONG_SELL_THS)

    # 3) 결과 출력
    print(f"\n--- Phase 3: Results ---")

    # Table 1: Avg Annualized Return %
    print_heatmap_table(
        "TABLE 1: Average Annualized Return (%)",
        results, STRONG_BUY_THS, STRONG_SELL_THS,
        value_key='avg_ann_return', fmt='.2f', suffix='%', highlight_max=True
    )

    # Table 2: Avg Final Portfolio $
    print_heatmap_table(
        "TABLE 2: Average Final Portfolio ($)",
        results, STRONG_BUY_THS, STRONG_SELL_THS,
        value_key='avg_final_portfolio', fmt='.0f', suffix='', highlight_max=True
    )

    # Table 3: Strong Buy Count / Total Buy Count Ratio %
    print_heatmap_table(
        "TABLE 3: Strong Buy Ratio (%) = Strong Buys / Total Buys",
        results, STRONG_BUY_THS, STRONG_SELL_THS,
        value_key='strong_buy_ratio', fmt='.1f', suffix='%', highlight_max=False
    )

    # Table 4: Strong Sell Count / Total Sell Count Ratio %
    print_heatmap_table(
        "TABLE 4: Strong Sell Ratio (%) = Strong Sells / Total Sells",
        results, STRONG_BUY_THS, STRONG_SELL_THS,
        value_key='strong_sell_ratio', fmt='.1f', suffix='%', highlight_max=False
    )

    # Table 5: Avg MDD
    print_heatmap_table(
        "TABLE 5: Average Max Drawdown (%)",
        results, STRONG_BUY_THS, STRONG_SELL_THS,
        value_key='avg_mdd', fmt='.1f', suffix='%', highlight_max=False
    )

    # Ranking
    print_ranking(results, STRONG_BUY_THS, STRONG_SELL_THS, top_n=10)

    elapsed = time.time() - t0
    print(f"\n  Total elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  Done.")


if __name__ == '__main__':
    main()
