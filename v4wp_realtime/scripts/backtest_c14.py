"""
C14 Backtest: PRICE_FILTER_RELAX (ATR q 66->55) + LATE_SELL_BLOCK
=================================================================
- ATR quantile relaxed from 66 to 55
- Suppress sell signals when price already dropped >5% from 20-day rolling high
- Monthly $100 DCA, initial $1000
- All tickers from watchlist.json + benchmarks
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime

# Project root
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _project_root)

from real_market_backtest import (
    download_data,
    calc_v4_score,
    detect_signal_events,
    build_price_filter,
)

# ── Configuration ─────────────────────────────────────────────
CACHE_DIR = str(Path(_project_root) / 'cache')

V4_WINDOW = 20
SIGNAL_THRESHOLD = 0.15
COOLDOWN = 5
ER_QUANTILE = 66
ATR_QUANTILE = 55        # C14: relaxed from 66
LOOKBACK = 252

INITIAL_CASH = 1000.0
MONTHLY_ADD = 100.0
STRONG_BUY_TH = 0.15     # |peak_val| >= 0.15
STRONG_SELL_TH = -0.25    # peak_val <= -0.25
BUY_STRONG_PCT = 0.50
BUY_NORMAL_PCT = 0.30
SELL_STRONG_PCT = 0.10
SELL_NORMAL_PCT = 0.05
LATE_SELL_DROP_TH = 0.05  # 5% below 20-day rolling high

START_ALL = '2007-01-01'
END_ALL = '2026-03-31'

MARKET_PERIODS = [
    {'name': '1.리먼위기',   'start': '2007-01-01', 'end': '2009-12-31', 'desc': '서브프라임 -> 리먼 파산 -> 대폭락'},
    {'name': '2.회복장',     'start': '2009-03-01', 'end': '2015-12-31', 'desc': 'QE 양적완화 -> 장기 상승장'},
    {'name': '3.금리충격',   'start': '2018-01-01', 'end': '2019-12-31', 'desc': 'Fed 금리인상 -> 크리스마스 폭락'},
    {'name': '4.코로나',     'start': '2020-01-01', 'end': '2021-12-31', 'desc': 'COVID-19 폭락 -> V자 회복'},
    {'name': '5.인플레하락',  'start': '2022-01-01', 'end': '2023-06-30', 'desc': '인플레 -> 급격 금리인상 -> 기술주 폭락'},
    {'name': '6.AI+관세',   'start': '2023-07-01', 'end': '2026-03-31', 'desc': 'ChatGPT 랠리 -> 트럼프 관세 쇼크'},
    {'name': '7.전체',      'start': '2007-01-01', 'end': '2026-03-31', 'desc': '금융위기부터 현재까지'},
]


def load_tickers():
    """Load tickers + sectors from watchlist.json"""
    wl_path = os.path.join(_project_root, 'v4wp_realtime', 'config', 'watchlist.json')
    with open(wl_path, 'r', encoding='utf-8') as f:
        wl = json.load(f)
    ticker_map = {}
    for t, info in wl['tickers'].items():
        ticker_map[t] = info.get('sector', 'Unknown')
    for b in wl.get('benchmarks', []):
        if b not in ticker_map:
            ticker_map[b] = 'Benchmark'
    return ticker_map


def run_backtest_single(ticker, sector, period_start, period_end):
    """
    Run C14 backtest for a single ticker in a given period.
    Returns dict with results or None if data insufficient.
    """
    try:
        df_full = download_data(ticker, start=START_ALL, end=END_ALL, cache_dir=CACHE_DIR)
    except Exception as e:
        print(f"    [SKIP] {ticker}: {e}")
        return None

    if len(df_full) < 100:
        print(f"    [SKIP] {ticker}: insufficient data ({len(df_full)} days)")
        return None

    # Compute V4 score on FULL data
    v4_full = calc_v4_score(df_full, w=V4_WINDOW)

    # Build price filter on FULL data with C14 relaxed ATR
    pf = build_price_filter(df_full, er_q=ER_QUANTILE, atr_q=ATR_QUANTILE, lookback=LOOKBACK)

    # Detect events on FULL data, then filter by price filter
    events_all = detect_signal_events(v4_full, th=SIGNAL_THRESHOLD, cooldown=COOLDOWN)
    events_filtered = [e for e in events_all if pf(e['peak_idx'])]

    # LATE_SELL_BLOCK: compute 20-day rolling high of Close
    rolling_high_20 = df_full['Close'].rolling(20, min_periods=1).max()

    # Slice to period
    mask = (df_full.index >= period_start) & (df_full.index <= period_end)
    df_period = df_full.loc[mask]
    if len(df_period) < 10:
        return None

    period_start_ts = df_period.index[0]
    period_end_ts = df_period.index[-1]

    # Map period indices
    full_start_loc = df_full.index.get_loc(period_start_ts)
    full_end_loc = df_full.index.get_loc(period_end_ts)

    # Filter events to period range
    period_events = [e for e in events_filtered
                     if full_start_loc <= e['peak_idx'] <= full_end_loc]

    # ── Simulate trading ─────────────────────────────────────
    cash = INITIAL_CASH
    shares = 0.0
    total_deposited = INITIAL_CASH
    n_buys = 0
    n_sells = 0
    n_blocked_sells = 0

    # Monthly DCA tracking
    months_added = set()

    # Portfolio value tracking for MDD
    portfolio_values = []
    dates_track = []

    # Track open positions (simplified: average cost basis)
    cost_basis = 0.0

    # Process day by day
    event_dict = {}
    for e in period_events:
        idx = e['peak_idx']
        if idx not in event_dict:
            event_dict[idx] = []
        event_dict[idx].append(e)

    for i in range(len(df_period)):
        date = df_period.index[i]
        price = df_period['Close'].iloc[i]
        full_idx = df_full.index.get_loc(date)

        # Monthly $100 DCA on first trading day of each month
        month_key = (date.year, date.month)
        if month_key not in months_added and month_key != (period_start_ts.year, period_start_ts.month) or \
           (month_key == (period_start_ts.year, period_start_ts.month) and i > 0):
            # Add monthly contribution (skip the very first month's first day since initial cash already given)
            pass

        if month_key not in months_added:
            if i > 0:  # Don't add on the first day (initial cash already covers it)
                cash += MONTHLY_ADD
                total_deposited += MONTHLY_ADD
            months_added.add(month_key)

        # Check for signal events on this day
        if full_idx in event_dict:
            for ev in event_dict[full_idx]:
                if ev['type'] == 'bottom':
                    # BUY signal
                    is_strong = abs(ev['peak_val']) >= STRONG_BUY_TH
                    buy_pct = BUY_STRONG_PCT if is_strong else BUY_NORMAL_PCT
                    buy_amount = cash * buy_pct
                    if buy_amount > 1.0 and price > 0:
                        new_shares = buy_amount / price
                        # Update cost basis
                        total_cost = cost_basis * shares + buy_amount
                        shares += new_shares
                        cost_basis = total_cost / shares if shares > 0 else 0
                        cash -= buy_amount
                        n_buys += 1

                elif ev['type'] == 'top':
                    # SELL signal - check LATE_SELL_BLOCK
                    rh = rolling_high_20.iloc[full_idx] if full_idx < len(rolling_high_20) else price
                    if price < rh * (1 - LATE_SELL_DROP_TH):
                        # BLOCKED: price already dropped >5% from 20-day high
                        n_blocked_sells += 1
                        continue

                    is_strong = ev['peak_val'] <= STRONG_SELL_TH
                    sell_pct = SELL_STRONG_PCT if is_strong else SELL_NORMAL_PCT
                    sell_shares = shares * sell_pct
                    if sell_shares > 0.0001 and price > 0:
                        proceeds = sell_shares * price
                        shares -= sell_shares
                        cash += proceeds
                        n_sells += 1

        # Track portfolio value
        portfolio_val = cash + shares * price
        portfolio_values.append(portfolio_val)
        dates_track.append(date)

    # Final values
    final_price = df_period['Close'].iloc[-1]
    final_portfolio = cash + shares * final_price
    pnl = final_portfolio - total_deposited
    profit_rate = (pnl / total_deposited * 100) if total_deposited > 0 else 0

    # Annualized return
    n_days = (period_end_ts - period_start_ts).days
    n_years = n_days / 365.25
    if n_years > 0 and total_deposited > 0:
        # Use simple annualized: (final/deposited)^(1/years) - 1
        ratio = final_portfolio / total_deposited
        if ratio > 0:
            ann_return = (ratio ** (1 / n_years) - 1) * 100
        else:
            ann_return = -100.0
    else:
        ann_return = 0.0

    # MDD
    if len(portfolio_values) > 0:
        pv_arr = np.array(portfolio_values)
        running_max = np.maximum.accumulate(pv_arr)
        drawdowns = (pv_arr - running_max) / running_max
        mdd = drawdowns.min() * 100  # negative percentage
    else:
        mdd = 0.0

    # Win rate: based on whether final portfolio > total deposited
    win = 1 if pnl > 0 else 0

    return {
        'ticker': ticker,
        'sector': sector,
        'n_buys': n_buys,
        'n_sells': n_sells,
        'n_blocked_sells': n_blocked_sells,
        'open_positions': shares,
        'total_deposited': total_deposited,
        'final_portfolio': final_portfolio,
        'cash_remaining': cash,
        'pnl': pnl,
        'profit_rate': profit_rate,
        'ann_return': ann_return,
        'mdd': mdd,
        'win': win,
    }


def print_period_results(period_name, period_desc, results):
    """Print results for a single period"""
    if not results:
        print(f"\n  기간에 해당하는 데이터 없음\n")
        return

    print(f"\n{'='*120}")
    print(f"  기간: {period_name} | {period_desc}")
    print(f"{'='*120}")

    # Summary statistics
    n = len(results)
    avg_deposited = np.mean([r['total_deposited'] for r in results])
    avg_final = np.mean([r['final_portfolio'] for r in results])
    avg_pnl = np.mean([r['pnl'] for r in results])
    avg_profit = np.mean([r['profit_rate'] for r in results])
    avg_ann = np.mean([r['ann_return'] for r in results])
    avg_mdd = np.mean([r['mdd'] for r in results])
    win_rate = sum(r['win'] for r in results) / n * 100
    total_buys = sum(r['n_buys'] for r in results)
    total_sells = sum(r['n_sells'] for r in results)
    total_blocked = sum(r['n_blocked_sells'] for r in results)

    print(f"\n  [요약]")
    print(f"  총 종목수: {n}")
    print(f"  평균 입금액: ${avg_deposited:,.2f}")
    print(f"  평균 최종 포트폴리오: ${avg_final:,.2f}")
    print(f"  평균 손익: ${avg_pnl:,.2f}")
    print(f"  평균 수익률: {avg_profit:.2f}%")
    print(f"  평균 연환산 수익률: {avg_ann:.2f}%")
    print(f"  평균 MDD: {avg_mdd:.2f}%")
    print(f"  승률: {win_rate:.1f}% ({sum(r['win'] for r in results)}/{n})")
    print(f"  총 매수 신호: {total_buys} | 총 매도 신호: {total_sells} | 차단된 매도: {total_blocked}")

    # Individual stock data sorted by annualized return (descending)
    sorted_results = sorted(results, key=lambda r: r['ann_return'], reverse=True)

    print(f"\n  [개별 종목 데이터] (연환산 수익률 내림차순)")
    print(f"  {'─'*118}")
    header = (f"  {'Ticker':<7} {'Sector':<12} {'매수':>4} {'보유':>8} {'입금액':>12} "
              f"{'최종포트':>12} {'잔여현금':>12} {'손익':>12} {'수익률':>8} "
              f"{'연환산':>8} {'MDD':>8} {'승패':>4}")
    print(header)
    print(f"  {'─'*118}")

    for r in sorted_results:
        win_str = 'WIN' if r['win'] else 'LOSE'
        print(f"  {r['ticker']:<7} {r['sector']:<12} {r['n_buys']:>4} "
              f"{r['open_positions']:>8.2f} ${r['total_deposited']:>10,.2f} "
              f"${r['final_portfolio']:>10,.2f} ${r['cash_remaining']:>10,.2f} "
              f"${r['pnl']:>10,.2f} {r['profit_rate']:>7.2f}% "
              f"{r['ann_return']:>7.2f}% {r['mdd']:>7.2f}% {win_str:>4}")

    print(f"  {'─'*118}")


def main():
    print("=" * 120)
    print("  C14 백테스트: PRICE_FILTER_RELAX (ATR q=55) + LATE_SELL_BLOCK (5% drop block)")
    print("  V4_WINDOW=20, SIGNAL_THRESHOLD=0.15, COOLDOWN=5")
    print(f"  ER_q={ER_QUANTILE}, ATR_q={ATR_QUANTILE}, LOOKBACK={LOOKBACK}")
    print(f"  초기자금=${INITIAL_CASH}, 월적립=${MONTHLY_ADD}")
    print("=" * 120)

    # Load tickers
    ticker_map = load_tickers()
    all_tickers = list(ticker_map.keys())
    print(f"\n  총 {len(all_tickers)}개 종목: {', '.join(all_tickers)}")

    # Pre-download all data
    print(f"\n{'='*80}")
    print(f"  데이터 다운로드/캐시 로드")
    print(f"{'='*80}")

    # Run backtest for each period
    all_period_results = {}

    for period in MARKET_PERIODS:
        p_name = period['name']
        p_start = period['start']
        p_end = period['end']
        p_desc = period['desc']

        print(f"\n\n{'#'*120}")
        print(f"  기간 처리 중: {p_name} ({p_start} ~ {p_end})")
        print(f"{'#'*120}")

        results = []
        for ticker in all_tickers:
            sector = ticker_map[ticker]
            r = run_backtest_single(ticker, sector, p_start, p_end)
            if r is not None:
                results.append(r)

        all_period_results[p_name] = results
        print_period_results(p_name, p_desc, results)

    # ── Cross-tables ─────────────────────────────────────────────

    # 4. Cross-table: periods x annualized return
    print(f"\n\n{'='*120}")
    print(f"  [크로스테이블 1] 기간별 x 종목별 연환산 수익률 (%)")
    print(f"{'='*120}")

    period_names = [p['name'] for p in MARKET_PERIODS]
    all_ticker_set = sorted(set(t for results in all_period_results.values() for r in results for t in [r['ticker']]))

    header = f"  {'Ticker':<7}"
    for pn in period_names:
        header += f" {pn:>12}"
    print(header)
    print(f"  {'─' * (7 + 13 * len(period_names))}")

    for ticker in all_ticker_set:
        line = f"  {ticker:<7}"
        for pn in period_names:
            results = all_period_results.get(pn, [])
            r = next((x for x in results if x['ticker'] == ticker), None)
            if r:
                line += f" {r['ann_return']:>11.2f}%"
            else:
                line += f" {'N/A':>12}"
        print(line)

    # 5. Cross-table: periods x avg final portfolio / total deposited
    print(f"\n\n{'='*120}")
    print(f"  [크로스테이블 2] 기간별 x 종목별 최종포트/입금액 비율")
    print(f"{'='*120}")

    header = f"  {'Ticker':<7}"
    for pn in period_names:
        header += f" {pn:>12}"
    print(header)
    print(f"  {'─' * (7 + 13 * len(period_names))}")

    for ticker in all_ticker_set:
        line = f"  {ticker:<7}"
        for pn in period_names:
            results = all_period_results.get(pn, [])
            r = next((x for x in results if x['ticker'] == ticker), None)
            if r and r['total_deposited'] > 0:
                ratio = r['final_portfolio'] / r['total_deposited']
                line += f" {ratio:>11.3f}x"
            else:
                line += f" {'N/A':>12}"
        print(line)

    # 6. Cross-table: periods x MDD
    print(f"\n\n{'='*120}")
    print(f"  [크로스테이블 3] 기간별 x 종목별 MDD (%)")
    print(f"{'='*120}")

    header = f"  {'Ticker':<7}"
    for pn in period_names:
        header += f" {pn:>12}"
    print(header)
    print(f"  {'─' * (7 + 13 * len(period_names))}")

    for ticker in all_ticker_set:
        line = f"  {ticker:<7}"
        for pn in period_names:
            results = all_period_results.get(pn, [])
            r = next((x for x in results if x['ticker'] == ticker), None)
            if r:
                line += f" {r['mdd']:>11.2f}%"
            else:
                line += f" {'N/A':>12}"
        print(line)

    # 7. Ranking by full-period annualized return
    full_period_name = '7.전체'
    full_results = all_period_results.get(full_period_name, [])

    if full_results:
        print(f"\n\n{'='*120}")
        print(f"  [종목 랭킹] 전체기간 연환산 수익률 순위")
        print(f"{'='*120}")

        ranked = sorted(full_results, key=lambda r: r['ann_return'], reverse=True)
        print(f"  {'순위':>4} {'Ticker':<7} {'Sector':<12} {'연환산':>8} {'수익률':>8} {'MDD':>8} {'매수':>4} {'매도':>4} {'차단':>4}")
        print(f"  {'─'*70}")
        for i, r in enumerate(ranked, 1):
            print(f"  {i:>4} {r['ticker']:<7} {r['sector']:<12} {r['ann_return']:>7.2f}% "
                  f"{r['profit_rate']:>7.2f}% {r['mdd']:>7.2f}% {r['n_buys']:>4} {r['n_sells']:>4} {r['n_blocked_sells']:>4}")

    # 8. Sector analysis
    if full_results:
        print(f"\n\n{'='*120}")
        print(f"  [섹터 분석] 전체기간 섹터별 평균 수익률")
        print(f"{'='*120}")

        sectors = {}
        for r in full_results:
            s = r['sector']
            if s not in sectors:
                sectors[s] = []
            sectors[s].append(r)

        print(f"  {'Sector':<12} {'종목수':>6} {'평균연환산':>10} {'평균수익률':>10} {'평균MDD':>8} {'승률':>8}")
        print(f"  {'─'*60}")

        for s in sorted(sectors.keys()):
            sr = sectors[s]
            n = len(sr)
            avg_ann = np.mean([r['ann_return'] for r in sr])
            avg_pf = np.mean([r['profit_rate'] for r in sr])
            avg_mdd = np.mean([r['mdd'] for r in sr])
            wr = sum(r['win'] for r in sr) / n * 100
            print(f"  {s:<12} {n:>6} {avg_ann:>9.2f}% {avg_pf:>9.2f}% {avg_mdd:>7.2f}% {wr:>7.1f}%")

    # 9. Best/worst 5 tickers
    if full_results:
        print(f"\n\n{'='*120}")
        print(f"  [BEST 5] 전체기간 최고 수익 종목")
        print(f"{'='*120}")
        top5 = sorted(full_results, key=lambda r: r['ann_return'], reverse=True)[:5]
        for i, r in enumerate(top5, 1):
            print(f"  {i}. {r['ticker']} ({r['sector']}) | 연환산: {r['ann_return']:.2f}% | "
                  f"수익률: {r['profit_rate']:.2f}% | MDD: {r['mdd']:.2f}% | "
                  f"입금: ${r['total_deposited']:,.2f} -> 최종: ${r['final_portfolio']:,.2f}")

        print(f"\n  [WORST 5] 전체기간 최저 수익 종목")
        print(f"  {'─'*100}")
        bot5 = sorted(full_results, key=lambda r: r['ann_return'])[:5]
        for i, r in enumerate(bot5, 1):
            print(f"  {i}. {r['ticker']} ({r['sector']}) | 연환산: {r['ann_return']:.2f}% | "
                  f"수익률: {r['profit_rate']:.2f}% | MDD: {r['mdd']:.2f}% | "
                  f"입금: ${r['total_deposited']:,.2f} -> 최종: ${r['final_portfolio']:,.2f}")

    # 10. Summary statistics
    print(f"\n\n{'='*120}")
    print(f"  [종합 통계]")
    print(f"{'='*120}")

    for pn in period_names:
        results = all_period_results.get(pn, [])
        if not results:
            continue
        n = len(results)
        avg_ann = np.mean([r['ann_return'] for r in results])
        med_ann = np.median([r['ann_return'] for r in results])
        std_ann = np.std([r['ann_return'] for r in results])
        avg_mdd = np.mean([r['mdd'] for r in results])
        wr = sum(r['win'] for r in results) / n * 100
        total_blocked = sum(r['n_blocked_sells'] for r in results)
        total_sells = sum(r['n_sells'] for r in results)
        total_buys = sum(r['n_buys'] for r in results)

        print(f"\n  {pn}:")
        print(f"    종목수={n}, 평균연환산={avg_ann:.2f}%, 중앙값={med_ann:.2f}%, "
              f"표준편차={std_ann:.2f}%")
        print(f"    평균MDD={avg_mdd:.2f}%, 승률={wr:.1f}%")
        print(f"    총매수={total_buys}, 총매도={total_sells}, 차단매도={total_blocked}")

    print(f"\n\n{'='*120}")
    print(f"  C14 백테스트 완료!")
    print(f"{'='*120}\n")


if __name__ == '__main__':
    main()
