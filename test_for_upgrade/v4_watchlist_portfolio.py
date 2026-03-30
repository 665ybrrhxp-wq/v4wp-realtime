"""현재 watchlist 전체: 매수 신호당 $100 투자 시 포트폴리오 성과 백테스트

- GEO-OP 파이프라인 (AND-GEO + PF + BUY_DD_GATE)
- 매수 신호 발생 시 $100 매입, 데이터 끝까지 보유
- 종목별/연도별/전체 연평균수익률 산출
"""
import sys, json
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import pandas as pd
from datetime import datetime
from real_market_backtest import (
    download_data, calc_v4_score, detect_signal_events,
    build_price_filter, smooth_earnings_volume,
)

# -- 설정 로드 --
with open(Path(_root) / 'v4wp_realtime' / 'config' / 'watchlist.json') as f:
    wl = json.load(f)

TICKERS = list(wl['tickers'].keys())
SECTORS = {t: v['sector'] for t, v in wl['tickers'].items()}
BENCHMARKS = wl.get('benchmarks', [])
P = wl['params']


def run_pipeline(ticker, df):
    """프로덕션 파이프라인 재현: 매수 신호만 추출."""
    df = smooth_earnings_volume(df, ticker=ticker)
    score = calc_v4_score(df, w=P['v4_window'], divgate_days=P['divgate_days'])
    events = detect_signal_events(score, th=P['signal_threshold'], cooldown=P['cooldown'])
    pf = build_price_filter(df, er_q=P['er_quantile'], atr_q=P['atr_quantile'],
                            lookback=P['lookback'])

    # 매수만 + PF 필터
    buys = [e for e in events if e['type'] == 'bottom' and pf(e['peak_idx'])]

    # BUY_DD_GATE
    close = df['Close'].values
    rh = df['Close'].rolling(P['buy_dd_lookback'], min_periods=1).max().values

    filtered = []
    for e in buys:
        pidx = e['peak_idx']
        if pidx >= len(close):
            continue
        dd = (rh[pidx] - close[pidx]) / rh[pidx] if rh[pidx] > 0 else 0
        if dd >= P['buy_dd_threshold']:
            # Duration 확인
            duration = e.get('duration', e['end_idx'] - e['start_idx'] + 1)
            if duration >= P['confirm_days']:
                filtered.append(e)

    return filtered


def backtest_ticker(ticker, df, signals):
    """종목별 $100/signal 투자 성과 계산."""
    close = df['Close'].values
    dates = df.index
    last_price = close[-1]
    last_date = dates[-1]

    trades = []
    for e in signals:
        pidx = e['peak_idx']
        buy_price = close[pidx]
        buy_date = dates[pidx]
        shares = 100.0 / buy_price
        final_value = shares * last_price
        pnl = final_value - 100.0
        hold_days = (last_date - buy_date).days

        trades.append({
            'ticker': ticker,
            'sector': SECTORS.get(ticker, ''),
            'buy_date': buy_date.strftime('%Y-%m-%d'),
            'buy_price': buy_price,
            'final_price': last_price,
            'invested': 100.0,
            'final_value': final_value,
            'pnl': pnl,
            'return_pct': pnl,  # $100 기준이므로 pnl = %
            'hold_days': hold_days,
        })
    return trades


# ==========================================================
print('=' * 70)
print('  V4_wP Watchlist Portfolio Backtest ($100 per signal)')
print('=' * 70)

all_trades = []
ticker_summary = []

for ticker in TICKERS:
    try:
        df = download_data(ticker, start='2020-01-01',
                           end=datetime.now().strftime('%Y-%m-%d'),
                           cache_dir=str(Path(_root) / 'cache'))
        if df is None or len(df) < 200:
            print(f'  {ticker}: insufficient data, skip')
            continue

        signals = run_pipeline(ticker, df)
        trades = backtest_ticker(ticker, df, signals)
        all_trades.extend(trades)

        if trades:
            total_inv = sum(t['invested'] for t in trades)
            total_val = sum(t['final_value'] for t in trades)
            total_pnl = total_val - total_inv
            avg_ret = np.mean([t['return_pct'] for t in trades])
            avg_hold = np.mean([t['hold_days'] for t in trades])
            # 연환산: (final/invested)^(365/avg_hold) - 1
            if avg_hold > 0:
                ann_ret = ((total_val / total_inv) ** (365.0 / avg_hold) - 1) * 100
            else:
                ann_ret = 0
            ticker_summary.append({
                'ticker': ticker,
                'sector': SECTORS.get(ticker, ''),
                'n_signals': len(trades),
                'total_invested': total_inv,
                'total_value': total_val,
                'total_pnl': total_pnl,
                'avg_return': avg_ret,
                'avg_hold_days': avg_hold,
                'ann_return': ann_ret,
            })
            print(f'  {ticker:6s}: {len(trades):3d} signals, '
                  f'avg ret {avg_ret:+.1f}%, ann {ann_ret:+.1f}%')
        else:
            print(f'  {ticker:6s}:   0 signals')
            ticker_summary.append({
                'ticker': ticker, 'sector': SECTORS.get(ticker, ''),
                'n_signals': 0, 'total_invested': 0, 'total_value': 0,
                'total_pnl': 0, 'avg_return': 0, 'avg_hold_days': 0, 'ann_return': 0,
            })
    except Exception as ex:
        print(f'  {ticker}: ERROR - {ex}')

# -- 벤치마크 (QQQ, VOO) Buy & Hold --
print(f'\n{"-" * 70}')
print('  Benchmark: Buy & Hold (같은 기간)')
bench_results = {}
for bm in BENCHMARKS:
    try:
        df = download_data(bm, start='2020-01-01',
                           end=datetime.now().strftime('%Y-%m-%d'),
                           cache_dir=str(Path(_root) / 'cache'))
        if df is not None and len(df) > 200:
            first_price = df['Close'].iloc[0]
            last_price = df['Close'].iloc[-1]
            total_days = (df.index[-1] - df.index[0]).days
            bh_return = (last_price / first_price - 1) * 100
            bh_ann = ((last_price / first_price) ** (365.0 / total_days) - 1) * 100
            bench_results[bm] = {'total': bh_return, 'ann': bh_ann, 'days': total_days}
            print(f'  {bm}: {bh_return:+.1f}% total, {bh_ann:+.1f}% ann ({total_days}d)')
    except Exception as ex:
        print(f'  {bm}: ERROR - {ex}')

# -- DCA 비교 (매월 $100 QQQ/VOO) --
print(f'\n{"-" * 70}')
print('  Benchmark: Monthly $100 DCA')
for bm in BENCHMARKS:
    try:
        df = download_data(bm, start='2020-01-01',
                           end=datetime.now().strftime('%Y-%m-%d'),
                           cache_dir=str(Path(_root) / 'cache'))
        if df is None or len(df) < 200:
            continue
        # 매월 첫 거래일 $100 투자
        monthly = df.resample('MS').first().dropna()
        total_shares = 0
        total_invested = 0
        for date, row in monthly.iterrows():
            total_shares += 100.0 / row['Close']
            total_invested += 100.0
        final_value = total_shares * df['Close'].iloc[-1]
        dca_return = (final_value / total_invested - 1) * 100
        n_months = len(monthly)
        dca_ann = ((final_value / total_invested) ** (12.0 / n_months) - 1) * 100
        print(f'  {bm}: ${total_invested:.0f} invested -> ${final_value:.0f} '
              f'({dca_return:+.1f}%), ann {dca_ann:+.1f}% ({n_months}mo)')
    except Exception:
        pass

# ==========================================================
# 전체 포트폴리오 요약
print(f'\n{"=" * 70}')
print('  PORTFOLIO SUMMARY')
print(f'{"=" * 70}')

if all_trades:
    total_invested = sum(t['invested'] for t in all_trades)
    total_value = sum(t['final_value'] for t in all_trades)
    total_pnl = total_value - total_invested
    avg_return = np.mean([t['return_pct'] for t in all_trades])
    avg_hold = np.mean([t['hold_days'] for t in all_trades])
    win_rate = sum(1 for t in all_trades if t['pnl'] > 0) / len(all_trades) * 100

    # 포트폴리오 전체 연환산
    if avg_hold > 0:
        port_ann = ((total_value / total_invested) ** (365.0 / avg_hold) - 1) * 100
    else:
        port_ann = 0

    print(f'  총 신호 수:     {len(all_trades)}')
    print(f'  총 투자금:      ${total_invested:,.0f}')
    print(f'  현재 평가금:    ${total_value:,.0f}')
    print(f'  총 수익:        ${total_pnl:+,.0f} ({total_pnl/total_invested*100:+.1f}%)')
    print(f'  평균 수익률:    {avg_return:+.1f}% (per signal)')
    print(f'  승률:           {win_rate:.1f}%')
    print(f'  평균 보유일:    {avg_hold:.0f}일')
    print(f'  연평균수익률:   {port_ann:+.1f}%')

    # 종목별 표
    print(f'\n{"-" * 70}')
    print(f'  {"Ticker":6s} {"Sector":8s} {"#Sig":>5s} {"Invested":>10s} '
          f'{"Value":>10s} {"PnL":>10s} {"AvgRet":>8s} {"AnnRet":>8s}')
    print(f'  {"-"*6} {"-"*8} {"-"*5} {"-"*10} {"-"*10} {"-"*10} {"-"*8} {"-"*8}')
    for s in sorted(ticker_summary, key=lambda x: -x['ann_return']):
        if s['n_signals'] == 0:
            print(f'  {s["ticker"]:6s} {s["sector"]:8s}     0          -          -          -        -        -')
        else:
            print(f'  {s["ticker"]:6s} {s["sector"]:8s} {s["n_signals"]:5d} '
                  f'${s["total_invested"]:>9,.0f} ${s["total_value"]:>9,.0f} '
                  f'${s["total_pnl"]:>+9,.0f} {s["avg_return"]:>+7.1f}% {s["ann_return"]:>+7.1f}%')

    # 연도별 분석
    print(f'\n{"-" * 70}')
    print('  연도별 신호 성과:')
    for t in all_trades:
        t['year'] = t['buy_date'][:4]
    years = sorted(set(t['year'] for t in all_trades))
    for yr in years:
        yr_trades = [t for t in all_trades if t['year'] == yr]
        yr_inv = sum(t['invested'] for t in yr_trades)
        yr_val = sum(t['final_value'] for t in yr_trades)
        yr_avg = np.mean([t['return_pct'] for t in yr_trades])
        yr_win = sum(1 for t in yr_trades if t['pnl'] > 0) / len(yr_trades) * 100
        print(f'    {yr}: {len(yr_trades):3d} signals, avg ret {yr_avg:+.1f}%, '
              f'win {yr_win:.0f}%, ${yr_inv:,.0f} -> ${yr_val:,.0f}')

print(f'\n{"=" * 70}')
