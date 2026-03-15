"""
전체 17종목 V4 매수 신호 적중률 분석
30d, 60d, 90d forward return hit rate
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from real_market_backtest import calc_v4_score, detect_signal_events, build_price_filter

BUY_CONFIRM = 3
LATE_SELL_DROP_TH = 0.05

TICKERS = ['TSLA','PLTR','NVDA','AVGO','AMZN','GOOGL','JOBY','HIMS',
           'TEM','RKLB','PGY','COIN','HOOD','IONQ','PL','QQQ','VOO']

def download_max(ticker):
    t = yf.Ticker(ticker)
    df = t.history(period='max', auto_adjust=True)
    if df is None or len(df) == 0:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

print("=" * 110)
print("  V4 매수 신호 적중률 분석 (전체 종목, 전체 기간)")
print("=" * 110)
print()

all_results = []
all_buy_details = []

for tk in TICKERS:
    print(f"  Processing {tk}...", end=" ", flush=True)
    df = download_max(tk)
    if df is None or len(df) < 100:
        print("SKIP (insufficient data)")
        continue

    score = calc_v4_score(df, w=20)
    events = detect_signal_events(score, th=0.15, cooldown=5)
    pf = build_price_filter(df, er_q=66, atr_q=55, lookback=252)
    close = df['Close'].values
    dates = df.index
    rh20 = df['Close'].rolling(20, min_periods=1).max().values
    sarr = score.values
    n = len(close)

    # Filter events (same as production)
    filtered = []
    for ev in events:
        if not pf(ev['peak_idx']):
            continue
        if ev['type'] == 'top':
            pidx = ev['peak_idx']
            if rh20[pidx] > 0 and (rh20[pidx] - close[pidx]) / rh20[pidx] > LATE_SELL_DROP_TH:
                continue
        ev['duration'] = ev['end_idx'] - ev['start_idx'] + 1
        filtered.append(ev)

    # Find buy signal dates (duration >= BUY_CONFIRM)
    buy_signals = []
    for ev in filtered:
        if ev['type'] != 'bottom':
            continue
        si, ei = ev['start_idx'], ev['end_idx']
        dur = ev['duration']
        ci = si + BUY_CONFIRM - 1  # confirmation index
        if ci <= ei and dur >= BUY_CONFIRM:
            buy_signals.append(ci)

    if not buy_signals:
        print(f"NO BUY SIGNALS")
        continue

    # Calculate forward returns
    returns_30 = []
    returns_60 = []
    returns_90 = []

    for idx in buy_signals:
        p = close[idx]
        d_str = dates[idx].strftime('%Y-%m-%d')

        # 30d
        idx30 = min(idx + 30, n - 1)
        r30 = ((close[idx30] / p) - 1) * 100
        # 60d
        idx60 = min(idx + 60, n - 1)
        r60 = ((close[idx60] / p) - 1) * 100
        # 90d
        idx90 = min(idx + 90, n - 1)
        r90 = ((close[idx90] / p) - 1) * 100

        # Skip if we don't have enough future data
        has_30 = (idx + 30) < n
        has_60 = (idx + 60) < n
        has_90 = (idx + 90) < n

        if has_30:
            returns_30.append(r30)
        if has_60:
            returns_60.append(r60)
        if has_90:
            returns_90.append(r90)

        all_buy_details.append({
            'ticker': tk,
            'date': d_str,
            'price': p,
            'r30': r30 if has_30 else None,
            'r60': r60 if has_60 else None,
            'r90': r90 if has_90 else None,
        })

    n_buys = len(buy_signals)
    pos_30 = sum(1 for r in returns_30 if r > 0)
    pos_60 = sum(1 for r in returns_60 if r > 0)
    pos_90 = sum(1 for r in returns_90 if r > 0)

    n30 = len(returns_30)
    n60 = len(returns_60)
    n90 = len(returns_90)

    hr30 = pos_30 / n30 * 100 if n30 > 0 else 0
    hr60 = pos_60 / n60 * 100 if n60 > 0 else 0
    hr90 = pos_90 / n90 * 100 if n90 > 0 else 0

    avg_r30 = np.mean(returns_30) if returns_30 else 0
    avg_r60 = np.mean(returns_60) if returns_60 else 0
    avg_r90 = np.mean(returns_90) if returns_90 else 0

    period = f"{dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}"

    all_results.append({
        'ticker': tk,
        'period': period,
        'n_days': len(df),
        'n_buys': n_buys,
        'n30': n30, 'pos_30': pos_30, 'hr30': hr30, 'avg_r30': avg_r30,
        'n60': n60, 'pos_60': pos_60, 'hr60': hr60, 'avg_r60': avg_r60,
        'n90': n90, 'pos_90': pos_90, 'hr90': hr90, 'avg_r90': avg_r90,
    })

    print(f"Buys={n_buys:>3d}  30d={pos_30}/{n30}({hr30:.0f}%)  60d={pos_60}/{n60}({hr60:.0f}%)  90d={pos_90}/{n90}({hr90:.0f}%)")

# ── Summary Table ──
print()
print("=" * 110)
print(f"  {'Ticker':<8s} {'Period':<28s} {'Days':>6s} {'Buys':>5s} "
      f"{'30d HR':>8s} {'30d Avg':>8s} {'60d HR':>8s} {'60d Avg':>8s} {'90d HR':>8s} {'90d Avg':>8s}")
print("-" * 110)

for r in all_results:
    print(f"  {r['ticker']:<8s} {r['period']:<28s} {r['n_days']:>6d} {r['n_buys']:>5d} "
          f"{r['pos_30']:>3d}/{r['n30']:<3d}{r['hr30']:>3.0f}% {r['avg_r30']:>+7.1f}% "
          f"{r['pos_60']:>3d}/{r['n60']:<3d}{r['hr60']:>3.0f}% {r['avg_r60']:>+7.1f}% "
          f"{r['pos_90']:>3d}/{r['n90']:<3d}{r['hr90']:>3.0f}% {r['avg_r90']:>+7.1f}%")

# ── Grand Totals ──
print("-" * 110)
total_buys = sum(r['n_buys'] for r in all_results)
total_pos30 = sum(r['pos_30'] for r in all_results)
total_n30 = sum(r['n30'] for r in all_results)
total_pos60 = sum(r['pos_60'] for r in all_results)
total_n60 = sum(r['n60'] for r in all_results)
total_pos90 = sum(r['pos_90'] for r in all_results)
total_n90 = sum(r['n90'] for r in all_results)

grand_hr30 = total_pos30 / total_n30 * 100 if total_n30 else 0
grand_hr60 = total_pos60 / total_n60 * 100 if total_n60 else 0
grand_hr90 = total_pos90 / total_n90 * 100 if total_n90 else 0

grand_avg30 = np.mean([r['avg_r30'] for r in all_results])
grand_avg60 = np.mean([r['avg_r60'] for r in all_results])
grand_avg90 = np.mean([r['avg_r90'] for r in all_results])

print(f"  {'TOTAL':<8s} {'':28s} {'':>6s} {total_buys:>5d} "
      f"{total_pos30:>3d}/{total_n30:<3d}{grand_hr30:>3.0f}% {grand_avg30:>+7.1f}% "
      f"{total_pos60:>3d}/{total_n60:<3d}{grand_hr60:>3.0f}% {grand_avg60:>+7.1f}% "
      f"{total_pos90:>3d}/{total_n90:<3d}{grand_hr90:>3.0f}% {grand_avg90:>+7.1f}%")

# ── Category Breakdown ──
categories = {
    'Benchmark (QQQ,VOO)': ['QQQ', 'VOO'],
    'Big Tech (NVDA,AVGO,GOOGL,AMZN,TSLA)': ['NVDA', 'AVGO', 'GOOGL', 'AMZN', 'TSLA'],
    'Growth (PLTR,RKLB,PGY,HIMS,HOOD)': ['PLTR', 'RKLB', 'PGY', 'HIMS', 'HOOD'],
    'Speculative (COIN,IONQ,JOBY,PL,TEM)': ['COIN', 'IONQ', 'JOBY', 'PL', 'TEM'],
}

print()
print("=" * 110)
print("  카테고리별 적중률")
print("=" * 110)

for cat_name, cat_tickers in categories.items():
    cat_results = [r for r in all_results if r['ticker'] in cat_tickers]
    if not cat_results:
        continue
    cp30 = sum(r['pos_30'] for r in cat_results)
    cn30 = sum(r['n30'] for r in cat_results)
    cp60 = sum(r['pos_60'] for r in cat_results)
    cn60 = sum(r['n60'] for r in cat_results)
    cp90 = sum(r['pos_90'] for r in cat_results)
    cn90 = sum(r['n90'] for r in cat_results)
    cb = sum(r['n_buys'] for r in cat_results)

    chr30 = cp30/cn30*100 if cn30 else 0
    chr60 = cp60/cn60*100 if cn60 else 0
    chr90 = cp90/cn90*100 if cn90 else 0

    cavg30 = np.mean([r['avg_r30'] for r in cat_results])
    cavg60 = np.mean([r['avg_r60'] for r in cat_results])
    cavg90 = np.mean([r['avg_r90'] for r in cat_results])

    print(f"  {cat_name:<42s} Buys={cb:>3d}  "
          f"30d={cp30}/{cn30}({chr30:.0f}%) avg{cavg30:>+.1f}%  "
          f"60d={cp60}/{cn60}({chr60:.0f}%) avg{cavg60:>+.1f}%  "
          f"90d={cp90}/{cn90}({chr90:.0f}%) avg{cavg90:>+.1f}%")

# ── Per-ticker buy signal details (top 5 best and worst) ──
valid_details = [d for d in all_buy_details if d['r90'] is not None]
sorted_by_90 = sorted(valid_details, key=lambda x: x['r90'], reverse=True)

print()
print("=" * 110)
print("  V4 매수 신호 TOP 10 (90일 수익률 기준)")
print("-" * 90)
print(f"  {'Ticker':<8s} {'Date':<12s} {'Price':>10s} {'30d':>8s} {'60d':>8s} {'90d':>8s}")
for d in sorted_by_90[:10]:
    print(f"  {d['ticker']:<8s} {d['date']:<12s} ${d['price']:>8.2f} {d['r30']:>+7.1f}% {d['r60']:>+7.1f}% {d['r90']:>+7.1f}%")

print()
print("  V4 매수 신호 WORST 10 (90일 수익률 기준)")
print("-" * 90)
print(f"  {'Ticker':<8s} {'Date':<12s} {'Price':>10s} {'30d':>8s} {'60d':>8s} {'90d':>8s}")
for d in sorted_by_90[-10:]:
    print(f"  {d['ticker']:<8s} {d['date']:<12s} ${d['price']:>8.2f} {d['r30']:>+7.1f}% {d['r60']:>+7.1f}% {d['r90']:>+7.1f}%")

print()
print("Done.")
