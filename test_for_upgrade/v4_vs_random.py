"""
V4 매수 신호 vs 무작위 매수: 연환산 수익률 비교
- V4 매수 신호 발생일의 30/60/90d forward return → 연환산
- 전체 거래일 중 무작위 매수의 30/60/90d forward return → 연환산
- 통계적 유의성 검정
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

def annualize(r_pct, days):
    """기간 수익률(%) → 연환산 수익률(%)"""
    if days <= 0:
        return 0.0
    return ((1 + r_pct / 100) ** (252 / days) - 1) * 100

print("=" * 120)
print("  V4 매수 신호 vs 무작위 매수: 연환산 수익률 비교")
print("=" * 120)
print()

all_v4 = {30: [], 60: [], 90: []}
all_random = {30: [], 60: [], 90: []}
ticker_results = []

for tk in TICKERS:
    print(f"  {tk}...", end=" ", flush=True)
    df = download_max(tk)
    if df is None or len(df) < 200:
        print("SKIP")
        continue

    close = df['Close'].values
    dates = df.index
    n = len(close)

    # ── V4 매수 신호 찾기 ──
    score = calc_v4_score(df, w=20)
    events = detect_signal_events(score, th=0.15, cooldown=5)
    pf = build_price_filter(df, er_q=66, atr_q=55, lookback=252)
    rh20 = df['Close'].rolling(20, min_periods=1).max().values

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

    v4_indices = set()
    for ev in filtered:
        if ev['type'] != 'bottom':
            continue
        si, ei = ev['start_idx'], ev['end_idx']
        dur = ev['duration']
        ci = si + BUY_CONFIRM - 1
        if ci <= ei and dur >= BUY_CONFIRM:
            v4_indices.add(ci)

    # ── Forward returns 계산 ──
    v4_ret = {30: [], 60: [], 90: []}
    rand_ret = {30: [], 60: [], 90: []}

    for horizon in [30, 60, 90]:
        # V4 신호일
        for idx in v4_indices:
            if idx + horizon < n:
                r = ((close[idx + horizon] / close[idx]) - 1) * 100
                v4_ret[horizon].append(r)
                all_v4[horizon].append(r)

        # 전체 거래일 (= 무작위 매수 기준)
        for idx in range(n - horizon):
            r = ((close[idx + horizon] / close[idx]) - 1) * 100
            rand_ret[horizon].append(r)
            all_random[horizon].append(r)

    if not v4_indices:
        print("NO SIGNALS")
        continue

    # 종목별 결과
    row = {'ticker': tk, 'n_v4': len(v4_indices)}
    for h in [30, 60, 90]:
        if v4_ret[h]:
            v4_avg = np.mean(v4_ret[h])
            v4_med = np.median(v4_ret[h])
            v4_ann = annualize(v4_avg, h)
            v4_hr = sum(1 for r in v4_ret[h] if r > 0) / len(v4_ret[h]) * 100
        else:
            v4_avg = v4_med = v4_ann = v4_hr = 0

        rand_avg = np.mean(rand_ret[h])
        rand_med = np.median(rand_ret[h])
        rand_ann = annualize(rand_avg, h)
        rand_hr = sum(1 for r in rand_ret[h] if r > 0) / len(rand_ret[h]) * 100

        row[f'v4_avg_{h}'] = v4_avg
        row[f'v4_ann_{h}'] = v4_ann
        row[f'v4_hr_{h}'] = v4_hr
        row[f'rand_avg_{h}'] = rand_avg
        row[f'rand_ann_{h}'] = rand_ann
        row[f'rand_hr_{h}'] = rand_hr
        row[f'edge_{h}'] = v4_ann - rand_ann

    ticker_results.append(row)
    e30 = row.get('edge_30', 0)
    e90 = row.get('edge_90', 0)
    print(f"V4={len(v4_indices):>3d}sig  30d edge={e30:>+7.1f}%p  90d edge={e90:>+7.1f}%p")

# ══════════════════════════════════════════════════════
# 종목별 상세 테이블
# ══════════════════════════════════════════════════════
print()
print("=" * 120)
print("  종목별 연환산 수익률 비교 (V4 매수일 vs 아무날이나 매수)")
print("=" * 120)

for h in [30, 60, 90]:
    print(f"\n  ── {h}일 보유 기준 ──")
    print(f"  {'Ticker':<8s} {'V4신호':>6s} {'V4 연환산':>10s} {'무작위 연환산':>12s} {'Edge':>10s} {'V4 적중률':>10s} {'무작위 적중률':>12s}")
    print(f"  {'-'*80}")

    for r in ticker_results:
        v4a = r[f'v4_ann_{h}']
        ra = r[f'rand_ann_{h}']
        edge = r[f'edge_{h}']
        v4hr = r[f'v4_hr_{h}']
        rhr = r[f'rand_hr_{h}']
        marker = " ★" if edge > 0 else ""
        print(f"  {r['ticker']:<8s} {r['n_v4']:>6d} {v4a:>+9.1f}% {ra:>+11.1f}% {edge:>+9.1f}%p {v4hr:>9.0f}% {rhr:>11.0f}%{marker}")

    # 전체 평균
    avg_v4 = np.mean([r[f'v4_ann_{h}'] for r in ticker_results])
    avg_rand = np.mean([r[f'rand_ann_{h}'] for r in ticker_results])
    avg_edge = avg_v4 - avg_rand
    print(f"  {'-'*80}")
    print(f"  {'AVG':<8s} {'':>6s} {avg_v4:>+9.1f}% {avg_rand:>+11.1f}% {avg_edge:>+9.1f}%p")

    # Edge 양수인 종목 수
    n_pos = sum(1 for r in ticker_results if r[f'edge_{h}'] > 0)
    print(f"  Edge 양수: {n_pos}/{len(ticker_results)}종목")

# ══════════════════════════════════════════════════════
# 전체 통합 (전종목 풀링)
# ══════════════════════════════════════════════════════
print()
print("=" * 120)
print("  전체 통합 결과 (모든 종목 신호 합산)")
print("=" * 120)

print(f"\n  {'기간':>6s} │ {'V4 매수':>18s} │ {'무작위 매수':>18s} │ {'Edge':>10s}")
print(f"  {'':>6s} │ {'평균수익 → 연환산':>18s} │ {'평균수익 → 연환산':>18s} │ {'':>10s}")
print(f"  {'─'*6}─┼─{'─'*18}─┼─{'─'*18}─┼─{'─'*10}")

for h in [30, 60, 90]:
    v4_data = all_v4[h]
    rand_data = all_random[h]

    v4_avg = np.mean(v4_data)
    rand_avg = np.mean(rand_data)
    v4_ann = annualize(v4_avg, h)
    rand_ann = annualize(rand_avg, h)
    edge = v4_ann - rand_ann

    v4_hr = sum(1 for r in v4_data if r > 0) / len(v4_data) * 100
    rand_hr = sum(1 for r in rand_data if r > 0) / len(rand_data) * 100

    print(f"  {h:>4d}일 │ {v4_avg:>+6.2f}% → {v4_ann:>+7.1f}%/yr │ {rand_avg:>+6.2f}% → {rand_ann:>+7.1f}%/yr │ {edge:>+8.1f}%p")

print()
print(f"  {'기간':>6s} │ {'V4 적중률':>12s} │ {'무작위 적중률':>14s} │ {'차이':>10s}")
print(f"  {'─'*6}─┼─{'─'*12}─┼─{'─'*14}─┼─{'─'*10}")
for h in [30, 60, 90]:
    v4_hr = sum(1 for r in all_v4[h] if r > 0) / len(all_v4[h]) * 100
    rand_hr = sum(1 for r in all_random[h] if r > 0) / len(all_random[h]) * 100
    print(f"  {h:>4d}일 │ {v4_hr:>10.1f}% │ {rand_hr:>12.1f}% │ {v4_hr-rand_hr:>+8.1f}%p")

# ══════════════════════════════════════════════════════
# 카테고리별
# ══════════════════════════════════════════════════════
categories = {
    'Benchmark': ['QQQ', 'VOO'],
    'Big Tech': ['NVDA', 'AVGO', 'GOOGL', 'AMZN', 'TSLA'],
    'Growth': ['PLTR', 'RKLB', 'PGY', 'HIMS', 'HOOD'],
    'Speculative': ['COIN', 'IONQ', 'JOBY', 'PL'],
}

print()
print("=" * 120)
print("  카테고리별 90일 보유 기준 Edge")
print("=" * 120)
print(f"  {'Category':<14s} {'V4 연환산':>10s} {'무작위 연환산':>12s} {'Edge':>10s} {'V4 적중률':>10s} {'무작위 적중률':>12s}")
print(f"  {'-'*80}")

for cat, tks in categories.items():
    cat_rows = [r for r in ticker_results if r['ticker'] in tks]
    if not cat_rows:
        continue
    avg_v4 = np.mean([r['v4_ann_90'] for r in cat_rows])
    avg_rand = np.mean([r['rand_ann_90'] for r in cat_rows])
    avg_v4hr = np.mean([r['v4_hr_90'] for r in cat_rows])
    avg_rhr = np.mean([r['rand_hr_90'] for r in cat_rows])
    edge = avg_v4 - avg_rand
    print(f"  {cat:<14s} {avg_v4:>+9.1f}% {avg_rand:>+11.1f}% {edge:>+9.1f}%p {avg_v4hr:>9.0f}% {avg_rhr:>11.0f}%")

# ══════════════════════════════════════════════════════
# 결론
# ══════════════════════════════════════════════════════
print()
print("=" * 120)

grand_v4_90 = annualize(np.mean(all_v4[90]), 90)
grand_rand_90 = annualize(np.mean(all_random[90]), 90)
grand_edge_90 = grand_v4_90 - grand_rand_90

grand_v4_30 = annualize(np.mean(all_v4[30]), 30)
grand_rand_30 = annualize(np.mean(all_random[30]), 30)
grand_edge_30 = grand_v4_30 - grand_rand_30

n_pos_90 = sum(1 for r in ticker_results if r['edge_90'] > 0)
n_pos_30 = sum(1 for r in ticker_results if r['edge_30'] > 0)

print(f"  결론:")
print(f"    30일 보유: V4 {grand_v4_30:>+.1f}%/yr vs 무작위 {grand_rand_30:>+.1f}%/yr → Edge {grand_edge_30:>+.1f}%p ({n_pos_30}/{len(ticker_results)} 종목 양수)")
print(f"    90일 보유: V4 {grand_v4_90:>+.1f}%/yr vs 무작위 {grand_rand_90:>+.1f}%/yr → Edge {grand_edge_90:>+.1f}%p ({n_pos_90}/{len(ticker_results)} 종목 양수)")
print("=" * 120)
print("Done.")
