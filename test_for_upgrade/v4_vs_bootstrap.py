"""V4 매수 vs 무작위 부트스트랩 매수: 공정한 비교

핵심: V4가 N번 매수했으면, 무작위도 N번씩 10,000회 반복 추출해서 비교.
"V4의 N번 매수 수익률이 무작위 N번 매수보다 나은가?"
"""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

import numpy as np
import yfinance as yf
import pandas as pd
from real_market_backtest import calc_v4_score, detect_signal_events, build_price_filter, smooth_earnings_volume

BUY_CONFIRM = 3
LATE_SELL_DROP_TH = 0.05
DIVGATE_DAYS = 3
N_BOOTSTRAP = 10000

TICKERS = ['TSLA','PLTR','NVDA','AVGO','AMZN','GOOGL','JOBY','HIMS',
           'TEM','RKLB','PGY','COIN','HOOD','IONQ','PL','QQQ','VOO']

def download_max(ticker):
    t = yf.Ticker(ticker)
    df = t.history(period='max', auto_adjust=True)
    if df is None or len(df)==0: return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

print('=' * 110)
print('  V4 매수 vs 무작위 부트스트랩 (10,000회 반복)')
print('  방법: V4가 N번 매수 -> 무작위도 N번씩 10,000세트 추출 -> 평균수익률 분포와 비교')
print('=' * 110)
print()

results = []

# 전체 통합용
all_v4_returns = {30: [], 60: [], 90: []}
all_possible_returns = {30: [], 60: [], 90: []}
all_v4_indices_info = []  # (ticker, indices, close, n)

for tk in TICKERS:
    print(f'  {tk}...', end=' ', flush=True)
    df = download_max(tk)
    if df is None or len(df) < 200:
        print('SKIP'); continue

    df_smooth = smooth_earnings_volume(df, ticker=tk)
    score = calc_v4_score(df_smooth, w=20, divgate_days=DIVGATE_DAYS)
    events = detect_signal_events(score, th=0.15, cooldown=5)
    pf = build_price_filter(df_smooth, er_q=66, atr_q=55, lookback=252)

    close = df['Close'].values
    n = len(close)
    rh20 = df['Close'].rolling(20, min_periods=1).max().values

    filtered = []
    for ev in events:
        if not pf(ev['peak_idx']): continue
        if ev['type'] == 'top':
            pidx = ev['peak_idx']
            if rh20[pidx]>0 and (rh20[pidx]-close[pidx])/rh20[pidx] > LATE_SELL_DROP_TH:
                continue
        ev['duration'] = ev['end_idx']-ev['start_idx']+1
        filtered.append(ev)

    v4_indices = []
    for ev in filtered:
        if ev['type']!='bottom': continue
        si, ei = ev['start_idx'], ev['end_idx']
        dur = ev['duration']
        ci = si + BUY_CONFIRM - 1
        if ci <= ei and dur >= BUY_CONFIRM:
            v4_indices.append(ci)
    v4_indices = sorted(set(v4_indices))

    if len(v4_indices) < 3:
        print(f'n={len(v4_indices)} (too few)'); continue

    row = {'tk': tk, 'n_signals': len(v4_indices)}

    for horizon in [30, 60, 90]:
        # V4 forward returns
        v4_valid = [idx for idx in v4_indices if idx + horizon < n]
        if len(v4_valid) < 3:
            continue
        v4_rets = np.array([((close[idx+horizon]/close[idx])-1)*100 for idx in v4_valid])
        v4_mean = np.mean(v4_rets)
        v4_median = np.median(v4_rets)
        v4_hr = np.mean(v4_rets > 0) * 100
        n_v4 = len(v4_valid)

        all_v4_returns[horizon].extend(v4_rets.tolist())

        # 무작위 가능한 모든 인덱스의 forward return
        all_idx = np.arange(n - horizon)
        all_rets = ((close[all_idx + horizon] / close[all_idx]) - 1) * 100

        all_possible_returns[horizon].extend(all_rets.tolist())

        # 부트스트랩: n_v4개씩 10,000번 무작위 추출
        rng = np.random.default_rng(42)
        boot_means = np.zeros(N_BOOTSTRAP)
        boot_medians = np.zeros(N_BOOTSTRAP)
        boot_hrs = np.zeros(N_BOOTSTRAP)
        for i in range(N_BOOTSTRAP):
            sample = rng.choice(all_rets, size=n_v4, replace=True)
            boot_means[i] = np.mean(sample)
            boot_medians[i] = np.median(sample)
            boot_hrs[i] = np.mean(sample > 0) * 100

        # p-value: V4 평균이 부트스트랩 분포에서 어디에 위치하는지
        # "무작위 N번 매수의 평균이 V4만큼 혹은 더 높을 확률"
        p_mean = np.mean(boot_means >= v4_mean)
        p_median = np.mean(boot_medians >= v4_median)
        p_hr = np.mean(boot_hrs >= v4_hr)

        # 백분위
        percentile_mean = np.mean(boot_means < v4_mean) * 100
        percentile_hr = np.mean(boot_hrs < v4_hr) * 100

        row[f'n_{horizon}'] = n_v4
        row[f'v4_mean_{horizon}'] = v4_mean
        row[f'v4_med_{horizon}'] = v4_median
        row[f'v4_hr_{horizon}'] = v4_hr
        row[f'boot_mean_{horizon}'] = np.mean(boot_means)
        row[f'boot_med_{horizon}'] = np.mean(boot_medians)
        row[f'boot_hr_{horizon}'] = np.mean(boot_hrs)
        row[f'p_mean_{horizon}'] = p_mean
        row[f'p_hr_{horizon}'] = p_hr
        row[f'pctile_mean_{horizon}'] = percentile_mean
        row[f'pctile_hr_{horizon}'] = percentile_hr

    results.append(row)
    # 간단 출력
    if f'p_mean_90' in row:
        pct = row.get('pctile_mean_90', 0)
        p = row.get('p_mean_90', 1)
        sig = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.10 else ''
        print(f"n={row['n_signals']:>3d}  V4={row['v4_mean_90']:>+7.1f}%  무작위={row['boot_mean_90']:>+7.1f}%  상위{100-pct:.0f}%  p={p:.3f} {sig}")
    else:
        print('insufficient data')


# ================================================================
# 종목별 상세 (90일)
# ================================================================
print()
print('=' * 110)
print('  [1] 종목별 부트스트랩 비교 (90일 보유)')
print('  V4의 평균수익률이 무작위 10,000세트 중 몇 %보다 높은지 (백분위)')
print('=' * 110)
print(f"  {'Ticker':<7s} {'N':>4s} | {'V4 평균':>8s} {'부트 평균':>9s} {'백분위':>7s} {'p-val':>7s} | {'V4 적중':>7s} {'부트 적중':>9s} {'백분위':>7s} {'p-val':>7s} | 판정")
print(f'  {"-"*105}')

for r in sorted(results, key=lambda x: -x.get('pctile_mean_90', 0)):
    if f'p_mean_90' not in r: continue
    vm = r['v4_mean_90']
    bm = r['boot_mean_90']
    pm = r['pctile_mean_90']
    pvm = r['p_mean_90']
    vh = r['v4_hr_90']
    bh = r['boot_hr_90']
    ph = r['pctile_hr_90']
    pvh = r['p_hr_90']

    if pm >= 95: verdict = 'V4 유의 우세'
    elif pm >= 75: verdict = 'V4 다소 우세'
    elif pm >= 25: verdict = '무차별'
    else: verdict = 'V4 열세'

    sig_m = '***' if pvm<0.01 else '**' if pvm<0.05 else '*' if pvm<0.10 else ''
    sig_h = '***' if pvh<0.01 else '**' if pvh<0.05 else '*' if pvh<0.10 else ''

    print(f"  {r['tk']:<7s} {r['n_90']:>4d} | {vm:>+7.1f}% {bm:>+8.1f}% {pm:>6.1f}% {pvm:>6.3f}{sig_m:<3s} | {vh:>6.1f}% {bh:>8.1f}% {ph:>6.1f}% {pvh:>6.3f}{sig_h:<3s} | {verdict}")

n_sig_win = sum(1 for r in results if r.get('p_mean_90', 1) < 0.05 and r.get('v4_mean_90', 0) > r.get('boot_mean_90', 0))
n_sig_lose = sum(1 for r in results if r.get('p_mean_90', 1) < 0.05 and r.get('v4_mean_90', 0) <= r.get('boot_mean_90', 0))
n_upper = sum(1 for r in results if r.get('pctile_mean_90', 50) >= 75)
n_lower = sum(1 for r in results if r.get('pctile_mean_90', 50) < 25)
n_total = sum(1 for r in results if 'p_mean_90' in r)
print(f'  {"-"*105}')
print(f'  p<0.05 유의: V4우세 {n_sig_win}개, V4열세 {n_sig_lose}개, 무차별 {n_total-n_sig_win-n_sig_lose}개')
print(f'  상위 25% 이상: {n_upper}/{n_total}종목  |  하위 25%: {n_lower}/{n_total}종목')


# ================================================================
# 30일, 60일
# ================================================================
for horizon in [30, 60]:
    print()
    print(f'  -- {horizon}일 보유 기준 --')
    print(f"  {'Ticker':<7s} {'N':>4s} | {'V4 평균':>8s} {'부트':>8s} {'백분위':>7s} {'p':>6s} | {'V4 적중':>7s} {'부트':>7s} {'백분위':>7s} | 판정")
    for r in sorted(results, key=lambda x: -x.get(f'pctile_mean_{horizon}', 0)):
        k = f'p_mean_{horizon}'
        if k not in r: continue
        vm = r[f'v4_mean_{horizon}']
        bm = r[f'boot_mean_{horizon}']
        pm = r[f'pctile_mean_{horizon}']
        pv = r[f'p_mean_{horizon}']
        vh = r[f'v4_hr_{horizon}']
        bh = r[f'boot_hr_{horizon}']
        ph = r[f'pctile_hr_{horizon}']
        if pm >= 95: verdict = 'V4 유의 우세'
        elif pm >= 75: verdict = 'V4 다소 우세'
        elif pm >= 25: verdict = '무차별'
        else: verdict = 'V4 열세'
        sig = '***' if pv<0.01 else '**' if pv<0.05 else '*' if pv<0.10 else ''
        print(f"  {r['tk']:<7s} {r.get(f'n_{horizon}',0):>4d} | {vm:>+7.1f}% {bm:>+7.1f}% {pm:>6.1f}% {pv:>5.3f}{sig:<3s} | {vh:>6.1f}% {bh:>6.1f}% {ph:>6.1f}% | {verdict}")


# ================================================================
# 전체 통합 부트스트랩
# ================================================================
print()
print('=' * 110)
print('  [2] 전체 통합 부트스트랩 (전 종목 합산)')
print('=' * 110)

rng = np.random.default_rng(42)

for horizon in [30, 60, 90]:
    v4_arr = np.array(all_v4_returns[horizon])
    all_arr = np.array(all_possible_returns[horizon])
    n_v4 = len(v4_arr)

    if n_v4 < 5:
        continue

    v4_mean = np.mean(v4_arr)
    v4_med = np.median(v4_arr)
    v4_hr = np.mean(v4_arr > 0) * 100

    boot_means = np.zeros(N_BOOTSTRAP)
    boot_meds = np.zeros(N_BOOTSTRAP)
    boot_hrs = np.zeros(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        sample = rng.choice(all_arr, size=n_v4, replace=True)
        boot_means[i] = np.mean(sample)
        boot_meds[i] = np.median(sample)
        boot_hrs[i] = np.mean(sample > 0) * 100

    p_mean = np.mean(boot_means >= v4_mean)
    p_med = np.mean(boot_meds >= v4_med)
    p_hr = np.mean(boot_hrs >= v4_hr)
    pctile_mean = np.mean(boot_means < v4_mean) * 100
    pctile_hr = np.mean(boot_hrs < v4_hr) * 100

    sig_m = '***' if p_mean<0.01 else '**' if p_mean<0.05 else '*' if p_mean<0.10 else 'n.s.'
    sig_h = '***' if p_hr<0.01 else '**' if p_hr<0.05 else '*' if p_hr<0.10 else 'n.s.'

    print(f'  {horizon}일 보유 (V4 {n_v4}건):')
    print(f'    V4 평균: {v4_mean:>+.2f}%   무작위 부트 평균: {np.mean(boot_means):>+.2f}%')
    print(f'    V4 백분위: {pctile_mean:.1f}%  (상위 {100-pctile_mean:.1f}%)  p={p_mean:.4f} {sig_m}')
    print(f'    V4 적중률: {v4_hr:.1f}%   무작위 부트: {np.mean(boot_hrs):.1f}%')
    print(f'    적중률 백분위: {pctile_hr:.1f}%  p={p_hr:.4f} {sig_h}')
    print(f'    부트 평균 95% CI: [{np.percentile(boot_means, 2.5):>+.2f}%, {np.percentile(boot_means, 97.5):>+.2f}%]')
    v4_in_ci = np.percentile(boot_means, 2.5) <= v4_mean <= np.percentile(boot_means, 97.5)
    print(f'    V4 평균이 95% CI 안에 있는가: {"YES (무작위와 구별 불가)" if v4_in_ci else "NO (유의한 차이)"}')
    print()


# ================================================================
# 최종 판정
# ================================================================
print('=' * 110)
print('  [3] 최종 판정')
print('=' * 110)

v4_90 = np.array(all_v4_returns[90])
all_90 = np.array(all_possible_returns[90])
n_v4_90 = len(v4_90)
v4_mean_90 = np.mean(v4_90)

boot_90 = np.zeros(N_BOOTSTRAP)
for i in range(N_BOOTSTRAP):
    boot_90[i] = np.mean(rng.choice(all_90, size=n_v4_90, replace=True))

p_final = np.mean(boot_90 >= v4_mean_90)
pctile_final = np.mean(boot_90 < v4_mean_90) * 100

print()
print(f'  V4 매수 544건의 90일 평균수익률: {v4_mean_90:+.2f}%')
print(f'  무작위 544건 x 10,000세트의 평균수익률 분포:')
print(f'    평균: {np.mean(boot_90):+.2f}%')
print(f'    표준편차: {np.std(boot_90):.2f}%')
print(f'    5%~95% 범위: [{np.percentile(boot_90, 5):+.2f}%, {np.percentile(boot_90, 95):+.2f}%]')
print()
print(f'  V4의 위치: 상위 {100-pctile_final:.1f}% (백분위 {pctile_final:.1f}%)')
print(f'  p-value: {p_final:.4f}')
print()

if p_final < 0.01:
    print('  ==> V4 매수 타이밍이 무작위보다 유의하게 우수 (p<0.01)')
elif p_final < 0.05:
    print('  ==> V4 매수 타이밍이 무작위보다 유의하게 우수 (p<0.05)')
elif p_final < 0.10:
    print('  ==> V4 매수 타이밍이 무작위보다 다소 우수한 경향 (p<0.10)')
elif pctile_final >= 60:
    print('  ==> V4가 무작위보다 약간 나은 경향이 있으나 통계적으로 유의하지 않음')
else:
    print('  ==> V4 매수 타이밍이 무작위와 구별 불가')

print()
print('  해석 가이드:')
print('    백분위 95%+ : V4가 무작위 10,000세트 중 상위 5%에 해당 -> 유의한 우위')
print('    백분위 75~95%: 무작위보다 나은 경향이 있으나 확실하지 않음')
print('    백분위 25~75%: 무작위와 구별 불가')
print('    백분위 25% 미만: 오히려 무작위보다 못함')
print()
print('=' * 110)
