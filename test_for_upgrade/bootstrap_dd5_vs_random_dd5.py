"""V4+dd5% vs 무작위+dd5% vs 무작위 전체: V4의 진짜 기여도 분리

3가지 비교:
  A) 무작위 전체: 아무 날이나 매수 (기존 baseline)
  B) 무작위+dd5%: dd>=5%인 날 중 무작위 매수 (dd5% 조건만의 효과)
  C) V4+dd5%: V4 신호 + dd>=5% (V4의 추가 기여)

핵심 질문: C가 B보다 나은가? → V4가 dd5% 위에 추가 가치를 주는가?
"""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from real_market_backtest import (
    calc_v4_score, detect_signal_events, build_price_filter,
    smooth_earnings_volume,
)

BUY_CONFIRM = 3
LATE_SELL_DROP_TH = 0.05
DIVGATE_DAYS = 3
N_BOOTSTRAP = 10000
DD_TH = 0.05

TICKERS = ['TSLA','PLTR','NVDA','AVGO','AMZN','GOOGL','JOBY','HIMS',
           'TEM','RKLB','PGY','COIN','HOOD','IONQ','PL','QQQ','VOO']


def download_max(ticker):
    t = yf.Ticker(ticker)
    df = t.history(period='max', auto_adjust=True)
    if df is None or len(df) == 0:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


print('=' * 130)
print('  V4+dd5% vs 무작위+dd5% vs 무작위전체: V4의 진짜 기여도 분리')
print('=' * 130)
print('  A) 무작위 전체: 모든 거래일에서 매수')
print('  B) 무작위+dd5%: 20일 고점 대비 5%+ 하락한 날에서만 매수')
print('  C) V4+dd5%: V4 신호 + dd>=5% 조건')
print('  질문: C-B > 0이면 V4가 dd5% 위에 추가 가치 있음')
print('=' * 130)
print()

# 데이터 수집
ticker_info = {}

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

    # V4 필터된 이벤트
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

    # V4+dd5% 매수 인덱스
    v4_dd5_indices = []
    for ev in filtered:
        if ev['type'] != 'bottom':
            continue
        si, ei = ev['start_idx'], ev['end_idx']
        dur = ev['duration']
        ci = si + BUY_CONFIRM - 1
        if ci <= ei and dur >= BUY_CONFIRM and ci < n:
            if rh20[ci] > 0 and (rh20[ci] - close[ci]) / rh20[ci] >= DD_TH:
                v4_dd5_indices.append(ci)
    v4_dd5_indices = sorted(set(v4_dd5_indices))

    # 무작위+dd5% 가능 인덱스: dd>=5%인 모든 거래일
    dd5_all_indices = []
    for i in range(n):
        if rh20[i] > 0 and (rh20[i] - close[i]) / rh20[i] >= DD_TH:
            dd5_all_indices.append(i)

    td = {}
    for horizon in [30, 60, 90]:
        # A) 모든 거래일 forward return
        all_idx = np.arange(n - horizon)
        all_rets = ((close[all_idx + horizon] / close[all_idx]) - 1) * 100

        # B) dd5% 날들의 forward return
        dd5_rets = []
        for idx in dd5_all_indices:
            if idx + horizon < n:
                dd5_rets.append(((close[idx + horizon] / close[idx]) - 1) * 100)
        dd5_rets = np.array(dd5_rets)

        # C) V4+dd5% forward return
        v4_dd5_rets = []
        for idx in v4_dd5_indices:
            if idx + horizon < n:
                v4_dd5_rets.append(((close[idx + horizon] / close[idx]) - 1) * 100)
        v4_dd5_rets = np.array(v4_dd5_rets)

        td[horizon] = {
            'all_rets': all_rets,
            'dd5_rets': dd5_rets,
            'v4_dd5_rets': v4_dd5_rets,
        }

    ticker_info[tk] = td
    print(f'전체일={n}  dd5%일={len(dd5_all_indices)}({len(dd5_all_indices)/n*100:.0f}%)  V4+dd5%={len(v4_dd5_indices)}')

print()


# ═══════════════════════════════════════════════════════
# [1] 종목별 90일: 3가지 비교
# ═══════════════════════════════════════════════════════
print('=' * 130)
print('  [1] 종목별 90일 평균수익률 비교: A(무작위전체) vs B(무작위+dd5%) vs C(V4+dd5%)')
print('=' * 130)
print(f"  {'Ticker':<7s} | {'A:무작위전체':>13s} {'N':>6s} | {'B:무작위+dd5%':>14s} {'N':>6s} | {'C:V4+dd5%':>12s} {'N':>5s} | {'C-A':>7s} {'C-B':>7s} | V4 추가기여")
print(f'  {"-"*120}')

results_90 = []
for tk in TICKERS:
    if tk not in ticker_info:
        continue
    td = ticker_info[tk][90]

    a_mean = np.mean(td['all_rets'])
    a_n = len(td['all_rets'])
    b_mean = np.mean(td['dd5_rets']) if len(td['dd5_rets']) > 0 else 0
    b_n = len(td['dd5_rets'])

    if len(td['v4_dd5_rets']) >= 3:
        c_mean = np.mean(td['v4_dd5_rets'])
        c_n = len(td['v4_dd5_rets'])
        c_hr = np.mean(td['v4_dd5_rets'] > 0) * 100
    else:
        c_mean = 0
        c_n = len(td['v4_dd5_rets'])
        c_hr = 0

    diff_ca = c_mean - a_mean
    diff_cb = c_mean - b_mean if b_n > 0 else 0

    if c_n >= 3:
        if diff_cb > 3:
            verdict = 'V4 추가가치 O'
        elif diff_cb > 0:
            verdict = 'V4 약간 기여'
        else:
            verdict = 'V4 추가가치 X'
    else:
        verdict = '(신호부족)'

    results_90.append({
        'tk': tk, 'a_mean': a_mean, 'b_mean': b_mean, 'c_mean': c_mean,
        'a_n': a_n, 'b_n': b_n, 'c_n': c_n, 'diff_cb': diff_cb,
    })

    print(f"  {tk:<7s} | {a_mean:>+12.1f}% {a_n:>6d} | {b_mean:>+13.1f}% {b_n:>6d} |"
          f" {c_mean:>+11.1f}% {c_n:>5d} | {diff_ca:>+6.1f}% {diff_cb:>+6.1f}% | {verdict}")

print(f'  {"-"*120}')

# 전체 평균
a_avg = np.mean([r['a_mean'] for r in results_90])
b_avg = np.mean([r['b_mean'] for r in results_90 if r['b_n'] > 0])
c_avg = np.mean([r['c_mean'] for r in results_90 if r['c_n'] >= 3])
print(f"  {'AVG':<7s} | {a_avg:>+12.1f}% {'':>6s} | {b_avg:>+13.1f}% {'':>6s} |"
      f" {c_avg:>+11.1f}% {'':>5s} | {c_avg-a_avg:>+6.1f}% {c_avg-b_avg:>+6.1f}% |")
print()


# ═══════════════════════════════════════════════════════
# [2] 부트스트랩: V4+dd5%(C) vs 무작위+dd5%(B) 풀에서 추출
# ═══════════════════════════════════════════════════════
print('=' * 130)
print('  [2] 핵심 부트스트랩: V4+dd5% vs 무작위+dd5% 풀에서 N개 추출')
print('  (dd5% 조건을 통제한 상태에서 V4가 추가 가치를 주는가?)')
print('=' * 130)
print()

rng = np.random.default_rng(42)

for horizon in [30, 60, 90]:
    # 전 종목 합산
    all_v4_dd5 = []
    all_dd5_pool = []
    all_any_pool = []

    for tk in ticker_info:
        td = ticker_info[tk][horizon]
        all_v4_dd5.extend(td['v4_dd5_rets'].tolist())
        all_dd5_pool.extend(td['dd5_rets'].tolist())
        all_any_pool.extend(td['all_rets'].tolist())

    v4_arr = np.array(all_v4_dd5)
    dd5_pool = np.array(all_dd5_pool)
    any_pool = np.array(all_any_pool)
    n_v4 = len(v4_arr)

    if n_v4 < 5:
        continue

    v4_mean = np.mean(v4_arr)
    v4_hr = np.mean(v4_arr > 0) * 100

    # 부트스트랩 1: V4+dd5% vs 무작위 전체
    boot_any = np.zeros(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        boot_any[i] = np.mean(rng.choice(any_pool, size=n_v4, replace=True))
    p_vs_any = np.mean(boot_any >= v4_mean)
    pctile_vs_any = np.mean(boot_any < v4_mean) * 100

    # 부트스트랩 2: V4+dd5% vs 무작위+dd5% (핵심!)
    boot_dd5 = np.zeros(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        boot_dd5[i] = np.mean(rng.choice(dd5_pool, size=n_v4, replace=True))
    p_vs_dd5 = np.mean(boot_dd5 >= v4_mean)
    pctile_vs_dd5 = np.mean(boot_dd5 < v4_mean) * 100

    dd5_pool_mean = np.mean(dd5_pool)
    dd5_pool_hr = np.mean(dd5_pool > 0) * 100

    sig_any = '***' if p_vs_any < 0.01 else '**' if p_vs_any < 0.05 else '*' if p_vs_any < 0.10 else 'n.s.'
    sig_dd5 = '***' if p_vs_dd5 < 0.01 else '**' if p_vs_dd5 < 0.05 else '*' if p_vs_dd5 < 0.10 else 'n.s.'

    print(f'  {horizon}일 보유 (V4+dd5% {n_v4}건):')
    print(f'    V4+dd5% 평균: {v4_mean:>+.2f}%  적중률: {v4_hr:.1f}%')
    print()
    print(f'    vs 무작위 전체 ({len(any_pool):,}건, 평균 {np.mean(any_pool):+.2f}%):')
    print(f'      부트 백분위: {pctile_vs_any:.1f}%  p={p_vs_any:.4f} {sig_any}')
    print(f'      부트 95% CI: [{np.percentile(boot_any, 2.5):+.2f}%, {np.percentile(boot_any, 97.5):+.2f}%]')
    print()
    print(f'    vs 무작위+dd5% ({len(dd5_pool):,}건, 평균 {dd5_pool_mean:+.2f}%, 적중률 {dd5_pool_hr:.1f}%):')
    print(f'      부트 백분위: {pctile_vs_dd5:.1f}%  p={p_vs_dd5:.4f} {sig_dd5}')
    print(f'      부트 95% CI: [{np.percentile(boot_dd5, 2.5):+.2f}%, {np.percentile(boot_dd5, 97.5):+.2f}%]')
    print()


# ═══════════════════════════════════════════════════════
# [3] 종목별 부트스트랩: V4+dd5% vs 무작위+dd5%
# ═══════════════════════════════════════════════════════
print('=' * 130)
print('  [3] 종목별 부트스트랩: V4+dd5% vs 무작위+dd5% (90일)')
print('  (dd5% 조건 통제 후 V4의 순수 기여)')
print('=' * 130)
print(f"  {'Ticker':<7s} | {'V4+dd5%':>5s} {'V4평균':>8s} {'적중률':>6s} | {'dd5%풀':>6s} {'풀평균':>8s} {'풀적중':>6s}"
      f" | {'pctile':>7s} {'p-val':>7s}  | 판정")
print(f'  {"-"*105}')

tk_results = []
for tk in TICKERS:
    if tk not in ticker_info:
        continue
    td = ticker_info[tk][90]

    if len(td['v4_dd5_rets']) < 3 or len(td['dd5_rets']) < 10:
        continue

    v4_arr = td['v4_dd5_rets']
    dd5_pool = td['dd5_rets']
    v4_mean = np.mean(v4_arr)
    v4_hr = np.mean(v4_arr > 0) * 100
    pool_mean = np.mean(dd5_pool)
    pool_hr = np.mean(dd5_pool > 0) * 100
    n_v4 = len(v4_arr)

    boot = np.zeros(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        boot[i] = np.mean(rng.choice(dd5_pool, size=n_v4, replace=True))
    p = np.mean(boot >= v4_mean)
    pctile = np.mean(boot < v4_mean) * 100

    sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''

    if pctile >= 90:
        verdict = 'V4 유의 추가가치'
    elif pctile >= 70:
        verdict = 'V4 다소 기여'
    elif pctile >= 30:
        verdict = 'V4 무차별'
    else:
        verdict = 'V4 오히려 손해'

    tk_results.append({'tk': tk, 'pctile': pctile, 'p': p, 'v4_mean': v4_mean, 'pool_mean': pool_mean})

    print(f"  {tk:<7s} | {n_v4:>5d} {v4_mean:>+7.1f}% {v4_hr:>5.1f}% | {len(dd5_pool):>6d} {pool_mean:>+7.1f}% {pool_hr:>5.1f}%"
          f" | {pctile:>6.1f}% {p:>6.3f}{sig:<3s} | {verdict}")

print(f'  {"-"*105}')

n_v4_helps = sum(1 for r in tk_results if r['pctile'] >= 50)
n_v4_sig = sum(1 for r in tk_results if r['p'] < 0.05)
n_v4_trend = sum(1 for r in tk_results if r['p'] < 0.10)
avg_pctile = np.mean([r['pctile'] for r in tk_results])
print(f'  V4 기여 양수 (>50%): {n_v4_helps}/{len(tk_results)}종목')
print(f'  V4 유의 (p<0.05): {n_v4_sig}/{len(tk_results)}종목')
print(f'  V4 경향 (p<0.10): {n_v4_trend}/{len(tk_results)}종목')
print(f'  평균 백분위: {avg_pctile:.1f}%')
print()


# ═══════════════════════════════════════════════════════
# [4] 최종 결론
# ═══════════════════════════════════════════════════════
print('=' * 130)
print('  [4] 최종 결론: 수익률 기여 분해')
print('=' * 130)
print()

# 전체 통합 수치
all_any = np.concatenate([ticker_info[tk][90]['all_rets'] for tk in ticker_info])
all_dd5 = np.concatenate([ticker_info[tk][90]['dd5_rets'] for tk in ticker_info if len(ticker_info[tk][90]['dd5_rets']) > 0])
all_v4dd5 = np.concatenate([ticker_info[tk][90]['v4_dd5_rets'] for tk in ticker_info if len(ticker_info[tk][90]['v4_dd5_rets']) > 0])

print(f'  90일 보유 수익률 분해:')
print(f'    A. 무작위 전체:     {np.mean(all_any):>+8.2f}%  (적중률 {np.mean(all_any>0)*100:.1f}%)  N={len(all_any):,}')
print(f'    B. 무작위+dd5%:     {np.mean(all_dd5):>+8.2f}%  (적중률 {np.mean(all_dd5>0)*100:.1f}%)  N={len(all_dd5):,}')
print(f'    C. V4+dd5%:         {np.mean(all_v4dd5):>+8.2f}%  (적중률 {np.mean(all_v4dd5>0)*100:.1f}%)  N={len(all_v4dd5)}')
print()
print(f'    dd5% 조건의 기여 (B-A):  {np.mean(all_dd5)-np.mean(all_any):>+8.2f}%p')
print(f'    V4의 추가 기여 (C-B):    {np.mean(all_v4dd5)-np.mean(all_dd5):>+8.2f}%p')
print(f'    합계 (C-A):              {np.mean(all_v4dd5)-np.mean(all_any):>+8.2f}%p')
print()

dd5_contrib = np.mean(all_dd5) - np.mean(all_any)
v4_contrib = np.mean(all_v4dd5) - np.mean(all_dd5)
total = dd5_contrib + v4_contrib

if total > 0:
    print(f'    기여 비율: dd5% {dd5_contrib/total*100:.0f}% + V4 {v4_contrib/total*100:.0f}% = 100%')
else:
    print(f'    기여 비율: 계산 불가 (총 기여 <= 0)')

print()
print('=' * 130)
print('Done.')
