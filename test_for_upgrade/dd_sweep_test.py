"""Drawdown 파라미터 스윕: 기간(lookback) × 하락률(threshold) 전수조사

2차원 그리드:
  lookback: 5, 10, 15, 20, 30, 40, 50일 고점 기준
  threshold: 1%, 2%, 3%, 5%, 7%, 10%, 15%, 20%

각 조합에 대해:
  1) V4+dd vs 무작위 전체 (p_vs_any)
  2) V4+dd vs 무작위+dd  (p_vs_dd) — V4의 순수 기여
  3) 신호 수, 평균수익률, 적중률
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
HORIZON = 90

LOOKBACKS = [5, 10, 15, 20, 30, 40, 50]
THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]

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
print('  Drawdown 파라미터 스윕: lookback × threshold 전수조사')
print(f'  lookback: {LOOKBACKS}')
print(f'  threshold: {[f"{t*100:.0f}%" for t in THRESHOLDS]}')
print(f'  보유기간: {HORIZON}일 | 부트스트랩: {N_BOOTSTRAP}회')
print('=' * 130)
print()

# ═══════════════════════════════════════════════════════
# 데이터 수집
# ═══════════════════════════════════════════════════════
ticker_cache = {}

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
    rh20_default = df['Close'].rolling(20, min_periods=1).max().values

    # 각 lookback별 rolling high 미리 계산
    rolling_highs = {}
    for lb in LOOKBACKS:
        rolling_highs[lb] = df['Close'].rolling(lb, min_periods=1).max().values

    # V4 필터된 매수 인덱스 (dd 무관)
    filtered = []
    for ev in events:
        if not pf(ev['peak_idx']):
            continue
        if ev['type'] == 'top':
            pidx = ev['peak_idx']
            if rh20_default[pidx] > 0 and (rh20_default[pidx] - close[pidx]) / rh20_default[pidx] > LATE_SELL_DROP_TH:
                continue
        ev['duration'] = ev['end_idx'] - ev['start_idx'] + 1
        filtered.append(ev)

    v4_buy_indices = []
    for ev in filtered:
        if ev['type'] != 'bottom':
            continue
        si, ei = ev['start_idx'], ev['end_idx']
        dur = ev['duration']
        ci = si + BUY_CONFIRM - 1
        if ci <= ei and dur >= BUY_CONFIRM:
            v4_buy_indices.append(ci)
    v4_buy_indices = sorted(set(v4_buy_indices))

    # forward return (전체)
    all_idx = np.arange(n - HORIZON)
    all_fwd = ((close[all_idx + HORIZON] / close[all_idx]) - 1) * 100

    ticker_cache[tk] = {
        'close': close, 'n': n, 'rolling_highs': rolling_highs,
        'v4_buy_indices': v4_buy_indices, 'all_fwd': all_fwd,
    }
    print(f'OK (V4={len(v4_buy_indices)}sig)')

print(f'\n  {len(ticker_cache)}/{len(TICKERS)} tickers loaded\n')


# ═══════════════════════════════════════════════════════
# 스윕 실행
# ═══════════════════════════════════════════════════════
rng = np.random.default_rng(42)

grid_results = []

total_combos = len(LOOKBACKS) * len(THRESHOLDS)
combo_idx = 0

for lb in LOOKBACKS:
    for th in THRESHOLDS:
        combo_idx += 1
        print(f'\r  [{combo_idx}/{total_combos}] lb={lb}d th={th*100:.0f}%...', end='', flush=True)

        all_v4_dd = []
        all_dd_pool = []
        all_any_pool = []

        for tk in ticker_cache:
            tc = ticker_cache[tk]
            close = tc['close']
            n = tc['n']
            rh = tc['rolling_highs'][lb]

            # V4+dd 매수
            for idx in tc['v4_buy_indices']:
                if idx + HORIZON >= n:
                    continue
                if rh[idx] > 0 and (rh[idx] - close[idx]) / rh[idx] >= th:
                    all_v4_dd.append(((close[idx + HORIZON] / close[idx]) - 1) * 100)

            # dd 조건인 모든 날의 forward return
            for i in range(n - HORIZON):
                if rh[i] > 0 and (rh[i] - close[i]) / rh[i] >= th:
                    fwd = ((close[i + HORIZON] / close[i]) - 1) * 100
                    all_dd_pool.append(fwd)

            all_any_pool.extend(tc['all_fwd'].tolist())

        v4_arr = np.array(all_v4_dd)
        dd_pool = np.array(all_dd_pool)
        any_pool = np.array(all_any_pool)
        n_v4 = len(v4_arr)

        if n_v4 < 5:
            grid_results.append({
                'lb': lb, 'th': th, 'n_v4': n_v4, 'n_dd_pool': len(dd_pool),
                'v4_mean': 0, 'v4_hr': 0, 'dd_mean': 0, 'any_mean': np.mean(any_pool),
                'p_vs_any': 1.0, 'p_vs_dd': 1.0,
                'pctile_vs_any': 0, 'pctile_vs_dd': 0,
            })
            continue

        v4_mean = np.mean(v4_arr)
        v4_hr = np.mean(v4_arr > 0) * 100

        # 부트스트랩 vs 전체
        boot_any = np.zeros(N_BOOTSTRAP)
        for i in range(N_BOOTSTRAP):
            boot_any[i] = np.mean(rng.choice(any_pool, size=n_v4, replace=True))
        p_vs_any = float(np.mean(boot_any >= v4_mean))
        pctile_vs_any = float(np.mean(boot_any < v4_mean) * 100)

        # 부트스트랩 vs dd 풀
        if len(dd_pool) >= n_v4:
            boot_dd = np.zeros(N_BOOTSTRAP)
            for i in range(N_BOOTSTRAP):
                boot_dd[i] = np.mean(rng.choice(dd_pool, size=n_v4, replace=True))
            p_vs_dd = float(np.mean(boot_dd >= v4_mean))
            pctile_vs_dd = float(np.mean(boot_dd < v4_mean) * 100)
        else:
            p_vs_dd = 1.0
            pctile_vs_dd = 0

        grid_results.append({
            'lb': lb, 'th': th, 'n_v4': n_v4, 'n_dd_pool': len(dd_pool),
            'v4_mean': v4_mean, 'v4_hr': v4_hr,
            'dd_mean': np.mean(dd_pool) if len(dd_pool) > 0 else 0,
            'any_mean': np.mean(any_pool),
            'p_vs_any': p_vs_any, 'p_vs_dd': p_vs_dd,
            'pctile_vs_any': pctile_vs_any, 'pctile_vs_dd': pctile_vs_dd,
        })

print('\n')


# ═══════════════════════════════════════════════════════
# [1] 히트맵: V4+dd vs 무작위 전체 (p_vs_any)
# ═══════════════════════════════════════════════════════
print('=' * 130)
print('  [1] V4+dd vs 무작위 전체: p-value 히트맵 (90일)')
print('  (값이 작을수록 V4+dd가 유의하게 우세)')
print('=' * 130)

header = f"  {'lb\\th':>6s}"
for th in THRESHOLDS:
    header += f"  {th*100:>5.0f}%"
print(header)
print(f"  {'-'*6}" + f"{'':->8s}" * len(THRESHOLDS))

for lb in LOOKBACKS:
    line = f"  {lb:>4d}d "
    for th in THRESHOLDS:
        r = next((r for r in grid_results if r['lb'] == lb and r['th'] == th), None)
        if r and r['n_v4'] >= 5:
            p = r['p_vs_any']
            sig = '***' if p < 0.01 else '** ' if p < 0.05 else '*  ' if p < 0.10 else '   '
            line += f"  {p:>.3f}{sig}"
        else:
            line += f"  {'N/A':>8s}"
    print(line)

print()


# ═══════════════════════════════════════════════════════
# [2] 히트맵: V4+dd vs 무작위+dd (V4 순수 기여, p_vs_dd)
# ═══════════════════════════════════════════════════════
print('=' * 130)
print('  [2] V4+dd vs 무작위+dd: p-value 히트맵 (90일)')
print('  (V4의 순수 추가 기여. 값이 작을수록 V4가 dd위에 추가가치)')
print('=' * 130)

print(header)
print(f"  {'-'*6}" + f"{'':->8s}" * len(THRESHOLDS))

for lb in LOOKBACKS:
    line = f"  {lb:>4d}d "
    for th in THRESHOLDS:
        r = next((r for r in grid_results if r['lb'] == lb and r['th'] == th), None)
        if r and r['n_v4'] >= 5:
            p = r['p_vs_dd']
            sig = '***' if p < 0.01 else '** ' if p < 0.05 else '*  ' if p < 0.10 else '   '
            line += f"  {p:>.3f}{sig}"
        else:
            line += f"  {'N/A':>8s}"
    print(line)

print()


# ═══════════════════════════════════════════════════════
# [3] 히트맵: 신호 수
# ═══════════════════════════════════════════════════════
print('=' * 130)
print('  [3] 신호 수 히트맵')
print('=' * 130)

print(header)
print(f"  {'-'*6}" + f"{'':->8s}" * len(THRESHOLDS))

for lb in LOOKBACKS:
    line = f"  {lb:>4d}d "
    for th in THRESHOLDS:
        r = next((r for r in grid_results if r['lb'] == lb and r['th'] == th), None)
        if r:
            line += f"  {r['n_v4']:>8d}"
        else:
            line += f"  {'N/A':>8s}"
    print(line)

print()


# ═══════════════════════════════════════════════════════
# [4] 히트맵: V4+dd 평균수익률
# ═══════════════════════════════════════════════════════
print('=' * 130)
print('  [4] V4+dd 평균수익률(%) 히트맵 (90일)')
print('=' * 130)

print(header)
print(f"  {'-'*6}" + f"{'':->8s}" * len(THRESHOLDS))

for lb in LOOKBACKS:
    line = f"  {lb:>4d}d "
    for th in THRESHOLDS:
        r = next((r for r in grid_results if r['lb'] == lb and r['th'] == th), None)
        if r and r['n_v4'] >= 5:
            line += f"  {r['v4_mean']:>+7.1f}%"
        else:
            line += f"  {'N/A':>8s}"
    print(line)

print()


# ═══════════════════════════════════════════════════════
# [5] 히트맵: V4+dd 적중률
# ═══════════════════════════════════════════════════════
print('=' * 130)
print('  [5] V4+dd 적중률(%) 히트맵 (90일)')
print('=' * 130)

print(header)
print(f"  {'-'*6}" + f"{'':->8s}" * len(THRESHOLDS))

for lb in LOOKBACKS:
    line = f"  {lb:>4d}d "
    for th in THRESHOLDS:
        r = next((r for r in grid_results if r['lb'] == lb and r['th'] == th), None)
        if r and r['n_v4'] >= 5:
            line += f"  {r['v4_hr']:>7.1f}%"
        else:
            line += f"  {'N/A':>8s}"
    print(line)

print()


# ═══════════════════════════════════════════════════════
# [6] 종합 스코어 (N × 유의성 × 수익률 밸런스)
# ═══════════════════════════════════════════════════════
print('=' * 130)
print('  [6] 종합 랭킹: 실용성 스코어 = -log10(p_vs_dd) × min(N/50, 1)')
print('  (유의성 높고 신호 수 적절한 조합 = 높은 점수)')
print('=' * 130)

scored = []
for r in grid_results:
    if r['n_v4'] < 5 or r['p_vs_dd'] >= 1.0:
        continue
    import math
    sig_score = -math.log10(max(r['p_vs_dd'], 1e-10))
    n_factor = min(r['n_v4'] / 50, 1.0)  # 50개 이상이면 만점
    practical_score = sig_score * n_factor
    scored.append({**r, 'score': practical_score, 'sig_score': sig_score, 'n_factor': n_factor})

scored.sort(key=lambda x: -x['score'])

print(f"  {'순위':>4s} {'lookback':>8s} {'threshold':>10s} | {'N':>5s} {'V4평균':>8s} {'적중률':>6s}"
      f" | {'p_vs_any':>8s} {'p_vs_dd':>8s} | {'점수':>6s}")
print(f'  {"-"*90}')

for i, r in enumerate(scored[:20]):
    sig_any = '***' if r['p_vs_any'] < 0.01 else '**' if r['p_vs_any'] < 0.05 else '*' if r['p_vs_any'] < 0.10 else ''
    sig_dd = '***' if r['p_vs_dd'] < 0.01 else '**' if r['p_vs_dd'] < 0.05 else '*' if r['p_vs_dd'] < 0.10 else ''
    print(f"  {i+1:>4d} {r['lb']:>6d}d {r['th']*100:>8.0f}%  | {r['n_v4']:>5d} {r['v4_mean']:>+7.1f}% {r['v4_hr']:>5.1f}%"
          f" | {r['p_vs_any']:>7.4f}{sig_any:<3s} {r['p_vs_dd']:>7.4f}{sig_dd:<3s} | {r['score']:>5.2f}")

print()


# ═══════════════════════════════════════════════════════
# [7] 최종 결론
# ═══════════════════════════════════════════════════════
print('=' * 130)
print('  [7] 최종 결론')
print('=' * 130)
print()

# p_vs_any < 0.05인 조합 수
n_sig_any = sum(1 for r in grid_results if r['n_v4'] >= 5 and r['p_vs_any'] < 0.05)
n_sig_dd = sum(1 for r in grid_results if r['n_v4'] >= 5 and r['p_vs_dd'] < 0.05)
n_valid = sum(1 for r in grid_results if r['n_v4'] >= 5)

print(f'  총 조합: {len(LOOKBACKS)} × {len(THRESHOLDS)} = {total_combos}')
print(f'  유효 조합 (N>=5): {n_valid}')
print(f'  vs 무작위 전체 유의 (p<0.05): {n_sig_any}/{n_valid}')
print(f'  vs 무작위+dd 유의 (p<0.05):   {n_sig_dd}/{n_valid}')
print()

if scored:
    best = scored[0]
    print(f'  최적 조합: lookback={best["lb"]}d, threshold={best["th"]*100:.0f}%')
    print(f'    신호 수: {best["n_v4"]}')
    print(f'    V4+dd 평균수익률: {best["v4_mean"]:+.2f}%')
    print(f'    적중률: {best["v4_hr"]:.1f}%')
    print(f'    vs 무작위 전체: p={best["p_vs_any"]:.4f}')
    print(f'    vs 무작위+dd:   p={best["p_vs_dd"]:.4f}')
    print()

    # p_vs_dd < 0.05인 조합 중 N이 가장 큰 것
    sig_dd_results = [r for r in grid_results if r['n_v4'] >= 5 and r['p_vs_dd'] < 0.05]
    if sig_dd_results:
        best_practical = max(sig_dd_results, key=lambda x: x['n_v4'])
        print(f'  실용적 최적 (p_vs_dd<0.05 중 N 최대):')
        print(f'    lookback={best_practical["lb"]}d, threshold={best_practical["th"]*100:.0f}%')
        print(f'    신호 수: {best_practical["n_v4"]}')
        print(f'    V4+dd 평균수익률: {best_practical["v4_mean"]:+.2f}%')
        print(f'    vs 무작위+dd: p={best_practical["p_vs_dd"]:.4f}')

print()
print('=' * 130)
print('Done.')
