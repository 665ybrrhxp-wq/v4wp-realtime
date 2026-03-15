"""V4 + dd>=5% 부트스트랩 종목별 상세 분석

기존 부트스트랩(전체 신호)과 dd5% 필터 적용 후를 종목별로 비교.
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
DD_TH = 0.05  # 20일 고점 대비 5%

TICKERS = ['TSLA','PLTR','NVDA','AVGO','AMZN','GOOGL','JOBY','HIMS',
           'TEM','RKLB','PGY','COIN','HOOD','IONQ','PL','QQQ','VOO']


def download_max(ticker):
    t = yf.Ticker(ticker)
    df = t.history(period='max', auto_adjust=True)
    if df is None or len(df) == 0:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def bootstrap(v4_rets, all_rets, n_boot=N_BOOTSTRAP, seed=42):
    if len(v4_rets) < 3:
        return None
    rng = np.random.default_rng(seed)
    v4_mean = np.mean(v4_rets)
    v4_hr = np.mean(np.array(v4_rets) > 0) * 100
    n = len(v4_rets)
    boot_means = np.zeros(n_boot)
    boot_hrs = np.zeros(n_boot)
    for i in range(n_boot):
        s = rng.choice(all_rets, size=n, replace=True)
        boot_means[i] = np.mean(s)
        boot_hrs[i] = np.mean(s > 0) * 100
    return {
        'n': n,
        'v4_mean': v4_mean,
        'v4_hr': v4_hr,
        'boot_mean': np.mean(boot_means),
        'boot_hr': np.mean(boot_hrs),
        'p_mean': np.mean(boot_means >= v4_mean),
        'p_hr': np.mean(boot_hrs >= v4_hr),
        'pctile_mean': np.mean(boot_means < v4_mean) * 100,
        'pctile_hr': np.mean(boot_hrs < v4_hr) * 100,
        'ci_lo': np.percentile(boot_means, 2.5),
        'ci_hi': np.percentile(boot_means, 97.5),
    }


print('=' * 130)
print('  V4 + dd>=5% 부트스트랩 종목별 상세 (기존 vs dd5% 비교)')
print('=' * 130)
print()

# 데이터 수집
ticker_data = {}  # {tk: {all_rets_h, v4_all_h, v4_dd5_h}}

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
    dates = df.index
    n = len(close)
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

    # 매수 인덱스 수집
    buy_indices_all = []
    buy_indices_dd5 = []
    for ev in filtered:
        if ev['type'] != 'bottom':
            continue
        si, ei = ev['start_idx'], ev['end_idx']
        dur = ev['duration']
        ci = si + BUY_CONFIRM - 1
        if ci <= ei and dur >= BUY_CONFIRM:
            buy_indices_all.append(ci)
            # dd5% 체크
            if ci < n and rh20[ci] > 0:
                dd = (rh20[ci] - close[ci]) / rh20[ci]
                if dd >= DD_TH:
                    buy_indices_dd5.append(ci)

    buy_indices_all = sorted(set(buy_indices_all))
    buy_indices_dd5 = sorted(set(buy_indices_dd5))

    td = {}
    for horizon in [30, 60, 90]:
        # 전체 가능한 forward return
        all_idx = np.arange(n - horizon)
        all_rets = ((close[all_idx + horizon] / close[all_idx]) - 1) * 100

        # V4 전체
        v4_all = []
        for idx in buy_indices_all:
            if idx + horizon < n:
                v4_all.append(((close[idx + horizon] / close[idx]) - 1) * 100)

        # V4 + dd5%
        v4_dd5 = []
        for idx in buy_indices_dd5:
            if idx + horizon < n:
                v4_dd5.append(((close[idx + horizon] / close[idx]) - 1) * 100)

        td[horizon] = {
            'all_rets': all_rets,
            'v4_all': np.array(v4_all),
            'v4_dd5': np.array(v4_dd5),
        }

    ticker_data[tk] = td
    print(f'전체={len(buy_indices_all)}  dd5%={len(buy_indices_dd5)}  ({len(buy_indices_dd5)/max(len(buy_indices_all),1)*100:.0f}% 생존)')

print()

# ═══════════════════════════════════════════════════════
# [1] 종목별 90일 비교: 기존 vs dd5%
# ═══════════════════════════════════════════════════════
print('=' * 130)
print('  [1] 종목별 90일 부트스트랩: 기존 V4 vs V4+dd5%')
print('=' * 130)
print(f"  {'Ticker':<7s} | {'--- 기존 V4 ---':^40s} | {'--- V4 + dd5% ---':^40s} | {'비교'}")
print(f"  {'':7s} | {'N':>4s} {'V4평균':>8s} {'부트':>8s} {'pctile':>7s} {'p':>7s} | {'N':>4s} {'V4평균':>8s} {'부트':>8s} {'pctile':>7s} {'p':>7s} | ")
print(f'  {"-"*125}')

results_90 = []

for tk in TICKERS:
    if tk not in ticker_data:
        continue
    td = ticker_data[tk][90]

    r_all = bootstrap(td['v4_all'], td['all_rets'])
    r_dd5 = bootstrap(td['v4_dd5'], td['all_rets'])

    if r_all is None:
        continue

    sig_all = '***' if r_all['p_mean'] < 0.01 else '**' if r_all['p_mean'] < 0.05 else '*' if r_all['p_mean'] < 0.10 else ''

    if r_dd5 is not None:
        sig_dd5 = '***' if r_dd5['p_mean'] < 0.01 else '**' if r_dd5['p_mean'] < 0.05 else '*' if r_dd5['p_mean'] < 0.10 else ''

        if r_dd5['pctile_mean'] > r_all['pctile_mean'] + 5:
            compare = 'dd5% 승'
        elif r_dd5['pctile_mean'] < r_all['pctile_mean'] - 5:
            compare = '기존 승'
        else:
            compare = '비슷'

        print(f"  {tk:<7s} | {r_all['n']:>4d} {r_all['v4_mean']:>+7.1f}% {r_all['boot_mean']:>+7.1f}% {r_all['pctile_mean']:>6.1f}% {r_all['p_mean']:>6.3f}{sig_all:<3s}"
              f" | {r_dd5['n']:>4d} {r_dd5['v4_mean']:>+7.1f}% {r_dd5['boot_mean']:>+7.1f}% {r_dd5['pctile_mean']:>6.1f}% {r_dd5['p_mean']:>6.3f}{sig_dd5:<3s}"
              f" | {compare}")

        results_90.append({
            'tk': tk, 'n_all': r_all['n'], 'n_dd5': r_dd5['n'] if r_dd5 else 0,
            'pctile_all': r_all['pctile_mean'], 'pctile_dd5': r_dd5['pctile_mean'] if r_dd5 else 0,
            'p_all': r_all['p_mean'], 'p_dd5': r_dd5['p_mean'] if r_dd5 else 1,
            'v4_mean_all': r_all['v4_mean'], 'v4_mean_dd5': r_dd5['v4_mean'] if r_dd5 else 0,
            'v4_hr_all': r_all['v4_hr'], 'v4_hr_dd5': r_dd5['v4_hr'] if r_dd5 else 0,
        })
    else:
        print(f"  {tk:<7s} | {r_all['n']:>4d} {r_all['v4_mean']:>+7.1f}% {r_all['boot_mean']:>+7.1f}% {r_all['pctile_mean']:>6.1f}% {r_all['p_mean']:>6.3f}{sig_all:<3s}"
              f" | {'N/A (신호 부족)':^40s} |")
        results_90.append({
            'tk': tk, 'n_all': r_all['n'], 'n_dd5': 0,
            'pctile_all': r_all['pctile_mean'], 'pctile_dd5': 0,
            'p_all': r_all['p_mean'], 'p_dd5': 1,
            'v4_mean_all': r_all['v4_mean'], 'v4_mean_dd5': 0,
            'v4_hr_all': r_all['v4_hr'], 'v4_hr_dd5': 0,
        })

print(f'  {"-"*125}')
n_sig_all = sum(1 for r in results_90 if r['p_all'] < 0.05)
n_sig_dd5 = sum(1 for r in results_90 if r['p_dd5'] < 0.05 and r['n_dd5'] >= 3)
n_better = sum(1 for r in results_90 if r['n_dd5'] >= 3 and r['pctile_dd5'] > r['pctile_all'])
n_valid_dd5 = sum(1 for r in results_90 if r['n_dd5'] >= 3)
print(f'  기존: p<0.05 = {n_sig_all}/{len(results_90)}종목')
print(f'  dd5%: p<0.05 = {n_sig_dd5}/{n_valid_dd5}종목 (유효)')
print(f'  dd5%가 백분위 더 높은 종목: {n_better}/{n_valid_dd5}')
print()


# ═══════════════════════════════════════════════════════
# [2] 30일/60일도
# ═══════════════════════════════════════════════════════
for horizon in [30, 60]:
    print(f'  -- {horizon}일 보유 --')
    print(f"  {'Ticker':<7s} | {'기존 N':>5s} {'V4평균':>8s} {'pctile':>7s} {'p':>7s} | {'dd5% N':>6s} {'V4평균':>8s} {'pctile':>7s} {'p':>7s} | {'비교'}")
    print(f'  {"-"*100}')

    for tk in TICKERS:
        if tk not in ticker_data:
            continue
        td = ticker_data[tk][horizon]
        r_all = bootstrap(td['v4_all'], td['all_rets'])
        r_dd5 = bootstrap(td['v4_dd5'], td['all_rets'])
        if r_all is None:
            continue

        sig_all = '***' if r_all['p_mean'] < 0.01 else '**' if r_all['p_mean'] < 0.05 else '*' if r_all['p_mean'] < 0.10 else ''

        if r_dd5 is not None:
            sig_dd5 = '***' if r_dd5['p_mean'] < 0.01 else '**' if r_dd5['p_mean'] < 0.05 else '*' if r_dd5['p_mean'] < 0.10 else ''
            compare = 'dd5%승' if r_dd5['pctile_mean'] > r_all['pctile_mean'] + 5 else '기존승' if r_dd5['pctile_mean'] < r_all['pctile_mean'] - 5 else '비슷'
            print(f"  {tk:<7s} | {r_all['n']:>5d} {r_all['v4_mean']:>+7.1f}% {r_all['pctile_mean']:>6.1f}% {r_all['p_mean']:>6.3f}{sig_all:<3s}"
                  f" | {r_dd5['n']:>6d} {r_dd5['v4_mean']:>+7.1f}% {r_dd5['pctile_mean']:>6.1f}% {r_dd5['p_mean']:>6.3f}{sig_dd5:<3s}"
                  f" | {compare}")
        else:
            print(f"  {tk:<7s} | {r_all['n']:>5d} {r_all['v4_mean']:>+7.1f}% {r_all['pctile_mean']:>6.1f}% {r_all['p_mean']:>6.3f}{sig_all:<3s}"
                  f" | {'N/A':>6s} {'':>8s} {'':>7s} {'':>7s}   |")

    print()


# ═══════════════════════════════════════════════════════
# [3] 전체 통합 부트스트랩
# ═══════════════════════════════════════════════════════
print('=' * 130)
print('  [3] 전체 통합 부트스트랩 (전 종목 합산)')
print('=' * 130)

rng = np.random.default_rng(42)

for horizon in [30, 60, 90]:
    # 전 종목 합산
    all_v4_all = []
    all_v4_dd5 = []
    all_possible = []

    for tk in ticker_data:
        td = ticker_data[tk][horizon]
        all_v4_all.extend(td['v4_all'].tolist())
        all_v4_dd5.extend(td['v4_dd5'].tolist())
        all_possible.extend(td['all_rets'].tolist())

    all_possible = np.array(all_possible)
    v4_all_arr = np.array(all_v4_all)
    v4_dd5_arr = np.array(all_v4_dd5)

    print(f'\n  {horizon}일 보유:')
    print(f'  {"":4s} {"":12s} {"N":>5s} {"V4평균":>9s} {"적중률":>7s} | {"부트평균":>9s} {"백분위":>7s} {"p-val":>8s} {"95% CI":>20s}')

    for label, v4_arr in [('기존 V4', v4_all_arr), ('V4+dd5%', v4_dd5_arr)]:
        if len(v4_arr) < 5:
            continue
        v4_mean = np.mean(v4_arr)
        v4_hr = np.mean(v4_arr > 0) * 100
        n_v4 = len(v4_arr)

        boot_means = np.zeros(N_BOOTSTRAP)
        for i in range(N_BOOTSTRAP):
            boot_means[i] = np.mean(rng.choice(all_possible, size=n_v4, replace=True))

        p = np.mean(boot_means >= v4_mean)
        pctile = np.mean(boot_means < v4_mean) * 100
        ci_lo = np.percentile(boot_means, 2.5)
        ci_hi = np.percentile(boot_means, 97.5)
        sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else 'n.s.'

        print(f'    {label:12s} {n_v4:>5d} {v4_mean:>+8.2f}% {v4_hr:>6.1f}% |'
              f' {np.mean(boot_means):>+8.2f}% {pctile:>6.1f}% {p:>7.4f}{sig:>4s}'
              f' [{ci_lo:>+7.2f}%, {ci_hi:>+7.2f}%]')


# ═══════════════════════════════════════════════════════
# [4] 종합 판정
# ═══════════════════════════════════════════════════════
print()
print()
print('=' * 130)
print('  [4] 종합 판정')
print('=' * 130)
print()

# 90일 기준 비교표
print('  90일 보유 기준 비교:')
print(f'  {"":20s} {"기존 V4":>12s} {"V4+dd5%":>12s}')
print(f'  {"-"*50}')

# 전체 통합 수치
all_v4_all_90 = []
all_v4_dd5_90 = []
all_possible_90 = []
for tk in ticker_data:
    td = ticker_data[tk][90]
    all_v4_all_90.extend(td['v4_all'].tolist())
    all_v4_dd5_90.extend(td['v4_dd5'].tolist())
    all_possible_90.extend(td['all_rets'].tolist())

all_possible_90 = np.array(all_possible_90)

# 기존
r1 = bootstrap(np.array(all_v4_all_90), all_possible_90)
r2 = bootstrap(np.array(all_v4_dd5_90), all_possible_90)

print(f'  {"신호 수":20s} {r1["n"]:>12d} {r2["n"]:>12d}')
print(f'  {"평균 수익률":20s} {r1["v4_mean"]:>+11.2f}% {r2["v4_mean"]:>+11.2f}%')
print(f'  {"적중률":20s} {r1["v4_hr"]:>11.1f}% {r2["v4_hr"]:>11.1f}%')
print(f'  {"부트 백분위":20s} {r1["pctile_mean"]:>11.1f}% {r2["pctile_mean"]:>11.1f}%')
print(f'  {"p-value":20s} {r1["p_mean"]:>12.4f} {r2["p_mean"]:>12.4f}')
sig1 = 'YES (p<0.05)' if r1['p_mean'] < 0.05 else 'NO'
sig2 = 'YES (p<0.05)' if r2['p_mean'] < 0.05 else 'NO'
print(f'  {"통계적 유의":20s} {sig1:>12s} {sig2:>12s}')
print(f'  {"95% CI 밖?":20s} {"YES" if r1["v4_mean"]>r1["ci_hi"] else "NO":>12s} {"YES" if r2["v4_mean"]>r2["ci_hi"] else "NO":>12s}')
print()

# 종목별 유의한 수
n_all_sig = sum(1 for r in results_90 if r['p_all'] < 0.05)
n_dd5_sig = sum(1 for r in results_90 if r['p_dd5'] < 0.05 and r['n_dd5'] >= 3)
n_all_trend = sum(1 for r in results_90 if r['p_all'] < 0.10)
n_dd5_trend = sum(1 for r in results_90 if r['p_dd5'] < 0.10 and r['n_dd5'] >= 3)
n_valid = sum(1 for r in results_90 if r['n_dd5'] >= 3)

print(f'  종목별 유의성 (90일):')
print(f'    p<0.05: 기존 {n_all_sig}/{len(results_90)}종목  dd5% {n_sig_dd5}/{n_valid}종목')
print(f'    p<0.10: 기존 {n_all_trend}/{len(results_90)}종목  dd5% {n_dd5_trend}/{n_valid}종목')
print()

print('=' * 130)
print('Done.')
