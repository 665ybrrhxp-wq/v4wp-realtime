"""V4 신호 품질 필터 테스트: 어떤 조건을 추가하면 부트스트랩 유의성에 도달하는가?

핵심 질문: V4 전체 신호(82.7th pctile, p=0.173)에서 고품질 신호만 골라내면 p<0.05가 가능한가?

테스트할 필터:
  F1: V4 스코어 크기 (score >= 0.25, 0.35, 0.50)
  F2: 가격 위치 - 20일 고점 대비 하락률 (drawdown >= 5%, 10%, 15%)
  F3: 변동성 레짐 - ATR 상위 50%, 33% 구간에서만
  F4: 서브지표 합의 - S_Force, S_Div가 같은 방향
  F5: 복합 - F1 + F2 조합
"""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from real_market_backtest import (
    calc_v4_score, calc_v4_subindicators, detect_signal_events,
    build_price_filter, smooth_earnings_volume,
    calc_atr_percentile
)

BUY_CONFIRM = 3
LATE_SELL_DROP_TH = 0.05
DIVGATE_DAYS = 3
N_BOOTSTRAP = 10000
HORIZON = 90  # 90일 보유 기준

TICKERS = ['TSLA','PLTR','NVDA','AVGO','AMZN','GOOGL','JOBY','HIMS',
           'TEM','RKLB','PGY','COIN','HOOD','IONQ','PL','QQQ','VOO']


def download_max(ticker):
    t = yf.Ticker(ticker)
    df = t.history(period='max', auto_adjust=True)
    if df is None or len(df) == 0:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def bootstrap_test(v4_rets, all_rets, n_boot=N_BOOTSTRAP, seed=42):
    """부트스트랩 테스트: V4 수익률 vs 무작위 N회 추출 10,000세트"""
    if len(v4_rets) < 3:
        return {'n': len(v4_rets), 'p': 1.0, 'pctile': 50.0,
                'v4_mean': 0, 'boot_mean': 0, 'v4_hr': 0}
    rng = np.random.default_rng(seed)
    v4_mean = np.mean(v4_rets)
    v4_hr = np.mean(np.array(v4_rets) > 0) * 100
    n_v4 = len(v4_rets)

    boot_means = np.zeros(n_boot)
    for i in range(n_boot):
        sample = rng.choice(all_rets, size=n_v4, replace=True)
        boot_means[i] = np.mean(sample)

    p = np.mean(boot_means >= v4_mean)
    pctile = np.mean(boot_means < v4_mean) * 100

    return {
        'n': n_v4,
        'p': p,
        'pctile': pctile,
        'v4_mean': v4_mean,
        'boot_mean': np.mean(boot_means),
        'v4_hr': v4_hr,
    }


# ═══════════════════════════════════════════════════════
# 데이터 수집: 각 종목별 신호 + 메타데이터 수집
# ═══════════════════════════════════════════════════════
print('=' * 120)
print('  V4 신호 품질 필터 테스트')
print('  목표: 어떤 추가 조건이 V4 매수 신호의 부트스트랩 유의성을 높이는가?')
print('=' * 120)
print()

# 각 신호별 메타데이터 수집
all_signals = []  # list of dict: {ticker, idx, score_at_buy, drawdown_20d, atr_pctile, subind_agree, fwd_ret}
all_possible_rets = []  # 전 종목의 모든 가능한 90d forward return

for tk in TICKERS:
    print(f'  {tk}...', end=' ', flush=True)
    df = download_max(tk)
    if df is None or len(df) < 200:
        print('SKIP')
        continue

    df_smooth = smooth_earnings_volume(df, ticker=tk)
    score = calc_v4_score(df_smooth, w=20, divgate_days=DIVGATE_DAYS)
    subind = calc_v4_subindicators(df_smooth, w=20, divgate_days=DIVGATE_DAYS)
    events = detect_signal_events(score, th=0.15, cooldown=5)
    pf = build_price_filter(df_smooth, er_q=66, atr_q=55, lookback=252)

    close = df['Close'].values
    n = len(close)
    rh20 = df['Close'].rolling(20, min_periods=1).max().values
    rh50 = df['Close'].rolling(50, min_periods=1).max().values
    atr_pct = calc_atr_percentile(df, 14, 252).values

    # 필터 적용된 이벤트
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

    # 매수 신호 추출 + 메타데이터
    for ev in filtered:
        if ev['type'] != 'bottom':
            continue
        si, ei = ev['start_idx'], ev['end_idx']
        dur = ev['duration']
        ci = si + BUY_CONFIRM - 1
        if ci > ei or dur < BUY_CONFIRM:
            continue
        if ci + HORIZON >= n:
            continue

        # 메타데이터 수집
        score_at_buy = score.iloc[ci] if ci < len(score) else 0
        peak_score = ev['peak_val']

        # 20일 고점 대비 하락률
        dd_20 = (rh20[ci] - close[ci]) / rh20[ci] if rh20[ci] > 0 else 0
        # 50일 고점 대비 하락률
        dd_50 = (rh50[ci] - close[ci]) / rh50[ci] if rh50[ci] > 0 else 0

        # ATR 백분위
        atr_p = atr_pct[ci] if ci < len(atr_pct) else 0.5

        # 서브지표 합의: s_force와 s_div가 같은 방향(양수)인지
        if ci < len(subind):
            s_force = subind['s_force'].iloc[ci]
            s_div = subind['s_div'].iloc[ci]
            s_conc = subind['s_conc'].iloc[ci]
            # 매수 신호이므로 양수가 좋음. 3개 중 2개 이상 양수면 합의
            n_pos = sum(1 for v in [s_force, s_div, s_conc] if v > 0)
            subind_agree = n_pos >= 2
            all_three = n_pos >= 3
        else:
            s_force = s_div = s_conc = 0
            subind_agree = False
            all_three = False

        # Forward return
        fwd_ret = ((close[ci + HORIZON] / close[ci]) - 1) * 100

        all_signals.append({
            'ticker': tk,
            'idx': ci,
            'score_at_buy': float(score_at_buy),
            'peak_score': float(peak_score),
            'duration': dur,
            'dd_20': dd_20,
            'dd_50': dd_50,
            'atr_pctile': atr_p,
            's_force': float(s_force),
            's_div': float(s_div),
            's_conc': float(s_conc),
            'subind_agree': subind_agree,
            'all_three': all_three,
            'fwd_ret': fwd_ret,
        })

    # 모든 가능한 forward return
    for idx in range(n - HORIZON):
        r = ((close[idx + HORIZON] / close[idx]) - 1) * 100
        all_possible_rets.append(r)

    print(f"n_signals={sum(1 for s in all_signals if s['ticker']==tk)}")

all_possible = np.array(all_possible_rets)
print(f'\n  총 신호: {len(all_signals)}개,  가능한 entry: {len(all_possible):,}개')
print()


# ═══════════════════════════════════════════════════════
# 필터별 부트스트랩 테스트
# ═══════════════════════════════════════════════════════
def run_filter(name, condition_fn):
    """조건 함수를 적용하여 필터된 신호로 부트스트랩 테스트"""
    filtered_rets = [s['fwd_ret'] for s in all_signals if condition_fn(s)]
    if len(filtered_rets) < 5:
        return {'name': name, 'n': len(filtered_rets), 'p': 1.0, 'pctile': 50.0,
                'v4_mean': 0, 'boot_mean': 0, 'v4_hr': 0, 'sig': ''}
    result = bootstrap_test(np.array(filtered_rets), all_possible)
    sig = '***' if result['p'] < 0.01 else '**' if result['p'] < 0.05 else '*' if result['p'] < 0.10 else ''
    result['name'] = name
    result['sig'] = sig
    return result


print('=' * 120)
print('  [1] 개별 필터 테스트 (90일 보유)')
print('=' * 120)
print(f"  {'필터':40s} {'N':>5s} | {'V4평균':>8s} {'부트평균':>8s} | {'백분위':>7s} {'p-val':>8s} | {'적중률':>6s} | 판정")
print(f'  {"-"*115}')

filters = [
    # 기준선: 필터 없음
    ('기준: 전체 신호 (필터 없음)', lambda s: True),

    # F1: V4 스코어 크기
    ('F1a: score_at_buy >= 0.20', lambda s: s['score_at_buy'] >= 0.20),
    ('F1b: score_at_buy >= 0.30', lambda s: s['score_at_buy'] >= 0.30),
    ('F1c: peak_score >= 0.40', lambda s: s['peak_score'] >= 0.40),
    ('F1d: peak_score >= 0.60', lambda s: s['peak_score'] >= 0.60),

    # F2: 가격 위치 (20일 고점 대비 하락)
    ('F2a: drawdown_20d >= 3%', lambda s: s['dd_20'] >= 0.03),
    ('F2b: drawdown_20d >= 5%', lambda s: s['dd_20'] >= 0.05),
    ('F2c: drawdown_20d >= 10%', lambda s: s['dd_20'] >= 0.10),
    ('F2d: drawdown_50d >= 10%', lambda s: s['dd_50'] >= 0.10),
    ('F2e: drawdown_50d >= 15%', lambda s: s['dd_50'] >= 0.15),
    ('F2f: drawdown_50d >= 20%', lambda s: s['dd_50'] >= 0.20),

    # F3: 변동성 레짐
    ('F3a: ATR 백분위 > 0.50 (상위 50%)', lambda s: s['atr_pctile'] > 0.50),
    ('F3b: ATR 백분위 > 0.67 (상위 33%)', lambda s: s['atr_pctile'] > 0.67),
    ('F3c: ATR 백분위 > 0.80 (상위 20%)', lambda s: s['atr_pctile'] > 0.80),

    # F4: 서브지표 합의
    ('F4a: 2/3 서브지표 양수', lambda s: s['subind_agree']),
    ('F4b: 3/3 서브지표 모두 양수', lambda s: s['all_three']),
    ('F4c: S_Force > 0 (힘 양수)', lambda s: s['s_force'] > 0),
    ('F4d: S_Div > 0 (발산 양수)', lambda s: s['s_div'] > 0),

    # F5: Duration 강화
    ('F5a: duration >= 4일', lambda s: s['duration'] >= 4),
    ('F5b: duration >= 5일', lambda s: s['duration'] >= 5),
    ('F5c: duration >= 7일', lambda s: s['duration'] >= 7),

    # F6: 복합 필터
    ('F6a: score>=0.30 + dd_20>=5%', lambda s: s['peak_score'] >= 0.30 and s['dd_20'] >= 0.05),
    ('F6b: score>=0.30 + ATR>0.50', lambda s: s['peak_score'] >= 0.30 and s['atr_pctile'] > 0.50),
    ('F6c: dd_20>=5% + ATR>0.50', lambda s: s['dd_20'] >= 0.05 and s['atr_pctile'] > 0.50),
    ('F6d: dd_50>=10% + 2/3합의', lambda s: s['dd_50'] >= 0.10 and s['subind_agree']),
    ('F6e: score>=0.40 + dd_50>=10%', lambda s: s['peak_score'] >= 0.40 and s['dd_50'] >= 0.10),
    ('F6f: dd_20>=5% + duration>=4', lambda s: s['dd_20'] >= 0.05 and s['duration'] >= 4),
    ('F6g: score>=0.30 + dd_20>=5% + ATR>0.50',
     lambda s: s['peak_score'] >= 0.30 and s['dd_20'] >= 0.05 and s['atr_pctile'] > 0.50),
    ('F6h: score>=0.40 + dd_50>=15%', lambda s: s['peak_score'] >= 0.40 and s['dd_50'] >= 0.15),
    ('F6i: dd_50>=20% + ATR>0.67', lambda s: s['dd_50'] >= 0.20 and s['atr_pctile'] > 0.67),
]

results = []
for name, fn in filters:
    r = run_filter(name, fn)
    results.append(r)

    if r['pctile'] >= 95:
        verdict = 'V4 유의 우세'
    elif r['pctile'] >= 75:
        verdict = 'V4 다소 우세'
    elif r['pctile'] >= 25:
        verdict = '무차별'
    else:
        verdict = 'V4 열세'

    print(f"  {r['name']:40s} {r['n']:>5d} | {r['v4_mean']:>+7.1f}% {r['boot_mean']:>+7.1f}% |"
          f" {r['pctile']:>6.1f}% {r['p']:>7.4f}{r['sig']:<3s} | {r['v4_hr']:>5.1f}% | {verdict}")

# 유의한 필터 요약
print()
print('=' * 120)
print('  [2] 유의한 필터 TOP 10 (백분위 기준 내림차순)')
print('=' * 120)

top = sorted(results, key=lambda x: -x['pctile'])[:10]
print(f"  {'순위':>4s} {'필터':40s} {'N':>5s} | {'V4평균':>8s} {'부트':>8s} | {'백분위':>7s} {'p-val':>8s} |")
print(f'  {"-"*100}')
for i, r in enumerate(top):
    marker = ' <<<' if r['p'] < 0.05 else ' <<' if r['p'] < 0.10 else ''
    print(f"  {i+1:>4d} {r['name']:40s} {r['n']:>5d} | {r['v4_mean']:>+7.1f}% {r['boot_mean']:>+7.1f}% |"
          f" {r['pctile']:>6.1f}% {r['p']:>7.4f}{r['sig']:<3s} |{marker}")


# ═══════════════════════════════════════════════════════
# [3] 최고 필터 상세 분석
# ═══════════════════════════════════════════════════════
print()
print('=' * 120)
print('  [3] 최고 필터 상세 분석')
print('=' * 120)

best = max(results, key=lambda x: x['pctile'])
print(f"  최고 필터: {best['name']}")
print(f"  신호 수: {best['n']}개")
print(f"  V4 평균: {best['v4_mean']:+.2f}%  vs  무작위: {best['boot_mean']:+.2f}%")
print(f"  백분위: {best['pctile']:.1f}%  p-value: {best['p']:.4f}")
print(f"  적중률: {best['v4_hr']:.1f}%")
print()

# 유의성에 가장 가까운 필터(N >= 20이면서 p가 가장 낮은)
practical = [r for r in results if r['n'] >= 20]
if practical:
    best_practical = min(practical, key=lambda x: x['p'])
    print(f"  실용적 최적 (N>=20): {best_practical['name']}")
    print(f"  신호 수: {best_practical['n']}개")
    print(f"  V4 평균: {best_practical['v4_mean']:+.2f}%  vs  무작위: {best_practical['boot_mean']:+.2f}%")
    print(f"  백분위: {best_practical['pctile']:.1f}%  p-value: {best_practical['p']:.4f}")
    print(f"  적중률: {best_practical['v4_hr']:.1f}%")


# ═══════════════════════════════════════════════════════
# [4] 30d/60d에서도 검증
# ═══════════════════════════════════════════════════════
print()
print('=' * 120)
print('  [4] 최고 필터를 30d/60d에서도 검증')
print('=' * 120)

# 재계산을 위해 30d/60d forward return 추가 수집이 필요
# 이미 90d 기준으로 메타데이터를 모았으므로, 별도로 다시 수집
for check_horizon in [30, 60]:
    print(f'\n  -- {check_horizon}일 보유 --')

    # 해당 horizon에 대한 forward return 재계산
    v4_rets_h = []
    all_rets_h = []

    for tk in TICKERS:
        df = download_max(tk)
        if df is None or len(df) < 200:
            continue
        close = df['Close'].values
        n = len(close)

        # V4 신호의 해당 horizon forward return
        for s in all_signals:
            if s['ticker'] != tk:
                continue
            idx = s['idx']
            if idx + check_horizon < n:
                r = ((close[idx + check_horizon] / close[idx]) - 1) * 100
                v4_rets_h.append((s, r))

        # 모든 가능한 entry
        for idx in range(n - check_horizon):
            r = ((close[idx + check_horizon] / close[idx]) - 1) * 100
            all_rets_h.append(r)

    all_arr_h = np.array(all_rets_h)

    # 상위 5개 필터 검증
    top5_filters = sorted(results, key=lambda x: -x['pctile'])[:5]
    for tf in top5_filters:
        # 해당 필터 조건 다시 적용
        fname = tf['name']
        # filters 리스트에서 해당 조건 함수 찾기
        cond_fn = None
        for fn_name, fn in filters:
            if fn_name == fname:
                cond_fn = fn
                break
        if cond_fn is None:
            continue

        filtered_rets = [r for s, r in v4_rets_h if cond_fn(s)]
        if len(filtered_rets) < 5:
            continue
        bt = bootstrap_test(np.array(filtered_rets), all_arr_h)
        sig = '***' if bt['p'] < 0.01 else '**' if bt['p'] < 0.05 else '*' if bt['p'] < 0.10 else ''
        print(f"    {fname:40s} N={bt['n']:>4d}  V4={bt['v4_mean']:>+7.1f}%  부트={bt['boot_mean']:>+7.1f}%"
              f"  pctile={bt['pctile']:>5.1f}%  p={bt['p']:.4f}{sig}")


# ═══════════════════════════════════════════════════════
# [5] 종합 결론
# ═══════════════════════════════════════════════════════
print()
print('=' * 120)
print('  [5] 종합 결론')
print('=' * 120)
print()

n_sig_005 = sum(1 for r in results if r['p'] < 0.05)
n_sig_010 = sum(1 for r in results if r['p'] < 0.10)
n_above_90 = sum(1 for r in results if r['pctile'] >= 90)
n_above_80 = sum(1 for r in results if r['pctile'] >= 80)

print(f"  테스트한 필터 수: {len(results)}")
print(f"  p < 0.05 (유의): {n_sig_005}개")
print(f"  p < 0.10 (경향): {n_sig_010}개")
print(f"  백분위 >= 90%: {n_above_90}개")
print(f"  백분위 >= 80%: {n_above_80}개")
print()

if n_sig_005 > 0:
    sig_filters = [r for r in results if r['p'] < 0.05]
    print("  >>> 유의한 필터 발견!")
    for r in sig_filters:
        print(f"      {r['name']}: p={r['p']:.4f}, 백분위={r['pctile']:.1f}%, N={r['n']}, V4평균={r['v4_mean']:+.1f}%")
    print()
    print("  결론: V4 + 추가 필터 조합으로 통계적으로 유의한 매수 타이밍 가능")
elif n_sig_010 > 0:
    trend_filters = [r for r in results if r['p'] < 0.10]
    print("  >>> 유의성에 근접한 필터 발견 (p < 0.10)")
    for r in trend_filters:
        print(f"      {r['name']}: p={r['p']:.4f}, 백분위={r['pctile']:.1f}%, N={r['n']}, V4평균={r['v4_mean']:+.1f}%")
    print()
    print("  결론: V4 + 추가 필터로 경향성 존재, 추가 개선 여지 있음")
else:
    print("  >>> 어떤 필터도 유의성에 도달하지 못함")
    print("  결론: V4 매수 신호 자체의 타이밍 edge가 제한적")
    print("         → 리스크 관리/심리 보조 도구로 활용하는 것이 적절")

print()
print('=' * 120)
print('Done.')
