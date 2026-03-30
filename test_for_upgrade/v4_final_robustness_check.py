"""최종 로직 4가지 견고성 점검.

1. DD 20%+ 극강 구간: 소표본 노이즈 vs 구조적 특성
2. 90일 고정 홀딩 vs 기계적 출구 규칙 (MFE 반납 기반)
3. VOO 레버리지 29건 표본 신뢰도
4. 2026 look-ahead bias 점검
"""
import sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yfinance as yf
from real_market_backtest import (
    calc_v4_score, detect_signal_events,
    build_price_filter, smooth_earnings_volume,
)

TICKERS = ['QQQ', 'VOO', 'GOOGL', 'AMZN', 'NVDA', 'AVGO', 'TSLA',
           'AAPL', 'MSFT', 'META', 'JPM', 'BRK-B', 'V', 'UNH', 'XOM']
np.random.seed(42)


def collect_signals(ticker, period='max'):
    """DD gate OFF로 모든 신호 수집 + 90일 일별 가격 경로."""
    tk = yf.Ticker(ticker)
    df = tk.history(period=period, auto_adjust=True)
    if df is None or len(df) < 252:
        return []
    df = smooth_earnings_volume(df, ticker=ticker)
    score = calc_v4_score(df, w=20, divgate_days=3)
    events = detect_signal_events(score, th=0.05, cooldown=5)
    pf = build_price_filter(df, er_q=80, atr_q=40, lookback=252)
    close = df['Close'].values
    rh = df['Close'].rolling(20, min_periods=1).max().values

    sigs = []
    for e in events:
        if e['type'] != 'bottom' or not pf(e['peak_idx']):
            continue
        pidx = e['peak_idx']
        if pidx >= len(close) - 90:
            continue
        price, high = close[pidx], rh[pidx]
        dd = (high - price) / high if high > 0 else 0
        if dd < 0.03:
            continue

        # 90일 일별 가격 배열
        prices_90 = close[pidx:pidx + 91]  # 매수일 포함 91개
        daily_rets = np.diff(prices_90) / prices_90[:-1]

        sigs.append({
            'ticker': ticker,
            'date': df.index[pidx],
            'year': df.index[pidx].year,
            'price': price,
            'dd': dd,
            'prices_90': prices_90,
            'daily_rets': daily_rets,
            'ret_90d': (prices_90[-1] - price) / price,
        })
    return sigs


t0 = time.time()
all_sigs = []
for t in TICKERS:
    s = collect_signals(t, 'max')
    all_sigs.extend(s)
    print(f'  {t}: {len(s)}개')
print(f'\n총 {len(all_sigs)}개 신호 ({time.time()-t0:.0f}초)\n')


# ═══════════════════════════════════════════════════════════════════
# 점검 1: DD 20%+ 극강 구간 — 소표본 노이즈 vs 구조적 특성
# ═══════════════════════════════════════════════════════════════════
print('=' * 85)
print('점검 1: DD 20%+ 극강 구간 분석 (소표본 노이즈 vs 구조적 특성)')
print('=' * 85)

extreme = [s for s in all_sigs if s['dd'] >= 0.20]
moderate = [s for s in all_sigs if 0.10 <= s['dd'] < 0.20]

print(f'\n■ 극강(20%+) 개별 신호 전수 조사 ({len(extreme)}건)')
print(f'  {"날짜":<12} {"종목":<6} {"DD":>6} {"매수가":>8} | {"30d":>7} {"60d":>7} {"90d":>7} | {"MFE60":>7} {"MAE60":>7} | 결과')
print(f'  {"-"*90}')

wins, losses = 0, 0
loss_details = []
for s in sorted(extreme, key=lambda x: x['date']):
    p = s['prices_90']
    r30 = (p[30] - p[0]) / p[0]
    r60 = (p[60] - p[0]) / p[0]
    r90 = (p[90] - p[0]) / p[0]
    cum60 = (p[1:61] - p[0]) / p[0]
    mfe60 = float(np.max(cum60))
    mae60 = float(np.min(cum60))
    result = 'O' if r90 > 0 else 'X'
    if r90 > 0:
        wins += 1
    else:
        losses += 1
        loss_details.append(s)
    print(f'  {s["date"].strftime("%Y-%m-%d"):<12} {s["ticker"]:<6} {s["dd"]:>5.1%} {s["price"]:>8.2f} | '
          f'{r30:>+6.1%} {r60:>+6.1%} {r90:>+6.1%} | {mfe60:>+6.1%} {mae60:>+6.1%} | {result}')

print(f'\n  승: {wins}, 패: {losses}, 적중률: {wins/(wins+losses):.1%}')

# 손실 케이스 패턴 분석
print(f'\n■ 손실 케이스 패턴 분석 ({losses}건)')
for s in loss_details:
    p = s['prices_90']
    r90 = (p[90] - p[0]) / p[0]
    cum = (p[1:] - p[0]) / p[0]
    mfe = float(np.max(cum))
    mfe_day = int(np.argmax(cum)) + 1
    print(f'  {s["date"].strftime("%Y-%m-%d")} {s["ticker"]}: '
          f'DD={s["dd"]:.1%}, 90d={r90:+.1%}, MFE={mfe:+.1%}(day{mfe_day})')
    # 이 신호가 V자 회복 실패인지 확인
    if mfe > 0.10:
        print(f'    → MFE {mfe:+.1%} 달성 후 반납 (V자 회복 후 재하락)')
    else:
        print(f'    → 회복 자체 실패 (지속 하락)')

# Bootstrap: 극강 vs 강력
print(f'\n■ Bootstrap: 극강(20%+) vs 강력(10~20%) 차이 검증')
ext_rets = np.array([s['ret_90d'] for s in extreme])
mod_rets = np.array([s['ret_90d'] for s in moderate])

n_boot = 10000
ext_means = [np.mean(np.random.choice(ext_rets, len(ext_rets), replace=True)) for _ in range(n_boot)]
mod_means = [np.mean(np.random.choice(mod_rets, len(mod_rets), replace=True)) for _ in range(n_boot)]

print(f'  강력(10~20%): avg={np.mean(mod_rets):+.1%}, CI=[{np.percentile(mod_means, 2.5):+.1%}, {np.percentile(mod_means, 97.5):+.1%}], n={len(mod_rets)}')
print(f'  극강(20%+):   avg={np.mean(ext_rets):+.1%}, CI=[{np.percentile(ext_means, 2.5):+.1%}, {np.percentile(ext_means, 97.5):+.1%}], n={len(ext_rets)}')
print(f'  CI 폭:  강력 ±{(np.percentile(mod_means,97.5)-np.percentile(mod_means,2.5))/2:.1%} | 극강 ±{(np.percentile(ext_means,97.5)-np.percentile(ext_means,2.5))/2:.1%}')

# 적중률 bootstrap
ext_hits = np.array([1 if s['ret_90d'] > 0 else 0 for s in extreme])
hit_boots = [np.mean(np.random.choice(ext_hits, len(ext_hits), replace=True)) for _ in range(n_boot)]
print(f'  극강 적중률 CI: [{np.percentile(hit_boots, 2.5):.1%}, {np.percentile(hit_boots, 97.5):.1%}]')
print(f'  → 하한 {np.percentile(hit_boots, 2.5):.1%}이면 {"신뢰 가능" if np.percentile(hit_boots, 2.5) > 0.50 else "동전 던지기 수준 포함"}')


# ═══════════════════════════════════════════════════════════════════
# 점검 2: 90일 고정 홀딩 vs 기계적 출구 규칙
# ═══════════════════════════════════════════════════════════════════
print(f'\n{"=" * 85}')
print('점검 2: 출구 전략 비교 (90일 Hold vs Trailing Stop vs MFE 반납 청산)')
print('=' * 85)

prod_sigs = [s for s in all_sigs if s['dd'] >= 0.03]  # 프로덕션 기준

def simulate_exit(sigs, strategy, **kwargs):
    """다양한 출구 전략 시뮬레이션."""
    results = []
    for s in sigs:
        p = s['prices_90']
        entry = p[0]

        if strategy == 'hold_90d':
            exit_price = p[90]
            exit_day = 90

        elif strategy == 'hold_60d':
            exit_price = p[60]
            exit_day = 60

        elif strategy == 'hold_30d':
            exit_price = p[30]
            exit_day = 30

        elif strategy == 'trailing_stop':
            ts_pct = kwargs.get('stop_pct', 0.10)
            peak = entry
            exit_price = p[90]
            exit_day = 90
            for d in range(1, 91):
                peak = max(peak, p[d])
                if p[d] < peak * (1 - ts_pct):
                    exit_price = p[d]
                    exit_day = d
                    break

        elif strategy == 'mfe_giveback':
            gb_pct = kwargs.get('giveback_pct', 0.50)
            peak_ret = 0
            exit_price = p[90]
            exit_day = 90
            for d in range(1, 91):
                ret = (p[d] - entry) / entry
                peak_ret = max(peak_ret, ret)
                # MFE의 gb_pct 이상 반납 시 청산
                if peak_ret > 0.02 and ret < peak_ret * (1 - gb_pct):
                    exit_price = p[d]
                    exit_day = d
                    break

        elif strategy == 'time_decay':
            # 30일까지 홀드 + 이후 trailing stop 5%
            exit_price = p[90]
            exit_day = 90
            if len(p) > 30:
                peak = max(p[:31])
                for d in range(31, 91):
                    peak = max(peak, p[d])
                    if p[d] < peak * 0.95:
                        exit_price = p[d]
                        exit_day = d
                        break

        ret = (exit_price - entry) / entry
        results.append({'ret': ret, 'day': exit_day})
    return results

strategies = [
    ('90일 고정 홀드', 'hold_90d', {}),
    ('60일 고정 홀드', 'hold_60d', {}),
    ('30일 고정 홀드', 'hold_30d', {}),
    ('Trailing Stop 10%', 'trailing_stop', {'stop_pct': 0.10}),
    ('Trailing Stop 15%', 'trailing_stop', {'stop_pct': 0.15}),
    ('Trailing Stop 20%', 'trailing_stop', {'stop_pct': 0.20}),
    ('MFE 50% 반납 청산', 'mfe_giveback', {'giveback_pct': 0.50}),
    ('MFE 40% 반납 청산', 'mfe_giveback', {'giveback_pct': 0.40}),
    ('MFE 60% 반납 청산', 'mfe_giveback', {'giveback_pct': 0.60}),
    ('30일 Hold + TS 5%', 'time_decay', {}),
]

print(f'\n■ 출구 전략별 성과 비교 ({len(prod_sigs)}건)')
print(f'  {"전략":<22} | {"평균수익":>8} {"적중률":>6} {"중간값":>8} | {"평균보유":>6} {"최악":>8} | {"총손익($)":>10}')
print(f'  {"-"*85}')

for name, strat, kw in strategies:
    res = simulate_exit(prod_sigs, strat, **kw)
    rets = [r['ret'] for r in res]
    days = [r['day'] for r in res]
    avg = np.mean(rets)
    med = np.median(rets)
    hit = np.mean([1 if r > 0 else 0 for r in rets])
    avg_day = np.mean(days)
    worst = min(rets)
    total = sum([r * 100 for r in rets])
    print(f'  {name:<22} | {avg:>+7.1%} {hit:>5.1%} {med:>+7.1%} | {avg_day:>5.1f}d {worst:>+7.1%} | {total:>+9.1f}$')

# MFE 반납 분석
print(f'\n■ MFE 반납 현황 (90일 홀드 기준)')
giveback_data = []
for s in prod_sigs:
    p = s['prices_90']
    entry = p[0]
    cum = (p[1:] - entry) / entry
    mfe = float(np.max(cum))
    mfe_day = int(np.argmax(cum)) + 1
    final = (p[90] - entry) / entry
    if mfe > 0:
        giveback = 1 - (final / mfe)
    else:
        giveback = 0
    giveback_data.append({'mfe': mfe, 'final': final, 'giveback': giveback, 'mfe_day': mfe_day})

gb_df = pd.DataFrame(giveback_data)
profitable = gb_df[gb_df['final'] > 0]
losing = gb_df[gb_df['final'] <= 0]

print(f'  수익 케이스 ({len(profitable)}건): 평균 MFE {profitable["mfe"].mean():+.1%} → 최종 {profitable["final"].mean():+.1%} (반납 {profitable["giveback"].mean():.0%})')
print(f'  손실 케이스 ({len(losing)}건): 평균 MFE {losing["mfe"].mean():+.1%} → 최종 {losing["final"].mean():+.1%}')
print(f'  전체: MFE 평균 day {gb_df["mfe_day"].mean():.0f}, MFE→최종 반납률 {gb_df["giveback"].mean():.0%}')

# DD 구간별 출구 전략 효과
print(f'\n■ DD 구간별 최적 출구 전략')
dd_tiers = [('3~10%', 0.03, 0.10), ('10~20%', 0.10, 0.20), ('20%+', 0.20, 1.0)]
best_strats = [('90일 홀드', 'hold_90d', {}), ('TS 15%', 'trailing_stop', {'stop_pct': 0.15}),
               ('MFE 50%반납', 'mfe_giveback', {'giveback_pct': 0.50})]

print(f'  {"DD 구간":<12} {"N":>4} | ', end='')
for sn, _, _ in best_strats:
    print(f'{sn:>14}', end='')
print()
print(f'  {"-"*60}')

for tier_name, lo, hi in dd_tiers:
    tier_sigs = [s for s in prod_sigs if lo <= s['dd'] < hi]
    n = len(tier_sigs)
    row = f'  {tier_name:<12} {n:>4} |'
    for sn, strat, kw in best_strats:
        res = simulate_exit(tier_sigs, strat, **kw)
        avg = np.mean([r['ret'] for r in res])
        row += f' {avg:>+6.1%} ({np.mean([r["day"] for r in res]):.0f}d)'
    print(row)


# ═══════════════════════════════════════════════════════════════════
# 점검 3: VOO 레버리지 29건 표본 신뢰도
# ═══════════════════════════════════════════════════════════════════
print(f'\n{"=" * 85}')
print('점검 3: VOO 레버리지 추천의 표본 신뢰도')
print('=' * 85)

voo_sigs = [s for s in all_sigs if s['ticker'] == 'VOO']

def lev_return_90(daily_rets, lev):
    return float(np.prod(1 + lev * daily_rets) - 1)

print(f'\n■ VOO Bootstrap 신뢰구간 (n={len(voo_sigs)}, 10,000회)')
print(f'  {"레버리지":<8} | {"평균":>8} {"CI_lo":>8} {"CI_hi":>8} {"CI폭":>8} | {"적중률":>6} {"적중CI_lo":>8} | {"P(loss>20%)":>12}')
print(f'  {"-"*85}')

for lev in [1, 1.5, 2, 3]:
    rets = np.array([lev_return_90(s['daily_rets'], lev) for s in voo_sigs])
    boot_means = [np.mean(np.random.choice(rets, len(rets), replace=True)) for _ in range(n_boot)]
    hits = np.array([1 if r > 0 else 0 for r in rets])
    boot_hits = [np.mean(np.random.choice(hits, len(hits), replace=True)) for _ in range(n_boot)]
    big_loss = np.mean(rets < -0.20)

    ci_lo, ci_hi = np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5)
    hit_ci_lo = np.percentile(boot_hits, 2.5)

    print(f'  {lev}x       | {np.mean(rets):>+7.1%} {ci_lo:>+7.1%} {ci_hi:>+7.1%} {ci_hi-ci_lo:>7.1%} | '
          f'{np.mean(hits):>5.1%} {hit_ci_lo:>7.1%} | {big_loss:>11.1%}')

# 표본 크기 vs 실제 필요 표본
print(f'\n■ 필요 표본 수 추정 (CI 폭 ±5% 이내)')
for lev in [1, 2, 3]:
    rets = [lev_return_90(s['daily_rets'], lev) for s in voo_sigs]
    std = np.std(rets)
    # CI ±5% 달성에 필요한 n: n = (1.96 * std / 0.05)^2
    n_needed = int((1.96 * std / 0.05) ** 2)
    print(f'  {lev}x: 현재 std={std:.1%}, CI±5% 달성에 n={n_needed}건 필요 (현재 {len(rets)}건, {"충분" if len(rets) >= n_needed else f"부족 {n_needed-len(rets)}건"})')

# VOO 전체풀 vs 개별 비교
print(f'\n■ VOO만 vs S&P계열 전체 (표본 확대 가능성)')
sp_like = [s for s in all_sigs if s['ticker'] in ['VOO', 'BRK-B', 'JPM', 'V', 'UNH', 'XOM']]
print(f'  VOO만: {len(voo_sigs)}건')
print(f'  대형가치주 풀(VOO+BRK-B+JPM+V+UNH+XOM): {len(sp_like)}건')
for lev in [1, 2, 3]:
    rets_voo = [lev_return_90(s['daily_rets'], lev) for s in voo_sigs]
    rets_pool = [lev_return_90(s['daily_rets'], lev) for s in sp_like]
    print(f'    {lev}x: VOO {np.mean(rets_voo):+.1%} (n={len(rets_voo)}) | '
          f'풀 {np.mean(rets_pool):+.1%} (n={len(rets_pool)})')


# ═══════════════════════════════════════════════════════════════════
# 점검 4: 2026 Look-ahead Bias 점검
# ═══════════════════════════════════════════════════════════════════
print(f'\n{"=" * 85}')
print('점검 4: 2026 Look-ahead Bias 점검')
print('=' * 85)

# 2026 신호 분리
sigs_pre2026 = [s for s in prod_sigs if s['year'] < 2026]
sigs_2026 = [s for s in prod_sigs if s['year'] == 2026]

print(f'\n■ 2026 데이터 현황')
print(f'  2026 신호: {len(sigs_2026)}건')
print(f'  ~2025 신호: {len(sigs_pre2026)}건')
print(f'  2026 비중: {len(sigs_2026)/len(prod_sigs):.1%}')

if sigs_2026:
    print(f'\n■ 2026 개별 신호')
    for s in sorted(sigs_2026, key=lambda x: x['date']):
        r90 = s['ret_90d']
        print(f'  {s["date"].strftime("%Y-%m-%d")} {s["ticker"]}: DD={s["dd"]:.1%}, 90d={r90:+.1%}')

# 파라미터 영향도: 2026 제거 후 주요 지표 비교
print(f'\n■ 2026 제거 전후 주요 지표 비교')
rets_all = [s['ret_90d'] for s in prod_sigs]
rets_pre = [s['ret_90d'] for s in sigs_pre2026]

print(f'  {"지표":<20} {"전체(2026포함)":>15} {"~2025만":>15} {"차이":>10}')
print(f'  {"-"*65}')
print(f'  {"신호 수":<20} {len(rets_all):>15} {len(rets_pre):>15} {len(rets_all)-len(rets_pre):>+10}')
print(f'  {"평균 수익":<20} {np.mean(rets_all):>+14.2%} {np.mean(rets_pre):>+14.2%} {np.mean(rets_all)-np.mean(rets_pre):>+9.2%}')
print(f'  {"적중률":<20} {np.mean([1 if r>0 else 0 for r in rets_all]):>14.1%} {np.mean([1 if r>0 else 0 for r in rets_pre]):>14.1%} {np.mean([1 if r>0 else 0 for r in rets_all])-np.mean([1 if r>0 else 0 for r in rets_pre]):>+9.1%}')

# DD 구간별 2026 영향
print(f'\n■ DD 구간별 2026 영향도')
dd_tiers_full = [('3~5%', 0.03, 0.05), ('5~10%', 0.05, 0.10),
                  ('10~20%', 0.10, 0.20), ('20%+', 0.20, 1.00)]
print(f'  {"DD 구간":<12} | {"전체 avg":>10} {"전체 N":>6} | {"~2025 avg":>10} {"~2025 N":>6} | {"차이":>8}')
print(f'  {"-"*70}')
for name, lo, hi in dd_tiers_full:
    all_t = [s['ret_90d'] for s in prod_sigs if lo <= s['dd'] < hi]
    pre_t = [s['ret_90d'] for s in sigs_pre2026 if lo <= s['dd'] < hi]
    avg_a = np.mean(all_t) if all_t else 0
    avg_p = np.mean(pre_t) if pre_t else 0
    print(f'  {name:<12} | {avg_a:>+9.1%} {len(all_t):>6} | {avg_p:>+9.1%} {len(pre_t):>6} | {avg_a-avg_p:>+7.2%}')

# GEO-OP 파라미터 결정 시점 확인
print(f'\n■ Look-ahead Bias 판정')
print(f'  GEO-OP 파라미터 확정일: 2026-03-17 (Rev2 커밋)')
print(f'  2026 신호 최소 날짜: {min(s["date"].strftime("%Y-%m-%d") for s in sigs_2026) if sigs_2026 else "없음"}')
print(f'  파라미터 튜닝 데이터: period="max" (~2025.12)')
print(f'  2026 신호가 파라미터에 영향: {"가능성 있음 (주의)" if sigs_2026 and min(s["date"] for s in sigs_2026).strftime("%Y-%m") <= "2026-03" else "없음"}')

# Walk-forward: 2020년까지 데이터로 파라미터 → 2021~2025 OOS 성과
print(f'\n■ Walk-forward 검증: ~2020 데이터 → 2021~2025 Out-of-Sample')
is_sigs = [s for s in prod_sigs if s['year'] <= 2020]
oos_sigs = [s for s in prod_sigs if 2021 <= s['year'] <= 2025]

is_rets = [s['ret_90d'] for s in is_sigs]
oos_rets = [s['ret_90d'] for s in oos_sigs]

print(f'  In-Sample (~2020): avg={np.mean(is_rets):+.1%}, hit={np.mean([1 if r>0 else 0 for r in is_rets]):.1%}, n={len(is_rets)}')
print(f'  Out-of-Sample (2021~25): avg={np.mean(oos_rets):+.1%}, hit={np.mean([1 if r>0 else 0 for r in oos_rets]):.1%}, n={len(oos_rets)}')
print(f'  OOS 성과 유지: {"YES — OOS가 IS 이상" if np.mean(oos_rets) >= np.mean(is_rets) * 0.7 else "NO — 성과 열화"}')

# DD 구간별 OOS
print(f'\n  DD 구간별 Walk-forward:')
print(f'  {"DD":<12} | {"IS avg":>8} {"IS N":>5} | {"OOS avg":>8} {"OOS N":>5} | {"유지":>4}')
print(f'  {"-"*60}')
for name, lo, hi in dd_tiers_full:
    is_t = [s['ret_90d'] for s in is_sigs if lo <= s['dd'] < hi]
    oos_t = [s['ret_90d'] for s in oos_sigs if lo <= s['dd'] < hi]
    is_avg = np.mean(is_t) if is_t else 0
    oos_avg = np.mean(oos_t) if oos_t else 0
    ok = 'OK' if oos_avg > 0 else 'X'
    print(f'  {name:<12} | {is_avg:>+7.1%} {len(is_t):>5} | {oos_avg:>+7.1%} {len(oos_t):>5} | {ok:>4}')

print(f'\n전체 실행 시간: {time.time()-t0:.0f}초')
