"""DD 신뢰도 × 레버리지 최적 조합 실험.

목표: 4단계 DD 신뢰도(보통/양호/강력/극강) 중
      어디서부터 레버리지(2x, 3x) 매수가 유리한지 검증.

핵심: 레버리지 ETF는 일별 수익률을 곱하므로
      단순히 "2 × 90d수익"이 아니라 경로 의존적 변동성 감쇠 반영.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf
from real_market_backtest import (
    calc_v4_score, calc_v4_subindicators, detect_signal_events,
    build_price_filter, smooth_earnings_volume,
)

# ── 설정 ──
TICKERS_MAX = ['QQQ', 'VOO', 'GOOGL', 'AMZN', 'NVDA', 'AVGO', 'TSLA',
               'AAPL', 'MSFT', 'META', 'JPM', 'BRK-B', 'V', 'UNH', 'XOM']
PARAMS = dict(signal_threshold=0.05, er_quantile=80, atr_quantile=40,
              divgate_days=3, confirm_days=1, cooldown=5,
              buy_dd_lookback=20, buy_dd_threshold=0.0)  # DD gate OFF → 모든 신호 수집

LEVERAGES = [1, 1.5, 2, 3]
HORIZONS = [30, 60, 90]
DD_TIERS = [
    ('보통 (3~5%)',  0.03, 0.05),
    ('양호 (5~10%)', 0.05, 0.10),
    ('강력 (10~20%)', 0.10, 0.20),
    ('극강 (20%+)',  0.20, 1.00),
]


def collect_signals_with_paths(ticker, period='max'):
    """신호 수집 + 신호 후 90일 일별 가격 경로 저장."""
    tk = yf.Ticker(ticker)
    df = tk.history(period=period, auto_adjust=True)
    if df is None or len(df) < 252:
        return []

    df = smooth_earnings_volume(df, ticker=ticker)
    score = calc_v4_score(df, w=20, divgate_days=3)
    events = detect_signal_events(score, th=0.05, cooldown=5)
    pf = build_price_filter(df, er_q=80, atr_q=40, lookback=252)
    subind = calc_v4_subindicators(df, w=20, divgate_days=3)

    close = df['Close'].values
    rolling_high = df['Close'].rolling(20, min_periods=1).max().values

    signals = []
    for e in events:
        if e['type'] != 'bottom':
            continue
        if not pf(e['peak_idx']):
            continue

        pidx = e['peak_idx']
        if pidx >= len(close) - 90:  # 90일 forward 필요
            continue

        price = close[pidx]
        rh = rolling_high[pidx]
        dd = (rh - price) / rh if rh > 0 else 0

        if dd < 0.03:  # 3% 미만은 DD gate에서 차단되므로 제외
            continue

        # 신호 후 90일 일별 수익률 경로
        daily_returns = []
        for d in range(1, 91):
            if pidx + d < len(close):
                dr = (close[pidx + d] - close[pidx + d - 1]) / close[pidx + d - 1]
                daily_returns.append(dr)

        if len(daily_returns) < 90:
            continue

        signals.append({
            'ticker': ticker,
            'date': df.index[pidx].strftime('%Y-%m-%d'),
            'dd_pct': dd,
            'score': float(score.iloc[pidx]),
            'price': price,
            'daily_returns': np.array(daily_returns),
            # 단순 forward returns
            'fwd_30d': (close[pidx + 30] - price) / price,
            'fwd_60d': (close[pidx + 60] - price) / price,
            'fwd_90d': (close[pidx + 90] - price) / price,
        })

    return signals


def calc_leveraged_returns(daily_returns, leverage):
    """일별 수익률에 레버리지를 적용한 누적 수익률 계산.

    레버리지 ETF 방식: 매일 leverage × daily_return.
    변동성 감쇠(volatility decay) 자연 반영.
    """
    cum = np.cumprod(1 + leverage * daily_returns)
    return {
        30: cum[29] - 1 if len(cum) >= 30 else np.nan,
        60: cum[59] - 1 if len(cum) >= 60 else np.nan,
        90: cum[89] - 1 if len(cum) >= 90 else np.nan,
    }


def calc_mfe_mae(daily_returns, leverage):
    """레버리지 적용 후 MFE(최대유리편찬)/MAE(최대불리편찬) 60일 기준."""
    cum = np.cumprod(1 + leverage * daily_returns[:60]) - 1
    return float(np.max(cum)), float(np.min(cum))


def bootstrap_ci(vals, n_boot=5000):
    """Bootstrap 95% CI."""
    if len(vals) < 3:
        return np.mean(vals), np.nan, np.nan
    means = [np.mean(np.random.choice(vals, len(vals), replace=True))
             for _ in range(n_boot)]
    return np.mean(vals), np.percentile(means, 2.5), np.percentile(means, 97.5)


def get_dd_tier(dd):
    for name, lo, hi in DD_TIERS:
        if lo <= dd < hi:
            return name
    return None


# ── 메인 ──
print("DD 신뢰도 × 레버리지 최적 조합 실험")
print("=" * 80)

t0 = time.time()
all_signals = []
for ticker in TICKERS_MAX:
    sigs = collect_signals_with_paths(ticker, period='max')
    print(f"  {ticker}: {len(sigs)}개 신호")
    all_signals.extend(sigs)

print(f"\n총 {len(all_signals)}개 신호 수집 ({time.time()-t0:.0f}초)\n")

# ── 실험 1: DD 신뢰도 × 레버리지 수익률 ──
print("=" * 80)
print("1. DD 신뢰도 × 레버리지 90일 수익률")
print("=" * 80)

header = f"{'DD 신뢰도':<16} {'N':>5}"
for lev in LEVERAGES:
    header += f" | {lev}x수익  {lev}x적중  {lev}xMFE  {lev}xMAE"
print(header)
print("-" * len(header))

tier_results = {}  # tier → {lev → [returns]}

for tier_name, dd_lo, dd_hi in DD_TIERS:
    sigs = [s for s in all_signals if dd_lo <= s['dd_pct'] < dd_hi]
    n = len(sigs)
    tier_results[tier_name] = {}

    row = f"{tier_name:<16} {n:>5}"
    for lev in LEVERAGES:
        rets_90 = []
        mfes = []
        maes = []
        for s in sigs:
            lr = calc_leveraged_returns(s['daily_returns'], lev)
            rets_90.append(lr[90])
            mfe, mae = calc_mfe_mae(s['daily_returns'], lev)
            mfes.append(mfe)
            maes.append(mae)

        tier_results[tier_name][lev] = rets_90
        avg = np.mean(rets_90) if rets_90 else 0
        hit = np.mean([1 if r > 0 else 0 for r in rets_90]) if rets_90 else 0
        avg_mfe = np.mean(mfes) if mfes else 0
        avg_mae = np.mean(maes) if maes else 0

        row += f" | {avg:+6.1%}  {hit:5.1%}  {avg_mfe:+6.1%}  {avg_mae:+6.1%}"

    print(row)


# ── 실험 2: 레버리지 효율 (1x 대비 증분) ──
print(f"\n{'=' * 80}")
print("2. 레버리지 효율: 1x 대비 수익 증분 (90d)")
print("=" * 80)
print(f"{'DD 신뢰도':<16} {'N':>5} | {'1x→1.5x':>10} {'1x→2x':>10} {'1x→3x':>10} | {'최적':>6}")
print("-" * 85)

for tier_name, dd_lo, dd_hi in DD_TIERS:
    sigs = [s for s in all_signals if dd_lo <= s['dd_pct'] < dd_hi]
    n = len(sigs)

    base_avg = np.mean(tier_results[tier_name][1]) if tier_results[tier_name][1] else 0
    deltas = {}
    best_lev = 1
    best_delta = 0

    row = f"{tier_name:<16} {n:>5} |"
    for lev in [1.5, 2, 3]:
        avg = np.mean(tier_results[tier_name][lev]) if tier_results[tier_name][lev] else 0
        delta = avg - base_avg
        deltas[lev] = delta
        row += f" {delta:>+9.1%}"
        if delta > best_delta:
            best_delta = delta
            best_lev = lev

    row += f" | {best_lev}x"
    print(row)


# ── 실험 3: 리스크 조정 수익 (수익/MAE) ──
print(f"\n{'=' * 80}")
print("3. 리스크 효율: 수익/MAE 비율 (높을수록 좋음)")
print("=" * 80)
print(f"{'DD 신뢰도':<16} {'N':>5} | {'1x':>8} {'1.5x':>8} {'2x':>8} {'3x':>8} | {'최적':>6}")
print("-" * 80)

for tier_name, dd_lo, dd_hi in DD_TIERS:
    sigs = [s for s in all_signals if dd_lo <= s['dd_pct'] < dd_hi]
    n = len(sigs)
    row = f"{tier_name:<16} {n:>5} |"
    best_lev = 1
    best_ratio = 0

    for lev in LEVERAGES:
        rets = []
        maes = []
        for s in sigs:
            lr = calc_leveraged_returns(s['daily_returns'], lev)
            rets.append(lr[90])
            _, mae = calc_mfe_mae(s['daily_returns'], lev)
            maes.append(abs(mae))

        avg_ret = np.mean(rets) if rets else 0
        avg_mae = np.mean(maes) if maes else 1
        ratio = avg_ret / avg_mae if avg_mae > 0 else 0
        row += f" {ratio:>7.2f}x"

        if ratio > best_ratio:
            best_ratio = ratio
            best_lev = lev

    row += f" | {best_lev}x"
    print(row)


# ── 실험 4: 손실 확률 (90일 후 마이너스일 확률) ──
print(f"\n{'=' * 80}")
print("4. 손실 확률: 90일 후 원금 손실 비율")
print("=" * 80)
print(f"{'DD 신뢰도':<16} {'N':>5} | {'1x':>8} {'1.5x':>8} {'2x':>8} {'3x':>8}")
print("-" * 70)

for tier_name, dd_lo, dd_hi in DD_TIERS:
    sigs = [s for s in all_signals if dd_lo <= s['dd_pct'] < dd_hi]
    n = len(sigs)
    row = f"{tier_name:<16} {n:>5} |"

    for lev in LEVERAGES:
        losses = []
        for s in sigs:
            lr = calc_leveraged_returns(s['daily_returns'], lev)
            losses.append(1 if lr[90] < 0 else 0)
        loss_pct = np.mean(losses) if losses else 0
        row += f" {loss_pct:>7.1%}"

    print(row)


# ── 실험 5: Bootstrap 유의성 — 레버리지 프리미엄 ──
print(f"\n{'=' * 80}")
print("5. Bootstrap 검증: 2x 레버리지가 1x보다 유의하게 좋은가?")
print("=" * 80)
print(f"{'DD 신뢰도':<16} {'N':>5} | {'1x avg':>10} {'2x avg':>10} {'Delta':>10} {'CI_lo':>8} {'CI_hi':>8} {'유의':>6}")
print("-" * 85)

np.random.seed(42)
for tier_name, dd_lo, dd_hi in DD_TIERS:
    sigs = [s for s in all_signals if dd_lo <= s['dd_pct'] < dd_hi]
    n = len(sigs)

    rets_1x = np.array([s['fwd_90d'] for s in sigs])
    rets_2x = []
    for s in sigs:
        lr = calc_leveraged_returns(s['daily_returns'], 2)
        rets_2x.append(lr[90])
    rets_2x = np.array(rets_2x)

    # Bootstrap: 2x - 1x 차이의 CI
    deltas = rets_2x - rets_1x
    mean_delta = np.mean(deltas)
    boot_deltas = [np.mean(np.random.choice(deltas, len(deltas), replace=True))
                   for _ in range(10000)]
    ci_lo = np.percentile(boot_deltas, 2.5)
    ci_hi = np.percentile(boot_deltas, 97.5)
    sig = "YES" if ci_lo > 0 else "NO"

    print(f"{tier_name:<16} {n:>5} | {np.mean(rets_1x):>+9.1%} {np.mean(rets_2x):>+9.1%} "
          f"{mean_delta:>+9.1%} {ci_lo:>+7.1%} {ci_hi:>+7.1%} {sig:>6}")


# ── 실험 6: 30/60/90일 기간별 최적 레버리지 ──
print(f"\n{'=' * 80}")
print("6. 보유 기간별 최적 레버리지")
print("=" * 80)

for horizon in HORIZONS:
    print(f"\n  ── {horizon}일 보유 ──")
    print(f"  {'DD 신뢰도':<16} {'N':>5} | {'1x':>8} {'1.5x':>8} {'2x':>8} {'3x':>8} | {'최적':>6}")
    print(f"  {'-' * 75}")

    for tier_name, dd_lo, dd_hi in DD_TIERS:
        sigs = [s for s in all_signals if dd_lo <= s['dd_pct'] < dd_hi]
        n = len(sigs)
        row = f"  {tier_name:<16} {n:>5} |"
        best_lev = 1
        best_ret = -999

        for lev in LEVERAGES:
            rets = []
            for s in sigs:
                lr = calc_leveraged_returns(s['daily_returns'], lev)
                rets.append(lr[horizon])
            avg = np.mean(rets) if rets else 0
            row += f" {avg:>+7.1%}"
            if avg > best_ret:
                best_ret = avg
                best_lev = lev

        row += f" | {best_lev}x"
        print(row)


# ── 실험 7: 최악 시나리오 (하위 10% 손실) ──
print(f"\n{'=' * 80}")
print("7. 최악 시나리오: 90일 수익 하위 10% 평균 (Tail Risk)")
print("=" * 80)
print(f"{'DD 신뢰도':<16} {'N':>5} | {'1x':>10} {'1.5x':>10} {'2x':>10} {'3x':>10}")
print("-" * 70)

for tier_name, dd_lo, dd_hi in DD_TIERS:
    sigs = [s for s in all_signals if dd_lo <= s['dd_pct'] < dd_hi]
    n = len(sigs)
    row = f"{tier_name:<16} {n:>5} |"

    for lev in LEVERAGES:
        rets = []
        for s in sigs:
            lr = calc_leveraged_returns(s['daily_returns'], lev)
            rets.append(lr[90])
        if rets:
            sorted_rets = sorted(rets)
            tail_n = max(1, len(sorted_rets) // 10)
            tail_avg = np.mean(sorted_rets[:tail_n])
        else:
            tail_avg = 0
        row += f" {tail_avg:>+9.1%}"

    print(row)


# ── 최종 결론 ──
print(f"\n{'=' * 80}")
print("최종 결론: DD 신뢰도별 레버리지 추천")
print("=" * 80)

for tier_name, dd_lo, dd_hi in DD_TIERS:
    sigs = [s for s in all_signals if dd_lo <= s['dd_pct'] < dd_hi]
    n = len(sigs)

    # 각 레버리지별 90일 수익, 적중률, tail risk 계산
    lev_stats = {}
    for lev in LEVERAGES:
        rets = []
        for s in sigs:
            lr = calc_leveraged_returns(s['daily_returns'], lev)
            rets.append(lr[90])
        avg_ret = np.mean(rets)
        hit = np.mean([1 if r > 0 else 0 for r in rets])
        sorted_rets = sorted(rets)
        tail_n = max(1, len(sorted_rets) // 10)
        tail = np.mean(sorted_rets[:tail_n])
        lev_stats[lev] = {'ret': avg_ret, 'hit': hit, 'tail': tail}

    # 추천: 적중률 60% 이상 + tail > -30% + 수익 극대화
    recommended = 1
    for lev in [3, 2, 1.5]:
        s = lev_stats[lev]
        if s['hit'] >= 0.60 and s['tail'] > -0.40:
            recommended = lev
            break

    print(f"\n  {tier_name} (n={n})")
    for lev in LEVERAGES:
        s = lev_stats[lev]
        marker = " ◀ 추천" if lev == recommended else ""
        print(f"    {lev}x: 수익 {s['ret']:+.1%}, 적중 {s['hit']:.1%}, "
              f"최악10% {s['tail']:+.1%}{marker}")

print(f"\n전체 실행 시간: {time.time()-t0:.0f}초")
