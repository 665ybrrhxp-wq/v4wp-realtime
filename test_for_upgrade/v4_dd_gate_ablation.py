"""
DD Gate 제거 실험 (Ablation Test)
=================================
GEO-OP 파이프라인에서 BUY_DD_GATE만 ON/OFF 비교.

비교군:
  A) DD Gate ON  (20d 고점 대비 3% 이상 하락 필수) — 현재 프로덕션
  B) DD Gate OFF (DD 조건 없이 모든 신호 허용)
"""
import sys, os, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import yfinance as yf
from real_market_backtest import (
    calc_v4_score, calc_v4_subindicators,
    detect_signal_events, build_price_filter, smooth_earnings_volume,
)

# ═══════════════════════════════════════════════════════════
# GEO-OP 파라미터 (DD gate만 토글)
# ═══════════════════════════════════════════════════════════
TICKERS = {
    'TSLA': 'Tech', 'NVDA': 'Tech', 'AVGO': 'Tech',
    'AMZN': 'Tech', 'GOOGL': 'Tech', 'PLTR': 'Tech',
    'COIN': 'Fintech', 'HOOD': 'Fintech',
    'HIMS': 'Growth', 'RKLB': 'Growth', 'JOBY': 'Growth',
    'IONQ': 'Quantum',
    'QQQ': 'Benchmark', 'VOO': 'Benchmark',
}

SIGNAL_TH = 0.05
COOLDOWN = 5
ER_Q = 80
ATR_Q = 40
LOOKBACK = 252
CONFIRM_DAYS = 1
DIVGATE = 3
BUY_DD_LB = 20
BUY_DD_TH = 0.03  # 프로덕션 DD 게이트


def download_data(ticker, years=5):
    t = yf.Ticker(ticker)
    df = t.history(period=f'{years}y', auto_adjust=True)
    if df is None or len(df) == 0:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.dropna(subset=['Close', 'Volume'])
    df = df[df['Close'] > 0]
    return df


def get_buy_signals(df, score, use_dd_gate=True):
    """매수 신호 추출. use_dd_gate=False면 DD 체크 건너뜀."""
    events = detect_signal_events(score, th=SIGNAL_TH, cooldown=COOLDOWN)
    pf = build_price_filter(df, er_q=ER_Q, atr_q=ATR_Q, lookback=LOOKBACK)

    close = df['Close'].values
    rolling_high = pd.Series(close).rolling(BUY_DD_LB, min_periods=1).max().values
    n = len(df)
    buy_indices = []
    blocked = 0

    for ev in events:
        if ev['type'] != 'bottom':
            continue
        if not pf(ev['peak_idx']):
            continue
        dur = ev['end_idx'] - ev['start_idx'] + 1
        ci = ev['start_idx'] + CONFIRM_DAYS - 1
        if ci > ev['end_idx'] or dur < CONFIRM_DAYS or ci >= n:
            continue

        if use_dd_gate:
            pidx = ev['peak_idx']
            rh = rolling_high[pidx]
            dd = (rh - close[pidx]) / rh if rh > 0 else 0
            if dd < BUY_DD_TH:
                blocked += 1
                continue

        buy_indices.append(ci)

    return buy_indices, blocked


def evaluate_signals(close, buy_indices):
    """신호 품질 평가."""
    n = len(close)
    fwd_30 = []; fwd_90 = []; fwd_180 = []
    hits_90 = 0; total_90 = 0
    max_dds = []

    for idx in buy_indices:
        if idx + 30 < n:
            fwd_30.append((close[idx + 30] / close[idx] - 1) * 100)
        if idx + 90 < n:
            total_90 += 1
            fr = (close[idx + 90] / close[idx] - 1) * 100
            fwd_90.append(fr)
            if fr > 0: hits_90 += 1
        if idx + 180 < n:
            fwd_180.append((close[idx + 180] / close[idx] - 1) * 100)
        # MDD within 90d
        end_i = min(idx + 90, n)
        if end_i > idx + 1:
            min_p = min(close[idx:end_i])
            max_dds.append((min_p / close[idx] - 1) * 100)

    return {
        'n_signals': len(buy_indices),
        'hit_rate_90': (hits_90 / total_90 * 100) if total_90 > 0 else 0,
        'avg_fwd_30': np.mean(fwd_30) if fwd_30 else 0,
        'avg_fwd_90': np.mean(fwd_90) if fwd_90 else 0,
        'med_fwd_90': np.median(fwd_90) if fwd_90 else 0,
        'avg_fwd_180': np.mean(fwd_180) if fwd_180 else 0,
        'avg_mdd': np.mean(max_dds) if max_dds else 0,
        'worst_mdd': min(max_dds) if max_dds else 0,
        'fwd_90_list': fwd_90,
        'fwd_30_list': fwd_30,
        'n_evaluated_90': total_90,
    }


def main():
    sep = '=' * 120
    print(sep)
    print("  DD Gate Ablation Test: ON (3%) vs OFF")
    print("  GEO-OP 파이프라인 동일, DD 게이트만 토글")
    print(sep)

    # Load data
    print("\n  데이터 로딩...")
    data = {}
    for tk in TICKERS:
        df = download_data(tk, years=5)
        if df is None or len(df) < 300:
            print(f"    {tk}: SKIP")
            continue
        df = smooth_earnings_volume(df, ticker=tk)
        data[tk] = df
        print(f"    {tk}: {len(df)} bars ({len(df)/252:.1f}yr)")
    print(f"  {len(data)} tickers loaded.\n")

    # ═══════════════════════════════════════════════════════
    # Run both modes
    # ═══════════════════════════════════════════════════════
    results_on = {}   # DD gate ON
    results_off = {}  # DD gate OFF

    print(f"  {'Ticker':<8} │ {'DD ON':>6} {'DD OFF':>7} {'Blocked':>8} │ "
          f"{'Hit90 ON':>9} {'Hit90 OFF':>10} │ {'Fwd90 ON':>9} {'Fwd90 OFF':>10} │ "
          f"{'MDD ON':>7} {'MDD OFF':>8}")
    print("  " + "-" * 105)

    for tk, df in data.items():
        score = calc_v4_score(df, w=20, divgate_days=DIVGATE)
        close = df['Close'].values

        # DD gate ON (production)
        idx_on, _ = get_buy_signals(df, score, use_dd_gate=True)
        eval_on = evaluate_signals(close, idx_on)
        results_on[tk] = eval_on

        # DD gate OFF
        idx_off, blocked = get_buy_signals(df, score, use_dd_gate=False)
        eval_off = evaluate_signals(close, idx_off)
        results_off[tk] = eval_off

        print(f"  {tk:<8} │ {eval_on['n_signals']:>5}개 {eval_off['n_signals']:>6}개 {blocked:>7}건 │ "
              f"{eval_on['hit_rate_90']:>8.1f}% {eval_off['hit_rate_90']:>9.1f}% │ "
              f"{eval_on['avg_fwd_90']:>+8.1f}% {eval_off['avg_fwd_90']:>+9.1f}% │ "
              f"{eval_on['avg_mdd']:>+6.1f}% {eval_off['avg_mdd']:>+7.1f}%")

    # ═══════════════════════════════════════════════════════
    # Aggregate Summary
    # ═══════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  전체 합산 비교")
    print(sep)

    total_on = sum(r['n_signals'] for r in results_on.values())
    total_off = sum(r['n_signals'] for r in results_off.values())

    all_fwd90_on = []; all_fwd90_off = []
    all_fwd30_on = []; all_fwd30_off = []
    for tk in results_on:
        all_fwd90_on.extend(results_on[tk]['fwd_90_list'])
        all_fwd90_off.extend(results_off[tk]['fwd_90_list'])
        all_fwd30_on.extend(results_on[tk]['fwd_30_list'])
        all_fwd30_off.extend(results_off[tk]['fwd_30_list'])

    n_eval_on = sum(r['n_evaluated_90'] for r in results_on.values())
    n_eval_off = sum(r['n_evaluated_90'] for r in results_off.values())
    hits_on = sum(1 for f in all_fwd90_on if f > 0)
    hits_off = sum(1 for f in all_fwd90_off if f > 0)

    hit_on = (hits_on / n_eval_on * 100) if n_eval_on > 0 else 0
    hit_off = (hits_off / n_eval_off * 100) if n_eval_off > 0 else 0

    print(f"""
  ┌──────────────────────────────────────────────────────┐
  │                DD Gate ON (3%)    DD Gate OFF         │
  ├──────────────────────────────────────────────────────┤
  │  총 신호 수       {total_on:>8}개       {total_off:>8}개         │
  │  평가 가능(90d)   {n_eval_on:>8}개       {n_eval_off:>8}개         │
  │                                                      │
  │  적중률 (90일)    {hit_on:>8.1f}%       {hit_off:>8.1f}%         │
  │  평균 Fwd 30일    {np.mean(all_fwd30_on) if all_fwd30_on else 0:>+8.1f}%       {np.mean(all_fwd30_off) if all_fwd30_off else 0:>+8.1f}%         │
  │  평균 Fwd 90일    {np.mean(all_fwd90_on) if all_fwd90_on else 0:>+8.1f}%       {np.mean(all_fwd90_off) if all_fwd90_off else 0:>+8.1f}%         │
  │  중앙값 Fwd 90일  {np.median(all_fwd90_on) if all_fwd90_on else 0:>+8.1f}%       {np.median(all_fwd90_off) if all_fwd90_off else 0:>+8.1f}%         │
  └──────────────────────────────────────────────────────┘""")

    # ═══════════════════════════════════════════════════════
    # DD gate OFF에서 추가된 신호 분석
    # ═══════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  DD Gate가 차단한 신호들의 품질 (OFF에만 존재하는 신호)")
    print(sep)

    # 차단된 신호 = OFF에 있지만 ON에 없는 것
    blocked_fwd90 = []
    blocked_fwd30 = []
    for tk in data:
        score = calc_v4_score(data[tk], w=20, divgate_days=DIVGATE)
        close = data[tk]['Close'].values
        n = len(close)

        idx_on_set = set(get_buy_signals(data[tk], score, True)[0])
        idx_off, _ = get_buy_signals(data[tk], score, False)

        for idx in idx_off:
            if idx not in idx_on_set:
                # 이 신호는 DD gate에 의해 차단됨
                if idx + 90 < n:
                    blocked_fwd90.append((close[idx + 90] / close[idx] - 1) * 100)
                if idx + 30 < n:
                    blocked_fwd30.append((close[idx + 30] / close[idx] - 1) * 100)

    if blocked_fwd90:
        blocked_hits = sum(1 for f in blocked_fwd90 if f > 0)
        blocked_hit_rate = blocked_hits / len(blocked_fwd90) * 100
        print(f"""
  차단된 신호 수: {len(blocked_fwd90)}개 (90일 평가 가능)

  차단된 신호 적중률 (90일): {blocked_hit_rate:.1f}%
  차단된 신호 Fwd 90일: 평균 {np.mean(blocked_fwd90):+.1f}%, 중앙값 {np.median(blocked_fwd90):+.1f}%
  차단된 신호 Fwd 30일: 평균 {np.mean(blocked_fwd30):+.1f}%

  Fwd 90일 분포:
    min  = {min(blocked_fwd90):+.1f}%
    25%  = {np.percentile(blocked_fwd90, 25):+.1f}%
    50%  = {np.percentile(blocked_fwd90, 50):+.1f}%
    75%  = {np.percentile(blocked_fwd90, 75):+.1f}%
    max  = {max(blocked_fwd90):+.1f}%""")

        # 차단 vs 통과 비교
        print(f"""
  ┌──────────────────────────────────────────────────────┐
  │  비교            통과 신호          차단된 신호        │
  ├──────────────────────────────────────────────────────┤
  │  적중률(90d)     {hit_on:>8.1f}%          {blocked_hit_rate:>8.1f}%        │
  │  Fwd 90d 평균    {np.mean(all_fwd90_on):>+8.1f}%          {np.mean(blocked_fwd90):>+8.1f}%        │
  │  Fwd 90d 중앙    {np.median(all_fwd90_on):>+8.1f}%          {np.median(blocked_fwd90):>+8.1f}%        │
  │  Fwd 30d 평균    {np.mean(all_fwd30_on):>+8.1f}%          {np.mean(blocked_fwd30):>+8.1f}%        │
  └──────────────────────────────────────────────────────┘""")

        diff_hit = hit_on - blocked_hit_rate
        diff_fwd = np.mean(all_fwd90_on) - np.mean(blocked_fwd90)
        print(f"\n  결론:")
        if diff_hit > 3:
            print(f"    DD Gate가 적중률을 {diff_hit:+.1f}%p 개선 → 유지 권장")
        elif diff_hit > -3:
            print(f"    DD Gate 효과 미미 ({diff_hit:+.1f}%p) → 제거 검토 가능")
        else:
            print(f"    DD Gate가 오히려 좋은 신호를 차단 ({diff_hit:+.1f}%p) → 제거 고려")

        if diff_fwd > 2:
            print(f"    Fwd 90일 수익률 차이: {diff_fwd:+.1f}%p → DD Gate가 노이즈 필터링 역할")
        elif diff_fwd > -2:
            print(f"    Fwd 90일 수익률 차이: {diff_fwd:+.1f}%p → 유의미한 차이 없음")
        else:
            print(f"    Fwd 90일 수익률 차이: {diff_fwd:+.1f}%p → 차단된 신호가 더 좋았음")
    else:
        print("\n  차단된 신호 없음 (DD gate가 아무것도 차단하지 않음)")

    print(f"\n{sep}")
    print("  Done.")
    print(sep)


if __name__ == '__main__':
    main()
