"""
매도 신호 대안 테스트: S_Force > 0 AND S_Div < 0 (베어리시 다이버전스)
=====================================================================
가격 모멘텀은 양(+)인데 거래량 발산이 음(-) → "가격은 오르지만 거래량이 안 따라옴"
→ 스마트머니 이탈, 고점 임박 신호?

비교 대상:
  A) 현재 매도: S_Force < 0 AND S_Div < 0 (score < -th)  → 이미 무효 판정
  B) 신규 매도: S_Force > 0 AND S_Div < 0 (divergence)    → 테스트 대상
  C) 추가 매도: S_Force < 0 AND S_Div > 0 (역방향 발산)   → 참고용
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
    calc_v4_subindicators, smooth_earnings_volume,
)

TICKERS = {
    'TSLA': 'Tech', 'NVDA': 'Tech', 'AVGO': 'Tech',
    'AMZN': 'Tech', 'GOOGL': 'Tech', 'PLTR': 'Tech',
    'COIN': 'Fintech', 'HOOD': 'Fintech',
    'HIMS': 'Growth', 'RKLB': 'Growth', 'JOBY': 'Growth',
    'IONQ': 'Quantum',
    'QQQ': 'Benchmark', 'VOO': 'Benchmark',
}

DIVGATE = 3


def download_max(ticker):
    t = yf.Ticker(ticker)
    df = t.history(period='max', auto_adjust=True)
    if df is None or len(df) == 0:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.dropna(subset=['Close', 'Volume'])
    df = df[df['Close'] > 0]
    return df


def detect_divergence_sells(subind, close_arr, mode='force_pos_div_neg',
                            strength_th=0.0, min_streak=3, cooldown=10,
                            rolling_high_lb=20, min_drop_from_high=0.0):
    """
    다이버전스 기반 매도 신호 감지.

    mode:
      'force_pos_div_neg': S_Force > 0 AND S_Div < 0  (가격↑ 거래량↓)
      'force_neg_div_pos': S_Force < 0 AND S_Div > 0  (가격↓ 거래량↑)
      'original':          S_Force < 0 AND S_Div < 0  (기존 매도)

    strength_th: |S_Force| > th AND |S_Div| > th 조건
    min_streak: 조건 연속 N일 이상
    cooldown: 신호 간 최소 간격
    """
    n = len(subind)
    sf = subind['s_force'].values
    sd = subind['s_div'].values

    # Condition mask
    if mode == 'force_pos_div_neg':
        cond = (sf > strength_th) & (sd < -strength_th)
    elif mode == 'force_neg_div_pos':
        cond = (sf < -strength_th) & (sd > strength_th)
    elif mode == 'original':
        cond = (sf < -strength_th) & (sd < -strength_th)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Streak detection
    streak = np.zeros(n, dtype=int)
    for i in range(1, n):
        if cond[i]:
            streak[i] = streak[i-1] + 1
        else:
            streak[i] = 0

    # Detect signal events: when streak ends (transitions from >= min_streak to < min_streak)
    # Or: emit signal at the END of the streak
    signals = []
    last_signal_idx = -cooldown - 1

    i = 0
    while i < n:
        if streak[i] >= min_streak:
            # Start of a valid streak region
            start = i - streak[i] + 1
            # Find end of this streak
            while i + 1 < n and cond[i + 1]:
                i += 1
            end = i

            # Signal at end of streak (when divergence ends)
            sig_idx = end
            if sig_idx - last_signal_idx >= cooldown and sig_idx < n:
                # Compute divergence strength = |S_Force| * |S_Div| (geometric)
                div_strength = np.sqrt(abs(sf[sig_idx]) * abs(sd[sig_idx]))
                duration = end - start + 1

                signals.append({
                    'idx': sig_idx,
                    'start_idx': start,
                    'duration': duration,
                    'strength': div_strength,
                    's_force': sf[sig_idx],
                    's_div': sd[sig_idx],
                    'price': close_arr[sig_idx],
                })
                last_signal_idx = sig_idx
        i += 1

    return signals


def calc_forward_returns(signals, close_arr, n):
    """Forward returns 계산"""
    periods = [7, 14, 30, 60, 90, 180]
    for sig in signals:
        idx = sig['idx']
        for d in periods:
            fi = idx + d
            if fi < n:
                sig[f'fwd_{d}d'] = (close_arr[fi] / close_arr[idx] - 1) * 100
            else:
                sig[f'fwd_{d}d'] = None

        # MDD within 90d
        end_i = min(idx + 90, n)
        if end_i > idx + 1:
            min_p = min(close_arr[idx:end_i])
            sig['mdd_90'] = (min_p / close_arr[idx] - 1) * 100
        else:
            sig['mdd_90'] = None
    return signals


def print_summary(label, signals, mode_desc):
    """모드별 요약 출력"""
    periods = [7, 14, 30, 60, 90, 180]

    print(f"\n  [{label}] {mode_desc}")
    print(f"  총 신호: {len(signals)}건")

    if not signals:
        return

    print(f"\n  {'기간':<8} │ {'N':>5} {'하락비율':>8} {'평균변동':>9} {'중앙값':>9} │ {'최대하락':>9} {'최대상승':>9}")
    print(f"  {'-'*70}")

    for d in periods:
        vals = [s[f'fwd_{d}d'] for s in signals if s[f'fwd_{d}d'] is not None]
        if not vals:
            continue
        n_down = sum(1 for v in vals if v < 0)
        print(f"  {d:>3}일    │ {len(vals):>5} {n_down/len(vals)*100:>7.1f}% {np.mean(vals):>+8.1f}% {np.median(vals):>+8.1f}% │ "
              f"{min(vals):>+8.1f}% {max(vals):>+8.1f}%")

    # Key metrics
    fwd30 = [s['fwd_30d'] for s in signals if s['fwd_30d'] is not None]
    fwd90 = [s['fwd_90d'] for s in signals if s['fwd_90d'] is not None]
    mdds = [s['mdd_90'] for s in signals if s['mdd_90'] is not None]

    if fwd30:
        drop5 = sum(1 for v in fwd30 if v < -5) / len(fwd30) * 100
        drop10 = sum(1 for v in fwd30 if v < -10) / len(fwd30) * 100
        rise10 = sum(1 for v in fwd30 if v > 10) / len(fwd30) * 100
        print(f"\n  30일 후: 5%+하락={drop5:.1f}% | 10%+하락={drop10:.1f}% | 10%+상승={rise10:.1f}%")

    if mdds:
        print(f"  MDD(90일): 평균={np.mean(mdds):+.1f}% 중앙값={np.median(mdds):+.1f}% 최악={min(mdds):+.1f}%")


def main():
    sep = '=' * 130
    dash = '-' * 130

    print(sep)
    print("  매도 신호 대안 테스트: 다이버전스 기반")
    print("  A) 기존: S_Force<0 & S_Div<0 (both negative)")
    print("  B) 신규: S_Force>0 & S_Div<0 (가격↑ 거래량↓ = 베어리시 다이버전스)")
    print("  C) 참고: S_Force<0 & S_Div>0 (가격↓ 거래량↑)")
    print(sep)

    # Load data
    print("\n  데이터 로딩...")
    data = {}
    for tk in TICKERS:
        df = download_max(tk)
        if df is None or len(df) < 300:
            continue
        df = smooth_earnings_volume(df, ticker=tk)
        data[tk] = df
        print(f"    {tk}: {len(df)} bars")
    print(f"  {len(data)} tickers loaded.\n")

    # Test configs
    configs = [
        ('A-기존', 'original', 0.0, 3),
        ('B-다이버전스', 'force_pos_div_neg', 0.0, 3),
        ('B2-강한다이버전스', 'force_pos_div_neg', 0.1, 3),
        ('B3-매우강한', 'force_pos_div_neg', 0.2, 3),
        ('B4-장기다이버전스', 'force_pos_div_neg', 0.0, 5),
        ('B5-장기+강한', 'force_pos_div_neg', 0.1, 5),
        ('C-역발산', 'force_neg_div_pos', 0.0, 3),
    ]

    # ═══════════════════════════════════════════════════════
    # [1] 전체 비교 요약
    # ═══════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  [1] 모드별 전체 비교 요약")
    print(sep)

    all_results = {}

    for label, mode, th, streak in configs:
        all_sigs = []
        for tk, df in data.items():
            subind = calc_v4_subindicators(df, w=20, divgate_days=DIVGATE)
            close = df['Close'].values
            sigs = detect_divergence_sells(subind, close, mode=mode,
                                           strength_th=th, min_streak=streak)
            sigs = calc_forward_returns(sigs, close, len(close))
            for s in sigs:
                s['ticker'] = tk
                s['date'] = df.index[s['idx']].strftime('%Y-%m-%d')
            all_sigs.extend(sigs)

        all_results[label] = all_sigs
        print_summary(label, all_sigs, f"mode={mode}, th={th}, streak>={streak}")

    # ═══════════════════════════════════════════════════════
    # [2] 핵심 비교 테이블
    # ═══════════════════════════════════════════════════════
    print(f"\n\n{sep}")
    print("  [2] 핵심 비교 테이블")
    print(sep)

    print(f"\n  {'Config':<20} │ {'N':>6} │ {'하락30d':>8} {'하락90d':>8} │ {'Avg30d':>8} {'Avg90d':>8} {'Med90d':>8} │ {'AvgMDD90':>9}")
    print(f"  {dash}")

    for label in all_results:
        sigs = all_results[label]
        fwd30 = [s['fwd_30d'] for s in sigs if s['fwd_30d'] is not None]
        fwd90 = [s['fwd_90d'] for s in sigs if s['fwd_90d'] is not None]
        mdds = [s['mdd_90'] for s in sigs if s['mdd_90'] is not None]

        d30 = sum(1 for v in fwd30 if v < 0) / len(fwd30) * 100 if fwd30 else 0
        d90 = sum(1 for v in fwd90 if v < 0) / len(fwd90) * 100 if fwd90 else 0
        a30 = np.mean(fwd30) if fwd30 else 0
        a90 = np.mean(fwd90) if fwd90 else 0
        m90 = np.median(fwd90) if fwd90 else 0
        amdd = np.mean(mdds) if mdds else 0

        # Highlight best
        marker = ""
        if d90 > 45: marker = " ★"

        print(f"  {label:<20} │ {len(sigs):>6} │ {d30:>7.1f}% {d90:>7.1f}% │ {a30:>+7.1f}% {a90:>+7.1f}% {m90:>+7.1f}% │ {amdd:>+8.1f}%{marker}")

    # ═══════════════════════════════════════════════════════
    # [3] B-다이버전스 티커별 상세
    # ═══════════════════════════════════════════════════════
    print(f"\n\n{sep}")
    print("  [3] B-다이버전스 (S_Force>0 & S_Div<0, streak>=3) 티커별")
    print(sep)

    b_sigs = all_results['B-다이버전스']

    print(f"\n  {'Ticker':<8} {'Sect':<10} │ {'N':>5} │ {'하락30d':>8} {'하락90d':>8} │ {'Avg30d':>8} {'Avg90d':>8} │ {'AvgMDD':>8}")
    print(f"  {dash}")

    for tk in TICKERS:
        if tk not in data:
            continue
        tk_sigs = [s for s in b_sigs if s['ticker'] == tk]
        if not tk_sigs:
            continue

        fwd30 = [s['fwd_30d'] for s in tk_sigs if s['fwd_30d'] is not None]
        fwd90 = [s['fwd_90d'] for s in tk_sigs if s['fwd_90d'] is not None]
        mdds = [s['mdd_90'] for s in tk_sigs if s['mdd_90'] is not None]

        d30 = sum(1 for v in fwd30 if v < 0) / len(fwd30) * 100 if fwd30 else 0
        d90 = sum(1 for v in fwd90 if v < 0) / len(fwd90) * 100 if fwd90 else 0
        a30 = np.mean(fwd30) if fwd30 else 0
        a90 = np.mean(fwd90) if fwd90 else 0
        amdd = np.mean(mdds) if mdds else 0

        print(f"  {tk:<8} {TICKERS[tk]:<10} │ {len(tk_sigs):>5} │ {d30:>7.1f}% {d90:>7.1f}% │ {a30:>+7.1f}% {a90:>+7.1f}% │ {amdd:>+7.1f}%")

    # ═══════════════════════════════════════════════════════
    # [4] B-다이버전스 개별 신호 (최근 2024~)
    # ═══════════════════════════════════════════════════════
    print(f"\n\n{sep}")
    print("  [4] B-다이버전스 최근 신호 (2024~ 주요 종목)")
    print(sep)

    recent = [s for s in b_sigs if s['date'] >= '2024-01-01']
    recent.sort(key=lambda x: x['date'])

    print(f"\n  {'#':>4} {'Ticker':>8} {'Date':>12} {'Price':>10} │ {'Force':>7} {'Div':>7} {'Str':>6} {'Dur':>4} │ "
          f"{'7d':>7} {'30d':>7} {'90d':>7} {'180d':>7} │ {'MDD90':>7} │ {'판정'}")
    print(f"  {'-'*135}")

    for i, s in enumerate(recent, 1):
        fwd_parts = []
        for d in [7, 30, 90, 180]:
            v = s[f'fwd_{d}d']
            fwd_parts.append(f"{v:>+6.1f}%" if v is not None else f"{'N/A':>7}")

        mdd = f"{s['mdd_90']:>+6.1f}%" if s['mdd_90'] is not None else f"{'N/A':>7}"

        fwd90 = s['fwd_90d']
        if fwd90 is not None:
            if fwd90 < -15: result = "★★ GREAT SELL"
            elif fwd90 < -5: result = "★ good sell"
            elif fwd90 < 0: result = "ok"
            elif fwd90 < 10: result = "miss"
            elif fwd90 < 30: result = "MISS"
            else: result = "★★ BIG MISS"
        else:
            result = ""

        print(f"  {i:>4} {s['ticker']:>8} {s['date']:>12} ${s['price']:>8.2f} │ {s['s_force']:>+6.3f} {s['s_div']:>+6.3f} "
              f"{s['strength']:>5.3f} {s['duration']:>4} │ {' '.join(fwd_parts)} │ {mdd} │ {result}")

    # ═══════════════════════════════════════════════════════
    # [5] Strength 구간별 성과 (B-다이버전스)
    # ═══════════════════════════════════════════════════════
    print(f"\n\n{sep}")
    print("  [5] B-다이버전스 Strength(=sqrt(|Force|*|Div|)) 구간별 성과")
    print(sep)

    str_bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.0)]
    print(f"\n  {'Strength':<12} │ {'N':>5} {'하락30d':>8} {'하락90d':>8} │ {'Avg30d':>8} {'Avg90d':>8} {'Med90d':>8} │ {'AvgMDD':>8}")
    print(f"  {'-'*95}")

    for lo, hi in str_bins:
        group = [s for s in b_sigs if lo <= s['strength'] < hi]
        if not group:
            continue

        fwd30 = [s['fwd_30d'] for s in group if s['fwd_30d'] is not None]
        fwd90 = [s['fwd_90d'] for s in group if s['fwd_90d'] is not None]
        mdds = [s['mdd_90'] for s in group if s['mdd_90'] is not None]

        d30 = sum(1 for v in fwd30 if v < 0) / len(fwd30) * 100 if fwd30 else 0
        d90 = sum(1 for v in fwd90 if v < 0) / len(fwd90) * 100 if fwd90 else 0
        a30 = np.mean(fwd30) if fwd30 else 0
        a90 = np.mean(fwd90) if fwd90 else 0
        m90 = np.median(fwd90) if fwd90 else 0
        amdd = np.mean(mdds) if mdds else 0

        print(f"  {lo:.1f}-{hi:.1f}      │ {len(group):>5} {d30:>7.1f}% {d90:>7.1f}% │ {a30:>+7.1f}% {a90:>+7.1f}% {m90:>+7.1f}% │ {amdd:>+7.1f}%")

    # ═══════════════════════════════════════════════════════
    # [6] Duration 구간별 성과 (B-다이버전스)
    # ═══════════════════════════════════════════════════════
    print(f"\n\n{sep}")
    print("  [6] B-다이버전스 Duration(연속일수) 구간별 성과")
    print(sep)

    dur_bins = [(3, 5), (6, 10), (11, 20), (21, 50), (51, 999)]
    dur_labels = ['3-5일', '6-10일', '11-20일', '21-50일', '51일+']

    print(f"\n  {'Duration':<10} │ {'N':>5} {'하락30d':>8} {'하락90d':>8} │ {'Avg30d':>8} {'Avg90d':>8} {'Med90d':>8} │ {'AvgStr':>8}")
    print(f"  {'-'*90}")

    for (lo, hi), label in zip(dur_bins, dur_labels):
        group = [s for s in b_sigs if lo <= s['duration'] <= hi]
        if not group:
            continue

        fwd30 = [s['fwd_30d'] for s in group if s['fwd_30d'] is not None]
        fwd90 = [s['fwd_90d'] for s in group if s['fwd_90d'] is not None]

        d30 = sum(1 for v in fwd30 if v < 0) / len(fwd30) * 100 if fwd30 else 0
        d90 = sum(1 for v in fwd90 if v < 0) / len(fwd90) * 100 if fwd90 else 0
        a30 = np.mean(fwd30) if fwd30 else 0
        a90 = np.mean(fwd90) if fwd90 else 0
        m90 = np.median(fwd90) if fwd90 else 0
        avg_str = np.mean([s['strength'] for s in group])

        print(f"  {label:<10} │ {len(group):>5} {d30:>7.1f}% {d90:>7.1f}% │ {a30:>+7.1f}% {a90:>+7.1f}% {m90:>+7.1f}% │ {avg_str:>7.3f}")

    # ═══════════════════════════════════════════════════════
    # [7] GRAND SUMMARY
    # ═══════════════════════════════════════════════════════
    print(f"\n\n{sep}")
    print("  GRAND SUMMARY")
    print(sep)

    # Compare the three main approaches
    for label, desc in [('A-기존', 'S_Force<0 & S_Div<0'),
                        ('B-다이버전스', 'S_Force>0 & S_Div<0'),
                        ('C-역발산', 'S_Force<0 & S_Div>0')]:
        sigs = all_results.get(label, [])
        fwd30 = [s['fwd_30d'] for s in sigs if s['fwd_30d'] is not None]
        fwd90 = [s['fwd_90d'] for s in sigs if s['fwd_90d'] is not None]
        mdds = [s['mdd_90'] for s in sigs if s['mdd_90'] is not None]

        d30 = sum(1 for v in fwd30 if v < 0) / len(fwd30) * 100 if fwd30 else 0
        d90 = sum(1 for v in fwd90 if v < 0) / len(fwd90) * 100 if fwd90 else 0
        a90 = np.mean(fwd90) if fwd90 else 0
        m90 = np.median(fwd90) if fwd90 else 0
        amdd = np.mean(mdds) if mdds else 0

        verdict = ""
        if d90 > 55: verdict = "★ 유효"
        elif d90 > 45: verdict = "약간 유효"
        elif d90 > 35: verdict = "미약"
        else: verdict = "무효"

        print(f"\n  [{label}] {desc}")
        print(f"    신호 수: {len(sigs)}건")
        print(f"    하락 30d: {d30:.1f}% | 하락 90d: {d90:.1f}%")
        print(f"    Avg 90d: {a90:+.1f}% | Med 90d: {m90:+.1f}%")
        print(f"    Avg MDD: {amdd:+.1f}%")
        print(f"    판정: {verdict}")

    # Best config
    print(f"\n  {'─'*70}")
    best_label = None
    best_d90 = 0
    for label, sigs in all_results.items():
        fwd90 = [s['fwd_90d'] for s in sigs if s['fwd_90d'] is not None]
        if fwd90:
            d90 = sum(1 for v in fwd90 if v < 0) / len(fwd90) * 100
            if d90 > best_d90:
                best_d90 = d90
                best_label = label

    if best_label:
        print(f"  최적 Config: [{best_label}] — 90일 하락률 {best_d90:.1f}%")

    print(f"\n  Done.\n")


if __name__ == '__main__':
    main()
