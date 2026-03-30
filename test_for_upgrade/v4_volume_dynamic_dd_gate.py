"""
Volume-Dynamic DD Gate 실험
============================
BUY_DD_GATE의 lookback을 고정 20일 대신,
"거래량이 늘어나는 기간"을 기반으로 동적 설정.

가설: 거래량 증가 기간이 길면 큰손 매집 기간이 길다는 의미 →
      그 기간의 고점 대비 충분히 빠져야 진입 가치가 있다.

3가지 거래량 증가 기간 정의:
  A. V_norm > 1 연속 일수 (거래량 > 20일 평균)
  B. S_Div > 0 연속 일수 (거래량모멘텀 > 가격모멘텀)
  C. Volume 20일 이동평균 상승 연속 일수

비교 대상:
  - Baseline: 고정 20일 lookback / 3% threshold (현 프로덕션)
  - Dynamic A/B/C: 동적 lookback / 동일 3% threshold
  - Dynamic + 조정 threshold: 기간 비례 threshold
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
    calc_pv_divergence,
)


# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════
TICKERS = {
    'TSLA': 'Tech', 'NVDA': 'Tech', 'AVGO': 'Tech',
    'AMZN': 'Tech', 'GOOGL': 'Tech', 'PLTR': 'Tech',
    'COIN': 'Fintech', 'HOOD': 'Fintech', 'SOFI': 'Fintech',
    'HIMS': 'Growth', 'RKLB': 'Growth', 'JOBY': 'Growth',
    'TEM': 'Growth', 'CRCL': 'Growth', 'PGY': 'Growth',
    'IONQ': 'Quantum', 'PL': 'Space', 'ASTS': 'Space',
    'QQQ': 'Benchmark', 'VOO': 'Benchmark',
}

# GEO-OP production params
SIGNAL_TH = 0.05
COOLDOWN = 5
ER_Q = 80
ATR_Q = 40
LOOKBACK = 252
CONFIRM_DAYS = 1
DIVGATE = 3

# Baseline DD gate
BASELINE_DD_LB = 20
BASELINE_DD_TH = 0.03

# Dynamic DD gate params
MIN_LOOKBACK = 5     # 최소 lookback (너무 짧으면 무의미)
MAX_LOOKBACK = 60    # 최대 lookback 캡
DEFAULT_LOOKBACK = 20  # 거래량 증가 없을 때 기본값

FORWARD_HORIZONS = [30, 60, 90]


# ═══════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════
def download_ticker(ticker, years=5):
    t = yf.Ticker(ticker)
    df = t.history(period=f'{years}y', auto_adjust=True)
    if df is None or len(df) == 0:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.dropna(subset=['Close', 'Volume'])
    df = df[df['Close'] > 0]
    return df


# ═══════════════════════════════════════════════════════════
# 거래량 증가 기간 계산 (3가지 방법)
# ═══════════════════════════════════════════════════════════
def calc_vol_surge_days_vnorm(df):
    """방법 A: V_norm > 1 (거래량 > 20일 평균) 연속 일수"""
    v_norm = df['Volume'] / df['Volume'].rolling(20).mean()
    v_norm = v_norm.fillna(0)
    n = len(df)
    streak = np.zeros(n)
    for i in range(n):
        if v_norm.iloc[i] > 1.0:
            streak[i] = (streak[i-1] + 1) if i > 0 else 1
        else:
            streak[i] = 0
    return streak


def calc_vol_surge_days_sdiv(df, w=20, divgate_days=3):
    """방법 B: S_Div > 0 (거래량모멘텀 > 가격모멘텀) 연속 일수"""
    pv_div = calc_pv_divergence(df, w)
    raw_div = np.array([np.clip(pv_div.iloc[i] / 3, -1, 1) for i in range(len(df))])
    n = len(df)
    streak = np.zeros(n)
    for i in range(n):
        if raw_div[i] > 0:
            streak[i] = (streak[i-1] + 1) if i > 0 else 1
        else:
            streak[i] = 0
    return streak


def calc_vol_surge_days_vma(df):
    """방법 C: Volume 20일 이동평균이 상승 중인 연속 일수"""
    vma = df['Volume'].rolling(20).mean()
    vma_diff = vma.diff().fillna(0)
    n = len(df)
    streak = np.zeros(n)
    for i in range(n):
        if vma_diff.iloc[i] > 0:
            streak[i] = (streak[i-1] + 1) if i > 0 else 1
        else:
            streak[i] = 0
    return streak


def get_dynamic_lookback(surge_days_at_idx, min_lb=MIN_LOOKBACK, max_lb=MAX_LOOKBACK,
                         default_lb=DEFAULT_LOOKBACK):
    """거래량 증가 일수 → DD Gate lookback 변환"""
    if surge_days_at_idx < min_lb:
        return default_lb  # 증가 기간이 너무 짧으면 기본값
    return int(min(surge_days_at_idx, max_lb))


# ═══════════════════════════════════════════════════════════
# DD Gate 변형 적용
# ═══════════════════════════════════════════════════════════
def apply_dd_gate_baseline(events, df, dd_lb=BASELINE_DD_LB, dd_th=BASELINE_DD_TH):
    """Baseline: 고정 lookback / 고정 threshold"""
    close = df['Close'].values
    rolling_high = pd.Series(close).rolling(dd_lb, min_periods=1).max().values

    passed = []
    blocked = []
    for ev in events:
        pidx = ev['peak_idx']
        if pidx >= len(close):
            continue
        rh = rolling_high[pidx]
        dd = (rh - close[pidx]) / rh if rh > 0 else 0
        ev_copy = dict(ev)
        ev_copy['dd'] = dd
        ev_copy['dd_lookback'] = dd_lb
        if dd >= dd_th:
            passed.append(ev_copy)
        else:
            blocked.append(ev_copy)
    return passed, blocked


def apply_dd_gate_dynamic(events, df, surge_days, dd_th=BASELINE_DD_TH,
                           proportional_th=False):
    """Dynamic: 거래량 증가 기간 기반 lookback"""
    close = df['Close'].values
    n = len(close)

    passed = []
    blocked = []
    for ev in events:
        pidx = ev['peak_idx']
        if pidx >= n:
            continue

        # 동적 lookback: 거래량 증가 일수 기반
        lb = get_dynamic_lookback(surge_days[pidx])

        # 해당 lookback 기간 rolling high 계산
        start_idx = max(0, pidx - lb + 1)
        rh = max(close[start_idx:pidx+1])
        dd = (rh - close[pidx]) / rh if rh > 0 else 0

        # 비례 threshold: 기간이 길면 더 큰 하락 요구
        if proportional_th:
            # 기본 3% + 기간 비례 (20일 기준, 최대 +2%)
            adjusted_th = dd_th + (lb - DEFAULT_LOOKBACK) * 0.001
            adjusted_th = max(dd_th * 0.5, min(adjusted_th, dd_th * 2.0))
            effective_th = adjusted_th
        else:
            effective_th = dd_th

        ev_copy = dict(ev)
        ev_copy['dd'] = dd
        ev_copy['dd_lookback'] = lb
        ev_copy['dd_threshold'] = effective_th
        ev_copy['surge_days'] = surge_days[pidx]

        if dd >= effective_th:
            passed.append(ev_copy)
        else:
            blocked.append(ev_copy)
    return passed, blocked


# ═══════════════════════════════════════════════════════════
# Forward Return 계산
# ═══════════════════════════════════════════════════════════
def calc_fwd_returns(events, df, horizons=FORWARD_HORIZONS):
    """각 신호의 미래 수익률 계산"""
    close = df['Close'].values
    n = len(close)
    results = []

    for ev in events:
        pidx = ev['peak_idx']
        if pidx >= n:
            continue
        price = close[pidx]

        row = {
            'peak_idx': pidx,
            'peak_date': df.index[pidx].strftime('%Y-%m-%d'),
            'price': price,
            'score': ev.get('peak_val', 0),
            'dd': ev.get('dd', 0),
            'dd_lookback': ev.get('dd_lookback', 0),
            'surge_days': ev.get('surge_days', 0),
        }

        for h in horizons:
            end_idx = min(pidx + h, n - 1)
            if end_idx <= pidx:
                row[f'fwd_{h}d'] = np.nan
                row[f'hit_{h}d'] = np.nan
            else:
                ret = (close[end_idx] - price) / price * 100
                row[f'fwd_{h}d'] = ret
                row[f'hit_{h}d'] = 1 if ret > 0 else 0

        results.append(row)

    return pd.DataFrame(results) if results else pd.DataFrame()


# ═══════════════════════════════════════════════════════════
# Main Backtest
# ═══════════════════════════════════════════════════════════
def run_single_ticker(ticker, sector):
    """단일 종목 전체 파이프라인"""
    df = download_ticker(ticker, years=5)
    if df is None or len(df) < 252:
        return None

    # 전처리
    df = smooth_earnings_volume(df, ticker)

    # 스코어 계산
    score = calc_v4_score(df, w=20, divgate_days=DIVGATE)

    # 이벤트 감지 + 가격필터
    events = detect_signal_events(score, th=SIGNAL_TH, cooldown=COOLDOWN)
    pf = build_price_filter(df, er_q=ER_Q, atr_q=ATR_Q, lookback=LOOKBACK)
    buy_events = [e for e in events if e['type'] == 'bottom' and pf(e['peak_idx'])]

    if len(buy_events) == 0:
        return None

    # 거래량 증가 일수 계산 (3가지)
    surge_vnorm = calc_vol_surge_days_vnorm(df)
    surge_sdiv = calc_vol_surge_days_sdiv(df)
    surge_vma = calc_vol_surge_days_vma(df)

    results = {}

    # ── Baseline (고정 20d/3%) ──
    passed_base, blocked_base = apply_dd_gate_baseline(buy_events, df)
    fwd_base = calc_fwd_returns(passed_base, df)
    if not fwd_base.empty:
        fwd_base['method'] = 'Baseline (20d/3%)'
    results['baseline'] = {
        'passed': len(passed_base), 'blocked': len(blocked_base),
        'fwd': fwd_base
    }

    # ── Dynamic A: V_norm 연속 ──
    passed_a, blocked_a = apply_dd_gate_dynamic(buy_events, df, surge_vnorm)
    fwd_a = calc_fwd_returns(passed_a, df)
    if not fwd_a.empty:
        fwd_a['method'] = 'Dynamic-A (V_norm)'
    results['dynamic_a'] = {
        'passed': len(passed_a), 'blocked': len(blocked_a),
        'fwd': fwd_a
    }

    # ── Dynamic B: S_Div 연속 ──
    passed_b, blocked_b = apply_dd_gate_dynamic(buy_events, df, surge_sdiv)
    fwd_b = calc_fwd_returns(passed_b, df)
    if not fwd_b.empty:
        fwd_b['method'] = 'Dynamic-B (S_Div)'
    results['dynamic_b'] = {
        'passed': len(passed_b), 'blocked': len(blocked_b),
        'fwd': fwd_b
    }

    # ── Dynamic C: VMA 상승 ──
    passed_c, blocked_c = apply_dd_gate_dynamic(buy_events, df, surge_vma)
    fwd_c = calc_fwd_returns(passed_c, df)
    if not fwd_c.empty:
        fwd_c['method'] = 'Dynamic-C (VMA)'
    results['dynamic_c'] = {
        'passed': len(passed_c), 'blocked': len(blocked_c),
        'fwd': fwd_c
    }

    # ── Dynamic B + 비례 Threshold ──
    passed_bp, blocked_bp = apply_dd_gate_dynamic(buy_events, df, surge_sdiv,
                                                    proportional_th=True)
    fwd_bp = calc_fwd_returns(passed_bp, df)
    if not fwd_bp.empty:
        fwd_bp['method'] = 'Dynamic-B+PropTH'
    results['dynamic_b_prop'] = {
        'passed': len(passed_bp), 'blocked': len(blocked_bp),
        'fwd': fwd_bp
    }

    return {
        'ticker': ticker, 'sector': sector,
        'total_buy_events': len(buy_events),
        'results': results
    }


# ═══════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════
def print_summary(all_results):
    methods = ['baseline', 'dynamic_a', 'dynamic_b', 'dynamic_c', 'dynamic_b_prop']
    method_labels = {
        'baseline': 'Baseline (20d/3%)',
        'dynamic_a': 'Dynamic-A (V_norm>1 연속)',
        'dynamic_b': 'Dynamic-B (S_Div>0 연속)',
        'dynamic_c': 'Dynamic-C (VMA 상승 연속)',
        'dynamic_b_prop': 'Dynamic-B + 비례 TH',
    }

    print("\n" + "=" * 90)
    print("Volume-Dynamic DD Gate 실험 결과")
    print("=" * 90)

    # ── 종목별 상세 ──
    print("\n■ 종목별 신호 수 (통과 / 차단)")
    print("-" * 90)
    header = f"{'Ticker':<8}"
    for m in methods:
        label = method_labels[m].split('(')[0].strip()
        header += f"{'':>2}{label:>14}"
    print(header)
    print("-" * 90)

    for r in all_results:
        line = f"{r['ticker']:<8}"
        for m in methods:
            p = r['results'][m]['passed']
            b = r['results'][m]['blocked']
            line += f"  {p:>5}/{b:<5}"
        print(line)

    # ── 전체 집계 ──
    print("\n\n■ 전체 집계 — Forward Return & Hit Rate")
    print("=" * 90)

    for m in methods:
        all_fwd = []
        for r in all_results:
            fwd_df = r['results'][m]['fwd']
            if not fwd_df.empty:
                all_fwd.append(fwd_df)

        if not all_fwd:
            print(f"\n  {method_labels[m]}: 신호 없음")
            continue

        combined = pd.concat(all_fwd, ignore_index=True)
        total = len(combined)

        print(f"\n  ┌─ {method_labels[m]}")
        print(f"  │  총 신호: {total}")

        for h in FORWARD_HORIZONS:
            col_fwd = f'fwd_{h}d'
            col_hit = f'hit_{h}d'
            if col_fwd in combined.columns:
                valid = combined[col_fwd].dropna()
                if len(valid) > 0:
                    avg_ret = valid.mean()
                    med_ret = valid.median()
                    hit = combined[col_hit].dropna().mean() * 100
                    print(f"  │  {h:>3}일: 평균 {avg_ret:+.2f}%  중앙값 {med_ret:+.2f}%  적중률 {hit:.1f}% ({int(combined[col_hit].dropna().sum())}/{len(valid)})")

        # Lookback 분포
        if 'dd_lookback' in combined.columns:
            lb_vals = combined['dd_lookback']
            print(f"  │  DD Lookback: 평균 {lb_vals.mean():.1f}일  범위 [{lb_vals.min():.0f}~{lb_vals.max():.0f}]")

        if 'surge_days' in combined.columns:
            sg_vals = combined['surge_days']
            if sg_vals.sum() > 0:
                print(f"  │  거래량 증가기간: 평균 {sg_vals.mean():.1f}일  범위 [{sg_vals.min():.0f}~{sg_vals.max():.0f}]")

        print(f"  └{'─' * 60}")

    # ── 방법 간 차이 분석 ──
    print("\n\n■ Baseline 대비 변화 (Delta)")
    print("=" * 90)

    # Baseline stats
    base_fwd_all = []
    for r in all_results:
        fwd_df = r['results']['baseline']['fwd']
        if not fwd_df.empty:
            base_fwd_all.append(fwd_df)

    if not base_fwd_all:
        print("  Baseline 신호 없음, 비교 불가")
        return

    base_combined = pd.concat(base_fwd_all, ignore_index=True)
    base_total = len(base_combined)

    for m in methods:
        if m == 'baseline':
            continue

        all_fwd = []
        for r in all_results:
            fwd_df = r['results'][m]['fwd']
            if not fwd_df.empty:
                all_fwd.append(fwd_df)

        if not all_fwd:
            continue

        combined = pd.concat(all_fwd, ignore_index=True)
        total = len(combined)

        print(f"\n  {method_labels[m]} vs Baseline:")
        print(f"  신호 수: {base_total} → {total} ({total - base_total:+d})")

        for h in FORWARD_HORIZONS:
            col_fwd = f'fwd_{h}d'
            col_hit = f'hit_{h}d'
            if col_fwd in combined.columns and col_fwd in base_combined.columns:
                b_ret = base_combined[col_fwd].dropna().mean()
                d_ret = combined[col_fwd].dropna().mean()
                b_hit = base_combined[col_hit].dropna().mean() * 100
                d_hit = combined[col_hit].dropna().mean() * 100
                print(f"    {h:>3}일 수익: {b_ret:+.2f}% → {d_ret:+.2f}% (Δ{d_ret-b_ret:+.2f}%p)  "
                      f"적중률: {b_hit:.1f}% → {d_hit:.1f}% (Δ{d_hit-b_hit:+.1f}%p)")

    # ── Blocked 신호 분석: 차단된 신호가 실제로 좋았는지? ──
    print("\n\n■ 차단 신호 분석 — 차단된 신호가 실제로 나빴는가?")
    print("=" * 90)
    print("  (Baseline에서 통과했으나 Dynamic에서 차단 / 그 반대)")

    for m in ['dynamic_b']:  # S_Div 기반만 상세 분석
        print(f"\n  {method_labels[m]}:")

        only_base = 0     # baseline 통과, dynamic 차단
        only_dynamic = 0  # baseline 차단, dynamic 통과
        both = 0          # 둘 다 통과

        only_base_rets = []
        only_dynamic_rets = []

        for r in all_results:
            base_passed_set = set(e['peak_idx'] for e in
                                  [ev for ev in r['results']['baseline']['fwd'].to_dict('records')]
                                  if 'peak_idx' in e) if not r['results']['baseline']['fwd'].empty else set()

            dyn_passed_set = set(e['peak_idx'] for e in
                                  [ev for ev in r['results'][m]['fwd'].to_dict('records')]
                                  if 'peak_idx' in e) if not r['results'][m]['fwd'].empty else set()

            both += len(base_passed_set & dyn_passed_set)

            ob = base_passed_set - dyn_passed_set
            od = dyn_passed_set - base_passed_set
            only_base += len(ob)
            only_dynamic += len(od)

            # 90일 수익 비교
            if not r['results']['baseline']['fwd'].empty:
                for _, row in r['results']['baseline']['fwd'].iterrows():
                    if row['peak_idx'] in ob and not pd.isna(row.get('fwd_90d', np.nan)):
                        only_base_rets.append(row['fwd_90d'])

            if not r['results'][m]['fwd'].empty:
                for _, row in r['results'][m]['fwd'].iterrows():
                    if row['peak_idx'] in od and not pd.isna(row.get('fwd_90d', np.nan)):
                        only_dynamic_rets.append(row['fwd_90d'])

        print(f"    양쪽 다 통과: {both}")
        print(f"    Baseline만 통과 (Dynamic이 추가 차단): {only_base}")
        if only_base_rets:
            avg_ob = np.mean(only_base_rets)
            print(f"      → 차단된 신호의 90일 평균 수익: {avg_ob:+.2f}% "
                  f"({'좋은 차단 ✓' if avg_ob < 0 else '잘못된 차단 ✗'})")
        print(f"    Dynamic만 통과 (Baseline이 차단했던 것): {only_dynamic}")
        if only_dynamic_rets:
            avg_od = np.mean(only_dynamic_rets)
            print(f"      → 추가 통과 신호의 90일 평균 수익: {avg_od:+.2f}% "
                  f"({'좋은 추가 ✓' if avg_od > 0 else '나쁜 추가 ✗'})")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Volume-Dynamic DD Gate 실험")
    print("=" * 60)
    print(f"Tickers: {len(TICKERS)}")
    print(f"Baseline: {BASELINE_DD_LB}d lookback / {BASELINE_DD_TH*100:.0f}% threshold")
    print(f"Dynamic lookback range: [{MIN_LOOKBACK}, {MAX_LOOKBACK}]")
    print()

    all_results = []
    for ticker, sector in TICKERS.items():
        print(f"  Processing {ticker}...", end=" ")
        try:
            result = run_single_ticker(ticker, sector)
            if result:
                total = result['total_buy_events']
                bp = result['results']['baseline']['passed']
                print(f"OK ({total} events, baseline {bp} passed)")
                all_results.append(result)
            else:
                print("SKIP (no data or events)")
        except Exception as e:
            print(f"ERROR: {e}")

    if all_results:
        print_summary(all_results)
    else:
        print("\nNo results to summarize.")
