"""V4_wP Adaptive Regime Backtest
====================================
레짐별로 시그널 감지 파라미터를 동적으로 조절하는 백테스트.

비교 전략:
  Baseline       : 모든 레짐에 동일 파라미터 (현행)
  Adaptive-Mild  : BEAR에서 threshold/dd_gate 소폭 완화
  Adaptive-Agg   : BEAR에서 대폭 완화 + BULL에서 강화
  Adaptive+Weight: 파라미터 완화 + 역발상 가중치 결합
"""
import sys
import os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)

from real_market_backtest import (
    download_data, smooth_earnings_volume,
    calc_v4_score, calc_v4_subindicators,
    build_price_filter,
)
from v4wp_realtime.config.settings import SECTOR_ETF_MAP, load_watchlist


# ── Regime Classification ────────────────────────────────────────────

def classify_regime(qqq_ret20, vix_change_20d):
    if qqq_ret20 is None or vix_change_20d is None:
        return 'UNKNOWN'
    if qqq_ret20 < -0.05 or vix_change_20d > 0.30:
        return 'BEAR_STRONG'
    if qqq_ret20 < 0:
        return 'BEAR_WEAK'
    if qqq_ret20 > 0.05 and vix_change_20d < 0.10:
        return 'BULL_STRONG'
    return 'BULL_WEAK'


# ── Adaptive Signal Detection ────────────────────────────────────────

def detect_signals_adaptive(df, scores, subind, base_params, price_filter_fn,
                            qqq_ret20_series, vix_change_20d_series,
                            regime_adjustments):
    """레짐별 파라미터를 동적으로 조절하는 시그널 감지.

    regime_adjustments: dict of regime -> {threshold_mult, dd_mult, confirm_days}
      threshold_mult: base threshold에 곱할 배수 (0.8 = 20% 완화)
      dd_mult: dd_threshold에 곱할 배수 (0.67 = 33% 완화)
      confirm_days: 해당 레짐의 confirm_days (None이면 기본값)
    """
    base_threshold = base_params.get('signal_threshold', 0.05) * 0.5
    base_dd = base_params.get('buy_dd_threshold', 0.03)
    base_confirm = base_params.get('confirm_days', 1)
    cooldown = base_params.get('cooldown', 5)
    dd_lookback = base_params.get('buy_dd_lookback', 20)

    n = len(df)
    signals = []
    last_signal_idx = -cooldown - 1

    in_zone = False
    zone_start = 0
    zone_peak_idx = 0
    zone_peak_val = 0

    for i in range(60, n):
        # 현재 시점의 레짐 판단
        peak_ts = df.index[i]
        mask_q = qqq_ret20_series.index <= peak_ts
        mask_v = vix_change_20d_series.index <= peak_ts
        qqq_val = float(qqq_ret20_series.loc[mask_q].iloc[-1]) if mask_q.any() else None
        vix_val = None
        if mask_v.any():
            v = vix_change_20d_series.loc[mask_v].iloc[-1]
            if pd.notna(v):
                vix_val = float(v)

        regime = classify_regime(qqq_val, vix_val)
        adj = regime_adjustments.get(regime, {})

        # 레짐별 파라미터 조절
        threshold = base_threshold * adj.get('threshold_mult', 1.0)
        dd_threshold = base_dd * adj.get('dd_mult', 1.0)
        confirm_days = adj.get('confirm_days', base_confirm)

        val = scores.iloc[i]

        if val > threshold:
            if not in_zone:
                in_zone = True
                zone_start = i
                zone_peak_idx = i
                zone_peak_val = val
            else:
                if val > zone_peak_val:
                    zone_peak_idx = i
                    zone_peak_val = val
        else:
            if in_zone:
                duration = zone_peak_idx - zone_start + 1

                if duration >= confirm_days:
                    peak_idx = zone_peak_idx
                    if peak_idx - last_signal_idx > cooldown:
                        lb = max(0, peak_idx - dd_lookback)
                        high_nd = df['Close'].iloc[lb:peak_idx+1].max()
                        close = df['Close'].iloc[peak_idx]
                        dd_pct = (high_nd - close) / high_nd if high_nd > 0 else 0

                        if dd_pct >= dd_threshold:
                            pf_ok = price_filter_fn(peak_idx) if price_filter_fn else True
                            if pf_ok:
                                s_force = float(subind['s_force'].iloc[peak_idx])
                                s_div = float(subind['s_div'].iloc[peak_idx])

                                signals.append({
                                    'peak_idx': peak_idx,
                                    'peak_date': df.index[peak_idx].strftime('%Y-%m-%d'),
                                    'peak_val': float(zone_peak_val),
                                    'start_val': float(scores.iloc[zone_start]),
                                    'close_price': float(close),
                                    's_force': s_force,
                                    's_div': s_div,
                                    'dd_pct': round(dd_pct * 100, 2),
                                    'duration': duration,
                                    'regime': regime,
                                    'qqq_ret20': qqq_val,
                                    'vix_change_20d': vix_val,
                                    'threshold_used': round(threshold, 5),
                                    'dd_threshold_used': round(dd_threshold, 4),
                                    'confirm_used': confirm_days,
                                })
                                last_signal_idx = peak_idx

                in_zone = False

    return signals


def calc_forward_returns(df, signals):
    n = len(df)
    for sig in signals:
        idx = sig['peak_idx']
        entry = sig['close_price']
        if idx + 5 < n:
            sig['return_5d'] = round((df['Close'].iloc[idx + 5] - entry) / entry * 100, 2)
        else:
            sig['return_5d'] = None
        if idx + 20 < n:
            sig['return_20d'] = round((df['Close'].iloc[idx + 20] - entry) / entry * 100, 2)
        else:
            sig['return_20d'] = None
        if idx + 90 < n:
            sig['return_90d'] = round((df['Close'].iloc[idx + 90] - entry) / entry * 100, 2)
            min_price = df['Close'].iloc[idx+1:idx+91].min()
            sig['max_dd_90d'] = round((min_price - entry) / entry * 100, 2)
        else:
            sig['return_90d'] = None
            sig['max_dd_90d'] = None
    return signals


# ── 전략 정의 ────────────────────────────────────────────────────────

STRATEGIES = {
    'Baseline (현행)': {
        'adjustments': {
            'BEAR_STRONG':  {'threshold_mult': 1.0, 'dd_mult': 1.0, 'confirm_days': 1},
            'BEAR_WEAK':    {'threshold_mult': 1.0, 'dd_mult': 1.0, 'confirm_days': 1},
            'BULL_WEAK':    {'threshold_mult': 1.0, 'dd_mult': 1.0, 'confirm_days': 1},
            'BULL_STRONG':  {'threshold_mult': 1.0, 'dd_mult': 1.0, 'confirm_days': 1},
            'UNKNOWN':      {'threshold_mult': 1.0, 'dd_mult': 1.0, 'confirm_days': 1},
        },
        'weights': {'BEAR_STRONG': 1.0, 'BEAR_WEAK': 1.0, 'BULL_WEAK': 1.0, 'BULL_STRONG': 1.0, 'UNKNOWN': 1.0},
    },
    'Adaptive-Mild': {
        'adjustments': {
            'BEAR_STRONG':  {'threshold_mult': 0.85, 'dd_mult': 0.80, 'confirm_days': 1},  # 15% 완화
            'BEAR_WEAK':    {'threshold_mult': 0.95, 'dd_mult': 0.90, 'confirm_days': 1},  # 5% 완화
            'BULL_WEAK':    {'threshold_mult': 1.0,  'dd_mult': 1.0,  'confirm_days': 1},  # 현행
            'BULL_STRONG':  {'threshold_mult': 1.10, 'dd_mult': 1.15, 'confirm_days': 2},  # 10% 강화
            'UNKNOWN':      {'threshold_mult': 1.0,  'dd_mult': 1.0,  'confirm_days': 1},
        },
        'weights': {'BEAR_STRONG': 1.0, 'BEAR_WEAK': 1.0, 'BULL_WEAK': 1.0, 'BULL_STRONG': 1.0, 'UNKNOWN': 1.0},
    },
    'Adaptive-Agg': {
        'adjustments': {
            'BEAR_STRONG':  {'threshold_mult': 0.70, 'dd_mult': 0.67, 'confirm_days': 0},  # 30% 완화, 즉시
            'BEAR_WEAK':    {'threshold_mult': 0.85, 'dd_mult': 0.80, 'confirm_days': 1},  # 15% 완화
            'BULL_WEAK':    {'threshold_mult': 1.05, 'dd_mult': 1.05, 'confirm_days': 1},  # 5% 강화
            'BULL_STRONG':  {'threshold_mult': 1.20, 'dd_mult': 1.30, 'confirm_days': 2},  # 20% 강화
            'UNKNOWN':      {'threshold_mult': 1.0,  'dd_mult': 1.0,  'confirm_days': 1},
        },
        'weights': {'BEAR_STRONG': 1.0, 'BEAR_WEAK': 1.0, 'BULL_WEAK': 1.0, 'BULL_STRONG': 1.0, 'UNKNOWN': 1.0},
    },
    'Adaptive+Weight': {
        'adjustments': {
            'BEAR_STRONG':  {'threshold_mult': 0.85, 'dd_mult': 0.80, 'confirm_days': 1},
            'BEAR_WEAK':    {'threshold_mult': 0.95, 'dd_mult': 0.90, 'confirm_days': 1},
            'BULL_WEAK':    {'threshold_mult': 1.0,  'dd_mult': 1.0,  'confirm_days': 1},
            'BULL_STRONG':  {'threshold_mult': 1.10, 'dd_mult': 1.15, 'confirm_days': 2},
            'UNKNOWN':      {'threshold_mult': 1.0,  'dd_mult': 1.0,  'confirm_days': 1},
        },
        'weights': {'BEAR_STRONG': 1.0, 'BEAR_WEAK': 0.8, 'BULL_WEAK': 0.6, 'BULL_STRONG': 0.4, 'UNKNOWN': 0.6},
    },
    'Adaptive-Agg+Weight': {
        'adjustments': {
            'BEAR_STRONG':  {'threshold_mult': 0.70, 'dd_mult': 0.67, 'confirm_days': 0},
            'BEAR_WEAK':    {'threshold_mult': 0.85, 'dd_mult': 0.80, 'confirm_days': 1},
            'BULL_WEAK':    {'threshold_mult': 1.05, 'dd_mult': 1.05, 'confirm_days': 1},
            'BULL_STRONG':  {'threshold_mult': 1.20, 'dd_mult': 1.30, 'confirm_days': 2},
            'UNKNOWN':      {'threshold_mult': 1.0,  'dd_mult': 1.0,  'confirm_days': 1},
        },
        'weights': {'BEAR_STRONG': 1.0, 'BEAR_WEAK': 0.8, 'BULL_WEAK': 0.5, 'BULL_STRONG': 0.0, 'UNKNOWN': 0.5},
    },
}


def main():
    wl = load_watchlist()
    tickers = list(wl['tickers'].keys())
    params = wl.get('params', {})
    START, END = '2020-01-01', '2026-12-31'

    print('=' * 75)
    print('  V4_wP Adaptive Regime Backtest')
    print('  (레짐별 파라미터 동적 조절 + 역발상 가중치)')
    print('=' * 75)

    # QQQ + VIX 사전 다운로드
    print('\n  Downloading QQQ + VIX...')
    qqq_df = download_data('QQQ', start=START, end=END)
    qqq_ret20 = qqq_df['Close'] / qqq_df['Close'].shift(20) - 1.0

    import yfinance as yf
    vix_df = yf.download('^VIX', start=START, end=END, progress=False)
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = vix_df.columns.get_level_values(0)
    vix_close = vix_df['Close']
    vix_change_20d = vix_close / vix_close.shift(20) - 1.0
    print(f'    QQQ: {len(qqq_df)} days, VIX: {len(vix_df)} days')

    # 종목 데이터 사전 준비
    print('\n  Loading ticker data...')
    ticker_cache = {}
    for ticker in tickers:
        try:
            df = download_data(ticker, start=START, end=END)
            if df is None or len(df) < 200:
                continue
            df = smooth_earnings_volume(df, ticker)
            scores = calc_v4_score(df)
            subind = calc_v4_subindicators(df)
            pf = build_price_filter(df,
                                    er_q=params.get('er_percentile', 80),
                                    atr_q=params.get('atr_percentile', 40))
            ticker_cache[ticker] = (df, scores, subind, pf)
            print(f'    {ticker}: {len(df)} days')
        except Exception as e:
            print(f'    {ticker}: ERROR - {e}')

    print(f'\n  Loaded {len(ticker_cache)} tickers')

    # ═══════════════════════════════════════════════════════════════════
    # 전략별 백테스트
    # ═══════════════════════════════════════════════════════════════════
    results = {}

    for strat_name, strat_config in STRATEGIES.items():
        adjustments = strat_config['adjustments']
        weights = strat_config['weights']

        all_signals = []
        for ticker, (df, scores, subind, pf) in ticker_cache.items():
            signals = detect_signals_adaptive(
                df, scores, subind, params, pf,
                qqq_ret20, vix_change_20d,
                adjustments
            )
            signals = calc_forward_returns(df, signals)
            for s in signals:
                s['ticker'] = ticker
            all_signals.extend(signals)

        completed = [s for s in all_signals if s.get('return_90d') is not None]
        results[strat_name] = {
            'all_signals': all_signals,
            'completed': completed,
            'weights': weights,
        }

    # ═══════════════════════════════════════════════════════════════════
    # 1. 전략 요약 비교
    # ═══════════════════════════════════════════════════════════════════
    print('\n' + '=' * 75)
    print('  STRATEGY COMPARISON')
    print('=' * 75)

    header = (f'  {"Strategy":<22s} {"Sigs":>5s} {"90d":>5s} {"WR%":>6s} '
              f'{"AvgR90":>8s} {"MedR90":>8s} {"WtdR90":>8s} {"AvgDD":>8s} {"CapU":>6s}')
    print(f'\n{header}')
    print('  ' + '-' * 73)

    for strat_name, data in results.items():
        completed = data['completed']
        weights = data['weights']
        if not completed:
            print(f'  {strat_name:<22s} {"0":>5s}')
            continue

        # 가중치 적용
        active = [s for s in completed if weights.get(s['regime'], 0) > 0]
        if not active:
            continue

        r90 = [s['return_90d'] for s in active]
        dd = [s['max_dd_90d'] for s in active if s.get('max_dd_90d') is not None]
        wins = sum(1 for r in r90 if r > 0)

        total_weight = sum(weights.get(s['regime'], 0) for s in active)
        weighted_r90 = sum(weights.get(s['regime'], 0) * s['return_90d'] for s in active)
        avg_weighted = weighted_r90 / total_weight if total_weight > 0 else 0
        cap_util = total_weight / len(active) * 100 if active else 0

        print(f'  {strat_name:<22s} {len(data["all_signals"]):>5d} {len(completed):>5d} '
              f'{wins/len(active)*100:>5.1f}% {np.mean(r90):>+7.1f}% '
              f'{np.median(r90):>+7.1f}% {avg_weighted:>+7.1f}% '
              f'{np.mean(dd):>+7.1f}% {cap_util:>5.1f}%')

    # ═══════════════════════════════════════════════════════════════════
    # 2. 레짐별 시그널 수 변화 (Baseline vs Adaptive)
    # ═══════════════════════════════════════════════════════════════════
    print('\n' + '=' * 75)
    print('  SIGNAL COUNT BY REGIME (Baseline vs Adaptive)')
    print('=' * 75)

    regime_order = ['BEAR_STRONG', 'BEAR_WEAK', 'BULL_WEAK', 'BULL_STRONG', 'UNKNOWN']
    header = f'  {"Regime":<14s}'
    for strat_name in STRATEGIES:
        short = strat_name.split('(')[0].strip()[:12]
        header += f' {short:>12s}'
    print(f'\n{header}')
    print('  ' + '-' * (14 + 13 * len(STRATEGIES)))

    for regime in regime_order:
        line = f'  {regime:<14s}'
        for strat_name, data in results.items():
            count = sum(1 for s in data['completed'] if s['regime'] == regime)
            line += f' {count:>12d}'
        print(line)

    # Total
    line = f'  {"TOTAL":<14s}'
    for strat_name, data in results.items():
        line += f' {len(data["completed"]):>12d}'
    print(line)

    # ═══════════════════════════════════════════════════════════════════
    # 3. Adaptive 전략의 레짐별 상세 성과
    # ═══════════════════════════════════════════════════════════════════
    print('\n' + '=' * 75)
    print('  REGIME-LEVEL DETAIL (per strategy)')
    print('=' * 75)

    for strat_name in ['Baseline (현행)', 'Adaptive-Mild', 'Adaptive-Agg']:
        data = results[strat_name]
        completed = data['completed']
        print(f'\n  --- {strat_name} ---')

        for regime in regime_order:
            sigs = [s for s in completed if s['regime'] == regime]
            if not sigs:
                continue
            r90 = [s['return_90d'] for s in sigs]
            r5 = [s['return_5d'] for s in sigs if s['return_5d'] is not None]
            dd = [s['max_dd_90d'] for s in sigs if s.get('max_dd_90d') is not None]
            wins = sum(1 for r in r90 if r > 0)
            print(f'    {regime:14s}: {len(sigs):3d} sigs, WR={wins/len(sigs)*100:5.1f}%, '
                  f'avg90={np.mean(r90):+6.1f}%, med90={np.median(r90):+6.1f}%, '
                  f'avgDD={np.mean(dd):+6.1f}%')

    # ═══════════════════════════════════════════════════════════════════
    # 4. 새로 추가된 시그널 분석 (Adaptive에만 있는 시그널)
    # ═══════════════════════════════════════════════════════════════════
    print('\n' + '=' * 75)
    print('  NEW SIGNALS ANALYSIS (Adaptive-only signals)')
    print('=' * 75)

    baseline_keys = set(
        (s['ticker'], s['peak_date'])
        for s in results['Baseline (현행)']['completed']
    )

    for strat_name in ['Adaptive-Mild', 'Adaptive-Agg']:
        data = results[strat_name]
        new_sigs = [s for s in data['completed']
                    if (s['ticker'], s['peak_date']) not in baseline_keys]

        if not new_sigs:
            print(f'\n  {strat_name}: No new signals (동일)')
            continue

        r90 = [s['return_90d'] for s in new_sigs]
        wins = sum(1 for r in r90 if r > 0)

        print(f'\n  {strat_name}: {len(new_sigs)} new signals')
        print(f'    Win Rate: {wins}/{len(new_sigs)} = {wins/len(new_sigs)*100:.1f}%')
        print(f'    Avg 90d: {np.mean(r90):+.1f}%  Med 90d: {np.median(r90):+.1f}%')

        # 레짐별 분포
        for regime in regime_order:
            rsigs = [s for s in new_sigs if s['regime'] == regime]
            if rsigs:
                rr = [s['return_90d'] for s in rsigs]
                w = sum(1 for r in rr if r > 0)
                print(f'      {regime}: {len(rsigs)} sigs, WR={w/len(rsigs)*100:.0f}%, avg={np.mean(rr):+.1f}%')

        # 상위/하위 시그널
        new_sigs_sorted = sorted(new_sigs, key=lambda s: s['return_90d'], reverse=True)
        print(f'\n    Top 5 new signals:')
        for s in new_sigs_sorted[:5]:
            print(f'      {s["ticker"]:6s} {s["peak_date"]}  {s["regime"]:14s}  '
                  f'90d={s["return_90d"]:+6.1f}%  th={s["threshold_used"]:.4f}  dd={s["dd_threshold_used"]:.3f}')
        print(f'    Bottom 5 new signals:')
        for s in new_sigs_sorted[-5:]:
            print(f'      {s["ticker"]:6s} {s["peak_date"]}  {s["regime"]:14s}  '
                  f'90d={s["return_90d"]:+6.1f}%  th={s["threshold_used"]:.4f}  dd={s["dd_threshold_used"]:.3f}')

    # ═══════════════════════════════════════════════════════════════════
    # 5. 제거된 시그널 분석 (BULL_STRONG 강화로 사라진 시그널)
    # ═══════════════════════════════════════════════════════════════════
    print('\n' + '=' * 75)
    print('  REMOVED SIGNALS (BULL_STRONG tightening)')
    print('=' * 75)

    for strat_name in ['Adaptive-Mild', 'Adaptive-Agg']:
        strat_keys = set(
            (s['ticker'], s['peak_date'])
            for s in results[strat_name]['completed']
        )
        removed = [s for s in results['Baseline (현행)']['completed']
                   if (s['ticker'], s['peak_date']) not in strat_keys]

        if not removed:
            print(f'\n  {strat_name}: No removed signals')
            continue

        r90 = [s['return_90d'] for s in removed]
        wins = sum(1 for r in r90 if r > 0)

        print(f'\n  {strat_name}: {len(removed)} removed signals')
        print(f'    Win Rate: {wins}/{len(removed)} = {wins/len(removed)*100:.1f}%')
        print(f'    Avg 90d: {np.mean(r90):+.1f}%  Med 90d: {np.median(r90):+.1f}%')
        print(f'    (이 시그널들이 제거되어 전체 성과가 개선되면 = 좋은 필터링)')

    # ═══════════════════════════════════════════════════════════════════
    # 6. 파라미터 사용량 분석
    # ═══════════════════════════════════════════════════════════════════
    print('\n' + '=' * 75)
    print('  PARAMETER USAGE ANALYSIS')
    print('=' * 75)

    for strat_name in ['Adaptive-Mild', 'Adaptive-Agg']:
        data = results[strat_name]
        print(f'\n  {strat_name}:')
        for regime in regime_order:
            sigs = [s for s in data['completed'] if s['regime'] == regime]
            if sigs:
                avg_th = np.mean([s['threshold_used'] for s in sigs])
                avg_dd = np.mean([s['dd_threshold_used'] for s in sigs])
                avg_cf = np.mean([s['confirm_used'] for s in sigs])
                print(f'    {regime:14s}: th={avg_th:.4f}  dd_gate={avg_dd:.3f}  '
                      f'confirm={avg_cf:.1f}d  ({len(sigs)} sigs)')

    print('\n' + '=' * 75)
    print('  CONCLUSION')
    print('=' * 75)
    print("""
  비교 포인트:
  1. Adaptive는 BEAR에서 시그널을 더 많이 포착하는가?
  2. 추가된 시그널의 품질(승률, 수익)은 기존과 비슷한가?
  3. BULL 강화로 제거된 시그널은 실제로 나쁜 시그널이었는가?
  4. 파라미터 완화 + 가중치 결합이 단독보다 나은가?
""")


if __name__ == '__main__':
    main()
