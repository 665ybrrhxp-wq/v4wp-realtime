"""V4_wP Forward Return Backtest
================================
Post-Mortem 엔진 검증용: 시그널 발동 후 5d/20d/90d 실제 수익률 분석.

현재 프로덕션 알고리즘(AND-GEO + DD Gate + Duration) 기준,
워치리스트 전 종목 대상 히스토리컬 백테스트.
"""
import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트
_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)

from real_market_backtest import (
    download_data, smooth_earnings_volume,
    calc_v4_score, calc_v4_subindicators,
    build_price_filter,
)
from v4wp_realtime.config.settings import SECTOR_ETF_MAP


def detect_signals(df, scores, subind, params, price_filter_fn=None):
    """시그널 감지 (프로덕션과 동일 로직).

    Returns: list of dict (peak_date, peak_idx, peak_val, s_force, s_div,
                            dd_pct, duration, close_price)
    """
    threshold = params.get('signal_threshold', 0.05) * 0.5  # bottom threshold
    cooldown = params.get('cooldown', 5)
    confirm_days = params.get('confirm_days', 1)
    dd_lookback = params.get('buy_dd_lookback', 20)
    dd_threshold = params.get('buy_dd_threshold', 0.03)

    n = len(df)
    signals = []
    last_signal_idx = -cooldown - 1

    # 연속 threshold 초과 구간 감지
    in_zone = False
    zone_start = 0
    zone_peak_idx = 0
    zone_peak_val = 0

    for i in range(60, n):
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

                # Duration 확인
                if duration >= confirm_days:
                    peak_idx = zone_peak_idx

                    # Cooldown 체크
                    if peak_idx - last_signal_idx > cooldown:
                        # DD Gate 체크
                        lb = max(0, peak_idx - dd_lookback)
                        high_nd = df['Close'].iloc[lb:peak_idx+1].max()
                        close = df['Close'].iloc[peak_idx]
                        dd_pct = (high_nd - close) / high_nd if high_nd > 0 else 0

                        if dd_pct >= dd_threshold:
                            # Price Filter 체크
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
                                })
                                last_signal_idx = peak_idx

                in_zone = False

    return signals


def calc_forward_returns(df, signals):
    """각 시그널의 5d/20d/90d forward return + max_dd_90d 계산."""
    n = len(df)

    for sig in signals:
        idx = sig['peak_idx']
        entry = sig['close_price']

        # 5d forward return
        if idx + 5 < n:
            sig['return_5d'] = round((df['Close'].iloc[idx + 5] - entry) / entry * 100, 2)
        else:
            sig['return_5d'] = None

        # 20d forward return
        if idx + 20 < n:
            sig['return_20d'] = round((df['Close'].iloc[idx + 20] - entry) / entry * 100, 2)
        else:
            sig['return_20d'] = None

        # 90d forward return + max DD
        if idx + 90 < n:
            sig['return_90d'] = round((df['Close'].iloc[idx + 90] - entry) / entry * 100, 2)
            min_price = df['Close'].iloc[idx+1:idx+91].min()
            sig['max_dd_90d'] = round((min_price - entry) / entry * 100, 2)
        else:
            sig['return_90d'] = None
            sig['max_dd_90d'] = None

    return signals


def main():
    from v4wp_realtime.config.settings import load_watchlist

    wl = load_watchlist()
    tickers = list(wl['tickers'].keys()) + wl.get('benchmarks', [])
    params = wl.get('params', {})

    print('=' * 70)
    print('  V4_wP Forward Return Backtest (5d / 20d / 90d)')
    print('=' * 70)

    # ── 시장/섹터 ETF 데이터 사전 다운로드 ──
    print('\n  Downloading market/sector ETF data...')
    etf_data = {}
    for etf in ['QQQ'] + list(set(v for v in SECTOR_ETF_MAP.values() if v)):
        try:
            edf = download_data(etf, start='2020-01-01', end='2026-12-31')
            if edf is not None and len(edf) >= 40:
                etf_data[etf] = edf['Close'] / edf['Close'].shift(20) - 1.0
                print(f'    {etf}: {len(edf)} days')
        except Exception:
            pass
    print()

    all_signals = []

    for ticker in tickers:
        try:
            df = download_data(ticker, start='2020-01-01', end='2026-12-31')
            if df is None or len(df) < 200:
                print(f'  {ticker}: insufficient data, skip')
                continue

            df = smooth_earnings_volume(df, ticker)
            scores = calc_v4_score(df)
            subind = calc_v4_subindicators(df)

            er_q = params.get('er_percentile', 80)
            atr_q = params.get('atr_percentile', 40)
            pf = build_price_filter(df, er_q=er_q, atr_q=atr_q)

            signals = detect_signals(df, scores, subind, params, pf)
            signals = calc_forward_returns(df, signals)

            for s in signals:
                s['ticker'] = ticker
                s['sector'] = wl['tickers'].get(ticker, {}).get('sector', 'Benchmark')

                # 시장/섹터 컨텍스트 enrichment
                peak_ts = df.index[s['peak_idx']]
                # QQQ (시장)
                if 'QQQ' in etf_data:
                    qqq_r = etf_data['QQQ']
                    mask = qqq_r.index <= peak_ts
                    if mask.any():
                        val = qqq_r.loc[mask].iloc[-1]
                        if pd.notna(val):
                            s['market_return_20d'] = round(float(val), 4)
                if 'market_return_20d' not in s:
                    s['market_return_20d'] = None

                # 섹터 ETF
                sector = s['sector']
                sector_etf = SECTOR_ETF_MAP.get(sector)
                if sector_etf is None and ticker in ('QQQ', 'VOO'):
                    # 벤치마크는 자기 자신의 20d 수익률 사용
                    ret20 = df['Close'] / df['Close'].shift(20) - 1.0
                    val = ret20.iloc[s['peak_idx']]
                    if pd.notna(val):
                        s['sector_return_20d'] = round(float(val), 4)
                elif sector_etf and sector_etf in etf_data:
                    sec_r = etf_data[sector_etf]
                    mask = sec_r.index <= peak_ts
                    if mask.any():
                        val = sec_r.loc[mask].iloc[-1]
                        if pd.notna(val):
                            s['sector_return_20d'] = round(float(val), 4)
                if 'sector_return_20d' not in s:
                    s['sector_return_20d'] = None

            all_signals.extend(signals)

            # 90d 완료된 시그널만 통계
            completed = [s for s in signals if s['return_90d'] is not None]
            if completed:
                wins = sum(1 for s in completed if s['return_90d'] > 0)
                avg_r90 = np.mean([s['return_90d'] for s in completed])
                print(f'  {ticker:6s}: {len(signals):3d} signals, '
                      f'{len(completed):3d} completed, '
                      f'win {wins}/{len(completed)} ({wins/len(completed)*100:.0f}%), '
                      f'avg 90d {avg_r90:+.1f}%')
            else:
                print(f'  {ticker:6s}: {len(signals):3d} signals, 0 completed (all recent)')

        except Exception as e:
            print(f'  {ticker}: ERROR - {e}')

    # ── 전체 통계 ──
    completed = [s for s in all_signals if s['return_90d'] is not None]

    print()
    print('=' * 70)
    print('  OVERALL STATISTICS')
    print('=' * 70)
    print(f'  Total signals: {len(all_signals)}')
    print(f'  90d completed: {len(completed)}')

    if not completed:
        print('  No completed signals for analysis.')
        return

    wins = sum(1 for s in completed if s['return_90d'] > 0)

    r5 = [s['return_5d'] for s in completed if s['return_5d'] is not None]
    r20 = [s['return_20d'] for s in completed if s['return_20d'] is not None]
    r90 = [s['return_90d'] for s in completed]
    dd90 = [s['max_dd_90d'] for s in completed if s['max_dd_90d'] is not None]

    print(f'\n  Win Rate (90d): {wins}/{len(completed)} = {wins/len(completed)*100:.1f}%')
    print(f'  Avg Return:  5d={np.mean(r5):+.2f}%  20d={np.mean(r20):+.2f}%  90d={np.mean(r90):+.2f}%')
    print(f'  Med Return:  5d={np.median(r5):+.2f}%  20d={np.median(r20):+.2f}%  90d={np.median(r90):+.2f}%')
    print(f'  Avg MaxDD(90d): {np.mean(dd90):.2f}%')
    print(f'  Med MaxDD(90d): {np.median(dd90):.2f}%')

    # ── Return Distribution ──
    print(f'\n  90d Return Distribution:')
    bins = [(-999, -20), (-20, -10), (-10, -5), (-5, 0), (0, 5), (5, 10), (10, 20), (20, 50), (50, 999)]
    for lo, hi in bins:
        count = sum(1 for r in r90 if lo <= r < hi)
        pct = count / len(r90) * 100
        bar = '#' * int(pct / 2)
        label = f'{lo:+d}~{hi:+d}%' if hi < 999 else f'{lo:+d}%+'
        print(f'    {label:>10s}: {count:3d} ({pct:5.1f}%) {bar}')

    # ── DD Gate 구간별 성과 ──
    print(f'\n  DD Gate Performance:')
    dd_bins = [(3, 5), (5, 10), (10, 20), (20, 100)]
    for lo, hi in dd_bins:
        subset = [s for s in completed if lo <= s['dd_pct'] < hi]
        if subset:
            w = sum(1 for s in subset if s['return_90d'] > 0)
            avg = np.mean([s['return_90d'] for s in subset])
            print(f'    DD {lo:2d}-{hi:2d}%: {len(subset):3d} signals, '
                  f'win {w}/{len(subset)} ({w/len(subset)*100:.0f}%), '
                  f'avg 90d {avg:+.1f}%')

    # ── 연도별 성과 ──
    print(f'\n  Annual Performance:')
    for year in sorted(set(s['peak_date'][:4] for s in completed)):
        subset = [s for s in completed if s['peak_date'][:4] == year]
        w = sum(1 for s in subset if s['return_90d'] > 0)
        avg = np.mean([s['return_90d'] for s in subset])
        print(f'    {year}: {len(subset):3d} signals, '
              f'win {w}/{len(subset)} ({w/len(subset)*100:.0f}%), '
              f'avg 90d {avg:+.1f}%')

    # ── Post-Mortem Verdict Accuracy 시뮬레이션 ──
    print(f'\n  Verdict Accuracy (BUY threshold +5%):')
    buy_correct = sum(1 for s in completed if s['return_90d'] >= 5)
    buy_over = sum(1 for s in completed if 0 < s['return_90d'] < 5)
    buy_wrong = sum(1 for s in completed if s['return_90d'] <= 0)
    print(f'    CORRECT (>=+5%):      {buy_correct:3d} ({buy_correct/len(completed)*100:.1f}%)')
    print(f'    OVERCONFIDENT (0~5%): {buy_over:3d} ({buy_over/len(completed)*100:.1f}%)')
    print(f'    OVERCONFIDENT (<0%):  {buy_wrong:3d} ({buy_wrong/len(completed)*100:.1f}%)')

    # ── 5d Early Warning 분석 ──
    print(f'\n  5d Early Warning (5d return as predictor of 90d):')
    r5_pos = [s for s in completed if s['return_5d'] is not None and s['return_5d'] > 0]
    r5_neg = [s for s in completed if s['return_5d'] is not None and s['return_5d'] <= 0]
    if r5_pos:
        w5p = sum(1 for s in r5_pos if s['return_90d'] > 0)
        avg5p = np.mean([s['return_90d'] for s in r5_pos])
        print(f'    5d > 0:  {len(r5_pos):3d} signals → 90d win {w5p/len(r5_pos)*100:.0f}%, avg {avg5p:+.1f}%')
    if r5_neg:
        w5n = sum(1 for s in r5_neg if s['return_90d'] > 0)
        avg5n = np.mean([s['return_90d'] for s in r5_neg])
        print(f'    5d <= 0: {len(r5_neg):3d} signals → 90d win {w5n/len(r5_neg)*100:.0f}%, avg {avg5n:+.1f}%')

    # ── Cosine Similarity 유효성 검증: 6D vs 8D ──
    print(f'\n  Similarity Validation: 6D vs 8D Comparison')
    print(f'  (6D = signal features only, 8D = + market/sector context)')
    from v4wp_realtime.core.similarity import _cosine_similarity

    def _build_6d(sig):
        """6D 벡터 (기존)."""
        s_force = sig['s_force'] or 0
        s_div = sig['s_div'] or 0
        peak_val = sig['peak_val'] or 0
        start_val = sig.get('start_val') or 0
        ratio = (start_val / peak_val) if peak_val > 0 else 0
        dd_pct = sig.get('dd_pct')
        duration = sig.get('duration')
        if dd_pct is not None and duration is not None:
            dd_norm = min(dd_pct / 30.0, 1.0) if dd_pct else 0  # dd_pct is in %
            dur_norm = min(duration / 30.0, 1.0) if duration else 0
            return np.array([s_force, s_div, peak_val, ratio, dd_norm, dur_norm])
        return np.array([s_force, s_div, peak_val, ratio])

    def _build_8d(sig):
        """8D 벡터 (시장/섹터 컨텍스트 포함)."""
        v6 = _build_6d(sig)
        if len(v6) < 6:
            return v6  # dd/duration 없으면 4D 폴백
        mkt = sig.get('market_return_20d')
        sec = sig.get('sector_return_20d')
        if mkt is not None and sec is not None:
            mkt_norm = float(np.tanh(mkt / 0.10))
            sec_norm = float(np.tanh(sec / 0.10))
            return np.append(v6, [mkt_norm, sec_norm])
        return v6

    def _run_sim_validation(completed, build_fn, label):
        vectors = [build_fn(s) for s in completed]
        # 각 시그널의 best match와 유사도 수집
        pairs = []
        for i in range(1, len(completed)):
            best_sim, best_j = -1, -1
            for j in range(i):
                sim = _cosine_similarity(vectors[i], vectors[j])
                if sim > best_sim:
                    best_sim = sim
                    best_j = j
            if best_j >= 0:
                oi = completed[i]['return_90d'] > 0
                oj = completed[best_j]['return_90d'] > 0
                pairs.append((best_sim, oi == oj))

        for thresh in [0.80, 0.90, 0.95, 0.99]:
            subset = [(s, o) for s, o in pairs if s > thresh]
            if subset:
                same = sum(1 for _, o in subset if o)
                print(f'    {label} >{thresh:.0%}: '
                      f'{len(subset):3d} pairs, same outcome {same/len(subset)*100:.0f}%')

    _run_sim_validation(completed, _build_6d, '6D')
    _run_sim_validation(completed, _build_8d, '8D')

    print()
    print('=' * 70)


if __name__ == '__main__':
    main()
