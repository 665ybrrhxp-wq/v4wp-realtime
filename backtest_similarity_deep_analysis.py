"""6D vs 8D 유사도 정밀 비교 분석
===================================
코사인 유사도에 시장/섹터 컨텍스트를 추가했을 때
실제 예측력이 얼마나 개선되는지 다각도 분석.

분석 항목:
  1. 유사도 분포 비교 (6D vs 8D)
  2. 임계값별 정확도 곡선
  3. 시장 레짐별 분석 (강세/약세/횡보)
  4. 섹터별 분석
  5. DD 구간별 분석
  6. 오판(False Positive) 패턴 분석
  7. 수익률 예측력 (유사 시그널 평균 수익 vs 실제 수익)
  8. 종합 결론
"""
import sys
import os
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
from v4wp_realtime.config.settings import SECTOR_ETF_MAP
from v4wp_realtime.core.similarity import _cosine_similarity


# ── 벡터 빌더 ──
def build_6d(sig):
    s_force = sig['s_force'] or 0
    s_div = sig['s_div'] or 0
    peak_val = sig['peak_val'] or 0
    start_val = sig.get('start_val') or 0
    ratio = (start_val / peak_val) if peak_val > 0 else 0
    dd_pct = sig.get('dd_pct')
    duration = sig.get('duration')
    if dd_pct is not None and duration is not None:
        dd_norm = min(dd_pct / 30.0, 1.0) if dd_pct else 0
        dur_norm = min(duration / 30.0, 1.0) if duration else 0
        return np.array([s_force, s_div, peak_val, ratio, dd_norm, dur_norm])
    return np.array([s_force, s_div, peak_val, ratio])


def build_8d(sig):
    v6 = build_6d(sig)
    if len(v6) < 6:
        return v6
    mkt = sig.get('market_return_20d')
    sec = sig.get('sector_return_20d')
    if mkt is not None and sec is not None:
        mkt_norm = float(np.tanh(mkt / 0.10))
        sec_norm = float(np.tanh(sec / 0.10))
        return np.append(v6, [mkt_norm, sec_norm])
    return v6


# ── 시그널 감지 (backtest_forward_returns.py 동일) ──
def detect_signals(df, scores, subind, params, price_filter_fn=None):
    threshold = params.get('signal_threshold', 0.05) * 0.5
    cooldown = params.get('cooldown', 5)
    confirm_days = params.get('confirm_days', 1)
    dd_lookback = params.get('buy_dd_lookback', 20)
    dd_threshold = params.get('buy_dd_threshold', 0.03)

    n = len(df)
    signals = []
    last_signal_idx = -cooldown - 1
    in_zone = False
    zone_start = zone_peak_idx = 0
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
                                signals.append({
                                    'peak_idx': peak_idx,
                                    'peak_date': df.index[peak_idx].strftime('%Y-%m-%d'),
                                    'peak_val': float(zone_peak_val),
                                    'start_val': float(scores.iloc[zone_start]),
                                    'close_price': float(close),
                                    's_force': float(subind['s_force'].iloc[peak_idx]),
                                    's_div': float(subind['s_div'].iloc[peak_idx]),
                                    'dd_pct': round(dd_pct * 100, 2),
                                    'duration': duration,
                                })
                                last_signal_idx = peak_idx
                in_zone = False
    return signals


def calc_forward_returns(df, signals):
    n = len(df)
    for sig in signals:
        idx = sig['peak_idx']
        entry = sig['close_price']
        sig['return_5d'] = round((df['Close'].iloc[idx+5] - entry) / entry * 100, 2) if idx+5 < n else None
        sig['return_20d'] = round((df['Close'].iloc[idx+20] - entry) / entry * 100, 2) if idx+20 < n else None
        if idx+90 < n:
            sig['return_90d'] = round((df['Close'].iloc[idx+90] - entry) / entry * 100, 2)
            sig['max_dd_90d'] = round((df['Close'].iloc[idx+1:idx+91].min() - entry) / entry * 100, 2)
        else:
            sig['return_90d'] = None
            sig['max_dd_90d'] = None
    return signals


def classify_market_regime(mkt_ret):
    """시장 레짐 분류: 강세/횡보/약세."""
    if mkt_ret is None:
        return 'UNKNOWN'
    if mkt_ret > 0.05:
        return 'BULL'
    elif mkt_ret < -0.05:
        return 'BEAR'
    else:
        return 'NEUTRAL'


def main():
    from v4wp_realtime.config.settings import load_watchlist

    wl = load_watchlist()
    tickers = list(wl['tickers'].keys()) + wl.get('benchmarks', [])
    params = wl.get('params', {})

    print('=' * 70)
    print('  6D vs 8D 유사도 정밀 비교 분석')
    print('=' * 70)

    # ── ETF 데이터 다운로드 ──
    print('\n  [준비] 시장/섹터 ETF 데이터 다운로드...')
    etf_data = {}
    for etf in ['QQQ'] + list(set(v for v in SECTOR_ETF_MAP.values() if v)):
        try:
            edf = download_data(etf, start='2020-01-01', end='2026-12-31')
            if edf is not None and len(edf) >= 40:
                etf_data[etf] = edf['Close'] / edf['Close'].shift(20) - 1.0
        except Exception:
            pass

    # ── 시그널 수집 ──
    print('  [준비] 전 종목 시그널 수집 중...\n')
    all_signals = []

    for ticker in tickers:
        try:
            df = download_data(ticker, start='2020-01-01', end='2026-12-31')
            if df is None or len(df) < 200:
                continue
            df = smooth_earnings_volume(df, ticker)
            scores = calc_v4_score(df)
            subind = calc_v4_subindicators(df)
            pf = build_price_filter(df,
                                    er_q=params.get('er_percentile', 80),
                                    atr_q=params.get('atr_percentile', 40))
            signals = detect_signals(df, scores, subind, params, pf)
            signals = calc_forward_returns(df, signals)

            for s in signals:
                s['ticker'] = ticker
                s['sector'] = wl['tickers'].get(ticker, {}).get('sector', 'Benchmark')

                # 시장 컨텍스트
                peak_ts = df.index[s['peak_idx']]
                s['market_return_20d'] = None
                if 'QQQ' in etf_data:
                    qqq_r = etf_data['QQQ']
                    mask = qqq_r.index <= peak_ts
                    if mask.any():
                        val = qqq_r.loc[mask].iloc[-1]
                        if pd.notna(val):
                            s['market_return_20d'] = round(float(val), 4)

                # 섹터 컨텍스트
                s['sector_return_20d'] = None
                sector_etf = SECTOR_ETF_MAP.get(s['sector'])
                if sector_etf is None and ticker in ('QQQ', 'VOO'):
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

                # 시장 레짐
                s['regime'] = classify_market_regime(s['market_return_20d'])

            all_signals.extend(signals)
        except Exception as e:
            print(f'  {ticker}: ERROR - {e}')

    completed = [s for s in all_signals if s['return_90d'] is not None]
    has_context = [s for s in completed if s.get('market_return_20d') is not None
                   and s.get('sector_return_20d') is not None]

    print(f'  총 시그널: {len(all_signals)}개')
    print(f'  90일 완료: {len(completed)}개')
    print(f'  시장/섹터 컨텍스트 보유: {len(has_context)}개')

    # ── 벡터 사전 계산 ──
    vecs_6d = [build_6d(s) for s in completed]
    vecs_8d = [build_8d(s) for s in completed]

    # 각 시그널에 대해 최고 유사 매치 찾기
    matches_6d = []  # (i, best_j, sim, same_outcome)
    matches_8d = []

    for i in range(1, len(completed)):
        # 6D
        best_sim6, best_j6 = -1, -1
        for j in range(i):
            s = _cosine_similarity(vecs_6d[i], vecs_6d[j])
            if s > best_sim6:
                best_sim6, best_j6 = s, j

        # 8D
        best_sim8, best_j8 = -1, -1
        for j in range(i):
            s = _cosine_similarity(vecs_8d[i], vecs_8d[j])
            if s > best_sim8:
                best_sim8, best_j8 = s, j

        oi = completed[i]['return_90d'] > 0
        oj6 = completed[best_j6]['return_90d'] > 0
        oj8 = completed[best_j8]['return_90d'] > 0

        matches_6d.append({
            'i': i, 'j': best_j6, 'sim': best_sim6,
            'same': oi == oj6, 'ret_i': completed[i]['return_90d'],
            'ret_j': completed[best_j6]['return_90d'],
            'regime': completed[i]['regime'],
            'sector': completed[i]['sector'],
            'dd_pct': completed[i]['dd_pct'],
            'ticker_i': completed[i]['ticker'],
            'ticker_j': completed[best_j6]['ticker'],
        })
        matches_8d.append({
            'i': i, 'j': best_j8, 'sim': best_sim8,
            'same': oi == oj8, 'ret_i': completed[i]['return_90d'],
            'ret_j': completed[best_j8]['return_90d'],
            'regime': completed[i]['regime'],
            'sector': completed[i]['sector'],
            'dd_pct': completed[i]['dd_pct'],
            'ticker_i': completed[i]['ticker'],
            'ticker_j': completed[best_j8]['ticker'],
        })

    # ═══════════════════════════════════════════════════════════
    # 1. 유사도 분포 비교
    # ═══════════════════════════════════════════════════════════
    print('\n')
    print('=' * 70)
    print('  1. 유사도 분포 비교')
    print('=' * 70)

    sims_6d = [m['sim'] for m in matches_6d]
    sims_8d = [m['sim'] for m in matches_8d]

    print(f'\n  {"구간":>12s}  {"6D 매치수":>10s}  {"8D 매치수":>10s}  {"차이":>8s}')
    print(f'  {"-"*12}  {"-"*10}  {"-"*10}  {"-"*8}')
    for lo, hi, label in [(0.99, 1.01, '99%+'), (0.95, 0.99, '95-99%'),
                           (0.90, 0.95, '90-95%'), (0.80, 0.90, '80-90%'),
                           (0.0, 0.80, '<80%')]:
        c6 = sum(1 for s in sims_6d if lo <= s < hi)
        c8 = sum(1 for s in sims_8d if lo <= s < hi)
        print(f'  {label:>12s}  {c6:>10d}  {c8:>10d}  {c8-c6:>+8d}')

    print(f'\n  6D 평균 유사도: {np.mean(sims_6d):.4f}  (중앙값: {np.median(sims_6d):.4f})')
    print(f'  8D 평균 유사도: {np.mean(sims_8d):.4f}  (중앙값: {np.median(sims_8d):.4f})')

    # ═══════════════════════════════════════════════════════════
    # 2. 임계값별 정확도 곡선
    # ═══════════════════════════════════════════════════════════
    print('\n')
    print('=' * 70)
    print('  2. 임계값별 예측 정확도 (same-outcome %)')
    print('=' * 70)

    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99]
    print(f'\n  {"임계값":>8s}  {"6D 매치":>8s}  {"6D 정확":>8s}  {"8D 매치":>8s}  {"8D 정확":>8s}  {"차이":>6s}')
    print(f'  {"-"*8}  {"-"*8}  {"-"*8}  {"-"*8}  {"-"*8}  {"-"*6}')

    for t in thresholds:
        sub6 = [m for m in matches_6d if m['sim'] > t]
        sub8 = [m for m in matches_8d if m['sim'] > t]
        acc6 = sum(1 for m in sub6 if m['same']) / len(sub6) * 100 if sub6 else 0
        acc8 = sum(1 for m in sub8 if m['same']) / len(sub8) * 100 if sub8 else 0
        diff = acc8 - acc6
        marker = ' <<' if diff > 3 else ''
        print(f'  {t:>8.0%}  {len(sub6):>8d}  {acc6:>7.1f}%  {len(sub8):>8d}  {acc8:>7.1f}%  {diff:>+5.1f}%{marker}')

    # ═══════════════════════════════════════════════════════════
    # 3. 시장 레짐별 분석
    # ═══════════════════════════════════════════════════════════
    print('\n')
    print('=' * 70)
    print('  3. 시장 레짐별 유사도 예측 정확도')
    print('     BULL: QQQ 20일 수익률 > +5%')
    print('     BEAR: QQQ 20일 수익률 < -5%')
    print('     NEUTRAL: -5% ~ +5%')
    print('=' * 70)

    for regime in ['BULL', 'NEUTRAL', 'BEAR']:
        sub6 = [m for m in matches_6d if m['regime'] == regime and m['sim'] > 0.90]
        sub8 = [m for m in matches_8d if m['regime'] == regime and m['sim'] > 0.90]

        if not sub6:
            print(f'\n  {regime}: 데이터 부족')
            continue

        acc6 = sum(1 for m in sub6 if m['same']) / len(sub6) * 100
        acc8 = sum(1 for m in sub8 if m['same']) / len(sub8) * 100

        avg_ret6 = np.mean([m['ret_i'] for m in sub6])
        avg_ret8 = np.mean([m['ret_i'] for m in sub8])

        print(f'\n  {regime} ({len(sub6)}건):')
        print(f'    6D 정확도: {acc6:.1f}%  (평균 실제수익 {avg_ret6:+.1f}%)')
        print(f'    8D 정확도: {acc8:.1f}%  (평균 실제수익 {avg_ret8:+.1f}%)')
        print(f'    차이: {acc8-acc6:+.1f}%p')

        # 8D에서 레짐 매치 여부 분석
        regime_match = 0
        regime_mismatch = 0
        for m in matches_8d:
            if m['regime'] != regime:
                continue
            j_regime = completed[m['j']]['regime']
            if m['regime'] == j_regime:
                regime_match += 1
            else:
                regime_mismatch += 1
        if regime_match + regime_mismatch > 0:
            total_rm = regime_match + regime_mismatch
            print(f'    8D 매치의 레짐 일치율: {regime_match}/{total_rm} ({regime_match/total_rm*100:.0f}%)')

    # ═══════════════════════════════════════════════════════════
    # 4. 섹터별 분석
    # ═══════════════════════════════════════════════════════════
    print('\n')
    print('=' * 70)
    print('  4. 섹터별 유사도 예측 정확도 (>90% 유사도)')
    print('=' * 70)

    sectors = sorted(set(m['sector'] for m in matches_6d))
    print(f'\n  {"섹터":>12s}  {"건수":>5s}  {"6D 정확":>8s}  {"8D 정확":>8s}  {"차이":>6s}  {"8D 동일섹터매치":>15s}')
    print(f'  {"-"*12}  {"-"*5}  {"-"*8}  {"-"*8}  {"-"*6}  {"-"*15}')

    for sector in sectors:
        sub6 = [m for m in matches_6d if m['sector'] == sector and m['sim'] > 0.90]
        sub8 = [m for m in matches_8d if m['sector'] == sector and m['sim'] > 0.90]

        if len(sub6) < 5:
            continue

        acc6 = sum(1 for m in sub6 if m['same']) / len(sub6) * 100
        acc8 = sum(1 for m in sub8 if m['same']) / len(sub8) * 100

        # 8D에서 같은 섹터로 매칭되는 비율
        same_sector = sum(1 for m in sub8 if completed[m['j']]['sector'] == sector)
        ss_pct = same_sector / len(sub8) * 100 if sub8 else 0

        diff = acc8 - acc6
        print(f'  {sector:>12s}  {len(sub6):>5d}  {acc6:>7.1f}%  {acc8:>7.1f}%  {diff:>+5.1f}%  {ss_pct:>14.0f}%')

    # ═══════════════════════════════════════════════════════════
    # 5. DD 구간별 분석
    # ═══════════════════════════════════════════════════════════
    print('\n')
    print('=' * 70)
    print('  5. DD 구간별 유사도 예측 정확도 (>90% 유사도)')
    print('=' * 70)

    dd_bins = [(3, 5, 'DD 3-5%'), (5, 10, 'DD 5-10%'), (10, 20, 'DD 10-20%'), (20, 100, 'DD 20%+')]
    print(f'\n  {"DD 구간":>12s}  {"건수":>5s}  {"6D 정확":>8s}  {"8D 정확":>8s}  {"차이":>6s}')
    print(f'  {"-"*12}  {"-"*5}  {"-"*8}  {"-"*8}  {"-"*6}')

    for lo, hi, label in dd_bins:
        sub6 = [m for m in matches_6d if lo <= m['dd_pct'] < hi and m['sim'] > 0.90]
        sub8 = [m for m in matches_8d if lo <= m['dd_pct'] < hi and m['sim'] > 0.90]

        if len(sub6) < 5:
            continue

        acc6 = sum(1 for m in sub6 if m['same']) / len(sub6) * 100
        acc8 = sum(1 for m in sub8 if m['same']) / len(sub8) * 100
        diff = acc8 - acc6
        print(f'  {label:>12s}  {len(sub6):>5d}  {acc6:>7.1f}%  {acc8:>7.1f}%  {diff:>+5.1f}%')

    # ═══════════════════════════════════════════════════════════
    # 6. 오판 패턴 분석 (False Positive)
    # ═══════════════════════════════════════════════════════════
    print('\n')
    print('=' * 70)
    print('  6. 오판 패턴 분석 (유사도 >95%인데 반대 결과)')
    print('=' * 70)

    fp_6d = [m for m in matches_6d if m['sim'] > 0.95 and not m['same']]
    fp_8d = [m for m in matches_8d if m['sim'] > 0.95 and not m['same']]

    print(f'\n  6D 오판: {len(fp_6d)}건 / {sum(1 for m in matches_6d if m["sim"] > 0.95)}건')
    print(f'  8D 오판: {len(fp_8d)}건 / {sum(1 for m in matches_8d if m["sim"] > 0.95)}건')

    # 6D 오판이지만 8D에서 정확한 케이스
    if fp_6d:
        fixed_by_8d = 0
        broken_by_8d = 0
        for m6 in fp_6d:
            idx = m6['i'] - 1  # matches 인덱스 (i는 1부터 시작)
            if idx < len(matches_8d):
                m8 = matches_8d[idx]
                if m8['same']:
                    fixed_by_8d += 1

        # 6D 정확인데 8D에서 오판된 케이스
        correct_6d = [m for m in matches_6d if m['sim'] > 0.95 and m['same']]
        for m6 in correct_6d:
            idx = m6['i'] - 1
            if idx < len(matches_8d):
                m8 = matches_8d[idx]
                if not m8['same']:
                    broken_by_8d += 1

        print(f'\n  6D 오판 → 8D 정확 (수정됨): {fixed_by_8d}건')
        print(f'  6D 정확 → 8D 오판 (퇴화됨): {broken_by_8d}건')
        print(f'  순 개선: {fixed_by_8d - broken_by_8d:+d}건')

    # 오판 시그널의 시장 레짐 분포
    if fp_6d:
        print(f'\n  6D 오판의 시장 레짐 분포:')
        for regime in ['BULL', 'NEUTRAL', 'BEAR']:
            cnt = sum(1 for m in fp_6d if m['regime'] == regime)
            total = sum(1 for m in matches_6d if m['regime'] == regime and m['sim'] > 0.95)
            if total > 0:
                print(f'    {regime}: {cnt}/{total} ({cnt/total*100:.0f}% 오판율)')

    if fp_8d:
        print(f'\n  8D 오판의 시장 레짐 분포:')
        for regime in ['BULL', 'NEUTRAL', 'BEAR']:
            cnt = sum(1 for m in fp_8d if m['regime'] == regime)
            total = sum(1 for m in matches_8d if m['regime'] == regime and m['sim'] > 0.95)
            if total > 0:
                print(f'    {regime}: {cnt}/{total} ({cnt/total*100:.0f}% 오판율)')

    # ═══════════════════════════════════════════════════════════
    # 7. 수익률 예측력 분석
    # ═══════════════════════════════════════════════════════════
    print('\n')
    print('=' * 70)
    print('  7. 수익률 예측력 (유사 시그널의 수익률 → 실제 수익률 예측)')
    print('=' * 70)

    # Top-3 유사 시그널의 평균 수익률로 현재 시그널 수익률 예측
    def get_top3_avg_return(completed, vecs, i, n_top=3):
        """i번째 시그널에 대해 과거 top-3 유사 시그널의 평균 90d 수익률."""
        sims = []
        for j in range(i):
            if completed[j]['return_90d'] is not None:
                s = _cosine_similarity(vecs[i], vecs[j])
                sims.append((s, completed[j]['return_90d']))
        sims.sort(key=lambda x: x[0], reverse=True)
        top = sims[:n_top]
        if not top:
            return None
        return np.mean([r for _, r in top])

    # MAE (Mean Absolute Error) 계산
    errors_6d = []
    errors_8d = []
    for i in range(5, len(completed)):  # 최소 5개 과거 시그널 확보
        actual = completed[i]['return_90d']
        pred6 = get_top3_avg_return(completed, vecs_6d, i)
        pred8 = get_top3_avg_return(completed, vecs_8d, i)
        if pred6 is not None:
            errors_6d.append(abs(actual - pred6))
        if pred8 is not None:
            errors_8d.append(abs(actual - pred8))

    if errors_6d:
        print(f'\n  Top-3 유사 시그널 평균 수익률로 예측:')
        print(f'    6D MAE: {np.mean(errors_6d):.2f}%p  (중앙값: {np.median(errors_6d):.2f}%p)')
        print(f'    8D MAE: {np.mean(errors_8d):.2f}%p  (중앙값: {np.median(errors_8d):.2f}%p)')
        print(f'    개선: {np.mean(errors_6d) - np.mean(errors_8d):+.2f}%p')

    # 방향 예측 정확도 (유사 시그널 평균이 양수면 실제도 양수?)
    correct_dir_6d = 0
    correct_dir_8d = 0
    total_dir = 0
    for i in range(5, len(completed)):
        actual = completed[i]['return_90d'] > 0
        pred6 = get_top3_avg_return(completed, vecs_6d, i)
        pred8 = get_top3_avg_return(completed, vecs_8d, i)
        if pred6 is not None and pred8 is not None:
            total_dir += 1
            if (pred6 > 0) == actual:
                correct_dir_6d += 1
            if (pred8 > 0) == actual:
                correct_dir_8d += 1

    if total_dir > 0:
        print(f'\n  Top-3 유사 시그널 방향 예측 (양수/음수):')
        print(f'    6D: {correct_dir_6d}/{total_dir} ({correct_dir_6d/total_dir*100:.1f}%)')
        print(f'    8D: {correct_dir_8d}/{total_dir} ({correct_dir_8d/total_dir*100:.1f}%)')

    # ═══════════════════════════════════════════════════════════
    # 8. 연도별 추이
    # ═══════════════════════════════════════════════════════════
    print('\n')
    print('=' * 70)
    print('  8. 연도별 유사도 예측 정확도 추이 (>90% 유사도)')
    print('=' * 70)

    years = sorted(set(completed[m['i']]['peak_date'][:4] for m in matches_6d))
    print(f'\n  {"연도":>6s}  {"건수":>5s}  {"6D 정확":>8s}  {"8D 정확":>8s}  {"차이":>6s}  {"시장 평균":>10s}')
    print(f'  {"-"*6}  {"-"*5}  {"-"*8}  {"-"*8}  {"-"*6}  {"-"*10}')

    for year in years:
        sub6 = [m for m in matches_6d
                if completed[m['i']]['peak_date'][:4] == year and m['sim'] > 0.90]
        sub8 = [m for m in matches_8d
                if completed[m['i']]['peak_date'][:4] == year and m['sim'] > 0.90]

        if len(sub6) < 5:
            continue

        acc6 = sum(1 for m in sub6 if m['same']) / len(sub6) * 100
        acc8 = sum(1 for m in sub8 if m['same']) / len(sub8) * 100
        avg_mkt = np.mean([completed[m['i']].get('market_return_20d', 0) or 0 for m in sub6])
        diff = acc8 - acc6
        print(f'  {year:>6s}  {len(sub6):>5d}  {acc6:>7.1f}%  {acc8:>7.1f}%  {diff:>+5.1f}%  {avg_mkt*100:>+9.1f}%')

    # ═══════════════════════════════════════════════════════════
    # 9. 8D가 매치를 변경한 케이스 분석
    # ═══════════════════════════════════════════════════════════
    print('\n')
    print('=' * 70)
    print('  9. 8D가 매치 대상을 변경한 케이스 분석')
    print('=' * 70)

    changed = 0
    changed_better = 0
    changed_worse = 0
    changed_neutral = 0
    for idx in range(len(matches_6d)):
        m6, m8 = matches_6d[idx], matches_8d[idx]
        if m6['j'] != m8['j']:
            changed += 1
            if m8['same'] and not m6['same']:
                changed_better += 1
            elif not m8['same'] and m6['same']:
                changed_worse += 1
            else:
                changed_neutral += 1

    total = len(matches_6d)
    print(f'\n  전체 매치 중 대상 변경: {changed}/{total} ({changed/total*100:.1f}%)')
    print(f'    개선 (6D 오판 → 8D 정확): {changed_better}건')
    print(f'    퇴화 (6D 정확 → 8D 오판): {changed_worse}건')
    print(f'    중립 (결과 동일): {changed_neutral}건')
    print(f'    순 효과: {changed_better - changed_worse:+d}건')

    # ═══════════════════════════════════════════════════════════
    # 10. 종합 결론
    # ═══════════════════════════════════════════════════════════
    print('\n')
    print('=' * 70)
    print('  10. 종합 결론')
    print('=' * 70)

    # 핵심 지표 계산
    acc6_90 = sum(1 for m in matches_6d if m['sim'] > 0.90 and m['same']) / max(1, sum(1 for m in matches_6d if m['sim'] > 0.90)) * 100
    acc8_90 = sum(1 for m in matches_8d if m['sim'] > 0.90 and m['same']) / max(1, sum(1 for m in matches_8d if m['sim'] > 0.90)) * 100
    acc6_99 = sum(1 for m in matches_6d if m['sim'] > 0.99 and m['same']) / max(1, sum(1 for m in matches_6d if m['sim'] > 0.99)) * 100
    acc8_99 = sum(1 for m in matches_8d if m['sim'] > 0.99 and m['same']) / max(1, sum(1 for m in matches_8d if m['sim'] > 0.99)) * 100

    print(f'\n  [유사도 >90%] 6D: {acc6_90:.1f}% → 8D: {acc8_90:.1f}% ({acc8_90-acc6_90:+.1f}%p)')
    print(f'  [유사도 >99%] 6D: {acc6_99:.1f}% → 8D: {acc8_99:.1f}% ({acc8_99-acc6_99:+.1f}%p)')

    if errors_6d and errors_8d:
        mae_diff = np.mean(errors_6d) - np.mean(errors_8d)
        print(f'  [MAE 개선]    {mae_diff:+.2f}%p (낮을수록 좋음)')

    print(f'  [매치 변경]   {changed}건 중 순 개선 {changed_better - changed_worse:+d}건')

    if acc8_99 > acc6_99 + 3:
        print(f'\n  결론: 8D 벡터는 고유사도(>99%) 구간에서 유의미한 개선을 보임.')
        print(f'        시장/섹터 컨텍스트가 "정밀 매칭"에 기여.')
    elif acc8_90 > acc6_90 + 1:
        print(f'\n  결론: 8D 벡터가 전반적으로 소폭 개선.')
    else:
        print(f'\n  결론: 8D 벡터의 전반적 개선 효과는 제한적.')
        print(f'        코사인 유사도의 구조적 한계 가능성 있음.')

    print()
    print('=' * 70)


if __name__ == '__main__':
    main()
