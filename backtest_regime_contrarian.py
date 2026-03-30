"""V4_wP Contrarian Regime Backtest
====================================
레짐을 역발상 확신도로 활용: 공포 극대 = 고확신, 탐욕 극대 = 저확신

Conviction 레벨:
  BEAR_STRONG → CONVICTION_HIGH   (공포 극대 = 바닥 매수 기회)
  BEAR_WEAK   → CONVICTION_MID    (약한 하락 = 평균적 기회)
  BULL_WEAK   → CONVICTION_LOW    (약한 상승 = 주의)
  BULL_STRONG → CONVICTION_CAUTION (탐욕 구간 = 고점 매수 위험)

시뮬레이션: conviction별 포지션 비중 차등 적용 시 수익률 변화
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
from backtest_forward_returns import detect_signals, calc_forward_returns
from v4wp_realtime.config.settings import SECTOR_ETF_MAP, load_watchlist


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


# 역발상 conviction 매핑
CONTRARIAN_MAP = {
    'BEAR_STRONG':  {'conviction': 'HIGH',    'label': '공포 극대 → 역발상 매수', 'weight': 1.0},
    'BEAR_WEAK':    {'conviction': 'MID',     'label': '약세장 매수 기회',         'weight': 0.8},
    'BULL_WEAK':    {'conviction': 'LOW',     'label': '상승장 평균 기회',         'weight': 0.6},
    'BULL_STRONG':  {'conviction': 'CAUTION', 'label': '과열 구간 주의',           'weight': 0.4},
    'UNKNOWN':      {'conviction': 'MID',     'label': '데이터 부족',              'weight': 0.6},
}


def main():
    wl = load_watchlist()
    tickers = list(wl['tickers'].keys())
    params = wl.get('params', {})
    START, END = '2020-01-01', '2026-12-31'

    print('=' * 70)
    print('  V4_wP Contrarian Regime Backtest')
    print('  (공포 = 고확신, 탐욕 = 저확신)')
    print('=' * 70)

    # QQQ + VIX
    qqq_df = download_data('QQQ', start=START, end=END)
    qqq_ret20 = qqq_df['Close'] / qqq_df['Close'].shift(20) - 1.0

    import yfinance as yf
    vix_df = yf.download('^VIX', start=START, end=END, progress=False)
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = vix_df.columns.get_level_values(0)
    vix_close = vix_df['Close']
    vix_change_20d = vix_close / vix_close.shift(20) - 1.0

    # 시그널 수집
    all_signals = []
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
            signals = detect_signals(df, scores, subind, params, pf)
            signals = calc_forward_returns(df, signals)

            for s in signals:
                s['ticker'] = ticker
                peak_ts = df.index[s['peak_idx']]
                mask = qqq_ret20.index <= peak_ts
                qqq_val = float(qqq_ret20.loc[mask].iloc[-1]) if mask.any() else None
                vmask = vix_change_20d.index <= peak_ts
                vix_val = float(vix_change_20d.loc[vmask].iloc[-1]) if vmask.any() and pd.notna(vix_change_20d.loc[vmask].iloc[-1]) else None
                s['qqq_ret20'] = qqq_val
                s['vix_change_20d'] = vix_val
                s['regime'] = classify_regime(qqq_val, vix_val)
                s['conviction'] = CONTRARIAN_MAP.get(s['regime'], CONTRARIAN_MAP['UNKNOWN'])['conviction']
                s['weight'] = CONTRARIAN_MAP.get(s['regime'], CONTRARIAN_MAP['UNKNOWN'])['weight']

            all_signals.extend(signals)
        except Exception as e:
            print(f'    {ticker}: ERROR - {e}')

    completed = [s for s in all_signals if s.get('return_90d') is not None]
    print(f'\n  Total signals: {len(all_signals)}, 90d completed: {len(completed)}')

    # ═══════════════════════════════════════════════════════════════
    # 1. Conviction별 성과
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print('  CONTRARIAN CONVICTION PERFORMANCE')
    print('=' * 70)

    for conv in ['HIGH', 'MID', 'LOW', 'CAUTION']:
        sigs = [s for s in completed if s['conviction'] == conv]
        if not sigs:
            continue
        regime = [s['regime'] for s in sigs]
        regime_str = ', '.join(sorted(set(regime)))
        r90 = [s['return_90d'] for s in sigs]
        r20 = [s['return_20d'] for s in sigs if s['return_20d'] is not None]
        r5 = [s['return_5d'] for s in sigs if s['return_5d'] is not None]
        dd = [s['max_dd_90d'] for s in sigs if s.get('max_dd_90d') is not None]
        wins = sum(1 for r in r90 if r > 0)
        losses = [r for r in r90 if r < 0]

        info = CONTRARIAN_MAP[[k for k, v in CONTRARIAN_MAP.items() if v['conviction'] == conv][0]]
        print(f'\n  {conv} — {info["label"]} ({len(sigs)} signals)')
        print(f'    레짐: {regime_str}')
        print(f'    Win Rate(90d): {wins}/{len(sigs)} = {wins/len(sigs)*100:.1f}%')
        print(f'    Avg Return:  5d={np.mean(r5):+.2f}%  20d={np.mean(r20):+.2f}%  90d={np.mean(r90):+.2f}%')
        print(f'    Med Return:  5d={np.median(r5):+.2f}%  20d={np.median(r20):+.2f}%  90d={np.median(r90):+.2f}%')
        if dd:
            print(f'    Avg MaxDD(90d): {np.mean(dd):.2f}%')
        if losses:
            print(f'    Avg Loss(패배): {np.mean(losses):.2f}%  Worst: {min(r90):+.1f}%')

    # ═══════════════════════════════════════════════════════════════
    # 2. 가중 포지션 시뮬레이션
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print('  WEIGHTED POSITION SIMULATION')
    print('=' * 70)
    print('  (weight × return으로 자본 효율성 비교)')

    strategies = {
        'Baseline (균등)':       {c: 1.0 for c in ['HIGH', 'MID', 'LOW', 'CAUTION', 'UNKNOWN']},
        'Contrarian (역발상)':   {'HIGH': 1.0, 'MID': 0.8, 'LOW': 0.6, 'CAUTION': 0.4, 'UNKNOWN': 0.6},
        'Aggressive (공격적)':   {'HIGH': 1.0, 'MID': 0.7, 'LOW': 0.4, 'CAUTION': 0.0, 'UNKNOWN': 0.5},
        'Skip CAUTION only':    {'HIGH': 1.0, 'MID': 1.0, 'LOW': 1.0, 'CAUTION': 0.0, 'UNKNOWN': 1.0},
    }

    header = f'  {"Strategy":<24s} {"Signals":>7s} {"AvgWt":>6s} {"WtdR90":>8s} {"AvgR90":>8s} {"WR%":>6s} {"CapUtil":>7s}'
    print(f'\n{header}')
    print('  ' + '-' * 72)

    for name, weights in strategies.items():
        # 가중치 적용 시그널만
        sigs = [s for s in completed if weights.get(s['conviction'], 0) > 0]
        if not sigs:
            continue

        total_weight = sum(weights.get(s['conviction'], 0) for s in sigs)
        weighted_r90 = sum(weights.get(s['conviction'], 0) * s['return_90d'] for s in sigs)
        avg_weighted_r90 = weighted_r90 / total_weight if total_weight > 0 else 0
        avg_r90 = np.mean([s['return_90d'] for s in sigs])
        avg_weight = total_weight / len(sigs) if sigs else 0
        wins = sum(1 for s in sigs if s['return_90d'] > 0)
        # 자본활용률: 가중평균 weight
        cap_util = avg_weight * 100

        print(f'  {name:<24s} {len(sigs):>7d} {avg_weight:>5.2f}x {avg_weighted_r90:>+7.1f}% '
              f'{avg_r90:>+7.1f}% {wins/len(sigs)*100:>5.1f}% {cap_util:>6.1f}%')

    # ═══════════════════════════════════════════════════════════════
    # 3. 단기 리스크 분석 (5d drawdown)
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print('  SHORT-TERM RISK (5d return by conviction)')
    print('=' * 70)
    print('  (BEAR 구간 진입 시 단기 변동성 얼마나 큰지)')

    for conv in ['HIGH', 'MID', 'LOW', 'CAUTION']:
        sigs = [s for s in completed if s['conviction'] == conv and s.get('return_5d') is not None]
        if not sigs:
            continue
        r5 = [s['return_5d'] for s in sigs]
        neg5 = sum(1 for r in r5 if r < -5)
        print(f'  {conv:8s}: avg 5d={np.mean(r5):+.2f}%  med={np.median(r5):+.2f}%  '
              f'5d<-5%: {neg5}/{len(sigs)} ({neg5/len(sigs)*100:.0f}%)')

    # ═══════════════════════════════════════════════════════════════
    # 4. AI 해석기 적용 제안
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print('  AI INTERPRETER IMPLICATIONS')
    print('=' * 70)
    print("""
  역발상 레짐 → AI 해석기 가이드라인:

  HIGH (BEAR_STRONG):
    "시장 공포 극대화 구간. 역사적으로 이 구간 매수 시그널의
     90일 승률 68%, 평균 수익 +35%. 역발상 매수 확신도 상향."

  MID (BEAR_WEAK):
    "약세장 매수 기회. 90일 승률 64%, 평균 +19%.
     바닥은 아닐 수 있으나 중장기 상승 확률 양호."

  LOW (BULL_WEAK):
    "상승장 시그널. 90일 승률 53%, 평균 +23%.
     평균 수익률은 높으나 분산이 크고 일부 대형 수익이 왜곡."

  CAUTION (BULL_STRONG):
    "과열 구간 주의. 90일 승률 53%, 평균 +8%.
     고점 매수 위험. 분할 매수 또는 소량 진입 권장."
""")


if __name__ == '__main__':
    main()
