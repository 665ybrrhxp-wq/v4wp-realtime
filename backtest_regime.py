"""V4_wP Market Regime Backtest
================================
매크로 레짐 필터링이 시그널 수익률을 얼마나 개선하는지 검증.

비교:
  Baseline     : 모든 시그널 (현행 시스템)
  Skip BEAR_S  : BEAR_STRONG 시그널 제외
  Skip ALL_BEAR: BEAR_STRONG + BEAR_WEAK 시그널 제외
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
from backtest_forward_returns import detect_signals, calc_forward_returns
from v4wp_realtime.config.settings import SECTOR_ETF_MAP, load_watchlist


# ── Regime Classification ────────────────────────────────────────────

def classify_regime(qqq_ret20, vix_change_20d):
    """매크로 레짐 4단계 분류.

    Args:
        qqq_ret20: QQQ 20일 수익률 (소수, e.g. -0.022 = -2.2%)
        vix_change_20d: VIX 20일 변화율 (소수, e.g. 0.24 = +24%)

    Returns:
        str: BULL_STRONG | BULL_WEAK | BEAR_WEAK | BEAR_STRONG
    """
    if qqq_ret20 is None or vix_change_20d is None:
        return 'UNKNOWN'

    # BEAR_STRONG: 극단적 하락 또는 극단적 공포
    if qqq_ret20 < -0.05 or vix_change_20d > 0.30:
        return 'BEAR_STRONG'

    # BEAR_WEAK: 하락장이지만 공포 수준은 보통
    if qqq_ret20 < 0:
        return 'BEAR_WEAK'

    # BULL_STRONG: 강한 상승 + 낮은 공포
    if qqq_ret20 > 0.05 and vix_change_20d < 0.10:
        return 'BULL_STRONG'

    # BULL_WEAK: 나머지 (약한 상승 또는 상승+약간의 공포)
    return 'BULL_WEAK'


# ── Main ─────────────────────────────────────────────────────────────

def main():
    wl = load_watchlist()
    tickers = list(wl['tickers'].keys())
    params = wl.get('params', {})

    START = '2020-01-01'
    END = '2026-12-31'

    print('=' * 70)
    print('  V4_wP Market Regime Backtest')
    print('=' * 70)

    # ── 1. QQQ + VIX 다운로드 ──
    print('\n  Downloading QQQ + VIX...')
    qqq_df = download_data('QQQ', start=START, end=END)
    qqq_ret20 = qqq_df['Close'] / qqq_df['Close'].shift(20) - 1.0

    try:
        import yfinance as yf
        vix_df = yf.download('^VIX', start=START, end=END, progress=False)
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = vix_df.columns.get_level_values(0)
        vix_close = vix_df['Close']
        vix_change_20d = vix_close / vix_close.shift(20) - 1.0
        print(f'    VIX: {len(vix_df)} days')
    except Exception as e:
        print(f'    VIX download failed: {e}')
        vix_change_20d = pd.Series(dtype=float)

    # ── 2. 종목별 시그널 탐지 + 레짐 매핑 ──
    print('\n  Scanning tickers...')
    all_signals = []

    for ticker in tickers:
        try:
            df = download_data(ticker, start=START, end=END)
            if df is None or len(df) < 200:
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
                peak_ts = df.index[s['peak_idx']]

                # QQQ 20d return
                mask = qqq_ret20.index <= peak_ts
                qqq_val = float(qqq_ret20.loc[mask].iloc[-1]) if mask.any() else None

                # VIX 20d change
                vix_val = None
                if len(vix_change_20d) > 0:
                    vmask = vix_change_20d.index <= peak_ts
                    if vmask.any():
                        v = vix_change_20d.loc[vmask].iloc[-1]
                        if pd.notna(v):
                            vix_val = float(v)

                s['qqq_ret20'] = qqq_val
                s['vix_change_20d'] = vix_val
                s['regime'] = classify_regime(qqq_val, vix_val)

            all_signals.extend(signals)
            print(f'    {ticker:6s}: {len(signals):3d} signals')

        except Exception as e:
            print(f'    {ticker}: ERROR - {e}')

    # ── 3. 분석 ──
    completed = [s for s in all_signals if s.get('return_90d') is not None]

    print(f'\n  Total signals: {len(all_signals)}')
    print(f'  90d completed: {len(completed)}')

    if not completed:
        print('  No completed signals.')
        return

    # ── 레짐 분포 ──
    regimes = {}
    for s in completed:
        r = s['regime']
        regimes.setdefault(r, []).append(s)

    print('\n' + '=' * 70)
    print('  REGIME DISTRIBUTION (90d completed signals)')
    print('=' * 70)

    for regime in ['BULL_STRONG', 'BULL_WEAK', 'BEAR_WEAK', 'BEAR_STRONG', 'UNKNOWN']:
        sigs = regimes.get(regime, [])
        if not sigs:
            continue

        r90 = [s['return_90d'] for s in sigs]
        r20 = [s['return_20d'] for s in sigs if s['return_20d'] is not None]
        r5 = [s['return_5d'] for s in sigs if s['return_5d'] is not None]
        wins = sum(1 for r in r90 if r > 0)
        dd = [s['max_dd_90d'] for s in sigs if s.get('max_dd_90d') is not None]

        print(f'\n  {regime} ({len(sigs)} signals, {len(sigs)/len(completed)*100:.0f}%)')
        print(f'    Win Rate(90d): {wins}/{len(sigs)} = {wins/len(sigs)*100:.1f}%')
        if r5:
            print(f'    Avg Return:  5d={np.mean(r5):+.2f}%  20d={np.mean(r20):+.2f}%  90d={np.mean(r90):+.2f}%')
            print(f'    Med Return:  5d={np.median(r5):+.2f}%  20d={np.median(r20):+.2f}%  90d={np.median(r90):+.2f}%')
        if dd:
            print(f'    Avg MaxDD(90d): {np.mean(dd):.2f}%')

        # 대표 시그널
        worst = min(sigs, key=lambda s: s['return_90d'])
        best = max(sigs, key=lambda s: s['return_90d'])
        print(f'    Best:  {best["ticker"]} {best["peak_date"]} → 90d {best["return_90d"]:+.1f}%')
        print(f'    Worst: {worst["ticker"]} {worst["peak_date"]} → 90d {worst["return_90d"]:+.1f}%')

    # ── 전략 비교 ──
    print('\n' + '=' * 70)
    print('  STRATEGY COMPARISON')
    print('=' * 70)

    strategies = {
        'Baseline (현행)': completed,
        'Skip BEAR_STRONG': [s for s in completed if s['regime'] != 'BEAR_STRONG'],
        'Skip ALL BEAR': [s for s in completed if s['regime'] not in ('BEAR_STRONG', 'BEAR_WEAK')],
        'BULL only (STRONG)': [s for s in completed if s['regime'] == 'BULL_STRONG'],
    }

    header = f'  {"Strategy":<22s} {"Signals":>7s} {"Win%":>6s} {"5d":>7s} {"20d":>7s} {"90d":>7s} {"MedDD":>7s}'
    print(f'\n{header}')
    print('  ' + '-' * 68)

    for name, sigs in strategies.items():
        if not sigs:
            print(f'  {name:<22s} {"0":>7s}')
            continue

        r5 = [s['return_5d'] for s in sigs if s['return_5d'] is not None]
        r20 = [s['return_20d'] for s in sigs if s['return_20d'] is not None]
        r90 = [s['return_90d'] for s in sigs]
        dd = [s['max_dd_90d'] for s in sigs if s.get('max_dd_90d') is not None]
        wins = sum(1 for r in r90 if r > 0)

        print(f'  {name:<22s} {len(sigs):>7d} {wins/len(sigs)*100:>5.1f}% '
              f'{np.mean(r5):>+6.1f}% {np.mean(r20):>+6.1f}% {np.mean(r90):>+6.1f}% '
              f'{np.median(dd):>+6.1f}%')

    # ── 필터링으로 제거된 시그널 상세 ──
    bear_strong = regimes.get('BEAR_STRONG', [])
    if bear_strong:
        print(f'\n  --- BEAR_STRONG에서 제거될 시그널 ({len(bear_strong)}개) ---')
        for s in sorted(bear_strong, key=lambda x: x['peak_date']):
            qqq = s['qqq_ret20']
            vix = s['vix_change_20d']
            print(f'    {s["ticker"]:6s} {s["peak_date"]}  '
                  f'QQQ={qqq*100:+.1f}% VIX={vix*100 if vix else 0:+.1f}%  '
                  f'→ 5d={s.get("return_5d", "?"):>+6}  '
                  f'20d={s.get("return_20d", "?"):>+6}  '
                  f'90d={s.get("return_90d", "?"):>+6}')

    bear_weak = regimes.get('BEAR_WEAK', [])
    if bear_weak:
        print(f'\n  --- BEAR_WEAK에서 제거될 시그널 ({len(bear_weak)}개) ---')
        for s in sorted(bear_weak, key=lambda x: x['peak_date']):
            qqq = s['qqq_ret20']
            vix = s['vix_change_20d']
            print(f'    {s["ticker"]:6s} {s["peak_date"]}  '
                  f'QQQ={qqq*100:+.1f}% VIX={vix*100 if vix else 0:+.1f}%  '
                  f'→ 5d={s.get("return_5d", "?"):>+6}  '
                  f'20d={s.get("return_20d", "?"):>+6}  '
                  f'90d={s.get("return_90d", "?"):>+6}')


if __name__ == '__main__':
    main()
