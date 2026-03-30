"""8D 벡터 다중공선성(Multicollinearity) 분석
==============================================
8개 변수 간 상관관계, VIF, 조건수, 독립 정보량 확인.
"""
import sys, os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from real_market_backtest import (
    download_data, smooth_earnings_volume,
    calc_v4_score, calc_v4_subindicators, build_price_filter,
)
from v4wp_realtime.config.settings import SECTOR_ETF_MAP, load_watchlist


def main():
    wl = load_watchlist()
    tickers = list(wl['tickers'].keys()) + wl.get('benchmarks', [])
    params = wl.get('params', {})

    # ETF 데이터
    etf_data = {}
    for etf in ['QQQ'] + list(set(v for v in SECTOR_ETF_MAP.values() if v)):
        edf = download_data(etf, start='2020-01-01', end='2026-12-31')
        if edf is not None and len(edf) >= 40:
            etf_data[etf] = edf['Close'] / edf['Close'].shift(20) - 1.0

    # 시그널 수집 (8D 벡터 구성)
    rows = []
    for ticker in tickers:
        try:
            df = download_data(ticker, start='2020-01-01', end='2026-12-31')
            if df is None or len(df) < 200:
                continue
            df = smooth_earnings_volume(df, ticker)
            scores = calc_v4_score(df)
            subind = calc_v4_subindicators(df)

            threshold = params.get('signal_threshold', 0.05) * 0.5
            cooldown = params.get('cooldown', 5)
            dd_lookback = params.get('buy_dd_lookback', 20)
            dd_threshold = params.get('buy_dd_threshold', 0.03)
            n = len(df)
            last_sig = -cooldown - 1
            in_zone = False
            zs = zpi = 0
            zpv = 0

            for i in range(60, n):
                val = scores.iloc[i]
                if val > threshold:
                    if not in_zone:
                        in_zone = True
                        zs = i
                        zpi = i
                        zpv = val
                    elif val > zpv:
                        zpi = i
                        zpv = val
                else:
                    if in_zone:
                        dur = zpi - zs + 1
                        if dur >= 1 and zpi - last_sig > cooldown:
                            lb = max(0, zpi - dd_lookback)
                            high_nd = df['Close'].iloc[lb:zpi + 1].max()
                            close = df['Close'].iloc[zpi]
                            dd = (high_nd - close) / high_nd if high_nd > 0 else 0
                            if dd >= dd_threshold:
                                peak_ts = df.index[zpi]
                                mkt = None
                                if 'QQQ' in etf_data:
                                    m = etf_data['QQQ'].index <= peak_ts
                                    if m.any():
                                        v = etf_data['QQQ'].loc[m].iloc[-1]
                                        if pd.notna(v):
                                            mkt = float(v)

                                sector = wl['tickers'].get(ticker, {}).get('sector', 'Benchmark')
                                sec = None
                                se = SECTOR_ETF_MAP.get(sector)
                                if se is None and ticker in ('QQQ', 'VOO'):
                                    r20 = df['Close'] / df['Close'].shift(20) - 1
                                    v = r20.iloc[zpi]
                                    if pd.notna(v):
                                        sec = float(v)
                                elif se and se in etf_data:
                                    m = etf_data[se].index <= peak_ts
                                    if m.any():
                                        v = etf_data[se].loc[m].iloc[-1]
                                        if pd.notna(v):
                                            sec = float(v)

                                if mkt is not None and sec is not None:
                                    rows.append({
                                        's_force': float(subind['s_force'].iloc[zpi]),
                                        's_div': float(subind['s_div'].iloc[zpi]),
                                        'peak_val': float(zpv),
                                        'ratio': float(scores.iloc[zs] / zpv) if zpv > 0 else 0,
                                        'dd_norm': min(dd * 100 / 30, 1.0),
                                        'dur_norm': min(dur / 30, 1.0),
                                        'mkt_norm': float(np.tanh(mkt / 0.10)),
                                        'sec_norm': float(np.tanh(sec / 0.10)),
                                    })
                                last_sig = zpi
                        in_zone = False
        except Exception:
            pass

    print(f'8D vector: {len(rows)} signals collected')
    print()

    cols = ['s_force', 's_div', 'peak_val', 'ratio', 'dd_norm', 'dur_norm', 'mkt_norm', 'sec_norm']
    labels = ['S_Force', 'S_Div', 'Peak_Val', 'Start/Pk', 'DD_Norm', 'Dur_Norm', 'Mkt_Norm', 'Sec_Norm']
    mat = pd.DataFrame(rows, columns=cols)

    # ════════════════════════════════════════
    # 1. 상관행렬
    # ════════════════════════════════════════
    print('=' * 78)
    print('  1. Correlation Matrix (Pearson)')
    print('=' * 78)
    corr = mat.corr()
    print()
    header = '            ' + ''.join(f'{l:>10s}' for l in labels)
    print(header)
    print('  ' + '-' * (10 + 10 * len(labels)))
    for i, l in enumerate(labels):
        row_str = f'  {l:>8s} |'
        for j in range(len(labels)):
            v = corr.iloc[i, j]
            if i == j:
                row_str += f'     ---  '
            elif abs(v) > 0.7:
                row_str += f'  {v:>+.3f}**'
            elif abs(v) > 0.5:
                row_str += f'  {v:>+.3f}* '
            else:
                row_str += f'  {v:>+.3f}  '
        print(row_str)

    print()
    print('  ** |r| > 0.7 (high),  * |r| > 0.5 (moderate)')

    # 높은 상관 쌍 정리
    print()
    high_pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) > 0.3:
                high_pairs.append((labels[i], labels[j], r))
    high_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    print('  Notable pairs (|r| > 0.3):')
    if high_pairs:
        for a, b, r in high_pairs:
            level = 'HIGH' if abs(r) > 0.7 else 'MED' if abs(r) > 0.5 else 'LOW'
            print(f'    {a:>8s} <-> {b:<8s}  r = {r:+.3f}  [{level}]')
    else:
        print('    None')

    # ════════════════════════════════════════
    # 2. VIF
    # ════════════════════════════════════════
    print()
    print('=' * 78)
    print('  2. VIF (Variance Inflation Factor)')
    print('     > 10: severe multicollinearity')
    print('     > 5 : warning')
    print('     < 5 : OK')
    print('=' * 78)
    print()

    from numpy.linalg import lstsq
    X = mat.values

    for i in range(X.shape[1]):
        y = X[:, i]
        others = np.delete(X, i, axis=1)
        ones = np.column_stack([others, np.ones(len(y))])
        coef, _, _, _ = lstsq(ones, y, rcond=None)
        y_pred = ones @ coef
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        vif = 1 / (1 - r2) if r2 < 1 else float('inf')
        status = 'SEVERE' if vif > 10 else 'WARN' if vif > 5 else 'OK'
        bar = '#' * min(int(vif), 30)
        print(f'  {labels[i]:>10s}  VIF = {vif:>6.2f}  [{status:>6s}]  {bar}')

    # ════════════════════════════════════════
    # 3. 조건수
    # ════════════════════════════════════════
    print()
    print('=' * 78)
    print('  3. Condition Number')
    print('     > 30 : suspected multicollinearity')
    print('     > 100: severe')
    print('=' * 78)
    print()

    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    eigenvalues = np.linalg.eigvalsh(X_std.T @ X_std / len(X_std))
    eigenvalues = np.sort(eigenvalues)[::-1]

    cn = np.sqrt(eigenvalues[0] / eigenvalues[-1]) if eigenvalues[-1] > 0 else float('inf')
    status = 'SEVERE' if cn > 100 else 'SUSPECTED' if cn > 30 else 'OK'
    print(f'  Condition Number: {cn:.2f}  [{status}]')

    print()
    print('  Eigenvalue distribution (PCA):')
    cum = 0
    for k, ev in enumerate(eigenvalues):
        pct = ev / eigenvalues.sum() * 100
        cum += pct
        bar = '#' * int(pct / 2)
        print(f'    PC{k + 1}: {ev:.4f} ({pct:5.1f}%, cum {cum:5.1f}%) {bar}')

    # ════════════════════════════════════════
    # 4. 신규 차원의 독립 정보량
    # ════════════════════════════════════════
    print()
    print('=' * 78)
    print('  4. Independent Information of Market/Sector Dimensions')
    print('     (How much info do mkt/sec add beyond the original 6D?)')
    print('=' * 78)
    print()

    for target_idx, target_label in [(6, 'Mkt_Norm'), (7, 'Sec_Norm')]:
        y = X[:, target_idx]
        others = X[:, :6]
        ones = np.column_stack([others, np.ones(len(y))])
        coef, _, _, _ = lstsq(ones, y, rcond=None)
        y_pred = ones @ coef
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        indep = (1 - r2) * 100
        print(f'  {target_label} predicted by 6D: R2 = {r2:.3f}')
        print(f'    -> Independent info: {indep:.1f}%')
        if indep > 80:
            print(f'    -> Most of {target_label} is NEW information (good)')
        elif indep > 50:
            print(f'    -> Moderate new information')
        else:
            print(f'    -> Mostly redundant with existing 6D')
        print()

    # mkt <-> sec
    r_ms = corr.loc['mkt_norm', 'sec_norm']
    print(f'  Mkt_Norm <-> Sec_Norm correlation: r = {r_ms:.3f}')
    if abs(r_ms) > 0.7:
        print('  -> HIGH correlation: consider keeping only one')
    elif abs(r_ms) > 0.5:
        print('  -> MODERATE correlation: partial overlap, both add some value')
    else:
        print('  -> LOW correlation: both dimensions are independently useful')

    # ════════════════════════════════════════
    # 5. 종합 판정
    # ════════════════════════════════════════
    print()
    print('=' * 78)
    print('  5. Summary')
    print('=' * 78)
    print()

    max_vif = 0
    for i in range(X.shape[1]):
        y = X[:, i]
        others = np.delete(X, i, axis=1)
        ones = np.column_stack([others, np.ones(len(y))])
        coef, _, _, _ = lstsq(ones, y, rcond=None)
        y_pred = ones @ coef
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        vif = 1 / (1 - r2) if r2 < 1 else float('inf')
        if vif > max_vif:
            max_vif = vif

    problems = []
    if max_vif > 10:
        problems.append(f'VIF max = {max_vif:.1f} (>10)')
    if cn > 30:
        problems.append(f'Condition number = {cn:.1f} (>30)')
    if abs(r_ms) > 0.7:
        problems.append(f'Mkt/Sec correlation = {r_ms:.3f} (>0.7)')

    if problems:
        print('  Issues found:')
        for p in problems:
            print(f'    - {p}')
    else:
        print('  No multicollinearity issues detected.')
        print('  All 8 dimensions provide sufficiently independent information.')


if __name__ == '__main__':
    main()
