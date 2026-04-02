"""VIX Factor 추가 효과 테스트
================================
기존 7D-noPV에 VIX 20일 변화율을 추가한 8D-VIX의 효과 검증.

비교 대상:
  A) 7D-noPV : S_Force, S_Div, Start/Pk, DD, Dur, Mkt, Sec       (현재)
  B) 8D-VIX  : S_Force, S_Div, Start/Pk, DD, Dur, Mkt, Sec, VIX  (VIX 추가)
  C) 8D-VIXonly: S_Force, S_Div, Start/Pk, DD, Dur, Mkt, VIX     (Sec→VIX 교체)
  D) 7D-VIXreplace: S_Force, S_Div, Start/Pk, DD, Dur, Mkt, VIX  (동일하지만 다른 정규화)

추가 분석:
  - VIX vs 기존 변수 상관관계
  - VIX의 독립 정보량
  - 시장 레짐별 VIX 효과
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
from v4wp_realtime.core.similarity import _cosine_similarity


def collect_signals():
    wl = load_watchlist()
    tickers = list(wl['tickers'].keys()) + wl.get('benchmarks', [])
    params = wl.get('params', {})

    # ETF + VIX 데이터
    etf_data = {}
    for etf in ['QQQ'] + list(set(v for v in SECTOR_ETF_MAP.values() if v)):
        edf = download_data(etf, start='2020-01-01', end='2026-12-31')
        if edf is not None and len(edf) >= 40:
            etf_data[etf] = edf['Close'] / edf['Close'].shift(20) - 1.0

    # VIX 다운로드 (^VIX는 download_data 캐시 문제 → 직접 로드)
    vix_change_20d = None
    try:
        _base = os.path.dirname(os.path.abspath(__file__))
        vix_cache = os.path.join(_base, 'cache', 'VIX_2020-01-01_2026-12-31.csv')
        if os.path.exists(vix_cache):
            # yfinance MultiIndex CSV: skip first 2 header rows
            vix_raw = pd.read_csv(vix_cache, skiprows=[1, 2], index_col=0, parse_dates=True)
            close_col = vix_raw.iloc[:, 0].astype(float)  # Close column
            close_col.index = pd.to_datetime(close_col.index)
        else:
            import yfinance as yf
            vix_df = yf.download('^VIX', start='2020-01-01', end='2026-12-31', progress=False)
            close_col = vix_df['Close']
            if hasattr(close_col, 'columns'):
                close_col = close_col.iloc[:, 0]
        if len(close_col) >= 40:
            vix_change_20d = close_col / close_col.shift(20) - 1.0
            print(f'  VIX: {len(close_col)} days loaded')
    except Exception as e:
        print(f'  VIX load failed: {e}')

    all_signals = []
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
                        in_zone = True; zs = i; zpi = i; zpv = val
                    elif val > zpv:
                        zpi = i; zpv = val
                else:
                    if in_zone:
                        dur = zpi - zs + 1
                        if dur >= 1 and zpi - last_sig > cooldown:
                            lb = max(0, zpi - dd_lookback)
                            high_nd = df['Close'].iloc[lb:zpi+1].max()
                            close = df['Close'].iloc[zpi]
                            dd = (high_nd - close) / high_nd if high_nd > 0 else 0
                            if dd >= dd_threshold:
                                peak_ts = df.index[zpi]
                                sig = {
                                    'peak_idx': zpi,
                                    'peak_date': df.index[zpi].strftime('%Y-%m-%d'),
                                    'peak_val': float(zpv),
                                    'start_val': float(scores.iloc[zs]),
                                    'close_price': float(close),
                                    's_force': float(subind['s_force'].iloc[zpi]),
                                    's_div': float(subind['s_div'].iloc[zpi]),
                                    'dd_pct': round(dd, 6),
                                    'duration': dur,
                                    'ticker': ticker,
                                    'sector': wl['tickers'].get(ticker, {}).get('sector', 'Benchmark'),
                                    'market_return_20d': None,
                                    'sector_return_20d': None,
                                    'vix_change_20d': None,
                                }
                                # 90d forward
                                if zpi + 90 < n:
                                    sig['return_90d'] = round(
                                        (df['Close'].iloc[zpi+90] - close) / close * 100, 2)
                                else:
                                    sig['return_90d'] = None

                                # Market
                                if 'QQQ' in etf_data:
                                    m = etf_data['QQQ'].index <= peak_ts
                                    if m.any():
                                        v = etf_data['QQQ'].loc[m].iloc[-1]
                                        if pd.notna(v):
                                            sig['market_return_20d'] = float(v)
                                # Sector
                                se = SECTOR_ETF_MAP.get(sig['sector'])
                                if se is None and ticker in ('QQQ', 'VOO'):
                                    r20 = df['Close'] / df['Close'].shift(20) - 1
                                    v = r20.iloc[zpi]
                                    if pd.notna(v):
                                        sig['sector_return_20d'] = float(v)
                                elif se and se in etf_data:
                                    m = etf_data[se].index <= peak_ts
                                    if m.any():
                                        v = etf_data[se].loc[m].iloc[-1]
                                        if pd.notna(v):
                                            sig['sector_return_20d'] = float(v)
                                # VIX
                                if vix_change_20d is not None:
                                    m = vix_change_20d.index <= peak_ts
                                    if m.any():
                                        v = vix_change_20d.loc[m].iloc[-1]
                                        if pd.notna(v):
                                            sig['vix_change_20d'] = float(v)

                                all_signals.append(sig)
                            last_sig = zpi
                        in_zone = False
        except Exception:
            pass

    return all_signals


# ── 벡터 빌더 ──

def vec_7d_nopv(s):
    """A) 현재 7D-noPV."""
    sf, sd = s['s_force'] or 0, s['s_div'] or 0
    pv = s['peak_val'] or 0
    ratio = (s.get('start_val') or 0) / pv if pv > 0 else 0
    dd = min((s.get('dd_pct') or 0) / 0.30, 1.0)
    dur = min((s.get('duration') or 0) / 30.0, 1.0)
    mkt, sec = s.get('market_return_20d'), s.get('sector_return_20d')
    if mkt is not None and sec is not None:
        return np.array([sf, sd, ratio, dd, dur, np.tanh(mkt/0.10), np.tanh(sec/0.10)])
    return np.array([sf, sd, ratio, dd, dur])

def vec_8d_vix(s):
    """B) 8D: 7D + VIX."""
    sf, sd = s['s_force'] or 0, s['s_div'] or 0
    pv = s['peak_val'] or 0
    ratio = (s.get('start_val') or 0) / pv if pv > 0 else 0
    dd = min((s.get('dd_pct') or 0) / 0.30, 1.0)
    dur = min((s.get('duration') or 0) / 30.0, 1.0)
    mkt = s.get('market_return_20d')
    sec = s.get('sector_return_20d')
    vix = s.get('vix_change_20d')
    if mkt is not None and sec is not None and vix is not None:
        # VIX: tanh(변화율/0.30) — VIX는 변동이 크므로 30% 기준
        return np.array([sf, sd, ratio, dd, dur,
                         np.tanh(mkt/0.10), np.tanh(sec/0.10), np.tanh(vix/0.30)])
    if mkt is not None and sec is not None:
        return np.array([sf, sd, ratio, dd, dur, np.tanh(mkt/0.10), np.tanh(sec/0.10)])
    return np.array([sf, sd, ratio, dd, dur])

def vec_7d_vix_replace(s):
    """C) 7D: Sec를 VIX로 교체."""
    sf, sd = s['s_force'] or 0, s['s_div'] or 0
    pv = s['peak_val'] or 0
    ratio = (s.get('start_val') or 0) / pv if pv > 0 else 0
    dd = min((s.get('dd_pct') or 0) / 0.30, 1.0)
    dur = min((s.get('duration') or 0) / 30.0, 1.0)
    mkt = s.get('market_return_20d')
    vix = s.get('vix_change_20d')
    if mkt is not None and vix is not None:
        return np.array([sf, sd, ratio, dd, dur, np.tanh(mkt/0.10), np.tanh(vix/0.30)])
    return np.array([sf, sd, ratio, dd, dur])


VARIANTS = [
    ('A) 7D-noPV',     vec_7d_nopv,       'S_F,S_D,Ratio,DD,Dur,Mkt,Sec'),
    ('B) 8D+VIX',      vec_8d_vix,        'S_F,S_D,Ratio,DD,Dur,Mkt,Sec,VIX'),
    ('C) 7D-VIXrepl',  vec_7d_vix_replace, 'S_F,S_D,Ratio,DD,Dur,Mkt,VIX'),
]


def run_test(completed, build_fn):
    vecs = [build_fn(s) for s in completed]
    pairs = []
    for i in range(1, len(completed)):
        best_sim, best_j = -1, -1
        for j in range(i):
            s = _cosine_similarity(vecs[i], vecs[j])
            if s > best_sim:
                best_sim, best_j = s, j
        oi = completed[i]['return_90d'] > 0
        oj = completed[best_j]['return_90d'] > 0
        pairs.append((best_sim, oi == oj, completed[i]))

    results = {}
    for t in [0.80, 0.90, 0.95, 0.97, 0.99]:
        sub = [(s, o) for s, o, _ in pairs if s > t]
        if sub:
            results[t] = (len(sub), sum(1 for _, o in sub if o) / len(sub) * 100)
        else:
            results[t] = (0, 0)

    # Top-3 direction
    correct = total = 0
    for i in range(5, len(completed)):
        sims = []
        for j in range(i):
            if completed[j]['return_90d'] is not None:
                sims.append((_cosine_similarity(vecs[i], vecs[j]), completed[j]['return_90d']))
        sims.sort(reverse=True)
        top3 = sims[:3]
        if top3:
            total += 1
            if (np.mean([r for _, r in top3]) > 0) == (completed[i]['return_90d'] > 0):
                correct += 1
    dir_acc = correct / total * 100 if total else 0

    # MAE
    errors = []
    for i in range(5, len(completed)):
        sims = []
        for j in range(i):
            if completed[j]['return_90d'] is not None:
                sims.append((_cosine_similarity(vecs[i], vecs[j]), completed[j]['return_90d']))
        sims.sort(reverse=True)
        top3 = sims[:3]
        if top3:
            errors.append(abs(completed[i]['return_90d'] - np.mean([r for _, r in top3])))
    mae = np.mean(errors) if errors else 0

    # 레짐별 분석 (>95% 유사도)
    regime_results = {}
    for regime in ['BULL', 'NEUTRAL', 'BEAR']:
        sub = [(s, o) for s, o, sig in pairs
               if s > 0.95 and classify_regime(sig.get('market_return_20d')) == regime]
        if len(sub) >= 10:
            regime_results[regime] = (len(sub), sum(1 for _, o in sub if o) / len(sub) * 100)

    return results, dir_acc, mae, regime_results


def classify_regime(mkt_ret):
    if mkt_ret is None:
        return 'UNKNOWN'
    if mkt_ret > 0.05:
        return 'BULL'
    elif mkt_ret < -0.05:
        return 'BEAR'
    return 'NEUTRAL'


def main():
    print('=' * 78)
    print('  VIX Factor Test: 7D-noPV vs 8D+VIX vs 7D-VIXreplace')
    print('=' * 78)

    print('\n  Collecting signals...')
    all_signals = collect_signals()
    completed = [s for s in all_signals if s['return_90d'] is not None]
    has_vix = sum(1 for s in completed if s.get('vix_change_20d') is not None)
    print(f'  Total: {len(all_signals)}, completed: {len(completed)}, with VIX: {has_vix}')

    wins = sum(1 for s in completed if s['return_90d'] > 0)
    wr = wins / len(completed)
    rand = wr**2 + (1-wr)**2
    print(f'  Win rate: {wr*100:.1f}%, random same-outcome: {rand*100:.1f}%')

    # ═══════════════════════════════════════
    # 1. VIX vs 기존 변수 상관관계
    # ═══════════════════════════════════════
    print('\n')
    print('=' * 78)
    print('  1. VIX 20d Change vs Other Variables (Correlation)')
    print('=' * 78)

    vix_vals = []
    other_vals = {'S_Force': [], 'S_Div': [], 'DD_pct': [], 'Duration': [],
                  'Mkt_20d': [], 'Sec_20d': [], 'Return_90d': []}

    for s in completed:
        if s.get('vix_change_20d') is not None and s.get('market_return_20d') is not None:
            vix_vals.append(s['vix_change_20d'])
            other_vals['S_Force'].append(s['s_force'] or 0)
            other_vals['S_Div'].append(s['s_div'] or 0)
            other_vals['DD_pct'].append(s.get('dd_pct') or 0)
            other_vals['Duration'].append(s.get('duration') or 0)
            other_vals['Mkt_20d'].append(s['market_return_20d'])
            other_vals['Sec_20d'].append(s.get('sector_return_20d') or 0)
            other_vals['Return_90d'].append(s['return_90d'])

    if vix_vals:
        vix_arr = np.array(vix_vals)
        print(f'\n  VIX 20d change stats: mean={np.mean(vix_arr)*100:+.1f}%, '
              f'std={np.std(vix_arr)*100:.1f}%, '
              f'range=[{np.min(vix_arr)*100:+.0f}%, {np.max(vix_arr)*100:+.0f}%]')
        print()
        print(f'  {"Variable":>12s}  {"Corr with VIX":>14s}  {"Interpretation":>20s}')
        print(f'  {"-"*12}  {"-"*14}  {"-"*20}')

        for name, vals in other_vals.items():
            r = np.corrcoef(vix_arr, vals)[0, 1]
            if abs(r) > 0.7:
                interp = 'HIGH (redundant?)'
            elif abs(r) > 0.5:
                interp = 'MODERATE'
            elif abs(r) > 0.3:
                interp = 'WEAK'
            else:
                interp = 'INDEPENDENT (good)'
            print(f'  {name:>12s}  {r:>+13.3f}  {interp:>20s}')

    # ═══════════════════════════════════════
    # 2. VIF 비교
    # ═══════════════════════════════════════
    print('\n')
    print('=' * 78)
    print('  2. VIF Comparison')
    print('=' * 78)

    from numpy.linalg import lstsq

    for name, fn, desc in VARIANTS:
        vecs = [fn(s) for s in completed[:300]]
        max_dim = max(len(v) for v in vecs)
        vecs_same = np.array([v for v in vecs if len(v) == max_dim])
        if len(vecs_same) < 50:
            continue

        max_vif = 0
        vifs = []
        for i in range(vecs_same.shape[1]):
            y = vecs_same[:, i]
            others = np.delete(vecs_same, i, axis=1)
            ones = np.column_stack([others, np.ones(len(y))])
            coef, _, _, _ = lstsq(ones, y, rcond=None)
            y_pred = ones @ coef
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            vif = 1 / (1 - r2) if r2 < 1 else float('inf')
            vifs.append(vif)
            if vif > max_vif:
                max_vif = vif

        print(f'\n  {name} ({desc}):')
        dim_labels = desc.split(',')
        for idx, vif in enumerate(vifs):
            label = dim_labels[idx] if idx < len(dim_labels) else f'd{idx}'
            status = 'SEVERE' if vif > 10 else 'WARN' if vif > 5 else 'OK'
            print(f'    {label:>6s}: VIF = {vif:>6.2f}  [{status}]')
        print(f'    Max VIF: {max_vif:.2f}')

    # ═══════════════════════════════════════
    # 3. 유사도 예측력 비교
    # ═══════════════════════════════════════
    print('\n')
    print('=' * 78)
    print('  3. Similarity Prediction Accuracy')
    print('=' * 78)

    all_results = {}
    all_dir = {}
    all_mae = {}
    all_regime = {}

    for name, fn, desc in VARIANTS:
        print(f'\n  Testing {name}...')
        results, dir_acc, mae, regime = run_test(completed, fn)
        all_results[name] = results
        all_dir[name] = dir_acc
        all_mae[name] = mae
        all_regime[name] = regime

    thresholds = [0.90, 0.95, 0.97, 0.99]
    for t in thresholds:
        print(f'\n  --- Threshold > {t:.0%} ---')
        print(f'  {"Variant":>16s}  {"Pairs":>6s}  {"Accuracy":>8s}  {"vs Random":>10s}')
        print(f'  {"-"*16}  {"-"*6}  {"-"*8}  {"-"*10}')
        best_acc = max(all_results[n][t][1] for n, _, _ in VARIANTS if all_results[n][t][0] > 10)
        for name, _, _ in VARIANTS:
            n_p, acc = all_results[name][t]
            vs = acc - rand * 100
            marker = ' <-- BEST' if acc == best_acc and n_p > 10 else ''
            print(f'  {name:>16s}  {n_p:>6d}  {acc:>7.1f}%  {vs:>+9.1f}%p{marker}')

    # ═══════════════════════════════════════
    # 4. Direction + MAE
    # ═══════════════════════════════════════
    print('\n')
    print('=' * 78)
    print('  4. Top-3 Direction Prediction & MAE')
    print('=' * 78)

    print(f'\n  {"Variant":>16s}  {"Direction":>10s}  {"MAE":>8s}')
    print(f'  {"-"*16}  {"-"*10}  {"-"*8}')
    for name, _, _ in VARIANTS:
        print(f'  {name:>16s}  {all_dir[name]:>9.1f}%  {all_mae[name]:>7.2f}')

    # ═══════════════════════════════════════
    # 5. 시장 레짐별 분석
    # ═══════════════════════════════════════
    print('\n')
    print('=' * 78)
    print('  5. Accuracy by Market Regime (>95% similarity)')
    print('=' * 78)

    for regime in ['BULL', 'NEUTRAL', 'BEAR']:
        print(f'\n  {regime}:')
        for name, _, _ in VARIANTS:
            if regime in all_regime[name]:
                n_p, acc = all_regime[name][regime]
                print(f'    {name:>16s}: {n_p:>4d} pairs, {acc:.1f}%')
            else:
                print(f'    {name:>16s}: insufficient data')

    # ═══════════════════════════════════════
    # 6. VIX 구간별 90d 수익률 (VIX 자체의 예측력)
    # ═══════════════════════════════════════
    print('\n')
    print('=' * 78)
    print('  6. VIX 20d Change vs 90d Return (VIX predictive power)')
    print('=' * 78)

    vix_sigs = [s for s in completed if s.get('vix_change_20d') is not None]
    if vix_sigs:
        bins = [(-9, -0.20, 'VIX -20%+'), (-0.20, -0.05, 'VIX -5~-20%'),
                (-0.05, 0.05, 'VIX -5~+5%'), (0.05, 0.20, 'VIX +5~+20%'),
                (0.20, 0.50, 'VIX +20~50%'), (0.50, 9, 'VIX +50%+')]

        print(f'\n  {"VIX 구간":>14s}  {"건수":>5s}  {"승률":>6s}  {"평균 90d":>10s}  {"해석":>10s}')
        print(f'  {"-"*14}  {"-"*5}  {"-"*6}  {"-"*10}  {"-"*10}')

        for lo, hi, label in bins:
            sub = [s for s in vix_sigs if lo <= s['vix_change_20d'] < hi]
            if len(sub) >= 5:
                w = sum(1 for s in sub if s['return_90d'] > 0)
                avg = np.mean([s['return_90d'] for s in sub])
                wr_pct = w / len(sub) * 100
                interpretation = 'strong' if avg > 30 else 'good' if avg > 10 else 'weak' if avg > 0 else 'bad'
                print(f'  {label:>14s}  {len(sub):>5d}  {wr_pct:>5.0f}%  {avg:>+9.1f}%  {interpretation:>10s}')

    # ═══════════════════════════════════════
    # 7. 종합
    # ═══════════════════════════════════════
    print('\n')
    print('=' * 78)
    print('  7. Overall Ranking')
    print('=' * 78)

    print(f'\n  {"Variant":>16s}  {">95%":>6s}  {">99%":>6s}  {"Dir":>6s}  {"MAE":>6s}  {"Score":>6s}')
    print(f'  {"-"*16}  {"-"*6}  {"-"*6}  {"-"*6}  {"-"*6}  {"-"*6}')

    ranked = []
    for name, _, desc in VARIANTS:
        _, acc95 = all_results[name][0.95]
        _, acc99 = all_results[name][0.99]
        d = all_dir[name]
        m = all_mae[name]
        score = (acc95 + acc99 + d + (100 - m)) / 4
        ranked.append((name, acc95, acc99, d, m, score, desc))

    ranked.sort(key=lambda x: x[5], reverse=True)
    for name, a95, a99, d, m, score, desc in ranked:
        marker = ' <-- BEST' if score == ranked[0][5] else ''
        print(f'  {name:>16s}  {a95:>5.1f}%  {a99:>5.1f}%  {d:>5.1f}%  {m:>5.1f}  {score:>5.1f}{marker}')

    print(f'\n  Winner: {ranked[0][0]}')
    print(f'  -> {ranked[0][6]}')
    print()
    print('=' * 78)


if __name__ == '__main__':
    main()
