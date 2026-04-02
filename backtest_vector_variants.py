"""벡터 변형 전수 비교 테스트
================================
다중공선성 문제(Peak_Val VIF=18, Mkt/Sec r=0.79)를 해결하는
여러 벡터 구성의 유사도 예측력을 비교.

테스트 변형:
  A) 6D-orig : S_Force, S_Div, Peak_Val, Start/Pk, DD, Dur       (기존)
  B) 8D      : + Mkt, Sec                                         (현재)
  C) 7D-noPV : S_Force, S_Div, Start/Pk, DD, Dur, Mkt, Sec       (Peak_Val 제거)
  D) 6D-comb : S_Force, S_Div, Start/Pk, DD, Dur, MktSecAvg      (합산)
  E) 5D-clean: S_Force, S_Div, Start/Pk, DD, Dur                  (기본만)
  F) 7D-1mkt : S_Force, S_Div, Peak_Val, Start/Pk, DD, Dur, Mkt  (섹터 제거)
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


# ── 벡터 빌더 (6종) ──

def vec_6d_orig(s):
    """A) 기존 6D: Peak_Val 포함, 시장 없음."""
    sf, sd = s['s_force'] or 0, s['s_div'] or 0
    pv = s['peak_val'] or 0
    ratio = (s.get('start_val') or 0) / pv if pv > 0 else 0
    dd = min((s.get('dd_pct') or 0) / 0.30, 1.0)
    dur = min((s.get('duration') or 0) / 30.0, 1.0)
    return np.array([sf, sd, pv, ratio, dd, dur])

def vec_8d(s):
    """B) 현재 8D: 전부 포함."""
    v = vec_6d_orig(s)
    mkt, sec = s.get('market_return_20d'), s.get('sector_return_20d')
    if mkt is not None and sec is not None:
        return np.append(v, [np.tanh(mkt / 0.10), np.tanh(sec / 0.10)])
    return v

def vec_7d_nopv(s):
    """C) 7D: Peak_Val 제거 + 시장/섹터."""
    sf, sd = s['s_force'] or 0, s['s_div'] or 0
    pv = s['peak_val'] or 0
    ratio = (s.get('start_val') or 0) / pv if pv > 0 else 0
    dd = min((s.get('dd_pct') or 0) / 0.30, 1.0)
    dur = min((s.get('duration') or 0) / 30.0, 1.0)
    mkt, sec = s.get('market_return_20d'), s.get('sector_return_20d')
    if mkt is not None and sec is not None:
        return np.array([sf, sd, ratio, dd, dur, np.tanh(mkt/0.10), np.tanh(sec/0.10)])
    return np.array([sf, sd, ratio, dd, dur])

def vec_6d_comb(s):
    """D) 6D: Peak_Val 제거 + 시장/섹터 합산."""
    sf, sd = s['s_force'] or 0, s['s_div'] or 0
    pv = s['peak_val'] or 0
    ratio = (s.get('start_val') or 0) / pv if pv > 0 else 0
    dd = min((s.get('dd_pct') or 0) / 0.30, 1.0)
    dur = min((s.get('duration') or 0) / 30.0, 1.0)
    mkt, sec = s.get('market_return_20d'), s.get('sector_return_20d')
    if mkt is not None and sec is not None:
        env = np.tanh(((mkt + sec) / 2) / 0.10)
        return np.array([sf, sd, ratio, dd, dur, env])
    return np.array([sf, sd, ratio, dd, dur])

def vec_5d_clean(s):
    """E) 5D: 순수 시그널 특성만 (중복 없음)."""
    sf, sd = s['s_force'] or 0, s['s_div'] or 0
    pv = s['peak_val'] or 0
    ratio = (s.get('start_val') or 0) / pv if pv > 0 else 0
    dd = min((s.get('dd_pct') or 0) / 0.30, 1.0)
    dur = min((s.get('duration') or 0) / 30.0, 1.0)
    return np.array([sf, sd, ratio, dd, dur])

def vec_7d_1mkt(s):
    """F) 7D: Peak_Val 유지 + 시장(QQQ)만."""
    v = vec_6d_orig(s)
    mkt = s.get('market_return_20d')
    if mkt is not None:
        return np.append(v, [np.tanh(mkt / 0.10)])
    return v


VARIANTS = [
    ('A) 6D-orig',  vec_6d_orig,  'S_Force,S_Div,PeakVal,Start/Pk,DD,Dur'),
    ('B) 8D',       vec_8d,       'S_Force,S_Div,PeakVal,Start/Pk,DD,Dur,Mkt,Sec'),
    ('C) 7D-noPV',  vec_7d_nopv,  'S_Force,S_Div,Start/Pk,DD,Dur,Mkt,Sec'),
    ('D) 6D-comb',  vec_6d_comb,  'S_Force,S_Div,Start/Pk,DD,Dur,MktSecAvg'),
    ('E) 5D-clean', vec_5d_clean, 'S_Force,S_Div,Start/Pk,DD,Dur'),
    ('F) 7D-1mkt',  vec_7d_1mkt,  'S_Force,S_Div,PeakVal,Start/Pk,DD,Dur,Mkt'),
]


def collect_signals():
    """전 종목 시그널 수집 + 시장/섹터 enrichment."""
    wl = load_watchlist()
    tickers = list(wl['tickers'].keys()) + wl.get('benchmarks', [])
    params = wl.get('params', {})

    etf_data = {}
    for etf in ['QQQ'] + list(set(v for v in SECTOR_ETF_MAP.values() if v)):
        edf = download_data(etf, start='2020-01-01', end='2026-12-31')
        if edf is not None and len(edf) >= 40:
            etf_data[etf] = edf['Close'] / edf['Close'].shift(20) - 1.0

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
                                if pf is None or pf(zpi):
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
                                    }
                                    # 90d forward return
                                    if zpi + 90 < n:
                                        sig['return_90d'] = round(
                                            (df['Close'].iloc[zpi+90] - close) / close * 100, 2)
                                    else:
                                        sig['return_90d'] = None

                                    # Market context
                                    if 'QQQ' in etf_data:
                                        m = etf_data['QQQ'].index <= peak_ts
                                        if m.any():
                                            v = etf_data['QQQ'].loc[m].iloc[-1]
                                            if pd.notna(v):
                                                sig['market_return_20d'] = float(v)
                                    # Sector context
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

                                    all_signals.append(sig)
                                last_sig = zpi
                        in_zone = False
        except Exception:
            pass

    return all_signals


def run_similarity_test(completed, build_fn):
    """유사도 테스트: best-match의 same-outcome 비율 + Top3 방향 예측."""
    vecs = [build_fn(s) for s in completed]

    # Best-match pairs
    pairs = []  # (sim, same_outcome)
    for i in range(1, len(completed)):
        best_sim, best_j = -1, -1
        for j in range(i):
            s = _cosine_similarity(vecs[i], vecs[j])
            if s > best_sim:
                best_sim, best_j = s, j
        oi = completed[i]['return_90d'] > 0
        oj = completed[best_j]['return_90d'] > 0
        pairs.append((best_sim, oi == oj))

    # 임계값별 정확도
    results = {}
    for t in [0.80, 0.90, 0.95, 0.97, 0.99]:
        sub = [(s, o) for s, o in pairs if s > t]
        if sub:
            acc = sum(1 for _, o in sub if o) / len(sub) * 100
            results[t] = (len(sub), acc)
        else:
            results[t] = (0, 0)

    # Top-3 방향 예측
    correct_dir = 0
    total_dir = 0
    for i in range(5, len(completed)):
        sims = []
        for j in range(i):
            if completed[j]['return_90d'] is not None:
                s = _cosine_similarity(vecs[i], vecs[j])
                sims.append((s, completed[j]['return_90d']))
        sims.sort(key=lambda x: x[0], reverse=True)
        top3 = sims[:3]
        if top3:
            pred = np.mean([r for _, r in top3]) > 0
            actual = completed[i]['return_90d'] > 0
            total_dir += 1
            if pred == actual:
                correct_dir += 1

    dir_acc = correct_dir / total_dir * 100 if total_dir > 0 else 0

    # Top-3 MAE
    errors = []
    for i in range(5, len(completed)):
        sims = []
        for j in range(i):
            if completed[j]['return_90d'] is not None:
                s = _cosine_similarity(vecs[i], vecs[j])
                sims.append((s, completed[j]['return_90d']))
        sims.sort(key=lambda x: x[0], reverse=True)
        top3 = sims[:3]
        if top3:
            pred = np.mean([r for _, r in top3])
            errors.append(abs(completed[i]['return_90d'] - pred))
    mae = np.mean(errors) if errors else 0

    return results, dir_acc, mae


def main():
    print('=' * 78)
    print('  Vector Variant Comparison Test')
    print('  (All approaches for multicollinearity resolution)')
    print('=' * 78)

    print('\n  Collecting signals...')
    all_signals = collect_signals()
    completed = [s for s in all_signals if s['return_90d'] is not None]
    print(f'  Total: {len(all_signals)}, 90d completed: {len(completed)}')

    # ── 전체 승률 (랜덤 기대값 계산용) ──
    wins = sum(1 for s in completed if s['return_90d'] > 0)
    wr = wins / len(completed)
    random_acc = wr ** 2 + (1 - wr) ** 2
    print(f'  Base win rate: {wr*100:.1f}% -> random same-outcome: {random_acc*100:.1f}%')

    # ═══════════════════════════════════════════════════════
    # 1. 임계값별 정확도 비교표
    # ═══════════════════════════════════════════════════════
    print('\n')
    print('=' * 78)
    print('  1. Same-Outcome Accuracy by Similarity Threshold')
    print('=' * 78)

    all_results = {}
    all_dir = {}
    all_mae = {}

    for name, fn, desc in VARIANTS:
        print(f'\n  Testing {name}...')
        results, dir_acc, mae = run_similarity_test(completed, fn)
        all_results[name] = results
        all_dir[name] = dir_acc
        all_mae[name] = mae

    # 표 출력
    thresholds = [0.80, 0.90, 0.95, 0.97, 0.99]

    for t in thresholds:
        print(f'\n  --- Threshold > {t:.0%} ---')
        print(f'  {"Variant":>14s}  {"Pairs":>6s}  {"Accuracy":>8s}  {"vs Random":>10s}')
        print(f'  {"-"*14}  {"-"*6}  {"-"*8}  {"-"*10}')

        best_acc = 0
        rows = []
        for name, _, _ in VARIANTS:
            n_pairs, acc = all_results[name][t]
            vs_rand = acc - random_acc * 100
            rows.append((name, n_pairs, acc, vs_rand))
            if acc > best_acc and n_pairs > 10:
                best_acc = acc

        for name, n_pairs, acc, vs_rand in rows:
            marker = ' <-- BEST' if acc == best_acc and n_pairs > 10 else ''
            print(f'  {name:>14s}  {n_pairs:>6d}  {acc:>7.1f}%  {vs_rand:>+9.1f}%p{marker}')

    # ═══════════════════════════════════════════════════════
    # 2. Top-3 방향 예측 정확도
    # ═══════════════════════════════════════════════════════
    print('\n')
    print('=' * 78)
    print('  2. Top-3 Similar Signals -> Direction Prediction')
    print('     (Top-3 avg return > 0 predicts actual > 0?)')
    print('=' * 78)

    print(f'\n  {"Variant":>14s}  {"Direction Acc":>14s}  {"vs Random":>10s}')
    print(f'  {"-"*14}  {"-"*14}  {"-"*10}')

    best_dir = max(all_dir.values())
    for name, _, desc in VARIANTS:
        d = all_dir[name]
        vs = d - wr * 100  # vs base win rate
        marker = ' <-- BEST' if d == best_dir else ''
        print(f'  {name:>14s}  {d:>13.1f}%  {vs:>+9.1f}%p{marker}')

    # ═══════════════════════════════════════════════════════
    # 3. MAE (예측 오차)
    # ═══════════════════════════════════════════════════════
    print('\n')
    print('=' * 78)
    print('  3. Top-3 Similar Signals -> Return MAE')
    print('     (Lower = better prediction of actual 90d return)')
    print('=' * 78)

    print(f'\n  {"Variant":>14s}  {"MAE":>10s}')
    print(f'  {"-"*14}  {"-"*10}')

    best_mae = min(all_mae.values())
    for name, _, _ in VARIANTS:
        m = all_mae[name]
        marker = ' <-- BEST' if m == best_mae else ''
        print(f'  {name:>14s}  {m:>9.2f}%p{marker}')

    # ═══════════════════════════════════════════════════════
    # 4. 종합 점수 (정규화 랭킹)
    # ═══════════════════════════════════════════════════════
    print('\n')
    print('=' * 78)
    print('  4. Overall Ranking')
    print('     Score = avg of: >95% acc, >99% acc, direction acc, (100-MAE)')
    print('=' * 78)

    scores = {}
    for name, _, desc in VARIANTS:
        _, acc95 = all_results[name][0.95]
        _, acc99 = all_results[name][0.99]
        dir_acc = all_dir[name]
        mae = all_mae[name]
        # 종합 점수 (높을수록 좋음)
        score = (acc95 + acc99 + dir_acc + (100 - mae)) / 4
        scores[name] = {
            'acc95': acc95, 'acc99': acc99,
            'dir': dir_acc, 'mae': mae,
            'total': score, 'desc': desc,
        }

    ranked = sorted(scores.items(), key=lambda x: x[1]['total'], reverse=True)

    print(f'\n  {"Rank":>4s}  {"Variant":>14s}  {">95%":>6s}  {">99%":>6s}  {"Dir":>6s}  {"MAE":>6s}  {"Score":>6s}  Description')
    print(f'  {"-"*4}  {"-"*14}  {"-"*6}  {"-"*6}  {"-"*6}  {"-"*6}  {"-"*6}  {"-"*20}')

    for rank, (name, s) in enumerate(ranked, 1):
        medal = ['1st', '2nd', '3rd'][rank-1] if rank <= 3 else f'{rank}th'
        print(f'  {medal:>4s}  {name:>14s}  {s["acc95"]:>5.1f}%  {s["acc99"]:>5.1f}%  {s["dir"]:>5.1f}%  {s["mae"]:>5.1f}  {s["total"]:>5.1f}  {s["desc"]}')

    winner = ranked[0][0]
    print(f'\n  Winner: {winner}')
    print(f'  -> {scores[winner]["desc"]}')

    # ═══════════════════════════════════════════════════════
    # 5. VIF 비교 (winner)
    # ═══════════════════════════════════════════════════════
    print('\n')
    print('=' * 78)
    print('  5. VIF Check for Top-3 Variants')
    print('=' * 78)

    from numpy.linalg import lstsq

    for rank, (name, fn, desc) in enumerate([(n, f, d) for n, f, d in VARIANTS if n in [r[0] for r in ranked[:3]]]):
        vecs = [fn(s) for s in completed[:200]]  # 200개 샘플
        # 동일 차원만
        dims = set(len(v) for v in vecs)
        max_dim = max(dims)
        vecs_same = [v for v in vecs if len(v) == max_dim]

        if len(vecs_same) < 50:
            continue

        X = np.array(vecs_same)
        n_vars = X.shape[1]

        print(f'\n  {name} ({desc}):')
        max_vif = 0
        for i in range(n_vars):
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
            status = 'SEVERE' if vif > 10 else 'WARN' if vif > 5 else 'OK'
            print(f'    dim{i}: VIF = {vif:>6.2f}  [{status}]')
        print(f'    Max VIF: {max_vif:.2f}')

    print('\n' + '=' * 78)


if __name__ == '__main__':
    main()
