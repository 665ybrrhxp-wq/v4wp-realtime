"""
V4 Three Findings Deep Dive
============================
3가지 발견을 최대 데이터(period='max')와 bootstrap 검증으로 규명.

발견 1: DD Gate 0~3% 역설 — 차단 구간이 5~10% 통과 구간보다 수익 좋음
발견 2: 저거래량 신호가 오히려 최고 수익 (V_norm 0.7~1.0)
발견 3: 10일+ 연속 V_norm>1 = 압도적 성과 (n=10, 검증 필요)
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
LONG_HISTORY = {
    'QQQ': 'Benchmark', 'VOO': 'Benchmark',
    'GOOGL': 'Tech', 'AMZN': 'Tech', 'NVDA': 'Tech',
    'AVGO': 'Tech', 'TSLA': 'Tech',
}

EXTRA_LARGECAP = {
    'AAPL': 'LargeCap', 'MSFT': 'LargeCap', 'META': 'LargeCap',
    'JPM': 'LargeCap', 'BRK-B': 'LargeCap', 'V': 'LargeCap',
    'UNH': 'LargeCap', 'XOM': 'LargeCap',
}

FULL_WATCHLIST = {
    'TSLA': 'Tech', 'NVDA': 'Tech', 'AVGO': 'Tech',
    'AMZN': 'Tech', 'GOOGL': 'Tech', 'PLTR': 'Tech',
    'COIN': 'Fintech', 'HOOD': 'Fintech', 'SOFI': 'Fintech',
    'HIMS': 'Growth', 'RKLB': 'Growth', 'JOBY': 'Growth',
    'PGY': 'Growth',
    'IONQ': 'Quantum', 'PL': 'Space', 'ASTS': 'Space',
    'QQQ': 'Benchmark', 'VOO': 'Benchmark',
}

# Production params (GEO-OP)
SIGNAL_TH = 0.05
COOLDOWN = 5
ER_Q = 80
ATR_Q = 40
LOOKBACK_PF = 252
DIVGATE = 3

HORIZONS = [10, 20, 30, 60, 90]
N_BOOTSTRAP = 10000


# ═══════════════════════════════════════════════════════════
# Data Download
# ═══════════════════════════════════════════════════════════
def download_max(ticker):
    t = yf.Ticker(ticker)
    df = t.history(period='max', auto_adjust=True)
    if df is None or len(df) == 0:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.dropna(subset=['Close', 'Volume'])
    df = df[df['Close'] > 0]
    return df


def download_5y(ticker):
    t = yf.Ticker(ticker)
    df = t.history(period='5y', auto_adjust=True)
    if df is None or len(df) == 0:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.dropna(subset=['Close', 'Volume'])
    df = df[df['Close'] > 0]
    return df


# ═══════════════════════════════════════════════════════════
# Bootstrap Helpers
# ═══════════════════════════════════════════════════════════
def bootstrap_ci(vals, n_boot=N_BOOTSTRAP, seed=42):
    if len(vals) < 2:
        m = np.mean(vals) if len(vals) == 1 else np.nan
        return m, np.nan, np.nan
    rng = np.random.default_rng(seed)
    arr = np.array(vals)
    means = np.array([np.mean(rng.choice(arr, size=len(arr), replace=True))
                      for _ in range(n_boot)])
    return float(np.mean(arr)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_vs_pool(signal_vals, pool_vals, n_boot=N_BOOTSTRAP, seed=42):
    if len(signal_vals) < 2 or len(pool_vals) < 2:
        return {'n': len(signal_vals), 'signal_mean': np.mean(signal_vals) if len(signal_vals) else np.nan,
                'p_value': np.nan, 'pctile': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan}
    rng = np.random.default_rng(seed)
    sig_mean = float(np.mean(signal_vals))
    n = len(signal_vals)
    pool = np.array(pool_vals)
    boot_means = np.array([np.mean(rng.choice(pool, size=n, replace=True))
                           for _ in range(n_boot)])
    return {
        'n': n,
        'signal_mean': sig_mean,
        'pool_mean': float(np.mean(pool)),
        'p_value': float(np.mean(boot_means >= sig_mean)),
        'pctile': float(np.mean(boot_means < sig_mean) * 100),
        'ci_lo': float(np.percentile(boot_means, 2.5)),
        'ci_hi': float(np.percentile(boot_means, 97.5)),
    }


# ═══════════════════════════════════════════════════════════
# Shared Signal Collection (DD gate 전)
# ═══════════════════════════════════════════════════════════
def collect_raw_signals(ticker, df, sector='Unknown'):
    if df is None or len(df) < 252:
        return None

    df = smooth_earnings_volume(df, ticker)
    score = calc_v4_score(df, w=20, divgate_days=DIVGATE)
    sub = calc_v4_subindicators(df, w=20, divgate_days=DIVGATE)

    events = detect_signal_events(score, th=SIGNAL_TH, cooldown=COOLDOWN)
    pf = build_price_filter(df, er_q=ER_Q, atr_q=ATR_Q, lookback=LOOKBACK_PF)
    buy_events = [e for e in events if e['type'] == 'bottom' and pf(e['peak_idx'])]

    if not buy_events:
        return None

    close = df['Close'].values
    n = len(close)

    # Precompute rolling highs for multiple lookbacks
    rolling_highs = {}
    for lb in [5, 10, 15, 20, 30]:
        rolling_highs[lb] = pd.Series(close).rolling(lb, min_periods=1).max().values

    # V_norm
    v_norm = (df['Volume'] / df['Volume'].rolling(20).mean()).fillna(1).values

    # SMA200
    sma200 = df['Close'].rolling(200, min_periods=200).mean().values

    # V_norm streak
    surge_vnorm = np.zeros(n)
    for i in range(n):
        if v_norm[i] > 1.0:
            surge_vnorm[i] = (surge_vnorm[i-1] + 1) if i > 0 else 1

    # S_Div streak
    pv_div = calc_pv_divergence(df, 20)
    raw_div = np.array([np.clip(pv_div.iloc[i] / 3, -1, 1) for i in range(n)])
    surge_sdiv = np.zeros(n)
    for i in range(n):
        if raw_div[i] > 0:
            surge_sdiv[i] = (surge_sdiv[i-1] + 1) if i > 0 else 1

    rows = []
    for ev in buy_events:
        pidx = ev['peak_idx']
        if pidx >= n or pidx < 200:
            continue
        price = close[pidx]
        date = df.index[pidx]

        # DD for multiple lookbacks
        dds = {}
        for lb in [5, 10, 15, 20, 30]:
            rh = rolling_highs[lb][pidx]
            dds[lb] = (rh - price) / rh if rh > 0 else 0

        # Forward returns
        fwd = {}
        hit = {}
        for h in HORIZONS:
            end_idx = min(pidx + h, n - 1)
            if end_idx > pidx:
                ret = (close[end_idx] - price) / price * 100
                fwd[h] = ret
                hit[h] = 1 if ret > 0 else 0
            else:
                fwd[h] = np.nan
                hit[h] = np.nan

        # MFE/MAE 60d
        mfe_60 = mae_60 = np.nan
        end_60 = min(pidx + 60, n - 1)
        if end_60 > pidx:
            wh = df['High'].values[pidx+1:end_60+1]
            wl = df['Low'].values[pidx+1:end_60+1]
            if len(wh) > 0:
                mfe_60 = (max(wh) - price) / price * 100
                mae_60 = (price - min(wl)) / price * 100

        # Trend 20d
        trend_20d = (close[pidx] - close[max(0, pidx-20)]) / close[max(0, pidx-20)] * 100

        row = {
            'ticker': ticker, 'sector': sector,
            'date': date, 'year': date.year,
            'price': price, 'score': ev['peak_val'],
            's_force': sub['s_force'].iloc[pidx],
            's_div': sub['s_div'].iloc[pidx],
            'duration': ev['duration'],
            'v_norm': v_norm[pidx],
            'surge_vnorm': surge_vnorm[pidx],
            'surge_sdiv': surge_sdiv[pidx],
            'ma200_above': 1 if (not np.isnan(sma200[pidx]) and close[pidx] > sma200[pidx]) else 0,
            'trend_20d': trend_20d,
            'mfe_60': mfe_60, 'mae_60': mae_60,
        }
        for lb in [5, 10, 15, 20, 30]:
            row[f'dd_{lb}d'] = dds[lb]
        for h in HORIZONS:
            row[f'fwd_{h}d'] = fwd[h]
            row[f'hit_{h}d'] = hit[h]

        rows.append(row)

    return pd.DataFrame(rows) if rows else None


# ═══════════════════════════════════════════════════════════
# EXPERIMENT 1: DD Gate Threshold Sweep
# ═══════════════════════════════════════════════════════════
def exp1_dd_sweep(data):
    print("\n" + "=" * 100)
    print("실험 1: DD Gate Threshold Sweep")
    print("=" * 100)

    lookbacks = [5, 10, 15, 20, 30]
    thresholds = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]

    # ── Table 1-A: 90d Return Heatmap ──
    print("\n■ 1-A: 90일 평균 수익률 히트맵 (lookback x threshold)")
    print(f"{'':>8}", end="")
    for th in thresholds:
        print(f"  {th*100:>5.0f}%", end="")
    print(f"  {'NONE':>7}")
    print("-" * 75)

    # No-gate baseline
    no_gate_vals = data['fwd_90d'].dropna()
    no_gate_ret = no_gate_vals.mean()
    no_gate_n = len(no_gate_vals)

    best_ret = -999
    best_combo = ""

    for lb in lookbacks:
        dd_col = f'dd_{lb}d'
        print(f"  {lb:>3}d  ", end="")
        for th in thresholds:
            subset = data[data[dd_col] >= th]
            vals = subset['fwd_90d'].dropna()
            if len(vals) >= 3:
                ret = vals.mean()
                if ret > best_ret:
                    best_ret = ret
                    best_combo = f"{lb}d/{th*100:.0f}%"
                print(f"  {ret:>+5.1f}%", end="")
            else:
                print(f"  {'n/a':>6}", end="")
        print(f"  {no_gate_ret:>+6.1f}%")

    print(f"\n  최적 조합: {best_combo} ({best_ret:+.1f}%)")

    # ── Table 1-B: Signal Count Heatmap ──
    print("\n■ 1-B: 신호 수 히트맵")
    print(f"{'':>8}", end="")
    for th in thresholds:
        print(f"  {th*100:>5.0f}%", end="")
    print(f"  {'NONE':>7}")
    print("-" * 75)

    for lb in lookbacks:
        dd_col = f'dd_{lb}d'
        print(f"  {lb:>3}d  ", end="")
        for th in thresholds:
            n = len(data[data[dd_col] >= th])
            print(f"  {n:>6}", end="")
        print(f"  {len(data):>7}")

    # ── Table 1-C: DD Band Analysis (lb=20) ──
    print("\n\n■ 1-C: DD 구간별 상세 (lookback=20d) — 역설 검증")
    print(f"  {'DD 구간':<18} {'N':>5} | {'30d':>8} {'60d':>8} {'90d':>8} | "
          f"{'Hit30':>6} {'Hit60':>6} {'Hit90':>6} | {'MFE60':>7} {'MAE60':>7} | {'CI_90d':>18}")
    print(f"  {'-'*110}")

    dd_bands = [(0, 0.01, '0~1%'), (0.01, 0.02, '1~2%'), (0.02, 0.03, '2~3% (차단)'),
                (0.03, 0.05, '3~5%'), (0.05, 0.10, '5~10%'),
                (0.10, 0.20, '10~20%'), (0.20, 1.0, '20%+')]

    pool_90 = data['fwd_90d'].dropna().values

    for lo, hi, label in dd_bands:
        subset = data[(data['dd_20d'] >= lo) & (data['dd_20d'] < hi)]
        if len(subset) < 2:
            continue
        n = len(subset)
        r30 = subset['fwd_30d'].dropna().mean()
        r60 = subset['fwd_60d'].dropna().mean()
        r90 = subset['fwd_90d'].dropna().mean()
        h30 = subset['hit_30d'].dropna().mean() * 100
        h60 = subset['hit_60d'].dropna().mean() * 100
        h90 = subset['hit_90d'].dropna().mean() * 100
        mfe = subset['mfe_60'].dropna().mean()
        mae = subset['mae_60'].dropna().mean()

        _, ci_lo, ci_hi = bootstrap_ci(subset['fwd_90d'].dropna().values)
        ci_str = f"[{ci_lo:+.1f},{ci_hi:+.1f}]"

        print(f"  {label:<18} {n:>5} | {r30:>+7.1f}% {r60:>+7.1f}% {r90:>+7.1f}% | "
              f"{h30:>5.1f}% {h60:>5.1f}% {h90:>5.1f}% | {mfe:>+6.1f}% {mae:>6.1f}% | {ci_str:>18}")

    # ── Table 1-D: Current vs No-Gate vs Optimal ──
    print("\n\n■ 1-D: 프로덕션 vs DD제거 vs 최적")
    configs = [
        ('Current (20d/3%)', data[data['dd_20d'] >= 0.03]),
        ('DD 제거 (0%)', data),
        ('DD 1%', data[data['dd_20d'] >= 0.01]),
        ('DD 2%', data[data['dd_20d'] >= 0.02]),
    ]

    print(f"  {'설정':<20} {'N':>5} | {'90d ret':>9} {'Hit90':>7} | {'MFE60':>8} {'MAE60':>8} | {'Bootstrap p':>12}")
    print(f"  {'-'*85}")

    for label, subset in configs:
        if len(subset) < 2:
            continue
        n = len(subset)
        r90 = subset['fwd_90d'].dropna().mean()
        h90 = subset['hit_90d'].dropna().mean() * 100
        mfe = subset['mfe_60'].dropna().mean()
        mae = subset['mae_60'].dropna().mean()
        bs = bootstrap_vs_pool(subset['fwd_90d'].dropna().values, pool_90)
        print(f"  {label:<20} {n:>5} | {r90:>+8.1f}% {h90:>6.1f}% | {mfe:>+7.1f}% {mae:>7.1f}% | p={bs['p_value']:.4f}")

    # ── Table 1-E: Per-Ticker ──
    print("\n\n■ 1-E: 종목별 — Current(3%) vs DD제거(0%)")
    print(f"  {'Ticker':<8} {'Years':>5} | {'3% N':>5} {'3% 90d':>8} {'3% Hit':>7} | "
          f"{'0% N':>5} {'0% 90d':>8} {'0% Hit':>7} | {'Delta':>7}")
    print(f"  {'-'*82}")

    for tk in sorted(data['ticker'].unique()):
        tk_data = data[data['ticker'] == tk]
        base = tk_data[tk_data['dd_20d'] >= 0.03]
        nogate = tk_data

        years = (tk_data['date'].max() - tk_data['date'].min()).days / 365.25

        b_n = len(base)
        n_n = len(nogate)
        b_r = base['fwd_90d'].dropna().mean() if b_n > 0 else np.nan
        n_r = nogate['fwd_90d'].dropna().mean() if n_n > 0 else np.nan
        b_h = base['hit_90d'].dropna().mean() * 100 if b_n > 0 else np.nan
        n_h = nogate['hit_90d'].dropna().mean() * 100 if n_n > 0 else np.nan

        delta = (n_r - b_r) if not np.isnan(b_r) and not np.isnan(n_r) else np.nan
        d_s = f"{delta:>+6.1f}%" if not np.isnan(delta) else "  N/A"

        b_r_s = f"{b_r:>+7.1f}%" if not np.isnan(b_r) else "    N/A"
        n_r_s = f"{n_r:>+7.1f}%" if not np.isnan(n_r) else "    N/A"
        b_h_s = f"{b_h:>6.1f}%" if not np.isnan(b_h) else "   N/A"
        n_h_s = f"{n_h:>6.1f}%" if not np.isnan(n_h) else "   N/A"

        print(f"  {tk:<8} {years:>4.1f}y | {b_n:>5} {b_r_s} {b_h_s} | "
              f"{n_n:>5} {n_r_s} {n_h_s} | {d_s}")


# ═══════════════════════════════════════════════════════════
# EXPERIMENT 2: Low Volume Signal Analysis
# ═══════════════════════════════════════════════════════════
def exp2_volume_anatomy(data):
    print("\n\n" + "=" * 100)
    print("실험 2: 저거래량 신호 원인 분석")
    print("=" * 100)

    vnorm_bands = [(0, 0.5, 'V<0.5 (매우조용)'),
                   (0.5, 0.7, '0.5~0.7 (조용)'),
                   (0.7, 1.0, '0.7~1.0 (평균이하)'),
                   (1.0, 1.5, '1.0~1.5 (평균)'),
                   (1.5, 2.5, '1.5~2.5 (높음)'),
                   (2.5, 99, '2.5+ (스파이크)')]

    pool_90 = data['fwd_90d'].dropna().values

    # ── Table 2-A: V_norm Breakdown ──
    print("\n■ 2-A: V_norm 구간별 전체 성과 + Bootstrap CI")
    print(f"  {'V_norm':<20} {'N':>5} | {'30d':>8} {'60d':>8} {'90d':>8} | "
          f"{'Hit90':>6} {'MFE60':>7} {'MAE60':>7} | {'CI_90d':>18} {'p_val':>7}")
    print(f"  {'-'*105}")

    for lo, hi, label in vnorm_bands:
        subset = data[(data['v_norm'] >= lo) & (data['v_norm'] < hi)]
        if len(subset) < 2:
            continue
        n = len(subset)
        r30 = subset['fwd_30d'].dropna().mean()
        r60 = subset['fwd_60d'].dropna().mean()
        r90 = subset['fwd_90d'].dropna().mean()
        h90 = subset['hit_90d'].dropna().mean() * 100
        mfe = subset['mfe_60'].dropna().mean()
        mae = subset['mae_60'].dropna().mean()

        _, ci_lo, ci_hi = bootstrap_ci(subset['fwd_90d'].dropna().values)
        bs = bootstrap_vs_pool(subset['fwd_90d'].dropna().values, pool_90)

        print(f"  {label:<20} {n:>5} | {r30:>+7.1f}% {r60:>+7.1f}% {r90:>+7.1f}% | "
              f"{h90:>5.1f}% {mfe:>+6.1f}% {mae:>6.1f}% | [{ci_lo:>+.1f},{ci_hi:>+.1f}] {bs['p_value']:>6.3f}")

    # ── Table 2-B: Sub-indicator Breakdown ──
    print("\n\n■ 2-B: V_norm × 서브지표 분해 (메커니즘 단서)")
    print(f"  {'V_norm':<20} {'N':>5} | {'S_Force':>8} {'S_Div':>8} {'Score':>7} | "
          f"{'DD_20d':>7} {'Trend20d':>9} | {'Duration':>8}")
    print(f"  {'-'*85}")

    for lo, hi, label in vnorm_bands:
        subset = data[(data['v_norm'] >= lo) & (data['v_norm'] < hi)]
        if len(subset) < 2:
            continue
        n = len(subset)
        print(f"  {label:<20} {n:>5} | "
              f"{subset['s_force'].mean():>+7.3f} {subset['s_div'].mean():>+7.3f} "
              f"{subset['score'].mean():>6.3f} | "
              f"{subset['dd_20d'].mean()*100:>6.1f}% {subset['trend_20d'].mean():>+8.1f}% | "
              f"{subset['duration'].mean():>7.1f}d")

    # ── Table 2-C: V_norm x DD Cross ──
    print("\n\n■ 2-C: V_norm × DD 교차분석 (90d 수익)")
    dd_bins = [(0, 0.03, '<3%'), (0.03, 0.10, '3~10%'), (0.10, 0.20, '10~20%'), (0.20, 1.0, '20%+')]

    print(f"  {'V_norm \\ DD':<20}", end="")
    for _, _, dl in dd_bins:
        print(f" | {dl:>14}", end="")
    print()
    print(f"  {'-'*75}")

    for vlo, vhi, vlab in vnorm_bands:
        print(f"  {vlab:<20}", end="")
        for dlo, dhi, _ in dd_bins:
            subset = data[(data['v_norm'] >= vlo) & (data['v_norm'] < vhi) &
                         (data['dd_20d'] >= dlo) & (data['dd_20d'] < dhi)]
            if len(subset) >= 3:
                r90 = subset['fwd_90d'].dropna().mean()
                h90 = subset['hit_90d'].dropna().mean() * 100
                print(f" | {r90:>+5.1f}% h{h90:.0f}% n{len(subset)}", end="")
            else:
                n = len(subset)
                print(f" | {'n=' + str(n):>14}", end="")
        print()

    # ── Table 2-D: Per-Ticker V_norm ──
    print("\n\n■ 2-D: 종목별 V_norm 분포")
    print(f"  {'Ticker':<8} {'avg_vn':>7} {'med_vn':>7} {'%<1.0':>6} | {'90d ret':>8} {'Hit90':>6}")
    print(f"  {'-'*55}")

    for tk in sorted(data['ticker'].unique()):
        tk_data = data[data['ticker'] == tk]
        if len(tk_data) < 2:
            continue
        print(f"  {tk:<8} {tk_data['v_norm'].mean():>6.2f} {tk_data['v_norm'].median():>6.2f} "
              f"{(tk_data['v_norm'] < 1.0).mean()*100:>5.1f}% | "
              f"{tk_data['fwd_90d'].dropna().mean():>+7.1f}% "
              f"{tk_data['hit_90d'].dropna().mean()*100:>5.1f}%")

    # ── Table 2-E: Market Context ──
    print("\n\n■ 2-E: 시장 맥락 비교 — 저거래량 vs 고거래량")
    low_v = data[data['v_norm'] < 1.0]
    high_v = data[data['v_norm'] >= 1.5]

    if len(low_v) >= 3 and len(high_v) >= 3:
        print(f"  {'지표':<25} {'V_norm<1.0':>12} {'V_norm>=1.5':>12} {'Delta':>10}")
        print(f"  {'-'*62}")

        comparisons = [
            ('N (신호 수)', len(low_v), len(high_v)),
            ('MA200 위 (%)', low_v['ma200_above'].mean()*100, high_v['ma200_above'].mean()*100),
            ('평균 DD_20d (%)', low_v['dd_20d'].mean()*100, high_v['dd_20d'].mean()*100),
            ('평균 Trend_20d (%)', low_v['trend_20d'].mean(), high_v['trend_20d'].mean()),
            ('평균 Score', low_v['score'].mean(), high_v['score'].mean()),
            ('평균 S_Force', low_v['s_force'].mean(), high_v['s_force'].mean()),
            ('평균 S_Div', low_v['s_div'].mean(), high_v['s_div'].mean()),
            ('신호 Duration (d)', low_v['duration'].mean(), high_v['duration'].mean()),
            ('90d 수익 (%)', low_v['fwd_90d'].dropna().mean(), high_v['fwd_90d'].dropna().mean()),
            ('90d 적중률 (%)', low_v['hit_90d'].dropna().mean()*100, high_v['hit_90d'].dropna().mean()*100),
            ('MFE_60d (%)', low_v['mfe_60'].dropna().mean(), high_v['mfe_60'].dropna().mean()),
            ('MAE_60d (%)', low_v['mae_60'].dropna().mean(), high_v['mae_60'].dropna().mean()),
        ]

        for label, v_low, v_high in comparisons:
            delta = v_high - v_low
            print(f"  {label:<25} {v_low:>11.1f} {v_high:>11.1f} {delta:>+9.1f}")

    # ── Table 2-F: Year-by-Year ──
    print("\n\n■ 2-F: 연도별 V_norm 효과 일관성")
    print(f"  {'Year':<6} | {'N_low':>5} {'ret_low':>8} | {'N_high':>6} {'ret_high':>9} | {'Delta':>8}")
    print(f"  {'-'*60}")

    for yr in sorted(data['year'].unique()):
        yr_data = data[data['year'] == yr]
        low = yr_data[yr_data['v_norm'] < 1.0]
        high = yr_data[yr_data['v_norm'] >= 1.0]

        n_l = len(low)
        n_h = len(high)
        r_l = low['fwd_90d'].dropna().mean() if n_l > 0 else np.nan
        r_h = high['fwd_90d'].dropna().mean() if n_h > 0 else np.nan
        delta = (r_l - r_h) if not np.isnan(r_l) and not np.isnan(r_h) else np.nan

        r_l_s = f"{r_l:>+7.1f}%" if not np.isnan(r_l) else "    N/A"
        r_h_s = f"{r_h:>+8.1f}%" if not np.isnan(r_h) else "     N/A"
        d_s = f"{delta:>+7.1f}%" if not np.isnan(delta) else "    N/A"

        print(f"  {yr:<6} | {n_l:>5} {r_l_s} | {n_h:>6} {r_h_s} | {d_s}")

    # ── Mechanism Summary ──
    print("\n\n■ 메커니즘 요약")
    if len(low_v) >= 3 and len(high_v) >= 3:
        dd_diff = low_v['dd_20d'].mean() - high_v['dd_20d'].mean()
        div_diff = low_v['s_div'].mean() - high_v['s_div'].mean()
        force_diff = low_v['s_force'].mean() - high_v['s_force'].mean()
        trend_diff = low_v['trend_20d'].mean() - high_v['trend_20d'].mean()

        explanations = []
        if dd_diff > 0.02:
            explanations.append(f"  1. 낙폭 동반: 저V_norm 신호는 평균 DD가 {dd_diff*100:+.1f}%p 더 큼 → 더 깊은 눌림목에서 발생")
        if div_diff > 0.05:
            explanations.append(f"  2. S_Div 강화: 저V_norm 신호의 S_Div가 {div_diff:+.3f} 더 높음 → 가격대비 거래량 괴리가 더 강함")
        if force_diff < -0.05:
            explanations.append(f"  3. S_Force 약화: 저V_norm 신호의 S_Force가 {force_diff:+.3f} 낮음 → AND-GEO에서 F가 낮으면 더 보수적 신호")
        if trend_diff < -3:
            explanations.append(f"  4. 하락 추세: 저V_norm 신호 전 20일 추세가 {trend_diff:+.1f}%p 더 하락 → 진짜 바닥에 가까움")
        if dd_diff <= 0.02 and abs(div_diff) <= 0.05:
            explanations.append("  ? 명확한 단일 원인 없음 — 복합적 효과로 추정")

        for e in explanations:
            print(e)


# ═══════════════════════════════════════════════════════════
# EXPERIMENT 3: Long Volume Surge Significance
# ═══════════════════════════════════════════════════════════
def exp3_surge_significance(data):
    print("\n\n" + "=" * 100)
    print("실험 3: 장기 거래량 증가 통계적 유의성")
    print("=" * 100)
    print(f"  총 신호: {len(data)}, 총 종목: {data['ticker'].nunique()}")

    streak_bands = [(0, 0, '0일 (없음)'),
                    (1, 4, '1~4일 (짧은)'),
                    (5, 9, '5~9일'),
                    (10, 14, '10~14일'),
                    (15, 19, '15~19일'),
                    (20, 99, '20일+')]

    pool_90 = data['fwd_90d'].dropna().values

    # ── Table 3-A: Streak Breakdown ──
    print("\n■ 3-A: V_norm>1 연속일 구간별 성과 + Bootstrap")
    print(f"  {'Streak':<16} {'N':>5} | {'30d':>8} {'60d':>8} {'90d':>8} | "
          f"{'Hit90':>6} {'MFE60':>7} {'MAE60':>7} | {'CI_90d':>18} {'p_val':>7} {'유의':>4}")
    print(f"  {'-'*115}")

    for lo, hi, label in streak_bands:
        subset = data[(data['surge_vnorm'] >= lo) & (data['surge_vnorm'] <= hi)]
        if len(subset) < 2:
            continue
        n = len(subset)
        r30 = subset['fwd_30d'].dropna().mean()
        r60 = subset['fwd_60d'].dropna().mean()
        r90 = subset['fwd_90d'].dropna().mean()
        h90 = subset['hit_90d'].dropna().mean() * 100
        mfe = subset['mfe_60'].dropna().mean()
        mae = subset['mae_60'].dropna().mean()

        _, ci_lo, ci_hi = bootstrap_ci(subset['fwd_90d'].dropna().values)
        bs = bootstrap_vs_pool(subset['fwd_90d'].dropna().values, pool_90)
        sig = "***" if bs['p_value'] < 0.01 else "**" if bs['p_value'] < 0.05 else "*" if bs['p_value'] < 0.10 else ""

        print(f"  {label:<16} {n:>5} | {r30:>+7.1f}% {r60:>+7.1f}% {r90:>+7.1f}% | "
              f"{h90:>5.1f}% {mfe:>+6.1f}% {mae:>6.1f}% | [{ci_lo:>+.1f},{ci_hi:>+.1f}] {bs['p_value']:>6.3f} {sig:>4}")

    # ── Table 3-B: Per-Ticker ──
    print("\n\n■ 3-B: 종목별 장기 streak 분포")
    print(f"  {'Ticker':<8} {'Years':>5} {'N_total':>7} {'N_10+':>6} {'N_15+':>6} | "
          f"{'ret_10+':>8} {'ret_other':>9} | {'Delta':>8}")
    print(f"  {'-'*72}")

    for tk in sorted(data['ticker'].unique()):
        tk_data = data[data['ticker'] == tk]
        years = (tk_data['date'].max() - tk_data['date'].min()).days / 365.25
        n_total = len(tk_data)
        n_10 = len(tk_data[tk_data['surge_vnorm'] >= 10])
        n_15 = len(tk_data[tk_data['surge_vnorm'] >= 15])

        long_streak = tk_data[tk_data['surge_vnorm'] >= 10]
        other = tk_data[tk_data['surge_vnorm'] < 10]

        r_long = long_streak['fwd_90d'].dropna().mean() if len(long_streak) > 0 else np.nan
        r_other = other['fwd_90d'].dropna().mean() if len(other) > 0 else np.nan
        delta = (r_long - r_other) if not np.isnan(r_long) and not np.isnan(r_other) else np.nan

        r_l_s = f"{r_long:>+7.1f}%" if not np.isnan(r_long) else "    N/A"
        r_o_s = f"{r_other:>+8.1f}%" if not np.isnan(r_other) else "     N/A"
        d_s = f"{delta:>+7.1f}%" if not np.isnan(delta) else "    N/A"

        print(f"  {tk:<8} {years:>4.1f}y {n_total:>7} {n_10:>6} {n_15:>6} | {r_l_s} {r_o_s} | {d_s}")

    # ── Table 3-C: Individual 10+ Streak Signals ──
    long_signals = data[data['surge_vnorm'] >= 10].sort_values('date')
    print(f"\n\n■ 3-C: 10일+ streak 개별 신호 리스트 (총 {len(long_signals)}개)")
    print(f"  {'Ticker':<7} {'Date':>11} {'Streak':>6} {'Score':>6} {'S_F':>6} {'S_D':>6} "
          f"{'DD20':>5} {'V_n':>5} | {'30d':>7} {'60d':>7} {'90d':>7}")
    print(f"  {'-'*90}")

    for _, r in long_signals.iterrows():
        r30 = f"{r['fwd_30d']:>+6.1f}%" if not np.isnan(r['fwd_30d']) else "   N/A"
        r60 = f"{r['fwd_60d']:>+6.1f}%" if not np.isnan(r['fwd_60d']) else "   N/A"
        r90 = f"{r['fwd_90d']:>+6.1f}%" if not np.isnan(r['fwd_90d']) else "   N/A"
        print(f"  {r['ticker']:<7} {str(r['date'])[:10]:>11} {r['surge_vnorm']:>5.0f}d "
              f"{r['score']:>5.3f} {r['s_force']:>5.3f} {r['s_div']:>5.3f} "
              f"{r['dd_20d']*100:>4.1f}% {r['v_norm']:>4.1f} | {r30} {r60} {r90}")

    # ── Table 3-D: Streak vs Characteristics ──
    print("\n\n■ 3-D: Streak 길이별 신호 특성")
    print(f"  {'구간':<16} {'Score':>6} {'S_Force':>8} {'S_Div':>7} {'DD20%':>6} {'V_norm':>7} {'MA200%':>7}")
    print(f"  {'-'*65}")

    for lo, hi, label in streak_bands:
        subset = data[(data['surge_vnorm'] >= lo) & (data['surge_vnorm'] <= hi)]
        if len(subset) < 2:
            continue
        print(f"  {label:<16} {subset['score'].mean():>5.3f} "
              f"{subset['s_force'].mean():>+7.3f} {subset['s_div'].mean():>+6.3f} "
              f"{subset['dd_20d'].mean()*100:>5.1f}% {subset['v_norm'].mean():>6.2f} "
              f"{subset['ma200_above'].mean()*100:>6.1f}%")

    # ── Table 3-E: Decade Distribution ──
    print("\n\n■ 3-E: 연대별 분포 (최근 현상인가?)")
    decades = [(1990, 1999, '1990s'), (2000, 2004, '2000-04'),
               (2005, 2009, '2005-09'), (2010, 2014, '2010-14'),
               (2015, 2019, '2015-19'), (2020, 2024, '2020-24'),
               (2025, 2030, '2025+')]

    print(f"  {'연대':<10} {'N_total':>7} {'N_10+':>6} {'pct_10+':>8} | {'ret_10+':>8} {'ret_other':>9}")
    print(f"  {'-'*55}")

    for yr_lo, yr_hi, label in decades:
        subset = data[(data['year'] >= yr_lo) & (data['year'] <= yr_hi)]
        if len(subset) == 0:
            continue
        n_total = len(subset)
        n_10 = len(subset[subset['surge_vnorm'] >= 10])
        pct = n_10 / n_total * 100 if n_total > 0 else 0

        long_s = subset[subset['surge_vnorm'] >= 10]
        other_s = subset[subset['surge_vnorm'] < 10]
        r_l = long_s['fwd_90d'].dropna().mean() if len(long_s) > 0 else np.nan
        r_o = other_s['fwd_90d'].dropna().mean() if len(other_s) > 0 else np.nan

        r_l_s = f"{r_l:>+7.1f}%" if not np.isnan(r_l) else "    N/A"
        r_o_s = f"{r_o:>+8.1f}%" if not np.isnan(r_o) else "     N/A"

        print(f"  {label:<10} {n_total:>7} {n_10:>6} {pct:>7.1f}% | {r_l_s} {r_o_s}")

    # ── Table 3-F: Alternative Definitions ──
    print("\n\n■ 3-F: 대안 정의 민감도")
    print(f"  {'정의':<30} {'N_10+':>6} {'90d ret':>8} {'Hit90':>6} {'p_val':>7}")
    print(f"  {'-'*62}")

    # Alt definitions: recompute streaks
    for alt_label, alt_col in [('V_norm>1.0 (원래)', 'surge_vnorm'),
                                ('S_Div>0 (SDdiv)', 'surge_sdiv')]:
        long_s = data[data[alt_col] >= 10]
        if len(long_s) < 2:
            print(f"  {alt_label:<30} {len(long_s):>6}   (표본 부족)")
            continue
        r90 = long_s['fwd_90d'].dropna().mean()
        h90 = long_s['hit_90d'].dropna().mean() * 100
        bs = bootstrap_vs_pool(long_s['fwd_90d'].dropna().values, pool_90)
        print(f"  {alt_label:<30} {len(long_s):>6} {r90:>+7.1f}% {h90:>5.1f}% {bs['p_value']:>6.3f}")


# ═══════════════════════════════════════════════════════════
# CROSS-REFERENCE ANALYSIS
# ═══════════════════════════════════════════════════════════
def cross_reference(data):
    print("\n\n" + "=" * 100)
    print("교차 분석 (Cross-Reference)")
    print("=" * 100)

    # ── X-A: DD x V_norm 2D ──
    print("\n■ X-A: DD × V_norm 2D 히트맵 (90d 수익 / N)")
    dd_bins = [(0, 0.03, '<3%'), (0.03, 0.10, '3~10%'), (0.10, 0.20, '10~20%'), (0.20, 1.0, '20%+')]
    vn_bins = [(0, 1.0, 'V<1.0'), (1.0, 1.5, '1.0~1.5'), (1.5, 99, 'V>=1.5')]

    print(f"  {'DD \\ V_norm':<12}", end="")
    for _, _, vl in vn_bins:
        print(f" | {vl:>16}", end="")
    print()
    print(f"  {'-'*65}")

    for dlo, dhi, dl in dd_bins:
        print(f"  {dl:<12}", end="")
        for vlo, vhi, _ in vn_bins:
            subset = data[(data['dd_20d'] >= dlo) & (data['dd_20d'] < dhi) &
                         (data['v_norm'] >= vlo) & (data['v_norm'] < vhi)]
            if len(subset) >= 3:
                r90 = subset['fwd_90d'].dropna().mean()
                h90 = subset['hit_90d'].dropna().mean() * 100
                print(f" | {r90:>+5.1f}% h{h90:.0f}% n{len(subset)}", end="")
            else:
                print(f" |   n={len(subset):>12}", end="")
        print()

    # ── X-B: Streak vs V_norm ──
    print("\n\n■ X-B: Streak vs V_norm — 긴 surge 후 V_norm이 떨어지나?")
    streak_bins = [(0, 0, '0일'), (1, 4, '1~4일'), (5, 9, '5~9일'), (10, 99, '10일+')]

    print(f"  {'Streak':<12} {'avg_vnorm':>9} {'med_vnorm':>10} {'%_vnorm<1':>10}")
    print(f"  {'-'*45}")

    for lo, hi, label in streak_bins:
        subset = data[(data['surge_vnorm'] >= lo) & (data['surge_vnorm'] <= hi)]
        if len(subset) < 2:
            continue
        print(f"  {label:<12} {subset['v_norm'].mean():>8.2f} "
              f"{subset['v_norm'].median():>9.2f} "
              f"{(subset['v_norm'] < 1.0).mean()*100:>9.1f}%")

    # ── X-C: 0-3% DD 신호 특성 ──
    print("\n\n■ X-C: 0-3% DD 신호의 V_norm / Streak 특성")
    blocked = data[data['dd_20d'] < 0.03]
    passed = data[data['dd_20d'] >= 0.03]

    if len(blocked) >= 2 and len(passed) >= 2:
        print(f"  {'지표':<25} {'DD<3% (차단)':>12} {'DD>=3% (통과)':>14} {'Delta':>8}")
        print(f"  {'-'*62}")
        comparisons = [
            ('N', len(blocked), len(passed)),
            ('avg V_norm', blocked['v_norm'].mean(), passed['v_norm'].mean()),
            ('avg Streak (vnorm)', blocked['surge_vnorm'].mean(), passed['surge_vnorm'].mean()),
            ('avg Score', blocked['score'].mean(), passed['score'].mean()),
            ('avg S_Force', blocked['s_force'].mean(), passed['s_force'].mean()),
            ('avg S_Div', blocked['s_div'].mean(), passed['s_div'].mean()),
            ('MA200 위 (%)', blocked['ma200_above'].mean()*100, passed['ma200_above'].mean()*100),
            ('90d ret (%)', blocked['fwd_90d'].dropna().mean(), passed['fwd_90d'].dropna().mean()),
            ('Hit90 (%)', blocked['hit_90d'].dropna().mean()*100, passed['hit_90d'].dropna().mean()*100),
        ]
        for label, v1, v2 in comparisons:
            delta = v2 - v1
            print(f"  {label:<25} {v1:>11.2f} {v2:>13.2f} {delta:>+7.2f}")

    # ── X-D: Triple-Factor Filter ──
    print("\n\n■ X-D: 조합 필터 테스트")
    pool_90 = data['fwd_90d'].dropna().values

    filters = [
        ('Baseline (DD>=3%)', data[data['dd_20d'] >= 0.03]),
        ('DD>=10%', data[data['dd_20d'] >= 0.10]),
        ('V_norm<1.0', data[data['v_norm'] < 1.0]),
        ('Streak>=10d', data[data['surge_vnorm'] >= 10]),
        ('DD>=10% AND V<1.0', data[(data['dd_20d'] >= 0.10) & (data['v_norm'] < 1.0)]),
        ('DD>=10% AND Streak>=10', data[(data['dd_20d'] >= 0.10) & (data['surge_vnorm'] >= 10)]),
        ('V<1.0 AND Streak>=10', data[(data['v_norm'] < 1.0) & (data['surge_vnorm'] >= 10)]),
        ('ALL THREE', data[(data['dd_20d'] >= 0.10) & (data['v_norm'] < 1.0) & (data['surge_vnorm'] >= 10)]),
        ('DD제거 (모든신호)', data),
    ]

    print(f"  {'필터':<28} {'N':>5} | {'90d':>8} {'Hit90':>7} {'MFE60':>8} | {'CI_90d':>18} {'p_val':>7}")
    print(f"  {'-'*90}")

    for label, subset in filters:
        if len(subset) < 2:
            print(f"  {label:<28} {len(subset):>5} | (표본 부족)")
            continue
        n = len(subset)
        r90 = subset['fwd_90d'].dropna().mean()
        h90 = subset['hit_90d'].dropna().mean() * 100
        mfe = subset['mfe_60'].dropna().mean()
        _, ci_lo, ci_hi = bootstrap_ci(subset['fwd_90d'].dropna().values)
        bs = bootstrap_vs_pool(subset['fwd_90d'].dropna().values, pool_90)
        print(f"  {label:<28} {n:>5} | {r90:>+7.1f}% {h90:>6.1f}% {mfe:>+7.1f}% | "
              f"[{ci_lo:>+.1f},{ci_hi:>+.1f}] {bs['p_value']:>6.3f}")

    # ── X-E: Final Summary ──
    print("\n\n■ X-E: 최종 결론 요약")
    print(f"  {'발견':<35} {'확인':>6} {'N':>5} {'효과크기':>10} {'p-value':>8} {'실용적 함의':>20}")
    print(f"  {'-'*95}")

    # F1: DD paradox
    band_03 = data[(data['dd_20d'] >= 0) & (data['dd_20d'] < 0.03)]
    band_510 = data[(data['dd_20d'] >= 0.05) & (data['dd_20d'] < 0.10)]
    if len(band_03) >= 3 and len(band_510) >= 3:
        r03 = band_03['fwd_90d'].dropna().mean()
        r510 = band_510['fwd_90d'].dropna().mean()
        diff1 = r03 - r510
        confirmed1 = "YES" if diff1 > 0 else "NO"
        print(f"  {'F1: DD 0-3% > 5-10%':<35} {confirmed1:>6} "
              f"{len(band_03)+len(band_510):>5} {diff1:>+9.1f}% {'N/A':>8} "
              f"{'TH 하향 검토':>20}")

    # F2: Low volume
    low_v = data[data['v_norm'] < 1.0]
    high_v = data[data['v_norm'] >= 1.0]
    if len(low_v) >= 3 and len(high_v) >= 3:
        r_low = low_v['fwd_90d'].dropna().mean()
        r_high = high_v['fwd_90d'].dropna().mean()
        diff2 = r_low - r_high
        bs2 = bootstrap_vs_pool(low_v['fwd_90d'].dropna().values, pool_90)
        confirmed2 = "YES" if diff2 > 3 and bs2['p_value'] < 0.1 else "WEAK" if diff2 > 0 else "NO"
        print(f"  {'F2: V_norm<1 우수':<35} {confirmed2:>6} "
              f"{len(low_v):>5} {diff2:>+9.1f}% {bs2['p_value']:>7.3f} "
              f"{'V_norm 필터 불필요':>20}")

    # F3: Long streak
    long_s = data[data['surge_vnorm'] >= 10]
    other_s = data[data['surge_vnorm'] < 10]
    if len(long_s) >= 3:
        r_long = long_s['fwd_90d'].dropna().mean()
        r_other = other_s['fwd_90d'].dropna().mean()
        diff3 = r_long - r_other
        bs3 = bootstrap_vs_pool(long_s['fwd_90d'].dropna().values, pool_90)
        confirmed3 = "YES" if bs3['p_value'] < 0.05 else "WEAK" if bs3['p_value'] < 0.10 else "NO"
        print(f"  {'F3: Streak 10+ 프리미엄':<35} {confirmed3:>6} "
              f"{len(long_s):>5} {diff3:>+9.1f}% {bs3['p_value']:>7.3f} "
              f"{'Streak 추가확인?':>20}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    import time
    t0 = time.time()

    print("V4 Three Findings Deep Dive")
    print("=" * 60)

    # ── Collect Data ──
    all_data = []

    # Long history + Extra largecap (period='max')
    max_tickers = {**LONG_HISTORY, **EXTRA_LARGECAP}
    print(f"\n[1/2] Long-history + Extra LargeCap ({len(max_tickers)} tickers, period='max')")
    for ticker, sector in max_tickers.items():
        print(f"  {ticker}...", end=" ", flush=True)
        try:
            df = download_max(ticker)
            if df is not None:
                years = len(df) / 252
                result = collect_raw_signals(ticker, df, sector)
                if result is not None and len(result) > 0:
                    result['pool'] = 'max'
                    all_data.append(result)
                    print(f"OK ({len(df)} bars, {years:.1f}y, {len(result)} signals)")
                else:
                    print(f"OK ({len(df)} bars) but no signals")
            else:
                print("SKIP (no data)")
        except Exception as e:
            print(f"ERROR: {e}")

    # Full watchlist (period='5y', skip already downloaded)
    already = set(max_tickers.keys())
    remaining = {k: v for k, v in FULL_WATCHLIST.items() if k not in already}
    print(f"\n[2/2] Remaining watchlist ({len(remaining)} tickers, period='5y')")
    for ticker, sector in remaining.items():
        print(f"  {ticker}...", end=" ", flush=True)
        try:
            df = download_5y(ticker)
            if df is not None:
                result = collect_raw_signals(ticker, df, sector)
                if result is not None and len(result) > 0:
                    result['pool'] = '5y'
                    all_data.append(result)
                    print(f"OK ({len(df)} bars, {len(result)} signals)")
                else:
                    print(f"OK ({len(df)} bars) but no signals")
            else:
                print("SKIP")
        except Exception as e:
            print(f"ERROR: {e}")

    if not all_data:
        print("\nNo data collected. Exiting.")
        sys.exit(1)

    data = pd.concat(all_data, ignore_index=True)
    elapsed_dl = time.time() - t0
    print(f"\n총 {len(data)} 신호 수집 완료 ({data['ticker'].nunique()} 종목, {elapsed_dl:.1f}초)")

    # Max-only data for Experiment 3
    max_data = data[data['pool'] == 'max']
    print(f"  Max-history 풀: {len(max_data)} 신호 ({max_data['ticker'].nunique()} 종목)")

    # ── Run Experiments ──
    exp1_dd_sweep(data)
    exp2_volume_anatomy(data)
    exp3_surge_significance(max_data)
    cross_reference(data)

    elapsed = time.time() - t0
    print(f"\n\n전체 실행 시간: {elapsed:.1f}초")
