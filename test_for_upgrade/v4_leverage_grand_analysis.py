"""
V4 Leverage Grand Analysis: 1x / 2x / 3x 통합 비교 + 새로운 발견
=================================================================
모든 레버리지 배수를 한번에 돌려서:
1. 최적 레버리지 배수 탐색 (1x ~ 4x)
2. Risk-adjusted edge (edge / worst-year-loss)
3. V4 신호 정확도와 레버리지 효과의 관계
4. 상승장/하락장 분리 분석
5. 최근 3년 vs 전체 기간 비교
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
    calc_v4_score, detect_signal_events, build_price_filter,
    smooth_earnings_volume,
)

# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════
TICKERS = {
    'TSLA': 'Tech', 'NVDA': 'Tech', 'AVGO': 'Tech',
    'AMZN': 'Tech', 'GOOGL': 'Tech', 'PLTR': 'Tech',
    'COIN': 'Fintech', 'HOOD': 'Fintech',
    'HIMS': 'Growth', 'RKLB': 'Growth', 'JOBY': 'Growth',
    'IONQ': 'Quantum',
    'QQQ': 'Benchmark', 'VOO': 'Benchmark',
}

MONTHLY_DEPOSIT = 500.0
SIGNAL_BUY_PCT = 0.50
EXPENSE_RATIO_DAILY = 0.0095 / 252
LEVERAGE_LEVELS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

V4_W = 20; SIGNAL_TH = 0.15; COOLDOWN = 5
ER_Q = 66; ATR_Q = 55; LOOKBACK = 252
DIVGATE = 3; CONFIRM = 3


def download_max(ticker):
    t = yf.Ticker(ticker)
    df = t.history(period='max', auto_adjust=True)
    if df is None or len(df) == 0:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def build_synthetic_lev(close, leverage):
    daily_ret = np.diff(close) / close[:-1]
    lev_price = np.zeros(len(close))
    lev_price[0] = close[0]
    for i in range(1, len(close)):
        lev_ret = leverage * daily_ret[i - 1] - EXPENSE_RATIO_DAILY
        lev_price[i] = lev_price[i - 1] * (1 + lev_ret)
        if lev_price[i] < 0.001:
            lev_price[i] = 0.001
    return lev_price


def get_buy_signal_indices(df, ticker):
    try:
        df_s = smooth_earnings_volume(df, ticker=ticker)
    except Exception:
        df_s = df.copy()
    score = calc_v4_score(df_s, w=V4_W, divgate_days=DIVGATE)
    events = detect_signal_events(score, th=SIGNAL_TH, cooldown=COOLDOWN)
    pf = build_price_filter(df_s, er_q=ER_Q, atr_q=ATR_Q, lookback=LOOKBACK)

    n = len(df_s)
    buys = set()
    for ev in events:
        if ev['type'] != 'bottom':
            continue
        if not pf(ev['peak_idx']):
            continue
        dur = ev['end_idx'] - ev['start_idx'] + 1
        ci = ev['start_idx'] + CONFIRM - 1
        if ci <= ev['end_idx'] and dur >= CONFIRM and ci < n:
            buys.add(ci)
    return buys


def calc_forward_hit_rate(close, buy_indices, horizon=60):
    """V4 signal accuracy: % of signals with positive forward return"""
    hits = 0; total = 0
    n = len(close)
    for idx in buy_indices:
        if idx + horizon < n:
            total += 1
            if close[idx + horizon] > close[idx]:
                hits += 1
    return (hits / total * 100) if total > 0 else 0


def simulate(close, close_lev, buy_signals, dates):
    n = len(close)
    month_periods = pd.Series(dates).dt.to_period('M')
    month_map = {}
    for i, mp in enumerate(month_periods):
        key = str(mp)
        if key not in month_map:
            month_map[key] = {'first': i, 'last': i, 'period': mp}
        else:
            month_map[key]['last'] = i
    sorted_months = sorted(month_map.keys())

    cash_a = 0.0; sh_1x_a = 0.0; sh_lx_a = 0.0; sig_count = 0
    cash_b = 0.0; sh_1x_b = 0.0
    total_dep = 0.0
    yr_data = {}; prev_yr = None

    def pf_a(idx):
        return sh_1x_a * close[idx] + sh_lx_a * close_lev[idx] + cash_a
    def pf_b(idx):
        return sh_1x_b * close[idx] + cash_b

    for mk in sorted_months:
        mm = month_map[mk]
        fi, li = mm['first'], mm['last']
        yr = mm['period'].year

        if yr != prev_yr:
            if prev_yr is not None:
                ref = fi - 1 if fi > 0 else fi
                yr_data[prev_yr]['end_a'] = pf_a(ref)
                yr_data[prev_yr]['end_b'] = pf_b(ref)
            yr_data[yr] = {
                'start_a': pf_a(fi), 'start_b': pf_b(fi),
                'deposits': 0.0, 'end_a': 0, 'end_b': 0, 'sigs': 0,
            }
            prev_yr = yr

        cash_a += MONTHLY_DEPOSIT; cash_b += MONTHLY_DEPOSIT
        total_dep += MONTHLY_DEPOSIT
        yr_data[yr]['deposits'] += MONTHLY_DEPOSIT

        for day_idx in range(fi, li + 1):
            if day_idx in buy_signals and cash_a > 1.0:
                amt = cash_a * SIGNAL_BUY_PCT
                sh_lx_a += amt / close_lev[day_idx]
                cash_a -= amt
                sig_count += 1
                yr_data[yr]['sigs'] += 1

        if cash_a > 1.0:
            sh_1x_a += cash_a / close[li]; cash_a = 0.0
        if cash_b > 1.0:
            sh_1x_b += cash_b / close[li]; cash_b = 0.0

    yr_data[prev_yr]['end_a'] = pf_a(n - 1)
    yr_data[prev_yr]['end_b'] = pf_b(n - 1)

    final_a = pf_a(n - 1); final_b = pf_b(n - 1)

    yr_results = []
    for yr in sorted(yr_data.keys()):
        yd = yr_data[yr]
        da = yd['start_a'] + yd['deposits'] * 0.5
        db = yd['start_b'] + yd['deposits'] * 0.5
        ra = ((yd['end_a'] - yd['start_a'] - yd['deposits']) / da * 100) if da > 10 else 0
        rb = ((yd['end_b'] - yd['start_b'] - yd['deposits']) / db * 100) if db > 10 else 0
        yr_results.append({'yr': yr, 'ret_a': ra, 'ret_b': rb, 'diff': ra - rb, 'sigs': yd['sigs']})

    avg_a = np.mean([r['ret_a'] for r in yr_results])
    avg_b = np.mean([r['ret_b'] for r in yr_results])

    # Lev position weight
    val_lx = sh_lx_a * close_lev[-1]
    pct_lx = val_lx / final_a * 100 if final_a > 0 else 0

    # Worst year diff
    worst_diff = min(r['diff'] for r in yr_results)
    worst_yr = min(yr_results, key=lambda x: x['diff'])

    # Recent 3 years
    recent = [r for r in yr_results if r['yr'] >= 2023]
    recent_edge = np.mean([r['diff'] for r in recent]) if recent else 0

    # Bull/bear split
    bull_yrs = [r for r in yr_results if r['ret_b'] > 5]
    bear_yrs = [r for r in yr_results if r['ret_b'] < -5]
    flat_yrs = [r for r in yr_results if -5 <= r['ret_b'] <= 5]

    return {
        'final_a': final_a, 'final_b': final_b,
        'avg_ann_a': avg_a, 'avg_ann_b': avg_b,
        'edge': avg_a - avg_b,
        'signals': sig_count,
        'total_dep': total_dep,
        'yr_results': yr_results,
        'wins': sum(1 for r in yr_results if r['diff'] > 0.5),
        'losses': sum(1 for r in yr_results if r['diff'] < -0.5),
        'worst_diff': worst_diff,
        'worst_yr': worst_yr['yr'],
        'pct_lev': pct_lx,
        'recent_edge': recent_edge,
        'bull_edge': np.mean([r['diff'] for r in bull_yrs]) if bull_yrs else 0,
        'bear_edge': np.mean([r['diff'] for r in bear_yrs]) if bear_yrs else 0,
        'flat_edge': np.mean([r['diff'] for r in flat_yrs]) if flat_yrs else 0,
        'n_bull': len(bull_yrs), 'n_bear': len(bear_yrs), 'n_flat': len(flat_yrs),
    }


# ═══════════════════════════════════════════════════════════
# Main: Run all leverage levels for all tickers
# ═══════════════════════════════════════════════════════════
print("=" * 120)
print("  V4 LEVERAGE GRAND ANALYSIS: 1x ~ 4x Comprehensive Comparison")
print("=" * 120)
print()

# master[ticker] = { lev: result_dict, ... }
master = {}
ticker_meta = {}

for tk, sector in TICKERS.items():
    print(f"  {tk}...", end=" ", flush=True)
    df = download_max(tk)
    if df is None or len(df) < 300:
        print("SKIP")
        continue

    close = df['Close'].values
    dates = df.index
    n = len(close)

    buy_signals = get_buy_signal_indices(df, tk)

    # Volatility
    daily_rets = np.diff(close) / close[:-1]
    ann_vol = np.std(daily_rets) * np.sqrt(252) * 100

    # V4 hit rate (60d forward)
    hit_rate_60 = calc_forward_hit_rate(close, buy_signals, 60)
    hit_rate_90 = calc_forward_hit_rate(close, buy_signals, 90)

    # Avg forward return at signal
    fwd_rets_60 = []
    for idx in buy_signals:
        if idx + 60 < n:
            fwd_rets_60.append((close[idx + 60] / close[idx] - 1) * 100)
    avg_fwd_60 = np.mean(fwd_rets_60) if fwd_rets_60 else 0

    ticker_meta[tk] = {
        'sector': sector, 'ann_vol': ann_vol,
        'hit_rate_60': hit_rate_60, 'hit_rate_90': hit_rate_90,
        'avg_fwd_60': avg_fwd_60,
        'n_signals': len(buy_signals),
        'n_years': (dates[-1] - dates[0]).days / 365.25,
    }

    master[tk] = {}
    edges = []
    for lev in LEVERAGE_LEVELS:
        if lev == 1.0:
            close_lev = close.copy()
        else:
            close_lev = build_synthetic_lev(close, lev)
        res = simulate(close, close_lev, buy_signals, dates)
        master[tk][lev] = res
        edges.append(f"{lev:.1f}x={res['edge']:+.1f}")

    print(f"vol={ann_vol:.0f}% hit={hit_rate_60:.0f}% " + "  ".join(edges))


# ═══════════════════════════════════════════════════════════
#  1. OPTIMAL LEVERAGE PER TICKER
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 120}")
print(f"  [1] OPTIMAL LEVERAGE: Edge by Leverage Level")
print(f"{'=' * 120}")

header = f"  {'Ticker':<7s} {'Vol':>5s} {'Hit%':>5s}"
for lev in LEVERAGE_LEVELS:
    header += f" {f'{lev:.1f}x Edge':>10s}"
header += f"  {'Optimal':>8s} {'Risk-adj':>9s}"
print(header)
print(f"  {'=' * 118}")

optimal_data = []
for tk in master:
    m = ticker_meta[tk]
    line = f"  {tk:<7s} {m['ann_vol']:>4.0f}% {m['hit_rate_60']:>4.0f}%"

    best_lev = 1.0; best_edge = -999
    best_risk_adj = -999; best_risk_lev = 1.0

    for lev in LEVERAGE_LEVELS:
        edge = master[tk][lev]['edge']
        worst = master[tk][lev]['worst_diff']
        line += f" {edge:>+9.2f}%p"

        if edge > best_edge:
            best_edge = edge; best_lev = lev

        # Risk-adjusted: edge / abs(worst_diff), higher = better
        risk_adj = edge / abs(worst) if abs(worst) > 0.1 else edge
        if risk_adj > best_risk_adj:
            best_risk_adj = risk_adj; best_risk_lev = lev

    line += f"  {best_lev:>6.1f}x  {best_risk_lev:>6.1f}x"
    print(line)

    optimal_data.append({
        'ticker': tk, 'best_lev': best_lev, 'best_risk_lev': best_risk_lev,
        'best_edge': best_edge, 'best_risk_adj': best_risk_adj,
        'vol': m['ann_vol'], 'hit': m['hit_rate_60'],
    })


# ═══════════════════════════════════════════════════════════
#  2. RISK ANALYSIS: Worst Year by Leverage
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 120}")
print(f"  [2] RISK: Worst Year Extra Loss by Leverage Level")
print(f"{'=' * 120}")

header = f"  {'Ticker':<7s}"
for lev in LEVERAGE_LEVELS:
    header += f" {f'{lev:.1f}x Worst':>12s}"
print(header)
print(f"  {'=' * 85}")

for tk in master:
    line = f"  {tk:<7s}"
    for lev in LEVERAGE_LEVELS:
        w = master[tk][lev]['worst_diff']
        line += f" {w:>+11.2f}%p"
    print(line)


# ═══════════════════════════════════════════════════════════
#  3. MARGINAL RETURN: Each +1x leverage increment
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 120}")
print(f"  [3] MARGINAL RETURN: Additional Edge per +1x Leverage Increment")
print(f"      (Diminishing returns? Or accelerating?)")
print(f"{'=' * 120}")

print(f"  {'Ticker':<7s} {'1x->2x':>10s} {'2x->3x':>10s} {'3x->4x':>10s} {'Pattern':>15s}")
print(f"  {'=' * 60}")

patterns = {'diminishing': 0, 'accelerating': 0, 'peaked_2x': 0, 'peaked_3x': 0, 'negative': 0}

for tk in master:
    e1 = master[tk][1.0]['edge']
    e2 = master[tk][2.0]['edge']
    e3 = master[tk][3.0]['edge']
    e4 = master[tk][4.0]['edge']

    m12 = e2 - e1  # marginal 1x->2x
    m23 = e3 - e2  # marginal 2x->3x
    m34 = e4 - e3  # marginal 3x->4x

    if m12 > 0 and m23 > 0 and m34 > 0:
        if m23 < m12 and m34 < m23:
            pat = "DIMINISHING"
            patterns['diminishing'] += 1
        elif m23 > m12:
            pat = "ACCELERATING"
            patterns['accelerating'] += 1
        else:
            pat = "MIXED_UP"
            patterns['diminishing'] += 1
    elif m12 > 0 and m23 > 0 and m34 <= 0:
        pat = "PEAK@3x"
        patterns['peaked_3x'] += 1
    elif m12 > 0 and m23 <= 0:
        pat = "PEAK@2x"
        patterns['peaked_2x'] += 1
    elif m12 <= 0:
        pat = "NO BENEFIT"
        patterns['negative'] += 1
    else:
        pat = "OTHER"

    print(f"  {tk:<7s} {m12:>+9.2f}%p {m23:>+9.2f}%p {m34:>+9.2f}%p {pat:>15s}")

print(f"\n  Pattern distribution:")
for p, c in sorted(patterns.items(), key=lambda x: -x[1]):
    if c > 0:
        print(f"    {p:<15s}: {c}/{len(master)} tickers")


# ═══════════════════════════════════════════════════════════
#  4. BULL vs BEAR MARKET: When does leverage help?
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 120}")
print(f"  [4] BULL vs BEAR MARKET: Edge in Different Market Conditions (2x leverage)")
print(f"      Bull: DCA return > +5%  |  Bear: DCA return < -5%  |  Flat: -5% ~ +5%")
print(f"{'=' * 120}")

print(f"  {'Ticker':<7s} {'Bull Edge':>10s} {'(N)':>4s} {'Bear Edge':>10s} {'(N)':>4s} {'Flat Edge':>10s} {'(N)':>4s} {'Insight':>20s}")
print(f"  {'=' * 80}")

bull_edges_all = []; bear_edges_all = []; flat_edges_all = []

for tk in master:
    r = master[tk][2.0]
    be = r['bull_edge']; bre = r['bear_edge']; fe = r['flat_edge']
    nb = r['n_bull']; nbr = r['n_bear']; nf = r['n_flat']

    if nb > 0: bull_edges_all.append(be)
    if nbr > 0: bear_edges_all.append(bre)
    if nf > 0: flat_edges_all.append(fe)

    # Insight
    if be > 3 and bre > 0:
        insight = "ALL-WEATHER"
    elif be > 3 and bre < -3:
        insight = "BULL-ONLY"
    elif be < 0 and bre > 0:
        insight = "CONTRARIAN"
    elif be > 0 and bre < 0:
        insight = "TREND-FOLLOW"
    else:
        insight = "MIXED"

    print(f"  {tk:<7s} {be:>+9.2f}%p {nb:>3d}  {bre:>+9.2f}%p {nbr:>3d}  {fe:>+9.2f}%p {nf:>3d}  {insight:>20s}")

print(f"  {'=' * 80}")
avg_bull = np.mean(bull_edges_all) if bull_edges_all else 0
avg_bear = np.mean(bear_edges_all) if bear_edges_all else 0
avg_flat = np.mean(flat_edges_all) if flat_edges_all else 0
print(f"  {'AVG':<7s} {avg_bull:>+9.2f}%p      {avg_bear:>+9.2f}%p      {avg_flat:>+9.2f}%p")


# ═══════════════════════════════════════════════════════════
#  5. V4 ACCURACY vs LEVERAGE BENEFIT
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 120}")
print(f"  [5] V4 SIGNAL ACCURACY vs LEVERAGE BENEFIT")
print(f"      Does better signal accuracy = more leverage benefit?")
print(f"{'=' * 120}")

print(f"  {'Ticker':<7s} {'Hit60':>6s} {'Hit90':>6s} {'AvgFwd60':>9s} {'1x Edge':>9s} {'2x Edge':>9s} {'3x Edge':>9s} {'Lev Mult':>9s}")
print(f"  {'=' * 75}")

hit_rates = []
lev_multipliers = []

for tk in sorted(master.keys(), key=lambda x: -ticker_meta[x]['hit_rate_60']):
    m = ticker_meta[tk]
    e1 = master[tk][1.0]['edge']
    e2 = master[tk][2.0]['edge']
    e3 = master[tk][3.0]['edge']
    # Leverage multiplier: how much does 2x edge exceed 1x edge?
    lev_mult = e2 / e1 if abs(e1) > 0.1 else 0

    hit_rates.append(m['hit_rate_60'])
    lev_multipliers.append(e2)

    print(f"  {tk:<7s} {m['hit_rate_60']:>5.0f}% {m['hit_rate_90']:>5.0f}% {m['avg_fwd_60']:>+8.2f}% "
          f"{e1:>+8.2f}%p {e2:>+8.2f}%p {e3:>+8.2f}%p {lev_mult:>8.2f}x")

# Correlation
if len(hit_rates) > 2:
    corr_hit_edge = np.corrcoef(hit_rates, lev_multipliers)[0, 1]
    print(f"\n  Correlation (Hit Rate 60d vs 2x Edge): {corr_hit_edge:.3f}")

    vols = [ticker_meta[tk]['ann_vol'] for tk in master]
    fwds = [ticker_meta[tk]['avg_fwd_60'] for tk in master]
    edges_2x = [master[tk][2.0]['edge'] for tk in master]

    corr_vol = np.corrcoef(vols, edges_2x)[0, 1]
    corr_fwd = np.corrcoef(fwds, edges_2x)[0, 1]
    print(f"  Correlation (Volatility vs 2x Edge):   {corr_vol:.3f}")
    print(f"  Correlation (Avg Fwd Return vs 2x Edge): {corr_fwd:.3f}")


# ═══════════════════════════════════════════════════════════
#  6. RECENT PERFORMANCE: 2023-2026 vs Full Period
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 120}")
print(f"  [6] RECENT vs FULL: 2023-2026 Edge vs Full Period Edge (2x leverage)")
print(f"{'=' * 120}")

print(f"  {'Ticker':<7s} {'Full Edge':>10s} {'2023-26':>10s} {'Improving?':>12s}")
print(f"  {'=' * 50}")

improving = 0; degrading = 0

for tk in master:
    full = master[tk][2.0]['edge']
    recent = master[tk][2.0]['recent_edge']
    trend = "BETTER" if recent > full + 1 else ("WORSE" if recent < full - 1 else "STABLE")
    if trend == "BETTER": improving += 1
    elif trend == "WORSE": degrading += 1

    print(f"  {tk:<7s} {full:>+9.2f}%p {recent:>+9.2f}%p {trend:>12s}")

print(f"\n  Improving: {improving}  Stable: {len(master)-improving-degrading}  Degrading: {degrading}")


# ═══════════════════════════════════════════════════════════
#  7. PORTFOLIO WEIGHT EVOLUTION
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 120}")
print(f"  [7] LEVERAGE POSITION WEIGHT in Final Portfolio")
print(f"{'=' * 120}")

print(f"  {'Ticker':<7s} {'1.5x Wt':>8s} {'2x Wt':>8s} {'3x Wt':>8s} {'4x Wt':>8s}")
print(f"  {'=' * 45}")

for tk in sorted(master.keys(), key=lambda x: -master[x][2.0]['pct_lev']):
    line = f"  {tk:<7s}"
    for lev in [1.5, 2.0, 3.0, 4.0]:
        pct = master[tk][lev]['pct_lev']
        line += f" {pct:>7.1f}%"
    print(line)


# ═══════════════════════════════════════════════════════════
#  8. EFFICIENCY FRONTIER: Edge vs Max Drawdown
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 120}")
print(f"  [8] EFFICIENCY: Edge / |Worst Year Loss| (higher = better risk-adjusted)")
print(f"{'=' * 120}")

header = f"  {'Ticker':<7s}"
for lev in LEVERAGE_LEVELS:
    header += f" {f'{lev:.1f}x Eff':>10s}"
header += f"  {'Best Eff @':>10s}"
print(header)
print(f"  {'=' * 85}")

eff_best = {}
for tk in master:
    line = f"  {tk:<7s}"
    best_eff = -999; best_eff_lev = 1.0
    for lev in LEVERAGE_LEVELS:
        edge = master[tk][lev]['edge']
        worst = master[tk][lev]['worst_diff']
        eff = edge / abs(worst) if abs(worst) > 0.1 else 0
        line += f" {eff:>+9.3f} "

        if eff > best_eff:
            best_eff = eff; best_eff_lev = lev

    line += f"  {best_eff_lev:>6.1f}x"
    print(line)
    eff_best[tk] = best_eff_lev

# Distribution
from collections import Counter
eff_dist = Counter(eff_best.values())
print(f"\n  Most efficient leverage distribution:")
for lev, cnt in sorted(eff_dist.items()):
    bar = "#" * (cnt * 4)
    print(f"    {lev:.1f}x: {cnt:>2d} tickers {bar}")


# ═══════════════════════════════════════════════════════════
#  GRAND SUMMARY
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 120}")
print(f"  GRAND SUMMARY: KEY FINDINGS")
print(f"{'=' * 120}")

# Average edge by leverage
print(f"\n  [A] Average Edge by Leverage Level (across {len(master)} tickers):")
print(f"  {'Leverage':>10s} {'Avg Edge':>10s} {'Median':>10s} {'Positive':>10s} {'Avg W/L':>10s}")
print(f"  {'=' * 55}")

for lev in LEVERAGE_LEVELS:
    edges = [master[tk][lev]['edge'] for tk in master]
    wins = sum(master[tk][lev]['wins'] for tk in master)
    losses = sum(master[tk][lev]['losses'] for tk in master)
    n_pos = sum(1 for e in edges if e > 0)
    print(f"  {lev:>8.1f}x {np.mean(edges):>+9.2f}%p {np.median(edges):>+9.2f}%p "
          f"{n_pos:>5d}/{len(master)} {wins:>4d}W/{losses}L")

# Marginal returns
print(f"\n  [B] Marginal Return per +1x Leverage (average across tickers):")
prev_avg = np.mean([master[tk][1.0]['edge'] for tk in master])
for lev in LEVERAGE_LEVELS[1:]:
    curr_avg = np.mean([master[tk][lev]['edge'] for tk in master])
    marginal = curr_avg - prev_avg
    print(f"    {LEVERAGE_LEVELS[LEVERAGE_LEVELS.index(lev)-1]:.1f}x -> {lev:.1f}x: {marginal:>+.2f}%p")
    prev_avg = curr_avg

# Market condition
print(f"\n  [C] Average Edge by Market Condition (2x leverage):")
print(f"    Bull market (DCA > +5%):  {avg_bull:>+.2f}%p")
print(f"    Bear market (DCA < -5%):  {avg_bear:>+.2f}%p")
print(f"    Flat market (-5%~+5%):    {avg_flat:>+.2f}%p")

# Correlations summary
print(f"\n  [D] What Predicts Leverage Benefit?")
if len(hit_rates) > 2:
    print(f"    V4 Hit Rate (60d) vs 2x Edge:  r = {corr_hit_edge:>+.3f}")
    print(f"    Volatility vs 2x Edge:          r = {corr_vol:>+.3f}")
    print(f"    Avg Forward Return vs 2x Edge:  r = {corr_fwd:>+.3f}")

# Efficiency
print(f"\n  [E] Most Risk-Efficient Leverage:")
for lev, cnt in sorted(eff_dist.items()):
    pct = cnt / len(master) * 100
    print(f"    {lev:.1f}x: {cnt} tickers ({pct:.0f}%)")

print()
print("=" * 120)
print("  Done.")
