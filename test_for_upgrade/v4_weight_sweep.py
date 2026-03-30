"""
V4 Weight Sweep: S_conc 제거 후 S_Force:S_Div 최적 가중치 탐색
================================================================
F weight: 0.30 ~ 0.90 (step 0.05)  → D weight = 1.0 - F
총 13개 가중치 조합 + CURRENT(3지표) 비교
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
    calc_pv_divergence, calc_pv_concordance, calc_pv_force_macd,
    detect_signal_events, build_price_filter, smooth_earnings_volume,
)

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
LEVERAGE = 2.0
EXPENSE_RATIO_DAILY = 0.0095 / 252

V4_W = 20; SIGNAL_TH = 0.15; COOLDOWN = 5
ER_Q = 66; ATR_Q = 55; LOOKBACK = 252
DIVGATE = 3; CONFIRM = 3

# Weight sweep: F from 0.30 to 0.90 step 0.05
F_WEIGHTS = [round(x, 2) for x in np.arange(0.30, 0.91, 0.05)]
# Plus current 3-indicator formula as baseline
SWEEP_NAMES = ['CUR_3IND'] + [f'F{int(fw*100):02d}' for fw in F_WEIGHTS]


def download_max(ticker):
    t = yf.Ticker(ticker)
    df = t.history(period='max', auto_adjust=True)
    if df is None or len(df) == 0:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def precompute_components(df, w=20, divgate_days=3):
    n = len(df)
    pv_div = calc_pv_divergence(df, w)
    pv_conc = calc_pv_concordance(df, w)
    pv_fh = calc_pv_force_macd(df)

    raw_div = np.array([np.clip(pv_div.iloc[i] / 3, -1, 1) for i in range(n)])
    consec = np.ones(n)
    for i in range(1, n):
        if (raw_div[i] > 0 and raw_div[i-1] > 0) or \
           (raw_div[i] < 0 and raw_div[i-1] < 0):
            consec[i] = consec[i-1] + 1
        else:
            consec[i] = 1

    return raw_div, consec, pv_conc, pv_fh


def calc_score_sweep(df, raw_div, consec, pv_conc, pv_fh,
                     w=20, divgate_days=3, f_weight=None):
    """f_weight=None → current 3-indicator formula. Otherwise 2-indicator."""
    n = len(df)
    scores = np.zeros(n)
    for i in range(max(60, w), n):
        s_div = raw_div[i] if consec[i] >= divgate_days else 0.0
        s_conc = pv_conc.iloc[i]
        fhr_std = pv_fh.iloc[max(0, i - w):i].std() + 1e-10
        s_force = np.clip(pv_fh.iloc[i] / (2 * fhr_std), -1, 1)

        if f_weight is None:
            # Current 3-indicator
            dire = 0.45 * s_force + 0.30 * s_div + 0.25 * s_conc
            act = sum([abs(s_div) > 0.1, abs(s_conc) > 0.1, abs(s_force) > 0.1])
            mm = {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.2}
        else:
            d_weight = 1.0 - f_weight
            dire = f_weight * s_force + d_weight * s_div
            act = sum([abs(s_div) > 0.1, abs(s_force) > 0.1])
            mm = {0: 0.5, 1: 1.0, 2: 2.2}

        mult = mm.get(act, 1.0)
        scores[i] = dire * mult

    return pd.Series(scores, index=df.index)


def build_synthetic_2x(close):
    daily_ret = np.diff(close) / close[:-1]
    lev_price = np.zeros(len(close))
    lev_price[0] = close[0]
    for i in range(1, len(close)):
        lev_ret = LEVERAGE * daily_ret[i - 1] - EXPENSE_RATIO_DAILY
        lev_price[i] = lev_price[i - 1] * (1 + lev_ret)
        if lev_price[i] < 0.001:
            lev_price[i] = 0.001
    return lev_price


def get_buy_signals(df_s, score_series):
    events = detect_signal_events(score_series, th=SIGNAL_TH, cooldown=COOLDOWN)
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


def simulate(close, close_2x, buy_signals, dates):
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

    cash_a = 0.0; sh_1x_a = 0.0; sh_2x_a = 0.0
    cash_b = 0.0; sh_1x_b = 0.0
    sig_count = 0
    yr_data = {}; prev_yr = None

    def pf_a(idx):
        return sh_1x_a * close[idx] + sh_2x_a * close_2x[idx] + cash_a
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
                'deposits': 0.0,
            }
            prev_yr = yr

        cash_a += MONTHLY_DEPOSIT; cash_b += MONTHLY_DEPOSIT
        yr_data[yr]['deposits'] += MONTHLY_DEPOSIT

        for day_idx in range(fi, li + 1):
            if day_idx in buy_signals and cash_a > 1.0:
                amt = cash_a * SIGNAL_BUY_PCT
                sh_2x_a += amt / close_2x[day_idx]
                cash_a -= amt
                sig_count += 1

        if cash_a > 1.0:
            sh_1x_a += cash_a / close[li]; cash_a = 0.0
        if cash_b > 1.0:
            sh_1x_b += cash_b / close[li]; cash_b = 0.0

    yr_data[prev_yr]['end_a'] = pf_a(n - 1)
    yr_data[prev_yr]['end_b'] = pf_b(n - 1)

    yr_results = []
    for yr in sorted(yr_data.keys()):
        yd = yr_data[yr]
        da = yd['start_a'] + yd['deposits'] * 0.5
        db = yd.get('start_b', 0) + yd['deposits'] * 0.5
        # Need start_b
        pass

    # Simpler: just recompute
    # Re-do with proper tracking
    cash_a = 0.0; sh_1x_a = 0.0; sh_2x_a = 0.0
    cash_b = 0.0; sh_1x_b = 0.0
    sig_count = 0
    yr_data = {}; prev_yr = None

    for mk in sorted_months:
        mm = month_map[mk]
        fi, li = mm['first'], mm['last']
        yr = mm['period'].year

        if yr != prev_yr:
            if prev_yr is not None:
                ref = fi - 1 if fi > 0 else fi
                yr_data[prev_yr]['end_a'] = sh_1x_a * close[ref] + sh_2x_a * close_2x[ref] + cash_a
                yr_data[prev_yr]['end_b'] = sh_1x_b * close[ref] + cash_b
            yr_data[yr] = {
                'start_a': sh_1x_a * close[fi] + sh_2x_a * close_2x[fi] + cash_a,
                'start_b': sh_1x_b * close[fi] + cash_b,
                'deposits': 0.0,
            }
            prev_yr = yr

        cash_a += MONTHLY_DEPOSIT; cash_b += MONTHLY_DEPOSIT
        yr_data[yr]['deposits'] += MONTHLY_DEPOSIT

        for day_idx in range(fi, li + 1):
            if day_idx in buy_signals and cash_a > 1.0:
                amt = cash_a * SIGNAL_BUY_PCT
                sh_2x_a += amt / close_2x[day_idx]
                cash_a -= amt
                sig_count += 1

        if cash_a > 1.0:
            sh_1x_a += cash_a / close[li]; cash_a = 0.0
        if cash_b > 1.0:
            sh_1x_b += cash_b / close[li]; cash_b = 0.0

    yr_data[prev_yr]['end_a'] = sh_1x_a * close[n-1] + sh_2x_a * close_2x[n-1] + cash_a
    yr_data[prev_yr]['end_b'] = sh_1x_b * close[n-1] + cash_b

    yr_results = []
    for yr in sorted(yr_data.keys()):
        yd = yr_data[yr]
        da = yd['start_a'] + yd['deposits'] * 0.5
        db = yd['start_b'] + yd['deposits'] * 0.5
        ra = ((yd['end_a'] - yd['start_a'] - yd['deposits']) / da * 100) if da > 10 else 0
        rb = ((yd['end_b'] - yd['start_b'] - yd['deposits']) / db * 100) if db > 10 else 0
        yr_results.append({'yr': yr, 'ret_a': ra, 'ret_b': rb, 'diff': ra - rb})

    worst_diff = min(r['diff'] for r in yr_results)
    bull_yrs = [r for r in yr_results if r['ret_b'] > 5]
    bear_yrs = [r for r in yr_results if r['ret_b'] < -5]

    edge = np.mean([r['ret_a'] for r in yr_results]) - \
           np.mean([r['ret_b'] for r in yr_results])

    return {
        'edge': edge,
        'wins': sum(1 for r in yr_results if r['diff'] > 0.5),
        'losses': sum(1 for r in yr_results if r['diff'] < -0.5),
        'worst_diff': worst_diff,
        'sig_count': sig_count,
        'bull_edge': np.mean([r['diff'] for r in bull_yrs]) if bull_yrs else 0,
        'bear_edge': np.mean([r['diff'] for r in bear_yrs]) if bear_yrs else 0,
        'efficiency': edge / abs(worst_diff) if abs(worst_diff) > 0.1 else 0,
        'yr_results': yr_results,
    }


# ===============================================================
# Main — Precompute once, sweep weights
# ===============================================================
print("=" * 130)
print("  V4 WEIGHT SWEEP: S_Force:S_Div 가중치 최적화 (S_conc 제거)")
print("  F weight: 0.30 ~ 0.90 (step 0.05) | D weight = 1.0 - F")
print("=" * 130)

# Store: ticker -> sweep_name -> result
master = {}

for tk in TICKERS:
    print(f"  {tk}...", end=" ", flush=True)
    df = download_max(tk)
    if df is None or len(df) < 300:
        print("SKIP"); continue

    try:
        df_s = smooth_earnings_volume(df, ticker=tk)
    except Exception:
        df_s = df.copy()

    close = df_s['Close'].values
    dates = df_s.index
    close_2x = build_synthetic_2x(close)
    raw_div, consec, pv_conc, pv_fh = precompute_components(df_s, w=V4_W, divgate_days=DIVGATE)

    master[tk] = {}

    # Current 3-indicator
    score = calc_score_sweep(df_s, raw_div, consec, pv_conc, pv_fh,
                             w=V4_W, divgate_days=DIVGATE, f_weight=None)
    buys = get_buy_signals(df_s, score)
    master[tk]['CUR_3IND'] = simulate(close, close_2x, buys, dates)

    # Sweep F weights
    for fw in F_WEIGHTS:
        name = f'F{int(fw*100):02d}'
        score = calc_score_sweep(df_s, raw_div, consec, pv_conc, pv_fh,
                                 w=V4_W, divgate_days=DIVGATE, f_weight=fw)
        buys = get_buy_signals(df_s, score)
        master[tk][name] = simulate(close, close_2x, buys, dates)

    cur = master[tk]['CUR_3IND']
    # Find best sweep
    best_name = max(F_WEIGHTS, key=lambda fw: master[tk][f'F{int(fw*100):02d}']['efficiency'])
    best = master[tk][f'F{int(best_name*100):02d}']
    print(f"sigs={cur['sig_count']:>3d} edge={cur['edge']:>+.1f}%  "
          f"BEST=F{int(best_name*100):02d} sigs={best['sig_count']:>3d} "
          f"edge={best['edge']:>+.1f}% eff={best['efficiency']:>+.3f}")

tks = list(master.keys())


# ===============================================================
# SECTION 1: Efficiency Heatmap (main result)
# ===============================================================
print(f"\n{'=' * 130}")
print(f"  SECTION 1: Efficiency by F:D Weight (higher = better)")
print(f"  F weight shown. D weight = 1-F. CUR = current 3-indicator (0.45F+0.30D+0.25C)")
print(f"{'=' * 130}")

# Header
print(f"\n  {'Ticker':<7s} {'CUR':>7s}", end="")
for fw in F_WEIGHTS:
    print(f" {f'F{int(fw*100)}':>6s}", end="")
print(f"  {'Best':>6s} {'BestEff':>8s}")
print(f"  {'=' * (7 + 7 + len(F_WEIGHTS)*7 + 16)}")

# Per-ticker row
all_effs = {name: [] for name in SWEEP_NAMES}
ticker_bests = {}

for tk in tks:
    cur_eff = master[tk]['CUR_3IND']['efficiency']
    all_effs['CUR_3IND'].append(cur_eff)

    line = f"  {tk:<7s} {cur_eff:>+6.3f}"

    best_fw = None; best_eff = cur_eff; best_name_tk = 'CUR'
    for fw in F_WEIGHTS:
        name = f'F{int(fw*100):02d}'
        eff = master[tk][name]['efficiency']
        all_effs[name].append(eff)

        # Mark relative to current
        delta = eff - cur_eff
        if delta > 0.03:
            mark = "+"
        elif delta < -0.03:
            mark = "-"
        else:
            mark = " "
        line += f" {eff:>+5.3f}{mark}"

        if eff > best_eff:
            best_eff = eff; best_fw = fw; best_name_tk = f'F{int(fw*100)}'

    ticker_bests[tk] = best_name_tk
    line += f"  {best_name_tk:>6s} {best_eff:>+7.3f}"
    print(line)

# Average row
print(f"  {'-' * (7 + 7 + len(F_WEIGHTS)*7 + 16)}")
line = f"  {'AVG':<7s} {np.mean(all_effs['CUR_3IND']):>+6.3f}"
avg_by_fw = {}
for fw in F_WEIGHTS:
    name = f'F{int(fw*100):02d}'
    avg = np.mean(all_effs[name])
    avg_by_fw[fw] = avg
    delta = avg - np.mean(all_effs['CUR_3IND'])
    mark = "+" if delta > 0.01 else ("-" if delta < -0.01 else " ")
    line += f" {avg:>+5.3f}{mark}"
print(line)


# ===============================================================
# SECTION 2: Summary Table — Edge, Bear, Bull, Efficiency
# ===============================================================
print(f"\n{'=' * 130}")
print(f"  SECTION 2: Summary — Edge, Efficiency, Bear/Bull")
print(f"{'=' * 130}")

print(f"\n  {'Weight':<10s} {'F:D':>8s} {'AvgSig':>7s} {'Edge':>8s} {'Bear':>8s} "
      f"{'Bull':>8s} {'Worst':>8s} {'Effic':>8s} {'W/L':>7s} {'vs CUR':>8s}")
print(f"  {'=' * 100}")

# Current
s_cur = {
    'sigs': np.mean([master[tk]['CUR_3IND']['sig_count'] for tk in tks]),
    'edge': np.mean([master[tk]['CUR_3IND']['edge'] for tk in tks]),
    'bear': np.mean([master[tk]['CUR_3IND']['bear_edge'] for tk in tks]),
    'bull': np.mean([master[tk]['CUR_3IND']['bull_edge'] for tk in tks]),
    'worst': np.mean([master[tk]['CUR_3IND']['worst_diff'] for tk in tks]),
    'eff': np.mean([master[tk]['CUR_3IND']['efficiency'] for tk in tks]),
    'wins': sum(master[tk]['CUR_3IND']['wins'] for tk in tks),
    'losses': sum(master[tk]['CUR_3IND']['losses'] for tk in tks),
}

print(f"  {'CUR_3IND':<10s} {'45:30:25':>8s} {s_cur['sigs']:>6.0f} {s_cur['edge']:>+7.2f}% "
      f"{s_cur['bear']:>+7.2f}% {s_cur['bull']:>+7.2f}% {s_cur['worst']:>+7.2f}% "
      f"{s_cur['eff']:>+7.3f} {s_cur['wins']:>3d}/{s_cur['losses']:<3d}  baseline")

sweep_summary = {}
for fw in F_WEIGHTS:
    name = f'F{int(fw*100):02d}'
    dw = 1.0 - fw
    s = {
        'sigs': np.mean([master[tk][name]['sig_count'] for tk in tks]),
        'edge': np.mean([master[tk][name]['edge'] for tk in tks]),
        'bear': np.mean([master[tk][name]['bear_edge'] for tk in tks]),
        'bull': np.mean([master[tk][name]['bull_edge'] for tk in tks]),
        'worst': np.mean([master[tk][name]['worst_diff'] for tk in tks]),
        'eff': np.mean([master[tk][name]['efficiency'] for tk in tks]),
        'wins': sum(master[tk][name]['wins'] for tk in tks),
        'losses': sum(master[tk][name]['losses'] for tk in tks),
    }
    sweep_summary[fw] = s
    delta_eff = s['eff'] - s_cur['eff']

    marker = ""
    if delta_eff > 0.05: marker = " *** BEST ***"
    elif delta_eff > 0.02: marker = " ** BETTER **"
    elif delta_eff > 0.01: marker = " * good *"
    elif delta_eff > 0.005: marker = " mild+"
    elif delta_eff < -0.03: marker = " [HARMFUL]"
    elif delta_eff < -0.01: marker = " [worse]"

    fd_str = f"{int(fw*100)}:{int(dw*100)}"
    print(f"  {name:<10s} {fd_str:>8s} {s['sigs']:>6.0f} {s['edge']:>+7.2f}% "
          f"{s['bear']:>+7.2f}% {s['bull']:>+7.2f}% {s['worst']:>+7.2f}% "
          f"{s['eff']:>+7.3f} {s['wins']:>3d}/{s['losses']:<3d} {delta_eff:>+7.3f}{marker}")


# ===============================================================
# SECTION 3: Top 5 weights deep dive
# ===============================================================
print(f"\n{'=' * 130}")
print(f"  SECTION 3: Top 5 Weights — Per-ticker Breakdown")
print(f"{'=' * 130}")

ranked_fw = sorted(F_WEIGHTS, key=lambda fw: sweep_summary[fw]['eff'], reverse=True)
top5 = ranked_fw[:5]

print(f"\n  Top 5 by avg efficiency:")
for i, fw in enumerate(top5):
    s = sweep_summary[fw]
    print(f"    #{i+1} F{int(fw*100):02d} ({int(fw*100)}:{int((1-fw)*100)})  "
          f"edge={s['edge']:>+.2f}%  eff={s['eff']:>+.3f}  "
          f"bear={s['bear']:>+.2f}%  bull={s['bull']:>+.2f}%")

print(f"\n  Per-ticker efficiency for top 5:")
print(f"  {'Ticker':<7s} {'CUR':>7s}", end="")
for fw in top5:
    print(f" {f'F{int(fw*100)}':>7s}", end="")
print(f"  {'Best':>7s}")
print(f"  {'=' * (7 + 7 + len(top5)*8 + 8)}")

top5_wins = {'CUR': 0}
for fw in top5:
    top5_wins[f'F{int(fw*100)}'] = 0

for tk in tks:
    cur_eff = master[tk]['CUR_3IND']['efficiency']
    line = f"  {tk:<7s} {cur_eff:>+6.3f}"
    best_n = 'CUR'; best_e = cur_eff
    for fw in top5:
        name = f'F{int(fw*100):02d}'
        eff = master[tk][name]['efficiency']
        delta = eff - cur_eff
        m = "+" if delta > 0.01 else ("-" if delta < -0.01 else " ")
        line += f" {eff:>+6.3f}{m}"
        if eff > best_e:
            best_e = eff; best_n = f'F{int(fw*100)}'
    top5_wins[best_n] = top5_wins.get(best_n, 0) + 1
    line += f"  {best_n:>7s}"
    print(line)

print(f"\n  Wins among top 5 + CUR:")
for k in sorted(top5_wins, key=lambda x: top5_wins[x], reverse=True):
    if top5_wins[k] > 0:
        print(f"    {k:<8s}: {top5_wins[k]:>2d}/{len(tks)} tickers")


# ===============================================================
# SECTION 4: QQQ/VOO Year-by-Year for Top 3
# ===============================================================
print(f"\n{'=' * 130}")
print(f"  SECTION 4: QQQ & VOO Year-by-Year — CUR vs Top 3 weights")
print(f"{'=' * 130}")

top3 = ranked_fw[:3]

for tk in ['QQQ', 'VOO']:
    if tk not in tks:
        continue
    print(f"\n  {tk}:")
    all_years = sorted(set(r['yr'] for r in master[tk]['CUR_3IND']['yr_results']))
    print(f"  {'Year':>6s} {'CUR':>8s}", end="")
    for fw in top3:
        print(f" {f'F{int(fw*100)}':>8s}", end="")
    print()
    print(f"  {'=' * (6 + 8 + len(top3)*9 + 2)}")

    for yr in all_years:
        line = f"  {yr:>6d}"
        yr_r = [r for r in master[tk]['CUR_3IND']['yr_results'] if r['yr'] == yr]
        line += f" {yr_r[0]['diff']:>+7.1f}%" if yr_r else f" {'N/A':>8s}"
        for fw in top3:
            name = f'F{int(fw*100):02d}'
            yr_r = [r for r in master[tk][name]['yr_results'] if r['yr'] == yr]
            line += f" {yr_r[0]['diff']:>+7.1f}%" if yr_r else f" {'N/A':>8s}"
        print(line)

    print(f"  {'-' * (6 + 8 + len(top3)*9 + 2)}")
    line = f"  {'AVG':>6s}"
    avg_cur = np.mean([r['diff'] for r in master[tk]['CUR_3IND']['yr_results']])
    line += f" {avg_cur:>+7.1f}%"
    for fw in top3:
        name = f'F{int(fw*100):02d}'
        avg = np.mean([r['diff'] for r in master[tk][name]['yr_results']])
        line += f" {avg:>+7.1f}%"
    print(line)


# ===============================================================
# SECTION 5: Final Verdict
# ===============================================================
print(f"\n{'=' * 130}")
print(f"  FINAL VERDICT")
print(f"{'=' * 130}")

best_fw = ranked_fw[0]
best_s = sweep_summary[best_fw]
delta = best_s['eff'] - s_cur['eff']

print(f"\n  Current (3-indicator): 0.45*F + 0.30*D + 0.25*C")
print(f"    Edge: {s_cur['edge']:>+.2f}%  Efficiency: {s_cur['eff']:>+.3f}  "
      f"Bear: {s_cur['bear']:>+.2f}%  Bull: {s_cur['bull']:>+.2f}%")

print(f"\n  BEST (2-indicator): {int(best_fw*100)}% Force + {int((1-best_fw)*100)}% Div")
print(f"    Edge: {best_s['edge']:>+.2f}%  Efficiency: {best_s['eff']:>+.3f}  "
      f"Bear: {best_s['bear']:>+.2f}%  Bull: {best_s['bull']:>+.2f}%")
print(f"    Delta efficiency: {delta:>+.3f}")

# Runner ups
print(f"\n  Top 5 weights (by efficiency):")
for i, fw in enumerate(ranked_fw[:5]):
    s = sweep_summary[fw]
    d = s['eff'] - s_cur['eff']
    print(f"    #{i+1} F:{int(fw*100):02d} D:{int((1-fw)*100):02d}  "
          f"edge={s['edge']:>+.2f}%  eff={s['eff']:>+.3f} ({d:>+.3f})")

# Check if clear winner or plateau
top_effs = [sweep_summary[fw]['eff'] for fw in ranked_fw[:5]]
spread = max(top_effs) - min(top_effs)
print(f"\n  Top-5 efficiency spread: {spread:.3f}")
if spread < 0.02:
    print(f"  → Flat plateau: weights don't matter much in the top range")
elif spread < 0.05:
    print(f"  → Mild preference for top weight, but not dramatic")
else:
    print(f"  → Clear winner: {int(best_fw*100)}:{int((1-best_fw)*100)} is meaningfully better")

print(f"\n{'=' * 130}")
print("  Done.")
