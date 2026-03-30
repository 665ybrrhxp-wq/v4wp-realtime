"""
V4 Velocity Weight Sweep: 속도 기반 S_Force + S_Div 최적 가중치 탐색
====================================================================
force = v_norm × p_vel (속도, 1차 미분) — S_conc 제거
F weight: 0.20 ~ 0.90 (step 0.05) → D weight = 1.0 - F
+ VEL_3ind (3지표 현재 가중치) 비교
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
    calc_pv_divergence, calc_pv_concordance,
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

F_WEIGHTS = [round(x, 2) for x in np.arange(0.20, 0.91, 0.05)]


def download_max(ticker):
    t = yf.Ticker(ticker)
    df = t.history(period='max', auto_adjust=True)
    if df is None or len(df) == 0:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def calc_force_macd_vel(df, fast=12, slow=26, signal=9):
    p_vel = df['Close'].pct_change().fillna(0)
    v_norm = df['Volume'] / df['Volume'].rolling(20).mean().replace(0, np.nan).fillna(df['Volume'])
    force = v_norm * p_vel
    fm = force.ewm(span=fast, adjust=False).mean() - force.ewm(span=slow, adjust=False).mean()
    fs = fm.ewm(span=signal, adjust=False).mean()
    return (fm - fs).rename('PV_Force_Hist')


def calc_force_macd_acc(df, fast=12, slow=26, signal=9):
    p_vel = df['Close'].pct_change().fillna(0)
    p_acc = p_vel.diff().fillna(0)
    v_norm = df['Volume'] / df['Volume'].rolling(20).mean().replace(0, np.nan).fillna(df['Volume'])
    force = v_norm * p_acc
    fm = force.ewm(span=fast, adjust=False).mean() - force.ewm(span=slow, adjust=False).mean()
    fs = fm.ewm(span=signal, adjust=False).mean()
    return (fm - fs).rename('PV_Force_Hist')


def precompute(df, w=20, divgate_days=3):
    n = len(df)
    pv_div = calc_pv_divergence(df, w)
    pv_conc = calc_pv_concordance(df, w)
    pv_fh_vel = calc_force_macd_vel(df)
    pv_fh_acc = calc_force_macd_acc(df)

    raw_div = np.array([np.clip(pv_div.iloc[i] / 3, -1, 1) for i in range(n)])
    consec = np.ones(n)
    for i in range(1, n):
        if (raw_div[i] > 0 and raw_div[i-1] > 0) or \
           (raw_div[i] < 0 and raw_div[i-1] < 0):
            consec[i] = consec[i-1] + 1
        else:
            consec[i] = 1

    return raw_div, consec, pv_conc, pv_fh_vel, pv_fh_acc


def calc_score(df, raw_div, consec, pv_conc, pv_fh,
               w=20, divgate_days=3, mode='2ind', f_weight=0.70):
    n = len(df)
    scores = np.zeros(n)
    for i in range(max(60, w), n):
        s_div = raw_div[i] if consec[i] >= divgate_days else 0.0
        s_conc = pv_conc.iloc[i]
        fhr_std = pv_fh.iloc[max(0, i - w):i].std() + 1e-10
        s_force = np.clip(pv_fh.iloc[i] / (2 * fhr_std), -1, 1)

        if mode == '3ind':
            dire = 0.45 * s_force + 0.30 * s_div + 0.25 * s_conc
            act = sum([abs(s_div) > 0.1, abs(s_conc) > 0.1, abs(s_force) > 0.1])
            mm = {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.2}
        else:
            d_weight = 1.0 - f_weight
            dire = f_weight * s_force + d_weight * s_div
            act = sum([abs(s_div) > 0.1, abs(s_force) > 0.1])
            mm = {0: 0.5, 1: 1.0, 2: 2.2}

        scores[i] = dire * mm.get(act, 1.0)
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
# Main
# ===============================================================
print("=" * 140)
print("  V4 VELOCITY WEIGHT SWEEP: 속도 기반 S_Force:S_Div 최적 가중치")
print("  force = v_norm x p_vel (1차 미분) | F: 0.20~0.90 step 0.05 | D = 1-F")
print("  + ACC_3ind(현재) + VEL_3ind(속도+3지표) 비교")
print("=" * 140)

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
    raw_div, consec, pv_conc, pv_fh_vel, pv_fh_acc = precompute(
        df_s, w=V4_W, divgate_days=DIVGATE)

    master[tk] = {}

    # ACC_3ind (현재 알고리즘)
    score = calc_score(df_s, raw_div, consec, pv_conc, pv_fh_acc,
                       w=V4_W, divgate_days=DIVGATE, mode='3ind')
    buys = get_buy_signals(df_s, score)
    master[tk]['ACC_CUR'] = simulate(close, close_2x, buys, dates)

    # VEL_3ind
    score = calc_score(df_s, raw_div, consec, pv_conc, pv_fh_vel,
                       w=V4_W, divgate_days=DIVGATE, mode='3ind')
    buys = get_buy_signals(df_s, score)
    master[tk]['VEL_3ind'] = simulate(close, close_2x, buys, dates)

    # VEL 2ind sweep
    for fw in F_WEIGHTS:
        name = f'VF{int(fw*100):02d}'
        score = calc_score(df_s, raw_div, consec, pv_conc, pv_fh_vel,
                           w=V4_W, divgate_days=DIVGATE, mode='2ind', f_weight=fw)
        buys = get_buy_signals(df_s, score)
        master[tk][name] = simulate(close, close_2x, buys, dates)

    cur = master[tk]['ACC_CUR']
    best_fw = max(F_WEIGHTS, key=lambda fw: master[tk][f'VF{int(fw*100):02d}']['efficiency'])
    best = master[tk][f'VF{int(best_fw*100):02d}']
    print(f"CUR eff={cur['efficiency']:>+.3f}  "
          f"BEST=VF{int(best_fw*100):02d} eff={best['efficiency']:>+.3f} edge={best['edge']:>+.1f}%")

tks = list(master.keys())
ALL_NAMES = ['ACC_CUR', 'VEL_3ind'] + [f'VF{int(fw*100):02d}' for fw in F_WEIGHTS]


# ===============================================================
# SECTION 1: Efficiency Heatmap
# ===============================================================
print(f"\n{'=' * 140}")
print(f"  SECTION 1: Efficiency by VEL F:D Weight")
print(f"  ACC_CUR = 현재 (가속도+3지표) | VEL_3ind = 속도+3지표 | VFxx = 속도+2지표(F:D)")
print(f"{'=' * 140}")

print(f"\n  {'Ticker':<7s} {'CUR':>7s} {'V3i':>7s}", end="")
for fw in F_WEIGHTS:
    print(f" {f'VF{int(fw*100)}':>6s}", end="")
print(f"  {'Best':>6s} {'Eff':>7s}")
print(f"  {'=' * (7 + 7 + 7 + len(F_WEIGHTS)*7 + 15)}")

for tk in tks:
    cur_eff = master[tk]['ACC_CUR']['efficiency']
    v3_eff = master[tk]['VEL_3ind']['efficiency']
    line = f"  {tk:<7s} {cur_eff:>+6.3f} {v3_eff:>+6.3f}"

    best_name = 'ACC_CUR'; best_eff = cur_eff
    if v3_eff > best_eff:
        best_eff = v3_eff; best_name = 'V3i'

    for fw in F_WEIGHTS:
        name = f'VF{int(fw*100):02d}'
        eff = master[tk][name]['efficiency']
        delta = eff - cur_eff
        m = "+" if delta > 0.03 else ("-" if delta < -0.03 else " ")
        line += f" {eff:>+5.3f}{m}"
        if eff > best_eff:
            best_eff = eff; best_name = f'VF{int(fw*100)}'

    line += f"  {best_name:>6s} {best_eff:>+6.3f}"
    print(line)

# Average
print(f"  {'-' * (7 + 7 + 7 + len(F_WEIGHTS)*7 + 15)}")
line = f"  {'AVG':<7s}"
cur_avg = np.mean([master[tk]['ACC_CUR']['efficiency'] for tk in tks])
v3_avg = np.mean([master[tk]['VEL_3ind']['efficiency'] for tk in tks])
line += f" {cur_avg:>+6.3f} {v3_avg:>+6.3f}"
for fw in F_WEIGHTS:
    name = f'VF{int(fw*100):02d}'
    avg = np.mean([master[tk][name]['efficiency'] for tk in tks])
    delta = avg - cur_avg
    m = "+" if delta > 0.02 else ("-" if delta < -0.02 else " ")
    line += f" {avg:>+5.3f}{m}"
print(line)


# ===============================================================
# SECTION 2: Summary Table
# ===============================================================
print(f"\n{'=' * 140}")
print(f"  SECTION 2: Summary — Edge, Efficiency, Bear/Bull")
print(f"{'=' * 140}")

print(f"\n  {'Strategy':<10s} {'F:D':>8s} {'AvgSig':>7s} {'Edge':>8s} {'Bear':>8s} "
      f"{'Bull':>8s} {'Worst':>8s} {'Effic':>8s} {'W/L':>7s} {'vs CUR':>8s}")
print(f"  {'=' * 105}")

summary = {}
for name in ALL_NAMES:
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
    summary[name] = s

base_eff = summary['ACC_CUR']['eff']

# Print ACC_CUR
s = summary['ACC_CUR']
print(f"  {'ACC_CUR':<10s} {'45:30:25':>8s} {s['sigs']:>6.0f} {s['edge']:>+7.2f}% "
      f"{s['bear']:>+7.2f}% {s['bull']:>+7.2f}% {s['worst']:>+7.2f}% "
      f"{s['eff']:>+7.3f} {s['wins']:>3d}/{s['losses']:<3d}  baseline")

# VEL_3ind
s = summary['VEL_3ind']
d = s['eff'] - base_eff
print(f"  {'VEL_3ind':<10s} {'45:30:25':>8s} {s['sigs']:>6.0f} {s['edge']:>+7.2f}% "
      f"{s['bear']:>+7.2f}% {s['bull']:>+7.2f}% {s['worst']:>+7.2f}% "
      f"{s['eff']:>+7.3f} {s['wins']:>3d}/{s['losses']:<3d} {d:>+7.3f}")

print(f"  {'-' * 105}")

for fw in F_WEIGHTS:
    name = f'VF{int(fw*100):02d}'
    s = summary[name]
    d = s['eff'] - base_eff
    dw = 1.0 - fw
    fd = f"{int(fw*100)}:{int(dw*100)}"

    marker = ""
    if d > 0.06: marker = " *** BEST ***"
    elif d > 0.04: marker = " ** GREAT **"
    elif d > 0.02: marker = " * good *"
    elif d > 0.005: marker = " mild+"
    elif d < -0.03: marker = " [HARMFUL]"
    elif d < -0.01: marker = " [worse]"

    print(f"  {name:<10s} {fd:>8s} {s['sigs']:>6.0f} {s['edge']:>+7.2f}% "
          f"{s['bear']:>+7.2f}% {s['bull']:>+7.2f}% {s['worst']:>+7.2f}% "
          f"{s['eff']:>+7.3f} {s['wins']:>3d}/{s['losses']:<3d} {d:>+7.3f}{marker}")


# ===============================================================
# SECTION 3: Per-ticker winners
# ===============================================================
print(f"\n{'=' * 140}")
print(f"  SECTION 3: Per-ticker Best VEL Weight")
print(f"{'=' * 140}")

eff_winners = {}
for tk in tks:
    best_name = 'ACC_CUR'; best_eff = master[tk]['ACC_CUR']['efficiency']
    for name in ALL_NAMES:
        eff = master[tk][name]['efficiency']
        if eff > best_eff:
            best_eff = eff; best_name = name
    eff_winners[best_name] = eff_winners.get(best_name, 0) + 1

print(f"\n  Winner distribution:")
for k in sorted(eff_winners, key=lambda x: eff_winners[x], reverse=True):
    print(f"    {k:<12s}: {eff_winners[k]:>2d}/{len(tks)} tickers")


# ===============================================================
# SECTION 4: QQQ/VOO Year-by-Year
# ===============================================================
print(f"\n{'=' * 140}")
print(f"  SECTION 4: QQQ & VOO Year-by-Year")
print(f"{'=' * 140}")

# Find top 3 VEL weights
ranked_vf = sorted(
    [f'VF{int(fw*100):02d}' for fw in F_WEIGHTS],
    key=lambda x: summary[x]['eff'], reverse=True
)
top3_vf = ranked_vf[:3]
compare = ['ACC_CUR', 'VEL_3ind'] + top3_vf

for tk in ['QQQ', 'VOO']:
    if tk not in tks:
        continue
    print(f"\n  {tk}:")
    all_years = sorted(set(r['yr'] for r in master[tk]['ACC_CUR']['yr_results']))

    print(f"  {'Year':>6s}", end="")
    for sn in compare:
        label = sn[:10]
        print(f" {label:>10s}", end="")
    print()
    print(f"  {'=' * (6 + len(compare)*11)}")

    for yr in all_years:
        line = f"  {yr:>6d}"
        for sn in compare:
            yr_r = [r for r in master[tk][sn]['yr_results'] if r['yr'] == yr]
            if yr_r:
                line += f" {yr_r[0]['diff']:>+9.1f}%"
            else:
                line += f" {'N/A':>10s}"
        print(line)

    print(f"  {'-' * (6 + len(compare)*11)}")
    line = f"  {'AVG':>6s}"
    for sn in compare:
        avg = np.mean([r['diff'] for r in master[tk][sn]['yr_results']])
        line += f" {avg:>+9.1f}%"
    print(line)


# ===============================================================
# SECTION 5: Final Verdict
# ===============================================================
print(f"\n{'=' * 140}")
print(f"  FINAL VERDICT")
print(f"{'=' * 140}")

# Overall ranking
all_ranked = sorted(ALL_NAMES, key=lambda x: summary[x]['eff'], reverse=True)

print(f"\n  Top 10 by efficiency:")
for i, name in enumerate(all_ranked[:10]):
    s = summary[name]
    d = s['eff'] - base_eff
    print(f"    #{i+1:>2d} {name:<12s} edge={s['edge']:>+.2f}%  eff={s['eff']:>+.3f} ({d:>+.3f})  "
          f"bear={s['bear']:>+.2f}%  bull={s['bull']:>+.2f}%  W/L={s['wins']}/{s['losses']}")

# Identify plateau
top5_effs = [summary[n]['eff'] for n in all_ranked[:5]]
spread = max(top5_effs) - min(top5_effs)
print(f"\n  Top-5 efficiency spread: {spread:.4f}")
if spread < 0.01:
    print(f"  → Very flat plateau: multiple weights perform similarly")
elif spread < 0.03:
    print(f"  → Mild plateau: top weights clustered")
else:
    print(f"  → Clear gradient: weight matters")

# Best single
best = all_ranked[0]
bs = summary[best]
print(f"\n  RECOMMENDED: {best}")
print(f"    Edge: {bs['edge']:>+.2f}%  Efficiency: {bs['eff']:>+.3f}")
print(f"    vs Current: {bs['eff']-base_eff:>+.3f} efficiency improvement")

print(f"\n{'=' * 140}")
print("  Done.")
