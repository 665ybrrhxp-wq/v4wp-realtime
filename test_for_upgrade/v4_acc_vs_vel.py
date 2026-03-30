"""
V4 Acceleration vs Velocity Test
=================================
현재: force = v_norm × p_acc (가속도, 2차 미분)
대안: force = v_norm × p_vel (속도, 1차 미분)

각각에 대해:
  - 3지표 현재 가중치 (0.45F+0.30D+0.25C)
  - 2지표 최적 가중치 sweep (F: 0.30~0.80, step 0.10)
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


def download_max(ticker):
    t = yf.Ticker(ticker)
    df = t.history(period='max', auto_adjust=True)
    if df is None or len(df) == 0:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def calc_force_macd_acc(df, fast=12, slow=26, signal=9):
    """현재: acceleration (2차 미분)"""
    p_vel = df['Close'].pct_change().fillna(0)
    p_acc = p_vel.diff().fillna(0)
    v_norm = df['Volume'] / df['Volume'].rolling(20).mean().replace(0, np.nan).fillna(df['Volume'])
    force = v_norm * p_acc
    fm = force.ewm(span=fast, adjust=False).mean() - force.ewm(span=slow, adjust=False).mean()
    fs = fm.ewm(span=signal, adjust=False).mean()
    return (fm - fs).rename('PV_Force_Hist')


def calc_force_macd_vel(df, fast=12, slow=26, signal=9):
    """대안: velocity (1차 미분)"""
    p_vel = df['Close'].pct_change().fillna(0)
    v_norm = df['Volume'] / df['Volume'].rolling(20).mean().replace(0, np.nan).fillna(df['Volume'])
    force = v_norm * p_vel
    fm = force.ewm(span=fast, adjust=False).mean() - force.ewm(span=slow, adjust=False).mean()
    fs = fm.ewm(span=signal, adjust=False).mean()
    return (fm - fs).rename('PV_Force_Hist')


def precompute(df, w=20, divgate_days=3):
    n = len(df)
    pv_div = calc_pv_divergence(df, w)
    pv_conc = calc_pv_concordance(df, w)
    pv_fh_acc = calc_force_macd_acc(df)
    pv_fh_vel = calc_force_macd_vel(df)

    raw_div = np.array([np.clip(pv_div.iloc[i] / 3, -1, 1) for i in range(n)])
    consec = np.ones(n)
    for i in range(1, n):
        if (raw_div[i] > 0 and raw_div[i-1] > 0) or \
           (raw_div[i] < 0 and raw_div[i-1] < 0):
            consec[i] = consec[i-1] + 1
        else:
            consec[i] = 1

    return raw_div, consec, pv_conc, pv_fh_acc, pv_fh_vel


def calc_score(df, raw_div, consec, pv_conc, pv_fh,
               w=20, divgate_days=3, mode='3ind', f_weight=0.45):
    """
    mode='3ind': 0.45F+0.30D+0.25C (현재), activity 3개
    mode='2ind': f_weight*F + (1-f_weight)*D, activity 2개
    """
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
# Define all strategies
# ===============================================================
# Strategy format: (name, force_type, mode, f_weight)
F_SWEEP = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

strategies = []
# 3-indicator (current weights)
strategies.append(('ACC_3ind', 'acc', '3ind', 0.45))
strategies.append(('VEL_3ind', 'vel', '3ind', 0.45))
# 2-indicator weight sweep
for fw in F_SWEEP:
    strategies.append((f'ACC_F{int(fw*100)}', 'acc', '2ind', fw))
    strategies.append((f'VEL_F{int(fw*100)}', 'vel', '2ind', fw))


# ===============================================================
# Main
# ===============================================================
print("=" * 130)
print("  V4 ACCELERATION vs VELOCITY TEST")
print("  ACC: force = v_norm x p_acc (가속도, 2차미분)  |  VEL: force = v_norm x p_vel (속도, 1차미분)")
print("  3ind = 0.45F+0.30D+0.25C  |  2ind = F weight sweep (0.30~0.80)")
print("=" * 130)

master = {}  # ticker -> strategy_name -> result

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
    raw_div, consec, pv_conc, pv_fh_acc, pv_fh_vel = precompute(
        df_s, w=V4_W, divgate_days=DIVGATE)

    master[tk] = {}
    for sname, ftype, mode, fw in strategies:
        pv_fh = pv_fh_acc if ftype == 'acc' else pv_fh_vel
        score = calc_score(df_s, raw_div, consec, pv_conc, pv_fh,
                          w=V4_W, divgate_days=DIVGATE, mode=mode, f_weight=fw)
        buys = get_buy_signals(df_s, score)
        master[tk][sname] = simulate(close, close_2x, buys, dates)

    # Quick summary
    a3 = master[tk]['ACC_3ind']
    v3 = master[tk]['VEL_3ind']
    print(f"ACC_3ind: edge={a3['edge']:>+.1f}% eff={a3['efficiency']:>+.3f}  "
          f"VEL_3ind: edge={v3['edge']:>+.1f}% eff={v3['efficiency']:>+.3f}")

tks = list(master.keys())


# ===============================================================
# SECTION 1: ACC vs VEL — 3-indicator (현재 구조)
# ===============================================================
print(f"\n{'=' * 130}")
print(f"  SECTION 1: ACC vs VEL — 3-indicator (0.45F + 0.30D + 0.25C)")
print(f"{'=' * 130}")

print(f"\n  {'Ticker':<7s} {'ACC sigs':>9s} {'ACC edge':>10s} {'ACC eff':>9s}  "
      f"{'VEL sigs':>9s} {'VEL edge':>10s} {'VEL eff':>9s}  {'Winner':>8s}")
print(f"  {'=' * 85}")

acc3_wins = 0; vel3_wins = 0
for tk in tks:
    a = master[tk]['ACC_3ind']
    v = master[tk]['VEL_3ind']
    w = "ACC" if a['efficiency'] > v['efficiency'] else "VEL"
    if a['efficiency'] > v['efficiency']:
        acc3_wins += 1
    else:
        vel3_wins += 1
    print(f"  {tk:<7s} {a['sig_count']:>8d} {a['edge']:>+9.2f}% {a['efficiency']:>+8.3f}  "
          f"{v['sig_count']:>8d} {v['edge']:>+9.2f}% {v['efficiency']:>+8.3f}  {w:>8s}")

print(f"  {'-' * 85}")
a_avg_edge = np.mean([master[tk]['ACC_3ind']['edge'] for tk in tks])
a_avg_eff = np.mean([master[tk]['ACC_3ind']['efficiency'] for tk in tks])
v_avg_edge = np.mean([master[tk]['VEL_3ind']['edge'] for tk in tks])
v_avg_eff = np.mean([master[tk]['VEL_3ind']['efficiency'] for tk in tks])
print(f"  {'AVG':<7s} {'':>9s} {a_avg_edge:>+9.2f}% {a_avg_eff:>+8.3f}  "
      f"{'':>9s} {v_avg_edge:>+9.2f}% {v_avg_eff:>+8.3f}  "
      f"{'ACC' if a_avg_eff > v_avg_eff else 'VEL':>8s}")
print(f"\n  ACC wins: {acc3_wins}/{len(tks)}  |  VEL wins: {vel3_wins}/{len(tks)}")


# ===============================================================
# SECTION 2: Weight Sweep — ACC vs VEL (2-indicator, no S_conc)
# ===============================================================
print(f"\n{'=' * 130}")
print(f"  SECTION 2: Weight Sweep — ACC vs VEL (2-indicator, S_conc 제거)")
print(f"{'=' * 130}")

print(f"\n  {'F:D Weight':<12s}", end="")
print(f" {'ACC edge':>10s} {'ACC eff':>9s}", end="")
print(f" {'VEL edge':>10s} {'VEL eff':>9s}", end="")
print(f" {'Winner':>8s} {'Best eff':>9s}")
print(f"  {'=' * 70}")

# Also include 3ind as reference
ref_a = {'edge': a_avg_edge, 'eff': a_avg_eff}
print(f"  {'CUR(3ind)':<12s} {ref_a['edge']:>+9.2f}% {ref_a['eff']:>+8.3f}"
      f" {v_avg_edge:>+9.2f}% {v_avg_eff:>+8.3f}  {'ref':>8s}")
print(f"  {'-' * 70}")

best_overall_name = None; best_overall_eff = -999

for fw in F_SWEEP:
    a_name = f'ACC_F{int(fw*100)}'
    v_name = f'VEL_F{int(fw*100)}'

    a_edge = np.mean([master[tk][a_name]['edge'] for tk in tks])
    a_eff = np.mean([master[tk][a_name]['efficiency'] for tk in tks])
    v_edge = np.mean([master[tk][v_name]['edge'] for tk in tks])
    v_eff = np.mean([master[tk][v_name]['efficiency'] for tk in tks])

    w = "ACC" if a_eff > v_eff else "VEL"
    best_eff = max(a_eff, v_eff)
    best_name = a_name if a_eff >= v_eff else v_name

    if best_eff > best_overall_eff:
        best_overall_eff = best_eff
        best_overall_name = best_name

    fd = f"{int(fw*100)}:{int((1-fw)*100)}"
    print(f"  {fd:<12s} {a_edge:>+9.2f}% {a_eff:>+8.3f}"
          f" {v_edge:>+9.2f}% {v_eff:>+8.3f}  {w:>8s} {best_eff:>+8.3f}")


# ===============================================================
# SECTION 3: Per-ticker — Best ACC config vs Best VEL config
# ===============================================================
print(f"\n{'=' * 130}")
print(f"  SECTION 3: Per-ticker — ACC vs VEL 최적 가중치 비교")
print(f"{'=' * 130}")

# Find each ticker's best ACC and best VEL
print(f"\n  {'Ticker':<7s} {'Best ACC':>12s} {'ACC eff':>9s}  {'Best VEL':>12s} {'VEL eff':>9s}  {'Overall':>10s}")
print(f"  {'=' * 75}")

acc_total_wins = 0; vel_total_wins = 0

for tk in tks:
    # Best ACC
    best_a_name = 'ACC_3ind'; best_a_eff = master[tk]['ACC_3ind']['efficiency']
    for fw in F_SWEEP:
        name = f'ACC_F{int(fw*100)}'
        eff = master[tk][name]['efficiency']
        if eff > best_a_eff:
            best_a_eff = eff; best_a_name = name

    # Best VEL
    best_v_name = 'VEL_3ind'; best_v_eff = master[tk]['VEL_3ind']['efficiency']
    for fw in F_SWEEP:
        name = f'VEL_F{int(fw*100)}'
        eff = master[tk][name]['efficiency']
        if eff > best_v_eff:
            best_v_eff = eff; best_v_name = name

    overall = "ACC" if best_a_eff >= best_v_eff else "VEL"
    if best_a_eff >= best_v_eff:
        acc_total_wins += 1
    else:
        vel_total_wins += 1

    print(f"  {tk:<7s} {best_a_name:>12s} {best_a_eff:>+8.3f}  "
          f"{best_v_name:>12s} {best_v_eff:>+8.3f}  {overall:>10s}")

print(f"\n  Overall: ACC wins {acc_total_wins}/{len(tks)} | VEL wins {vel_total_wins}/{len(tks)}")


# ===============================================================
# SECTION 4: Full ranking — All strategies
# ===============================================================
print(f"\n{'=' * 130}")
print(f"  SECTION 4: Full Ranking — All Strategies by Efficiency")
print(f"{'=' * 130}")

all_strats = {}
for sname, ftype, mode, fw in strategies:
    avg_edge = np.mean([master[tk][sname]['edge'] for tk in tks])
    avg_eff = np.mean([master[tk][sname]['efficiency'] for tk in tks])
    avg_bear = np.mean([master[tk][sname]['bear_edge'] for tk in tks])
    avg_bull = np.mean([master[tk][sname]['bull_edge'] for tk in tks])
    avg_worst = np.mean([master[tk][sname]['worst_diff'] for tk in tks])
    avg_sigs = np.mean([master[tk][sname]['sig_count'] for tk in tks])
    total_w = sum(master[tk][sname]['wins'] for tk in tks)
    total_l = sum(master[tk][sname]['losses'] for tk in tks)
    all_strats[sname] = {
        'edge': avg_edge, 'eff': avg_eff, 'bear': avg_bear,
        'bull': avg_bull, 'worst': avg_worst, 'sigs': avg_sigs,
        'wins': total_w, 'losses': total_l,
        'ftype': ftype, 'mode': mode, 'fw': fw,
    }

ranked = sorted(all_strats.keys(), key=lambda x: all_strats[x]['eff'], reverse=True)

print(f"\n  {'Rank':>4s} {'Strategy':<14s} {'Type':>4s} {'Sigs':>5s} {'Edge':>8s} "
      f"{'Bear':>7s} {'Bull':>7s} {'Worst':>7s} {'Effic':>8s} {'W/L':>7s} {'vs CUR':>8s}")
print(f"  {'=' * 100}")

cur_eff = all_strats['ACC_3ind']['eff']
for i, sname in enumerate(ranked):
    s = all_strats[sname]
    delta = s['eff'] - cur_eff
    marker = ""
    if sname == 'ACC_3ind':
        marker = " <-- CURRENT"
    elif delta > 0.05:
        marker = " ***"
    elif delta > 0.02:
        marker = " **"
    elif delta < -0.03:
        marker = " [bad]"

    print(f"  {i+1:>4d} {sname:<14s} {s['ftype']:>4s} {s['sigs']:>4.0f} {s['edge']:>+7.2f}% "
          f"{s['bear']:>+6.2f}% {s['bull']:>+6.2f}% {s['worst']:>+6.2f}% {s['eff']:>+7.3f} "
          f"{s['wins']:>3d}/{s['losses']:<3d} {delta:>+7.3f}{marker}")


# ===============================================================
# SECTION 5: QQQ/VOO Year-by-Year — Top ACC vs Top VEL
# ===============================================================
print(f"\n{'=' * 130}")
print(f"  SECTION 5: QQQ & VOO Year-by-Year — CUR vs Top ACC vs Top VEL")
print(f"{'=' * 130}")

# Top ACC (excluding 3ind)
top_acc = sorted([s for s in ranked if all_strats[s]['ftype'] == 'acc'],
                 key=lambda x: all_strats[x]['eff'], reverse=True)[0]
top_vel = sorted([s for s in ranked if all_strats[s]['ftype'] == 'vel'],
                 key=lambda x: all_strats[x]['eff'], reverse=True)[0]

compare = ['ACC_3ind', top_acc, top_vel]
# Remove duplicates
compare = list(dict.fromkeys(compare))

for tk in ['QQQ', 'VOO']:
    if tk not in tks:
        continue
    print(f"\n  {tk}:")
    all_years = sorted(set(r['yr'] for r in master[tk]['ACC_3ind']['yr_results']))

    print(f"  {'Year':>6s}", end="")
    for sn in compare:
        print(f" {sn:>12s}", end="")
    print()
    print(f"  {'=' * (6 + len(compare)*13)}")

    for yr in all_years:
        line = f"  {yr:>6d}"
        for sn in compare:
            yr_r = [r for r in master[tk][sn]['yr_results'] if r['yr'] == yr]
            if yr_r:
                line += f" {yr_r[0]['diff']:>+11.1f}%"
            else:
                line += f" {'N/A':>12s}"
        print(line)

    print(f"  {'-' * (6 + len(compare)*13)}")
    line = f"  {'AVG':>6s}"
    for sn in compare:
        avg = np.mean([r['diff'] for r in master[tk][sn]['yr_results']])
        line += f" {avg:>+11.1f}%"
    print(line)


# ===============================================================
# SECTION 6: Final Verdict
# ===============================================================
print(f"\n{'=' * 130}")
print(f"  FINAL VERDICT")
print(f"{'=' * 130}")

best = ranked[0]
bs = all_strats[best]
print(f"\n  OVERALL BEST: {best}")
print(f"    Type: {'Acceleration' if bs['ftype']=='acc' else 'Velocity'}")
print(f"    Edge: {bs['edge']:>+.2f}%  Efficiency: {bs['eff']:>+.3f} ({bs['eff']-cur_eff:>+.3f} vs CUR)")
print(f"    Bear: {bs['bear']:>+.2f}%  Bull: {bs['bull']:>+.2f}%  W/L: {bs['wins']}/{bs['losses']}")

# Best per type
best_acc = sorted([s for s in ranked if all_strats[s]['ftype']=='acc'],
                  key=lambda x: all_strats[x]['eff'], reverse=True)[0]
best_vel = sorted([s for s in ranked if all_strats[s]['ftype']=='vel'],
                  key=lambda x: all_strats[x]['eff'], reverse=True)[0]

ba = all_strats[best_acc]; bv = all_strats[best_vel]
print(f"\n  Best ACC: {best_acc}  eff={ba['eff']:>+.3f}  edge={ba['edge']:>+.2f}%")
print(f"  Best VEL: {best_vel}  eff={bv['eff']:>+.3f}  edge={bv['edge']:>+.2f}%")
print(f"  Difference: ACC-VEL = {ba['eff']-bv['eff']:>+.3f} efficiency")

if ba['eff'] > bv['eff'] + 0.02:
    print(f"\n  → Acceleration(가속도) 유지가 더 좋음")
elif bv['eff'] > ba['eff'] + 0.02:
    print(f"\n  → Velocity(속도)로 변경이 더 좋음")
else:
    print(f"\n  → 두 방식 차이가 미미함 (±0.02 이내)")

print(f"\n{'=' * 130}")
print("  Done.")
