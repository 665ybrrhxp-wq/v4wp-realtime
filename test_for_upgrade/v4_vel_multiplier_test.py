"""
V4 Velocity Multiplier Test: Activity Multiplier 유/무 비교
============================================================
현재 2지표 multiplier: {0: 0.5, 1: 1.0, 2: 2.2}
테스트: multiplier 제거 (항상 1.0)

VEL 기반, F weight sweep 0.30~0.85 step 0.05
각각 WITH mult / WITHOUT mult
+ ACC_CUR (현재) 비교
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

F_WEIGHTS = [round(x, 2) for x in np.arange(0.30, 0.86, 0.05)]


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
               w=20, divgate_days=3, mode='2ind', f_weight=0.70, use_mult=True):
    n = len(df)
    scores = np.zeros(n)
    for i in range(max(60, w), n):
        s_div = raw_div[i] if consec[i] >= divgate_days else 0.0
        s_conc = pv_conc.iloc[i]
        fhr_std = pv_fh.iloc[max(0, i - w):i].std() + 1e-10
        s_force = np.clip(pv_fh.iloc[i] / (2 * fhr_std), -1, 1)

        if mode == '3ind':
            dire = 0.45 * s_force + 0.30 * s_div + 0.25 * s_conc
            if use_mult:
                act = sum([abs(s_div) > 0.1, abs(s_conc) > 0.1, abs(s_force) > 0.1])
                mm = {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.2}
                mult = mm.get(act, 1.0)
            else:
                mult = 1.0
        else:
            d_weight = 1.0 - f_weight
            dire = f_weight * s_force + d_weight * s_div
            if use_mult:
                act = sum([abs(s_div) > 0.1, abs(s_force) > 0.1])
                mm = {0: 0.5, 1: 1.0, 2: 2.2}
                mult = mm.get(act, 1.0)
            else:
                mult = 1.0

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
print("  V4 MULTIPLIER TEST: Activity Multiplier 유 vs 무")
print("  WITH mult: {0:0.5, 1:1.0, 2:2.2} | WITHOUT mult: always 1.0")
print("  VEL 기반, F weight 0.30~0.85 step 0.05 + ACC_CUR baseline")
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

    # ACC_CUR (현재)
    score = calc_score(df_s, raw_div, consec, pv_conc, pv_fh_acc,
                       w=V4_W, divgate_days=DIVGATE, mode='3ind', use_mult=True)
    buys = get_buy_signals(df_s, score)
    master[tk]['ACC_CUR'] = simulate(close, close_2x, buys, dates)

    # VEL sweep: WITH mult and WITHOUT mult
    for fw in F_WEIGHTS:
        for use_m, suffix in [(True, 'M'), (False, 'N')]:
            name = f'V{suffix}{int(fw*100):02d}'
            score = calc_score(df_s, raw_div, consec, pv_conc, pv_fh_vel,
                               w=V4_W, divgate_days=DIVGATE, mode='2ind',
                               f_weight=fw, use_mult=use_m)
            buys = get_buy_signals(df_s, score)
            master[tk][name] = simulate(close, close_2x, buys, dates)

    cur = master[tk]['ACC_CUR']
    print(f"CUR eff={cur['efficiency']:>+.3f}")

tks = list(master.keys())


# ===============================================================
# SECTION 1: WITH mult vs WITHOUT mult — Summary
# ===============================================================
print(f"\n{'=' * 140}")
print(f"  SECTION 1: Multiplier ON vs OFF — Summary per Weight")
print(f"  VM = VEL+Multiplier | VN = VEL+NoMultiplier")
print(f"{'=' * 140}")

print(f"\n  {'F:D':>8s}  {'VM edge':>9s} {'VM eff':>8s} {'VM bear':>8s}  "
      f"{'VN edge':>9s} {'VN eff':>8s} {'VN bear':>8s}  {'Winner':>7s} {'vs CUR':>8s}")
print(f"  {'=' * 95}")

# ACC_CUR reference
cur_eff = np.mean([master[tk]['ACC_CUR']['efficiency'] for tk in tks])
cur_edge = np.mean([master[tk]['ACC_CUR']['edge'] for tk in tks])
print(f"  {'CUR':>8s}  {cur_edge:>+8.2f}% {cur_eff:>+7.3f} {'':>8s}  "
      f"{'':>9s} {'':>8s} {'':>8s}  {'base':>7s}")
print(f"  {'-' * 95}")

summary_m = {}
summary_n = {}

for fw in F_WEIGHTS:
    nm = f'VM{int(fw*100):02d}'
    nn = f'VN{int(fw*100):02d}'

    m_edge = np.mean([master[tk][nm]['edge'] for tk in tks])
    m_eff = np.mean([master[tk][nm]['efficiency'] for tk in tks])
    m_bear = np.mean([master[tk][nm]['bear_edge'] for tk in tks])
    m_bull = np.mean([master[tk][nm]['bull_edge'] for tk in tks])
    m_worst = np.mean([master[tk][nm]['worst_diff'] for tk in tks])
    m_sigs = np.mean([master[tk][nm]['sig_count'] for tk in tks])
    m_wins = sum(master[tk][nm]['wins'] for tk in tks)
    m_losses = sum(master[tk][nm]['losses'] for tk in tks)

    n_edge = np.mean([master[tk][nn]['edge'] for tk in tks])
    n_eff = np.mean([master[tk][nn]['efficiency'] for tk in tks])
    n_bear = np.mean([master[tk][nn]['bear_edge'] for tk in tks])
    n_bull = np.mean([master[tk][nn]['bull_edge'] for tk in tks])
    n_worst = np.mean([master[tk][nn]['worst_diff'] for tk in tks])
    n_sigs = np.mean([master[tk][nn]['sig_count'] for tk in tks])
    n_wins = sum(master[tk][nn]['wins'] for tk in tks)
    n_losses = sum(master[tk][nn]['losses'] for tk in tks)

    summary_m[fw] = {'edge': m_edge, 'eff': m_eff, 'bear': m_bear, 'bull': m_bull,
                     'worst': m_worst, 'sigs': m_sigs, 'wins': m_wins, 'losses': m_losses}
    summary_n[fw] = {'edge': n_edge, 'eff': n_eff, 'bear': n_bear, 'bull': n_bull,
                     'worst': n_worst, 'sigs': n_sigs, 'wins': n_wins, 'losses': n_losses}

    winner = "MULT" if m_eff >= n_eff else "NO_M"
    best_eff = max(m_eff, n_eff)
    delta = best_eff - cur_eff

    fd = f"{int(fw*100)}:{int((1-fw)*100)}"
    marker = ""
    if delta > 0.06: marker = "***"
    elif delta > 0.04: marker = "**"
    elif delta > 0.02: marker = "*"

    print(f"  {fd:>8s}  {m_edge:>+8.2f}% {m_eff:>+7.3f} {m_bear:>+7.2f}%  "
          f"{n_edge:>+8.2f}% {n_eff:>+7.3f} {n_bear:>+7.2f}%  {winner:>7s} {delta:>+7.3f} {marker}")


# ===============================================================
# SECTION 2: Per-ticker — MULT vs NO_MULT winners
# ===============================================================
print(f"\n{'=' * 140}")
print(f"  SECTION 2: Per-ticker — Multiplier 유/무 비교 (각 티커별 최적 가중치)")
print(f"{'=' * 140}")

print(f"\n  {'Ticker':<7s} {'CUR eff':>8s}  {'Best M':>10s} {'M eff':>8s}  "
      f"{'Best N':>10s} {'N eff':>8s}  {'Overall':>8s}")
print(f"  {'=' * 75}")

mult_total = 0; nomult_total = 0; cur_total = 0

for tk in tks:
    c_eff = master[tk]['ACC_CUR']['efficiency']

    # Best with mult
    best_m_name = None; best_m_eff = -999
    for fw in F_WEIGHTS:
        nm = f'VM{int(fw*100):02d}'
        eff = master[tk][nm]['efficiency']
        if eff > best_m_eff:
            best_m_eff = eff; best_m_name = nm

    # Best without mult
    best_n_name = None; best_n_eff = -999
    for fw in F_WEIGHTS:
        nn = f'VN{int(fw*100):02d}'
        eff = master[tk][nn]['efficiency']
        if eff > best_n_eff:
            best_n_eff = eff; best_n_name = nn

    if c_eff >= best_m_eff and c_eff >= best_n_eff:
        overall = "CUR"; cur_total += 1
    elif best_m_eff >= best_n_eff:
        overall = "MULT"; mult_total += 1
    else:
        overall = "NO_MULT"; nomult_total += 1

    print(f"  {tk:<7s} {c_eff:>+7.3f}  {best_m_name:>10s} {best_m_eff:>+7.3f}  "
          f"{best_n_name:>10s} {best_n_eff:>+7.3f}  {overall:>8s}")

print(f"\n  CUR: {cur_total}/{len(tks)} | MULT: {mult_total}/{len(tks)} | NO_MULT: {nomult_total}/{len(tks)}")


# ===============================================================
# SECTION 3: Full ranking — all strategies
# ===============================================================
print(f"\n{'=' * 140}")
print(f"  SECTION 3: Full Ranking — Top 15")
print(f"{'=' * 140}")

all_names = ['ACC_CUR']
for fw in F_WEIGHTS:
    all_names.append(f'VM{int(fw*100):02d}')
    all_names.append(f'VN{int(fw*100):02d}')

all_summary = {}
all_summary['ACC_CUR'] = {
    'edge': cur_edge, 'eff': cur_eff,
    'bear': np.mean([master[tk]['ACC_CUR']['bear_edge'] for tk in tks]),
    'bull': np.mean([master[tk]['ACC_CUR']['bull_edge'] for tk in tks]),
    'sigs': np.mean([master[tk]['ACC_CUR']['sig_count'] for tk in tks]),
    'wins': sum(master[tk]['ACC_CUR']['wins'] for tk in tks),
    'losses': sum(master[tk]['ACC_CUR']['losses'] for tk in tks),
    'worst': np.mean([master[tk]['ACC_CUR']['worst_diff'] for tk in tks]),
}
for fw in F_WEIGHTS:
    for sfx, src in [('M', summary_m), ('N', summary_n)]:
        name = f'V{sfx}{int(fw*100):02d}'
        all_summary[name] = src[fw]

ranked = sorted(all_names, key=lambda x: all_summary[x]['eff'], reverse=True)

print(f"\n  {'Rank':>4s} {'Strategy':<10s} {'Mult':>5s} {'Sigs':>5s} {'Edge':>8s} "
      f"{'Bear':>7s} {'Bull':>7s} {'Worst':>7s} {'Effic':>8s} {'W/L':>7s} {'vs CUR':>8s}")
print(f"  {'=' * 95}")

for i, name in enumerate(ranked[:15]):
    s = all_summary[name]
    d = s['eff'] - cur_eff
    is_mult = "YES" if name.startswith('VM') else ("YES*" if name == 'ACC_CUR' else "NO")
    marker = ""
    if name == 'ACC_CUR': marker = " <-- CUR"
    elif d > 0.06: marker = " ***"
    elif d > 0.04: marker = " **"

    print(f"  {i+1:>4d} {name:<10s} {is_mult:>5s} {s['sigs']:>4.0f} {s['edge']:>+7.2f}% "
          f"{s['bear']:>+6.2f}% {s['bull']:>+6.2f}% {s['worst']:>+6.2f}% {s['eff']:>+7.3f} "
          f"{s['wins']:>3d}/{s['losses']:<3d} {d:>+7.3f}{marker}")


# ===============================================================
# SECTION 4: QQQ/VOO Year-by-Year
# ===============================================================
print(f"\n{'=' * 140}")
print(f"  SECTION 4: QQQ & VOO Year-by-Year — CUR vs Best MULT vs Best NO_MULT")
print(f"{'=' * 140}")

# Best MULT and NO_MULT overall
best_mult = sorted([n for n in ranked if n.startswith('VM')],
                   key=lambda x: all_summary[x]['eff'], reverse=True)[0]
best_nomult = sorted([n for n in ranked if n.startswith('VN')],
                     key=lambda x: all_summary[x]['eff'], reverse=True)[0]

compare = ['ACC_CUR', best_mult, best_nomult]

for tk in ['QQQ', 'VOO']:
    if tk not in tks:
        continue
    print(f"\n  {tk}:")
    all_years = sorted(set(r['yr'] for r in master[tk]['ACC_CUR']['yr_results']))

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
# SECTION 5: Final Verdict
# ===============================================================
print(f"\n{'=' * 140}")
print(f"  FINAL VERDICT")
print(f"{'=' * 140}")

bm = all_summary[best_mult]
bn = all_summary[best_nomult]

print(f"\n  Current (ACC+3ind+mult): eff={cur_eff:>+.3f}  edge={cur_edge:>+.2f}%")
print(f"\n  Best WITH multiplier:    {best_mult}")
print(f"    eff={bm['eff']:>+.3f} ({bm['eff']-cur_eff:>+.3f})  edge={bm['edge']:>+.2f}%  "
      f"bear={bm['bear']:>+.2f}%  bull={bm['bull']:>+.2f}%")
print(f"\n  Best WITHOUT multiplier: {best_nomult}")
print(f"    eff={bn['eff']:>+.3f} ({bn['eff']-cur_eff:>+.3f})  edge={bn['edge']:>+.2f}%  "
      f"bear={bn['bear']:>+.2f}%  bull={bn['bull']:>+.2f}%")

diff = bm['eff'] - bn['eff']
if diff > 0.02:
    print(f"\n  → Multiplier 유지가 더 좋음 (차이: {diff:>+.3f})")
elif diff < -0.02:
    print(f"\n  → Multiplier 제거가 더 좋음 (차이: {diff:>+.3f})")
else:
    print(f"\n  → 차이 미미 ({diff:>+.3f}). Multiplier 유/무 큰 영향 없음")

print(f"\n{'=' * 140}")
print("  Done.")
