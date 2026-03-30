"""
V4 VN60 Full Backtest & Analysis
=================================
VN60: 속도 기반 60%Force + 40%Div, S_conc 제거, Multiplier 제거
vs ACC_CUR (현재): 가속도 기반 45F+30D+25C, Multiplier 있음

상세 분석:
  1. 전체 14 티커 백테스트 비교
  2. 연도별 수익률 전체 출력
  3. 신호 품질 분석 (90d forward return)
  4. 신호 타이밍 비교 (어떤 신호가 다른가)
  5. 최종 자산 비교
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


def calc_force_vel(df, fast=12, slow=26, signal=9):
    p_vel = df['Close'].pct_change().fillna(0)
    v_norm = df['Volume'] / df['Volume'].rolling(20).mean().replace(0, np.nan).fillna(df['Volume'])
    force = v_norm * p_vel
    fm = force.ewm(span=fast, adjust=False).mean() - force.ewm(span=slow, adjust=False).mean()
    fs = fm.ewm(span=signal, adjust=False).mean()
    return (fm - fs).rename('PV_Force_Hist')


def calc_force_acc(df, fast=12, slow=26, signal=9):
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
    pv_fh_vel = calc_force_vel(df)
    pv_fh_acc = calc_force_acc(df)

    raw_div = np.array([np.clip(pv_div.iloc[i] / 3, -1, 1) for i in range(n)])
    consec = np.ones(n)
    for i in range(1, n):
        if (raw_div[i] > 0 and raw_div[i-1] > 0) or \
           (raw_div[i] < 0 and raw_div[i-1] < 0):
            consec[i] = consec[i-1] + 1
        else:
            consec[i] = 1
    return raw_div, consec, pv_conc, pv_fh_vel, pv_fh_acc


def calc_score_cur(df, raw_div, consec, pv_conc, pv_fh_acc, w=20, divgate_days=3):
    """현재: ACC + 3ind + multiplier"""
    n = len(df)
    scores = np.zeros(n)
    for i in range(max(60, w), n):
        s_div = raw_div[i] if consec[i] >= divgate_days else 0.0
        s_conc = pv_conc.iloc[i]
        fhr_std = pv_fh_acc.iloc[max(0, i - w):i].std() + 1e-10
        s_force = np.clip(pv_fh_acc.iloc[i] / (2 * fhr_std), -1, 1)
        dire = 0.45 * s_force + 0.30 * s_div + 0.25 * s_conc
        act = sum([abs(s_div) > 0.1, abs(s_conc) > 0.1, abs(s_force) > 0.1])
        mm = {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.2}
        scores[i] = dire * mm.get(act, 1.0)
    return pd.Series(scores, index=df.index)


def calc_score_vn60(df, raw_div, consec, pv_fh_vel, w=20, divgate_days=3):
    """VN60: VEL + 2ind(60:40) + no multiplier"""
    n = len(df)
    scores = np.zeros(n)
    for i in range(max(60, w), n):
        s_div = raw_div[i] if consec[i] >= divgate_days else 0.0
        fhr_std = pv_fh_vel.iloc[max(0, i - w):i].std() + 1e-10
        s_force = np.clip(pv_fh_vel.iloc[i] / (2 * fhr_std), -1, 1)
        dire = 0.60 * s_force + 0.40 * s_div
        scores[i] = dire  # no multiplier
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


def simulate_full(close, close_2x, buy_signals, dates):
    """Full simulation returning detailed yearly + final portfolio data."""
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
    sig_count = 0; total_deposited = 0.0
    sig_2x_invested = 0.0; sig_1x_invested = 0.0
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
                'deposits': 0.0, 'sigs_this_yr': 0,
            }
            prev_yr = yr

        cash_a += MONTHLY_DEPOSIT; cash_b += MONTHLY_DEPOSIT
        yr_data[yr]['deposits'] += MONTHLY_DEPOSIT
        total_deposited += MONTHLY_DEPOSIT

        for day_idx in range(fi, li + 1):
            if day_idx in buy_signals and cash_a > 1.0:
                amt = cash_a * SIGNAL_BUY_PCT
                sh_2x_a += amt / close_2x[day_idx]
                cash_a -= amt
                sig_count += 1
                sig_2x_invested += amt
                yr_data[yr]['sigs_this_yr'] += 1

        if cash_a > 1.0:
            sig_1x_invested += cash_a
            sh_1x_a += cash_a / close[li]; cash_a = 0.0
        if cash_b > 1.0:
            sh_1x_b += cash_b / close[li]; cash_b = 0.0

    yr_data[prev_yr]['end_a'] = sh_1x_a * close[n-1] + sh_2x_a * close_2x[n-1] + cash_a
    yr_data[prev_yr]['end_b'] = sh_1x_b * close[n-1] + cash_b

    final_a = sh_1x_a * close[n-1] + sh_2x_a * close_2x[n-1] + cash_a
    final_b = sh_1x_b * close[n-1] + cash_b

    yr_results = []
    for yr in sorted(yr_data.keys()):
        yd = yr_data[yr]
        da = yd['start_a'] + yd['deposits'] * 0.5
        db = yd['start_b'] + yd['deposits'] * 0.5
        ra = ((yd['end_a'] - yd['start_a'] - yd['deposits']) / da * 100) if da > 10 else 0
        rb = ((yd['end_b'] - yd['start_b'] - yd['deposits']) / db * 100) if db > 10 else 0
        yr_results.append({
            'yr': yr, 'ret_a': ra, 'ret_b': rb, 'diff': ra - rb,
            'end_a': yd['end_a'], 'end_b': yd['end_b'],
            'sigs': yd['sigs_this_yr'],
        })

    worst_diff = min(r['diff'] for r in yr_results)
    bull_yrs = [r for r in yr_results if r['ret_b'] > 5]
    bear_yrs = [r for r in yr_results if r['ret_b'] < -5]
    edge = np.mean([r['ret_a'] for r in yr_results]) - \
           np.mean([r['ret_b'] for r in yr_results])

    return {
        'edge': edge,
        'final_a': final_a, 'final_b': final_b,
        'total_deposited': total_deposited,
        'sig_count': sig_count,
        'sig_2x_pct': sig_2x_invested / total_deposited * 100 if total_deposited > 0 else 0,
        'wins': sum(1 for r in yr_results if r['diff'] > 0.5),
        'losses': sum(1 for r in yr_results if r['diff'] < -0.5),
        'worst_diff': worst_diff,
        'bull_edge': np.mean([r['diff'] for r in bull_yrs]) if bull_yrs else 0,
        'bear_edge': np.mean([r['diff'] for r in bear_yrs]) if bear_yrs else 0,
        'efficiency': edge / abs(worst_diff) if abs(worst_diff) > 0.1 else 0,
        'yr_results': yr_results,
    }


def get_forward_returns(close, dates, buy_indices):
    n = len(close)
    results = []
    for idx in sorted(buy_indices):
        fwd30 = (close[min(idx+30, n-1)] / close[idx] - 1) * 100 if idx+30 < n else None
        fwd60 = (close[min(idx+60, n-1)] / close[idx] - 1) * 100 if idx+60 < n else None
        fwd90 = (close[min(idx+90, n-1)] / close[idx] - 1) * 100 if idx+90 < n else None
        results.append({
            'idx': idx, 'date': dates[idx], 'price': close[idx],
            'fwd_30': fwd30, 'fwd_60': fwd60, 'fwd_90': fwd90,
        })
    return results


# ===============================================================
# Main
# ===============================================================
print("=" * 140)
print("  V4 VN60 FULL BACKTEST & ANALYSIS")
print("  VN60: VEL(속도) + 60%F:40%D + NoConc + NoMult")
print("  vs CUR: ACC(가속도) + 45%F:30%D:25%C + Mult{0:0.5,1:1.0,2:1.5,3:2.2}")
print("=" * 140)

master = {}
signals = {}
fwd = {}

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

    # CUR
    score_cur = calc_score_cur(df_s, raw_div, consec, pv_conc, pv_fh_acc,
                                w=V4_W, divgate_days=DIVGATE)
    buys_cur = get_buy_signals(df_s, score_cur)
    res_cur = simulate_full(close, close_2x, buys_cur, dates)
    fwd_cur = get_forward_returns(close, dates, buys_cur)

    # VN60
    score_vn60 = calc_score_vn60(df_s, raw_div, consec, pv_fh_vel,
                                  w=V4_W, divgate_days=DIVGATE)
    buys_vn60 = get_buy_signals(df_s, score_vn60)
    res_vn60 = simulate_full(close, close_2x, buys_vn60, dates)
    fwd_vn60 = get_forward_returns(close, dates, buys_vn60)

    master[tk] = {'CUR': res_cur, 'VN60': res_vn60}
    signals[tk] = {'CUR': buys_cur, 'VN60': buys_vn60}
    fwd[tk] = {'CUR': fwd_cur, 'VN60': fwd_vn60}

    print(f"CUR: sigs={res_cur['sig_count']:>3d} edge={res_cur['edge']:>+.1f}%  "
          f"VN60: sigs={res_vn60['sig_count']:>3d} edge={res_vn60['edge']:>+.1f}%  "
          f"delta={res_vn60['efficiency']-res_cur['efficiency']:>+.3f}")

tks = list(master.keys())


# ===============================================================
# SECTION 1: 전체 비교 요약
# ===============================================================
print(f"\n{'=' * 140}")
print(f"  SECTION 1: CUR vs VN60 — 전체 비교")
print(f"{'=' * 140}")

print(f"\n  {'Ticker':<7s} {'Sector':<8s}"
      f" {'CUR sig':>8s} {'CUR edge':>9s} {'CUR eff':>8s} {'CUR W/L':>8s}"
      f" {'VN60 sig':>9s} {'VN60 edge':>10s} {'VN60 eff':>9s} {'VN60 W/L':>9s}"
      f" {'dEdge':>7s} {'dEff':>7s} {'Winner':>7s}")
print(f"  {'=' * 125}")

for tk in tks:
    c = master[tk]['CUR']; v = master[tk]['VN60']
    de = v['edge'] - c['edge']
    deff = v['efficiency'] - c['efficiency']
    w = "VN60" if deff > 0.01 else ("CUR" if deff < -0.01 else "TIE")
    sect = TICKERS.get(tk, '')
    print(f"  {tk:<7s} {sect:<8s}"
          f" {c['sig_count']:>7d} {c['edge']:>+8.2f}% {c['efficiency']:>+7.3f} {c['wins']:>3d}/{c['losses']:<3d}"
          f" {v['sig_count']:>8d} {v['edge']:>+9.2f}% {v['efficiency']:>+8.3f} {v['wins']:>3d}/{v['losses']:<3d}"
          f" {de:>+6.2f} {deff:>+6.3f} {w:>7s}")

# Averages
print(f"  {'-' * 125}")
for label, key in [('AVG', None)]:
    c_edge = np.mean([master[tk]['CUR']['edge'] for tk in tks])
    c_eff = np.mean([master[tk]['CUR']['efficiency'] for tk in tks])
    c_sigs = np.mean([master[tk]['CUR']['sig_count'] for tk in tks])
    v_edge = np.mean([master[tk]['VN60']['edge'] for tk in tks])
    v_eff = np.mean([master[tk]['VN60']['efficiency'] for tk in tks])
    v_sigs = np.mean([master[tk]['VN60']['sig_count'] for tk in tks])
    print(f"  {'AVG':<7s} {'':8s}"
          f" {c_sigs:>7.0f} {c_edge:>+8.2f}% {c_eff:>+7.3f} {'':>8s}"
          f" {v_sigs:>8.0f} {v_edge:>+9.2f}% {v_eff:>+8.3f} {'':>9s}"
          f" {v_edge-c_edge:>+6.2f} {v_eff-c_eff:>+6.3f}")

vn60_wins = sum(1 for tk in tks if master[tk]['VN60']['efficiency'] > master[tk]['CUR']['efficiency'] + 0.01)
cur_wins = sum(1 for tk in tks if master[tk]['CUR']['efficiency'] > master[tk]['VN60']['efficiency'] + 0.01)
ties = len(tks) - vn60_wins - cur_wins
print(f"\n  VN60 wins: {vn60_wins}/{len(tks)} | CUR wins: {cur_wins}/{len(tks)} | Ties: {ties}/{len(tks)}")


# ===============================================================
# SECTION 2: 최종 자산 비교
# ===============================================================
print(f"\n{'=' * 140}")
print(f"  SECTION 2: Final Portfolio Value ($500/month 적립)")
print(f"{'=' * 140}")

print(f"\n  {'Ticker':<7s} {'Deposited':>10s} {'CUR(2x+1x)':>12s} {'CUR(1xDCA)':>12s} "
      f"{'VN60(2x+1x)':>13s} {'VN60(1xDCA)':>12s} {'CUR gain':>10s} {'VN60 gain':>10s} {'Better':>8s}")
print(f"  {'=' * 110}")

for tk in tks:
    c = master[tk]['CUR']; v = master[tk]['VN60']
    dep = c['total_deposited']
    cg = (c['final_a'] / dep - 1) * 100
    vg = (v['final_a'] / dep - 1) * 100
    better = "VN60" if v['final_a'] > c['final_a'] else "CUR"
    print(f"  {tk:<7s} ${dep:>9,.0f} ${c['final_a']:>11,.0f} ${c['final_b']:>11,.0f} "
          f"${v['final_a']:>12,.0f} ${v['final_b']:>11,.0f} {cg:>+9.1f}% {vg:>+9.1f}% {better:>8s}")


# ===============================================================
# SECTION 3: 연도별 Edge (CUR vs VN60) — 모든 티커
# ===============================================================
print(f"\n{'=' * 140}")
print(f"  SECTION 3: Year-by-Year Edge — All Tickers")
print(f"{'=' * 140}")

for tk in tks:
    c = master[tk]['CUR']; v = master[tk]['VN60']
    print(f"\n  {tk} ({TICKERS.get(tk, '')}):")
    print(f"  {'Year':>6s} {'CUR ret':>8s} {'1xDCA':>8s} {'CUR edge':>9s} "
          f"{'VN60 ret':>9s} {'VN60 edge':>10s} {'VN60-CUR':>9s} {'Sigs C/V':>9s}")
    print(f"  {'=' * 80}")

    for cr, vr in zip(c['yr_results'], v['yr_results']):
        delta = vr['diff'] - cr['diff']
        marker = " **" if abs(delta) > 3 else (" *" if abs(delta) > 1 else "")
        # Find VN60 sig count for this year
        vs = vr.get('sigs', 0)
        cs = cr.get('sigs', 0)
        print(f"  {cr['yr']:>6d} {cr['ret_a']:>+7.1f}% {cr['ret_b']:>+7.1f}% {cr['diff']:>+8.1f}% "
              f"{vr['ret_a']:>+8.1f}% {vr['diff']:>+9.1f}% {delta:>+8.1f}%{marker} {cs:>3d}/{vs:<3d}")

    print(f"  {'-' * 80}")
    c_avg = np.mean([r['diff'] for r in c['yr_results']])
    v_avg = np.mean([r['diff'] for r in v['yr_results']])
    print(f"  {'AVG':>6s} {'':>8s} {'':>8s} {c_avg:>+8.1f}% {'':>9s} {v_avg:>+9.1f}% {v_avg-c_avg:>+8.1f}%")


# ===============================================================
# SECTION 4: 신호 품질 분석 (Forward Return)
# ===============================================================
print(f"\n{'=' * 140}")
print(f"  SECTION 4: Signal Quality — 90d Forward Return Analysis")
print(f"{'=' * 140}")

print(f"\n  {'Ticker':<7s} {'CUR':>5s}  {'avg30':>7s} {'avg60':>7s} {'avg90':>7s} {'win90':>6s}"
      f"  {'VN60':>5s}  {'avg30':>7s} {'avg60':>7s} {'avg90':>7s} {'win90':>6s}"
      f"  {'Better':>7s}")
print(f"  {'=' * 100}")

total_cur_fwd = []; total_vn60_fwd = []

for tk in tks:
    fc = fwd[tk]['CUR']; fv = fwd[tk]['VN60']

    def calc_stats(flist):
        f30 = [f['fwd_30'] for f in flist if f['fwd_30'] is not None]
        f60 = [f['fwd_60'] for f in flist if f['fwd_60'] is not None]
        f90 = [f['fwd_90'] for f in flist if f['fwd_90'] is not None]
        a30 = np.mean(f30) if f30 else 0
        a60 = np.mean(f60) if f60 else 0
        a90 = np.mean(f90) if f90 else 0
        w90 = sum(1 for x in f90 if x > 0) / len(f90) * 100 if f90 else 0
        return a30, a60, a90, w90, f90

    ca30, ca60, ca90, cw90, cf90 = calc_stats(fc)
    va30, va60, va90, vw90, vf90 = calc_stats(fv)
    total_cur_fwd.extend(cf90)
    total_vn60_fwd.extend(vf90)

    better = "VN60" if va90 > ca90 else "CUR"
    print(f"  {tk:<7s} {len(fc):>4d}  {ca30:>+6.1f}% {ca60:>+6.1f}% {ca90:>+6.1f}% {cw90:>5.0f}%"
          f"  {len(fv):>4d}  {va30:>+6.1f}% {va60:>+6.1f}% {va90:>+6.1f}% {vw90:>5.0f}%"
          f"  {better:>7s}")

print(f"  {'-' * 100}")
ca = np.mean(total_cur_fwd) if total_cur_fwd else 0
cw = sum(1 for x in total_cur_fwd if x > 0) / len(total_cur_fwd) * 100 if total_cur_fwd else 0
va = np.mean(total_vn60_fwd) if total_vn60_fwd else 0
vw = sum(1 for x in total_vn60_fwd if x > 0) / len(total_vn60_fwd) * 100 if total_vn60_fwd else 0
print(f"  {'TOTAL':<7s} {len(total_cur_fwd):>4d}  {'':>7s} {'':>7s} {ca:>+6.1f}% {cw:>5.0f}%"
      f"  {len(total_vn60_fwd):>4d}  {'':>7s} {'':>7s} {va:>+6.1f}% {vw:>5.0f}%")


# ===============================================================
# SECTION 5: 신호 겹침/차이 분석
# ===============================================================
print(f"\n{'=' * 140}")
print(f"  SECTION 5: Signal Overlap — CUR vs VN60")
print(f"{'=' * 140}")

print(f"\n  {'Ticker':<7s} {'CUR':>5s} {'VN60':>5s} {'Shared':>7s} {'OnlyCUR':>8s} {'OnlyVN60':>9s} "
      f"{'OnlyCUR avg90':>14s} {'OnlyVN60 avg90':>15s}")
print(f"  {'=' * 85}")

total_shared = 0; total_only_c = 0; total_only_v = 0
all_only_c_fwd = []; all_only_v_fwd = []

for tk in tks:
    sc = signals[tk]['CUR']; sv = signals[tk]['VN60']
    shared = sc & sv
    only_c = sc - sv
    only_v = sv - sc

    total_shared += len(shared)
    total_only_c += len(only_c)
    total_only_v += len(only_v)

    # Forward returns for only_c and only_v
    close = None
    fc_map = {f['idx']: f['fwd_90'] for f in fwd[tk]['CUR']}
    fv_map = {f['idx']: f['fwd_90'] for f in fwd[tk]['VN60']}

    oc_fwd = [fc_map[i] for i in only_c if i in fc_map and fc_map[i] is not None]
    ov_fwd = [fv_map[i] for i in only_v if i in fv_map and fv_map[i] is not None]
    all_only_c_fwd.extend(oc_fwd)
    all_only_v_fwd.extend(ov_fwd)

    oc_avg = f"{np.mean(oc_fwd):>+6.1f}%" if oc_fwd else "   N/A"
    ov_avg = f"{np.mean(ov_fwd):>+6.1f}%" if ov_fwd else "   N/A"

    print(f"  {tk:<7s} {len(sc):>5d} {len(sv):>5d} {len(shared):>7d} {len(only_c):>8d} {len(only_v):>9d} "
          f"{oc_avg:>14s} {ov_avg:>15s}")

print(f"  {'-' * 85}")
oc_total_avg = f"{np.mean(all_only_c_fwd):>+.1f}%" if all_only_c_fwd else "N/A"
ov_total_avg = f"{np.mean(all_only_v_fwd):>+.1f}%" if all_only_v_fwd else "N/A"
print(f"  {'TOTAL':<7s} {'':>5s} {'':>5s} {total_shared:>7d} {total_only_c:>8d} {total_only_v:>9d} "
      f"{oc_total_avg:>14s} {ov_total_avg:>15s}")

if all_only_c_fwd and all_only_v_fwd:
    oc_wr = sum(1 for x in all_only_c_fwd if x > 0) / len(all_only_c_fwd) * 100
    ov_wr = sum(1 for x in all_only_v_fwd if x > 0) / len(all_only_v_fwd) * 100
    print(f"\n  Signals ONLY in CUR (removed by VN60): avg={np.mean(all_only_c_fwd):>+.1f}%  "
          f"winrate={oc_wr:.0f}%  n={len(all_only_c_fwd)}")
    print(f"  Signals ONLY in VN60 (added by VN60):   avg={np.mean(all_only_v_fwd):>+.1f}%  "
          f"winrate={ov_wr:.0f}%  n={len(all_only_v_fwd)}")


# ===============================================================
# SECTION 6: Bear vs Bull 구간 성과
# ===============================================================
print(f"\n{'=' * 140}")
print(f"  SECTION 6: Bear vs Bull Market Performance")
print(f"{'=' * 140}")

print(f"\n  {'Ticker':<7s} {'CUR Bear':>9s} {'CUR Bull':>9s} {'VN60 Bear':>10s} {'VN60 Bull':>10s} "
      f"{'Bear diff':>10s} {'Bull diff':>10s}")
print(f"  {'=' * 75}")

for tk in tks:
    c = master[tk]['CUR']; v = master[tk]['VN60']
    print(f"  {tk:<7s} {c['bear_edge']:>+8.2f}% {c['bull_edge']:>+8.2f}% "
          f"{v['bear_edge']:>+9.2f}% {v['bull_edge']:>+9.2f}% "
          f"{v['bear_edge']-c['bear_edge']:>+9.2f}% {v['bull_edge']-c['bull_edge']:>+9.2f}%")

print(f"  {'-' * 75}")
c_bear = np.mean([master[tk]['CUR']['bear_edge'] for tk in tks])
c_bull = np.mean([master[tk]['CUR']['bull_edge'] for tk in tks])
v_bear = np.mean([master[tk]['VN60']['bear_edge'] for tk in tks])
v_bull = np.mean([master[tk]['VN60']['bull_edge'] for tk in tks])
print(f"  {'AVG':<7s} {c_bear:>+8.2f}% {c_bull:>+8.2f}% "
      f"{v_bear:>+9.2f}% {v_bull:>+9.2f}% "
      f"{v_bear-c_bear:>+9.2f}% {v_bull-c_bull:>+9.2f}%")


# ===============================================================
# SECTION 7: 2x 레버리지 배분 비율
# ===============================================================
print(f"\n{'=' * 140}")
print(f"  SECTION 7: 2x Leverage Allocation")
print(f"{'=' * 140}")

print(f"\n  {'Ticker':<7s} {'CUR sigs':>9s} {'CUR 2x%':>8s} {'VN60 sigs':>10s} {'VN60 2x%':>9s}")
print(f"  {'=' * 50}")

for tk in tks:
    c = master[tk]['CUR']; v = master[tk]['VN60']
    print(f"  {tk:<7s} {c['sig_count']:>8d} {c['sig_2x_pct']:>7.1f}% "
          f"{v['sig_count']:>9d} {v['sig_2x_pct']:>8.1f}%")


# ===============================================================
# FINAL SUMMARY
# ===============================================================
print(f"\n{'=' * 140}")
print(f"  FINAL SUMMARY: VN60 vs CUR")
print(f"{'=' * 140}")

c_eff = np.mean([master[tk]['CUR']['efficiency'] for tk in tks])
v_eff = np.mean([master[tk]['VN60']['efficiency'] for tk in tks])
c_edge = np.mean([master[tk]['CUR']['edge'] for tk in tks])
v_edge = np.mean([master[tk]['VN60']['edge'] for tk in tks])
c_sigs = np.mean([master[tk]['CUR']['sig_count'] for tk in tks])
v_sigs = np.mean([master[tk]['VN60']['sig_count'] for tk in tks])

print(f"""
  +--------------------------------------------------+
  |  Metric              CUR(현재)        VN60        |
  +--------------------------------------------------+
  |  Formula       45F+30D+25C     60F+40D(no C)     |
  |  Force type    Acceleration    Velocity           |
  |  Multiplier    YES             NO                 |
  |  Avg Signals   {c_sigs:>6.0f}          {v_sigs:>6.0f}             |
  |  Avg Edge      {c_edge:>+6.2f}%         {v_edge:>+6.2f}%            |
  |  Efficiency    {c_eff:>+6.3f}         {v_eff:>+6.3f}            |
  |  Bear Edge     {c_bear:>+6.2f}%         {v_bear:>+6.2f}%            |
  |  Bull Edge     {c_bull:>+6.2f}%         {v_bull:>+6.2f}%            |
  |  Improvement   ---             {v_eff-c_eff:>+6.3f} eff       |
  |                                {v_edge-c_edge:>+6.2f}% edge      |
  +--------------------------------------------------+
""")

print(f"{'=' * 140}")
print("  Done.")
