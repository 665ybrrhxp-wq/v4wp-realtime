"""
V4 No-Conc Test: S_conc 제거 후 S_Force + S_Div만으로 Score 계산
================================================================
S_conc(가격-거래량 상관관계)는 바닥 신호에서 음수가 자연스러운데,
이것이 Score를 깎아먹는 역효과를 낸다.
→ S_conc를 제거하고 F+D만으로 계산하면 어떤가?

테스트 공식:
  CURRENT:    0.45*F + 0.30*D + 0.25*C  (현재)
  NO_C_KEEP:  0.45*F + 0.30*D           (C 제거, 가중치 유지)
  NO_C_PROP:  0.60*F + 0.40*D           (비례 재분배)
  NO_C_EQUAL: 0.50*F + 0.50*D           (동일 가중치)
  NO_C_F70:   0.70*F + 0.30*D           (Force 중심)
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

FORMULA_NAMES = ['CURRENT', 'NO_C_KEEP', 'NO_C_PROP', 'NO_C_EQUAL', 'NO_C_F70']
FORMULA_DESC = {
    'CURRENT':    '0.45F + 0.30D + 0.25C',
    'NO_C_KEEP':  '0.45F + 0.30D (drop C)',
    'NO_C_PROP':  '0.60F + 0.40D (proportional)',
    'NO_C_EQUAL': '0.50F + 0.50D (equal)',
    'NO_C_F70':   '0.70F + 0.30D (force-heavy)',
}


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


def calc_score_variant(df, raw_div, consec, pv_conc, pv_fh,
                       w=20, divgate_days=3, formula='CURRENT'):
    n = len(df)
    scores = np.zeros(n)
    for i in range(max(60, w), n):
        s_div = raw_div[i] if consec[i] >= divgate_days else 0.0
        s_conc = pv_conc.iloc[i]
        fhr_std = pv_fh.iloc[max(0, i - w):i].std() + 1e-10
        s_force = np.clip(pv_fh.iloc[i] / (2 * fhr_std), -1, 1)

        if formula == 'CURRENT':
            dire = 0.45 * s_force + 0.30 * s_div + 0.25 * s_conc
            # Activity: all 3 components
            act = sum([abs(s_div) > 0.1, abs(s_conc) > 0.1, abs(s_force) > 0.1])
            mm = {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.2}
        else:
            # No-conc formulas
            if formula == 'NO_C_KEEP':
                dire = 0.45 * s_force + 0.30 * s_div
            elif formula == 'NO_C_PROP':
                dire = 0.60 * s_force + 0.40 * s_div
            elif formula == 'NO_C_EQUAL':
                dire = 0.50 * s_force + 0.50 * s_div
            elif formula == 'NO_C_F70':
                dire = 0.70 * s_force + 0.30 * s_div

            # Activity: only 2 components (F, D)
            act = sum([abs(s_div) > 0.1, abs(s_force) > 0.1])
            mm = {0: 0.5, 1: 1.0, 2: 2.2}  # 0->0.5, 1->1.0, 2(both active)->2.2

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
                'deposits': 0.0, 'end_a': 0, 'end_b': 0,
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

    final_a = pf_a(n - 1); final_b = pf_b(n - 1)

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
        'final_a': final_a, 'final_b': final_b,
        'wins': sum(1 for r in yr_results if r['diff'] > 0.5),
        'losses': sum(1 for r in yr_results if r['diff'] < -0.5),
        'worst_diff': worst_diff,
        'sig_count': sig_count,
        'bull_edge': np.mean([r['diff'] for r in bull_yrs]) if bull_yrs else 0,
        'bear_edge': np.mean([r['diff'] for r in bear_yrs]) if bear_yrs else 0,
        'efficiency': edge / abs(worst_diff) if abs(worst_diff) > 0.1 else 0,
        'yr_results': yr_results,
    }


def get_forward_returns(close, dates, buy_indices):
    n = len(close)
    results = []
    for idx in sorted(buy_indices):
        fwd = (close[min(idx + 90, n - 1)] / close[idx] - 1) * 100 \
              if idx + 90 < n else None
        results.append({'idx': idx, 'date': dates[idx], 'price': close[idx], 'fwd_90': fwd})
    return results


# ===============================================================
# Main
# ===============================================================
print("=" * 140)
print("  V4 NO-CONC TEST: S_conc 제거, S_Force + S_Div만으로 Score 계산")
print("  CURRENT: 0.45F+0.30D+0.25C | NO_C_KEEP: 0.45F+0.30D | NO_C_PROP: 0.60F+0.40D")
print("  NO_C_EQUAL: 0.50F+0.50D | NO_C_F70: 0.70F+0.30D")
print("=" * 140)

master = {}
signal_sets = {}
fwd_data = {}

for tk, sector in TICKERS.items():
    print(f"  {tk}...", end=" ", flush=True)
    df = download_max(tk)
    if df is None or len(df) < 300:
        print("SKIP")
        continue

    try:
        df_s = smooth_earnings_volume(df, ticker=tk)
    except Exception:
        df_s = df.copy()

    close = df_s['Close'].values
    dates = df_s.index
    close_2x = build_synthetic_2x(close)

    raw_div, consec, pv_conc, pv_fh = precompute_components(
        df_s, w=V4_W, divgate_days=DIVGATE)

    master[tk] = {}
    signal_sets[tk] = {}
    fwd_data[tk] = {}

    for fname in FORMULA_NAMES:
        score = calc_score_variant(
            df_s, raw_div, consec, pv_conc, pv_fh,
            w=V4_W, divgate_days=DIVGATE, formula=fname)
        buys = get_buy_signals(df_s, score)
        res = simulate(close, close_2x, buys, dates)
        master[tk][fname] = res
        signal_sets[tk][fname] = buys
        fwd_data[tk][fname] = get_forward_returns(close, dates, buys)

    cur = master[tk]['CURRENT']
    print(f"sigs={cur['sig_count']:>3d}  edge={cur['edge']:>+.1f}%  ", end="")
    for fn in FORMULA_NAMES[1:]:
        r = master[tk][fn]
        print(f"{fn}={r['sig_count']}/{r['edge']:>+.1f}  ", end="")
    print()

tks = list(master.keys())


# ===============================================================
# SECTION 1: Signal Count & Edge
# ===============================================================
print(f"\n{'=' * 140}")
print(f"  SECTION 1: Signal Count & Leverage Edge per Formula")
print(f"{'=' * 140}")

print(f"\n  {'Ticker':<7s}", end="")
for fn in FORMULA_NAMES:
    print(f"  {fn:>16s}", end="")
print()
print(f"  {'':7s}", end="")
for fn in FORMULA_NAMES:
    print(f"  {'sigs/edge%p':>16s}", end="")
print()
print(f"  {'=' * 95}")

for tk in tks:
    line = f"  {tk:<7s}"
    for fn in FORMULA_NAMES:
        r = master[tk][fn]
        line += f"  {r['sig_count']:>5d}/{r['edge']:>+6.1f}%p"
    print(line)

print(f"  {'-' * 95}")
line = f"  {'AVG':<7s}"
for fn in FORMULA_NAMES:
    avg_sig = np.mean([master[tk][fn]['sig_count'] for tk in tks])
    avg_edge = np.mean([master[tk][fn]['edge'] for tk in tks])
    line += f"  {avg_sig:>5.0f}/{avg_edge:>+6.1f}%p"
print(line)


# ===============================================================
# SECTION 2: Summary — Efficiency + Bear/Bull
# ===============================================================
print(f"\n{'=' * 140}")
print(f"  SECTION 2: Efficiency (Edge/|WorstLoss|) + Bear/Bull Edge")
print(f"{'=' * 140}")

print(f"\n  {'Formula':<12s} {'Desc':<28s} {'AvgSig':>7s} {'AvgEdge':>9s} "
      f"{'Bear':>8s} {'Bull':>8s} {'Worst':>8s} {'Effic':>8s} {'W/L':>6s} {'vs CUR':>8s}")
print(f"  {'=' * 115}")

formula_summary = {}
for fn in FORMULA_NAMES:
    avg_edge = np.mean([master[tk][fn]['edge'] for tk in tks])
    avg_bear = np.mean([master[tk][fn]['bear_edge'] for tk in tks])
    avg_bull = np.mean([master[tk][fn]['bull_edge'] for tk in tks])
    avg_worst = np.mean([master[tk][fn]['worst_diff'] for tk in tks])
    avg_eff = np.mean([master[tk][fn]['efficiency'] for tk in tks])
    avg_sigs = np.mean([master[tk][fn]['sig_count'] for tk in tks])
    total_w = sum(master[tk][fn]['wins'] for tk in tks)
    total_l = sum(master[tk][fn]['losses'] for tk in tks)

    formula_summary[fn] = {
        'edge': avg_edge, 'bear': avg_bear, 'bull': avg_bull,
        'worst': avg_worst, 'eff': avg_eff, 'sigs': avg_sigs,
        'wins': total_w, 'losses': total_l,
    }

base = formula_summary['CURRENT']
for fn in FORMULA_NAMES:
    s = formula_summary[fn]
    delta_eff = s['eff'] - base['eff']
    marker = ""
    if fn != 'CURRENT':
        if delta_eff > 0.05: marker = " *** BEST ***"
        elif delta_eff > 0.02: marker = " ** BETTER **"
        elif delta_eff > 0.005: marker = " * mild + *"
        elif delta_eff < -0.05: marker = " [HARMFUL]"
        elif delta_eff < -0.02: marker = " [worse]"

    print(f"  {fn:<12s} {FORMULA_DESC[fn]:<28s} {s['sigs']:>6.0f} {s['edge']:>+8.2f}% "
          f"{s['bear']:>+7.2f}% {s['bull']:>+7.2f}% {s['worst']:>+7.2f}% {s['eff']:>+7.3f} "
          f"{s['wins']:>3d}/{s['losses']:<3d} {delta_eff:>+7.3f}{marker}")


# ===============================================================
# SECTION 3: Per-ticker Efficiency
# ===============================================================
print(f"\n{'=' * 140}")
print(f"  SECTION 3: Per-ticker Efficiency (higher = better risk-adjusted)")
print(f"{'=' * 140}")

print(f"\n  {'Ticker':<7s}", end="")
for fn in FORMULA_NAMES:
    print(f"  {fn:>12s}", end="")
print(f"  {'Best':>12s}")
print(f"  {'=' * 82}")

eff_winners = {fn: 0 for fn in FORMULA_NAMES}
for tk in tks:
    line = f"  {tk:<7s}"
    best_fn = 'CURRENT'
    best_eff = master[tk]['CURRENT']['efficiency']
    for fn in FORMULA_NAMES:
        eff = master[tk][fn]['efficiency']
        delta = eff - master[tk]['CURRENT']['efficiency']
        m = "+" if delta > 0.01 else ("-" if delta < -0.01 else " ")
        line += f"  {eff:>+7.3f}({m})"
        if eff > best_eff:
            best_eff = eff; best_fn = fn
    eff_winners[best_fn] += 1
    line += f"   {best_fn}"
    print(line)

print(f"\n  Formula wins:")
for fn in sorted(eff_winners.keys(), key=lambda x: eff_winners[x], reverse=True):
    if eff_winners[fn] > 0:
        print(f"    {fn:<12s}: {eff_winners[fn]:>2d}/{len(tks)} tickers")


# ===============================================================
# SECTION 4: Signal Difference Analysis
# ===============================================================
print(f"\n{'=' * 140}")
print(f"  SECTION 4: Signal Changes — What happens when we remove S_conc?")
print(f"{'=' * 140}")

for fn in FORMULA_NAMES[1:]:
    total_current = 0; total_new = 0
    shared = 0; only_current = 0; only_new = 0
    fwd_shared = []; fwd_removed = []; fwd_added = []

    for tk in tks:
        cur_set = signal_sets[tk]['CURRENT']
        new_set = signal_sets[tk][fn]
        total_current += len(cur_set)
        total_new += len(new_set)
        s = cur_set & new_set
        oc = cur_set - new_set
        on = new_set - cur_set
        shared += len(s)
        only_current += len(oc)
        only_new += len(on)

        cur_fwd = {f['idx']: f['fwd_90'] for f in fwd_data[tk]['CURRENT']}
        new_fwd = {f['idx']: f['fwd_90'] for f in fwd_data[tk][fn]}

        for idx in s:
            if idx in cur_fwd and cur_fwd[idx] is not None:
                fwd_shared.append(cur_fwd[idx])
        for idx in oc:
            if idx in cur_fwd and cur_fwd[idx] is not None:
                fwd_removed.append(cur_fwd[idx])
        for idx in on:
            if idx in new_fwd and new_fwd[idx] is not None:
                fwd_added.append(new_fwd[idx])

    def stats(lst):
        if not lst: return 0, 0, 0
        return np.mean(lst), sum(1 for x in lst if x > 0) / len(lst) * 100, len(lst)

    avg_s, wr_s, n_s = stats(fwd_shared)
    avg_r, wr_r, n_r = stats(fwd_removed)
    avg_a, wr_a, n_a = stats(fwd_added)

    print(f"\n  --- {fn} ({FORMULA_DESC[fn]}) ---")
    print(f"    CUR: {total_current} sigs | {fn}: {total_new} sigs | "
          f"Shared: {shared} | Removed: {only_current} | Added: {only_new}")
    print(f"    90d Forward Returns:")
    print(f"      Shared (n={n_s:>3d}):  avg={avg_s:>+6.1f}%  winrate={wr_s:.0f}%")
    if n_r > 0:
        bad_r = sum(1 for x in fwd_removed if x <= 0)
        print(f"      Removed (n={n_r:>3d}): avg={avg_r:>+6.1f}%  winrate={wr_r:.0f}%  "
              f"bad precision={bad_r}/{n_r} ({bad_r/n_r*100:.0f}%)")
    if n_a > 0:
        good_a = sum(1 for x in fwd_added if x > 0)
        print(f"      Added (n={n_a:>3d}):   avg={avg_a:>+6.1f}%  winrate={wr_a:.0f}%  "
              f"good signals={good_a}/{n_a} ({good_a/n_a*100:.0f}%)")


# ===============================================================
# SECTION 5: QQQ/VOO Year-by-Year
# ===============================================================
print(f"\n{'=' * 140}")
print(f"  SECTION 5: QQQ & VOO Year-by-Year Edge")
print(f"{'=' * 140}")

for tk in ['QQQ', 'VOO']:
    if tk not in tks:
        continue
    print(f"\n  {tk}:")
    all_years = sorted(set(r['yr'] for r in master[tk]['CURRENT']['yr_results']))
    print(f"  {'Year':>6s}", end="")
    for fn in FORMULA_NAMES:
        label = fn[:10]
        print(f"  {label:>10s}", end="")
    print()
    print(f"  {'=' * 65}")

    for yr in all_years:
        line = f"  {yr:>6d}"
        for fn in FORMULA_NAMES:
            yr_r = [r for r in master[tk][fn]['yr_results'] if r['yr'] == yr]
            if yr_r:
                line += f"  {yr_r[0]['diff']:>+9.1f}%"
            else:
                line += f"  {'N/A':>10s}"
        print(line)

    print(f"  {'-' * 65}")
    line = f"  {'AVG':>6s}"
    for fn in FORMULA_NAMES:
        avg = np.mean([r['diff'] for r in master[tk][fn]['yr_results']])
        line += f"  {avg:>+9.1f}%"
    print(line)


# ===============================================================
# SECTION 6: Final Verdict
# ===============================================================
print(f"\n{'=' * 140}")
print(f"  FINAL VERDICT")
print(f"{'=' * 140}")

ranked = sorted(FORMULA_NAMES,
                key=lambda fn: formula_summary[fn]['eff'],
                reverse=True)

for i, fn in enumerate(ranked):
    s = formula_summary[fn]
    de = s['edge'] - base['edge']
    deff = s['eff'] - base['eff']
    dsig = s['sigs'] - base['sigs']

    star = " <-- CURRENT" if fn == 'CURRENT' else ""
    print(f"\n  #{i+1} {fn} ({FORMULA_DESC[fn]}){star}")
    print(f"      Signals:    {s['sigs']:.0f}  Edge: {s['edge']:>+.2f}%p  "
          f"Efficiency: {s['eff']:>+.3f}  ({deff:>+.3f} vs CUR)")
    print(f"      Bear: {s['bear']:>+.2f}%  Bull: {s['bull']:>+.2f}%  "
          f"W/L: {s['wins']}/{s['losses']}")

print(f"\n{'=' * 140}")
print("  Done.")
