"""
V4 ConcGate Test: S_conc에도 DivGate처럼 duration 적용
=========================================================
현재: S_div만 3일 연속 같은 부호여야 활성화 (DivGate)
      S_conc는 매일 raw 값 그대로 사용

테스트: S_conc에도 N일 연속 같은 부호 조건 적용
        N=1 (현재), 2, 3, 5, 7, 10
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
    calc_v4_score, calc_v4_subindicators,
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

CONCGATE_DAYS = [1, 2, 3, 5, 7, 10]


def download_max(ticker):
    t = yf.Ticker(ticker)
    df = t.history(period='max', auto_adjust=True)
    if df is None or len(df) == 0:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def calc_v4_score_concgate(df, w=20, divgate_days=3, concgate_days=1):
    """V4 score with ConcGate: S_conc도 N일 연속 같은 부호여야 활성화."""
    n = len(df)
    pv_div = calc_pv_divergence(df, w)
    pv_conc = calc_pv_concordance(df, w)
    pv_fh = calc_pv_force_macd(df)

    # DivGate: 연속 같은 부호 일수
    raw_div = np.array([np.clip(pv_div.iloc[i] / 3, -1, 1) for i in range(n)])
    div_consec = np.ones(n)
    for i in range(1, n):
        if (raw_div[i] > 0 and raw_div[i-1] > 0) or \
           (raw_div[i] < 0 and raw_div[i-1] < 0):
            div_consec[i] = div_consec[i-1] + 1
        else:
            div_consec[i] = 1

    # ConcGate: S_conc 연속 같은 부호 일수
    raw_conc = np.array([pv_conc.iloc[i] for i in range(n)])
    conc_consec = np.ones(n)
    for i in range(1, n):
        if (raw_conc[i] > 0 and raw_conc[i-1] > 0) or \
           (raw_conc[i] < 0 and raw_conc[i-1] < 0):
            conc_consec[i] = conc_consec[i-1] + 1
        else:
            conc_consec[i] = 1

    scores = np.zeros(n)
    for i in range(max(60, w), n):
        s_div = raw_div[i] if div_consec[i] >= divgate_days else 0.0
        s_conc = raw_conc[i] if conc_consec[i] >= concgate_days else 0.0
        fhr_std = pv_fh.iloc[max(0, i - w):i].std() + 1e-10
        s_force = np.clip(pv_fh.iloc[i] / (2 * fhr_std), -1, 1)

        dire = 0.45 * s_force + 0.30 * s_div + 0.25 * s_conc
        act = sum([abs(s_div) > 0.1, abs(s_conc) > 0.1, abs(s_force) > 0.1])
        mm = {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.2}
        scores[i] = dire * mm.get(act, 1.0)

    return pd.Series(scores, index=df.index)


def calc_subindicators_concgate(df, w=20, divgate_days=3, concgate_days=1):
    """V4 sub-indicators with ConcGate."""
    n = len(df)
    pv_div = calc_pv_divergence(df, w)
    pv_conc = calc_pv_concordance(df, w)
    pv_fh = calc_pv_force_macd(df)

    raw_div = np.array([np.clip(pv_div.iloc[i] / 3, -1, 1) for i in range(n)])
    div_consec = np.ones(n)
    for i in range(1, n):
        if (raw_div[i] > 0 and raw_div[i-1] > 0) or \
           (raw_div[i] < 0 and raw_div[i-1] < 0):
            div_consec[i] = div_consec[i-1] + 1
        else:
            div_consec[i] = 1

    raw_conc = np.array([pv_conc.iloc[i] for i in range(n)])
    conc_consec = np.ones(n)
    for i in range(1, n):
        if (raw_conc[i] > 0 and raw_conc[i-1] > 0) or \
           (raw_conc[i] < 0 and raw_conc[i-1] < 0):
            conc_consec[i] = conc_consec[i-1] + 1
        else:
            conc_consec[i] = 1

    arr_force = np.zeros(n)
    arr_div = np.zeros(n)
    arr_conc = np.zeros(n)
    arr_conc_raw = np.zeros(n)

    for i in range(max(60, w), n):
        s_div = raw_div[i] if div_consec[i] >= divgate_days else 0.0
        s_conc = raw_conc[i] if conc_consec[i] >= concgate_days else 0.0
        fhr_std = pv_fh.iloc[max(0, i - w):i].std() + 1e-10
        s_force = np.clip(pv_fh.iloc[i] / (2 * fhr_std), -1, 1)

        arr_force[i] = s_force
        arr_div[i] = s_div
        arr_conc[i] = s_conc
        arr_conc_raw[i] = raw_conc[i]

    return pd.DataFrame({
        's_force': arr_force, 's_div': arr_div,
        's_conc': arr_conc, 's_conc_raw': arr_conc_raw,
    }, index=df.index)


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


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
print("=" * 140)
print("  V4 ConcGate TEST: S_conc에 Duration 적용 (DivGate처럼)")
print("  현재: S_div만 3일 연속 → 활성화.  S_conc는 매일 raw 사용.")
print(f"  테스트: ConcGate = {CONCGATE_DAYS} 일 연속 같은 부호")
print("=" * 140)

master = {}       # ticker -> concgate_days -> result
signal_sets = {}  # ticker -> concgate_days -> buy indices
conc_stats = {}   # ticker -> concgate_days -> s_conc distribution at signal time

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

    master[tk] = {}
    signal_sets[tk] = {}
    conc_stats[tk] = {}

    for cg in CONCGATE_DAYS:
        score = calc_v4_score_concgate(df_s, w=V4_W, divgate_days=DIVGATE, concgate_days=cg)
        buys = get_buy_signals(df_s, score)
        res = simulate(close, close_2x, buys, dates)
        master[tk][cg] = res
        signal_sets[tk][cg] = buys

        # S_conc stats at signal time
        subind = calc_subindicators_concgate(df_s, w=V4_W, divgate_days=DIVGATE, concgate_days=cg)
        conc_vals = []
        conc_raw_vals = []
        conc_zero_count = 0
        for idx in buys:
            if idx < len(subind):
                conc_vals.append(subind['s_conc'].iloc[idx])
                conc_raw_vals.append(subind['s_conc_raw'].iloc[idx])
                if subind['s_conc'].iloc[idx] == 0:
                    conc_zero_count += 1
        conc_stats[tk][cg] = {
            'avg': np.mean(conc_vals) if conc_vals else 0,
            'avg_raw': np.mean(conc_raw_vals) if conc_raw_vals else 0,
            'neg_pct': sum(1 for c in conc_vals if c < 0) / len(conc_vals) * 100 if conc_vals else 0,
            'zero_pct': conc_zero_count / len(conc_vals) * 100 if conc_vals else 0,
        }

    base = master[tk][1]
    print(f"sigs={base['sig_count']:>3d}  edge={base['edge']:>+.1f}%  ", end="")
    for cg in CONCGATE_DAYS[1:]:
        r = master[tk][cg]
        print(f"CG{cg}={r['sig_count']:>3d}/{r['edge']:>+.1f}  ", end="")
    print()

tks = list(master.keys())


# ═══════════════════════════════════════════════════════════
# SECTION 1: Signal Count & Edge
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 140}")
print(f"  SECTION 1: Signal Count / Leverage Edge per ConcGate Duration")
print(f"{'=' * 140}")

print(f"\n  {'Ticker':<7s}", end="")
for cg in CONCGATE_DAYS:
    label = "CUR(1)" if cg == 1 else f"CG={cg}"
    print(f"  {label:>14s}", end="")
print()
print(f"  {'':7s}", end="")
for _ in CONCGATE_DAYS:
    print(f"  {'sigs/edge%p':>14s}", end="")
print()
print(f"  {'=' * 100}")

for tk in tks:
    line = f"  {tk:<7s}"
    for cg in CONCGATE_DAYS:
        r = master[tk][cg]
        line += f"  {r['sig_count']:>5d}/{r['edge']:>+5.1f}%"
    print(line)

print(f"  {'-' * 100}")
line = f"  {'AVG':<7s}"
for cg in CONCGATE_DAYS:
    avg_sig = np.mean([master[tk][cg]['sig_count'] for tk in tks])
    avg_edge = np.mean([master[tk][cg]['edge'] for tk in tks])
    line += f"  {avg_sig:>5.0f}/{avg_edge:>+5.1f}%"
print(line)


# ═══════════════════════════════════════════════════════════
# SECTION 2: Efficiency + Bear/Bull
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 140}")
print(f"  SECTION 2: Summary — Edge, Efficiency, Bear/Bull Split")
print(f"{'=' * 140}")

print(f"\n  {'ConcGate':>8s} {'Avg Sigs':>9s} {'Avg Edge':>10s} {'Bear':>8s} {'Bull':>8s} "
      f"{'Worst':>8s} {'Effic':>8s} {'W/L':>7s} {'vs CG=1':>8s}")
print(f"  {'=' * 85}")

summary = {}
for cg in CONCGATE_DAYS:
    avg_edge = np.mean([master[tk][cg]['edge'] for tk in tks])
    avg_bear = np.mean([master[tk][cg]['bear_edge'] for tk in tks])
    avg_bull = np.mean([master[tk][cg]['bull_edge'] for tk in tks])
    avg_worst = np.mean([master[tk][cg]['worst_diff'] for tk in tks])
    avg_eff = np.mean([master[tk][cg]['efficiency'] for tk in tks])
    avg_sigs = np.mean([master[tk][cg]['sig_count'] for tk in tks])
    total_w = sum(master[tk][cg]['wins'] for tk in tks)
    total_l = sum(master[tk][cg]['losses'] for tk in tks)

    summary[cg] = {
        'edge': avg_edge, 'bear': avg_bear, 'bull': avg_bull,
        'worst': avg_worst, 'eff': avg_eff, 'sigs': avg_sigs,
        'wins': total_w, 'losses': total_l,
    }

base = summary[1]
for cg in CONCGATE_DAYS:
    s = summary[cg]
    de = s['eff'] - base['eff']
    marker = ""
    if cg != 1:
        if de > 0.03: marker = " *** BEST ***"
        elif de > 0.01: marker = " ** BETTER **"
        elif de > 0.005: marker = " * mild + *"
        elif de < -0.03: marker = " [HARMFUL]"
        elif de < -0.01: marker = " [worse]"

    label = "CUR(1)" if cg == 1 else f"CG={cg}"
    print(f"  {label:>8s} {s['sigs']:>8.0f} {s['edge']:>+9.2f}% {s['bear']:>+7.2f}% {s['bull']:>+7.2f}% "
          f"{s['worst']:>+7.2f}% {s['eff']:>+7.3f} {s['wins']:>3d}/{s['losses']:<3d} {de:>+7.3f}{marker}")


# ═══════════════════════════════════════════════════════════
# SECTION 3: Per-ticker Efficiency
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 140}")
print(f"  SECTION 3: Per-ticker Efficiency Comparison")
print(f"{'=' * 140}")

print(f"\n  {'Ticker':<7s}", end="")
for cg in CONCGATE_DAYS:
    label = "CUR(1)" if cg == 1 else f"CG={cg}"
    print(f"  {label:>10s}", end="")
print(f"  {'Best':>10s}")
print(f"  {'=' * 80}")

eff_winners = {cg: 0 for cg in CONCGATE_DAYS}

for tk in tks:
    line = f"  {tk:<7s}"
    best_cg = 1; best_eff = master[tk][1]['efficiency']
    for cg in CONCGATE_DAYS:
        eff = master[tk][cg]['efficiency']
        delta = eff - master[tk][1]['efficiency']
        m = "+" if delta > 0.01 else ("-" if delta < -0.01 else " ")
        line += f"  {eff:>+6.3f}({m})"
        if eff > best_eff:
            best_eff = eff; best_cg = cg
    eff_winners[best_cg] += 1
    label = "CUR(1)" if best_cg == 1 else f"CG={best_cg}"
    line += f"   {label}"
    print(line)

print(f"\n  Winners:")
for cg in sorted(eff_winners.keys(), key=lambda x: eff_winners[x], reverse=True):
    if eff_winners[cg] > 0:
        label = "CUR(1)" if cg == 1 else f"CG={cg}"
        print(f"    {label:>8s}: {eff_winners[cg]:>2d}/{len(tks)} tickers")


# ═══════════════════════════════════════════════════════════
# SECTION 4: S_conc Distribution at Signal Time
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 140}")
print(f"  SECTION 4: S_conc at Signal Time — How ConcGate changes the distribution")
print(f"{'=' * 140}")

print(f"\n  {'ConcGate':>8s} {'Avg S_conc':>11s} {'S_conc<0 %':>11s} {'S_conc=0 %':>11s} {'Interpretation':<30s}")
print(f"  {'=' * 80}")

for cg in CONCGATE_DAYS:
    all_avg = np.mean([conc_stats[tk][cg]['avg'] for tk in tks])
    all_neg = np.mean([conc_stats[tk][cg]['neg_pct'] for tk in tks])
    all_zero = np.mean([conc_stats[tk][cg]['zero_pct'] for tk in tks])

    if cg == 1:
        interp = "현재: raw S_conc 전부 사용"
    elif all_zero > 50:
        interp = f"S_conc {all_zero:.0f}% 무효화 → 2지표 체계"
    elif all_neg < 30:
        interp = f"음수 S_conc {100-all_neg-all_zero:.0f}%만 통과"
    else:
        interp = f"부분적 필터링"

    label = "CUR(1)" if cg == 1 else f"CG={cg}"
    print(f"  {label:>8s} {all_avg:>+10.3f} {all_neg:>10.0f}% {all_zero:>10.0f}%  {interp}")

# Per-ticker detail for key ConcGate values
print(f"\n  --- Per-ticker S_conc=0 rate (gated to zero) ---")
print(f"  {'Ticker':<7s}", end="")
for cg in [3, 5, 7]:
    print(f"  CG={cg:>2d}", end="")
print()
for tk in tks:
    line = f"  {tk:<7s}"
    for cg in [3, 5, 7]:
        line += f"  {conc_stats[tk][cg]['zero_pct']:>5.0f}%"
    print(line)


# ═══════════════════════════════════════════════════════════
# SECTION 5: Signal Overlap Analysis
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 140}")
print(f"  SECTION 5: Signal Changes — ConcGate vs Current (CG=1)")
print(f"{'=' * 140}")

for cg in CONCGATE_DAYS[1:]:
    total_cur = 0; total_new = 0; shared = 0; only_cur = 0; only_new = 0

    for tk in tks:
        cur = signal_sets[tk][1]
        new = signal_sets[tk][cg]
        total_cur += len(cur); total_new += len(new)
        shared += len(cur & new)
        only_cur += len(cur - new)
        only_new += len(new - cur)

    print(f"  CG={cg}: Current={total_cur}  New={total_new}  "
          f"Shared={shared}  Removed={only_cur}  Added={only_new}  "
          f"Net={total_new-total_cur:>+d}")


# ═══════════════════════════════════════════════════════════
# SECTION 6: QQQ/VOO Year-by-Year
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 140}")
print(f"  SECTION 6: QQQ & VOO Year-by-Year Edge")
print(f"{'=' * 140}")

for tk in ['QQQ', 'VOO']:
    if tk not in tks:
        continue
    print(f"\n  {tk}:")
    all_years = sorted(set(r['yr'] for r in master[tk][1]['yr_results']))
    print(f"  {'Year':>6s}", end="")
    for cg in CONCGATE_DAYS:
        label = "CUR" if cg == 1 else f"CG{cg}"
        print(f"  {label:>7s}", end="")
    print()
    print(f"  {'=' * 55}")
    for yr in all_years:
        line = f"  {yr:>6d}"
        for cg in CONCGATE_DAYS:
            yr_r = [r for r in master[tk][cg]['yr_results'] if r['yr'] == yr]
            if yr_r:
                line += f"  {yr_r[0]['diff']:>+6.1f}%"
            else:
                line += f"  {'N/A':>7s}"
        print(line)

    print(f"  {'-' * 55}")
    line = f"  {'AVG':>6s}"
    for cg in CONCGATE_DAYS:
        avg = np.mean([r['diff'] for r in master[tk][cg]['yr_results']])
        line += f"  {avg:>+6.1f}%"
    print(line)


# ═══════════════════════════════════════════════════════════
# FINAL VERDICT
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 140}")
print(f"  FINAL VERDICT")
print(f"{'=' * 140}")

ranked = sorted(CONCGATE_DAYS[1:],
                key=lambda cg: summary[cg]['eff'] - base['eff'],
                reverse=True)

for i, cg in enumerate(ranked):
    s = summary[cg]
    de = s['edge'] - base['edge']
    deff = s['eff'] - base['eff']
    dbear = s['bear'] - base['bear']
    dsig = s['sigs'] - base['sigs']

    print(f"\n  #{i+1} ConcGate={cg}일")
    print(f"      Signals:    {base['sigs']:.0f} -> {s['sigs']:.0f} ({dsig:>+.0f})")
    print(f"      Edge:       {base['edge']:>+.2f} -> {s['edge']:>+.2f}%p ({de:>+.2f})")
    print(f"      Efficiency: {base['eff']:>+.3f} -> {s['eff']:>+.3f} ({deff:>+.3f})")
    print(f"      Bear edge:  {base['bear']:>+.2f} -> {s['bear']:>+.2f}%p ({dbear:>+.2f})")
    print(f"      W/L:        {base['wins']}/{base['losses']} -> {s['wins']}/{s['losses']}")

print(f"\n{'=' * 140}")
print("  Done.")
