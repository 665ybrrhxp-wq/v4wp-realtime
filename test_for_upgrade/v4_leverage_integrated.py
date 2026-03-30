"""
V4 Leverage Integrated Strategy: 내부 신호 품질 기반 vs 외부 레짐 필터
======================================================================
기존 문제: V4+2x는 상승장에서만 +13%p, 하락/횡보장에서 -3%p

전략 비교:
  A) ALWAYS_2x:  항상 2x (baseline)
  B) SCORE_MAG:  V4 score 크기에 따라 배수 조절 (약한 신호=1x, 강한 신호=2x)
  C) ACTIVITY:   Activity multiplier 3 (3개 지표 일치)일 때만 2x
  D) DURATION:   Signal duration >= 5일이면 2x (확실한 신호만)
  E) SUBIND:     s_force > 0 AND s_conc > 0 (방향 일치)일 때만 2x
  F) COMPOSITE:  score_mag + activity + duration 종합 점수로 결정
  G) MA200:      외부 레짐: 종가 > 200일 SMA일 때만 2x
  H) COMBO:      V4 내부(COMPOSITE) + 외부(MA200) 둘 다 충족시 2x
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
    calc_v4_score, calc_v4_subindicators, detect_signal_events,
    build_price_filter, smooth_earnings_volume,
)

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


def get_enriched_signals(df, ticker):
    """
    V4 신호 + 내부 품질 지표를 함께 반환.
    Returns: dict { signal_idx: { 'score', 'act', 's_force', 's_div', 's_conc', 'duration', ... } }
    """
    try:
        df_s = smooth_earnings_volume(df, ticker=ticker)
    except Exception:
        df_s = df.copy()

    score = calc_v4_score(df_s, w=V4_W, divgate_days=DIVGATE)
    subind = calc_v4_subindicators(df_s, w=V4_W, divgate_days=DIVGATE)
    events = detect_signal_events(score, th=SIGNAL_TH, cooldown=COOLDOWN)
    pf = build_price_filter(df_s, er_q=ER_Q, atr_q=ATR_Q, lookback=LOOKBACK)

    n = len(df_s)
    signals = {}

    for ev in events:
        if ev['type'] != 'bottom':
            continue
        if not pf(ev['peak_idx']):
            continue
        dur = ev['end_idx'] - ev['start_idx'] + 1
        ci = ev['start_idx'] + CONFIRM - 1
        if ci > ev['end_idx'] or dur < CONFIRM or ci >= n:
            continue

        pidx = ev['peak_idx']
        signals[ci] = {
            'score': ev['peak_val'],
            'duration': dur,
            's_force': subind['s_force'].iloc[pidx] if pidx < len(subind) else 0,
            's_div': subind['s_div'].iloc[pidx] if pidx < len(subind) else 0,
            's_conc': subind['s_conc'].iloc[pidx] if pidx < len(subind) else 0,
            'act': subind['act'].iloc[pidx] if pidx < len(subind) else 0,
            'peak_idx': pidx,
        }

    return signals


# ═══════════════════════════════════════════════════════════
# Leverage Decision Functions
# Each returns True (use 2x) or False (use 1x) given signal info + market data
# ═══════════════════════════════════════════════════════════

def decide_always(sig_info, close, idx, sma200):
    """항상 2x"""
    return True


def decide_score_mag(sig_info, close, idx, sma200):
    """V4 score >= 0.3이면 2x (상위 ~50% 강한 신호만)"""
    return sig_info['score'] >= 0.30


def decide_activity(sig_info, close, idx, sma200):
    """Activity == 3 (3개 sub-indicator 모두 활성)일 때만 2x"""
    return sig_info['act'] >= 3


def decide_duration(sig_info, close, idx, sma200):
    """Signal duration >= 5일 (확실한 신호만) 2x"""
    return sig_info['duration'] >= 5


def decide_subind_align(sig_info, close, idx, sma200):
    """s_force > 0 AND s_conc > 0 (방향 일치) 일 때만 2x"""
    return sig_info['s_force'] > 0 and sig_info['s_conc'] > 0


def decide_composite(sig_info, close, idx, sma200):
    """종합 신호 품질 점수: score_mag + activity + duration
    quality = score/0.5 + act/3 + min(dur,10)/10
    quality >= 1.5 이면 2x"""
    sc = min(abs(sig_info['score']) / 0.5, 1.0)
    ac = sig_info['act'] / 3.0
    du = min(sig_info['duration'], 10) / 10.0
    quality = sc + ac + du
    return quality >= 1.5


def decide_ma200(sig_info, close, idx, sma200):
    """외부 레짐: 종가 > 200일 SMA"""
    if sma200 is None or np.isnan(sma200[idx]):
        return True
    return close[idx] > sma200[idx]


def decide_combo_v4_ma(sig_info, close, idx, sma200):
    """V4 내부(composite) + 외부(MA200) 둘 다 충족"""
    v4_ok = decide_composite(sig_info, close, idx, sma200)
    ma_ok = decide_ma200(sig_info, close, idx, sma200)
    return v4_ok and ma_ok


def decide_v4_or_ma(sig_info, close, idx, sma200):
    """V4 내부(composite) OR 외부(MA200) 하나라도 충족"""
    v4_ok = decide_composite(sig_info, close, idx, sma200)
    ma_ok = decide_ma200(sig_info, close, idx, sma200)
    return v4_ok or ma_ok


STRATEGIES = {
    'ALWAYS_2x':  decide_always,
    'SCORE_MAG':  decide_score_mag,
    'ACTIVITY3':  decide_activity,
    'DURATION5':  decide_duration,
    'SUBIND_ALN': decide_subind_align,
    'COMPOSITE':  decide_composite,
    'MA200':      decide_ma200,
    'V4+MA(AND)': decide_combo_v4_ma,
    'V4|MA(OR)':  decide_v4_or_ma,
}

STRAT_NAMES = list(STRATEGIES.keys())


def simulate(close, close_2x, enriched_signals, dates, decide_fn, sma200):
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
    total_dep = 0.0; sig_2x = 0; sig_1x = 0

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
        total_dep += MONTHLY_DEPOSIT
        yr_data[yr]['deposits'] += MONTHLY_DEPOSIT

        for day_idx in range(fi, li + 1):
            if day_idx in enriched_signals and cash_a > 1.0:
                sig_info = enriched_signals[day_idx]
                amt = cash_a * SIGNAL_BUY_PCT

                if decide_fn(sig_info, close, day_idx, sma200):
                    sh_2x_a += amt / close_2x[day_idx]
                    sig_2x += 1
                else:
                    sh_1x_a += amt / close[day_idx]
                    sig_1x += 1
                cash_a -= amt

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

    avg_a = np.mean([r['ret_a'] for r in yr_results])
    avg_b = np.mean([r['ret_b'] for r in yr_results])
    worst_diff = min(r['diff'] for r in yr_results)

    bull_yrs = [r for r in yr_results if r['ret_b'] > 5]
    bear_yrs = [r for r in yr_results if r['ret_b'] < -5]
    flat_yrs = [r for r in yr_results if -5 <= r['ret_b'] <= 5]

    edge = avg_a - avg_b

    return {
        'edge': edge,
        'final_a': final_a, 'final_b': final_b,
        'wins': sum(1 for r in yr_results if r['diff'] > 0.5),
        'losses': sum(1 for r in yr_results if r['diff'] < -0.5),
        'worst_diff': worst_diff,
        'sig_2x': sig_2x, 'sig_1x': sig_1x,
        'bull_edge': np.mean([r['diff'] for r in bull_yrs]) if bull_yrs else 0,
        'bear_edge': np.mean([r['diff'] for r in bear_yrs]) if bear_yrs else 0,
        'flat_edge': np.mean([r['diff'] for r in flat_yrs]) if flat_yrs else 0,
        'efficiency': edge / abs(worst_diff) if abs(worst_diff) > 0.1 else 0,
    }


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
print("=" * 140)
print("  V4 LEVERAGE INTEGRATED STRATEGY: Internal Signal Quality vs External Regime Filter")
print("  Signal fires -> decide leverage via V4 internals (score, activity, duration, sub-indicators)")
print("=" * 140)

master = {}

for tk, sector in TICKERS.items():
    print(f"  {tk}...", end=" ", flush=True)
    df = download_max(tk)
    if df is None or len(df) < 300:
        print("SKIP")
        continue

    close = df['Close'].values
    dates = df.index

    close_2x = build_synthetic_2x(close)
    enriched = get_enriched_signals(df, tk)
    sma200 = pd.Series(close).rolling(200, min_periods=200).mean().values

    master[tk] = {'sector': sector}
    parts = []
    for sname, sfunc in STRATEGIES.items():
        res = simulate(close, close_2x, enriched, dates, sfunc, sma200)
        master[tk][sname] = res
        parts.append(f"{sname}={res['edge']:+.1f}")
    print("  ".join(parts))

    # Signal quality stats
    if enriched:
        scores = [s['score'] for s in enriched.values()]
        acts = [s['act'] for s in enriched.values()]
        durs = [s['duration'] for s in enriched.values()]
        master[tk]['_stats'] = {
            'n_sig': len(enriched),
            'avg_score': np.mean(scores), 'med_score': np.median(scores),
            'avg_act': np.mean(acts), 'pct_act3': sum(1 for a in acts if a >= 3) / len(acts) * 100,
            'avg_dur': np.mean(durs), 'pct_dur5': sum(1 for d in durs if d >= 5) / len(durs) * 100,
        }


# ═══════════════════════════════════════════════════════════
# [1] Overall Edge Comparison
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 140}")
print(f"  [1] EDGE COMPARISON: All Strategies")
print(f"{'=' * 140}")

header = f"  {'Ticker':<7s}"
for sn in STRAT_NAMES:
    label = sn[:9]
    header += f" {label:>10s}"
print(header)
print(f"  {'=' * 100}")

for tk in master:
    if '_stats' not in master[tk]:
        continue
    line = f"  {tk:<7s}"
    for sn in STRAT_NAMES:
        e = master[tk][sn]['edge']
        line += f" {e:>+9.2f}%p"
    print(line)

# Average
line = f"  {'AVG':<7s}"
tks = [tk for tk in master if '_stats' in master[tk]]
for sn in STRAT_NAMES:
    avg = np.mean([master[tk][sn]['edge'] for tk in tks])
    line += f" {avg:>+9.2f}%p"
print(f"  {'=' * 100}")
print(line)


# ═══════════════════════════════════════════════════════════
# [2] Bear Market Edge
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 140}")
print(f"  [2] BEAR MARKET EDGE (DCA < -5% years): The Problem We're Solving")
print(f"{'=' * 140}")

header = f"  {'Ticker':<7s}"
for sn in STRAT_NAMES:
    label = sn[:9]
    header += f" {label:>10s}"
print(header)
print(f"  {'=' * 100}")

for tk in tks:
    line = f"  {tk:<7s}"
    for sn in STRAT_NAMES:
        be = master[tk][sn]['bear_edge']
        line += f" {be:>+9.2f}%p"
    print(line)

line = f"  {'AVG':<7s}"
for sn in STRAT_NAMES:
    avg = np.mean([master[tk][sn]['bear_edge'] for tk in tks])
    line += f" {avg:>+9.2f}%p"
print(f"  {'=' * 100}")
print(line)


# ═══════════════════════════════════════════════════════════
# [3] Bull Market Edge
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 140}")
print(f"  [3] BULL MARKET EDGE (DCA > +5% years): What We Keep")
print(f"{'=' * 140}")

header = f"  {'Ticker':<7s}"
for sn in STRAT_NAMES:
    label = sn[:9]
    header += f" {label:>10s}"
print(header)
print(f"  {'=' * 100}")

for tk in tks:
    line = f"  {tk:<7s}"
    for sn in STRAT_NAMES:
        bue = master[tk][sn]['bull_edge']
        line += f" {bue:>+9.2f}%p"
    print(line)

line = f"  {'AVG':<7s}"
for sn in STRAT_NAMES:
    avg = np.mean([master[tk][sn]['bull_edge'] for tk in tks])
    line += f" {avg:>+9.2f}%p"
print(f"  {'=' * 100}")
print(line)


# ═══════════════════════════════════════════════════════════
# [4] Worst Year Risk
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 140}")
print(f"  [4] RISK: Worst Year Extra Loss")
print(f"{'=' * 140}")

header = f"  {'Ticker':<7s}"
for sn in STRAT_NAMES:
    label = sn[:9]
    header += f" {label:>10s}"
print(header)
print(f"  {'=' * 100}")

for tk in tks:
    line = f"  {tk:<7s}"
    for sn in STRAT_NAMES:
        w = master[tk][sn]['worst_diff']
        line += f" {w:>+9.2f}%p"
    print(line)

line = f"  {'AVG':<7s}"
for sn in STRAT_NAMES:
    avg = np.mean([master[tk][sn]['worst_diff'] for tk in tks])
    line += f" {avg:>+9.2f}%p"
print(f"  {'=' * 100}")
print(line)


# ═══════════════════════════════════════════════════════════
# [5] Efficiency: Edge / |Worst Loss|
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 140}")
print(f"  [5] EFFICIENCY: Edge / |Worst Year Loss| (higher = better risk-adjusted)")
print(f"{'=' * 140}")

header = f"  {'Ticker':<7s}"
for sn in STRAT_NAMES:
    label = sn[:9]
    header += f" {label:>10s}"
print(header)
print(f"  {'=' * 100}")

for tk in tks:
    line = f"  {tk:<7s}"
    for sn in STRAT_NAMES:
        eff = master[tk][sn]['efficiency']
        line += f" {eff:>+9.3f} "
    print(line)

line = f"  {'AVG':<7s}"
for sn in STRAT_NAMES:
    avg = np.mean([master[tk][sn]['efficiency'] for tk in tks])
    line += f" {avg:>+9.3f} "
print(f"  {'=' * 100}")
print(line)


# ═══════════════════════════════════════════════════════════
# [6] Signal Routing
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 140}")
print(f"  [6] SIGNAL ROUTING: % of signals that get 2x leverage")
print(f"{'=' * 140}")

header = f"  {'Ticker':<7s} {'Total':>5s}"
for sn in STRAT_NAMES[1:]:
    label = sn[:9]
    header += f" {label:>9s}"
print(header)
print(f"  {'=' * 95}")

for tk in tks:
    total = master[tk]['ALWAYS_2x']['sig_2x']
    line = f"  {tk:<7s} {total:>5d}"
    for sn in STRAT_NAMES[1:]:
        s2x = master[tk][sn]['sig_2x']
        pct = s2x / total * 100 if total > 0 else 0
        line += f" {pct:>8.0f}%"
    print(line)

line = f"  {'AVG':<7s} {'':>5s}"
for sn in STRAT_NAMES[1:]:
    pcts = []
    for tk in tks:
        total = master[tk]['ALWAYS_2x']['sig_2x']
        s2x = master[tk][sn]['sig_2x']
        if total > 0:
            pcts.append(s2x / total * 100)
    line += f" {np.mean(pcts):>8.0f}%"
print(f"  {'=' * 95}")
print(line)


# ═══════════════════════════════════════════════════════════
# [7] Signal Quality Stats
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 140}")
print(f"  [7] SIGNAL QUALITY DISTRIBUTION (for understanding filter impact)")
print(f"{'=' * 140}")

print(f"  {'Ticker':<7s} {'Signals':>7s} {'AvgScore':>9s} {'MedScore':>9s} {'AvgAct':>7s} {'Act>=3':>7s} {'AvgDur':>7s} {'Dur>=5':>7s}")
print(f"  {'=' * 65}")
for tk in tks:
    st = master[tk]['_stats']
    print(f"  {tk:<7s} {st['n_sig']:>7d} {st['avg_score']:>8.3f} {st['med_score']:>8.3f} "
          f"{st['avg_act']:>6.1f} {st['pct_act3']:>6.0f}% {st['avg_dur']:>6.1f} {st['pct_dur5']:>6.0f}%")


# ═══════════════════════════════════════════════════════════
# GRAND SUMMARY
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 140}")
print(f"  GRAND SUMMARY")
print(f"{'=' * 140}")

print(f"\n  {'Strategy':<12s} {'Type':>8s} {'Avg Edge':>10s} {'Bear Edge':>10s} {'Bull Edge':>10s} "
      f"{'Worst Loss':>11s} {'Efficiency':>11s} {'2x Rate':>8s}")
print(f"  {'=' * 85}")

for sn in STRAT_NAMES:
    avg_edge = np.mean([master[tk][sn]['edge'] for tk in tks])
    avg_bear = np.mean([master[tk][sn]['bear_edge'] for tk in tks])
    avg_bull = np.mean([master[tk][sn]['bull_edge'] for tk in tks])
    avg_worst = np.mean([master[tk][sn]['worst_diff'] for tk in tks])
    avg_eff = np.mean([master[tk][sn]['efficiency'] for tk in tks])

    if sn == 'ALWAYS_2x':
        sig_pct = 100.0
    else:
        pcts = []
        for tk in tks:
            total = master[tk]['ALWAYS_2x']['sig_2x']
            s2x = master[tk][sn]['sig_2x']
            if total > 0:
                pcts.append(s2x / total * 100)
        sig_pct = np.mean(pcts)

    stype = "INTERNAL" if sn in ('SCORE_MAG','ACTIVITY3','DURATION5','SUBIND_ALN','COMPOSITE') else (
            "EXTERNAL" if sn == 'MA200' else (
            "HYBRID" if sn in ('V4+MA(AND)','V4|MA(OR)') else "BASELINE"))

    print(f"  {sn:<12s} {stype:>8s} {avg_edge:>+9.2f}%p {avg_bear:>+9.2f}%p {avg_bull:>+9.2f}%p "
          f"{avg_worst:>+10.2f}%p {avg_eff:>+10.3f}  {sig_pct:>6.0f}%")

# Delta vs baseline
baseline_edge = np.mean([master[tk]['ALWAYS_2x']['edge'] for tk in tks])
baseline_bear = np.mean([master[tk]['ALWAYS_2x']['bear_edge'] for tk in tks])
baseline_worst = np.mean([master[tk]['ALWAYS_2x']['worst_diff'] for tk in tks])
baseline_eff = np.mean([master[tk]['ALWAYS_2x']['efficiency'] for tk in tks])

print(f"\n  Delta vs ALWAYS_2x baseline:")
print(f"  {'Strategy':<12s} {'Edge':>10s} {'Bear Fix':>10s} {'Worst Fix':>11s} {'Eff Change':>11s}")
print(f"  {'=' * 55}")

for sn in STRAT_NAMES[1:]:
    de = np.mean([master[tk][sn]['edge'] for tk in tks]) - baseline_edge
    db = np.mean([master[tk][sn]['bear_edge'] for tk in tks]) - baseline_bear
    dw = np.mean([master[tk][sn]['worst_diff'] for tk in tks]) - baseline_worst
    deff = np.mean([master[tk][sn]['efficiency'] for tk in tks]) - baseline_eff

    # Stars
    stars = ""
    if db > 0.5: stars += " [bear+]"
    if dw > 1.0: stars += " [risk-]"
    if deff > 0.1: stars += " [eff+]"
    if de > -0.5 and db > 0.5 and deff > 0: stars += " *** WINNER ***"

    print(f"  {sn:<12s} {de:>+9.2f}%p {db:>+9.2f}%p {dw:>+10.2f}%p {deff:>+10.3f} {stars}")

# Per-ticker best
print(f"\n  Per-ticker best efficiency (excluding baseline):")
from collections import Counter
best_map = {}
for tk in tks:
    best_sn = max(STRAT_NAMES[1:], key=lambda sn: master[tk][sn]['efficiency'])
    best_map[tk] = best_sn
    be = master[tk][best_sn]['efficiency']
    base_e = master[tk]['ALWAYS_2x']['efficiency']
    print(f"    {tk:<7s}: {best_sn:<12s} (eff={be:+.3f} vs baseline={base_e:+.3f})")

dist = Counter(best_map.values())
print(f"\n  Distribution of best strategy:")
for sn, cnt in sorted(dist.items(), key=lambda x: -x[1]):
    print(f"    {sn:<12s}: {cnt}/{len(tks)} tickers")

print()
print("=" * 140)
print("  Done.")
