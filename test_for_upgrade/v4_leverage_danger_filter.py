"""
V4 Leverage Danger Filter — Individual Condition Test
======================================================
각 위험 조건을 개별적으로 테스트해서 실제 영향 측정:

  1) SHALLOW:   60일 고점 대비 3% 미만 하락 → 1x
  2) RISEN:     20일 추세 > +3% → 1x
  3) NEAR_ATH:  전고점 5% 이내 → 1x
  4) SCONC_NEG: S_conc < 0 → 1x

각 조건이:
  - 몇 개의 신호를 걸러내는지
  - 걸러낸 신호의 실제 forward return은 어떤지
  - Edge와 Efficiency에 얼마나 영향 주는지
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
    try:
        df_s = smooth_earnings_volume(df, ticker=ticker)
    except Exception:
        df_s = df.copy()

    close = df_s['Close'].values
    dates = df_s.index
    n = len(df_s)

    score = calc_v4_score(df_s, w=V4_W, divgate_days=DIVGATE)
    subind = calc_v4_subindicators(df_s, w=V4_W, divgate_days=DIVGATE)
    events = detect_signal_events(score, th=SIGNAL_TH, cooldown=COOLDOWN)
    pf = build_price_filter(df_s, er_q=ER_Q, atr_q=ATR_Q, lookback=LOOKBACK)

    rolling_high_60 = pd.Series(close).rolling(60, min_periods=1).max().values

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
        buy_idx = ci
        price = close[buy_idx]

        # Danger conditions
        dd_60h = (rolling_high_60[buy_idx] - price) / rolling_high_60[buy_idx] * 100
        trend_20d = (close[buy_idx] / close[max(0, buy_idx - 20)] - 1) * 100
        ath = np.max(close[:buy_idx + 1])
        near_ath = (ath - price) / ath * 100 < 5
        s_conc = subind['s_conc'].iloc[pidx] if pidx < len(subind) else 0

        # Danger flags
        is_shallow = dd_60h < 3
        is_risen = trend_20d > 3
        is_ath = near_ath
        is_sconc_neg = s_conc < 0

        danger_count = sum([is_shallow, is_risen, is_ath, is_sconc_neg])

        # Forward returns for individual analysis
        fwd_90 = None
        if buy_idx + 90 < n:
            fwd_90 = (close[buy_idx + 90] / price - 1) * 100

        signals[ci] = {
            'score': ev['peak_val'],
            'duration': dur,
            's_conc': s_conc,
            'dd_60h': dd_60h,
            'trend_20d': trend_20d,
            'near_ath': near_ath,
            'is_shallow': is_shallow,
            'is_risen': is_risen,
            'is_ath': is_ath,
            'is_sconc_neg': is_sconc_neg,
            'danger_count': danger_count,
            'fwd_90': fwd_90,
        }

    return signals


# ── Danger filter strategies (순수 개별 조건) ──
def decide_always(sig):
    return True

def decide_shallow(sig):
    """dd_60h < 3% → 1x"""
    return not sig['is_shallow']

def decide_risen(sig):
    """trend_20d > 3% → 1x"""
    return not sig['is_risen']

def decide_near_ath(sig):
    """near_ath (ATH 대비 5% 이내) → 1x"""
    return not sig['is_ath']

def decide_sconc_neg(sig):
    """s_conc < 0 → 1x"""
    return not sig['is_sconc_neg']

# ── 2개 조합 (OR: 둘 중 하나라도 → 1x) ──
def decide_SH_RI(sig):
    return not (sig['is_shallow'] or sig['is_risen'])

def decide_SH_ATH(sig):
    return not (sig['is_shallow'] or sig['is_ath'])

def decide_SH_SC(sig):
    return not (sig['is_shallow'] or sig['is_sconc_neg'])

def decide_RI_ATH(sig):
    return not (sig['is_risen'] or sig['is_ath'])

def decide_RI_SC(sig):
    return not (sig['is_risen'] or sig['is_sconc_neg'])

def decide_ATH_SC(sig):
    return not (sig['is_ath'] or sig['is_sconc_neg'])

# ── 3개 조합 (OR: 셋 중 하나라도 → 1x) ──
def decide_SH_RI_ATH(sig):
    return not (sig['is_shallow'] or sig['is_risen'] or sig['is_ath'])

def decide_SH_RI_SC(sig):
    return not (sig['is_shallow'] or sig['is_risen'] or sig['is_sconc_neg'])

def decide_SH_ATH_SC(sig):
    return not (sig['is_shallow'] or sig['is_ath'] or sig['is_sconc_neg'])

def decide_RI_ATH_SC(sig):
    return not (sig['is_risen'] or sig['is_ath'] or sig['is_sconc_neg'])

# ── 4개 전부 (OR: 하나라도 → 1x) ──
def decide_ALL4(sig):
    return not (sig['is_shallow'] or sig['is_risen'] or sig['is_ath'] or sig['is_sconc_neg'])

# ── AND 조합 (모두 충족시에만 → 1x) ──
def decide_ANY2(sig):
    """4개 중 2개 이상 → 1x"""
    return sig['danger_count'] < 2

def decide_ANY3(sig):
    """4개 중 3개 이상 → 1x"""
    return sig['danger_count'] < 3


STRATEGIES = {
    # Baseline
    'ALWAYS':     decide_always,
    # 개별 (4)
    'SHALLOW':    decide_shallow,
    'RISEN':      decide_risen,
    'NEAR_ATH':   decide_near_ath,
    'SCONC_NEG':  decide_sconc_neg,
    # 2개 OR 조합 (6)
    'SH+RI':      decide_SH_RI,
    'SH+ATH':     decide_SH_ATH,
    'SH+SC':      decide_SH_SC,
    'RI+ATH':     decide_RI_ATH,
    'RI+SC':      decide_RI_SC,
    'ATH+SC':     decide_ATH_SC,
    # 3개 OR 조합 (4)
    'SH+RI+ATH':  decide_SH_RI_ATH,
    'SH+RI+SC':   decide_SH_RI_SC,
    'SH+ATH+SC':  decide_SH_ATH_SC,
    'RI+ATH+SC':  decide_RI_ATH_SC,
    # 4개 전부
    'ALL_4':      decide_ALL4,
    # N개 이상 충족
    'ANY_2of4':   decide_ANY2,
    'ANY_3of4':   decide_ANY3,
}
STRAT_NAMES = list(STRATEGIES.keys())
CONDITION_KEYS = {
    'SHALLOW': 'is_shallow',
    'RISEN': 'is_risen',
    'NEAR_ATH': 'is_ath',
    'SCONC_NEG': 'is_sconc_neg',
}


def simulate(close, close_2x, enriched_signals, dates, decide_fn):
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
                sig = enriched_signals[day_idx]
                amt = cash_a * SIGNAL_BUY_PCT
                if decide_fn(sig):
                    sh_2x_a += amt / close_2x[day_idx]; sig_2x += 1
                else:
                    sh_1x_a += amt / close[day_idx]; sig_1x += 1
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
        'efficiency': edge / abs(worst_diff) if abs(worst_diff) > 0.1 else 0,
    }


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
print("=" * 140)
print("  V4 LEVERAGE DANGER FILTER: Individual + Combination Test")
print("  4 Conditions: SH=Shallow(<3%) | RI=Risen(>3%) | ATH=NearATH(<5%) | SC=S_conc<0")
print("  Filter logic: danger → 1x, safe → 2x")
print("=" * 140)

master = {}
all_signals = {}

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

    all_signals[tk] = enriched

    master[tk] = {}
    for sname, sfunc in STRATEGIES.items():
        res = simulate(close, close_2x, enriched, dates, sfunc)
        master[tk][sname] = res

    total = len(enriched)
    baseline = master[tk]['ALWAYS']['edge']
    print(f"sigs={total:>3d}  base_edge={baseline:>+.1f}%p  done")

tks = list(master.keys())

# ── Build summary for all strategies ──
summary = {}
for sn in STRAT_NAMES:
    avg_edge = np.mean([master[tk][sn]['edge'] for tk in tks])
    avg_bear = np.mean([master[tk][sn]['bear_edge'] for tk in tks])
    avg_bull = np.mean([master[tk][sn]['bull_edge'] for tk in tks])
    avg_worst = np.mean([master[tk][sn]['worst_diff'] for tk in tks])
    avg_eff = np.mean([master[tk][sn]['efficiency'] for tk in tks])

    if sn == 'ALWAYS':
        filt_pct = 0.0
    else:
        pcts = []
        for tk in tks:
            total_s = master[tk]['ALWAYS']['sig_2x']
            s2x = master[tk][sn]['sig_2x']
            if total_s > 0:
                pcts.append((total_s - s2x) / total_s * 100)
        filt_pct = np.mean(pcts)

    n_improved = sum(1 for tk in tks if master[tk][sn]['efficiency'] > master[tk]['ALWAYS']['efficiency'] + 0.005)
    n_worsened = sum(1 for tk in tks if master[tk][sn]['efficiency'] < master[tk]['ALWAYS']['efficiency'] - 0.005)

    summary[sn] = {
        'sn': sn, 'edge': avg_edge, 'bear': avg_bear, 'bull': avg_bull,
        'worst': avg_worst, 'eff': avg_eff, 'filt_pct': filt_pct,
        'n_improved': n_improved, 'n_worsened': n_worsened,
    }

base = summary['ALWAYS']


# ═══════════════════════════════════════════════════════════
# SECTION 1: Grand Strategy Table (all 18)
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 140}")
print(f"  ALL STRATEGIES — Filter Rate, Edge, Bear/Bull Split, Efficiency")
print(f"  Filter=danger→1x.  Efficiency = Edge / |Worst Year Loss|")
print(f"{'=' * 140}")

print(f"\n  {'Strategy':<13s} {'Filt%':>6s} {'Edge':>8s} {'Bear':>8s} {'Bull':>8s} "
      f"{'Worst':>8s} {'Effic':>8s} {'vs BASE':>8s} {'Imp':>4s} {'Wrs':>4s} {'Verdict':<16s}")
print(f"  {'=' * 105}")

# Group labels
groups = [
    ('--- Baseline ---', ['ALWAYS']),
    ('--- Individual (1 condition OR) ---', ['SHALLOW', 'RISEN', 'NEAR_ATH', 'SCONC_NEG']),
    ('--- 2-Combo (OR) ---', ['SH+RI', 'SH+ATH', 'SH+SC', 'RI+ATH', 'RI+SC', 'ATH+SC']),
    ('--- 3-Combo (OR) ---', ['SH+RI+ATH', 'SH+RI+SC', 'SH+ATH+SC', 'RI+ATH+SC']),
    ('--- 4-All / N-of-4 ---', ['ALL_4', 'ANY_2of4', 'ANY_3of4']),
]

for group_label, group_strats in groups:
    print(f"  {group_label}")
    for sn in group_strats:
        s = summary[sn]
        delta_eff = s['eff'] - base['eff']
        if sn == 'ALWAYS':
            verdict = "(baseline)"
        elif delta_eff > 0.03:
            verdict = "*** STRONG ***"
        elif delta_eff > 0.01:
            verdict = "** GOOD **"
        elif delta_eff > 0.005:
            verdict = "* mild +"
        elif delta_eff > -0.005:
            verdict = "~ neutral"
        elif delta_eff > -0.02:
            verdict = "mild -"
        else:
            verdict = "HARMFUL"

        print(f"  {sn:<13s} {s['filt_pct']:>5.0f}% {s['edge']:>+7.2f}% {s['bear']:>+7.2f}% {s['bull']:>+7.2f}% "
              f"{s['worst']:>+7.2f}% {s['eff']:>+7.3f} {delta_eff:>+7.3f} {s['n_improved']:>3d} {s['n_worsened']:>3d}  {verdict}")


# ═══════════════════════════════════════════════════════════
# SECTION 2: Per-ticker Best Strategy
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 140}")
print(f"  Per-Ticker: ALWAYS vs Best Combination")
print(f"{'=' * 140}")

print(f"\n  {'Ticker':<7s} {'ALWAYS':>12s}  {'Best Filter':>13s} {'Edge':>8s} {'Effic':>8s} {'delta_eff':>10s}")
print(f"  {'=' * 70}")

best_wins = {}
for tk in tks:
    be = master[tk]['ALWAYS']['edge']
    beff = master[tk]['ALWAYS']['efficiency']
    best_sn = 'ALWAYS'; best_eff = beff
    for sn in STRAT_NAMES[1:]:
        eff = master[tk][sn]['efficiency']
        if eff > best_eff:
            best_eff = eff; best_sn = sn
    best_edge = master[tk][best_sn]['edge']
    delta = best_eff - beff

    if best_sn not in best_wins:
        best_wins[best_sn] = 0
    best_wins[best_sn] += 1

    marker = " ***" if delta > 0.05 else (" **" if delta > 0.02 else "")
    print(f"  {tk:<7s} {be:>+6.1f}/{beff:>.3f}  {best_sn:>13s} {best_edge:>+7.1f}% {best_eff:>+7.3f} {delta:>+9.3f}{marker}")

print(f"\n  Strategy wins:")
for sn in sorted(best_wins.keys(), key=lambda x: best_wins[x], reverse=True):
    print(f"    {sn:<13s}: {best_wins[sn]:>2d}/{len(tks)} tickers")


# ═══════════════════════════════════════════════════════════
# SECTION 3: Top 10 Ranking by Efficiency
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 140}")
print(f"  RANKING: Top Strategies by Efficiency Improvement over ALWAYS")
print(f"{'=' * 140}")

ranked = sorted([s for sn, s in summary.items() if sn != 'ALWAYS'],
                key=lambda x: x['eff'] - base['eff'], reverse=True)

print(f"\n  {'Rank':>4s}  {'Strategy':<13s}  {'Filt%':>6s}  {'Edge':>9s}  {'dEdge':>7s}  {'Effic':>8s}  {'dEff':>7s}  "
      f"{'Bear':>8s}  {'dBear':>7s}  {'Imp/Wrs':>8s}")
print(f"  {'=' * 100}")

for i, r in enumerate(ranked[:10]):
    de = r['edge'] - base['edge']
    deff = r['eff'] - base['eff']
    dbear = r['bear'] - base['bear']
    print(f"  {i+1:>4d}  {r['sn']:<13s}  {r['filt_pct']:>5.0f}%  {r['edge']:>+8.2f}%  {de:>+6.2f}%  {r['eff']:>+7.3f}  {deff:>+6.3f}  "
          f"{r['bear']:>+7.2f}%  {dbear:>+6.2f}%  {r['n_improved']:>2d}/{r['n_worsened']:>2d}")

print(f"\n  ... Bottom 3:")
for i, r in enumerate(ranked[-3:]):
    de = r['edge'] - base['edge']
    deff = r['eff'] - base['eff']
    dbear = r['bear'] - base['bear']
    print(f"  {len(ranked)-2+i:>4d}  {r['sn']:<13s}  {r['filt_pct']:>5.0f}%  {r['edge']:>+8.2f}%  {de:>+6.2f}%  {r['eff']:>+7.3f}  {deff:>+6.3f}  "
          f"{r['bear']:>+7.2f}%  {dbear:>+6.2f}%  {r['n_improved']:>2d}/{r['n_worsened']:>2d}")


# ═══════════════════════════════════════════════════════════
# SECTION 4: Top 3 Deep Dive
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 140}")
print(f"  TOP 3 DEEP DIVE")
print(f"{'=' * 140}")

for rank, r in enumerate(ranked[:3]):
    sn = r['sn']
    de = r['edge'] - base['edge']
    deff = r['eff'] - base['eff']
    dbear = r['bear'] - base['bear']
    dbull = r['bull'] - base['bull']
    dworst = r['worst'] - base['worst']

    print(f"\n  #{rank+1} {sn}")
    print(f"  {'-' * 80}")
    print(f"    Filter rate:      {r['filt_pct']:.0f}% of signals downgraded to 1x")
    print(f"    Edge:             {base['edge']:>+.2f} -> {r['edge']:>+.2f}%p  ({de:>+.2f}%p)")
    print(f"    Efficiency:       {base['eff']:>+.3f} -> {r['eff']:>+.3f}  ({deff:>+.3f})")
    print(f"    Bear edge:        {base['bear']:>+.2f} -> {r['bear']:>+.2f}%p  ({dbear:>+.2f}%p improvement)")
    print(f"    Bull edge:        {base['bull']:>+.2f} -> {r['bull']:>+.2f}%p  ({dbull:>+.2f}%p sacrifice)")
    print(f"    Worst year:       {base['worst']:>+.2f} -> {r['worst']:>+.2f}%p  ({dworst:>+.2f}%p improvement)")
    print(f"    Tickers improved: {r['n_improved']}/{len(tks)}  Worsened: {r['n_worsened']}/{len(tks)}")

    # Per-ticker detail
    print(f"\n    {'Ticker':<7s}  {'ALWAYS':>12s}  {sn:>12s}  {'delta':>8s}")
    for tk in tks:
        ae = master[tk]['ALWAYS']['edge']
        aeff = master[tk]['ALWAYS']['efficiency']
        se = master[tk][sn]['edge']
        seff = master[tk][sn]['efficiency']
        d = seff - aeff
        m = " ***" if d > 0.05 else (" ++" if d > 0.01 else (" --" if d < -0.01 else ""))
        print(f"    {tk:<7s}  {ae:>+5.1f}/{aeff:>.3f}  {se:>+5.1f}/{seff:>.3f}  {d:>+7.3f}{m}")


print(f"\n{'=' * 140}")
print("  Done.")
