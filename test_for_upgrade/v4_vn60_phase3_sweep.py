"""
VN60 Phase 3: S_Force / S_Div 가중치 비율 Sweep
=================================================
실험⑥: W_force / W_div 비율 (합계=1.0)
  - 0.80/0.20, 0.70/0.30, 0.60/0.40(현재), 0.50/0.50, 0.40/0.60, 0.30/0.70

백테스트 전략 (기존 VN60 동일):
  - 월 $500 입금
  - 시그널 시 가용자금의 50% → 2x / 3x 레버리지 매수
  - 월말 잔여 자금 → 본주(1x) 매수
  - DCA = 월 $500 순수 1x 매수 (벤치마크)

고정 파라미터 (Phase 1~2 결론):
  - MACD: (12/26/9)
  - ForceNorm: /2.0 × std
  - DivClip: /3.0
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
    calc_pv_divergence,
    detect_signal_events, build_price_filter, smooth_earnings_volume,
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
EXPENSE_2X = 0.0095 / 252
EXPENSE_3X = 0.0100 / 252

V4_W = 20; SIGNAL_TH = 0.15; COOLDOWN = 5
ER_Q = 66; ATR_Q = 55; LOOKBACK = 252
DIVGATE = 3; CONFIRM = 3
BUY_DD_LOOKBACK = 20; BUY_DD_THRESHOLD = 0.05
DIV_CLIP = 3.0      # Phase 1 결론
FORCE_NORM = 2.0     # Phase 1~2 결론
MACD_FAST = 12; MACD_SLOW = 26; MACD_SIGNAL = 9  # Phase 2 결론

# Sweep 대상: W_force / W_div 비율 (합계=1.0)
WEIGHT_COMBOS = [
    (0.80, 0.20, "80/20"),
    (0.70, 0.30, "70/30"),
    (0.60, 0.40, "60/40"),   # CURRENT
    (0.50, 0.50, "50/50"),
    (0.40, 0.60, "40/60"),
    (0.30, 0.70, "30/70"),
]

CURRENT_WEIGHTS = (0.60, 0.40, "60/40")


# ═══════════════════════════════════════════════════════════
# Core Functions
# ═══════════════════════════════════════════════════════════
def download_max(ticker):
    t = yf.Ticker(ticker)
    df = t.history(period='max', auto_adjust=True)
    if df is None or len(df) == 0:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.dropna(subset=['Close', 'Volume'])
    df = df[df['Close'] > 0]
    return df


def calc_force_macd_vel(df, fast=12, slow=26, signal=9):
    """VN60 Force: 속도(velocity) 기반 MACD histogram"""
    p_vel = df['Close'].pct_change().fillna(0)
    v_norm = df['Volume'] / df['Volume'].rolling(20).mean().replace(0, np.nan).fillna(df['Volume'])
    force = v_norm * p_vel
    fm = force.ewm(span=fast, adjust=False).mean() - force.ewm(span=slow, adjust=False).mean()
    fs = fm.ewm(span=signal, adjust=False).mean()
    return (fm - fs).rename('PV_Force_Hist')


def build_synthetic_lev(close, leverage, expense_daily):
    daily_ret = np.diff(close) / close[:-1]
    lev_price = np.zeros(len(close))
    lev_price[0] = close[0]
    for i in range(1, len(close)):
        lev_ret = leverage * daily_ret[i - 1] - expense_daily
        lev_price[i] = lev_price[i - 1] * (1 + lev_ret)
        if lev_price[i] < 0.001:
            lev_price[i] = 0.001
    return lev_price


def calc_score_vn60(pv_div_raw, consec_arr, pv_fh_vel, n, w, divgate_days, force_norm, w_force, w_div):
    """VN60 스코어: 가중치 비율 파라미터화"""
    scores = np.zeros(n)
    for i in range(max(60, w), n):
        s_div = pv_div_raw[i] if consec_arr[i] >= divgate_days else 0.0
        fhr_std = pv_fh_vel.iloc[max(0, i - w):i].std() + 1e-10
        s_force = np.clip(pv_fh_vel.iloc[i] / (force_norm * fhr_std), -1, 1)
        scores[i] = w_force * s_force + w_div * s_div
    return scores


def precompute_div(pv_div, n, div_clip):
    """DivGate pre-computation (1회만)"""
    raw_div = np.array([np.clip(pv_div.iloc[i] / div_clip, -1, 1) for i in range(n)])
    consec = np.zeros(n, dtype=int)
    for i in range(1, n):
        if raw_div[i] != 0 and np.sign(raw_div[i]) == np.sign(raw_div[i - 1]):
            consec[i] = consec[i - 1] + 1
        elif raw_div[i] != 0:
            consec[i] = 1
    return raw_div, consec


def get_buy_signals(df_s, score_arr):
    score_series = pd.Series(score_arr, index=df_s.index)
    events = detect_signal_events(score_series, th=SIGNAL_TH, cooldown=COOLDOWN)
    pf = build_price_filter(df_s, er_q=ER_Q, atr_q=ATR_Q, lookback=LOOKBACK)
    close_vals = df_s['Close'].values
    rolling_high = pd.Series(close_vals).rolling(BUY_DD_LOOKBACK, min_periods=1).max().values
    n = len(df_s)
    buy_indices = []
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
        rh = rolling_high[pidx]
        dd = (rh - close_vals[pidx]) / rh if rh > 0 else 0
        if dd < BUY_DD_THRESHOLD:
            continue
        buy_indices.append(ci)
    return buy_indices


def simulate(close, close_2x, close_3x, buy_indices, dates):
    n = len(close)
    month_periods = pd.Series(dates).dt.to_period('M')
    month_map = {}
    for i, mp in enumerate(month_periods):
        key = str(mp)
        if key not in month_map:
            month_map[key] = {'first': i, 'last': i}
        else:
            month_map[key]['last'] = i
    sorted_months = sorted(month_map.keys())
    buy_set = set(buy_indices)

    cash_a = 0.0; sh_1x_a = 0.0; sh_2x_a = 0.0
    cash_b = 0.0; sh_1x_b = 0.0; sh_3x_b = 0.0
    cash_c = 0.0; sh_1x_c = 0.0
    yr_data = {}; prev_yr = None; total_dep = 0.0

    def pf_a(idx): return sh_1x_a * close[idx] + sh_2x_a * close_2x[idx] + cash_a
    def pf_b(idx): return sh_1x_b * close[idx] + sh_3x_b * close_3x[idx] + cash_b
    def pf_c(idx): return sh_1x_c * close[idx] + cash_c

    for mk in sorted_months:
        mm = month_map[mk]
        fi, li = mm['first'], mm['last']
        yr = int(mk[:4])
        if yr != prev_yr:
            if prev_yr is not None:
                ref = fi - 1 if fi > 0 else fi
                yr_data[prev_yr]['end_a'] = pf_a(ref)
                yr_data[prev_yr]['end_b'] = pf_b(ref)
                yr_data[prev_yr]['end_c'] = pf_c(ref)
            yr_data[yr] = {
                'start_a': pf_a(fi), 'start_b': pf_b(fi), 'start_c': pf_c(fi),
                'deposits': 0.0, 'sigs': 0,
            }
            prev_yr = yr
        cash_a += MONTHLY_DEPOSIT; cash_b += MONTHLY_DEPOSIT; cash_c += MONTHLY_DEPOSIT
        yr_data[yr]['deposits'] += MONTHLY_DEPOSIT; total_dep += MONTHLY_DEPOSIT
        for day_idx in range(fi, li + 1):
            if day_idx in buy_set:
                if cash_a > 1.0:
                    amt = cash_a * SIGNAL_BUY_PCT
                    sh_2x_a += amt / close_2x[day_idx]; cash_a -= amt
                if cash_b > 1.0:
                    amt = cash_b * SIGNAL_BUY_PCT
                    sh_3x_b += amt / close_3x[day_idx]; cash_b -= amt
                yr_data[yr]['sigs'] += 1
        if cash_a > 1.0: sh_1x_a += cash_a / close[li]; cash_a = 0.0
        if cash_b > 1.0: sh_1x_b += cash_b / close[li]; cash_b = 0.0
        if cash_c > 1.0: sh_1x_c += cash_c / close[li]; cash_c = 0.0

    if prev_yr is not None:
        yr_data[prev_yr]['end_a'] = pf_a(n - 1)
        yr_data[prev_yr]['end_b'] = pf_b(n - 1)
        yr_data[prev_yr]['end_c'] = pf_c(n - 1)

    yr_results = []
    for yr in sorted(yr_data.keys()):
        yd = yr_data[yr]
        if 'end_a' not in yd:
            continue
        rets = {}
        for mode in ['a', 'b', 'c']:
            d = yd[f'start_{mode}'] + yd['deposits'] * 0.5
            if d > 10:
                val = (yd[f'end_{mode}'] - yd[f'start_{mode}'] - yd['deposits']) / d * 100
                rets[mode] = val if np.isfinite(val) else 0.0
            else:
                rets[mode] = 0
        yr_results.append({
            'yr': yr,
            'ret_2x': rets['a'], 'ret_3x': rets['b'], 'ret_dca': rets['c'],
            'edge_2x': rets['a'] - rets['c'], 'edge_3x': rets['b'] - rets['c'],
            'sigs': yd['sigs'],
        })

    # Hit rate & forward return
    hits_90 = 0; total_90 = 0; fwd_90 = []
    for idx in buy_indices:
        if idx + 90 < n:
            total_90 += 1
            fr = (close[idx + 90] / close[idx] - 1) * 100
            fwd_90.append(fr)
            if fr > 0: hits_90 += 1

    avg_2x = np.nanmean([r['ret_2x'] for r in yr_results])
    avg_3x = np.nanmean([r['ret_3x'] for r in yr_results])
    avg_dca = np.nanmean([r['ret_dca'] for r in yr_results])
    edge_2x = avg_2x - avg_dca; edge_3x = avg_3x - avg_dca
    worst_2x = min(r['edge_2x'] for r in yr_results) if yr_results else 0
    worst_3x = min(r['edge_3x'] for r in yr_results) if yr_results else 0

    bear_yrs = [r for r in yr_results if r['ret_dca'] < -5]
    bull_yrs = [r for r in yr_results if r['ret_dca'] > 15]

    return {
        'yr_results': yr_results,
        'avg_2x': avg_2x, 'avg_3x': avg_3x, 'avg_dca': avg_dca,
        'edge_2x': edge_2x, 'edge_3x': edge_3x,
        'worst_2x': worst_2x, 'worst_3x': worst_3x,
        'eff_2x': edge_2x / abs(worst_2x) if abs(worst_2x) > 0.1 else 0,
        'eff_3x': edge_3x / abs(worst_3x) if abs(worst_3x) > 0.1 else 0,
        'total_sigs': sum(r['sigs'] for r in yr_results),
        'hit_rate_90': (hits_90 / total_90 * 100) if total_90 > 0 else 0,
        'avg_fwd_90': np.mean(fwd_90) if fwd_90 else 0,
        'n_years': len(yr_results),
        'final_a': pf_a(n - 1), 'final_b': pf_b(n - 1), 'final_c': pf_c(n - 1),
        'total_dep': total_dep,
        'bear_edge_2x': np.mean([r['edge_2x'] for r in bear_yrs]) if bear_yrs else 0,
        'bull_edge_2x': np.mean([r['edge_2x'] for r in bull_yrs]) if bull_yrs else 0,
        'n_bear': len(bear_yrs), 'n_bull': len(bull_yrs),
    }


# ═══════════════════════════════════════════════════════════
# Data Download (1회)
# ═══════════════════════════════════════════════════════════
print("=" * 130)
print("  VN60 PHASE 3: Weight Ratio Sweep (W_force / W_div)")
print("  전략: 월 $500 → 시그널 시 50% 레버리지(2x/3x) 매수 → 월말 잔여금 1x 매수")
print(f"  고정: MACD(12/26/9), FN/2.0, DivClip/3.0 (Phase 1~2 결론)")
print("=" * 130)

print("\n  데이터 로딩...")
ticker_data = {}

for tk in TICKERS:
    df = download_max(tk)
    if df is None or len(df) < 300:
        continue
    try:
        df_s = smooth_earnings_volume(df, ticker=tk)
    except Exception:
        df_s = df.copy()

    close = df_s['Close'].values
    dates = df_s.index
    n = len(df_s)
    close_2x = build_synthetic_lev(close, 2.0, EXPENSE_2X)
    close_3x = build_synthetic_lev(close, 3.0, EXPENSE_3X)

    # Div (1회)
    pv_div = calc_pv_divergence(df_s, V4_W)
    raw_div, consec = precompute_div(pv_div, n, DIV_CLIP)

    # Force (1회 - MACD 고정)
    pv_fh_vel = calc_force_macd_vel(df_s, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)

    ticker_data[tk] = (df_s, close, close_2x, close_3x, dates, n, raw_div, consec, pv_fh_vel)
    print(f"    {tk}: {n} bars, {(dates[-1]-dates[0]).days/365.25:.1f}yr")

tks = list(ticker_data.keys())
print(f"  {len(tks)} tickers loaded.\n")


# ═══════════════════════════════════════════════════════════
# Run All Combos: 6 weight ratios
# ═══════════════════════════════════════════════════════════
print("  전체 조합 시뮬레이션 실행...")
master = {}  # master[(label)] = {ticker: result}

for w_force, w_div, label in WEIGHT_COMBOS:
    results = {}
    for tk in tks:
        df_s, close, close_2x, close_3x, dates, n, raw_div, consec, pv_fh_vel = ticker_data[tk]
        score_arr = calc_score_vn60(raw_div, consec, pv_fh_vel, n,
                                    w=V4_W, divgate_days=DIVGATE, force_norm=FORCE_NORM,
                                    w_force=w_force, w_div=w_div)
        buys = get_buy_signals(df_s, score_arr)
        results[tk] = simulate(close, close_2x, close_3x, buys, dates)
    master[label] = results

    avg_2x = np.nanmean([results[tk]['avg_2x'] for tk in tks])
    avg_3x = np.nanmean([results[tk]['avg_3x'] for tk in tks])
    avg_dca = np.nanmean([results[tk]['avg_dca'] for tk in tks])
    print(f"    W={label}  2x={avg_2x:>+6.2f}%  3x={avg_3x:>+6.2f}%  DCA={avg_dca:>+6.2f}%")

CURRENT_KEY = "60/40"


# ═══════════════════════════════════════════════════════════
# [1] 전체 조합 랭킹 (연평균수익률)
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 130}")
print(f"  [1] 전체 조합 랭킹: 14 티커 평균 연평균수익률")
print(f"{'=' * 130}")

print(f"\n  {'Rank':>4s} {'W_force/W_div':>14s}"
      f" │ {'VN60+2x':>9s} {'VN60+3x':>9s} {'DCA':>9s}"
      f" │ {'Edge2x':>8s} {'Edge3x':>8s} {'Eff2x':>7s}"
      f" │ {'Hit90':>6s} {'Fwd90':>8s} {'Sig/yr':>7s}"
      f" │ {'Note':>12s}")
print(f"  {'=' * 120}")

all_labels = [lbl for _, _, lbl in WEIGHT_COMBOS]
ranking = sorted(all_labels,
                 key=lambda k: np.nanmean([master[k][tk]['avg_2x'] for tk in tks]),
                 reverse=True)

for i, label in enumerate(ranking):
    m = master[label]
    a2 = np.nanmean([m[tk]['avg_2x'] for tk in tks])
    a3 = np.nanmean([m[tk]['avg_3x'] for tk in tks])
    ad = np.nanmean([m[tk]['avg_dca'] for tk in tks])
    e2 = np.nanmean([m[tk]['edge_2x'] for tk in tks])
    e3 = np.nanmean([m[tk]['edge_3x'] for tk in tks])
    ef = np.nanmean([m[tk]['eff_2x'] for tk in tks])
    hr = np.nanmean([m[tk]['hit_rate_90'] for tk in tks])
    fw = np.nanmean([m[tk]['avg_fwd_90'] for tk in tks])
    sy = np.nanmean([m[tk]['total_sigs'] / max(m[tk]['n_years'], 1) for tk in tks])

    note = ""
    if label == CURRENT_KEY: note = "◀ CURRENT"
    elif i == 0: note = "★ BEST"

    print(f"  {i+1:>4d} {label:>14s}"
          f" │ {a2:>+8.2f}% {a3:>+8.2f}% {ad:>+8.2f}%"
          f" │ {e2:>+7.2f}% {e3:>+7.2f}% {ef:>+6.3f}"
          f" │ {hr:>5.1f}% {fw:>+7.1f}% {sy:>6.1f}"
          f" │ {note:>12s}")


# ═══════════════════════════════════════════════════════════
# [2] 가중치별 상세 메트릭 비교
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 130}")
print(f"  [2] 가중치별 상세 메트릭 비교")
print(f"{'=' * 130}")

header_labels = [lbl for _, _, lbl in WEIGHT_COMBOS]
print(f"\n  {'Metric':<20s}", end="")
for lbl in header_labels:
    marker = " ◀" if lbl == CURRENT_KEY else "  "
    print(f"  {lbl+marker:>10s}", end="")
print()
print(f"  {'-' * (20 + 12 * len(header_labels))}")

for metric_name, metric_key in [("연평균수익률 2x", 'avg_2x'),
                                 ("연평균수익률 3x", 'avg_3x'),
                                 ("Edge 2x (%p)", 'edge_2x'),
                                 ("Edge 3x (%p)", 'edge_3x'),
                                 ("Efficiency 2x", 'eff_2x'),
                                 ("Hit Rate 90d (%)", 'hit_rate_90'),
                                 ("Fwd Return 90d", 'avg_fwd_90'),
                                 ("시그널/년", None)]:
    print(f"  {metric_name:<20s}", end="")
    for lbl in header_labels:
        m = master[lbl]
        if metric_key:
            val = np.nanmean([m[tk][metric_key] for tk in tks])
        else:
            val = np.nanmean([m[tk]['total_sigs'] / max(m[tk]['n_years'], 1) for tk in tks])

        if metric_key == 'eff_2x':
            print(f"  {val:>+9.3f}", end="")
        elif metric_key is None:
            print(f"  {val:>9.1f}", end="")
        else:
            print(f"  {val:>+8.2f}%", end="")
    print()


# ═══════════════════════════════════════════════════════════
# [3] 티커별 연평균수익률: BEST vs CURRENT
# ═══════════════════════════════════════════════════════════
best_label = ranking[0]

print(f"\n{'=' * 130}")
print(f"  [3] 티커별 연평균수익률: BEST({best_label}) vs CURRENT(60/40)")
print(f"{'=' * 130}")

print(f"\n  {'Ticker':<7s} {'Sect':<8s} {'Yrs':>4s}"
      f" │ ──── CURRENT(60/40) ────"
      f" │ ───── BEST({best_label}) ─────"
      f" │ ── 차이 ──")
print(f"  {'':7s} {'':8s} {'':>4s}"
      f" │ {'2x':>8s} {'3x':>8s} {'DCA':>8s}"
      f" │ {'2x':>8s} {'3x':>8s} {'DCA':>8s}"
      f" │ {'d2x':>7s} {'d3x':>7s}")
print(f"  {'=' * 100}")

d2_all = []; d3_all = []
for tk in tks:
    c = master[CURRENT_KEY][tk]; n_ = master[best_label][tk]
    d2 = n_['avg_2x'] - c['avg_2x']; d3 = n_['avg_3x'] - c['avg_3x']
    d2_all.append(d2); d3_all.append(d3)
    sect = TICKERS.get(tk, '')
    print(f"  {tk:<7s} {sect:<8s} {c['n_years']:>4d}"
          f" │ {c['avg_2x']:>+7.1f}% {c['avg_3x']:>+7.1f}% {c['avg_dca']:>+7.1f}%"
          f" │ {n_['avg_2x']:>+7.1f}% {n_['avg_3x']:>+7.1f}% {n_['avg_dca']:>+7.1f}%"
          f" │ {d2:>+6.2f} {d3:>+6.2f}")

print(f"  {'-' * 100}")
ca2 = np.nanmean([master[CURRENT_KEY][tk]['avg_2x'] for tk in tks])
ca3 = np.nanmean([master[CURRENT_KEY][tk]['avg_3x'] for tk in tks])
cad = np.nanmean([master[CURRENT_KEY][tk]['avg_dca'] for tk in tks])
na2 = np.nanmean([master[best_label][tk]['avg_2x'] for tk in tks])
na3 = np.nanmean([master[best_label][tk]['avg_3x'] for tk in tks])
nad = np.nanmean([master[best_label][tk]['avg_dca'] for tk in tks])
print(f"  {'AVG':<7s} {'':8s} {'':>4s}"
      f" │ {ca2:>+7.1f}% {ca3:>+7.1f}% {cad:>+7.1f}%"
      f" │ {na2:>+7.1f}% {na3:>+7.1f}% {nad:>+7.1f}%"
      f" │ {np.mean(d2_all):>+6.2f} {np.mean(d3_all):>+6.2f}")

better = sum(1 for d in d2_all if d > 0.05)
worse = sum(1 for d in d2_all if d < -0.05)
print(f"\n  2x 기준: {better}개 개선 / {worse}개 악화 / {len(tks)-better-worse}개 동일")


# ═══════════════════════════════════════════════════════════
# [4] 시그널 품질 비교
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 130}")
print(f"  [4] 시그널 품질: BEST({best_label}) vs CURRENT(60/40)")
print(f"{'=' * 130}")

print(f"\n  {'Ticker':<7s}"
      f" │ {'CUR sig':>8s} {'CUR hit90':>9s} {'CUR fwd90':>10s}"
      f" │ {'NEW sig':>8s} {'NEW hit90':>9s} {'NEW fwd90':>10s}"
      f" │ {'dHit':>6s} {'dFwd':>7s}")
print(f"  {'=' * 90}")

for tk in tks:
    c = master[CURRENT_KEY][tk]; n_ = master[best_label][tk]
    dh = n_['hit_rate_90'] - c['hit_rate_90']
    df_ = n_['avg_fwd_90'] - c['avg_fwd_90']
    print(f"  {tk:<7s}"
          f" │ {c['total_sigs']:>7d} {c['hit_rate_90']:>8.1f}% {c['avg_fwd_90']:>+9.2f}%"
          f" │ {n_['total_sigs']:>7d} {n_['hit_rate_90']:>8.1f}% {n_['avg_fwd_90']:>+9.2f}%"
          f" │ {dh:>+5.1f} {df_:>+6.2f}")

print(f"  {'-' * 90}")
ch = np.nanmean([master[CURRENT_KEY][tk]['hit_rate_90'] for tk in tks])
cf = np.nanmean([master[CURRENT_KEY][tk]['avg_fwd_90'] for tk in tks])
nh = np.nanmean([master[best_label][tk]['hit_rate_90'] for tk in tks])
nf = np.nanmean([master[best_label][tk]['avg_fwd_90'] for tk in tks])
print(f"  {'AVG':<7s}"
      f" │ {'':>8s} {ch:>8.1f}% {cf:>+9.2f}%"
      f" │ {'':>8s} {nh:>8.1f}% {nf:>+9.2f}%"
      f" │ {nh-ch:>+5.1f} {nf-cf:>+6.2f}")


# ═══════════════════════════════════════════════════════════
# [5] 최종 자산 비교
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 130}")
print(f"  [5] 최종 자산가치: BEST({best_label}) vs CURRENT(60/40)")
print(f"{'=' * 130}")

print(f"\n  {'Ticker':<7s} {'입금':>9s}"
      f" │ {'CUR 2x':>11s} {'CUR 3x':>11s} {'DCA':>11s}"
      f" │ {'NEW 2x':>11s} {'NEW 3x':>11s}"
      f" │ {'d2x$':>10s} {'Better':>7s}")
print(f"  {'=' * 105}")

tc2=0; tc3=0; tcd=0; tn2=0; tn3=0; tnd=0; tdep=0
for tk in tks:
    c = master[CURRENT_KEY][tk]; n_ = master[best_label][tk]
    dep = c['total_dep']
    d2 = n_['final_a'] - c['final_a']
    b = "BEST" if d2 > 0 else "CUR"
    tc2 += c['final_a']; tc3 += c['final_b']; tcd += c['final_c']
    tn2 += n_['final_a']; tn3 += n_['final_b']; tnd += n_['final_c']
    tdep += dep
    print(f"  {tk:<7s} ${dep:>8,.0f}"
          f" │ ${c['final_a']:>10,.0f} ${c['final_b']:>10,.0f} ${c['final_c']:>10,.0f}"
          f" │ ${n_['final_a']:>10,.0f} ${n_['final_b']:>10,.0f}"
          f" │ ${d2:>+9,.0f} {b:>7s}")

print(f"  {'-' * 105}")
print(f"  {'TOTAL':<7s} ${tdep:>8,.0f}"
      f" │ ${tc2:>10,.0f} ${tc3:>10,.0f} ${tcd:>10,.0f}"
      f" │ ${tn2:>10,.0f} ${tn3:>10,.0f}"
      f" │ ${tn2-tc2:>+9,.0f} {'BEST' if tn2>tc2 else 'CUR':>7s}")


# ═══════════════════════════════════════════════════════════
# [6] QQQ / VOO 연도별
# ═══════════════════════════════════════════════════════════
for target in ['QQQ', 'VOO']:
    if target not in tks:
        continue

    print(f"\n{'=' * 130}")
    print(f"  [6] {target} 연도별: CURRENT(60/40) vs BEST({best_label})")
    print(f"{'=' * 130}")

    cur = master[CURRENT_KEY][target]
    new = master[best_label][target]

    print(f"\n  {'Year':>6s}"
          f" │ {'CUR 2x':>8s} {'CUR 3x':>8s} {'DCA':>8s} {'Ce2x':>6s}"
          f" │ {'NEW 2x':>8s} {'NEW 3x':>8s} {'DCA':>8s} {'Ne2x':>6s}"
          f" │ {'d2x':>6s}")
    print(f"  {'-' * 90}")

    for cr, nr in zip(cur['yr_results'], new['yr_results']):
        d2 = nr['ret_2x'] - cr['ret_2x']
        regime = ""
        if cr['ret_dca'] < -5: regime = " B"
        elif cr['ret_dca'] > 15: regime = " U"
        print(f"  {cr['yr']:>6d}"
              f" │ {cr['ret_2x']:>+7.1f}% {cr['ret_3x']:>+7.1f}% {cr['ret_dca']:>+7.1f}% {cr['edge_2x']:>+5.1f}%"
              f" │ {nr['ret_2x']:>+7.1f}% {nr['ret_3x']:>+7.1f}% {nr['ret_dca']:>+7.1f}% {nr['edge_2x']:>+5.1f}%"
              f" │ {d2:>+5.1f}{regime}")

    print(f"  {'-' * 90}")
    ca = np.nanmean([r['ret_2x'] for r in cur['yr_results']])
    na_ = np.nanmean([r['ret_2x'] for r in new['yr_results']])
    print(f"  {'AVG':>6s} │ {ca:>+7.1f}% {'':>8s} {'':>8s} {'':>6s}"
          f" │ {na_:>+7.1f}% {'':>8s} {'':>8s} {'':>6s} │ {na_-ca:>+5.2f}")

    print(f"\n  {target}: CUR=${cur['final_a']:,.0f}  BEST=${new['final_a']:,.0f}"
          f"  (${new['final_a']-cur['final_a']:+,.0f})"
          f"  Hit90: {cur['hit_rate_90']:.1f}% → {new['hit_rate_90']:.1f}%")


# ═══════════════════════════════════════════════════════════
# [7] Bear/Bull 국면 분석
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 130}")
print(f"  [7] 시장 국면별 Edge(2x vs DCA): BEST({best_label}) vs CURRENT(60/40)")
print(f"{'=' * 130}")

print(f"\n  {'Ticker':<7s}"
      f" │ {'CUR Bear':>9s} {'CUR Bull':>9s}"
      f" │ {'NEW Bear':>9s} {'NEW Bull':>9s}"
      f" │ {'dBear':>7s} {'dBull':>7s}")
print(f"  {'=' * 70}")

for tk in tks:
    c = master[CURRENT_KEY][tk]; n_ = master[best_label][tk]
    db = n_['bear_edge_2x'] - c['bear_edge_2x']
    dbu = n_['bull_edge_2x'] - c['bull_edge_2x']
    print(f"  {tk:<7s}"
          f" │ {c['bear_edge_2x']:>+8.2f}% {c['bull_edge_2x']:>+8.2f}%"
          f" │ {n_['bear_edge_2x']:>+8.2f}% {n_['bull_edge_2x']:>+8.2f}%"
          f" │ {db:>+6.2f} {dbu:>+6.2f}")

print(f"  {'-' * 70}")
cb = np.nanmean([master[CURRENT_KEY][tk]['bear_edge_2x'] for tk in tks])
cbu = np.nanmean([master[CURRENT_KEY][tk]['bull_edge_2x'] for tk in tks])
nb = np.nanmean([master[best_label][tk]['bear_edge_2x'] for tk in tks])
nbu = np.nanmean([master[best_label][tk]['bull_edge_2x'] for tk in tks])
print(f"  {'AVG':<7s}"
      f" │ {cb:>+8.2f}% {cbu:>+8.2f}%"
      f" │ {nb:>+8.2f}% {nbu:>+8.2f}%"
      f" │ {nb-cb:>+6.2f} {nbu-cbu:>+6.2f}")


# ═══════════════════════════════════════════════════════════
# [8] 전체 가중치별 티커 히트맵 (Edge 2x)
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 130}")
print(f"  [8] 가중치별 티커 Edge(2x vs DCA) 히트맵 (%p)")
print(f"{'=' * 130}")

print(f"\n  {'Ticker':<7s}", end="")
for _, _, lbl in WEIGHT_COMBOS:
    marker = " ◀" if lbl == CURRENT_KEY else "  "
    print(f"  {lbl+marker:>8s}", end="")
print(f"  {'Best':>6s}")
print(f"  {'-' * (7 + 10 * len(WEIGHT_COMBOS) + 8)}")

for tk in tks:
    print(f"  {tk:<7s}", end="")
    best_w = max(all_labels, key=lambda l: master[l][tk]['edge_2x'])
    for _, _, lbl in WEIGHT_COMBOS:
        e = master[lbl][tk]['edge_2x']
        print(f"  {e:>+7.2f}%", end="")
    print(f"  {best_w:>6s}")

print(f"  {'-' * (7 + 10 * len(WEIGHT_COMBOS) + 8)}")
print(f"  {'AVG':<7s}", end="")
for _, _, lbl in WEIGHT_COMBOS:
    e = np.nanmean([master[lbl][tk]['edge_2x'] for tk in tks])
    print(f"  {e:>+7.2f}%", end="")
best_avg_w = max(all_labels, key=lambda l: np.nanmean([master[l][tk]['edge_2x'] for tk in tks]))
print(f"  {best_avg_w:>6s}")

# Count best per ticker
print(f"\n  티커별 최적 가중치 분포:")
from collections import Counter
best_per_tk = [max(all_labels, key=lambda l: master[l][tk]['edge_2x']) for tk in tks]
cnt = Counter(best_per_tk)
for lbl in [l for _, _, l in WEIGHT_COMBOS]:
    c = cnt.get(lbl, 0)
    bar = "█" * c
    print(f"    {lbl}: {c:>2d}개 {bar}")


# ═══════════════════════════════════════════════════════════
# GRAND SUMMARY
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 130}")
print(f"  GRAND SUMMARY: Phase 3 결론")
print(f"{'=' * 130}")

print(f"\n  [A] 가중치별 성과 요약:")
for _, _, lbl in WEIGHT_COMBOS:
    m = master[lbl]
    a2 = np.nanmean([m[tk]['avg_2x'] for tk in tks])
    e2 = np.nanmean([m[tk]['edge_2x'] for tk in tks])
    hr = np.nanmean([m[tk]['hit_rate_90'] for tk in tks])
    sy = np.nanmean([m[tk]['total_sigs'] / max(m[tk]['n_years'], 1) for tk in tks])
    marker = " ◀ CURRENT" if lbl == CURRENT_KEY else ""
    print(f"    W={lbl}  →  2x={a2:>+.2f}%  edge={e2:>+.2f}%  hit90={hr:.1f}%  sig/yr={sy:.1f}{marker}")

print(f"""
  +---------------------------------------------------------------------+
  |                    14 티커 평균 연평균수익률                          |
  +---------------------------------------------------------------------+
  |  Config                VN60+2x    VN60+3x    DCA       Edge2x      |
  +---------------------------------------------------------------------+
  |  CURRENT (60/40)       {ca2:>+7.2f}%   {ca3:>+7.2f}%   {cad:>+7.2f}%   {ca2-cad:>+6.2f}%     |
  +---------------------------------------------------------------------+
  |  BEST ({best_label:5s})          {na2:>+7.2f}%   {na3:>+7.2f}%   {nad:>+7.2f}%   {na2-nad:>+6.2f}%     |
  +---------------------------------------------------------------------+
  |  DELTA                 {na2-ca2:>+7.2f}%   {na3-ca3:>+7.2f}%   {nad-cad:>+7.2f}%   {(na2-nad)-(ca2-cad):>+6.2f}%     |
  +---------------------------------------------------------------------+
  |  Hit Rate 90d:  {ch:.1f}% → {nh:.1f}%  ({nh-ch:>+.1f}%p)                         |
  |  Ticker W/L:    {better} improved / {worse} degraded / {len(tks)-better-worse} same                |
  |  Total Assets:  ${tn2-tc2:>+,.0f}                                   |
  +---------------------------------------------------------------------+
""")

print("=" * 130)
print("  Done.")
