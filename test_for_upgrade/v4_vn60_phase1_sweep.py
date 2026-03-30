"""
VN60 Phase 1 Parameter Sweep
=============================
실험 A: Div 클리핑 팩터 sweep (Force norm = /2 고정)
실험 B: Force 정규화 팩터 sweep (Div clip = /3 고정)
실험 C: 상위 조합 교차 검증

백테스트 전략 (기존 VN60과 동일):
  - 월 $500 입금
  - 시그널 시 가용자금의 50% → 2x 레버리지 매수
  - 시그널 시 가용자금의 50% → 3x 레버리지 매수 (별도 포트폴리오)
  - 월말 잔여 자금 → 본주(1x) 매수
  - DCA = 월 $500 순수 1x 매수 (벤치마크)

측정: Edge vs DCA, Efficiency, Hit Rate(90d), 시그널 빈도
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
EXPENSE_2X = 0.0095 / 252   # ~0.95% annual
EXPENSE_3X = 0.0100 / 252   # ~1.00% annual

V4_W = 20; SIGNAL_TH = 0.15; COOLDOWN = 5
ER_Q = 66; ATR_Q = 55; LOOKBACK = 252
DIVGATE = 3; CONFIRM = 3
BUY_DD_LOOKBACK = 20; BUY_DD_THRESHOLD = 0.05

# Sweep 대상
DIV_CLIP_VALUES = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
FORCE_NORM_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0]

# 현재 기본값
DEFAULT_DIV_CLIP = 3.0
DEFAULT_FORCE_NORM = 2.0


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
    """합성 레버리지 가격 시리즈"""
    daily_ret = np.diff(close) / close[:-1]
    lev_price = np.zeros(len(close))
    lev_price[0] = close[0]
    for i in range(1, len(close)):
        lev_ret = leverage * daily_ret[i - 1] - expense_daily
        lev_price[i] = lev_price[i - 1] * (1 + lev_ret)
        if lev_price[i] < 0.001:
            lev_price[i] = 0.001
    return lev_price


def calc_score_vn60(df, pv_div, pv_fh_vel, w, divgate_days, div_clip, force_norm):
    """VN60 스코어: 파라미터화된 div_clip, force_norm"""
    n = len(df)

    # raw_div with parameterized clip factor
    raw_div = np.array([np.clip(pv_div.iloc[i] / div_clip, -1, 1) for i in range(n)])

    # DivGate: 연속 같은 부호 일수
    consec = np.zeros(n, dtype=int)
    for i in range(1, n):
        if raw_div[i] != 0 and np.sign(raw_div[i]) == np.sign(raw_div[i - 1]):
            consec[i] = consec[i - 1] + 1
        elif raw_div[i] != 0:
            consec[i] = 1

    scores = np.zeros(n)
    for i in range(max(60, w), n):
        s_div = raw_div[i] if consec[i] >= divgate_days else 0.0
        fhr_std = pv_fh_vel.iloc[max(0, i - w):i].std() + 1e-10
        s_force = np.clip(pv_fh_vel.iloc[i] / (force_norm * fhr_std), -1, 1)
        scores[i] = 0.60 * s_force + 0.40 * s_div

    return pd.Series(scores, index=df.index)


def get_buy_signals(df_s, score_series):
    """시그널 감지 + 필터 (DD Gate 포함)"""
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
    """DCA / VN60+2x / VN60+3x 동시 시뮬레이션"""
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

    # A: VN60+2x, B: VN60+3x, C: Pure DCA
    cash_a = 0.0; sh_1x_a = 0.0; sh_2x_a = 0.0
    cash_b = 0.0; sh_1x_b = 0.0; sh_3x_b = 0.0
    cash_c = 0.0; sh_1x_c = 0.0
    yr_data = {}; prev_yr = None

    def pf_a(idx):
        return sh_1x_a * close[idx] + sh_2x_a * close_2x[idx] + cash_a
    def pf_b(idx):
        return sh_1x_b * close[idx] + sh_3x_b * close_3x[idx] + cash_b
    def pf_c(idx):
        return sh_1x_c * close[idx] + cash_c

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
        yr_data[yr]['deposits'] += MONTHLY_DEPOSIT

        for day_idx in range(fi, li + 1):
            if day_idx in buy_set:
                if cash_a > 1.0:
                    amt = cash_a * SIGNAL_BUY_PCT
                    sh_2x_a += amt / close_2x[day_idx]
                    cash_a -= amt
                if cash_b > 1.0:
                    amt = cash_b * SIGNAL_BUY_PCT
                    sh_3x_b += amt / close_3x[day_idx]
                    cash_b -= amt
                yr_data[yr]['sigs'] += 1

        if cash_a > 1.0:
            sh_1x_a += cash_a / close[li]; cash_a = 0.0
        if cash_b > 1.0:
            sh_1x_b += cash_b / close[li]; cash_b = 0.0
        if cash_c > 1.0:
            sh_1x_c += cash_c / close[li]; cash_c = 0.0

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
            'edge_2x': rets['a'] - rets['c'],
            'edge_3x': rets['b'] - rets['c'],
            'sigs': yd['sigs'],
        })

    total_sigs = sum(r['sigs'] for r in yr_results)
    avg_2x = np.nanmean([r['ret_2x'] for r in yr_results])
    avg_3x = np.nanmean([r['ret_3x'] for r in yr_results])
    avg_dca = np.nanmean([r['ret_dca'] for r in yr_results])
    edge_2x = avg_2x - avg_dca
    edge_3x = avg_3x - avg_dca
    worst_2x = min(r['edge_2x'] for r in yr_results) if yr_results else 0
    worst_3x = min(r['edge_3x'] for r in yr_results) if yr_results else 0
    eff_2x = edge_2x / abs(worst_2x) if abs(worst_2x) > 0.1 else 0
    eff_3x = edge_3x / abs(worst_3x) if abs(worst_3x) > 0.1 else 0

    # Hit rate (90d forward)
    n_close = len(close)
    hits_90 = 0; total_90 = 0
    for idx in buy_indices:
        if idx + 90 < n_close:
            total_90 += 1
            if close[idx + 90] > close[idx]:
                hits_90 += 1
    hit_rate_90 = (hits_90 / total_90 * 100) if total_90 > 0 else 0

    return {
        'yr_results': yr_results,
        'avg_2x': avg_2x, 'avg_3x': avg_3x, 'avg_dca': avg_dca,
        'edge_2x': edge_2x, 'edge_3x': edge_3x,
        'worst_2x': worst_2x, 'worst_3x': worst_3x,
        'eff_2x': eff_2x, 'eff_3x': eff_3x,
        'total_sigs': total_sigs,
        'hit_rate_90': hit_rate_90,
        'n_years': len(yr_results),
    }


# ═══════════════════════════════════════════════════════════
# Data Download (1회만)
# ═══════════════════════════════════════════════════════════
print("=" * 120)
print("  VN60 PHASE 1 PARAMETER SWEEP")
print("  Experiment A: Div Clip Factor [1.5 ~ 4.0]  (Force Norm = /2 fixed)")
print("  Experiment B: Force Norm Factor [1.0 ~ 3.0] (Div Clip = /3 fixed)")
print("  Experiment C: Top combinations cross-validation")
print("=" * 120)

print("\n  Downloading data...")
ticker_data = {}  # {ticker: (df_s, close, close_2x, close_3x, dates, pv_div, pv_fh_vel)}

for tk in TICKERS:
    df = download_max(tk)
    if df is None or len(df) < 300:
        print(f"    {tk}: SKIP")
        continue
    try:
        df_s = smooth_earnings_volume(df, ticker=tk)
    except Exception:
        df_s = df.copy()

    close = df_s['Close'].values
    dates = df_s.index
    close_2x = build_synthetic_lev(close, 2.0, EXPENSE_2X)
    close_3x = build_synthetic_lev(close, 3.0, EXPENSE_3X)

    # Pre-compute raw indicators (공통)
    pv_div = calc_pv_divergence(df_s, V4_W)
    pv_fh_vel = calc_force_macd_vel(df_s)

    ticker_data[tk] = (df_s, close, close_2x, close_3x, dates, pv_div, pv_fh_vel)
    print(f"    {tk}: OK ({len(df_s)} bars)")

tks = list(ticker_data.keys())
print(f"\n  {len(tks)} tickers loaded.\n")


def run_sweep(tks, ticker_data, div_clip, force_norm):
    """하나의 (div_clip, force_norm) 조합에 대해 전체 티커 백테스트"""
    results = {}
    for tk in tks:
        df_s, close, close_2x, close_3x, dates, pv_div, pv_fh_vel = ticker_data[tk]
        score = calc_score_vn60(df_s, pv_div, pv_fh_vel,
                                w=V4_W, divgate_days=DIVGATE,
                                div_clip=div_clip, force_norm=force_norm)
        buys = get_buy_signals(df_s, score)
        res = simulate(close, close_2x, close_3x, buys, dates)
        results[tk] = res
    return results


# ═══════════════════════════════════════════════════════════
# EXPERIMENT A: Div Clip Factor Sweep
# ═══════════════════════════════════════════════════════════
print("=" * 120)
print("  EXPERIMENT A: Div Clip Factor Sweep  (Force Norm = /2.0 fixed)")
print("  clip(pv_div / X, -1, 1) 에서 X를 변경")
print("=" * 120)

exp_a = {}  # {div_clip: {ticker: result}}
for dc in DIV_CLIP_VALUES:
    exp_a[dc] = run_sweep(tks, ticker_data, div_clip=dc, force_norm=DEFAULT_FORCE_NORM)
    avg_edge = np.nanmean([exp_a[dc][tk]['edge_2x'] for tk in tks])
    avg_sigs = np.nanmean([exp_a[dc][tk]['total_sigs'] for tk in tks])
    print(f"    DivClip=/{dc:.1f}  avg_edge_2x={avg_edge:>+.2f}%p  avg_sigs={avg_sigs:.0f}")

# Experiment A 결과 테이블
print(f"\n  {'':7s}", end="")
for dc in DIV_CLIP_VALUES:
    marker = " *" if dc == DEFAULT_DIV_CLIP else ""
    print(f" {'/' + f'{dc:.1f}':>5s}{marker:2s}", end="")
print()
print(f"  {'=' * (8 + 8 * len(DIV_CLIP_VALUES))}")

# Per-ticker edge (2x)
for tk in tks:
    print(f"  {tk:<7s}", end="")
    best_dc = max(DIV_CLIP_VALUES, key=lambda d: exp_a[d][tk]['eff_2x'])
    for dc in DIV_CLIP_VALUES:
        e = exp_a[dc][tk]['edge_2x']
        mark = " <" if dc == best_dc else "  "
        print(f" {e:>+5.1f}%{mark}", end="")
    print()

print(f"  {'-' * (8 + 8 * len(DIV_CLIP_VALUES))}")

# Average row
print(f"  {'AVG':<7s}", end="")
for dc in DIV_CLIP_VALUES:
    avg_e = np.nanmean([exp_a[dc][tk]['edge_2x'] for tk in tks])
    print(f" {avg_e:>+5.2f}%p", end="")
print()

# Summary table for Exp A
print(f"\n  Experiment A Summary (2x leverage):")
print(f"  {'DivClip':>8s} {'Edge2x':>8s} {'Edge3x':>8s} {'Eff2x':>8s} {'Eff3x':>8s}"
      f" {'Hit90':>7s} {'Sigs/yr':>8s}")
print(f"  {'-' * 58}")

for dc in DIV_CLIP_VALUES:
    avg_e2 = np.nanmean([exp_a[dc][tk]['edge_2x'] for tk in tks])
    avg_e3 = np.nanmean([exp_a[dc][tk]['edge_3x'] for tk in tks])
    avg_ef2 = np.nanmean([exp_a[dc][tk]['eff_2x'] for tk in tks])
    avg_ef3 = np.nanmean([exp_a[dc][tk]['eff_3x'] for tk in tks])
    avg_hr = np.nanmean([exp_a[dc][tk]['hit_rate_90'] for tk in tks])
    avg_sy = np.nanmean([exp_a[dc][tk]['total_sigs'] / max(exp_a[dc][tk]['n_years'], 1) for tk in tks])
    marker = " ◀ current" if dc == DEFAULT_DIV_CLIP else ""
    print(f"  /{dc:>6.1f} {avg_e2:>+7.2f}% {avg_e3:>+7.2f}% {avg_ef2:>+7.3f} {avg_ef3:>+7.3f}"
          f" {avg_hr:>6.1f}% {avg_sy:>7.1f}{marker}")


# ═══════════════════════════════════════════════════════════
# EXPERIMENT B: Force Norm Factor Sweep
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 120}")
print("  EXPERIMENT B: Force Norm Factor Sweep  (Div Clip = /3.0 fixed)")
print("  clip(force / (X * std), -1, 1) 에서 X를 변경")
print("=" * 120)

exp_b = {}
for fn in FORCE_NORM_VALUES:
    exp_b[fn] = run_sweep(tks, ticker_data, div_clip=DEFAULT_DIV_CLIP, force_norm=fn)
    avg_edge = np.nanmean([exp_b[fn][tk]['edge_2x'] for tk in tks])
    avg_sigs = np.nanmean([exp_b[fn][tk]['total_sigs'] for tk in tks])
    print(f"    ForceNorm=/{fn:.1f}  avg_edge_2x={avg_edge:>+.2f}%p  avg_sigs={avg_sigs:.0f}")

# Per-ticker edge (2x)
print(f"\n  {'':7s}", end="")
for fn in FORCE_NORM_VALUES:
    marker = " *" if fn == DEFAULT_FORCE_NORM else ""
    print(f" {'/' + f'{fn:.1f}':>5s}{marker:2s}", end="")
print()
print(f"  {'=' * (8 + 8 * len(FORCE_NORM_VALUES))}")

for tk in tks:
    print(f"  {tk:<7s}", end="")
    best_fn = max(FORCE_NORM_VALUES, key=lambda f: exp_b[f][tk]['eff_2x'])
    for fn in FORCE_NORM_VALUES:
        e = exp_b[fn][tk]['edge_2x']
        mark = " <" if fn == best_fn else "  "
        print(f" {e:>+5.1f}%{mark}", end="")
    print()

print(f"  {'-' * (8 + 8 * len(FORCE_NORM_VALUES))}")
print(f"  {'AVG':<7s}", end="")
for fn in FORCE_NORM_VALUES:
    avg_e = np.nanmean([exp_b[fn][tk]['edge_2x'] for tk in tks])
    print(f" {avg_e:>+5.2f}%p", end="")
print()

# Summary table for Exp B
print(f"\n  Experiment B Summary (2x leverage):")
print(f"  {'FNorm':>8s} {'Edge2x':>8s} {'Edge3x':>8s} {'Eff2x':>8s} {'Eff3x':>8s}"
      f" {'Hit90':>7s} {'Sigs/yr':>8s}")
print(f"  {'-' * 58}")

for fn in FORCE_NORM_VALUES:
    avg_e2 = np.nanmean([exp_b[fn][tk]['edge_2x'] for tk in tks])
    avg_e3 = np.nanmean([exp_b[fn][tk]['edge_3x'] for tk in tks])
    avg_ef2 = np.nanmean([exp_b[fn][tk]['eff_2x'] for tk in tks])
    avg_ef3 = np.nanmean([exp_b[fn][tk]['eff_3x'] for tk in tks])
    avg_hr = np.nanmean([exp_b[fn][tk]['hit_rate_90'] for tk in tks])
    avg_sy = np.nanmean([exp_b[fn][tk]['total_sigs'] / max(exp_b[fn][tk]['n_years'], 1) for tk in tks])
    marker = " ◀ current" if fn == DEFAULT_FORCE_NORM else ""
    print(f"  /{fn:>6.1f} {avg_e2:>+7.2f}% {avg_e3:>+7.2f}% {avg_ef2:>+7.3f} {avg_ef3:>+7.3f}"
          f" {avg_hr:>6.1f}% {avg_sy:>7.1f}{marker}")


# ═══════════════════════════════════════════════════════════
# EXPERIMENT C: Top Combination Cross-Validation
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 120}")
print("  EXPERIMENT C: Top Combinations Cross-Validation")
print("=" * 120)

# A에서 상위 3개 div_clip 선택 (eff_2x 기준)
a_ranking = sorted(DIV_CLIP_VALUES,
                   key=lambda dc: np.nanmean([exp_a[dc][tk]['eff_2x'] for tk in tks]),
                   reverse=True)
top_dc = a_ranking[:3]

# B에서 상위 3개 force_norm 선택
b_ranking = sorted(FORCE_NORM_VALUES,
                   key=lambda fn: np.nanmean([exp_b[fn][tk]['eff_2x'] for tk in tks]),
                   reverse=True)
top_fn = b_ranking[:3]

print(f"  Top DivClip from A:  {[f'/{d:.1f}' for d in top_dc]}")
print(f"  Top ForceNorm from B: {[f'/{f:.1f}' for f in top_fn]}")
print(f"  Testing {len(top_dc) * len(top_fn)} combinations...\n")

exp_c = {}  # {(dc, fn): {ticker: result}}
for dc in top_dc:
    for fn in top_fn:
        key = (dc, fn)
        # 이미 A 또는 B에서 계산한 조합은 재사용
        if fn == DEFAULT_FORCE_NORM and dc in exp_a:
            exp_c[key] = exp_a[dc]
        elif dc == DEFAULT_DIV_CLIP and fn in exp_b:
            exp_c[key] = exp_b[fn]
        else:
            exp_c[key] = run_sweep(tks, ticker_data, div_clip=dc, force_norm=fn)

        avg_e2 = np.nanmean([exp_c[key][tk]['edge_2x'] for tk in tks])
        avg_ef2 = np.nanmean([exp_c[key][tk]['eff_2x'] for tk in tks])
        avg_hr = np.nanmean([exp_c[key][tk]['hit_rate_90'] for tk in tks])
        print(f"    DC=/{dc:.1f} FN=/{fn:.1f}  edge_2x={avg_e2:>+.2f}%p  eff_2x={avg_ef2:>+.3f}  hit90={avg_hr:.1f}%")

# Exp C 결과 매트릭스
print(f"\n  Cross-Validation Matrix: Edge 2x (%p)")
print(f"  {'':>12s}", end="")
for fn in top_fn:
    print(f" {'FN/'+f'{fn:.1f}':>9s}", end="")
print()
print(f"  {'-' * (12 + 10 * len(top_fn))}")

for dc in top_dc:
    print(f"  DC/{dc:<6.1f}  ", end="")
    for fn in top_fn:
        key = (dc, fn)
        avg_e = np.nanmean([exp_c[key][tk]['edge_2x'] for tk in tks])
        print(f" {avg_e:>+8.2f}%p", end="")
    print()

print(f"\n  Cross-Validation Matrix: Efficiency 2x")
print(f"  {'':>12s}", end="")
for fn in top_fn:
    print(f" {'FN/'+f'{fn:.1f}':>9s}", end="")
print()
print(f"  {'-' * (12 + 10 * len(top_fn))}")

for dc in top_dc:
    print(f"  DC/{dc:<6.1f}  ", end="")
    for fn in top_fn:
        key = (dc, fn)
        avg_ef = np.nanmean([exp_c[key][tk]['eff_2x'] for tk in tks])
        print(f" {avg_ef:>+8.3f} ", end="")
    print()

print(f"\n  Cross-Validation Matrix: Hit Rate 90d (%)")
print(f"  {'':>12s}", end="")
for fn in top_fn:
    print(f" {'FN/'+f'{fn:.1f}':>9s}", end="")
print()
print(f"  {'-' * (12 + 10 * len(top_fn))}")

for dc in top_dc:
    print(f"  DC/{dc:<6.1f}  ", end="")
    for fn in top_fn:
        key = (dc, fn)
        avg_hr = np.nanmean([exp_c[key][tk]['hit_rate_90'] for tk in tks])
        print(f" {avg_hr:>8.1f}% ", end="")
    print()


# ═══════════════════════════════════════════════════════════
# BEST COMBINATION: Per-Ticker Detail
# ═══════════════════════════════════════════════════════════
# 전체 eff_2x 기준 최적 조합
all_combos = list(exp_c.keys())
best_combo = max(all_combos,
                 key=lambda k: np.nanmean([exp_c[k][tk]['eff_2x'] for tk in tks]))
best_dc, best_fn = best_combo

# 현재 기본값 조합
baseline_key = (DEFAULT_DIV_CLIP, DEFAULT_FORCE_NORM)
if baseline_key not in exp_c:
    exp_c[baseline_key] = run_sweep(tks, ticker_data,
                                     div_clip=DEFAULT_DIV_CLIP,
                                     force_norm=DEFAULT_FORCE_NORM)

print(f"\n{'=' * 120}")
print(f"  BEST COMBINATION: DC=/{best_dc:.1f}  FN=/{best_fn:.1f}")
print(f"  vs CURRENT:       DC=/{DEFAULT_DIV_CLIP:.1f}  FN=/{DEFAULT_FORCE_NORM:.1f}")
print(f"{'=' * 120}")

print(f"\n  {'Ticker':<7s} {'Sect':<8s}"
      f" │ {'CUR e2x':>8s} {'CUR e3x':>8s} {'CUR ef2':>8s} {'CUR hr':>7s} {'CUR sig':>7s}"
      f" │ {'NEW e2x':>8s} {'NEW e3x':>8s} {'NEW ef2':>8s} {'NEW hr':>7s} {'NEW sig':>7s}"
      f" │ {'de2x':>7s} {'def2':>7s}")
print(f"  {'=' * 115}")

delta_edges = []; delta_effs = []

for tk in tks:
    cur = exp_c[baseline_key][tk]
    new = exp_c[best_combo][tk]
    de = new['edge_2x'] - cur['edge_2x']
    df_ = new['eff_2x'] - cur['eff_2x']
    delta_edges.append(de)
    delta_effs.append(df_)
    sect = TICKERS.get(tk, '')

    cur_sy = cur['total_sigs'] / max(cur['n_years'], 1)
    new_sy = new['total_sigs'] / max(new['n_years'], 1)

    print(f"  {tk:<7s} {sect:<8s}"
          f" │ {cur['edge_2x']:>+7.2f}% {cur['edge_3x']:>+7.2f}% {cur['eff_2x']:>+7.3f} {cur['hit_rate_90']:>6.1f}% {cur_sy:>6.1f}"
          f" │ {new['edge_2x']:>+7.2f}% {new['edge_3x']:>+7.2f}% {new['eff_2x']:>+7.3f} {new['hit_rate_90']:>6.1f}% {new_sy:>6.1f}"
          f" │ {de:>+6.2f} {df_:>+6.3f}")

print(f"  {'-' * 115}")

# Average
cur_avg_e2 = np.nanmean([exp_c[baseline_key][tk]['edge_2x'] for tk in tks])
cur_avg_e3 = np.nanmean([exp_c[baseline_key][tk]['edge_3x'] for tk in tks])
cur_avg_ef = np.nanmean([exp_c[baseline_key][tk]['eff_2x'] for tk in tks])
cur_avg_hr = np.nanmean([exp_c[baseline_key][tk]['hit_rate_90'] for tk in tks])
new_avg_e2 = np.nanmean([exp_c[best_combo][tk]['edge_2x'] for tk in tks])
new_avg_e3 = np.nanmean([exp_c[best_combo][tk]['edge_3x'] for tk in tks])
new_avg_ef = np.nanmean([exp_c[best_combo][tk]['eff_2x'] for tk in tks])
new_avg_hr = np.nanmean([exp_c[best_combo][tk]['hit_rate_90'] for tk in tks])

print(f"  {'AVG':<7s} {'':8s}"
      f" │ {cur_avg_e2:>+7.2f}% {cur_avg_e3:>+7.2f}% {cur_avg_ef:>+7.3f} {cur_avg_hr:>6.1f}% {'':>7s}"
      f" │ {new_avg_e2:>+7.2f}% {new_avg_e3:>+7.2f}% {new_avg_ef:>+7.3f} {new_avg_hr:>6.1f}% {'':>7s}"
      f" │ {np.nanmean(delta_edges):>+6.2f} {np.nanmean(delta_effs):>+6.3f}")

# Win/Loss
new_wins = sum(1 for d in delta_effs if d > 0.005)
new_loses = sum(1 for d in delta_effs if d < -0.005)
ties = len(tks) - new_wins - new_loses
print(f"\n  New wins: {new_wins}/{len(tks)}  |  Current wins: {new_loses}/{len(tks)}  |  Ties: {ties}/{len(tks)}")


# ═══════════════════════════════════════════════════════════
# QQQ / VOO 상세 (Best Combo)
# ═══════════════════════════════════════════════════════════
for target in ['QQQ', 'VOO']:
    if target not in tks:
        continue

    print(f"\n{'=' * 120}")
    print(f"  {target} Year-by-Year: CURRENT (DC/{DEFAULT_DIV_CLIP} FN/{DEFAULT_FORCE_NORM})"
          f" vs BEST (DC/{best_dc} FN/{best_fn})")
    print(f"{'=' * 120}")

    cur_yr = exp_c[baseline_key][target]['yr_results']
    new_yr = exp_c[best_combo][target]['yr_results']

    print(f"\n  {'Year':>6s} │ {'CUR 2x':>8s} {'CUR 3x':>8s} {'CUR DCA':>8s} {'Ce2x':>7s}"
          f" │ {'NEW 2x':>8s} {'NEW 3x':>8s} {'NEW DCA':>8s} {'Ne2x':>7s}"
          f" │ {'d_e2x':>7s} {'Sigs':>5s}")
    print(f"  {'-' * 100}")

    for cr, nr in zip(cur_yr, new_yr):
        de = nr['edge_2x'] - cr['edge_2x']
        print(f"  {cr['yr']:>6d} │ {cr['ret_2x']:>+7.1f}% {cr['ret_3x']:>+7.1f}% {cr['ret_dca']:>+7.1f}% {cr['edge_2x']:>+6.1f}%"
              f" │ {nr['ret_2x']:>+7.1f}% {nr['ret_3x']:>+7.1f}% {nr['ret_dca']:>+7.1f}% {nr['edge_2x']:>+6.1f}%"
              f" │ {de:>+6.1f}% {nr['sigs']:>4d}")

    print(f"  {'-' * 100}")
    ca2 = np.nanmean([r['edge_2x'] for r in cur_yr])
    na2 = np.nanmean([r['edge_2x'] for r in new_yr])
    print(f"  {'AVG':>6s} │ {'':>8s} {'':>8s} {'':>8s} {ca2:>+6.1f}%"
          f" │ {'':>8s} {'':>8s} {'':>8s} {na2:>+6.1f}%"
          f" │ {na2-ca2:>+6.1f}%")


# ═══════════════════════════════════════════════════════════
# GRAND SUMMARY
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 120}")
print(f"  GRAND SUMMARY")
print(f"{'=' * 120}")

# Rank all tested combos
all_tested = {}

# From Exp A (all dc × default fn)
for dc in DIV_CLIP_VALUES:
    key = (dc, DEFAULT_FORCE_NORM)
    if key not in all_tested:
        avg_ef = np.nanmean([exp_a[dc][tk]['eff_2x'] for tk in tks])
        avg_e2 = np.nanmean([exp_a[dc][tk]['edge_2x'] for tk in tks])
        avg_e3 = np.nanmean([exp_a[dc][tk]['edge_3x'] for tk in tks])
        avg_hr = np.nanmean([exp_a[dc][tk]['hit_rate_90'] for tk in tks])
        avg_sg = np.nanmean([exp_a[dc][tk]['total_sigs'] / max(exp_a[dc][tk]['n_years'], 1) for tk in tks])
        all_tested[key] = {'eff_2x': avg_ef, 'edge_2x': avg_e2, 'edge_3x': avg_e3,
                           'hit90': avg_hr, 'sigs_yr': avg_sg}

# From Exp B (default dc × all fn)
for fn in FORCE_NORM_VALUES:
    key = (DEFAULT_DIV_CLIP, fn)
    if key not in all_tested:
        avg_ef = np.nanmean([exp_b[fn][tk]['eff_2x'] for tk in tks])
        avg_e2 = np.nanmean([exp_b[fn][tk]['edge_2x'] for tk in tks])
        avg_e3 = np.nanmean([exp_b[fn][tk]['edge_3x'] for tk in tks])
        avg_hr = np.nanmean([exp_b[fn][tk]['hit_rate_90'] for tk in tks])
        avg_sg = np.nanmean([exp_b[fn][tk]['total_sigs'] / max(exp_b[fn][tk]['n_years'], 1) for tk in tks])
        all_tested[key] = {'eff_2x': avg_ef, 'edge_2x': avg_e2, 'edge_3x': avg_e3,
                           'hit90': avg_hr, 'sigs_yr': avg_sg}

# From Exp C (cross combos)
for key in exp_c:
    if key not in all_tested:
        avg_ef = np.nanmean([exp_c[key][tk]['eff_2x'] for tk in tks])
        avg_e2 = np.nanmean([exp_c[key][tk]['edge_2x'] for tk in tks])
        avg_e3 = np.nanmean([exp_c[key][tk]['edge_3x'] for tk in tks])
        avg_hr = np.nanmean([exp_c[key][tk]['hit_rate_90'] for tk in tks])
        avg_sg = np.nanmean([exp_c[key][tk]['total_sigs'] / max(exp_c[key][tk]['n_years'], 1) for tk in tks])
        all_tested[key] = {'eff_2x': avg_ef, 'edge_2x': avg_e2, 'edge_3x': avg_e3,
                           'hit90': avg_hr, 'sigs_yr': avg_sg}

# Sort by eff_2x
ranked = sorted(all_tested.items(), key=lambda x: x[1]['eff_2x'], reverse=True)

print(f"\n  {'Rank':>4s} {'DC':>6s} {'FN':>6s} │ {'Eff2x':>8s} {'Edge2x':>8s} {'Edge3x':>8s}"
      f" {'Hit90':>7s} {'Sig/yr':>7s} │ {'Note':>10s}")
print(f"  {'=' * 75}")

for i, (key, v) in enumerate(ranked):
    dc, fn = key
    note = ""
    if dc == DEFAULT_DIV_CLIP and fn == DEFAULT_FORCE_NORM:
        note = "◀ CURRENT"
    elif i == 0:
        note = "★ BEST"
    print(f"  {i+1:>4d} /{dc:>4.1f} /{fn:>4.1f} │ {v['eff_2x']:>+7.3f} {v['edge_2x']:>+7.2f}% {v['edge_3x']:>+7.2f}%"
          f" {v['hit90']:>6.1f}% {v['sigs_yr']:>6.1f} │ {note:>10s}")

# Final recommendation
best_key, best_val = ranked[0]
cur_val = all_tested.get((DEFAULT_DIV_CLIP, DEFAULT_FORCE_NORM), {})

print(f"""
  +------------------------------------------------------------------+
  |  CURRENT:  DC=/{DEFAULT_DIV_CLIP:.1f}  FN=/{DEFAULT_FORCE_NORM:.1f}                               |
  |    Edge 2x:  {cur_val.get('edge_2x', 0):>+7.2f}%p    Eff: {cur_val.get('eff_2x', 0):>+7.3f}                  |
  |    Edge 3x:  {cur_val.get('edge_3x', 0):>+7.2f}%p    Hit90: {cur_val.get('hit90', 0):>5.1f}%                |
  +------------------------------------------------------------------+
  |  BEST:     DC=/{best_key[0]:.1f}  FN=/{best_key[1]:.1f}                               |
  |    Edge 2x:  {best_val['edge_2x']:>+7.2f}%p    Eff: {best_val['eff_2x']:>+7.3f}                  |
  |    Edge 3x:  {best_val['edge_3x']:>+7.2f}%p    Hit90: {best_val['hit90']:>5.1f}%                |
  +------------------------------------------------------------------+
  |  DELTA:                                                          |
  |    Eff:    {best_val['eff_2x'] - cur_val.get('eff_2x', 0):>+7.3f}                                        |
  |    Edge2x: {best_val['edge_2x'] - cur_val.get('edge_2x', 0):>+7.2f}%p                                     |
  |    Edge3x: {best_val['edge_3x'] - cur_val.get('edge_3x', 0):>+7.2f}%p                                     |
  +------------------------------------------------------------------+
""")

print("=" * 120)
print("  Done.")
