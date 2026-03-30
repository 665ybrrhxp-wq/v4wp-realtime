"""
VN60 Phase 4B: AND-GEO + 파이프라인 완화 Sweep
================================================
AND-GEO가 이미 보수적(둘 다 양수 필요)이므로
후속 파이프라인 필터를 개별/조합 완화해서 최적 조합 탐색

비교 대상:
  CUR      - 가중합 60/40 + 표준 파이프라인 (기준)
  GEO      - AND-GEO + 표준 파이프라인 (Phase 4 결과)
  GEO-T10  - threshold 0.15→0.10 (bot_th 0.075→0.05)
  GEO-T05  - threshold 0.15→0.05 (bot_th 0.075→0.025)
  GEO-PF   - Price Filter 완화 (ER<80%, ATR>40%)
  GEO-DD3  - BUY_DD_GATE 5%→3%
  GEO-C2   - Confirmation 3d→2d
  GEO-C1   - Confirmation 3d→1d
  GEO-LT   - 복합 완화: th=0.10 + DD3% + confirm=2d
  GEO-OP   - 최대 완화: th=0.05 + PF완화 + DD3% + confirm=1d
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
LOOKBACK = 252
V4_W = 20; DIVGATE = 3; DIV_CLIP = 3.0; FORCE_NORM = 2.0
MACD_FAST = 12; MACD_SLOW = 26; MACD_SIGNAL = 9

# 10가지 설정: score 방식 + 파이프라인 파라미터
CONFIGS = [
    {"key": "CUR",     "desc": "가중합+표준파이프(기준)", "score": "weighted",
     "th": 0.15, "er_q": 66, "atr_q": 55, "dd_th": 0.05, "dd_lb": 20, "confirm": 3, "cd": 5},
    {"key": "GEO",     "desc": "AND-GEO 표준파이프",    "score": "and_geo",
     "th": 0.15, "er_q": 66, "atr_q": 55, "dd_th": 0.05, "dd_lb": 20, "confirm": 3, "cd": 5},
    {"key": "GEO-T10", "desc": "GEO + th=0.10",       "score": "and_geo",
     "th": 0.10, "er_q": 66, "atr_q": 55, "dd_th": 0.05, "dd_lb": 20, "confirm": 3, "cd": 5},
    {"key": "GEO-T05", "desc": "GEO + th=0.05",       "score": "and_geo",
     "th": 0.05, "er_q": 66, "atr_q": 55, "dd_th": 0.05, "dd_lb": 20, "confirm": 3, "cd": 5},
    {"key": "GEO-PF",  "desc": "GEO + PF완화(80/40)",  "score": "and_geo",
     "th": 0.15, "er_q": 80, "atr_q": 40, "dd_th": 0.05, "dd_lb": 20, "confirm": 3, "cd": 5},
    {"key": "GEO-DD3", "desc": "GEO + DD>=3%",        "score": "and_geo",
     "th": 0.15, "er_q": 66, "atr_q": 55, "dd_th": 0.03, "dd_lb": 20, "confirm": 3, "cd": 5},
    {"key": "GEO-C2",  "desc": "GEO + confirm=2d",    "score": "and_geo",
     "th": 0.15, "er_q": 66, "atr_q": 55, "dd_th": 0.05, "dd_lb": 20, "confirm": 2, "cd": 5},
    {"key": "GEO-C1",  "desc": "GEO + confirm=1d",    "score": "and_geo",
     "th": 0.15, "er_q": 66, "atr_q": 55, "dd_th": 0.05, "dd_lb": 20, "confirm": 1, "cd": 5},
    {"key": "GEO-LT",  "desc": "GEO 복합: T10+DD3+C2", "score": "and_geo",
     "th": 0.10, "er_q": 66, "atr_q": 55, "dd_th": 0.03, "dd_lb": 20, "confirm": 2, "cd": 5},
    {"key": "GEO-OP",  "desc": "GEO 최대: T05+PF+DD3+C1","score": "and_geo",
     "th": 0.05, "er_q": 80, "atr_q": 40, "dd_th": 0.03, "dd_lb": 20, "confirm": 1, "cd": 5},
]
CURRENT_KEY = "CUR"
CONFIG_KEYS = [c["key"] for c in CONFIGS]


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


def precompute_div(pv_div, n, div_clip):
    raw_div = np.array([np.clip(pv_div.iloc[i] / div_clip, -1, 1) for i in range(n)])
    consec = np.zeros(n, dtype=int)
    for i in range(1, n):
        if raw_div[i] != 0 and np.sign(raw_div[i]) == np.sign(raw_div[i - 1]):
            consec[i] = consec[i - 1] + 1
        elif raw_div[i] != 0:
            consec[i] = 1
    return raw_div, consec


def calc_components(raw_div, consec, pv_fh_vel, n, w, divgate_days, force_norm):
    s_force = np.zeros(n)
    s_div = np.zeros(n)
    for i in range(max(60, w), n):
        s_div[i] = raw_div[i] if consec[i] >= divgate_days else 0.0
        fhr_std = pv_fh_vel.iloc[max(0, i - w):i].std() + 1e-10
        s_force[i] = np.clip(pv_fh_vel.iloc[i] / (force_norm * fhr_std), -1, 1)
    return s_force, s_div


def score_weighted(sf, sd):
    return 0.60 * sf + 0.40 * sd


def score_and_geo(sf, sd):
    both_pos = (sf > 0) & (sd > 0)
    return np.where(both_pos, np.sqrt(sf * sd), 0.0)


# ═══════════════════════════════════════════════════════════
# Parameterized Signal Detection
# ═══════════════════════════════════════════════════════════
pf_cache = {}  # (tk, er_q, atr_q) -> price_filter_func


def get_buy_signals_param(df_s, score_arr, tk, th, cd, er_q, atr_q, dd_lb, dd_th, confirm):
    """파이프라인 파라미터 전부 외부 지정"""
    score_series = pd.Series(score_arr, index=df_s.index)
    events = detect_signal_events(score_series, th=th, cooldown=cd)

    # Price Filter (캐시)
    pf_key = (tk, er_q, atr_q)
    if pf_key not in pf_cache:
        pf_cache[pf_key] = build_price_filter(df_s, er_q=er_q, atr_q=atr_q, lookback=LOOKBACK)
    pf = pf_cache[pf_key]

    close_vals = df_s['Close'].values
    rolling_high = pd.Series(close_vals).rolling(dd_lb, min_periods=1).max().values
    n = len(df_s)
    buy_indices = []

    for ev in events:
        if ev['type'] != 'bottom':
            continue
        if not pf(ev['peak_idx']):
            continue
        dur = ev['end_idx'] - ev['start_idx'] + 1
        ci = ev['start_idx'] + confirm - 1
        if ci > ev['end_idx'] or dur < confirm or ci >= n:
            continue
        pidx = ev['peak_idx']
        rh = rolling_high[pidx]
        dd = (rh - close_vals[pidx]) / rh if rh > 0 else 0
        if dd < dd_th:
            continue
        buy_indices.append(ci)
    return buy_indices


# ═══════════════════════════════════════════════════════════
# Simulation (동일)
# ═══════════════════════════════════════════════════════════
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

    bear_yrs = [r for r in yr_results if r['ret_dca'] < -5]
    bull_yrs = [r for r in yr_results if r['ret_dca'] > 15]

    return {
        'yr_results': yr_results,
        'avg_2x': avg_2x, 'avg_3x': avg_3x, 'avg_dca': avg_dca,
        'edge_2x': edge_2x, 'edge_3x': edge_3x,
        'worst_2x': worst_2x,
        'eff_2x': edge_2x / abs(worst_2x) if abs(worst_2x) > 0.1 else 0,
        'total_sigs': sum(r['sigs'] for r in yr_results),
        'hit_rate_90': (hits_90 / total_90 * 100) if total_90 > 0 else 0,
        'avg_fwd_90': np.mean(fwd_90) if fwd_90 else 0,
        'n_years': len(yr_results),
        'final_a': pf_a(n - 1), 'final_b': pf_b(n - 1), 'final_c': pf_c(n - 1),
        'total_dep': total_dep,
        'bear_edge_2x': np.mean([r['edge_2x'] for r in bear_yrs]) if bear_yrs else 0,
        'bull_edge_2x': np.mean([r['edge_2x'] for r in bull_yrs]) if bull_yrs else 0,
    }


# ═══════════════════════════════════════════════════════════
# Data Download
# ═══════════════════════════════════════════════════════════
print("=" * 140)
print("  VN60 PHASE 4B: AND-GEO + Pipeline Relaxation Sweep")
print("  AND-GEO가 보수적이므로 후속 파이프라인 완화 시 성과 변화 탐색")
print("  전략: 월 $500 → 시그널 시 50% 레버리지(2x/3x) 매수 → 월말 잔여금 1x 매수")
print("=" * 140)

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

    pv_div = calc_pv_divergence(df_s, V4_W)
    raw_div, consec = precompute_div(pv_div, n, DIV_CLIP)
    pv_fh_vel = calc_force_macd_vel(df_s, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    s_force, s_div = calc_components(raw_div, consec, pv_fh_vel, n,
                                      w=V4_W, divgate_days=DIVGATE, force_norm=FORCE_NORM)

    ticker_data[tk] = {
        'df_s': df_s, 'close': close, 'close_2x': close_2x, 'close_3x': close_3x,
        'dates': dates, 'n': n, 's_force': s_force, 's_div': s_div,
    }
    print(f"    {tk}: {n} bars, {(dates[-1]-dates[0]).days/365.25:.1f}yr")

tks = list(ticker_data.keys())
print(f"  {len(tks)} tickers loaded.\n")


# ═══════════════════════════════════════════════════════════
# Run All Configs
# ═══════════════════════════════════════════════════════════
print("  전체 설정 시뮬레이션 실행...")
master = {}

for cfg in CONFIGS:
    ck = cfg["key"]
    results = {}
    for tk in tks:
        td = ticker_data[tk]
        if cfg["score"] == "weighted":
            score_arr = score_weighted(td['s_force'], td['s_div'])
        else:
            score_arr = score_and_geo(td['s_force'], td['s_div'])

        buys = get_buy_signals_param(
            td['df_s'], score_arr, tk,
            th=cfg["th"], cd=cfg["cd"],
            er_q=cfg["er_q"], atr_q=cfg["atr_q"],
            dd_lb=cfg["dd_lb"], dd_th=cfg["dd_th"],
            confirm=cfg["confirm"],
        )
        results[tk] = simulate(td['close'], td['close_2x'], td['close_3x'], buys, td['dates'])
    master[ck] = results

    a2 = np.nanmean([results[tk]['avg_2x'] for tk in tks])
    a3 = np.nanmean([results[tk]['avg_3x'] for tk in tks])
    ad = np.nanmean([results[tk]['avg_dca'] for tk in tks])
    sy = np.nanmean([results[tk]['total_sigs'] / max(results[tk]['n_years'], 1) for tk in tks])
    hr = np.nanmean([results[tk]['hit_rate_90'] for tk in tks])
    print(f"    {ck:<8s}  2x={a2:>+6.2f}%  3x={a3:>+6.2f}%  DCA={ad:>+6.2f}%  sig/yr={sy:.1f}  hit90={hr:.1f}%")


# Rankings
ranking_2x = sorted(CONFIG_KEYS,
                     key=lambda k: np.nanmean([master[k][tk]['avg_2x'] for tk in tks]),
                     reverse=True)
ranking_3x = sorted(CONFIG_KEYS,
                     key=lambda k: np.nanmean([master[k][tk]['avg_3x'] for tk in tks]),
                     reverse=True)
best_2x = ranking_2x[0]
best_3x = ranking_3x[0]


# ═══════════════════════════════════════════════════════════
# [1] 전체 랭킹: 2x + 3x 동시 표시
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 150}")
print(f"  [1] 전체 설정 랭킹 (2x 기준, 3x 병행)")
print(f"{'=' * 150}")

print(f"\n  {'Rk':>2s} {'Config':<8s} {'Description':<24s} {'변경점':<16s}"
      f" │ {'2x':>8s} {'3x':>8s} {'DCA':>8s}"
      f" │ {'E2x':>6s} {'E3x':>6s} {'Eff2x':>6s}"
      f" │ {'Hit90':>6s} {'Fwd90':>7s} {'S/yr':>5s}"
      f" │ {'Note':>10s}")
print(f"  {'=' * 140}")

for i, ck in enumerate(ranking_2x):
    m = master[ck]
    cfg = next(c for c in CONFIGS if c["key"] == ck)
    a2 = np.nanmean([m[tk]['avg_2x'] for tk in tks])
    a3 = np.nanmean([m[tk]['avg_3x'] for tk in tks])
    ad = np.nanmean([m[tk]['avg_dca'] for tk in tks])
    e2 = np.nanmean([m[tk]['edge_2x'] for tk in tks])
    e3 = np.nanmean([m[tk]['edge_3x'] for tk in tks])
    ef = np.nanmean([m[tk]['eff_2x'] for tk in tks])
    hr = np.nanmean([m[tk]['hit_rate_90'] for tk in tks])
    fw = np.nanmean([m[tk]['avg_fwd_90'] for tk in tks])
    sy = np.nanmean([m[tk]['total_sigs'] / max(m[tk]['n_years'], 1) for tk in tks])

    # 변경점 요약
    changes = []
    if cfg["score"] == "and_geo": changes.append("GEO")
    if cfg["th"] != 0.15: changes.append(f"T{cfg['th']:.2f}")
    if cfg["er_q"] != 66: changes.append(f"PF↓")
    if cfg["dd_th"] != 0.05: changes.append(f"DD{int(cfg['dd_th']*100)}%")
    if cfg["confirm"] != 3: changes.append(f"C{cfg['confirm']}")
    chg_str = "+".join(changes) if changes else "기준"

    note = ""
    if ck == CURRENT_KEY: note = "◀ CURRENT"
    elif i == 0: note = "★ BEST"

    print(f"  {i+1:>2d} {ck:<8s} {cfg['desc']:<24s} {chg_str:<16s}"
          f" │ {a2:>+7.2f}% {a3:>+7.2f}% {ad:>+7.2f}%"
          f" │ {e2:>+5.2f} {e3:>+5.2f} {ef:>+5.3f}"
          f" │ {hr:>5.1f}% {fw:>+6.1f}% {sy:>4.1f}"
          f" │ {note:>10s}")


# ═══════════════════════════════════════════════════════════
# [2] 상세 메트릭 비교 (컴팩트)
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 150}")
print(f"  [2] 설정별 상세 메트릭 비교")
print(f"{'=' * 150}")

print(f"\n  {'Metric':<16s}", end="")
for ck in CONFIG_KEYS:
    tag = "◀" if ck == CURRENT_KEY else " "
    print(f" {ck+tag:>9s}", end="")
print()
print(f"  {'-' * (16 + 10 * len(CONFIG_KEYS))}")

for metric_name, metric_key in [("2x 연수익률", 'avg_2x'),
                                 ("3x 연수익률", 'avg_3x'),
                                 ("Edge 2x", 'edge_2x'),
                                 ("Edge 3x", 'edge_3x'),
                                 ("Efficiency", 'eff_2x'),
                                 ("Hit90", 'hit_rate_90'),
                                 ("Fwd90", 'avg_fwd_90'),
                                 ("Sig/yr", None)]:
    print(f"  {metric_name:<16s}", end="")
    for ck in CONFIG_KEYS:
        m = master[ck]
        if metric_key:
            val = np.nanmean([m[tk][metric_key] for tk in tks])
        else:
            val = np.nanmean([m[tk]['total_sigs'] / max(m[tk]['n_years'], 1) for tk in tks])
        if metric_key == 'eff_2x':
            print(f" {val:>+8.3f}", end="")
        elif metric_key is None:
            print(f" {val:>8.1f}", end="")
        else:
            print(f" {val:>+7.2f}%", end="")
    print()


# ═══════════════════════════════════════════════════════════
# [3] 파이프라인 파라미터 영향도 분석
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 150}")
print(f"  [3] 파이프라인 파라미터 영향도 (GEO 대비 변화)")
print(f"{'=' * 150}")

geo_ref = master["GEO"]
geo_a2 = np.nanmean([geo_ref[tk]['avg_2x'] for tk in tks])
geo_a3 = np.nanmean([geo_ref[tk]['avg_3x'] for tk in tks])
geo_e2 = np.nanmean([geo_ref[tk]['edge_2x'] for tk in tks])
geo_hr = np.nanmean([geo_ref[tk]['hit_rate_90'] for tk in tks])
geo_fw = np.nanmean([geo_ref[tk]['avg_fwd_90'] for tk in tks])
geo_sy = np.nanmean([geo_ref[tk]['total_sigs'] / max(geo_ref[tk]['n_years'], 1) for tk in tks])

print(f"\n  GEO 기준: 2x={geo_a2:>+.2f}%  3x={geo_a3:>+.2f}%  edge2x={geo_e2:>+.2f}%  hit90={geo_hr:.1f}%  sig/yr={geo_sy:.1f}\n")

print(f"  {'Config':<8s} {'변경 내용':<20s}"
      f" │ {'d2x':>7s} {'d3x':>7s} {'dEdge2x':>8s} {'dHit90':>7s} {'dFwd90':>8s} {'dSig/yr':>8s}"
      f" │ {'평가':>8s}")
print(f"  {'-' * 105}")

for ck in CONFIG_KEYS:
    if ck in ("CUR", "GEO"):
        continue
    m = master[ck]
    cfg = next(c for c in CONFIGS if c["key"] == ck)
    a2 = np.nanmean([m[tk]['avg_2x'] for tk in tks])
    a3 = np.nanmean([m[tk]['avg_3x'] for tk in tks])
    e2 = np.nanmean([m[tk]['edge_2x'] for tk in tks])
    hr = np.nanmean([m[tk]['hit_rate_90'] for tk in tks])
    fw = np.nanmean([m[tk]['avg_fwd_90'] for tk in tks])
    sy = np.nanmean([m[tk]['total_sigs'] / max(m[tk]['n_years'], 1) for tk in tks])

    d2 = a2 - geo_a2; d3 = a3 - geo_a3
    de = e2 - geo_e2; dh = hr - geo_hr
    df_ = fw - geo_fw; ds = sy - geo_sy

    # 평가: 2x 또는 3x가 CUR보다 높으면 유망
    cur_a2 = np.nanmean([master[CURRENT_KEY][tk]['avg_2x'] for tk in tks])
    cur_a3 = np.nanmean([master[CURRENT_KEY][tk]['avg_3x'] for tk in tks])
    if a2 > cur_a2 or a3 > cur_a3:
        verdict = "★유망"
    elif a2 > geo_a2:
        verdict = "↑개선"
    else:
        verdict = "─"

    changes = []
    if cfg["th"] != 0.15: changes.append(f"th={cfg['th']:.2f}")
    if cfg["er_q"] != 66: changes.append(f"PF({cfg['er_q']}/{cfg['atr_q']})")
    if cfg["dd_th"] != 0.05: changes.append(f"DD>={int(cfg['dd_th']*100)}%")
    if cfg["confirm"] != 3: changes.append(f"confirm={cfg['confirm']}d")
    chg_desc = ", ".join(changes)

    print(f"  {ck:<8s} {chg_desc:<20s}"
          f" │ {d2:>+6.2f}% {d3:>+6.2f}% {de:>+7.2f}% {dh:>+6.1f}% {df_:>+7.1f}% {ds:>+7.1f}"
          f" │ {verdict:>8s}")


# ═══════════════════════════════════════════════════════════
# [4] BEST vs CUR 티커별 비교
# ═══════════════════════════════════════════════════════════
# 2x/3x 각각의 best 찾기
print(f"\n{'=' * 150}")
print(f"  [4] 티커별 비교: CUR vs BEST-2x({best_2x}) vs BEST-3x({best_3x})")
print(f"{'=' * 150}")

print(f"\n  {'Ticker':<7s} {'Sect':<8s}"
      f" │ {'CUR 2x':>8s} {'CUR 3x':>8s}"
      f" │ {'B2x 2x':>8s} {'B2x 3x':>8s}"
      f" │ {'d2x':>6s} {'d3x':>6s}"
      f" │ {'CUR sig':>7s} {'B2x sig':>7s}")
print(f"  {'=' * 100}")

d2_all = []; d3_all = []
for tk in tks:
    c = master[CURRENT_KEY][tk]
    b2 = master[best_2x][tk]
    d2 = b2['avg_2x'] - c['avg_2x']; d3 = b2['avg_3x'] - c['avg_3x']
    d2_all.append(d2); d3_all.append(d3)
    sect = TICKERS.get(tk, '')
    print(f"  {tk:<7s} {sect:<8s}"
          f" │ {c['avg_2x']:>+7.1f}% {c['avg_3x']:>+7.1f}%"
          f" │ {b2['avg_2x']:>+7.1f}% {b2['avg_3x']:>+7.1f}%"
          f" │ {d2:>+5.2f} {d3:>+5.2f}"
          f" │ {c['total_sigs']:>6d} {b2['total_sigs']:>6d}")

print(f"  {'-' * 100}")
ca2 = np.nanmean([master[CURRENT_KEY][tk]['avg_2x'] for tk in tks])
ca3 = np.nanmean([master[CURRENT_KEY][tk]['avg_3x'] for tk in tks])
ba2 = np.nanmean([master[best_2x][tk]['avg_2x'] for tk in tks])
ba3 = np.nanmean([master[best_2x][tk]['avg_3x'] for tk in tks])
print(f"  {'AVG':<7s} {'':8s}"
      f" │ {ca2:>+7.1f}% {ca3:>+7.1f}%"
      f" │ {ba2:>+7.1f}% {ba3:>+7.1f}%"
      f" │ {np.mean(d2_all):>+5.2f} {np.mean(d3_all):>+5.2f}")

better_2x = sum(1 for d in d2_all if d > 0.05)
worse_2x = sum(1 for d in d2_all if d < -0.05)
print(f"\n  2x 기준: {better_2x}개 개선 / {worse_2x}개 악화 / {len(tks)-better_2x-worse_2x}개 동일")


# ═══════════════════════════════════════════════════════════
# [5] 시그널 품질: CUR vs GEO vs BEST
# ═══════════════════════════════════════════════════════════
# 가장 유망한 GEO variant 찾기 (CUR 능가하는 것 중 hit rate 최고)
geo_variants = [ck for ck in CONFIG_KEYS if ck not in ("CUR",)]
best_geo = max(geo_variants,
               key=lambda k: np.nanmean([master[k][tk]['avg_2x'] for tk in tks]))

print(f"\n{'=' * 150}")
print(f"  [5] 시그널 품질: CUR vs GEO(표준) vs {best_geo}")
print(f"{'=' * 150}")

print(f"\n  {'Ticker':<7s}"
      f" │ {'CUR sig':>7s} {'hit':>5s} {'fwd90':>8s}"
      f" │ {'GEO sig':>7s} {'hit':>5s} {'fwd90':>8s}"
      f" │ {best_geo+' sig':>10s} {'hit':>5s} {'fwd90':>8s}")
print(f"  {'=' * 95}")

for tk in tks:
    c = master[CURRENT_KEY][tk]
    g = master["GEO"][tk]
    b = master[best_geo][tk]
    print(f"  {tk:<7s}"
          f" │ {c['total_sigs']:>6d} {c['hit_rate_90']:>4.0f}% {c['avg_fwd_90']:>+7.1f}%"
          f" │ {g['total_sigs']:>6d} {g['hit_rate_90']:>4.0f}% {g['avg_fwd_90']:>+7.1f}%"
          f" │ {b['total_sigs']:>9d} {b['hit_rate_90']:>4.0f}% {b['avg_fwd_90']:>+7.1f}%")

print(f"  {'-' * 95}")
for label, mk in [("CUR", CURRENT_KEY), ("GEO", "GEO"), (best_geo, best_geo)]:
    m = master[mk]
    hr = np.nanmean([m[tk]['hit_rate_90'] for tk in tks])
    fw = np.nanmean([m[tk]['avg_fwd_90'] for tk in tks])
    sy = np.nanmean([m[tk]['total_sigs'] / max(m[tk]['n_years'], 1) for tk in tks])
    print(f"  {label:<7s}: avg hit90={hr:.1f}%, avg fwd90={fw:>+.1f}%, sig/yr={sy:.1f}")


# ═══════════════════════════════════════════════════════════
# [6] 최종 자산 비교
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 150}")
print(f"  [6] 최종 자산가치: CUR vs {best_2x}")
print(f"{'=' * 150}")

print(f"\n  {'Ticker':<7s} {'입금':>9s}"
      f" │ {'CUR 2x':>11s} {'CUR 3x':>11s} {'DCA':>11s}"
      f" │ {'NEW 2x':>11s} {'NEW 3x':>11s}"
      f" │ {'d2x$':>10s} {'d3x$':>10s}")
print(f"  {'=' * 115}")

tc2=0; tc3=0; tcd=0; tn2=0; tn3=0; tdep=0
for tk in tks:
    c = master[CURRENT_KEY][tk]; n_ = master[best_2x][tk]
    dep = c['total_dep']
    d2 = n_['final_a'] - c['final_a']
    d3 = n_['final_b'] - c['final_b']
    tc2 += c['final_a']; tc3 += c['final_b']; tcd += c['final_c']
    tn2 += n_['final_a']; tn3 += n_['final_b']
    tdep += dep
    print(f"  {tk:<7s} ${dep:>8,.0f}"
          f" │ ${c['final_a']:>10,.0f} ${c['final_b']:>10,.0f} ${c['final_c']:>10,.0f}"
          f" │ ${n_['final_a']:>10,.0f} ${n_['final_b']:>10,.0f}"
          f" │ ${d2:>+9,.0f} ${d3:>+9,.0f}")

print(f"  {'-' * 115}")
print(f"  {'TOTAL':<7s} ${tdep:>8,.0f}"
      f" │ ${tc2:>10,.0f} ${tc3:>10,.0f} ${tcd:>10,.0f}"
      f" │ ${tn2:>10,.0f} ${tn3:>10,.0f}"
      f" │ ${tn2-tc2:>+9,.0f} ${tn3-tc3:>+9,.0f}")


# ═══════════════════════════════════════════════════════════
# [7] QQQ / VOO 연도별: CUR vs BEST
# ═══════════════════════════════════════════════════════════
for target in ['QQQ', 'VOO']:
    if target not in tks:
        continue
    print(f"\n{'=' * 150}")
    print(f"  [7] {target} 연도별: CUR vs {best_2x}")
    print(f"{'=' * 150}")

    cur = master[CURRENT_KEY][target]
    new = master[best_2x][target]

    print(f"\n  {'Year':>6s}"
          f" │ {'CUR 2x':>8s} {'CUR 3x':>8s} {'DCA':>8s} {'Ce2x':>6s}"
          f" │ {'NEW 2x':>8s} {'NEW 3x':>8s} {'Ne2x':>6s} {'Ne3x':>6s}"
          f" │ {'d2x':>5s} {'d3x':>5s}")
    print(f"  {'-' * 100}")

    for cr, nr in zip(cur['yr_results'], new['yr_results']):
        d2 = nr['ret_2x'] - cr['ret_2x']
        d3 = nr['ret_3x'] - cr['ret_3x']
        regime = ""
        if cr['ret_dca'] < -5: regime = " B"
        elif cr['ret_dca'] > 15: regime = " U"
        print(f"  {cr['yr']:>6d}"
              f" │ {cr['ret_2x']:>+7.1f}% {cr['ret_3x']:>+7.1f}% {cr['ret_dca']:>+7.1f}% {cr['edge_2x']:>+5.1f}%"
              f" │ {nr['ret_2x']:>+7.1f}% {nr['ret_3x']:>+7.1f}% {nr['edge_2x']:>+5.1f}% {nr['edge_3x']:>+5.1f}%"
              f" │ {d2:>+4.1f} {d3:>+4.1f}{regime}")

    print(f"  {'-' * 100}")
    ca_ = np.nanmean([r['ret_2x'] for r in cur['yr_results']])
    na_ = np.nanmean([r['ret_2x'] for r in new['yr_results']])
    ca3_ = np.nanmean([r['ret_3x'] for r in cur['yr_results']])
    na3_ = np.nanmean([r['ret_3x'] for r in new['yr_results']])
    print(f"  {'AVG':>6s} │ {ca_:>+7.1f}% {ca3_:>+7.1f}% │ {na_:>+7.1f}% {na3_:>+7.1f}% │ {na_-ca_:>+4.2f} {na3_-ca3_:>+4.2f}")
    print(f"\n  {target}: CUR 2x=${cur['final_a']:,.0f}  3x=${cur['final_b']:,.0f}"
          f"  →  {best_2x} 2x=${new['final_a']:,.0f}  3x=${new['final_b']:,.0f}"
          f"  Hit90: {cur['hit_rate_90']:.1f}% → {new['hit_rate_90']:.1f}%")


# ═══════════════════════════════════════════════════════════
# GRAND SUMMARY
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 150}")
print(f"  GRAND SUMMARY: Phase 4B — AND-GEO 파이프라인 완화 효과")
print(f"{'=' * 150}")

cur_m = master[CURRENT_KEY]
cur_2x = np.nanmean([cur_m[tk]['avg_2x'] for tk in tks])
cur_3x = np.nanmean([cur_m[tk]['avg_3x'] for tk in tks])
cur_dca = np.nanmean([cur_m[tk]['avg_dca'] for tk in tks])
cur_hr = np.nanmean([cur_m[tk]['hit_rate_90'] for tk in tks])
cur_sy = np.nanmean([cur_m[tk]['total_sigs'] / max(cur_m[tk]['n_years'], 1) for tk in tks])

print(f"\n  [A] 전체 설정 요약 (2x 순):")
for ck in ranking_2x:
    m = master[ck]
    a2 = np.nanmean([m[tk]['avg_2x'] for tk in tks])
    a3 = np.nanmean([m[tk]['avg_3x'] for tk in tks])
    e2 = np.nanmean([m[tk]['edge_2x'] for tk in tks])
    hr = np.nanmean([m[tk]['hit_rate_90'] for tk in tks])
    sy = np.nanmean([m[tk]['total_sigs'] / max(m[tk]['n_years'], 1) for tk in tks])
    tag = " ◀ CUR" if ck == CURRENT_KEY else ""
    beat = " ★" if a2 > cur_2x and ck != CURRENT_KEY else ""
    print(f"    {ck:<8s}  2x={a2:>+.2f}%  3x={a3:>+.2f}%  edge={e2:>+.2f}%  hit90={hr:.1f}%  sig/yr={sy:.1f}{tag}{beat}")

# CUR 능가하는 GEO variant 확인
beats_cur_2x = [ck for ck in CONFIG_KEYS if ck != CURRENT_KEY
                and np.nanmean([master[ck][tk]['avg_2x'] for tk in tks]) > cur_2x]
beats_cur_3x = [ck for ck in CONFIG_KEYS if ck != CURRENT_KEY
                and np.nanmean([master[ck][tk]['avg_3x'] for tk in tks]) > cur_3x]

b2x_m = master[best_2x]
b2x_2x = np.nanmean([b2x_m[tk]['avg_2x'] for tk in tks])
b2x_3x = np.nanmean([b2x_m[tk]['avg_3x'] for tk in tks])
b2x_hr = np.nanmean([b2x_m[tk]['hit_rate_90'] for tk in tks])

print(f"""
  [B] CUR 능가 여부:
    2x 기준: {len(beats_cur_2x)}개 능가 {beats_cur_2x if beats_cur_2x else '없음'}
    3x 기준: {len(beats_cur_3x)}개 능가 {beats_cur_3x if beats_cur_3x else '없음'}

  +--------------------------------------------------------------------------+
  |  Config            VN60+2x   VN60+3x   Edge2x  Hit90  Fwd90  Sig/yr    |
  +--------------------------------------------------------------------------+
  |  CUR (가중합)       {cur_2x:>+7.2f}%  {cur_3x:>+7.2f}%  {cur_2x-cur_dca:>+5.2f}%  {cur_hr:>4.1f}%         {cur_sy:.1f}    |
  +--------------------------------------------------------------------------+
  |  {best_2x:<8s}(best2x)  {b2x_2x:>+7.2f}%  {b2x_3x:>+7.2f}%  {b2x_2x-cur_dca:>+5.2f}%  {b2x_hr:>4.1f}%         {np.nanmean([b2x_m[tk]['total_sigs']/max(b2x_m[tk]['n_years'],1) for tk in tks]):.1f}    |
  +--------------------------------------------------------------------------+
  |  DELTA 2x          {b2x_2x-cur_2x:>+7.2f}%  {b2x_3x-cur_3x:>+7.2f}%                              |
  |  Ticker W/L:       {better_2x} improved / {worse_2x} degraded / {len(tks)-better_2x-worse_2x} same             |
  |  Total Assets:     2x: ${tn2-tc2:>+,.0f}  3x: ${tn3-tc3:>+,.0f}     |
  +--------------------------------------------------------------------------+
""")

print("=" * 150)
print("  Done.")
