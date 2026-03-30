"""
VN60+GEO-OP 프로덕션 코드 기준 전체 백테스트
=============================================
현재 코드(AND-GEO + 완화 파이프라인) 기준 상세 분석

전략: 월 $500 입금 → 시그널 시 50% 레버리지(2x/3x) 매수 → 월말 잔여 1x 매수
비교: DCA (월 $500 전액 1x 매수)
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
    calc_v4_score, calc_v4_subindicators,
    detect_signal_events, build_price_filter, smooth_earnings_volume,
)

# ═══════════════════════════════════════════════════════════
# Configuration — 현재 프로덕션 파라미터 (GEO-OP)
# ═══════════════════════════════════════════════════════════
TICKERS = {
    'TSLA': 'Tech', 'NVDA': 'Tech', 'AVGO': 'Tech',
    'AMZN': 'Tech', 'GOOGL': 'Tech', 'PLTR': 'Tech',
    'COIN': 'Fintech', 'HOOD': 'Fintech',
    'HIMS': 'Growth', 'RKLB': 'Growth', 'JOBY': 'Growth',
    'IONQ': 'Quantum',
    'QQQ': 'Benchmark', 'VOO': 'Benchmark',
}

# GEO-OP pipeline params (from watchlist.json)
SIGNAL_TH = 0.05
COOLDOWN = 5
ER_Q = 80
ATR_Q = 40
LOOKBACK = 252
CONFIRM_DAYS = 1
BUY_DD_LB = 20
BUY_DD_TH = 0.03
DIVGATE = 3

MONTHLY_DEPOSIT = 500.0
SIGNAL_BUY_PCT = 0.50
EXPENSE_2X = 0.0095 / 252
EXPENSE_3X = 0.0100 / 252


# ═══════════════════════════════════════════════════════════
# Data & Score
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


def build_synthetic_lev(close, leverage, expense_daily):
    daily_ret = np.diff(close) / close[:-1]
    lev = np.zeros(len(close))
    lev[0] = close[0]
    for i in range(1, len(close)):
        lr = leverage * daily_ret[i - 1] - expense_daily
        lev[i] = lev[i - 1] * (1 + lr)
        if lev[i] < 0.001:
            lev[i] = 0.001
    return lev


def get_buy_signals(df, score, tk):
    """프로덕션 파이프라인: Score→Event→PF→DD_GATE→Confirm"""
    events = detect_signal_events(score, th=SIGNAL_TH, cooldown=COOLDOWN)
    pf = build_price_filter(df, er_q=ER_Q, atr_q=ATR_Q, lookback=LOOKBACK)

    close = df['Close'].values
    rolling_high = pd.Series(close).rolling(BUY_DD_LB, min_periods=1).max().values
    n = len(df)
    buy_indices = []

    for ev in events:
        if ev['type'] != 'bottom':
            continue
        if not pf(ev['peak_idx']):
            continue
        dur = ev['end_idx'] - ev['start_idx'] + 1
        ci = ev['start_idx'] + CONFIRM_DAYS - 1
        if ci > ev['end_idx'] or dur < CONFIRM_DAYS or ci >= n:
            continue
        pidx = ev['peak_idx']
        rh = rolling_high[pidx]
        dd = (rh - close[pidx]) / rh if rh > 0 else 0
        if dd < BUY_DD_TH:
            continue
        buy_indices.append(ci)
    return buy_indices, events


# ═══════════════════════════════════════════════════════════
# Simulation
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

    # Signal quality
    hits_90 = 0; total_90 = 0; fwd_90 = []; fwd_30 = []; fwd_180 = []
    max_drawdowns = []
    for idx in buy_indices:
        if idx + 90 < n:
            total_90 += 1
            fr90 = (close[idx + 90] / close[idx] - 1) * 100
            fwd_90.append(fr90)
            if fr90 > 0: hits_90 += 1
        if idx + 30 < n:
            fwd_30.append((close[idx + 30] / close[idx] - 1) * 100)
        if idx + 180 < n:
            fwd_180.append((close[idx + 180] / close[idx] - 1) * 100)
        # Max drawdown after signal (within 90 days)
        end_i = min(idx + 90, n)
        if end_i > idx + 1:
            peak = close[idx]
            min_after = min(close[idx:end_i])
            max_drawdowns.append((min_after / peak - 1) * 100)

    avg_2x = np.nanmean([r['ret_2x'] for r in yr_results])
    avg_3x = np.nanmean([r['ret_3x'] for r in yr_results])
    avg_dca = np.nanmean([r['ret_dca'] for r in yr_results])

    bear_yrs = [r for r in yr_results if r['ret_dca'] < -5]
    bull_yrs = [r for r in yr_results if r['ret_dca'] > 15]
    flat_yrs = [r for r in yr_results if -5 <= r['ret_dca'] <= 15]

    return {
        'yr_results': yr_results,
        'avg_2x': avg_2x, 'avg_3x': avg_3x, 'avg_dca': avg_dca,
        'edge_2x': avg_2x - avg_dca, 'edge_3x': avg_3x - avg_dca,
        'total_sigs': sum(r['sigs'] for r in yr_results),
        'hit_rate_90': (hits_90 / total_90 * 100) if total_90 > 0 else 0,
        'avg_fwd_30': np.mean(fwd_30) if fwd_30 else 0,
        'avg_fwd_90': np.mean(fwd_90) if fwd_90 else 0,
        'avg_fwd_180': np.mean(fwd_180) if fwd_180 else 0,
        'median_fwd_90': np.median(fwd_90) if fwd_90 else 0,
        'avg_max_dd': np.mean(max_drawdowns) if max_drawdowns else 0,
        'n_years': len(yr_results),
        'final_a': pf_a(n - 1), 'final_b': pf_b(n - 1), 'final_c': pf_c(n - 1),
        'total_dep': total_dep,
        'bear_edge': np.mean([r['edge_2x'] for r in bear_yrs]) if bear_yrs else 0,
        'bull_edge': np.mean([r['edge_2x'] for r in bull_yrs]) if bull_yrs else 0,
        'flat_edge': np.mean([r['edge_2x'] for r in flat_yrs]) if flat_yrs else 0,
        'fwd_90_list': fwd_90,
        'fwd_30_list': fwd_30,
        'buy_indices': buy_indices,
    }


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    sep = '=' * 130
    print(sep)
    print("  VN60+GEO-OP 프로덕션 코드 기준 전체 백테스트")
    print("  Score: AND-GEO (sqrt(S_Force*S_Div) when both>0)")
    print(f"  Pipeline: th={SIGNAL_TH}, PF(ER<{ER_Q}%/ATR>{ATR_Q}%), DD>={BUY_DD_TH*100:.0f}%, confirm={CONFIRM_DAYS}d")
    print(f"  전략: 월 ${MONTHLY_DEPOSIT:.0f} → 시그널 시 {SIGNAL_BUY_PCT*100:.0f}% 레버리지(2x/3x) 매수 → 월말 잔여 1x")
    print(sep)

    # Load data
    print("\n  데이터 로딩...")
    data = {}
    for tk in TICKERS:
        df = download_max(tk)
        if df is None or len(df) < 300:
            print(f"    {tk}: SKIP (insufficient data)")
            continue
        df = smooth_earnings_volume(df, ticker=tk)
        n_years = len(df) / 252
        data[tk] = df
        print(f"    {tk}: {len(df)} bars, {n_years:.1f}yr")
    print(f"  {len(data)} tickers loaded.\n")

    # Run backtest per ticker
    all_results = {}
    all_signal_details = {}
    print("  백테스트 실행...")

    for tk, df in data.items():
        score = calc_v4_score(df, w=20, divgate_days=DIVGATE)
        subind = calc_v4_subindicators(df, w=20, divgate_days=DIVGATE)
        buy_idx, all_events = get_buy_signals(df, score, tk)

        close = df['Close'].values
        close_2x = build_synthetic_lev(close, 2, EXPENSE_2X)
        close_3x = build_synthetic_lev(close, 3, EXPENSE_3X)

        res = simulate(close, close_2x, close_3x, buy_idx, df.index)
        all_results[tk] = res

        # Collect signal details
        sig_details = []
        for idx in buy_idx:
            sig_details.append({
                'date': df.index[idx].strftime('%Y-%m-%d'),
                'price': close[idx],
                's_force': subind['s_force'].iloc[idx],
                's_div': subind['s_div'].iloc[idx],
                'score': score.iloc[idx],
                'fwd_30': (close[min(idx+30, len(close)-1)] / close[idx] - 1) * 100,
                'fwd_90': (close[min(idx+90, len(close)-1)] / close[idx] - 1) * 100 if idx+90 < len(close) else None,
            })
        all_signal_details[tk] = sig_details

        n_yr = res['n_years']
        sig_yr = res['total_sigs'] / n_yr if n_yr > 0 else 0
        print(f"    {tk:6s}  2x={res['avg_2x']:+.1f}%  3x={res['avg_3x']:+.1f}%  DCA={res['avg_dca']:+.1f}%  "
              f"sig={res['total_sigs']}({sig_yr:.1f}/yr)  hit90={res['hit_rate_90']:.0f}%")

    # ═══════════════════════════════════════════════════════
    # [1] 티커별 연평균 수익률 랭킹
    # ═══════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  [1] 티커별 연평균 수익률 (CAGR-like, 연도별 평균)")
    print(sep)
    print(f"\n  {'Ticker':<8} {'Sect':<10} │ {'2x':>8} {'3x':>8} {'DCA':>8} │ {'Edge2x':>7} {'Edge3x':>7} │ "
          f"{'Hit90':>6} {'Fwd30':>7} {'Fwd90':>7} {'Fwd180':>7} │ {'Sig':>4} {'S/yr':>5} {'Years':>5}")
    print("  " + "-" * 120)

    sorted_tks = sorted(all_results.keys(), key=lambda t: all_results[t]['avg_2x'], reverse=True)
    for tk in sorted_tks:
        r = all_results[tk]
        sec = TICKERS.get(tk, '?')
        n_yr = r['n_years']
        s_yr = r['total_sigs'] / n_yr if n_yr > 0 else 0
        print(f"  {tk:<8} {sec:<10} │ {r['avg_2x']:+7.1f}% {r['avg_3x']:+7.1f}% {r['avg_dca']:+7.1f}% │ "
              f"{r['edge_2x']:+6.1f}% {r['edge_3x']:+6.1f}% │ "
              f"{r['hit_rate_90']:5.0f}% {r['avg_fwd_30']:+6.1f}% {r['avg_fwd_90']:+6.1f}% {r['avg_fwd_180']:+6.1f}% │ "
              f"{r['total_sigs']:4d} {s_yr:5.1f} {n_yr:5.1f}")

    # Averages
    avg_2x = np.mean([all_results[t]['avg_2x'] for t in all_results])
    avg_3x = np.mean([all_results[t]['avg_3x'] for t in all_results])
    avg_dca = np.mean([all_results[t]['avg_dca'] for t in all_results])
    avg_hit = np.mean([all_results[t]['hit_rate_90'] for t in all_results])
    avg_f90 = np.mean([all_results[t]['avg_fwd_90'] for t in all_results])
    print("  " + "-" * 120)
    print(f"  {'AVG':<8} {'':10} │ {avg_2x:+7.1f}% {avg_3x:+7.1f}% {avg_dca:+7.1f}% │ "
          f"{avg_2x-avg_dca:+6.1f}% {avg_3x-avg_dca:+6.1f}% │ {avg_hit:5.0f}%")

    # ═══════════════════════════════════════════════════════
    # [2] 시그널 품질 상세
    # ═══════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  [2] 시그널 품질 상세 분석")
    print(sep)

    all_fwd90 = []
    all_fwd30 = []
    for tk in all_results:
        all_fwd90.extend(all_results[tk]['fwd_90_list'])
        all_fwd30.extend(all_results[tk]['fwd_30_list'])

    total_sigs = sum(all_results[t]['total_sigs'] for t in all_results)
    print(f"\n  전체 시그널 수: {total_sigs}")
    print(f"  평균 Hit Rate (90일): {avg_hit:.1f}%")
    if all_fwd90:
        print(f"  Fwd 90일 수익률: 평균 {np.mean(all_fwd90):+.1f}%, 중앙값 {np.median(all_fwd90):+.1f}%")
        print(f"  Fwd 90일 분포: min={min(all_fwd90):+.1f}% / 25%={np.percentile(all_fwd90,25):+.1f}% / "
              f"50%={np.percentile(all_fwd90,50):+.1f}% / 75%={np.percentile(all_fwd90,75):+.1f}% / max={max(all_fwd90):+.1f}%")
    if all_fwd30:
        hit30 = sum(1 for f in all_fwd30 if f > 0) / len(all_fwd30) * 100
        print(f"  Fwd 30일 Hit Rate: {hit30:.1f}% (평균 {np.mean(all_fwd30):+.1f}%)")

    # ═══════════════════════════════════════════════════════
    # [3] 최종 자산가치 비교
    # ═══════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  [3] 최종 자산가치 비교")
    print(sep)
    print(f"\n  {'Ticker':<8} {'입금':>12} │ {'GEO-OP 2x':>14} {'GEO-OP 3x':>14} {'DCA':>14} │ {'수익률2x':>10} {'수익률3x':>10} {'수익률DCA':>10}")
    print("  " + "-" * 110)

    total_dep_all = 0; total_2x = 0; total_3x = 0; total_dca = 0
    for tk in sorted_tks:
        r = all_results[tk]
        dep = r['total_dep']
        total_dep_all += dep; total_2x += r['final_a']; total_3x += r['final_b']; total_dca += r['final_c']
        roi_2x = (r['final_a'] / dep - 1) * 100
        roi_3x = (r['final_b'] / dep - 1) * 100
        roi_dca = (r['final_c'] / dep - 1) * 100
        print(f"  {tk:<8} ${dep:>10,.0f} │ ${r['final_a']:>12,.0f} ${r['final_b']:>12,.0f} ${r['final_c']:>12,.0f} │ "
              f"{roi_2x:>+9.0f}% {roi_3x:>+9.0f}% {roi_dca:>+9.0f}%")

    print("  " + "-" * 110)
    roi_t2x = (total_2x / total_dep_all - 1) * 100
    roi_t3x = (total_3x / total_dep_all - 1) * 100
    roi_tdca = (total_dca / total_dep_all - 1) * 100
    print(f"  {'TOTAL':<8} ${total_dep_all:>10,.0f} │ ${total_2x:>12,.0f} ${total_3x:>12,.0f} ${total_dca:>12,.0f} │ "
          f"{roi_t2x:>+9.0f}% {roi_t3x:>+9.0f}% {roi_tdca:>+9.0f}%")

    # ═══════════════════════════════════════════════════════
    # [4] QQQ & VOO 연도별 상세
    # ═══════════════════════════════════════════════════════
    for bench in ['QQQ', 'VOO']:
        if bench not in all_results:
            continue
        r = all_results[bench]
        print(f"\n{sep}")
        print(f"  [4] {bench} 연도별 상세")
        print(sep)
        print(f"\n  {'Year':>6} │ {'2x':>8} {'3x':>8} {'DCA':>8} │ {'Edge2x':>7} {'Edge3x':>7} │ {'Sig':>4} │ {'Note'}")
        print("  " + "-" * 75)

        for yr in r['yr_results']:
            note = ''
            if yr['ret_dca'] < -20: note = 'CRASH'
            elif yr['ret_dca'] < -5: note = 'BEAR'
            elif yr['ret_dca'] > 30: note = 'BOOM'
            elif yr['ret_dca'] > 15: note = 'BULL'

            marker = ''
            if yr['edge_2x'] > 5: marker = ' ★'
            elif yr['edge_2x'] < -3: marker = ' ▼'

            print(f"  {yr['yr']:>6} │ {yr['ret_2x']:+7.1f}% {yr['ret_3x']:+7.1f}% {yr['ret_dca']:+7.1f}% │ "
                  f"{yr['edge_2x']:+6.1f}% {yr['edge_3x']:+6.1f}% │ {yr['sigs']:4d} │ {note}{marker}")

        print("  " + "-" * 75)
        print(f"  {'AVG':>6} │ {r['avg_2x']:+7.1f}% {r['avg_3x']:+7.1f}% {r['avg_dca']:+7.1f}% │ "
              f"{r['edge_2x']:+6.1f}% {r['edge_3x']:+6.1f}% │")

    # ═══════════════════════════════════════════════════════
    # [5] 시장 환경별 성과
    # ═══════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  [5] 시장 환경별 Edge 분석 (2x 기준)")
    print(sep)

    bear_edges = []; bull_edges = []; flat_edges = []
    for tk in all_results:
        r = all_results[tk]
        bear_edges.append(r['bear_edge'])
        bull_edges.append(r['bull_edge'])
        flat_edges.append(r['flat_edge'])

    print(f"\n  BEAR 시장 (DCA<-5%):  평균 Edge = {np.mean(bear_edges):+.2f}%p")
    print(f"  FLAT 시장 (-5~15%):    평균 Edge = {np.mean(flat_edges):+.2f}%p")
    print(f"  BULL 시장 (DCA>15%):   평균 Edge = {np.mean(bull_edges):+.2f}%p")

    print(f"\n  {'Ticker':<8} │ {'Bear Edge':>10} {'Flat Edge':>10} {'Bull Edge':>10}")
    print("  " + "-" * 45)
    for tk in sorted_tks:
        r = all_results[tk]
        print(f"  {tk:<8} │ {r['bear_edge']:>+9.1f}% {r['flat_edge']:>+9.1f}% {r['bull_edge']:>+9.1f}%")

    # ═══════════════════════════════════════════════════════
    # [6] 시그널 타이밍 분석
    # ═══════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  [6] 시그널 타이밍 & 패턴 분석")
    print(sep)

    # Month distribution of signals
    all_months = []
    all_years_sig = []
    for tk in all_signal_details:
        for s in all_signal_details[tk]:
            dt = pd.Timestamp(s['date'])
            all_months.append(dt.month)
            all_years_sig.append(dt.year)

    if all_months:
        month_counts = pd.Series(all_months).value_counts().sort_index()
        print(f"\n  월별 시그널 분포:")
        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        for m in range(1, 13):
            cnt = month_counts.get(m, 0)
            bar = '#' * int(cnt)
            print(f"    {month_names[m-1]:3s}  {cnt:3d}  {bar}")

        # Year distribution
        year_counts = pd.Series(all_years_sig).value_counts().sort_index()
        print(f"\n  연도별 시그널 수:")
        for yr, cnt in year_counts.items():
            bar = '#' * int(cnt / 2)
            print(f"    {yr}  {cnt:3d}  {bar}")

    # Score distribution at signal time
    all_scores_at_sig = []
    all_force_at_sig = []
    all_div_at_sig = []
    for tk in all_signal_details:
        for s in all_signal_details[tk]:
            all_scores_at_sig.append(s['score'])
            all_force_at_sig.append(s['s_force'])
            all_div_at_sig.append(s['s_div'])

    if all_scores_at_sig:
        print(f"\n  시그널 시점 Score 분포:")
        print(f"    Score:  평균={np.mean(all_scores_at_sig):.3f}  중앙값={np.median(all_scores_at_sig):.3f}  "
              f"[{min(all_scores_at_sig):.3f} ~ {max(all_scores_at_sig):.3f}]")
        print(f"    Force:  평균={np.mean(all_force_at_sig):.3f}  중앙값={np.median(all_force_at_sig):.3f}")
        print(f"    Div:    평균={np.mean(all_div_at_sig):.3f}  중앙값={np.median(all_div_at_sig):.3f}")

    # ═══════════════════════════════════════════════════════
    # [7] 재미있는 발견들
    # ═══════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  [7] 흥미로운 발견")
    print(sep)

    # Finding 1: Best single signal
    best_sig = None; best_fwd = -999
    worst_sig = None; worst_fwd = 999
    for tk in all_signal_details:
        for s in all_signal_details[tk]:
            if s['fwd_90'] is not None:
                if s['fwd_90'] > best_fwd:
                    best_fwd = s['fwd_90']; best_sig = (tk, s)
                if s['fwd_90'] < worst_fwd:
                    worst_fwd = s['fwd_90']; worst_sig = (tk, s)

    if best_sig:
        tk, s = best_sig
        print(f"\n  [A] 역대 최고 시그널:")
        print(f"      {tk} {s['date']} @ ${s['price']:.2f}")
        print(f"      Score={s['score']:.3f} (Force={s['s_force']:.3f}, Div={s['s_div']:.3f})")
        print(f"      90일 후 수익률: {s['fwd_90']:+.1f}%")

    if worst_sig:
        tk, s = worst_sig
        print(f"\n  [B] 역대 최악 시그널:")
        print(f"      {tk} {s['date']} @ ${s['price']:.2f}")
        print(f"      Score={s['score']:.3f} (Force={s['s_force']:.3f}, Div={s['s_div']:.3f})")
        print(f"      90일 후 수익률: {s['fwd_90']:+.1f}%")

    # Finding 2: Highest ROI ticker
    best_roi_tk = max(all_results, key=lambda t: all_results[t]['final_a'] / all_results[t]['total_dep'])
    r = all_results[best_roi_tk]
    roi = (r['final_a'] / r['total_dep'] - 1) * 100
    print(f"\n  [C] 최고 총수익률 (2x):")
    print(f"      {best_roi_tk}: ${r['total_dep']:,.0f} 투입 → ${r['final_a']:,.0f} ({roi:+,.0f}%)")

    # Finding 3: Edge consistency
    consistent = []
    for tk in all_results:
        r = all_results[tk]
        positive_edge_yrs = sum(1 for yr in r['yr_results'] if yr['edge_2x'] > 0)
        total_yrs = len(r['yr_results'])
        if total_yrs > 0:
            consistent.append((tk, positive_edge_yrs / total_yrs * 100, positive_edge_yrs, total_yrs))

    print(f"\n  [D] Edge 일관성 (2x Edge>0인 연도 비율):")
    consistent.sort(key=lambda x: x[1], reverse=True)
    for tk, pct, pos, tot in consistent:
        bar = '█' * int(pct / 10) + '░' * (10 - int(pct / 10))
        print(f"      {tk:6s}  {bar}  {pct:.0f}% ({pos}/{tot}년)")

    # Finding 4: Consecutive wins
    print(f"\n  [E] 시그널 연속 성공 기록 (90일 수익>0):")
    for tk in sorted_tks[:5]:
        fwd_list = all_results[tk]['fwd_90_list']
        if not fwd_list:
            continue
        max_streak = 0; curr = 0
        for f in fwd_list:
            if f > 0:
                curr += 1; max_streak = max(max_streak, curr)
            else:
                curr = 0
        print(f"      {tk:6s}  최대 연속 성공: {max_streak}회 (전체 {len(fwd_list)} 시그널)")

    # Finding 5: 3x vs 2x divergence
    print(f"\n  [F] 3x 레버리지 초과수익 (3x Edge - 2x Edge):")
    lev_diff = []
    for tk in sorted_tks:
        r = all_results[tk]
        diff = r['edge_3x'] - r['edge_2x']
        lev_diff.append((tk, diff, r['edge_2x'], r['edge_3x']))
    lev_diff.sort(key=lambda x: x[1], reverse=True)
    for tk, diff, e2, e3 in lev_diff:
        arrow = '▲' if diff > 0 else '▽'
        print(f"      {tk:6s}  2x={e2:+5.1f}%  3x={e3:+5.1f}%  차이={diff:+5.1f}%p {arrow}")

    # Finding 6: Signal timing vs drawdown
    print(f"\n  [G] 시그널 후 최대 낙폭 (MDD within 90d):")
    for tk in sorted_tks:
        r = all_results[tk]
        if r['avg_max_dd'] != 0:
            mdd = r['avg_max_dd']
        else:
            mdd = 0
        # Recalculate from signal details
        mdds = []
        close = data[tk]['Close'].values
        for idx in r['buy_indices']:
            end_i = min(idx + 90, len(close))
            if end_i > idx + 1:
                min_p = min(close[idx:end_i])
                mdds.append((min_p / close[idx] - 1) * 100)
        if mdds:
            print(f"      {tk:6s}  평균 MDD={np.mean(mdds):+.1f}%  최악={min(mdds):+.1f}%  "
                  f"(시그널 {len(mdds)}회)")

    # ═══════════════════════════════════════════════════════
    # GRAND SUMMARY
    # ═══════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  GRAND SUMMARY: VN60+GEO-OP 프로덕션 코드")
    print(sep)

    # Win/loss count
    win_2x = sum(1 for t in all_results if all_results[t]['edge_2x'] > 0)
    win_3x = sum(1 for t in all_results if all_results[t]['edge_3x'] > 0)
    total_tk = len(all_results)

    print(f"""
  +{'─'*70}+
  │  전략: AND-GEO + 완화 파이프라인 (GEO-OP)                            │
  │  Score = sqrt(S_Force × S_Div) when both > 0, else 0               │
  │  Pipeline: th=0.05 / PF(ER<80%,ATR>40%) / DD>=3% / confirm=1d     │
  +{'─'*70}+
  │                                                                      │
  │  평균 연수익률:  2x = {avg_2x:+.1f}%   3x = {avg_3x:+.1f}%   DCA = {avg_dca:+.1f}%{'':>12}│
  │  평균 Edge:      2x = {avg_2x-avg_dca:+.1f}%p  3x = {avg_3x-avg_dca:+.1f}%p{'':>25}│
  │  DCA 초과 티커:  2x = {win_2x}/{total_tk}개     3x = {win_3x}/{total_tk}개{'':>25}│
  │                                                                      │
  │  시그널 품질:    Hit90 = {avg_hit:.0f}%   Fwd90 = {avg_f90:+.1f}%{'':>25}│
  │  총 자산:        2x = ${total_2x:>14,.0f}  (투입: ${total_dep_all:>10,.0f}){'':>7}│
  │                  3x = ${total_3x:>14,.0f}{'':>32}│
  │                  DCA = ${total_dca:>14,.0f}{'':>32}│
  +{'─'*70}+
""")

    print("  Done.\n")


if __name__ == '__main__':
    main()
