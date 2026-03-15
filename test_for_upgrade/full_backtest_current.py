"""
V4_wP 현행 시스템 전체 백테스트 (DivGate_3d + Earnings Vol Filter)
================================================================
- smooth_earnings_volume: 실적발표일 거래량 스무딩
- calc_v4_score(divgate_days=3): DivGate_3d 적용
- Duration 기반 확인: 매수 3d→100%, 매도 3d→5%
- LATE_SELL_BLOCK: 20일 고점 대비 5% 이상 하락 시 매도 차단
- 전체 기간(max history) + Modified Dietz 연도별 수익률

출력: 종목별 × 연도별 수익률, Alpha, 종합 요약
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from real_market_backtest import (
    calc_v4_score, detect_signal_events, build_price_filter,
    smooth_earnings_volume,
)

# =====================================================
# PARAMETERS (watchlist.json 프로덕션 설정과 동일)
# =====================================================
TICKERS = {
    'TSLA': 'Tech', 'PLTR': 'Tech', 'NVDA': 'Tech', 'AVGO': 'Tech',
    'AMZN': 'Tech', 'GOOGL': 'Tech', 'JOBY': 'Growth', 'HIMS': 'Growth',
    'TEM': 'Growth', 'RKLB': 'Growth', 'PGY': 'Growth', 'COIN': 'Fintech',
    'HOOD': 'Fintech', 'IONQ': 'Quantum', 'PL': 'Space',
}
BENCHMARKS = ['QQQ', 'VOO']
ALL_TICKERS = list(TICKERS.keys()) + BENCHMARKS

INITIAL = 1000.0
MONTHLY = 100.0

# V4 signal params
V4_WINDOW = 20
SIGNAL_TH = 0.15
COOLDOWN = 5
ER_Q = 66
ATR_Q = 55
LOOKBACK = 252
DIVGATE_DAYS = 3
EARNINGS_VOL_FILTER = True

# Duration strategy params (현행 최종)
CONFIRM_DAYS = 3
BUY_CONFIRMED_PCT = 1.00
SELL_CONFIRM_DAYS = 3
SELL_CONFIRMED_PCT = 0.05
LATE_SELL_DROP_TH = 0.05


def download_max(ticker):
    """yfinance max history 다운로드"""
    t = yf.Ticker(ticker)
    df = t.history(period='max', auto_adjust=True)
    if df is None or len(df) == 0:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


# =====================================================
# DATA LOADING + SIGNAL DETECTION (현행 파이프라인)
# =====================================================
print("=" * 120)
print("  V4_wP 현행 시스템 전체 백테스트 (DivGate_3d + Earnings Vol Filter)")
print("=" * 120)
print(f"  DivGate: {DIVGATE_DAYS}d | Earnings Filter: {EARNINGS_VOL_FILTER}")
print(f"  매수: {CONFIRM_DAYS}d confirm → {BUY_CONFIRMED_PCT:.0%} | 매도: {SELL_CONFIRM_DAYS}d confirm → {SELL_CONFIRMED_PCT:.0%}")
print(f"  DCA: ${INITIAL:.0f} initial + ${MONTHLY:.0f}/month")
print()
print("  Loading data...", flush=True)

data = {}
load_errors = []

for tk in ALL_TICKERS:
    print(f"    {tk}...", end=' ', flush=True)
    try:
        df = download_max(tk)
        if df is None or len(df) < 200:
            load_errors.append((tk, 'insufficient data'))
            print("SKIP")
            continue

        # [현행 파이프라인] 실적발표일 거래량 스무딩
        df_smooth = smooth_earnings_volume(df, ticker=tk) if EARNINGS_VOL_FILTER else df

        # [현행 파이프라인] DivGate_3d 적용 V4 스코어
        score = calc_v4_score(df_smooth, w=V4_WINDOW, divgate_days=DIVGATE_DAYS)
        events = detect_signal_events(score, th=SIGNAL_TH, cooldown=COOLDOWN)
        pf = build_price_filter(df_smooth, er_q=ER_Q, atr_q=ATR_Q, lookback=LOOKBACK)

        close = df['Close'].values  # 원본 가격으로 시뮬레이션
        rolling_high = df['Close'].rolling(20, min_periods=1).max().values

        filtered = []
        blocked_sells = 0
        for ev in events:
            if not pf(ev['peak_idx']):
                continue
            # LATE_SELL_BLOCK
            if ev['type'] == 'top':
                pidx = ev['peak_idx']
                if pidx < len(close):
                    price = close[pidx]
                    rh = rolling_high[pidx]
                    if rh > 0 and (rh - price) / rh > LATE_SELL_DROP_TH:
                        blocked_sells += 1
                        continue

            ev['duration'] = ev['end_idx'] - ev['start_idx'] + 1
            filtered.append(ev)

        data[tk] = {
            'df': df, 'events': filtered, 'score': score,
            'blocked_sells': blocked_sells,
        }
        n_buy = sum(1 for e in filtered if e['type'] == 'bottom')
        n_sell = sum(1 for e in filtered if e['type'] == 'top')
        print(f"OK ({len(df)} bars, {len(df.index[0].strftime('%Y'))}~, {n_buy}B/{n_sell}S)")

    except Exception as e:
        load_errors.append((tk, str(e)))
        print(f"ERROR: {e}")

print(f"\n  {len(data)}/{len(ALL_TICKERS)} tickers loaded")
if load_errors:
    for tk, err in load_errors:
        print(f"    [WARN] {tk}: {err}")
print()


# =====================================================
# SIMULATION: 현행 Duration 기반 전략
# =====================================================
def simulate(events, df):
    """현행 시스템과 동일한 Duration 기반 시뮬레이션.

    매수: sidx + confirm_days - 1 시점에 zone 유효하면 100% 매수
    매도: sidx + sell_confirm_days - 1 시점에 zone 유효하면 5% 매도
    """
    close = df['Close'].values
    dates = df.index
    n = len(close)

    price0 = close[0]
    shares = INITIAL / price0
    cash = 0.0
    shares_bnh = INITIAL / price0
    cash_bnh = 0.0

    month_done = {dates[0].strftime('%Y-%m')}
    trade_log = []

    yearly_data = {}
    current_year = dates[0].year

    def init_year(yr, v4_val, bnh_val):
        yearly_data[yr] = {
            'start_v4': v4_val, 'start_bnh': bnh_val,
            'flows_v4': [], 'flows_bnh': [],
            'end_v4': 0, 'end_bnh': 0,
        }

    init_year(current_year, INITIAL, INITIAL)

    for i in range(1, n):
        price = close[i]
        yr = dates[i].year

        # Year transition
        if yr != current_year:
            prev_price = close[i - 1]
            yearly_data[current_year]['end_v4'] = shares * prev_price + cash
            yearly_data[current_year]['end_bnh'] = shares_bnh * prev_price + cash_bnh
            current_year = yr
            init_year(yr, shares * prev_price + cash, shares_bnh * prev_price + cash_bnh)

        # Monthly DCA
        ym = dates[i].strftime('%Y-%m')
        if ym not in month_done:
            month_done.add(ym)
            day_of_year = (dates[i] - pd.Timestamp(year=yr, month=1, day=1)).days
            total_days = (pd.Timestamp(year=yr, month=12, day=31) -
                         pd.Timestamp(year=yr, month=1, day=1)).days + 1
            weight = 1 - day_of_year / total_days

            cash += MONTHLY
            cash_bnh += MONTHLY
            shares_bnh += MONTHLY / price

            yearly_data[yr]['flows_v4'].append((weight, MONTHLY))
            yearly_data[yr]['flows_bnh'].append((weight, MONTHLY))

        # Trade on signals
        for ev in events:
            sidx = ev['start_idx']
            eidx = ev['end_idx']
            duration = ev.get('duration', eidx - sidx + 1)

            if ev['type'] == 'bottom':
                # 매수: confirm_days 시점에 zone 유효하면 실행
                confirm_idx = sidx + CONFIRM_DAYS - 1
                if i == confirm_idx and confirm_idx <= eidx and duration >= CONFIRM_DAYS and cash > 0:
                    amount = cash * BUY_CONFIRMED_PCT
                    new_shares = amount / price
                    shares += new_shares
                    cash -= amount
                    trade_log.append({
                        'date': dates[i].strftime('%Y-%m-%d'),
                        'type': 'BUY_CONFIRMED',
                        'price': price, 'amount': amount,
                        'duration': duration,
                    })

            elif ev['type'] == 'top':
                # 매도: sell_confirm_days 시점에 zone 유효하면 실행
                confirm_idx = sidx + SELL_CONFIRM_DAYS - 1
                if i == confirm_idx and confirm_idx <= eidx and duration >= SELL_CONFIRM_DAYS and shares > 0:
                    sell_shares = shares * SELL_CONFIRMED_PCT
                    proceeds = sell_shares * price
                    cash += proceeds
                    shares -= sell_shares
                    trade_log.append({
                        'date': dates[i].strftime('%Y-%m-%d'),
                        'type': 'SELL_CONFIRMED',
                        'price': price, 'amount': proceeds,
                        'duration': duration,
                    })

    # Close final year
    yearly_data[current_year]['end_v4'] = shares * close[-1] + cash
    yearly_data[current_year]['end_bnh'] = shares_bnh * close[-1] + cash_bnh

    # Modified Dietz returns
    year_results = {}
    for yr, yd in sorted(yearly_data.items()):
        sv4, ev4 = yd['start_v4'], yd['end_v4']
        sbnh, ebnh = yd['start_bnh'], yd['end_bnh']

        sum_wf_v4 = sum(w * f for w, f in yd['flows_v4'])
        sum_f_v4 = sum(f for _, f in yd['flows_v4'])
        denom_v4 = sv4 + sum_wf_v4
        ret_v4 = ((ev4 - sv4 - sum_f_v4) / denom_v4 * 100) if denom_v4 > 0 else 0

        sum_wf_bnh = sum(w * f for w, f in yd['flows_bnh'])
        sum_f_bnh = sum(f for _, f in yd['flows_bnh'])
        denom_bnh = sbnh + sum_wf_bnh
        ret_bnh = ((ebnh - sbnh - sum_f_bnh) / denom_bnh * 100) if denom_bnh > 0 else 0

        year_results[yr] = (ret_v4, ret_bnh, ret_v4 - ret_bnh)

    final_v4 = shares * close[-1] + cash
    final_bnh = shares_bnh * close[-1] + cash_bnh
    total_invested = INITIAL + MONTHLY * (len(month_done) - 1)

    return year_results, trade_log, {
        'final_v4': final_v4, 'final_bnh': final_bnh,
        'total_invested': total_invested,
        'shares': shares, 'cash': cash,
        'total_return_v4': (final_v4 / total_invested - 1) * 100,
        'total_return_bnh': (final_bnh / total_invested - 1) * 100,
    }


# =====================================================
# RUN ALL TICKERS
# =====================================================
all_yearly = {}
all_trades = {}
all_stats = {}

tickers_ordered = [tk for tk in ALL_TICKERS if tk in data]

for tk in tickers_ordered:
    yr_res, trades, stats = simulate(data[tk]['events'], data[tk]['df'])
    all_yearly[tk] = yr_res
    all_trades[tk] = trades
    all_stats[tk] = stats


# =====================================================
# REPORT 1: 종목별 V4 연수익률 테이블
# =====================================================
# 전체 연도 수집 (종목별 시작 연도가 다르므로)
all_years = sorted(set(yr for tk in tickers_ordered for yr in all_yearly[tk]))

# 데이터 기간이 충분한 연도만 (최소 200일 = 이미 필터됨)
print("=" * 140)
print("  [1] 종목별 V4 전략 연수익률 (%) - Modified Dietz")
print("=" * 140)

header = f"  {'Ticker':<7s} {'Sector':<8s} {'Start':>5s}"
# 최근 10년만 표시 (이전은 일부 종목만 있어서)
display_years = [y for y in all_years if y >= 2015]
if len(display_years) < len(all_years):
    header += " | <2015"
for yr in display_years:
    header += f" {yr:>7d}"
header += f" | {'AVG':>7s} {'W/L':>5s}"
print(header)
print("  " + "-" * (len(header) - 2))

overall_yearly = {yr: [] for yr in all_years}

for tk in tickers_ordered:
    sector = TICKERS.get(tk, 'Bench')
    tk_years = sorted(all_yearly[tk].keys())
    start_yr = tk_years[0] if tk_years else 0

    line = f"  {tk:<7s} {sector:<8s} {start_yr:>5d}"

    # 2015 이전 평균
    if len(display_years) < len(all_years):
        early = [all_yearly[tk][yr][0] for yr in tk_years if yr < 2015]
        if early:
            line += f" {np.mean(early):>+5.0f}%"
        else:
            line += f" {'':>6s}"

    alphas_all = []
    wins = 0
    for yr in display_years:
        if yr in all_yearly[tk]:
            v4, bnh, alpha = all_yearly[tk][yr]
            line += f" {v4:>+6.1f}%"
            alphas_all.append(v4)
            overall_yearly[yr].append(v4)
            if v4 > 0:
                wins += 1
        else:
            line += f" {'':>7s}"

    # 전체 기간 평균
    all_v4 = [all_yearly[tk][yr][0] for yr in tk_years]
    avg = np.mean(all_v4) if all_v4 else 0
    line += f" | {avg:>+6.1f}% {sum(1 for v in all_v4 if v > 0)}/{len(all_v4)}"
    print(line)

# 전체 평균 행
print("  " + "-" * (len(header) - 2))
line = f"  {'AVG':<7s} {'':8s} {'':>5s}"
if len(display_years) < len(all_years):
    early_all = [v for yr in all_years if yr < 2015 for v in overall_yearly.get(yr, [])]
    if early_all:
        line += f" {np.mean(early_all):>+5.0f}%"
    else:
        line += f" {'':>6s}"

avg_list = []
for yr in display_years:
    vals = overall_yearly[yr]
    if vals:
        m = np.mean(vals)
        line += f" {m:>+6.1f}%"
        avg_list.append(m)
    else:
        line += f" {'':>7s}"
line += f" | {np.mean(avg_list):>+6.1f}%" if avg_list else ""
print(line)
print()


# =====================================================
# REPORT 2: Alpha (V4 - BnH) 테이블
# =====================================================
print("=" * 140)
print("  [2] 종목별 Alpha (V4 - BnH) (%) - 양수=V4 우세")
print("=" * 140)

header2 = f"  {'Ticker':<7s} {'Sector':<8s} {'Start':>5s}"
if len(display_years) < len(all_years):
    header2 += " | <2015"
for yr in display_years:
    header2 += f" {yr:>7d}"
header2 += f" | {'AVG':>7s} {'W/L':>5s}"
print(header2)
print("  " + "-" * (len(header2) - 2))

alpha_yearly = {yr: [] for yr in all_years}

for tk in tickers_ordered:
    sector = TICKERS.get(tk, 'Bench')
    tk_years = sorted(all_yearly[tk].keys())
    start_yr = tk_years[0] if tk_years else 0

    line = f"  {tk:<7s} {sector:<8s} {start_yr:>5d}"

    if len(display_years) < len(all_years):
        early = [all_yearly[tk][yr][2] for yr in tk_years if yr < 2015]
        if early:
            line += f" {np.mean(early):>+5.0f}%"
        else:
            line += f" {'':>6s}"

    all_alphas = []
    wins = 0
    for yr in display_years:
        if yr in all_yearly[tk]:
            _, _, alpha = all_yearly[tk][yr]
            color_mark = ""
            line += f" {alpha:>+6.1f}%"
            all_alphas.append(alpha)
            alpha_yearly[yr].append(alpha)
            if alpha > 0:
                wins += 1
        else:
            line += f" {'':>7s}"

    total_alphas = [all_yearly[tk][yr][2] for yr in tk_years]
    avg = np.mean(total_alphas) if total_alphas else 0
    line += f" | {avg:>+6.1f}% {sum(1 for a in total_alphas if a > 0)}/{len(total_alphas)}"
    print(line)

print("  " + "-" * (len(header2) - 2))
line = f"  {'AVG':<7s} {'':8s} {'':>5s}"
if len(display_years) < len(all_years):
    early_alphas = [a for yr in all_years if yr < 2015 for a in alpha_yearly.get(yr, [])]
    if early_alphas:
        line += f" {np.mean(early_alphas):>+5.0f}%"
    else:
        line += f" {'':>6s}"

avg_alphas = []
for yr in display_years:
    vals = alpha_yearly[yr]
    if vals:
        m = np.mean(vals)
        line += f" {m:>+6.1f}%"
        avg_alphas.append(m)
    else:
        line += f" {'':>7s}"
line += f" | {np.mean(avg_alphas):>+6.1f}%" if avg_alphas else ""
print(line)
print()


# =====================================================
# REPORT 3: 종목별 상세 (최종 포트폴리오 + 매매 통계)
# =====================================================
print("=" * 120)
print("  [3] 종목별 최종 포트폴리오")
print("=" * 120)
print(f"  {'Ticker':<7s} {'Sector':<8s} {'Period':>12s} {'Invested':>10s} {'V4 Final':>10s} {'BnH Final':>10s}"
      f" {'V4 Ret':>8s} {'BnH Ret':>8s} {'Alpha':>8s} {'Buys':>5s} {'Sells':>5s}")
print("  " + "-" * 108)

total_inv = total_v4 = total_bnh = 0

for tk in tickers_ordered:
    st = all_stats[tk]
    sector = TICKERS.get(tk, 'Bench')
    df = data[tk]['df']
    period = f"{df.index[0].year}-{df.index[-1].year}"

    trades = all_trades[tk]
    n_buys = sum(1 for t in trades if 'BUY' in t['type'])
    n_sells = sum(1 for t in trades if 'SELL' in t['type'])

    total_inv += st['total_invested']
    total_v4 += st['final_v4']
    total_bnh += st['final_bnh']

    alpha_total = st['total_return_v4'] - st['total_return_bnh']

    print(f"  {tk:<7s} {sector:<8s} {period:>12s} ${st['total_invested']:>9,.0f} ${st['final_v4']:>9,.0f} ${st['final_bnh']:>9,.0f}"
          f" {st['total_return_v4']:>+7.1f}% {st['total_return_bnh']:>+7.1f}% {alpha_total:>+7.1f}%"
          f" {n_buys:>5d} {n_sells:>5d}")

print("  " + "-" * 108)
ret_v4_total = (total_v4 / total_inv - 1) * 100
ret_bnh_total = (total_bnh / total_inv - 1) * 100
print(f"  {'TOTAL':<7s} {'':8s} {'':>12s} ${total_inv:>9,.0f} ${total_v4:>9,.0f} ${total_bnh:>9,.0f}"
      f" {ret_v4_total:>+7.1f}% {ret_bnh_total:>+7.1f}% {ret_v4_total - ret_bnh_total:>+7.1f}%")
print()


# =====================================================
# REPORT 4: 연도별 전체 요약 (가중평균)
# =====================================================
print("=" * 120)
print("  [4] 연도별 전체 요약 (전종목 평균)")
print("=" * 120)
print(f"  {'Year':>6s} │ {'종목수':>5s} │ {'V4 수익률':>10s} │ {'BnH 수익률':>10s} │ {'Alpha':>8s} │ {'Win':>5s} │ {'시장환경'}")
print(f"  {'─'*6}─┼─{'─'*5}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*8}─┼─{'─'*5}─┼─{'─'*20}")

# 시장 환경 라벨
market_labels = {
    1997: '닷컴 초기',
    1998: '닷컴 호황', 1999: '닷컴 버블', 2000: '닷컴 붕괴',
    2001: '침체', 2002: '바닥', 2003: '회복', 2004: '성장',
    2005: '안정', 2006: '과열', 2007: '서브프라임', 2008: '금융위기',
    2009: '바닥반등', 2010: '회복', 2011: '유럽위기', 2012: '안정',
    2013: '강세', 2014: '안정', 2015: '중국쇼크', 2016: '트럼프랠리',
    2017: '저변동', 2018: '금리쇼크', 2019: '회복', 2020: '코로나+V반등',
    2021: '과열', 2022: '베어마켓', 2023: 'AI랠리', 2024: 'AI확산',
    2025: '조정', 2026: '현재진행',
}

yearly_summary = []
for yr in all_years:
    v4_vals = [all_yearly[tk][yr][0] for tk in tickers_ordered if yr in all_yearly[tk]]
    bnh_vals = [all_yearly[tk][yr][1] for tk in tickers_ordered if yr in all_yearly[tk]]
    alpha_vals = [all_yearly[tk][yr][2] for tk in tickers_ordered if yr in all_yearly[tk]]

    if not v4_vals:
        continue

    n_tk = len(v4_vals)
    avg_v4 = np.mean(v4_vals)
    avg_bnh = np.mean(bnh_vals)
    avg_alpha = np.mean(alpha_vals)
    wins = sum(1 for a in alpha_vals if a > 0)
    label = market_labels.get(yr, '')

    yearly_summary.append({
        'year': yr, 'n': n_tk, 'v4': avg_v4, 'bnh': avg_bnh,
        'alpha': avg_alpha, 'wins': wins, 'total': n_tk,
    })

    print(f"  {yr:>6d} │ {n_tk:>5d} │ {avg_v4:>+9.2f}% │ {avg_bnh:>+9.2f}% │ {avg_alpha:>+7.2f}% │ {wins:>2d}/{n_tk:<2d} │ {label}")

print(f"  {'─'*6}─┼─{'─'*5}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*8}─┼─{'─'*5}─┼─{'─'*20}")

# Grand average
all_v4_r = [s['v4'] for s in yearly_summary]
all_bnh_r = [s['bnh'] for s in yearly_summary]
all_alpha_r = [s['alpha'] for s in yearly_summary]
print(f"  {'AVG':>6s} │ {'':>5s} │ {np.mean(all_v4_r):>+9.2f}% │ {np.mean(all_bnh_r):>+9.2f}% │ {np.mean(all_alpha_r):>+7.2f}% │ {'':>5s} │")
print()


# =====================================================
# REPORT 5: 종합 분석
# =====================================================
print("=" * 120)
print("  [5] 종합 분석")
print("=" * 120)

# 전 기간 평균
all_v4_flat = []
all_bnh_flat = []
all_alpha_flat = []
for tk in tickers_ordered:
    for yr in all_yearly[tk]:
        v4, bnh, alpha = all_yearly[tk][yr]
        all_v4_flat.append(v4)
        all_bnh_flat.append(bnh)
        all_alpha_flat.append(alpha)

print(f"  전 기간 (종목×연도 관측치 {len(all_alpha_flat)}개):")
print(f"    V4 평균 연수익률:    {np.mean(all_v4_flat):>+8.2f}%")
print(f"    BnH 평균 연수익률:   {np.mean(all_bnh_flat):>+8.2f}%")
print(f"    평균 Alpha:          {np.mean(all_alpha_flat):>+8.2f}%")
print(f"    Alpha 양수 비율:     {sum(1 for a in all_alpha_flat if a > 0)}/{len(all_alpha_flat)}"
      f" ({sum(1 for a in all_alpha_flat if a > 0)/len(all_alpha_flat)*100:.1f}%)")
print(f"    Alpha 중앙값:        {np.median(all_alpha_flat):>+8.2f}%")
print()

# 기간별 분석
print("  기간별 Alpha:")
for period_name, year_range in [
    ("초기 (~2014)", range(1990, 2015)),
    ("2015~2019", range(2015, 2020)),
    ("2020~2026", range(2020, 2027)),
    ("하락장 (2008,2022)", [2008, 2022]),
    ("상승장 (2013,2017,2021)", [2013, 2017, 2021]),
    ("V반등 (2009,2020)", [2009, 2020]),
]:
    period_alphas = [all_yearly[tk][yr][2] for tk in tickers_ordered
                     for yr in all_yearly[tk] if yr in year_range]
    if period_alphas:
        print(f"    {period_name:<30s}: Alpha {np.mean(period_alphas):>+7.2f}%"
              f"  (Win {sum(1 for a in period_alphas if a > 0)}/{len(period_alphas)},"
              f" {sum(1 for a in period_alphas if a > 0)/len(period_alphas)*100:.0f}%)")

print()

# 종목별 랭킹
print("  종목별 평균 Alpha 랭킹:")
tk_rankings = []
for tk in tickers_ordered:
    alphas = [all_yearly[tk][yr][2] for yr in all_yearly[tk]]
    v4_rets = [all_yearly[tk][yr][0] for yr in all_yearly[tk]]
    bnh_rets = [all_yearly[tk][yr][1] for yr in all_yearly[tk]]
    n_years = len(alphas)
    wins = sum(1 for a in alphas if a > 0)
    tk_rankings.append((tk, np.mean(alphas), np.mean(v4_rets), np.mean(bnh_rets), wins, n_years))

tk_rankings.sort(key=lambda x: -x[1])

for rank, (tk, avg_a, avg_v4, avg_bnh, wins, n_yrs) in enumerate(tk_rankings, 1):
    sector = TICKERS.get(tk, 'Bench')
    start_yr = min(all_yearly[tk].keys())
    print(f"    {rank:>2d}. {tk:<6s} ({sector:<8s} {start_yr}~): V4={avg_v4:>+7.1f}% BnH={avg_bnh:>+7.1f}%"
          f"  Alpha={avg_a:>+7.2f}%  W/L={wins}/{n_yrs}")

print()

# V4가 잘 되는/안 되는 연도
print("  V4 Alpha Best/Worst 연도 (전종목 평균):")
sorted_years = sorted(yearly_summary, key=lambda x: -x['alpha'])

print("    [Best 5]")
for s in sorted_years[:5]:
    label = market_labels.get(s['year'], '')
    print(f"      {s['year']}: Alpha {s['alpha']:>+7.2f}%  V4={s['v4']:>+7.1f}% BnH={s['bnh']:>+7.1f}%  ({label})")

print("    [Worst 5]")
for s in sorted_years[-5:]:
    label = market_labels.get(s['year'], '')
    print(f"      {s['year']}: Alpha {s['alpha']:>+7.2f}%  V4={s['v4']:>+7.1f}% BnH={s['bnh']:>+7.1f}%  ({label})")

print()
print("=" * 120)
print("  Backtest complete.")
print("=" * 120)
