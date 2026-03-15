"""
V4_wP 백테스트: 기존 vs dd>=5% 필터 비교
===========================================
두 가지 버전을 나란히 실행하여 Alpha 변화 비교:
  A) 기존: 모든 V4 매수 신호 실행
  B) dd5%: 20일 고점 대비 5%+ 하락 시에만 매수 실행
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
# PARAMETERS
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

V4_WINDOW = 20
SIGNAL_TH = 0.15
COOLDOWN = 5
ER_Q = 66
ATR_Q = 55
LOOKBACK = 252
DIVGATE_DAYS = 3

CONFIRM_DAYS = 3
BUY_CONFIRMED_PCT = 1.00
SELL_CONFIRM_DAYS = 3
SELL_CONFIRMED_PCT = 0.05
LATE_SELL_DROP_TH = 0.05

# 신규 파라미터
BUY_DRAWDOWN_TH = 0.05  # 20일 고점 대비 5% 하락 시에만 매수


def download_max(ticker):
    t = yf.Ticker(ticker)
    df = t.history(period='max', auto_adjust=True)
    if df is None or len(df) == 0:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


# =====================================================
# DATA LOADING
# =====================================================
print("=" * 130)
print("  V4_wP 백테스트: 기존 vs Drawdown >= 5% 필터 비교")
print("=" * 130)
print(f"  기존: 모든 V4 매수 신호 실행")
print(f"  dd5%: 매수 시점에 20일 고점 대비 5%+ 하락 상태일 때만 실행")
print()

data = {}
for tk in ALL_TICKERS:
    print(f"    {tk}...", end=' ', flush=True)
    try:
        df = download_max(tk)
        if df is None or len(df) < 200:
            print("SKIP"); continue

        df_smooth = smooth_earnings_volume(df, ticker=tk)
        score = calc_v4_score(df_smooth, w=V4_WINDOW, divgate_days=DIVGATE_DAYS)
        events = detect_signal_events(score, th=SIGNAL_TH, cooldown=COOLDOWN)
        pf = build_price_filter(df_smooth, er_q=ER_Q, atr_q=ATR_Q, lookback=LOOKBACK)

        close = df['Close'].values
        rolling_high = df['Close'].rolling(20, min_periods=1).max().values

        filtered = []
        for ev in events:
            if not pf(ev['peak_idx']):
                continue
            if ev['type'] == 'top':
                pidx = ev['peak_idx']
                if pidx < len(close):
                    price = close[pidx]
                    rh = rolling_high[pidx]
                    if rh > 0 and (rh - price) / rh > LATE_SELL_DROP_TH:
                        continue
            ev['duration'] = ev['end_idx'] - ev['start_idx'] + 1
            filtered.append(ev)

        data[tk] = {'df': df, 'events': filtered, 'rolling_high': rolling_high}
        n_buy = sum(1 for e in filtered if e['type'] == 'bottom')
        n_sell = sum(1 for e in filtered if e['type'] == 'top')
        print(f"OK ({len(df)} bars, {n_buy}B/{n_sell}S)")
    except Exception as e:
        print(f"ERROR: {e}")

print(f"\n  {len(data)}/{len(ALL_TICKERS)} tickers loaded\n")


# =====================================================
# SIMULATION
# =====================================================
def simulate(events, df, rolling_high, use_dd_filter=False):
    """
    use_dd_filter=True면 매수 시점에 dd >= 5% 조건 추가.
    매수 안 된 경우 cash가 계속 쌓이다가 다음 기회에 투입됨.
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
    n_buys = 0
    n_sells = 0
    n_skipped_buys = 0  # dd 필터로 스킵된 매수

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

        if yr != current_year:
            prev_price = close[i - 1]
            yearly_data[current_year]['end_v4'] = shares * prev_price + cash
            yearly_data[current_year]['end_bnh'] = shares_bnh * prev_price + cash_bnh
            current_year = yr
            init_year(yr, shares * prev_price + cash, shares_bnh * prev_price + cash_bnh)

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

        for ev in events:
            sidx = ev['start_idx']
            eidx = ev['end_idx']
            duration = ev.get('duration', eidx - sidx + 1)

            if ev['type'] == 'bottom':
                confirm_idx = sidx + CONFIRM_DAYS - 1
                if i == confirm_idx and confirm_idx <= eidx and duration >= CONFIRM_DAYS and cash > 0:
                    # dd 필터 체크
                    if use_dd_filter:
                        rh = rolling_high[i]
                        dd = (rh - price) / rh if rh > 0 else 0
                        if dd < BUY_DRAWDOWN_TH:
                            n_skipped_buys += 1
                            continue  # 하락 부족 → 매수 스킵

                    amount = cash * BUY_CONFIRMED_PCT
                    shares += amount / price
                    cash -= amount
                    n_buys += 1

            elif ev['type'] == 'top':
                confirm_idx = sidx + SELL_CONFIRM_DAYS - 1
                if i == confirm_idx and confirm_idx <= eidx and duration >= SELL_CONFIRM_DAYS and shares > 0:
                    sell_shares = shares * SELL_CONFIRMED_PCT
                    cash += sell_shares * price
                    shares -= sell_shares
                    n_sells += 1

    yearly_data[current_year]['end_v4'] = shares * close[-1] + cash
    yearly_data[current_year]['end_bnh'] = shares_bnh * close[-1] + cash_bnh

    # Modified Dietz
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

    return year_results, {
        'final_v4': final_v4, 'final_bnh': final_bnh,
        'total_invested': total_invested,
        'total_return_v4': (final_v4 / total_invested - 1) * 100,
        'total_return_bnh': (final_bnh / total_invested - 1) * 100,
        'n_buys': n_buys, 'n_sells': n_sells, 'n_skipped': n_skipped_buys,
        'cash_remaining': cash,
    }


# =====================================================
# RUN BOTH VERSIONS
# =====================================================
results_old = {}  # 기존
results_new = {}  # dd5%

tickers_ordered = [tk for tk in ALL_TICKERS if tk in data]

for tk in tickers_ordered:
    d = data[tk]
    yr_old, stats_old = simulate(d['events'], d['df'], d['rolling_high'], use_dd_filter=False)
    yr_new, stats_new = simulate(d['events'], d['df'], d['rolling_high'], use_dd_filter=True)
    results_old[tk] = {'yearly': yr_old, 'stats': stats_old}
    results_new[tk] = {'yearly': yr_new, 'stats': stats_new}


# =====================================================
# REPORT 1: 종목별 매매 횟수 비교
# =====================================================
print("=" * 130)
print("  [1] 종목별 매매 횟수 비교")
print("=" * 130)
print(f"  {'Ticker':<7s} {'Sector':<8s} | {'기존 매수':>8s} {'기존 매도':>8s} | {'dd5% 매수':>9s} {'dd5% 매도':>9s} {'스킵':>5s} | {'감소율':>6s}")
print(f"  {'-'*90}")

total_old_buys = total_new_buys = total_skipped = 0
for tk in tickers_ordered:
    sector = TICKERS.get(tk, 'Bench')
    so = results_old[tk]['stats']
    sn = results_new[tk]['stats']
    reduction = (1 - sn['n_buys'] / so['n_buys'] if so['n_buys'] > 0 else 0) * 100

    total_old_buys += so['n_buys']
    total_new_buys += sn['n_buys']
    total_skipped += sn['n_skipped']

    print(f"  {tk:<7s} {sector:<8s} | {so['n_buys']:>8d} {so['n_sells']:>8d} |"
          f" {sn['n_buys']:>9d} {sn['n_sells']:>9d} {sn['n_skipped']:>5d} | {reduction:>5.0f}%")

print(f"  {'-'*90}")
total_reduction = (1 - total_new_buys / total_old_buys if total_old_buys > 0 else 0) * 100
print(f"  {'TOTAL':<7s} {'':8s} | {total_old_buys:>8d} {'':>8s} |"
      f" {total_new_buys:>9d} {'':>9s} {total_skipped:>5d} | {total_reduction:>5.0f}%")
print()


# =====================================================
# REPORT 2: 종목별 Alpha 비교 (기존 vs dd5%)
# =====================================================
print("=" * 130)
print("  [2] 종목별 전 기간 Alpha 비교: 기존 vs dd5%")
print("=" * 130)
print(f"  {'Ticker':<7s} {'Sector':<8s} {'Period':>10s} | {'기존 V4':>8s} {'기존 BnH':>8s} {'기존 Alpha':>10s}"
      f" | {'dd5% V4':>8s} {'dd5% BnH':>8s} {'dd5% Alpha':>10s} | {'차이':>7s}")
print(f"  {'-'*120}")

for tk in tickers_ordered:
    sector = TICKERS.get(tk, 'Bench')
    df = data[tk]['df']
    period = f"{df.index[0].year}-{df.index[-1].year}"

    so = results_old[tk]['stats']
    sn = results_new[tk]['stats']

    alpha_old = so['total_return_v4'] - so['total_return_bnh']
    alpha_new = sn['total_return_v4'] - sn['total_return_bnh']
    diff = alpha_new - alpha_old

    marker = ' <<' if diff > 0 else ''
    print(f"  {tk:<7s} {sector:<8s} {period:>10s} | {so['total_return_v4']:>+7.1f}% {so['total_return_bnh']:>+7.1f}% {alpha_old:>+9.1f}%"
          f" | {sn['total_return_v4']:>+7.1f}% {sn['total_return_bnh']:>+7.1f}% {alpha_new:>+9.1f}% | {diff:>+6.1f}%{marker}")

print(f"  {'-'*120}")

# 전체 평균
old_alphas = [results_old[tk]['stats']['total_return_v4'] - results_old[tk]['stats']['total_return_bnh'] for tk in tickers_ordered]
new_alphas = [results_new[tk]['stats']['total_return_v4'] - results_new[tk]['stats']['total_return_bnh'] for tk in tickers_ordered]
print(f"  {'AVG':<7s} {'':8s} {'':>10s} | {'':>8s} {'':>8s} {np.mean(old_alphas):>+9.1f}%"
      f" | {'':>8s} {'':>8s} {np.mean(new_alphas):>+9.1f}% | {np.mean(new_alphas)-np.mean(old_alphas):>+6.1f}%")

n_improved = sum(1 for i in range(len(tickers_ordered)) if new_alphas[i] > old_alphas[i])
print(f"  dd5%가 개선된 종목: {n_improved}/{len(tickers_ordered)}")
print()


# =====================================================
# REPORT 3: 연도별 Alpha 비교
# =====================================================
all_years = sorted(set(yr for tk in tickers_ordered for yr in results_old[tk]['yearly']))

print("=" * 130)
print("  [3] 연도별 Alpha 비교: 기존 vs dd5% (전종목 평균)")
print("=" * 130)
print(f"  {'Year':>6s} | {'기존 V4':>8s} {'기존 BnH':>8s} {'기존 Alpha':>10s} | {'dd5% V4':>8s} {'dd5% BnH':>8s} {'dd5% Alpha':>10s} | {'차이':>7s} {'판정'}")
print(f"  {'-'*110}")

market_labels = {
    1997: '닷컴초기', 1998: '닷컴호황', 1999: '버블', 2000: '붕괴',
    2001: '침체', 2002: '바닥', 2003: '회복', 2007: '서브프라임', 2008: '금융위기',
    2009: '바닥반등', 2010: '회복', 2011: '유럽위기', 2013: '강세', 2015: '중국쇼크',
    2018: '금리쇼크', 2019: '회복', 2020: '코로나V반등', 2021: '과열',
    2022: '베어마켓', 2023: 'AI랠리', 2024: 'AI확산', 2025: '조정', 2026: '현재',
}

yearly_alpha_old = []
yearly_alpha_new = []

for yr in all_years:
    old_vals = [results_old[tk]['yearly'][yr] for tk in tickers_ordered if yr in results_old[tk]['yearly']]
    new_vals = [results_new[tk]['yearly'][yr] for tk in tickers_ordered if yr in results_new[tk]['yearly']]

    if not old_vals:
        continue

    avg_v4_old = np.mean([v[0] for v in old_vals])
    avg_bnh_old = np.mean([v[1] for v in old_vals])
    avg_alpha_old = np.mean([v[2] for v in old_vals])

    avg_v4_new = np.mean([v[0] for v in new_vals])
    avg_bnh_new = np.mean([v[1] for v in new_vals])
    avg_alpha_new = np.mean([v[2] for v in new_vals])

    diff = avg_alpha_new - avg_alpha_old
    label = market_labels.get(yr, '')
    verdict = 'dd5%승' if diff > 0.5 else '기존승' if diff < -0.5 else '동일'

    yearly_alpha_old.append(avg_alpha_old)
    yearly_alpha_new.append(avg_alpha_new)

    print(f"  {yr:>6d} | {avg_v4_old:>+7.1f}% {avg_bnh_old:>+7.1f}% {avg_alpha_old:>+9.2f}%"
          f" | {avg_v4_new:>+7.1f}% {avg_bnh_new:>+7.1f}% {avg_alpha_new:>+9.2f}% | {diff:>+6.2f}% {verdict:6s} {label}")

print(f"  {'-'*110}")
avg_old = np.mean(yearly_alpha_old)
avg_new = np.mean(yearly_alpha_new)
print(f"  {'AVG':>6s} | {'':>8s} {'':>8s} {avg_old:>+9.2f}%"
      f" | {'':>8s} {'':>8s} {avg_new:>+9.2f}% | {avg_new-avg_old:>+6.2f}%")
n_yr_improved = sum(1 for i in range(len(yearly_alpha_old)) if yearly_alpha_new[i] > yearly_alpha_old[i])
print(f"  dd5%가 나은 연도: {n_yr_improved}/{len(yearly_alpha_old)}")
print()


# =====================================================
# REPORT 4: 기간별 분석
# =====================================================
print("=" * 130)
print("  [4] 기간별 Alpha 비교")
print("=" * 130)

for period_name, year_range in [
    ("전 기간", range(1990, 2030)),
    ("초기 (~2014)", range(1990, 2015)),
    ("2015~2019", range(2015, 2020)),
    ("2020~2026", range(2020, 2027)),
    ("하락장 (2008,2022)", [2008, 2022]),
    ("상승장 (2013,2017,2021)", [2013, 2017, 2021]),
    ("V반등 (2009,2020)", [2009, 2020]),
]:
    old_a = [results_old[tk]['yearly'][yr][2] for tk in tickers_ordered
             for yr in results_old[tk]['yearly'] if yr in year_range]
    new_a = [results_new[tk]['yearly'][yr][2] for tk in tickers_ordered
             for yr in results_new[tk]['yearly'] if yr in year_range]
    if old_a:
        diff = np.mean(new_a) - np.mean(old_a)
        marker = ' <<' if diff > 0 else ''
        print(f"  {period_name:<30s}: 기존 {np.mean(old_a):>+7.2f}%  dd5% {np.mean(new_a):>+7.2f}%  차이 {diff:>+6.2f}%{marker}")

print()


# =====================================================
# REPORT 5: 최종 포트폴리오 비교
# =====================================================
print("=" * 130)
print("  [5] 최종 포트폴리오 비교")
print("=" * 130)
print(f"  {'Ticker':<7s} | {'투자금':>10s} | {'기존V4':>10s} {'기존수익':>8s} | {'dd5%V4':>10s} {'dd5%수익':>8s} | {'BnH':>10s} {'BnH수익':>8s} | {'dd5%잔여현금':>12s}")
print(f"  {'-'*115}")

total_inv = total_old_v4 = total_new_v4 = total_bnh = 0
for tk in tickers_ordered:
    so = results_old[tk]['stats']
    sn = results_new[tk]['stats']

    total_inv += so['total_invested']
    total_old_v4 += so['final_v4']
    total_new_v4 += sn['final_v4']
    total_bnh += so['final_bnh']

    print(f"  {tk:<7s} | ${so['total_invested']:>9,.0f} | ${so['final_v4']:>9,.0f} {so['total_return_v4']:>+7.1f}%"
          f" | ${sn['final_v4']:>9,.0f} {sn['total_return_v4']:>+7.1f}%"
          f" | ${so['final_bnh']:>9,.0f} {so['total_return_bnh']:>+7.1f}%"
          f" | ${sn['cash_remaining']:>10,.0f}")

print(f"  {'-'*115}")
print(f"  {'TOTAL':<7s} | ${total_inv:>9,.0f} | ${total_old_v4:>9,.0f} {(total_old_v4/total_inv-1)*100:>+7.1f}%"
      f" | ${total_new_v4:>9,.0f} {(total_new_v4/total_inv-1)*100:>+7.1f}%"
      f" | ${total_bnh:>9,.0f} {(total_bnh/total_inv-1)*100:>+7.1f}%")
print()


# =====================================================
# REPORT 6: 종합 결론
# =====================================================
print("=" * 130)
print("  [6] 종합 결론")
print("=" * 130)
print()

# 전체 종목×연도 기준
all_old_alpha = [results_old[tk]['yearly'][yr][2] for tk in tickers_ordered for yr in results_old[tk]['yearly']]
all_new_alpha = [results_new[tk]['yearly'][yr][2] for tk in tickers_ordered for yr in results_new[tk]['yearly']]

print(f"  전체 (종목×연도 {len(all_old_alpha)}개):")
print(f"    기존 평균 Alpha:    {np.mean(all_old_alpha):>+8.2f}%")
print(f"    dd5% 평균 Alpha:    {np.mean(all_new_alpha):>+8.2f}%")
print(f"    차이:               {np.mean(all_new_alpha)-np.mean(all_old_alpha):>+8.2f}%p")
print()
print(f"    기존 Alpha 양수 비율: {sum(1 for a in all_old_alpha if a>0)}/{len(all_old_alpha)}"
      f" ({sum(1 for a in all_old_alpha if a>0)/len(all_old_alpha)*100:.1f}%)")
print(f"    dd5% Alpha 양수 비율: {sum(1 for a in all_new_alpha if a>0)}/{len(all_new_alpha)}"
      f" ({sum(1 for a in all_new_alpha if a>0)/len(all_new_alpha)*100:.1f}%)")
print()

# 매매 횟수 비교
print(f"  매매 횟수:")
print(f"    기존 매수: {total_old_buys}회")
print(f"    dd5% 매수: {total_new_buys}회 (스킵 {total_skipped}회)")
print(f"    매수 감소율: {total_reduction:.0f}%")
print()

# 포트폴리오 비교
print(f"  포트폴리오 총합:")
print(f"    투자금 합계:   ${total_inv:>12,.0f}")
print(f"    기존 V4 합계:  ${total_old_v4:>12,.0f} ({(total_old_v4/total_inv-1)*100:>+.1f}%)")
print(f"    dd5% V4 합계:  ${total_new_v4:>12,.0f} ({(total_new_v4/total_inv-1)*100:>+.1f}%)")
print(f"    BnH 합계:      ${total_bnh:>12,.0f} ({(total_bnh/total_inv-1)*100:>+.1f}%)")
print()

if np.mean(all_new_alpha) > np.mean(all_old_alpha):
    print(f"  결론: dd5% 필터가 기존 대비 Alpha {np.mean(all_new_alpha)-np.mean(all_old_alpha):>+.2f}%p 개선")
else:
    print(f"  결론: dd5% 필터가 기존 대비 Alpha {np.mean(all_new_alpha)-np.mean(all_old_alpha):>+.2f}%p 변화")

print()
print("=" * 130)
print("Done.")
