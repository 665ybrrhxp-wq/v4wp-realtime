"""
V4 Signal + Month-end DCA  vs  Pure Month-end DCA
==================================================
VOO & QQQ 백테스트 비교

전략 A (V4+DCA):
  - 매월 초 $500 입금
  - V4 매수 신호 발생 시: 가용 자금의 50% 매수
  - 월 말: 남은 자금 전부 매수

전략 B (순수 DCA):
  - 매월 초 $500 입금
  - 월 말: $500 전부 매수
"""
import sys, os, warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import yfinance as yf
from real_market_backtest import (
    calc_v4_score, detect_signal_events, build_price_filter,
    smooth_earnings_volume,
)

# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════
TICKERS = ['VOO', 'QQQ']
MONTHLY_DEPOSIT = 500.0
SIGNAL_BUY_PCT = 0.50  # V4 signal -> 가용자금의 50% 매수

# V4 production parameters
V4_W = 20
SIGNAL_TH = 0.15
COOLDOWN = 5
ER_Q = 66
ATR_Q = 55
LOOKBACK = 252
DIVGATE = 3
CONFIRM = 3


def download_max(ticker):
    t = yf.Ticker(ticker)
    df = t.history(period='max', auto_adjust=True)
    if df is None or len(df) == 0:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def get_buy_signal_indices(df, ticker):
    """Return set of bar indices with confirmed V4 buy signals"""
    try:
        df_s = smooth_earnings_volume(df, ticker=ticker)
    except Exception:
        df_s = df.copy()
    score = calc_v4_score(df_s, w=V4_W, divgate_days=DIVGATE)
    events = detect_signal_events(score, th=SIGNAL_TH, cooldown=COOLDOWN)
    pf = build_price_filter(df_s, er_q=ER_Q, atr_q=ATR_Q, lookback=LOOKBACK)

    n = len(df)
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


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
print("=" * 100)
print("  V4 Signal + Month-end DCA  vs  Pure Month-end DCA")
print(f"  Monthly deposit: ${MONTHLY_DEPOSIT:.0f} | Signal buy: {SIGNAL_BUY_PCT:.0%} of cash")
print("=" * 100)

for tk in TICKERS:
    print(f"\n{'━' * 100}")
    print(f"  {tk}")
    print(f"{'━' * 100}")

    df = download_max(tk)
    if df is None or len(df) < 200:
        print("  Insufficient data")
        continue

    close = df['Close'].values
    dates = df.index
    n = len(close)

    print(f"  Period: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')} ({n} bars)")

    # V4 buy signals
    buy_signals = get_buy_signal_indices(df, tk)
    print(f"  V4 buy signals: {len(buy_signals)}")

    # Build month map: period -> (first_idx, last_idx)
    month_periods = pd.Series(dates).dt.to_period('M')
    month_map = {}
    for i, mp in enumerate(month_periods):
        key = str(mp)
        if key not in month_map:
            month_map[key] = {'first': i, 'last': i, 'period': mp}
        else:
            month_map[key]['last'] = i
    sorted_month_keys = sorted(month_map.keys())

    # ── Simulation ──
    cash_a = 0.0;  shares_a = 0.0
    cash_b = 0.0;  shares_b = 0.0
    total_deposited = 0.0
    signal_buy_count = 0
    signal_buy_total_amount = 0.0

    # Year tracking
    year_snapshots = {}  # yr -> {start_a, start_b, deposits, end_a, end_b, signal_buys}
    prev_yr = None

    for mk in sorted_month_keys:
        mm = month_map[mk]
        fi, li = mm['first'], mm['last']
        yr = mm['period'].year

        # Year transition
        if yr != prev_yr:
            if prev_yr is not None:
                # Close previous year
                year_snapshots[prev_yr]['end_a'] = shares_a * close[fi - 1] + cash_a if fi > 0 else 0
                year_snapshots[prev_yr]['end_b'] = shares_b * close[fi - 1] + cash_b if fi > 0 else 0
            year_snapshots[yr] = {
                'start_a': shares_a * close[fi] + cash_a,
                'start_b': shares_b * close[fi] + cash_b,
                'deposits': 0.0, 'end_a': 0, 'end_b': 0, 'signal_buys': 0,
            }
            prev_yr = yr

        # Month start: deposit
        cash_a += MONTHLY_DEPOSIT
        cash_b += MONTHLY_DEPOSIT
        total_deposited += MONTHLY_DEPOSIT
        year_snapshots[yr]['deposits'] += MONTHLY_DEPOSIT

        # Strategy A: check V4 signals during the month
        for day_idx in range(fi, li + 1):
            if day_idx in buy_signals and cash_a > 1.0:
                buy_amt = cash_a * SIGNAL_BUY_PCT
                shares_a += buy_amt / close[day_idx]
                cash_a -= buy_amt
                signal_buy_count += 1
                signal_buy_total_amount += buy_amt
                year_snapshots[yr]['signal_buys'] += 1

        # Strategy A: month-end buy remaining
        if cash_a > 1.0:
            shares_a += cash_a / close[li]
            cash_a = 0.0

        # Strategy B: month-end buy all
        if cash_b > 1.0:
            shares_b += cash_b / close[li]
            cash_b = 0.0

    # Close final year
    fp = close[-1]
    year_snapshots[prev_yr]['end_a'] = shares_a * fp + cash_a
    year_snapshots[prev_yr]['end_b'] = shares_b * fp + cash_b

    final_a = shares_a * fp + cash_a
    final_b = shares_b * fp + cash_b

    avg_price_a = total_deposited / shares_a if shares_a > 0 else 0
    avg_price_b = total_deposited / shares_b if shares_b > 0 else 0

    total_ret_a = (final_a / total_deposited - 1) * 100
    total_ret_b = (final_b / total_deposited - 1) * 100

    # ══════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════
    print(f"\n  {'─' * 96}")
    print(f"  {'':30s} {'V4+월말DCA':>20s} {'순수 월말DCA':>20s} {'차이':>15s}")
    print(f"  {'─' * 96}")
    print(f"  {'총 입금액':<28s}   ${total_deposited:>17,.0f} ${total_deposited:>17,.0f}")
    print(f"  {'최종 포트폴리오':<25s}   ${final_a:>17,.2f} ${final_b:>17,.2f} ${final_a - final_b:>+12,.2f}")
    print(f"  {'총 수익률':<28s}   {total_ret_a:>17.2f}% {total_ret_b:>17.2f}% {total_ret_a - total_ret_b:>+12.2f}%p")
    print(f"  {'보유 주수':<28s}   {shares_a:>17.4f} {shares_b:>17.4f} {shares_a - shares_b:>+12.4f}")
    print(f"  {'평균 매수가':<25s}   ${avg_price_a:>17.2f} ${avg_price_b:>17.2f} ${avg_price_a - avg_price_b:>+12.2f}")
    print(f"  {'─' * 96}")
    print(f"  {'V4 신호 매수 횟수':<25s}   {signal_buy_count:>17d}")
    print(f"  {'V4 신호 매수 총액':<22s}   ${signal_buy_total_amount:>17,.2f} ({signal_buy_total_amount / total_deposited * 100:.1f}% of deposits)")
    print(f"  {'총 월수':<28s}   {len(sorted_month_keys):>17d}")

    # ══════════════════════════════════════════════════════
    # Yearly breakdown
    # ══════════════════════════════════════════════════════
    print(f"\n  {'─' * 96}")
    print(f"  연도별 수익률 비교 (Modified Dietz)")
    print(f"  {'─' * 96}")
    print(f"  {'연도':>6s} {'V4+DCA':>12s} {'순수DCA':>12s} {'차이':>10s} {'V4신호':>8s} {'판정':>10s}")
    print(f"  {'─' * 70}")

    yr_results = []
    for yr in sorted(year_snapshots.keys()):
        yd = year_snapshots[yr]
        sa, sb = yd['start_a'], yd['start_b']
        ea, eb = yd['end_a'], yd['end_b']
        dep = yd['deposits']
        sig = yd['signal_buys']

        denom_a = sa + dep * 0.5
        denom_b = sb + dep * 0.5

        ret_a = ((ea - sa - dep) / denom_a * 100) if denom_a > 10 else 0
        ret_b = ((eb - sb - dep) / denom_b * 100) if denom_b > 10 else 0
        diff = ret_a - ret_b

        yr_results.append((yr, ret_a, ret_b, diff, sig))

        if diff > 0.5:
            verdict = "V4 WIN"
        elif diff < -0.5:
            verdict = "DCA WIN"
        else:
            verdict = "DRAW"
        print(f"  {yr:>6d} {ret_a:>+10.2f}% {ret_b:>+10.2f}% {diff:>+8.2f}%p {sig:>6d}   {verdict}")

    # Averages
    if yr_results:
        avg_a = np.mean([r[1] for r in yr_results])
        avg_b = np.mean([r[2] for r in yr_results])
        avg_diff = avg_a - avg_b
        wins_v4 = sum(1 for r in yr_results if r[3] > 0.5)
        wins_dca = sum(1 for r in yr_results if r[3] < -0.5)
        draws = len(yr_results) - wins_v4 - wins_dca

        print(f"  {'─' * 70}")
        print(f"  {'AVG':>6s} {avg_a:>+10.2f}% {avg_b:>+10.2f}% {avg_diff:>+8.2f}%p")
        print(f"  {'MED':>6s} {np.median([r[1] for r in yr_results]):>+10.2f}% "
              f"{np.median([r[2] for r in yr_results]):>+10.2f}% "
              f"{np.median([r[3] for r in yr_results]):>+8.2f}%p")
        print(f"\n  V4 승: {wins_v4}  |  DCA 승: {wins_dca}  |  무승부: {draws}  (총 {len(yr_results)}년)")

    # ══════════════════════════════════════════════════════
    # V4 signal dates detail
    # ══════════════════════════════════════════════════════
    if buy_signals:
        signal_list = sorted(buy_signals)
        print(f"\n  {'─' * 96}")
        print(f"  V4 매수 신호 상세 (총 {len(signal_list)}건)")
        print(f"  {'─' * 96}")
        print(f"  {'날짜':<14s} {'매수가':>10s} {'V4 Score':>10s}")
        print(f"  {'─' * 40}")

        try:
            df_s = smooth_earnings_volume(df, ticker=tk)
        except Exception:
            df_s = df.copy()
        score = calc_v4_score(df_s, w=V4_W, divgate_days=DIVGATE)
        score_vals = score.values

        for idx in signal_list:
            d = dates[idx].strftime('%Y-%m-%d')
            p = close[idx]
            sv = score_vals[idx] if idx < len(score_vals) else 0
            print(f"  {d:<14s} ${p:>9.2f} {sv:>+9.4f}")

    # ══════════════════════════════════════════════════════
    # Market condition analysis
    # ══════════════════════════════════════════════════════
    if yr_results:
        print(f"\n  {'─' * 96}")
        print(f"  시장 상황별 V4 효과")
        print(f"  {'─' * 96}")

        market_labels = {
            2010: 'QE회복', 2011: '유럽위기', 2012: '안정', 2013: '강세',
            2014: '안정', 2015: '중국쇼크', 2016: '트럼프랠리', 2017: '저변동',
            2018: '금리쇼크', 2019: '회복', 2020: '코로나+V반등', 2021: '과열',
            2022: '베어마켓', 2023: 'AI랠리', 2024: 'AI확산', 2025: '관세쇼크',
            2026: '현재',
            1999: '닷컴호황', 2000: '닷컴붕괴', 2001: '침체', 2002: '바닥',
            2003: '회복', 2004: '성장', 2005: '안정', 2006: '과열',
            2007: '서브프라임', 2008: '금융위기', 2009: '바닥반등',
        }

        # Best/Worst years for V4
        sorted_by_diff = sorted(yr_results, key=lambda x: -x[3])

        print(f"  [V4가 가장 효과적이었던 연도]")
        for yr, ra, rb, diff, sig in sorted_by_diff[:5]:
            label = market_labels.get(yr, '')
            print(f"    {yr} ({label}): V4={ra:>+.2f}% DCA={rb:>+.2f}% 차이={diff:>+.2f}%p (신호 {sig}건)")

        print(f"\n  [V4가 불리했던 연도]")
        for yr, ra, rb, diff, sig in sorted_by_diff[-5:]:
            label = market_labels.get(yr, '')
            print(f"    {yr} ({label}): V4={ra:>+.2f}% DCA={rb:>+.2f}% 차이={diff:>+.2f}%p (신호 {sig}건)")

        # Bear vs Bull
        bear_years = [2008, 2022, 2018, 2011, 2015, 2000, 2001, 2002]
        bull_years = [2013, 2017, 2019, 2021, 2023, 2024]
        crisis_years = [2008, 2020, 2022]

        for label, target_years in [
            ("하락장", bear_years),
            ("상승장", bull_years),
            ("위기", crisis_years),
        ]:
            matches = [r for r in yr_results if r[0] in target_years]
            if matches:
                avg_d = np.mean([r[3] for r in matches])
                wins = sum(1 for r in matches if r[3] > 0)
                print(f"\n  {label} ({len(matches)}년): 평균 V4 차이 = {avg_d:>+.2f}%p  (V4 승 {wins}/{len(matches)})")


# ══════════════════════════════════════════════════════
# Grand Summary
# ══════════════════════════════════════════════════════
print(f"\n{'━' * 100}")
print(f"  GRAND SUMMARY")
print(f"{'━' * 100}")
print(f"""
  전략 비교:
    전략 A (V4+DCA): V4 매수 신호 시 가용자금의 50% 즉시 매수 + 월말 잔액 전부 매수
    전략 B (순수 DCA): 매월 말 $500 전부 매수

  핵심: 두 전략 모두 매월 동일 금액($500) 투자.
        차이는 '월 중 V4 신호에 일부 선매수' vs '월말 일괄 매수'
        V4 신호가 저점을 잡으면 A가 유리, 고점을 잡으면 B가 유리.
""")
print("=" * 100)
print("  Done.")
