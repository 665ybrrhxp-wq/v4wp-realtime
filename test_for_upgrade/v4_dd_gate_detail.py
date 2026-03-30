"""
DD Gate ON/OFF 상세 분석
========================
GEO-OP 파이프라인에서 BUY_DD_GATE만 ON/OFF하고:
  1) 종목별 신호 수/적중률/수익률 비교
  2) 차단된 신호의 품질 심층 분석
  3) DD% 분포 (신호 시점의 drawdown 크기)
  4) 연도별 분석 (시장 환경에 따른 DD gate 효과)
  5) 포트폴리오 시뮬레이션 (ON vs OFF 누적수익)
  6) 승/패 분포 + 리스크 지표
"""
import sys, os, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import yfinance as yf
from collections import defaultdict
from real_market_backtest import (
    calc_v4_score, calc_v4_subindicators,
    detect_signal_events, build_price_filter, smooth_earnings_volume,
)

# ═══════════════════════════════════════════════════════════
# GEO-OP 파라미터
# ═══════════════════════════════════════════════════════════
TICKERS = {
    'TSLA': 'Tech', 'NVDA': 'Tech', 'AVGO': 'Tech',
    'AMZN': 'Tech', 'GOOGL': 'Tech', 'PLTR': 'Tech',
    'COIN': 'Fintech', 'HOOD': 'Fintech',
    'HIMS': 'Growth', 'RKLB': 'Growth', 'JOBY': 'Growth',
    'IONQ': 'Quantum',
    'QQQ': 'Benchmark', 'VOO': 'Benchmark',
}

SIGNAL_TH = 0.05
COOLDOWN = 5
ER_Q = 80
ATR_Q = 40
LOOKBACK = 252
CONFIRM_DAYS = 1
DIVGATE = 3
BUY_DD_LB = 20
BUY_DD_TH = 0.03


def download_data(ticker, years=5):
    t = yf.Ticker(ticker)
    df = t.history(period=f'{years}y', auto_adjust=True)
    if df is None or len(df) == 0:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.dropna(subset=['Close', 'Volume'])
    df = df[df['Close'] > 0]
    return df


def get_buy_signals_detailed(df, score, use_dd_gate=True):
    """매수 신호 추출 + 각 신호의 DD% 정보 포함."""
    events = detect_signal_events(score, th=SIGNAL_TH, cooldown=COOLDOWN)
    pf = build_price_filter(df, er_q=ER_Q, atr_q=ATR_Q, lookback=LOOKBACK)

    close = df['Close'].values
    dates = df.index
    rolling_high = pd.Series(close).rolling(BUY_DD_LB, min_periods=1).max().values
    n = len(df)
    signals = []  # list of dicts
    blocked_signals = []

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
        dd_pct = (rh - close[pidx]) / rh * 100 if rh > 0 else 0
        price = close[ci]
        date = dates[ci]

        sig_info = {
            'idx': ci, 'price': price, 'date': date,
            'dd_pct': dd_pct, 'peak_score': ev['peak_val'],
        }

        if use_dd_gate and dd_pct < BUY_DD_TH * 100:
            blocked_signals.append(sig_info)
            continue

        signals.append(sig_info)

    return signals, blocked_signals


def evaluate_signals_detailed(close, dates, signals):
    """신호별 상세 평가."""
    n = len(close)
    results = []

    for s in signals:
        idx = s['idx']
        entry_price = close[idx]
        entry_date = dates[idx]
        year = entry_date.year

        fwd_30 = ((close[idx + 30] / entry_price - 1) * 100) if idx + 30 < n else None
        fwd_60 = ((close[idx + 60] / entry_price - 1) * 100) if idx + 60 < n else None
        fwd_90 = ((close[idx + 90] / entry_price - 1) * 100) if idx + 90 < n else None
        fwd_180 = ((close[idx + 180] / entry_price - 1) * 100) if idx + 180 < n else None

        # MDD within 90d
        end_i = min(idx + 90, n)
        mdd_90 = 0
        if end_i > idx + 1:
            min_p = min(close[idx:end_i])
            mdd_90 = (min_p / entry_price - 1) * 100

        # Max gain within 90d
        max_gain_90 = 0
        if end_i > idx + 1:
            max_p = max(close[idx:end_i])
            max_gain_90 = (max_p / entry_price - 1) * 100

        results.append({
            **s,
            'year': year,
            'fwd_30': fwd_30, 'fwd_60': fwd_60,
            'fwd_90': fwd_90, 'fwd_180': fwd_180,
            'mdd_90': mdd_90, 'max_gain_90': max_gain_90,
        })

    return results


def portfolio_sim(close, dates, signals, init_cash=10000):
    """간단한 포트폴리오 시뮬레이션: 신호마다 균등 매수, 90일 후 매도."""
    n = len(close)
    trades = []
    for s in signals:
        idx = s['idx']
        sell_idx = min(idx + 90, n - 1)
        ret = (close[sell_idx] / close[idx] - 1)
        trades.append({
            'entry_date': dates[idx],
            'exit_date': dates[sell_idx],
            'entry_price': close[idx],
            'exit_price': close[sell_idx],
            'return': ret * 100,
        })

    if not trades:
        return {'total_return': 0, 'avg_return': 0, 'trades': 0}

    # 균등배분: 각 트레이드에 동일 금액 배분
    avg_ret = np.mean([t['return'] for t in trades])
    total_ret = sum(t['return'] for t in trades) / len(trades) * len(trades)
    # 복리 시뮬
    capital = init_cash
    for t in trades:
        allocation = capital / max(len([tt for tt in trades if tt['entry_date'] >= t['entry_date']]), 1)
        capital += allocation * t['return'] / 100

    return {
        'total_return': (capital / init_cash - 1) * 100,
        'avg_return': avg_ret,
        'trades': len(trades),
        'win_trades': sum(1 for t in trades if t['return'] > 0),
        'loss_trades': sum(1 for t in trades if t['return'] <= 0),
    }


def print_section(title):
    sep = '=' * 100
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)


def main():
    print('=' * 100)
    print("  DD Gate ON/OFF 상세 분석")
    print("  GEO-OP 프로덕션 파이프라인, DD Gate(20d/3%)만 토글")
    print('=' * 100)

    # ═══════════════════════════════════════════════════════
    # 1. 데이터 로딩
    # ═══════════════════════════════════════════════════════
    print("\n  데이터 로딩...")
    data = {}
    for tk in TICKERS:
        df = download_data(tk, years=5)
        if df is None or len(df) < 300:
            print(f"    {tk}: SKIP (insufficient data)")
            continue
        df = smooth_earnings_volume(df, ticker=tk)
        data[tk] = df
        print(f"    {tk}: {len(df)} bars ({len(df)/252:.1f}yr) | "
              f"{df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  {len(data)} tickers loaded.\n")

    # ═══════════════════════════════════════════════════════
    # 2. 종목별 신호 생성 + 평가
    # ═══════════════════════════════════════════════════════
    all_on = []     # 전체 ON 신호
    all_off = []    # 전체 OFF 신호
    all_blocked = [] # 차단된 신호

    print_section("1. 종목별 비교 (DD Gate ON vs OFF)")
    print(f"  {'Ticker':<8} {'섹터':<10} │ {'ON':>4} {'OFF':>4} {'차단':>4} │ "
          f"{'적중ON':>7} {'적중OFF':>8} │ {'Fwd90ON':>8} {'Fwd90OFF':>9} │ "
          f"{'MDD ON':>7} {'MDD OFF':>8} │ {'판정'}")
    print("  " + "-" * 100)

    for tk, df in data.items():
        score = calc_v4_score(df, w=20, divgate_days=DIVGATE)
        close = df['Close'].values
        dates = df.index

        # ON
        sigs_on, blocked = get_buy_signals_detailed(df, score, use_dd_gate=True)
        evals_on = evaluate_signals_detailed(close, dates, sigs_on)
        for e in evals_on:
            e['ticker'] = tk
            e['sector'] = TICKERS[tk]
        all_on.extend(evals_on)

        # OFF
        sigs_off, _ = get_buy_signals_detailed(df, score, use_dd_gate=False)
        evals_off = evaluate_signals_detailed(close, dates, sigs_off)
        for e in evals_off:
            e['ticker'] = tk
            e['sector'] = TICKERS[tk]
        all_off.extend(evals_off)

        # Blocked
        evals_blocked = evaluate_signals_detailed(close, dates, blocked)
        for e in evals_blocked:
            e['ticker'] = tk
            e['sector'] = TICKERS[tk]
        all_blocked.extend(evals_blocked)

        # 종목별 통계
        on_90 = [e['fwd_90'] for e in evals_on if e['fwd_90'] is not None]
        off_90 = [e['fwd_90'] for e in evals_off if e['fwd_90'] is not None]
        blk_90 = [e['fwd_90'] for e in evals_blocked if e['fwd_90'] is not None]
        on_mdd = [e['mdd_90'] for e in evals_on if e['mdd_90'] is not None]
        off_mdd = [e['mdd_90'] for e in evals_off if e['mdd_90'] is not None]

        hit_on = (sum(1 for x in on_90 if x > 0) / len(on_90) * 100) if on_90 else 0
        hit_off = (sum(1 for x in off_90 if x > 0) / len(off_90) * 100) if off_90 else 0

        # 판정
        diff = hit_on - hit_off
        if diff > 5:
            verdict = "ON 우세"
        elif diff < -5:
            verdict = "OFF 우세"
        else:
            verdict = "동등"

        print(f"  {tk:<8} {TICKERS[tk]:<10} │ {len(sigs_on):>4} {len(sigs_off):>4} {len(blocked):>4} │ "
              f"{hit_on:>6.1f}% {hit_off:>7.1f}% │ "
              f"{np.mean(on_90) if on_90 else 0:>+7.1f}% {np.mean(off_90) if off_90 else 0:>+8.1f}% │ "
              f"{np.mean(on_mdd) if on_mdd else 0:>+6.1f}% {np.mean(off_mdd) if off_mdd else 0:>+7.1f}% │ "
              f"{verdict}")

    # ═══════════════════════════════════════════════════════
    # 3. 전체 합산
    # ═══════════════════════════════════════════════════════
    print_section("2. 전체 합산 비교")

    on_90 = [e['fwd_90'] for e in all_on if e['fwd_90'] is not None]
    off_90 = [e['fwd_90'] for e in all_off if e['fwd_90'] is not None]
    blk_90 = [e['fwd_90'] for e in all_blocked if e['fwd_90'] is not None]

    on_30 = [e['fwd_30'] for e in all_on if e['fwd_30'] is not None]
    off_30 = [e['fwd_30'] for e in all_off if e['fwd_30'] is not None]
    blk_30 = [e['fwd_30'] for e in all_blocked if e['fwd_30'] is not None]

    on_180 = [e['fwd_180'] for e in all_on if e['fwd_180'] is not None]
    off_180 = [e['fwd_180'] for e in all_off if e['fwd_180'] is not None]
    blk_180 = [e['fwd_180'] for e in all_blocked if e['fwd_180'] is not None]

    on_mdd = [e['mdd_90'] for e in all_on]
    off_mdd = [e['mdd_90'] for e in all_off]
    blk_mdd = [e['mdd_90'] for e in all_blocked]

    hit_on = sum(1 for x in on_90 if x > 0) / len(on_90) * 100 if on_90 else 0
    hit_off = sum(1 for x in off_90 if x > 0) / len(off_90) * 100 if off_90 else 0
    hit_blk = sum(1 for x in blk_90 if x > 0) / len(blk_90) * 100 if blk_90 else 0

    print(f"""
  ┌────────────────────────────────────────────────────────────────────┐
  │  지표                DD Gate ON (3%)    DD Gate OFF    차단된 신호   │
  ├────────────────────────────────────────────────────────────────────┤
  │  총 신호 수          {len(all_on):>8}개        {len(all_off):>8}개    {len(all_blocked):>8}개   │
  │  90d 평가 가능       {len(on_90):>8}개        {len(off_90):>8}개    {len(blk_90):>8}개   │
  │                                                                    │
  │  적중률 30일         {sum(1 for x in on_30 if x>0)/len(on_30)*100 if on_30 else 0:>8.1f}%       {sum(1 for x in off_30 if x>0)/len(off_30)*100 if off_30 else 0:>8.1f}%   {sum(1 for x in blk_30 if x>0)/len(blk_30)*100 if blk_30 else 0:>8.1f}%   │
  │  적중률 90일         {hit_on:>8.1f}%       {hit_off:>8.1f}%   {hit_blk:>8.1f}%   │
  │                                                                    │
  │  평균 Fwd 30일       {np.mean(on_30) if on_30 else 0:>+8.1f}%       {np.mean(off_30) if off_30 else 0:>+8.1f}%   {np.mean(blk_30) if blk_30 else 0:>+8.1f}%   │
  │  평균 Fwd 90일       {np.mean(on_90) if on_90 else 0:>+8.1f}%       {np.mean(off_90) if off_90 else 0:>+8.1f}%   {np.mean(blk_90) if blk_90 else 0:>+8.1f}%   │
  │  중앙값 Fwd 90일     {np.median(on_90) if on_90 else 0:>+8.1f}%       {np.median(off_90) if off_90 else 0:>+8.1f}%   {np.median(blk_90) if blk_90 else 0:>+8.1f}%   │
  │  평균 Fwd 180일      {np.mean(on_180) if on_180 else 0:>+8.1f}%       {np.mean(off_180) if off_180 else 0:>+8.1f}%   {np.mean(blk_180) if blk_180 else 0:>+8.1f}%   │
  │                                                                    │
  │  평균 MDD (90일)     {np.mean(on_mdd) if on_mdd else 0:>+8.1f}%       {np.mean(off_mdd) if off_mdd else 0:>+8.1f}%   {np.mean(blk_mdd) if blk_mdd else 0:>+8.1f}%   │
  │  최악 MDD (90일)     {min(on_mdd) if on_mdd else 0:>+8.1f}%       {min(off_mdd) if off_mdd else 0:>+8.1f}%   {min(blk_mdd) if blk_mdd else 0:>+8.1f}%   │
  └────────────────────────────────────────────────────────────────────┘""")

    # ═══════════════════════════════════════════════════════
    # 4. 승/패 분포
    # ═══════════════════════════════════════════════════════
    print_section("3. 승/패 분포 (Fwd 90일)")

    def print_win_loss(label, fwd_list):
        if not fwd_list:
            print(f"  {label}: 데이터 없음")
            return
        wins = [x for x in fwd_list if x > 0]
        losses = [x for x in fwd_list if x <= 0]
        big_wins = [x for x in fwd_list if x > 30]
        big_losses = [x for x in fwd_list if x < -20]

        print(f"\n  [{label}] ({len(fwd_list)}건)")
        print(f"    승: {len(wins)}건 ({len(wins)/len(fwd_list)*100:.1f}%)"
              f"  |  패: {len(losses)}건 ({len(losses)/len(fwd_list)*100:.1f}%)")
        if wins:
            print(f"    승 평균: +{np.mean(wins):.1f}%  |  승 중앙: +{np.median(wins):.1f}%")
        if losses:
            print(f"    패 평균: {np.mean(losses):.1f}%  |  패 중앙: {np.median(losses):.1f}%")
        print(f"    대박(>+30%): {len(big_wins)}건  |  대손(<-20%): {len(big_losses)}건")

        # 수익률 구간 분포
        bins = [(-999, -30), (-30, -20), (-20, -10), (-10, 0),
                (0, 10), (10, 20), (20, 30), (30, 50), (50, 100), (100, 999)]
        bin_labels = ['<-30%', '-30~-20', '-20~-10', '-10~0',
                      '0~+10', '+10~+20', '+20~+30', '+30~+50', '+50~+100', '>+100%']
        counts = []
        for lo, hi in bins:
            c = sum(1 for x in fwd_list if lo <= x < hi)
            counts.append(c)

        print(f"    분포:")
        max_c = max(counts) if counts else 1
        for i, (label_b, c) in enumerate(zip(bin_labels, counts)):
            bar = '█' * int(c / max_c * 20) if max_c > 0 else ''
            pct = c / len(fwd_list) * 100
            print(f"      {label_b:>10}: {c:>3}건 ({pct:>5.1f}%) {bar}")

    print_win_loss("DD Gate ON (통과 신호)", on_90)
    print_win_loss("DD Gate OFF (전체 신호)", off_90)
    print_win_loss("차단된 신호 (ON에 없고 OFF에만 있는)", blk_90)

    # ═══════════════════════════════════════════════════════
    # 5. DD% 분포 분석
    # ═══════════════════════════════════════════════════════
    print_section("4. 신호 시점 Drawdown(%) 분포")

    on_dd = [e['dd_pct'] for e in all_on]
    off_dd = [e['dd_pct'] for e in all_off]
    blk_dd = [e['dd_pct'] for e in all_blocked]

    print(f"\n  DD Gate ON 신호의 DD% (매수 시점 고점 대비 하락폭):")
    if on_dd:
        print(f"    min={min(on_dd):.1f}%, 25%={np.percentile(on_dd,25):.1f}%, "
              f"중앙={np.median(on_dd):.1f}%, 75%={np.percentile(on_dd,75):.1f}%, "
              f"max={max(on_dd):.1f}%")

    print(f"\n  차단된 신호의 DD% (이 신호들은 DD가 3% 미만):")
    if blk_dd:
        print(f"    min={min(blk_dd):.1f}%, 25%={np.percentile(blk_dd,25):.1f}%, "
              f"중앙={np.median(blk_dd):.1f}%, 75%={np.percentile(blk_dd,75):.1f}%, "
              f"max={max(blk_dd):.1f}%")

    # DD 구간별 수익률
    print(f"\n  DD% 구간별 평균 Fwd 90일 수익률 (전체 OFF 신호 기준):")
    dd_bins = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 20), (20, 100)]
    dd_labels = ['0~1%', '1~2%', '2~3%', '3~5%', '5~10%', '10~20%', '20%+']
    print(f"    {'DD구간':>8} {'신호수':>6} {'적중률':>7} {'Fwd90평균':>10} {'Fwd90중앙':>10} {'MDD평균':>9}")
    print(f"    " + "-" * 55)
    for (lo, hi), label in zip(dd_bins, dd_labels):
        sigs = [e for e in all_off if lo <= e['dd_pct'] < hi and e['fwd_90'] is not None]
        if not sigs:
            print(f"    {label:>8} {0:>6}   {'--':>6}  {'--':>9}  {'--':>9}  {'--':>8}")
            continue
        fwd = [e['fwd_90'] for e in sigs]
        mdd = [e['mdd_90'] for e in sigs]
        hit = sum(1 for x in fwd if x > 0) / len(fwd) * 100
        print(f"    {label:>8} {len(sigs):>6}   {hit:>5.1f}%  {np.mean(fwd):>+8.1f}%  "
              f"{np.median(fwd):>+8.1f}%  {np.mean(mdd):>+7.1f}%")

    # ═══════════════════════════════════════════════════════
    # 6. 연도별 분석
    # ═══════════════════════════════════════════════════════
    print_section("5. 연도별 분석")

    years_set = sorted(set(e['year'] for e in all_off))
    print(f"  {'연도':>6} │ {'ON신호':>6} {'OFF신호':>7} {'차단':>5} │ "
          f"{'적중ON':>7} {'적중OFF':>8} {'적중차단':>8} │ "
          f"{'Fwd90ON':>8} {'Fwd90OFF':>9}")
    print(f"  " + "-" * 85)

    for yr in years_set:
        yr_on = [e for e in all_on if e['year'] == yr and e['fwd_90'] is not None]
        yr_off = [e for e in all_off if e['year'] == yr and e['fwd_90'] is not None]
        yr_blk = [e for e in all_blocked if e['year'] == yr and e['fwd_90'] is not None]

        yr_on_all = [e for e in all_on if e['year'] == yr]
        yr_off_all = [e for e in all_off if e['year'] == yr]
        yr_blk_all = [e for e in all_blocked if e['year'] == yr]

        hit_on_y = sum(1 for e in yr_on if e['fwd_90'] > 0) / len(yr_on) * 100 if yr_on else 0
        hit_off_y = sum(1 for e in yr_off if e['fwd_90'] > 0) / len(yr_off) * 100 if yr_off else 0
        hit_blk_y = sum(1 for e in yr_blk if e['fwd_90'] > 0) / len(yr_blk) * 100 if yr_blk else 0

        fwd_on_y = np.mean([e['fwd_90'] for e in yr_on]) if yr_on else 0
        fwd_off_y = np.mean([e['fwd_90'] for e in yr_off]) if yr_off else 0

        print(f"  {yr:>6} │ {len(yr_on_all):>5}개 {len(yr_off_all):>6}개 {len(yr_blk_all):>4}건 │ "
              f"{hit_on_y:>6.1f}% {hit_off_y:>7.1f}% {hit_blk_y:>7.1f}% │ "
              f"{fwd_on_y:>+7.1f}% {fwd_off_y:>+8.1f}%")

    # ═══════════════════════════════════════════════════════
    # 7. 섹터별 분석
    # ═══════════════════════════════════════════════════════
    print_section("6. 섹터별 분석")

    sectors = sorted(set(TICKERS.values()))
    print(f"  {'섹터':<12} │ {'ON신호':>6} {'OFF신호':>7} {'차단':>5} │ "
          f"{'적중ON':>7} {'적중OFF':>8} │ {'Fwd90ON':>8} {'Fwd90OFF':>9}")
    print(f"  " + "-" * 75)

    for sec in sectors:
        sec_on = [e for e in all_on if e['sector'] == sec and e['fwd_90'] is not None]
        sec_off = [e for e in all_off if e['sector'] == sec and e['fwd_90'] is not None]
        sec_blk = [e for e in all_blocked if e['sector'] == sec]

        sec_on_all = [e for e in all_on if e['sector'] == sec]
        sec_off_all = [e for e in all_off if e['sector'] == sec]

        hit_on_s = sum(1 for e in sec_on if e['fwd_90'] > 0) / len(sec_on) * 100 if sec_on else 0
        hit_off_s = sum(1 for e in sec_off if e['fwd_90'] > 0) / len(sec_off) * 100 if sec_off else 0

        fwd_on_s = np.mean([e['fwd_90'] for e in sec_on]) if sec_on else 0
        fwd_off_s = np.mean([e['fwd_90'] for e in sec_off]) if sec_off else 0

        print(f"  {sec:<12} │ {len(sec_on_all):>5}개 {len(sec_off_all):>6}개 {len(sec_blk):>4}건 │ "
              f"{hit_on_s:>6.1f}% {hit_off_s:>7.1f}% │ "
              f"{fwd_on_s:>+7.1f}% {fwd_off_s:>+8.1f}%")

    # ═══════════════════════════════════════════════════════
    # 8. 차단된 신호 개별 목록
    # ═══════════════════════════════════════════════════════
    print_section("7. 차단된 신호 개별 목록 (DD Gate에 의해 차단)")

    if all_blocked:
        # 정렬: 날짜순
        sorted_blocked = sorted(all_blocked, key=lambda x: x['date'])
        print(f"  {'#':>3} {'Ticker':<8} {'날짜':>12} {'가격':>10} {'DD%':>6} {'Score':>7} │ "
              f"{'Fwd30':>7} {'Fwd90':>7} {'Fwd180':>8} {'MDD90':>7} {'판정':>6}")
        print(f"  " + "-" * 95)

        for i, e in enumerate(sorted_blocked, 1):
            fwd30_s = f"{e['fwd_30']:>+6.1f}%" if e['fwd_30'] is not None else "  N/A "
            fwd90_s = f"{e['fwd_90']:>+6.1f}%" if e['fwd_90'] is not None else "  N/A "
            fwd180_s = f"{e['fwd_180']:>+6.1f}%" if e['fwd_180'] is not None else "   N/A "
            mdd_s = f"{e['mdd_90']:>+6.1f}%" if e['mdd_90'] is not None else "  N/A "

            if e['fwd_90'] is not None:
                verdict = "WIN" if e['fwd_90'] > 0 else "LOSS"
            else:
                verdict = "N/A"

            print(f"  {i:>3} {e['ticker']:<8} {e['date'].strftime('%Y-%m-%d'):>12} "
                  f"${e['price']:>8.2f} {e['dd_pct']:>5.1f}% {e['peak_score']:>6.3f} │ "
                  f"{fwd30_s} {fwd90_s} {fwd180_s} {mdd_s} {verdict:>6}")
    else:
        print("  차단된 신호 없음")

    # ═══════════════════════════════════════════════════════
    # 9. 리스크 조정 지표
    # ═══════════════════════════════════════════════════════
    print_section("8. 리스크 조정 지표")

    def risk_metrics(label, fwd_list, mdd_list):
        if not fwd_list or len(fwd_list) < 3:
            print(f"  [{label}] 데이터 부족")
            return
        avg = np.mean(fwd_list)
        std = np.std(fwd_list)
        sharpe = avg / std if std > 0 else 0
        downside = [x for x in fwd_list if x < 0]
        down_std = np.std(downside) if len(downside) > 1 else 1
        sortino = avg / down_std if down_std > 0 else 0
        calmar = avg / abs(min(mdd_list)) if mdd_list and min(mdd_list) < 0 else 0
        profit_factor = (sum(x for x in fwd_list if x > 0) /
                        abs(sum(x for x in fwd_list if x < 0))) if sum(x for x in fwd_list if x < 0) != 0 else 999

        print(f"\n  [{label}] ({len(fwd_list)}건)")
        print(f"    평균 수익률:    {avg:>+8.2f}%")
        print(f"    표준편차:       {std:>8.2f}%")
        print(f"    Sharpe-like:    {sharpe:>8.3f}  (평균/표준편차)")
        print(f"    Sortino-like:   {sortino:>8.3f}  (평균/하방편차)")
        print(f"    Calmar-like:    {calmar:>8.3f}  (평균/최대MDD)")
        print(f"    Profit Factor:  {profit_factor:>8.2f}  (총이익/총손실)")

    risk_metrics("DD Gate ON", on_90, on_mdd)
    risk_metrics("DD Gate OFF", off_90, off_mdd)
    risk_metrics("차단된 신호", blk_90, blk_mdd)

    # ═══════════════════════════════════════════════════════
    # 10. 최종 결론
    # ═══════════════════════════════════════════════════════
    print_section("9. 최종 결론")

    diff_hit = hit_on - hit_off
    diff_fwd = np.mean(on_90) - np.mean(off_90) if on_90 and off_90 else 0
    diff_med = np.median(on_90) - np.median(off_90) if on_90 and off_90 else 0
    diff_mdd = np.mean(on_mdd) - np.mean(off_mdd) if on_mdd and off_mdd else 0

    print(f"""
  DD Gate ON vs OFF 차이:
    적중률 차이 (90d):     {diff_hit:>+6.1f}%p
    평균 Fwd 90d 차이:     {diff_fwd:>+6.1f}%p
    중앙값 Fwd 90d 차이:   {diff_med:>+6.1f}%p
    평균 MDD 차이:         {diff_mdd:>+6.1f}%p (양수 = ON이 MDD 작음)
    신호 수 차이:          {len(all_on) - len(all_off):>+6}개 (ON이 더 적음)
    차단 비율:             {len(all_blocked)/(len(all_blocked)+len(all_on))*100 if (len(all_blocked)+len(all_on)) > 0 else 0:.1f}% ({len(all_blocked)}건/{len(all_blocked)+len(all_on)}건)
""")

    # 종합 판정
    score_on = 0
    if diff_hit > 2: score_on += 2
    elif diff_hit > 0: score_on += 1
    elif diff_hit < -2: score_on -= 2
    else: score_on -= 1

    if diff_fwd > 3: score_on += 2
    elif diff_fwd > 0: score_on += 1
    elif diff_fwd < -3: score_on -= 2
    else: score_on -= 1

    if diff_med > 3: score_on += 2
    elif diff_med > 0: score_on += 1
    elif diff_med < -3: score_on -= 2
    else: score_on -= 1

    if diff_mdd > 0: score_on += 1  # ON이 MDD 적으면 +
    else: score_on -= 1

    print(f"  종합 스코어: {score_on:+d} (양수=ON 유지 권장, 음수=OFF 검토)")

    if score_on >= 3:
        print("  >>> DD Gate 유지 강력 권장 (ON이 모든 지표에서 우세)")
    elif score_on >= 1:
        print("  >>> DD Gate 유지 권장 (ON이 약간 우세)")
    elif score_on >= -1:
        print("  >>> DD Gate 효과 미미 (유지/제거 모두 합리적)")
    elif score_on >= -3:
        print("  >>> DD Gate 제거 검토 (OFF가 약간 우세)")
    else:
        print("  >>> DD Gate 제거 권장 (OFF가 모든 지표에서 우세)")

    blk_hit = hit_blk
    if blk_90:
        blk_fwd_avg = np.mean(blk_90)
        print(f"\n  차단된 신호 평가:")
        print(f"    차단된 {len(blk_90)}건 중 {sum(1 for x in blk_90 if x > 0)}건 수익 "
              f"({blk_hit:.1f}% 적중)")
        if blk_fwd_avg > np.mean(on_90):
            print(f"    차단된 신호 평균({blk_fwd_avg:+.1f}%) > 통과 신호 평균({np.mean(on_90):+.1f}%)")
            print(f"    → DD Gate가 좋은 기회를 놓치고 있을 수 있음")
        else:
            print(f"    차단된 신호 평균({blk_fwd_avg:+.1f}%) < 통과 신호 평균({np.mean(on_90):+.1f}%)")
            print(f"    → DD Gate가 저품질 신호를 정확히 걸러냄")

    print(f"\n{'=' * 100}")
    print("  Done.")
    print('=' * 100)


if __name__ == '__main__':
    main()
