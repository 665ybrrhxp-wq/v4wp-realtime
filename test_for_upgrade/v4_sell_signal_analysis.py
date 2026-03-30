"""
VN60+GEO-OP 매도 신호 품질 분석
================================
현재 알고리즘의 매도(top) 신호가 실제로 효과적인지 검증.

분석 항목:
1. 전체 매도 신호 수 및 필터 통과/차단 비교
2. 매도 신호 후 Forward Return (가격이 실제로 하락했는가?)
3. 매도 신호 vs 매수 신호 비교
4. 매도 신호 스코어 구간별 성과
5. 매도 안 했으면 어땠을까? (Holding vs Selling)
6. 티커별 매도 신호 상세
7. 최근 매도 신호 리스트
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
# Configuration — GEO-OP 프로덕션 파라미터
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
DIVGATE = 3
BUY_DD_LB = 20
BUY_DD_TH = 0.03
LATE_SELL_DROP_TH = 0.05
SELL_CONFIRM_DAYS = 3


def download_max(ticker):
    t = yf.Ticker(ticker)
    df = t.history(period='max', auto_adjust=True)
    if df is None or len(df) == 0:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.dropna(subset=['Close', 'Volume'])
    df = df[df['Close'] > 0]
    return df


def get_all_signals(df, score, tk):
    """모든 이벤트 추출 + 필터링 상태 태깅"""
    events = detect_signal_events(score, th=SIGNAL_TH, cooldown=COOLDOWN)
    pf = build_price_filter(df, er_q=ER_Q, atr_q=ATR_Q, lookback=LOOKBACK)

    close = df['Close'].values
    rolling_high_sell = pd.Series(close).rolling(20, min_periods=1).max().values
    rolling_high_buy = pd.Series(close).rolling(BUY_DD_LB, min_periods=1).max().values
    n = len(df)

    tagged = []
    for ev in events:
        pidx = ev['peak_idx']
        if pidx >= n:
            continue

        tag = {**ev}
        tag['price'] = close[pidx]
        tag['date'] = df.index[pidx].strftime('%Y-%m-%d')

        # Price filter
        pf_pass = pf(pidx)
        tag['pf_pass'] = pf_pass

        if ev['type'] == 'top':
            # LATE_SELL_BLOCK check
            rh = rolling_high_sell[pidx]
            drop_pct = (rh - close[pidx]) / rh if rh > 0 else 0
            tag['drop_pct'] = drop_pct
            tag['late_sell_blocked'] = drop_pct > LATE_SELL_DROP_TH
            tag['dd_blocked'] = False

            # Duration check
            dur = ev['end_idx'] - ev['start_idx'] + 1
            tag['duration'] = dur
            tag['confirmed'] = dur >= SELL_CONFIRM_DAYS

            # Final status
            if not pf_pass:
                tag['status'] = 'PF_BLOCKED'
            elif tag['late_sell_blocked']:
                tag['status'] = 'LATE_SELL_BLOCKED'
            elif not tag['confirmed']:
                tag['status'] = 'PENDING'
            else:
                tag['status'] = 'ACTIVE'

        elif ev['type'] == 'bottom':
            rh = rolling_high_buy[pidx]
            dd = (rh - close[pidx]) / rh if rh > 0 else 0
            tag['drop_pct'] = dd
            tag['late_sell_blocked'] = False
            tag['dd_blocked'] = dd < BUY_DD_TH

            dur = ev['end_idx'] - ev['start_idx'] + 1
            tag['duration'] = dur
            tag['confirmed'] = dur >= 1  # confirm_days=1 for buy

            if not pf_pass:
                tag['status'] = 'PF_BLOCKED'
            elif tag['dd_blocked']:
                tag['status'] = 'DD_BLOCKED'
            elif not tag['confirmed']:
                tag['status'] = 'PENDING'
            else:
                tag['status'] = 'ACTIVE'

        # Forward returns (매도 신호: 가격 하락이면 성공)
        for days in [7, 14, 30, 60, 90, 180]:
            fwd_idx = pidx + days
            if fwd_idx < n:
                tag[f'fwd_{days}d'] = (close[fwd_idx] / close[pidx] - 1) * 100
            else:
                tag[f'fwd_{days}d'] = None

        # MDD after signal (within 90d)
        end_i = min(pidx + 90, n)
        if end_i > pidx + 1:
            min_p = min(close[pidx:end_i])
            max_p = max(close[pidx:end_i])
            tag['mdd_90'] = (min_p / close[pidx] - 1) * 100
            tag['max_gain_90'] = (max_p / close[pidx] - 1) * 100
        else:
            tag['mdd_90'] = None
            tag['max_gain_90'] = None

        tagged.append(tag)

    return tagged


def main():
    sep = '=' * 130
    dash = '-' * 130

    print(sep)
    print("  VN60+GEO-OP 매도 신호 품질 분석")
    print(f"  Score: AND-GEO → 매도 = score < -{SIGNAL_TH} (both S_Force < 0 AND S_Div < 0)")
    print(f"  Pipeline: th={SIGNAL_TH}, PF(ER<{ER_Q}%/ATR>{ATR_Q}%), LATE_SELL_BLOCK={LATE_SELL_DROP_TH*100:.0f}%, "
          f"sell_confirm={SELL_CONFIRM_DAYS}d")
    print(sep)

    # ─── 데이터 로딩 ───
    print("\n  데이터 로딩...")
    data = {}
    for tk in TICKERS:
        df = download_max(tk)
        if df is None or len(df) < 300:
            print(f"    {tk}: SKIP")
            continue
        df = smooth_earnings_volume(df, ticker=tk)
        data[tk] = df
        print(f"    {tk}: {len(df)} bars ({len(df)/252:.1f}yr)")
    print(f"  {len(data)} tickers loaded.\n")

    # ─── 신호 추출 ───
    all_sells = {}  # ticker -> list of sell events
    all_buys = {}
    all_events_raw = {}

    for tk, df in data.items():
        score = calc_v4_score(df, w=20, divgate_days=DIVGATE)
        subind = calc_v4_subindicators(df, w=20, divgate_days=DIVGATE)
        tagged = get_all_signals(df, score, tk)

        sells = [e for e in tagged if e['type'] == 'top']
        buys = [e for e in tagged if e['type'] == 'bottom']
        all_sells[tk] = sells
        all_buys[tk] = buys
        all_events_raw[tk] = {'subind': subind, 'score': score}

    # ═════════════════════════════════════════════════════════
    # [1] 매도 신호 총량 및 필터 분석
    # ═════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  [1] 매도 신호 총량 및 필터 분석")
    print(sep)

    total_sell = 0; active_sell = 0; pf_blocked = 0; late_blocked = 0; pending_sell = 0
    total_buy = 0; active_buy = 0

    print(f"\n  {'Ticker':<8} │ {'총 매도':>7} {'ACTIVE':>7} {'PF차단':>7} {'LATE차단':>8} {'PENDING':>8} │ "
          f"{'총 매수':>7} {'ACTIVE':>7} │ {'매도/매수 비율':>14}")
    print(f"  {dash}")

    for tk in TICKERS:
        if tk not in all_sells:
            continue
        sells = all_sells[tk]
        buys = all_buys[tk]

        s_total = len(sells)
        s_active = sum(1 for e in sells if e['status'] == 'ACTIVE')
        s_pf = sum(1 for e in sells if e['status'] == 'PF_BLOCKED')
        s_late = sum(1 for e in sells if e['status'] == 'LATE_SELL_BLOCKED')
        s_pend = sum(1 for e in sells if e['status'] == 'PENDING')

        b_total = len(buys)
        b_active = sum(1 for e in buys if e['status'] == 'ACTIVE')

        ratio = f"{s_active}/{b_active}" if b_active > 0 else "N/A"

        total_sell += s_total; active_sell += s_active
        pf_blocked += s_pf; late_blocked += s_late; pending_sell += s_pend
        total_buy += b_total; active_buy += b_active

        print(f"  {tk:<8} │ {s_total:>7} {s_active:>7} {s_pf:>7} {s_late:>8} {s_pend:>8} │ "
              f"{b_total:>7} {b_active:>7} │ {ratio:>14}")

    print(f"  {dash}")
    ratio_total = f"{active_sell}/{active_buy}" if active_buy > 0 else "N/A"
    print(f"  {'TOTAL':<8} │ {total_sell:>7} {active_sell:>7} {pf_blocked:>7} {late_blocked:>8} {pending_sell:>8} │ "
          f"{total_buy:>7} {active_buy:>7} │ {ratio_total:>14}")

    pct_active = active_sell / total_sell * 100 if total_sell > 0 else 0
    pct_pf = pf_blocked / total_sell * 100 if total_sell > 0 else 0
    pct_late = late_blocked / total_sell * 100 if total_sell > 0 else 0
    pct_pend = pending_sell / total_sell * 100 if total_sell > 0 else 0

    print(f"\n  매도 신호 비율: ACTIVE {pct_active:.1f}% | PF차단 {pct_pf:.1f}% | LATE차단 {pct_late:.1f}% | PENDING {pct_pend:.1f}%")

    # ═════════════════════════════════════════════════════════
    # [2] ACTIVE 매도 신호 Forward Return 분석
    # ═════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  [2] ACTIVE 매도 신호: 가격이 실제로 하락했는가?")
    print(f"      (매도 후 가격이 내린 = 매도 성공, 올랐으면 = 매도 실패)")
    print(sep)

    # Collect all active sells
    active_sells_all = []
    for tk in TICKERS:
        if tk not in all_sells:
            continue
        for e in all_sells[tk]:
            if e['status'] == 'ACTIVE':
                e['ticker'] = tk
                active_sells_all.append(e)

    # Forward return analysis
    periods = [7, 14, 30, 60, 90, 180]
    print(f"\n  {'기간':<8} │ {'N':>5} {'하락비율':>8} {'평균변동':>9} {'중앙값':>9} {'최소(최대하락)':>14} {'최대(최대상승)':>14}")
    print(f"  {'-'*80}")

    for days in periods:
        vals = [e[f'fwd_{days}d'] for e in active_sells_all if e[f'fwd_{days}d'] is not None]
        if not vals:
            continue
        n_down = sum(1 for v in vals if v < 0)
        down_rate = n_down / len(vals) * 100
        avg_v = np.mean(vals)
        med_v = np.median(vals)
        min_v = min(vals)
        max_v = max(vals)
        print(f"  {days:>3}일    │ {len(vals):>5} {down_rate:>7.1f}% {avg_v:>+8.1f}% {med_v:>+8.1f}% {min_v:>+13.1f}% {max_v:>+13.1f}%")

    # 매도 성공 기준: 30일 내 -5% 이상 하락
    sells_30 = [e for e in active_sells_all if e['fwd_30d'] is not None]
    if sells_30:
        drop_5pct = sum(1 for e in sells_30 if e['fwd_30d'] < -5)
        drop_10pct = sum(1 for e in sells_30 if e['fwd_30d'] < -10)
        rise_10pct = sum(1 for e in sells_30 if e['fwd_30d'] > 10)
        print(f"\n  30일 후 성과 분석 (N={len(sells_30)}):")
        print(f"    5% 이상 하락: {drop_5pct}건 ({drop_5pct/len(sells_30)*100:.1f}%) — 매도 정확")
        print(f"    10% 이상 하락: {drop_10pct}건 ({drop_10pct/len(sells_30)*100:.1f}%) — 매도 매우 정확")
        print(f"    10% 이상 상승: {rise_10pct}건 ({rise_10pct/len(sells_30)*100:.1f}%) — 매도 실패 (기회 상실)")

    # MDD analysis: 매도 후 90일 내 최대 낙폭
    sells_mdd = [e for e in active_sells_all if e['mdd_90'] is not None]
    if sells_mdd:
        mdds = [e['mdd_90'] for e in sells_mdd]
        max_gains = [e['max_gain_90'] for e in sells_mdd]
        print(f"\n  매도 후 90일 내 MDD / MaxGain:")
        print(f"    MDD 평균: {np.mean(mdds):+.1f}%   중앙값: {np.median(mdds):+.1f}%   최악: {min(mdds):+.1f}%")
        print(f"    MaxGain 평균: {np.mean(max_gains):+.1f}%   중앙값: {np.median(max_gains):+.1f}%   최고: {max(max_gains):+.1f}%")
        # 매도 안 했으면 더 좋았을 비율 (max_gain > |mdd|)
        better_hold = sum(1 for e in sells_mdd if e['max_gain_90'] > abs(e['mdd_90']))
        print(f"    '매도 안 했으면 더 좋았을' 비율 (MaxGain > |MDD|): {better_hold}/{len(sells_mdd)} = {better_hold/len(sells_mdd)*100:.1f}%")

    # ═════════════════════════════════════════════════════════
    # [3] 차단된 매도 vs 통과된 매도 비교
    # ═════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  [3] 차단된 매도 vs ACTIVE 매도 — 차단이 올바른 결정이었는가?")
    print(sep)

    for status_name, status_code in [('ACTIVE', 'ACTIVE'), ('LATE_SELL_BLOCKED', 'LATE_SELL_BLOCKED'),
                                      ('PF_BLOCKED', 'PF_BLOCKED'), ('PENDING', 'PENDING')]:
        group = []
        for tk in all_sells:
            for e in all_sells[tk]:
                if e['status'] == status_code:
                    group.append(e)

        if not group:
            continue

        fwd30 = [e['fwd_30d'] for e in group if e['fwd_30d'] is not None]
        fwd90 = [e['fwd_90d'] for e in group if e['fwd_90d'] is not None]

        if fwd30:
            down30 = sum(1 for v in fwd30 if v < 0) / len(fwd30) * 100
        else:
            down30 = 0

        if fwd90:
            down90 = sum(1 for v in fwd90 if v < 0) / len(fwd90) * 100
        else:
            down90 = 0

        print(f"\n  [{status_name}] — {len(group)}건")
        if fwd30:
            print(f"    30일 후: 하락비율={down30:.1f}%  평균={np.mean(fwd30):+.1f}%  중앙값={np.median(fwd30):+.1f}%")
        if fwd90:
            print(f"    90일 후: 하락비율={down90:.1f}%  평균={np.mean(fwd90):+.1f}%  중앙값={np.median(fwd90):+.1f}%")

        # LATE_SELL_BLOCKED인 경우: 이미 하락한 후라 매도 차단 → 실제로 반등했는가?
        if status_code == 'LATE_SELL_BLOCKED' and fwd30:
            rebound = sum(1 for v in fwd30 if v > 0) / len(fwd30) * 100
            print(f"    → 이미 하락 후 차단됨. 30일 후 반등 비율: {rebound:.1f}% → {'차단 정당' if rebound > 50 else '차단 부적절'}")

    # ═════════════════════════════════════════════════════════
    # [4] 매도 vs 매수 신호 비교
    # ═════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  [4] 매도 신호 vs 매수 신호 — 예측력 비교")
    print(sep)

    buy_actives = []
    for tk in all_buys:
        for e in all_buys[tk]:
            if e['status'] == 'ACTIVE':
                e['ticker'] = tk
                buy_actives.append(e)

    print(f"\n  {'':15} │ {'매수 (ACTIVE)':>40} │ {'매도 (ACTIVE)':>40}")
    print(f"  {'':15} │ {'N':>6} {'정확률':>8} {'평균':>8} {'중앙값':>8} │ {'N':>6} {'정확률':>8} {'평균':>8} {'중앙값':>8}")
    print(f"  {'-'*100}")

    for days in periods:
        buy_vals = [e[f'fwd_{days}d'] for e in buy_actives if e[f'fwd_{days}d'] is not None]
        sell_vals = [e[f'fwd_{days}d'] for e in active_sells_all if e[f'fwd_{days}d'] is not None]

        if buy_vals:
            b_n = len(buy_vals)
            b_hit = sum(1 for v in buy_vals if v > 0) / b_n * 100  # 매수 정확 = 가격 상승
            b_avg = np.mean(buy_vals)
            b_med = np.median(buy_vals)
        else:
            b_n = 0; b_hit = 0; b_avg = 0; b_med = 0

        if sell_vals:
            s_n = len(sell_vals)
            s_hit = sum(1 for v in sell_vals if v < 0) / s_n * 100  # 매도 정확 = 가격 하락
            s_avg = np.mean(sell_vals)
            s_med = np.median(sell_vals)
        else:
            s_n = 0; s_hit = 0; s_avg = 0; s_med = 0

        label = f"  {days:>3}일 후"
        print(f"  {label:<15} │ {b_n:>6} {b_hit:>7.1f}% {b_avg:>+7.1f}% {b_med:>+7.1f}% │ "
              f"{s_n:>6} {s_hit:>7.1f}% {s_avg:>+7.1f}% {s_med:>+7.1f}%")

    # ═════════════════════════════════════════════════════════
    # [5] 매도 스코어 구간별 성과
    # ═════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  [5] 매도 신호 스코어(절대값) 구간별 성과")
    print(sep)

    # Score bins for sell signals (absolute value)
    bins = [(0.05, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.50), (0.50, 1.00)]
    print(f"\n  {'Score |값|':<15} │ {'N':>5} {'하락30d':>8} {'하락90d':>8} │ {'Avg30d':>8} {'Avg90d':>8} {'Med90d':>8} │ {'AvgForce':>9} {'AvgDiv':>9}")
    print(f"  {'-'*110}")

    for lo, hi in bins:
        group = [e for e in active_sells_all if lo <= abs(e['peak_val']) < hi]
        if not group:
            continue

        fwd30 = [e['fwd_30d'] for e in group if e['fwd_30d'] is not None]
        fwd90 = [e['fwd_90d'] for e in group if e['fwd_90d'] is not None]

        d30 = sum(1 for v in fwd30 if v < 0) / len(fwd30) * 100 if fwd30 else 0
        d90 = sum(1 for v in fwd90 if v < 0) / len(fwd90) * 100 if fwd90 else 0

        avg30 = np.mean(fwd30) if fwd30 else 0
        avg90 = np.mean(fwd90) if fwd90 else 0
        med90 = np.median(fwd90) if fwd90 else 0

        # Avg force/div at signal
        forces = [abs(all_events_raw[e['ticker']]['subind']['s_force'].iloc[e['peak_idx']])
                   for e in group if e['ticker'] in all_events_raw and e['peak_idx'] < len(all_events_raw[e['ticker']]['subind'])]
        divs = [abs(all_events_raw[e['ticker']]['subind']['s_div'].iloc[e['peak_idx']])
                 for e in group if e['ticker'] in all_events_raw and e['peak_idx'] < len(all_events_raw[e['ticker']]['subind'])]

        avg_f = np.mean(forces) if forces else 0
        avg_d = np.mean(divs) if divs else 0

        print(f"  {lo:.2f}-{hi:.2f}      │ {len(group):>5} {d30:>7.1f}% {d90:>7.1f}% │ {avg30:>+7.1f}% {avg90:>+7.1f}% {med90:>+7.1f}% │ {avg_f:>+8.3f} {avg_d:>+8.3f}")

    # ═════════════════════════════════════════════════════════
    # [6] 티커별 매도 신호 성과
    # ═════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  [6] 티커별 매도 신호 성과 (ACTIVE만)")
    print(sep)

    print(f"\n  {'Ticker':<8} {'Sect':<10} │ {'총매도':>6} {'ACTIVE':>7} │ {'하락30d':>8} {'하락90d':>8} │ "
          f"{'Avg30d':>8} {'Avg90d':>8} │ {'Avg MDD90':>10} {'AvgScore':>9}")
    print(f"  {dash}")

    for tk in TICKERS:
        if tk not in all_sells:
            continue
        sells = all_sells[tk]
        active = [e for e in sells if e['status'] == 'ACTIVE']

        if not active:
            print(f"  {tk:<8} {TICKERS[tk]:<10} │ {len(sells):>6} {0:>7} │ {'N/A':>8} {'N/A':>8} │ {'N/A':>8} {'N/A':>8} │")
            continue

        fwd30 = [e['fwd_30d'] for e in active if e['fwd_30d'] is not None]
        fwd90 = [e['fwd_90d'] for e in active if e['fwd_90d'] is not None]
        mdds = [e['mdd_90'] for e in active if e['mdd_90'] is not None]
        scores = [abs(e['peak_val']) for e in active]

        d30 = sum(1 for v in fwd30 if v < 0) / len(fwd30) * 100 if fwd30 else 0
        d90 = sum(1 for v in fwd90 if v < 0) / len(fwd90) * 100 if fwd90 else 0
        avg30 = np.mean(fwd30) if fwd30 else 0
        avg90 = np.mean(fwd90) if fwd90 else 0
        avg_mdd = np.mean(mdds) if mdds else 0
        avg_sc = np.mean(scores) if scores else 0

        print(f"  {tk:<8} {TICKERS[tk]:<10} │ {len(sells):>6} {len(active):>7} │ {d30:>7.1f}% {d90:>7.1f}% │ "
              f"{avg30:>+7.1f}% {avg90:>+7.1f}% │ {avg_mdd:>+9.1f}% {avg_sc:>8.3f}")

    # ═════════════════════════════════════════════════════════
    # [7] 개별 매도 신호 상세 리스트 (ACTIVE)
    # ═════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  [7] 개별 매도 신호 상세 리스트 (ACTIVE)")
    print(sep)

    for tk in TICKERS:
        if tk not in all_sells:
            continue
        active = [e for e in all_sells[tk] if e['status'] == 'ACTIVE']
        if not active:
            continue

        subind = all_events_raw[tk]['subind']

        print(f"\n  [{tk}] ({TICKERS[tk]}) ACTIVE 매도 신호 — {len(active)}건")
        print(f"    {'#':>4} {'Date':>12} {'Price':>10} │ {'|Score|':>7} {'Force':>7} {'Div':>7} {'Dur':>4} │ "
              f"{'7d':>7} {'14d':>7} {'30d':>7} {'60d':>7} {'90d':>7} {'180d':>7} │ {'MDD90':>7} │ {'판정'}")
        print(f"    {'-'*130}")

        n_correct = 0; n_total_90 = 0
        for i, e in enumerate(active, 1):
            pidx = e['peak_idx']
            sc = abs(e['peak_val'])

            if pidx < len(subind):
                sf = subind['s_force'].iloc[pidx]
                sd = subind['s_div'].iloc[pidx]
            else:
                sf = sd = 0

            fwd_str = []
            for days in periods:
                v = e[f'fwd_{days}d']
                if v is not None:
                    fwd_str.append(f"{v:>+6.1f}%")
                else:
                    fwd_str.append(f"{'N/A':>7}")

            mdd = f"{e['mdd_90']:>+6.1f}%" if e['mdd_90'] is not None else f"{'N/A':>7}"

            # 판정: 매도 후 90일 내 가격 변동 기준
            fwd90 = e['fwd_90d']
            if fwd90 is not None:
                n_total_90 += 1
                if fwd90 < -15:
                    result = "★★ GREAT SELL"
                    n_correct += 1
                elif fwd90 < -5:
                    result = "★ good sell"
                    n_correct += 1
                elif fwd90 < 0:
                    result = "ok sell"
                    n_correct += 1
                elif fwd90 < 10:
                    result = "miss (small)"
                elif fwd90 < 30:
                    result = "MISS"
                else:
                    result = "★★ BIG MISS"
            else:
                result = ""

            print(f"    {i:>4} {e['date']:>12} ${e['price']:>8.2f} │ {sc:>6.3f} {sf:>+6.3f} {sd:>+6.3f} {e['duration']:>4} │ "
                  f"{' '.join(fwd_str)} │ {mdd} │ {result}")

        if n_total_90 > 0:
            print(f"    {'-'*130}")
            print(f"    매도 정확률 (90일 후 하락): {n_correct}/{n_total_90} = {n_correct/n_total_90*100:.1f}%")

    # ═════════════════════════════════════════════════════════
    # [8] 차단된 매도 (LATE_SELL_BLOCKED) 상세
    # ═════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  [8] 차단된 매도 (LATE_SELL_BLOCKED) 상세 — 차단이 올바른 결정이었는가?")
    print(sep)

    for tk in TICKERS:
        if tk not in all_sells:
            continue
        blocked = [e for e in all_sells[tk] if e['status'] == 'LATE_SELL_BLOCKED']
        if not blocked:
            continue

        print(f"\n  [{tk}] LATE_SELL_BLOCKED — {len(blocked)}건")
        print(f"    {'#':>4} {'Date':>12} {'Price':>10} {'Drop%':>7} │ {'30d':>7} {'90d':>7} {'180d':>7} │ {'판정'}")
        print(f"    {'-'*85}")

        for i, e in enumerate(blocked, 1):
            fwd30 = f"{e['fwd_30d']:>+6.1f}%" if e['fwd_30d'] is not None else f"{'N/A':>7}"
            fwd90 = f"{e['fwd_90d']:>+6.1f}%" if e['fwd_90d'] is not None else f"{'N/A':>7}"
            fwd180 = f"{e['fwd_180d']:>+6.1f}%" if e['fwd_180d'] is not None else f"{'N/A':>7}"

            # 판정: 차단 후 실제로 반등했으면 차단 정당
            if e['fwd_90d'] is not None:
                if e['fwd_90d'] > 10:
                    result = "★ 차단 정당 (반등)"
                elif e['fwd_90d'] > 0:
                    result = "차단 적절 (소폭 반등)"
                elif e['fwd_90d'] > -10:
                    result = "차단 중립"
                else:
                    result = "차단 실패 (계속 하락)"
            else:
                result = ""

            print(f"    {i:>4} {e['date']:>12} ${e['price']:>8.2f} {e['drop_pct']*100:>6.1f}% │ {fwd30} {fwd90} {fwd180} │ {result}")

    # ═════════════════════════════════════════════════════════
    # [9] 매도 Duration 분석
    # ═════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  [9] 매도 신호 Duration(지속일수)별 성과")
    print(sep)

    dur_bins = [(1, 2), (3, 5), (6, 10), (11, 20), (21, 999)]
    dur_labels = ['1-2일', '3-5일', '6-10일', '11-20일', '21일+']

    print(f"\n  {'Duration':<10} │ {'N':>5} {'하락30d':>8} {'하락90d':>8} │ {'Avg30d':>8} {'Avg90d':>8} {'Med90d':>8} │ {'AvgScore':>9}")
    print(f"  {'-'*90}")

    # Use ALL sell signals (not just active) for duration analysis
    all_sell_flat = []
    for tk in all_sells:
        for e in all_sells[tk]:
            e['ticker'] = tk
            all_sell_flat.append(e)

    for (lo, hi), label in zip(dur_bins, dur_labels):
        group = [e for e in all_sell_flat if lo <= e['duration'] <= hi]
        if not group:
            continue

        fwd30 = [e['fwd_30d'] for e in group if e['fwd_30d'] is not None]
        fwd90 = [e['fwd_90d'] for e in group if e['fwd_90d'] is not None]

        d30 = sum(1 for v in fwd30 if v < 0) / len(fwd30) * 100 if fwd30 else 0
        d90 = sum(1 for v in fwd90 if v < 0) / len(fwd90) * 100 if fwd90 else 0
        avg30 = np.mean(fwd30) if fwd30 else 0
        avg90 = np.mean(fwd90) if fwd90 else 0
        med90 = np.median(fwd90) if fwd90 else 0
        avg_sc = np.mean([abs(e['peak_val']) for e in group])

        print(f"  {label:<10} │ {len(group):>5} {d30:>7.1f}% {d90:>7.1f}% │ {avg30:>+7.1f}% {avg90:>+7.1f}% {med90:>+7.1f}% │ {avg_sc:>8.3f}")

    # ═════════════════════════════════════════════════════════
    # [10] 최근 매도 신호 (2024~)
    # ═════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  [10] 2024년 이후 매도 신호 (전체 — 상태별)")
    print(sep)

    recent_sells = []
    for tk in TICKERS:
        if tk not in all_sells:
            continue
        for e in all_sells[tk]:
            if e['date'] >= '2024-01-01':
                e['ticker'] = tk
                recent_sells.append(e)

    recent_sells.sort(key=lambda x: x['date'])

    print(f"\n  {'#':>4} {'Ticker':>8} {'Date':>12} {'Price':>10} {'Status':>18} │ "
          f"{'|Score|':>7} {'Dur':>4} {'Drop%':>7} │ {'30d':>7} {'90d':>7} {'180d':>7} │ {'판정'}")
    print(f"  {'-'*135}")

    for i, e in enumerate(recent_sells, 1):
        sc = abs(e['peak_val'])
        fwd30 = f"{e['fwd_30d']:>+6.1f}%" if e['fwd_30d'] is not None else f"{'N/A':>7}"
        fwd90 = f"{e['fwd_90d']:>+6.1f}%" if e['fwd_90d'] is not None else f"{'N/A':>7}"
        fwd180 = f"{e['fwd_180d']:>+6.1f}%" if e['fwd_180d'] is not None else f"{'N/A':>7}"

        if e['fwd_90d'] is not None:
            if e['status'] == 'ACTIVE':
                if e['fwd_90d'] < -5:
                    result = "★ 매도 성공"
                elif e['fwd_90d'] < 0:
                    result = "매도 ok"
                elif e['fwd_90d'] > 20:
                    result = "★ 매도 실패!"
                else:
                    result = "매도 miss"
            elif e['status'] == 'LATE_SELL_BLOCKED':
                if e['fwd_90d'] > 0:
                    result = "차단 정당"
                else:
                    result = "차단 실패"
            else:
                result = ""
        else:
            result = ""

        print(f"  {i:>4} {e['ticker']:>8} {e['date']:>12} ${e['price']:>8.2f} {e['status']:>18} │ "
              f"{sc:>6.3f} {e['duration']:>4} {e['drop_pct']*100:>6.1f}% │ {fwd30} {fwd90} {fwd180} │ {result}")

    # ═════════════════════════════════════════════════════════
    # [11] GRAND SUMMARY
    # ═════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  GRAND SUMMARY: 매도 신호 품질 평가")
    print(sep)

    # Collect key stats
    a_30 = [e['fwd_30d'] for e in active_sells_all if e['fwd_30d'] is not None]
    a_90 = [e['fwd_90d'] for e in active_sells_all if e['fwd_90d'] is not None]

    if a_30:
        sell_acc_30 = sum(1 for v in a_30 if v < 0) / len(a_30) * 100
    else:
        sell_acc_30 = 0
    if a_90:
        sell_acc_90 = sum(1 for v in a_90 if v < 0) / len(a_90) * 100
    else:
        sell_acc_90 = 0

    # Buy accuracy for comparison
    b_90 = [e['fwd_90d'] for e in buy_actives if e['fwd_90d'] is not None]
    if b_90:
        buy_acc_90 = sum(1 for v in b_90 if v > 0) / len(b_90) * 100
    else:
        buy_acc_90 = 0

    avg_sell_30 = np.mean(a_30) if a_30 else 0
    avg_sell_90 = np.mean(a_90) if a_90 else 0

    print(f"""
  +{'─'*75}+
  │  매도 신호 품질 종합 평가                                                 │
  +{'─'*75}+
  │                                                                           │
  │  ACTIVE 매도 신호 수:     {active_sell:>6}건 (전체 {total_sell}건 중 {pct_active:.0f}%){'':>16}│
  │  ACTIVE 매수 신호 수:     {active_buy:>6}건 (비교용){'':>30}│
  │                                                                           │
  │  매도 정확률 (30일):      {sell_acc_30:>6.1f}% (가격 하락 비율){'':>23}│
  │  매도 정확률 (90일):      {sell_acc_90:>6.1f}%{'':>40}│
  │  매수 정확률 (90일):      {buy_acc_90:>6.1f}% (비교: 가격 상승 비율){'':>15}│
  │                                                                           │
  │  매도 후 평균 변동 (30일): {avg_sell_30:>+6.1f}%{'':>39}│
  │  매도 후 평균 변동 (90일): {avg_sell_90:>+6.1f}%{'':>39}│
  │                                                                           │""")

    if sell_acc_90 > 55:
        verdict = "매도 신호 유효 — 알고리즘 활용 가능"
    elif sell_acc_90 > 45:
        verdict = "매도 신호 약간 유효 — 보조 지표로만 활용"
    else:
        verdict = "매도 신호 무효 — 랜덤 수준, 매도 알고리즘 재설계 필요"

    print(f"  │  결론: {verdict:<64}│")
    print(f"  +{'─'*75}+\n")

    print("  Done.\n")


if __name__ == '__main__':
    main()
