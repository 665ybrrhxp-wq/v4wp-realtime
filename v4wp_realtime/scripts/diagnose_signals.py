#!/usr/bin/env python3
"""
V4_wP 신호 진단 스크립트
========================
SPY & QQQ에서 V4_wP 시그널의 품질을 종합 분석하여
BAD 시그널, MISSED 기회, 시스템 약점을 진단한다.

실행: python diagnose_signals.py
"""

import sys
import io
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# UTF-8 출력
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── 프로젝트 루트 설정 ──
_project_root = r'C:\Users\user\OneDrive\Desktop\거래량 백테스트'
sys.path.insert(0, _project_root)

from pathlib import Path
CACHE_DIR = str(Path(_project_root) / 'cache')

# ── real_market_backtest 에서 필요한 함수 import ──
from real_market_backtest import (
    download_data,
    calc_v4_score,
    calc_v4_subindicators,
    detect_signal_events,
    build_price_filter,
    calc_efficiency_ratio,
    calc_atr_percentile,
    calc_pv_divergence,
    calc_pv_concordance,
    calc_pv_force_macd,
)

# ══════════════════════════════════════════════════════════════
# 파라미터
# ══════════════════════════════════════════════════════════════
V4_WINDOW      = 20
SIGNAL_THRESHOLD = 0.15
COOLDOWN       = 5
ER_QUANTILE    = 66
ATR_QUANTILE   = 66
LOOKBACK       = 252

DATA_START     = '2007-01-01'
DATA_END       = '2026-03-31'

TICKERS = ['SPY', 'QQQ']

# Forward-return 평가 기준
GOOD_BUY_THRESH   =  5.0   # 60일 내 +5%
BAD_BUY_THRESH    = -5.0   # 20일 내 -5%
GOOD_SELL_THRESH  = -5.0   # 60일 내 -5%
BAD_SELL_THRESH   = 10.0   # 60일 내 +10%
MISSED_DROP_THRESH = 15.0  # 고점 대비 15% 하락
MISSED_RECOVERY    = 15.0  # 저점 대비 15% 회복

HORIZONS = [5, 10, 20, 60, 120]


# ══════════════════════════════════════════════════════════════
# 1. 데이터 로드 및 신호 생성
# ══════════════════════════════════════════════════════════════

def load_and_compute(ticker):
    """데이터 로드, V4 스코어/서브지표, 가격필터 계산"""
    df = download_data(ticker, start=DATA_START, end=DATA_END, cache_dir=CACHE_DIR)
    v4_score = calc_v4_score(df, w=V4_WINDOW)
    v4_sub   = calc_v4_subindicators(df, w=V4_WINDOW)
    pf       = build_price_filter(df, er_q=ER_QUANTILE, atr_q=ATR_QUANTILE, lookback=LOOKBACK)

    # ER / ATR 원시값 (진단용)
    er_vals  = calc_efficiency_ratio(df, 20).values
    atr_vals = calc_atr_percentile(df, 14, 252).values

    return df, v4_score, v4_sub, pf, er_vals, atr_vals


def get_filtered_events(v4_score, pf):
    """가격필터 적용 후 시그널 이벤트 리스트 반환"""
    raw_events = detect_signal_events(v4_score, th=SIGNAL_THRESHOLD, cooldown=COOLDOWN)
    filtered   = [e for e in raw_events if pf(e['peak_idx'])]
    return raw_events, filtered


# ══════════════════════════════════════════════════════════════
# 2. Forward Return 계산
# ══════════════════════════════════════════════════════════════

def compute_forward_stats(events, df):
    """
    각 이벤트에 대해 여러 horizon의 수익률, MDD, 최대상승 계산.
    Returns: list of dict
    """
    close = df['Close'].values
    high  = df['High'].values
    low   = df['Low'].values
    n     = len(df)
    results = []

    for ev in events:
        pidx = ev['peak_idx']
        pprice = close[pidx]
        sig_type = ev['type']
        avail = n - 1 - pidx

        rec = {
            'type': sig_type,
            'peak_idx': pidx,
            'peak_date': df.index[pidx].strftime('%Y-%m-%d'),
            'peak_price': pprice,
            'peak_val': ev['peak_val'],
        }

        for h in HORIZONS:
            end_idx = min(pidx + h, n - 1)
            actual_h = end_idx - pidx
            if actual_h < 1:
                rec[f'ret_{h}'] = np.nan
                rec[f'max_dd_{h}'] = np.nan
                rec[f'max_up_{h}'] = np.nan
                continue

            # forward return
            rec[f'ret_{h}'] = (close[end_idx] - pprice) / pprice * 100

            # max drawdown from entry (always negative concept)
            window_low  = low[pidx+1 : end_idx+1]
            window_high = high[pidx+1 : end_idx+1]
            if len(window_low) == 0:
                rec[f'max_dd_{h}'] = np.nan
                rec[f'max_up_{h}'] = np.nan
                continue

            rec[f'max_dd_{h}'] = (window_low.min() - pprice) / pprice * 100
            rec[f'max_up_{h}'] = (window_high.max() - pprice) / pprice * 100

        rec['avail_days'] = avail
        results.append(rec)

    return results


# ══════════════════════════════════════════════════════════════
# 3. 신호 분류 (GOOD / BAD)
# ══════════════════════════════════════════════════════════════

def classify_signal(rec):
    """
    개별 신호를 GOOD / BAD / NEUTRAL로 분류.
    Returns: (category, reason)
    """
    sig = rec['type']

    if sig == 'bottom':  # BUY 시그널
        ret_60  = rec.get('ret_60',  np.nan)
        dd_20   = rec.get('max_dd_20', np.nan)
        up_60   = rec.get('max_up_60', np.nan)

        if not np.isnan(ret_60) and ret_60 >= GOOD_BUY_THRESH:
            return 'GOOD_BUY', f'60일 수익 {ret_60:+.1f}%'
        if not np.isnan(up_60) and up_60 >= GOOD_BUY_THRESH:
            return 'GOOD_BUY', f'60일 내 최대상승 {up_60:+.1f}%'
        if not np.isnan(dd_20) and dd_20 <= BAD_BUY_THRESH:
            return 'BAD_BUY', f'20일 내 최대하락 {dd_20:.1f}%'
        return 'NEUTRAL_BUY', f'60일 수익 {ret_60:+.1f}%' if not np.isnan(ret_60) else 'N/A'

    else:  # TOP → SELL 시그널
        ret_60  = rec.get('ret_60',  np.nan)
        up_60   = rec.get('max_up_60', np.nan)
        dd_60   = rec.get('max_dd_60', np.nan)

        if not np.isnan(dd_60) and dd_60 <= GOOD_SELL_THRESH:
            return 'GOOD_SELL', f'60일 내 최대하락 {dd_60:.1f}%'
        if not np.isnan(ret_60) and ret_60 <= GOOD_SELL_THRESH:
            return 'GOOD_SELL', f'60일 수익 {ret_60:+.1f}%'
        if not np.isnan(up_60) and up_60 >= BAD_SELL_THRESH:
            return 'BAD_SELL', f'60일 내 +{up_60:.1f}% 상승 놓침'
        if not np.isnan(ret_60) and ret_60 >= BAD_SELL_THRESH:
            return 'BAD_SELL', f'60일 후 +{ret_60:.1f}% (너무 일찍 매도)'
        return 'NEUTRAL_SELL', f'60일 수익 {ret_60:+.1f}%' if not np.isnan(ret_60) else 'N/A'


def check_late_sell(rec, df):
    """SELL 시그널이 이미 고점에서 많이 하락한 후 나왔는지 확인"""
    pidx = rec['peak_idx']
    close = df['Close'].values
    if pidx < 60:
        return False, 0.0
    lookback_high = max(close[max(0, pidx-60):pidx])
    drop_from_peak = (close[pidx] - lookback_high) / lookback_high * 100
    if drop_from_peak < -5.0:  # 이미 5% 이상 하락
        return True, drop_from_peak
    return False, drop_from_peak


# ══════════════════════════════════════════════════════════════
# 4. MISSED 기회 탐지
# ══════════════════════════════════════════════════════════════

def find_major_peaks_troughs(df, min_drop_pct=15.0, min_recovery_pct=15.0, window=120):
    """
    가격의 주요 고점/저점 탐지.
    고점→저점 하락률이 min_drop_pct 이상이고,
    저점→회복이 min_recovery_pct 이상인 쌍을 찾는다.
    """
    close = df['Close'].values
    n = len(close)

    # 로컬 고점/저점 탐지 (rolling window 기반)
    peaks = []
    troughs = []

    hw = window // 2  # half window

    for i in range(hw, n - hw):
        local_window = close[i - hw : i + hw + 1]
        if close[i] == local_window.max():
            peaks.append(i)
        if close[i] == local_window.min():
            troughs.append(i)

    # 중복 제거: 연속 고점 중 최고점만 유지
    filtered_peaks = []
    if peaks:
        group = [peaks[0]]
        for i in range(1, len(peaks)):
            if peaks[i] - peaks[i-1] <= hw:
                group.append(peaks[i])
            else:
                best = max(group, key=lambda x: close[x])
                filtered_peaks.append(best)
                group = [peaks[i]]
        best = max(group, key=lambda x: close[x])
        filtered_peaks.append(best)

    filtered_troughs = []
    if troughs:
        group = [troughs[0]]
        for i in range(1, len(troughs)):
            if troughs[i] - troughs[i-1] <= hw:
                group.append(troughs[i])
            else:
                best = min(group, key=lambda x: close[x])
                filtered_troughs.append(best)
                group = [troughs[i]]
        best = min(group, key=lambda x: close[x])
        filtered_troughs.append(best)

    # 주요 하락 구간 찾기
    major_bottoms = []
    for p_idx in filtered_peaks:
        for t_idx in filtered_troughs:
            if t_idx <= p_idx:
                continue
            if t_idx - p_idx > 500:  # 너무 먼 조합 skip
                break
            drop = (close[t_idx] - close[p_idx]) / close[p_idx] * 100
            if drop <= -min_drop_pct:
                # 저점 이후 회복 확인
                recovery_end = min(t_idx + 252, n - 1)
                if recovery_end > t_idx:
                    future_high = close[t_idx+1:recovery_end+1].max() if t_idx+1 <= recovery_end else close[t_idx]
                    recovery = (future_high - close[t_idx]) / close[t_idx] * 100
                    if recovery >= min_recovery_pct:
                        major_bottoms.append({
                            'peak_idx': p_idx,
                            'peak_date': df.index[p_idx].strftime('%Y-%m-%d'),
                            'peak_price': close[p_idx],
                            'trough_idx': t_idx,
                            'trough_date': df.index[t_idx].strftime('%Y-%m-%d'),
                            'trough_price': close[t_idx],
                            'drop_pct': drop,
                            'recovery_pct': recovery,
                        })
                        break  # 한 고점에 대해 첫 유효 저점만

    # 주요 고점 (저점→고점 상승 후 하락)
    major_tops = []
    for t_idx in filtered_troughs:
        for p_idx in filtered_peaks:
            if p_idx <= t_idx:
                continue
            if p_idx - t_idx > 500:
                break
            rise = (close[p_idx] - close[t_idx]) / close[t_idx] * 100
            if rise >= min_recovery_pct:
                # 고점 이후 하락 확인
                drop_end = min(p_idx + 252, n - 1)
                if drop_end > p_idx:
                    future_low = close[p_idx+1:drop_end+1].min() if p_idx+1 <= drop_end else close[p_idx]
                    drop_after = (future_low - close[p_idx]) / close[p_idx] * 100
                    if drop_after <= -min_drop_pct:
                        major_tops.append({
                            'trough_idx': t_idx,
                            'peak_idx': p_idx,
                            'peak_date': df.index[p_idx].strftime('%Y-%m-%d'),
                            'peak_price': close[p_idx],
                            'rise_pct': rise,
                            'subsequent_drop': drop_after,
                        })
                        break

    return major_bottoms, major_tops


def check_missed_opportunities(major_bottoms, major_tops, events, df, window=40):
    """
    주요 전환점 근처에 시그널이 있었는지 확인.
    window 거래일 이내에 적절한 시그널이 없으면 MISSED.
    """
    missed_bottoms = []
    missed_tops = []

    buy_events = [e for e in events if e['type'] == 'bottom']
    sell_events = [e for e in events if e['type'] == 'top']

    for mb in major_bottoms:
        t_idx = mb['trough_idx']
        # t_idx ± window 내에 buy 시그널 있는지?
        found = False
        for be in buy_events:
            if abs(be['peak_idx'] - t_idx) <= window:
                found = True
                break
        if not found:
            missed_bottoms.append(mb)

    for mt in major_tops:
        p_idx = mt['peak_idx']
        found = False
        for se in sell_events:
            if abs(se['peak_idx'] - p_idx) <= window:
                found = True
                break
        if not found:
            missed_tops.append(mt)

    return missed_bottoms, missed_tops


# ══════════════════════════════════════════════════════════════
# 5. BAD 시그널 원인 분석
# ══════════════════════════════════════════════════════════════

def analyze_bad_signal(rec, df, v4_sub, er_vals, atr_vals, pf):
    """
    BAD 시그널의 원인을 서브지표 수준에서 분석.
    """
    pidx = rec['peak_idx']
    analysis = {}

    # 서브지표 값
    if pidx < len(v4_sub):
        row = v4_sub.iloc[pidx]
        analysis['s_force'] = row['s_force']
        analysis['s_div']   = row['s_div']
        analysis['s_conc']  = row['s_conc']
        analysis['act']     = int(row['act'])
        analysis['score']   = row['score']

        # 어떤 지표가 가장 큰 기여를 했나?
        contrib_force = 0.30 * row['s_force']
        contrib_div   = 0.40 * row['s_div']
        contrib_conc  = 0.30 * row['s_conc']

        contribs = {
            'Force(30%)':  contrib_force,
            'Divergence(40%)': contrib_div,
            'Concordance(30%)': contrib_conc,
        }
        # 신호 방향에 기여한 정도
        if rec['type'] == 'bottom':
            # BUY: 양의 기여가 큰 지표가 주범
            dominant = max(contribs, key=lambda k: contribs[k])
        else:
            # SELL: 음의 기여가 큰 지표가 주범
            dominant = min(contribs, key=lambda k: contribs[k])

        analysis['dominant_indicator'] = dominant
        analysis['contributions'] = contribs

        # Activity Multiplier
        act_mult = {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.2}
        analysis['activity_multiplier'] = act_mult.get(int(row['act']), 1.0)

    # 가격필터 상태
    if pidx < len(er_vals):
        analysis['ER'] = er_vals[pidx]
        analysis['ATR_pct'] = atr_vals[pidx]
        analysis['price_filter_pass'] = pf(pidx)

    # 시그널 전후 스코어 추이 (5일 전~5일 후)
    s_start = max(0, pidx - 5)
    s_end   = min(len(v4_sub), pidx + 6)
    if s_end > s_start:
        analysis['score_context'] = v4_sub['score'].iloc[s_start:s_end].tolist()

    # 볼륨 이상 여부
    if pidx >= 20 and pidx < len(df):
        vol = df['Volume'].values
        vol_avg = np.mean(vol[pidx-20:pidx])
        vol_now = vol[pidx]
        analysis['vol_ratio'] = vol_now / vol_avg if vol_avg > 0 else np.nan
        analysis['vol_unusual'] = abs(analysis['vol_ratio'] - 1.0) > 0.5

    return analysis


# ══════════════════════════════════════════════════════════════
# 6. 가격필터 미적용 비교
# ══════════════════════════════════════════════════════════════

def analyze_price_filter_impact(raw_events, filtered_events, df):
    """가격필터에 의해 제거된 시그널의 품질 분석"""
    filtered_peaks = {e['peak_idx'] for e in filtered_events}
    removed = [e for e in raw_events if e['peak_idx'] not in filtered_peaks]

    removed_stats = compute_forward_stats(removed, df)
    good_removed = 0
    bad_removed  = 0
    for rs in removed_stats:
        cat, _ = classify_signal(rs)
        if 'GOOD' in cat:
            good_removed += 1
        elif 'BAD' in cat:
            bad_removed += 1

    return {
        'total_removed': len(removed),
        'good_removed': good_removed,
        'bad_removed': bad_removed,
        'removed_details': removed_stats,
    }


# ══════════════════════════════════════════════════════════════
# 7. 리포트 출력
# ══════════════════════════════════════════════════════════════

def print_separator(char='=', width=120):
    print(char * width)

def print_header(title, char='=', width=120):
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)


def run_diagnosis(ticker):
    """단일 종목에 대한 전체 진단 실행"""
    print_header(f"{ticker} V4_wP 신호 진단 보고서", '█', 120)

    # ── 데이터 로드 ──
    print(f"\n[데이터 로드]")
    df, v4_score, v4_sub, pf, er_vals, atr_vals = load_and_compute(ticker)
    print(f"  기간: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  거래일수: {len(df)}")

    # ── 시그널 생성 ──
    raw_events, filtered_events = get_filtered_events(v4_score, pf)
    print(f"\n[시그널 생성]")
    print(f"  원본 시그널: {len(raw_events)}개")
    print(f"  가격필터 후: {len(filtered_events)}개 (제거: {len(raw_events)-len(filtered_events)}개)")

    buy_events  = [e for e in filtered_events if e['type'] == 'bottom']
    sell_events = [e for e in filtered_events if e['type'] == 'top']
    print(f"  BUY 시그널:  {len(buy_events)}개")
    print(f"  SELL 시그널: {len(sell_events)}개")

    # ── Forward Return 계산 ──
    fr_results = compute_forward_stats(filtered_events, df)

    # ── 신호 분류 ──
    categories = {}
    all_classified = []
    for rec in fr_results:
        cat, reason = classify_signal(rec)
        rec['category'] = cat
        rec['reason'] = reason

        # Late sell 체크
        if rec['type'] == 'top':
            is_late, drop_from_peak = check_late_sell(rec, df)
            rec['is_late_sell'] = is_late
            rec['drop_from_recent_peak'] = drop_from_peak
            if is_late and 'GOOD' not in cat:
                cat = 'LATE_SELL'
                rec['category'] = cat
                rec['reason'] = f'고점에서 이미 {drop_from_peak:.1f}% 하락 후 시그널'

        categories.setdefault(cat, []).append(rec)
        all_classified.append(rec)

    # ══════════════════════════════════════════════════════════
    # 섹션 A: 신호 정확도 통계
    # ══════════════════════════════════════════════════════════
    print_header(f"[A] {ticker} 신호 정확도 통계")

    total = len(fr_results)
    cat_counts = {k: len(v) for k, v in categories.items()}

    print(f"\n  {'분류':<20} {'개수':>5} {'비율':>8}")
    print(f"  {'-'*35}")
    for cat_name in ['GOOD_BUY', 'BAD_BUY', 'NEUTRAL_BUY', 'GOOD_SELL', 'BAD_SELL', 'NEUTRAL_SELL', 'LATE_SELL']:
        cnt = cat_counts.get(cat_name, 0)
        pct = cnt / total * 100 if total > 0 else 0
        print(f"  {cat_name:<20} {cnt:>5} {pct:>7.1f}%")
    print(f"  {'-'*35}")
    print(f"  {'합계':<20} {total:>5} {100.0:>7.1f}%")

    # BUY 정확도
    n_buy = len(buy_events)
    good_buys = cat_counts.get('GOOD_BUY', 0)
    bad_buys  = cat_counts.get('BAD_BUY', 0)
    if n_buy > 0:
        print(f"\n  BUY 시그널 적중률: {good_buys}/{n_buy} = {good_buys/n_buy*100:.1f}%")
        print(f"  BUY 시그널 실패율: {bad_buys}/{n_buy} = {bad_buys/n_buy*100:.1f}%")

    # SELL 정확도
    n_sell = len(sell_events)
    good_sells = cat_counts.get('GOOD_SELL', 0)
    bad_sells  = cat_counts.get('BAD_SELL', 0)
    late_sells = cat_counts.get('LATE_SELL', 0)
    if n_sell > 0:
        print(f"  SELL 시그널 적중률: {good_sells}/{n_sell} = {good_sells/n_sell*100:.1f}%")
        print(f"  SELL 시그널 실패율: {bad_sells}/{n_sell} = {bad_sells/n_sell*100:.1f}%")
        print(f"  SELL 시그널 지연률: {late_sells}/{n_sell} = {late_sells/n_sell*100:.1f}%")

    # ── 호라이즌별 평균 수익률 ──
    print(f"\n  [BUY 시그널 후 평균 수익률]")
    print(f"  {'Horizon':>10}", end='')
    for h in HORIZONS:
        print(f"  {h}일", end='')
    print()

    buy_recs = [r for r in fr_results if r['type'] == 'bottom']
    if buy_recs:
        print(f"  {'평균수익':>10}", end='')
        for h in HORIZONS:
            vals = [r[f'ret_{h}'] for r in buy_recs if not np.isnan(r.get(f'ret_{h}', np.nan))]
            avg = np.mean(vals) if vals else np.nan
            print(f"  {avg:+.2f}%" if not np.isnan(avg) else "    N/A", end='')
        print()

        print(f"  {'최대DD':>10}", end='')
        for h in HORIZONS:
            vals = [r[f'max_dd_{h}'] for r in buy_recs if not np.isnan(r.get(f'max_dd_{h}', np.nan))]
            avg = np.mean(vals) if vals else np.nan
            print(f"  {avg:.2f}%" if not np.isnan(avg) else "    N/A", end='')
        print()

    print(f"\n  [SELL 시그널 후 평균 가격변동]")
    sell_recs = [r for r in fr_results if r['type'] == 'top']
    if sell_recs:
        print(f"  {'평균변동':>10}", end='')
        for h in HORIZONS:
            vals = [r[f'ret_{h}'] for r in sell_recs if not np.isnan(r.get(f'ret_{h}', np.nan))]
            avg = np.mean(vals) if vals else np.nan
            print(f"  {avg:+.2f}%" if not np.isnan(avg) else "    N/A", end='')
        print()

        print(f"  {'놓친상승':>10}", end='')
        for h in HORIZONS:
            vals = [r[f'max_up_{h}'] for r in sell_recs if not np.isnan(r.get(f'max_up_{h}', np.nan))]
            avg = np.mean(vals) if vals else np.nan
            print(f"  {avg:+.2f}%" if not np.isnan(avg) else "    N/A", end='')
        print()

    # ══════════════════════════════════════════════════════════
    # 섹션 B: 최악의 BAD 시그널 목록
    # ══════════════════════════════════════════════════════════
    print_header(f"[B] {ticker} 최악의 BAD 시그널 상세")

    # BAD BUY (최대 하락 순)
    bad_buy_list = categories.get('BAD_BUY', [])
    if bad_buy_list:
        bad_buy_sorted = sorted(bad_buy_list, key=lambda r: r.get('max_dd_20', 0))
        print(f"\n  ── BAD BUY 시그널 ({len(bad_buy_list)}개) ──")
        print(f"  {'날짜':<12} {'가격':>8} {'V4스코어':>9} {'20D DD':>8} {'60D수익':>8} {'120D수익':>9} {'원인분석'}")
        print(f"  {'-'*100}")
        for rec in bad_buy_sorted[:15]:
            analysis = analyze_bad_signal(rec, df, v4_sub, er_vals, atr_vals, pf)
            dominant = analysis.get('dominant_indicator', '?')
            act_m = analysis.get('activity_multiplier', 1.0)
            vol_ratio = analysis.get('vol_ratio', np.nan)

            dd_20 = rec.get('max_dd_20', np.nan)
            ret_60 = rec.get('ret_60', np.nan)
            ret_120 = rec.get('ret_120', np.nan)

            reason_parts = []
            reason_parts.append(f"주도: {dominant}")
            if act_m >= 2.0:
                reason_parts.append(f"ActMult={act_m:.1f}")
            if not np.isnan(vol_ratio) and analysis.get('vol_unusual', False):
                reason_parts.append(f"Vol비율={vol_ratio:.1f}x")

            print(f"  {rec['peak_date']:<12} {rec['peak_price']:>8.2f} {rec['peak_val']:>+9.3f} "
                  f"{dd_20:>+7.1f}% {ret_60:>+7.1f}% {ret_120:>+8.1f}%  "
                  f"{'|'.join(reason_parts)}")

            # 서브지표 내역
            contribs = analysis.get('contributions', {})
            if contribs:
                parts = [f"{k}={v:+.3f}" for k, v in contribs.items()]
                print(f"  {'':>12} 서브지표: {', '.join(parts)}, Act={analysis.get('act',0)}")

    # BAD SELL (놓친 상승 순)
    bad_sell_list = categories.get('BAD_SELL', [])
    if bad_sell_list:
        bad_sell_sorted = sorted(bad_sell_list, key=lambda r: -r.get('max_up_60', 0))
        print(f"\n  ── BAD SELL 시그널 ({len(bad_sell_list)}개) ──")
        print(f"  {'날짜':<12} {'가격':>8} {'V4스코어':>9} {'60D상승':>8} {'120D변동':>9} {'원인분석'}")
        print(f"  {'-'*100}")
        for rec in bad_sell_sorted[:15]:
            analysis = analyze_bad_signal(rec, df, v4_sub, er_vals, atr_vals, pf)
            dominant = analysis.get('dominant_indicator', '?')

            up_60 = rec.get('max_up_60', np.nan)
            ret_120 = rec.get('ret_120', np.nan)

            reason_parts = [f"주도: {dominant}"]
            act_m = analysis.get('activity_multiplier', 1.0)
            if act_m >= 2.0:
                reason_parts.append(f"ActMult={act_m:.1f}")

            print(f"  {rec['peak_date']:<12} {rec['peak_price']:>8.2f} {rec['peak_val']:>+9.3f} "
                  f"{up_60:>+7.1f}% {ret_120:>+8.1f}%  "
                  f"{'|'.join(reason_parts)}")
            contribs = analysis.get('contributions', {})
            if contribs:
                parts = [f"{k}={v:+.3f}" for k, v in contribs.items()]
                print(f"  {'':>12} 서브지표: {', '.join(parts)}, Act={analysis.get('act',0)}")

    # LATE SELL
    late_sell_list = categories.get('LATE_SELL', [])
    if late_sell_list:
        print(f"\n  ── LATE SELL 시그널 ({len(late_sell_list)}개) ──")
        print(f"  {'날짜':<12} {'가격':>8} {'V4스코어':>9} {'고점대비':>8} {'이유'}")
        print(f"  {'-'*80}")
        for rec in late_sell_list:
            dfp = rec.get('drop_from_recent_peak', 0)
            print(f"  {rec['peak_date']:<12} {rec['peak_price']:>8.2f} {rec['peak_val']:>+9.3f} "
                  f"{dfp:>+7.1f}%  {rec['reason']}")

    # ══════════════════════════════════════════════════════════
    # 섹션 C: MISSED 기회
    # ══════════════════════════════════════════════════════════
    print_header(f"[C] {ticker} 놓친 주요 기회 (MISSED)")

    major_bottoms, major_tops = find_major_peaks_troughs(df,
        min_drop_pct=MISSED_DROP_THRESH, min_recovery_pct=MISSED_RECOVERY, window=120)

    missed_bottoms, missed_tops = check_missed_opportunities(
        major_bottoms, major_tops, filtered_events, df, window=40)

    print(f"\n  주요 저점 {len(major_bottoms)}개 발견, 이중 BUY 시그널 MISSED: {len(missed_bottoms)}개")
    print(f"  주요 고점 {len(major_tops)}개 발견, 이중 SELL 시그널 MISSED: {len(missed_tops)}개")

    if missed_bottoms:
        print(f"\n  ── MISSED BOTTOM (매수 기회 놓침) ──")
        print(f"  {'저점날짜':<12} {'고점→저점':>12} {'하락률':>8} {'회복률':>8}  원인분석")
        print(f"  {'-'*90}")
        for mb in missed_bottoms:
            t_idx = mb['trough_idx']
            # 그 시점의 V4 스코어와 서브지표 확인
            if t_idx < len(v4_sub):
                row = v4_sub.iloc[t_idx]
                sc = row['score']
                sf = row['s_force']
                sd = row['s_div']
                scc = row['s_conc']
                act = int(row['act'])

                # 왜 시그널이 안 나왔나?
                reasons = []
                bot_th = SIGNAL_THRESHOLD * 0.5
                if sc <= bot_th:
                    reasons.append(f"V4스코어={sc:+.3f} (임계값 {bot_th:.3f} 미달)")
                else:
                    # 스코어는 충분한데 가격필터에 막힘?
                    if not pf(t_idx):
                        reasons.append(f"가격필터 차단 (ER={er_vals[t_idx]:.3f}, ATR={atr_vals[t_idx]:.3f})")
                    else:
                        reasons.append(f"V4스코어={sc:+.3f} 통과했지만 시점이 안 맞음")

                # 서브지표 중 어떤 것이 부족했나?
                if abs(sf) <= 0.1:
                    reasons.append("Force 비활성")
                if abs(sd) <= 0.1:
                    reasons.append("Divergence 비활성")
                if abs(scc) <= 0.1:
                    reasons.append("Concordance 비활성")

                print(f"  {mb['trough_date']:<12} "
                      f"{mb['peak_date']:>12} "
                      f"{mb['drop_pct']:>+7.1f}% {mb['recovery_pct']:>+7.1f}%  "
                      f"{'|'.join(reasons)}")
                print(f"  {'':>12} 서브: Force={sf:+.3f}, Div={sd:+.3f}, Conc={scc:+.3f}, Act={act}")
            else:
                print(f"  {mb['trough_date']:<12} "
                      f"{mb['peak_date']:>12} "
                      f"{mb['drop_pct']:>+7.1f}% {mb['recovery_pct']:>+7.1f}%  분석 불가")

    if missed_tops:
        print(f"\n  ── MISSED TOP (매도 기회 놓침) ──")
        print(f"  {'고점날짜':<12} {'이전상승':>8} {'이후하락':>8}  원인분석")
        print(f"  {'-'*80}")
        for mt in missed_tops:
            p_idx = mt['peak_idx']
            if p_idx < len(v4_sub):
                row = v4_sub.iloc[p_idx]
                sc = row['score']
                sf = row['s_force']
                sd = row['s_div']
                scc = row['s_conc']
                act = int(row['act'])

                reasons = []
                if sc >= -SIGNAL_THRESHOLD:
                    reasons.append(f"V4스코어={sc:+.3f} (매도임계 {-SIGNAL_THRESHOLD:.3f} 미달)")
                else:
                    if not pf(p_idx):
                        reasons.append(f"가격필터 차단")
                    else:
                        reasons.append(f"V4스코어={sc:+.3f} 충분했지만 시점 불일치")

                print(f"  {mt['peak_date']:<12} "
                      f"{mt['rise_pct']:>+7.1f}% {mt['subsequent_drop']:>+7.1f}%  "
                      f"{'|'.join(reasons)}")
                print(f"  {'':>12} 서브: Force={sf:+.3f}, Div={sd:+.3f}, Conc={scc:+.3f}, Act={act}")
            else:
                print(f"  {mt['peak_date']:<12} {mt['rise_pct']:>+7.1f}% {mt['subsequent_drop']:>+7.1f}%  분석 불가")

    # ══════════════════════════════════════════════════════════
    # 섹션 D: 가격필터 영향 분석
    # ══════════════════════════════════════════════════════════
    print_header(f"[D] {ticker} 가격필터 영향 분석")

    pf_impact = analyze_price_filter_impact(raw_events, filtered_events, df)
    n_removed = pf_impact['total_removed']
    print(f"\n  가격필터에 의해 제거된 시그널: {n_removed}개")
    print(f"  그 중 GOOD 시그널이었을 것: {pf_impact['good_removed']}개")
    print(f"  그 중 BAD  시그널이었을 것: {pf_impact['bad_removed']}개")
    n_neutral_removed = n_removed - pf_impact['good_removed'] - pf_impact['bad_removed']
    print(f"  그 중 NEUTRAL이었을 것:     {n_neutral_removed}개")

    if n_removed > 0:
        good_filter_pct = pf_impact['bad_removed'] / n_removed * 100
        bad_filter_pct  = pf_impact['good_removed'] / n_removed * 100
        print(f"\n  필터 효과: BAD 제거(정확)={good_filter_pct:.1f}%, GOOD 제거(오류)={bad_filter_pct:.1f}%")

    # ══════════════════════════════════════════════════════════
    # 섹션 E: 서브지표 실패 패턴 분석
    # ══════════════════════════════════════════════════════════
    print_header(f"[E] {ticker} 서브지표별 실패 패턴 분석")

    # BAD 시그널들의 주도 지표 통계
    bad_all = categories.get('BAD_BUY', []) + categories.get('BAD_SELL', []) + categories.get('LATE_SELL', [])

    if bad_all:
        dominant_counts = {}
        act_mult_high = 0
        vol_unusual_count = 0

        for rec in bad_all:
            analysis = analyze_bad_signal(rec, df, v4_sub, er_vals, atr_vals, pf)
            dom = analysis.get('dominant_indicator', 'Unknown')
            dominant_counts[dom] = dominant_counts.get(dom, 0) + 1
            if analysis.get('activity_multiplier', 1.0) >= 2.0:
                act_mult_high += 1
            if analysis.get('vol_unusual', False):
                vol_unusual_count += 1

        print(f"\n  BAD 시그널 총 {len(bad_all)}개의 주도 지표 분포:")
        for indicator, cnt in sorted(dominant_counts.items(), key=lambda x: -x[1]):
            pct = cnt / len(bad_all) * 100
            print(f"    {indicator:<25} {cnt:>3}개 ({pct:.1f}%)")

        print(f"\n  Activity Multiplier 2.2x (3개 지표 모두 활성): {act_mult_high}개 ({act_mult_high/len(bad_all)*100:.1f}%)")
        print(f"  비정상 거래량 동반: {vol_unusual_count}개 ({vol_unusual_count/len(bad_all)*100:.1f}%)")

    # 지표별 False Signal 기여도 분석 (GOOD vs BAD에서 각 지표의 평균값)
    print(f"\n  [GOOD vs BAD 시그널의 서브지표 평균 비교]")
    good_all = categories.get('GOOD_BUY', []) + categories.get('GOOD_SELL', [])

    if good_all and bad_all:
        print(f"  {'지표':<25} {'GOOD 평균':>12} {'BAD 평균':>12} {'차이':>10}")
        print(f"  {'-'*60}")
        for metric_name, col_key in [('Force (s_force)', 's_force'),
                                       ('Divergence (s_div)', 's_div'),
                                       ('Concordance (s_conc)', 's_conc'),
                                       ('Activity Count', 'act'),
                                       ('Score', 'score')]:
            good_vals = []
            bad_vals = []
            for rec in good_all:
                pidx = rec['peak_idx']
                if pidx < len(v4_sub):
                    good_vals.append(abs(v4_sub.iloc[pidx][col_key]))
            for rec in bad_all:
                pidx = rec['peak_idx']
                if pidx < len(v4_sub):
                    bad_vals.append(abs(v4_sub.iloc[pidx][col_key]))

            g_avg = np.mean(good_vals) if good_vals else np.nan
            b_avg = np.mean(bad_vals) if bad_vals else np.nan
            diff = g_avg - b_avg if not (np.isnan(g_avg) or np.isnan(b_avg)) else np.nan

            g_str = f"{g_avg:.4f}" if not np.isnan(g_avg) else "N/A"
            b_str = f"{b_avg:.4f}" if not np.isnan(b_avg) else "N/A"
            d_str = f"{diff:+.4f}" if not np.isnan(diff) else "N/A"
            print(f"  {metric_name:<25} {g_str:>12} {b_str:>12} {d_str:>10}")

    # ══════════════════════════════════════════════════════════
    # 섹션 F: 전체 시그널 목록 (시간순)
    # ══════════════════════════════════════════════════════════
    print_header(f"[F] {ticker} 전체 시그널 목록 (시간순)")

    print(f"\n  {'날짜':<12} {'종류':>6} {'V4':>8} {'분류':<15} {'5D':>7} {'10D':>7} {'20D':>7} {'60D':>7} {'120D':>8} {'이유'}")
    print(f"  {'-'*120}")

    sorted_recs = sorted(all_classified, key=lambda r: r['peak_idx'])
    for rec in sorted_recs:
        sig_label = 'BUY' if rec['type'] == 'bottom' else 'SELL'
        cat = rec.get('category', '?')

        # 색 코드 없이 카테고리 마킹
        marker = ''
        if 'GOOD' in cat:
            marker = '  [O]'
        elif 'BAD' in cat or 'LATE' in cat:
            marker = '  [X]'
        else:
            marker = '  [-]'

        r5  = rec.get('ret_5',   np.nan)
        r10 = rec.get('ret_10',  np.nan)
        r20 = rec.get('ret_20',  np.nan)
        r60 = rec.get('ret_60',  np.nan)
        r120 = rec.get('ret_120', np.nan)

        r5s  = f"{r5:+.1f}%" if not np.isnan(r5) else "  N/A"
        r10s = f"{r10:+.1f}%" if not np.isnan(r10) else "  N/A"
        r20s = f"{r20:+.1f}%" if not np.isnan(r20) else "  N/A"
        r60s = f"{r60:+.1f}%" if not np.isnan(r60) else "  N/A"
        r120s= f"{r120:+.1f}%" if not np.isnan(r120) else "   N/A"

        print(f"  {rec['peak_date']:<12} {sig_label:>6} {rec['peak_val']:>+8.3f} {cat:<15} "
              f"{r5s:>7} {r10s:>7} {r20s:>7} {r60s:>7} {r120s:>8} {marker}")

    # ══════════════════════════════════════════════════════════
    # 섹션 G: 시스템 약점 요약
    # ══════════════════════════════════════════════════════════
    print_header(f"[G] {ticker} 시스템 약점 요약 및 개선 제안")

    weaknesses = []

    # 1. BAD BUY 비율
    if n_buy > 0 and bad_buys / n_buy > 0.20:
        weaknesses.append(
            f"BUY 실패율 높음 ({bad_buys/n_buy*100:.0f}%): "
            f"바닥 시그널 임계값(현재 th*0.5={SIGNAL_THRESHOLD*0.5:.3f})이 너무 낮아 "
            f"노이즈 구간에서도 매수 시그널 발생"
        )

    # 2. BAD SELL 비율
    if n_sell > 0 and bad_sells / n_sell > 0.20:
        weaknesses.append(
            f"SELL 실패율 높음 ({bad_sells/n_sell*100:.0f}%): "
            f"상승 추세 도중 일시적 거래량 변화를 하락 전환으로 오인"
        )

    # 3. LATE SELL
    if n_sell > 0 and late_sells / n_sell > 0.15:
        weaknesses.append(
            f"SELL 지연 비율 높음 ({late_sells/n_sell*100:.0f}%): "
            f"가격이 이미 고점에서 상당히 하락한 후에야 매도 시그널 발생. "
            f"Divergence 지표의 후행성이 주 원인"
        )

    # 4. MISSED 분석
    if len(missed_bottoms) > 0:
        weaknesses.append(
            f"주요 저점 {len(missed_bottoms)}개 MISSED: "
            f"급락장에서 거래량 패턴이 평소와 다를 때 시그널 감도 부족. "
            f"가격필터의 과도한 제한도 원인 가능"
        )

    if len(missed_tops) > 0:
        weaknesses.append(
            f"주요 고점 {len(missed_tops)}개 MISSED: "
            f"점진적 고점 형성 시 거래량 변화가 충분히 감지되지 않음"
        )

    # 5. 가격필터 과잉 제거
    if pf_impact['good_removed'] > 2:
        weaknesses.append(
            f"가격필터가 GOOD 시그널 {pf_impact['good_removed']}개 제거: "
            f"ER/ATR 조건이 일부 유효한 시그널까지 차단. "
            f"현재 ER<{ER_QUANTILE}pct & ATR>{ATR_QUANTILE}pct 기준의 재조정 필요"
        )

    # 6. 서브지표 분석
    if bad_all:
        for rec in bad_all[:3]:
            analysis = analyze_bad_signal(rec, df, v4_sub, er_vals, atr_vals, pf)

    # 7. Divergence 후행성
    div_dominant = sum(1 for rec in bad_all
                       if analyze_bad_signal(rec, df, v4_sub, er_vals, atr_vals, pf).get('dominant_indicator', '') == 'Divergence(40%)') if bad_all else 0
    if bad_all and div_dominant / len(bad_all) > 0.30:
        weaknesses.append(
            f"Divergence 지표가 BAD 시그널의 {div_dominant/len(bad_all)*100:.0f}% 주도: "
            f"가중치 40%인 Divergence가 20일 모멘텀 비교 방식으로 인해 "
            f"추세 전환 구간에서 잘못된 방향 신호를 생성"
        )

    # Activity Multiplier 문제
    if bad_all and act_mult_high / len(bad_all) > 0.30:
        weaknesses.append(
            f"Activity Multiplier 과증폭: BAD 시그널의 {act_mult_high/len(bad_all)*100:.0f}%에서 "
            f"3개 지표 모두 활성(2.2x 배율). "
            f"지표들이 동시에 '틀린 방향'으로 활성화될 때 스코어가 과도하게 증폭됨"
        )

    print()
    if weaknesses:
        for i, w in enumerate(weaknesses, 1):
            print(f"  [{i}] {w}")
            print()
    else:
        print("  특별한 약점이 발견되지 않았습니다.")

    # 개선 제안
    print(f"\n  ── 개선 제안 ──")
    suggestions = [
        "BUY 시그널 임계값 상향: 현재 th*0.5 대신 th*0.7 이상으로 상향하여 노이즈 필터링 강화",
        "Divergence 가중치 조정: 40%→30%로 낮추고, Force를 30%→35%, Concordance를 30%→35%로",
        "SELL 시그널에 가격 모멘텀 확인 추가: 최근 20일 가격 추세가 실제 하락인지 확인 후 시그널 생성",
        "Activity Multiplier 상한 조정: 3개 동시 활성 시 2.2x→1.8x로 줄여 과증폭 방지",
        "가격필터 조건 완화: ATR_pct 퍼센타일을 66→55로 낮춰 급락장에서의 MISSED 줄이기",
        "LATE SELL 방지: 최근 고점 대비 이미 -5% 이상 하락한 상태에서의 SELL 시그널 억제",
        "주기적 자기상관 검사: 동일 방향 시그널이 짧은 간격으로 반복될 때 cooldown을 동적으로 확대",
    ]
    for i, s in enumerate(suggestions, 1):
        print(f"  {i}. {s}")

    print()
    return {
        'all_classified': all_classified,
        'categories': categories,
        'missed_bottoms': missed_bottoms,
        'missed_tops': missed_tops,
        'pf_impact': pf_impact,
        'n_buy': n_buy,
        'n_sell': n_sell,
    }


# ══════════════════════════════════════════════════════════════
# 8. 크로스 티커 비교
# ══════════════════════════════════════════════════════════════

def print_cross_ticker_summary(results_all):
    """SPY vs QQQ 비교 요약"""
    print_header("크로스 티커 비교 요약 (SPY vs QQQ)", '█', 120)

    print(f"\n  {'지표':<35}", end='')
    for ticker in TICKERS:
        print(f"  {ticker:>10}", end='')
    print()
    print(f"  {'-'*60}")

    metrics = [
        ('총 시그널 수', lambda r: len(r['all_classified'])),
        ('BUY 시그널', lambda r: r['n_buy']),
        ('SELL 시그널', lambda r: r['n_sell']),
        ('GOOD BUY', lambda r: len(r['categories'].get('GOOD_BUY', []))),
        ('BAD BUY', lambda r: len(r['categories'].get('BAD_BUY', []))),
        ('GOOD SELL', lambda r: len(r['categories'].get('GOOD_SELL', []))),
        ('BAD SELL', lambda r: len(r['categories'].get('BAD_SELL', []))),
        ('LATE SELL', lambda r: len(r['categories'].get('LATE_SELL', []))),
        ('MISSED BOTTOM', lambda r: len(r['missed_bottoms'])),
        ('MISSED TOP', lambda r: len(r['missed_tops'])),
        ('가격필터 제거 (GOOD)', lambda r: r['pf_impact']['good_removed']),
        ('가격필터 제거 (BAD)', lambda r: r['pf_impact']['bad_removed']),
    ]

    for label, fn in metrics:
        print(f"  {label:<35}", end='')
        for ticker in TICKERS:
            val = fn(results_all[ticker])
            print(f"  {val:>10}", end='')
        print()

    # BUY/SELL 적중률
    print(f"\n  {'적중률':<35}", end='')
    for ticker in TICKERS:
        r = results_all[ticker]
        n_buy = r['n_buy']
        good_buy = len(r['categories'].get('GOOD_BUY', []))
        rate = good_buy / n_buy * 100 if n_buy > 0 else 0
        print(f"  {rate:>9.1f}%", end='')
    print(" (BUY)")

    print(f"  {'':35}", end='')
    for ticker in TICKERS:
        r = results_all[ticker]
        n_sell = r['n_sell']
        good_sell = len(r['categories'].get('GOOD_SELL', []))
        rate = good_sell / n_sell * 100 if n_sell > 0 else 0
        print(f"  {rate:>9.1f}%", end='')
    print(" (SELL)")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print_header("V4_wP 신호 진단 시스템", '█', 120)
    print(f"\n  파라미터:")
    print(f"    V4_WINDOW={V4_WINDOW}, SIGNAL_THRESHOLD={SIGNAL_THRESHOLD}, COOLDOWN={COOLDOWN}")
    print(f"    ER_QUANTILE={ER_QUANTILE}, ATR_QUANTILE={ATR_QUANTILE}, LOOKBACK={LOOKBACK}")
    print(f"    데이터 기간: {DATA_START} ~ {DATA_END}")
    print(f"    대상: {', '.join(TICKERS)}")
    print(f"    GOOD BUY 기준: 60일 내 +{GOOD_BUY_THRESH}%")
    print(f"    BAD BUY 기준: 20일 내 {BAD_BUY_THRESH}%")
    print(f"    GOOD SELL 기준: 60일 내 {GOOD_SELL_THRESH}%")
    print(f"    BAD SELL 기준: 60일 내 +{BAD_SELL_THRESH}%")

    results_all = {}
    for ticker in TICKERS:
        results_all[ticker] = run_diagnosis(ticker)

    # 크로스 비교
    print_cross_ticker_summary(results_all)

    print_header("진단 완료", '█', 120)
    print()


if __name__ == '__main__':
    main()
