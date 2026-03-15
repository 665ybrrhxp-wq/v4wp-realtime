"""real_market_backtest.py 핵심 함수 래핑 + Duration 기반 신호 분류 (DivGate_3d, Earnings Vol Filter)"""
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (real_market_backtest import용)
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from real_market_backtest import (
    download_data,
    calc_v4_score,
    calc_v4_subindicators,
    detect_signal_events,
    build_price_filter,
    calc_forward_returns,
    smooth_earnings_volume,
)


def fetch_data(ticker, years=3, cache_dir=None):
    """최근 N년 데이터 다운로드"""
    from datetime import datetime, timedelta
    if cache_dir is None:
        cache_dir = str(Path(_project_root) / 'cache')
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
    return download_data(ticker, start=start, end=end, cache_dir=cache_dir)


def analyze_ticker(ticker, df, params):
    """단일 종목 분석: 스코어 + 신호 + 필터 + BUY_DD_GATE + LATE_SELL_BLOCK 적용.

    BUY_DD_GATE: N일 고점 대비 dd_threshold% 이상 하락일 때만 매수 신호 통과.
    LATE_SELL_BLOCK: 20일 고점 대비 drop_th% 이상 하락 시 매도 차단.

    Returns: dict with keys:
        score: V4 score Series
        subindicators: DataFrame (s_force, s_div, s_conc, act, score)
        all_events: list of all signal events
        filtered_events: list of price-filtered + dd_gated + late_sell_blocked events
        blocked_sells: list of blocked sell events (for reference)
        blocked_buys: list of dd-gated buy events (for reference)
        price_filter_fn: the filter function
    """
    w = params.get('v4_window', 20)
    th = params.get('signal_threshold', 0.15)
    cd = params.get('cooldown', 5)
    er_q = params.get('er_quantile', 66)
    atr_q = params.get('atr_quantile', 55)
    lb = params.get('lookback', 252)
    late_sell_drop_th = params.get('late_sell_drop_th', 0.05)
    divgate_days = params.get('divgate_days', 3)
    earnings_vol_filter = params.get('earnings_vol_filter', True)
    buy_dd_lookback = params.get('buy_dd_lookback', 20)
    buy_dd_threshold = params.get('buy_dd_threshold', 0.05)

    # 실적발표일 거래량 스무딩 (전처리)
    if earnings_vol_filter:
        df = smooth_earnings_volume(df, ticker=ticker)

    score = calc_v4_score(df, w=w, divgate_days=divgate_days)
    subind = calc_v4_subindicators(df, w=w, divgate_days=divgate_days)
    events = detect_signal_events(score, th=th, cooldown=cd)
    pf = build_price_filter(df, er_q=er_q, atr_q=atr_q, lookback=lb)

    filtered = [e for e in events if pf(e['peak_idx'])]

    # 롤링 고점 계산 (매수 DD 게이트 + 매도 LATE_SELL_BLOCK 공용)
    close = df['Close'].values
    rolling_high_sell = df['Close'].rolling(20, min_periods=1).max().values
    rolling_high_buy = df['Close'].rolling(buy_dd_lookback, min_periods=1).max().values

    active_events = []
    blocked_sells = []
    blocked_buys = []
    for e in filtered:
        pidx = e['peak_idx']
        if pidx >= len(close):
            continue

        price = close[pidx]

        if e['type'] == 'top':
            # LATE_SELL_BLOCK: 20일 고점 대비 drop_th% 이상 하락 시 매도 차단
            rh = rolling_high_sell[pidx]
            drop_pct = (rh - price) / rh if rh > 0 else 0
            if drop_pct > late_sell_drop_th:
                blocked_sells.append(e)
                continue
        elif e['type'] == 'bottom':
            # BUY_DD_GATE: N일 고점 대비 dd_threshold% 이상 하락일 때만 매수 허용
            rh = rolling_high_buy[pidx]
            dd = (rh - price) / rh if rh > 0 else 0
            if dd < buy_dd_threshold:
                blocked_buys.append(e)
                continue

        active_events.append(e)

    return {
        'score': score,
        'subindicators': subind,
        'all_events': events,
        'filtered_events': active_events,
        'blocked_sells': blocked_sells,
        'blocked_buys': blocked_buys,
        'price_filter_fn': pf,
    }


def classify_signal(ev, params):
    """Duration 기반 신호 분류 (매수/매도 모두 duration 확인).

    - 매수: duration >= confirm_days -> CONFIRMED (100% 매수)
            duration < confirm_days -> PENDING (대기, 스킵)
    - 매도: duration >= sell_confirm_days -> SELL_CONFIRMED (5% 매도)
            duration < sell_confirm_days -> PENDING (대기, 스킵)

    Returns: dict with:
        tier: str ('CONFIRMED', 'SELL_CONFIRMED', 'PENDING')
        label: str
        action_pct: float (매수/매도 비율)
        is_strong: bool
    """
    confirm_days = params.get('confirm_days', 3)
    sell_confirm_days = params.get('sell_confirm_days', 3)
    buy_confirmed = params.get('buy_confirmed_pct', 1.00)
    sell_confirmed = params.get('sell_confirmed_pct', 0.05)

    duration = ev.get('duration', ev['end_idx'] - ev['start_idx'] + 1)

    if ev['type'] == 'bottom':
        if duration >= confirm_days:
            return {
                'tier': 'CONFIRMED',
                'is_strong': True,
                'label': 'BUY (CONFIRMED)',
                'action_pct': buy_confirmed,
            }
        else:
            return {
                'tier': 'PENDING',
                'is_strong': False,
                'label': 'BUY (PENDING)',
                'action_pct': 0.0,
            }
    else:
        if duration >= sell_confirm_days:
            return {
                'tier': 'SELL_CONFIRMED',
                'is_strong': True,
                'label': 'SELL (CONFIRMED)',
                'action_pct': sell_confirmed,
            }
        else:
            return {
                'tier': 'PENDING',
                'is_strong': False,
                'label': 'SELL (PENDING)',
                'action_pct': 0.0,
            }


def get_latest_score_data(df, subind, n_days=1):
    """최근 N일 스코어 데이터를 dict 리스트로 반환 (daily_scores 테이블 저장용)"""
    rows = []
    for i in range(-n_days, 0):
        idx = len(df) + i
        if idx < 0:
            continue
        date_str = df.index[idx].strftime('%Y-%m-%d')
        rows.append({
            'date': date_str,
            'score': float(subind['score'].iloc[idx]) if idx < len(subind) else None,
            's_force': float(subind['s_force'].iloc[idx]) if idx < len(subind) else None,
            's_div': float(subind['s_div'].iloc[idx]) if idx < len(subind) else None,
            's_conc': float(subind['s_conc'].iloc[idx]) if idx < len(subind) else None,
            'close_price': float(df['Close'].iloc[idx]),
        })
    return rows
