"""real_market_backtest.py 핵심 함수 래핑 + C25 신호 분류"""
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
    """단일 종목 분석: 스코어 + 신호 + 필터 + LATE_SELL_BLOCK 적용.

    Returns: dict with keys:
        score: V4 score Series
        subindicators: DataFrame (s_force, s_div, s_conc, act, score)
        all_events: list of all signal events
        filtered_events: list of price-filtered + late_sell_blocked events
        blocked_sells: list of blocked sell events (for reference)
        price_filter_fn: the filter function
    """
    w = params.get('v4_window', 20)
    th = params.get('signal_threshold', 0.15)
    cd = params.get('cooldown', 5)
    er_q = params.get('er_quantile', 66)
    atr_q = params.get('atr_quantile', 55)
    lb = params.get('lookback', 252)
    late_sell_drop_th = params.get('late_sell_drop_th', 0.05)

    score = calc_v4_score(df, w=w)
    subind = calc_v4_subindicators(df, w=w)
    events = detect_signal_events(score, th=th, cooldown=cd)
    pf = build_price_filter(df, er_q=er_q, atr_q=atr_q, lookback=lb)

    filtered = [e for e in events if pf(e['peak_idx'])]

    # LATE_SELL_BLOCK: 20일 롤링 고점 대비 drop_th% 이상 하락 시 매도 차단
    close = df['Close'].values
    rolling_high_20 = df['Close'].rolling(20, min_periods=1).max().values

    active_events = []
    blocked_sells = []
    for e in filtered:
        if e['type'] == 'top':
            pidx = e['peak_idx']
            if pidx < len(close):
                price = close[pidx]
                rh = rolling_high_20[pidx]
                drop_pct = (rh - price) / rh if rh > 0 else 0
                if drop_pct > late_sell_drop_th:
                    blocked_sells.append(e)
                    continue
        active_events.append(e)

    return {
        'score': score,
        'subindicators': subind,
        'all_events': events,
        'filtered_events': active_events,
        'blocked_sells': blocked_sells,
        'price_filter_fn': pf,
    }


def classify_signal(ev, params):
    """C25 신호 강도 분류.

    Returns: dict with:
        is_strong: bool
        label: str ('STRONG BUY', 'BUY', 'STRONG SELL', 'SELL')
        action_pct: float (매수/매도 비율)
    """
    strong_buy_th = params.get('strong_buy_th', 0.25)
    strong_sell_th = params.get('strong_sell_th', -0.25)
    buy_normal = params.get('buy_normal_pct', 0.40)
    buy_strong = params.get('buy_strong_pct', 0.60)
    sell_normal = params.get('sell_normal_pct', 0.05)
    sell_strong = params.get('sell_strong_pct', 0.10)

    if ev['type'] == 'bottom':
        is_strong = abs(ev['peak_val']) >= strong_buy_th
        return {
            'is_strong': is_strong,
            'label': 'STRONG BUY' if is_strong else 'BUY',
            'action_pct': buy_strong if is_strong else buy_normal,
        }
    else:
        is_strong = ev['peak_val'] <= strong_sell_th
        return {
            'is_strong': is_strong,
            'label': 'STRONG SELL' if is_strong else 'SELL',
            'action_pct': sell_strong if is_strong else sell_normal,
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
