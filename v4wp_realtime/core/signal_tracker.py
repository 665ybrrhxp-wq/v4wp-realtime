"""신호 중복 제거 + 신규 판별"""
from datetime import datetime, timedelta
from v4wp_realtime.data.store import get_existing_signal_dates


def is_new_signal(conn, ticker, signal_type, peak_date_str, window_days=3):
    """기존 DB에 동일(또는 근접) 신호가 있는지 확인.
    +-window_days 범위 내 같은 ticker+type 신호가 있으면 중복.
    Returns True if new signal.
    """
    peak_date = datetime.strptime(peak_date_str, '%Y-%m-%d')
    start = (peak_date - timedelta(days=window_days)).strftime('%Y-%m-%d')
    end = (peak_date + timedelta(days=window_days)).strftime('%Y-%m-%d')

    existing = get_existing_signal_dates(conn, ticker, signal_type, (start, end))
    return len(existing) == 0


def extract_recent_events(events, df, lookback_days=10):
    """최근 N일 이내 발생한 이벤트만 추출.
    Returns list of event dicts with peak_date added.
    """
    if not events or len(df) == 0:
        return []

    cutoff = df.index[-1] - timedelta(days=lookback_days)
    recent = []
    for ev in events:
        peak_date = df.index[ev['peak_idx']]
        if peak_date >= cutoff:
            ev_copy = dict(ev)
            ev_copy['peak_date'] = peak_date.strftime('%Y-%m-%d')
            recent.append(ev_copy)
    return recent
