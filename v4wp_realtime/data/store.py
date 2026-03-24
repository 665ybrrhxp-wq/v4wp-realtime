"""SQLite 데이터 저장소"""
import sqlite3
from pathlib import Path
from v4wp_realtime.config.settings import DB_PATH, DATA_DIR


def _try_publish(event_type, data):
    """이벤트 버스로 발행 시도 (API 서버 미실행 시 무시)."""
    try:
        from v4wp_realtime.api.event_bus import publish
        publish(event_type, data)
    except Exception:
        pass  # API 서버 미실행 환경 (GitHub Actions 등)


SCHEMA_PATH = Path(__file__).parent / 'schema.sql'


def get_connection(db_path=None):
    """SQLite 연결 반환"""
    path = db_path or DB_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=None):
    """스키마 초기화 (테이블 없으면 생성)"""
    conn = get_connection(db_path)
    with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()


def upsert_daily_scores(conn, rows):
    """일별 스코어 upsert.
    rows: list of dict with keys: date, ticker, score, s_force, s_div, s_conc,
          close_price, er, atr_pct
    """
    conn.executemany(
        """INSERT OR REPLACE INTO daily_scores
           (date, ticker, score, s_force, s_div, s_conc, close_price, er, atr_pct)
           VALUES (:date, :ticker, :score, :s_force, :s_div, :s_conc,
                   :close_price, :er, :atr_pct)""",
        rows
    )
    conn.commit()

    # SSE 이벤트 발행
    tickers = list({r['ticker'] for r in rows if isinstance(r, dict)})
    if tickers:
        _try_publish("scores_updated", {"tickers": tickers, "count": len(rows)})


def insert_signal_event(conn, event):
    """신호 이벤트 삽입 (중복 시 무시).
    Returns: True if inserted, False if duplicate.
    """
    try:
        conn.execute(
            """INSERT INTO signal_events
               (ticker, signal_type, peak_date, peak_val, start_val, close_price,
                detected_date, notified, commentary, s_force, s_div, s_conc, er, atr_pct,
                signal_tier, action_pct)
               VALUES (:ticker, :signal_type, :peak_date, :peak_val, :start_val, :close_price,
                       :detected_date, :notified, :commentary,
                       :s_force, :s_div, :s_conc, :er, :atr_pct,
                       :signal_tier, :action_pct)""",
            event
        )
        conn.commit()

        # SSE 이벤트 발행
        ticker = event.get('ticker') if isinstance(event, dict) else None
        _try_publish("signal_detected", {
            "ticker": ticker,
            "signal_type": event.get('signal_type') if isinstance(event, dict) else None,
            "peak_date": event.get('peak_date') if isinstance(event, dict) else None,
        })

        return True
    except sqlite3.IntegrityError:
        return False


def mark_notified(conn, event_id):
    """알림 전송 완료 표시"""
    conn.execute(
        "UPDATE signal_events SET notified = 1 WHERE id = ?",
        (event_id,)
    )
    conn.commit()


def get_recent_signals(conn, ticker=None, days=30):
    """최근 N일 신호 조회"""
    query = """SELECT * FROM signal_events
               WHERE peak_date >= date('now', ?) """
    params = [f'-{days} days']
    if ticker:
        query += " AND ticker = ?"
        params.append(ticker)
    query += " ORDER BY peak_date DESC"
    return conn.execute(query, params).fetchall()


def get_existing_signal_dates(conn, ticker, signal_type, date_range):
    """기존 신호 날짜 조회 (중복 판별용)"""
    start, end = date_range
    rows = conn.execute(
        """SELECT peak_date FROM signal_events
           WHERE ticker = ? AND signal_type = ?
           AND peak_date BETWEEN ? AND ?""",
        (ticker, signal_type, start, end)
    ).fetchall()
    return [r['peak_date'] for r in rows]


def log_scan_run(conn, run_date, n_tickers, n_new, n_alerts, duration, status):
    """스캔 실행 로그"""
    conn.execute(
        """INSERT INTO scan_runs
           (run_date, n_tickers, n_new_signals, n_alerts_sent, duration_sec, status)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (run_date, n_tickers, n_new, n_alerts, duration, status)
    )
    conn.commit()


def get_latest_scores(conn, n_days=30):
    """최근 N일 스코어 전체 조회 (대시보드용)"""
    return conn.execute(
        """SELECT * FROM daily_scores
           WHERE date >= date('now', ?)
           ORDER BY date DESC, ticker""",
        (f'-{n_days} days',)
    ).fetchall()
