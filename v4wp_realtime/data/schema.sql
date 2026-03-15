-- V4_wP Realtime: SQLite Schema

-- 일별 스코어 (대시보드 히트맵용)
CREATE TABLE IF NOT EXISTS daily_scores (
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    score REAL,
    s_force REAL,
    s_div REAL,
    s_conc REAL,
    close_price REAL,
    er REAL,
    atr_pct REAL,
    PRIMARY KEY (date, ticker)
);

-- 신호 이벤트 (중복 제거 + 알림 추적)
CREATE TABLE IF NOT EXISTS signal_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    peak_date TEXT NOT NULL,
    peak_val REAL,
<<<<<<< HEAD
    start_val REAL,
=======
>>>>>>> 187a32a6aa96e6dada11f8fbf85eaa48a75ec451
    close_price REAL,
    detected_date TEXT,
    notified INTEGER DEFAULT 0,
    commentary TEXT,
    s_force REAL,
    s_div REAL,
    s_conc REAL,
    er REAL,
    atr_pct REAL,
<<<<<<< HEAD
    signal_tier TEXT,
    action_pct REAL,
=======
>>>>>>> 187a32a6aa96e6dada11f8fbf85eaa48a75ec451
    UNIQUE(ticker, signal_type, peak_date)
);

-- 스캔 실행 로그
CREATE TABLE IF NOT EXISTS scan_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date TEXT NOT NULL,
    n_tickers INTEGER,
    n_new_signals INTEGER,
    n_alerts_sent INTEGER,
    duration_sec REAL,
    status TEXT
);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_daily_scores_date ON daily_scores(date);
CREATE INDEX IF NOT EXISTS idx_daily_scores_ticker ON daily_scores(ticker);
CREATE INDEX IF NOT EXISTS idx_signal_events_ticker ON signal_events(ticker);
CREATE INDEX IF NOT EXISTS idx_signal_events_peak_date ON signal_events(peak_date);
CREATE INDEX IF NOT EXISTS idx_signal_events_notified ON signal_events(notified);
