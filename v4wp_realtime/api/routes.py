"""FastAPI 엔드포인트 — V4_wP Signal Dashboard API"""
import asyncio
import json
import sqlite3
from pathlib import Path
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse, JSONResponse

from v4wp_realtime.data.store import get_connection, init_db
from v4wp_realtime.config.settings import load_watchlist, DB_PATH
from v4wp_realtime.api.event_bus import subscribe, unsubscribe, subscriber_count

app = FastAPI(title="V4_wP Signal API", version="1.0.0")

# 빌드된 Mini App 정적 파일 서빙
_webapp_dist = Path(__file__).resolve().parent.parent / "webapp" / "dist"
if _webapp_dist.exists():
    app.mount("/app", StaticFiles(directory=str(_webapp_dist), html=True), name="webapp")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _db():
    """DB 연결 반환. DB 파일 없으면 초기화."""
    if not DB_PATH.exists():
        init_db()
    return get_connection()


def _rows_to_dicts(rows):
    """sqlite3.Row 리스트 → dict 리스트"""
    return [dict(r) for r in rows]


# ── 1. GET /api/watchlist ──────────────────────────────────────────────
@app.get("/api/watchlist")
def get_watchlist():
    """각 종목의 최신 스코어 + 전일대비 변화율 + 최근 시그널."""
    wl = load_watchlist()
    tickers = list(wl["tickers"].keys())
    result = []

    try:
        conn = _db()
    except Exception:
        return result

    try:
        for ticker in tickers:
            sector = wl["tickers"][ticker].get("sector", "")

            # 최근 2일 스코어 (변화율 계산용)
            rows = conn.execute(
                """SELECT date, score, s_force, close_price, er, atr_pct
                   FROM daily_scores
                   WHERE ticker = ?
                   ORDER BY date DESC LIMIT 2""",
                (ticker,),
            ).fetchall()

            if not rows:
                result.append({
                    "ticker": ticker, "sector": sector,
                    "close_price": None, "prev_close": None, "change_pct": None,
                    "score": None, "s_force": None, "er": None, "atr_pct": None,
                    "date": None, "recent_signal": None,
                })
                continue

            latest = dict(rows[0])
            prev = dict(rows[1]) if len(rows) > 1 else None

            change_pct = None
            if prev and prev["close_price"] and latest["close_price"]:
                change_pct = round(
                    (latest["close_price"] - prev["close_price"])
                    / prev["close_price"] * 100, 2
                )

            # 최근 시그널 1개
            sig_row = conn.execute(
                """SELECT signal_type, peak_date, signal_tier, s_force, peak_val
                   FROM signal_events
                   WHERE ticker = ?
                   ORDER BY peak_date DESC LIMIT 1""",
                (ticker,),
            ).fetchone()

            recent_signal = None
            if sig_row:
                s = dict(sig_row)
                recent_signal = {
                    "direction": "LONG" if s["signal_type"] == "bottom" else "SHORT",
                    "peak_date": s["peak_date"],
                    "tier": s["signal_tier"],
                    "s_force": s["s_force"],
                    "peak_val": s["peak_val"],
                }

            result.append({
                "ticker": ticker,
                "sector": sector,
                "date": latest["date"],
                "close_price": latest["close_price"],
                "prev_close": prev["close_price"] if prev else None,
                "change_pct": change_pct,
                "score": latest["score"],
                "s_force": latest["s_force"],
                "er": latest["er"],
                "atr_pct": latest["atr_pct"],
                "recent_signal": recent_signal,
            })
    finally:
        conn.close()

    return result


# ── 2. GET /api/signals/{ticker} ───────────────────────────────────────
@app.get("/api/signals/{ticker}")
def get_signals(ticker: str, days: int = Query(default=90, ge=1, le=365)):
    """해당 종목의 시그널 히스토리."""
    try:
        conn = _db()
    except Exception:
        return []

    try:
        rows = conn.execute(
            """SELECT
                 peak_date   AS date,
                 signal_type,
                 close_price,
                 s_force,
                 s_div,
                 s_conc,
                 er,
                 atr_pct,
                 peak_val,
                 signal_tier,
                 action_pct,
                 commentary,
                 detected_date,
                 interpretation
               FROM signal_events
               WHERE ticker = ?
                 AND peak_date >= date('now', ?)
               ORDER BY peak_date DESC""",
            (ticker.upper(), f"-{days} days"),
        ).fetchall()
    finally:
        conn.close()

    result = []
    for r in _rows_to_dicts(rows):
        r["direction"] = "LONG" if r["signal_type"] == "bottom" else "SHORT"
        r["squeeze"] = bool(r["s_conc"] and abs(r["s_conc"]) > 0)
        if r.get("interpretation"):
            try:
                r["interpretation"] = json.loads(r["interpretation"])
            except (json.JSONDecodeError, TypeError):
                r["interpretation"] = None
        result.append(r)

    return result


# ── 3. GET /api/indicators/{ticker} ───────────────────────────────────
@app.get("/api/indicators/{ticker}")
def get_indicators(ticker: str):
    """최신 지표 상태: AND-GEO 핵심 지표 + 신호 파이프라인 상태."""
    wl = load_watchlist()
    params = wl.get("params", {})
    sig_th = params.get("signal_threshold", 0.05)
    bot_th = sig_th * 0.5  # bottom signal threshold
    dd_lookback = params.get("buy_dd_lookback", 20)
    dd_threshold = params.get("buy_dd_threshold", 0.03)
    confirm_days = params.get("confirm_days", 1)

    try:
        conn = _db()
    except Exception:
        return {"ticker": ticker.upper(), "data": None, "filters": []}

    try:
        row = conn.execute(
            """SELECT date, score, s_force, s_div, close_price, er, atr_pct
               FROM daily_scores
               WHERE ticker = ?
               ORDER BY date DESC LIMIT 1""",
            (ticker.upper(),),
        ).fetchone()

        # 파이프라인 상태 계산용: 최근 N일 스코어
        recent = conn.execute(
            """SELECT score, close_price FROM daily_scores
               WHERE ticker = ?
               ORDER BY date DESC LIMIT ?""",
            (ticker.upper(), max(dd_lookback, 30)),
        ).fetchall()
    finally:
        conn.close()

    if not row:
        return {"ticker": ticker.upper(), "data": None, "filters": []}

    d = dict(row)

    s_force = d.get("s_force") or 0
    s_div = d.get("s_div") or 0
    score = d.get("score") or 0
    close = d.get("close_price") or 0
    er = d.get("er")
    atr_pct = d.get("atr_pct")

    and_geo_active = s_force > 0 and s_div > 0

    # ── 신호 파이프라인 상태 ──
    # 1) 연속 threshold 초과 일수
    streak = 0
    for r in recent:  # DESC 순서 (최근→과거)
        if r["score"] and r["score"] > bot_th:
            streak += 1
        else:
            break

    # 2) DD Gate: N일 고점 대비 낙폭
    prices = [r["close_price"] for r in recent[:dd_lookback]
              if r["close_price"]]
    high_20d = max(prices) if prices else close
    dd_pct = (high_20d - close) / high_20d if high_20d > 0 else 0

    pipeline = {
        "above_threshold": score > bot_th,
        "threshold": round(bot_th, 4),
        "streak_days": streak,
        "confirm_days": confirm_days,
        "duration_ok": streak >= confirm_days,
        "dd_pct": round(dd_pct * 100, 2),
        "dd_threshold": round(dd_threshold * 100, 2),
        "dd_ok": dd_pct >= dd_threshold,
    }

    # 가격 필터 (값이 있을 때만)
    filters = []
    if er is not None:
        filters.append({
            "label": "ER (efficiency ratio)",
            "value": round(er, 4),
            "desc": "낮을수록 반전 가능성 ↑",
        })
    if atr_pct is not None:
        filters.append({
            "label": "ATR%",
            "value": round(atr_pct, 2),
            "desc": "높을수록 변동성 충분",
        })

    return {
        "ticker": ticker.upper(),
        "data": d,
        "score": round(score, 4),
        "s_force": round(s_force, 4),
        "s_div": round(s_div, 4),
        "and_geo_active": and_geo_active,
        "pipeline": pipeline,
        "filters": filters,
    }


# ── 3b. GET /api/interpretation/{ticker} ──────────────────────────────
@app.get("/api/interpretation/{ticker}")
def get_interpretation(ticker: str):
    """최신 AI 멀티 페르소나 해석."""
    try:
        conn = _db()
    except Exception:
        return {"ticker": ticker.upper(), "interpretation": None}

    try:
        row = conn.execute(
            """SELECT interpretation, peak_date, signal_type, signal_tier
               FROM signal_events
               WHERE ticker = ? AND interpretation IS NOT NULL
               ORDER BY peak_date DESC LIMIT 1""",
            (ticker.upper(),),
        ).fetchone()
    finally:
        conn.close()

    if not row or not row["interpretation"]:
        return {"ticker": ticker.upper(), "interpretation": None}

    try:
        interp = json.loads(row["interpretation"])
    except (json.JSONDecodeError, TypeError):
        interp = None

    return {
        "ticker": ticker.upper(),
        "peak_date": row["peak_date"],
        "signal_type": row["signal_type"],
        "signal_tier": row["signal_tier"],
        "interpretation": interp,
    }


# ── 3c. GET /api/postmortem/{ticker} ───────────────────────────────────
@app.get("/api/postmortem/{ticker}")
def get_postmortem(ticker: str):
    """시그널 사후 검증 결과 + 페르소나 정확도."""
    try:
        conn = _db()
    except Exception:
        return {"ticker": ticker.upper(), "stats": None}

    try:
        from v4wp_realtime.core.postmortem import get_postmortem_stats
        stats = get_postmortem_stats(conn, ticker.upper())
    finally:
        conn.close()

    return {"ticker": ticker.upper(), **stats}


# ── 3d. GET /api/similar-signals/{ticker} ─────────────────────────────
@app.get("/api/similar-signals/{ticker}")
def get_similar_signals(ticker: str):
    """최신 시그널의 유사 과거 시그널 Top 5."""
    try:
        conn = _db()
    except Exception:
        return {"ticker": ticker.upper(), "similar": []}

    try:
        # 해당 티커의 최신 시그널 조회
        row = conn.execute(
            """SELECT id, s_force, s_div, peak_val, start_val, dd_pct, duration,
                      market_return_20d, vix_change_20d
               FROM signal_events
               WHERE ticker = ?
               ORDER BY peak_date DESC LIMIT 1""",
            (ticker.upper(),),
        ).fetchone()

        if not row:
            return {"ticker": ticker.upper(), "similar": []}

        from v4wp_realtime.core.similarity import find_similar_signals
        signal = dict(row)
        similar = find_similar_signals(conn, signal, exclude_id=row['id'])
    finally:
        conn.close()

    return {"ticker": ticker.upper(), "similar": similar}


# ── 4. GET /api/chart-data/{ticker} ──────────────────────────────────
@app.get("/api/chart-data/{ticker}")
def get_chart_data(ticker: str, days: int = Query(default=60, ge=1, le=252)):
    """N일치 시계열 데이터 (차트용)."""
    try:
        conn = _db()
    except Exception:
        return {"ticker": ticker.upper(), "days": days, "data": [], "signals": []}

    try:
        # 일별 스코어
        rows = conn.execute(
            """SELECT date, close_price, score, s_force, s_div, s_conc, er, atr_pct
               FROM daily_scores
               WHERE ticker = ?
                 AND date >= date('now', ?)
               ORDER BY date ASC""",
            (ticker.upper(), f"-{days} days"),
        ).fetchall()

        # 같은 기간의 시그널
        sig_rows = conn.execute(
            """SELECT peak_date AS date, signal_type, close_price AS entry_price,
                      s_force, peak_val, signal_tier, action_pct
               FROM signal_events
               WHERE ticker = ?
                 AND peak_date >= date('now', ?)
               ORDER BY peak_date ASC""",
            (ticker.upper(), f"-{days} days"),
        ).fetchall()
    finally:
        conn.close()

    signals = []
    for s in _rows_to_dicts(sig_rows):
        s["direction"] = "LONG" if s["signal_type"] == "bottom" else "SHORT"
        signals.append(s)

    return {
        "ticker": ticker.upper(),
        "days": days,
        "data": _rows_to_dicts(rows),
        "signals": signals,
    }


# ── 5. GET /api/stream/scores — SSE 실시간 스트림 ─────────────────────
@app.get("/api/stream/scores")
async def stream_scores(request: Request):
    """Server-Sent Events 스트림.

    클라이언트는 EventSource로 연결하면:
    - 스코어 업데이트 시 "scores_updated" 이벤트 수신
    - 새 시그널 시 "signal_detected" 이벤트 수신
    - 30초마다 heartbeat (연결 유지)
    """

    async def event_generator():
        queue = subscribe()
        try:
            while True:
                # 클라이언트 연결 끊김 감지
                if await request.is_disconnected():
                    break

                try:
                    event = await asyncio.wait_for(queue.get(), timeout=25.0)
                    # SSE 형식: event: type\ndata: json\n\n
                    evt_type = event.get("type", "message")
                    evt_data = json.dumps(event.get("data", {}))
                    yield f"event: {evt_type}\ndata: {evt_data}\n\n"
                except asyncio.TimeoutError:
                    # heartbeat — 연결 유지용 빈 코멘트
                    yield f": heartbeat {subscriber_count()} clients\n\n"
        finally:
            unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx 버퍼링 방지
        },
    )


# ── 6. GET /api/stream/status — SSE 상태 확인 ────────────────────────
@app.get("/api/stream/status")
def stream_status():
    """현재 SSE 구독자 수."""
    return {"subscribers": subscriber_count()}


# ── 7. POST /api/stream/notify — 수동 이벤트 발행 (스캔 트리거용) ─────
@app.post("/api/stream/notify")
def stream_notify(event_type: str = "scores_updated", tickers: str = ""):
    """수동으로 SSE 이벤트 발행.

    사용 예:
        curl -X POST "http://localhost:8000/api/stream/notify?event_type=scores_updated&tickers=NVDA,TSLA"

    외부 프로세스(daily_scan, cron)에서 DB 업데이트 후 이 엔드포인트를 호출하면
    같은 API 서버 프로세스의 이벤트 버스를 통해 SSE 클라이언트에 브로드캐스트됨.
    """
    from v4wp_realtime.api.event_bus import publish

    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    data = {"tickers": ticker_list, "count": len(ticker_list)} if ticker_list else {}
    delivered = publish(event_type, data)
    return {"delivered": delivered, "event_type": event_type, "subscribers": subscriber_count()}
