/**
 * V4_wP Signal API client
 *
 * 모드 자동 감지:
 *   - VITE_API_BASE_URL 설정 시 → 라이브 API (개발/서버 모드)
 *   - 미설정 시 → 정적 JSON 파일 (GitHub Pages 모드)
 */

const BASE = import.meta.env.VITE_API_BASE_URL || "";
const IS_STATIC = !BASE;

async function fetchJSON(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`API ${res.status}: ${path}`);
  return res.json();
}

/** 워치리스트 */
export function fetchWatchlist() {
  if (IS_STATIC) return fetchJSON("./data/watchlist.json");
  return fetchJSON("/api/watchlist");
}

/** 종목별 차트 데이터 */
export function fetchChartData(ticker, days = 60) {
  if (IS_STATIC) return fetchJSON(`./data/chart/${ticker.toUpperCase()}.json`);
  return fetchJSON(`/api/chart-data/${ticker}?days=${days}`);
}

/** 종목별 최신 지표 상태 */
export function fetchIndicators(ticker) {
  if (IS_STATIC) return fetchJSON(`./data/indicators/${ticker.toUpperCase()}.json`);
  return fetchJSON(`/api/indicators/${ticker}`);
}

/** 스파크라인 데이터 (정적 모드용) */
export function fetchSparkline(ticker) {
  return fetchJSON(`./data/spark/${ticker.toUpperCase()}.json`);
}

// ── SSE (라이브 모드 전용) ────────────────────────────────────────────

/**
 * SSE 스트림 연결 (정적 모드에서는 no-op)
 */
export function connectSSE(onEvent) {
  if (IS_STATIC) {
    return { close() {} };
  }

  const url = `${BASE}/api/stream/scores`;
  let es = null;
  let retryTimer = null;
  let closed = false;

  function connect() {
    if (closed) return;

    es = new EventSource(url);

    es.addEventListener("scores_updated", (e) => {
      try { onEvent("scores_updated", JSON.parse(e.data)); } catch {}
    });

    es.addEventListener("signal_detected", (e) => {
      try { onEvent("signal_detected", JSON.parse(e.data)); } catch {}
    });

    es.onerror = () => {
      es.close();
      if (!closed) {
        retryTimer = setTimeout(connect, 5000);
      }
    };
  }

  connect();

  return {
    close() {
      closed = true;
      if (retryTimer) clearTimeout(retryTimer);
      if (es) es.close();
    },
  };
}
