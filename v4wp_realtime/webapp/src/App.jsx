import { useState, useEffect, useCallback, useRef } from "react";
import { initTelegramApp, isTelegram, getStartParam, showBackButton, hideBackButton, haptic } from "./telegram";
import { fetchWatchlist, fetchChartData, fetchSparkline, connectSSE } from "./api";
import WatchlistBar from "./components/WatchlistBar";
import TickerDetail from "./components/TickerDetail";

export default function App() {
  const [watchlist, setWatchlist] = useState([]);
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [sparkCache, setSparkCache] = useState({});
  const initDone = useRef(false);

  // ── 초기화 ──
  useEffect(() => {
    if (initDone.current) return;
    initDone.current = true;

    initTelegramApp();

    const start = getStartParam();
    if (start?.ticker) {
      setSelected(start.ticker);
    }

    fetchWatchlist()
      .then((data) => {
        setWatchlist(data);
        if (!start?.ticker && data.length > 0) {
          setSelected(data[0].ticker);
        }
        loadSparklines(data.map((d) => d.ticker));
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  // ── SSE 실시간 업데이트 ──
  const [lastUpdate, setLastUpdate] = useState(0);

  useEffect(() => {
    const sse = connectSSE((eventType, data) => {
      if (eventType === "scores_updated") {
        fetchWatchlist().then(setWatchlist).catch(() => {});
        if (data?.tickers) {
          data.tickers.forEach((t) => {
            fetchChartData(t, 20)
              .then((res) => {
                if (res?.data) {
                  setSparkCache((prev) => ({
                    ...prev,
                    [t]: res.data.map((d) => d.close_price).filter(Boolean),
                  }));
                }
              })
              .catch(() => {});
          });
        }
        setLastUpdate(Date.now());
      } else if (eventType === "signal_detected") {
        fetchWatchlist().then(setWatchlist).catch(() => {});
        setLastUpdate(Date.now());
      }
    });
    return () => sse.close();
  }, []);

  // ── 스파크라인 ──
  async function loadSparklines(tickers) {
    const cache = {};
    for (let i = 0; i < tickers.length; i += 5) {
      const batch = tickers.slice(i, i + 5);
      const results = await Promise.allSettled(
        batch.map((t) =>
          fetchSparkline(t)
            .then((prices) => ({ type: "spark", data: prices }))
            .catch(() =>
              fetchChartData(t, 20).then((res) => ({
                type: "chart",
                data: res?.data?.map((d) => d.close_price).filter(Boolean) || [],
              }))
            )
        )
      );
      results.forEach((r, idx) => {
        if (r.status === "fulfilled" && r.value?.data?.length) {
          cache[batch[idx]] = r.value.data;
        }
      });
    }
    setSparkCache(cache);
  }

  // ── 종목 선택 ──
  const handleSelect = useCallback((ticker) => {
    haptic("impact");
    setSelected(ticker);
  }, []);

  // ── Telegram BackButton ──
  useEffect(() => {
    if (!isTelegram || !selected) return;
    showBackButton(() => {
      haptic("impact");
      if (watchlist.length > 0) {
        setSelected(watchlist[0].ticker);
      }
    });
    return () => hideBackButton();
  }, [selected, watchlist]);

  const current = watchlist.find((w) => w.ticker === selected);

  if (loading) {
    return (
      <Shell>
        <div style={{ padding: 60, textAlign: "center", color: "var(--tg-hint)" }}>
          <Spinner /> Loading...
        </div>
      </Shell>
    );
  }

  if (error) {
    return (
      <Shell>
        <div style={{ padding: 40, textAlign: "center", color: "var(--tg-destructive)", fontSize: 13 }}>
          {error}
        </div>
      </Shell>
    );
  }

  return (
    <Shell>
      {/* ── Header ── */}
      <div
        style={{
          padding: "12px 14px",
          background: "var(--tg-header-bg)",
          display: "flex",
          alignItems: "center",
          gap: 10,
          borderBottom: "1px solid rgba(255,255,255,0.06)",
        }}
      >
        <div
          style={{
            width: 28,
            height: 28,
            borderRadius: 7,
            background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 13,
            fontWeight: 800,
            color: "#fff",
            flexShrink: 0,
          }}
        >
          V4
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 14, fontWeight: 700, color: "var(--tg-text)", letterSpacing: -0.3 }}>
            V4_wP Signal Dashboard
          </div>
          <div style={{ fontSize: 10, color: "var(--tg-hint)", marginTop: 1 }}>
            Rev2 · Walk-Forward Validated
          </div>
        </div>
        <div style={{ display: "flex", gap: 12, flexShrink: 0 }}>
          <StatChip label="Tickers" value={watchlist.length} color="var(--tg-accent)" />
        </div>
      </div>

      {/* ── Watchlist Bar ── */}
      <WatchlistBar
        items={watchlist}
        selected={selected}
        onSelect={handleSelect}
        chartCache={sparkCache}
      />

      {/* ── Ticker Header ── */}
      {current && (
        <div
          style={{
            padding: "10px 14px",
            borderBottom: "1px solid rgba(255,255,255,0.06)",
          }}
        >
          <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
            <span style={{ fontSize: 20, fontWeight: 800, color: "var(--tg-text)", letterSpacing: -0.5 }}>
              {selected}
            </span>
            {current.close_price != null && (
              <span
                style={{
                  fontSize: 18,
                  fontWeight: 600,
                  fontFamily: "'JetBrains Mono', monospace",
                  color: "var(--tg-text)",
                }}
              >
                ${current.close_price.toFixed(2)}
              </span>
            )}
            {current.change_pct != null && (
              <span
                style={{
                  fontSize: 13,
                  fontWeight: 600,
                  fontFamily: "'JetBrains Mono', monospace",
                  color: current.change_pct >= 0 ? "#34d399" : "#f87171",
                }}
              >
                {current.change_pct >= 0 ? "▲" : "▼"}{" "}
                {current.change_pct >= 0 ? "+" : ""}
                {current.change_pct.toFixed(2)}%
              </span>
            )}
          </div>
        </div>
      )}

      {/* ── Content (single scroll) ── */}
      <div style={{ flex: 1, padding: "10px 12px", overflow: "auto", paddingBottom: 40 }}>
        {selected && <TickerDetail key={`detail-${selected}-${lastUpdate}`} ticker={selected} />}
      </div>
    </Shell>
  );
}

function Shell({ children }) {
  return (
    <div
      style={{
        minHeight: "100vh",
        background: "var(--tg-bg)",
        color: "var(--tg-text)",
        fontFamily: "'Inter', -apple-system, sans-serif",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {children}
    </div>
  );
}

function StatChip({ label, value, color }) {
  return (
    <div style={{ textAlign: "center" }}>
      <div style={{ fontSize: 9, color: "var(--tg-hint)", textTransform: "uppercase", letterSpacing: 0.8 }}>
        {label}
      </div>
      <div style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, fontSize: 13, color }}>
        {value}
      </div>
    </div>
  );
}

function Spinner() {
  return (
    <span
      style={{
        display: "inline-block",
        width: 16,
        height: 16,
        border: "2px solid var(--tg-hint)",
        borderTopColor: "var(--tg-btn)",
        borderRadius: "50%",
        animation: "spin 0.6s linear infinite",
        marginRight: 8,
        verticalAlign: "middle",
      }}
    />
  );
}
