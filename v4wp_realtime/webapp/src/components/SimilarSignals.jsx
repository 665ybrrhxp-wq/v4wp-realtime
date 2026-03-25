import { useState, useEffect } from "react";
import { fetchSimilarSignals } from "../api";

const mono = "'JetBrains Mono', monospace";

export default function SimilarSignals({ ticker }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetchSimilarSignals(ticker)
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [ticker]);

  if (loading || !data?.similar?.length) return null;

  const similar = data.similar;
  const completed = similar.filter((s) => s.return_90d != null);
  const wins = completed.filter((s) => s.return_90d > 0).length;
  const avgR90 = completed.length
    ? (completed.reduce((a, s) => a + s.return_90d, 0) / completed.length).toFixed(1)
    : null;

  return (
    <div
      style={{
        background: "var(--tg-section-bg)",
        borderRadius: 10,
        padding: "12px 10px 8px",
        border: "1px solid rgba(255,255,255,0.06)",
      }}
    >
      {/* Label */}
      <div
        style={{
          fontSize: 10,
          color: "var(--tg-hint)",
          marginBottom: 8,
          paddingLeft: 2,
          fontWeight: 600,
          letterSpacing: 0.5,
          textTransform: "uppercase",
        }}
      >
        SIMILAR SIGNALS
      </div>

      {/* Summary */}
      {completed.length > 0 && (
        <div
          style={{
            fontSize: 11,
            color: "var(--tg-subtitle)",
            marginBottom: 8,
            padding: "4px 8px",
            borderRadius: 6,
            background: "var(--tg-bg)",
          }}
        >
          {similar.length}건 중 {completed.length}건 확인:{" "}
          <span style={{ color: "#34d399", fontWeight: 700 }}>{wins}W</span>
          <span style={{ color: "#f87171", fontWeight: 700 }}> {completed.length - wins}L</span>
          {avgR90 && (
            <span style={{ fontFamily: mono, marginLeft: 6 }}>
              avg {avgR90 >= 0 ? "+" : ""}{avgR90}%
            </span>
          )}
        </div>
      )}

      {/* Signal Cards */}
      {similar.map((s, i) => (
        <div
          key={i}
          style={{
            padding: "8px 10px",
            borderRadius: 8,
            marginBottom: 4,
            background: "var(--tg-bg)",
            border: "1px solid rgba(255,255,255,0.06)",
          }}
        >
          {/* Header */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 3 }}>
            <span style={{ fontSize: 11, fontWeight: 700, fontFamily: mono, color: "var(--tg-text)" }}>
              {s.ticker} {s.peak_date?.slice(5)}
            </span>
            <span
              style={{
                fontSize: 10,
                fontWeight: 700,
                fontFamily: mono,
                color: "#818cf8",
              }}
            >
              {(s.similarity * 100).toFixed(0)}%
            </span>
          </div>

          {/* Features */}
          <div style={{ display: "flex", gap: 8, fontSize: 10, color: "var(--tg-hint)", marginBottom: 3 }}>
            <span>F:<span style={{ fontFamily: mono, color: s.s_force > 0 ? "#34d399" : "#f87171" }}>{s.s_force?.toFixed(2)}</span></span>
            <span>D:<span style={{ fontFamily: mono, color: s.s_div > 0 ? "#818cf8" : "#f87171" }}>{s.s_div?.toFixed(2)}</span></span>
            <span>Score:<span style={{ fontFamily: mono, color: "var(--tg-text)" }}>{s.peak_val?.toFixed(4)}</span></span>
          </div>

          {/* Result */}
          <div style={{ fontSize: 10, fontFamily: mono }}>
            {s.return_90d != null ? (
              <>
                <span style={{ color: "var(--tg-hint)" }}>90d: </span>
                <span style={{ color: s.return_90d >= 0 ? "#34d399" : "#f87171", fontWeight: 700 }}>
                  {s.return_90d >= 0 ? "+" : ""}{s.return_90d}%
                </span>
                <span
                  style={{
                    marginLeft: 6,
                    fontSize: 9,
                    fontWeight: 800,
                    color: s.outcome === "WIN" ? "#34d399" : "#f87171",
                  }}
                >
                  {s.outcome}
                </span>
              </>
            ) : (
              <span style={{ color: "var(--tg-hint)" }}>90d: pending</span>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
