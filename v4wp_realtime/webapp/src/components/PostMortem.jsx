import { useState, useEffect } from "react";
import { fetchPostMortem } from "../api";

const mono = "'JetBrains Mono', monospace";

const OUTCOME_STYLE = {
  WIN: { color: "#34d399", bg: "#0d3320" },
  LOSS: { color: "#f87171", bg: "#3d1320" },
  PENDING: { color: "var(--tg-hint)", bg: "var(--tg-secondary-bg)" },
};

const ACCURACY_COLOR = {
  CORRECT: "#34d399",
  OVERCONFIDENT: "#f87171",
  UNDERCONFIDENT: "#fbbf24",
};

export default function PostMortem({ ticker }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetchPostMortem(ticker)
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [ticker]);

  if (loading || !data || data.total_completed === 0) return null;

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
        POST-MORTEM
      </div>

      {/* Summary Stats */}
      <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
        <StatChip
          label="Win Rate"
          value={`${data.win_rate}%`}
          color={data.win_rate >= 60 ? "#34d399" : data.win_rate >= 40 ? "#fbbf24" : "#f87171"}
        />
        <StatChip
          label="Avg 90d"
          value={`${data.avg_return_90d >= 0 ? "+" : ""}${data.avg_return_90d}%`}
          color={data.avg_return_90d >= 0 ? "#34d399" : "#f87171"}
        />
        <StatChip label="Completed" value={`${data.total_completed}`} color="var(--tg-hint)" />
      </div>

      {/* Signal Results */}
      {data.signals.slice(0, 5).map((sig, i) => (
        <SignalResult key={i} sig={sig} />
      ))}
    </div>
  );
}

function SignalResult({ sig }) {
  const outcome = OUTCOME_STYLE[sig.outcome] || OUTCOME_STYLE.PENDING;

  return (
    <div
      style={{
        padding: "8px 10px",
        borderRadius: 8,
        marginBottom: 4,
        background: "var(--tg-bg)",
        border: "1px solid rgba(255,255,255,0.06)",
      }}
    >
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
        <span style={{ fontSize: 11, fontWeight: 700, fontFamily: mono, color: "var(--tg-text)" }}>
          {sig.ticker} {sig.peak_date?.slice(5)}
        </span>
        <span
          style={{
            fontSize: 10,
            fontWeight: 800,
            padding: "1px 6px",
            borderRadius: 4,
            color: outcome.color,
            background: outcome.bg,
          }}
        >
          {sig.outcome}
        </span>
      </div>

      {/* Returns */}
      <div style={{ display: "flex", gap: 10, marginBottom: 4 }}>
        <ReturnChip label="5d" value={sig.return_5d} />
        <ReturnChip label="20d" value={sig.return_20d} />
        <ReturnChip label="90d" value={sig.return_90d} />
        {sig.max_dd_90d != null && (
          <span style={{ fontSize: 10, fontFamily: mono, color: "#f87171" }}>
            MaxDD {sig.max_dd_90d}%
          </span>
        )}
      </div>

      {/* Persona Accuracy */}
      {sig.persona_scores && Object.keys(sig.persona_scores).length > 0 && (
        <div style={{ display: "flex", gap: 8, fontSize: 10, color: "var(--tg-hint)" }}>
          {["force_expert", "div_expert", "chairman"].map((key) => {
            const p = sig.persona_scores[key];
            if (!p) return null;
            const label = key === "force_expert" ? "F" : key === "div_expert" ? "D" : "C";
            const accColor = ACCURACY_COLOR[p.accuracy] || "var(--tg-hint)";
            return (
              <span key={key} style={{ color: "var(--tg-hint)" }}>
                <span style={{ fontWeight: 700, color: "var(--tg-text)" }}>{label}:{p.conviction}</span>{" "}
                <span style={{ color: accColor }}>{p.accuracy === "CORRECT" ? "O" : p.accuracy === "OVERCONFIDENT" ? "X" : "?"}</span>
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}

function ReturnChip({ label, value }) {
  if (value == null) return null;
  const color = value >= 0 ? "#34d399" : "#f87171";
  return (
    <span style={{ fontSize: 10, fontFamily: mono, color: "var(--tg-hint)" }}>
      <span>{label}:</span>{" "}
      <span style={{ color, fontWeight: 600 }}>{value >= 0 ? "+" : ""}{value}%</span>
    </span>
  );
}

function StatChip({ label, value, color }) {
  return (
    <div
      style={{
        flex: 1,
        padding: "5px 6px",
        borderRadius: 6,
        background: "var(--tg-bg)",
        border: "1px solid rgba(255,255,255,0.06)",
        textAlign: "center",
      }}
    >
      <div style={{ fontSize: 8, color: "var(--tg-hint)", textTransform: "uppercase", letterSpacing: 0.3 }}>
        {label}
      </div>
      <div style={{ fontSize: 13, fontWeight: 800, fontFamily: mono, color }}>
        {value}
      </div>
    </div>
  );
}
