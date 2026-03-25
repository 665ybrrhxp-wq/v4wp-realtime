import { useState, useEffect } from "react";
import { fetchInterpretation } from "../api";

const mono = "'JetBrains Mono', monospace";

const PERSONA_CONFIG = {
  force_expert: {
    icon: "F",
    gradient: "linear-gradient(135deg, #34d399, #059669)",
    accentColor: "#34d399",
  },
  div_expert: {
    icon: "D",
    gradient: "linear-gradient(135deg, #818cf8, #6366f1)",
    accentColor: "#818cf8",
  },
  chairman: {
    icon: "C",
    gradient: "linear-gradient(135deg, #fbbf24, #f59e0b)",
    accentColor: "#fbbf24",
  },
};

const VERDICT_STYLE = {
  STRONG_BUY: { label: "STRONG BUY", color: "#34d399", bg: "#0d3320", border: "#16653440" },
  BUY: { label: "BUY", color: "#34d399", bg: "#0d3320", border: "#16653440" },
  CAUTIOUS_BUY: { label: "CAUTIOUS", color: "#fbbf24", bg: "#422006", border: "#92400e40" },
  HOLD: { label: "HOLD", color: "#f87171", bg: "#3d1320", border: "#7f1d1d40" },
};

export default function AIInterpretation({ ticker }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    setLoading(true);
    setExpanded(false);
    fetchInterpretation(ticker)
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [ticker]);

  if (loading || !data?.interpretation) return null;

  const interp = data.interpretation;
  const verdict = VERDICT_STYLE[interp.final_verdict] || VERDICT_STYLE.HOLD;

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
        AI SIGNAL INTERPRETATION
        {data.peak_date && (
          <span style={{ marginLeft: 6, opacity: 0.7 }}>{data.peak_date.slice(5)}</span>
        )}
      </div>

      {/* Verdict Banner */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "8px 10px",
          borderRadius: 8,
          background: verdict.bg,
          border: `1px solid ${verdict.border}`,
          marginBottom: 8,
        }}
      >
        <span style={{ fontSize: 13, fontWeight: 800, color: verdict.color, fontFamily: mono }}>
          {verdict.label}
        </span>
        <span style={{ fontSize: 11, color: "var(--tg-hint)", fontFamily: mono }}>
          Confidence {interp.confidence_score}%
        </span>
      </div>

      {/* Chairman (always visible) */}
      <PersonaCard persona={interp.chairman} config={PERSONA_CONFIG.chairman} />

      {/* Expand toggle */}
      <button
        onClick={() => setExpanded((v) => !v)}
        style={{
          width: "100%",
          padding: "6px 0",
          margin: "4px 0",
          background: "none",
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 6,
          color: "var(--tg-hint)",
          fontSize: 11,
          fontWeight: 600,
          cursor: "pointer",
        }}
      >
        {expanded ? "접기" : "전문가 의견 보기"} {expanded ? "\u25B2" : "\u25BC"}
      </button>

      {/* Experts (expandable) */}
      {expanded && (
        <>
          <PersonaCard persona={interp.force_expert} config={PERSONA_CONFIG.force_expert} />
          <PersonaCard persona={interp.div_expert} config={PERSONA_CONFIG.div_expert} />
        </>
      )}

      {/* Risk Note */}
      {interp.risk_note && (
        <div
          style={{
            marginTop: 6,
            padding: "6px 8px",
            borderRadius: 6,
            background: "rgba(248,113,113,0.06)",
            border: "1px solid rgba(248,113,113,0.12)",
            fontSize: 11,
            color: "#f87171",
            lineHeight: 1.5,
          }}
        >
          {interp.risk_note}
        </div>
      )}
    </div>
  );
}

function PersonaCard({ persona, config }) {
  if (!persona) return null;

  return (
    <div
      style={{
        padding: "10px 12px",
        borderRadius: 8,
        marginBottom: 6,
        background: "var(--tg-bg)",
        border: "1px solid rgba(255,255,255,0.06)",
      }}
    >
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
        <div
          style={{
            width: 22,
            height: 22,
            borderRadius: 6,
            background: config.gradient,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 11,
            fontWeight: 800,
            color: "#fff",
            flexShrink: 0,
          }}
        >
          {config.icon}
        </div>
        <span style={{ fontSize: 11, fontWeight: 700, color: config.accentColor }}>
          {persona.persona_name}
        </span>
        <ConvictionDots level={persona.conviction} color={config.accentColor} />
      </div>

      {/* Analysis */}
      <div
        style={{ fontSize: 12, color: "var(--tg-text)", lineHeight: 1.7 }}
        dangerouslySetInnerHTML={{ __html: formatAnalysis(persona.analysis) }}
      />

      {/* Key Point */}
      <div
        style={{
          marginTop: 6,
          fontSize: 10,
          fontWeight: 600,
          color: config.accentColor,
          opacity: 0.8,
        }}
      >
        {persona.key_point}
      </div>
    </div>
  );
}

function ConvictionDots({ level, color }) {
  return (
    <div style={{ display: "flex", gap: 3, marginLeft: "auto" }}>
      {[1, 2, 3, 4, 5].map((i) => (
        <div
          key={i}
          style={{
            width: 6,
            height: 6,
            borderRadius: "50%",
            background: i <= level ? color : "rgba(255,255,255,0.1)",
          }}
        />
      ))}
    </div>
  );
}

function formatAnalysis(text) {
  if (!text) return "";
  return text
    .replace(/\*\*\[([^\]]+)\]\*\*/g, '<span style="color:var(--tg-hint);font-size:10px;font-weight:700;letter-spacing:0.3px">[$1]</span>')
    .replace(/\n/g, "<br/>");
}
