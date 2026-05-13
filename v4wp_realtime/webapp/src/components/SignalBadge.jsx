export default function SignalBadge({ direction, tier, ageDays }) {
  const passed = tier === "CONFIRMED";

  // 31일 초과 신호는 표시하지 않음
  if (ageDays != null && ageDays > 30) return null;

  const recent = ageDays != null && ageDays > 5 && ageDays <= 14;
  const stale = ageDays != null && ageDays > 14;

  const baseOpacity = !passed ? 0.5 : stale ? 0.5 : recent ? 0.75 : 1;
  const bg = !passed || stale
    ? "#2a2a35"
    : direction === "LONG" ? "#0d3320" : "#3d1320";
  const fg = !passed || stale
    ? "#888"
    : direction === "LONG" ? "#34d399" : "#f87171";
  const border = !passed || stale
    ? "#333"
    : direction === "LONG" ? "#166534" : "#7f1d1d";

  return (
    <span
      style={{
        display: "inline-block",
        padding: "2px 6px",
        borderRadius: 4,
        fontSize: 10,
        fontWeight: 700,
        background: bg,
        color: fg,
        border: `1px solid ${border}`,
        letterSpacing: 0.5,
        fontFamily: "'JetBrains Mono', monospace",
        opacity: baseOpacity,
        whiteSpace: "nowrap",
      }}
    >
      {direction}
      {!passed && " \u2715"}
      {ageDays != null && ageDays > 1 && (
        <span style={{ marginLeft: 4, opacity: 0.7, fontSize: 9 }}>
          {ageDays}d
        </span>
      )}
    </span>
  );
}
