export default function SignalBadge({ direction, tier }) {
  const passed = tier === "CONFIRMED";
  const bg = !passed ? "#2a2a35" : direction === "LONG" ? "#0d3320" : "#3d1320";
  const fg = !passed ? "#555" : direction === "LONG" ? "#34d399" : "#f87171";
  const border = !passed ? "#333" : direction === "LONG" ? "#166534" : "#7f1d1d";
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
        opacity: passed ? 1 : 0.5,
        whiteSpace: "nowrap",
      }}
    >
      {direction}
      {!passed && " \u2715"}
    </span>
  );
}
