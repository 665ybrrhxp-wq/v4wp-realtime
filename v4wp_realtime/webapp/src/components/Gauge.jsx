export default function Gauge({ value, min = -1, max = 1, label, color }) {
  const safeVal = value ?? 0;
  const pct = ((safeVal - min) / (max - min)) * 100;
  return (
    <div style={{ marginBottom: 10 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: 11,
          color: "var(--tg-hint)",
          marginBottom: 3,
        }}
      >
        <span>{label}</span>
        <span style={{ color, fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>
          {safeVal.toFixed(3)}
        </span>
      </div>
      <div
        style={{
          height: 5,
          background: "var(--tg-bg)",
          borderRadius: 3,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${Math.min(100, Math.max(0, pct))}%`,
            height: "100%",
            background: color,
            borderRadius: 3,
            transition: "width 0.5s ease",
          }}
        />
      </div>
    </div>
  );
}
