import { useState, useEffect } from "react";
import { fetchSignals } from "../api";
import SignalBadge from "./SignalBadge";

const mono = "'JetBrains Mono', monospace";

export default function SignalsView({ ticker }) {
  const [signals, setSignals] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchSignals(ticker, 180)
      .then((res) => { if (!cancelled) setSignals(res); })
      .catch(() => { if (!cancelled) setSignals([]); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [ticker]);

  if (loading) {
    return <div style={{ padding: 40, textAlign: "center", color: "var(--tg-hint)", fontSize: 12 }}>Loading...</div>;
  }

  if (signals.length === 0) {
    return (
      <div style={{ padding: 40, textAlign: "center", color: "var(--tg-hint)", fontSize: 12 }}>
        No signals for {ticker}
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {signals.map((s, i) => (
        <SignalCard key={i} signal={s} />
      ))}
    </div>
  );
}

function SignalCard({ signal }) {
  const s = signal;
  const passed = s.signal_tier === "CONFIRMED";

  return (
    <div
      style={{
        background: "var(--tg-section-bg)",
        borderRadius: 10,
        padding: "12px 14px",
        border: "1px solid rgba(255,255,255,0.06)",
        opacity: passed ? 1 : 0.5,
      }}
    >
      {/* Row 1: date + badge + entry */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontFamily: mono, fontSize: 12, color: "var(--tg-text)" }}>
            {s.date}
          </span>
          <SignalBadge direction={s.direction} tier={s.signal_tier} />
        </div>
        <span style={{ fontFamily: mono, fontSize: 13, fontWeight: 700, color: "var(--tg-text)" }}>
          ${s.close_price?.toFixed(2)}
        </span>
      </div>

      {/* Row 2: indicators grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 6 }}>
        <Metric label="s_force" value={s.s_force?.toFixed(3)} color={s.s_force > 0 ? "#34d399" : "#f87171"} />
        <Metric label="SQZ" value={s.squeeze ? "ON" : "OFF"} color={s.squeeze ? "#fbbf24" : "var(--tg-hint)"} />
        <Metric label="ER" value={s.er?.toFixed(3) ?? "—"} color="var(--tg-link)" />
        <Metric label="ATR%" value={s.atr_pct != null ? `${s.atr_pct.toFixed(1)}%` : "—"} color="#fbbf24" />
      </div>

      {/* Row 3: commentary (if any) */}
      {s.commentary && (
        <div style={{ marginTop: 8, fontSize: 11, color: "var(--tg-hint)", lineHeight: 1.5, borderTop: "1px solid rgba(255,255,255,0.05)", paddingTop: 8 }}>
          {s.commentary}
        </div>
      )}
    </div>
  );
}

function Metric({ label, value, color }) {
  return (
    <div>
      <div style={{ fontSize: 9, color: "var(--tg-hint)", textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 2 }}>
        {label}
      </div>
      <div style={{ fontFamily: mono, fontSize: 12, fontWeight: 600, color }}>
        {value}
      </div>
    </div>
  );
}
