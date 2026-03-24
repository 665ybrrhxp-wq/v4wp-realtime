import { useState, useEffect } from "react";
import { fetchIndicators } from "../api";
import Gauge from "./Gauge";

const mono = "'JetBrains Mono', monospace";

export default function IndicatorsView({ ticker }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchIndicators(ticker)
      .then((res) => { if (!cancelled) setData(res); })
      .catch(() => { if (!cancelled) setData(null); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [ticker]);

  if (loading) {
    return <div style={{ padding: 40, textAlign: "center", color: "var(--tg-hint)", fontSize: 12 }}>Loading...</div>;
  }
  if (!data?.data) {
    return <div style={{ padding: 40, textAlign: "center", color: "var(--tg-hint)", fontSize: 12 }}>No indicator data</div>;
  }

  const sForce = data.s_force ?? 0;
  const sDiv = data.s_div ?? 0;
  const score = data.score ?? 0;
  const andGeo = data.and_geo_active;
  const filters = data.filters || [];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {/* V4_wP Score (AND-GEO) */}
      <Card>
        <div
          style={{
            textAlign: "center",
            padding: "14px 0 10px",
          }}
        >
          <div style={{ fontSize: 10, color: "var(--tg-hint)", textTransform: "uppercase", letterSpacing: 1, marginBottom: 6 }}>
            V4_wP Score
          </div>
          <div
            style={{
              fontSize: 36,
              fontWeight: 800,
              fontFamily: mono,
              color: score > 0 ? "#34d399" : score < 0 ? "#f87171" : "var(--tg-hint)",
              lineHeight: 1,
            }}
          >
            {score.toFixed(3)}
          </div>
          <div style={{ fontSize: 10, color: "var(--tg-hint)", marginTop: 6 }}>
            = √(s_force × s_div)
          </div>
        </div>

        {/* AND-GEO 상태 */}
        <div
          style={{
            margin: "4px 0 2px",
            padding: "8px 12px",
            borderRadius: 6,
            textAlign: "center",
            fontSize: 11,
            fontWeight: 700,
            background: andGeo ? "#0d3320" : "var(--tg-secondary-bg)",
            color: andGeo ? "#34d399" : "var(--tg-hint)",
            border: `1px solid ${andGeo ? "#166534" : "rgba(255,255,255,0.06)"}`,
          }}
        >
          AND-GEO: {andGeo ? "ACTIVE (s_force > 0 AND s_div > 0)" : "INACTIVE"}
        </div>
      </Card>

      {/* s_force + s_div Gauges */}
      <Card>
        <SectionLabel>Sub-Indicators</SectionLabel>
        <Gauge
          value={sForce}
          min={-1}
          max={1}
          label="s_force (PV Momentum)"
          color={sForce > 0 ? "#34d399" : "#f87171"}
        />
        <div style={{ fontSize: 9, color: "var(--tg-hint)", padding: "2px 4px 8px", lineHeight: 1.4 }}>
          거래량 × 가격가속도 MACD — 양수면 매수 모멘텀
        </div>

        <Gauge
          value={sDiv}
          min={-1}
          max={1}
          label="s_div (PV Divergence)"
          color={sDiv > 0 ? "#818cf8" : "#f87171"}
        />
        <div style={{ fontSize: 9, color: "var(--tg-hint)", padding: "2px 4px 4px", lineHeight: 1.4 }}>
          가격-거래량 다이버전스 — 양수면 거래량이 가격보다 강세
        </div>
      </Card>

      {/* 가격 필터 (ER, ATR%) — 값이 있을 때만 */}
      {filters.length > 0 && (
        <Card>
          <SectionLabel>Price Filters</SectionLabel>
          {filters.map((f, i) => (
            <div
              key={i}
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "8px 0",
                borderBottom: i < filters.length - 1 ? "1px solid rgba(255,255,255,0.05)" : "none",
              }}
            >
              <div>
                <div style={{ fontSize: 12, color: "var(--tg-subtitle)" }}>{f.label}</div>
                <div style={{ fontSize: 9, color: "var(--tg-hint)", marginTop: 1 }}>{f.desc}</div>
              </div>
              <span
                style={{
                  fontSize: 13,
                  fontWeight: 700,
                  fontFamily: mono,
                  color: "var(--tg-text)",
                }}
              >
                {f.value}
              </span>
            </div>
          ))}
        </Card>
      )}
    </div>
  );
}

function Card({ children }) {
  return (
    <div
      style={{
        background: "var(--tg-section-bg)",
        borderRadius: 10,
        padding: 14,
        border: "1px solid rgba(255,255,255,0.06)",
      }}
    >
      {children}
    </div>
  );
}

function SectionLabel({ children }) {
  return (
    <div
      style={{
        fontSize: 10,
        color: "var(--tg-hint)",
        fontWeight: 600,
        letterSpacing: 0.5,
        textTransform: "uppercase",
        marginBottom: 12,
      }}
    >
      {children}
    </div>
  );
}
