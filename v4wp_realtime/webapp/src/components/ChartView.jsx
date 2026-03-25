import { useState, useEffect } from "react";
import {
  ComposedChart, Area, Line, BarChart, Bar, XAxis, YAxis,
  Tooltip, ResponsiveContainer, ReferenceLine, Cell,
} from "recharts";
import { fetchChartData } from "../api";
import SignalBadge from "./SignalBadge";

export default function ChartView({ ticker }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchChartData(ticker, 60)
      .then((res) => { if (!cancelled) setData(res); })
      .catch(() => { if (!cancelled) setData(null); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [ticker]);

  if (loading) return <Loading />;
  if (!data || !data.data || data.data.length === 0) return <Empty text="No chart data" />;

  const chartData = data.data.map((d) => ({
    date: d.date?.slice(5) || d.date,
    close: d.close_price,
    sForce: d.s_force,
  }));

  const forceData = chartData.map((d) => ({
    date: d.date,
    sForce: d.sForce,
    fill: d.sForce >= 0 ? "#34d399" : "#f87171",
  }));

  const signals = data.signals || [];
  const signalDates = new Set(signals.map((s) => s.date?.slice(5)));

  const tooltipStyle = {
    background: "var(--tg-secondary-bg)",
    border: "1px solid rgba(255,255,255,0.1)",
    borderRadius: 8,
    fontSize: 11,
    color: "var(--tg-text)",
  };
  const tooltipLabelStyle = { color: "var(--tg-hint)", fontSize: 10 };
  const tooltipItemStyle = { color: "var(--tg-text)" };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {/* Price chart */}
      <Card label={`PRICE · ${data.days || 60}D`}>
        <ResponsiveContainer width="100%" height={200}>
          <ComposedChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
            <XAxis
              dataKey="date"
              tick={{ fontSize: 9, fill: "var(--tg-hint)" }}
              tickLine={false}
              axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
              interval={Math.floor(chartData.length / 5)}
            />
            <YAxis
              domain={["auto", "auto"]}
              tick={{ fontSize: 9, fill: "var(--tg-hint)" }}
              tickLine={false}
              axisLine={false}
              width={42}
            />
            <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} formatter={(v) => [typeof v === "number" ? v.toFixed(2) : v]} />
            <defs>
              <linearGradient id="priceGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#818cf8" stopOpacity={0.15} />
                <stop offset="100%" stopColor="#818cf8" stopOpacity={0} />
              </linearGradient>
            </defs>
            <Area type="monotone" dataKey="close" fill="url(#priceGrad)" stroke="none" />
            <Line type="monotone" dataKey="close" stroke="#818cf8" strokeWidth={2} dot={false} />
            {chartData
              .filter((d) => signalDates.has(d.date))
              .map((d, i) => (
                <ReferenceLine
                  key={i}
                  x={d.date}
                  stroke="#34d399"
                  strokeDasharray="3 3"
                  strokeWidth={1}
                />
              ))}
          </ComposedChart>
        </ResponsiveContainer>

        {/* Signal legend */}
        {signals.length > 0 && (
          <div style={{ display: "flex", gap: 8, padding: "6px 4px 2px", flexWrap: "wrap" }}>
            {signals.map((s, i) => (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 10, color: "var(--tg-hint)" }}>
                <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#34d399" }} />
                {s.date?.slice(5)}
                <SignalBadge direction={s.direction} tier={s.signal_tier} />
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* s_force bar */}
      <Card label="S_FORCE · PV Momentum">
        <ResponsiveContainer width="100%" height={80}>
          <BarChart data={forceData} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
            <XAxis dataKey="date" tick={false} axisLine={{ stroke: "rgba(255,255,255,0.06)" }} />
            <YAxis
              domain={[-1, 1]}
              tick={{ fontSize: 8, fill: "var(--tg-hint)" }}
              tickLine={false}
              axisLine={false}
              width={24}
              ticks={[-1, 0, 1]}
            />
            <ReferenceLine y={0} stroke="rgba(255,255,255,0.08)" />
            <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} />
            <Bar dataKey="sForce" radius={[2, 2, 0, 0]}>
              {forceData.map((d, i) => (
                <Cell key={i} fill={d.fill} fillOpacity={0.7} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  );
}

function Card({ label, children }) {
  return (
    <div
      style={{
        background: "var(--tg-section-bg)",
        borderRadius: 10,
        padding: "12px 10px 6px",
        border: "1px solid rgba(255,255,255,0.06)",
      }}
    >
      <div
        style={{
          fontSize: 10,
          color: "var(--tg-hint)",
          marginBottom: 6,
          paddingLeft: 2,
          fontWeight: 600,
          letterSpacing: 0.5,
          textTransform: "uppercase",
        }}
      >
        {label}
      </div>
      {children}
    </div>
  );
}

function Loading() {
  return (
    <div style={{ padding: 40, textAlign: "center", color: "var(--tg-hint)", fontSize: 12 }}>
      Loading...
    </div>
  );
}

function Empty({ text }) {
  return (
    <div style={{ padding: 40, textAlign: "center", color: "var(--tg-hint)", fontSize: 12 }}>
      {text}
    </div>
  );
}
