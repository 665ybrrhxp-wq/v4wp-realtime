import { useState, useEffect, useRef } from "react";
import {
  ComposedChart, Area, Line, BarChart, Bar, XAxis, YAxis,
  Tooltip, ReferenceLine, Cell,
} from "recharts";
import { fetchChartData, fetchIndicators } from "../api";
import Gauge from "./Gauge";
import SignalBadge from "./SignalBadge";
import AIInterpretation from "./AIInterpretation";
import PostMortem from "./PostMortem";
import SimilarSignals from "./SimilarSignals";

const mono = "'JetBrains Mono', monospace";
const PPD = 10; // pixels per data point

const tooltipStyle = {
  background: "var(--tg-secondary-bg)",
  border: "1px solid rgba(255,255,255,0.1)",
  borderRadius: 8,
  fontSize: 11,
  color: "var(--tg-text)",
};
const tooltipLabelStyle = { color: "var(--tg-hint)", fontSize: 10 };
const tooltipItemStyle = { color: "var(--tg-text)" };

export default function TickerDetail({ ticker }) {
  const [chart, setChart] = useState(null);
  const [ind, setInd] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);

    Promise.all([
      fetchChartData(ticker, 120),
      fetchIndicators(ticker),
    ])
      .then(([chartRes, indRes]) => {
        if (cancelled) return;
        setChart(chartRes);
        setInd(indRes);
      })
      .catch(() => {
        if (!cancelled) { setChart(null); setInd(null); }
      })
      .finally(() => { if (!cancelled) setLoading(false); });

    return () => { cancelled = true; };
  }, [ticker]);

  if (loading) return <Msg>Loading...</Msg>;
  if (!chart?.data?.length) return <Msg>No data for {ticker}</Msg>;

  const days = chart.data;
  const signals = chart.signals || [];
  const score = ind?.score ?? 0;
  const sForce = ind?.s_force ?? 0;
  const sDiv = ind?.s_div ?? 0;
  const andGeo = ind?.and_geo_active ?? false;
  const pipe = ind?.pipeline;

  // 차트용 데이터 매핑
  const priceData = days.map((d) => ({
    date: d.date?.slice(5),
    close: d.close_price,
  }));

  const forceData = days.map((d) => ({
    date: d.date?.slice(5),
    val: d.s_force,
  }));

  const divData = days.map((d) => ({
    date: d.date?.slice(5),
    val: d.s_div,
  }));

  const signalDates = new Set(signals.map((s) => s.date?.slice(5)));
  // 고정 폭: 데이터 포인트 × PPD + Y축 여백
  const chartW = days.length * PPD + 50;
  const xInterval = Math.max(Math.floor(days.length / 8) - 1, 0);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {/* ── 1. Score + AND-GEO ── */}
      <Card>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div style={{ textAlign: "center", flexShrink: 0, minWidth: 90 }}>
            <div style={{ fontSize: 9, color: "var(--tg-hint)", textTransform: "uppercase", letterSpacing: 0.8 }}>
              V4_wP Score
            </div>
            <div
              style={{
                fontSize: 30,
                fontWeight: 800,
                fontFamily: mono,
                color: score > 0 ? "#34d399" : score < 0 ? "#f87171" : "var(--tg-hint)",
                lineHeight: 1.2,
              }}
            >
              {score.toFixed(3)}
            </div>
            <div
              style={{
                marginTop: 4,
                fontSize: 9,
                fontWeight: 700,
                padding: "2px 8px",
                borderRadius: 4,
                display: "inline-block",
                background: andGeo ? "#0d3320" : "var(--tg-secondary-bg)",
                color: andGeo ? "#34d399" : "var(--tg-hint)",
                border: `1px solid ${andGeo ? "#166534" : "rgba(255,255,255,0.06)"}`,
              }}
            >
              AND-GEO {andGeo ? "ON" : "OFF"}
            </div>
          </div>

          <div style={{ flex: 1, minWidth: 0 }}>
            <Gauge value={sForce} min={-1} max={1} label="s_force" color={sForce > 0 ? "#34d399" : "#f87171"} />
            <Gauge value={sDiv} min={-1} max={1} label="s_div" color={sDiv > 0 ? "#818cf8" : "#f87171"} />
          </div>
        </div>
      </Card>

      {/* ── 1b. Signal Pipeline ── */}
      {pipe && andGeo && (
        <div style={{ display: "flex", gap: 6 }}>
          <PipeChip
            ok={pipe.above_threshold}
            label="Threshold"
            detail={pipe.above_threshold ? `${score.toFixed(3)} > ${pipe.threshold}` : `${score.toFixed(3)} < ${pipe.threshold}`}
          />
          <PipeChip
            ok={pipe.duration_ok}
            label="Duration"
            detail={`${pipe.streak_days}d / ${pipe.confirm_days}d`}
          />
          <PipeChip
            ok={pipe.dd_ok}
            label="DD Gate"
            detail={`-${pipe.dd_pct}% / -${pipe.dd_threshold}%`}
          />
        </div>
      )}

      {/* ── 1c. AI Signal Interpretation ── */}
      <AIInterpretation ticker={ticker} />

      {/* ── 1d. Post-Mortem ── */}
      <PostMortem ticker={ticker} />

      {/* ── 1e. Similar Signals ── */}
      <SimilarSignals ticker={ticker} />

      {/* ── 2. Price Chart (좌우 스크롤) ── */}
      <Card label={`PRICE · ${days.length}D`}>
        <ScrollBox>
          <ComposedChart width={chartW} height={200} data={priceData} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
            <XAxis
              dataKey="date"
              tick={{ fontSize: 9, fill: "var(--tg-hint)" }}
              tickLine={false}
              axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
              interval={xInterval}
            />
            <YAxis
              domain={["auto", "auto"]}
              tick={{ fontSize: 9, fill: "var(--tg-hint)" }}
              tickLine={false}
              axisLine={false}
              width={38}
            />
            <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} formatter={(v) => [typeof v === "number" ? `$${v.toFixed(2)}` : v]} />
            <defs>
              <linearGradient id="priceGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#818cf8" stopOpacity={0.15} />
                <stop offset="100%" stopColor="#818cf8" stopOpacity={0} />
              </linearGradient>
            </defs>
            <Area type="monotone" dataKey="close" fill="url(#priceGrad)" stroke="none" />
            <Line type="monotone" dataKey="close" stroke="#818cf8" strokeWidth={1.5} dot={false} />
            {priceData
              .filter((d) => signalDates.has(d.date))
              .map((d, i) => (
                <ReferenceLine key={i} x={d.date} stroke="#34d399" strokeDasharray="3 3" strokeWidth={1} />
              ))}
          </ComposedChart>
        </ScrollBox>

        {signals.length > 0 && (
          <div style={{ display: "flex", gap: 8, padding: "6px 4px 0", flexWrap: "wrap" }}>
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

      {/* ── 3. s_force 시계열 (좌우 스크롤) ── */}
      <Card label="S_FORCE · PV Momentum">
        <ScrollBox>
          <BarChart width={chartW} height={80} data={forceData} margin={{ top: 2, right: 8, bottom: 0, left: 0 }}>
            <XAxis dataKey="date" tick={false} axisLine={{ stroke: "rgba(255,255,255,0.06)" }} />
            <YAxis domain={[-1, 1]} tick={{ fontSize: 8, fill: "var(--tg-hint)" }} tickLine={false} axisLine={false} width={20} ticks={[-1, 0, 1]} />
            <ReferenceLine y={0} stroke="rgba(255,255,255,0.08)" />
            <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} />
            <Bar dataKey="val" radius={[1, 1, 0, 0]}>
              {forceData.map((d, i) => (
                <Cell key={i} fill={d.val >= 0 ? "#34d399" : "#f87171"} fillOpacity={0.7} />
              ))}
            </Bar>
          </BarChart>
        </ScrollBox>
      </Card>

      {/* ── 4. s_div 시계열 (좌우 스크롤) ── */}
      <Card label="S_DIV · PV Divergence">
        <ScrollBox>
          <BarChart width={chartW} height={80} data={divData} margin={{ top: 2, right: 8, bottom: 0, left: 0 }}>
            <XAxis dataKey="date" tick={false} axisLine={{ stroke: "rgba(255,255,255,0.06)" }} />
            <YAxis domain={[-1, 1]} tick={{ fontSize: 8, fill: "var(--tg-hint)" }} tickLine={false} axisLine={false} width={20} ticks={[-1, 0, 1]} />
            <ReferenceLine y={0} stroke="rgba(255,255,255,0.08)" />
            <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} />
            <Bar dataKey="val" radius={[1, 1, 0, 0]}>
              {divData.map((d, i) => (
                <Cell key={i} fill={d.val >= 0 ? "#818cf8" : "#f87171"} fillOpacity={0.7} />
              ))}
            </Bar>
          </BarChart>
        </ScrollBox>
      </Card>

      {/* ── 5. Signal History ── */}
      {signals.length > 0 && (
        <Card label={`SIGNALS · ${signals.length}`}>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {signals.map((s, i) => (
              <SignalRow key={i} signal={s} />
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}

/* ── ScrollBox: 좌우 스크롤 컨테이너 (최신 데이터가 먼저 보이도록 오른쪽 끝으로 스크롤) ── */
function ScrollBox({ children }) {
  const ref = useRef(null);
  useEffect(() => {
    if (ref.current) {
      ref.current.scrollLeft = ref.current.scrollWidth;
    }
  }, [children]);
  return (
    <div
      ref={ref}
      style={{
        overflowX: "auto",
        WebkitOverflowScrolling: "touch",
      }}
    >
      {children}
    </div>
  );
}

/* ── Signal Row (compact) ── */
function SignalRow({ signal }) {
  const s = signal;
  const confirmed = s.signal_tier === "CONFIRMED";
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "8px 10px",
        borderRadius: 8,
        background: "var(--tg-bg)",
        opacity: confirmed ? 1 : 0.5,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ fontFamily: mono, fontSize: 11, color: "var(--tg-subtitle)" }}>
          {s.date}
        </span>
        <SignalBadge direction={s.direction} tier={s.signal_tier} />
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <Metric label="force" value={s.s_force?.toFixed(2)} color={s.s_force > 0 ? "#34d399" : "#f87171"} />
        <span style={{ fontFamily: mono, fontSize: 12, fontWeight: 700, color: "var(--tg-text)" }}>
          ${s.entry_price?.toFixed(2) ?? "—"}
        </span>
      </div>
    </div>
  );
}

function Metric({ label, value, color }) {
  return (
    <div style={{ textAlign: "right" }}>
      <div style={{ fontSize: 8, color: "var(--tg-hint)", textTransform: "uppercase" }}>{label}</div>
      <div style={{ fontFamily: mono, fontSize: 11, fontWeight: 600, color }}>{value}</div>
    </div>
  );
}

/* ── Shared UI ── */
function Card({ label, children }) {
  return (
    <div
      style={{
        background: "var(--tg-section-bg)",
        borderRadius: 10,
        padding: label ? "12px 10px 8px" : 14,
        border: "1px solid rgba(255,255,255,0.06)",
      }}
    >
      {label && (
        <div style={{ fontSize: 10, color: "var(--tg-hint)", marginBottom: 8, paddingLeft: 2, fontWeight: 600, letterSpacing: 0.5, textTransform: "uppercase" }}>
          {label}
        </div>
      )}
      {children}
    </div>
  );
}

function PipeChip({ ok, label, detail }) {
  return (
    <div
      style={{
        flex: 1,
        padding: "6px 8px",
        borderRadius: 8,
        background: "var(--tg-section-bg)",
        border: `1px solid ${ok ? "#166534" : "rgba(255,255,255,0.06)"}`,
        textAlign: "center",
      }}
    >
      <div style={{ fontSize: 9, fontWeight: 700, color: ok ? "#34d399" : "#f87171", marginBottom: 2 }}>
        {ok ? "PASS" : "FAIL"} {label}
      </div>
      <div style={{ fontSize: 9, fontFamily: mono, color: "var(--tg-hint)" }}>
        {detail}
      </div>
    </div>
  );
}

function Msg({ children }) {
  return <div style={{ padding: 40, textAlign: "center", color: "var(--tg-hint)", fontSize: 12 }}>{children}</div>;
}
