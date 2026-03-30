import { useState, useMemo, useEffect } from "react";
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, Cell, Area, ComposedChart } from "recharts";

// ── Mock Data Generator ──
function generateMockData() {
  const tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "AVGO", "CRM"];
  const now = Date.now();
  const DAY = 86400000;

  const priceData = {};
  const signals = [];

  tickers.forEach((ticker) => {
    const base = 100 + Math.random() * 400;
    const days = [];
    let price = base;

    for (let i = 60; i >= 0; i--) {
      const vol = (0.5 + Math.random()) * 1e6;
      const change = (Math.random() - 0.48) * price * 0.025;
      price = Math.max(price * 0.7, price + change);
      const o = price + (Math.random() - 0.5) * 3;
      const c = price;
      const h = Math.max(o, c) + Math.random() * 4;
      const l = Math.min(o, c) - Math.random() * 4;

      const sForce = Math.sin(i * 0.15 + tickers.indexOf(ticker)) * 0.6 + (Math.random() - 0.5) * 0.5;
      const squeeze = Math.abs(sForce) > 0.5 ? (sForce > 0 ? 1 : -1) : 0;
      const er = 0.2 + Math.random() * 0.6;
      const atrPct = 1.5 + Math.random() * 4;

      days.push({
        date: new Date(now - i * DAY).toISOString().slice(5, 10),
        dayIndex: 60 - i,
        open: +o.toFixed(2),
        close: +c.toFixed(2),
        high: +h.toFixed(2),
        low: +l.toFixed(2),
        volume: Math.round(vol),
        sForce: +sForce.toFixed(3),
        squeeze,
        er: +er.toFixed(3),
        atrPct: +atrPct.toFixed(2),
      });
    }
    priceData[ticker] = days;

    // Generate 1-3 signals per ticker
    const numSignals = 1 + Math.floor(Math.random() * 3);
    for (let s = 0; s < numSignals; s++) {
      const dayIdx = 10 + Math.floor(Math.random() * 45);
      const d = days[dayIdx];
      if (!d) continue;
      const direction = d.sForce > 0 ? "LONG" : "SHORT";
      const mfe = +(1 + Math.random() * 6).toFixed(2);
      const mae = +(0.3 + Math.random() * 2.5).toFixed(2);
      signals.push({
        ticker,
        date: d.date,
        dayIndex: dayIdx,
        direction,
        sForce: d.sForce,
        squeeze: d.squeeze,
        er: d.er,
        atrPct: d.atrPct,
        entryPrice: d.close,
        mfe,
        mae,
        pnl: direction === "LONG" ? +(mfe - mae + Math.random() * 2).toFixed(2) : +(mfe - mae + Math.random() * 2).toFixed(2),
        passed: Math.random() > 0.3,
      });
    }
  });

  return { tickers, priceData, signals };
}

// ── Tiny Sparkline ──
function Sparkline({ data, color, height = 32, width = 90 }) {
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const points = data.map((v, i) => `${(i / (data.length - 1)) * width},${height - ((v - min) / range) * height}`).join(" ");
  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" />
    </svg>
  );
}

// ── Gauge Component ──
function Gauge({ value, min = -1, max = 1, label, color }) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#8a8f98", marginBottom: 3 }}>
        <span>{label}</span>
        <span style={{ color, fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>{value.toFixed(3)}</span>
      </div>
      <div style={{ height: 4, background: "#1e2028", borderRadius: 2, overflow: "hidden" }}>
        <div style={{ width: `${Math.min(100, Math.max(0, pct))}%`, height: "100%", background: color, borderRadius: 2, transition: "width 0.5s ease" }} />
      </div>
    </div>
  );
}

// ── Signal Badge ──
function SignalBadge({ direction, passed }) {
  const bg = !passed ? "#2a2a35" : direction === "LONG" ? "#0d3320" : "#3d1320";
  const fg = !passed ? "#555" : direction === "LONG" ? "#34d399" : "#f87171";
  const border = !passed ? "#333" : direction === "LONG" ? "#166534" : "#7f1d1d";
  return (
    <span
      style={{
        display: "inline-block", padding: "2px 8px", borderRadius: 4, fontSize: 11,
        fontWeight: 700, background: bg, color: fg, border: `1px solid ${border}`,
        letterSpacing: 0.5, fontFamily: "'JetBrains Mono', monospace",
        opacity: passed ? 1 : 0.5,
      }}
    >
      {direction} {!passed && "✕"}
    </span>
  );
}

// ── Main Dashboard ──
export default function V4Dashboard() {
  const [data] = useState(() => generateMockData());
  const [selectedTicker, setSelectedTicker] = useState("NVDA");
  const [tab, setTab] = useState("chart");
  const [animIn, setAnimIn] = useState(false);

  useEffect(() => { setTimeout(() => setAnimIn(true), 50); }, []);

  const tickerData = data.priceData[selectedTicker] || [];
  const tickerSignals = data.signals.filter((s) => s.ticker === selectedTicker);
  const latestDay = tickerData[tickerData.length - 1];
  const prevDay = tickerData[tickerData.length - 2];
  const priceChange = latestDay && prevDay ? latestDay.close - prevDay.close : 0;
  const pctChange = prevDay ? (priceChange / prevDay.close) * 100 : 0;

  // Summary stats
  const totalSignals = data.signals.length;
  const passedSignals = data.signals.filter((s) => s.passed);
  const hitRate = passedSignals.length > 0 ? (passedSignals.filter((s) => s.pnl > 0).length / passedSignals.length * 100) : 0;
  const avgMfeMAE = passedSignals.length > 0
    ? (passedSignals.reduce((a, s) => a + s.mfe / s.mae, 0) / passedSignals.length) : 0;

  const chartData = tickerData.map((d) => ({
    ...d,
    candleColor: d.close >= d.open ? "#34d399" : "#f87171",
    bodyTop: Math.max(d.open, d.close),
    bodyBot: Math.min(d.open, d.close),
    bodyHeight: Math.abs(d.close - d.open),
    signal: tickerSignals.find((s) => s.date === d.date) || null,
  }));

  const forceData = tickerData.map((d) => ({
    date: d.date,
    sForce: d.sForce,
    fill: d.sForce >= 0 ? "#34d399" : "#f87171",
  }));

  return (
    <div
      style={{
        minHeight: "100vh", background: "#0c0d12", color: "#e2e4e9",
        fontFamily: "'Inter', 'Pretendard', -apple-system, sans-serif",
        opacity: animIn ? 1 : 0, transition: "opacity 0.6s ease",
      }}
    >
      {/* ── Header ── */}
      <div style={{
        padding: "16px 20px", borderBottom: "1px solid #1a1b22",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        background: "linear-gradient(180deg, #10111a 0%, #0c0d12 100%)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 32, height: 32, borderRadius: 8,
            background: "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 16, fontWeight: 800, color: "#fff",
          }}>V4</div>
          <div>
            <div style={{ fontSize: 15, fontWeight: 700, letterSpacing: -0.3 }}>V4_wP Signal Dashboard</div>
            <div style={{ fontSize: 11, color: "#6b7080", marginTop: 1 }}>Rev2 · Walk-Forward Validated</div>
          </div>
        </div>
        <div style={{ display: "flex", gap: 16, fontSize: 12 }}>
          <div style={{ textAlign: "center" }}>
            <div style={{ color: "#6b7080", fontSize: 10, textTransform: "uppercase", letterSpacing: 1 }}>Signals</div>
            <div style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, color: "#a78bfa" }}>{totalSignals}</div>
          </div>
          <div style={{ textAlign: "center" }}>
            <div style={{ color: "#6b7080", fontSize: 10, textTransform: "uppercase", letterSpacing: 1 }}>Hit Rate</div>
            <div style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, color: "#34d399" }}>{hitRate.toFixed(1)}%</div>
          </div>
          <div style={{ textAlign: "center" }}>
            <div style={{ color: "#6b7080", fontSize: 10, textTransform: "uppercase", letterSpacing: 1 }}>MFE/MAE</div>
            <div style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, color: "#fbbf24" }}>{avgMfeMAE.toFixed(2)}</div>
          </div>
        </div>
      </div>

      <div style={{ display: "flex", height: "calc(100vh - 65px)" }}>
        {/* ── Sidebar: Watchlist ── */}
        <div style={{
          width: 220, borderRight: "1px solid #1a1b22", overflowY: "auto",
          background: "#0e0f16", flexShrink: 0,
        }}>
          <div style={{ padding: "12px 14px 8px", fontSize: 10, color: "#6b7080", textTransform: "uppercase", letterSpacing: 1.5, fontWeight: 600 }}>
            Watchlist
          </div>
          {data.tickers.map((t) => {
            const td = data.priceData[t];
            const last = td[td.length - 1];
            const prev = td[td.length - 2];
            const chg = ((last.close - prev.close) / prev.close * 100);
            const isSelected = t === selectedTicker;
            const recentSignal = data.signals.filter(s => s.ticker === t).slice(-1)[0];

            return (
              <div
                key={t}
                onClick={() => setSelectedTicker(t)}
                style={{
                  padding: "10px 14px", cursor: "pointer",
                  background: isSelected ? "#1a1b26" : "transparent",
                  borderLeft: isSelected ? "2px solid #6366f1" : "2px solid transparent",
                  transition: "all 0.15s ease",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontWeight: 700, fontSize: 13, letterSpacing: -0.2 }}>{t}</span>
                  <span style={{
                    fontSize: 11, fontFamily: "'JetBrains Mono', monospace",
                    color: chg >= 0 ? "#34d399" : "#f87171", fontWeight: 600,
                  }}>
                    {chg >= 0 ? "+" : ""}{chg.toFixed(2)}%
                  </span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 4 }}>
                  <Sparkline data={td.slice(-20).map(d => d.close)} color={chg >= 0 ? "#34d399" : "#f87171"} width={80} height={20} />
                  {recentSignal && <SignalBadge direction={recentSignal.direction} passed={recentSignal.passed} />}
                </div>
              </div>
            );
          })}
        </div>

        {/* ── Main Content ── */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
          {/* Ticker Header */}
          <div style={{ padding: "14px 20px", borderBottom: "1px solid #1a1b22", display: "flex", alignItems: "center", gap: 20 }}>
            <div>
              <span style={{ fontSize: 22, fontWeight: 800, letterSpacing: -0.5 }}>{selectedTicker}</span>
              <span style={{
                marginLeft: 12, fontSize: 20, fontWeight: 600,
                fontFamily: "'JetBrains Mono', monospace",
              }}>
                ${latestDay?.close.toFixed(2)}
              </span>
              <span style={{
                marginLeft: 8, fontSize: 14, fontWeight: 600,
                fontFamily: "'JetBrains Mono', monospace",
                color: priceChange >= 0 ? "#34d399" : "#f87171",
              }}>
                {priceChange >= 0 ? "▲" : "▼"} {Math.abs(priceChange).toFixed(2)} ({pctChange >= 0 ? "+" : ""}{pctChange.toFixed(2)}%)
              </span>
            </div>
            <div style={{ marginLeft: "auto", display: "flex", gap: 4 }}>
              {["chart", "signals", "indicators"].map((t) => (
                <button
                  key={t}
                  onClick={() => setTab(t)}
                  style={{
                    padding: "5px 14px", border: "none", borderRadius: 6, cursor: "pointer",
                    fontSize: 12, fontWeight: 600, textTransform: "capitalize",
                    background: tab === t ? "#6366f1" : "#1a1b26",
                    color: tab === t ? "#fff" : "#8a8f98",
                    transition: "all 0.15s ease",
                  }}
                >
                  {t}
                </button>
              ))}
            </div>
          </div>

          {/* Chart Area */}
          <div style={{ flex: 1, padding: "12px 16px", overflow: "auto" }}>
            {tab === "chart" && (
              <div>
                {/* Price Chart */}
                <div style={{
                  background: "#12131c", borderRadius: 10, padding: "16px 12px 8px",
                  border: "1px solid #1e1f2e", marginBottom: 12,
                }}>
                  <div style={{ fontSize: 11, color: "#6b7080", marginBottom: 8, paddingLeft: 4, fontWeight: 600, letterSpacing: 0.5 }}>
                    PRICE · 60D
                  </div>
                  <ResponsiveContainer width="100%" height={240}>
                    <ComposedChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: 4 }}>
                      <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#4a4f5e" }} tickLine={false} axisLine={{ stroke: "#1e1f2e" }} interval={9} />
                      <YAxis domain={["auto", "auto"]} tick={{ fontSize: 10, fill: "#4a4f5e" }} tickLine={false} axisLine={false} width={50} />
                      <Tooltip
                        contentStyle={{ background: "#1a1b26", border: "1px solid #2a2b3a", borderRadius: 8, fontSize: 12, color: "#e2e4e9" }}
                        formatter={(v, name) => [typeof v === "number" ? v.toFixed(2) : v, name]}
                      />
                      <Area type="monotone" dataKey="close" fill="url(#priceGrad)" stroke="none" />
                      <Line type="monotone" dataKey="close" stroke="#818cf8" strokeWidth={2} dot={false} />
                      {chartData.filter(d => d.signal).map((d, i) => (
                        <ReferenceLine
                          key={i} x={d.date}
                          stroke={d.signal.direction === "LONG" ? "#34d399" : "#f87171"}
                          strokeDasharray="3 3" strokeWidth={1}
                        />
                      ))}
                      <defs>
                        <linearGradient id="priceGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#818cf8" stopOpacity={0.15} />
                          <stop offset="100%" stopColor="#818cf8" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                    </ComposedChart>
                  </ResponsiveContainer>
                  {/* Signal markers legend */}
                  {tickerSignals.length > 0 && (
                    <div style={{ display: "flex", gap: 12, padding: "8px 4px 4px", flexWrap: "wrap" }}>
                      {tickerSignals.map((s, i) => (
                        <div key={i} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: "#8a8f98" }}>
                          <div style={{ width: 8, height: 8, borderRadius: "50%", background: s.direction === "LONG" ? "#34d399" : "#f87171" }} />
                          {s.date} <SignalBadge direction={s.direction} passed={s.passed} />
                          <span style={{ fontFamily: "'JetBrains Mono', monospace", color: s.pnl > 0 ? "#34d399" : "#f87171" }}>
                            {s.pnl > 0 ? "+" : ""}{s.pnl.toFixed(2)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* s_force Bar Chart */}
                <div style={{
                  background: "#12131c", borderRadius: 10, padding: "16px 12px 8px",
                  border: "1px solid #1e1f2e",
                }}>
                  <div style={{ fontSize: 11, color: "#6b7080", marginBottom: 8, paddingLeft: 4, fontWeight: 600, letterSpacing: 0.5 }}>
                    S_FORCE · PV Momentum
                  </div>
                  <ResponsiveContainer width="100%" height={100}>
                    <BarChart data={forceData} margin={{ top: 4, right: 8, bottom: 0, left: 4 }}>
                      <XAxis dataKey="date" tick={false} axisLine={{ stroke: "#1e1f2e" }} />
                      <YAxis domain={[-1, 1]} tick={{ fontSize: 9, fill: "#4a4f5e" }} tickLine={false} axisLine={false} width={30} ticks={[-1, 0, 1]} />
                      <ReferenceLine y={0} stroke="#2a2b3a" />
                      <Tooltip contentStyle={{ background: "#1a1b26", border: "1px solid #2a2b3a", borderRadius: 8, fontSize: 12 }} />
                      <Bar dataKey="sForce" radius={[2, 2, 0, 0]}>
                        {forceData.map((d, i) => (
                          <Cell key={i} fill={d.fill} fillOpacity={0.7} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {tab === "signals" && (
              <div style={{ background: "#12131c", borderRadius: 10, border: "1px solid #1e1f2e", overflow: "hidden" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                  <thead>
                    <tr style={{ borderBottom: "1px solid #1e1f2e" }}>
                      {["Date", "Dir", "Entry", "s_force", "SQZ", "ER", "ATR%", "MFE", "MAE", "P&L"].map((h) => (
                        <th key={h} style={{
                          padding: "10px 12px", textAlign: "left", fontSize: 10, color: "#6b7080",
                          textTransform: "uppercase", letterSpacing: 1, fontWeight: 600,
                        }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {tickerSignals.map((s, i) => (
                      <tr key={i} style={{ borderBottom: "1px solid #15161f", opacity: s.passed ? 1 : 0.4 }}>
                        <td style={{ padding: "8px 12px", fontFamily: "'JetBrains Mono', monospace" }}>{s.date}</td>
                        <td style={{ padding: "8px 12px" }}><SignalBadge direction={s.direction} passed={s.passed} /></td>
                        <td style={{ padding: "8px 12px", fontFamily: "'JetBrains Mono', monospace" }}>${s.entryPrice.toFixed(2)}</td>
                        <td style={{ padding: "8px 12px", fontFamily: "'JetBrains Mono', monospace", color: s.sForce > 0 ? "#34d399" : "#f87171" }}>{s.sForce.toFixed(3)}</td>
                        <td style={{ padding: "8px 12px" }}>
                          <span style={{
                            display: "inline-block", width: 8, height: 8, borderRadius: "50%",
                            background: s.squeeze !== 0 ? "#fbbf24" : "#2a2b3a",
                          }} />
                        </td>
                        <td style={{ padding: "8px 12px", fontFamily: "'JetBrains Mono', monospace" }}>{s.er.toFixed(3)}</td>
                        <td style={{ padding: "8px 12px", fontFamily: "'JetBrains Mono', monospace" }}>{s.atrPct.toFixed(1)}%</td>
                        <td style={{ padding: "8px 12px", fontFamily: "'JetBrains Mono', monospace", color: "#34d399" }}>+{s.mfe.toFixed(2)}%</td>
                        <td style={{ padding: "8px 12px", fontFamily: "'JetBrains Mono', monospace", color: "#f87171" }}>-{s.mae.toFixed(2)}%</td>
                        <td style={{
                          padding: "8px 12px", fontFamily: "'JetBrains Mono', monospace", fontWeight: 700,
                          color: s.pnl > 0 ? "#34d399" : "#f87171",
                        }}>
                          {s.pnl > 0 ? "+" : ""}{s.pnl.toFixed(2)}%
                        </td>
                      </tr>
                    ))}
                    {tickerSignals.length === 0 && (
                      <tr><td colSpan={10} style={{ padding: 40, textAlign: "center", color: "#4a4f5e" }}>No signals for {selectedTicker}</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            )}

            {tab === "indicators" && latestDay && (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <div style={{ background: "#12131c", borderRadius: 10, padding: 16, border: "1px solid #1e1f2e" }}>
                  <div style={{ fontSize: 11, color: "#6b7080", fontWeight: 600, letterSpacing: 0.5, textTransform: "uppercase", marginBottom: 14 }}>
                    Current State
                  </div>
                  <Gauge value={latestDay.sForce} min={-1} max={1} label="s_force" color={latestDay.sForce > 0 ? "#34d399" : "#f87171"} />
                  <Gauge value={latestDay.er} min={0} max={1} label="Efficiency Ratio" color="#818cf8" />
                  <Gauge value={latestDay.atrPct / 10} min={0} max={1} label={`ATR% (${latestDay.atrPct}%)`} color="#fbbf24" />
                  <div style={{ marginTop: 12, display: "flex", alignItems: "center", gap: 8, fontSize: 12 }}>
                    <span style={{ color: "#6b7080" }}>Squeeze:</span>
                    <span style={{
                      padding: "2px 10px", borderRadius: 4, fontSize: 11, fontWeight: 700,
                      fontFamily: "'JetBrains Mono', monospace",
                      background: latestDay.squeeze !== 0 ? "#3d2e00" : "#1a1b26",
                      color: latestDay.squeeze !== 0 ? "#fbbf24" : "#4a4f5e",
                      border: `1px solid ${latestDay.squeeze !== 0 ? "#6b5300" : "#2a2b3a"}`,
                    }}>
                      {latestDay.squeeze !== 0 ? "FIRING" : "OFF"}
                    </span>
                  </div>
                </div>
                <div style={{ background: "#12131c", borderRadius: 10, padding: 16, border: "1px solid #1e1f2e" }}>
                  <div style={{ fontSize: 11, color: "#6b7080", fontWeight: 600, letterSpacing: 0.5, textTransform: "uppercase", marginBottom: 14 }}>
                    Filter Check
                  </div>
                  {[
                    { label: "s_force magnitude", val: Math.abs(latestDay.sForce), thresh: 0.3, pass: Math.abs(latestDay.sForce) > 0.3 },
                    { label: "ER percentile", val: latestDay.er, thresh: 0.4, pass: latestDay.er > 0.4 },
                    { label: "ATR% percentile", val: latestDay.atrPct / 10, thresh: 0.3, pass: latestDay.atrPct / 10 > 0.3 },
                    { label: "Squeeze confirm", val: latestDay.squeeze !== 0 ? 1 : 0, thresh: 0.5, pass: latestDay.squeeze !== 0 },
                  ].map((f, i) => (
                    <div key={i} style={{
                      display: "flex", alignItems: "center", justifyContent: "space-between",
                      padding: "8px 0", borderBottom: i < 3 ? "1px solid #1a1b22" : "none",
                    }}>
                      <span style={{ fontSize: 12, color: "#8a8f98" }}>{f.label}</span>
                      <span style={{
                        fontSize: 11, fontWeight: 700, padding: "2px 8px", borderRadius: 4,
                        fontFamily: "'JetBrains Mono', monospace",
                        background: f.pass ? "#0d3320" : "#2a1215",
                        color: f.pass ? "#34d399" : "#f87171",
                      }}>
                        {f.pass ? "PASS" : "FAIL"}
                      </span>
                    </div>
                  ))}
                  <div style={{
                    marginTop: 16, padding: "10px 14px", borderRadius: 8, textAlign: "center",
                    fontSize: 13, fontWeight: 700, letterSpacing: 0.5,
                    fontFamily: "'JetBrains Mono', monospace",
                    background: "linear-gradient(135deg, #1a0d33 0%, #0d1a33 100%)",
                    border: "1px solid #2a2b5a",
                    color: "#a78bfa",
                  }}>
                    V4_wP COMPOSITE: {(Math.abs(latestDay.sForce) * 0.4 + latestDay.er * 0.3 + (latestDay.squeeze !== 0 ? 0.3 : 0)).toFixed(3)}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
