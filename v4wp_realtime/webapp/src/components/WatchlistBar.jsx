import Sparkline from "./Sparkline";
import SignalBadge from "./SignalBadge";

/**
 * 가로 스크롤 워치리스트 (모바일 최적화)
 * - 각 종목 칩: 티커 + 변화율 + 스파크라인 + 시그널뱃지
 */
export default function WatchlistBar({ items, selected, onSelect, chartCache }) {
  if (!items || items.length === 0) return null;

  return (
    <div
      style={{
        display: "flex",
        overflowX: "auto",
        gap: 8,
        padding: "10px 12px",
        borderBottom: "1px solid rgba(255,255,255,0.06)",
        background: "var(--tg-header-bg)",
        WebkitOverflowScrolling: "touch",
        scrollbarWidth: "none",
      }}
    >
      {items.map((item) => {
        const isActive = item.ticker === selected;
        const chg = item.change_pct;
        const chgColor = chg != null && chg >= 0 ? "#34d399" : "#f87171";
        const sparkData = chartCache[item.ticker];

        return (
          <div
            key={item.ticker}
            onClick={() => onSelect(item.ticker)}
            style={{
              flexShrink: 0,
              padding: "8px 12px",
              borderRadius: 10,
              cursor: "pointer",
              background: isActive ? "var(--tg-btn)" : "var(--tg-secondary-bg)",
              border: isActive
                ? "1px solid var(--tg-btn)"
                : "1px solid rgba(255,255,255,0.06)",
              transition: "all 0.15s ease",
              minWidth: 110,
            }}
          >
            {/* 상단: 티커 + 변화율 */}
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                gap: 8,
              }}
            >
              <span
                style={{
                  fontWeight: 700,
                  fontSize: 13,
                  color: isActive ? "var(--tg-btn-text)" : "var(--tg-text)",
                }}
              >
                {item.ticker}
              </span>
              {chg != null && (
                <span
                  style={{
                    fontSize: 10,
                    fontFamily: "'JetBrains Mono', monospace",
                    fontWeight: 600,
                    color: isActive ? "var(--tg-btn-text)" : chgColor,
                  }}
                >
                  {chg >= 0 ? "+" : ""}
                  {chg.toFixed(1)}%
                </span>
              )}
            </div>

            {/* 하단: 스파크라인 + 뱃지 */}
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginTop: 6,
                gap: 6,
              }}
            >
              {sparkData && sparkData.length > 1 ? (
                <Sparkline
                  data={sparkData}
                  color={isActive ? "rgba(255,255,255,0.7)" : chgColor}
                  width={50}
                  height={16}
                />
              ) : (
                <div style={{ width: 50 }} />
              )}
              {item.recent_signal && (
                <SignalBadge
                  direction={item.recent_signal.direction}
                  tier={item.recent_signal.tier}
                />
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
