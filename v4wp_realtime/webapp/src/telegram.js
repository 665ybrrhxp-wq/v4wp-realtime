/**
 * Telegram Web App SDK wrapper
 *
 * - 테마 색상을 CSS 변수로 매핑
 * - BackButton / MainButton 제어
 * - initData에서 start_param(딥링크) 파싱
 */

const tg = window.Telegram?.WebApp;

/** Telegram 환경인지 여부 */
export const isTelegram = !!tg;

/** 앱 초기화: expand + 테마 적용 */
export function initTelegramApp() {
  if (!tg) return;
  tg.ready();
  tg.expand();
  applyTheme();
  tg.onEvent("themeChanged", applyTheme);
}

/** Telegram themeParams → CSS 변수 매핑 */
function applyTheme() {
  if (!tg) return;
  const p = tg.themeParams;
  const root = document.documentElement;

  // 다크모드 기본값 (Telegram 테마가 없으면 폴백)
  const map = {
    "--tg-bg": p.bg_color || "#0c0d12",
    "--tg-text": p.text_color || "#e2e4e9",
    "--tg-hint": p.hint_color || "#6b7080",
    "--tg-link": p.link_color || "#818cf8",
    "--tg-btn": p.button_color || "#6366f1",
    "--tg-btn-text": p.button_text_color || "#ffffff",
    "--tg-secondary-bg": p.secondary_bg_color || "#12131c",
    "--tg-header-bg": p.header_bg_color || "#10111a",
    "--tg-section-bg": p.section_bg_color || "#12131c",
    "--tg-accent": p.accent_text_color || "#a78bfa",
    "--tg-subtitle": p.subtitle_text_color || "#8a8f98",
    "--tg-destructive": p.destructive_text_color || "#f87171",
  };

  Object.entries(map).forEach(([k, v]) => root.style.setProperty(k, v));
}

/**
 * 딥링크 파싱.
 * 우선순위 1: Telegram start_param (t.me 딥링크 경로) — 예: "NVDA_2026-03-16"
 * 우선순위 2: URL 쿼리스트링 (web_app 버튼 경로) — 예: "?ticker=NVDA&peak_date=2026-03-16"
 */
export function getStartParam() {
  const raw = tg?.initDataUnsafe?.start_param;
  if (raw) {
    const sep = raw.indexOf("_");
    if (sep === -1) return { ticker: raw.toUpperCase() };
    return {
      ticker: raw.slice(0, sep).toUpperCase(),
      peakDate: raw.slice(sep + 1),
    };
  }

  if (typeof window !== "undefined" && window.location?.search) {
    const params = new URLSearchParams(window.location.search);
    const ticker = params.get("ticker");
    if (ticker) {
      return {
        ticker: ticker.toUpperCase(),
        peakDate: params.get("peak_date") || null,
      };
    }
  }
  return null;
}

/** BackButton 표시/숨김 */
export function showBackButton(onClick) {
  if (!tg) return;
  tg.BackButton.show();
  tg.BackButton.onClick(onClick);
}

export function hideBackButton() {
  if (!tg) return;
  tg.BackButton.hide();
  tg.BackButton.offClick();
}

/** MainButton 제어 */
export function showMainButton(text, onClick) {
  if (!tg) return;
  tg.MainButton.setText(text);
  tg.MainButton.show();
  tg.MainButton.onClick(onClick);
}

export function hideMainButton() {
  if (!tg) return;
  tg.MainButton.hide();
  tg.MainButton.offClick();
}

/** haptic feedback */
export function haptic(type = "impact") {
  if (!tg?.HapticFeedback) return;
  if (type === "impact") tg.HapticFeedback.impactOccurred("light");
  else if (type === "success") tg.HapticFeedback.notificationOccurred("success");
  else if (type === "error") tg.HapticFeedback.notificationOccurred("error");
}
