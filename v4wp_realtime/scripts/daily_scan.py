"""GitHub Actions 진입점: 일일 스캔 실행 (매수 신호 전용)

신호 데이터 영속화:
  - data/signals_history.json: 전체 신호 누적 (git 추적)
  - 스캔 전 JSON 로드 → SQLite 복원 → 중복 제거 → 스캔 후 JSON 업데이트
"""
import sys
import json
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from v4wp_realtime.core.scanner import run_scan
from v4wp_realtime.config.settings import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, CLAUDE_API_KEY, SIGNALS_JSON,
)
from v4wp_realtime.data.store import get_connection, init_db, insert_signal_event


def load_signal_history():
    """JSON에서 기존 신호 히스토리 로드."""
    if not SIGNALS_JSON.exists():
        return []
    try:
        with open(SIGNALS_JSON, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []


def restore_signals_to_db(signals):
    """기존 JSON 신호를 SQLite에 복원 (중복 제거용).

    GitHub Actions는 매 실행마다 빈 DB로 시작하므로,
    JSON 히스토리의 기존 신호를 DB에 넣어야 is_new_signal()이 작동함.
    """
    if not signals:
        return 0
    conn = get_connection()
    restored = 0
    for s in signals:
        # DB insert에 필요한 최소 필드 보장
        event = {
            'ticker': s.get('ticker', ''),
            'signal_type': s.get('signal_type', 'bottom'),
            'peak_date': s.get('peak_date', ''),
            'peak_val': s.get('peak_val', 0),
            'start_val': s.get('start_val', 0),
            'close_price': s.get('close_price', 0),
            'detected_date': s.get('detected_date', ''),
            'notified': 1,  # 이미 알림 완료
            'commentary': s.get('commentary'),
            's_force': s.get('s_force', 0),
            's_div': s.get('s_div', 0),
            's_conc': s.get('s_conc', 0),
            'er': s.get('er'),
            'atr_pct': s.get('atr_pct'),
            'signal_tier': s.get('signal_tier', 'CONFIRMED'),
            'action_pct': s.get('action_pct', 1.0),
        }
        if insert_signal_event(conn, event):
            restored += 1
    conn.close()
    return restored


def save_signal_history(existing, new_signals):
    """신호 히스토리 JSON 저장 (기존 + 신규)."""
    all_signals = existing + new_signals
    SIGNALS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SIGNALS_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_signals, f, indent=2, default=str, ensure_ascii=False)
    return len(all_signals)


def main():
    # 알림 함수 설정
    alert_fn = None
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        from v4wp_realtime.alerts.telegram_bot import send_signal_alert
        alert_fn = send_signal_alert

    # AI 코멘터리 함수 설정
    commentary_fn = None
    if CLAUDE_API_KEY:
        from v4wp_realtime.ai.commentary import generate_commentary
        commentary_fn = generate_commentary

    print('=' * 60)
    print('  V4_wP Daily Scan (BUY only)')
    print('=' * 60)
    print(f'  Telegram: {"ON" if alert_fn else "OFF"}')
    print(f'  AI Commentary: {"ON" if commentary_fn else "OFF"}')

    # 1) DB 초기화 + JSON 히스토리에서 기존 신호 복원
    init_db()
    history = load_signal_history()
    restored = restore_signals_to_db(history)
    print(f'  Signal history: {len(history)} loaded, {restored} restored to DB')
    print()

    # 2) 스캔 실행
    results = run_scan(alert_fn=alert_fn, commentary_fn=commentary_fn)

    print(f'\n  Scan Complete:')
    print(f'  Scanned: {results["scanned"]} tickers')
    print(f'  New signals: {len(results["new_signals"])}')
    print(f'  Duration: {results["duration_sec"]:.1f}s')

    if results['new_signals']:
        print(f'\n  New Signals:')
        for s in results['new_signals']:
            print(f'    \U0001f7e2 {s["ticker"]} ({s["sector"]}) '
                  f'BUY | Score: {s["peak_val"]:.3f} | '
                  f'${s["close_price"]:.2f} | {s.get("duration", 0)}d')

    if results['errors']:
        print(f'\n  Errors ({len(results["errors"])}):')
        for ticker, err in results['errors']:
            print(f'    {ticker}: {err}')

    # 3) JSON 히스토리 업데이트
    if results['new_signals']:
        total = save_signal_history(history, results['new_signals'])
        print(f'\n  Signal history updated: {total} total signals')
    else:
        print(f'\n  No new signals — history unchanged ({len(history)} total)')


if __name__ == '__main__':
    main()
