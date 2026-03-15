"""GitHub Actions 진입점: 일일 스캔 실행"""
import sys
import os
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from v4wp_realtime.core.scanner import run_scan
from v4wp_realtime.config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, CLAUDE_API_KEY


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
    print('  V4_wP Daily Scan')
    print('=' * 60)
    print(f'  Telegram: {"ON" if alert_fn else "OFF"}')
    print(f'  AI Commentary: {"ON" if commentary_fn else "OFF"}')
    print()

    results = run_scan(alert_fn=alert_fn, commentary_fn=commentary_fn)

    print(f'\n  Scan Complete:')
    print(f'  Scanned: {results["scanned"]} tickers')
    print(f'  New signals: {len(results["new_signals"])}')
    print(f'  Duration: {results["duration_sec"]:.1f}s')

    if results['new_signals']:
        print(f'\n  New Signals:')
        for s in results['new_signals']:
            emoji = '🟢' if s['signal_type'] == 'bottom' else '🔴'
            print(f'    {emoji} {s["ticker"]} ({s["sector"]}) '
                  f'{s["signal_type"].upper()} | Score: {s["peak_val"]:.3f} | '
                  f'${s["close_price"]:.2f}')

    if results['errors']:
        print(f'\n  Errors ({len(results["errors"])}):')
        for ticker, err in results['errors']:
            print(f'    {ticker}: {err}')

    # JSON 백업
    import json
    from v4wp_realtime.config.settings import SIGNALS_JSON
    if results['new_signals']:
        backup = []
        if SIGNALS_JSON.exists():
            try:
                with open(SIGNALS_JSON, 'r') as f:
                    backup = json.load(f)
            except Exception:
                pass
        backup.extend(results['new_signals'])
        SIGNALS_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(SIGNALS_JSON, 'w') as f:
            json.dump(backup, f, indent=2, default=str)


if __name__ == '__main__':
    main()
