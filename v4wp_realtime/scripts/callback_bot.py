"""Telegram 인라인 키보드 콜백 핸들러.

사용법:
    python v4wp_realtime/scripts/callback_bot.py

daily_scan과 별도로 실행. 버튼 눌림을 감지하고 요약/상세 토글 처리.
Ctrl+C로 종료.
"""
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from v4wp_realtime.alerts.telegram_bot import run_callback_handler


if __name__ == '__main__':
    print('=' * 40)
    print('  V4_wP Callback Bot')
    print('  Press Ctrl+C to stop')
    print('=' * 40)
    run_callback_handler()
