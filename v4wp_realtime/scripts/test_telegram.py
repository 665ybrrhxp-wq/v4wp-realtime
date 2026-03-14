"""텔레그램 수동 테스트"""
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from v4wp_realtime.alerts.telegram_bot import send_test_message, send_signal_alert
from v4wp_realtime.config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID


def main():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print('ERROR: Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables')
        print('  export TELEGRAM_BOT_TOKEN="your-token"')
        print('  export TELEGRAM_CHAT_ID="your-chat-id"')
        return

    print('Sending test message...')
    result = send_test_message()
    print(f'OK: message_id={result["result"]["message_id"]}')

    # 샘플 신호 테스트
    print('\nSending sample signal...')
    sample = {
        'ticker': 'AAPL',
        'sector': 'Tech',
        'signal_type': 'bottom',
        'peak_val': 0.342,
        'close_price': 227.48,
        's_force': 0.65,
        's_div': 0.78,
        's_conc': 0.42,
        'peak_date': '2026-03-09',
        'commentary': 'AAPL 거래량 바닥 신호 — 단기 반등 가능성을 시사합니다.',
    }
    result = send_signal_alert(sample)
    print(f'OK: message_id={result["result"]["message_id"]}')


if __name__ == '__main__':
    main()
