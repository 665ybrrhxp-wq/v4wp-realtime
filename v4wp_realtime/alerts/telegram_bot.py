"""Telegram Bot API를 통한 알림 전송"""
import requests
from v4wp_realtime.config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from v4wp_realtime.alerts.message_formatter import format_signal_message, format_scan_summary


TELEGRAM_API = 'https://api.telegram.org/bot{token}/sendMessage'


def _send_message(text, parse_mode=None):
    """Telegram 메시지 전송"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise ValueError('TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID required')

    url = TELEGRAM_API.format(token=TELEGRAM_BOT_TOKEN)
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': text,
    }
    if parse_mode:
        payload['parse_mode'] = parse_mode

    resp = requests.post(url, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def send_signal_alert(signal):
    """개별 신호 알림 전송"""
    msg = format_signal_message(signal)
    return _send_message(msg)


def send_scan_summary(results):
    """스캔 결과 요약 전송"""
    msg = format_scan_summary(results)
    return _send_message(msg)


def send_test_message():
    """테스트 메시지 전송"""
    return _send_message('\U0001f916 V4_wP Realtime Alert System: Test message OK!')
