"""버그 수정사항 검증용 텔레그램 전송 테스트.

확인 대상:
  D-1) Haiku model ID — 실제 AI 호출로 fallback 아닌 문장이 나오는지
  D-2) Mini App deep link — Dashboard 버튼 URL에 ticker/peak_date 둘 다 포함
  M-1) blocked_buys 종목명 — DD차단 리스트에 티커+DD% 표시
  M-2) AI 위원회 (interpretation) — 카드 하단에 verdict + chairman key_point
  M-3) 레짐 라인 — 카드에 "🌍 레짐 ..." 줄

사용법:
  TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID 환경변수 또는 v4wp_realtime/.env 설정 후:

      python v4wp_realtime/scripts/test_bugfix_telegram.py

  CLAUDE_API_KEY가 없으면 D-1만 스킵, 나머지는 정상 전송.
  TELEGRAM_WEBAPP_URL이 없으면 D-2 검사 스킵.
"""
import sys
import json
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from v4wp_realtime.config.settings import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    TELEGRAM_WEBAPP_URL, CLAUDE_API_KEY,
)
from v4wp_realtime.alerts.telegram_bot import (
    _send_message, _dashboard_button, send_signal_alert,
)
from v4wp_realtime.alerts.message_formatter import (
    format_signal_message, format_scan_summary,
)


SEPARATOR = '\n' + '=' * 60 + '\n'


def _make_signal_with_all_fields():
    """M-2/M-3 + 모든 신규 필드를 채운 가짜 신호."""
    return {
        'ticker': 'NVDA',
        'sector': 'Tech',
        'signal_type': 'bottom',
        'peak_date': '2026-05-12',
        'peak_val': -0.18,
        'start_val': -0.05,
        'duration': 5,
        'close_price': 500.0,
        'current_price': 510.0,
        'detected_date': '2026-05-13',
        's_force': 0.42,
        's_div': 0.31,
        's_conc': 0.0,
        'dd_pct': 0.062,
        'action_pct': 0.5,
        'signal_tier': 'CONFIRMED',
        'signal_label': 'BUY (CONFIRMED)',
        'is_strong': True,
        'commentary': 'NVDA 거래량 반전 감지 — 단기 매수 기회.',
        # M-3 대상
        'market_regime': 'BEAR_STRONG',
        'market_return_20d': -0.035,
        'sector_return_20d': -0.028,
        'vix_change_20d': 0.42,
        # M-2 대상 — 실제 scanner는 JSON 문자열로 저장하므로 동일 포맷
        'interpretation': json.dumps({
            'final_verdict': 'STRONG_BUY',
            'confidence_score': 82,
            'chairman': {
                'persona_name': 'AI 의장',
                'key_point': '공포 극대화 + S_Force 강함 → 역발상 매수 타이밍',
                'analysis': '...',
                'conviction': 5,
            },
        }, ensure_ascii=False),
    }


def _make_summary_with_blocked():
    """M-1 대상 — blocked_buys 리스트가 채워진 results."""
    sig = _make_signal_with_all_fields()
    return {
        'date': '2026-05-13',
        'scanned': 30,
        'duration_sec': 12.3,
        'new_signals': [sig],
        'errors': [],
        'blocked_buys': [
            {'ticker': 'AAPL', 'dd_pct': 0.025},
            {'ticker': 'TSLA', 'dd_pct': 0.028},
            {'ticker': 'MSFT', 'dd_pct': 0.022},
        ],
        'watch_alerts': [],
        'market_event': None,
    }


def test_d1_commentary():
    """D-1: 실제 Anthropic API 호출 — fallback인지 실제 응답인지 확인."""
    if not CLAUDE_API_KEY:
        print('  SKIP: CLAUDE_API_KEY not set')
        return None

    from v4wp_realtime.ai.commentary import generate_commentary
    from v4wp_realtime.ai.prompt_templates import get_fallback

    sig = _make_signal_with_all_fields()
    out = generate_commentary(sig)
    fallback = get_fallback(sig)

    print(f'  Output:   {out}')
    print(f'  Fallback: {fallback}')
    if out == fallback:
        print('  RESULT: ❌ STILL FALLBACK — model ID 또는 API key 확인 필요')
        return False
    print('  RESULT: ✅ AI commentary 정상 (fallback 아님)')
    return True


def test_d2_dashboard_url():
    """D-2: _dashboard_button이 ticker + peak_date 둘 다 URL에 넣는지."""
    if not TELEGRAM_WEBAPP_URL:
        print('  SKIP: TELEGRAM_WEBAPP_URL not set (fallback t.me 경로는 별도 검증)')
        return None

    btn = _dashboard_button(ticker='NVDA', peak_date='2026-05-12')
    if not btn:
        print('  RESULT: ❌ 버튼 생성 실패')
        return False
    btn_obj = btn[0]
    url = btn_obj.get('web_app', {}).get('url', '')
    print(f'  Button URL: {url}')

    ok_ticker = 'ticker=NVDA' in url
    ok_date = 'peak_date=2026-05-12' in url
    if ok_ticker and ok_date:
        print('  RESULT: ✅ ticker + peak_date 둘 다 포함')
        return True
    print(f'  RESULT: ❌ ticker={ok_ticker} peak_date={ok_date}')
    return False


def test_send_card_to_telegram():
    """M-2/M-3 시각 검증 — 실제 텔레그램으로 카드 전송."""
    sig = _make_signal_with_all_fields()
    msg = format_signal_message(sig)
    print('  Rendered card preview:')
    print('  ' + '-' * 40)
    for line in msg.split('\n'):
        print('  ' + line)
    print('  ' + '-' * 40)

    # send_signal_alert (chart_bytes=None) → format_signal_message + Dashboard 버튼
    result = send_signal_alert(sig, chart_bytes=None)
    msg_id = result.get('result', {}).get('message_id')
    print(f'  Sent: message_id={msg_id}')
    return msg_id is not None


def test_send_summary_to_telegram():
    """M-1 시각 검증 — 일일 요약(blocked_buys 포함) 전송."""
    results = _make_summary_with_blocked()
    msg = format_scan_summary(results)
    print('  Rendered summary preview:')
    print('  ' + '-' * 40)
    for line in msg.split('\n'):
        print('  ' + line)
    print('  ' + '-' * 40)

    api_result = _send_message(msg, parse_mode='HTML')
    msg_id = api_result.get('result', {}).get('message_id')
    print(f'  Sent: message_id={msg_id}')
    return msg_id is not None


def main():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print('ERROR: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 환경변수 미설정')
        print('       v4wp_realtime/.env 파일에 추가하거나 export 후 재실행')
        sys.exit(1)

    print(SEPARATOR + '[1/4] D-1: Haiku model ID' + SEPARATOR)
    test_d1_commentary()

    print(SEPARATOR + '[2/4] D-2: Dashboard URL params' + SEPARATOR)
    test_d2_dashboard_url()

    print(SEPARATOR + '[3/4] M-2 + M-3: 카드 — 레짐 + AI 위원회' + SEPARATOR)
    test_send_card_to_telegram()

    print(SEPARATOR + '[4/4] M-1: 일일 요약 — blocked_buys 종목' + SEPARATOR)
    test_send_summary_to_telegram()

    print(SEPARATOR + '완료 — 텔레그램에서 직접 카드를 확인하세요.' + SEPARATOR)
    print('체크포인트:')
    print('  카드:')
    print('    □ "🌍 레짐  공포 극대 → 역발상 매수  (QQQ 20d -3.5%)" 줄  ← M-3')
    print('    □ "🏛 🔥 강력 매수 (확신도 82/100)" 줄                      ← M-2')
    print('    □ "⤷ 공포 극대화 + S_Force 강함 …" 줄                     ← M-2')
    print('    □ Dashboard 버튼 → 누르면 NVDA가 선택된 상태로 열림        ← D-2')
    print('  요약:')
    print('    □ "🚫 DD게이트 차단 3건: AAPL(2.5%), TSLA(2.8%), …" 줄    ← M-1')


if __name__ == '__main__':
    main()
