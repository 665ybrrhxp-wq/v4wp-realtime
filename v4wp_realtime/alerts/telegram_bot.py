"""Telegram Bot API를 통한 알림 전송 (인라인 키보드 + 콜백 지원)

Mini App 딥링크 연동:
  - 개별 시그널 알림에 "Dashboard" 버튼 추가
    → https://t.me/{bot_username}/{miniapp_short}?startapp={ticker}_{peak_date}
  - 일일 리포트에 "View Dashboard" 버튼 추가
    → https://t.me/{bot_username}/{miniapp_short}

BotFather 설정 방법:
  1. /mybots → 봇 선택 → Bot Settings → Menu Button
     - 또는 /setmenubutton 명령
     - Web App URL: https://your-server.com/app/  (FastAPI 서버 주소)
  2. /newapp → 봇 선택 → Web App URL 입력 → Short Name 설정
     - Short Name이 TELEGRAM_MINIAPP_SHORT 환경변수 값과 일치해야 함
  3. .env에 추가:
     TELEGRAM_BOT_USERNAME=your_bot_username  (@ 제외)
     TELEGRAM_MINIAPP_SHORT=app               (기본값)
"""
import io
import json
import time
import requests
from v4wp_realtime.config.settings import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DATA_DIR,
    TELEGRAM_BOT_USERNAME, TELEGRAM_MINIAPP_SHORT,
)
from v4wp_realtime.alerts.message_formatter import (
    format_signal_message, format_signal_compact,
    format_signals_summary, format_scan_summary,
)


# ── Telegram API 엔드포인트 ──
_BASE = 'https://api.telegram.org/bot{token}'
TELEGRAM_API_MSG = _BASE + '/sendMessage'
TELEGRAM_API_PHOTO = _BASE + '/sendPhoto'
TELEGRAM_API_MEDIA_GROUP = _BASE + '/sendMediaGroup'
TELEGRAM_API_EDIT_TEXT = _BASE + '/editMessageText'
TELEGRAM_API_EDIT_CAPTION = _BASE + '/editMessageCaption'
TELEGRAM_API_DELETE = _BASE + '/deleteMessage'
TELEGRAM_API_ANSWER_CB = _BASE + '/answerCallbackQuery'
TELEGRAM_API_UPDATES = _BASE + '/getUpdates'

# 콜백 데이터 / 차트 저장 경로
CALLBACK_STORE = DATA_DIR / 'callback_data.json'
CHART_DIR = DATA_DIR / 'charts'


# ════════════════════════════════════════════
# 기본 전송
# ════════════════════════════════════════════

def _send_message(text, parse_mode=None, reply_markup=None):
    """Telegram 메시지 전송."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise ValueError('TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID required')

    url = TELEGRAM_API_MSG.format(token=TELEGRAM_BOT_TOKEN)
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': text,
    }
    if parse_mode:
        payload['parse_mode'] = parse_mode
    if reply_markup:
        payload['reply_markup'] = json.dumps(reply_markup)

    resp = requests.post(url, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _send_photo(photo_bytes, caption=None, parse_mode=None, reply_markup=None,
                chat_id=None):
    """Telegram 사진 + 캡션 전송."""
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError('TELEGRAM_BOT_TOKEN required')

    target_chat = chat_id or TELEGRAM_CHAT_ID
    if not target_chat:
        raise ValueError('chat_id required')

    url = TELEGRAM_API_PHOTO.format(token=TELEGRAM_BOT_TOKEN)
    data = {'chat_id': target_chat}
    if caption:
        data['caption'] = caption
    if parse_mode:
        data['parse_mode'] = parse_mode
    if reply_markup:
        data['reply_markup'] = json.dumps(reply_markup)

    photo_bytes.seek(0)
    files = {'photo': ('chart.png', photo_bytes, 'image/png')}
    resp = requests.post(url, data=data, files=files, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _send_media_group(items):
    """Telegram 앨범(MediaGroup) 전송. 최대 10개."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise ValueError('TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID required')

    url = TELEGRAM_API_MEDIA_GROUP.format(token=TELEGRAM_BOT_TOKEN)

    media = []
    files = {}
    for i, (photo_bytes, caption, parse_mode) in enumerate(items):
        attach_name = f'photo_{i}'
        entry = {
            'type': 'photo',
            'media': f'attach://{attach_name}',
        }
        if caption:
            entry['caption'] = caption
        if parse_mode:
            entry['parse_mode'] = parse_mode
        media.append(entry)
        photo_bytes.seek(0)
        files[attach_name] = (f'chart_{i}.png', photo_bytes, 'image/png')

    data = {
        'chat_id': TELEGRAM_CHAT_ID,
        'media': json.dumps(media),
    }

    resp = requests.post(url, data=data, files=files, timeout=60)
    resp.raise_for_status()
    return resp.json()


# ════════════════════════════════════════════
# 메시지 편집 / 삭제
# ════════════════════════════════════════════

def _edit_message_caption(chat_id, message_id, caption, parse_mode=None, reply_markup=None):
    """사진 메시지의 캡션 편집."""
    url = TELEGRAM_API_EDIT_CAPTION.format(token=TELEGRAM_BOT_TOKEN)
    payload = {'chat_id': chat_id, 'message_id': message_id}
    if caption:
        payload['caption'] = caption
    if parse_mode:
        payload['parse_mode'] = parse_mode
    if reply_markup:
        payload['reply_markup'] = json.dumps(reply_markup)

    resp = requests.post(url, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _edit_message_text(chat_id, message_id, text, parse_mode=None, reply_markup=None):
    """텍스트 메시지 편집."""
    url = TELEGRAM_API_EDIT_TEXT.format(token=TELEGRAM_BOT_TOKEN)
    payload = {'chat_id': chat_id, 'message_id': message_id, 'text': text}
    if parse_mode:
        payload['parse_mode'] = parse_mode
    if reply_markup:
        payload['reply_markup'] = json.dumps(reply_markup)

    resp = requests.post(url, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _delete_message(chat_id, message_id):
    """메시지 삭제."""
    url = TELEGRAM_API_DELETE.format(token=TELEGRAM_BOT_TOKEN)
    payload = {'chat_id': chat_id, 'message_id': message_id}
    resp = requests.post(url, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _answer_callback(callback_query_id, text=None):
    """콜백 쿼리 응답 (버튼 로딩 해제)."""
    url = TELEGRAM_API_ANSWER_CB.format(token=TELEGRAM_BOT_TOKEN)
    payload = {'callback_query_id': callback_query_id}
    if text:
        payload['text'] = text
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception:
        pass


# ════════════════════════════════════════════
# 인라인 키보드 빌더
# ════════════════════════════════════════════

def _keyboard(rows):
    """인라인 키보드 마크업 생성.

    각 row는 (text, data) 튜플 리스트.
    data가 http:// 또는 https://로 시작하면 url 버튼, 아니면 callback_data 버튼.
    """
    return {
        'inline_keyboard': [
            [
                {'text': text, 'url': data}
                if isinstance(data, str) and data.startswith('http')
                else {'text': text, 'callback_data': data}
                for text, data in row
            ]
            for row in rows
        ]
    }


def _miniapp_url(ticker=None, peak_date=None):
    """Mini App 딥링크 URL 생성.

    Returns: URL string 또는 None (BOT_USERNAME 미설정 시)
    """
    if not TELEGRAM_BOT_USERNAME:
        return None
    base = f'https://t.me/{TELEGRAM_BOT_USERNAME}/{TELEGRAM_MINIAPP_SHORT}'
    if ticker and peak_date:
        return f'{base}?startapp={ticker}_{peak_date}'
    elif ticker:
        return f'{base}?startapp={ticker}'
    return base


def _detail_button(ticker, peak_date):
    return [('\U0001f4ca 상세 보기', f'd:{ticker}:{peak_date}')]


def _compact_button(ticker, peak_date):
    return [('\U0001f4cb 요약', f'c:{ticker}:{peak_date}')]


def _summary_button(detected_date):
    return [('\u2190 전체 요약', f's:{detected_date}')]


def _close_button():
    return [('\u2715 닫기', 'x:close')]


def _dashboard_button(ticker=None, peak_date=None):
    """Mini App 딥링크 버튼. BOT_USERNAME 미설정 시 빈 리스트 반환."""
    url = _miniapp_url(ticker, peak_date)
    if not url:
        return []
    return [('\U0001f4f1 Dashboard', url)]


# ════════════════════════════════════════════
# 차트 디스크 저장/로드 (콜백 시 재전송용)
# ════════════════════════════════════════════

def _save_chart(ticker, peak_date, chart_bytes):
    """차트 PNG를 디스크에 저장."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    path = CHART_DIR / f'{ticker}_{peak_date}.png'
    chart_bytes.seek(0)
    with open(path, 'wb') as f:
        f.write(chart_bytes.read())
    chart_bytes.seek(0)


def _load_chart(ticker, peak_date):
    """저장된 차트 PNG 로드. 없으면 None."""
    path = CHART_DIR / f'{ticker}_{peak_date}.png'
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return io.BytesIO(f.read())


# ════════════════════════════════════════════
# 콜백 데이터 저장/로드
# ════════════════════════════════════════════

def save_callback_data(signals, detected_date):
    """콜백 처리를 위한 신호 데이터 저장."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    store = {}
    if CALLBACK_STORE.exists():
        try:
            with open(CALLBACK_STORE, 'r', encoding='utf-8') as f:
                store = json.load(f)
        except Exception:
            store = {}

    for s in signals:
        key = f'{s["ticker"]}:{s["peak_date"]}'
        store[key] = s

    summary_key = f'_summary:{detected_date}'
    store[summary_key] = {
        'signal_keys': [f'{s["ticker"]}:{s["peak_date"]}' for s in signals],
        'detected_date': detected_date,
    }

    with open(CALLBACK_STORE, 'w', encoding='utf-8') as f:
        json.dump(store, f, indent=2, default=str, ensure_ascii=False)


def _load_callback_store():
    """콜백 데이터 로드."""
    if not CALLBACK_STORE.exists():
        return {}
    try:
        with open(CALLBACK_STORE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


# ════════════════════════════════════════════
# 신호 전송 (인라인 키보드 포함)
# ════════════════════════════════════════════

def send_signal_alert(signal, chart_bytes=None):
    """개별 신호 알림 전송.

    chart_bytes가 있으면 차트 + 컴팩트 캡션 + [상세][Dashboard] 버튼.
    없으면 상세 텍스트 + [Dashboard] 버튼.
    """
    ticker = signal['ticker']
    peak_date = signal['peak_date']
    dash_btn = _dashboard_button(ticker, peak_date)

    if chart_bytes:
        _save_chart(ticker, peak_date, chart_bytes)
        compact = format_signal_compact(signal)
        rows = [_detail_button(ticker, peak_date)]
        if dash_btn:
            rows.append(dash_btn)
        kb = _keyboard(rows)
        return _send_photo(chart_bytes, caption=compact, parse_mode='HTML',
                           reply_markup=kb)

    msg = format_signal_message(signal)
    rows = []
    if dash_btn:
        rows.append(dash_btn)
    kb = _keyboard(rows) if rows else None
    return _send_message(msg, parse_mode='HTML', reply_markup=kb)


def send_signal_album(signal_chart_pairs):
    """여러 신호를 요약 메시지 + 인라인 키보드로 전송.

    - 차트를 디스크에 저장 (콜백 시 재전송용)
    - 요약 텍스트 1개 + 종목별 버튼
    - 버튼 탭 → 차트+상세 분석을 새 메시지로 전송
    """
    with_chart = [(s, c) for s, c in signal_chart_pairs if c is not None]
    without_chart = [(s, c) for s, c in signal_chart_pairs if c is None]
    results = []

    # 차트 디스크 저장
    for s, c in with_chart:
        if c:
            _save_chart(s['ticker'], s['peak_date'], c)

    if len(with_chart) == 1:
        # 단일 → 차트+캡션+키보드
        s, c = with_chart[0]
        results.append(send_signal_alert(s, chart_bytes=c))

    elif len(with_chart) > 1:
        # 다중 → 요약 메시지 + 종목별 버튼 + Dashboard (앨범 없음)
        all_signals = [s for s, _ in with_chart[:10]]
        summary_msg = format_signals_summary(all_signals)

        buttons = []
        row = []
        for s, _ in with_chart[:10]:
            row.append((f'\U0001f4ca {s["ticker"]}',
                         f'd:{s["ticker"]}:{s["peak_date"]}'))
            if len(row) == 2:
                buttons.append(row)
                row = []
        if row:
            buttons.append(row)

        # Mini App Dashboard 버튼 (전체 대시보드)
        dash_btn = _dashboard_button()
        if dash_btn:
            buttons.append(dash_btn)

        detected_date = all_signals[0].get('detected_date', all_signals[0]['peak_date'])
        kb = _keyboard(buttons)
        results.append(_send_message(summary_msg, parse_mode='HTML', reply_markup=kb))

        save_callback_data(all_signals, detected_date)

        # 10개 초과 시 나머지 개별
        for s, c in with_chart[10:]:
            results.append(send_signal_alert(s, chart_bytes=c))

    # 차트 없는 신호
    for s, _ in without_chart:
        results.append(send_signal_alert(s))

    # 단일 신호도 콜백 데이터 저장
    if len(with_chart) == 1:
        s = with_chart[0][0]
        detected_date = s.get('detected_date', s['peak_date'])
        save_callback_data([s], detected_date)

    return results


# ════════════════════════════════════════════
# 콜백 처리
# ════════════════════════════════════════════

def handle_callback(update):
    """인라인 키보드 콜백 처리.

    callback_data 형식:
      'd:{ticker}:{peak_date}' → 상세 (차트+텍스트)
      'c:{ticker}:{peak_date}' → 요약으로 복귀 (사진 캡션)
      's:{detected_date}'      → 전체 요약으로 복귀
      'x:close'                → 메시지 삭제
    """
    cb = update.get('callback_query')
    if not cb:
        return

    cb_id = cb['id']
    data = cb.get('data', '')
    msg = cb.get('message', {})
    chat_id = msg.get('chat', {}).get('id')
    message_id = msg.get('message_id')
    is_photo = 'photo' in msg

    if not chat_id or not message_id:
        _answer_callback(cb_id)
        return

    store = _load_callback_store()
    parts = data.split(':')
    action = parts[0] if parts else ''

    try:
        if action == 'd' and len(parts) >= 3:
            ticker = parts[1]
            peak_date = ':'.join(parts[2:])
            key = f'{ticker}:{peak_date}'
            signal = store.get(key)

            if not signal:
                _answer_callback(cb_id, '데이터를 찾을 수 없습니다')
                return

            detail = format_signal_message(signal)
            detected_date = signal.get('detected_date', peak_date)

            if is_photo:
                # 사진 메시지 → editMessageCaption (차트 이미 표시됨)
                kb = _keyboard([_compact_button(ticker, peak_date)])
                _edit_message_caption(chat_id, message_id, detail,
                                     parse_mode='HTML', reply_markup=kb)
            else:
                # 텍스트 메시지(요약) → 차트+상세를 새 메시지로 전송
                chart_bytes = _load_chart(ticker, peak_date)
                if chart_bytes:
                    kb = _keyboard([_close_button()])
                    _send_photo(chart_bytes, caption=detail, parse_mode='HTML',
                                reply_markup=kb, chat_id=chat_id)
                else:
                    # 차트 없으면 텍스트만
                    kb = _keyboard([_close_button()])
                    _send_message(detail, parse_mode='HTML', reply_markup=kb)

            _answer_callback(cb_id, f'{ticker} 상세')

        elif action == 'c' and len(parts) >= 3:
            # 컴팩트로 복귀 (사진 메시지 캡션)
            ticker = parts[1]
            peak_date = ':'.join(parts[2:])
            key = f'{ticker}:{peak_date}'
            signal = store.get(key)

            if not signal:
                _answer_callback(cb_id, '데이터를 찾을 수 없습니다')
                return

            compact = format_signal_compact(signal)
            kb = _keyboard([_detail_button(ticker, peak_date)])
            _edit_message_caption(chat_id, message_id, compact,
                                 parse_mode='HTML', reply_markup=kb)
            _answer_callback(cb_id)

        elif action == 's' and len(parts) >= 2:
            # 전체 요약 복귀
            detected_date = ':'.join(parts[1:])
            summary_key = f'_summary:{detected_date}'
            summary_info = store.get(summary_key)

            if not summary_info:
                _answer_callback(cb_id, '요약을 찾을 수 없습니다')
                return

            signal_keys = summary_info.get('signal_keys', [])
            signals = [store[k] for k in signal_keys if k in store]

            if not signals:
                _answer_callback(cb_id, '신호 데이터 없음')
                return

            summary_msg = format_signals_summary(signals)

            buttons = []
            row = []
            for s in signals:
                row.append((f'\U0001f4ca {s["ticker"]}',
                            f'd:{s["ticker"]}:{s["peak_date"]}'))
                if len(row) == 2:
                    buttons.append(row)
                    row = []
            if row:
                buttons.append(row)

            kb = _keyboard(buttons)
            _edit_message_text(chat_id, message_id, summary_msg,
                              parse_mode='HTML', reply_markup=kb)
            _answer_callback(cb_id, '전체 요약')

        elif action == 'x':
            # 메시지 삭제
            _delete_message(chat_id, message_id)
            _answer_callback(cb_id)

        else:
            _answer_callback(cb_id)

    except Exception as e:
        _answer_callback(cb_id, f'오류: {str(e)[:30]}')


def run_callback_handler(poll_interval=2, timeout=None):
    """콜백 폴링 루프."""
    if not TELEGRAM_BOT_TOKEN:
        print('ERROR: TELEGRAM_BOT_TOKEN required')
        return

    url = TELEGRAM_API_UPDATES.format(token=TELEGRAM_BOT_TOKEN)
    offset = 0
    start = time.time()
    print(f'Callback handler started (poll={poll_interval}s)')

    while True:
        if timeout and (time.time() - start) > timeout:
            print(f'Timeout ({timeout}s) reached')
            break

        try:
            params = {'offset': offset, 'timeout': 30,
                      'allowed_updates': ['callback_query']}
            resp = requests.get(url, params=params, timeout=35)
            resp.raise_for_status()
            updates = resp.json().get('result', [])

            for update in updates:
                offset = update['update_id'] + 1
                if 'callback_query' in update:
                    cb = update['callback_query']
                    user = cb.get('from', {})
                    data = cb.get('data', '')
                    print(f'  Callback: {data} from {user.get("first_name", "?")}')
                    handle_callback(update)

        except requests.exceptions.Timeout:
            continue
        except KeyboardInterrupt:
            print('\nStopped by user')
            break
        except Exception as e:
            print(f'  Error: {e}')
            time.sleep(poll_interval)


# ════════════════════════════════════════════
# 기존 인터페이스
# ════════════════════════════════════════════

def send_scan_summary(results):
    """스캔 결과 요약 전송 (HTML) + Dashboard 버튼."""
    msg = format_scan_summary(results)
    dash_btn = _dashboard_button()
    if dash_btn:
        kb = _keyboard([dash_btn])
        return _send_message(msg, parse_mode='HTML', reply_markup=kb)
    return _send_message(msg, parse_mode='HTML')


def send_test_message():
    """테스트 메시지 전송."""
    return _send_message('\U0001f916 V4_wP Realtime Alert System: Test message OK!')
