"""텔레그램 수동 테스트 (단일 차트 + 앨범 + 인라인 키보드)"""
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from v4wp_realtime.alerts.telegram_bot import (
    send_test_message, send_signal_alert, send_signal_album,
    run_callback_handler,
)
from v4wp_realtime.config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID


def _make_chart(ticker, df, analysis):
    """차트 BytesIO 생성."""
    from v4wp_realtime.alerts.chart_generator import generate_signal_chart

    subind = analysis['subindicators']
    events = analysis['filtered_events']

    if events:
        ev = events[-1]
    else:
        ev = {
            'peak_idx': len(df) - 1,
            'start_idx': len(df) - 5,
            'end_idx': len(df) - 1,
            'peak_val': 0.1,
            'dd_pct': 0.04,
            'peak_date': df.index[-1].strftime('%Y-%m-%d'),
        }

    chart_bytes = generate_signal_chart(ticker, df, subind, ev)
    return chart_bytes, ev


def _make_signal(ticker, df, ev, subind, sector='Test'):
    """신호 dict 생성."""
    peak_idx = ev['peak_idx']
    from datetime import datetime
    return {
        'ticker': ticker,
        'sector': sector,
        'signal_type': 'bottom',
        'peak_val': float(ev.get('peak_val', 0.1)),
        'start_val': 0.0,
        'close_price': float(df['Close'].iloc[peak_idx]),
        'duration': 3,
        's_force': float(subind['s_force'].iloc[peak_idx]),
        's_div': float(subind['s_div'].iloc[peak_idx]),
        's_conc': 0.0,
        'dd_pct': ev.get('dd_pct', 0.04),
        'peak_date': ev.get('peak_date', df.index[peak_idx].strftime('%Y-%m-%d')),
        'detected_date': datetime.now().strftime('%Y-%m-%d'),
        'action_pct': 1.0,
        'signal_tier': 'CONFIRMED',
        'signal_label': 'BUY (CONFIRMED)',
        'is_strong': True,
        'commentary': f'{ticker} 거래량 반전 — 단기 매수 기회 시사.',
    }


def main():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print('ERROR: Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables')
        return

    from v4wp_realtime.core.indicators import fetch_data, analyze_ticker
    from v4wp_realtime.config.settings import load_watchlist

    wl = load_watchlist()
    params = wl['params']

    # 1) 기본 테스트
    print('1. Sending test message...')
    result = send_test_message()
    print(f'   OK: message_id={result["result"]["message_id"]}')

    # 2) 단일 차트 + 인라인 키보드
    print('\n2. Single chart + inline keyboard (VOO)...')
    df = fetch_data('VOO', years=3)
    if df is not None and len(df) > 200:
        analysis = analyze_ticker('VOO', df, params)
        chart_bytes, ev = _make_chart('VOO', df, analysis)
        signal = _make_signal('VOO', df, ev, analysis['subindicators'], 'Benchmark')

        print(f'   Chart: {chart_bytes.getbuffer().nbytes / 1024:.0f} KB')
        result = send_signal_alert(signal, chart_bytes=chart_bytes)
        print(f'   OK: message_id={result["result"]["message_id"]}')
        print('   (tap [상세 보기] button to toggle)')

    # 3) 앨범 + 요약 메시지 + 인라인 키보드
    print('\n3. Album + summary keyboard (VOO, QQQ, AAPL)...')
    album_tickers = ['VOO', 'QQQ', 'AAPL']
    pairs = []

    for ticker in album_tickers:
        print(f'   Downloading {ticker}...')
        df = fetch_data(ticker, years=3)
        if df is None or len(df) < 200:
            print(f'   SKIP: {ticker} insufficient data')
            continue

        analysis = analyze_ticker(ticker, df, params)
        chart_bytes, ev = _make_chart(ticker, df, analysis)
        signal = _make_signal(ticker, df, ev, analysis['subindicators'])
        pairs.append((signal, chart_bytes))
        print(f'   {ticker} chart: {chart_bytes.getbuffer().nbytes / 1024:.0f} KB')

    if len(pairs) >= 2:
        print(f'   Sending album ({len(pairs)} charts) + summary...')
        results = send_signal_album(pairs)
        print(f'   OK: {len(results)} API calls')
        print('   Album: swipe left/right')
        print('   Summary: tap ticker buttons for detail')
    elif pairs:
        send_signal_alert(pairs[0][0], chart_bytes=pairs[0][1])

    # 4) 콜백 핸들러 (30초간 버튼 테스트)
    print('\n4. Listening for button presses (30 sec)...')
    print('   Tap the buttons in Telegram now!')
    run_callback_handler(timeout=30)

    print('\nDone!')


if __name__ == '__main__':
    main()
