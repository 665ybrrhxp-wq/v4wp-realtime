<<<<<<< HEAD
"""Telegram 메시지 포맷팅 (Duration 기반 확인 매수/매도)"""
=======
"""Telegram 메시지 포맷팅 (C25 카드형)"""
>>>>>>> 187a32a6aa96e6dada11f8fbf85eaa48a75ec451


def format_signal_message(signal):
    """신호 데이터를 Telegram 카드형 메시지로 포맷.

    Args:
<<<<<<< HEAD
        signal: dict with keys: ticker, sector, signal_type, peak_val, start_val,
                duration, close_price, s_force, s_div, s_conc, peak_date, commentary,
                signal_label, signal_tier, is_strong, action_pct
    """
    label = signal.get('signal_label', signal['signal_type'].upper())
    tier = signal.get('signal_tier', 'CONFIRMED')
    action_pct = signal.get('action_pct')
    sig_type = signal['signal_type']
    duration = signal.get('duration', 0)

    if sig_type == 'bottom':
        header_emoji = '\U0001f525' if tier == 'CONFIRMED' else '\U0001f7e2'
        action_text = f'\uac00\uc6a9\uc790\uae08\uc758 {action_pct:.0%} \ub9e4\uc218' if action_pct else ''
    else:
        header_emoji = '\U0001f534' if tier == 'SELL_CONFIRMED' else '\U0001f7e1'
        action_text = f'\ubcf4\uc720\ub7c9\uc758 {action_pct:.0%} \ub9e4\ub3c4' if action_pct else ''

    score_val = signal.get('start_val', signal.get('peak_val', 0))
=======
        signal: dict with keys: ticker, sector, signal_type, peak_val,
                close_price, s_force, s_div, s_conc, peak_date, commentary,
                signal_label, is_strong, action_pct
    """
    label = signal.get('signal_label', signal['signal_type'].upper())
    is_strong = signal.get('is_strong', False)
    action_pct = signal.get('action_pct')
    sig_type = signal['signal_type']

    # 신호 타입별 이모지 + 액션 텍스트
    if sig_type == 'bottom':
        header_emoji = '\u2b50' if is_strong else '\U0001f7e2'
        action_text = f'\uac00\uc6a9\uc790\uae08\uc758 {action_pct:.0%} \ub9e4\uc218' if action_pct else ''
    else:
        header_emoji = '\U0001f6a8' if is_strong else '\U0001f534'
        action_text = f'\ubcf4\uc720\ub7c9\uc758 {action_pct:.0%} \ub9e4\ub3c4' if action_pct else ''

    # 스코어 방향 표시
    score_val = signal['peak_val']
>>>>>>> 187a32a6aa96e6dada11f8fbf85eaa48a75ec451
    score_bar = _score_bar(score_val)

    lines = [
        '\u2501' * 20,
        f'{header_emoji} {label}',
        '\u2501' * 20,
        '',
        f'\U0001f4cc {signal["ticker"]} ({signal.get("sector", "")})',
        f'\U0001f4b2 ${signal["close_price"]:.2f}',
        '',
        f'\U0001f4ca V4 Score: {score_val:.3f}  {score_bar}',
<<<<<<< HEAD
        f'\u23f1 Duration: {duration}d',
=======
>>>>>>> 187a32a6aa96e6dada11f8fbf85eaa48a75ec451
        f'\u2523 Force:  {signal["s_force"]:+.2f}',
        f'\u2523 Div:    {signal["s_div"]:+.2f}',
        f'\u2517 Conc:   {signal["s_conc"]:+.2f}',
    ]

    if action_text:
        lines.append('')
        lines.append(f'\U0001f4b0 Action: {action_text}')

    lines.append(f'\U0001f4c5 {signal["peak_date"]}')

    if signal.get('commentary'):
        lines.append('')
        lines.append(f'\U0001f4ac {signal["commentary"]}')

    lines.append('\u2501' * 20)

    return '\n'.join(lines)


def _score_bar(score, width=10):
    """스코어를 시각적 바로 표현. -1 ~ +1 범위."""
    clamped = max(-1.0, min(1.0, score))
    mid = width // 2
    pos = int((clamped + 1) / 2 * width)
    bar = ['\u2591'] * width
    bar[mid] = '\u2502'
    if pos < mid:
        for i in range(pos, mid):
            bar[i] = '\u2593'
    elif pos > mid:
        for i in range(mid + 1, min(pos + 1, width)):
            bar[i] = '\u2593'
    return '[' + ''.join(bar) + ']'


def format_scan_summary(results):
<<<<<<< HEAD
    """스캔 결과 요약 메시지 (Duration 기반 카드형)"""
    n_new = len(results['new_signals'])
    n_err = len(results['errors'])
    n_blocked = len(results.get('blocked_sells', []))
    n_blocked_buys = len(results.get('blocked_buys', []))
=======
    """스캔 결과 요약 메시지 (C25 카드형)"""
    n_new = len(results['new_signals'])
    n_err = len(results['errors'])
    n_blocked = len(results.get('blocked_sells', []))

    strong_signals = [s for s in results['new_signals'] if s.get('is_strong')]
    normal_signals = [s for s in results['new_signals'] if not s.get('is_strong')]
>>>>>>> 187a32a6aa96e6dada11f8fbf85eaa48a75ec451

    buy_signals = [s for s in results['new_signals'] if s.get('signal_type') == 'bottom']
    sell_signals = [s for s in results['new_signals'] if s.get('signal_type') == 'top']

    lines = [
        '\u2501' * 20,
<<<<<<< HEAD
        f'\U0001f4ca V4_wP Daily Report',
=======
        f'\U0001f4ca V4_wP C25 Daily Report',
>>>>>>> 187a32a6aa96e6dada11f8fbf85eaa48a75ec451
        '\u2501' * 20,
        '',
        f'\U0001f4c5 {results["date"]}',
        f'\U0001f50d \uc2a4\uce94: {results["scanned"]}\uc885\ubaa9 | \uc18c\uc694: {results["duration_sec"]:.1f}s',
    ]

    if n_err > 0:
        lines.append(f'\u26a0\ufe0f \uc624\ub958: {n_err}\uac74')

    lines.append('')

    if n_new > 0:
        lines.append(f'\U0001f4e2 \uc2e0\uaddc \uc2e0\ud638: {n_new}\uac74')
        if buy_signals:
<<<<<<< HEAD
            tickers = ', '.join(f'{s["ticker"]}({s.get("duration",0)}d)' for s in buy_signals)
            lines.append(f'  \U0001f525 \ub9e4\uc218(CONFIRMED): {len(buy_signals)}\uac74 ({tickers})')
        if sell_signals:
            tickers = ', '.join(f'{s["ticker"]}({s.get("duration",0)}d)' for s in sell_signals)
            lines.append(f'  \U0001f534 \ub9e4\ub3c4(CONFIRMED): {len(sell_signals)}\uac74 ({tickers})')
    else:
        lines.append('\u2705 \uc624\ub298 \uc2e0\uaddc \uc2e0\ud638 \uc5c6\uc74c')

    if n_blocked_buys > 0:
        lines.append(f'  \U0001f6ab \ub9e4\uc218 \ucc28\ub2e8(DD_GATE): {n_blocked_buys}\uac74')
=======
            tickers = ', '.join(s['ticker'] for s in buy_signals)
            lines.append(f'  \U0001f7e2 \ub9e4\uc218: {len(buy_signals)}\uac74 ({tickers})')
        if sell_signals:
            tickers = ', '.join(s['ticker'] for s in sell_signals)
            lines.append(f'  \U0001f534 \ub9e4\ub3c4: {len(sell_signals)}\uac74 ({tickers})')
        if strong_signals:
            lines.append(f'  \u2b50 \uac15\ud55c \uc2e0\ud638: {len(strong_signals)}\uac74')
    else:
        lines.append('\u2705 \uc624\ub298 \uc2e0\uaddc \uc2e0\ud638 \uc5c6\uc74c')

>>>>>>> 187a32a6aa96e6dada11f8fbf85eaa48a75ec451
    if n_blocked > 0:
        lines.append(f'  \U0001f6ab \ub9e4\ub3c4 \ucc28\ub2e8(LATE_SELL): {n_blocked}\uac74')

    lines.append('')
    lines.append('\u2501' * 20)

    return '\n'.join(lines)
