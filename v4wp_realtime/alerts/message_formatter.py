"""Telegram 메시지 포맷팅 (HTML mode)"""


def format_signal_message(signal):
    """개별 매수 신호 → Telegram HTML 카드."""
    action_pct = signal.get('action_pct')
    duration = signal.get('duration', 0)
    score_val = signal.get('start_val', signal.get('peak_val', 0))

    emoji = '\U0001f7e2'  # 🟢
    action_text = f'가용자금의 {action_pct:.0%} 매수' if action_pct else ''
    direction = 'BUY'

    # 스코어 강도 바 (5칸)
    bar = _score_dots(score_val)

    lines = [
        f'{emoji} <b>{signal["ticker"]}</b>  {direction}',
        '',
        f'  가격    <code>${signal["close_price"]:.2f}</code>',
        f'  스코어  <code>{score_val:+.3f}</code>  {bar}',
        f'  지속    <code>{duration}일</code>',
        '',
        f'  Force <code>{signal["s_force"]:+.2f}</code>  '
        f'Div <code>{signal["s_div"]:+.2f}</code>',
    ]

    if action_text:
        lines.append('')
        lines.append(f'  \u27a1 <b>{action_text}</b>')

    lines.append(f'  {signal["peak_date"]}')

    if signal.get('commentary'):
        lines.append('')
        lines.append(f'  <i>{signal["commentary"]}</i>')

    return '\n'.join(lines)


def _score_dots(score):
    """스코어 강도를 5단계 도트로 표현."""
    abs_s = min(abs(score), 1.0)
    filled = round(abs_s * 5)
    return '\U0001f7e2' * filled + '\u26aa' * (5 - filled)  # 🟢⚪


def format_scan_summary(results):
    """스캔 요약 → Telegram HTML 카드."""
    n_new = len(results['new_signals'])
    n_err = len(results['errors'])
    n_blocked_buys = len(results.get('blocked_buys', []))

    lines = [
        f'\U0001f4ca <b>V4_wP Daily Report</b>',
        f'{results["date"]}  |  {results["scanned"]}종목  |  {results["duration_sec"]:.0f}s',
        '',
    ]

    if n_err > 0:
        lines.append(f'\u26a0 오류 {n_err}건')

    if n_new > 0:
        tickers = ', '.join(
            f'<b>{s["ticker"]}</b>({s.get("duration",0)}d)' for s in results['new_signals']
        )
        lines.append(f'\U0001f7e2 매수 {n_new}건: {tickers}')
    else:
        lines.append('\u2705 신규 신호 없음')

    if n_blocked_buys > 0:
        lines.append(f'\U0001f6ab DD게이트 차단: {n_blocked_buys}건')

    return '\n'.join(lines)
