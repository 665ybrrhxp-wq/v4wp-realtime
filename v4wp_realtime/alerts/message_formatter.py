"""Telegram 메시지 포맷팅 (HTML mode)"""


def format_signal_message(signal):
    """개별 신호 → Telegram HTML 카드."""
    tier = signal.get('signal_tier', 'CONFIRMED')
    action_pct = signal.get('action_pct')
    sig_type = signal['signal_type']
    duration = signal.get('duration', 0)
    score_val = signal.get('start_val', signal.get('peak_val', 0))

    if sig_type == 'bottom':
        emoji = '\U0001f7e2'  # 🟢
        action_text = f'가용자금의 {action_pct:.0%} 매수' if action_pct else ''
        direction = 'BUY'
    else:
        emoji = '\U0001f534'  # 🔴
        action_text = f'보유량의 {action_pct:.0%} 매도' if action_pct else ''
        direction = 'SELL'

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
        f'Div <code>{signal["s_div"]:+.2f}</code>  '
        f'Conc <code>{signal["s_conc"]:+.2f}</code>',
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
    if score >= 0:
        return '\U0001f7e2' * filled + '\u26aa' * (5 - filled)  # 🟢⚪
    else:
        return '\U0001f534' * filled + '\u26aa' * (5 - filled)  # 🔴⚪


def format_scan_summary(results):
    """스캔 요약 → Telegram HTML 카드."""
    n_new = len(results['new_signals'])
    n_err = len(results['errors'])
    n_blocked_sells = len(results.get('blocked_sells', []))
    n_blocked_buys = len(results.get('blocked_buys', []))

    buy_signals = [s for s in results['new_signals'] if s.get('signal_type') == 'bottom']
    sell_signals = [s for s in results['new_signals'] if s.get('signal_type') == 'top']

    lines = [
        f'\U0001f4ca <b>V4_wP Daily Report</b>',
        f'{results["date"]}  |  {results["scanned"]}종목  |  {results["duration_sec"]:.0f}s',
        '',
    ]

    if n_err > 0:
        lines.append(f'\u26a0 오류 {n_err}건')

    if n_new > 0:
        if buy_signals:
            tickers = ', '.join(
                f'<b>{s["ticker"]}</b>({s.get("duration",0)}d)' for s in buy_signals
            )
            lines.append(f'\U0001f7e2 매수 {len(buy_signals)}건: {tickers}')
        if sell_signals:
            tickers = ', '.join(
                f'<b>{s["ticker"]}</b>({s.get("duration",0)}d)' for s in sell_signals
            )
            lines.append(f'\U0001f534 매도 {len(sell_signals)}건: {tickers}')
    else:
        lines.append('\u2705 신규 신호 없음')

    # 차단 요약 (한 줄로)
    blocked_parts = []
    if n_blocked_buys > 0:
        blocked_parts.append(f'매수 {n_blocked_buys}')
    if n_blocked_sells > 0:
        blocked_parts.append(f'매도 {n_blocked_sells}')
    if blocked_parts:
        lines.append(f'\U0001f6ab 차단: {" / ".join(blocked_parts)}건')

    return '\n'.join(lines)
