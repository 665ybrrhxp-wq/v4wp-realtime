"""Telegram 메시지 포맷팅 (HTML mode)"""


def format_signal_compact(signal):
    """사진 캡션용 요약 (단일 신호)."""
    dd_pct = signal.get('dd_pct', 0.0)
    dd_icon = _dd_icon(dd_pct)
    score = signal.get('start_val', signal.get('peak_val', 0))
    action_pct = signal.get('action_pct')

    lines = [
        f'\U0001f7e2 <b>{signal["ticker"]}</b>  BUY',
        '',
        f'<code>${signal["close_price"]:.2f}</code>  '
        f'\u2502  DD <code>{dd_pct:.1%}</code> {dd_icon}',
        f'Score <code>{score:+.3f}</code> {_score_bar(score)}',
    ]

    if action_pct:
        lines.append(f'\u27a1 <b>{action_pct:.0%} 매수</b>  \u2502  {signal["peak_date"]}')

    return '\n'.join(lines)


def format_signal_message(signal):
    """매수 신호 상세 카드."""
    action_pct = signal.get('action_pct')
    duration = signal.get('duration', 0)
    score_val = signal.get('start_val', signal.get('peak_val', 0))

    dd_pct = signal.get('dd_pct', 0.0)
    dd_label = _dd_confidence(dd_pct)

    lines = [
        f'\U0001f7e2 <b>{signal["ticker"]}</b>  BUY',
        '',
        f'\U0001f4b0 가격       <code>${signal["close_price"]:.2f}</code>',
        f'\U0001f4c9 낙폭       <code>{dd_pct:.1%}</code>  {dd_label}',
        f'\U0001f4ca 스코어    <code>{score_val:+.3f}</code>  {_score_bar(score_val)}',
        f'\u23f1 지속       <code>{duration}일</code>',
        '',
        f'\u26a1 Force <code>{signal["s_force"]:+.3f}</code>'
        f'  \u2502  '
        f'\U0001f4d0 Div <code>{signal["s_div"]:+.3f}</code>',
    ]

    if action_pct:
        lines.append('')
        lines.append(
            f'\u27a1 <b>가용자금의 {action_pct:.0%} 매수</b>'
        )

    lines.append(f'\U0001f4c5  {signal["peak_date"]}')

    if signal.get('commentary'):
        lines.append('')
        lines.append(f'\U0001f4a1 <i>{signal["commentary"]}</i>')

    return '\n'.join(lines)


def format_signals_summary(signals):
    """다중 신호 요약 (인라인 키보드 메시지)."""
    n = len(signals)
    lines = [
        f'\U0001f4ca <b>V4_wP 매수 신호 {n}건</b>',
        '',
    ]
    for s in signals:
        dd_pct = s.get('dd_pct', 0.0)
        dd_icon = _dd_icon(dd_pct)
        score = s.get('start_val', s.get('peak_val', 0))
        lines.append(
            f'\U0001f7e2 <b>{s["ticker"]}</b>'
            f'  <code>${s["close_price"]:>8.2f}</code>'
            f'  DD <code>{dd_pct:.1%}</code>{dd_icon}'
            f'  {_score_bar(score)}'
        )
    lines.append('')
    lines.append('\u2195 종목 버튼 \u2192 차트 + 상세 분석')
    return '\n'.join(lines)


def _score_bar(score):
    """스코어 강도를 5단계 게이지 바로 표현."""
    abs_s = min(abs(score), 1.0)
    filled = round(abs_s * 5)
    return '\u25b0' * filled + '\u25b1' * (5 - filled)


def _dd_icon(dd_pct):
    """DD 이모지 (요약용)."""
    if dd_pct >= 0.20:
        return '\U0001f525'
    elif dd_pct >= 0.10:
        return '\U0001f4aa'
    elif dd_pct >= 0.05:
        return '\U0001f44d'
    else:
        return '\u2705'


def _dd_confidence(dd_pct):
    """DD 신뢰도 라벨.

    실험 근거 (2,040 신호, bootstrap 검증):
      3~5%  → 90d +9.1%, Hit 68.6%  (보통)
      5~10% → 90d +9.9%, Hit 68.6%  (양호)
      10~20% → 90d +20.7%, Hit 72.0% (강력, p<0.01)
      20%+  → 90d +33.5%, Hit 61.2%  (극강, 변동성 주의)
    """
    if dd_pct >= 0.20:
        return '\U0001f525 극강(변동주의)'
    elif dd_pct >= 0.10:
        return '\U0001f4aa 강력'
    elif dd_pct >= 0.05:
        return '\U0001f44d 양호'
    else:
        return '\u2705 보통'


def format_watch_alerts(watch_alerts):
    """DD Gate 근접 종목 워치 알림 (복수)."""
    n = len(watch_alerts)
    lines = [
        f'\u23f3 <b>DD Gate Watch {n}건</b>',
        '',
    ]
    for w in watch_alerts:
        dd_pct = w['dd_pct'] * 100
        dd_th = w['dd_threshold'] * 100
        gap = dd_th - dd_pct
        lines.append(
            f'\u23f3 <b>{w["ticker"]}</b>'
            f'  DD <code>{dd_pct:.1f}%</code>'
            f'  (threshold {dd_th:.1f}%,'
            f' <b>{gap:.1f}%p</b> \ub0a8\uc74c)'
        )
    lines.append('')
    lines.append('\U0001f50d DD Gate \ud1b5\uacfc \uc784\ubc15 \u2014 \uc8fc\uc2dc \ud544\uc694')
    return '\n'.join(lines)


def format_market_event(market_event, signals):
    """Cross-Ticker Market-Level Event 알림 포맷."""
    event_type = market_event['event_type']
    n = market_event['n_signals']
    n_sec = market_event['n_sectors']

    type_icon = {
        'MARKET_WIDE': '\U0001f525\U0001f525\U0001f525',
        'BROAD_SIGNAL': '\U0001f525\U0001f525',
        'SECTOR_CLUSTER': '\u26a0',
    }
    type_label = {
        'MARKET_WIDE': '매크로 전환점 감지',
        'BROAD_SIGNAL': '광범위 매수 신호',
        'SECTOR_CLUSTER': '섹터 집중 신호',
    }
    boost_label = {
        'STRONG': '\U0001f4aa 확신도 상향 (3개+ 섹터 분산)',
        'MILD': '\U0001f44d 확신도 소폭 상향 (2개 섹터)',
        'NONE': '\u26a0 동일 섹터 집중 — 상관관계 주의',
    }

    icon = type_icon.get(event_type, '\U0001f4ca')
    label = type_label.get(event_type, event_type)
    boost = boost_label.get(market_event['conviction_boost'], '')

    lines = [
        f'{icon} <b>Market Event: {label}</b>',
        '',
        f'\U0001f4ca <b>{n}종목</b> 동시 매수 신호 | <b>{n_sec}개 섹터</b>',
    ]

    # 섹터별 종목 표시
    for sector, tickers in market_event['sectors'].items():
        lines.append(f'  \u2022 {sector}: {", ".join(tickers)}')

    lines.append('')
    lines.append(boost)

    regime = market_event.get('regime', 'UNKNOWN')
    if regime != 'UNKNOWN' and regime != 'MIXED':
        from v4wp_realtime.core.regime import get_conviction
        conv = get_conviction(regime)
        lines.append(f'\U0001f30d 레짐: {regime} ({conv["label_kr"]})')

    # 종목별 요약
    lines.append('')
    for s in signals:
        score = s.get('start_val', s.get('peak_val', 0))
        dd_pct = s.get('dd_pct', 0)
        lines.append(
            f'\U0001f7e2 <b>{s["ticker"]}</b>'
            f'  <code>${s["close_price"]:>8.2f}</code>'
            f'  DD <code>{dd_pct:.1%}</code>'
            f'  {_score_bar(score)}'
        )

    return '\n'.join(lines)


def format_scan_summary(results):
    """스캔 요약 → Telegram HTML 카드."""
    n_new = len(results['new_signals'])
    n_err = len(results['errors'])
    n_blocked = len(results.get('blocked_buys', []))

    lines = [
        f'\U0001f4ca <b>V4_wP Daily Report</b>',
        f'{results["date"]}  \u2502  '
        f'{results["scanned"]}종목  \u2502  '
        f'{results["duration_sec"]:.0f}s',
        '',
    ]

    if n_err > 0:
        lines.append(f'\u26a0 오류 {n_err}건')

    if n_new > 0:
        tickers = ', '.join(
            f'<b>{s["ticker"]}</b>({s.get("duration",0)}d)'
            for s in results['new_signals']
        )
        lines.append(f'\U0001f7e2 매수 {n_new}건: {tickers}')

        # Market Event 표시
        me = results.get('market_event')
        if me:
            type_label = {
                'MARKET_WIDE': '\U0001f525 매크로 전환점',
                'BROAD_SIGNAL': '\U0001f525 광범위 신호',
                'SECTOR_CLUSTER': '\u26a0 섹터 집중',
            }
            lines.append(f'{type_label.get(me["event_type"], "")} '
                         f'({me["n_sectors"]}개 섹터 동시)')
    else:
        lines.append('\u2705 신규 신호 없음')

    if n_blocked > 0:
        lines.append(f'\U0001f6ab DD\uac8c\uc774\ud2b8 \ucc28\ub2e8: {n_blocked}\uac74')

    n_watch = len(results.get('watch_alerts', []))
    if n_watch > 0:
        watch_tickers = ', '.join(
            f'<b>{w["ticker"]}</b>' for w in results['watch_alerts']
        )
        lines.append(f'\u23f3 DD\uadfc\uc811: {n_watch}\uac74 ({watch_tickers})')

    return '\n'.join(lines)
