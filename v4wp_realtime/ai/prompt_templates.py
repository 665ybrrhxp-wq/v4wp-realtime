"""Claude API 프롬프트 템플릿"""

SYSTEM_PROMPT = """You are a concise Korean stock market analyst.
Given a V4_wP volume indicator signal, write ONE sentence (max 50 characters in Korean) summarizing the signal's implication.
- Bottom signal: bullish/buy bias
- Top signal: bearish/sell bias
Focus on what the volume pattern suggests about near-term price action.
Reply in Korean only. No emoji. No english."""

SIGNAL_PROMPT_TEMPLATE = """Signal:
- Ticker: {ticker} ({sector})
- Type: {signal_type} ({signal_label})
- Score: {peak_val:.3f}
- Price: ${close_price:.2f}
- Sub-indicators: Force={s_force:.2f}, Div={s_div:.2f}, Conc={s_conc:.2f}
- Date: {peak_date}

Recent score trend (last 5 days):
{score_trend}

Write a single Korean sentence (max 50 chars) about this signal."""


FALLBACK_TEMPLATES = {
    'bottom': '{ticker} 거래량 바닥 신호 감지 — 단기 반등 가능성.',
    'top': '{ticker} 거래량 천장 신호 감지 — 단기 조정 가능성.',
}


def build_prompt(signal, context=None):
    """프롬프트 생성"""
    score_trend = 'N/A'
    if context and context.get('score_history'):
        scores = context['score_history'][-5:]
        trend_lines = [f'  {s["date"]}: {s["score"]:.3f}' for s in scores if s.get('score') is not None]
        score_trend = '\n'.join(trend_lines) if trend_lines else 'N/A'

    signal_label = '매수 신호' if signal['signal_type'] == 'bottom' else '매도 신호'

    return SIGNAL_PROMPT_TEMPLATE.format(
        ticker=signal['ticker'],
        sector=signal.get('sector', ''),
        signal_type=signal['signal_type'],
        signal_label=signal_label,
        peak_val=signal['peak_val'],
        close_price=signal['close_price'],
        s_force=signal['s_force'],
        s_div=signal['s_div'],
        s_conc=signal['s_conc'],
        peak_date=signal['peak_date'],
        score_trend=score_trend,
    )


def get_fallback(signal):
    """API 실패 시 템플릿 기반 폴백"""
    template = FALLBACK_TEMPLATES.get(signal['signal_type'], '{ticker} V4_wP 신호 감지.')
    return template.format(ticker=signal['ticker'])
