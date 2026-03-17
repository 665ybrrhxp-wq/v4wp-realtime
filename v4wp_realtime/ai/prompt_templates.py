"""Claude API 프롬프트 템플릿"""

SYSTEM_PROMPT = """You are a concise Korean stock market analyst.
Given a V4_wP volume indicator buy signal, write ONE sentence (max 50 characters in Korean) summarizing the signal's implication.
Focus on what the volume pattern suggests about near-term bullish price action.
Reply in Korean only. No emoji. No english."""

SIGNAL_PROMPT_TEMPLATE = """Signal:
- Ticker: {ticker} ({sector})
- Type: {signal_type} ({signal_label}) [{signal_tier}]
- Entry Score: {start_val:.3f} (peak: {peak_val:.3f})
- Price: ${close_price:.2f}
- Sub-indicators: Force={s_force:.2f}, Div={s_div:.2f}
- Date: {peak_date}

Recent score trend (last 5 days):
{score_trend}

Write a single Korean sentence (max 50 chars) about this signal."""


FALLBACK_TEMPLATES = {
    'bottom': '{ticker} 거래량 바닥 신호 감지 — 단기 반등 가능성.',
}


def build_prompt(signal, context=None):
    """프롬프트 생성"""
    score_trend = 'N/A'
    if context and context.get('score_history'):
        scores = context['score_history'][-5:]
        trend_lines = [f'  {s["date"]}: {s["score"]:.3f}' for s in scores if s.get('score') is not None]
        score_trend = '\n'.join(trend_lines) if trend_lines else 'N/A'

    signal_label = '매수 신호'
    signal_tier = signal.get('signal_tier', 'NORMAL')

    return SIGNAL_PROMPT_TEMPLATE.format(
        ticker=signal['ticker'],
        sector=signal.get('sector', ''),
        signal_type=signal['signal_type'],
        signal_label=signal_label,
        signal_tier=signal_tier,
        start_val=signal.get('start_val', 0),
        peak_val=signal['peak_val'],
        close_price=signal['close_price'],
        s_force=signal['s_force'],
        s_div=signal['s_div'],
        peak_date=signal['peak_date'],
        score_trend=score_trend,
    )


def get_fallback(signal):
    """API 실패 시 템플릿 기반 폴백"""
    template = FALLBACK_TEMPLATES.get(signal['signal_type'], '{ticker} V4_wP 신호 감지.')
    return template.format(ticker=signal['ticker'])
