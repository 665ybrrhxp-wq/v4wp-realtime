"""Multi-Persona AI Signal Interpreter using Claude Sonnet with prompt caching.

Uses Anthropic structured output (Pydantic) to guarantee valid JSON.
System prompt is cached across calls within a 5-minute TTL window,
saving ~90% on input token costs for the 2nd-20th ticker in a scan.
"""

from v4wp_realtime.config.settings import CLAUDE_API_KEY
from v4wp_realtime.ai.interpreter_prompt import (
    INTERPRETER_SYSTEM_PROMPT,
    build_interpreter_prompt,
)
from v4wp_realtime.ai.schema import SignalInterpretation

# Module-level client singleton — reuses TCP connection across calls
_client = None


def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    return _client


def generate_interpretation(signal: dict, context: dict = None) -> dict | None:
    """Generate multi-persona interpretation for a buy signal.

    Uses prompt caching: system prompt (~5000 tokens) is cached on first call,
    subsequent calls within the 5-min TTL read from cache (0.1x cost).

    Args:
        signal: dict with ticker, peak_val, s_force, s_div, close_price, etc.
        context: dict with score_history, recent_events

    Returns:
        dict: Serialized SignalInterpretation, or None on failure
    """
    if not CLAUDE_API_KEY:
        return None

    client = _get_client()
    user_prompt = build_interpreter_prompt(signal, context)

    system_block = [{
        'type': 'text',
        'text': INTERPRETER_SYSTEM_PROMPT,
        'cache_control': {'type': 'ephemeral'},
    }]

    # 1st attempt
    try:
        response = client.messages.parse(
            model='claude-sonnet-4-6',
            max_tokens=1024,
            system=system_block,
            messages=[{'role': 'user', 'content': user_prompt}],
            output_format=SignalInterpretation,
        )
        _log_cache_stats(response, signal.get('ticker', ''))
        return response.parsed_output.model_dump()

    except Exception as e1:
        # Retry once with corrective instruction
        try:
            response = client.messages.parse(
                model='claude-sonnet-4-6',
                max_tokens=1024,
                system=system_block,
                messages=[
                    {'role': 'user', 'content': user_prompt},
                    {'role': 'assistant', 'content': '(이전 응답이 스키마를 위반했습니다)'},
                    {'role': 'user', 'content': (
                        'JSON 스키마를 정확히 따라주세요. '
                        'conviction은 1-5 정수, final_verdict는 '
                        'STRONG_BUY/BUY/CAUTIOUS_BUY/HOLD 중 하나, '
                        'confidence_score는 공식에 따라 1-100 정수.'
                    )},
                ],
                output_format=SignalInterpretation,
            )
            _log_cache_stats(response, signal.get('ticker', ''))
            return response.parsed_output.model_dump()

        except Exception as e2:
            print(f'  [AI] Interpretation failed for {signal.get("ticker")}: '
                  f'1st={e1}, 2nd={e2}')
            return None


def _log_cache_stats(response, ticker: str):
    """Log prompt cache hit/miss for monitoring."""
    usage = response.usage
    cache_create = getattr(usage, 'cache_creation_input_tokens', 0)
    cache_read = getattr(usage, 'cache_read_input_tokens', 0)
    if cache_create:
        print(f'  [AI] {ticker}: cache WRITE {cache_create} tokens')
    elif cache_read:
        print(f'  [AI] {ticker}: cache HIT {cache_read} tokens')
