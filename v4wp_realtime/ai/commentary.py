"""Claude API를 통한 한 줄 평 생성"""
from v4wp_realtime.config.settings import CLAUDE_API_KEY
from v4wp_realtime.ai.prompt_templates import SYSTEM_PROMPT, build_prompt, get_fallback


def generate_commentary(signal, context=None):
    """신호에 대한 한 줄 평 생성.

    Args:
        signal: dict with signal data
        context: dict with score_history, recent_events

    Returns:
        str: Korean commentary (1 sentence)
    """
    if not CLAUDE_API_KEY:
        return get_fallback(signal)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

        user_prompt = build_prompt(signal, context)

        response = client.messages.create(
            model='claude-haiku-4-5-20241022',
            max_tokens=100,
            system=SYSTEM_PROMPT,
            messages=[{'role': 'user', 'content': user_prompt}],
        )

        text = response.content[0].text.strip()
        # 50자 초과 시 자르기
        if len(text) > 60:
            text = text[:57] + '...'
        return text

    except Exception as e:
        print(f'  [AI] Commentary failed for {signal["ticker"]}: {e}')
        return get_fallback(signal)
