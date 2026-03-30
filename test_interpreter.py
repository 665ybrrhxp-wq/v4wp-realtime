"""AI Multi-Persona Interpreter 로컬 테스트 스크립트.

실행 방법:
  cd v4wp_rev4
  python test_interpreter.py

필요 조건:
  - .env 파일에 ANTHROPIC_API_KEY (또는 CLAUDE_API_KEY) 설정
  - pip install anthropic>=0.42 pydantic>=2.0
"""

import json
import sys
import os

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(__file__))

from v4wp_realtime.ai.signal_interpreter import generate_interpretation

# 샘플 신호 데이터 (NVDA 매수 신호 예시)
SAMPLE_SIGNAL = {
    'ticker': 'NVDA',
    'sector': 'Technology',
    'signal_type': 'buy',
    'signal_tier': 'CONFIRMED',
    'start_val': 0.0312,
    'peak_val': 0.0478,
    'close_price': 118.45,
    's_force': 0.4523,
    's_div': 0.2814,
    'dd_pct': 8.32,
    'duration': 3,
    'peak_date': '2026-03-20',
}

SAMPLE_CONTEXT = {
    'score_history': [
        {'date': '2026-03-11', 'score': 0.0000, 's_force': -0.12, 's_div': 0.00},
        {'date': '2026-03-12', 'score': 0.0000, 's_force': 0.05, 's_div': 0.00},
        {'date': '2026-03-13', 'score': 0.0000, 's_force': 0.18, 's_div': 0.08},
        {'date': '2026-03-14', 'score': 0.0000, 's_force': 0.25, 's_div': 0.12},
        {'date': '2026-03-17', 'score': 0.0178, 's_force': 0.31, 's_div': 0.19},
        {'date': '2026-03-18', 'score': 0.0245, 's_force': 0.38, 's_div': 0.24},
        {'date': '2026-03-19', 'score': 0.0312, 's_force': 0.42, 's_div': 0.27},
        {'date': '2026-03-20', 'score': 0.0478, 's_force': 0.45, 's_div': 0.28},
    ],
}


def main():
    print("=" * 60)
    print("  V4_wP AI Multi-Persona Interpreter Test")
    print("=" * 60)
    print()

    # API key 확인
    from v4wp_realtime.config.settings import CLAUDE_API_KEY
    if not CLAUDE_API_KEY:
        print("[ERROR] API key not found!")
        print("  .env 파일에 ANTHROPIC_API_KEY=sk-ant-... 를 설정하세요.")
        print(f"  .env 경로: {os.path.join(os.path.dirname(__file__), 'v4wp_realtime', '.env')}")
        return

    print(f"[OK] API key found: {CLAUDE_API_KEY[:12]}...")
    print(f"[INFO] Ticker: {SAMPLE_SIGNAL['ticker']}")
    print(f"[INFO] Score: {SAMPLE_SIGNAL['peak_val']:.4f}, "
          f"S_Force: {SAMPLE_SIGNAL['s_force']:.4f}, "
          f"S_Div: {SAMPLE_SIGNAL['s_div']:.4f}")
    print()
    print("[CALL] generate_interpretation() ...")
    print()

    result = generate_interpretation(SAMPLE_SIGNAL, SAMPLE_CONTEXT)

    if result is None:
        print("[FAIL] None returned -- API error or parse failure")
        return

    print("[OK] Interpretation received!")
    print()

    # Verdict + Confidence
    print(f"  Verdict:    {result['final_verdict']}")
    print(f"  Confidence: {result['confidence_score']}%")
    print()

    # 각 페르소나 출력
    for key in ['force_expert', 'div_expert', 'chairman']:
        p = result[key]
        dots = '*' * p['conviction'] + '.' * (5 - p['conviction'])
        print(f"  [{p['persona_name']}] conviction={p['conviction']}/5 [{dots}]")
        print(f"    {p['analysis']}")
        print(f"    >> {p['key_point']}")
        print()

    # Risk note
    print(f"  Risk: {result['risk_note']}")
    print()

    # Confidence 검증
    f_c = result['force_expert']['conviction']
    d_c = result['div_expert']['conviction']
    ch_c = result['chairman']['conviction']
    expected = round((f_c * 30 + d_c * 30 + ch_c * 40) / 100 * 20)
    actual = result['confidence_score']
    match = abs(expected - actual) <= 2  # 2점 오차 허용
    print(f"  Confidence check: F={f_c} D={d_c} C={ch_c} => expected={expected}, actual={actual}, {'OK' if match else 'MISMATCH'}")

    # JSON 전체 출력
    print()
    print("-" * 60)
    print("Full JSON:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
