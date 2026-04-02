"""Post-Mortem 엔진: 시그널 발동 후 실제 수익률 추적 + 페르소나 정확도 채점."""
import json


def run_postmortem(conn):
    """return_5d/20d/90d가 NULL인 시그널에 대해 forward return 계산.

    daily_scores에서 peak_date 이후 실존하는 N번째 행 기준 (거래일 기반).
    매 daily_scan 실행 시 호출.

    Returns:
        dict: {"updated_5d": int, "updated_20d": int, "updated_90d": int}
    """
    stats = {"updated_5d": 0, "updated_20d": 0, "updated_90d": 0}

    # return이 하나라도 NULL인 시그널 조회
    pending = conn.execute(
        """SELECT id, ticker, peak_date, close_price, signal_type,
                  return_5d, return_20d, return_90d, postmortem, interpretation
           FROM signal_events
           WHERE (return_5d IS NULL OR return_20d IS NULL OR return_90d IS NULL)
             AND close_price > 0"""
    ).fetchall()

    for sig in pending:
        sig_id = sig['id']
        ticker = sig['ticker']
        peak_date = sig['peak_date']
        entry_price = sig['close_price']

        if not entry_price or entry_price <= 0:
            continue

        # peak_date 이후 거래일 가격 (최대 90일분)
        future_prices = conn.execute(
            """SELECT close_price FROM daily_scores
               WHERE ticker = ? AND date > ?
               ORDER BY date ASC LIMIT 90""",
            (ticker, peak_date)
        ).fetchall()

        n_available = len(future_prices)
        updates = {}

        # 5거래일 (OFFSET 4 = 5번째 행)
        if sig['return_5d'] is None and n_available >= 5:
            r5 = (future_prices[4]['close_price'] - entry_price) / entry_price * 100
            updates['return_5d'] = round(r5, 2)
            stats['updated_5d'] += 1

        # 20거래일
        if sig['return_20d'] is None and n_available >= 20:
            r20 = (future_prices[19]['close_price'] - entry_price) / entry_price * 100
            updates['return_20d'] = round(r20, 2)
            stats['updated_20d'] += 1

        # 90거래일 → 최종 판정
        if sig['return_90d'] is None and n_available >= 90:
            r90 = (future_prices[89]['close_price'] - entry_price) / entry_price * 100
            updates['return_90d'] = round(r90, 2)

            # max_dd_90d: 90거래일 내 최대 낙폭
            min_price = min(p['close_price'] for p in future_prices[:90])
            max_dd = (min_price - entry_price) / entry_price * 100
            updates['max_dd_90d'] = round(max_dd, 2)

            # 최종 판정 생성
            pm = _build_postmortem(sig, updates, future_prices)
            updates['postmortem'] = json.dumps(pm, ensure_ascii=False)
            stats['updated_90d'] += 1

        if updates:
            set_clause = ", ".join(f"{k} = ?" for k in updates)
            values = list(updates.values()) + [sig_id]
            conn.execute(
                f"UPDATE signal_events SET {set_clause} WHERE id = ?",
                values
            )

    if any(v > 0 for v in stats.values()):
        conn.commit()

    return stats


def _build_postmortem(sig, updates, future_prices):
    """90d return이 채워졌을 때 최종 판정 JSON 생성."""
    r5 = updates.get('return_5d') or sig['return_5d']
    r20 = updates.get('return_20d') or sig['return_20d']
    r90 = updates['return_90d']
    max_dd = updates.get('max_dd_90d', 0)

    outcome = "WIN" if r90 > 0 else "LOSS"

    # interpretation에서 verdict/conviction 추출
    interp = _parse_interpretation(sig['interpretation'])
    verdict = interp.get('final_verdict', 'BUY')
    verdict_accuracy = _judge_verdict(verdict, r90)

    persona_scores = {}
    for key in ['force_expert', 'div_expert', 'chairman']:
        p = interp.get(key, {})
        conviction = p.get('conviction', 3)
        accuracy = _judge_persona(conviction, r90)
        persona_scores[key] = {
            "conviction": conviction,
            "accuracy": accuracy,
            "detail": f"확신도 {conviction} → 90일 {r90:+.1f}% ({accuracy})",
        }

    return {
        "return_5d": r5,
        "return_20d": r20,
        "return_90d": r90,
        "max_dd_90d": max_dd,
        "outcome": outcome,
        "verdict_accuracy": verdict_accuracy,
        "persona_scores": persona_scores,
    }


def _parse_interpretation(interp_str):
    """interpretation JSON 문자열 파싱. 실패 시 빈 dict."""
    if not interp_str:
        return {}
    try:
        return json.loads(interp_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def _judge_verdict(verdict, r90):
    """final_verdict vs 실제 90d 수익률 판정.

    | Verdict       | 기대     | 실제 ≥ 기대 | 실제 < 기대 but > 0 | 실제 < 0      |
    |STRONG_BUY     | +15%     | CORRECT      | OVERCONFIDENT       | OVERCONFIDENT |
    |BUY            | +5%      | CORRECT      | OVERCONFIDENT       | OVERCONFIDENT |
    |CAUTIOUS_BUY   | 0%       | CORRECT      | CORRECT             | OVERCONFIDENT |
    |HOLD           | N/A      | UNDERCONFIDENT| UNDERCONFIDENT     | CORRECT       |
    """
    thresholds = {
        'STRONG_BUY': 15.0,
        'BUY': 5.0,
        'CAUTIOUS_BUY': 0.0,
    }
    if verdict == 'HOLD':
        return "CORRECT" if r90 <= 0 else "UNDERCONFIDENT"

    threshold = thresholds.get(verdict, 5.0)  # 기본 BUY
    if r90 >= threshold:
        return "CORRECT"
    else:
        return "OVERCONFIDENT"


def _judge_persona(conviction, r90):
    """conviction vs 실제 90d 수익률로 페르소나 정확도 판정.

    | Conviction | 수익 > 0                                    | 손실 ≤ 0                          |
    |4-5 (높음)  | CORRECT                                    | OVERCONFIDENT                    |
    |3 (중간)    | r90 > +15% → UNDERCONFIDENT, else CORRECT  | r90 < -5% → OVERCONFIDENT, else CORRECT |
    |1-2 (낮음)  | r90 > +5% → UNDERCONFIDENT                 | CORRECT                          |
    """
    if conviction >= 4:
        return "CORRECT" if r90 > 0 else "OVERCONFIDENT"
    elif conviction == 3:
        if r90 > 0:
            return "UNDERCONFIDENT" if r90 > 15 else "CORRECT"
        else:
            return "OVERCONFIDENT" if r90 < -5 else "CORRECT"
    else:  # 1-2
        if r90 > 0:
            return "UNDERCONFIDENT" if r90 > 5 else "CORRECT"
        else:
            return "CORRECT"


def run_decay_analysis(conn):
    """시그널 발동 후 5거래일 스코어 감쇠 분석.

    daily_scores에서 peak_date 이후 5거래일의 score를 조회하여,
    시그널이 유지되는지(CONFIRMED) 또는 즉시 소멸하는지(FALSE_POSITIVE) 분류.

    분류 기준:
      - score_5d_avg >= peak_val × 0.50 → CONFIRMED (시그널 유지)
      - score_5d_avg >= peak_val × 0.20 → FADING (감쇠 중, 관찰 필요)
      - score_5d_avg <  peak_val × 0.20 → FALSE_POSITIVE (위양성 의심)

    Returns:
        dict: {"classified": int, "confirmed": int, "fading": int, "false_positive": int}
    """
    stats = {"classified": 0, "confirmed": 0, "fading": 0, "false_positive": 0}

    # decay_class가 NULL이고 peak_val > 0인 시그널 조회
    pending = conn.execute(
        """SELECT id, ticker, peak_date, peak_val
           FROM signal_events
           WHERE decay_class IS NULL AND peak_val > 0"""
    ).fetchall()

    for sig in pending:
        sig_id = sig['id']
        ticker = sig['ticker']
        peak_date = sig['peak_date']
        peak_val = sig['peak_val']

        # peak_date 이후 5거래일 score 조회
        future_scores = conn.execute(
            """SELECT score, s_force, s_div FROM daily_scores
               WHERE ticker = ? AND date > ?
               ORDER BY date ASC LIMIT 5""",
            (ticker, peak_date)
        ).fetchall()

        if len(future_scores) < 5:
            continue  # 아직 5거래일이 경과하지 않음

        scores = [r['score'] for r in future_scores if r['score'] is not None]
        if not scores:
            continue

        avg_score = sum(scores) / len(scores)
        ratio = avg_score / peak_val if peak_val > 0 else 0

        if ratio >= 0.50:
            decay_class = 'CONFIRMED'
            stats['confirmed'] += 1
        elif ratio >= 0.20:
            decay_class = 'FADING'
            stats['fading'] += 1
        else:
            decay_class = 'FALSE_POSITIVE'
            stats['false_positive'] += 1

        conn.execute(
            "UPDATE signal_events SET decay_class = ?, score_5d_avg = ? WHERE id = ?",
            (decay_class, round(avg_score, 6), sig_id)
        )
        stats['classified'] += 1

    if stats['classified'] > 0:
        conn.commit()

    return stats


def get_decay_context(conn, ticker, peak_date):
    """특정 시그널의 decay 분류 + 5일 스코어 추이를 반환 (AI 해석기용).

    Returns:
        dict or None: {
            "decay_class": str,
            "score_5d_avg": float,
            "score_trend": [float, ...],  # 5일간 score 리스트
            "s_force_trend": [float, ...],
            "s_div_trend": [float, ...],
        }
    """
    row = conn.execute(
        """SELECT decay_class, score_5d_avg FROM signal_events
           WHERE ticker = ? AND peak_date = ? AND decay_class IS NOT NULL""",
        (ticker, peak_date)
    ).fetchone()

    if not row:
        return None

    future_scores = conn.execute(
        """SELECT score, s_force, s_div FROM daily_scores
           WHERE ticker = ? AND date > ?
           ORDER BY date ASC LIMIT 5""",
        (ticker, peak_date)
    ).fetchall()

    return {
        "decay_class": row['decay_class'],
        "score_5d_avg": row['score_5d_avg'],
        "score_trend": [r['score'] or 0 for r in future_scores],
        "s_force_trend": [r['s_force'] or 0 for r in future_scores],
        "s_div_trend": [r['s_div'] or 0 for r in future_scores],
    }


def get_postmortem_stats(conn, ticker=None):
    """Post-mortem 집계 통계.

    Returns:
        dict: {
            "signals": [...],
            "win_rate": float,
            "avg_return_90d": float,
            "total_completed": int,
            "persona_accuracy": {"force_expert": {"correct": n, "total": n}, ...}
        }
    """
    query = """SELECT * FROM signal_events WHERE return_90d IS NOT NULL"""
    params = []
    if ticker:
        query += " AND ticker = ?"
        params.append(ticker)
    query += " ORDER BY peak_date DESC"

    rows = conn.execute(query, params).fetchall()
    if not rows:
        return {
            "signals": [],
            "win_rate": 0,
            "avg_return_90d": 0,
            "total_completed": 0,
            "persona_accuracy": {},
        }

    signals = []
    wins = 0
    total_r90 = 0
    persona_stats = {
        "force_expert": {"correct": 0, "total": 0},
        "div_expert": {"correct": 0, "total": 0},
        "chairman": {"correct": 0, "total": 0},
    }

    for row in rows:
        pm = _parse_interpretation(row['postmortem'])
        sig_info = {
            "ticker": row['ticker'],
            "peak_date": row['peak_date'],
            "close_price": row['close_price'],
            "return_5d": row['return_5d'],
            "return_20d": row['return_20d'],
            "return_90d": row['return_90d'],
            "max_dd_90d": row['max_dd_90d'],
            "outcome": pm.get('outcome', 'UNKNOWN'),
            "verdict_accuracy": pm.get('verdict_accuracy', 'UNKNOWN'),
            "persona_scores": pm.get('persona_scores', {}),
        }
        signals.append(sig_info)

        if pm.get('outcome') == 'WIN':
            wins += 1
        total_r90 += row['return_90d']

        for key in persona_stats:
            ps = pm.get('persona_scores', {}).get(key, {})
            if ps.get('accuracy'):
                persona_stats[key]['total'] += 1
                if ps['accuracy'] == 'CORRECT':
                    persona_stats[key]['correct'] += 1

    n = len(rows)
    return {
        "signals": signals,
        "win_rate": round(wins / n * 100, 1) if n else 0,
        "avg_return_90d": round(total_r90 / n, 2) if n else 0,
        "total_completed": n,
        "persona_accuracy": persona_stats,
    }
