"""유사 시그널 검색 엔진: 코사인 유사도 기반 과거 시그널 매칭."""
import numpy as np


def _build_vector(sig):
    """시그널에서 특성 벡터 구축 (7D-VIXrepl).

    Peak_Val 제거 (S_Force×S_Div 파생 → VIF=18).
    Sector ETF → VIX 20일 변화로 교체 (Mkt/Sec 상관 r=0.79 해소, VIF max=1.86).

    7차원 (full context):
      [s_force, s_div, start/peak ratio,
       dd_pct_norm, duration_norm, market_norm, vix_norm]
    5차원 폴백 (market/vix 없음):
      [s_force, s_div, start/peak ratio, dd_pct_norm, duration_norm]
    2차원 폴백 (dd_pct/duration도 NULL):
      [s_force, s_div]
    """
    s_force = sig['s_force'] or 0
    s_div = sig['s_div'] or 0
    peak_val = sig['peak_val'] or 0
    start_val = sig.get('start_val') or 0
    ratio = (start_val / peak_val) if peak_val > 0 else 0

    dd_pct = sig.get('dd_pct')
    duration = sig.get('duration')

    if dd_pct is not None and duration is not None:
        dd_norm = min(dd_pct / 0.30, 1.0) if dd_pct else 0
        dur_norm = min(duration / 30.0, 1.0) if duration else 0

        mkt_ret = sig.get('market_return_20d')
        vix_chg = sig.get('vix_change_20d')

        if mkt_ret is not None and vix_chg is not None:
            mkt_norm = float(np.tanh(mkt_ret / 0.10))
            # VIX: 변동폭이 크므로 30% 기준 정규화
            vix_norm = float(np.tanh(vix_chg / 0.30))
            return np.array([s_force, s_div, ratio,
                             dd_norm, dur_norm, mkt_norm, vix_norm])
        else:
            return np.array([s_force, s_div, ratio, dd_norm, dur_norm])
    else:
        return np.array([s_force, s_div])


def _cosine_similarity(a, b):
    """코사인 유사도 (차원이 다르면 공통 차원만 사용)."""
    min_dim = min(len(a), len(b))
    a, b = a[:min_dim], b[:min_dim]
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def find_similar_signals(conn, signal, top_n=5, lookback_days=365, exclude_id=None):
    """현재 시그널과 유사한 과거 시그널 Top N 검색 (크로스 티커).

    Args:
        conn: SQLite connection
        signal: dict with s_force, s_div, peak_val, start_val, dd_pct, duration
        top_n: 반환할 유사 시그널 수
        lookback_days: 최근 N일 내 시그널만 대상
        exclude_id: 제외할 시그널 ID (자기 자신)

    Returns:
        list of dict: [{ticker, peak_date, similarity, return_90d, outcome, ...}, ...]
    """
    current_vec = _build_vector(signal)

    # lookback 범위 내 과거 시그널 조회
    rows = conn.execute(
        """SELECT id, ticker, peak_date, peak_val, start_val, close_price,
                  s_force, s_div, dd_pct, duration,
                  market_return_20d, sector_return_20d, vix_change_20d,
                  return_5d, return_20d, return_90d, max_dd_90d, postmortem
           FROM signal_events
           WHERE peak_date >= date('now', ?)
           ORDER BY peak_date ASC""",
        (f'-{lookback_days} days',)
    ).fetchall()

    candidates = []
    for row in rows:
        if exclude_id and row['id'] == exclude_id:
            continue

        row_dict = dict(row)
        candidate_vec = _build_vector(row_dict)
        sim = _cosine_similarity(current_vec, candidate_vec)

        candidates.append({
            "id": row['id'],
            "ticker": row['ticker'],
            "peak_date": row['peak_date'],
            "peak_val": row['peak_val'],
            "close_price": row['close_price'],
            "s_force": row['s_force'],
            "s_div": row['s_div'],
            "dd_pct": row['dd_pct'],
            "duration": row['duration'],
            "similarity": round(sim, 4),
            "return_5d": row['return_5d'],
            "return_20d": row['return_20d'],
            "return_90d": row['return_90d'],
            "max_dd_90d": row['max_dd_90d'],
            "outcome": _extract_outcome(row['postmortem']),
        })

    # 유사도 내림차순 정렬, Top N
    candidates.sort(key=lambda x: x['similarity'], reverse=True)
    return candidates[:top_n]


def _extract_outcome(postmortem_str):
    """postmortem JSON에서 outcome 추출."""
    if not postmortem_str:
        return None
    try:
        import json
        pm = json.loads(postmortem_str)
        return pm.get('outcome')
    except (ValueError, TypeError):
        return None


def build_similar_signals_context(similar_signals):
    """AI 프롬프트용 유사 시그널 텍스트 생성.

    Returns:
        str or None: 유사 시그널이 있으면 텍스트, 없으면 None
    """
    if not similar_signals:
        return None

    completed = [s for s in similar_signals if s.get('return_90d') is not None]
    wins = sum(1 for s in completed if s['return_90d'] > 0)
    avg_r90 = sum(s['return_90d'] for s in completed) / len(completed) if completed else 0

    lines = []
    if completed:
        lines.append(f"과거 유사 시그널 {len(similar_signals)}건 중 "
                      f"결과 확인 {len(completed)}건: "
                      f"승률 {wins}/{len(completed)}, 평균 90일 수익률 {avg_r90:+.1f}%")
    else:
        lines.append(f"과거 유사 시그널 {len(similar_signals)}건 (아직 90일 미경과, 결과 대기 중)")

    for s in similar_signals:
        r90_str = f"{s['return_90d']:+.1f}%" if s['return_90d'] is not None else "대기중"
        outcome_str = s['outcome'] or "PENDING"
        lines.append(
            f"  - {s['ticker']} {s['peak_date']} | "
            f"유사도 {s['similarity']:.0%} | "
            f"F:{s['s_force']:.2f} D:{s['s_div']:.2f} Score:{s['peak_val']:.4f} | "
            f"90d: {r90_str} ({outcome_str})"
        )

    return "\n".join(lines)
