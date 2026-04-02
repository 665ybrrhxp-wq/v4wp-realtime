"""Cross-Ticker Correlation Alert: 동시 다발 시그널 감지 + 매크로 전환점 분류.

3개 이상 종목에서 동시에 매수 시그널 발생 시 "Market-Level Event"로 분류.
섹터 다양성에 따라 conviction 가중치를 부여:
  - 단일 섹터 집중 → SECTOR_CLUSTER (같은 베팅, 주의)
  - 2개 섹터 → BROAD_SIGNAL (확산 초기)
  - 3개+ 섹터 → MARKET_WIDE (매크로 전환점 가능성)
"""


def analyze_market_event(new_signals):
    """당일 신규 시그널을 분석하여 Market-Level Event 여부를 판정.

    Args:
        new_signals: list of signal dicts from scanner.py results['new_signals']

    Returns:
        dict or None: None if not a market event, else:
        {
            "is_market_event": True,
            "n_signals": int,
            "n_sectors": int,
            "event_type": str,  # SECTOR_CLUSTER / BROAD_SIGNAL / MARKET_WIDE
            "conviction_boost": str,  # NONE / MILD / STRONG
            "sectors": {sector: [tickers]},
            "regime": str,  # 공통 market_regime (또는 MIXED)
            "summary": str,  # 한줄 요약
        }
    """
    if len(new_signals) < 3:
        return None

    # 섹터별 그룹핑
    sectors = {}
    for s in new_signals:
        sector = s.get('sector', 'Unknown')
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append(s['ticker'])

    n_signals = len(new_signals)
    n_sectors = len(sectors)

    # 공통 레짐 판별
    regimes = [s.get('market_regime', 'UNKNOWN') for s in new_signals]
    unique_regimes = set(r for r in regimes if r != 'UNKNOWN')
    regime = unique_regimes.pop() if len(unique_regimes) == 1 else 'MIXED'

    # 이벤트 유형 + conviction 가중치 결정
    if n_sectors >= 3:
        event_type = 'MARKET_WIDE'
        conviction_boost = 'STRONG'
    elif n_sectors == 2:
        event_type = 'BROAD_SIGNAL'
        conviction_boost = 'MILD'
    else:
        event_type = 'SECTOR_CLUSTER'
        conviction_boost = 'NONE'

    # 요약 생성
    sector_str = ", ".join(f"{k}({len(v)})" for k, v in sectors.items())
    ticker_str = ", ".join(s['ticker'] for s in new_signals)

    type_kr = {
        'MARKET_WIDE': '매크로 전환점',
        'BROAD_SIGNAL': '광범위 매수 신호',
        'SECTOR_CLUSTER': '섹터 집중 신호',
    }
    boost_kr = {
        'STRONG': '확신도 상향',
        'MILD': '확신도 소폭 상향',
        'NONE': '동일 섹터 집중 — 분산 주의',
    }

    summary = (
        f"{type_kr[event_type]}: {n_signals}종목 동시 매수 "
        f"({n_sectors}개 섹터: {sector_str}) "
        f"| {boost_kr[conviction_boost]}"
    )

    return {
        "is_market_event": True,
        "n_signals": n_signals,
        "n_sectors": n_sectors,
        "event_type": event_type,
        "conviction_boost": conviction_boost,
        "sectors": sectors,
        "tickers": ticker_str,
        "regime": regime,
        "summary": summary,
    }


def query_historical_market_events(conn, min_signals=3, lookback_days=365):
    """과거 동시 다발 시그널 이력 조회 + 성과 분석.

    Returns:
        list of dict: [{date, n_signals, tickers, avg_return_90d, win_rate}, ...]
    """
    rows = conn.execute(
        """SELECT peak_date, COUNT(*) as n,
                  GROUP_CONCAT(ticker) as tickers,
                  AVG(return_90d) as avg_r90,
                  SUM(CASE WHEN return_90d > 0 THEN 1 ELSE 0 END) as wins
           FROM signal_events
           WHERE peak_date >= date('now', ?)
             AND signal_type = 'bottom'
           GROUP BY peak_date
           HAVING COUNT(*) >= ?
           ORDER BY peak_date DESC""",
        (f'-{lookback_days} days', min_signals)
    ).fetchall()

    events = []
    for r in rows:
        n = r['n']
        wins = r['wins'] or 0
        events.append({
            "date": r['peak_date'],
            "n_signals": n,
            "tickers": r['tickers'],
            "avg_return_90d": round(r['avg_r90'], 2) if r['avg_r90'] else None,
            "win_rate": round(wins / n * 100, 1) if n > 0 else None,
        })

    return events
