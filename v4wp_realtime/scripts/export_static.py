"""DB 데이터를 정적 JSON 파일로 내보내기 (GitHub Pages용)

daily_scan.py 실행 후 호출하여 webapp/dist/data/ 에 JSON 파일 생성.
프론트엔드가 API 서버 대신 이 JSON 파일에서 데이터를 읽음.

생성 파일:
  dist/data/watchlist.json          - GET /api/watchlist 동일
  dist/data/chart/{TICKER}.json     - GET /api/chart-data/{ticker} 동일
  dist/data/indicators/{TICKER}.json - GET /api/indicators/{ticker} 동일
"""
import sys
import json
import math
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from v4wp_realtime.config.settings import DB_PATH, load_watchlist
from v4wp_realtime.data.store import get_connection, init_db

# 출력 디렉토리
DIST_DATA = Path(__file__).resolve().parent.parent / 'webapp' / 'dist' / 'data'


def _safe(val):
    """NaN/Infinity → None (JSON 호환)."""
    if val is None:
        return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return val


def export_watchlist(conn, wl):
    """watchlist.json 생성."""
    tickers = list(wl['tickers'].keys())
    result = []

    for ticker in tickers:
        sector = wl['tickers'][ticker].get('sector', '')

        rows = conn.execute(
            """SELECT date, score, s_force, close_price, er, atr_pct
               FROM daily_scores
               WHERE ticker = ?
               ORDER BY date DESC LIMIT 2""",
            (ticker,),
        ).fetchall()

        if not rows:
            result.append({
                'ticker': ticker, 'sector': sector,
                'close_price': None, 'prev_close': None, 'change_pct': None,
                'score': None, 's_force': None, 'er': None, 'atr_pct': None,
                'date': None, 'recent_signal': None,
            })
            continue

        latest = dict(rows[0])
        prev = dict(rows[1]) if len(rows) > 1 else None

        change_pct = None
        if prev and prev['close_price'] and latest['close_price']:
            change_pct = round(
                (latest['close_price'] - prev['close_price'])
                / prev['close_price'] * 100, 2
            )

        sig_row = conn.execute(
            """SELECT signal_type, peak_date, signal_tier, s_force, peak_val
               FROM signal_events
               WHERE ticker = ?
               ORDER BY peak_date DESC LIMIT 1""",
            (ticker,),
        ).fetchone()

        recent_signal = None
        if sig_row:
            s = dict(sig_row)
            recent_signal = {
                'direction': 'LONG' if s['signal_type'] == 'bottom' else 'SHORT',
                'peak_date': s['peak_date'],
                'tier': s['signal_tier'],
                's_force': _safe(s['s_force']),
                'peak_val': _safe(s['peak_val']),
            }

        result.append({
            'ticker': ticker,
            'sector': sector,
            'date': latest['date'],
            'close_price': _safe(latest['close_price']),
            'prev_close': _safe(prev['close_price']) if prev else None,
            'change_pct': _safe(change_pct),
            'score': _safe(latest['score']),
            's_force': _safe(latest['s_force']),
            'er': _safe(latest.get('er')),
            'atr_pct': _safe(latest.get('atr_pct')),
            'recent_signal': recent_signal,
        })

    out = DIST_DATA / 'watchlist.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)
    print(f'  watchlist.json: {len(result)} tickers')
    return tickers


def export_chart_data(conn, ticker, days=60):
    """chart/{TICKER}.json 생성."""
    rows = conn.execute(
        """SELECT date, close_price, score, s_force, s_div, s_conc, er, atr_pct
           FROM daily_scores
           WHERE ticker = ?
             AND date >= date('now', ?)
           ORDER BY date ASC""",
        (ticker, f'-{days} days'),
    ).fetchall()

    sig_rows = conn.execute(
        """SELECT peak_date AS date, signal_type, close_price AS entry_price,
                  s_force, peak_val, signal_tier, action_pct
           FROM signal_events
           WHERE ticker = ?
             AND peak_date >= date('now', ?)
           ORDER BY peak_date ASC""",
        (ticker, f'-{days} days'),
    ).fetchall()

    signals = []
    for s in [dict(r) for r in sig_rows]:
        s['direction'] = 'LONG' if s['signal_type'] == 'bottom' else 'SHORT'
        # NaN 방지
        for k in s:
            s[k] = _safe(s[k])
        signals.append(s)

    data = []
    for r in [dict(r) for r in rows]:
        for k in r:
            r[k] = _safe(r[k])
        data.append(r)

    result = {'ticker': ticker, 'days': days, 'data': data, 'signals': signals}

    out = DIST_DATA / 'chart' / f'{ticker}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)


def export_indicators(conn, ticker, params=None):
    """indicators/{TICKER}.json 생성."""
    params = params or {}
    sig_th = params.get('signal_threshold', 0.05)
    bot_th = sig_th * 0.5
    dd_lookback = params.get('buy_dd_lookback', 20)
    dd_threshold = params.get('buy_dd_threshold', 0.03)
    confirm_days = params.get('confirm_days', 1)

    row = conn.execute(
        """SELECT date, score, s_force, s_div, close_price, er, atr_pct
           FROM daily_scores
           WHERE ticker = ?
           ORDER BY date DESC LIMIT 1""",
        (ticker,),
    ).fetchone()

    recent = conn.execute(
        """SELECT score, close_price FROM daily_scores
           WHERE ticker = ?
           ORDER BY date DESC LIMIT ?""",
        (ticker, max(dd_lookback, 30)),
    ).fetchall()

    if not row:
        result = {'ticker': ticker, 'data': None, 'score': 0,
                  's_force': 0, 's_div': 0, 'and_geo_active': False,
                  'pipeline': None, 'filters': []}
    else:
        d = dict(row)
        for k in d:
            d[k] = _safe(d[k])

        s_force = d.get('s_force') or 0
        s_div = d.get('s_div') or 0
        score = d.get('score') or 0
        close = d.get('close_price') or 0
        er = d.get('er')
        atr_pct = d.get('atr_pct')

        # 파이프라인 상태
        streak = 0
        for r in recent:
            if r['score'] and r['score'] > bot_th:
                streak += 1
            else:
                break

        prices = [_safe(r['close_price']) for r in recent[:dd_lookback]
                  if r['close_price']]
        high_20d = max(prices) if prices else close
        dd_pct = (high_20d - close) / high_20d if high_20d > 0 else 0

        pipeline = {
            'above_threshold': score > bot_th,
            'threshold': round(bot_th, 4),
            'streak_days': streak,
            'confirm_days': confirm_days,
            'duration_ok': streak >= confirm_days,
            'dd_pct': round(dd_pct * 100, 2),
            'dd_threshold': round(dd_threshold * 100, 2),
            'dd_ok': dd_pct >= dd_threshold,
        }

        filters = []
        if er is not None:
            filters.append({'label': 'ER (efficiency ratio)',
                            'value': round(er, 4), 'desc': '낮을수록 반전 가능성 ↑'})
        if atr_pct is not None:
            filters.append({'label': 'ATR%',
                            'value': round(atr_pct, 2), 'desc': '높을수록 변동성 충분'})

        result = {
            'ticker': ticker,
            'data': d,
            'score': round(score, 4),
            's_force': round(s_force, 4),
            's_div': round(s_div, 4),
            'and_geo_active': s_force > 0 and s_div > 0,
            'pipeline': pipeline,
            'filters': filters,
        }

    out = DIST_DATA / 'indicators' / f'{ticker}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)


def main():
    if not DB_PATH.exists():
        print('  No DB found — initializing')
        init_db()

    conn = get_connection()
    wl = load_watchlist()

    print('Export static data → webapp/dist/data/')
    tickers = export_watchlist(conn, wl)

    wl_params = wl.get('params', {})
    for t in tickers:
        export_chart_data(conn, t)
        export_indicators(conn, t, wl_params)

    # 스파크라인용 (20일 데이터)
    spark_dir = DIST_DATA / 'spark'
    spark_dir.mkdir(parents=True, exist_ok=True)
    for t in tickers:
        rows = conn.execute(
            """SELECT close_price FROM daily_scores
               WHERE ticker = ? AND date >= date('now', '-20 days')
               ORDER BY date ASC""",
            (t,),
        ).fetchall()
        prices = [_safe(r['close_price']) for r in rows if r['close_price']]
        with open(spark_dir / f'{t}.json', 'w') as f:
            json.dump(prices, f)

    conn.close()
    print(f'  chart/indicators/spark: {len(tickers)} tickers each')
    print('  Done!')


if __name__ == '__main__':
    main()
