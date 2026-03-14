"""히스토리 초기 적재 — 과거 신호를 DB에 backfill"""
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from v4wp_realtime.config.settings import load_watchlist
from v4wp_realtime.core.indicators import fetch_data, analyze_ticker, get_latest_score_data
from v4wp_realtime.data.store import (
    get_connection, init_db, upsert_daily_scores, insert_signal_event,
)


def backfill(years=3, score_days=60):
    """과거 데이터 backfill.
    Args:
        years: 데이터 기간
        score_days: 최근 N일 스코어 저장
    """
    wl = load_watchlist()
    params = wl['params']
    params['data_years'] = years
    today = datetime.now().strftime('%Y-%m-%d')

    init_db()
    conn = get_connection()

    all_tickers = list(wl['tickers'].keys()) + wl.get('benchmarks', [])
    n_signals = 0
    n_scores = 0

    print(f'Backfilling {len(all_tickers)} tickers ({years}y data, {score_days}d scores)...')

    for ticker in all_tickers:
        try:
            df = fetch_data(ticker, years=years)
            if df is None or len(df) < 200:
                print(f'  {ticker}: skip (insufficient data)')
                continue

            analysis = analyze_ticker(ticker, df, params)
            sector = wl['tickers'].get(ticker, {}).get('sector', 'Benchmark')

            # 스코어 저장
            score_rows = get_latest_score_data(df, analysis['subindicators'], n_days=score_days)
            for row in score_rows:
                row['ticker'] = ticker
                row['er'] = None
                row['atr_pct'] = None
            upsert_daily_scores(conn, score_rows)
            n_scores += len(score_rows)

            # 필터링된 신호 저장
            for ev in analysis['filtered_events']:
                peak_date = df.index[ev['peak_idx']].strftime('%Y-%m-%d')
                peak_idx = ev['peak_idx']
                subind = analysis['subindicators']

                event = {
                    'ticker': ticker,
                    'signal_type': ev['type'],
                    'peak_date': peak_date,
                    'peak_val': float(ev['peak_val']),
                    'close_price': float(df['Close'].iloc[peak_idx]),
                    'detected_date': today,
                    'notified': 1,  # backfill은 이미 알림 처리됨
                    'commentary': None,
                    's_force': float(subind['s_force'].iloc[peak_idx]),
                    's_div': float(subind['s_div'].iloc[peak_idx]),
                    's_conc': float(subind['s_conc'].iloc[peak_idx]),
                    'er': None,
                    'atr_pct': None,
                }
                if insert_signal_event(conn, event):
                    n_signals += 1

            print(f'  {ticker}: {len(analysis["filtered_events"])} signals, {len(score_rows)} scores')

        except Exception as e:
            print(f'  {ticker}: ERROR - {e}')

    conn.close()
    print(f'\nBackfill complete: {n_signals} signals, {n_scores} score rows')


if __name__ == '__main__':
    backfill()
