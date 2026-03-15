<<<<<<< HEAD
"""일일 스캔 오케스트레이션 (V4 Duration 기반 알고리즘)"""
=======
"""일일 스캔 오케스트레이션 (C25 알고리즘)"""
>>>>>>> 187a32a6aa96e6dada11f8fbf85eaa48a75ec451
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from v4wp_realtime.config.settings import load_watchlist
from v4wp_realtime.core.indicators import fetch_data, analyze_ticker, classify_signal, get_latest_score_data
from v4wp_realtime.core.signal_tracker import is_new_signal, extract_recent_events
from v4wp_realtime.data.store import (
    get_connection, init_db, upsert_daily_scores,
    insert_signal_event, log_scan_run,
)


def run_scan(alert_fn=None, commentary_fn=None, dry_run=False):
<<<<<<< HEAD
    """전체 워치리스트 스캔 (V4 Duration 기반 알고리즘).

    파이프라인:
      1. smooth_earnings_volume: 실적발표일 ±1일 비정상 거래량 보정
      2. calc_v4_score: DivGate_3d 적용된 V4 스코어 산출
      3. detect_signal_events: threshold 기반 신호 감지
      4. build_price_filter: ER + ATR 필터 (비정상 구간 제외)
      5. BUY_DD_GATE: 20일 고점 대비 5% 이상 하락일 때만 매수 허용
      6. LATE_SELL_BLOCK: 20일 고점 대비 5% 이상 하락 시 매도 차단
      7. classify_signal (Duration 기반):
         - 매수: 3일 이상 지속 확인(CONFIRMED) → 100% 풀매수
         - 매도: 3일 이상 지속 확인(SELL_CONFIRMED) → 5% 매도
         - 미확인(PENDING) → 무시
=======
    """전체 워치리스트 스캔 (C25 알고리즘).

    C25 변경사항:
      - ATR quantile: 55 (relaxed)
      - LATE_SELL_BLOCK: 20일 고점 대비 5% 이상 하락 시 매도 차단
      - 강한 매수 임계값: |peak_val| >= 0.25
      - 매수 비율: 일반 40% / 강한 60%
      - 워치리스트에서 PGY, BA, INTC 제거됨
>>>>>>> 187a32a6aa96e6dada11f8fbf85eaa48a75ec451

    Args:
        alert_fn: callable(signal_dict) -> bool, 알림 전송 함수
        commentary_fn: callable(signal_dict, context) -> str, AI 코멘터리 생성
        dry_run: True면 DB/알림 없이 결과만 반환

    Returns:
        dict with scan results
    """
    wl = load_watchlist()
    tickers = wl['tickers']
    params = wl['params']
    today = datetime.now().strftime('%Y-%m-%d')

    conn = None
    if not dry_run:
        init_db()
        conn = get_connection()

    start_time = datetime.now()
    results = {
        'date': today,
        'scanned': 0,
        'errors': [],
        'new_signals': [],
        'blocked_sells': [],
<<<<<<< HEAD
        'blocked_buys': [],
=======
>>>>>>> 187a32a6aa96e6dada11f8fbf85eaa48a75ec451
        'scores': [],
    }

    all_tickers = list(tickers.keys()) + wl.get('benchmarks', [])

    for ticker in all_tickers:
        try:
            df = fetch_data(ticker, years=params.get('data_years', 3))
            if df is None or len(df) < 200:
                results['errors'].append((ticker, 'insufficient data'))
                continue

            analysis = analyze_ticker(ticker, df, params)
            results['scanned'] += 1

            # 일별 스코어 저장
            score_rows = get_latest_score_data(df, analysis['subindicators'], n_days=5)
            for row in score_rows:
                row['ticker'] = ticker
                row['er'] = None
                row['atr_pct'] = None
            results['scores'].extend(score_rows)

            if not dry_run and conn and score_rows:
                upsert_daily_scores(conn, score_rows)

            # 차단된 매도 신호 기록
            for ev in analysis.get('blocked_sells', []):
                peak_date = df.index[ev['peak_idx']].strftime('%Y-%m-%d')
                results['blocked_sells'].append({
                    'ticker': ticker,
                    'peak_date': peak_date,
                    'peak_val': ev['peak_val'],
                    'close_price': float(df['Close'].iloc[ev['peak_idx']]),
                })

<<<<<<< HEAD
            # 차단된 매수 신호 기록 (DD 게이트 미통과)
            for ev in analysis.get('blocked_buys', []):
                peak_date = df.index[ev['peak_idx']].strftime('%Y-%m-%d')
                results['blocked_buys'].append({
                    'ticker': ticker,
                    'peak_date': peak_date,
                    'peak_val': ev['peak_val'],
                    'close_price': float(df['Close'].iloc[ev['peak_idx']]),
                })

=======
>>>>>>> 187a32a6aa96e6dada11f8fbf85eaa48a75ec451
            # 최근 신호 추출 (필터 적용된 것만)
            recent = extract_recent_events(analysis['filtered_events'], df, lookback_days=10)
            sector = tickers.get(ticker, {}).get('sector', 'Benchmark')

            for ev in recent:
                peak_idx = ev['peak_idx']
                subind = analysis['subindicators']

<<<<<<< HEAD
                # Duration 기반 신호 분류
                classification = classify_signal(ev, params)

                # PENDING 매수 신호는 건너뛰기 (3일 미확인)
                if classification['tier'] == 'PENDING':
                    continue

=======
                # C25 신호 분류
                classification = classify_signal(ev, params)

>>>>>>> 187a32a6aa96e6dada11f8fbf85eaa48a75ec451
                signal_data = {
                    'ticker': ticker,
                    'sector': sector,
                    'signal_type': ev['type'],
                    'peak_date': ev['peak_date'],
                    'peak_val': float(ev['peak_val']),
<<<<<<< HEAD
                    'start_val': float(ev.get('start_val', 0)),
                    'duration': ev.get('duration', ev['end_idx'] - ev['start_idx'] + 1),
=======
>>>>>>> 187a32a6aa96e6dada11f8fbf85eaa48a75ec451
                    'close_price': float(df['Close'].iloc[peak_idx]),
                    'detected_date': today,
                    'notified': 0,
                    'commentary': None,
                    's_force': float(subind['s_force'].iloc[peak_idx]),
                    's_div': float(subind['s_div'].iloc[peak_idx]),
                    's_conc': float(subind['s_conc'].iloc[peak_idx]),
                    'er': None,
                    'atr_pct': None,
<<<<<<< HEAD
                    # Duration 기반 분류
                    'signal_tier': classification['tier'],
=======
                    # C25 추가 필드
>>>>>>> 187a32a6aa96e6dada11f8fbf85eaa48a75ec451
                    'is_strong': classification['is_strong'],
                    'signal_label': classification['label'],
                    'action_pct': classification['action_pct'],
                }

                # 신규 판별
                if dry_run or (conn and is_new_signal(conn, ticker, ev['type'], ev['peak_date'])):
                    # AI 코멘터리
                    if commentary_fn:
                        try:
                            context = {
                                'score_history': score_rows,
                                'recent_events': recent,
                            }
                            signal_data['commentary'] = commentary_fn(signal_data, context)
                        except Exception:
                            pass

                    # DB 저장
                    if not dry_run and conn:
                        inserted = insert_signal_event(conn, signal_data)
                        if not inserted:
                            continue

                    results['new_signals'].append(signal_data)

                    # 알림 전송
                    if alert_fn and not dry_run:
                        try:
                            alert_fn(signal_data)
                            if conn:
                                # mark notified
                                conn.execute(
                                    """UPDATE signal_events SET notified = 1
                                       WHERE ticker = ? AND signal_type = ? AND peak_date = ?""",
                                    (ticker, ev['type'], ev['peak_date'])
                                )
                                conn.commit()
                        except Exception as e:
                            results['errors'].append((ticker, f'alert error: {e}'))

        except Exception as e:
            results['errors'].append((ticker, str(e)))

    duration = (datetime.now() - start_time).total_seconds()
    results['duration_sec'] = duration

    if not dry_run and conn:
        log_scan_run(
            conn, today, results['scanned'],
            len(results['new_signals']), len(results['new_signals']),
            duration, 'success'
        )
        conn.close()

    return results
