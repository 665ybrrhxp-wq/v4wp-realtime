"""일일 스캔 오케스트레이션 (V4 Duration 기반 알고리즘)"""
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
import pandas as pd
from v4wp_realtime.config.settings import load_watchlist, SECTOR_ETF_MAP
from v4wp_realtime.core.indicators import fetch_data, analyze_ticker, classify_signal, get_latest_score_data
from v4wp_realtime.core.signal_tracker import is_new_signal, extract_recent_events
from v4wp_realtime.core.regime import classify_regime, get_conviction
from v4wp_realtime.data.store import (
    get_connection, init_db, upsert_daily_scores,
    insert_signal_event, log_scan_run,
)


def run_scan(alert_fn=None, commentary_fn=None, interpretation_fn=None, dry_run=False):
    """전체 워치리스트 스캔 (VN60+GEO-OP 알고리즘, 매수 전용).

    파이프라인:
      1. smooth_earnings_volume: 실적발표일 ±1일 비정상 거래량 보정
      2. calc_v4_score: AND-GEO 방식 (S_Force×S_Div 기하평균) 스코어 산출
      3. detect_signal_events: threshold(0.05) 기반 신호 감지
      4. build_price_filter: ER<80% + ATR>40% 필터
      5. BUY_DD_GATE: 20일 고점 대비 3% 이상 하락일 때만 매수 허용
      6. classify_signal (Duration 기반):
         - 1일 이상 지속 확인(CONFIRMED) → 100% 풀매수
         - 미확인(PENDING) → 무시

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
    buy_dd_threshold = params.get('buy_dd_threshold', 0.03)

    results = {
        'date': today,
        'scanned': 0,
        'errors': [],
        'new_signals': [],
        'blocked_buys': [],
        'watch_alerts': [],
        'scores': [],
    }

    all_tickers = list(tickers.keys()) + wl.get('benchmarks', [])

    # ── 시장/섹터 ETF + VIX 20일 변화율 사전 계산 ──
    qqq_ret20 = None
    sector_ret20_map = {}
    vix_change_20d_series = None
    try:
        qqq_df = fetch_data('QQQ', years=params.get('data_years', 3))
        if qqq_df is not None and len(qqq_df) >= 40:
            qqq_ret20 = qqq_df['Close'] / qqq_df['Close'].shift(20) - 1.0
        for etf in set(v for v in SECTOR_ETF_MAP.values() if v):
            etf_df = fetch_data(etf, years=params.get('data_years', 3))
            if etf_df is not None and len(etf_df) >= 40:
                sector_ret20_map[etf] = etf_df['Close'] / etf_df['Close'].shift(20) - 1.0
    except Exception:
        pass
    try:
        import yfinance as yf
        vix_df = yf.download('^VIX', period=f'{params.get("data_years", 3)}y', progress=False)
        if vix_df is not None and len(vix_df) >= 40:
            vix_close = vix_df['Close']
            if hasattr(vix_close, 'columns'):
                vix_close = vix_close.iloc[:, 0]
            vix_change_20d_series = vix_close / vix_close.shift(20) - 1.0
    except Exception:
        pass

    # 앨범 전송용 배치 수집
    pending_alerts = []  # list of (signal_data, chart_bytes)

    for ticker in all_tickers:
        try:
            df = fetch_data(ticker, years=params.get('data_years', 3))
            if df is None or len(df) < 200:
                results['errors'].append((ticker, 'insufficient data'))
                continue

            analysis = analyze_ticker(ticker, df, params)
            results['scanned'] += 1

            # 일별 스코어 저장
            score_rows = get_latest_score_data(df, analysis['subindicators'], n_days=252)
            for row in score_rows:
                row['ticker'] = ticker
                row['er'] = None
                row['atr_pct'] = None
            results['scores'].extend(score_rows)

            if not dry_run and conn and score_rows:
                upsert_daily_scores(conn, score_rows)

            # 차단된 매수 신호 기록 (DD 게이트 미통과)
            recent_blocked = extract_recent_events(
                analysis.get('blocked_buys', []), df, lookback_days=10
            )
            for ev in recent_blocked:
                blocked = {
                    'ticker': ticker,
                    'sector': tickers.get(ticker, {}).get('sector', 'Benchmark'),
                    'peak_date': ev['peak_date'],
                    'peak_val': ev['peak_val'],
                    'close_price': float(df['Close'].iloc[ev['peak_idx']]),
                    'dd_pct': ev.get('dd_pct', 0),
                    'dd_threshold': buy_dd_threshold,
                    's_force': float(analysis['subindicators']['s_force'].iloc[ev['peak_idx']]),
                    's_div': float(analysis['subindicators']['s_div'].iloc[ev['peak_idx']]),
                }
                results['blocked_buys'].append(blocked)

                # Watch Alert: threshold의 80% 이상이면 근접 알림 대상
                if ev.get('dd_pct', 0) >= buy_dd_threshold * 0.8:
                    results['watch_alerts'].append(blocked)

            # 최근 신호 추출 (필터 적용된 것만)
            recent = extract_recent_events(analysis['filtered_events'], df, lookback_days=10)
            sector = tickers.get(ticker, {}).get('sector', 'Benchmark')

            for ev in recent:
                peak_idx = ev['peak_idx']
                subind = analysis['subindicators']

                # Duration 기반 신호 분류
                classification = classify_signal(ev, params)

                # PENDING 매수 신호는 건너뛰기 (3일 미확인)
                if classification['tier'] == 'PENDING':
                    continue

                signal_data = {
                    'ticker': ticker,
                    'sector': sector,
                    'signal_type': ev['type'],
                    'peak_date': ev['peak_date'],
                    'peak_val': float(ev['peak_val']),
                    'start_val': float(ev.get('start_val', 0)),
                    'duration': ev.get('duration', ev['end_idx'] - ev['start_idx'] + 1),
                    'close_price': float(df['Close'].iloc[peak_idx]),
                    'detected_date': today,
                    'notified': 0,
                    'commentary': None,
                    's_force': float(subind['s_force'].iloc[peak_idx]),
                    's_div': float(subind['s_div'].iloc[peak_idx]),
                    's_conc': 0.0,  # GEO-OP에서 미사용 (DB 호환)
                    'dd_pct': ev.get('dd_pct', 0.0),
                    'er': None,
                    'atr_pct': None,
                    # Duration 기반 분류
                    'signal_tier': classification['tier'],
                    'is_strong': classification['is_strong'],
                    'signal_label': classification['label'],
                    'action_pct': classification['action_pct'],
                    'market_return_20d': None,
                    'sector_return_20d': None,
                    'vix_change_20d': None,
                }

                # 시장/섹터 컨텍스트
                peak_ts = df.index[peak_idx]
                if qqq_ret20 is not None:
                    mask = qqq_ret20.index <= peak_ts
                    if mask.any():
                        val = qqq_ret20.loc[mask].iloc[-1]
                        if pd.notna(val):
                            signal_data['market_return_20d'] = round(float(val), 4)
                sector_etf = SECTOR_ETF_MAP.get(sector)
                if sector_etf is None and ticker in ('QQQ', 'VOO'):
                    ret20 = df['Close'] / df['Close'].shift(20) - 1.0
                    val = ret20.iloc[peak_idx]
                    if pd.notna(val):
                        signal_data['sector_return_20d'] = round(float(val), 4)
                elif sector_etf and sector_etf in sector_ret20_map:
                    s_series = sector_ret20_map[sector_etf]
                    mask = s_series.index <= peak_ts
                    if mask.any():
                        val = s_series.loc[mask].iloc[-1]
                        if pd.notna(val):
                            signal_data['sector_return_20d'] = round(float(val), 4)
                if vix_change_20d_series is not None:
                    mask = vix_change_20d_series.index <= peak_ts
                    if mask.any():
                        val = vix_change_20d_series.loc[mask].iloc[-1]
                        if pd.notna(val):
                            signal_data['vix_change_20d'] = round(float(val), 4)

                # 매크로 레짐 분류 (역발상 확신도)
                signal_data['market_regime'] = classify_regime(
                    signal_data.get('market_return_20d'),
                    signal_data.get('vix_change_20d'),
                )

                # 신규 판별
                if dry_run or (conn and is_new_signal(conn, ticker, ev['type'], ev['peak_date'])):
                    # AI 코멘터리 (Telegram용 한줄평)
                    context = {
                        'score_history': score_rows,
                        'recent_events': recent,
                    }

                    # 유사 시그널 검색 (AI 의장에게 추가 맥락 제공)
                    if not dry_run and conn:
                        try:
                            from v4wp_realtime.core.similarity import (
                                find_similar_signals, build_similar_signals_context,
                            )
                            similar = find_similar_signals(conn, signal_data)
                            if similar:
                                sim_text = build_similar_signals_context(similar)
                                if sim_text:
                                    context['similar_signals_text'] = sim_text
                        except Exception:
                            pass
                    # 페르소나 과거 실적 (AI 환류)
                    if not dry_run and conn:
                        try:
                            from v4wp_realtime.core.postmortem import get_postmortem_stats
                            pm = get_postmortem_stats(conn, ticker)
                            if pm.get('total_completed', 0) >= 5:
                                context['persona_accuracy'] = pm['persona_accuracy']
                                context['total_completed'] = pm['total_completed']
                        except Exception:
                            pass

                    if commentary_fn:
                        try:
                            signal_data['commentary'] = commentary_fn(signal_data, context)
                        except Exception:
                            pass

                    # AI 해석기 (Dashboard용 멀티 페르소나)
                    if interpretation_fn:
                        try:
                            import json as _json
                            interp = interpretation_fn(signal_data, context)
                            if interp:
                                signal_data['interpretation'] = _json.dumps(
                                    interp, ensure_ascii=False
                                )
                        except Exception:
                            pass
                    if 'interpretation' not in signal_data:
                        signal_data['interpretation'] = None

                    # DB 저장
                    if not dry_run and conn:
                        inserted = insert_signal_event(conn, signal_data)
                        if not inserted:
                            continue

                    results['new_signals'].append(signal_data)

                    # 차트 생성 + 알림 배치 수집
                    chart_bytes = None
                    if alert_fn and not dry_run:
                        try:
                            from v4wp_realtime.alerts.chart_generator import generate_signal_chart
                            chart_bytes = generate_signal_chart(
                                ticker, df, analysis['subindicators'], ev
                            )
                        except Exception:
                            pass  # 차트 실패 시 텍스트만 전송

                        pending_alerts.append((signal_data, chart_bytes))

        except Exception as e:
            results['errors'].append((ticker, str(e)))

    # ── 앨범 전송 (스캔 완료 후 일괄) ──
    if pending_alerts and alert_fn and not dry_run:
        try:
            from v4wp_realtime.alerts.telegram_bot import send_signal_album
            send_signal_album(pending_alerts)

            # mark all notified
            if conn:
                for signal_data, _ in pending_alerts:
                    conn.execute(
                        """UPDATE signal_events SET notified = 1
                           WHERE ticker = ? AND signal_type = ? AND peak_date = ?""",
                        (signal_data['ticker'], signal_data['signal_type'],
                         signal_data['peak_date'])
                    )
                conn.commit()
        except Exception as e:
            results['errors'].append(('album_send', f'alert error: {e}'))

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
