"""V4_wP Streamlit Dashboard"""
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from datetime import datetime, timedelta

from v4wp_realtime.config.settings import DB_PATH, load_watchlist

st.set_page_config(
    page_title='V4_wP Dashboard',
    page_icon='\U0001f4ca',
    layout='wide',
)


@st.cache_resource
def get_db():
    """SQLite 연결 (read-only)"""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def load_scores(conn, days=30):
    """최근 N일 스코어"""
    df = pd.read_sql_query(
        f"SELECT * FROM daily_scores WHERE date >= date('now', '-{days} days') ORDER BY date, ticker",
        conn
    )
    return df


def load_signals(conn, days=90):
    """최근 N일 신호"""
    df = pd.read_sql_query(
        f"SELECT * FROM signal_events WHERE peak_date >= date('now', '-{days} days') ORDER BY peak_date DESC",
        conn
    )
    return df


def load_scan_runs(conn, limit=30):
    """스캔 로그"""
    df = pd.read_sql_query(
        f"SELECT * FROM scan_runs ORDER BY run_date DESC LIMIT {limit}",
        conn
    )
    return df


# ========== PAGES ==========

def page_watchlist():
    """Watchlist: 전 종목 현재 스코어"""
    st.header('Watchlist')

    conn = get_db()
    df = load_scores(conn, days=5)

    if df.empty:
        st.warning('No score data. Run backfill first: `python -m v4wp_realtime.scripts.backfill`')
        return

    # 최신 날짜 스코어
    latest_date = df['date'].max()
    latest = df[df['date'] == latest_date].copy()
    st.caption(f'Latest data: {latest_date}')

    wl = load_watchlist()

    # 섹터 추가
    latest['sector'] = latest['ticker'].map(lambda t: wl['tickers'].get(t, {}).get('sector', 'Benchmark'))

    # 신호 상태 컬럼
    def score_status(row):
        s = row['score']
        if s is None or pd.isna(s):
            return '-'
        if s > 0.075:
            return '\U0001f7e2 BUY'
        elif s < -0.15:
            return '\U0001f534 SELL'
        return '\u26aa Neutral'

    latest['status'] = latest.apply(score_status, axis=1)

    # 표시
    cols = ['ticker', 'sector', 'score', 's_force', 's_div', 's_conc', 'close_price', 'status']
    display = latest[cols].sort_values('score', ascending=False).reset_index(drop=True)

    # 컬럼 포맷
    st.dataframe(
        display,
        column_config={
            'ticker': st.column_config.TextColumn('Ticker', width='small'),
            'sector': st.column_config.TextColumn('Sector', width='small'),
            'score': st.column_config.NumberColumn('Score', format='%.3f'),
            's_force': st.column_config.NumberColumn('Force', format='%.2f'),
            's_div': st.column_config.NumberColumn('Div', format='%.2f'),
            's_conc': st.column_config.NumberColumn('Conc', format='%.2f'),
            'close_price': st.column_config.NumberColumn('Price', format='$%.2f'),
            'status': st.column_config.TextColumn('Status', width='small'),
        },
        hide_index=True,
        use_container_width=True,
    )


def page_signals():
    """Signals: 최근 신호 목록"""
    st.header('Recent Signals')

    conn = get_db()
    days = st.slider('Period (days)', 7, 180, 30)
    df = load_signals(conn, days=days)

    if df.empty:
        st.info(f'No signals in last {days} days.')
        return

    st.caption(f'{len(df)} signals found')

    # 타입별 필터
    sig_types = st.multiselect('Signal Type', ['bottom', 'top'], default=['bottom', 'top'])
    df = df[df['signal_type'].isin(sig_types)]

    # 표시
    display_cols = ['peak_date', 'ticker', 'signal_type', 'peak_val', 'close_price',
                    's_force', 's_div', 's_conc', 'commentary']
    st.dataframe(
        df[display_cols],
        column_config={
            'peak_date': st.column_config.TextColumn('Date'),
            'ticker': st.column_config.TextColumn('Ticker'),
            'signal_type': st.column_config.TextColumn('Type'),
            'peak_val': st.column_config.NumberColumn('Score', format='%.3f'),
            'close_price': st.column_config.NumberColumn('Price', format='$%.2f'),
            's_force': st.column_config.NumberColumn('Force', format='%.2f'),
            's_div': st.column_config.NumberColumn('Div', format='%.2f'),
            's_conc': st.column_config.NumberColumn('Conc', format='%.2f'),
            'commentary': st.column_config.TextColumn('AI Commentary', width='large'),
        },
        hide_index=True,
        use_container_width=True,
    )

    # 종목별 신호 수
    st.subheader('Signals by Ticker')
    counts = df.groupby(['ticker', 'signal_type']).size().reset_index(name='count')
    fig = px.bar(counts, x='ticker', y='count', color='signal_type',
                 barmode='group', color_discrete_map={'bottom': '#22c55e', 'top': '#ef4444'})
    fig.update_layout(height=350, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)


def page_heatmap():
    """Heatmap: 종목 x 날짜 스코어 히트맵"""
    st.header('Score Heatmap')

    conn = get_db()
    days = st.slider('Period (days)', 7, 60, 30, key='hm_days')
    df = load_scores(conn, days=days)

    if df.empty:
        st.warning('No data available.')
        return

    # 피벗
    pivot = df.pivot_table(index='ticker', columns='date', values='score', aggfunc='first')
    pivot = pivot.sort_index()

    fig = px.imshow(
        pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0,
        aspect='auto',
        labels={'color': 'V4 Score'},
    )
    fig.update_layout(height=max(400, len(pivot) * 18), margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)


def page_pnl():
    """P&L: 신호별 수익 추적"""
    st.header('P&L Tracker')

    conn = get_db()
    df = load_signals(conn, days=365)

    if df.empty:
        st.info('No signal data for P&L tracking.')
        return

    st.caption(f'Tracking {len(df)} signals (buy $100 per signal, mark-to-market)')

    # 간단한 P&L: bottom 신호마다 $100 매수, 현재가 기준 평가
    bottom_signals = df[df['signal_type'] == 'bottom'].copy()
    if bottom_signals.empty:
        st.info('No bottom (buy) signals.')
        return

    # 현재가 가져오기 (최신 스코어 테이블에서)
    latest_prices = pd.read_sql_query(
        "SELECT ticker, close_price FROM daily_scores WHERE date = (SELECT MAX(date) FROM daily_scores) ",
        conn
    )
    price_map = dict(zip(latest_prices['ticker'], latest_prices['close_price']))

    pnl_rows = []
    for _, row in bottom_signals.iterrows():
        ticker = row['ticker']
        buy_price = row['close_price']
        current_price = price_map.get(ticker)
        if buy_price and current_price and buy_price > 0:
            shares = 100.0 / buy_price
            value = shares * current_price
            pnl = value - 100
            pnl_rows.append({
                'date': row['peak_date'],
                'ticker': ticker,
                'buy_price': buy_price,
                'current_price': current_price,
                'pnl': pnl,
                'roi': pnl,  # % of $100
            })

    if not pnl_rows:
        st.info('No P&L data to display.')
        return

    pnl_df = pd.DataFrame(pnl_rows)

    col1, col2, col3, col4 = st.columns(4)
    total_invested = len(pnl_df) * 100
    total_pnl = pnl_df['pnl'].sum()
    col1.metric('Total Invested', f'${total_invested:,.0f}')
    col2.metric('Total P&L', f'${total_pnl:+,.0f}')
    col3.metric('ROI', f'{total_pnl / total_invested * 100:+.1f}%')
    col4.metric('Win Rate', f'{(pnl_df["pnl"] > 0).mean() * 100:.0f}%')

    # 종목별 P&L
    by_ticker = pnl_df.groupby('ticker')['pnl'].sum().sort_values(ascending=False).reset_index()
    fig = px.bar(by_ticker, x='ticker', y='pnl',
                 color='pnl', color_continuous_scale='RdYlGn',
                 color_continuous_midpoint=0)
    fig.update_layout(height=400, margin=dict(t=20), title='P&L by Ticker')
    st.plotly_chart(fig, use_container_width=True)

    # 상세 테이블
    st.subheader('Signal Details')
    st.dataframe(
        pnl_df,
        column_config={
            'date': st.column_config.TextColumn('Signal Date'),
            'ticker': st.column_config.TextColumn('Ticker'),
            'buy_price': st.column_config.NumberColumn('Buy Price', format='$%.2f'),
            'current_price': st.column_config.NumberColumn('Current', format='$%.2f'),
            'pnl': st.column_config.NumberColumn('P&L', format='$%.2f'),
            'roi': st.column_config.NumberColumn('ROI%', format='%.1f%%'),
        },
        hide_index=True,
        use_container_width=True,
    )


# ========== MAIN ==========

def main():
    st.sidebar.title('\U0001f4ca V4_wP Dashboard')

    page = st.sidebar.radio('Page', ['Watchlist', 'Signals', 'Heatmap', 'P&L'])

    # DB 상태
    if DB_PATH.exists():
        st.sidebar.success(f'DB: {DB_PATH.name}')
        conn = get_db()
        runs = load_scan_runs(conn, limit=1)
        if not runs.empty:
            st.sidebar.caption(f'Last scan: {runs.iloc[0]["run_date"]}')
    else:
        st.sidebar.error('DB not found. Run backfill first.')

    if page == 'Watchlist':
        page_watchlist()
    elif page == 'Signals':
        page_signals()
    elif page == 'Heatmap':
        page_heatmap()
    elif page == 'P&L':
        page_pnl()


if __name__ == '__main__':
    main()
