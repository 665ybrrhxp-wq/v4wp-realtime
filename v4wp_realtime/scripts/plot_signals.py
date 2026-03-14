"""
V4_wP C25 Signal Visualization — Interactive Plotly Charts
==========================================================
SPY & QQQ: Price + V4 Score + Buy/Sell Signals
C25 = C14 base + 강매수TH=0.25 + 워치정리 + 매수40%/60%

Signal pipeline:
  1. download_data() -> OHLCV
  2. calc_v4_score() -> V4 score series
  3. detect_signal_events() -> raw events
  4. build_price_filter() -> ER+ATR(q=55) filter
  5. LATE_SELL_BLOCK: block sells where price < 20d_high * 0.95

Strong thresholds (C25):
  - Strong Buy:  abs(peak_val) >= 0.25  (changed from 0.15)
  - Strong Sell:  peak_val <= -0.25
  - Normal Buy: 40% of cash / Strong Buy: 60% of cash
"""

import sys
import os
from pathlib import Path

# Project root so we can import from real_market_backtest
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from real_market_backtest import (
    download_data,
    calc_v4_score,
    detect_signal_events,
    build_price_filter,
)

# ── Configuration ──────────────────────────────────────────────
TICKERS = ['SPY', 'QQQ']
START = '2007-01-01'
END = '2026-03-31'
CACHE_DIR = str(Path(_project_root) / 'cache')

V4_WINDOW = 20
SIGNAL_THRESHOLD = 0.15
COOLDOWN = 5
ER_QUANTILE = 66
ATR_QUANTILE = 55          # C14: relaxed from 66
LOOKBACK = 252

STRONG_BUY_TH = 0.25       # C25: abs(peak_val) >= 0.25 (was 0.15)
STRONG_SELL_TH = -0.25      # peak_val <= -0.25
BUY_NORMAL_PCT = 0.40       # C25: 40% of cash (was 30%)
BUY_STRONG_PCT = 0.60       # C25: 60% of cash (was 50%)

LATE_SELL_DROP_TH = 0.05    # C14: 5% below 20-day rolling high -> block sell

OUTPUT_DIR = str(Path(_project_root) / 'output')

# V4 score panel threshold lines
BUY_LINE_TH = SIGNAL_THRESHOLD * 0.5   # 0.075
SELL_LINE_TH = -SIGNAL_THRESHOLD        # -0.15


def process_ticker(ticker):
    """Download data, compute V4 score, detect & filter signals, apply LATE_SELL_BLOCK."""
    print(f"\n{'='*60}")
    print(f"  Processing {ticker} (C25)")
    print(f"{'='*60}")

    # 1. Data
    df = download_data(ticker, start=START, end=END, cache_dir=CACHE_DIR)

    # 2. V4 score
    print(f"  Computing V4 score (window={V4_WINDOW})...")
    v4 = calc_v4_score(df, w=V4_WINDOW)

    # 3. Detect signal events
    print(f"  Detecting signal events (th={SIGNAL_THRESHOLD}, cooldown={COOLDOWN})...")
    events = detect_signal_events(v4, th=SIGNAL_THRESHOLD, cooldown=COOLDOWN)
    print(f"  Raw events: {len(events)}")

    # 4. Price filter (C14: ATR q=55)
    print(f"  Building price filter (ER_q={ER_QUANTILE}, ATR_q={ATR_QUANTILE}, lookback={LOOKBACK})...")
    pf = build_price_filter(df, er_q=ER_QUANTILE, atr_q=ATR_QUANTILE, lookback=LOOKBACK)
    filtered = [e for e in events if pf(e['peak_idx'])]
    print(f"  After price filter: {len(filtered)} (removed {len(events)-len(filtered)})")

    # 5. LATE_SELL_BLOCK: compute 20-day rolling high
    rolling_high_20 = df['Close'].rolling(20, min_periods=1).max()

    active_events = []
    blocked_events = []

    for e in filtered:
        if e['type'] == 'top':
            idx = e['peak_idx']
            price = df['Close'].iloc[idx]
            rh = rolling_high_20.iloc[idx]
            if price < rh * (1 - LATE_SELL_DROP_TH):
                blocked_events.append(e)
            else:
                active_events.append(e)
        else:
            active_events.append(e)

    print(f"  After LATE_SELL_BLOCK: {len(active_events)} active, {len(blocked_events)} blocked sells")

    return df, v4, active_events, blocked_events


def classify_event(ev):
    """
    Classify an event into one of four visual categories.
    Returns (category, is_strong)
    """
    if ev['type'] == 'bottom':
        is_strong = abs(ev['peak_val']) >= STRONG_BUY_TH
        return ('strong_buy' if is_strong else 'normal_buy'), is_strong
    else:
        is_strong = ev['peak_val'] <= STRONG_SELL_TH
        return ('strong_sell' if is_strong else 'normal_sell'), is_strong


def build_chart(ticker, df, v4, events, blocked_events):
    """Build an interactive Plotly figure with price, V4 score, signal markers, and blocked sells."""
    print(f"  Building chart for {ticker}...")

    dates = df.index
    close = df['Close'].values
    v4_vals = v4.values

    # ── Separate events by category ─────────────────────────────
    cats = {
        'normal_buy':  {'dates': [], 'prices': [], 'texts': [], 'vals': []},
        'strong_buy':  {'dates': [], 'prices': [], 'texts': [], 'vals': []},
        'normal_sell': {'dates': [], 'prices': [], 'texts': [], 'vals': []},
        'strong_sell': {'dates': [], 'prices': [], 'texts': [], 'vals': []},
    }

    for ev in events:
        cat, _ = classify_event(ev)
        idx = ev['peak_idx']
        d = dates[idx]
        p = close[idx]
        v = ev['peak_val']
        cats[cat]['dates'].append(d)
        cats[cat]['prices'].append(p)
        cats[cat]['vals'].append(v)
        cats[cat]['texts'].append(
            f"Date: {d.strftime('%Y-%m-%d')}<br>"
            f"Price: ${p:.2f}<br>"
            f"V4 Score: {v:.3f}<br>"
            f"Type: {cat.replace('_', ' ').title()}"
        )

    # Blocked sell events
    blocked_data = {'dates': [], 'prices': [], 'texts': [], 'vals': []}
    for ev in blocked_events:
        idx = ev['peak_idx']
        d = dates[idx]
        p = close[idx]
        v = ev['peak_val']
        blocked_data['dates'].append(d)
        blocked_data['prices'].append(p)
        blocked_data['vals'].append(v)
        blocked_data['texts'].append(
            f"Date: {d.strftime('%Y-%m-%d')}<br>"
            f"Price: ${p:.2f}<br>"
            f"V4 Score: {v:.3f}<br>"
            f"Type: BLOCKED SELL (Late Sell Block)"
        )

    # ── Statistics ───────────────────────────────────────────────
    n_buy   = len(cats['normal_buy']['dates'])
    n_sbuy  = len(cats['strong_buy']['dates'])
    n_sell  = len(cats['normal_sell']['dates'])
    n_ssell = len(cats['strong_sell']['dates'])
    n_blocked = len(blocked_data['dates'])
    total   = n_buy + n_sbuy + n_sell + n_ssell + n_blocked

    sbuy_pct  = (n_sbuy  / total * 100) if total > 0 else 0
    ssell_pct = (n_ssell / total * 100) if total > 0 else 0
    blocked_pct = (n_blocked / total * 100) if total > 0 else 0

    # ── Create subplots: 2 rows, shared x-axis ──────────────────
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.65, 0.35],
        subplot_titles=(None, None),
    )

    # ── Row 1: Price line ────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=dates, y=close,
            mode='lines',
            name=f'{ticker} Close',
            line=dict(color='#42A5F5', width=1.5),
            hovertemplate='%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>',
        ),
        row=1, col=1,
    )

    # ── Row 1: Signal markers ────────────────────────────────────
    # Normal Buy (40%)
    if cats['normal_buy']['dates']:
        fig.add_trace(
            go.Scatter(
                x=cats['normal_buy']['dates'],
                y=cats['normal_buy']['prices'],
                mode='markers',
                name=f'Buy ({BUY_NORMAL_PCT*100:.0f}%)',
                marker=dict(
                    symbol='triangle-up',
                    size=10,
                    color='rgba(0, 230, 118, 0.65)',
                    line=dict(width=1, color='#00C853'),
                ),
                text=cats['normal_buy']['texts'],
                hoverinfo='text',
            ),
            row=1, col=1,
        )

    # Strong Buy
    if cats['strong_buy']['dates']:
        fig.add_trace(
            go.Scatter(
                x=cats['strong_buy']['dates'],
                y=cats['strong_buy']['prices'],
                mode='markers',
                name=f'STRONG Buy ({BUY_STRONG_PCT*100:.0f}%)',
                marker=dict(
                    symbol='triangle-up',
                    size=16,
                    color='#00E676',
                    line=dict(width=2, color='#00C853'),
                ),
                hovertext=cats['strong_buy']['texts'],
                hoverinfo='text',
            ),
            row=1, col=1,
        )
        for i, (d, p) in enumerate(zip(cats['strong_buy']['dates'],
                                        cats['strong_buy']['prices'])):
            fig.add_annotation(
                x=d, y=p,
                text="<b>BUY</b>",
                showarrow=True,
                arrowhead=0,
                arrowcolor='rgba(0,0,0,0)',
                ax=0, ay=-25,
                font=dict(size=10, color='#00E676', family='Arial Black'),
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='#00E676',
                borderwidth=1,
                borderpad=2,
                xref='x', yref='y',
            )

    # Normal Sell
    if cats['normal_sell']['dates']:
        fig.add_trace(
            go.Scatter(
                x=cats['normal_sell']['dates'],
                y=cats['normal_sell']['prices'],
                mode='markers',
                name='Sell',
                marker=dict(
                    symbol='triangle-down',
                    size=8,
                    color='rgba(255, 82, 82, 0.85)',
                    line=dict(width=1, color='#D50000'),
                ),
                text=cats['normal_sell']['texts'],
                hoverinfo='text',
            ),
            row=1, col=1,
        )

    # Strong Sell
    if cats['strong_sell']['dates']:
        fig.add_trace(
            go.Scatter(
                x=cats['strong_sell']['dates'],
                y=cats['strong_sell']['prices'],
                mode='markers',
                name='STRONG Sell',
                marker=dict(
                    symbol='triangle-down',
                    size=16,
                    color='#FF1744',
                    line=dict(width=2, color='#D50000'),
                ),
                hovertext=cats['strong_sell']['texts'],
                hoverinfo='text',
            ),
            row=1, col=1,
        )
        for i, (d, p) in enumerate(zip(cats['strong_sell']['dates'],
                                        cats['strong_sell']['prices'])):
            fig.add_annotation(
                x=d, y=p,
                text="<b>SELL</b>",
                showarrow=True,
                arrowhead=0,
                arrowcolor='rgba(0,0,0,0)',
                ax=0, ay=25,
                font=dict(size=10, color='#FF1744', family='Arial Black'),
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='#FF1744',
                borderwidth=1,
                borderpad=2,
                xref='x', yref='y',
            )

    # BLOCKED Sell (gray X markers)
    if blocked_data['dates']:
        fig.add_trace(
            go.Scatter(
                x=blocked_data['dates'],
                y=blocked_data['prices'],
                mode='markers',
                name='BLOCKED Sell',
                marker=dict(
                    symbol='x',
                    size=10,
                    color='rgba(158, 158, 158, 0.85)',
                    line=dict(width=2, color='#9E9E9E'),
                ),
                hovertext=blocked_data['texts'],
                hoverinfo='text',
            ),
            row=1, col=1,
        )
        for i, (d, p) in enumerate(zip(blocked_data['dates'],
                                        blocked_data['prices'])):
            fig.add_annotation(
                x=d, y=p,
                text="<b>BLOCKED</b>",
                showarrow=True,
                arrowhead=0,
                arrowcolor='rgba(0,0,0,0)',
                ax=0, ay=25,
                font=dict(size=9, color='#9E9E9E', family='Arial Black'),
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='#9E9E9E',
                borderwidth=1,
                borderpad=2,
                xref='x', yref='y',
            )

    # ── Row 2: V4 Score (filled areas) ──────────────────────────
    # Positive fill (green)
    v4_pos = np.where(v4_vals > 0, v4_vals, 0)
    fig.add_trace(
        go.Scatter(
            x=dates, y=v4_pos,
            mode='lines',
            name='V4 (+)',
            fill='tozeroy',
            fillcolor='rgba(0, 230, 118, 0.20)',
            line=dict(color='rgba(0, 230, 118, 0.5)', width=0.5),
            hoverinfo='skip',
            showlegend=False,
        ),
        row=2, col=1,
    )

    # Negative fill (red)
    v4_neg = np.where(v4_vals < 0, v4_vals, 0)
    fig.add_trace(
        go.Scatter(
            x=dates, y=v4_neg,
            mode='lines',
            name='V4 (-)',
            fill='tozeroy',
            fillcolor='rgba(255, 82, 82, 0.20)',
            line=dict(color='rgba(255, 82, 82, 0.5)', width=0.5),
            hoverinfo='skip',
            showlegend=False,
        ),
        row=2, col=1,
    )

    # V4 score line on top
    fig.add_trace(
        go.Scatter(
            x=dates, y=v4_vals,
            mode='lines',
            name='V4 Score',
            line=dict(color='#CE93D8', width=1.2),
            hovertemplate='%{x|%Y-%m-%d}<br>V4: %{y:.3f}<extra></extra>',
        ),
        row=2, col=1,
    )

    # ── Threshold dashed lines on V4 subplot ─────────────────────
    fig.add_hline(
        y=BUY_LINE_TH, line_dash='dash', line_color='rgba(0,230,118,0.4)',
        line_width=1,
        annotation_text=f'Buy th={BUY_LINE_TH:.3f}',
        annotation_position='top left',
        annotation_font_size=10,
        annotation_font_color='rgba(0,230,118,0.6)',
        row=2, col=1,
    )
    fig.add_hline(
        y=STRONG_BUY_TH, line_dash='dash', line_color='rgba(0,200,83,0.8)',
        line_width=1.5,
        annotation_text=f'Strong Buy={STRONG_BUY_TH:.2f} (60%)',
        annotation_position='top right',
        annotation_font_size=10,
        annotation_font_color='#00C853',
        row=2, col=1,
    )
    fig.add_hline(
        y=SELL_LINE_TH, line_dash='dash', line_color='rgba(255,82,82,0.6)',
        line_width=1,
        annotation_text=f'Sell th={SELL_LINE_TH:.3f}',
        annotation_position='bottom left',
        annotation_font_size=10,
        annotation_font_color='#FF5252',
        row=2, col=1,
    )
    fig.add_hline(
        y=STRONG_SELL_TH, line_dash='dash', line_color='rgba(213,0,0,0.7)',
        line_width=1.2,
        annotation_text=f'Strong Sell={STRONG_SELL_TH:.3f}',
        annotation_position='bottom left',
        annotation_font_size=10,
        annotation_font_color='#FF1744',
        row=2, col=1,
    )
    fig.add_hline(
        y=0, line_color='rgba(255,255,255,0.3)', line_width=0.8,
        row=2, col=1,
    )

    # ── Signal diamonds on V4 score subplot ──────────────────────
    v4_signal_groups = {
        'buy': {'dates': [], 'vals': [], 'texts': [], 'colors': [], 'sizes': [], 'symbols': []},
        'sell': {'dates': [], 'vals': [], 'texts': [], 'colors': [], 'sizes': [], 'symbols': []},
        'blocked': {'dates': [], 'vals': [], 'texts': [], 'colors': [], 'sizes': [], 'symbols': []},
    }
    for ev in events:
        cat, is_strong = classify_event(ev)
        idx = ev['peak_idx']
        d = dates[idx]
        v = ev['peak_val']
        side = 'buy' if 'buy' in cat else 'sell'
        v4_signal_groups[side]['dates'].append(d)
        v4_signal_groups[side]['vals'].append(v)
        v4_signal_groups[side]['colors'].append(
            '#00E676' if side == 'buy' else '#FF1744'
        )
        v4_signal_groups[side]['sizes'].append(10 if is_strong else 6)
        v4_signal_groups[side]['symbols'].append('diamond' if is_strong else 'diamond-open')
        v4_signal_groups[side]['texts'].append(
            f"{d.strftime('%Y-%m-%d')}<br>"
            f"V4={v:.3f}<br>"
            f"{cat.replace('_',' ').title()}"
        )

    # Blocked sells on V4 subplot
    for ev in blocked_events:
        idx = ev['peak_idx']
        d = dates[idx]
        v = ev['peak_val']
        v4_signal_groups['blocked']['dates'].append(d)
        v4_signal_groups['blocked']['vals'].append(v)
        v4_signal_groups['blocked']['colors'].append('#9E9E9E')
        v4_signal_groups['blocked']['sizes'].append(8)
        v4_signal_groups['blocked']['symbols'].append('x')
        v4_signal_groups['blocked']['texts'].append(
            f"{d.strftime('%Y-%m-%d')}<br>"
            f"V4={v:.3f}<br>"
            f"BLOCKED SELL"
        )

    for side, grp in v4_signal_groups.items():
        if grp['dates']:
            fig.add_trace(
                go.Scatter(
                    x=grp['dates'],
                    y=grp['vals'],
                    mode='markers',
                    marker=dict(
                        symbol=grp['symbols'],
                        size=grp['sizes'],
                        color=grp['colors'],
                        line=dict(width=1, color='white'),
                    ),
                    showlegend=False,
                    hovertemplate='%{text}<extra></extra>',
                    text=grp['texts'],
                ),
                row=2, col=1,
            )

    # ── Stats annotation box (top-right corner) ──────────────────
    stats_text = (
        f"<b>C25 Signal Stats</b> (강매수TH={STRONG_BUY_TH})<br>"
        f"Total signals: {total}<br>"
        f"<span style='color:#00E676'>Strong Buy (|pv|≥{STRONG_BUY_TH}): {n_sbuy} ({sbuy_pct:.1f}%)</span><br>"
        f"Normal Buy: {n_buy}<br>"
        f"<span style='color:#FF1744'>Strong Sell: {n_ssell} ({ssell_pct:.1f}%)</span><br>"
        f"Normal Sell: {n_sell}<br>"
        f"<span style='color:#9E9E9E'>Blocked Sell: {n_blocked} ({blocked_pct:.1f}%)</span><br>"
        f"Buy ratio: Normal {BUY_NORMAL_PCT*100:.0f}% / Strong {BUY_STRONG_PCT*100:.0f}%"
    )

    fig.add_annotation(
        text=stats_text,
        xref='paper', yref='paper',
        x=0.99, y=0.98,
        xanchor='right', yanchor='top',
        showarrow=False,
        font=dict(size=12, color='#E0E0E0', family='Consolas, monospace'),
        align='left',
        bordercolor='#555',
        borderwidth=1,
        borderpad=8,
        bgcolor='rgba(30, 30, 50, 0.85)',
    )

    # ── Range selector buttons ───────────────────────────────────
    range_buttons = [
        dict(count=1,  label='1M',  step='month', stepmode='backward'),
        dict(count=3,  label='3M',  step='month', stepmode='backward'),
        dict(count=6,  label='6M',  step='month', stepmode='backward'),
        dict(count=1,  label='1Y',  step='year',  stepmode='backward'),
        dict(count=3,  label='3Y',  step='year',  stepmode='backward'),
        dict(count=5,  label='5Y',  step='year',  stepmode='backward'),
        dict(label='ALL', step='all'),
    ]

    # ── Layout ───────────────────────────────────────────────────
    title_kr = f"{ticker} C25 매매신호 차트 (강매수TH=0.25 + 매수40%/60% + Late Sell Block)"
    period_str = f"{dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}"

    fig.update_layout(
        title=dict(
            text=f'<b>{title_kr}</b><br><span style="font-size:13px;color:#999">{period_str}</span>',
            font=dict(size=22, family='Arial, sans-serif', color='#E0E0E0'),
            x=0.5,
            xanchor='center',
        ),
        template='plotly_dark',
        height=900,
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='rgba(30,30,50,0.9)',
            font_size=12,
            font_family='Consolas, monospace',
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.04,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(30,30,50,0.7)',
            bordercolor='#555',
            borderwidth=1,
            font=dict(size=11, color='#E0E0E0'),
        ),
        margin=dict(l=60, r=30, t=110, b=60),
        dragmode='zoom',
    )

    # ── X-axis: range slider on row 2, range selector on row 1 ──
    fig.update_xaxes(
        rangeslider=dict(visible=False),
        rangeselector=dict(
            buttons=range_buttons,
            bgcolor='rgba(40,40,60,0.8)',
            activecolor='#7B1FA2',
            bordercolor='#555',
            borderwidth=1,
            font=dict(size=11, color='#E0E0E0'),
            x=0, y=1.08,
        ),
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikethickness=1,
        spikecolor='#888',
        spikedash='dot',
        row=1, col=1,
    )

    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.05, bgcolor='rgba(40,40,60,0.4)'),
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikethickness=1,
        spikecolor='#888',
        spikedash='dot',
        row=2, col=1,
    )

    # ── Y-axes ───────────────────────────────────────────────────
    fig.update_yaxes(
        title_text='Price ($)',
        title_font=dict(size=13, color='#AAA'),
        gridcolor='rgba(255,255,255,0.08)',
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikethickness=1,
        spikecolor='#888',
        spikedash='dot',
        row=1, col=1,
    )
    fig.update_yaxes(
        title_text='V4 Score',
        title_font=dict(size=13, color='#AAA'),
        gridcolor='rgba(255,255,255,0.08)',
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikethickness=1,
        spikecolor='#888',
        spikedash='dot',
        row=2, col=1,
    )

    return fig


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for ticker in TICKERS:
        df, v4, events, blocked_events = process_ticker(ticker)
        fig = build_chart(ticker, df, v4, events, blocked_events)

        out_path = os.path.join(OUTPUT_DIR, f'{ticker}_C25_signals.html')
        fig.write_html(
            out_path,
            include_plotlyjs='cdn',
            full_html=True,
            config={
                'scrollZoom': True,
                'displayModeBar': True,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
                'displaylogo': False,
                'responsive': True,
            },
            default_width='100%',
            default_height='900px',
        )
        print(f"\n  >> Saved: {out_path}")

    print(f"\n{'='*60}")
    print(f"  All C25 charts generated in {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
