"""
V4 Leverage Timeline: QQQ, VOO — 언제 2x를 샀고, 언제 1x를 샀는지 시각화
"""
import sys, os, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
from real_market_backtest import (
    calc_v4_score, detect_signal_events, build_price_filter,
    smooth_earnings_volume,
)

MONTHLY_DEPOSIT = 500.0
SIGNAL_BUY_PCT = 0.50
V4_W = 20; SIGNAL_TH = 0.15; COOLDOWN = 5
ER_Q = 66; ATR_Q = 55; LOOKBACK = 252
DIVGATE = 3; CONFIRM = 3


def download_max(ticker):
    t = yf.Ticker(ticker)
    df = t.history(period='max', auto_adjust=True)
    if df is None or len(df) == 0:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def get_buy_signal_indices(df, ticker):
    try:
        df_s = smooth_earnings_volume(df, ticker=ticker)
    except Exception:
        df_s = df.copy()
    score = calc_v4_score(df_s, w=V4_W, divgate_days=DIVGATE)
    events = detect_signal_events(score, th=SIGNAL_TH, cooldown=COOLDOWN)
    pf = build_price_filter(df_s, er_q=ER_Q, atr_q=ATR_Q, lookback=LOOKBACK)
    n = len(df_s)
    buys = set()
    for ev in events:
        if ev['type'] != 'bottom':
            continue
        if not pf(ev['peak_idx']):
            continue
        dur = ev['end_idx'] - ev['start_idx'] + 1
        ci = ev['start_idx'] + CONFIRM - 1
        if ci <= ev['end_idx'] and dur >= CONFIRM and ci < n:
            buys.add(ci)
    return buys


def get_month_end_indices(dates):
    """Get month-end trading day indices"""
    month_periods = pd.Series(dates).dt.to_period('M')
    month_map = {}
    for i, mp in enumerate(month_periods):
        key = str(mp)
        if key not in month_map:
            month_map[key] = {'last': i}
        else:
            month_map[key]['last'] = i
    return [v['last'] for v in month_map.values()]


def simulate_and_track(close, dates, buy_signals):
    """Track every buy event: V4 signal (2x) vs month-end DCA (1x)"""
    n = len(close)
    month_periods = pd.Series(dates).dt.to_period('M')
    month_map = {}
    for i, mp in enumerate(month_periods):
        key = str(mp)
        if key not in month_map:
            month_map[key] = {'first': i, 'last': i}
        else:
            month_map[key]['last'] = i
    sorted_months = sorted(month_map.keys())

    cash = 0.0
    buys_2x = []  # (date, price, amount)
    buys_1x = []  # (date, price, amount)

    for mk in sorted_months:
        mm = month_map[mk]
        fi, li = mm['first'], mm['last']

        cash += MONTHLY_DEPOSIT

        # V4 signals -> 2x leverage
        for day_idx in range(fi, li + 1):
            if day_idx in buy_signals and cash > 1.0:
                amt = cash * SIGNAL_BUY_PCT
                buys_2x.append({
                    'date': dates[day_idx],
                    'price': close[day_idx],
                    'amount': amt,
                    'idx': day_idx,
                })
                cash -= amt

        # Month-end -> 1x
        if cash > 1.0:
            buys_1x.append({
                'date': dates[li],
                'price': close[li],
                'amount': cash,
                'idx': li,
            })
            cash = 0.0

    return buys_2x, buys_1x


# ═══════════════════════════════════════════════════════════
# Generate charts
# ═══════════════════════════════════════════════════════════

for tk in ['QQQ', 'VOO']:
    print(f"  Processing {tk}...")
    df = download_max(tk)
    if df is None:
        continue

    close = df['Close'].values
    dates = df.index
    buy_signals = get_buy_signal_indices(df, tk)
    buys_2x, buys_1x = simulate_and_track(close, dates, buy_signals)

    # MA200
    sma200 = pd.Series(close).rolling(200, min_periods=200).mean().values

    # ── Chart ──
    fig, axes = plt.subplots(3, 1, figsize=(20, 14), height_ratios=[5, 1.2, 1.2],
                              gridspec_kw={'hspace': 0.08})
    fig.suptitle(f'{tk}: V4 Signal (2x Leverage) vs Month-end DCA (1x)',
                 fontsize=16, fontweight='bold', y=0.98)

    ax1 = axes[0]
    ax_amt = axes[1]
    ax_cum = axes[2]

    # --- Panel 1: Price + Buy Points ---
    ax1.plot(dates, close, color='#333333', linewidth=0.8, alpha=0.9, label=f'{tk} Price')
    ax1.plot(dates, sma200, color='#999999', linewidth=0.7, alpha=0.5, linestyle='--', label='200d SMA')

    # Month-end DCA (1x) - small gray dots
    if buys_1x:
        d1x = [b['date'] for b in buys_1x]
        p1x = [b['price'] for b in buys_1x]
        ax1.scatter(d1x, p1x, color='#AAAAAA', s=12, alpha=0.4, zorder=3, label=f'1x DCA ({len(buys_1x)})')

    # V4 signal (2x) - large red triangles
    if buys_2x:
        d2x = [b['date'] for b in buys_2x]
        p2x = [b['price'] for b in buys_2x]
        ax1.scatter(d2x, p2x, color='#E74C3C', s=80, marker='^', zorder=5,
                    edgecolors='darkred', linewidths=0.5, label=f'2x V4 Signal ({len(buys_2x)})')

    ax1.set_ylabel('Price ($)', fontsize=11)
    ax1.set_yscale('log')
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim(dates[0], dates[-1])

    # Year labels
    years = sorted(set(d.year for d in dates))
    for yr in years:
        if yr % 2 == 0:
            yr_start = pd.Timestamp(f'{yr}-01-01')
            ax1.axvline(yr_start, color='#DDDDDD', linewidth=0.5, alpha=0.5)

    # Shade bear markets (simple: below MA200)
    above = close > sma200
    in_bear = False; bear_start = None
    for i in range(200, len(close)):
        if not above[i] and not in_bear:
            bear_start = dates[i]; in_bear = True
        elif above[i] and in_bear:
            ax1.axvspan(bear_start, dates[i], color='#FFE0E0', alpha=0.3)
            in_bear = False
    if in_bear:
        ax1.axvspan(bear_start, dates[-1], color='#FFE0E0', alpha=0.3)

    # --- Panel 2: Buy amounts bar chart ---
    # 2x buys
    if buys_2x:
        ax_amt.bar([b['date'] for b in buys_2x], [b['amount'] for b in buys_2x],
                   color='#E74C3C', alpha=0.8, width=5, label='2x Leverage Buy')
    # 1x buys (thinner)
    if buys_1x:
        ax_amt.bar([b['date'] for b in buys_1x], [b['amount'] for b in buys_1x],
                   color='#AAAAAA', alpha=0.3, width=3, label='1x DCA Buy')

    ax_amt.set_ylabel('$ Bought', fontsize=9)
    ax_amt.legend(loc='upper left', fontsize=8)
    ax_amt.set_xlim(dates[0], dates[-1])
    ax_amt.grid(True, alpha=0.2)

    # --- Panel 3: Cumulative invested in 2x vs 1x ---
    all_events = []
    for b in buys_2x:
        all_events.append((b['date'], b['amount'], '2x'))
    for b in buys_1x:
        all_events.append((b['date'], b['amount'], '1x'))
    all_events.sort(key=lambda x: x[0])

    cum_2x = 0; cum_1x = 0
    cum_dates = []; cum_2x_vals = []; cum_1x_vals = []
    for dt, amt, typ in all_events:
        if typ == '2x':
            cum_2x += amt
        else:
            cum_1x += amt
        cum_dates.append(dt)
        cum_2x_vals.append(cum_2x)
        cum_1x_vals.append(cum_1x)

    ax_cum.fill_between(cum_dates, 0, cum_2x_vals, color='#E74C3C', alpha=0.4, label='Cumulative 2x invested')
    ax_cum.fill_between(cum_dates, cum_2x_vals,
                        [a + b for a, b in zip(cum_2x_vals, [0]*len(cum_1x_vals))],
                        color='#E74C3C', alpha=0.0)
    total_line = [a + b for a, b in zip(cum_2x_vals, cum_1x_vals)]

    ax_cum.fill_between(cum_dates, cum_2x_vals, total_line, color='#AAAAAA', alpha=0.3, label='Cumulative 1x invested')
    ax_cum.plot(cum_dates, total_line, color='#333333', linewidth=0.8)

    final_2x_pct = cum_2x / (cum_2x + cum_1x) * 100 if (cum_2x + cum_1x) > 0 else 0
    ax_cum.set_ylabel('Cumulative $', fontsize=9)
    ax_cum.set_xlabel('Date', fontsize=10)
    ax_cum.legend(loc='upper left', fontsize=8)
    ax_cum.set_xlim(dates[0], dates[-1])
    ax_cum.grid(True, alpha=0.2)

    # Stats annotation
    total_invested = cum_2x + cum_1x
    stats_text = (
        f"Total Invested: ${total_invested:,.0f}\n"
        f"2x Leverage: ${cum_2x:,.0f} ({final_2x_pct:.0f}%)\n"
        f"1x DCA: ${cum_1x:,.0f} ({100-final_2x_pct:.0f}%)\n"
        f"V4 Signals: {len(buys_2x)} | DCA Buys: {len(buys_1x)}"
    )
    ax1.text(0.98, 0.02, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), f'v4_leverage_timeline_{tk}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

print("\n  Done.")
