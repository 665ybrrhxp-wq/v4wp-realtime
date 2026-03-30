"""
V4 Signal Anatomy: 좋은 신호 vs 고점 신호 — 왜 고점에서 사는가?
================================================================
각 V4 매수 신호의:
- Forward return (30d, 60d, 90d)
- V4 내부 상태 (score, sub-indicators, activity, duration)
- 시장 위치 (고점 대비 drawdown, MA200 위/아래, 최근 추세)
를 분석해서 고점 매수의 원인을 규명.
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
from real_market_backtest import (
    calc_v4_score, calc_v4_subindicators, detect_signal_events,
    build_price_filter, smooth_earnings_volume,
)

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


def analyze_signals(df, ticker):
    try:
        df_s = smooth_earnings_volume(df, ticker=ticker)
    except Exception:
        df_s = df.copy()

    close = df_s['Close'].values
    dates = df_s.index
    n = len(close)

    score = calc_v4_score(df_s, w=V4_W, divgate_days=DIVGATE)
    subind = calc_v4_subindicators(df_s, w=V4_W, divgate_days=DIVGATE)
    events = detect_signal_events(score, th=SIGNAL_TH, cooldown=COOLDOWN)
    pf = build_price_filter(df_s, er_q=ER_Q, atr_q=ATR_Q, lookback=LOOKBACK)

    sma200 = pd.Series(close).rolling(200, min_periods=200).mean().values
    rolling_high_60 = pd.Series(close).rolling(60, min_periods=1).max().values
    rolling_high_120 = pd.Series(close).rolling(120, min_periods=1).max().values

    signals = []
    for ev in events:
        if ev['type'] != 'bottom':
            continue
        if not pf(ev['peak_idx']):
            continue
        dur = ev['end_idx'] - ev['start_idx'] + 1
        ci = ev['start_idx'] + CONFIRM - 1
        if ci > ev['end_idx'] or dur < CONFIRM or ci >= n:
            continue

        pidx = ev['peak_idx']
        buy_idx = ci
        price = close[buy_idx]

        # Forward returns
        fwd = {}
        for h in [30, 60, 90]:
            if buy_idx + h < n:
                fwd[h] = (close[buy_idx + h] / price - 1) * 100
            else:
                fwd[h] = None

        # Market position
        dd_from_60h = (rolling_high_60[buy_idx] - price) / rolling_high_60[buy_idx] * 100
        dd_from_120h = (rolling_high_120[buy_idx] - price) / rolling_high_120[buy_idx] * 100
        above_ma200 = close[buy_idx] > sma200[buy_idx] if not np.isnan(sma200[buy_idx]) else None

        # Recent trend: 20-day return before signal
        trend_20d = (close[buy_idx] / close[max(0, buy_idx-20)] - 1) * 100

        # Was this near ATH? (within 5% of all-time high up to that point)
        ath = np.max(close[:buy_idx+1])
        near_ath = (ath - price) / ath * 100 < 5

        signals.append({
            'date': dates[buy_idx],
            'price': price,
            'buy_idx': buy_idx,
            'peak_idx': pidx,
            'score': ev['peak_val'],
            'duration': dur,
            's_force': subind['s_force'].iloc[pidx] if pidx < len(subind) else 0,
            's_div': subind['s_div'].iloc[pidx] if pidx < len(subind) else 0,
            's_conc': subind['s_conc'].iloc[pidx] if pidx < len(subind) else 0,
            'act': subind['act'].iloc[pidx] if pidx < len(subind) else 0,
            'fwd_30': fwd.get(30), 'fwd_60': fwd.get(60), 'fwd_90': fwd.get(90),
            'dd_60h': dd_from_60h,
            'dd_120h': dd_from_120h,
            'above_ma200': above_ma200,
            'trend_20d': trend_20d,
            'near_ath': near_ath,
        })

    return signals


for tk in ['QQQ', 'VOO']:
    print(f"\n{'=' * 130}")
    print(f"  {tk}: V4 SIGNAL ANATOMY")
    print(f"{'=' * 130}")

    df = download_max(tk)
    if df is None:
        continue

    signals = analyze_signals(df, tk)
    if not signals:
        continue

    # Classify: good (90d fwd > 0) vs bad (90d fwd <= 0)
    good = [s for s in signals if s['fwd_90'] is not None and s['fwd_90'] > 0]
    bad = [s for s in signals if s['fwd_90'] is not None and s['fwd_90'] <= 0]
    na = [s for s in signals if s['fwd_90'] is None]

    # ── Full signal table ──
    print(f"\n  Total signals: {len(signals)} (Good: {len(good)}, Bad: {len(bad)}, Recent: {len(na)})")
    print(f"\n  {'Date':>12s} {'Price':>8s} {'Score':>6s} {'Act':>4s} {'Dur':>4s} "
          f"{'S_F':>6s} {'S_D':>6s} {'S_C':>6s} "
          f"{'DD60':>6s} {'DD120':>6s} {'MA200':>6s} {'Trnd20':>7s} {'ATH?':>5s} "
          f"{'30d':>7s} {'60d':>7s} {'90d':>7s} {'Result':>7s}")
    print(f"  {'=' * 128}")

    for s in signals:
        f30 = f"{s['fwd_30']:>+6.1f}%" if s['fwd_30'] is not None else "   N/A"
        f60 = f"{s['fwd_60']:>+6.1f}%" if s['fwd_60'] is not None else "   N/A"
        f90 = f"{s['fwd_90']:>+6.1f}%" if s['fwd_90'] is not None else "   N/A"
        result = "GOOD" if (s['fwd_90'] is not None and s['fwd_90'] > 0) else (
                 "BAD" if (s['fwd_90'] is not None and s['fwd_90'] <= 0) else "?")
        ma = "Above" if s['above_ma200'] else "Below" if s['above_ma200'] is not None else "N/A"
        ath = "YES" if s['near_ath'] else ""

        marker = " ***" if (s['fwd_90'] is not None and s['fwd_90'] < -10) else ""

        print(f"  {s['date'].strftime('%Y-%m-%d'):>12s} ${s['price']:>7.1f} {s['score']:>5.2f} {s['act']:>3.0f} {s['duration']:>4d} "
              f"{s['s_force']:>+5.2f} {s['s_div']:>+5.2f} {s['s_conc']:>+5.2f} "
              f"{s['dd_60h']:>5.1f}% {s['dd_120h']:>5.1f}% {ma:>6s} {s['trend_20d']:>+6.1f}% {ath:>5s} "
              f"{f30} {f60} {f90} {result:>6s}{marker}")

    # ── Statistical comparison ──
    print(f"\n  {'=' * 130}")
    print(f"  GOOD vs BAD Signal Comparison (90d forward return)")
    print(f"  {'=' * 130}")

    def avg_field(lst, field):
        vals = [s[field] for s in lst if s[field] is not None and not (isinstance(s[field], float) and np.isnan(s[field]))]
        return np.mean(vals) if vals else 0

    metrics = ['score', 'act', 'duration', 's_force', 's_div', 's_conc',
               'dd_60h', 'dd_120h', 'trend_20d']

    print(f"  {'Metric':<15s} {'GOOD (n={len(good)})':>15s} {'BAD (n={len(bad)})':>15s} {'Difference':>12s} {'Insight':>25s}")
    print(f"  {'=' * 90}")

    for m in metrics:
        g = avg_field(good, m)
        b = avg_field(bad, m)
        diff = g - b

        # Insight
        if m == 'score':
            insight = "Higher score = better" if diff > 0.05 else "Similar"
        elif m == 'act':
            insight = "More aligned = better" if diff > 0.1 else "Similar"
        elif m == 'duration':
            insight = "Longer = better" if diff > 1 else "Similar"
        elif m == 'dd_60h':
            insight = "DEEPER dip = better" if diff > 1 else ("HIGH ENTRY = worse" if diff < -1 else "Similar")
        elif m == 'dd_120h':
            insight = "DEEPER dip = better" if diff > 1 else ("HIGH ENTRY = worse" if diff < -1 else "Similar")
        elif m == 'trend_20d':
            insight = "Rising into signal" if diff > 1 else ("Falling = better" if diff < -1 else "Similar")
        elif m == 's_force':
            insight = "Stronger momentum" if diff > 0.05 else "Similar"
        elif m == 's_div':
            insight = "More divergence" if diff > 0.05 else "Similar"
        elif m == 's_conc':
            insight = "Better alignment" if diff > 0.05 else "Similar"
        else:
            insight = ""

        print(f"  {m:<15s} {g:>+14.3f} {b:>+14.3f} {diff:>+11.3f}  {insight}")

    # MA200 distribution
    g_above = sum(1 for s in good if s['above_ma200'] == True)
    b_above = sum(1 for s in bad if s['above_ma200'] == True)
    g_pct = g_above / len(good) * 100 if good else 0
    b_pct = b_above / len(bad) * 100 if bad else 0
    print(f"  {'above_ma200':<15s} {g_pct:>13.0f}% {b_pct:>13.0f}% {g_pct-b_pct:>+10.0f}%p  {'Regime matters' if abs(g_pct-b_pct) > 10 else 'Similar'}")

    # Near ATH
    g_ath = sum(1 for s in good if s['near_ath']) / len(good) * 100 if good else 0
    b_ath = sum(1 for s in bad if s['near_ath']) / len(bad) * 100 if bad else 0
    print(f"  {'near_ath':<15s} {g_ath:>13.0f}% {b_ath:>13.0f}% {g_ath-b_ath:>+10.0f}%p  {'ATH = danger' if b_ath > g_ath + 10 else 'Similar'}")

    # ── Key finding: what's unique about bad signals? ──
    print(f"\n  {'=' * 130}")
    print(f"  BAD SIGNALS (90d loss) — Detail:")
    print(f"  {'=' * 130}")

    for s in bad:
        f90 = s['fwd_90']
        reasons = []
        if s['dd_60h'] < 3:
            reasons.append(f"60d High -{s['dd_60h']:.1f}% (shallow dip)")
        if s['near_ath']:
            reasons.append("Near ATH")
        if s['above_ma200']:
            reasons.append("Above MA200")
        if s['trend_20d'] > 5:
            reasons.append(f"20d trend +{s['trend_20d']:.1f}% (already risen)")
        if s['act'] < 3:
            reasons.append(f"Activity={s['act']:.0f} (not all aligned)")
        if s['s_conc'] < 0:
            reasons.append(f"S_conc={s['s_conc']:.2f} (price-vol disagree)")

        reason_str = " | ".join(reasons) if reasons else "No obvious red flag"
        print(f"  {s['date'].strftime('%Y-%m-%d')} ${s['price']:.1f}  90d={f90:+.1f}%  {reason_str}")


    # ── Chart: good vs bad signals on price ──
    fig, ax = plt.subplots(figsize=(18, 7))
    close = df['Close'].values
    dates = df.index

    ax.plot(dates, close, color='#333', linewidth=0.7, alpha=0.8)
    ax.set_yscale('log')

    for s in good:
        ax.scatter(s['date'], s['price'], color='#27AE60', s=60, marker='^',
                   zorder=5, edgecolors='darkgreen', linewidths=0.3)
    for s in bad:
        ax.scatter(s['date'], s['price'], color='#E74C3C', s=100, marker='v',
                   zorder=6, edgecolors='darkred', linewidths=0.5)
        ax.annotate(f"{s['fwd_90']:+.0f}%", (s['date'], s['price']),
                    textcoords="offset points", xytext=(0, -14),
                    fontsize=7, color='red', ha='center', fontweight='bold')

    ax.scatter([], [], color='#27AE60', s=60, marker='^', label=f'Good 90d (n={len(good)})')
    ax.scatter([], [], color='#E74C3C', s=100, marker='v', label=f'Bad 90d (n={len(bad)})')

    ax.set_title(f'{tk}: Good Signals (green) vs Bad Signals (red) — 90d Forward Return',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Price ($)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.2)

    out = os.path.join(os.path.dirname(__file__), f'v4_signal_anatomy_{tk}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Chart saved: {out}")

print(f"\n{'=' * 130}")
print("  Done.")
