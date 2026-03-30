"""
VN60+GEO-OP 상세 데이터 출력
============================
전 티커 연도별 상세 + 개별 시그널 리스트
"""
import sys, os, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import yfinance as yf
from real_market_backtest import (
    calc_v4_score, calc_v4_subindicators,
    detect_signal_events, build_price_filter, smooth_earnings_volume,
)

TICKERS = {
    'TSLA': 'Tech', 'NVDA': 'Tech', 'AVGO': 'Tech',
    'AMZN': 'Tech', 'GOOGL': 'Tech', 'PLTR': 'Tech',
    'COIN': 'Fintech', 'HOOD': 'Fintech',
    'HIMS': 'Growth', 'RKLB': 'Growth', 'JOBY': 'Growth',
    'IONQ': 'Quantum', 'QQQ': 'Benchmark', 'VOO': 'Benchmark',
}

SIGNAL_TH = 0.05; COOLDOWN = 5; ER_Q = 80; ATR_Q = 40
LOOKBACK = 252; CONFIRM_DAYS = 1; BUY_DD_LB = 20; BUY_DD_TH = 0.03; DIVGATE = 3
MONTHLY_DEPOSIT = 500.0; SIGNAL_BUY_PCT = 0.50
EXPENSE_2X = 0.0095 / 252; EXPENSE_3X = 0.0100 / 252


def download_max(ticker):
    t = yf.Ticker(ticker)
    df = t.history(period='max', auto_adjust=True)
    if df is None or len(df) == 0: return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.dropna(subset=['Close', 'Volume'])
    df = df[df['Close'] > 0]
    return df


def build_synthetic_lev(close, leverage, expense_daily):
    daily_ret = np.diff(close) / close[:-1]
    lev = np.zeros(len(close)); lev[0] = close[0]
    for i in range(1, len(close)):
        lr = leverage * daily_ret[i - 1] - expense_daily
        lev[i] = lev[i - 1] * (1 + lr)
        if lev[i] < 0.001: lev[i] = 0.001
    return lev


def get_buy_signals(df, score, tk):
    events = detect_signal_events(score, th=SIGNAL_TH, cooldown=COOLDOWN)
    pf = build_price_filter(df, er_q=ER_Q, atr_q=ATR_Q, lookback=LOOKBACK)
    close = df['Close'].values
    rolling_high = pd.Series(close).rolling(BUY_DD_LB, min_periods=1).max().values
    n = len(df)
    buy_indices = []
    for ev in events:
        if ev['type'] != 'bottom': continue
        if not pf(ev['peak_idx']): continue
        dur = ev['end_idx'] - ev['start_idx'] + 1
        ci = ev['start_idx'] + CONFIRM_DAYS - 1
        if ci > ev['end_idx'] or dur < CONFIRM_DAYS or ci >= n: continue
        pidx = ev['peak_idx']
        rh = rolling_high[pidx]
        dd = (rh - close[pidx]) / rh if rh > 0 else 0
        if dd < BUY_DD_TH: continue
        buy_indices.append(ci)
    return buy_indices, events


def simulate_yearly(close, close_2x, close_3x, buy_indices, dates):
    n = len(close)
    month_periods = pd.Series(dates).dt.to_period('M')
    month_map = {}
    for i, mp in enumerate(month_periods):
        key = str(mp)
        if key not in month_map: month_map[key] = {'first': i, 'last': i}
        else: month_map[key]['last'] = i
    sorted_months = sorted(month_map.keys())
    buy_set = set(buy_indices)
    cash_a = 0.0; sh_1x_a = 0.0; sh_2x_a = 0.0
    cash_b = 0.0; sh_1x_b = 0.0; sh_3x_b = 0.0
    cash_c = 0.0; sh_1x_c = 0.0
    yr_data = {}; prev_yr = None; total_dep = 0.0
    def pf_a(idx): return sh_1x_a * close[idx] + sh_2x_a * close_2x[idx] + cash_a
    def pf_b(idx): return sh_1x_b * close[idx] + sh_3x_b * close_3x[idx] + cash_b
    def pf_c(idx): return sh_1x_c * close[idx] + cash_c
    for mk in sorted_months:
        mm = month_map[mk]; fi, li = mm['first'], mm['last']; yr = int(mk[:4])
        if yr != prev_yr:
            if prev_yr is not None:
                ref = fi - 1 if fi > 0 else fi
                yr_data[prev_yr]['end_a'] = pf_a(ref)
                yr_data[prev_yr]['end_b'] = pf_b(ref)
                yr_data[prev_yr]['end_c'] = pf_c(ref)
            yr_data[yr] = {'start_a': pf_a(fi), 'start_b': pf_b(fi), 'start_c': pf_c(fi), 'deposits': 0.0, 'sigs': 0}
            prev_yr = yr
        cash_a += MONTHLY_DEPOSIT; cash_b += MONTHLY_DEPOSIT; cash_c += MONTHLY_DEPOSIT
        yr_data[yr]['deposits'] += MONTHLY_DEPOSIT; total_dep += MONTHLY_DEPOSIT
        for day_idx in range(fi, li + 1):
            if day_idx in buy_set:
                if cash_a > 1.0: amt = cash_a * SIGNAL_BUY_PCT; sh_2x_a += amt / close_2x[day_idx]; cash_a -= amt
                if cash_b > 1.0: amt = cash_b * SIGNAL_BUY_PCT; sh_3x_b += amt / close_3x[day_idx]; cash_b -= amt
                yr_data[yr]['sigs'] += 1
        if cash_a > 1.0: sh_1x_a += cash_a / close[li]; cash_a = 0.0
        if cash_b > 1.0: sh_1x_b += cash_b / close[li]; cash_b = 0.0
        if cash_c > 1.0: sh_1x_c += cash_c / close[li]; cash_c = 0.0
    if prev_yr is not None:
        yr_data[prev_yr]['end_a'] = pf_a(n - 1)
        yr_data[prev_yr]['end_b'] = pf_b(n - 1)
        yr_data[prev_yr]['end_c'] = pf_c(n - 1)
    yr_results = []
    for yr in sorted(yr_data.keys()):
        yd = yr_data[yr]
        if 'end_a' not in yd: continue
        rets = {}
        for mode in ['a', 'b', 'c']:
            d = yd[f'start_{mode}'] + yd['deposits'] * 0.5
            if d > 10:
                val = (yd[f'end_{mode}'] - yd[f'start_{mode}'] - yd['deposits']) / d * 100
                rets[mode] = val if np.isfinite(val) else 0.0
            else: rets[mode] = 0
        yr_results.append({
            'yr': yr, 'ret_2x': rets['a'], 'ret_3x': rets['b'], 'ret_dca': rets['c'],
            'edge_2x': rets['a'] - rets['c'], 'edge_3x': rets['b'] - rets['c'], 'sigs': yd['sigs'],
            'val_2x': yd['end_a'], 'val_3x': yd['end_b'], 'val_dca': yd['end_c'],
            'dep': yd['deposits'],
        })
    return yr_results, pf_a(n-1), pf_b(n-1), pf_c(n-1), total_dep


def main():
    sep = '=' * 140
    sep2 = '-' * 140

    print(sep)
    print("  VN60+GEO-OP 전체 상세 데이터")
    print(f"  Pipeline: th={SIGNAL_TH}, PF(ER<{ER_Q}%/ATR>{ATR_Q}%), DD>={BUY_DD_TH*100:.0f}%, confirm={CONFIRM_DAYS}d")
    print(sep)

    # Load
    print("\n  데이터 로딩...")
    data = {}
    for tk in TICKERS:
        df = download_max(tk)
        if df is None or len(df) < 300: continue
        df = smooth_earnings_volume(df, ticker=tk)
        data[tk] = df
        print(f"    {tk}: {len(df)} bars ({df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')})")

    # ═══════════════════════════════════════════════════════
    # SECTION 1: 전 티커 연도별 상세
    # ═══════════════════════════════════════════════════════
    all_yr = {}
    all_sigs = {}

    for tk, df in data.items():
        score = calc_v4_score(df, w=20, divgate_days=DIVGATE)
        subind = calc_v4_subindicators(df, w=20, divgate_days=DIVGATE)
        buy_idx, all_events = get_buy_signals(df, score, tk)
        close = df['Close'].values
        close_2x = build_synthetic_lev(close, 2, EXPENSE_2X)
        close_3x = build_synthetic_lev(close, 3, EXPENSE_3X)
        yr_results, fa, fb, fc, td = simulate_yearly(close, close_2x, close_3x, buy_idx, df.index)
        all_yr[tk] = {'yr_results': yr_results, 'final_2x': fa, 'final_3x': fb, 'final_dca': fc, 'total_dep': td}

        # Collect signal details
        sigs = []
        n = len(df)
        for idx in buy_idx:
            fwd = {}
            for d in [7, 14, 30, 60, 90, 180]:
                if idx + d < n:
                    fwd[d] = (close[idx + d] / close[idx] - 1) * 100
                else:
                    fwd[d] = None
            # MDD within 90d
            end_i = min(idx + 90, n)
            mdd = (min(close[idx:end_i]) / close[idx] - 1) * 100 if end_i > idx + 1 else 0
            # 20d high drawdown at entry
            rh = pd.Series(close).rolling(20, min_periods=1).max().values
            dd_at_entry = (rh[idx] - close[idx]) / rh[idx] * 100 if rh[idx] > 0 else 0

            sigs.append({
                'idx': idx, 'date': df.index[idx].strftime('%Y-%m-%d'),
                'price': close[idx],
                'score': score.iloc[idx],
                's_force': subind['s_force'].iloc[idx],
                's_div': subind['s_div'].iloc[idx],
                'dd_entry': dd_at_entry,
                'fwd': fwd, 'mdd_90': mdd,
            })
        all_sigs[tk] = sigs

    # Print yearly detail per ticker
    for tk in data:
        yr_data = all_yr[tk]
        sec = TICKERS[tk]
        print(f"\n{sep}")
        print(f"  [{tk}] ({sec}) 연도별 상세  |  총 입금: ${yr_data['total_dep']:,.0f}")
        print(f"  최종: 2x=${yr_data['final_2x']:,.0f}  3x=${yr_data['final_3x']:,.0f}  DCA=${yr_data['final_dca']:,.0f}")
        print(sep)
        print(f"  {'Year':>6} │ {'2x':>9} {'3x':>9} {'DCA':>9} │ {'Edge2x':>8} {'Edge3x':>8} │ "
              f"{'Sig':>4} │ {'누적2x':>14} {'누적3x':>14} {'누적DCA':>14} │ Note")
        print(f"  {sep2}")

        cum_dep = 0
        for yr in yr_data['yr_results']:
            cum_dep += yr['dep']
            note = ''
            if yr['ret_dca'] < -20: note = 'CRASH'
            elif yr['ret_dca'] < -5: note = 'BEAR'
            elif yr['ret_dca'] > 50: note = 'BOOM!'
            elif yr['ret_dca'] > 30: note = 'BOOM'
            elif yr['ret_dca'] > 15: note = 'BULL'

            marker = ''
            if yr['edge_2x'] > 10: marker = ' ★★'
            elif yr['edge_2x'] > 5: marker = ' ★'
            elif yr['edge_2x'] < -5: marker = ' ▼'

            print(f"  {yr['yr']:>6} │ {yr['ret_2x']:>+8.1f}% {yr['ret_3x']:>+8.1f}% {yr['ret_dca']:>+8.1f}% │ "
                  f"{yr['edge_2x']:>+7.1f}% {yr['edge_3x']:>+7.1f}% │ {yr['sigs']:>4} │ "
                  f"${yr['val_2x']:>13,.0f} ${yr['val_3x']:>13,.0f} ${yr['val_dca']:>13,.0f} │ {note}{marker}")

        # yearly avg
        yrs = yr_data['yr_results']
        if yrs:
            a2 = np.mean([y['ret_2x'] for y in yrs])
            a3 = np.mean([y['ret_3x'] for y in yrs])
            ad = np.mean([y['ret_dca'] for y in yrs])
            print(f"  {sep2}")
            print(f"  {'AVG':>6} │ {a2:>+8.1f}% {a3:>+8.1f}% {ad:>+8.1f}% │ {a2-ad:>+7.1f}% {a3-ad:>+7.1f}% │ "
                  f"{sum(y['sigs'] for y in yrs):>4} │")

    # ═══════════════════════════════════════════════════════
    # SECTION 2: 전 티커 개별 시그널 리스트
    # ═══════════════════════════════════════════════════════
    for tk in data:
        sigs = all_sigs[tk]
        if not sigs: continue
        sec = TICKERS[tk]
        print(f"\n{sep}")
        print(f"  [{tk}] ({sec}) 개별 시그널 리스트 — 총 {len(sigs)}회")
        print(sep)
        print(f"  {'#':>3} {'Date':>12} {'Price':>10} │ {'Score':>6} {'Force':>6} {'Div':>6} {'DD%':>6} │ "
              f"{'7d':>7} {'14d':>7} {'30d':>7} {'60d':>7} {'90d':>7} {'180d':>7} │ {'MDD90':>7} │ Result")
        print(f"  {sep2}")

        total_win = 0; total_eval = 0
        for i, s in enumerate(sigs):
            f = s['fwd']
            def fmt(v): return f"{v:>+6.1f}%" if v is not None else "    N/A"

            result = ''
            if f[90] is not None:
                total_eval += 1
                if f[90] > 50: result = '★★ BIG WIN'; total_win += 1
                elif f[90] > 20: result = '★ WIN'; total_win += 1
                elif f[90] > 0: result = 'win'; total_win += 1
                elif f[90] > -10: result = 'small loss'
                elif f[90] > -30: result = 'LOSS'
                else: result = '★★ BIG LOSS'

            print(f"  {i+1:>3} {s['date']:>12} ${s['price']:>9.2f} │ "
                  f"{s['score']:>5.3f} {s['s_force']:>+5.3f} {s['s_div']:>+5.3f} {s['dd_entry']:>5.1f}% │ "
                  f"{fmt(f[7])} {fmt(f[14])} {fmt(f[30])} {fmt(f[60])} {fmt(f[90])} {fmt(f[180])} │ "
                  f"{s['mdd_90']:>+6.1f}% │ {result}")

        if total_eval > 0:
            print(f"  {sep2}")
            print(f"  Hit Rate (90d): {total_win}/{total_eval} = {total_win/total_eval*100:.1f}%")
            fwd90_valid = [s['fwd'][90] for s in sigs if s['fwd'][90] is not None]
            if fwd90_valid:
                print(f"  Fwd 90d: 평균={np.mean(fwd90_valid):+.1f}%  중앙값={np.median(fwd90_valid):+.1f}%  "
                      f"최고={max(fwd90_valid):+.1f}%  최악={min(fwd90_valid):+.1f}%")

    # ═══════════════════════════════════════════════════════
    # SECTION 3: 크로스 티커 연도 매트릭스 (Edge 2x)
    # ═══════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  [MATRIX] 티커 × 연도 Edge 2x (%p)")
    print(sep)

    # Collect all years
    all_years = set()
    for tk in all_yr:
        for yr in all_yr[tk]['yr_results']:
            all_years.add(yr['yr'])
    all_years = sorted(all_years)

    # Header
    tk_list = list(data.keys())
    hdr = f"  {'Year':>6} │"
    for tk in tk_list:
        hdr += f" {tk:>6}"
    hdr += " │   AVG"
    print(hdr)
    print(f"  {'-' * (10 + 7 * len(tk_list) + 10)}")

    for yr in all_years:
        line = f"  {yr:>6} │"
        edges = []
        for tk in tk_list:
            found = None
            for y in all_yr[tk]['yr_results']:
                if y['yr'] == yr:
                    found = y['edge_2x']
                    break
            if found is not None:
                if found > 5: line += f" \033[92m{found:>+5.1f}\033[0m"
                elif found < -3: line += f" \033[91m{found:>+5.1f}\033[0m"
                else: line += f" {found:>+5.1f}"
                edges.append(found)
            else:
                line += "      -"
        avg_e = np.mean(edges) if edges else 0
        line += f" │ {avg_e:>+5.1f}"
        print(line)

    # Ticker avg row
    line = f"  {'AVG':>6} │"
    for tk in tk_list:
        yrs = all_yr[tk]['yr_results']
        if yrs:
            avg = np.mean([y['edge_2x'] for y in yrs])
            line += f" {avg:>+5.1f}"
        else:
            line += "      -"
    line += " │"
    print(f"  {'-' * (10 + 7 * len(tk_list) + 10)}")
    print(line)

    # ═══════════════════════════════════════════════════════
    # SECTION 4: 시그널 스코어 구간별 성과
    # ═══════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  [SCORE BINS] 시그널 스코어 구간별 90일 성과")
    print(sep)

    all_sig_flat = []
    for tk in all_sigs:
        for s in all_sigs[tk]:
            if s['fwd'][90] is not None:
                all_sig_flat.append({'tk': tk, 'score': s['score'], 'fwd90': s['fwd'][90],
                                     'force': s['s_force'], 'div': s['s_div'], 'mdd': s['mdd_90']})

    if all_sig_flat:
        df_sig = pd.DataFrame(all_sig_flat)
        bins = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.01]
        labels = ['0.00-0.10', '0.10-0.20', '0.20-0.30', '0.30-0.50', '0.50-0.70', '0.70-1.00']
        df_sig['bin'] = pd.cut(df_sig['score'], bins=bins, labels=labels, right=False)

        print(f"\n  {'Score 구간':<12} │ {'N':>5} {'Hit90':>7} {'Avg':>8} {'Med':>8} {'Min':>8} {'Max':>8} │ {'AvgMDD':>8} │ {'AvgForce':>9} {'AvgDiv':>8}")
        print(f"  {'-' * 95}")

        for label in labels:
            grp = df_sig[df_sig['bin'] == label]
            if len(grp) == 0:
                print(f"  {label:<12} │ {0:>5} {'':>7} {'':>8} {'':>8} {'':>8} {'':>8} │ {'':>8} │")
                continue
            hit = (grp['fwd90'] > 0).mean() * 100
            print(f"  {label:<12} │ {len(grp):>5} {hit:>6.1f}% {grp['fwd90'].mean():>+7.1f}% "
                  f"{grp['fwd90'].median():>+7.1f}% {grp['fwd90'].min():>+7.1f}% {grp['fwd90'].max():>+7.1f}% │ "
                  f"{grp['mdd'].mean():>+7.1f}% │ {grp['force'].mean():>+8.3f} {grp['div'].mean():>+7.3f}")

        print(f"  {'-' * 95}")
        print(f"  {'TOTAL':<12} │ {len(df_sig):>5} {(df_sig['fwd90']>0).mean()*100:>6.1f}% "
              f"{df_sig['fwd90'].mean():>+7.1f}% {df_sig['fwd90'].median():>+7.1f}%")

    # ═══════════════════════════════════════════════════════
    # SECTION 5: 시그널 DD(진입낙폭) 구간별 성과
    # ═══════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  [DD BINS] 진입 시 낙폭(20d고점 대비) 구간별 90일 성과")
    print(sep)

    if all_sig_flat:
        all_dd_flat = []
        for tk in all_sigs:
            for s in all_sigs[tk]:
                if s['fwd'][90] is not None:
                    all_dd_flat.append({'dd': s['dd_entry'], 'fwd90': s['fwd'][90], 'score': s['score']})

        df_dd = pd.DataFrame(all_dd_flat)
        dd_bins = [0, 3, 5, 8, 12, 20, 100]
        dd_labels = ['3-5%', '5-8%', '8-12%', '12-20%', '20%+']
        # Actually redo with proper bins
        dd_bins2 = [3, 5, 8, 12, 20, 100]
        dd_labels2 = ['3-5%', '5-8%', '8-12%', '12-20%', '20%+']
        df_dd['bin'] = pd.cut(df_dd['dd'], bins=[3, 5, 8, 12, 20, 100], labels=dd_labels2, right=False)

        print(f"\n  {'DD 구간':<12} │ {'N':>5} {'Hit90':>7} {'Avg Fwd90':>10} {'Med Fwd90':>10} {'Avg Score':>10}")
        print(f"  {'-' * 62}")

        for label in dd_labels2:
            grp = df_dd[df_dd['bin'] == label]
            if len(grp) == 0:
                print(f"  {label:<12} │ {0:>5}")
                continue
            hit = (grp['fwd90'] > 0).mean() * 100
            print(f"  {label:<12} │ {len(grp):>5} {hit:>6.1f}% {grp['fwd90'].mean():>+9.1f}% "
                  f"{grp['fwd90'].median():>+9.1f}% {grp['score'].mean():>9.3f}")

    # ═══════════════════════════════════════════════════════
    # SECTION 6: 최근 시그널 (2024-2026)
    # ═══════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  [RECENT] 2024년 이후 시그널 (전 티커)")
    print(sep)
    print(f"\n  {'#':>3} {'Ticker':>8} {'Date':>12} {'Price':>10} │ {'Score':>6} {'Force':>6} {'Div':>6} {'DD%':>6} │ "
          f"{'30d':>7} {'90d':>7} {'180d':>7} │ Result")
    print(f"  {sep2}")

    recent = []
    for tk in all_sigs:
        for s in all_sigs[tk]:
            if s['date'] >= '2024-01-01':
                recent.append({**s, 'tk': tk})
    recent.sort(key=lambda x: x['date'])

    for i, s in enumerate(recent):
        f = s['fwd']
        def fmt(v): return f"{v:>+6.1f}%" if v is not None else "    N/A"
        result = ''
        if f[90] is not None:
            if f[90] > 50: result = '★★ BIG WIN'
            elif f[90] > 20: result = '★ WIN'
            elif f[90] > 0: result = 'win'
            elif f[90] > -10: result = 'small loss'
            else: result = 'LOSS'

        print(f"  {i+1:>3} {s['tk']:>8} {s['date']:>12} ${s['price']:>9.2f} │ "
              f"{s['score']:>5.3f} {s['s_force']:>+5.3f} {s['s_div']:>+5.3f} {s['dd_entry']:>5.1f}% │ "
              f"{fmt(f[30])} {fmt(f[90])} {fmt(f[180])} │ {result}")

    print(f"\n  Done.\n")


if __name__ == '__main__':
    main()
