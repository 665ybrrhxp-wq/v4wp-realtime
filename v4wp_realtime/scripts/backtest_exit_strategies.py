"""
가용자금 비율 기반 매매 전략 백테스트
=====================================
- 시작 가용자금: $1000
- 매월 가용자금에 $100 추가 입금 (자동 매수 아님, 신호 대기)
- BOTTOM 신호 시: 가용자금의 N% 매수
- TOP 신호 시: 보유량의 N% 매도 → 매도대금 가용자금 편입

실행:
  cd "거래량 백테스트"
  python -m v4wp_realtime.scripts.backtest_exit_strategies
"""

import sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

_script_dir = Path(__file__).resolve().parent
_project_root = str(_script_dir.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd
import json, warnings
warnings.filterwarnings('ignore')

from real_market_backtest import (
    download_data, calc_v4_score, detect_signal_events, build_price_filter,
)

# ============================================================
# 설정
# ============================================================
V4_WINDOW = 20
SIGNAL_THRESHOLD = 0.15
COOLDOWN = 5
ER_QUANTILE = 66
ATR_QUANTILE = 66
LOOKBACK = 252
INITIAL_CASH = 1000.0
MONTHLY_DEPOSIT = 100.0   # 매월 가용자금에 추가 입금 (자동매수 아님)

MARKET_PERIODS = [
    {'name': '1.리먼위기',   'start': '2007-01-01', 'end': '2009-12-31',
     'desc': '서브프라임 -> 리먼 파산 -> 대폭락'},
    {'name': '2.회복장',     'start': '2009-03-01', 'end': '2015-12-31',
     'desc': 'QE 양적완화 -> 장기 상승장'},
    {'name': '3.금리충격',   'start': '2018-01-01', 'end': '2019-12-31',
     'desc': 'Fed 금리인상 -> 크리스마스 폭락'},
    {'name': '4.코로나',     'start': '2020-01-01', 'end': '2021-12-31',
     'desc': 'COVID-19 폭락 -> V자 회복'},
    {'name': '5.인플레하락',  'start': '2022-01-01', 'end': '2023-06-30',
     'desc': '인플레 -> 급격 금리인상 -> 기술주 폭락'},
    {'name': '6.AI+관세',   'start': '2023-07-01', 'end': '2026-03-31',
     'desc': 'ChatGPT 랠리 -> 트럼프 관세 쇼크'},
    {'name': '7.전체',      'start': '2007-01-01', 'end': '2026-03-31',
     'desc': '금융위기부터 현재까지'},
]

# 강한 신호 임계값
STRONG_BUY_TH  = 0.15
STRONG_SELL_TH = -0.25


# ============================================================
# 전략 정의
# ============================================================
STRATEGIES = [
    # ── 유저 제안: 가용자금 비율 기반 ──
    {'id': 'NEW',  'name': '유저안: 30%/50%매수,5%/10%매도',
     'buy_pct_normal': 0.30, 'buy_pct_strong': 0.50,
     'sell_pct_normal': 0.05, 'sell_pct_strong': 0.10},

    # ── 변형: 매수 비율 ──
    {'id': 'N2',  'name': '20%/40%매수,5%/10%매도',
     'buy_pct_normal': 0.20, 'buy_pct_strong': 0.40,
     'sell_pct_normal': 0.05, 'sell_pct_strong': 0.10},

    {'id': 'N3',  'name': '40%/60%매수,5%/10%매도',
     'buy_pct_normal': 0.40, 'buy_pct_strong': 0.60,
     'sell_pct_normal': 0.05, 'sell_pct_strong': 0.10},

    # ── 변형: 매도 비율 ──
    {'id': 'N4',  'name': '30%/50%매수,3%/7%매도',
     'buy_pct_normal': 0.30, 'buy_pct_strong': 0.50,
     'sell_pct_normal': 0.03, 'sell_pct_strong': 0.07},

    {'id': 'N5',  'name': '30%/50%매수,10%/20%매도',
     'buy_pct_normal': 0.30, 'buy_pct_strong': 0.50,
     'sell_pct_normal': 0.10, 'sell_pct_strong': 0.20},

    # ── 변형: 강신호만 ──
    {'id': 'N6',  'name': '강매수만50%,강매도만10%',
     'buy_pct_normal': 0.0, 'buy_pct_strong': 0.50,
     'sell_pct_normal': 0.0, 'sell_pct_strong': 0.10},

    # ── 변형: 동일비율 ──
    {'id': 'N7',  'name': '전매수30%,전매도5%',
     'buy_pct_normal': 0.30, 'buy_pct_strong': 0.30,
     'sell_pct_normal': 0.05, 'sell_pct_strong': 0.05},

    {'id': 'N8',  'name': '전매수30%,전매도10%',
     'buy_pct_normal': 0.30, 'buy_pct_strong': 0.30,
     'sell_pct_normal': 0.10, 'sell_pct_strong': 0.10},

    # ── 변형: 공격적 ──
    {'id': 'N9',  'name': '50%/70%매수,5%/10%매도',
     'buy_pct_normal': 0.50, 'buy_pct_strong': 0.70,
     'sell_pct_normal': 0.05, 'sell_pct_strong': 0.10},

    {'id': 'N10', 'name': '30%/50%매수,5%/15%매도',
     'buy_pct_normal': 0.30, 'buy_pct_strong': 0.50,
     'sell_pct_normal': 0.05, 'sell_pct_strong': 0.15},
]


# ============================================================
# 백테스트 엔진
# ============================================================

def run_backtest(df, events, strat):
    """가용자금 비율 기반 백테스트 (매월 $100 적립)"""

    sorted_events = sorted(events, key=lambda e: e['peak_idx'])
    event_by_idx = {}
    for ev in sorted_events:
        event_by_idx.setdefault(ev['peak_idx'], []).append(ev)

    buy_pct_normal = strat['buy_pct_normal']
    buy_pct_strong = strat['buy_pct_strong']
    sell_pct_normal = strat['sell_pct_normal']
    sell_pct_strong = strat['sell_pct_strong']

    # 포트폴리오 상태
    cash = INITIAL_CASH
    total_deposited = INITIAL_CASH  # 총 입금액 (초기 + 적립)
    positions = []
    trades = []
    n_buys = 0
    n_strong_buys, n_weak_buys = 0, 0
    n_strong_sells, n_weak_sells = 0, 0
    peak_portfolio = INITIAL_CASH
    max_drawdown = 0.0
    last_deposit_month = None  # 월 중복 방지

    for i in range(len(df)):
        price = df['Close'].iloc[i]
        date = df.index[i]

        # 매월 첫 거래일: 가용자금에 $100 추가 (자동매수 아님, 현금으로 대기)
        cur_month = (date.year, date.month)
        if last_deposit_month is None:
            last_deposit_month = cur_month  # 시작월은 이미 INITIAL_CASH 포함
        elif cur_month != last_deposit_month:
            cash += MONTHLY_DEPOSIT          # 가용자금(현금)에 추가
            total_deposited += MONTHLY_DEPOSIT
            last_deposit_month = cur_month

        if i in event_by_idx:
            for ev in event_by_idx[i]:
                pv = ev['peak_val']

                if ev['type'] == 'bottom':
                    is_strong = abs(pv) >= STRONG_BUY_TH
                    buy_pct = buy_pct_strong if is_strong else buy_pct_normal

                    if buy_pct <= 0 or cash <= 1:  # $1 미만이면 스킵
                        continue

                    amt = cash * buy_pct
                    cash -= amt
                    shares = amt / price
                    positions.append({
                        'buy_price': price, 'shares': shares,
                        'buy_date': date, 'buy_idx': i,
                    })
                    n_buys += 1
                    if is_strong:
                        n_strong_buys += 1
                    else:
                        n_weak_buys += 1

                elif ev['type'] == 'top' and positions:
                    is_strong = pv <= STRONG_SELL_TH
                    sell_pct = sell_pct_strong if is_strong else sell_pct_normal

                    if sell_pct <= 0:
                        continue

                    remaining = []
                    for pos in positions:
                        ss = pos['shares'] * sell_pct
                        ks = pos['shares'] * (1 - sell_pct)
                        if ss > 1e-10:
                            revenue = ss * price
                            cost = ss * pos['buy_price']
                            trades.append({
                                'buy_date': pos['buy_date'], 'sell_date': date,
                                'buy_price': pos['buy_price'], 'sell_price': price,
                                'shares_sold': ss, 'cost': cost, 'revenue': revenue,
                                'pnl': revenue - cost,
                                'pnl_pct': (price / pos['buy_price'] - 1) * 100,
                                'hold_days': (date - pos['buy_date']).days,
                            })
                            cash += revenue
                        if ks > 1e-10:
                            remaining.append({**pos, 'shares': ks})
                    positions = remaining

                    if is_strong:
                        n_strong_sells += 1
                    else:
                        n_weak_sells += 1

        # MDD 추적
        holdings_val = sum(p['shares'] * price for p in positions)
        portfolio_val = cash + holdings_val
        if portfolio_val > peak_portfolio:
            peak_portfolio = portfolio_val
        dd = (peak_portfolio - portfolio_val) / peak_portfolio * 100 if peak_portfolio > 0 else 0
        if dd > max_drawdown:
            max_drawdown = dd

    # 최종 평가
    last_price = df['Close'].iloc[-1]
    holdings_val = sum(p['shares'] * last_price for p in positions)
    final_portfolio = cash + holdings_val
    total_pnl = final_portfolio - total_deposited  # 순수익 = 최종 - 총입금
    profit_rate = total_pnl / total_deposited * 100 if total_deposited > 0 else 0

    n_t = len(trades)
    n_w = sum(1 for t in trades if t['pnl'] > 0)
    ad = (df.index[-1] - df.index[0]).days
    ay = ad / 365.25 if ad > 0 else 1.0

    # CAGR: (최종/총입금)^(1/년) - 1 은 적립식에 부적합
    # 대신 XIRR 근사 또는 단순 수익률 사용
    # 여기서는 총수익률과 연환산 수익률 표시
    tr = total_pnl / total_deposited if total_deposited > 0 else 0
    ann_return = ((final_portfolio / total_deposited) ** (1 / ay) - 1) if (ay > 0 and total_deposited > 0 and final_portfolio > 0) else 0

    return {
        'strategy': strat.get('name', ''),
        'initial_cash': INITIAL_CASH,
        'total_deposited': total_deposited,
        'final_cash': cash,
        'final_holdings': holdings_val,
        'final_portfolio': final_portfolio,
        'total_pnl': total_pnl,
        'profit_rate': profit_rate,
        'ann_return': ann_return * 100,
        'actual_years': ay,
        'max_drawdown': max_drawdown,
        'n_buys': n_buys,
        'n_strong_buys': n_strong_buys, 'n_weak_buys': n_weak_buys,
        'n_strong_sells': n_strong_sells, 'n_weak_sells': n_weak_sells,
        'n_closed_trades': n_t,
        'n_open_positions': len(positions),
        'win_rate': (n_w / n_t * 100) if n_t > 0 else 0.0,
        'trades': trades,
    }


# ============================================================
# 분석
# ============================================================

def run_period(tickers, sectors, period, cache_dir):
    start, end = period['start'], period['end']
    strat_results = {s['id']: {} for s in STRATEGIES}
    failed, skipped = [], []

    for ticker in tickers:
        try:
            df = download_data(ticker, start=start, end=end, cache_dir=cache_dir)
            if len(df) < 100:
                skipped.append(ticker)
                continue
            score = calc_v4_score(df, w=V4_WINDOW)
            events = detect_signal_events(score, th=SIGNAL_THRESHOLD, cooldown=COOLDOWN)
            pf = build_price_filter(df, er_q=ER_QUANTILE, atr_q=ATR_QUANTILE, lookback=LOOKBACK)
            filtered = [e for e in events if pf(e['peak_idx'])]

            if sum(1 for e in filtered if e['type'] == 'bottom') == 0:
                for s in STRATEGIES:
                    strat_results[s['id']][ticker] = None
                continue

            for s in STRATEGIES:
                r = run_backtest(df, filtered, s)
                strat_results[s['id']][ticker] = r
        except Exception:
            failed.append(ticker)

    return strat_results, failed, skipped


def aggregate(strat_results):
    sums = []
    for s in STRATEGIES:
        res = strat_results[s['id']]
        has_trades = {t: r for t, r in res.items() if r is not None and r['n_buys'] > 0}
        if not has_trades:
            sums.append({'id': s['id'], 'name': s['name'],
                'tickers': 0, 'avg_deposited': 0, 'avg_final': 0,
                'avg_pnl': 0, 'avg_profit_rate': 0, 'avg_ann_return': 0,
                'actual_years': 0, 'closed_trades': 0, 'win_rate': 0,
                'max_dd': 0, 'avg_cash': 0,
                'n_strong_buys': 0, 'n_weak_buys': 0,
                'n_strong_sells': 0, 'n_weak_sells': 0})
            continue

        n = len(has_trades)
        vals = list(has_trades.values())
        all_t = [t for r in vals for t in r['trades']]
        nc = len(all_t)
        nw = sum(1 for t in all_t if t['pnl'] > 0)

        sums.append({'id': s['id'], 'name': s['name'],
            'tickers': n,
            'avg_deposited': np.mean([r['total_deposited'] for r in vals]),
            'avg_final': np.mean([r['final_portfolio'] for r in vals]),
            'avg_pnl': np.mean([r['total_pnl'] for r in vals]),
            'avg_profit_rate': np.mean([r['profit_rate'] for r in vals]),
            'avg_ann_return': np.mean([r['ann_return'] for r in vals]),
            'actual_years': vals[0]['actual_years'],
            'closed_trades': nc,
            'win_rate': (nw / nc * 100) if nc > 0 else 0,
            'max_dd': np.mean([r['max_drawdown'] for r in vals]),
            'avg_cash': np.mean([r['final_cash'] for r in vals]),
            'n_strong_buys': sum(r['n_strong_buys'] for r in vals),
            'n_weak_buys': sum(r['n_weak_buys'] for r in vals),
            'n_strong_sells': sum(r['n_strong_sells'] for r in vals),
            'n_weak_sells': sum(r['n_weak_sells'] for r in vals),
        })
    return sums


def fp(v, w=9):
    return f"{'+' if v >= 0 else '-'}${abs(v):>{w-2}.1f}"

def fc(v, w=8):
    return f"{'+' if v >= 0 else ''}{v:>{w-1}.2f}%"


# ============================================================
# 메인
# ============================================================

def main():
    config_path = _script_dir.parent / 'config' / 'watchlist.json'
    with open(config_path, encoding='utf-8') as f:
        config = json.load(f)

    tickers = list(config['tickers'].keys()) + config.get('benchmarks', [])
    sectors = {t: config['tickers'][t]['sector'] for t in config['tickers']}
    for b in config.get('benchmarks', []):
        sectors[b] = 'Benchmark'

    cache_dir = str(Path(_project_root) / 'cache')
    W = 160

    print(f"{'='*W}")
    print(f"  가용자금 비율 기반 매매 전략 백테스트")
    print(f"  시작 가용자금: ${INITIAL_CASH:.0f} + 매월 ${MONTHLY_DEPOSIT:.0f} 가용자금 추가입금 (신호 없으면 현금 보유)")
    print(f"  강매수: |peak|>={STRONG_BUY_TH} / 강매도: peak<={STRONG_SELL_TH}")
    print(f"{'='*W}")
    print(f"\n  전략 목록:")
    for s in STRATEGIES:
        print(f"  {s['id']:<5} {s['name']:<38} "
              f"매수: 가용{s['buy_pct_normal']*100:.0f}%/{s['buy_pct_strong']*100:.0f}%  "
              f"매도: 보유{s['sell_pct_normal']*100:.0f}%/{s['sell_pct_strong']*100:.0f}%")

    all_data = {}

    for period in MARKET_PERIODS:
        print(f"\n{'='*W}")
        print(f"  [{period['name']}]  {period['start']} ~ {period['end']}  |  {period['desc']}")
        print(f"{'='*W}")

        sr, failed, skipped = run_period(tickers, sectors, period, cache_dir)
        sums = aggregate(sr)
        all_data[period['name']] = (sr, sums, failed, skipped)

        if skipped:
            print(f"  건너뜀: {', '.join(skipped)}")
        if failed:
            print(f"  실패: {', '.join(failed)}")

        # 전략 요약
        print(f"\n  {'ID':<5} {'전략':<35} {'종목':>3} {'총입금':>8} {'최종$':>9} "
              f"{'순이익':>9} {'수익률':>8} {'연환산':>8} {'MDD':>6} {'승률':>6} "
              f"{'강B':>3} {'약B':>3}")
        print(f"  {'-'*(W-4)}")
        for row in sums:
            if row['tickers'] == 0:
                print(f"  {row['id']:<5} {row['name']:<35} {'(없음)':>3}")
                continue
            print(f"  {row['id']:<5} {row['name']:<35} {row['tickers']:>3} "
                  f"${row['avg_deposited']:>7.0f} ${row['avg_final']:>8.1f} "
                  f"{fp(row['avg_pnl'],9)} "
                  f"{fc(row['avg_profit_rate'])} {fc(row['avg_ann_return'])} "
                  f"{row['max_dd']:>5.1f}% {row['win_rate']:>5.1f}% "
                  f"{row['n_strong_buys']:>3} {row['n_weak_buys']:>3}")

        # 종목 상세
        for s in STRATEGIES:
            results = sr[s['id']]
            valid = {t: r for t, r in results.items()
                     if r is not None and r['n_buys'] > 0}
            if not valid:
                continue

            print(f"\n  --- [{s['id']}] {s['name']} ---")
            print(f"  {'종목':<6} {'섹터':<10} {'BUY':>3} {'OPN':>3} "
                  f"{'총입금':>7} {'최종$':>8} {'현금$':>7} {'순이익':>9} "
                  f"{'수익률':>8} {'연환산':>8} {'MDD':>6} {'승률':>6}")
            print(f"  {'-'*(W-4)}")

            for t in sorted(valid.keys(), key=lambda t: valid[t]['ann_return'], reverse=True):
                r = valid[t]
                print(f"  {t:<6} {sectors.get(t,'?'):<10} "
                      f"{r['n_buys']:>3} {r['n_open_positions']:>3} "
                      f"${r['total_deposited']:>6.0f} ${r['final_portfolio']:>7.1f} "
                      f"${r['final_cash']:>6.1f} "
                      f"{fp(r['total_pnl'],8)} "
                      f"{fc(r['profit_rate'])} {fc(r['ann_return'])} "
                      f"{r['max_drawdown']:>5.1f}% {r['win_rate']:>5.1f}%")

    # ==================================================================
    # 핵심 크로스테이블: 연환산 수익률
    # ==================================================================
    print(f"\n\n{'='*W}")
    print(f"  [핵심] 시장상황 x 전략 평균 연환산 수익률 (%) 크로스테이블")
    print(f"{'='*W}")

    hdr = f"  {'상황':<12}"
    for s in STRATEGIES:
        hdr += f" {s['id']:>6}"
    hdr += f"  {'BEST':>5}"
    print(hdr)
    print(f"  {'-'*(W-4)}")

    period_bests = []
    for period in MARKET_PERIODS:
        _, sums, _, _ = all_data[period['name']]
        row = f"  {period['name']:<12}"
        best_c, best_id = -999, ''
        for si, s in enumerate(STRATEGIES):
            c = sums[si]['avg_ann_return']
            row += f" {c:>5.1f}%"
            if c > best_c:
                best_c, best_id = c, s['id']
        row += f"  {best_id:>5}"
        print(row)
        period_bests.append((period['name'], best_id, best_c))

    # ==================================================================
    # 최종 포트폴리오 크로스테이블
    # ==================================================================
    print(f"\n{'='*W}")
    print(f"  시장상황 x 전략 평균 최종 포트폴리오 ($) / (총입금 $)")
    print(f"{'='*W}")

    hdr = f"  {'상황':<12}"
    for s in STRATEGIES:
        hdr += f" {s['id']:>10}"
    print(hdr)
    print(f"  {'-'*(W-4)}")

    for period in MARKET_PERIODS:
        _, sums, _, _ = all_data[period['name']]
        row = f"  {period['name']:<12}"
        for si in range(len(STRATEGIES)):
            f_val = sums[si]['avg_final']
            d_val = sums[si]['avg_deposited']
            row += f" ${f_val:>5.0f}/{d_val:>3.0f}"
        print(row)

    # ==================================================================
    # MDD 크로스테이블
    # ==================================================================
    print(f"\n{'='*W}")
    print(f"  시장상황 x 전략 평균 MDD (%)")
    print(f"{'='*W}")

    hdr = f"  {'상황':<12}"
    for s in STRATEGIES:
        hdr += f" {s['id']:>6}"
    print(hdr)
    print(f"  {'-'*(W-4)}")

    for period in MARKET_PERIODS:
        _, sums, _, _ = all_data[period['name']]
        row = f"  {period['name']:<12}"
        for si in range(len(STRATEGIES)):
            row += f" {sums[si]['max_dd']:>5.1f}%"
        print(row)

    # ==================================================================
    # 순위
    # ==================================================================
    print(f"\n{'='*W}")
    print(f"  전 기간 평균 연환산 수익률 순위")
    print(f"{'='*W}")

    ranking = []
    for si, s in enumerate(STRATEGIES):
        anns, finals, deps, mdds = [], [], [], []
        for period in MARKET_PERIODS:
            _, sums, _, _ = all_data[period['name']]
            anns.append(sums[si]['avg_ann_return'])
            finals.append(sums[si]['avg_final'])
            deps.append(sums[si]['avg_deposited'])
            mdds.append(sums[si]['max_dd'])
        ranking.append({'id': s['id'], 'name': s['name'],
            'avg_ann': np.mean(anns), 'min_ann': min(anns), 'max_ann': max(anns),
            'std_ann': np.std(anns), 'anns': anns,
            'avg_final': np.mean(finals), 'avg_dep': np.mean(deps),
            'avg_mdd': np.mean(mdds)})
    ranking.sort(key=lambda x: x['avg_ann'], reverse=True)

    print(f"  {'#':>2} {'ID':<5} {'전략':<35} {'평균연환산':>9} {'최소':>7} {'최대':>7} "
          f"{'편차':>6} {'평균$':>7} {'MDD':>6}  시장상황별")
    print(f"  {'-'*(W-4)}")
    for i, r in enumerate(ranking, 1):
        det = " ".join(f"{c:>5.1f}" for c in r['anns'])
        marker = ' <-- 유저안' if r['id'] == 'NEW' else ''
        print(f"  {i:>2} {r['id']:<5} {r['name']:<35} {fc(r['avg_ann'],9)} "
              f"{r['min_ann']:>6.2f}% {r['max_ann']:>6.2f}% "
              f"{r['std_ann']:>5.2f}% ${r['avg_final']:>6.0f} "
              f"{r['avg_mdd']:>5.1f}%  [{det}]{marker}")

    pn = " / ".join(p['name'] for p in MARKET_PERIODS)
    print(f"\n  (순서: {pn})")

    # ==================================================================
    # 최적 & 유저안 비교
    # ==================================================================
    print(f"\n{'='*W}")
    print(f"  시장상황별 최적 전략")
    print(f"{'='*W}")
    for pn, bi, bc in period_bests:
        bn = next((s['name'] for s in STRATEGIES if s['id'] == bi), bi)
        print(f"  {pn:<14} -> {bi:<5} {bn:<35} 연환산 {fc(bc)}")

    # 유저안 vs 1위
    new_idx = next(i for i, s in enumerate(STRATEGIES) if s['id'] == 'NEW')
    print(f"\n{'='*W}")
    print(f"  유저안(NEW) vs 각 전략 연환산 수익률 차이 (%p)")
    print(f"{'='*W}")
    hdr = f"  {'상황':<14}"
    for s in STRATEGIES:
        if s['id'] == 'NEW':
            hdr += f" {'NEW':>6}"
        else:
            hdr += f" {s['id']:>6}"
    print(hdr)
    print(f"  {'-'*(W-4)}")

    for period in MARKET_PERIODS:
        _, sums, _, _ = all_data[period['name']]
        nc = sums[new_idx]['avg_ann_return']
        row = f"  {period['name']:<14}"
        for si, s in enumerate(STRATEGIES):
            if s['id'] == 'NEW':
                row += f" {nc:>5.1f}%"
            else:
                d = nc - sums[si]['avg_ann_return']
                row += f" {'+' if d>=0 else ''}{d:>4.1f}p"
        print(row)

    print(f"\n\n  백테스트 완료!")


if __name__ == '__main__':
    main()
