"""매수 신호 차트 생성 (Telegram 전송용)"""
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import pandas as pd


# 한글 폰트 설정 (Windows: Malgun Gothic)
def _setup_font():
    for name in ['Malgun Gothic', 'NanumGothic', 'AppleGothic']:
        matches = fm.findSystemFonts(fontpaths=None)
        for fp in matches:
            try:
                prop = fm.FontProperties(fname=fp)
                if name.lower() in prop.get_name().lower():
                    plt.rcParams['font.family'] = prop.get_name()
                    plt.rcParams['axes.unicode_minus'] = False
                    return
            except Exception:
                continue
    # fallback: 영어만 사용
    plt.rcParams['axes.unicode_minus'] = False

_setup_font()


# 다크 테마 색상
BG = '#0d1117'
SURFACE = '#161b22'
GRID = '#21262d'
TEXT = '#c9d1d9'
TEXT_DIM = '#8b949e'
BLUE = '#58a6ff'
GREEN = '#3fb950'
RED = '#f85149'
ORANGE = '#f0883e'
PURPLE = '#a371f7'


def generate_signal_chart(ticker, df, subind, ev):
    """매수 신호 차트 PNG를 BytesIO로 반환.

    Args:
        ticker: 종목 코드
        df: OHLCV DataFrame (DatetimeIndex)
        subind: DataFrame (s_force, s_div, score)
        ev: 신호 이벤트 dict (peak_idx, dd_pct, peak_val 등)

    Returns:
        io.BytesIO containing PNG image
    """
    peak_idx = ev['peak_idx']
    dd_pct = ev.get('dd_pct', 0)

    # 최근 60거래일 범위
    start = max(0, peak_idx - 55)
    end = min(len(df), peak_idx + 5)
    slc = slice(start, end)

    dates = df.index[slc]
    close = df['Close'].values[slc]
    score = subind['score'].values[slc]
    s_force = subind['s_force'].values[slc]
    s_div = subind['s_div'].values[slc]

    # 20일 롤링 고점 (DD Gate 기준선)
    rolling_high = pd.Series(df['Close'].values[slc]).rolling(20, min_periods=1).max().values

    # 신호 위치 (슬라이스 내 인덱스)
    sig_local = peak_idx - start
    sig_date = dates[sig_local] if sig_local < len(dates) else dates[-1]
    sig_price = close[sig_local] if sig_local < len(close) else close[-1]
    sig_score = score[sig_local] if sig_local < len(score) else 0

    # 신호 구간 (start_idx ~ end_idx)
    ev_start_local = max(0, ev.get('start_idx', peak_idx) - start)
    ev_end_local = min(len(dates) - 1, ev.get('end_idx', peak_idx) - start)

    # ── Figure 설정 ──
    fig = plt.figure(figsize=(10, 6), facecolor=BG)
    gs = gridspec.GridSpec(2, 1, height_ratios=[7, 3], hspace=0.08)

    # ════════════════════════════════════════════
    # 상단: 가격 + 20d 고점선 + 매수 신호
    # ════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(BG)

    # 종가 라인
    ax1.plot(dates, close, color=BLUE, linewidth=1.5, label='종가', zorder=3)

    # 20일 고점선
    ax1.plot(dates, rolling_high, color=RED, linewidth=0.8,
             linestyle='--', alpha=0.6, label='20일 고점')

    # DD 구간 음영 (고점 ~ 종가 사이)
    ax1.fill_between(dates, close, rolling_high,
                     where=(rolling_high > close),
                     color=RED, alpha=0.07, label=f'DD 구간')

    # 매수 신호 마커
    ax1.scatter([sig_date], [sig_price], marker='^', s=200,
                color=GREEN, edgecolors='white', linewidths=0.8,
                zorder=5, label=f'BUY ${sig_price:.2f}')

    # 신호 구간 배경
    if ev_start_local < ev_end_local and ev_end_local < len(dates):
        ax1.axvspan(dates[ev_start_local], dates[ev_end_local],
                    color=GREEN, alpha=0.05)

    # 가격 어노테이션
    ax1.annotate(f'${sig_price:.2f}\nDD {dd_pct:.1%}',
                 xy=(sig_date, sig_price),
                 xytext=(15, -25), textcoords='offset points',
                 fontsize=9, color=GREEN, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.2))

    # 스타일
    ax1.set_ylabel('Price ($)', fontsize=10, color=TEXT_DIM)
    ax1.tick_params(colors=TEXT_DIM, labelsize=9)
    ax1.grid(True, color=GRID, alpha=0.5, linewidth=0.5)
    ax1.legend(loc='upper left', fontsize=8, facecolor=SURFACE,
               edgecolor=GRID, labelcolor=TEXT)
    ax1.set_xlim(dates[0], dates[-1])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_color(GRID)
    ax1.spines['left'].set_color(GRID)
    ax1.set_xticklabels([])

    # 타이틀
    peak_date_str = ev.get('peak_date', df.index[peak_idx].strftime('%Y-%m-%d'))
    ax1.set_title(f'{ticker}  BUY Signal  {peak_date_str}',
                  fontsize=14, fontweight='bold', color=TEXT,
                  loc='left', pad=12)

    # ════════════════════════════════════════════
    # 하단: V4 Score + S_Force / S_Div
    # ════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_facecolor(BG)

    ax2.plot(dates, score, color=BLUE, linewidth=1.5, label='Score')
    ax2.plot(dates, s_force, color=ORANGE, linewidth=0.9, alpha=0.7, label='S_Force')
    ax2.plot(dates, s_div, color=PURPLE, linewidth=0.9, alpha=0.7, label='S_Div')

    # Threshold 점선
    ax2.axhline(0.025, color=TEXT_DIM, linewidth=0.7, linestyle=':', alpha=0.5)
    ax2.axhline(0, color=GRID, linewidth=0.5)

    # 신호 지점 마커
    ax2.scatter([sig_date], [sig_score], marker='o', s=60,
                color=GREEN, edgecolors='white', linewidths=0.6, zorder=5)

    # 스코어 값 어노테이션
    ax2.annotate(f'{sig_score:+.3f}',
                 xy=(sig_date, sig_score),
                 xytext=(10, 8), textcoords='offset points',
                 fontsize=8, color=GREEN, fontweight='bold')

    # 스타일
    ax2.set_ylabel('Score', fontsize=10, color=TEXT_DIM)
    ax2.tick_params(colors=TEXT_DIM, labelsize=9)
    ax2.grid(True, color=GRID, alpha=0.5, linewidth=0.5)
    ax2.legend(loc='upper left', fontsize=8, facecolor=SURFACE,
               edgecolor=GRID, labelcolor=TEXT, ncol=3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_color(GRID)
    ax2.spines['left'].set_color(GRID)

    # X축 날짜 포맷
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax2.get_xticklabels(), rotation=0, ha='center')

    # ── PNG 출력 ──
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf
