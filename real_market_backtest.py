"""
Volume Indicator Ecosystem — Real Market Backtest
==================================================
SPY & QQQ (2000-01-01 ~ 2025-12-31)

실행 방법:
  pip install yfinance pandas numpy matplotlib
  python real_market_backtest.py

포함된 Ground Truth 전환점:
  - 닷컴 버블 붕괴 (2000-2002)
  - 2003 회복
  - 2007 금융위기 전조
  - 2008 리먼 사태
  - 2009 바닥 및 회복
  - 2011 유럽 위기
  - 2015-2016 중국 쇼크
  - 2018 금리 충격
  - 2020 코로나 크래시 & V-Recovery
  - 2022 인플레이션/금리 하락
  - 2023 AI 랠리 시작

아키텍처 3종 비교: V3 (분리형) vs V4 (융합형) vs V4+ (하이브리드)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')
import os
from datetime import datetime

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. DATA DOWNLOAD
# ============================================================

def download_data(ticker, start='2000-01-01', end='2025-12-31', cache_dir='./cache'):
    """yfinance로 데이터 다운로드 + 로컬 캐시"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'{ticker}_{start}_{end}.csv')
    
    if os.path.exists(cache_file):
        print(f"  Loading cached {ticker} data...")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    else:
        print(f"  Downloading {ticker} from yfinance...")
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, auto_adjust=True)
        # yfinance 최신 버전 호환: MultiIndex 처리
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.to_csv(cache_file)
        print(f"  Saved to {cache_file}")
    
    # 컬럼 정규화
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if 'open' in cl: col_map[c] = 'Open'
        elif 'high' in cl: col_map[c] = 'High'
        elif 'low' in cl: col_map[c] = 'Low'
        elif 'close' in cl: col_map[c] = 'Close'
        elif 'volume' in cl: col_map[c] = 'Volume'
    df = df.rename(columns=col_map)
    
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    for r in required:
        if r not in df.columns:
            raise ValueError(f"Missing column: {r}. Available: {list(df.columns)}")
    
    df = df[required].dropna()
    df['Volume'] = df['Volume'].astype(float)
    
    # Volume이 0인 날 제거 (휴장일 등)
    df = df[df['Volume'] > 0]
    
    print(f"  {ticker}: {len(df)} trading days ({df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')})")
    return df


# ============================================================
# 2. GROUND TRUTH — 실제 시장 전환점
# ============================================================

def get_transition_points(ticker):
    """
    역사적으로 알려진 주요 전환점.
    날짜는 '사후적으로 확인된' 고점/저점 근처.
    실제 시장에서는 정확한 날을 아는 것이 불가능하므로,
    ±10 거래일 윈도우 내 감지를 성공으로 판정.
    """
    if ticker == 'SPY':
        return [
            # (날짜, 타입, 이벤트 설명)
            ('2000-03-24', 'top',    'Dot-com Bubble Peak'),
            ('2002-10-09', 'bottom', 'Dot-com Bottom'),
            ('2007-10-09', 'top',    'Pre-GFC Peak'),
            ('2009-03-09', 'bottom', 'GFC Bottom (Lehman)'),
            ('2011-04-29', 'top',    'Pre-Euro Crisis'),
            ('2011-10-03', 'bottom', 'Euro Crisis Bottom'),
            ('2015-05-21', 'top',    'Pre-China Shock'),
            ('2016-02-11', 'bottom', 'China Shock Bottom'),
            ('2018-09-20', 'top',    'Pre-Rate Shock'),
            ('2018-12-24', 'bottom', 'Rate Shock Bottom (Xmas Eve)'),
            ('2020-02-19', 'top',    'Pre-COVID Peak'),
            ('2020-03-23', 'bottom', 'COVID Bottom'),
            ('2022-01-03', 'top',    'Pre-Inflation Bear'),
            ('2022-10-12', 'bottom', 'Inflation Bear Bottom'),
            ('2025-02-19', 'top',    'Tariff Shock Peak'),
        ]
    elif ticker == 'QQQ':
        return [
            ('2000-03-27', 'top',    'Dot-com Bubble Peak (Tech)'),
            ('2002-10-07', 'bottom', 'Dot-com Bottom (Tech)'),
            ('2007-10-31', 'top',    'Pre-GFC Peak (Tech)'),
            ('2009-03-09', 'bottom', 'GFC Bottom (Tech)'),
            ('2011-04-28', 'top',    'Pre-Euro Crisis (Tech)'),
            ('2011-10-03', 'bottom', 'Euro Crisis Bottom (Tech)'),
            ('2015-07-20', 'top',    'Pre-China Shock (Tech)'),
            ('2016-02-11', 'bottom', 'China Shock Bottom (Tech)'),
            ('2018-10-01', 'top',    'Pre-Rate Shock (Tech)'),
            ('2018-12-24', 'bottom', 'Rate Shock Bottom (Tech)'),
            ('2020-02-19', 'top',    'Pre-COVID Peak (Tech)'),
            ('2020-03-23', 'bottom', 'COVID Bottom (Tech)'),
            ('2021-11-19', 'top',    'Pre-Tech Bear Peak'),
            ('2022-10-13', 'bottom', 'Tech Bear Bottom'),
            ('2025-02-19', 'top',    'Tariff Shock Peak (Tech)'),
        ]
    else:
        return []

def match_transitions_to_index(transitions, df):
    """날짜 문자열을 DataFrame 인덱스 위치로 변환"""
    matched = []
    for date_str, tp_type, desc in transitions:
        target = pd.Timestamp(date_str)
        
        # 정확한 날짜가 없을 수 있으므로 가장 가까운 거래일 찾기
        idx_loc = df.index.searchsorted(target)
        idx_loc = min(idx_loc, len(df) - 1)
        
        # 인덱스가 유효한 범위 내에 있는지 확인
        if 0 <= idx_loc < len(df):
            matched.append((idx_loc, tp_type, desc, df.index[idx_loc].strftime('%Y-%m-%d')))
    
    return matched


# ============================================================
# 3. ALL INDICATORS
# ============================================================

def calc_obv(df):
    return (np.sign(df['Close'].diff()).fillna(0) * df['Volume']).cumsum().rename('OBV')

def calc_mfi(df, period=14):
    tp = (df['High']+df['Low']+df['Close'])/3
    raw = tp*df['Volume']; d = tp.diff()
    pos = (raw*(d>0).astype(float)).rolling(period).sum()
    neg = (raw*(d<0).astype(float)).rolling(period).sum()
    return (100-100/(1+pos/neg.replace(0,np.nan))).fillna(50).rename('MFI')

def calc_vr(df, period=20):
    d=df['Close'].diff()
    up=(df['Volume']*(d>0).astype(float)).rolling(period).sum()
    dn=(df['Volume']*(d<0).astype(float)).rolling(period).sum()
    un=(df['Volume']*(d==0).astype(float)).rolling(period).sum()
    return (((up+0.5*un)/(dn+0.5*un).replace(0,np.nan))*100).fillna(150).rename('VR')

def calc_cmf(df, period=20):
    mfm=((df['Close']-df['Low'])-(df['High']-df['Close']))/(df['High']-df['Low']).replace(0,np.nan)
    return (mfm.fillna(0)*df['Volume']).rolling(period).sum().div(df['Volume'].rolling(period).sum()).fillna(0).rename('CMF')

def calc_pvi_nvi(df):
    pvi,nvi=[1000.0],[1000.0]; pct=df['Close'].pct_change().fillna(0)
    vup=df['Volume']>df['Volume'].shift(1)
    for i in range(1,len(df)):
        if vup.iloc[i]: pvi.append(pvi[-1]*(1+pct.iloc[i])); nvi.append(nvi[-1])
        else: pvi.append(pvi[-1]); nvi.append(nvi[-1]*(1+pct.iloc[i]))
    return pd.Series(pvi,index=df.index,name='PVI'), pd.Series(nvi,index=df.index,name='NVI')

def calc_vrsi(df, period=14):
    d=df['Volume'].diff(); g=d.where(d>0,0.0).ewm(span=period,adjust=False).mean()
    l=(-d).where(d<0,0.0).ewm(span=period,adjust=False).mean()
    return (100-100/(1+g/l.replace(0,np.nan))).fillna(50).rename('V_RSI')

def calc_vsto(df, period=14, smooth=3):
    v=df['Volume']; lo=v.rolling(period).min(); hi=v.rolling(period).max()
    k=((v-lo)/(hi-lo).replace(0,np.nan)*100).fillna(50)
    return k.rename('V_Sto_K'), k.rolling(smooth).mean().rename('V_Sto_D')

def calc_vmacd(df, fast=12, slow=26, signal=9):
    v=df['Volume']
    vm=v.ewm(span=fast,adjust=False).mean()-v.ewm(span=slow,adjust=False).mean()
    vs=vm.ewm(span=signal,adjust=False).mean()
    return vm.rename('V_MACD'), vs.rename('V_Signal'), (vm-vs).rename('V_Histogram')

def calc_vbb(df, period=20, ns=2):
    v=df['Volume']; m=v.rolling(period).mean(); s=v.rolling(period).std()
    u,l=m+ns*s,m-ns*s
    return ((v-l)/(u-l).replace(0,np.nan)).fillna(0.5).rename('V_BB_PctB'), \
           ((u-l)/m.replace(0,np.nan)).fillna(0).rename('V_BandWidth')

def calc_vcci(df, period=20):
    v=df['Volume']; s=v.rolling(period).mean()
    mad=v.rolling(period).apply(lambda x:np.mean(np.abs(x-np.mean(x))),raw=True)
    return ((v-s)/(0.015*mad).replace(0,np.nan)).fillna(0).rename('V_CCI')

def calc_vadx(df, period=14):
    vc=df['Volume'].diff().fillna(0)
    vu=vc.where(vc>0,0.0).ewm(span=period,adjust=False).mean()
    vd=(-vc).where(vc<0,0.0).ewm(span=period,adjust=False).mean()
    va=vc.abs().ewm(span=period,adjust=False).mean().replace(0,np.nan)
    pdi,ndi=vu/va,vd/va; dx=((pdi-ndi).abs()/(pdi+ndi).replace(0,np.nan)*100)
    return dx.ewm(span=period,adjust=False).mean().fillna(0).rename('V_ADX')


# ============================================================
# 4. V3 ARCHITECTURE (분리형)
# ============================================================

def calc_v3_score(df, w=20):
    n=len(df)
    obv=calc_obv(df); mfi=calc_mfi(df); vr=calc_vr(df); cmf=calc_cmf(df)
    pvi,nvi=calc_pvi_nvi(df); vrsi=calc_vrsi(df)
    vsk,_=calc_vsto(df); _,_,vhist=calc_vmacd(df)
    vbb,vbw=calc_vbb(df); vcci=calc_vcci(df); vadx=calc_vadx(df)
    
    scores=np.zeros(n)
    for i in range(max(60,w),n):
        # Cluster A
        obv_r=(obv.iloc[i]-obv.iloc[i-w])/(abs(obv.iloc[i-w])+1e-10)
        a_raw=np.mean([np.clip(obv_r*10,-1,1),(mfi.iloc[i]-50)/50,np.clip((vr.iloc[i]-150)/150,-1,1)])
        # Cluster B
        b_raw=np.clip(cmf.iloc[i]*2,-1,1)
        # Cluster C
        pr=(pvi.iloc[i]-pvi.iloc[i-w])/(abs(pvi.iloc[i-w])+1e-10)
        nr=(nvi.iloc[i]-nvi.iloc[i-w])/(abs(nvi.iloc[i-w])+1e-10)
        c_raw=np.clip((nr-pr)*50,-1,1)
        
        # D → A* health check
        vrsi_d=(vrsi.iloc[i]-50)/50; conc=vrsi_d*a_raw
        dh=1.0+0.5*min(abs(conc),1.0) if conc>0 else 1.0-0.5*min(abs(conc),1.0)
        # D → B* participation
        hr=abs(vhist.iloc[max(0,i-w):i]).mean()+1e-10
        vmacd_d=np.clip(vhist.iloc[i]/hr,-1,1); bc2=vmacd_d*b_raw
        dp=1.0+0.4*min(abs(bc2),1.0) if bc2>0 else 1.0-0.4*min(abs(bc2),1.0)
        # D → C* conviction
        vsn=(vsk.iloc[i]-50)/50; cl=c_raw*(-vsn)
        dc=1.0+0.6*min(abs(cl),1.0) if cl>0 else 1.0-0.3*min(abs(cl),1.0)
        
        a_s=a_raw*dh; b_s=b_raw*dp; c_s=c_raw*dc
        
        # E amplifier
        sp=1.0+0.5*max(abs(vbb.iloc[i]-0.5)*2,min(abs(vcci.iloc[i])/200,1.0))
        tr=1.0+0.4*min(vadx.iloc[i]/50,1.0)
        bww=vbw.iloc[max(0,i-60):i]
        sq=1.3 if(len(bww)>10 and (bww<vbw.iloc[i]).mean()<0.15) else 1.0
        e_a=sp*tr*sq
        
        dire=0.45*a_s+0.30*b_s+0.25*c_s
        act=sum([abs(a_s)>0.1,abs(b_s)>0.1,abs(c_s)>0.1])
        mm={0:0.5,1:1.0,2:1.5,3:2.2}
        scores[i]=dire*mm.get(act,1.0)*e_a
    
    return pd.Series(scores,index=df.index,name='V3')


# ============================================================
# 5. V4 ARCHITECTURE (융합형)
# ============================================================

def calc_pv_rsi(df, period=14):
    p_ret=df['Close'].pct_change().fillna(0)
    v_ratio=df['Volume']/df['Volume'].rolling(20).mean().replace(0,np.nan).fillna(df['Volume'])
    pv_force=p_ret*v_ratio
    delta=pv_force.diff()
    g=delta.where(delta>0,0.0).ewm(span=period,adjust=False).mean()
    l=(-delta).where(delta<0,0.0).ewm(span=period,adjust=False).mean()
    return (100-100/(1+g/l.replace(0,np.nan))).fillna(50).rename('PV_RSI')

def calc_pv_divergence(df, period=20):
    p_mom=df['Close'].pct_change(period).fillna(0)
    v_mom=df['Volume'].pct_change(period).fillna(0)
    p_std=p_mom.rolling(period*2).std().replace(0,np.nan).fillna(1)
    v_std=v_mom.rolling(period*2).std().replace(0,np.nan).fillna(1)
    return ((v_mom/v_std)-(p_mom/p_std)).rename('PV_DivIdx')

def calc_pv_concordance(df, period=20):
    p=df['Close'].pct_change().fillna(0)
    v=df['Volume'].pct_change().fillna(0)
    return p.rolling(period).corr(v).fillna(0).rename('PV_Concordance')

def calc_pv_force_macd(df, fast=12, slow=26, signal=9):
    p_vel=df['Close'].pct_change().fillna(0); p_acc=p_vel.diff().fillna(0)
    v_norm=df['Volume']/df['Volume'].rolling(20).mean().replace(0,np.nan).fillna(df['Volume'])
    force=v_norm*p_acc
    fm=force.ewm(span=fast,adjust=False).mean()-force.ewm(span=slow,adjust=False).mean()
    fs=fm.ewm(span=signal,adjust=False).mean()
    return (fm-fs).rename('PV_Force_Hist')

def calc_pv_squeeze(df, period=20, ns=2):
    pm=df['Close'].rolling(period).mean(); ps=df['Close'].rolling(period).std()
    pbw=(2*ns*ps/pm.replace(0,np.nan)).fillna(0)
    vm=df['Volume'].rolling(period).mean(); vs=df['Volume'].rolling(period).std()
    vbw=(2*ns*vs/vm.replace(0,np.nan)).fillna(0)
    pp=pbw.rolling(60).rank(pct=True).fillna(0.5)
    vp=vbw.rolling(60).rank(pct=True).fillna(0.5)
    dbl=((pp<0.20)&(vp<0.20)).astype(float)
    dvg=((pp<0.20)&(vp>0.80))|((pp>0.80)&(vp<0.20))
    return dbl.rename('PV_DblSqueeze'), dvg.astype(float).rename('PV_Divergent')

# ============================================================
# 5B. V4_wP: Price Filter (ER + ATR rolling percentile)
# ============================================================

def calc_efficiency_ratio(df, period=20):
    """Price Efficiency Ratio: direction / path (0~1)"""
    direction = (df['Close'] - df['Close'].shift(period)).abs()
    path = df['Close'].diff().abs().rolling(period).sum()
    er = (direction / path.replace(0, np.nan)).fillna(0).clip(0, 1)
    return er.rename('ER')


def calc_atr_percentile(df, atr_period=14, rank_period=252):
    """ATR Percentile Rank (0~1)"""
    high = df['High']; low = df['Low']; close = df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_period, adjust=False).mean()
    atr_pct = atr.rolling(rank_period).rank(pct=True).fillna(0.5)
    return atr_pct.rename('ATR_pct')


def build_price_filter(df, er_q=66, atr_q=66, lookback=252):
    """
    V4_wP 가격 필터 구축: rolling percentile 기반.
    Returns: function(peak_idx) -> bool (True = 통과)
    """
    er = calc_efficiency_ratio(df, 20).values
    atr = calc_atr_percentile(df, 14, 252).values
    n = len(df)

    # rolling percentile 미리 계산
    er_thresh = np.full(n, np.nan)
    atr_thresh = np.full(n, np.nan)
    for i in range(lookback, n):
        lb = max(0, i - lookback)
        er_window = er[lb:i]
        atr_window = atr[lb:i]
        if len(er_window) >= 60:
            er_thresh[i] = np.percentile(er_window, er_q)
            atr_thresh[i] = np.percentile(atr_window, atr_q)

    def price_filter(peak_idx):
        if peak_idx >= n or np.isnan(er_thresh[peak_idx]) or np.isnan(atr_thresh[peak_idx]):
            return False
        return er[peak_idx] < er_thresh[peak_idx] and atr[peak_idx] > atr_thresh[peak_idx]

    return price_filter


def smooth_earnings_volume(df, ticker=None, spike_mult=3.0, buffer_days=1, median_window=20):
    """실적발표일 전후 거래량 스무딩 (single-day 스파이크 제거).

    2단계 감지:
      1) yfinance earnings_dates (최근 ~2.5년)
      2) 거래량 > spike_mult × 20일 중앙값 (오래된 데이터 보완)
    감지된 날짜 ±buffer_days의 Volume을 rolling median으로 교체.
    원본 df를 수정하지 않고 복사본 반환.
    """
    df = df.copy()
    vol = df['Volume'].copy()
    rolling_med = vol.rolling(median_window, min_periods=5).median()

    earnings_mask = pd.Series(False, index=df.index)

    # Tier 1: yfinance 실적발표일
    if ticker:
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            edates = t.earnings_dates
            if edates is not None and len(edates) > 0:
                for ed in edates.index:
                    ed_naive = ed.tz_localize(None) if hasattr(ed, 'tz_localize') and ed.tzinfo else ed
                    for offset in range(-buffer_days, buffer_days + 1):
                        target = ed_naive + pd.Timedelta(days=offset)
                        if target in df.index:
                            earnings_mask[target] = True
        except Exception:
            pass

    # Tier 2: 거래량 스파이크 감지 (Tier 1 미커버 구간)
    spike_threshold = rolling_med * spike_mult
    spikes = (vol > spike_threshold) & (~earnings_mask) & (rolling_med > 0)
    for spike_idx in df.index[spikes]:
        for offset in range(-buffer_days, buffer_days + 1):
            target = spike_idx + pd.Timedelta(days=offset)
            if target in df.index:
                earnings_mask[target] = True

    # 감지된 날: Volume → rolling median으로 교체
    mask_valid = earnings_mask & rolling_med.notna()
    df.loc[mask_valid, 'Volume'] = rolling_med[mask_valid]

    return df


def calc_v4_score(df, w=20, divgate_days=3):
    """VN60+GEO-OP: 2지표(force/div) AND-GEO 결합.
    score = sqrt(S_Force × S_Div) when both > 0 (buy territory)
    score = -sqrt(|S_Force| × |S_Div|) when both < 0 (sell territory)
    score = 0 otherwise (mixed → no signal)
    DivGate: S_Div가 divgate_days일 연속 같은 부호여야 활성화."""
    subind = calc_v4_subindicators(df, w=w, divgate_days=divgate_days)
    return subind['score'].rename('V4')


def calc_v4_subindicators(df, w=20, divgate_days=3):
    """VN60+GEO-OP 내부 지표값을 매 bar마다 계산하여 DataFrame으로 반환.
    AND-GEO 결합: sqrt(S_Force × S_Div) when both same sign, else 0."""
    n = len(df)
    pv_div = calc_pv_divergence(df, w)
    pv_fh = calc_pv_force_macd(df)

    # DivGate: 연속 같은 부호 일수
    raw_div = np.array([np.clip(pv_div.iloc[i] / 3, -1, 1) for i in range(n)])
    consec = np.ones(n)
    for i in range(1, n):
        if raw_div[i] > 0 and raw_div[i-1] > 0:
            consec[i] = consec[i-1] + 1
        elif raw_div[i] < 0 and raw_div[i-1] < 0:
            consec[i] = consec[i-1] + 1
        else:
            consec[i] = 1

    arr_force = np.zeros(n)
    arr_div = np.zeros(n)
    arr_div_raw = np.zeros(n)
    arr_score = np.zeros(n)

    for i in range(max(60, w), n):
        s_div_raw = raw_div[i]
        s_div = s_div_raw if consec[i] >= divgate_days else 0.0
        fhr_std = pv_fh.iloc[max(0, i - w):i].std() + 1e-10
        s_force = np.clip(pv_fh.iloc[i] / (2 * fhr_std), -1, 1)

        # AND-GEO combination
        if s_force > 0 and s_div > 0:
            score = np.sqrt(s_force * s_div)
        elif s_force < 0 and s_div < 0:
            score = -np.sqrt(abs(s_force) * abs(s_div))
        else:
            score = 0.0

        arr_force[i] = s_force
        arr_div[i] = s_div
        arr_div_raw[i] = s_div_raw
        arr_score[i] = score

    return pd.DataFrame({
        's_force': arr_force,
        's_div': arr_div, 's_div_raw': arr_div_raw,
        's_conc': np.zeros(n),  # backward compat (GEO-OP에서 미사용)
        'act': np.zeros(n),     # backward compat (GEO-OP에서 미사용)
        'score': arr_score,
    }, index=df.index)


# ============================================================
# 6. V4+ HYBRID
# ============================================================

def calc_v4p_score(v3, v4):
    n=len(v3); scores=np.zeros(n)
    for i in range(60,n):
        a=v3.iloc[i]*v4.iloc[i]  # agreement
        avg=(v3.iloc[i]+v4.iloc[i])/2
        if a>0: scores[i]=avg*(1.0+0.5*min(abs(a),1.0))
        else: scores[i]=avg*0.7
    return pd.Series(scores,index=v3.index,name='V4+')


# ============================================================
# 7. EVALUATION
# ============================================================

def evaluate(score, transitions, df, lead_w=40, lag_w=15, th=0.15):
    """실제 시장은 합성 데이터보다 노이즈가 크므로 윈도우를 넓힘"""
    rows = []
    for tp_idx, tp_type, desc, actual_date in transitions:
        if tp_idx < 80 or tp_idx >= len(df)-15:
            continue
        
        s = max(0, tp_idx - lead_w)
        e = min(len(df), tp_idx + lag_w)
        w = score.iloc[s:e]
        
        if tp_type == 'top':
            ext = w.min(); idx = w.idxmin()
            det = ext < -th
        else:
            ext = w.max(); idx = w.idxmax()
            chg = w.diff(5).max()
            chg_idx = w.diff(5).idxmax()
            det = ext > th * 0.5 or (chg > th if not pd.isna(chg) else False)
            if chg > th and not pd.isna(chg_idx):
                idx = chg_idx
        
        # lead_days는 인덱스 위치 차이
        det_pos = df.index.get_loc(idx) if det else tp_idx
        lead = tp_idx - det_pos
        
        rows.append({
            'tp_idx': tp_idx, 'type': tp_type, 'desc': desc,
            'actual_date': actual_date,
            'detected': det, 'lead': lead,
            'strength': abs(ext), 'det_idx': det_pos,
            'det_date': df.index[det_pos].strftime('%Y-%m-%d') if det else '-',
        })
    
    return pd.DataFrame(rows)


# ============================================================
# 7B. FALSE POSITIVE ANALYSIS
# ============================================================

def detect_signal_events(score, th=0.15, cooldown=5):
    """
    스코어 시계열에서 모든 신호 이벤트를 탐지.
    - top_signal: score < -th 연속 구간 (매도 경고)
    - bottom_signal: score > th*0.5 연속 구간 (매수 기회)
    cooldown: 이벤트 종료 후 N일 내 재발생은 같은 이벤트로 병합

    Returns: list of dict {type, start_idx, end_idx, peak_idx, peak_val, start_val, duration}
    """
    arr = score.values
    n = len(arr)
    events = []

    # --- Top signals (score < -th) ---
    i = 0
    while i < n:
        if not np.isnan(arr[i]) and arr[i] < -th:
            start = i
            peak_val = arr[i]
            peak_idx = i
            # 연속 구간 + 쿨다운 병합
            while i < n:
                if not np.isnan(arr[i]) and arr[i] < -th:
                    if arr[i] < peak_val:
                        peak_val = arr[i]
                        peak_idx = i
                    end = i
                    i += 1
                else:
                    # 쿨다운 구간 확인
                    gap = 0
                    while gap < cooldown and i + gap < n:
                        if not np.isnan(arr[i + gap]) and arr[i + gap] < -th:
                            break
                        gap += 1
                    if gap < cooldown and i + gap < n:
                        # 쿨다운 내 재발생 → 계속
                        i += gap
                    else:
                        break
            events.append({
                'type': 'top', 'start_idx': start, 'end_idx': end,
                'peak_idx': peak_idx, 'peak_val': peak_val,
                'start_val': arr[start],
                'duration': end - start + 1,
            })
        else:
            i += 1

    # --- Bottom signals (score > th*0.5) ---
    bot_th = th * 0.5
    i = 0
    while i < n:
        if not np.isnan(arr[i]) and arr[i] > bot_th:
            start = i
            peak_val = arr[i]
            peak_idx = i
            while i < n:
                if not np.isnan(arr[i]) and arr[i] > bot_th:
                    if arr[i] > peak_val:
                        peak_val = arr[i]
                        peak_idx = i
                    end = i
                    i += 1
                else:
                    gap = 0
                    while gap < cooldown and i + gap < n:
                        if not np.isnan(arr[i + gap]) and arr[i + gap] > bot_th:
                            break
                        gap += 1
                    if gap < cooldown and i + gap < n:
                        i += gap
                    else:
                        break
            events.append({
                'type': 'bottom', 'start_idx': start, 'end_idx': end,
                'peak_idx': peak_idx, 'peak_val': peak_val,
                'start_val': arr[start],
                'duration': end - start + 1,
            })
        else:
            i += 1

    return sorted(events, key=lambda e: e['start_idx'])


def classify_signals(events, transitions, df, lead_w=40, lag_w=15):
    """
    신호 이벤트를 TP / FP 로 분류.
    - TP: 이벤트의 peak_idx가 전환점의 [tp-lead_w, tp+lag_w] 윈도우 안에 있고 타입도 일치
    - FP: 어떤 전환점 윈도우에도 해당하지 않음

    Returns: dict with keys:
      - events: list of events with 'label' field added ('TP' or 'FP')
      - tp_count: {top: N, bottom: N} 전환점 수
      - tp_matched: {top: set, bottom: set} 매칭된 전환점 인덱스들
    """
    # 전환점별 윈도우 구성
    tp_windows = []
    for tp_idx, tp_type, desc, actual_date in transitions:
        if tp_idx < 80 or tp_idx >= len(df) - 15:
            continue
        tp_windows.append({
            'tp_idx': tp_idx, 'type': tp_type,
            'start': max(0, tp_idx - lead_w),
            'end': min(len(df), tp_idx + lag_w),
        })

    tp_matched = {'top': set(), 'bottom': set()}
    tp_count = {'top': 0, 'bottom': 0}
    for w in tp_windows:
        tp_count[w['type']] += 1

    classified_events = []
    for ev in events:
        peak = ev['peak_idx']
        ev_type = ev['type']
        matched = False
        for w in tp_windows:
            if w['type'] == ev_type and w['start'] <= peak <= w['end']:
                matched = True
                tp_matched[ev_type].add(w['tp_idx'])
                break
        ev_copy = dict(ev)
        ev_copy['label'] = 'TP' if matched else 'FP'
        classified_events.append(ev_copy)

    return {
        'events': classified_events,
        'tp_count': tp_count,
        'tp_matched': tp_matched,
    }


def calc_precision_recall(classification, n_trading_days):
    """
    분류 결과에서 Precision, Recall, F1, FP/year 계산.
    top/bottom 별도 + 전체 합산.

    Returns: dict with top/bottom/overall metrics
    """
    events = classification['events']
    tp_count = classification['tp_count']
    tp_matched = classification['tp_matched']

    years = n_trading_days / 252.0
    result = {}

    for sig_type in ['top', 'bottom', 'overall']:
        if sig_type == 'overall':
            tp = sum(1 for e in events if e['label'] == 'TP')
            fp = sum(1 for e in events if e['label'] == 'FP')
            fn = (tp_count['top'] - len(tp_matched['top'])) + \
                 (tp_count['bottom'] - len(tp_matched['bottom']))
        else:
            type_events = [e for e in events if e['type'] == sig_type]
            tp = sum(1 for e in type_events if e['label'] == 'TP')
            fp = sum(1 for e in type_events if e['label'] == 'FP')
            fn = tp_count[sig_type] - len(tp_matched[sig_type])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fp_per_year = fp / years if years > 0 else 0.0

        result[sig_type] = {
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision, 'recall': recall, 'f1': f1,
            'fp_per_year': fp_per_year,
            'total_signals': tp + fp,
        }

    return result


def run_fp_analysis(score, transitions, df, th=0.15, cooldown=5, lead_w=40, lag_w=15,
                    price_filter=None):
    """단일 스코어에 대해 FP 분석 전체 파이프라인 실행"""
    events = detect_signal_events(score, th=th, cooldown=cooldown)
    if price_filter is not None:
        events = [e for e in events if price_filter(e['peak_idx'])]
    classification = classify_signals(events, transitions, df, lead_w=lead_w, lag_w=lag_w)
    metrics = calc_precision_recall(classification, len(df))
    return {
        'events': classification['events'],
        'metrics': metrics,
    }


def run_threshold_sensitivity(score, transitions, df, cooldown=5, lead_w=40, lag_w=15):
    """임계값을 변화시키며 Precision-Recall 트레이드오프 계산"""
    thresholds = np.arange(0.05, 0.55, 0.05)
    results = []
    for th in thresholds:
        fp_result = run_fp_analysis(score, transitions, df, th=th,
                                     cooldown=cooldown, lead_w=lead_w, lag_w=lag_w)
        m = fp_result['metrics']['overall']
        results.append({
            'threshold': th,
            'precision': m['precision'],
            'recall': m['recall'],
            'f1': m['f1'],
            'fp_per_year': m['fp_per_year'],
            'total_signals': m['total_signals'],
        })
    return pd.DataFrame(results)


# ============================================================
# 7C. FORWARD RETURN + MFE/MAE ANALYSIS
# ============================================================

def calc_forward_returns(events, df, horizons=(5, 10, 20, 40, 60)):
    """
    각 신호 이벤트에 대해 Forward Return, MFE, MAE를 계산.

    Args:
        events: detect_signal_events() 반환값
        df: OHLCV DataFrame
        horizons: 측정할 미래 거래일 수 튜플

    Returns:
        list[dict] — 각 dict:
          type, peak_idx, peak_date, peak_price, peak_val, start_val,
          forward_returns{h: %}, mfe{h: %}, mae{h: %},
          grade ('Strong'|'Moderate'|'Weak'|'Failed'),
          truncated (bool), available_days (int)
    """
    n = len(df)
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    results = []

    for ev in events:
        peak_idx = ev['peak_idx']
        peak_price = close[peak_idx]
        sig_type = ev['type']
        available_days = n - 1 - peak_idx

        fr = {}
        mfe = {}
        mae = {}

        for h in horizons:
            end_idx = min(peak_idx + h, n - 1)
            actual_h = end_idx - peak_idx

            if actual_h < 1:
                fr[h] = np.nan
                mfe[h] = np.nan
                mae[h] = np.nan
                continue

            # Forward return (Close 기준)
            ret = (close[end_idx] - peak_price) / peak_price * 100

            # 윈도우 High/Low (peak 다음날 ~ end_idx)
            w_high = high[peak_idx + 1: end_idx + 1]
            w_low = low[peak_idx + 1: end_idx + 1]

            if len(w_high) == 0:
                fr[h] = np.nan
                mfe[h] = np.nan
                mae[h] = np.nan
                continue

            if sig_type == 'top':
                # Top: 하락이 유리, 상승이 불리
                mfe[h] = (peak_price - w_low.min()) / peak_price * 100
                mae[h] = (w_high.max() - peak_price) / peak_price * 100
            else:
                # Bottom: 상승이 유리, 하락이 불리
                mfe[h] = (w_high.max() - peak_price) / peak_price * 100
                mae[h] = (peak_price - w_low.min()) / peak_price * 100

            fr[h] = ret

        # 등급: 60일 MFE 기준
        mfe_60 = mfe.get(60, np.nan)
        if np.isnan(mfe_60) or mfe_60 < 1.0:
            grade = 'Failed'
        elif mfe_60 < 2.0:
            grade = 'Weak'
        elif mfe_60 < 5.0:
            grade = 'Moderate'
        else:
            grade = 'Strong'

        results.append({
            'type': sig_type,
            'peak_idx': peak_idx,
            'peak_date': df.index[peak_idx].strftime('%Y-%m-%d'),
            'peak_price': peak_price,
            'peak_val': ev['peak_val'],
            'start_val': ev.get('start_val', None),
            'forward_returns': fr,
            'mfe': mfe,
            'mae': mae,
            'grade': grade,
            'truncated': available_days < max(horizons),
            'available_days': available_days,
        })

    return results


def run_forward_return_analysis(score, df, th=0.15, cooldown=5,
                                 horizons=(5, 10, 20, 40, 60)):
    """
    Forward Return + MFE/MAE 분석 파이프라인.

    Returns:
        dict with 'events' (list) + 'summary' (dict):
          summary: n_events, n_top, n_bottom, grade_dist,
                   avg_fr, avg_mfe, avg_mae, avg_fr_top, avg_fr_bottom,
                   hit_rate, profit_factor
    """
    events = detect_signal_events(score, th=th, cooldown=cooldown)
    fr_results = calc_forward_returns(events, df, horizons=horizons)

    n_events = len(fr_results)
    if n_events == 0:
        empty_h = {h: np.nan for h in horizons}
        return {
            'events': [],
            'summary': {
                'n_events': 0, 'n_top': 0, 'n_bottom': 0,
                'grade_dist': {'Strong': 0, 'Moderate': 0, 'Weak': 0, 'Failed': 0},
                'avg_fr': empty_h, 'avg_mfe': empty_h, 'avg_mae': empty_h,
                'avg_fr_top': empty_h, 'avg_fr_bottom': empty_h,
                'hit_rate': 0.0, 'profit_factor': 0.0,
            }
        }

    grade_dist = {'Strong': 0, 'Moderate': 0, 'Weak': 0, 'Failed': 0}
    for r in fr_results:
        grade_dist[r['grade']] += 1

    avg_fr, avg_mfe, avg_mae = {}, {}, {}
    avg_fr_top, avg_fr_bottom = {}, {}

    for h in horizons:
        vals_fr = [r['forward_returns'][h] for r in fr_results
                   if not np.isnan(r['forward_returns'][h])]
        vals_mfe = [r['mfe'][h] for r in fr_results
                    if not np.isnan(r['mfe'][h])]
        vals_mae = [r['mae'][h] for r in fr_results
                    if not np.isnan(r['mae'][h])]

        avg_fr[h] = np.mean(vals_fr) if vals_fr else np.nan
        avg_mfe[h] = np.mean(vals_mfe) if vals_mfe else np.nan
        avg_mae[h] = np.mean(vals_mae) if vals_mae else np.nan

        top_fr = [r['forward_returns'][h] for r in fr_results
                  if r['type'] == 'top' and not np.isnan(r['forward_returns'][h])]
        bot_fr = [r['forward_returns'][h] for r in fr_results
                  if r['type'] == 'bottom' and not np.isnan(r['forward_returns'][h])]
        avg_fr_top[h] = np.mean(top_fr) if top_fr else np.nan
        avg_fr_bottom[h] = np.mean(bot_fr) if bot_fr else np.nan

    n_good = grade_dist['Strong'] + grade_dist['Moderate']
    hit_rate = n_good / n_events * 100

    mfe_good = sum(r['mfe'][60] for r in fr_results
                   if r['grade'] in ('Strong', 'Moderate')
                   and not np.isnan(r['mfe'].get(60, np.nan)))
    mae_bad = sum(r['mae'][60] for r in fr_results
                  if r['grade'] == 'Failed'
                  and not np.isnan(r['mae'].get(60, np.nan)))
    profit_factor = mfe_good / mae_bad if mae_bad > 0 else np.inf

    n_top = sum(1 for r in fr_results if r['type'] == 'top')
    n_bottom = sum(1 for r in fr_results if r['type'] == 'bottom')

    return {
        'events': fr_results,
        'summary': {
            'n_events': n_events,
            'n_top': n_top,
            'n_bottom': n_bottom,
            'grade_dist': grade_dist,
            'avg_fr': avg_fr,
            'avg_mfe': avg_mfe,
            'avg_mae': avg_mae,
            'avg_fr_top': avg_fr_top,
            'avg_fr_bottom': avg_fr_bottom,
            'hit_rate': hit_rate,
            'profit_factor': profit_factor,
        }
    }


# ============================================================
# 8. VISUALIZATION — 실제 시장 차트
# ============================================================

def plot_real_market(df, v3s, v4s, v4ps, transitions, ev3, ev4, ev4p, ticker, output_dir='.'):
    """메인 비교 차트"""
    fig = plt.figure(figsize=(30, 36))
    gs = gridspec.GridSpec(6, 1, height_ratios=[3.5, 1.5, 1.5, 1.5, 1.5, 3], hspace=0.10)
    
    dates = df.index
    x = np.arange(len(df))
    
    def add_tp(ax, use_x=True):
        for tp_idx, tp_type, desc, actual_date in transitions:
            if tp_idx < 0 or tp_idx >= len(df): continue
            c = '#c0392b' if tp_type == 'top' else '#27ae60'
            pos = tp_idx if use_x else dates[tp_idx]
            ax.axvline(pos, color=c, lw=0.8, ls='--', alpha=0.4)
    
    # 대통령/이벤트 배경 (대략적)
    events = [
        (2000, 2003, '#ffcccc', 'Dot-com\nBurst'),
        (2007, 2009, '#ffcccc', 'GFC'),
        (2010, 2011.8, '#ccffcc', 'Recovery'),
        (2020, 2020.5, '#ffcccc', 'COVID'),
        (2022, 2022.8, '#ffffcc', 'Rate\nHikes'),
    ]
    
    # P1: Price + Volume
    ax1 = fig.add_subplot(gs[0])
    ax1.semilogy(x, df['Close'], '#2c3e50', lw=1.0, label=f'{ticker} Price (log)')
    
    for tp_idx, tp_type, desc, actual_date in transitions:
        if tp_idx < 0 or tp_idx >= len(df): continue
        m = 'v' if tp_type == 'top' else '^'
        c = '#c0392b' if tp_type == 'top' else '#27ae60'
        ax1.scatter(tp_idx, df['Close'].iloc[tp_idx], marker=m, s=120, c=c, zorder=5, edgecolors='k', lw=0.8)
        # 이벤트 라벨
        y_pos = df['Close'].iloc[tp_idx] * (1.15 if tp_type == 'top' else 0.85)
        ax1.annotate(f"{actual_date[:7]}\n{desc}", (tp_idx, df['Close'].iloc[tp_idx]),
                    textcoords="offset points", xytext=(0, 20 if tp_type=='top' else -20),
                    fontsize=5.5, ha='center', va='bottom' if tp_type=='top' else 'top',
                    color=c, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=c, lw=0.5))
    
    ax1v = ax1.twinx()
    ax1v.bar(x, df['Volume'], alpha=0.08, color='steelblue', width=1.0)
    ax1v.set_ylabel('Volume', color='steelblue', fontsize=9)
    
    # x축을 날짜로 변환
    tick_years = range(df.index[0].year, df.index[-1].year + 1, 2)
    tick_positions = []
    tick_labels = []
    for y in tick_years:
        ts = pd.Timestamp(f'{y}-01-01')
        idx = df.index.searchsorted(ts)
        if idx < len(df):
            tick_positions.append(idx)
            tick_labels.append(str(y))
    
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, fontsize=8)
    ax1.set_ylabel('Price (log scale)', fontsize=11)
    ax1.set_title(f'{ticker} — Volume Indicator Ecosystem Real Market Backtest (2000-2025)\n'
                  f'V3 (Separated) vs V4 (Fused) vs V4+ (Hybrid)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_xlim(0, len(df))
    add_tp(ax1)
    
    # P2-P4: Three architecture scores
    arch_info = [
        (v3s, ev3, 'V3 (Separated)', '#3498db'),
        (v4s, ev4, 'V4 (Fused)', '#e67e22'),
        (v4ps, ev4p, 'V4+ (Hybrid)', '#8e44ad'),
    ]
    
    for pi, (score, ev, label, color) in enumerate(arch_info):
        ax = fig.add_subplot(gs[1 + pi], sharex=ax1)
        add_tp(ax)
        ax.fill_between(x, score, 0, where=score > 0, alpha=0.4, color='#27ae60')
        ax.fill_between(x, score, 0, where=score < 0, alpha=0.4, color='#c0392b')
        ax.plot(x, score, '#2c3e50', lw=0.6)
        ax.axhline(0, color='gray', lw=0.5)
        
        # Detection markers
        for _, r in ev.iterrows():
            if r['detected']:
                di = int(r['det_idx'])
                m = 'v' if r['type'] == 'top' else '^'
                c = '#c0392b' if r['type'] == 'top' else '#27ae60'
                ax.scatter(di, score.iloc[di], marker=m, s=120, c=c, zorder=5, edgecolors='k', lw=1.2)
        
        ax.set_ylabel(label, fontsize=9, fontweight='bold', color=color)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=7)
    
    # P5: Score difference + consensus zone
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    add_tp(ax5)
    
    # V3와 V4의 합의도 표시
    agreement = v3s * v4s
    ax5.fill_between(x, agreement, 0, where=agreement > 0, alpha=0.3, color='#27ae60', label='Agreement (same dir)')
    ax5.fill_between(x, agreement, 0, where=agreement < 0, alpha=0.3, color='#e74c3c', label='Disagreement')
    ax5.axhline(0, color='gray', lw=0.5)
    ax5.set_ylabel('V3×V4\nAgreement', fontsize=9)
    ax5.legend(loc='upper left', fontsize=7)
    ax5.set_xticks(tick_positions)
    ax5.set_xticklabels(tick_labels, fontsize=7)
    
    # P6: Summary Table
    ax6 = fig.add_subplot(gs[5])
    ax6.axis('off')
    txt = build_real_summary(ev3, ev4, ev4p, ticker)
    ax6.text(0.01, 0.97, txt, transform=ax6.transAxes, fontsize=8,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', fc='#f8f9fa', ec='#dee2e6'))
    
    filepath = os.path.join(output_dir, f'real_backtest_{ticker}.png')
    plt.savefig(filepath, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")
    return filepath


def build_real_summary(ev3, ev4, ev4p, ticker):
    txt = "=" * 110 + "\n"
    txt += f"  {ticker} REAL MARKET BACKTEST — V3 vs V4 vs V4+\n"
    txt += "=" * 110 + "\n\n"
    
    for label, ev in [("V3 (Separated: A*,B*,C* + E)", ev3),
                       ("V4 (Fused: PV-RSI, PV-Div, PV-Conc, PV-Force)", ev4),
                       ("V4+ (Hybrid: V3+V4 consensus)", ev4p)]:
        if len(ev) == 0: continue
        dr = ev['detected'].mean() * 100
        al = ev.loc[ev['detected'], 'lead'].mean() if ev['detected'].any() else 0
        avs = ev.loc[ev['detected'], 'strength'].mean() if ev['detected'].any() else 0
        tops = ev[ev['type'] == 'top']
        bots = ev[ev['type'] == 'bottom']
        tr = tops['detected'].mean() * 100 if len(tops) > 0 else 0
        tl = tops.loc[tops['detected'], 'lead'].mean() if len(tops) > 0 and tops['detected'].any() else 0
        br = bots['detected'].mean() * 100 if len(bots) > 0 else 0
        bl = bots.loc[bots['detected'], 'lead'].mean() if len(bots) > 0 and bots['detected'].any() else 0
        
        txt += f"  [{label}]\n"
        txt += f"  Overall: Det={dr:.0f}%  Lead={al:.1f}d  Strength={avs:.3f}\n"
        txt += f"  Tops: Det={tr:.0f}% Lead={tl:.1f}d  |  Bottoms: Det={br:.0f}% Lead={bl:.1f}d\n\n"
    
    txt += "-" * 110 + "\n"
    txt += f"  {'Date':<12} {'Type':<7} {'Event':<30}  "
    txt += f"{'V3':>4} {'V3 Lead':>8} {'V3 Str':>7}  "
    txt += f"{'V4':>4} {'V4 Lead':>8} {'V4 Str':>7}  "
    txt += f"{'V4+':>4} {'V4+Lead':>8} {'V4+Str':>7}  {'Best':>5}\n"
    txt += "-" * 110 + "\n"
    
    wins = {'V3': 0, 'V4': 0, 'V4+': 0, 'MISS': 0}
    
    for i in range(len(ev3)):
        r3, r4, rp = ev3.iloc[i], ev4.iloc[i], ev4p.iloc[i]
        
        d3 = "Y" if r3['detected'] else "N"
        l3 = f"{r3['lead']:.0f}d" if r3['detected'] else "-"
        s3 = f"{r3['strength']:.3f}" if r3['detected'] else "-"
        d4 = "Y" if r4['detected'] else "N"
        l4 = f"{r4['lead']:.0f}d" if r4['detected'] else "-"
        s4 = f"{r4['strength']:.3f}" if r4['detected'] else "-"
        dp = "Y" if rp['detected'] else "N"
        lp = f"{rp['lead']:.0f}d" if rp['detected'] else "-"
        sp = f"{rp['strength']:.3f}" if rp['detected'] else "-"
        
        best_sc = {}
        for nm, r in [('V3', r3), ('V4', r4), ('V4+', rp)]:
            if r['detected']:
                best_sc[nm] = r['lead'] * 0.5 + r['strength'] * 10
        best = max(best_sc, key=best_sc.get) if best_sc else "MISS"
        wins[best] = wins.get(best, 0) + 1
        
        txt += f"  {r3['actual_date']:<12} {r3['type']:<7} {r3['desc'][:30]:<30}  "
        txt += f"{d3:>4} {l3:>8} {s3:>7}  "
        txt += f"{d4:>4} {l4:>8} {s4:>7}  "
        txt += f"{dp:>4} {lp:>8} {sp:>7}  {best:>5}\n"
    
    txt += "-" * 110 + "\n"
    txt += f"  SCORECARD:  V3: {wins.get('V3',0)}  |  V4: {wins.get('V4',0)}  |  V4+: {wins.get('V4+',0)}  |  All Missed: {wins.get('MISS',0)}\n"
    
    return txt


def plot_metrics_real(evals_dict, output_dir='.'):
    """SPY와 QQQ 성능 비교 (하나의 차트)"""
    fig, axes = plt.subplots(2, 3, figsize=(22, 11))
    
    for row_idx, (ticker, evals) in enumerate(evals_dict.items()):
        ev3, ev4, ev4p = evals
        names = ['V3\n(Sep)', 'V4\n(Fuse)', 'V4+\n(Hyb)']
        colors = ['#3498db', '#e67e22', '#8e44ad']
        evs = [ev3, ev4, ev4p]
        
        # Detection Rate
        rates = [e['detected'].mean()*100 if len(e)>0 else 0 for e in evs]
        bars = axes[row_idx, 0].bar(names, rates, color=colors, alpha=0.7, edgecolor='k')
        for b in bars:
            axes[row_idx, 0].text(b.get_x()+b.get_width()/2, b.get_height()+1,
                                  f'{b.get_height():.0f}%', ha='center', fontweight='bold', fontsize=11)
        axes[row_idx, 0].set_ylim(0, 115)
        axes[row_idx, 0].set_ylabel(f'{ticker}\nDetection Rate (%)')
        if row_idx == 0: axes[row_idx, 0].set_title('Detection Rate', fontweight='bold')
        
        # Lead Days
        leads = [e.loc[e['detected'],'lead'].mean() if e['detected'].any() else 0 for e in evs]
        bc = ['green' if l>0 else 'red' for l in leads]
        bars = axes[row_idx, 1].bar(names, leads, color=bc, alpha=0.7, edgecolor='k')
        for b in bars:
            axes[row_idx, 1].text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                                  f'{b.get_height():.1f}d', ha='center', fontweight='bold', fontsize=11)
        axes[row_idx, 1].axhline(0, color='k', lw=0.5)
        axes[row_idx, 1].set_ylabel('Lead Days')
        if row_idx == 0: axes[row_idx, 1].set_title('Avg Lead Time', fontweight='bold')
        
        # Strength
        strs = [e.loc[e['detected'],'strength'].mean() if e['detected'].any() else 0 for e in evs]
        bars = axes[row_idx, 2].bar(names, strs, color=colors, alpha=0.7, edgecolor='k')
        for b in bars:
            axes[row_idx, 2].text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                                  f'{b.get_height():.3f}', ha='center', fontweight='bold', fontsize=10)
        axes[row_idx, 2].set_ylabel('Signal Strength')
        if row_idx == 0: axes[row_idx, 2].set_title('Avg Strength', fontweight='bold')
    
    fig.suptitle('SPY vs QQQ — Real Market Performance Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'real_backtest_metrics.png')
    plt.savefig(filepath, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")
    return filepath


# ============================================================
# MAIN
# ============================================================

def run_single_ticker(ticker, output_dir='.'):
    """하나의 티커에 대한 전체 백테스트 실행"""
    print(f"\n{'='*60}")
    print(f"  Processing {ticker}")
    print(f"{'='*60}")
    
    # 1. Data
    print("\n[1/5] Loading data...")
    df = download_data(ticker)
    
    # 2. Transitions
    print("[2/5] Matching transition points...")
    raw_transitions = get_transition_points(ticker)
    transitions = match_transitions_to_index(raw_transitions, df)
    print(f"  Matched {len(transitions)} transition points:")
    for tp_idx, tp_type, desc, actual_date in transitions:
        price = df['Close'].iloc[tp_idx]
        print(f"    {actual_date}  {tp_type:>6}  ${price:>8.2f}  {desc}")
    
    # 3. Compute scores
    print("\n[3/5] Computing architecture scores...")
    print("  V3 (separated)...", end=' ')
    v3 = calc_v3_score(df)
    print("done")
    print("  V4 (fused)...", end=' ')
    v4 = calc_v4_score(df)
    print("done")
    print("  V4+ (hybrid)...", end=' ')
    v4p = calc_v4p_score(v3, v4)
    print("done")
    
    # 4. Evaluate
    print("[4/5] Evaluating...")
    ev3 = evaluate(v3, transitions, df)
    ev4 = evaluate(v4, transitions, df)
    ev4p = evaluate(v4p, transitions, df)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"  {ticker} RESULTS")
    print(f"{'='*60}")
    
    for label, ev in [("V3 (Separated)", ev3), ("V4 (Fused)", ev4), ("V4+ (Hybrid)", ev4p)]:
        dr = ev['detected'].mean()*100 if len(ev) > 0 else 0
        al = ev.loc[ev['detected'],'lead'].mean() if len(ev)>0 and ev['detected'].any() else 0
        avs = ev.loc[ev['detected'],'strength'].mean() if len(ev)>0 and ev['detected'].any() else 0
        print(f"\n  [{label}]")
        print(f"  Detection: {dr:.0f}%  Lead: {al:.1f}d  Strength: {avs:.3f}")
        for _, r in ev.iterrows():
            st = "YES" if r['detected'] else " NO"
            ld = f"{r['lead']:>4.0f}d" if r['detected'] else "    -"
            print(f"    {r['actual_date']}  {r['type']:>6}  {st}  Lead:{ld}  Str:{r['strength']:.3f}  {r['desc'][:35]}")
    
    # 5. Visualize
    print(f"\n[5/5] Generating {ticker} charts...")
    fpath = plot_real_market(df, v3, v4, v4p, transitions, ev3, ev4, ev4p, ticker, output_dir)
    
    return ev3, ev4, ev4p


if __name__ == '__main__':
    print("=" * 65)
    print("  Volume Indicator Ecosystem — Real Market Backtest")
    print("  SPY & QQQ (2000-2025)")
    print("  V3 (Separated) vs V4 (Fused) vs V4+ (Hybrid)")
    print("=" * 65)
    
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    
    all_evals = {}
    
    for ticker in ['SPY', 'QQQ']:
        ev3, ev4, ev4p = run_single_ticker(ticker, output_dir)
        all_evals[ticker] = (ev3, ev4, ev4p)
    
    # Combined metrics chart
    print("\n[FINAL] Generating combined metrics chart...")
    plot_metrics_real(all_evals, output_dir)
    
    print("\n" + "=" * 65)
    print("  ALL COMPLETE")
    print("=" * 65)
    print(f"\n  Output directory: {os.path.abspath(output_dir)}")
    print(f"  Files:")
    for f in os.listdir(output_dir):
        if f.endswith('.png'):
            print(f"    - {f}")
