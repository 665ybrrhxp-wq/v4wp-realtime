# V4_wP 실시간 트레이딩 시그널 시스템 — 전체 구조 및 로직 가이드

> 최종 업데이트: 2026-03-15 (DivGate_3d + 실적발표일 거래량 필터 적용)

## 1. 시스템 개요

미국 주식 15종목을 매일 스캔하여, **거래량-가격 융합 지표(V4_wP)**로 매수/매도 시점을 감지하고 Telegram으로 알림을 보내는 시스템이다. Claude AI가 한 줄 코멘터리를 생성하고, Streamlit 대시보드로 시각화한다.

### 핵심 철학

6단계 최적화를 거쳐 내린 결론:

- **V4는 "DCA를 이기는 매매 시스템"이 아니라 "리스크 모니터링 & 심리 보조 도구"**
- 17종목, 최대 28년 백테스트에서 어떤 타이밍 전략도 DCA(Dollar Cost Averaging)를 이기지 못함
- V4의 진짜 가치: **매수 적중률 70%+ (90일 기준)**, 하락장 방어, 심리적 매수 근거 제공

### 워치리스트 (15종목 + 벤치마크 2개)

| 섹터 | 종목 |
|---|---|
| Tech | TSLA, PLTR, NVDA, AVGO, AMZN, GOOGL |
| Growth | JOBY, HIMS, TEM, RKLB, PGY |
| Fintech | COIN, HOOD |
| Quantum | IONQ |
| Space | PL |
| 벤치마크 | QQQ, VOO |

---

## 2. V4_wP 지표 — 핵심 수학

### 2-1. 3개의 서브 지표

V4 스코어는 **가격과 거래량을 동시에 분석**하는 3개 서브 지표의 가중 합산이다.

#### (1) S_Force — 거래량 힘 (가중치 45%)
```
PV_Force = (거래량 / 20일 평균 거래량) × 가격 가속도
가격 가속도 = 일간 수익률의 차분 (2차 미분)

MACD 방식으로 처리:
  Fast EMA(12) - Slow EMA(26) → Signal EMA(9) → Histogram

S_Force = clip(Histogram / (2 × 20일 std), -1, +1)
```
- 거래량이 터지면서 가격이 가속하는 구간을 포착
- +1에 가까우면 강한 매수세, -1에 가까우면 강한 매도세

#### (2) S_Divergence — 가격-거래량 괴리 (가중치 30%) + DivGate_3d
```
가격 모멘텀 = 20일 수익률 / (40일 rolling std)
거래량 모멘텀 = 20일 거래량 변화율 / (40일 rolling std)

PV_DivIdx = 거래량 모멘텀 - 가격 모멘텀
S_Div_raw = clip(PV_DivIdx / 3, -1, +1)
```
- 거래량은 늘어나는데 가격이 안 오르면 → 양수 (바닥 근처, 매수 기회)
- 거래량은 줄어드는데 가격만 오르면 → 음수 (천장 근처, 매도 경고)

**DivGate_3d (신규):**
```
같은 부호의 S_Div_raw가 3일 연속 유지되어야 S_Div 활성화.
3일 미만 → S_Div = 0 (노이즈로 간주하고 무시)

예시:
  Day 1: S_Div_raw = +0.3 → consec = 1 → S_Div = 0 (비활성)
  Day 2: S_Div_raw = +0.2 → consec = 2 → S_Div = 0 (비활성)
  Day 3: S_Div_raw = +0.4 → consec = 3 → S_Div = +0.4 (활성!)
  Day 4: S_Div_raw = -0.1 → consec = 1 → S_Div = 0 (부호 변경, 리셋)
```
효과: 1~2일짜리 단발성 괴리 신호를 걸러내어 90일 적중률 +0.5%p, Edge +0.7%p 개선.

#### (3) S_Concordance — 방향 일치도 (가중치 25%)
```
S_Conc = 20일 rolling 상관계수(일간 수익률, 일간 거래량 변화율)
```
- +1이면 가격과 거래량이 같은 방향 (건강한 추세)
- -1이면 역방향 (추세 약화 신호)

### 2-2. 실적발표일 거래량 스무딩 (전처리)

V4 스코어 계산 전, 실적발표일 전후의 비정상적 거래량 스파이크를 제거한다.

```
2단계 감지:
  1) yfinance earnings_dates (최근 ~2.5년 실적발표 일정)
  2) 거래량 > 3× 20일 중앙값 (오래된 데이터의 스파이크 감지)

감지된 날짜 ±1 거래일의 Volume → 20일 rolling median으로 교체
```
- 실적발표 당일/전후의 거래량은 실적 서프라이즈에 의한 것이므로 가격-거래량 관계 분석에 노이즈
- 원본 데이터를 수정하지 않고 복사본에서 처리

### 2-3. 최종 V4 스코어 계산

```python
방향값 = 0.45 × S_Force + 0.30 × S_Div(gated) + 0.25 × S_Conc

활성 지표 수 = |S_Force| > 0.1 인가? + |S_Div| > 0.1 인가? + |S_Conc| > 0.1 인가?

Activity Multiplier:
  0개 활성 → ×0.5  (신호 약화)
  1개 활성 → ×1.0  (기본)
  2개 활성 → ×1.5  (증폭)
  3개 활성 → ×2.2  (강력 증폭)

V4_Score = 방향값 × Activity Multiplier
```

**핵심 아이디어**: 3개 지표가 모두 같은 방향을 가리키면 ×2.2 배로 증폭시켜 강한 신호를 만든다. 하나만 반응하면 노이즈로 간주해 약화시킨다.

---

## 3. 신호 감지 파이프라인

전체 흐름:
```
OHLCV 데이터 (yfinance, 3년치)
  → [0] smooth_earnings_volume()  실적발표일 거래량 스무딩
  → [1] calc_v4_score()           V4 스코어 시계열 생성 (DivGate_3d 포함)
  → [2] detect_signal_events()    임계값 기반 이벤트 추출
  → [3] build_price_filter()      가격 필터 (잡음 제거)
  → [4] LATE_SELL_BLOCK           뒤늦은 매도 차단
  → [5] classify_signal()         Duration 기반 확인 (3일 연속)
  → [6] is_new_signal()           중복 제거 (±3일)
  → Telegram 알림 + AI 코멘터리
```

### 3-1. 신호 이벤트 추출 (`detect_signal_events`)

V4 스코어 시계열에서 임계값을 넘는 연속 구간을 찾아낸다.

```
매도 신호 (top):    V4_Score < -0.15 인 연속 구간
매수 신호 (bottom): V4_Score > +0.075 인 연속 구간  (0.15 × 0.5)
```

- 매수 임계값이 매도보다 낮은 이유: 바닥 신호는 약하게 시작되므로 더 민감하게 잡는다
- **쿨다운 5일**: 신호가 끊겼다가 5일 내 다시 나타나면 같은 이벤트로 병합한다
- 구간 내 가장 극단값이 나온 날짜(`peak_idx`)와 그 값(`peak_val`)을 기록한다
- **duration** = 구간 길이 (end_idx - start_idx + 1)

### 3-2. 가격 필터 (`build_price_filter`)

거래량 신호만으로는 노이즈가 많으므로, **가격 움직임의 질**을 추가로 검증한다.

```
통과 조건 (두 가지 모두 충족해야 함):

(1) ER < 66 퍼센타일
    ER(Efficiency Ratio) = |20일 방향 이동| / |20일 경로 합|
    ER이 낮다 = 가격이 방향 없이 왔다갔다 = 전환점 근처일 가능성

(2) ATR_pct > 55 퍼센타일
    ATR = 14일 평균 변동폭의 252일 퍼센타일 순위
    ATR이 높다 = 변동성이 충분하다 = 의미 있는 움직임
```

**요약**: "방향이 혼란스럽고(ER 낮음) + 변동성은 큰(ATR 높음)" 구간에서 나온 신호만 통과시킨다. 조용한 횡보 구간의 잡음 신호를 걸러낸다.

### 3-3. LATE_SELL_BLOCK — 뒤늦은 매도 차단

```
매도 신호가 나왔을 때:
  현재가와 20일 롤링 고점을 비교
  (롤링 고점 - 현재가) / 롤링 고점 > 5% 이면 → 매도 차단

이유: 이미 5% 이상 빠진 뒤에 매도하면 손실만 확정시키므로,
      "너무 늦은 매도"는 하지 않는 것이 낫다.
```

### 3-4. 신호 확인 — Duration 기반 (`classify_signal`)

필터를 통과한 신호를 **지속 일수(duration)**로 확인한다.

```
매수 (bottom):
  duration >= 3일 → CONFIRMED → 100% 풀매수 실행
  duration < 3일  → PENDING   → 무시 (아직 확인 안 됨)

매도 (top):
  duration >= 3일 → SELL_CONFIRMED → 보유량의 5% 매도
  duration < 3일  → PENDING        → 무시
```

| 분류 | 조건 | 행동 |
|---|---|---|
| **BUY (CONFIRMED)** | 매수 신호 3일+ 지속 | 가용자금의 **100%** 매수 |
| **SELL (CONFIRMED)** | 매도 신호 3일+ 지속 | 보유량의 **5%** 매도 |
| PENDING | 지속 3일 미만 | 무시 (알림 안 보냄) |

**설계 원칙**:
- **매수는 과감하게(100%)**: 3일 연속 확인된 매수 신호는 신뢰도가 높으므로 전액 투입
- **매도는 최소한으로(5%)**: 14가지 매도 전략 비교 결과, "매도를 안 하는 게 최고"(-11.81%)이며 5% 매도가 2순위(-11.92%). 장기 우상향하는 미국 주식시장에서 조기 매도로 수익을 놓치는 것을 방지
- **PENDING = 노이즈**: 1~2일짜리 단발성 신호는 false positive 확률이 높으므로 무시

### 3-5. 중복 제거 (`is_new_signal`)

같은 종목, 같은 타입(top/bottom)의 신호가 **±3일 이내**에 이미 DB에 있으면 중복으로 판단하여 무시한다.

---

## 4. 진입 시점 판단 — 7단계 요약

**"이 주식을 지금 사야 하는가?"** 에 대한 답을 내리는 과정:

```
[전처리]
0. 실적발표일 전후 거래량이 비정상적인가?               → 거래량 스무딩

[지표 계산]
1. 거래량이 터지면서 가격이 방향을 바꾸고 있는가?        → S_Force  (45%)
2. 거래량과 가격이 괴리를 보이고, 3일 이상 지속되는가?    → S_Div + DivGate (30%)
3. 가격과 거래량이 같은 방향으로 움직이는가?             → S_Conc  (25%)
4. 위 3개 지표가 동시에 반응하는가?                     → Activity Multiplier

[필터링]
5. 가격이 방향 없이 출렁이면서 변동성은 큰 구간인가?     → Price Filter
6. 이미 크게 빠진 뒤 뒤늦은 매도는 아닌가?              → LATE_SELL_BLOCK

[확인]
7. 이 신호가 3일 이상 지속되었는가?                     → Duration Confirm
```

**모든 조건을 통과한 경우에만** Telegram 알림이 발송된다.

---

## 5. 프로덕션 파라미터 (최종 설정)

```json
{
  "v4_window": 20,             // V4 계산 윈도우 (거래일)
  "signal_threshold": 0.15,    // 매도 신호 임계값 (매수는 ×0.5 = 0.075)
  "cooldown": 5,               // 이벤트 병합 쿨다운 (일)
  "er_quantile": 66,           // ER 필터 퍼센타일
  "atr_quantile": 55,          // ATR 필터 퍼센타일
  "lookback": 252,             // 필터 롤링 윈도우 (1년)
  "data_years": 3,             // 데이터 수집 기간
  "confirm_days": 3,           // 매수 확인 기간 (일)
  "buy_confirmed_pct": 1.00,   // 확인된 매수 비율 (100%)
  "sell_confirm_days": 3,      // 매도 확인 기간 (일)
  "sell_confirmed_pct": 0.05,  // 확인된 매도 비율 (5%)
  "late_sell_drop_th": 0.05,   // LATE_SELL 차단 하락률
  "divgate_days": 3,           // S_Div 활성화 최소 연속일
  "earnings_vol_filter": true  // 실적발표일 거래량 스무딩 ON/OFF
}
```

---

## 6. 백테스트 성과 요약

### V4 매수 적중률 (v4_hit_rate_all.py)

| 기간 | 전체 | 30d | 60d | 90d |
|---|---|---|---|---|
| 신호 수 | 551 | - | - | - |
| 적중률 | - | 64% | 68% | 71% |
| 평균 수익 | - | - | - | +30.4% |

### V4 vs 무작위 매수 (v4_vs_random.py, 90일 기준)

| 지표 | V4 매수 | 무작위 매수 | Edge |
|---|---|---|---|
| 연환산 수익률 | +57.0%/yr | +46.9%/yr | **+10.0%p** |
| 적중률 | 70.8% | 65.3% | +5.5%p |
| Edge 양수 종목 | 13/16 | - | - |

### 연도별 V4 Alpha (BnH 대비)

| 시장 상황 | 대표 연도 | V4 성과 | 설명 |
|---|---|---|---|
| 하락장 | 2022 | **+23.94%** | 베어마켓에서 매수 자제 |
| 과열장 | 2021 | **+8.24%** | 과열 구간 경고 |
| 금융위기 | 2008 | **+1.19%** | 하락장 방어 |
| V자 반등 | 2020 | -21.95% | 반등 시점 놓침 |
| AI 랠리 | 2023 | -13.22% | 후발 진입 |
| 초기 폭등 | 1997-98 | -185~795% | AMZN 초기 (극단 케이스) |

---

## 7. 시스템 아키텍처

### 파일 구조 및 의존성

```
real_market_backtest.py          ← 핵심 수학 엔진
  함수: download_data, calc_v4_score(DivGate), calc_v4_subindicators,
        detect_signal_events, build_price_filter, smooth_earnings_volume
        ↑
v4wp_realtime/
├── core/
│   ├── indicators.py            ← 수학 엔진 래핑 + Duration 기반 classify_signal()
│   ├── scanner.py               ← 일일 스캔 오케스트레이션 (전체 파이프라인)
│   └── signal_tracker.py        ← 중복 제거 (±3일 윈도우)
├── config/
│   ├── settings.py              ← 경로 설정 + .env 로딩
│   └── watchlist.json           ← 종목 리스트 + 파라미터 (14개)
├── data/
│   ├── store.py                 ← SQLite CRUD
│   └── schema.sql               ← 테이블 정의 (3개)
├── alerts/
│   ├── telegram_bot.py          ← Telegram 전송
│   └── message_formatter.py     ← 카드형 메시지 포맷
├── ai/
│   ├── commentary.py            ← Claude API 한 줄 평
│   └── prompt_templates.py      ← 프롬프트 템플릿
├── dashboard/
│   └── app.py                   ← Streamlit 대시보드
└── scripts/
    ├── daily_scan.py            ← 진입점 (매일 실행)
    ├── backfill.py              ← 초기 데이터 적재
    └── test_telegram.py         ← Telegram 연결 테스트
```

### 데이터 흐름

```
[yfinance]  →  OHLCV 데이터  →  [캐시 CSV]
                    ↓
            거래량 스무딩 (smooth_earnings_volume)
                    ↓
            V4 스코어 계산 + DivGate (real_market_backtest.py)
                    ↓
            신호 감지 + 필터링 + Duration 확인 (indicators.py)
                    ↓
            중복 제거 (signal_tracker.py)
                    ↓
         ┌──────────┼──────────┐
         ↓          ↓          ↓
    [SQLite DB]  [Telegram]  [Claude AI]
         ↓                      ↓
    [Streamlit]           한 줄 코멘터리
    대시보드               → Telegram 메시지에 포함
```

### DB 테이블 (SQLite)

| 테이블 | 용도 | PK |
|---|---|---|
| `daily_scores` | 일별 V4 스코어 + 서브지표 | (date, ticker) |
| `signal_events` | 감지된 신호 이벤트 + 코멘터리 | (ticker, signal_type, peak_date) |
| `scan_runs` | 스캔 실행 로그 | id (auto) |

---

## 8. 실행 방법

### 환경 설정
```bash
pip install -r requirements.txt

# .env 파일 (v4wp_realtime/.env)
ANTHROPIC_API_KEY=sk-ant-...     # Claude AI 코멘터리용
TELEGRAM_BOT_TOKEN=123456:ABC... # Telegram 봇 토큰
TELEGRAM_CHAT_ID=123456789       # 알림 수신 채팅 ID
```

### 일일 운영
```bash
# 1. 초기 데이터 적재 (최초 1회)
python -m v4wp_realtime.scripts.backfill

# 2. 매일 스캔 (장 마감 후 실행, GitHub Actions cron 가능)
python -m v4wp_realtime.scripts.daily_scan

# 3. 대시보드 (선택)
streamlit run v4wp_realtime/dashboard/app.py

# 4. Telegram 연결 테스트
python -m v4wp_realtime.scripts.test_telegram
```

---

## 9. Telegram 알림 메시지 예시

```
━━━━━━━━━━━━━━━━━━━━
✅ BUY (CONFIRMED)
━━━━━━━━━━━━━━━━━━━━

📌 NVDA (Tech)
💲 $125.50

📊 V4 Score: 0.312  [░░░░░│▓▓▓░░]
┣ Force:  +0.45
┣ Div:    +0.28  (3d+ gated)
┗ Conc:   +0.31

⏱ Duration: 5일 (확인됨)
💰 Action: 가용자금의 100% 매수
📅 2026-03-13

💬 거래량 급증과 함께 가격 반전 조짐이 뚜렷하다.
━━━━━━━━━━━━━━━━━━━━
```

---

## 10. 최적화 이력

| Phase | 변경 | 결과 |
|---|---|---|
| 1. C25 | peak_val 기반 4등급 분류 | look-ahead bias 발견 |
| 2. V2 | peak_val → start_val | bias 제거, alpha 음수 |
| 3. V3~V4 | start_val → Duration 기반 2등급 | 단순화, alpha -12.01% |
| 4. 매도 최적화 | 14가지 전략 비교 | "매도 안 함"이 최고 |
| 5. Hybrid DCA | DCA+BIL 하이브리드 | auto=100%(BnH) 최적 |
| 6. Golden Cross | V4+이평선 GC | V4only가 더 나음 |
| 7. DivGate_3d | S_Div 3일 연속 게이트 | 90d Edge +0.7%p |
| 8. 실적발표 필터 | 실적일 거래량 스무딩 | 노이즈 감소 |

---

## 11. 한계 및 참고사항

- **DCA > V4**: 생존한 종목에서 DCA는 항상 V4를 이김. V4는 리스크 모니터링 도구로 활용
- **생존자 편향**: 현재 살아있는 종목만 테스트. 망한 종목(Enron, FTX 등) 미포함
- **백테스트 ≠ 실전**: 과거 데이터 기반이며 미래 수익을 보장하지 않음
- **거래비용 미반영**: 실제 거래비용/슬리피지 미포함
- **거래량 기반 한계**: 유동성이 낮은 종목(소형주)에서는 노이즈가 클 수 있음
- **매도 비율 5%**: 장기 우상향 전제 전략이므로, 하락장에서는 포지션 축소가 느림
- **하락장 성과**: 2008(-46%), 2020(-22%) 등 급락장에서 V4는 반등 타이밍을 놓칠 수 있음
