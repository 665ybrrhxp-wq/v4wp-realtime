# V4_wP — 버그 수정 및 개선 작업 가이드

> 작성일: 2026-05-13
> 대상 커밋: 현재 main
> 범위: `v4wp_realtime/` 전체 + 루트 `real_market_backtest.py` + `.github/workflows/`

이 문서는 전체 코드 리뷰 결과로 발견된 버그와 개선 사항을 **Tier 별로 정리**한 작업 명세서입니다.
각 항목은 다음 구조를 따릅니다:

- **증상** — 사용자/시스템 관점에서 무엇이 잘못되는가
- **원인** — 정확한 파일/라인 + 근거
- **수정** — 무엇을 어떻게 바꿔야 하는가 (코드 발췌 + 변경 방향)
- **검증** — 수정이 적용됐는지 확인하는 방법
- **효과** — 수정 후 시스템에 어떤 변화가 오는가

---

## 📋 우선순위 한눈에 보기

| Tier | ID | 제목 | 영향 | LOC |
|---|---|---|---|---|
| 🔴 0 | D-1 | Haiku 모델 ID 오타 | AI commentary 전체 미동작 추정 | 1줄 |
| 🔴 0 | D-2 | Mini App deep link ticker 파싱 누락 | 알림→Dashboard 컨텍스트 손실 | ~10줄 |
| 🔴 0 | D-3 | yfinance end-date 1일 지연 | 모든 신호/가격 1일 늦음 | ~5줄 |
| 🟡 1 | Q-1 | silent `except: pass` 14곳 | 버그 은닉 (D-1이 발견 안 된 이유) | ~14곳 |
| 🟡 1 | D-4 | VIX `period=` vs 주식 `end=` 시간축 불일치 | 신호 당일 VIX 누락 | ~5줄 |
| 🟡 1 | M-1 | blocked_buys 종목명 미표기 | 차단 종목 모니터링 불가 | 1줄 |
| 🟡 1 | M-2 | interpretation Telegram 미노출 | AI 비용 가치 회수 실패 | ~10줄 |
| 🟡 1 | M-3 | 카드에 레짐 컨텍스트 없음 | 패닉/과열 day 구분 불가 | ~5줄 |
| 🟡 1 | UX-1 | Mini App에서 LONG 뱃지/AI 위원회 무한 노출 | 옛 신호가 현재 신호로 오인 | ~20줄 |
| 🟢 2 | DC-1 | `_send_media_group` 데드 함수 | 코드 부채 | -31줄 |
| 🟢 2 | DC-2 | `_summary_button` 데드 함수 | 코드 부채 | -2줄 |
| 🟢 2 | DC-3 | 콜백 시스템 전체 미배포 | 코드 부채 ~200줄 | -200줄 |
| 🟢 2 | DC-4 | FastAPI 서버 미사용 | 코드 부채 ~500줄 | -515줄 |
| 🟢 2 | DC-5 | SSE 인프라 인-프로세스 한정 | 코드 부채 | -80줄 |
| 🟢 2 | S-3 | postmortem/decay 결과 매 스캔 재계산 | 비용 + 일관성 | ~20줄 |
| ⚪ 3 | P-1 | interpretation을 PENDING에도 호출 | AI 비용 | ~3줄 |
| ⚪ 3 | P-2 | postmortem이 완료 신호도 매번 훑음 | 쿼리 비용 | ~3줄 |
| ⚪ 3 | P-3 | ETF/VIX 다운로드 직렬 | wall-time | ~10줄 |
| ⚪ 3 | S-1 | signals_history.json 무한 누적 | git/IO 비용 | ~5줄 |
| ⚪ 3 | S-2 | `data/charts/*.png` 청소 정책 없음 | 디스크 (DC-3 해결 시 자동 해소) | - |
| ⚪ 3 | Q-2~5 | 코드 품질 잡다 | 가독성/유지보수 | - |

---

# 🔴 Tier 0 — 즉시 수정 (기능이 실제로 망가져 있음)

## D-1. Haiku 모델 ID 오타

### 증상
- AI 한 줄 코멘트가 모든 신호에서 **fallback 텍스트만** 출력될 가능성이 매우 높음.
- Telegram 알림에서 신호 카드의 `💡 ...` 라인이 항상 같은 템플릿 패턴 (예: "강한 매수 신호", "주가 반전 가능성").
- GitHub Actions 로그에 `[AI] Commentary failed for {ticker}: Error code: 404 ...not_found_error...` 가 매 신호마다 찍혀 있을 것.

### 원인
[`v4wp_realtime/ai/commentary.py:26`](v4wp_realtime/ai/commentary.py)
```python
model='claude-haiku-4-5-20241022',
```
- `claude-haiku-4-5` 패밀리는 **2025년 10월** 릴리스 (`claude-haiku-4-5-20251001`).
- `20241022`는 **Haiku 3.5**의 릴리스 날짜.
- 패밀리명과 날짜 스탬프 조합이 존재하지 않는 모델 ID → Anthropic API 404.
- [`commentary.py:38-40`](v4wp_realtime/ai/commentary.py)의 `except Exception: return get_fallback(signal)` 이 에러를 흡수해서 사용자 눈에 보이지 않음.

### 수정
**`v4wp_realtime/ai/commentary.py:26`**
```python
# Before
model='claude-haiku-4-5-20241022',

# After (둘 중 하나)
model='claude-haiku-4-5',              # alias (권장 — 자동 최신)
# or
model='claude-haiku-4-5-20251001',     # 명시적 버전 핀
```

### 검증
1. 로컬에서 dry-run:
   ```bash
   python -c "from v4wp_realtime.ai.commentary import generate_commentary; \
              print(generate_commentary({'ticker':'NVDA','signal_type':'bottom', \
              'peak_val':-0.18,'s_force':0.4,'s_div':0.3,'close_price':500}))"
   ```
   → fallback 템플릿이 아닌 종목 맞춤 문장이 나와야 함.
2. 다음 GitHub Actions 실행 후 로그에서 `[AI] Commentary failed` 메시지가 사라졌는지 확인.

### 효과
- 시스템 핵심 기능(AI 코멘트)이 **사실상 켜짐**.
- 사용자가 받는 카드의 정보 밀도가 즉시 향상.

---

## D-2. Mini App deep link에서 ticker 컨텍스트 손실

### 증상
- Telegram 알림에서 "📱 Dashboard" 버튼을 누르면 Mini App이 열리지만, 알림 종목이 아닌 **워치리스트 첫 종목**으로 진입.

### 원인
경로 1 — Telegram 봇:
[`v4wp_realtime/alerts/telegram_bot.py:246-258`](v4wp_realtime/alerts/telegram_bot.py)
```python
def _dashboard_button(ticker=None, peak_date=None):
    if TELEGRAM_WEBAPP_URL:                    # ← 워크플로에서 항상 set됨
        url = TELEGRAM_WEBAPP_URL.rstrip('/')
        if ticker:
            url += f'?ticker={ticker}'         # ← ?ticker= 만, peak_date는 누락
        return [{'text': '...', 'web_app': {'url': url}}]
    # fallback (t.me 딥링크) — startapp={ticker}_{peak_date} 형식
    url = _miniapp_url(ticker, peak_date)
    ...
```

경로 2 — Mini App:
[`v4wp_realtime/webapp/src/telegram.js:49-59`](v4wp_realtime/webapp/src/telegram.js)
```javascript
export function getStartParam() {
  if (!tg) return null;
  const raw = tg.initDataUnsafe?.start_param;  // ← t.me 딥링크에서만 채워짐
  if (!raw) return null;
  ...
}
```

[`v4wp_realtime/webapp/src/App.jsx:22-25`](v4wp_realtime/webapp/src/App.jsx)
```javascript
const start = getStartParam();
if (start?.ticker) {
  setSelected(start.ticker);
}
// ← location.search (?ticker=) 를 읽는 fallback 없음
```

`TELEGRAM_WEBAPP_URL`이 설정된 경로(현재 워크플로의 기본 경로)는 URL 쿼리스트링으로 ticker를 전달하는데, Mini App은 그걸 읽지 않음.

### 수정

**1. `v4wp_realtime/webapp/src/telegram.js`** — `getStartParam`에 URL 쿼리 fallback 추가
```javascript
export function getStartParam() {
  // 우선순위 1: Telegram start_param (t.me 딥링크)
  const raw = tg?.initDataUnsafe?.start_param;
  if (raw) {
    const sep = raw.indexOf("_");
    if (sep === -1) return { ticker: raw.toUpperCase() };
    return { ticker: raw.slice(0, sep).toUpperCase(), peakDate: raw.slice(sep + 1) };
  }

  // 우선순위 2: URL 쿼리스트링 (web_app 버튼 경로)
  const params = new URLSearchParams(window.location.search);
  const ticker = params.get('ticker');
  if (ticker) {
    return {
      ticker: ticker.toUpperCase(),
      peakDate: params.get('peak_date') || null,
    };
  }
  return null;
}
```

**2. `v4wp_realtime/alerts/telegram_bot.py:246-258`** — peak_date도 URL에 포함
```python
def _dashboard_button(ticker=None, peak_date=None):
    if TELEGRAM_WEBAPP_URL:
        url = TELEGRAM_WEBAPP_URL.rstrip('/')
        params = []
        if ticker:
            params.append(f'ticker={ticker}')
        if peak_date:
            params.append(f'peak_date={peak_date}')
        if params:
            url += '?' + '&'.join(params)
        return [{'text': '\U0001f4f1 Dashboard', 'web_app': {'url': url}}]
    ...
```

### 검증
1. 실제 알림 받은 후 Dashboard 버튼 클릭 → 알림 종목이 선택된 상태로 열리는지 확인.
2. 브라우저 콘솔 (Telegram Web App 디버거)에서 `window.location.search` 가 `?ticker=NVDA&peak_date=2026-05-13` 형태인지 확인.

### 효과
- 매번 워치리스트에서 종목을 다시 찾는 마찰 제거.
- "AAPL 신호 알림 → AAPL 상세 분석" 흐름이 1탭으로 단축.

---

## D-3. yfinance `end` 1일 지연 (모든 신호/가격이 어제 데이터)

### 증상
- Telegram 카드의 "현재 $X" 가격이 항상 **어제 종가**.
- 오늘 발생한 신호의 `peak_date`가 어제로 표기.
- 가격 라인의 변동률(▲▼ %)이 신호가 며칠 전인 경우에만 보임 (`current == close` 이면 [`message_formatter.py:98`](v4wp_realtime/alerts/message_formatter.py)에서 변동률 줄 자체가 생략).

### 원인
[`v4wp_realtime/core/indicators.py:26-28`](v4wp_realtime/core/indicators.py)
```python
end = datetime.now().strftime('%Y-%m-%d')
start = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
return download_data(ticker, start=start, end=end, cache_dir=cache_dir)
```
- yfinance `end` 파라미터는 **exclusive**.
- 워크플로 cron은 UTC 21:00 = EDT 17:00 / EST 16:00 ([daily_scan.yml:7](.github/workflows/daily_scan.yml)) → 미국장 마감(16:00 ET) 직후~1시간 후.
- 오늘 봉이 yfinance에 이미 올라와 있는데도 `end='오늘'` exclusive 때문에 잘라냄.

영향 받는 변수:
- `df['Close'].iloc[-1]` → 어제 종가 ([scanner.py:158](v4wp_realtime/core/scanner.py))
- `peak_idx`가 df 마지막 행인 신호 → `close_price`도 어제 ([scanner.py:157](v4wp_realtime/core/scanner.py))

### 수정

**`v4wp_realtime/core/indicators.py:21-28`**
```python
def fetch_data(ticker, years=3, cache_dir=None):
    """최근 N년 데이터 다운로드 (period 모드 — 오늘 봉 포함)."""
    from datetime import datetime, timedelta, date

    if cache_dir is None:
        cache_dir = str(Path(_project_root) / 'cache')

    # period= 방식으로 통일 — end exclusive 문제 회피.
    # download_data가 start/end를 받으므로 end를 내일로 설정해 오늘 봉을 포함.
    end = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
    df = download_data(ticker, start=start, end=end, cache_dir=cache_dir)

    # 인트라데이 안전장치: 만약 마지막 행이 오늘 날짜인데
    # 현재 시각이 미국장 마감(20:00 UTC EST / 21:00 UTC EDT) 이전이면
    # 미완성 봉이므로 drop. Actions cron 시간상 일반적으로 해당 없음.
    if df is not None and len(df) > 0:
        last_date = df.index[-1].date() if hasattr(df.index[-1], 'date') else None
        if last_date == date.today():
            # 현재 UTC 시각이 21:00 이전이면 미완성 봉 가능
            if datetime.utcnow().hour < 21:
                df = df.iloc[:-1]
    return df
```

**중요**: `real_market_backtest.py:48`의 캐시 키가 `{ticker}_{start}_{end}.csv` 형태인데, end가 매일 바뀌므로 캐시가 사실상 동작 안 함(어차피 Actions에선 매번 fresh 컨테이너). 캐시 효율을 원하면 S-1 항목 참조.

### 검증
1. 로컬:
   ```bash
   python -c "from v4wp_realtime.core.indicators import fetch_data; \
              df = fetch_data('AAPL'); print(df.index[-1], df['Close'].iloc[-1])"
   ```
   → 오늘 날짜와 오늘 종가가 나와야 함 (미국장 마감 후 실행 시).
2. 다음 스캔 결과 Telegram 카드에서 "📅 {today}" 형태로 오늘 날짜가 표시되는지 확인.

### 효과
- 신호 발생일이 실제 발생일과 일치.
- "현재가" 라벨이 실제 현재가(=오늘 종가)를 가리킴.
- 매수 결정이 하루 빨라짐.

---

# 🟡 Tier 1 — 높은 가치, 낮은 위험

## Q-1. Silent `except: pass` 제거

### 증상
- D-1처럼 중요한 기능이 망가져도 사용자/개발자가 발견 불가.
- 디버깅 시 "왜 코멘트가 fallback이지?" → 원인 추적까지 시간 소모.

### 원인
다음 위치에서 예외가 silently 흡수:

| 파일:라인 | 컨텍스트 |
|---|---|
| [`scanner.py:75-76`](v4wp_realtime/core/scanner.py) | QQQ/섹터 ETF 다운로드 실패 |
| [`scanner.py:85-86`](v4wp_realtime/core/scanner.py) | VIX 다운로드 실패 |
| [`scanner.py:231-232`](v4wp_realtime/core/scanner.py) | similarity 검색 실패 |
| [`scanner.py:241-242`](v4wp_realtime/core/scanner.py) | postmortem stats 실패 |
| [`scanner.py:250-251`](v4wp_realtime/core/scanner.py) | decay context 실패 |
| [`scanner.py:256-257`](v4wp_realtime/core/scanner.py) | commentary_fn 실패 ← **D-1 은닉**|
| [`scanner.py:268-269`](v4wp_realtime/core/scanner.py) | interpretation_fn 실패 |
| [`scanner.py:289-290`](v4wp_realtime/core/scanner.py) | chart 생성 실패 |
| [`scanner.py:302-303`](v4wp_realtime/core/scanner.py) | cross_ticker 실패 |
| [`store.py:11-13`](v4wp_realtime/data/store.py) | event_bus publish (의도된 silent — DC-5 해결 시 같이 제거) |

### 수정
모든 위치를 다음 패턴으로 교체:
```python
# Before
except Exception:
    pass

# After
except Exception as e:
    print(f'  [scanner] {context_label} failed for {ticker}: {e}')
```

특히 `commentary_fn` / `interpretation_fn` 실패는 **신호당 1회 print** 정도면 충분 (스팸 안 됨).

### 검증
- Actions 로그를 grep으로 검사: `grep "failed" actions.log`
- D-1 수정 전 단계에서 모든 신호에 대해 `[scanner] commentary failed` 가 찍히는지 확인.

### 효과
- D-1 같은 silent failure가 즉시 가시화.
- 신규 기능 추가 시 회귀 탐지 시간 단축.

---

## D-4. VIX `period=` vs 주식 `end=` 시간축 불일치

### 증상
- 신호 발생 당일의 VIX 데이터가 누락 → 레짐 분류가 어제 VIX 기준.
- 특히 패닉 day(VIX 급등)에서 신호가 발생할 때 정확도 저하.

### 원인
[`v4wp_realtime/core/scanner.py:78-86`](v4wp_realtime/core/scanner.py)
```python
import yfinance as yf
vix_df = yf.download('^VIX', period=f'{params.get("data_years", 3)}y', progress=False)
```
- VIX만 `period=` 모드 → end 미지정 → 가장 최근 데이터 포함.
- 주식은 [`indicators.py:fetch_data`](v4wp_realtime/core/indicators.py) → end exclusive (D-3 적용 전 기준).
- D-3 수정 후엔 자동으로 정렬되지만, 코드 일관성을 위해 VIX도 같은 진입점 사용 권장.

### 수정

**`v4wp_realtime/core/scanner.py:77-86`** — `fetch_data` 사용으로 통일
```python
try:
    vix_df = fetch_data('^VIX', years=params.get('data_years', 3))
    if vix_df is not None and len(vix_df) >= 40:
        vix_close = vix_df['Close']
        if hasattr(vix_close, 'columns'):
            vix_close = vix_close.iloc[:, 0]
        vix_change_20d_series = vix_close / vix_close.shift(20) - 1.0
except Exception as e:
    print(f'  [scanner] VIX fetch failed: {e}')
```

### 검증
- 스캔 결과 `signal_data['vix_change_20d']`가 신호 당일 시점의 값과 일치 (직전 어제 값이 아닌).

### 효과
- D-3와 결합 시 레짐 분류가 신호 발생 당일 정보 완전 반영.

---

## M-1. blocked_buys 종목명 미표기

### 증상
- 일일 리포트에 "🚫 DD게이트 차단: 3건" 표시만, **어떤 종목이 차단됐는지** 알 수 없음.
- DD 임박 종목 다음날 모니터링 불가.

### 원인
[`v4wp_realtime/alerts/message_formatter.py:267-268`](v4wp_realtime/alerts/message_formatter.py)
```python
if n_blocked > 0:
    lines.append(f'\U0001f6ab DD\uac8c\uc774\ud2b8 \ucc28\ub2e8: {n_blocked}\uac74')
```

### 수정
```python
if n_blocked > 0:
    blocked = results.get('blocked_buys', [])
    tickers = ', '.join(f'<b>{b["ticker"]}</b>({b.get("dd_pct",0)*100:.1f}%)'
                        for b in blocked[:10])
    lines.append(f'\U0001f6ab DD게이트 차단 {n_blocked}건: {tickers}')
```

### 검증
- 다음 차단 발생 시 카드에 종목 코드 + DD% 동반 표시 확인.

### 효과
- 차단 종목이 "임박"(DD 2.5% 등) 상태였음을 사용자가 인지 → 다음날 진입 타이밍 모니터링 가능.

---

## M-2. AI interpretation (멀티 페르소나) Telegram 미노출

### 증상
- Anthropic API 비용 들여서 생성한 `interpretation` JSON이 Mini App에서만 보이고, Telegram 메인 알림에는 사용되지 않음.
- Mini App을 안 여는 사용자는 페르소나 분석을 영원히 못 봄.

### 원인
[`v4wp_realtime/alerts/telegram_bot.py:329-354`](v4wp_realtime/alerts/telegram_bot.py)의 `send_signal_alert`에서 `signal['interpretation']` 참조 없음.

[`v4wp_realtime/alerts/message_formatter.py:format_signal_message`](v4wp_realtime/alerts/message_formatter.py)도 `commentary`만 카드에 포함.

### 수정 방향
`format_signal_message`에 interpretation 요약 1~2줄 추가:

```python
# format_signal_message 마지막 부분
interp_str = signal.get('interpretation')
if interp_str:
    try:
        import json as _json
        interp = _json.loads(interp_str) if isinstance(interp_str, str) else interp_str
        verdict = interp.get('final_verdict', '')
        conf = interp.get('confidence_score', 0)
        chairman_summary = interp.get('chairman', {}).get('summary', '')[:60]
        if verdict:
            verdict_kr = {
                'STRONG_BUY': '🔥 강력 매수',
                'BUY': '✅ 매수',
                'CAUTIOUS_BUY': '⚠ 신중 매수',
                'HOLD': '⏸ 관망',
            }.get(verdict, verdict)
            lines.append('')
            lines.append(f'\U0001f3db <b>{verdict_kr}</b> (확신도 {conf}/100)')
            if chairman_summary:
                lines.append(f'\u2937 <i>{chairman_summary}</i>')
    except Exception:
        pass
```

### 검증
- AI interpretation이 생성된 신호에서 카드에 "🏛 매수 (확신도 78/100)" 줄이 보이는지 확인.

### 효과
- AI 호출 비용 대비 가치 회수.
- Mini App 의존성 감소.

---

## M-3. 카드에 레짐 컨텍스트 없음

### 증상
- 백테스트 검증상 BEAR_STRONG (승률 67.9%, 90d +35.2%) vs BULL_STRONG (52.6%, +7.9%)는 **완전히 다른 품질**의 신호인데, 카드에는 같은 "🟢 BUY"로 보임.

### 원인
[`v4wp_realtime/alerts/message_formatter.py:30-69`](v4wp_realtime/alerts/message_formatter.py)의 `format_signal_message`가 `signal['market_regime']`을 참조하지 않음. ([`scanner.py:207`](v4wp_realtime/core/scanner.py)에서 채워지긴 함)

### 수정
`format_signal_message`에 레짐 라인 추가 (Force/Div 라인 다음):

```python
regime = signal.get('market_regime', 'UNKNOWN')
if regime != 'UNKNOWN':
    from v4wp_realtime.core.regime import get_conviction
    conv = get_conviction(regime)
    mkt = signal.get('market_return_20d')
    mkt_str = f'{mkt*100:+.1f}%' if mkt is not None else 'N/A'
    lines.append(f'\U0001f30d 레짐  <b>{conv["label_kr"]}</b>  (QQQ 20d {mkt_str})')
```

### 검증
- 카드에 "🌍 레짐 공포 극대 → 역발상 매수 (QQQ 20d -3.2%)" 같은 줄이 추가되는지 확인.

### 효과
- 동일 스코어 신호라도 레짐별로 사용자가 다른 비중을 결정할 수 있음.
- 백테스트에서 검증된 conviction 차이가 UI에 노출.

---

## UX-1. Mini App에서 옛 신호가 현재 신호처럼 무한 노출

### 증상
워치리스트 종목 칩의 **LONG 뱃지**와 종목 상세의 **AI SIGNAL INTERPRETATION** 카드(페르소나 위원회)가, 한 번 신호가 발생하면 **다음 신호가 같은 종목에 또 뜰 때까지 영원히 표시**됩니다.

구체적으로:
- 90일 전 신호의 LONG 뱃지가 오늘도 풀 컬러로 보임.
- 1년 전 페르소나 분석(`STRONG_BUY`, `Confidence 78%`)이 오늘도 매수 의사결정에 사용될 수 있음.
- `signal_tier === "CONFIRMED"` 면 회색/페이드도 안 됨. PENDING은 DB에 저장조차 안 되므로([`scanner.py:146-147`](v4wp_realtime/core/scanner.py:146)) 모든 저장 신호는 풀 컬러로 영원히 살아 있음.

### 원인
두 위치 모두 **시간 필터 없는** `ORDER BY peak_date DESC LIMIT 1` 패턴을 사용:

**1) LONG 뱃지 데이터 소스** — [`export_static.py:72-78`](v4wp_realtime/scripts/export_static.py:72)
```sql
SELECT signal_type, peak_date, signal_tier, s_force, peak_val
FROM signal_events
WHERE ticker = ?
ORDER BY peak_date DESC LIMIT 1
```

**2) AI 위원회 데이터 소스** — [`export_static.py:292-298`](v4wp_realtime/scripts/export_static.py:292)
```sql
SELECT interpretation, peak_date, signal_type, signal_tier
FROM signal_events
WHERE ticker = ? AND interpretation IS NOT NULL
ORDER BY peak_date DESC LIMIT 1
```

UI 측에서도 신호 나이를 무시:
- [`WatchlistBar.jsx:100-105`](v4wp_realtime/webapp/src/components/WatchlistBar.jsx:100) — `item.recent_signal` truthy면 무조건 렌더
- [`AIInterpretation.jsx:45`](v4wp_realtime/webapp/src/components/AIInterpretation.jsx:45) — `if (loading || !data?.interpretation) return null;`
- [`SignalBadge.jsx:2-5`](v4wp_realtime/webapp/src/components/SignalBadge.jsx:2) — `tier === "CONFIRMED"`만 체크, age 무관

다른 곳에서는 시간 윈도를 의식적으로 두는데(`postmortem.py` 90일, `similarity.py` 365일, `extract_recent_events` 10일) **watchlist/interpretation 추출만 누락**.

### 수정 방향 — Option B (신호 나이 노출 + UI에서 페이드)

> 데이터 자체는 보존하고 UI에서 신선도를 시각화. 신호 나이 정보가 매수 결정에 중요한 요소이므로 명시적으로 노출.

#### 1) Backend — 신호 나이 필드 추가

**`v4wp_realtime/scripts/export_static.py:72-103`** — `export_watchlist` 수정
```python
sig_row = conn.execute(
    """SELECT signal_type, peak_date, signal_tier, s_force, peak_val,
              julianday('now') - julianday(peak_date) AS age_days
       FROM signal_events
       WHERE ticker = ?
       ORDER BY peak_date DESC LIMIT 1""",
    (ticker,),
).fetchone()

recent_signal = None
if sig_row:
    s = dict(sig_row)
    recent_signal = {
        'direction': 'LONG' if s['signal_type'] == 'bottom' else 'SHORT',
        'peak_date': s['peak_date'],
        'tier': s['signal_tier'],
        's_force': _safe(s['s_force']),
        'peak_val': _safe(s['peak_val']),
        'age_days': int(s['age_days']) if s['age_days'] is not None else None,  # ← 추가
    }
```

**`v4wp_realtime/scripts/export_static.py:290-318`** — `export_interpretation` 수정
```python
row = conn.execute(
    """SELECT interpretation, peak_date, signal_type, signal_tier,
              julianday('now') - julianday(peak_date) AS age_days
       FROM signal_events
       WHERE ticker = ? AND interpretation IS NOT NULL
       ORDER BY peak_date DESC LIMIT 1""",
    (ticker,),
).fetchone()

# ... 기존 처리 ...
result = {
    'ticker': ticker,
    'peak_date': row['peak_date'],
    'signal_type': row['signal_type'],
    'signal_tier': row['signal_tier'],
    'age_days': int(row['age_days']) if row['age_days'] is not None else None,  # ← 추가
    'interpretation': interp,
}
```

#### 2) Frontend — 신선도 등급 정책

| age_days | LONG 뱃지 | AI 위원회 카드 |
|---|---|---|
| 0~5일 (FRESH) | 🟢 풀 컬러 LONG | 🟢 풀 컬러 카드 |
| 6~14일 (RECENT) | 🟡 라벨 옆 "5d" 표시, 약간 페이드 | 🟡 상단 "5일 전" 노란 배지 |
| 15~30일 (STALE) | ⚪ 회색 + opacity 0.5 + "12d" | ⚪ "12일 전 — 참고용" 회색 배지 + 카드 opacity 0.6 |
| 31일+ (EXPIRED) | ❌ 뱃지 미표시 | ❌ 카드 미렌더 (`return null`) |

#### 3) Frontend — `SignalBadge.jsx` 수정

```jsx
export default function SignalBadge({ direction, tier, ageDays }) {
  const passed = tier === "CONFIRMED";

  // 31일 초과 시 렌더 안 함
  if (ageDays != null && ageDays > 30) return null;

  const fresh = ageDays == null || ageDays <= 5;
  const recent = ageDays != null && ageDays > 5 && ageDays <= 14;
  const stale = ageDays != null && ageDays > 14;

  const baseOpacity = !passed ? 0.5 : stale ? 0.5 : recent ? 0.75 : 1;
  const bg = !passed || stale ? "#2a2a35"
           : direction === "LONG" ? "#0d3320" : "#3d1320";
  const fg = !passed || stale ? "#888"
           : direction === "LONG" ? "#34d399" : "#f87171";
  const border = !passed || stale ? "#333"
               : direction === "LONG" ? "#166534" : "#7f1d1d";

  return (
    <span style={{
      display: "inline-block",
      padding: "2px 6px",
      borderRadius: 4,
      fontSize: 10,
      fontWeight: 700,
      background: bg,
      color: fg,
      border: `1px solid ${border}`,
      letterSpacing: 0.5,
      fontFamily: "'JetBrains Mono', monospace",
      opacity: baseOpacity,
      whiteSpace: "nowrap",
    }}>
      {direction}
      {!passed && " \u2715"}
      {ageDays != null && ageDays > 1 && (
        <span style={{ marginLeft: 4, opacity: 0.7, fontSize: 9 }}>
          {ageDays}d
        </span>
      )}
    </span>
  );
}
```

#### 4) Frontend — `WatchlistBar.jsx:100-105` 수정

```jsx
{item.recent_signal && (
  <SignalBadge
    direction={item.recent_signal.direction}
    tier={item.recent_signal.tier}
    ageDays={item.recent_signal.age_days}    // ← 추가
  />
)}
```

#### 5) Frontend — `AIInterpretation.jsx` 수정

```jsx
// L45 부근 — 31일 초과 시 렌더 안 함
if (loading || !data?.interpretation) return null;
if (data.age_days != null && data.age_days > 30) return null;

const interp = data.interpretation;
const verdict = VERDICT_STYLE[interp.final_verdict] || VERDICT_STYLE.HOLD;
const ageDays = data.age_days;
const isStale = ageDays != null && ageDays > 14;
const isRecent = ageDays != null && ageDays > 5 && ageDays <= 14;

return (
  <div style={{
    background: "var(--tg-section-bg)",
    borderRadius: 10,
    padding: "12px 10px 8px",
    border: "1px solid rgba(255,255,255,0.06)",
    opacity: isStale ? 0.6 : 1,        // ← 추가
  }}>
    {/* Age Warning Badge */}
    {ageDays != null && ageDays > 5 && (
      <div style={{
        display: "inline-block",
        padding: "2px 8px",
        borderRadius: 4,
        marginBottom: 8,
        fontSize: 10,
        fontWeight: 700,
        background: isStale ? "rgba(136,136,136,0.15)" : "rgba(251,191,36,0.15)",
        color: isStale ? "#888" : "#fbbf24",
        border: `1px solid ${isStale ? "#444" : "#92400e40"}`,
        fontFamily: mono,
      }}>
        {ageDays}일 전 신호 {isStale && "— 참고용"}
      </div>
    )}

    {/* Label */}
    <div style={{
      fontSize: 10,
      color: "var(--tg-hint)",
      marginBottom: 8,
      ...
    }}>
      AI SIGNAL INTERPRETATION
      {data.peak_date && (
        <span style={{ marginLeft: 6, opacity: 0.7 }}>{data.peak_date.slice(5)}</span>
      )}
    </div>

    {/* ... 나머지는 그대로 ... */}
  </div>
);
```

### 검증
1. **단위 테스트**: 임의의 시그널을 `peak_date` 6/15/35일 전으로 DB에 넣고 export 실행 후 결과 JSON에 `age_days: 6 / 15 / 35` 가 박혀 있는지 확인.
2. **시각 검증**:
   - 5일 전 신호: 풀 컬러
   - 10일 전 신호: 노란 "10d" 라벨 + 약간 페이드
   - 20일 전 신호: 회색 + opacity 0.5
   - 35일 전 신호: 뱃지 안 보임, AI 위원회 카드 미렌더
3. **시계열 시뮬레이션**: 동일 신호를 `julianday('now')` 대신 고정 날짜로 두고 1~40일 시뮬레이션해서 단계 전환 확인.

### 효과
- **의사결정 품질 향상**: 5일 전 신호와 어제 신호가 시각적으로 구분 → 사용자가 신선도를 인지한 상태로 매수 결정.
- **정보 손실 없음**: 데이터는 DB/JSON에 그대로 보존, UI 표현만 변경. 30일 이내라면 누구나 신호 정보 접근 가능.
- **decay 시스템과 자연 결합**: 이미 [`run_decay_analysis`](v4wp_realtime/core/postmortem.py:178)가 5거래일 후 `FADING`/`FALSE_POSITIVE`를 판정함. 추후 `decay_class`도 export에 포함하면 "10일 전 신호인데 FADING" 같은 컨텍스트도 노출 가능.
- **AI 비용 가치 회수**: AI 위원회 분석이 30일 이내에만 표시 → 1년 전 분석을 오늘 신호로 오해하는 사례 차단.

### 후속 개선 (선택)
- `signals_history.json`이 무한 누적되면([S-1](#s-1-signals_historyjson-보존-정책)) DB 복원 시 옛 신호도 다 들어와 쿼리가 느려짐. UX-1로 화면에선 안 보이지만 DB 부담은 그대로 → S-1 함께 처리 권장.
- 색상 임계값(5/14/30일)은 매수 전략에 따라 조정. 단기 매매면 3/7/14일, 중기면 현재 값, 장기면 7/21/60일.

---

# 🟢 Tier 2 — 구조 정리 (LOC 감축)

## DC-1. `_send_media_group` 데드 함수 제거

### 근거
[`v4wp_realtime/alerts/telegram_bot.py:100-130`](v4wp_realtime/alerts/telegram_bot.py)에 31줄 정의되어 있으나 호출처 0.
`send_signal_album`([L357](v4wp_realtime/alerts/telegram_bot.py))은 단일은 `_send_photo`, 다중은 `_send_message`(요약 텍스트)로 처리. 앨범 전송 코드 경로가 존재하지 않음.

### 수정
함수 전체 삭제. `TELEGRAM_API_MEDIA_GROUP` 상수([L38](v4wp_realtime/alerts/telegram_bot.py))도 함께 제거.

---

## DC-2. `_summary_button` 데드 함수 제거

### 근거
[`v4wp_realtime/alerts/telegram_bot.py:238-239`](v4wp_realtime/alerts/telegram_bot.py)에 정의. `s:` 콜백 핸들러([L504-537](v4wp_realtime/alerts/telegram_bot.py))는 존재하나, 이 버튼을 키보드에 부착하는 코드는 어디에도 없음.

### 수정
함수 삭제. `handle_callback`의 `s:` 분기도 DC-3와 함께 제거.

---

## DC-3. 콜백 시스템 전체 제거

### 근거
- [`v4wp_realtime/scripts/callback_bot.py`](v4wp_realtime/scripts/callback_bot.py) — long-poll 호스팅 필요.
- [`.github/workflows/daily_scan.yml`](.github/workflows/daily_scan.yml) — 워크플로에 호스팅 단계 없음, cron만 존재.
- 사용자 환경에 별도 서버/머신을 두지 않는 한 콜백은 **물리적으로 동작 불가**.
- Mini App이 모든 상호작용을 대체.

### 수정 범위
**삭제할 코드:**
1. [`v4wp_realtime/scripts/callback_bot.py`](v4wp_realtime/scripts/callback_bot.py) — 파일 전체 (25줄)
2. [`v4wp_realtime/alerts/telegram_bot.py`](v4wp_realtime/alerts/telegram_bot.py):
   - `_send_message`/`_send_photo` 외 `_edit_*`, `_delete_message`, `_answer_callback` (L137-185)
   - `_detail_button`, `_compact_button`, `_close_button` (L230-243)
   - `_save_chart`, `_load_chart` (L265-281)
   - `save_callback_data`, `_load_callback_store` (L288-322)
   - `handle_callback`, `run_callback_handler` (L426-590)
   - 모든 API 상수 중 콜백 관련 (`_EDIT_TEXT`, `_EDIT_CAPTION`, `_DELETE`, `_ANSWER_CB`, `_UPDATES`)
   - `send_signal_album`에서 `save_callback_data` 호출 제거 (L403, L417)
3. `import io`, `import time` 사용처 점검 (`io.BytesIO`가 다른 곳에 쓰이면 유지)

**유지할 코드:**
- `_dashboard_button`, `_keyboard`, `_miniapp_url` — Mini App 진입 버튼은 유지
- `send_signal_alert`, `send_signal_album`, `send_watch_alert`, `send_market_event_alert`, `send_scan_summary` — 메인 알림 경로

### 검증
- `grep -r "handle_callback\|callback_bot\|callback_data" v4wp_realtime/` → 결과 없음
- 다음 스캔에서 Telegram 알림이 정상 전송되는지 확인.

### 효과
- 약 200줄 감축.
- `data/callback_data.json`, `data/charts/` 의 누적 문제 자동 해소 (S-2).
- "이 봇 콜백 왜 안 되지" 라는 혼란 제거.

---

## DC-4. FastAPI 서버 전체 제거

### 근거
- [`v4wp_realtime/api/routes.py`](v4wp_realtime/api/routes.py) (477줄)의 모든 엔드포인트가 [`v4wp_realtime/scripts/export_static.py`](v4wp_realtime/scripts/export_static.py) (398줄)에서 동일 데이터를 정적 JSON으로 생성.
- 배포 모델은 GitHub Pages 정적 (워크플로 `peaceiris/actions-gh-pages@v4` 단계).
- [`webapp/src/api.js:10`](v4wp_realtime/webapp/src/api.js): `IS_STATIC = !BASE && import.meta.env.PROD` → 프로덕션은 항상 정적 모드.

### 수정 범위
**삭제:**
1. [`v4wp_realtime/api/routes.py`](v4wp_realtime/api/routes.py) (477줄)
2. [`v4wp_realtime/scripts/run_api.py`](v4wp_realtime/scripts/run_api.py) (37줄)
3. [`v4wp_realtime/api/__init__.py`](v4wp_realtime/api/__init__.py) (DC-5와 함께 디렉토리 자체 제거 가능)

**`requirements.txt`** — `fastapi`, `uvicorn`, `starlette` 의존성 제거 (다른 곳에 사용 없는지 확인).

**유지 옵션:**
- 로컬 개발용 핫리로드가 필요하면 `python -m http.server` + `vite dev`로 대체 가능.

### 검증
- `grep -r "from v4wp_realtime.api" v4wp_realtime/` → DC-5 (event_bus) 외 결과 없음
- 다음 배포에서 Mini App이 정상 동작 확인.

### 효과
- 515줄 감축.
- "이 라우트는 export_static과 어떻게 다르지?" 혼란 제거.

---

## DC-5. SSE 인프라 제거

### 근거
- [`v4wp_realtime/api/event_bus.py`](v4wp_realtime/api/event_bus.py)의 `_subscribers`는 in-process set.
- daily_scan.py는 별도 프로세스 → publish가 빈 set에 발행 → no-op.
- 클라이언트 [`webapp/src/api.js:65-67`](v4wp_realtime/webapp/src/api.js)는 정적 모드에서 즉시 close.
- GitHub Pages 배포에선 publisher/subscriber 둘 다 도달 불가.

### 수정 범위
**삭제:**
1. [`v4wp_realtime/api/event_bus.py`](v4wp_realtime/api/event_bus.py) (80줄)
2. [`v4wp_realtime/data/store.py:7-13, 81-83, 109-115`](v4wp_realtime/data/store.py) — `_try_publish` 호출 전부
3. [`v4wp_realtime/webapp/src/api.js:59-104`](v4wp_realtime/webapp/src/api.js) — `connectSSE` 전체
4. [`v4wp_realtime/webapp/src/App.jsx:39-67`](v4wp_realtime/webapp/src/App.jsx) — SSE useEffect 블록

### 검증
- `grep -r "_try_publish\|connectSSE\|event_bus" v4wp_realtime/` → 결과 없음
- Mini App이 정상 로드되는지 확인 (SSE 의존이 사라져도 초기 fetch는 동작).

### 효과
- 80줄 + 60줄 감축.
- "왜 실시간 업데이트가 안 되지?" 혼란 제거.
- DC-4와 함께 `v4wp_realtime/api/` 디렉토리 통째로 제거 가능.

---

## S-3. postmortem/decay 결과 JSON 직렬화

### 증상
- 매 GitHub Actions 실행마다 `run_postmortem` / `run_decay_analysis`가 **모든 NULL 신호**를 재계산.
- 결정론적이라 결과는 같지만, 200+ 신호 × 5/20/90d 쿼리 = 매 스캔 +수백 SQL.

### 원인
[`v4wp_realtime/core/scanner.py:149-176`](v4wp_realtime/core/scanner.py)의 `signal_data` 딕셔너리에 다음 필드 없음:
- `return_5d`, `return_20d`, `return_90d`
- `max_dd_90d`
- `postmortem`
- `decay_class`, `score_5d_avg`

[`v4wp_realtime/scripts/daily_scan.py:46-71`](v4wp_realtime/scripts/daily_scan.py)의 `restore_signals_to_db` 도 이 필드를 JSON에서 꺼내지 않음 → 그래서 매 실행마다 NULL 상태로 시작 → 재계산.

### 수정

**1. `daily_scan.py:restore_signals_to_db`** — 직렬화된 필드 복원
```python
event = {
    # ... 기존 필드 ...
    'return_5d': s.get('return_5d'),
    'return_20d': s.get('return_20d'),
    'return_90d': s.get('return_90d'),
    'max_dd_90d': s.get('max_dd_90d'),
    'postmortem': s.get('postmortem'),
    'decay_class': s.get('decay_class'),
    'score_5d_avg': s.get('score_5d_avg'),
}
```

**2. `daily_scan.py`** — 스캔 + postmortem 완료 후, JSON 저장 전에 DB에서 최신 값 다시 읽어 history에 머지
```python
def main():
    ...
    # 기존 단계 7 직전에 추가:
    # postmortem이 채운 필드를 history에 반영
    pm_conn = get_connection()
    for sig in history:
        row = pm_conn.execute(
            """SELECT return_5d, return_20d, return_90d, max_dd_90d,
                      postmortem, decay_class, score_5d_avg
               FROM signal_events
               WHERE ticker=? AND signal_type=? AND peak_date=?""",
            (sig['ticker'], sig['signal_type'], sig['peak_date'])
        ).fetchone()
        if row:
            for k in row.keys():
                if row[k] is not None:
                    sig[k] = row[k]
    pm_conn.close()

    # 그 다음에 새 시그널 추가하고 save_signal_history(history, results['new_signals'])
```

**3. `insert_signal_event` SQL** — 필드가 누락되지 않도록 [`store.py:90-119`](v4wp_realtime/data/store.py)에 `return_*`, `max_dd_90d`, `postmortem`, `decay_class`, `score_5d_avg` 컬럼 추가 (이미 schema에는 있음, INSERT 문에만 추가)

### 검증
- 2일 연속 스캔 후 `data/signals_history.json` 의 오래된 신호에 `return_5d`, `return_90d` 값이 누적되어 있는지 확인.
- Actions 로그에서 `Post-Mortem: 5d=N, 20d=M, 90d=K`의 N/M/K 합이 점진적으로 줄어드는지 (=재계산 감소) 확인.

### 효과
- 매 스캔당 ~수백 SQL 쿼리 절감.
- Mini App의 postmortem 데이터가 일관성 보장 (매 스캔마다 같은 값으로 재생산 ≠ 영속화된 값과는 다름).

---

# ⚪ Tier 3 — 백로그

## P-1. interpretation을 CONFIRMED tier에만 호출

### 근거
[`v4wp_realtime/core/scanner.py:146-147`](v4wp_realtime/core/scanner.py)에서 PENDING은 `continue`로 스킵되지만, 그 직후 [L260-269](v4wp_realtime/core/scanner.py)의 `interpretation_fn`은 무조건 호출.

실제로는 PENDING이 위에서 걸러져서 영향이 없을 수 있음. 코드 확인 필요.

**현재 동작 검증:**
```bash
grep -n "PENDING\|interpretation_fn" v4wp_realtime/core/scanner.py
```

만약 모든 신호가 CONFIRMED만 진입한다면 P-1 자체가 불필요. 우선 검증 후 결정.

---

## P-2. postmortem 시간 필터

### 수정
[`v4wp_realtime/core/postmortem.py:17-23`](v4wp_realtime/core/postmortem.py)의 pending 쿼리에 시간 상한 추가:

```python
pending = conn.execute(
    """SELECT id, ticker, peak_date, close_price, signal_type,
              return_5d, return_20d, return_90d, postmortem, interpretation
       FROM signal_events
       WHERE (return_5d IS NULL OR return_20d IS NULL OR return_90d IS NULL)
         AND close_price > 0
         AND peak_date >= date('now', '-120 days')"""  -- ← 추가
).fetchall()
```

90거래일이 지나면 더 이상 업데이트할 게 없으므로 120일 전 신호는 스킵.

---

## P-3. ETF/VIX 다운로드 병렬화

### 수정
[`v4wp_realtime/core/scanner.py:67-86`](v4wp_realtime/core/scanner.py)을 `concurrent.futures.ThreadPoolExecutor`로 묶음:

```python
from concurrent.futures import ThreadPoolExecutor

context_tickers = ['QQQ', '^VIX'] + list(set(v for v in SECTOR_ETF_MAP.values() if v))
with ThreadPoolExecutor(max_workers=4) as ex:
    futures = {t: ex.submit(fetch_data, t, params.get('data_years', 3))
               for t in context_tickers}
    results = {t: f.result() for t, f in futures.items()}

# 이후 results['QQQ'], results['^VIX'], results['XLK'] 등으로 사용
```

---

## S-1. signals_history.json 보존 정책

### 수정 방향
스캔 후 `save_signal_history`에서 90일 지난 신호는 별도 아카이브로 분리:

```python
def save_signal_history(existing, new_signals):
    from datetime import datetime, timedelta
    cutoff = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    all_signals = existing + new_signals
    active = [s for s in all_signals if s.get('peak_date', '') >= cutoff]
    archived = [s for s in all_signals if s.get('peak_date', '') < cutoff]

    SIGNALS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SIGNALS_JSON, 'w', encoding='utf-8') as f:
        json.dump(active, f, indent=2, default=str, ensure_ascii=False)

    if archived:
        archive_path = SIGNALS_JSON.parent / 'signals_archive.jsonl'
        with open(archive_path, 'a', encoding='utf-8') as f:
            for s in archived:
                f.write(json.dumps(s, default=str, ensure_ascii=False) + '\n')

    return len(active)
```

JSONL 아카이브는 append-only라 git 충돌 위험 낮음.

---

## Q-2. 한글 escape sequence 평문화

### 수정
[`message_formatter.py:161, 164, 268, 275`](v4wp_realtime/alerts/message_formatter.py)의 `\ub0a8\uc74c`, `\ud544\uc694` 등을 그냥 한글로:

```python
# Before
f'{gap:.1f}%p</b> \ub0a8\uc74c)'
# After
f'{gap:.1f}%p</b> 남음)'
```

기능 동일, 검색성/가독성 향상.

---

## Q-3. `start_val` vs `peak_val` 라벨 명확화

### 수정
[`message_formatter.py:9, 34, 83`](v4wp_realtime/alerts/message_formatter.py) 등에서 "Score" 라벨 옆에 표시되는 값이 `start_val`임을 명확히. 가능하면 두 값 모두 표시:

```python
'📊 스코어 시작 <code>{start_val:+.3f}</code> → 정점 <code>{peak_val:+.3f}</code>'
```

또는 한 값만 보여줄 거면 `peak_val`이 더 자연스러움 (이벤트 강도의 대표값).

---

## Q-4. naive `datetime.now()` 명시화

### 수정
[`scanner.py:48, 324`](v4wp_realtime/core/scanner.py), [`indicators.py:26`](v4wp_realtime/core/indicators.py) 등을 `datetime.now(timezone.utc)`로 통일. Actions는 UTC라 결과는 같지만 의도가 명확해짐.

---

## Q-5. 워크플로 git pull 전략 검토

### 근거
[`.github/workflows/daily_scan.yml:50`](.github/workflows/daily_scan.yml):
```yaml
git pull --rebase -X theirs origin main || (git rebase --abort && git pull --no-rebase origin main)
```

cron 1회/일 환경에선 안전하지만 `workflow_dispatch` 동시 실행 시 위험. concurrency group 추가 권장:

```yaml
concurrency:
  group: daily-scan-${{ github.workflow }}
  cancel-in-progress: false
```

워크플로 최상단에 추가하면 동시 실행 자체를 막아줌.

---

# 📦 권장 PR 묶음

각 PR은 독립적으로 mergeable하도록 설계.

## PR #1 — Tier 0 버그 픽스 + Q-1
- D-1, D-2, D-3, Q-1 한 묶음.
- 모두 작은 변경 + 즉시 가치.
- **단위 테스트 1개 필수**: `fetch_data('AAPL')`의 마지막 날짜가 미국장 기준 오늘인지 검증.

## PR #2 — Tier 1 메시지 개선
- D-4, M-1, M-2, M-3.
- 텔레그램 알림 카드 품질 직접 향상.

## PR #3 — 콜백/API/SSE 제거
- DC-1, DC-2, DC-3, DC-4, DC-5.
- 약 700~800 LOC 감축.
- Mini App 동작 확인 후 머지.

## PR #4 — postmortem 영속화
- S-3.
- 단독 PR (스키마 영향 가능성).

## PR #5 — 백로그 (P/S/Q)
- 우선순위에 따라 분리.

---

# ✅ 머지 전 체크리스트

각 PR 머지 전 다음을 확인:

- [ ] 로컬 dry-run: `python v4wp_realtime/scripts/daily_scan.py` 가 에러 없이 완료
- [ ] `grep -rn "TODO\|XXX\|FIXME" v4wp_realtime/` 결과가 PR과 무관
- [ ] `data/signals_history.json` 스키마 호환성 (PR이 새 필드 추가 시 기존 항목에 NULL 허용)
- [ ] 워크플로 dry-run: `act -W .github/workflows/daily_scan.yml` (act 설치 시)
- [ ] Telegram 카드 시각 확인 (실제 알림 1건 받아보기)
- [ ] Mini App 빌드: `cd v4wp_realtime/webapp && npm run build` 성공

---

# 📚 참고 — 검증 결과 요약 (전체 리뷰에서)

## 데이터 흐름 ✅ 정상
- JSON ↔ DB 라운드트립 (`restore_signals_to_db` → 스캔 → `save_signal_history`)
- DB `UNIQUE(ticker, signal_type, peak_date)` 제약 + `is_new_signal` ±3일 이중 방어
- 매 스캔 252일치 `daily_scores` 재구축 → postmortem/decay에 충분
- 레짐 분류 시 `pd.notna` 가드

## 데이터 흐름 ❌ 누수
- D-1 AI commentary (fallback only 추정)
- D-2 Mini App deep link
- D-3 yfinance end exclusive
- D-4 VIX 시간축 1일 불일치
- D-7 / S-3 postmortem 매 스캔 재계산 (이 문서에선 S-3로 통합)
- SSE 전 경로 (DC-5에서 제거)

## 의도된 동작 (수정 대상 아님)
- 매도 신호(`signal_type='top'`) 미사용 — SYSTEM_GUIDE 6장 백테스트 결과로 의도적 OFF
- Streamlit dashboard — 개발자 도구로 유지
- `is_new_signal` ±3일 + DB PK 이중방어 — 의도된 클러스터 압축

---

문서 끝.
