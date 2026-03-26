"""매크로 레짐 분류 + 역발상 확신도 매핑.

QQQ 20일 수익률 + VIX 20일 변화율로 시장 환경을 4단계로 분류하고,
역발상 논리에 따라 확신도(conviction)를 부여한다.

백테스트 검증 결과 (461 시그널, 2020-2026):
  BEAR_STRONG → HIGH:    승률 67.9%, 90d +35.2%, MaxDD -14.6%
  BEAR_WEAK   → MID:     승률 63.7%, 90d +19.0%, MaxDD -17.7%
  BULL_WEAK   → LOW:     승률 52.5%, 90d +22.7%, MaxDD -19.0%
  BULL_STRONG → CAUTION: 승률 52.6%, 90d  +7.9%, MaxDD -21.0%
"""


def classify_regime(qqq_ret20, vix_change_20d):
    """매크로 레짐 4단계 분류.

    Args:
        qqq_ret20: QQQ 20일 수익률 (소수, e.g. -0.022 = -2.2%)
        vix_change_20d: VIX 20일 변화율 (소수, e.g. 0.24 = +24%)

    Returns:
        str: BULL_STRONG | BULL_WEAK | BEAR_WEAK | BEAR_STRONG | UNKNOWN
    """
    if qqq_ret20 is None or vix_change_20d is None:
        return 'UNKNOWN'
    if qqq_ret20 < -0.05 or vix_change_20d > 0.30:
        return 'BEAR_STRONG'
    if qqq_ret20 < 0:
        return 'BEAR_WEAK'
    if qqq_ret20 > 0.05 and vix_change_20d < 0.10:
        return 'BULL_STRONG'
    return 'BULL_WEAK'


# 역발상 conviction 매핑
CONTRARIAN_MAP = {
    'BEAR_STRONG': {
        'conviction': 'HIGH',
        'weight': 1.0,
        'label_kr': '공포 극대 → 역발상 매수',
        'color': '#34d399',
    },
    'BEAR_WEAK': {
        'conviction': 'MID',
        'weight': 0.8,
        'label_kr': '약세장 매수 기회',
        'color': '#60a5fa',
    },
    'BULL_WEAK': {
        'conviction': 'LOW',
        'weight': 0.6,
        'label_kr': '상승장 평균 기회',
        'color': '#fbbf24',
    },
    'BULL_STRONG': {
        'conviction': 'CAUTION',
        'weight': 0.4,
        'label_kr': '과열 구간 주의',
        'color': '#f87171',
    },
    'UNKNOWN': {
        'conviction': 'MID',
        'weight': 0.6,
        'label_kr': '데이터 부족',
        'color': '#9ca3af',
    },
}


def get_conviction(regime):
    """레짐에서 역발상 확신도 정보를 반환."""
    return CONTRARIAN_MAP.get(regime, CONTRARIAN_MAP['UNKNOWN'])
