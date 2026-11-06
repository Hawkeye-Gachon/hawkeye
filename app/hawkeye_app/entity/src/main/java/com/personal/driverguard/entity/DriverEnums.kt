package com.personal.driverguard.entity

/** 분석 방식. 실제 분석 기록과 시연 기록은 저장·리포트에서 항상 구분한다. */
enum class AnalysisMode { REAL, DEMO }

/**
 * 화면에 보여 주는 종합 상태. 지표별 관측 가능 여부와 활성 판정을 보존한 채 표시용으로만 집계한다.
 * 어느 상태도 졸음이나 안전을 '확정'하지 않는다.
 */
enum class DisplayState {
    /** 캘리브레이션 확인·첫 유효 관측 대기 */
    PREPARING,

    /** 관찰 중(검출된 징후 없음 ≠ 안전 보장) */
    OBSERVING,

    /** 주의 징후 감지(지속 눈 감김, 시각적 부주의 의심) */
    CAUTION,

    /** 졸음 의심에 따른 휴식 권고 */
    REST_RECOMMENDED,

    /** 분석 불가 */
    UNAVAILABLE,

    /** 휴식 중이거나 앱이 화면에서 벗어나 분석이 멈춤 */
    PAUSED,
}

/** 주의 상태의 원인. 화면 안내 문구를 고르는 데 쓴다. */
enum class CautionCause { EYE_CLOSURE, HEAD_AWAY }

/** 졸음 판정 단계. 확률이 아니라 규칙 충족 여부다. */
enum class DrowsinessLevel {
    NONE,

    /** 졸음 의심: 눈 감김 비율 상승, 반복적 눈 감김, 또는 입 벌림 반복 + 눈 감김 */
    SUSPECTED,

    /** 졸음 의심 + 보조 징후(입 벌림 반복, 실험적 끄덕임)가 함께 관찰됨 */
    SUSPECTED_WITH_SUPPORT,
}

/** 이벤트·판정에 기록하는 단계. 선언 순서가 심각도 순서다. */
enum class AlertLevel { INFO, CAUTION, REST }

enum class EventType {
    /** 지속적 눈 감김(양쪽 눈, 유효 관측 연속) */
    EYE_CLOSURE,

    /** 지속적 입 벌림. 사용자에게는 '하품 의심'으로 표시. 말하기·웃음과 구분할 수 없다. */
    MOUTH_OPEN,

    /** 기준 자세 대비 고개 방향 이탈(시선 추적 아님) */
    HEAD_AWAY,

    /** 반복적 고개 숙임(실험) */
    HEAD_NOD,

    /** 분석 불가 구간 */
    ANALYSIS_UNAVAILABLE,

    /** 판정: 졸음 의심 */
    DROWSINESS_SUSPECTED,

    /** 판정: 시각적 부주의(전방 주시 이탈) 의심 */
    DISTRACTION_SUSPECTED,
    ;

    val isJudgment: Boolean get() = this == DROWSINESS_SUSPECTED || this == DISTRACTION_SUSPECTED
}

enum class EventEndReason {
    /** 지표가 정상 범위로 돌아옴 */
    RECOVERED,

    /** 프레임 공백으로 관측이 끊김(정상 복귀 아님) */
    OBSERVATION_GAP,

    /** 해당 지표를 계산할 수 없게 됨(정상 복귀 아님) */
    OBSERVATION_INVALID,

    /** 판정 근거가 유지 시간 동안 다시 나타나지 않음 */
    HOLD_EXPIRED,
    REST_STARTED,
    APP_PAUSED,
    TRIP_ENDED,

    /** 앱이 비정상 종료되어 다음 실행 때 정리됨 */
    ABNORMAL_TERMINATION,
}

/** 분석 중단 사유 */
enum class InterruptionReason { REST, APP_BACKGROUND, CAMERA_ERROR }

/** 재생할 경고. priority 가 높을수록 재생 중인 낮은 경고를 끊고 재생한다. */
enum class AlertKind(val priority: Int, val withTone: Boolean) {
    UNAVAILABLE(priority = 1, withTone = false),
    HEAD_CAUTION(priority = 2, withTone = true),
    EYE_CAUTION(priority = 3, withTone = true),
    REST(priority = 4, withTone = true),
}
