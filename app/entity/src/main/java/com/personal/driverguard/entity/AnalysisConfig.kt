package com.personal.driverguard.entity

import kotlinx.serialization.Serializable

/**
 * 분석·판정·경고에 쓰는 모든 임계값과 시간 창. 시간은 단조 시계 기준 밀리초다.
 *
 * 주의: 모든 기본값은 동작 확인을 위한 개발 시작값이다. 논문이나 실차 데이터로 검증된 안전 기준이 아니다.
 * 정차 환경에서 오경고·미검출·감지 지연을 측정해 다시 정해야 한다.
 * "보정 필요" 표시가 있는 값은 사람·카메라에 따라 차이가 커서 캘리브레이션 기준값에 상대적으로 쓰거나 개별 조정이 필요하다.
 */
@Serializable
data class AnalysisConfig(
    // ── 프레임 ──
    /** 카메라 프레임 중 분석에 제출하는 최대 빈도. 카메라 FPS 와 별개다. 10~15fps 를 개발 목표로 잡았다. */
    val targetAnalysisFps: Int = 12,
    /** 화면 지표 갱신 간격. 분석 빈도보다 길게 잡아 불필요한 recomposition 을 줄인다. */
    val uiRefreshMs: Long = 250,
    /** 이웃한 두 관측 사이가 이 간격 이하일 때만 연속 관측으로 잇는다. 더 길면 그 사이는 '관측 없음'이다. */
    val maxContinuityGapMs: Long = 300,

    // ── 관측 품질 ──
    /** 얼굴 너비 / 이미지 너비 가 이보다 작으면 FACE_TOO_SMALL */
    val minFaceWidthRatio: Double = 0.12,
    /** 얼굴 경계가 이미지 밖으로 이 비율 이상 나가면 잘린 얼굴(FACE_TOO_SMALL 과 같은 '관측 부족'으로 처리) */
    val maxFaceOutsideRatio: Double = 0.08,
    /** 눈 가로(눈꼬리 사이) 픽셀 길이가 이보다 작으면 EAR 분모가 불안정하다고 보고 무효 처리 */
    val minEyeWidthPx: Double = 6.0,
    /** 입 가로 픽셀 길이 최소값 */
    val minMouthWidthPx: Double = 10.0,
    /** 얼굴 영역 평균 밝기(0~255)가 이보다 낮으면 LOW_LIGHT_OR_LOW_CONTRAST */
    val minFaceLuma: Double = 30.0,
    /** 얼굴 영역 밝기 표준편차가 이보다 낮으면 대비 부족 */
    val minFaceLumaStdDev: Double = 8.0,
    /** solvePnP 평균 재투영 오차 / 얼굴 너비 가 이보다 크면 자세 무효 */
    val maxReprojectionErrorRatio: Double = 0.06,
    /** 기준 자세 대비 yaw 가 이보다 크면 눈·입 비율이 원근 왜곡돼 해당 지표만 무효(EXCESSIVE_HEAD_ROTATION) */
    val metricMaxYawDeg: Double = 30.0,
    val metricMaxPitchDeg: Double = 25.0,

    // ── 눈 (보정 필요) ──
    /** 눈 뜬 기준 EAR 대비 이 비율 아래로 내려가면 닫힘 후보(눈 감김 캘리브레이션을 안 했을 때) */
    val earCloseRatio: Double = 0.70,
    /** 눈 뜬 기준 EAR 대비 이 비율 위로 올라가면 열림 후보. earCloseRatio 보다 커야 한다(히스테리시스). */
    val earOpenRatio: Double = 0.85,
    /** 눈 감김 캘리브레이션을 했다면: 감은 EAR + (뜬 EAR − 감은 EAR) × 이 비율 아래를 닫힘으로 본다 */
    val earCloseFraction: Double = 0.35,
    val earOpenFraction: Double = 0.55,
    /** EAR 중앙값 필터 길이(프레임). 1 이면 끈다. 짧은 눈 감김을 지우지 않도록 3 이하로 둔다. */
    val earMedianWindow: Int = 3,
    /** 연속된 유효 관측에서 양쪽 눈 감김이 이 시간 이상 이어지면 '지속적 눈 감김' 이벤트 */
    val eyeClosureEventMs: Long = 1_500,

    // ── 눈 감김 비율(PERCLOS 유사 지표) ──
    val closedRatioWindowMs: Long = 60_000,
    /** 평가 창 대비 유효 관측 시간 비율이 이보다 낮으면 비율 판정을 보류한다 */
    val closedRatioMinCoverage: Double = 0.80,
    /** 눈 감김 비율이 이 값 이상이면 졸음 의심(휴식 권고). 보정 필요. */
    val closedRatioDrowsyThreshold: Double = 0.15,

    // ── 반복 ──
    /** [repeatWindowMs] 안에 지속적 눈 감김 이벤트가 이 횟수 이상이면 졸음 의심 */
    val repeatedClosureCount: Int = 3,
    val repeatWindowMs: Long = 300_000,

    // ── 입 (보정 필요) ──
    /** 다문 입 기준 비율 + 이 값보다 크면 벌림 후보 */
    val mouthOpenDelta: Double = 0.30,
    /** 다문 입 기준 비율 + 이 값보다 작아지면 다시 닫힘. mouthOpenDelta 보다 작아야 한다. */
    val mouthCloseDelta: Double = 0.18,
    val mouthOpenEventMs: Long = 2_000,
    /** 입 벌림 이벤트가 끝난 뒤 이 시간 안에 시작된 벌림은 새 이벤트로 세지 않는다 */
    val mouthRefractoryMs: Long = 5_000,
    /** [repeatWindowMs] 안에 입 벌림 이벤트가 이 횟수 이상이고 눈 감김 이벤트도 있으면 졸음 의심(보조 규칙) */
    val yawnSupportCount: Int = 2,

    // ── 고개 방향 ──
    val headYawAwayDeg: Double = 25.0,
    val headPitchAwayDeg: Double = 20.0,
    /** 진입 각도 × 이 비율 안으로 돌아와야 복귀로 본다(진입·복귀 분리) */
    val headReturnFactor: Double = 0.8,
    val headAwayEventMs: Long = 2_000,

    // ── 고개 숙임 반복(실험) ──
    val nodEnabled: Boolean = true,
    /** 기준 대비 이 각도 이상 숙였다가 절반 아래로 돌아오면 끄덕임 1회 */
    val nodPitchDeg: Double = 12.0,
    val nodMinMs: Long = 150,
    val nodMaxMs: Long = 1_500,
    val nodRepeatCount: Int = 3,
    val nodWindowMs: Long = 60_000,
    /** 끄덕임 반복을 졸음 판정 상향에 쓸지. 검증 전이므로 기본은 끈다. */
    val nodContributesToDrowsiness: Boolean = false,

    // ── 분석 불가 ──
    /** 눈 지표를 쓸 수 없는 상태가 이 시간 이상 이어지면 '분석 불가'로 안내한다 */
    val unavailableNoticeMs: Long = 3_000,

    // ── 판정 유지 ──
    /** 지속 눈 감김·고개 방향 이탈이 끝난 뒤 주의 표시와 시각적 부주의 의심을 이 시간 동안 유지한다 */
    val cautionHoldMs: Long = 3_000,
    /** 졸음 의심(휴식 권고)은 마지막 근거로부터 이 시간 동안 유지한다. 휴식을 시작하면 해제된다. */
    val drowsinessHoldMs: Long = 60_000,

    // ── 경고 반복 제한 ──
    val alertRepeatMs: Long = 20_000,
    val restAlertRepeatMs: Long = 60_000,
    val unavailableAlertRepeatMs: Long = 60_000,

    // ── 위치 ──
    /** 이벤트 시각과 이보다 차이 나는 위치는 붙이지 않는다 */
    val maxLocationAgeMs: Long = 10_000,
    val maxLocationAccuracyM: Double = 50.0,

    // ── 캘리브레이션 ──
    val calibrationTargetMs: Long = 5_000,
    val calibrationMaxMs: Long = 10_000,
    val calibrationMinSamples: Int = 25,
    /** 수집 시간 대비 유효 샘플 시간이 이 비율 이상이어야 목표 시간에 완료한다 */
    val calibrationMinCoverage: Double = 0.6,
    /** 기준 자세 샘플들의 yaw·pitch 사분위 범위가 이보다 크면 자세가 안정적이지 않다고 본다 */
    val calibrationMaxPoseIqrDeg: Double = 6.0,
    val closedEyeCalibrationMs: Long = 3_000,
) {
    fun validate(): List<String> = buildList {
        if (targetAnalysisFps !in 1..30) add("분석 제출 빈도는 1~30fps 여야 합니다")
        if (uiRefreshMs !in 50..2_000) add("화면 갱신 간격은 50~2000ms 여야 합니다")
        if (maxContinuityGapMs !in 50..2_000) add("연속 관측 최대 간격은 50~2000ms 여야 합니다")
        if (minFaceWidthRatio !in 0.01..0.9) add("최소 얼굴 너비 비율은 0.01~0.9 여야 합니다")
        if (earCloseRatio !in 0.1..0.99 || earOpenRatio !in 0.1..1.5) add("EAR 비율은 0.1 이상이어야 합니다")
        if (earCloseRatio >= earOpenRatio) add("EAR 닫힘 비율은 열림 비율보다 작아야 합니다")
        if (earCloseFraction >= earOpenFraction) add("EAR 닫힘 위치는 열림 위치보다 작아야 합니다")
        if (earMedianWindow !in 1..5) add("EAR 중앙값 필터는 1~5 프레임이어야 합니다")
        if (eyeClosureEventMs <= 0 || mouthOpenEventMs <= 0 || headAwayEventMs <= 0) add("이벤트 지속 시간은 0보다 커야 합니다")
        if (closedRatioWindowMs < 10_000) add("눈 감김 비율 평가 창은 10초 이상이어야 합니다")
        if (closedRatioMinCoverage !in 0.1..1.0) add("최소 유효 관측률은 0.1~1.0 이어야 합니다")
        if (closedRatioDrowsyThreshold !in 0.01..1.0) add("눈 감김 비율 기준은 0.01~1.0 이어야 합니다")
        if (repeatedClosureCount < 1 || yawnSupportCount < 1 || nodRepeatCount < 1) add("반복 횟수는 1 이상이어야 합니다")
        if (repeatWindowMs <= 0 || nodWindowMs <= 0) add("반복 판단 구간은 0보다 커야 합니다")
        if (mouthCloseDelta >= mouthOpenDelta) add("입 닫힘 기준은 벌림 기준보다 작아야 합니다")
        if (headYawAwayDeg !in 1.0..90.0 || headPitchAwayDeg !in 1.0..90.0) add("고개 각도 기준은 1~90도여야 합니다")
        if (headReturnFactor !in 0.1..1.0) add("복귀 비율은 0.1~1.0 이어야 합니다")
        if (nodMinMs >= nodMaxMs) add("끄덕임 최소 시간은 최대 시간보다 작아야 합니다")
        if (cautionHoldMs < 0 || drowsinessHoldMs < 0) add("판정 유지 시간은 음수일 수 없습니다")
        if (unavailableNoticeMs <= 0) add("분석 불가 안내 지연은 0보다 커야 합니다")
        if (alertRepeatMs <= 0 || restAlertRepeatMs <= 0 || unavailableAlertRepeatMs <= 0) add("경고 반복 제한은 0보다 커야 합니다")
        if (calibrationTargetMs <= 0 || calibrationMaxMs < calibrationTargetMs) add("캘리브레이션 최대 시간은 목표 시간 이상이어야 합니다")
        if (calibrationMinSamples < 5) add("캘리브레이션 최소 샘플은 5 이상이어야 합니다")
    }

    /** 여정에 남길 설정 식별자. 값이 하나라도 다르면 달라진다. */
    fun versionTag(): String = "v$SCHEMA_VERSION-" + hashCode().toUInt().toString(16).padStart(8, '0')

    companion object {
        const val SCHEMA_VERSION = 1
    }
}
