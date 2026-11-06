package com.personal.driverguard.views.devsettings

import androidx.annotation.StringRes
import com.personal.driverguard.R
import com.personal.driverguard.entity.AnalysisConfig

/**
 * 개발 화면에서 편집하는 설정 칸. 정차·개발 환경 전용 화면이라 항목 이름은 설정 키와 함께 바로 적는다.
 * 모든 값은 개발 시작값이며 검증된 안전 기준이 아니다.
 */
class ConfigField(
    val key: String,
    val label: String,
    val unit: String,
    val read: (AnalysisConfig) -> String,
    val write: (AnalysisConfig, String) -> AnalysisConfig?,
)

class ConfigGroup(@param:StringRes val title: Int, val fields: List<ConfigField>)

private fun long(key: String, label: String, get: (AnalysisConfig) -> Long, set: AnalysisConfig.(Long) -> AnalysisConfig) =
    ConfigField(key, label, "ms", { get(it).toString() }, { c, t -> t.trim().toLongOrNull()?.let { c.set(it) } })

private fun int(key: String, label: String, unit: String, get: (AnalysisConfig) -> Int, set: AnalysisConfig.(Int) -> AnalysisConfig) =
    ConfigField(key, label, unit, { get(it).toString() }, { c, t -> t.trim().toIntOrNull()?.let { c.set(it) } })

private fun dbl(key: String, label: String, unit: String, get: (AnalysisConfig) -> Double, set: AnalysisConfig.(Double) -> AnalysisConfig) =
    ConfigField(key, label, unit, { get(it).toString() }, { c, t -> t.trim().toDoubleOrNull()?.let { c.set(it) } })

private fun bool(key: String, label: String, get: (AnalysisConfig) -> Boolean, set: AnalysisConfig.(Boolean) -> AnalysisConfig) =
    ConfigField(key, label, "1/0", { if (get(it)) "1" else "0" }, { c, t ->
        when (t.trim()) {
            "1", "true" -> c.set(true)
            "0", "false" -> c.set(false)
            else -> null
        }
    })

val configGroups = listOf(
    ConfigGroup(
        R.string.group_eyes,
        listOf(
            dbl("earCloseRatio", "닫힘: 뜬 EAR × (보정 필요)", "배", { it.earCloseRatio }) { copy(earCloseRatio = it) },
            dbl("earOpenRatio", "열림: 뜬 EAR × (보정 필요)", "배", { it.earOpenRatio }) { copy(earOpenRatio = it) },
            dbl("earCloseFraction", "닫힘 위치(감음 보정 시)", "0~1", { it.earCloseFraction }) { copy(earCloseFraction = it) },
            dbl("earOpenFraction", "열림 위치(감음 보정 시)", "0~1", { it.earOpenFraction }) { copy(earOpenFraction = it) },
            int("earMedianWindow", "EAR 중앙값 필터", "프레임", { it.earMedianWindow }) { copy(earMedianWindow = it) },
            long("eyeClosureEventMs", "지속 눈 감김 이벤트", { it.eyeClosureEventMs }) { copy(eyeClosureEventMs = it) },
        ),
    ),
    ConfigGroup(
        R.string.group_ratio,
        listOf(
            long("closedRatioWindowMs", "눈 감김 비율 평가 창", { it.closedRatioWindowMs }) { copy(closedRatioWindowMs = it) },
            dbl("closedRatioMinCoverage", "최소 유효 관측률", "0~1", { it.closedRatioMinCoverage }) { copy(closedRatioMinCoverage = it) },
            dbl("closedRatioDrowsyThreshold", "졸음 의심 비율 (보정 필요)", "0~1", { it.closedRatioDrowsyThreshold }) { copy(closedRatioDrowsyThreshold = it) },
            int("repeatedClosureCount", "반복 눈 감김 횟수", "회", { it.repeatedClosureCount }) { copy(repeatedClosureCount = it) },
            long("repeatWindowMs", "반복 판단 구간", { it.repeatWindowMs }) { copy(repeatWindowMs = it) },
            long("drowsinessHoldMs", "졸음 의심 유지", { it.drowsinessHoldMs }) { copy(drowsinessHoldMs = it) },
        ),
    ),
    ConfigGroup(
        R.string.group_mouth,
        listOf(
            dbl("mouthOpenDelta", "벌림: 다문 입 + (보정 필요)", "비율", { it.mouthOpenDelta }) { copy(mouthOpenDelta = it) },
            dbl("mouthCloseDelta", "닫힘: 다문 입 +", "비율", { it.mouthCloseDelta }) { copy(mouthCloseDelta = it) },
            long("mouthOpenEventMs", "입 벌림 최소 지속", { it.mouthOpenEventMs }) { copy(mouthOpenEventMs = it) },
            long("mouthRefractoryMs", "재발생 최소 간격", { it.mouthRefractoryMs }) { copy(mouthRefractoryMs = it) },
            int("yawnSupportCount", "입 벌림 반복(보조 규칙)", "회", { it.yawnSupportCount }) { copy(yawnSupportCount = it) },
        ),
    ),
    ConfigGroup(
        R.string.group_head,
        listOf(
            dbl("headYawAwayDeg", "좌우 이탈 각도", "도", { it.headYawAwayDeg }) { copy(headYawAwayDeg = it) },
            dbl("headPitchAwayDeg", "상하 이탈 각도", "도", { it.headPitchAwayDeg }) { copy(headPitchAwayDeg = it) },
            dbl("headReturnFactor", "복귀 비율", "배", { it.headReturnFactor }) { copy(headReturnFactor = it) },
            long("headAwayEventMs", "이탈 최소 지속", { it.headAwayEventMs }) { copy(headAwayEventMs = it) },
            long("cautionHoldMs", "주의 표시 유지", { it.cautionHoldMs }) { copy(cautionHoldMs = it) },
            bool("nodEnabled", "고개 숙임 반복(실험)", { it.nodEnabled }) { copy(nodEnabled = it) },
            dbl("nodPitchDeg", "숙임 각도(실험)", "도", { it.nodPitchDeg }) { copy(nodPitchDeg = it) },
            bool("nodContributesToDrowsiness", "숙임을 졸음 상향에 사용", { it.nodContributesToDrowsiness }) { copy(nodContributesToDrowsiness = it) },
        ),
    ),
    ConfigGroup(
        R.string.group_frames,
        listOf(
            int("targetAnalysisFps", "분석 제출 빈도", "fps", { it.targetAnalysisFps }) { copy(targetAnalysisFps = it) },
            long("uiRefreshMs", "개발 지표 갱신", { it.uiRefreshMs }) { copy(uiRefreshMs = it) },
            long("maxContinuityGapMs", "연속 관측 최대 간격", { it.maxContinuityGapMs }) { copy(maxContinuityGapMs = it) },
            long("unavailableNoticeMs", "분석 불가 안내 지연", { it.unavailableNoticeMs }) { copy(unavailableNoticeMs = it) },
            dbl("minFaceWidthRatio", "최소 얼굴 너비", "비율", { it.minFaceWidthRatio }) { copy(minFaceWidthRatio = it) },
            dbl("metricMaxYawDeg", "눈·입 지표 최대 yaw", "도", { it.metricMaxYawDeg }) { copy(metricMaxYawDeg = it) },
            dbl("maxReprojectionErrorRatio", "재투영 오차 / 얼굴 너비", "비율", { it.maxReprojectionErrorRatio }) { copy(maxReprojectionErrorRatio = it) },
        ),
    ),
    ConfigGroup(
        R.string.group_alerts,
        listOf(
            long("alertRepeatMs", "주의 경고 반복 제한", { it.alertRepeatMs }) { copy(alertRepeatMs = it) },
            long("restAlertRepeatMs", "휴식 권고 반복 제한", { it.restAlertRepeatMs }) { copy(restAlertRepeatMs = it) },
            long("unavailableAlertRepeatMs", "분석 불가 안내 반복 제한", { it.unavailableAlertRepeatMs }) { copy(unavailableAlertRepeatMs = it) },
        ),
    ),
)

val allConfigFields = configGroups.flatMap { it.fields }
