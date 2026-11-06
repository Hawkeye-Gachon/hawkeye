package com.personal.driverguard.views.components

import androidx.annotation.StringRes
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.CheckCircle
import androidx.compose.material.icons.rounded.Face
import androidx.compose.material.icons.rounded.Notifications
import androidx.compose.material.icons.rounded.Warning
import androidx.compose.ui.graphics.vector.ImageVector
import com.personal.driverguard.R
import com.personal.driverguard.domain.analysis.calibration.CalibrationFailure
import com.personal.driverguard.domain.engine.JudgmentRules
import com.personal.driverguard.entity.AlertLevel
import com.personal.driverguard.entity.AnalysisMode
import com.personal.driverguard.entity.CautionCause
import com.personal.driverguard.entity.DisplayState
import com.personal.driverguard.entity.EventEndReason
import com.personal.driverguard.entity.EventType
import com.personal.driverguard.entity.InterruptionReason
import com.personal.driverguard.entity.QualityFlag
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@StringRes
fun modeLabel(mode: AnalysisMode) = when (mode) {
    AnalysisMode.REAL -> R.string.mode_real
    AnalysisMode.DEMO -> R.string.mode_demo
}

@StringRes
fun displayLabel(state: DisplayState, cause: CautionCause?) = when (state) {
    DisplayState.PREPARING -> R.string.state_preparing
    DisplayState.OBSERVING -> R.string.state_observing
    DisplayState.CAUTION -> if (cause == CautionCause.HEAD_AWAY) R.string.state_caution_head else R.string.state_caution_eyes
    DisplayState.REST_RECOMMENDED -> R.string.state_rest
    DisplayState.UNAVAILABLE -> R.string.state_unavailable
    DisplayState.PAUSED -> R.string.state_paused
}

fun displayIcon(state: DisplayState): ImageVector = when (state) {
    DisplayState.OBSERVING, DisplayState.PREPARING, DisplayState.PAUSED -> Icons.Rounded.CheckCircle
    DisplayState.CAUTION -> Icons.Rounded.Warning
    DisplayState.REST_RECOMMENDED -> Icons.Rounded.Notifications
    DisplayState.UNAVAILABLE -> Icons.Rounded.Face
}

@StringRes
fun displayGuidance(state: DisplayState, cause: CautionCause?) = when (state) {
    DisplayState.PREPARING -> R.string.guidance_preparing
    DisplayState.OBSERVING -> R.string.guidance_observing
    DisplayState.CAUTION -> if (cause == CautionCause.HEAD_AWAY) R.string.guidance_caution_head else R.string.guidance_caution_eyes
    DisplayState.REST_RECOMMENDED -> R.string.guidance_rest
    DisplayState.UNAVAILABLE -> R.string.guidance_unavailable
    DisplayState.PAUSED -> R.string.guidance_paused
}

@StringRes
fun qualityLabel(flag: QualityFlag) = when (flag) {
    QualityFlag.VALID -> R.string.quality_valid
    QualityFlag.FACE_NOT_FOUND -> R.string.quality_face_not_found
    QualityFlag.MULTIPLE_FACES -> R.string.quality_multiple_faces
    QualityFlag.FACE_TOO_SMALL -> R.string.quality_face_too_small
    QualityFlag.LANDMARKS_INVALID -> R.string.quality_landmarks_invalid
    QualityFlag.EXCESSIVE_HEAD_ROTATION -> R.string.quality_excessive_rotation
    QualityFlag.LOW_LIGHT_OR_LOW_CONTRAST -> R.string.quality_low_light
    QualityFlag.FRAME_GAP -> R.string.quality_frame_gap
    QualityFlag.CALIBRATION_REQUIRED -> R.string.quality_calibration_required
    QualityFlag.ANALYSIS_ERROR -> R.string.quality_analysis_error
}

@StringRes
fun eventTypeLabel(type: EventType) = when (type) {
    EventType.EYE_CLOSURE -> R.string.event_eye_closure
    EventType.MOUTH_OPEN -> R.string.event_mouth_open
    EventType.HEAD_AWAY -> R.string.event_head_away
    EventType.HEAD_NOD -> R.string.event_head_nod
    EventType.ANALYSIS_UNAVAILABLE -> R.string.event_unavailable
    EventType.DROWSINESS_SUSPECTED -> R.string.event_drowsiness
    EventType.DISTRACTION_SUSPECTED -> R.string.event_distraction
}

@StringRes
fun alertLevelLabel(level: AlertLevel) = when (level) {
    AlertLevel.INFO -> R.string.level_info
    AlertLevel.CAUTION -> R.string.level_caution
    AlertLevel.REST -> R.string.level_rest
}

@StringRes
fun endReasonLabel(reason: EventEndReason) = when (reason) {
    EventEndReason.RECOVERED -> R.string.end_recovered
    EventEndReason.OBSERVATION_GAP -> R.string.end_gap
    EventEndReason.OBSERVATION_INVALID -> R.string.end_invalid
    EventEndReason.HOLD_EXPIRED -> R.string.end_hold_expired
    EventEndReason.REST_STARTED -> R.string.end_rest
    EventEndReason.APP_PAUSED -> R.string.end_app_paused
    EventEndReason.TRIP_ENDED -> R.string.end_trip_ended
    EventEndReason.ABNORMAL_TERMINATION -> R.string.end_abnormal
}

@StringRes
fun interruptionLabel(reason: InterruptionReason) = when (reason) {
    InterruptionReason.REST -> R.string.interruption_rest
    InterruptionReason.APP_BACKGROUND -> R.string.interruption_background
    InterruptionReason.CAMERA_ERROR -> R.string.interruption_camera
}

@StringRes
fun ruleLabel(rule: String): Int? = when (rule) {
    JudgmentRules.EYE_SUSTAINED -> R.string.rule_eye_sustained
    JudgmentRules.CLOSED_RATIO -> R.string.rule_closed_ratio
    JudgmentRules.REPEATED_CLOSURE -> R.string.rule_repeated_closure
    JudgmentRules.YAWN_WITH_CLOSURE -> R.string.rule_yawn_with_closure
    JudgmentRules.SUPPORT_YAWN -> R.string.rule_support_yawn
    JudgmentRules.SUPPORT_NOD -> R.string.rule_support_nod
    JudgmentRules.HEAD_AWAY -> R.string.rule_head_away
    JudgmentRules.MOUTH_SUSTAINED -> R.string.rule_mouth_sustained
    JudgmentRules.NOD_REPEATED -> R.string.rule_nod_repeated
    JudgmentRules.UNAVAILABLE -> R.string.rule_unavailable
    else -> null
}

@StringRes
fun calibrationFailureLabel(failure: CalibrationFailure) = when (failure) {
    CalibrationFailure.NOT_ENOUGH_VALID_SAMPLES -> R.string.calib_fail_samples
    CalibrationFailure.POSE_UNSTABLE -> R.string.calib_fail_pose
    CalibrationFailure.EYES_NOT_OPEN -> R.string.calib_fail_eyes_open
    CalibrationFailure.EYES_NOT_CLOSED -> R.string.calib_fail_eyes_closed
}

fun formatDateTime(epochMs: Long): String =
    SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date(epochMs))

fun formatTime(epochMs: Long): String = SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date(epochMs))

fun formatDuration(ms: Long): String {
    if (ms < 60_000) return String.format(Locale.getDefault(), "%.1f초", ms / 1000.0)
    val totalSeconds = ms / 1000
    val h = totalSeconds / 3600
    val m = totalSeconds % 3600 / 60
    val s = totalSeconds % 60
    return if (h > 0) "${h}시간 ${m}분 ${s}초" else "${m}분 ${s}초"
}

/** 경과 시간 시계 표기(예: 05:09, 1:02:03) */
fun formatClock(ms: Long): String {
    val totalSeconds = (ms / 1000).coerceAtLeast(0)
    val h = totalSeconds / 3600
    val m = totalSeconds % 3600 / 60
    val s = totalSeconds % 60
    return if (h > 0) String.format(Locale.US, "%d:%02d:%02d", h, m, s) else String.format(Locale.US, "%02d:%02d", m, s)
}

fun formatDistance(meters: Double): String =
    if (meters >= 1000) String.format(Locale.getDefault(), "%.2f km", meters / 1000) else "${meters.toInt()} m"

fun formatMs(value: Double?): String = value?.let { String.format(Locale.getDefault(), "%.1f ms", it) } ?: "-"
