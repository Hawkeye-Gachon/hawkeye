package com.personal.driverguard.domain.analysis.calibration

import com.personal.driverguard.entity.AnalysisConfig
import com.personal.driverguard.entity.CalibrationBaseline
import com.personal.driverguard.entity.FaceObservation
import com.personal.driverguard.entity.Mat3
import com.personal.driverguard.entity.QualityFlag
import kotlin.math.abs

enum class CalibrationFailure {
    /** 유효 샘플이 부족함(최대 시간까지 연장해도). 함께 주는 품질 사유가 가장 흔했던 원인이다. */
    NOT_ENOUGH_VALID_SAMPLES,

    /** 기준 자세가 흔들림 */
    POSE_UNSTABLE,

    /** 눈 뜬 기준 EAR 이 비정상적으로 낮음(눈을 감고 있었거나 랜드마크가 눈꺼풀을 못 잡음) */
    EYES_NOT_OPEN,

    /** 눈 감김 캘리브레이션에서 EAR 이 충분히 내려가지 않음 */
    EYES_NOT_CLOSED,
}

sealed interface CalibrationState {
    data class Collecting(
        val elapsedMs: Long,
        /** 목표 시간 대비 유효 샘플 시간(0~1) */
        val progress: Double,
        val validSamples: Int,
        /** 가장 최근 무효 사유. 화면 안내에 쓴다. */
        val lastIssue: QualityFlag?,
    ) : CalibrationState

    data class Completed(val baseline: CalibrationBaseline) : CalibrationState

    data class Failed(val reason: CalibrationFailure, val dominantIssue: QualityFlag?) : CalibrationState
}

/**
 * 정차 상태에서 평소 운전 자세의 기준값을 측정한다.
 *
 * - 목표 시간([AnalysisConfig.calibrationTargetMs]) 동안 수집하고, 유효 샘플이 부족하면 최대 시간까지 연장한다.
 *   그래도 부족하면 완료하지 않고 실패 사유를 돌려준다.
 * - 순간 깜빡임·이상값에 흔들리지 않도록 평균 대신 중앙값을 쓰고, 사분위 범위로 변동성을 확인한다.
 * - 기준 회전은 중앙값 각도에 가장 가까운 실제 샘플의 회전 행렬을 쓴다(오일러각 평균으로 행렬을 만들지 않는다).
 */
class CalibrationManager(private val config: AnalysisConfig) {

    private class Sample(
        val leftEar: Double,
        val rightEar: Double,
        val mouth: Double,
        val rotation: Mat3,
        val yaw: Double,
        val pitch: Double,
        val roll: Double,
        val faceWidth: Double,
    )

    private val samples = mutableListOf<Sample>()
    private val issues = mutableMapOf<QualityFlag, Int>()
    private var startMs: Long? = null
    private var lastValidMs: Long? = null
    private var validMs = 0L
    private var lastIssue: QualityFlag? = null

    var state: CalibrationState = CalibrationState.Collecting(0, 0.0, 0, null)
        private set

    fun reset() {
        samples.clear()
        issues.clear()
        startMs = null
        lastValidMs = null
        validMs = 0
        lastIssue = null
        state = CalibrationState.Collecting(0, 0.0, 0, null)
    }

    fun add(obs: FaceObservation): CalibrationState {
        if (state !is CalibrationState.Collecting) return state
        val start = startMs ?: obs.timestampMs.also { startMs = it }
        val issue = issueOf(obs)
        if (issue == null) {
            val pose = obs.pose!!
            samples += Sample(obs.leftEar!!, obs.rightEar!!, obs.mouthOpenRatio!!, pose.rotation, pose.yawDeg, pose.pitchDeg, pose.rollDeg, obs.faceWidthRatio ?: 0.0)
            lastValidMs?.let { prev ->
                val dt = obs.timestampMs - prev
                if (dt in 1..config.maxContinuityGapMs) validMs += dt
            }
            lastValidMs = obs.timestampMs
            lastIssue = null
        } else {
            issues[issue] = (issues[issue] ?: 0) + 1
            lastValidMs = null
            lastIssue = issue
        }
        val elapsed = obs.timestampMs - start
        state = decide(elapsed)
        return state
    }

    private fun issueOf(obs: FaceObservation): QualityFlag? = listOf(obs.faceQuality, obs.eyeQuality, obs.mouthQuality, obs.poseQuality)
        .firstOrNull { !it.isValid }
        ?: if (obs.leftEar == null || obs.rightEar == null || obs.mouthOpenRatio == null || obs.pose == null) QualityFlag.LANDMARKS_INVALID else null

    private fun decide(elapsed: Long): CalibrationState {
        val enough = samples.size >= config.calibrationMinSamples
        val coverageOk = validMs >= config.calibrationTargetMs * config.calibrationMinCoverage
        if (elapsed >= config.calibrationTargetMs && enough && coverageOk) return finish()
        if (elapsed >= config.calibrationMaxMs) {
            return if (enough) finish() else CalibrationState.Failed(CalibrationFailure.NOT_ENOUGH_VALID_SAMPLES, issues.maxByOrNull { it.value }?.key)
        }
        val progress = (validMs.toDouble() / config.calibrationTargetMs).coerceIn(0.0, 0.99)
        return CalibrationState.Collecting(elapsed, progress, samples.size, lastIssue)
    }

    private fun finish(): CalibrationState {
        val yawIqr = iqr(samples.map { it.yaw })
        val pitchIqr = iqr(samples.map { it.pitch })
        if (yawIqr > config.calibrationMaxPoseIqrDeg || pitchIqr > config.calibrationMaxPoseIqrDeg) {
            return CalibrationState.Failed(CalibrationFailure.POSE_UNSTABLE, null)
        }
        val left = median(samples.map { it.leftEar })
        val right = median(samples.map { it.rightEar })
        if (left < MIN_OPEN_EAR || right < MIN_OPEN_EAR) return CalibrationState.Failed(CalibrationFailure.EYES_NOT_OPEN, null)
        val yaw = median(samples.map { it.yaw })
        val pitch = median(samples.map { it.pitch })
        val roll = median(samples.map { it.roll })
        val reference = samples.minBy { abs(it.yaw - yaw) + abs(it.pitch - pitch) + abs(it.roll - roll) }
        return CalibrationState.Completed(
            CalibrationBaseline(
                leftEarOpen = left,
                rightEarOpen = right,
                mouthClosedRatio = median(samples.map { it.mouth }),
                poseRotation = reference.rotation,
                yawDeg = yaw,
                pitchDeg = pitch,
                rollDeg = roll,
                faceWidthRatio = median(samples.map { it.faceWidth }),
                sampleCount = samples.size,
                earIqr = iqr(samples.map { (it.leftEar + it.rightEar) / 2 }),
                yawIqrDeg = yawIqr,
                pitchIqrDeg = pitchIqr,
            ),
        )
    }

    companion object {
        /** 눈을 뜬 상태의 EAR 이 이보다 낮으면 기준값으로 쓰지 않는다(개발용 하한) */
        const val MIN_OPEN_EAR = 0.12

        fun median(values: List<Double>): Double = percentile(values, 0.5)

        /** 선형 보간 백분위수(p 는 0~1) */
        fun percentile(values: List<Double>, p: Double): Double {
            require(values.isNotEmpty()) { "값이 없습니다" }
            val sorted = values.sorted()
            val pos = p * (sorted.size - 1)
            val lo = pos.toInt()
            val hi = minOf(lo + 1, sorted.size - 1)
            return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo)
        }

        fun iqr(values: List<Double>): Double = percentile(values, 0.75) - percentile(values, 0.25)
    }
}

/**
 * 선택적인 눈 감김 캘리브레이션. 정차 상태에서만 안내한다.
 * 눈을 감은 상태의 EAR 중앙값을 측정해 닫힘·열림 임계값을 두 기준값 사이에 둔다.
 * 하지 않으면 눈 뜬 기준값 비율만으로 임계값을 만들며, 이는 충분히 보정된 값이 아니다.
 */
class ClosedEyeCalibration(private val config: AnalysisConfig, private val baseline: CalibrationBaseline) {
    private val left = mutableListOf<Double>()
    private val right = mutableListOf<Double>()
    private var startMs: Long? = null

    var result: CalibrationState? = null
        private set

    /** 진행률(0~1). 완료되면 [result] 가 채워진다. */
    fun add(obs: FaceObservation): Double {
        if (result != null) return 1.0
        val start = startMs ?: obs.timestampMs.also { startMs = it }
        val l = obs.leftEar
        val r = obs.rightEar
        if (obs.eyeQuality.isValid && l != null && r != null) {
            left.add(l)
            right.add(r)
        }
        val elapsed = obs.timestampMs - start
        if (elapsed >= config.closedEyeCalibrationMs) result = finish()
        return (elapsed.toDouble() / config.closedEyeCalibrationMs).coerceIn(0.0, 1.0)
    }

    private fun finish(): CalibrationState {
        if (left.size < MIN_SAMPLES) return CalibrationState.Failed(CalibrationFailure.NOT_ENOUGH_VALID_SAMPLES, QualityFlag.LANDMARKS_INVALID)
        val l = CalibrationManager.median(left)
        val r = CalibrationManager.median(right)
        if (l > baseline.leftEarOpen * MAX_CLOSED_RATIO || r > baseline.rightEarOpen * MAX_CLOSED_RATIO) {
            return CalibrationState.Failed(CalibrationFailure.EYES_NOT_CLOSED, null)
        }
        return CalibrationState.Completed(baseline.copy(leftEarClosed = l, rightEarClosed = r))
    }

    private companion object {
        const val MIN_SAMPLES = 10

        /** 감은 EAR 이 뜬 EAR 의 이 비율보다 크면 제대로 감지 않은 것으로 본다 */
        const val MAX_CLOSED_RATIO = 0.8
    }
}
