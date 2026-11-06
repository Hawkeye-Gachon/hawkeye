package com.personal.driverguard.domain.analysis.temporal

import com.personal.driverguard.entity.AnalysisConfig
import com.personal.driverguard.entity.CalibrationBaseline

enum class EyeState { OPEN, CLOSED, UNKNOWN }

/** 눈별 닫힘·열림 임계값. close < open (히스테리시스). */
data class EyeThresholds(
    val leftClose: Double,
    val leftOpen: Double,
    val rightClose: Double,
    val rightOpen: Double,
    /** 눈 감김 캘리브레이션 결과로 만든 값인지 */
    val fromClosedCalibration: Boolean,
) {
    companion object {
        fun from(baseline: CalibrationBaseline, config: AnalysisConfig): EyeThresholds {
            val lc = baseline.leftEarClosed
            val rc = baseline.rightEarClosed
            return if (lc != null && rc != null) {
                EyeThresholds(
                    leftClose = lc + (baseline.leftEarOpen - lc) * config.earCloseFraction,
                    leftOpen = lc + (baseline.leftEarOpen - lc) * config.earOpenFraction,
                    rightClose = rc + (baseline.rightEarOpen - rc) * config.earCloseFraction,
                    rightOpen = rc + (baseline.rightEarOpen - rc) * config.earOpenFraction,
                    fromClosedCalibration = true,
                )
            } else {
                EyeThresholds(
                    leftClose = baseline.leftEarOpen * config.earCloseRatio,
                    leftOpen = baseline.leftEarOpen * config.earOpenRatio,
                    rightClose = baseline.rightEarOpen * config.earCloseRatio,
                    rightOpen = baseline.rightEarOpen * config.earOpenRatio,
                    fromClosedCalibration = false,
                )
            }
        }
    }
}

/**
 * 양쪽 눈의 EAR 에 짧은 중앙값 필터와 히스테리시스를 적용해 눈 상태를 정한다.
 *
 * - 닫힘: 두 눈이 모두 닫힘 임계값 아래(한쪽만 감은 윙크는 닫힘이 아니다)
 * - 열림: 어느 한쪽이라도 열림 임계값 위
 * - 중간 구간: 이전 상태 유지(처음이면 열림으로 본다)
 * 관측이 끊기거나 눈 지표가 무효가 되면 [reset] 해서 이전 상태·필터 값을 새 관측에 이어 쓰지 않는다.
 */
class EyeStateTracker(private val thresholds: EyeThresholds, private val medianWindow: Int) {
    private val leftBuffer = ArrayDeque<Double>()
    private val rightBuffer = ArrayDeque<Double>()

    var state: EyeState = EyeState.UNKNOWN
        private set
    var filteredLeft: Double? = null
        private set
    var filteredRight: Double? = null
        private set

    fun update(leftEar: Double, rightEar: Double): EyeState {
        val l = push(leftBuffer, leftEar)
        val r = push(rightBuffer, rightEar)
        filteredLeft = l
        filteredRight = r
        val bothClosed = l < thresholds.leftClose && r < thresholds.rightClose
        val anyOpen = l > thresholds.leftOpen || r > thresholds.rightOpen
        state = when {
            bothClosed -> EyeState.CLOSED
            anyOpen -> EyeState.OPEN
            state == EyeState.UNKNOWN -> EyeState.OPEN
            else -> state
        }
        return state
    }

    fun reset() {
        leftBuffer.clear()
        rightBuffer.clear()
        filteredLeft = null
        filteredRight = null
        state = EyeState.UNKNOWN
    }

    private fun push(buffer: ArrayDeque<Double>, value: Double): Double {
        buffer.addLast(value)
        while (buffer.size > medianWindow) buffer.removeFirst()
        val sorted = buffer.sorted()
        return if (sorted.size % 2 == 1) sorted[sorted.size / 2] else (sorted[sorted.size / 2 - 1] + sorted[sorted.size / 2]) / 2
    }
}
