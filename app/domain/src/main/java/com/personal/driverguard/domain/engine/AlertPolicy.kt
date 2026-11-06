package com.personal.driverguard.domain.engine

import com.personal.driverguard.entity.AlertKind
import com.personal.driverguard.entity.AnalysisConfig

/**
 * 엔진의 경고 요청 중 실제로 재생할 것을 고른다.
 *
 * - 종류별로 반복 제한 시간을 따로 둔다. 낮은 단계의 반복 제한이 높은 단계(휴식 권고)를 막지 않는다.
 * - 한 번에 하나만 고른다: 요청 중 우선순위가 가장 높고 반복 제한에 걸리지 않은 것.
 * - 이벤트 기록과는 분리돼 있다. 여기서 버려진 경고도 이벤트는 엔진이 이미 기록했다.
 */
class AlertPolicy(private val config: AnalysisConfig) {

    private val lastPlayedMs = mutableMapOf<AlertKind, Long>()

    fun select(requests: Set<AlertKind>, nowMs: Long): AlertKind? {
        val kind = requests.sortedByDescending { it.priority }.firstOrNull { ready(it, nowMs) } ?: return null
        lastPlayedMs[kind] = nowMs
        return kind
    }

    fun reset() = lastPlayedMs.clear()

    private fun ready(kind: AlertKind, nowMs: Long): Boolean {
        val last = lastPlayedMs[kind] ?: return true
        return nowMs - last >= intervalOf(kind)
    }

    private fun intervalOf(kind: AlertKind): Long = when (kind) {
        AlertKind.EYE_CAUTION, AlertKind.HEAD_CAUTION -> config.alertRepeatMs
        AlertKind.REST -> config.restAlertRepeatMs
        AlertKind.UNAVAILABLE -> config.unavailableAlertRepeatMs
    }
}

/**
 * 경고음·음성이 겹치지 않게 재생 여부를 정한다.
 * 재생 중에는 더 높은 우선순위만 끊고 들어갈 수 있고, 같거나 낮은 것은 버린다.
 */
class AlertArbiter {

    enum class Decision { PLAY, INTERRUPT_AND_PLAY, DROP }

    var playing: AlertKind? = null
        private set

    fun offer(kind: AlertKind): Decision {
        val current = playing
        return when {
            current == null -> {
                playing = kind
                Decision.PLAY
            }
            kind.priority > current.priority -> {
                playing = kind
                Decision.INTERRUPT_AND_PLAY
            }
            else -> Decision.DROP
        }
    }

    /** 재생이 끝났거나 취소됐을 때 부른다. 이미 다른 경고로 바뀌었으면 무시한다. */
    fun finished(kind: AlertKind) {
        if (playing == kind) playing = null
    }

    fun clear() {
        playing = null
    }
}
