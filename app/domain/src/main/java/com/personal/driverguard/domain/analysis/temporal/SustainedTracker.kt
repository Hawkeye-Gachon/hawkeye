package com.personal.driverguard.domain.analysis.temporal

import com.personal.driverguard.domain.engine.EventLedger
import com.personal.driverguard.entity.AlertLevel
import com.personal.driverguard.entity.EventEndReason
import com.personal.driverguard.entity.EventType
import java.util.Locale

/** 이벤트에 남길 지표 극값(최솟값·최댓값) */
class MetricExtremes {
    private val values = linkedMapOf<String, Double>()

    fun min(name: String, value: Double?) {
        if (value == null || !value.isFinite()) return
        values[name] = values[name]?.let { minOf(it, value) } ?: value
    }

    fun max(name: String, value: Double?) {
        if (value == null || !value.isFinite()) return
        values[name] = values[name]?.let { maxOf(it, value) } ?: value
    }

    fun clear() = values.clear()

    fun format(): String = values.entries.joinToString(", ") { (k, v) -> "$k=" + String.format(Locale.US, "%.3f", v) }
}

/**
 * '조건이 연속된 유효 관측에서 일정 시간 이상 유지되면 이벤트'를 추적한다.
 * 생명주기: 후보 시작(조건 첫 관측) → 지속 조건 충족(확인, 레코드 생성) → 종료 조건(레코드 닫힘).
 *
 * 지속 시간은 프레임 수가 아니라 타임스탬프 차이로 계산한다.
 * 관측 공백·지표 무효로 끊기면 [breakOff] 로 닫고, 종료 사유에 정상 복귀가 아님을 남긴다.
 */
class SustainedTracker(
    val type: EventType,
    private val minDurationMs: Long,
    private val level: AlertLevel,
    private val rule: String,
    private val ledger: EventLedger,
) {
    var candidateStartMs: Long? = null
        private set
    private var lastActiveMs: Long? = null
    var key: Long? = null
        private set
    var lastConfirmedEndMs: Long? = null
        private set
    val extremes = MetricExtremes()

    val active: Boolean get() = candidateStartMs != null
    val confirmed: Boolean get() = key != null

    fun durationMs(nowMs: Long): Long = candidateStartMs?.let { nowMs - it } ?: 0L

    /** 조건이 관측됐다. [record] 로 지표 극값을 남긴다. 이번 호출로 이벤트가 확인됐으면 true. */
    fun onActive(ts: Long, record: MetricExtremes.() -> Unit = {}): Boolean {
        val start = candidateStartMs ?: ts.also {
            candidateStartMs = it
            extremes.clear()
        }
        extremes.record()
        lastActiveMs = ts
        if (key == null && ts - start >= minDurationMs) {
            key = ledger.open(type, start, ts, level, listOf(rule), extremes.format())
            return true
        }
        return false
    }

    /** 조건이 해소된 유효 관측. 확인된 이벤트면 [ts] 에 정상 복귀로 닫는다. */
    fun onInactive(ts: Long) = close(ts, EventEndReason.RECOVERED)

    /** 관측이 끊겼거나(공백·무효) 분석을 멈춘다. 확인된 이벤트면 [endMs] 에 [reason] 으로 닫는다. */
    fun breakOff(endMs: Long, reason: EventEndReason) = close(endMs, reason)

    /** 진행 중인 이벤트의 지표 요약을 갱신한다 */
    fun refreshMetrics() {
        key?.let { ledger.update(it, level, listOf(rule), extremes.format()) }
    }

    private fun close(endMs: Long, reason: EventEndReason) {
        val start = candidateStartMs
        val k = key
        if (k != null && start != null) {
            val end = maxOf(endMs, start)
            ledger.close(k, end, end - start, reason, extremes.format())
            lastConfirmedEndMs = end
        }
        candidateStartMs = null
        lastActiveMs = null
        key = null
    }
}

/**
 * 실험적 고개 숙임(끄덕임) 검출. 기준 대비 pitch 가 [nodPitchDeg] 이상 내려갔다가
 * 절반 아래로 돌아오는 데 걸린 시간이 [minMs, maxMs] 이면 1회로 센다.
 * 고개를 숙였다는 사실만으로 졸음을 확정하지 않는다. 기본 설정에서는 졸음 판정에 쓰지 않는다.
 */
class NodDetector(
    private val nodPitchDeg: Double,
    private val minMs: Long,
    private val maxMs: Long,
    private val windowMs: Long,
) {
    private var downSinceMs: Long? = null
    private val nods = ArrayDeque<Long>()

    /** 끄덕임 1회가 완료됐으면 true */
    fun update(relativePitchDeg: Double, ts: Long): Boolean {
        val down = downSinceMs
        if (down == null) {
            if (relativePitchDeg > nodPitchDeg) downSinceMs = ts
            return false
        }
        if (relativePitchDeg < nodPitchDeg / 2) {
            downSinceMs = null
            val duration = ts - down
            if (duration in minMs..maxMs) {
                nods.addLast(ts)
                return true
            }
        }
        return false
    }

    fun count(nowMs: Long): Int {
        while (nods.isNotEmpty() && nowMs - nods.first() > windowMs) nods.removeFirst()
        return nods.size
    }

    fun firstNodMs(): Long? = nods.firstOrNull()

    fun clearCount() = nods.clear()

    /** 관측이 끊기면 진행 중인 숙임은 버린다(공백을 숙임 시간으로 보지 않는다) */
    fun reset() {
        downSinceMs = null
    }
}
