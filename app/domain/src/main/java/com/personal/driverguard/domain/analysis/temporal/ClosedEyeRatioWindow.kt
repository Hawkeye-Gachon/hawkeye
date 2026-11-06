package com.personal.driverguard.domain.analysis.temporal

enum class RatioStatus {
    /** 평가 창 길이만큼 관측 시간이 쌓이지 않음 */
    MEASURING,

    /** 유효 관측률이 기준보다 낮아 판정 보류 */
    LOW_COVERAGE,
    AVAILABLE,
}

data class ClosedEyeRatio(
    val status: RatioStatus,
    /** 유효 관찰 구간 중 눈 감김 시간 / 유효 관찰 시간. 유효 관찰이 없으면 null */
    val ratio: Double?,
    /** 유효 관찰 시간 / 평가 창 길이 */
    val coverage: Double,
    /** 실제로 평가에 쓴 창 길이(측정 중이면 창보다 짧다) */
    val windowUsedMs: Long,
    val validMs: Long,
    val closedMs: Long,
)

/**
 * 최근 [windowMs] 동안의 눈 감김 비율(TemporalFeatureAccumulator).
 * 표준 PERCLOS 가 아니라 EAR 임계값 기반의 'PERCLOS 유사 지표'다(눈꺼풀 폐쇄 정의·측정 조건을 검증하지 않았다).
 *
 * 관측 → 시간 구간 변환 규칙(엔진이 적용한 뒤 [add] 로 넘긴다):
 *  - 관측 i 의 눈 상태는 [t_i, t_{i+1}) 구간을 대표한다. 단 t_{i+1} − t_i 가 연속 관측 기준을 넘으면 그 구간은 관측 없음이다.
 *  - 눈 지표가 무효인 구간, 얼굴 미검출 구간, 프레임 공백은 분자와 유효 분모 모두에서 빠진다.
 * 창 경계에 걸친 구간은 경계에서 잘라 계산한다.
 */
class ClosedEyeRatioWindow(
    private val windowMs: Long,
    private val minCoverage: Double,
) {
    private class Segment(val start: Long, val end: Long, val closed: Boolean)

    private val segments = ArrayDeque<Segment>()
    private var originMs: Long? = null

    /** 평가 창의 시작 기준(세션 시작·휴식 후 재개 시각). 이보다 이전 구간은 없다. */
    fun reset(originMs: Long) {
        segments.clear()
        this.originMs = originMs
    }

    /** 유효한 눈 상태 구간 [start, end) */
    fun add(start: Long, end: Long, closed: Boolean) {
        if (end <= start) return
        val last = segments.lastOrNull()
        if (last != null && last.closed == closed && last.end == start) {
            segments.removeLast()
            segments.addLast(Segment(last.start, end, closed))
        } else {
            segments.addLast(Segment(start, end, closed))
        }
    }

    fun evaluate(nowMs: Long): ClosedEyeRatio {
        val windowStart = nowMs - windowMs
        while (segments.isNotEmpty() && segments.first().end <= windowStart) segments.removeFirst()
        var valid = 0L
        var closed = 0L
        for (s in segments) {
            val from = maxOf(s.start, windowStart)
            val to = minOf(s.end, nowMs)
            if (to <= from) continue
            valid += to - from
            if (s.closed) closed += to - from
        }
        val origin = originMs ?: nowMs
        val used = (nowMs - maxOf(origin, windowStart)).coerceAtLeast(0)
        val coverage = valid.toDouble() / windowMs
        val status = when {
            used < windowMs -> RatioStatus.MEASURING
            coverage < minCoverage -> RatioStatus.LOW_COVERAGE
            else -> RatioStatus.AVAILABLE
        }
        return ClosedEyeRatio(status, if (valid > 0) closed.toDouble() / valid else null, coverage, used, valid, closed)
    }
}
