package com.personal.driverguard.domain.platform

/**
 * 시간 출처. 지속 시간 계산은 반드시 [monotonicMs] 로 하고, [wallClockMs] 는 저장·표시용으로만 쓴다.
 */
interface TimeSource {
    /** 단조 증가 시계(Android 에서는 SystemClock.elapsedRealtime) */
    fun monotonicMs(): Long

    /** 벽시계(epoch ms). 사용자가 시간을 바꾸면 뒤로 갈 수 있다. */
    fun wallClockMs(): Long

    /** 단조 시계 시각을 벽시계 시각으로 바꾼다. */
    fun toWallClock(monotonicMs: Long): Long = wallClockMs() - (monotonicMs() - monotonicMs)
}
