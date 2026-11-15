package com.personal.driverguard.ui.state

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.job
import kotlinx.coroutines.withContext

/**
 * 말하기와 듣기의 차례를 정한다. 스피커와 마이크는 하나라서 한 번에 한 가지만 한다
 * (겹치면 안내가 중간에 끊기거나, 안내 음성을 운전자의 대답으로 듣는다).
 *
 *  - 급한 음성(숫자가 큰 [priority])은 덜 급한 음성을 끊고 먼저 나간다. 끊긴 쪽의 [turn] 은 null 을 돌려준다.
 *  - 같은 급이나 덜 급한 음성은 앞의 것이 끝날 때까지 기다린다. 기다리는 것끼리는 급한 순, 먼저 온 순이다.
 *  - [skipIfBusy] 면 같은 급 이상이 진행 중일 때 기다리지 않고 건너뛴다(반복되는 경고가 쌓이지 않게).
 *  - 차례가 끊기거나 부른 쪽이 취소되면 [stop] 을 불러 나오던 소리와 듣기를 멈춘다.
 *
 * Android 에 의존하지 않는다(단위 테스트로 검증).
 */
class VoiceQueue(private val stop: () -> Unit) {
    private class Turn(val priority: Int, val seq: Long) {
        val granted = CompletableDeferred<Unit>()
        @Volatile var job: Job? = null
        @Volatile var preempted = false
    }

    private val lock = Any()
    private val waiting = ArrayList<Turn>()
    private var running: Turn? = null
    private var seq = 0L

    /** [priority] 보다 급한 음성이 진행 중이거나, 기다리는 음성이 있는지 */
    fun busyFor(priority: Int): Boolean = synchronized(lock) {
        waiting.isNotEmpty() || (running?.let { it.priority > priority } == true)
    }

    /** 차례를 받아 [block] 을 실행한다. 더 급한 음성에 밀려 끊기거나 [skipIfBusy] 로 건너뛰면 null. */
    suspend fun <T> turn(priority: Int, skipIfBusy: Boolean = false, block: suspend () -> T): T? {
        val turn: Turn
        synchronized(lock) {
            val current = running
            if (skipIfBusy && current != null && current.priority >= priority) return null
            turn = Turn(priority, seq++)
            if (current == null) {
                running = turn
                turn.granted.complete(Unit)
            } else {
                waiting += turn
                if (current.priority < priority) {
                    current.preempted = true
                    current.job?.cancel()
                }
            }
        }
        try {
            turn.granted.await()
        } catch (e: CancellationException) {
            synchronized(lock) {
                waiting.remove(turn)
                if (running === turn) handOff(turn)
            }
            throw e
        }
        val job = SupervisorJob(currentCoroutineContext().job)
        turn.job = job
        if (turn.preempted) job.cancel()
        return try {
            withContext(job) { block() }
        } catch (e: CancellationException) {
            // 부른 쪽이 취소된 것이면 그대로 올려 보내고, 더 급한 음성에 밀린 것이면 null
            currentCoroutineContext().ensureActive()
            null
        } finally {
            if (job.isCancelled) runCatching { stop() }
            job.complete()
            synchronized(lock) { handOff(turn) }
        }
    }

    /** [done] 이 끝났으면 기다리는 것 중 가장 급하고 먼저 온 것에 차례를 넘긴다. lock 안에서 부른다. */
    private fun handOff(done: Turn) {
        if (running !== done) return
        val next = waiting.maxWithOrNull(compareBy<Turn> { it.priority }.thenByDescending { it.seq })
        running = next
        if (next != null) {
            waiting.remove(next)
            next.granted.complete(Unit)
        }
    }

    companion object {
        const val QUIZ = 0
        const val DIALOG = 1
        const val LOOK_ALERT = 2
        const val DROWSY_ALERT = 3
        const val EMERGENCY = 4
    }
}
