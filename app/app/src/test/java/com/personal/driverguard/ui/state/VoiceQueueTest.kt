package com.personal.driverguard.ui.state

import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class VoiceQueueTest {
    private val log = mutableListOf<String>()
    private var stops = 0
    private val queue = VoiceQueue { stops++ }

    /** [ms] 동안 말하는 것처럼 기다린다 */
    private suspend fun speak(name: String, ms: Long) {
        log += "$name 시작"
        delay(ms)
        log += "$name 끝"
    }

    @Test
    fun sameLevelWaitsInOrderWithoutOverlapping() = runTest {
        val a = async { queue.turn(VoiceQueue.DIALOG) { speak("확인했습니다", 3_000) } }
        val b = async { queue.turn(VoiceQueue.DIALOG) { speak("경유지 질문", 2_000) } }
        advanceUntilIdle()
        assertEquals(listOf("확인했습니다 시작", "확인했습니다 끝", "경유지 질문 시작", "경유지 질문 끝"), log)
        assertTrue(a.await() != null && b.await() != null)
        assertEquals("끊긴 것이 없으면 멈추지 않는다", 0, stops)
    }

    @Test
    fun urgentAlertCutsInAndTheCutTurnReturnsNull() = runTest {
        val question = async { queue.turn(VoiceQueue.DIALOG) { speak("경유지 질문", 5_000); "응" } }
        runCurrent()
        delay(1_000)
        val alert = async { queue.turn(VoiceQueue.LOOK_ALERT) { speak("전방 주시", 2_000) } }
        advanceUntilIdle()
        assertNull("경고에 끊긴 질문은 대답이 없다", question.await())
        assertTrue(alert.await() != null)
        assertEquals(listOf("경유지 질문 시작", "전방 주시 시작", "전방 주시 끝"), log)
        assertEquals("끊긴 소리를 멈춘다", 1, stops)
    }

    @Test
    fun urgentTurnJumpsAheadOfWaitingTurns() = runTest {
        launch { queue.turn(VoiceQueue.DIALOG) { speak("확인했습니다", 3_000) } }
        runCurrent()
        launch { queue.turn(VoiceQueue.QUIZ) { speak("퀴즈", 1_000) } }
        launch { queue.turn(VoiceQueue.DIALOG) { speak("경유지 질문", 1_000) } }
        runCurrent()
        launch { queue.turn(VoiceQueue.DROWSY_ALERT) { speak("졸음 경고", 1_000) } }
        advanceUntilIdle()
        assertEquals(
            "졸음 경고가 끊고 들어가고, 그다음 급한 순서(질문 → 퀴즈)",
            listOf("확인했습니다 시작", "졸음 경고 시작", "졸음 경고 끝", "경유지 질문 시작", "경유지 질문 끝", "퀴즈 시작", "퀴즈 끝"),
            log,
        )
    }

    @Test
    fun repeatedAlertIsSkippedWhileOneIsSpeaking() = runTest {
        launch { queue.turn(VoiceQueue.DROWSY_ALERT, skipIfBusy = true) { speak("졸음 경고", 3_000) } }
        runCurrent()
        val again = async { queue.turn(VoiceQueue.DROWSY_ALERT, skipIfBusy = true) { speak("졸음 경고 2", 3_000) } }
        val look = async { queue.turn(VoiceQueue.LOOK_ALERT, skipIfBusy = true) { speak("전방 주시", 1_000) } }
        advanceUntilIdle()
        assertNull(again.await())
        assertNull("더 급한 경고가 나오는 중이면 덜 급한 경고는 건너뛴다", look.await())
        assertEquals(listOf("졸음 경고 시작", "졸음 경고 끝"), log)
    }

    @Test
    fun cancelledCallerStopsSoundAndLetsTheNextOneGo() = runTest {
        val screen = launch { queue.turn(VoiceQueue.DIALOG) { speak("경유지 질문", 5_000) } }
        runCurrent()
        val next = async { queue.turn(VoiceQueue.DIALOG) { speak("도착 안내", 1_000) } }
        delay(1_000)
        screen.cancel() // 화면을 떠남
        advanceUntilIdle()
        assertEquals(listOf("경유지 질문 시작", "도착 안내 시작", "도착 안내 끝"), log)
        assertEquals(1, stops)
        assertTrue(next.await() != null)
    }

    @Test
    fun waitingCallerThatLeavesDoesNotBlockTheQueue() = runTest {
        val gate = CompletableDeferred<Unit>()
        launch { queue.turn(VoiceQueue.DIALOG) { gate.await(); log += "첫 음성 끝" } }
        runCurrent()
        val leaving = launch { queue.turn(VoiceQueue.DIALOG) { log += "떠난 쪽" } }
        val staying = async { queue.turn(VoiceQueue.DIALOG) { log += "남은 쪽" } }
        runCurrent()
        leaving.cancel()
        gate.complete(Unit)
        advanceUntilIdle()
        assertEquals(listOf("첫 음성 끝", "남은 쪽"), log)
        assertTrue(staying.await() != null)
    }

    @Test
    fun busyForTellsTheQuizToWait() = runTest {
        launch { queue.turn(VoiceQueue.DIALOG) { delay(1_000) } }
        runCurrent()
        assertTrue(queue.busyFor(VoiceQueue.QUIZ))
        advanceUntilIdle()
        assertTrue(!queue.busyFor(VoiceQueue.QUIZ))
    }
}
