package com.personal.driverguard.ui.state

import com.personal.driverguard.data.remote.EventType
import com.personal.driverguard.data.remote.NearestPlace
import com.personal.driverguard.domain.platform.NaviResult
import com.personal.driverguard.domain.platform.NaviRouteState
import com.personal.driverguard.platform.CabinAlert
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * 관찰 중의 흐름이 한 번에 하나씩, 정해진 순서와 간격으로 진행되는지 가상 시간으로 확인한다.
 * 가짜 음성은 한 문장에 2초, 대답은 미리 넣어 둔 대로 돌려준다.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class DriveDirectorTest {
    private val place = NearestPlace(70_000, "REST_AREA", "죽전휴게소", 37.331, 127.104, 11_600, 520)
    private val questions = listOf(
        DrowsyQuiz.Question("안전벨트는 뒷좌석에서도 매야 한다.", true, "안전벨트는 모든 좌석에서 매야 합니다."),
        DrowsyQuiz.Question("비 오는 날에는 제동 거리가 짧아진다.", false, "비가 오면 제동 거리가 길어집니다."),
        DrowsyQuiz.Question("졸음쉼터는 졸릴 때 잠깐 쉬어 가는 곳이다.", true, "졸음쉼터는 졸릴 때 잠깐 쉬어 가는 곳입니다."),
    )

    private class Fake(private val scope: TestScope, private val place: NearestPlace) : DrivePorts {
        /** "시각 종류: 내용" 순서대로 */
        val log = mutableListOf<String>()
        var open = true
        var known = true
        /** 듣기 한 번마다 (대답까지 걸리는 시간, 들은 말). 비어 있으면 끝까지 못 듣는다. */
        val answers = ArrayDeque<Pair<Long, String?>>()
        var navi: NaviRouteState? = NaviRouteState(routing = true, destinationName = "부산", waypointNames = emptyList())
        var addResults = ArrayDeque<NaviResult?>()
        var pending: String? = null
        var drowsy = 0
        var inattention = 0
        var dialed = 0

        fun now() = scope.testScheduler.currentTime
        private fun note(kind: String, text: String = "") { log += "${now()} $kind${if (text.isEmpty()) "" else ": $text"}" }

        override suspend fun voiceReady() = true
        override suspend fun speak(text: String) { note("말", text); delay(SPEECH_MS) }
        override suspend fun hear(windowMs: Long): String? {
            note("듣기")
            val (after, text) = answers.removeFirstOrNull() ?: (windowMs to null)
            delay(minOf(after, windowMs))
            return if (after < windowMs) text else null
        }
        override suspend fun alarm(durationMs: Long) { note("경고음"); delay(durationMs) }
        override fun eyesKnown() = known
        override fun eyesOpen() = open
        override fun cabinAlert(level: CabinAlert) = note("차량", level.name)
        override fun cabinClearInattention() = note("차량", "주황 해제")
        override fun cabinRest() = note("차량", "휴식 모드")
        override fun cabinRestore() = note("차량", "원래대로")
        override fun record(eventType: String) { if (eventType == EventType.DROWSINESS) drowsy++ else inattention++ }
        override fun drowsinessCount() = drowsy
        override fun inattentionSuggestDue() = inattention > 0 && inattention % 3 == 0
        override suspend fun findRestStop() = RestLookup(place, null, navi)
        override suspend fun addRestStop(place: NearestPlace, destination: Boolean, trigger: String): NaviResult? {
            delay(500)
            val result = if (addResults.isEmpty()) NaviResult.Added else addResults.removeFirst()
            if (result == NaviResult.Added) pending = place.name
            note("내비", result.toString())
            return result
        }
        override fun pendingRestName() = pending
        override fun dial119(): Boolean { dialed++; note("119 전화 화면"); return true }
        override fun locationText() = "위도 37.33100, 경도 127.10400"
        override fun nowMs() = now()
    }

    private fun TestScope.start(fake: Fake): DriveDirector {
        val director = DriveDirector(fake, questions)
        backgroundScope.launch(UnconfinedTestDispatcher(testScheduler)) { director.scene.collect { fake.log += "${fake.now()} 장면: $it" } }
        backgroundScope.launch { director.run() }
        runCurrent()
        return director
    }

    private fun Fake.timeOf(prefix: String): Long =
        log.first { it.substringAfter(' ').startsWith(prefix) }.substringBefore(' ').toLong()

    private fun Fake.spoken(): List<String> = log.filter { " 말: " in it }.map { it.substringAfter(" 말: ") }

    @Test
    fun eyesOpenedAfterTheWarningThenRestStopQuizAndArrivalFollowInOrder() = runTest {
        val fake = Fake(this, place)
        fake.answers += listOf(1_000L to "응", 1_500L to "오", 1_500L to "엑스")
        val d = start(fake)
        d.onDrowsy()
        advanceTimeBy(SETTLE_MS)

        // 경고 2초 + 눈 뜸 확인 1초 → 경고가 꺼지고 3초 뒤에 묻는다
        assertEquals(0L, fake.timeOf("장면: DROWSY"))
        val calm = fake.timeOf("장면: OBSERVE", after = 1)
        assertEquals(3_000L, calm)
        assertEquals("경고가 꺼지고 3초 뒤에 경유지를 묻는다", calm + 3_000, fake.timeOf("말: 죽전휴게소를 경유지로 추가할까요?"))
        assertTrue("눈을 떴으므로 119 확인은 하지 않는다", fake.log.none { "UNRESPONSIVE" in it || "경고음" in it })

        // 추가 안내가 끝나고 6초 뒤에 첫 문제, 대답을 다 듣고 정답을 말한 뒤 15초 쉬고 다음 문제
        val addedEnd = fake.timeOf("말: 죽전휴게소를 경유지로 추가했습니다.") + SPEECH_MS
        val first = fake.timeOf("말: ${DrowsyQuiz.INTRO}")
        assertEquals("추가한 뒤 6초 뒤에 퀴즈", addedEnd + 6_000, first)
        val firstFeedbackEnd = fake.timeOf("말: 정답입니다!") + SPEECH_MS
        assertEquals(first + SPEECH_MS + 1_500 + SPEECH_MS, firstFeedbackEnd)
        assertEquals(firstFeedbackEnd + DriveDirector.QUIZ_GAP_MS, fake.timeOf("말: ${DrowsyQuiz.NEXT}"))
        // 쉼터에 도착할 때까지 계속 낸다
        assertEquals(2, fake.spoken().count { it.startsWith("정답입니다") })
        assertTrue(fake.spoken().count { it.startsWith(DrowsyQuiz.NEXT) } >= 2)

        // 도착: 안내와 함께 휴식 모드. 다시 출발을 누를 때까지 기다린다.
        fake.pending = null
        d.onArrived("죽전휴게소", "REST_AREA")
        advanceTimeBy(SETTLE_MS)
        assertEquals(DriveScene.ARRIVED, d.scene.value)
        assertTrue(fake.log.any { it.endsWith("차량: 휴식 모드") })
        assertTrue(fake.spoken().last().startsWith("죽전휴게소에 도착했습니다. 휴식에 알맞은 온도로 맞추고 의자를 눕힐게요."))
        d.tapDepart()
        advanceTimeBy(SETTLE_MS)
        assertEquals(DriveScene.OBSERVE, d.scene.value)
        assertTrue(fake.log.takeLast(3).any { it.endsWith("차량: 원래대로") })
        assertNull(d.arrival.value)
    }

    @Test
    fun eyesKeptClosedFor5SecondsLeadsTo119CheckAndOkLeadsToRestStop() = runTest {
        val fake = Fake(this, place).apply { open = false }
        fake.answers += listOf(2_000L to "괜찮아", 1_000L to "응")
        val d = start(fake)
        d.onDrowsy()
        advanceTimeBy(40_000)

        val warningEnd = fake.timeOf("말: ${DriveDirector.DROWSY_TEXT}") + SPEECH_MS
        assertEquals("경고가 끝나고 5초 동안 계속 감고 있어야 119 확인이 뜬다", warningEnd + DriveDirector.WATCH_MS, fake.timeOf("장면: UNRESPONSIVE"))
        val countdown = fake.timeOf("장면: COUNTDOWN")
        assertTrue("경고음과 안내가 끝난 뒤에 10초를 센다", countdown >= fake.timeOf("말: 괜찮으세요?") + SPEECH_MS)

        // 괜찮다고 한 뒤에야 경유지를 묻는다
        val said = fake.spoken()
        assertTrue(said.indexOf("확인했습니다.") in 0 until said.indexOfFirst { it.startsWith("죽전휴게소를 경유지로 추가할까요?") })
        val calm = fake.log.first { it.endsWith("장면: OBSERVE") && it.substringBefore(' ').toLong() > countdown }.substringBefore(' ').toLong()
        assertEquals("경고가 꺼지고 3초 뒤에 묻는다", calm + DriveDirector.AFTER_OK_MS, fake.timeOf("말: 죽전휴게소를 경유지로 추가할까요?"))
        assertEquals(0, fake.dialed)
        assertEquals("죽전휴게소", fake.pending)
    }

    @Test
    fun noResponseOpensTheDialerAndDoesNotAskAboutARestStop() = runTest {
        val fake = Fake(this, place).apply { open = false }
        val d = start(fake)
        d.onDrowsy()
        advanceTimeBy(SETTLE_MS)

        assertEquals(1, fake.dialed)
        assertEquals(DriveScene.REPORTED, d.scene.value)
        assertEquals(true, d.emergency.value.dialed)
        assertEquals("10초를 다 센 뒤에 연다", fake.timeOf("장면: COUNTDOWN") + DriveDirector.COUNTDOWN_MS, fake.timeOf("119 전화 화면"))
        assertTrue(fake.spoken().none { "경유지" in it })
        d.tapDismiss()
        advanceTimeBy(SETTLE_MS)
        assertEquals(DriveScene.OBSERVE, d.scene.value)
    }

    @Test
    fun touchingOkDuringTheCountdownCancelsTheReport() = runTest {
        val fake = Fake(this, place).apply { open = false }
        val d = start(fake)
        d.onDrowsy()
        advanceTimeBy(14_000)
        assertEquals(DriveScene.COUNTDOWN, d.scene.value)
        assertTrue(d.emergency.value.secondsLeft in 1..10)
        d.tapOk()
        advanceTimeBy(1_000)
        assertEquals(0, fake.dialed)
        assertEquals("확인했습니다.", fake.spoken().last())
    }

    @Test
    fun newDetectionsDuringAFlowAreNotTakenAndNothingOverlaps() = runTest {
        val fake = Fake(this, place)
        fake.answers += listOf(1_000L to "응", 1_000L to "오", 1_000L to "엑스")
        val d = start(fake)
        d.onDrowsy()
        // 흐름이 진행되는 동안(경고 → 질문 → 추가 안내, 약 14초)과 끝난 직후에도 감지가 계속 들어온다
        repeat(60) { advanceTimeBy(500); d.onDrowsy() }
        advanceTimeBy(SETTLE_MS)

        assertEquals("졸음은 한 번만 기록된다", 1, fake.drowsy)
        assertEquals(1, fake.spoken().count { it == DriveDirector.DROWSY_TEXT })
        assertEquals(1, fake.spoken().count { it.startsWith("죽전휴게소를 경유지로 추가할까요?") })
        assertEquals("퀴즈도 끊기지 않고 이어진다", 2, fake.spoken().count { it.startsWith("정답입니다") })
        // 말하기·듣기·경고음은 앞의 것이 끝난 뒤에만 시작한다
        val voice = fake.log.filter { " 말: " in it || it.endsWith(" 듣기") || it.endsWith(" 경고음") }.map { it.substringBefore(' ').toLong() to (" 말: " in it) }
        voice.zipWithNext().forEach { (a, b) -> if (a.second) assertTrue("말이 끝나기 전에 다음 음성이 시작됨: $a → $b", b.first >= a.first + SPEECH_MS) }
    }

    @Test
    fun tappingTheButtonWhileListeningAddsRightAway() = runTest {
        val fake = Fake(this, place)
        val d = start(fake)
        d.onDrowsy()
        advanceTimeBy(9_000) // 경고 → 3초 → 질문 → 듣는 중
        assertEquals(true, d.voice.value.listening)
        d.tapChoice(true)
        advanceTimeBy(600)
        assertEquals("죽전휴게소", fake.pending)
        assertEquals(DriveScene.SUGGEST_ADDED, d.scene.value)
    }

    @Test
    fun sayingNoGoesBackWithoutAddingOrQuizzing() = runTest {
        val fake = Fake(this, place)
        fake.answers += 1_000L to "아니"
        val d = start(fake)
        d.onDrowsy()
        advanceTimeBy(SETTLE_MS)
        assertNull(fake.pending)
        assertEquals(DriveScene.OBSERVE, d.scene.value)
        assertEquals("알겠습니다. 다음에 다시 안내할게요.", fake.spoken().last())
    }

    @Test
    fun naviFailureCanBeRetriedFromTheScreen() = runTest {
        val fake = Fake(this, place)
        fake.answers += 1_000L to "응"
        fake.addResults += listOf(NaviResult.Failed("내비가 응답하지 않았습니다"), NaviResult.Added)
        val d = start(fake)
        d.onDrowsy()
        advanceTimeBy(12_000)
        assertEquals(DriveScene.SUGGEST_FAIL, d.scene.value)
        d.tapChoice(true) // 다시 시도
        advanceTimeBy(1_000)
        assertEquals("죽전휴게소", fake.pending)
    }

    @Test
    fun aRestStopAlreadyOnTheRouteIsNotAskedAgain() = runTest {
        val fake = Fake(this, place).apply { pending = "죽전휴게소" }
        val d = start(fake)
        d.onDrowsy()
        advanceTimeBy(12_000)
        assertTrue(fake.spoken().none { "추가할까요" in it })
        assertEquals("죽전휴게소까지 조금만 더 힘내세요.", fake.spoken().last())
        assertEquals(DriveScene.OBSERVE, d.scene.value)
    }

    @Test
    fun drowsinessInterruptsTheQuizAndTheSameQuestionComesBackLater() = runTest {
        val fake = Fake(this, place)
        fake.answers += 1_000L to "응"
        val d = start(fake)
        d.onDrowsy()
        advanceTimeBy(45_000) // 경유지를 넣고 첫 문제를 지나 두 번째 문제를 말하는 중
        val second = fake.spoken().last()
        assertTrue(second, second.startsWith(DrowsyQuiz.NEXT) && second.endsWith(questions[1].text))
        d.onDrowsy()
        advanceTimeBy(30_000)
        assertEquals(2, fake.drowsy)
        val said = fake.spoken()
        val warningAt = said.indexOf(DriveDirector.DROWSY_REPEATED_TEXT)
        assertTrue("경고가 퀴즈를 끊고 나온다", warningAt > said.indexOf(second))
        assertEquals("경유지를 다시 묻지 않는다", 1, said.count { "추가할까요" in it })
        assertTrue("쉼터까지 가자고만 한다", said.drop(warningAt).any { it == "죽전휴게소까지 조금만 더 힘내세요." })
        assertTrue("같은 문제를 다시 낸다", said.drop(warningAt).any { it.endsWith(questions[1].text) })
    }

    @Test
    fun drowsinessRightAfterAFlowIsNotTakenForAWhile() = runTest {
        val fake = Fake(this, place)
        fake.answers += listOf(1_000L to "아니", 1_000L to "아니")
        val d = start(fake)
        d.onDrowsy()
        advanceTimeBy(15_000) // 경고 → 질문 → '아니' → 끝(약 11초)
        assertEquals(DriveScene.OBSERVE, d.scene.value)
        d.onDrowsy()
        advanceTimeBy(2_000)
        assertEquals("방금 확인한 직후에는 다시 경고하지 않는다", 1, fake.drowsy)
        advanceTimeBy(DriveDirector.DROWSY_QUIET_MS)
        d.onDrowsy()
        advanceTimeBy(1_000)
        assertEquals("쉬는 시간이 지나면 다시 받는다", 2, fake.drowsy)
    }

    @Test
    fun lookingAwayShowsTheAlertUntilTheDriverLooksBack() = runTest {
        val fake = Fake(this, place)
        val d = start(fake)
        d.onLookAway()
        advanceTimeBy(5_000)
        assertEquals(DriveScene.LOOK, d.scene.value)
        assertEquals(listOf(DriveDirector.LOOK_TEXT), fake.spoken())
        d.onLookBack()
        advanceTimeBy(500)
        assertEquals(DriveScene.OBSERVE, d.scene.value)
        assertTrue(fake.log.any { it.endsWith("차량: 주황 해제") })
        assertEquals(1, fake.inattention)
    }

    @Test
    fun thirdLookAwaySuggestsARestStopThreeSecondsAfterLookingBack() = runTest {
        val fake = Fake(this, place).apply { inattention = 2 }
        fake.answers += 1_000L to "아니"
        val d = start(fake)
        d.onLookAway()
        advanceTimeBy(3_000)
        d.onLookBack()
        advanceTimeBy(200)
        val back = fake.now()
        advanceTimeBy(SETTLE_MS)
        val asked = fake.timeOf("말: 죽전휴게소를 경유지로 추가할까요?")
        assertTrue("정면을 본 뒤 3초쯤 지나 묻는다: ${asked - back}", asked - back in 2_800..3_100)
        assertTrue(fake.log.any { it.endsWith("장면: SUGGEST_REPEAT") })
    }

    @Test
    fun withoutEyeStateThe119CheckIsSkipped() = runTest {
        val fake = Fake(this, place).apply { known = false; open = false }
        fake.answers += 1_000L to "아니"
        val d = start(fake)
        d.onDrowsy()
        advanceTimeBy(SETTLE_MS)
        assertFalse(fake.log.any { "UNRESPONSIVE" in it })
        assertEquals(0, fake.dialed)
    }

    private fun Fake.timeOf(prefix: String, after: Int): Long =
        log.filter { it.substringAfter(' ').startsWith(prefix) }[after].substringBefore(' ').toLong()

    private companion object {
        const val SPEECH_MS = 2_000L
        /** 흐름 하나가 끝나고도 남는 시간 */
        const val SETTLE_MS = 90_000L
    }
}
