package com.personal.driverguard.ui.state

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class DrowsyQuizTest {
    @Test
    fun hearsOAndXInTheWaysPeopleSayThem() {
        listOf("오", "오!", "O", "5", "동그라미", "맞아요", "네", "응", "오 입니다").forEach { assertEquals(it, true, DrowsyQuiz.parse(it)) }
        listOf("엑스", "X", "엑스요", "틀렸어", "아니", "아니오", "땡").forEach { assertEquals(it, false, DrowsyQuiz.parse(it)) }
    }

    @Test
    fun unclearOrMissingSpeechIsNotAnAnswer() {
        listOf(null, "", "  ", "음", "잠깐만", "뭐라고").forEach { assertNull("$it", DrowsyQuiz.parse(it)) }
    }

    @Test
    fun feedbackSaysTheRightAnswerWhenWrongOrSilent() {
        val q = DrowsyQuiz.Question("비 오는 날에는 제동 거리가 짧아진다.", false, "비가 오면 제동 거리가 길어집니다.")
        assertEquals("정답입니다!", DrowsyQuiz.feedback(q, false))
        assertTrue(DrowsyQuiz.feedback(q, true).contains("정답은 엑스입니다. 비가 오면 제동 거리가 길어집니다."))
        assertTrue(DrowsyQuiz.feedback(q, null).startsWith("대답을 듣지 못했어요."))
    }

    @Test
    fun questionBankIsBalancedAndShortEnoughToSay() {
        val qs = DrowsyQuiz.questions
        assertTrue(qs.size >= 10)
        assertEquals("O 와 X 가 반반이어야 찍어서 맞히기 어렵다", qs.size / 2, qs.count { it.answer })
        assertTrue("문제는 한 문장으로 짧게", qs.all { it.text.length <= 30 && it.text.endsWith(".") })
        assertEquals("같은 문제가 없어야 한다", qs.size, qs.map { it.text }.toSet().size)
    }
}
