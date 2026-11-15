package com.personal.driverguard.ui.state

import com.personal.driverguard.platform.VoiceGuide
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/** 음성 안내에 쓰는 조사 고르기와, 쉼터 제안에 대한 대답(예/아니오) 알아듣기 */
class KoreanParticleTest {
    @Test
    fun particlesFollowTheFinalConsonant() {
        assertEquals("를", HawkeyeViewModel.josaReul("기흥휴게소"))
        assertEquals("을", HawkeyeViewModel.josaReul("안성 졸음쉼터 앞"))
        assertEquals("로", HawkeyeViewModel.josaRo("기흥휴게소"))
        assertEquals("로", HawkeyeViewModel.josaRo("서울"))
        assertEquals("으로", HawkeyeViewModel.josaRo("부산"))
    }

    @Test
    fun yesAndNoAreHeardInTheWaysPeopleSayThem() {
        listOf("응", "네", "어", "그래", "맞아", "그렇게 해", "추가해 줘", "좋아요").forEach { assertEquals(it, true, VoiceGuide.yesNo(it)) }
        listOf("아니", "아니요", "싫어", "나중에", "괜찮아", "하지 마").forEach { assertEquals(it, false, VoiceGuide.yesNo(it)) }
        listOf(null, "", "  ").forEach { assertNull(VoiceGuide.yesNo(it)) }
    }
}
