package com.personal.driverguard.platform

import android.media.AudioManager
import android.media.ToneGenerator
import kotlinx.coroutines.delay

/**
 * '이제 말하세요' 신호음. 음성 질문이 끝나고 대답을 듣기 직전에 짧게 울린다.
 * 신호음이 끝난 뒤에 듣기 시작하므로 앞선 음성 안내나 신호음 자체가 대답으로 잡히지 않는다.
 * 소리를 낼 수 없는 기기에서는 같은 시간만 쉬고 넘어간다.
 */
object ListenCue {
    private const val BEEP_MS = 160
    /** 신호음이 끝나고 스피커 잔향이 사라질 때까지 기다리는 시간 */
    private const val SETTLE_MS = 320L

    suspend fun beep() {
        val tone = runCatching { ToneGenerator(AudioManager.STREAM_MUSIC, 85) }.getOrNull()
        try {
            tone?.let { runCatching { it.startTone(ToneGenerator.TONE_PROP_BEEP, BEEP_MS) } }
            delay(BEEP_MS + SETTLE_MS)
        } finally {
            tone?.let { runCatching { it.release() } }
        }
    }
}
