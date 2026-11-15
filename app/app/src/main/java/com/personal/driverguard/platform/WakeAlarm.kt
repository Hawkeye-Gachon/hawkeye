package com.personal.driverguard.platform

import android.media.AudioManager
import android.media.ToneGenerator
import kotlinx.coroutines.delay

/**
 * 반응이 없는 운전자를 깨우는 큰 경고음. 알람 음량 채널로 낸다(미디어 음량을 줄여 둬도 들리게).
 * 기기에서 소리를 낼 수 없으면 조용히 넘어간다(음성 안내와 차량 조치는 따로 나간다).
 */
class WakeAlarm {
    private var tone: ToneGenerator? = null

    /** [durationMs] 동안 끊어 울린다. 취소되면 바로 멈춘다. */
    suspend fun ring(durationMs: Long) {
        val generator = runCatching { ToneGenerator(AudioManager.STREAM_ALARM, ToneGenerator.MAX_VOLUME) }.getOrNull() ?: run {
            delay(durationMs)
            return
        }
        tone = generator
        try {
            var left = durationMs
            while (left > 0) {
                runCatching { generator.startTone(ToneGenerator.TONE_CDMA_ALERT_CALL_GUARD, BEEP_MS) }
                delay(STEP_MS.coerceAtMost(left))
                left -= STEP_MS
            }
        } finally {
            stop()
        }
    }

    fun stop() {
        tone?.let { runCatching { it.stopTone(); it.release() } }
        tone = null
    }

    private companion object {
        const val BEEP_MS = 450
        const val STEP_MS = 600L
    }
}
