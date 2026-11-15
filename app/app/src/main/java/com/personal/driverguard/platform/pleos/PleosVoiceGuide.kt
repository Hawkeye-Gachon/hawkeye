package com.personal.driverguard.platform.pleos

import ai.pleos.playground.stt.SpeechToText
import ai.pleos.playground.stt.listener.ResultListener
import ai.pleos.playground.tts.TextToSpeech
import ai.pleos.playground.tts.constant.ErrorType
import ai.pleos.playground.tts.constant.Mode
import ai.pleos.playground.tts.listener.EventListener
import ai.pleos.playground.tts.listener.OnInitializeListener
import android.content.Context
import android.util.Log
import com.personal.driverguard.platform.VoiceGuide
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.withTimeoutOrNull
import java.util.Locale

/**
 * Pleos TextToSpeech·SpeechToText. 안내는 차량 스피커로 나가고, 대답은 차량 마이크로 듣는다.
 * HYBRID 모드(서버가 안 되면 기기 안에서 처리). 앱 키 등록(registerApp)은 하지 않는다. 서버 인식이 키를 요구하면 기기 처리로만 동작한다.
 * 경고는 다른 안내보다 먼저 말하도록 speak(text, true) 로 보낸다.
 */
class PleosVoiceGuide(context: Context) : VoiceGuide {
    private val appContext = context.applicationContext
    private val tts by lazy { TextToSpeech(appContext, Mode.HYBRID) }
    private val stt by lazy { SpeechToText(appContext, ai.pleos.playground.stt.constant.Mode.HYBRID) }

    @Volatile private var ttsReady = false
    @Volatile private var sttReady = false
    @Volatile private var unavailable = false

    override suspend fun connect(): Boolean {
        if (ttsReady) return true
        if (unavailable) return false
        val ok = CompletableDeferred<Boolean>()
        try {
            tts.initialize(object : OnInitializeListener {
                override fun onSuccess() { ok.complete(true) }
                override fun onFailure(error: ErrorType) { Log.w(TAG, "TTS 초기화 실패: $error"); ok.complete(false) }
            })
        } catch (e: Throwable) {
            Log.w(TAG, "Pleos TTS 를 쓸 수 없습니다", e)
            ok.complete(false)
        }
        ttsReady = withTimeoutOrNull(CONNECT_TIMEOUT_MS) { ok.await() } ?: false
        if (!ttsReady) { unavailable = true; return false }
        runCatching { tts.setLocale(Locale.KOREA) }
        // 음성 인식은 실패해도 안내는 쓴다
        val sttOk = CompletableDeferred<Boolean>()
        try {
            stt.initialize(object : ai.pleos.playground.stt.listener.OnInitializeListener {
                override fun onSuccess() { sttOk.complete(true) }
                override fun onFailure(error: ai.pleos.playground.stt.constant.ErrorType) { Log.w(TAG, "STT 초기화 실패: $error"); sttOk.complete(false) }
            })
        } catch (e: Throwable) {
            sttOk.complete(false)
        }
        sttReady = withTimeoutOrNull(CONNECT_TIMEOUT_MS) { sttOk.await() } ?: false
        if (sttReady) runCatching { stt.setLocale(Locale.KOREA) }
        return true
    }

    override suspend fun speak(text: String) {
        if (!ttsReady) return
        val done = CompletableDeferred<Unit>()
        val listener = object : EventListener {
            override fun onStart() = Unit
            override fun onReady() = Unit
            override fun onDone() { done.complete(Unit) }
            override fun onError(error: ErrorType) { done.complete(Unit) }
            override fun onStop() { done.complete(Unit) }
            override fun onUpdatedRms(rms: Double) = Unit
        }
        runCatching {
            tts.addEventListener(listener)
            tts.speak(text, true)
        }.onFailure { done.complete(Unit) }
        withTimeoutOrNull(SPEAK_TIMEOUT_MS) { done.await() }
        runCatching { tts.removeEventListener(listener) }
    }

    override suspend fun listen(timeoutMs: Long): String? {
        if (!sttReady) return null
        val result = CompletableDeferred<String?>()
        var last: String? = null
        val listener = object : ResultListener {
            override fun onUpdated(text: String, isFinal: Boolean) {
                last = text
                if (isFinal) result.complete(text)
            }
            override fun onUpdatedEpdData(start: Long, end: Long) = Unit
            override fun onStartedRecognition() = Unit
            override fun onEndedRecognition() { result.complete(last) }
            override fun onError(error: ai.pleos.playground.stt.constant.ErrorType) { result.complete(null) }
            override fun onReady() = Unit
            override fun onUpdatedRms(rms: Double) = Unit
        }
        return try {
            stt.addListener(listener)
            stt.request()
            withTimeoutOrNull(timeoutMs) { result.await() } ?: last
        } catch (e: Throwable) {
            Log.w(TAG, "음성 인식 실패", e)
            null
        } finally {
            runCatching { stt.stop() }
            runCatching { stt.removeListener(listener) }
        }
    }

    override fun stop() {
        runCatching { tts.stop() }
        runCatching { stt.stop() }
    }

    override fun release() {
        if (ttsReady) runCatching { tts.release() }
        if (sttReady) runCatching { stt.release() }
        ttsReady = false
        sttReady = false
    }

    private companion object {
        const val TAG = "PleosVoiceGuide"
        const val CONNECT_TIMEOUT_MS = 3_000L
        const val SPEAK_TIMEOUT_MS = 12_000L
    }
}
