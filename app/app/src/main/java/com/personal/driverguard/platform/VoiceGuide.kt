package com.personal.driverguard.platform

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import androidx.core.content.ContextCompat
import com.personal.driverguard.platform.pleos.PleosVoiceGuide
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import java.util.Locale
import java.util.UUID
import kotlin.coroutines.resume

/**
 * 음성 안내(TTS)와 짧은 음성 응답(STT). 운전 중 화면을 누르지 않고 경고를 듣고, '응'으로 답하게 한다.
 * 구현: Pleos TextToSpeech·SpeechToText(차량 스피커·마이크). Pleos 가 아니면 Android 기본 음성으로 대신한다([AutoVoiceGuide]).
 */
interface VoiceGuide {
    suspend fun connect(): Boolean

    /** 말하기가 끝날 때까지 기다린다(시간 제한 있음) */
    suspend fun speak(text: String)

    /** 한 번 듣고 인식한 문장을 돌려준다. 못 들었거나 쓸 수 없으면 null. */
    suspend fun listen(timeoutMs: Long): String?

    fun stop()

    fun release()

    companion object {
        private val YES = listOf("응", "네", "예", "그래", "좋아", "추가", "어 ", "웅", "콜", "오케이", "해 줘", "해줘", "넣어")
        private val NO = listOf("아니", "싫어", "괜찮아", "나중", "됐어", "필요 없")

        /** 짧은 대답을 예/아니오로 본다. 판단할 수 없으면 null. */
        fun yesNo(text: String?): Boolean? {
            val t = text?.trim()?.lowercase() ?: return null
            if (t.isEmpty()) return null
            if (NO.any { t.contains(it) }) return false
            if (YES.any { t.contains(it.trim()) } || t == "어" || t == "yes" || t == "ok") return true
            return null
        }
    }
}

/** Pleos 음성을 먼저 쓰고, 안 되면 Android 기본 음성으로 대신한다. */
class AutoVoiceGuide(context: Context) : VoiceGuide {
    private val pleos = runCatching { PleosVoiceGuide(context) }.getOrNull()
    private val android = AndroidVoiceGuide(context)
    @Volatile private var active: VoiceGuide? = null

    override suspend fun connect(): Boolean {
        active?.let { return true }
        val p = pleos
        if (p != null && runCatching { p.connect() }.getOrDefault(false)) { active = p; return true }
        if (android.connect()) { active = android; return true }
        return false
    }

    override suspend fun speak(text: String) { if (connect()) active?.speak(text) }
    override suspend fun listen(timeoutMs: Long): String? = if (connect()) active?.listen(timeoutMs) else null
    override fun stop() { active?.stop() }
    override fun release() { runCatching { pleos?.release() }; android.release() }
}

/** Android 기본 TextToSpeech·SpeechRecognizer. 시연 태블릿에서 태블릿 스피커로 안내한다. */
class AndroidVoiceGuide(context: Context) : VoiceGuide {
    private val appContext = context.applicationContext
    private var tts: TextToSpeech? = null
    private var ready: CompletableDeferred<Boolean>? = null
    private var recognizer: SpeechRecognizer? = null

    override suspend fun connect(): Boolean {
        val r = ready ?: CompletableDeferred<Boolean>().also { d ->
            ready = d
            withContext(Dispatchers.Main) {
                tts = runCatching {
                    TextToSpeech(appContext) { status ->
                        val ok = status == TextToSpeech.SUCCESS && tts?.setLanguage(Locale.KOREAN).let { it == null || it >= TextToSpeech.LANG_AVAILABLE }
                        d.complete(ok)
                    }
                }.getOrElse { d.complete(false); null }
            }
        }
        return withTimeoutOrNull(CONNECT_TIMEOUT_MS) { r.await() } ?: false
    }

    override suspend fun speak(text: String) {
        val t = tts ?: return
        val id = UUID.randomUUID().toString()
        val done = CompletableDeferred<Unit>()
        t.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) = Unit
            override fun onDone(utteranceId: String?) { if (utteranceId == id) done.complete(Unit) }
            @Deprecated("Deprecated in Java")
            override fun onError(utteranceId: String?) { done.complete(Unit) }
            override fun onStop(utteranceId: String?, interrupted: Boolean) { done.complete(Unit) }
        })
        t.speak(text, TextToSpeech.QUEUE_FLUSH, null, id)
        withTimeoutOrNull(SPEAK_TIMEOUT_MS) { done.await() }
    }

    override suspend fun listen(timeoutMs: Long): String? {
        val granted = ContextCompat.checkSelfPermission(appContext, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
        if (!granted || !SpeechRecognizer.isRecognitionAvailable(appContext)) return null
        return withContext(Dispatchers.Main) {
            withTimeoutOrNull(timeoutMs) {
                suspendCancellableCoroutine { cont ->
                    val sr = SpeechRecognizer.createSpeechRecognizer(appContext).also { recognizer = it }
                    fun finish(text: String?) {
                        if (cont.isActive) cont.resume(text)
                        runCatching { sr.destroy() }
                        if (recognizer === sr) recognizer = null
                    }
                    sr.setRecognitionListener(object : RecognitionListener {
                        override fun onResults(results: Bundle?) =
                            finish(results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull())
                        override fun onError(error: Int) = finish(null)
                        override fun onReadyForSpeech(params: Bundle?) = Unit
                        override fun onBeginningOfSpeech() = Unit
                        override fun onRmsChanged(rmsdB: Float) = Unit
                        override fun onBufferReceived(buffer: ByteArray?) = Unit
                        override fun onEndOfSpeech() = Unit
                        override fun onPartialResults(partialResults: Bundle?) = Unit
                        override fun onEvent(eventType: Int, params: Bundle?) = Unit
                    })
                    val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
                        .putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                        .putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ko-KR")
                        .putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
                    runCatching { sr.startListening(intent) }.onFailure { Log.w(TAG, "음성 인식 시작 실패", it); finish(null) }
                    cont.invokeOnCancellation { runCatching { sr.cancel(); sr.destroy() } }
                }
            }
        }
    }

    override fun stop() {
        runCatching { tts?.stop() }
        runCatching { recognizer?.cancel() }
    }

    override fun release() {
        runCatching { tts?.shutdown() }
        tts = null
        ready = null
    }

    private companion object {
        const val TAG = "AndroidVoiceGuide"
        const val CONNECT_TIMEOUT_MS = 4_000L
        const val SPEAK_TIMEOUT_MS = 12_000L
    }
}
