package com.personal.driverguard.platform

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
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
        private val YES = listOf(
            "응", "네", "예", "그래", "좋아", "추가", "어 ", "웅", "콜", "오케이", "해 줘", "해줘", "넣어",
            "맞", "그렇", "그럼", "당연", "부탁", "좋습니다", "좋죠", "예스", "ㅇㅇ",
        )
        private val NO = listOf("아니", "싫어", "괜찮아", "나중", "됐어", "필요 없", "안 해", "안해", "하지 마", "하지마", "말고")

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

/** Android 기본 TextToSpeech·SpeechRecognizer. Pleos 음성을 쓸 수 없는 기기에서 기기 스피커로 안내한다. */
class AndroidVoiceGuide(context: Context) : VoiceGuide {
    private val appContext = context.applicationContext
    private var tts: TextToSpeech? = null
    private var ready: CompletableDeferred<Boolean>? = null
    private var recognizer: SpeechRecognizer? = null
    private var voiceChosen = false

    override suspend fun connect(): Boolean {
        val r = ready ?: CompletableDeferred<Boolean>().also { d ->
            ready = d
            withContext(Dispatchers.Main) {
                tts = runCatching {
                    TextToSpeech(appContext) { status ->
                        if (status != TextToSpeech.SUCCESS) Log.w(TAG, "음성 엔진 초기화 실패: $status")
                        // 음성 고르기는 엔진 연결이 끝난 뒤여야 하므로(이 콜백 안에서는 tts 가 아직 없을 수 있다) 아래에서 한다
                        d.complete(status == TextToSpeech.SUCCESS)
                    }
                }.getOrElse { Log.w(TAG, "음성 엔진을 만들지 못했습니다", it); d.complete(false); null }
            }
        }
        val ok = withTimeoutOrNull(CONNECT_TIMEOUT_MS) { r.await() } ?: false
        if (ok && !voiceChosen) {
            voiceChosen = true
            tts?.let { chooseKoreanVoice(it) }
        }
        return ok
    }

    /**
     * 기기 안에서 합성하는 한국어 음성을 고른다.
     * 엔진 기본값이 서버 음성이면(예: Google TTS 의 ko-kr-x-ism-server) 네트워크 상태에 따라 합성이 실패해 아무 소리도 나지 않는다.
     * 차량이나 태블릿은 네트워크가 없을 수 있으므로 설치된 오프라인 음성을 우선한다. 없으면 엔진 기본 한국어를 쓴다.
     */
    private fun chooseKoreanVoice(t: TextToSpeech) {
        val korean = runCatching { t.voices }.getOrNull().orEmpty().filter { it.locale.language == Locale.KOREAN.language }
        val offline = korean.filter { !it.isNetworkConnectionRequired && TextToSpeech.Engine.KEY_FEATURE_NOT_INSTALLED !in it.features }
        val pick = offline.maxByOrNull { it.quality }
        if (pick != null) {
            val set = runCatching { t.setVoice(pick) }.getOrDefault(TextToSpeech.ERROR)
            Log.i(TAG, "한국어 오프라인 음성 ${pick.name} 선택(결과 $set, 후보 ${offline.size}/${korean.size})")
            if (set == TextToSpeech.SUCCESS) return
        }
        val lang = runCatching { t.setLanguage(Locale.KOREAN) }.getOrDefault(TextToSpeech.LANG_NOT_SUPPORTED)
        Log.w(TAG, "오프라인 한국어 음성이 없어 엔진 기본 음성을 씁니다(setLanguage=$lang, 한국어 음성 ${korean.map { it.name }})")
    }

    override suspend fun speak(text: String) {
        val t = tts ?: return
        val id = UUID.randomUUID().toString()
        val done = CompletableDeferred<Unit>()
        t.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) = Unit
            override fun onDone(utteranceId: String?) { if (utteranceId == id) done.complete(Unit) }
            // 앞 문장을 끊고 말하면 그 문장의 중단·오류 알림이 이 리스너로 온다. 자기 문장의 알림에만 끝난 것으로 본다
            // (그러지 않으면 아직 말하는 중인데 끝난 것으로 보고 듣기를 시작해, 안내 음성을 대답으로 듣게 된다).
            @Deprecated("Deprecated in Java")
            override fun onError(utteranceId: String?) { if (utteranceId == id) done.complete(Unit) }
            override fun onError(utteranceId: String?, errorCode: Int) {
                if (utteranceId != id) return
                Log.w(TAG, "음성 합성 실패(오류 $errorCode, 음성 ${runCatching { t.voice?.name }.getOrNull()})")
                done.complete(Unit)
            }
            override fun onStop(utteranceId: String?, interrupted: Boolean) { if (utteranceId == id) done.complete(Unit) }
        })
        t.speak(text, TextToSpeech.QUEUE_FLUSH, null, id)
        withTimeoutOrNull(SPEAK_TIMEOUT_MS) { done.await() }
    }

    override suspend fun listen(timeoutMs: Long): String? {
        val granted = ContextCompat.checkSelfPermission(appContext, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
        if (!granted) { Log.w(TAG, "마이크 권한이 없어 대답을 들을 수 없습니다"); return null }
        if (!SpeechRecognizer.isRecognitionAvailable(appContext)) { Log.w(TAG, "이 기기에는 음성 인식 서비스가 없습니다"); return null }
        return withContext(Dispatchers.Main) {
            // 인식이 끝나기 전에 시간이 다 되거나 오류로 끝나도, 그때까지 알아들은 말이 있으면 그것을 돌려준다
            var partial: String? = null
            withTimeoutOrNull(timeoutMs) {
                suspendCancellableCoroutine { cont ->
                    val sr = createRecognizer().also { recognizer = it }
                    fun finish(text: String?) {
                        if (cont.isActive) cont.resume(text)
                        runCatching { sr.destroy() }
                        if (recognizer === sr) recognizer = null
                    }
                    sr.setRecognitionListener(object : RecognitionListener {
                        override fun onResults(results: Bundle?) =
                            finish(results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()?.takeIf { it.isNotBlank() } ?: partial)
                        override fun onPartialResults(partialResults: Bundle?) {
                            partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()?.takeIf { it.isNotBlank() }?.let { partial = it }
                        }
                        override fun onError(error: Int) {
                            // 6(말이 없음)·7(알아듣지 못함)은 흔하다. 그 밖의 오류는 원인을 찾을 수 있게 남긴다.
                            Log.i(TAG, "음성 인식 끝: 오류 $error${if (onDevice) "(기기 내 인식)" else ""}")
                            noteError(error)
                            finish(partial)
                        }
                        override fun onReadyForSpeech(params: Bundle?) = Unit
                        override fun onBeginningOfSpeech() = Unit
                        override fun onRmsChanged(rmsdB: Float) = Unit
                        override fun onBufferReceived(buffer: ByteArray?) = Unit
                        override fun onEndOfSpeech() = Unit
                        override fun onEvent(eventType: Int, params: Bundle?) = Unit
                    })
                    val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
                        .putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                        .putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ko-KR")
                        .putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
                        .putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
                    runCatching { sr.startListening(intent) }.onFailure { Log.w(TAG, "음성 인식 시작 실패", it); finish(null) }
                    cont.invokeOnCancellation { runCatching { sr.cancel(); sr.destroy() } }
                }
            } ?: partial
        }
    }

    /** 서버 인식이 네트워크 문제로 실패하면 기기 안에서 인식하는 쪽으로 바꿔 본다(지원하는 기기에서만). */
    private var onDevice = false
    private var onDeviceUnusable = false

    private fun createRecognizer(): SpeechRecognizer {
        if (onDevice && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            runCatching { return SpeechRecognizer.createOnDeviceSpeechRecognizer(appContext) }
                .onFailure { onDevice = false; onDeviceUnusable = true }
        }
        return SpeechRecognizer.createSpeechRecognizer(appContext)
    }

    private fun noteError(error: Int) {
        val network = error == SpeechRecognizer.ERROR_NETWORK || error == SpeechRecognizer.ERROR_NETWORK_TIMEOUT ||
            error == SpeechRecognizer.ERROR_SERVER || error == SpeechRecognizer.ERROR_SERVER_DISCONNECTED
        if (onDevice) {
            // 기기 내 인식이 한국어를 지원하지 않거나 쓸 수 없으면 다시 서버 인식으로 돌아간다
            if (error != SpeechRecognizer.ERROR_NO_MATCH && error != SpeechRecognizer.ERROR_SPEECH_TIMEOUT) {
                onDevice = false
                onDeviceUnusable = true
            }
        } else if (network && !onDeviceUnusable && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S &&
            runCatching { SpeechRecognizer.isOnDeviceRecognitionAvailable(appContext) }.getOrDefault(false)
        ) {
            Log.w(TAG, "서버 음성 인식이 네트워크 문제로 실패해 기기 내 인식으로 바꿉니다")
            onDevice = true
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
        voiceChosen = false
    }

    private companion object {
        const val TAG = "AndroidVoiceGuide"
        const val CONNECT_TIMEOUT_MS = 4_000L
        const val SPEAK_TIMEOUT_MS = 12_000L
    }
}
