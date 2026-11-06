package com.personal.driverguard.platform.android

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Build
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import com.personal.driverguard.R
import com.personal.driverguard.domain.engine.AlertArbiter
import com.personal.driverguard.domain.platform.IAlertOutput
import com.personal.driverguard.entity.AlertKind
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeoutOrNull
import java.util.Locale
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap

/**
 * AlertController: 경고음 + TextToSpeech.
 *
 * - 우선순위: 재생 중에는 더 높은 우선순위 경고만 끊고 들어간다([AlertArbiter]). 반복 제한은 앞단 AlertPolicy 가 한다.
 * - 경고음이 끝난 뒤 음성을 재생해 둘이 겹치지 않는다.
 * - 재생하는 동안만 오디오 포커스(일시·다른 소리 줄이기)를 잡고 끝나면 반환한다.
 * - TTS 초기화 실패나 한국어 음성이 없으면 경고음만 낸다([voiceAvailable] = false).
 * - ViewModel 이 소유하므로 화면 회전으로 다시 만들어지지 않아 같은 경고가 다시 재생되지 않는다.
 * - [cancelAll] 은 재생 중·대기 중 경고를 멈춘다(휴식 시작·주행 종료).
 * 모든 상태는 메인 스레드에서만 바뀐다.
 */
class AndroidAlertOutput(context: Context) : IAlertOutput {

    private val appContext = context.applicationContext
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private val arbiter = AlertArbiter()
    private var playback: Job? = null
    private val pending = ConcurrentHashMap<String, CompletableDeferred<Unit>>()
    private val audioManager = appContext.getSystemService(AudioManager::class.java)
    private val attributes = AudioAttributes.Builder()
        .setUsage(AudioAttributes.USAGE_ASSISTANCE_NAVIGATION_GUIDANCE)
        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
        .build()
    private val focusRequest: AudioFocusRequest? = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
        AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK).setAudioAttributes(attributes).build()
    } else {
        null
    }

    @Volatile
    override var voiceAvailable = false
        private set

    private val tone: ToneGenerator? = runCatching { ToneGenerator(AudioManager.STREAM_ALARM, TONE_VOLUME) }.getOrNull()
    private val tts = TextToSpeech(appContext) { status -> onTtsInit(status) }

    override fun play(kind: AlertKind) {
        scope.launch {
            when (arbiter.offer(kind)) {
                AlertArbiter.Decision.DROP -> return@launch
                AlertArbiter.Decision.INTERRUPT_AND_PLAY -> {
                    playback?.cancel()
                    stopOutputs()
                }
                AlertArbiter.Decision.PLAY -> Unit
            }
            playback = scope.launch {
                requestFocus()
                try {
                    if (kind.withTone && tone != null) {
                        tone.startTone(toneOf(kind), TONE_MS)
                        delay(TONE_MS + TONE_GAP_MS)
                    }
                    speak(appContext.getString(messageOf(kind)))
                } finally {
                    arbiter.finished(kind)
                    if (arbiter.playing == null) abandonFocus()
                }
            }
        }
    }

    override fun cancelAll() {
        scope.launch {
            playback?.cancel()
            stopOutputs()
            arbiter.clear()
            abandonFocus()
        }
    }

    override fun release() {
        playback?.cancel()
        stopOutputs()
        abandonFocus()
        scope.cancel()
        tone?.release()
        tts.shutdown()
    }

    private fun onTtsInit(status: Int) {
        if (status != TextToSpeech.SUCCESS) return
        val language = tts.setLanguage(Locale.KOREAN)
        tts.setAudioAttributes(attributes)
        tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) = Unit
            override fun onDone(utteranceId: String?) = complete(utteranceId)

            @Deprecated("Deprecated in Java")
            override fun onError(utteranceId: String?) = complete(utteranceId)
            override fun onError(utteranceId: String?, errorCode: Int) = complete(utteranceId)
            override fun onStop(utteranceId: String?, interrupted: Boolean) = complete(utteranceId)
        })
        voiceAvailable = language != TextToSpeech.LANG_MISSING_DATA && language != TextToSpeech.LANG_NOT_SUPPORTED
    }

    private fun complete(utteranceId: String?) {
        utteranceId?.let { pending.remove(it)?.complete(Unit) }
    }

    private suspend fun speak(text: String) {
        if (!voiceAvailable) return
        val id = UUID.randomUUID().toString()
        val done = CompletableDeferred<Unit>()
        pending[id] = done
        try {
            if (tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, id) != TextToSpeech.SUCCESS) return
            withTimeoutOrNull(SPEECH_TIMEOUT_MS) { done.await() }
        } finally {
            pending.remove(id)
        }
    }

    @Suppress("DEPRECATION")
    private fun requestFocus() {
        val am = audioManager ?: return
        if (focusRequest != null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            am.requestAudioFocus(focusRequest)
        } else {
            am.requestAudioFocus(null, AudioManager.STREAM_ALARM, AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK)
        }
    }

    @Suppress("DEPRECATION")
    private fun abandonFocus() {
        val am = audioManager ?: return
        if (focusRequest != null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            am.abandonAudioFocusRequest(focusRequest)
        } else {
            am.abandonAudioFocus(null)
        }
    }

    private fun stopOutputs() {
        tone?.stopTone()
        if (voiceAvailable) tts.stop()
    }

    private fun toneOf(kind: AlertKind) = when (kind) {
        AlertKind.REST -> ToneGenerator.TONE_CDMA_ABBR_ALERT
        AlertKind.EYE_CAUTION, AlertKind.HEAD_CAUTION, AlertKind.UNAVAILABLE -> ToneGenerator.TONE_PROP_BEEP2
    }

    private fun messageOf(kind: AlertKind) = when (kind) {
        AlertKind.EYE_CAUTION -> R.string.alert_eye_caution
        AlertKind.HEAD_CAUTION -> R.string.alert_head_caution
        AlertKind.REST -> R.string.alert_rest
        AlertKind.UNAVAILABLE -> R.string.alert_unavailable
    }

    private companion object {
        const val TONE_VOLUME = 90
        const val TONE_MS = 500
        const val TONE_GAP_MS = 150L
        const val SPEECH_TIMEOUT_MS = 10_000L
    }
}
