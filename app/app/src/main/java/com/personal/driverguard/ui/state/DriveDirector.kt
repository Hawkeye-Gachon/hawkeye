package com.personal.driverguard.ui.state

import com.personal.driverguard.data.remote.EventType
import com.personal.driverguard.data.remote.NearestPlace
import com.personal.driverguard.data.remote.TriggerType
import com.personal.driverguard.domain.platform.NaviResult
import com.personal.driverguard.domain.platform.NaviRouteState
import com.personal.driverguard.platform.CabinAlert
import com.personal.driverguard.platform.VoiceGuide
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.async
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.selects.select
import kotlinx.coroutines.withTimeoutOrNull

/** 관찰 중에 보여 줄 화면. [DriveDirector] 만 바꾸고, 화면 쪽은 이 값을 따라간다. */
enum class DriveScene {
    OBSERVE, LOOK, DROWSY, UNRESPONSIVE, COUNTDOWN, REPORTED,
    SUGGEST, SUGGEST_REPEAT, SUGGEST_EXISTING, SUGGEST_NO_DEST, SUGGEST_FULL, SUGGEST_FAIL, SUGGEST_NO_PLACE, SUGGEST_ADDED,
    ARRIVED,
}

/** 쉼터 조회 결과. [navi] 는 차량 내비 상태(연결되지 않았으면 null). */
data class RestLookup(val place: NearestPlace?, val reason: String?, val navi: NaviRouteState?)

/** 진행자가 바깥(음성·차량·서버·기록)에 시키는 일. 앱에서는 [HawkeyeViewModel] 이, 단위 테스트에서는 가짜가 맡는다. */
interface DrivePorts {
    // 음성: 말이 끝날 때까지, 대답을 듣거나 시간이 다 될 때까지 기다린다
    suspend fun voiceReady(): Boolean
    suspend fun speak(text: String)
    /** 신호음 뒤 [windowMs] 동안 듣는다. 못 들었으면 null. */
    suspend fun hear(windowMs: Long): String?
    suspend fun alarm(durationMs: Long)

    // 운전자
    /** 눈 상태를 알 수 있는지(감지기가 돌고 있는지) */
    fun eyesKnown(): Boolean
    /** 얼굴이 보이고 눈을 뜨고 있는지 */
    fun eyesOpen(): Boolean

    // 차량 실내(기다리지 않는다. 반영된 조치는 화면에 따로 표시된다)
    fun cabinAlert(level: CabinAlert)
    fun cabinClearInattention()
    /** 쉼터 도착: 졸음 대응으로 바꾼 것을 되돌리고, 휴식에 맞게 온도와 등받이를 바꾼다 */
    fun cabinRest()
    /** 다시 출발: 휴식에 맞게 바꾼 것을 되돌린다 */
    fun cabinRestore()

    // 기록
    fun record(eventType: String)
    fun drowsinessCount(): Int
    /** 전방 주시 이탈이 쉼터를 제안할 만큼 쌓였는지 */
    fun inattentionSuggestDue(): Boolean

    // 쉼터
    suspend fun findRestStop(): RestLookup
    /** 차량 내비에 쉼터를 넣는다. 차량 내비가 없는 기기면 null. */
    suspend fun addRestStop(place: NearestPlace, destination: Boolean, trigger: String): NaviResult?
    /** 경유지로 넣고 아직 들르지 않은 쉼터 이름. 없으면 null. */
    fun pendingRestName(): String?

    // 119
    fun dial119(): Boolean
    fun locationText(): String?

    fun nowMs(): Long

    /** 흐름 하나가 예상하지 못한 오류로 끝났을 때(기록용). 진행자는 관찰 화면으로 돌아가 계속 돈다. */
    fun fault(error: Throwable) = Unit
}

/**
 * 관찰 중에 일어나는 일을 **한 번에 하나씩, 정해진 순서대로** 진행한다.
 *
 * 감지·119 확인·경유지 질문·퀴즈·도착 안내가 따로따로 움직이면 음성과 화면이 서로 끼어든다.
 * 그래서 이 진행자 하나가 [run] 안에서 차례대로만 진행하고, 진행 중에 들어온 새 감지는 받지 않는다
 * (눈을 계속 감고 있으면 진행 중인 졸음 흐름이 119 확인으로 이어 간다).
 * 졸음 흐름이 끝난 뒤에도 [DROWSY_QUIET_MS] 동안은 새 졸음 감지를 받지 않는다(방금 확인했는데 곧바로 다시 경고하지 않게).
 *
 * 졸음 한 건의 순서:
 *  1. 졸음 경고(화면·음성·차량 조치)
 *  2. 경고 음성이 끝난 뒤 [WATCH_MS] 동안 지켜본다. 눈을 뜨면 4번으로, 계속 감고 있으면 3번으로.
 *  3. 119 확인: 경고음과 안내 → [COUNTDOWN_MS] 동안 대답·눈 뜸·화면 누름을 기다린다.
 *     반응이 없으면 119 전화 화면을 열고 끝낸다. 반응하면 "확인했습니다".
 *  4. 경고를 끄고 [AFTER_OK_MS] 뒤에 쉼터를 경유지로 넣을지 묻는다(이미 넣은 쉼터가 있으면 묻지 않는다).
 *  5. 넣었으면 [QUIZ_FIRST_MS] 뒤부터 쉼터에 도착할 때까지 [QUIZ_GAP_MS] 간격으로 음성 퀴즈를 낸다.
 *  6. 쉼터에 도착하면 도착 안내와 함께 휴식에 맞게 온도와 등받이를 바꾼다.
 *
 * Android 에 의존하지 않는다(시간은 코루틴 delay 와 [DrivePorts.nowMs] 만 쓴다. 단위 테스트는 가상 시간으로 검증한다).
 * 한 스레드(메인)에서만 쓴다.
 */
class DriveDirector(
    private val ports: DrivePorts,
    private val questions: List<DrowsyQuiz.Question> = DrowsyQuiz.questions.shuffled(),
) {
    private sealed interface Trigger {
        data object Drowsy : Trigger
        data object LookAway : Trigger
        data object Arrived : Trigger
    }

    /** 대기 없는 통로: 진행자가 기다리는 중일 때만 받는다. 다른 일을 하는 중에 온 감지는 버려진다. */
    private val inbox = Channel<Trigger>(Channel.RENDEZVOUS)
    private val lookingAway = MutableStateFlow(false)
    private var arrived: ArrivalUi? = null
    /** 이 시각까지는 새 졸음 감지를 받지 않는다(졸음 흐름이 끝난 직후. 방금 괜찮다고 확인했는데 곧바로 다시 경고하지 않게). */
    private var quietUntil = 0L

    private val _scene = MutableStateFlow(DriveScene.OBSERVE)
    val scene: StateFlow<DriveScene> = _scene.asStateFlow()
    private val _suggestion = MutableStateFlow(SuggestionUi())
    val suggestion: StateFlow<SuggestionUi> = _suggestion.asStateFlow()
    private val _voice = MutableStateFlow(VoiceUi())
    val voice: StateFlow<VoiceUi> = _voice.asStateFlow()
    private val _emergency = MutableStateFlow(EmergencyUi())
    val emergency: StateFlow<EmergencyUi> = _emergency.asStateFlow()
    private val _arrival = MutableStateFlow<ArrivalUi?>(null)
    val arrival: StateFlow<ArrivalUi?> = _arrival.asStateFlow()

    // 화면 버튼. 그 버튼을 기다리는 단계에서만 값이 있다.
    private var okTap: CompletableDeferred<Unit>? = null
    private var dismissTap: CompletableDeferred<Unit>? = null
    private var choiceTap: CompletableDeferred<Boolean>? = null
    private var closeTap: CompletableDeferred<Unit>? = null
    private var departTap: CompletableDeferred<Unit>? = null

    // 퀴즈
    private var quizAt: Long? = null
    private var quizIndex = 0
    private var quizIntroduced = false
    /** 방금 쉼터를 넣어 퀴즈 시각을 정했는지(그 흐름이 끝날 때 시각을 다시 미루지 않는다) */
    private var legJustStarted = false

    // ── 바깥에서 알리는 것 ──

    fun onDrowsy() { inbox.trySend(Trigger.Drowsy) }
    fun onLookAway() { lookingAway.value = true; inbox.trySend(Trigger.LookAway) }
    fun onLookBack() { lookingAway.value = false }
    fun onArrived(name: String, type: String?) { arrived = ArrivalUi(name, type); inbox.trySend(Trigger.Arrived) }

    /** 119 확인 화면의 '괜찮습니다' */
    fun tapOk() { okTap?.complete(Unit) }
    /** 119 전화 화면을 연 뒤의 '관찰로 돌아가기' */
    fun tapDismiss() { dismissTap?.complete(Unit) }
    /** 쉼터 제안의 '경유지 추가'·'다시 시도'(true), '나중에'·'닫기'(false) */
    fun tapChoice(yes: Boolean) { choiceTap?.complete(yes) ?: closeTap?.complete(Unit) }
    /** 결과 화면의 '닫기'·'확인' */
    fun tapClose() { closeTap?.complete(Unit) ?: choiceTap?.complete(false) }
    /** 쉼터 도착 화면의 '휴식 마치고 다시 출발' */
    fun tapDepart() { departTap?.complete(Unit) }

    /** 새 관찰을 시작하기 전과 끝낸 뒤에 부른다. 지난 관찰의 상태를 지운다. */
    fun reset() {
        lookingAway.value = false
        arrived = null
        quietUntil = 0
        okTap = null; dismissTap = null; choiceTap = null; closeTap = null; departTap = null
        quizAt = null; quizIndex = 0; quizIntroduced = false; legJustStarted = false
        _scene.value = DriveScene.OBSERVE
        _suggestion.value = SuggestionUi()
        _voice.value = VoiceUi()
        _emergency.value = EmergencyUi()
        _arrival.value = null
    }

    /** 관찰이 끝날 때까지 돈다(취소로 끝낸다). */
    suspend fun run() {
        while (true) {
            try {
                step()
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                // 흐름 하나가 잘못돼도 남은 관찰 동안 감지가 멈추지 않게 한다
                ports.fault(e)
                okTap = null; dismissTap = null; choiceTap = null; closeTap = null; departTap = null
                _voice.value = VoiceUi()
                _scene.value = DriveScene.OBSERVE
                delay(FAULT_PAUSE_MS)
            }
        }
    }

    /** 다음 한 가지(도착 안내, 전방 주시 경고, 졸음 흐름, 퀴즈 한 문제)를 끝까지 진행한다 */
    private suspend fun step() {
        val a = arrived
        if (a != null) {
            arrived = null
            arrivalScript(a)
            return
        }
        // 다른 흐름을 진행하는 동안 시작된 전방 주시 이탈이 아직 이어지고 있으면 지금 알린다
        if (lookingAway.value) {
            lookScript()
            return
        }
        val wait = quizAt?.let { (it - ports.nowMs()).coerceAtLeast(0) }
        val trigger = if (wait == null) inbox.receive() else withTimeoutOrNull(wait) { inbox.receive() }
        if (trigger == null) quizStep() else dispatch(trigger)
    }

    private suspend fun dispatch(trigger: Trigger) {
        if (!accepts(trigger)) return
        when (trigger) {
            Trigger.Drowsy -> drowsyScript()
            Trigger.LookAway -> lookScript()
            Trigger.Arrived -> Unit // 도착은 run 의 맨 위에서 처리한다
        }
    }

    /** 지금 받아도 되는 감지인지. 졸음 흐름이 끝난 직후의 쉬는 시간에는 새 졸음 감지를 받지 않는다. */
    private fun accepts(trigger: Trigger): Boolean = trigger != Trigger.Drowsy || ports.nowMs() >= quietUntil

    // ── 전방 주시 이탈 ──

    private suspend fun lookScript() {
        ports.record(EventType.INATTENTION)
        _scene.value = DriveScene.LOOK
        ports.cabinAlert(CabinAlert.INATTENTION)
        var takeover = false
        coroutineScope {
            var talk = launch { ports.speak(LOOK_TEXT) }
            while (lookingAway.value && arrived == null) {
                when (val t = withTimeoutOrNull(POLL_MS) { inbox.receive() }) {
                    // 고개를 돌린 채 눈까지 감으면 졸음으로 넘어간다
                    Trigger.Drowsy -> if (accepts(t)) { takeover = true; break }
                    // 계속 돌리고 있어 다시 감지되면 한 번 더 말한다
                    Trigger.LookAway -> if (!talk.isActive) talk = launch { ports.speak(LOOK_TEXT) }
                    else -> Unit
                }
            }
            if (takeover) talk.cancel()
        }
        ports.cabinClearInattention()
        if (takeover) {
            drowsyScript()
            return
        }
        _scene.value = DriveScene.OBSERVE
        // 전방 주시 이탈이 쌓이면 쉼터를 제안한다(이미 넣은 쉼터가 있으면 하지 않는다)
        if (arrived == null && ports.inattentionSuggestDue() && ports.pendingRestName() == null) {
            delay(AFTER_OK_MS)
            coroutineScope { suggestScript(TriggerType.INATTENTION_REPEATED, async { find(TriggerType.INATTENTION_REPEATED) }) }
        }
        resumeQuiz()
    }

    // ── 졸음 ──

    private suspend fun drowsyScript() {
        ports.record(EventType.DROWSINESS)
        val repeated = ports.drowsinessCount() >= 2
        _scene.value = DriveScene.DROWSY
        ports.cabinAlert(if (repeated) CabinAlert.DROWSY_REPEATED else CabinAlert.DROWSY)
        coroutineScope {
            // 쉼터는 미리 찾아 둔다(물을 때 기다리지 않게). 이미 넣은 쉼터가 있으면 찾지 않는다.
            val lookup = if (ports.pendingRestName() == null) async { find(TriggerType.DROWSINESS) } else null
            ports.speak(if (repeated) DROWSY_REPEATED_TEXT else DROWSY_TEXT)
            // 경고가 끝난 뒤에도 계속 눈을 감고 있는지 지켜본다. 눈 상태를 모르면(감지기 꺼짐) 119 확인은 하지 않는다.
            val awake = !ports.eyesKnown() || watchEyes(WATCH_MS)
            if (!awake) {
                if (!emergencyCheck()) {
                    lookup?.cancel()
                    reportScript()
                    return@coroutineScope
                }
                ports.speak("확인했습니다.")
            }
            // 경고를 끄고, 잠시 뒤에 쉼터를 묻는다
            _scene.value = DriveScene.OBSERVE
            delay(AFTER_OK_MS)
            val pending = ports.pendingRestName()
            if (lookup == null || pending != null) {
                lookup?.cancel()
                ports.speak("${pending ?: "쉼터"}까지 조금만 더 힘내세요.")
            } else {
                suggestScript(TriggerType.DROWSINESS, lookup)
            }
        }
        quietUntil = ports.nowMs() + DROWSY_QUIET_MS
        resumeQuiz()
    }

    /** [ms] 동안 지켜본다. 눈을 [OPEN_CONFIRM_MS] 이상 계속 뜨면 바로 true, 끝까지 그러지 못하면 false. */
    private suspend fun watchEyes(ms: Long): Boolean {
        var waited = 0L
        var open = 0L
        while (waited < ms) {
            delay(POLL_MS)
            waited += POLL_MS
            open = if (ports.eyesOpen()) open + POLL_MS else 0
            if (open >= OPEN_CONFIRM_MS) return true
        }
        return false
    }

    /** 119 확인. 운전자가 반응하면 true, 끝까지 반응이 없으면 false. */
    private suspend fun emergencyCheck(): Boolean {
        _emergency.value = EmergencyUi()
        _scene.value = DriveScene.UNRESPONSIVE
        ports.cabinAlert(CabinAlert.DROWSY_REPEATED)
        // 경고음과 안내. 그사이에도 눈을 뜨거나 화면을 누르면 반응으로 본다.
        val early = awaitResponse(windowMs = null, listen = false) {
            ports.alarm(WAKE_ALARM_MS)
            ports.speak("괜찮으세요? 대답이 없으면 10초 뒤 119 전화 화면을 엽니다.")
        }
        if (early) return true
        _scene.value = DriveScene.COUNTDOWN
        return awaitResponse(windowMs = COUNTDOWN_MS, listen = true)
    }

    /**
     * 운전자의 반응(눈을 [OPEN_CONFIRM_MS] 이상 뜸, 화면의 '괜찮습니다', [listen] 이면 음성 대답)을 기다린다. 반응하면 true.
     * [announce] 가 있으면 그 안내가 끝날 때까지, 없으면 [windowMs] 동안 기다린다(남은 초는 화면에 보여 준다).
     */
    private suspend fun awaitResponse(windowMs: Long?, listen: Boolean, announce: (suspend () -> Unit)? = null): Boolean = coroutineScope {
        val tap = CompletableDeferred<Unit>().also { okTap = it }
        val talk = announce?.let { launch { it() } }
        val heard = if (listen && windowMs != null) async { !ports.hear(windowMs).isNullOrBlank() } else null
        var waited = 0L
        var open = 0L
        var result: Boolean? = null
        try {
            while (result == null) {
                result = when {
                    tap.isCompleted || open >= OPEN_CONFIRM_MS -> true
                    heard != null && heard.isCompleted && heard.await() -> true
                    talk != null && !talk.isActive -> false
                    windowMs != null && waited >= windowMs -> false
                    else -> null
                }
                if (result != null) break
                if (windowMs != null) _emergency.update { it.copy(secondsLeft = ((windowMs - waited + 999) / 1000).toInt()) }
                delay(POLL_MS)
                waited += POLL_MS
                open = if (ports.eyesOpen()) open + POLL_MS else 0
            }
        } finally {
            talk?.cancel()
            heard?.cancel()
            okTap = null
        }
        result == true
    }

    /** 반응이 없어 119 전화 화면을 연다. 화면에서 닫을 때까지 기다린다. */
    private suspend fun reportScript() {
        val dialed = ports.dial119()
        _emergency.value = EmergencyUi(dialed = dialed, location = ports.locationText())
        _scene.value = DriveScene.REPORTED
        val dismiss = CompletableDeferred<Unit>().also { dismissTap = it }
        try {
            coroutineScope {
                val talk = launch {
                    ports.speak(if (dialed) "119 전화 화면을 열었습니다. 통화 버튼을 눌러 주세요." else "전화 화면을 열지 못했습니다. 주변에 도움을 요청해 주세요.")
                }
                dismiss.await()
                talk.cancel()
            }
        } finally {
            dismissTap = null
        }
        _emergency.value = EmergencyUi()
        _scene.value = DriveScene.OBSERVE
    }

    // ── 쉼터 제안 ──

    private suspend fun find(trigger: String): RestLookup {
        _suggestion.value = SuggestionUi(loading = true, trigger = trigger)
        val found = ports.findRestStop()
        _suggestion.value = SuggestionUi(trigger = trigger, place = found.place, noPlaceReason = found.reason, navi = found.navi)
        return found
    }

    private suspend fun suggestScript(trigger: String, lookup: Deferred<RestLookup>) {
        _voice.value = VoiceUi()
        _scene.value = if (trigger == TriggerType.INATTENTION_REPEATED) DriveScene.SUGGEST_REPEAT else DriveScene.SUGGEST
        val found = lookup.await()
        val place = found.place
        if (place == null) {
            _scene.value = DriveScene.SUGGEST_NO_PLACE
            resultScene("가까운 쉼터를 찾지 못했습니다. 안전한 곳에 정차해 주세요.")
            return
        }
        val navi = found.navi
        var destination = false
        when {
            navi == null -> Unit
            !navi.routing -> { destination = true; _scene.value = DriveScene.SUGGEST_NO_DEST }
            navi.waypointsFull -> {
                _scene.value = DriveScene.SUGGEST_FULL
                resultScene("경유지를 더 추가할 수 없습니다. 정차한 뒤 내비에서 경유지를 정리해 주세요.")
                return
            }
            trigger == TriggerType.DROWSINESS && navi.waypointNames.isNotEmpty() -> _scene.value = DriveScene.SUGGEST_EXISTING
        }
        if (!confirm(place.name, destination)) return
        // 차량 내비에 넣는다. 실패하면 다시 시도할 수 있고, 목적지가 없으면 쉼터로 길 안내를 다시 묻는다.
        while (true) {
            _suggestion.update { it.copy(busy = true, failure = null) }
            val result = ports.addRestStop(place, destination, trigger)
            _suggestion.update { it.copy(busy = false, failure = (result as? NaviResult.Failed)?.reason) }
            _voice.value = VoiceUi()
            when (result) {
                // null: 차량 내비가 없는 기기. 화면 흐름은 추가된 것처럼 보여 준다.
                null, NaviResult.Added -> {
                    added(place.name, destination)
                    return
                }
                NaviResult.NoDestination -> {
                    destination = true
                    _scene.value = DriveScene.SUGGEST_NO_DEST
                    if (!confirm(place.name, true)) return
                }
                NaviResult.Full -> {
                    _scene.value = DriveScene.SUGGEST_FULL
                    resultScene("경유지를 더 추가할 수 없습니다. 정차한 뒤 내비에서 경유지를 정리해 주세요.")
                    return
                }
                is NaviResult.Failed -> {
                    _scene.value = DriveScene.SUGGEST_FAIL
                    val retry = choose(timeoutMs = RESULT_MS) { ports.speak("경유지를 추가하지 못했습니다. 다시 시도하려면 화면을 눌러 주세요."); null }
                    if (retry != true) {
                        _scene.value = DriveScene.OBSERVE
                        return
                    }
                }
            }
        }
    }

    /** 쉼터를 넣을지 묻는다. '응'(또는 버튼)이면 true. 아니면 관찰 화면으로 돌아가고 false. */
    private suspend fun confirm(name: String, destination: Boolean): Boolean {
        val question = if (destination) "${name}${HawkeyeViewModel.josaRo(name)} 길 안내를 시작할까요? 응 또는 아니 라고 답해 주세요."
        else "${name}${HawkeyeViewModel.josaReul(name)} 경유지로 추가할까요? 응 또는 아니 라고 답해 주세요."
        val voiced = ports.voiceReady()
        val yes = choose(timeoutMs = if (voiced) TAP_AFTER_VOICE_MS else TAP_ONLY_MS) {
            if (!voiced) return@choose null
            _voice.value = VoiceUi(asking = true)
            ports.speak(question)
            _voice.value = VoiceUi(listening = true)
            var heard = ports.hear(ANSWER_MS)
            var answer = VoiceGuide.yesNo(heard)
            // 무슨 말은 했는데 알아듣지 못했으면 한 번 더 묻는다
            if (heard != null && answer == null) {
                _voice.value = VoiceUi(asking = true)
                ports.speak("잘 못 들었어요. 응 또는 아니 라고 답해 주세요.")
                _voice.value = VoiceUi(listening = true)
                heard = ports.hear(ANSWER_MS) ?: heard
                answer = VoiceGuide.yesNo(heard)
            }
            _voice.value = VoiceUi(heard = heard, answer = answer, done = true)
            answer
        }
        if (yes == true) return true
        _voice.value = VoiceUi()
        if (yes == false) ports.speak("알겠습니다. 다음에 다시 안내할게요.")
        _scene.value = DriveScene.OBSERVE
        return false
    }

    /**
     * 음성([dialog], 대답을 못 얻으면 null)과 화면 버튼 중 먼저 온 답을 받는다.
     * 음성으로 답을 못 얻으면 [timeoutMs] 동안 버튼을 더 기다린다. 끝내 없으면 null.
     */
    private suspend fun choose(timeoutMs: Long, dialog: suspend () -> Boolean?): Boolean? = coroutineScope {
        val tap = CompletableDeferred<Boolean>().also { choiceTap = it }
        val spoken = async { dialog() }
        try {
            select<Boolean?> {
                tap.onAwait { it }
                spoken.onAwait { answer -> answer ?: withTimeoutOrNull(timeoutMs) { tap.await() } }
            }
        } finally {
            spoken.cancel()
            choiceTap = null
        }
    }

    /** 결과 화면(쉼터 없음·가득 참): 안내를 말하고, 닫기를 누르거나 [RESULT_MS] 가 지나면 관찰 화면으로 돌아간다. */
    private suspend fun resultScene(text: String) {
        val close = CompletableDeferred<Unit>().also { closeTap = it }
        try {
            coroutineScope {
                val talk = launch { ports.speak(text) }
                withTimeoutOrNull(RESULT_MS) { close.await() }
                talk.cancel()
            }
        } finally {
            closeTap = null
        }
        _scene.value = DriveScene.OBSERVE
    }

    private suspend fun added(name: String, destination: Boolean) {
        if (destination) {
            _scene.value = DriveScene.OBSERVE
            ports.speak("${name}${HawkeyeViewModel.josaRo(name)} 안내를 시작합니다.")
        } else {
            _scene.value = DriveScene.SUGGEST_ADDED
            ports.speak("${name}${HawkeyeViewModel.josaReul(name)} 경유지로 추가했습니다.")
        }
        val at = ports.nowMs()
        if (!destination) {
            val close = CompletableDeferred<Unit>().also { closeTap = it }
            try { withTimeoutOrNull(ADDED_SHOW_MS) { close.await() } } finally { closeTap = null }
            _scene.value = DriveScene.OBSERVE
        }
        // 쉼터로 가는 동안 퀴즈를 낸다(추가 안내가 끝나고 QUIZ_FIRST_MS 뒤부터)
        quizAt = if (ports.pendingRestName() != null) at + QUIZ_FIRST_MS else null
        legJustStarted = quizAt != null
    }

    // ── 퀴즈 ──

    /** 다른 흐름이 끝난 뒤: 아직 갈 쉼터가 있고 이번 구간의 퀴즈가 남았으면 잠시 뒤에 이어서 낸다 */
    private fun resumeQuiz() {
        if (legJustStarted) legJustStarted = false
        else if (quizAt != null) quizAt = ports.nowMs() + QUIZ_FIRST_MS
    }

    private suspend fun quizStep() {
        if (ports.pendingRestName() == null || !ports.voiceReady()) { quizAt = null; return }
        // 눈을 감았거나 정면을 보지 않을 때는 내지 않고 잠시 뒤에 다시 본다
        if (lookingAway.value || (ports.eyesKnown() && !ports.eyesOpen())) {
            quizAt = ports.nowMs() + QUIZ_RETRY_MS
            return
        }
        val q = questions[quizIndex % questions.size]
        val interrupted = interruptible {
            ports.speak((if (quizIntroduced) DrowsyQuiz.NEXT else DrowsyQuiz.INTRO) + " " + q.text)
            quizIntroduced = true
            val heard = ports.hear(ANSWER_MS)
            var said = DrowsyQuiz.parse(heard)
            if (heard != null && said == null) {
                ports.speak("오 또는 엑스로 답해 주세요.")
                said = DrowsyQuiz.parse(ports.hear(ANSWER_MS))
            }
            ports.speak(DrowsyQuiz.feedback(q, said))
        }
        if (interrupted != null) {
            // 경고가 먼저다. 같은 문제는 그 흐름이 끝난 뒤에 다시 낸다.
            quizAt = ports.nowMs() + QUIZ_FIRST_MS
            dispatch(interrupted)
            return
        }
        quizIndex++
        quizAt = ports.nowMs() + QUIZ_GAP_MS
    }

    /** [block] 을 진행하다가 새 감지나 도착이 오면 멈추고 그것을 돌려준다. 끝까지 했으면 null. */
    private suspend fun interruptible(block: suspend () -> Unit): Trigger? = coroutineScope {
        val work = async { block() }
        var trigger: Trigger? = null
        while (true) {
            val t = select<Trigger?> {
                work.onAwait { null }
                inbox.onReceive { it }
            } ?: break
            // 받지 않는 감지(쉬는 시간의 졸음)로는 멈추지 않는다
            if (accepts(t)) {
                trigger = t
                work.cancelAndJoin()
                break
            }
        }
        trigger
    }

    // ── 쉼터 도착 ──

    private suspend fun arrivalScript(a: ArrivalUi) {
        quizAt = null
        _arrival.value = a
        _scene.value = DriveScene.ARRIVED
        ports.cabinRest()
        val depart = CompletableDeferred<Unit>().also { departTap = it }
        try {
            coroutineScope {
                val talk = launch { ports.speak("${a.name}에 도착했습니다. 휴식에 알맞은 온도로 맞추고 의자를 눕힐게요. 충분히 쉬고 출발하세요.") }
                depart.await()
                talk.cancel()
            }
        } finally {
            departTap = null
        }
        ports.cabinRestore()
        _arrival.value = null
        _scene.value = DriveScene.OBSERVE
        // 아직 들를 쉼터가 남아 있으면 그쪽으로 가는 동안 다시 퀴즈를 낸다
        if (ports.pendingRestName() != null) quizAt = ports.nowMs() + QUIZ_FIRST_MS
    }

    companion object {
        const val LOOK_TEXT = "전방을 주시해 주세요."
        const val DROWSY_TEXT = "졸음이 감지됐습니다. 시원한 바깥 공기를 들이고 있어요."
        const val DROWSY_REPEATED_TEXT = "졸음이 반복되고 있습니다. 창문을 조금 열었어요."

        /** 상태를 확인하는 간격 */
        const val POLL_MS = 100L
        /** 졸음 경고 음성이 끝난 뒤 눈을 뜨는지 지켜보는 시간. 끝까지 감고 있으면 119 확인으로 넘어간다. */
        const val WATCH_MS = 5_000L
        /** 눈을 이만큼 계속 뜨고 있어야 뜬 것으로 본다(잠깐 떴다 감는 것은 치지 않는다) */
        const val OPEN_CONFIRM_MS = 1_000L
        /** 119 확인에서 경고음을 울리는 시간 */
        const val WAKE_ALARM_MS = 2_000L
        /** 119 전화 화면을 열기 전 대답을 기다리는 시간 */
        const val COUNTDOWN_MS = 10_000L
        /** 괜찮다고 확인되고 경고가 꺼진 뒤 쉼터를 묻기까지 */
        const val AFTER_OK_MS = 3_000L
        /** 질문 뒤 대답을 기다리는 시간 */
        const val ANSWER_MS = 8_000L
        /** 음성으로 답을 못 얻었을 때 버튼을 더 기다리는 시간 / 음성을 쓸 수 없을 때 버튼을 기다리는 시간 */
        const val TAP_AFTER_VOICE_MS = 10_000L
        const val TAP_ONLY_MS = 25_000L
        /** 결과 화면(쉼터 없음·가득 참·실패)을 보여 주는 시간 */
        const val RESULT_MS = 8_000L
        /** '경유지로 추가했습니다' 화면을 보여 주는 시간 */
        const val ADDED_SHOW_MS = 3_000L
        /** 경유지를 넣고 나서(또는 다른 흐름이 끝난 뒤) 퀴즈를 내기까지 */
        const val QUIZ_FIRST_MS = 6_000L
        /** 문제 사이 쉬는 시간 */
        const val QUIZ_GAP_MS = 15_000L
        /** 눈을 감았거나 정면을 보지 않아 문제를 못 낼 때 다시 보는 간격 */
        const val QUIZ_RETRY_MS = 1_000L
        /** 흐름이 오류로 끝났을 때 다음 흐름까지 쉬는 시간 */
        const val FAULT_PAUSE_MS = 1_000L
        /** 졸음 흐름이 끝난 뒤 새 졸음 감지를 받지 않는 시간 */
        const val DROWSY_QUIET_MS = 20_000L
    }
}
