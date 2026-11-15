package com.personal.driverguard.ui.state

import android.app.Application
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.personal.driverguard.PlatformLocator
import com.personal.driverguard.RepositoryLocator
import com.personal.driverguard.data.remote.AgreeItem
import com.personal.driverguard.data.remote.ApiResult
import com.personal.driverguard.data.remote.CalendarResponse
import com.personal.driverguard.data.remote.DailyReport
import com.personal.driverguard.data.remote.DriveListItem
import com.personal.driverguard.data.remote.DriveReport
import com.personal.driverguard.data.remote.EventType
import com.personal.driverguard.data.remote.HawkeyeApi
import com.personal.driverguard.data.remote.MeResponse
import com.personal.driverguard.data.remote.NavMode
import com.personal.driverguard.data.remote.NearbyRequest
import com.personal.driverguard.data.remote.NearestPlace
import com.personal.driverguard.data.remote.PlaceType
import com.personal.driverguard.data.remote.SessionEvent
import com.personal.driverguard.data.remote.TriggerType
import com.personal.driverguard.data.remote.WeeklyReport
import com.personal.driverguard.data.remote.getOrNull
import com.personal.driverguard.domain.engine.DistanceAccumulator
import com.personal.driverguard.domain.platform.NaviResult
import com.personal.driverguard.domain.platform.NaviRouteState
import com.personal.driverguard.domain.platform.RestStop
import com.personal.driverguard.platform.CabinAction
import com.personal.driverguard.platform.CabinAlert
import com.personal.driverguard.platform.CabinStatus
import com.personal.driverguard.platform.Detection
import com.personal.driverguard.platform.DetectorUi
import com.personal.driverguard.platform.DrowsyDetector
import com.personal.driverguard.ui.design.DriverFrames
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import com.personal.driverguard.platform.ListenCue
import com.personal.driverguard.platform.VoiceGuide
import com.personal.driverguard.platform.WakeAlarm
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.emitAll
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.withTimeoutOrNull
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.time.LocalDate
import java.time.YearMonth

// ── 화면 상태 ──

/** 입력 폼(로그인·회원가입) 공통 상태 */
data class FormUi(
    val busy: Boolean = false,
    /** 폼 전체 오류(예: 로그인 실패, 네트워크 오류) */
    val message: String? = null,
    /** 입력란별 오류(백엔드 fieldErrors + 앱 검사) */
    val fieldErrors: Map<String, String> = emptyMap(),
)

data class TermsItemUi(val termsId: Long, val title: String, val required: Boolean, val content: String?)

data class TermsUi(
    val loading: Boolean = false,
    val items: List<TermsItemUi> = emptyList(),
    val checks: Map<Long, Boolean> = emptyMap(),
    val busy: Boolean = false,
    val error: String? = null,
) {
    val requiredOk: Boolean get() = items.isNotEmpty() && items.filter { it.required }.all { checks[it.termsId] == true }
    val allOn: Boolean get() = items.isNotEmpty() && items.all { checks[it.termsId] == true }
}

data class HomeUi(
    val loading: Boolean = false,
    val me: MeResponse? = null,
    val lastDrive: DriveListItem? = null,
    val weekly: WeeklyReport? = null,
    val error: String? = null,
)

/** 관찰 중 화면에 보이는 값 */
data class ObserveUi(
    val active: Boolean = false,
    val startedMonotonicMs: Long = 0,
    val inattentionCount: Int = 0,
    val drowsinessCount: Int = 0,
    val locationEnabled: Boolean = false,
)

/** 쉼터 제안 상태 */
data class SuggestionUi(
    val loading: Boolean = false,
    /** 백엔드 TriggerType */
    val trigger: String = TriggerType.DROWSINESS,
    val place: NearestPlace? = null,
    /** 쉼터를 못 찾았을 때 이유(화면 문구) */
    val noPlaceReason: String? = null,
    /** 차량 내비 상태. Pleos 내비에 연결되지 않았으면 null */
    val navi: NaviRouteState? = null,
    val busy: Boolean = false,
    val failure: String? = null,
)

data class EndUi(val busy: Boolean = false, val error: String? = null)

data class ReportUi(val loading: Boolean = false, val report: DriveReport? = null, val error: String? = null)

data class DailyUi(val loading: Boolean = false, val data: DailyReport? = null, val error: String? = null)

data class WeeklyUi(val loading: Boolean = false, val data: WeeklyReport? = null, val error: String? = null)

data class CalendarUi(val month: YearMonth = YearMonth.now(SEOUL), val data: CalendarResponse? = null, val loading: Boolean = false)

/** 차량이 한 조치와 차량에서 읽은 상태(관찰 화면 표시용) */
data class CabinUi(
    val level: CabinAlert? = null,
    val actions: List<CabinAction> = emptyList(),
    val status: CabinStatus = CabinStatus(),
)

/** 음성으로 묻고 답을 듣는 상태(쉼터 제안 화면) */
/** 쉼터 제안의 음성 대화 상태. [done] 은 묻고 듣기가 끝났는지(대답을 못 들었어도 true). */
data class VoiceUi(val asking: Boolean = false, val listening: Boolean = false, val heard: String? = null, val answer: Boolean? = null, val done: Boolean = false)

/**
 * 반응 없음 대응 상태(긴급 화면).
 * [dialed] 는 119 전화 화면을 열었는지(아직 시도 전이면 null), [location] 은 신고할 때 알려 줄 현재 위치.
 */
data class EmergencyUi(
    val secondsLeft: Int = 0,
    val dialed: Boolean? = null,
    val location: String? = null,
)

/** 들른 쉼터(쉼터 도착 화면) */
data class ArrivalUi(val name: String, val type: String?)

/** 다음에 열 화면(내비게이션은 화면 쪽이 한다) */
enum class Next { LOGIN, TERMS, HOME }

/**
 * 앱 전체 상태와 백엔드·차량 내비 호출을 맡는다. 화면은 상태를 읽고, 동작은 여기 함수를 부른 뒤 결과에 따라 이동한다.
 *
 * 연결 범위:
 *  - 졸음·전방 주시 이탈은 카메라 영상을 기기 안에서 판정한다([DrowsyDetector]).
 *  - 감지된 뒤의 흐름(경고 → 119 확인 → 쉼터 제안 → 퀴즈 → 도착 안내)은 [DriveDirector] 가 한 번에 하나씩 순서대로 진행한다.
 *  - 차량 개입은 Pleos Vehicle SDK(무드등·공조·창문·시트), 음성은 Pleos TTS·STT(없으면 Android 기본 음성), 쉼터 경유지는 NaviHelper.
 *  - 위치는 Pleos FusedLocation, 없으면 일반 Android 위치.
 */
class HawkeyeViewModel(app: Application) : AndroidViewModel(app) {

    private val context: Context get() = getApplication()
    val api: HawkeyeApi = RepositoryLocator.provideApi(app)
    private val navigation by lazy { PlatformLocator.provideVehicleNavigation(app) }
    private val cabin by lazy { PlatformLocator.provideVehicleCabin(app) }
    private val voice: VoiceGuide by lazy { PlatformLocator.provideVoiceGuide(app) }
    private val location = LocationTracker(app)

    private val pending = PendingUploads(app.getSharedPreferences("pending_uploads", Context.MODE_PRIVATE))

    val loggedIn: StateFlow<Boolean> = api.loggedIn
    val sessionEvents: SharedFlow<SessionEvent> = api.sessionEvents

    private val _login = MutableStateFlow(FormUi())
    val login = _login.asStateFlow()
    private val _signup = MutableStateFlow(FormUi())
    val signup = _signup.asStateFlow()
    private val _terms = MutableStateFlow(TermsUi())
    val terms = _terms.asStateFlow()
    private val _home = MutableStateFlow(HomeUi())
    val home = _home.asStateFlow()
    private val _observe = MutableStateFlow(ObserveUi())
    val observe = _observe.asStateFlow()
    private val _end = MutableStateFlow(EndUi())
    val end = _end.asStateFlow()
    private val _report = MutableStateFlow(ReportUi())
    val report = _report.asStateFlow()
    private val _daily = MutableStateFlow(DailyUi())
    val daily = _daily.asStateFlow()
    private val _weekly = MutableStateFlow(WeeklyUi())
    val weekly = _weekly.asStateFlow()
    private val _calendar = MutableStateFlow(CalendarUi())
    val calendar = _calendar.asStateFlow()
    private val _cabinUi = MutableStateFlow(CabinUi())
    val cabinUi = _cabinUi.asStateFlow()
    private val _detectorUi = MutableStateFlow(DetectorUi())
    val detectorUi = _detectorUi.asStateFlow()
    private var detector: DrowsyDetector? = null
    private var detectorJob: Job? = null
    private var speedFrom: GeoFix? = null

    private val wakeAlarm = WakeAlarm()

    /** 이번 주행에서 경유지로 넣고 아직 들르지 않은 쉼터 이름. 없으면 null. */
    @Volatile private var pendingRestName: String? = null

    private var recorder: DriveRecorder? = null
    private var locationJob: Job? = null
    private var cabinStatusJob: Job? = null
    private var cabinJob: Job? = null
    private var directorJob: Job? = null
    private var checkpointJob: Job? = null
    /** 중간 저장을 바로 해 달라는 요청. 저장은 [checkpointJob] 안에서만 해서, 종료할 때 그 작업만 끝내면 뒤늦은 저장이 없다. */
    private val checkpointRequests = Channel<Unit>(Channel.CONFLATED)
    private var dailyJob: Job? = null
    private var weeklyJob: Job? = null
    private var calendarJob: Job? = null

    /** 경유지로 넣은 쉼터 이름(도착 안내용) */
    private val addedPlaces = mutableMapOf<Long, NearestPlace>()

    /**
     * 주행 리포트의 지도에 쉼터를 찍기 위한 좌표(위도, 경도). 리포트 응답에는 쉼터 좌표가 없어서,
     * 방금 끝난 주행에서 넣은 쉼터만 알 수 있다. 모르면 null.
     */
    fun restPlaceLocation(placeId: Long): Pair<Double, Double>? = addedPlaces[placeId]?.let { it.latitude to it.longitude }

    // ════════════════════════ 시작 · 인증 ════════════════════════

    /** 스플래시: 저장된 로그인으로 이어 가고, 올리지 못한 주행 기록이 있으면 다시 올린다. */
    suspend fun bootstrap(): Next {
        if (!api.loggedIn.value) return Next.LOGIN
        return when (val r = api.resumeSession()) {
            is ApiResult.Success -> {
                viewModelScope.launch { retryPendingUploads() }
                if (r.data.termsAgreementRequired) Next.TERMS else Next.HOME
            }
            // 네트워크 문제면 저장된 로그인으로 홈에 들어간다(기록 화면은 각자 오류를 보여 준다)
            is ApiResult.Failure -> if (r.code == ApiResult.NETWORK) Next.HOME else Next.LOGIN
        }
    }

    suspend fun login(email: String, password: String): Next? {
        val errors = buildMap {
            if (email.isBlank()) put("email", "이메일을 입력해 주세요.")
            if (password.isBlank()) put("password", "비밀번호를 입력해 주세요.")
        }
        if (errors.isNotEmpty()) {
            _login.value = FormUi(fieldErrors = errors)
            return null
        }
        _login.value = FormUi(busy = true)
        return when (val r = api.login(email.trim(), password)) {
            is ApiResult.Success -> {
                pending.owner = email
                _login.value = FormUi()
                if (r.data.termsAgreementRequired) Next.TERMS else Next.HOME
            }
            is ApiResult.Failure -> {
                _login.value = FormUi(message = r.message, fieldErrors = r.fieldErrors)
                null
            }
        }
    }

    suspend fun signup(name: String, email: String, password: String, passwordConfirm: String, ageConfirmed: Boolean): Next? {
        val errors = buildMap {
            if (name.isBlank()) put("name", "이름을 입력해 주세요.")
            if (email.isBlank()) put("email", "이메일을 입력해 주세요.")
            if (password.isBlank()) put("password", "비밀번호를 입력해 주세요.")
            if (password != passwordConfirm) put("passwordConfirm", "비밀번호가 일치하지 않습니다.")
            if (!ageConfirmed) put("ageConfirmed", "만 14세 이상만 가입할 수 있습니다.")
        }
        if (errors.isNotEmpty()) {
            _signup.value = FormUi(fieldErrors = errors)
            return null
        }
        _signup.value = FormUi(busy = true)
        return when (val r = api.signup(email.trim(), name.trim(), password, ageConfirmed)) {
            is ApiResult.Success -> {
                pending.owner = email
                _signup.value = FormUi()
                if (r.data.termsAgreementRequired) Next.TERMS else Next.HOME
            }
            is ApiResult.Failure -> {
                val fields = r.fieldErrors.toMutableMap()
                if (r.code == "EMAIL_DUPLICATED") fields["email"] = r.message
                _signup.value = FormUi(message = r.message.takeIf { fields.isEmpty() }, fieldErrors = fields)
                null
            }
        }
    }

    fun clearForms() {
        _login.value = FormUi()
        _signup.value = FormUi()
    }

    suspend fun logout() {
        stopDriveLocally()
        api.logout()
        pending.owner = null
        _home.value = HomeUi()
    }

    // ════════════════════════ 약관 ════════════════════════

    /** 약관 목록·전문과(로그인했으면) 내 동의 현황을 불러온다. 전문을 읽고 돌아와도 선택은 유지된다. */
    fun loadTerms() {
        if (_terms.value.items.isNotEmpty() || _terms.value.loading) return
        _terms.update { it.copy(loading = true, error = null) }
        viewModelScope.launch {
            val list = api.terms()
            val summaries = list.getOrNull()?.terms
            if (summaries == null) {
                _terms.update { it.copy(loading = false, error = (list as ApiResult.Failure).message) }
                return@launch
            }
            val details = summaries.map { s -> async { api.termsDetail(s.termsId).getOrNull()?.content } }.map { it.await() }
            val mine = if (api.loggedIn.value) api.myAgreements().getOrNull()?.agreements.orEmpty() else emptyList()
            val agreed = mine.associate { it.termsId to (it.agreed == true) }
            _terms.update {
                it.copy(
                    loading = false,
                    items = summaries.mapIndexed { i, s -> TermsItemUi(s.termsId, s.title, s.required, details[i]) },
                    checks = summaries.associate { s -> s.termsId to (it.checks[s.termsId] ?: agreed[s.termsId] ?: false) },
                )
            }
        }
    }

    fun toggleTerm(termsId: Long) = _terms.update { it.copy(checks = it.checks + (termsId to !(it.checks[termsId] ?: false))) }

    fun toggleAllTerms() = _terms.update { t -> val v = !t.allOn; t.copy(checks = t.items.associate { it.termsId to v }) }

    suspend fun agreeTerms(): Boolean {
        val t = _terms.value
        if (!t.requiredOk) return false
        _terms.update { it.copy(busy = true, error = null) }
        val r = api.agree(t.items.map { AgreeItem(it.termsId, t.checks[it.termsId] == true) })
        _terms.update { it.copy(busy = false, error = (r as? ApiResult.Failure)?.message) }
        return r is ApiResult.Success && !r.data.termsAgreementRequired
    }

    // ════════════════════════ 홈 ════════════════════════

    fun loadHome() {
        _home.update { it.copy(loading = true, error = null) }
        viewModelScope.launch {
            val me = async { api.me() }
            val last = async { api.drives(size = 1) }
            val week = async { api.weekly(null) }
            val meR = me.await()
            _home.value = HomeUi(
                loading = false,
                me = meR.getOrNull(),
                lastDrive = last.await().getOrNull()?.drives?.firstOrNull(),
                weekly = week.await().getOrNull(),
                error = (meR as? ApiResult.Failure)?.message,
            )
        }
    }

    // ════════════════════════ 관찰 ════════════════════════

    /** '관찰 시작'(보는 방향 설정 완료 후). 주행을 등록하고 위치 기록을 시작하고, 차량 상태를 읽기 시작한다. */
    fun startDrive() {
        if (recorder != null) return
        val rec = DriveRecorder(startedMonotonicMs = SystemClock.elapsedRealtime())
        recorder = rec
        addedPlaces.clear()
        pendingRestName = null
        director.reset()
        _cabinUi.value = CabinUi()
        _observe.value = ObserveUi(active = true, startedMonotonicMs = rec.startedMonotonicMs, locationEnabled = location.hasPermission())
        _report.value = ReportUi()
        _end.value = EndUi()
        // 위치: Pleos 차량 위치(없으면 Android 위치)
        val fixes = vehicleLocation()
        locationJob = viewModelScope.launch {
            fixes.collect {
                rec.onLocation(it)
                detector?.speedKmh = speedKmh(it)
                announceArrivals(rec)
            }
        }
        // 앱이 강제 종료돼도 그때까지의 기록이 남도록 주기적으로 기기에 저장한다(다음 실행 때 올린다).
        checkpointJob = viewModelScope.launch {
            while (true) {
                // 주기가 되거나, 감지가 기록돼 바로 저장해 달라는 요청이 오면 저장한다
                withTimeoutOrNull(CHECKPOINT_MS) { checkpointRequests.receive() }
                checkpoint(rec)
            }
        }
        // 감지·119 확인·쉼터 제안·퀴즈·도착 안내는 진행자가 한 번에 하나씩 순서대로 진행한다
        directorJob = viewModelScope.launch { director.run() }
        // 카메라 영상으로 졸음·전방 주시 이탈을 감지한다
        startDetector()
        // 등록에 실패해도 관찰은 계속한다. 종료할 때 다시 등록한다.
        viewModelScope.launch { api.startDrive(rec.clientDriveId, rec.startedAt).getOrNull()?.let { rec.driveId = it.driveId } }
        // 차량 상태(안전벨트·크루즈·기어)를 읽고, 안전벨트를 안 맸으면 음성으로 알린다
        cabinStatusJob = viewModelScope.launch {
            val c = cabin
            if (!c.connect()) return@launch
            if (c.status.value.seatBeltUnbuckled == true) launch { say("안전벨트를 매 주세요.") }
            c.status.collect { st -> _cabinUi.update { it.copy(status = st) } }
        }
    }

    /** 진행 중인 주행을 지금까지의 내용으로 기기에 저장한다. 기록은 메인 스레드에서 읽고, 쓰기는 백그라운드에서 한다. */
    private suspend fun checkpoint(rec: DriveRecorder) {
        val item = PendingUpload(rec.clientDriveId, rec.startedAt, rec.driveId, rec.toEndRequest(), pending.owner)
        withContext(Dispatchers.IO) { pending.put(item) }
    }

    /** 감지기를 준비하고 카메라 프레임을 잇는다. 실패하면 감지 없이 관찰만 계속한다(화면에 사유를 보여 준다). */
    private fun startDetector() {
        speedFrom = null
        val d = DrowsyDetector(context) { detection -> viewModelScope.launch { onDetection(detection) } }
        detector = d
        detectorJob = viewModelScope.launch {
            launch { d.ui.collect { _detectorUi.value = it } }
            val ok = withContext(Dispatchers.Default) { d.start() }
            if (ok && detector === d) DriverFrames.sink = d::analyze
        }
    }

    /** 감지 결과는 진행자에게만 알린다. 진행자가 다른 흐름을 진행하는 중이면 새 감지는 받지 않는다. */
    private fun onDetection(detection: Detection) {
        if (recorder == null) return
        when (detection) {
            Detection.DROWSY -> director.onDrowsy()
            Detection.LOOK_AWAY -> director.onLookAway()
            Detection.LOOK_BACK -> director.onLookBack()
        }
    }

    /** 위치가 속도를 주면 그 값을, 아니면 앞 위치와의 거리로 계산한 값을 km/h 로 */
    private fun speedKmh(fix: GeoFix): Double? {
        val prev = speedFrom
        speedFrom = fix
        fix.speedMps?.let { return it * 3.6 }
        prev ?: return null
        val sec = (fix.monotonicMs - prev.monotonicMs) / 1000.0
        if (sec <= 0) return null
        return DistanceAccumulator.haversineMeters(prev.latitude, prev.longitude, fix.latitude, fix.longitude) / sec * 3.6
    }

    /** Pleos 차량 위치를 쓰고, 지원하지 않으면 일반 Android 위치 */
    private fun vehicleLocation(): Flow<GeoFix> = flow {
        val fused = PlatformLocator.provideFusedLocation(context)
        if (fused.isAvailable()) emitAll(fused.updates()) else emitAll(location.updates())
    }

    /** 쉼터에 멈춰 방문으로 기록되면 진행자에게 알린다(도착 안내는 진행 중인 흐름이 끝난 뒤에 나온다) */
    private fun announceArrivals(rec: DriveRecorder) {
        val ids = rec.takeArrivals()
        if (ids.isEmpty()) return
        val place = addedPlaces[ids.last()]
        val name = place?.name ?: "쉼터"
        if (rec.pendingStop() == null) pendingRestName = null
        director.onArrived(name, place?.type)
    }

    // ════════════════════════ 진행자 ════════════════════════

    /** 진행자가 시키는 일(음성·차량·서버·기록)을 실제로 한다 */
    private val ports = object : DrivePorts {
        override suspend fun voiceReady(): Boolean = runCatching { voice.connect() }.getOrDefault(false)
        override suspend fun speak(text: String) = say(text)
        override suspend fun hear(windowMs: Long): String? = voiceTurn(VoiceQueue.DIALOG) { listenAnswer(windowMs) }
        override suspend fun alarm(durationMs: Long) { voiceTurn(VoiceQueue.DIALOG) { wakeAlarm.ring(durationMs) } }

        override fun eyesKnown(): Boolean = _detectorUi.value.running
        override fun eyesOpen(): Boolean {
            val d = _detectorUi.value
            return d.running && d.faceVisible && (d.closure?.let { it < EYES_OPEN_BELOW } ?: false)
        }

        override fun cabinAlert(level: CabinAlert) = cabinDo {
            // 더 낮은 단계로는 내려가지 않는다(졸음 대응 중에 전방 주시 이탈이 와도 빨간 무드등을 주황으로 바꾸지 않는다)
            val current = _cabinUi.value.level
            if (current != null && level.ordinal < current.ordinal) return@cabinDo
            val done = cabin.alert(level)
            val voiced = runCatching { voice.connect() }.getOrDefault(false)
            _cabinUi.update { it.copy(level = level, actions = (if (voiced) listOf(CabinAction(CabinAction.Kind.VOICE, "음성 안내")) else emptyList()) + done) }
        }

        override fun cabinClearInattention() = cabinDo {
            // 졸음 단계면 쉼터에 들를 때까지 유지한다
            if (_cabinUi.value.level != CabinAlert.INATTENTION) return@cabinDo
            cabin.restore()
            _cabinUi.update { it.copy(level = null, actions = emptyList()) }
        }

        override fun cabinRest() = cabinDo {
            cabin.restore()
            val done = cabin.rest()
            _cabinUi.update { it.copy(level = null, actions = done) }
        }

        override fun cabinRestore() = cabinDo {
            cabin.restore()
            _cabinUi.update { it.copy(level = null, actions = emptyList()) }
        }

        override fun record(eventType: String) { signal(eventType) }
        override fun drowsinessCount(): Int = recorder?.drowsinessCount ?: 0
        override fun inattentionSuggestDue(): Boolean = inattentionDue

        override suspend fun findRestStop(): RestLookup = lookupRestStop()
        override suspend fun addRestStop(place: NearestPlace, destination: Boolean, trigger: String): NaviResult? = addToNavigation(place, destination, trigger)
        override fun pendingRestName(): String? = if (recorder?.pendingStop() != null) pendingRestName ?: "쉼터" else null

        override fun dial119(): Boolean = openDialer()
        override fun locationText(): String? = recorder?.lastFix?.let { "위도 %.5f, 경도 %.5f".format(it.latitude, it.longitude) }

        override fun nowMs(): Long = SystemClock.elapsedRealtime()
        override fun fault(error: Throwable) { Log.e(VOICE_TAG, "관찰 중 흐름이 오류로 끝났습니다", error) }
    }

    private val director = DriveDirector(ports)

    /** 관찰 중에 보여 줄 화면. 화면 쪽은 이 값을 따라 바뀐다. */
    val scene: StateFlow<DriveScene> = director.scene
    val suggestion: StateFlow<SuggestionUi> = director.suggestion
    val voiceUi: StateFlow<VoiceUi> = director.voice
    val emergency: StateFlow<EmergencyUi> = director.emergency
    val arrival: StateFlow<ArrivalUi?> = director.arrival

    /** 차량 조작은 요청한 순서대로 하나씩 한다(진행자는 끝나기를 기다리지 않는다) */
    private fun cabinDo(block: suspend () -> Unit) {
        val previous = cabinJob
        cabinJob = viewModelScope.launch {
            previous?.join()
            runCatching { block() }
        }
    }

    // ── 화면 버튼 ──

    /** 119 확인 화면의 '괜찮습니다' */
    fun respondToEmergency() = director.tapOk()

    /** 119 전화 화면을 연 뒤의 '관찰로 돌아가기' */
    fun dismissEmergency() = director.tapDismiss()

    /** 쉼터 제안의 '경유지 추가'·'쉼터로 길 안내'·'다시 시도'(true), '나중에'·'닫기'(false) */
    fun answerSuggestion(yes: Boolean) = director.tapChoice(yes)

    /** 쉼터 제안 결과 화면의 '닫기'·'확인' */
    fun closeSuggestion() = director.tapClose()

    /** 쉼터 도착 화면의 '휴식 마치고 다시 출발'. 휴식에 맞게 바꿨던 실내를 되돌린다. */
    fun resumeAfterRest() = director.tapDepart()

    /**
     * 119 가 입력된 전화 화면을 연다. 일반 앱은 긴급 번호로 바로 전화를 걸 수 없어(Android 제한) 통화 버튼은 사람이 눌러야 한다.
     * 전화 앱이 없는 기기(일부 태블릿·차량)에서는 열지 못하고 false.
     */
    private fun openDialer(): Boolean = runCatching {
        context.startActivity(Intent(Intent.ACTION_DIAL, Uri.parse("tel:$EMERGENCY_NUMBER")).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
        true
    }.getOrDefault(false)

    // ════════════════════════ 음성 ════════════════════════
    //
    // 말하기와 듣기는 한 번에 하나만 한다(스피커와 마이크가 하나라서, 겹치면 안내가 끊기거나 안내 음성을 대답으로 듣는다).
    // 진행자가 순서대로 부르므로 보통은 겹칠 일이 없고, 그 밖의 음성(안전벨트 안내)과도 [VoiceQueue] 로 차례를 지킨다.

    private val voiceQueue = VoiceQueue { runCatching { voice.stop() } }

    /**
     * 음성 차례를 받아 [block] 을 실행한다([VoiceQueue]). 더 급한 음성에 밀려 끊기면 null.
     * [block] 안에서는 [speakNow]·[listenAnswer] 를 쓴다([say] 는 차례를 다시 받으려 해서 쓰면 안 된다).
     */
    private suspend fun <T> voiceTurn(priority: Int, skipIfBusy: Boolean = false, block: suspend () -> T): T? =
        voiceQueue.turn(priority, skipIfBusy, block)

    /** 음성 안내 한 문장(차례를 받아서). 쓸 수 없으면 조용히 넘어간다. */
    private suspend fun say(text: String, priority: Int = VoiceQueue.DIALOG, skipIfBusy: Boolean = false) {
        voiceTurn(priority, skipIfBusy) { speakNow(text) }
    }

    /** 차례 안에서 말한다. 말이 끝날 때까지 기다린다. */
    private suspend fun speakNow(text: String) {
        // 음성 순서를 기기 로그로 확인할 수 있게 남긴다(운전자 음성·영상은 남기지 않는다)
        Log.i(VOICE_TAG, "말하기 시작: $text")
        runCatching { if (voice.connect()) voice.speak(text) }
        Log.i(VOICE_TAG, "말하기 끝")
    }

    /**
     * 차례 안에서 대답 하나를 듣는다. 신호음을 낸 뒤 [windowMs] 동안 듣고, 인식기가 일찍 끝나도(잠깐 조용했거나 알아듣지 못함)
     * 시간이 남아 있으면 다시 듣는다. 들은 말이 없으면 null.
     */
    private suspend fun listenAnswer(windowMs: Long): String? {
        ListenCue.beep()
        Log.i(VOICE_TAG, "듣기 시작(${windowMs}ms)")
        val heard = listenLoop(windowMs)
        Log.i(VOICE_TAG, "듣기 끝: ${if (heard == null) "못 들음" else "들음(${heard.length}자)"}")
        return heard
    }

    private suspend fun listenLoop(windowMs: Long): String? {
        val deadline = SystemClock.elapsedRealtime() + windowMs
        var quickEnds = 0
        while (true) {
            val left = deadline - SystemClock.elapsedRealtime()
            if (left < MIN_LISTEN_MS) return null
            val started = SystemClock.elapsedRealtime()
            val heard = runCatching { voice.listen(left) }.getOrNull()
            if (!heard.isNullOrBlank()) return heard
            // 인식기를 쓸 수 없는 상태(권한·네트워크 등)면 바로 끝나므로, 계속 바로 끝나면 그만 듣는다
            if (SystemClock.elapsedRealtime() - started < QUICK_END_MS) {
                if (++quickEnds >= 3) return null
                delay(QUICK_END_MS)
            } else {
                quickEnds = 0
            }
        }
    }

    /**
     * 감지 1건을 주행 기록에 남긴다.
     * 전방 주시 이탈이 3회 누적될 때마다 true 를 돌려준다(정면 주시가 확인되면 쉼터를 제안).
     */
    fun signal(type: String): Boolean {
        val rec = recorder ?: return false
        rec.recordEvent(type)
        // 감지 기록은 바로 저장한다(다음 주기 저장 전에 앱이 꺼져도 남도록)
        checkpointRequests.trySend(Unit)
        _observe.update { it.copy(inattentionCount = rec.inattentionCount, drowsinessCount = rec.drowsinessCount) }
        return type == EventType.INATTENTION && rec.inattentionCount > 0 && rec.inattentionCount % INATTENTION_SUGGEST_EVERY == 0
    }

    /** 이번 주행의 전방 주시 이탈이 3회 단위로 쌓였는지(정면 주시 확인 시 제안 여부) */
    val inattentionDue: Boolean
        get() = (recorder?.inattentionCount ?: 0).let { it > 0 && it % INATTENTION_SUGGEST_EVERY == 0 }

    // ════════════════════════ 쉼터 ════════════════════════

    /** 현재 위치로 가장 가까운 쉼터를 묻고, 차량 내비 상태를 확인한다 */
    private suspend fun lookupRestStop(): RestLookup = coroutineScope {
        val navi = async { runCatching { navigation.routeState() }.getOrNull() }
        val fix = recorder?.lastFix
            ?: return@coroutineScope RestLookup(null, "현재 위치를 확인할 수 없어 쉼터를 찾지 못했습니다.", navi.await())
        val r = api.nearbyRestPlace(NearbyRequest(fix.latitude, fix.longitude, fix.bearingDeg?.toDouble(), fix.speedMps?.toDouble()))
        val place = r.getOrNull()?.place
        val reason = when {
            r is ApiResult.Failure -> r.message
            place != null -> null
            r.getOrNull()?.onHighway == false -> "고속도로가 아닌 곳에서는 쉼터를 안내하지 않습니다. 안전한 곳에 정차해 주세요."
            else -> "진행 방향에서 바로 들어갈 수 있는 쉼터를 확인하지 못했습니다. 안전한 곳에 정차해 주세요."
        }
        RestLookup(place, reason, navi.await())
    }

    /**
     * 차량 내비에 쉼터를 넣는다([destination] 이면 쉼터로 길 안내). Pleos 내비에 연결되지 않으면 null.
     * 내비가 반영을 확인하면 주행 기록에 경유지를 남긴다.
     */
    private suspend fun addToNavigation(place: NearestPlace, destination: Boolean, trigger: String): NaviResult? {
        fun record() {
            addedPlaces[place.placeId] = place
            pendingRestName = place.name
            recorder?.recordWaypoint(
                place.placeId, if (destination) NavMode.DESTINATION else NavMode.WAYPOINT, trigger,
                latitude = place.latitude, longitude = place.longitude,
            )
        }
        val nav = navigation
        if (!nav.connect()) return null
        val stop = RestStop(place.name, placeTypeLabel(place.type), place.latitude, place.longitude)
        val result = if (destination) nav.routeToRestStop(stop) else nav.addRestStop(stop)
        if (result == NaviResult.Added) record()
        return result
    }

    // ════════════════════════ 종료 · 리포트 ════════════════════════

    /** '종료하고 리포트 보기'. 기록을 올리고 리포트를 받는다. 실패하면 기기에 남겨 두고 다음 실행 때 다시 올린다. */
    suspend fun endDrive(): Boolean {
        val rec = recorder ?: return false
        if (_end.value.busy) return false
        _end.value = EndUi(busy = true)
        // 올리는 도중 화면을 벗어나도(호출한 쪽이 취소돼도) 차량 원복·업로드·기기 저장은 끝까지 마친다
        return withContext(NonCancellable) {
            // 중간 저장이 종료 뒤에 끼어들어 이미 올린 기록을 다시 남기지 않도록, 끝날 때까지 기다린다
            checkpointJob?.cancelAndJoin()
            checkpointJob = null
            // 진행 중이던 흐름(경고·질문·퀴즈)을 멈춘다
            directorJob?.cancelAndJoin()
            directorJob = null
            runCatching { voice.stop() }
            cabinJob?.cancelAndJoin()
            runCatching { cabin.restore() }
            val body = rec.toEndRequest()
            // 올리기 전에 먼저 기기에 남긴다(업로드 중 앱이 꺼져도 다음 실행 때 올라간다)
            pending.put(PendingUpload(rec.clientDriveId, rec.startedAt, rec.driveId, body, pending.owner))
            val driveId = rec.driveId ?: api.startDrive(rec.clientDriveId, rec.startedAt).getOrNull()?.driveId?.also { rec.driveId = it }
            val result = driveId?.let { api.endDrive(it, body) }
            val report = result?.getOrNull()
            if (report == null) {
                pending.put(PendingUpload(rec.clientDriveId, rec.startedAt, driveId, body, pending.owner))
                stopDriveLocally()
                _end.value = EndUi(error = (result as? ApiResult.Failure)?.message ?: "서버에 연결할 수 없습니다.")
                return@withContext false
            }
            pending.remove(rec.clientDriveId)
            stopDriveLocally()
            _end.value = EndUi()
            _report.value = ReportUi(report = report)
            true
        }
    }

    /** 주행 목록·일간 리포트에서 연 리포트 */
    fun openReport(driveId: Long) {
        _report.value = ReportUi(loading = true)
        viewModelScope.launch {
            val r = api.driveReport(driveId)
            _report.value = ReportUi(report = r.getOrNull(), error = (r as? ApiResult.Failure)?.message)
        }
    }

    private fun stopDriveLocally() {
        DriverFrames.sink = null
        detectorJob?.cancel()
        detectorJob = null
        detector?.let { d -> viewModelScope.launch(Dispatchers.Default) { d.close() } }
        detector = null
        _detectorUi.value = DetectorUi()
        directorJob?.cancel()
        directorJob = null
        cabinJob?.cancel()
        cabinJob = null
        wakeAlarm.stop()
        pendingRestName = null
        checkpointJob?.cancel()
        checkpointJob = null
        cabinStatusJob?.cancel()
        cabinStatusJob = null
        locationJob?.cancel()
        locationJob = null
        recorder = null
        _observe.value = ObserveUi()
        _cabinUi.value = CabinUi()
        director.reset()
    }

    // ════════════════════════ 이전 기록 ════════════════════════

    /** [date] 가 null 이면 가장 최근 주행일 */
    fun loadDaily(date: LocalDate?) {
        dailyJob?.cancel()
        _daily.update { it.copy(loading = true, error = null) }
        dailyJob = viewModelScope.launch {
            val r = api.daily(date?.toString())
            _daily.value = DailyUi(data = r.getOrNull() ?: _daily.value.data, error = (r as? ApiResult.Failure)?.message)
        }
    }

    /** [weekStart] 는 월요일. null 이면 가장 최근 주행일이 속한 주 */
    fun loadWeekly(weekStart: LocalDate?) {
        weeklyJob?.cancel()
        _weekly.update { it.copy(loading = true, error = null) }
        weeklyJob = viewModelScope.launch {
            val r = api.weekly(weekStart?.toString())
            _weekly.value = WeeklyUi(data = r.getOrNull() ?: _weekly.value.data, error = (r as? ApiResult.Failure)?.message)
        }
    }

    fun loadCalendar(month: YearMonth) {
        calendarJob?.cancel()
        _calendar.value = CalendarUi(month = month, loading = true)
        calendarJob = viewModelScope.launch {
            _calendar.value = CalendarUi(month = month, data = api.calendar(month.toString()).getOrNull())
        }
    }

    // ════════════════════════ 올리지 못한 기록 ════════════════════════

    /**
     * 지금 계정의 올리지 못한 기록을 다시 올린다. 하나씩 올리고, 끝난 것만 보관함에서 지운다
     * (올리는 사이 새로 저장된 기록을 덮어쓰지 않는다). 진행 중인 주행의 중간 저장본은 건드리지 않는다.
     */
    private suspend fun retryPendingUploads() {
        for (p in pending.uploadable(except = recorder?.clientDriveId)) {
            if (p.clientDriveId == recorder?.clientDriveId) continue
            val id = p.driveId ?: api.startDrive(p.clientDriveId, p.startedAt).getOrNull()?.driveId
            val r = id?.let { api.endDrive(it, p.body) }
            // 성공했거나, 주행이 없어졌거나(404) 형식 오류(400)면 다시 시도해도 소용없으니 지운다
            val done = r is ApiResult.Success || (r is ApiResult.Failure && r.status in 400..404 && r.status != 401)
            if (done) withContext(Dispatchers.IO) { pending.remove(p.clientDriveId) }
        }
    }

    override fun onCleared() {
        DriverFrames.sink = null
        detector?.close()
        locationJob?.cancel()
        cabinStatusJob?.cancel()
        wakeAlarm.stop()
        // 관찰 중에 화면이 닫혀도 지금까지의 기록을 남기고, 바꿔 둔 차량 실내는 원래대로 돌린다
        // (viewModelScope 는 이미 취소됐으므로 앱 범위에서 마친다)
        val rec = recorder
        val item = rec?.let { PendingUpload(it.clientDriveId, it.startedAt, it.driveId, it.toEndRequest(), pending.owner) }
        val vehicle = cabin
        PlatformLocator.appScope.launch {
            item?.let { pending.put(it) }
            runCatching { vehicle.restore() }
        }
    }

    companion object {
        /** 전방 주시 이탈이 이 횟수만큼 쌓일 때마다 쉼터를 제안한다 */
        const val INATTENTION_SUGGEST_EVERY = 3
        /** 이 값보다 눈 감김 정도(0 뜸 ~ 1 감음)가 낮으면 눈을 뜬 것으로 본다 */
        const val EYES_OPEN_BELOW = 0.5
        const val EMERGENCY_NUMBER = "119"
        private const val VOICE_TAG = "HawkeyeVoice"
        /** 남은 시간이 이보다 짧으면 더 듣지 않는다 */
        const val MIN_LISTEN_MS = 1_200L
        /** 인식기가 이보다 빨리 끝나면 쓸 수 없는 상태로 본다 */
        const val QUICK_END_MS = 600L
        /** 진행 중인 주행을 기기에 저장하는 주기 */
        const val CHECKPOINT_MS = 30_000L

        /** 받침이 있으면 '을', 없으면 '를' */
        fun josaReul(word: String) = if (hasBatchim(word)) "을" else "를"

        /** 받침이 있으면 '으로'(ㄹ 받침은 '로'), 없으면 '로' */
        fun josaRo(word: String): String {
            val c = word.lastOrNull() ?: return "로"
            if (c !in '가'..'힣') return "로"
            val jong = (c - '가') % 28
            return if (jong == 0 || jong == 8) "로" else "으로"
        }

        private fun hasBatchim(word: String): Boolean {
            val c = word.lastOrNull() ?: return false
            return c in '가'..'힣' && (c - '가') % 28 != 0
        }

        fun placeTypeLabel(type: String?) = when (type) {
            PlaceType.REST_AREA -> "휴게소"
            PlaceType.DROWSY_SHELTER -> "졸음쉼터"
            else -> "쉼터"
        }
    }
}
