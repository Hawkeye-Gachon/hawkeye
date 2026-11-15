package com.personal.driverguard.ui.state

import android.app.Application
import android.content.Context
import android.os.SystemClock
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
import com.personal.driverguard.domain.platform.NaviResult
import com.personal.driverguard.domain.platform.NaviRouteState
import com.personal.driverguard.domain.platform.RestStop
import com.personal.driverguard.platform.CabinAction
import com.personal.driverguard.platform.CabinAlert
import com.personal.driverguard.platform.CabinStatus
import com.personal.driverguard.platform.DemoVehicleCabin
import com.personal.driverguard.platform.Detection
import com.personal.driverguard.platform.DetectorUi
import com.personal.driverguard.platform.DrowsyDetector
import com.personal.driverguard.ui.design.DriverFrames
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import com.personal.driverguard.platform.DemoVehicleNavigation
import com.personal.driverguard.platform.VoiceGuide
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.emitAll
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.Job
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.serialization.builtins.ListSerializer
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
data class VoiceUi(val asking: Boolean = false, val listening: Boolean = false, val heard: String? = null, val answer: Boolean? = null)

/** 들른 쉼터(쉼터 도착 화면) */
data class ArrivalUi(val name: String, val type: String?)

/** 다음에 열 화면(내비게이션은 화면 쪽이 한다) */
enum class Next { LOGIN, TERMS, HOME }

/**
 * 앱 전체 상태와 백엔드·차량 내비 호출을 맡는다. 화면은 상태를 읽고, 동작은 여기 함수를 부른 뒤 결과에 따라 이동한다.
 *
 * 연결 범위(임시 포함):
 *  - 졸음·전방 주시 이탈 '감지'는 AI 추론 서버가 아직 없어 화면의 시연 신호가 [detectDrowsiness]·[detectInattention] 을 부른다.
 *    AI 연동 시 같은 함수를 쓴다.
 *  - 감지되면 단계별로 차량이 개입한다: 음성 안내(Pleos TTS), 무드등·공조·창문·시트(Pleos Vehicle SDK). 쉼터 제안은 음성으로 묻고
 *    '응'이라고 답하면 경유지로 넣는다(Pleos STT, NaviHelper). 쉼터에 멈추면 도착으로 기록하고 실내를 원래대로 돌린다.
 *  - 위치는 Pleos FusedLocation, 없으면 일반 Android 위치.
 *  - [DemoMode] 가 켜지면 서버·차량·위치가 모두 시연용 가짜로 바뀐다(DemoServer, DemoVehicleNavigation, DemoVehicleCabin, DriveSimulator).
 */
class HawkeyeViewModel(app: Application) : AndroidViewModel(app) {

    private val context: Context get() = getApplication()
    val api: HawkeyeApi = RepositoryLocator.provideApi(app)
    private val realNavigation by lazy { PlatformLocator.provideVehicleNavigation(app) }
    private val realCabin by lazy { PlatformLocator.provideVehicleCabin(app) }
    private val demoNavigation = DemoVehicleNavigation()
    private val demoCabin = DemoVehicleCabin()
    private val voice: VoiceGuide by lazy { PlatformLocator.provideVoiceGuide(app) }
    private val location = LocationTracker(app)

    /** 시연 모드면 가짜 차량 내비·실내, 아니면 Pleos */
    private val navigation get() = if (DemoMode.enabled) demoNavigation else realNavigation
    private val cabin get() = if (DemoMode.enabled) demoCabin else realCabin
    val demoMode: StateFlow<Boolean> = DemoMode.on
    private val pendingStore = app.getSharedPreferences("pending_uploads", Context.MODE_PRIVATE)

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
    private val _suggestion = MutableStateFlow(SuggestionUi())
    val suggestion = _suggestion.asStateFlow()
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
    private val _voiceUi = MutableStateFlow(VoiceUi())
    val voiceUi = _voiceUi.asStateFlow()
    private val _arrival = MutableStateFlow<ArrivalUi?>(null)
    val arrival = _arrival.asStateFlow()

    /** 쉼터에 도착함 — 화면 쪽이 받아 쉼터 도착 화면으로 이동한다 */
    private val _arrivals = MutableSharedFlow<Unit>(extraBufferCapacity = 2)
    val arrivals: SharedFlow<Unit> = _arrivals

    /** 카메라 영상 감지 결과 — 화면 쪽이 받아 경고 화면으로 이동한다 */
    private val _detections = MutableSharedFlow<Detection>(extraBufferCapacity = 8)
    val detections: SharedFlow<Detection> = _detections
    /** 졸음 경고가 난 횟수(같은 화면에서 다시 경고가 나면 화면이 표시 시간을 다시 센다) */
    private val _drowsyAlerts = MutableStateFlow(0)
    val drowsyAlerts = _drowsyAlerts.asStateFlow()
    private val _detectorUi = MutableStateFlow(DetectorUi())
    val detectorUi = _detectorUi.asStateFlow()
    private var detector: DrowsyDetector? = null
    private var detectorJob: Job? = null
    private var lastDrowsyMs = 0L
    private var speedFrom: GeoFix? = null

    private var recorder: DriveRecorder? = null
    private var simulator: DriveSimulator? = null
    private var locationJob: Job? = null
    private var cabinStatusJob: Job? = null
    private var interveneJob: Job? = null

    /** 경유지로 넣은 쉼터 이름(도착 안내용) */
    private val addedPlaces = mutableMapOf<Long, NearestPlace>()

    init {
        DemoMode.init(app)
    }

    // ════════════════════════ 시연 모드 ════════════════════════

    /** 시연 모드를 켜고 끈다. 진행 중인 관찰을 멈추고 로그아웃한 뒤 바꾼다(화면은 로그인으로). */
    suspend fun setDemoMode(on: Boolean) {
        if (DemoMode.enabled == on) return
        stopDriveLocally()
        api.logout()
        DemoMode.set(context, on)
        _home.value = HomeUi()
        _terms.value = TermsUi()
        _daily.value = DailyUi()
        _weekly.value = WeeklyUi()
        _calendar.value = CalendarUi()
        _report.value = ReportUi()
        clearForms()
    }


    // ════════════════════════ 시작 · 인증 ════════════════════════

    /** 스플래시: 저장된 로그인으로 이어 가고, 올리지 못한 주행 기록이 있으면 다시 올린다. */
    suspend fun bootstrap(): Next {
        if (!api.loggedIn.value) return Next.LOGIN
        return when (val r = api.resumeSession()) {
            is ApiResult.Success -> {
                // 시연 모드에서는 실제 서버로 올릴 기록을 건드리지 않는다
                if (!DemoMode.enabled) viewModelScope.launch { retryPendingUploads() }
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
        val demo = DemoMode.enabled
        val rec = DriveRecorder(startedMonotonicMs = SystemClock.elapsedRealtime())
        recorder = rec
        addedPlaces.clear()
        _arrival.value = null
        _cabinUi.value = CabinUi()
        if (demo) demoNavigation.clear()
        _observe.value = ObserveUi(
            active = true, startedMonotonicMs = rec.startedMonotonicMs,
            locationEnabled = demo || DebugFlags.DUMMY_ROUTE || location.hasPermission(),
        )
        _report.value = ReportUi()
        _end.value = EndUi()
        // 위치: 시연은 경부고속도로 시속 80km 모의 주행(경유지를 넣으면 6초 뒤 도착), debug 서버 시험은 시속 100km,
        // 실제는 Pleos 차량 위치(없으면 Android 위치)
        val sim = when {
            demo -> DriveSimulator(DemoRoute.load(context), arriveWithinMs = DriveSimulator.DEMO_ARRIVE_MS)
            DebugFlags.DUMMY_ROUTE -> DriveSimulator(DemoRoute.load(context), DriveSimulator.TEST_SPEED_MPS, restMs = DriveSimulator.TEST_REST_MS)
            else -> null
        }
        simulator = sim
        val fixes = sim?.fixes { rec.pendingStop()?.let { LatLng(it.first, it.second) } } ?: vehicleLocation()
        locationJob = viewModelScope.launch {
            fixes.collect {
                rec.onLocation(it)
                detector?.speedKmh = speedKmh(it)
                announceArrivals(rec)
            }
        }
        // 카메라 영상으로 졸음·전방 주시 이탈을 감지한다. 시연 모드도 감지는 실제 AI 를 쓴다(위치·차량·서버만 가짜).
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

    /** 감지기를 준비하고 카메라 프레임을 잇는다. 실패하면 감지 없이 관찰만 계속한다(화면에 사유를 보여 준다). */
    private fun startDetector() {
        lastDrowsyMs = 0L
        speedFrom = null
        val d = DrowsyDetector(context) { detection -> viewModelScope.launch { onDetection(detection) } }
        detector = d
        detectorJob = viewModelScope.launch {
            launch { d.ui.collect { _detectorUi.value = it } }
            val ok = withContext(Dispatchers.Default) { d.start() }
            if (ok && detector === d) DriverFrames.sink = d::analyze
        }
    }

    private fun onDetection(detection: Detection) {
        if (recorder == null) return
        when (detection) {
            Detection.DROWSY -> {
                // 경고는 감지될 때마다 다시 낸다(재경고 간격은 판정 라이브러리의 rules.json 이 정한다).
                // 다만 한 번의 졸음이 여러 건으로 세어지지 않게, 직전 기록 뒤 짧은 시간 안의 감지는 새로 기록하지 않는다.
                val now = SystemClock.elapsedRealtime()
                val sameEpisode = lastDrowsyMs != 0L && now - lastDrowsyMs < DROWSY_REPEAT_MS
                lastDrowsyMs = now
                detectDrowsiness(record = !sameEpisode)
            }
            Detection.LOOK_AWAY -> detectInattention()
            Detection.LOOK_BACK -> inattentionCleared()
        }
        _detections.tryEmit(detection)
    }

    /** 위치가 속도를 주면 그 값을, 아니면 앞 위치와의 거리로 계산한 값을 km/h 로 */
    private fun speedKmh(fix: GeoFix): Double? {
        val prev = speedFrom
        speedFrom = fix
        fix.speedMps?.let { return it * 3.6 }
        prev ?: return null
        val sec = (fix.monotonicMs - prev.monotonicMs) / 1000.0
        if (sec <= 0) return null
        return DemoRoute.dist(LatLng(prev.latitude, prev.longitude), LatLng(fix.latitude, fix.longitude)) / sec * 3.6
    }

    /** Pleos 차량 위치를 쓰고, 지원하지 않으면 일반 Android 위치 */
    private fun vehicleLocation(): Flow<GeoFix> = flow {
        val fused = PlatformLocator.provideFusedLocation(context)
        if (fused.isAvailable()) emitAll(fused.updates()) else emitAll(location.updates())
    }

    /** 쉼터에 멈춰 방문으로 기록되면 도착 화면을 띄우고, 실내를 원래대로 돌리고, 음성으로 알린다 */
    private fun announceArrivals(rec: DriveRecorder) {
        val ids = rec.takeArrivals()
        if (ids.isEmpty()) return
        val place = addedPlaces[ids.last()]
        val name = place?.name ?: "쉼터"
        _arrival.value = ArrivalUi(name, place?.type)
        _arrivals.tryEmit(Unit)
        if (DemoMode.enabled) demoNavigation.clear(name)
        interveneJob?.cancel()
        viewModelScope.launch {
            cabin.restore()
            _cabinUi.update { it.copy(level = null, actions = emptyList()) }
            say("${name}에 도착했습니다. 충분히 쉬고 출발하세요.")
        }
    }

    /** 쉼터 도착 화면의 '다시 출발'. 시연 모의 주행은 경로로 돌아가 이어 달린다. */
    fun resumeAfterRest() {
        simulator?.resume()
        _arrival.value = null
    }

    /** 졸음 감지: 기록하고, 단계에 맞게 차량이 개입하고, 가까운 쉼터를 찾는다 */
    fun detectDrowsiness(record: Boolean = true) {
        _drowsyAlerts.value += 1
        if (record) signal(EventType.DROWSINESS)
        val repeated = (recorder?.drowsinessCount ?: 1) >= 2
        // 이어지는 같은 졸음이면 이미 찾은 쉼터를 그대로 쓴다
        if (record || _suggestion.value.place == null) openSuggestion(TriggerType.DROWSINESS)
        intervene(
            if (repeated) CabinAlert.DROWSY_REPEATED else CabinAlert.DROWSY,
            if (repeated) "졸음이 반복되고 있습니다. 창문을 조금 열었어요. 가까운 쉼터에서 꼭 쉬어 가세요."
            else "졸음이 감지됐습니다. 시원한 바깥 공기를 들이고 있어요. 가까운 쉼터에서 쉬어 가세요.",
        )
    }

    /** 전방 주시 이탈 감지: 기록하고 음성·무드등으로 알린다 */
    fun detectInattention() {
        signal(EventType.INATTENTION)
        intervene(CabinAlert.INATTENTION, "전방을 주시해 주세요.")
    }

    /** 정면을 다시 봤을 때: 부주의 단계의 무드등만 되돌린다(졸음 단계면 쉼터에 들를 때까지 유지) */
    fun inattentionCleared() {
        if (_cabinUi.value.level != CabinAlert.INATTENTION) return
        viewModelScope.launch {
            cabin.restore()
            _cabinUi.update { it.copy(level = null, actions = emptyList()) }
        }
    }

    /** 차량 개입: 음성 안내와 실내 조작을 함께 하고, 반영된 조치를 화면에 보여 준다. 더 낮은 단계로는 내려가지 않는다. */
    private fun intervene(level: CabinAlert, message: String) {
        val current = _cabinUi.value.level
        val apply = current == null || level.ordinal >= current.ordinal
        interveneJob = viewModelScope.launch {
            val spoken = async { say(message) }
            if (apply) {
                val done = cabin.alert(level)
                val voiced = voice.connect()
                _cabinUi.update { ui ->
                    ui.copy(level = level, actions = (if (voiced) listOf(CabinAction(CabinAction.Kind.VOICE, "음성 안내")) else emptyList()) + done)
                }
            }
            spoken.await()
        }
    }

    /** 음성 안내. 쓸 수 없으면 조용히 넘어간다. */
    private suspend fun say(text: String) {
        runCatching { if (voice.connect()) voice.speak(text) }
    }

    /**
     * 감지 신호 1건(지금은 화면의 시연 신호, 나중에는 AI 추론 결과).
     * 전방 주시 이탈이 3회 누적될 때마다 true 를 돌려준다(정면 주시가 확인되면 쉼터를 제안).
     */
    fun signal(type: String): Boolean {
        val rec = recorder ?: return false
        rec.recordEvent(type)
        _observe.update { it.copy(inattentionCount = rec.inattentionCount, drowsinessCount = rec.drowsinessCount) }
        return type == EventType.INATTENTION && rec.inattentionCount > 0 && rec.inattentionCount % INATTENTION_SUGGEST_EVERY == 0
    }

    /** 이번 주행의 전방 주시 이탈이 3회 단위로 쌓였는지(정면 주시 확인 시 제안 여부) */
    val inattentionDue: Boolean
        get() = (recorder?.inattentionCount ?: 0).let { it > 0 && it % INATTENTION_SUGGEST_EVERY == 0 }

    // ════════════════════════ 쉼터 제안 ════════════════════════

    /** 제안 화면을 열 때: 현재 위치로 가장 가까운 쉼터를 묻고, 차량 내비 상태를 확인한다. */
    fun openSuggestion(trigger: String) {
        _suggestion.value = SuggestionUi(loading = true, trigger = trigger)
        viewModelScope.launch {
            val fix = recorder?.lastFix
            val navi = async { navigation.routeState() }
            if (fix == null) {
                _suggestion.update { it.copy(loading = false, noPlaceReason = "현재 위치를 확인할 수 없어 쉼터를 찾지 못했습니다.", navi = navi.await()) }
                return@launch
            }
            val r = api.nearbyRestPlace(NearbyRequest(fix.latitude, fix.longitude, fix.bearingDeg?.toDouble(), fix.speedMps?.toDouble()))
            val reason = when {
                r is ApiResult.Failure -> r.message
                r.getOrNull()?.place != null -> null
                r.getOrNull()?.onHighway == false -> "고속도로가 아닌 곳에서는 쉼터를 안내하지 않습니다. 안전한 곳에 정차해 주세요."
                else -> "진행 방향에서 바로 들어갈 수 있는 쉼터를 확인하지 못했습니다. 안전한 곳에 정차해 주세요."
            }
            val place = r.getOrNull()?.place
            _suggestion.update { it.copy(loading = false, place = place, noPlaceReason = reason, navi = navi.await()) }
        }
    }

    /**
     * 차량 내비에 쉼터를 넣는다([destination] 이면 쉼터로 길 안내). Pleos 내비에 연결되지 않으면 null(화면은 시연 흐름으로 넘어간다).
     * 내비가 반영을 확인하면 주행 기록에 경유지를 남긴다.
     */
    suspend fun addRestStop(destination: Boolean): NaviResult? {
        val place = _suggestion.value.place ?: return NaviResult.Failed("쉼터 정보가 없습니다")
        fun record() {
            addedPlaces[place.placeId] = place
            recorder?.recordWaypoint(
                place.placeId, if (destination) NavMode.DESTINATION else NavMode.WAYPOINT, _suggestion.value.trigger,
                latitude = place.latitude, longitude = place.longitude,
            )
        }
        val nav = navigation
        if (!nav.connect()) {
            // 테스트용(debug): Pleos 내비가 없는 탭에서도 경유지 추가·쉼터 도착 기록을 시험할 수 있게 기록만 남긴다
            if (DebugFlags.DUMMY_ROUTE) record()
            return null
        }
        _suggestion.update { it.copy(busy = true, failure = null) }
        val stop = RestStop(place.name, placeTypeLabel(place.type), place.latitude, place.longitude)
        val result = if (destination) nav.routeToRestStop(stop) else nav.addRestStop(stop)
        if (result == NaviResult.Added) {
            record()
            viewModelScope.launch { say(if (destination) "${place.name}${josaRo(place.name)} 안내를 시작합니다." else "${place.name}${josaReul(place.name)} 경유지로 추가했습니다.") }
        }
        _suggestion.update { it.copy(busy = false, failure = (result as? NaviResult.Failed)?.reason) }
        return result
    }

    /**
     * 쉼터를 경유지로 넣을지 음성으로 묻고 한 번 듣는다. '응'이면 true, '아니'면 false, 못 들었으면 null.
     * 음성을 쓸 수 없으면 바로 null(화면 버튼으로 고른다).
     */
    suspend fun askToAddRestStop(): Boolean? {
        val place = _suggestion.value.place ?: return null
        if (!runCatching { voice.connect() }.getOrDefault(false)) return null
        _voiceUi.value = VoiceUi(asking = true)
        voice.speak("${place.name}${josaReul(place.name)} 경유지로 추가할까요? 응 이라고 답해 주세요.")
        _voiceUi.value = VoiceUi(listening = true)
        val heard = voice.listen(LISTEN_MS)
        val answer = VoiceGuide.yesNo(heard)
        _voiceUi.value = VoiceUi(heard = heard, answer = answer)
        return answer
    }

    fun stopVoice() {
        runCatching { voice.stop() }
        _voiceUi.value = VoiceUi()
    }

    // ════════════════════════ 종료 · 리포트 ════════════════════════

    /** '종료하고 리포트 보기'. 기록을 올리고 리포트를 받는다. 실패하면 기기에 남겨 두고 다음 실행 때 다시 올린다. */
    suspend fun endDrive(): Boolean {
        val rec = recorder ?: return false
        _end.value = EndUi(busy = true)
        interveneJob?.cancel()
        runCatching { voice.stop() }
        cabin.restore()
        val body = rec.toEndRequest()
        val driveId = rec.driveId ?: api.startDrive(rec.clientDriveId, rec.startedAt).getOrNull()?.driveId?.also { rec.driveId = it }
        val result = driveId?.let { api.endDrive(it, body) }
        val report = result?.getOrNull()
        if (report == null) {
            savePending(PendingUpload(rec.clientDriveId, rec.startedAt, driveId, body))
            stopDriveLocally()
            _end.value = EndUi(error = (result as? ApiResult.Failure)?.message ?: "서버에 연결할 수 없습니다.")
            return false
        }
        stopDriveLocally()
        _end.value = EndUi()
        _report.value = ReportUi(report = report)
        return true
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
        interveneJob?.cancel()
        cabinStatusJob?.cancel()
        cabinStatusJob = null
        locationJob?.cancel()
        locationJob = null
        simulator = null
        recorder = null
        if (DemoMode.enabled) demoNavigation.clear()
        _observe.value = ObserveUi()
        _cabinUi.value = CabinUi()
        _voiceUi.value = VoiceUi()
        _arrival.value = null
    }

    // ════════════════════════ 이전 기록 ════════════════════════

    /** [date] 가 null 이면 가장 최근 주행일 */
    fun loadDaily(date: LocalDate?) {
        _daily.update { it.copy(loading = true, error = null) }
        viewModelScope.launch {
            val r = api.daily(date?.toString())
            _daily.value = DailyUi(data = r.getOrNull() ?: _daily.value.data, error = (r as? ApiResult.Failure)?.message)
        }
    }

    /** [weekStart] 는 월요일. null 이면 가장 최근 주행일이 속한 주 */
    fun loadWeekly(weekStart: LocalDate?) {
        _weekly.update { it.copy(loading = true, error = null) }
        viewModelScope.launch {
            val r = api.weekly(weekStart?.toString())
            _weekly.value = WeeklyUi(data = r.getOrNull() ?: _weekly.value.data, error = (r as? ApiResult.Failure)?.message)
        }
    }

    fun loadCalendar(month: YearMonth) {
        _calendar.value = CalendarUi(month = month, loading = true)
        viewModelScope.launch {
            _calendar.value = CalendarUi(month = month, data = api.calendar(month.toString()).getOrNull())
        }
    }

    // ════════════════════════ 올리지 못한 기록 ════════════════════════

    private val pendingSerializer = ListSerializer(PendingUpload.serializer())

    private fun loadPending(): List<PendingUpload> = pendingStore.getString(KEY_PENDING, null)
        ?.let { runCatching { HawkeyeApi.json.decodeFromString(pendingSerializer, it) }.getOrNull() }
        .orEmpty()

    private fun savePending(item: PendingUpload) {
        val list = loadPending().filterNot { it.clientDriveId == item.clientDriveId } + item
        pendingStore.edit().putString(KEY_PENDING, HawkeyeApi.json.encodeToString(pendingSerializer, list)).apply()
    }

    private suspend fun retryPendingUploads() {
        val remaining = loadPending().filterNot { p ->
            val id = p.driveId ?: api.startDrive(p.clientDriveId, p.startedAt).getOrNull()?.driveId
            val r = id?.let { api.endDrive(it, p.body) }
            // 성공했거나, 주행이 없어졌거나(404) 형식 오류(400)면 다시 시도해도 소용없으니 지운다
            r is ApiResult.Success || (r is ApiResult.Failure && r.status in 400..404 && r.status != 401)
        }
        pendingStore.edit().putString(KEY_PENDING, HawkeyeApi.json.encodeToString(pendingSerializer, remaining)).apply()
    }

    override fun onCleared() {
        DriverFrames.sink = null
        detector?.close()
        locationJob?.cancel()
        cabinStatusJob?.cancel()
    }

    companion object {
        /** 전방 주시 이탈이 이 횟수만큼 쌓일 때마다 쉼터를 제안한다 */
        const val INATTENTION_SUGGEST_EVERY = 3
        /** 직전 졸음 감지 뒤 이 시간 안의 감지는 같은 졸음으로 보고 기록 횟수를 늘리지 않는다(경고는 다시 낸다) */
        const val DROWSY_REPEAT_MS = 10_000L
        /** 쉼터 제안에 음성으로 답을 기다리는 시간 */
        const val LISTEN_MS = 6_000L
        private const val KEY_PENDING = "pending"

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
