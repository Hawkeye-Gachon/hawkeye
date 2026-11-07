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
import kotlinx.coroutines.Job
import kotlinx.coroutines.async
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

/** 다음에 열 화면(내비게이션은 화면 쪽이 한다) */
enum class Next { LOGIN, TERMS, HOME }

/**
 * 앱 전체 상태와 백엔드·차량 내비 호출을 맡는다. 화면은 상태를 읽고, 동작은 여기 함수를 부른 뒤 결과에 따라 이동한다.
 *
 * 연결 범위(임시 포함):
 *  - 졸음·전방 주시 이탈 '감지'는 AI 추론 서버가 아직 없어 화면의 시연 신호가 [signal] 을 부른다. AI 연동 시 같은 함수를 쓴다.
 *  - 위치는 일반 Android 위치(Pleos FusedLocation 은 준비 중), 경유지 추가는 Pleos NaviHelper.
 */
class HawkeyeViewModel(app: Application) : AndroidViewModel(app) {

    private val context: Context get() = getApplication()
    val api: HawkeyeApi = RepositoryLocator.provideApi(app)
    private val navigation = PlatformLocator.provideVehicleNavigation(app)
    private val location = LocationTracker(app)
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

    private var recorder: DriveRecorder? = null
    private var locationJob: Job? = null

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

    /** '관찰 시작'(보는 방향 설정 완료 후). 주행을 등록하고 위치 기록을 시작한다. */
    fun startDrive() {
        if (recorder != null) return
        val rec = DriveRecorder(startedMonotonicMs = SystemClock.elapsedRealtime())
        recorder = rec
        _observe.value = ObserveUi(active = true, startedMonotonicMs = rec.startedMonotonicMs, locationEnabled = location.hasPermission())
        _report.value = ReportUi()
        _end.value = EndUi()
        locationJob = viewModelScope.launch { location.updates().collect { rec.onLocation(it) } }
        // 등록에 실패해도 관찰은 계속한다. 종료할 때 다시 등록한다.
        viewModelScope.launch { api.startDrive(rec.clientDriveId, rec.startedAt).getOrNull()?.let { rec.driveId = it.driveId } }
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
            _suggestion.update { it.copy(loading = false, place = r.getOrNull()?.place, noPlaceReason = reason, navi = navi.await()) }
        }
    }

    /**
     * 차량 내비에 쉼터를 넣는다([destination] 이면 쉼터로 길 안내). Pleos 내비에 연결되지 않으면 null(화면은 시연 흐름으로 넘어간다).
     * 내비가 반영을 확인하면 주행 기록에 경유지를 남긴다.
     */
    suspend fun addRestStop(destination: Boolean): NaviResult? {
        val place = _suggestion.value.place ?: return NaviResult.Failed("쉼터 정보가 없습니다")
        if (!navigation.connect()) return null
        _suggestion.update { it.copy(busy = true, failure = null) }
        val stop = RestStop(place.name, placeTypeLabel(place.type), place.latitude, place.longitude)
        val result = if (destination) navigation.routeToRestStop(stop) else navigation.addRestStop(stop)
        if (result == NaviResult.Added) {
            recorder?.recordWaypoint(place.placeId, if (destination) NavMode.DESTINATION else NavMode.WAYPOINT, _suggestion.value.trigger)
        }
        _suggestion.update { it.copy(busy = false, failure = (result as? NaviResult.Failed)?.reason) }
        return result
    }

    // ════════════════════════ 종료 · 리포트 ════════════════════════

    /** '종료하고 리포트 보기'. 기록을 올리고 리포트를 받는다. 실패하면 기기에 남겨 두고 다음 실행 때 다시 올린다. */
    suspend fun endDrive(): Boolean {
        val rec = recorder ?: return false
        _end.value = EndUi(busy = true)
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
        locationJob?.cancel()
        locationJob = null
        recorder = null
        _observe.value = ObserveUi()
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
        locationJob?.cancel()
    }

    companion object {
        /** 전방 주시 이탈이 이 횟수만큼 쌓일 때마다 쉼터를 제안한다 */
        const val INATTENTION_SUGGEST_EVERY = 3
        private const val KEY_PENDING = "pending"

        fun placeTypeLabel(type: String?) = when (type) {
            PlaceType.REST_AREA -> "휴게소"
            PlaceType.DROWSY_SHELTER -> "졸음쉼터"
            else -> "쉼터"
        }
    }
}
