package com.personal.driverguard.ui.state

import android.content.Context
import com.personal.driverguard.data.remote.AgreeRequest
import com.personal.driverguard.data.remote.AgreementItem
import com.personal.driverguard.data.remote.AgreementsResponse
import com.personal.driverguard.data.remote.DriveListResponse
import com.personal.driverguard.data.remote.DriveReport
import com.personal.driverguard.data.remote.EndDriveRequest
import com.personal.driverguard.data.remote.EventType
import com.personal.driverguard.data.remote.HawkeyeApi
import com.personal.driverguard.data.remote.MeResponse
import com.personal.driverguard.data.remote.NearbyRequest
import com.personal.driverguard.data.remote.NearbyResponse
import com.personal.driverguard.data.remote.NearestPlace
import com.personal.driverguard.data.remote.SignupRequest
import com.personal.driverguard.data.remote.SignupResponse
import com.personal.driverguard.data.remote.StartDriveRequest
import com.personal.driverguard.data.remote.StartDriveResponse
import com.personal.driverguard.data.remote.TermsDetail
import com.personal.driverguard.data.remote.TermsListResponse
import com.personal.driverguard.data.remote.TermsSummary
import com.personal.driverguard.data.remote.TokenResponse
import kotlinx.serialization.KSerializer
import okhttp3.Interceptor
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.Protocol
import okhttp3.Request
import okhttp3.Response
import okhttp3.ResponseBody.Companion.toResponseBody
import okio.Buffer
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.YearMonth
import java.time.temporal.ChronoUnit
import kotlin.math.roundToInt

/**
 * 시연 모드의 가짜 백엔드. [DemoMode] 가 켜져 있으면 모든 API 요청에 서버 대신 응답하고, 꺼져 있으면 그대로 서버로 보낸다.
 * 응답 형식은 실제 백엔드와 같아서 앱의 나머지 코드는 시연인지 모른다.
 *
 *  - 로그인: 아무 이메일·비밀번호나 받는다(비어 있으면 앱이 먼저 막는다). 회원가입은 약관 동의 화면으로 이어진다.
 *  - 쉼터 조회: 받은 위치를 경부고속도로 경로에 맞춰 진행 방향 앞의 첫 쉼터를 돌려준다([DemoRoute]).
 *  - 주행 종료: 올린 기록으로 리포트를 만들고, 이전 기록에도 넣는다([FakeHistory.addRecorded]).
 *  - 이전 기록: 이번 주·지난주 예시 기록([FakeHistory]).
 */
class DemoServer(private val loadRoute: () -> DemoRoute) : Interceptor {
    private val route by lazy { loadRoute() }
    private val json = HawkeyeApi.json

    override fun intercept(chain: Interceptor.Chain): Response {
        if (!DemoMode.enabled) return chain.proceed(chain.request())
        val req = chain.request()
        return try {
            handle(req)
        } catch (e: Exception) {
            respond(req, 500, """{"status":500,"code":"DEMO_ERROR","message":"시연 응답을 만들지 못했습니다: ${e.message?.replace("\"", "'")}"}""")
        }
    }

    private fun handle(req: Request): Response {
        val path = req.url.encodedPath.removePrefix("/")
        val body = req.body?.let { b -> Buffer().also { b.writeTo(it) }.readUtf8() }.orEmpty()
        val method = req.method
        fun ok(text: String) = respond(req, 200, text)
        fun <T> ok(serializer: KSerializer<T>, value: T) = ok(json.encodeToString(serializer, value))
        val noContent = respond(req, 204, "")
        return when {
            // ── 인증 ──
            path == "api/auth/signup" -> {
                val r = json.decodeFromString(SignupRequest.serializer(), body)
                State.name = r.name
                State.email = r.email
                State.agreed = false
                ok(SignupResponse.serializer(), SignupResponse(USER_ID, r.email, r.name, "Bearer", ACCESS, 3600, REFRESH, termsAgreementRequired = true))
            }
            path == "api/auth/login" || path == "api/auth/refresh" ->
                ok(TokenResponse.serializer(), TokenResponse("Bearer", ACCESS, 3600, REFRESH, termsAgreementRequired = !State.agreed))
            path == "api/auth/logout" -> noContent
            // ── 사용자·약관 ──
            path == "api/users/me" && method == "DELETE" -> noContent
            path == "api/users/me" ->
                ok(MeResponse.serializer(), MeResponse(USER_ID, State.email, State.name, "ACTIVE", "2026-10-01T09:00:00+09:00", !State.agreed))
            path == "api/terms" -> ok(TermsListResponse.serializer(), TermsListResponse(TERMS.map { it.summary() }))
            path.startsWith("api/terms/") -> {
                val id = path.substringAfterLast('/').toLong()
                val t = TERMS.first { it.termsId == id }
                ok(TermsDetail.serializer(), TermsDetail(t.termsId, t.code, t.title, 1, t.required, EFFECTIVE, t.content))
            }
            path == "api/users/me/terms-agreements" && method == "POST" -> {
                val r = json.decodeFromString(AgreeRequest.serializer(), body)
                State.checks = r.agreements.associate { it.termsId to it.agreed }
                State.agreed = TERMS.filter { it.required }.all { State.checks[it.termsId] == true }
                ok(AgreementsResponse.serializer(), agreements())
            }
            path == "api/users/me/terms-agreements" -> ok(AgreementsResponse.serializer(), agreements())
            // ── 쉼터 ──
            path == "api/rest-places/nearby" -> ok(NearbyResponse.serializer(), nearby(json.decodeFromString(NearbyRequest.serializer(), body)))
            // ── 주행 ──
            path == "api/drives" && method == "POST" -> {
                val r = json.decodeFromString(StartDriveRequest.serializer(), body)
                val id = State.nextDriveId++
                State.started[id] = r.startedAt
                ok(StartDriveResponse.serializer(), StartDriveResponse(id, "IN_PROGRESS", r.startedAt))
            }
            path.matches(Regex("api/drives/\\d+/end")) -> {
                val id = path.split('/')[2].toLong()
                val report = endReport(id, json.decodeFromString(EndDriveRequest.serializer(), body))
                FakeHistory.addRecorded(report)
                ok(DriveReport.serializer(), report)
            }
            path.matches(Regex("api/drives/\\d+")) -> {
                val report = FakeHistory.report(path.substringAfterLast('/').toLong(), route)
                if (report == null) respond(req, 404, """{"status":404,"code":"DRIVE_NOT_FOUND","message":"주행 기록을 찾을 수 없습니다."}""")
                else ok(DriveReport.serializer(), report)
            }
            path == "api/drives" -> {
                val size = req.url.queryParameter("size")?.toIntOrNull() ?: 20
                ok(DriveListResponse.serializer(), DriveListResponse(FakeHistory.recent(size)))
            }
            // ── 이전 기록 ──
            path == "api/reports/calendar" ->
                ok(com.personal.driverguard.data.remote.CalendarResponse.serializer(), FakeHistory.calendar(YearMonth.parse(req.url.queryParameter("month"))))
            path == "api/reports/daily" ->
                ok(com.personal.driverguard.data.remote.DailyReport.serializer(), FakeHistory.daily(req.url.queryParameter("date")?.let(LocalDate::parse)))
            path == "api/reports/weekly" ->
                ok(com.personal.driverguard.data.remote.WeeklyReport.serializer(), FakeHistory.weekly(req.url.queryParameter("weekStart")?.let(LocalDate::parse)))
            else -> respond(req, 404, """{"status":404,"code":"NOT_FOUND","message":"시연 모드에서 지원하지 않는 요청입니다."}""")
        }
    }

    /** 위치를 경로에 맞춰 진행 방향 앞의 첫 쉼터. 경로에서 멀면(고속도로가 아님) 쉼터를 주지 않는다. */
    internal fun nearby(r: NearbyRequest): NearbyResponse {
        val (at, off) = route.project(r.lat, r.lng)
        if (off > ON_ROUTE_M) return NearbyResponse(directionApplied = true, onHighway = false, distanceSource = "ROAD", place = null)
        val rest = route.nextRest(at) ?: return NearbyResponse(directionApplied = true, onHighway = true, distanceSource = "ROAD", place = null)
        val meters = (rest.routeMeters - at).roundToInt()
        return NearbyResponse(
            directionApplied = true, onHighway = true, distanceSource = "ROAD",
            place = NearestPlace(rest.placeId, rest.type, rest.name, rest.lat, rest.lng, meters, (meters / DriveSimulator.DEMO_SPEED_MPS).roundToInt()),
        )
    }

    /** 올린 주행 기록으로 리포트를 만든다(백엔드와 같은 형식) */
    internal fun endReport(driveId: Long, b: EndDriveRequest): DriveReport {
        val startedAt = State.started[driveId] ?: b.endedAt
        val real = ChronoUnit.SECONDS.between(OffsetDateTime.parse(startedAt), OffsetDateTime.parse(b.endedAt)).toInt().coerceAtLeast(0)
        // 시연은 쉼터까지 6초 만에 빨리 감으므로, 리포트의 주행 시간은 이동 거리를 시속 80km로 달린 시간 이상으로 맞춘다
        val observed = maxOf(real, (b.distanceMeters / DriveSimulator.DEMO_SPEED_MPS).roundToInt())
        val endedAt = OffsetDateTime.parse(startedAt).plusSeconds(observed.toLong()).toString()
        val pts = b.route
        return DriveReport(
            driveId = driveId, status = "ENDED", startedAt = startedAt, endedAt = endedAt,
            observedSeconds = observed, distanceMeters = b.distanceMeters,
            inattentionCount = b.events.count { it.type == EventType.INATTENTION },
            drowsinessCount = b.events.count { it.type == EventType.DROWSINESS },
            route = DriveReport.Route(
                pts.firstOrNull()?.let { DriveReport.Point(it[0], it[1]) },
                pts.lastOrNull()?.let { DriveReport.Point(it[0], it[1]) },
                pts,
            ),
            events = b.events.map { DriveReport.EventItem(it.type, it.detectedAt, it.latitude, it.longitude) },
            waypoints = b.waypoints.mapIndexed { i, w ->
                val rest = route.rest(w.placeId)
                DriveReport.WaypointItem(i + 1L, w.placeId, rest?.name, rest?.type, w.mode, w.trigger, w.addedAt, visited = w.visitedAt != null, visitedAt = w.visitedAt)
            },
            skipped = DriveReport.Skipped(0, 0),
        )
    }

    private fun agreements() = AgreementsResponse(
        termsAgreementRequired = !State.agreed,
        agreements = TERMS.map { t ->
            val agreed = if (State.agreed && State.checks.isEmpty()) true else State.checks[t.termsId]
            AgreementItem(t.termsId, t.code, t.title, 1, t.required, agreed, if (agreed == true) EFFECTIVE else null)
        },
    )

    private fun respond(req: Request, code: Int, text: String): Response = Response.Builder()
        .request(req)
        .protocol(Protocol.HTTP_1_1)
        .code(code)
        .message(if (code < 400) "OK" else "Error")
        .body(text.toResponseBody(JSON))
        .build()

    /** 시연 계정 상태(앱이 켜져 있는 동안) */
    private object State {
        @Volatile var name = "김하늘"
        @Volatile var email = "haneul.kim@mail.com"
        @Volatile var agreed = true
        @Volatile var checks: Map<Long, Boolean> = emptyMap()
        @Volatile var nextDriveId = 9_000_001L
        val started = java.util.concurrent.ConcurrentHashMap<Long, String>()
    }

    private class Term(val termsId: Long, val code: String, val title: String, val required: Boolean, val content: String) {
        fun summary() = TermsSummary(termsId, code, title, 1, required, EFFECTIVE)
    }

    companion object {
        fun create(context: Context): DemoServer {
            val app = context.applicationContext
            return DemoServer { DemoRoute.load(app) }
        }

        private val JSON = "application/json".toMediaType()
        private const val USER_ID = 1L
        private const val ACCESS = "demo-access-token"
        private const val REFRESH = "demo-refresh-token"
        private const val EFFECTIVE = "2026-10-01T00:00:00+09:00"
        /** 이보다 경로에서 멀면 고속도로 밖으로 본다 */
        private const val ON_ROUTE_M = 500.0

        private val TERMS = listOf(
            Term(1, "SERVICE", "서비스 이용약관", true, "# 제1조 목적\n이 약관은 Hawkeye가 제공하는 운전자 상태 관찰, 경고, 쉼터 안내, 주행 리포트 서비스의 이용 조건을 정합니다.\n\n# 제2조 서비스의 한계\n경고와 안내는 운전을 돕는 보조 수단입니다. 안전 운전의 책임은 운전자에게 있습니다."),
            Term(2, "PRIVACY", "개인정보 수집·이용", true, "# 수집 항목\n이름, 이메일, 주행 기록(관찰 시작·종료 시각, 이동 거리, 감지된 이벤트의 시각·유형)\n\n# 이용 목적과 보관\n주행 리포트와 주간 통계를 만드는 데 씁니다. 탈퇴하면 바로 지웁니다."),
            Term(3, "LOCATION", "위치정보 이용", true, "쉼터를 찾고 이동 경로와 이벤트 위치를 기록하는 데 씁니다. 관찰 중에만 수집합니다."),
            Term(4, "FACE", "얼굴 영상 분석", true, "# 분석 항목\n눈 감김 비율, 눈을 감고 있던 시간, 머리 방향. 결과는 추정이며 의학적 진단이 아닙니다.\n\n# 보관\n영상과 사진은 저장하지 않고 태블릿 안에서만 분석합니다."),
            Term(5, "NOTIFICATION", "주행 리포트 알림", false, "주간 통계와 반복되는 감지 양상을 알림으로 보내 드립니다. 동의하지 않아도 모든 기능을 쓸 수 있습니다."),
        )
    }
}
