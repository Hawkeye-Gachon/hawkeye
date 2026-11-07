package com.personal.driverguard.data.remote

import kotlinx.coroutines.test.runTest
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/**
 * 백엔드 응답 형식(backend/ 의 DTO·README 예시)을 흉내 낸 가짜 서버로 클라이언트 동작을 확인한다.
 * TokenStore 는 SharedPreferences 를 쓰므로 여기서는 메모리 구현으로 대신한다.
 */
class HawkeyeApiTest {
    private lateinit var server: MockWebServer
    private lateinit var tokens: TokenStore
    private lateinit var api: HawkeyeApi

    @Before
    fun setUp() {
        server = MockWebServer().apply { start() }
        tokens = FakeTokenStore.create()
        api = HawkeyeApi(server.url("/").toString(), tokens)
    }

    @After
    fun tearDown() = server.shutdown()

    private fun json(code: Int, body: String) = MockResponse().setResponseCode(code).setHeader("Content-Type", "application/json").setBody(body)

    @Test
    fun loginSavesTokensAndSendsBearerAfterwards() = runTest {
        server.enqueue(json(200, """{"grantType":"Bearer","accessToken":"A1","accessTokenExpiresIn":1800,"refreshToken":"R1","termsAgreementRequired":true}"""))
        server.enqueue(json(200, """{"userId":7,"email":"a@b.com","name":"김하늘","status":"ACTIVE","createdAt":"2026-11-07T10:00:00+09:00","termsAgreementRequired":false}"""))

        val login = api.login("a@b.com", "drive1234")
        assertTrue(login is ApiResult.Success && login.data.termsAgreementRequired)
        assertEquals("A1", tokens.accessToken)
        assertTrue(api.loggedIn.value)

        val loginReq = server.takeRequest()
        assertEquals("/api/auth/login", loginReq.path)
        assertNull("인증 API 에는 토큰을 붙이지 않는다", loginReq.getHeader("Authorization"))

        val me = api.me()
        assertEquals("김하늘", me.getOrNull()?.name)
        assertEquals("Bearer A1", server.takeRequest().getHeader("Authorization"))
    }

    @Test
    fun expiredAccessTokenIsRefreshedOnceAndRetried() = runTest {
        tokens.save("OLD", "R1")
        server.enqueue(json(401, """{"status":401,"code":"UNAUTHORIZED","message":"인증이 필요합니다."}"""))
        server.enqueue(json(200, """{"grantType":"Bearer","accessToken":"NEW","accessTokenExpiresIn":1800,"refreshToken":"R2","termsAgreementRequired":false}"""))
        server.enqueue(json(200, """{"month":"2026-11","driveDates":["2026-11-02","2026-11-06"]}"""))

        val result = api.calendar("2026-11")
        assertEquals(listOf("2026-11-02", "2026-11-06"), result.getOrNull()?.driveDates)

        assertEquals("Bearer OLD", server.takeRequest().getHeader("Authorization"))
        val refresh = server.takeRequest()
        assertEquals("/api/auth/refresh", refresh.path)
        assertTrue(refresh.body.readUtf8().contains("\"refreshToken\":\"R1\""))
        assertEquals("Bearer NEW", server.takeRequest().getHeader("Authorization"))
        assertEquals("R2", tokens.refreshToken)
    }

    @Test
    fun failedRefreshLogsOut() = runTest {
        tokens.save("OLD", "R1")
        server.enqueue(json(401, """{"status":401,"code":"UNAUTHORIZED"}"""))
        server.enqueue(json(401, """{"status":401,"code":"INVALID_REFRESH_TOKEN"}"""))

        val result = api.me()
        assertTrue(result is ApiResult.Failure && result.status == 401)
        assertNull(tokens.refreshToken)
        assertTrue(!api.loggedIn.value)
    }

    @Test
    fun validationErrorKeepsFieldMessages() = runTest {
        server.enqueue(
            json(
                400,
                """{"timestamp":"2026-11-07T01:00:00Z","status":400,"error":"Bad Request","code":"VALIDATION_FAILED","message":"입력값을 확인해 주세요.","path":"/api/auth/signup",
                   "fieldErrors":[{"field":"email","message":"이메일 형식이 아닙니다. 예: name@mail.com"}]}""",
            ),
        )
        val result = api.signup("haneul.kim@mail", "김하늘", "drive1234", true)
        result as ApiResult.Failure
        assertEquals("VALIDATION_FAILED", result.code)
        assertEquals("이메일 형식이 아닙니다. 예: name@mail.com", result.fieldErrors["email"])
    }

    @Test
    fun termsRequiredIsReportedAsFailure() = runTest {
        tokens.save("A", "R")
        server.enqueue(json(403, """{"status":403,"code":"TERMS_AGREEMENT_REQUIRED","message":"필수 약관 동의가 필요합니다."}"""))
        val result = api.daily(null)
        assertEquals("TERMS_AGREEMENT_REQUIRED", (result as ApiResult.Failure).code)
    }

    @Test
    fun networkErrorBecomesFailure() = runTest {
        server.shutdown()
        val result = api.terms()
        assertEquals(ApiResult.NETWORK, (result as ApiResult.Failure).code)
    }

    @Test
    fun endDriveSendsBodyAndParsesReport() = runTest {
        tokens.save("A", "R")
        server.enqueue(
            json(
                200,
                """{"driveId":12,"status":"ENDED","startedAt":"2026-11-06T14:30:00+09:00","endedAt":"2026-11-06T15:12:18+09:00",
                   "observedSeconds":2538,"distanceMeters":28400,"inattentionCount":3,"drowsinessCount":1,
                   "route":{"start":{"latitude":37.1,"longitude":127.1},"end":{"latitude":36.9,"longitude":127.2},"points":[[37.1,127.1],[36.9,127.2]]},
                   "events":[{"type":"DROWSINESS","detectedAt":"2026-11-06T14:55:00+09:00","latitude":37.0,"longitude":127.15}],
                   "waypoints":[{"waypointId":3,"placeId":41,"placeName":"천안 졸음쉼터","placeType":"DROWSY_SHELTER","mode":"WAYPOINT","trigger":"DROWSINESS","addedAt":"2026-11-06T14:56:00+09:00","visited":false}],
                   "skipped":{"events":0,"waypoints":0}}""",
            ),
        )
        val body = EndDriveRequest(
            endedAt = "2026-11-06T15:12:18+09:00",
            distanceMeters = 28400,
            events = listOf(EventRequest(EventType.DROWSINESS, "2026-11-06T14:55:00+09:00", 37.0, 127.15)),
            route = listOf(listOf(37.1, 127.1), listOf(36.9, 127.2)),
            waypoints = listOf(WaypointRequest(41, NavMode.WAYPOINT, TriggerType.DROWSINESS, "2026-11-06T14:56:00+09:00")),
        )
        val report = api.endDrive(12, body).getOrNull()!!
        assertEquals(2538, report.observedSeconds)
        assertEquals("천안 졸음쉼터", report.waypoints.single().placeName)
        assertEquals(2, report.route?.points?.size)

        val sent = server.takeRequest()
        assertEquals("/api/drives/12/end", sent.path)
        val text = sent.body.readUtf8()
        assertTrue(text.contains("\"route\":[[37.1,127.1],[36.9,127.2]]"))
        // visitedAt 은 없으면 아예 보내지 않거나 null 이어야 한다(백엔드는 둘 다 받는다)
        assertTrue(!text.contains("\"visitedAt\":\"\""))
    }

    @Test
    fun nearbyWithoutPlaceIsParsed() = runTest {
        tokens.save("A", "R")
        server.enqueue(json(200, """{"directionApplied":true,"onHighway":false,"distanceSource":"ROAD","place":null}"""))
        val res = api.nearbyRestPlace(NearbyRequest(37.5, 127.0, 90.0, 20.0)).getOrNull()!!
        assertEquals(false, res.onHighway)
        assertNull(res.place)
    }
}
