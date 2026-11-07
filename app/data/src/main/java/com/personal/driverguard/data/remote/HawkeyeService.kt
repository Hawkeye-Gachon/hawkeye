package com.personal.driverguard.data.remote

import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.HTTP
import retrofit2.http.POST
import retrofit2.http.Path
import retrofit2.http.Query

/** DriverGuard 백엔드 엔드포인트. 요청·응답 형식은 ApiModels.kt, 동작 설명은 backend/README.md 와 Swagger 참고. */
interface HawkeyeService {

    // ── 인증(토큰 불필요) ──
    @POST("api/auth/signup")
    suspend fun signup(@Body body: SignupRequest): Response<SignupResponse>

    @POST("api/auth/login")
    suspend fun login(@Body body: LoginRequest): Response<TokenResponse>

    @POST("api/auth/refresh")
    suspend fun refresh(@Body body: RefreshTokenRequest): Response<TokenResponse>

    @POST("api/auth/logout")
    suspend fun logout(@Body body: RefreshTokenRequest): Response<Unit>

    // ── 사용자 ──
    @GET("api/users/me")
    suspend fun me(): Response<MeResponse>

    @HTTP(method = "DELETE", path = "api/users/me", hasBody = true)
    suspend fun withdraw(@Body body: WithdrawRequest): Response<Unit>

    // ── 약관(목록·전문은 토큰 불필요) ──
    @GET("api/terms")
    suspend fun terms(): Response<TermsListResponse>

    @GET("api/terms/{termsId}")
    suspend fun termsDetail(@Path("termsId") termsId: Long): Response<TermsDetail>

    @GET("api/users/me/terms-agreements")
    suspend fun myAgreements(): Response<AgreementsResponse>

    @POST("api/users/me/terms-agreements")
    suspend fun agree(@Body body: AgreeRequest): Response<AgreementsResponse>

    // ── 쉼터 ──
    @POST("api/rest-places/nearby")
    suspend fun nearbyRestPlace(@Body body: NearbyRequest): Response<NearbyResponse>

    // ── 주행 ──
    @POST("api/drives")
    suspend fun startDrive(@Body body: StartDriveRequest): Response<StartDriveResponse>

    @POST("api/drives/{driveId}/end")
    suspend fun endDrive(@Path("driveId") driveId: Long, @Body body: EndDriveRequest): Response<DriveReport>

    @GET("api/drives/{driveId}")
    suspend fun driveReport(@Path("driveId") driveId: Long): Response<DriveReport>

    @GET("api/drives")
    suspend fun drives(@Query("size") size: Int? = null, @Query("cursor") cursor: String? = null): Response<DriveListResponse>

    // ── 이전 기록 ──
    @GET("api/reports/calendar")
    suspend fun calendar(@Query("month") month: String): Response<CalendarResponse>

    @GET("api/reports/daily")
    suspend fun daily(@Query("date") date: String? = null): Response<DailyReport>

    @GET("api/reports/weekly")
    suspend fun weekly(@Query("weekStart") weekStart: String? = null): Response<WeeklyReport>
}
