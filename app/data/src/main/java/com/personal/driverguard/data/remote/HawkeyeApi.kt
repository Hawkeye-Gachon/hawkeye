package com.personal.driverguard.data.remote

import android.content.Context
import android.content.SharedPreferences
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.serialization.json.Json
import okhttp3.Authenticator
import okhttp3.Interceptor
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Route
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Response
import retrofit2.Retrofit
import retrofit2.converter.kotlinx.serialization.asConverterFactory
import java.io.IOException
import java.util.concurrent.TimeUnit

/** API 호출 결과. 실패는 서버가 준 공통 오류 형식(code 로 분기)이거나 네트워크 오류다. */
sealed interface ApiResult<out T> {
    data class Success<T>(val data: T) : ApiResult<T>

    data class Failure(
        /** HTTP 상태. 네트워크 오류면 0 */
        val status: Int,
        /** 백엔드 ErrorCode(예: LOGIN_FAILED, VALIDATION_FAILED). 네트워크 오류면 NETWORK */
        val code: String,
        val message: String,
        /** 입력값 검증 실패일 때 필드별 사유(field → message) */
        val fieldErrors: Map<String, String> = emptyMap(),
    ) : ApiResult<Nothing>

    companion object {
        const val NETWORK = "NETWORK"
    }
}

inline fun <T> ApiResult<T>.onFailure(block: (ApiResult.Failure) -> Unit): ApiResult<T> {
    if (this is ApiResult.Failure) block(this)
    return this
}

fun <T> ApiResult<T>.getOrNull(): T? = (this as? ApiResult.Success)?.data

/** 로그인 상태 변화 중 화면 이동이 필요한 것 */
enum class SessionEvent {
    /** 리프레시 토큰까지 만료돼 다시 로그인해야 함 */
    LOGGED_OUT,

    /** 현재 필수 약관에 동의하지 않아 기능 API 가 403 TERMS_AGREEMENT_REQUIRED 를 줌 */
    TERMS_REQUIRED,
}

/** 토큰을 실제로 보관하는 곳. 앱은 SharedPreferences, 단위 테스트는 메모리를 쓴다. */
interface TokenPersistence {
    fun load(): Pair<String?, String?>
    fun save(access: String, refresh: String)
    fun clear()
}

/** 앱 전용 SharedPreferences 에 토큰을 둔다. 영상·얼굴 정보는 저장하지 않는다. */
class PrefsTokenPersistence(context: Context) : TokenPersistence {
    private val prefs: SharedPreferences = context.applicationContext.getSharedPreferences("auth", Context.MODE_PRIVATE)

    override fun load() = prefs.getString(KEY_ACCESS, null) to prefs.getString(KEY_REFRESH, null)

    override fun save(access: String, refresh: String) {
        prefs.edit().putString(KEY_ACCESS, access).putString(KEY_REFRESH, refresh).apply()
    }

    override fun clear() {
        prefs.edit().clear().apply()
    }

    private companion object {
        const val KEY_ACCESS = "access_token"
        const val KEY_REFRESH = "refresh_token"
    }
}

/** 현재 토큰. 여러 스레드(요청·재발급)에서 읽고 쓴다. */
class TokenStore(private val persistence: TokenPersistence) {
    @Volatile var accessToken: String?
        private set

    @Volatile var refreshToken: String?
        private set

    init {
        val (a, r) = persistence.load()
        accessToken = a
        refreshToken = r
    }

    val hasSession: Boolean get() = refreshToken != null

    @Synchronized
    fun save(access: String, refresh: String) {
        accessToken = access
        refreshToken = refresh
        persistence.save(access, refresh)
    }

    @Synchronized
    fun clear() {
        accessToken = null
        refreshToken = null
        persistence.clear()
    }

    companion object {
        fun create(context: Context) = TokenStore(PrefsTokenPersistence(context))
    }
}

/**
 * DriverGuard 백엔드 클라이언트.
 *
 * - 모든 요청에 accessToken 을 붙인다(인증 API 제외).
 * - 401 이면 refreshToken 으로 한 번 재발급하고 다시 보낸다. 재발급도 실패하면 토큰을 지우고 [SessionEvent.LOGGED_OUT].
 * - 403 TERMS_AGREEMENT_REQUIRED 면 [SessionEvent.TERMS_REQUIRED] 를 알린다(앱은 약관 동의 화면으로).
 * - 결과는 예외 대신 [ApiResult] 로 돌려준다.
 */
class HawkeyeApi(
    baseUrl: String,
    private val tokens: TokenStore,
    debugLogging: Boolean = false,
) {
    private val _events = MutableSharedFlow<SessionEvent>(extraBufferCapacity = 4)
    val sessionEvents: SharedFlow<SessionEvent> = _events.asSharedFlow()

    private val _loggedIn = MutableStateFlow(tokens.hasSession)
    val loggedIn: StateFlow<Boolean> = _loggedIn.asStateFlow()

    private val normalizedBaseUrl = if (baseUrl.endsWith("/")) baseUrl else "$baseUrl/"
    private val refreshLock = Any()

    /** 재발급 전용(인증기·인증 헤더 없음) */
    private val bareClient = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(15, TimeUnit.SECONDS)
        .build()

    private val authHeader = Interceptor { chain ->
        val request = chain.request()
        val token = tokens.accessToken
        if (token == null || request.url.encodedPath.startsWith("/api/auth/")) {
            chain.proceed(request)
        } else {
            chain.proceed(request.newBuilder().header("Authorization", "Bearer $token").build())
        }
    }

    private val refreshOn401 = Authenticator { _: Route?, response: okhttp3.Response ->
        val failedRequest = response.request
        if (failedRequest.url.encodedPath.startsWith("/api/auth/")) return@Authenticator null
        // 이미 한 번 재시도한 요청이면 포기한다
        if (response.priorResponse != null) return@Authenticator null
        val usedToken = failedRequest.header("Authorization")?.removePrefix("Bearer ")
        val newAccess = synchronized(refreshLock) {
            val current = tokens.accessToken
            // 다른 요청이 그사이 재발급했으면 그 토큰을 쓴다
            if (current != null && current != usedToken) current else refreshBlocking()
        } ?: run {
            expireSession()
            return@Authenticator null
        }
        failedRequest.newBuilder().header("Authorization", "Bearer $newAccess").build()
    }

    private val client = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(20, TimeUnit.SECONDS)
        .addInterceptor(authHeader)
        .authenticator(refreshOn401)
        .apply {
            if (debugLogging) addInterceptor(HttpLoggingInterceptor().apply { level = HttpLoggingInterceptor.Level.BASIC })
        }
        .build()

    private val service: HawkeyeService = Retrofit.Builder()
        .baseUrl(normalizedBaseUrl)
        .client(client)
        .addConverterFactory(json.asConverterFactory("application/json".toMediaType()))
        .build()
        .create(HawkeyeService::class.java)

    // ── 인증 ──
    suspend fun signup(email: String, name: String, password: String, ageConfirmed: Boolean): ApiResult<SignupResponse> =
        call { service.signup(SignupRequest(email, name, password, ageConfirmed)) }.also { r ->
            r.getOrNull()?.let { saveTokens(it.accessToken, it.refreshToken) }
        }

    suspend fun login(email: String, password: String): ApiResult<TokenResponse> =
        call { service.login(LoginRequest(email, password)) }.also { r ->
            r.getOrNull()?.let { saveTokens(it.accessToken, it.refreshToken) }
        }

    /** 앱을 다시 켰을 때 저장된 refreshToken 으로 로그인 상태를 이어 간다 */
    suspend fun resumeSession(): ApiResult<TokenResponse> {
        val refresh = tokens.refreshToken ?: return ApiResult.Failure(401, "UNAUTHORIZED", "로그인이 필요합니다.")
        return call { service.refresh(RefreshTokenRequest(refresh)) }.also { r ->
            when (r) {
                is ApiResult.Success -> saveTokens(r.data.accessToken, r.data.refreshToken)
                is ApiResult.Failure -> if (r.status == 401) expireSession(notify = false)
            }
        }
    }

    suspend fun logout(): ApiResult<Unit> {
        val refresh = tokens.refreshToken
        val result = if (refresh != null) call { service.logout(RefreshTokenRequest(refresh)) } else ApiResult.Success(Unit)
        // 서버 응답과 상관없이 기기의 토큰은 지운다(accessToken 은 만료 전까지 유효하므로)
        expireSession(notify = false)
        return result
    }

    suspend fun withdraw(password: String): ApiResult<Unit> =
        call { service.withdraw(WithdrawRequest(password)) }.also { if (it is ApiResult.Success) expireSession(notify = false) }

    // ── 사용자·약관 ──
    suspend fun me() = call { service.me() }
    suspend fun terms() = call { service.terms() }
    suspend fun termsDetail(termsId: Long) = call { service.termsDetail(termsId) }
    suspend fun myAgreements() = call { service.myAgreements() }
    suspend fun agree(items: List<AgreeItem>) = call { service.agree(AgreeRequest(items)) }

    // ── 쉼터·주행·기록 ──
    suspend fun nearbyRestPlace(request: NearbyRequest) = call { service.nearbyRestPlace(request) }
    suspend fun startDrive(clientDriveId: String, startedAt: String) = call { service.startDrive(StartDriveRequest(clientDriveId, startedAt)) }
    suspend fun endDrive(driveId: Long, body: EndDriveRequest) = call { service.endDrive(driveId, body) }
    suspend fun driveReport(driveId: Long) = call { service.driveReport(driveId) }
    suspend fun drives(size: Int? = null, cursor: String? = null) = call { service.drives(size, cursor) }
    suspend fun calendar(month: String) = call { service.calendar(month) }
    suspend fun daily(date: String? = null) = call { service.daily(date) }
    suspend fun weekly(weekStart: String? = null) = call { service.weekly(weekStart) }

    private fun saveTokens(access: String, refresh: String) {
        tokens.save(access, refresh)
        _loggedIn.value = true
    }

    private fun expireSession(notify: Boolean = true) {
        tokens.clear()
        _loggedIn.value = false
        if (notify) _events.tryEmit(SessionEvent.LOGGED_OUT)
    }

    /** OkHttp 인증기 스레드에서 동기로 재발급한다. 실패하면 null. */
    private fun refreshBlocking(): String? {
        val refresh = tokens.refreshToken ?: return null
        val body = json.encodeToString(RefreshTokenRequest.serializer(), RefreshTokenRequest(refresh))
            .toRequestBody("application/json".toMediaType())
        val request = Request.Builder().url(normalizedBaseUrl + "api/auth/refresh").post(body).build()
        return try {
            bareClient.newCall(request).execute().use { res ->
                if (!res.isSuccessful) return null
                val token = json.decodeFromString(TokenResponse.serializer(), res.body?.string().orEmpty())
                tokens.save(token.accessToken, token.refreshToken)
                token.accessToken
            }
        } catch (_: Exception) {
            null
        }
    }

    private suspend fun <T> call(block: suspend () -> Response<T>): ApiResult<T> = try {
        val res = block()
        if (res.isSuccessful) {
            @Suppress("UNCHECKED_CAST")
            ApiResult.Success(res.body() ?: Unit as T)
        } else {
            val error = parseError(res.errorBody()?.string(), res.code())
            if (error.code == "TERMS_AGREEMENT_REQUIRED") _events.tryEmit(SessionEvent.TERMS_REQUIRED)
            error
        }
    } catch (e: IOException) {
        ApiResult.Failure(0, ApiResult.NETWORK, "서버에 연결할 수 없습니다. 네트워크 상태를 확인해 주세요.")
    } catch (e: Exception) {
        ApiResult.Failure(0, "CLIENT_ERROR", e.message ?: "요청을 처리하지 못했습니다.")
    }

    private fun parseError(raw: String?, status: Int): ApiResult.Failure {
        val body = raw?.let { runCatching { json.decodeFromString(ApiErrorBody.serializer(), it) }.getOrNull() }
        return ApiResult.Failure(
            status = status,
            code = body?.code ?: "HTTP_$status",
            message = body?.message ?: "요청을 처리하지 못했습니다. ($status)",
            fieldErrors = body?.fieldErrors.orEmpty().associate { it.field to it.message },
        )
    }

    companion object {
        val json = Json {
            ignoreUnknownKeys = true
            explicitNulls = false
            coerceInputValues = true
        }
    }
}
