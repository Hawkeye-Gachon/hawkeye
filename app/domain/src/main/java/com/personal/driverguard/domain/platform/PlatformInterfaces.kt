package com.personal.driverguard.domain.platform

import com.personal.driverguard.entity.AlertKind
import com.personal.driverguard.entity.LocationFix
import kotlinx.coroutines.flow.Flow

/*
 * 플랫폼 연동 지점.
 * 현재 구현은 일반 Android(app 모듈의 platform.android 패키지)뿐이다.
 * Pleos Connect 전용 SDK 가 제공되면 같은 인터페이스로 구현체를 추가하고 PlatformLocator 에서 교체한다.
 * 카메라 입력은 CameraX 바인딩이 화면 생명주기에 묶여 있어 app 모듈의 camera 패키지에 있다.
 */

/** 선택적 위치 수집(LocationProvider). 권한이 없으면 [locationUpdates] 는 아무것도 내보내지 않고 끝난다. */
interface ILocationSource {
    fun hasPermission(): Boolean

    fun locationUpdates(): Flow<LocationFix>
}

/**
 * 경고음·음성 출력(AlertController). 구현체는 경고가 겹쳐 재생되지 않게 해야 한다(AlertArbiter 참고).
 * 반복 제한은 AlertPolicy 가 앞에서 처리하므로 구현체는 받은 경고를 재생만 한다.
 */
interface IAlertOutput {
    /** TTS 를 쓸 수 있는지. 초기화에 실패하면 경고음만 낸다. */
    val voiceAvailable: Boolean

    fun play(kind: AlertKind)

    /** 재생 중·대기 중인 경고를 모두 멈춘다(주행 종료·휴식 시작). */
    fun cancelAll()

    fun release()
}

/**
 * 휴식 장소 안내. 지도 SDK/API 가 없으므로 앱이 장소를 추천하지 않는다.
 * 일반 Android 구현은 설치된 지도 앱에 '휴게소' 검색을 넘기는 것까지만 한다.
 */
interface IRestAreaGuide {
    fun isAvailable(): Boolean

    /** 외부 앱으로 검색을 열었으면 true */
    fun openRestAreaSearch(): Boolean
}

/** 휴식 장소(졸음쉼터·휴게소). 지금은 앱에 둔 예시 값이고, 백엔드가 현재 위치·진행 방향으로 골라 줄 예정이다. */
data class RestStop(
    val name: String,
    /** 예: "졸음쉼터", "휴게소" */
    val kind: String,
    val latitude: Double,
    val longitude: Double,
    val address: String? = null,
)

/** 차량 내비의 현재 안내 상태 */
data class NaviRouteState(
    /** 목적지 안내 중인지 */
    val routing: Boolean,
    val destinationName: String?,
    /** 이미 들어 있는 경유지 이름(순서대로) */
    val waypointNames: List<String>,
) {
    val waypointsFull: Boolean get() = waypointNames.size >= MAX_WAYPOINTS

    companion object {
        /** 차량 내비가 받는 경유지 최대 개수(Pleos NaviHelper 기준) */
        const val MAX_WAYPOINTS = 2
    }
}

/** 쉼터 추가 요청 결과. 시안의 경유지 제안 화면 상태와 1:1 로 대응한다. */
sealed interface NaviResult {
    /** 차량 내비가 반영을 확인함 */
    data object Added : NaviResult

    /** 안내 중인 목적지가 없음 → '쉼터로 길 안내'를 제안한다 */
    data object NoDestination : NaviResult

    /** 경유지가 이미 가득 참 */
    data object Full : NaviResult

    data class Failed(val reason: String) : NaviResult
}

/**
 * 차량 기본 내비 연동(NavigationController). 앱은 들를 쉼터를 정해 전달만 하고, 지도와 길 안내는 차량 내비가 맡는다.
 * 구현: Pleos Connect NaviHelper. Pleos 차량·에뮬레이터가 아니면 [connect] 가 false 를 돌려준다.
 */
interface IVehicleNavigation {
    /** 차량 내비 서비스에 연결한다. 이미 연결됐으면 바로 true. */
    suspend fun connect(): Boolean

    /** 현재 안내 상태. 연결되지 않았거나 응답이 없으면 null. */
    suspend fun routeState(): NaviRouteState?

    /** 안내 중인 경로에 쉼터를 경유지로 넣는다. 기존 경유지가 있으면 쉼터를 먼저 들르도록 넣는다. */
    suspend fun addRestStop(stop: RestStop): NaviResult

    /** 안내 중인 목적지가 없을 때 쉼터를 목적지로 길 안내를 시작한다. */
    suspend fun routeToRestStop(stop: RestStop): NaviResult

    fun release()
}
