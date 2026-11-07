package com.personal.driverguard.platform.pleos

import ai.pleos.playground.navi.constants.NaviErrorCode
import ai.pleos.playground.navi.constants.RouteOption
import ai.pleos.playground.navi.constants.WaypointCmdType
import ai.pleos.playground.navi.constants.WaypointIndex
import ai.pleos.playground.navi.data.DestinationInfo
import ai.pleos.playground.navi.data.RequestWaypointInfo
import ai.pleos.playground.navi.data.RouteInfo
import ai.pleos.playground.navi.data.RouteStartInfo
import ai.pleos.playground.navi.data.RouteStateInfo
import ai.pleos.playground.navi.data.RouteWithWaypointsRequest
import ai.pleos.playground.navi.data.WaypointChangedInfo
import ai.pleos.playground.navi.data.WaypointInfo
import ai.pleos.playground.navi.helper.NaviHelper
import ai.pleos.playground.navi.helper.listener.NaviHelperEventListener
import android.content.Context
import android.util.Log
import com.personal.driverguard.domain.platform.IVehicleNavigation
import com.personal.driverguard.domain.platform.NaviResult
import com.personal.driverguard.domain.platform.NaviRouteState
import com.personal.driverguard.domain.platform.RestStop
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withTimeoutOrNull
import kotlin.coroutines.resume

/**
 * Pleos Connect NaviHelper 로 차량 기본 내비에 쉼터를 넣는다.
 *
 * NaviHelper 는 요청 메서드가 값을 돌려주지 않고 [NaviHelperEventListener] 콜백으로 결과를 준다.
 * 여기서는 요청 하나를 보내고 대응하는 콜백(또는 onError)을 기다리는 식으로 순서대로 처리한다([lock]).
 * 응답이 없으면 시간 제한 뒤 실패로 본다.
 *
 * 문서로 확인하지 못한 동작(에뮬레이터에서 확인 필요):
 *  - addWaypoint 가 진행 중인 안내에 바로 반영되는지(재탐색이 따로 필요한지)
 *  - 경유지가 가득 찼을 때 전용 오류가 오는지(없어서 요청 전에 개수를 먼저 확인한다)
 */
class PleosVehicleNavigation(context: Context) : IVehicleNavigation {

    private val appContext = context.applicationContext

    // 만드는 것부터 실패할 수 있어(Pleos 가 아닌 기기) connect() 의 try 안에서 처음 만든다
    private val helper by lazy { NaviHelper(appContext) }
    private val lock = Mutex()

    @Volatile
    private var connected = false

    /** 한 번 연결에 실패하면 이번 실행에서는 다시 시도하지 않는다(일반 폰·에뮬레이터에서 매번 기다리지 않도록) */
    @Volatile
    private var unavailable = false

    // 기다리는 응답. 콜백 스레드에서 채운다.
    @Volatile private var routeStateReply: CompletableDeferred<RouteStateInfo>? = null
    @Volatile private var destinationReply: CompletableDeferred<DestinationInfo>? = null
    @Volatile private var waypointReply: CompletableDeferred<List<WaypointInfo>>? = null

    /** 경로를 바꾸는 요청(경유지 추가·길 안내)의 결과 */
    @Volatile private var changeReply: CompletableDeferred<NaviResult>? = null

    private val listener = object : NaviHelperEventListener {
        override fun onRouteStateInfo(routeStateInfo: RouteStateInfo) {
            routeStateReply?.complete(routeStateInfo)
        }

        override fun onDestinationInfo(destinationInfo: DestinationInfo) {
            destinationReply?.complete(destinationInfo)
        }

        override fun onWayPointInfo(waypointInfo: List<WaypointInfo>) {
            waypointReply?.complete(waypointInfo)
        }

        override fun onWaypointChanged(waypointChangedInfo: WaypointChangedInfo) {
            if (waypointChangedInfo.cmdType == WaypointCmdType.ADD) changeReply?.complete(NaviResult.Added)
        }

        override fun onRouteStarted(routeStartInfo: RouteStartInfo) {
            changeReply?.complete(NaviResult.Added)
        }

        override fun onError(errorCode: NaviErrorCode) {
            Log.w(TAG, "NaviHelper 오류: $errorCode")
            changeReply?.complete(NaviResult.Failed(errorCode.name))
        }
    }

    override suspend fun connect(): Boolean {
        if (connected) return true
        if (unavailable) return false
        return lock.withLock {
            if (connected) return@withLock true
            if (unavailable) return@withLock false
            val ok = withTimeoutOrNull(CONNECT_TIMEOUT_MS) {
                suspendCancellableCoroutine { cont ->
                    try {
                        helper.initialize(
                            { if (cont.isActive) cont.resume(true) },
                            { e ->
                                Log.w(TAG, "NaviHelper 연결 실패", e)
                                if (cont.isActive) cont.resume(false)
                            },
                        )
                    } catch (e: Exception) {
                        // Pleos 차량·에뮬레이터가 아니면 내비 서비스가 없다
                        Log.w(TAG, "NaviHelper 를 쓸 수 없습니다", e)
                        if (cont.isActive) cont.resume(false)
                    }
                }
            } ?: false
            if (ok) helper.addListener(listener) else unavailable = true
            connected = ok
            ok
        }
    }

    override suspend fun routeState(): NaviRouteState? {
        if (!connect()) return null
        return lock.withLock { queryRouteState() }
    }

    override suspend fun addRestStop(stop: RestStop): NaviResult {
        if (!connect()) return NaviResult.Failed("차량 내비에 연결할 수 없습니다")
        return lock.withLock {
            val state = queryRouteState() ?: return@withLock NaviResult.Failed("차량 내비가 응답하지 않습니다")
            when {
                !state.routing -> NaviResult.NoDestination
                state.waypointsFull -> NaviResult.Full
                state.waypointNames.isEmpty() -> awaitChange { helper.addWaypoint(stop.toWaypoint(WaypointIndex.FIRST)) }
                else -> {
                    // 기존 경유지가 있으면 쉼터를 먼저 들르도록 목적지·경유지를 다시 요청한다: 현재 위치 → 쉼터 → 기존 경유지 → 목적지
                    val destination = query(::destinationReply) { helper.getDestinationInfo() }
                        ?: return@withLock NaviResult.Failed("목적지 정보를 받지 못했습니다")
                    val existing = query(::waypointReply) { helper.getWaypointInfo() }.orEmpty()
                    val waypoints = listOf(stop.toWaypoint(WaypointIndex.FIRST)) +
                        existing.take(1).map { it.toRequest(WaypointIndex.SECOND) }
                    awaitChange { helper.requestRouteWithWaypoints(RouteWithWaypointsRequest(destination.toRoute(), waypoints)) }
                }
            }
        }
    }

    override suspend fun routeToRestStop(stop: RestStop): NaviResult {
        if (!connect()) return NaviResult.Failed("차량 내비에 연결할 수 없습니다")
        return lock.withLock { awaitChange { helper.requestRoute(stop.toRoute()) } }
    }

    override fun release() {
        if (!connected) return
        connected = false
        runCatching {
            helper.removeListener(listener)
            helper.release()
        }
    }

    /** 안내 상태·목적지 이름·경유지 목록을 차례로 묻는다. [lock] 안에서 부른다. */
    private suspend fun queryRouteState(): NaviRouteState? {
        val state = query(::routeStateReply) { helper.getRouteStateInfo() } ?: return null
        if (!state.isRouting) return NaviRouteState(routing = false, destinationName = null, waypointNames = emptyList())
        val destination = query(::destinationReply) { helper.getDestinationInfo() }
        val waypoints = query(::waypointReply) { helper.getWaypointInfo() }.orEmpty()
        return NaviRouteState(routing = true, destinationName = destination?.name, waypointNames = waypoints.map { it.name })
    }

    private suspend fun <T> query(slot: kotlin.reflect.KMutableProperty0<CompletableDeferred<T>?>, request: () -> Unit): T? {
        val reply = CompletableDeferred<T>()
        slot.set(reply)
        return try {
            request()
            withTimeoutOrNull(QUERY_TIMEOUT_MS) { reply.await() }
        } catch (e: Exception) {
            Log.w(TAG, "NaviHelper 조회 실패", e)
            null
        } finally {
            slot.set(null)
        }
    }

    private suspend fun awaitChange(request: () -> Unit): NaviResult {
        val reply = CompletableDeferred<NaviResult>()
        changeReply = reply
        return try {
            request()
            withTimeoutOrNull(CHANGE_TIMEOUT_MS) { reply.await() } ?: NaviResult.Failed("차량 내비가 응답하지 않습니다")
        } catch (e: Exception) {
            Log.w(TAG, "NaviHelper 요청 실패", e)
            NaviResult.Failed(e.message ?: e.javaClass.simpleName)
        } finally {
            changeReply = null
        }
    }

    private fun RestStop.toWaypoint(index: WaypointIndex) = RequestWaypointInfo(
        latitude = latitude,
        longitude = longitude,
        poiName = name,
        waypointIndex = index,
        poiId = "",
        address = address ?: "",
        poiSubId = "0",
        poiProvider = "",
    )

    private fun RestStop.toRoute() = RouteInfo(
        latitude = latitude,
        longitude = longitude,
        displayLatitude = null,
        displayLongitude = null,
        poiName = name,
        poiId = "",
        address = address ?: "",
        poiSubId = "0",
        routeOption = RouteOption.RECOMMENDED,
        poiProvider = "",
        isMobileRequest = false,
    )

    private fun WaypointInfo.toRequest(index: WaypointIndex) = RequestWaypointInfo(
        latitude = latitude,
        longitude = longitude,
        poiName = name,
        waypointIndex = index,
        poiId = poiId,
        address = address ?: "",
        poiSubId = poiSubId ?: "0",
        poiProvider = poiProvider,
    )

    private fun DestinationInfo.toRoute() = RouteInfo(
        latitude = latitude,
        longitude = longitude,
        displayLatitude = displayLatitude,
        displayLongitude = displayLongitude,
        poiName = name,
        poiId = poiId,
        address = address ?: "",
        poiSubId = poiSubId ?: "0",
        routeOption = RouteOption.RECOMMENDED,
        poiProvider = poiProvider,
        isMobileRequest = false,
    )

    private companion object {
        const val TAG = "PleosNavi"
        const val CONNECT_TIMEOUT_MS = 3_000L
        const val QUERY_TIMEOUT_MS = 3_000L
        const val CHANGE_TIMEOUT_MS = 10_000L
    }
}
