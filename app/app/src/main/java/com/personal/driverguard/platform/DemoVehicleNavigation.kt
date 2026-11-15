package com.personal.driverguard.platform

import com.personal.driverguard.domain.platform.IVehicleNavigation
import com.personal.driverguard.domain.platform.NaviResult
import com.personal.driverguard.domain.platform.NaviRouteState
import com.personal.driverguard.domain.platform.RestStop
import kotlinx.coroutines.delay

/** 시연 모드의 차량 내비: 부산까지 안내 중이고, 쉼터를 넣으면 반영된 것으로 한다(실제 내비 조작 없음). */
class DemoVehicleNavigation : IVehicleNavigation {
    private val waypoints = mutableListOf<String>()

    override suspend fun connect(): Boolean = true

    override suspend fun routeState(): NaviRouteState = synchronized(waypoints) { NaviRouteState(routing = true, destinationName = DESTINATION, waypointNames = waypoints.toList()) }

    override suspend fun addRestStop(stop: RestStop): NaviResult {
        delay(RESPONSE_MS)
        synchronized(waypoints) {
            if (waypoints.size >= NaviRouteState.MAX_WAYPOINTS) return NaviResult.Full
            waypoints.add(0, stop.name)
        }
        return NaviResult.Added
    }

    override suspend fun routeToRestStop(stop: RestStop): NaviResult {
        delay(RESPONSE_MS)
        return NaviResult.Added
    }

    /** 쉼터에 들렀거나 주행을 끝내면 비운다 */
    fun clear(name: String? = null) = synchronized(waypoints) { if (name == null) waypoints.clear() else waypoints.remove(name) }

    override fun release() = Unit

    companion object {
        const val DESTINATION = "부산"
        private const val RESPONSE_MS = 500L
    }
}
