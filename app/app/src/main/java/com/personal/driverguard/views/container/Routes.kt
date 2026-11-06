package com.personal.driverguard.views.container

import kotlinx.serialization.Serializable

// Navigation Compose 타입 세이프 라우트
@Serializable
data object HomeRoute

/** 주행(실제 분석). 캘리브레이션을 마친 뒤에만 들어온다. */
@Serializable
data object DrivingRoute

/** 실제 분석 전 카메라 위치·기준 자세 확인 */
@Serializable
data object CalibrationRoute

@Serializable
data object TripListRoute

@Serializable
data class TripDetailRoute(val tripId: Long)

@Serializable
data object DevSettingsRoute
