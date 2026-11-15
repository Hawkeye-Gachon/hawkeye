package com.personal.driverguard.ui.state

import com.personal.driverguard.BuildConfig

/**
 * 개발용 스위치. 시연 영상용 가짜 동작은 [DemoMode](앱 안 스위치)로 켜고 끈다. 여기 값은 코드에서만 바꾼다.
 */
object DebugFlags {
    /**
     * 클릭 진행 모드: 앱을 시안 확인 모드로 시작해, 서버 없이 버튼만 눌러도 다음 화면으로 넘어간다(화면 확인용).
     * 화면 값은 시안 예시 값이다.
     */
    val CLICK_THROUGH: Boolean = false

    /**
     * 서버 시험용 모의 위치(debug 빌드): 시연 모드가 아닐 때도 실제 GPS 대신 경부고속도로를 시속 100km로 달리는 위치를
     * 1초마다 실제 서버로 올린다. 추가한 쉼터에는 들러 20초 정차한다([DriveSimulator]).
     * 실제 차량에서 debug 빌드로 시험할 때는 false 로 바꿔야 차량 위치를 쓴다.
     */
    val DUMMY_ROUTE: Boolean = BuildConfig.DEBUG
}
