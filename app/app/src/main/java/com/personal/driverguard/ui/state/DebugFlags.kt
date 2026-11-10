package com.personal.driverguard.ui.state

import com.personal.driverguard.BuildConfig

/**
 * UI 테스트용 임시 기능 스위치. debug 빌드에서만 켜지고 release 에는 적용되지 않는다.
 * 실제 서버 데이터로 확인하려면 해당 값을 false 로 바꾼다.
 */
object DebugFlags {
    /**
     * 클릭 진행 모드: 앱을 시안 확인 모드로 시작해, 서버 없이 버튼만 눌러도 다음 화면으로 넘어간다(화면 확인용).
     * 화면 값은 시안 예시 값이다. 실제 로그인·서버 데이터로 확인하려면 false 로 바꾼다.
     */
    val CLICK_THROUGH: Boolean = false

    /** 관찰 시작 3초 뒤 졸음 경고, 다시 3초 뒤 쉼터 경유지 추천을 자동으로 띄운다(AI 연동 전 화면 확인용). */
    val AUTO_WARNINGS: Boolean = BuildConfig.DEBUG

    /** 홈의 이전 기록·이전 기록 화면(일간·주간·달력·주행 리포트)에 이번 주와 지난주 가짜 기록을 보여 준다([FakeHistory]). */
    val FAKE_HISTORY: Boolean = BuildConfig.DEBUG

    /** 관찰 중 실제 GPS 대신 경부고속도로(양재→기흥)를 시속 100km로 달리는 모의 위치를 1초마다 찍는다([DummyRoute.drive]). */
    val DUMMY_ROUTE: Boolean = BuildConfig.DEBUG

    /** 쉼터 추천 시 위치가 없으면 더미 위치로 묻고, 서버가 쉼터를 못 찾으면 더미 쉼터를 쓴다([DummyRoute.restPlace]). */
    val DUMMY_REST_PLACE: Boolean = BuildConfig.DEBUG
}
