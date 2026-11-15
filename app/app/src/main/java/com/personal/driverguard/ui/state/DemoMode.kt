package com.personal.driverguard.ui.state

import android.content.Context
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * 시연 모드. 실제 차량·서버 없이 시연 영상을 찍기 위한 모드로, 켜면 앱 전체가 가짜로 동작한다.
 *
 *  - 서버: 앱 안의 [DemoServer] 가 대신 응답한다(로그인·약관·쉼터 조회·주행 기록·이전 기록, 이번 주·지난주 예시 기록 포함).
 *  - 위치: 경부고속도로 서울(양재IC)→부산 방향을 시속 80km로 달리는 모의 주행([DriveSimulator]).
 *  - 쉼터: 진행 방향 앞의 실제 휴게소·졸음쉼터. 경유지로 추가하면 6초 뒤 도착한다.
 *  - 차량: 내비([DemoVehicleNavigation])와 실내 개입([com.personal.driverguard.platform.DemoVehicleCabin])은 반영된 것처럼 보여 준다.
 *  - 감지: 실제와 같이 카메라 영상을 AI 로 판정한다. 모의 주행이 시속 80km 라 저속 경고 억제에 걸리지 않는다.
 *    (카메라 카드를 길게 누르면 감지를 직접 넣는 시연 메뉴가 열린다.)
 *  - 음성 안내는 태블릿 스피커로 실제로 나온다(Android 기본 음성).
 *
 * 켜고 끄기는 로그인 화면과 홈 화면의 '시연 모드' 스위치로 하고, 앱을 다시 켜도 유지된다.
 */
object DemoMode {
    private const val PREFS = "demo_mode"
    private const val KEY_ON = "on"

    private val _on = MutableStateFlow(false)
    val on: StateFlow<Boolean> = _on.asStateFlow()
    val enabled: Boolean get() = _on.value

    @Volatile private var loaded = false

    fun init(context: Context) {
        if (loaded) return
        loaded = true
        _on.value = context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getBoolean(KEY_ON, false)
    }

    fun set(context: Context, value: Boolean) {
        context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putBoolean(KEY_ON, value).apply()
        _on.value = value
    }

    /** 테스트용 */
    fun setForTest(value: Boolean) {
        loaded = true
        _on.value = value
    }
}
