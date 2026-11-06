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
