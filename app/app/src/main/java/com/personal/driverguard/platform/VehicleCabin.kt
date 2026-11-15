package com.personal.driverguard.platform

import kotlinx.coroutines.flow.StateFlow

/** 감지 단계별 차량 개입 수준 */
enum class CabinAlert {
    /** 전방 주시 이탈: 무드등 주황 */
    INATTENTION,

    /** 졸음 1회: 무드등 빨강, 에어컨 켜고 온도를 낮추고 바람을 세게, 내기 순환을 꺼 바깥 공기를 들인다 */
    DROWSY,

    /** 졸음 반복: 위에 더해 운전석 창문을 조금 열고 운전석 시트 통풍을 최대로 */
    DROWSY_REPEATED,
}

/** 차량이 실제로 한 조치 한 건(화면 표시용) */
data class CabinAction(val kind: Kind, val text: String) {
    enum class Kind { VOICE, AIR, LIGHT, WINDOW, SEAT }
}

/** 차량에서 읽은 상태. 읽지 못한 값은 null. */
data class CabinStatus(
    val connected: Boolean = false,
    /** 운전석 안전벨트 미착용 경고 */
    val seatBeltUnbuckled: Boolean? = null,
    /** 크루즈 컨트롤이 작동 중인지 */
    val cruiseActive: Boolean? = null,
    /** P / R / N / D */
    val gear: String? = null,
)

/**
 * 차량 실내 개입(공조·무드등·창문·시트)과 차량 상태 읽기.
 * 구현: Pleos Vehicle SDK([com.personal.driverguard.platform.pleos.PleosVehicleCabin]).
 * 권한이나 차량 지원이 없는 조작은 건너뛰고, 실제로 반영된 조치만 돌려준다.
 */
interface VehicleCabin {
    val status: StateFlow<CabinStatus>

    /** 차량 서비스에 연결한다. Pleos 차량이 아니면 false. */
    suspend fun connect(): Boolean

    /** [level] 에 맞게 차량을 조작하고, 반영된 조치를 돌려준다. 처음 조작 전 상태는 [restore] 를 위해 기억한다. */
    suspend fun alert(level: CabinAlert): List<CabinAction>

    /**
     * 쉼터에 도착해 쉴 때: 실내 온도를 [CabinPlan.REST_TEMP_C] 로 맞추고 바람을 약하게 하고, 주차 상태면 운전석 등받이를 눕힌다.
     * 반영된 조치를 돌려준다. 바꾸기 전 상태는 [restore] 를 위해 기억한다.
     */
    suspend fun rest(): List<CabinAction>

    /** 바꾼 것을 처음 상태로 돌린다(쉼터 도착·다시 출발·관찰 종료) */
    suspend fun restore()

    fun release()
}

/** 차량 개입에 쓰는 값 */
object CabinPlan {
    /** 졸음 때 낮추는 온도 */
    const val COOL_TEMP_C = 18f

    /** 졸음 반복 때 여는 운전석 창문 비율 */
    const val WINDOW_OPEN_PERCENT = 20

    /** 쉴 때 맞추는 실내 온도 */
    const val REST_TEMP_C = 23f
}
