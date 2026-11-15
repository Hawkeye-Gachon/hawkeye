package com.personal.driverguard.platform.pleos

import kotlinx.coroutines.withContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.NonCancellable
import ai.pleos.playground.vehicle.constant.control.MoveControl
import ai.pleos.playground.vehicle.Vehicle
import ai.pleos.playground.vehicle.api.model.area.SeatArea
import ai.pleos.playground.vehicle.api.model.area.WindowArea
import ai.pleos.playground.vehicle.constant.area.SeatValue
import ai.pleos.playground.vehicle.constant.area.WindowValue
import ai.pleos.playground.vehicle.constant.control.CruiseControl
import ai.pleos.playground.vehicle.constant.control.LightSwitch
import ai.pleos.playground.vehicle.constant.control.VehicleGear
import ai.pleos.playground.vehicle.constant.unit.TemperatureUnit
import ai.pleos.playground.vehicle.listener.CruiseControlStateListener
import ai.pleos.playground.vehicle.listener.GearSelectionListener
import ai.pleos.playground.vehicle.listener.SeatBeltNotBuckledWarningListener
import android.content.Context
import android.util.Log
import com.personal.driverguard.platform.CabinAction
import com.personal.driverguard.platform.CabinAlert
import com.personal.driverguard.platform.CabinStatus
import com.personal.driverguard.platform.CabinPlan
import com.personal.driverguard.platform.VehicleCabin
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withTimeoutOrNull
import kotlin.coroutines.resume
import kotlin.math.roundToInt

/**
 * Pleos Vehicle SDK 로 졸음 단계별 실내 개입을 한다.
 *
 *  - 공조(HVAC, CONTROL_CAR_CLIMATE): 전원·에어컨 켜기, 온도 18°C, 바람 최대, 내기 순환 끄기(바깥 공기)
 *  - 무드등(Light, CONTROL_CAR_LIGHTS): 주황(부주의)·빨강(졸음). 주행 모드에 따른 자동 색은 잠시 끈다.
 *  - 창문(Window, CONTROL_CAR_WINDOWS): 운전석 창문 20% 열기(졸음 반복)
 *  - 시트(Seat, CONTROL_CAR_SEATS): 운전석 통풍 최대(졸음 반복)
 *  - 읽기: 안전벨트 미착용 경고(CAR_SAFETY), 크루즈 컨트롤(CAR_DRIVE), 기어(CAR_DRIVE)
 *
 * SDK 는 모든 결과를 onComplete/onFailed 콜백으로 준다. 여기서는 호출 하나를 콜백이 올 때까지(시간 제한) 기다리는 식으로 쓴다.
 * 권한이 없거나 차량이 지원하지 않으면 onFailed 가 오고, 그 조작은 건너뛴다(조치 목록에서 빠진다).
 * 처음 조작하기 전의 값을 읽어 두었다가 [restore] 에서 되돌린다(읽지 못한 값은 건드리지 않는다).
 *
 * 문서로 확인하지 못한 점(차량에서 확인 필요): 무드등 색 값 형식(여기서는 ARGB 정수), 창문 위치 값의 범위(능력 조회로 최대값을 받아 비율로 계산).
 */
class PleosVehicleCabin(context: Context) : VehicleCabin {

    private val appContext = context.applicationContext
    private val lock = Mutex()

    @Volatile private var vehicle: Vehicle? = null
    @Volatile private var unavailable = false

    private val _status = MutableStateFlow(CabinStatus())
    override val status: StateFlow<CabinStatus> = _status.asStateFlow()

    /** 처음 조작 전 값. null 이면 읽지 못했거나 아직 조작하지 않음. */
    private class Saved {
        var taken = false
        var hvacPower: Boolean? = null
        var ac: Boolean? = null
        var temperature: Float? = null
        var fan: Int? = null
        var recirculation: Boolean? = null
        var ambientSwitch: LightSwitch? = null
        var ambientColor: Int? = null
        var ambientAuto: Boolean? = null
        var window: Int? = null
        var seatVentilation: Int? = null
        /** 쉴 때 등받이를 뒤로 움직인 시간(ms). 되돌릴 때 같은 시간만큼 앞으로 움직인다. */
        var reclinedMs = 0L
    }

    private var saved = Saved()

    private val gearListener = object : GearSelectionListener {
        override fun onSelected(gear: VehicleGear?) = _status.update { it.copy(gear = gear?.label()) }
        override fun onFailed(e: Exception) = Unit
    }
    private val seatBeltListener = object : SeatBeltNotBuckledWarningListener {
        override fun onSeatBeltNotBuckledWarningUpdated(warning: Boolean?) = _status.update { it.copy(seatBeltUnbuckled = warning) }
        override fun onFailed(e: Exception) = Unit
    }
    private val cruiseListener = object : CruiseControlStateListener {
        override fun onCruiseControlStateUpdated(state: CruiseControl?) = _status.update { it.copy(cruiseActive = state == CruiseControl.ACTIVATED) }
        override fun onFailed(e: Exception) = Unit
    }

    override suspend fun connect(): Boolean = lock.withLock {
        if (vehicle != null) return@withLock true
        if (unavailable) return@withLock false
        val v = try {
            Vehicle(appContext).also { it.initialize() }
        } catch (e: Throwable) {
            Log.w(TAG, "Vehicle SDK 를 쓸 수 없습니다", e)
            unavailable = true
            return@withLock false
        }
        // 서비스 연결은 비동기라 기어를 한 번 읽어 실제로 응답하는지 확인한다
        var gear: VehicleGear? = null
        repeat(CONNECT_TRIES) {
            gear = call<VehicleGear> { ok, fail -> v.getGear().getCurrentGearSelectionState(ok, fail) }
            if (gear != null) return@repeat
        }
        if (gear == null) {
            Log.w(TAG, "Vehicle 서비스가 응답하지 않습니다(Pleos 차량이 아님)")
            runCatching { v.release() }
            unavailable = true
            return@withLock false
        }
        vehicle = v
        val belt = call<Boolean> { ok, fail -> v.getSafety().hasSeatBeltNotBuckledWarning(DRIVER, ok, fail) }
        val cruise = call<CruiseControl> { ok, fail -> v.getDrivingMode().getCruiseControlState(ok, fail) }
        _status.value = CabinStatus(connected = true, seatBeltUnbuckled = belt, cruiseActive = cruise?.let { it == CruiseControl.ACTIVATED }, gear = gear?.label())
        runCatching {
            v.getGear().registerGearSelection(gearListener)
            v.getSafety().registerSeatBeltNotBuckledWarning(DRIVER, seatBeltListener)
            v.getDrivingMode().registerCruiseControlState(cruiseListener)
        }.onFailure { Log.w(TAG, "차량 상태 구독 실패", it) }
        true
    }

    override suspend fun alert(level: CabinAlert): List<CabinAction> {
        if (!connect()) return emptyList()
        val v = vehicle ?: return emptyList()
        return lock.withLock {
            saveOnce(v)
            val done = mutableListOf<CabinAction>()
            val red = level != CabinAlert.INATTENTION
            if (setAmbient(v, if (red) RED else AMBER)) done += CabinAction(CabinAction.Kind.LIGHT, if (red) "무드등 빨강" else "무드등 주황")
            if (red) {
                val parts = mutableListOf<String>()
                set { ok, fail -> v.getHVAC().setHvacPower(DRIVER, true, { _, _ -> ok(Unit) }, fail) }
                if (set { ok, fail -> v.getHVAC().setHvacAc(DRIVER, true, { _, _ -> ok(Unit) }, fail) } and
                    set { ok, fail -> v.getHVAC().setHvacTemperature(DRIVER, CabinPlan.COOL_TEMP_C, TemperatureUnit.CELSIUS, { _, _ -> ok(Unit) }, fail) }
                ) parts += "에어컨 ${CabinPlan.COOL_TEMP_C.toInt()}°C"
                if (set { ok, fail -> v.getHVAC().setHvacRecirculation(DRIVER, false, { _, _ -> ok(Unit) }, fail) }) parts += "바깥 공기"
                val fanMax = call<Map<SeatArea, ai.pleos.playground.vehicle.api.model.capability.AreaBounds<Int>>> { ok, fail ->
                    v.getHVAC().checkHvacFanSpeedCapabilities(ok, fail)
                }?.get(DRIVER)?.maxValue ?: DEFAULT_FAN_MAX
                if (set { ok, fail -> v.getHVAC().setHvacFanSpeed(DRIVER, fanMax, { _, _ -> ok(Unit) }, fail) }) parts += "바람 세게"
                if (parts.isNotEmpty()) done += CabinAction(CabinAction.Kind.AIR, parts.joinToString(", "))
            }
            if (level == CabinAlert.DROWSY_REPEATED) {
                val windowMax = call<Map<WindowArea, ai.pleos.playground.vehicle.api.model.capability.AreaBounds<Int>>> { ok, fail ->
                    v.getWindow().checkWindowPositionCapabilities(ok, fail)
                }?.get(DRIVER_WINDOW)?.maxValue ?: 100
                val open = (windowMax * CabinPlan.WINDOW_OPEN_PERCENT / 100.0).roundToInt()
                if (set { ok, fail -> v.getWindow().setWindowPosition(DRIVER_WINDOW, open, { _, _ -> ok(Unit) }, fail) }) {
                    done += CabinAction(CabinAction.Kind.WINDOW, "운전석 창문 ${CabinPlan.WINDOW_OPEN_PERCENT}% 열림")
                }
                val ventMax = call<Map<SeatArea, ai.pleos.playground.vehicle.api.model.capability.AreaBounds<Int>>> { ok, fail ->
                    v.getSeat().checkSeatVentilationCapabilities(ok, fail)
                }?.get(DRIVER)?.maxValue ?: DEFAULT_VENT_MAX
                if (set { ok, fail -> v.getSeat().setHvacSeatVentilation(DRIVER, ventMax, { _, _ -> ok(Unit) }, fail) }) {
                    done += CabinAction(CabinAction.Kind.SEAT, "운전석 시트 통풍 최대")
                }
            }
            done
        }
    }

    /**
     * 쉴 때의 실내: 온도 [CabinPlan.REST_TEMP_C], 바람 약하게. 기어가 P 일 때만 운전석 등받이를 눕힌다
     * (주행 중에 시트가 움직이면 위험하다).
     *
     * 등받이는 '이 각도로'가 아니라 '이 방향으로 움직여라'로만 조작할 수 있다(moveSeatBackrestAngle + MoveControl).
     * 그래서 뒤로 움직이기 시작하고 [RECLINE_MS] 뒤에 멈춘다. 되돌릴 때는 같은 시간만큼 앞으로 움직인다.
     * 문서로 확인하지 못한 점(차량에서 확인 필요): BACKWARD 가 눕히는 방향인지, 움직이는 속도.
     */
    override suspend fun rest(): List<CabinAction> {
        if (!connect()) return emptyList()
        val v = vehicle ?: return emptyList()
        return lock.withLock {
            saveOnce(v)
            val done = mutableListOf<CabinAction>()
            val parts = mutableListOf<String>()
            set { ok, fail -> v.getHVAC().setHvacPower(DRIVER, true, { _, _ -> ok(Unit) }, fail) }
            if (set { ok, fail -> v.getHVAC().setHvacTemperature(DRIVER, CabinPlan.REST_TEMP_C, TemperatureUnit.CELSIUS, { _, _ -> ok(Unit) }, fail) }) {
                parts += "실내 ${CabinPlan.REST_TEMP_C.toInt()}°C"
            }
            if (set { ok, fail -> v.getHVAC().setHvacFanSpeed(DRIVER, REST_FAN, { _, _ -> ok(Unit) }, fail) }) parts += "바람 약하게"
            if (parts.isNotEmpty()) done += CabinAction(CabinAction.Kind.AIR, parts.joinToString(", "))
            if (_status.value.gear == "P" && moveBackrest(v, MoveControl.BACKWARD, RECLINE_MS)) {
                saved.reclinedMs = RECLINE_MS
                done += CabinAction(CabinAction.Kind.SEAT, "운전석 등받이 눕힘")
            }
            done
        }
    }

    /** 등받이를 [direction] 으로 [ms] 동안 움직이고 멈춘다. 움직이기 시작했으면 true(멈춤 명령은 실패해도 한 번 더 보낸다). */
    private suspend fun moveBackrest(v: Vehicle, direction: MoveControl, ms: Long): Boolean {
        if (!set { ok, fail -> v.getSeat().moveSeatBackrestAngle(DRIVER, direction, { _, _ -> ok(Unit) }, fail) }) return false
        try {
            delay(ms)
        } finally {
            withContext(NonCancellable) {
                if (!set { ok, fail -> v.getSeat().moveSeatBackrestAngle(DRIVER, MoveControl.STOP, { _, _ -> ok(Unit) }, fail) }) {
                    set { ok, fail -> v.getSeat().moveSeatBackrestAngle(DRIVER, MoveControl.STOP, { _, _ -> ok(Unit) }, fail) }
                }
            }
        }
        return true
    }

    override suspend fun restore() {
        val v = vehicle ?: return
        lock.withLock {
            val s = saved
            if (!s.taken) return@withLock
            // 눕힌 등받이는 주차 상태일 때만 세운다(그렇지 않으면 운전자가 직접 세운다)
            if (s.reclinedMs > 0 && _status.value.gear == "P") moveBackrest(v, MoveControl.FORWARD, s.reclinedMs)
            s.window?.let { w -> set { ok, fail -> v.getWindow().setWindowPosition(DRIVER_WINDOW, w, { _, _ -> ok(Unit) }, fail) } }
            s.seatVentilation?.let { x -> set { ok, fail -> v.getSeat().setHvacSeatVentilation(DRIVER, x, { _, _ -> ok(Unit) }, fail) } }
            s.recirculation?.let { x -> set { ok, fail -> v.getHVAC().setHvacRecirculation(DRIVER, x, { _, _ -> ok(Unit) }, fail) } }
            s.fan?.let { x -> set { ok, fail -> v.getHVAC().setHvacFanSpeed(DRIVER, x, { _, _ -> ok(Unit) }, fail) } }
            s.temperature?.let { x -> set { ok, fail -> v.getHVAC().setHvacTemperature(DRIVER, x, TemperatureUnit.CELSIUS, { _, _ -> ok(Unit) }, fail) } }
            s.ac?.let { x -> set { ok, fail -> v.getHVAC().setHvacAc(DRIVER, x, { _, _ -> ok(Unit) }, fail) } }
            s.hvacPower?.let { x -> set { ok, fail -> v.getHVAC().setHvacPower(DRIVER, x, { _, _ -> ok(Unit) }, fail) } }
            s.ambientColor?.let { x -> set { ok, fail -> v.getLight().setAmbientLightBrightnessColor(DRIVER, x, { _, _ -> ok(Unit) }, fail) } }
            s.ambientSwitch?.let { x -> set { ok, fail -> v.getLight().setAmbientLightSwitch(DRIVER, x, { _, _ -> ok(Unit) }, fail) } }
            s.ambientAuto?.let { x -> set { ok, fail -> v.getLight().setAmbientColorAutoByDriveModeEnabled(x, { ok(Unit) }, fail) } }
            saved = Saved()
        }
    }

    override fun release() {
        val v = vehicle ?: return
        runCatching {
            v.getGear().unregisterGearSelection(gearListener)
            v.getSafety().unregisterSeatBeltNotBuckledWarning(DRIVER, seatBeltListener)
            v.getDrivingMode().unregisterCruiseControlState(cruiseListener)
            v.release()
        }
        vehicle = null
    }

    private suspend fun saveOnce(v: Vehicle) {
        if (saved.taken) return
        val s = Saved().apply { taken = true }
        s.hvacPower = call { ok, fail -> v.getHVAC().getHvacPowerState(DRIVER, ok, fail) }
        s.ac = call { ok, fail -> v.getHVAC().getHvacAcState(DRIVER, ok, fail) }
        s.temperature = call { ok, fail -> v.getHVAC().getHvacTemperature(DRIVER, TemperatureUnit.CELSIUS, ok, fail) }
        s.fan = call { ok, fail -> v.getHVAC().getHvacFanSpeed(DRIVER, ok, fail) }
        s.recirculation = call { ok, fail -> v.getHVAC().getHvacRecirculationState(DRIVER, ok, fail) }
        s.ambientSwitch = call { ok, fail -> v.getLight().getAmbientLightSwitch(DRIVER, ok, fail) }
        s.ambientColor = call { ok, fail -> v.getLight().getAmbientLightBrightnessColor(DRIVER, ok, fail) }
        s.ambientAuto = call { ok, fail -> v.getLight().getAmbientColorAutoByDriveModeEnabled(ok, fail) }
        s.window = call { ok, fail -> v.getWindow().getCurrentWindowPosition(DRIVER_WINDOW, ok, fail) }
        s.seatVentilation = call { ok, fail -> v.getSeat().getHvacSeatVentilationState(DRIVER, ok, fail) }
        saved = s
    }

    private suspend fun setAmbient(v: Vehicle, color: Int): Boolean {
        set { ok, fail -> v.getLight().setAmbientColorAutoByDriveModeEnabled(false, { ok(Unit) }, fail) }
        set { ok, fail -> v.getLight().setAmbientLightSwitch(DRIVER, LightSwitch.ON, { _, _ -> ok(Unit) }, fail) }
        return set { ok, fail -> v.getLight().setAmbientLightBrightnessColor(DRIVER, color, { _, _ -> ok(Unit) }, fail) }
    }

    /** 조작 하나. 성공 콜백이 오면 true. */
    private suspend fun set(block: (ok: (Unit?) -> Unit, fail: (Exception) -> Unit) -> Unit): Boolean = call<Unit>(block) != null

    /** 콜백 API 하나를 기다린다. 실패·시간 초과·예외면 null. */
    private suspend fun <T> call(block: (ok: (T?) -> Unit, fail: (Exception) -> Unit) -> Unit): T? =
        withTimeoutOrNull(CALL_TIMEOUT_MS) {
            suspendCancellableCoroutine<T?> { cont ->
                try {
                    block({ if (cont.isActive) cont.resume(it) }, { e ->
                        Log.w(TAG, "차량 호출 실패: ${e.message}")
                        if (cont.isActive) cont.resume(null)
                    })
                } catch (e: Throwable) {
                    Log.w(TAG, "차량 호출 예외", e)
                    if (cont.isActive) cont.resume(null)
                }
            }
        }

    private fun VehicleGear.label(): String? = when (this) {
        VehicleGear.GEAR_PARK -> "P"
        VehicleGear.GEAR_REVERSE -> "R"
        VehicleGear.GEAR_NEUTRAL -> "N"
        VehicleGear.GEAR_UNKNOWN -> null
        else -> "D"
    }

    private companion object {
        const val TAG = "PleosVehicleCabin"
        const val CALL_TIMEOUT_MS = 2_000L
        const val CONNECT_TRIES = 3
        const val DEFAULT_FAN_MAX = 6
        const val DEFAULT_VENT_MAX = 3
        /** 쉴 때의 바람 세기 */
        const val REST_FAN = 1
        /** 등받이를 눕히느라 움직이는 시간 */
        const val RECLINE_MS = 4_000L
        val DRIVER = SeatArea(SeatValue.ROW_1_LEFT)
        val DRIVER_WINDOW = WindowArea(WindowValue.ROW_1_LEFT)
        val RED = 0xFFFF2E24.toInt()
        val AMBER = 0xFFFF9F0A.toInt()
    }
}
