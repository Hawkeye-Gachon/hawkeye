package com.personal.driverguard

import android.content.Context
import com.personal.driverguard.domain.platform.IVehicleNavigation
import com.personal.driverguard.platform.AutoVoiceGuide
import com.personal.driverguard.platform.VehicleCabin
import com.personal.driverguard.platform.VoiceGuide
import com.personal.driverguard.platform.pleos.PleosFusedLocation
import com.personal.driverguard.platform.pleos.PleosVehicleCabin
import com.personal.driverguard.platform.pleos.PleosVehicleNavigation
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob

/** 차량·음성·위치 구현을 만들어 준다. 앱 전체에서 하나씩만 쓴다. */
object PlatformLocator {

    /** 화면 수명과 무관하게 끝까지 마쳐야 하는 작업(차량 실내 원복, 주행 기록 저장) */
    val appScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    @Volatile private var vehicleNavigation: IVehicleNavigation? = null
    @Volatile private var vehicleCabin: VehicleCabin? = null
    @Volatile private var voiceGuide: VoiceGuide? = null
    @Volatile private var fusedLocation: PleosFusedLocation? = null

    /** 차량 기본 내비 연동(Pleos NaviHelper) */
    fun provideVehicleNavigation(context: Context): IVehicleNavigation =
        vehicleNavigation ?: synchronized(this) {
            vehicleNavigation ?: PleosVehicleNavigation(context.applicationContext).also { vehicleNavigation = it }
        }

    /** 차량 실내 개입(Pleos Vehicle SDK: 공조·무드등·창문·시트)과 차량 상태 읽기 */
    fun provideVehicleCabin(context: Context): VehicleCabin =
        vehicleCabin ?: synchronized(this) {
            vehicleCabin ?: PleosVehicleCabin(context.applicationContext).also { vehicleCabin = it }
        }

    /** 음성 안내·응답(Pleos TTS·STT, 없으면 Android 기본 음성) */
    fun provideVoiceGuide(context: Context): VoiceGuide =
        voiceGuide ?: synchronized(this) {
            voiceGuide ?: AutoVoiceGuide(context.applicationContext).also { voiceGuide = it }
        }

    /** 차량 위치(Pleos FusedLocation) */
    fun provideFusedLocation(context: Context): PleosFusedLocation =
        fusedLocation ?: synchronized(this) {
            fusedLocation ?: PleosFusedLocation(context.applicationContext).also { fusedLocation = it }
        }

    /** 테스트에서 차량 내비 구현을 바꿔 끼운다(Robolectric 에서는 NaviHelper 서비스에 연결할 수 없다). */
    fun overrideVehicleNavigation(navigation: IVehicleNavigation) {
        vehicleNavigation = navigation
    }

    /** 테스트에서 차량 실내·음성 구현을 바꿔 끼운다(Robolectric 에는 Pleos 서비스가 없다). */
    fun overrideCabinAndVoice(cabin: VehicleCabin, voice: VoiceGuide) {
        vehicleCabin = cabin
        voiceGuide = voice
    }
}
