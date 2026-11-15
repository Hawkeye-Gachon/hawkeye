package com.personal.driverguard

import android.content.Context
import com.personal.driverguard.domain.platform.IAlertOutput
import com.personal.driverguard.domain.platform.ILocationSource
import com.personal.driverguard.domain.platform.IRestAreaGuide
import com.personal.driverguard.domain.platform.IVehicleNavigation
import com.personal.driverguard.domain.platform.TimeSource
import com.personal.driverguard.platform.android.AndroidAlertOutput
import com.personal.driverguard.platform.android.AndroidLocationSource
import com.personal.driverguard.platform.android.AndroidRestAreaGuide
import com.personal.driverguard.platform.android.AndroidTimeSource
import com.personal.driverguard.platform.pleos.PleosVehicleNavigation
import com.personal.driverguard.runtime.AnalysisStatsHolder
import com.personal.driverguard.runtime.CalibrationStore
import com.personal.driverguard.runtime.MetricLog
import com.personal.driverguard.runtime.ModelAssetHolder
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob

/**
 * 플랫폼 구현과 분석 런타임을 제공한다.
 * Pleos Connect 전용 구현이 생기면 이곳에서 일반 Android 구현과 교체한다.
 */
object PlatformLocator {

    /** 화면 수명과 무관하게 끝까지 마쳐야 하는 작업(여정 마무리 저장 등) */
    val appScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    val timeSource: TimeSource = AndroidTimeSource
    val analysisStats = AnalysisStatsHolder()
    val calibrationStore = CalibrationStore()

    @Volatile
    private var modelAssets: ModelAssetHolder? = null

    @Volatile
    private var metricLog: MetricLog? = null

    fun provideModelAssets(context: Context): ModelAssetHolder =
        modelAssets ?: synchronized(this) {
            modelAssets ?: ModelAssetHolder(context.applicationContext).also { modelAssets = it }
        }

    fun provideMetricLog(context: Context): MetricLog =
        metricLog ?: synchronized(this) {
            metricLog ?: MetricLog(context.applicationContext).also { metricLog = it }
        }

    fun provideLocationSource(context: Context): ILocationSource = AndroidLocationSource(context.applicationContext)

    fun provideRestAreaGuide(context: Context): IRestAreaGuide = AndroidRestAreaGuide(context.applicationContext)

    @Volatile
    private var vehicleNavigation: IVehicleNavigation? = null

    /** 차량 기본 내비 연동(Pleos NaviHelper). 앱 전체에서 하나만 쓴다. */
    fun provideVehicleNavigation(context: Context): IVehicleNavigation =
        vehicleNavigation ?: synchronized(this) {
            vehicleNavigation ?: PleosVehicleNavigation(context.applicationContext).also { vehicleNavigation = it }
        }

    /** 테스트에서 차량 내비 구현을 바꿔 끼운다(Robolectric 에서는 NaviHelper 서비스 연결이 가짜라 실제 구현을 쓸 수 없다). */
    fun overrideVehicleNavigation(navigation: IVehicleNavigation) {
        vehicleNavigation = navigation
    }

    @Volatile private var vehicleCabin: com.personal.driverguard.platform.VehicleCabin? = null
    @Volatile private var voiceGuide: com.personal.driverguard.platform.VoiceGuide? = null
    @Volatile private var fusedLocation: com.personal.driverguard.platform.pleos.PleosFusedLocation? = null

    /** 차량 실내 개입(Pleos Vehicle SDK: 공조·무드등·창문·시트)과 차량 상태 읽기 */
    fun provideVehicleCabin(context: Context): com.personal.driverguard.platform.VehicleCabin =
        vehicleCabin ?: synchronized(this) {
            vehicleCabin ?: com.personal.driverguard.platform.pleos.PleosVehicleCabin(context.applicationContext).also { vehicleCabin = it }
        }

    /** 음성 안내·응답(Pleos TTS·STT, 없으면 Android 기본 음성) */
    fun provideVoiceGuide(context: Context): com.personal.driverguard.platform.VoiceGuide =
        voiceGuide ?: synchronized(this) {
            voiceGuide ?: com.personal.driverguard.platform.AutoVoiceGuide(context.applicationContext).also { voiceGuide = it }
        }

    /** 차량 위치(Pleos FusedLocation) */
    fun provideFusedLocation(context: Context): com.personal.driverguard.platform.pleos.PleosFusedLocation =
        fusedLocation ?: synchronized(this) {
            fusedLocation ?: com.personal.driverguard.platform.pleos.PleosFusedLocation(context.applicationContext).also { fusedLocation = it }
        }

    /** 테스트에서 차량 실내·음성 구현을 바꿔 끼운다(Robolectric 에는 Pleos 서비스가 없다) */
    fun overrideCabinAndVoice(cabin: com.personal.driverguard.platform.VehicleCabin, voice: com.personal.driverguard.platform.VoiceGuide) {
        vehicleCabin = cabin
        voiceGuide = voice
    }

    /** 주행 화면마다 새로 만들고 화면이 끝나면 release 한다. */
    fun createAlertOutput(context: Context): IAlertOutput = AndroidAlertOutput(context)
}
