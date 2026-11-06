package com.personal.driverguard

import android.content.Context
import com.personal.driverguard.domain.platform.IAlertOutput
import com.personal.driverguard.domain.platform.ILocationSource
import com.personal.driverguard.domain.platform.IRestAreaGuide
import com.personal.driverguard.domain.platform.TimeSource
import com.personal.driverguard.platform.android.AndroidAlertOutput
import com.personal.driverguard.platform.android.AndroidLocationSource
import com.personal.driverguard.platform.android.AndroidRestAreaGuide
import com.personal.driverguard.platform.android.AndroidTimeSource
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

    /** 주행 화면마다 새로 만들고 화면이 끝나면 release 한다. */
    fun createAlertOutput(context: Context): IAlertOutput = AndroidAlertOutput(context)
}
