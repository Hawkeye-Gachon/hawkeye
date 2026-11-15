package com.personal.driverguard

import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onRoot
import com.github.takahirom.roborazzi.captureRoboImage
import com.personal.driverguard.ui.HawkeyeApp
import com.personal.driverguard.ui.Screen
import com.personal.driverguard.domain.platform.IVehicleNavigation
import com.personal.driverguard.domain.platform.NaviResult
import com.personal.driverguard.domain.platform.NaviRouteState
import com.personal.driverguard.domain.platform.RestStop
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.ParameterizedRobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode

/** Renders every screen at 1280×800dp so it can be compared with the design board. */
@RunWith(ParameterizedRobolectricTestRunner::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@Config(sdk = [34], qualifiers = "w1280dp-h800dp-land-mdpi")
class ScreensTest(private val screen: Screen) {
    @get:Rule val compose = createComposeRule()

    /** 차량 내비가 없는 환경(시안 흐름 그대로). */
    @Before fun noVehicleNavigation() {
        PlatformLocator.overrideVehicleNavigation(object : IVehicleNavigation {
            override suspend fun connect() = false
            override suspend fun routeState(): NaviRouteState? = null
            override suspend fun addRestStop(stop: RestStop): NaviResult = NaviResult.Failed("테스트")
            override suspend fun routeToRestStop(stop: RestStop): NaviResult = NaviResult.Failed("테스트")
            override fun release() = Unit
        })
        // Pleos 차량·음성 서비스가 없는 환경
        PlatformLocator.overrideCabinAndVoice(object : com.personal.driverguard.platform.VehicleCabin {
            override val status = kotlinx.coroutines.flow.MutableStateFlow(com.personal.driverguard.platform.CabinStatus())
            override suspend fun connect() = false
            override suspend fun alert(level: com.personal.driverguard.platform.CabinAlert) = emptyList<com.personal.driverguard.platform.CabinAction>()
            override suspend fun rest() = emptyList<com.personal.driverguard.platform.CabinAction>()
            override suspend fun restore() = Unit
            override fun release() = Unit
        }, object : com.personal.driverguard.platform.VoiceGuide {
            override suspend fun connect() = false
            override suspend fun speak(text: String) = Unit
            override suspend fun listen(timeoutMs: Long): String? = null
            override fun stop() = Unit
            override fun release() = Unit
        })
    }

    @Test fun render() {
        compose.mainClock.autoAdvance = false
        compose.setContent { HawkeyeApp(start = screen) }
        compose.mainClock.advanceTimeBy(600)
        compose.onRoot().captureRoboImage("build/screens/${"%02d".format(screen.ordinal + 1)}-${screen.name}.png")
    }

    companion object {
        @JvmStatic
        @ParameterizedRobolectricTestRunner.Parameters(name = "{0}")
        fun screens() = Screen.entries.map { arrayOf<Any>(it) }
    }
}
