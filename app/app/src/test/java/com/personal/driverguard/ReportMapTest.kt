package com.personal.driverguard

import androidx.compose.foundation.layout.Box
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onRoot
import com.github.takahirom.roborazzi.captureRoboImage
import com.personal.driverguard.ui.design.LiveRouteMap
import com.personal.driverguard.ui.design.RoutePin
import com.personal.driverguard.ui.design.RouteStop
import com.personal.driverguard.ui.theme.NColor
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode
import kotlin.math.cos
import kotlin.math.sin

/**
 * 주행 리포트의 경로 지도를 그려 이미지로 남긴다(build/screens/report-map-*.png).
 * 경로는 테스트용으로 만든 굽은 길이다(실제 도로가 아니다).
 */
@RunWith(RobolectricTestRunner::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@Config(sdk = [34], qualifiers = "w1280dp-h800dp-land-mdpi")
class ReportMapTest {
    @get:Rule val compose = createComposeRule()

    /** 남남동쪽으로 [meters] 를 가는 굽은 길. 25m 마다 한 점. */
    private fun road(meters: Int): List<List<Double>> = (0..meters / 25).map { i ->
        val s = i * 25.0
        val side = 420 * sin(s / 860) + 110 * sin(s / 400 + 1.3)
        val north = -s * cos(.35) + side * sin(.35)
        val east = s * sin(.35) + side * cos(.35)
        listOf(37.40 + north / 111_320, 127.10 + east / (111_320 * cos(Math.toRadians(37.40))))
    }

    private fun pin(road: List<List<Double>>, at: Int, drowsy: Boolean, label: String) =
        road[at / 25].let { RoutePin(it[0], it[1], if (drowsy) NColor.danger else NColor.alert, label) }

    /** 감지가 출발 직후에 몰려 있고, 쉼터에 들렀다가 조금 더 간 주행 */
    @Test fun driveWithARestStop() {
        val road = road(8_700)
        val pins = listOf(pin(road, 250, false, "오후 5:26"), pin(road, 475, true, "오후 5:26"), pin(road, 1_300, true, "오후 5:27"))
        val stop = road[7_500 / 25].let { RouteStop(it[0] + .0006, it[1] + .0008, "죽전휴게소", visited = true) }
        compose.setContent { Box { LiveRouteMap(road, pins, "출발 오후 5:26", "도착 오후 5:39", listOf(stop)) } }
        compose.onRoot().captureRoboImage("build/screens/report-map-rest.png")
    }

    /** 쉼터 없이 끝낸 짧은 주행(약 1분, 1.3km) */
    @Test fun shortDrive() {
        val road = road(1_300)
        compose.setContent { Box { LiveRouteMap(road, listOf(pin(road, 400, false, "오후 5:26")), "출발 오후 5:26", "도착 오후 5:27") } }
        compose.onRoot().captureRoboImage("build/screens/report-map-short.png")
    }

    /** 위치 기록이 없는 주행 */
    @Test fun noLocation() {
        compose.setContent { Box { LiveRouteMap(emptyList(), emptyList(), "출발", "도착") } }
        compose.onRoot().captureRoboImage("build/screens/report-map-empty.png")
    }
}
