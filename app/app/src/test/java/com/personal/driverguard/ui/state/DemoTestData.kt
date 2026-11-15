package com.personal.driverguard.ui.state

import java.io.File

/** 시연용 경부고속도로 경로·쉼터(앱 assets 파일을 그대로 읽는다. 단위 테스트의 작업 폴더는 app 모듈) */
object DemoTestData {
    val route: DemoRoute by lazy {
        DemoRoute.parse(File("src/main/assets/demo/gyeongbu_busan.json").readText(), File("src/main/assets/demo/gyeongbu_busan_rest.json").readText())
    }
}
