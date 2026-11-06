package com.personal.driverguard.views.theme

import androidx.compose.ui.graphics.Color

/*
 * 원시(primitive) 색 토큰 — "블루아워 드라이브".
 * 새벽 고속도로의 블루아워 하늘에서 가져왔다. 이전 버전보다 하늘·면을 한두 단계 밝혔다(깊은 밤 → 해 뜨기 직전).
 * 앱에서 raw 색 값이 존재하는 유일한 파일이며, 화면은 Theme.kt 의 의미 토큰만 참조한다.
 * 대비 수치는 night1(#13223D) 바탕 기준.
 */
internal object Palette {
    // 하늘 — 위에서 지평선 쪽으로
    val night0 = Color(0xFF0F1C33)
    val night1 = Color(0xFF13223D)
    val dusk2 = Color(0xFF1B2D4F)
    val dusk3 = Color(0xFF24395F)
    val horizon = Color(0xFF3A5A8E)
    val haze = Color(0xFF4A6390)

    // 글자 — skyWhite 14.2:1, mist 7.8:1, mistDim 5.4:1 (night1 기준, dusk2 위에서는 12.3 / 6.7 / 4.7)
    val skyWhite = Color(0xFFEEF3FB)
    val mist = Color(0xFFA5B7D5)
    val mistDim = Color(0xFF8598B8)

    // 조작 강조(주요 동작) — 상태색과 섞지 않는다
    val ice = Color(0xFFAFCBFF)
    val iceDeep = Color(0xFF07101F)

    // 지평선 빛 — 상태 전용 예약색(앰버·레드는 '지금 행동' 상태에만)
    val glowClear = Color(0xFF5BC8B1)
    val glowDawn = Color(0xFFF3A65B)
    val glowSunset = Color(0xFFEC5A5F)
    val glowSlate = Color(0xFF8E9BB6)
    val glowRest = Color(0xFF9C9AF8)

    // 시연 표시 — 상태 예약색과 겹치지 않는 라벤더
    val lavender = Color(0xFFD3C4FF)

    val errorText = Color(0xFFFFA3A3)
    val white = Color(0xFFFFFFFF)
    val black = Color(0xFF000000)
}
