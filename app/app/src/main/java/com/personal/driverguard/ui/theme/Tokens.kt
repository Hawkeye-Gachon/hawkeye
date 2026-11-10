package com.personal.driverguard.ui.theme

import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.em
import androidx.compose.ui.unit.sp
import com.personal.driverguard.R

/** Generated from handoff/tokens.json. Do not edit by hand. */
object NColor {
    /** 모든 화면의 바탕(보드 바깥 여백 포함). 2026-11 요청으로 검은색에서 흰색으로 바꿨다. */
    val canvas = Color(0xFFFFFFFF)
    val frost = Color(0xFFF6F6F9)
    val frost2 = Color(0xFFECEDF3)
    val white = Color(0xFFFFFFFF)
    val ink = Color(0xFF171717)
    val ink2 = Color(0xFF4A4C55)
    val ink3 = Color(0xFF6B7482)
    val ink4 = Color(0xFFA4A8AE)
    val line = Color(0x1A131417)
    val line2 = Color(0x33131417)
    val fill = Color(0x0D131417)
    val fill2 = Color(0x1A131417)
    val onNight = Color(0xFFFFFFFF)
    val onNight2 = Color(0xD6FFFFFF)
    val onNight3 = Color(0xA3FFFFFF)
    /** 주 강조색. 2026-11 Hawkeye 시안에 맞춰 보라(#5A46FA)에서 로고 남색으로 바꿨다(이름은 코드 호환을 위해 유지). */
    val indigo = Color(0xFF0B2A5C)
    val indigoHi = Color(0xFF3D7EF5)
    /** Hawkeye 로고 남색. 로그인·회원가입의 주 버튼에 쓴다(시안 02 로그인, 05 회원가입). */
    val navy = Color(0xFF0B2A5C)
    val navyPressed = Color(0xFF08204A)
    val alert = Color(0xFFFF8A00)
    val danger = Color(0xFFF62E24)
    val dangerInk = Color(0xFFD9190F)
    val alarm = Color(0xFFE3241B)
    val camBg = Color(0xFF0A0B0D)
    val mapBg = Color(0xFFEEF0F6)
}

val AstaSans = FontFamily(
    Font(R.font.asta_light, FontWeight.Light),
    Font(R.font.asta_regular, FontWeight.Normal),
    Font(R.font.asta_medium, FontWeight.Medium),
    Font(R.font.asta_semibold, FontWeight.SemiBold),
)

object NType {
    val clock = TextStyle(fontFamily = AstaSans, fontWeight = FontWeight(300), fontSize = 76.sp, lineHeight = 80.sp, letterSpacing = (-0.04).em)
    val alarm = TextStyle(fontFamily = AstaSans, fontWeight = FontWeight(600), fontSize = 44.sp, lineHeight = 54.sp, letterSpacing = (-0.03).em)
    val headline = TextStyle(fontFamily = AstaSans, fontWeight = FontWeight(300), fontSize = 40.sp, lineHeight = 52.sp, letterSpacing = (-0.025).em)
    val metric = TextStyle(fontFamily = AstaSans, fontWeight = FontWeight(300), fontSize = 38.sp, lineHeight = 46.sp, letterSpacing = (-0.03).em)
    val count = TextStyle(fontFamily = AstaSans, fontWeight = FontWeight(300), fontSize = 34.sp, lineHeight = 42.sp, letterSpacing = (-0.02).em)
    val title = TextStyle(fontFamily = AstaSans, fontWeight = FontWeight(500), fontSize = 30.sp, lineHeight = 40.sp, letterSpacing = (-0.02).em)
    val subtitle = TextStyle(fontFamily = AstaSans, fontWeight = FontWeight(600), fontSize = 22.sp, lineHeight = 30.sp, letterSpacing = (-0.01).em)
    val button = TextStyle(fontFamily = AstaSans, fontWeight = FontWeight(600), fontSize = 22.sp, lineHeight = 28.sp, letterSpacing = (-0.01).em)
    val peak = TextStyle(fontFamily = AstaSans, fontWeight = FontWeight(600), fontSize = 20.sp, lineHeight = 28.sp, letterSpacing = (-0.01).em)
    val body = TextStyle(fontFamily = AstaSans, fontWeight = FontWeight(400), fontSize = 18.sp, lineHeight = 27.sp, letterSpacing = (0).em)
    val row = TextStyle(fontFamily = AstaSans, fontWeight = FontWeight(400), fontSize = 17.sp, lineHeight = 24.sp, letterSpacing = (0).em)
    val small = TextStyle(fontFamily = AstaSans, fontWeight = FontWeight(400), fontSize = 16.sp, lineHeight = 23.sp, letterSpacing = (0).em)
    val label = TextStyle(fontFamily = AstaSans, fontWeight = FontWeight(400), fontSize = 15.sp, lineHeight = 20.sp, letterSpacing = (0).em)
    val caption = TextStyle(fontFamily = AstaSans, fontWeight = FontWeight(400), fontSize = 14.sp, lineHeight = 20.sp, letterSpacing = (0).em)
}

object NDimen {
    val screenW = 1280.dp
    val screenH = 800.dp
    val margin = 24.dp
    val gutter = 24.dp
    val sceneW = 688.dp
    val sceneH = 752.dp
    val sheetW = 520.dp
    val sheetH = 752.dp
    val cardFullW = 1232.dp
    val cardRadius = 32.dp
    val sheetPadding = 40.dp
    val sheetPicH = 250.dp
    val sheetPicSmH = 200.dp
    val buttonH = 72.dp
    val buttonRadius = 24.dp
    val buttonHHero = 88.dp
    val buttonRadiusHero = 28.dp
    val buttonSmH = 52.dp
    val buttonSmRadius = 16.dp
    val fieldH = 60.dp
    val fieldRadius = 16.dp
    val check = 30.dp
    val checkRadius = 10.dp
    val segH = 48.dp
    val segSmH = 40.dp
    val segRadius = 20.dp
    val icon = 24.dp
    const val iconStroke = 1.5f
    val gridBg = 48.dp
    val edgeLine = 3.dp
    val edgeBlur1 = 70.dp
    val edgeBlur2 = 220.dp
}
