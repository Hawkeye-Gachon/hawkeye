package com.personal.driverguard.views.theme

import android.provider.Settings
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.Surface
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.Immutable
import androidx.compose.runtime.remember
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.intl.LocaleList
import androidx.compose.ui.text.style.LineBreak
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.em
import androidx.compose.ui.unit.sp
import com.personal.driverguard.R
import com.personal.driverguard.entity.DisplayState

/*
 * 의미(semantic) 토큰 계층 — 블루아워 드라이브.
 * 콘텐츠층: 하늘 필드(SkyBackground)와 불투명 네이비 재질(surface*).
 * 조작층: 하늘 위에 뜨는 유리(GlassStyle). 화면은 이 파일의 토큰만 쓴다.
 * 다크 고정: 차량 실내(야간·터널·역광)에서 쓰는 장면이 기준이다. 이전보다 하늘을 밝게 잡았다.
 */

val Pretendard = FontFamily(
    Font(R.font.pretendard_regular, FontWeight.Normal),
    Font(R.font.pretendard_medium, FontWeight.Medium),
    Font(R.font.pretendard_semibold, FontWeight.SemiBold),
    Font(R.font.pretendard_bold, FontWeight.Bold),
)

private val BlueHourScheme = darkColorScheme(
    primary = Palette.ice,
    onPrimary = Palette.iceDeep,
    primaryContainer = Palette.horizon,
    onPrimaryContainer = Palette.skyWhite,
    secondary = Palette.mist,
    onSecondary = Palette.night1,
    // 지정하지 않으면 Material 기본 보라가 진행 막대 트랙·스낵바 등에 섞여 들어온다
    secondaryContainer = Palette.dusk3,
    onSecondaryContainer = Palette.skyWhite,
    tertiary = Palette.ice,
    onTertiary = Palette.iceDeep,
    tertiaryContainer = Palette.horizon,
    onTertiaryContainer = Palette.skyWhite,
    background = Palette.night1,
    onBackground = Palette.skyWhite,
    surface = Palette.dusk2,
    onSurface = Palette.skyWhite,
    surfaceVariant = Palette.dusk3,
    onSurfaceVariant = Palette.mist,
    surfaceContainerLowest = Palette.night0,
    surfaceContainerLow = Palette.night1,
    surfaceContainer = Palette.dusk2,
    surfaceContainerHigh = Palette.dusk3,
    surfaceContainerHighest = Palette.horizon,
    inverseSurface = Palette.skyWhite,
    inverseOnSurface = Palette.night1,
    inversePrimary = Palette.horizon,
    outline = Palette.mistDim,
    outlineVariant = Palette.haze,
    error = Palette.errorText,
    onError = Palette.night1,
)

/** 지평선 빛(상태)과 시연 표시 색. 상태는 언제나 문구·아이콘이 함께 전달하고, 색은 그 확인이다. */
@Immutable
data class StatusColors(
    val clear: Color,
    val dawn: Color,
    val sunset: Color,
    val slate: Color,
    val resting: Color,
    val demo: Color,
    val onDemo: Color,
) {
    fun glow(state: DisplayState): Color = when (state) {
        DisplayState.OBSERVING -> clear
        DisplayState.CAUTION -> dawn
        DisplayState.REST_RECOMMENDED -> sunset
        DisplayState.UNAVAILABLE, DisplayState.PREPARING -> slate
        DisplayState.PAUSED -> resting
    }
}

private val DefaultStatusColors = StatusColors(
    clear = Palette.glowClear,
    dawn = Palette.glowDawn,
    sunset = Palette.glowSunset,
    slate = Palette.glowSlate,
    resting = Palette.glowRest,
    demo = Palette.lavender,
    onDemo = Palette.night1,
)

/** 하늘 필드 색 단계 */
@Immutable
data class SkyColors(val zenith: Color, val mid: Color, val horizon: Color)

/** 조작층 유리. blur 를 못 쓰는 기기·고대비 설정에서는 fallback 으로 불투명하게 그린다. */
@Immutable
data class GlassStyle(
    val blur: Dp,
    val tint: Color,
    val highlight: Color,
    val edge: Color,
    /** 고대비 설정(투명도 줄이기)일 때의 완전 불투명 채움 */
    val fallback: Color,
    /** 블러를 못 쓰는 기기(API 30 이하)의 채움 */
    val translucent: Color,
    /** 유리 위 보조 버튼 채움 */
    val fill: Color,
)

/** 8dp 셀 간격 */
@Immutable
data class Spacing(
    val xxs: Dp = 4.dp,
    val xs: Dp = 8.dp,
    val s: Dp = 12.dp,
    val m: Dp = 16.dp,
    val l: Dp = 24.dp,
    val xl: Dp = 32.dp,
    val xxl: Dp = 48.dp,
)

@Immutable
data class Sizes(
    val touchMin: Dp = 48.dp,
    val hairline: Dp = 1.dp,
    /** 모든 버튼·분할 컨트롤·입력 칸·아이콘 버튼의 높이(하나만 쓴다) */
    val control: Dp = 56.dp,
    val glassBar: Dp = 64.dp,
    val iconS: Dp = 18.dp,
    val iconM: Dp = 24.dp,
    val iconL: Dp = 40.dp,
    val iconXl: Dp = 64.dp,
    val drivingRailWidth: Dp = 232.dp,
    val contentMaxWidth: Dp = 840.dp,
    val homeStartPaneMaxWidth: Dp = 520.dp,
    val wideBreakpoint: Dp = 600.dp,
)

/** 애플식 큰 반경. 캡슐은 RoundedCornerShape(50). */
private val SkyShapes = Shapes(
    extraSmall = RoundedCornerShape(8.dp),
    small = RoundedCornerShape(12.dp),
    medium = RoundedCornerShape(16.dp),
    large = RoundedCornerShape(22.dp),
    extraLarge = RoundedCornerShape(32.dp),
)

/**
 * 한국어 줄바꿈: 어절 단위(WordBreak.Phrase)로 끊어 낱말 중간에서 끊기지 않게 한다.
 * Android 13(API 33) 이상에서 동작하고, 그 아래 버전은 기존 방식으로 끊는다.
 */
private val KoreanLineBreak = LineBreak(LineBreak.Strategy.HighQuality, LineBreak.Strictness.Default, LineBreak.WordBreak.Phrase)

private fun style(size: Int, line: Int, weight: FontWeight, tracking: Double = 0.0) = TextStyle(
    fontFamily = Pretendard,
    fontSize = size.sp,
    lineHeight = line.sp,
    fontWeight = weight,
    letterSpacing = tracking.em,
    lineBreak = KoreanLineBreak,
    localeList = LocaleList("ko"),
)

/** Apple 텍스트 스타일(Large Title·Title·Headline·Body·Footnote)을 Material 역할에 대응 */
private val SkyTypography = Typography(
    displayLarge = style(56, 62, FontWeight.Bold, -0.025),
    displayMedium = style(46, 52, FontWeight.Bold, -0.025),
    displaySmall = style(38, 44, FontWeight.Bold, -0.02),
    headlineLarge = style(34, 41, FontWeight.Bold, -0.015),
    headlineMedium = style(28, 34, FontWeight.Bold, -0.01),
    headlineSmall = style(22, 28, FontWeight.SemiBold, -0.005),
    titleLarge = style(20, 25, FontWeight.SemiBold),
    titleMedium = style(17, 22, FontWeight.SemiBold),
    titleSmall = style(15, 20, FontWeight.SemiBold),
    bodyLarge = style(17, 24, FontWeight.Normal),
    bodyMedium = style(15, 21, FontWeight.Normal),
    bodySmall = style(13, 18, FontWeight.Normal),
    labelLarge = style(17, 22, FontWeight.SemiBold),
    labelMedium = style(13, 18, FontWeight.Medium),
    labelSmall = style(12, 16, FontWeight.Medium),
)

private val LocalStatusColors = staticCompositionLocalOf { DefaultStatusColors }
private val LocalSpacing = staticCompositionLocalOf { Spacing() }
private val LocalSizes = staticCompositionLocalOf { Sizes() }
private val LocalSky = staticCompositionLocalOf { SkyColors(Palette.night0, Palette.dusk3, Palette.horizon) }
private val LocalGlass = staticCompositionLocalOf {
    GlassStyle(28.dp, Palette.night1.copy(alpha = 0.40f), Palette.white.copy(alpha = 0.08f), Palette.white.copy(alpha = 0.30f), Palette.dusk2, Palette.dusk2.copy(alpha = 0.94f), Palette.white.copy(alpha = 0.13f))
}
private val LocalSkyDetails = staticCompositionLocalOf { SkyDetails(reduceTransparency = false, reduceMotion = false) }

/** 사용자 접근성 설정 — 투명도(고대비 글자 설정을 신호로 사용)와 애니메이션 */
@Immutable
data class SkyDetails(val reduceTransparency: Boolean, val reduceMotion: Boolean)

val MaterialTheme.status: StatusColors
    @Composable get() = LocalStatusColors.current
val MaterialTheme.spacing: Spacing
    @Composable get() = LocalSpacing.current
val MaterialTheme.sizes: Sizes
    @Composable get() = LocalSizes.current
val MaterialTheme.sky: SkyColors
    @Composable get() = LocalSky.current
val MaterialTheme.glass: GlassStyle
    @Composable get() = LocalGlass.current
val MaterialTheme.details: SkyDetails
    @Composable get() = LocalSkyDetails.current

/** 고정폭 숫자(시간·지속 시간 정렬용) */
val TabularNumbers = TextStyle(fontFeatureSettings = "tnum")

@Composable
fun DriverGuardTheme(content: @Composable () -> Unit) {
    val resolver = LocalContext.current.contentResolver
    val details = remember(resolver) {
        SkyDetails(
            // Android 에는 '투명도 줄이기'가 없어 '고대비 텍스트'를 같은 신호로 쓴다
            reduceTransparency = Settings.Secure.getInt(resolver, "high_text_contrast_enabled", 0) == 1,
            reduceMotion = Settings.Global.getFloat(resolver, Settings.Global.ANIMATOR_DURATION_SCALE, 1f) == 0f,
        )
    }
    CompositionLocalProvider(
        LocalStatusColors provides DefaultStatusColors,
        LocalSpacing provides Spacing(),
        LocalSizes provides Sizes(),
        LocalSkyDetails provides details,
    ) {
        MaterialTheme(colorScheme = BlueHourScheme, typography = SkyTypography, shapes = SkyShapes) {
            Surface(color = MaterialTheme.colorScheme.background, contentColor = MaterialTheme.colorScheme.onBackground, content = content)
        }
    }
}
