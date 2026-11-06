package com.personal.driverguard.views.components

import android.os.Build
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.snap
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.WindowInsetsSides
import androidx.compose.foundation.layout.asPaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.only
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.ArrowBack
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.compositionLocalOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.BlendMode
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.CompositingStrategy
import androidx.compose.ui.graphics.Shape
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.scale
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.selected
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.LayoutDirection
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import com.personal.driverguard.R
import com.personal.driverguard.entity.AnalysisMode
import com.personal.driverguard.views.theme.TabularNumbers
import com.personal.driverguard.views.theme.details
import com.personal.driverguard.views.theme.glass
import com.personal.driverguard.views.theme.sizes
import com.personal.driverguard.views.theme.sky
import com.personal.driverguard.views.theme.spacing
import com.personal.driverguard.views.theme.status
import dev.chrisbanes.haze.HazeProgressive
import dev.chrisbanes.haze.HazeState
import dev.chrisbanes.haze.HazeStyle
import dev.chrisbanes.haze.HazeTint
import dev.chrisbanes.haze.hazeEffect
import dev.chrisbanes.haze.hazeSource
import dev.chrisbanes.haze.rememberHazeState

/* ------------------------------------------------------------------ 하늘 필드(콘텐츠층) */

/** 화면 높이 대비 지평선 위치 */
const val HORIZON_FRACTION = 0.68f

/**
 * 추상 블루아워 하늘과 지평선 빛. 빛의 색과 세기가 운전자 상태를 말한다(시그니처).
 * 상태가 바뀌면 빛이 지평선에서 천천히 차오른다. 애니메이션 끄기 설정이면 즉시 바뀐다.
 * 사진을 쓰지 않으므로 카메라 화면·글자와 경쟁하지 않는다.
 */
@Composable
fun SkyBackground(
    glow: Color,
    intensity: Float,
    modifier: Modifier = Modifier,
    dim: Float = 0f,
) {
    val reduceMotion = MaterialTheme.details.reduceMotion
    val sky = MaterialTheme.sky
    val color by animateColorAsState(glow, if (reduceMotion) snap() else tween(GLOW_RISE_MS), label = "glow")
    val strength by animateFloatAsState(intensity, if (reduceMotion) snap() else tween(GLOW_RISE_MS), label = "glowStrength")
    Canvas(modifier.fillMaxSize()) {
        val horizonY = size.height * HORIZON_FRACTION
        // 하늘: 천정 → 지평선(블루아워) → 땅(다시 어둡게)
        drawRect(
            Brush.verticalGradient(
                0f to sky.zenith,
                HORIZON_FRACTION * 0.55f to sky.mid,
                HORIZON_FRACTION to sky.horizon,
                HORIZON_FRACTION + 0.02f to sky.mid,
                1f to sky.zenith,
            ),
        )
        // 지평선의 얇은 밝은 띠
        drawRect(
            Brush.verticalGradient(
                0f to Color.Transparent,
                0.5f to sky.horizon.copy(alpha = 0.9f),
                1f to Color.Transparent,
                startY = horizonY - size.height * 0.05f,
                endY = horizonY + size.height * 0.015f,
            ),
            topLeft = Offset(0f, horizonY - size.height * 0.05f),
            size = androidx.compose.ui.geometry.Size(size.width, size.height * 0.065f),
        )
        val center = Offset(size.width / 2f, horizonY)
        val radius = size.width * 0.75f
        scale(scaleX = 1f, scaleY = 0.42f, pivot = center) {
            drawCircle(
                brush = Brush.radialGradient(
                    0f to color.copy(alpha = 0.85f * strength),
                    0.35f to color.copy(alpha = 0.38f * strength),
                    1f to Color.Transparent,
                    center = center,
                    radius = radius,
                ),
                radius = radius,
                center = center,
            )
        }
        if (dim > 0f) drawRect(Color.Black.copy(alpha = dim))
    }
}

private const val GLOW_RISE_MS = 700

/* ------------------------------------------------------------------ 유리(조작층) */

val LocalHaze = compositionLocalOf<HazeState?> { null }

/** 유리 컨테이너 안인지. 안쪽 보조 버튼은 유리를 겹치지 않고 옅은 채움만 쓴다. */
private val LocalInGlass = compositionLocalOf { false }

/**
 * 유리 가장자리(스페큘러 림). 위쪽은 빛을 받아 밝고, 아래쪽은 희미하게 다시 반사된다.
 */
@Composable
fun glassRim(strength: Float = 1f): Brush {
    val g = MaterialTheme.glass
    return Brush.verticalGradient(
        0f to g.edge.copy(alpha = g.edge.alpha * strength),
        0.45f to g.highlight.copy(alpha = 0.03f * strength),
        1f to g.highlight.copy(alpha = 0.10f * strength),
    )
}

/**
 * Liquid Glass 근사. 조작·탐색 요소(떠 있는 기능층)에만 쓴다. 콘텐츠 카드·목록·카메라 영상에는 쓰지 않는다.
 * 블러: Android 12 이상에서 실제 배경 블러, 그 아래는 반투명 대체, 고대비(투명도 줄이기 대용)에서는 불투명.
 */
@Composable
fun GlassSurface(
    modifier: Modifier = Modifier,
    shape: Shape = MaterialTheme.shapes.large,
    content: @Composable BoxScope.() -> Unit,
) {
    val glass = MaterialTheme.glass
    val haze = LocalHaze.current
    val opaque = haze == null || MaterialTheme.details.reduceTransparency
    val base = modifier.clip(shape)
    val surfaced = if (opaque) {
        base.background(glass.fallback)
    } else {
        base.hazeEffect(
            state = haze,
            style = HazeStyle(
                backgroundColor = glass.fallback,
                tints = listOf(HazeTint(glass.tint)),
                blurRadius = glass.blur,
                noiseFactor = 0.04f,
                fallbackTint = HazeTint(glass.translucent),
            ),
        )
    }
    Box(
        surfaced
            // 위쪽 안쪽 광택: 유리 윗면이 빛을 받는 느낌
            .background(Brush.verticalGradient(0f to glass.highlight.copy(alpha = 0.10f), 0.5f to Color.Transparent))
            .border(MaterialTheme.sizes.hairline, glassRim(), shape),
    ) {
        CompositionLocalProvider(LocalInGlass provides true) { content() }
    }
}

/**
 * 하늘 위에 콘텐츠를 올리고 유리 상단 바·하단 바를 띄우는 화면 틀.
 * 콘텐츠는 바 아래로 스크롤되어 들어가고, 바 뒤의 스크롤 가장자리 효과가 그 콘텐츠를 흐리게 한다.
 */
@Composable
fun SkyScaffold(
    /** 없으면 상태 표시줄 높이의 스크롤 가장자리 띠를 둬서 스크롤된 내용이 상태 표시줄 아이콘과 겹치지 않게 한다 */
    topBar: @Composable () -> Unit = { StatusBarEdge() },
    bottomBar: @Composable () -> Unit = {},
    glow: Color = MaterialTheme.status.clear,
    glowIntensity: Float = 0.35f,
    /** 글이 많은 화면은 하늘을 이만큼 가라앉힌다. 밝아진 하늘에 맞춰 이전(0.45)보다 낮췄다. */
    dim: Float = 0.2f,
    content: @Composable (PaddingValues) -> Unit,
) {
    val haze = rememberHazeState()
    val density = LocalDensity.current
    var topPx by remember { mutableIntStateOf(0) }
    var bottomPx by remember { mutableIntStateOf(0) }
    CompositionLocalProvider(LocalHaze provides haze) {
        Box(Modifier.fillMaxSize()) {
            Box(Modifier.fillMaxSize().hazeSource(haze)) {
                SkyBackground(glow, glowIntensity, dim = dim)
                val insets = WindowInsets.safeDrawing.asPaddingValues()
                content(
                    PaddingValues(
                        top = with(density) { topPx.toDp() },
                        bottom = maxOf(with(density) { bottomPx.toDp() }, insets.calculateBottomPadding()),
                        start = insets.calculateLeftPadding(LayoutDirection.Ltr),
                        end = insets.calculateRightPadding(LayoutDirection.Ltr),
                    ),
                )
            }
            Box(Modifier.align(Alignment.TopCenter).fillMaxWidth().onSizeChanged { topPx = it.height }) { topBar() }
            Box(Modifier.align(Alignment.BottomCenter).fillMaxWidth().onSizeChanged { bottomPx = it.height }) { bottomBar() }
        }
    }
}

/** 상태 표시줄 뒤의 스크롤 가장자리 띠 */
@Composable
fun StatusBarEdge() {
    Box(Modifier.fillMaxWidth().scrollEdge(fromTop = true).windowInsetsPadding(WindowInsets.safeDrawing.only(WindowInsetsSides.Top)))
}

/** 바 여백 + 추가 여백. 스크롤 안쪽 여백(contentPadding)으로 써서 내용이 바 아래로 흘러 들어가게 한다. */
fun PaddingValues.plus(extra: Dp): PaddingValues = PaddingValues(
    start = calculateLeftPadding(LayoutDirection.Ltr) + extra,
    top = calculateTopPadding() + extra,
    end = calculateRightPadding(LayoutDirection.Ltr) + extra,
    bottom = calculateBottomPadding() + extra,
)

/**
 * 스크롤 가장자리 효과. 바 아래로 스크롤된 콘텐츠를 점점 흐리고 어둡게 해 떠 있는 조작부와 구분한다.
 * 장식이 아니며 단색 판을 깔지 않는다.
 */
@Composable
fun Modifier.scrollEdge(fromTop: Boolean): Modifier {
    val haze = LocalHaze.current
    val sky = MaterialTheme.sky
    // 실제 블러(API 31+)가 있으면 흐림이 구분을 맡아 옅게 덮고, 없으면 덮개가 그 역할을 대신해 거의 불투명하게 깐다
    val canBlur = haze != null && !MaterialTheme.details.reduceTransparency && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S
    val mid = if (canBlur) 0.55f else 0.9f
    val edge = if (canBlur) 0.85f else 0.97f
    val fade = if (fromTop) {
        Brush.verticalGradient(0f to sky.zenith.copy(alpha = edge), 0.75f to sky.zenith.copy(alpha = mid), 1f to Color.Transparent)
    } else {
        Brush.verticalGradient(0f to Color.Transparent, 0.25f to sky.zenith.copy(alpha = mid), 1f to sky.zenith.copy(alpha = edge))
    }
    val blurred = if (canBlur) {
        hazeEffect(state = haze) {
            blurRadius = SCROLL_EDGE_BLUR
            progressive = HazeProgressive.verticalGradient(startIntensity = if (fromTop) 1f else 0f, endIntensity = if (fromTop) 0f else 1f)
        }
    } else {
        this
    }
    return blurred.background(fade)
}

private val SCROLL_EDGE_BLUR = 20.dp

/**
 * 떠 있는 상단 바: 유리 원형 뒤로 버튼, 가운데 제목, 오른쪽 동작. 바 자체는 판이 아니라 스크롤 가장자리 효과다.
 * [below] 는 바 아래 줄(분할 컨트롤 등). 유리 캡슐에 올린다.
 */
@Composable
fun GlassTopBar(
    title: String,
    onBack: (() -> Unit)?,
    actions: @Composable RowScope.() -> Unit = {},
    below: (@Composable BoxScope.() -> Unit)? = null,
) {
    Column(
        Modifier
            .fillMaxWidth()
            .scrollEdge(fromTop = true)
            .windowInsetsPadding(WindowInsets.safeDrawing.only(WindowInsetsSides.Top + WindowInsetsSides.Horizontal))
            .padding(horizontal = MaterialTheme.spacing.m, vertical = MaterialTheme.spacing.xs),
        verticalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.xs),
    ) {
        Box(Modifier.fillMaxWidth().heightIn(min = MaterialTheme.sizes.glassBar)) {
            if (onBack != null) {
                GlassIconButton(Icons.AutoMirrored.Rounded.ArrowBack, stringResource(R.string.action_back), onBack, Modifier.align(Alignment.CenterStart))
            }
            Text(title, style = MaterialTheme.typography.titleMedium, modifier = Modifier.align(Alignment.Center).semantics { heading() })
            Row(Modifier.align(Alignment.CenterEnd), verticalAlignment = Alignment.CenterVertically, content = actions)
        }
        if (below != null) {
            Box(Modifier.fillMaxWidth().padding(bottom = MaterialTheme.spacing.xs), contentAlignment = Alignment.Center) {
                GlassSurface(Modifier.widthIn(max = MaterialTheme.sizes.contentMaxWidth).fillMaxWidth(), shape = CapsuleShape) { below() }
            }
        }
    }
}

/* ------------------------------------------------------------------ 버튼 */

/**
 * 누름 반응. 눌린 동안 살짝 줄어들고 밝아진다(버튼에는 반드시 누름 상태가 있어야 한다).
 * 애니메이션 끄기 설정이면 크기 변화 없이 밝기만 바뀐다.
 */
@Composable
private fun Modifier.liquidPress(interaction: MutableInteractionSource, enabled: Boolean): Modifier {
    val pressed by interaction.collectIsPressedAsState()
    val reduceMotion = MaterialTheme.details.reduceMotion
    val active = pressed && enabled
    val scale by animateFloatAsState(
        if (active && !reduceMotion) PRESS_SCALE else 1f,
        if (reduceMotion) snap() else spring(dampingRatio = 0.55f, stiffness = 700f),
        label = "press",
    )
    val glow by animateFloatAsState(if (active) 1f else 0f, if (reduceMotion) snap() else tween(PRESS_MS), label = "pressGlow")
    val highlight = MaterialTheme.glass.highlight
    return graphicsLayer {
        scaleX = scale
        scaleY = scale
    }.drawWithContent {
        drawContent()
        if (glow > 0f) drawRect(highlight.copy(alpha = 0.14f * glow))
    }
}

private const val PRESS_SCALE = 0.96f
private const val PRESS_MS = 120
private const val DISABLED_ALPHA = 0.4f

/**
 * 강조 동작(화면당 하나): 색을 입힌 유리(Glass Prominent). 색은 글자가 아니라 바탕에 둔다.
 * 모든 버튼은 높이 56dp, 좌우 여백 24dp 로 같다. 우선순위는 크기가 아니라 스타일로 구분한다.
 */
@Composable
fun PrimaryButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier.fillMaxWidth(),
    enabled: Boolean = true,
) {
    val colors = MaterialTheme.colorScheme
    val interaction = remember { MutableInteractionSource() }
    Box(
        modifier
            .heightIn(min = MaterialTheme.sizes.control)
            .liquidPress(interaction, enabled)
            // 비활성은 채움까지 함께 흐리게(글자만 흐려지면 눌리는 버튼처럼 보인다)
            .alpha(if (enabled) 1f else DISABLED_ALPHA)
            .clip(CapsuleShape)
            .background(Brush.verticalGradient(0f to lerp(colors.primary, Color.White, 0.22f), 1f to colors.primary))
            .border(MaterialTheme.sizes.hairline, glassRim(strength = 1.6f), CapsuleShape)
            .clickable(interactionSource = interaction, indication = null, enabled = enabled, role = Role.Button, onClick = onClick)
            .padding(horizontal = MaterialTheme.spacing.l, vertical = MaterialTheme.spacing.xs),
        contentAlignment = Alignment.Center,
    ) {
        Text(text, style = MaterialTheme.typography.labelLarge, color = colors.onPrimary, textAlign = TextAlign.Center)
    }
}

/** 보조 동작: 무색 유리 캡슐(글자는 단색). 유리 컨테이너 안에서는 유리를 겹치지 않고 옅은 채움만 쓴다. */
@Composable
fun SecondaryButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier.fillMaxWidth(),
    enabled: Boolean = true,
    leading: ImageVector? = null,
) {
    val interaction = remember { MutableInteractionSource() }
    val inGlass = LocalInGlass.current
    Row(
        modifier
            .heightIn(min = MaterialTheme.sizes.control)
            .liquidPress(interaction, enabled)
            .alpha(if (enabled) 1f else DISABLED_ALPHA)
            .clip(CapsuleShape)
            .background(if (inGlass) MaterialTheme.glass.fill.copy(alpha = MaterialTheme.glass.fill.alpha * 0.6f) else MaterialTheme.glass.fill)
            .then(if (inGlass) Modifier else Modifier.border(MaterialTheme.sizes.hairline, glassRim(strength = 0.8f), CapsuleShape))
            .clickable(interactionSource = interaction, indication = null, enabled = enabled, role = Role.Button, onClick = onClick)
            .padding(horizontal = MaterialTheme.spacing.l, vertical = MaterialTheme.spacing.xs),
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        leading?.let {
            Icon(it, contentDescription = null, modifier = Modifier.size(MaterialTheme.sizes.iconS))
            Spacer(Modifier.size(MaterialTheme.spacing.xs))
        }
        Text(text, style = MaterialTheme.typography.labelLarge, textAlign = TextAlign.Center)
    }
}

/**
 * 아이콘만 있는 보조 버튼(56dp 원). 유리 컨테이너 안에 넣는 용도라 유리를 겹치지 않고 보조 버튼과 같은 채움을 쓴다.
 * [selected] 면 켜짐 상태(밝은 채움). 스크린 리더 이름 필수.
 */
@Composable
fun SecondaryIconButton(
    icon: ImageVector,
    contentDescription: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    selected: Boolean = false,
) {
    val interaction = remember { MutableInteractionSource() }
    val colors = MaterialTheme.colorScheme
    Box(
        modifier
            .size(MaterialTheme.sizes.control)
            .liquidPress(interaction, enabled = true)
            .clip(CircleShape)
            .background(if (selected) colors.onBackground else MaterialTheme.glass.fill)
            .clickable(interactionSource = interaction, indication = null, role = Role.Button, onClick = onClick)
            .semantics {
                this.contentDescription = contentDescription
                this.selected = selected
            },
        contentAlignment = Alignment.Center,
    ) {
        Icon(icon, contentDescription = null, tint = if (selected) colors.background else colors.onBackground, modifier = Modifier.size(MaterialTheme.sizes.iconM))
    }
}

/** 떠 있는 원형 유리 아이콘 버튼(뒤로 등). 56dp, 스크린 리더 이름 필수. [selected] 면 켜짐 상태로 보인다. */
@Composable
fun GlassIconButton(
    icon: ImageVector,
    contentDescription: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    selected: Boolean = false,
) {
    val interaction = remember { MutableInteractionSource() }
    GlassSurface(
        modifier
            .size(MaterialTheme.sizes.control)
            .liquidPress(interaction, enabled = true)
            .clickable(interactionSource = interaction, indication = null, role = Role.Button, onClick = onClick)
            .semantics {
                this.contentDescription = contentDescription
                this.selected = selected
            },
        shape = CircleShape,
    ) {
        Icon(
            icon,
            contentDescription = null,
            tint = if (selected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onBackground,
            modifier = Modifier.align(Alignment.Center).size(MaterialTheme.sizes.iconM),
        )
    }
}

val CapsuleShape = RoundedCornerShape(50)

/* ------------------------------------------------------------------ 분할 컨트롤 */

/**
 * Apple식 분할 컨트롤. 라디오 그룹으로 동작하고, 선택은 미끄러지는 밝은 유리 썸과 굵기로 함께 표시한다.
 * 애니메이션 끄기 설정이면 썸이 바로 옮겨 간다.
 */
@Composable
fun <T> SegmentedControl(
    options: List<T>,
    selected: T,
    label: @Composable (T) -> String,
    onSelect: (T) -> Unit,
    modifier: Modifier = Modifier,
    /** false 면 선택 표시(썸)를 그리지 않는다(여러 줄로 나눈 묶음에서 다른 줄이 선택된 경우) */
    showThumb: Boolean = true,
) {
    val index = options.indexOf(selected).coerceAtLeast(0)
    val reduceMotion = MaterialTheme.details.reduceMotion
    val position by animateFloatAsState(
        index.toFloat(),
        if (reduceMotion) snap() else spring(dampingRatio = 0.78f, stiffness = 420f),
        label = "segmentThumb",
    )
    val thumb = MaterialTheme.colorScheme.onBackground
    val rimTop = MaterialTheme.glass.highlight.copy(alpha = 0.9f)
    // 유리 컨테이너 안에서는 자체 채움·테두리를 빼서 유리를 겹치지 않는다
    val inGlass = LocalInGlass.current
    Row(
        modifier
            .clip(CapsuleShape)
            .then(
                if (inGlass) {
                    Modifier
                } else {
                    Modifier.background(MaterialTheme.glass.fill).border(MaterialTheme.sizes.hairline, glassRim(strength = 0.7f), CapsuleShape)
                },
            )
            .padding(MaterialTheme.spacing.xxs)
            .drawBehind {
                if (!showThumb) return@drawBehind
                // 미끄러지는 밝은 유리 썸
                val w = size.width / options.size
                val topLeft = Offset(w * position, 0f)
                val thumbSize = androidx.compose.ui.geometry.Size(w, size.height)
                val radius = CornerRadius(size.height / 2f)
                drawRoundRect(Brush.verticalGradient(0f to thumb, 1f to thumb.copy(alpha = 0.86f)), topLeft, thumbSize, radius)
                drawRoundRect(
                    Brush.verticalGradient(0f to rimTop, 1f to rimTop.copy(alpha = 0.15f)),
                    topLeft,
                    thumbSize,
                    radius,
                    style = Stroke(width = 1.dp.toPx()),
                )
            }
            .selectableGroup(),
    ) {
        options.forEachIndexed { i, option ->
            val isSelected = showThumb && i == index
            Box(
                Modifier
                    .weight(1f)
                    .heightIn(min = MaterialTheme.sizes.control - MaterialTheme.spacing.xs)
                    .clip(CapsuleShape)
                    .selectable(selected = isSelected, onClick = { onSelect(option) }, role = Role.RadioButton)
                    .padding(horizontal = MaterialTheme.spacing.xxs),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    label(option),
                    style = if (isSelected) MaterialTheme.typography.labelLarge else MaterialTheme.typography.bodyLarge,
                    color = if (isSelected) MaterialTheme.colorScheme.background else MaterialTheme.colorScheme.onBackground,
                    textAlign = TextAlign.Center,
                    maxLines = 1,
                )
            }
        }
    }
}

/* ------------------------------------------------------------------ 묶음 목록(콘텐츠층 재질) */

/** iOS 설정 앱처럼 둥근 묶음. 콘텐츠층이라 유리 대신 불투명 재질을 쓴다. */
@Composable
fun InsetGroup(
    modifier: Modifier = Modifier,
    header: String? = null,
    footer: String? = null,
    content: @Composable ColumnScope.() -> Unit,
) {
    Column(modifier.fillMaxWidth()) {
        header?.let { SectionHeader(it, Modifier.padding(start = MaterialTheme.spacing.m, bottom = MaterialTheme.spacing.xs)) }
        Column(
            Modifier
                .fillMaxWidth()
                .clip(MaterialTheme.shapes.large)
                .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.94f)),
            content = content,
        )
        footer?.let {
            Text(
                it,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(start = MaterialTheme.spacing.m, end = MaterialTheme.spacing.m, top = MaterialTheme.spacing.xs),
            )
        }
    }
}

@Composable
fun GroupRow(
    title: String,
    modifier: Modifier = Modifier,
    subtitle: String? = null,
    leading: (@Composable () -> Unit)? = null,
    trailing: (@Composable () -> Unit)? = null,
    onClick: (() -> Unit)? = null,
    showDivider: Boolean = false,
) {
    Column {
        if (showDivider) {
            HorizontalDivider(
                Modifier.padding(start = if (leading != null) MaterialTheme.spacing.xxl + MaterialTheme.spacing.xs else MaterialTheme.spacing.m),
                color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.6f),
            )
        }
        Row(
            modifier
                .fillMaxWidth()
                .heightIn(min = MaterialTheme.sizes.control)
                .then(if (onClick != null) Modifier.clickable(role = Role.Button, onClick = onClick) else Modifier)
                .padding(horizontal = MaterialTheme.spacing.m, vertical = MaterialTheme.spacing.s),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.s),
        ) {
            leading?.invoke()
            Column(Modifier.weight(1f)) {
                Text(title, style = MaterialTheme.typography.bodyLarge)
                subtitle?.let { Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant) }
            }
            trailing?.invoke()
        }
    }
}

/** 여러 줄을 한 묶음으로 보이게 하는 모서리: 첫 줄은 위, 마지막 줄은 아래만 둥글다 */
@Composable
fun groupedRowShape(index: Int, count: Int): Shape {
    val r = GROUP_CORNER
    return when {
        count == 1 -> RoundedCornerShape(r)
        index == 0 -> RoundedCornerShape(topStart = r, topEnd = r)
        index == count - 1 -> RoundedCornerShape(bottomStart = r, bottomEnd = r)
        else -> RoundedCornerShape(0)
    }
}

/** 묶음 목록의 한 줄 배경 */
@Composable
fun Modifier.groupedRow(index: Int, count: Int): Modifier =
    this.clip(groupedRowShape(index, count)).background(MaterialTheme.colorScheme.surface.copy(alpha = 0.94f))

private val GROUP_CORNER = 22.dp

@Composable
fun SectionHeader(text: String, modifier: Modifier = Modifier) {
    Text(
        text = text,
        style = MaterialTheme.typography.labelMedium,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        modifier = modifier.semantics { heading() },
    )
}

@Composable
fun KeyValue(label: String, value: String, modifier: Modifier = Modifier) {
    Column(modifier.semantics(mergeDescendants = true) {}) {
        Text(label, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(value, style = MaterialTheme.typography.titleMedium.merge(TabularNumbers))
    }
}

/* ------------------------------------------------------------------ 배지·안내 */

/** 실제 분석/시연 구분 캡슐. 시연은 상태 예약색과 겹치지 않는 라벤더로 칠한다. [detailed] 면 '시연 · 실제 분석 아님'. */
@Composable
fun ModeBadge(mode: AnalysisMode, modifier: Modifier = Modifier) {
    val demo = mode == AnalysisMode.DEMO
    Text(
        text = stringResource(modeLabel(mode)),
        maxLines = 1,
        overflow = TextOverflow.Ellipsis,
        modifier = modifier
            .clip(CapsuleShape)
            .background(if (demo) MaterialTheme.status.demo else MaterialTheme.glass.fill)
            .padding(horizontal = MaterialTheme.spacing.s, vertical = MaterialTheme.spacing.xxs),
        color = if (demo) MaterialTheme.status.onDemo else MaterialTheme.colorScheme.onBackground,
        style = MaterialTheme.typography.labelMedium,
    )
}

/** 시연 기록임을 알리는 띠 */
@Composable
fun DemoNotice(text: String, modifier: Modifier = Modifier) {
    Text(
        text = text,
        modifier = modifier
            .clip(MaterialTheme.shapes.medium)
            .background(MaterialTheme.status.demo)
            .padding(horizontal = MaterialTheme.spacing.m, vertical = MaterialTheme.spacing.xs),
        color = MaterialTheme.status.onDemo,
        style = MaterialTheme.typography.titleSmall,
    )
}

/** 빈 상태·오류 상태: 아이콘, 제목, 설명, 복구 동작 하나 */
@Composable
fun MessageState(
    icon: ImageVector,
    title: String,
    body: String?,
    modifier: Modifier = Modifier,
    iconTint: Color = MaterialTheme.colorScheme.onSurfaceVariant,
    actionLabel: String? = null,
    onAction: (() -> Unit)? = null,
) {
    // 안내는 하늘 위가 아니라 콘텐츠 재질 위에 올려 읽기 쉽게 한다
    Box(modifier.fillMaxSize().padding(MaterialTheme.spacing.l), contentAlignment = Alignment.Center) {
        Column(
            Modifier
                .widthIn(max = MaterialTheme.sizes.homeStartPaneMaxWidth)
                .clip(MaterialTheme.shapes.extraLarge)
                .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.94f))
                .padding(MaterialTheme.spacing.l),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.s, Alignment.CenterVertically),
        ) {
            Icon(icon, contentDescription = null, tint = iconTint, modifier = Modifier.size(MaterialTheme.sizes.iconL))
            Text(title, style = MaterialTheme.typography.headlineSmall, textAlign = TextAlign.Center)
            body?.let {
                Text(it, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant, textAlign = TextAlign.Center)
            }
            if (actionLabel != null && onAction != null) {
                SecondaryButton(actionLabel, onAction, Modifier.padding(top = MaterialTheme.spacing.xs))
            }
        }
    }
}

@Composable
fun LoadingState(text: String, modifier: Modifier = Modifier) {
    Column(
        modifier.fillMaxSize().semantics(mergeDescendants = true) {},
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.m, Alignment.CenterVertically),
    ) {
        CircularProgressIndicator(color = MaterialTheme.colorScheme.onBackground)
        Text(text, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

/* ------------------------------------------------------------------ 확인 대화상자 */

/** 되돌릴 수 없는 동작의 확인. 콘텐츠 재질 위에 유리 가장자리를 두르고, 확인 동작을 강조 버튼으로 둔다. */
@Composable
fun ConfirmDialog(
    title: String,
    message: String,
    confirmLabel: String,
    dismissLabel: String,
    onConfirm: () -> Unit,
    onDismiss: () -> Unit,
) {
    Dialog(onDismissRequest = onDismiss) {
        Column(
            Modifier
                .widthIn(max = MaterialTheme.sizes.homeStartPaneMaxWidth)
                .clip(MaterialTheme.shapes.extraLarge)
                .background(MaterialTheme.colorScheme.surfaceContainerHigh)
                .background(Brush.verticalGradient(0f to MaterialTheme.glass.highlight.copy(alpha = 0.08f), 0.4f to Color.Transparent))
                .border(MaterialTheme.sizes.hairline, glassRim(strength = 1.2f), MaterialTheme.shapes.extraLarge)
                .padding(MaterialTheme.spacing.l),
            verticalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.s),
        ) {
            Text(title, style = MaterialTheme.typography.headlineSmall, modifier = Modifier.semantics { heading() })
            Text(message, style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Row(Modifier.padding(top = MaterialTheme.spacing.s), horizontalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.s)) {
                SecondaryButton(dismissLabel, onDismiss, Modifier.weight(1f))
                PrimaryButton(confirmLabel, onConfirm, Modifier.weight(1f))
            }
        }
    }
}

/* ------------------------------------------------------------------ 기타 */

/** 스크롤 영역 아래 가장자리를 투명하게 흐려 '아래에 더 있음'을 보여 준다. 더 내려갈 내용이 없으면 그리지 않는다. */
fun Modifier.fadeBottomEdge(enabled: Boolean, fraction: Float = 0.12f): Modifier = if (!enabled) this else this
    .graphicsLayer { compositingStrategy = CompositingStrategy.Offscreen }
    .drawWithContent {
        drawContent()
        drawRect(
            brush = Brush.verticalGradient(0f to Color.Black, 1f - fraction to Color.Black, 1f to Color.Transparent),
            blendMode = BlendMode.DstIn,
        )
    }

@Composable
fun AdaptiveLayout(
    modifier: Modifier = Modifier,
    compact: @Composable () -> Unit,
    wide: @Composable () -> Unit,
) {
    BoxWithConstraints(modifier) {
        if (maxWidth >= MaterialTheme.sizes.wideBreakpoint) wide() else compact()
    }
}

@Composable
fun KeepScreenOn() {
    val view = LocalView.current
    DisposableEffect(view) {
        view.keepScreenOn = true
        onDispose { view.keepScreenOn = false }
    }
}

/** 시스템 '애니메이션 삭제' 설정을 따른다 */
@Composable
fun rememberReduceMotion(): Boolean = MaterialTheme.details.reduceMotion
