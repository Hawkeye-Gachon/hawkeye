package com.personal.driverguard.ui.design

import androidx.annotation.DrawableRes
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsFocusedAsState
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicText
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material3.Icon
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.scale
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.em
import androidx.compose.ui.unit.sp
import com.personal.driverguard.R
import com.personal.driverguard.ui.theme.AstaSans
import com.personal.driverguard.ui.theme.NColor
import com.personal.driverguard.ui.theme.NType

// ------------------------------------------------------------------ text styles used across screens
object S {
    val h1 = NType.headline.ko(NColor.ink)
    val h2 = NType.title.ko(NColor.ink)
    val h3 = NType.subtitle.ko(NColor.ink)
    val alarm = NType.alarm.ko(NColor.ink)
    val body = NType.body.ko(NColor.ink2)
    val small = NType.small.ko(NColor.ink3)
    val label = NType.label.ko(NColor.ink3)
    val caption = NType.caption.ko(NColor.ink3)
    val button = NType.button.ko(NColor.ink)
    val clock = NType.clock.ko(NColor.ink).copy(fontFeatureSettings = "tnum")
    val count = NType.count.ko(NColor.ink).copy(fontFeatureSettings = "tnum")
    val metric = NType.metric.ko(NColor.ink).copy(fontFeatureSettings = "tnum")
    val peak = NType.peak.ko(NColor.ink)
    val row = NType.row.ko(NColor.ink2)
    fun t(size: Int, line: Int, weight: Int = 400, color: Color = NColor.ink, tracking: Double = 0.0) =
        TextStyle(fontFamily = AstaSans, fontWeight = FontWeight(weight), fontSize = size.sp, lineHeight = line.sp, letterSpacing = tracking.em, color = color).ko()
}

/** CSS word-break: keep-all. Joins letters inside a word with U+2060 so Korean wraps only at spaces. */
fun keepAll(s: String): String {
    if (s.length < 2) return s
    val b = StringBuilder(s.length * 2)
    for (i in s.indices) {
        b.append(s[i])
        if (i < s.lastIndex && !s[i].isWhitespace() && !s[i + 1].isWhitespace() && (isHangul(s[i]) || isHangul(s[i + 1]))) b.append('\u2060')
    }
    return b.toString()
}
private fun isHangul(c: Char) = c in '\uAC00'..'\uD7A3'

fun keepAll(a: AnnotatedString): AnnotatedString = androidx.compose.ui.text.buildAnnotatedString {
    a.spanStyles.let { spans ->
        var pos = 0
        val starts = (spans.map { it.start } + spans.map { it.end } + listOf(0, a.length)).distinct().sorted()
        for (k in 0 until starts.size - 1) {
            val st = starts[k]; val en = starts[k + 1]; if (st >= en) continue
            val seg = a.text.substring(st, en)
            val styles = spans.filter { it.start <= st && it.end >= en }
            styles.forEach { pushStyle(it.item) }
            append(keepAll(seg))
            // keep a joiner across segment boundaries inside a word
            if (en < a.length && !a.text[en - 1].isWhitespace() && !a.text[en].isWhitespace()) append('\u2060')
            styles.forEach { _ -> pop() }
            pos = en
        }
    }
}

@Composable
fun T(text: String, style: TextStyle, modifier: Modifier = Modifier, align: TextAlign? = null, maxLines: Int = Int.MAX_VALUE) =
    BasicText(keepAll(text), modifier, style = if (align != null) style.copy(textAlign = align) else style, maxLines = maxLines)

@Composable
fun T(text: AnnotatedString, style: TextStyle, modifier: Modifier = Modifier) = BasicText(keepAll(text), modifier, style = style)

@Composable
fun Ic(@DrawableRes id: Int, color: Color, size: Dp = 24.dp, modifier: Modifier = Modifier) =
    Icon(painterResource(id), contentDescription = null, tint = color, modifier = modifier.size(size))

@Composable
fun Mark(size: Dp = 34.dp, textStyle: TextStyle = S.t(26, 30, 600, tracking = -.03), gap: Dp = 12.dp) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(gap)) {
        Image(painterResource(R.drawable.ic_mark), null, Modifier.size(size))
        T("앱이름", textStyle)
    }
}

// ------------------------------------------------------------------ cards
val CardShape = RoundedCornerShape(32.dp)

@Composable
fun BoxScope.SceneCard(modifier: Modifier = Modifier, content: @Composable BoxScope.() -> Unit) =
    Box(Modifier.at(24.dp, 24.dp, 688.dp, 752.dp).clip(CardShape).then(modifier), content = content)

@Composable
fun BoxScope.Sheet(color: Color = NColor.frost, content: @Composable ColumnScope.() -> Unit) =
    Column(
        Modifier.at(736.dp, 24.dp, 520.dp, 752.dp)
            .shadow(36.dp, CardShape, ambientColor = Color.Black.copy(alpha = .5f), spotColor = Color.Black.copy(alpha = .7f))
            .clip(CardShape).background(color),
        content = content,
    )

@Composable
fun BoxScope.FullCard(modifier: Modifier = Modifier, content: @Composable BoxScope.() -> Unit) =
    Box(Modifier.at(24.dp, 24.dp, 1232.dp, 752.dp).clip(CardShape).background(NColor.frost).then(modifier), content = content)

/** Sheet body with the design's padding; [bottom] stays pinned to the bottom like `.acts`. */
@Composable
fun ColumnScope.SheetBody(
    top: Dp = 32.dp, horizontal: Dp = 40.dp, bottomPad: Dp = 40.dp, gap: Dp = 14.dp,
    content: @Composable ColumnScope.() -> Unit, bottom: @Composable ColumnScope.() -> Unit = {},
) {
    Column(Modifier.weight(1f).fillMaxWidth().padding(start = horizontal, end = horizontal, top = top, bottom = bottomPad)) {
        Column(verticalArrangement = Arrangement.spacedBy(gap), content = content)
        Spacer(Modifier.weight(1f))
        Column(verticalArrangement = Arrangement.spacedBy(12.dp), content = bottom)
    }
}

@Composable
fun SheetPic(@DrawableRes img: Int, height: Dp = 250.dp, glass: Boolean = false, headScale: Boolean = false, biasX: Float = 0f, biasY: Float = 0f, tint: Color? = null, overlay: @Composable BoxScope.() -> Unit = {}) {
    Box(Modifier.fillMaxWidth().height(height).then(if (glass) Modifier.glassBg() else Modifier)) {
        Image(
            painterResource(img), null,
            contentScale = androidx.compose.ui.layout.ContentScale.Crop,
            alignment = androidx.compose.ui.BiasAlignment(biasX, biasY),
            colorFilter = tint?.let { androidx.compose.ui.graphics.ColorFilter.tint(it, androidx.compose.ui.graphics.BlendMode.Saturation) },
            modifier = Modifier.fillMaxSize().then(
                if (headScale) Modifier.graphicsLayerScale(.9f, .26f, 1f) else Modifier
            ),
        )
        overlay()
    }
}

fun Modifier.graphicsLayerScale(s: Float, ox: Float, oy: Float): Modifier =
    this.graphicsLayer { scaleX = s; scaleY = s; transformOrigin = androidx.compose.ui.graphics.TransformOrigin(ox, oy) }

// ------------------------------------------------------------------ buttons
enum class Kind { Primary, Ghost, OnImage, OnDark }

@Composable
fun Btn(
    text: String, modifier: Modifier = Modifier, kind: Kind = Kind.Primary, enabled: Boolean = true,
    height: Dp = 72.dp, radius: Dp = 24.dp, style: TextStyle = S.button, horizontal: Dp = 40.dp,
    @DrawableRes icon: Int? = null, onClick: () -> Unit = {},
) {
    val src = remember { MutableInteractionSource() }
    val pressed by src.collectIsPressedAsState()
    val (bg, fg) = when {
        !enabled -> NColor.fill to NColor.ink4
        kind == Kind.Primary -> (if (pressed) Color(0xFF4D39F0) else NColor.indigo) to Color.White
        kind == Kind.Ghost -> (if (pressed) NColor.fill2 else NColor.fill) to NColor.ink
        kind == Kind.OnImage -> Color.White.copy(alpha = .72f) to NColor.ink
        else -> Color.White.copy(alpha = if (pressed) .26f else .16f) to Color.White
    }
    Row(
        modifier.height(height).scale(if (pressed && enabled) .985f else 1f).clip(RoundedCornerShape(radius)).background(bg)
            .clickable(src, null, enabled = enabled, onClick = onClick).padding(horizontal = horizontal),
        horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterHorizontally),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (icon != null) Ic(icon, fg)
        T(text, style.copy(color = fg), maxLines = 1)
    }
}

@Composable
fun BackBtn(text: String?, modifier: Modifier = Modifier, kind: Kind = Kind.Ghost, @DrawableRes icon: Int = R.drawable.ic_back, onClick: () -> Unit) =
    Btn(text ?: "", modifier, kind, height = 52.dp, radius = 16.dp, style = S.t(18, 24, 600), horizontal = if (text == null) 14.dp else 18.dp, icon = icon, onClick = onClick)

// ------------------------------------------------------------------ form
enum class FieldState { Normal, Error }

@Composable
fun Field(
    label: String, value: String, onChange: (String) -> Unit = {}, password: Boolean = false,
    state: FieldState = FieldState.Normal, message: String? = null, placeholder: String? = null, forceFocus: Boolean = false,
) {
    val src = remember { MutableInteractionSource() }
    val focused by src.collectIsFocusedAsState()
    val border = when {
        state == FieldState.Error -> BorderStroke(2.dp, NColor.danger)
        focused || forceFocus -> BorderStroke(2.dp, NColor.ink)
        else -> BorderStroke(1.dp, NColor.line2)
    }
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        T(label, S.t(16, 22, 500, NColor.ink2))
        BasicTextField(
            value = value, onValueChange = onChange, singleLine = true, interactionSource = src,
            textStyle = S.t(if (password) 16 else 19, 24, color = NColor.ink, tracking = if (password) .25 else 0.0),
            visualTransformation = if (password) PasswordVisualTransformation('●') else VisualTransformation.None,
            cursorBrush = SolidColor(NColor.indigo),
            modifier = Modifier.fillMaxWidth().height(60.dp).clip(RoundedCornerShape(16.dp)).background(Color.White).border(border, RoundedCornerShape(16.dp)),
            decorationBox = { inner ->
                Box(Modifier.fillMaxSize().padding(horizontal = 20.dp), contentAlignment = Alignment.CenterStart) {
                    if (value.isEmpty() && placeholder != null) T(placeholder, S.t(19, 24, color = NColor.ink4))
                    inner()
                }
            },
        )
        if (message != null) T(message, S.t(15, 20, color = if (state == FieldState.Error) NColor.dangerInk else NColor.ink3))
    }
}

@Composable
fun CheckBox(checked: Boolean) {
    Box(
        Modifier.size(30.dp).clip(RoundedCornerShape(10.dp))
            .background(if (checked) NColor.indigo else Color.Transparent)
            .border(2.dp, if (checked) NColor.indigo else NColor.line2, RoundedCornerShape(10.dp)),
        contentAlignment = Alignment.Center,
    ) { if (checked) Ic(R.drawable.ic_check, Color.White, 20.dp) }
}

@Composable
fun Segmented(items: List<String>, selected: Int, small: Boolean = false, onSelect: (Int) -> Unit) {
    Row(Modifier.clip(RoundedCornerShape(20.dp)).background(NColor.fill).padding(4.dp), horizontalArrangement = Arrangement.spacedBy(2.dp)) {
        items.forEachIndexed { i, s ->
            val sel = i == selected
            Box(
                Modifier.height(if (small) 40.dp else 48.dp)
                    .then(if (sel) Modifier.shadow(4.dp, RoundedCornerShape(if (small) 14.dp else 16.dp), ambientColor = Color(0x33131417), spotColor = Color(0x33131417)) else Modifier)
                    .clip(RoundedCornerShape(if (small) 14.dp else 16.dp))
                    .background(if (sel) Color.White else Color.Transparent)
                    .clickable { onSelect(i) }
                    .padding(horizontal = if (small) 16.dp else 22.dp),
                contentAlignment = Alignment.Center,
            ) { T(s, S.t(if (small) 15 else 17, 22, 600, if (sel) NColor.ink else NColor.ink3)) }
        }
    }
}

// ------------------------------------------------------------------ route sequence (현재 위치 → 쉼터 → 집)
enum class Stop { Here, Rest, Was, Dest }

@Composable
fun Seq(stops: List<Pair<Stop, String>>, modifier: Modifier = Modifier) {
    val n = stops.size
    Row(
        modifier.fillMaxWidth().drawBehind {
            val seg = size.width / n
            drawRect(NColor.line2, topLeft = Offset(seg / 2, 8.dp.toPx()), size = Size(size.width - seg, 2.dp.toPx()))
        },
    ) {
        stops.forEach { (kind, label) ->
            Column(Modifier.weight(1f), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Box(Modifier.size(18.dp).drawBehind {
                    val r = size.minDimension / 2
                    when (kind) {
                        Stop.Rest -> { drawCircle(NColor.indigo.copy(alpha = .16f), r + 6.dp.toPx()); drawCircle(NColor.indigo, r) }
                        Stop.Was -> drawCircle(NColor.ink3, r)
                        Stop.Dest -> drawCircle(NColor.ink, r)
                        Stop.Here -> { drawCircle(Color.White, r); drawCircle(NColor.ink4, r - 1.dp.toPx(), style = Stroke(2.dp.toPx())) }
                    }
                })
                T(label, S.t(15, 20, if (kind == Stop.Rest) 600 else 400, if (kind == Stop.Rest) NColor.ink else NColor.ink2), align = TextAlign.Center)
            }
        }
    }
}

// ------------------------------------------------------------------ info rows
@Composable
fun InfoRows(rows: List<Pair<Int, AnnotatedString>>, onRed: Boolean = false, modifier: Modifier = Modifier) {
    val line = if (onRed) Color.White.copy(alpha = .28f) else NColor.line
    Column(modifier.fillMaxWidth().drawBehind { drawRect(line, size = Size(size.width, 1.dp.toPx())) }) {
        rows.forEach { (icon, text) ->
            Row(
                Modifier.fillMaxWidth().drawBehind { drawRect(line, topLeft = Offset(0f, size.height - 1.dp.toPx()), size = Size(size.width, 1.dp.toPx())) }.padding(vertical = 14.dp),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Ic(icon, if (onRed) Color.White.copy(alpha = .8f) else NColor.ink3, 22.dp, Modifier.padding(top = 1.dp))
                T(text, S.row.copy(color = if (onRed) Color.White.copy(alpha = .92f) else NColor.ink2))
            }
        }
    }
}

fun boldParts(vararg parts: Pair<String, Boolean>, boldColor: Color = NColor.ink) = androidx.compose.ui.text.buildAnnotatedString {
    parts.forEach { (s, b) -> if (b) pushStyle(androidx.compose.ui.text.SpanStyle(fontWeight = FontWeight(600), color = boldColor)).also { append(s); pop() } else append(s) }
}

// ------------------------------------------------------------------ status
@Composable
fun StateLine(text: String, color: Color = NColor.indigo) {
    val t = rememberInfiniteTransition(label = "breathe")
    val p by t.animateFloat(0f, 1f, infiniteRepeatable(tween(2400)), label = "p")
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
        Canvas(Modifier.size(10.dp)) {
            drawCircle(color.copy(alpha = .5f * (1 - p)), radius = size.minDimension / 2 + 12.dp.toPx() * p)
            drawCircle(color)
        }
        T(text, S.t(18, 24, 600, NColor.ink2))
    }
}

@Composable
fun Counts(items: List<Pair<String, String>>, gap: Dp = 40.dp, valueStyle: TextStyle = S.count) {
    Row(horizontalArrangement = Arrangement.spacedBy(gap)) {
        items.forEach { (v, l) -> Column(verticalArrangement = Arrangement.spacedBy(2.dp)) { T(v, valueStyle); T(l, S.label) } }
    }
}

// ------------------------------------------------------------------ edge glow (SPEC.md)
enum class Glow { None, Orange, Red, RedFaint }

@Composable
fun EdgeGlow(glow: Glow) {
    if (glow == Glow.None) return
    val color = if (glow == Glow.Orange) NColor.alert else Color(0xFFFF2E24)
    val pulse = glow == Glow.Red
    val t = rememberInfiniteTransition(label = "edge")
    val a by t.animateFloat(if (pulse) .62f else 1f, 1f, infiniteRepeatable(tween(1400, easing = FastOutSlowInEasing), RepeatMode.Reverse), label = "a")
    val k = if (glow == Glow.RedFaint) .4f else 1f
    Canvas(Modifier.fillMaxSize()) {
        val al = (if (pulse) a else 1f) * k
        drawRect(color.copy(alpha = .95f * al), style = Stroke((if (glow == Glow.Orange) 3.dp else 4.dp).toPx()))
        val layers = if (glow == Glow.Orange) listOf(64.dp to .55f, 200.dp to .18f) else listOf(40.dp to .7f, 100.dp to .26f)
        layers.forEach { (w, kk) ->
            val px = w.toPx(); val c = color.copy(alpha = kk * al)
            drawRect(Brush.verticalGradient(listOf(c, Color.Transparent), 0f, px), size = Size(size.width, px))
            drawRect(Brush.verticalGradient(listOf(Color.Transparent, c), size.height - px, size.height), topLeft = Offset(0f, size.height - px), size = Size(size.width, px))
            drawRect(Brush.horizontalGradient(listOf(c, Color.Transparent), 0f, px), size = Size(px, size.height))
            drawRect(Brush.horizontalGradient(listOf(Color.Transparent, c), size.width - px, size.width), topLeft = Offset(size.width - px, 0f), size = Size(px, size.height))
        }
    }
}

// ------------------------------------------------------------------ dialog scrim
@Composable
fun Scrim() = Box(Modifier.fillMaxSize().background(Color(0x9906070A)))

@Composable
fun Dot(color: Color, size: Dp = 12.dp) = Box(Modifier.size(size).clip(CircleShape).background(color))

@Composable
fun HLine(modifier: Modifier = Modifier) = Box(modifier.fillMaxWidth().height(1.dp).background(NColor.line))

@Composable
fun Gap(h: Dp) = Spacer(Modifier.height(h))

@Composable
fun RowScope.Fill() = Spacer(Modifier.weight(1f))

@Composable
fun WGap(w: Dp) = Spacer(Modifier.width(w))
