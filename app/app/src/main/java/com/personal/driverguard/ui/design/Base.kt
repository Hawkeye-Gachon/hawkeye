package com.personal.driverguard.ui.design

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.size
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.layout
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.style.LineBreak
import androidx.compose.ui.text.style.LineHeightStyle
import androidx.compose.ui.unit.Constraints
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.dp
import com.personal.driverguard.ui.theme.NColor
import kotlin.math.min

/** The design is drawn on a 1280 × 800dp board and scaled to whatever display the app runs on. */
const val BOARD_W = 1280
const val BOARD_H = 800

@Composable
fun DesignBoard(content: @Composable BoxScope.() -> Unit) {
    BoxWithConstraints(Modifier.fillMaxSize().background(NColor.canvas), contentAlignment = Alignment.Center) {
        val scale = min(constraints.maxWidth / BOARD_W.toFloat(), constraints.maxHeight / BOARD_H.toFloat())
        CompositionLocalProvider(LocalDensity provides Density(scale, 1f)) {
            Box(
                Modifier
                    .size(BOARD_W.dp, BOARD_H.dp)
                    .drawBehind {
                        drawRect(NColor.canvas)
                        drawRect(
                            Brush.radialGradient(
                                listOf(Color(0xFF16171C), NColor.canvas),
                                center = Offset(size.width * .3f, 0f),
                                radius = size.width * .72f,
                            )
                        )
                    },
                content = content,
            )
        }
    }
}

/** Absolute placement inside a parent Box, like the design's position:absolute boxes. */
fun Modifier.at(x: Dp, y: Dp, w: Dp, h: Dp): Modifier = this.layout { m, c ->
    val pw = w.roundToPx(); val ph = h.roundToPx()
    val p = m.measure(Constraints.fixed(pw, ph))
    val W = if (c.hasBoundedWidth) c.maxWidth else pw
    val H = if (c.hasBoundedHeight) c.maxHeight else ph
    layout(W, H) { p.place(x.roundToPx(), y.roundToPx()) }
}

/** Frost background with the 48dp grid and a soft white light (behind transparent 3D renders). */
fun Modifier.glassBg(): Modifier = this.drawBehind {
    drawRect(NColor.frost)
    drawRect(Brush.radialGradient(listOf(Color.White, Color.White.copy(alpha = 0f)), center = Offset(size.width * .45f, size.height * .45f), radius = size.maxDimension * .5f))
    val step = 48.dp.toPx(); val line = Color(0x0D131417); val w = 1.dp.toPx()
    var x = -1f
    while (x < size.width) { drawRect(line, topLeft = Offset(x, 0f), size = androidx.compose.ui.geometry.Size(w, size.height)); x += step }
    var y = -1f
    while (y < size.height) { drawRect(line, topLeft = Offset(0f, y), size = androidx.compose.ui.geometry.Size(size.width, w)); y += step }
}

private val trim = LineHeightStyle(LineHeightStyle.Alignment.Center, LineHeightStyle.Trim.None)

/** Korean text keeps words together, like CSS word-break: keep-all. */
fun TextStyle.ko(color: Color? = null, size: TextUnit? = null, line: TextUnit? = null): TextStyle =
    this.copy(
        color = color ?: this.color,
        fontSize = size ?: this.fontSize,
        lineHeight = line ?: this.lineHeight,
        lineHeightStyle = trim,
        lineBreak = LineBreak(LineBreak.Strategy.Balanced, LineBreak.Strictness.Strict, LineBreak.WordBreak.Phrase),
    )
