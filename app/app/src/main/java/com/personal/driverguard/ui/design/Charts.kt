package com.personal.driverguard.ui.design

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.foundation.layout.size
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.PathMeasure
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.TextMeasurer
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.personal.driverguard.ui.theme.NColor
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

/**
 * Bar chart from the design: null = 관찰 기록 없음 (dotted baseline), 0 = 관찰했지만 0회 (4dp stub),
 * peak bars in indigo, a dark tooltip above [tipIndex].
 */
@Composable
fun BarChart(
    values: List<Int?>, labels: (Int) -> String?, width: Dp, height: Dp,
    peak: Set<Int> = emptySet(), tipIndex: Int? = null, tip: String? = null,
    padL: Dp = 30.dp, padT: Dp = 56.dp, padB: Dp = 36.dp, gap: Dp? = null, axis: Boolean = true,
    onBar: ((Int) -> Unit)? = null,
) {
    val tm = rememberTextMeasurer()
    val n = values.size
    Canvas(
        Modifier.size(width, height).then(
            if (onBar != null) Modifier.pointerInput(values) {
                detectTapGestures { p ->
                    val pl = padL.toPx(); val g = (gap ?: if (n > 12) 6.dp else 16.dp).toPx()
                    val bw = (size.width - pl - g * (n - 1)) / n
                    val i = ((p.x - pl) / (bw + g)).toInt()
                    if (i in 0 until n) onBar(i)
                }
            } else Modifier
        )
    ) {
        val W = size.width; val H = size.height
        val pl = padL.toPx(); val pt = padT.toPx(); val pb = padB.toPx()
        val cw = W - pl; val ch = H - pb - pt
        val mx = max(1, values.maxOf { it ?: 0 })
        val top = max(2, mx)
        val step = if (top <= 5) 1 else if (top <= 10) 2 else 5
        fun y(v: Int) = pt + ch - v.toFloat() / top * ch
        val small = S.t(14, 18, color = NColor.ink3)
        if (axis) {
            var t = 0
            while (t <= top) {
                drawRect(NColor.line, Offset(pl, y(t)), Size(cw, 1.dp.toPx()))
                text(tm, "$t", small, Offset(0f, y(t) - 9.dp.toPx()))
                t += step
            }
        }
        val g = (gap ?: if (n > 12) 6.dp else 16.dp).toPx()
        val bw = (cw - g * (n - 1)) / n
        values.forEachIndexed { i, v ->
            val x = pl + i * (bw + g)
            when {
                v == null -> drawLine(NColor.ink4, Offset(x + 1, y(0) - 1), Offset(x + bw - 1, y(0) - 1), 2.dp.toPx(), StrokeCap.Round, PathEffect.dashPathEffect(floatArrayOf(2.dp.toPx(), 4.dp.toPx())))
                v == 0 -> drawRoundRect(NColor.line2, Offset(x, y(0) - 4.dp.toPx()), Size(bw, 4.dp.toPx()), CornerRadius(2.dp.toPx()))
                else -> {
                    val pk = i in peak
                    drawRoundRect(if (pk) NColor.indigo else Color(0x33131417), Offset(x, y(v)), Size(bw, y(0) - y(v)), CornerRadius(min(6.dp.toPx(), bw / 2)))
                    if (n <= 12) {
                        val st = S.t(16, 20, 600, if (pk) NColor.indigo else NColor.ink2)
                        val r = tm.measure("${v}회", st)
                        drawText(r, topLeft = Offset(x + bw / 2 - r.size.width / 2, y(v) - 10.dp.toPx() - r.size.height))
                    }
                }
            }
            labels(i)?.let { l ->
                val r = tm.measure(l, small)
                val lx = if (n > 12) x else x + bw / 2 - r.size.width / 2
                drawText(r, topLeft = Offset(lx, H - r.size.height - 2.dp.toPx()))
            }
        }
        if (tipIndex != null && tip != null) {
            val v = values[tipIndex] ?: 0
            val x = pl + tipIndex * (bw + g) + bw / 2
            val ty = y(v) - 16.dp.toPx()
            val st = S.t(15, 18, 600, Color.White)
            val r = tm.measure(tip, st)
            val tw = r.size.width + 32.dp.toPx()
            val tx = (x - tw / 2).coerceIn(pl, W - tw)
            drawRoundRect(NColor.ink, Offset(tx, ty - 42.dp.toPx()), Size(tw, 38.dp.toPx()), CornerRadius(12.dp.toPx()))
            val p = Path().apply { moveTo(x - 7.dp.toPx(), ty - 5.dp.toPx()); lineTo(x, ty + 2.dp.toPx()); lineTo(x + 7.dp.toPx(), ty - 5.dp.toPx()); close() }
            drawPath(p, NColor.ink)
            drawText(r, topLeft = Offset(tx + tw / 2 - r.size.width / 2, ty - 23.dp.toPx() - r.size.height / 2))
        }
    }
}

private fun DrawScope.text(tm: TextMeasurer, s: String, st: TextStyle, at: Offset) = drawText(tm.measure(s, st), topLeft = at)

/** The trip map from the report: grid, context roads, driven route, events, rest stop. 688 × 752. */
@Composable
fun RouteMap() {
    val tm = rememberTextMeasurer()
    Canvas(Modifier.size(688.dp, 752.dp)) {
        val d = density
        fun P(x: Float, y: Float) = Offset(x * d, y * d)
        drawRect(NColor.mapBg)
        var gx = 0f; while (gx <= 688) { drawRect(Color(0xFFE2E5EE), P(gx, 0f), Size(1 * d, 752 * d)); gx += 43 }
        var gy = 0f; while (gy <= 752) { drawRect(Color(0xFFE2E5EE), P(0f, gy), Size(688 * d, 1 * d)); gy += 43 }
        fun cubic(vararg pts: Float) = Path().apply {
            moveTo(pts[0] * d, pts[1] * d)
            var i = 2
            while (i + 5 < pts.size + 0) { cubicTo(pts[i] * d, pts[i + 1] * d, pts[i + 2] * d, pts[i + 3] * d, pts[i + 4] * d, pts[i + 5] * d); i += 6 }
        }
        val roads = listOf(
            cubic(-20f, 210f, 160f, 230f, 300f, 170f, 460f, 200f, 620f, 230f, 640f, 260f, 720f, 240f),
            cubic(90f, -20f, 120f, 200f, 80f, 420f, 140f, 780f),
            cubic(540f, -20f, 520f, 220f, 590f, 460f, 560f, 780f),
            cubic(-20f, 560f, 200f, 540f, 420f, 600f, 720f, 570f),
            cubic(300f, -20f, 330f, 300f, 300f, 500f, 360f, 780f),
        )
        roads.forEach {
            drawPath(it, Color(0xFFDCE0EA), style = Stroke(13 * d, cap = StrokeCap.Round))
            drawPath(it, Color.White, style = Stroke(10 * d, cap = StrokeCap.Round))
        }
        val route = cubic(190f, 150f, 230f, 230f, 250f, 280f, 300f, 340f, 350f, 400f, 380f, 440f, 410f, 500f, 440f, 560f, 450f, 580f, 470f, 610f)
        drawPath(route, Color.White, style = Stroke(22 * d, cap = StrokeCap.Round))
        drawPath(route, NColor.indigo.copy(alpha = .18f), style = Stroke(16 * d, cap = StrokeCap.Round))
        drawPath(route, NColor.indigo, style = Stroke(5 * d, cap = StrokeCap.Round))
        drawPath(cubic(470f, 610f, 500f, 630f, 530f, 640f, 560f, 646f), NColor.indigo, style = Stroke(3 * d, cap = StrokeCap.Round, pathEffect = PathEffect.dashPathEffect(floatArrayOf(2 * d, 8 * d))))
        val pm = PathMeasure().apply { setPath(route, false) }
        fun at(f: Float) = pm.getPosition(pm.length * f)
        fun tag(p: Offset, s: String, end: Boolean, color: Color = NColor.ink2, weight: Int = 500) {
            val st = S.t(15, 18, weight, color)
            val r = tm.measure(s, st)
            val x = if (end) p.x - r.size.width else p.x
            drawText(r, topLeft = Offset(x, p.y - r.size.height / 2))
        }
        fun pin(f: Float, c: Color, s: String, end: Boolean, ring: Boolean = false) {
            val p = at(f)
            if (ring) drawCircle(Color(0x24F62E24), 22 * d, p)
            drawCircle(Color.White, 13 * d, p); drawCircle(c, 10 * d, p)
            tag(Offset(p.x + (if (end) -(if (ring) 30 else 18) else 18) * d, p.y), s, end)
        }
        pin(.16f, NColor.alert, "오후 2:37", false)
        pin(.36f, NColor.alert, "오후 2:44", true)
        pin(.57f, NColor.alert, "오후 2:52", false)
        pin(.76f, NColor.danger, "오후 2:55", true, ring = true)
        val w = at(.8f)
        drawCircle(Color.White, 10 * d, w); drawCircle(NColor.indigo, 7 * d, w)
        tag(Offset(w.x + 16 * d, w.y), "오후 2:56 경유지 추가", false, NColor.indigo, 600)
        // rest stop pin
        val pinPath = Path().apply {
            val ox = 548 * d; val oy = 608 * d
            moveTo(ox + 14 * d, oy + 36 * d)
            cubicTo(ox + 14 * d, oy + 36 * d, ox + 2 * d, oy + 25.6f * d, ox + 2 * d, oy + 15.6f * d)
            cubicTo(ox + 2 * d, oy + 9 * d, ox + 7.4f * d, oy + 3.6f * d, ox + 14 * d, oy + 3.6f * d)
            cubicTo(ox + 20.6f * d, oy + 3.6f * d, ox + 26 * d, oy + 9 * d, ox + 26 * d, oy + 15.6f * d)
            cubicTo(ox + 26 * d, oy + 25.6f * d, ox + 14 * d, oy + 36 * d, ox + 14 * d, oy + 36 * d)
            close()
        }
        drawPath(pinPath, NColor.indigo); drawPath(pinPath, Color.White, style = Stroke(2.5f * d))
        drawCircle(Color.White, 4.5f * d, P(562f, 623f))
        tag(P(586f, 628f), "천안 졸음쉼터", false, NColor.ink, 600)
        val s = at(0f); val e = at(1f)
        drawCircle(Color.White, 11 * d, s); drawCircle(NColor.indigo, 11 * d, s, style = Stroke(5 * d))
        tag(Offset(s.x + 20 * d, s.y), "출발 오후 2:30", false, NColor.ink, 600)
        drawCircle(Color.White, 13 * d, e); drawCircle(NColor.ink, 11 * d, e)
        tag(Offset(e.x - 18 * d, e.y + 26 * d), "도착 오후 3:12", true, NColor.ink, 600)
    }
}

fun peakOf(v: List<Int?>): Set<Int> { val m = v.maxOf { it ?: 0 }; return if (m == 0) emptySet() else v.indices.filter { v[it] == m }.toSet() }

fun slotName(h: Int): String {
    val a = if (h < 12) "오전" else "오후"
    val s = if (h % 12 == 0) 12 else h % 12
    val e = if ((h + 1) % 12 == 0) 12 else (h + 1) % 12
    return "$a $s~${e}시"
}

fun mins(m: Int) = if (m >= 60) "${m / 60}시간 ${m % 60}분" else "${m}분"

@Suppress("unused") private fun ceilInt(f: Float) = ceil(f).toInt()

/** 실제 경로 지도에 찍을 감지 위치 */
class RoutePin(val latitude: Double, val longitude: Double, val color: Color, val label: String)

/**
 * 실제 주행 경로(위도·경도 목록)를 리포트 카드 크기(688 × 752)에 맞춰 그린다.
 * 지도 SDK 를 쓰지 않으므로 도로·지명 없이 격자 위에 경로 모양과 감지 위치만 보여 준다.
 * 위도·경도를 등장방형으로 투영하고(경도에 cos(위도)를 곱함) 가로세로 비율을 지킨 채 가운데 맞춘다.
 */
@Composable
fun LiveRouteMap(points: List<List<Double>>, pins: List<RoutePin>, startLabel: String, endLabel: String) {
    val tm = rememberTextMeasurer()
    Canvas(Modifier.size(688.dp, 752.dp)) {
        val d = density
        drawRect(NColor.mapBg)
        var gx = 0f; while (gx <= 688) { drawRect(Color(0xFFE2E5EE), Offset(gx * d, 0f), Size(1 * d, 752 * d)); gx += 43 }
        var gy = 0f; while (gy <= 752) { drawRect(Color(0xFFE2E5EE), Offset(0f, gy * d), Size(688 * d, 1 * d)); gy += 43 }
        val all = points.map { it[0] to it[1] } + pins.map { it.latitude to it.longitude }
        if (all.isEmpty()) {
            val st = S.t(17, 24, 500, NColor.ink3)
            val r = tm.measure("위치 기록이 없어 경로를 표시하지 않습니다", st)
            drawText(r, topLeft = Offset(size.width / 2 - r.size.width / 2, size.height / 2 - r.size.height / 2))
            return@Canvas
        }
        val midLat = all.map { it.first }.average()
        val k = kotlin.math.cos(Math.toRadians(midLat))
        val xs = all.map { it.second * k }
        val ys = all.map { it.first }
        val minX = xs.min(); val maxX = xs.max(); val minY = ys.min(); val maxY = ys.max()
        // 위쪽 안내 상자와 아래쪽 범례를 피해 여백을 둔다
        val left = 70 * d; val right = size.width - 70 * d; val top = 150 * d; val bottom = size.height - 110 * d
        val spanX = (maxX - minX).takeIf { it > 1e-9 } ?: 1e-9
        val spanY = (maxY - minY).takeIf { it > 1e-9 } ?: 1e-9
        val scale = min((right - left) / spanX, (bottom - top) / spanY)
        val ox = left + ((right - left) - spanX * scale) / 2
        val oy = top + ((bottom - top) - spanY * scale) / 2
        fun P(lat: Double, lng: Double) = Offset((ox + (lng * k - minX) * scale).toFloat(), (oy + (maxY - lat) * scale).toFloat())
        fun tag(p: Offset, s: String, color: Color = NColor.ink2, weight: Int = 600) {
            val r = tm.measure(s, S.t(15, 18, weight, color))
            val x = (p.x + 18 * d).coerceAtMost(size.width - r.size.width - 8 * d)
            drawText(r, topLeft = Offset(x, p.y - r.size.height / 2))
        }
        if (points.size >= 2) {
            val path = Path().apply {
                val f = P(points[0][0], points[0][1]); moveTo(f.x, f.y)
                points.drop(1).forEach { val q = P(it[0], it[1]); lineTo(q.x, q.y) }
            }
            drawPath(path, Color.White, style = Stroke(22 * d, cap = StrokeCap.Round, join = androidx.compose.ui.graphics.StrokeJoin.Round))
            drawPath(path, NColor.indigo.copy(alpha = .18f), style = Stroke(16 * d, cap = StrokeCap.Round, join = androidx.compose.ui.graphics.StrokeJoin.Round))
            drawPath(path, NColor.indigo, style = Stroke(5 * d, cap = StrokeCap.Round, join = androidx.compose.ui.graphics.StrokeJoin.Round))
        }
        pins.forEach { pin ->
            val p = P(pin.latitude, pin.longitude)
            drawCircle(Color.White, 13 * d, p); drawCircle(pin.color, 10 * d, p)
            tag(p, pin.label, NColor.ink2, 500)
        }
        points.firstOrNull()?.let { s0 ->
            val s = P(s0[0], s0[1])
            drawCircle(Color.White, 11 * d, s); drawCircle(NColor.indigo, 11 * d, s, style = Stroke(5 * d))
            tag(s, startLabel, NColor.ink)
        }
        if (points.size >= 2) points.last().let { e0 ->
            val e = P(e0[0], e0[1])
            drawCircle(Color.White, 13 * d, e); drawCircle(NColor.ink, 11 * d, e)
            tag(Offset(e.x, e.y + 26 * d), endLabel, NColor.ink)
        }
    }
}
