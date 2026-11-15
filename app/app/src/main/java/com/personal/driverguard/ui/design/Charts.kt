package com.personal.driverguard.ui.design

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.foundation.layout.size
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.StrokeJoin
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
import com.personal.driverguard.domain.engine.DistanceAccumulator
import com.personal.driverguard.ui.theme.NColor
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
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

/** 실제 경로 지도에 찍을 쉼터(경유지로 넣은 곳). [visited] 는 실제로 들러 쉬었는지. */
class RouteStop(val latitude: Double, val longitude: Double, val name: String, val visited: Boolean)

/**
 * 실제 주행 경로(위도·경도 목록)를 리포트 카드 크기(688 × 752)에 맞춰 그린다.
 *
 * 지도 SDK 를 쓰지 않으므로 주변 도로와 지명은 없다. 그 대신 기록에서 알 수 있는 것만으로 지도처럼 읽히게 한다:
 * 진행 방향 표시, 일정 거리마다의 눈금, 축척, 북쪽 표시, 경유지로 넣은 쉼터. 측정하지 않은 것(주변 도로 등)은 그리지 않는다.
 * 글자는 서로와 경로·핀을 가리지 않는 자리를 찾아 놓고, 핀에서 떨어져 놓이면 가는 선으로 잇는다
 * (감지가 한곳에 몰려도 겹치지 않는다).
 * 위도·경도를 등장방형으로 투영하고(경도에 cos(위도)를 곱함) 가로세로 비율을 지킨 채 가운데 맞춘다.
 */
@Composable
fun LiveRouteMap(points: List<List<Double>>, pins: List<RoutePin>, startLabel: String, endLabel: String, stops: List<RouteStop> = emptyList()) {
    val tm = rememberTextMeasurer()
    Canvas(Modifier.size(688.dp, 752.dp)) {
        val d = density
        drawRect(NColor.mapBg)
        var gx = 0f; while (gx <= 688) { drawRect(Color(0xFFE2E5EE), Offset(gx * d, 0f), Size(1 * d, 752 * d)); gx += 43 }
        var gy = 0f; while (gy <= 752) { drawRect(Color(0xFFE2E5EE), Offset(0f, gy * d), Size(688 * d, 1 * d)); gy += 43 }
        val all = points.map { it[0] to it[1] } + pins.map { it.latitude to it.longitude } + stops.map { it.latitude to it.longitude }
        if (all.isEmpty()) {
            val st = S.t(17, 24, 500, NColor.ink3)
            val r = tm.measure("위치 기록이 없어 경로를 표시하지 않습니다", st)
            drawText(r, topLeft = Offset(size.width / 2 - r.size.width / 2, size.height / 2 - r.size.height / 2))
            return@Canvas
        }
        val midLat = all.map { it.first }.average()
        val k = cos(Math.toRadians(midLat))
        val xs = all.map { it.second * k }
        val ys = all.map { it.first }
        val minX = xs.min(); val maxX = xs.max(); val minY = ys.min(); val maxY = ys.max()
        // 위쪽 안내 상자와 아래쪽 범례, 오른쪽에 놓일 글자를 생각해 여백을 둔다
        val left = 84 * d; val right = size.width - 120 * d; val top = 150 * d; val bottom = size.height - 120 * d
        val spanX = (maxX - minX).takeIf { it > 1e-9 } ?: 1e-9
        val spanY = (maxY - minY).takeIf { it > 1e-9 } ?: 1e-9
        val scale = min((right - left) / spanX, (bottom - top) / spanY)
        val ox = left + ((right - left) - spanX * scale) / 2
        val oy = top + ((bottom - top) - spanY * scale) / 2
        fun P(lat: Double, lng: Double) = Offset((ox + (lng * k - minX) * scale).toFloat(), (oy + (maxY - lat) * scale).toFloat())

        // ── 경로 ──
        val route = points.map { P(it[0], it[1]) }
        val path = Path().apply {
            route.firstOrNull()?.let { moveTo(it.x, it.y) }
            route.drop(1).forEach { lineTo(it.x, it.y) }
        }
        val pm = PathMeasure().apply { setPath(path, false) }
        val length = if (route.size >= 2) pm.length else 0f
        // 글자가 경로를 덮지 않게 하려고 경로를 따라 촘촘히 찍어 둔 점
        val samples = if (length > 0f) List((length / (8 * d)).toInt() + 1) { pm.getPosition(min(it * 8 * d, length)) } else route
        if (length > 0f) {
            fun line(w: Float) = Stroke(w * d, cap = StrokeCap.Round, join = StrokeJoin.Round)
            drawPath(path, Color.White, style = line(22f))
            drawPath(path, NColor.indigo.copy(alpha = .16f), style = line(16f))
            val a = route.first(); val b = route.last()
            // 출발(남색)에서 도착(파랑)으로 옅어지는 선. 출발과 도착이 같은 자리면 한 색으로 그린다.
            if ((a - b).getDistance() > 4 * d) drawPath(path, Brush.linearGradient(listOf(NColor.indigo, NColor.indigoHi), start = a, end = b), style = line(7f))
            else drawPath(path, NColor.indigo, style = line(7f))
            // 진행 방향
            var at = 46 * d
            while (at < length - 30 * d) {
                val p = pm.getPosition(at); val t = pm.getTangent(at)
                val ang = atan2(t.y, t.x)
                val tip = Offset(p.x + 2.4f * d * cos(ang), p.y + 2.4f * d * sin(ang))
                for (side in listOf(.62f, -.62f)) {
                    val back = Offset(tip.x - 4.4f * d * cos(ang + side), tip.y - 4.4f * d * sin(ang + side))
                    drawLine(Color.White, tip, back, 1.7f * d, cap = StrokeCap.Round)
                }
                at += 78 * d
            }
        }

        // ── 일정 거리마다의 눈금 ──
        val cum = DoubleArray(points.size)
        for (i in 1 until points.size) cum[i] = cum[i - 1] + DistanceAccumulator.haversineMeters(points[i - 1][0], points[i - 1][1], points[i][0], points[i][1])
        val total = cum.lastOrNull() ?: 0.0
        val step = when {
            total <= 2_500 -> 500.0
            total <= 6_000 -> 1_000.0
            total <= 16_000 -> 2_000.0
            total <= 45_000 -> 5_000.0
            total <= 120_000 -> 10_000.0
            else -> 50_000.0
        }
        fun along(m: Double): Offset {
            val i = cum.indexOfFirst { it >= m }.coerceAtLeast(1)
            val seg = cum[i] - cum[i - 1]
            val t = if (seg <= 0) 0.0 else (m - cum[i - 1]) / seg
            return P(points[i - 1][0] + (points[i][0] - points[i - 1][0]) * t, points[i - 1][1] + (points[i][1] - points[i - 1][1]) * t)
        }
        val ticks = mutableListOf<Pair<Offset, String>>()
        var m = step
        while (points.size >= 2 && m < total - step * .4 && ticks.size < 12) {
            ticks += along(m) to (if (m < 1_000) "${m.toInt()}m" else if (m % 1_000 == 0.0) "${(m / 1_000).toInt()}km" else "%.1fkm".format(m / 1_000))
            m += step
        }
        ticks.forEach { (p, _) -> drawCircle(Color.White, 5 * d, p); drawCircle(NColor.indigo, 5 * d, p, style = Stroke(1.6f * d)) }

        // ── 쉼터, 감지 위치, 출발·도착 ──
        val stopAt = stops.map { P(it.latitude, it.longitude) }
        stopAt.forEach { p ->
            val pin = Path().apply {
                val x = p.x - 14 * d; val y = p.y - 36 * d
                moveTo(x + 14 * d, y + 36 * d)
                cubicTo(x + 14 * d, y + 36 * d, x + 2 * d, y + 25.6f * d, x + 2 * d, y + 15.6f * d)
                cubicTo(x + 2 * d, y + 9 * d, x + 7.4f * d, y + 3.6f * d, x + 14 * d, y + 3.6f * d)
                cubicTo(x + 20.6f * d, y + 3.6f * d, x + 26 * d, y + 9 * d, x + 26 * d, y + 15.6f * d)
                cubicTo(x + 26 * d, y + 25.6f * d, x + 14 * d, y + 36 * d, x + 14 * d, y + 36 * d)
                close()
            }
            drawPath(pin, NColor.indigo); drawPath(pin, Color.White, style = Stroke(2.5f * d))
            drawCircle(Color.White, 4.5f * d, Offset(p.x, p.y - 20.4f * d))
        }
        val pinAt = pins.map { P(it.latitude, it.longitude) }
        pins.forEachIndexed { i, pin ->
            // 졸음은 옅은 빨간 테를 둘러 부주의와 한눈에 구분한다
            if (pin.color == NColor.danger) drawCircle(Color(0x24F62E24), 22 * d, pinAt[i])
            drawCircle(Color.White, 13 * d, pinAt[i]); drawCircle(pin.color, 10 * d, pinAt[i])
        }
        val startAt = route.firstOrNull()
        val endAt = route.lastOrNull()?.takeIf { route.size >= 2 }
        startAt?.let { drawCircle(Color.White, 11 * d, it); drawCircle(NColor.indigo, 11 * d, it, style = Stroke(5 * d)) }
        endAt?.let { drawCircle(Color.White, 13 * d, it); drawCircle(NColor.ink, 11 * d, it) }

        // ── 북쪽 표시와 축척 ──
        val north = Offset(size.width - 44 * d, 50 * d)
        drawCircle(Color.White.copy(alpha = .92f), 20 * d, north)
        drawCircle(Color(0x1F131417), 20 * d, north, style = Stroke(1 * d))
        drawPath(Path().apply { moveTo(north.x, north.y - 13 * d); lineTo(north.x - 5.5f * d, north.y - 2 * d); lineTo(north.x + 5.5f * d, north.y - 2 * d); close() }, NColor.indigo)
        tm.measure("N", S.t(12, 14, 700, NColor.ink2)).let { drawText(it, topLeft = Offset(north.x - it.size.width / 2, north.y)) }
        val pxPerMeter = (scale / 111_320.0).toFloat()
        val barMeters = listOf(50_000, 20_000, 10_000, 5_000, 2_000, 1_000, 500, 200, 100, 50).firstOrNull { it * pxPerMeter <= 120 * d } ?: 50
        val barWidth = barMeters * pxPerMeter
        val bar = Offset(size.width - 28 * d - barWidth, size.height - 34 * d)
        drawLine(NColor.ink3, bar, Offset(bar.x + barWidth, bar.y), 2 * d)
        for (x in listOf(bar.x, bar.x + barWidth)) drawLine(NColor.ink3, Offset(x, bar.y - 5 * d), Offset(x, bar.y + 1 * d), 2 * d)
        tm.measure(if (barMeters < 1_000) "${barMeters}m" else "${barMeters / 1_000}km", S.t(13, 16, 600, NColor.ink3)).let {
            drawText(it, topLeft = Offset(bar.x + barWidth / 2 - it.size.width / 2, bar.y - 8 * d - it.size.height))
        }

        // ── 글자: 이미 있는 것(안내 상자, 범례, 축척, 북쪽 표시, 핀)과 경로를 피해 놓는다 ──
        val taken = mutableListOf(
            Rect(24 * d, 20 * d, 344 * d, 130 * d),
            Rect(24 * d, size.height - 84 * d, 250 * d, size.height - 20 * d),
            Rect(north.x - 26 * d, north.y - 26 * d, north.x + 26 * d, north.y + 26 * d),
            Rect(bar.x - 10 * d, bar.y - 32 * d, size.width, size.height),
        )
        (listOfNotNull(startAt, endAt) + pinAt).forEach { taken += Rect(it.x - 15 * d, it.y - 15 * d, it.x + 15 * d, it.y + 15 * d) }
        stopAt.forEach { taken += Rect(it.x - 15 * d, it.y - 38 * d, it.x + 15 * d, it.y + 4 * d) }
        ticks.forEach { (p, _) -> taken += Rect(p.x - 7 * d, p.y - 7 * d, p.x + 7 * d, p.y + 7 * d) }

        /** [anchor] 옆에 글자를 놓는다. [optional] 이면 바로 옆 자리가 비어 있을 때만 놓는다(거리 눈금). */
        fun label(anchor: Offset, text: String, style: TextStyle, optional: Boolean = false) {
            val r = tm.measure(text, style)
            val w = r.size.width + 18 * d; val h = r.size.height + 8 * d
            val gap = (if (optional) 12 else 20) * d
            var spot: Rect? = null
            search@ for (row in if (optional) listOf(0) else listOf(0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6)) {
                for (side in listOf(1, -1)) {
                    val shift = gap + abs(row) * 6 * d
                    val x = if (side > 0) anchor.x + shift else anchor.x - shift - w
                    val y = anchor.y - h / 2 + row * (h + 6 * d)
                    val rect = Rect(x, y, x + w, y + h)
                    if (rect.left < 8 * d || rect.right > size.width - 8 * d || rect.top < 8 * d || rect.bottom > size.height - 8 * d) continue
                    if (taken.any { it.overlaps(rect) }) continue
                    val wide = rect.inflate(5 * d)
                    if (samples.any { wide.contains(it) }) continue
                    spot = rect
                    break@search
                }
            }
            val rect = spot ?: if (optional) return else {
                // 빈자리가 없으면 화면 안으로만 맞춰 오른쪽에 놓는다
                val x = (anchor.x + gap).coerceAtMost(size.width - w - 8 * d)
                Rect(x, anchor.y - h / 2, x + w, anchor.y + h / 2)
            }
            taken += rect
            val near = Offset(if (rect.left > anchor.x) rect.left else rect.right, rect.center.y)
            if (abs(rect.center.y - anchor.y) > 2 * d) drawLine(Color(0x73131417), anchor, near, 1.2f * d)
            drawRoundRect(Color.White.copy(alpha = .95f), rect.topLeft, rect.size, CornerRadius(10 * d))
            drawRoundRect(Color(0x1F131417), rect.topLeft, rect.size, CornerRadius(10 * d), style = Stroke(1 * d))
            drawText(r, topLeft = Offset(rect.left + 9 * d, rect.top + 4 * d))
        }
        startAt?.let { label(it, startLabel, S.t(15, 18, 600, NColor.ink)) }
        endAt?.let { label(it, endLabel, S.t(15, 18, 600, NColor.ink)) }
        stops.forEachIndexed { i, stop -> label(Offset(stopAt[i].x, stopAt[i].y - 18 * d), if (stop.visited) "${stop.name} · 휴식함" else stop.name, S.t(15, 18, 600, NColor.indigo)) }
        pins.forEachIndexed { i, pin -> label(pinAt[i], pin.label, S.t(15, 18, 500, NColor.ink2)) }
        ticks.forEach { (p, text) -> label(p, text, S.t(13, 16, 600, NColor.ink3), optional = true) }
    }
}
