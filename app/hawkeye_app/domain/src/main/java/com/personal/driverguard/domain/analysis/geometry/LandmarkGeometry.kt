package com.personal.driverguard.domain.analysis.geometry

import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

data class Point2(val x: Double, val y: Double) {
    val isFinite: Boolean get() = x.isFinite() && y.isFinite()
}

data class Point3(val x: Double, val y: Double, val z: Double)

data class Bounds(val left: Double, val top: Double, val right: Double, val bottom: Double) {
    val width: Double get() = right - left
    val height: Double get() = bottom - top
}

/**
 * 한 얼굴의 랜드마크를 '회전 보정된 분석 이미지'의 픽셀 좌표로 담는다.
 * 정규화 좌표 x 에는 이미지 너비를, y 에는 높이를 곱한다. 종횡비가 다른 이미지에서 정규화 좌표 거리를 그대로 쓰지 않기 위해서다.
 * 정규화 z 는 담지 않는다(실제 깊이·픽셀 단위가 아니므로 2D 거리와 섞지 않는다).
 */
class LandmarkPoints(
    val imageWidth: Int,
    val imageHeight: Int,
    private val xs: DoubleArray,
    private val ys: DoubleArray,
) {
    init {
        require(xs.size == ys.size) { "x·y 개수가 다릅니다" }
    }

    val size: Int get() = xs.size

    operator fun get(index: Int): Point2 = Point2(xs[index], ys[index])

    fun allFinite(): Boolean = xs.all { it.isFinite() } && ys.all { it.isFinite() }

    fun bounds(): Bounds {
        var l = Double.POSITIVE_INFINITY
        var t = Double.POSITIVE_INFINITY
        var r = Double.NEGATIVE_INFINITY
        var b = Double.NEGATIVE_INFINITY
        for (i in xs.indices) {
            l = min(l, xs[i]); r = max(r, xs[i])
            t = min(t, ys[i]); b = max(b, ys[i])
        }
        return Bounds(l, t, r, b)
    }

    companion object {
        /** 정규화 좌표(0~1) 배열을 픽셀 좌표로 바꾼다. [normalized] 는 x0,y0,x1,y1,… 순서. */
        fun fromNormalized(normalized: FloatArray, imageWidth: Int, imageHeight: Int): LandmarkPoints {
            val n = normalized.size / 2
            return LandmarkPoints(
                imageWidth,
                imageHeight,
                DoubleArray(n) { normalized[it * 2] * imageWidth.toDouble() },
                DoubleArray(n) { normalized[it * 2 + 1] * imageHeight.toDouble() },
            )
        }
    }
}

/** 기하학 계산 결과. 분모가 너무 작거나 좌표가 유한하지 않으면 value 가 null 이다. */
data class RatioResult(val value: Double?, val denominatorPx: Double)

object LandmarkGeometry {

    fun distance(a: Point2, b: Point2): Double = hypot(a.x - b.x, a.y - b.y)

    /**
     * EAR = (|p2−p6| + |p3−p5|) / (2·|p1−p4|)  (Soukupová & Čech, 2016)
     * [eye] 는 p1..p6 순서의 인덱스. 눈 너비(|p1−p4|)가 [minWidthPx] 보다 작으면 무효.
     */
    fun eyeAspectRatio(points: LandmarkPoints, eye: IntArray, minWidthPx: Double): RatioResult {
        require(eye.size == 6) { "EAR 은 6점이 필요합니다" }
        val p = eye.map { points[it] }
        if (p.any { !it.isFinite }) return RatioResult(null, Double.NaN)
        val width = distance(p[0], p[3])
        if (!(width >= minWidthPx)) return RatioResult(null, width)
        val vertical = distance(p[1], p[5]) + distance(p[2], p[4])
        val ear = vertical / (2.0 * width)
        return RatioResult(ear.takeIf { it.isFinite() }, width)
    }

    /**
     * 입 벌림 비율 = 입술 안쪽 위·아래 점 쌍들의 세로 거리 평균 / 입꼬리 사이 가로 너비.
     * 점 하나가 흔들려도 비율이 튀지 않도록 세 쌍을 평균한다.
     */
    fun mouthOpenRatio(
        points: LandmarkPoints,
        verticalPairs: List<Pair<Int, Int>>,
        rightCorner: Int,
        leftCorner: Int,
        minWidthPx: Double,
    ): RatioResult {
        val r = points[rightCorner]
        val l = points[leftCorner]
        if (!r.isFinite || !l.isFinite) return RatioResult(null, Double.NaN)
        val width = distance(r, l)
        if (!(width >= minWidthPx)) return RatioResult(null, width)
        val gaps = verticalPairs.map { (top, bottom) -> distance(points[top], points[bottom]) }
        if (gaps.any { !it.isFinite() }) return RatioResult(null, width)
        return RatioResult(gaps.average() / width, width)
    }

    /** 경계 상자가 이미지 밖으로 나간 넓이 비율(0 이면 완전히 안쪽) */
    fun outsideRatio(bounds: Bounds, imageWidth: Int, imageHeight: Int): Double {
        val area = bounds.width * bounds.height
        if (!(area > 0)) return 1.0
        val iw = (min(bounds.right, imageWidth.toDouble()) - max(bounds.left, 0.0)).coerceAtLeast(0.0)
        val ih = (min(bounds.bottom, imageHeight.toDouble()) - max(bounds.top, 0.0)).coerceAtLeast(0.0)
        return 1.0 - (iw * ih) / area
    }
}
