package com.personal.driverguard.ui.state

import android.content.Context
import com.personal.driverguard.domain.engine.DistanceAccumulator
import kotlinx.serialization.Serializable
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.builtins.serializer
import kotlinx.serialization.json.Json
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin

data class LatLng(val lat: Double, val lng: Double)

/** 경로 위 쉼터. [routeMeters] 는 경로 시작(양재IC)에서 쉼터 진입 지점까지 거리. */
data class RestSpot(val placeId: Long, val name: String, val type: String, val lat: Double, val lng: Double, val routeMeters: Double)

/**
 * 시연·시험용 경부고속도로 부산 방향 경로(양재IC → 구서IC, 약 416km)와 그 위의 휴게소·졸음쉼터.
 *
 * 데이터: assets/demo/gyeongbu_busan.json(경로 [[위도, 경도], ...]), gyeongbu_busan_rest.json(쉼터).
 * 출처는 OpenStreetMap(© OpenStreetMap contributors, ODbL)의 경부고속도로(ref=1) 부산 방향 차로와 휴게소·졸음쉼터 위치다.
 * 방향이 표시되지 않은 일부 쉼터는 실제로는 반대 방향일 수 있다(시연용).
 */
class DemoRoute(val points: List<LatLng>, rests: List<RawRest>) {

    @Serializable
    class RawRest(val name: String, val type: String, val lat: Double, val lon: Double)

    private val cum = DoubleArray(points.size).also { c ->
        for (i in 1 until points.size) c[i] = c[i - 1] + dist(points[i - 1], points[i])
    }

    val lengthMeters: Double get() = cum.last()

    val rests: List<RestSpot> = rests.mapIndexed { i, r ->
        RestSpot(PLACE_ID_BASE + i, r.name, r.type, r.lat, r.lon, cum[nearestIndex(r.lat, r.lon)])
    }.sortedBy { it.routeMeters }

    fun nearestIndex(lat: Double, lng: Double): Int {
        var best = 0
        var bestD = Double.MAX_VALUE
        val p = LatLng(lat, lng)
        for (i in points.indices) {
            val d = dist(points[i], p)
            if (d < bestD) { bestD = d; best = i }
        }
        return best
    }

    /** 경로에서 가장 가까운 지점까지 시작부터의 거리와, 그 지점까지 떨어진 거리 */
    fun project(lat: Double, lng: Double): Pair<Double, Double> {
        val i = nearestIndex(lat, lng)
        return cum[i] to dist(points[i], LatLng(lat, lng))
    }

    fun positionAt(meters: Double): LatLng {
        if (meters <= 0) return points.first()
        if (meters >= lengthMeters) return points.last()
        var lo = 0
        var hi = cum.size - 1
        while (hi - lo > 1) {
            val mid = (lo + hi) / 2
            if (cum[mid] <= meters) lo = mid else hi = mid
        }
        val seg = cum[hi] - cum[lo]
        val t = if (seg == 0.0) 0.0 else (meters - cum[lo]) / seg
        return lerp(points[lo], points[hi], t)
    }

    fun bearingAt(meters: Double): Float = bearing(positionAt(meters - 30), positionAt(meters + 30))

    /** [fromMeters] 보다 [minAheadMeters] 이상 앞에 있는 첫 쉼터 */
    fun nextRest(fromMeters: Double, minAheadMeters: Double = MIN_AHEAD_M): RestSpot? = rests.firstOrNull { it.routeMeters >= fromMeters + minAheadMeters }

    fun rest(placeId: Long): RestSpot? = rests.firstOrNull { it.placeId == placeId }

    companion object {
        /** 시연 쉼터 ID 시작값(서버 ID 와 겹치지 않게) */
        const val PLACE_ID_BASE = 70_000L
        const val MIN_AHEAD_M = 1_000.0
        private const val ROUTE_ASSET = "demo/gyeongbu_busan.json"
        private const val REST_ASSET = "demo/gyeongbu_busan_rest.json"
        private val json = Json { ignoreUnknownKeys = true }

        @Volatile private var cached: DemoRoute? = null

        fun load(context: Context): DemoRoute = cached ?: synchronized(this) {
            cached ?: parse(
                context.assets.open(ROUTE_ASSET).bufferedReader().use { it.readText() },
                context.assets.open(REST_ASSET).bufferedReader().use { it.readText() },
            ).also { cached = it }
        }

        fun parse(routeJson: String, restJson: String): DemoRoute {
            val pts = json.decodeFromString(ListSerializer(ListSerializer(Double.serializer())), routeJson).map { LatLng(it[0], it[1]) }
            val rests = json.decodeFromString(ListSerializer(RawRest.serializer()), restJson)
            return DemoRoute(pts, rests)
        }

        fun dist(a: LatLng, b: LatLng): Double = DistanceAccumulator.haversineMeters(a.lat, a.lng, b.lat, b.lng)

        fun lerp(a: LatLng, b: LatLng, t: Double) = LatLng(a.lat + (b.lat - a.lat) * t, a.lng + (b.lng - a.lng) * t)

        fun bearing(a: LatLng, b: LatLng): Float {
            val p1 = Math.toRadians(a.lat)
            val p2 = Math.toRadians(b.lat)
            val dl = Math.toRadians(b.lng - a.lng)
            val y = sin(dl) * cos(p2)
            val x = cos(p1) * sin(p2) - sin(p1) * cos(p2) * cos(dl)
            return ((Math.toDegrees(atan2(y, x)) + 360) % 360).toFloat()
        }
    }
}
