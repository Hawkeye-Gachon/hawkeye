package com.contest.DriverGuard.restplace.service;

import java.util.List;

public final class GeoUtils {

    private static final double EARTH_RADIUS_METERS = 6_371_008.8;

    private GeoUtils() {
    }

    /** 두 좌표 사이 직선 거리(m, haversine) */
    public static double distanceMeters(double lat1, double lng1, double lat2, double lng2) {
        double phi1 = Math.toRadians(lat1);
        double phi2 = Math.toRadians(lat2);
        double dPhi = Math.toRadians(lat2 - lat1);
        double dLambda = Math.toRadians(lng2 - lng1);
        double a = Math.sin(dPhi / 2) * Math.sin(dPhi / 2)
                + Math.cos(phi1) * Math.cos(phi2) * Math.sin(dLambda / 2) * Math.sin(dLambda / 2);
        return 2 * EARTH_RADIUS_METERS * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    }

    /** 1 → 2 방위각 (북쪽 0도, 시계 방향 0~360) */
    public static double bearingDegrees(double lat1, double lng1, double lat2, double lng2) {
        double phi1 = Math.toRadians(lat1);
        double phi2 = Math.toRadians(lat2);
        double dLambda = Math.toRadians(lng2 - lng1);
        double y = Math.sin(dLambda) * Math.cos(phi2);
        double x = Math.cos(phi1) * Math.sin(phi2) - Math.sin(phi1) * Math.cos(phi2) * Math.cos(dLambda);
        return (Math.toDegrees(Math.atan2(y, x)) + 360) % 360;
    }

    /** 진행 방향 기준 상대 각도 (-180~180, 양수면 오른쪽) */
    public static double relativeAngle(double targetBearing, double heading) {
        return ((targetBearing - heading + 540) % 360) - 180;
    }

    /**
     * 경로(path, [경도, 위도] 목록)가 지점(lat, lng)에 가장 가까이 지나는 곳.
     *
     * @return {경로까지 거리(m), 출발점부터 그 곳까지 경로를 따라 간 거리(m)}
     */
    public static double[] closestOnPath(List<double[]> path, double lat, double lng) {
        return closestOnPath(path, lat, lng, Double.MAX_VALUE);
    }

    /**
     * 경로가 지점에 nearRadius 안까지 다가갔다가, 그 뒤 다시 awayMargin 이상 멀어진 적이 있는지 (= 지나쳤다가 되돌아옴).
     * 반대 차선 휴게소로 가는 경로는 휴게소 옆을 지나 다음 IC 까지 갔다가 돌아오므로 해당됨.
     *
     * @return 지나쳤다면 가장 가까이 지나간 지점까지 경로를 따라 간 거리(m), 아니면 -1
     */
    public static double passesThenReturns(List<double[]> path, double lat, double lng,
                                           double nearRadius, double awayMargin) {
        double kx = Math.cos(Math.toRadians(lat)) * 111_320;
        double ky = 110_540;
        double min = Double.MAX_VALUE;
        double minAlong = 0;
        double along = 0;
        for (int i = 0; i < path.size(); i++) {
            double[] p = path.get(i);
            if (i > 0) {
                double[] prev = path.get(i - 1);
                along += Math.hypot((p[0] - prev[0]) * kx, (p[1] - prev[1]) * ky);
            }
            double d = Math.hypot((p[0] - lng) * kx, (p[1] - lat) * ky);
            if (d < min) {
                min = d;
                minAlong = along;
            } else if (min <= nearRadius && d >= min + awayMargin) {
                return minAlong;
            }
        }
        return -1;
    }

    /**
     * 경로 시작/끝 부분의 진행 방향 (북쪽 0도, 시계 방향).
     *
     * @param fromEnd true 면 마지막 meters 구간(도착 직전), false 면 처음 meters 구간
     * @return 구간 양 끝을 잇는 방위각. 경로가 너무 짧으면 NaN
     */
    public static double pathBearing(List<double[]> path, double meters, boolean fromEnd) {
        if (path.size() < 2) {
            return Double.NaN;
        }
        int step = fromEnd ? -1 : 1;
        int start = fromEnd ? path.size() - 1 : 0;
        double[] first = path.get(start);
        double acc = 0;
        int i = start;
        while (i + step >= 0 && i + step < path.size() && acc < meters) {
            acc += distanceMeters(path.get(i)[1], path.get(i)[0], path.get(i + step)[1], path.get(i + step)[0]);
            i += step;
        }
        if (acc < meters / 3) {
            return Double.NaN;
        }
        double[] other = path.get(i);
        return fromEnd
                ? bearingDegrees(other[1], other[0], first[1], first[0])
                : bearingDegrees(first[1], first[0], other[1], other[0]);
    }

    /**
     * 도착 직전 tailMeters 구간에서, 기준 방향과 reversedAngle 넘게 반대로 진행한 길이(m).
     * 기준 방향 = 그보다 앞 구간(도착 tailMeters ~ tailMeters+refMeters 전)의 진행 방향.
     * 휴게소 직전에서 연결로를 돌아 거꾸로 들어가는(회차) 모양을 찾는 데 씀.
     */
    public static double reversedLengthNearEnd(List<double[]> path, double tailMeters, double refMeters,
                                               double reversedAngle) {
        int n = path.size();
        if (n < 3) {
            return 0;
        }
        // 끝에서부터 누적 거리
        double[] fromEnd = new double[n];
        for (int i = n - 2; i >= 0; i--) {
            fromEnd[i] = fromEnd[i + 1] + distanceMeters(path.get(i)[1], path.get(i)[0], path.get(i + 1)[1], path.get(i + 1)[0]);
        }
        int refStart = -1;
        int refEnd = -1;
        for (int i = 0; i < n; i++) {
            if (refStart < 0 && fromEnd[i] <= tailMeters + refMeters) {
                refStart = i;
            }
            if (fromEnd[i] <= tailMeters) {
                refEnd = i;
                break;
            }
        }
        if (refStart < 0 || refEnd <= refStart) {
            return 0;   // 경로가 너무 짧아 기준 방향을 못 구함
        }
        double reference = bearingDegrees(path.get(refStart)[1], path.get(refStart)[0], path.get(refEnd)[1], path.get(refEnd)[0]);

        double reversed = 0;
        for (int i = refEnd; i + 1 < n; i++) {
            double len = fromEnd[i] - fromEnd[i + 1];
            if (len <= 0) {
                continue;
            }
            double b = bearingDegrees(path.get(i)[1], path.get(i)[0], path.get(i + 1)[1], path.get(i + 1)[0]);
            if (Math.abs(relativeAngle(b, reference)) > reversedAngle) {
                reversed += len;
            }
        }
        return reversed;
    }

    /** closestOnPath 와 같지만 출발점부터 maxAlong(m) 까지의 구간만 봄 */
    public static double[] closestOnPath(List<double[]> path, double lat, double lng, double maxAlong) {
        // 지점 주변 평면 근사 (수십 km 범위에서 오차 무시 가능)
        double kx = Math.cos(Math.toRadians(lat)) * 111_320;
        double ky = 110_540;
        double best = Double.MAX_VALUE;
        double bestAlong = 0;
        double along = 0;
        for (int i = 0; i + 1 < path.size() && along <= maxAlong; i++) {
            double ax = (path.get(i)[0] - lng) * kx;
            double ay = (path.get(i)[1] - lat) * ky;
            double dx = (path.get(i + 1)[0] - lng) * kx - ax;
            double dy = (path.get(i + 1)[1] - lat) * ky - ay;
            double len2 = dx * dx + dy * dy;
            double len = Math.sqrt(len2);
            double t = len2 == 0 ? 0 : Math.max(0, Math.min(1, -(ax * dx + ay * dy) / len2));
            double d = Math.hypot(ax + t * dx, ay + t * dy);
            if (d < best) {
                best = d;
                bestAlong = along + t * len;
            }
            along += len;
        }
        return new double[]{best, bestAlong};
    }
}
