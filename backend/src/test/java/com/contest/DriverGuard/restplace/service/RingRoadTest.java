package com.contest.DriverGuard.restplace.service;

import org.junit.jupiter.api.Test;

import static com.contest.DriverGuard.restplace.service.RingRoad.Rotation.CCW;
import static com.contest.DriverGuard.restplace.service.RingRoad.Rotation.CW;
import static org.assertj.core.api.Assertions.assertThat;

/**
 * 서울외곽순환 회전 판정. 좌표는 rest_places 실제 값.
 * 정답은 카카오 실제 경로(시계·반시계)에서 각 장소가 어느 방향 차선의 오른쪽에 붙어 있는지로 확인한 값 (2026-11-06)
 */
class RingRoadTest {

    private final RingRoad ring = RingRoad.SEOUL;

    @Test
    void 노선명() {
        assertThat(RingRoad.of("수도권제1순환")).isSameAs(ring);   // 카카오 도로명 "수도권제1순환고속도로"
        assertThat(RingRoad.of("서울외곽")).isSameAs(ring);        // 졸음쉼터 데이터 "서울외곽선"
        assertThat(RingRoad.of("경부")).isNull();
    }

    @Test
    void 휴게소는_이름의_방향_지명_쪽으로_짧게_도는_회전() {
        // 같은 "일산"이라도 원의 어느 쪽이냐에 따라 반대
        assertThat(ring.restAreaRotation(37.62798, 127.13909, "일산")).isEqualTo(CCW);   // 구리(일산): 동쪽, 북쪽으로
        assertThat(ring.restAreaRotation(37.39338, 127.02272, "일산")).isEqualTo(CW);    // 의왕(일산): 남쪽, 서쪽으로
        assertThat(ring.restAreaRotation(37.5, 127.0, "부산")).isNull();
    }

    @Test
    void 졸음쉼터는_기점_종점이_노선_진행_방향() {
        // 위치와 상관없이 판교→일산 = 시계 (동쪽의 구리남양주도 시계: 남쪽 차선)
        assertThat(ring.shelterRotation("판교", "일산")).isEqualTo(CW);
        assertThat(ring.shelterRotation("일산", "판교")).isEqualTo(CCW);
        // 성남 "퇴계원기점 + 구리종점" (순환도로에서 약 1km 떨어진 좌표) → 판정 안 함
        assertThat(ring.shelterRotation("퇴계원", "구리")).isNull();
        assertThat(ring.shelterRotation(null, "일산")).isNull();
    }

    @Test
    void 차가_도는_방향() {
        // 남쪽 구간(의왕)에서 서쪽으로 = 시계, 동쪽으로 = 반시계
        assertThat(ring.travelRotation(37.3845, 126.9913, 247)).isEqualTo(CW);
        assertThat(ring.travelRotation(37.3803, 126.9806, 65)).isEqualTo(CCW);
        // 동쪽 구간(구리)에서 북쪽으로 = 반시계
        assertThat(ring.travelRotation(37.6099, 127.1442, 341)).isEqualTo(CCW);
    }

    @Test
    void 앞에_있는_곳만_둘레_거리로() {
        // 의왕 부근에서 시계 방향(서쪽): 시흥 쉼터는 앞, 청계 쉼터는 뒤
        double ahead = ring.aheadMeters(37.3845, 126.9913, 37.45071, 126.80384, CW);
        assertThat(ahead).isBetween(15_000.0, 30_000.0);
        assertThat(ring.aheadMeters(37.3845, 126.9913, 37.39327, 127.02231, CW)).isEqualTo(-1);
    }
}
