package com.contest.DriverGuard.restplace.service;

import com.contest.DriverGuard.restplace.dao.RestPlaceDao;
import com.contest.DriverGuard.restplace.dto.RestPlaceDto.RestPlace;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 활성 쉼터 목록(약 450건)을 메모리에 보관. 주변 조회 때마다 DB 를 치지 않기 위함.
 * 앱 시작 시 + 수집이 끝날 때 reload().
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class RestPlaceCache {

    private final RestPlaceDao restPlaceDao;

    private volatile Snapshot snapshot = new Snapshot(List.of(), Map.of());

    /**
     * @param restAreaDirections 노선명 → 그 노선 휴게소들의 방향 표기 모음 (예: 경부선 → {부산, 서울})
     */
    private record Snapshot(List<RestPlace> places, Map<String, Set<String>> restAreaDirections) {
    }

    @EventListener(ApplicationReadyEvent.class)
    public void reload() {
        List<RestPlace> places = List.copyOf(restPlaceDao.findAllActive());
        Map<String, Set<String>> directions = new HashMap<>();
        for (RestPlace p : places) {
            if (p.getPlaceType() == RestPlace.PlaceType.REST_AREA && p.getDirection() != null) {
                directions.computeIfAbsent(p.getRouteName(), k -> new HashSet<>()).add(p.getDirection());
            }
        }
        directions.replaceAll((k, v) -> Set.copyOf(v));
        snapshot = new Snapshot(places, Map.copyOf(directions));
        log.info("rest places loaded: {}", places.size());
    }

    public List<RestPlace> getAll() {
        return snapshot.places();
    }

    /** 노선의 휴게소 방향 표기 모음 (없으면 빈 집합) */
    public Set<String> restAreaDirections(String routeName) {
        return snapshot.restAreaDirections().getOrDefault(routeName, Set.of());
    }
}
