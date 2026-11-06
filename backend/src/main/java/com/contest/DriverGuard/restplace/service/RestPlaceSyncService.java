package com.contest.DriverGuard.restplace.service;

import com.contest.DriverGuard.restplace.dto.RestPlaceDto.RestPlace;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.scheduling.annotation.Async;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * 졸음쉼터(공공데이터포털 API) + 휴게소(도로공사 API) 주기 수집.
 * driverguard.rest-place.sync-enabled=true 일 때만 동작 (없거나 false 면 이 빈 자체가 없음 → 수집 안 함).
 */
@Slf4j
@Service
@RequiredArgsConstructor
@ConditionalOnProperty(name = "driverguard.rest-place.sync-enabled", havingValue = "true")
public class RestPlaceSyncService {

    private final DrowsyShelterApiReader drowsyShelterApiReader;
    private final ExRestAreaApiReader exRestAreaApiReader;
    private final RestPlaceWriter restPlaceWriter;
    private final RestPlaceCache restPlaceCache;

    /** 처음 띄운 서버처럼 DB 가 비어 있으면 다음 주기까지 기다리지 않고 바로 한 번 수집 (앱 시작은 막지 않음) */
    @Async
    @EventListener(ApplicationReadyEvent.class)
    public void syncIfEmpty() {
        restPlaceCache.reload();
        if (restPlaceCache.getAll().isEmpty()) {
            log.info("rest_places is empty: initial sync");
            syncAll();
        }
    }

    /**
     * synchronized: 시작 시 수집과 주기 수집이 겹치면 서로의 synced_at 기준으로
     * 상대가 막 반영한 행을 비활성화할 수 있어서 한 번에 하나만 돌게 함.
     */
    @Scheduled(cron = "${driverguard.rest-place.sync-cron}", zone = "Asia/Seoul")
    public synchronized void syncAll() {
        sync(RestPlace.PlaceType.DROWSY_SHELTER, drowsyShelterApiReader::read);
        sync(RestPlace.PlaceType.REST_AREA, exRestAreaApiReader::read);
        restPlaceCache.reload();   // 새로 수집한 데이터를 주변 조회에 바로 반영
    }

    private void sync(RestPlace.PlaceType type, RowSource source) {
        List<RestPlace> rows;
        try {
            rows = source.read();
        } catch (Exception e) {
            log.error("{} sync failed, keep existing data", type, e);
            return;
        }
        if (rows.isEmpty()) {
            log.warn("{} sync returned 0 rows, keep existing data", type);
            return;
        }
        restPlaceWriter.replace(type, rows);
    }

    @FunctionalInterface
    private interface RowSource {
        List<RestPlace> read() throws Exception;
    }
}
