package com.contest.DriverGuard.restplace.service;

import com.contest.DriverGuard.restplace.dao.RestPlaceDao;
import com.contest.DriverGuard.restplace.dto.RestPlaceDto.RestPlace;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import java.time.OffsetDateTime;
import java.util.List;

/**
 * 수집 결과를 한 트랜잭션으로 DB에 반영.
 * RestPlaceSyncService 안에서 호출하면 self-invocation 이라 @Transactional 이 안 걸려서 별도 빈으로 분리.
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class RestPlaceWriter {

    private final RestPlaceDao restPlaceDao;

    @Transactional
    public void replace(RestPlace.PlaceType type, List<RestPlace> rows) {
        OffsetDateTime syncedAt = OffsetDateTime.now();
        for (RestPlace row : rows) {
            restPlaceDao.upsert(row, syncedAt);
        }
        int deactivated = restPlaceDao.deactivateNotSynced(type, syncedAt);
        log.info("{}: {} upserted, {} deactivated", type, rows.size(), deactivated);
    }
}