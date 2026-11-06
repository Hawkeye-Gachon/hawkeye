package com.contest.DriverGuard.restplace.dao;

import com.contest.DriverGuard.restplace.dto.RestPlaceDto;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.time.OffsetDateTime;
import java.util.List;

@Mapper
public interface RestPlaceDao {

    /** 주변 조회용: 활성 상태이고 좌표가 있는 곳 전체 */
    List<RestPlaceDto.RestPlace> findAllActive();

    /** (place_type, source_key) 기준 INSERT, 이미 있으면 UPDATE + active = TRUE */
    int upsert(@Param("row") RestPlaceDto.RestPlace row,
               @Param("syncedAt") OffsetDateTime syncedAt);

    /** 이번 수집(syncedAt)에 포함되지 않은 곳을 비활성화 */
    int deactivateNotSynced(@Param("placeType") RestPlaceDto.RestPlace.PlaceType placeType,
                            @Param("syncedAt") OffsetDateTime syncedAt);
}
