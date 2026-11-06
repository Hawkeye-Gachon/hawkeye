package com.contest.DriverGuard.report.dao;

import com.contest.DriverGuard.drive.dto.DriveDto;
import com.contest.DriverGuard.report.dto.ReportDto;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.time.OffsetDateTime;
import java.util.List;

@Mapper
public interface ReportDao {
    /** [from, to) 구간의 시간대 통계 행 (종료된 주행만 행이 있음) */
    List<ReportDto.StatRow> findStats(@Param("userId") Long userId,
                                      @Param("from") OffsetDateTime from,
                                      @Param("to") OffsetDateTime to);

    /** [from, to) 와 관찰 시간이 겹치는 주행, 최신순 */
    List<DriveDto.ListItem> findDrives(@Param("userId") Long userId,
                                       @Param("from") OffsetDateTime from,
                                       @Param("to") OffsetDateTime to);

    /** 가장 최근 종료 주행의 시작 시각. 없으면 null */
    OffsetDateTime findLatestStartedAt(@Param("userId") Long userId);
}
