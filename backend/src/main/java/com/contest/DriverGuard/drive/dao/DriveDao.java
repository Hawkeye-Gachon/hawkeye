package com.contest.DriverGuard.drive.dao;

import com.contest.DriverGuard.drive.dto.DriveDto;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.time.OffsetDateTime;
import java.util.List;

@Mapper
public interface DriveDao {
    /** 같은 사용자의 주행 시작 요청을 순서대로 처리하기 위해 users 행을 잠금 (트랜잭션 끝까지) */
    Long lockUser(@Param("userId") Long userId);

    /** 없으면 null */
    DriveDto.Drive findByClientDriveId(@Param("userId") Long userId, @Param("clientDriveId") String clientDriveId);

    /** 진행 중인 주행 (사용자당 최대 1개). 없으면 null */
    DriveDto.Drive findInProgress(@Param("userId") Long userId);

    /** 본인 주행만. 없거나 다른 사용자 주행이면 null */
    DriveDto.Drive findById(@Param("driveId") Long driveId, @Param("userId") Long userId);

    /** findById + 행 잠금 (종료 업로드 중복 처리 방지) */
    DriveDto.Drive findByIdForUpdate(@Param("driveId") Long driveId, @Param("userId") Long userId);

    /**
     * 종료된 주행 목록, 최근 시작순. cursor(startedAt, driveId) 가 있으면 그 다음부터.
     * 다음 페이지 여부를 알기 위해 limit 은 size + 1 로 넘김
     */
    List<DriveDto.Drive> findEndedPage(@Param("userId") Long userId,
                                       @Param("cursorStartedAt") OffsetDateTime cursorStartedAt,
                                       @Param("cursorDriveId") Long cursorDriveId,
                                       @Param("limit") int limit);

    /** insert 후 생성된 drive_id 가 drive.driveId 에 채워짐 */
    void insertDrive(DriveDto.Drive drive);

    /** IN_PROGRESS → ABANDONED */
    void abandon(@Param("driveId") Long driveId);

    /** 종료 업로드 결과 저장 → ENDED */
    void endDrive(DriveDto.Drive drive);

    /** 사용자의 주행 전부 삭제 (하위 기록은 CASCADE). 탈퇴용 */
    void deleteByUserId(@Param("userId") Long userId);

    // ---------------- 하위 기록 ----------------

    void insertEvents(@Param("driveId") Long driveId, @Param("events") List<DriveDto.Event> events);

    /** rest_places 에서 장소명·종류·좌표를 복사해 저장. 장소가 없거나 좌표가 없으면 0 반환 (저장 안 함) */
    int insertWaypoint(DriveDto.Waypoint waypoint);

    void insertHourlyStats(@Param("stats") List<DriveDto.HourlyStat> stats);

    List<DriveDto.Event> findEvents(@Param("driveId") Long driveId);

    List<DriveDto.Waypoint> findWaypoints(@Param("driveId") Long driveId);
}
