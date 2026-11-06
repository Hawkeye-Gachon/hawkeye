package com.contest.DriverGuard.drive.service;

import com.contest.DriverGuard.drive.dao.DriveDao;
import com.contest.DriverGuard.drive.dto.DriveDto;
import com.contest.DriverGuard.exception.ApiException;
import com.contest.DriverGuard.exception.ErrorCode;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import tools.jackson.core.type.TypeReference;
import tools.jackson.databind.json.JsonMapper;

import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.UUID;

@Service
@RequiredArgsConstructor
public class DriveService {

    /** 기기 시계 오차 허용 */
    private static final Duration CLOCK_SKEW = Duration.ofMinutes(5);
    /** 시작 요청이 실패했던 주행을 나중에 등록할 수 있는 기간 */
    private static final Duration LATE_START_LIMIT = Duration.ofDays(7);
    /** 관찰 1회 최대 길이 */
    private static final Duration MAX_DRIVE = Duration.ofHours(24);
    /** 이동 거리 상한 확인용 속도 (70 m/s ≈ 250 km/h) */
    private static final int MAX_SPEED_MPS = 70;

    private final DriveDao driveDao;
    private final JsonMapper jsonMapper;

    /**
     * 관찰 시작 (주행 등록).
     * - 같은 clientDriveId 재전송 → 기존 주행 그대로 반환 (created = false)
     * - 진행 중 주행보다 늦게 시작 → 기존 주행 ABANDONED, 새 주행 IN_PROGRESS
     * - 진행 중 주행보다 일찍 시작 (실패했던 예전 주행을 나중에 등록) → 기존 주행 유지, 새 주행을 ABANDONED 로 생성
     */
    @Transactional
    public DriveDto.StartResponse start(Long userId, DriveDto.StartRequest request) {
        String clientDriveId = UUID.fromString(request.getClientDriveId()).toString();   // 소문자로 정규화
        OffsetDateTime startedAt = request.getStartedAt();

        // 같은 사용자의 시작 요청은 한 번에 하나씩 (재전송·동시 요청에도 진행 중 1개 보장)
        driveDao.lockUser(userId);

        DriveDto.Drive existing = driveDao.findByClientDriveId(userId, clientDriveId);
        if (existing != null) {
            return DriveDto.StartResponse.builder()
                    .driveId(existing.getDriveId())
                    .status(existing.getStatus())
                    .startedAt(existing.getStartedAt())
                    .created(false)
                    .build();
        }

        OffsetDateTime now = OffsetDateTime.now();
        if (startedAt.isBefore(now.minus(LATE_START_LIMIT)) || startedAt.isAfter(now.plus(CLOCK_SKEW))) {
            throw ApiException.invalidField("startedAt", "startedAt 은 현재로부터 7일 전 ~ 5분 후 사이여야 합니다.");
        }

        DriveDto.Drive.Status status = DriveDto.Drive.Status.IN_PROGRESS;
        Long abandonedDriveId = null;

        DriveDto.Drive inProgress = driveDao.findInProgress(userId);
        if (inProgress != null) {
            if (startedAt.isBefore(inProgress.getStartedAt())) {
                status = DriveDto.Drive.Status.ABANDONED;   // 뒤늦은 등록. 앱이 이어서 종료 API 로 기록을 올림
            } else {
                driveDao.abandon(inProgress.getDriveId());
                abandonedDriveId = inProgress.getDriveId();
            }
        }

        DriveDto.Drive drive = DriveDto.Drive.builder()
                .userId(userId)
                .clientDriveId(clientDriveId)
                .status(status)
                .startedAt(startedAt)
                .build();
        driveDao.insertDrive(drive);

        return DriveDto.StartResponse.builder()
                .driveId(drive.getDriveId())
                .status(status)
                .startedAt(startedAt)
                .abandonedDriveId(abandonedDriveId)
                .created(true)
                .build();
    }

    /**
     * 관찰 종료 · 기록 업로드. 한 트랜잭션에서 감지·경유지·시간대 통계 저장 후 ENDED 로 변경하고 리포트 반환.
     * - 이미 ENDED 면 본문을 무시하고 기존 리포트 반환 (재전송 안전)
     * - IN_PROGRESS, ABANDONED 모두 받음
     * - 주행 기간 밖·중복 항목, 없는 장소는 400 대신 건너뛰고 skipped 로 알려 줌
     */
    @Transactional
    public DriveDto.Report end(Long userId, Long driveId, DriveDto.EndRequest request) {
        DriveDto.Drive drive = driveDao.findByIdForUpdate(driveId, userId);
        if (drive == null) {
            throw new ApiException(ErrorCode.DRIVE_NOT_FOUND);
        }
        if (drive.getStatus() == DriveDto.Drive.Status.ENDED) {
            return buildReport(drive, null);
        }

        OffsetDateTime startedAt = drive.getStartedAt();
        OffsetDateTime endedAt = request.getEndedAt();
        validateEnd(startedAt, endedAt, request.getDistanceMeters(), request.getRoute());

        // 감지 기록: 기간 밖·중복 제외
        List<DriveDto.Event> events = new ArrayList<>();
        Set<String> eventKeys = new HashSet<>();
        for (DriveDto.EventRequest e : request.getEvents()) {
            if (!within(e.getDetectedAt(), startedAt, endedAt)
                    || !eventKeys.add(e.getType() + "|" + e.getDetectedAt().toInstant())) {
                continue;
            }
            boolean hasCoord = e.getLatitude() != null && e.getLongitude() != null;   // 한쪽만 있으면 위치 없음으로
            events.add(new DriveDto.Event(null, driveId, e.getType(), e.getDetectedAt(),
                    hasCoord ? e.getLatitude() : null, hasCoord ? e.getLongitude() : null));
        }
        events.sort(Comparator.comparing(e -> e.getDetectedAt().toInstant()));
        int skippedEvents = request.getEvents().size() - events.size();
        if (!events.isEmpty()) {
            driveDao.insertEvents(driveId, events);
        }

        // 경유지: 기간 밖·중복·없는 장소 제외. 방문 시각이 추가 시각 ~ 종료 시각 밖이면 방문 기록만 무시
        int skippedWaypoints = 0;
        Set<String> waypointKeys = new HashSet<>();
        for (DriveDto.WaypointRequest w : request.getWaypoints()) {
            if (!within(w.getAddedAt(), startedAt, endedAt)
                    || !waypointKeys.add(w.getPlaceId() + "|" + w.getAddedAt().toInstant())) {
                skippedWaypoints++;
                continue;
            }
            OffsetDateTime visitedAt = within(w.getVisitedAt(), w.getAddedAt(), endedAt) ? w.getVisitedAt() : null;
            int inserted = driveDao.insertWaypoint(DriveDto.Waypoint.builder()
                    .driveId(driveId)
                    .placeId(w.getPlaceId())
                    .navMode(w.getMode())
                    .triggerType(w.getTrigger())
                    .addedAt(w.getAddedAt())
                    .visited(visitedAt != null)
                    .visitedAt(visitedAt)
                    .build());
            if (inserted == 0) {
                skippedWaypoints++;
            }
        }

        // 시간대 통계 (리포트·달력용)
        List<DriveDto.HourlyStat> stats = HourlyStatsCalculator.calculate(
                driveId, userId, startedAt, endedAt, request.getDistanceMeters(), events);
        driveDao.insertHourlyStats(stats);

        drive.setStatus(DriveDto.Drive.Status.ENDED);
        drive.setEndedAt(endedAt);
        drive.setObservedSeconds(stats.stream().mapToInt(DriveDto.HourlyStat::getObservedSeconds).sum());
        drive.setDistanceMeters(request.getDistanceMeters());
        drive.setInattentionCount((int) events.stream()
                .filter(e -> e.getEventType() == DriveDto.EventType.INATTENTION).count());
        drive.setDrowsinessCount((int) events.stream()
                .filter(e -> e.getEventType() == DriveDto.EventType.DROWSINESS).count());
        drive.setRouteJson(jsonMapper.writeValueAsString(request.getRoute()));
        driveDao.endDrive(drive);

        return buildReport(drive, new DriveDto.Skipped(skippedEvents, skippedWaypoints));
    }

    /** 주행 상세 리포트. 종료 업로드 전이면 409 */
    @Transactional(readOnly = true)
    public DriveDto.Report getReport(Long userId, Long driveId) {
        DriveDto.Drive drive = driveDao.findById(driveId, userId);
        if (drive == null) {
            throw new ApiException(ErrorCode.DRIVE_NOT_FOUND);
        }
        if (drive.getStatus() != DriveDto.Drive.Status.ENDED) {
            throw new ApiException(ErrorCode.DRIVE_NOT_ENDED);
        }
        return buildReport(drive, null);
    }

    /** 종료된 주행 목록 (최근 시작순, 커서 페이지) */
    @Transactional(readOnly = true)
    public DriveDto.ListResponse getList(Long userId, Integer size, String cursor) {
        int pageSize = size == null ? 20 : size;
        if (pageSize < 1 || pageSize > 50) {
            throw ApiException.invalidField("size", "size 는 1~50 이어야 합니다.");
        }

        OffsetDateTime cursorStartedAt = null;
        Long cursorDriveId = null;
        if (cursor != null && !cursor.isBlank()) {
            try {
                String[] parts = new String(Base64.getUrlDecoder().decode(cursor), StandardCharsets.UTF_8).split("\\|");
                cursorStartedAt = OffsetDateTime.parse(parts[0]);
                cursorDriveId = Long.parseLong(parts[1]);
            } catch (RuntimeException e) {
                throw ApiException.invalidField("cursor", "cursor 가 올바르지 않습니다.");
            }
        }

        List<DriveDto.Drive> rows = driveDao.findEndedPage(userId, cursorStartedAt, cursorDriveId, pageSize + 1);
        String nextCursor = null;
        if (rows.size() > pageSize) {
            rows = rows.subList(0, pageSize);
            DriveDto.Drive last = rows.get(pageSize - 1);
            nextCursor = Base64.getUrlEncoder().withoutPadding().encodeToString(
                    (last.getStartedAt() + "|" + last.getDriveId()).getBytes(StandardCharsets.UTF_8));
        }
        return new DriveDto.ListResponse(rows.stream().map(DriveDto.ListItem::from).toList(), nextCursor);
    }

    private void validateEnd(OffsetDateTime startedAt, OffsetDateTime endedAt, int distanceMeters,
                             List<List<Double>> route) {
        if (endedAt.isBefore(startedAt)) {
            throw ApiException.invalidField("endedAt", "endedAt 은 관찰 시작 시각 이후여야 합니다.");
        }
        if (endedAt.isAfter(OffsetDateTime.now().plus(CLOCK_SKEW))) {
            throw ApiException.invalidField("endedAt", "endedAt 이 현재 시각보다 늦습니다.");
        }
        Duration observed = Duration.between(startedAt, endedAt);
        if (observed.compareTo(MAX_DRIVE) > 0) {
            throw ApiException.invalidField("endedAt", "관찰 시간은 최대 24시간입니다.");
        }
        if (distanceMeters > Math.max(1, observed.toSeconds()) * MAX_SPEED_MPS) {
            throw ApiException.invalidField("distanceMeters", "관찰 시간에 비해 이동 거리가 너무 깁니다.");
        }
        for (List<Double> p : route) {
            if (p == null || p.size() != 2 || p.get(0) == null || p.get(1) == null
                    || Math.abs(p.get(0)) > 90 || Math.abs(p.get(1)) > 180) {
                throw ApiException.invalidField("route", "route 는 [위도, 경도] 배열이어야 합니다.");
            }
        }
    }

    /** from ≤ t ≤ to (t 가 null 이면 false) */
    private static boolean within(OffsetDateTime t, OffsetDateTime from, OffsetDateTime to) {
        if (t == null) {
            return false;
        }
        Instant i = t.toInstant();
        return !i.isBefore(from.toInstant()) && !i.isAfter(to.toInstant());
    }

    private DriveDto.Report buildReport(DriveDto.Drive drive, DriveDto.Skipped skipped) {
        List<List<Double>> points = drive.getRouteJson() == null
                ? List.of()
                : jsonMapper.readValue(drive.getRouteJson(), new TypeReference<List<List<Double>>>() { });
        DriveDto.Point start = points.isEmpty() ? null : new DriveDto.Point(points.get(0).get(0), points.get(0).get(1));
        List<Double> lastPoint = points.isEmpty() ? null : points.get(points.size() - 1);
        DriveDto.Point end = lastPoint == null ? null : new DriveDto.Point(lastPoint.get(0), lastPoint.get(1));

        List<DriveDto.EventItem> events = driveDao.findEvents(drive.getDriveId()).stream()
                .map(e -> new DriveDto.EventItem(e.getEventType(), e.getDetectedAt(), e.getLatitude(), e.getLongitude()))
                .toList();
        List<DriveDto.WaypointItem> waypoints = driveDao.findWaypoints(drive.getDriveId()).stream()
                .map(w -> DriveDto.WaypointItem.builder()
                        .waypointId(w.getWaypointId())
                        .placeId(w.getPlaceId())
                        .placeName(w.getPlaceName())
                        .placeType(w.getPlaceType())
                        .mode(w.getNavMode())
                        .trigger(w.getTriggerType())
                        .addedAt(w.getAddedAt())
                        .visited(Boolean.TRUE.equals(w.getVisited()))
                        .visitedAt(w.getVisitedAt())
                        .build())
                .toList();

        return DriveDto.Report.builder()
                .driveId(drive.getDriveId())
                .status(drive.getStatus())
                .startedAt(drive.getStartedAt())
                .endedAt(drive.getEndedAt())
                .observedSeconds(drive.getObservedSeconds())
                .distanceMeters(drive.getDistanceMeters())
                .inattentionCount(drive.getInattentionCount())
                .drowsinessCount(drive.getDrowsinessCount())
                .route(new DriveDto.Route(start, end, points))
                .events(events)
                .waypoints(waypoints)
                .skipped(skipped)
                .build();
    }

    /** 탈퇴: 주행 기록 전부 삭제 (감지·경유지·시간대 통계는 CASCADE) */
    @Transactional
    public void deleteAllByUser(Long userId) {
        driveDao.deleteByUserId(userId);
    }
}
