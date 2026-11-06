package com.contest.DriverGuard.restplace.controller;

import com.contest.DriverGuard.configuration.SwaggerConfig;
import com.contest.DriverGuard.restplace.dto.RestPlaceDto;
import com.contest.DriverGuard.restplace.service.NearbyRestPlaceService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.security.SecurityRequirement;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@Tag(name = "RestPlace", description = "휴게소/졸음쉼터")
@SecurityRequirement(name = SwaggerConfig.BEARER)
@RestController
@RequestMapping("/api/rest-places")
@RequiredArgsConstructor
public class RestPlaceController {

    private final NearbyRestPlaceService nearbyRestPlaceService;

    @Operation(summary = "가장 가까운 휴게소/졸음쉼터 조회",
            description = "졸음 감지 시 호출. 현재 위치·진행 방향 기준으로 지금 차선에서 바로 들어갈 수 있는 곳 중 "
                    + "실제 도로 거리(카카오 길찾기)가 가장 가까운 1곳을 반환 (반대 차선 제외). "
                    + "앱은 place 의 latitude/longitude 를 경유지로 추가. "
                    + "heading 은 speed >= 2.8m/s(약 10km/h) 일 때만 반영. "
                    + "고속도로 전용: onHighway=false 면 일반도로라 안내하지 않음 (place=null, 앱은 일반 정차 안내). "
                    + "onHighway=null 은 근처에 휴게소가 없거나 길찾기 실패로 확인 못함. "
                    + "distanceSource=UNAVAILABLE 이면 길찾기 실패로 확인 불가 (place=null). "
                    + "400: 좌표 범위 오류")
    @PostMapping("/nearby")
    public ResponseEntity<RestPlaceDto.NearbyResponse> nearby(@Valid @RequestBody RestPlaceDto.NearbyRequest request) {
        return ResponseEntity.ok(nearbyRestPlaceService.findNearest(request));
    }
}