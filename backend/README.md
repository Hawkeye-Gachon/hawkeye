# DriverGuard (Hawkeye 백엔드)

운전자 상태 관찰 앱 **Hawkeye**의 백엔드 서버입니다.
앱이 차량 내 카메라로 **전방 주시 이탈**과 **졸음**을 감지하면, 서버는

- 지금 달리는 차선에서 **바로 들어갈 수 있는 가장 가까운 휴게소·졸음쉼터 한 곳**을 찾아 주고 (앱이 차량 내비 경유지로 추가)
- 주행이 끝나면 감지 기록·경로·경유지를 받아 **주행 리포트**와 **일간·주간 통계**를 만들어 줍니다.

## 기술 스택

| 구분 | 사용 |
|---|---|
| 언어 · 프레임워크 | Java 17, Spring Boot 4.0.9 (Web MVC, Security, Validation) |
| 데이터 | PostgreSQL 15, MyBatis 4.0.1 |
| 인증 | JWT (jjwt 0.12.6) accessToken 30분 + DB 저장 refreshToken 14일 (재발급 시 교체) |
| API 문서 | springdoc-openapi 3.0.3 (Swagger UI) |
| 외부 API | 카카오모빌리티 자동차 길찾기, 공공데이터포털 졸음쉼터 표준데이터, 한국도로공사 휴게소 정보 |

## 주요 기능

| 기능 | API |
|---|---|
| 회원가입(가입 즉시 로그인) · 로그인 · 토큰 재발급 · 로그아웃 | `POST /api/auth/signup` `login` `refresh` `logout` |
| 내 정보 · 회원 탈퇴 | `GET` `DELETE /api/users/me` |
| 약관 목록 · 전문 · 내 동의 현황 · 동의 저장 | `GET /api/terms` `GET /api/terms/{id}` `GET` `POST /api/users/me/terms-agreements` |
| 졸음 감지 시 가장 가까운 쉴 곳 | `POST /api/rest-places/nearby` |
| 관찰 시작 · 종료(기록 업로드) · 주행 리포트 · 주행 목록 | `POST /api/drives` `POST /api/drives/{id}/end` `GET /api/drives/{id}` `GET /api/drives` |
| 달력 · 일간 리포트 · 주간 통계 | `GET /api/reports/calendar` `daily` `weekly` |

요청·응답 형식과 오류 코드는 실행 후 Swagger UI(`/swagger-ui.html`)에서 확인할 수 있습니다.

## 프로젝트 구조

도메인별로 `controller → service → dao(MyBatis) → mapper XML` 순서로 나누고, 요청·응답·테이블 행은 도메인마다 `XxxDto` 한 파일의 중첩 클래스로 둡니다.

```
src/main/java/com/contest/DriverGuard
├── auth/          회원가입 · 로그인 · 토큰 · 탈퇴 처리
├── user/          사용자 (내 정보)
├── terms/         약관 · 약관 동의
├── drive/         관찰 시작 · 종료 업로드 · 주행 리포트 (HourlyStatsCalculator: 시간대 집계)
├── report/        달력 · 일간 · 주간 통계
├── restplace/     휴게소·졸음쉼터 수집과 주변 조회 (NearbyRestPlaceService, RingRoad, KakaoDirectionsClient)
├── security/      JWT 필터 · 보안 설정 · 약관 미동의 차단(TermsAgreementInterceptor)
├── exception/     ErrorCode · ApiException · 공통 에러 응답
├── configuration/ Swagger · Web MVC 설정
└── property/      설정값 바인딩 (JWT)
src/main/resources
├── db/schema.sql          테이블 정의 (PostgreSQL)
├── db/terms-sample.sql    약관 데이터 (개발용)
└── mybatis/*Mapper.xml    SQL
```

## 핵심 설계

### 1. 반대 차선 쉼터를 절대 안내하지 않기
고속도로에서 반대 차선의 휴게소를 안내하면 운전자가 다음 IC까지 가서 돌아와야 합니다. 그래서 **확인되지 않으면 안내하지 않는 것**을 원칙으로 판정합니다.

- **휴게소**: 카카오 길찾기를 현재 진행 방향으로 출발시켜 경로 모양을 봅니다. 휴게소를 지나쳤다가 되돌아와 도착하면 반대 차선으로 보고 제외합니다.
- **졸음쉼터**: 카카오 지도에 진입로가 없어 길찾기로 직접 판정할 수 없습니다. 대신 내 방향으로 확인된 휴게소까지 가는 경로 위(60m 이내)에 있고 방향 표기가 같은 쉼터만 인정합니다.
- **순환고속도로(서울외곽)**: 같은 "일산 방향"이라도 원의 위치에 따라 시계·반시계가 바뀝니다. 그래서 차가 도는 방향과 장소의 방향을 좌표로 비교합니다(`RingRoad`).
- 고속도로 위가 아니거나, 길찾기가 실패하거나, 판정이 엇갈리면 결과를 비우고 앱은 일반 정차 안내를 표시합니다.

검증 결과는 다음과 같습니다.
- 고속도로 12개 방향을 15km 간격으로, 서울외곽순환은 양방향 한 바퀴씩 실제 카카오 길찾기로 확인했고 반대 차선 결과는 0건이었습니다.
- 서울외곽순환의 정답은 데이터 표기가 아니라 카카오 경로에서 계산한 실제 차선 위치(우측통행)로 정했습니다.
- 녹화한 카카오 응답 72개 상황은 오프라인 회귀 테스트로 매번 다시 확인합니다.

### 2. 주행 중에는 서버와 통신하지 않는 기록 방식
- 관찰을 시작할 때 주행만 등록합니다(`clientDriveId` UUID).
- 감지 기록, 앱이 단순화한 경로, 이동 거리, 경유지 추가·방문 기록은 앱이 로컬에 저장했다가 **종료할 때 한 번에** 올립니다.
- 같은 요청을 다시 보내도 한 번만 저장됩니다. 시작은 `clientDriveId`로, 종료는 주행 상태와 행 잠금으로 막습니다.
- 종료 업로드 없이 다음 관찰이 시작되면 이전 주행은 `ABANDONED`가 되고, 나중에 업로드되면 `ENDED`로 바뀝니다.
- 주행 기간 밖이나 중복된 항목은 주행 전체를 거부하지 않고 건너뛴 뒤 `skipped`로 알려 줍니다.

### 3. 리포트 집계
- 종료 업로드 때 주행을 **서울 기준 1시간 구간**으로 나눠 `drive_hourly_stats`에 저장합니다. 일간·주간·달력은 이 표만 읽습니다.
- 감지 횟수는 실제 감지 시각의 구간에 셉니다. 관찰 시간은 구간마다 나누고, 이동 거리는 관찰 시간 비율로 나누되 합계가 정확히 맞도록 배분합니다.
- 자정을 넘긴 주행은 두 날짜에 나눠 집계하고, 주간 주행 횟수에는 한 번만 셉니다.

### 4. 약관 버전 관리
- 약관을 개정하면 새 버전 행을 추가합니다. 사용자가 현재 시행 중인 필수 약관에 동의하지 않았으면 로그인 응답의 `termsAgreementRequired`가 `true`가 됩니다.
- 그 상태로 기능 API를 호출하면 `403 TERMS_AGREEMENT_REQUIRED`가 나갑니다. 앱은 이 두 가지로 약관 동의 화면으로 이동합니다.

### 5. 공통 에러 응답
모든 오류를 같은 형식으로 응답하며, 앱은 `code`로 분기하고 `fieldErrors`를 입력란 아래에 표시합니다.
```json
{
  "status": 400, "error": "Bad Request", "code": "VALIDATION_FAILED", "message": "입력값을 확인해 주세요.",
  "path": "/api/auth/signup",
  "fieldErrors": [{ "field": "email", "message": "이메일 형식이 아닙니다. 예: name@mail.com" }]
}
```

## 실행 방법

1. PostgreSQL에 `DriverGuard` 데이터베이스를 만들고 `db/schema.sql`, `db/terms-sample.sql`을 차례로 실행합니다.
2. `src/main/resources/application.properties.sample`을 같은 폴더에 `application.properties`로 복사한 뒤 DB 접속 정보를 맞추고, 아래 값을 환경변수로 넣습니다. (`application.properties`는 비밀 값이 들어가므로 git에 올리지 않습니다.)

   | 환경변수 | 용도 |
   |---|---|
   | `JWT_SECRET` | JWT 서명 키 (Base64, 256bit 이상) |
   | `KAKAO_REST_API_KEY` | 카카오모빌리티 길찾기 (쉼터 조회) |
   | `DATA_GO_KR_SERVICE_KEY` | 공공데이터포털 졸음쉼터 데이터 (Decoding 키) |
   | `EX_API_KEY` | 한국도로공사 휴게소 데이터 |

3. 서버를 실행합니다.
   ```bash
   ./mvnw spring-boot:run
   ```
   DB의 휴게소·졸음쉼터 데이터가 비어 있으면 시작할 때 한 번 수집하고, 이후에는 매주 월요일 04:00에 갱신합니다.
4. Swagger UI: http://localhost:8080/swagger-ui.html
   로그인 응답의 `accessToken`을 우측 상단 **Authorize**에 넣으면 인증이 필요한 API를 호출할 수 있습니다.

## 테스트

```bash
./mvnw test
```

| 테스트 | 개수 | 내용 |
|---|---|---|
| API 통합 테스트 (`*ApiTest`) | 31 | 회원·약관·탈퇴·주행·리포트를 실제 HTTP 요청으로 흐름대로 검사. 로컬 DB를 쓰고 테스트마다 롤백 |
| 쉼터 조회 회귀 테스트 (`NearbyRestPlaceServiceTest`) | 78 | 녹화한 카카오 응답으로 반대 차선 결과가 없는지 검사 (카카오 호출 없음) |
| 순환도로 판정 (`RingRoadTest`) | 5 | 서울외곽순환 시계·반시계 판정 |
| 시간대 집계 (`HourlyStatsCalculatorTest`) | 5 | 구간 분할, 자정 경계, 거리 배분 합계 |
| 컨텍스트 로드 | 1 | |

통합 테스트는 `schema.sql`과 `terms-sample.sql`이 적용된 로컬 DB가 필요합니다.

## 데이터 출처
- 졸음쉼터: 공공데이터포털 「전국졸음쉼터표준데이터」
- 휴게소: 한국도로공사 공공데이터 포털 휴게소 정보
- 도로 거리·경로: 카카오모빌리티 자동차 길찾기 API
