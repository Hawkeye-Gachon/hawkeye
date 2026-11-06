# DriverGuard API 실행 기록 (2026-11-06 20:43 KST)

로컬 서버(localhost:8080)에서 실제로 호출한 요청·응답 기록 (명세서 예시 원본).
토큰은 `eyJ…` / `…` 로 줄여 표기. 시각은 실행 시점 기준으로 만든 값.

## 시나리오 1. 처음 가입할 때

스플래시 → 로그인 → 회원가입 → 약관 동의 → 홈

#### 회원가입
```http
POST /api/auth/signup

{"email":"driver1@example.com","name":"김하늘","password":"Test1234!","ageConfirmed":true}
```
**201**
```json
{"userId":1,"email":"driver1@example.com","name":"김하늘","grantType":"Bearer","accessToken":"eyJ…","accessTokenExpiresIn":1800,"refreshToken":"…","termsAgreementRequired":true}
```

#### 약관 동의 화면 진입
```http
GET /api/terms
```
**200**
```json
{"terms":[{"termsId":1,"code":"SERVICE","title":"서비스 이용약관","version":1,"required":true,"effectiveAt":"2026-11-01T00:00:00+09:00"},{"termsId":2,"code":"PRIVACY","title":"개인정보 수집 · 이용","version":1,"required":true,"effectiveAt":"2026-11-01T00:00:00+09:00"},{"termsId":3,"code":"CAMERA","title":"운전자 카메라 영상 처리","version":1,"required":true,"effectiveAt":"2026-11-01T00:00:00+09:00"},{"termsId":4,"code":"LOCATION","title":"위치정보 이용","version":1,"required":true,"effectiveAt":"2026-11-01T00:00:00+09:00"},{"termsId":5,"code":"MARKETING","title":"이벤트 · 혜택 알림 수신","version":1,"required":false,"effectiveAt":"2026-11-01T00:00:00+09:00"}]}
```

#### 전문 보기
```http
GET /api/terms/1
```
**200**
```json
{"termsId":1,"code":"SERVICE","title":"서비스 이용약관","version":1,"required":true,"effectiveAt":"2026-11-01T00:00:00+09:00","content":"# 제1조 (목적)\n이 약관은 Hawkeye(이하 “서비스”)가 제공하는 운전자 상태 관찰 및 휴식 안내 서비스의 이용 조건과 절차, 회사와 이용자의 권리·의무 및 책임 사항을 정하는 것을 목적으로 합니다.\n\n# 제2조 (정의)\n“관찰”이란 차량 내 운전자 카메라 영상을 이용해 전방 주시 이탈과 졸음 징후를 판단하는 기능을 말합니다. “경유지 추가”란 차량 기본 내비게이션의 안내 경로에 휴게소 또는 졸음쉼터를 추가하도록 요청하는 기능을 말합니다.\n\n# 제3조 (서비스의 내용)\n서비스는 관찰 결과에 따라 화면 경고와 경고음을 제공하며, 휴식이 필요하다고 판단되는 경우 가까운 쉼터를 경유지로 제안합니다.\n\n# 제4조 (영상 정보의 처리)\n예시 문구입니다.\n\n# 제5조 (이용자의 의무)\n예시 문구입니다.\n\n# 제6조 (책임의 제한)\n예시 문구입니다.\n\n# 부칙\n이 약관은 2026년 11월 1일부터 시행합니다."}
```

#### 동의 전 기능 API 호출 → 403
```http
POST /api/rest-places/nearby
Authorization: Bearer eyJ…

{"lat":37.073253,"lng":127.132555,"heading":340,"speed":27}
```
**403**
```json
{"timestamp":"2026-11-06T11:43:04.053888300Z","status":403,"error":"Forbidden","code":"TERMS_AGREEMENT_REQUIRED","message":"필수 약관 동의가 필요합니다.","path":"/api/rest-places/nearby"}
```

#### 동의하고 시작하기 (필수 4개 동의, 선택 미동의)
```http
POST /api/users/me/terms-agreements
Authorization: Bearer eyJ…

{"agreements":[{"termsId":1,"agreed":true},{"termsId":2,"agreed":true},{"termsId":3,"agreed":true},{"termsId":4,"agreed":true},{"termsId":5,"agreed":false}]}
```
**200**
```json
{"termsAgreementRequired":false,"agreements":[{"termsId":1,"code":"SERVICE","title":"서비스 이용약관","version":1,"required":true,"agreed":true,"agreedAt":"2026-11-06T20:43:04.321411+09:00"},{"termsId":2,"code":"PRIVACY","title":"개인정보 수집 · 이용","version":1,"required":true,"agreed":true,"agreedAt":"2026-11-06T20:43:04.321411+09:00"},{"termsId":3,"code":"CAMERA","title":"운전자 카메라 영상 처리","version":1,"required":true,"agreed":true,"agreedAt":"2026-11-06T20:43:04.321411+09:00"},{"termsId":4,"code":"LOCATION","title":"위치정보 이용","version":1,"required":true,"agreed":true,"agreedAt":"2026-11-06T20:43:04.321411+09:00"},{"termsId":5,"code":"MARKETING","title":"이벤트 · 혜택 알림 수신","version":1,"required":false,"agreed":false,"agreedAt":null}]}
```

#### 홈: 최근 주행 미리보기 (기록 없음)
```http
GET /api/drives?size=1
Authorization: Bearer eyJ…
```
**200**
```json
{"drives":[],"nextCursor":null}
```

#### 내 정보
```http
GET /api/users/me
Authorization: Bearer eyJ…
```
**200**
```json
{"userId":1,"email":"driver1@example.com","name":"김하늘","status":"ACTIVE","createdAt":"2026-11-06T20:43:03.249219+09:00","termsAgreementRequired":false}
```

## 시나리오 2. 기존 회원 로그인 · 토큰 재발급 · 약관 재동의

#### 로그인
```http
POST /api/auth/login

{"email":"driver1@example.com","password":"Test1234!"}
```
**200**
```json
{"grantType":"Bearer","accessToken":"eyJ…","accessTokenExpiresIn":1800,"refreshToken":"…","termsAgreementRequired":false}
```

#### 토큰 재발급 (refreshToken 교체)
```http
POST /api/auth/refresh

{"refreshToken":"…"}
```
**200**
```json
{"grantType":"Bearer","accessToken":"eyJ…","accessTokenExpiresIn":1800,"refreshToken":"…","termsAgreementRequired":false}
```

#### 이미 사용한 refreshToken 재사용 → 401
```http
POST /api/auth/refresh

{"refreshToken":"…"}
```
**401**
```json
{"timestamp":"2026-11-06T11:43:05.623482100Z","status":401,"error":"Unauthorized","code":"INVALID_REFRESH_TOKEN","message":"유효하지 않거나 만료된 리프레시 토큰입니다.","path":"/api/auth/refresh"}
```

(DB 에 서비스 이용약관 v2 를 시행 상태로 추가 → 재동의가 필요한 상황)

#### 약관 개정 후 내 정보 → termsAgreementRequired=true
```http
GET /api/users/me
Authorization: Bearer eyJ…
```
**200**
```json
{"userId":1,"email":"driver1@example.com","name":"김하늘","status":"ACTIVE","createdAt":"2026-11-06T20:43:03.249219+09:00","termsAgreementRequired":true}
```

#### 기능 API 호출 → 403
```http
GET /api/drives?size=1
Authorization: Bearer eyJ…
```
**403**
```json
{"timestamp":"2026-11-06T11:43:06.777572Z","status":403,"error":"Forbidden","code":"TERMS_AGREEMENT_REQUIRED","message":"필수 약관 동의가 필요합니다.","path":"/api/drives"}
```

#### 내 동의 현황 (이전 선택 미리 채우기)
```http
GET /api/users/me/terms-agreements
Authorization: Bearer eyJ…
```
**200**
```json
{"termsAgreementRequired":true,"agreements":[{"termsId":8,"code":"SERVICE","title":"서비스 이용약관","version":2,"required":true,"agreed":false,"agreedAt":null},{"termsId":2,"code":"PRIVACY","title":"개인정보 수집 · 이용","version":1,"required":true,"agreed":true,"agreedAt":"2026-11-06T20:43:04.321411+09:00"},{"termsId":3,"code":"CAMERA","title":"운전자 카메라 영상 처리","version":1,"required":true,"agreed":true,"agreedAt":"2026-11-06T20:43:04.321411+09:00"},{"termsId":4,"code":"LOCATION","title":"위치정보 이용","version":1,"required":true,"agreed":true,"agreedAt":"2026-11-06T20:43:04.321411+09:00"},{"termsId":5,"code":"MARKETING","title":"이벤트 · 혜택 알림 수신","version":1,"required":false,"agreed":false,"agreedAt":null}]}
```

#### 예전 버전(termsId 1)으로 동의 → 400
```http
POST /api/users/me/terms-agreements
Authorization: Bearer eyJ…

{"agreements":[{"termsId":1,"agreed":true}]}
```
**400**
```json
{"timestamp":"2026-11-06T11:43:07.057790200Z","status":400,"error":"Bad Request","code":"TERMS_NOT_CURRENT","message":"현재 시행 중인 약관이 아닙니다. 약관을 다시 확인해 주세요.","path":"/api/users/me/terms-agreements"}
```

#### 새 버전(termsId 8)에 동의
```http
POST /api/users/me/terms-agreements
Authorization: Bearer eyJ…

{"agreements":[{"termsId":8,"agreed":true}]}
```
**200**
```json
{"termsAgreementRequired":false,"agreements":[{"termsId":8,"code":"SERVICE","title":"서비스 이용약관","version":2,"required":true,"agreed":true,"agreedAt":"2026-11-06T20:43:08.099328+09:00"},{"termsId":2,"code":"PRIVACY","title":"개인정보 수집 · 이용","version":1,"required":true,"agreed":true,"agreedAt":"2026-11-06T20:43:04.321411+09:00"},{"termsId":3,"code":"CAMERA","title":"운전자 카메라 영상 처리","version":1,"required":true,"agreed":true,"agreedAt":"2026-11-06T20:43:04.321411+09:00"},{"termsId":4,"code":"LOCATION","title":"위치정보 이용","version":1,"required":true,"agreed":true,"agreedAt":"2026-11-06T20:43:04.321411+09:00"},{"termsId":5,"code":"MARKETING","title":"이벤트 · 혜택 알림 수신","version":1,"required":false,"agreed":false,"agreedAt":null}]}
```

(테스트용 v2 약관과 그 동의 기록은 삭제해 원래 약관 상태로 복원)

## 시나리오 3. 관찰 시작

관찰 시작 시각 2026-11-06T19:58:09+09:00 (실행 시점 45분 전으로 등록)

#### 관찰 시작
```http
POST /api/drives
Authorization: Bearer eyJ…

{"clientDriveId":"3d159653-6702-4412-aed7-66120077d400","startedAt":"2026-11-06T19:58:09+09:00"}
```
**201**
```json
{"driveId":1,"status":"IN_PROGRESS","startedAt":"2026-11-06T19:58:09+09:00","abandonedDriveId":null}
```

#### 응답을 못 받아 같은 clientDriveId 로 재전송 → 200, 같은 driveId
```http
POST /api/drives
Authorization: Bearer eyJ…

{"clientDriveId":"3d159653-6702-4412-aed7-66120077d400","startedAt":"2026-11-06T19:58:09+09:00"}
```
**200**
```json
{"driveId":1,"status":"IN_PROGRESS","startedAt":"2026-11-06T19:58:09+09:00","abandonedDriveId":null}
```

#### 관찰 중 리포트 조회 → 409
```http
GET /api/drives/1
Authorization: Bearer eyJ…
```
**409**
```json
{"timestamp":"2026-11-06T11:43:09.707785400Z","status":409,"error":"Conflict","code":"DRIVE_NOT_ENDED","message":"아직 종료되지 않은 주행입니다.","path":"/api/drives/1"}
```

## 시나리오 4~6. 관찰 중

부주의·졸음 감지와 경유지 추가 결과는 앱이 로컬에 저장. 주행 중 서버 호출은 쉼터 찾기뿐

#### 졸음 감지 → 쉼터 찾기 (경부고속도로 북행 오산 부근)
```http
POST /api/rest-places/nearby
Authorization: Bearer eyJ…

{"lat":37.073253,"lng":127.132555,"heading":340,"speed":27}
```
**200**
```json
{"directionApplied":true,"onHighway":true,"distanceSource":"ROAD","place":{"placeId":1211,"type":"DROWSY_SHELTER","name":"오산","latitude":37.1542659,"longitude":127.0861027,"roadDistanceMeters":10313,"durationSeconds":550}}
```

## 시나리오 7. 관찰 종료 → 리포트

감지 5건 중 1건은 같은 시각 중복(건너뜀). 경유지는 쉼터 반경 체류로 앱이 방문 확인한 visitedAt 포함

#### 종료하고 리포트 보기
```http
POST /api/drives/1/end
Authorization: Bearer eyJ…

{"endedAt":"2026-11-06T20:40:10+09:00","distanceMeters":28400,"events":[{"type":"INATTENTION","detectedAt":"2026-11-06T20:08:10+09:00","latitude":37.0951,"longitude":127.1228},{"type":"INATTENTION","detectedAt":"2026-11-06T20:15:11+09:00","latitude":37.1102,"longitude":127.115},{"type":"INATTENTION","detectedAt":"2026-11-06T20:15:11+09:00","latitude":37.1102,"longitude":127.115},{"type":"DROWSINESS","detectedAt":"2026-11-06T20:21:11+09:00","latitude":37.1301,"longitude":127.1002},{"type":"INATTENTION","detectedAt":"2026-11-06T20:31:11+09:00","latitude":37.161,"longitude":127.085}],"route":[[37.073253,127.132555],[37.1102,127.115],[37.1542659,127.0861027],[37.2351,127.1052]],"waypoints":[{"placeId":1211,"mode":"WAYPOINT","trigger":"DROWSINESS","addedAt":"2026-11-06T20:22:11+09:00","visitedAt":"2026-11-06T20:28:11+09:00"}]}
```
**200**
```json
{"driveId":1,"status":"ENDED","startedAt":"2026-11-06T19:58:09+09:00","endedAt":"2026-11-06T20:40:10+09:00","observedSeconds":2521,"distanceMeters":28400,"inattentionCount":3,"drowsinessCount":1,"route":{"start":{"latitude":37.073253,"longitude":127.132555},"end":{"latitude":37.2351,"longitude":127.1052},"points":[[37.073253,127.132555],[37.1102,127.115],[37.1542659,127.0861027],[37.2351,127.1052]]},"events":[{"type":"INATTENTION","detectedAt":"2026-11-06T20:08:10+09:00","latitude":37.0951,"longitude":127.1228},{"type":"INATTENTION","detectedAt":"2026-11-06T20:15:11+09:00","latitude":37.1102,"longitude":127.115},{"type":"DROWSINESS","detectedAt":"2026-11-06T20:21:11+09:00","latitude":37.1301,"longitude":127.1002},{"type":"INATTENTION","detectedAt":"2026-11-06T20:31:11+09:00","latitude":37.161,"longitude":127.085}],"waypoints":[{"waypointId":1,"placeId":1211,"placeName":"오산","placeType":"DROWSY_SHELTER","mode":"WAYPOINT","trigger":"DROWSINESS","addedAt":"2026-11-06T20:22:11+09:00","visited":true,"visitedAt":"2026-11-06T20:28:11+09:00"}],"skipped":{"events":1,"waypoints":0}}
```

#### 응답을 못 받아 재전송 → 같은 리포트 (skipped 없음)
```http
POST /api/drives/1/end
Authorization: Bearer eyJ…

{"endedAt":"2026-11-06T20:40:10+09:00","distanceMeters":28400,"events":[{"type":"INATTENTION","detectedAt":"2026-11-06T20:08:10+09:00","latitude":37.0951,"longitude":127.1228},{"type":"INATTENTION","detectedAt":"2026-11-06T20:15:11+09:00","latitude":37.1102,"longitude":127.115},{"type":"INATTENTION","detectedAt":"2026-11-06T20:15:11+09:00","latitude":37.1102,"longitude":127.115},{"type":"DROWSINESS","detectedAt":"2026-11-06T20:21:11+09:00","latitude":37.1301,"longitude":127.1002},{"type":"INATTENTION","detectedAt":"2026-11-06T20:31:11+09:00","latitude":37.161,"longitude":127.085}],"route":[[37.073253,127.132555],[37.1102,127.115],[37.1542659,127.0861027],[37.2351,127.1052]],"waypoints":[{"placeId":1211,"mode":"WAYPOINT","trigger":"DROWSINESS","addedAt":"2026-11-06T20:22:11+09:00","visitedAt":"2026-11-06T20:28:11+09:00"}]}
```
**200**
```json
{"driveId":1,"status":"ENDED","startedAt":"2026-11-06T19:58:09+09:00","endedAt":"2026-11-06T20:40:10+09:00","observedSeconds":2521,"distanceMeters":28400,"inattentionCount":3,"drowsinessCount":1,"route":{"start":{"latitude":37.073253,"longitude":127.132555},"end":{"latitude":37.2351,"longitude":127.1052},"points":[[37.073253,127.132555],[37.1102,127.115],[37.1542659,127.0861027],[37.2351,127.1052]]},"events":[{"type":"INATTENTION","detectedAt":"2026-11-06T20:08:10+09:00","latitude":37.0951,"longitude":127.1228},{"type":"INATTENTION","detectedAt":"2026-11-06T20:15:11+09:00","latitude":37.1102,"longitude":127.115},{"type":"DROWSINESS","detectedAt":"2026-11-06T20:21:11+09:00","latitude":37.1301,"longitude":127.1002},{"type":"INATTENTION","detectedAt":"2026-11-06T20:31:11+09:00","latitude":37.161,"longitude":127.085}],"waypoints":[{"waypointId":1,"placeId":1211,"placeName":"오산","placeType":"DROWSY_SHELTER","mode":"WAYPOINT","trigger":"DROWSINESS","addedAt":"2026-11-06T20:22:11+09:00","visited":true,"visitedAt":"2026-11-06T20:28:11+09:00"}]}
```

#### 리포트 → 홈 (미리보기 갱신)
```http
GET /api/drives?size=1
Authorization: Bearer eyJ…
```
**200**
```json
{"drives":[{"driveId":1,"startedAt":"2026-11-06T19:58:09+09:00","endedAt":"2026-11-06T20:40:10+09:00","observedSeconds":2521,"distanceMeters":28400,"inattentionCount":3,"drowsinessCount":1}],"nextCursor":null}
```

## 시나리오 8. 업로드하지 못한 주행

앱에 남아 있던 지난 주행을 나중에 등록·업로드. 마지막은 강제 종료로 남은 주행이 다음 관찰 시작 때 ABANDONED 가 되었다가 나중에 업로드되는 경우

#### 시작 요청이 실패했던 주행 등록 (2026-11-04 08:05)
```http
POST /api/drives
Authorization: Bearer eyJ…

{"clientDriveId":"13baf507-4bcc-45e3-ae4a-c5fc00767c00","startedAt":"2026-11-04T08:05:00+09:00"}
```
**201**
```json
{"driveId":2,"status":"IN_PROGRESS","startedAt":"2026-11-04T08:05:00+09:00","abandonedDriveId":null}
```

#### 그 주행 업로드
```http
POST /api/drives/2/end
Authorization: Bearer eyJ…

{"endedAt":"2026-11-04T08:52:42+09:00","distanceMeters":32800,"events":[{"type":"INATTENTION","detectedAt":"2026-11-04T08:12:10+09:00"},{"type":"INATTENTION","detectedAt":"2026-11-04T08:24:51+09:00"},{"type":"INATTENTION","detectedAt":"2026-11-04T08:31:05+09:00"}],"route":[[37.4,127.1],[37.5,127.03]],"waypoints":[]}
```
**200**
```json
{"driveId":2,"status":"ENDED","startedAt":"2026-11-04T08:05:00+09:00","endedAt":"2026-11-04T08:52:42+09:00","observedSeconds":2862,"distanceMeters":32800,"inattentionCount":3,"drowsinessCount":0,"route":{"start":{"latitude":37.4,"longitude":127.1},"end":{"latitude":37.5,"longitude":127.03},"points":[[37.4,127.1],[37.5,127.03]]},"events":[{"type":"INATTENTION","detectedAt":"2026-11-04T08:12:10+09:00","latitude":null,"longitude":null},{"type":"INATTENTION","detectedAt":"2026-11-04T08:24:51+09:00","latitude":null,"longitude":null},{"type":"INATTENTION","detectedAt":"2026-11-04T08:31:05+09:00","latitude":null,"longitude":null}],"waypoints":[],"skipped":{"events":0,"waypoints":0}}
```

#### 자정을 넘긴 주행 등록 (2026-11-05 23:30)
```http
POST /api/drives
Authorization: Bearer eyJ…

{"clientDriveId":"14ec1ab0-64df-482a-ad4e-af8e007cdc00","startedAt":"2026-11-05T23:30:00+09:00"}
```
**201**
```json
{"driveId":3,"status":"IN_PROGRESS","startedAt":"2026-11-05T23:30:00+09:00","abandonedDriveId":null}
```

#### 그 주행 업로드 (다음날 00:30 종료)
```http
POST /api/drives/3/end
Authorization: Bearer eyJ…

{"endedAt":"2026-11-06T00:30:00+09:00","distanceMeters":61000,"events":[{"type":"INATTENTION","detectedAt":"2026-11-05T23:48:00+09:00"},{"type":"DROWSINESS","detectedAt":"2026-11-06T00:14:30+09:00"}],"route":[],"waypoints":[]}
```
**200**
```json
{"driveId":3,"status":"ENDED","startedAt":"2026-11-05T23:30:00+09:00","endedAt":"2026-11-06T00:30:00+09:00","observedSeconds":3600,"distanceMeters":61000,"inattentionCount":1,"drowsinessCount":1,"route":{"start":null,"end":null,"points":[]},"events":[{"type":"INATTENTION","detectedAt":"2026-11-05T23:48:00+09:00","latitude":null,"longitude":null},{"type":"DROWSINESS","detectedAt":"2026-11-06T00:14:30+09:00","latitude":null,"longitude":null}],"waypoints":[],"skipped":{"events":0,"waypoints":0}}
```

#### 관찰 시작 후 앱 강제 종료 (종료 업로드 없음)
```http
POST /api/drives
Authorization: Bearer eyJ…

{"clientDriveId":"3998fec0-497b-4dcf-aa5f-59b6008f5a00","startedAt":"2026-11-06T20:41:13+09:00"}
```
**201**
```json
{"driveId":4,"status":"IN_PROGRESS","startedAt":"2026-11-06T20:41:13+09:00","abandonedDriveId":null}
```

#### 다음 관찰 시작 → 이전 주행이 ABANDONED (abandonedDriveId)
```http
POST /api/drives
Authorization: Bearer eyJ…

{"clientDriveId":"1e7a0bab-3fe4-449c-ad67-de1c005e9a00","startedAt":"2026-11-06T20:42:13+09:00"}
```
**201**
```json
{"driveId":5,"status":"IN_PROGRESS","startedAt":"2026-11-06T20:42:13+09:00","abandonedDriveId":4}
```

#### ABANDONED 주행을 나중에 업로드 → ENDED
```http
POST /api/drives/4/end
Authorization: Bearer eyJ…

{"endedAt":"2026-11-06T20:41:43+09:00","distanceMeters":400,"events":[],"route":[],"waypoints":[]}
```
**200**
```json
{"driveId":4,"status":"ENDED","startedAt":"2026-11-06T20:41:13+09:00","endedAt":"2026-11-06T20:41:43+09:00","observedSeconds":30,"distanceMeters":400,"inattentionCount":0,"drowsinessCount":0,"route":{"start":null,"end":null,"points":[]},"events":[],"waypoints":[],"skipped":{"events":0,"waypoints":0}}
```

#### 진행 중이던 주행 종료
```http
POST /api/drives/5/end
Authorization: Bearer eyJ…

{"endedAt":"2026-11-06T20:43:04+09:00","distanceMeters":700,"events":[{"type":"INATTENTION","detectedAt":"2026-11-06T20:42:44+09:00"}],"route":[[37.2351,127.1052],[37.2401,127.1001]],"waypoints":[]}
```
**200**
```json
{"driveId":5,"status":"ENDED","startedAt":"2026-11-06T20:42:13+09:00","endedAt":"2026-11-06T20:43:04+09:00","observedSeconds":51,"distanceMeters":700,"inattentionCount":1,"drowsinessCount":0,"route":{"start":{"latitude":37.2351,"longitude":127.1052},"end":{"latitude":37.2401,"longitude":127.1001},"points":[[37.2351,127.1052],[37.2401,127.1001]]},"events":[{"type":"INATTENTION","detectedAt":"2026-11-06T20:42:44+09:00","latitude":null,"longitude":null}],"waypoints":[],"skipped":{"events":0,"waypoints":0}}
```

## 시나리오 9. 이전 기록 조회

#### 이전 기록 진입 (date 생략 → 가장 최근 주행일)
```http
GET /api/reports/daily
Authorization: Bearer eyJ…
```
**200**
```json
{"date":"2026-11-06","summary":{"driveCount":4,"observedSeconds":4402,"distanceMeters":60000,"inattentionCount":4,"drowsinessCount":2},"hourly":[{"hour":0,"observedSeconds":1800,"inattentionCount":0,"drowsinessCount":1},{"hour":1,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":2,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":3,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":4,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":5,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":6,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":7,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":8,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":9,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":10,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":11,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":12,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":13,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":14,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":15,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":16,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":17,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":18,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":19,"observedSeconds":111,"inattentionCount":0,"drowsinessCount":0},{"hour":20,"observedSeconds":2491,"inattentionCount":4,"drowsinessCount":1},{"hour":21,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":22,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":23,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0}],"peak":{"inattention":{"hours":[20],"count":4},"drowsiness":{"hours":[0,20],"count":1}},"drives":[{"driveId":5,"startedAt":"2026-11-06T20:42:13+09:00","endedAt":"2026-11-06T20:43:04+09:00","observedSeconds":51,"distanceMeters":700,"inattentionCount":1,"drowsinessCount":0},{"driveId":4,"startedAt":"2026-11-06T20:41:13+09:00","endedAt":"2026-11-06T20:41:43+09:00","observedSeconds":30,"distanceMeters":400,"inattentionCount":0,"drowsinessCount":0},{"driveId":1,"startedAt":"2026-11-06T19:58:09+09:00","endedAt":"2026-11-06T20:40:10+09:00","observedSeconds":2521,"distanceMeters":28400,"inattentionCount":3,"drowsinessCount":1},{"driveId":3,"startedAt":"2026-11-05T23:30:00+09:00","endedAt":"2026-11-06T00:30:00+09:00","observedSeconds":3600,"distanceMeters":61000,"inattentionCount":1,"drowsinessCount":1}]}
```

#### 달력 열기
```http
GET /api/reports/calendar?month=2026-11
Authorization: Bearer eyJ…
```
**200**
```json
{"month":"2026-11","driveDates":["2026-11-04","2026-11-05","2026-11-06"]}
```

#### 날짜 선택 (2026-11-05: 자정 넘긴 주행의 앞부분)
```http
GET /api/reports/daily?date=2026-11-05
Authorization: Bearer eyJ…
```
**200**
```json
{"date":"2026-11-05","summary":{"driveCount":1,"observedSeconds":1800,"distanceMeters":30500,"inattentionCount":1,"drowsinessCount":0},"hourly":[{"hour":0,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":1,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":2,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":3,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":4,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":5,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":6,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":7,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":8,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":9,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":10,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":11,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":12,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":13,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":14,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":15,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":16,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":17,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":18,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":19,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":20,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":21,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":22,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":23,"observedSeconds":1800,"inattentionCount":1,"drowsinessCount":0}],"peak":{"inattention":{"hours":[23],"count":1},"drowsiness":{"hours":[],"count":0}},"drives":[{"driveId":3,"startedAt":"2026-11-05T23:30:00+09:00","endedAt":"2026-11-06T00:30:00+09:00","observedSeconds":3600,"distanceMeters":61000,"inattentionCount":1,"drowsinessCount":1}]}
```

#### 요일 막대 누름 (2026-11-04)
```http
GET /api/reports/daily?date=2026-11-04
Authorization: Bearer eyJ…
```
**200**
```json
{"date":"2026-11-04","summary":{"driveCount":1,"observedSeconds":2862,"distanceMeters":32800,"inattentionCount":3,"drowsinessCount":0},"hourly":[{"hour":0,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":1,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":2,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":3,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":4,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":5,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":6,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":7,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":8,"observedSeconds":2862,"inattentionCount":3,"drowsinessCount":0},{"hour":9,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":10,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":11,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":12,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":13,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":14,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":15,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":16,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":17,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":18,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":19,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":20,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":21,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":22,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":23,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0}],"peak":{"inattention":{"hours":[8],"count":3},"drowsiness":{"hours":[],"count":0}},"drives":[{"driveId":2,"startedAt":"2026-11-04T08:05:00+09:00","endedAt":"2026-11-04T08:52:42+09:00","observedSeconds":2862,"distanceMeters":32800,"inattentionCount":3,"drowsinessCount":0}]}
```

#### 주행 항목 누름 (시나리오 7 주행)
```http
GET /api/drives/1
Authorization: Bearer eyJ…
```
**200**
```json
{"driveId":1,"status":"ENDED","startedAt":"2026-11-06T19:58:09+09:00","endedAt":"2026-11-06T20:40:10+09:00","observedSeconds":2521,"distanceMeters":28400,"inattentionCount":3,"drowsinessCount":1,"route":{"start":{"latitude":37.073253,"longitude":127.132555},"end":{"latitude":37.2351,"longitude":127.1052},"points":[[37.073253,127.132555],[37.1102,127.115],[37.1542659,127.0861027],[37.2351,127.1052]]},"events":[{"type":"INATTENTION","detectedAt":"2026-11-06T20:08:10+09:00","latitude":37.0951,"longitude":127.1228},{"type":"INATTENTION","detectedAt":"2026-11-06T20:15:11+09:00","latitude":37.1102,"longitude":127.115},{"type":"DROWSINESS","detectedAt":"2026-11-06T20:21:11+09:00","latitude":37.1301,"longitude":127.1002},{"type":"INATTENTION","detectedAt":"2026-11-06T20:31:11+09:00","latitude":37.161,"longitude":127.085}],"waypoints":[{"waypointId":1,"placeId":1211,"placeName":"오산","placeType":"DROWSY_SHELTER","mode":"WAYPOINT","trigger":"DROWSINESS","addedAt":"2026-11-06T20:22:11+09:00","visited":true,"visitedAt":"2026-11-06T20:28:11+09:00"}]}
```

#### 주간 통계 탭 (weekStart 생략)
```http
GET /api/reports/weekly
Authorization: Bearer eyJ…
```
**200**
```json
{"weekStart":"2026-11-02","weekEnd":"2026-11-08","summary":{"driveCount":5,"observedSeconds":9064,"distanceMeters":123300,"inattentionCount":8,"drowsinessCount":2},"daily":[{"date":"2026-11-02","dayOfWeek":"MONDAY","driveCount":0,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"date":"2026-11-03","dayOfWeek":"TUESDAY","driveCount":0,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"date":"2026-11-04","dayOfWeek":"WEDNESDAY","driveCount":1,"observedSeconds":2862,"inattentionCount":3,"drowsinessCount":0},{"date":"2026-11-05","dayOfWeek":"THURSDAY","driveCount":1,"observedSeconds":1800,"inattentionCount":1,"drowsinessCount":0},{"date":"2026-11-06","dayOfWeek":"FRIDAY","driveCount":4,"observedSeconds":4402,"inattentionCount":4,"drowsinessCount":2},{"date":"2026-11-07","dayOfWeek":"SATURDAY","driveCount":0,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"date":"2026-11-08","dayOfWeek":"SUNDAY","driveCount":0,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0}],"hourly":[{"hour":0,"observedSeconds":1800,"inattentionCount":0,"drowsinessCount":1},{"hour":1,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":2,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":3,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":4,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":5,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":6,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":7,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":8,"observedSeconds":2862,"inattentionCount":3,"drowsinessCount":0},{"hour":9,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":10,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":11,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":12,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":13,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":14,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":15,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":16,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":17,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":18,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":19,"observedSeconds":111,"inattentionCount":0,"drowsinessCount":0},{"hour":20,"observedSeconds":2491,"inattentionCount":4,"drowsinessCount":1},{"hour":21,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":22,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":23,"observedSeconds":1800,"inattentionCount":1,"drowsinessCount":0}],"peak":{"inattention":{"hours":[20],"count":4},"drowsiness":{"hours":[0,20],"count":1}}}
```

#### 이전 주로 이동
```http
GET /api/reports/weekly?weekStart=2026-10-26
Authorization: Bearer eyJ…
```
**200**
```json
{"weekStart":"2026-10-26","weekEnd":"2026-11-01","summary":{"driveCount":0,"observedSeconds":0,"distanceMeters":0,"inattentionCount":0,"drowsinessCount":0},"daily":[{"date":"2026-10-26","dayOfWeek":"MONDAY","driveCount":0,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"date":"2026-10-27","dayOfWeek":"TUESDAY","driveCount":0,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"date":"2026-10-28","dayOfWeek":"WEDNESDAY","driveCount":0,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"date":"2026-10-29","dayOfWeek":"THURSDAY","driveCount":0,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"date":"2026-10-30","dayOfWeek":"FRIDAY","driveCount":0,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"date":"2026-10-31","dayOfWeek":"SATURDAY","driveCount":0,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"date":"2026-11-01","dayOfWeek":"SUNDAY","driveCount":0,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0}],"hourly":[{"hour":0,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":1,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":2,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":3,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":4,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":5,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":6,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":7,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":8,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":9,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":10,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":11,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":12,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":13,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":14,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":15,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":16,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":17,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":18,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":19,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":20,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":21,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":22,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0},{"hour":23,"observedSeconds":0,"inattentionCount":0,"drowsinessCount":0}],"peak":{"inattention":{"hours":[],"count":0},"drowsiness":{"hours":[],"count":0}}}
```

#### 주행 목록 (size=2, 첫 페이지)
```http
GET /api/drives?size=2
Authorization: Bearer eyJ…
```
**200**
```json
{"drives":[{"driveId":5,"startedAt":"2026-11-06T20:42:13+09:00","endedAt":"2026-11-06T20:43:04+09:00","observedSeconds":51,"distanceMeters":700,"inattentionCount":1,"drowsinessCount":0},{"driveId":4,"startedAt":"2026-11-06T20:41:13+09:00","endedAt":"2026-11-06T20:41:43+09:00","observedSeconds":30,"distanceMeters":400,"inattentionCount":0,"drowsinessCount":0}],"nextCursor":"MjAyNi0xMS0wNlQxMTo0MToxM1p8NA"}
```

#### 주행 목록 (다음 페이지)
```http
GET /api/drives?size=2&cursor=MjAyNi0xMS0wNlQxMTo0MToxM1p8NA
Authorization: Bearer eyJ…
```
**200**
```json
{"drives":[{"driveId":1,"startedAt":"2026-11-06T19:58:09+09:00","endedAt":"2026-11-06T20:40:10+09:00","observedSeconds":2521,"distanceMeters":28400,"inattentionCount":3,"drowsinessCount":1},{"driveId":3,"startedAt":"2026-11-05T23:30:00+09:00","endedAt":"2026-11-06T00:30:00+09:00","observedSeconds":3600,"distanceMeters":61000,"inattentionCount":1,"drowsinessCount":1}],"nextCursor":"MjAyNi0xMS0wNVQxNDozMFp8Mw"}
```

## 시나리오 10. 회원 탈퇴

driver1 의 기록은 남겨 두려고 별도 계정 driver2 로 진행

#### 가입
```http
POST /api/auth/signup

{"email":"driver2@example.com","name":"이도로","password":"Test1234!","ageConfirmed":true}
```
**201**
```json
{"userId":2,"email":"driver2@example.com","name":"이도로","grantType":"Bearer","accessToken":"eyJ…","accessTokenExpiresIn":1800,"refreshToken":"…","termsAgreementRequired":true}
```

#### 약관 동의
```http
POST /api/users/me/terms-agreements
Authorization: Bearer eyJ…

{"agreements":[{"termsId":1,"agreed":true},{"termsId":2,"agreed":true},{"termsId":3,"agreed":true},{"termsId":4,"agreed":true},{"termsId":5,"agreed":false}]}
```
**200**
```json
{"termsAgreementRequired":false,"agreements":[{"termsId":1,"code":"SERVICE","title":"서비스 이용약관","version":1,"required":true,"agreed":true,"agreedAt":"2026-11-06T20:43:16.262705+09:00"},{"termsId":2,"code":"PRIVACY","title":"개인정보 수집 · 이용","version":1,"required":true,"agreed":true,"agreedAt":"2026-11-06T20:43:16.262705+09:00"},{"termsId":3,"code":"CAMERA","title":"운전자 카메라 영상 처리","version":1,"required":true,"agreed":true,"agreedAt":"2026-11-06T20:43:16.262705+09:00"},{"termsId":4,"code":"LOCATION","title":"위치정보 이용","version":1,"required":true,"agreed":true,"agreedAt":"2026-11-06T20:43:16.262705+09:00"},{"termsId":5,"code":"MARKETING","title":"이벤트 · 혜택 알림 수신","version":1,"required":false,"agreed":false,"agreedAt":null}]}
```

#### 비밀번호 틀림 → 400
```http
DELETE /api/users/me
Authorization: Bearer eyJ…

{"password":"Wrong1234!"}
```
**400**
```json
{"timestamp":"2026-11-06T11:43:16.544579800Z","status":400,"error":"Bad Request","code":"PASSWORD_MISMATCH","message":"비밀번호가 일치하지 않습니다.","path":"/api/users/me"}
```

#### 탈퇴
```http
DELETE /api/users/me
Authorization: Bearer eyJ…

{"password":"Test1234!"}
```
**204**
```json

```

#### 탈퇴 후 남은 accessToken 으로 호출 → 404
```http
GET /api/users/me
Authorization: Bearer eyJ…
```
**404**
```json
{"timestamp":"2026-11-06T11:43:17.319099200Z","status":404,"error":"Not Found","code":"USER_NOT_FOUND","message":"사용자를 찾을 수 없습니다.","path":"/api/users/me"}
```

#### 탈퇴한 계정으로 로그인 → 401
```http
POST /api/auth/login

{"email":"driver2@example.com","password":"Test1234!"}
```
**401**
```json
{"timestamp":"2026-11-06T11:43:17.521844Z","status":401,"error":"Unauthorized","code":"LOGIN_FAILED","message":"이메일 또는 비밀번호가 올바르지 않습니다.","path":"/api/auth/login"}
```

#### 로그아웃 (driver1)
```http
POST /api/auth/logout

{"refreshToken":"…"}
```
**204**
```json

```

## 오류 응답 예시

#### 회원가입 입력값 오류 (입력란별 fieldErrors)
```http
POST /api/auth/signup

{"email":"haneul.kim@mail","name":"","password":"12345678","ageConfirmed":false}
```
**400**
```json
{"timestamp":"2026-11-06T11:43:17.913901200Z","status":400,"error":"Bad Request","code":"VALIDATION_FAILED","message":"입력값을 확인해 주세요.","path":"/api/auth/signup","fieldErrors":[{"field":"ageConfirmed","message":"만 14세 이상만 가입할 수 있습니다."},{"field":"email","message":"이메일 형식이 아닙니다. 예: name@mail.com"},{"field":"name","message":"이름을 입력해 주세요."},{"field":"password","message":"비밀번호는 영문과 숫자를 섞어 8~64자로 입력해 주세요."}]}
```

#### 이메일 중복
```http
POST /api/auth/signup

{"email":"driver1@example.com","name":"김하늘","password":"Test1234!","ageConfirmed":true}
```
**409**
```json
{"timestamp":"2026-11-06T11:43:18.123772400Z","status":409,"error":"Conflict","code":"EMAIL_DUPLICATED","message":"이미 사용 중인 이메일입니다.","path":"/api/auth/signup"}
```

#### 로그인 실패
```http
POST /api/auth/login

{"email":"driver1@example.com","password":"Wrong1234!"}
```
**401**
```json
{"timestamp":"2026-11-06T11:43:18.391627100Z","status":401,"error":"Unauthorized","code":"LOGIN_FAILED","message":"이메일 또는 비밀번호가 올바르지 않습니다.","path":"/api/auth/login"}
```

#### 토큰 없음
```http
GET /api/users/me
```
**401**
```json
{"timestamp":"2026-11-06T11:43:18.606066500Z","status":401,"error":"Unauthorized","code":"UNAUTHORIZED","message":"인증이 필요합니다.","path":"/api/users/me"}
```

#### 필수 약관을 false 로 보냄
```http
POST /api/users/me/terms-agreements
Authorization: Bearer eyJ…

{"agreements":[{"termsId":5,"agreed":true},{"termsId":1,"agreed":false}]}
```
**400**
```json
{"timestamp":"2026-11-06T11:43:18.761280500Z","status":400,"error":"Bad Request","code":"TERMS_REQUIRED_NOT_AGREED","message":"필수 약관에 모두 동의해 주세요.","path":"/api/users/me/terms-agreements"}
```

#### 없는 약관
```http
GET /api/terms/999
```
**404**
```json
{"timestamp":"2026-11-06T11:43:18.994435400Z","status":404,"error":"Not Found","code":"TERMS_NOT_FOUND","message":"약관을 찾을 수 없습니다.","path":"/api/terms/999"}
```

#### 관찰 시작 시각 범위 밖 (1시간 뒤)
```http
POST /api/drives
Authorization: Bearer eyJ…

{"clientDriveId":"306a1e5c-03c5-4729-ad7f-0228002cd200","startedAt":"2026-11-06T21:43:19+09:00"}
```
**400**
```json
{"timestamp":"2026-11-06T11:43:19.196907400Z","status":400,"error":"Bad Request","code":"VALIDATION_FAILED","message":"입력값을 확인해 주세요.","path":"/api/drives","fieldErrors":[{"field":"startedAt","message":"startedAt 은 현재로부터 7일 전 ~ 5분 후 사이여야 합니다."}]}
```

#### 시간대 없는 시각
```http
POST /api/drives
Authorization: Bearer eyJ…

{"clientDriveId":"306a1e5c-03c5-4729-ad7f-0228002cd200","startedAt":"2026-11-06T08:00:00"}
```
**400**
```json
{"timestamp":"2026-11-06T11:43:19.428069100Z","status":400,"error":"Bad Request","code":"VALIDATION_FAILED","message":"입력값을 확인해 주세요.","path":"/api/drives","fieldErrors":[{"field":"startedAt","message":"값의 형식이 올바르지 않습니다."}]}
```

#### 종료 업로드 값 오류 (감지 종류)
```http
POST /api/drives/1/end
Authorization: Bearer eyJ…

{"endedAt":"2026-11-06T10:00:00+09:00","distanceMeters":0,"events":[{"type":"SLEEP","detectedAt":"2026-11-06T09:00:00+09:00"}],"route":[],"waypoints":[]}
```
**400**
```json
{"timestamp":"2026-11-06T11:43:19.660339500Z","status":400,"error":"Bad Request","code":"VALIDATION_FAILED","message":"입력값을 확인해 주세요.","path":"/api/drives/1/end","fieldErrors":[{"field":"events[0].type","message":"값의 형식이 올바르지 않습니다."}]}
```

#### 없는 주행
```http
GET /api/drives/99999
Authorization: Bearer eyJ…
```
**404**
```json
{"timestamp":"2026-11-06T11:43:19.909319300Z","status":404,"error":"Not Found","code":"DRIVE_NOT_FOUND","message":"주행 기록을 찾을 수 없습니다.","path":"/api/drives/99999"}
```

#### 주간 통계 weekStart 가 월요일 아님
```http
GET /api/reports/weekly?weekStart=2026-11-03
Authorization: Bearer eyJ…
```
**400**
```json
{"timestamp":"2026-11-06T11:43:20.065370100Z","status":400,"error":"Bad Request","code":"INVALID_WEEK_START","message":"weekStart 는 월요일이어야 합니다.","path":"/api/reports/weekly"}
```

#### 날짜 형식 오류
```http
GET /api/reports/daily?date=2026-13-01
Authorization: Bearer eyJ…
```
**400**
```json
{"timestamp":"2026-11-06T11:43:20.236259400Z","status":400,"error":"Bad Request","code":"VALIDATION_FAILED","message":"입력값을 확인해 주세요.","path":"/api/reports/daily","fieldErrors":[{"field":"date","message":"date 는 yyyy-MM-dd 형식이어야 합니다."}]}
```

#### 쉼터 찾기 좌표 범위 오류
```http
POST /api/rest-places/nearby
Authorization: Bearer eyJ…

{"lat":999,"lng":127.0}
```
**400**
```json
{"timestamp":"2026-11-06T11:43:20.425507800Z","status":400,"error":"Bad Request","code":"VALIDATION_FAILED","message":"입력값을 확인해 주세요.","path":"/api/rest-places/nearby","fieldErrors":[{"field":"lat","message":"lat 은 -90 ~ 90 이어야 합니다."}]}
```
