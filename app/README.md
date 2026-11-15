# Hawkeye (Android 앱)

**졸음을 알리는 데서 끝나지 않고, 차가 직접 운전자를 깨워 쉼터까지 데려가는 차량용 앱입니다.**

기존 졸음 경고는 소리를 한 번 내고 끝납니다. 운전자는 경고를 듣고도 계속 달립니다. Hawkeye는 경고 다음을 맡습니다. 졸음이 감지되면 차량의 음성·조명·공조·창문·시트를 단계적으로 움직여 운전자를 깨우고, 진행 방향의 가장 가까운 쉼터를 차량 내비 경유지로 넣어 실제 휴식까지 이어 줍니다.

현대자동차그룹 차량용 앱 개발 공모전(Pleos Connect) 출품작입니다. 이 폴더(`app/`)는 그중 Android 앱이며, 졸음 판정 라이브러리는 [`../ai/`](../ai), 서버는 [`../backend/`](../backend), 화면 시안은 [`../design/`](../design)에 있습니다.

## 한 번의 주행에서 일어나는 일

| 순서 | 앱이 하는 일 |
| --- | --- |
| 1. 출발 전 | 운전자가 평소 자세로 정면을 바라보면 그 방향을 기준으로 등록합니다. |
| 2. 관찰 | 전면 카메라 영상을 기기 안에서 분석해 눈 감김과 머리 방향을 추적합니다. |
| 3. 전방 주시 이탈 | 3초 이상 옆을 보면 음성으로 알리고 무드등을 주황으로 바꿉니다. 정면을 다시 보면 스스로 닫힙니다. |
| 4. 졸음 | 눈 감김이 이어지면 음성과 함께 차량 실내를 바꿔 운전자를 깨웁니다. |
| 5. 반응 확인 | 경고 뒤에도 5초 동안 눈을 뜨지 않으면 경고음과 함께 괜찮은지 묻고, 10초 동안 대답이 없으면 119 전화 화면을 엽니다. |
| 6. 쉼터 제안 | 괜찮다고 확인되면 진행 방향의 가장 가까운 휴게소·졸음쉼터를 음성으로 묻고, "응"이라고 답하면 차량 내비 경유지로 넣습니다. |
| 7. 쉼터까지 | 쉼터로 가는 동안 운전 상식 O/X 퀴즈를 음성으로 내 운전자가 깨어 있게 돕습니다. |
| 8. 휴식 | 그 쉼터 300m 안에서 멈추면 도착으로 기록하고, 휴식에 맞게 실내 온도를 맞춥니다. 주차(P)하면 등받이를 눕힙니다. 다시 출발하면 되돌립니다. |
| 9. 리포트 | 주행 경로 위에 감지 위치와 들른 쉼터를 보여 주고, 일간·주간 통계로 쌓습니다. |

4번부터 8번까지는 한 번에 하나씩, 앞 단계가 끝난 뒤에만 다음으로 넘어갑니다. 음성 안내와 질문이 서로 겹치지 않고, 진행 중에는 새 경고가 끼어들지 않습니다.

## Android 앱에서 만든 것

### 1. 차량이 직접 개입합니다

감지 단계에 따라 Pleos Vehicle SDK로 차량을 조작합니다. 단계가 올라갈수록 조치가 더해집니다.

| 단계 | 차량 조치 |
| --- | --- |
| 전방 주시 이탈 | 음성 안내, 무드등 주황 |
| 졸음 | 음성 안내, 무드등 빨강, 에어컨 켜기, 온도 18°C, 바람 최대, 내기 순환을 꺼 바깥 공기 유입 |
| 졸음 반복 | 위 조치에 더해 운전석 창문 20% 열기, 운전석 시트 통풍 최대 |
| 쉼터 도착 | 졸음 대응을 되돌리고, 실내 23°C·바람 약하게, 주차 상태(P)면 운전석 등받이 눕힘 |
| 다시 출발·관찰 종료 | 모두 조작 전 상태로 되돌림 |

- 조작하기 전의 값을 읽어 두었다가 그대로 되돌립니다. 앱이 차의 설정을 바꿔 놓은 채 끝나지 않습니다.
- 권한이 없거나 차량이 지원하지 않는 조작은 건너뛰고, **실제로 반영된 조치만** 화면에 표시합니다. 차가 왜 갑자기 바람을 세게 트는지 운전자가 알 수 있습니다.
- 차량 상태도 읽습니다. 안전벨트를 매지 않았으면 음성으로 알리고, 크루즈 주행 중인지와 기어 단을 관찰 화면에 보여 줍니다.

### 2. 쉼터를 차량 내비에 넣습니다

지도와 길 안내를 앱에서 다시 만들지 않고, Pleos NaviHelper로 차량 기본 내비에 쉼터를 넣습니다. 운전자는 평소 보던 내비 화면에서 그대로 안내를 받습니다. 내비 상태에 따라 다르게 동작합니다.

| 내비 상태 | 동작 |
| --- | --- |
| 목적지 안내 중 | 쉼터를 경유지로 추가 |
| 이미 경유지가 있음 | 쉼터를 먼저 들르도록 `현재 위치 → 쉼터 → 기존 경유지 → 목적지` 순서로 다시 요청 |
| 안내 중인 목적지 없음 | 쉼터를 목적지로 길 안내 시작 |
| 경유지가 가득 참 | 추가할 수 없음을 알리고 정차 후 정리하도록 안내 |
| 내비 응답 없음 | 실패를 알리고 다시 시도. 관찰은 계속됩니다. |

### 3. 화면을 누르지 않아도 됩니다

- 쉼터 제안은 Pleos TextToSpeech로 묻고 SpeechToText로 대답을 듣습니다. 질문이 끝나면 신호음이 울리고, "응" 한마디면 경유지가 추가됩니다. 알아듣지 못하면 한 번 더 묻습니다.
- 대답이 없으면 스스로 관찰 화면으로 돌아갑니다.
- 쉼터로 가는 동안의 퀴즈도 문제와 대답을 음성으로만 주고받습니다. 화면에는 띄우지 않습니다.
- 말하기와 듣기는 한 번에 하나만 합니다. 안내가 끝나기 전에 다음 안내가 시작되거나, 안내 음성을 운전자의 대답으로 듣는 일이 없습니다.

### 4. 반응이 없으면 119로 이어 줍니다

- 졸음 경고 뒤에도 눈을 뜨지 않으면 경고음과 함께 "괜찮으세요?"라고 묻습니다.
- 눈을 뜨거나, 대답하거나, 화면을 누르면 취소되고 쉼터 제안으로 넘어갑니다.
- 10초 동안 반응이 없으면 119가 입력된 전화 화면을 열고 현재 위치(위도·경도)를 화면에 보여 줍니다. Android는 일반 앱이 긴급 번호로 바로 전화를 걸지 못하게 하므로, 통화 버튼은 사람이 눌러야 합니다.

### 5. 영상은 기기 밖으로 나가지 않습니다

- 감지는 모두 기기 안에서 합니다. 카메라 프레임 → MediaPipe 얼굴 랜드마크 → `ai/kotlin` 졸음 판정 라이브러리 순서로 처리하며(초당 약 15프레임), 영상과 사진은 저장하지도 서버로 보내지도 않습니다.
- 주행 기록(감지 시각과 위치, 단순화한 경로, 들른 쉼터)은 앱 안에 모았다가 종료할 때 한 번에 올립니다. 올리지 못하면 기기에 남겨 두었다가 다음 실행 때 다시 올립니다.
- 사람마다 눈 크기와 앉는 자세가 다르므로, 시작 후 약 1분 동안 운전자의 평소 상태를 익혀 판정에 반영합니다. 그동안에도 졸음과 고개 돌림은 감지합니다.
- 저속에서는 경고하지 않습니다(전방 주시·눈 감김 시속 20km 미만, 졸음 시속 50km 미만). 신호 대기 중에 옆을 본다고 경고가 울리지 않습니다.

### 6. Pleos 차량이 아니어도 멈추지 않습니다

Pleos 연동은 모두 인터페이스 뒤에 두었습니다. 연결에 실패하면 해당 기능만 건너뛰거나 Android 기본 기능으로 바꿔 동작합니다.

| 기능 | Pleos 차량 | 그 밖의 기기 |
| --- | --- | --- |
| 음성 안내·응답 | 차량 스피커·마이크 | Android 기본 음성 |
| 위치 | 차량 위치(FusedLocation) | Android 위치 |
| 실내 조치·내비 | 실제 조작 | 건너뜀 |

### 7. 차량 화면에 맞춘 UI

- 차량 센터 디스플레이(가로 1280 × 800dp)를 기준으로 그렸고, 다른 크기에서는 같은 비율로 확대·축소합니다.
- 주행 중 화면은 모두 왼쪽 카메라 카드와 오른쪽 시트의 같은 구성입니다. 경고가 떠도 배치는 그대로이고 내용만 바뀝니다.
- 경고는 화면 가장자리의 빛으로 먼저 전합니다(전방 주시 이탈은 주황, 졸음은 깜빡이는 빨강). 전방을 보고 있어도 주변 시야로 알 수 있습니다.
- 서체는 42dot의 Asta Sans를 씁니다.

배치 이유와 시안 파일은 [`../design/README.md`](../design/README.md)에 정리했습니다.

## 사용한 Pleos Connect SDK

| SDK | 버전 | 용도 |
| --- | --- | --- |
| NaviHelper | 3.0.5 | 안내 상태·경유지 조회, 쉼터 경유지 추가, 쉼터로 길 안내 |
| Vehicle | 2.3.2 | 공조(HVAC), 무드등(Light), 창문(Window), 시트 통풍(Seat) 제어와 기어·크루즈·안전벨트 상태 읽기 |
| TextToSpeech | 3.3.1 | 경고와 쉼터 제안 음성 안내 |
| SpeechToText | 3.3.1 | 쉼터 제안에 대한 음성 응답 인식 |
| FusedLocation | 1.1.3 | 차량 위치 |

## 기술 구성

| 구분 | 내용 |
| --- | --- |
| 언어·UI | Kotlin, Jetpack Compose |
| 얼굴 분석 | MediaPipe Tasks Vision 0.10.35(얼굴 랜드마크), `ai/kotlin` 졸음 판정 라이브러리 |
| 서버 통신 | Retrofit, OkHttp |
| 테스트 | 단위 테스트, 전체 화면을 1280 × 800dp로 그려 시안과 비교하는 화면 스냅숏 테스트(Roborazzi) |
| 지원 버전 | minSdk 28, targetSdk 36 |

## 실행 방법

### 1. APK 설치

release APK는 저장소 루트의 [`apk/`](../apk) 폴더에 있습니다. 실제 기기와 차량(arm64-v8a)용입니다.

```sh
adb install -r apk/Hawk_Hawkeye_release.apk
```

### 2. 서버 연결

로그인, 쉼터 조회, 주행 기록, 리포트는 백엔드 서버(`../backend/`)가 필요합니다. 서버 주소는 빌드할 때 정합니다(아래 3번).

### 3. 소스에서 빌드

필요한 것: JDK 17, Android SDK(compileSdk 36). 저장소를 받을 때 `ai/` 폴더가 함께 있어야 합니다(졸음 판정 라이브러리를 포함 빌드로 씁니다).

```sh
cd app
./gradlew :app:assembleRelease      # app/build/outputs/apk/release/app-release.apk
./gradlew :app:testDebugUnitTest    # 단위 테스트와 화면 스냅숏
```

네이티브 라이브러리(MediaPipe, ONNX Runtime 등)가 커서 APK는 실제 기기와 차량(arm64-v8a)용으로만 만듭니다. x86_64 에뮬레이터에서 돌리려면 `app/build.gradle.kts`의 `abiFilters`에 `x86_64`를 더해 다시 만듭니다. release APK는 시험용으로 Android debug 키로 서명합니다.

실제 서버에 연결하려면 `app/local.properties`에 서버 주소를 적습니다. 적지 않으면 `http://10.0.2.2:8080/`(에뮬레이터에서 본 개발 PC)을 씁니다.

```properties
hawkeye.apiBaseUrl=http://<서버 주소>:<포트>/
```

## 테스트 환경과 필요한 설정

- **화면 규격**: 가로 고정, 1280 × 800dp 기준으로 설계했고 화면 크기에 맞춰 비율대로 확대·축소합니다.
- **Android**: minSdk 28, targetSdk 36.
- **확인한 기기**: 갤럭시 탭 S9(SM-X710, Android 16 / One UI 8.0, 화면 2560 × 1600)에서 카메라 감지, 음성 안내와 응답, 화면 흐름을 확인했습니다. 이때 차량과 서버의 응답은 개발용 가상 구현으로 대신했고, 그 구현은 제출본에서 뺐습니다.
- **Pleos Connect 에뮬레이터**: 아직 실행하지 못했습니다. Pleos SDK 호출 코드는 SDK 라이브러리에 맞춰 작성하고 빌드했으며, 차량 제어 권한(`CONTROL_CAR_*`)과 실차 동작은 2차 심사 전에 검증할 예정입니다.
- **권한**: 카메라(필수), 위치, 마이크(음성 응답), Pleos 차량 권한(내비·공조·조명·창문·시트·안전·주행·위치·음성).
- **네트워크**: 서버 연결이 필요합니다. Pleos가 아닌 기기에서는 음성 안내와 음성 인식에도 네트워크나 기기에 내려받은 한국어 음성 데이터가 필요합니다.

## 알아둘 점

- 졸음 판정은 현재 눈 감김 규칙(PERCLOS, 감김 지속 시간)으로 동작합니다. 학습된 모델은 아직 포함되지 않았습니다(`ai/README.md`).
- 경고는 주행 속도가 낮으면 나가지 않습니다(시선·감김 시속 20km 미만, 졸음 시속 50km 미만). 세워 둔 차나 책상 위에서는 경고가 나지 않습니다.
- 이 앱의 경고와 안내는 운전을 돕는 보조 수단이며 안전 시스템이 아닙니다.

## 코드 구조

### 모듈

| 모듈 | 내용 |
| --- | --- |
| `app/` | 화면(Jetpack Compose), 앱 상태, Pleos 연동 |
| `analysis/` | MediaPipe 얼굴 랜드마크 추론 |
| `data/` | 서버 API 클라이언트(Retrofit), 로그인 토큰 저장과 갱신 |
| `domain/`, `entity/` | Android에 의존하지 않는 순수 Kotlin 인터페이스와 자료형 |
| `../ai/kotlin` | 졸음 판정 라이브러리. Gradle 포함 빌드(`includeBuild`)로 가져옵니다. |

### 감지에서 차량 조작까지

```text
카메라 프레임
   │  CameraX, 약 15fps, 한 번에 한 프레임만 처리
   ▼
DrowsyDetector ── MediaPipe 얼굴 랜드마크 ──▶ DrowsinessMonitor(ai/kotlin)
   │  졸음 / 전방 주시 이탈 / 정면 복귀
   ▼
DriveDirector   경고 → 반응 확인(119) → 쉼터 제안 → 퀴즈 → 도착 안내를 한 번에 하나씩
   │
   ├─▶ VoiceGuide          음성 안내, 대답 듣기
   ├─▶ VehicleCabin        무드등·공조·창문·시트, 차량 상태 읽기
   ├─▶ IVehicleNavigation  쉼터 경유지 추가
   ├─▶ HawkeyeApi          쉼터 조회, 주행 기록 업로드
   └─▶ DriveRecorder       감지·경로·쉼터 방문 기록
   ▼  StateFlow
화면(Compose)  진행 단계를 따라 바뀜
```

### 주요 코드

경로는 `app/src/main/java/com/personal/driverguard/` 기준입니다.

| 위치 | 역할 |
| --- | --- |
| [`ui/Nav.kt`](app/src/main/java/com/personal/driverguard/ui/Nav.kt) | 전체 화면 목록(`Screen`)과 화면 이동. 관찰 중의 화면은 진행 단계를 따라갑니다. |
| [`ui/screens/`](app/src/main/java/com/personal/driverguard/ui/screens) | 화면. `Start`(로그인·가입·약관), `Home`, `Observe`(설정·관찰·경고·119 확인·쉼터 제안·도착), `Records`(리포트·통계) |
| [`ui/design/`](app/src/main/java/com/personal/driverguard/ui/design), [`ui/theme/Tokens.kt`](app/src/main/java/com/personal/driverguard/ui/theme/Tokens.kt) | 공통 컴포넌트(카드, 시트, 버튼, 가장자리 빛, 차트, 경로 지도)와 시안의 색·글자·치수 토큰 |
| [`ui/state/DriveDirector.kt`](app/src/main/java/com/personal/driverguard/ui/state/DriveDirector.kt) | 감지 뒤의 흐름을 정해진 순서와 간격으로 진행합니다. |
| [`ui/state/HawkeyeViewModel.kt`](app/src/main/java/com/personal/driverguard/ui/state/HawkeyeViewModel.kt) | 앱 전체 상태. 음성·차량·내비·서버를 실제로 호출합니다. |
| [`ui/state/VoiceQueue.kt`](app/src/main/java/com/personal/driverguard/ui/state/VoiceQueue.kt) | 말하기와 듣기가 겹치지 않게 차례를 정합니다. |
| [`ui/state/DrowsyQuiz.kt`](app/src/main/java/com/personal/driverguard/ui/state/DrowsyQuiz.kt) | 음성 O/X 퀴즈의 문제와 대답 알아듣기 |
| [`ui/state/DriveRecorder.kt`](app/src/main/java/com/personal/driverguard/ui/state/DriveRecorder.kt), [`ui/state/PendingUploads.kt`](app/src/main/java/com/personal/driverguard/ui/state/PendingUploads.kt) | 한 번의 주행 기록과 쉼터 도착 판정, 아직 올리지 못한 기록 보관 |
| [`platform/DrowsyDetector.kt`](app/src/main/java/com/personal/driverguard/platform/DrowsyDetector.kt) | 카메라 프레임을 MediaPipe와 판정 라이브러리에 잇는 감지기 |
| [`platform/VehicleCabin.kt`](app/src/main/java/com/personal/driverguard/platform/VehicleCabin.kt), [`platform/VoiceGuide.kt`](app/src/main/java/com/personal/driverguard/platform/VoiceGuide.kt) | 차량 실내 조작과 음성의 인터페이스, 개입 단계 정의 |
| [`platform/pleos/`](app/src/main/java/com/personal/driverguard/platform/pleos) | Pleos SDK 구현: `PleosVehicleCabin`, `PleosVehicleNavigation`, `PleosVoiceGuide`, `PleosFusedLocation` |
| [`PlatformLocator.kt`](app/src/main/java/com/personal/driverguard/PlatformLocator.kt) | 차량·음성·위치 구현을 만들어 주는 곳 |

### 구현에서 신경 쓴 점

- **한 번에 하나씩 진행**: 경고, 119 확인, 쉼터 질문, 퀴즈, 도착 안내가 따로 움직이면 음성과 화면이 서로 끼어듭니다. 그래서 한 곳(`DriveDirector`)에서 순서대로만 진행하고, 순서와 간격은 가상 시간으로 돌리는 단위 테스트로 확인합니다.
- **구현을 바꿔 끼우는 구조**: 차량 기능은 `VehicleCabin`, `VoiceGuide`, `IVehicleNavigation` 인터페이스로만 부릅니다. 그 뒤에 Pleos 구현과 Android 기본 구현이 있어 화면과 상태 코드는 지금 차량인지 태블릿인지 모릅니다.
- **콜백을 순서 있는 호출로**: Pleos SDK는 결과를 콜백으로 줍니다. 호출 하나를 코루틴으로 감싸 응답이 올 때까지 기다리고, 시간 안에 오지 않으면 실패로 처리합니다. 요청이 겹치지 않도록 한 번에 하나씩 보냅니다.
- **실패해도 관찰은 계속**: 차량 연결, 내비 응답, 감지기, 서버 등록 중 어느 하나가 실패해도 나머지는 계속 동작합니다. 감지기가 멈추면 화면에 사유를 보여 줍니다.
- **기록을 잃지 않게**: 주행 중에도 기록을 기기에 주기적으로 저장하고, 올리지 못한 기록은 다음 실행 때 같은 계정으로만 다시 올립니다.
- **프레임이 밀리지 않게**: 처리 중에 들어온 프레임은 버립니다. MediaPipe가 결과를 돌려주지 않는 경우에도 일정 시간이 지나면 다음 프레임을 받습니다.
- **화면은 시안과 대조**: 모든 화면을 1280 × 800dp로 그려 이미지로 남기는 테스트가 있어, 화면이 시안에서 벗어나면 바로 알 수 있습니다.
