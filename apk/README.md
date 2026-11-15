# APK

Hawkeye Android 앱의 설치 파일입니다. 소스는 [`../app/`](../app)에 있습니다.

| 파일 | 용도 |
| --- | --- |
| [`Hawk_Hawkeye_release.apk`](Hawk_Hawkeye_release.apk) | 실제 기기와 차량(arm64-v8a) |

파일 이름은 공모전 규칙(`팀명_앱이름_release.apk`)을 따랐습니다. 팀명은 Hawk, 앱 이름은 Hawkeye입니다.

## 먼저: 백엔드 서버가 준비돼 있어야 합니다

**이 앱은 백엔드 서버가 켜져 있고 앱이 그 주소로 연결될 수 있어야 동작합니다.** 서버가 없으면 로그인 화면에서 "서버에 연결할 수 없습니다"가 뜨고 그다음으로 넘어가지 않습니다.

- 서버가 맡는 것: 회원가입·로그인, 약관, 가까운 쉼터 조회, 주행 기록 저장, 리포트와 통계.
- 서버 실행 방법과 필요한 설정(PostgreSQL, 외부 API 키)은 [`../backend/README.md`](../backend/README.md)에 있습니다.
- 앱이 연결할 서버 주소는 **빌드할 때** 정해져 APK 안에 들어갑니다. 여기 올린 APK는 팀 개발 서버 주소로 빌드한 것이라, 그 서버가 꺼져 있으면 동작하지 않습니다.
- 다른 서버에 붙이려면 `app/local.properties`에 주소를 적고 APK를 다시 만듭니다.

  ```properties
  hawkeye.apiBaseUrl=http://<서버 주소>:<포트>/
  ```

  ```sh
  cd app && ./gradlew :app:assembleRelease
  ```

  주소를 적지 않으면 `http://10.0.2.2:8080/`(에뮬레이터에서 본 개발 PC)을 씁니다.

## 설치

```sh
adb install -r apk/Hawk_Hawkeye_release.apk
```

## 알아둘 점

- 버전 1.0.0, Android 9(API 28) 이상, arm64-v8a 기기.
- x86_64 에뮬레이터(Intel·AMD PC)에는 설치되지 않습니다. 그 환경에서 돌리려면 `app/app/build.gradle.kts`의 `abiFilters`에 `x86_64`를 더해 다시 만듭니다. Apple Silicon Mac의 에뮬레이터에는 이 파일이 그대로 설치됩니다.
- 경고는 주행 속도가 낮으면 나가지 않습니다(전방 주시·눈 감김 시속 20km 미만, 졸음 시속 50km 미만).
- 시험용으로 Android debug 키로 서명했습니다. 다른 키로 서명한 빌드를 설치하려면 먼저 이 앱을 지워야 합니다.
- 다시 만들려면 `app` 폴더에서 `./gradlew :app:assembleRelease`를 실행합니다. 결과는 `app/app/build/outputs/apk/release/app-release.apk`입니다.
