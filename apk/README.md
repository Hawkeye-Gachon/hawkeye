# APK

Hawkeye Android 앱의 설치 파일입니다. 소스는 [`../app/`](../app)에 있습니다.

| 파일 | 용도 |
| --- | --- |
| [`Hawkeye_v0.2.0_arm64-v8a_release.apk`](Hawkeye_v0.2.0_arm64-v8a_release.apk) | 실제 기기와 차량(arm64) |
| [`Hawkeye_v0.2.0_x86_64_release.apk`](Hawkeye_v0.2.0_x86_64_release.apk) | x86_64 에뮬레이터 |

Apple Silicon Mac의 에뮬레이터는 arm64 파일을 씁니다.

## 설치

```sh
adb install -r apk/Hawkeye_v0.2.0_arm64-v8a_release.apk
```

## 알아둘 점

- 버전 0.2.0, Android 9(API 28) 이상.
- 로그인, 쉼터 조회, 주행 기록, 리포트는 백엔드 서버([`../backend/`](../backend))에 연결돼야 동작합니다. 서버 주소는 빌드할 때 정해집니다.
- 경고는 주행 속도가 낮으면 나가지 않습니다(전방 주시·눈 감김 시속 20km 미만, 졸음 시속 50km 미만).
- 시험용으로 Android debug 키로 서명했습니다. 다른 키로 서명한 빌드를 설치하려면 먼저 이 앱을 지워야 합니다.
- 다시 만들려면 `app` 폴더에서 `./gradlew :app:assembleRelease`를 실행합니다. 결과는 `app/app/build/outputs/apk/release/`에 생깁니다.
