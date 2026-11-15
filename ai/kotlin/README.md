# 졸음 판정 Kotlin 모듈 (`drowsy-core`, `drowsy-onnx`)

## 이 모듈은 무엇인가

Python 참조 구현(`drowsy/`의 일곱 모듈: types, signals, calibrate, blink, step_features, rules, pipeline)을 Kotlin으로
옮긴 순수 JVM 라이브러리다. 카메라와 MediaPipe는 앱이 가진다. 앱은 MediaPipe 결과 하나마다 `update`를 한 번 부르고
`DriverState`를 받는다.

- `:core`(아티팩트 `drowsy-core`): 파이프라인 이식, 두 설정 파일(`feature_spec_v0.json`, `rules.json`) 읽기와 검사,
  시작 검사.
- `:onnx`(아티팩트 `drowsy-onnx`): ONNX 모델 어댑터와 앱이 부르는 입구 `hawkeye.drowsy.DrowsinessMonitor`. `../export/`의
  세 파일(`drowsiness_v0.onnx`, `feature_spec_v0.json`, `rules.json`)을 JAR의 `hawkeye/drowsy/bundle/`에 바이트 그대로 넣는다.
- 두 모듈은 Android와 MediaPipe를 import하지 않는다. ONNX Runtime은 `compileOnly`라서 앱이 넣는다.
- 그룹은 `hawkeye`, 버전은 `0.1.0`이다. 컴파일러는 Kotlin 2.1.21이고 언어·API는 2.0 모드, `kotlin-stdlib` 2.0.21,
  `jvmTarget` 11이다.

## 앱이 더하는 줄

팀 저장소에서 이 저장소는 `ai/`가 되고, 이 Gradle 프로젝트는 `ai/kotlin/`이다. 앱의 Gradle 루트 `app/`에서 아래 줄을
더한다. 이 형태는 버리는 사본(`app/` 자리의 소비자 + `ai/` 사본)에서 Gradle 8.14, Kotlin Gradle 플러그인 2.2.21,
`jvmToolchain(17)`, JDK 17로 확인했다. 순수 JVM 모듈에서는 `main`을 돌렸고, AGP 8.12.1 Android 라이브러리 모듈
(compileSdk 35, minSdk 28)에서는 `assembleDebug`가 통과했다.

`app/settings.gradle.kts`:

```kotlin
includeBuild("../ai/kotlin") {
    dependencySubstitution {
        substitute(module("hawkeye:drowsy-onnx")).using(project(":onnx"))
    }
}
```

프로젝트 이름이 `core`, `onnx`라서(JAR 이름만 `drowsy-*`) `dependencySubstitution` 블록이 있어야 한다. 없으면
`Could not find hawkeye:drowsy-onnx:0.1.0.`으로 멈춘다.

판정 모듈의 `build.gradle.kts`(예: MediaPipe를 쓰는 모듈):

```kotlin
implementation("hawkeye:drowsy-onnx:0.1.0")
implementation("com.microsoft.onnxruntime:onnxruntime-android:1.28.0")
implementation("com.google.mediapipe:tasks-vision:0.10.35")   // 앱에 이미 있다
```

- 순수 JVM 모듈(데스크톱 `main`)에서는 `onnxruntime-android` 대신 `runtimeOnly("com.microsoft.onnxruntime:onnxruntime:1.28.0")`을
  넣는다. 위 확인의 JVM 소비자가 이 줄로 돌았다. Android 모듈의 로컬 단위 시험(JVM)에서 돌려 보지는 않았다.
- `drowsy-core`는 `drowsy-onnx`를 따라 들어온다(`api`). 따로 적지 않는다.
- `kotlin-stdlib` 2.0.21은 앱의 2.2.21로 올라간다(확인: 소비자의 `dependencies` 출력).
- `ai/export/`에 세 파일이 있어야 한다. 하나라도 없으면 `:onnx:processResources`가 멈춘다.
- 앱의 빌드는 이 프로젝트의 시험을 돌리지 않는다.

## MediaPipe 결과 → `update` (변환 함수)

이 함수를 앱 쪽에 둔다(이 모듈에는 MediaPipe 의존이 없다). 위 확인에서 `tasks-vision` 0.10.35에 대고 컴파일했다.

```kotlin
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import hawkeye.drowsy.DrowsinessMonitor
import hawkeye.drowsy.core.session.DriverState

fun DrowsinessMonitor.updateFromMediaPipe(
    result: FaceLandmarkerResult,
    imageWidth: Int,
    imageHeight: Int,
    speedKmh: Double?,
): DriverState? {
    val tMs = result.timestampMs()
    val faces = result.faceLandmarks()
    if (faces.isEmpty()) return updateNoFace(tMs, imageWidth, imageHeight, speedKmh)

    val landmarks = faces[0]
    val x = FloatArray(landmarks.size) { landmarks[it].x() }
    val y = FloatArray(landmarks.size) { landmarks[it].y() }

    val blendshapes = result.faceBlendshapes()
        .orElseThrow { IllegalStateException("FaceLandmarker needs setOutputFaceBlendshapes(true)") }[0]
    val names = blendshapes.map { it.categoryName() }
    val scores = FloatArray(blendshapes.size) { blendshapes[it].score() }

    val matrix = result.facialTransformationMatrixes()
        .orElseThrow { IllegalStateException("FaceLandmarker needs setOutputFacialTransformationMatrixes(true)") }[0]

    return update(tMs, imageWidth, imageHeight, x, y, names, scores, matrix, speedKmh)
}
```

함수가 기대는 것(SPEC 1장):

- 얼굴 0만 쓴다. 얼굴 목록이 비었으면 얼굴 없는 프레임이다. 그런 결과도 빠짐없이 넣는다(G10).
- 랜드마커는 `setOutputFaceBlendshapes(true)`, `setOutputFacialTransformationMatrixes(true)`로 만든다(G4). 아니면 위 함수가
  `IllegalStateException`을 던진다. 실행 모드는 `LIVE_STREAM`, `setNumFaces(1)`, 신뢰도 임계값은 기본값이다(G2, G3, G5).
  모델 파일은 공개 `face_landmarker.task`(float16 v1, sha256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`)다(G1).
- 랜드마크 x, y는 MediaPipe가 준 정규화 좌표 그대로다. 478점 모델이어야 한다(홍채 468~477). 아니면 `update`가
  `IllegalArgumentException`을 던진다.
- 행렬은 MediaPipe가 준 16개를 순서 그대로(열 우선) 넘긴다.
- `imageWidth`, `imageHeight`는 MediaPipe에 넘긴, 똑바로 세운 그림의 크기(px)다(G9).
- 그림은 `detectAsync`에 넘기기 전에 앱이 똑바로 세운다. `ImageProcessingOptions`의 `rotationDegrees`는 쓰지 않는다(G6).
- 분석 프레임은 좌우 반전하지 않는다. 반전은 미리보기에만 건다(G7). 반전된 입력이면 yaw와 roll의 부호가 뒤집히고
  왼쪽·오른쪽 신호가 서로 바뀐다(SPEC 5.6절). 라이브러리는 이것을 알아챌 수 없다.

## 호출하는 곳 (예)

이 코드도 위 확인에서 컴파일했다. 기기에서 돌리지는 않았다.

```kotlin
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import hawkeye.drowsy.DrowsinessMonitor
import hawkeye.drowsy.core.DriverStateEvent
import hawkeye.drowsy.core.check.sha256Hex
import hawkeye.drowsy.core.session.DriverState

/** 운전 한 번 = 인스턴스 하나. taskFileBytes = MediaPipe에 준 face_landmarker.task 파일의 바이트. */
class DrowsyWatch(taskFileBytes: ByteArray, private val onState: (DriverState) -> Unit,
                  private val onAlert: (DriverStateEvent) -> Unit) : AutoCloseable {
    private val monitor = DrowsinessMonitor.open(sha256Hex(taskFileBytes))   // 시작 검사에 떨어지면 IllegalStateException
    @Volatile var speedMetersPerSecond: Float? = null                       // 차량 속도. 모르면 null

    /** FaceLandmarker(LIVE_STREAM) 결과마다 한 번. width, height = MediaPipe에 준 세운 그림의 크기. */
    fun onResult(result: FaceLandmarkerResult, width: Int, height: Int) {
        val kmh = speedMetersPerSecond?.let { it.toDouble() * 3.6 }
        val state = monitor.updateFromMediaPipe(result, width, height, kmh) ?: return   // null = 버린 프레임
        state.primaryAlert?.let(onAlert)                                     // 이번 프레임에 새로 난 가장 급한 경고
        onState(state)                                                       // active, calibration 등을 그린다
    }

    override fun close() = monitor.close()                                   // 운전이 끝날 때
}
```

- `open(mediapipeModelSha256)`은 JAR에 든 세 파일로 연다. 부를 때마다 클래스 경로에서 다시 읽는다.
- 세 파일을 앱이 가지고 싶으면 `open(modelFileName, modelBytes, featureSpecBytes, rulesBytes, mediapipeModelSha256)`을 쓴다.
- `mediapipeModelSha256`은 앱이 실제로 읽은 `face_landmarker.task`의 sha256이다. 소문자 16진수 64자이고 있는 그대로 비교한다.
- 시작 검사(SPEC 15장: 해시, 설정, 모델 입출력 비교)에 떨어지면 `IllegalStateException`이다. 메시지에 문제를 모두 적는다.
- 속도는 km/h다. m/s 값이면 3.6을 곱한다. 모르면 null이다(null이면 속도로 경고를 막지 않는다). Float m/s를
  `toDouble()` 한 뒤 곱하면 경계값이 조금 어긋난다. 예: 20 km/h가 Float m/s 5.55555534362793으로 오면 ×3.6은
  19.999999237060546이라 20 km/h 기준 경고가 막힌다(계획 K26, numpy float32로 다시 계산함).
- `tMs`는 양수이고 엄격히 늘어나는 실제 ms다. 카메라 프레임 시각을 쓴다. 프레임 번호를 쓰지 않는다(G8). 카메라 시각이
  ns이면 1,000,000으로 나눈다. `tMs`가 직전에 받아들인 값 이하인 프레임은 파이프라인이 버린다. `update`는 null을
  돌려주고 상태는 바뀌지 않으며 `counters.droppedFrames`만 하나 는다(SPEC 6장). 위 함수는 `result.timestampMs()`를 쓴다
  (= `detectAsync`에 넘긴 값, G9).
- 호출은 한 번에 하나씩이다. 겹치면 `IllegalStateException("update must be called one at a time")`이다. 새 운전(새
  운전자)마다 새 인스턴스를 연다(G11). 새 인스턴스는 보정을 처음부터 한다.
- `close()`는 ONNX 세션을 닫는다. 두 번 불러도 된다. 다른 스레드의 `update`가 돌고 있으면 끝나기를 기다린다. 닫은 뒤의
  `update`는 `IllegalStateException("the DrowsinessMonitor is closed")`다.
- 파이프라인이 예외를 내면 그 예외를 그대로 던지고 멈춘다. 그 뒤의 `update`와 `updateNoFace`는 모두
  `IllegalStateException("the DrowsinessMonitor stopped after a processing error: ...")`(cause = 그 예외)이다.
  `lastState`, `counters`는 멈춘 뒤에도 읽히고 `close()`도 부를 수 있다. 계속하려면 새 인스턴스를 연다(새 보정).
- 인자 검사(필요한 랜드마크·블렌드셰이프가 없음, 행렬이 16개가 아님)의 `IllegalArgumentException`은 상태를 바꾸지
  않으므로 멈추지 않는다. 그러나 파이프라인이 낸 예외도 `IllegalArgumentException`일 수 있다(예: 아래 "알려진 차이"의 NaN
  정렬). 예외의 종류로 둘을 가리지 않는다. 다음 호출이 위의 "stopped after a processing error" 예외를 던지면 멈춘 것이다.
- NaN과 무한대는 검사하지 않는다. 넣지 않는 것은 앱의 일이다(SPEC 2장, 아래 "알려진 차이").

## `DriverState`가 담는 것

받아들인 프레임마다 하나다.

- `alerts`: 이번 프레임에 나는 경고(우선순위 순서). 경고는 한 번만 나므로 매 프레임 확인한다. `primaryAlert`는 그 첫째
  (가장 급한 것)이고 없으면 null이다. 속도가 `rules.json`의 `min_speed_kmh`보다 작으면 그 종류의 경고는 막힌다
  (`gaze_away_long`, `vats`, `microsleep`, `sleep`은 20.0, `drowsy`는 50.0, `calibrating`, `unable_to_analyze`는 0.0).
- `events`: 이번 프레임에 난 이벤트(onset, end). 속도와 무관하다.
- `active`: 지금 진행 중인 이벤트. `rules.json`의 `alert_priority` 순서(sleep, microsleep, gaze_away_long, vats, drowsy,
  unable_to_analyze, calibrating)다. `sleep` onset은 `microsleep`을 지운다. `isActive(type)`로 물을 수 있다.
- `calibration`: 기준선이 잡히기 전에는 `CALIBRATING`, 그 뒤에는 `OK`, `OFF_AXIS`, `FALLBACK_GLOBAL`. 시작 뒤 적어도
  60초는 `CALIBRATING`이다. 그때까지 얼굴 프레임 300개, 얼굴 시간 비율 0.5, EAR 분위수 0.12~0.45를 채우지 못하면 30초씩
  늘리고, 180초에도 못 채우면 전역 상수 기준선(`FALLBACK_GLOBAL`)을 쓴다(`feature_spec_v0.json`의 `pipeline.baseline`).
- 그 밖에 `tMs`, `sessionMs`(첫 상태부터 잰 ms), `gazeZone`, `baseline`, `counters`(계수 다섯), `speedKmh`(이 프레임이
  쓴 속도), `output`(파이프라인 출력 그대로).
- 졸음 판정은 지금 PERCLOS 규칙에서 나온다(`rules.json`의 `drowsiness.source: perclos`). v0 모델은 무작위 가중치라
  `output.probs`의 확률은 뜻이 없다. 화면에 졸음 정도로 보이지 않는다.

## 시험

`kotlin/`에서 JDK 17로 `./gradlew build`를 돌린다. 시험은 `../tests/vectors/`(벡터 102개), `../export/`, Python 모듈과
설정 파일(참조 지문 계산)을 그 자리에서 읽는다. `:onnx:test`는 ONNX Runtime 데스크톱 JAR로 모델을 돌린다. 모든 시험은
`ORT_DISABLE_TELEMETRY=1`로 돈다.

`.diffdata/`는 Python 참조 구현을 돌려 내보낸 차등 비교 자료다(git 제외, 형식은 `difftools/FORMATS.md`). 그 자료를 읽는
시험(부품 호출 기록 C-*, 변조 파일, 진행 중 이벤트 기대값, 묶음 여덟 S-quick, S-odd, S-edge, F-stat, F-inj, F-cal, F-calib,
G-sig)은 `READY_small.txt`나 `READY_corpora.txt`가 없으면 건너뛴다. 그래서 저장소를 새로 받은 사본에서는 그 시험이
건너뛰어지고 나머지 시험으로 `./gradlew build`가 통과한다. 자료가 다른 참조 파일에서 만들어졌으면(reference_id나
depends_on이 다르면) 건너뛰지 않고 실패한다.

차등 비교 자료는 이식 작업을 한 PC에서 만들었고 저장소에 없다. 자료를 만드는 내보내기 명령(`difftools/export.py`의 `small`,
`corpora`, `full`)은 생성기 사본 폴더 `difftools/vendor/`를 읽는데, 그 폴더도 저장소에 없다(git 제외, 그 PC의 경로가
들어 있다). 그래서 저장소만으로는 자료를 다시 만들 수 없다.
그 폴더가 있는 PC에서 다시 만드는 명령(저장소 루트에서):

```
nice -n 19 .venv/bin/python kotlin/difftools/export.py small
nice -n 19 .venv/bin/python kotlin/difftools/export.py corpora --workers 10
nice -n 19 .venv/bin/python kotlin/difftools/export.py check
```

## 가정

- 버전 0.1.0은 임시로 고른 값이다. 팀이 정한 것이 아니다.
- A1: 사칙연산, `sqrt`, `floor`, `ceil`, 비교는 JVM·Android와 CPython에서 같은 double을 낸다. JVM에서는 STATE 차등 비교가
  모든 수를 `==`로 비교했고 알려진 차이 밖의 차이는 0이었다. Android에서는 확인하지 않았다.
- A2: `atan2`, `hypot`, `sin`, `cos`(StrictMath, 계획 K20)는 CPython과 마지막 몇 자리만 다르다. 잰 가장 큰 차이는 부품
  자료에서 2.84e-14다. 그 차이가 임계값을 가르는 경우가 있다(아래 "알려진 차이").
- A3: JVM의 `String.toDouble()`은 가장 가까운 double을 준다(숫자 표 10,020개에서 비트 차이 0).
- A4: `sortedWith`(객체 목록 정렬)는 안정하다(`KotlinSemanticsTest`).
- A8: MediaPipe `LIVE_STREAM`의 결과 콜백이 하나씩 온다(위 호출 예가 그렇게 쓴다). 확인하지 않았다. 겹치면 `update`가
  `IllegalStateException`을 던진다.
- 정렬할 값에 NaN이 없다(아래 "알려진 차이").
- A6: ONNX Runtime 데스크톱 JAR(`compileOnly`)와 `onnxruntime-android`의 `ai.onnxruntime` 클래스가 같아서 이 코드가 기기에서
  돈다. 기기에서 돌려 보지 않았다.
- A7: `OrtEnvironment`는 프로세스에 하나이고 라이브러리가 닫지 않는다.
- A14: `java-test-fixtures` 플러그인을 Kotlin/JVM 모듈에 쓸 수 있다(확인).
- A15: binary-compatibility-validator 0.18.2가 KGP 2.1.21에서 돌고 `nonPublicMarkers`가 표시된 선언을 뺀다(확인).
- A16: `@DrowsyInternalApi`(`@RequiresOptIn`)는 Java 호출자를 막지 못한다. 앱은 그 표시가 붙은 선언(`DriveSession`,
  `PreviewPlayer`, `ImageGeometry`)과 `internal` 선언을 쓰지 않는다. Java에서는 바이트코드에서 공개인 internal 함수
  (`openChecked$onnx` 등)에도 닿을 수 있다.
- A17: Android의 `StrictMath`가 fdlibm을 따른다. 확인하지 않았다.
- A18: ONNX Runtime 1.28.0(이 모듈)과 1.31.1(Python 참조)이 v0 파일에서 같은 확률을 낸다. 잰 값: 출력 900개가 비트까지 같다.
- JVM의 Double → Float 변환이 numpy `astype(float32)`와 같다. 잰 값: 창 값 612,000개와 변환 표 426개에서 차이 0.
- A23: KGP가 stdlib를 저절로 넣지 않게 하는 속성 이름이 `kotlin.stdlib.default.dependency`다(확인).
- A24: 2.0 모드로 만든 클래스를 Kotlin 2.0.0 컴파일러가 읽는다. 클래스의 메타데이터 버전은 `mv=[2,0,0]`이다(확인).
  2.0.0 컴파일러로 읽어 보지는 않았다. Kotlin 2.2.21 소비자는 읽었다(확인).
- A29: ONNX Runtime Java에 실행 중 버전 API(`OrtEnvironment.getVersion()`)가 있다. 1.28.0 데스크톱 JAR에서 "1.28.0"을
  돌려줬다.
- A32: Gradle은 선언하지 않은 입력 파일의 변경을 모른다. 그래서 벡터, `export/`, 참조 파일, `.diffdata`의 READY·index·세션
  파일을 시험 입력으로 선언했다.
- A35: ART의 `Double.parseDouble`이 가장 가까운 double을 준다. 시작 검사의 상수 비트 지문(`CONSTANT_BITS_VS_VERIFIED`)이 기기마다
  본다.
- A36: `-Xjdk-release=11`이 Java 11 밖의 JDK API를 컴파일 오류로 만든다(확인).
- 설정 읽기: `pipeline.pose.*` 문자열(`order`, `yaw_pos`, `pitch_pos`, `roll_pos`, `matrix_layout`, `angle_unit`)은 Python이
  검사하지 않으므로 Kotlin도 검사하지 않는다.
- 설정 읽기 메시지: Python `repr`의 출력 가능 글자 판정은 Python 3.11.16(unicodedata 14.0.0)에서 만든 표(`PyPrintable.kt`)를
  쓴다. 참조의 Python 판이 바뀌면 표를 다시 만든다. 판정은 바뀌지 않고 메시지 글자만 바뀐다.
- `StepFeatures.k`(스텝 번호)는 Int 범위 안이다. 넘으면 `ArithmeticException`이다.
- `OnnxDrowsinessModel.predict`는 창이 정확히 `window_steps x 17`이어야 한다(Python `reshape`는 원소 수만 본다. 파이프라인은
  언제나 그 모양을 준다).
- 벡터 묶음 지문(`VerifiedConstants.VECTOR_SET`): `tests/vectors/*.json` 파일마다 `<파일 이름> <sha256>` 줄, 이름 순, `\n`으로
  이은 문자열의 sha256이다(계획이 잇는 규칙을 정하지 않았다). `VerifiedConstants`의 값은 손으로 적었고,
  `VerifiedConstantsTest`가 벡터와 번들 파일에서 다시 계산해 다르면 실패한다. 벡터나 `export/`가 바뀌면 사람이 새 값을 적는다.
- `DrowsinessMonitor`: `RawFrames.of`의 인자 거부에서는 멈추지 않는다. 참조 도구(`drowsy/live.py:1625-1631`)는 결과 변환
  오류에서도 세션을 끝낸다.
- `PreviewPlayer`의 재생 자료에서 실수 칸의 정수 글자는 |값| <= 2^53이다.
- 시험: 벡터 실행기(`VectorRunner`)는 `testFixtures`에 있다. 벡터의 `config_overrides`는 `export/rules.json`의 JSON 나무에
  적용한다.
- 시험(차등 비교, `DiffReplay.kt`):
  - RAW·SIG 모드의 좁은 허용 오차는 A2에 기댄다.
  - STATE 모드는 버려질 프레임도 그대로 넣는다(받아들임 단계에서 버려진다).
  - Python이 예외를 낸 세션의 계수는 예외가 난 자리로 비교한다(파이프라인의 검출기인지 규칙 엔진의 검출기인지).
  - roundHalfUp 차이로 세려면 closure_valid 행에 유한하지 않은 closure가 있어야 한다.
  - NaN 기준선 차이로 세려면 `BaselineEstimator` 마감에서 TimSort가 던지고, 그 앞에 유한하지 않은 표본 값이 있어야 한다.
  - STATE 모드의 `updateFromState`는 `baselineDiagnostics`를 채우지 않는다(판정에는 쓰이지 않는다).

## Python 참조와 알려진 차이

- 반올림 도우미(`blink.py:339` `_round_half_up`)에 NaN이나 무한대가 오면 Python은 예외(ValueError, OverflowError)를 내고
  Kotlin `floor(x + 0.5).toLong()`은 예외 없이 0이나 끝값을 낸다. 이대로 두기로 정했다(2026-11-15). 차등 자료
  S-odd 500세션 가운데 18세션이 이 경우다. 그 세션들은 Python이 예외를 낸 프레임 앞까지 모두 같았다.
- 기준선 표본 목록(원소 32개 이상)에 NaN이 있으면 Kotlin의 정렬은 다른 값을 내거나 `IllegalArgumentException("Comparison
  method violates its general contract!")`을 던지고, Python의 `sorted`는 예외 없이 계속한다. 잰 예: 600개 얼굴 프레임
  가운데 blink_l 하나가 NaN이면 Kotlin은 60000 ms 마감에서 멈추고 Python은 status ok 기준선을 낸다. 검사를 더하지 않았다.
  차등 묶음의 RAW·SIG 세션에는 이런 입력이 없다. 입구 `DrowsinessMonitor`는 파이프라인이 예외를 내면 멈춘다. 그 뒤의
  `update`, `updateNoFace`는 모두 `IllegalStateException`(cause = 그 예외)으로 거부되고, 계속하려면 새 인스턴스를 연다
  (새 보정). 그래서 정렬이 던지면 보정 표본의 NaN 하나가 Kotlin에서는 그 세션을 끝낸다.
- 입구는 유한하지 않은 값을 검사하지 않는다(검사를 더하지 않기로 했다). 그래서 위 두 차이가 남는다. MediaPipe 결과에 NaN이나
  무한대가 있으면 앱이 넣지 않는다.
- yaw_rel이 정확히 ±40도에 걸리는 경우(예: yaw ±40, pitch −7.5, roll 3.25, 단위 기준 회전): JVM `StrictMath.atan2`가 CPython보다
  1 ulp 커서 `closure_valid`가 뒤집힌다(Kotlin false, Python true). 그대로 둔다(A2, K20).
- 내부 JSON 읽기(`MiniJson`)는 Python `json.loads`보다 엄격하다. 중복 키, BOM, 값 뒤의 남은 글자, Long 밖의 정수 글자를
  거부하고, 중첩 깊이를 512로 제한한다. Python의 한계는 호출 깊이에 따라 달라진다(잰 값: 맨 위에서 993, 프레임 200개
  아래에서 793). 지금 파일의 깊이는 feature_spec 4, rules 3, 벡터 최대 6이다. JSON 문법 오류의 메시지는 Python
  `JSONDecodeError`의 글이 아니다.
- 설정의 정수 범위: `Int`로 받는 칸에 Int 범위 밖의 값이 오면 Kotlin은 거부한다(Kotlin만의 메시지 "integer N is outside
  the 32-bit range of this field"). Python은 정수에 한계가 없어 받는다. 차등 자료 C-loader에서 이 차이는 `flags` 표지가 붙은
  문서에만 있다. 이대로 둘지는 아직 정하지 않았다.
- Kotlin은 `pipeline_hash`를 다시 계산하지 않는다(SPEC 15장). Python은 읽을 때 다시 계산해 맞지 않으면 거부한다. 시작
  검사의 H3(`H3_SPEC_VS_VERIFIED`)와 H2, 상수 비트 지문이 그런 파일을 거부한다.
- `RawFrames.of`는 필요한 블렌드셰이프가 없으면 그 자리에서 거부한다(Python은 signals 단계에서 예외를 낸다). x와 y,
  이름과 점수의 길이가 다르면 거부한다.
- 이 입구에는 모델 파일 고정값이 없다. ONNX 가중치 한 바이트가 바뀐 모델도 시작 검사를 통과한다(변조 경우 SC2). 시작 때
  자기 시험 추론도 하지 않는다.
- `update`는 Float 배열을 받는다. MediaPipe도 Float를 주므로 실제 입력에서는 차이가 없다. 벡터의 Double 입력을 Float로
  줄여 넣으면 출력이 마지막 자리에서 달라진다(e2e 벡터에서 잰 가장 큰 차이 4.2e-6, 벡터 허용 오차 안).

## 확인하지 않은 것

- Android 기기나 에뮬레이터에서 한 번도 돌리지 않았다. 실제 MediaPipe 결과를 넣어 보지 않았다. 변환 함수와 호출 예는
  `tasks-vision` 0.10.35에 대고 컴파일만 했다.
- Android에서 `getResourceAsStream`으로 번들 파일을 읽는 것, R8 축소 빌드, `onnxruntime-android`에서의 추론, ART의 수 읽기
  (A35), Android `StrictMath`의 비트(A17)를 확인하지 않았다. 확인한 것은 빌드까지다: Android 라이브러리 소비자의
  실행 경로 의존에서 `drowsy-onnx-0.1.0.jar`의 Java 리소스로 `hawkeye/drowsy/bundle/`의 세 파일이 보이고, 그 sha256이
  `export/`와 같다. 라이브러리 AAR 자체에는 의존의 리소스가 들어가지 않는다(AAR의 성질).
- compileSdk 36(팀 앱의 값)으로는 빌드하지 않았다. 이 PC의 SDK에는 platforms;android-35와 build-tools;35.0.0만 있다.
  JDK 17 말고 다른 JDK(예: Android Studio에 든 JDK)로 Gradle을 돌려 보지 않았다.
- 팀 앱의 세 번째 Maven 저장소는 넣지 않고 확인했다.
- v0 모델은 무작위 가중치다. 학습한 모델이 오면 시험(벡터, `VerifiedConstants`)을 다시 돌린다.
- 차등 비교: S-full(12,000세션)은 돌리지 않았다. 계획 9.2.5절의 걸침 검토 장치는 만들지 않았다(RAW·SIG의 이산 차이가 0이라
  필요한 경우가 없었다). k가 건너뛸 때 모델 창을 비우는 줄(`DriverPipeline`)을 고정하는 시험이 없다.
- 시간, CPU, 메모리는 재지 않았다.
