# 이식 명세 (SPEC): 운전자 상태 파이프라인

상태: **초판 (M1, spec v0)**. 무작위 가중치 모델 `drowsiness_v0.onnx`로 연동을 시작하기 위한 문서다.
2026-11-14 개정: 검토에서 고친 동작 넷(10.2절의 거슬러 보기와 반올림, 12.4절의 `vats` 신뢰도와 인정 구간 뒤의 onset)과
2026-11-13의 소유자 결정 넷(10.2절의 감김 길이, 13장의 미뤄 둔 경고, 7.2절의 스텝마다 모델 호출, 12.7절의 끊김 뒤
다시 시작)을 반영했다. 상수와 두 해시는 그대로이고 벡터는 다시 만들었다(16.1절). 같은 날의 두 번째 검증 뒤에는 참조
구현을 바꾸지 않고 벡터 13개를 더했고(16.1절), 12.4절과 17.2절의 문장 둘을 코드에 맞게 고쳤으며, 20장의 목록을 README의
목록과 맞췄다. 세 번째 검증 뒤에도 참조 구현은 그대로 두고 벡터 8개를 더했다(16.1절. 앞의 81개는 바이트 단위로 같다).
설정을 읽을 때의 검사에 조건 하나(`hold_ms < clock.reset_gap_ms`, 10.1절)가 더해졌고(지금 상수는 통과하므로 해시는
그대로다), 벡터가 고정하지 않는 동작의 목록(16.1절 끝)을 채웠으며, 20장에서는 항목마다 제안된 대안을 적고 3번의 한
문장을 "정비에서 정한 것"으로 옮기고 9번과 12번의 문장을 고쳤다. 같은 날 소유자가 20장의 첫 질문(눈꺼풀이 이탈 임계와
진입 임계 사이에 머무는 동안 감김이 끝나지 않는 것)에 "지금대로 둠"이라고 답했다. 코드, 벡터, 두 해시는 그대로이고, 그
결정과 알려진 한계는 20장 머리의 결정 기록과 10.5절 7번에 적었다. 20장의 1번은 비워 두고 나머지 번호는 그대로 둔다.
네 번째 검증 뒤에도 참조 구현은 그대로 두고 벡터 13개를 더했다(16.1절. 앞의 89개는 바이트 단위로 같다). 참조 구현의 한
줄을 일부러 바꿔 보는 점검을 이식 대상 모듈 여섯 전부로 넓혔고, 신호·기준선·스텝 특징 쪽에서 벡터가 고정하지 않는
동작을 16.1절 끝의 목록에 더했다. 설정을 읽을 때의 검사에 조건 둘(`yawn.cancel_gap_ms`와 `nod.cancel_gap_ms`가
`clock.reset_gap_ms`보다 작아야 한다. 11.5절)이 더해졌다(지금 상수는 통과하므로 해시는 그대로다). 11.5절의 의사 코드에는
하품·끄덕임 계수기의 `reset()`이 하는 일을 적었다(11.6절이 부르는데 내용이 빠져 있었다. 동작은 그대로다).

Android/Kotlin 팀이 Python 참조 구현을 옮길 때 보는 명세다. 계획서(`docs/PLAN.md`)가 아니라 **이식 대상 모듈의
코드**(`drowsy/types.py`, `signals.py`, `calibrate.py`, `blink.py`, `step_features.py`, `rules.py`, `pipeline.py`)와
`config/feature_spec.json`, `config/rules.yaml`에서 뽑았다.

- 문서와 코드가 다르면 코드와 `tests/vectors/*.json`이 맞다. 다른 곳을 찾으면 알려 달라.
- 설계 근거는 PLAN.md에 있고 여기서는 되풀이하지 않는다. "PLAN 4.3절"처럼 가리키기만 한다.
- 이 문서에는 성능 수치가 없다. 숫자는 상수, 정의, 또는 합성 입력에 대한 참조 구현의 출력뿐이다.
- 계획서보다 코드가 더 구체적으로 정한 동작에는 "(코드의 가정)"이라고 적었다. 소유자의 결정을 기다리는 동작은
  해당 절에 코드가 하는 대로 적고 20장에 모았다.

---

## 0. 기준 파일, 해시, 표기

### 0.1 앱이 받는 파일

| 파일 | 내용 | 해시 |
|------|------|------|
| `export/drowsiness_v{N}.onnx` | 졸음 모델. 메타데이터에 아래 해시들이 들어 있다(14장) | 벡터의 `model.sha256` |
| `export/feature_spec_v{N}.json` | 특징 목록, 표준화 상수, `pipeline` 블록(이식 코드의 상수 전부), `pipeline_hash` | 파일 바이트의 sha256 = `feature_spec_hash` |
| `export/rules.json` | 규칙 엔진과 경고 결정의 임계값. `python -m drowsy.spec export-rules --out export/rules.json`으로 만든다 | 파일 바이트의 sha256 = `rules_hash` |
| `tests/vectors/*.json` | 이식 코드를 맞춰 볼 입력과 기대 출력(16장) | 각 파일에 `pipeline_hash`, `rules_hash` |
| `face_landmarker.task` | MediaPipe 모델. 저장소에 없다. 공개 표준 파일(float16 v1)을 쓴다 | `mediapipe.model_sha256` |

상수는 코드에 박지 말고 두 JSON에서 읽는다. spec v0의 잠정 상수(19장)는 실제 데이터를 본 뒤 한 번 바뀌고, 그때
`pipeline_hash`와 벡터가 함께 바뀐다. `export/` 폴더의 파일별 해시는 `export/README.md`에도 적혀 있다.

이 문서를 쓴 시점의 값:

<!-- check:hashes -->

| 항목 | 값 |
|------|----|
| `spec_version` | 0 |
| `pipeline_hash` | `650495e5c8a158db9a071457713f852af846e54e32d80352376592768f9c31a6` |
| `rules_hash` | `9c2d05f29c4c4339d4759425f996c38137677d7216ad73c27091e75fb6cec7b4` |
| `mediapipe_model_sha256` | `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` |

### 0.2 참조 구현과 이 문서의 대응

| Python 모듈 | 내용 | 절 |
|-------------|------|----|
| `drowsy/types.py` | 자료형과 문자열 상수 | 3 |
| `drowsy/signals.py` | EAR, 머리 자세, 홍채 비율, 눈동자 합성값 | 4, 5 |
| `drowsy/pipeline.py` | 프레임 하나의 처리 순서, 모델 호출 | 6, 7, 14 |
| `drowsy/calibrate.py` | 기준선, closure, 프레임 상태 | 8, 9 |
| `drowsy/blink.py` | 감김 구간 검출기 | 10 |
| `drowsy/step_features.py` | 시간 누적, 스텝 특징 17개, 하품, 끄덕임 | 11 |
| `drowsy/rules.py` | 규칙 엔진, 경고 결정 | 12, 13 |

### 0.3 표기

- 상수는 JSON 경로로 적는다. `closure.rho`는 `feature_spec`의 `pipeline.closure.rho`, `forward_zone.debounce_ms`는
  `rules.json`의 같은 경로다. `clock` 블록은 두 파일에 다 있고 값이 같아야 한다(다르면 파이프라인 생성이 실패한다).
  `closure` 블록도 두 파일에 다 있지만 내용이 다르다(pipeline 쪽은 closure 식의 상수, rules 쪽은 규칙용 검출기와 임계 시간).
- `` `경로`(값) ``처럼 괄호 안에 적은 값은 지금 설정의 값이다. 저장소의 테스트(`tests/test_spec_doc.py`)가 이 표기와
  상수 표의 값을 설정 파일과 비교한다.
- `t`는 프레임의 `t_ms`, `clip01(x) = min(max(x, 0), 1)`, `round_half_up(x) = floor(x + 0.5)`.
- 의사 코드의 필드 이름은 Python 구현과 벡터 JSON의 이름(snake_case) 그대로다.

---

## 1. 앱이 보장해야 하는 것 (파이프라인을 부르기 전)

참조 구현의 학습 특징은 아래 전제에서 뽑은 것이고 벡터도 이 전제를 따른다. Android 실기기에서는 아직 확인하지 못했다.

| # | 보장할 것 | 어기면 |
|---|-----------|--------|
| G1 | 모델 파일은 공개 표준 `face_landmarker.task`(float16 v1). 파일의 sha256이 `mediapipe.model_sha256`과 같다 | 랜드마크와 블렌드셰이프의 분포가 달라진다 |
| G2 | 실행 모드는 `LIVE_STREAM`(`detectAsync`). 학습 추출은 같은 그래프를 쓰는 `VIDEO` 모드다. `IMAGE` 모드는 쓰지 않는다 | 추적과 평활이 달라 신호가 달라진다 |
| G3 | `numFaces = 1` (`mediapipe.num_faces`) | 1이 아니면 MediaPipe의 시간 평활이 꺼진다 |
| G4 | `setOutputFaceBlendshapes(true)`, `setOutputFacialTransformationMatrixes(true)` | 블렌드셰이프와 행렬이 없으면 `RawFrame`을 만들 수 없다 |
| G5 | 검출·추적 신뢰도 임계값은 기본값 그대로 둔다(참조 추출 `drowsy/extract.py`가 바꾸지 않는다) | 얼굴 유무 판정이 달라진다 |
| G6 | 비트맵을 **똑바로 세운 뒤** `detectAsync`에 넘긴다. `ImageProcessingOptions`의 `rotationDegrees`는 쓰지 않는다 | 랜드마크와 행렬이 센서 방향 좌표로 나오고 블렌드셰이프 값이 달라진다 |
| G7 | 분석 프레임을 **좌우 반전하지 않는다**. 반전은 미리보기에만 건다 | yaw·roll의 부호가 뒤집히고 Left/Right 신호가 서로 바뀐다(5.6절) |
| G8 | `detectAsync`에 넘기는 타임스탬프는 **양수**이고 **엄격히 증가**하는 **실제 ms**다. 카메라 프레임 시각을 권한다. 프레임 번호를 넘기지 않는다 | 0 이하는 MediaPipe의 평활을 흐트러뜨리고, 같거나 작은 값은 MediaPipe가 거부한다. 프레임 번호는 평활과 모든 시간 계산을 틀리게 한다 |
| G9 | `RawFrame.t_ms`는 결과 콜백의 `timestampMs()`(= 넘긴 값), `w`·`h`는 MediaPipe에 넘긴(세운 뒤의) 영상 크기다 | EAR과 모든 시간이 틀린다 |
| G10 | 콜백으로 받은 결과는 얼굴이 없는 것도 빠짐없이 `update`에 넣는다(`face = false`) | 얼굴 없는 시간이 "데이터 없음"으로 바뀌고, 끊긴 시간 판정이 달라진다 |
| G11 | `update`는 타임스탬프 순서로, 한 번에 하나씩 부른다. 세션(운전 한 번)마다 `DriverPipeline`을 새로 만든다 | 기준선은 인스턴스당 한 번만 잡는다. 상태 머신은 동시 호출을 가정하지 않는다 |
| G12 | 차량 속도는 km/h, 모르면 null | null이면 속도 억제를 하지 않는다(13장) |

결과에서 `RawFrame`을 만드는 법:

```
face = result.faceLandmarks()가 비어 있지 않다      // 얼굴 유무는 이것으로만 판정한다
face == false: RawFrame(t_ms, false, w, h, lm = {}, bs = {}, m = [])
face == true:
    lm[i] = (landmark[i].x, landmark[i].y)           // 4.1절의 인덱스 44개. 정규화 좌표 그대로(Double)
    bs[name] = score                                 // category_name으로 찾는다. 4.2절의 11개가 반드시 있어야 한다
    m = facialTransformationMatrixes()[0]            // float[16]을 순서 그대로(열 우선, 5.3절)
```

`face = true`인데 필요한 랜드마크나 블렌드셰이프가 없으면 참조 구현은 예외를 낸다. 행렬의 길이가 16이 아니어도 예외다.

---

## 2. 숫자 규칙

| 규칙 | 내용 |
|------|------|
| 실수 | 모두 `Double`. `Float`는 모델 입력 텐서와 모델 출력에만 나온다 |
| 시간 | 정수 ms, `Long`. `t_ms`는 `Int` 범위를 넘을 수 있다(벡터 `steps_five_minutes_large_t0`) |
| 시간 누적 | ms 합은 `Long`으로 정확히 더한다. `Double`로 바꾸는 것은 나눗셈 직전이다 |
| 수학 함수 | `sqrt`, `atan2`, `hypot`, `sin`, `cos`, `floor`, `ceil`만 쓴다. 다른 함수(`asin`, `acos`, `exp`, `pow`)는 쓰지 않는다 |
| 각도 변환 | 상수를 곱한다: `RAD_TO_DEG = 180.0 / PI`, `DEG_TO_RAD = PI / 180.0`. `Math.toDegrees`를 쓰지 않는다 |
| 반올림 | `round_half_up(x) = floor(x + 0.5)`를 `Long`으로. 감김 검출기의 보간 시각에만 쓴다. 두 표본 사이의 오프셋(ms)만 반올림하고 직전 표본의 시각(`Long`)에 정수로 더한다. 절대 시각을 `Double`로 더한 뒤 반올림하지 않는다(10.2절). 짝수 쪽으로 반올림하는 함수(`kotlin.math.round`)는 쓰지 않는다 |
| 분위수 순위 | `ceil(q * n)`. `q * n`은 `Double` 곱 그대로 올림한다(8.1절) |
| 비교 연산 | 이 문서에 적힌 그대로(`>=`와 `>`, `<`와 `<=`를 바꾸지 않는다) |
| 0 나눗셈 | 분모가 0이 되는 비율은 0으로 정의한다(식마다 조건을 적었다) |
| 분산 | 제곱근 전에 0 이상으로 자른다 |
| 없는 값 | NaN과 무한대를 쓰지 않는다. 값이 없으면 0.0을 넣고 옆의 `Boolean`으로 구분한다. 선택 항목(홍채 비율)만 null이다 |
| 식의 계산 순서 | 적힌 순서대로 왼쪽부터. 예: `pitch * pitch * ms`는 `(pitch * pitch) * ms` |
| 합산 순서 | 스텝 통 안에서는 프레임이 온 순서로, 구간 합은 오래된 스텝부터 더한다(11.3절). 더한 값을 들고 다니며 빼지 않는다 |
| 유한성 검사 | `x - x == 0.0`(= `Double.isFinite()`) |
| `-0.0` | 0.0과 같은 값이다. 벡터 파일에 `-0.0`이 나온다 |

벡터의 허용 오차(16장)는 수학 함수의 마지막 자리 차이를 덮을 만큼 넉넉하다. 실수로 계산되는 값이 판정을 가르는
자리(임계값에 붙은 closure·각도·눈동자 합성값, x.5 ms에 걸친 보간 시각)는 벡터에 없다. 등호 자리를 벡터가 고정하는
것은 정수 ms의 비교, 입력으로 그대로 들어오는 차량 속도, 정수 ms의 몫으로 정확히 나오는 `perclos_60`이다(16.1절).

---

## 3. 자료형

모두 불변 값이다. 시각과 길이는 정수 ms이고 **원래 `t_ms` 시계**로 적는다. 세션 시간(6장)으로 적는 것은
`Baseline.t_complete_ms`와 `StepFeatures.k` 둘뿐이다. `_l`은 MediaPipe Left, `_r`은 Right다(5.5절).

### 3.1 `RawFrame` (앱이 만든다)

| 필드 | 형 | 내용 |
|------|----|------|
| `t_ms` | Long | 프레임 타임스탬프 |
| `face` | Boolean | 얼굴이 검출됐는지(결과 리스트가 비었는지로만 판정) |
| `w`, `h` | Int | MediaPipe에 넘긴 영상의 너비와 높이(픽셀) |
| `lm` | `Map<Int, (Double, Double)>` | 랜드마크 인덱스 → 정규화 좌표 `(x, y)`. x는 너비, y는 높이로 정규화된 값. 4.1절의 44개. 더 있어도 된다 |
| `bs` | `Map<String, Double>` | 블렌드셰이프 `category_name` → 값. 4.2절의 11개는 반드시 있어야 한다 |
| `m` | Double 16개 | 얼굴 변환 행렬 4x4의 열 우선 값 |

얼굴이 없으면 `lm`과 `bs`는 비어 있고 `m`은 길이 0이다.

### 3.2 `FrameSignals` (`signals.compute`의 출력, 5장)

| 필드 | 형 | 내용 | 얼굴 없음 |
|------|----|------|-----------|
| `t_ms` | Long | `RawFrame.t_ms` | 같음 |
| `face` | Boolean | | false |
| `ear_l`, `ear_r` | Double | 픽셀 좌표의 2D EAR. 쓸 수 없는 눈은 0.0 | 0.0 |
| `eye_w_l`, `eye_w_r` | Double | 눈 폭 `dist(p1, p4)`(픽셀). `signals.min_eye_width_px` 미만이면 "쓸 수 없는 눈". 별도 플래그는 없다 | 0.0 |
| `blink_l`, `blink_r` | Double | `eyeBlinkLeft`, `eyeBlinkRight` 값 그대로 | 0.0 |
| `eye_look` | Double 8개 | `eyeLook*` 값. 순서는 `EYE_LOOK_ORDER`(3.9절) | 0.0 여덟 개 |
| `jaw_open` | Double | `jawOpen` 값 그대로 | 0.0 |
| `yaw`, `pitch`, `roll` | Double | 행렬에서 구한 절대 오일러 각(도) | 0.0 |
| `iris_h_l`, `iris_h_r`, `iris_v_l`, `iris_v_r` | Double? | 홍채 비율. 쓸 수 없는 눈은 null | null |

홍채 비율은 `signals` 벡터가 비교하지만, 그 뒤의 계산(기준선, closure, 특징, 규칙)은 쓰지 않는다.

### 3.3 `Baseline` (8장)

| 필드 | 형 | 내용 |
|------|----|------|
| `e_l`, `e_r` | Double | 각 눈 EAR의 90 분위수(뜬 눈 기준) |
| `b_l`, `b_r` | Double | 각 눈 `eyeBlink`의 10 분위수(뜬 눈 기준) |
| `r0` | Double 9개 | 기준 회전 3x3, 행 우선. `R0[i][j] = r0[3*i + j]` |
| `jaw0`, `gaze_h0`, `gaze_v0` | Double | `jaw_open`, `gaze_h`, `gaze_v`의 중앙값 |
| `t_complete_ms` | Long | 기준선이 완성된 **세션 시간** `T_b`. 60000, 90000, …, 180000. 전역 상수로 바로 만든 것은 0 |
| `status` | String | `ok`, `off_axis`, `fallback_global` 중 하나 |

### 3.4 `FrameState` (`calibrate.frame_state`의 출력, 9장)

| 필드 | 형 | 내용 | 얼굴 없음 |
|------|----|------|-----------|
| `t_ms` | Long | `RawFrame.t_ms` | 같음 |
| `face` | Boolean | | false |
| `closure` | Double | 0~1. `closure_valid = false`이면 0.0이고 뜻이 없다 | 0.0 |
| `closure_valid` | Boolean | 얼굴 있음, 쓸 수 있는 눈이 하나 이상, `abs(yaw_rel) <= closure.valid_max_abs_yaw_deg` | false |
| `provisional` | Boolean | 기준선 완성 전에 전역 상수로 계산한 프레임이면 true. 완성된 뒤(전역 상수로 대체된 경우 포함)에는 false | 호출자가 준 값 |
| `yaw_rel`, `pitch_rel` | Double | `Euler(R0ᵀ·R)`의 yaw, pitch(도). 얼굴이 있으면 항상 채운다 | 0.0 |
| `pose_valid` | Boolean | 얼굴이 있고 기준선 `status`가 `fallback_global`이 아니다. 시선 규칙이 자세를 써도 되는지 | false |
| `gaze_h_rel`, `gaze_v_rel` | Double | `gaze_h − gaze_h0`, `gaze_v − gaze_v0` | 0.0 |
| `jaw_rel` | Double | `jaw_open − jaw0` | 0.0 |

기준선 완성 전에는 전역 상수 기준선(`status = fallback_global`)으로 계산하므로 `pose_valid`는 저절로 false다.
스텝 특징(`pitch_std_60`, 끄덕임)은 `pose_valid`가 아니라 `face`만 보고 `pitch_rel`을 쓴다.

### 3.5 `ClosureRun` (감김 구간, 10장)

| 필드 | 형 | 내용 |
|------|----|------|
| `start_ms` | Long | closure가 진입 임계를 그 구간에서 처음 위로 지난 시각(보간했으면 반올림한 값) |
| `end_ms` | Long | 나가기 전 마지막으로 진입 임계를 아래로 지난 시각. 잘린 구간은 진입 임계 이상이었던 마지막 유효 프레임의 시각 |
| `duration_ms` | Long | 구간 안에서 closure가 진입 임계 **이상이던 시간의 합**(10.4절의 토막). 구간 도중에 진입 임계 아래로 내려갔다가 다시 올라온 사이는 넣지 않는다. `end_ms − start_ms` 이하이고, 다시 올라온 적이 없는 구간에서는 같다 |
| `amplitude` | Double | 구간 중 최대 closure |
| `peak_closing_velocity`, `peak_opening_velocity` | Double | closure 증가율과 감소율의 최댓값(1/s, 0 이상). `velocity_valid = false`이면 0.0 |
| `velocity_valid` | Boolean | 잘리지 않았고 두 속도가 모두 0보다 크게 측정됐다 |
| `truncated` | Boolean | 유효 프레임이 끊겨서 닫힌 구간 |

### 3.6 `StepFeatures` (11장)

| 필드 | 형 | 내용 |
|------|----|------|
| `k` | Int | 세션 스텝 번호. 스텝 k는 세션 시간 `[1000(k−1), 1000k)`를 대표한다. `T_b = 60000`이면 처음 나오는 스텝은 61 |
| `t_end_ms` | Long | 스텝 끝 시각(원래 시계) = 세션 원점 + `1000 * k` |
| `full_lookback` | Boolean | 60초 조회 구간이 다 찼는지(11.2절의 `emit`). true인 스텝만 모델 창에 들어간다 |
| `values` | Double 17개 | 11.4절 순서의 원시 특징. 표준화도 log1p도 하지 않은 값. 모두 유한 |

### 3.7 `DriverStateEvent` (12장)

| 필드 | 형 | 내용 |
|------|----|------|
| `type` | String | `EVENT_TYPES` 중 하나 |
| `severity` | Int | 0~3. `events.<type>.severity` |
| `t_start_ms` | Long | 상태가 시작된 시각(원래 시계). 종류별 정의는 12.8절 |
| `duration_ms` | Long | 발행 시각 − `t_start_ms`. 발행 시각은 프레임 이벤트는 그 프레임의 `t_ms`, `drowsy`는 스텝의 `t_end_ms`다(10초 초과 끊김이 끝내는 `drowsy` end만 그 프레임의 `t_ms`, 12.7절) |
| `confidence` | Double | 12.9절 |
| `source` | String | `SOURCES` 중 하나 |
| `phase` | String | `onset`(임계 시간 도달) 또는 `end`(상태 종료) |
| `provisional` | Boolean | 발행한 프레임의 `FrameState.provisional`. `drowsy`는 항상 false |

### 3.8 `PipelineOutput` (`DriverPipeline.update`의 출력, 7장)

| 필드 | 형 | 내용 |
|------|----|------|
| `state` | FrameState | 이 프레임의 상태 |
| `runs` | ClosureRun 목록 | 이 프레임에서 끝난 **특징용** 감김 구간. 0개 또는 1개 |
| `steps` | StepFeatures 목록 | 이 프레임이 발행시킨 스텝. 오래된 것부터 |
| `step_probs` | (Double 3개 또는 null)의 목록 | `steps`와 길이와 순서가 같다. 그 스텝에서 모델을 불렀으면 그 스텝까지의 창으로 얻은 `(alert, low_vigilant, drowsy)`, 부르지 않았으면(모델 없음, 창이 덜 참, 입력이 유한하지 않음) null |
| `probs` | Double 3개 또는 null | 이 프레임이 스텝을 발행시켰으면 `step_probs`의 마지막 원소, 스텝이 없으면 null |
| `events` | DriverStateEvent 목록 | 이 프레임에서 발행된 상태 이벤트(발행 순서, 7.3절) |
| `alerts` | DriverStateEvent 목록 | 실제로 경고할 이벤트(13장). 이 프레임의 onset이거나, 앞선 프레임의 onset을 미뤘다가 내는 사본, 되풀이 경고의 사본이다(사본은 `duration_ms`만 다르다) |

스텝을 하나만 발행시킨 프레임에서는 `probs`가 "이 프레임에서 모델을 불렀을 때만 확률"이다. 프레임이 끊겼다가 와서
스텝을 여럿 발행시킨 프레임에서는 스텝마다 모델을 부르므로 `step_probs`에 확률이 여러 개 있을 수 있다.

### 3.9 문자열 상수

<!-- check:enums -->

| 이름 | 값 (순서 고정) |
|------|----------------|
| `EVENT_TYPES` | `calibrating`, `unable_to_analyze`, `gaze_away_long`, `vats`, `drowsy`, `microsleep`, `sleep` |
| `PHASES` | `onset`, `end` |
| `SOURCES` | `face`, `head_pose`, `eye`, `model`, `perclos`, `closure` |
| `BASELINE_STATUSES` | `ok`, `off_axis`, `fallback_global` |
| `CLASS_ORDER` | `alert`, `low_vigilant`, `drowsy` |
| `DROWSY_SOURCES` | `model`, `perclos`, `either` |
| `EYE_LOOK_ORDER` | `in_l`, `out_l`, `up_l`, `down_l`, `in_r`, `out_r`, `up_r`, `down_r` |

`probs[2]`가 `p_drowsy`다. `DROWSY_SOURCES`는 `drowsiness.source`의 값이다.

---

## 4. 인덱스 표

### 4.1 랜드마크 (`pipeline.landmarks`)

<!-- check:landmarks -->

| 키 | 인덱스 (순서 고정) | 뜻 |
|----|--------------------|----|
| `ear_r` | 33, 160, 158, 133, 153, 144 | R(MediaPipe RIGHT_EYE)의 EAR 점 p1..p6 |
| `ear_l` | 362, 385, 387, 263, 373, 380 | L(MediaPipe LEFT_EYE)의 EAR 점 p1..p6 |
| `iris_r` | 468, 469, 470, 471, 472 | R 홍채: 중심, +x, 위, −x, 아래 |
| `iris_l` | 473, 474, 475, 476, 477 | L 홍채: 같은 순서 |
| `corner_outer_r` | 33 | R 바깥 눈 꼬리(관자놀이 쪽) |
| `corner_inner_r` | 133 | R 안쪽 눈 꼬리(코 쪽) |
| `corner_outer_l` | 263 | L 바깥 눈 꼬리 |
| `corner_inner_l` | 362 | L 안쪽 눈 꼬리 |
| `contour_r` | 33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7 | R 눈 윤곽 16점: 바깥 꼬리에서 윗눈꺼풀을 따라 안쪽 꼬리까지, 아랫눈꺼풀로 돌아옴 |
| `contour_l` | 263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249 | L 눈 윤곽 16점: 같은 순서 |
| `face_width` | 234, 454 | 얼굴 폭 측정용 2점 |

- EAR의 p1과 p4는 눈 꼬리다. R은 p1 = 바깥(33), p4 = 안쪽(133)이고 L은 p1 = 안쪽(362), p4 = 바깥(263)이다.
  (p2, p6)과 (p3, p5)가 위아래 짝이다.
- `RawFrame.lm`의 계약은 윤곽 32 + 홍채 10 + 얼굴 폭 2 = 44개다. 이식 코드가 실제로 읽는 것은 EAR 12점과 홍채 중심
  2점(468, 473)뿐이다. 나머지는 추출 파일과 품질 통계가 쓰고 이식 코드의 출력에는 영향이 없다.

### 4.2 블렌드셰이프 (`pipeline.blendshapes`)

참조 구현은 이름(`category_name`)으로 값을 찾는다. 쓰는 것은 11개다.

<!-- check:blendshape-names -->

| 키 | 이름 |
|----|------|
| `eye_blink_l` | `eyeBlinkLeft` |
| `eye_blink_r` | `eyeBlinkRight` |
| `eye_look` | `eyeLookInLeft`, `eyeLookOutLeft`, `eyeLookUpLeft`, `eyeLookDownLeft`, `eyeLookInRight`, `eyeLookOutRight`, `eyeLookUpRight`, `eyeLookDownRight` |
| `jaw_open` | `jawOpen` |

`eye_look`의 순서가 `FrameSignals.eye_look`의 순서다. `eyeBlinkLeft`는 L 집합(362…)과, `eyeBlinkRight`는 R 집합(33…)과
짝이다. `eyeLookIn*`은 코 쪽, `eyeLookOut*`은 관자놀이 쪽이다.

52개 전체(`blendshapes.order`, 위치가 MediaPipe의 `Category.index`):

<!-- check:blendshapes -->

| # | 이름 | # | 이름 | # | 이름 | # | 이름 |
|---|------|---|------|---|------|---|------|
| 0 | `_neutral` | 1 | `browDownLeft` | 2 | `browDownRight` | 3 | `browInnerUp` |
| 4 | `browOuterUpLeft` | 5 | `browOuterUpRight` | 6 | `cheekPuff` | 7 | `cheekSquintLeft` |
| 8 | `cheekSquintRight` | 9 | `eyeBlinkLeft` | 10 | `eyeBlinkRight` | 11 | `eyeLookDownLeft` |
| 12 | `eyeLookDownRight` | 13 | `eyeLookInLeft` | 14 | `eyeLookInRight` | 15 | `eyeLookOutLeft` |
| 16 | `eyeLookOutRight` | 17 | `eyeLookUpLeft` | 18 | `eyeLookUpRight` | 19 | `eyeSquintLeft` |
| 20 | `eyeSquintRight` | 21 | `eyeWideLeft` | 22 | `eyeWideRight` | 23 | `jawForward` |
| 24 | `jawLeft` | 25 | `jawOpen` | 26 | `jawRight` | 27 | `mouthClose` |
| 28 | `mouthDimpleLeft` | 29 | `mouthDimpleRight` | 30 | `mouthFrownLeft` | 31 | `mouthFrownRight` |
| 32 | `mouthFunnel` | 33 | `mouthLeft` | 34 | `mouthLowerDownLeft` | 35 | `mouthLowerDownRight` |
| 36 | `mouthPressLeft` | 37 | `mouthPressRight` | 38 | `mouthPucker` | 39 | `mouthRight` |
| 40 | `mouthRollLower` | 41 | `mouthRollUpper` | 42 | `mouthShrugLower` | 43 | `mouthShrugUpper` |
| 44 | `mouthSmileLeft` | 45 | `mouthSmileRight` | 46 | `mouthStretchLeft` | 47 | `mouthStretchRight` |
| 48 | `mouthUpperUpLeft` | 49 | `mouthUpperUpRight` | 50 | `noseSneerLeft` | 51 | `noseSneerRight` |

---

## 5. 좌표, 각도, 좌우 규약과 원시 신호

<!-- check:pipeline -->

| 경로 | 값 | 뜻 |
|------|----|----|
| `signals.min_eye_width_px` | 6.0 | 눈 폭이 이 값 미만이면 쓸 수 없는 눈 |
| `pose.order` | `Ry*Rx*Rz` | `R = Ry(yaw)·Rx(pitch)·Rz(roll)` |
| `pose.yaw_pos` | `subject_left` | yaw 양수 = 피험자가 자기 왼쪽으로 고개를 돌림 |
| `pose.pitch_pos` | `chin_down` | pitch 양수 = 턱이 내려감 |
| `pose.roll_pos` | `ccw_in_image` | roll 양수 = 영상에서 반시계 방향 |
| `pose.matrix_layout` | `column_major` | `M[i][j] = m[4*j + i]` |
| `pose.angle_unit` | `deg` | 각도는 모두 도 |
| `mediapipe.model_sha256` | `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` | G1 |
| `mediapipe.num_faces` | 1 | G3 |

`pose` 블록은 규약을 해시에 묶어 두는 기록이고 계산식은 아래로 고정이다.

### 5.1 픽셀 좌표와 거리

랜드마크는 x가 너비, y가 높이로 따로 정규화돼 있다. 거리와 비율은 모두 픽셀 좌표에서 구한다. 정규화 좌표로 EAR을
구하면 비정사각 프레임에서 값이 틀린다(벡터 `signals_ear_non_square`).

```
px(i)      = (lm[i].x * w, lm[i].y * h)
dist(a, b) = hypot(a.x - b.x, a.y - b.y)
```

### 5.2 EAR과 홍채 비율

```
눈 하나 (p1..p6 = 4.1절의 ear_r 또는 ear_l 순서, 픽셀 좌표):
width = dist(p1, p4)
width < signals.min_eye_width_px 이면: 쓸 수 없는 눈. ear = 0.0, iris_h = iris_v = null, eye_w = width (측정값 그대로)
아니면:
    ear = (dist(p2, p6) + dist(p3, p5)) / (2.0 * width)
    outer = px(corner_outer), inner = px(corner_inner), iris = px(홍채 중심)        // R: 33, 133, 468   L: 263, 362, 473
    a = inner - outer;  d = iris - outer;  n = a.x*a.x + a.y*a.y
    n <= 0.0 이면 (iris_h, iris_v) = (0.0, 0.0)
    아니면 iris_h = (d.x*a.x + d.y*a.y) / n;   iris_v = (a.x*d.y - a.y*d.x) / n
```

- `eye_aspect_ratio` 함수 자체는 `width <= 0.0`이면 0.0을 돌려준다. `compute`는 폭이 하한 미만인 눈에서 이 함수를
  부르지 않는다.
- 폭이 정확히 6.0이면 쓸 수 있는 눈이다(`<`).
- `iris_h`는 바깥 꼬리에서 0, 안쪽 꼬리에서 1이다. 두 눈은 `a`의 방향이 반대라서 같은 시선 이동에 부호가 반대로 움직인다.
  참조 구현은 눈별 값만 내고 합치지 않는다.

### 5.3 변환 행렬과 오일러 각

MediaPipe의 `float[16]`은 열 우선이다. Python의 `M[i][j]`(행 i, 열 j)와 같은 값은 `m[4*j + i]`다. 행 우선으로 읽으면
전치 행렬이 되어 각도 부호가 대략 뒤집히는데 오류는 나지 않는다(벡터 `signals_official_matrix`로 확인한다).

```
yaw   = atan2( m[8],  m[10])            * RAD_TO_DEG      // atan2(M[0][2], M[2][2])
pitch = atan2(-m[9],  hypot(m[1], m[5])) * RAD_TO_DEG     // atan2(-M[1][2], hypot(M[1][0], M[1][1]))
roll  = atan2( m[1],  m[5])             * RAD_TO_DEG      // atan2(M[1][0], M[1][1])
```

3x3 회전은 행 우선 9개 값 `r`로 주고받는다(`R[i][j] = r[3*i + j]`).

```
행렬에서 회전 블록:  r = (m[0], m[4], m[8],  m[1], m[5], m[9],  m[2], m[6], m[10])
r에서 각도:          yaw = atan2(r[2], r[8]);  pitch = atan2(-r[5], hypot(r[3], r[4]));  roll = atan2(r[3], r[4])   (각각 * RAD_TO_DEG)

각도(도)에서 회전 R = Ry(yaw)·Rx(pitch)·Rz(roll):
cy = cos(yaw * DEG_TO_RAD);   sy = sin(yaw * DEG_TO_RAD)
cp = cos(pitch * DEG_TO_RAD); sp = sin(pitch * DEG_TO_RAD)
cr = cos(roll * DEG_TO_RAD);  sr = sin(roll * DEG_TO_RAD)
r = ( cy*cr + sy*sp*sr,   sy*sp*cr - cy*sr,   sy*cp,
      cp*sr,              cp*cr,              -sp,
      cy*sp*sr - sy*cr,   sy*sr + cy*sp*cr,   cy*cp )

기준 회전 대비 회전 Rrel = R0ᵀ·R:
rrel[3*i + j] = r0[i]*r[j] + r0[3 + i]*r[3 + j] + r0[6 + i]*r[6 + j]        (i, j = 0..2, 이 순서로 더한다)
```

기준 자세 대비 각도는 오일러 각의 단순 차가 아니라 `Euler(R0ᵀ·R)`이다.

### 5.4 눈동자 합성값

`el` = `FrameSignals.eye_look`(순서 `in_l, out_l, up_l, down_l, in_r, out_r, up_r, down_r`).

```
gaze_h = ((el[in_r] + el[out_l]) - (el[out_r] + el[in_l])) / 2.0      // 양수 = 피험자가 자기 왼쪽을 봄 (yaw와 같은 부호)
gaze_v = (el[up_l] + el[up_r]) / 2.0 - (el[down_l] + el[down_r]) / 2.0   // 양수 = 위쪽
```

정면에서도 0이 아니므로 기준선 대비 변화(`gaze_h_rel`, `gaze_v_rel`)만 쓴다.

### 5.5 좌우와 부호

- `_l`은 MediaPipe Left(L 집합 362…, `*Left` 블렌드셰이프), `_r`은 Right(R 집합 33…, `*Right`)다. 반전하지 않은
  영상에서 피험자 자신의 좌우다.
- 부호(비반전 영상): yaw 양수 = 피험자가 자기 왼쪽으로 돌림, pitch 양수 = 턱이 내려감, roll 양수 = 영상에서 반시계 방향.
- 행렬의 평행이동 성분(`m[12..14]`)은 쓰지 않는다.

### 5.6 좌우 반전 입력이 들어오면

yaw와 roll의 부호가 뒤집히고 pitch는 그대로이며, L과 R의 신호가 서로 바뀐다(벡터 `signals_mirrored_matrix`,
`signals_one_eye_closed`). closure(9장)와 시선 영역 판정(절댓값 비교)은 반전해도 값이 같게 설계돼 있지만, 눈별 신호와
`gaze_h`의 부호, `signals` 벡터의 기대값은 달라진다. 명세는 비반전(G7)이다.

### 5.7 `signals.compute`

```
compute(frame):
  frame.face == false: FrameSignals(t_ms, face = false, 실수 필드 모두 0.0, eye_look = 0.0 여덟 개, 홍채 비율 null)
  아니면:
    눈 l, r 각각 5.2절                              // 한쪽이 쓸 수 없는 눈이어도 다른 쪽은 계산한다
    blink_l = bs["eyeBlinkLeft"];  blink_r = bs["eyeBlinkRight"]
    eye_look = 4.2절 eye_look 이름 순서의 값 8개
    jaw_open = bs["jawOpen"]
    (yaw, pitch, roll) = 5.3절 (frame.m에서)
```

블렌드셰이프와 각도는 눈 폭과 무관하게 채운다. 이 함수는 상태가 없다.

---

## 6. 타임스탬프, 세션 시간, 유지 규칙

<!-- check:pipeline -->

| 경로 | 값 | 단위 | 뜻 |
|------|----|------|----|
| `clock.step_ms` | 1000 | ms | 스텝 길이 |
| `clock.max_hold_ms` | 200 | ms | 한 프레임의 상태를 다음 프레임까지 인정하는 최대 시간 |
| `clock.reset_gap_ms` | 10000 | ms | 프레임 간격이 이 값을 **넘으면** 검출기와 누적기를 초기화 |

- **버리는 프레임**: `t_ms`가 직전에 받아들인 프레임의 `t_ms` 이하이면 버리고 센다(`dropped_frames`). `update`는 null을
  돌려주고 다른 상태는 하나도 바뀌지 않는다(벡터 `runs_duplicate_timestamp`).
- **세션 원점** = 처음 받아들인 프레임의 `t_ms`. **세션 시간** = `t_ms` − 원점. 첫 프레임의 세션 시간은 0이다.
- **유지 규칙**: 한 프레임의 상태(얼굴 유무, closure, 자세)는 다음 프레임이 올 때까지 유지된 것으로 본다. 단
  `held = min(다음 t − 이번 t, clock.max_hold_ms)`까지만이다. 넘는 부분은 "데이터 없음"이고 얼굴 없는 시간과 똑같이
  취급한다(어느 누적값에도 들어가지 않는다). 그래서 **시간 누적은 다음 프레임이 도착했을 때** 한다. 세션의 마지막
  프레임은 어디에도 더해지지 않는다.
- **비율은 모두 시간으로** 계산한다. 프레임 수로 세는 것은 기준선의 "얼굴 프레임 300개"와 분위수 표본뿐이다.
- **스텝**: 스텝 k는 세션 시간 `[1000(k−1), 1000k)`를 대표한다. 세션 시간이 `1000k` 이상인 첫 프레임이 스텝 k를
  발행시키고, 그 프레임은 스텝 k에 들어가지 않는다(정확히 `1000k`에 온 프레임은 스텝 k+1에 속한다).
- **10초 초과 끊김**: 받아들인 프레임 사이의 간격이 `clock.reset_gap_ms`(10000)를 넘으면(`>`. 정확히 10000 ms는
  넘지 않은 것이다) 특징용 감김 검출기와 스텝 누적기(11.6절)를 초기화하고, 규칙 엔진은 진행 중이던 상태를 끝낸 뒤
  세션을 시작할 때처럼 다시 시작한다(12.7절. `calibrating`만 남는다). 기준선과 기준선 추정기는 그대로다. 새
  기준선은 새 `DriverPipeline`에서만 잡는다.
- **10초 이하의 끊김**은 초기화하지 않는다. 끊긴 시간 가운데 `clock.max_hold_ms`를 넘는 부분은 "데이터 없음"이고,
  그동안 끝난 스텝을 다음 프레임에서 한꺼번에 발행한다(11.6절). 끊김의 길이에 따라 일어나는 일은 17.2절에 모았다.
- 구성 요소가 받는 시계: 감김 검출기와 규칙 엔진은 원래 `t_ms`, 기준선 추정기와 스텝 누적기(하품·끄덕임 포함)는 세션
  시간으로 계산한다. 모든 식이 시각의 차만 쓰므로 결과는 원점과 무관하다(벡터 `steps_five_minutes_large_t0`).

---

## 7. 프레임 하나의 처리 순서 (`DriverPipeline`)

### 7.1 생성

```
DriverPipeline(spec, rules, baseline = null, feature_start_ms = null, model = null)
  rules.clock != spec.clock                         -> 오류
  baseline == null 인데 feature_start_ms != null    -> 오류
  baseline != null:
      feature_start = feature_start_ms ?: baseline.t_complete_ms
      feature_start < 0 또는 clock.step_ms의 배수가 아님 -> 오류
  baseline == null:
      estimator = BaselineEstimator(spec)            // feature_start는 기준선이 확정될 때 정해진다

  global_baseline = 전역 상수 기준선 (8.4절)
  detector   = ClosureRunDetector(pipeline.run_detector)      // 특징용
  aggregator = 없음 (처음 필요할 때 만든다)
  engine = RuleEngine(rules);  policy = AlertPolicy(rules)
  window = 빈 목록;  window_has_k = false;  window_last_k = 0
```

- 앱은 `baseline`을 주지 않는다(세션 앞부분에서 보정한다). 기준선을 넣는 경로는 학습과 벡터용이다. 이때는 첫 프레임부터
  그 기준선을 쓰고(`provisional = false`, `calibrating` 이벤트 없음) 특징 누적은 `feature_start`(세션 시간)부터다.
- `model`은 120 x 17 원시 스텝 특징(가장 오래된 스텝이 먼저)을 받아 확률 3개를 돌려주는 함수다(14장). null이면
  `step_probs`의 원소와 `probs`는 항상 null이다.

### 7.2 `update(frame, speed_kmh)`

```
 1. has_frame && frame.t_ms <= prev_t:  dropped_frames += 1;  return null

 2. long_gap = false
    !has_frame:  has_frame = true;  origin = frame.t_ms
    아니면 frame.t_ms - prev_t > clock.reset_gap_ms 이면 long_gap = true
    prev_t = frame.t_ms;  frames_accepted += 1;  session_ms = frame.t_ms - origin

 3. sig = signals.compute(frame)                                             // 5.7절

 4. baseline == null && estimator != null:
        done = estimator.update(sig)                                         // 8.2절
        done != null:  baseline = done;  feature_start = done.t_complete_ms
    baseline == null:  state = frame_state(sig, global_baseline, provisional = true)     // 9장
    아니면:            state = frame_state(sig, baseline,        provisional = false)

 5. run = null;  steps = []
    baseline != null && session_ms >= feature_start:
        aggregator == null:  aggregator = StepAggregator(spec, start_ms = feature_start, origin_ms = origin)
        long_gap:  detector.reset()
        run   = detector.update(frame.t_ms, state.closure, state.closure_valid)          // 10장
        steps = aggregator.update(state, run, detector.ongoing_ms())                     // 11장

 6. events = [];  step_probs = [];  probs = null
    steps_emitted += steps.size
    steps의 각 step에 대해 (오래된 것부터):
        push_step(step)
        probs = null
        model != null 이고 window.size == step_features.window_steps:  probs = call_model()   // 그 스텝까지의 창
        step_probs += probs
        events += engine.on_step(step, probs)                                            // 12.6절

 7. events += engine.update(state)                                                       // 12.1~12.5절, 12.7절

 8. alerts = policy.decide(events, speed_kmh, frame.t_ms)                                // 13장

 9. return PipelineOutput(state, runs = run이 있으면 [run] 아니면 [], steps, step_probs,
                          probs,                              // 6단계가 끝났을 때의 값: 마지막 스텝의 확률, 스텝이 없으면 null
                          events, alerts)
```

```
push_step(step):
    !step.full_lookback:                               window를 비운다
    아니면 window_has_k && step.k != window_last_k + 1:  window를 비운다
    step.full_lookback:  window에 step.values를 덧붙이고, 길이가 window_steps를 넘으면 맨 앞을 버린다
    window_has_k = true;  window_last_k = step.k

call_model():
    window의 값 중 유한하지 않은 것이 하나라도 있으면:  nonfinite_windows += 1;  return null
    out = model(window)                 // window[0]이 가장 오래된 스텝
    out.size != 3 -> 오류
    model_calls += 1;  return (out[0], out[1], out[2])
```

알아 둘 것:

- 4단계: 기준선이 확정되는 프레임은 기준선 표본에 들어가지 않고, 그 프레임의 상태부터 새 기준선으로 계산한다
  (`provisional = false`). 같은 프레임에서 5단계가 처음 실행된다.
- 5단계: 기준선 완성 전에는 특징용 검출기와 누적기를 돌리지 않는다. 규칙 엔진(7단계)은 첫 프레임부터 돈다.
  누적의 시작은 기준선을 완성한 프레임의 시각이 아니라 `T_b`(`feature_start`)다. 프레임이 끊겨서 그 프레임이 `T_b`보다
  1초 넘게 늦게 오면, 그 사이에 끝난 스텝을 빈 스텝으로 발행한다(11.6절. 벡터 `rules_calibrating_large_t0`: 61333의
  프레임이 스텝 61을 발행한다).
- 5단계의 `long_gap`은 2단계의 "초과" 비교다. 간격이 정확히 `clock.reset_gap_ms`이면 검출기를 초기화하지 않는다(벡터
  `steps_gap_exactly_10s`).
- 6단계: 모델은 **스텝마다** 부른다(2026-11-13 소유자 결정). 그 스텝을 창에 넣은 뒤 창에 든 가장 최근
  `step_features.window_steps`(120)개 스텝이 모두 `full_lookback`이고 번호가 이어져 있으면, 그 스텝까지의 창으로
  한 번 부르고 그 확률을 그 스텝의 `on_step`에 준다. 프레임이 끊겼다가 와서 스텝 n개가 한꺼번에 나오면 모델을 최대
  n번 부른다(10초 이하의 끊김이므로 n은 10 이하다). 뒤 스텝의 창은 앞 스텝의 창을 한 칸 민 것이다. 스텝을 하나만
  발행시키는 프레임에서는 프레임당 한 번이다. `T_b = 60000`이면 `full_lookback`인 첫 스텝이 120, 첫 호출이 스텝
  239다(벡터 `e2e_250s`). 끊김 뒤의 프레임이 스텝 셋을 발행하고 모델을 세 번 부르는 경우는 벡터
  `rules_drowsy_model_stall`이 고정한다.
- 6단계와 7단계의 순서: 그 프레임이 발행시킨 스텝을 모두 처리한 뒤에 그 프레임 자신을 규칙 엔진에 넣는다. 10초 초과
  끊김 뒤의 첫 프레임에서 규칙 엔진의 초기화(12.7절)는 7단계 안에서 일어난다. 그 프레임은 스텝을 발행시키지 않거나
  (누적기가 같은 조건으로 다시 시작한다), 누적기가 그 프레임에서 처음 프레임을 받는 경우에만 `full_lookback`이 아닌
  빈 스텝을 발행시키므로, 6단계의 `drowsy` 이벤트가 초기화의 end보다 앞에 오는 일은 없다.
- 세는 값: `frames_accepted`, `dropped_frames`, `steps_emitted`, `model_calls`, `nonfinite_windows`. 벡터의
  `expected.counters`와 비교한다. `model_calls`와 `nonfinite_windows`는 스텝마다 세므로 한 프레임에서 여럿 늘 수 있다.

### 7.3 한 프레임 안의 이벤트 순서

1. `drowsy` (발행된 스텝 순서대로)
2. 10초 초과 끊김 뒤의 초기화가 내는 end: `drowsy`, `unable_to_analyze`, `gaze_away_long`, `microsleep` 또는 `sleep`
   순서(12.7절)
3. `calibrating`
4. `unable_to_analyze`
5. `gaze_away_long`
6. `vats` (10초 초과 끊김 뒤의 첫 프레임에서 진행 중이던 `vats`의 end는 여기서 난다)
7. 감김: 앞 감김의 end, `microsleep` onset, `sleep` onset

벡터 `rules_gap_over_10s`(2번의 `gaze_away_long`, `sleep`과 6번의 `vats`), `rules_gap_over_10s_drowsy`(2번의 `drowsy`,
`gaze_away_long`), `rules_alert_priority`(5번과 7번이 같은 프레임)가 이 순서를 고정한다. 경고(`alerts`)의 순서는
이벤트의 순서가 아니라 우선순위다(13장).

---

## 8. 기준선 (`BaselineEstimator`)

<!-- check:pipeline -->

| 경로 | 값 | 단위 | 뜻 |
|------|----|------|----|
| `baseline.duration_ms` | 60000 | ms | 첫 마감(세션 시간) |
| `baseline.extend_step_ms` | 30000 | ms | 완성 조건 미달 시 마감을 늦추는 폭 |
| `baseline.max_duration_ms` | 180000 | ms | 마지막 마감. 여기서도 미달이면 전역 상수로 대체 |
| `baseline.min_face_frames` | 300 | 개 | 얼굴 프레임 수 하한 |
| `baseline.min_face_ratio` | 0.5 | 비율 | 얼굴 검출 비율(시간 기준) 하한 |
| `baseline.e_min` | 0.12 | | E의 허용 범위 하한 |
| `baseline.e_max` | 0.45 | | E의 허용 범위 상한 |
| `baseline.ear_quantile` | 0.9 | | E = EAR의 이 분위수 |
| `baseline.blink_quantile` | 0.1 | | B = `eyeBlink`의 이 분위수 |
| `baseline.blink_record_quantiles` | 0.5, 0.9 | | 분포 기록용(`Baseline`에는 들어가지 않는다) |
| `baseline.off_axis_yaw_deg` | 15.0 | 도 | `abs(yaw0)`이 이 값을 **넘으면** `status = off_axis` |

### 8.1 분위수와 중앙값

```
quantile(values, q):                      // 최근접 순위. 값을 보간하지 않는다
    n = values.size (0이면 오류);  s = 오름차순 정렬
    rank = ceil(q * n) 를 정수로;  rank < 1 이면 1;  rank > n 이면 n
    return s[rank - 1]

median(values):                           // 중앙값만 정의가 다르다
    n = values.size (0이면 오류);  s = 오름차순 정렬
    n이 홀수: s[n / 2]                     (정수 나눗셈)
    n이 짝수: (s[n/2 - 1] + s[n/2]) / 2.0
```

중앙값을 최근접 순위로 하지 않는 이유는 좌우 반전 입력에서 `median(−x) = −median(x)`가 정확히 성립해야 `R0`이 정확히
대칭이 되기 때문이다. 두 정의가 틀리면 벡터 `baseline_normal`에서 드러난다.

### 8.2 수집과 확정

추정기는 처음 받은 프레임의 `t_ms`를 원점으로 삼고 세션 시간 `t`로 계산한다. 수집 구간은 `[0, 마감)`이다.

```
상태: done = false;  deadline = baseline.duration_ms;  has_prev = false;  prev_t = 0;  prev_face = false
      face_ms = 0 (Long);  표본 목록 ear_l, ear_r, blink_l, blink_r, yaw, pitch, roll, jaw, gaze_h, gaze_v

update(sig) -> Baseline?:
    done:  return null
    !has_prev:  origin = sig.t_ms
    t = sig.t_ms - origin
    has_prev && t <= prev_t -> 오류                       // 호출자가 그런 프레임을 버린다

    held = 0                                              // 직전 프레임이 대표한 얼굴 시간
    has_prev && prev_face:  held = min(t - prev_t, clock.max_hold_ms)

    while t >= deadline:                                  // 이 프레임을 넣기 전에 마감을 검사한다
        held_before = min(deadline - prev_t, held)        // 마감을 넘는 부분은 세지 않는다
        b = try_complete(face_ms + held_before)
        b != null:  done = true;  return b
        deadline = min(deadline + baseline.extend_step_ms, baseline.max_duration_ms)

    face_ms += held
    sig.face:  collect(sig)
    has_prev = true;  prev_t = t;  prev_face = sig.face
    return null

collect(sig):                                             // 얼굴 프레임만
    sig.eye_w_l >= signals.min_eye_width_px:  ear_l += sig.ear_l          // 쓸 수 있는 눈의 EAR만 표본에 넣는다
    sig.eye_w_r >= signals.min_eye_width_px:  ear_r += sig.ear_r
    blink_l += sig.blink_l;  blink_r += sig.blink_r
    yaw += sig.yaw;  pitch += sig.pitch;  roll += sig.roll;  jaw += sig.jaw_open
    gaze_h += gaze_h(sig.eye_look);  gaze_v += gaze_v(sig.eye_look)        // 5.4절

try_complete(face_ms) -> Baseline?:
    n_face = blink_l.size                                 // 얼굴 프레임 수
    face_ratio = face_ms / deadline                       (Double 나눗셈)
    e_in_range = false
    ear_l과 ear_r이 둘 다 비어 있지 않으면:
        e_l = quantile(ear_l, baseline.ear_quantile);  e_r = quantile(ear_r, baseline.ear_quantile)
        e_in_range = e_min <= e_l && e_l <= e_max && e_min <= e_r && e_r <= e_max
    complete = n_face >= baseline.min_face_frames && face_ratio >= baseline.min_face_ratio && e_in_range

    !complete && deadline < baseline.max_duration_ms:  return null          // 연장
    !complete:  return 전역 상수 기준선(t_complete_ms = deadline, status = fallback_global)     // 대체 (8.4절)

    yaw0 = median(yaw);  pitch0 = median(pitch);  roll0 = median(roll)
    status = abs(yaw0) > baseline.off_axis_yaw_deg 이면 off_axis, 아니면 ok
    return Baseline(e_l, e_r,
                    b_l = quantile(blink_l, baseline.blink_quantile), b_r = quantile(blink_r, baseline.blink_quantile),
                    r0 = Ry(yaw0)·Rx(pitch0)·Rz(roll0)  (5.3절),
                    jaw0 = median(jaw), gaze_h0 = median(gaze_h), gaze_v0 = median(gaze_v),
                    t_complete_ms = deadline, status)
```

### 8.3 정리

- 완성 조건 셋: 얼굴 프레임 수 ≥ `baseline.min_face_frames`(300), 얼굴 검출 비율(시간 기준) ≥ `baseline.min_face_ratio`(0.5),
  두 눈 모두 `baseline.e_min`(0.12) ≤ E ≤ `baseline.e_max`(0.45).
- 얼굴 검출 비율의 분자는 유지 규칙(6장)으로 센 얼굴 시간이고 분모는 마감 값이다(데이터 없는 시간이 분모에 들어간다).
- 마감 시각에 정확히 온 프레임은 수집 구간에 들어가지 않는다. 마감 이상인 첫 프레임이 오기 전에는 확정되지 않는다.
- 연장하면 표본은 버리지 않고 `[0, 새 마감)` 전체를 쓴다. 마감은 60000, 90000, 120000, 150000, 180000이다.
- 프레임이 끊겼다가 와서 마감을 여러 개 지나면 지난 마감을 순서대로 검사한다. `T_b`는 조건을 만족한 첫 마감이다.
- 한쪽 눈이라도 쓸 수 있는 표본이 하나도 없으면 E의 범위 조건을 어긴 것으로 본다.
- `T_b` = `t_complete_ms`는 확정을 **결정한 마감 값**이다(확정을 알아챈 프레임의 시각이 아니다). 대체된 경우에는 180000이다.
- `status = off_axis`는 표시일 뿐이고 계산은 `ok`와 같다. `fallback_global`에서는 `pose_valid`가 항상 false라 시선 규칙이
  세션 내내 꺼진다(12.3절).
- 참조 구현은 확정할 때 진단용 기록(`BaselineDiagnostics`: 얼굴 프레임 수, 얼굴 검출 비율, `eyeBlink`의 50·90 분위수)을
  남긴다. 판정에 쓰이지 않고 벡터에도 없다. 옮기지 않아도 된다.
- 가정(README에 명시): 세션 시작 시점의 운전자는 각성 상태이고 전방을 본다.

### 8.4 전역 상수 기준선

<!-- check:pipeline -->

| 경로 | 값 | 뜻 |
|------|----|----|
| `global_constants.e` | 0.27 | `e_l = e_r` |
| `global_constants.b` | 0.22 | `b_l = b_r` |
| `global_constants.jaw0` | 0.08 | |
| `global_constants.gaze_h0` | 0.0 | |
| `global_constants.gaze_v0` | 0.0 | |
| `global_constants.r0` | 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 | 단위 행렬(행 우선) |

`status`는 항상 `fallback_global`이다. 쓰는 곳은 둘이다: 기준선 완성 전의 임시 계산(`t_complete_ms = 0`,
`provisional = true`)과 보정 실패 시의 대체(`t_complete_ms = 180000`, `provisional = false`).

---

## 9. closure와 프레임 상태 (`calibrate.frame_state`)

<!-- check:pipeline -->

| 경로 | 값 | 뜻 |
|------|----|----|
| `closure.rho` | 0.35 | EAR 쪽: 뜬 눈 기준의 이 비율까지 좁아지면 1 (잠정) |
| `closure.b_closed` | 0.6 | 블렌드셰이프 쪽: 감긴 눈의 `eyeBlink` 수준 (잠정) |
| `closure.b_span_min` | 0.2 | 분모 `b_closed − B`의 하한 |
| `closure.w` | 0.5 | EAR 쪽 비중 |
| `closure.far_eye_zero_yaw_deg` | 25.0 | `abs(yaw_rel)`이 이 값 이상이면 가까운 눈만 쓴다 |
| `closure.far_eye_ramp_deg` | 10.0 | 먼 쪽 눈 비중이 1에서 0으로 줄어드는 구간 폭 |
| `closure.valid_max_abs_yaw_deg` | 40.0 | `abs(yaw_rel)`이 이 값을 **넘으면** closure 무효 |

```
eye_closure(ear, blink, E, B):
    a = 0.0
    span_e = E * (1.0 - closure.rho)
    span_e > 0.0:  a = clip01((E - ear) / span_e)
    span_b = closure.b_closed - B
    span_b < closure.b_span_min:  span_b = closure.b_span_min
    s = clip01((blink - B) / span_b)
    return closure.w * a + (1.0 - closure.w) * s

frame_state(sig, b, provisional):
    sig.face == false:
        return FrameState(t_ms, face = false, closure = 0.0, closure_valid = false, provisional,
                          yaw_rel = 0.0, pitch_rel = 0.0, pose_valid = false, 나머지 0.0)

    r = Ry(sig.yaw)·Rx(sig.pitch)·Rz(sig.roll)             // 각도에서 다시 만든다 (5.3절)
    (yaw_rel, pitch_rel, _) = Euler(b.r0ᵀ · r)

    usable_l = sig.eye_w_l >= signals.min_eye_width_px
    usable_r = sig.eye_w_r >= signals.min_eye_width_px

    closure = 0.0;  closure_valid = false
    (usable_l || usable_r) && abs(yaw_rel) <= closure.valid_max_abs_yaw_deg:
        closure_valid = true
        usable_l && usable_r:
            c_l = eye_closure(sig.ear_l, sig.blink_l, b.e_l, b.b_l)
            c_r = eye_closure(sig.ear_r, sig.blink_r, b.e_r, b.b_r)
            w_far = clip01((closure.far_eye_zero_yaw_deg - abs(yaw_rel)) / closure.far_eye_ramp_deg)
            yaw_rel > 0.0:  c_far = c_l;  c_near = c_r      // 먼 쪽 눈: yaw_rel > 0이면 L
            아니면:          c_far = c_r;  c_near = c_l      // yaw_rel이 정확히 0이면 R을 먼 쪽으로 친다(w_far = 1이라 결과는 같다)
            closure = (c_near + w_far * c_far) / (1.0 + w_far)
        usable_l만:  closure = eye_closure(L)               // 먼 쪽 눈 가중을 하지 않는다
        usable_r만:  closure = eye_closure(R)

    return FrameState(t_ms, face = true, closure, closure_valid, provisional, yaw_rel, pitch_rel,
                      pose_valid = (b.status != fallback_global),
                      gaze_h_rel = gaze_h(sig.eye_look) - b.gaze_h0,
                      gaze_v_rel = gaze_v(sig.eye_look) - b.gaze_v0,
                      jaw_rel    = sig.jaw_open - b.jaw0)
```

- `R`은 행렬에서 직접 꺼내지 않고 `FrameSignals`의 `(yaw, pitch, roll)`로 **다시 만든다**(코드의 가정). MediaPipe의 회전
  블록은 float32라 정확히 직교하지 않으므로, 행렬에서 직접 꺼내면 마지막 자리가 달라질 수 있다. 같은 순서로 구현한다.
- 유효하지 않은 프레임의 closure는 0.0이고 `closure_valid = false`다. 보간하거나 이전 값을 이어 쓰지 않는다.
- 얼굴이 있으면 `yaw_rel`, `pitch_rel`, `gaze_*_rel`, `jaw_rel`은 closure가 무효여도 채운다.
- 전역 상수 기준선에서는 `R0`이 단위 행렬이라 `yaw_rel`, `pitch_rel`이 절대 각도와 같다.
- 15도까지는 두 눈 평균(`w_far = 1`), 25도부터는 가까운 눈만(`w_far = 0`) 쓴다. 벡터 `closure_yaw_weighting`.

---

## 10. 감김 구간 검출기 (`ClosureRunDetector`)

closure에 히스테리시스를 거는 상태 머신이다. 같은 클래스를 설정만 바꿔 두 번 쓴다.

<!-- check:pipeline -->

| 경로 (특징용) | 값 | 단위 | 뜻 |
|---------------|----|------|----|
| `run_detector.enter_thr` | 0.5 | | closure가 이 값 **이상**이면 구간 시작 |
| `run_detector.exit_thr` | 0.3 | | 이 값 **미만**이면 구간 종료 |
| `run_detector.min_duration_ms` | 50 | ms | `duration_ms`가 이 값 **미만**인 구간은 버린다 |
| `run_detector.max_dt_ms` | 200 | ms | 두 유효 표본의 간격이 이 값 **이하**일 때만 보간과 속도 계산에 쓴다 |
| `run_detector.hold_ms` | 500 | ms | 마지막 유효 표본 뒤로 이 값을 **넘게** 끊기면 구간을 닫는다 |
| `run_detector.velocity_lookback_samples` | 2 | 개 | 감는 속도는 진입 이 표본 수 전부터 본다 |

<!-- check:rules -->

| 경로 (규칙용, `rules.json`) | 값 | 단위 |
|-----------------------------|----|------|
| `closure.detector.enter_thr` | 0.7 | |
| `closure.detector.exit_thr` | 0.5 | |
| `closure.detector.min_duration_ms` | 50 | ms |
| `closure.detector.max_dt_ms` | 200 | ms |
| `closure.detector.hold_ms` | 500 | ms |
| `closure.detector.velocity_lookback_samples` | 2 | 개 |

### 10.1 호출 계약

- 받아들인 프레임마다 `update(t, closure, valid)`를 한 번 부른다. 얼굴이 없거나 closure가 유효하지 않은 프레임도
  `valid = false`로 넘긴다(끊긴 시간과 HOLD 판정에 필요하다). 이때 closure 값은 읽지 않는다.
- `t`는 호출마다 엄격히 증가해야 한다. 아니면 오류다(그런 프레임은 `DriverPipeline`이 미리 버린다).
- `update` 한 번은 구간을 최대 하나 돌려준다.
- 판정은 프레임이 도착할 때만 한다.
- 프레임 간격이 `clock.reset_gap_ms`를 넘으면 호출자가 `reset()`을 부른 뒤 그 프레임을 넘긴다(7.2절 5단계, 12.7절).
- 설정은 `max_dt_ms <= hold_ms`여야 한다. 참조 구현은 두 검출기 설정(`pipeline.run_detector`, 규칙의
  `closure.detector`)을 읽을 때 이를 어기면 거부한다(지금 값은 200과 500). 그래야 `hold_ms`를 넘는 끊김으로 구간이
  잘렸을 때 그 끊김을 건너는 쌍이 언제나 "쓸 수 없는 쌍"이 된다. `max_dt_ms`는 0보다 커야 하므로 `hold_ms`도 0일 수 없다.
- 설정은 `hold_ms < clock.reset_gap_ms`이기도 해야 한다(지금 값은 500과 10000). 두 검출기 설정 모두 같은 파일의
  `clock` 블록과 비교하고, 같거나 크면 설정을 읽을 때 거부한다(오류 문장에 두 값이 적힌다). 그래야 초기화를 일으키는
  끊김(`clock.reset_gap_ms` 초과)이 언제나 진행 중인 구간을 자르는 끊김(`hold_ms` 초과)이기도 하다. `hold_ms`가 그
  이상이면 끊김 앞의 감김과 뒤의 감김을 가르는 것이 호출자의 `reset()` 호출 하나뿐이 된다(규칙 엔진에서 그 호출을
  빠뜨리면 두 감김이 하나로 이어져 `microsleep`과 `sleep`의 `t_start_ms`가 끊김 앞으로 간다). 조건을 지키는 설정에서는
  규칙 엔진이 그 호출을 빠뜨려도 이벤트가 같아서, 그 호출은 벡터로 고정되지 않는다(12.7절). 두 조건을 합치면
  `0 < max_dt_ms <= hold_ms < clock.reset_gap_ms`다. 같은 까닭의 조건이 하품과 끄덕임의 계수기에도 있다:
  `yawn.cancel_gap_ms`와 `nod.cancel_gap_ms`도 `clock.reset_gap_ms`보다 작아야 한다(11.5절).

### 10.2 알고리즘

```
상태 변수
  state ∈ {OPEN, CLOSED, HOLD} = OPEN
  has_frame = false, last_t                    // 마지막으로 받은 프레임(유효 여부와 무관). 순서 검사에만 쓴다
  has_prev = false, prev_t, prev_c             // 직전 "유효" 표본. 무효 프레임이 와도 지우지 않는다
  recent_up[velocity_lookback_samples] = 0.0   // 최근 쌍들의 closure 증가율(0 이상), 오래된 것부터.
                                               // 쓸 수 없는 쌍이 오면 모두 0.0으로 지운다
  // 진행 중인 구간 (CLOSED, HOLD에서만 뜻이 있다)
  start_ms                                     // 구간의 시작 = 첫 토막의 시작
  sealed_ms, span_start_ms                     // 끝난 토막들의 길이 합, 지금 토막의 시작 (10.4절)
  last_closed_ms                               // closure가 enter 이상이던 마지막 유효 프레임의 시각
  last_down_ms                                 // 마지막으로 enter를 아래로 지난 시각
  amplitude, closing_peak, opening_peak, pending_up

update(t, c, valid) -> ClosureRun?:
    has_frame && t <= last_t -> 오류
    has_frame = true;  last_t = t
    run = null

    // (1) 끊김 검사. 끊긴 시간 = t − 마지막 유효 표본의 시각
    state != OPEN && t - prev_t > hold_ms:  run = finish(truncated = true)           // state는 OPEN이 된다

    // (2) 무효 프레임
    !valid:
        state == CLOSED:  state = HOLD
        return run

    // (3) 직전 유효 표본과의 쌍
    pair_ok = has_prev && t - prev_t <= max_dt_ms                                    // "쓸 수 있는 쌍"
    up = 0.0;  down = 0.0
    pair_ok:
        rate = (c - prev_c) * 1000.0 / (t - prev_t)                                  // 1/s
        rate > 0.0:  up = rate
        rate < 0.0:  down = -rate
    아니면:
        recent_up을 모두 0.0으로                       // 쓸 수 없는 쌍: 거슬러 보기를 여기서 끊는다
    recent_up을 한 칸 앞으로 밀고 마지막 칸에 up을 넣는다

    // (4) 전이
    state == OPEN:
        c >= enter_thr:  begin(t, c, pair_ok)
    아니면:                                             // CLOSED 또는 HOLD이고 끊긴 시간 <= hold_ms
        resumed = (state == HOLD);  state = CLOSED
        c > amplitude:                                 // 정점이 이 표본으로 옮겨졌다
            amplitude = c
            closing_peak = max(closing_peak, pending_up, up)
            pending_up = 0.0;  opening_peak = 0.0
        아니면:
            pending_up = max(pending_up, up);  opening_peak = max(opening_peak, down)
        c >= enter_thr:
            prev_c < enter_thr:                        // 다시 감김: enter 아래로 내려갔다가 구간이 끝나기 전에 다시 올라왔다
                sealed_ms += last_down_ms - span_start_ms          // 앞 토막을 마지막으로 내려간 시각에서 닫아 더한다
                span_start_ms = up_crossing(t, c, pair_ok)         // 이번에 올라온 시각에서 새 토막을 연다
            last_closed_ms = t
        아니면:
            prev_c >= enter_thr:  last_down_ms = down_crossing(t, c, pair_ok)
            c < exit_thr:  run = finish(truncated = resumed)

    has_prev = true;  prev_t = t;  prev_c = c
    return run

begin(t, c, pair_ok):                                  // OPEN -> CLOSED. 진입 임계 이상을 처음 본 표본
    start_ms = up_crossing(t, c, pair_ok)
    closing_peak = recent_up의 최댓값                   // 방금 넣은 진입 쌍까지 포함
    state = CLOSED;  sealed_ms = 0;  span_start_ms = start_ms
    last_closed_ms = t;  last_down_ms = t;  amplitude = c
    opening_peak = 0.0;  pending_up = 0.0

up_crossing(t, c, pair_ok):                            // c >= enter_thr. 구간의 시작과 뒤 토막의 시작에 쓴다
    !pair_ok || prev_c >= enter_thr:  return t         // 보간할 수 없으면 이 프레임의 시각
    fraction = (enter_thr - prev_c) / (c - prev_c)
    return prev_t + round_half_up(fraction * (t - prev_t))

down_crossing(t, c, pair_ok):                          // prev_c >= enter_thr > c
    !pair_ok:  return prev_t
    fraction = (prev_c - enter_thr) / (prev_c - c)
    return prev_t + round_half_up(fraction * (t - prev_t))

finish(truncated) -> ClosureRun?:
    state = OPEN
    end_ms = truncated 이면 last_closed_ms, 아니면 last_down_ms
    duration_ms = sealed_ms + (end_ms - span_start_ms)  // 끝난 토막들 + 마지막 토막. 토막이 하나면 end_ms − start_ms
    duration_ms < min_duration_ms:  return null         // 버린다
    velocity_valid = !truncated && closing_peak > 0.0 && opening_peak > 0.0
    return ClosureRun(start_ms, end_ms, duration_ms, amplitude,
                      peak_closing_velocity = velocity_valid 이면 closing_peak, 아니면 0.0,
                      peak_opening_velocity = velocity_valid 이면 opening_peak, 아니면 0.0,
                      velocity_valid, truncated)

ongoing_ms():    state == OPEN 이면 0, 아니면 sealed_ms + (last_closed_ms - span_start_ms)
run_start_ms():  state == OPEN 이면 0, 아니면 start_ms
reset():         state = OPEN;  has_frame = false;  has_prev = false;  recent_up을 모두 0.0으로
                 (진행 중이던 구간은 발행하지 않고 버린다)
```

보간식의 계산 순서는 적힌 그대로다: 나눗셈으로 `fraction`을 구하고, 간격(정수 ms를 `Double`로 바꾼 값)을 곱하고,
`floor(x + 0.5)`로 정수를 만든 다음 `prev_t`(`Long`)에 정수로 더한다. 반올림하는 값이 두 closure와 간격만으로
정해지므로 `t_ms`의 원점(첫 프레임의 절대 시각)이 달라도 결과가 같다. 절대 시각에 더한 뒤 반올림하면 `t_ms`가 큰
값일 때 `Double`의 마지막 자리가 잘려 1 ms 달라질 수 있다(벡터 `runs_rounding_large_t0`: 그 경우 49 ms가 되어
버려지는 구간이, 옳게 구현하면 50 ms로 남는다).

### 10.3 상태 전이표

`c` = closure, `enter` = `enter_thr`, `exit` = `exit_thr`. 끊긴 시간 = 이 프레임의 `t` − 마지막 유효 표본의 시각.

| 상태 | 유효, `c >= enter` | 유효, `exit <= c < enter` | 유효, `c < exit` | 무효 프레임 |
|------|--------------------|---------------------------|------------------|-------------|
| `OPEN` | `CLOSED`로. `start_ms` 기록 | 그대로 | 그대로 | 그대로 |
| `CLOSED`, 끊긴 시간 ≤ `hold_ms` | 그대로. 정점과 속도 갱신. 직전 유효 표본이 `enter` 미만이었으면 앞 토막을 닫고 새 토막을 연다. `last_closed_ms = t` | 그대로. 직전 유효 표본이 `enter` 이상이었으면 `last_down_ms` 기록 | (직전 유효 표본이 `enter` 이상이었으면 `last_down_ms` 기록 후) 구간 발행, `OPEN`으로 | `HOLD`로 |
| `HOLD`, 끊긴 시간 ≤ `hold_ms` | `CLOSED`로 복귀. 이하 `CLOSED` 행과 같다(새 토막 포함) | `CLOSED`로 복귀. 이하 `CLOSED` 행과 같다 | 구간 발행(`truncated = true`), `OPEN`으로 | 그대로 |
| `CLOSED` 또는 `HOLD`, 끊긴 시간 > `hold_ms` | 구간 발행(`truncated = true`) 뒤 `OPEN` 행으로 처리: 새 구간이 이 표본에서 시작한다 | 구간 발행(`truncated = true`), `OPEN` | 구간 발행(`truncated = true`), `OPEN` | 구간 발행(`truncated = true`), `OPEN` |

"구간 발행"은 `duration_ms >= min_duration_ms`일 때만 `ClosureRun`이 나간다는 뜻이다. 짧으면 아무것도 돌려주지 않고
상태만 `OPEN`이 된다.

### 10.4 시각, 길이, 속도의 정의

| 값 | 정의 |
|----|------|
| 쓸 수 있는 쌍 | 직전 유효 표본과 이번 유효 표본. 간격이 `max_dt_ms` 이하일 때만. 사이에 무효 프레임이 끼어 있어도 된다. 직전 유효 표본이 없거나 간격이 `max_dt_ms`를 넘으면 "쓸 수 없는 쌍"이다 |
| `start_ms` | 직전 유효 표본(`enter` 미만)과 진입 표본 사이에서 closure가 `enter`를 지난 시각의 선형 보간. 쓸 수 없는 쌍이면 진입 표본의 시각 |
| `end_ms` (잘리지 않음) | 마지막으로 `enter`를 아래로 지난 시각의 선형 보간. 쓸 수 없는 쌍이면 `enter` 이상이었던 마지막 유효 표본의 시각 |
| `end_ms` (잘림) | `enter` 이상이었던 마지막 유효 프레임의 시각(`last_closed_ms`). 그 뒤에 `enter` 아래로 내려간 표본을 봤어도(아래로 지난 시각을 보간해 두었어도) 같다 |
| 보간한 시각 | 직전 표본의 시각 + `round_half_up(fraction * 간격)`. `fraction`은 두 표본 사이에서 `enter`에 닿는 비율(0~1)이다. 원점과 무관하다 |
| 토막 | 구간 안에서 closure가 `enter` 이상이던 한 덩어리. `enter`를 위로 지난 시각에 시작해서 그다음에 `enter`를 아래로 지난 시각에 끝난다. 첫 토막의 시작이 `start_ms`다. 구간이 이어지는 중에(`CLOSED`, 또는 `HOLD`에서 복귀한 프레임) 직전 유효 표본이 `enter` 미만이고 이번 표본이 `enter` 이상이면 앞 토막이 끝나고 새 토막이 시작한다. 새 토막의 시작은 `start_ms`와 같은 방법으로 구한다(쓸 수 있는 쌍이면 보간, 아니면 그 프레임의 시각) |
| `duration_ms` | 토막 길이의 합 = `sealed_ms + (end_ms − span_start_ms)`. 잘린 구간도 같은 식이고 마지막 토막이 `end_ms`(= `last_closed_ms`)에서 끝난 것으로 센다. `end_ms − start_ms` 이하이고, 토막이 하나뿐이면 같다. closure가 `exit`와 `enter` 사이에 머문 시간은 구간을 끝내지 않지만 길이에도 들어가지 않는다 |
| `amplitude` | 구간에 속한 유효 표본의 최대 closure |
| 정점 | 최대 closure에 **처음** 도달한 표본(`c > amplitude`일 때만 옮긴다) |
| `peak_closing_velocity` | 진입 표본의 `velocity_lookback_samples`개 전 표본부터 정점 표본까지의 쓸 수 있는 쌍 중 증가율 최댓값. 진입 전의 쌍은 진입 표본까지 쓸 수 있는 쌍이 끊기지 않고 이어진 것만 본다(쓸 수 없는 쌍을 만나면 그보다 오래된 쌍은 버린다) |
| `peak_opening_velocity` | 정점 표본부터 구간을 닫은 표본까지의 쓸 수 있는 쌍 중 감소율 최댓값 |
| `ongoing_ms()` | 진행 중인 구간에서 closure가 `enter` 이상이던 시간의 합 = `sealed_ms + (last_closed_ms − span_start_ms)`. 끝은 `enter` 이상이었던 마지막 유효 프레임의 시각이다(보간하지 않는다). `OPEN`이면 0. 한 구간 안에서는 줄지 않는다 |
| `run_start_ms()` | 진행 중인 구간의 `start_ms`(구간이 끝나 발행될 `ClosureRun.start_ms`와 같은 값). `OPEN`이면 0이고 뜻이 없다. 토막이 여럿이면 `t − ongoing_ms()`는 `start_ms`가 아니므로, 구간의 시작이 필요한 쪽(규칙 엔진, 12.5절)은 이 값을 읽는다 |

### 10.5 경계 사례

1. 끊긴 시간이 정확히 `hold_ms`이면 같은 구간으로 잇는다(`>`만 끊는다).
2. 끊긴 시간은 `HOLD`뿐 아니라 `CLOSED`에서도 검사한다(코드의 가정). 무효 프레임 없이 프레임 자체가 `hold_ms` 넘게
   끊겨도 구간을 닫는다. 이때 다음 프레임이 감긴 표본이면 한 번의 `update`에서 잘린 구간이 나가고 새 구간이 시작한다
   (벡터 `runs_truncated_by_gap`).
3. `HOLD`에서 돌아온 첫 유효 표본이 `exit` 미만이면 끊긴 시간이 `hold_ms` 이하여도 `truncated = true`이고 `end_ms`는
   `last_closed_ms`다.
4. `min_duration_ms`는 잘린 구간에도 적용한다. 버려진 구간은 스텝 특징 어디에도 들어가지 않는다.
5. 세션(또는 초기화 뒤)의 첫 유효 표본이 `enter` 이상이면 `start_ms`는 그 표본의 시각이다.
6. 한 표본짜리 깜빡임은 앞뒤 표본으로 보간한 길이가 `duration_ms`다(벡터 `runs_one_sample_blink`). 감긴 표본이
   하나뿐인 채로 잘리면 `end_ms`는 그 표본의 시각이고 `duration_ms`는 보간한 시작부터 그 시각까지다. 그 길이가
   `min_duration_ms` 미만이면 버린다(벡터 `runs_one_sample_truncated`: 83 ms는 남고 47 ms는 버려진다).
7. `ongoing_ms()`는 눈꺼풀이 `exit`와 `enter` 사이에 머무는 동안과 `HOLD` 동안에는 늘어나지 않는다. 다시 `enter` 이상인
   표본이 와도 머문 시간이 뒤늦게 더해지지 않는다: 새 토막이 그 표본 쪽에서 시작할 뿐이다(2026-11-13 소유자 결정).
   지금 상수에서는 한 프레임에 늘어나는 양이 `hold_ms`(500) 이하다(이어진 감긴 표본 사이는 끊긴 시간까지, 새 토막이
   열리는 프레임에서는 앞 토막의 끝과 새 토막의 시작을 보간한 두 조각까지). 벡터 `runs_lingering_lids`,
   `runs_multi_span`, `steps_multi_span_runs`, `rules_closure_dwell_below_1s`.
   구간 자체는 closure가 `exit` 아래로 내려가야(또는 끊겨서) 끝난다. 그래서 눈꺼풀이 `exit`와 `enter` 사이에 머문 채
   깜빡이면 그 깜빡임들이 한 구간으로 이어지고, `enter` 이상이던 시간이 그 구간의 길이에 계속 쌓인다. 이것은
   2026-11-14 소유자 결정으로 지금대로 둔 동작이고 알려진 한계다. 이식 코드는 이 누적을 그대로 재현해야 한다(잰 사례와
   고르지 않은 대안은 20장 머리의 결정 기록).
8. 두 표본이 모두 `enter` 이상이면 그 사이의 끊김(`hold_ms` 이하)은 토막 안이다. 그 시간은 길이에 들어간다. `enter`
   아래로 내려간 표본을 본 뒤 `HOLD`를 거쳐 `enter` 이상으로 돌아오면 새 토막이고, 그 쌍이 쓸 수 없는 쌍이면 새 토막은
   돌아온 프레임의 시각에서 시작한다(벡터 `runs_multi_span`의 둘째 구간).
9. 토막이 여럿인 구간이 잘리면 `duration_ms = sealed_ms + (last_closed_ms − span_start_ms)`다(벡터
   `runs_multi_span_truncated`). `min_duration_ms`는 이 합에 적용한다.
10. closure가 정확히 `enter`인 표본은 감긴 표본이다(`>=`). 그런 표본은 토막을 나누지 않는다.
11. 감는 속도의 거슬러 보기는 **쓸 수 없는 쌍에서 끊는다**. 직전 유효 표본과의 간격이 `max_dt_ms`를 넘으면 그보다 앞에서
    잰 증가율을 모두 버리므로, 무효 프레임이나 프레임 누락이 길게 이어진 뒤, `hold_ms`를 넘는 끊김으로 구간이 잘린 뒤,
    `reset()` 뒤의 첫 유효 표본에서 시작한 구간에는 끊김 앞의 증가율이 넘어오지 않는다. 사이에 무효 프레임이 끼어 있어도
    간격이 `max_dt_ms` 이하인 쌍은 쓸 수 있는 쌍이라 끊지 않는다(코드의 가정). 벡터 `runs_velocity_after_gap`.
12. 진입 쌍이 쓸 수 없는 쌍이어도(감는 모습을 보지 못한 구간) 구간 안에서 정점까지 오른 쌍이 있으면 그 증가율이 감는
    속도다(코드의 가정. 20장 3번).
13. `velocity_valid = false`인 경우: 잘린 구간, 감는 쪽이나 뜨는 쪽에 쓸 수 있는 쌍의 변화가 없는 구간.
14. `reset()`은 진행 중이던 구간을 발행하지 않는다.
15. 보간식의 분모는 0이 되지 않는다(`up_crossing`에서는 `prev_c < enter <= c`, `down_crossing`에서는 `prev_c >= enter > c`).
16. 정점이 뒤의 표본으로 옮겨지면(더 큰 closure가 나오면) 그때까지의 쌍은 모두 감는 쪽이다. 앞 정점 뒤에서 잰 증가율
    (`pending_up`)도 감는 속도의 후보가 되고, 그때까지 잰 뜨는 속도는 버리고 새 정점 뒤부터 다시 잰다. 정점을 넘지 못한
    증가율은 그 구간의 감는 속도가 아니고 다음 구간으로도 넘어가지 않는다(`begin`이 `pending_up`을 0으로 둔다). 같은
    최댓값이 다시 나와도 정점은 옮기지 않는다. 벡터 `runs_two_humps`가 네 경우를 고정한다. `pending_up`은 앞 정점 뒤에서
    잰 증가율 가운데 **가장 큰 값**이다(마지막 값이 아니다). 정점이 옮겨지기 전에 증가가 두 번 있고 큰 것이 먼저인
    구간에서 갈린다(벡터 `runs_two_rises_before_peak`: 3.64/s, 0.91/s의 순서로 올랐다가 정점이 옮겨지는 구간의 감는
    속도는 3.64/s다. 마지막 값만 들고 있으면 3.4/s가 된다).

---

## 11. 시간 누적과 스텝 특징 (`StepAggregator`)

<!-- check:pipeline -->

| 경로 | 값 | 단위 | 뜻 |
|------|----|------|----|
| `step_features.window_steps` | 120 | 스텝 | 모델 입력 창 길이 |
| `step_features.lookback_long_ms` | 60000 | ms | 긴 조회 구간. `full_lookback`의 기준이기도 하다 |
| `step_features.lookback_short_ms` | 10000 | ms | 짧은 조회 구간 |
| `step_features.perclos_thr` | 0.8 | | closure가 이 값 **이상**인 시간을 센다 (잠정) |
| `step_features.perclos_min_valid_ms_60` | 5000 | ms | 유효 시간이 이 값 미만이면 `perclos_60 = 0` |
| `step_features.perclos_min_valid_ms_10` | 2000 | ms | 유효 시간이 이 값 미만이면 `perclos_10 = 0` |
| `step_features.blink_rate_min_valid_ms` | 5000 | ms | 유효 시간이 이 값 미만이면 `blink_rate_60 = 0` |
| `step_features.blink_duration_cap_ms` | 2000 | ms | 지속 시간 평균과 표준편차에서 이벤트별 상한 |
| `step_features.long_closure_min_ms` | 500 | ms | 이 길이 **이상**이면 긴 감김 |
| `step_features.max_closure_cap_ms` | 10000 | ms | `max_closure_ms_*`의 상한 |
| `step_features.pitch_std_min_face_ms` | 5000 | ms | 얼굴 시간이 이 값 미만이면 `pitch_std_60 = 0` |

### 11.1 스텝 통

스텝마다 누적값을 담는 통을 하나 둔다. 끝난 스텝의 통을 오래된 것부터 최대 60개(`lookback_long_ms / step_ms`) 보관한다.

| 통의 필드 | 형 | 내용 |
|-----------|----|------|
| `valid_ms` | Long | closure가 유효한 시간 |
| `closed_ms` | Long | 그중 `closure >= step_features.perclos_thr`인 시간 |
| `closure_sum` | Double | Σ `closure * ms` (유효 시간에 대해) |
| `face_ms` | Long | 얼굴이 있는 시간 |
| `pitch_sum` | Double | Σ `pitch_rel * ms` (얼굴 시간에 대해) |
| `pitch_sq_sum` | Double | Σ `pitch_rel * pitch_rel * ms` |
| `n_events` | Int | 잘리지 않은 감김 구간 수 |
| `dur_sum`, `dur_sq_sum` | Long | Σ `d`, Σ `d * d`. `d = min(duration_ms, step_features.blink_duration_cap_ms)`. `duration_ms`는 `ClosureRun`의 값 그대로다(진입 임계 이상이던 시간의 합, 10.4절) |
| `n_velocity` | Int | 그중 `velocity_valid`인 구간 수 |
| `closing_vel_sum`, `opening_vel_sum` | Double | 그 구간들의 두 속도 합 |
| `n_long` | Int | `duration_ms >= step_features.long_closure_min_ms`인 구간 수(잘린 것 포함) |
| `max_run_ms` | Long | 이 통에 귀속된 구간의 최대 `duration_ms`(잘린 것 포함, 상한을 걸지 않은 값) |
| `n_yawn`, `n_nod` | Int | 하품, 끄덕임 수 |

### 11.2 `update(state, run, ongoing_ms)`

받아들인 프레임마다, 특징용 검출기를 갱신한 **뒤에** 부른다. `run`은 그 프레임에서 검출기가 돌려준 구간, `ongoing_ms`는
그 프레임을 처리한 뒤의 `detector.ongoing_ms()`다.

```
StepAggregator(spec, start_ms, origin_ms):
    start_ms < 0 또는 clock.step_ms의 배수가 아님 -> 오류
    accum_start = start_ms                    // 지금 누적을 시작한 세션 시간. 초기화하면 뒤로 옮겨진다
    cur_k = start_ms / step_ms + 1;  cur = 새 통;  history = []
    has_prev = false;  prev_t = start_ms      // 프레임을 받기 전에는 start_ms에 "데이터 없음" 프레임이 있다고 본다
    prev_face = false;  prev_valid = false;  prev_closure = 0.0;  prev_pitch = 0.0;  prev_ongoing = 0
    yawn = 하품 계수기;  nod = 끄덕임 검출기   // 11.5절

update(state, run, ongoing_ms) -> StepFeatures 목록:
    t = state.t_ms - origin_ms
    t < start_ms:  return []                              // 누적 시작 전의 프레임은 무시한다
    has_prev && t <= prev_t -> 오류

    steps = []
    t - prev_t > clock.reset_gap_ms:  restart(t)          // 11.6절. 이 프레임은 스텝을 발행하지 않는다
    아니면:  advance_to(t, steps)

    // 이 프레임에서 생긴 이벤트는 이 프레임이 속한 스텝의 통(cur)에 넣는다
    run != null:  add_run(run)
    yawn.update(t, state.jaw_rel, valid = state.face):   cur.n_yawn += 1
    nod.update(t, state.pitch_rel, valid = state.face):  cur.n_nod += 1

    has_prev = true;  prev_t = t;  prev_face = state.face;  prev_valid = state.closure_valid
    prev_closure = state.closure;  prev_pitch = state.pitch_rel;  prev_ongoing = ongoing_ms
    return steps

advance_to(t, steps):                 // 직전 프레임의 상태가 유지된 구간을 누적하면서 t까지 끝난 스텝을 발행한다
    pos = prev_t;  hold_end = prev_t
    has_prev:  hold_end = prev_t + min(t - prev_t, clock.max_hold_ms)
    while cur_k * step_ms <= t:
        boundary = cur_k * step_ms
        pos < hold_end:
            piece_end = min(hold_end, boundary)            // 스텝 경계에서 정확히 나눈다
            add_time(piece_end - pos);  pos = piece_end
        steps += emit()
    pos < hold_end:  add_time(hold_end - pos)

add_time(ms):                         // 직전 프레임의 상태를 ms만큼 cur에 더한다
    prev_face:
        cur.face_ms += ms
        cur.pitch_sum += prev_pitch * ms
        cur.pitch_sq_sum += prev_pitch * prev_pitch * ms
    prev_valid:
        cur.valid_ms += ms
        cur.closure_sum += prev_closure * ms
        prev_closure >= step_features.perclos_thr:  cur.closed_ms += ms

add_run(run):
    !run.truncated:
        d = min(run.duration_ms, step_features.blink_duration_cap_ms)
        cur.n_events += 1;  cur.dur_sum += d;  cur.dur_sq_sum += d * d
        run.velocity_valid:
            cur.n_velocity += 1
            cur.closing_vel_sum += run.peak_closing_velocity;  cur.opening_vel_sum += run.peak_opening_velocity
    run.duration_ms >= step_features.long_closure_min_ms:  cur.n_long += 1
    run.duration_ms > cur.max_run_ms:  cur.max_run_ms = run.duration_ms

emit() -> StepFeatures:
    history에 cur를 덧붙이고, 60개를 넘으면 맨 앞을 버린다
    k = cur_k;  end_ms = k * step_ms
    step = StepFeatures(k, t_end_ms = origin_ms + end_ms,
                        full_lookback = (end_ms >= accum_start + step_features.lookback_long_ms),
                        values = features())               // 11.3, 11.4절
    cur = 새 통;  cur_k = k + 1
    return step
```

### 11.3 구간 합

스텝이 끝날 때 `history`의 통을 **오래된 것부터** 더한다. 60초 구간은 `history` 전체(최대 60개), 10초 구간은 마지막
10개(`lookback_short_ms / step_ms`)다. 방금 끝난 스텝의 통이 마지막이다.

| 기호 | 60초 구간의 합 | 10초 구간에도 필요한가 |
|------|----------------|------------------------|
| `V` | Σ `valid_ms` | 예 |
| `C` | Σ `closed_ms` | 예 |
| `S` | Σ `closure_sum` | 예 |
| `F` | Σ `face_ms` | 예 |
| `P`, `P2` | Σ `pitch_sum`, Σ `pitch_sq_sum` | |
| `N`, `D`, `D2` | Σ `n_events`, Σ `dur_sum`, Σ `dur_sq_sum` | |
| `NV`, `CV`, `OV` | Σ `n_velocity`, Σ `closing_vel_sum`, Σ `opening_vel_sum` | |
| `L` | Σ `n_long` | |
| `M` | max `max_run_ms` | 예 |
| `Y`, `ND` | Σ `n_yawn`, Σ `n_nod` | |

`G` = `prev_ongoing`: 이 스텝을 발행시킨 프레임이 **아니라 그 직전 프레임**에서 받은 `ongoing_ms`다(코드의 가정). 스텝 k의
특징은 세션 시간이 `1000k`보다 작은 프레임까지의 정보만 쓴다. `std(mean, mean_sq) = sqrt(max(mean_sq − mean * mean, 0.0))`.

### 11.4 특징 17개 (순서 고정)

아래 식에서 60은 60초 구간의 합, 10은 10초 구간의 합을 뜻한다(예: `V60`, `V10`). 정수 합은 나눗셈 직전에 `Double`로 바꾼다.

<!-- check:features -->

| # | 이름 | 구간(s) | 단위 | 변환 | 정의 불가 값 | 식 |
|---|------|---------|------|------|--------------|----|
| 0 | `perclos_60` | 60 | ratio | none | 0.0 | `V60 >= perclos_min_valid_ms_60`이면 `C60 / V60` |
| 1 | `perclos_10` | 10 | ratio | none | 0.0 | `V10 >= perclos_min_valid_ms_10`이면 `C10 / V10` |
| 2 | `closure_mean_60` | 60 | 0-1 | none | 0.0 | `V60 > 0`이면 `S60 / V60` |
| 3 | `closure_mean_10` | 10 | 0-1 | none | 0.0 | `V10 > 0`이면 `S10 / V10` |
| 4 | `blink_rate_60` | 60 | 1/min | none | 0.0 | `V60 >= blink_rate_min_valid_ms`이면 `N * 60000.0 / V60` |
| 5 | `blink_dur_mean_60` | 60 | ms | none | 0.0 | `N > 0`이면 `D / N` |
| 6 | `blink_dur_std_60` | 60 | ms | none | 0.0 | `N >= 2`이면 `std(D / N, D2 / N)` |
| 7 | `closing_vel_mean_60` | 60 | 1/s | none | 0.0 | `NV > 0`이면 `CV / NV` |
| 8 | `opening_vel_mean_60` | 60 | 1/s | none | 0.0 | `NV > 0`이면 `OV / NV` |
| 9 | `long_closure_count_60` | 60 | count | log1p | 0.0 | `L` + (`G >= long_closure_min_ms`이면 1) |
| 10 | `max_closure_ms_60` | 60 | ms | log1p | 0.0 | `min(max(M60, G), max_closure_cap_ms)` |
| 11 | `max_closure_ms_10` | 10 | ms | log1p | 0.0 | `min(max(M10, G), max_closure_cap_ms)` |
| 12 | `yawn_count_60` | 60 | count | log1p | 0.0 | `Y` |
| 13 | `pitch_std_60` | 60 | deg | none | 0.0 | `F60 >= pitch_std_min_face_ms`이면 `std(P / F60, P2 / F60)` |
| 14 | `nod_count_60` | 60 | count | log1p | 0.0 | `ND` |
| 15 | `face_missing_ratio_60` | 60 | ratio | none | 0.0 | `(60000 − F60) / 60000` |
| 16 | `face_missing_ratio_10` | 10 | ratio | none | 0.0 | `(10000 − F10) / 10000` |

- "변환"은 ONNX 그래프 안에서 표준화 전에 하는 것이다(14.2절). 앱은 변환하지 않은 원시 값을 넣는다.
- 조건을 만족하지 않으면 "정의 불가 값"이다. 15, 16번은 항상 정의된다(분모는 `lookback_long_ms`, `lookback_short_ms`).
- "이벤트"(4~8번)는 구간 안의 스텝에 귀속된, **잘리지 않은** 특징용 감김 구간이다. 구간은 그것을 발행한 프레임이 속한
  스텝에 귀속한다(구간이 시작한 스텝이 아니다).
- 9~11번은 잘린 구간과 진행 중인 감김(`G`)을 포함한다. 진행 중인 감김은 통에 넣지 않고 발행할 때마다 더하므로, 끝나서
  통에 들어간 뒤에 두 번 세지 않는다. 10, 11번의 `M`은 구간의 `duration_ms`다(스텝 경계에서 자르지 않는다).
- 감김의 길이를 쓰는 특징(5, 6, 9~11번)은 `ClosureRun.duration_ms`와 `ongoing_ms()`만 쓴다. 둘 다 closure가 진입
  임계(0.5) 이상이던 시간의 합이다. `end_ms − start_ms`는 어디에도 쓰지 않는다. 그래서 깜빡이고 반쯤 뜬 채(0.3~0.5)로
  몇 초 있다가 다시 깜빡이면 이벤트는 하나(4번)이고 길이는 두 깜빡임의 합이다(벡터 `steps_multi_span_runs`).
- 누적을 시작하기 전의 시간은 "데이터 없음"으로 센다(코드의 가정). 조회 구간이 덜 찬 스텝의 `face_missing_ratio_60`은
  1에 가깝고, 유효 시간이 모자란 동안 `perclos_60`과 `blink_rate_60`은 0이다.
- 13번은 `pose_valid`가 아니라 얼굴이 있는 시간 전체의 `pitch_rel`을 쓴다. 가만히 있으면 분산이 반올림 오차로 음수가 될
  수 있어 0으로 자른다(벡터 `steps_gap_over_10s`).
- 입력이 유한하면 17개 값은 모두 유한하다. 분모가 0이 되는 경우(얼굴이 없는 스텝, closure가 무효인 스텝, 이벤트가
  없는 스텝)는 벡터 `steps_zero_division`이 고정한다.

### 11.5 하품과 끄덕임

<!-- check:pipeline -->

| 경로 | 값 | 단위 | 뜻 |
|------|----|------|----|
| `yawn.enter_thr` | 0.4 | | `jaw_rel`이 이 값 **이상**이면 시작 (잠정) |
| `yawn.exit_thr` | 0.25 | | 이 값 **미만**이면 끝 |
| `yawn.count_at_ms` | 2000 | ms | 길이가 이 값에 도달하면 한 번 센다 |
| `yawn.cancel_gap_ms` | 500 | ms | 무효 구간이 이 값을 **넘으면** 취소 |
| `nod.enter_deg` | 10.0 | 도 | `d`가 이 값 **이상**이면 시작 (잠정) |
| `nod.exit_deg` | 5.0 | 도 | 이 값 **미만**이면 끝 |
| `nod.min_duration_ms` | 200 | ms | 끝날 때 길이가 이 값 이상 |
| `nod.max_duration_ms` | 3000 | ms | 이고 이 값 이하이면 센다 |
| `nod.ema_tau_ms` | 10000 | ms | 느린 평균의 시정수 |
| `nod.ema_max_dt_ms` | 200 | ms | 느린 평균 갱신에 쓰는 간격의 상한 |
| `nod.cancel_gap_ms` | 500 | ms | 무효 구간이 이 값을 **넘으면** 취소 |

둘 다 같은 계수기를 쓴다. 시각은 세션 시간이고 길이는 프레임 시각의 차다(보간하지 않는다). `valid`는 `state.face`다.

```
ThresholdEventCounter(enter_thr, exit_thr, min_duration_ms, max_duration_ms, count_at_end, cancel_gap_ms)
상태: active = false;  start = 0;  last_valid = 0;  counted = false

update(t, value, valid) -> Boolean:              // 이 프레임에서 이벤트를 하나 셌으면 true
    active && t - last_valid > cancel_gap_ms:  active = false        // 무효 구간이 길면 취소 (프레임이 끊긴 시간 포함)
    !valid:  return false
    hit = false
    !active:
        value >= enter_thr:  active = true;  start = t;  counted = false
    아니면 value < exit_thr:
        active = false
        count_at_end:  dur = t - start;  hit = (dur >= min_duration_ms && dur <= max_duration_ms)
    아니면 !count_at_end && !counted:
        t - start >= min_duration_ms:  counted = true;  hit = true
    last_valid = t
    return hit

reset():  active = false;  start = 0;  last_valid = 0;  counted = false      // 만들었을 때의 값. 진행 중이던 판정을 버린다
```

- **하품**: `ThresholdEventCounter(yawn.enter_thr, yawn.exit_thr, yawn.count_at_ms, yawn.count_at_ms, count_at_end = false,
  yawn.cancel_gap_ms)`에 `jaw_rel`을 넣는다. 값이 `exit` 이상으로 유지되는 프레임에서 길이가 `yawn.count_at_ms`(2000)
  이상이 되면 한 번 센다. `exit` 미만으로 내려간 프레임에서는 길이가 기준을 넘었어도 세지 않는다.
- **끄덕임**: `ThresholdEventCounter(nod.enter_deg, nod.exit_deg, nod.min_duration_ms, nod.max_duration_ms,
  count_at_end = true, nod.cancel_gap_ms)`에 `d`를 넣는다. 끝나는 프레임에서 길이가 범위 안이면 센다.

```
NodDetector 상태: has_mean = false;  mean = 0.0;  prev_valid_t = 0

update(t, pitch_rel, valid) -> Boolean:
    !valid:  return counter.update(t, 0.0, false)                   // 느린 평균은 갱신하지 않는다
    d = 0.0
    has_mean:
        d = pitch_rel - mean                                        // d를 먼저 계산하고
        dt = min(t - prev_valid_t, nod.ema_max_dt_ms)
        alpha = dt / (nod.ema_tau_ms + dt)                          (Double 나눗셈)
        mean += alpha * (pitch_rel - mean)                          // 그다음 평균을 갱신한다
    아니면:
        has_mean = true;  mean = pitch_rel                          // 첫 유효 표본. d = 0
    prev_valid_t = t
    return counter.update(t, d, true)

reset():  counter.reset();  has_mean = false;  mean = 0.0;  prev_valid_t = 0    // 안의 계수기와 느린 평균을 모두 처음 값으로
```

`pitch_rel`은 턱이 내려갈 때 양수이므로 끄덕임은 고개가 느린 평균보다 10도 이상 **내려간** 경우다. 취소돼도 느린 평균은
그대로다.

설정은 `yawn.cancel_gap_ms < clock.reset_gap_ms`이고 `nod.cancel_gap_ms < clock.reset_gap_ms`여야 한다(지금 값은 500,
500, 10000). 참조 구현은 `pipeline` 블록을 읽을 때 이를 어기면 거부한다(오류 문장에 두 값이 적힌다). 그래야 초기화를
일으키는 끊김(11.6절)이 언제나 진행 중인 판정을 취소하는 끊김이기도 해서, 끊김 앞의 하품이나 끄덕임과 뒤의 것이 계수기의
`reset()` 호출과 무관하게 서로 다른 이벤트다(감김 검출기의 `hold_ms`와 같은 까닭이다. 10.1절). 조건을 지키는 설정에서는
그 호출을 빠뜨려도 출력이 같아서 벡터로 고정되지 않는다. `nod.ema_max_dt_ms`에는 이런 조건이 없다: 느린 평균의 간격
상한은 초기화와 얽히지 않는다(끊김 뒤에 느린 평균을 버리는 것은 이 값과 무관하게 드러나는 동작이고 11.6절의 벡터가
고정한다).

이 절의 경계는 벡터 셋이 고정한다(16.1절). `steps_yawn_edges`: 시작 임계에 닿지 않은 벌림, 무효 구간 800 ms와 정확히
500 ms, 정확히 2000 ms가 된 프레임, 끝 임계 아래로 내려간 프레임. `steps_nod_edges`: 길이 정확히 200 ms와 3000 ms,
끄덕임 도중의 얼굴 소실. `steps_nod_slow_mean`: 느린 평균의 초기값, 빼는 방향, `alpha`의 식과 `dt`의 상한.

### 11.6 10초 초과 끊김 뒤의 초기화

`update`에서 `t − prev_t`가 `clock.reset_gap_ms`(10000)를 넘으면(`>`):

```
restart(t):
    cur_k = t / step_ms + 1                          (정수 나눗셈)
    accum_start = (cur_k - 1) * step_ms              // 새 프레임이 속한 스텝의 시작 경계
    cur = 새 통;  history = [];  has_prev = false;  prev_ongoing = 0
    yawn.reset();  nod.reset()                       // 느린 평균도 버린다
```

- 끊긴 동안의 스텝은 발행하지 않는다. 끊기기 직전에 채우던 스텝도 버린다. 스텝 번호 k는 세션 시간 그대로라 번호가 건너뛴다.
- 끄덕임의 느린 평균은 끊김 뒤 첫 유효 표본에서 다시 시작한다(그 표본이 새 초기값이고 `d`는 0이다). 끊김 앞의 평균을
  남기면 끊김 뒤의 자세가 끄덕임으로 세어진다(벡터 `steps_nod_after_long_gap`).
- `full_lookback`은 새 `accum_start` + 60초부터 다시 true가 된다. 모델 창도 비워진다(7.2절 `push_step`).
- 첫 프레임이 `start_ms`보다 늦게 오면 그 사이에 끝난 스텝은 빈 스텝으로 발행한다. 10초 넘게 늦으면 위 초기화를 따른다.
- 10초 이하의 끊김에서는 그 사이에 끝난 스텝을 모두 발행한다(벡터 `steps_five_minutes`: 프레임 하나가 스텝 8개를 발행).
- 경계는 "초과"다. 간격이 정확히 10000 ms이면 특징용 검출기(7.2절 5번)도 누적기도 초기화하지 않는다: 진행 중이던
  감김은 그 프레임에서 잘린 구간으로 나오고 끊긴 동안의 스텝 열 개가 발행된다. 10001 ms이면 둘 다 초기화한다(벡터
  `steps_gap_exactly_10s`가 두 간격을 한 세션에 담았다).

#### 끊긴 동안 발행되는 스텝 (10초 이하의 끊김)

프레임 하나가 스텝을 여럿 발행할 때 그 스텝들은 모두 **끊기기 직전 마지막 프레임**까지의 정보로 만든다.

- 시간 누적: 마지막 프레임의 상태는 `clock.max_hold_ms`까지만 인정되므로, 끊긴 동안의 스텝에는 새로 더해지는 시간이
  없다. 10초 구간과 60초 구간의 유효 시간과 얼굴 시간은 스텝이 지날수록 줄어든다.
- 진행 중인 감김 `G`는 그 스텝들 모두에서 끊기기 직전 프레임의 `ongoing_ms`다. 끊긴 동안 감김이 어떻게 됐는지는
  알 수 없으므로 늘리지도 0으로 되돌리지도 않는다(코드의 가정). 그래서 눈을 감은 채 끊기면 끊긴 동안의 스텝마다
  `long_closure_count_60`, `max_closure_ms_60`, `max_closure_ms_10`에 그 감김이 들어간다.
- 끊김이 `run_detector.hold_ms`를 넘었으면 끊김 뒤 첫 프레임에서 검출기가 그 감김을 잘린 구간으로 내고, 그 프레임이
  속한 스텝의 통에 들어가며 `G`는 0이 된다. 두 번 세지는 않지만, `max_closure_ms_10`에는 10초가 아니라 끊긴 시간 +
  10초 동안 남는다.
- 10초 구간의 유효 시간이 줄어드는데 감긴 시간은 구간에 남아 있으므로 `perclos_10`이 올라간다. 유효 시간이
  `step_features.perclos_min_valid_ms_10`(2000) 아래로 내려가거나 감긴 시간이 구간을 벗어나야 0이 된다.

벡터 `steps_closure_across_gap`: 12.0초에 눈을 감고 12.8초의 프레임 다음 프레임이 20.6초에 온다. 그 프레임이 스텝
13~20을 발행하고, 그 스텝들의 `max_closure_ms_60`과 `max_closure_ms_10`은 889, `long_closure_count_60`은 1이다. 잘린
구간(889 ms)은 스텝 21에 들어가고 `max_closure_ms_10`은 스텝 30까지 889다(스텝 31에서 0). `perclos_10`은 스텝 13의
0.1에서 스텝 21~22의 0.4167까지 올랐다가 스텝 23에서 0이 된다. 이 동작은 바꾸지 않기로 정하고 문서로 남긴 것이다
(20장 첫머리의 "정비에서 정한 것").

### 11.7 세션 시간 흐름 (`T_b` = 기준선 완성 시각)

| 세션 시간 | 동작 |
|-----------|------|
| 0 ~ `T_b` | 기준선 수집. `calibrating`. 스텝 없음. 감김 규칙은 전역 상수로 계산한 closure로 동작(`provisional = true`). 시선 규칙은 꺼져 있다 |
| `T_b` 이상인 첫 프레임부터 | 개인 기준선으로 closure 계산. 특징용 검출기와 누적 시작. 첫 스텝은 `T_b / 1000 + 1` |
| 스텝 끝 ≥ `T_b` + 60000 | `full_lookback = true`. PERCLOS 졸음 판정이 이 스텝들만 쓴다 |
| `full_lookback` 스텝이 연속 120개 | 스텝이 나올 때마다 모델을 부른다. `T_b = 60000`이면 스텝 239부터 |

---

## 12. 규칙 엔진 (`RuleEngine`)

`update(state)`를 받아들인 프레임마다 한 번(첫 프레임부터), `on_step(step, probs)`를 스텝이 발행될 때마다 부른다.
시각은 원래 `t_ms`다. `t`가 직전 값 이하이면 오류다.

<!-- check:rules -->

| 경로 (`rules.json`) | 값 | 단위 | 뜻 |
|---------------------|----|------|----|
| `clock.step_ms` | 1000 | ms | `pipeline.clock`과 같은 값이어야 한다 |
| `clock.max_hold_ms` | 200 | ms | |
| `clock.reset_gap_ms` | 10000 | ms | |

### 12.1 공통: 유효 시간 누적과 `IntervalWindow`

프레임마다 맨 먼저, 직전 프레임이 유지된 시간(6장)을 더한다.

```
update(state):
    t = state.t_ms;  events = []
    !has_prev:  first_t = t
    아니면:
        gap = t - prev_t;  held = min(gap, clock.max_hold_ms);  held_end = prev_t + held
        prev_face:           face_ms += held;  face_window.add(prev_t, held_end)
        prev_pose_valid:     pose_ms += held;  pose_window.add(prev_t, held_end)
        prev_closure_valid:  closure_valid_ms += held
        gap > clock.reset_gap_ms:  reset_after_gap(state, events)             // 12.7절
    rule_calibrating(state, events)                                           // 12.2절
    analyzable = rule_unable(state, events)                                   // 12.3절
    rule_gaze(state, analyzable, events)                                      // 12.4절
    rule_closure(state, analyzable, events)                                   // 12.5절
    has_prev = true;  prev_t = t
    prev_face = state.face;  prev_pose_valid = state.pose_valid;  prev_closure_valid = state.closure_valid
    return events
```

`face_ms`, `pose_ms`, `closure_valid_ms`는 세션 처음부터의 누적(Long)이다. 이벤트가 시작될 때의 값을 적어 두고 차로
`confidence`를 구한다. 10초 초과 끊김에서도 지우지 않는다. `face_window`는 길이 `unable_to_analyze.window_ms`,
`pose_window`는 길이 `vats.window_ms`의 `IntervalWindow`다. `pose_window`는 `vats`가 시작될 때 "`t_start_ms` 시점의
`pose_ms`"를 구하는 데만 쓴다(12.4절).

```
IntervalWindow(window_ms): 구간 [start, end)의 목록. 시간 순서로, 겹치지 않게 들어온다
    add(start, end):   end <= start 이면 무시. 마지막 구간의 end == start 이면 그 구간의 end를 end로 늘린다. 아니면 새 구간을 덧붙인다
    trim(now):         cut = now - window_ms. end <= cut 인 앞쪽 구간을 버린다. 남은 첫 구간의 start < cut 이면 start = cut
    total_ms():        Σ (end - start)
    since_ms(from):    Σ max(0, end - max(start, from))
    first_start_ms():  첫 구간의 start (비어 있을 때는 부르지 않는다)
    clear()
```

- `trim`은 끝이 창의 시작과 **같은** 구간도 버린다(`end <= cut`). 남기면 길이 0인 구간이 되어 합은 같지만
  `first_start_ms()`가 그 시각을 돌려준다(벡터 `rules_vats_window_edge`: `vats`의 `t_start_ms`가 갈린다. 12.4절).
- `since_ms`는 구간마다 `from` 뒤의 부분만 더한다: `from`보다 앞에서 끝난 구간은 0이고, `from` 뒤에서 시작한 구간은
  그 구간의 길이다(벡터 `rules_vats_confidence_face_lost`).

### 12.2 `calibrating`

```
rule_calibrating(state):
    !has_prev (첫 프레임):
        state.provisional:  calibrating = true;  calib_start = t
                            onset 발행 (t_start_ms = t, duration_ms = 0, confidence = 0.0, provisional = true)
    아니면 calibrating && !state.provisional:
        calibrating = false
        end 발행 (t_start_ms = calib_start, duration_ms = t - calib_start,
                  confidence = ratio(face_ms, t - calib_start), provisional = false)
```

`ratio(a, b)`는 `b <= 0`이면 0.0, 아니면 `a / b`다. 첫 프레임부터 `provisional = false`이면(기준선을 넣어 준 경우) 이벤트가
없다. 기준선이 전역 상수로 대체돼도 end가 나간다(180000 이상인 첫 프레임). 구간은 첫 프레임의 `t_ms`부터 재므로 첫
`t_ms`가 커도 `duration_ms`와 `confidence`가 같다(벡터 `rules_calibrating_large_t0`: 첫 `t_ms`가 1731465123457인 세션.
규칙 엔진과 경고 결정의 시각도 `Long`이어야 한다).

### 12.3 `unable_to_analyze`

<!-- check:rules -->

| 경로 | 값 | 단위 | 뜻 |
|------|----|------|----|
| `unable_to_analyze.window_ms` | 10000 | ms | 얼굴 시간 비율을 재는 창 |
| `unable_to_analyze.min_face_ratio` | 0.5 | 비율 | 비율이 이 값 **미만**이면 분석 불가 |

```
rule_unable(state) -> Boolean:                           // 이 프레임을 시선·감김 규칙이 써도 되는지
    face_window.trim(t)
    t - first_t < unable_to_analyze.window_ms:  return true           // 처음 10초는 판정하지 않는다 (코드의 가정).
                                                                      // first_t = 세션의 첫 프레임, 10초 초과 끊김이
                                                                      // 있었으면 그 뒤의 첫 프레임 (12.7절)
    face_ratio = face_window.total_ms() / unable_to_analyze.window_ms          (Double 나눗셈)
    !unable:
        face_ratio < unable_to_analyze.min_face_ratio:
            unable = true;  unable_start = t;  unable_face0 = face_ms
            onset 발행 (t_start_ms = t, duration_ms = 0, confidence = 0.0)
    아니면 face_ratio >= unable_to_analyze.min_face_ratio:
        unable = false
        end 발행 (t_start_ms = unable_start, duration_ms = t - unable_start,
                  confidence = ratio(face_ms - unable_face0, t - unable_start))
    return !unable
```

- 창에는 직전 프레임까지의 유지 시간만 들어 있다(이 프레임 자신의 얼굴 유무는 다음 프레임에서 더해진다).
- 비율이 정확히 0.5이면 분석 불가가 아니다.
- `first_t`에서 정확히 `window_ms`가 지난 프레임부터 판정한다(`<`). 벡터 `rules_unable_sparse_frames`. `first_t`는 0이
  아니라 첫 프레임의 `t_ms`다(벡터 `rules_calibrating_large_t0`: 0으로 두면 첫 프레임에서 분석 불가가 시작된다).
  첫 `t_ms`가 `window_ms`보다 작은 세션에서도 첫 프레임에서 적어 둔 값이어야 한다(벡터 `rules_unable_small_t0`: 첫
  `t_ms`가 5000이고 처음부터 얼굴이 없는 세션. onset은 `t_ms` 15000의 프레임에서 난다. `first_t`가 0으로 남아 있으면
  `t_ms` 10000의 프레임에서 난다. 12.1절의 `!has_prev` 갈래를 빠뜨리면 첫 `t_ms`가 큰 세션에서는 10초 초과 끊김의
  초기화가 `first_t`를 대신 적어 주므로 드러나지 않고, 첫 `t_ms`가 1인 세션에서도 1 ms 차이라 드러나지 않는다).
- **끊김과 분석 불가**. 데이터 없는 시간은 얼굴 없는 시간이다. 10초 이하의 끊김은 초기화하지 않으므로, 끊기기 전
  10초 동안 얼굴이 계속 있었더라도 끊김이 5200 ms를 넘으면 다음 프레임에서 분석 불가가 시작된다(끊긴 구간 중 얼굴
  시간으로 인정되는 것은 `clock.max_hold_ms` 200 ms뿐이라 창의 얼굴 시간이 `10000 − 끊김 + 200`이다). 경고는 속도로
  억제하지 않으므로 재경고 대기 중이 아니면 경고도 나간다. 벡터 `rules_gap_unable_boundary`(5200 ms는 시작하지 않고
  5201 ms는 시작한다), `rules_gap_exactly_10s`(정확히 10000 ms). 5.2초~10초의 끊김도 판정에서 빼야 하는지는 20장 6번.
- 10초를 **넘는** 끊김 뒤의 첫 프레임에서는 진행 중이던 분석 불가가 끝나고 창과 `first_t`가 다시 시작한다(12.7절).
  그 프레임과 이후 10초는 분석 불가가 아니다.
- 프레임 간격이 400 ms를 넘게 이어지면 프레임마다 200 ms만 인정되어 얼굴 시간 비율이 0.5 아래로 내려간다(벡터
  `rules_unable_sparse_frames`: 500 ms 간격).
- 분석 불가인 동안(`analyzable = false`) 전방 주시 표지는 UNKNOWN이고 규칙용 감김 검출기에는 무효 프레임을 준다.
  onset 프레임부터 적용되고, end 프레임부터 풀린다.
- 스텝 특징과 `drowsy` 판정은 분석 불가의 영향을 받지 않는다.

### 12.4 전방 주시, `gaze_away_long`, `vats`

<!-- check:rules -->

| 경로 | 값 | 단위 | 뜻 |
|------|----|------|----|
| `forward_zone.yaw_abs_max_deg` | 23.0 | 도 | `abs(yaw_rel)`이 이 값을 **넘으면** 영역 밖 (잠정) |
| `forward_zone.pitch_up_max_deg` | 25.0 | 도 | `pitch_rel < −값`이면 영역 밖 (잠정) |
| `forward_zone.pitch_down_max_deg` | 18.0 | 도 | `pitch_rel > 값`이면 영역 밖 (잠정) |
| `forward_zone.eye_aux_enabled` | true | | 눈동자 보조 판정을 쓸지 |
| `forward_zone.gaze_h_abs_max` | 0.25 | | `abs(gaze_h_rel)`이 이 값을 **넘으면** 영역 밖 (잠정) |
| `forward_zone.gaze_v_abs_max` | 0.35 | | `abs(gaze_v_rel)`이 이 값을 **넘으면** 영역 밖 (잠정) |
| `forward_zone.debounce_ms` | 250 | ms | 상태 변화 확정에 필요한 유지 시간 (잠정) |
| `forward_zone.face_loss_credit_ms` | 3000 | ms | 관측이 끊긴 뒤 마지막 표지를 이어받는 시간 (잠정) |
| `gaze_away_long.onset_ms` | 3000 | ms | 연속 영역 밖 시간이 이 값 **이상**이면 시작 |
| `vats.window_ms` | 30000 | ms | 영역 밖 시간을 더하는 창 |
| `vats.cumulative_ms` | 10000 | ms | 창 안의 합이 이 값 **이상**이면 시작 |
| `vats.reset_enabled` | false | | 켜면 전방 주시가 이어질 때 합을 지운다 |
| `vats.reset_forward_ms` | 2000 | ms | `reset_enabled`일 때만 쓴다 |

#### 관측

```
observe(state) -> IN 또는 OUT:
    head_out = abs(state.yaw_rel) > forward_zone.yaw_abs_max_deg
               || state.pitch_rel < -forward_zone.pitch_up_max_deg
               || state.pitch_rel >  forward_zone.pitch_down_max_deg
    head_out:  out_source = head_pose;  return OUT
    forward_zone.eye_aux_enabled
      && (abs(state.gaze_h_rel) > forward_zone.gaze_h_abs_max || abs(state.gaze_v_rel) > forward_zone.gaze_v_abs_max):
        out_source = eye;  return OUT
    return IN
```

`out_source`는 마지막 유효 영역 밖 관측의 출처다. 머리 조건이 성립하면 `head_pose`, 머리는 안인데 눈동자만 벗어나면 `eye`다.

#### 상태 머신 (`ForwardZone`)

- `zone` ∈ {IN, OUT}: 확정된 상태. 처음에는 IN.
- `label` ∈ {IN, OUT, UNKNOWN}: 이 프레임의 표지. 관측이 있으면 관측값, 없으면 이어받은 값 또는 UNKNOWN.
- 후보(`cand`): 지금 상태와 다른 표지가 처음 나타난 프레임. `forward_zone.debounce_ms`(250) 이상 이어지면 확정하고 상태의
  시작 시각을 후보의 시작으로 소급한다.

```
상태: zone = IN;  label = UNKNOWN;  confirmed_in = false
      out_start = 0;  out_pose = 0;  out_source = head_pose;  unknown_ms = 0;  in_start = 0
      has_prev = false;  prev_t = 0;  prev_observed = false;  cover = UNKNOWN;  lost = 0;  acct = 0
      cand = false;  cand_t = 0;  cand_pose = 0
      away = IntervalWindow(vats.window_ms)

update(state, analyzable, pose_ms):                 // pose_ms = 12.1절의 누적값(이 프레임까지)
    t = state.t_ms;  confirmed_in = false
    observed = analyzable && state.pose_valid
    obs = observed 이면 observe(state), 아니면 UNKNOWN

    !has_prev:  in_start = t
    아니면:
        prev_observed:  lost = prev_t + min(t - prev_t, clock.max_hold_ms)      // 마지막 유효 관측이 유지된 끝
        cover_end = lost + forward_zone.face_loss_credit_ms

        // (a) 지난 구간 [prev_t, t)의 정산
        cand:
            cover == UNKNOWN || t > cover_end:                                  // 후보가 확정되지 못하고 끊겼다
                zone == OUT:  unknown_ms += t - cand_t
                cand = false
        아니면 zone == OUT:
            away_end = acct
            cover == OUT:  away_end = max(acct, min(t, cover_end))
            away.add(acct, away_end)
            unknown_ms += t - away_end

    // (b) 이 프레임의 표지
    label = obs
    !observed && analyzable && has_prev && cover != UNKNOWN && t < lost + forward_zone.face_loss_credit_ms:
        label = cover                                                           // 이어받기
    cover = label

    // (c) 디바운스
    label != UNKNOWN:
        label == zone:
            cand:                                                               // 후보를 버린다. 상태는 끊기지 않은 것으로 본다
                zone == OUT:  away.add(cand_t, t)
                cand = false
        아니면:
            !cand:  cand = true;  cand_t = t;  cand_pose = pose_ms
            t - cand_t >= forward_zone.debounce_ms:
                cand = false;  zone = label
                label == OUT:  out_start = cand_t;  out_pose = cand_pose;  unknown_ms = 0;  away.add(cand_t, t)
                아니면:         in_start = cand_t;  confirmed_in = true

    acct = t;  has_prev = true;  prev_t = t;  prev_observed = observed

reset():  away.clear();  zone = IN;  label = UNKNOWN;  confirmed_in = false;  unknown_ms = 0
          has_prev = false;  prev_observed = false;  cover = UNKNOWN;  cand = false
```

- 관측이 없는 경우: `pose_valid = false`(얼굴 없음, 기준선 완성 전, 대체된 기준선)이거나 분석 불가 중.
- 이어받기: 관측이 없는 프레임은 직전 프레임의 표지를 이어받는다. 단 `t < lost + 3000`일 때만이고(`lost`는 마지막 유효
  관측의 시각 + 최대 200 ms), 그 뒤는 UNKNOWN이다. 마지막 표지가 IN이었어도 똑같이 이어받는다(코드의 가정).
  분석 불가 중인 프레임은 이어받지 않고 바로 UNKNOWN이며, 그 뒤로는 새 관측이 올 때까지 이어받을 것이 없다.
- 확정은 프레임이 도착할 때만 판정한다. 이어받은 표지로도 확정된다(고개를 빨리 돌려 250 ms가 차기 전에 얼굴이 사라져도
  영역 밖이 확정된다. 벡터 `rules_gaze_face_lost_in_turn`).
- UNKNOWN 표지는 상태를 바꾸지 않는다. 후보가 걸린 채 UNKNOWN 프레임을 만나면 그다음 프레임의 정산에서 후보를 버린다.
  그 사이의 한 프레임 동안은 `cand`가 true인 채로 남는다. 분석할 수 있는 프레임에서 표지가 UNKNOWN인데 후보가 걸려
  있는 것은 인정 구간이 끝나는 바로 그 시각(`t == lost + 3000`)에 프레임이 온 경우뿐이다(아래 이벤트의 조건이 쓴다).
- 이어받기의 끝은 `t < lost + 3000`이다(`<`). 정확히 `lost + 3000`에 온 프레임은 이어받지 않는다. 이어받은 표지가 IN이면
  그 표지로 영역 안이 확정될 수도 있다(벡터 `rules_gaze_credit_boundary`: 끝나기 50 ms 전의 프레임은 확정하고, 정확히
  끝에 온 프레임은 확정하지 않는다).
- 영역 밖 시간: 확정된 영역 밖 상태에서 표지가 OUT이던 시간(이어받은 시간 포함)이 `away`에 구간으로 들어간다. 프레임이
  끊긴 시간도 얼굴이 사라진 것과 같이 `lost + 3000`까지 영역 밖으로 친다. 영역 밖 상태가 시작된 뒤 세지 않은 시간은
  `unknown_ms`에 더한다. **연속 영역 밖 시간 = `t − out_start − unknown_ms`**.
- 영역 밖 상태에서 영역 안 후보가 걸렸다가 다시 OUT 표지가 오면, 후보 구간도 영역 밖 시간으로 친다(잠깐 앞을 본 것은
  연속을 끊지 않는다). 벡터 `rules_gaze_glance_and_credit_times`의 첫 장면: 200 ms의 정면을 포함한 10000 ms에서 `vats`가
  시작한다.
- 거꾸로 영역 안 상태에서 걸린 영역 밖 후보가 확정되지 못하고 IN 표지로 버려지면 그 후보 구간은 영역 밖 시간이
  **아니다**((c)에서 `away.add`는 `zone == OUT`일 때만 한다). 그래서 250 ms를 채우지 못하는 곁눈질은 아무리 되풀이해도
  `vats`가 되지 않는다(벡터 `rules_gaze_one_frame_glances`: 400 ms마다 프레임 하나에서만 고개를 돌리기를 31초 동안
  되풀이해도 이벤트가 없다).
- `lost`는 마지막 유효 관측의 시각 + `min(다음 프레임까지의 간격, 200)`이다. 다음 프레임이 100 ms 뒤에 왔으면 인정
  구간은 관측 + 3100 ms에 끝난다(같은 벡터의 둘째 장면). 걸린 후보는 `t > cover_end`일 때만 버린다: 정확히
  `cover_end`에 온 프레임에서는 후보가 살아 있고, 그 프레임의 표지가 후보와 같으면 확정된다(같은 벡터의 셋째 장면:
  영역 밖 프레임 하나, 정확히 3200 ms 뒤의 영역 밖 프레임에서 소급해 확정되고 `gaze_away_long`이 난다).
- `pose_valid`가 세션 내내 false이면(대체된 기준선) 표지가 항상 UNKNOWN이라 시선 이벤트가 나지 않는다.

#### 이벤트

```
rule_gaze(state, analyzable):
    zone.update(state, analyzable, pose_ms)

    // gaze_away_long
    gaze_active:
        zone.confirmed_in:  gaze_active = false;  end 발행
    아니면 zone.zone == OUT && (zone.label == OUT || (zone.label == UNKNOWN && analyzable && !zone.cand)):
        t - zone.out_start - zone.unknown_ms >= gaze_away_long.onset_ms:
            gaze_active = true;  gaze_start = zone.out_start;  gaze_pose0 = zone.out_pose;  gaze_source = zone.out_source
            onset 발행

    // vats
    vats.reset_enabled && zone.zone == IN && zone.label == IN && t - zone.in_start >= vats.reset_forward_ms:
        zone.away.clear()
    zone.away.trim(t);  pose_window.trim(t)
    away_ms = zone.away.total_ms()
    !vats_active:
        away_ms >= vats.cumulative_ms:
            vats_active = true;  vats_start = zone.away.first_start_ms()
            vats_pose0 = pose_ms - pose_window.since_ms(vats_start)       // vats_start 시점의 pose_ms
            vats_source = zone.out_source
            onset 발행
    아니면 away_ms < vats.cumulative_ms:
        vats_active = false;  end 발행

gaze_away_long 이벤트:  t_start_ms = gaze_start, duration_ms = t - gaze_start, source = gaze_source,
                        confidence = ratio(pose_ms - gaze_pose0, t - gaze_start)
vats 이벤트:            t_start_ms = vats_start, duration_ms = t - vats_start, source = vats_source,
                        confidence = ratio(pose_ms - vats_pose0, t - vats_start)
```

- `gaze_away_long`의 onset은 영역 밖 상태에서 연속 영역 밖 시간이 3000 ms 이상이 된 뒤 처음 도착한 프레임에서 난다.
  그 프레임의 표지는 OUT이거나, 분석 불가가 아닌 프레임의 UNKNOWN이어야 한다. 뒤쪽은 얼굴 소실 뒤의 인정 구간 안에서
  3000 ms가 찼는데 그 구간에 프레임이 하나도 오지 않은 경우다. UNKNOWN 동안에는 `unknown_ms`가 함께 늘어 연속 시간이
  멈춰 있으므로 일찍 나는 것이 아니고, `duration_ms`는 3000보다 프레임 간격 이상으로 클 수 있다(벡터
  `rules_gaze_credit_window_missed`: 3300). 표지가 IN인 프레임(영역 안 후보가 걸려 있다)과 분석 불가 중인 프레임
  (`analyzable = false`)에서는 나지 않는다.
- 표지가 UNKNOWN인 프레임에 영역 안 후보가 아직 걸려 있으면(`zone.cand`) 그 프레임에서는 판정하지 않는다(코드의 가정).
  후보 구간이 아직 `unknown_ms`에 들어가지 않아 연속 시간이 그만큼 크게 읽히기 때문이다(벡터
  `rules_gaze_credit_boundary`의 첫 장면). 후보는 그다음 프레임의 정산에서 버려진다. 판정이 그 프레임으로 넘어가는 것은
  그 프레임의 표지가 OUT이거나 분석 불가가 아닌 UNKNOWN일 때뿐이다. 그 프레임의 표지가 IN이면(운전자가 앞을 보고
  있다) 새 영역 안 후보가 걸리고 onset은 나지 않는다. 그래서 인정 구간이 끝나는 바로 그 시각에 프레임이 왔는지에 따라
  결과가 갈릴 수 있다. 이 조건을 그대로 둘지는 소유자의 확인을 기다린다(20장 9번에 잰 경계가 있다).
- 영역 안 후보가 인정 구간 뒤의 프레임에서 버려지면(`t > cover_end`) 그 후보 구간(`t − cand_t`)은 `unknown_ms`에 들어가
  연속 영역 밖 시간에서 빠지고, 상태는 영역 밖 그대로 이어진다. 2.0초 이탈, 앞을 본 프레임 하나, 3.3초의 프레임 끊김,
  다시 이탈이면 끊김 뒤 1000 ms에 onset이 난다(벡터 `rules_gaze_candidate_dropped`: `duration_ms` 6300).
- 분석 불가 중인 프레임은 표지가 UNKNOWN이라 그동안 연속 영역 밖 시간이 멈춘다. 영역 밖 상태에서 3초가 되기 전에 분석
  불가가 시작됐고 끝났을 때도 영역 밖이면, 멈췄던 값에서 이어서 센다(벡터 `rules_gaze_unable_pauses_timer`:
  1800 ms에서 멈췄다가 분석 불가가 끝난 뒤 1200 ms에 onset).
- `t_start_ms`는 영역 밖 후보가 처음 나타난 프레임이다. end는 영역 안이 **확정된** 프레임(영역 안 표지가 처음 나타난 뒤
  250 ms 이상 지난 첫 프레임)에서 난다. 정확히 250 ms 뒤의 프레임에서 확정한다(벡터 `rules_gaze_debounce_250ms`).
  얼굴이 사라져 UNKNOWN이 이어지는 동안에는 끝나지 않는다.
- `vats`의 `t_start_ms`는 onset 시점에 30초 창 안에 든 첫 영역 밖 구간의 시작이다(창의 시작에 걸친 구간이면 창의 시작
  시각). 창의 시작에서 정확히 끝나는 구간은 창 안에 없다(12.1절의 `trim`). 벡터 `rules_vats_window_edge`: 영역 밖 구간
  [1000, 2000)과 [21000, ...)이 있고 프레임이 3초 끊긴 뒤의 프레임(32000)에서 합이 9000에서 11000 ms로 건너뛴다. 그
  프레임에서 창의 시작이 2000이므로 `t_start_ms`는 21000, `duration_ms`는 11000이다(첫 구간을 남기면 2000과 30000).
  end 이벤트도 같은 `t_start_ms`와 `source`를 쓴다. 길이가 0인 구간은 `away`에 들어가지 않으므로(12.1절의
  `add`) 영역 밖 상태에서 표지가 UNKNOWN이던 프레임의 시각이 `t_start_ms`가 되는 일은 없다(벡터
  `rules_gaze_unable_pauses_timer`: 27초의 분석 불가 뒤에 시작된 `vats`의 `t_start_ms`는 분석 불가가 끝난 프레임이다).
- `vats`의 `confidence`는 다른 이벤트처럼 `[t_start_ms, 발행 시각]` 전체의 `pose_valid` 시간 비율이다. `vats_start`는
  onset 프레임보다 과거이므로 onset 때 `pose_window`(방금 30초로 자른 것. `vats_start`는 그 안에 있다)에서 `vats_start`
  뒤의 유효 시간을 구해 `pose_ms`에서 빼 둔다. 그 뒤로는 `pose_ms`의 차만 쓰므로 구간이 30초보다 길어도(end는 대개
  그렇다) 잘리지 않는다. 벡터 `rules_gap_exactly_10s`(37초 이어진 `vats`의 end: 25200 / 37000), `rules_vats`.
  `vats_start` 뒤에 `pose_valid`가 끊겼다 이어졌으면 `pose_window`에 구간이 여럿이고, `since_ms`는 구간마다 `vats_start`
  뒤의 부분만 더한다(벡터 `rules_vats_confidence_face_lost`: `vats_start` 앞에서 끝난 구간 하나, 걸친 구간 하나, 뒤에서
  시작한 구간 하나가 있는 세션의 onset `confidence`가 8000 / 10000이다).
- `vats`의 `source`는 onset 프레임의 `zone.out_source`다(벡터 `rules_forward_zone_edges`: `eye`).
- `vats.reset_enabled`는 기본이 꺼짐이다. 켜면 표지가 IN이고 영역 안 상태가 `vats.reset_forward_ms` 이상 이어진
  프레임에서 합을 지운다(정확히 2000 ms인 프레임 포함. 벡터 `rules_vats_reset_enabled`).
- 영역의 경계는 벡터 `rules_forward_zone_edges`가 바로 안쪽과 바깥쪽 값으로 고정한다(yaw ±22.99 / ±23.01도 등).
  눈동자 보조는 절댓값으로 비교한다: 그 벡터가 지나지 않는 두 방향(`gaze_h_rel` 양수, 아래를 보는 `gaze_v_rel` 음수)은
  `rules_forward_zone_eye_edges`가 고정한다. 정확히 경계에 놓인 값은 벡터에 없다(2장).

### 12.5 감김: `microsleep`, `sleep`

<!-- check:rules -->

| 경로 | 값 | 단위 | 뜻 |
|------|----|------|----|
| `closure.microsleep_ms` | 1000 | ms | 진행 중인 감김의 길이(closure가 0.7 이상이던 시간의 합)가 이 값 **이상**이면 `microsleep` |
| `closure.sleep_ms` | 3000 | ms | 이 값 **이상**이면 `sleep` ("3.0초 이상") |

규칙용 `ClosureRunDetector`(`closure.detector`, 진입 0.7 / 이탈 0.5)를 첫 프레임부터 돌린다. 특징용 검출기와는 별개의
인스턴스다.

```
상태: run_tracked = false;  run_start = 0;  run_first = 0;  run_valid0 = 0;  level = NONE      // NONE < MICROSLEEP < SLEEP

rule_closure(state, analyzable):
    valid = state.closure_valid && analyzable
    run = detector.update(t, state.closure, valid)

    closed_frame = valid && state.closure >= closure.detector.enter_thr && detector.state != OPEN
    ongoing = 0;  start = 0
    closed_frame:  ongoing = detector.ongoing_ms();  start = detector.run_start_ms()   // 10.4절. start는 그 감김의 start_ms

    // 기록해 둔 감김이 끝났는가 (같은 프레임에서 새 감김이 시작됐어도)
    ended = (run != null) || detector.state == OPEN
    closed_frame && run_tracked && start != run_start:  ended = true
    ended:
        level != NONE:  end 발행 (그 감김이 도달한 가장 높은 종류 하나);  level = NONE
        run_tracked = false

    // 임계 시간은 유효한 감김 프레임에서만 판정한다
    closed_frame:
        !run_tracked:  run_tracked = true;  run_start = start;  run_first = t;  run_valid0 = closure_valid_ms
        level < MICROSLEEP && ongoing >= closure.microsleep_ms:  level = MICROSLEEP;  microsleep onset 발행
        level < SLEEP      && ongoing >= closure.sleep_ms:       level = SLEEP;       sleep onset 발행

감김 이벤트:  t_start_ms = run_start, duration_ms = t - run_start, source = closure,
              confidence = ratio(closure_valid_ms - run_valid0, t - run_first), provisional = state.provisional
```

- 길이 `ongoing`은 그 감김에서 closure가 진입 임계(0.7) 이상이던 시간의 합이다(2026-11-13 소유자 결정). 이탈과 진입
  사이(0.5~0.7)에 머문 시간은 감김을 끝내지 않지만 길이에도 들어가지 않는다. 잠깐 감았다가 반쯤 뜬 채로 몇 초 있다가
  다시 감아도 그 몇 초가 뒤늦게 더해져 onset이 나지는 않는다(벡터 `rules_closure_dwell_below_1s`). 눈꺼풀이 이탈
  임계 아래로 한 번도 내려가지 않으면 감김이 이어지고 길이가 계속 쌓인다. 이것은 2026-11-14 소유자 결정으로 지금대로
  둔 동작이고 알려진 한계다(10.5절 7번. 잰 사례는 20장 머리의 결정 기록).
- `t_start_ms`는 검출기의 `start_ms`(그 감김에서 closure가 0.7을 처음 위로 지난 시각, 보간값)다. `microsleep`과 `sleep`이
  같은 값을 쓴다. `start`가 기록해 둔 `run_start`와 다르면 앞 감김이 끝나고 새 감김이 시작된 것이다. 앞 감김이
  `min_duration_ms` 미만이라 검출기가 구간을 내지 않고 버린 경우에도 이 비교로 알아본다(벡터
  `rules_closure_after_dropped_run`: 22 ms짜리 감김이 500 ms 넘는 끊김으로 버려진 프레임에서 새 감김이 시작하고, 그
  감김의 `t_start_ms`와 `confidence`는 새 감김의 것이다).
- `duration_ms`는 이벤트 공통 규칙대로 발행 시각 − `t_start_ms`다(코드의 가정). 감김 도중에 반쯤 뜬 시간이 있었으면
  onset을 낸 길이(`ongoing`)보다 길다. `vats`의 `duration_ms`가 영역 밖 시간의 합보다 긴 것과 같다(벡터
  `rules_closure_dwell_reaches_1s`: 길이가 정확히 1000 ms가 된 프레임의 onset이 `duration_ms` 5966).
- 한 감김은 `microsleep` onset, (3.0초에) `sleep` onset, 끝날 때 도달한 가장 높은 종류의 end **하나**를 낸다.
  `sleep`까지 간 감김에는 `microsleep` end가 없다.
- end의 `duration_ms`는 감김이 끝난 것을 알아챈 프레임(이탈 임계 아래로 내려간 프레임, 또는 끊겨서 닫힌 프레임)까지의
  길이다. 검출기의 `duration_ms`보다 길다.
- 끊긴 구간(500 ms 이하)은 앞뒤 표본이 모두 0.7 이상이면 길이에 들어가지만 그 안(무효 프레임)에서는 onset이 나지
  않는다. 다음 감긴 프레임에서 난다. 길이에는 들어가도 `confidence`의 분자에는 들어가지 않는다: 유효 시간은
  `closure_valid`인 프레임이 유지된 시간만 센다(벡터 `rules_microsleep_confidence_face_lost`: 얼굴 없는 프레임 셋이 낀
  감김의 onset `confidence`가 700 / 1000, end가 1300 / 1600이다).
- 한 프레임에 `ongoing`이 늘어나는 양은 500 ms 이하이므로, 지금 상수(1000과 3000)에서는 `microsleep`과 `sleep`의 onset이
  같은 프레임에 나지 않는다. 의사 코드의 순서(`microsleep` 먼저)는 그대로 둔다.
- 기준선 완성 전에는 전역 상수로 계산한 closure로 판정하고 이벤트에 `provisional = true`가 붙는다. 검출기는 기준선이
  완성될 때 초기화하지 않으므로, 완성 시점에 걸친 감김은 onset이 `provisional = true`, end가 `false`일 수 있다.
- 벡터: `rules_closure_900ms`, `_1100ms`, `_2500ms`, `_3000ms`, `_6000ms`(토막이 하나인 감김),
  `rules_closure_dwell_below_1s`, `rules_closure_dwell_reaches_1s`, `rules_closure_three_spans`(토막이 여럿인 감김).

### 12.6 `drowsy` (`on_step`)

<!-- check:rules -->

| 경로 | 값 | 단위 | 뜻 |
|------|----|------|----|
| `drowsiness.source` | `perclos` | | `model`, `perclos`, `either` 중 하나. 학습한 모델이 기준을 통과하기 전의 기본값은 `perclos` |
| `drowsiness.model.onset_p` | 0.6 | | `p_drowsy`가 이 값 **이상**인 스텝이 |
| `drowsiness.model.onset_steps` | 10 | 스텝 | 이만큼 연속이면 시작 |
| `drowsiness.model.end_p` | 0.4 | | `p_drowsy`가 이 값 **이하**인 스텝이 |
| `drowsiness.model.end_steps` | 10 | 스텝 | 이만큼 연속이면 종료 |
| `drowsiness.model.refractory_ms` | 60000 | ms | 종료 후 이 시간 동안은 다시 시작하지 않는다 |
| `drowsiness.perclos.onset_ratio` | 0.15 | | `perclos_60`이 이 값 **이상**인 스텝이 (잠정) |
| `drowsiness.perclos.onset_steps` | 10 | 스텝 | 이만큼 연속이면 시작 |
| `drowsiness.perclos.end_ratio` | 0.1 | | `perclos_60`이 이 값 **미만**인 스텝이 (잠정) |
| `drowsiness.perclos.end_steps` | 10 | 스텝 | 이만큼 연속이면 종료 |
| `drowsiness.perclos.refractory_ms` | 0 | ms | 없음 |

```
StepHysteresis(onset_steps, end_steps, refractory_ms)
상태: active = false;  start_ms = 0;  has_k = false;  last_k = 0;  onset_run = 0;  end_run = 0
      run_start = 0;  has_ended = false;  last_end = 0

update(k, t_end_ms, has_value, onset_hit, end_hit):
    has_k && k != last_k + 1:  onset_run = 0;  end_run = 0           // 스텝 번호가 이어지지 않는다
    has_k = true;  last_k = k
    !has_value:  onset_run = 0;  end_run = 0;  return                // 값이 없는 스텝. active는 그대로

    onset_hit:  (onset_run == 0 이면 run_start = t_end_ms);  onset_run += 1
    아니면:     onset_run = 0
    end_hit:    end_run += 1
    아니면:     end_run = 0

    !active:
        onset_run >= onset_steps && (!has_ended || t_end_ms - last_end >= refractory_ms):
            active = true;  start_ms = run_start
    아니면 end_run >= end_steps:
        active = false;  has_ended = true;  last_end = t_end_ms

reset():  모든 상태를 만들었을 때의 값으로 (active = false, start_ms = 0, has_k = false, last_k = 0,
          onset_run = 0, end_run = 0, run_start = 0, has_ended = false, last_end = 0)       // 12.7절이 부른다
```

```
on_step(step, probs) -> 이벤트 목록:
    use_model   = drowsiness.source ∈ {model, either}
    use_perclos = drowsiness.source ∈ {perclos, either}
    has_p = (probs != null);  p = has_p 이면 probs[2], 아니면 0.0
    perclos = step.values[0]

    use_model:    model_rule.update(step.k, step.t_end_ms, has_p,
                                    p >= drowsiness.model.onset_p, p <= drowsiness.model.end_p)
    use_perclos:  perclos_rule.update(step.k, step.t_end_ms, step.full_lookback,
                                      perclos >= drowsiness.perclos.onset_ratio, perclos < drowsiness.perclos.end_ratio)
    model_on = use_model && model_rule.active;  perclos_on = use_perclos && perclos_rule.active

    !drowsy_active && (model_on || perclos_on):
        drowsy_active = true
        model_on:  drowsy_source = model;    drowsy_start = model_rule.start_ms
        아니면:    drowsy_source = perclos;  drowsy_start = perclos_rule.start_ms
        return [onset]
    drowsy_active && !(model_on || perclos_on):
        drowsy_active = false
        return [end]
    return []

drowsy 이벤트:  t_start_ms = drowsy_start, duration_ms = step.t_end_ms - drowsy_start, source = drowsy_source,
                provisional = false,
                confidence = drowsy_source == perclos 이면 1.0 - step.values[15], 아니면 p
```

- `t_start_ms`는 시작 조건을 만족한 연속 구간의 **첫 스텝의 `t_end_ms`**다. onset은 열 번째 스텝에서 나가므로 재시작 금지에
  걸리지 않았다면 onset의 `duration_ms`는 9000이다.
- 발행 시각은 그 스텝의 `t_end_ms`이고 그 스텝을 발행시킨 프레임의 `t_ms`가 아니다. 실제 프레임은 스텝 경계에 정확히
  오지 않으므로 둘은 대개 다르다(벡터 `rules_drowsy_off_step_boundary`: 스텝 69를 세션 69050의 프레임이 발행하고
  onset의 `duration_ms`는 9000이다. 프레임 시각에서 재면 9050이다. 다른 벡터에서는 두 시각의 차가 1 ms 이하라 이
  차이가 드러나지 않는다). 그 프레임에서 바로 나가는 경고도 같은 값이다(13장).
- 모델 쪽: 확률이 없는 스텝(`probs = null`)은 값이 없는 스텝이다. 창이 덜 찬 동안의 스텝과 유한하지 않은 값 때문에
  추론을 건너뛴 스텝이 여기에 든다. 한 프레임에서 스텝이 여럿 나와도 스텝마다 확률을 받으므로(7.2절) 10초 이하의
  프레임 끊김은 연속 수를 끊지 않는다(벡터 `rules_drowsy_model_stall`).
- PERCLOS 쪽: `full_lookback`인 스텝만 쓴다(코드의 가정). 조회 구간이 덜 찬 스텝은 값이 없는 스텝이다.
- 재시작 금지 시간 동안에도 연속 수는 센다. 금지 시간이 끝난 첫 스텝에서 연속 수가 이미 찼으면 그 스텝에서 시작하고
  `t_start_ms`는 그 연속 구간의 첫 스텝 끝(금지 시간 안일 수 있다)이다.
- `either`: 둘 중 하나라도 판정 중이면 `drowsy` 상태이고 둘 다 끝나야 끝난다. onset의 `source`와 `t_start_ms`는 상태를 연
  쪽의 것이고(같은 스텝에서 함께 열리면 `model`), end도 같은 값을 쓴다. `model`이 연 상태가 확률이 없는 스텝에서 끝나면
  `confidence`는 0.0이다.
- 모델 출력은 `Float`다. 임계값과는 `Double`로 바꿔 비교한다.
- 10초 초과 끊김 뒤의 첫 프레임(`update`)에서 진행 중이던 `drowsy`가 끝나고 두 판정기가 `reset()`된다(12.7절).
  끊김 뒤에는 시작 조건을 처음부터 다시 채워야 새 `drowsy`가 시작되고 재시작 금지 시간도 이어지지 않는다.
- 등호 자리: `perclos_60`이 정확히 0.15이면 시작 조건이고(`>=`) 정확히 0.10이면 종료 조건이 아니다(`<`). 벡터
  `rules_drowsy_perclos_exact`가 정수 ms의 몫으로 이 두 값을 만든다. `p_drowsy`가 정확히 0.6이면 시작 조건이다
  (벡터 `rules_drowsy_model_stall`). 종료 쪽의 등호(`p_drowsy == 0.4`)는 벡터로 고정하지 않는다: 0.4는 `Float`로
  정확히 적을 수 없어서 스텁이 `Float`를 돌려주는 구현에서는 0.4보다 큰 값이 된다.
- 벡터: `rules_drowsy_perclos`, `rules_drowsy_perclos_exact`, `rules_drowsy_model_stub`, `rules_drowsy_model_stall`,
  `rules_gap_over_10s_drowsy`, `e2e_250s`, 그리고 `either`의 `rules_drowsy_either`(모델이 연 상태가 끝난 다음 스텝에
  PERCLOS가 상태를 연다: 둘째 onset의 `source`는 `perclos`이고 `confidence`는 `1 − face_missing_ratio_60`이다. 이 벡터와
  다른 벡터에서는 그 스텝의 #15와 #16이 같은 값이라, #15 대신 #16을 읽는 구현도 통과한다. 16.1절)와
  끊김 뒤의 모델 판정기를 보는 `rules_gap_over_10s_drowsy_model`(12.7절).

### 12.7 10초 초과 끊김 (`reset_after_gap`)

프레임 간격이 `clock.reset_gap_ms`를 넘으면 그 프레임에서 규칙 엔진이 진행 중이던 상태를 끝내고 세션을 시작할 때처럼
다시 시작한다(2026-11-13 소유자 결정). `update`에서 `gap > clock.reset_gap_ms`이면, 유효 시간 누적 직후에:

```
reset_after_gap(state, events):                          // t = state.t_ms
    // 졸음
    drowsy_active:
        drowsy_active = false
        drowsy end 발행 (t_start_ms = drowsy_start, duration_ms = t - drowsy_start, source = drowsy_source,
                         confidence = 0.0, provisional = false)
    model_rule.reset();  perclos_rule.reset()            // 12.6절

    // 분석 불가
    unable:
        unable = false
        unable_to_analyze end 발행 (t_start_ms = unable_start, duration_ms = t - unable_start,
                                    confidence = ratio(face_ms - unable_face0, t - unable_start))
    face_window.clear();  first_t = t                    // "처음 10초"를 이 프레임부터 다시 센다 (12.3절)

    // 시선
    gaze_active:      gaze_active = false;  gaze_away_long end 발행
    zone.reset()                                         // 전방 주시 상태와 영역 밖 누적을 지운다

    // 감김
    level != NONE:    감김 end 발행 (도달한 가장 높은 종류);  level = NONE
    detector.reset();  run_tracked = false
```

그 뒤 그 프레임을 평소 순서로 판정한다(12.1절의 `rule_calibrating`부터).

- end의 순서는 `drowsy`, `unable_to_analyze`, `gaze_away_long`, `microsleep` 또는 `sleep`이다. 발행 시각은 모두 그
  프레임의 `t_ms`이고 `duration_ms`는 끊긴 시간을 포함한다. `t_start_ms`와 `source`는 짝이 되는 onset의 것이다.
- `drowsy` end의 발행 시각은 스텝 끝이 아니라 그 프레임의 `t_ms`이고, 값을 실어 줄 스텝이 없으므로 `confidence`는
  0.0이다(코드의 가정). `provisional`은 다른 `drowsy` 이벤트처럼 false다.
- `vats`가 진행 중이었으면 누적이 지워졌으므로 같은 프레임의 12.4절 판정에서 end가 난다(위의 end들 뒤이고, 같은
  프레임에 `calibrating` end가 있으면 그 뒤다).
- 그 프레임부터 `unable_to_analyze.window_ms` 동안은 분석 불가가 아니므로 시선과 감김 규칙이 바로 돈다. 끊김 뒤 첫
  프레임부터 눈을 감고 있으면 그 프레임이 감김의 시작이고 1000 ms 뒤에 `microsleep`, 3000 ms 뒤에 `sleep`이 난다
  (벡터 `rules_gap_over_10s`).
- 남기는 것은 `calibrating` 상태(기준선은 유지된다)와 유효 시간 누적값(`face_ms`, `pose_ms`, `closure_valid_ms`)뿐이다.
  누적값은 차로만 쓰므로 끊김 전의 값이 끊김 뒤에 시작하는 이벤트의 `confidence`에 들어가지 않는다.
- `on_step`은 `update`보다 먼저 불리므로, 끊김 뒤의 첫 프레임이 발행시킨 스텝이 있다면 그 스텝은 초기화 전의 판정기로
  처리된다. `DriverPipeline`에서는 그런 스텝이 값을 갖지 못하므로(7.2절) 초기화 직전에 `drowsy`가 열리거나 닫히는
  일은 없다.
- 미뤄 둔 경고(13장)는 이 end들로 지워진다. 끊김 전에 속도에 막혀 있던 `drowsy` 경고가 끊김 뒤에 나가지 않는다
  (벡터 `rules_gap_over_10s_drowsy`).
- `zone.reset()`은 걸려 있던 전방 주시 후보도 버린다. 고개를 돌린 프레임 하나(영역 밖 후보)를 본 뒤 10초 넘게 끊기고
  끊김 뒤에도 고개를 돌리고 있으면, 영역 밖은 끊김 뒤의 첫 프레임부터 새로 센다(벡터 `rules_gap_over_10s_candidate`:
  그 프레임에서 정확히 3000 ms 뒤에 onset).
- 모델 판정기의 재시작 금지 기억(`has_ended`, `last_end`)도 지워진다. 끊김 뒤에 모델 창이 다시 차려면 179초가 걸리므로
  (11.6절, 7.2절) 기본 금지 시간 60초로는 차이가 드러나지 않는다. 벡터 `rules_gap_over_10s_drowsy_model`은
  `drowsiness.model.refractory_ms`를 600000으로 덮어써서, 끊김 전에 끝난 상태의 금지 시간 안에 새 onset이 나는 것을
  고정한다.
- 마지막 줄의 `detector.reset()`과 `run_tracked = false`는 설정 검증을 통과한 상수에서는 이벤트를 바꾸지 않는다.
  `closure.detector.hold_ms`(500)가 `clock.reset_gap_ms`보다 작아야 하므로(10.1절) 끊김 앞의 감김은 그 프레임의
  `detector.update`에서 어차피 잘려 끝나고, 그 프레임에서 시작한 감김은 시작 시각이 달라 `rule_closure`가 새 감김으로
  알아본다. 그래서 이 두 줄은 벡터로도 단위 테스트로도 고정되지 않는다. 의사 코드대로 둔다.
- 간격이 정확히 10000 ms이면 초기화하지 않는다(벡터 `rules_gap_exactly_10s`). 10초 이하의 끊김은 얼굴 없는 시간으로
  세므로 5200 ms를 넘으면 `unable_to_analyze`가 시작된다(12.3절).
- 벡터: `rules_gap_over_10s`(`gaze_away_long`, `vats`, `sleep`이 진행 중), `rules_gap_over_10s_drowsy`(`drowsy`와
  `gaze_away_long`), `rules_unable_sparse_frames`(`unable_to_analyze`가 진행 중이고, 끊김 뒤 정확히 10000 ms에 다시
  판정한다).

### 12.8 이벤트 표

<!-- check:events -->

| type | phase | onset 조건 | `t_start_ms` | end 조건 | source | severity | cooldown_ms |
|------|-------|------------|--------------|----------|--------|----------|-------------|
| `calibrating` | `onset`, `end` | 첫 프레임이 `provisional` | 첫 프레임 | `provisional = false`인 첫 프레임 | `face` | 0 | 0 |
| `unable_to_analyze` | `onset`, `end` | 최근 10초의 얼굴 시간 비율 < 0.5 (세션의 처음 10초와 10초 초과 끊김 뒤의 10초 제외) | 비율이 처음 0.5 아래로 내려간 프레임 | 비율 ≥ 0.5인 첫 프레임, 또는 10초 초과 끊김 | `face` | 1 | 30000 |
| `gaze_away_long` | `onset`, `end` | 연속 영역 밖 시간 ≥ 3000인 프레임. 표지가 OUT이거나, 분석 불가가 아니고 걸린 후보가 없는 UNKNOWN (12.4절) | 영역 밖이 처음 나타난 프레임 | 영역 안 확정, 또는 10초 초과 끊김 | `head_pose`, `eye` | 2 | 10000 |
| `vats` | `onset`, `end` | 최근 30초의 영역 밖 합 ≥ 10000 | 그 30초 안의 첫 영역 밖 구간 시작 | 합 < 10000인 첫 프레임(10초 초과 끊김 뒤의 첫 프레임 포함: 합이 지워진다) | `head_pose`, `eye` | 2 | 30000 |
| `drowsy` | `onset`, `end` | 12.6절 | 연속 구간의 첫 스텝 끝 | 12.6절, 또는 10초 초과 끊김(발행 시각은 그 프레임의 `t_ms`, `confidence` 0.0) | `model`, `perclos` | 2 | 300000 |
| `microsleep` | `onset`, `end` | 감긴 프레임에서 진행 중인 감김의 길이(0.7 이상이던 시간의 합) ≥ 1000 | closure가 0.7을 처음 위로 지난 시각 | 감김 종료 또는 10초 초과 끊김(`sleep`에 이르지 않은 경우만) | `closure` | 3 | 5000 |
| `sleep` | `onset`, `end` | 같은 감김의 길이가 ≥ 3000 | `microsleep`과 같음 | 감김 종료, 또는 10초 초과 끊김 | `closure` | 3 | 0 |

`cooldown_ms`는 경고 결정(13장)이 쓰는 값이고 상태 이벤트의 발행에는 영향이 없다. `sleep`은 대기 없이 3초마다 되풀이한다.

### 12.9 `confidence`

"그 이벤트 구간에서 해당 신호가 유효했던 시간 / 구간 길이"다. 유효 시간은 유지 규칙으로 센다. 구간 길이가 0이면 0.0이다.

| type | 분자 | 분모 |
|------|------|------|
| `calibrating` | `[t_start_ms, 발행 시각]`에서 얼굴이 있던 시간 | 발행 시각 − `t_start_ms` |
| `unable_to_analyze` | 같음 | 같음 |
| `gaze_away_long` | `[t_start_ms, 발행 시각]`에서 `pose_valid`였던 시간 | 같음 |
| `vats` | 같음(`[t_start_ms, 발행 시각]` 전체. 30초로 자르지 않는다) | 같음 |
| `microsleep`, `sleep` | `[감김을 처음 본 프레임, 발행 시각]`에서 `closure_valid`였던 시간 | 발행 시각 − 감김을 처음 본 프레임 |
| `drowsy` (`model`) | 발행한 스텝의 `p_drowsy` (확률이 없는 스텝이면 0.0) | |
| `drowsy` (`perclos`) | `1 − face_missing_ratio_60` (발행한 스텝) | |
| `drowsy` (10초 초과 끊김이 끝낸 end) | 0.0 | |

`calibrating`과 `unable_to_analyze`의 onset은 구간 길이가 0이라 `confidence = 0.0`이다. 감김의 분모는 `t_start_ms`(보간값)가
아니라 감긴 것을 처음 본 프레임의 시각부터 잰다. 1보다 작은 값을 고정하는 벡터: `rules_vats_confidence_face_lost`와
`rules_vats_window_edge`(`vats` onset), `rules_microsleep_confidence_face_lost`(`microsleep` onset과 end),
`rules_gaze_candidate_dropped`와 `rules_gaze_credit_window_missed`(`gaze_away_long` onset), `rules_gap_over_10s`(`sleep`
end), `rules_unable_small_t0`와 `rules_unable_sparse_frames`(`unable_to_analyze` end), `rules_calibrating_provisional`
(`calibrating` end).

### 12.10 표시용 상태

`gaze_zone`: 마지막 프레임의 표지가 UNKNOWN이면 `unknown`, 아니면 확정된 상태(`in` 또는 `out`)다. 판정에는 쓰지 않는다.

---

## 13. 경고 결정 (`AlertPolicy`)

상태 이벤트와 차량 속도를 받아 **실제로 경고할 이벤트**를 돌려준다. 프레임마다 `decide(events, speed_kmh, t)`를 한 번
부른다. 이벤트가 없어도 부른다(미뤄 둔 경고와 되풀이 경고의 시각을 재야 한다).

<!-- check:events -->

| type | severity | cooldown_ms | repeat_ms | min_speed_kmh |
|------|----------|-------------|-----------|---------------|
| `calibrating` | 0 | 0 | 0 | 0.0 |
| `unable_to_analyze` | 1 | 30000 | 0 | 0.0 |
| `gaze_away_long` | 2 | 10000 | 0 | 20.0 |
| `vats` | 2 | 30000 | 0 | 20.0 |
| `drowsy` | 2 | 300000 | 0 | 50.0 |
| `microsleep` | 3 | 5000 | 0 | 20.0 |
| `sleep` | 3 | 0 | 3000 | 20.0 |

값은 `events.<type>`에 있다. 0은 "없음"이다(대기 없음, 되풀이 없음, 속도 억제 없음).

<!-- check:alert-priority -->

| 순위 | type |
|------|------|
| 1 | `sleep` |
| 2 | `microsleep` |
| 3 | `gaze_away_long` |
| 4 | `vats` |
| 5 | `drowsy` |
| 6 | `unable_to_analyze` |
| 7 | `calibrating` |

```
상태: last_alert[type]    (마지막 경고 시각)
      pending[type]       (막혀서 미뤄 둔 onset 이벤트. 되풀이하지 않는 종류만)
      repeat_onset[type]  (이어지는 상태의 onset 이벤트. 되풀이하는 종류만)
      repeat_alert[type]  (그 상태에서 낸 마지막 경고 시각)

decide(events, speed_kmh, t) -> 경고 목록:
    passed = []

    // 1. 이 호출의 이벤트
    events의 각 event에 대해 (순서대로):
        event.phase == end:
            forget(event.type)
            event.type == sleep:  forget(microsleep)
            continue
        event.type == calibrating:  continue
        forget(event.type)                                   // onset: 그 종류의 지난 기록을 지우고 시작한다
        event.type == sleep:  forget(microsleep)
        events.<type>.repeat_ms > 0:  repeat_onset[type] = event
        allowed(type, speed_kmh, t):  mark(type, t);  passed += event
        아니면 events.<type>.repeat_ms == 0:  pending[type] = event

    // 2. 미뤄 둔 경고
    alert_priority의 각 type에 대해:
        type이 pending에 없다:  continue
        allowed(type, speed_kmh, t):
            mark(type, t)
            passed += pending[type]의 사본 (duration_ms = t - t_start_ms 로만 고친다)
            pending에서 type을 지운다

    // 3. 되풀이
    alert_priority의 각 type에 대해:
        type이 repeat_onset에 없다:  continue
        type이 repeat_alert에 있고 t - repeat_alert[type] < events.<type>.repeat_ms:  continue
        allowed(type, speed_kmh, t):
            mark(type, t)
            passed += repeat_onset[type]의 사본 (duration_ms = t - t_start_ms 로만 고친다)

    return alert_priority 순서로 passed를 정렬한 것 (같은 종류 안에서는 passed에 들어간 순서)

allowed(type, speed_kmh, t):
    type이 last_alert에 있고 t - last_alert[type] < events.<type>.cooldown_ms:  return false
    speed_kmh != null && events.<type>.min_speed_kmh > 0.0 && speed_kmh < events.<type>.min_speed_kmh:  return false
    return true

mark(type, t):
    last_alert[type] = t
    type이 repeat_onset에 있으면 repeat_alert[type] = t

forget(type):
    pending, repeat_onset, repeat_alert에서 type을 지운다          // last_alert는 남긴다
```

- 경고가 되는 것은 onset뿐이다. end는 상태가 끝났다는 것만 알린다. `calibrating`은 경고하지 않는다.
- onset이 온 호출에서 바로 나가는 경고는 그 onset 이벤트 그대로다(`passed += event`). `duration_ms`를 `t - t_start_ms`로
  고치는 것은 미뤘다 낸 경고와 되풀이 경고뿐이다. 프레임 이벤트는 발행 시각이 `t`라 구별되지 않지만 `drowsy`는 발행
  시각이 스텝 끝이라 달라진다(벡터 `rules_drowsy_off_step_boundary`: 바로 낸 경고의 `duration_ms`가 9050이 아니라
  9000이다).
- 재경고 대기는 실제로 낸 경고부터 센다(바로 낸 것, 미뤘다 낸 것, 되풀이 모두). 억제되거나 대기에 걸린 onset은 대기를
  새로 시작하지 않는다. 마지막 경고에서 정확히 `cooldown_ms`가 지난 프레임은 걸리지 않는다(`<`).
- 속도 억제는 경고에만 적용한다. 상태 이벤트(`events`)는 속도와 무관하게 그대로 나온다. 속도가 null이면 억제하지 않는다.
  속도가 기준과 같으면(`20.0`, `50.0`) 억제하지 않는다(`<`).
- **미뤄 둔 경고**(2026-11-13 소유자 결정. `calibrating`을 뺀 모든 종류): onset이 온 호출에서 대기나 속도에 막혀 경고하지
  못하면 그 onset을 종류별로 하나 미뤄 둔다. 이후의 호출마다, 그 호출의 이벤트를 모두 처리한 뒤에, 미뤄 둔 종류가 더는
  막히지 않으면 한 번 경고하고 지운다. 이 경고는 미뤄 둔 onset의 사본이고 `duration_ms`만 지금 시각 기준이다
  (`confidence`와 `provisional`은 onset 때의 값). 막힌 그 호출에서는 같은 조건으로 다시 보는 것이라 나가지 않는다.
- 상태가 끝나면 미뤄 둔 경고는 경고 없이 지워진다. 그 종류의 end가 온 호출에서 지우고(막힘이 풀린 바로 그 호출에 end가
  함께 와도 경고하지 않는다), `microsleep`은 `sleep`의 onset이나 end가 와도 지운다(한 감김은 도달한 가장 높은 종류의 end
  하나만 내므로 `sleep`까지 간 감김의 `microsleep`에는 자기 end가 오지 않는다). 다른 종류의 end는 지우지 않는다(벡터
  `rules_alert_pending_survives_end`: 미뤄 둔 `microsleep` 경고가 `gaze_away_long`의 end 뒤에도 남아 있다가 속도가
  오른 프레임에서 나간다). 다른 종류의 onset도 지우지 않는다(벡터 `rules_alert_pending_survives_onset`: 미뤄 둔
  `microsleep` 경고가 `gaze_away_long`의 onset 뒤에도 남아 있다가 속도가 오른 프레임에서 `gaze_away_long` 경고와 함께
  우선순위 순으로 나간다). 규칙 엔진은 상태를 끝낼 때 반드시 그
  종류의 end를 내고 10초 초과 끊김의 초기화도 end를 내므로(12.7절), 미뤄 둔 경고가 상태보다 오래 남지 않는다.
- 한 onset은 바로든 미뤄서든 한 번만 경고한다. 상태가 오래 이어져도 주기적으로 다시 경고하지 않는다(되풀이는 `sleep`뿐).
- 되풀이(`sleep`): 상태가 이어지는 동안 마지막 경고 뒤 `events.sleep.repeat_ms`(3000) 이상 지난 첫 프레임마다 다시
  경고한다. 되풀이 경고는 onset의 사본이고 `duration_ms`만 지금 시각 기준이다(`confidence`는 onset 때의 값).
  onset이 막혔으면 상태가 이어지는 동안 막힘이 풀린 첫 호출에서 경고하고 거기서부터 주기를 센다(미뤄 둔 경고와 결과가
  같아서 되풀이하는 종류는 `pending`에 넣지 않는다).
- 한 번의 호출에서 통과한 경고(바로 낸 것, 미뤘다 낸 것, 되풀이)는 모두 돌려준다(낮은 순위를 버리지 않는다). 앱은 첫
  원소를 대표로 쓰면 된다. 순서는 `alert_priority`이고 이벤트가 난 순서가 아니다(벡터 `rules_alert_priority`).
- 미뤄 둔 종류의 onset이 end 없이 다시 오면(규칙 엔진은 그렇게 내지 않는다) 새 onset으로 바꾼다(코드의 가정).
- 벡터: `rules_speed_gates`, `rules_cooldown`, `rules_sleep_repeat`, `rules_closure_6000ms`, 그리고 미뤄 둔 경고의
  `rules_alert_deferred_cooldown`(대기가 풀린 프레임, 대기가 풀리기 전에 끝난 상태), `rules_alert_blocked_microsleep_sleep`
  (`sleep`으로 넘어간 `microsleep`, 정확히 20.0 km/h), `rules_drowsy_perclos_exact`(속도에 막힌 `drowsy`, 정확히
  50.0 km/h), `rules_gap_over_10s`(속도에 막힌 `gaze_away_long`), `rules_unable_sparse_frames`(대기에 걸린
  `unable_to_analyze`), `rules_gap_over_10s_drowsy`(끊김으로 끝난 상태).

---

## 14. ONNX 모델

### 14.1 입출력 계약

<!-- check:onnx -->

| 항목 | 값 |
|------|----|
| `opset` | 17 |
| `ir_version` | 8 |
| `input_name` | `features` |
| `input_shape` | 1, 120, 17 |
| `output_name` | `probs` |
| `output_shape` | 1, 3 |
| `max_file_bytes` | 1000000 |

<!-- check:feature-spec -->

| `feature_spec` 경로 | 값 | 뜻 |
|---------------------|----|----|
| `step_sec` | 1 | 스텝 길이(초) |
| `window_steps` | 120 | 창 길이(스텝) |
| `class_order` | `alert`, `low_vigilant`, `drowsy` | 출력 순서 |
| `standardization.location` | `in_graph` | 표준화는 그래프 안에 있다 |
| `standardization.z_clip` | 5.0 | 표준화 뒤 ±이 값으로 자른다 |
| `standardization.std_floor` | 0.001 | 표준편차 하한 |

- 파일: `export/drowsiness_v{N}.onnx`. 단일 파일(외부 데이터 없음), 1 MB 미만, 배치 1 고정. 연산자 도메인은 기본
  도메인(`ai.onnx`) 하나다.
- 입력 `features`: `float32 [1, 120, 17]`. `features[0][i][j]`는 창의 i번째 스텝의 j번째 특징(11.4절의 #)이다.
  **`features[0][0]`이 가장 오래된 스텝, `features[0][119]`가 가장 최근 스텝**이다.
- 입력은 `StepFeatures.values` 그대로다(`Double`을 `Float`로 바꾸기만 한다). **표준화하지 않고, log1p도 하지 않는다.**
  앱이 한 번 더 표준화해도 오류가 나지 않으므로 주의한다.
- 출력 `probs`: `float32 [1, 3]`. `(alert, low_vigilant, drowsy)` 확률이고 합은 1이다.
- 호출 시점은 7.2절의 6단계다. 스텝이 발행될 때마다, 그 스텝까지 연속한 `full_lookback` 스텝 120개가 차 있으면 한 번.
  프레임이 끊겼다가 와서 스텝 n개를 한꺼번에 발행시킨 프레임에서는 최대 n번(10번 이하)이다.
- **유한성 검사**: 넣기 전에 120 x 17 값이 모두 유한한지 본다. 하나라도 아니면 추론을 건너뛰고 센다
  (`nonfinite_windows`). onnxruntime의 GRU는 NaN 입력에도 그럴듯한 확률을 낸다.
- 실행 설정: CPU, 연산 내부 스레드 1개(`intra_op_num_threads = 1`). 참조 구현과 벡터는 `inter_op_num_threads = 1`도 함께 둔다.

### 14.2 그래프 안의 처리 (앱은 하지 않는다)

```
x' = x                       변환이 none인 특징
x' = log(max(x, 0) + 1)      변환이 log1p인 특징 (#9, #10, #11, #12, #14)
z  = clip((x' − mean) / std, −z_clip, +z_clip)        mean, std는 feature_spec의 features[j] (std는 std_floor 이상)
logits = 신경망(z)
probs  = softmax(logits / temperature)                 temperature는 standardization.temperature
```

`mean`, `std`, `temperature`는 학습으로 정해져 버전마다 다르다. v0은 0.0, 1.0, 1.0이다. 그래프 상수와
`feature_spec_v{N}.json`의 값이 같은지는 내보낼 때 검사한다.

### 14.3 메타데이터 (`metadata_props`, 값은 모두 문자열)

<!-- check:metadata -->

| 키 | 값 | 앱이 할 일 |
|----|----|-----------|
| `model_version` | N (숫자 문자열) | `feature_spec_v{N}.json`의 N과 같다 |
| `feature_spec_hash` | `feature_spec_v{N}.json` 바이트의 sha256 | 번들한 파일의 해시와 비교(15장) |
| `step_sec` | `1` | 기대값과 비교 |
| `window_steps` | `120` | 기대값과 비교 |
| `class_order` | `alert,low_vigilant,drowsy` | 기대값과 비교 |
| `trained_on` | 학습 데이터 설명. v0은 `none (random weights)` | 표시용 |
| `commit` | git commit hash. 작업 트리가 깨끗하지 않으면 `-dirty`가 붙고, 커밋이 없으면 `uncommitted` | 표시용 |
| `standardization` | `in_graph` | 기대값과 비교 |
| `pipeline_hash` | `feature_spec`의 `pipeline_hash` | 이식 코드가 맞춘 값과 비교(15장) |
| `mediapipe_model_sha256` | `mediapipe.model_sha256` | `face_landmarker.task`의 해시와 비교(15장) |

feature_spec 사본은 버전마다 따로다: `drowsiness_v{N}.onnx` 옆의 `feature_spec_v{N}.json`. 계획서 7장의 표에는
`export/feature_spec.json`으로 적혀 있지만 구현은 버전을 붙인다(이 이름으로 확정할지는 소유자의 확인을 기다린다.
20장 10번). INT8 파일 `drowsiness_v{N}_int8.onnx`(TCN이 최종
모델일 때만 만든다)는 같은 버전의 사본을 같이 쓴다.

### 14.4 허용 연산자

<!-- check:operators -->

| 파일 | 연산자 |
|------|--------|
| `default` | Add, Clip, Concat, Constant, Conv, Div, Expand, GRU, Gather, Gemm, Log, Mul, Relu, Shape, Softmax, Sub, Transpose, Unsqueeze |
| `int8` | Add, Cast, Clip, Concat, Constant, ConvInteger, Div, DynamicQuantizeLinear, Gather, Log, MatMulInteger, Mul, Relu, Reshape, Softmax, Sub, Transpose |
| `dynamo` | Add, Clip, Concat, ConstantOfShape, Conv, Div, GRU, Gather, Gemm, Log, Mul, Relu, Slice, Softmax, Sub, Transpose, Unsqueeze |
| `v0` | Add, Clip, Concat, Constant, Conv, Div, Gather, Gemm, Log, Mul, Relu, Softmax, Sub, Transpose |

`default`는 기본 내보내기 경로에서 두 모델(TCN, GRU)이 쓰는 연산자의 합집합, `int8`은 양자화한 TCN, `dynamo`는 대체
내보내기 경로(`--dynamo`로 고를 때만 쓴다)의 목록이고, `v0`은 지금 있는 `drowsiness_v0.onnx`(TCN)가 실제로 쓰는 것이다.
내보내기는 파일 종류에 맞는 목록을 벗어나는 파일을 거부한다. 대상 `onnxruntime-android` 버전이 opset 17과 IR 8을
읽는지는 앱 팀이 확인한다.

`drowsiness_v0.onnx`는 무작위 가중치다. 입출력 계약과 연동을 확인하는 용도이고 판정에 쓸 수 없다.

---

## 15. 앱이 해야 하는 해시 검사

시작할 때 한 번 검사하고, 하나라도 다르면 판정을 시작하지 않는다.

| # | 비교 | 다르면 |
|---|------|--------|
| H1 | sha256(`face_landmarker.task`) = 메타데이터 `mediapipe_model_sha256` = `feature_spec`의 `pipeline.mediapipe.model_sha256` | 학습과 다른 MediaPipe 모델 |
| H2 | sha256(번들한 `feature_spec_v{N}.json` 바이트) = 메타데이터 `feature_spec_hash` | 모델과 다른 spec 파일 |
| H3 | `feature_spec`의 `pipeline_hash`(문자열) = 메타데이터 `pipeline_hash` = 이식 코드를 벡터로 맞출 때의 `pipeline_hash` | 이식 코드를 검증한 상수와 다른 상수 |
| H4 | sha256(번들한 `rules.json` 바이트) = 벡터의 `rules_hash`(빌드에 넣어 둔 값) | 검증한 규칙 설정과 다른 설정 |
| H5 | 메타데이터 `step_sec`, `window_steps`, `class_order`, `standardization`이 14.3절의 값 | 계약이 다른 모델 |

- 파일 해시는 **바이트 그대로** 계산한다. JSON을 다시 쓰거나 줄바꿈을 바꾸면 달라진다(저장소는 `.gitattributes`로 줄바꿈
  변환을 막는다).
- `pipeline_hash`는 Python이 `pipeline` 블록의 정규 JSON에서 계산한 값이다. Kotlin은 계산하지 않고 문자열만 비교한다.
- `rules_hash`는 모델 메타데이터에 없다. 규칙만 바꿔도 모델은 그대로다.
- 규칙 임계값을 바꿔 시험할 때는 `rules.json`을 고치고 H4의 기대값을 함께 바꾼다. `clock` 블록은 `feature_spec`과 같아야 한다.

---

## 16. 테스트 벡터 (`tests/vectors/*.json`)

형식, 벡터별 내용, 쓰는 순서는 `tests/vectors/README.md`에 있다. 여기에는 요점만 적는다.

- 입력은 모두 합성 값이다(실제 얼굴 이미지 없음). 기대값은 참조 구현의 출력이고 성능 수치가 아니다.
- 벡터 하나 = JSON 파일 하나: `frames`(= `RawFrame` 목록), `baseline`(넣어 줄 기준선 또는 null), `config_overrides`
  (규칙 설정에 덮어쓸 값), `speed`, `model`, `expected`. 지금 102개다.
- `expected.step_probs`는 `PipelineOutput.step_probs`를 스텝 순서대로 편 것이다: 발행된 스텝마다 `{t_ms, k, probs}`
  하나씩이고 `t_ms`는 그 스텝을 발행시킨 프레임, `probs`는 그 스텝에서 모델을 불렀으면 확률 3개, 아니면 null이다.
  `expected.probs`는 `PipelineOutput.probs`(그 프레임이 발행시킨 마지막 스텝의 확률)를 null이 아닌 프레임마다 적은
  것이다. 스텝마다 모델을 부르게 되면서 `step_probs`를 더했고 `probs`의 뜻과 모양은 그대로다.
- 먼저 `pipeline_hash`와 `rules_hash`를 본다(15장). 다르면 비교를 멈춘다.
- 묶음별로 `signals` → `baseline` → `runs` → `steps` → `rules` → `e2e` 순서로 통과시킨다. `model` 묶음은 ONNX 실행만 본다.
- `signals` 묶음은 프레임마다 `signals.compute`만, `baseline`과 `runs` 묶음은 `DriverPipeline` 없이도 확인할 수 있다.
  나머지는 `DriverPipeline(spec, rules, baseline, model)`에 프레임을 차례로 넣고 출력을 모아 비교한다.
- `lm`의 키는 문자열이므로 정수로 바꾼다. 숫자는 `Double`, `t_ms`는 `Long`으로 읽는다.
- 버린 프레임(`update`가 null)은 `expected.closure`에 원소가 없다.

허용 오차. 실수는 `abs(실제 − 기대) <= abs + rel * abs(기대)`, 시각은 `±time_abs_ms`로 비교한다.

<!-- check:tolerances -->

| 항목 | abs | rel | time_abs_ms | angle_abs_deg | confidence_abs |
|------|-----|-----|-------------|---------------|----------------|
| `signals` | 1e-6 | | | 1e-3 | |
| `closure` | 1e-6 | | | | |
| `baseline` | 1e-6 | | | | |
| `runs` | | 1e-4 | 1 | | |
| `steps` | 1e-5 | 1e-5 | | | |
| `model_inputs` | 1e-5 | 1e-5 | | | |
| `step_probs` | 1e-4 | | | | |
| `probs` | 1e-4 | | | | |
| `events` | | | 1 | | 1e-4 |
| `alerts` | | | 1 | | 1e-4 |

- `signals`의 `angle_abs_deg`는 `yaw`, `pitch`, `roll`에 쓴다. `runs`의 `time_abs_ms`는 `start_ms`, `end_ms`, `duration_ms`에,
  `rel`은 `amplitude`와 두 속도에 쓴다. `events`와 `alerts`의 `time_abs_ms`는 `t_start_ms`, `duration_ms`에 쓴다.
- 표에 없는 값(불리언, 문자열, 스텝 번호 `k`, `t_end_ms`, 출력을 낸 프레임의 `t_ms`, `counters`, 목록의 길이와 순서)은
  정확히 같아야 한다.
- 저장소의 `python -m pytest tests/test_vectors.py`가 벡터를 다시 만들어 파일과 바이트 단위로 같은지 본다. 참조
  구현이나 상수가 바뀌면 벡터도 다시 나간다.

### 16.1 2026-11-14 개정에서 달라진 벡터

`pipeline_hash`와 `rules_hash`는 그대로다. 해시가 같아도 아래 파일은 내용이 달라졌으므로 벡터를 통째로 다시 받는다.

기존 43개 가운데 달라진 것:

| 벡터 | 달라진 것 | 까닭 |
|------|-----------|------|
| `rules_vats` | `vats` end의 `confidence` 0.9709 → 1.0 | 구간 전체의 `pose_valid` 시간 비율로 고쳤다(12.4절) |
| `runs_lingering_lids` | 첫 구간의 `duration_ms` 1225 → 757, 설명 | 길이는 진입 임계 이상이던 시간의 합(10.4절) |
| `rules_drowsy_model_stub`, `e2e_250s` | `expected.step_probs`와 그 허용 오차가 더해졌다. 다른 값은 같다 | 스텝마다 모델 호출(7.2절) |
| `rules_calibrating_provisional` | 보정 중 40~42초의 프레임 12개를 얼굴 없는 프레임으로 바꿨다. `calibrating` end의 `confidence` 1.0 → 0.9667, 설명 | 신뢰도가 1.0뿐이던 것을 고정하려고 입력을 바꿨다 |
| `runs_dropped_frames` | 설명만(프레임을 확률 0.35로 버렸고 66개 중 29개가 빠졌다) | 문구 정정 |

새 벡터 25개가 고정하는 것(자세한 내용은 각 파일의 `description`과 `tests/vectors/README.md`):

| 주제 | 벡터 |
|------|------|
| 감는 속도의 거슬러 보기를 쓸 수 없는 쌍에서 끊는다 (10.5절 11번) | `runs_velocity_after_gap` |
| 보간 시각의 반올림은 원점과 무관하다 (10.2절) | `runs_rounding_large_t0` |
| 감김의 길이 = 진입 임계 이상이던 시간의 합 (10.4절, 11.4절, 12.5절) | `runs_multi_span`, `runs_multi_span_truncated`, `steps_multi_span_runs`, `rules_closure_dwell_below_1s`, `rules_closure_dwell_reaches_1s`, `rules_closure_three_spans` |
| 끊긴 동안 발행되는 스텝 (11.6절) | `steps_closure_across_gap` |
| 0으로 나누게 되는 비율 (11.4절) | `steps_zero_division` |
| 미뤄 둔 경고 (13장) | `rules_alert_deferred_cooldown`, `rules_alert_blocked_microsleep_sleep`, `rules_drowsy_perclos_exact`, `rules_gap_over_10s`, `rules_unable_sparse_frames` |
| 스텝마다 모델 호출 (7.2절) | `rules_drowsy_model_stall` |
| 10초 초과 끊김 뒤 다시 시작 (12.7절) | `rules_gap_over_10s`, `rules_gap_over_10s_drowsy`, `rules_unable_sparse_frames` |
| 10초 이하의 끊김은 얼굴 없는 시간 (12.3절) | `rules_gap_exactly_10s`, `rules_gap_unable_boundary` |
| 인정 구간과 `gaze_away_long` (12.4절) | `rules_gaze_credit_window_missed`, `rules_gaze_credit_boundary` |
| 영역의 경계, 디바운스, `vats`의 `source`와 초기화 옵션 (12.4절) | `rules_forward_zone_edges`, `rules_gaze_debounce_250ms`, `rules_vats_reset_enabled` |
| 30초보다 긴 `vats`의 `confidence` (12.4절) | `rules_gap_exactly_10s` |
| 한 프레임의 경고 둘과 우선순위 (13장) | `rules_alert_priority`, `rules_gap_over_10s` |
| 등호 자리: `perclos_60` 0.15와 0.10, 속도 20.0과 50.0 km/h, 재경고 대기, 디바운스 250 ms, 처음 10초, 10000 ms와 10001 ms의 끊김 | `rules_drowsy_perclos_exact`, `rules_alert_blocked_microsleep_sleep`, `rules_alert_deferred_cooldown`, `rules_gaze_debounce_250ms`, `rules_unable_sparse_frames`, `rules_gap_exactly_10s`, `rules_gap_over_10s` |

진행 중인 구간에서 closure가 진입 임계 아래, 이탈 임계 위에 머무는 장면이 든 벡터 여섯은 2026-11-14 소유자 결정으로
확정된 동작을 고정한다: 그런 머묾은 구간을 끝내지 않는다(20장 머리의 결정 기록). 다섯(`runs_lingering_lids`,
`runs_multi_span`, `steps_multi_span_runs`, `rules_closure_dwell_below_1s`, `rules_closure_dwell_reaches_1s`)에는
`hold_ms`(500)보다 긴 머묾이 있고, `runs_multi_span_truncated`의 둘째 구간은 진입 임계 아래로 내려간 뒤 얼굴이 사라져
잘린다. 결정을 기다리는 질문에 걸린 벡터가 아니다.

두 번째 검증 뒤에 더한 벡터 13개. 참조 구현의 동작은 바뀌지 않았고 위의 68개는 바이트 단위로 그대로다. 표의 앞
여덟 줄은 저장소의 단위 테스트만 잡던 동작이고(벡터만 돌려 보는 이식 코드에서는 틀려도 드러나지 않았다), 뒤의 일곱 줄은
단위 테스트도 잡지 못하던 동작이다(참조 구현의 한 줄을 일부러 바꾼 변이가 벡터와 단위 테스트를 모두 통과했다):

| 고정하는 것 | 벡터 |
|-------------|------|
| 정점이 뒤의 표본으로 옮겨지는 구간의 감는 속도와 뜨는 속도, 같은 최댓값이 다시 나올 때의 정점 (10.5절 16번) | `runs_two_humps` |
| 감긴 표본이 하나뿐인 채로 잘린 구간의 `end_ms`와 길이, 잘린 구간에도 적용되는 `min_duration_ms` (10.5절 6번) | `runs_one_sample_truncated` |
| 분석 불가 동안 멈추는 연속 영역 밖 시간 (12.4절) | `rules_gaze_unable_pauses_timer` |
| 10초 초과 끊김이 지우는 모델 판정기의 재시작 금지 기억 (12.7절) | `rules_gap_over_10s_drowsy_model` |
| 미뤄 둔 `microsleep` 경고는 다른 종류의 end로 지워지지 않는다 (13장) | `rules_alert_pending_survives_end` |
| 잠깐 앞을 본 구간도 영역 밖 시간이다. 얼굴이 사라진 시각은 다음 프레임의 시각과 관측 + 200 ms 중 이른 쪽이다. 후보는 인정 구간을 넘을 때만 버린다 (12.4절) | `rules_gaze_glance_and_credit_times` |
| 버려진 짧은 감김 뒤에 같은 프레임에서 시작한 새 감김의 `t_start_ms`와 `confidence` (12.5절) | `rules_closure_after_dropped_run` |
| 첫 `t_ms`가 큰 값인 세션의 규칙과 경고: "처음 10초"와 `calibrating`의 구간은 첫 프레임 대비다 (12.2절, 12.3절) | `rules_calibrating_large_t0` |
| 기준선을 완성한 프레임이 `T_b`보다 1초 넘게 늦게 와도 누적은 `T_b`부터다 (7.2절, 11.6절) | `rules_calibrating_large_t0` |
| 특징 쪽의 초기화 경계: 정확히 10000 ms와 10001 ms (11.6절, 7.2절 5번) | `steps_gap_exactly_10s` |
| 10초 초과 끊김이 버리는 전방 주시 후보 (12.7절) | `rules_gap_over_10s_candidate` |
| 인정 구간 뒤에 버려진 영역 안 후보의 구간은 영역 밖 시간이 아니다 (12.4절) | `rules_gaze_candidate_dropped` |
| 눈동자 보조의 나머지 두 방향: `gaze_h_rel` 양수, `gaze_v_rel` 음수 (12.4절) | `rules_forward_zone_eye_edges` |
| `either`에서 PERCLOS가 연 상태의 `source` (12.6절) | `rules_drowsy_either` |
| 길이가 0인 영역 밖 구간은 `vats`의 `t_start_ms`가 되지 않는다 (12.1절, 12.4절) | `rules_gaze_unable_pauses_timer` |

세 번째 검증 뒤에 더한 벡터 8개. 이번에도 참조 구현의 동작은 바뀌지 않았고 위의 81개는 바이트 단위로 그대로다. 참조
구현의 한 줄을 일부러 바꾸는 점검(이번에는 조건을 항상 참이나 거짓으로 바꾸기와 호출 한 줄 지우기도 했다)에서
벡터 81개를 모두 통과하던 자리들이다. 표의 앞 여섯 줄은 저장소의 단위 테스트만 잡던 것이고, 일곱째 줄은 단위 테스트도
잡지 못하던 것이며, 마지막 줄은 그 뒤의 정비에서 같은 점검을 되풀이하다 찾은 것이다(단위 테스트는 잡지만 벡터로는
드러나지 않았고, 실제 프레임에서는 늘 일어나는 경우다). 가운데 칸은 그 줄을 틀리게 옮긴 구현이 이 벡터에서 내는
값이다(참조 구현의 사본으로 돌려 본 값):

| 고정하는 것 | 틀린 구현의 출력 (옳은 값) | 벡터 |
|-------------|---------------------------|------|
| 30초 창의 시작에서 정확히 끝나는 영역 밖 구간은 창에 없다 (12.1절의 `trim`, 12.4절) | `vats` onset의 `t_start_ms` 2000, `duration_ms` 30000 (21000, 11000) | `rules_vats_window_edge` |
| 정점이 옮겨지기 전의 증가율은 가장 큰 값을 들고 있는다 (10.5절 16번) | `peak_closing_velocity` 3.4 (3.64) | `runs_two_rises_before_peak` |
| `vats` onset의 `confidence`: `pose_valid` 구간마다 `t_start_ms` 뒤의 부분만 더한다 (12.1절의 `since_ms`, 12.4절) | 1.4 또는 0.74 (0.8) | `rules_vats_confidence_face_lost` |
| 영역 안 상태에서 버려진 영역 밖 후보의 구간은 영역 밖 시간이 아니다 (12.4절) | `vats` onset과 경고 (이벤트 없음) | `rules_gaze_one_frame_glances` |
| 감김 이벤트의 `confidence`는 `closure_valid`인 프레임이 유지된 시간만 센다 (12.5절, 12.9절) | 1.0 (onset 0.7, end 0.8125) | `rules_microsleep_confidence_face_lost` |
| 미뤄 둔 `microsleep` 경고는 다른 종류의 onset으로 지워지지 않는다 (13장) | 경고 하나 (둘) | `rules_alert_pending_survives_onset` |
| "처음 10초"는 첫 `t_ms`가 작은 세션에서도 첫 프레임부터 잰다 (12.1절, 12.3절) | `unable_to_analyze` onset이 `t_ms` 10000의 프레임 (15000) | `rules_unable_small_t0` |
| `drowsy` 이벤트의 발행 시각은 스텝 끝이고, 바로 낸 경고는 onset 이벤트 그대로다 (12.6절, 13장) | 경고의 `duration_ms` 9050 (9000) | `rules_drowsy_off_step_boundary` |

네 번째 검증 뒤에 더한 벡터 13개. 참조 구현의 동작은 바뀌지 않았고 위의 89개는 바이트 단위로 그대로다. 그때까지 참조
구현의 한 줄을 일부러 바꾸는 점검은 감김 검출, 규칙, 파이프라인 모듈만 보았다. 같은 점검을 신호, 기준선과 closure, 스텝
특징 모듈에 돌렸을 때 벡터 89개를 모두 통과하던 자리 가운데 이식 코드의 출력이 달라지는 것들을 고정한다. 가운데 칸은 그
줄을 틀리게 옮긴 구현이 이 벡터에서 내는 값이다(참조 구현의 사본으로 돌려 본 값). 첫 칸에 (※)가 붙은 줄은 그때 저장소의
단위 테스트도 잡지 못하던 자리다(네 번째 검증이 짚은 스텝 특징의 다섯 동작과 홍채 비율. 단위 테스트도 함께 더했다):

| 고정하는 것 | 틀린 구현의 출력 (옳은 값) | 벡터 |
|-------------|---------------------------|------|
| (※) 홍채 비율은 눈 축이 영상에서 기울어도 같다: `n`, `iris_h`, `iris_v`의 식에서 `a.y`가 든 항 (5.2절) | 첫 프레임의 `iris_h_l` 1.6429 또는 0.1096, `iris_v_l` 0.4472 (0.46, 0.02) | `signals_iris_tilted_eyes` |
| 표본이 홀수 개이면 중앙값은 가운데 표본 하나다 (8.1절) | 기준 pitch 2도와 `jaw0` 0.07: 완성 뒤 둘째 프레임의 `pitch_rel`이 -2.0 (4도와 0.09, -4.0) | `baseline_odd_sample_count` |
| Right 눈의 표본이 하나도 없으면 완성하지 않는다. E의 범위는 두 눈 모두에서 본다. 쓸 수 없는 Right 눈의 EAR은 표본이 아니다 (8.2절의 `collect`와 `try_complete`) | `T_b` 60000 또는 90000, 또는 세션이 끝날 때까지 완성되지 않음 (120000) | `baseline_odd_sample_count` |
| 얼굴 시간: 프레임 하나가 대표하는 시간은 `clock.max_hold_ms`까지이고 마감을 넘는 부분과 얼굴 없는 프레임은 세지 않는다 (8.2절의 `held`와 `held_before`) | `T_b` 60000 (90000) | `baseline_burst_frames` |
| 얼굴 검출 비율이 정확히 0.5이면 완성한다('이상'). 마감 직전 프레임이 대표한 시간도 더한다 (8.2절) | 세션이 끝날 때까지 완성되지 않음 (`T_b` 90000) | `baseline_burst_frames` |
| 얼굴 프레임이 정확히 300개이면 완성한다('이상'). Left 눈의 표본이 없는 마감에서는 분위수를 구하지 않고, 쓸 수 없는 Left 눈의 EAR은 표본이 아니다 (8.2절) | 세션이 끝날 때까지 완성되지 않음, 또는 60초 마감에서 빈 목록의 분위수를 구하려다 나는 오류 (`T_b` 90000, `e_l` 0.30) | `baseline_exactly_300_frames` |
| 블렌드셰이프 쪽 분모의 하한 `closure.b_span_min` (9장의 `span_b`) | closure 0.25, 0.375 (0.125, 0.325) | `closure_blink_span_floor` |
| 하품: 시작 임계에 닿아야 시작한다. 무효 구간이 500 ms를 '넘으면' 취소한다. 2000 ms '이상'이 된 프레임에서 한 번 세고, 끝 임계 아래로 내려간 프레임에서는 세지 않는다 (11.5절) | 스텝 15의 `yawn_count_60` 1 또는 3 (2) | `steps_yawn_edges` |
| (※) 끄덕임: 길이 200 ms '이상' 3000 ms '이하'이면 센다. 얼굴 없는 프레임은 값을 내지 않고, 무효 구간이 500 ms를 '넘으면' 취소한다 (11.5절. ※는 얼굴 없는 프레임 쪽) | 스텝 8의 `nod_count_60` 2 또는 4 (3) | `steps_nod_edges` |
| (※) 느린 평균: 첫 유효 표본이 초기값이다. `d = pitch_rel - mean`. `alpha = dt / (nod.ema_tau_ms + dt)`, `dt`는 `nod.ema_max_dt_ms`까지 (11.5절. ※는 빼는 방향) | 스텝 8의 `nod_count_60` 0 (1) | `steps_nod_slow_mean` |
| (※) 10초 초과 끊김 뒤에는 끄덕임의 느린 평균도 버린다 (11.6절의 `nod.reset()`) | 스텝 13과 14의 `nod_count_60` 1 (0) | `steps_nod_after_long_gap` |
| (※) 한 스텝의 통에는 그 스텝에 귀속된 감김 가운데 가장 긴 것이 남는다 (11.2절의 `max_run_ms`) | 스텝 1의 `max_closure_ms_60`과 `max_closure_ms_10` 88 (388) | `steps_two_closures_in_one_step` |
| `max_closure_ms_60`과 `max_closure_ms_10`의 상한 `step_features.max_closure_cap_ms`: 진행 중인 감김에도, 끝난 감김에도 (11.4절) | 스텝 11~13의 값 10489, 11489, 11578 (모두 10000) | `steps_closure_above_cap` |
| 긴 감김의 기준은 '이상'이다: 끝난 감김과 진행 중인 감김 모두 (11.2절의 `n_long`, 11.4절 #9) | `long_closure_count_60` 0, 1, 1 또는 1, 1, 2 (1, 2, 2) | `steps_closure_exactly_500ms` |
| (※) `pitch_std_60`의 얼굴 시간 기준은 '이상'이다 (11.4절 #13) | 스텝 5의 `pitch_std_60` 0.0 (2.449490) | `steps_pitch_std_at_5000ms` |

홍채 비율은 기록용 신호다. 이식 대상의 다른 모듈(기준선과 closure, 감김 검출, 스텝 특징, 규칙, 파이프라인)은
`iris_h`, `iris_v`를 읽지 않는다. 그래서 5.2절의 그 식이 틀려도 `signals` 묶음 밖의 벡터와 이벤트는 달라지지 않는다. 그
식을 고정하는 것은 벡터 `signals_iris_tilted_eyes`와 저장소의 단위 테스트 `test_iris_ratio_on_a_tilted_eye_axis`
(`tests/test_signals.py`)다. 다른 신호 벡터는 두 눈 꼬리의 높이가 같아서(`a.y = 0`) 그 항이 0이다.

`config_overrides`가 있는 벡터는 다섯이다: `rules_drowsy_model_stub`, `rules_drowsy_model_stall`,
`rules_gap_over_10s_drowsy_model`(`drowsiness.source = model`. 마지막 것은 `drowsiness.model.refractory_ms = 600000`도),
`rules_drowsy_either`(`drowsiness.source = either`), `rules_vats_reset_enabled`(`vats.reset_enabled = true`).

**벡터가 고정하지 않는 동작.** 벡터 102개를 모두 통과해도 아래는 확인된 것이 아니다. 이식할 때 이 문서의 의사 코드와
비교 연산자를 그대로 옮긴다. 참조 구현의 한 줄을 일부러 바꾸는 점검에서 벡터는 모두 통과하고 저장소의 단위 테스트만
실패한 자리를 빠짐없이 옮겨 까닭별로 묶었고, 이 문서만 보고 옮긴 사본을 일부러 틀리게 한 점검에서 찾은 자리(PERCLOS
`drowsy`의 `confidence`가 읽는 인덱스, 분위수의 순위, 모델 창을 미는 순서)도 더했다. 이 목록은 그런 점검이 찾은 자리를 모은
것이고, 이식 코드가 틀릴 수 있는 자리를 모두 담은 것은 아니다.

점검은 이렇게 했다. 대상은 이식 대상 모듈 가운데 계산이 있는 여섯이다(`drowsy/types.py`는 자료형과 상수뿐이다). 한 번에
한 줄만 바꾼 사본을 만들고, 사본마다 `python -m drowsy.vectors verify`를 돌린 뒤 저장소의 단위 테스트(여섯 모듈의 테스트
파일. 거기까지 통과한 사본에는 `tests/test_vectors.py`도)를 돌렸다. 바꾸는 방법은 두 묶음이다. 기본 묶음은 비교 연산자,
`and`와 `or`, 참과 거짓, 덧셈과 뺄셈을 하나씩 바꾸고 상태를 적는 줄을 하나씩 지운다. 확장 묶음은 조건을 항상 참이나
거짓으로 두고, 호출 한 줄을 지우고, 1씩 세는 줄을 2씩 세게 한다. 아래는 벡터 102개에 대한 결과다. "벡터가 잡음"은
`verify`가 실패한 사본, "단위 테스트만 잡음"은 `verify`는 통과하고 단위 테스트가 실패한 사본, "둘 다 통과"는 어느 쪽도
실패하지 않은 사본이다:

<!-- check:mutation-check -->

| 모듈 | 묶음 | 바꾼 사본 | 벡터가 잡음 | 단위 테스트만 잡음 | 둘 다 통과 |
|------|------|-----------|-------------|--------------------|------------|
| `drowsy/signals.py` | 기본 | 45 | 40 | 5 | 0 |
| `drowsy/signals.py` | 확장 | 18 | 10 | 8 | 0 |
| `drowsy/calibrate.py` | 기본 | 101 | 74 | 17 | 10 |
| `drowsy/calibrate.py` | 확장 | 61 | 50 | 9 | 2 |
| `drowsy/blink.py` | 기본 | 101 | 72 | 16 | 13 |
| `drowsy/blink.py` | 확장 | 53 | 50 | 2 | 1 |
| `drowsy/step_features.py` | 기본 | 178 | 140 | 8 | 30 |
| `drowsy/step_features.py` | 확장 | 113 | 106 | 3 | 4 |
| `drowsy/rules.py` | 기본 | 398 | 336 | 22 | 40 |
| `drowsy/rules.py` | 확장 | 233 | 210 | 12 | 9 (그 밖에 2개는 문법에 맞지 않는다) |
| `drowsy/pipeline.py` | 기본 | 64 | 55 | 4 | 5 |
| `drowsy/pipeline.py` | 확장 | 55 | 42 | 10 | 3 |

네 번째 검증 때의 벡터 89개만 놓고 세면 "벡터가 잡음"은 `signals.py`가 37과 10, `calibrate.py`가 59와 41,
`step_features.py`가 119와 96이었다(기본, 확장 순서. 같은 실행에서 새 벡터 13개만 실패한 사본을 빼고 센 값이고, 네 번째
검증이 89개로 잰 수와 같다). 나머지 세 모듈의 수는 새 벡터와 무관하게 같다.

"둘 다 통과"한 사본 117개는 출력이 달라지지 않는 것들이다(상수가 설정 검증을 통과하는 한. 10.1절, 11.5절). 117개
모두 무작위 세션을 원본과 사본에 넣어 출력을 비교했고 차이가 없었다. 신호, 기준선, 스텝 특징 모듈의
46개는 줄마다 까닭을 확인했다. 네 가지다. 같은 값일 때 아무 일도 하지 않는 비교(상한과 최댓값을 구하는 자리의 `>`와
`>=`, 0으로 자르는 자리의 `<`와 `<=`, 길이가 0인 시간을 더하는 것). 읽기 전에 다시 적히는 상태(초기화가 0으로 되돌리는
시각과 표시, 만들 때의 처음 값). 설정 검증이 막아 둔 경우에만 쓰이는 줄(초기화를 일으키는 끊김에서는 이미 취소된 뒤인
계수기의 `reset()`, 순위나 마감을 범위 안으로 자르는 줄). 그리고 `pitch_sum`을 더하는 부호(11.2절, 11.3절): 이 합은
`pitch_std_60`에서 제곱으로만 쓰여 부호가 바뀌어도 값이 같다. 감김 검출, 규칙, 파이프라인 모듈의 71개는 앞선 검증에서
줄마다 확인한 것과 같은 수다. 아래 목록은 "단위 테스트만 잡음"의 자리를 모은 것이다.

벡터 형식에 담을 자리가 없는 것:

- 표시용 상태 `gaze_zone`(12.10절). 형식에 이 값을 적는 항목이 없다. 틀려도 이벤트와 경고는 같다. 벡터에 담을지는
  20장 12번에 있다.
- `t_ms`가 0 이하인 프레임. 벡터의 `t_ms`는 양수다(1장 G8). 첫 `t_ms`가 0인 세션에서만 갈리는 자리가 둘 있다. 규칙
  엔진이 감김의 길이와 시작을 감긴 프레임에서만 읽고 기록하는 것(12.5절의 `closed_frame` 조건들. 뜬 프레임에서도
  기록하면 시작 시각 0이 적히는데, 감김이 정확히 `t_ms` 0에서 시작할 때만 그 기록이 새 감김과 구별되지 않는다)과, 감김
  검출기의 시각 순서 검사가 첫 프레임에는 걸리지 않는 것(10.1절. 처음 상태를 "직전 프레임 있음, 시각 0"으로 두면
  `t_ms`가 0 이하인 첫 프레임을 거부한다).
- `DriverPipeline`을 만들 때의 인자 검사 셋(7.1절의 "오류" 세 줄)과, 모델 함수가 확률 3개를 돌려주지 않을 때의 오류.
  참조 구현에는 이 세션에서 확정한 기준선의 판정 통계를 읽는 속성이 하나 더 있는데 학습 쪽에서만 쓰고 이 문서에는 없다.
- 넣어 준 기준선의 누적 시작 시각이 0보다 큰 경우(7.1절의 `feature_start`): 그 시각 전의 프레임은 특징용 검출기와
  누적기에 넣지 않는다(7.2절 5번). 벡터에 넣어 준 기준선은 모두 `t_complete_ms`가 0이고 `feature_start_ms` 인자는 쓰지
  않는다. 형식으로는 담을 수 있지만(`baseline.t_complete_ms`) 학습용 경로이고 앱은 기준선을 넣지 않아서 두지 않았다.
- 입력의 길이가 맞지 않을 때의 오류: 변환 행렬이 16개 값이 아닐 때, 회전이 9개 값이 아닐 때, EAR의 점이 6개가 아닐 때,
  빈 목록의 분위수와 중앙값(5.2절, 5.3절, 8.1절). 벡터는 오류를 기대값으로 담지 못한다. 기준선의 진단용 기록
  (`BaselineDiagnostics`, 8.3절)도 형식에 자리가 없다.

`DriverPipeline`을 거쳐서는 일어나지 않는 것(구성 요소를 따로 부를 때만 드러난다):

- `StepHysteresis`가 값이 없는 스텝이나 번호가 건너뛴 스텝에서 두 연속 수(`onset_run`, `end_run`)를 0으로 되돌리는
  것과, `reset()`이 연속 수를 0으로 되돌리는 것(12.6절). `DriverPipeline`에서 그런 스텝은 세션의 처음(연속 수가 아직
  0이다)과 10초 초과 끊김 뒤(판정기가 통째로 `reset()`된 뒤이고, 끊김 뒤의 첫 스텝들은 값이 없다)에만 나온다. 특징 값이
  유한하지 않아 추론을 건너뛰는 경우(7.2절의 `nonfinite_windows`)는 합성 프레임으로 만들 수 없다.
- `sleep`의 end가 미뤄 둔 `microsleep` 경고를 지우는 것(13장). 규칙 엔진은 `sleep`의 onset을 먼저 내고 그때 이미
  지워지므로, 엔진을 거친 이벤트로는 그 자리에 미뤄 둔 `microsleep`이 남아 있을 수 없다.
- 같은 종류의 onset이 end 없이 다시 왔을 때 앞의 기록을 지우는 것(13장의 onset 쪽 `forget(event.type)`). 규칙 엔진은
  그런 순서로 내지 않는다.
- `t_ms`가 직전 값 이하인 프레임의 거부. `RuleEngine.update`와 `ClosureRunDetector.update`는 아무 상태도 바꾸기 전에
  오류를 낸다. `DriverPipeline`이 그런 프레임을 먼저 버린다(7.2절 1번).
- `ClosureRunDetector.reset()` 뒤의 첫 표본이 쌍을 이루지 않는 것과, `OPEN`에서 `run_start_ms()`가 0인 것(10.2절,
  10.4절). `reset()`은 10초 초과 끊김 뒤에만 불리고 그 간격은 `max_dt_ms`를 넘으므로 쌍은 어차피 쓸 수 없다. 규칙
  엔진은 `OPEN`에서 `run_start_ms()`를 읽지 않는다. `reset()`은 시각 순서 검사도 처음 상태로 되돌린다(10.2절의
  `has_frame = false`. 직전 프레임이 없는 것으로 보므로 `reset()` 뒤의 첫 `t`는 앞의 `t` 이하여도 받는다. 단위 테스트
  `test_interpolation_needs_gap_within_max_dt`가 고정한다). `DriverPipeline`에서는 `t_ms`가 늘 증가하므로 드러나지 않는다.
- `BaselineEstimator.update`와 `StepAggregator.update`가 직전 값 이하의 `t`에 내는 오류(8.2절, 11.2절), `StepAggregator`가
  `start_ms`를 검사하는 것과 `start_ms` 앞의 프레임을 무시하는 것(11.2절). `DriverPipeline`은 그런 프레임을 먼저 버리고,
  `start_ms`로는 언제나 스텝 경계인 `T_b`를 주며, 누적기는 `T_b` 이후의 프레임만 받는다(7.2절).
- 기준선이 확정된 뒤의 `BaselineEstimator.update`가 null을 돌려주는 것(8.2절의 `done`). `DriverPipeline`은 확정 뒤에
  추정기를 부르지 않는다(7.2절 4번).
- `eye_aspect_ratio`가 폭 0에서 0.0을 돌려주는 것과 홍채 비율의 `n <= 0.0` 처리(5.2절). `compute`는 폭이 6 px 미만인
  눈에서 둘 다 계산하지 않고, 두 눈 꼬리가 EAR의 p1, p4와 같은 점이라 `n`은 폭의 제곱이다.
- 변환 행렬에서 회전 블록을 바로 꺼내는 식(5.3절의 "행렬에서 회전 블록"). `compute`는 행렬에서 각도를 구하고
  `frame_state`는 그 각도에서 회전을 다시 만들므로(9장) 파이프라인은 이 식을 지나지 않는다. 참조 구현에서는 도우미
  함수와 단위 테스트만 쓴다.
- 분위수의 순위를 1 이상 n 이하로 자르는 것(8.1절). 설정 검증이 분위수를 0 초과 1 이하로 제한하므로 `ceil(q * n)`은
  언제나 그 범위 안이다.
- `eye_closure`에서 `E * (1 - rho)`가 0 이하일 때 EAR 쪽을 0으로 두는 것(9장의 `span_e > 0.0`). 파이프라인이 만든
  기준선의 E는 `baseline.e_min` 이상이고 전역 상수의 E도 그렇다. 넣어 준 기준선의 E가 0일 때만 이 자리에 온다.
- 하품·끄덕임 계수기의 `reset()`이 진행 중인 판정을 버리는 것과 `NodDetector`의 `reset()`이 느린 평균의 값을 0.0으로
  되돌리는 것(11.5절, 11.6절). `reset()`은 10초 초과 끊김 뒤에만 불리고 그 끊김은 `cancel_gap_ms`를 넘으므로(11.5절의
  설정 조건) 판정은 다음 프레임에서 어차피 취소된다. 느린 평균의 값은 `has_mean = false`만으로 다음 유효 표본에서 다시
  정해진다(느린 평균을 버리는 것 자체는 벡터 `steps_nod_after_long_gap`이 고정한다).
- 내부 기록만 달라지고 출력은 같은 것 셋. 10초 초과 끊김의 초기화가 얼굴 창을 비우는 것(비우지 않아도 다음 판정까지
  10초 사이에 옛 구간이 창에서 빠진다), `pose_window.trim`(자르지 않아도 `since_ms`가 `vats_start` 앞의 구간을 세지
  않는다. 메모리만 는다), 막힌 `sleep` onset을 `pending`에 넣지 않는 것(넣어도 같은 호출의 되풀이 쪽이 건너뛰어 경고는
  하나다).

지금 상수와 다른 설정에서만 드러나는 것(벡터는 `pipeline` 상수를 덮어쓰지 않고, 규칙 설정은 위의 다섯 벡터만 덮어쓴다):

- `velocity_lookback_samples`(2)가 다른 값일 때의 거슬러 보기 버퍼(10.2절의 `recent_up`).
- `forward_zone.eye_aux_enabled = false`일 때 눈동자 보조 판정을 건너뛰는 것(12.4절의 `observe`).
- `events.sleep.repeat_ms`가 `closure.sleep_ms`보다 긴 설정에서, 끝난 `sleep` 상태의 되풀이 시각이 다음 상태로 넘어가지
  않는 것(13장의 `forget`이 `repeat_alert`도 지운다). 지금 상수(둘 다 3000)에서는 새 `sleep`이 앞 상태의 마지막 경고에서
  3000 ms 넘게 지난 뒤에야 시작하므로 남아 있어도 차이가 없다.
- 속도가 음수일 때 기준이 0인 종류(`unable_to_analyze`)를 억제하지 않는 것(13장의 `min_speed_kmh > 0.0`). 벡터의
  속도는 0 이상이다.
- 길이가 0인 구간의 `ratio`가 0.0인 것(12.2절, 12.9절). 지금 상수에서는 규칙 엔진이 `ratio`를 부르는 자리의 분모가 0이
  되지 않는다(`calibrating`과 `unable_to_analyze`의 onset은 0.0을 직접 적는다). `closure.microsleep_ms`가
  `closure.detector.max_dt_ms`(200) 이하인 설정(설정 검증은 받아들인다)에서는 감김을 처음 본 프레임에서 `microsleep`
  onset이 날 수 있고, 그 `confidence`는 분모(발행 시각 − 감김을 처음 본 프레임)가 0이라 0.0이다. 지금 값(1000)에서는
  그런 onset이 나지 않는다.

벡터가 지나지 않는 자리:

- 기준선의 얼굴 시간에서, 마감 직전의 프레임이 마감에서 200 ms 넘게 떨어진 얼굴 프레임이면 그 프레임이 대표한 200 ms까지만,
  얼굴 없는 프레임이면 0 ms만 마감 전의 얼굴 시간에 더하는 것(8.2절 `held_before = min(deadline - prev_t, held)`의 `held`
  쪽. 마감에서 자르는 쪽은 `baseline_burst_frames`가 고정한다). 벡터에 그런 마감이 있지만 그 프레임과 마감 사이를 모두
  세어도 판정이 같다. 판정이 갈리는 마감은 벡터에 넣지 않았다. 새 보정 세션은 필요 없다: `baseline_burst_frames`의
  프레임에서 하나를 빼고 76개를 더하면(파일이 약 0.14 MB 커진다) 그 `min`을 빠뜨린 사본에서 `T_b`가 120000 대신 90000이
  된다. 벡터를 다시 만들 때 더할 수 있다. 지금은 저장소의 단위 테스트가 고정한다. 그 `min`을 빠뜨린 사본은 무작위 보정
  세션 60개 가운데 4개에서 출력이 달랐다. 이식할 때 빠뜨리지 않는다.
- PERCLOS가 연 `drowsy`의 `confidence`가 `values[15]`(`face_missing_ratio_60`)를 쓰는 것(12.6절, 12.9절). perclos
  `drowsy` 이벤트가 있는 벡터에서는 발행한 스텝의 #15와 #16이 늘 같아서 `values[16]`(`face_missing_ratio_10`)을 읽는
  구현도 통과한다. 그런 사본은 무작위 세션에서 `confidence`가 달랐다(0.6663, 옳은 값 0.6815). 저장소의 단위 테스트
  `test_perclos_drowsy_onset_and_end`(`tests/test_rules.py`)가 고정한다. 이식할 때 인덱스를 그대로 옮긴다.
- 분위수의 순위가 정확히 정수가 되는 `q * n`(8.1절의 `ceil`). 벡터에서 그런 n(360)은 이웃한 표본이 같은 값이라
  `floor(q * n) + 1`로 옮긴 구현도 통과한다. 무작위 보정 세션에서는 E가 달라진다(n = 330: 0.277511, 옳은 값 0.277428).
  저장소의 단위 테스트가 고정한다.
- 모델 창이 첫 호출 뒤에 미는 순서(7.2절 `push_step`, 14.1절). `expected.model_inputs`는 첫 창 하나뿐이고, 그 뒤의 창은
  무작위 가중치 v0 모델의 확률(허용 오차 1e-4)로만 비교된다. 맨 앞 대신 둘째 스텝을 버리는 구현도 통과한다(확률 차
  6e-8). 학습한 모델에서는 출력이 달라진다. 저장소의 단위 테스트 `test_model_window_is_120_by_17_oldest_first`
  (`tests/test_pipeline.py`)가 마지막 창까지 비교해 고정한다.
- 인정 구간이 끝나는 바로 그 시각에 온 관측 없는 프레임(12.4절의 `!zone.cand` 조건. 7200 ms와 7201 ms에서 결과가
  갈리는 경계). 소유자의 결정을 기다리는 동작이라(20장 9번) 지금 동작은 단위 테스트 하나만 고정하고, 결정이 나면 그
  경계의 벡터를 더한다.
- 감김 검출기의 처음 상태에 "직전 유효 표본 없음"이 들어 있는 것(10.2절의 `has_prev = false`. 10.5절 5번). 벡터의 첫
  `t_ms`는 거의 모두 1이라, 처음 상태를 "직전 표본 있음, 시각 0, closure 0"으로 잘못 둔 구현은 첫 표본과 1 ms 간격의
  쌍을 만든다. 그 쌍의 증가율은 세션의 첫 표본이나 둘째 표본에서 시작하는 구간의 감는 속도에만 들어가는데 벡터에는
  그런 구간이 없다. 실제 타임스탬프에서는 그 간격이 `max_dt_ms`를 넘어 쌍이 되지 않으므로 차이가 없다.

실수로 계산한 값이 임계에 정확히 놓여야 드러나는 등호 자리도 벡터가 고정하지 않는다: 전방 주시 영역의 다섯 비교
(`abs(yaw_rel) > 23.0` 등)와 두 감김 검출기의 진입·이탈 임계(10.2절의 `>=`와 `<`, 12.5절의 `closed_frame`). 그런 표본은
수학 함수의 마지막 자리 차이로 갈리므로 두지 않는다(바로 안쪽과 바깥쪽 값으로 고정한다). 신호, 기준선, 스텝 특징 쪽의
같은 자리는 눈 폭이 정확히 6 px인 눈(5.2절의 `<`, 8.2절의 `collect`와 9장의 `>=`), `abs(yaw_rel)`이 정확히 40도인
프레임(9장의 `<=`), E가 정확히 `baseline.e_min`이나 `baseline.e_max`인 기준선과 기준 yaw가 정확히 15도인 기준선(8.2절),
하품과 끄덕임의 시작·끝 임계(11.5절의 `>=`와 `<`), closure가 정확히 `step_features.perclos_thr`인 프레임(11.2절)이다.
벡터에는 그 임계에서 떨어진 값만 있고, 등호는 저장소의 단위 테스트가 고정한다. `p_drowsy <= 0.4`의 등호는
0.4를 `Float`로 정확히 적을 수 없어서 두지 않는다. 이 자리들은 이 문서의 비교 연산자를 그대로 옮긴다(2장).

---

## 17. 프레임률: 15 fps와 프레임 누락에서의 동작

시간 누적은 프레임률과 무관하게 정의돼 있다. 프레임률에 따라 달라지는 것은 아래다. 17.1절과 17.2절은 정의에서 나오는
성질이고, 17.3절은 합성 깜빡임 파형으로 돌린 모사의 결과다.

### 17.1 프레임률에 따라 달라지는 것

| 항목 | 15 fps(간격 약 67 ms)에서 |
|------|---------------------------|
| 시간 누적 | 간격이 `clock.max_hold_ms`(200) 이하이면 빠지는 시간이 없다. 프레임 하나가 빠져도(간격 약 133 ms) 같다. 두 개가 연달아 빠지면 간격이 200 ms 안팎이라 경계에 걸리고, 200 ms를 넘는 부분만 "데이터 없음"이 된다 |
| 감김 검출 | 진입 임계 이상인 표본이 하나는 있어야 구간이 된다. 두 프레임 사이에 통째로 들어간 깜빡임은 보이지 않는다 |
| `start_ms`, `end_ms` | 앞뒤 표본의 선형 보간이라 표본 간격보다 잘게 나온다. 눈꺼풀이 직선으로 움직이지 않는 만큼 오차가 있다(17.3절) |
| 짧은 깜빡임 | 한 표본짜리 깜빡임은 보간한 길이가 `run_detector.min_duration_ms`(50) 이상일 때만 남는다 |
| `amplitude` | 표본 중 최댓값이라 정점이 표본 사이에 떨어지면 작게 읽힌다 |
| 속도 | 표본 사이의 평균 변화율이다. 간격이 길수록, 쌍이 정점을 걸칠수록 작게 읽히고 위상에 따라 흔들린다. 특징 #7, #8이 프레임률에 의존한다 |
| 임계 시간 판정 | 프레임이 도착할 때만 본다. `microsleep`·`sleep`·`gaze_away_long`의 onset은 조건을 만족한 뒤 첫 해당 프레임에서 난다(늦어도 한 프레임 간격). 얼굴 소실 뒤의 인정 구간에서 3초가 찬 `gaze_away_long`만은 그 구간에 프레임이 없으면 구간이 끝난 뒤의 첫 프레임에서 나므로 더 늦을 수 있고, 그 프레임에서 운전자가 앞을 보고 있으면 나지 않는다(12.4절) |
| 디바운스 | `forward_zone.debounce_ms`(250) 이상 지난 첫 프레임에서 확정한다. 15 fps에서는 후보 뒤 네 번째 프레임(약 267 ms) |
| 스텝 발행 | 스텝 k는 세션 시간이 `1000k` 이상인 첫 프레임에서 나온다(늦어도 한 프레임 간격). 모델 호출도 같다 |

같은 눈꺼풀 움직임을 30 fps와 15 fps로 본 예는 벡터 `runs_blinks_30fps`와 `runs_blinks_15fps`다(직선으로 움직이는 합성
눈꺼풀. 15 fps에서는 120 ms 깜빡임이 0.5 이상인 표본 하나뿐이고 보간한 길이가 50 ms 미만이라 버려진다). 프레임이
무작위로 빠진 경우는 `runs_dropped_frames`, 한 표본짜리 깜빡임은 `runs_one_sample_blink`가 고정한다.

### 17.2 프레임 간격이 길어질 때 (상수에서 나오는 값)

| 간격 | 일어나는 일 |
|------|-------------|
| 200 ms 초과 | 넘는 부분이 "데이터 없음". 감김 검출기는 그 쌍으로 보간과 속도 계산을 하지 않고, 감는 속도의 거슬러 보기를 거기서 끊는다(10.5절 11번) |
| 얼굴 프레임이 평균 초당 5개 미만 | 60초 안에 얼굴 프레임 300개를 못 채워 기준선이 연장된다 |
| 400 ms 초과가 이어짐 | 얼굴 시간 비율이 0.5 아래로 내려가 `unable_to_analyze`(벡터 `rules_unable_sparse_frames`) |
| 500 ms 초과 | 진행 중이던 감김이 잘린다(`truncated`). 하품·끄덕임 판정이 취소된다 |
| 1000 ms 초과 | 한 프레임이 스텝을 둘 이상 발행할 수 있다. 스텝마다 모델을 부르고 그 확률을 그 스텝의 졸음 판정에 주므로 연속 수는 끊기지 않는다(7.2절, 12.6절). 그 스텝들의 특징은 끊기기 직전 프레임까지의 정보로 만든다(11.6절). 한 프레임의 추론 횟수는 최대 10번이다 |
| 3200 ms 초과 | 걸려 있던 전방 주시 후보가 버려진다. 영역 밖 시간은 마지막 관측 + 최대 3200 ms까지만 친다. 그 안에서 연속 3초가 찼으면 다음 프레임에서 `gaze_away_long`이 시작된다. 단 그 프레임이 분석 불가 중이 아니고 그 프레임의 표지가 IN이 아닐 때다: 얼굴이 없거나 여전히 영역 밖이면 시작되고, 운전자가 앞을 보고 있으면(표지 IN) 시작되지 않는다(12.4절) |
| 5200 ms 초과, 10000 ms 이하 | 세션의 처음 10초가 지난 뒤라면 다음 프레임에서 `unable_to_analyze`가 시작되고(끊기기 전 10초 동안 얼굴이 계속 있었어도) 얼굴 시간이 다시 5초 쌓일 때까지 시선·감김 규칙이 멈춘다(12.3절, 20장 6번) |
| 정확히 10000 ms | 초기화하지 않는다(위 행과 같다). 특징 쪽도 같다: 진행 중이던 감김은 잘린 구간으로 나오고 끊긴 동안의 스텝 열 개가 발행된다(벡터 `steps_gap_exactly_10s`, `rules_gap_exactly_10s`) |
| 10000 ms 초과 | 초기화(6장). 스텝 누적과 모델 창이 다시 시작한다: `full_lookback`은 새 누적 시작 + 60초부터, 다음 모델 호출은 그 뒤 연속 120스텝이 찼을 때다. 규칙 엔진은 진행 중이던 상태를 끝내고 다시 시작하며 그 프레임부터 10초 동안 `unable_to_analyze`를 판정하지 않는다(12.7절) |

### 17.3 깜빡임 길이 오차의 모사 (합성 파형)

**실제 눈을 잰 값이 아니다.** 합성 깜빡임 파형을 프레임률과 위상을 바꿔 표본화하고 특징용 `ClosureRunDetector`
(진입 0.5 / 이탈 0.3, `min_duration_ms` 50, `max_dt_ms` 200)에 넣어 잰 모사다. MediaPipe의 시간 평활과 신호 잡음은
들어 있지 않다. 성능 수치가 아니다.

- 명령: `python -m drowsy.signal_check blink-sim --config config/default.yaml --run-name <이름>`. 아래 값은 2026-11-14에
  이 명령을 시드 0(기본값. 프레임 누락 패턴에만 쓰인다)으로 돌린 결과이고, `reports/<이름>/blink_sim/summary.md`와
  `blink_sim.csv`에 같은 값이 있다. 저장소의 테스트(`tests/test_spec_doc.py`)가 같은 모사를 다시 돌려 아래 표와 비교한다.
- 파형: 감는 구간이 전체의 35%인 비대칭 코사인 모양(`signal_check.BlinkShape`). 머무는 구간 없음, 뜬 눈의 closure는 0.
- 길이(closure가 진입 임계 이상인 시간, 검출기가 재는 `duration_ms`의 참값): 60, 80, 100, 150, 200, 300, 400 ms.
  진폭: 0.7, 0.85, 1.0. 위상 50개(프레임 주기를 고르게 나눈 값). 누락이 있는 조건은 위상마다 누락 패턴을 4번 뽑는다.
- 조건: `30fps`, `15fps`, `15fps_drop20`(15 fps에서 프레임마다 확률 0.2로 누락).
- 시각 방식: `interpolated`는 검출기 그대로(앞뒤 표본의 선형 보간, 10.2절), `frame_time`은 같은 검출기에
  `max_dt_ms = 1`을 줘서 보간을 끈 것이다(시작은 진입 임계 이상을 처음 본 프레임, 끝은 진입 임계 이상이었던 마지막
  프레임의 시각).
- 오차 = 검출기의 `duration_ms` − 참값. 치우침은 평균, 중앙 절대 오차는 절대값의 중앙값, 95%는 절대값의 95 분위수
  (최근접 순위)이고 검출된 깜빡임만으로 계산한다. 합친 행은 길이와 진폭의 모든 조합에 같은 비중을 준 값이다. 실제
  깜빡임 길이의 분포가 아니다.

조건별(길이와 진폭을 모두 합침):

<!-- check:blink-sim-conditions -->

| 조건 | 시각 방식 | 시행 | 검출 | 치우침 (ms) | 중앙 절대 오차 (ms) | 절대 오차 95% (ms) |
|------|-----------|------|------|-------------|---------------------|--------------------|
| `30fps` | `interpolated` | 1050 | 1050 | -0.9 | 1.0 | 4.0 |
| `30fps` | `frame_time` | 1050 | 809 | -31.9 | 33.0 | 50.0 |
| `15fps` | `interpolated` | 1050 | 951 | -3.6 | 3.0 | 19.0 |
| `15fps` | `frame_time` | 1050 | 704 | -61.1 | 67.0 | 100.0 |
| `15fps_drop20` | `interpolated` | 4200 | 3502 | -6.0 | 6.0 | 43.0 |
| `15fps_drop20` | `frame_time` | 4200 | 2473 | -81.1 | 67.0 | 167.0 |

15 fps의 길이별(진폭을 합침). `-`는 검출된 깜빡임이 없어 값이 없다는 뜻이다:

<!-- check:blink-sim-durations -->

| 조건 | 길이 (ms) | 시각 방식 | 시행 | 검출 | 치우침 (ms) | 중앙 절대 오차 (ms) | 절대 오차 95% (ms) |
|------|-----------|-----------|------|------|-------------|---------------------|--------------------|
| `15fps` | 60 | `interpolated` | 150 | 51 | -0.3 | 5.0 | 9.0 |
| `15fps` | 60 | `frame_time` | 150 | 0 | - | - | - |
| `15fps` | 80 | `interpolated` | 150 | 150 | -8.6 | 12.5 | 27.0 |
| `15fps` | 80 | `frame_time` | 150 | 29 | -13.3 | 13.0 | 14.0 |
| `15fps` | 100 | `interpolated` | 150 | 150 | -5.7 | 9.0 | 21.0 |
| `15fps` | 100 | `frame_time` | 150 | 75 | -33.3 | 33.0 | 34.0 |
| `15fps` | 150 | `interpolated` | 150 | 150 | -3.3 | 4.0 | 11.0 |
| `15fps` | 150 | `frame_time` | 150 | 150 | -67.3 | 83.0 | 84.0 |
| `15fps` | 200 | `interpolated` | 150 | 150 | -2.3 | 2.0 | 7.0 |
| `15fps` | 200 | `frame_time` | 150 | 150 | -66.7 | 67.0 | 67.0 |
| `15fps` | 300 | `interpolated` | 150 | 150 | -1.5 | 1.0 | 4.0 |
| `15fps` | 300 | `frame_time` | 150 | 150 | -66.7 | 67.0 | 100.0 |
| `15fps` | 400 | `interpolated` | 150 | 150 | -1.1 | 1.0 | 3.0 |
| `15fps` | 400 | `frame_time` | 150 | 150 | -66.7 | 67.0 | 67.0 |

진폭과 속도(보간하는 검출기, 연속 파형 대비). 진폭의 비 = 검출기의 `amplitude` / 파형의 진폭, 속도의 비 =
`peak_closing_velocity`, `peak_opening_velocity` / 연속 파형의 최대 속도(`velocity_valid`인 깜빡임만):

<!-- check:blink-sim-ratios -->

| 조건 | 진폭의 비 | 감는 속도의 비 | 뜨는 속도의 비 | 속도가 유효한 깜빡임 | 검출된 깜빡임 |
|------|-----------|----------------|----------------|----------------------|---------------|
| `30fps` | 0.985 | 0.888 | 0.964 | 1050 | 1050 |
| `15fps` | 0.960 | 0.752 | 0.898 | 951 | 951 |
| `15fps_drop20` | 0.952 | 0.702 | 0.856 | 3455 | 3502 |

이 모사에서 읽을 수 있는 것: 보간을 하면 15 fps에서도 길이의 치우침이 몇 ms 안이고, 프레임 시각만 쓰면 프레임 간격
하나쯤(약 67 ms) 짧게 읽힌다. 60 ms 깜빡임은 15 fps에서 세 번에 한 번꼴로만 검출된다. 감는 속도는 프레임률이 낮을수록
작게 읽힌다. 실제 기기에서는 MediaPipe의 시간 평활과 LIVE_STREAM의 프레임 누락 때문에 짧은 깜빡임의 진폭과 속도가 더
작게 읽힐 수 있다(PLAN 2.1절, 2.6절). 그 크기는 재지 못했다. 학습은 15 fps로 다시 추출한 특징을 함께 쓴다(PLAN 5.2절).

---

## 18. 알려진 교란과 거치 위치

크기에 대한 조사 결과는 PLAN 2.1절, 2.2절, 4.3절에 있다. 여기에는 이식 코드의 동작으로 드러나는 것만 적는다.

| 교란 | 무엇이 틀어지나 | 지금의 대응 |
|------|-----------------|-------------|
| 아래 시선(계기판, 휴대전화) | EAR이 줄고 `eyeBlink`가 올라 closure가 오른다. PERCLOS와 감김 규칙이 반응할 수 있다 | 규칙용 임계(0.7 / 0.5)를 특징용(0.5 / 0.3)보다 높게 둠. pitch 보정은 없다 |
| 고개 숙임(아래 pitch) | 위와 같다. `pitch_rel > 18`이면 시선 규칙도 영역 밖으로 본다 | 없음. 문서화만 |
| 하품 | 하품 중에 EAR이 내려가 closure가 오른다 | 없음. 하품 수는 별도 특징(#12) |
| 고개 돌림(yaw) | 먼 쪽 눈의 EAR이 부풀어 감김이 작게 읽힌다 | 먼 쪽 눈 가중(9장). `abs(yaw_rel) > 40`이면 closure 무효 |
| 미러·계기판 확인 | 머리 자세만 보므로 주행에 필요한 시선도 영역 밖이다 | 없음(범위 밖). 250 ms 디바운스와 3초 기준만 있다 |
| 눈만 움직이는 시선 이탈 | 머리 조건으로는 놓친다 | 눈동자 보조(`source = eye`). 임계값은 공개 근거가 없는 잠정값이다 |
| 운전자가 옆으로 옮겨 앉음 | 얼굴이 프레임 안에서 옮겨 가면 `yaw_rel`, `pitch_rel`이 변한다 | 없음. 기준선은 세션당 한 번만 잡는다 |
| 정면에서 벗어난 거치 | 기준선의 `abs(yaw0) > 15`이면 먼 쪽 눈 규칙이 맞지 않을 수 있다 | `status = off_axis`로 표시만 한다. 계산은 같다 |
| 세션 시작 때 졸리거나 딴 곳을 봄 | 기준선이 틀어져 closure와 상대 자세가 모두 치우친다 | 없음. 가정으로 문서화(8.3절) |
| 안경, 가는 눈 | 조사에서 확인하지 못했다 | E 범위 검사(0.12~0.45)에 걸리면 연장·대체 |
| 좌우 반전, 회전 옵션, 프레임 번호 타임스탬프 | 1장 | 앱이 보장 |
| GPU delegate | CPU와의 수치 차이를 재지 못했다 | 앱 팀이 실기기에서 벡터 밖의 확인을 한다 |

거치 위치: 카메라는 운전자 정면에서 좌우 15도 이내(`baseline.off_axis_yaw_deg`(15.0))를 권한다. 얼굴이 프레임에서 너무
작으면 검출되지 않는다(PLAN 1.3절). 시선 규칙은 실내 영상 데이터로 검증할 수 없어 합성 벡터로만 확인했다.

---

## 19. 잠정 상수

"잠정"은 공개 근거가 없거나 실제 눈 감김으로 확인하지 못한 값이다. 실제 데이터를 본 뒤 **한 번** 확정한다(PLAN 11장 E0).
`pipeline` 블록의 값이 바뀌면 `spec_version`이 1이 되고 `pipeline_hash`와 벡터가 바뀐다. `rules.json`의 값이 바뀌면
`rules_hash`가 바뀐다.

| 구분 | 상수 | 상태 |
|------|------|------|
| closure 식 | `closure.rho`, `closure.b_closed` | 잠정 |
| 감김 임계 | `run_detector.enter_thr`, `run_detector.exit_thr`, 규칙용 `closure.detector.enter_thr`, `closure.detector.exit_thr`, `step_features.perclos_thr` | 잠정 |
| 전역 상수 | `global_constants.e`, `global_constants.b`, `global_constants.jaw0` | 설계값. E0에서 영상별 통계의 중앙값으로 한 번 바꾼다 |
| 하품, 끄덕임 | `yawn`과 `nod` 블록 전체 | 잠정 |
| 전방 주시 | `forward_zone` 블록 전체 | 잠정 |
| PERCLOS 졸음 판정 | `drowsiness.perclos` 블록 | 잠정 |
| 분석 불가 | `unable_to_analyze` 블록 | 잠정(설계값) |
| 경고 정책 | `events.<type>`의 `severity`, `cooldown_ms`, `repeat_ms` | 설계값 |
| 졸음 판정원 | `drowsiness.source` | 실제 데이터의 교차검증이 기준을 통과할 때까지 `perclos` |
| 표준화 | `features[j].mean`, `features[j].std`, `standardization.temperature` | 학습으로 정해진다. v0은 0.0, 1.0, 1.0 |

계획서와 설정 파일이 잠정으로 표시하지 않은 것: `clock`, `baseline`, `closure.w`와 먼 쪽 눈 가중,
`gaze_away_long.onset_ms`, `vats.window_ms`, `vats.cumulative_ms`, `closure.microsleep_ms`, `closure.sleep_ms`,
`drowsiness.model`(사전 고정), `events.<type>.min_speed_kmh`. 특징의 목록과 순서(11.4절)도 고정이다. E0의 통과 기준에
미달하면 표시되지 않은 상수도 그때 한 번 고칠 수 있다(PLAN 11장).

---

## 20. 소유자의 결정을 기다리는 동작

코드가 하는 대로 위에 적었고 벡터도 그대로 고정한다. 결정이 지금 동작과 다르면 코드, 벡터, 이 문서가 함께 바뀐다. 그전에는
**코드대로** 옮긴다. 행 번호는 2026-11-14의 이식 대상 모듈 기준이다. 잰 값은 모두 합성 입력으로 참조 구현을 돌린 것이다.

2026-11-13에 결정되어 이 목록에서 빠진 것: 감김의 길이(10.4절), 막힌 경고를 뒤늦게 내는 것(13장), 한 프레임의 여러
스텝과 모델 호출(7.2절), 10초 넘는 끊김 뒤의 분석 불가와 `drowsy`(12.7절). 검토에서 결함으로 판정해 고친 것: `vats` end의
`confidence`(12.4절), 감는 속도의 거슬러 보기(10.5절 11번), 보간 시각의 원점 의존(10.2절), 인정 구간에 프레임이 없을 때의
`gaze_away_long`(12.4절).

2026-11-14에 결정되어 이 목록에서 빠진 것(이 장의 1번이던 질문. 번호는 비워 둔다): **반쯤 뜬 눈꺼풀은 감김을 끝내지
않는다.** 소유자의 답은 "지금대로 둠"이다. 감김의 길이는 closure가 진입 임계 이상이던 시간의 합이지만(2026-11-13 결정)
구간 자체는 closure가 이탈 임계(규칙용 0.5, 특징용 0.3) 아래로 내려가야(또는 끊겨서) 끝난다. 그래서 눈꺼풀이 이탈
임계와 진입 임계(규칙용 0.7, 특징용 0.5) 사이에 머무는 동안에는 여러 깜빡임이 한 감김으로 이어지고, 진입 임계 이상이던
시간이 그 감김의 길이에 계속 쌓인다. 코드, 벡터, 두 해시는 그대로이고, 이식 코드는 이 누적을 그대로 재현해야 한다
(10.5절 7번, 12.5절). 알려진 한계이고, 잰 사례(합성 입력, 규칙 엔진)는 이렇다: 눈꺼풀이 0.6에 머문 채 보통의
깜빡임(0.9, 165 ms)을 4초마다 하면 `microsleep`이 24.1초(여섯 번째 깜빡임), `sleep`이 72.0초(열여덟 번째 깜빡임)에
난다. 이 값은 프레임 시각을 33 ms 간격으로 둔 것이고, 정확한 30 fps 타임스탬프로 두면 `sleep`이 68.1초(열일곱 번째
깜빡임)에 난다. 2026-11-13 결정 전에는 둘 다 두 번째 깜빡임인 8.0초에 났다. `sleep`이 시작된 뒤에는 눈꺼풀이 머무는
동안 `sleep` 경고가 3초마다 되풀이된다(33 ms 시계의 3분 각본에서 36번). 스텝 특징에서는 그 깜빡임들이 이벤트
하나다(`blink_rate_60`이 1을 센다). 함께 제시했지만 고르지 않은 대안은 "closure가 진입 임계 아래에 `hold_ms`(500)보다
오래 머물면 구간을 끝낸다"였다. 그 대안으로 바뀌었을(`runs_multi_span_truncated`는 대안의 정의에 따라) 벡터
여섯(`runs_lingering_lids`, `runs_multi_span`, `runs_multi_span_truncated`, `steps_multi_span_runs`,
`rules_closure_dwell_below_1s`, `rules_closure_dwell_reaches_1s`)은 이제 결정된 동작을 고정한다(16.1절). 저장소의 단위 테스트 `test_decided_case_of_lids_hovering_between_the_two_thresholds`
(`tests/test_rules.py`)가 위의 잰 사례를 고정한다. (`drowsy/blink.py` 252~257행, 181행)

정비에서 정한 것(결정을 기다리는 질문이 아니다. 구현하면서 정한 동작의 전체 목록과 되돌리는 방법은 README의
"구현하면서 정한 것" 절에 있다):

- **끊긴 동안 발행되는 스텝은 끊기기 직전의 진행 중 감김을 쓴다**(11.6절, 벡터 `steps_closure_across_gap`). 검토에서
  바꾸지 않고 문서로 남기기로 정한 동작이다. `perclos_10`이 끊긴 동안 올라가는 것이 문제가 되면
  `perclos_min_valid_ms_10`을 E0에서 올리는 방법이 있다(상수 변경이라 `pipeline_hash`가 바뀐다).
  (`drowsy/step_features.py` 478행)
- **감는 속도의 거슬러 보기는 쌍의 간격만으로 끊는다.** 직전 유효 표본과의 간격이 `max_dt_ms`를 넘는 쌍(쓸 수 없는
  쌍)에서만 끊고, 두 유효 표본 사이에 무효 프레임이 끼어 있어도 간격이 `max_dt_ms` 이하이면 끊지 않는다(10.5절 11번,
  벡터 `runs_velocity_after_gap`의 다섯째 구간). 검토에서 나온 고침(거슬러 보기를 끊김에서 끊는다)을 구현하면서 정한
  동작이다. (`drowsy/blink.py` 85~86행, 205~208행, 220~224행)

기다리는 질문은 아래 열하나(2~12번)다. 1번은 2026-11-14에 소유자가 결정해 비워 두었다(결정 기록은 이 장 머리의
"2026-11-14에 결정되어 이 목록에서 빠진 것"). 나머지 항목은 번호를 바꾸지 않는다. 항목에 적은 "제안된 대안"은 README
"소유자 결정 대기" 절의 같은 번호 항목에 있는 것과 같은 내용이다. 어느 쪽을 추천하는지와 그 근거는 README의 그 절에만
적는다.

2. **무효 프레임 뒤의 첫 유효 프레임에서 눈을 뜬 깜빡임은 잘린 구간이 된다.** `HOLD`에서 돌아온 표본이 이탈 임계
   미만이면 끊긴 시간이 짧아도 `truncated = true`라서 깜빡임 수와 길이 특징(#4~#8)에서 빠진다. 같은 자리의 프레임이
   그냥 빠졌을 때는 보통의 구간이다(10.5절 3번). 제안된 대안: 마지막 유효 표본에서 `max_dt_ms`(200) 안에 온 열린
   프레임이면 프레임이 빠진 경우처럼 보통의 구간으로 끝낸다. (`drowsy/blink.py` 262행)
3. **감긴 모습을 보지 못한 구간도 감는 속도를 가질 수 있다.** 진입 쌍이 쓸 수 없는 쌍이어도 구간 안에서 정점까지 오른
   쌍이 있으면 그 증가율(대개 잡음 크기)이 `peak_closing_velocity`가 되고 `velocity_valid = true`다(10.5절 12번).
   "`velocity_valid`에 쓸 수 있는 진입 쌍을 요구할 것인가"가 질문이다. 제안된 대안: 진입 쌍이 쓸 수 없는 쌍이면
   `velocity_valid = false`로 둔다. (`drowsy/blink.py` 87행, 319행)
4. **기준선은 두 눈 모두에 쓸 수 있는 표본이 하나 이상 있어야 완성된다.** 한쪽 눈이 수집 구간 내내 쓸 수 없으면 180초에
   전역 상수로 대체되고, 쓸 수 있는 표본이 하나뿐이어도 그 값이 그 눈의 E가 된다(8.3절). 제안된 대안: 한 눈으로도
   완성하거나, 눈마다 최소 표본 수를 둔다(이미 있는 `baseline.min_face_frames`를 쓰면 새 상수가 없고, 새 상수를 두면
   `pipeline_hash`가 바뀐다). (`drowsy/calibrate.py` 240행)
5. **한 프레임만 본 영역 밖(또는 안) 관측도 얼굴 소실의 인정 구간으로 확정된다.** 영역 밖 프레임 하나 뒤에 얼굴이 3.5초
   사라지면 `gaze_away_long`이 난다(`confidence`가 0에 가깝다). 빠른 고개 돌림을 놓치지 않으려는 가정이다(12.4절).
   제안된 대안: 확정에 실제로 관측한 250 ms를 요구한다(그러면 아주 빠른 고개 돌림을 놓친다).
   (`drowsy/rules.py` 187행, 296행)
6. **5.2초를 넘고 10초 이하인 끊김 뒤에는 `unable_to_analyze`가 시작된다.** 10초를 넘는 끊김은 다시 시작하도록
   결정됐지만(12.7절) 10초 이하는 그대로다: 끊기기 전에 얼굴이 계속 보였어도 다음 프레임에서 경고와 함께 시작되고 얼굴
   시간이 5초 쌓일 때까지 시선·감김 규칙이 멈춘다(12.3절, 벡터 `rules_gap_unable_boundary`, `rules_gap_exactly_10s`).
   잰 사례(30 fps): 30초 동안 눈을 뜨고 있다가 8초 끊기고, 끊김 뒤 첫 프레임부터 5초 동안 눈을 감는다. 첫 프레임에서
   `unable_to_analyze`가 경고와 함께 시작되어 5000 ms 뒤에 끝나고, 눈을 감고 있던 그 5초 동안 `microsleep`도 `sleep`도
   나지 않는다(같은 장면에서 끊김이 12초이면 1000 ms에 `microsleep`, 3000 ms에 `sleep`이 난다).
   이 범위도 판정에서 뺄 것인가가 남은 질문이다. 제안된 대안: 이 범위의 끊김 뒤에도 다시 시작하거나, 데이터 없는 시간을
   얼굴 없는 시간으로 세지 않는다. (`drowsy/rules.py` 44~47행, 659~670행)
7. **`unable_to_analyze`와 `vats`에는 종료 쪽 여유가 없다.** 시작과 종료가 같은 임계(비율 0.5, 누적 10초)라 임계
   근처에서는 onset과 end가 번갈아 난다. 경고는 재경고 대기가 막는다. 제안된 대안: 종료 임계를 따로 둔다(예: 얼굴 시간
   비율 0.6 이상, 영역 밖 누적 9초 미만). 규칙 설정에 새 값이 생기므로 `rules_hash`와 `rules.json`, rules 묶음의 벡터가
   바뀐다. (`drowsy/rules.py` 663~670행, 721~729행)
8. **2026-11-13 결정의 귀결(확인용).** (a) 재경고 대기(30초) 중에 시작된 `unable_to_analyze`는 대기가 풀릴 때 상태가
   이어지고 있으면 그때 경고가 나간다(벡터 `rules_unable_sparse_frames`). (b) 미뤄 둔 `gaze_away_long`·`vats` 경고는
   얼굴이 보이지 않는 프레임에서 나갈 수 있다(두 상태는 분석 불가 중에도 끝나지 않는다). (c) 미뤘다 내는 경고의
   `confidence`와 `provisional`은 onset 때의 값이다. (d) 프레임이 끊겼다가 온 프레임 하나에서 추론을 최대 10번 한다.
   (e) 미뤄 둔 경고는 그 **상태**가 끝날 때까지 남으므로, 운전자가 앞을 본 지 한참 뒤에 나갈 수 있다. 잰 사례(10 fps):
   10 km/h에서 1.0~12.0초 동안 옆을 보고 12.0초부터 앞을 본다. `vats`는 11.0초에 시작해 영역 밖 누적이 30초 창에서
   빠지는 32.1초까지 이어진다. 25.0초에 속도가 30 km/h가 되면 그 프레임에서 `vats` 경고가 나간다: 앞을 본 지 13초
   뒤다. (f) 정확히 10000 ms의 끊김은 초기화가 아니므로(12.7절), 끊기기 전에 속도에 막혀 있던 `gaze_away_long`·`vats`·
   `drowsy` 경고가 끊김 뒤 첫 프레임에서, 운전자가 앞을 보고 있어도, `unable_to_analyze` 경고와 함께 나갈 수 있다.
   (e), (f)에 제안된 대안: 영역 안이 확정되면 미뤄 둔 `vats`·`gaze_away_long` 경고를 지우거나, 조건이 그 프레임에도
   성립할 때만 미뤄 둔 경고를 낸다. (`drowsy/rules.py` 946~957행, `drowsy/pipeline.py` 195~201행)
9. **인정 구간 뒤의 `gaze_away_long` 조건에 구현이 더한 것.** 검토의 고침 계획은 "영역 밖 상태이고 표지가 OUT이거나
   분석 불가가 아닌 UNKNOWN"이었다. 구현은 여기에 "영역 안 후보가 걸려 있지 않다"를 더했다(12.4절, 벡터
   `rules_gaze_credit_boundary`의 첫 장면). 이 조건이 없으면 0.4초 이탈 뒤 앞을 본 프레임 하나, 그리고 정확히 3200 ms
   뒤의 얼굴 없는 프레임에서 잘못된 onset이 난다. 그 대신 1 ms 차이로 결과가 갈리는 자리가 생긴다. 잰 경계(10 fps):
   1000~3900에 영역 밖(3초가 4000에 찬다), 4000에 앞을 본 프레임 하나, 다음 프레임이 얼굴 없이 정확히 7200(인정 구간이
   끝나는 시각)에 오고 그 뒤로 앞을 보면 이벤트가 없다. 그 프레임이 7201에 오면 그 프레임에서 `t_start_ms` 1000인
   `gaze_away_long` onset과 경고가 난다. 제안된 대안: 인정 구간의 끝이나 그 뒤에 온 관측 없는 프레임에서는 걸린 후보를
   그 프레임에서 버리고, 이미 인정된 시간으로 그 프레임에서 판정한다. 이 대안으로 바뀌는 것: 12.4절의 의사 코드와
   설명, 벡터 `rules_gaze_credit_boundary`의 설명 문장(첫 장면. 파일은 다시 만들어지지만 `expected`는 같다). 저장된
   기대값은 하나도 바뀌지 않는다: 대안을 참조 구현의 사본에 넣어(`drowsy/rules.py` 262행의 조건에 "관측이 없고
   `t >= cover_end`"를 더했다) 지금의 벡터에 돌리면 102개가 모두 그대로 통과한다. 위의 7200 / 7201 ms 경계는 어느 벡터도
   지나지 않기 때문이고(지금은 단위 테스트 하나만 고정한다. 16.1절), 대안을 고르면 그 경계를 고정하는 벡터를 새로 더해야
   한다. (`drowsy/rules.py` 438~440행, 697행)
10. **feature_spec 사본의 이름.** 계획서 4.7절과 7장은 `export/feature_spec.json`이라고 적었고 지금 내보내기가 쓰는 이름은
    `export/feature_spec_v{N}.json`이다(14.3절). 이 이름으로 확정할지는 소유자의 확인을 기다린다.
    (`drowsy/export_onnx.py`의 `EXPORTED_SPEC_PATTERN`)
11. **v0 모델의 `commit` 메타데이터.** 저장소에 커밋이 없을 때 내보낸 `drowsiness_v0.onnx`는 `commit`이
    `uncommitted-dirty`다(14.3절). 첫 커밋 뒤에 v0을 다시 내보낼지(모델 파일의 sha256과 벡터 `model_v0_inputs`,
    `e2e_250s`의 `model.sha256`이 바뀐다), v1부터 해시를 넣을지를 기다린다.
12. **테스트 벡터의 형식과 크기.** 지금은 JSON 파일 102개, 합계 30,588,827 bytes다(`python -m drowsy.vectors generate`가
    쓴 크기. 처음에는 43개, 약 18 MB였고 두 번째 검증 전에는 68개, 22,650,508 bytes, 세 번째 검증 전에는 81개,
    26,908,941 bytes, 네 번째 검증 전에는 89개, 27,983,687 bytes였다). 다시 만들 때마다 그만큼을 새로 쓴다. 스텝마다 모델을 부르게 되면서 `expected.step_probs`
    항목이 형식에 더해졌는데(16장), 이것은 구현하면서 정한 것이고 형식 전체에 대한 결정은 아직 없다. 형식을 바꾸거나 긴
    세션을 줄일지는 정해지지 않았다. 함께 정할 것: 표시용 상태 `gaze_zone`(12.10절)을 벡터에 담을지. 지금 형식에는
    이 값을 적을 항목이 없어서 벡터로는 `gaze_zone`이 틀려도 드러나지 않고 저장소의 단위 테스트만 고정한다(16.1절).
    담으려면 `expected`에 프레임마다의 항목을 하나 더해야 하는데, 형식을 바꾸는 일이라 하지 않았다.

이 장은 이식 코드와 앱이 받는 파일에 걸린 질문만 적는다. 학습, 평가, 영상 추출 쪽에서 기다리는 결정(재학습 길이와 후보
점수의 규칙, 모델을 기본 졸음 판정원으로 둘지 가리는 판정에서 지표를 fold 평균으로 볼지 합쳐서 볼지, 다른 정규화 조건으로
돌린 실행을 부가로 둘지, 부가 실행에서 선택을 다시 할지, 회전을 가르지 못한 영상을 어떻게 확인할지)까지 담은 전체 목록은
README의 "소유자 결정 대기" 절에 있다.
