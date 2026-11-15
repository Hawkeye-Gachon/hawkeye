# 구현 계획서: 운전자 졸음 판정 모델과 ONNX 내보내기

상태: **승인됨 (v0.3, 2026-11-13)**. 1장의 추천안(Q1~Q8, D1~D16) 전부로 2026-11-13에 승인됐다. 구현하면서 달라진 점(구현하면서 정한 차이, 소유자의 답을 기다리는 질문, 소유자가 2026-11-13에 따로 정한 네 가지 동작과 2026-11-14에 정한 한 가지)은 14장에 있다. 승인한 본문은 고치지 않았고, 그 네 가지 결정으로 맞지 않게 된 4.4, 4.5, 8.1, 8.2절의 문장만 그 자리에서 고치고 "(2026-11-13 소유자 결정으로 변경, 14장)"을 붙였다. 2026-11-14의 결정(반쯤 뜬 눈꺼풀은 감김을 끝내지 않는다, "지금대로 둠")은 동작을 바꾸지 않았으므로 본문 문장은 그대로이고, 8.1절의 눈 감김 항목에 그 사실을 알리는 괄호 하나만 더했다. 2026-11-15에는 구현과 다르게 읽히던 본문 두 곳에 "(구현 메모, 2026-11-15)"로 시작하는 괄호를 붙였다(3.1절의 수학 함수 목록, 4.4절 표의 기준선 수집 행). 본문 문장은 그대로다. 14장은 2026-11-15의 코드 기준이다(4차 검증 뒤의 정비까지 반영). 1장의 Q·D 번호는 이 문서의 질문과 결정 번호이고, 14장의 O·G·R 번호는 정비 작업의 지시 번호라 14장에서 행마다 뜻을 적었다. 3차 검증 뒤와 4차 검증 뒤의 정비는 지시 번호 없이 무엇을 바꿨는지로 적었다. 14장에 나오는 `검토 <영역>-NN`(예: `검토 rules-06`)과 K1~K8은 2026-11-13 독립 검토의 지적 번호다. 그 검토 보고서는 저장소에 없으므로 번호 옆에 지적의 내용을 말로 적었다.

이 문서는 조사로 확인한 사실, 결정이 필요한 질문, 설계 제안, 모듈과 함수 시그니처, 특징 목록, 실험 순서를 담는다.

- "검증됨"은 1차 출처를 직접 읽었거나 스크래치 환경에서 실행해 재현한 것, "추정"은 2차 출처나 한 번의 관찰에 근거한 것이다.
- "잠정"이라고 적힌 설계 상수는 공개 근거가 없거나 실제 눈 감김으로 확인하지 못한 값이다. 실제 데이터를 본 뒤 한 번 확정한다(11장 E0).
- v0.3은 독립 검토 5건(명세 누락, 평가 타당성, 이식 가능성, 사실 일치, 내부 모순)의 지적을 반영했다.

---

## 1. 결정이 필요한 사항

### 1.1 답이 필요한 것

여기 항목은 답을 받은 뒤에 해당 작업을 시작한다. **Q4~Q7은 구현 시작 전에, Q1은 의존성 고정 전에, Q2·Q3·Q8은 실제 데이터 추출(M3) 전에** 필요하다. Q5와 Q6은 계약 변경이므로 답이 없으면 바꾸지 않는다(이벤트 필드 6개, 메타데이터 키 7개 그대로).

| # | 질문 | 추천안 | 배경 |
|---|------|--------|------|
| Q1 | Python `mediapipe` 버전을 무엇으로 고정할까. Android 앱의 `tasks-vision` 버전은 무엇인가. | `mediapipe==0.10.32` | 최신 1.2.3은 이 PC에 없는 시스템 라이브러리 `libGLESv2.so.2`가 필요하고(`sudo apt install libgles2`), 0.10.35부터는 실행 시 `play.googleapis.com` 조회를 시도한다. 0.10.32는 둘 다 없다. 버전 간 수치 차이는 IMAGE 모드에서 랜드마크 0.003 px 이하였다. VIDEO 모드는 0.10.21 대 1.2.3에서만 쟀고 최대 0.56 px·블렌드셰이프 0.016이었다. 0.10.32의 VIDEO 모드 비교는 M0에서 한 번 잰다. |
| Q2 | 앱이 쓰는 `face_landmarker.task`가 공개 표준 파일(float16 v1, sha256 `64184e22…bc9ff`)과 같은가. 파일 경로는 어디인가. | 표준 파일로 가정하고, 추출 시작 시 sha256을 검사한다 | 학습 특징 추출이 같은 파일을 써야 한다. 조사는 공개 표준 파일로 했다. |
| Q3 | 앱이 MediaPipe에 넘기는 프레임 전제를 아래처럼 명세에 고정해도 되는가. | (1) 비트맵을 똑바로 세워서 넘긴다(`rotationDegrees` 옵션 대신 비트맵 회전). (2) 좌우 반전하지 않는다(미리보기만 반전). (3) `numFaces=1`. (4) 타임스탬프는 양수이고 엄격히 증가하는 실제 ms(카메라 프레임 시각). | 공식 Android 샘플은 전면 카메라 프레임을 반전하고 `uptimeMillis()`를 넘긴다. 반전하면 좌우 신호와 yaw·roll 부호가 뒤집힌다. `rotationDegrees` 옵션을 쓰면 블렌드셰이프가 최대 0.19~0.43 달라진다. `numFaces`가 1이 아니면 시간 평활이 꺼진다. 모델 특징은 좌우 대칭으로 설계해 반전에는 영향받지 않게 한다(3.2절). |
| Q4 | 첫 졸음 확률이 세션 시작 약 4분 뒤에 나와도 되는가. | 4분 (기준선 60초 + 조회 구간 60초 + 창 120초). 기준선이 연장되면 최대 약 6분 | 3분으로 줄이려면 조회 구간이 덜 찬 스텝을 모델에 넣어야 해서 학습과 이식이 복잡해진다. 규칙(마이크로슬립·수면·시선 이탈)은 이 시간을 기다리지 않는다(4.4절). |
| Q5 | `DriverStateEvent`에 필드 두 개를 추가해도 되는가: `phase`(`onset`/`end`), `provisional`(참/거짓). | 추가 | 경고는 상태가 진행 중일 때(임계 시간 도달 시점) 나가야 하고 상태 종료도 알려야 한다. 지정하신 6개 필드만으로는 시작과 끝을 구분할 수 없다. `provisional`은 기준선 완성 전에 임시 상수로 판정한 이벤트를 표시한다(D4). |
| Q6 | 전처리를 ONNX 그래프 안에 넣고, `metadata_props`에 키 세 개를 추가해도 되는가. | 그래프 안에 포함: 일부 특징의 `log1p`, 표준화, ±5 자르기, 온도, 소프트맥스. 추가 키: `standardization=in_graph`, `pipeline_hash`, `mediapipe_model_sha256` | 그래프 안에 넣으면 앱은 원시 스텝 특징만 넣으면 된다. 앱이 한 번 더 표준화해도 오류가 나지 않으므로 메타데이터와 SPEC.md에 명시한다. 표준화와 소프트맥스의 실측 비용은 2~3 µs, 약 0.5 KB였다(`log1p`와 자르기는 M1에서 측정). |
| Q7 | 이 폴더(`hawkeye/`)를 저장소 루트로 삼고 `git init`을 해도 되는가. | 예. `hawkeye/pyproject.toml`, 패키지는 `hawkeye/drowsy/` | 메타데이터의 commit hash에 git 저장소가 필요하다. |
| Q8 | UTA-RLDD 사본이 이미 있는가. 공식 Drive zip인가 Kaggle 미러인가. | 경로를 받으면 M3부터 진행, 그전에는 합성 데이터로 M0~M2 완료 | 폴더 깊이와 이름 규칙이 사본마다 다를 수 있다. 로더는 둘 다 받도록 만든다. |

### 1.2 추천안으로 진행하되 바꾸고 싶으면 알려주실 것

답이 없으면 추천안으로 진행하고 README와 코드 주석에 가정으로 남긴다.

| # | 항목 | 추천안 |
|---|------|--------|
| D1 | 정규화 비교 실험에 세 번째 조건 추가 | 지정하신 두 조건(alert 영상 기준선, 전역 정규화)에 "각 영상 자신의 앞 60초 기준선"을 추가한다. 앱의 실제 동작(같은 세션 안에서 보정)과 같다. 내보낼 모델은 지정하신 alert 영상 기준선 조건으로 학습하고, 그 모델을 앱 조건(자기 세션 기준선)으로도 평가해 두 값을 함께 싣는다(6장). |
| D2 | 2~3초 감김의 처리, 수면 판정 비교 연산 | 1.0초에 마이크로슬립 시작, 계속 감겨 있으면 3.0초에 수면 시작(같은 `t_start_ms`). 수면은 "3.0초 이상". Euro NCAP 2026이 수면을 3초 이상으로 정의한다. 명세의 "3초 초과"와 다르므로 YAML 값으로 둔다. |
| D3 | 속도 억제 기준 | 시선 이탈·마이크로슬립·수면 경고는 20 km/h 미만에서 억제, 졸음 경고는 50 km/h 미만에서 억제(Euro NCAP 2026 값). 속도 입력이 `None`이면 억제하지 않는다. 억제는 경고에만 적용하고 상태 이벤트는 그대로 낸다. |
| D4 | 기준선 완성 전의 규칙 엔진 | `calibrating` 이벤트를 낸다. 감김 규칙은 전역 상수로 계산한 임시 closure로 동작시키고 이벤트에 `provisional=true`를 단다. 시선 규칙과 모델은 기준선 완성 후 시작한다. |
| D5 | 기준선 연장과 대체 | 명세는 "첫 60초"다. 60초 시점에 얼굴 프레임이 부족하면 30초씩 최대 180초까지 연장하고, 그래도 안 되면 전역 상수로 대체한다(4.2절). 연장된 만큼 4.4절의 시간표가 뒤로 밀린다. |
| D6 | VATS 누적의 초기화 | 명세대로 "최근 30초 중 누적 10초"만 쓰고 초기화는 끈다. Euro NCAP 2023의 "전방 주시 2초면 초기화"는 YAML 선택 값으로만 둔다. 초기화를 켜면 2.9초 이탈과 2.1초 주시를 반복해도 이벤트가 나지 않는다. |
| D7 | 영상 단위 정확도의 분모와 품질 기준 | 분모는 논리 영상 180개 전체(피험자 32, 49의 `10_1`/`10_2`는 한 영상으로 합침). 창이 하나도 안 나오는 영상은 오답으로 센다. 보조 행으로 품질 기준 통과 영상만 따로 싣는다. 품질 기준은 라벨과 무관하게 지금 고정한다: 얼굴 검출 비율(시간 기준) 0.7 이상이고 창이 1개 이상. 같은 모델로 평가 대상만 거른다. |
| D8 | 2클래스(alert vs drowsy) 보고 방식 | 내보내는 3클래스 모델을 그대로 쓰고, alert·drowsy 영상 120개만 대상으로 창마다 `p_alert`와 `p_drowsy` 중 큰 쪽을 고른 뒤 영상별 다수결을 한다. 별도 2클래스 모델은 학습하지 않는다. |
| D9 | "시간당 졸음 오판정 수"의 정의 | 주 수치는 판정 시작 횟수 기준(6장)이다. 창 단위 값(alert 영상에서 drowsy로 분류된 창 수 / 시간)도 함께 싣는다. |
| D10 | 내보내는 모델과 보고 수치의 관계 | 최종 모델은 60명 전체로 학습한다. 따라서 보고하는 수치는 같은 절차의 교차검증 추정치이고, 내보낸 파일 자체를 시험한 값이 아니다. 모델 카드에 그렇게 적는다. |
| D11 | 학습 모델이 약할 때의 기본 동작 | 교차검증 결과(앱 조건 평가 기준)가 다음을 모두 만족할 때만 규칙 엔진의 기본 졸음 판정원을 모델로 둔다: macro-F1이 LightGBM − 0.02 이상, 2클래스 영상 정확도 70% 이상, alert 영상 오경보 시간당 2회 이하이면서 판정이 한 번이라도 난 alert 영상이 10% 이하, drowsy 영상의 50% 이상에서 판정 발생. 아니면 ONNX는 계약대로 내보내되 모델 카드와 README에 "실험용"으로 적고 기본 판정원을 PERCLOS 규칙으로 둔다(`drowsiness.source: model | perclos | either`). 이 기준은 점추정 규칙이다. |
| D12 | 시선 규칙의 눈동자 보조 | 명세대로 `eyeLook` 기반 보조 판정을 켜고 그로 인한 이벤트는 `source=eye`로 구분한다. 임계값은 공개 근거가 없는 잠정값이다. |
| D13 | RLDD 얼굴 이미지 | 데모와 리포트는 신호와 타임라인만 그리고 얼굴 프레임은 저장하지 않는다. 데이터셋 페이지는 60명 중 36명만 얼굴 공개를 허용한다. |
| D14 | 테스트 벡터의 입력 | 실제 얼굴 이미지 없이 합성 랜드마크·신호만 쓴다. |
| D15 | 범위 밖으로 두는 것 | 무반응 운전자(감김 6초 이상), 휴대전화 사용 구분, 주행 관련 시선(미러·계기판)의 예외 처리. |
| D16 | 선택 항목 INT8 양자화 | TCN이 최종 모델이면 만들고 정확도 변화를 보고한다. GRU가 최종 모델이면 만들지 않는다(실측상 동적 양자화가 GRU 연산자를 양자화하지 않아 파일이 오히려 커졌다). |

### 1.3 앱 팀에 전달할 확인 요청 (구현을 막지는 않음)

- 카메라 종류(RGB 전면 카메라인지), 거치 위치, 분석 해상도, 실제 LIVE_STREAM 결과 프레임률. 얼굴 폭이 프레임 긴 변의 약 9% 아래로 내려가면 검출되지 않았다. 거치 위치는 정면에서 좌우 15도 이내를 권장한다(4.3절).
- CPU와 GPU 중 어느 delegate를 쓰는지. GPU 수치 차이는 측정하지 못했다.
- 사용할 `onnxruntime-android` 버전. opset 17, IR 8 파일은 Python ORT 1.15.1~1.31.1에서 모두 로드됐다.
- 대상 기기의 LIVE_STREAM 결과 타임스탬프 로그. 프레임 누락 모사에 실제 지연 분포를 쓰기 위한 것이다.
- UTA-RLDD에는 명시된 라이선스가 없다(논문 인용 조건과 얼굴 공개 제한만 있음). 공모전 규정이 학습 데이터 라이선스를 요구하는지 확인이 필요하다.

---

## 2. 조사로 확인한 사실

조사는 스크래치 폴더에서만 했고 프로젝트 폴더에는 이 문서 외에 쓴 것이 없다. 데이터셋은 받지 않았다.

### 2.1 랜드마크와 블렌드셰이프 (검증됨, 독립 재검증 완료)

| 항목 | 값 |
|------|----|
| EAR 식 | `(|p2-p6| + |p3-p5|) / (2|p1-p4|)`, 픽셀 좌표 `(x*W, y*H)`의 2D 거리 |
| R 집합 `p1..p6` (MediaPipe RIGHT_EYE) | 33, 160, 158, 133, 153, 144 |
| L 집합 `p1..p6` (MediaPipe LEFT_EYE) | 362, 385, 387, 263, 373, 380 |
| 홍채 | R 468(중심), 469~472 / L 473(중심), 474~477. 순서는 중심, +x, 위, -x, 아래 |
| 시선 대용치용 눈 꼬리 | R 바깥 33, 안쪽 133 / L 바깥 263, 안쪽 362 |
| 블렌드셰이프 인덱스 | 0 `_neutral`, 9 `eyeBlinkLeft`, 10 `eyeBlinkRight`, 11~18 `eyeLook{Down,In,Out,Up}{Left,Right}`, 25 `jawOpen` (52개 전체 표는 SPEC.md에 수록) |
| 짝 | `eyeBlinkRight`는 R 집합, `eyeBlinkLeft`는 L 집합과 짝이다. 반전 여부와 무관하다. |
| `eyeLookIn` / `eyeLookOut` | In은 코 쪽, Out은 관자놀이 쪽 |

계획에 영향을 주는 관찰:

- **좌우의 의미**: Left/Right는 반전하지 않은 영상에서 피험자 자신의 좌우다. 좌우 반전 입력은 해부학적 의미를 바꾸고, 회전은 바꾸지 않는다. 180도 뒤집힌 입력은 오류 없이 틀린 결과를 낸다.
- **좌표**: x와 y가 각각 너비와 높이로 정규화된다. 정규화 좌표로 EAR을 구하면 비정사각 프레임에서 값이 틀린다(같은 얼굴에서 0.157 대 0.231).
- **두 신호는 독립이 아니다**: 블렌드셰이프 모델의 입력은 랜드마크 146개의 x, y뿐이다. `eyeBlink`와 EAR은 같은 랜드마크의 두 함수다.
- **`eyeBlink`의 범위**: 뜬 눈에서 0.19~0.30이다. 눈꺼풀이 기하적으로 완전히 닫혀도 윗눈꺼풀이 내려온 경우 0.76~0.82, 양쪽이 가운데서 만난 경우 0.57~0.61, 아랫눈꺼풀만 올라온 경우 0.37~0.38에 그친다. 홍채 위치와 roll에도 반응한다.
- **머리 자세**: 강체 회전 모사에서 yaw 30도일 때 먼 쪽 눈의 2D EAR이 33% 커지고, 아래 방향 pitch 30도에서 22% 작아진다. `eyeBlink`도 아래 pitch에서 올라가고 yaw에서 먼 쪽 눈이 내려간다. 즉 yaw와 pitch는 두 신호를 같은 방향으로 치우치게 한다.
- **시간 평활**: VIDEO와 LIVE_STREAM은 `num_faces=1`일 때 one-euro 필터를 랜드마크에 건다. 짧은 감김의 진폭이 줄고(자극에 따라 56~94%) 뜨는 구간에 여러 프레임의 꼬리가 생긴다. `num_faces=2`면 평활이 꺼진다.
- **얼굴 없음**: 결과 리스트가 빈다. `face_present`는 리스트가 비었는지로만 판정한다.
- 이 실험들은 정면 사진 한두 장과 합성 변형으로 했다. 안경, 가는 눈, 실제 아래 시선, 실제 눈 감김은 검증하지 못했다.

### 2.2 변환 행렬과 각도 규약 (검증됨, 독립 재검증 완료)

- 행렬은 표준 얼굴 모델(cm, +X 피험자 왼쪽, +Y 위, +Z 얼굴 앞쪽)을 카메라 공간(+X 영상 오른쪽, +Y 영상 위, +Z 카메라 쪽)으로 옮긴다. 회전 블록의 스케일은 1이다.
- Android의 `float[16]`은 열 우선이다. Python의 `M[i][j]`와 같은 값은 `m[4*j + i]`다. 행 우선으로 잘못 읽으면 전치 행렬이 되어 각도 부호가 대략 뒤집힌다. Python에서 저장할 때는 `M.T.reshape(-1)`을 쓴다.
- 규약: `R = Ry(yaw) · Rx(pitch) · Rz(roll)`
  - `yaw = atan2(M[0][2], M[2][2])`
  - `pitch = atan2(-M[1][2], hypot(M[1][0], M[1][1]))`
  - `roll = atan2(M[1][0], M[1][1])`
  - 부호(비반전 영상): yaw 양수는 피험자가 자기 왼쪽으로 고개를 돌림, pitch 양수는 턱이 내려감, roll 양수는 영상에서 반시계 방향.
  - 부호는 대수, 수치 왕복, 영상 회전, 원근 변형, 독립적인 solvePnP 다섯 가지로 확인했다.
- 좌우 반전 입력은 yaw와 roll의 부호를 뒤집고 pitch는 그대로 둔다.
- 기준 자세 대비 상대 각도는 오일러 각의 단순 차가 아니라 `Euler(R0ᵀ · R)`로 구한다. 기준선이 (30, 15, 0)도일 때 단순 차는 pitch와 roll에 수 도의 오차를 낸다.
- 절대 pitch는 모델에 따라 달라진다(같은 랜드마크에서 solvePnP와 4도 차이). 절대 임계값을 쓰지 않고 기준선 대비 값만 쓴다.
- yaw와 pitch에는 얼굴이 프레임 안 어디에 있는지에 따른 항이 섞인다(가로 3배 캔버스의 가장자리에서 yaw 약 ±14도). 기준선은 고정된 부분만 없애므로 운전자가 옆으로 옮겨 앉으면 상대 yaw가 변한다.
- 행렬의 평행이동 성분은 가정된 화각과 프레임 크기에 좌우되므로 거리 특징으로 쓰지 않는다.
- 테스트 벡터에는 공식 기대 출력 파일의 행렬(기대 각도 2.24, 6.32, 0.99도)을 쓴다.

### 2.3 VIDEO 모드와 LIVE_STREAM의 일치, 타임스탬프 (검증됨, Android 실기기는 미실행)

- VIDEO와 LIVE_STREAM은 같은 그래프(이전 프레임 기반 추적, 평활)를 쓴다. 데스크톱 CPU에서 같은 프레임과 타임스탬프를 넣으면 결과가 비트 단위로 같았다(1.2.3에서 측정). LIVE_STREAM은 바쁠 때 프레임을 버리는 흐름 제한기만 추가로 가진다.
- IMAGE 모드는 결과가 달라서(랜드마크 최대 8 px) 학습 추출에 쓰면 안 된다.
- 평활 필터는 넘겨받은 타임스탬프로 샘플링 주기를 계산한다. 프레임 번호를 타임스탬프로 넘기면 신호가 달라진다.
- **타임스탬프는 양수여야 한다.** 첫 값이 정확히 0이면 두 번째 프레임이 기본 30 Hz로 평활되고, 음수면 평활이 꺼진다. 오프라인 추출은 `t_ms = 1 + (첫 프레임 대비 ms)`로 시작한다.
- `detect_for_video`는 엄격히 증가하는 정수 ms를 요구하고, 같거나 작은 값에는 `ValueError`를 낸다.
- 타임스탬프는 PyAV의 `pts * time_base`로 얻는다. PyAV는 회전 메타데이터를 적용하지 않으므로 `frame.rotation`을 보고 직접 세운다. OpenCV의 자동 회전은 버전마다 기본값이 달라서 쓰지 않는다. 합성 가변 프레임률 파일에서 오차 0 ms였고, 실제 휴대전화 녹화 파일로는 아직 확인하지 않았다.
- CPU에서 프레임당 추론 약 4 ms였다. 추출은 영상 단위 다중 프로세스로 충분하다.

### 2.4 UTA-RLDD (논문·데이터셋 페이지·공개 코드·미러 파일 목록 기준, 영상 자체는 미측정)

- 피험자 60명, 논리 영상 180개, 영상당 약 10분. 라벨은 본인 보고(alert는 KSS 1~3, low vigilant는 6~7, drowsy는 8~9).
- **공식 5-fold가 있다.** fold k는 피험자 `12(k-1)+1`~`12k`다. GroupKFold 대신 이것을 쓴다. 이 대응은 논문 서술과 미러의 폴더 이름으로 확인했다. 로더는 `FoldK` 폴더 이름이 있으면 그것과 일치하는지 검사한다.
- 파일은 182개다(미러 목록 기준). 피험자 32와 49의 drowsy 영상이 `10_1`, `10_2`로 나뉘어 있다. 확장자는 `.mp4`, `.MP4`, `.mov`, `.MOV`, `.m4v`가 섞여 있다.
- 원 논문은 프레임률이 항상 30 fps 미만이라고 적었다. Wolter 2025의 보고로는 182개 중 169개가 가변 프레임률이고 12~30 fps이며, 일부 영상은 회전 보정이 필요했고, 피험자 23은 얼굴이 작아 제외됐다.
- **65.2%**: 3클래스 영상 단위 정확도, 30회 깜빡임 시퀀스의 다수결, 5 fold 평균. 클래스별 재현율은 0.81 / 0.32 / 0.82, 사람 판정은 57.8%다. 공개 코드는 174개 영상만 평가하는 것으로 보이고 검증 분할이 없다는 점을 비교표 주석에 적는다.
- 피험자 분리 조건의 후속 연구: 3클래스 52.5%·2클래스 80.4%(Wolter 2025), 2클래스 55~63%(Magán 2022). Xu 2025는 86.5%를 보고했지만 관찰자 판정과 라벨이 다른 구간을 버린 뒤의 값이다. 90% 이상 결과는 프레임 단위 무작위 분할에서 나온다.
- 세션 교란 위험: 클래스마다 녹화 세션이 다르고, 60명 중 17명은 세 영상의 파일 크기가 2배 넘게 차이 난다. 모델이 졸음 대신 조명·카메라·프레임률을 배울 수 있다.

### 2.5 도구 체인 (스크래치 환경에서 실행해 확인)

- 함께 설치되어 동작한 고정 조합(Python 3.11.16): numpy 2.4.6, opencv-python-headless 5.0.0.93, av 18.1.0, torch 2.15.0(PyPI 기본 휠이 RTX 5090과 CPU 모두 동작), onnx 1.23.2, onnxruntime 1.31.1, onnxscript 0.7.2, lightgbm 4.7.0, scikit-learn 1.9.1, pandas 3.1.1, pyarrow 26.0.1, matplotlib 3.11.2, pyyaml 6.0.3, pytest 9.2.0. 이 조합은 mediapipe 1.2.3과 함께 검증했다. 0.10.32와의 조합은 M0에서 다시 해석해 확인한다.
- mediapipe가 `opencv-contrib-python`을 끌어와 headless 빌드를 덮어쓴다. uv의 의존성 제외 설정으로 해결됐다.
- 구형 TorchScript 내보내기는 opset 17, IR 8, 단일 파일을 그대로 쓴다. 새 dynamo 내보내기는 opset 17을 요청해도 일부 TCN을 조용히 opset 18로 남기고 IR 10을 쓴다. 내보낸 뒤 opset과 IR을 검사하는 단계가 필요하다.
- 시험 모델(F=24, 무작위 가중치, 그래프 안에 표준화와 소프트맥스 포함): TCN 31,811 파라미터·130~153 KB, GRU 17,475 파라미터·72 KB. PyTorch와 onnxruntime의 최대 절대 오차는 6.6e-7 이하, 단일 스레드 CPU 지연 중앙값은 0.056~0.075 ms였다.
- ORT는 자기 한도보다 높은 IR 버전의 모델을 거부한다. IR 8로 고정하면 시험한 모든 버전에서 로드됐다.
- 입력에 NaN이 하나 있으면 PyTorch GRU는 NaN을 내지만 ORT의 GRU는 그럴듯한 확률을 낸다. 스텝 특징은 모델에 들어가기 전에 반드시 유한값이어야 한다.

### 2.6 규칙 임계값의 근거

- Euro NCAP 2026 운전자 관여 프로토콜 v1.2를 직접 읽었다. 장시간 시선 이탈 3~4초, VATS 30초 중 누적 10초, 마이크로슬립 1~2초, 수면 3초 이상, 경고는 20 km/h부터(졸음은 50 km/h부터). 명세의 값과 맞는다.
- VATS 초기화 시간은 2026년판에서 제조사 선언 값이다. 2023년판은 전방 주시 2초 이상이었다(추정, 원문 PDF는 열리지 않았다).
- 전방 주시 영역의 yaw·pitch 임계값, 눈동자 보조 임계값, 하품, 끄덕임에는 검증된 공개 값이 없다. 기본값은 모두 잠정이며 YAML로 조정한다. 참고로 openpilot은 yaw 약 23도, 아래 pitch 약 18도를 쓴다.
- PERCLOS의 원 정의(P80)는 1분 중 눈꺼풀이 동공을 80% 이상 가린 시간의 비율이다. 기준선 정규화 closure가 0.8 이상인 시간 비율은 그 근사다.
- 속도와 진폭(모사): 빠른 깜빡임의 최대 감는 속도는 30 fps에서 참값의 약 70~79%, 15 fps에서 약 47~53%로 읽힌다. 진폭도 30 fps에서 85~91%, 15 fps에서 78~85%다.

---

## 3. 설계 원칙

### 3.1 코드 경로는 하나

앱이 MediaPipe에서 받는 것과 같은 입력(타임스탬프, 프레임 크기, 랜드마크, 블렌드셰이프, 행렬)을 프레임 하나씩 받아 신호, 정규화, 깜빡임, 스텝 특징, 규칙까지 계산하는 스트리밍 코드를 한 벌만 둔다. 학습 특징 생성, 데모, 테스트 벡터 생성이 모두 이 코드를 호출한다. numpy로 따로 벡터화한 구현을 만들지 않는다.

이식 대상 모듈은 표준 라이브러리만 쓴다. 이식 대상 모듈이 numpy를 import하면 실패하는 테스트를 둔다. 연산은 배정밀도(Kotlin `Double`), 시간은 정수 ms(`Long`), 쓰는 수학 함수는 `sqrt`, `atan2`, `hypot`, `sin`, `cos`뿐이고, float32는 모델 입력 텐서에서만 나온다. 반올림은 `floor(x + 0.5)`로 통일한다. (구현 메모, 2026-11-15: 구현은 `floor`(반올림)와 `ceil`(분위수의 순위)도 쓴다. 목록은 `docs/SPEC.md` 2장의 것이 맞다)

### 3.2 모델 특징은 좌우 대칭

데이터셋 영상이 반전됐는지 알 수 없고 앱이 반전할 수도 있다. 모델 입력은 좌우를 바꿔도 값이 같은 양으로만 만든다. 눈별 값은 원시 신호로만 보관한다.

### 3.3 시간 규칙

- **세션 시간** = `t_ms` − (처음 받아들인 프레임의 `t_ms`). 이벤트의 시각은 원래 `t_ms`로 적는다.
- `t_ms`가 이전 값 이하인 프레임은 버리고 센다.
- 한 프레임의 상태는 다음 프레임까지 유지되는 것으로 보되 최대 200 ms까지만 인정한다. 넘는 부분은 "데이터 없음"이고, 얼굴 없는 시간과 똑같이 취급한다.
- 구간이 스텝 경계에 걸치면 경계에서 정확히 나눈다. 모든 비율은 프레임 수가 아니라 시간으로 계산한다.
- 프레임 간격이 10초를 넘으면 검출기와 누적기를 초기화한다(기준선은 유지). 새 기준선은 앱이 새 세션을 시작할 때만 잡는다.
- 0으로 나누게 되는 비율은 0으로 정의하고, 분산은 제곱근 전에 0 이상으로 자른다.

### 3.4 숫자는 실행 결과만

합성 데이터 결과는 파이프라인 점검용이며 성능으로 보고하지 않는다. reports/의 모든 표는 생성한 명령과 설정 사본을 함께 남긴다.

---

## 4. 신호, 정규화, 특징 정의

### 4.1 프레임 입력과 원시 신호

- `RawFrame`(앱이 받는 것): `t_ms`, `face`, `w`, `h`, 랜드마크 부분집합의 정규화 좌표(눈 윤곽 32점, 홍채 10점, 얼굴 폭 측정용 2점), 블렌드셰이프 이름별 값, 열 우선 행렬 16개 값.
- `FrameSignals`(`signals.compute`가 계산): `t_ms`, `face`, `ear_l`, `ear_r`, `blink_l`, `blink_r`, `eye_look_{in,out,up,down}_{l,r}` 8개, `jaw_open`, `yaw`, `pitch`, `roll`, 눈 폭(픽셀) `eye_w_l`, `eye_w_r`, 선택 항목 홍채 비율 `iris_h_{l,r}`, `iris_v_{l,r}`.
- `_l`은 MediaPipe Left(L 집합과 `*Left` 블렌드셰이프), `_r`은 Right다.
- 눈 폭 `|p1−p4|`가 6 px 미만인 눈은 쓸 수 없는 눈으로 표시하고 EAR과 홍채 비율을 계산하지 않는다.
- 추출 parquet에는 `RawFrame` 내용과 블렌드셰이프 52개 전부를 저장한다. 규약을 바꿔도 MediaPipe를 다시 돌리지 않기 위해서다.
- 홍채 비율: `a = p_inner − p_outer`, `d = p_iris − p_outer`일 때 `h = dot(d,a)/|a|²`, `v = cross(a,d)/|a|²`. 두 눈을 합칠 때는 수평과 수직 모두 한쪽 부호를 뒤집는다.
- 눈동자 합성값(시선 규칙 보조): `gaze_h = ((eyeLookInRight + eyeLookOutLeft) − (eyeLookOutRight + eyeLookInLeft)) / 2`, `gaze_v = (eyeLookUp 두 눈 평균) − (eyeLookDown 두 눈 평균)`. 정면에서도 0이 아니므로 기준선 대비 변화만 쓴다.

### 4.2 기준선 (`Baseline`)

세션 시작 후 60초 동안 얼굴이 검출된 프레임으로 계산한다.

| 통계 | 정의 |
|------|------|
| `E_l`, `E_r` | 각 눈 EAR의 90 분위수(뜬 눈 기준) |
| `B_l`, `B_r` | 각 눈 `eyeBlink`의 10 분위수(뜬 눈 기준). 분포 기록용으로 50, 90 분위수도 저장 |
| `R0` | yaw, pitch, roll 중앙값으로 만든 기준 회전 |
| `jaw0`, `gaze_h0`, `gaze_v0` | 중앙값 |

- 분위수는 최근접 순위(`sorted[ceil(q·n) − 1]`)다. 중앙값만 표본이 짝수 개일 때 가운데 두 값의 평균으로 정의한다. 좌우 반전 입력에서 `R0`이 정확히 대칭이 되게 하기 위해서다.
- 완성 조건: 세션 시간 60초 경과, 얼굴 프레임 300개 이상, 얼굴 검출 비율(시간 기준) 0.5 이상, `E`가 0.12~0.45 안. 만족하지 못하면 30초씩 최대 180초까지 연장하고, 그래도 안 되면 전역 상수로 대체하며 `status=fallback_global`을 표시한다(D5). 대체 상태에서는 `R0`이 없으므로 시선 규칙을 끈다.
- 기준선의 `|yaw0|`이 15도를 넘으면 `status`에 `off_axis`를 표시한다(4.3절의 먼 쪽 눈 규칙이 맞지 않을 수 있다).
- 기준선이 완성된 세션 시간을 `T_b`라 한다. 정상이면 60,000 ms다.
- 가정(README에 명시): 운전 시작 시점의 운전자는 각성 상태이고 전방을 본다.

### 4.3 closure 정의 (제안)

```
눈 e마다:
a_e = clip((E_e − ear_e) / (E_e · (1 − RHO)), 0, 1)              RHO = 0.35       (잠정)
s_e = clip((blink_e − B_e) / max(B_CLOSED − B_e, 0.2), 0, 1)     B_CLOSED = 0.60  (잠정)
c_e = W · a_e + (1 − W) · s_e                                    W = 0.5

(yaw_rel, pitch_rel, roll_rel) = Euler(R0ᵀ · R)
w_far   = clip((25 − |yaw_rel|) / 10, 0, 1)        먼 쪽 눈: yaw_rel > 0이면 L, 아니면 R
closure = (c_near + w_far · c_far) / (1 + w_far)
쓸 수 없는 눈이 하나면 나머지 눈의 c_e를 그대로 쓴다.
유효 조건: 얼굴 있음, 쓸 수 있는 눈이 하나 이상, |yaw_rel| ≤ 40도
```

- 뜻: EAR 쪽은 "뜬 눈 기준 대비 얼마나 좁아졌나"를, 블렌드셰이프 쪽은 "뜬 눈 수준에서 감긴 수준까지 얼마나 갔나"를 0~1로 만든 뒤 평균한다.
- 평균을 고른 이유: 두 신호는 같은 랜드마크에서 나오므로 독립 측정이 아니다. 평균이 도움이 되는 것은 roll, 홍채 위치, 눈꺼풀이 움직이는 방식처럼 두 신호가 다르게 반응하는 경우뿐이다. yaw와 pitch는 둘을 같은 방향으로 치우치게 하므로 평균으로 줄지 않는다. `W`를 0, 0.5, 1로 바꾼 결과를 E1에서 보고하되, v1에서는 `W = 0.5`를 바꾸지 않는다.
- 먼 쪽 눈의 비중을 줄이는 이유: 고개를 돌리면 먼 쪽 눈의 EAR이 부풀어 감김이 작게 읽힌다. 15도까지는 두 눈 평균, 25도부터는 가까운 눈만 쓴다. 반전 입력에서는 yaw 부호와 눈 집합이 함께 바뀌므로 결과가 같다.
- 유효하지 않은 프레임의 closure는 "없음"이다. NaN을 쓰지 않고, 보간하거나 이전 값을 이어 쓰지 않는다.
- `RHO`, `B_CLOSED`와 아래에 나오는 감김 임계값(0.5, 0.7, 0.8)은 실제 눈 감김으로 검증되지 않았다. 조사에서 합성으로 감긴 눈 두 장에 이 식을 적용하면 closure가 0.61과 0.85로, 하나는 0.7과 0.8에 못 미친다. E0에서 실제 데이터의 깜빡임 정점 분포를 보고 한 번 확정한다. 이 확인은 라벨을 쓰지 않는다.
- 전역 상수(`E`, `B`, `jaw0`의 고정값, `R0`은 단위 행렬): 기준선 대체, 기준선 완성 전의 임시 closure, 전역 정규화 비교 조건에 같은 값을 쓴다. spec v0에서는 설계값(`E = 0.27`, `B = 0.22`, `jaw0 = 0.08`)이고, E0에서 전체 영상의 영상별 통계 중앙값으로 한 번 바꾼다. 라벨을 쓰지 않는 값이지만 시험 피험자의 영상도 들어가므로 리포트에 그렇게 밝히고, 전역 조건에서는 fold별 상수로 다시 계산한 결과를 민감도 점검으로 덧붙인다.
- 알려진 교란(문서화 대상): 아래 시선과 고개 숙임은 closure를 올린다. 하품 중에도 EAR이 내려간다. pitch 보정은 v1에 넣지 않는다.

### 4.4 세션 시간 흐름 (추천안, Q4)

스텝 k는 세션 시간 `[1000(k−1), 1000k)` ms를 대표한다. 정확히 `1000k`에 온 프레임은 스텝 k를 발행시키고 스텝 k+1에 속한다.

| 시점 | 동작 |
|------|------|
| 0 ~ `T_b` | 기준선 수집. `calibrating` 이벤트. 스텝은 내지 않는다. 감김 규칙은 임시 closure로 동작(D4). (구현 메모, 2026-11-15: 분석 불가 판정(8.1절)도 기준선과 무관하게 세션 처음 10초 뒤부터 돌고 경고를 낸다. `docs/SPEC.md` 12.3절) |
| `T_b` 이후 | closure, 깜빡임 검출, 누적 시작. `T_b` 이후에 시작하는 스텝부터 발행. 모든 규칙 동작 |
| 스텝 끝 ≥ `T_b` + 60초 | 조회 구간 60초가 다 찬 스텝. `StepFeatures.full_lookback = true` |
| 다 찬 스텝이 연속 120개 | 모델이 매초 확률을 출력. `T_b` = 60초면 첫 출력은 239초 |

- 프레임이 끊겼다가 한 번에 여러 스텝이 발행되면 스텝마다 그 스텝까지의 창으로 모델을 부른다(창이 120스텝으로 찼을 때). 한 프레임이 스텝 n개를 내면 추론이 최대 n번이다. (2026-11-13 소유자 결정으로 변경, 14장)
- 학습에서도 같은 흐름을 따른다. 영상(분할 영상은 부분)마다 "그 영상 자신의 기준선이 완성된 시각"을 `T_b`로 삼고, 세 정규화 조건 모두 그 시각부터 특징을 만든다. alert 영상 기준선 조건의 나머지 두 영상에는 alert 영상의 기준선을 넣어 주되 시작 시각은 자기 `T_b`를 쓴다. 이렇게 하면 기준선에 쓰인 프레임이 창에 들어가지 않고(원 논문도 보정에 쓴 깜빡임을 제외했다), 세 조건이 같은 창 집합을 쓴다.
- 600초 영상은 `T_b` = 60초일 때 보폭 1에서 창 362개, 보폭 10에서 37개가 나온다. 240초보다 짧은 영상은 창이 나오지 않는다. 기준선이 연장된 영상과 조건별·클래스별 창 수를 리포트에 싣는다.

### 4.5 스트리밍 깜빡임 검출기

closure에 히스테리시스를 거는 상태 머신 `ClosureRunDetector`다. 같은 클래스를 임계값만 바꿔 두 번 쓴다. 스텝 특징용은 진입 0.5 / 이탈 0.3, 규칙용(8장)은 진입 0.7 / 이탈 0.5다. 아래 설명은 특징용 값으로 적는다.

| 설정 | 기본값 | 뜻 |
|------|--------|----|
| `enter_thr` | 0.5 | 이 값 이상이면 감김 구간 시작 |
| `exit_thr` | 0.3 | 이 값 미만이면 구간 종료, 이벤트 발행 |
| `min_duration_ms` | 50 | 이보다 짧은 이벤트는 버림 |
| `max_dt_ms` | 200 | 이보다 긴 프레임 간격은 속도 계산과 시각 보간에서 제외 |
| `hold_ms` | 500 | 감긴 중 유효 프레임이 끊겨도 이만큼은 같은 구간으로 이음 |

상태 전이(프레임이 도착할 때만 판정한다):

| 상태 | 유효, c ≥ enter | 유효, exit ≤ c < enter | 유효, c < exit | 무효 프레임 |
|------|-----------------|------------------------|----------------|-------------|
| `OPEN` | `CLOSED`로. `start_ms` 기록 | 그대로 | 그대로 | 그대로 |
| `CLOSED` | 그대로. 정점 갱신. 직전 유효 표본이 enter 아래였으면 새 감긴 조각을 시작한다 (2026-11-13 소유자 결정으로 변경, 14장) | 그대로. 직전 표본이 enter 이상이었으면 `last_down_ms` 기록 | 이벤트 발행, `OPEN`으로 | `HOLD`로 |
| `HOLD` | 끊긴 시간 ≤ `hold_ms`면 `CLOSED`로 복귀 | 같음 | 이벤트 발행(`truncated=true`), `OPEN`으로 | 끊긴 시간 > `hold_ms`면 이벤트 발행(`truncated=true`), `OPEN`으로 |

- `HOLD`에서 끊긴 시간이 `hold_ms`를 넘은 뒤 도착한 프레임은 먼저 구간을 닫고(`truncated=true`), 그 프레임을 `OPEN` 상태에서 다시 처리한다.
- `start_ms`는 closure가 enter를 위로 지난 시각, `end_ms`는 나가기 전 마지막으로 enter를 아래로 지난 시각이다. 앞뒤 유효 프레임의 간격이 `max_dt_ms` 이하면 선형 보간하고, 아니면 프레임 시각을 쓴다. 잘린 이벤트의 `end_ms`는 closure가 enter 이상이었던 마지막 유효 프레임의 시각이다.
- 지속 시간은 "enter 넘게 감긴 시간", 곧 구간 안에서 closure가 enter 이상이었던 조각들의 길이의 합이다. 조각은 enter를 위로 지난 시각에 시작해 아래로 지난 시각에 끝나고(보간 규칙은 `start_ms`, `end_ms`와 같다), 조각이 하나면 `end_ms − start_ms`와 같다. `min_duration_ms`는 이 합에 적용한다. 히스테리시스는 상태 전이에만 쓰인다. (2026-11-13 소유자 결정으로 변경, 14장)
- 진행 중인 구간의 길이 `ongoing_ms(t)`는 지금까지의 감긴 조각들의 합이고, 진행 중인 조각은 closure가 enter 이상이었던 가장 최근 유효 프레임까지 센다. 눈꺼풀이 0.3~0.5에 머무는 동안에는 늘어나지 않고, 구간 안에서 줄지 않는다. 진행 중인 구간의 시작(`start_ms`)은 `run_start_ms()`로 얻는다. (2026-11-13 소유자 결정으로 변경, 14장)
- 이벤트 출력: `start_ms`, `end_ms`, `duration_ms`(위의 조각 합. 2026-11-13 소유자 결정으로 변경, 14장), `amplitude`(구간 중 최대 closure), `peak_closing_velocity`(진입 두 표본 전부터 정점까지 연속 유효 쌍의 closure 증가율 최댓값, 단위 1/s), `peak_opening_velocity`(정점부터 나갈 때까지의 감소율 최댓값), `velocity_valid`, `truncated`.
- 이벤트를 합치지 않는다. 발행이 지연되지 않게 하기 위해서다.
- 프레임 하나의 처리 순서는 closure 계산, 검출기 갱신, 스텝 발행이다. 이벤트는 그것을 발행한 프레임이 속한 스텝에 귀속한다.
- 15 fps에서는 깜빡임 하나가 1~6개 표본이다. 검출은 되지만 속도가 절반 가까이 낮게 읽히고 위상에 따라 흔들리며, 진폭도 줄어든다. 약 70 ms보다 짧은 깜빡임은 놓칠 수 있다. 보간이 지속 시간 오차를 얼마나 줄이는지는 이 정의로 M1에서 다시 모사해 SPEC.md에 수치와 함께 적는다.

### 4.6 스텝 특징 목록 (F = 17, 순서 고정)

구간은 스텝 끝 직전 60초 또는 10초다. "유효 시간"은 closure가 유효한 프레임이 대표하는 시간의 합이다. "이벤트"는 구간 안의 스텝에 귀속된, 잘리지 않은 특징용 감김 이벤트다.

| # | 이름 | 구간 | 단위 | 정의 | 정의 불가 시 |
|---|------|------|------|------|--------------|
| 0 | `perclos_60` | 60 s | 비율 | closure ≥ 0.8인 유효 시간 / 유효 시간 | 유효 시간 5초 미만이면 0 |
| 1 | `perclos_10` | 10 s | 비율 | 같음 | 유효 시간 2초 미만이면 0 |
| 2 | `closure_mean_60` | 60 s | 0~1 | closure의 시간 가중 평균 | 0 |
| 3 | `closure_mean_10` | 10 s | 0~1 | 같음 | 0 |
| 4 | `blink_rate_60` | 60 s | 회/분 | 이벤트 수 × 60000 / 유효 시간(ms) | 유효 시간 5초 미만이면 0 |
| 5 | `blink_dur_mean_60` | 60 s | ms | 이벤트 지속 시간(각각 상한 2000) 평균 | 이벤트 없으면 0 |
| 6 | `blink_dur_std_60` | 60 s | ms | 같은 값의 모표준편차 | 2개 미만이면 0 |
| 7 | `closing_vel_mean_60` | 60 s | 1/s | 속도가 유효한 이벤트의 최대 감는 속도 평균 | 0 |
| 8 | `opening_vel_mean_60` | 60 s | 1/s | 같은 방식의 최대 뜨는 속도 평균 | 0 |
| 9 | `long_closure_count_60` | 60 s | 회 | 길이 500 ms 이상인 감김 수(잘린 것, 진행 중인 것 포함) | 0 |
| 10 | `max_closure_ms_60` | 60 s | ms | 구간에 귀속된 감김과 진행 중인 감김 중 가장 긴 것의 전체 길이, 상한 10,000 | 0 |
| 11 | `max_closure_ms_10` | 10 s | ms | 같음 | 0 |
| 12 | `yawn_count_60` | 60 s | 회 | 하품 이벤트 수 | 0 |
| 13 | `pitch_std_60` | 60 s | 도 | 얼굴이 있는 프레임의 상대 pitch 시간 가중 표준편차 | 얼굴 시간 5초 미만이면 0 |
| 14 | `nod_count_60` | 60 s | 회 | 끄덕임 이벤트 수 | 0 |
| 15 | `face_missing_ratio_60` | 60 s | 비율 | (얼굴 없는 시간 + 데이터 없는 시간) / 60초 | 해당 없음 |
| 16 | `face_missing_ratio_10` | 10 s | 비율 | 같음, 10초 기준 | 해당 없음 |

- 명세에 적힌 12개 항목은 #0, #4~#10, #12~#15다. 추가한 것은 `closure_mean_60`(깜빡임 사이의 눈 뜬 정도. 문헌에서 유효하다고 보고됨)과 10초 구간 4개다.
- 10초 구간을 넣는 이유: 60초 이동 통계만으로 된 120스텝 수열은 매우 매끄러워서 시계열 모델이 볼 변화가 거의 없다.
- 프레임률, 해상도, 머리 자세의 평균 수준처럼 녹화 조건을 직접 드러내는 값은 특징에 넣지 않는다. RLDD에서는 클래스마다 녹화 세션이 달라 라벨이 샐 수 있다.
- 속도 특징(#7, #8)은 명세에 있어 유지하되, RLDD가 30 fps 미만이므로 이를 뺀 결과를 함께 보고한다.
- 모든 값은 유한하다(3.3절의 0 나눗셈과 분산 규칙, 4.1절의 눈 폭 규칙으로 보장하고 각각 테스트 벡터를 둔다). Kotlin 쪽도 모델 입력 전에 유한성을 검사하고, 유한하지 않으면 추론을 건너뛴다.
- 하품 이벤트(잠정): `jaw_open − jaw0 ≥ 0.40`으로 시작, 0.25 미만으로 끝, 2,000 ms에 도달하면 한 번 센다.
- 끄덕임 이벤트(잠정): `d = pitch_rel − m`이 10도 이상으로 시작, 5도 미만으로 끝, 지속 200~3,000 ms면 끝날 때 센다. `m`은 pitch_rel의 느린 평균으로, 유효 프레임마다 `d`를 먼저 계산한 뒤 `m += α·(pitch_rel − m)`, `α = dt / (10000 + dt)`(dt는 최대 200 ms)로 갱신한다. 초기값은 첫 유효 표본이고 무효 프레임에서는 갱신하지 않는다.
- 하품과 끄덕임 모두 무효 구간이 500 ms를 넘으면 진행 중이던 판정을 취소한다.
- 이 목록과 순서는 승인 시점에 고정한다(spec v0). E0에서는 잠정 상수만 한 번 바꿀 수 있다(spec v1). 그 뒤 식이나 목록을 바꾸는 것은 spec v2이며 사전에 확인을 받고 리포트에 밝힌다.

### 4.7 `feature_spec.json`과 해시

```
{
  "spec_version": 1,
  "step_sec": 1, "window_steps": 120,
  "class_order": ["alert", "low_vigilant", "drowsy"],
  "features": [{"index": 0, "name": "perclos_60", "window_sec": 60, "unit": "ratio",
                "formula": "...", "undefined_value": 0.0, "transform": "none|log1p",
                "mean": ..., "std": ...}, ...],
  "standardization": {"location": "in_graph", "z_clip": 5.0, "std_floor": 0.001, "temperature": ...},
  "pipeline": {                      // 이식 코드가 쓰는 상수 전부
    "landmarks": {"ear_r": [33,160,158,133,153,144], "ear_l": [362,385,387,263,373,380], ...},
    "pose": {"order": "Ry*Rx*Rz", "yaw_pos": "subject_left", "pitch_pos": "chin_down", "roll_pos": "ccw_in_image"},
    "clock": {...}, "baseline": {...}, "closure": {...}, "run_detector": {...},
    "yawn": {...}, "nod": {...}, "global_constants": {...},
    "mediapipe": {"model_sha256": "...", "num_faces": 1}
  },
  "pipeline_hash": "..."             // pipeline 블록의 해시 (Python이 계산, Kotlin은 문자열 비교만)
}
```

- `feature_spec_hash`는 모델과 함께 내보내는 이 파일 바이트의 sha256이다. 앱은 번들한 파일을 `MessageDigest`로 해시해 모델 메타데이터와 비교한다. JSON 정규화가 필요 없다. 줄바꿈 변환을 막으려고 해시 대상 파일에 `.gitattributes`의 `-text`를 건다.
- 평균과 표준편차는 학습 데이터로 계산해 이 파일과 ONNX 그래프 상수 양쪽에 넣고, 내보낼 때 두 값이 같은지 검사한다. 표준편차는 0.001 아래로 내려가지 않게 한다.
- `pipeline` 블록은 E0에서 확정한 뒤 바뀌지 않는다. 재학습으로 평균·표준편차가 바뀌어도 `pipeline_hash`와 이식 코드용 벡터는 유효하다.
- 규칙 임계값은 `config/rules.yaml`에서 관리하고, 앱에는 같은 내용을 `export/rules.json`으로 내보낸다. 그 파일의 해시 `rules_hash`를 규칙 벡터와 오경보 리포트에 적는다.

---

## 5. 모델과 학습

### 5.1 내보낼 후보

그래프 안의 처리 순서는 두 모델이 같다: `log1p(max(x, 0))`(#9~#12, #14에만, 상수 마스크로 구현) → `(x − mean) / std` → ±5 자르기 → 신경망 → 온도로 나누기 → 소프트맥스.

| 모델 | 구성 | 크기 |
|------|------|------|
| 인과 TCN | 잔차 블록 5개(블록당 인과 확장 합성곱 2개, 커널 3, 확장 1·2·4·8·16, 합성곱 바로 뒤 배치 정규화, ReLU, 드롭아웃), 마지막 스텝 읽기, 선형 3. 수용 영역 125스텝 | 채널 32일 때 약 3만 파라미터(F=24 시험 모델 실측 31,811) |
| 단방향 GRU | GRU 1층, 마지막 은닉, 드롭아웃, 선형 3 | 은닉 64일 때 약 1.6만(F=24 실측 17,475) |

둘 다 목표(10만 파라미터, 1 MB)보다 훨씬 작다. 배치 정규화를 합성곱 바로 뒤에 두면 내보낼 때 합성곱에 합쳐진다. 허용 연산자 목록은 M1에서 `drowsiness_v0.onnx`(위 처리를 모두 포함)를 내보내 확정하고 SPEC.md에 싣는다. INT8 파일은 목록이 따로다.

### 5.2 학습 설정

- 손실: 라벨 스무딩 0.1을 준 교차 엔트로피. 영상 라벨을 창이 물려받아 창 단위 잡음이 크기 때문이다.
- 클래스 불균형과 피험자 간 차이: 표집을 피험자, 영상, 창 순서로 균등하게 뽑는다. 피험자마다 클래스별 영상이 하나씩이므로 클래스와 피험자가 함께 균형을 이루고, 긴 영상이 학습을 지배하지 않는다. 여기에 개인 기준선 정규화, 좌우 대칭 특징, 작은 모델, 드롭아웃, 가중치 감쇠를 더한다.
- 최적화: AdamW, 학습률 1e-3, 가중치 감쇠 1e-2, 배치 64. 표집에 자연스러운 에폭이 없으므로 200 갱신을 한 단위로 삼아 최대 60단위, 내부 검증 macro-F1 기준 조기 종료(인내 10단위).
- 탐색 범위(사전 고정): TCN 채널 {16, 32} × 드롭아웃 {0.1, 0.3}, GRU 은닉 {32, 64} × 드롭아웃 {0.1, 0.3}.
- 창: 학습 보폭 5초(시작 위치 무작위 이동), 평가 보폭 1초.
- 증강: 표준화 공간의 가우시안 잡음(σ = 0.05), 프레임률 변형. 프레임률 변형은 두 가지다. (1) MediaPipe 앞단에서 흐름 제한기를 흉내 내 약 15 fps로 다시 추출한 특징. 같은 프레임을 받으면 VIDEO와 LIVE_STREAM의 결과가 같으므로 기기 동작을 그대로 재현한다. (2) 저장된 신호에서 프레임을 무작위로 10~30% 버리고 특징을 다시 계산. 근사임을 문서화한다.
- 확률 보정: 내부 검증 예측으로 온도 하나를 맞춰 그래프에 넣고, 보정 전후 ECE를 보고한다.
- 부가 실험: 순서형 소프트 라벨(이웃 클래스에 확률을 나눠 주는 방식)과 비교.

### 5.3 비교 기준선: LightGBM

창(120 × 17)마다 특징별 평균, 표준편차, 최소, 최대, 마지막 값, 선형 기울기를 계산해 102개 입력으로 쓴다. 같은 fold, 같은 창, 같은 내부 검증 절차를 쓴다. 탐색 범위는 `num_leaves` {7, 15} × `min_data_in_leaf` {20, 60}. 내보내지 않는다. 비교표와 D11의 기준이 되는 LightGBM은 alert 영상 기준선 조건, `W = 0.5`의 것으로 미리 정한다.

---

## 6. 평가 절차

- **fold**: 공식 5-fold. 바깥 fold k를 시험으로 두고, 나머지 4개 fold 안에서 검증 fold를 하나씩 돌려가며(학습 36명, 검증 12명) 모델 계열과 설정, 학습 길이(갱신 횟수), 온도를 고른다. 선택 기준은 내부 검증 macro-F1의 평균이고, 차이가 내부 fold 표준편차 안이면 파라미터가 적은 쪽을 고른다. 고른 것으로 48명 전체를 다시 학습해 시험 fold를 한 번만 평가한다. 이 "바깥 fold마다 안에서 고른" 행이 주 결과다. 계열별(TCN만, GRU만, LightGBM) 행도 같은 절차로 따로 싣는다.
- 시험 fold는 계열·설정 선택, 조기 종료, 표준화 상수, 온도 어디에도 쓰이지 않는다. 예외는 E0에서 라벨 없이 한 번 정하는 closure 상수와 전역 상수뿐이고(4.3절), 리포트에 밝힌다.
- **피험자 분리 검사**: 데이터 구성, fold 구성, 리포트 작성 시점에 `assert_subject_disjoint`를 호출한다. 일부러 겹치게 한 입력에서 예외가 나는지 테스트로 확인한다.
- **시드**: 다시 학습할 때 3개. 표에는 fold별 값(시드 평균), fold 평균 ± 표준편차, 시드 간 표준편차, 피험자 단위 부트스트랩 95% 구간을 싣는다.
- **지표**
  - 창 단위: macro-F1, 클래스별 재현율, 혼동 행렬. 영상마다 같은 비중을 주는 값을 주로, 창을 그대로 합친 값을 보조로 싣는다.
  - 영상 단위: 창 예측의 다수결(동률이면 평균 확률)로 구한 정확도, 클래스별 재현율, 혼동 행렬. 분모는 D7.
  - 2클래스: D8.
  - 오경보: alert 시험 영상에서 졸음 판정이 시작된 횟수를 평가 시간으로 나눈 값. 판정은 규칙 엔진과 같은 코드를 쓴다(`p_drowsy ≥ 0.6`이 10스텝 연속이면 시작, `≤ 0.4`가 10스텝 연속이면 종료, 종료 후 60초 동안은 다시 시작하지 않음). 임계값은 사전에 고정하고 시험 fold로 조정하지 않는다. 평가 가능한 alert 시간은 전체 약 6시간뿐이므로 5개 fold를 합친 횟수와 총 시간을 주 수치로 싣는다.
  - 판정은 한번 시작하면 유지되므로 횟수만으로는 부족하다. 함께 싣는 값: 판정이 한 번이라도 난 alert 영상 수(60개 중), alert 평가 시간 중 판정 상태였던 비율, 피험자 부트스트랩 구간, 창 단위 값(D9), 같은 판정기로 drowsy 영상 중 한 번이라도 판정된 비율, 임계값을 바꿔 본 곡선.
  - PERCLOS 규칙과 `either`도 같은 방식으로 평가해 모델과 나란히 싣는다(학습이 필요 없다).
- **앱 조건 평가**: alert 영상 기준선 조건으로 학습한 fold 모델을, 시험 fold의 "자기 세션 기준선" 특징으로도 평가한다(재학습 없음). alert 영상은 두 조건에서 같고 나머지 두 클래스만 달라진다. 원 조건 값과 이 값을 낙관·비관 범위로 모델 카드에 함께 싣고, D11은 이 값으로 판정한다.
- **alert 영상 기준선 조건의 한계**: alert 영상은 자기 세션의 기준선으로, 나머지 두 영상은 다른 세션의 기준선으로 정규화된다. 카메라나 조명 차이가 라벨과 함께 움직일 수 있고, 차량에서는 이런 차이가 없다. 위의 앱 조건 평가와 세 번째 조건(D1)이 그 크기를 보여준다.
- **교란 점검**(보고만 하고 선택에는 쓰지 않음): (1) 녹화 메타데이터(평균 프레임률, 해상도, 얼굴 크기, 얼굴 검출률, 밝기)만으로 LightGBM 학습. (2) 영상별 기준선 편차(뜬 눈 closure 중앙값, 상대 yaw·pitch 중앙값)만으로 학습. 어느 쪽이든 우연 수준을 크게 넘으면 리포트 맨 앞에 적는다.
- **비교표**: 우리 결과 옆에 문헌 값을 "문헌 보고값"으로 구분해 싣는다. HM-LSTM 65.2%(조건 주석 포함), 사람 57.8%, Wolter 2025, Magán 2022, Xu 2025(조건 주석 포함).
- **통계적 주의**: fold당 영상 36개라 영상 하나가 2.8%p다. TCN과 GRU의 차이는 짝지은 부트스트랩 구간이 0을 벗어날 때만 차이가 있다고 쓴다.
- **저장**: `reports/<실행명>/`에 설정 사본, 영상 목록, 제외 목록, fold별 JSON과 CSV, 요약 Markdown(fold별 표와 평균 ± 표준편차), 혼동 행렬, 오경보 표, 비교표, ONNX 일치 검증과 지연 기록.

---

## 7. ONNX 내보내기

- 계약 그대로: `export/drowsiness_v{N}.onnx`, opset 17, 배치 1, 입력 `features` float32 `[1,120,17]`, 출력 `probs` float32 `[1,3]`(alert, low_vigilant, drowsy). 입력은 표준화하지 않은 원시 스텝 특징이고, `features[0][0]`이 가장 오래된 스텝, `features[0][119]`가 가장 최근 스텝이다.
- 메타데이터(값은 모두 문자열):

| 키 | 예 |
|----|----|
| `model_version` | `1` |
| `feature_spec_hash` | `export/feature_spec.json` 바이트의 sha256 |
| `step_sec` | `1` |
| `window_steps` | `120` |
| `class_order` | `alert,low_vigilant,drowsy` |
| `trained_on` | `UTA-RLDD 60 subjects, arm=alert_baseline` (v0은 `none (random weights)`) |
| `commit` | git commit hash, 작업 트리가 깨끗하지 않으면 `-dirty`를 붙임 |
| (Q6) `standardization` | `in_graph` |
| (Q6) `pipeline_hash` | 4.7절 |
| (Q6) `mediapipe_model_sha256` | `64184e22…` |

- 내보내기: TorchScript 방식(`dynamo=False`)을 기본으로, dynamo 방식에 후처리(opset 17 변환, IR 8 설정)를 붙인 것을 대체 경로로 둔다.
- 내보낸 뒤 자동 검사: opset이 정확히 17, IR 8, 단일 파일, 입출력 이름·형·모양, 메타데이터 키, 파일 1 MB 미만, 허용 연산자 목록, 그래프 안 상수와 spec 값의 일치, 표준편차 하한.
- 일치 검증(pytest): 무작위 입력과 실제 창에서 PyTorch와 onnxruntime의 최대 절대 오차 < 1e-4.
- 지연: onnxruntime CPU, 연산 내부 스레드 1개, 1,000회 반복의 중앙값과 95 분위수를 기록한다.
- INT8(D16): `quantize_dynamic`, 가중치 형은 `QUInt8`. 각 fold 모델을 양자화해 시험 fold 지표 변화를 보고한다.
- M1에서 무작위 가중치로 `drowsiness_v0.onnx`를 먼저 내보낸다. 앱 팀이 입출력 계약으로 연동을 시작할 수 있게 하기 위한 것이다.

---

## 8. 규칙 엔진

프레임마다 `update()`를 받는 상태 머신들로 구성한다. 임계값은 `config/rules.yaml`에 출처와 함께 둔다. 이벤트는 임계 시간에 도달한 순간 `onset`이 나가고 상태가 끝날 때 `end`가 나간다. `duration_ms`는 발행 시점까지의 길이다.

### 8.1 내부 판정

- **전방 주시**: 상대 yaw 절댓값 ≤ 23도이고 상대 pitch가 위 25도~아래 18도 안이면 영역 안이다. 눈동자 보조(D12): `gaze_h`의 기준선 대비 변화 절댓값이 0.25를 넘거나 `gaze_v`의 변화 절댓값이 0.35를 넘으면 영역 밖으로 본다(잠정). 상태 변화는 250 ms 이상 유지돼야 확정되고, 확정되면 시작 시각을 변화가 처음 나타난 프레임으로 소급한다. 영역 밖 상태에서 얼굴이 사라지면 3초까지는 영역 밖으로 치고, 그 뒤에는 "알 수 없음"으로 두어 시선 타이머를 멈춘다.
- **눈 감김**: 4.5절의 검출기를 진입 0.7 / 이탈 0.5로 한 번 더 돌린다. 특징용보다 임계값이 높은 이유는 경고에 쓰이는 판정이라 반쯤 감긴 눈이나 아래 시선에 덜 반응해야 하기 때문이다. 임계 시간은 유효한 감김 프레임이 도착했을 때 `ongoing_ms`로 판정한다. 길이는 0.7 이상이었던 시간의 누적이다. 0.5~0.7에 머문 시간은 세지 않고, 끊긴 구간(500 ms 이하)은 앞뒤 유효 표본이 모두 0.7 이상일 때만 길이에 들어간다. 끊긴 구간 안에서는 시작 이벤트를 내지 않는다. (2026-11-13 소유자 결정으로 변경, 14장) (구간은 0.5 아래로 내려가야 끝나므로 눈꺼풀이 0.5~0.7에 머물면 여러 깜빡임의 0.7 이상 시간이 한 감김에 쌓인다. 2026-11-14 소유자 결정으로 그대로 둔 동작이고 알려진 한계다. 14장 표 (c)의 5번)
- **분석 불가**: 최근 10초의 얼굴 검출 비율(시간 기준, 데이터 없음 포함)이 0.5 미만. 지속되는 동안 시선과 감김의 누적을 멈춘다. 10초 이하의 끊김은 얼굴 없는 시간으로 센다(약 5.2초가 넘는 끊김 뒤에는 첫 프레임에서 시작된다). 10초 넘는 끊김 뒤에는 세션 시작 때처럼 창을 비우고 그 프레임부터 10초는 판정하지 않는다. (2026-11-13 소유자 결정으로 변경, 14장)

### 8.2 이벤트

| `type` | 시작 조건(기본값) | `t_start_ms` | 종료 | `severity` | `source` | 재경고 대기 |
|--------|-------------------|--------------|------|-----------|----------|-------------|
| `calibrating` | 세션 시작 | 첫 프레임 | 기준선 완성 | 0 | `face` | 없음 |
| `unable_to_analyze` | 8.1의 분석 불가 | 비율이 처음 0.5 아래로 내려간 프레임 | 비율 0.5 이상, 또는 10초 넘는 끊김 뒤 첫 프레임 (2026-11-13 소유자 결정으로 변경, 14장) | 1 | `face` | 30초 |
| `gaze_away_long` | 영역 밖이 연속 3.0초 | 영역 밖이 시작된 프레임 | 영역 안 확정 | 2 | `head_pose` 또는 `eye`(머리 조건이 성립하면 `head_pose`) | 10초 |
| `vats` | 최근 30초 중 영역 밖 누적 10초 | 그 30초 안의 첫 영역 밖 구간 시작 | 누적이 10초 아래로 | 2 | 위와 같음 | 30초 |
| `drowsy` | 모델: 6장의 판정. PERCLOS: `perclos_60 ≥ 0.15`가 10스텝 연속 | 연속 구간의 첫 스텝 끝 | 모델: 6장. PERCLOS: 0.10 미만이 10스텝 연속. 또는 10초 넘는 끊김 뒤 첫 프레임(confidence 0.0, 두 판정기도 처음 상태로) (2026-11-13 소유자 결정으로 변경, 14장) | 2 | `model` 또는 `perclos` | 300초 |
| `microsleep` | 눈 감김(0.7 이상이었던 시간의 누적)이 1.0초 도달 (2026-11-13 소유자 결정으로 변경, 14장) | 0.7을 위로 지난 시각 | 감김 종료 | 3 | `closure` | 5초 |
| `sleep` | 같은 감김의 누적이 3.0초 도달(`microsleep`과 같은 `t_start_ms`) (2026-11-13 소유자 결정으로 변경, 14장) | 같음 | 감김 종료 | 3 | `closure` | 없음(3초마다 반복) |

- 한 감김은 `microsleep` 시작, (3.0초에) `sleep` 시작, 감김이 끝날 때 도달한 가장 높은 종류의 `end` 하나를 낸다. `duration_ms`는 발행 시점까지의 길이(발행 시각 − `t_start_ms`)라서 시작을 일으킨 누적 감김 시간보다 길 수 있다. (2026-11-13 소유자 결정으로 변경, 14장)
- 이벤트 스키마: `DriverStateEvent{type, severity, t_start_ms, duration_ms, confidence, source}` + (Q5) `phase`, `provisional`. `confidence`는 규칙에서는 해당 구간의 유효 시간 비율, 모델에서는 `p_drowsy`다.
- 경고 결정(`AlertPolicy`): 입력은 상태 이벤트와 차량 속도, 출력은 실제로 경고할 이벤트다. 표의 재경고 대기, 속도 억제(D3), 우선순위(`sleep`, `microsleep`, `gaze_away_long`, `vats`, `drowsy` 순)를 적용한다. 재경고 대기나 속도 억제로 막힌 시작은 막힘이 풀렸을 때 그 상태가 계속 중이면 한 번 경고하고(시작 하나에 경고는 최대 한 번, 되풀이하는 `sleep`은 그대로), 상태가 끝나면 버린다. (2026-11-13 소유자 결정으로 변경, 14장)
- 시선 규칙의 한계: 머리 자세만으로는 눈만 움직이는 시선 이탈을 놓치고 미러 확인을 이탈로 본다. 운전자가 옆으로 옮겨 앉아도 상대 yaw가 변한다. RLDD는 화면을 보는 실내 영상이라 시선 규칙을 검증할 수 없다. 합성 벡터와 팀 수집 영상으로만 확인할 수 있다고 문서에 적는다.

---

## 9. 저장소 구조와 주요 시그니처

```
hawkeye/
  pyproject.toml, uv.lock, README.md, .gitattributes
  config/default.yaml, config/rules.yaml, config/feature_spec.json
  drowsy/
    types.py          # 아래 자료형
    spec.py           # feature_spec, rules 로드와 해시
    signals.py        # EAR, 머리 자세, 시선 대용치           (이식 대상, 표준 라이브러리만)
    calibrate.py      # 기준선, closure                       (이식 대상)
    blink.py          # ClosureRunDetector                    (이식 대상)
    step_features.py  # 하품·끄덕임 검출, 1초 스텝 특징        (이식 대상)
    rules.py          # 규칙 엔진, 경고 결정                  (이식 대상)
    pipeline.py       # 위 모듈을 잇는 DriverPipeline         (이식 대상)
    video_io.py       # PyAV 디코딩, 타임스탬프, 회전 보정
    extract.py        # 영상 → MediaPipe → RawFrame parquet, 프레임 누락 모사
    dataset.py        # RLDD 로더, 무라벨 차량 영상 로더, 재생, 창 생성, fold
    synth.py          # 합성 세션 생성기(테스트와 벡터용)
    models.py         # CausalTCN, GRUClassifier, ExportWrapper
    train.py, evaluate.py, baseline_lgbm.py
    export_onnx.py    # 내보내기, 메타데이터, 계약 검사, 일치 검증, 지연, INT8
    vectors.py        # 테스트 벡터 생성과 재검증
    demo.py
  tests/  (vectors/ 포함)
  docs/   PLAN.md, SPEC.md, MODEL_CARD.md
  export/ reports/
```

명세의 권장 구조에서 더한 것은 `types.py`, `spec.py`, `pipeline.py`, `video_io.py`, `synth.py`, `vectors.py`다.

```python
# types.py  (모두 불변 dataclass)
class RawFrame:      t_ms: int; face: bool; w: int; h: int; lm: dict[int, tuple[float, float]]; bs: dict[str, float]; m: tuple[float, ...]
class FrameSignals:  t_ms: int; face: bool; ear_l: float; ear_r: float; eye_w_l: float; eye_w_r: float; blink_l: float; blink_r: float
                     eye_look: tuple[float, ...]; jaw_open: float; yaw: float; pitch: float; roll: float   # + 선택 홍채 비율
class Baseline:      e_l: float; e_r: float; b_l: float; b_r: float; r0: tuple[float, ...]; jaw0: float; gaze_h0: float; gaze_v0: float
                     t_complete_ms: int; status: str
class FrameState:    t_ms: int; face: bool; closure: float; closure_valid: bool; provisional: bool
                     yaw_rel: float; pitch_rel: float; pose_valid: bool; gaze_h_rel: float; gaze_v_rel: float; jaw_rel: float
class ClosureRun:    start_ms: int; end_ms: int; duration_ms: int; amplitude: float
                     peak_closing_velocity: float; peak_opening_velocity: float; velocity_valid: bool; truncated: bool
class StepFeatures:  k: int; t_end_ms: int; full_lookback: bool; values: tuple[float, ...]
class DriverStateEvent: type: str; severity: int; t_start_ms: int; duration_ms: int; confidence: float; source: str
                        phase: str; provisional: bool                      # 뒤 두 개는 Q5
class PipelineOutput: state: FrameState; runs: tuple[ClosureRun, ...]; steps: tuple[StepFeatures, ...]
                      probs: tuple[float, float, float] | None; events: tuple[DriverStateEvent, ...]; alerts: tuple[DriverStateEvent, ...]

# signals.py
def compute(frame: RawFrame, spec: PipelineSpec) -> FrameSignals: ...
def eye_aspect_ratio(pts_px: Sequence[tuple[float, float]]) -> float: ...            # p1..p6 순서
def euler_from_colmajor(m: Sequence[float]) -> tuple[float, float, float]: ...       # yaw, pitch, roll (도)
def rotation_from_euler(yaw: float, pitch: float, roll: float) -> tuple[float, ...]: ...
def relative_euler(m: Sequence[float], r0: Sequence[float]) -> tuple[float, float, float]: ...
def iris_ratio(outer: tuple[float, float], inner: tuple[float, float], iris: tuple[float, float]) -> tuple[float, float]: ...

# calibrate.py
class BaselineEstimator:
    def __init__(self, spec: PipelineSpec) -> None: ...
    def update(self, s: FrameSignals) -> Baseline | None: ...        # 완성되는 프레임에서 한 번 반환
def global_baseline(spec: PipelineSpec) -> Baseline: ...
def frame_state(s: FrameSignals, b: Baseline, spec: PipelineSpec, provisional: bool) -> FrameState: ...

# blink.py
class ClosureRunDetector:
    def __init__(self, enter_thr: float, exit_thr: float, spec: PipelineSpec) -> None: ...
    def update(self, t_ms: int, closure: float, valid: bool) -> ClosureRun | None: ...
    def ongoing_ms(self) -> int: ...

# step_features.py
class ThresholdEventCounter:                                         # 하품, 끄덕임 공용
    def update(self, t_ms: int, value: float, valid: bool) -> bool: ...
class StepAggregator:
    def __init__(self, spec: PipelineSpec, start_ms: int) -> None: ...   # start_ms = T_b
    def update(self, state: FrameState, run: ClosureRun | None, ongoing_ms: int) -> list[StepFeatures]: ...

# rules.py
class RuleEngine:
    def __init__(self, cfg: RulesConfig) -> None: ...
    def update(self, state: FrameState) -> list[DriverStateEvent]: ...
    def on_step(self, step: StepFeatures, probs: Sequence[float] | None) -> list[DriverStateEvent]: ...
class AlertPolicy:
    def __init__(self, cfg: RulesConfig) -> None: ...
    def decide(self, events: Sequence[DriverStateEvent], speed_kmh: float | None, t_ms: int) -> list[DriverStateEvent]: ...

# pipeline.py
class DriverPipeline:
    def __init__(self, spec: PipelineSpec, rules: RulesConfig,
                 baseline: Baseline | None = None,          # 주면 보정을 건너뛰고 이 기준선을 쓴다(학습용)
                 feature_start_ms: int | None = None,       # 기준선을 줄 때 특징 누적을 시작할 세션 시간
                 model: Callable[[Sequence[Sequence[float]]], Sequence[float]] | None = None) -> None: ...
    def update(self, frame: RawFrame, speed_kmh: float | None = None) -> PipelineOutput: ...

# video_io.py / extract.py
def iter_video_frames(path: Path) -> Iterator[tuple[int, np.ndarray]]: ...          # (양수 t_ms, 똑바로 세운 RGB)
def flow_limiter_select(t_ms: Sequence[int], latency_ms: Sequence[float]) -> list[int]: ...   # 기기 프레임 누락 모사
def extract_video(path: Path, model_path: Path, out_parquet: Path, drop_sim: DropSimConfig | None = None) -> ExtractStats: ...

# dataset.py
def build_manifest(root: Path) -> pd.DataFrame: ...                  # subject, fold, label, part, path, 품질 통계
def scan_unlabeled(root: Path) -> pd.DataFrame: ...                  # 팀 수집 차량 영상(오경보 평가용)
def official_fold(subject_id: int) -> int: ...
def assert_subject_disjoint(*subject_sets: Iterable[int]) -> None: ...
def replay_video(frames: Iterable[RawFrame], spec: PipelineSpec, baseline: Baseline | None,
                 feature_start_ms: int | None) -> ReplayResult: ...
def build_step_table(manifest: pd.DataFrame, raw_dir: Path, spec: PipelineSpec,
                     arm: Literal["alert_baseline", "global", "session_baseline"],
                     global_override: Baseline | None = None) -> pd.DataFrame: ...
def make_windows(steps: pd.DataFrame, window_steps: int = 120, stride: int = 5) -> WindowSet: ...
def nested_splits(manifest: pd.DataFrame) -> Iterator[FoldSplit]: ...

# models.py
class CausalTCN(nn.Module):
    def __init__(self, n_features: int, channels: int = 32, kernel_size: int = 3,
                 dilations: Sequence[int] = (1, 2, 4, 8, 16), dropout: float = 0.1) -> None: ...
    def forward(self, z: Tensor) -> Tensor: ...                      # 표준화된 [B,120,F] -> logits [B,3]
class GRUClassifier(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.1) -> None: ...
class ExportWrapper(nn.Module):                                      # log1p, 표준화, 자르기, 온도, 소프트맥스 포함
    def __init__(self, net: nn.Module, spec: FeatureSpec, temperature: float) -> None: ...
    def forward(self, features: Tensor) -> Tensor: ...               # 원시 [1,120,F] -> probs [1,3]
def count_parameters(m: nn.Module) -> int: ...

# train.py / evaluate.py / baseline_lgbm.py
def fit(train: WindowSet, val: WindowSet | None, cfg: TrainConfig, seed: int) -> FitResult: ...
def run_nested_cv(cfg: ExperimentConfig) -> CVResult: ...
def fit_final(cfg: ExperimentConfig) -> FitResult: ...
def window_metrics(preds: pd.DataFrame) -> dict[str, float]: ...
def video_vote(preds: pd.DataFrame) -> pd.DataFrame: ...
def alarm_report(preds: pd.DataFrame, cfg: RulesConfig, source: Literal["model", "perclos", "either"]) -> AlarmReport: ...
def subject_bootstrap(preds: pd.DataFrame, metric: Callable[[pd.DataFrame], float], n: int = 10000) -> tuple[float, float]: ...
def window_stats(x: np.ndarray) -> np.ndarray: ...                   # [N,120,F] -> [N,6F]
def run_lgbm_cv(cfg: ExperimentConfig) -> CVResult: ...

# export_onnx.py
def export(model: ExportWrapper, spec_path: Path, out_path: Path, meta: Mapping[str, str]) -> Path: ...
def verify_contract(onnx_path: Path, spec_path: Path) -> None: ...
def parity_max_abs_diff(model: ExportWrapper, onnx_path: Path, inputs: np.ndarray) -> float: ...
def measure_latency(onnx_path: Path, runs: int = 1000) -> dict[str, float]: ...
def quantize_int8(onnx_path: Path, out_path: Path) -> Path: ...

# demo.py
def run_demo(video: Path, onnx_path: Path, out_dir: Path, speed_kmh: float | None = None) -> tuple[Path, Path]: ...
```

- 실행 명령은 모두 `python -m drowsy.<모듈> --config config/default.yaml` 형식이다.
- 데모: `python -m drowsy.demo --video path.mp4 [--model export/drowsiness_v1.onnx] [--out out/]`. 영상 자신의 앞 60초를 기준선으로 써서 전체 파이프라인을 돌린다. 출력은 타임라인 PNG(closure, 시선 이탈 여부, 졸음 확률, 규칙 이벤트 네 줄)와 이벤트 JSON(`DriverStateEvent` 목록)이다. 240초보다 짧은 영상은 확률 줄이 비고 그 이유를 그림에 적는다. 얼굴 프레임은 저장하지 않는다(D13).

---

## 10. 테스트, 테스트 벡터, SPEC.md

### 10.1 단위 테스트 (pytest, 합성 데이터)

- 신호: 표준 얼굴 모델 기하에서의 EAR, 비정사각 프레임, 폭이 0인 눈, 오일러 각 왕복, 열 우선 배열과 4×4의 일치, 반전 항등식, 공식 기대 행렬.
- 기준선: 분위수와 중앙값 정의, 유효 프레임 부족 시 연장과 대체.
- closure: 먼 쪽 눈 가중, 반전 불변성, 한쪽 눈만 쓸 수 있는 경우, 유효 조건.
- 감김 검출기: 30 fps와 15 fps의 합성 깜빡임, 프레임 누락, 중복 타임스탬프, 긴 감김, 0.3~0.5에 머무는 눈꺼풀, 감김 중 얼굴 소실, 전이표의 모든 칸.
- 스텝 특징: 손으로 계산한 기댓값, 초기 구간, 깜빡임 없는 구간, 얼굴 소실, 긴 끊김 뒤 여러 스텝 발행, 10초 넘는 끊김 뒤 초기화, 분산이 0인 구간, 첫 `t_ms`가 큰 임의 값인 경우, 같은 신호를 다른 프레임률로 넣었을 때의 근접성.
- 데이터: fold 매핑, 피험자 분리 검사(위반 입력에서 예외), 창 생성, 분할 영상(`10_1`, `10_2`) 처리, 조회 구간이 덜 찬 스텝이 창에 없는지.
- 모델: 120스텝 전체가 출력에 영향을 주는지, 합성곱 층이 미래 스텝을 보지 않는지, 파라미터 수.
- 내보내기: 계약 검사, 일치 검증, 메타데이터, 파일 크기, 표준편차가 0인 특징.
- 규칙: 경계값(감김 0.9·1.1·2.5·3.0·6초, 시선 2.9·3.1초, 얼굴이 사라지는 고개 돌림, VATS 누적, 모델 졸음 판정의 시작과 종료, PERCLOS 판정, 보정 중과 임시 판정, 속도 기준 위아래, 재경고 대기, 분석 불가).
- 이식 대상 모듈이 numpy를 import하지 않는지.
- MediaPipe가 필요한 테스트는 모델 파일이나 라이브러리가 없으면 건너뛴다.

### 10.2 테스트 벡터 (`tests/vectors/*.json`)

형식: `{name, pipeline_hash, rules_hash, baseline|null, config_overrides, frames:[{t_ms, face, w, h, lm:{인덱스:[x,y]}, bs:{이름:값}, m:[16]}], speed:[[t_ms, kmh]], expected:{signals, closure, runs, steps, probs, events, alerts}}`.

| 묶음 | 내용 | 허용 오차 |
|------|------|-----------|
| 신호 | 정사각·비정사각 프레임의 EAR, 공식 행렬과 반전·무작위 행렬의 각도, 한쪽 눈만 감은 경우(좌우 확인), 폭이 0인 눈 | 신호 1e-6, 각도 1e-3도 |
| 기준선과 closure | 정상, 연장, 대체, yaw에 따른 눈 가중, 얼굴 없음 | 1e-6 |
| 감김 검출 | 30 fps·15 fps·누락 포함, 한 표본짜리 깜빡임, 600 ms 감김, 끊겨서 잘린 감김, 중복 타임스탬프 | 시각 ±1 ms, 값 상대 1e-4 |
| 스텝 특징 | 5분 세션(초기 구간, 깜빡임 없는 구간, 얼굴 소실, 긴 끊김), 첫 `t_ms`가 큰 값인 세션 | 절대 1e-5 + 상대 1e-5 |
| 규칙 | 10.1의 경계값 전부, 경고 결정 출력 포함 | 종류·단계·순서 일치, 시각 ±1 ms |
| 모델 | `features [1,120,17]` 세 개와 `probs`, 모델 파일 해시 | 절대 1e-4 |
| 전 과정 | 약 250초의 프레임 → 스텝 → 120스텝 텐서 → `probs` → `drowsy` 시작과 종료 | 위와 같음 |

저장소의 pytest가 벡터를 다시 계산해 저장된 값과 비교하므로 참조 구현이 바뀌면 바로 드러난다.

### 10.3 SPEC.md 목차

앱이 지켜야 할 MediaPipe 설정과 프레임 전제(Q3), 타임스탬프와 세션 시간, 인덱스 표, 좌표·각도·좌우 규약, 기준선과 closure, 감김 검출기 전이표, 시간 누적 방식과 스텝 특징 표, 규칙과 이벤트 표, 경고 결정, ONNX 입출력과 메타데이터(원시 특징을 넣을 것, 스텝 순서, 스레드 1개 권장, 허용 연산자), 15 fps에서의 한계, 알려진 교란, 거치 위치 권장.

---

## 11. 실험과 납품 순서

각 단계는 통과 기준을 만족해야 다음으로 넘어간다. 데이터 경로가 없어도 M0~M2는 끝낼 수 있다.

| 단계 | 내용 | 통과 기준 |
|------|------|-----------|
| M0 뼈대 | `git init`, pyproject와 잠금 파일, 설정, spec v0. 고정한 mediapipe 버전의 일회성 점검(스크래치에서: 공식 행렬의 각도, 시작 타임스탬프 0과 1의 차이, VIDEO 모드 출력) | 의존성 설치 성공, MediaPipe 로드와 모델 sha256 확인 |
| M1 이식 대상 구현과 계약 고정 | signals, calibrate, blink, step_features, rules, pipeline, synth, 테스트 벡터, SPEC.md 초판, `drowsiness_v0.onnx` | pytest 통과. 벡터를 다시 만들어도 바이트 단위로 같음. ONNX 계약 검사와 일치 검증 통과. **이 시점에 앱 팀이 연동을 시작할 수 있다** |
| M2 합성 데이터 전 과정 | 합성 세션으로 학습, 평가, 내보내기. 데모와 README 초판 | 피험자 겹침을 넣으면 검사가 실패함. 데모가 PNG와 JSON을 씀. 합성 점수는 보고하지 않음 |
| M3 추출 | 영상 목록, 6개 영상 시험 추출(프레임당 시간, 얼굴 검출률, 회전 확인), 전체 추출(원본과 15 fps 모사) | 논리 영상 180개의 처리 결과와 제외 사유가 reports/에 있음. 타임스탬프가 양수이고 엄격히 증가 |
| E0 신호 점검과 상수 확정 | 잠정 상수 확정(한 번만), 전역 상수 계산, 기준선 통계 분포. 확정 뒤 교란 점검 두 가지 | 라벨을 쓰지 않고 전체 영상으로 판정: 영상의 90% 이상에서 기준선이 60초에 대체 없이 완성, 80% 이상에서 깜빡임 분당 5~40회, 80% 이상에서 깜빡임 정점 closure의 중앙값이 0.8 이상. 미달이면 상수를 한 번 더 고칠 수 있고, 식을 바꿔야 하면 spec v2로 확인을 받는다. 확정 후 `pipeline_hash`를 앱 팀에 알림 |
| E1 LightGBM | 정규화 세 조건. `W` 세 가지 비교는 보고용 | fold별 표 저장. 내부 검증 macro-F1 평균이 0.40 미만이면(우연은 0.33) 신경망으로 가기 전에 멈추고 보고 |
| E2 TCN 대 GRU | alert 영상 기준선 조건에서 중첩 교차검증, 보정, 오경보, 앱 조건 평가 | fold별 표, 평균 ± 표준편차, 부트스트랩 구간 저장. 결과가 어떻든 보고 |
| E3 부가 분석 | 나머지 정규화 조건, 속도 특징 제외, 순서형 라벨, 15 fps 모사 평가(설정은 E2 것을 고정) | 표 저장, "부가"로 표시 |
| E4 최종 모델 | 60명 전체에 같은 선택 절차(5-fold)를 한 번 적용해 계열·설정·학습 길이·온도를 정한 뒤 60명으로 학습. 내보내기, 일치 검증, 지연 측정, INT8, D11 판정 | `export/drowsiness_v1.onnx`, `export/feature_spec.json`, `export/rules.json`, 검증 기록 |
| M4 문서와 데모 확정 | MODEL_CARD.md, README, SPEC.md 확정, 실제 모델로 모델 벡터 재생성, 팀 수집 영상의 시간당 경고 수 | 체크리스트 전 항목 |

- 시험 fold 결과를 본 뒤에 식이나 특징을 고치면 그 사실과 이전 표를 리포트에 남긴다.
- 최종 모델의 시드는 0으로 고정한다. D11은 E2의 주 결과 행(앱 조건 평가)으로 판정하고 내보낸 파일로 판정하지 않는다.
- 데이터가 끝내 오지 않으면 M0~M2 산출물(이식 대상 구현, 규칙 엔진, PERCLOS, 벡터, 계약을 만족하는 무작위 가중치 ONNX, 데모, README)이 납품물이다. 이 경우 closure 상수는 실제 눈으로 검증되지 않은 잠정값이라고 문서에 적는다.

---

## 12. 위험과 한계

| 위험 | 대응 |
|------|------|
| 3클래스 성능이 낮게 나올 수 있다. 피험자 분리 조건의 문헌 결과는 대체로 50~65%이고 low vigilant 재현율은 우연 수준에 가깝다 | 기대 범위를 미리 문서화. D11의 기본 동작 규칙. 규칙 엔진과 PERCLOS는 모델과 무관하게 동작 |
| 세션 교란(조명·카메라·프레임률을 학습) | 녹화 조건을 드러내는 특징 제외, 교란 점검 두 가지, 앱 조건 평가, 세 번째 정규화 조건, 속도 특징 제외 결과 병기 |
| Android와 수치 불일치(반전, 회전 방식, 타임스탬프, `numFaces`, 버전, GPU delegate) | SPEC.md에 전제를 고정하고 테스트 벡터로 확인. 실기기는 앱 팀이 검증 |
| closure 상수와 감김 임계값이 실제 눈 감김으로 검증되지 않았다 | E0에서 확정, 통과 기준 미달 시 학습 전에 멈춤 |
| 아래 시선·고개 숙임·하품이 감김으로 읽힌다 | 문서화. 규칙용 임계값을 특징용보다 높게 둠. 팀 수집 영상으로 시간당 경고 수 확인 |
| 기기 프레임률과 평활 때문에 속도·진폭 특징이 학습 때와 달라진다 | 15 fps 모사 추출을 학습과 평가에 포함, 속도 특징 제외 결과 병기 |
| 오경보 수치의 근거가 alert 영상 약 6시간뿐이다 | 구간과 영상 수 기준 값을 함께 보고, 팀 수집 영상으로 보완 |
| RLDD는 실내에서 화면을 보는 영상이라 주행 환경으로 옮겨지지 않을 수 있다 | README와 모델 카드에 명시. 용도는 보조 안내이며 안전 시스템이 아님 |
| MediaPipe 모델 카드가 "생명에 관련된 결정에는 쓰지 말 것"이라고 명시한다 | 모델 카드에 인용 |
| 60초 기준선은 문헌의 10~15분보다 짧다 | 완성 조건 미달 시 연장, 대체 표시 |
| RLDD 라이선스가 명시돼 있지 않다 | 사용 여부는 팀 판단. 영상과 얼굴은 재배포하지 않음 |
| 실제 RLDD 파일의 회전·타임스탬프 이상 | 영상별 회전 자동 점검(0·90·180·270도 중 검출률 최고), 설정의 수동 보정표, 품질 리포트 |

---

## 13. 조사 과정에서 한 일 (투명성)

- 스크래치 폴더에 pip 패키지, 공개 `face_landmarker.task`, MediaPipe 공식 저장소의 테스트 이미지 2장과 표준 얼굴 모델 파일을 받아 실험했다. UTA-RLDD 영상은 받지 않았고 미러의 파일 목록 메타데이터만 조회했다.
- 조사 에이전트 하나가 공개 저장소(openpilot)의 파일 목록을 보려고 로그인된 `gh` CLI를 한 번 썼다. 프로젝트 정보는 보내지 않았지만 요청에 이 PC의 GitHub 계정 인증이 실렸다. 이후 작업에서는 인증 없는 조회만 쓴다.
- 1.x대 mediapipe는 시스템 라이브러리가 없어 스크래치 전용 대체 라이브러리로 돌렸다. 0.10.32와 0.10.21은 그런 조치 없이 실행했다.

---

## 14. 구현 중 달라진 점

이 장은 승인(2026-11-13) 뒤에 더한 것이고 2026-11-15의 코드 기준이다(세 검증자의 4차 검증 뒤 정비까지). 표 (a)는 계획과
구현의 차이 가운데 구현하면서 정한 것, (b)는 소유자의 답을 기다리는 질문, (c)는 소유자가 2026-11-13에 정한 네 가지 동작과
2026-11-14에 정한 한 가지(5번)다. (c)의 번호 1~5는 작업 기록의 D1~D5이고 1장의 D 번호와 다르다. 표 (a)의 O1~O14와 G1~G10은 정비(2026-11-13~14에 독립 검토의
지적을 고치고 빠진 항목을 채운 작업)의 지시 번호와 빠진 항목 번호이고, R1~R11은 2026-11-14에 세 검증자가 정비 결과를 다시
확인한 뒤(2차 검증) 내린 지시 번호다. 같은 날 세 검증자가 두 번 더 확인했고(3차, 4차 검증), 그 지적을 고친 "3차 검증 뒤의
정비"와 "4차 검증 뒤의 정비"(2026-11-14~15. 4차 검증은 막는 결함도 큰 결함도 찾지 않았고 작은 지적 13개를 냈다)는 지시 번호
없이 내용으로 적었다(표 (a)의 마지막 두 묶음). 행마다 뜻을 함께 적었다. `검토 <영역>-NN`(예: `검토 frame-04`,
`검토 rules-06`)과 `검토 K1`~`K8`은 2026-11-13 독립 검토가 붙인 지적 번호다. 그 검토 보고서는 저장소에 없으므로 번호만으로는
찾아볼 수 없고, 번호가 나오는 칸에는 지적의 내용을 말로 함께 적었다. 명령과 결과 파일의 설명은 README 4장, 동작 변경의 앱 쪽
영향은 README 8장과 13.3절에 있다. 표 (b)는 README 9장, `docs/SPEC.md` 20장(2~12번)과 같은 목록이고, 세 문서 모두 2026-11-14에 결정된 1번을 비워 두었다.
14.4절은 같은 날 소유자의 요청으로 더한 실시간 시험 도구이고, 계획과 구현의 차이가 아니라 이 계획서의 산출물 밖에 더한 것이다.

### 14.1 표 (a): 계획과 구현의 차이 (구현하면서 정한 것)

"되돌리기"는 소유자가 다르게 원할 때 바꿀 곳이다.

**9장의 시그니처와 7장의 파일 이름**

| 계획서 | 코드 | 이유 |
|--------|------|------|
| `ClosureRunDetector(enter_thr, exit_thr, spec)` | `ClosureRunDetector(settings: RunDetectorSpec)`, `run_start_ms() -> int` 추가 | 임계값과 시간 상수를 설정 묶음 하나로 받는다. `run_start_ms`는 (c)의 결정 1 |
| `StepAggregator(spec, start_ms)` | `StepAggregator(spec, start_ms, origin_ms)`, `update(state, run, ongoing_ms) -> list[StepFeatures]` | 세션 원점(첫 프레임의 `t_ms`)을 따로 받아 스텝 번호를 정한다 |
| `DriverPipeline.update(...) -> PipelineOutput` | `-> PipelineOutput \| None`. `PipelineOutput`에 `step_probs`(스텝마다 확률 또는 없음) | 직전 이하의 `t_ms`는 버리고 `None`. `step_probs`는 결정 3 |
| `RuleEngine`, `AlertPolicy`(시그니처 없음) | `RuleEngine(cfg).update(state)`, `on_step(step, probs)`, `AlertPolicy.decide(events, speed_kmh, t_ms)`, `StepHysteresis.reset()`, `ForwardZone.candidate_pending` | `reset`은 결정 4. `candidate_pending`(전방 주시 후보가 확정 전인지)은 O4(얼굴 소실 인정 구간의 시선 이탈)를 구현하면서 더한 조건이고 표 (b) 9번에서 확인을 기다린다 |
| `iter_video_frames(path)` | `iter_video_frames(path, rotation=None, counts=None)` | 수동 회전, 디코딩 계수. 표시 행렬에 뒤집기가 든 영상은 회전을 주지 않으면 거부 |
| `extract_video(path, model_path, out_parquet, drop_sim)` | 키워드 `spec`, `rotation`, `extra_metadata` 추가. `output_is_current(out_parquet, spec, drop_sim, rotation, source=None)`, `read_previous_timings(own_csv, pilot_csv=None) -> PreviousTimings` 추가 | 회전 점검 결과와 추출 요약을 parquet 메타데이터에 적는다. 명령의 네 입력(`--video`, `--input-dir`, `--dataset`, `--unlabeled`)이 모두 회전을 자동 점검하고, `--video`는 `--rotation`을 받는다(R5). 3차 검증 뒤: 원본의 파일 이름과 크기(`source_name`, `source_size_bytes`)를 적고 이어 하기가 크기를 비교한다. 건너뛴 영상의 처리 시간은 두 리포트 가운데 나중의 것에서 잇는다 |
| `replay_video(frames, spec, baseline, feature_start_ms)` | 키워드 `rules` 추가 | `DriverPipeline`이 규칙 설정을 요구한다 |
| `build_step_table(...) -> pd.DataFrame` | `-> StepTable`(스텝 표와 영상 품질 표), 키워드 `gate`, `workers`, `frame_drop`. `raw_content_digest(path, *, refresh=False)` | 품질 표(D7)를 같은 재생에서 만든다. 캐시는 `load_step_tables`, 프레임 누락 변형은 `load_frame_drop_tables`. 캐시의 내용 지문은 추출 파일 전체의 sha256이다(3차 검증 뒤) |
| `fit(train, val, cfg, seed)` | `fit(train, val, cfg, seed, extra_train: Sequence[TrainSource] \| None = None)`, 표집기 `BalancedSampler(metas, seed, extra_shares)` | 학습 전용 출처(15 fps 모사 추출, 프레임 누락 변형)를 몫과 함께 섞는다(G1) |
| `run_nested_cv(cfg)`, `fit_final(cfg)` | 키워드 `arm`, `families`, `fast`, `run_name`, `overwrite`, `allow_partial_fps_sim`, `rebuild`. `run_nested_cv`는 `variant`(`base`, `ordinal`, `no_velocity`), `fit_final`은 `out_dir`, `version`도. `CVData.rotation`, `print_rotation_check`, `rotation_note`, `rotation_record` 추가 | 명령의 옵션과 부가 실행(G2, G3). 학습 명령의 회전 점검 줄과 실행 기록의 `rotation_check`(3차 검증 뒤) |
| `alarm_report(preds, cfg, source)`, `subject_bootstrap(preds, metric, n)` | `alarm_report(preds, cfg, source, videos=None, sweep=())`, `subject_bootstrap(table, metric, n, seed, level)` | D7의 분모, 임계값 곡선, 재현성 |
| `run_lgbm_cv(cfg)` | 키워드 `arms`, `fast`, `run_name`, `closure_w`, `overwrite`, `rebuild`, `no_velocity`, `foldwise_global`. `run_lgbm_w_grid` 추가 | E1의 W 비교, 부가 실행(G3, G5) |
| `export(model, spec_path, out_path, meta)`, `verify_contract(onnx_path, spec_path)` | `export(..., dynamo=False)`, `verify_contract(..., allowed_ops=...)`, `verify_file_name`, `refuse_synthetic_export`, `refuse_fast_export`, `is_export_dir_or_below(out_dir, export_dir)`, `protected_export_dir(out_dir, export_dir) -> str \| None`, `REPO_ROOT`, `SHIPPED_EXPORT_DIRNAME` 추가 | 대체 내보내기 경로, INT8의 연산자 집합(파일 종류도 이것으로 안다), 파일 이름 검사, 내보내기 보호 장치와 그 기준 폴더(R11), 늘 지키는 저장소의 `export/`와 logits 노드 검사(3차 검증 뒤) |
| `run_demo(video, onnx_path, out_dir, speed_kmh)` | 키워드 `cfg`, `rotation`, `on_rotation`. `run_demo_raw`(추출 parquet 입력), `run_pipeline`(속도를 함수로도 받음), `VideoRotation`, `resolve_video_rotation`, `describe_video_rotation`, `RawRotation`, `read_raw_rotation(metadata)` 추가 | `--raw` 데모, 라벨 없는 영상 평가가 같은 재생을 쓴다. 영상 입력의 회전(`--rotation` 또는 자동 점검 한 번)을 정하고 출력, JSON, 그림에 적는다(R5). `--raw` 입력은 추출이 남긴 회전 기록을 한 줄로 알린다(3차 검증 뒤) |
| 7장 `export/feature_spec.json` | `export/feature_spec_v{N}.json`(모델 버전마다 하나) | 버전마다 짝을 남긴다. 소유자 확인 대기(표 (b) 10번) |
| 9장 "실행 명령은 모두 `--config` 형식" | `drowsy.spec`, `drowsy.vectors`는 `--spec`, `--rules`를 받는다 | 실험 설정이 아니라 명세 파일을 다룬다 |

**새 모듈, 명령, 설정 키, 형식**

| 무엇 | 내용 |
|------|------|
| `drowsy/signal_check.py` | E0 점검과 상수 제안(`e0`), 15 fps 깜빡임 길이 오차 모사(`blink-sim`) |
| `tests/test_leakage.py` | 결과만 보는 누설 회귀 시험. 피험자 넷(1, 13, 25, 26) 가운데 시험 fold 3의 피험자 25를 망가뜨린 사본과 원본으로 명령 여덟 개를 돌리고, fold 3의 선택 기록이 같은지와 손대지 않은 피험자 26의 예측 행이 모든 묶음(44개)에서 같은지 본다(R10). 3차 검증 뒤에는 26의 라벨 파일만 맞바꾼 셋째 데이터를 더해, 평가받는 피험자 자신의 라벨이 그 피험자의 확률에 닿지 않는지도 본다. 시험 37개, 다른 작업이 없는 이 PC에서 혼자 돌려 54.4초 |
| 새 명령과 옵션 | `extract --unlabeled`, `--run-name`; `dataset --rebuild`, `--frame-drop-variants`, `--frame-drop-seed`; `signal_check e0`, `blink-sim`; `baseline_lgbm --rebuild`, `cv --all-w`, `--no-velocity`, `--foldwise-global`, `--overwrite`; `train --rebuild`, `cv --ordinal`, `--no-velocity`, `cv`·`final --overwrite`, `--allow-partial-fps-sim`; `evaluate unlabeled`; `extract --rotation`(`--video`에만), `demo --rotation`(`--video`에만). `export_onnx --checkpoint`는 `--spec`, `--version`이 필수. `--run-name`이 있는 모든 명령은 폴더 이름 하나만 받는다(R7) |
| 새 설정 키 | `extraction.rotation_roll_margin_deg`(15.0), `extraction.unlabeled_rotation_overrides`; `training.ordinal_neighbor_mass`(0.2), `training.frame_drop_share`(0.25), `training.frame_drop_variants`(1). 모두 코드가 읽는다 |
| 바뀐 기본값 | `training.fps_sim_share` 0.5 → 0.25. 두 증강이 다 있으면 원본 추출의 몫이 0.5다 |
| 체크포인트 형식 | `drowsy-checkpoint-v2`: 학습한 spec 사본의 sha256, 합성 데이터 여부(`synthetic`), 축소 설정 여부(`fast`, 나중에 더한 키라 없으면 거짓으로 읽는다). v1은 거부한다 |
| 스텝 표 캐시 | 형식 4: 열쇠에 이식 코드와 데이터 모듈의 소스 지문(형식 2)과 추출 파일마다의 내용 지문을 더했다. 내용 지문은 형식 3(R10)에서 parquet 꼬리말(파일 끝의 열 구성과 열별 통계)의 sha256이었고, 형식 4(3차 검증 뒤)에서 파일 전체의 sha256이다(한 명령 안에서는 경로, 크기, 수정 시각별로 기억한다) |
| 리포트 형식 | 예측 표에 `feature_source`, `variant`, `supplementary` 열(옛 17열 파일도 읽는다). summary.md 8절(기준선과 창 수), 9절(부가 결과), `supplementary_*.csv`. `alerts.csv`에 `immediate_alerts`, `deferred_alerts`. 데모 JSON의 경고에 `kind` |
| 벡터 형식 | `expected.step_probs`(스텝마다의 확률). 벡터 형식은 소유자 결정 대기(표 (b) 12번)이지만 결정 3을 고정하려고 구현하면서 더했다(R9). 지금 102개, 30,588,827 bytes(3차 검증 뒤에 8개, 4차 검증 뒤에 13개를 더했고 앞의 81개와 89개는 바이트 단위로 그대로) |

**정비 작업의 지시(O1~O14)**

| 지시 | 구현 | 되돌리기 |
|------|------|----------|
| O1 감는 속도의 되돌아보기 | 쓸 수 없는 쌍(직전 유효 표본 없음 또는 간격 > `max_dt_ms`) 뒤의 유효 표본에서 `recent_up`(최근 상승률 버퍼)을 모두 비운다. spec 검증이 `max_dt_ms ≤ hold_ms`를 요구한다. 무효 프레임 하나만으로는 끊지 않는 것은 구현자의 해석이다(무효 프레임마다 끊는 안과 정비 전 벡터 43개와 무작위 3,137구간에서 차이 0. 지금은 벡터 `runs_velocity_after_gap`의 한 장면이 이 해석을 고정한다). "진입 쌍이 쓸 수 있어야 유효"는 넣지 않았다(표 (b) 3번) | 해석: `ClosureRunDetector.update`의 무효 프레임 갈래에서 버퍼를 비우고 시험 하나의 기댓값을 11.25 → 1.25로, 벡터 `runs_velocity_after_gap`을 다시 만든다 |
| O2 보간 시각 | `prev_t + round_half_up(fraction × (t − prev_t))`. 시각 원점에 따라 달라지지 않는다 | (결함 수정) |
| O3 VATS confidence | VATS 시작 때의 자세 유효 누적값을 두고 [t_start, 발행 시각]의 유효 시간 비율로 계산한다. 30초보다 긴 VATS의 `end`만 바뀐다 | (결함 수정) |
| O4 얼굴 소실 인정 구간의 시선 이탈 | 인정 구간에서 기한이 된 `gaze_away_long`을 다음 분석 가능 프레임에서 낸다. 구현자가 검증자의 조건에 "표지가 알 수 없음인 프레임에서는 앞을 보는 후보가 확정 전이 아닐 것"을 더했다. 더하지 않으면 0.4초 곁눈질이 정확히 3,200 ms 뒤 얼굴 없는 프레임에서 3.6초 이탈로 시작된다. 남는 귀결: 이탈이 3초에 닿은 뒤 앞을 본 프레임 하나와 3.2초 넘는 끊김 뒤 얼굴 없는 프레임이 오면 그 프레임에서 시작이 난다. 더한 조건은 2차 검증 뒤 소유자의 확인을 기다리는 질문으로 옮겼다(R1, 표 (b) 9번) | `rules.py`의 시선 규칙에서 `candidate_pending` 조건을 빼고 경계 시험 하나를 지운다. 벡터 `rules_gaze_credit_boundary`가 바뀐다 |
| O5 회전 점검 | 최고 얼굴 검출률(표본 하나 차이까지 같음) 다음 roll 중앙값의 절댓값이 가장 작은 각도. 가르지 못하면 `rotation_ambiguous`로 표시하고 메타데이터 각도를 둔다. 12장 마지막 행의 "검출률 최고"는 이 규칙으로 대체됐다. 2차 검증 뒤에는 모든 추출 입력과 영상 데모가 이 점검을 한다(R5). 구현자가 정한 것: 가르지 못한 각도에 메타데이터 각도가 없으면 roll이 가장 작은 각도를 고르고 표시한다(측정 8건 중 6건 맞음), 차량 영상은 보정표를 따로 둔다, 표시 행렬에 뒤집기가 든 영상은 수동 회전 없이 거부한다 | `extract.choose_rotation`에서 메타데이터 각도를 두고, 보정표를 하나로 합친다 |
| O6 실제 데이터 경로 | git에 넣지 않는 `config/local.yaml`(기본 설정의 사본)을 `--config`로 준다. 테스트의 설정은 `rldd_root`를 직접 적는다 | |
| O7 모델 카드 | 우리 결과의 범위를 예상한 문장을 지웠다. 문헌 값은 문헌 값으로 남겼다(12장 첫 행의 "기대 범위를 미리 문서화"는 하지 않는다) | |
| O8 protobuf | `demo.py`는 `google.protobuf`를 import하지 않는다. 읽을 수 없는 ONNX 파일은 `export_onnx.read_metadata`가 `ValueError`(ContractError)로 알린다 | |
| O9 체크포인트 | v2 형식. `--checkpoint`에 `--spec`, `--version` 필수. spec의 해시는 `export()`가 쓸 사본의 바이트로 비교한다(내용이 같고 들여쓰기만 다른 파일은 받는다). 합성 데이터 모델은 `paths.export_dir`과 그 아래로 내보내지 못한다(`train final`과 `export_onnx`가 같은 함수를 쓴다) | |
| O10 LightGBM | 원본 프레임률 추출로만 학습한다(15 fps 모사 추출, 프레임 누락 변형 모두 없이). 리포트 첫 문단과 명령 출력에 적는다 | |
| O11 검토 K5(끊긴 동안 발행되는 스텝) | 동작 그대로, 문서화만. SPEC.md 11.6절과 벡터 `steps_closure_across_gap`에 있다. 소유자에게 묻는 질문이 아니라 정비에서 정한 동작이다(R3) | |
| O12 오프라인 오경보 평가의 재시작(결정 4의 후속) | `evaluate.alarm_videos`가 한 세션 안에서 스텝 번호 k가 건너뛰는 곳(10초 넘는 끊김 뒤에만 생긴다)에서 판정 상태와 두 판정기를 새로 시작한다. 모델 판정이 앱과 같아졌다(무작위 450세션, 시작 946번, 불일치 0). 남은 차이(가정): PERCLOS와 either 판정은 예측 행이 시작되기 전 119스텝 안에서 `perclos_60`이 시작 비율에 닿으면 앱이 먼저 시작할 수 있다 | `alarm_videos`의 `elif step_k != previous_k + 1:` 갈래를 비운다 |
| O13 축소 설정 체크포인트(검토 training-10의 남은 부분) | 체크포인트 v2에 `fast`. `train final`이 설정에서 적고, `export_onnx --checkpoint`는 `fast`인 체크포인트를 `paths.export_dir`과 그 아래로 내보내지 못한다 | `export_onnx._run_cli`의 `if checkpoint.fast: refuse_fast_export(...)` 두 줄을 지운다 |
| O14 경고 종류와 모델 호출마다의 점(결정 2, 3의 후속) | 데모와 라벨 없는 영상 리포트가 경고를 `immediate`(바로), `deferred`(막혔다가 나중에), `repeat`(되풀이)로 나눈다. 데모는 모델 호출마다 확률 점 하나를 그린다 | `demo.AlertKinds` 대신 "그 프레임의 이벤트에 없는 경고 = 되풀이" 표시로, 점은 프레임마다 `out.probs` 하나로 되돌리고 `alerts.csv`의 두 열을 뺀다 |

**빠져 있던 계획 항목(G1~G10, 4.4절, E1, E4)**: 모두 구현했다.

| 항목 | 방법 | 되돌리기 또는 끄기 |
|------|------|---------------------|
| G1 증강 (2): 프레임 누락 변형(5.2절) | 세션마다 [`frame_drop_min`, `frame_drop_max`]에서 고르게 뽑은 비율로 저장된 프레임을 하나씩 버리고(첫 프레임은 남김, sha256 시드) `DriverPipeline`으로 다시 재생해 변형별로 캐시한다. 기준선과 T_b는 솎지 않은 프레임에서 구한다. 신경망 학습이 원본 0.5, 15 fps 모사 0.25, 변형 0.25의 몫으로 섞는다(가정). 학습 피험자의 창만 쓰고 검증, 시험, 온도는 원본 창만 쓴다. 표준화 상수도 같은 몫으로 가중한다 | `training.frame_drop_variants: 0`(남는 몫은 원본으로), 몫은 `fps_sim_share`, `frame_drop_share` |
| G2 순서형 소프트 라벨(부가) | `train cv --ordinal`: 라벨 스무딩 대신 [0.8, 0.2, 0], [0.1, 0.8, 0.1], [0, 0.2, 0.8](`ordinal_neighbor_mass` 0.2). 조기 종료와 온도는 정답 라벨로. `<실행>/ordinal/` | 명령을 돌리지 않는다 |
| G3 속도 특징 #7, #8을 뺀 실행(부가) | `train cv --no-velocity`: 표준화한 두 입력을 학습(잡음 증강 포함), 검증, 평가에서 0으로 둔다(F = 17 그대로). `baseline_lgbm cv --no-velocity`: 통계 열 12개를 뺀다(102 → 90) | 명령을 돌리지 않는다 |
| G4 15 fps 모사 추출에서의 평가(부가) | fold 모델을 시험 fold의 15 fps 추출로도 평가한다(학습 조건과 앱 조건). LightGBM fold 모델도 같다(검토의 빠진 항목 설명을 따라 더했다). 추출이 시험 fold 하나를 통째로 빠뜨리면 이유를 적고 건너뛴다 | 끄는 설정은 없다(그 추출이 있으면 돈다) |
| G5 fold별 전역 상수 민감도(4.3절, 부가) | `baseline_lgbm cv --arms global --foldwise-global`: 바깥 fold마다 학습 피험자로만 전역 상수(중앙값)를 구해 전역 조건의 스텝 표를 다시 만든다. spec 상수의 묶음이 주, fold별 상수의 묶음이 부가 | 명령을 돌리지 않는다 |
| G6 D16 fold 모델의 INT8 비교 | fold마다 다시 학습한 TCN을 임시 폴더로 내보내 QUInt8 동적 양자화하고, float ONNX와 INT8을 onnxruntime으로 시험 fold에 돌린다(`tcn_onnx`, `tcn_onnx_int8`, 짝지은 비교 둘). `train final`의 INT8 수치는 확률 차이뿐이라고 적는다 | `export.int8_if_tcn: false` |
| G7 M3 추출 리포트 | `reports/<실행>/extraction/`에 영상마다 상태, 사유, 프레임, 길이, fps, 가변 간격, 해상도, 회전과 이유, 모호 표시, 얼굴 비율, 프레임당 ms, 밝기. 시험 추출 리포트 포함. 밝기 등 영상 요약은 parquet 메타데이터 키(`SCHEMA_VERSION` 그대로) | |
| G8 E0 도구 | `signal_check e0`. 라벨, 클래스, fold를 계산 전에 버린다. 구현자가 정한 것: 판정 단위는 세션(파일)이라 RLDD에서 분모가 182, `off_axis` 기준선은 완성으로 침, 제안 상수는 소수 넷째 자리, 합성 데이터의 제안 파일 이름은 `feature_spec_proposed.synthetic.json` | `signal_check.py` |
| G9 M4 차량 영상 | `extract --unlabeled`, `evaluate unlabeled`. `demo`의 재생 함수를 다시 쓴다. 구현자가 정한 것: 분모는 얼굴이 검출된 시간, 경고 수에 나중에 낸 경고와 되풀이를 넣고 따로도 싣는다, 속도를 주지 않으면 억제하지 않는다 | `evaluate.run_unlabeled` |
| G10 15 fps 깜빡임 길이 오차 모사(4.5절) | `signal_check blink-sim`. 수치는 SPEC.md 17.3절에 모사로 표시해 실었고 `tests/test_spec_doc.py`가 다시 계산해 비교한다 | |
| 4.4절 리포트 항목 | `videos.csv`의 기준선 열, summary.md 8절(연장·대체 기준선 영상, alert 기준선 대체, 조건별·보폭별·클래스별 창 수), `run.json`의 기준선 요약 | |
| E1 W 비교를 한 명령으로 | `baseline_lgbm cv --all-w`, `lgbm_w_grid/`의 합친 표 | |
| E4 `rules.json`을 모델 옆에 | `final.json`에 `rules_hash` | |
| 누설 회귀 시험(6장의 피험자 분리) | `tests/test_leakage.py`. 저장소 사본에 일부러 넣은 누설 10가지 중 8가지를 잡는다(2차 검증 뒤에 더한 평가 때의 누설 둘, 곧 시험 fold로 표준화하기와 시험 fold로 온도 맞추기를 포함). 3차 검증 뒤에는 평가받는 피험자 자신의 라벨을 쓰는 누설 넷(자기 시험 라벨로 온도 다시 맞추기, 확률에 정답 라벨 섞기 세 곳)도 잡는다(작업자의 기록). 나머지 둘(15 fps·변형 출처를 학습 피험자로 거르지 않음)은 표집기와 표준화가 원본 학습 집합에 없는 영상을 쓰지 않아 출력이 바뀌지 않는다(선택 파일 16개와 예측 표 16개가 바이트 단위로 같았다). 그 경우는 `tests/test_train.py`의 단위 시험이 막는다 | 시험만 |
| 교란 점검 (1)의 녹화 메타데이터(6장) | 추출 메타데이터의 밝기, 해상도, 평균 fps를 읽고 가변 프레임률 여부를 더했다. 없으면(합성) 빈 열로 두고 리포트에 적는다 | `baseline_lgbm.RECORDING_COLUMNS` |

**2차 검증 뒤의 정비(R1~R11, 2026-11-14)**: 세 검증자가 정비 결과를 다시 확인한 뒤의 지시다. 이식 대상 모듈(`signals`,
`calibrate`, `blink`, `step_features`, `rules`, `pipeline`, `types`)은 한 바이트도 바꾸지 않았고, `pipeline_hash`, `rules_hash`,
`export/drowsiness_v0.onnx`, ONNX 계약은 그대로다.

| 지시 | 구현 | 되돌리기 |
|------|------|----------|
| R1 얼굴 소실 인정 구간 뒤 시선 이탈 조건 | 코드 변경 없음. O4에서 구현이 더한 조건("전방 주시 후보가 확정 전이 아닐 것")을 소유자 확인 대기로 옮기고, 잰 경계(7,200 ms이면 이벤트 없음, 7,201 ms이면 시작)를 적었다 | (질문, 표 (b) 9번) |
| R2 늦게 나가는 미뤄 둔 경고 | 코드 변경 없음. 결정 2의 귀결에 (e) 앞을 본 지 13초 뒤의 `vats` 경고, (f) 정확히 10초 끊김 뒤 함께 나가는 경고를 더했다 | (질문, 표 (b) 8번) |
| R3 기다리는 질문의 목록 하나 | 문서만. README 9장, `docs/SPEC.md` 20장(그때 1~12번. 1번이 2026-11-14에 결정된 뒤로는 2~12번), 이 장의 표 (b)가 같은 목록이다. "끊긴 동안 발행되는 스텝"(O11)은 질문이 아니라 정비에서 정한 동작으로 옮겼다 | |
| R4 (a) 회전을 가르지 못한 영상 알리기 | `python -m drowsy.dataset`, `signal_check e0`, `evaluate unlabeled`가 한 함수(`dataset.read_rotation_checks`)로 추출 parquet의 회전 점검 기록과 보정표를 읽어 한 줄(분류 다섯: 자동 점검, 수동 보정, 가르지 못했고 보정 없음, 보정표의 각도가 적용되지 않음, 기록 없음)과 표시된 영상 목록을 낸다(E0은 summary.md 절과 criteria.json, 라벨 없는 영상 평가는 summary.md 4절, 스텝 표 명령은 출력). 영상을 빼지 않고 종료 코드도 그대로다. (b)는 표 (b) 17번 | 세 명령에서 `read_rotation_checks`, `rotation_check_line`을 부르는 곳을 지운다 |
| R5 회전 점검이 없던 명령 | `extract --input-dir`, `--video`도 회전을 점검하고(보정표가 우선), `extract --video`와 `demo --video`가 `--rotation {0,90,180,270}`을 받는다(보정표보다 우선). 데모는 점검을 한 번 돌려 각도와 가르지 못함 여부를 첫 출력 줄, JSON(`rotation_source`, `rotation_ambiguous`, `rotation_check`, `display_flip`), 그림 머리말에 적는다. 뒤집기가 든 영상은 `--rotation`을 알려 주며 거부한다. 기록 없는 옛 `--input-dir`·`--video` 결과는 다시 추출한다 | `extract._extract_all`에서 점검 설정을 `--dataset`, `--unlabeled` 작업에만 주고, 두 명령의 `--rotation`과 `demo.resolve_video_rotation`을 뺀다 |
| R6 영상과 사본 설정의 git 제외 | `.gitignore`에 `videos/`, 영상 확장자 다섯 가지(`mp4`, `mov`, `m4v`, `avi`, `mkv`)의 소문자·대문자, `config/local*` | 그 줄들을 지운다 |
| R7 실행 이름 | `--run-name`이 있는 모든 명령이 다른 일을 하기 전에 `config.check_run_name`을 부른다. 빈 값, `.`, `..`, `/`·`\`가 든 이름은 종료 코드 2와 한 줄 `error: --run-name must be a plain folder name, got '<값>'` | 각 모듈 `main()`의 `check_run_name` 호출을 지운다 |
| R8 추출 처리 시간 | summary.md 6절이 이번 실행에서 잰 값과 전 리포트에서 이어 온 값을 따로 적고, "N개씩 동시에 처리했다"는 앞의 것에만 붙인다 | `extract.write_extraction_report`의 6절에서 `_timing_carried`로 나누는 부분을 하나로 합친다 |
| R9 벡터 형식의 `step_probs` | 구현자가 정한 것으로 적었다. 형식과 크기의 질문은 표 (b) 12번에 남는다 | `drowsy/vectors.py`에서 그 항목을 빼고 다시 만든다 |
| R10 누설 회귀 시험 | 위 "새 모듈" 표의 `tests/test_leakage.py` 행. 시험만 바뀌었다 | 시험 파일을 전의 판으로 |
| R10 계약 검사의 온도 | `verify_contract`: logits를 바꿀 수 있는 것은 Softmax 바로 앞의 `temperature` 상수로 나누기뿐이다. 기본 경로와 INT8 파일은 온도 1.0이어도 그 상수가 있어야 하고, 대체 경로(`--dynamo`) 파일만 온도 1.0일 때 없을 수 있다. logits가 Div나 Mul의 출력이면 거부한다. 계약 값은 그대로. 3차 검증 뒤에 logits를 내는 노드의 검사가 더해졌다(아래 묶음) | `export_onnx._check_graph_constants`에서 `temperature_may_be_absent` 갈래와 Div·Mul 검사를 뺀다 |
| R10 다른 조건 실행의 D11 | `train cv --arm`이 `training.export_arm`이 아니면 D11 판정을 하지 않고(`summary.json`의 `gate`가 `null`) 설명과 출력에 그렇게 적은 뒤 리포트를 끝까지 쓴다(전에는 학습 뒤 종료 코드 2) | `train.run_nested_cv`의 판정 조건에서 `not other_arm`을 뺀다 |
| R10 예측 표의 부가 표시 | `evaluate.validate_predictions`: 출처가 `native`가 아니거나 변형이 `base`가 아닌 묶음은 `supplementary`가 참이어야 하고, 열이 없으면 두 열에서 정한다. 기본 표시의 묶음(ONNX, INT8, 다른 조건 실행)은 가진 표시를 쓴다 | 그 검사와 채우기를 뺀다 |
| R10 스텝 표 캐시의 내용 지문 | 캐시 열쇠에 parquet 꼬리말의 sha256(위 "스텝 표 캐시" 행). `--rebuild`는 그대로 남는다. 3차 검증 뒤에 파일 전체의 sha256으로 바뀌었다(아래 묶음) | `dataset._source_fingerprint`에서 `raw_content_digest`를 빼고 `CACHE_FORMAT`을 2로 |
| R11 내보내기 보호 장치의 기준 폴더 | 상대 경로 `paths.export_dir`을 지금 폴더와 저장소 루트(`drowsy` 패키지의 부모) 둘 다에 대어 본다. `export_onnx.is_export_dir_or_below` 하나를 `train.fit_final`도 쓴다(`train._inside`는 없어졌다). 저장소 밖에서 명령을 돌려도 합성·축소 설정 모델이 저장소의 `export/`에 쓰이지 않는다. 3차 검증 뒤에는 설정이 다른 폴더를 가리켜도 저장소의 `export/`를 늘 지킨다(아래 묶음). 판단은 지금 `protected_export_dir`이 한다 | `protected_export_dir`에서 설정 값의 `REPO_ROOT` 기준 비교를 뺀다 |

**3차 검증 뒤의 정비(2026-11-14)**: 세 검증자가 2차 검증 뒤의 정비를 다시 확인한 뒤(3차 검증) 고친 것이다. 이식 대상 모듈
일곱은 이번에도 한 바이트도 바꾸지 않았고, `pipeline_hash`, `rules_hash`, `export/drowsiness_v0.onnx`, ONNX 계약, 앞의 벡터
81개는 그대로다.

| 무엇 | 구현 | 되돌리기 |
|------|------|----------|
| 벡터 8개 추가 | 참조 구현의 한 줄을 틀리게 바꿔도 앞의 81개가 모두 통과하던 자리를 고정한다: 30초 창의 시작에서 끝나는 구간(`rules_vats_window_edge`), 정점 전의 두 증가 가운데 큰 값(`runs_two_rises_before_peak`), 얼굴을 잠깐 놓친 `vats`와 `microsleep`의 `confidence`(`rules_vats_confidence_face_lost`, `rules_microsleep_confidence_face_lost`), 한 프레임 곁눈질(`rules_gaze_one_frame_glances`), 다른 종류의 시작 뒤에도 남는 미뤄 둔 경고(`rules_alert_pending_survives_onset`), 첫 `t_ms`가 작은 세션의 처음 10초(`rules_unable_small_t0`), 스텝 끝에서 나는 `drowsy`(`rules_drowsy_off_step_boundary`). SPEC 16.1절에 표와 "벡터가 고정하지 않는 동작"의 목록이 있다 | 시험과 벡터만이다(되돌릴 동작이 없다) |
| 감김 검출기 설정의 `hold_ms < clock.reset_gap_ms` | 설정을 읽을 때 두 검출기 모두에 검사한다(`spec._validate_run_detector`). 지금 값 500과 10,000은 통과하므로 해시가 그대로다. 어기는 설정에서는 10초 넘는 끊김 앞뒤의 감김을 가르는 것이 규칙 엔진의 `reset()` 호출 하나뿐이 된다 | 그 검사를 지운다 |
| 학습 명령의 회전 점검 줄 | `train`(cv, final)과 `baseline_lgbm`(cv, probes)이 학습 전에 회전 점검 줄과 표시된 영상을 출력하고, `run.json`·`final.json`·`probes.json`의 `rotation_check`와 리포트 설명에 남긴다. 알리기만 한다. README 4.9절의 작업 순서는 `python -m drowsy.dataset`(또는 E0)으로 시작해 목록을 학습 전에 본다 | 두 모듈 `main()`의 `print_rotation_check` 호출과 `train.load_cv_data`의 `rotation_note`를 지운다 |
| `--raw` 데모의 회전 줄 | 추출 parquet의 회전 기록(가르지 못함, 수동 각도, 기록 없음)을 한 줄로 출력하고 이벤트 JSON의 `source`에 옮겨 적는다. 보정표는 보지 않는다 | `demo.run_demo_raw`에서 `read_raw_rotation`을 쓰는 곳과 `main()`의 출력 한 줄을 지운다 |
| 원본 파일 크기로 이어 하기 판단 | 추출 결과에 원본의 파일 이름과 크기를 적고, 크기가 지금 영상과 다르면 네 입력 모두 다시 추출하며 `replace` 줄로 알린다. 건너뛴 결과의 원본 경로가 다르면 `skip` 줄에 적는다. 크기가 없는 옛 결과는 전처럼 다룬다(가정). 크기가 같은 다른 영상은 가려내지 못한다 | `extract._current_metadata`의 크기 비교를 지운다 |
| 이어 온 처리 시간의 출처 | 한 영상의 값이 시험 추출 리포트와 전체 추출 리포트의 전 판에 다 있으면 두 CSV의 수정 시각으로 나중의 것을 쓰고, 이어 온 값을 출처 리포트별로 적는다. 47열 CSV는 그대로 | `extract.PreviousTimings.lookup`이 전 판을 늘 먼저 쓰게 한다 |
| 스텝 표 캐시의 파일 전체 해시 | `CACHE_FORMAT` 4. 내용 지문이 파일 전체의 sha256이고 한 명령 안에서는 (경로, 크기, 수정 시각 ns)별로 기억한다. 합성 데이터 182개 파일 443 MB에서 첫 호출 224 ms(페이지 캐시를 비운 뒤 406 ms), 그 뒤 5.7 ms. 한 명령이 도는 동안 크기와 수정 시각을 그대로 둔 채 바뀐 파일은 알아채지 못한다. `--rebuild`는 남는다 | `raw_content_digest`를 꼬리말만 읽는 판으로, `CACHE_FORMAT`을 3으로 |
| 늘 지키는 `export/` | `export_onnx.protected_export_dir`이 설정의 `paths.export_dir`과 함께 저장소의 `export/`를 늘 내보내기 폴더로 본다(합성 데이터·축소 설정 모델의 거부). 연기 시험 설정으로 `--out-dir export`를 줘도 막힌다 | `protected_export_dir`에서 `SHIPPED_EXPORT_DIRNAME` 부분을 지운다 |
| 계약 검사의 logits 노드 | logits는 그래프의 유일한 Gemm(`alpha = beta = 1`, `transA = 0`, 상수 가중치와 편향)이거나, INT8 파일이면 양자화한 그 층의 끝인 편향 Add여야 한다. 계약 값은 그대로이고 정상적인 내보내기와 v0은 통과한다. 마지막 층의 가중치를 통째로 줄이거나 키운 파일은 잡지 못하고, 그것은 체크포인트와의 일치 검증이 지킨다(오류 문장과 설명에 적었다) | `_check_graph_constants`의 `_check_logits_producer` 호출을 지운다 |
| 누설 회귀 시험의 셋째 데이터 | 위 "새 모듈" 표의 `tests/test_leakage.py` 행. 시험만 바뀌었다 | 시험 파일에서 `relabelled`와 그 비교를 지운다 |
| 리포트의 D 번호 안내 | 교차검증과 라벨 없는 영상 평가의 summary.md 머리말에 "D 번호는 이 문서 1.2절의 결정 번호"라는 한 줄을 넣고, `속도 억제(D3)`를 `속도 억제(PLAN 1.2절 결정 D3)`로 적는다. 4차 검증 뒤에 두 리포트에 더 넣고 문장을 고쳤다(아래 묶음) | `evaluate.PLAN_DECISION_NOTE`를 넣는 곳을 지운다(지금은 네 곳. 아래 묶음) |
| `.gitignore` | `config/local*`, 어느 깊이의 `videos/`, 영상 확장자 다섯 가지의 모든 대소문자(`*.[mM][pP]4` 등). 일부러 넓게 잡았다(README 4.0절) | `/videos/`, `config/local*.yaml`, 소문자·대문자 두 줄씩으로 좁힌다 |
| 문서 | 감는 속도의 되돌아보기를 쌍의 간격으로만 끊는 것(O1 행)을 SPEC 20장에서도 "정비에서 정한 것"으로 옮겼다. 표 (b) 9번의 다른 안은 저장된 기댓값을 바꾸지 않는다(다른 안을 넣은 참조 구현의 사본에서 89개가 모두 통과). 표 (b) 12번에 `gaze_zone`을 벡터에 담을지를 함께 정할 것으로 더했다 | |

**4차 검증 뒤의 정비(2026-11-14~15)**: 세 검증자가 3차 검증 뒤의 정비를 다시 확인한 뒤(4차 검증) 작은 지적 13개를 마무리한
것이다. 중간에 PC가 다시 시작되어 끊겼다가 이어서 끝냈다. 이식 대상 모듈 일곱, `pipeline_hash`, `rules_hash`,
`export/drowsiness_v0.onnx`, ONNX 계약, 앞의 벡터 89개(합계 27,983,687 bytes)는 그대로다. 표 (b)의 16개 질문과 번호도 그대로다.

| 무엇 | 구현 | 되돌리기 |
|------|------|----------|
| 벡터 13개 추가와 한 줄 바꾸기 점검의 범위 | 한 줄 바꾸기 점검(참조 구현의 한 줄을 일부러 틀리게 바꾼 사본마다 벡터 재검증을 돌려 보는 변이 점검)을 3차 검증 뒤의 세 모듈(`blink`, `rules`, `pipeline`)에서 여섯 모듈(`signals`, `calibrate`, `step_features`까지)로 넓혔고, 앞의 89개가 모두 통과하던 자리를 벡터 13개로 고정했다(기울어진 눈의 홍채 비율, 블렌드셰이프 쪽 closure 분모의 하한, 손으로 놓은 기준선 세 세션, 하품과 끄덕임 판정의 경계, 끄덕임의 느린 평균, 10초 넘는 끊김 뒤의 끄덕임, 한 스텝 안의 감김 둘, 10초 상한을 넘는 감김, 정확히 500 ms인 감김, 얼굴 시간이 정확히 5초일 때의 `pitch_std_60`). 변이 1,420개 가운데 벡터가 1,185개를 잡고(앞의 89개로는 1,127개) pytest만이 116개를 잡고 117개가 남았다(남은 것은 모두 무작위 비교에서 출력 차이 없음). 모듈별 표는 SPEC 16.1절. 벡터로 고정하지 않고 단위 시험만 고정하는 자리(기준선 마감에 걸친 간격을 유지 시간으로 자르는 비교, 실수 임계값의 등호 등)도 SPEC 16.1절에 까닭과 함께 있다. `step_features.py`의 다섯 동작(끄덕임 도중의 얼굴 소실, 느린 평균의 부호, 10초 넘는 끊김 뒤의 끄덕임 초기화, 한 스텝 안 감김 둘의 큰 쪽, 정확히 5,000 ms의 얼굴 시간)에 단위 시험도 더했다 | 시험과 벡터만이다(되돌릴 동작이 없다) |
| 하품·끄덕임 계수기 설정의 `cancel_gap_ms < clock.reset_gap_ms` | 설정을 읽을 때 `pipeline.yawn.cancel_gap_ms`와 `pipeline.nod.cancel_gap_ms`가 `clock.reset_gap_ms`보다 작은지 검사하고, 어기면 두 값을 적은 설정 오류다. 지금 값 500과 10,000은 통과하므로 해시가 그대로다. 어기는 설정에서는 10초 넘는 끊김 앞뒤의 하품·끄덕임을 가르는 것이 계수기의 `reset()` 호출 하나뿐이 된다. `nod.ema_max_dt_ms`에는 같은 조건을 두지 않았다(그 값을 200, 9999, 10000, 20000으로 둔 세션 301개의 실험에서 초기화와 얽히지 않았다. `drowsy/spec.py`의 주석). SPEC 10.1절, 11.5절 | `drowsy/spec.py`의 두 `_check` 호출(지금 763~772줄)과 그것을 부르는 시험(`tests/test_spec.py`)을 지운다 |
| 계약 검사의 속성 읽기 | logits를 내는 Gemm의 `alpha`, `beta`, `transA`와 INT8 Cast의 `to`가 한 번만, 정해진 형으로, 값을 가진 채 있어야 하고 아니면 한 줄의 `ContractError`다. 형이 틀린 속성에서 `TypeError`가 나던 결함을 고쳤고, 값이 없는 INT 속성을 받지 않는 것은 구현자가 정했다(가정). 정상적인 내보내기 71가지와 v0은 통과한다. logits 노드 검사의 한 조건씩을 뺀 10가지 가운데 6가지를 잡는 시험을 `tests/test_export_onnx.py`에 더했고, 남은 넷은 시험이 고정하지 않는다(README 12.3절의 알려진 한계) | `export_onnx._attribute`에서 값이 없는 속성을 거부하는 조건을 빼고 그때는 기본값을 돌려주게 한다 |
| 리포트의 D 번호 안내(넓힘) | 같은 한 줄을 `train final`의 `final/summary.md`와 `baseline_lgbm cv --all-w`의 `lgbm_w_grid/w_grid.md`에도 넣었다(둘 다 일곱째 줄). 문장은 소유자 결정의 수를 세지 않게 고쳤다: "이 리포트의 D 번호(D 뒤에 숫자가 붙은 표기)는 계획서 `docs/PLAN.md` 1.2절의 표에서 승인된 결정의 번호다. 같은 문서 14장에 실린 소유자 결정(2026-11-13, 2026-11-14)은 따로 번호를 매기며, 그 번호와는 다르다." `drowsy/evaluate.py` 머리의 설명도 같이 고쳤다 | `evaluate.PLAN_DECISION_NOTE`를 넣는 네 곳(`evaluate.py`의 두 리포트, `train.py`, `baseline_lgbm.py`)을 지운다 |
| 누설 회귀 시험의 한계 | `tests/test_leakage.py`의 설명에 잡지 못하는 누설 셋을 적었다: 평가받는 피험자의 alert 대 나머지 라벨만 쓰는 누설, 리포트를 쓰는 시점(`drowsy.evaluate`)의 누설, `baseline_lgbm probes`와 `train final`(이 시험이 돌리지 않는다). 넷째 데이터는 만들지 않았다(시험이 이미 1분쯤 걸린다) | 시험의 설명만이다 |
| 회전 점검 기록을 읽는 명령의 설명 | `dataset.read_rotation_checks`의 설명이 그 함수를 쓰는 명령으로 `train`(cv, final)과 `baseline_lgbm`(cv, probes)도 적고, `demo --raw`는 이 함수가 아니라 `demo.read_raw_rotation`으로 그 파일 하나의 기록만 읽는다고 적는다. 설명 문장이라 동작은 그대로이고, 스텝 표 캐시의 코드 지문이 바뀌어 캐시를 한 번 다시 만들었다 | 설명만이다 |
| 문서 | 표 (b) 17번의 "지금 동작"을 지금 코드에 맞췄다(추출 리포트와 여섯 명령이 알린다). README의 학습 명령 회전 줄 설명(데이터를 읽기 전에 알리지만 멈추지 않는다), 얼굴 이미지 시험 수(고르는 열하나 가운데 셋), 이어 온 처리 시간의 한계(시험 추출을 줄여 다시 하면 첫 시험 추출의 값이 전 판의 줄에 섞인다)를 바로잡았다. 이 문서의 본문 두 곳에 구현 메모를 붙였다(머리말) | |

**그 밖에 구현하면서 정한 것**: 5.2절의 "학습 보폭 5초, 시작 위치 무작위 이동"을 영상의 보폭 1 창에서 고르게 뽑는 것으로
구현했다(같은 분포. 표집 순서는 피험자 → 영상 → 출처 → 창). 15 fps 모사 추출이 원본 세션을 다 덮지 못하면 학습을 시작하지
않는다. 같은 리포트 폴더에 다른 조건·축소 설정·데이터 종류의 리포트가 있으면 멈춘다. LightGBM 기준 리포트와 부가 실행의
주 실행이 같은 데이터와 코드에서 나온 것인지(스텝 표 내용 지문, 축소 설정, closure W, 예측 표의 데이터 표시)를 학습 전에
검사한다. 예측이 없는 fold가 있는 묶음은 평가하지 않는다(창이 하나도 없는 fold만 D7대로 모두 오답). 내보내기는 git 상태를
먼저 읽고, 임시 이름으로 쓴 뒤 계약 검사를 통과하면 바꿔 넣는다. `paths.rldd_root`를 주면 합성 표시가 있는 추출 결과를
거부하고, 나뉜 영상에 `_1`만 있으면 멈춘다. `.gitattributes`에서 `*.onnx`, `*.task`는 binary다. 벡터 `rules_calibrating_provisional`은
`calibrating` 종료의 confidence를 고정하려고 입력(40~42초의 프레임 12개를 얼굴 없는 프레임으로)을 바꿨다. 실수로 계산한 값이
임계값에 정확히 놓이는 비교(전방 주시 영역의 다섯 비교, `p_drowsy ≤ 0.4`)는 벡터로 고정하지 않고 SPEC.md의 연산자를 따른다.

### 14.2 표 (b): 소유자의 답을 기다리는 질문

2026-11-15, 4차 검증 뒤의 정비가 끝난 상태의 목록이다(질문과 번호는 3차 검증 뒤와 같고, 9번·12번의 벡터 수와 17번의 "지금 동작"만
고쳤다). README 9장과 같은 16개(2~17번)를 같은 번호로 싣고, 2~12번은
`docs/SPEC.md` 20장의 2~12번이다. 1번(반쯤 뜬 눈이 이어질 때 감김 누적에 끝을 둘까)은 2026-11-14에 소유자가 "지금대로 둠"으로
결정해 비워 두었다. 그 결정은 표 (c)의 5번, README 8장의 결정 5, SPEC 20장 머리의 "2026-11-14에 결정되어 이 목록에서 빠진 것"에
있다. 다른 항목은 번호를 바꾸지 않는다. 항목마다의 영향, 잰 사례, 추천의 근거는 README 9장에 있다. "다시 만들 것"은 답이 지금 동작과 다를 때 바꿔야 하는
것이다. 추천은 정비의 문서 단계가 적은 의견이고 답이 오기 전에는 코드에 들어가지 않는다.

| 질문 | 지금 동작 | 다른 안 | 추천 | 다시 만들 것 |
|------|-----------|---------|------|--------------|
| 2. 짧은 무효 구간 뒤 열린 프레임으로 끝난 깜빡임(검토 frame-04: 감김 검출기의 HOLD 종료) | 잘린 구간이 되어 깜빡임 특징 #4~#8에서 빠진다. 같은 자리의 프레임이 빠졌을 때는 보통 구간 | `max_dt_ms` 안이면 보통 깜빡임 | 다른 안 | `blink.py`, 벡터(runs, steps), 캐시, SPEC 10장 |
| 3. 감는 속도가 유효하려면 진입 쌍이 쓸 수 있어야 하나(검토 frame-01 2단계) | 진입 쌍이 끊김 너머여도 구간 안의 상승으로 감는 속도를 재고 유효로 친다(진입 쌍을 쓸 수 없는 구간 269개 중 33개 유효, 중앙값 2.00/s) | 진입 쌍이 쓸 수 없으면 `velocity_valid = false` | 다른 안 | `blink.py`와 시험, 벡터(runs, steps, `runs_velocity_after_gap`), 스텝 표 캐시, SPEC 10장, 4.5절 문장 |
| 4. 기준선에 두 눈이 다 필요한가(검토 frame-05) | 두 눈 모두 쓸 수 있는 표본이 필요하고 E는 표본 하나로도 정해진다 | 한 눈으로 완성, 눈마다 최소 표본 수 | 지금 동작, E0 결과를 본 뒤 다시 정함 | `calibrate.py`, 새 상수면 `pipeline_hash`·spec·v0 재내보내기·벡터 전부, SPEC 8장 |
| 5. 250 ms 미만 관측을 얼굴 소실 인정만으로 확정할까(검토 rules-06) | 확정한다(영역 밖 프레임 하나 뒤 3.5초 소실이면 `gaze_away_long`, confidence는 0에 가깝다) | 실제 관측 250 ms 요구 | 지금 동작 | `rules.py`, rules 벡터, SPEC 12.4절 |
| 6. 5.2초~10초 끊김도 분석 불가에서 뺄까(결정 4 뒤에 남음) | 그 끊김 뒤 첫 프레임에서 `unable_to_analyze`가 시작되고 경고가 나간다. 각본: 8초 끊김 뒤 눈을 감으면 분석 불가가 +5,000 ms까지 이어지고 그동안 마이크로슬립·수면이 나지 않는다(12초 끊김이면 +1,000 ms, +3,000 ms에 난다) | 더 짧은 끊김에도 재시작하거나 데이터 없는 시간을 얼굴 없는 시간으로 세지 않는다 | 다른 안 | `rules.py`(새 상수면 `rules_hash`와 `export/rules.json`), 벡터 `rules_gap_unable_boundary`·`rules_gap_exactly_10s`, SPEC 12.3·12.7·17장, 8.1절 문장 |
| 7. `unable_to_analyze`와 `vats`의 끝 히스테리시스(검토 rules-09) | 시작과 같은 임계값에서 끝난다 | 끝 임계값을 따로(예: 비율 0.6, 누적 9초) | 다른 안, E0 뒤 상수 확정 때 함께 | `rules.yaml`(새 값이면 `rules_hash`), `export/rules.json`, rules 벡터, SPEC 12장 |
| 8. 결정 2와 3의 귀결을 그대로 둘까 | (a) 재경고 대기 중 시작한 `unable_to_analyze`가 대기가 끝날 때 경고된다. (b) 미뤘다 내는 시선·VATS 경고가 얼굴이 보이지 않는 프레임에서 나갈 수 있다. (c) 미뤘다 내는 경고는 시작 때의 `confidence`, `provisional`을 가진다. (d) 끊겼다 온 프레임 하나가 추론을 최대 10번 한다. (e) 미뤄 둔 경고는 상태가 끝날 때까지 남는다: 10 fps, 10 km/h에서 1.0~12.0초 옆을 보고 12.0초부터 앞을 보면 25.0초에 30 km/h가 될 때 `vats` 경고가 나간다(앞을 본 지 13초 뒤). (f) 정확히 10,000 ms 끊김 뒤 첫 프레임에서 막혀 있던 `gaze_away_long`·`vats`·`drowsy` 경고가 `unable_to_analyze` 경고와 함께 나갈 수 있다 | (e)(f): 전방 주시가 확정되면 미뤄 둔 `vats`·`gaze_away_long` 경고를 지우거나, 조건이 그 프레임에도 성립할 때만 낸다. (a)~(d)에는 제안된 대안이 없다(확인받지 못할 때의 방향으로 적어 둔 것: 종류별로 미루지 않음, 얼굴이 보일 때만 냄, 발행 때 값으로 다시 계산, 추론 횟수 상한) | (a)~(d) 그대로, (e)(f)는 "조건이 그 프레임에도 성립할 때만" | `rules.py` `AlertPolicy` 또는 `pipeline.py`, 벡터 `rules_unable_sparse_frames`·`rules_alert_deferred_cooldown`·`rules_drowsy_model_stall` 등, SPEC 7.2·13장 |
| 9. 얼굴 소실 인정 구간 뒤 `gaze_away_long` 조건에 구현이 더한 것(R1) | 표지가 "알 수 없음"인 프레임에서는 전방 주시 후보가 확정 전이면 내지 않는다. 경계(10 fps): 1,000~3,900 ms 영역 밖, 4,000 ms에 앞을 본 프레임 하나, 다음 얼굴 없는 프레임이 7,200 ms이면 이벤트 없음, 7,201 ms이면 `t_start_ms` 1,000인 시작과 경고 | 인정 구간의 끝이나 그 뒤의 관측 없는 프레임에서 걸린 후보를 버리고 인정된 시간으로 판정한다 | 다른 안 | `rules.py`, SPEC 12.4절의 의사 코드와 설명, 벡터 `rules_gaze_credit_boundary`의 첫 장면 설명 문장(저장된 기댓값은 바뀌지 않는다: 다른 안을 넣은 참조 구현의 사본에서 102개가 모두 통과), 7,200 / 7,201 ms 경계를 고정하는 새 벡터, 경계 시험 |
| 10. 모델과 나가는 spec 파일 이름(검토 evalexport-11) | `feature_spec_v{N}.json` | `feature_spec.json` | 지금 동작 | `export_onnx.py`, `export/`, SPEC 14장, README |
| 11. v0을 첫 커밋 뒤 다시 내보낼까(검토 evalexport-01, docs-10) | `commit = uncommitted-dirty` | 첫 커밋 뒤 다시 내보냄 | 지금 동작(커밋 해시는 v1부터) | `export/drowsiness_v0.onnx`(sha256과 크기), 벡터 `model_v0_inputs`·`e2e_250s`, `export/README.md`, 모델 카드, README |
| 12. 테스트 벡터의 형식과 크기. 함께 정할 것: 표시용 상태 `gaze_zone`을 벡터에 담을지 | JSON 102개, 30,588,827 bytes, `step_probs` 칸 포함(R9). `gaze_zone`을 적을 항목이 없어 단위 시험만 고정한다 | 더 작은 형식이나 LFS. `gaze_zone`은 `expected`에 프레임마다의 항목으로 더한다 | 지금 형식. `gaze_zone`은 앱이 화면에 쓰기로 하면 담는다 | `vectors.py`, 벡터 전부, `tests/vectors/README.md`, SPEC 16장 |
| 13. 재학습 길이와 후보 점수(검토 training-04) | 분할별 조기 종료 단위의 중앙값, 분할별 최고 단위 macro-F1의 평균 | 분할 평균 검증 곡선이 최대인 단위 | 다른 안 | `train.py`, `baseline_lgbm.py`, 시험, 6장 문장, 모델 카드. 실제 결과 전에 정해야 한다 |
| 14. D11의 집계(검토 evalexport-10) | 분류 두 조건은 fold 평균, 오경보 세 조건은 fold를 합친 값 | 모두 합친 값(또는 모두 평균) | 지금 동작 | `evaluate.gate_inputs`, 시험, 모델 카드 9장 |
| 15. 다른 정규화 조건의 신경망 실행을 부가로 둘까 | `train cv --arm`이 `training.export_arm`이 아니면 그 실행의 묶음이 모두 부가(summary.md 9절)이고 D11 판정이 없다 | 그 조건의 리포트에서 주 결과로 | 지금 동작 | `train.run_nested_cv`, 시험, 그 리포트 |
| 16. E3의 "설정은 E2에 고정"을 어떻게 읽을까 | 부가 실행(순서형, 속도 제외)이 바깥 fold마다 안쪽 선택을 다시 한다 | E2에서 고른 후보, 학습 길이, 온도를 그대로 쓴다 | 다른 안 | `train.run_nested_cv`, 시험, 부가 리포트 |
| 17. 회전을 가르지 못한 영상을 어떻게 확인할까(R4 (b)) | 표시만 하고 추출 리포트와 여섯 명령(스텝 표, E0, `baseline_lgbm` cv·probes, `train` cv·final, 라벨 없는 영상 평가, `--raw` 데모)이 알린다(R4 (a), 3차 검증 뒤에 학습 명령과 `--raw` 데모가 더해졌다. README 4.5~4.9, 4.11, 4.12절). 영상을 빼지 않는다. 검증자의 클립 184개: 틀린 클립은 모두 표시됐고 표시 없이 틀린 것 0개, 거꾸로도 검출되는 얼굴은 모든 클립이 표시됨 | (가) 표시된 영상을 모두 사람이 보고 보정표에 적는다, (나) 고른 각도와 반대 각도의 검출 신뢰도를 비교하는 점검을 더한다 | (가), 표시가 너무 많으면 (나) | (가): 설정의 보정표와 다시 추출. (나): `extract.choose_rotation`, `ROTATION_CHECK_VERSION`(전체 다시 추출), 시험 |

정비에서 정한 것(질문이 아니다. README 9장 끝, `docs/SPEC.md` 20장의 같은 문단): (1) 2초~10초 끊긴 뒤의 프레임이 한꺼번에 내는
스텝들은 끊기기 직전의 진행 중 감김을 쓴다(O11, SPEC 11.6절). (2) 감는 속도의 되돌아보기는 쌍의 간격만으로 끊는다: 간격이
`max_dt_ms`를 넘는 쌍에서만 끊고 무효 프레임이 끼어 있어도 간격이 그 이하이면 끊지 않는다(표 (a)의 O1 행). 2차 검증 뒤에는
SPEC 20장 3번에 확인할 것으로 덧붙어 있었는데 3차 검증 뒤에 이 문단으로 옮겼다.

### 14.3 표 (c): 소유자 결정 (2026-11-13, 2026-11-14)

1~4번은 2026-11-13, 5번은 2026-11-14의 결정이다. 다섯 가지 모두 설정 값과 해시(`pipeline_hash` 650495e5…, `rules_hash` 9c2d05f2…, `export/drowsiness_v0.onnx`의 sha256)를
바꾸지 않았다. "다시 만든 것"은 2026-11-14의 상태다.

| 질문 | 전 동작 | 정한 동작 | 다시 만든 것 |
|------|---------|-----------|--------------|
| 1. 반쯤 뜬 구간이 감김 길이에 들어가나(검토 K3) | 길이 = 처음 진입부터 마지막으로 enter 아래로 내려간 시각까지. 이탈~진입에 머문 시간도 들어가서, 프레임 하나로 `ongoing_ms`가 305 ms에서 5,354 ms로 뛰고 마이크로슬립과 수면이 함께 시작될 수 있었다 | "0.7 이상이었던 시간만 더합니다." 길이 = enter 이상인 조각들의 합(`duration_ms`, `ongoing_ms()`). 구간의 시작과 끝, 진폭, 속도, 보간, 반올림은 그대로. 규칙 엔진은 구간 시작을 `run_start_ms()`로 얻는다. 4.5, 8.1, 8.2절의 문장을 고쳤다. 4.6절 #10의 "전체 길이"는 이 `duration_ms`다 | 코드와 단위 시험, 스텝 표 캐시(코드 지문으로 자동), 벡터(`runs_lingering_lids`의 `duration_ms` 1225 → 757, 새 벡터 여섯), SPEC 3.5·10·11.4·12.5절, 합성 데이터 전 과정 |
| 2. 막힌 경고가 다시 나가나(검토 K7) | 막힌 시작은 그 호출에서만 검사했고 되풀이하지 않는 종류는 나중에 경고하지 않았다 | "막힘이 풀렸을 때 상태가 계속 중이면 한 번 경고합니다." 종류마다 대기 중인 시작을 두고 허용되는 첫 호출에서 사본(`duration_ms = t − t_start_ms`)을 한 번 낸다. 끝이 오면 버린다(`microsleep`은 `sleep`의 시작·끝에서도). 주기적 재경고는 없다. 8.2절의 문장을 고쳤다 | 코드와 단위 시험. 기존 벡터는 그대로, 새 벡터 다섯(`rules_alert_deferred_cooldown` 등), SPEC 13장. 데모와 라벨 없는 영상 리포트의 경고 종류(표 (a) O14) |
| 3. 프레임이 1초 넘게 끊기면 모델 판정이 끊기나(검토 K6) | 한 프레임이 낸 여러 스텝 가운데 마지막에서만 모델을 불러 앞 스텝은 확률이 없었고, 연속 수가 끊겼다. 오프라인 평가와 앱이 달랐다 | "밀린 스텝마다 모델을 호출합니다." 창이 찼으면 스텝마다 그 시점의 창으로 부르고 그 확률을 `on_step`에 넘긴다. `PipelineOutput.step_probs` 추가, `probs`는 그 마지막 원소. 4.4절의 문장을 고쳤다 | 코드와 단위 시험, `evaluate.alarm_videos`의 설명과 회귀 시험, 벡터 형식의 `step_probs`(`rules_drowsy_model_stub`, `e2e_250s`에 더함)와 새 벡터 `rules_drowsy_model_stall`, SPEC 3.8·7.2절, 데모의 확률 점 |
| 4. 5초 넘는 끊김 직후의 분석 불가(검토 K4) | 10초 넘는 끊김 뒤에도 분석 불가 창이 그대로라 첫 프레임에서 `unable_to_analyze`가 시작되고 감김·시선 규칙이 멈췄다. 졸음 상태와 판정기도 남았다 | "10초 넘는 끊김 뒤에는 세션 시작 때처럼 창을 새로 시작해 바로 판정을 재개합니다." 첫 프레임에서 `drowsy`(confidence 0.0), `unable_to_analyze`, `gaze_away_long`, `microsleep`/`sleep` 순으로 끝내고 두 판정기와 얼굴 창을 초기화, 처음 10초 보호를 다시 센다. 보정 상태와 기준선은 그대로. 5.2초~10초 끊김은 그대로 분석 불가를 시작한다. 8.1, 8.2절의 문장을 고쳤다. 3.3절의 "검출기와 누적기를 초기화한다"는 이제 그대로 맞다 | 코드와 단위 시험. 기존 벡터는 그대로, 새 벡터 다섯(`rules_gap_over_10s` 등), SPEC 6·12.3·12.6~12.8·17장. 오프라인 오경보 평가의 재시작(표 (a) O12). 남은 질문은 표 (b) 6번 |
| 5. 반쯤 뜬 눈꺼풀이 감김을 끝내지 않는 것을 그대로 둘까(결정 1 뒤에 남은 질문, 표 (b)의 1번이던 것) | 구간은 closure가 이탈 임계값(규칙용 0.5, 특징용 0.3) 아래로 내려가야(또는 끊겨서) 끝나고, 두 임계값 사이에 머무는 동안 여러 깜빡임의 진입 임계값 이상 시간이 한 감김에 쌓인다 | "지금대로 둠"(2026-11-14). 동작을 바꾸지 않았고 알려진 한계로 적었다. 잰 사례(합성, 규칙 엔진): 눈꺼풀이 0.6에 머물고 4초마다 0.9로 165 ms 깜빡이면 마이크로슬립 24.1초(여섯째 깜빡임), 수면 72.0초(열여덟째, 33 ms 프레임 시계. 정확한 30 fps 시각이면 68.1초), 그 뒤 수면 경고가 3초마다 되풀이(3분 각본에서 36번). 결정 1 전에는 둘 다 둘째 깜빡임(8.0초). 고르지 않은 대안: closure가 진입 임계값 아래에 `hold_ms`(500 ms)보다 오래 머물면 구간을 끝낸다. 이식 코드는 이 누적을 그대로 재현해야 한다 | 코드, 벡터, 해시 모두 그대로. 문서만: SPEC 20장의 결정 기록과 10.5절 7번·12.5절·16.1절, `tests/vectors/README.md`, 단위 테스트 이름(`test_decided_case_of_lids_hovering_between_the_two_thresholds`), 이 문서의 8.1절 괄호. 벡터 여섯(`runs_lingering_lids`, `runs_multi_span`, `runs_multi_span_truncated`, `steps_multi_span_runs`, `rules_closure_dwell_below_1s`, `rules_closure_dwell_reaches_1s`)은 이제 결정된 동작을 고정한다 |

### 14.4 소유자 요청으로 더한 실시간 시험 도구 (2026-11-14)

| 무엇 | 내용 | 앱 팀에 가는 것에 대한 영향 |
|------|------|-----------------------------|
| `drowsy/live.py`(명령 `python -m drowsy.live`)와 `tests/test_live.py` | 2026-11-14에 소유자가 Windows에 붙은 웹캠으로 파이프라인을 실시간으로 시험해 보기를 요청했고, 소유자가 승인한 그 계획대로 더했다. Windows의 ffmpeg가 카메라 영상을 MPEG-TS(H.264)로 TCP `localhost`에 보내면 WSL 안에서 받아 프레임마다 MediaPipe와 `DriverPipeline`을 돌리고(데모와 같은 인자, 기준선은 세션의 앞 60초), 상태를 터미널과 127.0.0.1의 웹 페이지에 보여 주고, 끝에 프레임 표와 `--raw` 데모와 같은 그림·이벤트 JSON을 쓴다. 얼굴 화소는 디스크에 쓰지 않는다(D13). 데스크톱 시험 도구이고 **이 계획서가 승인한 산출물(11장)에 들지 않는다**. 실제 카메라, Windows의 ffmpeg, 브라우저, 실제 얼굴로는 실행하지 않았다. 설명은 README 4.13절 | 없다. 이식 대상 모듈 일곱, `config/`, `export/`, `tests/vectors/`, `docs/SPEC.md`, `pyproject.toml`, `uv.lock`은 이 작업 전의 사본과 바이트 단위로 같고(새 의존성 없음), `pipeline_hash`(650495e5…)와 `rules_hash`(9c2d05f2…)도 그대로다. 되돌리기: 두 파일을 지우고 README 4.13절과 이 절을 지운다 |
| 4차 검증 뒤의 고침(같은 두 파일, 2026-11-14) | 그 도구의 최종 점검이 낸 작은 지적 여섯을 고쳤다. `--help`와 설명은 접속이 거부될 때뿐 아니라 보내는 쪽을 시작했는데도 `still waiting` 줄만 나올 때도 `tcp://127.0.0.1:5599`를 써 보라고 적는다(Windows PowerShell의 TCP 접속은 `localhost`와 `127.0.0.1` 모두로 닿았고 웹 페이지도 두 이름에서 응답했지만, Windows의 ffmpeg로 영상을 보내 본 적은 없다). 접속 시간 초과의 오류 문장은 "아무도 접속하지 않음"과 "접속한 쪽이 MPEG-TS 영상을 보내지 않음"을 함께 적고 `-f mpegts`를 알린다. 0초로 적힐 기다림 알림은 내지 않는다. 처리 오류가 난 프레임을 `frames:` 줄과 표의 메타데이터 `n_failed`로 따로 센다(decoded = skipped + dropped + left + failed + processed). 파일도 듣는 소스도 아닌 소스에는 "소스를 멈추라"고 안내한다. 페이지 스크립트의 주석을 한국어로 바꿨다. 기다리는 동안의 SIGTERM·SIGHUP이 빈 출력 폴더를 남기는 것은 처리기를 두지 않고 설명에 적었다. 시험은 69개에서 75개가 됐다 | 없다(이 두 파일만 바뀌었다). 되돌리기: 두 파일을 이 고침 전의 판으로 |
