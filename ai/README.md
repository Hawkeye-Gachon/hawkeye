# hawkeye: 운전자 졸음 판정 모델과 ONNX 내보내기

전면 카메라로 본 운전자의 눈 감김, 시선, 머리 자세에서 졸음과 주의 이탈을 판정한다. 이 저장소에는 네 가지가 있다.

1. Android 앱 팀이 Kotlin으로 옮길 스트리밍 코드의 **Python 참조 구현**(표준 라이브러리만 쓴다)과 규칙 엔진
2. 그 코드를 옮기기 위한 **이식 명세** `docs/SPEC.md`와 **테스트 벡터** `tests/vectors/`(입력과 기댓값을 담은 JSON 102개)
3. 졸음 판정 모델의 **학습·평가 파이프라인**과 **ONNX 내보내기**
4. 실제 데이터를 받았을 때 돌릴 **추출, E0 점검(라벨 없이 신호를 확인하고 잠정 상수를 정하는 단계), 평가 도구**

설계와 근거는 `docs/PLAN.md`(계획서)에 있다. 계획서는 2026-11-13에 1장의 추천안 전부로 승인됐다. 구현하면서 계획과
달라진 점은 계획서 14장에 있다. 이 README에서 "PLAN 4.3절"은 계획서의 절이다. "Q2", "D11"처럼 글자와 숫자로 된 표기는
계획서 1장의 질문 번호(Q)와 승인된 결정 번호(D)이고, 처음 나오는 곳에 뜻을 함께 적었다. 자주 나오는 번호의 뜻은 아래와
같다.

| 번호 | 뜻 |
|------|----|
| Q1, Q2, Q3 | MediaPipe 버전 고정(0.10.32), 앱이 쓰는 `face_landmarker.task`가 공개 표준 파일인지, 앱이 MediaPipe에 넘기는 프레임의 전제(똑바로 세움, 좌우 반전 없음, 얼굴 1개, 증가하는 ms 타임스탬프) |
| Q8 | UTA-RLDD 사본의 위치(받으면 실제 데이터 단계를 시작한다) |
| D1 | 정규화 조건 셋(alert 영상 기준선, 전역 상수, 각 영상 자신의 앞 60초 기준선)을 비교하고, 내보낼 모델은 alert 영상 기준선 조건으로 학습한다 |
| D3 | 경고의 속도 억제: 시선·감김 경고는 20 km/h 미만, 졸음 경고는 50 km/h 미만에서 내지 않는다 |
| D4 | 기준선이 완성되기 전에는 감김 규칙이 임시 판정(`provisional`)으로 돌고 시선 규칙과 모델은 기다린다. 분석 불가 판정(`unable_to_analyze`, 최근 10초의 얼굴 비율)은 기준선과 무관하게 세션 처음 10초 뒤부터 돌고 경고도 낸다(`docs/SPEC.md` 12.3절) |
| D5 | 기준선은 처음 60초로 잡고, 모자라면 30초씩 최대 180초까지 늘리고, 그래도 안 되면 전역 상수로 대체한다 |
| D7 | 영상 단위 정확도의 분모는 논리 영상 180개 전체이고, 품질 기준(얼굴 검출 비율 0.7 이상)을 통과한 영상만의 값은 보조 행이다 |
| D8 | 2클래스(alert 대 drowsy) 영상 정확도는 따로 학습한 모델 없이, 내보내는 3클래스 모델의 `p_alert`와 `p_drowsy` 가운데 큰 쪽을 창마다 고른 뒤 영상별 다수결로 낸다(alert·drowsy 영상 120개) |
| D9 | 오경보는 판정 시작 횟수로 세고, 창 단위 값도 함께 싣는다 |
| D10 | 최종 모델은 60명 전체로 학습하므로 보고 수치는 같은 절차의 교차검증 추정치다 |
| D11 | 교차검증 결과가 정해 둔 다섯 조건을 모두 채울 때만 모델을 졸음 판정의 기본 판정원으로 둔다(지금은 PERCLOS 규칙) |
| D12 | 시선 규칙에 눈동자 방향(`eyeLook` 블렌드셰이프) 보조 판정을 쓴다 |
| D13 | 얼굴 프레임은 어디에도 저장하지 않는다(데모와 리포트는 신호와 타임라인만 그린다) |
| D15 | 범위 밖: 무반응 운전자, 휴대전화 사용, 미러·계기판 확인 |
| D16 | TCN이 최종 모델이면 INT8 양자화 파일도 만들고 지표 변화를 보고한다 |

2026-11-13에 소유자가 따로 정한 네 가지 동작은 번호가 겹치지 않도록 "소유자 결정 1~4"라고 적고, 2026-11-14에 소유자가
답한 반쯤 뜬 눈꺼풀의 동작("지금대로 둠")은 "소유자 결정 5"라고 적는다(8장).
"정비"는 2026-11-13~15에 독립 검토의 지적을 고치고 계획서에서 빠져 있던 항목을 채운 작업이다. 2026-11-14에는 세
검증자가 정비 결과를 세 번 더 확인했다. 두 번째 확인(2차 검증)의 지적을 고친 차례를 "2차 검증 뒤의 정비", 세 번째
확인(3차 검증)의 지적을 고친 차례를 "3차 검증 뒤의 정비", 네 번째 확인(4차 검증. 막는 결함과 큰 결함은 없었고 작은 지적
13개가 나왔다)의 지적을 마무리한 차례(2026-11-14~15)를 "4차 검증 뒤의 정비"라고 적는다. 4차 검증 뒤의 정비는 중간에 이
PC가 다시 시작되어 끊겼다가 이어서 끝냈다. 2026-11-13의 독립 검토가 붙인 지적 번호
(`검토 K5`, `검토 rules-06`처럼 K와 숫자, 또는 영역 이름과 숫자)가 코드 주석이나 계획서에 남아 있는 곳이 있다. 그 검토의
보고서는 저장소에 없으므로, 번호가 나오는 곳에는 지적의 내용을 말로 함께 적었다.

> **보조 안내용이다. 안전 시스템이 아니다.** 여기서 나오는 이벤트와 확률은 운전자에게 주는 참고 신호이고, 차량 제어나
> 사람의 생명에 관련된 결정의 근거로 쓸 수 없다. 조용하다고 해서 운전자가 깨어 있다는 뜻이 아니다.

## 1. 무엇이고 무엇이 아닌가

**있는 것**

- 프레임 하나씩 받아 신호 → 기준선 → closure(눈 감김 정도, 0은 뜬 눈, 1은 감은 눈) → 감김 구간 검출 → 1초 스텝 특징 17개 →
  규칙 이벤트 → 경고를 내는 `DriverPipeline`(`drowsy/pipeline.py`). 학습 특징 생성, E0 점검, 데모, 라벨 없는 영상 평가,
  테스트 벡터가 모두 이 코드 한 벌을 부른다(PLAN 3.1절).
- 규칙 엔진: 보정 중, 분석 불가, 장시간 시선 이탈, VATS(최근 30초 중 시선 이탈 누적 10초), 졸음(PERCLOS 또는 모델),
  마이크로슬립, 수면. 경고 결정(재경고 대기, 속도 억제, 막혔던 경고의 한 번 발행, 우선순위).
- 영상 추출(VIDEO 모드 FaceLandmarker, 모든 입력에 대한 회전 자동 점검, 추출 리포트), 회전을 가르지 못한 영상을 알리는
  점검(스텝 표 명령, E0, 학습과 LightGBM 명령, 라벨 없는 영상 평가, `--raw` 데모), E0 신호 점검과 상수 제안 도구, 15 fps
  깜빡임 길이 오차 모사.
- 모델(인과 TCN, 단방향 GRU), LightGBM 비교 기준선과 closure W 비교, 공식 5-fold 중첩 교차검증, 오경보 평가, 리포트,
  학습 증강 두 가지(약 15 fps 모사 추출, 프레임 누락 변형), 부가 분석(순서형 소프트 라벨, 속도 특징 제외, 15 fps 평가,
  fold별 전역 상수, fold 모델의 INT8 비교), 라벨 없는 차량 영상의 시간당 경고 수.
- ONNX 내보내기와 계약 검사, 테스트 벡터 102개, 합성 세션 생성기, 타임라인 데모, 웹캠 실시간 시험 도구(4.13절).

**없는 것**

- **학습된 모델이 없다.** `export/drowsiness_v0.onnx`는 입출력 계약 확인용이고 가중치가 무작위다.
- **성능 수치가 없다.** 실제 데이터(UTA-RLDD)를 아직 받지 않았다. 합성 데이터로 돌린 점수는 파이프라인이 도는지 보는
  용도이고 성능으로 보고하지 않는다. `docs/MODEL_CARD.md`의 지표 표는 모두 `미측정`이다.
- Android 코드가 없다. 앱 쪽 이식과 실기기 검증은 앱 팀의 일이다(13장).
- 데이터셋과 얼굴 이미지가 없다. 저장소에는 실제 얼굴 이미지나 영상을 넣지 않는다. 테스트와 벡터는 합성 랜드마크만 쓴다.

## 2. 지금 상태 (2026-11-15)

| 단계(PLAN 11장) | 상태 |
|------|------|
| M0 뼈대 | 됨. 고정 버전 설치, MediaPipe 0.10.32가 표준 모델을 읽는다. 일회성 점검 수치는 12.2절 |
| M1 이식 대상 구현과 계약 고정 | 됨. 정비로 바뀐 동작까지 `docs/SPEC.md`와 테스트 벡터 102개에 반영됐다(13.3절). 2차, 3차, 4차 검증 뒤의 정비는 이식 대상 모듈을 한 바이트도 바꾸지 않았고 시험과 벡터만 더했다(3차 검증 뒤에는 감김 검출기의 설정 검증 조건 하나를, 4차 검증 뒤에는 하품·끄덕임 계수기의 조건 둘을 더했다. 7장). 전체 테스트가 통과한다(12.1절) |
| M2 합성 데이터 전 과정 | 됨. 4차 검증 뒤의 정비가 끝난 코드로 2026-11-15(00:12~00:46)에 다시 돌렸고(실행 이름 `synth_m2`, `reports/synth_m2/`, `out/synth_m2/`, 네 명령 모두 종료 코드 0) 내보낸 모델, 체크포인트, 예측 파일 둘, 선택 기록 둘이 그 전의 실행과 바이트 단위로 같았다(12.1절). 이번 정비가 학습 결과를 바꾸지 않았다는 뜻이다. 점수는 보고하지 않는다 |
| M3 실제 데이터 추출 | 도구는 있다(추출, 시험 추출, 추출 리포트). 데이터 경로를 받지 않아 실행하지 않았다. 얼굴이 없는 시험 영상과, 공개 얼굴 사진 두 장으로 만든 짧은 클립(저장소 밖)으로만 돌렸다 |
| E0 잠정 상수 확정 | 도구는 있다(`python -m drowsy.signal_check e0`). 실제 데이터로는 실행하지 않았다 |
| E1~E4 실제 데이터 학습과 평가 | 명령은 모두 있다(LightGBM 기준선과 W 비교, 중첩 교차검증, 부가 분석, 최종 학습과 내보내기). 데이터가 없어 실행하지 않았다 |
| M4 문서 확정과 차량 영상 | 문서는 초판(이 README, `docs/GUIDE.md`, `docs/MODEL_CARD.md`, `docs/SPEC.md`). 차량 영상 명령은 있다(`python -m drowsy.evaluate unlabeled`) |

졸음 이벤트의 기본 판정원은 모델이 아니라 PERCLOS 규칙이다(`config/rules.yaml`의 `drowsiness.source: perclos`, D11).
PERCLOS는 최근 60초 가운데 closure가 0.8 이상이었던 시간의 비율이다.

## 3. 환경 준비

Python 3.11(`>=3.11,<3.12`)과 `uv.lock`에 고정한 버전을 쓴다.

```
uv sync                              # .venv에 고정 버전을 설치한다(pytest 포함)
uv run python -m pytest -q           # 또는 .venv/bin/python -m pytest -q
```

- 아래의 모든 명령은 **저장소 루트에서** 실행하고, `python`은 이 환경의 것이다(`uv run python ...` 또는 `.venv/bin/python ...`).
- 고정 버전: mediapipe 0.10.32, numpy 2.4.6, opencv-python-headless 5.0.0.93, av 18.1.0, torch 2.15.0, onnx 1.23.2,
  onnxruntime 1.31.1, onnxscript 0.7.2, lightgbm 4.7.0, scikit-learn 1.9.1, pandas 3.1.1, pyarrow 26.0.1, matplotlib 3.11.2,
  pyyaml 6.0.3, pytest 9.2.0. 이 PC의 `.venv`는 Python 3.11.16이다. `google.protobuf`는 직접 import하지 않는다.
- mediapipe를 0.10.32로 고정한 이유(Q1): 더 새로운 판은 시스템 라이브러리(`libGLESv2.so.2`)가 필요하거나 실행 시 외부
  조회를 시도한다. mediapipe가 끌어오는 `opencv-contrib-python`은 `pyproject.toml`의 `exclude-dependencies`로 뺀다.
- 이 README를 쓰면서 `uv sync`를 다시 실행하지는 않았다. 아래 명령은 이미 설치된 `.venv/bin/python`으로 확인했다.

### face_landmarker.task

영상에서 랜드마크를 뽑는 단계(추출, `--video` 데모)에만 필요하다.

- 놓는 곳: `models/face_landmarker.task` (`config/default.yaml`의 `paths.model_path`). `models/`는 git에 넣지 않는다.
- 파일: MediaPipe의 공개 표준 파일(float16, 버전 1), 3,758,596 bytes.
  `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`
- 기대하는 sha256: `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`
  (`sha256sum models/face_landmarker.task`로 확인한다)
- 추출과 데모는 시작할 때 파일의 sha256을 `config/feature_spec.json`의 `pipeline.mediapipe.model_sha256`과 비교하고,
  다르면 멈춘다. 앱이 실제로 다른 파일을 쓴다면(Q2) 그 값을 바꾸고 `python -m drowsy.spec update-hash`를 실행한다.
  그러면 `pipeline_hash`(이식 코드의 상수 전부에 대한 해시)가 바뀌므로 모델을 다시 내보내고 테스트 벡터를 다시 만들어야 한다.
- 이 파일이 없어도 테스트(`mediapipe` 표시가 붙은 테스트는 건너뛴다), 합성 데이터 전 과정, `--raw` 데모는 돈다.

## 4. 단계별 명령

각 명령의 옵션은 `python -m drowsy.<모듈> --help`(하위 명령이 있으면 `python -m drowsy.<모듈> <하위 명령> --help`)로
볼 수 있다. 여기 적은 옵션은 2026-11-15, 4차 검증 뒤의 정비가 끝난 코드에서 모든 모듈과 하위 명령 24개(실시간 시험 도구
`python -m drowsy.live` 포함)의 `--help`를 하나씩 돌려 확인했다(24개 모두 종료 코드 0). 4차 검증 뒤의 정비는 옵션을 하나도
바꾸지 않았고, `python -m drowsy.live --help` 끝의 안내 2번에 `127.0.0.1` 주소를 써 볼 경우 하나를 더했을 뿐이다(4.13절).
기본 설정은 `config/default.yaml`이고 `--config`의 기본값도 그 파일이다.

**실행 이름(`--run-name`)의 규칙.** 리포트를 쓰는 명령(`extract`, `signal_check e0`과 `blink-sim`, `baseline_lgbm`,
`train`, `evaluate unlabeled`)의 `--run-name`은 `reports/` 아래의 폴더 이름 하나여야 한다. 빈 값, `.`, `..`, 그리고 `/`나
`\`가 든 이름은 설정 파일을 읽기 전에 거부하고, 종료 코드 2와 한 줄
`error: --run-name must be a plain folder name, got '<값>'`을 낸다. 리포트 폴더 밖(예: 추적되는 `export/`)에 설정 사본을
쓰는 일을 막기 위해서다. `python -m drowsy.dataset`과 `python -m drowsy.demo`에는 실행 이름이 없다(데모의 `--out`은
출력 폴더 경로다).

### 4.0 실제 데이터를 쓸 때의 설정 파일 (`config/local.yaml`)

추적되는 `config/default.yaml`은 고치지 않는다. 그 파일의 `paths.rldd_root`는 `null`로 남아야 하고, 테스트 하나
(`tests/test_spec.py::test_default_config_paths`)가 그것을 확인한다. 실제 데이터의 경로는 git에 넣지 않는 사본에 적는다.

```
cp config/default.yaml config/local.yaml          # config/local*는 .gitignore에 있다
# config/local.yaml에서 paths.rldd_root(UTA-RLDD 사본), 필요하면 paths.unlabeled_root(팀 수집 차량 영상)를 적는다
python -m drowsy.extract --config config/local.yaml --dataset --pilot 6
```

- 4.5~4.11의 명령에는 모두 `--config config/local.yaml`을 준다. 아래 예시는 실제 데이터 단계에서는 `config/local.yaml`,
  합성 데이터 단계에서는 `config/default.yaml`로 적었다. `--data-root DIR`은 합성 데이터 폴더를 바꿀 뿐이고 합성 데이터를
  실제 데이터로 바꾸지 못한다.
- `config/default.yaml`에 키가 더해지면 `config/local.yaml`에도 같은 키를 옮겨 적는다. 없는 키가 있으면
  `load_config`가 `missing keys`로 멈춘다(정비에서는 `extraction.rotation_roll_margin_deg`,
  `extraction.unlabeled_rotation_overrides`, `training.ordinal_neighbor_mass`, `training.frame_drop_share`,
  `training.frame_drop_variants`가 더해졌다).
- 리포트는 설정 파일에 적힌 데이터 경로를 그대로 싣는다. `reports/`는 git에 넣지 않지만, 실제 데이터 리포트를 밖으로
  꺼낼 때는 경로를 확인한다.
- `paths.rldd_root`를 주면 저장된 추출 결과(parquet)에 합성 데이터 표시가 있는지 검사하고, 있으면 멈춘다.
- **영상과 사본 설정은 git에 넣지 않는다.** `.gitignore`의 해당 패턴은 일부러 넓게 잡았다. 실제 데이터의 경로나 얼굴
  영상이 실수로 커밋되는 것을 막는 쪽을 골랐기 때문이다.
  - `config/local*`: `config/` 안에서 이름이 `local`로 시작하는 파일은 모두 제외된다. `config/local.yaml`뿐 아니라 그
    백업(`config/local.yaml.bak`)이나 `config/locale.yaml`, `config/localhost.txt` 같은 이름도 제외되므로(2026-11-14에
    `git check-ignore`로 확인), 추적할 설정 파일에는 `local`로 시작하는 이름을 쓰지 않는다.
  - `videos/`: 이름이 `videos`인 폴더는 저장소의 어느 깊이에 있든(`tests/videos/`, `docs/videos/`도) 통째로 제외된다.
    추적할 파일을 그런 이름의 폴더에 두지 않는다.
  - 영상 확장자 다섯 가지(`mp4`, `mov`, `m4v`, `avi`, `mkv`)는 글자의 대소문자가 섞여도(`clip.MP4`, `clip.Mp4`) 어느
    폴더에서나 제외된다(`*.[mM][pP]4`처럼 글자마다 두 경우를 적은 패턴이다).
  아래 예시의 `videos/`와 `path.mp4`는 그래서 저장소 루트에 두어도 추적되지 않는다. 추적되지 않는 파일이 있으면 내보낸
  모델의 `commit` 메타데이터에 `-dirty`가 붙으므로, 다른 파일도 저장소 밖이나 제외된 폴더(`data/`, `out/`)에 둔다.
- **데이터셋은 이 저장소의 명령으로 내려받지 않는다.**

### 4.1 설정 검사

```
python -m drowsy.spec check                      # config/feature_spec.json의 pipeline_hash가 pipeline 블록과 맞는지 (--spec 파일)
python -m drowsy.spec update-hash                # pipeline 블록을 고친 뒤 pipeline_hash를 다시 계산해 파일에 쓴다
python -m drowsy.spec export-rules --out export/rules.json   # 규칙 설정을 앱용 JSON으로 쓰고 rules_hash를 출력한다
```

`drowsy.spec`과 `drowsy.vectors`는 `--config` 대신 `--spec`(feature_spec 파일), `--rules`(규칙 설정)를 받는다.

### 4.2 합성 데이터 (실제 데이터가 없을 때)

```
python -m drowsy.synth --config config/default.yaml --out data/synth --subjects 60 --minutes 6
python -m drowsy.synth --config config/default.yaml --out data/synth_fps15 --subjects 60 --minutes 6 --fps-sim 15
```

- 추출 결과와 같은 형식의 parquet을 RLDD와 같은 배치(`Fold{k}_part{p}/{NN}/{0|5|10}.parquet`)로 쓴다. 얼굴 이미지는 없다.
- 두 번째 명령은 같은 세션에서 기기의 흐름 제한기(MediaPipe 앞단에서 처리가 밀리면 새 프레임을 버리는 장치)가 버렸을
  프레임을 뺀 것이다(약 15 fps). 이 폴더가 있으면 신경망 학습이 표본의 일부(`training.fps_sim_share`)를 여기서 뽑는다.
- 그 밖의 옵션: `--seed`(기본 0), `--workers`, `--minutes`(기본 10). 출력 폴더에 parquet이 이미 있으면 거부한다.
- `paths.rldd_root`가 `null`이면 4.6~4.10의 명령은 `paths.synth_dir`(`data/synth`)을 쓴다.
- **합성 데이터의 점수는 성능이 아니다.** 합성 데이터로 만든 리포트와 출력은 첫 줄에 그 사실을 적는다.

### 4.3 테스트

```
python -m pytest -q                              # 전체(이 PC에서 가장 낮은 우선순위 nice -n 19로 약 6분 30초)
python -m pytest -q -m "not mediapipe"           # MediaPipe가 필요한 테스트를 빼고
python -m pytest -q tests/test_leakage.py        # 시험 fold 누설 회귀 시험만(이 PC에서 혼자 돌려 54초)
DROWSY_TEST_FACE_IMAGE=/path/to/face.png python -m pytest tests/test_extract.py tests/test_demo.py -k face
```

- 마지막 명령은 이름에 `face`가 든 시험 11개를 고른다. 그 가운데 정면 얼굴 이미지 한 장이 있어야 도는 것은 셋이다: 실제
  FaceLandmarker의 출력 형식·좌우 규약·회전 규칙(`test_real_landmarker_on_a_face_image`), 그리고 옆으로 누운 얼굴 클립을
  `extract --input-dir`·`extract --video`와 `demo --video`가 똑바로 세우는지(`test_real_cli_turns_a_sideways_face_clip_upright`,
  `test_cli_video_turns_a_sideways_face_clip_upright`). 나머지 여덟은 얼굴이 없는 입력으로 도는 보통의 시험이라 변수가
  없어도 돈다. 세 시험은 짧은 얼굴 클립을 pytest의 임시 폴더에만 쓴다. 저장소에는 얼굴 이미지를 넣지 않으므로 변수가
  없으면 셋을 건너뛴다(전체 테스트의 `3 skipped`가 이것이다).
- `tests/test_leakage.py`는 시험 fold에 든 정보가 학습이나 평가 쪽으로 새지 않는지 결과만 보고 확인하는 시험이다(구현을
  들여다보지 않는다). 작은 합성 데이터에서 피험자 넷(1, 13, 25, 26)을 쓰고 데이터를 세 벌 만든다. 시험 fold는 3이고 25와
  26이 그 fold의 시험 피험자다.
  - 원본(`clean`).
  - 망가뜨린 사본(`corrupted`): 피험자 25의 데이터만 다른 얼굴, 뒤섞은 클래스, 20초 더 긴 세션으로 바꾼다.
  - 라벨만 바꾼 사본(`relabelled`, 3차 검증 뒤에 더했다): 피험자 26의 low vigilant 영상과 drowsy 영상의 파일 이름을
    맞바꾼다. 프레임은 한 장도 다르지 않고 바뀌는 것은 26의 두 영상에 붙는 라벨뿐이다(alert 영상은 기준선을 잡는 데 정당하게
    쓰이므로 맞바꾸지 않는다).
  세 데이터로 명령 여덟 개(`train cv`, `baseline_lgbm cv`, `cv --arms global --foldwise-global`, `train cv --ordinal`,
  `train cv --no-velocity`, `baseline_lgbm cv --no-velocity`, `cv --all-w`, `train cv --arm global`)를 모두 돌린 뒤 본다.
  원본과 망가뜨린 사본 사이에서는 fold 3의 선택 기록(`selection.json`)이 같아야 하고(다른 fold는 달라야 한다), 손대지 않은
  피험자 26의 예측 행이 모든 예측 묶음(44개)에서 같아야 한다(확률은 1e-6 안, 나머지 열은 정확히). 원본과 라벨만 바꾼 사본
  사이에서는 26의 확률 행이 내용이 같은 행끼리 견주어 모든 묶음에서 같아야 한다. 평가받는 피험자 자신의 시험 라벨이 그
  피험자의 예측에 닿는 누설(자기 라벨로 온도를 다시 맞추기, 예측에 정답 라벨을 섞기)은 앞의 비교로는 보이지 않아서 더한
  비교다. 시험은 37개이고 2026-11-14에 다른 작업이 없는 이 PC에서 혼자 돌려 54.4초 걸렸다.
- 이 시험이 잡지 못하는 누설(알려진 한계. 4차 검증 뒤의 정비가 시험 파일의 설명에 적었다. 넷째 데이터는 만들지 않았다):
  - 평가받는 피험자의 "alert인지 아닌지" 라벨만 쓰는 누설. 26의 alert 영상은 세 데이터에서 모두 같으므로, 예측에 alert 여부만
    섞거나 시험 라벨을 alert 대 나머지로 줄여 온도를 다시 맞추는 누설은 26의 확률 행을 바꾸지 않는다.
  - 리포트를 쓰는 시점의 누설. 비교하는 것은 선택 기록(`selection.json`)과 예측 표(`predictions.parquet`)뿐이라,
    `drowsy.evaluate`가 예측 표에서 계산하는 것(창의 투표, 영상 판정, 경보 판정, 지표)에 시험 라벨을 쓰는 누설은 드러나지 않는다.
  - `baseline_lgbm probes`(교란 점검)와 `train final`(최종 모델)은 이 시험이 돌리지 않는다.
- 정비 작업자들이 저장소 사본에 일부러 누설을 넣어 이 시험을 확인했다(작업자의 기록. 문서 작성자가 다시 돌리지 않았다).
  2차 검증 뒤에는 누설 10가지 가운데 8가지를 잡았다. 잡지 못한 둘(15 fps 모사 추출과 프레임 누락 변형을 학습 피험자로
  거르지 않음)은 출력이 한 바이트도 바뀌지 않아서 결과만 보는 시험으로는 볼 것이 없고, `tests/test_train.py`의 단위 시험이
  따로 막는다. 3차 검증 뒤에는 검증자가 찾은 자기 라벨 누설 넷(피험자마다 자기 시험 라벨로 온도를 다시 맞추기, 예측 함수와
  신경망·LightGBM의 예측 표에서 확률에 정답 라벨을 섞기)과 평가 때의 누설 셋(시험 fold로 다시 표준화하기, 시험 fold로 온도
  맞추기, 시험 fold 전체의 평균 예측으로 LightGBM 확률을 다시 맞추기)을 하나씩 넣은 사본 일곱이 모두 이 파일에서 실패했고,
  고치지 않은 코드는 두 번 다 통과했다.
- 다른 무거운 작업과 동시에 돌릴 때는 `OMP_NUM_THREADS=2`처럼 스레드 수를 줄인다. `tests/test_train.py`의 torch 테스트가
  과부하에서 멈춘 듯 느려진 적이 있다.
- 지금의 결과는 12.1절에 있다.

### 4.4 테스트 벡터

```
python -m drowsy.vectors verify --dir tests/vectors       # expected를 다시 계산해 비교 (--model-dir export, --spec, --rules)
python -m drowsy.vectors generate --out tests/vectors     # 다시 만들기 (--model export/drowsiness_v0.onnx, --spec, --rules), 약 4초
```

`config/feature_spec.json`의 `pipeline` 블록, `config/rules.yaml`, 이식 대상 모듈, `export/drowsiness_v0.onnx` 가운데 하나라도
바뀌면 다시 만들고 차이를 검토한다. 지금 저장된 벡터는 102개(30,588,827 bytes)이고 지금 코드와 맞는다(`verify`가 102개
모두 통과하고, 다시 만들면 바이트 단위로 같다). 2차 검증 뒤의 정비가 13개를, 3차 검증 뒤의 정비가 8개를, 4차 검증 뒤의
정비가 13개(2,605,140 bytes)를 더했다. 그때마다 앞의 벡터(68개, 81개, 89개)는 바이트 단위로 그대로다(4차 검증 전의 89개는
합계 27,983,687 bytes로 그때와 같고, 다시 만들어도 같다). 4차 검증 뒤의 13개는 "한 줄 바꾸기 점검"(참조 구현의 한 줄을 일부러
틀리게 바꾼 사본 하나하나를 벡터 재검증에 돌려 보는 변이 점검)에서 앞의 89개가 모두 통과하던 `signals.py`, `calibrate.py`,
`step_features.py`의 자리를 고정한다: 기울어진 눈의 홍채 비율, 블렌드셰이프 쪽 closure의 분모(`b_closed − B`)의 하한, 기준선의 중앙값(표본 수가 홀수일 때),
프레임이 몰려 올 때의 얼굴 시간, 얼굴 프레임이 정확히 300개인 기준선, 하품과 끄덕임 판정의 경계, 끄덕임의 느린 평균,
10초 넘는 끊김 뒤의 끄덕임, 한 스텝 안의 감김 둘, 10초 상한을 넘는 감김, 정확히 500 ms인 감김, 얼굴 시간이 정확히 5초일 때의
`pitch_std_60`. 3차 검증 뒤의 8개는 참조 구현의 한 줄을 일부러 틀리게 바꿔도 앞의 벡터 81개가 모두
통과하던 자리(30초 창의 경계, 정점 전의 증가율, 얼굴을 잠깐 놓친 `vats`·`microsleep`의 `confidence`, 한 프레임 곁눈질, 다른
종류의 시작 뒤에도 남는 미뤄 둔 경고, 첫 `t_ms`가 작은 세션의 처음 10초, 스텝 끝과 프레임 시각이 다른 `drowsy`)를 고정한다.
형식과 벡터별 설명은 `tests/vectors/README.md`에, 정비에서 달라지거나 더해진 벡터의 목록과 벡터가 고정하지 않는 동작의
목록은 `docs/SPEC.md` 16.1절에 있다.

### 4.5 실제 데이터 추출 (M3)

```
python -m drowsy.extract --config config/local.yaml --dataset --pilot 6 --run-name <이름>        # 시험 추출
python -m drowsy.extract --config config/local.yaml --dataset --run-name <이름>                  # 전체 추출 → data/raw/native/
python -m drowsy.extract --config config/local.yaml --dataset --fps-sim 15 --run-name <이름>      # 기기 프레임 누락 모사 → data/raw/fps15/
python -m drowsy.extract --config config/local.yaml --unlabeled [--fps-sim 15] --run-name <이름>  # 팀 수집 차량 영상 → data/raw/unlabeled_native/
python -m drowsy.extract --config config/default.yaml --input-dir videos/ [--run-name <이름>]     # 폴더 하나
python -m drowsy.extract --config config/default.yaml --video path.mp4 [--rotation 90]            # 영상 하나(리포트 없음)
```

- 옵션: `--workers N`(기본 `extraction.workers`), `--pilot [N]`(기본 `extraction.pilot_videos` = 6, `--dataset`에만),
  `--rotation {0,90,180,270}`(`--video`에만, 아래 회전 항목), `--run-name`(기본 `extraction`, 4장 처음의 규칙).
  `--video`, `--input-dir`, `--dataset`, `--unlabeled` 중 하나만 준다.
- `videos/`와 `path.mp4`는 예시 경로다. 저장소 루트에 두어도 `.gitignore`가 영상을 제외한다(4.0절).
- VIDEO 모드 FaceLandmarker로 영상마다 RawFrame parquet 하나를 쓴다(랜드마크 44개, 블렌드셰이프 52개, 변환 행렬).
  얼굴 이미지는 저장하지 않는다. 밝기(프레임 휘도 평균과 중앙값), 원본의 길이, 평균 fps, 최소·최대 프레임 간격, 가변 간격
  여부, 해상도는 parquet 메타데이터에 적는다(열은 그대로, `SCHEMA_VERSION` 1). 원본 영상은 메타데이터의 세 키로 적는다:
  `source`(받은 경로 그대로), `source_name`(파일 이름), `source_size_bytes`(디코딩을 시작하기 전에 읽은 파일 크기, 바이트).
  뒤의 둘은 3차 검증 뒤의 정비가 더했다.
- **이어 하기**: 이미 같은 설정으로 만든 결과가 있는 영상은 건너뛰고 `skip <영상>: <결과 파일> is up to date` 줄을 낸다.
  네 입력 모두 결과에 적힌 원본 파일 크기를 지금 받은 영상의 크기와 비교한다. 크기가 다르면 이름만 같은 다른 영상이거나
  바뀐 파일로 보고 다시 추출하며, 먼저 `replace <영상>: <결과 파일> was extracted from <옛 경로> (<옛 크기> bytes); this video
  has <지금 크기> bytes, so it is extracted again and that result is overwritten` 줄을 낸다(결과 파일은 하나뿐이라 옛 영상의
  결과는 없어진다). 원본 경로는 비교하지 않는다(데이터 폴더를 옮겨도 다시 추출하지 않는다). 대신 건너뛴 결과에 적힌 경로가
  지금의 영상과 다르면 `skip` 줄 끝에 `(extracted from <옛 경로>, a file of the same size as this video: N bytes)`가 붙는다.
  한계: 크기가 바이트 단위로 같은 다른 영상(예: 같은 영상에 회전 태그만 바꿔 단 파일)은 가려내지 못한다(내용의 해시는 이어
  하기마다 모든 영상을 끝까지 읽어야 해서 구하지 않는다). 그런 영상으로 바꿨다면 결과 파일을 지우고 다시 실행한다. 크기가
  적혀 있지 않은 결과(3차 검증 뒤의 정비 전에 만든 추출)는 전처럼 크기를 비교하지 않는다(그때 `skip` 줄은 `the result
  records no source size, so the two files were not compared`라고 적는다). 스텝 표와 학습 명령은 두 키가 있든 없든 같은
  파일을 읽는다.
- **추출 리포트**: `--video`를 뺀 모든 실행이 `reports/<이름>/extraction/`에 `<태그>_videos.csv`(영상마다 한 행, 47열:
  상태, 실패 사유, 경고, 프레임 수, 길이, fps, 해상도, 고른 회전과 이유, 각도별 얼굴 검출률과 roll, 얼굴 비율, 프레임당 ms,
  밝기 등), `<태그>_summary.md`, `<태그>_config.yaml`을 쓴다. 태그는 `<입력 종류>_<추출 폴더 이름>`이다. 입력 종류는
  `dataset`, `pilot`(시험 추출), `input_dir`, `unlabeled`이고, 추출 폴더는 원본 프레임률이면 `native`, 누락 모사이면
  `fps15`(`--fps-sim`에 다른 값 N을 주면 `fps{N}`)다. 그래서 `dataset_native`, `dataset_fps15`, `pilot_native`,
  `pilot_fps15`, `input_dir_native`, `input_dir_fps15`, `unlabeled_native`, `unlabeled_fps15` 같은 이름이 나온다. 실패한
  영상이 있을 때, 모두 이미 추출됐을 때, 도중에 멈췄을 때도 쓴다. 파일 수는 RLDD의 기대값(182개, 논리 영상 180개)과 비교해
  적되 강제하지 않는다.
- **회전**: 네 입력(`--video`, `--input-dir`, `--dataset`, `--unlabeled`) 모두 영상마다 회전을 자동으로 점검한다. 휴대전화
  녹화에는 회전 정보(컨테이너의 회전 태그)가 없거나 틀린 것이 있고, 옆으로 누운 얼굴도 검출률 1.0으로 검출되기 때문이다.
  점검은 영상 앞 24초에서 12장을 뽑아 0·90·180·270도의 얼굴 검출률과 머리 roll(좌우로 기운 각도)의 중앙값을 잰다. 검출률이
  가장 높은 각도(표본 하나 차이까지 같은 것으로 본다) 가운데 |roll|이 가장 작은 각도를 고른다. roll 차이가
  `extraction.rotation_roll_margin_deg`(15도) 안이라 가르지 못하면, 또는 어느 각도에서도 얼굴을 못 찾으면 "가르지 못함"
  (`rotation_ambiguous`)으로 표시한다. 0도와 180도처럼 정반대인 두 각도는 랜드마커가 거꾸로 선 얼굴도 검출하면 roll이 거의
  같게 나와 이 점검으로 가를 수 없다. 그때는 회전 정보의 각도를 그대로 두고 표시만 한다.
- 수동 회전이 자동 점검보다 우선한다: `--dataset`, `--input-dir`은 `extraction.rotation_overrides`, `--unlabeled`는
  `extraction.unlabeled_rotation_overrides`(키는 입력 폴더 기준 상대 경로), `--video`는 `--rotation`(보정표의 같은 영상보다도
  우선한다. 보정표의 키는 파일 이름)을 쓴다. 각도는 그림을 똑바로 세우려면 반시계로 돌려야 하는 각도다. `--rotation`은 그
  실행에만 적용되고, `--video` 아닌 입력에 주면 종료 코드 2로 거부한다.
- 영상마다 출력하는 `done` 줄 끝에 각도를 어떻게 정했는지 적는다: `rotation: auto check chose N (metadata M; <사유>; 각도별
  검출률과 roll)`, 가르지 못했으면 `WARNING rotation ambiguous: look at the video and add it to the manual rotation table
  (<설정 키>)`(`--video`이면 `pass --rotation`), 수동 각도이면 `rotation: manual override`(`--rotation`이면
  `rotation: manual override (--rotation)`). 리포트(`--video` 제외)는 가르지 못한 영상을 summary.md의 "회전" 절 앞쪽에
  사유별로 모은다.
- 점검 기록은 추출 결과 parquet의 메타데이터 `rotation_check`에 남는다(자동 점검으로 각도를 정한 파일에만 있고, 수동 각도로
  만든 파일에는 없다). 2차 검증 뒤의 정비 전에 `--input-dir`이나 `--video`로 만든 결과에는 이 기록이 없으므로, 다음 실행에서
  다시 추출된다. 이 기록은 4.6~4.11의 명령(E0, 스텝 표, LightGBM, 학습, 라벨 없는 영상 평가)과 4.12의 `--raw` 데모가 읽는다.
- 측정(2차 검증의 검증자가 공개 얼굴 사진 두 장을 돌리거나 기울여 만든 클립 184개로 잰 것, 실제 휴대전화 녹화나 RLDD가
  아니다): 회전 정보가 없거나 틀린 클립 40개 중 35개가 맞았고, 틀린 5개는 모두 정확히 180도 차이였고 모두 표시됐다.
  기울인 클립 144개 중 41개가 틀렸고 모두 표시됐다. 표시 없이 틀린 클립은 184개 중 0개였다. 거꾸로 서도 검출되는 얼굴은
  맞든 틀리든 모든 클립이 표시됐다. 그래서 표시된 영상은 사람이 직접 보고 보정표에 각도를 적는다. 표시된 영상을 모두 볼지,
  더 엄격한 비교를 더할지는 9장 17번의 질문이다.
- 보정표의 키가 이번 입력의 어느 영상과도 맞지 않으면 시작하지 않고(종료 코드 2) 가장 가까운 경로를 알린다. 그래서
  데이터셋 영상의 보정표를 둔 설정으로 `--video`나 `--input-dir`을 돌리면 멈춘다.
- 표시 행렬(영상 파일에 든 화면 회전·반전 정보)에 좌우·상하 뒤집기가 든 영상은 수동 회전 없이는 추출하지 않는다. 메시지가
  `--rotation`(`--video`)이나 보정표(다른 입력)를 알려 준다. `--video`에서 그런 영상은 `FAILED` 줄과 종료 코드 1로 끝난다.
- **처리 시간**(summary.md 6절): 처리 시간은 결과 파일에 없으므로 건너뛴 영상의 값은 전에 쓴 리포트에서 이어 온다. 이번
  실행에서 잰 값과 이어 온 값을 따로 적고, 이어 온 값은 그 값이 적혀 있던 리포트별로 한 줄씩 적어 어디서 잰 값인지 밝힌다:
  시험 추출 리포트(`pilot_<폴더>_videos.csv`. 시험 추출은 영상을 한 번에 하나씩 처리한다), 또는 이 리포트의 전 판
  (`<태그>_videos.csv`. 그 실행이 영상을 몇 개씩 동시에 처리했는지는 남아 있지 않다). 전체 추출(`--dataset`)의 리포트에서
  한 영상의 값이 두 리포트에 다 있으면, 이번 실행이 리포트를 다시 쓰기 전에 읽은 두 CSV 파일의 수정 시각을 견주어 나중에
  쓴 리포트의 값을 쓴다(3차 검증 뒤의 정비. 전에는 전 판의 값이 늘 이겨서, 전체 추출 뒤에 시험 추출을 다시 해도 옛 값이
  나왔다). 두 값이 글자까지 같으면 전 판이 시험 추출의 값을 옮겨 적은 것이므로 시험 추출에서 잰 값으로 적는다. 이번 실행이
  추출했는데 처리 시간을 받지 못한 영상(실행이 끊김)에는 옛 값을 잇지 않는다. "영상을 한 번에 하나씩 / 최대 N개씩 동시에
  처리했다"는 문장은 이번 실행에서 잰 줄에만 붙는다. CSV의 47열 형식은 그대로다. 시험 추출의 값은 전체 추출 시간을
  어림하는 데 쓴다.
- 영상 목록(`dataset.build_manifest`): 나뉜 영상(`10_1`, `10_2`)에 `_1`만 있으면 사본이 덜 된 것으로 보고 멈춘다.
- 종료 코드: 0 성공, 1 실패한 영상이 있음(리포트는 쓴다), 2 시작할 수 없음.

### 4.6 E0: 라벨 없는 신호 점검과 상수 확정

```
python -m drowsy.signal_check e0 --config config/local.yaml --run-name <이름>
python -m drowsy.signal_check e0 --config config/local.yaml --run-name <이름> --override closure.rho=0.30 closure.b_closed=0.65
python -m drowsy.signal_check e0 --config config/local.yaml --run-name <이름> --fps-sim
python -m drowsy.signal_check blink-sim --config config/default.yaml --run-name <이름> [--seed 0]
```

- `e0`은 세션마다 `DriverPipeline`을 자기 기준선으로 돌려 PLAN 11장 E0의 세 기준(기준선 60초 완성 비율, 분당 깜빡임 5~40회
  비율, 깜빡임 정점 closure의 중앙값)을 `evaluation.signal_gate`로 판정하고, 전역 상수(E, B, jaw0: 기준선을 잡지 못한 영상에
  쓰는 뜬 눈 EAR, 뜬 눈 `eyeBlink`, 입 벌림 기준값)를 세션별 값의 중앙값으로 제안한다. 라벨, 클래스, fold는 계산 전에
  버린다. 결과: `reports/<이름>/e0/`의 summary.md, per_video.csv, criteria.json, config.yaml, 제안 spec
  (`feature_spec_proposed.json`, 합성 데이터면 `feature_spec_proposed.synthetic.json`). `config/feature_spec.json`은 절대
  고치지 않는다. 제안을 받아들이는 순서는 summary.md에 번호로 적힌다. 옵션 `--workers N`.
- `--override`는 `pipeline` 블록의 상수만 바꾼다(`config/rules.yaml`의 감김 임계값 0.7/0.5는 바꾸지 못한다). 잠정이 아닌
  상수를 바꾸면 리포트에 표시한다. `--fps-sim`은 `reports/<이름>/e0_fps_sim/`에 쓰고 제안 파일을 쓰지 않는다.
- **회전 점검 줄**: `E0 criteria ...` 줄 다음에 한 줄을 출력한다(4.7절의 줄과 같은 형식). 리포트에는 summary.md의 마지막 절
  "회전 점검"(분류별 파일 수와 표시된 영상 목록), criteria.json의 `rotation_check` 키가 있고, 표시된 영상이 있으면
  summary.md 앞쪽에 굵은 경고 문단이 붙는다. 표시된 영상은 빼지 않고 종료 코드도 바뀌지 않는다(알리기만 한다).
- 종료 코드: 0, 2 시작할 수 없음, 4 실제 데이터의 원본 프레임률에서 기준 미달(리포트는 쓴다).
- `blink-sim`은 가정한 깜빡임 파형으로 30 fps, 15 fps, 15 fps + 20% 누락에서 감김 길이 오차와 검출률을 보간 유무로 비교한다
  (`reports/<이름>/blink_sim/`). 눈을 잰 값이 아니라 모사다. 수치는 12.1절과 `docs/SPEC.md` 17.3절.

### 4.7 스텝 표

```
python -m drowsy.dataset --config config/default.yaml                              # 정규화 조건별 스텝 표를 캐시에 두고 창 수와 기준선 요약을 출력
python -m drowsy.dataset --config config/default.yaml --fps-sim                    # 누락 모사 추출의 스텝 표
python -m drowsy.dataset --config config/default.yaml --rebuild                    # 캐시를 무시하고 다시 만든다
python -m drowsy.dataset --config config/default.yaml --frame-drop-variants 0 --frame-drop-seed 0   # 프레임 누락 변형(증강 (2))
```

- 옵션: `--arms alert_baseline global session_baseline`(기본 `training.arms`), `--workers N`. 정규화 조건(arm)은 기준선을
  어디서 잡는지다: 피험자의 alert 영상, 전역 상수, 각 영상 자신의 앞 60초(D1).
- 스텝 특징은 저장된 프레임을 `DriverPipeline`에 하나씩 넣어서 얻는다. 뒤 단계의 명령도 캐시가 없으면 스스로 만든다.
- 캐시는 raw 폴더 옆의 `<폴더 이름>_steps/`(합성은 `data/synth_steps/`, 실제는 `data/raw/native_steps/`)에
  `{arm}_{pipeline_hash 앞 16자}[_g…][_fd{변형}_s{시드}_r{min}-{max}].steps.parquet`과 `.quality.parquet`으로 둔다. 열쇠
  (캐시가 지금 데이터와 코드로 만든 것인지 가리는 값)는 `pipeline_hash`, 조건, 추출 파일마다의 크기·수정 시각·**파일 전체의
  sha256**, 그리고 이식 코드와 데이터 모듈의 소스 지문이다(캐시 형식 4). 그 모듈의 파일이 바뀌면(설명 문장만 바뀌어도) 다시
  만든다.
  - 파일 전체의 sha256은 3차 검증 뒤의 정비가 넣었다. 2차 검증 뒤에는 parquet 꼬리말(footer: 파일 끝의 열 구성과 열별
    최솟값·최댓값)만 해시했는데, 두 프레임의 값을 맞바꾼 파일처럼 꼬리말이 같은 다른 내용을 가려내지 못했다. 형식 3 이하의
    캐시는 다음 실행에서 한 번 다시 만들어진다(합성 데이터에서 2026-11-14에 `python -m drowsy.dataset --config
    config/default.yaml`이 세 조건의 캐시를 다시 만드는 데 23.3초, 그 뒤의 실행은 0.82초와 0.84초).
  - 한 명령(프로세스) 안에서는 파일마다 (링크를 푼 경로, 크기, 수정 시각 ns)별로 해시를 기억해 두고 셋이 그대로인 파일은
    다시 읽지 않는다. 학습 한 번이 스텝 표를 여러 번 읽기 때문이다. 기억은 디스크에 남지 않는다.
  - 비용(2026-11-14, 이 PC, `data/synth`의 182개 파일 443,433,671 bytes, 지문을 만드는 함수만 잰 것): 프로세스의 첫 호출
    224 ms(파일들을 운영체제의 페이지 캐시에서 내보낸 뒤에는 406 ms), 그 뒤의 호출은 중앙값 5.7 ms. 비용은 추출 파일의 전체
    크기에 비례한다.
  - 알아채는 것: 내용이 한 바이트라도 바뀐 파일. 크기와 수정 시각을 되돌려 놓은 파일(`cp -p`, `rsync -t`로 덮어쓴 것)도
    다음 명령에서는 다시 읽으므로 알아챈다. 알아채지 못하는 것: 한 명령이 도는 동안 이미 읽은 파일을 크기와 수정 시각을 그대로
    둔 채 바꾼 경우, 그리고 Python·NumPy 판이 바뀐 경우. 그때는 `--rebuild`로 다시 만든다.
- **회전 점검 줄**: manifest 줄 다음(셋째 줄)에 추출 결과 파일들의 회전 점검 기록을 센 한 줄을 출력한다:
  `rotation check of N raw file(s): A decided by the automatic check, B extracted with a manual override, C ambiguous without
  a manual override, D with an override that was not applied, E without a rotation-check record`. C는 가르지 못했는데 보정표에
  각도가 없는 영상, D는 보정표에 적힌 각도가 추출 때 쓴 각도와 다른 영상(보정표를 고친 뒤 추출을 다시 돌리지 않음), E는
  기록이 없는 파일(합성 데이터, 2차 검증 뒤의 정비 전에 만든 추출)이다. C나 D가 있으면 줄 끝에
  `LOOK AT THE K FLAGGED VIDEO(S) before using this data`가 붙고, 이 명령은 리포트를 쓰지 않으므로 그 아래에 영상마다 한 줄씩
  모두 출력한다. 영상은 보정표에 각도를 적고, 그 각도가 추출 때와 다르면 추출을 다시 돌리면 목록에서 빠진다. 숫자는 명령에
  준 설정의 보정표로 센다(추출 때와 다른 설정을 주면 달라진다). 표시된 영상을 빼지 않고 종료 코드도 바꾸지 않는다.
  이 명령은 학습하지 않고 1초 안팎에 끝나므로(캐시가 있을 때), 실제 데이터를 추출한 뒤 **학습 전에 맨 먼저** 돌려 표시된
  영상의 목록을 본다(4.9절의 순서). `train`과 `baseline_lgbm`도 3차 검증 뒤의 정비부터 같은 줄을 데이터를 읽기 전에(출력의
  둘째 줄로) 내고 실행 기록에 남기지만(4.8, 4.9절), 알리기만 하고 멈추지 않고 그대로 학습을 이어 간다. 그래서 목록은 먼저
  이 명령(`python -m drowsy.dataset`)으로 본다.
- 프레임 누락 변형은 세션마다 `training.frame_drop_min`~`frame_drop_max`(0.10~0.30)에서 뽑은 비율로 저장된 프레임을
  하나씩 독립으로 버리고(첫 프레임은 남긴다) 다시 재생한 것이다. 기준선과 T_b(기준선이 완성된 시각)는 솎지 않은
  프레임에서 구한다. 저장된 랜드마크를 다시 쓰는 근사다. 신경망 학습(4.9, 4.10)은 처음 돌 때 `training.frame_drop_variants`개
  (기본 1개: 변형 0, 시드 0, 위 마지막 명령과 같은 캐시)를 만들어 쓴다. 변형 하나마다 전체 세션을 한 번 더 재생한다.

### 4.8 LightGBM 기준선, W 비교, 교란 점검 (E1)

```
python -m drowsy.baseline_lgbm --config config/local.yaml --run-name <이름> probes
python -m drowsy.baseline_lgbm --config config/local.yaml --run-name <이름> cv
python -m drowsy.baseline_lgbm --config config/local.yaml --run-name <이름> cv --all-w
python -m drowsy.baseline_lgbm --config config/local.yaml --run-name <이름> cv --no-velocity
python -m drowsy.baseline_lgbm --config config/local.yaml --run-name <이름> cv --arms global --foldwise-global
```

- 공통 옵션(`--config`, `--data-root DIR`, `--run-name`, `--rebuild`)은 하위 명령 앞에 둔다.
- **회전 점검 줄**(3차 검증 뒤의 정비): 두 하위 명령 모두 출력의 둘째 줄, 곧 데이터 설명 다음 줄에, 학습이나 재생을 시작하기
  전에 4.7절과 같은 회전 점검 줄을 낸다. 회전이 확인되지 않은 영상이 있으면 그 아래에 영상마다 한 줄을 적는다. 같은 값이
  실행 기록(`run.json`, `probes.json`)의 `rotation_check` 키(분류별 파일 수와 표시된 영상 목록)에 남고, 표시된 영상이 있으면
  리포트 설명(summary.md의 첫 문단)에 `ROTATION NOT CONFIRMED for K of N raw file(s) (...)` 문장이 붙는다. 영상을 빼지 않고
  종료 코드도 바꾸지 않는다. 원본 프레임률 추출의 기록만 읽는다(15 fps 모사 추출의 기록은 `python -m drowsy.dataset --fps-sim`이
  출력한다).
- `cv`의 옵션: `--arms ...`(기본 `lgbm.reference_arm`), `--closure-w W`(closure의 W를 바꾼 비교, 보고용), `--fast`,
  `--overwrite`, `--all-w`, `--no-velocity`, `--foldwise-global`. 결과는 `reports/<이름>/lgbm/`. 기준 실행(`--closure-w`
  없음)은 feature_spec의 `closure.w`가 `lgbm.reference_w`(0.5)와 같아야 시작한다. E1 통과 기준
  (`evaluation.lgbm_min_macro_f1`)에 못 미치면 종료 코드 4다. 15 fps 모사 추출이 있으면 fold 모델을 그 추출로도 평가해
  부가 묶음으로 싣는다.
- `cv --all-w`는 기준 조건으로 `lgbm.w_grid`(0, 0.5, 1)의 W를 모두 돌려 `reports/<이름>/lgbm_w_grid/`에 한 리포트와
  `w_grid.md`, `w_grid.csv`(W마다 한 행)를 쓴다. 기준 W의 묶음이 주 결과이고 나머지는 부가 결과다. `lgbm/`은 건드리지
  않고 종료 코드 4를 내지 않는다. `--arms`, `--closure-w`, `--no-velocity`, `--foldwise-global`과 함께 쓸 수 없다.
  `w_grid.md`의 일곱째 줄은 D 번호의 뜻을 알리는 줄(4.9절의 줄과 같은 문장)이다(4차 검증 뒤의 정비가 더했다. 설명 문단의
  D11이 무엇의 번호인지 알린다).
- `cv --no-velocity`(부가)는 속도 특징 #7, #8의 통계 열 12개를 빼고(102 → 90열) `reports/<이름>/lgbm_no_velocity/`에 쓴다.
- `cv --arms global --foldwise-global`(부가, PLAN 4.3절의 민감도 점검)은 전역 상수를 바깥 fold마다 그 fold의 학습 피험자로만
  다시 계산해 전역 조건의 스텝 표를 fold마다 새로 만들고 `reports/<이름>/lgbm_fold_global/`에 쓴다. spec의 상수로 만든 묶음이
  주 결과, fold별 상수의 묶음이 부가 결과다. 쓰인 상수는 `run.json`과 `selection.json`에 남는다.
- 두 부가 실행은 종료 코드 4를 내지 않고, 서로 함께 쓰거나 `--closure-w`와 함께 쓸 수 없다.
- LightGBM은 원본 프레임률 추출의 창으로만 학습한다(15 fps 모사 추출과 프레임 누락 변형을 섞지 않는다). 그래서 D11의 기준
  LightGBM은 신경망의 두 증강 없이 학습한 것이다. 리포트 첫 문단과 명령 출력에 적힌다.
- `probes`의 옵션: `--arm`, `--workers N`. 녹화 메타데이터(밝기, 해상도, 평균 fps, 가변 프레임률 여부)만으로, 그리고 영상별
  기준선 편차만으로 라벨을 맞혀 본다(세션 교란 점검). 메타데이터는 추출이 parquet에 적은 값을 읽고, 없으면(합성 데이터)
  빈 열로 두고 리포트에 그렇게 적는다. 결과는 `reports/<이름>/probes/`.
- 같은 폴더에 다른 조건, 축소 설정, closure W, 다른 종류의 데이터로 만든 리포트가 있으면 멈춘다(`--overwrite`로 덮어쓴다).
- 종료 코드 3은 피험자 겹침 검사(`assert_subject_disjoint`)에 걸려 멈춘 것이다.

### 4.9 중첩 교차검증 (E2, E3)

```
python -m drowsy.train --config config/local.yaml --run-name <이름> cv
python -m drowsy.train --config config/local.yaml --run-name <이름> cv --ordinal        # 부가: 순서형 소프트 라벨
python -m drowsy.train --config config/local.yaml --run-name <이름> cv --no-velocity    # 부가: 속도 특징 #7, #8을 0으로
```

- 옵션: `--arm`(기본 `training.export_arm` = `alert_baseline`), `--families tcn gru`, `--fast`, `--overwrite`,
  `--allow-partial-fps-sim`, `--ordinal` 또는 `--no-velocity`(둘 중 하나). 공통 옵션(`--config`, `--data-root`, `--run-name`,
  `--rebuild`)은 하위 명령 앞에 둔다.
- **한 실행 폴더의 작업 순서**(같은 `--run-name`):
  1. `python -m drowsy.dataset --config config/local.yaml`(4.7절. 또는 그 전에 `signal_check e0`, 4.6절): 스텝 표를 만들고
     회전 점검 줄과 회전이 확인되지 않은 영상의 목록을 본다. 목록이 비어 있지 않으면 그 영상을 직접 보고 보정표
     (`extraction.rotation_overrides`)에 각도를 적는다. 적은 각도가 추출 때의 각도와 다르면 추출(4.5절)을 다시 돌린다. 목록이
     빌 때까지 이 단계를 되풀이한다. 명령들은 알리기만 하고 학습을 막지 않는다(그것을 바꿀지는 9장 17번의 질문이다).
  2. `baseline_lgbm probes`, 3. `baseline_lgbm cv`, 4. `train cv`, 5. 부가 실행들, 6. `train final`.
  같은 `--run-name`으로 4.8의 `cv`를 먼저 돌려 두면 요약에 LightGBM 행과 D11 판정이 함께 실린다. 그 `lgbm/`이 같은
  데이터, 같은 축소 설정 여부, 같은 스텝 표(내용 지문), 같은 closure W로 만든 것인지는 학습 전에 검사하고, 아니면 멈춘다.
- **회전 점검 줄**(3차 검증 뒤의 정비): `cv`와 `final` 모두 출력의 둘째 줄에 4.8절과 같은 회전 점검 줄을 낸다. 실행 기록
  (`run.json`, `final/final.json`)의 `rotation_check`와 리포트 설명의 `ROTATION NOT CONFIRMED ...` 문장도 4.8절과 같다.
- **학습 표집**: 피험자 → 영상 → 출처 → 그 영상의 보폭 1 창 순서로 고르게 뽑는다. 출처의 몫은 원본 추출 0.5, 15 fps 모사
  추출 0.25(`training.fps_sim_share`), 프레임 누락 변형 0.25(`training.frame_drop_share`, 변형이 여럿이면 똑같이 나눈다)다.
  출처가 없거나 그 영상의 창이 없으면 그 몫은 원본으로 간다. 표준화 상수도 같은 몫으로 가중한다. 학습 피험자의 창만 쓰고,
  내부 검증, 시험, 온도 맞추기는 원본 창만 쓴다. 15 fps 모사 추출이 원본에 있는 세션을 다 덮지 못하면 시작하지 않는다
  (`--allow-partial-fps-sim`으로 풀고, 덮은 비율이 리포트에 적힌다).
- 결과는 `reports/<이름>/`: `summary.md`, `summary.json`, fold별 CSV와 JSON, `predictions.parquet`(parquet 메타데이터에
  데이터 종류), 오경보 표, `videos.csv`(기준선 상태, `t_b_ms`, 기준선 출처, alert 기준선 대체 여부 포함), `selection.json`
  (fold마다 다시 학습한 모델의 가중치 해시와 출처 포함), `timings.json`, `run.json`(학습 출처와 몫, 스텝 표 지문, 기준선 요약,
  부가 평가의 상태), 설정 사본. summary.md 8절은 기준선이 연장되거나 전역 상수로 대체된 영상과 조건별·보폭별·클래스별 창 수,
  9절은 부가 결과다. 부가 결과는 주 결과(1~7절)와 D11에 쓰지 않는다. `supplementary_per_fold.csv`,
  `supplementary_alarm.csv`, `supplementary_alarm_sweep.csv`는 늘 쓴다(부가 묶음이 없으면 머리줄만). summary.md의 일곱째
  줄(앞뒤에 빈 줄)은 ``이 리포트의 D 번호(D 뒤에 숫자가 붙은 표기)는 계획서 `docs/PLAN.md` 1.2절의 표에서 승인된 결정의
  번호다. 같은 문서 14장에 실린 소유자 결정(2026-11-13, 2026-11-14)은 따로 번호를 매기며, 그 번호와는 다르다.``이다(3차
  검증 뒤의 정비가 넣었고, 4차 검증 뒤의 정비가 소유자 결정의 수를 세지 않는 이 문장으로 고쳤다. 리포트 문구의 `D11 판정`,
  `(D7: ...)` 같은 표기를 위한 것이다). 같은 줄이 `train final`의 `final/summary.md`(4.10절), `baseline_lgbm cv --all-w`의
  `w_grid.md`(4.8절)의 일곱째 줄과 라벨 없는 영상 평가의 summary.md 다섯째 줄(4.11절)에도 있다.
- **주 실행에 함께 실리는 부가 묶음**: 15 fps 모사 추출이 있으면 fold 모델을 시험 fold의 그 추출로도 평가한다(학습 조건과
  앱 조건 둘 다). fold마다 다시 학습한 TCN을 임시 폴더로 내보내 INT8(QUInt8 동적 양자화)로 바꾸고, float ONNX와 INT8 파일을
  onnxruntime으로 시험 fold에 돌려 나란히 싣는다(`tcn_onnx`, `tcn_onnx_int8`. INT8 대 float ONNX, float ONNX 대 PyTorch의
  짝지은 비교). INT8 비교는 `export.int8_if_tcn`을 따른다.
- **부가 실행**(`--ordinal`, `--no-velocity`)은 `reports/<이름>/ordinal/`, `reports/<이름>/no_velocity/`에 따로 쓰고 모든 묶음이
  부가다. 주 실행의 파일은 읽기만 하고, 같은 폴더의 주 실행과 짝지은 비교를 싣는다. 주 실행이 다른 데이터·코드·축소
  설정으로 만든 것이면 학습 전에 멈춘다(종료 코드 2). 주 실행이 없으면 짝 없이 돈다. 순서형 소프트 라벨은 라벨 스무딩 대신
  정답 클래스가 이웃 클래스에 `training.ordinal_neighbor_mass`(0.2)를 나눠 주는 목표다(alert [0.8, 0.2, 0], low vigilant
  [0.1, 0.8, 0.1], drowsy [0, 0.2, 0.8]). 속도 특징 제외는 표준화한 입력 #7, #8을 학습, 검증, 평가 모두에서 0으로 둔다(입력은
  17개 그대로).
- `--arm`을 `training.export_arm`이 아닌 조건으로 주면 그 실행의 묶음은 모두 부가로 표시된다(9장 15번의 질문). 그런
  실행은 D11 판정을 하지 않는다: `summary.json`의 판정(`gate`)이 `null`이고, 리포트 설명 끝에 그 사실이 적히고, 명령이
  `supplementary analysis (training arm <조건> is not the export arm <내보낼 조건>): every group is supplementary and D11 is
  not evaluated`를 출력한 뒤 리포트를 끝까지 쓴다(2차 검증 전에는 학습을 마친 뒤 종료 코드 2로 멈췄다).
- 리포트를 쓰기 전에 예측 표를 검사한다(`evaluate.validate_predictions`): 출처(`feature_source`)가 원본(`native`)이 아니거나
  변형(`variant`)이 기본(`base`)이 아닌 묶음, 곧 15 fps 평가, 순서형, 속도 제외, fold별 전역 상수의 묶음은 부가
  (`supplementary`)로 표시돼 있어야 하고, 아니면 멈춘다. 그 열이 없는 옛 파일은 두 열에서 부가 여부를 정한다. 그런 묶음이
  주 결과나 D11에 섞이는 일을 막는다.
- 리포트 설명에는 closure 상수와 전역 상수가 E0에서 라벨 없이 시험 피험자를 포함한 전체 영상으로 한 번 정해진다는
  예외가 적힌다(PLAN 4.3절). `config.yaml`과 `rules.json` 사본은 바이트 사본이라 데이터 종류 표시가 없다.
- `--fast`는 설정의 `fast` 블록으로 탐색 범위와 학습 길이를 줄인 연기 시험용이다. 사전에 고정한 절차가 아니므로 그
  수치는 보고하지 않는다.
- 종료 코드: 0, 2 시작할 수 없음, 3 피험자 겹침.

### 4.10 최종 학습과 내보내기 (E4)

```
python -m drowsy.train --config config/local.yaml --run-name <이름> final       # → export/drowsiness_v1.onnx, feature_spec_v1.json, rules.json
python -m drowsy.export_onnx --random-init --kind tcn --version 0 --out-dir <폴더>
python -m drowsy.export_onnx --checkpoint reports/<이름>/final/final_model.pt --spec <내보낸 폴더>/feature_spec_v1.json --version 1 --out-dir <폴더>
```

- `final`의 옵션: `--out-dir DIR`(기본 `paths.export_dir` = `export/`), `--version N`(기본 1), `--families`, `--fast`,
  `--overwrite`, `--allow-partial-fps-sim`. 학습 표집은 4.9와 같다. 모델 옆에 규칙 설정 사본 `rules.json`을 쓰고 그
  `rules_hash`를 `reports/<이름>/final/final.json`에 적는다. 그 폴더에는 선택 기록, 학습 출처, `summary.md`, 체크포인트
  `final_model.pt`도 있다. TCN이면 INT8 파일 `drowsiness_v{N}_int8.onnx`도 만들고 float 모델과의 최대 확률 차이를 적는다.
  이것은 확률이 얼마나 움직였는지일 뿐 지표 변화가 아니다(지표 변화는 4.9의 fold 모델 INT8 묶음에 있다).
  `final/summary.md`의 일곱째 줄은 D 번호의 뜻을 알리는 줄(4.9절의 줄과 같은 문장)이다(4차 검증 뒤의 정비가 더했다. 아래
  항목의 D10, D16이 무엇의 번호인지 알린다).
- **합성 데이터로 학습한 모델과 `--fast` 모델은 내보내기 폴더와 그 아래 폴더로 내보내지 못한다.** 내보내기 폴더는 둘이다:
  설정의 `paths.export_dir`, 그리고 설정이 무엇을 가리키든 이 저장소의 `export/`(`drowsy` 패키지의 부모 폴더 아래의 `export`).
  `train final`도, `export_onnx --checkpoint`도 같은 함수(`export_onnx.protected_export_dir`. `is_export_dir_or_below`가 그
  판단을 참·거짓으로 돌려준다)로 검사한다. 합성 데이터로 돌릴 때는 `--out-dir out/<이름>`을 준다.
  - `paths.export_dir`이 상대 경로(기본 `export`)이면 지금 폴더 기준과 저장소 루트 기준 둘 다를 내보내기 폴더로 보므로,
    저장소 밖의 폴더에서 명령을 돌려도 저장소의 `export/`에 쓰지 못한다(2차 검증 뒤의 정비).
  - 저장소의 `export/`는 늘 지킨다(3차 검증 뒤의 정비). 연기 시험용 설정은 `paths.export_dir`을 임시 폴더로 적어 두는데,
    그런 설정으로 저장소 루트에서 `--out-dir export`를 주면 합성 모델이 커밋하는 폴더에 들어가던 것을 막는다. 거부 문장은
    `... must not be exported to the export folder of this repository (<경로>; it is protected whatever paths.export_dir
    names, here <설정의 값>) ...`이다.
  - 한계: 패키지를 다른 곳에 복사해 쓰면 그 복사본의 부모 폴더가 저장소 루트로 쓰이므로, 그때 원래 저장소의 `export/`는
    설정에 그 절대 경로를 적을 때만 보호된다. 실제 데이터와 정식 설정으로 학습한 모델은 이 검사와 무관하게 어느 폴더로든
    나간다.
- `export_onnx`의 옵션: `--random-init`(`--kind tcn|gru`, `--seed`), `--checkpoint`, `--version`, `--spec`, `--out-dir`
  (기본 `paths.export_dir`), `--dynamo`(대체 내보내기 경로). `--random-init`에 `--out-dir`을 주지 않으면 `export/`의 v0을
  덮어쓰므로 시험할 때는 늘 임시 폴더를 준다.
- `--checkpoint`에는 `--spec`과 `--version`이 꼭 필요하다. `--spec`은 그 학습이 모델 옆에 쓴 `feature_spec_v{N}.json`이다.
  체크포인트(형식 `drowsy-checkpoint-v2`)에는 학습한 spec 사본의 sha256, 합성 데이터 여부(`synthetic`), 축소 설정 여부
  (`fast`)가 있다. spec이 다르면 거부하고, 합성 데이터나 축소 설정 체크포인트는 위의 두 내보내기 폴더와 그 아래로 내보내지
  못한다. `fast` 키가 없는 v2 파일(그 키를 더하기 전에 만든 것)은 축소 설정이 아닌 것으로 읽는다. 형식 v1 체크포인트(정비
  전에 만든 것)는 거부하고 `train final`을 다시 돌리라고 알린다.
- 두 명령 모두 계약 검사(파일 이름 포함), PyTorch 대 onnxruntime 일치 검증, 지연 측정을 하고 결과를 출력한다.
  종료 코드: 0, 1 일치 검증 실패, 2 오류(한 줄로 알린다).
- 계약 검사(`verify_contract`)는 세 번에 걸쳐 엄격해졌다. 계약의 값(파일 이름, opset, 입출력, 메타데이터 키)은 그대로이고
  저장소의 v0과 이 저장소의 정상적인 내보내기는 모두 통과한다.
  - 온도(2차 검증 뒤): 신경망이 낸 점수(logits)를 바꿀 수 있는 것은 Softmax 바로 앞의 `temperature`라는 이름의 상수로 나누는
    것뿐이다. 기본 경로와 INT8 파일에는 그 상수가 꼭 있어야 하고(온도가 1.0이어도), 대체 경로(`--dynamo`)로 만든 파일만
    온도가 1.0일 때 그 상수가 없을 수 있다. 어느 경로든 logits가 Div나 Mul 노드의 출력이면 거부한다.
  - logits를 내는 노드(3차 검증 뒤): logits는 신경망의 마지막 선형 층 그대로여야 한다. 기본 경로와 대체 경로에서는 그래프의
    유일한 Gemm 노드(행렬 곱과 편향 더하기를 하는 ONNX 연산)의 출력이고, 그 Gemm은 `alpha = 1`, `beta = 1`, `transA = 0`이며
    가중치와 편향을 그래프의 float32 상수에서 읽는다. INT8 파일에서는 양자화기가 그 Gemm을 바꿔 쓴 모양(입력 양자화 →
    유일한 MatMulInteger → float32로 Cast → 배율 Mul → 상수 편향 Add)의 끝에 있는 편향 Add의 출력이다. 그래서 logits에 상수를
    곱한 뒤 값을 바꾸지 않는 노드(0을 더하는 Add, 입력이 하나인 Concat 등)를 끼우거나, 배율을 Gemm의 속성이나 둘째 Gemm에
    넣은 파일도 거부한다.
  - 속성 읽기(4차 검증 뒤): 위 Gemm의 세 속성(`alpha`, `beta`, `transA`)과 INT8 파일의 Cast가 바꾸는 형(`to`)은 한 번만,
    정해진 형(`alpha`·`beta`는 FLOAT, `transA`·`to`는 INT)으로, 값을 가진 채 있어야 한다. 아니면 한 줄짜리 계약 오류
    (`ContractError`)다. 전에는 형이 UNDEFINED이거나 문자열인 속성에서 계약 오류가 아니라 Python의 `TypeError`가 났고, 값이 없는
    INT 속성(`transA`)은 받아들였다(onnxruntime은 그 파일을 기본값으로 그대로 돌리므로 출력은 같았다). 이 저장소의 내보내기는
    모든 속성에 값을 적으므로 정상적인 파일은 그대로 통과한다(정상적인 내보내기 71가지와 v0, 정비 작업자의 기록).
  - 이 검사가 보지 못하는 것: 마지막 층의 가중치와 편향 자체를 통째로 줄이거나 키운 파일(logits에 상수를 곱한 것과 같다).
    그래프의 모양과 속성이 그대로이기 때문이다. 신경망 안쪽(Clip과 마지막 층 사이)도 보지 않으므로 클래스 순서 바꾸기도
    잡지 못한다. 가중치는 체크포인트의 PyTorch 모델과 출력을 견주는 일치 검증(`export_onnx --checkpoint`와 `train final`이
    내보낸 직후에 한다)이 지킨다. 받은 파일을 의심할 때도 체크포인트와 출력을 견준다. 오류 문장도 이 범위를 적는다.
- 내보내기는 git 상태를 파일을 쓰기 전에 읽는다(깨끗한 커밋에서 처음 내보낸 버전이 `-dirty`로 찍히지 않게). 모델과 spec
  사본은 임시 이름으로 쓴 뒤 계약 검사를 통과하면 바꿔 넣는다. 성공하면 같은 버전의 옛 INT8 파일을 지운다.

### 4.11 라벨 없는 차량 영상의 시간당 경고 수 (M4)

```
python -m drowsy.extract --config config/local.yaml --unlabeled --run-name <이름>
python -m drowsy.evaluate unlabeled --config config/local.yaml --run-name <이름> [--raw-dir DIR] [--model PATH] [--speed-kmh X]
```

- `paths.unlabeled_root` 아래의 영상을 추출해 `data/raw/unlabeled_native/`에 두고, 그 parquet을 모두 `DriverPipeline`으로
  다시 돌려 이벤트 종류별 상태 시작 수, 경고 수, 분석 가능한 시간당 경고 수를 낸다. 경고 수는 바로 낸 경고
  (`immediate_alerts`), 막혔다가 나중에 낸 경고(`deferred_alerts`, 소유자 결정 2), 되풀이 경고(`repeat_alerts`, 수면의 3초
  주기)의 합이고 세 수도 따로 싣는다. 분석 가능한 시간은 얼굴이 검출된 시간이다(세션 길이도 함께 싣는다).
- 결과: `reports/<이름>/unlabeled/`의 `alerts.csv`(첫 줄은 `#` 주석이므로 `pd.read_csv(path, skiprows=1)`로 읽는다),
  `summary.md`, `config.yaml`, `rules.json`. 기본 실행 이름은 `nested_cv`다.
- 기본 모델은 `demo.model_path`이고 그 파일이 없으면 `export/drowsiness_v0.onnx`(무작위 가중치)를 쓰며 리포트에 적는다.
  `drowsiness.source`가 `perclos`인 동안 모델은 경고에 영향이 없다.
- `--speed-kmh`를 주지 않으면 속도 억제를 하지 않는다. 주면 세션 내내 같은 속도다. summary.md의 다섯째 줄은 D 번호가 계획서
  1.2절의 결정 번호라는 줄(4.9절의 줄과 같은 문장)이고, 속도 억제는 `속도 억제(PLAN 1.2절 결정 D3)`로 적힌다(3차 검증 뒤의 정비. 2026-11-13의
  소유자 결정 3과 헷갈리지 않게 했다).
- 라벨이 없으므로 정확도가 아니라 경고 수다. 다시 돌릴 수 없는 파일은 이유와 함께 목록에 남고 합계에서 빠진다.
  종료 코드: 0 리포트를 씀, 2 시작할 수 없음.
- **회전 점검**: 마지막 출력 줄이 4.7절과 같은 형식의 회전 점검 줄이고, summary.md의 4절 "회전 점검"에 표시된 영상의 목록이,
  머리말 목록에 경고 한 줄이 붙는다. 보정표는 `extraction.unlabeled_rotation_overrides`와 비교한다(`--raw-dir`로 다른 폴더를
  줘도 같다). 표시된 영상도 합계에 들어간다. 그 영상의 경고 수는 각도를 확인하기 전에는 믿을 수 없다.

### 4.12 데모

```
python -m drowsy.demo --raw data/synth/Fold1_part1/01/10.parquet                    # 추출했거나 합성한 parquet (MediaPipe 불필요)
python -m drowsy.demo --video path.mp4 [--rotation 90] --model export/drowsiness_v0.onnx --out out/demo --speed-kmh 60
```

- `path.mp4`는 예시 경로다. 영상은 저장소 루트에 두어도 `.gitignore`가 제외한다(4.0절).

- 입력의 앞 60초를 기준선으로 써서 전체 파이프라인을 돌리고 `<입력 이름>_timeline.png`와 `<입력 이름>_events.json`을 쓴다.
- 타임라인은 가로축을 맞춘 네 줄이다: closure와 임계값, 전방 주시 상태, 모델 출력(`p_drowsy`와 옅은 `p_low_vigilant`, 모델
  호출마다 점 하나), 규칙 이벤트(막대는 상태가 이어진 시간, 색은 심각도, 임시 판정은 빗금). 경고는 세 가지 표식으로
  구분한다: 바로 낸 경고(`alert`), 막혔다가 나중에 낸 경고(`deferred alert`), 되풀이 경고(`repeat alert`).
- JSON에는 이벤트 목록, 경고 목록(경고마다 `kind`: `immediate`, `deferred`, `repeat`), 계수(`model_calls`, `alerts_immediate`,
  `alerts_deferred`, `alerts_repeat` 등)와 실행 메타데이터가 있다.
- 실행 전에 모델 메타데이터의 `feature_spec_hash`와 `pipeline_hash`를 spec 파일과 대조하고, 다르면 거부한다. 읽을 수 없는
  모델 파일도 거부한다(종료 코드 2).
- `--model`을 주지 않으면 설정의 `demo.model_path`(`export/drowsiness_v1.onnx`)를 쓰고, 그 파일이 아직 없으면
  `export/drowsiness_v0.onnx`(무작위 가중치)를 쓰며 그 사실을 알린다. `--out`의 기본은 `out/demo`다.
- 첫 모델 출력은 세션 239초에 나온다. 그보다 짧은 입력은 모델 줄이 비고 이유가 그림에 적힌다.
- `--speed-kmh`를 주면 경고의 속도 억제(시선·감김 20 km/h, 졸음 50 km/h)가 적용된다. 주지 않으면 억제하지 않는다.
- 마이크로슬립·수면 이벤트의 막대는 감김 구간 전체(반쯤 뜬 구간 포함, 소유자 결정 1)를 덮는다.
- **얼굴 프레임은 어디에도 저장하지 않는다**(D13). 같은 영상을 여러 번 볼 때는 4.5의 `--video`로 한 번 추출하고 `--raw`를 쓴다.
- **회전(`--video`)**: `--rotation {0,90,180,270}`을 주면 그 각도로 세운다(추출의 보정표와 같은 규약, 반시계 각도). 주지
  않으면 프레임을 처리하기 전에 추출과 같은 회전 자동 점검(4.5절)을 한 번 돌린다. 어느 쪽이든 표준 출력의 첫 줄이 회전 줄이다:
  `rotation: auto check chose N, ambiguous: yes|no (...)`(가르지 못했으면 `WARNING` 문장과 `--rotation`을 주라는 안내가
  붙는다) 또는 `rotation: N (given with --rotation; the automatic check was not run)`. 그래서 각도가 틀렸으면 긴 실행을 일찍
  멈출 수 있다. 가르지 못해도 실행은 계속한다. 데모는 추출 설정의 보정표를 읽지 않는다(그 키는 추출 입력 폴더 기준이다).
- JSON의 `source`에는 적용한 각도 `rotation`과 함께 `rotation_source`(`manual_override` 또는 `auto_check`),
  `rotation_ambiguous`(참/거짓), `rotation_check`(추출 parquet의 기록과 같은 점검 값. 수동 각도이면 `null`),
  `display_flip`(표시 행렬에 뒤집기가 있었는지)이 있다. 타임라인 그림의 머리말에 넷째 줄 `video rotation: N degrees
  counter-clockwise (automatic check | automatic check, AMBIGUOUS: ... | given with --rotation)`이 붙는다.
- 표시 행렬에 뒤집기가 든 영상은 `--rotation` 없이는 종료 코드 2로 거부하고, 메시지가 `--rotation`을 알려 준다. 주면 그
  각도로만 돌리고 거울상은 되돌리지 않으며 JSON에 `display_flip: true`를 적는다.
- `--rotation`을 `--raw`와 함께 주면 종료 코드 2다(parquet은 추출할 때 이미 세웠다).
- **회전(`--raw`, 3차 검증 뒤의 정비)**: 데모는 parquet을 돌리지 않고, 추출이 메타데이터에 남긴 회전 기록을 읽어 한 줄로
  알린다. 합성 데이터 표시 줄이 있으면 그 다음 줄이다. 줄은 위에서부터 처음 맞는 것이다:
  - 합성 세션: `rotation: no rotation-check record (synthetic session: there was no video to turn upright)`.
  - 점검 기록이 있다: `rotation: the extraction's auto check chose N, ambiguous: yes|no (...)`. 가르지 못했으면 `WARNING
    rotation ambiguous: ...`과, 영상을 직접 보고 각도를 손으로 줘서 다시 추출하라는 안내(`extract --video ... --rotation`,
    다른 입력은 보정표)가 붙고 그림의 머리말에도 한 줄이 붙는다. 고른 각도가 영상의 회전 정보와 다르면 `WARNING rotation
    differs from the video metadata`가 붙는다.
  - 기록이 없고 수동 각도로 추출됐다: `rotation: N (manual value recorded by the extraction; the automatic check was not run)`.
  - 기록도 수동 각도도 없다: 회전 정보의 각도만 썼다는 것과 점검하려면 다시 추출하라는 것, 또는 기록이 없다는 것.
  같은 문장이 JSON `source`의 `rotation_note`에 들어가고, `rotation`(적용된 각도), `rotation_source`(`auto_check`,
  `manual_override`, `video_metadata` 또는 `null`), `rotation_ambiguous`, `rotation_check`(점검 기록 또는 `null`)도 적힌다.
  한계: 데모는 설정의 보정표를 보지 않는다(보정표의 키는 추출 입력 폴더 기준이라 parquet 하나로는 어느 영상인지 정해지지
  않는다). 그래서 가르지 못한 영상을 보정표에 같은 각도로 적어 확인한 경우에도 데모에서는 가르지 못했다고 나온다. 보정표까지
  대조한 판정은 4.7절의 명령으로 본다.

### 4.13 실시간 시험: Windows의 웹캠 (`python -m drowsy.live`)

```
python -m drowsy.live [--source SOURCE] [--rotation {0,90,180,270}] [--port PORT] [--no-preview]
                      [--model PATH] [--speed-kmh X] [--out DIR] [--config PATH]
```

**무엇을 위한 것인가.** 카메라 앞에 앉아 파이프라인 전체(MediaPipe → `DriverPipeline` → 규칙 이벤트와 경고)를 실시간으로
돌려 보는 **데스크톱 시험 도구**다. 저장소는 WSL2(Windows 안의 Linux) 안에 있고 웹캠은 Windows에 붙어 있어서 WSL에서는
카메라가 보이지 않는다(`/dev/video*`가 없다). 그래서 Windows의 ffmpeg(영상 변환 프로그램)가 카메라 영상을 H.264로 압축해
MPEG-TS(방송용 영상 묶음 형식)로 TCP `localhost`에 보내고, 이 명령이 WSL 안에서 받아 프레임이 오는 대로 처리한다. 상태는
터미널(1초에 한 줄)과 Windows 브라우저로 여는 작은 웹 페이지에 나오고, 세션이 끝나면 그 세션의 프레임 표와, 그 표로 4.12절의
`--raw` 데모와 같은 타임라인 그림과 이벤트 JSON을 쓴다. 소유자의 요청으로 2026-11-14에 더한 것이고 승인된 계획의 산출물이
아니다(PLAN 14.4절). 이식 대상이 아니며 Android 팀에 가는 파일은 하나도 바꾸지 않았다. 파이프라인은 데모와 같은 인자로 만들어
(`DriverPipeline(spec, rules, None, None, model)`) 기준선을 주지 않으므로, 세션 자신의 앞 60초로 보정한다.

이 절에서 "검토", "수정 단계", "최종 점검"은 2026-11-14에 이 도구를 만들면서 거친 세 단계(만든 직후의 검토, 그 지적을 고친
단계, 마지막 확인)이고, README 맨 앞에서 말하는 2026-11-13의 독립 검토와는 다르다. "작업을 맡긴 쪽"은 이 도구를 만드는 작업을
지시하고 Windows 쪽의 접속을 직접 확인한 쪽이다.

**한 번만 하는 준비**

- WSL: `models/face_landmarker.task`(3장). `--video` 데모와 같이 시작할 때 sha256을 spec의 값과 대조한다.
- Windows PowerShell: ffmpeg를 설치하고, PowerShell 창을 새로 열어 H.264 인코더(libx264)가 들어 있는지 본다.
  ```
  winget search ffmpeg
  winget install --id Gyan.FFmpeg -e
  ffmpeg -hide_banner -encoders | findstr libx264
  ```
  가정: 이 세 명령과 패키지 이름 `Gyan.FFmpeg`는 이 README를 쓰며 실행하지 않았다(Windows에는 아직 ffmpeg가 없고, 이 작업은
  Windows 프로그램을 실행하지 않았다). `winget search`의 목록에서 다른 ffmpeg 패키지를 골라도 마지막 명령에 libx264 줄이 나오면 된다.

**순서** (괄호 안이 실행하는 곳)

1. (Windows PowerShell) 카메라 이름을 본다: `ffmpeg -hide_banner -list_devices true -f dshow -i dummy`. 출력에서 영상 장치의
   이름을 옮겨 적는다. 한글이 든 이름이 3번에서 찾아지지 않으면 같은 장치에 함께 나오는 대체 이름(`Alternative name`)을 쓴다.
   그 카메라가 내는 해상도와 프레임률은 `ffmpeg -hide_banner -list_options true -f dshow -i video="<카메라 이름>"`로 본다.
2. (WSL 터미널, 저장소 루트) `python -m drowsy.live`. 2초쯤 뒤(1.8초와 2.0초를 쟀다) 무엇을 돌리는지 알리는 시작 블록 10줄이
   나오고(아래), 보내는 쪽의 접속을 **120초까지** 기다린다. 기다리는 동안 30초마다 남은 시간을 알리는 줄이 나온다
   (`still waiting for the sender (90 s left)`, `(60 s left)`, `(30 s left)`. 남은 시간이 0초로 적힐 줄은 내지 않는다).
3. (Windows PowerShell, 2가 기다리는 동안) 보내는 쪽을 시작한다.
   ```
   ffmpeg -hide_banner -f dshow -video_size 640x480 -framerate 30 -i video="<카메라 이름>" -c:v libx264 -preset ultrafast -tune zerolatency -g 30 -pix_fmt yuv420p -f mpegts tcp://localhost:5599
   ```
   - 같은 명령이 시작 블록의 `sender example` 줄에 이 실행의 실제 포트로 나오고, `--help`의 끝에도 있다.
   - `-video_size`와 `-framerate`는 1번의 목록에 있는 조합으로 바꾼다.
   - **이 프로그램을 먼저** 실행한다. 먼저 시작한 ffmpeg는 접속이 거부되어 바로 끝나고 다시 시도하지 않는다(검토 때 이 PC의
     Linux ffmpeg 7.0.2로 본 동작). 120초 안에 영상이 오지 않으면 이 프로그램은 표준 오류에 `error: no video from a sender on
     <소스> within 120 s: either no sender connected, or a client stayed connected and sent no MPEG-TS video (ffmpeg: -f
     mpegts). Fix: ...`를 내고 종료 코드 2로 끝나므로 2번부터 다시 한다. 아무도 접속하지 않은 경우와, 접속한 쪽이 MPEG-TS
     영상을 보내지 않은 경우(예: ffmpeg 명령에 `-f mpegts`가 빠짐)는 이 프로그램이 가를 수 없어서 두 경우를 함께 적는다.
   - 주소: 2026-11-14에 Windows PowerShell에서 연 TCP 접속이 WSL 안의 듣는 프로그램에 `localhost`로도 `127.0.0.1`로도 닿았고,
     웹 페이지도 두 이름 모두에서 응답했다(작업을 맡긴 쪽이 확인한 것). Windows의 ffmpeg로 영상을 보내 본 적은 아직 없다.
     접속이 거부되거나, ffmpeg가 돌기 시작했는데도 이 프로그램에 `source opened` 줄이 나오지 않고 `still waiting` 줄만 이어지면
     끝의 주소를 `tcp://127.0.0.1:5599`로 바꿔 본다(`--help` 끝의 안내 2번도 이 두 경우를 적는다). 이 PC의 Linux ffmpeg 7.0.2는
     `tcp://localhost`로는 접속하지 못한 채 12초 동안 아무 말 없이 있다가 끝났고 `tcp://127.0.0.1`로는 접속했다(최종 점검.
     그 판은 이 이름을 풀지 못한다). Windows의 ffmpeg가 어느 쪽인지는 모른다. 시작 블록의 `sender example` 줄과 오류 문장은
     그대로 `localhost`를 적는다.
   - `-g 30`은 키프레임(앞 프레임 없이 풀 수 있는 프레임) 간격이다(30 fps에서 1초). 세션은 첫 키프레임에서 시작하므로 짧게
     둔다. 키프레임 옵션이 없으면 첫 프레임이 보내는 쪽을 시작한 지 7.92초 뒤에 왔고 `-g 30`이면 0.59초였다(검토, Linux
     ffmpeg, 합성 그림).
   - `-re`, 출력 쪽 `-r`, `-fps_mode cfr`은 붙이지 않는다. 파이프라인의 시간은 모두 프레임 시각에서 나오는데, 검토에서
     `-fps_mode cfr`은 프레임 시각을 고른 간격으로 바꾸고 빈 구간을 복제 프레임으로 채웠다.
   - ffmpeg가 `real-time buffer too full`을 알리면 `-i` 앞에 `-rtbufsize 100M`을 더한다.
   - 이 명령에서 `-i`까지의 카메라 입력 부분은 **실행해 보지 못했다**(옵션의 이름과 자리는 ffmpeg 공식 문서와 대조했다).
     그 뒤의 인코더와 출력 부분은 PyAV로 만든 보내는 쪽과 이 PC의 Linux ffmpeg 명령(주소 `127.0.0.1`, 합성 그림)으로 확인했다.
4. (Windows 브라우저) `http://localhost:8765/`를 연다. 카메라 미리보기를 보며 얼굴이 화면 가운데에 똑바로 오게 앉는다.
   미리보기는 MediaPipe에 넘긴 그대로의 그림이라 거울상이 아니다. 그림이 옆으로 누웠거나 거꾸로면 Ctrl+C로 끝내고
   `--rotation`(세우려면 반시계로 돌려야 하는 각도)을 줘서 다시 시작한다. 실시간 스트림에는 회전 자동 점검이 없다.
5. 첫 프레임부터 60초가 **기준선**이다(얼굴 프레임이 모자라면 30초씩 180초까지 늘어난다). 그동안은 눈을 평소처럼 뜨고 앞을
   본다. 기준선이 완성되기 전에는 감김 이벤트가 임시 판정(`provisional`)이고 옆을 봐도 이벤트가 나지 않는다(D4). 그 뒤에 눈을
   감거나 옆을 보면 이벤트가 난다(어떤 동작이 어떤 이벤트를 내는지는 `docs/GUIDE.md` 5.7절).
6. 끝낼 때는 WSL 터미널에서 Ctrl+C를 누르거나 보내는 쪽을 멈춘다. 세 파일은 그때 쓰이므로 `wrote` 줄이 나올 때까지 창을
   닫지 않는다. 이 프로그램을 먼저 끝내면 ffmpeg는 `Broken pipe`를 알리고 스스로 끝난다(검토, Linux ffmpeg).

**옵션** (2026-11-14에 `python -m drowsy.live --help`로 확인했고, 4차 검증 뒤의 정비가 끝난 2026-11-15에 다시 돌려 같았다)

| 옵션 | 기본값 | 뜻 |
|------|--------|----|
| `--source` | `tcp://127.0.0.1:5599?listen=1` | PyAV가 열 수 있는 것. 기본은 TCP로 듣고 MPEG-TS(H.264)를 받는다. `/dev/video0` 같은 Linux 카메라 장치(WSL에 장치가 없어 실행해 보지 못했다)나 영상 파일(그 파일의 프레임 시각대로 실제 속도로 재생한다. 파일 전체를 빠짐없이 돌리려면 `demo --video`)도 된다 |
| `--rotation` | 0 | 그림을 세우려고 반시계로 돌리는 각도. 0, 90, 180, 270 |
| `--port` | 8765 | 웹 페이지의 포트(127.0.0.1에만 묶는다). 0이면 비어 있는 포트를 시스템이 고른다 |
| `--no-preview` | 미리보기 있음 | 카메라 미리보기를 만들지도 내주지도 않는다(페이지에는 상태만) |
| `--model` | 데모와 같다 | 설정의 `demo.model_path`, 그 파일이 아직 없으면 `export/drowsiness_v0.onnx`(무작위 가중치)와 표준 오류의 알림. 대조도 데모와 같다(4.12절) |
| `--speed-kmh` | 없음 | 세션 내내 같은 차량 속도. 주지 않으면 경고를 속도로 억제하지 않는다(책상 앞의 시험에서는 주지 않는다) |
| `--out` | `<demo.out_dir>/live`(기본 설정에서 `out/live`) | 출력 폴더. 세션을 시작하기 **전에** 만들고 쓸 수 있는지 확인한다. 쓸 수 없으면 아무것도 시작하지 않고 종료 코드 2 |
| `--config` | `config/default.yaml` | 다른 모듈과 같은 실험 설정 |

기다리는 시간(접속 120초, 데이터 없음 10초), 상태 줄의 주기(1초), 한 세션의 프레임 상한(216,000프레임, 30 fps로 2시간)은
옵션이 아니라 `drowsy/live.py`의 상수다(11장).

**터미널에 나오는 것** (모두 영어. 오류와 모델 대체 알림만 표준 오류). 아래의 예는 2026-11-14 이 PC의 실행에서 옮겼다.

- 시작 블록: 도구 설명(Android에 관해 아무것도 검증하지 않는다는 문장), 모델(`... (version 0, trained_on: none (random weights)):
  the model output carries no meaning, only the rule events do`), 졸음 판정원(`drowsiness.source: perclos (drowsy events come from
  PERCLOS, not from the model)`), 소스와 기다리는 시간, 보내는 쪽의 명령 예와 카메라 이름을 보는 명령, 회전(`--rotation`을 줬는지),
  속도, 페이지 주소, 끝내는 법. 끝내는 법의 줄은 소스에 따라 다르다: 듣는 소스(기본값)는 `stop: press Ctrl+C in this
  terminal, or stop the sender. ...`, 영상 파일은 `... or wait for the source to end. ...`, 그 밖의 소스(`/dev/video` 장치,
  `listen=1`이 없는 `tcp://`·`udp://`)는 `... or stop the source. ...`다(4차 검증 뒤의 정비가 셋째 경우를 나눴다. 전에는 영상
  파일과 같은 문장이었다). 규칙 설정의 판정원을 `model`이나 `either`로 바꾸고 무작위 가중치 모델을 쓰면 졸음 이벤트에도 뜻이
  없다는 `WARNING`이 붙는다(기본 설정은 `perclos`).
- 접속과 첫 프레임: `source opened: the sender is connected and sends video; waiting for the first key frame`, 3초가 지나도
  프레임이 없으면 키프레임 간격(`-g 30`)을 보라는 한 줄, `first frame: 640x480; the session clock starts here`, 이어서 기준선
  안내 한 줄(`baseline: the first 60 s of the session are the reference for this driver ...`). 접속했다가 영상 없이 끊은 쪽(포트만
  확인한 접속 등)이 있으면 그 사실을 한 줄로 알리고 남은 시간 동안 다시 듣는다.
- 1초에 한 줄: `session 59.3 s, face yes, closure 0.00 (provisional), gaze unknown, baseline 59/60 s, received 30.0 fps,
  processed 30.0 fps, dropped 0, skipped 0`. 기준선이 완성되면 `closure 0.00, gaze forward, baseline ok (60 s)`처럼 바뀐다
  (`face`: 얼굴 검출, `closure`: 눈 감김 정도 0~1, `gaze`: 전방 주시 `forward`/`away`/`unknown`, `baseline`: 진행 또는 결과 `ok`,
  `off_axis`, `fallback_global`, `dropped`: 처리가 밀려 버린 프레임, `skipped`: 타임스탬프가 늘지 않아 버린 프레임).
- 이벤트와 경고는 날 때 한 줄씩: `event 73.1 s: microsleep onset (severity 3, started 72.1 s, duration 1.0 s, confidence 1.00,
  source closure)`, `alert 73.1 s: microsleep (immediate, severity 3, duration 1.0 s)`. 모델 출력은 터미널과 페이지에 보여 주지
  않는다(무작위 가중치. 이벤트 JSON과 그림에는 데모와 같이 들어간다).
- 끝: 멈추라는 요청이면 `stop requested: ending the session`, 그리고 `session ended: <이유>`, 프레임 계수 한 줄, `wrote
  out/live/live_<YYYYMMDD_HHMMSS>.parquet (<n> bytes)`, 데모 출력의 회전 줄이 이 세션에서 무슨 뜻인지 알리는 `replay: ...` 한
  줄, 그 뒤는 `python -m drowsy.demo --raw`의 출력 그대로(그림과 JSON의 `wrote` 줄 둘, 요약 줄)다. 2026-11-14에 4차 검증 뒤의
  정비가 PyAV 보내는 쪽으로 돌려 첫 프레임 6.0초 뒤에 Ctrl+C(SIGINT)로 끝낸 세션의 줄(작업자의 기록에서 옮김):
  `session ended: interrupted (Ctrl+C)`, `frames: decoded 182, skipped 0 (timestamp), dropped 0 (processing was behind), left 1
  (unprocessed at the end), failed 0 (processing error), processed 181, with a face 0`, 요약 줄 `frames 181, session 6.0 s,
  events 1, alerts 0, model outputs 0, drowsiness.source perclos`. 프레임 계수의 항등식은 decoded = skipped + dropped + left +
  failed + processed다. `failed`(4차 검증 뒤의 정비가 더했다)는 처리하다 오류가 난 프레임으로 0 또는 1이다(처리 오류는 세션을
  끝내므로). 그 프레임은 표에 들어가지 않는다. 전에는 그 프레임을 `left`(끝에 처리하지 못한 프레임)에 셌다.

**웹 페이지** (`http://localhost:8765/`. 표준 라이브러리의 HTTP 서버, 127.0.0.1에만 묶는다)

- 단계(기다림, 실행 중, 끝남), 세션 시간, 얼굴, closure, 전방 주시, 기준선, 초당 프레임 수, 버린 수, 카메라 미리보기, 최근
  이벤트와 경고 20개씩, 무엇을 돌리는지의 안내. 0.5초마다 `/status.json`을 읽어 다시 그린다. 미리보기는 `/preview.mjpg`(JPEG
  스트림, 긴 변 480 px)다.
- 미리보기는 **메모리에만** 있고 127.0.0.1에서만 내주며 디스크에 쓰지 않는다. `--no-preview`면 만들지 않는다(`/preview.mjpg`는 404).
- Host 머리말이 `localhost`, `127.0.0.1`, `[::1]`에 서버의 포트가 붙은 것이 아닌 요청은 403이다(브라우저에 열린 다른 웹 페이지가
  미리보기를 읽어 가는 길을 막는다). 프로그램이 답하지 않으면 페이지가 그렇게 알리고, 같은 포트로 다시 시작하면 스스로 이어 간다.
- 페이지는 실제 브라우저로 열어 보지 못했다. Windows PowerShell에서 `http://localhost:8765/`와 `/status.json`을 받아 200을 확인했고
  (2026-11-14에는 `127.0.0.1`이라는 이름으로도 페이지가 응답했다),
  페이지의 스크립트는 node에서 가짜 화면 객체로만 시험했다(검토와 수정 단계의 기록).

**끝에 쓰는 파일** (`--out` 아래, 줄기 `live_<YYYYMMDD_HHMMSS>`는 세션을 시작한 지역 시각. 같은 초에 시작한 두 세션은 덮어쓴다)

| 파일 | 내용 |
|------|------|
| `<줄기>.parquet` | 세션의 프레임 표. 추출 결과(4.5절)와 같은 형식이고 랜드마크 값만 담는다(화소 없음). 메타데이터에 `live: true`, `rotation_given`(`--rotation`을 줬는지), 프레임 계수(`n_decoded`, `n_skipped`, `n_dropped`, `n_left`, `n_failed`, `n_decode_errors`, `n_processed`, `n_face`. `n_decoded` = `n_skipped` + `n_dropped` + `n_left` + `n_failed` + `n_processed`. `n_failed`는 4차 검증 뒤에 더한 키다), `end_reason`이 있다 |
| `<줄기>_timeline.png` | 그 표에 `python -m drowsy.demo --raw`를 돌린 타임라인(4.12절) |
| `<줄기>_events.json` | 같은 실행의 이벤트 JSON. 터미널에 나온 이벤트와 경고가 그대로 들어 있다 |

- 같은 표를 나중에 `python -m drowsy.demo --raw out/live/<줄기>.parquet`로 다시 돌릴 수 있다. 실행에서는 다시 돌린 이벤트 JSON이
  세션의 것과 같았다(아래 "잰 것").
- 실시간 표에서 데모의 회전 줄 `rotation: N (manual value recorded by the extraction; ...)`과 `python -m drowsy.dataset`의 회전
  요약이 말하는 "수동 값"은 `--rotation`으로 준 값이거나 기본값 0이고, 아무것도 점검하지 않았다는 뜻이다. 줬는지는
  `rotation_given`에 있다. 실시간 표의 `n_dropped`는 처리가 밀려 버린 수이고, 추출 결과의 `n_dropped`(프레임 누락 모사)와 뜻이 다르다.
- 출력 폴더에 쓰는 것은 이 셋뿐이다. 다만 onnxruntime 라이브러리는 모델을 불러올 때 자기 파일(임시 폴더의 `.ses`,
  `~/.cache/Microsoft/DeveloperTools/.onnxruntime/` 아래의 작은 데이터베이스)을 남긴다. 화소는 들어 있지 않다. `python -m
  drowsy.demo`도 같다(최종 점검의 기록).

**끝나는 길과 종료 코드**

| 끝나는 길 | 결과 |
|-----------|------|
| Ctrl+C, `kill`(SIGTERM), 터미널 창을 닫음(SIGHUP), 보내는 쪽이 멈춤, 10초 동안 데이터 없음, 프레임 상한 | 프레임을 하나라도 처리했으면 세 파일을 쓰고 종료 코드 0. Ctrl+C를 여러 번 눌러도 같다. 창을 닫은 경우 파일은 다 쓰이지만 종료 코드는 뜻이 없다(가짜 터미널을 닫아 본 실행에서 120) |
| 세션 도중의 처리 오류(종류를 가리지 않는다) | 그때까지의 프레임으로 세 파일을 쓰고 종료 코드 2. 표준 오류의 줄은 `error: the session was ended by a processing error: <오류>`이고, 오류가 난 프레임은 `frames:` 줄의 `failed 1`로 센다. 미리보기나 페이지 상태를 만들다 난 오류는 한 줄로 알리고 세션을 계속한다 |
| 첫 프레임을 처리하기 전의 Ctrl+C, 보내는 쪽이 120초 안에 오지 않음, 쓸 수 없는 `--out`, 사용 중인 포트, 모델·설정의 오류 | 아무것도 쓰지 않고 종료 코드 2(오류 문장에 고칠 방법이 붙는다). 이 프로그램이 만든 출력 폴더는 지운다 |
| 보내는 쪽을 기다리는 동안의 `kill`(SIGTERM)이나 터미널 창 닫기(SIGHUP) | 파일은 쓰지 않지만 이 프로그램이 만든 **빈 출력 폴더가 남는다**(그런 신호에는 처리기를 두지 않기로 했다. 4차 검증 뒤의 정비가 `drowsy/live.py`의 설명에 이 예외를 적었다). 프로세스는 그 신호로 끝난다 |
| 시작하는 처음 2초쯤(라이브러리를 올리는 동안)의 Ctrl+C | Python의 `KeyboardInterrupt` traceback으로 끝난다(아직 아무것도 시작하지 않았다) |

**한계**

- 모델 `export/drowsiness_v0.onnx`는 무작위 가중치다. 그 출력은 판정이 아니고, 뜻이 있는 것은 규칙 이벤트뿐이다.
- 규칙의 임계값(감김 0.7·0.5, PERCLOS의 0.8 등, 전방 주시 영역)은 실제 눈과 시선으로 확인하지 못한 잠정값이다(7장). 이벤트가
  나지 않거나 예상과 다르게 나는 것이 이 도구의 결함이라는 뜻은 아니다.
- 그림은 H.264로 압축된 것이다(영상 파일에서 추출할 때와 같다). 압축하지 않은 카메라 프레임을 받는 앱과는 화소가 다르다.
  보내는 쪽의 타임스탬프가 실제 촬영 간격을 반영해야 한다(파이프라인의 시간은 모두 이 값에서 나온다).
- 데스크톱 시험 도구다. Android에 관해서는 아무것도 검증하지 않는다(LIVE_STREAM 모드, 기기의 프레임률, 지연 모두).
- **이 도구를 만든 사람들은 실제 카메라, Windows의 ffmpeg, 브라우저에 그려진 페이지, 실제 얼굴로 돌려 보지 못했다.** 아래의
  수치는 모두 이 PC 안에서 PyAV로 만든 보내는 쪽과 얼굴 없는 합성 그림으로 쟀다(얼굴이 있는 세션은 MediaPipe 대신 각본대로
  랜드마크를 내는 가짜 랜드마커로 흉내 냈다).
  얼굴이 있을 때의 MediaPipe 처리 시간, 200초보다 긴 실제 세션, `/dev/video` 소스도 확인하지 못했다.
- 화면의 상태는 실제보다 늦다(아래 "잰 것"의 중앙값 71.8 ms). 파이프라인의 시간은 스트림의 타임스탬프에서 나오므로 이 지연은 판정에
  들어가지 않는다. 받는 쪽은 버퍼링을 끄고 읽어 스트림 분석이 끝난 뒤의 첫 키프레임부터 보므로, 키프레임 간격이 30인
  스트림에서 처음 30프레임을 보지 못했다.
- 보내는 쪽이 접속한 채 MPEG-TS가 아닌 것을 계속 보내거나 아무것도 보내지 않으면, 이 프로그램은 120초까지 `still waiting`
  줄만 내고 시간 초과 오류로 끝난다. PyAV가 접속을 받은 때를 알려 주지 않아서 "아무도 오지 않음"과 "왔지만 영상이 아님"을
  가르지 못한다. 그래서 오류 문장이 두 경우를 함께 적고 `-f mpegts`를 알려 준다(4차 검증 뒤의 정비. 전에는 `no sender
  connected ...`라고만 적었다). 4차 검증 뒤의 정비가 접속 시간을 8초로 줄인 사본으로 확인했다: 접속한 채 아무것도 보내지 않는
  쪽과 글자를 보내는 쪽 모두 `(6 s left)`, `(4 s left)`, `(2 s left)` 줄 뒤에 그 오류 문장과 종료 코드 2로 끝났고 출력 폴더는
  남지 않았다.
- 보내는 쪽을 기다리는 동안 kill하거나 창을 닫으면 빈 출력 폴더가 남는다(위 "끝나는 길"의 표. 그 실행에서 폴더는 비어 있었다).

**잰 것** (2026-11-14, 이 PC, WSL2, localhost, PyAV 보내는 쪽 libx264 ultrafast·zerolatency·키프레임 30프레임마다, 얼굴 없는
640x480 그림 30 fps. 명령과 출처는 12.1절의 표): 200초 세션에서 6,000프레임을 처리하고 버린 프레임 0, 보내는 쪽이 프레임을
인코더에 넣은 때부터 처리가 끝날 때까지 중앙값 71.8 ms, 95 분위수 72.4 ms, 최대 87.6 ms, 프레임 하나의 처리(MediaPipe와
파이프라인, 얼굴 없음) 중앙값 1.33 ms, 최대 6.51 ms, 상주 메모리 시작 5초 뒤 793 MB, 끝(200초)에 805 MB. 세션 도중에
멈추라는 요청(Ctrl+C, SIGTERM, SIGHUP)부터 프로세스가 끝날 때까지(세 파일을 쓰는 시간 포함) 약 0.6~0.8초(다섯 번에
0.60~0.77초. 4차 검증 뒤의 정비가 가장 낮은 우선순위 `nice -n 19`로 돌린 Ctrl+C 한 번은 0.87초였으므로 0.8초를 넘을 수
있다), 보내는 쪽이 접속한 채 아무것도 보내지 않을 때의 Ctrl+C는 약 3초(2.72초와 2.92초), 첫 키프레임을 기다리는 동안은
0.51초(한 번). 얼굴 없는 그림에서는 기준선이 60초에 완성되지 않고 180초에 전역 상수로 대체됐다(`baseline fallback_global (180 s)`).
각본 얼굴로는 60.0초에 `calibrating end`, `baseline ok (60 s)`가 나왔다.

## 5. 저장소 구조

```
hawkeye/
  pyproject.toml, uv.lock      의존성 고정
  .gitignore, .gitattributes   git 제외 목록(일부러 넓게: config/local*, 어느 깊이의 videos/ 폴더, 대소문자를 가리지 않는 영상 확장자 다섯), 해시를 거는 파일의 줄바꿈 변환 금지, *.onnx와 *.task는 binary
  config/
    default.yaml               실험 설정: 경로, 추출, 창, 학습, LightGBM, 평가, 내보내기, 데모. 실제 데이터는 그 사본 local.yaml(git 제외)
    feature_spec.json          스텝 특징 17개의 정의와 순서, 이식 코드의 상수 전부(pipeline 블록), pipeline_hash
    rules.yaml                 규칙 엔진과 경고의 임계값(값마다 출처 주석)
  drowsy/
    types.py                   [이식] 자료형(RawFrame, FrameSignals, Baseline, FrameState, ClosureRun, StepFeatures, DriverStateEvent, PipelineOutput ...)
    signals.py                 [이식] EAR, 머리 자세, 시선 대용치
    calibrate.py               [이식] 기준선, closure
    blink.py                   [이식] 감김 검출기(ClosureRunDetector)
    step_features.py           [이식] 하품·끄덕임 검출, 1초 스텝 특징
    rules.py                   [이식] 규칙 엔진, 경고 결정
    pipeline.py                [이식] 위 모듈을 잇는 DriverPipeline
    spec.py, config.py         feature_spec·규칙·실험 설정의 로드, 검증, 해시
    video_io.py, extract.py    PyAV 디코딩과 회전 보정, 영상 → MediaPipe → RawFrame parquet, 프레임 누락 모사, 추출 리포트
    synth.py                   합성 세션 생성기(테스트, 벡터, 합성 데이터 전 과정)
    dataset.py                 영상 목록, 재생, 스텝 표와 캐시, 프레임 누락 변형, 창, 공식 fold, 피험자 분리 검사
    signal_check.py            E0 신호 점검과 상수 제안, 15 fps 깜빡임 길이 오차 모사
    models.py                  CausalTCN, GRUClassifier, ExportWrapper, 체크포인트(형식 v2)
    train.py, evaluate.py      학습(증강, 부가 실행)과 중첩 교차검증, 지표와 리포트, 라벨 없는 영상 평가
    baseline_lgbm.py           LightGBM 기준선, W 비교, 부가 실행, 교란 점검
    export_onnx.py             내보내기, 계약 검사, 일치 검증, 지연, INT8
    vectors.py                 테스트 벡터 생성과 재검증
    demo.py                    타임라인 데모
    live.py                    실시간 시험 도구(카메라 스트림 → MediaPipe → DriverPipeline, 터미널과 127.0.0.1의 웹 페이지, 끝에 프레임 표·타임라인·이벤트 JSON. 4.13절)
  tests/                       pytest(합성 데이터만, 누설 회귀 시험 test_leakage.py 포함), vectors/ 테스트 벡터 102개와 README
  docs/                        GUIDE.md(먼저 읽는 안내서), PLAN.md(계획서), SPEC.md(이식 명세), MODEL_CARD.md(모델 카드)
  export/                      앱에 넘기는 파일과 README
  models/     (git 제외)       face_landmarker.task
  data/       (git 제외)       raw/ 추출 결과, synth/ 합성 데이터, *_steps/ 스텝 표 캐시
  reports/    (git 제외)       실행별 리포트
  out/        (git 제외)       데모 출력, 합성 데이터로 학습한 모델
  videos/     (git 제외)       4.5, 4.12의 예시가 쓰는 영상 폴더(영상 확장자는 어느 폴더에서나 제외된다)
```

`[이식]`은 Android 팀이 Kotlin으로 옮기는 모듈이다. 표준 라이브러리만 쓴다.

## 6. 데이터에 대한 전제

이 프로젝트의 결과를 읽을 때 함께 알아야 하는 전제다.

1. **주행을 시작할 때 운전자는 각성 상태이고 전방을 본다.** 세션의 처음 60초로 개인 기준선(뜬 눈의 EAR과 `eyeBlink`,
   기준 머리 자세)을 잡고 이후의 모든 신호를 그 기준에 대어 본다. 처음부터 졸린 운전자, 처음 60초에 옆을 보고 있던
   운전자에게는 기준이 틀어진다. 얼굴 프레임이 모자라면 30초씩 최대 180초까지 연장하고, 그래도 안 되면 전역 상수로
   대체하며 그때는 시선 규칙을 끈다(D5).
2. **UTA-RLDD는 실내에서 화면을 보며 찍은 영상이고 운전 환경이 아니다.** 주행 중의 조명, 진동, 시선 이동, 거치 각도가
   없다. 여기서 학습한 모델이 차량으로 옮겨지지 않을 수 있다. 시선 규칙은 이 데이터로 검증할 수 없다.
3. **UTA-RLDD의 라벨은 본인 보고다.** alert는 KSS(졸음 자가 평가 척도) 1~3, low vigilant는 6~7, drowsy는 8~9이고 영상
   전체에 라벨 하나가 붙는다.
4. **UTA-RLDD에는 명시된 라이선스가 없다.** 논문 인용 조건과 얼굴 공개 제한만 있다. 사용 여부는 팀이 판단하고, 공모전
   규정이 학습 데이터 라이선스를 요구하는지 확인이 필요하다(PLAN 1.3절). 영상은 재배포하지 않는다.
5. **얼굴을 공개해도 되는 피험자는 60명 중 36명뿐이다.** 그래서 데모와 리포트는 신호와 타임라인만 그리고 얼굴 프레임을
   저장하지 않는다(D13). 추출 결과에도 이미지는 없고 랜드마크 좌표와 블렌드셰이프 값, 영상 요약(밝기 등)만 있다.
6. 앱이 MediaPipe에 넘기는 프레임(Q3): 비트맵을 똑바로 세워서 넘긴다(`rotationDegrees` 옵션을 쓰지 않는다), 좌우
   반전하지 않는다, `numFaces = 1`, 타임스탬프는 양수이고 엄격히 증가하는 실제 ms다. 학습 특징도 이 전제로 뽑는다.
7. 합성 데이터의 상태별 수치는 문헌의 방향만 따른 설계값이다. 실제 사람의 분포를 재현하지 않는다.
8. closure 상수와 전역 상수는 E0에서 라벨 없이, 시험 피험자의 영상을 포함한 전체 영상으로 한 번 정한다. 교차검증
   리포트의 설명에 이 예외가 적힌다(PLAN 4.3절).

UTA-RLDD에 대해 계획 단계에서 조사한 내용(공식 5-fold, 파일 182개, 가변 프레임률, 세션 교란 위험)은 PLAN 2.4절에 있다.
영상 자체는 아직 보지 못했다.

## 7. 잠정 상수

"잠정"은 공개 근거가 없거나 실제 눈 감김으로 확인하지 못한 값이다. 실제 데이터를 본 뒤 한 번 확정한다(PLAN 11장 E0,
4.6절의 명령. 라벨을 쓰지 않는 점검이다). 그때 `pipeline_hash`가 바뀌므로 벡터를 다시 만들고 앱 팀에 알린다.
정비에서 값은 하나도 바뀌지 않았다(`pipeline_hash` 650495e5…, `rules_hash` 9c2d05f2…).

| 상수 | 지금 값 | 있는 곳 | 상태 |
|------|---------|---------|------|
| closure의 `rho`(EAR이 뜬 눈 값의 이 비율까지 내려가면 완전히 감긴 것으로 본다) | 0.35 | `feature_spec.json` `pipeline.closure` | 잠정 |
| closure의 `b_closed`(`eyeBlink`가 이 값에 닿으면 완전히 감긴 것으로 본다) | 0.60 | 같은 곳 | 잠정 |
| closure의 `w`(EAR 쪽과 블렌드셰이프 쪽의 비중) | 0.5 | 같은 곳 | v1에서는 바꾸지 않는다. 0, 0.5, 1의 비교는 보고용(`baseline_lgbm cv --all-w`) |
| 먼 쪽 눈의 비중 | 상대 yaw 15도까지 두 눈 평균, 25도부터 가까운 눈만 | 같은 곳 | 설계값(4.3절의 제안) |
| closure 유효 범위 | 상대 yaw 절댓값 40도 이하 | 같은 곳 | 설계값 |
| 특징용 감김 검출 임계값 | 진입 0.5 / 이탈 0.3 | `pipeline.run_detector` | 잠정 |
| 규칙용 감김 검출 임계값 | 진입 0.7 / 이탈 0.5 | `rules.yaml` `closure.detector` | 잠정 |
| 감김 검출기의 시간 상수 | `max_dt_ms` 200 ≤ `hold_ms` 500 < `clock.reset_gap_ms` 10,000. 설정을 읽을 때 두 검출기 모두에 이 순서를 검사하고, 어기면 설정 오류다(`hold_ms < reset_gap_ms`는 3차 검증 뒤의 정비가 더했다. 10초 넘는 끊김이 늘 진행 중인 감김도 끊게 하려는 것이다. 지금 값은 통과하므로 해시는 그대로다) | `pipeline.run_detector`, `rules.yaml` `closure.detector`, `clock` | 설계값 |
| PERCLOS의 감김 기준 | closure 0.8 이상 | `pipeline.step_features.perclos_thr` | 잠정 |
| 전역 상수 `E`, `B`, `jaw0` | 0.27, 0.22, 0.08 | `pipeline.global_constants` | spec v0의 설계값. E0에서 세션별 통계의 중앙값으로 한 번 바꾼다 |
| 하품 | `jaw_open − jaw0` 0.40 이상으로 시작, 0.25 미만으로 끝, 2,000 ms에 한 번 센다 | `pipeline.yawn` | 잠정 |
| 끄덕임 | 느린 평균 대비 pitch 10도 이상으로 시작, 5도 미만으로 끝, 200~3,000 ms, 평균의 시상수 10초 | `pipeline.nod` | 잠정 |
| 하품·끄덕임 판정을 취소하는 끊김 | `yawn.cancel_gap_ms` 500, `nod.cancel_gap_ms` 500 < `clock.reset_gap_ms` 10,000. 설정을 읽을 때 둘 다 검사하고, 어기면 두 값을 적은 설정 오류다(4차 검증 뒤의 정비가 더했다. 감김 검출기의 `hold_ms`와 같은 까닭이다: 10초 넘는 끊김이 늘 진행 중인 하품·끄덕임 판정도 취소하게 한다. 지금 값은 통과하므로 해시는 그대로다) | `pipeline.yawn`, `pipeline.nod`, `clock` | 값은 잠정, 부등식은 설계값 |
| 전방 주시 영역 | 상대 yaw 절댓값 23도 이하, pitch 위 25도 ~ 아래 18도 | `rules.yaml` `forward_zone` | 잠정. 검증된 공개 값이 없다(openpilot은 yaw 약 23도, 아래 약 18도) |
| 눈동자 보조 판정 | `gaze_h` 변화 0.25, `gaze_v` 변화 0.35 초과 | 같은 곳 | 잠정(D12). 공개 근거 없음 |
| 전방 주시 확정 시간, 얼굴 소실 인정 시간 | 250 ms, 3,000 ms | 같은 곳 | 잠정(설계값) |
| PERCLOS 졸음 판정 | `perclos_60` 0.15 이상이 10스텝 연속이면 시작, 0.10 미만이 10스텝 연속이면 종료 | `rules.yaml` `drowsiness.perclos` | 잠정 |
| 분석 불가 | 최근 10초의 얼굴 검출 비율 0.5 미만 | `rules.yaml` `unable_to_analyze` | 잠정(설계값) |
| 이벤트별 심각도, 재경고 대기, 되풀이 주기 | `rules.yaml`의 `events` 표 | `rules.yaml` `events` | 설계값(공개 근거 없음) |
| 기준선 완성 조건 | 60초, 얼굴 프레임 300개, 얼굴 비율 0.5, `E`가 0.12~0.45, 30초씩 최대 180초까지 연장 | `pipeline.baseline` | 설계값(D5) |
| 기기 프레임 누락 모사의 처리 지연 | 평균 66.7 ms, 표준편차 8.0 ms | `default.yaml` `extraction.drop_sim` | 가정. 실기기 로그를 받으면 바꾼다 |
| 회전 점검 | 12장, 2초 간격, roll 여유 15도 | `default.yaml` `extraction` | 가정 |
| 학습 출처의 몫 | 원본 0.5, 15 fps 모사 0.25, 프레임 누락 변형 0.25(변형 1개) | `default.yaml` `training` | 가정(계획서에 없는 값) |
| 순서형 소프트 라벨의 이웃 몫 | 0.2 | `default.yaml` `training.ordinal_neighbor_mass` | 가정(부가 실행만 쓴다) |

잠정이 아닌 값(명세와 Euro NCAP 2026 프로토콜에서 온 것): 마이크로슬립 1.0초, 수면 3.0초 이상, 장시간 시선 이탈 3.0초,
VATS 30초 중 누적 10초, 경고의 속도 기준 20 km/h(졸음은 50 km/h). 모델 졸음 판정의 임계값(`p_drowsy` 0.6 이상이 10스텝
연속이면 시작, 0.4 이하가 10스텝 연속이면 종료, 종료 후 60초는 다시 시작하지 않음)은 사전에 고정한 값이고 시험 fold로
조정하지 않는다.

## 8. 소유자 결정 (2026-11-13, 2026-11-14)

검토에서 나온 동작 질문 네 가지를 소유자가 2026-11-13에 정했고, 이식 코드에 그대로 들어갔다. 작업 기록에서는 D1~D4로
불렀지만 계획서 1장의 D 번호와 겹치므로 여기서는 1~4로 적는다. 네 가지 모두 설정 값을 바꾸지 않았다(`pipeline_hash`,
`rules_hash`, `export/drowsiness_v0.onnx`의 sha256은 그대로다). `docs/SPEC.md`(10, 12, 13, 7장)와 테스트 벡터가 이 동작을
담고 있다. 아래 "확인"은 정비의 통합 단계가 실제 `RuleEngine`과 `DriverPipeline`으로 돌린 각본의 결과다(합성 신호, 30 fps).
2026-11-14에는 9장의 첫 질문(결정 1 뒤에 남은 반쯤 뜬 눈꺼풀의 질문)에 소유자가 "지금대로 둠"이라고 답했다. 그것이 이 장
끝의 결정 5이고, 코드도 벡터도 해시도 바뀌지 않았다.

### 결정 1: 감김 길이는 0.7(진입 임계값) 이상이었던 시간만 더한다

- **전**: 감김 구간의 길이(`ClosureRun.duration_ms`, 진행 중인 길이 `ongoing_ms()`)는 처음 진입한 시각부터 마지막으로 진입
  임계값 아래로 내려간 시각까지였다. 눈꺼풀이 이탈과 진입 사이(규칙용 0.5~0.7, 특징용 0.3~0.5)에 머문 시간도 길이에
  들어갔다. 검토의 재현에서는 프레임 하나 때문에 `ongoing_ms`가 305 ms에서 5,354 ms로 뛰어 마이크로슬립과 수면이 한
  프레임에 함께 시작됐다.
- **후**: 구간은 진입 임계값 이상인 "감긴 조각" 하나 이상으로 이뤄진다. 길이는 조각들의 합이다. 반쯤 뜬 구간에서는
  `ongoing_ms()`가 늘지 않고, 구간 안에서 줄지도 않는다. 조각이 하나뿐인 구간은 전과 비트 단위로 같다. 같은 재현에서
  이제 이벤트가 없고 구간 길이는 423 ms(규칙용)다.
- **확인**: 0.9로 300 ms, 0.6으로 5초, 0.9로 300 ms 감은 각본은 마이크로슬립·수면 없이 구간 하나(길이 592 ms)를 냈다. 둘째
  감김을 800 ms로 늘리면 누적이 1,000 ms에 닿은 첫 감긴 프레임(누적 1,018 ms)에서 마이크로슬립이 시작됐다.
- **앱 개발자가 알 것**
  - 검출기에 상태 `sealed_ms`(끝난 조각의 합), `span_start_ms`(지금 조각의 시작)와 `run_start_ms()`가 더해졌다. 규칙 엔진은
    구간 시작을 `t − ongoing_ms`가 아니라 `run_start_ms()`로 얻는다(그 등식이 더는 성립하지 않는다).
  - `duration_ms ≤ end_ms − start_ms`이고 조각이 하나일 때만 같다. `min_duration_ms`(50)는 합에 적용한다.
  - 반쯤 뜬 구간 양쪽의 두 깜빡임은 여전히 한 구간이다. `blink_rate_60`은 하나로 세고 `blink_dur_mean_60`에는 두 조각의
    합이 들어간다.
  - 마이크로슬립·수면은 여전히 감긴 프레임에서 `ongoing_ms() ≥ 1000 / 3000`으로 시작한다. `t_start_ms`는 구간의
    `start_ms`이고 `duration_ms`는 발행 시각 − `t_start_ms`라서 시작을 일으킨 누적 시간보다 길 수 있다.
  - `ongoing_ms`는 한 프레임에 최대 `hold_ms`(500 ms)만 늘어나므로 기본 임계값에서 두 시작이 한 프레임에 함께 나오지 않는다.
  - 눈꺼풀이 이탈 임계값 아래로 내려가지 않는 한 누적에 시간 제한이 없다. 소유자가 2026-11-14에 이 동작을 그대로 두기로
    정했다(이 장 끝의 결정 5. 잰 사례와 고르지 않은 대안도 거기에 있다).
  - 특징 #4~#11이 조각이 여럿인 구간에서 바뀐다. 정비 전에 만든 스텝 표 캐시는 다시 만든다(캐시의 코드 지문이 알아서 한다).

### 결정 2: 막힌 경고는 막힘이 풀렸을 때 상태가 계속 중이면 한 번 낸다

- **전**: 재경고 대기나 속도 억제에 걸린 시작(`onset`)은 그 호출에서만 검사했고, 되풀이하지 않는 종류는 나중에 경고하지
  않았다(되풀이하는 `sleep`만 뒤늦게 경고했다).
- **후**: `AlertPolicy`가 막힌 시작을 종류마다 "대기 중"으로 둔다. 뒤의 호출에서 그 종류가 허용되면 시작 이벤트의 사본을
  `duration_ms = t − t_start_ms`로 한 번 경고한다. 그 종류의 `end`가 오면 경고 없이 버린다. `microsleep`의 대기는
  `sleep`의 시작이나 끝에서도 버린다. 막힌 시작은 여전히 재경고 대기를 시작하지 않는다.
- **확인**: 45 km/h에서 시작한 졸음 상태가 60초 뒤 100 km/h가 되자 그 첫 프레임에서 경고 하나가 나갔다. 속도가 오르기 전에
  상태가 끝나면 경고가 없었다. 10 km/h에서 막힌 마이크로슬립이 수면으로 자라고 감김이 끝난 뒤 80 km/h가 되어도 늦은
  마이크로슬립 경고는 없었다.
- **앱 개발자가 알 것**
  - 되풀이하지 않는 종류는 시작 하나에 경고가 최대 한 번(즉시 또는 나중)이다. **졸음 상태가 한 시간 이어져도 경고는 한 번**이다.
    주기적인 재경고는 더하지 않았다. 되풀이하는 종류(`sleep`, 3초마다)는 그대로다.
  - 한 호출의 경고(즉시, 나중, 되풀이)는 `alert_priority` 순서이고, 같은 종류 안에서는 즉시, 나중, 되풀이 순이다.
  - 나중에 나가는 경고는 시작 이벤트의 `confidence`와 `provisional`을 그대로 가진다.
  - 결과로: 재경고 대기 30초 안에 시작한 `unable_to_analyze`는 대기가 끝날 때 상태가 이어지면 경고된다. 시선·VATS 상태는
    분석 불가 동안에도 열려 있으므로, 얼굴이 보이지 않는 동안 그 경고가 나갈 수 있다. 미뤄 둔 경고는 상태가 끝날 때까지
    남으므로 운전자가 앞을 본 지 한참 뒤에 나갈 수도 있다(9장 8번에서 확인을 기다린다).
  - `speed_kmh = None`은 전처럼 막지 않는다.
  - 데모와 라벨 없는 영상 리포트는 경고를 바로 낸 것, 나중에 낸 것, 되풀이로 나눠 적는다(4.11, 4.12).

### 결정 3: 프레임이 끊겼다 와서 스텝이 여러 개 나오면 스텝마다 모델을 부른다

- **전**: 한 프레임이 스텝 n개를 내면 모두 창에 넣은 뒤 마지막 스텝에서 모델을 한 번만 불렀다. 앞 스텝들은 확률이 없어
  모델 판정의 연속 수가 끊겼다(9초마다 2.1초 끊기면 졸음 시작이 나오지 않았다). 오프라인 오경보 평가는 모든 창에
  확률을 줘서 앱과 달랐다.
- **후**: 스텝마다 창에 넣고, 창이 120스텝으로 차 있으면 그 시점의 창으로 모델을 불러 그 스텝의 확률을
  `RuleEngine.on_step`에 넘긴다. `PipelineOutput`에 `step_probs`(스텝마다 확률 또는 없음)가 더해졌고 `probs`는 그 마지막
  원소다. 앱과 오프라인 평가가 같은 스텝에 확률을 준다.
- **확인**: 9초마다 2.1초 끊기는 각본(10 fps, 모델 확률 0.9 고정)에서 졸음 시작이 끊김 없는 각본과 같은 프레임에 나왔고,
  확률이 0.1로 내려가자 종료도 같은 프레임에 나왔다.
- **앱 개발자가 알 것**: 최대 10초 끊긴 뒤의 프레임 하나가 추론을 최대 10번 한다. `model_calls`와 유한하지 않은 입력의
  수는 스텝마다 센다. `RuleEngine.on_step`과 `StepHysteresis`(연속 스텝 수로 시작과 끝을 정하는 판정기)는 그대로이고,
  확률이 없는 스텝(창이 덜 참, 유한하지 않은 입력, 모델 없음)은 전처럼 연속 수를 끊는다. 데모는 모델 호출마다 점 하나를
  그린다.

### 결정 4: 10초 넘게 끊긴 뒤에는 규칙 엔진이 세션 시작 때처럼 다시 시작한다

- **전**: 10초(`clock.reset_gap_ms`) 넘는 끊김 뒤에는 전방 주시 영역, VATS 합, 감김 검출기만 초기화했다. 분석 불가 창은
  그대로 흘러가서 끊긴 직후 얼굴 시간이 0이라 `unable_to_analyze`가 바로 시작됐고, 그동안 감김과 시선 규칙이 멈췄다
  (12초 끊긴 뒤 눈을 감고 있어도 마이크로슬립이 늦었다). 졸음 상태와 두 판정기도 그대로였다.
- **후**: 그런 끊김 뒤 첫 프레임에서, 진행 중이던 상태를 이 순서로 끝낸다: `drowsy`(시작과 같은 `t_start_ms`와 `source`,
  발행 시각은 그 프레임, `confidence` 0.0), `unable_to_analyze`, `gaze_away_long`, `microsleep` 또는 `sleep`. 두
  `StepHysteresis`(모델, PERCLOS)를 처음 상태로 되돌리고 얼굴 창을 비우며, "세션 처음 10초는 분석 불가를 판정하지
  않는다"는 보호를 그 프레임부터 다시 센다. 그래서 감김과 시선 규칙이 바로 돈다. 보정 상태(`calibrating`)와 기준선은
  그대로다.
- **확인**: 12초 끊긴 뒤 처음부터 눈을 감은 각본(80 km/h)은 분석 불가 없이 +1,000 ms에 마이크로슬립, +3,000 ms에 수면을
  경고와 함께 냈다(새 세션과 같다). 20분 끊김 전에 열린 졸음 상태는 끊김 뒤 첫 프레임에서 끝났고(`confidence` 0.0) 그 뒤
  새 일화가 시작되어 경고됐다.
- **앱 개발자가 알 것**
  - **끊김이 약 5.2초에서 10초 사이이면 여전히 `unable_to_analyze`가 첫 프레임에서 시작하고 경고도 나간다**(데이터 없는
    시간을 얼굴 없는 시간으로 센다. PLAN 8.1절). 경계는 테스트가 고정한다: 5,200 ms 이벤트 없음, 5,201 ms 시작,
    10,000 ms 시작, 10,001 ms 재시작. 이 구간도 면제할지는 9장 6번의 질문이다.
  - 그 프레임의 이벤트 순서: 재시작이 끝낸 상태들(위 순서), 보정 종료, 분석 불가, 시선, VATS(비운 합에서 나오는 끝),
    감김. 그 프레임에서 확률을 가진 스텝은 나오지 않는다.
  - 졸음 상태가 끊김으로 끝나고 다시 시작하면 새 일화로 다시 경고할 수 있다. **한 졸음 일화 안에서는 여전히 경고가 한
    번**이다(결정 2).
  - 오프라인 오경보 평가(`evaluate.alarm_videos`)도 한 세션 안에서 스텝 번호가 건너뛰는 곳(10초 넘는 끊김 뒤에만 생긴다)에서
    판정을 새로 시작한다. 그래서 모델 판정은 앱과 같다. 구현하면서 정한 것이라 10장에 적었다.

### 결정 5 (2026-11-14): 반쯤 뜬 눈꺼풀은 감김을 끝내지 않는다 ("지금대로 둠")

- **질문**: 9장의 1번이던 질문이다(결정 1 뒤에 남은 것. 9장에서는 그 번호를 비워 두었다). 결정 1로 감김의 길이는 진입
  임계값(규칙용 0.7, 특징용 0.5) 이상이던 시간만 더하지만, 구간 자체는 closure가 이탈 임계값(규칙용 0.5, 특징용 0.3) 아래로
  내려가야(또는 끊겨서 잘려야) 끝난다. 그래서 눈꺼풀이 두 임계값 사이에 머무는 동안에는 보통의 깜빡임 여러 개가 한 감김
  구간으로 이어지고, 진입 임계값 이상이던 시간이 그 구간의 길이에 계속 쌓인다. 이것에 끝을 둘지를 물었다.
- **소유자의 답**: "지금대로 둠". 동작, 코드, 벡터, `pipeline_hash`, `rules_hash`가 모두 그대로다. **옮긴 코드도 이 누적을
  그대로 재현해야 한다**(`docs/SPEC.md` 10.5절 7번, 12.5절, 결정 기록은 SPEC 20장 머리의 "2026-11-14에 결정되어 이 목록에서
  빠진 것").
- **알려진 한계와 잰 사례**(합성 입력, 규칙 엔진): 눈꺼풀이 0.6에 머문 채 4초마다 보통의 깜빡임(0.9로 165 ms)을 하면
  마이크로슬립이 24.1초(여섯째 깜빡임), 수면이 72.0초(열여덟째 깜빡임)에 난다. 프레임 시각을 33 ms 간격으로 둔 값이고, 정확한
  30 fps 시각으로 두면 수면이 68.1초(열일곱째 깜빡임)에 난다. 결정 1 전에는 둘 다 둘째 깜빡임인 8.0초에 났다. 수면이 시작된
  뒤에는 눈꺼풀이 머무는 동안 수면 경고가 3초마다 되풀이된다(33 ms 시계의 3분 각본에서 36번). 스텝 특징에서는 그 깜빡임들이
  한 구간이라 `blink_rate_60`이 1을 센다. 이 사례는 단위 테스트
  `tests/test_rules.py::test_decided_case_of_lids_hovering_between_the_two_thresholds`가 고정하고, 2026-11-14의 통합 단계와 4차 검증
  뒤의 통합 단계(2026-11-15)가 같은 각본을 다시 돌려 같은 값(24.1초, 72.0초, 36번, 68.1초)을 얻었다.
- **고르지 않은 대안**: closure가 진입 임계값 아래에 `hold_ms`(500 ms)보다 오래 머물면 구간을 끝내는 안이었다(새 상수가
  없어 해시가 그대로인 안).
- **벡터**: 그 대안으로 바뀌었을 벡터 여섯(`runs_lingering_lids`, `runs_multi_span`, `runs_multi_span_truncated`,
  `steps_multi_span_runs`, `rules_closure_dwell_below_1s`, `rules_closure_dwell_reaches_1s`. `runs_multi_span_truncated`는
  대안을 어떻게 정의하느냐에 따라)은 이제 결정된 동작을 고정한다(SPEC 16.1절, `tests/vectors/README.md`).
- **앱 개발자가 알 것**: 반쯤 뜬 눈의 운전자에게는 짧은 깜빡임이 모여 마이크로슬립·수면 경고가 날 수 있다. 앱에서 따로
  거르지 않는다(거르면 벡터와 다르게 동작한다).

## 9. 소유자 결정 대기

아래 16가지 동작(2~17번)은 지금 그대로 두고 소유자의 답을 기다린다(2026-11-15, 4차 검증 뒤의 정비가 끝난 상태의 목록.
질문과 번호는 3차 검증 뒤와 같고, 4차 검증 뒤에는 9번·12번의 벡터 수와 17번의 "지금 동작"만 고쳤다. 3차 검증 뒤에는 9번과
12번의 문장과 3번의 범위가 2차 검증 뒤와 달라졌다). **1번은 2026-11-14에 소유자가 결정해 비워 두었다**(결정 기록은
8장의 결정 5와 `docs/SPEC.md` 20장 머리의 "2026-11-14에 결정되어 이 목록에서 빠진 것"). 다른 항목은 번호를 바꾸지 않으므로,
문서와 작업 기록에 적힌 "9장 N번"은 앞으로도 같은 질문을 가리킨다. 이 목록 하나를 세 문서가 함께 쓴다: 2~12번은
`docs/SPEC.md` 20장의 2~12번과 번호와 내용이 같고(이식 코드와 앱이 받는 파일에 걸린 질문), 13~17번은 학습, 평가, 영상 추출
쪽 질문이라 이식 명세에는 없다. 계획서 14장 표 (b)는 같은 16개를 같은 번호로 싣고, 답이 지금 동작과 다를 때 다시 만들어야
하는 것을 적는다. 동작을 바꾸는 질문(2~9번)의 "다른 안"은 SPEC 20장의 같은 번호에 적힌 "제안된 대안"과 같은 내용이다.
"추천"은 정비의 문서 단계가 근거와 함께 적은 의견이고, 답이 오기 전에는 어느 것도 코드에 들어가지 않는다. "잰 사례"는 합성
입력으로 참조 구현을 돌린 결과다.

2. **무효 프레임 뒤 첫 유효 프레임에서 눈을 뜬 깜빡임을 보통 깜빡임으로 칠까**
   - 영향: 얼굴 검출이 깜빡이는 세션에서 깜빡임 수·길이·속도 특징(#4~#8)이 실제보다 적게 잡힌다.
   - 지금 동작: 감김 검출기가 판정할 수 없는 프레임 때문에 구간을 보류하는 상태(`HOLD`)에서 돌아온 표본이 이탈 임계값
     미만이면, 끊긴 시간이 짧아도 그 깜빡임은 잘린 구간(`truncated`)이 되어 깜빡임 특징에서 빠진다. 같은 자리의 프레임이
     아예 빠졌을 때는 보통 구간이다.
   - 다른 안: 마지막 유효 표본에서 `max_dt_ms`(200 ms) 안에 온 열린 프레임이면 프레임이 빠진 경우처럼 보통 구간으로 끝낸다.
   - 추천: 다른 안. 같은 장면이 프레임이 무효였는지 빠졌는지에 따라 다르게 세지지 않게 한다.
3. **감긴 모습을 보지 못한 구간도 감는 속도를 가질까**(`velocity_valid`에 쓸 수 있는 진입 쌍을 요구할까)
   - 영향: 구간에 들어선 두 표본(진입 쌍)이 끊김 너머여도 구간 안의 상승으로 감는 속도를 재므로, 잡음 크기의 값이 특징 #7,
     #8에 들어간다.
   - 지금 동작: 진입 쌍을 쓸 수 없어도 구간 안에서 정점까지 오른 쌍이 있으면 그 증가율이 `peak_closing_velocity`가 되고
     `velocity_valid`가 참이다. (감는 속도의 되돌아보기를 쌍의 간격으로만 끊는 것은 이 질문이 아니라 구현하면서 정한 동작이다.
     이 장 끝의 "정비에서 정한 것"과 10장)
   - 잰 사례(검토의 측정): 진입 쌍을 쓸 수 없는 구간 269개 중 33개가 유효였고 중앙값은 2.00/s(대개 잡음 크기)였다.
   - 다른 안: 진입 쌍을 쓸 수 없으면 `velocity_valid = false`.
   - 추천: 다른 안. 보지 못한 감김의 속도를 특징에 넣지 않는다.
4. **기준선에 두 눈이 다 필요한가, 눈마다 최소 표본 수를 둘까**
   - 영향: 한쪽 눈이 기준선 수집 내내 쓸 수 없으면 180초 뒤 전역 상수로 대체되고, 쓸 수 있는 표본이 몇 개뿐이면 그 몇 개가
     그 눈의 뜬 눈 값(E)을 정해 closure가 치우친다.
   - 지금 동작: 두 눈 모두 쓸 수 있는 표본이 하나 이상 있어야 기준선이 완성되고, E는 표본 하나로도 정해진다.
   - 다른 안: 한 눈으로도 완성하거나, 눈마다 최소 표본 수를 둔다(기존 `baseline.min_face_frames`를 쓰면 새 상수가 없다. 새
     상수를 두면 `pipeline_hash`가 바뀐다).
   - 추천: 지금 동작을 두고, 실제 데이터의 E0 점검(기준선 완성 비율)을 본 뒤 정한다. 그 전에는 얼마나 자주 일어나는지 모른다.
5. **250 ms 미만으로 본 관측을 얼굴 소실 인정 시간만으로 확정할까**
   - 영향: 얼굴이 가려지기 직전의 잡음 프레임 하나가 전방 주시 영역 밖이면, 3.5초 소실 뒤 `gaze_away_long`이 나고 경고될 수
     있다(`confidence`는 0에 가깝다).
   - 지금 동작: 확정한다. 영역 밖(또는 안) 관측은 250 ms 확정 시간을 얼굴 소실 인정 시간(최대 3초)으로 채울 수 있다.
   - 다른 안: 확정에 실제로 관측한 250 ms를 요구한다. 아주 빠른 고개 돌림을 놓친다.
   - 추천: 지금 동작. 놓치는 쪽보다 낫고, 그런 이벤트는 `confidence`가 0에 가까워 앱이 구분할 수 있다.
6. **5.2초~10초 끊김도 `unable_to_analyze`(분석 불가)에서 뺄까**(소유자 결정 4 뒤에 남은 질문)
   - 영향: 끊기기 전에 얼굴이 계속 보였어도 끊김 뒤 첫 프레임에서 분석 불가가 경고와 함께 시작되고, 얼굴 시간이 5초 쌓일
     때까지 시선·감김 규칙이 멈춘다.
   - 지금 동작: 10초 넘는 끊김만 세션 시작처럼 다시 시작한다(결정 4). 10초 이하의 끊김은 데이터 없는 시간을 얼굴 없는
     시간으로 센다(PLAN 8.1절).
   - 잰 사례(30 fps): 30초 동안 눈을 뜨고 있다가 8초 끊기고, 끊김 뒤 첫 프레임부터 5초 동안 눈을 감는다. 첫 프레임에서 분석
     불가가 경고와 함께 시작되어 5,000 ms 뒤에 끝났고, 그 5초 동안 마이크로슬립도 수면도 나지 않았다. 같은 장면에서 끊김이
     12초이면 1,000 ms에 마이크로슬립, 3,000 ms에 수면이 났다.
   - 다른 안: 이 범위의 끊김 뒤에도 재시작하거나, 데이터 없는 시간을 얼굴 없는 시간으로 세지 않는다.
   - 추천: 다른 안. 잰 사례에서 끊김 직후 5초 동안 감은 눈을 놓친다.
7. **`unable_to_analyze`와 `vats`에 끝 여유(히스테리시스)를 둘까**
   - 영향: 시작과 끝이 같은 임계값이라 그 근처에서는 분당 수십 쌍의 시작·끝이 나고, 그동안 시선·감김 규칙이 절반쯤 멈춰
     마이크로슬립이 나뉘거나 빠질 수 있다. 경고는 재경고 대기가 막는다.
   - 지금 동작: 얼굴 비율 0.5, 이탈 누적 10초에서 시작하고 같은 값에서 끝난다.
   - 다른 안: 끝 임계값을 따로 둔다(예: 얼굴 비율 0.6 이상, 누적 9초 미만). `config/rules.yaml`의 새 값이라 `rules_hash`,
     `export/rules.json`, rules 벡터가 바뀐다.
   - 추천: 다른 안. 해시가 바뀌므로 E0 뒤 상수를 확정할 때 함께 바꾼다.
8. **소유자 결정 2와 3의 귀결을 그대로 둘까**(확인)
   - 영향과 지금 동작: (a) 재경고 대기 30초 안에 시작한 `unable_to_analyze`는 대기가 끝날 때 상태가 이어지면 경고된다.
     (b) 미뤘다 내는 `gaze_away_long`·`vats` 경고는 얼굴이 보이지 않는 프레임에서 나갈 수 있다. (c) 미뤘다 내는 경고의
     `confidence`와 `provisional`은 시작 때의 값이다. (d) 끊겼다 온 프레임 하나가 추론을 최대 10번 한다. (e) 미뤄 둔 경고는
     그 상태가 끝날 때까지 남으므로 운전자가 앞을 본 지 한참 뒤에 나갈 수 있다. (f) 정확히 10,000 ms의 끊김은 재시작이 아니므로,
     끊기기 전에 속도에 막혀 있던 `gaze_away_long`·`vats`·`drowsy` 경고가 끊김 뒤 첫 프레임에서, 운전자가 앞을 보고 있어도,
     `unable_to_analyze` 경고와 함께 나갈 수 있다.
   - 잰 사례 (e)(10 fps): 10 km/h에서 1.0~12.0초 동안 옆을 보고 12.0초부터 앞을 본다. `vats`는 11.0초에 시작해 이탈 누적이
     30초 창에서 빠지는 32.1초까지 이어진다. 25.0초에 속도가 30 km/h가 되자 그 프레임에서 `vats` 경고가 나갔다. 앞을 본 지
     13초 뒤다.
   - 다른 안((e), (f)에 대해): 전방 주시가 확정되면 미뤄 둔 `vats`·`gaze_away_long` 경고를 지우거나, 조건이 그 프레임에도
     성립할 때만 미뤄 둔 경고를 낸다.
   - (a)~(d)에는 제안된 대안이 없다. 확인을 받지 못할 때 바꿀 수 있는 방향으로 문서 단계가 적어 둔 것만 있다: (a)는 그 종류를
     미루지 않기, (b)는 얼굴이 보일 때만 내기, (c)는 발행 때의 값으로 다시 계산하기, (d)는 한 프레임의 추론 횟수에 상한 두기.
   - 추천: (a)~(d)는 그대로 확인하고, (e)와 (f)는 "조건이 그 프레임에도 성립할 때만 낸다"로 바꾼다. 앞을 보고 있는 운전자에게
     나가는 시선 경고는 오경보로 읽힌다.
9. **얼굴 소실 인정 구간 뒤의 `gaze_away_long` 조건에 구현이 더한 것을 그대로 둘까**
   - 영향: 인정 구간이 끝나는 시각과 1 ms 차이로 시선 이탈 이벤트가 나거나 나지 않는 경계가 있다.
   - 지금 동작: 인정 구간에서 기한이 된 시선 이탈은 다음 분석 가능 프레임에서 내되, 표지가 "알 수 없음"인 프레임에서는
     전방 주시 후보가 확정 전이면 내지 않는다(`!candidate_pending`). 검토의 고침 계획에는 없던, 구현이 더한 조건이다. 이 조건이
     없으면 0.4초 곁눈질 뒤 앞을 본 프레임 하나, 그리고 정확히 3,200 ms 뒤의 얼굴 없는 프레임에서 잘못된 시작이 난다.
   - 잰 사례(10 fps): 1,000~3,900 ms에 영역 밖(3초가 4,000 ms에 찬다), 4,000 ms에 앞을 본 프레임 하나, 다음 프레임이 얼굴
     없이 정확히 7,200 ms(인정 구간이 끝나는 시각)에 오고 그 뒤로 앞을 보면 이벤트가 없다. 그 프레임이 7,201 ms에 오면 그
     프레임에서 `t_start_ms` 1,000인 `gaze_away_long` 시작과 경고가 난다.
   - 다른 안: 인정 구간의 끝이나 그 뒤에 온 관측 없는 프레임에서는 걸린 후보를 버리고, 이미 인정된 시간으로 그 프레임에서
     판정한다. 바뀌는 것은 SPEC 12.4절의 의사 코드와 설명, 그리고 벡터 `rules_gaze_credit_boundary`의 첫 장면을 설명하는
     문장뿐이다. 저장된 기댓값은 하나도 바뀌지 않는다: 다른 안을 참조 구현의 사본에 넣어 지금의 벡터에 돌리면 102개가 모두
     그대로 통과한다(2026-11-14 확인. 7,200 / 7,201 ms 경계는 어느 벡터도 지나지 않고 단위 시험 하나만 고정한다). 그 사본에서는
     7,200 ms의 프레임에서도 `t_start_ms` 1,000인 시작이 나고, 이 조건이 막으려던 0.4초 곁눈질의 경우는 여전히 이벤트가 없다.
     다른 안을 고르면 그 경계를 고정하는 벡터를 새로 더해야 한다.
   - 추천: 다른 안. 1 ms 경계가 없어지고, 구현이 막으려던 잘못된 시작도 나지 않으며, 저장된 기댓값이 바뀌지 않는다.
10. **모델과 함께 나가는 spec 파일의 이름**
    - 영향: 앱이 번들하는 파일 이름이다.
    - 지금 동작: `export/feature_spec_v{N}.json`(모델 버전마다 하나). 과제 명세와 계획서(4.7절, 7장)는 `feature_spec.json`이라
      적었다.
    - 다른 안: `feature_spec.json`(그때는 두 버전이 한 폴더에 있을 때를 정해야 한다).
    - 추천: 지금 이름. 버전마다 모델과 spec의 짝이 남는다.
11. **v0 모델을 첫 커밋 뒤 다시 내보낼까**
    - 영향: v0의 메타데이터 `commit`이 자리 채움 값 `uncommitted-dirty`다.
    - 지금 동작: 그대로 둔다. 커밋 해시는 v1부터 들어간다.
    - 다른 안: 첫 커밋 뒤 다시 내보낸다. 파일의 sha256과 크기(검토의 임시 저장소에서 148,558 → 148,581 bytes)가 바뀌어 모델
      벡터 두 개(`model_v0_inputs`, `e2e_250s`의 `model.sha256`)와 문서 세 곳(`export/README.md`, `docs/MODEL_CARD.md`, 이
      README)이 바뀐다.
    - 추천: 지금 동작. v0은 무작위 가중치의 연동용 파일이고, 앱 팀이 이미 쓰는 sha256이 바뀌지 않는다.
12. **테스트 벡터의 형식과 크기**
    - 영향: 저장소 크기와, 벡터를 다시 만들 때마다 바뀌는 양.
    - 지금 동작: JSON 파일 102개, 합계 30,588,827 bytes이고(`python -m drowsy.vectors generate`가 쓴 크기) 다시 만들 때마다
      그만큼을 새로 쓴다(Git LFS 없음). 처음에는 43개 약 18 MB, 2차 검증 전에는 68개 22,650,508 bytes, 3차 검증 전에는 81개
      26,908,941 bytes, 4차 검증 전에는 89개 27,983,687 bytes였다. 4차 검증 뒤에 더한 13개 가운데 가장 큰 것은 손으로 놓은
      기준선 세션 셋(653,722 / 707,708 / 716,492 bytes)이다. 스텝마다의 확률(`expected.step_probs`) 항목은 결정 3을 고정하려고 구현하면서 더했다(10장).
    - 함께 정할 것: 표시용 상태 `gaze_zone`(앱이 운전자에게 보여 줄 시선 영역 상태, SPEC 12.10절)을 벡터에 담을지. 지금
      형식에는 그 값을 적을 항목이 없어서, 옮긴 코드의 `gaze_zone`이 틀려도 벡터로는 드러나지 않고 저장소의 단위 시험만
      고정한다(SPEC 16.1절). 담으려면 `expected`에 프레임마다의 항목을 하나 더해야 하는데, 형식을 바꾸는 일이라 하지 않았다.
    - 다른 안: 더 작은 형식(압축, 긴 세션 줄이기)이나 Git LFS. `gaze_zone`은 프레임마다의 항목으로 더한다.
    - 추천: 지금 형식. 앱 팀이 JSON을 그대로 읽고, 크기는 아직 저장소에 부담이 되는 수준이 아니다. 커지면 압축을 먼저 본다.
      `gaze_zone`은 앱이 화면에 쓰기로 하면 담는다(벡터를 모두 다시 만들어야 한다).
13. **재학습 길이와 후보 점수를 정하는 규칙**
    - 영향: 최종 모델과 fold 모델의 학습 길이, 그리고 어느 계열·설정을 고르는지가 이 규칙으로 정해진다. 실제 데이터 결과를
      보기 전에 정해야 한다(PLAN 11장: 절차는 미리 고정한다).
    - 지금 동작: 학습 길이 = 안쪽 분할별 조기 종료 단위의 중앙값, 후보 점수 = 분할별 최고 단위 macro-F1의 평균.
    - 다른 안: 분할을 평균한 검증 곡선이 최대인 단위를 학습 길이로, 그 최댓값을 점수로 쓴다(안쪽 학습을 같은 길이까지
      돌려야 한다).
    - 추천: 다른 안. 분할마다 잡음이 섞인 최고점의 중앙값보다 평균 곡선이 안정적이고, 점수와 학습 길이가 같은 곡선에서 나온다.
14. **D11 판정(모델을 기본 판정원으로 둘지)의 집계 방식**
    - 영향: macro-F1 차이가 기준(0.02) 근처이면 판정이 집계 방식에 달린다.
    - 지금 동작: 분류 두 조건(macro-F1 차이, 2클래스 영상 정확도)은 fold 평균, 오경보 세 조건은 5개 fold를 합친 값이다.
    - 다른 안: 모두 합친 값(또는 모두 평균).
    - 추천: 지금 동작. 분류 지표는 모델 카드가 fold 평균으로 보고하고, 오경보는 fold마다 수가 적어 합쳐야 안정적이다.
15. **`training.export_arm`이 아닌 정규화 조건으로 돌린 신경망 실행을 부가로 둘까**
    - 영향: 정규화 조건 셋의 비교(D1)를 어느 리포트의 어느 절에서 읽는지가 달라진다.
    - 지금 동작: `train cv --arm`에 내보낼 조건이 아닌 조건을 주면 그 실행의 묶음이 모두 부가로 표시되어 summary.md 9절에만
      실리고, D11 판정을 하지 않는다(4.9절).
    - 다른 안: 그 조건의 리포트에서 주 결과로 싣는다.
    - 추천: 지금 동작. 내보낼 모델과 D11은 한 조건으로만 정하고, 다른 조건은 비교용이다.
16. **E3 부가 실행의 "설정은 E2에 고정"을 어떻게 읽을까**
    - 영향: 지금 방식에서는 부가 실행(순서형 소프트 라벨, 속도 특징 제외)과 주 실행의 차이에 선택의 차이도 섞이고, 부가 실행
      하나가 주 실행만큼 걸린다.
    - 지금 동작: 부가 실행이 바깥 fold마다 안쪽 선택(계열, 설정, 학습 길이, 온도)을 다시 한다.
    - 다른 안: E2(주 실행)에서 고른 후보, 학습 길이, 온도를 그대로 쓴다.
    - 추천: 다른 안. 계획서 문장에 가깝고, 차이가 바꾼 처리 하나의 효과만 보인다.
17. **회전을 가르지 못한 영상을 어떻게 확인할까**
    - 영향: 거꾸로 서도 검출되는 얼굴의 영상이 회전 정보 없이 거꾸로 저장돼 있으면 180도 틀린 각도로 추출되고, 표시만 된 채
      그 신호가 학습과 평가에 쓰인다.
    - 지금 동작: 추출이 그런 영상을 "가르지 못함"으로 표시하고, 추출 리포트와 여섯 명령(스텝 표, E0, `baseline_lgbm`(cv,
      probes), `train`(cv, final), 라벨 없는 영상 평가, `--raw` 데모)이 표시된 영상을 알린다(4.5~4.9, 4.11, 4.12절). `--raw`
      데모 말고는 보정표에 각도가 없는 영상을 알리고, `--raw` 데모는 보정표를 보지 않고 그 파일의 기록만 알린다. 영상을 빼지
      않고 종료 코드도 바꾸지 않는다.
    - 잰 사례(2차 검증의 검증자, 공개 얼굴 사진 두 장으로 만든 클립 184개): 회전 정보가 없거나 틀린 클립 40개 중 35개가 맞았고
      틀린 5개는 모두 정확히 180도 차이였고 모두 표시됐다. 기울인 클립 144개 중 41개가 틀렸고 모두 표시됐다. 표시 없이 틀린
      클립은 0개였다. 거꾸로 서도 검출되는 얼굴은 맞든 틀리든 모든 클립이 표시됐다.
    - 다른 안: (가) 표시된 영상을 모두 사람이 보고 보정표에 각도를 적는다. (나) 고른 각도와 반대 각도의 검출 신뢰도를 비교하는
      더 엄격한 점검을 더한다(앞선 검증자의 측정: 똑바로 선 얼굴은 신뢰도 기준 0.8~0.9까지 검출되고 거꾸로 선 얼굴은 0.7에서
      놓쳤다).
    - 추천: (가). RLDD는 파일 182개라 표시된 영상을 다 볼 수 있는 양이다. 표시가 너무 많으면 (나)를 더한다.

**정비에서 정한 것(질문이 아니다)**. `docs/SPEC.md` 20장의 같은 이름의 문단과 같은 두 가지다.

- 2초~10초 끊긴 뒤의 프레임이 한꺼번에 내는 스텝들은 끊기기 직전의 진행 중 감김을 쓴다. 검토에서 동작을 바꾸지 않고
  문서로 남기기로 정했다(13.3절 끝, `docs/SPEC.md` 11.6절, 벡터 `steps_closure_across_gap`).
- 감는 속도의 되돌아보기는 쌍의 간격만으로 끊는다: 직전 유효 표본과의 간격이 `max_dt_ms`를 넘는 쌍에서만 끊고, 두 유효
  표본 사이에 무효 프레임이 끼어 있어도 간격이 `max_dt_ms` 이하이면 끊지 않는다(SPEC 10.5절 11번, 벡터
  `runs_velocity_after_gap`의 다섯째 구간). 검토의 고침(되돌아보기를 끊김에서 끊는다)을 구현하면서 정한 동작이다. 2차 검증
  뒤에는 SPEC이 3번 질문에 덧붙여 확인을 청했지만, 3차 검증 뒤에 README·계획서와 같게 이 문단으로 옮겼다. 되돌리는 방법은
  10장의 표에 있다.

구현자가 정한 그 밖의 동작은 10장에 있다.

## 10. 구현하면서 정한 것 (결정 대기가 아니다)

계획서가 정하지 않았거나 검토의 고침 계획과 다르게 구현자가 정한 동작이다. 지금 그대로 쓰되, 소유자가 다르게 원하면
아래 방법으로 되돌린다. 계획서 14장 표 (a)에 전체 목록이 있다.

| 무엇 | 정한 동작 | 되돌리는 방법 |
|------|-----------|---------------|
| 오프라인 오경보 평가의 재시작(결정 4에 맞춤) | `evaluate.alarm_videos`가 한 세션 안에서 스텝 번호 k가 건너뛰는 곳에서 판정 상태와 두 판정기를 새로 시작한다. k는 10초 넘는 끊김 뒤에만 건너뛰고, 그 프레임은 `RuleEngine`이 다시 시작하는 프레임이다. 무작위 450세션(시작 946번)에서 모델 판정이 앱과 모두 같았다. 남은 차이(가정): PERCLOS와 either 판정은 예측 행이 시작되기 전 119스텝 안에서 `perclos_60`이 시작 비율에 닿으면 앱이 먼저 시작할 수 있다(오프라인 평가는 예측 행만 본다) | `alarm_videos`의 `elif step_k != previous_k + 1:` 갈래를 비운다 |
| 축소 설정 체크포인트의 내보내기 거부 | 체크포인트 v2에 `fast`가 있고 `train final`이 설정에서 적는다. `export_onnx --checkpoint`는 `fast`인 체크포인트를 `paths.export_dir`과 그 아래로 내보내지 못한다. 키가 없는 파일은 축소 설정이 아닌 것으로 읽는다 | `export_onnx._run_cli`의 `if checkpoint.fast: refuse_fast_export(...)` 두 줄을 지운다 |
| 경고 종류 표시와 모델 호출마다의 점(결정 2, 3에 맞춤) | 데모와 라벨 없는 영상 리포트가 경고를 바로 낸 것, 나중에 낸 것, 되풀이로 나눈다. 데모는 모델 호출마다 확률 점 하나를 그린다 | `demo.AlertKinds` 대신 "그 프레임의 이벤트에 없는 경고 = 되풀이" 표시로, 점은 프레임마다 `out.probs` 하나로 되돌리고 `alerts.csv`의 `immediate_alerts`, `deferred_alerts` 열을 뺀다 |
| 학습 증강: 프레임 누락 변형을 섞고 출처의 몫을 원본 0.5, 15 fps 0.25, 변형 0.25로 | 4.9절. 몫은 계획서에 없는 값이다 | `config/local.yaml`에서 `training.frame_drop_variants: 0`(변형을 쓰지 않음), `training.fps_sim_share`와 `frame_drop_share`로 몫을 바꾼다. 남는 몫은 원본으로 간다 |
| 부가 실행: 순서형 소프트 라벨, 속도 특징 제외, fold별 전역 상수 | 각각 따로 돌리는 명령이고(4.8, 4.9) 주 결과를 바꾸지 않는다. 이웃 몫 0.2와 끝 클래스 규칙은 정한 값이다 | 그 명령을 돌리지 않는다 |
| 주 실행 안의 부가 묶음: 15 fps 평가, fold 모델의 INT8 비교 | `train cv`와 `baseline_lgbm cv`가 함께 만들어 부가로 싣는다. LightGBM fold 모델의 15 fps 평가는 검토의 빠진 항목 설명을 따라 더했다 | INT8 비교는 `export.int8_if_tcn: false`. 15 fps 평가에는 끄는 설정이 없고 그 추출이 있으면 늘 돈다 |
| 밝기를 읽는 교란 점검 | 교란 점검이 추출 메타데이터의 밝기, 해상도, 평균 fps, 가변 프레임률 여부를 읽는다(가변 프레임률 여부는 PLAN 6장 목록에 없던 것을 더했다) | `baseline_lgbm.RECORDING_COLUMNS`에서 `brightness`, `variable_fps`를 뺀다 |
| 감는 속도의 되돌아보기를 끊는 곳 | 쓸 수 없는 쌍(간격 > `max_dt_ms`)에서만 끊고 무효 프레임 하나만으로는 끊지 않는다. 무효 프레임마다 끊는 안과 정비 전 벡터 43개와 무작위 3,137구간에서 차이가 없었다 | 검출기의 무효 프레임 갈래에서 버퍼를 비우고 시험 하나의 기댓값(11.25 → 1.25)을 고치고 벡터 `runs_velocity_after_gap`을 다시 만든다 |
| 회전 점검의 두 선택 | 가르지 못한 각도에 메타데이터 각도가 없으면 roll이 가장 작은 각도를 고르고 표시한다(측정 8건 중 6건 맞음. 메타데이터를 따랐다면 8건 모두 옆으로 누웠다). 차량 영상은 보정표를 따로 쓴다 | `extract.choose_rotation`에서 메타데이터 각도를 두게 하고, 보정표를 하나로 합친다 |
| E0의 판정 단위와 반올림 | 세션(파일) 단위라 RLDD에서 분모가 182다. `off_axis` 기준선(고개를 돌린 채 완성된 기준선)은 완성으로 친다. 제안 상수는 소수 넷째 자리로 반올림한다 | `signal_check.py`(논리 영상 단위로 하려면 나뉜 영상을 합치는 규칙이 필요하다) |
| 라벨 없는 영상의 시간당 경고 수 | 분모는 얼굴이 검출된 시간, 경고 수에는 되풀이와 나중에 낸 경고도 넣는다(따로도 싣는다), 속도를 주지 않으면 억제하지 않는다 | `evaluate.run_unlabeled`의 합계 계산 |
| 벡터 형식의 `expected.step_probs` | 결정 3(스텝마다 모델 호출)을 벡터로 고정하려고 스텝마다의 확률을 벡터 형식에 더했다. 형식 전체에 대한 결정은 9장 12번에서 기다린다 | `drowsy/vectors.py`에서 그 항목을 쓰고 비교하는 부분을 빼고 벡터를 다시 만든다(그러면 결정 3이 벡터로 고정되지 않는다) |
| 모든 추출 입력의 회전 점검과 `--rotation`(2차 검증 뒤) | `extract --input-dir`과 `--video`도 회전을 점검하고, `extract --video`와 `demo --video`가 `--rotation`을 받으며 그 값이 보정표보다 우선한다. 데모는 점검을 한 번 돌려 각도와 가르지 못함 여부를 출력, JSON, 그림에 적는다(4.5, 4.12절). 점검 기록이 없는 옛 `--input-dir`·`--video` 결과는 다시 추출한다 | `extract._extract_all`에서 점검 설정을 `--dataset`, `--unlabeled` 작업에만 주고, 두 명령의 `--rotation`과 `demo.resolve_video_rotation`을 뺀다 |
| 회전 점검 기록을 읽는 세 명령(2차 검증 뒤) | `python -m drowsy.dataset`, `signal_check e0`, `evaluate unlabeled`가 한 함수(`dataset.read_rotation_checks`)로 기록과 보정표를 읽어 한 줄과 목록(리포트가 있으면 그 절)을 낸다. 분류는 다섯이다(검토가 요청한 셋에 "보정표에 있으나 추출에 적용되지 않음"과 "기록 없음"을 더했다). 표시된 영상은 빼지 않는다 | 세 명령에서 `read_rotation_checks`와 `rotation_check_line`을 부르는 곳을 지운다 |
| 실행 이름의 검사(2차 검증 뒤) | `--run-name`이 있는 모든 명령이 다른 일을 하기 전에 `config.check_run_name`을 부른다(4장 처음). `signal_check`와 `evaluate`는 라이브러리 함수 안에서도 검사한다 | 각 모듈 `main()`의 `check_run_name(...)` 호출을 지운다 |
| 추출 처리 시간의 분리(2차 검증 뒤) | summary.md 6절이 이번 실행에서 잰 값과 이어 온 값을 따로 적고, 동시 처리 문장은 앞의 것에만 붙인다. 합친 하나의 수치는 내지 않는다 | `extract.write_extraction_report`의 6절에서 `_timing_carried`로 나누는 부분을 하나로 합친다 |
| 이어 온 처리 시간의 출처(3차 검증 뒤) | 한 영상의 값이 시험 추출 리포트와 전체 추출 리포트의 전 판에 다 있으면 두 CSV의 수정 시각으로 나중의 것을 쓰고, 두 값이 같으면 시험 추출의 값으로 적는다. 이어 온 값은 출처 리포트별로 한 줄씩 적는다(4.5절). 47열 CSV는 그대로다 | `extract.PreviousTimings.lookup`에서 전 판의 값을 늘 먼저 쓰게 하고, 6절의 출처별 두 줄을 한 줄로 합친다 |
| 원본 파일 크기로 이어 하기 판단(3차 검증 뒤) | 추출 결과에 원본의 파일 이름과 크기를 적고, 크기가 지금 영상과 다르면 네 입력 모두 다시 추출하며 `replace` 줄로 알린다. 건너뛴 결과의 원본 경로가 다르면 `skip` 줄에 적는다. 크기가 없는 옛 결과는 전처럼 다룬다(4.5절) | `extract._current_metadata`에서 크기 비교 두 줄을 지운다(그러면 이름이 같은 다른 영상을 건너뛴다) |
| 회전 점검 줄을 내는 학습 명령(3차 검증 뒤) | `train`(cv, final)과 `baseline_lgbm`(cv, probes)이 학습 전에 회전 점검 줄과 표시된 영상을 출력하고, 실행 기록의 `rotation_check`와 리포트 설명에 남긴다(4.8, 4.9절). 알리기만 한다 | 두 모듈 `main()`의 `print_rotation_check(...)` 호출과 `train.load_cv_data`의 `rotation_note`를 지운다 |
| `--raw` 데모의 회전 줄(3차 검증 뒤) | 추출 parquet의 회전 기록을 한 줄로 출력하고 JSON `source`에 옮겨 적으며, 가르지 못한 기록이면 그림 머리말에도 적는다. 보정표는 보지 않는다(4.12절) | `demo.run_demo_raw`에서 `read_raw_rotation`의 결과를 쓰는 곳과 `main()`의 출력 한 줄을 지운다 |
| 리포트의 D 번호 안내(3차 검증 뒤, 4차 검증 뒤에 넓힘) | D 번호를 쓰는 리포트 넷의 머리말에 "D 번호는 계획서 1.2절의 결정 번호이고 계획서 14장의 소유자 결정(2026-11-13, 2026-11-14)과는 번호가 다르다"는 한 줄(`evaluate.PLAN_DECISION_NOTE`)을 넣는다: 교차검증의 summary.md와 라벨 없는 영상 평가의 summary.md(3차 검증 뒤), `train final`의 `final/summary.md`와 `baseline_lgbm cv --all-w`의 `w_grid.md`(4차 검증 뒤). 4차 검증 뒤에 문장을 소유자 결정의 수를 세지 않게 고쳤다(전에는 "네 가지 동작"이라 적어 결정 5와 맞지 않았다). `속도 억제(D3)`는 `속도 억제(PLAN 1.2절 결정 D3)`로 적는다 | `evaluate.PLAN_DECISION_NOTE`를 넣는 네 곳(`evaluate.py`의 두 곳, `train.py`, `baseline_lgbm.py`)을 지운다 |
| 감김 검출기 설정의 `hold_ms < clock.reset_gap_ms`(3차 검증 뒤) | 설정을 읽을 때 두 검출기 모두에 검사한다(7장). 지금 값(500, 10,000)은 통과하므로 해시는 그대로다. 어기는 설정에서는 끊김 앞뒤의 감김을 가르는 것이 규칙 엔진의 `reset()` 호출 하나뿐이 되어, 그 호출을 빠뜨린 이식 코드를 벡터가 잡지 못한다 | `spec._validate_run_detector`의 `hold_ms < clock.reset_gap_ms` 검사를 지운다 |
| 하품·끄덕임 계수기 설정의 `cancel_gap_ms < clock.reset_gap_ms`(4차 검증 뒤) | 설정을 읽을 때 `pipeline.yawn.cancel_gap_ms`와 `pipeline.nod.cancel_gap_ms`가 `clock.reset_gap_ms`보다 작은지 검사하고(7장), 어기면 `pipeline.yawn: need cancel_gap_ms < clock.reset_gap_ms (cancel_gap_ms is 20000, reset_gap_ms is 10000)`처럼 두 값을 적은 설정 오류(`SpecError`)다. 지금 값(500, 10,000)은 통과하므로 해시는 그대로다. 어기는 설정에서는 끊김 앞의 하품·끄덕임과 뒤의 것을 가르는 것이 끊김 뒤 계수기의 `reset()` 호출 하나뿐이 되어, 그 호출을 빠뜨린 이식 코드를 벡터가 잡지 못한다. 끄덕임의 느린 평균에 쓰는 `nod.ema_max_dt_ms`에는 같은 조건을 두지 않았다(실험으로 그 값이 초기화와 얽히지 않음을 확인했다는 설명이 `drowsy/spec.py`에 있다) | `drowsy/spec.py`의 두 `_check(... cancel_gap_ms < clock.reset_gap_ms ...)` 호출(지금 763~772줄)과 그것을 부르는 시험(`tests/test_spec.py`)을 지운다 |
| `.gitignore`의 영상과 사본 설정(2차 검증 뒤, 3차 검증 뒤에 넓힘) | 일부러 넓은 패턴: `config/local*`(이름이 `local`로 시작하는 모든 파일), 어느 깊이에 있든 `videos/` 폴더, 영상 확장자 다섯 가지의 모든 대소문자 조합(`*.[mM][pP]4` 등). 4.0절 | 좁히려면 `/videos/`(저장소 루트의 폴더만)와 `config/local*.yaml`로 바꾸고, 대소문자 패턴을 `*.mp4`, `*.MP4` 두 줄씩으로 되돌린다 |
| 누설 회귀 시험의 강화(2차 검증 뒤, 3차 검증 뒤) | `tests/test_leakage.py`가 손대지 않은 둘째 피험자, 명령 여덟 개, 예측 묶음 44개의 비교를 한다. 3차 검증 뒤에는 그 피험자의 라벨만 맞바꾼 셋째 데이터로 "평가받는 피험자 자신의 라벨이 예측에 닿지 않는다"를 더 본다(4.3절). 시험만 바뀌었고 출력에는 영향이 없다. 혼자 돌려 54.4초(다른 작업이 없는 이 PC) | 시험 파일에서 `relabelled` 데이터와 그 비교 셋을 지우거나, 시험 파일을 전의 판으로 되돌린다 |
| 계약 검사의 온도 규칙(2차 검증 뒤) | logits를 바꿀 수 있는 것은 `temperature` 상수로 나누기뿐이고, 기본 경로와 INT8 파일은 온도 1.0이어도 그 상수를 가져야 하며, logits가 Div나 Mul의 출력이면 거부한다(4.10절). 계약 값은 그대로다 | `export_onnx._check_graph_constants`에서 `temperature_may_be_absent` 갈래와 Div·Mul 검사를 뺀다 |
| 계약 검사의 logits 노드(3차 검증 뒤) | logits는 마지막 선형 층 그대로여야 한다: 기본·대체 경로는 그래프의 유일한 Gemm(`alpha = beta = 1`, `transA = 0`, 상수 가중치와 편향), INT8은 양자화한 그 층의 끝인 편향 Add(4.10절). 마지막 층의 가중치를 통째로 줄이거나 키운 파일은 잡지 못하고, 그것은 체크포인트와의 일치 검증이 지킨다. 저장소의 v0과 정상적인 내보내기는 모두 통과한다. 양자화기의 판이 바뀌어 INT8 모양이 달라지면 이 검사도 다시 맞춰야 한다(그때는 INT8 파일이 검사에서 걸리므로 조용히 지나가지 않는다) | `export_onnx._check_graph_constants`의 `_check_logits_producer(...)` 호출 한 줄을 지운다 |
| 계약 검사의 속성 읽기(4차 검증 뒤) | logits를 내는 Gemm의 `alpha`, `beta`, `transA`와 INT8 Cast의 `to`는 한 번만, 정해진 형으로, 값을 가진 채 있어야 하고 아니면 한 줄의 `ContractError`다(4.10절). 형이 틀린 속성에서 나던 `TypeError`를 계약 오류로 바꾼 결함 수정이고, 값이 없는 INT 속성을 받지 않는 것은 구현자가 정한 것이다(가정. onnxruntime은 그 파일을 기본값으로 돌리지만 이 저장소의 내보내기는 그런 파일을 만들지 않는다) | 값이 없는 속성을 다시 받으려면 `export_onnx._attribute`에서 `not attribute.HasField(field)` 조건을 빼고 그때는 `default`를 돌려주게 한다. 형 검사까지 빼면 `TypeError`가 다시 난다 |
| 내보내기 보호 장치의 기준 폴더(2차 검증 뒤) | 상대 경로 `paths.export_dir`을 지금 폴더와 저장소 루트 둘 다에 대어 본다(`export_onnx.is_export_dir_or_below`, `REPO_ROOT`). `drowsy/train.py`도 같은 함수를 쓴다 | `protected_export_dir`에서 `REPO_ROOT` 기준 비교(`configured.append(...)`)를 뺀다 |
| 늘 지키는 `export/`(3차 검증 뒤) | 설정의 `paths.export_dir`이 다른 폴더를 가리켜도 저장소의 `export/`(`REPO_ROOT / "export"`)를 내보내기 폴더로 본다(`export_onnx.protected_export_dir`, 4.10절). 늘 지키는 폴더는 그 하나이고, `out/`, `reports/`는 설정이 가리킬 때만 지킨다 | `protected_export_dir`에서 `SHIPPED_EXPORT_DIRNAME` 폴더를 보는 부분을 지운다 |
| 다른 조건 실행의 D11(2차 검증 뒤) | `train cv --arm`이 내보낼 조건이 아니면 D11 판정을 하지 않고(`gate`가 `null`) 리포트를 끝까지 쓴다(4.9절) | `train.run_nested_cv`에서 판정의 `not other_arm` 조건을 뺀다(그러면 그 실행은 판정에 필요한 행이 없어 멈춘다) |
| 예측 표의 부가 표시 검사(2차 검증 뒤) | 출처가 원본이 아니거나 변형이 기본이 아닌 묶음은 부가로 표시돼 있어야 하고, 표시가 없으면 두 열에서 정한다(4.9절). 기본 표시의 묶음(ONNX, INT8, 다른 조건 실행)은 가진 표시를 그대로 쓴다 | `evaluate.validate_predictions`의 그 검사와 채우기를 뺀다 |
| 스텝 표 캐시의 내용 지문(2차 검증 뒤에 꼬리말, 3차 검증 뒤에 파일 전체) | 캐시 열쇠에 추출 파일마다 **파일 전체**의 sha256을 넣고, 한 명령 안에서는 (경로, 크기, 수정 시각)별로 기억한다(캐시 형식 4, 4.7절). 2차 검증 뒤의 꼬리말 해시는 꼬리말이 같은 다른 내용을 놓쳤다. 비용은 합성 데이터 443 MB에서 명령의 첫 호출 0.22초(페이지 캐시를 비운 뒤 0.41초) | 꼬리말만 보던 2차 검증 뒤의 동작: `dataset.raw_content_digest`를 파일 꼬리만 읽는 판으로 되돌리고 `CACHE_FORMAT`을 3으로. 내용 지문 없이(크기와 수정 시각만): `_source_fingerprint`에서 `raw_content_digest`를 빼고 `CACHE_FORMAT`을 2로 |

## 11. 코드의 가정 목록

계획서가 정하지 않았거나 모호해서 구현하면서 정한 것이다. 코드에서는 docstring과 주석의 `가정` 줄에 적혀 있다.

**만든 방법**: 2026-11-15, 4차 검증 뒤의 정비가 끝난 코드에서 `grep -rn "가정" drowsy/ tests/ config/`가 찾은 412줄(코드와
설정 파일 388줄, 시험 파일 24줄)을 저장소 밖의 작은 스크립트(문서 단계의 `collect_r5b.py`. 같은 grep을 파일마다 돌린다)로
정리했다. 그 가운데 384줄이 가정 항목이고(30개 파일. 코드와 설정 367줄, 시험 파일 17줄), 28줄(코드 21줄, 시험 파일 7줄)은
새 가정이 아니라 다른 곳의 가정을 가리키거나 시험이 리포트 문구의 '가정'을 확인하는 줄이라 맨 끝에 따로 모았다.
3차 검증 뒤의 목록(항목 364줄, 가리키는 줄 20줄, 실시간 시험 도구 포함)과 견주면 달라진 것은 넷이다.
- 4차 검증 뒤의 정비가 더한 가정 항목 셋: `export_onnx.py:738`(값이 비어 있는 ONNX 속성도 거부한다), `vectors.py:82`,
  `vectors.py:87`(손으로 값을 고른 벡터의 설명).
- 실시간 시험 도구를 4차 검증 뒤에 고치면서 바뀐 항목 넷(`live.py:189`, `live.py:195`, `live.py:199`, `live.py:223`)과 새로
  생긴 가리키는 줄 하나(`live.py:465`).
- 시험 파일 묶음(맨 끝의 "시험 파일 (`tests/`)")을 새로 넣었다. 전에는 시험 파일의 줄을 목록에서 뺐다.
- 나머지 항목은 문장이 그대로이고 줄 번호만 지금 파일에 맞췄다.

항목은 `가정:`, `가정(…):`, `# 가정.`으로 시작하는 줄(또는 줄 중간의 그 표현)이고, 같은 문단의 다음 줄을 이어 붙였다. 시험
파일에서는 `(가정)`을 붙인 주석도 항목으로 셌다. 각 줄 앞의 `파일:줄`은 그 가정이 적힌 곳이고(시험 파일은 `tests/`를 붙여
적는다), 그 뒤는 그 줄이 속한 함수나 클래스다. 목록을 만든 뒤 두 방향으로 다시 확인했다(문서 단계의 `check_markers_r5b.py`):
모든 `파일:줄`이 실제로 `가정`이 든 줄을 가리키고, `drowsy/`, `tests/`, `config/`에서 `가정`이 든 줄마다(그러니 `가정:` 표지가
있는 줄마다) 목록에 줄이 정확히 하나 있다. 코드가 바뀌면 줄 번호가 어긋나므로, 다시 만들 때도 같은 grep으로 찾는다.
인용한 문장에 나오는 D 번호의 뜻은 이 README 맨 앞의 표에 있다. 그 밖에 인용문에 나오는 표기: E0~E4는 계획서 11장의 실험
단계(E0은 라벨 없는 신호 점검과 잠정 상수 확정, E1은 LightGBM 기준선, E2는 TCN 대 GRU의 중첩 교차검증, E3은 부가 분석, E4는
최종 모델), `R0`은 기준선이 잡은 기준 머리 자세(회전 행렬, PLAN 4.2절), `(검토 K5)`는 2026-11-13 독립 검토의 지적 번호로
"프레임이 끊겼다 와서 프레임 하나가 스텝을 여럿 낼 때 그 스텝들이 쓰는 진행 중 감김"에 관한 지적이다(9장 끝의 "정비에서
정한 것" 첫 항목. 검토 보고서는 저장소에 없다).

### 이식 대상 모듈 (Android 팀이 Kotlin으로 옮기는 코드)

**`drowsy/types.py`** (7개)

- `types.py:8` 모듈 설명 — 계획서는 이벤트 시각만 원래 t_ms로 적으라고 했고 나머지 자료형의 시계는 정하지 않았다. 시계를 바꾸는 지점을 없애려고 모든 자료형을 원래 t_ms로 통일한다. 세션 원점(처음 받아들인 프레임의 t_ms)이 필요한 계산(기준선 경과 시간, 스텝 경계와 k)은 그 값을 따로 받아서 쓴다.
- `types.py:73` `BASELINE_OK` — 계획서는 fallback_global과 off_axis만 이름을 정했다. 정상 완성은 "ok"로 적는다.
- `types.py:223` `Baseline` — 보정 없이 전역 상수로 바로 만든 기준선(`calibrate.global_baseline`)은 0으로 둔다.
- `types.py:228` `Baseline` — 분포 기록용 `eyeBlink` 50, 90 분위수는 계획서의 필드 목록에 없으므로 여기 두지 않는다.
- `types.py:258` `FrameState` — 계획서는 "대체 상태에서는 R0이 없으므로 시선 규칙을 끈다"(4.2절), "시선 규칙은 기준선 완성 후 시작한다"(D4)고만 했다. 규칙 엔진은 FrameState만 받으므로 두 경우 모두 이 필드를 False로 해서 전한다. 기준선 완성 전에는 전역 상수 기준선(status = fallback_global)으로 계산하므로 저절로 False가 된다.
- `types.py:291` `ClosureRun` — enter 이상이던 토막마다 반올림된 두 시각(위로 지난 시각, 아래로 지난 시각)의 차를 구해 더한다.
- `types.py:295` `ClosureRun` — 감는 속도와 뜨는 속도가 둘 다 측정됐을 때만 True로 한다(플래그가 하나뿐이므로).

**`drowsy/signals.py`** (3개)

- `signals.py:50` `eye_aspect_ratio` — 눈 폭 |p1−p4|가 0이면 0.0을 돌려준다(PLAN 3.3절: 0으로 나누게 되는 비율은 0). `compute`는 폭이 6 px 미만인 눈에서 이 함수를 부르지 않는다.
- `signals.py:73` `iris_ratio` — 두 눈 꼬리가 같은 점이면 (0.0, 0.0)을 돌려준다(PLAN 3.3절).
- `signals.py:220` `compute` — 홍채 비율은 쓸 수 있는 눈마다 항상 계산한다. `RawFrame.lm`이 홍채 랜드마크를 담는 것이 계약이다. face=True인데 필요한 랜드마크나 블렌드셰이프가 없으면 KeyError, 행렬 길이가 16이 아니면 ValueError가 난다.

**`drowsy/calibrate.py`** (10개)

- `calibrate.py:42` `nearest_rank_quantile` — 순위는 1 이상 n 이하로 자른다(q = 0이면 최솟값).
- `calibrate.py:136` `BaselineEstimator` — 얼굴 검출 비율(시간 기준)은 PLAN 3.3절의 시간 규칙으로 센다. 얼굴 프레임은 다음 프레임까지의 간격만큼, 최대 `clock.max_hold_ms`(200 ms)까지의 시간을 대표하고, 마감을 넘는 부분은 세지 않는다. 분모는 마감 값(데이터 없는 시간을 포함한 수집 구간 전체)이다.
- `calibrate.py:139` `BaselineEstimator` — E의 표본에는 쓸 수 있는 눈(눈 폭 ≥ 6 px)의 EAR만 넣는다. 쓸 수 없는 눈의 ear 0.0은 측정값이 아니다. 한쪽 눈이라도 쓸 수 있는 표본이 하나도 없으면 그 눈의 E를 정할 수 없으므로 조건 3을 어긴 것으로 본다.
- `calibrate.py:141` `BaselineEstimator` — 마감 시각에 정확히 온 프레임은 수집 구간에 넣지 않는다(스텝 경계 규칙과 같다. PLAN 4.4절).
- `calibrate.py:142` `BaselineEstimator` — 프레임이 한참 끊겼다가 와서 마감을 여러 번 지나면 지난 마감을 순서대로 하나씩 검사한다. T_b는 조건을 만족한 첫 마감이다.
- `calibrate.py:144` `BaselineEstimator` — 전역 상수로 대체한 기준선의 `t_complete_ms`는 대체를 결정한 마감(180000)이다. 특징 누적은 그때 시작한다.
- `calibrate.py:145` `BaselineEstimator` — `t_ms`가 직전 프레임 이하이면 ValueError를 낸다. 그런 프레임은 호출자가 버리고 넘기지 않는다.
- `calibrate.py:299` `eye_closure` — E·(1 − rho)가 0 이하이면 a_e = 0으로 둔다(PLAN 3.3절: 0으로 나누게 되는 비율은 0).
- `calibrate.py:328` `frame_state` — `FrameSignals`에는 변환 행렬이 없으므로 R은 (yaw, pitch, roll)로 다시 만든다.
- `calibrate.py:329` `frame_state` — yaw_rel이 정확히 0이면 R을 먼 쪽으로 친다. 이때 w_far = 1이라 결과는 두 눈 평균으로 같다.

**`drowsy/blink.py`** (13개)

- `blink.py:80` 모듈 설명 — 끊긴 시간은 HOLD뿐 아니라 CLOSED에서도 검사한다. 무효 프레임 없이 프레임 자체가 hold_ms 넘게 끊겨도 같은 구간으로 잇지 않는다(PLAN 3.3절: 데이터 없는 시간은 얼굴 없는 시간과 똑같이 취급한다). 따라서 CLOSED에서 끊긴 시간이 hold_ms를 넘은 무효 프레임은 HOLD를 거치지 않고 바로 구간을 닫는다.
- `blink.py:83` 모듈 설명 — 쌍과 보간은 "직전 유효 표본"만 본다. 두 유효 표본 사이에 무효 프레임이 끼어 있어도 간격이 max_dt_ms 이하이면 쌍으로 쓴다.
- `blink.py:85` 모듈 설명 — 감는 속도의 거슬러 보기는 쓸 수 없는 쌍에서만 끊는다. 위 가정대로 무효 프레임이 끼어 있어도 간격이 max_dt_ms 이하인 쌍은 쓸 수 있는 쌍이므로 끊지 않는다(PLAN 4.5절의 "연속 유효 쌍"을 시각 기준으로 읽은 것).
- `blink.py:87` 모듈 설명 — 진입 쌍이 쓸 수 없는 쌍이어도(감는 모습을 보지 못한 구간) 구간 안에서 정점까지 오른 쌍이 있으면 그 증가율을 감는 속도로 쓴다(계획서의 "정점까지" 그대로다. velocity_valid에 진입 쌍 조건을 더하지 않는다).
- `blink.py:89` 모듈 설명 — 보간할 수 없을 때 start_ms(와 뒤 토막의 시작)는 enter 이상을 처음 본 프레임의 시각, end_ms(와 앞 토막의 끝)는 enter 이상이었던 마지막 유효 프레임의 시각이다. 감긴 것을 직접 본 시간만 길이에 넣는다.
- `blink.py:91` 모듈 설명 — 잘린 구간의 end_ms는 그 전에 enter 아래로 내려간 표본을 봤더라도 enter 이상이었던 마지막 유효 프레임의 시각이다(계획서 문장 그대로).
- `blink.py:93` 모듈 설명 — 정점은 최대 closure에 처음 도달한 표본이다(같은 값이 다시 나와도 옮기지 않는다).
- `blink.py:94` 모듈 설명 — velocity_valid는 잘리지 않은 구간에서 0보다 큰 감는 속도와 뜨는 속도를 둘 다 쟀을 때만 True다.
- `blink.py:95` 모듈 설명 — min_duration_ms는 잘린 구간에도 적용한다.
- `blink.py:96` 모듈 설명 — `ongoing_ms()`의 끝은 enter 이상이었던 마지막 유효 프레임의 시각이다(프레임 시각, 보간하지 않음). 그래서 구간이 끝난 뒤의 duration_ms는 마지막 `ongoing_ms()`보다 보간한 만큼 길 수 있다.
- `blink.py:98` 모듈 설명 — `run_start_ms()`는 OPEN에서 0을 돌려준다(`ongoing_ms()`와 같다. 진행 중인 구간이 없으므로 뜻이 없는 값이다).
- `blink.py:99` 모듈 설명 — `t_ms`가 직전 값 이하인 호출은 조용히 넘기지 않고 ValueError로 거부한다(호출자가 버린다는 계약을 드러낸다).
- `blink.py:100` 모듈 설명 — `reset()`은 진행 중이던 구간을 발행하지 않고 버린다(10초 넘는 끊김 뒤에는 누적기도 함께 초기화된다).

**`drowsy/step_features.py`** (10개)

- `step_features.py:27` 모듈 설명 — 스텝 k의 특징은 t < 1000k인 프레임까지의 정보만 쓴다. "진행 중인 감김"(#9~#11)은 스텝을 발행시킨 프레임이 아니라 그 직전 프레임에서 받은 `ongoing_ms`다. 발행시킨 프레임의 값을 쓰면, 그 프레임에서 끝난 긴 감김이 스텝 k에서는 진행 중도 아니고 귀속되지도 않아(귀속은 스텝 k+1) 한 스텝 동안 사라진다. 프레임이 끊겼다가 와서 프레임 하나가 스텝을 여럿 발행할 때(10초 이하의 끊김)도 같은 규칙이다(검토 K5). 그 스텝들은 모두 끊기기 직전 마지막 프레임의 `ongoing_ms`를 쓴다. 끊긴 동안 감김이 어떻게 됐는지는 알 수 없으므로 늘리지도 0으로 되돌리지도 않는다(0으로 두면 감김이 끊긴 동안 사라졌다가 끊김 뒤에 잘린 구간으로 다시 나타난다). 눈을 감은 채 끊겼을 때의 결과는 다음과 같다. 동작은 바꾸지 않았고 문서로만 남긴다. · 끊긴 동안의 스텝마다 그 감김이 진행 중인 감김으로 들어간다(long_closure_count_60, max_closure_ms_60, max_closure_ms_10). 끊김 뒤 첫 프레임에서 검출기가 그 감김을 잘린 구간으로 내면 그 프레임이 속한 스텝의 통에 들어가고 진행 중인 값은 0이 되므로 두 번 세지는 않는다. 다만 max_closure_ms_10에는 10초가 아니라 끊긴 시간 + 10초 동안 남는다. · 끊긴 동안의 스텝에서는 10초 구간의 유효 시간이 줄어드는데 감긴 시간은 그대로 구간에 남아 perclos_10이 올라간다. 유효 시간이 `perclos_min_valid_ms_10`(2초) 아래로 내려가야 0이 된다. · 예(실행해서 확인한 값): 20.0초에 눈을 감고, 20.8초의 프레임 다음 프레임이 28.5초에 오면 스텝 21~28이 모두 그 프레임에서 발행되고 그 스텝들의 max_closure_ms_60과 max_closure_ms_10은 812, long_closure_count_60은 1이다. 잘린 구간(812 ms)은 스텝 29에 들어가고 max_closure_ms_10은 스텝 38까지 812다(스텝 39에서 0). perclos_10은 스텝 21의 0.100에서 스텝 29~30의 0.400까지 올랐다가 스텝 31에서 0이 된다.
- `step_features.py:45` 모듈 설명 — 누적을 시작하기 전의 시간(start_ms 이전, 초기화 이전)은 "데이터 없음"으로 센다. 그래서 조회 구간이 덜 찬 스텝의 face_missing_ratio는 1에 가깝고, 유효 시간이 모자란 동안 perclos 등은 정의 불가 값(0)이다.
- `step_features.py:47` 모듈 설명 — start_ms에는 "데이터 없음" 상태의 가상 프레임이 있다고 본다. 첫 프레임이 start_ms보다 늦게 오면 그 사이에 끝난 스텝은 빈 스텝으로 발행하고, 10초 넘게 늦으면 아래의 초기화 규칙을 따른다.
- `step_features.py:49` 모듈 설명 — 10초 넘는 끊김 뒤에는 끊긴 동안의 스텝을 발행하지 않는다(끊기기 직전에 채우던 스텝도 버린다). 새 프레임이 속한 스텝의 시작 경계를 새 누적 시작 시각으로 삼고 그 스텝부터 다시 발행한다. `full_lookback`도 새 누적 시작 시각 + 60초부터 다시 True가 된다. 스텝 번호 k는 세션 시간 그대로다.
- `step_features.py:52` 모듈 설명 — start_ms 이전의 프레임은 받아도 누적하지 않는다(빈 목록을 돌려준다).
- `step_features.py:53` 모듈 설명 — start_ms는 `clock.step_ms`의 배수여야 한다(T_b는 spec 검증으로 항상 배수다). 아니면 ValueError.
- `step_features.py:77` `ThresholdEventCounter` — 길이는 (지금 프레임 시각) − (enter 이상을 처음 본 프레임 시각)이다. 보간하지 않는다.
- `step_features.py:78` `ThresholdEventCounter` — 하품은 값이 exit 이상으로 유지되는 프레임에서만 센다. exit 미만으로 내려간 프레임에서는 길이가 기준을 넘었더라도 세지 않는다.
- `step_features.py:80` `ThresholdEventCounter` — 무효 구간은 무효 프레임이 없이 프레임 자체가 끊긴 시간도 포함한다(PLAN 3.3절).
- `step_features.py:165` `NodDetector` — dt는 직전 유효 표본과의 간격이고 `ema_max_dt_ms`(200 ms)를 넘으면 그 값으로 자른다.

**`drowsy/rules.py`** (18개)

- `rules.py:25` 모듈 설명 — 위 구간과 신호의 선택은 계획서의 "해당 구간의 유효 시간 비율"을 종류별로 풀어 쓴 것이다.
- `rules.py:26` 모듈 설명 — 끊김 초기화가 끝내는 drowsy `end`는 그 값을 실어 줄 스텝이 없으므로 confidence를 0.0으로 둔다.
- `rules.py:48` 모듈 설명 — `on_step`은 `update`보다 먼저 불리므로, 10초 넘는 끊김 뒤의 첫 프레임이 발행시킨 스텝이 있다면 그 스텝은 초기화 전의 판정기로 처리되고 그 뒤에 1번의 drowsy `end`가 온다. `DriverPipeline`에서는 그런 스텝이 값을 갖지 못한다(누적기가 같은 조건으로 다시 시작해 스텝이 없다. 누적기가 그 프레임에서 처음 프레임을 받는 경우에만 스텝이 나오는데 조회 구간이 덜 찬 빈 스텝이다). 그래서 끊김 뒤 첫 프레임의 스텝이 drowsy를 열거나 닫아 초기화 직전에 상태가 바뀌는 일은 없다.
- `rules.py:184` `ForwardZone` — 초기 상태는 영역 안이다.
- `rules.py:185` `ForwardZone` — 이어받기는 마지막 유효 관측이 영역 안이었을 때도 똑같이 한다(계획서는 영역 밖일 때만 적었다). 영역 밖 상태에서 잠깐 앞을 본 직후 얼굴이 사라지면 영역 안으로 확정된다.
- `rules.py:187` `ForwardZone` — 확정은 프레임이 도착할 때만 판정한다. 영역 밖 후보는 이어받은 표지로도 확정된다(고개를 빨리 돌려 250 ms가 차기 전에 얼굴이 사라지는 경우를 놓치지 않으려는 것이다).
- `rules.py:189` `ForwardZone` — 후보가 확정되지 못한 채 UNKNOWN 구간을 만나면 후보를 버리고, 후보 구간은 세지 않는다.
- `rules.py:190` `ForwardZone` — `out_source`는 마지막 유효 영역 밖 관측의 source다. 시선 이벤트는 시작할 때의 이 값을 source로 쓴다.
- `rules.py:348` `StepHysteresis` — 재시작 금지 시간 동안에도 연속 수는 센다. 금지 시간이 끝난 첫 스텝에서 연속 수가 이미 찼으면 그 스텝에서 시작하고, `start_ms`는 그 연속 구간의 첫 스텝 끝(금지 시간 안일 수 있다)이다.
- `rules.py:429` `RuleEngine` — 세션의 첫 프레임(끊김 초기화가 있었으면 그 뒤의 첫 프레임)부터 `window_ms`가 지나기 전에는 판정하지 않는다(창이 덜 차서 생기는 오판정을 막는다).
- `rules.py:438` `RuleEngine` — 표지가 UNKNOWN인 프레임에 영역 안 후보가 아직 걸려 있으면(`ForwardZone.candidate_pending`. 인정 구간이 끝나는 바로 그 시각에 프레임이 온 경우) 그 프레임에서는 판정하지 않는다. 후보 구간이 연속 시간에 섞여 잠깐 앞을 본 뒤의 끊김이 시선 이탈로 잡히는 것을 막는다. `t_start_ms`는 영역 밖 상태가 시작된 프레임. 영역 안이 확정된 프레임에서 끝난다.
- `rules.py:455` `RuleEngine` — `duration_ms`는 이벤트 공통 규칙대로 발행 시각 − `t_start_ms`다. 감김 도중에 반쯤 뜬 시간이 있었으면 시작 이벤트를 낸 누적 감김 시간(`ongoing_ms`)보다 길다(vats의 `duration_ms`가 영역 밖 시간의 합보다 긴 것과 같다).
- `rules.py:471` `RuleEngine` — PERCLOS 판정은 `full_lookback`인 스텝만 쓴다. 조회 구간이 덜 찬 스텝은 값이 없는 스텝으로 본다 (perclos_60이 60초 구간의 값일 때만 임계값과 비교한다).
- `rules.py:907` `AlertPolicy` — 같은 호출에서 통과한 경고는 모두 돌려준다(낮은 순위를 버리지 않는다). 앱은 첫 원소를 대표로 쓰면 된다.
- `rules.py:908` `AlertPolicy` — 억제되거나 대기에 걸린 `onset`은 재경고 대기를 새로 시작하지 않는다(대기는 실제로 낸 경고부터 센다).
- `rules.py:909` `AlertPolicy` — 미뤄 둔 종류의 `onset`이 `end` 없이 다시 오면(규칙 엔진은 그렇게 내지 않는다) 새 `onset`으로 바꾼다.
- `rules.py:910` `AlertPolicy` — sleep의 `onset`·`end`는 microsleep의 되풀이 기록도 함께 지운다(microsleep에 `repeat_ms`를 준 설정에서 감김이 끝난 뒤에도 microsleep 경고가 되풀이되는 것을 막는다. 기본 설정에서는 해당 없음).
- `rules.py:912` `AlertPolicy` — unable_to_analyze의 "속도 억제 없음"은 설정값(`min_speed_kmh: 0`)으로 표현한다.

**`drowsy/pipeline.py`** (5개)

- `pipeline.py:33` 모듈 설명 — 한 프레임의 이벤트는 스텝 이벤트(drowsy)를 먼저, 프레임 이벤트를 나중에 적는다 (스텝은 이 프레임보다 앞선 시간을 대표한다).
- `pipeline.py:35` 모듈 설명 — 10초 넘는 끊김 뒤의 첫 프레임에서 규칙 엔진의 초기화는 `RuleEngine.update` 안에서, 곧 그 프레임의 스텝을 처리한 뒤에 일어난다. 그 프레임의 이벤트는 초기화의 `end`(drowsy, unable_to_analyze, gaze_away_long, microsleep 또는 sleep 순), 그 프레임의 판정 순이다. 그 프레임이 발행시킨 스텝은 졸음 판정을 바꾸지 못한다: `StepAggregator`가 같은 끊김으로 다시 시작하므로 스텝이 없고, 누적기가 그 프레임에서 처음 프레임을 받는 경우 (끊김이 누적 시작 시각에 걸친 경우)에만 스텝이 나오는데 데이터가 없는 `full_lookback=False` 스텝이라 PERCLOS 판정에도 모델 창에도 값이 없다. 그래서 스텝 이벤트가 초기화의 drowsy `end`보다 앞에 오는 일은 없다.
- `pipeline.py:41` 모듈 설명 — 기준선을 주고 `feature_start_ms`를 주지 않으면 `baseline.t_complete_ms`를 쓴다.
- `pipeline.py:42` 모듈 설명 — `rules.clock`은 `spec.clock`과 같아야 한다(다르면 누적기와 규칙 엔진의 초기화 시점이 어긋난다). 아니면 ValueError.
- `pipeline.py:43` 모듈 설명 — 모델 입력에 유한하지 않은 값이 있으면 추론을 건너뛰고 센다(PLAN 4.6절: 유한하지 않으면 추론을 건너뛴다).

### 설정과 명세

**`drowsy/spec.py`** (6개)

- `spec.py:244` `GlobalConstants` — 계획서는 E, B, jaw0, R0만 정했다. gaze_h0과 gaze_v0은 0.0으로 둔다(대체 상태에서는 시선 규칙이 꺼진다).
- `spec.py:414` `PerclosDrowsyRules` — 계획서는 PERCLOS 판정의 재시작 금지 시간을 적지 않았다. 0(없음)으로 둔다.
- `spec.py:428` `DrowsinessRules` — 교차검증 기준(D11)을 통과하기 전에는 모델이 무작위 가중치이므로 기본값을 perclos로 둔다.
- `spec.py:448` `EventPolicy` — 표의 "없음"은 0으로 적는다. sleep의 "없음(3초마다 반복)"은 cooldown_ms = 0, repeat_ms = 3000이다.
- `spec.py:492` `RulesConfig` — 규칙 엔진은 이 설정만 받는다(`RuleEngine(cfg)`). 그래서 PipelineSpec에 있는 값 중 규칙 엔진이 쓰는 것(시간 규칙, 감김 검출기 설정)을 여기에도 둔다. 두 곳의 값이 같은지는 테스트가 검사한다.
- `spec.py:494` `RulesConfig` — 계획서의 경고 우선순위는 sleep, microsleep, gaze_away_long, vats, drowsy 다섯 종류만 정했다. alert_priority에는 그 뒤에 unable_to_analyze, calibrating을 두어 모든 종류를 싣는다.

**`drowsy/config.py`** (3개)

- `config.py:290` `_validate` — 학습 표본에서 두 증강의 몫을 뺀 나머지가 원본 추출의 몫이다. 합이 1을 채우면 원본에서 뽑을 것이 없다.
- `config.py:296` `_validate` — 이웃에 주는 확률이 0.5 이상이면 정답 클래스가 더는 가장 큰 목표가 아니다. 그 아래로만 받는다.
- `config.py:320` `_validate` — W 비교(`baseline_lgbm cv --all-w`)는 기준 W의 묶음을 주 결과로, 나머지를 그에 대한 부가 결과로 싣는다.

**`config/rules.yaml`** (2개)

- `rules.yaml:68` — `refractory_ms: 0` — 계획서는 PERCLOS 판정의 재시작 금지 시간을 적지 않았다. 0 = 없음
- `rules.yaml:92` — 계획서의 순서에 없는 unable_to_analyze와 calibrating은 그 뒤에 둔다.

**`config/default.yaml`** (15개)

- `default.yaml:25` — `rotation_check_frames: 12` — 추출의 모든 입력(--video, --input-dir, --dataset, --unlabeled)과 python -m drowsy.demo --video가 하는 영상별 회전 자동 점검(12장)에 쓰는 표본 프레임 수
- `default.yaml:26` — `rotation_check_interval_ms: 2000` — 표본 프레임 사이의 간격. 12장 x 2초면 영상 앞 24초를 본다
- `default.yaml:27` — `rotation_roll_margin_deg: 15.0` — 회전 점검에서 얼굴 검출 비율이 비슷한 각도들의 |roll 중앙값| 차이가 이 값 미만이면 가르지 못한 것으로 표시한다(리포트의 rotation_ambiguous)
- `default.yaml:31` — `latency_ms_mean: 66.7` — 프레임당 처리 지연 평균. 약 15 fps(1000 / 15)가 되게 한 값. 실기기 로그를 받으면 바꾼다(1.3절)
- `default.yaml:32` — `latency_ms_std: 8.0` — 지연의 표준편차
- `default.yaml:37` — `train_stride: 5` — 학습 보폭(초). LightGBM의 학습 창과 리포트의 창 수에 쓴다. 가정: 신경망은 영상의 보폭 1 창에서 균등하게 뽑는다(보폭 격자에서 시작 위치를 무작위로 옮기는 것과 같은 분포)
- `default.yaml:51` — `ordinal_neighbor_mass: 0.2` — 부가 실험(PLAN 11장 E3)의 순서형 소프트 라벨: 정답 클래스가 이웃 클래스에 나눠 주는 확률(끝 클래스는 하나뿐인 이웃에 전부, 가운데 클래스는 양쪽에 절반씩). 주 실행은 쓰지 않고 python -m drowsy.train ... cv --ordinal만 읽는다(라벨 스무딩 대신 쓴다)
- `default.yaml:55` — `fps_sim_share: 0.25` — 증강 (1): 누락 모사 추출(raw_dir/fps15 또는 synth_dir + _fps15)의 스텝 표가 있으면 학습 표본의 이 비율을 거기서 뽑는다. 없으면 쓰지 않고 그 몫은 원본 추출로 간다. 그 추출이 원본의 세션을 다 덮지 못하면 학습을 시작하지 않는다(--allow-partial-fps-sim으로 풀 수 있다)
- `default.yaml:56` — `frame_drop_share: 0.25` — 증강 (2): 프레임 누락 변형(원본 추출의 프레임을 frame_drop_min~max 비율로 버리고 다시 계산한 스텝 표)에서 뽑는 학습 표본의 비율. 변형이 여럿이면 똑같이 나눈다. 두 비율의 합은 1보다 작아야 하고 나머지(기본 0.5)가 원본 추출의 몫이다. 변형에 창이 없는 영상의 몫은 원본으로 간다. 신경망 학습(train cv, train final)만 섞고 LightGBM은 섞지 않는다
- `default.yaml:57` — `frame_drop_variants: 1` — 증강 (2)의 변형 수(변형 번호 0부터, 시드 0. python -m drowsy.dataset --frame-drop-variants 0이 만드는 것과 같은 캐시). 0이면 만들지 않고 그 몫은 원본 추출로 간다. 학습이 처음 돌 때 변형마다 전체 세션을 한 번 더 재생해 캐시에 둔다
- `default.yaml:73` — `learning_rate: 0.05` — 계획서에 없는 값
- `default.yaml:74` — `max_rounds: 300` — 부스팅 횟수 상한. 실제 횟수는 내부 검증으로 고른다
- `default.yaml:75` — `early_stopping_rounds: 30` — 계획서에 없는 값. 내부 검증 성능이 이 횟수 동안 나아지지 않으면 멈춘다
- `default.yaml:86` — `alarm_threshold_sweep: [0.5, 0.6, 0.7, 0.8, 0.9]` — "임계값을 바꿔 본 곡선"의 p_drowsy 시작 임계값. 주 수치는 rules.yaml의 0.6
- `default.yaml:113` — `parity_random_inputs: 200` — 일치 검증에 쓰는 무작위 입력 수

### 영상 입출력과 추출

**`drowsy/video_io.py`** (4개)

- `video_io.py:110` `display_matrix_has_flip` — 부가 정보의 정수는 libav가 메모리에 둔 그대로(기계의 바이트 순서)다.
- `video_io.py:145` `iter_video_frames` — 표시 행렬에 좌우·상하 뒤집기가 들어 있는 파일(첫 프레임으로 판정한다)은 rotation을 주지 않으면 VideoError다. libav가 알려 주는 회전은 뒤집기를 무시한 값이라, 좌우 뒤집기만 있는 똑바로 선 영상을 180도 돌려 거꾸로 세우기 때문이다. rotation을 주면 그 각도로만 돌리고 거울상은 되돌리지 않으며 `counts.display_flip`에 표시한다(모델 특징은 좌우 대칭이어서 거울상은 견디지만 거꾸로 선 얼굴은 못 견딘다).
- `video_io.py:149` `iter_video_frames` — 첫 프레임에 pts가 없으면 파일 전체를 프레임 번호 / 평균 프레임률 시계로 읽는다. 첫 프레임에는 pts가 있는데 뒤 프레임에 없으면 그 프레임은 버리고 센다(두 시계를 섞지 않는다).
- `video_io.py:200` `probe_video` — 패킷의 pts 집합이 디코딩된 프레임의 pts 집합과 같다고 본다(정상 파일에서 성립한다. 디코더가 버리도록 표시된 패킷은 뺀다). 추출이 실제로 본 프레임 수는 추출 결과의 통계에 따로 남는다.

**`drowsy/extract.py`** (34개)

- `extract.py:26` 모듈 설명 — 계획서는 저장할 내용만 정했고 열 이름, 자료형, 메타데이터 키는 정하지 않았다. 위 형식으로 고정하고 형식을 바꾸면 SCHEMA_VERSION을 올린다. 읽기는 `read_raw_frames`만 쓰면 되므로 다른 모듈은 열 이름에 의존하지 않는다.
- `extract.py:28` 모듈 설명 — 녹화 요약은 열이 아니라 메타데이터 키로 적는다. 표의 열은 그대로이므로 SCHEMA_VERSION을 올리지 않고 (합성 데이터와 `read_raw_frames`가 그대로 유효하다) 메타데이터 내용의 판은 `extract_version`으로 따로 센다.
- `extract.py:275` `FlowLimiter` — 프레임은 자기 타임스탬프 시각에 도착하고, 처리 시간은 호출자가 주는 지연 값이다(실제 추론 시간이 아니다).
- `extract.py:276` `FlowLimiter` — 처리가 끝나는 시각과 프레임 도착 시각이 정확히 같으면 처리가 먼저 끝난 것으로 본다.
- `extract.py:277` `FlowLimiter` — 입력이 끝났을 때 대기 중인 프레임은 처리된다(`flush`).
- `extract.py:338` `drop_sim_latencies` — 평균 latency_ms_mean, 표준편차 latency_ms_std인 정규분포에서 뽑고 음수는 0으로 자른다. 난수열은 drop_sim.seed만으로 정해지므로 모든 영상이 같은 지연 수열을 쓴다.
- `extract.py:548` `frame_brightness` — 밝기 = (0.299·ΣR + 0.587·ΣG + 0.114·ΣB) / (H·W). BT.601 가중치를 디코더가 낸 8비트 RGB 값(감마가 적용된 값)에 그대로 쓴다. 선형 광량으로 바꾸지 않고, 얼굴 영역이 아니라 MediaPipe에 넘긴 프레임 전체를 본다. 채널별 합을 정수로 구하므로 같은 프레임이면 값이 언제나 같다. 화소는 저장하지 않는다(PLAN 1.2절의 결정 D13).
- `extract.py:638` `extract_video` — 수동 회전 보정 값은 `frame.rotation`과 같은 규약(세우려면 반시계로 돌려야 하는 각도)이다.
- `extract.py:639` `extract_video` — 길이·프레임률·간격은 원본 영상의 것이다. 누락 모사 추출에서도 모사가 버리기 전의 프레임 시각으로 구한다 (저장된 행의 간격은 `t_ms` 열에서 다시 구할 수 있다). 밝기는 실제로 처리한 프레임만 본다.
- `extract.py:641` `extract_video` — 처리 시간은 메타데이터에 적지 않는다(같은 영상을 다시 추출하면 파일이 바이트 단위로 같아야 한다).
- `extract.py:728` `output_is_current` — 파일이 있고, 같은 모델 파일과 mediapipe 버전, 같은 누락 모사 설정, 같은 수동 회전 보정, 같은 랜드마크· 블렌드셰이프 목록, 같은 메타데이터 판(`EXTRACT_VERSION`)으로 만들었고, 결과에 적힌 원본 파일 크기 (`source_size_bytes`)가 source의 지금 크기와 같으면 최신으로 본다. 원본 경로는 비교하지 않는다(데이터 폴더를 옮겨도 다시 추출하지 않는다). 크기가 다르면 이름만 같은 다른 영상이거나 바뀐 파일이므로 다시 추출한다. 크기가 적혀 있는데 source를 읽을 수 없으면(파일이 없다) 최신으로 보지 않는다(추출을 시도해 그 오류가 드러나게 한다).
- `extract.py:733` `output_is_current` — 원본의 내용은 크기로만 비교한다(해시를 구하지 않는다. 이어 하기마다 모든 영상을 끝까지 읽지 않기 위해서다). 그래서 크기가 바이트 단위로 같은 다른 영상은 가려내지 못한다(예: 같은 영상에 회전 메타데이터만 바꿔 단 파일). 그런 영상으로 바꿨다면 결과 파일을 지우고 다시 실행한다.
- `extract.py:736` `output_is_current` — 원본 파일 크기가 적혀 있지 않은 결과(이 키를 적기 전의 추출 결과)는 전과 같이 다룬다: 크기를 비교하지 않고 나머지 조건만으로 최신 여부를 정한다. source를 주지 않아도 크기를 비교하지 않는다.
- `extract.py:835` `sample_frames` — 첫 프레임을 고르고, 그 뒤로는 직전에 고른 프레임보다 interval_ms 이상 늦은 첫 프레임을 고른다. 필요한 만큼 고르면 수열을 더 읽지 않는다(영상을 끝까지 디코딩하지 않는다).
- `extract.py:905` `choose_rotation` — 실제 랜드마커는 옆으로 누운 얼굴과 거꾸로 선 얼굴도 검출하므로(공개 시험 이미지 2장으로 잰 것) 검출 비율만으로는 방향을 가를 수 없다. 거꾸로 선 얼굴이 검출되면 똑바로 선 얼굴로 잘못 맞춰져 roll이 정방향과 몇 도 안에서 같게 나온다. 그런 영상에서 이 점검은 고른 각도와 그 반대 각도(0도와 180도)를 가르지 못한다. 그때는 가른 것처럼 다루지 않고 `ambiguous`로 표시해 사람이 확인하게 한다.
- `extract.py:909` `choose_rotation` — 3번에서 회전 메타데이터가 가르지 못한 각도들 안에 없을 때(메타데이터의 각도는 roll이 ±90 근처여서 이미 틀린 것으로 드러난 경우) 메타데이터를 따르지 않고 |roll|이 가장 작은 각도를 쓴다. 추출하려면 각도 하나를 정해야 해서 고른 것일 뿐 근거가 약하다(맞는 각도이거나 그 반대 각도다). 그래서 `ambiguous`로 표시되고 리포트의 맨 앞 목록에 오른다.
- `extract.py:913` `choose_rotation` — 어느 각도에서도 얼굴을 못 찾은 영상(4번)도 `ambiguous`로 표시한다. 점검이 아무것도 확인하지 못했기 때문이다.
- `extract.py:947` `check_rotation` — 표본은 영상 앞부분에서만 뽑는다(interval_ms 간격으로 n_frames장). 영상 전체를 디코딩하지 않기 위해서다.
- `extract.py:1069` `_job_current_metadata` — 회전을 자동으로 점검하는 작업은, 같은 점검 규칙 판(`ROTATION_CHECK_VERSION`)과 같은 점검 설정으로 자동 점검을 거쳐 만든 결과가 있으면 최신으로 본다(점검을 다시 돌려 각도를 비교하지 않는다. 같은 영상의 점검 결과는 같다). 점검 규칙이나 설정이 바뀌면 다시 추출한다. 점검 기록이 없는 결과(회전 메타데이터만 써서 만든 것, 수동 값으로 만든 것)도 다시 추출한다.
- `extract.py:1073` `_job_current_metadata` — 각도를 `--rotation`으로 준 작업은 수동 값으로 만든 결과(점검 기록이 없고 각도가 같은 것)만 최신으로 본다. 자동 점검으로 만든 결과는 각도가 같아도 다시 추출한다. `--rotation`은 그 실행의 명령줄에만 있으므로, 사람이 각도를 확인했다는 사실을 결과 파일에 남기는 길이 그것뿐이기 때문이다. 보정표로 준 각도는 설정 파일에 남으므로 각도만 같으면 최신으로 보고, 그때 파일의 점검 기록은 그대로 남는다(`output_is_current`).
- `extract.py:1103` `_skip_note` — "다르다"는 절대 경로로 바꾼 글자열이 다르다는 뜻이다. 결과에 상대 경로가 적혀 있으면 지금의 실행 위치를 기준으로 푼다(명령은 저장소 루트에서 실행한다고 본다). 심볼릭 링크는 풀지 않으므로, 같은 파일을 가리키는 다른 경로도 다르다고 알린다. 데이터 폴더를 옮긴 뒤의 이어 하기에서는 건너뛴 영상마다 옛 경로가 붙는다.
- `extract.py:1147` `raw_parquet_path` — 계획서는 raw_dir 아래의 배치를 정하지 않았다. 확장자만 다른 두 영상(`a.mp4`, `a.mov`)은 같은 경로가 되므로 명령이 시작 전에 오류로 알린다.
- `extract.py:1159` `unlabeled_raw_dir` — 라벨 없는 영상의 결과를 라벨 있는 데이터셋의 결과와 같은 폴더에 섞지 않는다. `--fps-sim`에 설정과 다른 프레임률을 주면 추출은 `unlabeled_fps{FPS}`에 쓰므로 이 함수가 가리키는 폴더와 달라진다(`drop_sim_for_fps`).
- `extract.py:1169` `drop_sim_for_fps` — FPS가 설정의 지연 평균이 뜻하는 프레임률(1000 / latency_ms_mean을 반올림한 값)과 같으면 설정을 그대로 쓴다. 다르면 지연 평균을 1000 / FPS로, 저장 폴더를 `fps{FPS}`로 바꾸고 표준편차와 시드는 유지한다.
- `extract.py:1199` `check_override_keys` — 맞지 않는 키를 조용히 무시하지 않는다. 보정표는 회전이 틀린 영상을 고치는 유일한 수단이어서, 대소문자나 폴더 깊이를 틀리게 적은 키가 무시되면 그 영상은 틀린 각도로 추출되고 실행은 성공으로 끝난다.
- `extract.py:1478` `_row_warnings` — `low_face_rate`는 프레임 기준 얼굴 검출 비율이 영상 품질 기준(PLAN 1.2절의 결정 D7: 얼굴 검출 비율과 창의 수로 평가 대상 영상을 거른다)의 값(`evaluation.quality_gate.min_face_ratio`) 미만일 때 붙인다. 그 기준 자체는 시간 기준 비율이고 평가 때 계산하므로, 이 표시는 미리 알려 주는 용도일 뿐 영상을 제외하지 않는다.
- `extract.py:1533` `PreviousTimings.lookup` — 두 리포트에 그 영상의 값이 다 있으면 나중에 쓴 리포트의 값을 쓴다. "나중"은 이번 실행이 리포트를 다시 쓰기 전에 읽은 두 CSV 파일의 수정 시각(ns)으로 정한다(같은 폴더의 두 파일이므로 같은 시계다). 시험 추출은 결과가 있어도 다시 추출하므로, 전체 추출 뒤에 시험 추출을 다시 했다면 지금 있는 결과 파일은 시험 추출이 만든 것이고 그 처리 시간은 시험 추출 리포트에 있다.
- `extract.py:1537` `PreviousTimings.lookup` — 두 리포트의 값이 글자까지 같으면 시험 추출에서 잰 값으로 본다(전체 추출의 리포트는 건너뛴 영상의 값을 시험 추출 리포트에서 옮겨 적으므로). 이어 하기는 이 리포트를 다시 써서 수정 시각을 새로 만들지만 옮겨 적은 값은 그대로이므로, 이어 하기를 몇 번 거듭해도 값과 출처가 바뀌지 않는다. 따로 잰 두 값이 유효 숫자 6자리까지 우연히 같으면 출처만 시험 추출로 적힌다(값은 같다).
- `extract.py:1585` `_report_rows` — 처리 시간은 결과 파일에 없으므로, 이번에 추출한 영상은 이번 실행의 값을, 건너뛴 영상(up_to_date)은 전에 쓴 리포트(같은 이름의 리포트, `--dataset`이면 같은 폴더의 시험 추출 리포트도)에 그 영상의 값이 있고 프레임 수가 같을 때 그 값을 쓴다(둘 다 있으면 `PreviousTimings.lookup`의 규칙). 없으면 비워 둔다. 이번 실행이 추출했는데 처리 시간을 받지 못한 영상(실행이 끊김)에는 전의 값을 잇지 않는다: 전 리포트의 값은 지금의 결과 파일을 만든 시간이 아니다.
- `extract.py:2001` `write_extraction_report` — 파일 이름의 tag는 `<입력 종류>_<추출 폴더 이름>`이다(dataset_native, pilot_native, dataset_fps15, input_dir_native, unlabeled_native 등). 원본 추출, 누락 모사 추출, 시험 추출의 리포트가 한 실행명 아래에서 서로 덮어쓰지 않게 하기 위해서다.
- `extract.py:2241` `main` — "앞 N개"는 목록 순서(피험자, 라벨, 부분)다. 결과는 전체 추출과 같은 자리에 쓰므로 전체 추출이 이어받는다.
- `extract.py:2249` `main` — 설정의 rotation_overrides 키는 `--input-dir`이나 데이터셋 루트 기준 상대 경로(슬래시 구분)이고, `--video`에서는 파일 이름이다. 값은 `frame.rotation`과 같은 규약(세우려면 반시계로 돌려야 하는 각도)이며 영상의 회전 메타데이터와 자동 점검 결과를 대신한다. `--video`의 `--rotation`도 같은 규약이고, 보정표에 그 영상이 있어도 `--rotation`이 이긴다(명령에 직접 준 값이 설정 파일보다 우선한다). `--rotation`은 그 실행에만 적용된다: 다음에 `--rotation` 없이 실행하면 자동 점검으로 다시 추출한다. 이번 입력의 어느 영상과도 맞지 않는 보정표 키가 있으면 시작하지 않는다 (`check_override_keys`).
- `extract.py:2255` `main` — 회전 자동 점검은 네 입력(`--video`, `--input-dir`, `--dataset`, `--unlabeled`) 모두에서 한다. 휴대전화로 찍은 영상은 회전 메타데이터가 없거나 틀릴 수 있고, 실제 랜드마커는 옆으로 누운 얼굴도 검출하므로(얼굴 검출 비율 1.0) 회전 메타데이터만 믿고 추출하면 틀린 방향의 신호가 아무 표시 없이 나온다. 회전 메타데이터만 써서 만든 전의 결과 (자동 점검 기록이 없는 `--video`, `--input-dir` 결과)는 최신으로 보지 않고 다시 추출한다.
- `extract.py:2259` `main` — `--unlabeled`의 보정표를 `rotation_overrides`와 따로 둔다. 한 설정 파일에 데이터셋과 라벨 없는 영상의 보정 값을 함께 적어도 서로를 "맞지 않는 키"로 거부하지 않게 하기 위해서다.

### 합성 데이터와 테스트 벡터

**`drowsy/synth.py`** (9개)

- `synth.py:156` `frame_from_signals` — 요청값의 범위는 검사하지 않는다(EAR ≥ 0, 눈 폭 > 0, |pitch| < 90을 호출자가 지킨다).
- `synth.py:238` `mirror_frame` — `frame.lm`은 `spec.landmarks.required_indices()`의 인덱스만 담고 있어야 한다(다른 인덱스는 짝을 모른다).
- `synth.py:290` `FaceProfile` — 구간에 적는 수준은 "기준선이 이 얼굴의 뜬 눈 값일 때의 closure"로 정의한다(`lid_signals`). 쉬는 자세가 0이 아니면 `HeadTurn`의 각도는 기준선 대비 각도(Euler(R0ᵀ·R))와 조금 다를 수 있다.
- `synth.py:346` `Blink` — 모양은 삼각형으로 정했다. 감김 검출기가 재는 길이(0.5를 넘는 시간)는 depth가 1일 때 duration_ms의 절반이다.
- `synth.py:463` `Scenario` — 각본 시나리오의 신호에는 잡음을 넣지 않는다(기대값을 손으로 계산할 수 있게). 잡음 있는 세션은 통계 생성기가 만든다.
- `synth.py:718` `generate_session` — 상태별 수치는 문헌의 방향(졸리면 깜빡임이 길어지고 긴 감김, 하품, 끄덕임이 늘어난다)만 따른 설계값이다. 실제 사람의 분포를 재현하지 않는다. 이 데이터의 점수는 파이프라인 점검용이다.
- `synth.py:934` `split_subjects` — 피험자가 49명 이상이면 RLDD와 같은 32번과 49번이다. 그보다 적으면 가운데 번호와 마지막 번호를 쓴다 (피험자가 한 명이면 그 한 명).
- `synth.py:962` `session_jobs` — 나뉜 부분은 각각 다른 세션과 같은 길이의 독립된 녹화다(부분마다 창이 나올 수 있는 길이가 되게).
- `synth.py:990` `simulate_device_drops` — 합성 프레임에는 MediaPipe의 시간 평활이 없으므로 프레임을 빼는 것이 누락 모사의 전부다 (실제 추출에서는 빠진 프레임 때문에 평활 결과도 달라진다).

**`drowsy/vectors.py`** (17개)

- `vectors.py:44` 모듈 설명 — 기준선을 넣어 줄 때 특징 누적을 시작할 세션 시간은 `baseline.t_complete_ms`로 전한다(별도 항목을 두지 않는다).
- `vectors.py:45` 모듈 설명 — config_overrides는 규칙 설정(`RulesConfig`)에만 적용한다. `pipeline` 블록의 상수는 pipeline_hash로 고정된 값이므로 벡터에서 바꾸지 않는다.
- `vectors.py:47` 모듈 설명 — 스텁 모델은 i번째 호출(0부터)에 probs[i]를 돌려주고, 목록이 끝나면 마지막 값을 되풀이한다.
- `vectors.py:48` 모듈 설명 — 모델을 스텝마다 부르게 되면서(2026-11-13 소유자 결정) 스텝별 확률을 `step_probs` 항목으로 따로 적는다. `probs` 항목의 뜻과 모양은 그대로 둔다(프레임의 마지막 스텝). 프레임이 스텝을 하나씩만 발행하는 벡터에서는 `step_probs`에서 null이 아닌 원소를 모은 것이 `probs`와 같다.
- `vectors.py:51` 모듈 설명 — 경계값 벡터의 일부는 각본 구간 대신 프레임의 시각과 값을 직접 정해서 만든다(`_hand_frames`). 정수 ms의 경계 (정확히 250 ms, 1000 ms, 10000 ms, 10001 ms)에 프레임을 놓으려는 것이다. 값은 계단 모양으로 바뀐다.
- `vectors.py:53` 모듈 설명 — 실수로 계산되는 값의 임계(closure, 각도, 눈동자 합성값)는 등호 자리를 벡터로 고정하지 않는다. 그 자리의 표본은 수학 함수의 마지막 자리 차이로 갈리므로 두지 않고(아래), 임계의 바로 안쪽과 바깥쪽 값으로 양쪽을 고정한다. 모델 확률의 종료 임계(0.4)도 같다: 모델 출력은 float32이고 0.4는 float32로 정확히 적을 수 없어서(0.4보다 큰 값이 된다), 스텁이 Double을 돌려주는 구현과 Float를 돌려주는 구현이 "이하"의 등호 자리에서 갈린다. 등호를 고정하는 것은 정수 ms의 비교, 입력으로 그대로 들어오는 값(차량 속도 20.0, 50.0), 정수 ms의 몫으로 정확히 나오는 값(perclos_60 0.15, 0.10), 그리고 시작 임계와 같은 p_drowsy 0.6이다(0.6을 float32로 적으면 0.6보다 커서 어느 구현에서든 시작 조건이므로 벡터에 둬도 옳은 구현이 떨어지지 않는다).
- `vectors.py:60` 모듈 설명 — 긴 세션(기준선, 스텝 특징, 졸음 판정, 전 과정)은 파일 크기를 줄이려고 3~10 fps의 프레임을 쓴다. 시간 누적은 프레임률과 무관하게 정의돼 있고(프레임 간격에서 `clock.max_hold_ms`(200 ms)를 넘는 부분만 "데이터 없음"이다), 프레임률에 따른 감김 검출은 감김 검출 묶음(30 fps, 15 fps, 누락)이 따로 고정한다.
- `vectors.py:63` 모듈 설명 — 이벤트의 confidence 허용 오차(절대 1e-4)는 계획서에 없어서 모델 확률의 허용 오차와 같게 정했다.
- `vectors.py:64` 모듈 설명 — 모델 묶음의 입력은 배정밀도로 적고, 텐서에 넣을 때 float32로 바꾼다(앱이 스텝 특징을 넣는 방식과 같다).
- `vectors.py:65` 모듈 설명 — 생성 결과가 바이트 단위로 같은 것은 같은 환경(같은 수학 라이브러리와 onnxruntime)에서의 일이다.
- `vectors.py:66` 모듈 설명 — 옳게 이식한 코드가 다른 기기에서도 같은 결과를 내도록, 판정이 갈리는 자리에 걸친 값은 벡터에 두지 않는다. 판정 임계값에서 1e-6 안에 있는 표본이 생기는 각본은 만들 때 거부하고(`_threshold_problem`), 보간한 시각이 반올림 경계(x.5 ms)에 붙지 않는지는 tests/test_vectors.py가 확인한다. 경계값 벡터는 정수 ms의 비교로 경계를 잡는다 (예: 감김 시작에서 정확히 3000 ms 뒤의 프레임).
- `vectors.py:70` 모듈 설명 — 기본 설정에서는 드러나지 않는 동작을 고정하려고 규칙 설정을 덮어쓴 벡터가 있다. `rules_gap_over_10s_drowsy_model`은 `drowsiness.model.refractory_ms`를 600000으로 둔다(10초 넘는 끊김 뒤에 모델 창이 다시 차려면 179초가 걸려서, 기본 60초로는 재시작 금지 기억이 지워졌는지 드러나지 않는다). 덮어쓴 값은 그 벡터 안에서만 쓰인다.
- `vectors.py:73` 모듈 설명 — `runs_two_humps`의 "같은 최댓값이 다시 나온 표본"은 같은 내용의 프레임을 두 번 쓴 것이라 어느 구현에서든 두 표본의 closure가 같다. 임계값과의 비교가 아니므로 위의 "판정이 갈리는 자리에 값을 두지 않는다"에 걸리지 않는다.
- `vectors.py:75` 모듈 설명 — sleep의 end가 미뤄 둔 microsleep 경고를 지우는 규칙은 벡터에 없다. 규칙 엔진은 sleep의 onset을 먼저 내고 그때 이미 지워지므로 엔진을 거친 이벤트로는 그 자리를 만들 수 없다. 대신 다른 종류의 end가 그 경고를 지우지 않는다는 것을 `rules_alert_pending_survives_end`가, 다른 종류의 onset도 지우지 않는다는 것을 `rules_alert_pending_survives_onset`이 고정한다.
- `vectors.py:79` 모듈 설명 — 첫 t_ms가 작은 세션(`rules_unable_small_t0`)의 첫 t_ms는 5000으로 정했다. 0보다 크고(벡터의 t_ms는 양수다) `unable_to_analyze.window_ms`(10000)보다 작아서, "처음 10초"를 첫 프레임이 아니라 t_ms 0부터 재는 구현이 5초 일찍 판정을 시작하는 값이다. 첫 t_ms가 1인 다른 벡터에서는 그 차이가 1 ms라 드러나지 않는다.
- `vectors.py:82` 모듈 설명 — 끄덕임의 느린 평균을 고정하는 벡터(`steps_nod_slow_mean`)는 끄덕임이 시작되는 프레임의 d = pitch_rel − m이 시작 임계 10도에서 0.028도 위에 오도록 pitch(25.25도)를 골랐다. m의 갱신 속도가 1%만 달라도(α = dt/τ로 구한 경우 d가 9.984) 끄덕임이 시작되지 않게 하려는 것이다. 등호 자리가 아니고 수학 함수의 마지막 자리 차이(1e-12 수준)보다 훨씬 멀어서 옳게 이식한 코드는 어느 기기에서든 같은 쪽에 놓인다. d는 저장되는 값이 아니라서 `_threshold_problem`이 보는 값에 들지 않는다. 손으로 계산한 값은 tests/test_vectors.py에 있다.
- `vectors.py:87` 모듈 설명 — 손으로 놓은 보정 세션 셋(`baseline_odd_sample_count`, `baseline_burst_frames`, `baseline_exactly_300_frames`)은 신호를 몇 가지 값으로만 둔다(분위수와 중앙값을 손으로 셀 수 있게). `baseline_burst_frames`의 얼굴 검출 비율 0.5는 정수 ms의 몫(45000 / 90000)으로 정확히 나오는 값이라 등호를 고정하고, `baseline_exactly_300_frames`는 프레임 수의 등호 (정확히 300개)를 고정한다. E의 범위 끝(0.12, 0.45)의 등호는 실수 임계라 벡터로 고정하지 않았다(단위 테스트가 고정한다).

### 데이터 구성과 E0 점검

**`drowsy/dataset.py`** (36개)

- `dataset.py:23` 모듈 설명 — 계획서는 식별자의 모양을 정하지 않았다. 위 형식으로 고정한다(피험자 번호는 두 자리 이상으로 0을 채운다).
- `dataset.py:143` `official_fold` — 60번을 넘는 피험자 번호(합성 데이터)에도 같은 식을 쓴다. 1 미만이면 ValueError.
- `dataset.py:218` `build_manifest` — `_N`은 세 라벨 모두에서 받는다(RLDD에는 10에만 있다).
- `dataset.py:219` `build_manifest` — `10_1`만 있고 `10_2`가 없으면 복사가 덜 된 것으로 보고 `ManifestError`를 낸다(영상의 절반만으로 논리 영상 하나를 만들지 않는다). 부분이 셋 이상일 때 마지막 부분이 빠진 것은 파일 이름만으로는 알 수 없어 검사하지 못한다.
- `dataset.py:229` `build_manifest` — 피험자 수는 검사하지 않는다(합성 데이터나 부분 사본도 받는다). 파일은 열어 보지 않는다.
- `dataset.py:371` `ReplayResult` — 계획서는 재생 결과로 스텝, 감김 구간, 이벤트, 기준선, 진단만 적었다. 품질 표(D7)에 필요한 프레임 계수와 얼굴 시간을 같은 재생에서 함께 센다(파이프라인이 낸 `FrameState.face`와 `t_ms`만 쓴다).
- `dataset.py:373` `ReplayResult` — 얼굴 검출 비율(시간 기준)의 분모는 받아들인 첫 프레임과 마지막 프레임 사이의 시간이다 (프레임이 끊긴 시간도 분모에 들어간다. 마지막 프레임은 뒤 프레임이 없어 시간을 대표하지 않는다).
- `dataset.py:416` `replay_video` — 계획서의 시그니처에는 규칙 인자가 없지만 `DriverPipeline`은 규칙 설정을 요구한다. 선택 인자로 받고 주지 않으면 저장소의 기본 규칙을 쓴다.
- `dataset.py:502` `frame_drop_plan` — 계획서는 "프레임을 무작위로 10~30% 버린다"만 적었다. 비율은 세션마다 한 번 균등 분포에서 뽑고, 프레임은 서로 독립으로 버린다(연속 누락을 따로 흉내 내지 않는다. 기기 동작의 모사는 증강 (1)이 맡는다).
- `dataset.py:504` `frame_drop_plan` — 난수는 넘파이가 아니라 표준 라이브러리의 메르센 트위스터를 쓴다(정수 시드의 `random()` 수열은 파이썬 판이 바뀌어도 유지된다고 문서가 보장한다).
- `dataset.py:622` `build_step_table` — 계획서의 시그니처는 표 하나를 돌려주지만, 스텝 표와 영상 품질 표를 `StepTable`로 묶어 돌려준다.
- `dataset.py:623` `build_step_table` — "쓸 수 있는 alert 기준선"은 alert 영상의 보정이 대체 없이 완성된 것(status가 ok 또는 off_axis)이다.
- `dataset.py:624` `build_step_table` — alert 영상이 나뉘어 있으면 첫 부분의 기준선을 피험자의 alert 기준선으로 쓴다.
- `dataset.py:625` `build_step_table` — 품질 기준의 "창 수"는 보폭 1에서 세고, 얼굴 검출 비율은 보정 구간을 포함한 영상 전체에서 잰다.
- `dataset.py:626` `build_step_table` — 추출 결과가 하나도 없으면 raw_dir이 틀린 것으로 보고 `ManifestError`를 낸다.
- `dataset.py:1018` `make_windows` — 이어진 구간마다 첫 창은 그 구간의 첫 window_steps개이고(앱이 처음 확률을 내는 시점과 같다) 그 뒤로 stride 스텝씩 옮긴다. 학습의 "시작 위치 무작위 이동"은 보폭 1로 만든 창에서 골라 쓰면 된다(`WindowSet.subset`).
- `dataset.py:1094` `nested_splits` — fold가 3개 미만이면 안쪽 분할의 학습 집합이 비므로 ValueError다.
- `dataset.py:1152` `code_fingerprint` — 지문은 소스 파일의 바이트다. 주석이나 문서 문자열만 바꿔도 캐시를 다시 만든다(안전한 쪽으로 틀린다). 특징을 바꾸지 않는 모듈(`rules`, `spec`, `extract`, `dataset`)도 재생 경로에 있어서 넣었다. 실행 중에 코드를 바꿔치기한 것(monkeypatch)이나 파이썬·넘파이 판이 바뀐 것은 알아채지 못한다. 그럴 때는 `--rebuild`를 쓴다.
- `dataset.py:1167` `step_cache_dir` — 계획서는 캐시의 위치를 정하지 않았다. raw_dir 안에 두지 않는 이유는 raw_dir이 `build_manifest`가 훑는 폴더일 수 있기 때문이다(합성 데이터).
- `dataset.py:1215` `raw_content_digest` — 한 프로세스가 도는 동안에는 추출 결과를 그런 식으로 바꾸지 않는다고 본다(추출과 학습은 서로 다른 명령이다). 바꿨다면 그 프로세스에서는 `rebuild=True`로 다시 만든다. 새로 실행한 명령은 스스로 알아챈다.
- `dataset.py:1217` `raw_content_digest` — 해시는 `code_fingerprint`와 같은 sha256이다(표준 라이브러리). 비용은 파일 크기에 비례한다. 합성 데이터 (`data/synth`, 182개 파일 443 MB)에서 2026-11-14에 이 PC로 잰 값: 지문 전체(`_source_fingerprint`)의 프로세스 첫 호출이 약 0.2초(파일들을 운영체제의 페이지 캐시에서 내보낸 뒤에는 약 0.4초), 그 뒤의 호출이 약 6 ms.
- `dataset.py:1321` `load_step_tables` — 캐시의 열쇠는 pipeline_hash와 조건이다(지시 사항). 여기에 데이터가 바뀐 것을 알아채기 위한 지문과, 특징을 만드는 코드가 바뀐 것을 알아채기 위한 코드 지문을 더했다(상수와 추출 결과가 그대로여도 코드를 고치면 옛 특징으로 학습하게 되는 것을 막는다). 추출 결과는 프로세스마다 한 번 파일 전체를 읽어 sha256을 구한다. 같은 프로세스가 도는 동안 크기와 수정 시각을 그대로 둔 채 내용을 바꾼 파일만 알아채지 못하고, 그럴 때는 `rebuild`를 쓴다.
- `dataset.py:1353` `load_frame_drop_tables` — 실제 기기에서 프레임이 빠지면 MediaPipe의 추적 결과(랜드마크와 블렌드셰이프 값)도 조금 달라지지만, 여기서는 남은 프레임의 저장된 값을 그대로 쓴다. 누락도 프레임마다 독립이라 기기의 누락 무늬와 다르다. 기기 동작을 그대로 재현하는 것은 증강 (1)(흐름 제한기를 흉내 내 다시 추출한 것)이고, 이것은 그 보조다.
- `dataset.py:1357` `load_frame_drop_tables` — 기준선과 T_b는 솎지 않은 원본 프레임에서 구한다. 조건마다 기준선을 대는 세션(자기 세션 또는 alert 영상)을 솎지 않고 앱처럼 보정해 얻은 기준선을, 솎은 프레임의 재생에 넣고 그 세션 자신의 (솎지 않은) T_b부터 특징을 누적한다. 그래서 원본 표와 다른 것은 특징에 들어가는 프레임뿐이고, 기준선 열과 누적 시작 시각은 원본 표와 같다.
- `dataset.py:1360` `load_frame_drop_tables` — 스텝 번호는 원본과 같은 격자지만 스텝의 집합은 조금 다를 수 있다. 마지막 프레임들이 버려지면 끝 스텝이 하나 덜 나오고, 원래 10초에 가깝던 끊김이 10초를 넘으면 누적이 초기화된다. 원본 표와 행을 맞추지 않는다.
- `dataset.py:1362` `load_frame_drop_tables` — 전역 조건은 spec의 전역 상수를 쓴다(`global_override`는 받지 않는다. fold별 상수 점검은 원본 표만 쓴다).
- `dataset.py:1363` `load_frame_drop_tables` — 누락 모사 추출(15 fps)의 표를 다시 솎는 것도 막지 않는다. raw_dir이 가리키는 추출 결과를 그대로 솎는다.
- `dataset.py:1364` `load_frame_drop_tables` — 변형의 품질 표에서 프레임 수와 얼굴 시간은 솎은 재생의 값이다(그 표가 실제로 재생한 것을 적는다). 스텝이 없는 세션(추출 결과 없음, 보정 미완)은 솎지 않고 `drop_rate`를 NaN으로 둔다.
- `dataset.py:1574` `read_rotation_checks` — "수동 보정이 있다"는 것은 보정표에 키가 있고 파일이 그 각도로 추출돼 있다는 뜻이다. 추출은 자동 점검이 고른 각도와 같은 각도를 보정표에 적으면 파일을 다시 만들지 않으므로(점검 기록이 그대로 남는다) 파일만으로는 사람이 확인했는지 알 수 없다. 그래서 보정표를 함께 받는다. 반대로 보정표의 각도와 파일의 각도가 다르면 사람이 정한 각도가 아직 적용되지 않은 것이므로 확인 필요로 올린다.
- `dataset.py:1578` `read_rotation_checks` — 보정표의 키와 파일은 확장자를 뗀 상대 경로로 맞춘다(추출 결과 폴더에는 영상의 확장자가 남지 않는다. 확장자만 다른 두 영상은 추출이 이미 거부한다). 다른 폴더의 추출 결과를 넘기면 키가 맞지 않아 보정이 없는 것으로 본다.
- `dataset.py:1580` `read_rotation_checks` — 읽을 수 없는 파일은 "기록 없음"에 센다(회전이 점검됐는지 알 수 없다는 뜻에서 같다).
- `dataset.py:1784` `open_dataset` — parquet 폴더를 `rldd_root`로 주는 것 자체는 막지 않는다(실제 추출 결과만 복사해 둔 배치일 수 있다).
- `dataset.py:1785` `open_dataset` — 있는데 추출 결과로 읽을 수 없는 파일(메타데이터 없음, 깨진 파일)도 여기서 `ManifestError`로 알린다.
- `dataset.py:1891` `baseline_window_summary` — 계획서는 "기준선이 연장된 영상과 조건별·클래스별 창 수를 리포트에 싣는다"만 적었다. 항목은 세션(파일) 단위로 싣는다. 나뉜 영상은 부분마다 T_b가 따로이기 때문이다(`video_id`로 묶을 수 있다).
- `dataset.py:1953` `main` — 회전이 확인되지 않은 영상이 있어도 그 영상을 빼지 않고 스텝 표를 만들며 종료 코드는 0이다. 알리기만 한다 (영상을 모두 직접 볼지, 더 엄격한 자동 점검을 넣을지는 소유자의 결정을 기다리는 항목이다).
- `dataset.py:1955` `main` — `--frame-drop-seed`의 기본값은 0이다. 캐시의 열쇠에 시드가 들어가므로, 여기서 미리 만든 변형을 학습이 쓰려면 학습 코드가 같은 시드와 변형 번호로 `load_frame_drop_tables`를 불러야 한다. 종료 코드: 0 성공, 2 시작할 수 없음(설정, 데이터 폴더, 추출 결과 문제).

**`drowsy/signal_check.py`** (26개)

- `signal_check.py:32` 모듈 설명 — E0의 판정 단위는 세션(파일)이다. 나뉜 영상의 부분(`10_1`, `10_2`)은 앱의 세션처럼 각자 자기 기준선으로 따로 재생되므로 따로 센다. RLDD라면 논리 영상 180개가 아니라 파일 182개가 분모다. 출력 파일 이름은 지시대로 per_video.csv이고 한 행이 세션 하나다.
- `signal_check.py:35` 모듈 설명 — `dataset.replay_video`는 프레임 상태를 돌려주지 않아 closure 유효 시간을 셀 수 없다. 그래서 같은 `DriverPipeline`을 프레임 하나씩 직접 부르는 `collect_session`을 둔다. 신호, closure, 감김 검출은 파이프라인이 낸 값만 쓴다(PLAN 3.1절). 겹치는 값(기준선, 감김 구간, 프레임 수, 얼굴 시간)이 `replay_video`와 같은지는 테스트가 본다.
- `signal_check.py:38` 모듈 설명 — 실제 데이터에서 통과 기준에 미달하면 종료 코드 4로 끝낸다(E1의 `baseline_lgbm cv`와 같은 방식. PLAN 12장: 미달 시 학습 전에 멈춘다). 합성 데이터에서는 파이프라인 점검일 뿐이므로 0이다.
- `signal_check.py:40` 모듈 설명 — 합성 데이터로 만든 제안 파일의 이름은 `feature_spec_proposed.synthetic.json`이고 "받아들인 뒤 실행할 것" 목록을 쓰지 않는다. spec 로더가 모르는 키를 거부해서 파일 안에 합성 표시를 넣을 수 없기 때문이다. 합성 세션에서 나온 상수가 `config/feature_spec.json`에 들어가면 안 된다.
- `signal_check.py:43` 모듈 설명 — `--fps-sim` 실행은 제안 파일을 쓰지 않는다. 전역 상수는 원본 프레임률 추출에서 정한다(PLAN 4.3절).
- `signal_check.py:303` `session_record` — 깜빡임은 특징용 검출기가 낸 잘리지 않은 구간 전부다(PLAN 4.6절의 "이벤트"). 길이로 거르지 않으므로 긴 감김도 들어간다.
- `signal_check.py:305` `session_record` — closure 유효 시간이 0이면 분당 횟수는 0이다(PLAN 3.3절: 0으로 나누게 되는 비율은 0). 기준선이 없는 세션은 특징 구간이 없으므로 분당 횟수와 깜빡임 통계가 NaN이다. 깜빡임이 없는 세션의 정점·길이 통계도 NaN이다.
- `signal_check.py:307` `session_record` — 사분위수는 최근접 순위, 중앙값은 PLAN 4.2절의 정의(짝수 개면 가운데 두 값의 평균)를 쓴다.
- `signal_check.py:409` `signal_table` — 추출 결과가 하나도 없으면 raw_dir이 틀린 것으로 보고 `ManifestError`를 낸다(`dataset.build_step_table`과 같다).
- `signal_check.py:457` `criterion_flags` — off_axis(거치 각도가 15도를 넘음)는 대체가 아니므로 완성으로 센다. 연장된 기준선(T_b > 60초)은 세지 않는다.
- `signal_check.py:458` `criterion_flags` — 경계는 포함한다(정확히 5회, 40회, 0.8이면 만족).
- `signal_check.py:459` `criterion_flags` — 값이 없는 세션(추출 결과 없음, 보정이 끝나기 전에 끝남, 깜빡임 없음)은 해당 조건을 만족하지 못한 것이다.
- `signal_check.py:477` `evaluate_criteria` — 세션이 하나도 없으면 비율은 0이고 미달이다.
- `signal_check.py:533` `constant_medians` — 세션별 통계는 E = (E_l + E_r) / 2, B = (B_l + B_r) / 2, jaw0이다(전역 상수는 두 눈에 같은 값을 쓴다).
- `signal_check.py:534` `constant_medians` — 개인 기준선이 완성된 세션(status ok 또는 off_axis. 연장된 것 포함)만 쓴다. 전역 상수로 대체된 세션의 값은 전역 상수 자신이라 넣으면 순환이 되고, 보정이 끝나지 않은 세션에는 값이 없다.
- `signal_check.py:536` `constant_medians` — 중앙값은 PLAN 4.2절의 정의다(짝수 개면 가운데 두 값의 평균).
- `signal_check.py:573` `global_constants` — 상수는 소수 넷째 자리로 반올림한다(`floor(x·10⁴ + 0.5) / 10⁴`). 설정 파일과 SPEC.md의 표에 같은 값을 그대로 적기 위해서다. 세션 사이의 차이에 비하면 무시할 크기다. 반올림 전 값은 `constant_medians`가 준다.
- `signal_check.py:647` `build_proposed_spec` — `global_constants.e`, `.b`, `.jaw0`을 덮어썼어도 제안 파일에는 constants(세션별 통계의 중앙값)가 들어간다. 덮어쓴 전역 상수는 평가에만 쓰인다.
- `signal_check.py:649` `build_proposed_spec` — `spec_version`은 `pipeline` 블록이 바뀔 때만 1 올린다. 바뀐 것이 없으면 그대로다.
- `signal_check.py:790` `run_e0` — 통과 기준은 "지금 spec + 덮어쓴 값"으로 판정한다. 제안 파일은 여기에 전역 상수까지 바꾼 것이다. 전역 상수는 기준선 완성 전의 임시 closure와 전역 상수로 대체된 세션에만 쓰이므로, 대체된 세션이 없으면 판정이 같다. 대체된 세션이 있으면 제안 상수를 `--override`로 넣어 한 번 더 돌려 확인하라고 요약에 적는다.
- `signal_check.py:793` `run_e0` — 회전이 확인되지 않은 영상이 있어도 통과 기준의 판정과 종료 코드는 바꾸지 않는다(그 영상을 빼지도 않는다). 리포트와 출력으로 알리기만 한다. 영상을 볼지, 더 엄격한 점검을 넣을지는 소유자의 결정을 기다리는 항목이다. 회전 점검 기록은 라벨과 무관하다: 목록의 `rel_path`와 추출 결과 파일, 설정의 수동 회전 보정표만 읽는다.
- `signal_check.py:1317` `BlinkShape` — 실제 눈꺼풀 궤적의 측정값이 아니다. 감는 쪽이 뜨는 쪽보다 빠르고(전체의 35%) 양 끝과 정점에서 속도가 0인 매끄러운 모양으로 정했다. 머무는 구간은 없다. 뜬 눈의 closure는 0이고 잡음과 MediaPipe의 시간 평활은 넣지 않았다.
- `signal_check.py:1463` `simulate_blinks` — 검출은 "잘리지 않은 구간이 하나 나왔다"로 센다. 프레임이 500 ms 넘게 끊겨 잘린 구간만 나온 시행은 검출되지 않은 것으로 세고 `n_truncated`에 적는다.
- `signal_check.py:1465` `simulate_blinks` — 길이 격자는 enter 넘게 감긴 시간으로 60~400 ms, 진폭은 0.7, 0.85, 1.0, 위상은 50개, 누락 확률은 0.2 (PLAN 5.2절의 10~30%의 가운데), 누락 패턴은 위상마다 4번이다. 합친 행은 격자의 모든 조합에 같은 비중을 준다. 실제 깜빡임 길이의 분포가 아니므로 합친 값은 격자에 따라 달라진다.
- `signal_check.py:1741` `main` — `--fps-sim` 실행은 참고용 점검이라 기준에 미달해도 0으로 끝낸다(판정은 원본 프레임률 추출로 한다).
- `signal_check.py:1742` `main` — 회전이 확인되지 않은 영상이 있어도 종료 코드는 바꾸지 않는다. 한 줄로 알리고 목록은 리포트에 싣는다.

### 모델, 학습, 평가

**`drowsy/models.py`** (3개)

- `models.py:154` `ExportWrapper` — 온도는 인자로 받은 값을 쓴다. spec의 `standardization.temperature`와 같은지는 내보낼 때 `drowsy.export_onnx.export`가 검사한다.
- `models.py:281` `load_checkpoint` — v1 체크포인트는 어떤 feature_spec으로 학습했는지와 합성 데이터 여부를 담고 있지 않아 확인할 방법이 없으므로, 경고만 하고 읽는 대신 거부한다.
- `models.py:283` `load_checkpoint` — `fast` 키가 없는 v2 체크포인트(이 키를 더하기 전에 쓴 파일)는 축소 설정이 아닌 것으로 읽는다(`fast=False`). 그 파일이 실제로 축소 설정으로 학습한 것이었는지는 파일만으로 알 수 없다. 키가 있으면 불이어야 한다.

**`drowsy/train.py`** (35개)

- `train.py:152` `TEMPERATURE_MIN` — 계획서에 없는 값이다. 내부 검증 예측이 모두 맞으면 음의 로그 가능도가 온도 0 쪽으로 계속 내려가므로 끝을 둔다.
- `train.py:176` `CONSTANTS_NOTE` — 이 문장은 절차를 적은 것이다. feature_spec의 상수가 실제로 E0(`python -m drowsy.signal_check e0`)를 거친
- `train.py:187` `FRAME_DROP_SEED` — 시드는 0으로 고정한다(`python -m drowsy.dataset --frame-drop-variants 0`의 기본 시드와 같아 그 명령이 만든
- `train.py:299` `ordinal_targets` — 계획서는 "이웃 클래스에 확률을 나눠 주는 방식"이라고만 했다. 나눠 주는 크기(`training.ordinal_neighbor_mass`, 0.2)와 끝 클래스의 규칙은 여기서 정한 것이다. 균일 라벨 스무딩은 함께 쓰지 않는다(이것이 그 자리를 대신한다).
- `train.py:325` `VideoMacroF1` — 조기 종료와 선택의 "내부 검증 macro-F1"은 리포트의 주 지표와 같은 영상 가중 값이다(창을 그대로 합친 값이 아니다).
- `train.py:390` `standardization_stats` — 영상들 사이의 비중은 같다. 피험자마다 영상 수가 다르면(영상이 빠진 피험자) 표집기의 "피험자 → 영상" 비중과 조금 다르다. 받은 창만 쓴다. 검증·시험 피험자의 창을 넘기지 않는 것은 호출하는 쪽(`fit`)의 일이고 `fit`은 학습 창만 넘긴다.
- `train.py:447` `fit_temperature` — 창을 그대로 합쳐(영상 가중 없이) 맞춘다. 보정 전후 ECE도 창을 그대로 합쳐 계산한다.
- `train.py:448` `fit_temperature` — 탐색 범위는 0.1 ~ 10이다(계획서에 없는 값). 예측이 모두 맞거나 모두 틀리면 범위의 끝이 나온다.
- `train.py:526` `BalancedSampler` — "학습 보폭 5초, 시작 위치 무작위 이동"(PLAN 5.2절)은 영상의 보폭 1 창을 균등하게 뽑는 것으로 구현한다. 보폭 격자의 칸을 고른 뒤 시작 위치를 0 ~ 보폭 − 1 스텝 옮기는 것과 같은 분포이고, 구간 끝의 창이 더 자주 뽑히는 치우침이 없다. 그래서 이 표집기는 `windows.train_stride`를 쓰지 않는다(그 값은 LightGBM의 학습 창과 리포트의 창 수에 쓰인다). 창 묶음은 보폭 1로 만든 것을 받는다. 다른 보폭의 묶음을 주면 받은 창들에서 균등하게 뽑는다.
- `train.py:530` `BalancedSampler` — 표집 대상인 피험자와 영상은 첫 묶음(원본 추출)에 창이 있는 것이다. 창이 없는 영상과 피험자, 나머지 묶음에만 창이 있는 영상은 뽑지 않는다.
- `train.py:757` `fit` — 내부 검증 macro-F1이 같으면 먼저 나온(짧은) 단위를 남긴다.
- `train.py:1008` `score_candidate` — 계획서는 분할별 학습 길이를 하나로 합치는 방법을 적지 않았다. 중앙값(짝수 개면 가운데 두 값의 평균을 반올림)을 쓴다.
- `train.py:1010` `score_candidate` — 후보의 점수는 분할마다 "그 분할에서 가장 좋았던 단위"의 macro-F1을 평균한 값이다(조기 종료의 결과 그대로).
- `train.py:1056` `choose` — "내부 fold 표준편차"는 평균이 가장 높은 후보의 fold별 값의 표본 표준편차다.
- `train.py:1176` `mix_sources` — 몫의 기본값(원본 0.5, 15 fps 모사 0.25, 프레임 누락 변형 0.25)은 계획서에 없는 값이다.
- `train.py:1248` `print_rotation_check` — 알리기만 하고 멈추지 않는다. 회전이 확인되지 않은 영상을 빼거나 학습을 막는 것은 소유자의 결정을 기다리는 항목이다 (`python -m drowsy.dataset`도 알리기만 한다).
- `train.py:1250` `print_rotation_check` — 원본 프레임률 추출의 기록만 읽는다. 15 fps 누락 모사 추출은 같은 영상을 같은 점검 설정으로 한 번 더 추출한 것이라 기록이 같을 것으로 본다. 두 추출 사이에 보정표를 고치고 한쪽만 다시 추출한 경우는 여기서 드러나지 않는다(그 추출의 기록은 `python -m drowsy.dataset --fps-sim`이 출력한다).
- `train.py:1304` `load_cv_data` — 합성 데이터에서는 fold 수를 보지 않는다(테스트와 연기 시험은 피험자 몇 명, fold 두세 개로 돈다).
- `train.py:1308` `load_cv_data` — 프레임이 덜 들어온 추출에서는 보정이 늦게 끝나 원본에서만 창이 나오는 영상이 정상적으로 있을 수 있다. 그런 경우에도 명시적으로 allow_partial_fps_sim을 주게 한다(조용히 넘어가지 않는다).
- `train.py:1311` `load_cv_data` — 변형은 이 호출이 원본 추출의 같은 파일에서 같은 목록으로 만든다. "덜 끝난 추출 폴더"가 있을 수 없으므로 세션 수를 대조할 이유가 없다. 솎은 뒤 창이 나오지 않는 세션(끝 스텝이 빠져 조회 구간이 모자라게 된 짧은 세션 등)은 있을 수 있고, 그 영상의 표본은 원본에서 뽑는다(`BalancedSampler`). 덮은 세션 수는 설명에 적는다.
- `train.py:1540` `cross_validate` — 내보내기나 양자화가 실패하면(계약 검사 포함) 그 사실을 notes에 적고 두 묶음을 모두 버린 채 주 결과는 끝까지 낸다. 부가 비교 하나 때문에 긴 학습의 결과를 잃지 않게 한다.
- `train.py:1550` `cross_validate` — 주 결과 행이 고른 후보가 그 계열의 계열별 선택과 같으면 다시 학습하지 않고 같은 예측을 `selected`로도 싣는다.
- `train.py:1551` `cross_validate` — 안쪽 선택의 학습은 시드 하나만 쓴다(6장은 다시 학습할 때의 시드 수만 정했다).
- `train.py:1853` `step_table_fingerprint` — 지문은 `STEP_COLUMNS` 전체(식별자, 스텝 번호, 특징 17개)를 표의 행 순서대로 넣는다. 목록의 세션 순서가 바뀌어도 값이 바뀐다(같은 데이터를 다른 순서로 읽은 것도 "다른 표"로 본다).
- `train.py:1917` `write_run_files` — 같은 폴더의 `config.yaml`과 `rules.json`(`evaluate.write_reports`가 쓴다)은 설정 파일의 바이트 사본이라 데이터 종류 표시가 없다. rules.json은 바이트의 해시가 rules_hash라서 한 줄도 더할 수 없다.
- `train.py:1946` `check_run_dir` — 같은 identity의 실행을 다시 돌리는 것(같은 조건의 재실행)은 덮어쓴다. 추출을 다시 했거나 코드를 고친 뒤의 재실행이 그런 경우다.
- `train.py:2000` `_reference_lgbm` — 지문이나 데이터 종류 표지가 없는 예전 형식의 `lgbm/`(이 검사들이 생기기 전에 만든 것)은 대조할 수 없으므로 받지 않는다. `baseline_lgbm cv`를 다시 돌리면 된다.
- `train.py:2101` `main_run_predictions` — 주 실행의 리포트가 있는데 대조에 실패하면(추출이나 코드가 바뀐 뒤 주 실행을 다시 돌리지 않은 경우) 짝 없이 진행하지 않고 멈춘다. 부가 실행의 목적이 주 실행과의 비교이기 때문이다. 주 실행이 아직 없으면 짝 없이 돈다.
- `train.py:2214` `run_nested_cv` — 주 결과(E2)는 내보낼 조건(`training.export_arm`, alert 영상 기준선)으로 학습한 실행 하나다. 다른 조건의 실행은 다른 실행 이름(폴더)에 쓰이고(`check_run_dir`), 그 리포트에는 주 결과 묶음이 없다.
- `train.py:2226` `run_nested_cv` — 설정 키 `export.int8_if_tcn`(최종 모델의 INT8 파일을 만들지)이 fold 모델의 INT8 비교도 함께 켜고 끈다. 따로 키를 두지 않았다. fold 모델의 내보내기는 임시 폴더에서 하고 파일을 남기지 않는다.
- `train.py:2235` `run_nested_cv` — 부가 실행끼리 겹친 조합(순서형 라벨 모델의 15 fps 평가 등)은 만들지 않는다. 계획서가 든 부가 분석은 넷이고 (나머지 정규화 조건, 속도 특징 제외, 순서형 라벨, 15 fps 모사 평가) 각각 주 실행과 하나만 다르게 둔다.
- `train.py:2237` `run_nested_cv` — 계획서 11장 E3의 "설정은 E2 것을 고정"을 "탐색 범위, 시드, 절차를 E2와 같게 둔다"로 읽는다. E2가 fold마다 고른 후보를 그대로 가져오지 않고, 바깥 fold마다 안쪽 선택을 다시 한다(선택 사항이 바뀌면 맞는 학습 길이와 온도가 달라지고, 그 값들을 시험 fold 없이 정하는 길은 안쪽 선택뿐이다).
- `train.py:2614` `fit_final` — `paths.rules`(config/rules.yaml)의 내용을 그대로 쓴다. D11 판정 결과에 따라 `drowsiness.source`를 바꾸는 일은 하지 않는다(규칙 파일을 고치는 것은 rules_hash를 바꾸는 결정이라 사람이 한다).
- `train.py:2629` `fit_final` — `trained_on`에 축소 설정 표시를 넣지 않는다(메타데이터 값의 형식을 바꾸지 않는다). 대신 위치를 막고 `final.json`의 `fast`에 적는다.
- `train.py:2818` `with_data_root` — `--data-root`는 설정이 가리키는 데이터 루트를 바꾼다. `paths.rldd_root`가 있으면 그것(영상 폴더)을, null이면 `paths.synth_dir`(합성 데이터 폴더)을 바꾼다. 이 옵션으로 합성 데이터 설정을 실제 데이터로 바꿀 수는 없다. 실제 데이터로 돌리려면 `config/default.yaml`을 git에 넣지 않는 `config/local.yaml`로 복사해 `paths.rldd_root`에 경로를 넣고 `--config config/local.yaml`을 준다. 추적되는 기본 설정은 고치지 않는다.

**`drowsy/baseline_lgbm.py`** (12개)

- `baseline_lgbm.py:105` `RECORDING_COLUMNS` — 가변 프레임률 표시(`variable_fps`)는 계획서 6장의 목록(평균 프레임률, 해상도, 얼굴 크기, 얼굴 검출률, 밝기)에
- `baseline_lgbm.py:116` `PROBE_MIN_DATA_IN_LEAF` — 교란 점검의 모델 크기는 계획서에 없다. 영상 180개짜리 표에 맞춰 작게 고정한다(선택에 쓰지 않으므로 탐색하지 않는다).
- `baseline_lgbm.py:330` `choose_lgbm` — LightGBM에는 "파라미터 수"가 없다. 잎이 적을수록, 잎 수가 같으면 `min_data_in_leaf`가 클수록 단순한 것으로 본다.
- `baseline_lgbm.py:379` `lgbm_cross_validate` — 신경망은 영상의 보폭 1 창을 균등하게 뽑아 보지만 LightGBM에는 에폭이 없으므로 학습 보폭 격자 위의 창만 쓴다.
- `baseline_lgbm.py:380` `lgbm_cross_validate` — 원본 프레임률 추출의 창으로만 학습한다. 신경망은 15 fps 누락 모사 추출과 프레임 누락 변형이 있으면 학습 표본의 일부를 거기서 뽑지만(증강 (1), (2)) 여기서는 섞지 않는다. 계획서 5.3절의 "같은 창"과 다른 점이고, 비교표와 D11의 기준 LightGBM은 그 증강 없이 학습한 것이다. 리포트 설명과 명령의 출력에 적는다(`NATIVE_ONLY_NOTE`).
- `baseline_lgbm.py:602` `e1_gate` — "내부 검증 macro-F1 평균"은 바깥 fold마다 고른 후보의 안쪽 fold 평균을 다시 바깥 fold에 대해 평균한 값이다. 시험 fold의 값은 쓰지 않는다.
- `baseline_lgbm.py:637` `run_lgbm_cv` — 기준 실행은 "closure W를 바꾸지 않은 실행"이다. E0에서 feature_spec의 W를 바꾸게 되면 `lgbm.reference_w`를 어떻게 둘지(계획서의 0.5를 지킬지, 확정한 W를 따를지)는 사람이 정해 설정에 적어야 하고, 그 전에는 시작하지 않는다.
- `baseline_lgbm.py:904` `run_lgbm_w_grid` — 계획서는 "W 세 가지 비교는 보고용"이라고만 했다. 기준 조건 하나에서만 돌리고, 기준 W가 아닌 묶음을 부가로 표시하며, E1 통과 기준(종료 코드 4)은 기준 실행에만 적용한다. 이 명령은 기준 미달이어도 멈추지 않는다.
- `baseline_lgbm.py:906` `run_lgbm_w_grid` — 영상 표(분모와 품질 판정)는 기준 W의 스텝 표에서 만든다. closure W는 기준선 완성 시각과 얼굴 검출 비율을 바꾸지 않으므로 W마다 같다(다르면 예측 표의 검사가 gate_pass 불일치로 멈춘다).
- `baseline_lgbm.py:1142` `_state_medians` — "뜬 눈 closure"는 특징용 감김 구간에 들지 않은(진입 임계값 미만) 프레임의 closure다.
- `baseline_lgbm.py:1194` `_recording_values` — 누락 모사 추출에서 메타데이터의 평균 프레임률은 모사가 버리기 전 원본 영상의 것이다. 교란 점검은 원본 프레임률 추출을 읽으므로 두 정의의 차이는 디코딩에 실패한 프레임뿐이다.
- `baseline_lgbm.py:1356` `confound_probe` — "우연 수준을 크게 넘는다"를 95% 구간의 아래 끝이 1/3을 넘는 것으로 정한다.

**`drowsy/evaluate.py`** (43개)

- `evaluate.py:52` 모듈 설명 — 계획서는 표의 모양을 정하지 않았다. `temperature` 열은 "보정 전후 ECE"를 예측 표만으로 계산하려고 둔 것이다 (softmax(z/T)를 T제곱해 정규화하면 softmax(z)가 된다. `untemper`).
- `evaluate.py:54` 모듈 설명 — 계획서는 부가 분석을 "부가로 표시"하라고만 했다(11장 E3). 표지 열 셋과 그 기본값, `_int8` 접미사는 여기서 정한 것이다.
- `evaluate.py:69` 모듈 설명 — 확률이 정확히 같으면 클래스 순서가 앞선 쪽이다.
- `evaluate.py:70` 모듈 설명 — 0으로 나누게 되는 비율(영상이 없는 클래스의 재현율, 평가 시간이 0일 때의 시간당 횟수 등)은 0.0이다. PLAN 3.3절이 특징에 정한 규칙을 지표에도 쓴다.
- `evaluate.py:72` 모듈 설명 — 표준편차는 표본 표준편차(n − 1로 나눔)이고 값이 하나뿐이면 NaN이다(리포트에는 n/a).
- `evaluate.py:335` `validate_predictions` — 표지가 둘 다 기본값인 묶음의 부가 여부는 표지로 알 수 없다(ONNX로 다시 돌린 예측, INT8 양자화 모델의 예측, 내보낼 조건이 아닌 정규화 조건의 실행은 계열 이름이나 실행 이름만 다르다). 그런 묶음은 적혀 있는 supplementary를 그대로 따르고, 값이 없으면 주 결과(False)로 본다. 부르는 쪽이 True를 직접 적어야 한다.
- `evaluate.py:338` `validate_predictions` — gate_pass는 feature_source마다 한 값이면 된다(다른 추출의 품질 표에서 온 값이 다를 수 있다). 리포트에서는 어느 묶음이든 `write_reports`에 넘긴 영상 표의 gate_pass를 쓴다(`_per_video`).
- `evaluate.py:471` `prediction_table` — supplementary를 주지 않으면(None) feature_source나 variant가 기본값이 아닐 때 True, 둘 다 기본값이면 False다. 표지를 달아 놓고 부가 표시를 잊은 묶음이 주 결과에 섞이지 않게 한다. 계열 이름만 다른 부가 묶음 (INT8)은 True를 직접 줘야 한다. 표지가 기본값이 아닌데 False를 직접 주면 ValueError다(`validate_predictions`).
- `evaluate.py:551` `video_table` — 계획서는 "기준선이 연장된 영상을 리포트에 싣는다"고만 했다. 나뉜 영상의 값을 합치는 방식(가장 늦은 T_b, 상태를 `/`로 이음)은 여기서 정한 것이다.
- `evaluate.py:774` `video_vote` — 표가 같으면 그 클래스들 중 영상 전체 창의 평균 확률이 큰 쪽이다. 가정: 그것도 정확히 같으면 클래스 순서가 앞선 쪽.
- `evaluate.py:791` `two_class` — 한 창에서 두 확률이 정확히 같으면 alert다. 영상의 표가 같을 때는 `video_vote`처럼 두 확률의 평균으로 가른다.
- `evaluate.py:792` `two_class` — D7의 분모 규칙을 여기에도 쓴다. videos를 주면 창이 없는 alert·drowsy 영상도 분모에 들어가고 오답이다. 돌려주는 표는 `video_vote`와 같은 열이고 low_vigilant 영상의 행이 없다(`votes_low_vigilant`는 항상 0). 정확도는 `video_metrics(votes, classes=TWO_CLASSES)`로 구한다.
- `evaluate.py:893` `AlarmReport` — 스텝을 처리한 뒤 판정 중이면 그 스텝을 판정 상태로 센다(시작한 스텝은 들어가고 종료한 스텝은 빠진다).
- `evaluate.py:896` `AlarmReport` — 시간으로 나눈 값이다. 가정: D9는 모델의 값만 정의한다. perclos와 either의 정의는 나란히 싣기 위해 맞춘 것이다.
- `evaluate.py:943` `alarm_videos` — 예측 표의 행(창이 나온 스텝)만 평가한다. 앱에서 PERCLOS 판정은 조회 구간이 찬 첫 스텝부터 돌아 모델보다 119스텝 먼저 시작하지만, 여기서는 세 판정원을 같은 평가 시간에서 비교한다.
- `evaluate.py:945` `alarm_videos` — 평가 시간은 행 수 × `clock.step_ms`다(보폭 1의 예측을 전제한다). 한 세션에 행이 둘 이상인데 한 스텝 간격인 행 쌍이 표 전체에 하나도 없으면 보폭이 1이 아닌 표로 보고 ValueError를 낸다(판정이 영원히 나지 않는 표다).
- `evaluate.py:962` `alarm_videos` — 건너뛴 크기는 보지 않는다. 파이프라인이 낼 수 없는 작은 건너뜀(손으로 만든 표, 행을 솎아 낸 표)도 같은 다시 시작으로 다룬다.
- `evaluate.py:980` `alarm_videos` — 앱은 창에 유한하지 않은 값이 있으면 추론을 건너뛰고 그 스텝을 값이 없는 스텝으로 본다. 예측 표의 행에는 언제나 확률이 있으므로 여기에는 그런 스텝이 없다(스텝 특징은 입력이 유한하면 유한하다. `drowsy.step_features`).
- `evaluate.py:1146` `alarm_sweep` — 계획서는 무엇을 바꾸는 곡선인지 적지 않았다. 시작 임계값만 바꾸고 종료 임계값과 스텝 수, 재시작 금지 시간은 설정 그대로 둔다. source는 model 또는 either다. perclos는 이 임계값과 무관하므로 임계값을 하나라도 주면 ValueError다. thresholds가 비어 있으면 빈 표다.
- `evaluate.py:1269` `subject_bootstrap` — 계획서는 구간의 방식을 적지 않았다. 백분위수 구간(선형 보간)을 쓰고, 피험자는 fold와 무관하게 추출한다.
- `evaluate.py:1310` `expected_calibration_error` — 계획서에 구간 수가 없다. 기본은 같은 폭 15개(`ECE_BINS`)다.
- `evaluate.py:1649` `evaluate_group` — 부트스트랩 구간은 "fold 평균"이 아니라 "fold를 합친 값"의 구간이다(피험자를 fold와 무관하게 추출한다). fold 크기가 같으면 정확도는 두 값이 같고 macro-F1은 조금 다를 수 있어 표에 두 값을 함께 싣는다.
- `evaluate.py:1651` `evaluate_group` — "시드 간 표준편차"는 시드마다 fold별 값을 평균한 뒤 그 평균들의 표준편차다. 시드마다 예측이 있는 fold가 다르면 ValueError다(시드 평균이 서로 다른 fold를 섞게 된다).
- `evaluate.py:1653` `evaluate_group` — 영상 표에 `n_windows` 열이 있고 어떤 fold의 영상이 모두 창 0개이면, 그 fold에 예측이 없는 것은 평가하지 않은 것이 아니라 창이 없는 것이다. 그 fold는 받아들이고 영상을 모두 오답으로 센다(D7. 학습 코드가 창이 없는 시험 fold를 건너뛰는 경우). `n_windows` 열이 없으면 예측이 없는 fold는 언제나 거부한다. 표지 열이 없는 표(검사를 거치지 않은 예전 형식)도 받는다. 그 묶음의 표지는 기본값이다.
- `evaluate.py:1779` `gate_inputs` — 계획서는 D11의 "교차검증 결과"가 어느 집계인지 적지 않았다. 분류 지표(창 단위 영상 가중 macro-F1, 2클래스 영상 정확도)는 리포트의 대표값인 fold 평균(시드 평균)을, 오경보 수치(시간당 횟수, 영상 비율)는 6장이 주 수치로 정한 "5개 fold를 합친 값"(시드 평균)을 쓴다.
- `evaluate.py:2277` `supplementary_pairs` — 계획서는 부가 분석의 비교 상대를 적지 않았다. "다른 점 하나를 되돌린 묶음"이라는 규칙은 여기서 정한 것이다.
- `evaluate.py:2509` `write_reports` — "첫 줄에 합성 데이터임을 적는다"를 summary.md뿐 아니라 모든 표 파일에 적용한다. CSV는 첫 줄이 `# 데이터 종류` 주석이므로 `pd.read_csv(path, skiprows=1)`로 읽고, JSON은 첫 키가 `data_note`다.
- `evaluate.py:2511` `write_reports` — 영상 표에 있더라도 fold_subjects에 없는 fold의 영상은 평가 대상이 아니다(일부 fold만 평가한 리포트).
- `evaluate.py:2512` `write_reports` — 문헌 행의 값과 주석은 PLAN 2.4절의 것이다. 2.4절은 Xu 2025의 86.5%가 몇 클래스 값인지 적지 않았다. 조사 기록(영상 단위 3클래스 정확도)에 따라 3클래스 열에 둔다.
- `evaluate.py:2514` `write_reports` — 부가 묶음도 주 결과의 영상 표를 분모와 품질 판정(gate_pass)으로 쓴다. 다른 추출(15 fps 모사)에서 창이 나오지 않은 영상은 그 묶음에서 오답이다(D7을 그대로 적용).
- `evaluate.py:2516` `write_reports` — "기준선이 연장된 영상"은 T_b가 기본 수집 시간(60초)보다 큰 영상이다. 전역 상수로 대체된 영상과 alert 영상 기준선을 쓰지 못한 영상(alert_fallback)도 같은 목록에 싣는다(계획서는 연장된 영상만 적었다).
- `evaluate.py:2907` `analysable_ms` — 계획서는 "시간당 경고 수"의 분모를 정하지 않았다. 얼굴이 없거나 프레임이 끊긴 시간에는 감김·시선·졸음 규칙이 판정할 것이 없으므로 그 시간을 분모에서 뺀다. 기준선 수집 구간은 넣는다(임시 closure로 감김 규칙이 동작하고 경고도 나간다. PLAN 1.2절의 결정 D4). 영상 길이 전체를 분모로 한 값이 필요하면 alerts.csv의 duration_h로 다시 나눈다.
- `evaluate.py:2966` `replay_unlabeled` — 영상 하나의 실패로 전체 실행을 멈추지 않는다. 실패한 파일은 리포트에 이유와 함께 적고 합계에서 뺀다.
- `evaluate.py:3244` `run_unlabeled` — 속도를 주지 않으면(차량 속도 기록이 없는 영상) 속도 억제를 적용하지 않는다. 모든 경고가 속도와 무관하게 나간 것으로 센다. 리포트에 그렇게 적는다.
- `evaluate.py:3246` `run_unlabeled` — 속도를 주면 세션 내내 같은 값이다(데모와 같다. 시간에 따라 변하는 속도 기록은 받지 않는다).
- `evaluate.py:3247` `run_unlabeled` — "경고 수"는 운전자에게 실제로 나간 경고의 수다. 상태가 시작된 프레임에서 바로 나간 경고, 재경고 대기나 속도 억제에 막혔다가 상태가 이어지는 동안 나중에 나간 경고(2026-11-13 소유자 결정 2), 상태가 이어지는 동안의 되풀이 경고(기본 설정에서 sleep의 3초 주기)를 모두 센다. 세 종류의 수는 `immediate_alerts`, `deferred_alerts`, `repeat_alerts`에 따로 적는다(셋의 합이 `alerts`다).
- `evaluate.py:3251` `run_unlabeled` — 입력 가운데 합성 표시가 붙은 파일이 하나라도 있으면 두 파일의 첫 줄에 합성 데이터임을 함께 적는다.
- `evaluate.py:3252` `run_unlabeled` — 리포트 폴더의 기본 이름은 교차검증·최종 학습과 같은 `nested_cv`다(같은 모델의 기록을 한 폴더에 모은다).
- `evaluate.py:3253` `run_unlabeled` — 회전이 확인되지 않은 영상도 재생하고 합계에 넣는다(빼지 않는다). 리포트의 머리말과 4절에서 알리기만 한다. 틀린 각도로 추출된 영상의 경고 수는 믿을 수 없으므로 그 영상을 직접 보고 수동 회전 보정표에 적어야 한다.
- `evaluate.py:3255` `run_unlabeled` — 회전 점검 기록은 `extraction.unlabeled_rotation_overrides`(`python -m drowsy.extract --unlabeled`의 보정표)와 대조한다. `--raw-dir`로 다른 추출 폴더를 줘도 같은 표를 쓴다(키가 맞지 않으면 보정이 없는 것으로 본다). 재생할 수 있는 파일이 하나도 없으면 `UnlabeledError`다(아무것도 쓰지 않는다). run_name이 폴더 이름 하나가 아니면 ValueError다(`config.check_run_name`).
- `evaluate.py:3357` `run_unlabeled` — 리포트 문구: 차량 속도: 주지 않았다. 가정: 속도 입력이 없으므로 경고의 속도 억제(PLAN 1.2절 결정 D3)를 적용하지 않았다. "아래 경고 수는 속도와 무관하게 낸 경고를 모두 센 값이다
- `evaluate.py:3362` `run_unlabeled` — 리포트 문구: 차량 속도: {speed_kmh:g} km/h. 가정: 세션 내내 같은 값으로 두고 경고의 속도 억제(PLAN 1.2절 결정 D3)를 "적용했다
- `evaluate.py:3366` `run_unlabeled` — 리포트 문구: 분석 가능한 시간(가정): 얼굴이 검출된 프레임이 대표한 시간의 합(다음 프레임까지, 최대 {max_hold_ms} ms). "얼굴이 없거나 프레임이 끊긴 시간은 뺀다. 기준선 수집 구간은 들어간다", "경고 수는 `AlertPolicy`가 실제로 낸 경고(재경고 대기와 되풀이 경고를 적용한 뒤)이고, 상태 이벤트 수는 규칙 엔진이 " "낸 이벤트 수다. 시간당 경고 수 = 경고 수 / 분석 가능한 시간", "경고 수 = 바로 낸 경고(상태가 시작된 프레임에서 나간 것) + 미뤘다 낸 경고(시작할 때는 재경고 대기나 속도 억제에 " "막혔다가, 상태가 이어지는 동안 막힘이 풀린 첫 프레임에서 나간 그 상태의 첫 경고) + 되풀이 경고(같은 상태에서 이미 " "경고가 나간 뒤의 경고)", f"`drowsy`는 기준선이 완성되고 조회 구간 {features.lookback_long_ms / 1000.0:g}초가 찬 스텝부터 판정한다(모델 출력은 " f"그런 스텝이 {features.window_steps}개 이어진 뒤부터 나온다). 그보다 짧은 영상에서는 `drowsy` 경고가 나올 수 없지만 " "그 영상의 시간도 분석 가능한 시간에 들어 있다",

### 내보내기와 데모

**`drowsy/export_onnx.py`** (18개)

- `export_onnx.py:275` `refuse_synthetic_export` — "아래 폴더"는 풀어 쓴 경로의 상위 폴더에 export_dir이 있는 경우다. 이름만 비슷한 이웃 폴더(`export2`)나 다른 곳의 같은 이름 폴더(`out/export`)는 막지 않는다.
- `export_onnx.py:323` `protected_export_dir` — 저장소 루트는 이 모듈 파일의 위치에서 정한다(`drowsy/export_onnx.py`의 두 단계 위 폴더). 패키지를 다른 곳에 복사해 쓰면 그 사본이 놓인 폴더가 기준이 되고, 원래 저장소의 `export/`는 절대 경로로 적은 설정으로만 지킬 수 있다.
- `export_onnx.py:325` `protected_export_dir` — 늘 지키는 폴더는 이름이 `export`인 그 폴더 하나다. 저장소 안의 다른 폴더(`out/`, `reports/`)는 설정이 가리킬 때만 지킨다. 실제 데이터와 정식 설정으로 학습한 모델은 이 검사와 무관하게 어느 폴더로든 나간다.
- `export_onnx.py:376` `git_commit` — 변경은 `git status --porcelain`의 출력이 비어 있지 않은 것으로 판정한다. 추적하지 않는 파일 (.gitignore에 걸리지 않은 것)도 변경으로 센다.
- `export_onnx.py:378` `git_commit` — git을 실행할 수 없거나 git 저장소가 아니면 "uncommitted"다("-dirty"는 붙지 않는다).
- `export_onnx.py:410` `_load_model` — onnx는 해석할 수 없는 바이트에 protobuf의 DecodeError를 낸다. protobuf는 onnx가 끌어오는 패키지이고 이 프로젝트의 의존성 목록(pyproject.toml)에 없으므로 그 형을 import하지 않고, OSError가 아닌 예외를 모두 "읽을 수 없는 파일"로 바꾼다.
- `export_onnx.py:570` `export` — 계획서는 온도를 spec에도 적고 그래프에도 넣는다. 둘이 다르면 문서와 모델이 어긋나므로 내보내지 않는다.
- `export_onnx.py:571` `export` — 사본과 모델을 바꿔 놓는 두 번의 `os.replace` 사이에 프로세스가 죽거나 두 번째가 실패하는 경우는 다루지 않는다(그때는 새 사본과 예전 모델이 남고, 계약 검사가 해시 불일치로 알려 준다).
- `export_onnx.py:573` `export` — 임시 파일 이름은 `.<파일 이름>.<프로세스 번호>.tmp`다. 프로세스가 강제로 종료되면 남을 수 있고 그때는 손으로 지운다. 같은 폴더에 두 프로세스가 동시에 내보내는 경우는 다루지 않는다.
- `export_onnx.py:575` `export` — 같은 버전을 다시 내보내면 가중치가 같더라도 예전 INT8 파일을 지운다. 그 파일이 어느 내보내기에서 나온 것인지 파일만으로는 알 수 없기 때문이다.
- `export_onnx.py:738` `_attribute` — 값이 비어 있는 속성도 거부한다. onnxruntime은 값이 없는 INT 속성(transA 등)을 기본값으로 읽어 그대로 실행하지만, 내보내기는 그런 파일을 만들지 않으므로 받지 않는 쪽을 택했다(내보낸 파일과 v0은 모든 속성에 값이 있다).
- `export_onnx.py:787` `_check_logits_producer` — INT8 파일의 모양은 이 프로젝트가 쓰는 onnxruntime의 `quantize_dynamic`이 Gemm에 대해 내는 모양이다. 양자화기가 다른 모양을 내는 버전으로 바뀌면 `ALLOWED_OPS_INT8`과 함께 이 검사도 다시 맞춰야 한다(그때는 INT8 파일이 계약 검사에서 걸리므로 조용히 지나가지 않는다).
- `export_onnx.py:894` `_check_graph_constants` — log1p의 Add는 두 입력의 순서를 가리지 않고(덧셈이라 결과가 같다), Softmax의 축은 −1과 1을 같은 것으로 받는다.
- `export_onnx.py:895` `_check_graph_constants` — 신경망은 두 계열 모두 선형 층 하나로 끝나고 그래프에 선형 층은 그것뿐이다(`drowsy.models`의 `head`). 그래서 로짓을 내는 노드를 그 층의 모양으로 고정한다. 마지막 층의 상수(가중치와 편향)를 통째로 줄이거나 키운 파일은 그래프의 모양이 같아서 이 검사로 가릴 수 없다. 그것은 체크포인트와의 일치 검증(`parity_max_abs_diff`)이 잡는다.
- `export_onnx.py:1021` `verify_contract` — 파일 종류를 따로 받지 않고 allowed_ops로 정한다. `ALLOWED_OPS_DYNAMO`와 같은 목록을 준 검사만 dynamo 경로의 파일로 보고, spec의 온도가 1.0일 때 온도 상수가 없는 그래프를 받는다(`_check_graph_constants`). `ALLOWED_OPS_INT8`과 같은 목록을 준 검사만 INT8 파일로 보고, 로짓을 내는 노드를 양자화한 선형 층의 모양과 견준다. 그 밖의 목록은 Gemm과 견준다. 검사 항목: 파일 크기 1 MB 미만, opset 정확히 17, IR 8, 외부 데이터 없음, 입출력 이름·형·모양, 메타데이터 열 개와 그 값, 허용 연산자, 그래프 안 상수와 spec 값의 일치, 표준편차 하한, 전처리·후처리 연산의 연결(`_check_graph_constants`의 설명: log1p(max(x, 0)) 사슬, 입력을 읽는 노드, Softmax의 축과 출력, Softmax 바로 앞에서 `temperature` 상수로 나누는 Div와 그 밖의 로짓 배율 금지, 로짓을 내는 노드가 신경망의 마지막 선형 층 그대로인지), onnx 검사기, onnxruntime 로드와 0 입력 한 번의 실행. 파일 이름은 보지 않는다(`verify_file_name`). ONNX로 해석할 수 없는 파일도 ContractError다.
- `export_onnx.py:1259` `_run_cli` — spec은 받은 파일의 바이트가 아니라 `export`가 모델 옆에 쓸 사본(고정된 형식)의 바이트로 대조한다.
- `export_onnx.py:1269` `_run_cli` — 체크포인트의 `synthetic`이 false여도 `trained_on`이 합성 표지로 시작하면 합성 모델로 본다.
- `export_onnx.py:1344` `main` — 여러 줄로 된 예외 메시지는 줄바꿈과 연속 공백을 공백 하나로 바꿔 한 줄로 만든다.

**`drowsy/demo.py`** (21개)

- `demo.py:85` 모듈 설명 — 계획서의 명령에는 `--raw`와 `--config`가 없다. `--raw`는 `--video` 대신 주는 입력이고(둘 중 하나만), `--config`는 다른 모듈과 같은 실험 설정 파일이다(feature_spec, 규칙, MediaPipe 모델 경로, 그림 해상도를 읽는다).
- `demo.py:87` 모듈 설명 — `--model`을 주지 않으면 설정의 `demo.model_path`를 쓰고, 그 파일이 아직 없으면(학습한 v1을 내보내기 전) `paths.export_dir`의 무작위 가중치 모델 `drowsiness_v0.onnx`를 쓰며 그 사실을 표준 오류에 알린다. `--out`을 주지 않으면 `demo.out_dir` 아래 `demo/`(기본 설정에서 `out/demo`)에 쓴다.
- `demo.py:90` 모듈 설명 — 파이프라인은 설정의 feature_spec(`paths.feature_spec`)으로 돌리고, 모델의 `pipeline_hash`가 그것과 다르면 거부한다. 표준화 상수는 그래프 안에 있으므로 모델과 함께 나간 사본(`feature_spec_v{N}.json`)은 해시 대조에만 쓴다.
- `demo.py:92` 모듈 설명 — 추출 parquet에 MediaPipe 모델 sha256이 적혀 있고 spec의 값과 다르면 거부한다(다른 랜드마크 모델의 신호다). 합성 parquet에는 그 값이 없으므로 검사하지 않는다.
- `demo.py:94` 모듈 설명 — `--speed-kmh`는 세션 내내 같은 값이다(명령은 시간에 따라 변하는 속도 입력을 받지 않는다. `run_pipeline`은 프레임 시각을 받아 속도를 돌려주는 함수도 받는다).
- `demo.py:96` 모듈 설명 — 경고의 종류는 종류별로 "지금 상태에서 경고가 이미 나갔는지"만 기억해서 가린다. 상태는 그 종류의 `onset` 이벤트에서 새로 시작한 것으로 본다(`AlertPolicy`가 새 `onset`에서 앞 상태의 미뤄 둔 경고와 되풀이 기록을 지우는 것과 같은 기준이다).
- `demo.py:99` 모듈 설명 — 이벤트 JSON의 경고에는 예전의 `repeat`(불) 대신 `kind`(세 값)만 적는다. 두 필드를 함께 두지 않는다 (`repeat`가 참이던 경고는 이제 `kind`가 `repeat`이거나 `deferred`다).
- `demo.py:101` 모듈 설명 — 그림에서 경고의 종류는 표식의 모양으로 구분한다(색은 하나다). 계획서는 표식을 정하지 않았다. 범례에는 그 종류의 경고가 하나도 없어도 세 종류를 늘 싣고, 범례가 길어진 만큼 이벤트 줄의 제목을 줄였다.
- `demo.py:103` 모듈 설명 — 전방 주시 줄은 프레임마다 `DriverPipeline.gaze_zone`을 읽어 그 프레임이 대표하는 시간(다음 프레임까지, 최대 `clock.max_hold_ms`)에 칠한다. 프레임이 대표하지 않는 시간은 "알 수 없음" 줄에 넣는다. `gaze_zone`은 확정된 상태(디바운스 250 ms 뒤)이므로, 영역 밖 표시는 시선 이벤트의 소급된 `t_start_ms`보다 그만큼 늦게 시작한다.
- `demo.py:106` 모듈 설명 — 이벤트 구간은 `t_start_ms`부터 `end` 이벤트의 `t_start_ms + duration_ms`까지다. `sleep`의 `end`는 같은 감김의 `microsleep` 구간도 닫는다(PLAN 8.2절: 한 감김은 도달한 가장 높은 종류의 `end` 하나만 낸다). 입력이 끝날 때까지 `end`가 없는 구간은 마지막 프레임까지 그린다.
- `demo.py:109` 모듈 설명 — 그림의 글자는 영어만 쓰고 글꼴은 matplotlib에 들어 있는 DejaVu Sans로 고정한다. 입력 파일 이름의 ASCII가 아닌 글자는 그림에서만 `\\uXXXX` 꼴로 바꿔 적는다(없는 글리프가 생기지 않게. JSON에는 원래 경로를 적는다).
- `demo.py:111` 모듈 설명 — 심각도 색은 0~3의 네 단계다. 규칙 설정이 그 밖의 심각도를 주면 가까운 쪽 끝 단계의 색으로 그린다.
- `demo.py:112` 모듈 설명 — 계획서는 출력 파일의 이름을 정하지 않았다. `<줄기>_timeline.png`, `<줄기>_events.json`으로 쓰고, 같은 이름의 파일이 있으면 덮어쓴다.
- `demo.py:114` 모듈 설명 — 합성 입력(`drowsy.synth`가 쓴 parquet)으로 만든 그림과 JSON은 첫 줄에 합성 데이터임을 적는다.
- `demo.py:115` 모듈 설명 — `--video`의 회전은 `--rotation`과 자동 점검 둘로만 정한다. 설정의 수동 회전 보정표(`extraction.rotation_overrides`)는 보지 않는다(그 표의 키는 추출 입력 폴더 기준 경로여서 영상 하나만 받는 데모에서는 어느 영상을 가리키는지 정해지지 않는다). 점검이 가르지 못해도 멈추지 않고 고른 각도로 실행하며 출력, 그림, JSON에 표시한다.
- `demo.py:118` 모듈 설명 — 표시 행렬에 뒤집기가 든 영상은 회전 메타데이터를 믿을 수 없어 자동 점검도 하지 않는다. `--rotation`이 없으면 거부하고(오류 문장이 `--rotation`을 알려 준다), 있으면 그 각도로만 돌리고 거울상은 되돌리지 않는다.
- `demo.py:120` 모듈 설명 — `--raw` 입력은 추출할 때 이미 세운 결과이므로 `--rotation`을 함께 주면 거부한다. 추출 때의 회전 정보는 parquet 메타데이터에서 읽어 한 줄로 출력하고 JSON의 `source`에 옮겨 적는다(위 "추출 결과의 회전").
- `demo.py:122` 모듈 설명 — `--raw`의 회전 줄은 그 파일의 메타데이터만 보고 정한다. 설정의 수동 회전 보정표는 보지 않는다(표의 키는 추출 입력 폴더 기준 경로여서 parquet 하나만 받는 데모에서는 어느 영상을 가리키는지 정해지지 않는다). 그래서 자동 점검이 가르지 못한 영상을 사람이 보정표에 같은 각도로 적어 확인한 경우에도(그때 추출은 파일을 다시 만들지 않는다) 여기서는 가르지 못했다고 나온다. 보정표까지 대조한 판정은 `python -m drowsy.dataset`, `signal_check e0`, `evaluate unlabeled`이 한다.
- `demo.py:126` 모듈 설명 — `--raw`의 회전 줄은 합성 데이터 표시 줄 다음에 출력한다(합성 입력이면 출력의 첫 줄이 합성 데이터 표시여야 한다). `--video`의 회전 줄은 프레임을 읽기 전에 정해지는 대로 출력하므로 첫 줄이다.
- `demo.py:128` 모듈 설명 — `--raw` 입력의 그림에는 점검이 가르지 못한 경우에만 회전 줄을 넣는다(합성 입력의 머리말은 이미 다섯 줄을 다 쓴다).
- `demo.py:345` `AlertKinds` — `onset`을 본 적 없는 종류의 사본이 오면(`AlertPolicy`는 그렇게 내지 않는다) 그 상태의 첫 경고로 본다.

### 실시간 시험 도구

**`drowsy/live.py`** (17개)

- `live.py:178` 모듈 설명 — 소스의 형식은 소스 문자열로 정한다(`plan_source`). `tcp://`와 `udp://`는 MPEG-TS와 지연을 줄이는 옵션(`STREAM_OPTIONS`)으로, `/dev/video`로 시작하면 v4l2로 열고, 그 밖은 PyAV가 내용을 보고 정하게 둔다. 형식을 고르는 옵션은 두지 않았다(승인된 옵션 목록에 없다).
- `live.py:181` 모듈 설명 — 영상 파일 소스는 타임스탬프대로 실제 속도로 재생한다(첫 프레임을 낸 때부터 `t_ms`만큼 지난 때에 그 프레임을 낸다). 파일은 디코딩이 실제 시간보다 훨씬 빨라서, 그대로 두면 가장 새 프레임만 넘기는 규칙이 대부분의 프레임을 버린다. 파일 전체를 빠짐없이 돌리려면 `python -m drowsy.demo --video`를 쓴다.
- `live.py:184` 모듈 설명 — 소스의 회전 메타데이터와 표시 행렬은 보지 않는다. `--rotation`으로 준 각도(기본 0)만 쓴다.
- `live.py:185` 모듈 설명 — 첫 프레임에 pts가 없거나 스트림에 time_base가 없으면 스트림 전체를 도착 시각 시계로 읽는다. 첫 프레임에는 pts가 있는데 뒤 프레임에 없으면 그 프레임은 버리고 센다(두 시계를 섞지 않는다. `video_io`와 같다).
- `live.py:187` 모듈 설명 — 풀 수 없는 패킷(`InvalidDataError`)은 건너뛰고 센다(`StreamCounts.decode_errors`). 스트림 하나의 깨진 패킷이 세션을 끝내지 않게 하려는 것이다.
- `live.py:189` 모듈 설명 — 보내는 쪽의 ffmpeg 명령 예 가운데 카메라 입력 부분(`-f dshow -video_size ... -framerate ... -i video=...`과 목록을 보는 두 명령)은 실행해 보지 못했다(Windows에 ffmpeg가 없고 Windows 프로그램을 실행하지 않았다). 옵션의 이름과 자리는 검토 때 ffmpeg 공식 문서와 대조했다. 그 뒤의 인코더와 출력 부분(libx264, ultrafast, zerolatency, `-g 30`, yuv420p, MPEG-TS, TCP)은 PyAV로 만든 보내는 쪽과, 검토 때 이 PC의 Linux ffmpeg 7.0.2 명령으로 확인했다(합성 그림). 그 Linux ffmpeg 명령은 주소를 `tcp://127.0.0.1:<포트>`로 적었다: 그 빌드는 `localhost`라는 이름으로는 접속하지 못했다 (2026-11-14의 점검에서 `tcp://localhost:<포트>`로는 12초 동안 접속하지 못하고 메시지 없이 끝났다).
- `live.py:195` 모듈 설명 — 보내는 쪽의 명령 예는 주소를 `localhost`로 적는다(`sender_target`). 2026-11-14에 Windows PowerShell의 TCP 접속이 WSL 안의 듣는 쪽에 `localhost`로도 `127.0.0.1`로도 닿았다(웹 페이지 요청은 `localhost`로 닿았다). 영상은 Windows에서 보내 본 적이 없으므로 Windows의 ffmpeg가 `localhost`로 접속하는지는 확인하지 못했다. 그래서 접속이 거부되거나 보내는 쪽을 시작했는데도 `still waiting` 줄만 나오면 `127.0.0.1`로 바꿔 보라고 안내한다(`--help`).
- `live.py:199` 모듈 설명 — 듣는 소스에 접속한 쪽이 영상 없이 끊으면 오류로 끝내지 않고 남은 시간 동안 다시 듣는다(`open_source`). PyAV는 접속을 받은 때를 알려 주지 않으므로 "source opened"는 스트림 분석까지 끝난 뒤에 나온다. 그래서 접속한 채 MPEG-TS가 아닌 것을 계속 보내거나 아무것도 보내지 않는 쪽이 있어도 접속 시간 초과까지 `still waiting` 줄만 나온다. 시간 초과 뒤의 오류 문장은 아무도 접속하지 않은 경우와 접속한 쪽이 MPEG-TS 영상을 보내지 않은 경우를 함께 적고 `-f mpegts`를 알려 준다.
- `live.py:204` 모듈 설명 — 기다리는 시간들(`CONNECT_TIMEOUT_S`, `STALL_TIMEOUT_S`), 미리보기와 상태 줄의 주기, 프레임 상한은 옵션이 아니라 이 모듈의 상수다(설정 파일과 승인된 옵션 목록을 바꾸지 않는다).
- `live.py:206` 모듈 설명 — `--model`을 주지 않았을 때의 모델 선택(설정의 `demo.model_path`, 그 파일이 없으면 `paths.export_dir`의 `drowsiness_v0.onnx`와 표준 오류의 알림)과 `--speed-kmh`의 범위 검사는 데모와 같다. 데모에서는 이 둘이 `main`과 `_demo` 안에 있어 부를 수 없으므로 같은 내용을 여기에 다시 적었다.
- `live.py:209` 모듈 설명 — `LiveSession.process_raw`는 `demo.run_pipeline`의 프레임 하나 분량과 같은 일을 한다(`DriverPipeline.update`, `AlertKinds`, `EventRecord`, `AlertRecord`, `ProbPoint`). `run_pipeline`은 입력이 끝나야 결과를 돌려주므로 프레임마다 부를 수 없다. 두 길이 같은 결과를 내는지는 테스트가 확인한다.
- `live.py:212` 모듈 설명 — 기준선 진행(`baseline 12/60 s`)의 분모는 지금 마감이다. 파이프라인이 마감을 내주지 않으므로 spec의 `baseline.duration_ms`, `extend_step_ms`, `max_duration_ms`로 다시 구한다(60초, 미달이면 30초씩 180초까지).
- `live.py:214` 모듈 설명 — 프레임 표의 메타데이터에서 끝나야 아는 값(프레임 계수, 끝난 이유)은 처리 루프가 끝날 때 채운다. `extract.write_raw_frames`가 메타데이터를 프레임을 다 받은 뒤에 읽는 것에 기댄다(테스트가 확인한다).
- `live.py:216` 모듈 설명 — 프레임 표에는 세션에 쓴 각도를 `rotation_override`에도 적는다(그 키의 뜻이 "소스의 회전 메타데이터 대신 쓴 각도"이고, 실시간 경로는 메타데이터를 보지 않는다). `--rotation`을 주지 않은 세션도 0을 그렇게 적는다. 그래서 끝의 데모 출력의 회전 줄은 "manual value recorded by the extraction"이고, `python -m drowsy.dataset`의 회전 요약도 실시간 표를 수동 값으로 센다. 실시간 표에서 그 말은 "`--rotation`으로 준 값이거나 기본값 0이고, 아무것도 점검하지 않았다"는 뜻이다. 줬는지는 `rotation_given`에 적고, 데모 출력 앞의 `replay: ...` 줄이 그 뜻을 알린다. 데모의 문장은 `drowsy.demo`에 있어 여기서 바꾸지 않았다. 녹화 요약(`extract_video`가 적는 밝기, 프레임 간격 등)과 `extract_version`은 적지 않는다.
- `live.py:223` 모듈 설명 — 프레임 하나를 처리하다 난 오류는 `Exception`이면 모두 "처리 오류"로 다룬다(그때까지의 프레임을 쓰고 2로 끝난다). 어떤 종류의 오류든 긴 세션을 표 없이 끝내는 것보다 낫다고 봤다. 오류가 난 그 프레임은 표에 들어가지 않고 끝의 프레임 계수에서 `failed`(메타데이터 `n_failed`)로 따로 센다. 파이프라인이 받아들인 뒤 줄을 내다 오류가 난 프레임은 표에 들어가므로 `processed`로 센다.
- `live.py:227` 모듈 설명 — 상태 줄은 실제 시간으로 `STATUS_INTERVAL_S`(1초)마다 한 번 출력하고, 초당 프레임 수는 그 사이에 받은·처리한 프레임 수를 지난 시간으로 나눈 값이다. 프레임이 오지 않는 동안에도 마지막 상태로 줄을 낸다(0.0 fps로 보인다).
- `live.py:229` 모듈 설명 — 출력 파일의 이름은 세션을 시작한 지역 시각으로 짓는다. 같은 초에 시작한 두 세션은 같은 이름을 쓴다(덮어쓴다).

### 시험 파일 (`tests/`)

**`tests/test_demo.py`** (1개)

- `tests/test_demo.py:797` `test_alert_kinds_classifier_follows_the_state_of_each_type` — 시작을 본 적 없는 종류의 사본은 그 상태의 첫 경고로 본다

**`tests/test_evaluate.py`** (1개)

- `tests/test_evaluate.py:1134` `test_alarm_restarts_where_the_step_number_jumps_inside_a_session` — (5) 건너뛴 크기는 보지 않는다(가정): 스텝 하나만 빠져도(251 → 253) 같은 다시 시작이다

**`tests/test_leakage.py`** (5개)

- `tests/test_leakage.py:35` 모듈 설명 — alert 영상(라벨 0)은 맞바꾸지 않는다. alert 영상 기준선 조건은 피험자의 alert 영상에서 기준선을 얻으므로(PLAN 4.3절), 그 파일을 바꾸면 라벨을 정당하게 쓰는 그 경로 때문에 26번의 특징이 달라져 비교할 수 없다.
- `tests/test_leakage.py:52` 모듈 설명 — 세션은 명목 18 fps로 만든다(`drowsy.synth`의 기본은 피험자마다 17~30 fps이고 구간마다 그 70%까지 내려간다). 프레임 수를 줄여 재생 시간을 줄이려는 것이고, 보려는 것(어느 피험자의 데이터가 어느 계산에 들어가는가)은 프레임률과 무관하다. 더 낮추지 않는 이유: 프레임 간격이 흐름 제한기의 처리 지연(평균 66.7 ms)보다 길면 15 fps 누락 모사 폴더가 원본과 같은 프레임을 담게 되어 두 출처가 구분되지 않는다.
- `tests/test_leakage.py:56` 모듈 설명 — 확률의 허용 오차는 1e-6이다. 25번의 창 수가 달라지면 시험 fold를 한 번에 넣는 창 수(배치 크기)가 달라지고, GRU는 배치 크기에 따라 float32 마지막 자리가 달라질 수 있다(누설이 아니다). 이 PC에서 무작위 가중치의 GRU(은닉 8)에 같은 창을 배치 크기만 바꿔 넣었을 때 확률이 5.96e-08까지 달랐고, 이 테스트의 두 실행에서는 26번의 확률이 44개 묶음 모두에서 차이 0이었다. 시험 fold의 통계가 평가에 섞이는 누설은 허용 오차보다 몇 자리 큰 차이를 낸다(평가 함수에 그런 누설 둘을 넣은 사본에서 이 파일이 실패하는 것을 확인했다). 라벨만 맞바꾼 비교에서는 26번의 창이 같은 배치 안에서 자리만 바뀐다. 같은 허용 오차를 쓴다.
- `tests/test_leakage.py:240` `run_commands` — 세 프로세스는 같은 환경 변수(스레드 수 포함)로 같은 기계에서 돈다. 비교하는 값(가중치 해시, 확률)은 실행마다 같은 값이 나오는 CPU 학습의 결과이므로, 한 프로세스에서 차례로 돌릴 때와 같은 비교가 된다. 세 번째 데이터셋을 더하고도 이 파일의 소요 시간이 늘지 않게 하려는 것이다(차례로 돌리면 데이터셋마다 약 45초가 든다).
- `tests/test_leakage.py:243` `run_commands` — 환경 변수 `OMP_NUM_THREADS`가 없으면 세 프로세스에 2로 준다(있으면 그 값 그대로). 셋이 동시에 도는데 저마다 모든 코어를 쓰려 들면 서로 밀려 오히려 느려진다. 이 시험의 모델과 표는 작아서 스레드를 늘려도 빨라지지 않는다.

**`tests/test_signal_check.py`** (1개)

- `tests/test_signal_check.py:1012` `test_e0_reports_videos_whose_rotation_is_ambiguous_without_an_override` — 01/5.mov도 보정표에 적으면 확인할 영상이 없다. 통과 기준의 판정과 종료 코드는 표시가 있을 때와 같다(가정)

**`tests/test_spec_doc.py`** (8개)

- `tests/test_spec_doc.py:11` 모듈 설명 — 상수 표의 첫 칸은 경로, 둘째 칸은 값이다. 값은 설정의 자료형에 맞춰 읽는다(정수는 소수점 없이, 실수는 소수점을 붙여서, 불리언은 true/false, 목록은 쉼표로 나눠서 적는다). 표의 칸 안에는 `|`를 쓰지 않는다.
- `tests/test_spec_doc.py:13` 모듈 설명 — 본문의 괄호 인용에서 점이 없는 이름(`hold_ms`)은 두 설정에서 같은 이름의 값을 모두 찾아 전부와 비교한다. 값이 서로 다른 이름을 점 없이 인용하면 실패하므로, 그런 이름은 문서에 전체 경로로 적어야 한다.
- `tests/test_spec_doc.py:15` 모듈 설명 — `pipeline` 블록과 규칙 설정의 값은 빠짐없이 표에 있어야 한다(랜드마크·블렌드셰이프·특징 이름·이벤트 정책은 각자의 표가 맡는다). 설정에 값을 더하고 문서를 고치지 않으면 여기서 드러난다.
- `tests/test_spec_doc.py:17` 모듈 설명 — 17.3절의 모사 표는 `signal_check.simulate_blinks`를 시드 0으로 다시 돌린 값과 비교한다. 문서는 반올림해 적으므로 적힌 자릿수의 절반까지 허용한다. 검출된 깜빡임이 없어 값이 없는 칸은 `-`로 적는다.
- `tests/test_spec_doc.py:19` 모듈 설명 — 본문에서 코드 조각으로 인용한 이름 가운데 `signals_`, `baseline_`, `runs_`, `steps_`, `rules_`, `e2e_`로 시작하는 것은 벡터 이름으로 본다(벡터가 아닌 식별자 둘은 테스트에 적어 둔다). 그런 이름은 모두 실제 벡터여야 한다.
- `tests/test_spec_doc.py:21` 모듈 설명 — 20장(소유자의 결정을 기다리는 동작)은 "## 20."으로 시작하는 줄부터 문서 끝까지이고, 그 안의 질문은 줄 맨 앞의 "N. **"으로 시작한다. 본문의 "20장 N번"은 그 번호를 가리킨다. 벡터의 수와 합계 크기는 20장에 "JSON 파일 N개, 합계 B bytes" 모양으로 적는다(B는 쉼표로 끊어 적는다). 결정된 질문의 번호는 비워 두고 다른 번호는 바꾸지 않는다(`RETIRED_SECTION_20_ITEMS`). 결정 기록은 20장 머리에 "<날짜>에 결정되어 이 목록에서 빠진 것"으로 적는다.
- `tests/test_spec_doc.py:25` 모듈 설명 — 20장의 항목 하나는 "N. **"으로 시작하는 줄부터 다음 항목 앞까지이고(마지막 항목은 맺는 문단 앞의 빈 줄까지), 대안은 "제안된 대안"이라는 말로 적는다. 9번은 대안을 넣은 참조 구현으로 돌려 본 벡터의 수를 "N개가 모두 그대로 통과"로 적는다.
- `tests/test_spec_doc.py:27` 모듈 설명 — README.md가 docs/SPEC.md를 이식 명세로 가리키는지는 "한 줄에 `docs/SPEC.md`와 '이식 명세'가 함께 있는가"로 본다.

**`tests/test_train.py`** (1개)

- `tests/test_train.py:622` `test_fast_profile_and_mixing_share_in_the_default_config` — 학습 표본의 몫(가정): 원본 0.5, 15 fps 누락 모사 0.25, 프레임 누락 변형 0.25(변형 하나).

### 다른 곳의 가정을 가리키기만 하는 줄

아래 줄에도 '가정'이라는 낱말이 있지만 새 가정이 아니다. 코드의 줄은 위 항목이나 같은 docstring의 다른 줄을 가리키고, 시험 파일의 줄은 위 항목을 가리키거나 리포트 문구에 '가정'이 적혀 있는지 확인하는 시험 줄(`assert`와 그 머리 주석)이다.

- `dataset.py:621` — - frame_drop: 주면 프레임 누락 변형을 만든다(캐시 없이. 뜻과 가정은 `load_frame_drop_tables`의 설명에 있다).
- `demo.py:522` — """`onset`과 `end`를 짝지어 구간으로 만든다(모듈 설명의 가정 참조)."""
- `evaluate.py:226` — ECE_BINS = 15  # 가정은 `expected_calibration_error`에 적었다
- `evaluate.py:975` — - perclos와 either에 남는 차이는 위의 첫 가정(행만 평가한다) 하나다. 앱의 PERCLOS 판정기는 세션의 처음과 10초 넘는
- `evaluate.py:1169` — 실행 하나의 예측 표를 받는다. 판정 방식과 가정은 `alarm_videos`, 수치의 뜻은 `AlarmReport`에 있다.
- `evaluate.py:1809` — # 값과 주석은 PLAN 2.4절의 것이다. Xu 2025를 3클래스 열에 둔 근거는 `write_reports`의 가정에 적었다.
- `evaluate.py:3121` — """라벨 없는 영상 리포트의 summary.md 줄들. facts는 머리말 아래의 항목(명령, 모델, 해시, 가정)이다.
- `evaluate.py:3238` — 판정원(`drowsiness.source`), 속도와 분석 가능한 시간에 대한 가정, 전체와 영상별 표(분석 가능한 시간, 이벤트
- `live.py:22` — 영상 파일 시험용. 파일의 타임스탬프대로 실제 속도로 재생한다(아래 가정).
- `live.py:23` — 보내는 쪽의 예(Windows PowerShell의 ffmpeg. 카메라 입력 부분은 실행해 보지 못했다. 아래 가정):
- `live.py:36` — 줄만 내면 주소를 `tcp://127.0.0.1:5599`로 바꿔 본다(아래 가정). (앞 줄에서 이어지는 문장의 끝: 접속이 거부되거나, 보내는 쪽을 시작했는데도 `still waiting` 줄만 나올 때)
- `live.py:323` — # 보내는 쪽의 예(Windows PowerShell의 ffmpeg). {target}은 이 프로그램이 듣는 주소다. 모듈 설명의 가정 참조
- `live.py:359` — """소스 문자열로 여는 방법을 정한다(모듈 설명의 가정 참조)."""
- `live.py:371` — 이 PC 자신을 가리키는 주소(127.0.0.1, 0.0.0.0, ::, ::1)는 `localhost`로 적는다(모듈 설명의 가정).
- `live.py:465` — 않았다(둘을 가릴 수 없다. 모듈 설명의 가정). (`open_source` 설명의 끝: 듣는 소스의 시간 초과 오류가 두 경우를 함께 적는 까닭)
- `live.py:1768` — 모듈 설명의 가정 참조). (`table_metadata` 설명의 마지막 줄: `rotation_given`의 뜻)
- `models.py:26` — `fast`는 v2에 나중에 더한 키다. 형식 이름은 그대로이고, 이 키가 없는 v2 파일도 읽는다(`load_checkpoint`의 가정).
- `signal_check.py:31` — 가정 (계획서가 정하지 않아서 정한 것. 함수별 가정은 각 docstring에 있다)
- `signal_check.py:133` — CONSTANT_DECIMALS = 4  # 제안 전역 상수의 소수 자릿수(가정은 `global_constants`에 적었다)
- `signal_check.py:1038` — "기준선으로 따로 재생하므로 따로 센다(가정)",
- `step_features.py:265` — 이 프레임이 발행시킨 스텝을 오래된 것부터 돌려준다. 시간 규칙과 가정은 모듈 설명에 있다.
- `tests/test_evaluate.py:3284` — # 가정: 속도를 주지 않았으므로 속도 억제가 없다. 분석 가능한 시간의 정의
- `tests/test_evaluate.py:3286` — assert "주지 않았다" in speed and "가정" in speed
- `tests/test_evaluate.py:3295` — assert any(line.startswith("- 분석 가능한 시간(가정): 얼굴이 검출된 프레임이 대표한 시간의 합") for line in lines)
- `tests/test_evaluate.py:3348` — assert "- 차량 속도: 10 km/h. 가정: 세션 내내 같은 값으로 두고 경고의 속도 억제(PLAN 1.2절 결정 D3)를 적용했다" in lines
- `tests/test_leakage.py:39` — - 평가받는 피험자의 alert 대 나머지 라벨만 쓰는 누설. 26번의 alert 파일은 세 벌에서 모두 같으므로(위의 가정), 예측에
- `tests/test_leakage.py:99` — PROBABILITY_TOLERANCE = 1e-6  # 26번의 확률이 두 실행에서 이 안에서 같아야 한다(모듈 설명의 가정)
- `tests/test_spec_doc.py:685` — assert "## 21." not in section  # 20장이 문서의 마지막 장이라는 가정을 붙잡는다

## 12. 확인한 것과 확인하지 못한 것

### 12.1 실행해서 확인한 것 (2026-11-14~15, 이 PC)

아래 표의 행은 두 차례의 기록이다. 결과 칸이 **(4차 검증 뒤)**로 시작하는 행은 4차 검증 뒤의 정비(2026-11-14~15)가 지금
코드에서 돌린 결과이고, 누가 돌렸는지(이 문서 단계, 통합 단계, 작업자)를 그 칸에 적었다. 나머지 행은 3차 검증 뒤의 정비에서
문서 단계가 문서를 쓰면서 돌린 결과다. 그 뒤에 4차 검증 뒤의 정비가 고친 코드 파일은 `drowsy/spec.py`(설정 검증 조건 둘),
`drowsy/export_onnx.py`(속성 읽기), `drowsy/train.py`·`drowsy/baseline_lgbm.py`·`drowsy/evaluate.py`(D 번호 안내 줄),
`drowsy/dataset.py`(설명 문장 하나), `drowsy/vectors.py`(벡터 13개), `drowsy/live.py`(4.13절)이고, 이식 대상 모듈 일곱은 그대로다.
그래서 학습 명령의 행은 지금 코드로 다시 돌린 것이 아니지만, 같은 명령들이 지금 코드에서 모두 종료 코드 0으로 돌았다(표
다음의 통합 단계 기록). 3차 검증 뒤의 문서 단계가 저장소 안에 쓴 `reports/fix_r4_docs/`는 그 뒤의 통합 단계가 지웠고, 4차
검증 뒤의 문서 단계는 저장소 안에 아무것도 쓰지 않았다(출력은 모두 임시 폴더). 표 다음에 통합 단계의 기록이 있다.
`<임시 설정>`은 `config/default.yaml`을 복사해 `paths.reports_dir`과 `paths.raw_dir`만 임시 폴더로 바꾼 것이다(4.0의 절차).
`<얼굴 없는 시험 영상>`은 얼굴이 없는 짧은 시험 영상(저장소 밖)이다. 합성 데이터와 시험 영상의 결과는 명령이 도는지만 보여
준다. 걸린 시간은 이 PC에서 한 번 잰 값이다. 4차 검증 뒤의 행은 다른 작업과 함께 돌았으므로 가장 낮은 우선순위
(`nice -n 19`)로 돌렸고, 그래서 다른 작업이 없을 때보다 느리다.

| 무엇을 | 명령 | 결과 |
|--------|------|------|
| 전체 테스트 | `python -m pytest -q` | **(4차 검증 뒤)** 이 문서 단계가 문서를 마친 뒤 `nice -n 19`로: `1743 passed, 3 skipped in 391.83s (0:06:31)`. 통합 단계의 처음과 끝: `1743 passed, 3 skipped in 386.88s (0:06:26)`, `1743 passed, 3 skipped in 393.98s (0:06:33)`(시험 1,746개, 파일 23개). 건너뛴 셋은 얼굴 이미지가 있어야 도는 시험(4.3절). 3차 검증 뒤에는 `1606 passed, 3 skipped in 318.02s (0:05:18)`였다 |
| 문서와 데모 테스트 | `python -m pytest tests/test_spec_doc.py tests/test_demo.py -q` | **(4차 검증 뒤)** 이 문서 단계가 문서를 마친 뒤 `nice -n 19 ... -p no:cacheprovider`로: `63 passed, 1 skipped in 17.07s`(건너뛴 하나는 위의 얼굴 이미지 시험). 3차 검증 뒤의 표에 있던 `60 passed, 1 skipped in 15.76s`는 기록이 남지 않아 다시 잰 것이다 |
| 누설 회귀 시험만 | `python -m pytest -q tests/test_leakage.py` | `37 passed in 54.38s`(다른 작업이 없을 때) |
| 테스트 벡터 재검증 | `python -m drowsy.vectors verify --dir tests/vectors` | **(4차 검증 뒤)** 통합 단계: `102 of 102 vectors verified in tests/vectors`, 1.21초와 1.27초 |
| 테스트 벡터 다시 만들기 | `python -m drowsy.vectors generate --out <임시 폴더>` | **(4차 검증 뒤)** 통합 단계: 102개, 30,588,827 bytes, 3.68초와 3.71초. `tests/vectors/`의 102개와 바이트 단위로 같다. 4차 검증 전의 89개는 합계 27,983,687 bytes로 그대로다 |
| 9장 9번의 다른 안 | 다른 안을 참조 구현의 사본에 넣는 저장소 밖 스크립트 | **(4차 검증 뒤)** 작업자(벡터와 명세 담당): 벡터 102개가 모두 그대로 통과. 3차 검증 뒤의 문서 단계: 7,200 ms와 7,201 ms 둘 다 `t_start_ms` 1,000인 시작, 0.4초 곁눈질 경우는 이벤트 없음(지금 코드: 7,200 ms 없음, 7,201 ms 시작) |
| feature_spec의 해시 | `python -m drowsy.spec check` (`--spec export/feature_spec_v0.json`도) | **(4차 검증 뒤)** 통합 단계와 작업자: `pipeline_hash` 650495e5…, `feature_spec_hash` f4031d09…로 그대로. 3차 검증 뒤의 출력: `pipeline_hash OK: 650495e5…`, `feature_spec_hash: f4031d09…` |
| 규칙 설정 내보내기 | `python -m drowsy.spec export-rules --out <임시 폴더>/rules.json` | **(4차 검증 뒤)** 통합 단계와 작업자: `rules_hash` 9c2d05f2…, `export/rules.json`과 바이트 단위로 같다 |
| v0 내보내기 | `python -m drowsy.export_onnx --random-init --kind tcn --version 0 --out-dir <임시 폴더>` | **(4차 검증 뒤)** 통합 단계: `verify_contract` 통과(logits 노드와 속성 검사 포함), 최대 절대 오차 1.192e-07, 148,558 bytes, `export/drowsiness_v0.onnx`(sha256 eb2f71a6…)와 바이트 단위로 같다, 1.57초와 1.64초 |
| GRU 내보내기 | `python -m drowsy.export_onnx --random-init --kind gru --version 0 --out-dir <임시 폴더>` | `verify_contract: OK`, 최대 절대 오차 5.960e-08, 68,343 bytes |
| 체크포인트 다시 내보내기 | `python -m drowsy.export_onnx --checkpoint reports/synth_m2/final/final_model.pt --spec out/synth_m2/feature_spec_v1.json --version 1 --out-dir <임시 폴더>` | **(4차 검증 뒤)** 통합 단계: 종료 코드 0, 최대 절대 오차 1.788e-07, 1.52초, `out/synth_m2/drowsiness_v1.onnx`(sha256 1a522878…, 23,732 bytes)와 바이트 단위로 같다 |
| 늘 지키는 `export/` | 패키지와 설정을 임시 폴더에 복사하고(그 폴더가 저장소 루트 역할), `paths.export_dir`을 다른 절대 경로로 바꾼 설정으로 그 폴더에서 위 명령을 `--out-dir export`로 | 종료 코드 2, `error: a model trained on synthetic data must not be exported to the export folder of this repository (...; it is protected whatever paths.export_dir names, here ...)`, 사본의 `export/`는 빈 채로 남음. 다른 폴더로 준 대조 실행은 종료 코드 0. 진짜 `export/`에는 명령을 겨누지 않았다(파일 넷의 sha256이 그대로다) |
| 계약 검사의 변이 | 3차 검증 뒤의 정비에서 쓴 변이 스크립트 둘을 임시 폴더에 복사해 돌림 | 정상적인 내보내기 40개(기본·대체 경로, 온도 1.0·0.8·1.7, 작은 설정 포함)와 저장소 v0이 모두 통과. 반드시 거부해야 할 변이 61가지를 파일 11개(대체 경로 둘은 해당하는 47가지)에 넣어 모두 거부. logits 배율을 숨긴 변이 7가지 가운데 마지막 층의 가중치를 통째로 키운 것만 11개 파일 모두에서 통과(4.10절에 적은 한계). **(4차 검증 뒤)** 작업자(학습·내보내기 담당)가 같은 변이 61가지를 파일 18개에 넣어 모두 거부했고 계약 오류가 아닌 예외는 0개였다(3차 검증 때의 검증자 실행에서는 `TypeError` 14개). 정상적인 내보내기 71가지 통과. 속성을 망가뜨린 40가지(형 UNDEFINED, 정수, 문자열, 텐서, 값 없음, 두 번)는 고치기 전에 계약 오류 27, 받아들임 1, `TypeError` 12였고 고친 뒤 40가지 모두 한 줄의 계약 오류였다 |
| 합성 세션의 데모 | `python -m drowsy.demo --raw data/synth/Fold1_part1/01/10.parquet --out <임시 폴더>` | 종료 코드 0, `10_timeline.png` 213,579 bytes, `10_events.json` 10,799 bytes. 둘째 줄이 `rotation: no rotation-check record (synthetic session: ...)`이고 같은 문장이 JSON의 `source.rotation_note`에 있다 |
| 추출 결과의 데모 | 아래 추출로 만든 parquet으로 `demo --raw` | 종료 코드 0, 회전 줄 `rotation: the extraction's auto check chose 0, ambiguous: yes (...; no_face; ...) WARNING rotation ambiguous: no face was found at any angle ...` |
| 영상 데모 | `python -m drowsy.demo --video <얼굴 없는 시험 영상> --out <임시 폴더>` | 종료 코드 0, 2.8초. 첫 줄이 `rotation: auto check chose 0, ambiguous: yes (...)` |
| 이름이 같은 두 영상의 추출 | `python -m drowsy.extract --config <임시 설정> --video <폴더 a>/clip.mp4`, 이어서 `--video <폴더 b>/clip.mp4`(크기가 다른 다른 영상), 같은 명령 한 번 더, 그리고 b를 다른 폴더로 복사한 것 | 넷 다 종료 코드 0. 둘째 실행이 `replace clip.mp4: ... was extracted from <폴더 a>/clip.mp4 (46928 bytes); this video has 8930 bytes, so it is extracted again ...` 뒤에 다시 추출했고, 셋째는 `skip ... is up to date`, 넷째는 `skip ... is up to date (extracted from <폴더 b>/clip.mp4, a file of the same size as this video: 8930 bytes)` |
| 스텝 표 | `python -m drowsy.dataset --config config/default.yaml`을 세 번 | 종료 코드 0. 첫 실행 23.3초(캐시 형식 4로 세 조건을 다시 만듦), 다음 두 번 0.82초와 0.84초. 셋째 줄이 `rotation check of 182 raw file(s): ... 182 without a rotation-check record`(합성 데이터에는 기록이 없다) |
| 캐시 지문의 비용 | 지문 함수만 재는 저장소 밖 스크립트(3차 검증 뒤의 정비에서 쓴 것을 복사), `data/synth` 182개 파일 443,433,671 bytes | 첫 호출 224 ms, 페이지 캐시에서 파일을 내보낸 뒤의 첫 호출 406 ms, 그 뒤 20번의 중앙값 5.7 ms |
| 교란 점검 | `python -m drowsy.baseline_lgbm --config config/default.yaml --run-name fix_r4_docs probes` | 종료 코드 0, 20.4초. 둘째 줄이 회전 점검 줄, `probes.json`에 `rotation_check`(182개 모두 기록 없음, 표시된 영상 없음) |
| 축소 설정 교차검증 | `python -m drowsy.train --config config/default.yaml --run-name fix_r4_docs cv --fast` | 종료 코드 0, 99.2초. 둘째 줄이 회전 점검 줄, `run.json`에 `rotation_check`, summary.md 머리말에 D 번호 안내 줄. 연기 시험이라 수치는 결과가 아니다 |
| E0 도구 | `python -m drowsy.signal_check e0 --config <임시 설정> --run-name fix_r4_docs` | 종료 코드 0, 10.1초, `E0 criteria ...` 줄 다음에 회전 점검 줄. 합성 데이터라 기준 충족은 파이프라인 점검일 뿐이다 |
| 깜빡임 길이 오차 모사 | `python -m drowsy.signal_check blink-sim --config <임시 설정> --run-name <이름>` | 0.6초, 아래 표. `blink_sim.csv`는 2차 검증 뒤 정비의 실행과 바이트 단위로 같다. **(4차 검증 뒤)** 이 문서 단계가 다시 돌려(`nice -n 19`, 0.67초) 같은 파일을 얻었다(sha256 f4fb0029…). 통합 단계가 쓴 사본이 있던 `reports/fix_round4/`는 지웠다 |
| 라벨 없는 영상 평가 | `python -m drowsy.evaluate unlabeled --config <임시 설정> --raw-dir data/synth/Fold1_part1/01 --run-name fix_r4_docs` | 종료 코드 0, 3개 중 3개를 다시 돌림, 마지막 줄이 회전 점검 줄. summary.md 다섯째 줄이 D 번호 안내 줄이고 속도 항목이 `속도 억제(PLAN 1.2절 결정 D3)`로 적힌다. 합성 세션이라 수치는 싣지 않는다 |
| 합성 데이터 생성 | `python -m drowsy.synth --config config/default.yaml --out <임시 폴더> --subjects 2 --minutes 1` | 종료 코드 0, 세션 8개, 0.7초 |
| 실행 이름의 거부 | `python -m drowsy.train --config <임시 설정> --run-name ../fix_r4_escape cv`, `python -m drowsy.baseline_lgbm --config <임시 설정> --run-name a/b probes` | 둘 다 종료 코드 2, 한 줄 `error: --run-name must be a plain folder name, got '...'` |
| `.gitignore` | `git check-ignore -v --no-index`로 예시 경로를 물음(저장소 상태는 바꾸지 않는다) | `config/local.yaml.bak`, `config/locale.yaml`, `config/localhost.txt`, `tests/videos/readme.md`, `docs/videos/x.txt`, `clip.Mp4`, `clip.MP4`, `a/b/c.mKv`, `path.mp4`가 모두 제외된다 |
| 모든 명령의 `--help` | 모듈과 하위 명령 24개(`python -m drowsy.live` 포함) | **(4차 검증 뒤)** 이 문서 단계: 24개 모두 종료 코드 0. `drowsy.live`의 출력은 그 도구를 고친 작업자가 남긴 출력과 같다 |
| MediaPipe 일회성 점검 | 12.2절 | 이번에는 다시 돌리지 않았다(2026-11-14의 앞 정비에서 돌린 출력이 2026-11-13과 같았다) |
| 실시간 시험 도구(4.13절) | `python -m pytest tests/test_live.py -q`. 4차 검증 뒤에 도구를 고친 작업자의 실행: 저장소 밖의 PyAV 보내는 쪽(libx264 ultrafast, zerolatency, 키프레임 30프레임마다, MPEG-TS를 TCP로, 얼굴 없는 640x480 그림, 30 fps)으로 `python -m drowsy.live --source tcp://127.0.0.1:<포트>?listen=1 --port 0 --out <임시 폴더>`, 세션 뒤 `python -m drowsy.demo --raw <세션의 표>` | **(4차 검증 뒤)** 이 문서 단계: `75 passed in 55.71s`(시험 75개. 셋은 `face_landmarker.task`로 실제 랜드마커를 돌리고 하나는 node로 페이지 스크립트를 돌린다. 3차 검증 뒤에 이 도구를 더한 작업의 기록은 69개). 도구를 고친 작업자: 보내는 쪽이 300프레임을 보내고 멈춘 세션은 종료 코드 0, `frames: decoded 270, ..., failed 0 (processing error), processed 270, with a face 0`, 출력 폴더에 세 파일만(34,714 / 130,666 / 2,989 bytes), `demo --raw`로 다시 돌린 이벤트와 경고가 세션의 것과 같았다. Ctrl+C로 끝낸 세션은 종료 코드 0, 신호부터 끝까지 0.87초, `decoded 182 = left 1 + processed 181`. 접속만 하고 영상을 보내지 않는 쪽 둘(아무것도 안 보냄, 글자를 보냄)은 접속 시간을 8초로 줄인 사본에서 두 경우를 함께 적은 오류와 종료 코드 2. 기다리는 동안의 SIGTERM·SIGHUP은 빈 출력 폴더를 남겼다. 3차 검증 뒤의 기록: 200초 세션 6,000프레임 처리, 버린 프레임 0, 지연 중앙값 71.8 ms(4.13절 "잰 것"). 실제 카메라, Windows의 ffmpeg, 브라우저, 실제 얼굴로는 돌리지 않았다 |

표의 추출과 데모 실행은 얼굴이 없는 시험 영상으로 돌렸으므로 회전 점검이 모두 "얼굴 없음, 가르지 못함"으로 끝났다. 회전 줄과
표시가 전달되는 길은 확인했지만 실제 얼굴로 각도를 고르는 것은 이 표에서 확인하지 않았다. 그것은 4.5절의 측정과, 얼굴 이미지가
있어야 도는 시험 셋(4.3절의 명령이 고르는 열하나 가운데 셋)이 보였다.

깜빡임 길이 오차 모사(가정한 깜빡임 파형, 합성. 눈을 잰 값이 아니고 모델 성능도 아니다). 오차 = 검출기의 `duration_ms` −
진입 임계값을 넘은 참 시간, 모든 길이와 진폭을 합친 값이다. 길이별 표는 `docs/SPEC.md` 17.3절에 있다.

| 조건 | 보간함: 검출 / 치우침 / 절대 오차 중앙값 / 95 분위수 | 프레임 시각: 검출 / 치우침 / 중앙값 / 95 분위수 |
|------|------|------|
| 30 fps | 1050/1050, −0.9 ms, 1.0 ms, 4.0 ms | 809/1050, −31.9 ms, 33.0 ms, 50.0 ms |
| 15 fps | 951/1050, −3.6 ms, 3.0 ms, 19.0 ms | 704/1050, −61.1 ms, 67.0 ms, 100.0 ms |
| 15 fps, 20% 누락 | 3502/4200, −6.0 ms, 6.0 ms, 43.0 ms | 2473/4200, −81.1 ms, 67.0 ms, 167.0 ms |

**4차 검증 뒤의 정비에서 통합 단계가 기록한 것**(2026-11-15 00:03~01:02, 이 차례의 마지막 코드, 이 PC. 문서 작성자가 다시
돌리지 않은 것. 기록은 그 단계의 결과와 로그에서 옮겼다. 긴 명령은 모두 가장 낮은 우선순위 `nice -n 19`로 돌렸으므로 3차
검증 뒤의 기록보다 느리다):

- 전체 테스트를 처음과 끝에 한 번씩 돌렸다: `1743 passed, 3 skipped in 386.88s (0:06:26)`, `1743 passed, 3 skipped in 393.98s
  (0:06:33)`(시험 1,746개, 파일 23개. 건너뛴 셋은 4.3절의 얼굴 이미지 시험). 고칠 것은 없었고 저장소의 파일을 하나도 고치지 않았다.
- 처음과 끝에 바뀌지 않았음을 확인한 것: `pipeline_hash` 650495e5…, `rules_hash` 9c2d05f2…, `feature_spec_hash` f4031d09…,
  `export/`의 파일(`drowsiness_v0.onnx` sha256 eb2f71a6…, 148,558 bytes, `feature_spec_v0.json`, `rules.json`, `README.md`. 설명서
  `README.md`는 그 뒤에 이 문서 단계가 고쳤다), 이식
  대상 모듈 일곱과 `config/`(2026-11-14 17시 30분쯤 떠 둔 저장소 사본과 바이트 단위로 같다). `vectors verify`는 102개 모두
  통과했고, 다시 만든 벡터는 102개 30,588,827 bytes로 `tests/vectors/`와 바이트 단위로 같았다. 그 사본에 있던 101개는 그대로이고
  4차 검증 전의 89개는 합계 27,983,687 bytes다. v0을 임시 폴더로 다시 내보낸 파일도 `export/`의 v0과 같았다.
- 스텝 표 캐시를 모두 다시 만들었다(32개 파일, 캐시 형식 4. `drowsy/spec.py`와 `drowsy/dataset.py`가 바뀌어 캐시의 코드 지문이
  달라졌기 때문이다). 걸린 시간(합성 데이터의 파이프라인 시간, 이 PC): 원본 세 조건 23.4초, 15 fps 세 조건 18.2초, 프레임
  누락 변형 0의 세 조건 26.6초(1,282,233프레임 가운데 254,111개를 뺐다). 캐시가 있을 때 `python -m drowsy.dataset --config
  config/default.yaml`은 M2가 도는 동안 0.97초와 0.94초였다.
- 합성 데이터 전 과정(M2)을 다시 돌렸다: `python -m drowsy.baseline_lgbm --config config/default.yaml --run-name synth_m2
  probes`, 같은 설정의 `baseline_lgbm ... cv`, `train ... cv`, `train ... final --out-dir out/synth_m2`. 네 명령 모두 종료 코드 0.
  **걸린 시간은 이 PC(GPU 하나)에서 합성 데이터로 잰 파이프라인 시간일 뿐 성능이 아니다**: 교란 점검 20.8초, LightGBM
  교차검증 33.2초, 중첩 교차검증 1,690.4초(그 가운데 안쪽 선택 1,297.1초, 다시 학습과 시험 87.4초, 15 fps 평가 1.5초, fold
  모델의 INT8 14.9초, 리포트 쓰기 297.6초), 최종 학습 272.1초(선택 265.5초, 학습 1.2초, 내보내기 0.7초), 합계 2,016.5초(00:12:36~
  00:46:12. 3차 검증 뒤에는 1,970.5초). 내보낸 모델(`out/synth_m2/drowsiness_v1.onnx`, sha256 1a522878…), 체크포인트, 예측 파일
  둘(`predictions.parquet`, `lgbm/predictions.parquet`), 선택 기록 둘(`selection.json`)과 `out/synth_m2/`의 세 파일이 그 전의
  M2 실행과 바이트 단위로 같았고, 실행이 쓴 파일 43개 가운데 35개가 같았다. 다른 8개는 설정 사본 둘(`config/default.yaml`의
  `extraction.rotation_overrides` 주석 하나를 3차 검증 뒤의 문서 단계가 그때의 M2 실행 뒤에 고쳤다. 값은 그대로다), `final/final.json`(시간과 지연의 값만), `final/summary.md`(새
  D 번호 안내 줄과 지연 줄), `summary.md`와 `lgbm/summary.md`(고친 안내 줄만), `timings.json` 둘이다. 이번 정비가 학습 결과를 바꾸지
  않았다는 뜻이다. 체크포인트는 v2(`synthetic` 참, `fast` 거짓)이고, 체크포인트를 임시 폴더로 다시 내보낸 모델도 같았다.
  `reports/synth_m2`는 파일 40개 21,233,682 bytes, `out/synth_m2`는 3개 37,851 bytes다.
- 축소 설정(`--fast`)의 주 실행, 부가 실행 네 가지, `cv --all-w`, `final --fast --out-dir out/fix_round5`가 실행 이름
  `fix_round5`로, 내보낼 조건이 아닌 조건의 실행(`training.export_arm: global`인 임시 설정)이 실행 이름 `fix_round5_other_arm`으로
  모두 종료 코드 0으로 돌았다(합계 449.4초). 출력은 `reports/fix_round5/`(파일 124개), `reports/fix_round5_other_arm/`(34개),
  `out/fix_round5/`(3개)에 있고, 축소 설정의 연기 시험이라 언제 지워도 된다. 수치는 결과가 아니다. 이번에는 `signal_check e0`,
  `blink-sim`, `evaluate unlabeled`, 추출, 데모를 통합 단계에서 다시 돌리지 않았으므로 그 출력 폴더는 없다.
- 바뀐 것을 하나씩 확인했다: D 번호 안내 줄은 D 번호를 쓰는 남은 .md 리포트 14개에서 모두 일곱째 줄에 한 번씩 있고 D 번호를
  쓰지 않는 리포트에는 없다. `pipeline.yawn.cancel_gap_ms`나 `pipeline.nod.cancel_gap_ms`를 20000으로 바꾼 설정은 두 값을 적은
  오류로 거부됐다. 속성 형이 UNDEFINED인 Gemm은 `ContractError`로 끝났다. 스텝 표 명령은 회전 점검 줄을 내고 캐시에서 읽었다.
- 소유자 결정 1~4와 결정 5의 확인 각본(8장) 여섯 개는 모두 종료 코드 0이고 출력의 크기가 3차 검증 뒤에 저장한 출력과 같았다
  (그 출력 파일은 PC를 다시 시작하며 없어져 내용은 견주지 못했다). 결정 5의 각본은 24.1초, 72.0초, 36번, 정확한 30 fps 시각에서
  68.1초를 냈다. 결정 1의 감김 조각 점검은 결정 전 코드의 사본이 없어져, 지금 검출기를 결정 전의 정의와 견주는 판으로 바꿔
  돌렸다(같은 무작위 세션의 구간 99,649개에서 불일치 0. 일부러 틀리게 한 판은 세션 20개에서 불일치 5,403개로 걸렸다).
- 피험자 겹침을 일부러 넣은 네 경우가 모두 거부됐다. 예측 묶음 72개(파일 11개) 가운데 부가 표시가 빠진 것은 0개였다.
- 지운 것: 앞 차례의 연기 시험 출력 네 폴더(`reports/fix_round4/`, `reports/fix_round4_other_arm/`,
  `reports/fix_round4_speed80/`, `out/fix_round4/`, 파일 180개). git에 넣을 후보는 167개(앞 차례의 151개, 그 뒤에 더한 파일 셋
  (`drowsy/live.py`, `tests/test_live.py`, `docs/GUIDE.md`), 이 차례의 새 벡터 13개. `git status --short --untracked-files=all`로 셌다)이고,
  `reports/`와 `out/` 아래에서 git이 추적할 파일은 `reports/.gitkeep` 하나뿐이다. `out/demo/`, `out/live/`, `videos/`는 건드리지
  않았다.
- 합성 데이터로 만든 리포트는 모두 첫 줄에 합성 데이터임을 적었고, 이 README, `docs/`, `export/`에는 그 실행의 점수가 없다.

정비 작업자들이 기록한 검사: 이식 코드의 변경마다 독립 참조 구현과의 무작위 비교에서 불일치 0, 변이 검사(일부러 넣은
결함이 시험에 걸리는지)가 모두 걸렸다(결함 수정 18/18, 결정 1 21/21, 결정 2~4 35/35, 오프라인 평가 재시작과 경고 종류
23/23). 2차 검증 뒤의 정비에서: 이식 모듈 세 개에 대한 변이 563개(`blink.py` 101, `rules.py` 398, `pipeline.py` 64) 가운데
462개를 벡터 재검증이, 43개를 pytest만이 잡았고, 남은 58개는 모두 동작이 같은 변이(같은 값을 내는 변경)로 분류됐다.
`docs/SPEC.md`만 보고 옮겨 적은 구현이 벡터 81개를 모두 통과했고(프레임 18,216개를 엄격히 비교해 불일치 0), 일부러 망가뜨린
규칙 21개를 모두 벡터가 잡았다. 계약 검사에 대한 변이 53개(파일 10개)에서 거부해야 할 것을 받아들인 경우는 없었고, 추출·데모의
새 동작에 대한 변이 16개는 모두 시험에 걸렸다.

3차 검증 뒤의 정비에서 작업자들이 기록한 검사(작업자의 기록. 문서 작성자가 다시 돌린 것은 위 표에 따로 적었다):
- 이식 모듈 세 개에 대한 변이(같은 변이 563개): 벡터 재검증이 463개(`blink.py` 72, `rules.py` 336, `pipeline.py` 55)를, pytest만이
  42개를 잡았고 58개가 남았다. 조건을 늘 참이나 거짓으로 두기, 호출 한 줄 지우기 같은 변이를 더한 판(341개 가운데 올바른
  Python 339개)에서는 벡터가 302개, pytest만이 24개를 잡고 13개가 남았다. 남은 변이는 출력이 같은 것으로 분류됐다(`docs/SPEC.md`
  16.1절 끝의 목록). 틀린 한 줄 여덟 자리(검증자가 짚은 일곱과 작업자가 더 찾은 하나)를 참조 구현의 사본에 하나씩 넣으면
  그때마다 새 벡터 가운데 하나가 실패했다(새 벡터 일곱이 쓰였다). 여덟째 새 벡터(`rules_drowsy_off_step_boundary`)는 같은
  점검을 되풀이하다 찾은 자리를 고정하고, SPEC만 보고 옮긴 구현에서 그 규칙을 망가뜨리면 실패했다.
- 스텝 표 캐시의 내용 지문과 리포트의 D 번호 안내에 대한 변이 21개, 추출과 데모의 새 동작(원본 크기 비교, 처리 시간의 출처,
  `--raw` 데모의 회전 줄)에 대한 변이 42개가 모두 시험에 걸렸다.
- `docs/SPEC.md`만 보고 옮겨 적은 구현이 벡터 89개를 모두 통과했고(프레임 18,974개를 엄격히 비교해 불일치 0), 일부러 망가뜨린
  규칙 21개를 모두 벡터가 잡았으며, SPEC 16.1절이 이름을 댄 벡터는 그 절의 규칙을 망가뜨린 9가지에서 모두 실패했다.
- 누설 회귀 시험: 4.3절 끝.

4차 검증 뒤의 정비에서 작업자들이 기록한 검사(작업자의 기록. 문서 작성자가 다시 돌리지 않았다):
- 한 줄 바꾸기 점검(변이 점검)을 이식 대상 모듈 여섯(`signals.py`, `calibrate.py`, `step_features.py`, `blink.py`, `rules.py`,
  `pipeline.py`)에 두 변이 생성기(기본 판과, 조건을 늘 참이나 거짓으로 두기·호출 지우기를 더한 판)로 모두 돌렸다(3차 검증 뒤에는
  뒤의 셋만). 변이 1,420개(올바른 Python이 아닌 2개 포함) 가운데 벡터 재검증이 1,185개를 잡았고(4차 검증 전의 벡터 89개로는
  1,127개), pytest만이 116개를 잡았고, 117개가 남았다. 모듈별 표는 `docs/SPEC.md` 16.1절의 `mutation-check` 표에 있다. 새로 다룬
  세 모듈의 수(변이 / 벡터가 잡음(89개 → 102개) / pytest만 / 남음): `signals.py` 기본 45 / 37 → 40 / 5 / 0, 확장 18 / 10 → 10 / 8 / 0;
  `calibrate.py` 기본 101 / 59 → 74 / 17 / 10, 확장 61 / 41 → 50 / 9 / 2; `step_features.py` 기본 178 / 119 → 140 / 8 / 30, 확장
  113 / 96 → 106 / 3 / 4. 나머지 셋은 3차 검증 뒤와 같다(`blink.py` 101 / 72 / 16 / 13과 53 / 50 / 2 / 1, `rules.py` 398 / 336 /
  22 / 40과 233 / 210 / 12 / 9, `pipeline.py` 64 / 55 / 4 / 5와 55 / 42 / 10 / 3). 남은 117개는 모두 무작위 비교에서 출력 차이가
  없었고, 새 세 모듈의 46개는 줄마다 같은 값을 내는 까닭을 적었다(뒤의 세 모듈의 71개는 비교만 다시 돌렸다). pytest만이 잡는 새
  세 모듈의 50개는 SPEC 16.1절의 "단위 시험만 고정하는 동작" 목록의 항목에 하나씩 대응한다.
- `docs/SPEC.md`만 보고 옮겨 적은 구현(5, 8, 9, 11장은 새로 옮겼다)이 벡터 102개를 모두 재현했고, 무작위 세션 700개(파이프라인
  프레임 약 307만 개)에서 저장소의 코드와 차이가 없었다. 그 과정에서 SPEC의 빈틈 하나(하품·끄덕임 계수기의 `reset()`이 하는 일,
  11.5절)를 찾아 의사 코드 두 줄로 채웠다(동작은 그대로). 문서의 규칙 37개를 하나씩 망가뜨리면 그때마다 그 규칙에 이름이 붙은
  새 벡터가 실패했다.
- 새 벡터 13개마다 그 벡터가 막으려는 틀린 한 줄을 넣으면 생성기가 벡터를 쓰기를 거부하고(57가지), 생성기의 검사를 빼고 만든
  파일은 손으로 계산한 시험에서 실패했다.
- 내보내기의 계약 검사: 위 표의 "계약 검사의 변이" 행. logits를 내는 노드 검사의 한 조건씩을 빼 본 10가지 가운데 6가지는 이제
  `tests/test_export_onnx.py`가 잡는다. 남은 넷은 12.3절의 알려진 한계에 있다.
- 실시간 시험 도구: 최종 점검이 짚은 변이 셋(회전 방향을 시계 방향으로, 처리 오류 뒤 종료 코드 0, 받은 fps와 처리한 fps를 맞바꿈)은
  새 시험 셋 가운데 각자의 시험 하나에서만 실패했고, 이번에 고친 동작의 변이 셋(0초 알림을 냄, 실패한 프레임을 `left`로 셈, 다른
  소스에 "끝을 기다리라"고 안내)도 모두 시험에 걸렸다.

### 12.2 M0 일회성 점검: mediapipe 0.10.32 (도구 점검, 모델 성능 아님)

PLAN 11장 M0과 Q1이 요구한 일회성 점검을 2026-11-13에 스크래치 폴더에서 돌렸고 2026-11-14에 다시 돌려 출력이 같았다(저장소
밖의 스크립트 `oneoff_check.py`, 3.2초, 네트워크 없음, 저장소에 쓰지 않음). 입력은 MediaPipe 공식 저장소의 시험 이미지 두
장(계획 단계에서 스크래치에 받은 것)과 그것으로 만든 40프레임 움직이는 수열이다. 얼굴 이미지는 저장소에 넣지 않았다.

- 버전과 모델: `mediapipe 0.10.32`, 모델 sha256 `64184e22…bc9ff`(spec의 값과 같다).
- 출력 형식(VIDEO 모드): 모든 프레임에서 랜드마크 478개, 블렌드셰이프 52개(이름과 순서가 spec과 같다), 4×4 변환 행렬, 값이
  모두 유한하다.
- 공식 시험 이미지의 머리 각도(열 우선 행렬, PLAN 2.2절 식; `signals.euler_from_colmajor`와 같은 값): sergey.png 600×600은
  yaw +0.10°, pitch +0.12°, roll −0.50°, lenna.png 400×400은 yaw +19.60°, pitch +13.29°, roll −2.87°. 행렬을 행 우선으로
  잘못 읽으면 lenna가 yaw −20.66°, pitch −11.51°, roll +7.28°가 된다. 같은 이미지를 10번 넣은 뒤의 값: sergey
  −0.06° / −0.51° / −0.56°, lenna +19.53° / +12.66° / −2.75°.
- 첫 타임스탬프 0과 1의 차이(40프레임, 600×600): 간격 33 ms에서 랜드마크 최대 0.0143 px, 블렌드셰이프 최대 0.00015,
  각도 최대 0.0024°(모두 두 번째 프레임). 간격 67 ms에서 0.8345 px, 0.00949, 0.1511°(두 번째 프레임), 20번째 프레임
  0.00129 px, 마지막 프레임 0.00007 px. 같은 조건을 다시 돌리면 같고, 시작 시각 1과 5,000,000은 차이가 0이다.

### 12.3 확인하지 못한 것

- **실제 데이터가 없다.** UTA-RLDD로 추출, E0, 학습, 평가를 한 적이 없다. 보고할 성능 수치가 없고, D11(모델을 기본
  판정원으로 둘지)도 판정하지 않았다.
- **closure 상수와 감김 임계값은 실제 눈 감김으로 검증되지 않았다.** 조사에서 합성으로 감긴 눈 두 장에 식을 적용했을 때
  closure가 0.61과 0.85였고, 하나는 0.7과 0.8에 못 미쳤다(PLAN 4.3절). E0을 통과하기 전에는 마이크로슬립·수면·PERCLOS
  판정이 실제 눈에서 어떻게 동작할지 알 수 없다.
- **Android에서 실행한 적이 없다.** Kotlin 이식 코드, LIVE_STREAM 모드, GPU delegate, `onnxruntime-android`, 대상 기기의
  프레임률과 타임스탬프는 모두 미확인이다. Python의 VIDEO 모드와 LIVE_STREAM이 같은 결과를 낸다는 것은 조사 단계에서
  데스크톱 CPU로만 확인했다(PLAN 2.3절).
- 실제 휴대전화 녹화 파일과 RLDD 파일의 타임스탬프와 회전 처리. 회전 규칙은 공개 사진 두 장으로 만든 클립으로만 쟀다.
  2차 검증 뒤의 정비에서 회전 정보 없이 옆으로 누워 저장된 그 클립(정비 전 코드는 0도로 추출해 roll 중앙값의 절댓값이
  82.8~90.6도였다)을 `extract --input-dir`과 `--video`가 90도·270도로 세워 roll −0.6도·−3.0도, 얼굴 검출률 1.000으로
  추출했고, 데모도 같은 각도를 골랐다. 거꾸로 저장됐지만 거꾸로도 검출되는 얼굴의 클립은 0도(180도 틀림)로 남고 모든
  출력에 "가르지 못함"으로 표시됐으며, `--rotation 180`을 주면 똑바로 섰다(roll −0.6도). 그 밖의 추출과 데모 실행은 얼굴이
  없는 시험 영상으로만 했다.
- 안경, 선글라스, 가는 눈, 야간, 실제 아래 시선에서의 신호.
- 시선 규칙(전방 주시 영역, VATS)의 임계값. RLDD로는 검증할 수 없고 팀이 수집한 차량 영상이 필요하다.
- 약 15 fps에서의 깜빡임 속도·진폭 특징이 기기에서 실제로 어떻게 읽히는지(12.1절의 모사는 MediaPipe의 시간 평활을
  넣지 않았다).
- 프레임 누락 변형, 부가 실행, fold별 전역 상수 실행이 실제 데이터에서 걸리는 시간. 변형 하나는 전체 세션을 한 번 더
  재생하고, fold별 전역 상수는 전역 조건의 스텝 표를 다섯 번 더 만든다.
- 합성 데이터의 교란 점검에서는 "영상별 기준선 편차" 쪽이 우연 수준을 넘는 것으로 표시되어, 합성 데이터 신경망 리포트의
  설명 문단이 그 경고로 시작한다. 합성 생성기의 성질로 보이며 따져 보지 않았다. 실제 데이터에서 같은 표시가 나오는지 봐야 한다.

**알려진 한계**(동작을 바꾸지 않고 적어 둔 것):

- 눈꺼풀이 이탈 임계값과 진입 임계값 사이(규칙용 0.5~0.7, 특징용 0.3~0.5)에 머무는 동안에는 보통의 깜빡임 여러 개가 한
  감김 구간으로 이어지고 감긴 시간이 쌓인다. 합성 각본(눈꺼풀 0.6, 4초마다 0.9로 165 ms 깜빡임)에서 마이크로슬립이 24.1초,
  수면이 72.0초(33 ms 프레임 시계. 정확한 30 fps 시각이면 68.1초)에 나고, 그 뒤 수면 경고가 3초마다 되풀이된다(3분에 36번).
  소유자가 2026-11-14에 이 동작을 그대로 두기로 정했다(8장의 결정 5).
- 회전이 확인되지 않은 영상은 여섯 명령(스텝 표, E0, `baseline_lgbm`, `train`, 라벨 없는 영상 평가, `--raw` 데모)이 알리기만
  하고 빼지 않는다. 학습 명령은 데이터를 읽기 전에 알리지만 멈추지 않고 학습을 이어 가므로, 학습 전에 `python -m drowsy.dataset`으로
  목록을 본다(4.9절의 순서). 학습 명령은 원본 프레임률 추출의 기록만 읽는다. 두 추출 사이에 보정표를 고치고 한쪽만 다시 추출한 경우는
  `python -m drowsy.dataset --fps-sim`이 따로 보여 준다.
- `--raw` 데모는 설정의 보정표를 보지 않으므로, 보정표에 같은 각도를 적어 확인한 영상도 "가르지 못함"으로 알린다(4.12절).
- 표시 행렬에 뒤집기가 든 영상은 `demo --video`에서 종료 코드 2, `extract --video`에서 종료 코드 1(`FAILED` 줄)로 끝난다.
  메시지는 같다.
- 회전 점검의 표본은 2초 간격이라, 2초보다 짧은 영상은 프레임 한 장으로 판정한다. 규칙은 같지만 근거가 얇다.
- 이어 하기는 원본 파일의 크기만 비교한다. 크기가 바이트 단위로 같은 다른 영상은 가려내지 못하고, 크기가 적히기 전에 만든
  결과는 크기를 비교하지 않는다(4.5절).
- 추출 리포트의 이어 온 처리 시간 가운데 "이 리포트의 전 판" 줄은 동시 처리 수가 다른 여러 실행의 값을 합칠 수 있다(예: 일꾼
  수를 바꿔 가며 두 번에 나눠 한 전체 추출). 행마다 측정 조건을 남기려면 47열 `<태그>_videos.csv` 형식을 바꿔야 해서 하지
  않았다. 시험 추출의 값은 시험 추출 리포트에 그 영상이 아직 있는 동안은 따로 한 줄이다. 그러나 전체 추출 뒤에 영상 수를 줄여
  시험 추출을 다시 하면(예: `--pilot 3` 뒤에 `--pilot 2`), 다시 쓴 시험 추출 리포트에서 빠진 영상의 값은 첫 시험 추출이 하나씩
  처리하며 잰 값인데도 "이 리포트의 전 판" 줄에 여러 개씩 동시에 처리한 값들과 함께 들어간다(4차 검증의 검증자가 재현). 이어 온
  값은 모두 그 결과 파일을 만든 실행의 값이고 그 줄은 측정 조건을 모른다고 적으므로 틀린 값은 아니다. 처리 시간은 추출 시간을
  어림하는 참고값일 뿐이다.
- 실행 이름의 검사는 명령줄만 막는다. `run_nested_cv`, `fit_final`, `run_lgbm_cv` 같은 라이브러리 함수를 직접 부르면 경로가 든
  이름도 받는다.
- 스텝 표 캐시는 한 명령이 도는 동안 크기와 수정 시각을 그대로 둔 채 바뀐 추출 파일과 Python·NumPy 판의 변화를 알아채지
  못한다(4.7절. 그때는 `--rebuild`).
- 계약 검사는 마지막 층의 가중치를 통째로 줄이거나 키운 파일과 신경망 안쪽의 변경을 잡지 못한다. 그것은 체크포인트와의 일치
  검증이 지킨다(4.10절).
- 내보내기 보호 장치는 합성 데이터와 축소 설정 모델에만 걸린다. 패키지를 다른 곳에 복사해 쓰면 원래 저장소의 `export/`는
  설정에 절대 경로를 적을 때만 보호된다(4.10절).
- 계약 검사 가운데 logits를 내는 노드의 조건 넷은 하나씩 빼도 `tests/test_export_onnx.py`가 실패하지 않는다(4차 검증 뒤의
  정비가 확인했다): Gemm의 `transA`를 보지 않기, INT8 파일에서 둘째 MatMulInteger를 허용하기, INT8 Cast가 바꾸는 형을 보지
  않기, 편향 Add가 아무 상수나 더하게 두기. 4차 검증 뒤의 정비가 끝난 뒤 그 결과를 한 번 더 확인한 검증자가(2026-11-15) 조건을
  하나씩 뺀 사본에 공격 28가지를 넣어 보았다. 편향 Add와 Cast 형의 조건을 뺀 사본은 출력이 바뀐 파일을 어느 것도 받아들이지 않았다. `transA`를
  보지 않는 사본은 마지막 층의 입력을 Transpose로 뒤집고 0.2를 곱한 뒤 `transA = 1`로 읽는 파일을(확률 차이 최대 0.33), 둘째
  MatMulInteger를 허용하는 사본은 가중치를 두 배로 한 둘째 MatMulInteger를 Cast에 이은 파일을(최대 0.21) 받아들였다. 두
  변화는 고치지 않은 검사도 받아들이는 변형(마지막 층 앞에 곱하기(Mul)를 넣은 파일, 마지막 층의 가중치를 그 자리에서 두 배로
  한 파일. 위의 계약 검사 한계)과 파일마다 값이 같으므로, 두 조건이 따로 막아 주는 출력 변화는 없다. 시험이 그 조건을
  고정하지 않는다는 한계다.
- INT8 파일의 Cast 속성 `to`가 한 번만 있고, 속성 형이 INT이고, 값을 가졌는지의 확인은 시험이 고정하지 않는다(시험은
  Gemm의 속성만 망가뜨려 본다). 그 확인을 빼도 망가진 `to`는 대부분 계약 오류로 끝나고 받아들여지는 파일은 없다.
- 계약 검사(`export_onnx.verify_contract`)는 일부 망가진 파일에서 `ContractError`가 아닌 예외를 그대로 낸다. 마지막 Gemm의
  `transB` 속성이 없거나 값이 비어 있는 파일과 마지막 층 상수의 모양이 틀린 파일은 onnx의 `InferenceError`나 onnxruntime의
  오류로 끝난다. 그런 파일이 받아들여지는 일은 없고, 사용자가 준 파일을 이 검사에 넘기는 명령도 없다(Python에서 직접 부를
  때만 해당한다).
- 누설 회귀 시험(`tests/test_leakage.py`)은 평가받는 피험자의 alert 대 나머지 라벨만 쓰는 누설, 리포트를 쓰는 시점(`drowsy.evaluate`)의
  누설, `baseline_lgbm probes`와 `train final`의 누설을 잡지 못한다(4.3절).
- 기준선의 얼굴 시간에서 마감 직전 프레임이 대표한 시간을 200 ms까지만(얼굴 없는 프레임이면 0) 세는 규칙(SPEC 8.2절
  `held_before`의 `min` 가운데 `held` 쪽, `calibrate.py` 197~198줄)은 테스트 벡터가 고정하지 않고 저장소의 단위 시험 다섯만
  고정한다. 이 `min`을 빠뜨린 이식 코드는 벡터 102개를 모두 통과하지만 기준선 완성 시각이 참조 구현과 달라질 수 있다(무작위
  보정 세션 60개 가운데 4개의 결과가 달랐다). 새 보정 세션 없이 기존 벡터 `baseline_burst_frames`에서 프레임 하나를 빼고
  76개(약 0.14 MB)를 더하면 고정할 수 있으므로, 벡터를 다시 만들 때 더할 수 있다(`docs/SPEC.md` 16.1절). 벡터
  `baseline_exactly_300_frames`는 60초 마감에서 두 까닭(얼굴 프레임 299개, 왼쪽 눈의 표본 없음)으로 동시에 실패하므로, 왼쪽
  눈의 표본 수 검사만 빠뜨린 이식 코드도 그 벡터를 통과할 수 있다(오른쪽 눈의 같은 경우는 `baseline_odd_sample_count`가 따로
  고정한다).
- 하품과 끄덕임 계수기에 넘기는 유효 표시가 `face`라는 것(SPEC 11.5절. `closure_valid`나 `pose_valid`가 아니다)은 테스트
  벡터도 단위 시험도 고정하지 않는다. 고개를 40도 넘게 돌린 채의 하품과 끄덕임, 전역 상수로 대체된 기준선에서의 끄덕임이 든
  벡터가 없어서, 다른 표시를 넘긴 이식 코드는 벡터 102개를 모두 통과하지만 그런 장면에서 `yawn_count_60`과 `nod_count_60`이
  달라진다. 또 기준선의 분위수 순위 `ceil(q·n)`에서 q·n이 정수로 떨어지는 경우, 연장된 마감에서 얼굴 검출 비율의 분모(그
  마감 값), 기준 yaw가 음수일 때의 `off_axis` 판정(`abs`)은 벡터가 고정하지 않고 저장소의 단위 시험만 고정한다.
- 실시간 시험 도구(4.13절): 보내는 쪽을 기다리는 동안 kill하거나 창을 닫으면 빈 출력 폴더가 남는다. 접속 시간 초과의 오류
  문장은 "아무도 오지 않음"과 "접속했지만 MPEG-TS 영상이 아님"을 가르지 못해 둘 다 적는다. 시작 블록의 보내는 쪽 명령과 오류
  문장은 `localhost`를 적고 `127.0.0.1`을 권하는 안내는 `--help`와 이 README에만 있다. Windows의 ffmpeg로 영상을 보내 본 적이 없다.

### 12.4 아직 하지 않은 것

**실제 데이터가 있어야 하는 것**: M3 추출과 그 리포트 검토(회전 표시 영상의 수동 보정 포함), E0 판정과 상수 확정(spec v1,
`pipeline_hash` 변경, 벡터 재생성, 앱 팀 통보), E1~E4의 실행(LightGBM 기준선과 W 비교, 중첩 교차검증, 부가 분석, 최종 학습),
MODEL_CARD 지표 표 채우기, D11 판정, 팀 수집 차량 영상의 시간당 경고 수.

**Android 쪽에서 해야 하는 것**: Kotlin 이식과 벡터 대조, LIVE_STREAM과 실기기 프레임률·타임스탬프, 기기에서의 추론 지연과
Python 대비 출력 일치.

**소유자의 답이 있어야 하는 것**: 9장의 질문 16개(2~17번. 1번은 2026-11-14에 결정되어 8장의 결정 5가 됐다).

## 13. Android 팀이 가져갈 것

### 13.1 파일

| 가져갈 것 | 어디에 | 쓰임 |
|-----------|--------|------|
| `docs/GUIDE.md` | `docs/` | **먼저 읽는 안내서.** 모델의 구조, 저장소와 문서를 읽는 순서, Android 앱에 잇는 법(6장: 번들할 파일, 시작할 때의 검사, MediaPipe 설정, 프레임마다 부르는 법, 체크리스트). SPEC.md와 코드를 따라 쓴 요약이고, 둘이 다르면 SPEC.md가 맞다 |
| `docs/SPEC.md` | `docs/` | **이식 명세.** 앱이 지킬 전제, 숫자 규칙, 자료형, 모듈별 의사 코드, 해시 검사, 벡터 쓰는 법, 결정을 기다리는 동작(20장). 문서와 코드가 다르면 코드와 벡터가 맞다 |
| `drowsiness_v{N}.onnx` | `export/` | 모델. 지금은 v0(무작위 가중치)으로 연동만 확인한다 |
| `feature_spec_v{N}.json` | `export/` | 모델과 짝인 명세. 스텝 특징의 순서와 정의, 이식 코드의 상수 전부(`pipeline` 블록) |
| `rules.json` | `export/` | 규칙 엔진과 경고의 임계값 |
| `tests/vectors/*.json` 102개 | `tests/vectors/` | 옮긴 코드의 출력을 맞춰 볼 입력과 기댓값 |
| 참조 구현 | `drowsy/types.py`, `signals.py`, `calibrate.py`, `blink.py`, `step_features.py`, `rules.py`, `pipeline.py` | SPEC.md가 설명하는 코드. 상수를 담는 자료형은 `drowsy/spec.py`의 `PipelineSpec`과 `RulesConfig`다 |

해시와 파일의 짝은 `export/README.md`에, 벡터의 형식·허용 오차·쓰는 순서는 `tests/vectors/README.md`("Android 팀이 쓰는 법")에 있다.
계획서 9장의 시그니처와 7장의 파일 이름(`export/feature_spec.json`)은 구현 전의 것이다. 지금의 것은 SPEC.md와
`export/feature_spec_v{N}.json`이고, 달라진 점은 계획서 14장에 있다.

### 13.2 순서

1. `export/`의 세 파일을 앱에 번들한다. 시작할 때 `docs/SPEC.md` 15장의 다섯 가지를 비교하고, 하나라도 다르면 판정을
   시작하지 않는다.
   - H1: 앱이 쓰는 `face_landmarker.task`의 sha256 = 모델 메타데이터의 `mediapipe_model_sha256` = `feature_spec`의
     `pipeline.mediapipe.model_sha256`.
   - H2: 번들한 `feature_spec_v{N}.json` 바이트의 sha256 = 메타데이터의 `feature_spec_hash`.
   - H3: 파일 안의 `pipeline_hash` = 메타데이터의 `pipeline_hash` = 이식 코드를 벡터로 맞출 때의 `pipeline_hash`.
   - H4: 번들한 `rules.json` 바이트의 sha256 = 벡터의 `rules_hash`(빌드에 넣어 둔 값. 모델 메타데이터에는 없다).
   - H5: 메타데이터의 `step_sec`(`1`), `window_steps`(`120`), `class_order`(`alert,low_vigilant,drowsy`), `standardization`
     (`in_graph`)이 SPEC 14.3절의 값.
2. `docs/SPEC.md`를 보고 이식 대상 모듈을 옮긴다. Python 모듈은 참조 구현이다. 이 모듈들은 표준 라이브러리만 쓴다(numpy를
   import하면 실패하는 테스트가 있다). 연산은 배정밀도(`Double`), 시간은 정수 ms(`Long`), 수학 함수는 `sqrt`, `atan2`,
   `hypot`, `sin`, `cos`, `floor`, `ceil`뿐이다(SPEC 2장. `asin`, `acos`, `exp`, `pow`는 쓰지 않는다). 반올림은
   `floor(x + 0.5)`이고 분위수의 순위는 `ceil(q * n)`이다. 상수는 코드에 적지 말고 `feature_spec_v{N}.json`의 `pipeline`
   블록과 `rules.json`에서 읽는다. 설정을 읽을 때 셋을 검사하고, 어기면 설정 오류로 시작하지 않는다.
   - 두 감김 검출기 모두에 `0 < max_dt_ms ≤ hold_ms < clock.reset_gap_ms`(SPEC 10.1절. 뒤의 부등식은 3차 검증 뒤에 더해졌다.
     지금 값 200, 500, 10,000은 통과한다).
   - 하품과 끄덕임 계수기에 `yawn.cancel_gap_ms < clock.reset_gap_ms`, `nod.cancel_gap_ms < clock.reset_gap_ms`(SPEC 10.1절과
     11.5절. 4차 검증 뒤에 더해졌다. 지금 값 500과 10,000은 통과한다).
   - `rules.json`의 `clock` 블록이 `feature_spec`의 `clock` 블록과 같을 것(SPEC 7.1절, 15장. 다르면 `DriverPipeline`을 만들 때
     오류다).
3. 벡터를 묶음 순서대로 맞춘다: `signals` → `baseline` → `runs` → `steps` → `rules` → `model` → `e2e`. 벡터의
   `pipeline_hash`와 `rules_hash`가 번들한 파일의 값과 다르면 상수가 다른 것이므로 비교를 멈춘다. 102개가 모두 통과해도 확인되지
   않는 동작(표시용 `gaze_zone`, 첫 `t_ms`가 0인 세션, 실수 임계값에 정확히 놓인 비교 연산자 등)의 목록이 SPEC.md 16.1절 끝에
   있으니, 그 자리는 SPEC.md의 의사 코드와 연산자를 그대로 옮긴다.
4. MediaPipe에 넘기는 프레임의 전제(6장의 6번)를 지킨다. 변환 행렬 `float[16]`은 열 우선이다(`M[i][j] = m[4*j + i]`).
5. 모델에는 표준화하지 않은 원시 스텝 특징 `[1, 120, 17]`을 가장 오래된 스텝부터 넣는다. 넣기 전에 값이 모두 유한한지
   보고, 아니면 추론을 건너뛴다. 한 프레임이 스텝을 여러 개 내면 스텝마다 그 시점의 창으로 부른다(결정 3). onnxruntime은
   스레드 1개를 권장한다.
6. `DriverPipeline.update`는 `t_ms`가 직전에 받아들인 프레임 이하이면 그 프레임을 버리고 `null`을 돌려준다.

### 13.3 정비에서 해시 변경 없이 바뀐 동작

정비에서 이식 코드의 동작이 여덟 군데 바뀌었고 설정 검증 조건이 세 차례 더해졌다(아래 표의 둘째와 셋째 행). `pipeline_hash`와
`rules_hash`는 그대로인데 결과가 바뀌는 변경이다.
모두 `docs/SPEC.md`와 벡터에 들어 있다. **벡터는 해시가 같아도 내용이 달라졌으므로 폴더를 통째로 다시 받는다**: 처음 43개
가운데 6개가 바뀌었고(`rules_vats`, `runs_lingering_lids`, `rules_drowsy_model_stub`, `e2e_250s`, `rules_calibrating_provisional`,
`runs_dropped_frames`) 새 벡터 25개가 더해져 68개가 됐다. 2차 검증 뒤의 정비는 동작을 바꾸지 않고 벡터만 13개 더했다(68개는
바이트 단위로 그대로, 81개). 3차 검증 뒤의 정비도 동작을 바꾸지 않고 벡터 8개를 더했다(81개는 바이트 단위로 그대로, 89개). 4차
검증 뒤의 정비도 동작을 바꾸지 않고 벡터 13개를 더했다(89개는 바이트 단위로 그대로, 지금 102개. 기준선, closure, 하품·끄덕임,
스텝 특징의 경계). 새 벡터는 끊김 경계, 미뤄 둔 경고, 감김 검출기의 정점과 잘림, 30초 창의 경계, 얼굴을 잠깐 놓친 구간의 `confidence`
같은, 전에는 단위 시험만 고정하던 동작을 벡터로 고정한다(SPEC.md 16.1절).

| 무엇 | 바뀐 것 | SPEC.md |
|------|---------|---------|
| 감는 속도의 되돌아보기 | 쓸 수 없는 쌍(직전 유효 표본이 없거나 간격 > `max_dt_ms`) 뒤의 유효 표본에서 `recent_up` 버퍼(최근 상승률)를 모두 0으로 비운 뒤 밀고 넣는다. 특징 #7, #8 | 10.2, 10.5절 |
| `max_dt_ms ≤ hold_ms < clock.reset_gap_ms` | 두 감김 검출기 설정의 필수 조건. 아니면 설정 오류. 앞의 부등식은 첫 정비, 뒤의 것은 3차 검증 뒤의 정비가 더했다(지금 값은 통과한다) | 10.1절 |
| `yawn.cancel_gap_ms`, `nod.cancel_gap_ms` < `clock.reset_gap_ms` | 하품·끄덕임 계수기 설정의 필수 조건. 아니면 두 값을 적은 설정 오류. 4차 검증 뒤의 정비가 더했다(지금 값 500과 10,000은 통과한다). 계수기의 `reset()`이 하는 일도 그때 SPEC 11.5절의 의사 코드에 적었다(동작은 그대로) | 10.1, 11.5절 |
| 보간 시각의 반올림 | `prev_t + round_half_up(fraction × (t − prev_t))`. 계산 순서(나눗셈, 곱셈, `floor(x + 0.5)`, 정수 더하기)를 지킨다. 시각 원점에 따라 1 ms가 달라지던 문제가 없어졌다 | 2, 10.2절 |
| VATS `end`의 confidence | VATS 시작 때 `vats_pose0 = pose_ms − pose_window.since_ms(vats_start)`를 두고, 모든 VATS 이벤트의 confidence = (pose_ms − vats_pose0) / (t − vats_start). 30초보다 긴 VATS의 `end`만 바뀐다 | 12.4, 12.9절 |
| 얼굴 소실 인정 구간에서 된 시선 이탈 | 시작 조건이 `zone == OUT && (label == OUT \|\| (label == UNKNOWN && analyzable && !candidate_pending))`. 인정 구간 안에 프레임이 없었어도 다음 분석 가능 프레임에서 `gaze_away_long`이 시작한다. `!candidate_pending`은 구현이 더한 조건이고 9장 9번에서 확인을 기다린다 | 12.4절 |
| 감김 길이(결정 1) | 8장 | 10.2~10.5, 11.4, 12.5절 |
| 막힌 경고(결정 2) | 8장. 대기 중 표와 버리는 규칙 | 13장 |
| 스텝마다 모델 호출(결정 3) | 8장. `PipelineOutput.step_probs`, 벡터의 `expected.step_probs` | 3.8, 7.2절 |
| 10초 넘는 끊김 뒤 재시작(결정 4) | 8장. `StepHysteresis.reset()`, 끝내는 순서 | 6, 12.3, 12.6~12.8절 |

바뀌지 않았지만 문서로만 남기기로 한 동작: 2~10초 끊긴 뒤의 프레임이 한꺼번에 내는 스텝들은 끊기기 전 마지막 프레임의
`ongoing_ms`를 쓴다(이중으로 세지는 않는다). 그래서 끊기기 직전의 감김이 `max_closure_ms_10`에 "끊긴 시간 + 10초" 동안
남고, 유효 시간이 줄어든 `perclos_10`은 끊김 뒤에 최대 `감긴 ms / 2000`까지 오른다(유효 시간 2초 미만이면 0). 검토의 재현:
0.8초 감김 뒤 7.7초 끊김에서 `perclos_10`이 0.10에서 0.40으로 올랐다. SPEC.md 11.6절과 벡터 `steps_closure_across_gap`에 있다.

### 13.4 바뀔 때 알려 줄 것

- E0에서 잠정 상수를 확정하면 `pipeline_hash`가 바뀐다(spec v1). `feature_spec`, `rules.json`, 벡터와 함께 그 spec으로 다시
  내보낸 모델 파일도 다시 받아야 한다. 모델 메타데이터에 `pipeline_hash`와 `feature_spec_hash`가 들어 있어서, 옛 모델을 새
  spec과 함께 번들하면 시작할 때의 검사(13.2절의 H2, H3)에서 판정이 시작되지 않는다. 빌드에 넣어 둔 해시도 바꾼다.
- 학습한 모델이 나오면 `drowsiness_v1.onnx`, `feature_spec_v1.json`(TCN이면 `drowsiness_v1_int8.onnx`)이 추가되고 모델
  벡터를 다시 만든다. 입출력 계약은 그대로다.
- 9장의 질문 가운데 앱에 걸린 것(2~9번은 이식 코드, 10번은 spec 파일 이름, 11번은 v0 모델 파일, 12번은 벡터 형식. SPEC.md
  20장의 같은 번호. 1번은 결정되어 비워 두었다: 8장의 결정 5)은 답에 따라 코드, 벡터, SPEC.md, `export/`가 함께 바뀐다. 해시가 바뀌지 않는 변경도 있으니 벡터 폴더를
  다시 받는다.
- 앱 팀에 확인을 요청한 것(카메라와 거치 위치, delegate, `onnxruntime-android` 버전, 기기의 타임스탬프 로그)은 PLAN 1.3절에 있다.

## 14. 문서

| 문서 | 내용 |
|------|------|
| `docs/GUIDE.md` | 안내서. 가장 먼저 읽는 입구 문서(모델 구조, 저장소·코드·문서를 읽는 순서, Android 앱에 잇는 법) |
| `docs/SPEC.md` | 이식 명세(앱이 지킬 전제와 규약, 모듈별 의사 코드, 해시 검사, 벡터, 프레임률 모사, 결정을 기다리는 동작) |
| `docs/PLAN.md` | 계획서(2026-11-13 승인): 조사로 확인한 사실, 결정(Q1~Q8, D1~D16), 신호와 특징의 정의, 모델, 평가 절차, 규칙, 단계와 통과 기준, 위험. 14장은 구현 중 달라진 점 |
| `docs/MODEL_CARD.md` | 모델 카드. 입출력, 구조와 v0의 실측 크기, 학습 절차(설계), 쓰임새와 한계, D11. 지표는 모두 `미측정` |
| `export/README.md` | `export/`의 파일과 해시의 짝, 내보내기의 보호 장치 |
| `tests/vectors/README.md` | 테스트 벡터의 형식, 허용 오차, 벡터별 설명, Android 팀이 쓰는 법 |
| `config/rules.yaml`, `config/default.yaml` | 값마다 출처와 잠정 여부를 주석으로 적었다 |
