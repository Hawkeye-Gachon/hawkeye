# export/ — 앱에 넘기는 파일

이 폴더의 파일은 Android 앱이 그대로 번들하는 산출물이다. 손으로 고치지 않는다(해시가 달라진다).
JSON 파일은 줄바꿈이 바뀌면 해시가 달라지므로 `.gitattributes`에서 `export/*.json -text`로 변환을 막아 두었다.
모델 파일은 `*.onnx binary`로 줄바꿈 변환, 텍스트 비교, 병합을 막았다(sha256이 벡터와 문서에 고정돼 있다).
이 파일들을 쓰는 코드를 옮기는 방법은 `docs/SPEC.md`(이식 명세)에 있다.

## 파일

| 파일 | 내용 | 만드는 명령 |
|------|------|-------------|
| `drowsiness_v0.onnx` | 입출력 계약 확인용 모델. **가중치가 무작위다(학습하지 않았다).** 인과 TCN, 시드 0, 148,558 bytes | `python -m drowsy.export_onnx --random-init --kind tcn --version 0` |
| `feature_spec_v0.json` | v0 모델과 함께 나간 feature_spec 사본. 스텝 특징 17개의 이름·순서·정의, 표준화 상수, 이식 코드가 쓰는 상수 전부(`pipeline` 블록) | 위 명령이 함께 쓴다 |
| `rules.json` | 규칙 엔진과 경고 결정의 임계값. `config/rules.yaml`과 같은 내용 | `python -m drowsy.spec export-rules --out export/rules.json` |

위 표의 첫 명령은 `--out-dir`이 없으면 이 폴더에 쓴다. 저장소의 v0을 바꾸지 않고 시험하려면 `--out-dir <임시 폴더>`를 준다.
2026-11-13, 2026-11-14(2차, 3차 검증 뒤의 정비가 끝난 코드 포함)와 2026-11-15(4차 검증 뒤의 정비가 끝난 코드)에 같은 명령을
임시 폴더로 돌린 파일은 이 폴더의 `drowsiness_v0.onnx`와 바이트 단위로 같았다.

학습한 모델을 내보내면(`python -m drowsy.train --config config/local.yaml --run-name <이름> final`) 버전마다
`drowsiness_v{N}.onnx`, `feature_spec_v{N}.json`이 따로 추가되고 `rules.json`도 그 실행의 규칙 설정으로 다시 쓰인다(내용이
같으면 바이트도 같다). TCN이면 `drowsiness_v{N}_int8.onnx`도 생긴다(같은 버전의 spec 사본을 같이 쓴다). 앞 버전의 파일과
해시는 그대로 남는다.

## 어느 해시가 어느 파일과 짝인가

| 해시 | 정의 | 적혀 있는 곳 | 지금 값 |
|------|------|--------------|---------|
| `feature_spec_hash` | `feature_spec_v0.json` **파일 바이트**의 sha256 | `drowsiness_v0.onnx`의 메타데이터 | `f4031d09592a99ba8b2c9e72883507c27e5004391a02be2437ed1a99464a4911` |
| `pipeline_hash` | `feature_spec_v0.json` 안 `pipeline` 블록의 해시(Python이 계산해 파일에 적는다. 앱은 문자열만 비교한다) | 같은 파일의 `pipeline_hash`, 모델 메타데이터, `tests/vectors/*.json` | `650495e5c8a158db9a071457713f852af846e54e32d80352376592768f9c31a6` |
| `rules_hash` | `rules.json` **파일 바이트**의 sha256 | `tests/vectors/*.json`, 오경보 리포트, 학습한 모델의 `final.json` | `9c2d05f29c4c4339d4759425f996c38137677d7216ad73c27091e75fb6cec7b4` |
| 모델 파일 sha256 | `drowsiness_v0.onnx` 바이트의 sha256 | `tests/vectors/model_v0_inputs.json`, `e2e_250s.json`의 `model.sha256` | `eb2f71a6e6defea6ea3c27da7c02ca244caba769b1ee241674158db045723842` |
| `mediapipe_model_sha256` | 학습 특징 추출에 쓴 `face_landmarker.task`의 sha256. 앱이 번들한 파일과 같아야 한다 | 모델 메타데이터, `feature_spec_v0.json`의 `pipeline.mediapipe.model_sha256` | `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` |

앱이 시작할 때 할 검사: 번들한 `feature_spec_v{N}.json`을 `MessageDigest`로 해시해 모델 메타데이터의
`feature_spec_hash`와 비교하고, 파일 안의 `pipeline_hash`를 메타데이터의 `pipeline_hash`와 문자열로 비교한다.
JSON을 정규화할 필요가 없다. 값 확인은 `sha256sum export/*`와 `python -m drowsy.spec check --spec export/feature_spec_v0.json`으로 한다.

2026-11-13~14의 정비에서 이식 코드의 동작이 바뀌었지만(README 8장, 13.3절) 설정 값은 하나도 바뀌지 않았으므로 위 해시는
모두 그대로다. 그래도 테스트 벡터는 **해시 변경 없이** 바뀌었다: 처음 43개 가운데 6개의 기댓값이나 설명이 달라졌고 새 벡터
25개가 더해져 68개가 됐다. 2026-11-14의 2차 검증 뒤 정비는 동작을 바꾸지 않고 벡터 13개를 더했다(앞의 68개는 바이트 단위로
그대로, 81개). 3차 검증 뒤의 정비도 동작을 바꾸지 않고 벡터 8개를 더했다(앞의 81개는 바이트 단위로 그대로, 89개,
27,983,687 bytes). 4차 검증 뒤의 정비도 동작을 바꾸지 않고 벡터 13개를 더했다(앞의 89개는 바이트 단위로 그대로, 지금 102개,
30,588,827 bytes. `docs/SPEC.md` 16.1절). 앱은 벡터 폴더를 통째로 다시 받는다. 4차 검증 뒤에는 설정 검증 조건 둘도 더해졌다
(`pipeline.yawn.cancel_gap_ms`와 `pipeline.nod.cancel_gap_ms`는 `clock.reset_gap_ms`보다 작아야 한다). 이 폴더의
`feature_spec_v0.json`의 값(500, 10,000)은 통과하므로 해시는 그대로다.
2026-11-14에 소유자가 반쯤 뜬 눈꺼풀의 동작(눈꺼풀이 이탈 임계값 위에 머무는 동안 여러 깜빡임이 한 감김으로 이어져 길이가
쌓이는 것)을 "지금대로 둠"으로 정했다(README 8장의 결정 5). 이 폴더의 파일, `rules.json`, 해시, 벡터는 하나도 바뀌지 않았고,
그 동작이 든 감김 벡터 여섯은 이제 결정된 동작을 고정한다.

## 내보내기의 보호 장치 (`drowsy.export_onnx`, `drowsy.train final`)

- **계약 검사**(`verify_contract`): opset 17, IR 8, 입력 `features` float32 `[1, 120, 17]`, 출력 `probs` float32 `[1, 3]`,
  메타데이터 키, 허용 연산자, 그래프 안 처리의 배선(특징 → 유지 마스크 곱 / ReLU → +1 → Log → log1p 마스크 곱 → 더하기 →
  평균 빼기 → 표준편차 나누기 → ±5 자르기, 클래스 축의 Softmax 하나가 `probs`)과 그 상수가 spec 사본과 같은지 본다. 파일
  이름(`drowsiness_v{model_version}.onnx`, `_int8.onnx`)은 `verify_file_name`이 따로 본다.
- **온도 규칙**(2차 검증 뒤의 정비에서 엄격해졌다): 신경망이 낸 점수(logits)를 바꿀 수 있는 것은 Softmax 바로 앞의
  `temperature`라는 이름의 상수로 나누는 것뿐이다. 기본 경로와 INT8 파일은 온도가 1.0이어도 그 상수가 있어야 하고, 대체
  경로(`--dynamo`)로 만든 파일만 온도가 1.0일 때 그 상수가 없을 수 있다. 어느 경로든 logits가 Div나 Mul 노드의 출력이면
  거부한다. 파일 종류는 검사에 준 연산자 집합으로 안다. 계약의 값은 하나도 바뀌지 않았고 이 폴더의 v0은 그대로 통과한다.
- **logits를 내는 노드**(3차 검증 뒤의 정비): logits는 신경망의 마지막 선형 층 그대로여야 한다. 기본·대체 경로는 그래프의
  유일한 Gemm(`alpha = 1`, `beta = 1`, `transA = 0`, 가중치와 편향은 그래프의 float32 상수), INT8 파일은 양자화기가 그 Gemm을
  바꿔 쓴 모양(입력 양자화 → 유일한 MatMulInteger → float32 Cast → 배율 Mul → 상수 편향 Add)의 끝인 편향 Add다. logits에
  상수를 곱한 뒤 값을 바꾸지 않는 노드를 끼우거나 배율을 Gemm 속성에 넣은 파일을 거부한다.
- **속성 읽기**(4차 검증 뒤의 정비): 그 Gemm의 `alpha`, `beta`, `transA`와 INT8 Cast의 `to`는 한 번만, 정해진 형(FLOAT 또는
  INT)으로, 값을 가진 채 있어야 한다. 아니면 한 줄의 계약 오류(`ContractError`)다. 전에는 형이 틀린 속성에서 Python의
  `TypeError`가 났고 값이 없는 INT 속성은 받아들였다. 이 폴더의 v0은 그대로 통과한다.
- **계약 검사가 보지 못하는 것**: 마지막 층의 가중치와 편향을 통째로 줄이거나 키운 파일(그래프 모양이 같다), 그리고 신경망
  안쪽(Clip과 마지막 층 사이). 가중치는 내보낸 직후의 PyTorch 대 onnxruntime 일치 검증이 지킨다. 받은 파일을 의심할 때는
  체크포인트와 출력을 견준다.
- **원자적 쓰기**: 모델과 spec 사본을 같은 폴더의 임시 이름으로 쓰고 계약 검사를 통과하면 바꿔 넣는다. 실패하면 앞의 모델,
  spec 사본, INT8 파일이 바이트 단위로 그대로 남는다. 성공하면 같은 버전의 옛 `drowsiness_v{N}_int8.onnx`를 지운다(INT8은
  그 뒤에 다시 만든다). 강제 종료되면 `.<파일 이름>.<pid>.tmp`가 남을 수 있으니 손으로 지운다.
- **커밋 해시**: git 상태를 아무 파일도 쓰기 전에 읽는다. 깨끗한 커밋에서 새 버전을 처음 내보내면 `-dirty`가 붙지 않는다.
- **합성 데이터와 축소 설정**: 합성 데이터로 학습한 체크포인트(`synthetic` 표시 또는 `trained_on`이 `SYNTHETIC`으로 시작)와
  축소 설정(`--fast`, 연기 시험)으로 학습한 체크포인트(`fast` 표시)는 이 폴더와 그 아래 폴더로 내보내지 못한다. 설정의
  `paths.export_dir`(기본값이 이 폴더)도, 설정이 다른 폴더를 가리킬 때의 이 폴더도(3차 검증 뒤의 정비. 연기 시험용 설정으로
  `--out-dir export`를 줘도 막힌다) 지킨다. `train final`과 `export_onnx --checkpoint`가 같은 함수
  (`export_onnx.protected_export_dir`, `is_export_dir_or_below`)로 검사한다. 경로는
  풀어서 비교하므로 `..`, 상대 경로, 심볼릭 링크로도 피할 수 없다. `paths.export_dir`이 상대 경로(기본 `export`)이면 지금
  폴더 기준과 저장소 루트(`drowsy` 패키지의 부모 폴더) 기준 둘 다를 이 폴더로 보므로, 저장소 밖의 폴더에서 명령을 돌려도
  이 폴더에 쓰지 못한다. 패키지를 다른 곳에 복사해 쓰면 그 복사본의 부모가 기준이 되므로, 그때 원래 저장소의 이 폴더는
  설정에 절대 경로를 적어야 보호된다. 거부되면 아무 파일도 쓰지 않고 종료 코드 2로 끝난다.
- **체크포인트 규칙**: `train final`이 쓰는 `reports/<이름>/final/final_model.pt`는 형식 `drowsy-checkpoint-v2`이고, 학습한
  spec 사본(`feature_spec_v{N}.json`)의 sha256, 합성 데이터 여부(`synthetic`), 축소 설정 여부(`fast`)를 담는다. `fast` 키가
  없는 v2 파일(그 키를 더하기 전에 만든 것)은 축소 설정이 아닌 것으로 읽는다. `export_onnx --checkpoint`에는 `--spec`(그
  학습이 모델 옆에 쓴 spec 사본)과 `--version`이 꼭 필요하고, 내보낼 spec 사본의 해시가 체크포인트의 값과 다르면 거부한다.
  형식 v1 체크포인트는 거부하고 `python -m drowsy.train final`을 다시 돌리라고 알린다. 예:
  `python -m drowsy.export_onnx --checkpoint reports/<이름>/final/final_model.pt --spec <그 실행의 내보낸 폴더>/feature_spec_v1.json --version 1 --out-dir <폴더>`.
  2026-11-14에 합성 데이터 전 과정의 체크포인트로 이 명령을 임시 폴더에 돌린 모델은 `train final`이 내보낸 모델과 바이트
  단위로 같았고, `--out-dir`을 빼면(이 폴더로) 거부됐다. 3차 검증 뒤의 정비가 끝난 코드로 다시 돌린 전 과정의 체크포인트에서도
  같았다(거부는 종료 코드 2, 이 폴더의 해시는 그대로). 2026-11-15에 4차 검증 뒤의 정비가 끝난 코드로 다시 돌린 전 과정의
  체크포인트를 임시 폴더로 내보낸 모델도 `train final`의 모델과 바이트 단위로 같았다(통합 단계의 기록).
- 명령의 종료 코드: 0 성공, 1 PyTorch 대 onnxruntime 일치 검증 실패(새 파일은 이미 바뀌어 들어간 뒤다), 2 오류(한 줄로 알린다).

## v0 모델에 대해 알아 둘 것

- 가중치가 무작위이므로 출력 확률에는 뜻이 없다. 앱 팀이 입출력 계약으로 연동을 시작하기 위한 파일이다.
  메타데이터 `trained_on`이 `none (random weights)`다.
- 계약: opset 17, IR 8, 단일 파일. 입력 `features` float32 `[1, 120, 17]`(표준화하지 않은 원시 스텝 특징,
  `features[0][0]`이 가장 오래된 스텝), 출력 `probs` float32 `[1, 3]`(alert, low_vigilant, drowsy).
  표준화, ±5 자르기, 온도, 소프트맥스는 그래프 안에 있다(`standardization=in_graph`). 앱에서 다시 표준화하지 않는다.
- v0의 표준화 상수는 평균 0, 표준편차 1, 온도 1이다(학습 전의 자리 채움 값).
- **메타데이터 `commit`은 `uncommitted-dirty`다(자리 채움 값).** 내보낼 때 저장소에 커밋이 없었다. 첫 커밋 뒤 v0을 다시
  내보낼지, 그대로 두고 v1부터 커밋 해시를 넣을지는 소유자의 결정을 기다린다(README 9장 11번, `docs/SPEC.md` 20장 11번). 다시 내보내면 `commit` 문자열이
  길어져 파일 크기와 sha256이 바뀐다(검토의 임시 저장소에서 148,558 → 148,581 bytes). 그러면
  `python -m drowsy.vectors generate --out tests/vectors`로 벡터(`model_v0_inputs`, `e2e_250s`의 `model.sha256`)를 다시 만들고
  이 문서, `docs/MODEL_CARD.md`, README의 크기와 sha256을 고친다.
- 졸음 판정의 기본 판정원은 모델이 아니라 PERCLOS 규칙이다(`rules.json`의 `drowsiness.source: perclos`).
  근거와 조건은 `docs/MODEL_CARD.md`의 D11 항목에 있다.
