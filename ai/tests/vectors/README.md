# 테스트 벡터 (`tests/vectors/*.json`)

입력은 모두 합성 값이다(실제 얼굴 이미지는 없다. PLAN 1.2절의 D14). 여기 있는 수치는 이식 코드의 계산이 참조 구현과 같은지
확인하는 용도이고 성능 수치가 아니다.

벡터 하나는 JSON 파일 하나다. 앱이 MediaPipe에서 받는 것과 같은 모양의 프레임과, Python 참조 구현
(`drowsy/`의 이식 대상 모듈)이 그 프레임에서 낸 출력을 함께 담는다. 쓰임은 둘이다.

- Android 팀: 같은 입력을 Kotlin 이식 코드에 넣고 출력이 `expected`와 허용 오차 안에서 같은지 본다. 옮길 규칙은
  `docs/SPEC.md`(이식 명세)에 있다.
- 이 저장소: `python -m pytest tests/test_vectors.py`가 벡터를 다시 만들어 파일과 바이트 단위로 같은지, `expected`를
  다시 계산해 허용 오차 안에서 같은지 본다. 참조 구현이나 설정이 바뀌면 여기서 드러난다.

벡터는 102개다(신호 9, 기준선과 closure 9, 감김 검출 15, 스텝 특징 15, 규칙 52, 전 과정 1, 모델 1). 2026-11-14에 다시
만들었고 같은 날 13개를, 이어서 8개를, 다시 13개를 더했다: 달라진 파일과 새 파일은 맨 아래 "2026-11-14에 달라진 것"에
있다. `pipeline_hash`와 `rules_hash`는 그대로다.

## 명령

저장소 루트에서 실행한다.

```
python -m drowsy.vectors generate --out tests/vectors     # 다시 만들기 (몇 초, 102개 합계 30,588,827 bytes)
python -m drowsy.vectors verify --dir tests/vectors       # expected를 다시 계산해 비교
```

- 선택 인자: `--spec config/feature_spec.json`, `--rules config/rules.yaml`, 생성은 `--model export/drowsiness_v0.onnx`,
  검증은 `--model-dir export`.
- 파일을 손으로 고치지 않는다. `config/feature_spec.json`의 `pipeline` 블록, `config/rules.yaml`, 이식 대상 모듈,
  `export/drowsiness_v0.onnx` 가운데 하나라도 바뀌면 다시 만든다. 모델을 다시 내보내면(가중치가 같아도 메타데이터가
  달라지면 파일 해시가 달라진다) `model_v0_inputs`와 `e2e_250s`가 달라진다.
- 같은 환경에서 다시 만들면 파일이 바이트 단위로 같다. 실수는 정확히 왕복하는 가장 짧은 십진 표기로 적는다.

## 파일 형식

PLAN 10.2절의 형식에 `bundle`, `description`, `tolerances`, `model`, `model_inputs`와 `expected`의 `baseline`,
`model_inputs`, `step_probs`, `counters`를 더했다. `step_probs`는 2026-11-14에 더한 항목이다(스텝마다 모델을 부르게
되어 스텝별 확률을 따로 적는다. `probs`의 뜻과 모양은 그대로다).

| 키 | 내용 |
|----|------|
| `name` | 파일 이름(확장자 제외) |
| `bundle` | `signals`, `baseline`, `runs`, `steps`, `rules`, `model`, `e2e` 중 하나 |
| `description` | 이 벡터가 고정하는 것 |
| `pipeline_hash` | 만든 시점의 `config/feature_spec.json`의 `pipeline_hash`. 앱이 번들한 spec 파일(`export/feature_spec_v{N}.json`)의 값과 다르면 벡터와 코드의 상수가 다른 것이다 |
| `rules_hash` | 만든 시점의 규칙 설정 해시(`export/rules.json` 바이트의 sha256과 같은 값). `config_overrides`를 적용하기 전의 값이다 |
| `tolerances` | `expected`의 항목별 허용 오차(아래 표) |
| `baseline` | 넣어 준 기준선 또는 `null`. 필드는 `Baseline`과 같다: `e_l, e_r, b_l, b_r, r0`(3x3 행 우선 9개), `jaw0, gaze_h0, gaze_v0, t_complete_ms, status`. 주면 보정을 건너뛰고 세션 시간 `t_complete_ms`(벡터에서는 0)부터 특징을 누적한다. `null`이면 세션 앞부분에서 보정한다 |
| `config_overrides` | 규칙 설정에 덮어쓸 값. 점으로 이은 경로 → 값. 예: `{"drowsiness.source": "model"}`. `pipeline` 상수는 덮어쓰지 않는다 |
| `model` | `null`(모델 없음), `{"kind": "stub", "probs": [[p_alert, p_low, p_drowsy], ...]}`, `{"kind": "onnx", "file": "drowsiness_v0.onnx", "sha256": "..."}` 중 하나 |
| `frames` | 프레임 목록. `{t_ms, face, w, h, lm, bs, m}`. `lm`은 랜드마크 인덱스(문자열 키) → 정규화 좌표 `[x, y]` 44개, `bs`는 블렌드셰이프 이름 → 값 11개, `m`은 변환 행렬의 열 우선 16개 값이다. 얼굴이 없으면 `lm`과 `bs`는 `{}`, `m`은 `[]`다. `t_ms`는 양수다 |
| `speed` | `[[t_ms, kmh], ...]`. `t_ms` 이상인 프레임부터 그 속도를 `update(frame, speed_kmh)`에 넣는다. 첫 항목보다 이른 프레임과 목록이 빈 벡터는 속도 입력이 없다(null) |
| `model_inputs` | 모델 묶음에만 있다. 원시 스텝 특징 `[120][17]`의 목록 |
| `expected` | 참조 구현의 출력. 벡터마다 해당하는 항목만 있다(아래) |

`expected`의 항목. 목록의 원소에 있는 `t_ms`는 그 출력을 낸 프레임의 `t_ms`다.

| 항목 | 내용 |
|------|------|
| `signals` | 프레임마다 `FrameSignals`의 필드. `eye_look`은 `in_l, out_l, up_l, down_l, in_r, out_r, up_r, down_r` 순서, 쓸 수 없는 눈의 홍채 비율은 `null` |
| `closure` | 받아들인 프레임마다 `FrameState`의 필드(`closure, closure_valid, provisional, yaw_rel, pitch_rel, pose_valid, gaze_h_rel, gaze_v_rel, jaw_rel`). 버린 프레임은 없다 |
| `baseline` | 세션에서 확정된 기준선(`Baseline`의 필드)과 그것을 처음 쓴 프레임의 `t_ms` |
| `runs` | 특징용 검출기(진입 0.5 / 이탈 0.3)가 발행한 `ClosureRun`의 필드 |
| `steps` | `StepFeatures`의 필드: `k, t_end_ms, full_lookback, values`(17개, spec 파일의 `features` 순서). 프레임 하나가 스텝 여러 개를 발행하면 `t_ms`가 같다 |
| `model_inputs` | 모델을 처음 부른 창 `{t_ms, k, features}`. `features`는 `[120][17]`이고 **`features[0]`이 가장 오래된 스텝(k − 119), `features[119]`가 가장 최근 스텝(k)**이다. ONNX 입력 `features[0][t][f]`에 그대로 넣는다 |
| `step_probs` | 발행된 스텝마다 `{t_ms, k, probs}` 하나씩(`PipelineOutput.step_probs`를 스텝 순서대로 편 것). `t_ms`는 그 스텝을 발행시킨 프레임, `probs`는 그 스텝에서 모델을 불렀으면 `[alert, low_vigilant, drowsy]`, 부르지 않았으면 `null`이다. 프레임 하나가 스텝을 여럿 발행했으면 같은 `t_ms`의 원소가 여럿이고 저마다 확률을 가질 수 있다 |
| `probs` | 마지막 스텝에 확률이 있는 프레임마다 `{t_ms, k, probs}`(`PipelineOutput.probs`: 그 프레임이 발행시킨 마지막 스텝의 확률). `k`는 그 스텝의 번호다. 프레임마다 스텝이 하나뿐인 벡터에서는 `step_probs`에서 `null`이 아닌 원소를 모은 것과 같다. 모델 묶음에서는 `{input, probs}` |
| `events` | 상태 이벤트 `DriverStateEvent`의 필드(`type, severity, t_start_ms, duration_ms, confidence, source, phase, provisional`), 발행 순서 |
| `alerts` | `AlertPolicy`가 실제로 경고하기로 한 이벤트, 같은 모양 |
| `counters` | `DriverPipeline`이 센 값: `frames_accepted, dropped_frames, steps_emitted, model_calls, nonfinite_windows` |

## 허용 오차 (PLAN 10.2절)

실수는 `|실제 − 기대| ≤ abs + rel·|기대|`, 시각은 `±time_abs_ms`로 비교한다. 파일의 `tolerances`에 같은 값이 들어 있다.

| 항목 | 허용 오차 |
|------|-----------|
| `signals` | 절대 1e-6, 각도(`yaw, pitch, roll`)는 절대 1e-3도 |
| `closure`, `baseline` | 절대 1e-6 |
| `runs` | 시각(`start_ms, end_ms, duration_ms`) ±1 ms, 값(`amplitude`, 속도) 상대 1e-4 |
| `steps`, `model_inputs` | 절대 1e-5 + 상대 1e-5 |
| `step_probs`, `probs` | 절대 1e-4 (`null`은 `null`이어야 한다) |
| `events`, `alerts` | 종류·단계·순서·source·severity·provisional 일치, 시각(`t_start_ms, duration_ms`) ±1 ms, `confidence` 절대 1e-4 |

표에 없는 값(불리언, 문자열, 스텝 번호 `k`, `t_end_ms`, 출력을 낸 프레임의 `t_ms`, `counters`, 목록의 길이)은 정확히
같아야 한다. `confidence`의 허용 오차는 계획서에 없어서 모델 확률과 같게 정했다.

## 묶음과 벡터

### signals (신호)

프레임마다 `signals.compute`의 출력을 비교한다. 상태가 없으므로 프레임 순서는 뜻이 없다.

| 벡터 | 고정하는 것 |
|------|-------------|
| `signals_ear_square` | 정사각 프레임의 EAR, 눈 폭, 홍채 비율, 블렌드셰이프 읽기. 좌우 값이 모두 다르다 |
| `signals_ear_non_square` | 비정사각 프레임 셋(가로, 세로, 작은 해상도). EAR은 픽셀 좌표 `(x·w, y·h)`에서 구해야 같은 값이 나온다 |
| `signals_official_matrix` | MediaPipe 공식 기대 출력의 행렬: yaw 2.24, pitch 6.32, roll 0.99도. 열 우선으로 읽어야 한다 |
| `signals_mirrored_matrix` | 같은 프레임과 좌우 반전한 프레임: yaw·roll의 부호가 뒤집히고 좌우 신호가 바뀐다 |
| `signals_random_matrices` | 무작위 자세 여덟 개(yaw ±170, pitch ±80, roll ±170도)와 무작위 평행이동 |
| `signals_one_eye_closed` | 한쪽 눈만 감은 프레임과 그 반전: Left/Right 짝(L 집합과 `eyeBlinkLeft`) |
| `signals_zero_width_eye` | 눈 폭 0 px, 5 px(쓸 수 없는 눈: ear 0.0, 홍채 `null`), 7 px(쓸 수 있음) |
| `signals_no_face` | 얼굴 없는 프레임: 모든 값 0.0 |
| `signals_iris_tilted_eyes` | 눈 축이 영상에서 기울어진 프레임 둘(두 눈을 36.87도와 53.13도로 돌렸다. 머리가 기울면 실제로도 이렇게 된다). 홍채 비율, EAR, 눈 폭은 축의 방향과 무관하다: `iris_h_l` 0.46, `iris_v_l` 0.02, `iris_h_r` 0.55, `iris_v_r` -0.03. 다른 신호 벡터는 두 눈 꼬리의 높이가 같아서 홍채 비율의 식에서 세로 성분이 든 항의 부호가 틀려도 드러나지 않는다(틀리면 여기서 `iris_h_l`이 1.64나 0.11, `iris_v_l`이 0.45가 된다) |

### baseline (기준선과 closure)

`expected.closure`(프레임 상태)와 `expected.baseline`을 비교한다. `DriverPipeline` 없이도 확인할 수 있다:
프레임마다 `signals.compute` → `BaselineEstimator.update` → 완성 전에는 전역 상수 기준선과 `provisional = true`,
완성된 프레임부터는 그 기준선과 `provisional = false`로 `frame_state`.

| 벡터 | 고정하는 것 |
|------|-------------|
| `baseline_normal` | 60초에 완성(status `ok`). 신호가 프레임마다 흩어져 있어서 분위수(최근접 순위)와 중앙값(짝수 표본은 가운데 두 값의 평균)의 정의가 틀리면 드러난다 |
| `baseline_extended` | 얼굴 프레임이 모자라 90초로 연장. 기준 yaw가 15도를 넘어 status `off_axis` |
| `baseline_fallback` | 180초까지 완성하지 못해 전역 상수로 대체(status `fallback_global`, `pose_valid`는 계속 false) |
| `closure_yaw_weighting` | 넣어 준 기준선(기준 자세 10, 5, 2도)에서 yaw에 따른 먼 쪽 눈 가중, 40도 초과 시 무효, 한쪽 눈만 쓸 수 있는 경우 |
| `closure_no_face` | 얼굴 없는 프레임의 상태(모두 0.0과 false) |
| `baseline_odd_sample_count` | 손으로 놓은 보정 세션(신호는 몇 가지 값뿐이다). 눈 조건으로 두 번 연장한다: 60초 마감에서는 Right 눈의 폭이 3 px이라 E의 표본이 없고, 90초 마감에서는 Right 눈의 E가 0.50으로 `e_max`(0.45)를 넘는다. 120초에 표본 331개(홀수)로 완성한다. 중앙값은 가운데 표본 하나다: 기준 pitch 4도, `jaw0` 0.09. 가운데 두 표본의 평균을 내는 구현은 pitch 2도, `jaw0` 0.07을 낸다. 눈 하나의 표본이 없는데도 완성하는 구현은 `T_b`가 60000이, E의 범위를 한쪽 눈에서만 보는 구현은 90000이 된다 |
| `baseline_burst_frames` | 프레임이 몰려서 오는 세션의 얼굴 검출 비율. 얼굴 프레임 하나는 다음 프레임까지, 최대 200 ms를 대표하고 마감을 넘는 부분은 세지 않는다. 60초 마감에서 얼굴 프레임은 301개지만 얼굴 시간이 29900 ms(비율 0.4983)라 연장하고, 90초 마감에서 정확히 45000 ms(비율 0.5, '이상')라 완성한다(`T_b` 90000). 200 ms 상한이 없거나, 마감 뒤까지 대표한 시간을 더하거나, 얼굴 없는 프레임이 대표한 시간을 더하면 60초에 완성되고, 비율을 '초과'로 비교하면 세션이 끝날 때까지 완성되지 않는다 |
| `baseline_exactly_300_frames` | 손으로 놓은 보정 세션(모든 프레임의 신호가 같다). 얼굴 프레임이 정확히 300개다: 0~59600 ms의 299개는 Left 눈의 폭이 3 px이라 쓸 수 없고, 300번째(60000 ms)에서만 Left 눈을 쓸 수 있으며, 그 뒤 90000 ms까지 프레임이 없다. 60초 마감에서는 프레임이 299개이고 Left 눈의 표본이 없어 연장한다. 90초 마감에서 프레임이 정확히 300개('이상')이고 얼굴 시간이 60000 ms(비율 0.6667)라 완성한다(`T_b` 90000, E_l은 표본 하나인 0.30). 프레임 수를 '초과'로 세거나 쓸 수 없는 Left 눈의 EAR(0.0)을 표본에 넣으면(E_l이 0.0이 된다) 세션이 끝날 때까지 완성되지 않고, Left 눈의 표본이 없는데 분위수를 구하려 들면 60초 마감에서 빈 목록의 오류가 난다 |
| `closure_blink_span_floor` | 넣어 준 기준선의 Left 눈 B가 0.50이라 `b_closed − B`(0.10)가 `b_span_min`(0.20)에 못 미친다: 분모는 하한 0.20이다. closure는 0.125, 0.325, 0.45. 하한을 빼면 첫 프레임이 0.25가 된다 |

### runs (감김 검출)

`expected.closure`와 `expected.runs`를 비교한다. 검출기만 확인하려면 `expected.closure`의
`(t_ms, closure, closure_valid)`를 `ClosureRunDetector`(진입 0.5 / 이탈 0.3)에 차례로 넣으면 `expected.runs`가 나온다.
기준선을 넣어 줬으므로 첫 프레임부터 검출기가 돈다.

| 벡터 | 고정하는 것 |
|------|-------------|
| `runs_blinks_30fps` | 30 fps의 깜빡임 넷(하나는 0.5에 못 미쳐 구간 없음): 보간한 시작·끝, 진폭, 속도 |
| `runs_blinks_15fps` | 같은 움직임을 15 fps로: 짧은 깜빡임은 보간한 길이가 50 ms 미만이라 버려진다 |
| `runs_dropped_frames` | 같은 움직임에서 프레임을 확률 0.35로 버리고(66개 중 29개가 빠졌다) 시각을 흔든 경우 |
| `runs_one_sample_blink` | 한 표본짜리 깜빡임: 구간이 되는 것, 너무 짧아 버려지는 것, 보이지 않는 것 |
| `runs_closure_600ms` | 600 ms 감김: `duration_ms`는 0.5를 지난 두 시각의 차 |
| `runs_truncated_by_gap` | 얼굴 소실 300 ms(같은 구간으로 이어짐), 프레임 끊김 700 ms와 얼굴 소실 700 ms(`truncated`) |
| `runs_duplicate_timestamp` | 같은 `t_ms`와 더 이른 `t_ms`의 프레임은 버리고 센다(`counters.dropped_frames = 2`) |
| `runs_lingering_lids` | 눈꺼풀이 0.3~0.5에 머무는 경우의 구간 경계. 감았다가 0.4에 머문 뒤 다시 감으면 한 구간이고 `duration_ms`는 0.5 이상이던 두 토막의 합이다 |
| `runs_velocity_after_gap` | 감는 속도의 거슬러 보기는 쓸 수 없는 쌍(간격 200 ms 초과)에서 끊긴다: 끊김이나 긴 무효 구간 뒤의 첫 표본에서 시작한 구간에 그 앞의 증가율이 넘어오지 않는다. 무효 프레임 하나를 건넌 67 ms 쌍은 이어진다 |
| `runs_rounding_large_t0` | 보간 시각 = 직전 표본의 시각 + floor(쌍 안의 오프셋 + 0.5). 첫 `t_ms`가 1.7e12인 세션에서 오프셋이 16.49993 ms다: 절대 시각에 더한 뒤 반올림하면 구간이 49 ms가 되어 버려지고, 옳게 구현하면 50 ms로 남는다 |
| `runs_multi_span` | 토막이 셋인 구간(`duration_ms` 908, `end_ms − start_ms` 1788). 토막 안의 끊김은 길이에 들어가고, `HOLD`에서 돌아온 프레임에서 시작한 토막은 그 프레임의 시각에서 시작한다 |
| `runs_multi_span_truncated` | 토막이 둘인 구간이 잘리는 세 경우: 잘린 구간의 `end_ms`와 `duration_ms` |
| `runs_two_humps` | 정점이 뒤의 표본으로 옮겨지는 구간의 두 속도: 정점을 넘지 못했던 증가율도 정점이 옮겨지면 감는 속도의 후보이고(3.64/s), 뜨는 속도는 새 정점 뒤부터 다시 잰다(3.7/s). 같은 최댓값이 다시 나와도 정점은 처음 닿은 표본이다. 정점을 넘지 못한 증가율은 다음 구간으로 넘어가지 않는다 |
| `runs_one_sample_truncated` | 감긴 표본이 하나뿐인 채로 잘린 구간: `end_ms`는 그 표본의 시각, `duration_ms`는 보간한 시작부터 그 시각까지(83 ms). 47 ms짜리는 잘린 구간이어도 버린다 |
| `runs_two_rises_before_peak` | 정점이 옮겨지기 전에 증가가 두 번 있는 구간(3.64/s, 그다음 0.91/s): 감는 속도는 둘 가운데 큰 3.64/s다. 마지막 증가율만 들고 있는 구현은 3.4/s를 낸다(`runs_two_humps`에는 그런 증가가 한 번뿐이다). 토막이 셋이고 `duration_ms` 239 |

### steps (스텝 특징)

`expected.steps`와 `expected.runs`를 비교한다. 프레임별 신호는 저장하지 않는다. 기준선을 넣어 줘서 스텝 1부터
나온다. 아래 표의 뒤 여덟은 15초 이하의 짧은 세션이고 프레임의 시각과 값을 손으로 놓았다(값은 계단 모양).

| 벡터 | 고정하는 것 |
|------|-------------|
| `steps_five_minutes` | 5분 세션: 초기 구간(스텝 59까지 `full_lookback` false), 깜빡임 없는 구간, 감긴 채 12초의 얼굴 소실(잘린 감김, 유효 시간이 없는 10초 구간의 비율은 0), 7.2초 끊김(프레임 하나가 스텝 8개를 발행), 하품, 끄덕임, 긴 감김, 200 ms를 넘는 프레임 간격 |
| `steps_five_minutes_large_t0` | 같은 세션을 `t_ms = 1731465123457`에서 시작. 스텝 번호와 값은 위와 같고 시각만 밀린다. `t_ms`는 `Long`이어야 한다 |
| `steps_gap_over_10s` | 10초 넘는 끊김: 끊긴 동안의 스텝은 없고, 새 프레임의 스텝부터 누적과 `full_lookback`이 다시 시작한다. 고개를 숙인 채 가만히 있어서 pitch의 분산이 반올림 오차로 음수가 된다(0 이상으로 자르지 않으면 `pitch_std_60`이 NaN) |
| `steps_multi_span_runs` | 토막이 여럿인 감김의 스텝 특징(12초): 길이를 쓰는 특징(#5, #6, #9~#11)은 0.5 이상이던 시간의 합을 쓴다. 0.4에 머무는 동안 진행 중인 길이가 늘지 않는다 |
| `steps_closure_across_gap` | 눈을 감은 채 프레임이 7.8초 끊긴다(10초 이하): 끊긴 동안의 스텝 여덟 개가 한 프레임에서 나오고 모두 끊기기 직전의 진행 중 감김을 쓴다. 잘린 구간은 두 번 세지 않는다. `perclos_10`이 끊긴 동안 올라간다 |
| `steps_zero_division` | 0으로 나누게 되는 비율은 0: 얼굴이 없는 스텝, 얼굴은 있고 closure가 무효인 스텝, 유효 시간이 하한에 못 미치는 스텝, 이벤트가 하나뿐인 표준편차 |
| `steps_gap_exactly_10s` | 특징 쪽의 초기화 경계. 정확히 10000 ms의 끊김: 초기화하지 않는다(진행 중이던 감김이 잘린 구간으로 나오고 끊긴 동안의 스텝 6~15가 한 프레임에서 나온다). 10001 ms의 끊김: 초기화한다(구간도 끊긴 동안의 스텝도 없고 스텝 31부터 다시 쌓는다) |
| `steps_yawn_edges` | 하품 계수의 경계 다섯 장면: 시작 임계에 닿지 않은 벌림(세지 않는다), 800 ms의 얼굴 소실(취소. 취소하지 않으면 하나를 더 센다), 정확히 500 ms의 얼굴 소실(이어진다), 정확히 2000 ms가 된 프레임(센다), 끝 임계 아래로 내려간 프레임이 정확히 2000 ms(세지 않는다). `yawn_count_60`은 스텝 1~8이 0, 9~11이 1, 12~15가 2 |
| `steps_nod_edges` | 끄덕임 계수의 경계 네 장면: 길이 정확히 200 ms(센다), 정확히 3000 ms(센다), 숙인 채 얼굴이 0.8초 사라짐(취소. 얼굴 없는 프레임을 pitch 0의 표본으로 넣으면 하나를 더 센다), 정확히 500 ms의 얼굴 소실(이어져서 800 ms로 센다). `nod_count_60`은 스텝 1~4가 1, 5~7이 2, 8이 3 |
| `steps_nod_slow_mean` | 끄덕임의 느린 평균: 첫 표본이 초기값, `d = pitch_rel − m`, `α = dt / (10000 + dt)`, `dt`의 상한 200 ms. 자세가 12도에서 20도로 바뀐 뒤 7.2초에 25.25도까지 숙이면 `d`가 10.028이라 끄덕임 하나다(`nod_count_60`이 스텝 8에서 1). 갱신 속도가 1%만 달라도(`α = dt / 10000`이면 9.984) 시작 임계 10에 못 미쳐 0이 된다. 빼는 방향이 틀리거나 `m`을 갱신하지 않아도 0이 된다 |
| `steps_nod_after_long_gap` | 10초 넘는 끊김 뒤에는 느린 평균도 처음 상태로 돌아간다: 끊김 뒤 첫 표본(12도)이 새 초기값이라 끄덕임이 없다(`nod_count_60`이 모두 0). 끊김 앞의 평균(0도)을 남기면 스텝 13에서 하나를 센다 |
| `steps_two_closures_in_one_step` | 한 스텝 안에서 끝난 감김 둘 가운데 긴 것이 그 스텝의 값이다: `max_closure_ms_60`과 `max_closure_ms_10`이 스텝 1에서 388, 스텝 2에서 488. 나중에 끝난 감김으로 덮어쓰면 스텝 1이 88이 된다 |
| `steps_closure_above_cap` | 11578 ms의 감김: `max_closure_ms_60`과 `max_closure_ms_10`은 진행 중일 때도 끝난 뒤에도 상한 10000이다(상한이 없으면 10489, 11489, 11578). `blink_dur_mean_60`은 깜빡임 길이의 상한 2000 |
| `steps_closure_exactly_500ms` | 길이가 정확히 500 ms인 감김은 긴 감김이다('이상'). 끝난 감김(150~650 ms)과, 스텝의 마지막 프레임에서 진행 중인 길이가 정확히 500 ms인 감김을 둘 다 센다: `long_closure_count_60`이 1, 2, 2. 끝난 감김을 '초과'로 세면 0, 1, 1, 진행 중인 감김을 '초과'로 세면 1, 1, 2 |
| `steps_pitch_std_at_5000ms` | `pitch_std_60`은 얼굴 시간이 정확히 5000 ms인 스텝부터 계산한다('이상'): 스텝 1~4는 0, 스텝 5는 sqrt(6) = 2.449490, 스텝 6은 2.357023. '초과'로 구현하면 스텝 5가 0이 된다. 프레임이 끊기지 않는 세션은 모두 다섯째 스텝에서 이 자리를 지난다 |

### rules (규칙과 경고)

`expected.events`와 `expected.alerts`를 비교한다. `rules_calibrating_provisional`과 `rules_calibrating_large_t0`만 보정부터
하고 나머지는 기준선을 넣어 준다(status `ok`라 시선 규칙이 동작한다).

| 벡터 | 고정하는 것 |
|------|-------------|
| `rules_closure_900ms` | 0.9초 감김: 이벤트 없음 |
| `rules_closure_1100ms` | 1.1초 감김: `microsleep` onset과 end |
| `rules_closure_2500ms` | 2.5초 감김: `microsleep`만 |
| `rules_closure_3000ms` | 감김 시작에서 정확히 3000 ms 뒤의 감긴 프레임에서 `sleep` onset(PLAN 1.2절의 D2: "3.0초 이상"). `microsleep` onset, `sleep` onset, `sleep` end |
| `rules_closure_6000ms` | 6초 감김: 첫 `sleep` 경고에서 3000 ms 이상 지난 첫 프레임에 경고가 한 번 되풀이된다 |
| `rules_gaze_2900ms` | 2.9초 시선 이탈: 이벤트 없음 |
| `rules_gaze_3100ms` | 3.1초 시선 이탈: 정확히 3000 ms에 `gaze_away_long` onset, 250 ms 확정 뒤 end. source `head_pose` |
| `rules_gaze_eye_only` | 눈만 돌린 이탈: source `eye` (계획서의 경우에 더한 벡터) |
| `rules_gaze_face_lost_in_turn` | 고개를 돌린 채 얼굴이 사라짐: 3초까지는 영역 밖으로 쳐서 얼굴이 없는 동안 onset |
| `rules_vats` | 30초 중 누적 10초: `vats` onset과 end, 속도 30 km/h에서 경고 |
| `rules_drowsy_model_stub` | 모델 졸음 판정(`drowsiness.source = model`, 스텁 확률): onset, end, 종료 후 60초 재시작 금지, 재경고 대기 |
| `rules_drowsy_perclos` | PERCLOS 졸음 판정: onset과 end, 속도 40 km/h에서 경고 억제 |
| `rules_calibrating_provisional` | `calibrating` onset과 end, 보정 중의 `provisional` microsleep, 시선 규칙은 완성 뒤에 시작 |
| `rules_unable_to_analyze` | 분석 불가 onset과 end, 그동안 감김 판정이 멈춤 |
| `rules_speed_gates` | 20 km/h 기준 위아래의 `microsleep`, `sleep`, `gaze_away_long`, 기준 아래의 `vats`. 억제된 `sleep`은 속도가 오르면 경고 |
| `rules_cooldown` | `microsleep` 재경고 대기 5초 |
| `rules_sleep_repeat` | `sleep` 경고가 3000 ms마다 되풀이 |
| `rules_closure_dwell_below_1s` | 0.9로 300 ms, 0.6으로 5초, 0.9로 300 ms: 한 감김이고 길이(0.7 이상이던 시간의 합)는 778 ms라 이벤트 없음 |
| `rules_closure_dwell_reaches_1s` | 같은 앞부분에 둘째 감김이 길다: 누적 감김 시간이 정확히 1000 ms가 된 프레임에서 `microsleep` onset. `duration_ms`(5966)는 감김의 시작부터라 머문 5초를 포함한다 |
| `rules_closure_three_spans` | 토막이 셋인 감김: 누적이 정확히 1000 ms와 3000 ms가 된 프레임에서 `microsleep`과 `sleep` |
| `rules_alert_deferred_cooldown` | 재경고 대기에 걸린 `microsleep` onset: 대기가 풀리기 전에 끝나면 경고 없음, 이어지면 첫 경고에서 정확히 5000 ms 뒤의 프레임에 한 번 경고. 미뤘다 낸 경고도 대기를 새로 시작한다 |
| `rules_alert_blocked_microsleep_sleep` | 속도에 막힌 `microsleep`이 `sleep`으로 넘어가면 `microsleep` 경고는 나가지 않는다. 정확히 20.0 km/h에서 `sleep` 경고 |
| `rules_drowsy_perclos_exact` | `perclos_60`이 정확히 0.15(시작 조건)와 정확히 0.10(종료 조건 아님)인 스텝. 45 km/h에서 막힌 `drowsy` 경고가 정확히 50.0 km/h가 된 프레임에서 한 번 나간다 |
| `rules_drowsy_model_stall` | 프레임이 2.6초 끊긴 뒤의 프레임이 스텝 셋을 발행하고 모델을 세 번 부른다(`expected.step_probs`). 끊김이 모델 졸음 판정의 연속 수를 끊지 않는다. `p_drowsy`가 정확히 0.6인 스텝은 시작 조건이다 |
| `rules_gap_over_10s` | 10001 ms의 끊김: 진행 중이던 `gaze_away_long`, `sleep`, `vats`가 그 프레임에서 이 순서로 끝나고 규칙 엔진이 다시 시작한다(정확히 1000 ms 뒤 `microsleep`, 3000 ms 뒤 `gaze_away_long`과 `sleep`, 분석 불가 없음). 한 프레임의 경고 둘은 우선순위 순. 속도에 막혔던 `gaze_away_long` 경고가 정확히 20.0 km/h에서 한 번 나간다 |
| `rules_gap_exactly_10s` | 같은 앞부분에서 끊김이 정확히 10000 ms: 초기화하지 않는다. `unable_to_analyze`가 시작되고 감김은 잘려 끝난다. 37초 이어진 `vats`의 end `confidence`는 구간 전체의 `pose_valid` 시간 비율(25200 / 37000) |
| `rules_gap_over_10s_drowsy` | 10초 넘는 끊김이 `drowsy`를 끝낸다(발행 시각은 그 프레임, `confidence` 0.0). 속도에 막혀 미뤄 둔 `drowsy` 경고는 나가지 않고, 끊김 뒤의 스텝에서 `drowsy`가 다시 시작되지 않는다 |
| `rules_gap_unable_boundary` | 끊김 5200 ms는 `unable_to_analyze`를 시작하지 않고 5201 ms는 시작한다. 분석 불가 중인 프레임에서는 `gaze_away_long`이 나지 않는다 |
| `rules_unable_sparse_frames` | 500 ms 간격의 프레임(프레임마다 200 ms만 인정): 정확히 10000 ms의 프레임에서 `unable_to_analyze`. 10초 넘는 끊김 뒤 끝나고 "처음 10초"가 다시 시작한다. 재경고 대기에 걸린 둘째 onset은 대기가 풀린 프레임에서 경고 |
| `rules_gaze_credit_window_missed` | 얼굴 소실 뒤의 인정 구간(3초)에 프레임이 오지 않은 경우: 구간이 끝난 뒤의 첫 프레임(얼굴 없음)에서 `gaze_away_long` onset |
| `rules_gaze_credit_boundary` | 인정 구간이 끝나는 바로 그 시각의 프레임은 표지를 이어받지 않는다. 영역 안 후보가 걸린 채로는 onset이 나지 않는다. 끝나기 50 ms 전의 프레임은 이어받은 표지로 영역 안을 확정한다 |
| `rules_gaze_debounce_250ms` | 200 ms의 정면은 확정되지 않고, 정확히 250 ms 뒤의 프레임에서 확정된다 |
| `rules_forward_zone_edges` | 영역의 경계 여섯(yaw ±23, pitch −25와 +18, gaze_h 0.25, gaze_v 0.35)의 바로 바깥 값과 안쪽 값. `vats`의 source `eye` |
| `rules_vats_reset_enabled` | `vats.reset_enabled = true`: 정면이 정확히 2000 ms 이어진 프레임에서 누적을 지우고, 1800 ms로는 지우지 않는다 |
| `rules_alert_priority` | 같은 프레임의 `gaze_away_long`과 `microsleep`: 이벤트는 판정 순서, 경고는 우선순위 순서 |
| `rules_gap_over_10s_candidate` | 10초 넘는 끊김은 걸려 있던 전방 주시 후보도 버린다: 고개를 돌린 프레임 하나 뒤 10001 ms 끊기면 영역 밖은 끊김 뒤의 첫 프레임부터 새로 세고, 정확히 3000 ms 뒤에 `gaze_away_long` onset |
| `rules_gaze_candidate_dropped` | 2.0초 이탈, 앞을 본 프레임 하나, 3.3초의 프레임 끊김, 다시 이탈: 버려진 영역 안 후보의 구간은 영역 밖 시간이 아니다. onset은 끊김 뒤 1000 ms(`t_start_ms`는 처음 이탈한 프레임, `duration_ms` 6300) |
| `rules_forward_zone_eye_edges` | 눈동자 보조의 나머지 두 방향: `gaze_h` +0.251 / +0.249, `gaze_v` −0.351 / −0.349(아래를 봄). 절댓값으로 비교한다 |
| `rules_gaze_unable_pauses_timer` | 영역 밖 1800 ms에서 분석 불가가 시작되면 연속 영역 밖 시간이 멈추고, 분석 불가가 끝난 뒤 1200 ms에 `gaze_away_long` onset. 27초의 분석 불가 뒤에 시작한 `vats`의 `t_start_ms`는 그때 30초 창 안의 첫 영역 밖 구간의 시작이다(분석 불가 동안의 프레임 시각이 아니다) |
| `rules_drowsy_either` | `drowsiness.source = either`: 모델이 연 상태(source `model`)가 끝난 다음 스텝에 PERCLOS가 상태를 연다. 둘째 onset의 source는 `perclos`, `confidence`는 `1 − face_missing_ratio_60` |
| `rules_gap_over_10s_drowsy_model` | 10초 넘는 끊김은 모델 판정기의 재시작 금지 기억도 지운다. `drowsiness.model.refractory_ms`를 600000으로 덮어쓴 세션에서, 끊김 전에 끝난 상태의 금지 시간 안(199초 뒤)에 새 onset이 난다. 끊김 뒤의 모델 호출은 스텝 388부터다 |
| `rules_alert_pending_survives_end` | 속도에 막혀 미뤄 둔 `microsleep` 경고는 `gaze_away_long`의 end로 지워지지 않고, 속도가 오른 프레임에서 나간다. 미뤄 둔 `gaze_away_long` 경고는 자기 end에서 지워진다 |
| `rules_closure_after_dropped_run` | 22 ms짜리 감김이 500 ms 넘는 끊김으로 버려지고 같은 프레임에서 새 감김이 시작한다: `microsleep`의 `t_start_ms`와 `confidence`는 새 감김의 것이다 |
| `rules_gaze_glance_and_credit_times` | (1) 200 ms의 정면은 영역 밖 시간에 들어간다: `vats`가 정확히 10000 ms에 시작. (2) 얼굴이 사라진 시각은 다음 프레임의 시각과 관측 + 200 ms 중 이른 쪽: 관측 + 3150 ms의 프레임은 표지를 이어받지 않는다. (3) 후보는 인정 구간을 넘을 때만 버린다: 정확히 관측 + 3200 ms의 영역 밖 프레임에서 소급해 확정되고 `gaze_away_long` onset |
| `rules_calibrating_large_t0` | 첫 `t_ms`가 1731465123457인 세션의 규칙과 경고(보정부터). 이벤트의 시각은 `Long`이어야 한다. "처음 10초"와 `calibrating`의 구간은 첫 프레임 대비다. 기준선을 완성한 프레임이 `T_b`보다 1.3초 늦게 와도 누적은 `T_b`부터라 그 프레임이 빈 스텝 61을 발행한다(`expected.steps`) |
| `rules_vats_window_edge` | 30초 창의 시작에서 정확히 끝나는 영역 밖 구간은 창에 없다. 프레임이 3초 끊긴 뒤의 프레임에서 영역 밖 누적이 9000에서 11000 ms로 건너뛰고, 그때 창의 시작이 첫 구간의 끝과 같다: `vats`의 `t_start_ms`는 둘째 구간의 시작(21000), `duration_ms` 11000, `confidence` 8200 / 11000. 첫 구간을 남기는 구현은 2000과 30000을 낸다 |
| `rules_vats_confidence_face_lost` | `vats` onset의 `confidence`가 1보다 작은 경우. 영역 밖인 동안 얼굴이 2초 사라졌다 돌아온다(3초 안이라 영역 밖으로 친다). `pose_valid` 구간이 셋(시작보다 앞에서 끝난 것, 걸친 것, 뒤에서 시작한 것)이고 구간마다 `t_start_ms` 뒤의 부분만 더해 8000 / 10000이다. 틀리게 더하면 1.4나 0.74가 된다 |
| `rules_gaze_one_frame_glances` | 400 ms마다 프레임 하나에서만 고개를 돌리기를 31초 동안 되풀이한다(돌린 프레임의 시간을 더하면 15.8초). 영역 밖이 한 번도 확정되지 않으므로 이벤트도 경고도 없다. 영역 안 상태에서 버려진 영역 밖 후보의 구간은 영역 밖 누적에 들어가지 않는다 |
| `rules_microsleep_confidence_face_lost` | 감긴 채 얼굴 없는 프레임이 셋(400 ms 끊김) 낀 감김. 끊긴 시간은 길이에는 들어가서 2000의 프레임에서 `microsleep` onset이 나지만, `confidence`는 `closure_valid`인 프레임이 유지된 시간만 세어 700 / 1000이다(end는 1300 / 1600). 모든 프레임을 세는 구현은 1.0을 낸다 |
| `rules_alert_pending_survives_onset` | 속도에 막혀 미뤄 둔 `microsleep` 경고는 그 뒤에 온 `gaze_away_long`의 onset으로 지워지지 않는다. 속도가 오른 프레임에서 두 경고가 우선순위 순(`microsleep`, `gaze_away_long`)으로 함께 나간다. `rules_alert_pending_survives_end`는 다른 종류의 end만 지난다 |
| `rules_unable_small_t0` | 첫 `t_ms`가 5000이고 처음부터 얼굴이 없는 세션. "처음 10초"는 `t_ms` 0이 아니라 첫 프레임부터 잰다: `unable_to_analyze` onset은 `t_ms` 15000의 프레임에서 난다(0부터 재면 `t_ms` 10000). 얼굴이 돌아와 5000 ms가 쌓인 프레임에서 end(`confidence` 5000 / 7000) |
| `rules_drowsy_off_step_boundary` | `rules_drowsy_perclos_exact`와 같은 감김인데 프레임이 스텝 경계보다 50 ms 늦게 온다(스텝 k를 세션 1000k + 50의 프레임이 발행한다). `drowsy` onset의 발행 시각은 그 프레임이 아니라 스텝의 끝이라 `duration_ms`가 9000이고(프레임에서 재면 9050), 60 km/h에서 바로 나간 경고도 onset 이벤트 그대로다. `expected.steps`도 담는다 |

`rules_calibrating_provisional`은 보정 중 40~42초에 얼굴이 없어서 `calibrating` end의 `confidence`가 58000 / 60000이다.

속도 기준 위아래는 종류별로 이렇게 나뉘어 있다: `microsleep`·`sleep`·`gaze_away_long`은 `rules_speed_gates`, `vats`는
아래가 `rules_speed_gates`이고 위가 `rules_vats`, `drowsy`(50 km/h)는 아래가 `rules_drowsy_perclos`이고 위가
`rules_drowsy_model_stub`과 `e2e_250s`다. 기준과 정확히 같은 속도(20.0, 50.0 km/h)는 억제하지 않는다:
`rules_alert_blocked_microsleep_sleep`, `rules_gap_over_10s`, `rules_drowsy_perclos_exact`.

`config_overrides`가 있는 벡터는 다섯이다: `rules_drowsy_model_stub`과 `rules_drowsy_model_stall`
(`drowsiness.source = model`), `rules_gap_over_10s_drowsy_model`(`drowsiness.source = model`과
`drowsiness.model.refractory_ms = 600000`), `rules_drowsy_either`(`drowsiness.source = either`),
`rules_vats_reset_enabled`(`vats.reset_enabled = true`).

### model (모델)

| 벡터 | 고정하는 것 |
|------|-------------|
| `model_v0_inputs` | `model.file`의 sha256과, 입력 셋(`model_inputs[i]`, `[120][17]`)에 대한 `probs`. 입력은 0으로 채운 창, `e2e_250s`의 첫 창, 범위를 벗어난 값이 섞인 무작위 창 |

### e2e (전 과정)

| 벡터 | 고정하는 것 |
|------|-------------|
| `e2e_250s` | 250초의 프레임 → 보정(60초) → 스텝 61~250 → 스텝 239의 120스텝 창 → ONNX `probs` → 이벤트와 경고 |

`drowsiness_v0.onnx`는 무작위 가중치라 이 세션에서 `p_drowsy`가 0.6에 닿지 않는다(`expected.probs`의 열두 값은
0.324~0.328이고, 닿지 않는다는 것은 생성할 때 확인한다). 그래서 이 벡터의 `drowsy` onset과 end는 기본 판정원인
PERCLOS 규칙이 낸 것이고, 모델 확률은 `probs`로만 확인한다. 모델이 판정을 내는 경로는 `rules_drowsy_model_stub`이
고정한다. 학습한 모델이 나오면(M4) 이 벡터를 다시 만든다.

## Android 팀이 쓰는 법

1. 파일을 읽어 `pipeline_hash`가 앱이 번들한 spec 파일의 값과 같은지, `rules_hash`가 `rules.json`의 sha256과 같은지
   본다. 다르면 상수가 다른 것이므로 비교를 멈춘다. spec 파일은 지금 내보내기가 `export/feature_spec_v{N}.json`이라는
   이름으로 쓴다(모델 `drowsiness_v{N}.onnx` 옆). 계획서는 `feature_spec.json`이라고 적었고, 버전을 붙인 이름으로
   확정할지는 소유자의 확인을 기다리고 있다(docs/SPEC.md 20장).
2. `frames`를 `RawFrame`으로 바꾼다. `lm`의 키는 문자열이므로 정수로 바꾼다. 숫자는 `Double`, `t_ms`는 `Long`으로 읽는다.
3. 묶음에 맞게 돌린다.
   - `signals`: 프레임마다 `signals.compute`.
   - `model`: `model_inputs[i]`를 `float32 [1,120,17]`로 바꿔 ONNX에 넣는다(`features[0]`이 가장 오래된 스텝).
     번들한 모델 파일의 sha256이 `model.sha256`과 같은지 먼저 본다.
   - 나머지: `DriverPipeline(spec, rules, baseline, model)`을 만들고 프레임을 차례로 `update(frame, speed)`에 넣는다.
     `rules`는 `config_overrides`를 적용한 것, `baseline`은 `null`이 아니면 그 값(특징 누적 시작은
     `baseline.t_complete_ms`), `speed`는 위 표의 규칙으로 고른 값이다.
     `model`이 `stub`이면 i번째 호출(0부터)에 `probs[i]`를 돌려주고 목록이 끝나면 마지막 값을 되풀이하는 함수를,
     `onnx`이면 실제 모델을 넣는다. 모델은 스텝마다 불리므로(프레임이 끊겼다가 온 프레임에서는 여러 번) 호출 순서는
     스텝 순서다.
4. 출력을 모아 `expected`의 항목과 비교한다. 파일에 있는 항목만 비교하면 된다. `update`가 버린 프레임(같거나 이른
   `t_ms`)은 `closure`에 원소가 없다.
5. 모듈을 하나씩 옮기는 동안에는 `signals` → `baseline` → `runs` → `steps` → `rules` → `e2e` 순서로 통과시키면 된다.
   `baseline`과 `runs` 묶음은 위에 적은 대로 파이프라인 없이도 확인할 수 있다.

## 알아 둘 것

- 긴 세션은 파일 크기를 줄이려고 3~10 fps의 프레임을 쓴다(스텝 특징과 전 과정 6 fps, PERCLOS 3~5 fps, 스텁 모델 3 fps).
  시간 누적은 프레임률과 무관하게 정의돼 있고, 프레임 간격이 200 ms를 넘는 부분은 "데이터 없음"으로 센다.
  프레임률에 따른 감김 검출은 `runs` 묶음(30 fps, 15 fps, 누락)이 고정한다.
- 2026-11-14에 더한 벡터는 대부분 프레임의 시각과 값을 손으로 놓았다(`t_ms` = 세션 시간 + 1, 값은 계단 모양으로
  바뀐다). 정수 ms의 경계(정확히 250, 1000, 3000, 5000, 10000, 10001 ms)에 프레임을 놓으려는 것이다. 설명에 적힌
  시각은 세션 시간이다. `runs_rounding_large_t0`과 `rules_calibrating_large_t0`은 첫 `t_ms`가 1731465123457이고
  `rules_unable_small_t0`은 5000이다(설명의 시각은 그 프레임 대비).
- 판정 임계값에 붙은 값은 벡터에 두지 않았다. 임계값에서 1e-6 안에 있는 표본이 생기는 각본은 생성이 거부하고,
  보간한 시각이 반올림 경계(x.5 ms)에서 1e-6 ms 안에 붙지 않는지는 테스트가 확인한다. 그래서 수학 함수의 마지막 자리
  차이로 결과가 갈리지 않는다. 일부러 x.5 가까이에 둔 `runs_rounding_large_t0`의 오프셋도 7e-5 ms 떨어져 있고, 이는
  두 표본의 closure가 허용 오차(1e-6)만큼 달라질 때 오프셋이 움직이는 폭(3.7e-5 ms)보다 크다.
- 등호 자리를 벡터가 고정하는 것: 정수 ms의 비교(예: `rules_closure_3000ms`와 `rules_gaze_3100ms`의 3000 ms, 재경고
  대기, 디바운스, 10000 ms와 10001 ms의 끊김), 입력으로 그대로 들어오는 차량 속도(20.0, 50.0 km/h), 정수 ms의 몫으로
  정확히 나오는 `perclos_60`(0.15, 0.10)과 기준선의 얼굴 검출 비율(0.5, `baseline_burst_frames`), 정수 개수의 비교(기준선의
  얼굴 프레임 300개, `baseline_exactly_300_frames`), 그리고 `p_drowsy` 0.6(스텁이 `Float`를 돌려주는 구현에서는 0.6f가 0.6보다
  커서 어느 쪽이든 시작 조건이다).
- 등호 자리를 벡터가 고정하지 않는 것: 전방 주시 영역의 비교 다섯(`abs(yaw_rel) > 23.0`, pitch 둘, gaze 둘)과 closure
  임계는 실수로 계산한 값이 임계에 정확히 놓여야 드러나므로 바로 안쪽과 바깥쪽 값으로만 고정한다
  (`rules_forward_zone_edges`). 같은 까닭으로 눈 폭이 정확히 6 px인 눈, `abs(yaw_rel)`이 정확히 40도인 프레임, E가 정확히
  `baseline.e_min`이나 `baseline.e_max`인 기준선, 기준 yaw가 정확히 15도인 기준선, 하품과 끄덕임의 시작·끝 임계에 놓인 값,
  closure가 정확히 `step_features.perclos_thr`인 프레임도 벡터에 두지 않았다(전체 목록은 docs/SPEC.md 16.1절 끝).
  `p_drowsy <= 0.4`의 등호는 0.4를 `Float`로 정확히 적을 수 없어서 두지 않았다. 이
  자리들은 docs/SPEC.md의 비교 연산자를 그대로 옮긴다.
- 각본 시나리오의 값(눈꺼풀 수준, 고개 각도)은 설계값이고 실제 사람의 분포가 아니다. closure 상수는 실제 눈 감김으로
  확정되기 전의 잠정값이다(PLAN 11장 E0). 상수가 확정되면 `pipeline_hash`가 바뀌고 벡터를 다시 만든다.
- 파일의 `-0.0`은 실수 0과 같은 값이다(회전 행렬의 `−sin(0)`에서 나온다).

## 2026-11-14에 달라진 것

`pipeline_hash`와 `rules_hash`는 그대로다(상수는 바뀌지 않았다). 참조 구현의 동작이 여덟 군데 달라졌다(docs/SPEC.md의
머리말과 16.1절).

기존 43개 가운데 달라진 파일:

| 벡터 | 달라진 것 |
|------|-----------|
| `rules_vats` | `vats` end의 `confidence` 0.970873786407767 → 1.0(구간 전체의 `pose_valid` 시간 비율로 고쳤다) |
| `runs_lingering_lids` | 첫 구간의 `duration_ms` 1225 → 757(0.5 이상이던 시간의 합), 설명 |
| `rules_drowsy_model_stub`, `e2e_250s` | `expected.step_probs`와 `tolerances.step_probs`가 더해졌다. 다른 값은 같다 |
| `rules_calibrating_provisional` | 입력을 바꿨다: 40~42초의 프레임 12개가 얼굴 없는 프레임이다. `calibrating` end의 `confidence` 1.0 → 0.9667, 설명 |
| `runs_dropped_frames` | 설명만 |

나머지 37개는 바이트 단위로 같다. 새 파일은 25개다(위의 표에 있다).

같은 날 두 번째 검증 뒤에 13개를 더했다. 참조 구현은 바뀌지 않았고 위의 68개는 바이트 단위로 그대로다(sha256으로
확인했다). 새 파일은 그때까지 벡터로는 드러나지 않던 동작을 고정한다. 참조 구현의 한 줄을 일부러 틀리게 바꾼 사본을
만들어 벡터에 돌려 보는 점검에서, 틀렸는데도 벡터를 모두 통과하던 자리들이다.
저장소의 단위 테스트만 잡던 것: `runs_two_humps`, `runs_one_sample_truncated`, `rules_gaze_unable_pauses_timer`(멈추는
타이머), `rules_gap_over_10s_drowsy_model`, `rules_alert_pending_survives_end`, `rules_closure_after_dropped_run`,
`rules_gaze_glance_and_credit_times`, `rules_calibrating_large_t0`(규칙 쪽의 큰 `t_ms`). 단위 테스트도 잡지 못하던 것:
`steps_gap_exactly_10s`, `rules_gap_over_10s_candidate`, `rules_gaze_candidate_dropped`, `rules_forward_zone_eye_edges`,
`rules_drowsy_either`, `rules_gaze_unable_pauses_timer`(`vats`의 `t_start_ms`), `rules_calibrating_large_t0`(늦게 온
기준선 완성 프레임). 합계 크기는 22,650,508 bytes(68개)에서 26,908,941 bytes(81개)가 됐다.

세 번째 검증 뒤에 8개를 더했다. 이번에도 참조 구현은 바뀌지 않았고 위의 81개는 바이트 단위로 그대로다(sha256으로
확인했다). 같은 점검에서(이번에는 조건을 항상 참이나 거짓으로 바꾸기와 호출 한 줄 지우기도 했다) 벡터 81개를
모두 통과하던 한 줄짜리 오류들이다. 저장소의 단위 테스트만 잡던 것 여섯: `rules_vats_window_edge`(30초 창의 시작에서
정확히 끝나는 구간), `runs_two_rises_before_peak`(정점이 옮겨지기 전의 두 증가), `rules_vats_confidence_face_lost`
(`pose_valid`가 끊긴 `vats`의 `confidence`), `rules_gaze_one_frame_glances`(확정되지 못한 곁눈질),
`rules_microsleep_confidence_face_lost`(얼굴이 사라진 감김의 `confidence`), `rules_alert_pending_survives_onset`(다른
종류의 onset과 미뤄 둔 경고). 단위 테스트도 잡지 못하던 것 하나: `rules_unable_small_t0`(첫 `t_ms`가 작은 세션의
"처음 10초"). 그 정비에서 같은 점검을 되풀이하다 찾은 것 하나: `rules_drowsy_off_step_boundary`(스텝 경계에서 벗어난
프레임이 내는 `drowsy`의 발행 시각과 바로 낸 경고. 다른 벡터에서는 `drowsy`를 낸 프레임의 시각이 스텝의 끝과 1 ms
이하로만 달라서, 발행 시각을 프레임에서 재는 구현이 통과했다). 합계 크기는 26,908,941 bytes(81개)에서
27,983,687 bytes(89개)가 됐다.

네 번째 검증 뒤에 13개를 더했다. 참조 구현은 바뀌지 않았고 위의 89개는 바이트 단위로 그대로다(89개의 합계가 그 전에
적어 둔 27,983,687 bytes와 같고, 벡터를 모두 다시 만들면 저장된 파일과 바이트 단위로 같다).
그때까지의 점검은 감김 검출, 규칙, 파이프라인 모듈만 보았다. 같은 점검을 신호, 기준선과 closure, 스텝 특징 모듈
(`drowsy/signals.py`, `calibrate.py`, `step_features.py`)에 돌렸더니 한 줄을 틀리게 바꿔도 벡터 89개를 모두 통과하는
자리가 더 있었고, 새 벡터는 그 가운데 이식 코드의 출력이 실제로 달라지는 자리를 고정한다. 신호 하나
(`signals_iris_tilted_eyes`), 기준선과 closure 넷(`baseline_odd_sample_count`, `baseline_burst_frames`,
`baseline_exactly_300_frames`, `closure_blink_span_floor`), 스텝 특징 여덟(`steps_yawn_edges`, `steps_nod_edges`,
`steps_nod_slow_mean`, `steps_nod_after_long_gap`, `steps_two_closures_in_one_step`, `steps_closure_above_cap`,
`steps_closure_exactly_500ms`, `steps_pitch_std_at_5000ms`)이다. 여섯은 그때 단위 테스트도 잡지 못하던 동작이다(기울어진
눈 축의 홍채 비율, 끄덕임 도중의 얼굴 소실, 느린 평균을 빼는 방향, 10초 넘는 끊김 뒤의 끄덕임 검출기, 한 스텝에서 끝난 두
감김 가운데 긴 것, 얼굴 시간이 정확히 5000 ms일 때의 `pitch_std_60`). 새 벡터마다 그 줄을 틀리게 옮긴 구현이 내는 값은 docs/SPEC.md 16.1절의 표에 있다. 합계 크기는
27,983,687 bytes(89개)에서 30,588,827 bytes(102개)가 됐다. 새 파일의 크기는 보정 세션 셋이 653,722, 707,708,
716,492 bytes(얼굴 프레임 300개가 있어야 기준선이 완성되므로 더 줄일 수 없다), 나머지 열이 6,577~112,137 bytes다.

벡터가 고정하지 않아 단위 테스트로만 남은 동작은 docs/SPEC.md 16.1절 끝에 있다.

감김 벡터 여섯은 2026-11-14의 소유자 결정을 고정한다. 진행 중인 감김에서 눈꺼풀이 진입 임계와 이탈 임계 사이에
머물러도 구간은 끝나지 않고, 진입 임계 이상이던 시간이 그 구간의 길이에 계속 쌓인다(소유자의 답은 "지금대로 둠"이다.
결정 기록, 알려진 한계, 잰 사례는 docs/SPEC.md 20장 머리의 "2026-11-14에 결정되어 이 목록에서 빠진 것"에 있다. 20장의
1번이던 질문이라 그 번호는 비워 두었다). 함께 제시했지만 고르지 않은 대안("진입 임계 아래에 `hold_ms`(500)보다 오래 머물면
구간을 끝낸다")을 골랐다면 바뀌었거나 그 대안의 정의에 따라 바뀌었을 벡터가 이 여섯이다. 다섯에는 한 구간 안에서
진입 임계 아래에 500 ms 넘게 머무는 장면이 있고(검출기에 기록을 붙여 모든 벡터에서 잰 값),
`runs_multi_span_truncated`의 둘째 구간은 진입 임계 아래로 내려간 뒤 얼굴이 사라져 잘린다. 이식 코드는 이 벡터들이
고정한 누적을 그대로 재현해야 한다:

| 벡터 | 진입 임계 아래에 머문 시간 |
|------|----------------------------|
| `runs_lingering_lids` | 둘째 구간이 0.5 아래로 내려간 뒤 555 ms 만에 끝난다(첫 구간의 머묾은 468 ms) |
| `runs_multi_span` | 첫 구간의 둘째 머묾 540 ms |
| `runs_multi_span_truncated` | 둘째 구간이 0.5 아래로 내려간 뒤 얼굴이 사라져 잘린다(머묾과 얼굴 소실이 이어진다) |
| `steps_multi_span_runs` | 구간 A 540 ms, 구간 C 2140 ms |
| `rules_closure_dwell_below_1s`, `rules_closure_dwell_reaches_1s` | 4966 ms |

머문 시간이 500 ms 이하인 `rules_closure_three_spans`(266 ms), `runs_multi_span`의 둘째 구간, `steps_multi_span_runs`의
구간 B는 대안으로도 바뀌지 않았을 장면이다. 두 번째 검증 뒤에 더한 13개, 세 번째 검증 뒤에 더한 8개, 네 번째 검증 뒤에
더한 13개에는 500 ms 넘는 머묾이 없다(같은 방법으로 다시 쟀다. 토막이 여럿인 `runs_two_humps`와
`runs_two_rises_before_peak`의 머묾은 모두 500 ms 이하다).

소유자의 결정을 기다리는 질문 가운데 여기서는 전방 주시 쪽의 둘만 설명한다. 다른 질문도 답에 따라 벡터가 다시
만들어진다: 어느 질문이 어느 벡터를 바꾸는지는 docs/PLAN.md 14.2절 표의 "다시 만들 것" 칸에 질문마다 적혀 있다. 답이
나오면 그 칸의 벡터를 다시 받는다. 전방 주시 쪽의 둘은 이렇다. 한 프레임만 본 관측을 인정 구간으로 확정할지(docs/SPEC.md 20장 5번)가
바뀌면 `rules_gaze_glance_and_credit_times`의 셋째 장면이 다시 만들어진다. 인정 구간이 끝나는 바로 그 시각의 관측 없는
프레임을 어떻게 다룰지(20장 9번)는 다르다: 제안된 대안을 골라도 저장된 기대값은 하나도 바뀌지 않는다(대안을 넣은 참조
구현의 사본에 벡터 102개를 돌리면 모두 그대로 통과한다). 결과가 갈리는 7200 / 7201 ms 경계를 지나는 벡터가 없기
때문이다. 그때는 `rules_gaze_credit_boundary`의 설명 문장(첫 장면)만 고쳐 다시 만들고, 그 경계를 고정하는 벡터를 새로
더한다.
