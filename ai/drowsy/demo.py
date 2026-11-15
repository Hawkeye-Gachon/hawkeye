"""데모: 영상 하나에 전체 파이프라인을 돌려 타임라인 PNG와 이벤트 JSON을 쓴다 (PLAN 9장 데모 항목). 이식 대상이 아니다.

흐름
    프레임 → `DriverPipeline`(영상 자신의 앞 60초가 기준선, 모델 콜백은 onnxruntime) → 타임라인 PNG + 이벤트 JSON
- 프레임을 얻는 길은 둘이다.
    `run_demo` / `--video`:   `drowsy.video_io`가 디코딩한 프레임을 `drowsy.extract`의 VIDEO 모드 FaceLandmarker에
                              하나씩 넣는다. 추출과 같은 함수를 쓰고, `paths.model_path`의 face_landmarker.task가 필요하다.
                              영상을 세우는 회전은 `--rotation`으로 주거나, 주지 않으면 추출과 같은 자동 점검으로 한 번
                              정한다(아래 "영상의 회전").
    `run_demo_raw` / `--raw`: 이미 추출했거나 `drowsy.synth`가 쓴 RawFrame parquet을 읽는다. MediaPipe가 필요 없다.
- 신호, closure, 감김 검출, 스텝 특징, 규칙, 경고는 여기서 계산하지 않는다. `DriverPipeline`이 낸 값을 그대로 그리고
  적는다(PLAN 3.1절).
- 모델을 쓰기 전에 메타데이터를 대조한다(`verify_model`). 어긋나면 실행하지 않는다.
- 얼굴 프레임은 저장하지 않는다(PLAN 1.2절의 결정 D13). 디코딩한 영상은 메모리에서 MediaPipe로만 가고, 쓰는 파일은
  PNG와 JSON 둘뿐이다.

출력 (`out_dir` 아래, 이름은 입력 파일 이름의 줄기에서 딴다)
- `<줄기>_timeline.png`: 가로축(세션 시간)을 맞춘 네 줄. 크기는 `FIGURE_SIZE_IN` x `demo.figure_dpi` 픽셀로 고정이다.
    1. closure와 임계값(PERCLOS 기준, 규칙용 진입·이탈, 특징용 진입·이탈). 기준선 완성 전의 임시 closure는 옅게 그린다.
    2. 전방 주시 상태: 영역 밖, 알 수 없음(자세 무효·분석 불가·프레임이 끊긴 시간).
    3. 모델 출력 p_drowsy와, 더 옅게 p_low_vigilant. 점은 모델을 부른 스텝마다 하나이고 가로 위치는 그 스텝의 끝
       시각이다(프레임이 끊겼다가 와서 프레임 하나가 스텝을 여럿 발행해도 스텝마다 점이 있다. 2026-11-13에 소유자가
       "밀린 스텝마다 모델을 호출한다"고 정했기 때문이다).
       출력이 하나도 없으면 줄을 비우고 이유를 적는다.
    4. 규칙 이벤트: 종류마다 한 줄, 시작부터 종료까지의 구간, 색은 심각도, 임시 판정은 빗금, 경고는 표식.
       표식의 모양이 경고의 종류다: 바로 낸 경고(채운 세모), 미뤘다 낸 경고(빈 세모), 되풀이 경고(세로 금).
- `<줄기>_events.json`:
    note               합성 입력이면 합성 데이터임을 알리는 문장(파일의 첫 항목)
    source             입력 종류와 경로, 합성 여부, 추출 정보. `--video` 입력이면 영상을 세운 회전도 적는다:
                       `rotation`(적용한 각도, 반시계), `rotation_source`(`manual_override`는 `--rotation`으로 준 값,
                       `auto_check`는 자동 점검이 고른 값), `rotation_ambiguous`(자동 점검이 각도를 가르지 못했는지),
                       `rotation_check`(점검이 잰 값. 추출 결과의 메타데이터와 같은 꼴이고 수동 값이면 null),
                       `display_flip`(표시 행렬에 뒤집기가 든 영상을 `--rotation`으로 돌려 읽었는지).
                       `--raw` 입력이면 추출 결과의 메타데이터에 적힌 회전을 같은 이름으로 옮겨 적는다: `rotation`
                       (추출에 적용된 각도. 적혀 있지 않으면 null), `rotation_source`(`auto_check`, `manual_override`,
                       점검 기록 없이 영상의 회전 메타데이터만 쓴 결과는 `video_metadata`, 회전이 적혀 있지 않으면 null),
                       `rotation_ambiguous`, `rotation_check`(점검 기록. 없으면 null), 그리고 `rotation_note`(명령이
                       출력하는 회전 줄 그대로. 아래 "추출 결과의 회전")
    model              모델 파일과 sha256, 메타데이터 열 개, 함께 나간 feature_spec 사본과 그 해시
    spec               실행에 쓴 feature_spec 파일과 sha256, pipeline_hash, 규칙 파일과 rules_hash
    drowsiness_source  규칙 엔진의 졸음 판정원(`drowsiness.source`. 기본값을 모델로 둘 조건은 PLAN 1.2절의 결정 D11)
    speed_kmh          넣어 준 차량 속도(없으면 null)
    session            받아들인 첫 프레임과 마지막 프레임의 t_ms, 길이
    baseline           기준선 값과 상태, 판정 통계(완성 전에 입력이 끝났으면 null)
    counters           파이프라인 계수와 얼굴 프레임 수, 이벤트 수, 경고 수(전체와 종류별: `alerts_immediate`,
                       `alerts_deferred`, `alerts_repeat`)
    probabilities      모델 출력 수, 첫 출력과 마지막 출력의 스텝 끝 시각, 출력이 없으면 그 이유
    events             `DriverStateEvent` 여덟 필드 + 발행한 프레임의 `t_ms` + `alert`(그 프레임에서 경고로 나갔는지.
                       막혔다가 나중에 나간 경고는 여기가 아니라 `alerts`에 `deferred`로 적힌다)
    alerts             실제로 낸 경고 전부. 여덟 필드 + 경고를 낸 프레임의 `t_ms` + `kind`(경고의 종류, 아래)
  시각은 모두 원래 t_ms 시계다(`baseline.t_complete_ms`만 세션 시간).

경고의 종류 (`kind`. `AlertPolicy`가 낸 경고를 `AlertKinds`가 가린다. 2026-11-13에 소유자가 "막힌 경고는 막힘이 풀렸을 때
상태가 계속 중이면 한 번 낸다"고 정한 뒤로 경고가 상태의 시작보다 늦게 나갈 수 있어서 구분한다)
    immediate   상태가 시작된 그 프레임에서 나간 경고(그 프레임의 이벤트 가운데 그 `onset`이 있다).
    deferred    그 상태의 첫 경고인데 시작한 프레임보다 나중에 나간 것. 시작할 때는 재경고 대기나 속도 억제에 막혔고,
                상태가 이어지는 동안 막힘이 풀린 첫 프레임에서 나갔다. 되풀이하는 종류(sleep)의 첫 경고가 막혔다가
                나간 것도 여기에 든다.
    repeat      같은 상태에서 이미 경고가 나간 뒤의 경고(기본 설정에서는 sleep의 3초 주기).
  deferred와 repeat는 그 `onset`의 사본이고 `duration_ms`만 경고를 낸 시각 기준이다.

영상의 회전 (`--video`만. 각도는 `frame.rotation`과 설정의 수동 회전 보정표와 같은 규약: 세우려면 반시계로 돌려야 하는 각도)
    `--rotation {0,90,180,270}`  그 각도로 돌린다. 영상의 회전 메타데이터와 자동 점검을 대신한다.
    주지 않으면                  `extract.check_rotation`을 한 번 돌려(설정의 `extraction.rotation_check_frames`,
                                 `rotation_check_interval_ms`, `rotation_roll_margin_deg`) 고른 각도를 쓴다.
  어느 쪽이든 고른 각도와 출처, 가르지 못했는지를 프레임을 읽기 전에 한 줄로 출력하고, 그림의 머리말과 JSON의 `source`에
  적는다. 점검은 고른 각도와 그 반대 각도(0도와 180도)를 가르지 못할 수 있다(`extract.choose_rotation`). 그때는
  `rotation_ambiguous`가 참이고 출력에 WARNING이 붙는다: 영상을 직접 보고 `--rotation`으로 다시 실행한다.

추출 결과의 회전 (`--raw`만. parquet은 추출할 때 이미 세웠으므로 데모는 돌리지 않고, 추출이 메타데이터에 남긴 기록을 읽어 알린다)
  합성 데이터 표시 줄 다음에 회전 줄을 하나 출력하고 같은 문장을 JSON의 `source.rotation_note`에 적는다(`read_raw_rotation`).
    점검 기록(`rotation_check`)이 있다          자동 점검이 고른 각도와 가르지 못했는지, 점검이 잰 값. 가르지 못했으면
                                               WARNING이 붙고 그림의 머리말에도 한 줄이 붙는다: 영상을 직접 보고 각도를 손으로
                                               줘서 다시 추출한다(`extract --video --rotation`, 그 밖의 입력은 수동 회전 보정표).
    점검 기록이 없고 수동 값으로 추출됐다       각도와, 수동 값이라는 것만 적는다(`--video`에 `--rotation`을 줬을 때의 줄과 같다).
    점검 기록이 없다                           기록이 없다고 적는다(합성 세션, 영상의 회전 메타데이터만 믿고 추출한 전의 결과).

명령 (저장소 루트에서 실행)
    python -m drowsy.demo --video path.mp4 [--rotation 90] [--model export/drowsiness_v0.onnx] [--out out/demo]
                          [--speed-kmh 60]
    python -m drowsy.demo --raw path.parquet [--model ...] [--out ...] [--speed-kmh ...]
종료 코드: 0 성공, 2 실행할 수 없음(모델 메타데이터 불일치, 모델 파일 없음, 읽을 수 없는 입력, 표시 행렬에 뒤집기가
든 영상에 `--rotation`을 주지 않음, `--raw`에 `--rotation`을 줌 등).

가정: 계획서의 명령에는 `--raw`와 `--config`가 없다. `--raw`는 `--video` 대신 주는 입력이고(둘 중 하나만),
  `--config`는 다른 모듈과 같은 실험 설정 파일이다(feature_spec, 규칙, MediaPipe 모델 경로, 그림 해상도를 읽는다).
가정: `--model`을 주지 않으면 설정의 `demo.model_path`를 쓰고, 그 파일이 아직 없으면(학습한 v1을 내보내기 전)
  `paths.export_dir`의 무작위 가중치 모델 `drowsiness_v0.onnx`를 쓰며 그 사실을 표준 오류에 알린다.
  `--out`을 주지 않으면 `demo.out_dir` 아래 `demo/`(기본 설정에서 `out/demo`)에 쓴다.
가정: 파이프라인은 설정의 feature_spec(`paths.feature_spec`)으로 돌리고, 모델의 `pipeline_hash`가 그것과 다르면
  거부한다. 표준화 상수는 그래프 안에 있으므로 모델과 함께 나간 사본(`feature_spec_v{N}.json`)은 해시 대조에만 쓴다.
가정: 추출 parquet에 MediaPipe 모델 sha256이 적혀 있고 spec의 값과 다르면 거부한다(다른 랜드마크 모델의 신호다).
  합성 parquet에는 그 값이 없으므로 검사하지 않는다.
가정: `--speed-kmh`는 세션 내내 같은 값이다(명령은 시간에 따라 변하는 속도 입력을 받지 않는다. `run_pipeline`은
  프레임 시각을 받아 속도를 돌려주는 함수도 받는다).
가정: 경고의 종류는 종류별로 "지금 상태에서 경고가 이미 나갔는지"만 기억해서 가린다. 상태는 그 종류의 `onset`
  이벤트에서 새로 시작한 것으로 본다(`AlertPolicy`가 새 `onset`에서 앞 상태의 미뤄 둔 경고와 되풀이 기록을 지우는
  것과 같은 기준이다).
가정: 이벤트 JSON의 경고에는 예전의 `repeat`(불) 대신 `kind`(세 값)만 적는다. 두 필드를 함께 두지 않는다
  (`repeat`가 참이던 경고는 이제 `kind`가 `repeat`이거나 `deferred`다).
가정: 그림에서 경고의 종류는 표식의 모양으로 구분한다(색은 하나다). 계획서는 표식을 정하지 않았다. 범례에는 그
  종류의 경고가 하나도 없어도 세 종류를 늘 싣고, 범례가 길어진 만큼 이벤트 줄의 제목을 줄였다.
가정: 전방 주시 줄은 프레임마다 `DriverPipeline.gaze_zone`을 읽어 그 프레임이 대표하는 시간(다음 프레임까지, 최대
  `clock.max_hold_ms`)에 칠한다. 프레임이 대표하지 않는 시간은 "알 수 없음" 줄에 넣는다. `gaze_zone`은 확정된
  상태(디바운스 250 ms 뒤)이므로, 영역 밖 표시는 시선 이벤트의 소급된 `t_start_ms`보다 그만큼 늦게 시작한다.
가정: 이벤트 구간은 `t_start_ms`부터 `end` 이벤트의 `t_start_ms + duration_ms`까지다. `sleep`의 `end`는 같은 감김의
  `microsleep` 구간도 닫는다(PLAN 8.2절: 한 감김은 도달한 가장 높은 종류의 `end` 하나만 낸다). 입력이 끝날 때까지
  `end`가 없는 구간은 마지막 프레임까지 그린다.
가정: 그림의 글자는 영어만 쓰고 글꼴은 matplotlib에 들어 있는 DejaVu Sans로 고정한다. 입력 파일 이름의 ASCII가
  아닌 글자는 그림에서만 `\\uXXXX` 꼴로 바꿔 적는다(없는 글리프가 생기지 않게. JSON에는 원래 경로를 적는다).
가정: 심각도 색은 0~3의 네 단계다. 규칙 설정이 그 밖의 심각도를 주면 가까운 쪽 끝 단계의 색으로 그린다.
가정: 계획서는 출력 파일의 이름을 정하지 않았다. `<줄기>_timeline.png`, `<줄기>_events.json`으로 쓰고, 같은 이름의
  파일이 있으면 덮어쓴다.
가정: 합성 입력(`drowsy.synth`가 쓴 parquet)으로 만든 그림과 JSON은 첫 줄에 합성 데이터임을 적는다.
가정: `--video`의 회전은 `--rotation`과 자동 점검 둘로만 정한다. 설정의 수동 회전 보정표(`extraction.rotation_overrides`)는
  보지 않는다(그 표의 키는 추출 입력 폴더 기준 경로여서 영상 하나만 받는 데모에서는 어느 영상을 가리키는지 정해지지
  않는다). 점검이 가르지 못해도 멈추지 않고 고른 각도로 실행하며 출력, 그림, JSON에 표시한다.
가정: 표시 행렬에 뒤집기가 든 영상은 회전 메타데이터를 믿을 수 없어 자동 점검도 하지 않는다. `--rotation`이 없으면
  거부하고(오류 문장이 `--rotation`을 알려 준다), 있으면 그 각도로만 돌리고 거울상은 되돌리지 않는다.
가정: `--raw` 입력은 추출할 때 이미 세운 결과이므로 `--rotation`을 함께 주면 거부한다. 추출 때의 회전 정보는
  parquet 메타데이터에서 읽어 한 줄로 출력하고 JSON의 `source`에 옮겨 적는다(위 "추출 결과의 회전").
가정: `--raw`의 회전 줄은 그 파일의 메타데이터만 보고 정한다. 설정의 수동 회전 보정표는 보지 않는다(표의 키는 추출 입력
  폴더 기준 경로여서 parquet 하나만 받는 데모에서는 어느 영상을 가리키는지 정해지지 않는다). 그래서 자동 점검이 가르지
  못한 영상을 사람이 보정표에 같은 각도로 적어 확인한 경우에도(그때 추출은 파일을 다시 만들지 않는다) 여기서는 가르지
  못했다고 나온다. 보정표까지 대조한 판정은 `python -m drowsy.dataset`, `signal_check e0`, `evaluate unlabeled`이 한다.
가정: `--raw`의 회전 줄은 합성 데이터 표시 줄 다음에 출력한다(합성 입력이면 출력의 첫 줄이 합성 데이터 표시여야 한다).
  `--video`의 회전 줄은 프레임을 읽기 전에 정해지는 대로 출력하므로 첫 줄이다.
가정: `--raw` 입력의 그림에는 점검이 가르지 못한 경우에만 회전 줄을 넣는다(합성 입력의 머리말은 이미 다섯 줄을 다 쓴다).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import textwrap
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import matplotlib
import numpy as np
import onnxruntime as ort
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator

from drowsy import export_onnx, extract, signals
from drowsy.calibrate import BaselineDiagnostics
from drowsy.config import ExperimentConfig, load_config
from drowsy.dataset import SYNTHETIC_NOTE
from drowsy.pipeline import DriverPipeline, ModelFn
from drowsy.rules import ZONE_IN, ZONE_OUT, ZONE_UNKNOWN
from drowsy.spec import FeatureSpec, PipelineSpec, RulesConfig, file_sha256, load_feature_spec, load_rules, rules_hash
from drowsy.types import (
    CLASS_DROWSY,
    CLASS_LOW_VIGILANT,
    DROWSY_SOURCE_PERCLOS,
    EVENT_MICROSLEEP,
    EVENT_SLEEP,
    EVENT_TYPES,
    PHASE_ONSET,
    Baseline,
    DriverStateEvent,
    RawFrame,
)
from drowsy.video_io import DecodeCounts, iter_video_frames, normalize_rotation

DEFAULT_CONFIG_PATH = "config/default.yaml"
FALLBACK_MODEL_NAME = "drowsiness_v0.onnx"  # demo.model_path의 파일이 아직 없을 때 쓰는 무작위 가중치 모델
DEFAULT_OUT_SUBDIR = "demo"
FIGURE_SIZE_IN: tuple[float, float] = (14.0, 9.0)  # 그림 크기(인치). 픽셀 크기는 여기에 demo.figure_dpi를 곱한 값이다

SOURCE_VIDEO = "video"
SOURCE_RAW = "raw_parquet"
NOTE_RECORDING = "Driver-state timeline of one recording. Assistive guidance only, not a safety system."

# 경고의 종류(`AlertRecord.kind`, 이벤트 JSON의 `alerts[].kind`). 뜻은 모듈 설명에 있다.
ALERT_IMMEDIATE = "immediate"
ALERT_DEFERRED = "deferred"
ALERT_REPEAT = "repeat"
ALERT_KINDS: tuple[str, ...] = (ALERT_IMMEDIATE, ALERT_DEFERRED, ALERT_REPEAT)

# 프레임의 `t_ms`를 받아 그 프레임의 차량 속도(km/h, 모르면 None)를 돌려주는 함수
SpeedFn = Callable[[int], float | None]


class DemoError(RuntimeError):
    """데모를 진행할 수 없을 때(모델 메타데이터 불일치, 모델 파일 없음, 빈 입력 등)."""


# ---------------------------------------------------------------------------
# 모델: 메타데이터 대조, onnxruntime 콜백
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """`verify_model`을 통과한 ONNX 모델."""

    path: Path
    sha256: str  # 모델 파일 바이트의 sha256
    metadata: dict[str, str]  # metadata_props 전부
    spec_copy: Path  # 모델과 함께 나간 feature_spec 사본(`feature_spec_v{N}.json`)
    feature_spec_hash: str  # 그 사본 바이트의 sha256. 메타데이터의 값과 같다


def verify_model(onnx_path: Path, running: FeatureSpec) -> ModelInfo:
    """모델의 메타데이터가 spec 파일들과 맞는지 대조한다. 어긋나면 `DemoError`다.

    - `feature_spec_hash`: 모델 옆의 사본 `feature_spec_v{model_version}.json` 바이트의 sha256과 같아야 한다.
    - `pipeline_hash`: 그 사본의 값, 그리고 running(파이프라인을 실제로 돌릴 feature_spec)의 값과 같아야 한다.
      다르면 모델이 학습한 스텝 특징과 지금 계산할 스텝 특징의 정의가 다르다.
    """
    onnx_path = Path(onnx_path)
    if not onnx_path.is_file():
        raise DemoError(
            f"ONNX model not found: {onnx_path}. Fix: pass --model, e.g. the random-weight contract model "
            f"export/{FALLBACK_MODEL_NAME} (python -m drowsy.export_onnx --random-init --kind tcn --version 0)."
        )
    try:
        metadata = export_onnx.read_metadata(onnx_path)
    except ValueError as exc:  # export_onnx.ContractError: "<경로>: not a readable ONNX file (...)"
        raise DemoError(str(exc)) from exc
    missing = [key for key in ("model_version", "feature_spec_hash", "pipeline_hash") if key not in metadata]
    if missing:
        raise DemoError(f"{onnx_path}: model metadata lacks {missing}; not a model exported by drowsy.export_onnx")
    try:
        spec_copy = export_onnx.exported_spec_path(onnx_path)
    except export_onnx.ContractError as exc:
        raise DemoError(str(exc)) from exc
    if not spec_copy.is_file():
        raise DemoError(
            f"{onnx_path}: the feature spec copy shipped with the model is missing ({spec_copy}). "
            "Fix: keep feature_spec_v{N}.json next to drowsiness_v{N}.onnx."
        )
    copy_hash = file_sha256(spec_copy)
    if copy_hash != metadata["feature_spec_hash"]:
        raise DemoError(
            f"feature_spec_hash mismatch: {onnx_path} was exported with feature spec "
            f"{metadata['feature_spec_hash']} but {spec_copy} hashes to {copy_hash}. "
            "Fix: use the model and the feature spec copy from the same export."
        )
    try:
        shipped = load_feature_spec(spec_copy)
    except ValueError as exc:  # SpecError
        raise DemoError(f"{spec_copy}: {exc}") from exc
    if shipped.pipeline_hash != metadata["pipeline_hash"]:
        raise DemoError(
            f"pipeline_hash mismatch: {onnx_path} metadata says {metadata['pipeline_hash']} but its feature spec "
            f"copy {spec_copy} says {shipped.pipeline_hash}"
        )
    if metadata["pipeline_hash"] != running.pipeline_hash:
        raise DemoError(
            f"pipeline_hash mismatch: {onnx_path} was exported for pipeline {metadata['pipeline_hash']} but the "
            f"feature spec used to run the pipeline has {running.pipeline_hash}. Fix: run with the feature spec "
            "the model was trained with, or export a model for the current pipeline constants."
        )
    return ModelInfo(
        path=onnx_path,
        sha256=file_sha256(onnx_path),
        metadata=metadata,
        spec_copy=spec_copy,
        feature_spec_hash=copy_hash,
    )


def make_model_fn(onnx_path: Path, window_steps: int, n_features: int) -> ModelFn:
    """`DriverPipeline`에 넣을 모델 함수. 원시 스텝 특징 창(가장 오래된 스텝이 먼저)을 받아 확률 3개를 돌려준다.

    onnxruntime CPU, 스레드 1개(앱에 권장하는 설정과 같다). 모델의 입력 모양이 [1, window_steps, n_features]가
    아니면 `DemoError`다.
    """
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
    expected = [1, window_steps, n_features]
    shape = list(session.get_inputs()[0].shape)
    if shape != expected:
        raise DemoError(f"{onnx_path}: model input shape is {shape}, the pipeline feeds {expected}")

    def model(window: Sequence[Sequence[float]]) -> Sequence[float]:
        tensor = np.asarray(window, dtype=np.float32).reshape(1, window_steps, n_features)
        output = session.run([export_onnx.OUTPUT_NAME], {export_onnx.INPUT_NAME: tensor})[0]
        return [float(value) for value in output[0]]

    return model


# ---------------------------------------------------------------------------
# 파이프라인 실행
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProbPoint:
    """모델 출력 하나(모델을 부른 스텝마다 하나). `t_end_ms`와 `k`는 그 스텝(창의 마지막 스텝)의 끝 시각과 번호다.

    `t_ms`는 그 스텝을 발행시킨 프레임이다. 프레임이 끊겼다가 와서 스텝을 여럿 발행했으면 여러 점의 `t_ms`가 같다.
    """

    t_ms: int
    t_end_ms: int
    k: int
    probs: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class EventRecord:
    """상태 이벤트 하나. `t_ms`는 발행한 프레임, `alert`는 그 프레임에서 경고로도 나갔는지다.

    막혔다가 나중에 나간 경고(`ALERT_DEFERRED`)는 `alert`를 바꾸지 않는다. 그 경고는 `DemoRun.alerts`에 있다.
    """

    t_ms: int
    event: DriverStateEvent
    alert: bool


@dataclass(frozen=True, slots=True)
class AlertRecord:
    """실제로 낸 경고 하나. `t_ms`는 경고를 낸 프레임, `kind`는 경고의 종류(`ALERT_KINDS`, 뜻은 모듈 설명)다."""

    t_ms: int
    event: DriverStateEvent
    kind: str


class AlertKinds:
    """프레임마다 `AlertPolicy`가 낸 경고의 종류(`ALERT_KINDS`)를 가린다. 세션마다 하나를 만든다.

    `classify(events, alerts)`를 받아들인 프레임마다, 그 프레임의 `PipelineOutput.events`와 `alerts`로 부른다.
    - 그 프레임의 이벤트 가운데 있는 경고(그 프레임에서 시작한 상태의 `onset` 자신)는 `ALERT_IMMEDIATE`다.
    - 그 밖의 경고는 앞선 프레임의 `onset`의 사본이다. 그 상태에서 처음 나가는 경고면 `ALERT_DEFERRED`, 이미 경고가
      나간 상태면 `ALERT_REPEAT`다.
    종류별로 "지금 상태에서 경고가 나갔는지"만 기억한다. 그 종류의 `onset` 이벤트가 오면 새 상태로 보고 지운다.
    가정: `onset`을 본 적 없는 종류의 사본이 오면(`AlertPolicy`는 그렇게 내지 않는다) 그 상태의 첫 경고로 본다.
    """

    def __init__(self) -> None:
        self._alerted: dict[str, bool] = {}  # 종류 -> 지금 상태에서 경고가 이미 나갔는지

    def classify(self, events: Sequence[DriverStateEvent], alerts: Sequence[DriverStateEvent]) -> list[str]:
        """alerts와 같은 순서로 그 종류를 돌려준다."""
        for event in events:
            if event.phase == PHASE_ONSET:
                self._alerted[event.type] = False
        kinds: list[str] = []
        for alert in alerts:
            if alert in events:
                kinds.append(ALERT_IMMEDIATE)
            elif self._alerted.get(alert.type, False):
                kinds.append(ALERT_REPEAT)
            else:
                kinds.append(ALERT_DEFERRED)
            self._alerted[alert.type] = True
        return kinds


@dataclass(frozen=True, slots=True)
class DemoRun:
    """`run_pipeline`의 결과. 프레임별 값은 받아들인 프레임마다 하나씩이고 길이가 모두 같다. 시각은 원래 t_ms 시계다."""

    t_ms: tuple[int, ...]
    closure: tuple[float, ...]
    closure_valid: tuple[bool, ...]
    provisional: tuple[bool, ...]
    face: tuple[bool, ...]
    gaze_zone: tuple[str, ...]  # 그 프레임까지의 `DriverPipeline.gaze_zone`
    probs: tuple[ProbPoint, ...]
    events: tuple[EventRecord, ...]
    alerts: tuple[AlertRecord, ...]
    baseline: Baseline | None  # 입력이 끝났을 때의 기준선. 보정이 끝나기 전이면 None
    diagnostics: BaselineDiagnostics | None
    counters: dict[str, int]  # frames_accepted, dropped_frames, steps_emitted, model_calls, nonfinite_windows

    @property
    def duration_ms(self) -> int:
        """받아들인 마지막 프레임과 첫 프레임의 t_ms 차."""
        return self.t_ms[-1] - self.t_ms[0]


def run_pipeline(
    frames: Iterable[RawFrame],
    spec: PipelineSpec,
    rules: RulesConfig,
    model: ModelFn | None,
    speed_kmh: float | SpeedFn | None,
) -> DemoRun:
    """프레임을 `DriverPipeline`에 차례로 넣고 그림과 JSON에 필요한 것을 모은다.

    기준선은 넣어 주지 않는다(앱처럼 세션 앞부분에서 보정한다). 받아들인 프레임이 하나도 없으면 `DemoError`다.
    - speed_kmh: 세션 내내 같은 차량 속도(None이면 속도 억제 없음), 또는 프레임의 `t_ms`를 받아 그 프레임의 속도를
      돌려주는 함수(`SpeedFn`). 명령과 `run_demo`, `run_demo_raw`는 상수만 넘긴다.
    - 모델 출력(`DemoRun.probs`)은 모델을 부른 스텝마다 하나다(`PipelineOutput.step_probs`의 None이 아닌 원소).
      그래서 그 수는 `counters["model_calls"]`와 같다.
    - 경고(`DemoRun.alerts`)에는 종류가 붙는다(`AlertKinds`).
    """
    pipe = DriverPipeline(spec, rules, None, None, model)
    kinds = AlertKinds()
    t_ms: list[int] = []
    closure: list[float] = []
    closure_valid: list[bool] = []
    provisional: list[bool] = []
    face: list[bool] = []
    gaze_zone: list[str] = []
    probs: list[ProbPoint] = []
    events: list[EventRecord] = []
    alerts: list[AlertRecord] = []

    for frame in frames:
        speed = speed_kmh(frame.t_ms) if callable(speed_kmh) else speed_kmh
        out = pipe.update(frame, speed)
        if out is None:
            continue
        state = out.state
        t_ms.append(state.t_ms)
        closure.append(state.closure)
        closure_valid.append(state.closure_valid)
        provisional.append(state.provisional)
        face.append(state.face)
        gaze_zone.append(pipe.gaze_zone)
        for step, step_probs in zip(out.steps, out.step_probs):
            if step_probs is not None:
                probs.append(ProbPoint(state.t_ms, step.t_end_ms, step.k, step_probs))
        for event in out.events:
            events.append(EventRecord(state.t_ms, event, event in out.alerts))
        for alert, kind in zip(out.alerts, kinds.classify(out.events, out.alerts)):
            alerts.append(AlertRecord(state.t_ms, alert, kind))

    if len(t_ms) == 0:
        raise DemoError("the input holds no frame")
    return DemoRun(
        t_ms=tuple(t_ms),
        closure=tuple(closure),
        closure_valid=tuple(closure_valid),
        provisional=tuple(provisional),
        face=tuple(face),
        gaze_zone=tuple(gaze_zone),
        probs=tuple(probs),
        events=tuple(events),
        alerts=tuple(alerts),
        baseline=pipe.baseline,
        diagnostics=pipe.baseline_diagnostics,
        counters={
            "frames_accepted": pipe.frames_accepted,
            "dropped_frames": pipe.dropped_frames,
            "steps_emitted": pipe.steps_emitted,
            "model_calls": pipe.model_calls,
            "nonfinite_windows": pipe.nonfinite_windows,
        },
    )


def first_output_session_ms(baseline: Baseline, spec: PipelineSpec) -> int:
    """기준선이 완성된 세션에서 첫 모델 출력이 나오는 세션 시간(PLAN 4.4절). T_b = 60초면 239,000 ms다.

    조회 구간이 다 찬 첫 스텝의 끝은 T_b + 60초이고, 그런 스텝이 `window_steps`개 이어져야 한다.
    """
    features = spec.step_features
    return baseline.t_complete_ms + features.lookback_long_ms + (features.window_steps - 1) * spec.clock.step_ms


def _seconds(duration_ms: int) -> str:
    """시간을 초로 적는다. 10초 미만이고 정수 초가 아니면 소수 한 자리까지 적는다."""
    if duration_ms >= 10_000 or duration_ms % 1000 == 0:
        return f"{duration_ms / 1000.0:.0f} s"
    return f"{duration_ms / 1000.0:.1f} s"


def probability_gap_reason(run: DemoRun, spec: PipelineSpec) -> str | None:
    """모델 출력이 하나도 없는 이유(그림과 JSON에 적는 영어 문장). 출력이 있으면 None이다."""
    if len(run.probs) > 0:
        return None
    if run.baseline is None:
        return (
            f"No model output: the baseline was not completed within this input ({_seconds(run.duration_ms)} of "
            f"session time; calibration takes at least {_seconds(spec.baseline.duration_ms)})."
        )
    needed_ms = first_output_session_ms(run.baseline, spec)
    if run.duration_ms < needed_ms:
        features = spec.step_features
        return (
            f"No model output: this input covers {_seconds(run.duration_ms)} of session time and the first output "
            f"needs {_seconds(needed_ms)} (baseline {_seconds(run.baseline.t_complete_ms)} + "
            f"{_seconds(features.lookback_long_ms)} lookback + {features.window_steps} steps)."
        )
    if run.counters["nonfinite_windows"] > 0:
        return "No model output: every complete window held a non-finite feature value, so inference was skipped."
    return (
        f"No model output: there was no run of {spec.step_features.window_steps} consecutive full-lookback steps "
        f"(a frame gap longer than {_seconds(spec.clock.reset_gap_ms)} restarts the window)."
    )


# ---------------------------------------------------------------------------
# 그림에 쓸 구간
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EventSpan:
    """이벤트 줄에 그리는 구간 하나. `ended`가 False이면 입력이 끝날 때까지 `end`가 오지 않은 것이다."""

    type: str
    severity: int
    start_ms: int
    end_ms: int
    provisional: bool
    ended: bool


def event_spans(events: Sequence[EventRecord], last_t_ms: int) -> list[EventSpan]:
    """`onset`과 `end`를 짝지어 구간으로 만든다(모듈 설명의 가정 참조)."""
    open_onsets: dict[str, DriverStateEvent] = {}
    spans: list[EventSpan] = []
    for record in events:
        event = record.event
        if event.phase == PHASE_ONSET:
            open_onsets[event.type] = event
            continue
        end_ms = event.t_start_ms + event.duration_ms
        closing = [event.type]
        if event.type == EVENT_SLEEP:
            closing.append(EVENT_MICROSLEEP)  # 같은 감김의 microsleep은 end를 따로 내지 않는다
        for name in closing:
            onset = open_onsets.pop(name, None)
            if onset is not None:
                spans.append(EventSpan(name, onset.severity, onset.t_start_ms, end_ms, onset.provisional, True))
    for name, onset in open_onsets.items():
        end_ms = last_t_ms if last_t_ms > onset.t_start_ms else onset.t_start_ms
        spans.append(EventSpan(name, onset.severity, onset.t_start_ms, end_ms, onset.provisional, False))
    return spans


def zone_spans(
    t_ms: Sequence[int], zones: Sequence[str], max_hold_ms: int
) -> dict[str, list[tuple[int, int]]]:
    """전방 주시 표지별 시간 구간 `[시작, 끝)`. 키는 `ZONE_IN`, `ZONE_OUT`, `ZONE_UNKNOWN`이다.

    프레임 i의 표지는 다음 프레임까지, 최대 max_hold_ms 동안 유지된다(PLAN 3.3절). 그 뒤 다음 프레임까지의 시간은
    `ZONE_UNKNOWN`에 넣는다. 마지막 프레임은 시간을 대표하지 않는다. 이어진 같은 표지의 구간은 하나로 합친다.
    """
    spans: dict[str, list[tuple[int, int]]] = {ZONE_IN: [], ZONE_OUT: [], ZONE_UNKNOWN: []}

    def add(zone: str, start_ms: int, end_ms: int) -> None:
        if end_ms <= start_ms:
            return
        pieces = spans[zone]
        if len(pieces) > 0 and pieces[-1][1] == start_ms:
            pieces[-1] = (pieces[-1][0], end_ms)
        else:
            pieces.append((start_ms, end_ms))

    for i in range(len(t_ms) - 1):
        held_end_ms = min(t_ms[i + 1], t_ms[i] + max_hold_ms)
        add(zones[i], t_ms[i], held_end_ms)
        add(ZONE_UNKNOWN, held_end_ms, t_ms[i + 1])
    return spans


# ---------------------------------------------------------------------------
# 타임라인 그림
# ---------------------------------------------------------------------------

# 색. 신호는 파랑 하나, 심각도는 같은 색상의 밝기 단계(밝음 → 어두움), 경고 표식만 주황이다.
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"
_WASH = "#f0efec"  # 기준선 수집 구간의 바탕
_SIGNAL = "#2a78d6"
_SIGNAL_LIGHT = "#86b6ef"
_UNKNOWN = "#c3c2b7"
_SEVERITY_RAMP: tuple[str, ...] = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
_ALERT = "#eb6834"
_PROVISIONAL_HATCH = "////"
# 경고 표식: 색은 하나이고 모양이 종류다(색만으로 구분하지 않는다). 종류 -> (범례 이름, Line2D 속성)
_ALERT_MARKS: dict[str, tuple[str, dict[str, Any]]] = {
    ALERT_IMMEDIATE: (
        "alert",
        {"marker": "v", "markersize": 6.5, "markerfacecolor": _ALERT, "markeredgecolor": _SURFACE,
         "markeredgewidth": 1.2},
    ),
    ALERT_DEFERRED: (
        "deferred alert",
        {"marker": "v", "markersize": 6.0, "markerfacecolor": _SURFACE, "markeredgecolor": _ALERT,
         "markeredgewidth": 1.5},
    ),
    ALERT_REPEAT: (
        "repeat alert",
        {"marker": "|", "markersize": 8.0, "markerfacecolor": _ALERT, "markeredgecolor": _ALERT,
         "markeredgewidth": 1.8},
    ),
}  # fmt: skip

_RC: dict[str, Any] = {
    "font.family": "DejaVu Sans",
    "font.size": 9.0,
    "text.color": _INK,
    "figure.facecolor": _SURFACE,
    "savefig.facecolor": _SURFACE,
    "axes.facecolor": _SURFACE,
    "axes.edgecolor": _AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": _INK_SECONDARY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": _AXIS,
    "ytick.color": _AXIS,
    "xtick.labelcolor": _INK_SECONDARY,
    "ytick.labelcolor": _INK_SECONDARY,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "hatch.linewidth": 1.2,
    "legend.fontsize": 8.0,
}
_TRACK_HEIGHTS: tuple[float, ...] = (3.0, 0.9, 2.0, 3.3)
_EVENT_BAR_HALF = 0.22  # 이벤트 막대 높이의 절반(줄 간격이 1)
_ALERT_OFFSET = 0.36  # 경고 표식을 줄 가운데에서 위로 올리는 정도
_TICK_STEPS_S: tuple[int, ...] = (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600)
_MAX_TICKS = 12
_MIN_SPAN_FRACTION = 0.002  # 아주 짧은 구간도 보이도록 가로축 범위의 이 비율만큼은 그린다
_TITLE_CHARS = 110  # 그림 위 굵은 줄(12 pt)과 작은 줄(9 pt)에 들어가는 글자 수
_NOTE_CHARS = 170


def _severity_color(severity: int) -> str:
    return _SEVERITY_RAMP[min(max(severity, 0), len(_SEVERITY_RAMP) - 1)]


def _ascii(text: str) -> str:
    """그림에 적을 글자열. ASCII가 아닌 글자는 `\\uXXXX` 꼴로 바꾼다(DejaVu Sans에 없는 글리프를 피한다)."""
    return text.encode("ascii", "backslashreplace").decode("ascii")


def _clip(text: str, limit: int) -> str:
    """limit 글자를 넘는 줄은 끝을 `...`으로 줄인다(그림 밖으로 나가지 않게)."""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _format_clock(value: float, _position: int) -> str:
    """가로축 눈금: 세션 시간(초)을 `m:ss`(1시간 이상이면 `h:mm:ss`)로 적는다."""
    total = int(math.floor(value + 0.5))
    if total < 0:
        return ""
    hours, rest = divmod(total, 3600)
    minutes, seconds = divmod(rest, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def _tick_step_s(x_max_s: float) -> float:
    for step in _TICK_STEPS_S:
        if x_max_s / step <= _MAX_TICKS:
            return float(step)
    return float(_TICK_STEPS_S[-1] * math.ceil(x_max_s / (_MAX_TICKS * _TICK_STEPS_S[-1])))


def _bars(spans: Sequence[tuple[int, int]], origin_ms: int, min_width_s: float) -> list[tuple[float, float]]:
    """`[시작, 끝)` ms 구간을 `broken_barh`의 (시작 초, 너비 초)로 바꾼다."""
    bars: list[tuple[float, float]] = []
    for start_ms, end_ms in spans:
        width_s = (end_ms - start_ms) / 1000.0
        bars.append(((start_ms - origin_ms) / 1000.0, width_s if width_s > min_width_s else min_width_s))
    return bars


def _right_label(ax: Axes, value: float, text: str) -> None:
    """임계값 선의 이름을 그림 오른쪽 바깥, 선과 같은 높이에 적는다."""
    ax.annotate(
        text,
        xy=(1.0, value),
        xycoords=("axes fraction", "data"),
        xytext=(5.0, 0.0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=8.0,
        color=_INK_SECONDARY,
        annotation_clip=False,
    )


def _threshold_lines(ax: Axes, thresholds: Sequence[tuple[float, str]]) -> None:
    """임계값마다 점선과 이름을 그린다. 값이 같은 임계값은 선 하나에 이름을 함께 적는다."""
    names_by_value: dict[float, list[str]] = {}
    for value, name in thresholds:
        names_by_value.setdefault(value, []).append(name)
    for value, names in names_by_value.items():
        ax.axhline(value, color=_INK_MUTED, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1.5)
        _right_label(ax, value, f"{value:.2f} {' / '.join(names)}")


def _legend(ax: Axes, handles: Sequence[Any]) -> None:
    """범례를 그림 위 오른쪽, 제목과 같은 줄에 가로로 놓는다(자료를 가리지 않는다)."""
    ax.legend(
        handles=list(handles),
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=len(handles),
        frameon=False,
        borderaxespad=0.25,
        handlelength=1.5,
        handletextpad=0.5,
        columnspacing=1.3,
        labelcolor=_INK_SECONDARY,
    )


def _closure_series(
    run: DemoRun, max_hold_ms: int
) -> tuple[np.ndarray, np.ma.MaskedArray, np.ma.MaskedArray]:
    """closure 선의 (세션 시간 초, 확정 closure, 임시 closure). 유효하지 않은 프레임은 가려서 선이 끊기게 한다.

    프레임 간격이 max_hold_ms를 넘는 곳에는 가린 점을 하나 끼워, 자료가 없는 시간을 선이 가로지르지 않게 한다.
    """
    t = np.asarray(run.t_ms, dtype=np.int64)
    values = np.asarray(run.closure, dtype=np.float64)
    hidden = ~np.asarray(run.closure_valid, dtype=bool)
    provisional = np.asarray(run.provisional, dtype=bool)
    gaps = np.flatnonzero(np.diff(t) > max_hold_ms)
    x = np.insert((t - t[0]) / 1000.0, gaps + 1, (t[gaps] + max_hold_ms - t[0]) / 1000.0)
    values = np.insert(values, gaps + 1, 0.0)
    hidden = np.insert(hidden, gaps + 1, True)
    provisional = np.insert(provisional, gaps + 1, provisional[gaps])
    return (
        x,
        np.ma.masked_array(values, mask=hidden | provisional),
        np.ma.masked_array(values, mask=hidden | ~provisional),
    )


def _draw_closure(ax: Axes, run: DemoRun, spec: PipelineSpec, rules: RulesConfig) -> None:
    x, final, provisional = _closure_series(run, spec.clock.max_hold_ms)
    ax.plot(x, provisional, color=_SIGNAL_LIGHT, linewidth=1.0, solid_joinstyle="round", zorder=2)
    ax.plot(x, final, color=_SIGNAL, linewidth=1.0, solid_joinstyle="round", zorder=2.5)
    detector = rules.closure.detector
    _threshold_lines(
        ax,
        (
            (spec.step_features.perclos_thr, "PERCLOS"),
            (detector.enter_thr, "rule enter"),
            (detector.exit_thr, "rule exit"),
            (spec.run_detector.enter_thr, "blink enter"),
            (spec.run_detector.exit_thr, "blink exit"),
        ),
    )
    ax.set_ylim(-0.03, 1.05)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("Eye closure (0 = open, 1 = closed)", loc="left", fontsize=10.0, pad=6.0)
    _legend(
        ax,
        (
            Line2D([], [], color=_SIGNAL, linewidth=1.6, label="closure"),
            Line2D([], [], color=_SIGNAL_LIGHT, linewidth=1.6, label="provisional (before baseline)"),
            Patch(facecolor=_WASH, edgecolor=_AXIS, linewidth=0.6, label="baseline window"),
        ),
    )


def _draw_gaze(ax: Axes, run: DemoRun, spec: PipelineSpec, min_width_s: float) -> None:
    spans = zone_spans(run.t_ms, run.gaze_zone, spec.clock.max_hold_ms)
    origin_ms = run.t_ms[0]
    ax.broken_barh(_bars(spans[ZONE_OUT], origin_ms, min_width_s), (0.7, 0.6), facecolors=_SIGNAL, linewidth=0)
    ax.broken_barh(_bars(spans[ZONE_UNKNOWN], origin_ms, 0.0), (-0.3, 0.6), facecolors=_UNKNOWN, linewidth=0)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([1.0, 0.0])
    ax.set_yticklabels(["away", "unknown / no data"])
    ax.tick_params(axis="y", length=0)
    ax.set_title("Gaze zone (blank = forward)", loc="left", fontsize=10.0, pad=6.0)


def _draw_probabilities(
    ax: Axes, run: DemoRun, rules: RulesConfig, title: str, empty_reason: str | None
) -> None:
    origin_ms = run.t_ms[0]
    # 스텝 번호가 이어진 구간마다 선을 따로 그린다(끊긴 곳을 선이 가로지르지 않게).
    start = 0
    for index in range(1, len(run.probs) + 1):
        if index < len(run.probs) and run.probs[index].k == run.probs[index - 1].k + 1:
            continue
        piece = run.probs[start:index]
        start = index
        x = [(point.t_end_ms - origin_ms) / 1000.0 for point in piece]
        for class_index, color, zorder in ((CLASS_LOW_VIGILANT, _SIGNAL_LIGHT, 2.0), (CLASS_DROWSY, _SIGNAL, 2.5)):
            y = [point.probs[class_index] for point in piece]
            # 점 하나짜리 구간은 선이 보이지 않으므로 표식을 찍는다
            marker = "o" if len(piece) == 1 else None
            ax.plot(x, y, color=color, linewidth=1.4, marker=marker, markersize=4.0, zorder=zorder)
    model_rule = rules.drowsiness.model
    _threshold_lines(ax, ((model_rule.onset_p, "drowsy onset"), (model_rule.end_p, "drowsy end")))
    if empty_reason is not None:
        ax.text(
            0.5,
            0.5,
            textwrap.fill(empty_reason, 90),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.5,
            color=_INK_SECONDARY,
            bbox={"facecolor": _SURFACE, "edgecolor": "none", "pad": 4.0},
            zorder=4,
        )
    ax.set_ylim(-0.03, 1.05)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_title(title, loc="left", fontsize=10.0, pad=6.0)
    _legend(
        ax,
        (
            Line2D([], [], color=_SIGNAL, linewidth=1.6, label="p_drowsy"),
            Line2D([], [], color=_SIGNAL_LIGHT, linewidth=1.6, label="p_low_vigilant"),
        ),
    )


def _draw_events(ax: Axes, run: DemoRun, rules: RulesConfig, min_width_s: float) -> None:
    origin_ms = run.t_ms[0]
    rows = {name: row for row, name in enumerate(EVENT_TYPES)}
    for span in event_spans(run.events, run.t_ms[-1]):
        bars = _bars([(span.start_ms, span.end_ms)], origin_ms, min_width_s)
        ax.broken_barh(
            bars,
            (rows[span.type] - _EVENT_BAR_HALF, 2.0 * _EVENT_BAR_HALF),
            facecolors=_severity_color(span.severity),
            edgecolors=_SURFACE,
            linewidth=0,
            hatch=_PROVISIONAL_HATCH if span.provisional else None,
            zorder=2,
        )
    for kind in ALERT_KINDS:  # 종류마다 표식 모양이 다르다. 그 종류의 경고가 없으면 그리지 않는다
        marked = [alert for alert in run.alerts if alert.kind == kind]
        if len(marked) > 0:
            ax.plot(
                [(alert.t_ms - origin_ms) / 1000.0 for alert in marked],
                # 짧은 구간을 가리지 않도록 표식은 막대 바로 위에 둔다(세로축이 뒤집혀 있어 빼면 위쪽이다)
                [rows[alert.event.type] - _ALERT_OFFSET for alert in marked],
                linestyle="none",
                zorder=3,
                **_ALERT_MARKS[kind][1],
            )
    ax.set_ylim(len(EVENT_TYPES) - 0.5, -0.5)
    ax.set_yticks(list(rows.values()))
    ax.set_yticklabels(list(rows.keys()))
    ax.tick_params(axis="y", length=0)
    # 제목은 짧게 둔다: 같은 줄 오른쪽의 범례(심각도 넷, 임시 판정, 경고 세 종류)가 축 너비의 3/4쯤을 쓴다
    ax.set_title("Rule events (bar = state duration)", loc="left", fontsize=10.0, pad=6.0)
    severities = sorted({rules.events.get(name).severity for name in EVENT_TYPES})
    handles: list[Any] = [
        Patch(facecolor=_severity_color(severity), linewidth=0, label=f"severity {severity}")
        for severity in severities
    ]
    handles.append(
        Patch(facecolor=_UNKNOWN, edgecolor=_SURFACE, linewidth=0, hatch=_PROVISIONAL_HATCH, label="provisional")
    )
    for kind in ALERT_KINDS:  # 범례에는 세 종류를 늘 싣는다
        label, style = _ALERT_MARKS[kind]
        handles.append(Line2D([], [], linestyle="none", label=label, **style))
    _legend(ax, handles)


def render_timeline(
    run: DemoRun,
    spec: PipelineSpec,
    rules: RulesConfig,
    titles: Sequence[str],
    notes: Sequence[str],
    probability_title: str,
    empty_reason: str | None,
) -> Figure:
    """네 줄 타임라인을 그린 `Figure`(Agg 캔버스)를 돌려준다. 축 순서는 closure, 전방 주시, 모델 출력, 규칙 이벤트다.

    titles는 그림 위에 굵게 적는 줄(합성 입력이면 호출자가 합성 데이터 표시를 첫 줄에 둔다), notes는 그 아래에
    작게 적는 줄이다. 굵은 줄 둘과 작은 줄 셋(굵은 줄이 하나면 작은 줄 넷)까지 들어가고, 그림 너비를 넘는 줄은 끝을
    줄여 적는다.
    pyplot을 쓰지 않으므로 전역 백엔드와 전역 그림 상태를 건드리지 않는다.
    """
    duration_s = run.duration_ms / 1000.0
    x_max_s = duration_s if duration_s > 1.0 else 1.0
    min_width_s = _MIN_SPAN_FRACTION * x_max_s
    with matplotlib.rc_context(_RC):
        fig = Figure(figsize=FIGURE_SIZE_IN)
        FigureCanvasAgg(fig)
        grid = fig.add_gridspec(
            len(_TRACK_HEIGHTS), 1, height_ratios=_TRACK_HEIGHTS, left=0.115, right=0.845, top=0.822, bottom=0.06,
            hspace=0.5,
        )  # fmt: skip
        ax_closure = fig.add_subplot(grid[0])
        ax_gaze = fig.add_subplot(grid[1], sharex=ax_closure)
        ax_probs = fig.add_subplot(grid[2], sharex=ax_closure)
        ax_events = fig.add_subplot(grid[3], sharex=ax_closure)
        axes = (ax_closure, ax_gaze, ax_probs, ax_events)

        _draw_closure(ax_closure, run, spec, rules)
        _draw_gaze(ax_gaze, run, spec, min_width_s)
        _draw_probabilities(ax_probs, run, rules, probability_title, empty_reason)
        _draw_events(ax_events, run, rules, min_width_s)

        # 기준선 수집 구간: 완성됐으면 T_b까지, 아니면 입력 전체
        wash_end_s = x_max_s if run.baseline is None else run.baseline.t_complete_ms / 1000.0
        for ax in axes:
            ax.axvspan(0.0, min(wash_end_s, x_max_s), facecolor=_WASH, linewidth=0, zorder=0)
            ax.grid(axis="x", color=_GRID, linewidth=0.6, zorder=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(axis="x", length=3.0)
        for ax in axes[:-1]:
            ax.tick_params(axis="x", labelbottom=False)
        ax_events.set_xlim(0.0, x_max_s)
        ax_events.xaxis.set_major_locator(MultipleLocator(_tick_step_s(x_max_s)))
        ax_events.xaxis.set_major_formatter(FuncFormatter(_format_clock))
        ax_events.set_xlabel("session time (m:ss from the first frame)")

        y = 0.985
        for line, bold in [(line, True) for line in titles] + [(line, False) for line in notes]:
            fig.text(
                0.012,
                y,
                _clip(_ascii(line), _TITLE_CHARS if bold else _NOTE_CHARS),
                ha="left",
                va="top",
                fontsize=12.0 if bold else 9.0,
                fontweight="bold" if bold else "normal",
                color=_INK if bold else _INK_SECONDARY,
            )
            y -= 0.032 if bold else 0.024
        fig.text(
            0.988,
            0.012,
            "Assistive guidance only, not a safety system.",
            ha="right",
            va="bottom",
            fontsize=8.0,
            color=_INK_MUTED,
        )
    return fig


def save_timeline(fig: Figure, png_path: Path, dpi: int) -> None:
    """그림을 PNG로 쓴다. 여백을 잘라 내지 않으므로 크기는 항상 `FIGURE_SIZE_IN` x dpi 픽셀이다."""
    with matplotlib.rc_context(_RC):
        fig.savefig(png_path, dpi=dpi, format="png")


# ---------------------------------------------------------------------------
# 이벤트 JSON
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """입력 하나의 정보. `details`는 종류별 부가 정보이고 JSON의 `source`에 그대로 들어간다."""

    kind: str  # SOURCE_VIDEO 또는 SOURCE_RAW
    path: Path
    synthetic: bool
    details: dict[str, Any]


def _baseline_summary(baseline: Baseline | None, diagnostics: BaselineDiagnostics | None) -> dict[str, Any] | None:
    if baseline is None:
        return None
    yaw0, pitch0, roll0 = signals.euler_from_rotation(baseline.r0)
    summary: dict[str, Any] = dataclasses.asdict(baseline)
    summary["r0"] = list(baseline.r0)
    summary["yaw0_deg"] = yaw0
    summary["pitch0_deg"] = pitch0
    summary["roll0_deg"] = roll0
    summary["diagnostics"] = None
    if diagnostics is not None:
        summary["diagnostics"] = {
            "n_face_frames": diagnostics.n_face_frames,
            "face_ratio": diagnostics.face_ratio,
            "blink_l_quantiles": list(diagnostics.blink_l_quantiles),
            "blink_r_quantiles": list(diagnostics.blink_r_quantiles),
        }
    return summary


def build_report(
    run: DemoRun,
    source: SourceInfo,
    model: ModelInfo,
    cfg: ExperimentConfig,
    feature_spec: FeatureSpec,
    rules: RulesConfig,
    speed_kmh: float | None,
    empty_reason: str | None,
) -> dict[str, Any]:
    """이벤트 JSON의 내용(모듈 설명의 형식)."""
    return {
        "note": SYNTHETIC_NOTE if source.synthetic else NOTE_RECORDING,
        "source": {
            "kind": source.kind,
            "path": str(source.path),
            "synthetic": source.synthetic,
            **source.details,
        },
        "model": {
            "file": str(model.path),
            "sha256": model.sha256,
            "feature_spec_copy": str(model.spec_copy),
            "feature_spec_hash": model.feature_spec_hash,
            "metadata": dict(model.metadata),
        },
        "spec": {
            "feature_spec": str(cfg.paths.feature_spec),
            "feature_spec_sha256": file_sha256(cfg.paths.feature_spec),
            "pipeline_hash": feature_spec.pipeline_hash,
            "rules": str(cfg.paths.rules),
            "rules_hash": rules_hash(rules),
        },
        "drowsiness_source": rules.drowsiness.source,
        "speed_kmh": speed_kmh,
        "session": {
            "first_t_ms": run.t_ms[0],
            "last_t_ms": run.t_ms[-1],
            "duration_ms": run.duration_ms,
        },
        "baseline": _baseline_summary(run.baseline, run.diagnostics),
        "counters": {
            **run.counters,
            "face_frames": sum(run.face),
            "events": len(run.events),
            "alerts": len(run.alerts),
            **{f"alerts_{kind}": sum(1 for record in run.alerts if record.kind == kind) for kind in ALERT_KINDS},
        },
        "probabilities": {
            "n_outputs": len(run.probs),
            "first_t_end_ms": run.probs[0].t_end_ms if len(run.probs) > 0 else None,
            "last_t_end_ms": run.probs[-1].t_end_ms if len(run.probs) > 0 else None,
            "empty_reason": empty_reason,
        },
        "events": [
            {"t_ms": record.t_ms, **dataclasses.asdict(record.event), "alert": record.alert} for record in run.events
        ],
        "alerts": [
            {"t_ms": record.t_ms, **dataclasses.asdict(record.event), "kind": record.kind} for record in run.alerts
        ],
    }


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------


def _header_lines(
    run: DemoRun, source: SourceInfo, model: ModelInfo, rules: RulesConfig, speed_kmh: float | None
) -> tuple[list[str], list[str]]:
    """그림 위에 적는 (굵은 줄, 작은 줄). 합성 입력이면 굵은 줄의 첫 줄이 합성 데이터 표시다.

    영상 입력이면 작은 줄이 하나 더 있다(영상을 세운 회전). 영상 입력은 합성일 수 없어 굵은 줄이 하나뿐이므로
    전체 높이는 합성 입력의 굵은 줄 둘과 작은 줄 셋을 넘지 않는다. parquet 입력은 추출 때의 자동 점검이 각도를
    가르지 못했다고 적혀 있을 때만(합성 입력이 아닐 때) 같은 자리에 한 줄이 붙는다.
    """
    titles: list[str] = []
    if source.synthetic:
        titles.append(SYNTHETIC_NOTE)
    titles.append(f"Driver state timeline: {source.path.name}")

    version = model.metadata["model_version"]
    trained_on = model.metadata.get("trained_on", "unknown")
    drowsy_source = rules.drowsiness.source
    if drowsy_source == DROWSY_SOURCE_PERCLOS:
        drowsy_source += " (the model output is shown but not used by the drowsy rule)"
    if run.baseline is None:
        baseline = "not completed (closure stays provisional, from global constants)"
    else:
        baseline = f"{run.baseline.status}, complete at {_seconds(run.baseline.t_complete_ms)}"
    speed = "not given (alerts are not speed-gated)" if speed_kmh is None else f"{speed_kmh:g} km/h (constant)"
    notes = [
        f"model: {model.path.name} (version {version}, trained_on: {trained_on})",
        f"drowsiness.source: {drowsy_source}",
        f"baseline: {baseline}   |   speed: {speed}",
    ]
    details = source.details
    if source.kind == SOURCE_VIDEO and "rotation_source" in details:  # 영상을 세운 회전과 그 출처를 넷째 줄에 적는다
        if details["rotation_source"] == extract.SOURCE_MANUAL:
            how = "given with --rotation"
        elif details["rotation_ambiguous"]:
            how = "automatic check, AMBIGUOUS: look at the video and pass --rotation"
        else:
            how = "automatic check"
        notes.append(f"video rotation: {details['rotation']} degrees counter-clockwise ({how})")
    elif source.kind == SOURCE_RAW and details.get("rotation_ambiguous") and not source.synthetic:
        # parquet 입력: 추출 때의 자동 점검이 각도를 가르지 못했다는 기록이 있을 때만 적는다
        angle = details.get("rotation")
        turned = "angle not recorded" if angle is None else f"{angle} degrees counter-clockwise"
        notes.append(
            f"video rotation: {turned} (automatic check at extraction, AMBIGUOUS: look at the video and extract it "
            "again with a manual rotation)"
        )
    return titles, notes


def _probability_title(model: ModelInfo) -> str:
    if model.metadata.get("trained_on") == export_onnx.TRAINED_ON_NONE:
        return "Model output (random weights: the values carry no meaning)"
    return "Model output"


def _demo(
    frames: Iterable[RawFrame],
    describe: Callable[[], SourceInfo],
    onnx_path: Path,
    out_dir: Path,
    speed_kmh: float | None,
    cfg: ExperimentConfig,
    feature_spec: FeatureSpec,
) -> tuple[Path, Path]:
    """검증한 모델로 파이프라인을 돌리고 두 파일을 쓴다. describe()는 프레임을 다 읽은 뒤 `SourceInfo`를 돌려준다."""
    if speed_kmh is not None and not (math.isfinite(speed_kmh) and speed_kmh >= 0.0):
        raise DemoError(f"speed must be a finite value >= 0 km/h, got {speed_kmh}")
    spec = feature_spec.pipeline
    rules = load_rules(cfg.paths.rules)
    model = verify_model(Path(onnx_path), feature_spec)
    model_fn = make_model_fn(model.path, spec.step_features.window_steps, len(spec.step_features.names))

    run = run_pipeline(frames, spec, rules, model_fn, speed_kmh)
    source = describe()
    empty_reason = probability_gap_reason(run, spec)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{source.path.stem}_timeline.png"
    json_path = out_dir / f"{source.path.stem}_events.json"

    titles, notes = _header_lines(run, source, model, rules, speed_kmh)
    fig = render_timeline(run, spec, rules, titles, notes, _probability_title(model), empty_reason)
    save_timeline(fig, png_path, cfg.demo.figure_dpi)
    report = build_report(run, source, model, cfg, feature_spec, rules, speed_kmh, empty_reason)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return png_path, json_path


@dataclass(frozen=True, slots=True)
class VideoRotation:
    """영상 입력을 세우는 데 쓴 회전과 그 출처(`resolve_video_rotation`의 결과). 각도는 `frame.rotation`과 같은 규약이다."""

    rotation: int  # 적용할 회전(세우려면 반시계로 돌려야 하는 각도, 0·90·180·270)
    source: str  # `extract.SOURCE_MANUAL`(호출자가 준 값) 또는 `extract.SOURCE_AUTO`(자동 점검이 고른 값)
    ambiguous: bool  # 자동 점검이 각도를 가르지 못했는지. 수동 값이면 False
    check: extract.RotationCheck | None  # 자동 점검의 결과. 수동 값이면 None
    record: dict[str, Any] | None  # 점검 결과를 추출 메타데이터의 `rotation_check`와 같은 꼴로 적은 것. 수동 값이면 None


def resolve_video_rotation(
    video: Path, cfg: ExperimentConfig, spec: PipelineSpec, rotation: int | None
) -> VideoRotation:
    """영상을 세울 회전을 정한다. rotation을 주면 그 값이고, None이면 추출과 같은 자동 점검을 한 번 돌린다.

    자동 점검은 `extract.check_rotation`을 설정의 `extraction.rotation_check_frames`, `rotation_check_interval_ms`,
    `rotation_roll_margin_deg`로 부른다(`paths.model_path`의 face_landmarker.task가 필요하다). 표시 행렬에 뒤집기가 든
    영상은 rotation을 주지 않으면 `VideoError`다.
    """
    if rotation is not None:
        return VideoRotation(normalize_rotation(rotation), extract.SOURCE_MANUAL, False, None, None)
    settings = extract.RotationCheckSettings(
        cfg.extraction.rotation_check_frames,
        cfg.extraction.rotation_check_interval_ms,
        cfg.extraction.rotation_roll_margin_deg,
    )
    check = extract.check_rotation(
        video, cfg.paths.model_path, spec, settings.frames, settings.interval_ms, settings.roll_margin_deg
    )
    record = extract.rotation_check_record(check, settings)
    return VideoRotation(check.rotation, extract.SOURCE_AUTO, check.ambiguous, check, record)


def describe_video_rotation(chosen: VideoRotation) -> str:
    """영상을 세운 회전을 알리는 한 줄(명령의 출력용, 영어): 각도, 출처, 가르지 못했는지, 점검이 잰 값."""
    if chosen.check is None:
        return f"rotation: {chosen.rotation} (given with --rotation; the automatic check was not run)"
    text = (
        f"rotation: auto check chose {chosen.rotation}, ambiguous: {'yes' if chosen.ambiguous else 'no'} "
        f"({extract.rotation_check_details(chosen.check)})"
    )
    if chosen.ambiguous:
        why = "the check could not tell this angle from another one (usually the opposite one)"
        if chosen.check.reason == extract.REASON_NO_FACE:
            why = "no face was found at any angle in the sampled frames, so the rotation metadata is used unverified"
        text += f" WARNING rotation ambiguous: {why}. Look at the video and run again with --rotation {{0,90,180,270}}"
    elif chosen.rotation != chosen.check.metadata_rotation:
        text += " WARNING rotation differs from the video metadata"
    return text


def _video_frames(
    video: Path, model_path: Path, spec: PipelineSpec, counts: DecodeCounts, rotation: int
) -> Iterator[RawFrame]:
    """영상을 rotation만큼 돌려 세워 VIDEO 모드 FaceLandmarker에 넣고 `RawFrame`을 하나씩 낸다. 프레임은 어디에도 저장하지 않는다."""
    landmarker = extract.make_landmarker(model_path, spec)
    try:
        for t_ms, rgb in iter_video_frames(video, rotation=rotation, counts=counts):
            yield extract.detect_frame(landmarker, t_ms, rgb, spec)
    finally:
        landmarker.close()


def run_demo(
    video: Path,
    onnx_path: Path,
    out_dir: Path,
    speed_kmh: float | None = None,
    *,
    cfg: ExperimentConfig | None = None,
    rotation: int | None = None,
    on_rotation: Callable[[VideoRotation], None] | None = None,
) -> tuple[Path, Path]:
    """영상 하나를 MediaPipe와 전체 파이프라인에 통과시켜 (타임라인 PNG 경로, 이벤트 JSON 경로)를 돌려준다.

    - 기준선은 영상 자신의 앞 60초다(부족하면 PLAN 1.2절의 결정 D5대로 30초씩 연장하거나 전역 상수로 대체한다).
    - onnx_path: 내보낸 모델. 메타데이터가 spec 파일들과 맞지 않으면 `DemoError`로 거부한다.
    - speed_kmh: 경고의 속도 억제(PLAN 1.2절의 결정 D3: 시선·감김 경고는 20 km/h, 졸음 경고는 50 km/h 미만에서 억제)에
      쓰는 차량 속도. None이면 억제하지 않는다.
    - cfg: 실험 설정. None이면 실행 위치의 `config/default.yaml`을 읽는다(저장소 루트에서 실행).
      `paths.feature_spec`, `paths.rules`, `paths.model_path`(face_landmarker.task), `demo.figure_dpi`, 그리고 회전
      자동 점검의 설정(`extraction.rotation_check_*`, `rotation_roll_margin_deg`)을 쓴다.
    - rotation: 영상을 세우려면 반시계로 돌려야 하는 각도(90의 배수). None이면 자동 점검으로 정한다
      (`resolve_video_rotation`). 표시 행렬에 뒤집기가 든 영상은 rotation을 주지 않으면 `VideoError`다.
    - on_rotation: 회전이 정해지면(모델 검증이 끝나고 프레임을 읽기 직전) 한 번 불린다. 명령은 이것으로 고른 각도를
      출력한다(`describe_video_rotation`).
    이벤트 JSON의 `source`에 적용한 회전(`rotation`), 출처(`rotation_source`), 가르지 못했는지(`rotation_ambiguous`),
    점검이 잰 값(`rotation_check`), 표시 행렬의 뒤집기(`display_flip`)를 적는다.
    """
    video = Path(video)
    if cfg is None:
        cfg = load_config(DEFAULT_CONFIG_PATH)
    if not video.is_file():
        raise DemoError(f"video not found: {video}")
    manual = None if rotation is None else normalize_rotation(rotation)
    feature_spec = load_feature_spec(cfg.paths.feature_spec)
    counts = DecodeCounts()
    chosen: list[VideoRotation] = []  # 프레임을 읽기 시작할 때 한 번 채운다

    def frames() -> Iterator[RawFrame]:
        chosen.append(resolve_video_rotation(video, cfg, feature_spec.pipeline, manual))
        if on_rotation is not None:
            on_rotation(chosen[0])
        yield from _video_frames(video, cfg.paths.model_path, feature_spec.pipeline, counts, chosen[0].rotation)

    def describe() -> SourceInfo:
        return SourceInfo(
            kind=SOURCE_VIDEO,
            path=video,
            synthetic=False,
            details={
                "n_decoded": counts.decoded,
                "n_skipped": counts.skipped,
                "rotation": counts.rotation,
                "rotation_source": chosen[0].source,
                "rotation_ambiguous": chosen[0].ambiguous,
                "rotation_check": chosen[0].record,
                "display_flip": counts.display_flip,
                "mediapipe_version": extract.mediapipe_version(),
                "mediapipe_model_sha256": feature_spec.pipeline.mediapipe.model_sha256,
            },
        )

    return _demo(frames(), describe, onnx_path, out_dir, speed_kmh, cfg, feature_spec)


@dataclass(frozen=True, slots=True)
class RawRotation:
    """추출 결과(parquet)의 메타데이터에 적힌 회전 정보(`read_raw_rotation`의 결과). 데모는 parquet을 돌리지 않는다."""

    rotation: int | None  # 추출에 적용된 회전(반시계, 0·90·180·270). 파일에 적혀 있지 않으면 None
    # `extract.SOURCE_AUTO`(자동 점검 기록이 있다), `extract.SOURCE_MANUAL`(기록 없이 수동 값으로 추출됐다),
    # `extract.SOURCE_METADATA`(기록도 수동 값도 없고 적용된 회전만 적혀 있다), None(회전이 적혀 있지 않다)
    source: str | None
    ambiguous: bool  # 점검 기록이 각도를 가르지 못했다고 하는지. 기록이 없으면 False
    record: dict[str, Any] | None  # 점검 기록(`rotation_check`) 그대로. 없으면 None
    note: str  # 명령이 출력하고 JSON의 `source.rotation_note`에 적는 한 줄(영어)


def _recorded_angle(value: Any) -> int | None:
    return value % 360 if isinstance(value, int) and not isinstance(value, bool) else None


def _check_from_record(rotation: int, record: Mapping[str, Any]) -> extract.RotationCheck | None:
    """점검 기록(`extract.rotation_check_record`의 꼴)을 `RotationCheck`로 되돌린다. 빠진 값이 있으면 None."""
    try:
        rates = {angle: float(record["rates"][str(angle)]) for angle in extract.ROTATION_CANDIDATES}
        rolls = {
            angle: None if record["rolls"][str(angle)] is None else float(record["rolls"][str(angle)])
            for angle in extract.ROTATION_CANDIDATES
        }
        return extract.RotationCheck(
            rotation=rotation,
            metadata_rotation=int(record["metadata_rotation"]),
            rates=rates,
            rolls=rolls,
            n_frames=int(record["n_frames"]),
            ambiguous=bool(record.get("ambiguous")),
            reason=str(record["reason"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def read_raw_rotation(metadata: Mapping[str, Any]) -> RawRotation:
    """추출 결과의 메타데이터(`extract.read_extract_metadata`)에서 회전 기록을 읽어 알릴 한 줄을 만든다.

    위에서부터 처음 맞는 것:
    1. 합성 표시(`synthetic: true`)가 있다: 영상이 없는 세션이다. 점검 기록이 없다고 적는다.
    2. 점검 기록(`rotation_check`)이 있다: 자동 점검이 고른 각도, 가르지 못했는지, 점검이 잰 값. 가르지 못했으면
       WARNING과 할 일(각도를 손으로 줘서 다시 추출)을 붙인다. 가르지는 못하지 않았지만 고른 각도가 영상의 회전
       메타데이터와 다르면 `--video` 입력과 같은 WARNING을 붙인다.
    3. 기록이 없고 `rotation_override`에 각도가 있다: 수동 값(보정표나 `extract --video --rotation`)으로 추출됐다.
       `--video`에 `--rotation`을 줬을 때처럼 각도와 수동 값이라는 것만 적는다.
    4. 기록도 수동 값도 없다: 기록이 없다고 적는다. 적용된 회전이 적혀 있으면(영상의 회전 메타데이터만 믿고 추출한 전의
       결과) 그 각도와, 점검하려면 다시 추출하라는 말을 함께 적는다.
    """
    applied = _recorded_angle(metadata.get("rotation"))
    override = _recorded_angle(metadata.get("rotation_override"))
    found = metadata.get(extract.ROTATION_CHECK_KEY)
    if metadata.get("synthetic") is True:
        note = "rotation: no rotation-check record (synthetic session: there was no video to turn upright)"
        return RawRotation(None, None, False, None, note)
    if isinstance(found, dict):
        angle = applied if applied is not None else override
        ambiguous = bool(found.get("ambiguous"))
        chosen = "an unrecorded angle" if angle is None else str(angle)
        check = None if angle is None else _check_from_record(angle, found)
        details = f"reason {found.get('reason') or 'not recorded'}" if check is None else extract.rotation_check_details(check)
        note = f"rotation: the extraction's auto check chose {chosen}, ambiguous: {'yes' if ambiguous else 'no'} ({details})"
        if ambiguous:
            why = "the check could not tell this angle from another one (usually the opposite one)"
            if found.get("reason") == extract.REASON_NO_FACE:
                why = "no face was found at any angle in the sampled frames, so the rotation metadata was used unverified"
            note += (
                f" WARNING rotation ambiguous: {why}. Look at the video and extract it again with the angle given by "
                "hand (python -m drowsy.extract --video ... --rotation {0,90,180,270}, or the manual rotation table "
                "of the config for the other inputs)"
            )
        elif check is not None and check.rotation != check.metadata_rotation:
            note += " WARNING rotation differs from the video metadata"
        return RawRotation(angle, extract.SOURCE_AUTO, ambiguous, dict(found), note)
    if override is not None:
        angle = applied if applied is not None else override
        note = f"rotation: {angle} (manual value recorded by the extraction; the automatic check was not run)"
        return RawRotation(angle, extract.SOURCE_MANUAL, False, None, note)
    if applied is not None:
        note = (
            f"rotation: {applied} from the video's rotation metadata, no rotation-check record (the extraction did "
            "not check the angle; extract the video again to run the check)"
        )
        return RawRotation(applied, extract.SOURCE_METADATA, False, None, note)
    note = "rotation: no rotation-check record (the file does not say how the picture was turned upright)"
    return RawRotation(None, None, False, None, note)


def run_demo_raw(
    raw_parquet: Path,
    onnx_path: Path,
    out_dir: Path,
    speed_kmh: float | None = None,
    *,
    cfg: ExperimentConfig | None = None,
) -> tuple[Path, Path]:
    """`run_demo`와 같지만 영상 대신 RawFrame parquet(추출 결과나 `drowsy.synth`의 세션)을 읽는다. MediaPipe가 필요 없다.

    parquet은 추출할 때 이미 세운 결과라서 돌리지 않는다. 추출이 메타데이터에 남긴 회전 기록을 읽어(`read_raw_rotation`)
    이벤트 JSON의 `source`에 `rotation`, `rotation_source`, `rotation_ambiguous`, `rotation_check`, `rotation_note`로 적는다.
    명령은 `rotation_note`를 한 줄로 출력한다. 자동 점검이 각도를 가르지 못했다는 기록이 있으면 그림의 머리말에도 적는다.
    """
    raw_parquet = Path(raw_parquet)
    if cfg is None:
        cfg = load_config(DEFAULT_CONFIG_PATH)
    if not raw_parquet.is_file():
        raise DemoError(f"raw parquet not found: {raw_parquet}")
    feature_spec = load_feature_spec(cfg.paths.feature_spec)
    metadata = extract.read_extract_metadata(raw_parquet)
    expected_sha = feature_spec.pipeline.mediapipe.model_sha256
    recorded_sha = metadata.get("model_sha256")
    if recorded_sha is not None and recorded_sha != expected_sha:
        raise DemoError(
            f"{raw_parquet} was extracted with MediaPipe model {recorded_sha} but the feature spec expects "
            f"{expected_sha}. Fix: extract the video again with the expected face_landmarker.task."
        )
    recorded_rotation = read_raw_rotation(metadata)

    def describe() -> SourceInfo:
        return SourceInfo(
            kind=SOURCE_RAW,
            path=raw_parquet,
            synthetic=metadata.get("synthetic") is True,
            details={
                "extract_source": metadata.get("source"),
                "mediapipe_version": metadata.get("mediapipe_version"),
                "mediapipe_model_sha256": recorded_sha,
                "drop_sim": metadata.get("drop_sim"),
                "rotation": recorded_rotation.rotation,
                "rotation_source": recorded_rotation.source,
                "rotation_ambiguous": recorded_rotation.ambiguous,
                "rotation_check": recorded_rotation.record,
                "rotation_note": recorded_rotation.note,
            },
        )

    return _demo(
        extract.read_raw_frames(raw_parquet), describe, onnx_path, out_dir, speed_kmh, cfg, feature_spec
    )


# ---------------------------------------------------------------------------
# 명령줄
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m drowsy.demo`. 종료 코드: 0 성공, 2 실행할 수 없음."""
    parser = argparse.ArgumentParser(
        prog="python -m drowsy.demo",
        description=(
            "Run the whole pipeline on one recording (its own first 60 s are the baseline) and write a timeline "
            "PNG and an events JSON. Face frames are never saved."
        ),
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="experiment config YAML")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--video", type=Path, help="video file; frames go through MediaPipe FaceLandmarker (needs paths.model_path)"
    )
    source.add_argument(
        "--raw",
        type=Path,
        help="already extracted or synthetic RawFrame parquet (MediaPipe is not needed). The rotation record that "
        "the extraction left in the file is printed as one line (ambiguous check, manual angle, or no record)",
    )
    parser.add_argument(
        "--rotation",
        type=int,
        choices=extract.ROTATION_CANDIDATES,
        default=None,
        help=(
            "with --video: turn the picture counter-clockwise by this angle to make it upright (same convention as "
            "the manual rotation tables of the config). Default: the automatic rotation check of the extraction "
            "picks the angle; it is printed together with whether the check was ambiguous"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=(
            "exported ONNX model (default: demo.model_path of the config; while that file does not exist, "
            f"export/{FALLBACK_MODEL_NAME})"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"output folder (default: <demo.out_dir>/{DEFAULT_OUT_SUBDIR}, i.e. out/{DEFAULT_OUT_SUBDIR})",
    )
    parser.add_argument(
        "--speed-kmh",
        type=float,
        default=None,
        help="constant vehicle speed for the alert speed gates (default: not given, alerts are not suppressed)",
    )
    args = parser.parse_args(argv)

    try:
        if args.rotation is not None and args.video is None:
            raise DemoError(
                "--rotation needs --video: a raw parquet was turned upright when it was extracted "
                "(python -m drowsy.extract)"
            )
        cfg = load_config(args.config)
        onnx_path = args.model
        if onnx_path is None:
            onnx_path = cfg.demo.model_path
            if not onnx_path.is_file():
                fallback = cfg.paths.export_dir / FALLBACK_MODEL_NAME
                print(
                    f"note: demo.model_path {onnx_path} does not exist yet; using the random-weight contract "
                    f"model {fallback}",
                    file=sys.stderr,
                )
                onnx_path = fallback
        out_dir = args.out if args.out is not None else cfg.demo.out_dir / DEFAULT_OUT_SUBDIR
        if args.video is not None:
            png_path, json_path = run_demo(
                args.video,
                onnx_path,
                out_dir,
                args.speed_kmh,
                cfg=cfg,
                rotation=args.rotation,
                # 긴 영상을 다 돌리기 전에 각도를 볼 수 있게, 정해지는 대로 바로 출력한다
                on_rotation=lambda chosen: print(describe_video_rotation(chosen), flush=True),
            )
        else:
            png_path, json_path = run_demo_raw(args.raw, onnx_path, out_dir, args.speed_kmh, cfg=cfg)
    except (DemoError, extract.ExtractError, av.error.FFmpegError, ValueError, OSError) as exc:
        # SpecError, ContractError, VideoError와 pyarrow의 형식 오류는 ValueError다
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = json.loads(json_path.read_text(encoding="utf-8"))
    print(report["note"])
    if args.raw is not None:  # 추출이 parquet에 남긴 회전 기록(가르지 못했는지, 기록이 없는지)을 한 줄로 알린다
        print(report["source"]["rotation_note"])
    print(f"wrote {png_path} ({png_path.stat().st_size} bytes)")
    print(f"wrote {json_path} ({json_path.stat().st_size} bytes)")
    print(
        f"frames {report['counters']['frames_accepted']}, session {report['session']['duration_ms'] / 1000.0:.1f} s, "
        f"events {report['counters']['events']}, alerts {report['counters']['alerts']}, "
        f"model outputs {report['probabilities']['n_outputs']}, drowsiness.source {report['drowsiness_source']}"
    )
    if report["probabilities"]["empty_reason"] is not None:
        print(report["probabilities"]["empty_reason"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
