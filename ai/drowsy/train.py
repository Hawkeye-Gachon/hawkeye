"""학습, 중첩 교차검증, 최종 모델 (PLAN 5.2절, 6장, 11장 E2~E4). 이식 대상이 아니다.

스텝 특징은 여기서 계산하지 않는다. `drowsy.dataset`이 `DriverPipeline`으로 만든 스텝 표와 창만 쓴다(PLAN 3.1절).

아래에 나오는 D7, D10, D11, D16은 계획서(docs/PLAN.md 1.2절 표)의 결정 번호다. D7: 영상 단위 정확도의 분모는 논리 영상
전체이고 창이 하나도 없는 영상은 오답으로 센다. D10: 최종 모델은 전원으로 학습하고 보고하는 수치는 교차검증 추정치다.
D11: 교차검증 결과(앱 조건 평가)가 정해 둔 조건을 모두 채울 때만 모델을 규칙 엔진의 기본 졸음 판정원으로 둔다.
D16: TCN이 최종 모델이면 INT8 파일도 만들고 정확도 변화를 보고한다.

구성
- `fit`: 모델 하나를 학습한다. 표집은 피험자 → 영상 → 출처 → 창 순서(`BalancedSampler`), AdamW, 200 갱신 = 1단위,
  내부 검증 macro-F1 기준 조기 종료, 라벨 스무딩, 표준화 공간의 가우시안 잡음. 표준화 상수는 학습 창에서만, 표집과
  같은 출처별 몫으로 계산하고(`standardization_stats`), 온도는 내부 검증 로짓으로 맞춘다(`fit_temperature`).
- 학습 표본의 출처(PLAN 5.2절 증강, `mix_sources`): 원본 추출, 15 fps 누락 모사 추출(`training.fps_sim_share`, 0.25),
  프레임 누락 변형(`training.frame_drop_share`, 0.25를 `training.frame_drop_variants`개가 나눈다). 남은 몫(0.5)이
  원본이고, 없는 출처의 몫은 원본으로 간다. 변형의 스텝 표는 `load_cv_data`가 `dataset.load_frame_drop_tables`로
  만든다. 어느 출처든 학습 피험자의 창만 학습에 들어가고, 검증·시험·온도는 원본 추출의 창만 쓴다.
- `cross_validate` / `run_nested_cv`: PLAN 6장의 중첩 교차검증. 바깥 fold마다 나머지 fold 안에서 검증 fold를
  하나씩 돌려 계열·설정·학습 길이·온도를 고르고, 고른 것으로 학습 fold 전체를 시드 3개로 다시 학습해 시험 fold를
  한 번 평가한다. 주 결과 행(`selected`)과 계열별 행(`tcn`, `gru`), 앱 조건 평가(alert 영상 기준선으로 학습한 fold
  모델을 시험 fold의 자기 세션 기준선 특징으로 평가, 재학습 없음)를 `drowsy.evaluate`의 예측 표로 만든다.
- 부가 결과(PLAN 11장 E3, D16. 리포트 9절에만 실리고 주 결과 표, 비교표, D11에는 쓰이지 않는다)
  - 15 fps 모사 평가: 그 추출이 있으면 같은 fold 모델을 시험 fold의 15 fps 추출 특징으로도 평가한다
    (`feature_source = fps15`, 학습 조건과 앱 조건 둘 다).
  - INT8(D16): `export.int8_if_tcn`이 켜져 있으면, 다시 학습한 계열별 TCN fold 모델을 임시 폴더에 내보내 양자화하고
    float 파일과 INT8 파일을 onnxruntime으로 시험 fold에 돌린다(계열 `tcn_onnx`, `tcn_onnx_int8`).
  - 순서형 소프트 라벨(`cv --ordinal`)과 속도 특징 제외(`cv --no-velocity`): 같은 절차를 그 선택 사항으로 한 번 더
    도는 부가 실행이다. 리포트는 실행 폴더의 하위 폴더 `ordinal/`, `no_velocity/`에 쓰고 주 리포트는 읽기만 한다.
    주 실행의 리포트가 있으면 그 묶음과 짝지은 비교를 싣는다.
  - 내보낼 조건(`training.export_arm`)이 아닌 정규화 조건으로 학습한 실행(`cv --arm ...`)은 묶음이 모두 부가 묶음이고
    D11을 판정하지 않는다.
- `fit_final`: 같은 선택 절차를 전체 피험자에 한 번(공식 fold) 적용하고 전원으로 시드 0 학습, 학습한 평균·표준편차·
  온도를 feature_spec 사본에 써서 `drowsy.export_onnx`로 내보낸다. 모델 옆에 규칙 설정 사본 `rules.json`도 쓴다.
  최종 모델의 INT8 파일에 대해서는 float 모델과의 확률 차이만 적는다(전원으로 학습해 시험 fold가 없다. 양자화에 따른
  지표 변화는 교차검증의 부가 묶음이 그 추정치다).

시험 fold가 선택, 조기 종료, 표준화 상수, 온도에 쓰이지 않게 하는 장치
- 안쪽 선택 작업(`_ScoreJob`)에는 시험 fold의 스텝을 아예 넣지 않는다(원본, 15 fps 추출, 프레임 누락 변형 모두).
- `fit`은 시작할 때, `predict_proba`는 평가할 때 `assert_subject_disjoint`를 부른다. 겹치면 `SubjectOverlapError`로
  멈춘다.
- 예외는 feature_spec의 closure 상수와 전역 상수뿐이다(`CONSTANTS_NOTE`). 리포트 설명에 그 사실을 적는다.
- 표집기와 표준화 상수는 원본 학습 묶음에 있는 영상만 대상으로 삼는다. 학습 전용 출처에 다른 피험자의 창이 섞여
  들어와도 표본과 상수에 닿지 않는다(두 번째 방어선).
- `tests/test_leakage.py`가 시험 fold의 추출 결과(원본, 15 fps 추출, 그리고 원본에서 만들어지는 프레임 누락 변형)를
  바꿔치기해도 그 fold의 선택 기록, 표준화 상수, 온도, 가중치 해시가 그대로인지 본다.

리포트 폴더 `{paths.reports_dir}/{실행명}/`
    summary.md, summary.json, per_fold.csv, ... (`evaluate.write_reports`)
    predictions.parquet     예측 표(신경망 묶음). parquet 메타데이터 `drowsy_predictions`에 데이터 종류를 적는다
    selection.json          바깥 fold별 안쪽 선택 기록(후보별 내부 검증 값, 고른 설정, 학습 길이, 온도)과 다시 학습한
                            모델마다의 표준화 상수, 온도, 가중치 해시, 출처별 몫(`refits`)
    timings.json            단계별 소요 시간
    run.json                실행 기록: 학습 조건, run_id, 축소 설정 여부, 선택 사항(`variant`), pipeline_hash, 스텝 표의
                            내용 지문, 기준선·창 요약, 학습 표본의 출처와 몫(`train_sources`), 부가 평가(15 fps, INT8)의
                            기록(`supplementary`), 추출 결과의 회전 점검 기록(`rotation_check`)
    ordinal/, no_velocity/  부가 실행(`cv --ordinal`, `cv --no-velocity`)의 같은 파일들. 그 실행의 묶음은 모두 부가 결과다
    lgbm/                   `python -m drowsy.baseline_lgbm cv`가 쓴다. 있으면 기준 LightGBM 행과 D11 판정을 함께 싣는다
    probes/                 `python -m drowsy.baseline_lgbm probes`가 쓴다. 우연 수준을 넘으면 리포트 맨 앞에 적는다
    final/                  `final` 명령의 선택 기록, 체크포인트, 내보내기 검증 기록
같은 실행 폴더에 다른 학습 조건, 축소 설정 여부, 다른 종류의 데이터(합성 대 실제)로 만든 리포트가 있으면 덮어쓰지 않고
멈춘다(`--overwrite`를 주면 덮어쓴다). `final/`과 부가 실행의 하위 폴더도 같다.

명령 (저장소 루트에서 실행. 공통 옵션은 하위 명령 앞에 둔다)
    python -m drowsy.train --config config/default.yaml [--data-root DIR] [--run-name NAME] [--rebuild] cv
        [--arm alert_baseline] [--families tcn gru] [--fast] [--overwrite] [--allow-partial-fps-sim]
        [--ordinal | --no-velocity]
    python -m drowsy.train --config config/default.yaml [--data-root DIR] [--run-name NAME] [--rebuild] final
        [--families tcn gru] [--fast] [--out-dir DIR] [--version N] [--overwrite] [--allow-partial-fps-sim]
합성 데이터로 만든 출력은 첫 줄에 합성 데이터임을 적는다. 합성 데이터의 점수는 파이프라인 점검용이다(PLAN 3.4절).
두 명령 모두 데이터 설명 다음 줄에, 학습을 시작하기 전에, 추출 결과의 회전 점검 기록을 구분별 파일 수로 출력한다
(`print_rotation_check`). 회전이 확인되지 않은 영상(자동 점검이 각도를 가르지 못했는데 수동 회전 보정이 없는 영상, 보정표의
각도가 적용되지 않은 영상)이 있으면 그 아래에 영상마다 한 줄씩 적고, 실행 기록의 `rotation_check`에 목록을, 리포트 설명에
그 수를 남긴다. 그런 영상을 빼지는 않는다. 영상을 보고 `extraction.rotation_overrides`를 채우는 것은 사람의 일이다.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import random
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import onnxruntime as ort
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import Tensor, nn

from drowsy import evaluate, export_onnx
from drowsy.config import (
    ARM_ALERT_BASELINE,
    ARM_SESSION_BASELINE,
    ARMS,
    ExperimentConfig,
    ExportSettings,
    TrainingSettings,
    apply_fast_profile,
    check_run_name,
    load_config,
)
from drowsy.dataset import (
    STEP_COLUMNS,
    SUBJECTS_PER_FOLD,
    SYNTHETIC_NOTE,
    DatasetSource,
    InnerSplit,
    RotationSummary,
    StepTable,
    SubjectOverlapError,
    WindowSet,
    assert_subject_disjoint,
    baseline_window_summary,
    dataset_rotation_checks,
    load_frame_drop_tables,
    load_step_tables,
    make_windows,
    nested_splits,
    open_dataset,
    rotation_check_line,
    rotation_flag_lines,
)
from drowsy.extract import OVERRIDES_KEY, ExtractError
from drowsy.models import KIND_GRU, KIND_TCN, KINDS, ExportWrapper, build_model, count_parameters, save_checkpoint
from drowsy.spec import (
    FeatureSpec,
    PipelineSpec,
    RulesConfig,
    export_rules_json,
    file_sha256,
    load_feature_spec,
    load_rules,
    read_feature_spec_dict,
    write_feature_spec,
)
from drowsy.types import CLASS_ORDER, F_CLOSING_VEL_MEAN_60, F_OPENING_VEL_MEAN_60, FEATURE_NAMES

OPTIMIZER_ADAMW = "adamw"
DEVICE_AUTO = "auto"

# 온도 탐색 범위.
# 가정: 계획서에 없는 값이다. 내부 검증 예측이 모두 맞으면 음의 로그 가능도가 온도 0 쪽으로 계속 내려가므로 끝을 둔다.
TEMPERATURE_MIN = 0.1
TEMPERATURE_MAX = 10.0
_GOLDEN_STEPS = 80  # 황금 분할 탐색 횟수. 구간 폭이 0.618^80배로 줄어든다

_CHUNK = 4096  # 전처리와 추론을 나눠 넣는 창 수

PREDICTIONS_FILE = "predictions.parquet"
PREDICTIONS_METADATA_KEY = b"drowsy_predictions"  # predictions.parquet의 스키마 메타데이터 키(데이터 종류 표지)
SELECTION_FILE = "selection.json"
TIMINGS_FILE = "timings.json"
RUN_INFO_FILE = "run.json"
LGBM_SUBDIR = "lgbm"
PROBES_SUBDIR = "probes"
PROBES_FILE = "probes.json"
FINAL_SUBDIR = "final"
FINAL_RECORD_FILE = "final.json"
FINAL_CHECKPOINT_FILE = "final_model.pt"
FINAL_RULES_FILE = "rules.json"  # 내보낸 모델 옆에 쓰는 규칙 설정 사본(PLAN 11장 E4)
DEFAULT_CV_RUN_NAME = "nested_cv"
DEFAULT_FINAL_VERSION = 1  # 학습한 모델은 1부터(v0은 무작위 가중치)
FAST_NOTE = "FAST profile (smoke run, not the pre-registered procedure)"
# PLAN 6장: 시험 fold는 선택, 조기 종료, 표준화 상수, 온도 어디에도 쓰이지 않는다. 예외는 E0에서 라벨 없이 한 번
# 정하는 closure 상수와 전역 상수뿐이고 리포트에 밝힌다. 교차검증 리포트의 설명(첫 문단)에 이 문장을 싣는다.
# 가정: 이 문장은 절차를 적은 것이다. feature_spec의 상수가 실제로 E0(`python -m drowsy.signal_check e0`)를 거친
# 값인지는 여기서 확인하지 않는다(합성 데이터 리포트에도 같은 문장이 실린다).
CONSTANTS_NOTE = (
    "Exception to test-fold isolation (PLAN 4.3, 6): the closure constants and global constants of the feature spec "
    "are fixed once at E0, without labels, on all videos including the test-fold subjects; selection, early "
    "stopping, standardization and temperature never use the test fold"
)

_PARITY_REAL_WINDOWS = 200  # 내보낸 모델의 일치 검증에 쓰는 실제 창 수의 상한

# 프레임 누락 변형(PLAN 5.2절 증강 (2))의 시드. 변형 번호는 0부터 `training.frame_drop_variants − 1`까지다.
# 가정: 시드는 0으로 고정한다(`python -m drowsy.dataset --frame-drop-variants 0`의 기본 시드와 같아 그 명령이 만든
# 캐시를 그대로 쓴다). 학습 시드마다 다른 변형을 만들지 않는다. 변형 하나를 만드는 데 전체 세션의 재생이 한 번 든다.
FRAME_DROP_SEED = 0
SOURCE_FRAME_DROP_PREFIX = "frame_drop_v"  # 프레임 누락 변형 v의 출처 이름: frame_drop_v0, frame_drop_v1, ...
# 15 fps 누락 모사 추출의 출처 이름. 그 추출로 평가한 부가 묶음의 `feature_source` 표지와 같은 값이다.
SOURCE_FPS_SIM = evaluate.FEATURE_SOURCE_FPS15

# 부가 실행(PLAN 11장 E3)의 선택 사항. 값은 예측 표의 `variant` 표지이고 run_id의 접미사와 리포트 하위 폴더 이름이다.
SUPPLEMENTARY_VARIANTS: tuple[str, ...] = (evaluate.VARIANT_ORDINAL, evaluate.VARIANT_NO_VELOCITY)
RUN_VARIANTS: tuple[str, ...] = (evaluate.VARIANT_BASE,) + SUPPLEMENTARY_VARIANTS
# 속도 특징 #7, #8 (PLAN 4.6절). `--no-velocity` 실행은 이 두 입력을 표준화 공간에서 0으로 둔다.
VELOCITY_FEATURES: tuple[int, ...] = (F_CLOSING_VEL_MEAN_60, F_OPENING_VEL_MEAN_60)
# D16의 fold 모델 비교: 내보낸 float 파일과 INT8 파일을 onnxruntime으로 돌린 예측의 계열 이름.
# INT8 쪽은 `evaluate.INT8_SUFFIX` 규칙을 따르므로 `evaluate.supplementary_pairs`가 둘을 짝짓는다.
FAMILY_TCN_ONNX = "tcn_onnx"
FAMILY_TCN_ONNX_INT8 = FAMILY_TCN_ONNX + evaluate.INT8_SUFFIX
# 최종 모델의 INT8 기록(`final.json`)에 적는 문장: 여기서 재는 것은 지표 변화가 아니라 확률 차이다.
INT8_FINAL_NOTE = (
    "probability drift only: the final model is trained on all subjects, so there is no held-out fold to measure a "
    "metric change on; the metric change of INT8 quantization is the cross-validation estimate of `train cv` "
    f"(supplementary groups {FAMILY_TCN_ONNX} and {FAMILY_TCN_ONNX_INT8})"
)


# ---------------------------------------------------------------------------
# 작은 도우미
# ---------------------------------------------------------------------------


def resolve_device(name: str) -> torch.device:
    """설정의 `training.device`(auto, cpu, cuda)를 장치로 바꾼다. auto는 GPU가 있으면 GPU, 없으면 CPU다."""
    if name == DEVICE_AUTO:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("training.device is cuda but no CUDA device is available")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    """파이썬, numpy, PyTorch(CPU와 GPU)의 난수 시드를 고정하고 cuDNN을 결정적 모드로 둔다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def window_subjects(windows: WindowSet) -> tuple[int, ...]:
    """창 묶음에 들어 있는 피험자 번호(오름차순)."""
    return tuple(sorted({int(value) for value in windows.meta["subject"].unique()}))


def subset_subjects(windows: WindowSet, subjects: Iterable[int]) -> WindowSet:
    """주어진 피험자의 창만 골라낸 묶음."""
    wanted = np.array(sorted({int(value) for value in subjects}), dtype=np.int64)
    return windows.subset(np.isin(windows.meta["subject"].to_numpy(dtype=np.int64), wanted))


def median_half_up(values: Sequence[int]) -> int:
    """정수들의 중앙값. 짝수 개면 가운데 두 값의 평균을 `floor(x + 0.5)`로 반올림한다."""
    if len(values) == 0:
        raise ValueError("median of an empty sequence")
    ordered = sorted(int(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return math.floor((ordered[middle - 1] + ordered[middle]) / 2.0 + 0.5)


def data_note(synthetic: bool) -> str:
    """리포트와 기록 파일의 첫 줄: 합성 데이터이면 `SYNTHETIC_NOTE`, 아니면 `evaluate.REAL_NOTE`."""
    return SYNTHETIC_NOTE if synthetic else evaluate.REAL_NOTE


def write_json(payload: Mapping[str, Any], path: Path) -> None:
    """기록 파일을 JSON으로 쓴다(들여쓰기 2, UTF-8). NaN과 무한대는 쓰지 못한다(None으로 바꿔서 넘긴다)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _progress(message: str) -> None:
    """긴 실행의 진행 상황을 표준 오류에 적는다(점수는 적지 않는다. 점수는 리포트 파일에만 둔다)."""
    print(message, file=sys.stderr, flush=True)


def weights_sha256(net: nn.Module) -> str:
    """신경망 가중치(파라미터와 버퍼)의 sha256 16진수. 선택 기록에 적어 두 실행의 모델이 같은지 대조하는 데 쓴다.

    `state_dict`의 항목을 이름순으로, 항목마다 `이름(UTF-8) + b"\\0" + dtype 이름 + b"\\0" + 모양 + b"\\0" + 값 바이트`를
    넣는다. CPU에서 같은 시드, 같은 입력으로 학습한 모델은 값이 같다.
    """
    digest = hashlib.sha256()
    for name, value in sorted(net.state_dict().items()):
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        digest.update(name.encode("utf-8") + b"\0" + str(array.dtype).encode("ascii") + b"\0")
        digest.update(",".join(str(size) for size in array.shape).encode("ascii") + b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def ordinal_targets(n_classes: int, neighbor_mass: float) -> np.ndarray:
    """순서형 소프트 라벨의 목표 분포 `[C, C]`(float64). 행이 정답 클래스, 열이 클래스다 (PLAN 5.2절 부가 실험).

    정답 클래스가 `1 − neighbor_mass`를 갖고, neighbor_mass를 순서상 이웃 클래스(번호가 1 차이)에 똑같이 나눠 준다.
    가운데 클래스는 양쪽에 절반씩, 끝 클래스는 하나뿐인 이웃에 전부다. 3클래스, 0.2이면
    alert `[0.8, 0.2, 0]`, low_vigilant `[0.1, 0.8, 0.1]`, drowsy `[0, 0.2, 0.8]`.
    가정: 계획서는 "이웃 클래스에 확률을 나눠 주는 방식"이라고만 했다. 나눠 주는 크기(`training.ordinal_neighbor_mass`,
    0.2)와 끝 클래스의 규칙은 여기서 정한 것이다. 균일 라벨 스무딩은 함께 쓰지 않는다(이것이 그 자리를 대신한다).
    """
    if n_classes < 2:
        raise ValueError(f"ordinal targets need at least 2 classes, got {n_classes}")
    if not 0.0 <= neighbor_mass < 0.5:
        raise ValueError(f"neighbor_mass must be within [0, 0.5), got {neighbor_mass!r}")
    target = np.zeros((n_classes, n_classes), dtype=np.float64)
    for index in range(n_classes):
        neighbors = [other for other in (index - 1, index + 1) if 0 <= other < n_classes]
        target[index, index] = 1.0 - neighbor_mass
        for other in neighbors:
            target[index, other] = neighbor_mass / len(neighbors)
    return target


# ---------------------------------------------------------------------------
# 내부 검증 지표
# ---------------------------------------------------------------------------


class VideoMacroF1:
    """창 예측에서 영상 가중 창 단위 macro-F1을 계산한다. 리포트의 주 지표(`window_macro_f1`)와 같은 정의다.

    계산은 `evaluate.vote_window_metrics`가 한다. 여기서는 영상별 표 수만 센다. 창이 속한 영상을 미리 찾아 두므로
    같은 창 묶음에 여러 번(학습 단위마다, 부스팅 회차마다) 부를 수 있다.
    가정: 조기 종료와 선택의 "내부 검증 macro-F1"은 리포트의 주 지표와 같은 영상 가중 값이다(창을 그대로 합친 값이 아니다).
    """

    def __init__(self, meta: pd.DataFrame) -> None:
        codes = meta.groupby(list(evaluate.VIDEO_KEYS), sort=False).ngroup().to_numpy()
        n_videos = int(codes.max()) + 1 if len(codes) else 0
        class_index = np.zeros(n_videos, dtype=np.int64)
        class_index[codes] = meta["class_index"].to_numpy(dtype=np.int64)
        self._codes = codes
        self._class_index = class_index
        self._n_windows = len(codes)

    def __call__(self, pred_class: np.ndarray) -> float:
        pred = np.asarray(pred_class, dtype=np.int64)
        if pred.shape != (self._n_windows,):
            raise ValueError(f"expected {self._n_windows} predictions, got shape {pred.shape}")
        votes = np.zeros((len(self._class_index), len(CLASS_ORDER)), dtype=np.int64)
        np.add.at(votes, (self._codes, pred), 1)
        table = pd.DataFrame({"class_index": self._class_index, "n_windows": votes.sum(axis=1)})
        for index, name in enumerate(CLASS_ORDER):
            table[f"votes_{name}"] = votes[:, index]
        return float(evaluate.vote_window_metrics(table).video_weighted.macro_f1)


# ---------------------------------------------------------------------------
# 표준화 상수, 온도
# ---------------------------------------------------------------------------


def _video_moments(
    windows: WindowSet, spec: FeatureSpec
) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]:
    """창 묶음의 영상 (피험자 번호, video_id)마다 특징별 (평균, 제곱 평균). log1p 열은 변환한 뒤의 값이다."""
    if windows.x.shape[0] == 0:
        return {}
    if windows.x.shape[2] != spec.n_features:
        raise ValueError(f"windows have {windows.x.shape[2]} features, the feature spec has {spec.n_features}")
    mask = np.array(spec.log1p_mask, dtype=bool)
    codes = windows.meta.groupby(list(evaluate.VIDEO_KEYS), sort=False).ngroup().to_numpy()
    first_rows = np.unique(codes, return_index=True)[1]  # 영상 번호 순서대로 그 영상의 첫 창
    subjects = windows.meta["subject"].to_numpy()[first_rows].tolist()
    video_ids = windows.meta["video_id"].to_numpy()[first_rows].tolist()
    moments: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    for code, (subject, video_id) in enumerate(zip(subjects, video_ids)):
        values = windows.x[codes == code].astype(np.float64).reshape(-1, spec.n_features)
        values[:, mask] = np.log1p(np.maximum(values[:, mask], 0.0))
        moments[(int(subject), str(video_id))] = (values.mean(axis=0), (values * values).mean(axis=0))
    return moments


def standardization_stats(
    windows: WindowSet | Sequence[WindowSet], spec: FeatureSpec, extra_shares: Sequence[float] | None = None
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """학습 창에서 특징별 평균과 표준편차를 계산한다 (PLAN 4.7절, 5.1절). `(mean, std)`를 돌려준다.

    - transform이 log1p인 열은 `log1p(max(x, 0))`를 적용한 뒤의 값으로 계산한다(그래프 안의 순서와 같다).
    - 영상마다 같은 비중이다: 영상별로 (창, 스텝) 전체의 평균과 제곱 평균을 구한 뒤 영상들에 대해 평균한다.
      표집이 영상을 균등하게 뽑으므로 긴 영상이 상수를 지배하지 않게 같은 가중을 쓴다.
    - 표준편차는 모표준편차이고 `standardization.std_floor`(0.001) 아래로 내려가지 않게 올린다.
    - 묶음을 여러 개 주면 첫 묶음이 원본 추출이고 나머지는 학습 전용 출처(15 fps 누락 모사 추출, 프레임 누락 변형)다.
      extra_shares는 나머지 묶음마다 하나씩, 표집기(`BalancedSampler`)가 그 묶음에서 뽑는 표본의 비율이다. 영상 v의
      값은 `Σ_e share_e · (묶음 e에서의 v) + (1 − Σ_e share_e) · (첫 묶음에서의 v)`이고, 합은 v의 창이 있는 나머지
      묶음에 대해서만 낸다(그 영상이 없는 묶음의 몫은 원본으로 간다. 표집기와 같은 규칙이다). 첫 묶음에 창이 없는
      영상은 표집되지 않으므로 넣지 않는다. 그래서 상수는 표집기가 실제로 뽑는 표본의 분포에서 계산한 값이 된다.
      extra_shares가 None이면 묶음들이 같은 몫(1 / 묶음 수)을 갖는다.
    가정: 영상들 사이의 비중은 같다. 피험자마다 영상 수가 다르면(영상이 빠진 피험자) 표집기의 "피험자 → 영상" 비중과
    조금 다르다.
    받은 창만 쓴다. 검증·시험 피험자의 창을 넘기지 않는 것은 호출하는 쪽(`fit`)의 일이고 `fit`은 학습 창만 넘긴다.
    """
    sets = [windows] if isinstance(windows, WindowSet) else list(windows)
    if len(sets) == 0:
        raise ValueError("cannot compute standardization constants without training windows")
    shares = [1.0 / len(sets)] * (len(sets) - 1) if extra_shares is None else [float(v) for v in extra_shares]
    if len(shares) != len(sets) - 1:
        raise ValueError(f"need one extra share per extra window set: {len(sets) - 1} sets, {len(shares)} shares")
    if any(share < 0.0 for share in shares) or sum(shares) >= 1.0:
        raise ValueError("extra shares must be >= 0 and sum to less than 1")
    primary = _video_moments(sets[0], spec)
    if not primary:
        raise ValueError("cannot compute standardization constants without training windows")
    extras = [_video_moments(item, spec) for item in sets[1:]]
    # 행의 순서: 첫 묶음의 영상들, 그다음 나머지 묶음의 영상들(묶음 순서). 가중치의 합은 영상마다 1이다.
    weights: list[float] = []
    first: list[np.ndarray] = []
    second: list[np.ndarray] = []
    for key, (mean_1, mean_2) in primary.items():
        used = sum(share for share, extra in zip(shares, extras) if share > 0.0 and key in extra)
        weights.append(1.0 - used)
        first.append(mean_1)
        second.append(mean_2)
    for share, extra in zip(shares, extras):
        for key, (mean_1, mean_2) in extra.items():
            if share > 0.0 and key in primary:
                weights.append(share)
                first.append(mean_1)
                second.append(mean_2)
    weight = np.array(weights, dtype=np.float64)[:, None]
    mean = np.sum(weight * np.stack(first), axis=0) / len(primary)
    square = np.sum(weight * np.stack(second), axis=0) / len(primary)
    variance = np.maximum(square - mean * mean, 0.0)  # 반올림으로 음수가 되지 않게 자른다
    std = np.maximum(np.sqrt(variance), spec.standardization.std_floor)
    return tuple(float(value) for value in mean), tuple(float(value) for value in std)


def with_standardization(
    spec: FeatureSpec, mean: Sequence[float], std: Sequence[float], temperature: float
) -> FeatureSpec:
    """평균, 표준편차, 온도를 바꾼 `FeatureSpec` 사본. 나머지(특징 목록, pipeline, 해시)는 그대로다."""
    if len(mean) != spec.n_features or len(std) != spec.n_features:
        raise ValueError(f"need {spec.n_features} means and stds, got {len(mean)} and {len(std)}")
    features = tuple(
        dataclasses.replace(feature, mean=float(m), std=float(s)) for feature, m, s in zip(spec.features, mean, std)
    )
    standardization = dataclasses.replace(spec.standardization, temperature=float(temperature))
    return dataclasses.replace(spec, features=features, standardization=standardization)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """로짓 `[N, 3]`과 정답 `[N]`에서 온도 하나를 맞춘다 (PLAN 5.2절 "확률 보정").

    `softmax(logits / T)`의 평균 음의 로그 가능도가 가장 작은 T를 `[TEMPERATURE_MIN, TEMPERATURE_MAX]`에서 찾는다.
    이 값은 1/T에 대해 볼록하므로 log T에 대한 황금 분할 탐색으로 충분하다. 같은 입력이면 같은 값이 나온다.
    가정: 창을 그대로 합쳐(영상 가중 없이) 맞춘다. 보정 전후 ECE도 창을 그대로 합쳐 계산한다.
    가정: 탐색 범위는 0.1 ~ 10이다(계획서에 없는 값). 예측이 모두 맞거나 모두 틀리면 범위의 끝이 나온다.
    """
    values = np.asarray(logits, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or target.shape != (values.shape[0],) or values.shape[0] == 0:
        raise ValueError(f"need logits [N, classes] and labels [N] with N >= 1, got {values.shape} and {target.shape}")
    if not np.isfinite(values).all():
        raise ValueError("logits must be finite")
    rows = np.arange(values.shape[0])

    def loss(log_t: float) -> float:
        scaled = values / math.exp(log_t)
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        return float(np.mean(np.log(np.exp(scaled).sum(axis=1)) - scaled[rows, target]))

    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    low, high = math.log(TEMPERATURE_MIN), math.log(TEMPERATURE_MAX)
    left, right = high - ratio * (high - low), low + ratio * (high - low)
    loss_left, loss_right = loss(left), loss(right)
    for _ in range(_GOLDEN_STEPS):
        if loss_left <= loss_right:
            high, right, loss_right = right, left, loss_left
            left = high - ratio * (high - low)
            loss_left = loss(left)
        else:
            low, left, loss_left = left, right, loss_right
            right = low + ratio * (high - low)
            loss_right = loss(right)
    return float(math.exp((low + high) / 2.0))


# ---------------------------------------------------------------------------
# 표집
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class _VideoIndex:
    """창 묶음 하나의 표집용 색인. 창을 (피험자, 영상) 순으로 정렬한 위치에서 센다."""

    keys: tuple[tuple[int, str], ...]  # 영상의 (피험자 번호, video_id), 정렬 순
    start: np.ndarray  # 영상별 첫 창의 정렬 위치
    count: np.ndarray  # 영상별 창 수
    order: np.ndarray  # 정렬 위치 → 묶음 안의 창 번호


def _video_index(meta: pd.DataFrame) -> _VideoIndex:
    frame = meta[["subject", "video_id"]].reset_index(drop=True)
    ordered = frame.sort_values(["subject", "video_id"], kind="stable")
    order = ordered.index.to_numpy(dtype=np.int64)
    subject = ordered["subject"].to_numpy(dtype=np.int64)
    video = ordered["video_id"].to_numpy()
    n = len(ordered)
    new_video = np.ones(n, dtype=bool)
    new_video[1:] = (subject[1:] != subject[:-1]) | (video[1:] != video[:-1])
    first = np.flatnonzero(new_video)
    keys = tuple((int(s), str(v)) for s, v in zip(subject[first].tolist(), video[first].tolist()))
    return _VideoIndex(keys=keys, start=first, count=np.diff(np.append(first, n)), order=order)


class BalancedSampler:
    """피험자 → 영상 → 창 순서로 균등하게 뽑는 표집기 (PLAN 5.2절).

    metas는 창 묶음들의 메타데이터다. 첫 묶음이 원본 추출이고, 나머지는 같은 영상을 다른 방식으로 만든 학습 전용
    묶음(15 fps 누락 모사 추출, 프레임 누락 변형)이다. extra_shares는 나머지 묶음마다 하나씩, 그 묶음에서 뽑을 표본의
    비율이다(기본 설정: 15 fps 모사 0.25, 프레임 누락 변형 0.25, 남은 0.5가 원본).

    한 표본을 뽑는 순서
      1. 첫 묶음의 피험자 하나를 균등하게.
      2. 그 피험자의 영상(첫 묶음에 창이 있는 것) 하나를 균등하게. 피험자마다 클래스별 영상이 하나씩이면 클래스도
         균형을 이룬다.
      3. 묶음: 나머지 묶음 e를 `extra_shares[e]`의 확률로, 첫 묶음을 남은 확률로 고른다. 고른 묶음에 그 영상의 창이
         없으면 첫 묶음에서 뽑는다(그 묶음의 몫이 원본으로 간다).
      4. 고른 묶음에서 그 영상의 창 하나를 균등하게(나뉜 영상은 부분들의 창을 합쳐서).
    `draw(n)`은 묶음들을 순서대로 이어 붙인 배열에서의 창 번호 n개를 돌려준다. 같은 시드면 같은 수열이다.

    피험자와 영상을 먼저 고르므로, 나머지 묶음이 일부 피험자의 창만 담고 있어도(추출이 덜 끝난 폴더) 피험자와 영상의
    비중은 그대로 균등하다. 그 묶음이 없는 영상의 표본이 원본에서 나올 뿐이다.
    가정: "학습 보폭 5초, 시작 위치 무작위 이동"(PLAN 5.2절)은 영상의 보폭 1 창을 균등하게 뽑는 것으로 구현한다.
    보폭 격자의 칸을 고른 뒤 시작 위치를 0 ~ 보폭 − 1 스텝 옮기는 것과 같은 분포이고, 구간 끝의 창이 더 자주 뽑히는
    치우침이 없다. 그래서 이 표집기는 `windows.train_stride`를 쓰지 않는다(그 값은 LightGBM의 학습 창과 리포트의 창
    수에 쓰인다). 창 묶음은 보폭 1로 만든 것을 받는다. 다른 보폭의 묶음을 주면 받은 창들에서 균등하게 뽑는다.
    가정: 표집 대상인 피험자와 영상은 첫 묶음(원본 추출)에 창이 있는 것이다. 창이 없는 영상과 피험자, 나머지 묶음에만
    창이 있는 영상은 뽑지 않는다.
    """

    def __init__(self, metas: Sequence[pd.DataFrame], seed: int, extra_shares: Sequence[float] = ()) -> None:
        if len(metas) == 0 or len(metas[0]) == 0:
            raise ValueError("the first window set given to the sampler needs at least one window")
        shares = [float(share) for share in extra_shares]
        if len(shares) != len(metas) - 1:
            raise ValueError(f"need one extra share per extra window set: {len(metas) - 1} sets, {len(shares)} shares")
        if any(share < 0.0 for share in shares) or sum(shares) >= 1.0:
            raise ValueError("extra shares must be >= 0 and sum to less than 1")
        primary = _video_index(metas[0])
        video_subject = np.array([key[0] for key in primary.keys], dtype=np.int64)
        _, subject_count = np.unique(video_subject, return_counts=True)  # 영상은 피험자 순으로 정렬되어 있다
        self._subject_start = np.cumsum(subject_count) - subject_count
        self._subject_count = subject_count
        position = {key: index for index, key in enumerate(primary.keys)}
        self._starts = [primary.start]
        self._counts = [primary.count]
        self._orders = [primary.order]
        for meta in metas[1:]:
            index = _video_index(meta)
            start = np.zeros(len(primary.keys), dtype=np.int64)
            count = np.zeros(len(primary.keys), dtype=np.int64)
            for key, first, size in zip(index.keys, index.start.tolist(), index.count.tolist()):
                if key in position:  # 첫 묶음에 없는 영상은 표집 대상이 아니다
                    start[position[key]] = first
                    count[position[key]] = size
            self._starts.append(start)
            self._counts.append(count)
            self._orders.append(index.order)
        sizes = [len(meta) for meta in metas]
        self._offsets = np.cumsum([0] + sizes[:-1])
        self._edges = np.cumsum(shares)  # 균등 난수 u가 [edges[e−1], edges[e])이면 나머지 묶음 e
        self._rng = np.random.default_rng(seed)

    def draw(self, n: int) -> np.ndarray:
        """창 번호 n개(int64). 번호는 묶음들을 이어 붙인 순서에서 센다."""
        rng = self._rng
        subject = rng.integers(0, len(self._subject_start), size=n)
        video = self._subject_start[subject] + rng.integers(0, self._subject_count[subject])
        source = np.zeros(n, dtype=np.int64)
        if len(self._edges) > 0 and self._edges[-1] > 0.0:
            picked = np.searchsorted(self._edges, rng.random(n), side="right")
            for extra in range(len(self._edges)):
                source[(picked == extra) & (self._counts[extra + 1][video] > 0)] = extra + 1
        out = np.empty(n, dtype=np.int64)
        for position in range(len(self._counts)):
            chosen = np.flatnonzero(source == position)
            if len(chosen) == 0:
                continue
            which = video[chosen]
            window = self._starts[position][which] + rng.integers(0, self._counts[position][which])
            out[chosen] = self._offsets[position] + self._orders[position][window]
        return out


# ---------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """`fit` 한 번의 설정.

    - kind, hyper: `build_model(kind, n_features, **hyper)`에 넘기는 계열과 하이퍼파라미터.
    - spec: 기준 feature_spec. log1p 열, z_clip, std_floor를 여기서 읽는다(평균·표준편차·온도는 학습이 정한다).
    - settings: 학습 설정(PLAN 5.2절).
    - fixed_units: 검증 없이 다시 학습할 때의 학습 길이(단위 수). 검증 창을 줄 때는 None이어야 한다.
    - temperature: 검증 없이 다시 학습할 때 쓸 온도(안쪽 검증에서 고른 값). None이면 1.0.
    - variant: 학습의 선택 사항(`RUN_VARIANTS`). `base`는 사전에 고정한 절차 그대로, `ordinal`은 라벨 스무딩 대신
      순서형 소프트 라벨(`ordinal_targets`), `no_velocity`는 속도 특징 #7, #8을 표준화 공간에서 0으로 둔다.
    """

    kind: str
    hyper: Mapping[str, Any]
    spec: FeatureSpec
    settings: TrainingSettings
    fixed_units: int | None = None
    temperature: float | None = None
    variant: str = evaluate.VARIANT_BASE


@dataclass(frozen=True, eq=False)
class TrainSource:
    """학습 전용 출처 하나의 창 묶음 (PLAN 5.2절 증강). `fit`의 extra_train 항목이다.

    - name: 출처 이름(`SOURCE_FPS_SIM`, `frame_drop_v0` 등). 기록에만 쓴다.
    - share: 표집기가 이 출처에서 뽑는 표본의 비율. windows: 그 출처의 창(보폭 1).
    """

    name: str
    share: float
    windows: WindowSet

    def of(self, subjects: Iterable[int]) -> TrainSource:
        """주어진 피험자의 창만 남긴 사본."""
        return TrainSource(self.name, self.share, subset_subjects(self.windows, subjects))


@dataclass(frozen=True, eq=False)
class StepSource:
    """학습 전용 출처 하나의 스텝 표. 창을 만들기 전 단계의 `TrainSource`다."""

    name: str
    share: float
    steps: pd.DataFrame

    def of(self, subjects: Iterable[int]) -> StepSource:
        """주어진 피험자의 스텝만 남긴 사본."""
        return StepSource(self.name, self.share, _steps_of(self.steps, subjects))

    def windows(self, window_steps: int) -> TrainSource:
        """보폭 1 창으로 바꾼 것."""
        return TrainSource(self.name, self.share, make_windows(self.steps, window_steps, 1))


@dataclass(frozen=True, eq=False)
class FitResult:
    """`fit`의 결과.

    - net: 학습한 신경망(CPU, eval 모드). 검증 창을 줬으면 내부 검증 macro-F1이 가장 높았던 단위의 가중치다.
    - spec: 학습 창에서 계산한 평균·표준편차와 온도를 넣은 feature_spec. `ExportWrapper(net, spec, temperature)`가
      내보내는 모듈이다(`export_wrapper`).
    - units: 채택한 학습 길이(단위 수). 검증이 있으면 가장 좋았던 단위, 없으면 `fixed_units`. updates = units ×
      updates_per_unit. units_run은 실제로 돈 단위 수(조기 종료의 인내 구간 포함).
    - history: 단위별 내부 검증 macro-F1(검증이 없으면 빈 튜플). best_macro_f1은 그 최댓값(검증이 없으면 NaN).
    - val_logits: 채택한 단위에서의 검증 창 로짓 `[N, 3]`(온도 적용 전), val_labels: 정답. 검증이 없으면 None.
    - train_subjects, val_subjects: 학습과 검증에 쓴 피험자. 평가할 때 겹침 검사에 쓴다.
    - held_features: 표준화 공간에서 0으로 둔 특징 번호(`no_velocity` 학습이면 #7, #8, 아니면 빈 튜플).
      `predict_proba`가 평가할 때도 같은 특징을 0으로 둔다.
    - sources: 학습 표본의 출처와 몫 `((이름, 몫), ...)`. 첫 항목이 원본 추출(`native`)이고 몫의 합은 1이다.
    """

    net: nn.Module
    kind: str
    hyper: dict[str, Any]
    spec: FeatureSpec
    temperature: float
    seed: int
    units: int
    units_run: int
    updates: int
    history: tuple[float, ...]
    best_macro_f1: float
    val_logits: np.ndarray | None
    val_labels: np.ndarray | None
    n_parameters: int
    train_subjects: tuple[int, ...]
    val_subjects: tuple[int, ...]
    held_features: tuple[int, ...] = ()
    sources: tuple[tuple[str, float], ...] = ()


def _hold_at_mean(wrapper: ExportWrapper, chunk: Tensor, features: Sequence[int]) -> Tensor:
    """원시 창 텐서 `[B, T, F]`의 사본에서 주어진 특징을 래퍼의 평균으로 바꾼다. features가 비어 있으면 그대로 돌려준다.

    그 특징은 `(x − mean) / std`에서 정확히 0이 된다(같은 float32 값을 뺀다). 속도 특징 제외 실행(`no_velocity`)이
    학습, 검증, 평가에서 그 입력을 표준화 공간의 0으로 두는 방법이다. 신경망의 입력 수 F와 래퍼(내보내는 모듈)는
    그대로다.
    log1p 열에는 쓸 수 없다(변환 뒤의 평균과 원시 값이 다르다. ValueError).
    """
    if len(features) == 0:
        return chunk
    held = chunk.clone()
    for feature in features:
        if float(wrapper.log1p_mask[feature]) != 0.0:
            raise ValueError(f"feature {feature} is a log1p column and cannot be held at its mean")
        held[..., feature] = wrapper.mean[feature]
    return held


def _preprocess(
    wrapper: ExportWrapper, arrays: Sequence[np.ndarray], device: torch.device, held_features: Sequence[int] = ()
) -> Tensor:
    """원시 창을 `ExportWrapper.preprocess`(log1p, 표준화, 자르기)에 통과시켜 장치에 올린다.

    학습과 내보낸 그래프가 같은 전처리 코드를 쓴다. held_features의 특징은 먼저 평균으로 바꿔 넣는다(`_hold_at_mean`).
    """
    parts: list[Tensor] = []
    with torch.no_grad():
        for array in arrays:
            for start in range(0, array.shape[0], _CHUNK):
                chunk = torch.from_numpy(np.ascontiguousarray(array[start : start + _CHUNK])).to(device)
                parts.append(wrapper.preprocess(_hold_at_mean(wrapper, chunk, held_features)))
    return torch.cat(parts) if len(parts) > 1 else parts[0]


def _logits(net: nn.Module, z: Tensor) -> np.ndarray:
    """eval 모드의 로짓 `[N, 3]`(float64). 부른 뒤 신경망은 eval 모드로 남는다."""
    net.eval()
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, z.shape[0], _CHUNK):
            parts.append(net(z[start : start + _CHUNK]).double().cpu().numpy())
    return np.concatenate(parts)


def fit(
    train: WindowSet,
    val: WindowSet | None,
    cfg: TrainConfig,
    seed: int,
    extra_train: Sequence[TrainSource] | None = None,
) -> FitResult:
    """모델 하나를 학습한다 (PLAN 5.2절).

    - train: 학습 피험자의 창(보폭 1, 원본 추출).
    - extra_train: 같은 피험자의, 학습 전용 출처들의 창(`TrainSource`). 15 fps 누락 모사 추출(증강 (1))과 프레임
      누락 변형(증강 (2))이다. 출처마다 표본의 `share`를 거기서 뽑고 나머지를 원본에서 뽑는다(기본 설정: 0.25, 0.25,
      원본 0.5). 창이 없는 출처와 몫이 0인 출처는 쓰지 않고 그 몫은 원본으로 간다. 쓰는 출처에 창이 없는 영상의
      표본도 원본에서 뽑는다(`BalancedSampler`). 스텝 표를 만드는 일(재추출, 프레임 솎기)은 `drowsy.dataset`이 하고
      이 함수는 창만 받는다.
    - val: 내부 검증 피험자의 창(원본 추출). 주면 단위(`updates_per_unit` 갱신)마다 영상 가중 macro-F1을 계산해,
      가장 높았던 단위의 가중치를 남기고 `patience_units` 단위 동안 나아지지 않으면 멈춘다(최대 `max_units`). 온도는
      그 단위의 검증 로짓으로 맞춘다. None이면 `cfg.fixed_units` 단위를 그대로 학습하고 온도는 `cfg.temperature`다.
    - 맨 먼저 학습 피험자(모든 출처)와 검증 피험자가 겹치지 않는지 검사한다. 겹치면 `SubjectOverlapError`이고 학습을
      시작하지 않는다.
    - 표준화 상수는 학습 창(train과 extra_train)에서만, 표집기가 뽑는 것과 같은 출처별 몫으로 계산한다
      (`standardization_stats`). 검증 창은 그 상수로 변환만 한다.
    - 손실은 라벨 스무딩을 준 교차 엔트로피, 최적화는 AdamW, 증강은 표준화 공간의 가우시안 잡음(`noise_sigma`)이다.
      `cfg.variant`가 `ordinal`이면 라벨 스무딩 대신 순서형 소프트 라벨(`ordinal_targets`)을 목표로 쓰고,
      `no_velocity`면 속도 특징 #7, #8을 학습과 검증에서 표준화 공간의 0으로 둔다(`_hold_at_mean`).
      조기 종료의 지표와 온도는 어느 경우에나 원래 라벨(하나의 정답)로 계산한다.
    - 시드는 가중치 초기화, 표집, 잡음, 드롭아웃을 모두 정한다. CPU에서는 같은 시드면 같은 가중치가 나온다.
    가정: 내부 검증 macro-F1이 같으면 먼저 나온(짧은) 단위를 남긴다.
    """
    settings = cfg.settings
    if cfg.variant not in RUN_VARIANTS:
        raise ValueError(f"TrainConfig.variant must be one of {RUN_VARIANTS}, got {cfg.variant!r}")
    if isinstance(extra_train, WindowSet):
        raise TypeError("extra_train takes TrainSource items (name, share, windows), not a bare WindowSet")
    sources = [item for item in (extra_train or ()) if item.share > 0.0 and len(item.windows) > 0]
    if len({item.name for item in sources}) != len(sources):
        raise ValueError(f"training sources must have distinct names, got {[item.name for item in sources]}")
    shares = [float(item.share) for item in sources]
    if sum(shares) >= 1.0:
        raise ValueError(f"the shares of the extra training sources must sum to less than 1, got {shares}")
    train_sets = [train] + [item.windows for item in sources]
    train_subjects = tuple(sorted(set().union(*(window_subjects(item) for item in train_sets))))
    val_subjects = window_subjects(val) if val is not None else ()
    assert_subject_disjoint(train_subjects, val_subjects)
    held_features = VELOCITY_FEATURES if cfg.variant == evaluate.VARIANT_NO_VELOCITY else ()

    if settings.optimizer != OPTIMIZER_ADAMW:
        raise ValueError(f"training.optimizer must be {OPTIMIZER_ADAMW!r}, got {settings.optimizer!r}")
    if len(train) == 0:
        raise ValueError("fit needs at least one training window")
    if val is None:
        if cfg.fixed_units is None or cfg.fixed_units < 1:
            raise ValueError("fit without validation windows needs TrainConfig.fixed_units >= 1")
    else:
        if cfg.fixed_units is not None or cfg.temperature is not None:
            raise ValueError("fixed_units and temperature are for refits without validation windows")
        if len(val) == 0:
            raise ValueError("the validation set has no windows")

    seed_everything(seed)
    device = resolve_device(settings.device)
    mean, std = standardization_stats(train_sets, cfg.spec, shares)
    net = build_model(cfg.kind, cfg.spec.n_features, **cfg.hyper)
    n_parameters = count_parameters(net)
    wrapper = ExportWrapper(net, with_standardization(cfg.spec, mean, std, 1.0), 1.0).to(device)

    z_train = _preprocess(wrapper, [item.x for item in train_sets], device, held_features)
    y_train = torch.from_numpy(np.concatenate([item.y for item in train_sets])).to(device)
    sampler = BalancedSampler([item.meta for item in train_sets], seed, shares)
    noise = torch.Generator(device=device)
    noise.manual_seed(seed)

    z_val: Tensor | None = None
    scorer: VideoMacroF1 | None = None
    if val is not None:
        z_val = _preprocess(wrapper, [val.x], device, held_features)
        scorer = VideoMacroF1(val.meta)

    optimizer = torch.optim.AdamW(net.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    if cfg.variant == evaluate.VARIANT_ORDINAL:
        # 목표가 클래스 확률이다. 라벨 스무딩은 함께 쓰지 않는다(`ordinal_targets`).
        soft = ordinal_targets(len(CLASS_ORDER), settings.ordinal_neighbor_mass)
        target_train = torch.from_numpy(soft).to(dtype=torch.float32, device=device)[y_train]
        loss_fn = nn.CrossEntropyLoss()
    else:
        target_train = y_train
        loss_fn = nn.CrossEntropyLoss(label_smoothing=settings.label_smoothing)
    updates_per_unit = settings.updates_per_unit
    batch_size = settings.batch_size
    max_units = settings.max_units if val is not None else int(cfg.fixed_units or 0)

    history: list[float] = []
    best_score = -math.inf
    best_unit = 0
    best_state: dict[str, Tensor] | None = None
    best_logits: np.ndarray | None = None
    units_run = 0
    for unit in range(1, max_units + 1):
        net.train()
        drawn = sampler.draw(updates_per_unit * batch_size)
        index = torch.from_numpy(drawn).to(device).view(updates_per_unit, batch_size)
        loss = torch.zeros((), device=device)
        for step in range(updates_per_unit):
            batch = index[step]
            z = z_train[batch]
            if settings.noise_sigma > 0.0:
                z = z + settings.noise_sigma * torch.randn(z.shape, generator=noise, device=device)
                if held_features:
                    z[..., list(held_features)] = 0.0  # 0으로 둔 입력에는 잡음도 넣지 않는다
            loss = loss_fn(net(z), target_train[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        units_run = unit
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"training loss is not finite after unit {unit} ({cfg.kind}, seed {seed})")
        if z_val is None or scorer is None:
            continue
        logits = _logits(net, z_val)
        score = scorer(np.argmax(logits, axis=1))
        history.append(score)
        if score > best_score:
            best_score, best_unit, best_logits = score, unit, logits
            best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
        elif unit - best_unit >= settings.patience_units:
            break

    val_labels: np.ndarray | None = None
    if val is not None:
        if best_state is None or best_logits is None:
            raise RuntimeError("validation macro-F1 was never finite; nothing to keep")
        net.load_state_dict(best_state)
        temperature = fit_temperature(best_logits, val.y)
        units = best_unit
        val_labels = np.asarray(val.y, dtype=np.int64)
    else:
        temperature = 1.0 if cfg.temperature is None else float(cfg.temperature)
        units = units_run
    net = net.cpu().eval()
    return FitResult(
        net=net,
        kind=cfg.kind,
        hyper=dict(cfg.hyper),
        spec=with_standardization(cfg.spec, mean, std, temperature),
        temperature=temperature,
        seed=seed,
        units=units,
        units_run=units_run,
        updates=units * updates_per_unit,
        history=tuple(history),
        best_macro_f1=best_score if val is not None else math.nan,
        val_logits=best_logits,
        val_labels=val_labels,
        n_parameters=n_parameters,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        held_features=tuple(held_features),
        sources=((evaluate.FEATURE_SOURCE_NATIVE, 1.0 - sum(shares)),)
        + tuple((item.name, float(item.share)) for item in sources),
    )


def export_wrapper(result: FitResult) -> ExportWrapper:
    """학습 결과로 내보내기용 래퍼(원시 특징 → 확률)를 만든다. 평가와 ONNX 내보내기가 같은 모듈을 쓴다."""
    return ExportWrapper(result.net, result.spec, result.temperature)


def predict_proba(result: FitResult, windows: WindowSet, device: torch.device | str = "cpu") -> np.ndarray:
    """학습에 쓰이지 않은 피험자의 창에 대한 확률 `[N, 3]`(float64, 온도 적용 뒤).

    평가 전에 `assert_subject_disjoint`로 이 창의 피험자가 그 모델의 학습·검증 피험자와 겹치지 않는지 검사한다.
    겹치면 `SubjectOverlapError`다. 추론은 `export_wrapper(result)`(내보내는 것과 같은 모듈)로 한다.
    `result.held_features`가 있으면(속도 특징 제외 학습) 그 특징을 학습 때처럼 표준화 공간의 0으로 두고 추론한다.
    """
    assert_subject_disjoint(set(result.train_subjects) | set(result.val_subjects), window_subjects(windows))
    wrapper = export_wrapper(result).to(device).eval()
    parts: list[np.ndarray] = [np.zeros((0, len(CLASS_ORDER)), dtype=np.float64)]
    with torch.no_grad():
        for start in range(0, len(windows), _CHUNK):
            chunk = torch.from_numpy(np.ascontiguousarray(windows.x[start : start + _CHUNK])).to(device)
            chunk = _hold_at_mean(wrapper, chunk, result.held_features)
            parts.append(wrapper(chunk).double().cpu().numpy())
    result.net.cpu()  # 래퍼가 같은 신경망을 장치로 옮겼다. 결과 객체는 CPU에 둔다
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# 안쪽 선택
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Candidate:
    """선택 후보 하나: 계열과 하이퍼파라미터. name은 기록과 로그에 쓰는 이름이다."""

    family: str
    hyper: dict[str, Any]
    name: str


def candidates(settings: TrainingSettings, families: Sequence[str] = KINDS) -> tuple[Candidate, ...]:
    """사전에 고정한 탐색 범위의 후보들 (PLAN 5.2절): TCN 채널 × 드롭아웃, GRU 은닉 × 드롭아웃."""
    out: list[Candidate] = []
    for family in families:
        if family == KIND_TCN:
            for channels in settings.tcn.channels:
                for dropout in settings.tcn.dropout:
                    hyper = {
                        "channels": int(channels),
                        "kernel_size": int(settings.tcn.kernel_size),
                        "dilations": [int(value) for value in settings.tcn.dilations],
                        "dropout": float(dropout),
                    }
                    out.append(Candidate(KIND_TCN, hyper, f"tcn(channels={channels},dropout={dropout:g})"))
        elif family == KIND_GRU:
            for hidden in settings.gru.hidden:
                for dropout in settings.gru.dropout:
                    hyper = {"hidden": int(hidden), "dropout": float(dropout)}
                    out.append(Candidate(KIND_GRU, hyper, f"gru(hidden={hidden},dropout={dropout:g})"))
        else:
            raise ValueError(f"unknown model family {family!r}, expected one of {KINDS}")
    if len({candidate.name for candidate in out}) != len(out):
        raise ValueError("the search grid lists the same candidate twice")
    return tuple(out)


@dataclass(frozen=True, slots=True)
class CandidateScore:
    """후보 하나의 안쪽 검증 결과.

    - val_folds: 검증으로 쓴 fold 번호들. fold_macro_f1, fold_units: 그 fold별 내부 검증 macro-F1(조기 종료가 고른
      단위의 값)과 그 단위 수.
    - mean_macro_f1, std_macro_f1: fold별 값의 평균과 표본 표준편차(fold가 하나면 0).
    - units: 다시 학습할 때의 학습 길이(단위 수). temperature: 다시 학습한 모델에 넣을 온도.
    """

    candidate: Candidate
    n_parameters: int
    val_folds: tuple[int, ...]
    fold_macro_f1: tuple[float, ...]
    mean_macro_f1: float
    std_macro_f1: float
    fold_units: tuple[int, ...]
    units: int
    temperature: float


def pooled_temperature(results: Sequence[FitResult]) -> float:
    """안쪽 fold들의 검증 로짓을 모아 온도 하나를 맞춘다 (PLAN 5.2절, 6장).

    각 결과의 검증 로짓은 그 모델이 학습에 쓰지 않은 피험자의 것이다. 시험 fold의 창은 들어 있지 않다.
    """
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for result in results:
        if result.val_logits is None or result.val_labels is None:
            raise ValueError("pooled_temperature needs fits that had validation windows")
        logits.append(result.val_logits)
        labels.append(result.val_labels)
    if not logits:
        raise ValueError("pooled_temperature needs at least one fit")
    return fit_temperature(np.concatenate(logits), np.concatenate(labels))


def score_candidate(
    candidate: Candidate,
    train_windows: WindowSet,
    val_windows: WindowSet,
    extra_windows: Sequence[TrainSource] | None,
    splits: Sequence[InnerSplit],
    base: TrainConfig,
    seed: int,
) -> CandidateScore:
    """후보 하나를 안쪽 분할마다 학습해 점수를 낸다 (PLAN 6장).

    분할마다 `fit(학습 피험자의 창, 검증 피험자의 창)`을 한 번 부른다. 학습 전용 출처(extra_windows)에서는 분할마다
    그 분할의 학습 피험자의 창만 골라 넘긴다. 학습 길이는 분할별로 조기 종료가 고른 단위
    수의 중앙값, 온도는 분할들의 검증 로짓을 모아 맞춘 값이다.
    가정: 계획서는 분할별 학습 길이를 하나로 합치는 방법을 적지 않았다. 중앙값(짝수 개면 가운데 두 값의 평균을
    반올림)을 쓴다.
    가정: 후보의 점수는 분할마다 "그 분할에서 가장 좋았던 단위"의 macro-F1을 평균한 값이다(조기 종료의 결과 그대로).
    """
    if len(splits) == 0:
        raise ValueError("score_candidate needs at least one inner split")
    cfg = dataclasses.replace(base, kind=candidate.family, hyper=candidate.hyper, fixed_units=None, temperature=None)
    results: list[FitResult] = []
    for split in splits:
        assert_subject_disjoint(split.train_subjects, split.val_subjects)
        extra = tuple(item.of(split.train_subjects) for item in (extra_windows or ()))
        results.append(
            fit(
                subset_subjects(train_windows, split.train_subjects),
                subset_subjects(val_windows, split.val_subjects),
                cfg,
                seed,
                extra_train=extra,
            )
        )
    scores = [float(result.best_macro_f1) for result in results]
    return CandidateScore(
        candidate=candidate,
        n_parameters=results[0].n_parameters,
        val_folds=tuple(int(split.val_fold) for split in splits),
        fold_macro_f1=tuple(scores),
        mean_macro_f1=float(np.mean(scores)),
        std_macro_f1=float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        fold_units=tuple(int(result.units) for result in results),
        units=median_half_up([result.units for result in results]),
        temperature=pooled_temperature(results),
    )


_Scored = TypeVar("_Scored")


def choose(
    scores: Sequence[_Scored],
    complexity: Callable[[_Scored], tuple[float, ...]],
    mean: Callable[[_Scored], float],
    std: Callable[[_Scored], float],
) -> _Scored:
    """안쪽 검증 점수로 후보 하나를 고른다 (PLAN 6장의 선택 기준과 동률 규칙).

    평균이 가장 높은 후보를 찾고, 그 후보와의 차이가 내부 fold 표준편차 안인 후보들 중 가장 단순한 것을 고른다.
    complexity가 작을수록 단순하다(신경망은 파라미터 수). 단순함도 같으면 평균이 높은 쪽, 그것도 같으면 목록에서
    앞선 쪽이다.
    가정: "내부 fold 표준편차"는 평균이 가장 높은 후보의 fold별 값의 표본 표준편차다.
    """
    if len(scores) == 0:
        raise ValueError("nothing to choose from")
    means = [float(mean(score)) for score in scores]
    if not all(math.isfinite(value) for value in means):
        raise ValueError("inner-validation scores must be finite")
    top = max(range(len(scores)), key=lambda index: means[index])  # 같으면 앞선 후보
    margin = float(std(scores[top]))
    eligible = [index for index in range(len(scores)) if means[top] - means[index] <= margin]
    best = min(eligible, key=lambda index: (complexity(scores[index]), -means[index]))
    return scores[best]


def choose_candidate(scores: Sequence[CandidateScore]) -> CandidateScore:
    """신경망 후보의 선택: `choose`에 파라미터 수를 단순함의 기준으로 준 것."""
    return choose(
        scores,
        complexity=lambda score: (float(score.n_parameters),),
        mean=lambda score: score.mean_macro_f1,
        std=lambda score: score.std_macro_f1,
    )


def _score_record(score: CandidateScore) -> dict[str, Any]:
    return {
        "name": score.candidate.name,
        "family": score.candidate.family,
        "hyper": score.candidate.hyper,
        "n_parameters": score.n_parameters,
        "val_folds": list(score.val_folds),
        "fold_macro_f1": [_finite_or_none(value) for value in score.fold_macro_f1],
        "mean_macro_f1": _finite_or_none(score.mean_macro_f1),
        "std_macro_f1": _finite_or_none(score.std_macro_f1),
        "fold_units": list(score.fold_units),
        "units": score.units,
        "temperature": score.temperature,
    }


@dataclass(frozen=True, eq=False)
class _ScoreJob:
    """안쪽 선택 작업 하나(후보 하나 × 안쪽 분할 전부).

    steps와 extra_steps에는 이 작업의 학습·검증 피험자의 스텝만 넣는다. 시험 fold의 스텝은 들어오지 않는다.
    extra_steps는 학습 전용 출처들(15 fps 누락 모사 추출, 프레임 누락 변형)이다.
    """

    candidate: Candidate
    splits: tuple[InnerSplit, ...]
    steps: pd.DataFrame
    extra_steps: tuple[StepSource, ...]
    base: TrainConfig
    window_steps: int
    eval_stride: int
    seed: int


def _score_job(job: _ScoreJob) -> CandidateScore:
    train_windows = make_windows(job.steps, job.window_steps, 1)
    val_windows = (
        train_windows if job.eval_stride == 1 else make_windows(job.steps, job.window_steps, job.eval_stride)
    )
    extra_windows = tuple(source.windows(job.window_steps) for source in job.extra_steps)
    return score_candidate(
        job.candidate, train_windows, val_windows, extra_windows, job.splits, job.base, job.seed
    )


def _steps_of(steps: pd.DataFrame, subjects: Iterable[int]) -> pd.DataFrame:
    wanted = sorted({int(value) for value in subjects})
    return steps[steps["subject"].isin(wanted)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class CVData:
    """교차검증과 최종 학습에 쓰는 표 묶음.

    - feature_spec: 기준 feature_spec(`paths.feature_spec`).
    - manifest: 세션 목록. fold는 이 표의 `fold` 열(공식 대응)이다. raw_dir: 추출 결과(RawFrame parquet)의 루트.
    - tables: 정규화 조건 → 스텝 표(원본 프레임률 추출).
    - extra_tables: 정규화 조건 → 두 번째 추출(15 fps 누락 모사)의 스텝 표. 없으면 빈 dict다. 학습 조건의 표는 학습
      표본에 섞고(증강 (1)), 시험 fold의 15 fps 평가(부가 결과)에도 쓴다.
    - synthetic, description: 합성 데이터인지와 한 줄 설명(`DatasetSource`의 값). 회전이 확인되지 않은 영상이 있으면
      데이터 설명 바로 뒤에 그 수가 붙어 있다(`rotation_note`). 리포트의 설명은 모두 이 문장으로 시작한다.
    - frame_drop_tables: 정규화 조건 → 프레임 누락 변형들의 스텝 표(증강 (2), 학습 전용). 없으면 빈 dict다.
    - rotation: 원본 추출 결과의 회전 점검 기록을 모은 것(`dataset.dataset_rotation_checks`). 손으로 만든 묶음(테스트)은 None이다.
    학습에 섞는 출처와 몫은 `mix_sources`가 이 표들과 설정에서 만든다.
    """

    feature_spec: FeatureSpec
    manifest: pd.DataFrame
    raw_dir: Path
    tables: dict[str, StepTable]
    extra_tables: dict[str, StepTable]
    synthetic: bool
    description: str
    frame_drop_tables: dict[str, tuple[StepTable, ...]] = dataclasses.field(default_factory=dict)
    rotation: RotationSummary | None = None


def frame_drop_source_name(table: StepTable, position: int) -> str:
    """프레임 누락 변형 표의 출처 이름 `frame_drop_v{변형 번호}`. 변형 설정이 없는 표(손으로 만든 표)는 목록에서의 순서다."""
    variant = table.frame_drop.variant if table.frame_drop is not None else position
    return f"{SOURCE_FRAME_DROP_PREFIX}{variant}"


def mix_sources(data: CVData, cfg: ExperimentConfig, arm: str) -> tuple[StepSource, ...]:
    """학습 조건 arm의 학습 전용 출처들과 그 표집 몫 (PLAN 5.2절 증강 (1), (2)).

    - 15 fps 누락 모사 추출(`data.extra_tables[arm]`): 몫 `training.fps_sim_share`.
    - 프레임 누락 변형(`data.frame_drop_tables[arm]`): 몫 `training.frame_drop_share`를 변형들이 똑같이 나눈다.
    표가 없거나 몫이 0인 출처는 넣지 않는다. 그 몫은 원본 추출로 간다(`fit`이 남은 몫을 원본에서 뽑는다).
    돌려주는 표에는 목록의 모든 피험자가 들어 있다. 학습 피험자만 골라 넘기는 것은 부르는 쪽이 한다
    (`StepSource.of`, `TrainSource.of`).
    가정: 몫의 기본값(원본 0.5, 15 fps 모사 0.25, 프레임 누락 변형 0.25)은 계획서에 없는 값이다.
    """
    sources: list[StepSource] = []
    if cfg.training.fps_sim_share > 0.0 and arm in data.extra_tables:
        sources.append(StepSource(SOURCE_FPS_SIM, float(cfg.training.fps_sim_share), data.extra_tables[arm].steps))
    variants = data.frame_drop_tables.get(arm, ())
    if cfg.training.frame_drop_share > 0.0 and len(variants) > 0:
        share = float(cfg.training.frame_drop_share) / len(variants)
        for position, table in enumerate(variants):
            sources.append(StepSource(frame_drop_source_name(table, position), share, table.steps))
    return tuple(sources)


def source_shares(sources: Sequence[StepSource]) -> dict[str, float]:
    """출처 이름 → 표집 몫. 원본 추출(`native`)이 남은 몫을 갖는다. 실행 기록에 적는 값이다."""
    shares = {evaluate.FEATURE_SOURCE_NATIVE: 1.0 - sum(source.share for source in sources)}
    shares.update({source.name: source.share for source in sources})
    return shares


def open_fps_sim(cfg: ExperimentConfig) -> DatasetSource | None:
    """두 번째 추출(15 fps 누락 모사)이 있으면 그 데이터를, 없으면 None을 돌려준다.

    위치는 `dataset.open_dataset(cfg, fps_sim=True)`가 찾는 곳과 같다: 실제 데이터는 `raw_dir/{drop_sim.subdir}`,
    합성 데이터는 `{synth_dir}_{drop_sim.subdir}`. 폴더가 없을 때만 None이고, 있는데 읽을 수 없으면 예외가 그대로 난다.
    """
    subdir = cfg.extraction.drop_sim.subdir
    if cfg.paths.rldd_root is not None:
        root = cfg.paths.raw_dir / subdir
    else:
        root = cfg.paths.synth_dir.with_name(f"{cfg.paths.synth_dir.name}_{subdir}")
    if not root.is_dir():
        return None
    return open_dataset(cfg, fps_sim=True)


def rotation_note(rotation: RotationSummary | None) -> str:
    """리포트 설명에 붙이는 말: 회전이 확인되지 않은 추출 결과의 수. 그런 파일이 없으면(기록을 읽지 않았어도) 빈 문자열이다.

    "확인되지 않았다"는 것은 `dataset.read_rotation_checks`가 확인 필요로 올린 파일이다: 추출의 회전 자동 점검이 각도를
    가르지 못했는데 수동 회전 보정(`extraction.rotation_overrides`)이 없는 것과, 보정표의 각도와 다른 각도로 추출돼 있는 것.
    목록은 실행 기록의 `rotation_check`(`rotation_record`)에 있고 명령이 시작할 때도 출력한다.
    """
    if rotation is None or not rotation.flagged:
        return ""
    return (
        f"; ROTATION NOT CONFIRMED for {len(rotation.flagged)} of {rotation.n_files} raw file(s) "
        f"({rotation.n_ambiguous} where the automatic rotation check of the extraction could not tell the angle and "
        f"no manual override exists, {rotation.n_override_not_applied} extracted at another angle than their manual "
        "override): these videos may be rotated wrongly and they are included in every number below as they were "
        "extracted; look at them and fill `" + OVERRIDES_KEY + "` (list: key `rotation_check` of the run record in "
        "this folder, or `python -m drowsy.dataset`)"
    )


def rotation_record(rotation: RotationSummary | None) -> dict[str, Any] | None:
    """실행 기록(run.json, final.json, probes.json)의 `rotation_check` 값. 기록을 읽지 않은 묶음이면 None이다.

    `signal_check e0`가 criteria.json에 적는 것과 같은 모양이다: 보정표의 설정 키(`overrides_key`), 구분별 파일 수
    (`n_files`, `n_missing`, `decided_by_check`, `manual_override`, `ambiguous_without_override`,
    `override_not_applied`, `no_record`), 확인이 필요한 영상의 목록(`flagged`).
    """
    return None if rotation is None else {"overrides_key": OVERRIDES_KEY, **rotation.as_dict()}


def print_rotation_check(cfg: ExperimentConfig, source: DatasetSource) -> RotationSummary:
    """추출 결과의 회전 점검 기록을 읽어 한 줄로 출력하고(`dataset.rotation_check_line`), 읽은 것을 돌려준다.

    `python -m drowsy.train`과 `python -m drowsy.baseline_lgbm`의 모든 하위 명령이 데이터 설명 다음 줄에, 학습이나 재생을
    시작하기 전에 부른다. 확인이 필요한 영상이 있으면 그 아래에 영상마다 한 줄씩 적는다(`dataset.rotation_flag_lines`.
    `python -m drowsy.dataset`의 출력과 같은 줄이다). 합성 데이터에는 점검 기록이 없으므로 모든 파일이 "without a
    rotation-check record"로 세어진 한 줄만 나온다.
    가정: 알리기만 하고 멈추지 않는다. 회전이 확인되지 않은 영상을 빼거나 학습을 막는 것은 소유자의 결정을 기다리는 항목이다
    (`python -m drowsy.dataset`도 알리기만 한다).
    가정: 원본 프레임률 추출의 기록만 읽는다. 15 fps 누락 모사 추출은 같은 영상을 같은 점검 설정으로 한 번 더 추출한 것이라
    기록이 같을 것으로 본다. 두 추출 사이에 보정표를 고치고 한쪽만 다시 추출한 경우는 여기서 드러나지 않는다(그 추출의
    기록은 `python -m drowsy.dataset --fps-sim`이 출력한다).
    """
    rotation = dataset_rotation_checks(cfg, source)
    print(rotation_check_line(rotation, "the lines below"))
    for line in rotation_flag_lines(rotation, OVERRIDES_KEY):
        print(line)
    sys.stdout.flush()
    return rotation


def _sessions_with_windows(quality: pd.DataFrame) -> set[str]:
    """품질 표에서 창이 하나라도 나오는 세션의 `session_id`."""
    return {str(value) for value in quality.loc[quality["n_windows"].to_numpy() > 0, "session_id"].tolist()}


def _listed(items: Sequence[str], limit: int = 10) -> str:
    shown = ", ".join(items[:limit])
    return shown if len(items) <= limit else f"{shown}, ... ({len(items)} in total)"


def load_cv_data(
    cfg: ExperimentConfig,
    arms: Sequence[str],
    mix_arms: Sequence[str] = (),
    pipeline: PipelineSpec | None = None,
    *,
    fps_eval_arms: Sequence[str] = (),
    allow_partial_fps_sim: bool = False,
    rebuild: bool = False,
) -> CVData:
    """설정이 가리키는 데이터의 스텝 표를 읽는다(없으면 만들어 캐시에 둔다).

    - arms: 읽을 정규화 조건들.
    - mix_arms: 학습 전용 출처의 스텝 표도 읽을 조건들(학습 조건). arms에 들어 있는 조건이어야 한다.
      - 두 번째 추출(15 fps 누락 모사)이 있고 `training.fps_sim_share`가 0보다 크면 그 스텝 표(`extra_tables`).
      - `training.frame_drop_share`가 0보다 크고 `training.frame_drop_variants`가 1 이상이면 프레임 누락 변형의
        스텝 표(`frame_drop_tables`). 원본 추출의 프레임을 솎아 다시 재생한 것이고(`dataset.load_frame_drop_tables`,
        시드 `FRAME_DROP_SEED`, 비율 `training.frame_drop_min`~`frame_drop_max`), 변형마다 따로 캐시된다.
    - fps_eval_arms: 두 번째 추출이 있으면 학습에 섞지 않더라도 그 스텝 표를 읽을 조건들(시험 fold의 15 fps 평가용.
      `training.fps_sim_share`가 0이어도 읽는다). arms에 들어 있어야 한다.
    - pipeline: 스텝 표를 만들 때 쓸 상수. None이면 feature_spec의 것이다(closure W 비교처럼 상수를 바꿔 볼 때만 준다).
    - allow_partial_fps_sim: 두 번째 추출이 원본 추출의 세션을 다 덮지 못해도 받아들인다(아래).
    - rebuild: 스텝 표 캐시를 읽지 않고 다시 만든다(`dataset.load_step_tables`의 rebuild).
    `paths.rldd_root`가 있으면 실제 데이터, null이면 합성 데이터다(`dataset.open_dataset`).
    원본 추출 결과의 회전 점검 기록도 읽어 `rotation`에 두고(`dataset.dataset_rotation_checks`), 회전이 확인되지 않은 파일이
    있으면 그 수를 설명에 붙인다(`rotation_note`). 이 묶음으로 만드는 리포트(교차검증, 최종 모델, LightGBM, 교란 점검)의
    설명은 모두 이 설명으로 시작하므로 그 말이 함께 실린다.

    설정과 데이터가 맞는지도 여기서 본다(어긋나면 ValueError).
    - `evaluation.subjects_per_fold`는 공식 fold 대응의 상수(`dataset.SUBJECTS_PER_FOLD`, 12)와 같아야 한다. fold는
      그 상수로 정해지고 설정 값으로 바뀌지 않는다.
    - 실제 데이터에서는 목록의 fold 수가 `evaluation.n_folds`와 같아야 한다.
      가정: 합성 데이터에서는 fold 수를 보지 않는다(테스트와 연기 시험은 피험자 몇 명, fold 두세 개로 돈다).
    - 두 번째 추출이 있으면, 원본 추출에서 창이 나오는 세션은 두 번째 추출에서도 창이 나와야 한다. 덜 끝난 추출
      폴더로 학습하는 일을 막는다. 일부만으로 학습하려면 allow_partial_fps_sim을 준다. 그때는 그 추출이 없는 영상의
      표본을 원본에서 뽑고(`BalancedSampler`), 덮은 세션 수를 설명에 적는다.
      가정: 프레임이 덜 들어온 추출에서는 보정이 늦게 끝나 원본에서만 창이 나오는 영상이 정상적으로 있을 수 있다.
      그런 경우에도 명시적으로 allow_partial_fps_sim을 주게 한다(조용히 넘어가지 않는다).
    - 프레임 누락 변형에는 그 검사를 하지 않는다.
      가정: 변형은 이 호출이 원본 추출의 같은 파일에서 같은 목록으로 만든다. "덜 끝난 추출 폴더"가 있을 수 없으므로
      세션 수를 대조할 이유가 없다. 솎은 뒤 창이 나오지 않는 세션(끝 스텝이 빠져 조회 구간이 모자라게 된 짧은 세션
      등)은 있을 수 있고, 그 영상의 표본은 원본에서 뽑는다(`BalancedSampler`). 덮은 세션 수는 설명에 적는다.
    """
    feature_spec = load_feature_spec(cfg.paths.feature_spec)
    if cfg.windows.window_steps != feature_spec.window_steps:
        raise ValueError(
            f"windows.window_steps is {cfg.windows.window_steps} but the feature spec has {feature_spec.window_steps}"
        )
    for arm in tuple(arms) + tuple(mix_arms) + tuple(fps_eval_arms):
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    stray = [arm for arm in mix_arms if arm not in arms]
    if stray:
        raise ValueError(f"mix_arms must be among arms, got {stray} outside {list(arms)}")
    stray = [arm for arm in fps_eval_arms if arm not in arms]
    if stray:
        raise ValueError(f"fps_eval_arms must be among arms, got {stray} outside {list(arms)}")
    if cfg.evaluation.subjects_per_fold != SUBJECTS_PER_FOLD:
        raise ValueError(
            f"evaluation.subjects_per_fold is {cfg.evaluation.subjects_per_fold} but the official fold mapping is "
            f"fixed at {SUBJECTS_PER_FOLD} subjects per fold (dataset.official_fold)"
        )
    spec = feature_spec.pipeline if pipeline is None else pipeline
    source = open_dataset(cfg)
    n_folds = int(source.manifest["fold"].nunique())
    if not source.synthetic and n_folds != cfg.evaluation.n_folds:
        raise ValueError(
            f"the dataset has {n_folds} fold(s) but evaluation.n_folds is {cfg.evaluation.n_folds}; "
            "the pre-registered procedure uses every official fold (set evaluation.n_folds if the subset is intended)"
        )
    gate = cfg.evaluation.quality_gate
    workers = cfg.extraction.workers
    tables = load_step_tables(
        source.manifest, source.raw_dir, spec, tuple(arms), gate=gate, workers=workers, rebuild=rebuild
    )
    extra_tables: dict[str, StepTable] = {}
    rotation = dataset_rotation_checks(cfg, source)
    description = source.description + rotation_note(rotation)
    mixing_fps = len(mix_arms) > 0 and cfg.training.fps_sim_share > 0.0
    fps_arms = tuple(dict.fromkeys((tuple(mix_arms) if mixing_fps else ()) + tuple(fps_eval_arms)))
    if len(fps_arms) > 0:
        extra = open_fps_sim(cfg)
        if extra is not None:
            unknown = sorted(set(extra.manifest["subject"].tolist()) - set(source.manifest["subject"].tolist()))
            if unknown:
                raise ValueError(f"the frame-drop simulated extraction holds subjects missing from the data: {unknown}")
            extra_tables = load_step_tables(
                extra.manifest, extra.raw_dir, spec, fps_arms, gate=gate, workers=workers, rebuild=rebuild
            )
            first = fps_arms[0]
            native_sessions = _sessions_with_windows(tables[first].quality)
            missing = sorted(native_sessions - _sessions_with_windows(extra_tables[first].quality))
            covered = len(native_sessions) - len(missing)
            if missing and not allow_partial_fps_sim:
                raise ValueError(
                    f"the frame-drop simulated extraction at {extra.raw_dir} has windows for only {covered} of the "
                    f"{len(native_sessions)} sessions that have windows in the native extraction (missing: "
                    f"{_listed(missing)}). Finish that extraction, or pass --allow-partial-fps-sim to train with "
                    "the sessions it holds"
                )
            coverage = f"windows for {covered} of {len(native_sessions)} sessions"
            if mixing_fps:
                description += (
                    f"; training mixes in the frame-drop simulated extraction at {extra.raw_dir} "
                    f"(share {cfg.training.fps_sim_share:g}, {coverage}"
                    + (", PARTIAL: the other sessions are drawn from the native extraction only" if missing else "")
                    + ")"
                )
            else:
                description += (
                    f"; the frame-drop simulated extraction at {extra.raw_dir} is not mixed into training "
                    f"and serves the supplementary 15 fps evaluation only ({coverage}"
                    + (", PARTIAL: videos without windows there count as wrong in that evaluation" if missing else "")
                    + ")"
                )
    frame_drop_tables: dict[str, tuple[StepTable, ...]] = {}
    n_variants = int(cfg.training.frame_drop_variants)
    if len(mix_arms) > 0 and cfg.training.frame_drop_share > 0.0 and n_variants > 0:
        built = [
            load_frame_drop_tables(
                source.manifest,
                source.raw_dir,
                spec,
                tuple(mix_arms),
                variant=variant,
                seed=FRAME_DROP_SEED,
                drop_min=cfg.training.frame_drop_min,
                drop_max=cfg.training.frame_drop_max,
                gate=gate,
                workers=workers,
                rebuild=rebuild,
            )
            for variant in range(n_variants)
        ]
        frame_drop_tables = {arm: tuple(tables_of[arm] for tables_of in built) for arm in mix_arms}
        first = mix_arms[0]
        native_sessions = _sessions_with_windows(tables[first].quality)
        covered = [
            len(native_sessions & _sessions_with_windows(table.quality)) for table in frame_drop_tables[first]
        ]
        description += (
            f"; training mixes in {n_variants} frame-drop variant(s) of the native extraction (PLAN 5.2 "
            f"augmentation 2, an approximation: stored frames dropped at a per-session rate within "
            f"{cfg.training.frame_drop_min:g}-{cfg.training.frame_drop_max:g}, seed {FRAME_DROP_SEED}, share "
            f"{cfg.training.frame_drop_share:g} in total, windows for "
            f"{', '.join(str(count) for count in covered)} of {len(native_sessions)} sessions)"
        )
    return CVData(
        feature_spec=feature_spec,
        manifest=source.manifest,
        raw_dir=source.raw_dir,
        tables=dict(tables),
        extra_tables=extra_tables,
        synthetic=source.synthetic,
        description=description,
        frame_drop_tables=frame_drop_tables,
        rotation=rotation,
    )


def _base_train_config(
    cfg: ExperimentConfig, spec: FeatureSpec, variant: str = evaluate.VARIANT_BASE
) -> TrainConfig:
    return TrainConfig(kind=KIND_TCN, hyper={}, spec=spec, settings=cfg.training, variant=variant)


# ---------------------------------------------------------------------------
# fold 모델의 INT8 비교 (D16)
# ---------------------------------------------------------------------------

FOLD_MODEL_VERSION = 0  # 임시 폴더에 쓰는 fold 모델 파일의 버전 번호. 파일은 평가가 끝나면 지운다
FOLD_MODEL_TRAINED_ON = "cross-validation fold model (temporary file, not for deployment)"
FOLD_MODEL_COMMIT = "not-exported"  # 임시 파일이라 git 상태를 읽지 않는다


def export_fold_model(result: FitResult, base_spec_path: Path, work_dir: Path) -> tuple[Path, Path]:
    """다시 학습한 fold 모델을 work_dir에 float ONNX와 INT8 ONNX로 쓴다. `(float 파일, INT8 파일)` (PLAN 7장, D16).

    `export_fit`과 같은 길을 쓴다: 학습한 평균·표준편차·온도를 넣은 feature_spec 사본을 쓰고(`write_trained_spec`),
    그 파일에서 읽은 값으로 만든 래퍼를 `export_onnx.export`로 내보낸 뒤(계약 검사 포함) `export_onnx.quantize_int8`로
    양자화하고 INT8 파일의 계약 검사(`ALLOWED_OPS_INT8`)를 한다. 어긋나면 `ContractError`(ValueError)다.
    work_dir은 부르는 쪽이 만든 임시 폴더여야 한다. 이 파일들은 배포물이 아니다(메타데이터의 trained_on에 그렇게 적는다).
    """
    work_dir = Path(work_dir)
    scratch = work_dir / "spec"
    scratch.mkdir(parents=True, exist_ok=True)
    trained_spec = scratch / export_onnx.exported_spec_name(FOLD_MODEL_VERSION)
    write_trained_spec(base_spec_path, result, trained_spec)
    spec = load_feature_spec(trained_spec)
    wrapper = ExportWrapper(result.net, spec, spec.standardization.temperature)
    float_path = work_dir / export_onnx.model_file_name(FOLD_MODEL_VERSION)
    meta = {"model_version": str(FOLD_MODEL_VERSION), "trained_on": FOLD_MODEL_TRAINED_ON, "commit": FOLD_MODEL_COMMIT}
    export_onnx.export(wrapper, trained_spec, float_path, meta)
    int8_path = export_onnx.quantize_int8(
        float_path, work_dir / export_onnx.model_file_name(FOLD_MODEL_VERSION, int8=True)
    )
    export_onnx.verify_contract(
        int8_path, work_dir / export_onnx.exported_spec_name(FOLD_MODEL_VERSION), export_onnx.ALLOWED_OPS_INT8
    )
    return float_path, int8_path


def onnx_predict(onnx_path: Path, x: np.ndarray, intra_op_threads: int = 1) -> np.ndarray:
    """ONNX 파일을 onnxruntime(CPU)으로 돌린 확률 `[N, 3]`(float64). 앱처럼 창 하나씩(배치 1) 넣는다.

    x는 원시 창 `[N, 120, 17]`이다. 계약의 입력 이름과 출력 이름(`features`, `probs`)을 쓴다.
    """
    windows = np.ascontiguousarray(x, dtype=np.float32)
    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op_threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
    out = np.zeros((windows.shape[0], len(CLASS_ORDER)), dtype=np.float64)
    for index in range(windows.shape[0]):
        feed = {export_onnx.INPUT_NAME: windows[index : index + 1]}
        out[index] = session.run([export_onnx.OUTPUT_NAME], feed)[0][0]
    return out


# ---------------------------------------------------------------------------
# 중첩 교차검증
# ---------------------------------------------------------------------------


def _video_tables_match(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    columns = list(evaluate.VIDEO_COLUMNS)
    return a[columns].reset_index(drop=True).equals(b[columns].reset_index(drop=True))


def cross_validate(
    data: CVData,
    cfg: ExperimentConfig,
    arm: str,
    families: Sequence[str],
    run_id: str,
    *,
    variant: str = evaluate.VARIANT_BASE,
    fold_int8: bool = False,
    notes: dict[str, Any] | None = None,
    supplementary: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, float]]:
    """PLAN 6장의 중첩 교차검증. `(예측 표, 바깥 fold별 선택 기록, 단계별 소요 시간)`을 돌려준다.

    바깥 fold k마다
      1. 시험 fold를 뺀 나머지 fold 안에서 검증 fold를 하나씩 돌려(`FoldSplit.inner`) 후보를 모두 학습한다.
         안쪽 학습의 시드는 `training.seeds`의 첫 값 하나다.
      2. 선택(`choose_candidate`): 계열별로 하나씩, 그리고 계열을 가리지 않고 하나(주 결과 행 `selected`).
      3. 고른 후보마다 학습 fold 전체로, 고른 학습 길이와 온도로, `training.seeds`의 시드마다 다시 학습한다.
      4. 시험 fold를 평가 보폭(1초)으로 한 번 평가한다. 학습 조건이 alert 영상 기준선이고 자기 세션 기준선 스텝 표가
         있으면, 같은 모델을 시험 fold의 자기 세션 기준선 특징으로도 평가한다(앱 조건 평가, 재학습 없음).
    예측 표의 묶음: `(run_id, 계열, arm)`과 `(run_id, selected, arm)`, 앱 조건 평가는 arm만 `session_baseline`이다.
    시험 fold는 1~3단계의 어떤 호출에도 들어가지 않는다(안쪽 선택 작업에는 시험 fold의 스텝을 넣지 않는다).

    학습 표본의 출처(`mix_sources`): 원본 추출에 15 fps 누락 모사 추출과 프레임 누락 변형을 섞는다. 어느 출처든
    1단계에는 그 안쪽 분할의 학습 피험자, 3단계에는 그 바깥 fold의 학습 피험자의 창만 넘긴다. 검증과 시험, 온도는
    원본 추출의 창만 쓴다.

    부가 묶음(리포트 9절에만 실린다. PLAN 11장 E3, D16)
    - variant가 `base`가 아니면(`ordinal`, `no_velocity`) 그 선택 사항으로 1~4단계를 그대로 돌고, 모든 묶음에 그
      `variant` 표지를 단다(전부 부가 묶음). 이 경우 아래 두 평가는 하지 않는다.
    - 15 fps 모사 평가: `data.extra_tables`에 그 추출의 스텝 표가 있는 평가 조건마다, 같은 fold 모델을 시험 fold의
      15 fps 추출 특징으로도 평가한다(`feature_source = fps15`, 재학습 없음). 원본 추출에는 창이 있는데 15 fps
      추출에는 창이 하나도 없는 시험 fold가 있으면 그 조건의 15 fps 평가는 통째로 건너뛴다(평가하지 않은 fold를
      오답으로 세지 않는다).
    - fold_int8(D16): 계열별 선택의 TCN 모델(모든 fold에 있다)을 다시 학습할 때마다 임시 폴더에 내보내 양자화하고
      (`export_fold_model`), float 파일과 INT8 파일을 onnxruntime으로 시험 fold의 원본 추출 특징에 돌린다. 계열 이름은
      `tcn_onnx`, `tcn_onnx_int8`이고 둘 다 부가 묶음이다. `selected`가 TCN일 때만 만들면 fold마다 있고 없고가
      달라져 묶음이 fold를 다 덮지 못하므로 계열별 TCN으로 만든다.
      가정: 내보내기나 양자화가 실패하면(계약 검사 포함) 그 사실을 notes에 적고 두 묶음을 모두 버린 채 주 결과는
      끝까지 낸다. 부가 비교 하나 때문에 긴 학습의 결과를 잃지 않게 한다.
    - supplementary가 True면 표지가 기본값인 묶음(원본 추출 특징, 선택 사항 없음)도 부가 묶음으로 표시한다. 내보낼
      조건이 아닌 정규화 조건으로 학습한 실행(E3의 "나머지 정규화 조건")이 쓴다.
    notes에 dict를 주면 부가 평가의 기록(선택 사항, 출처별 몫, 15 fps 평가 여부, INT8 비교의 상태)을 채워 준다.
    선택 기록에는 시험 fold에 따라 달라지는 값을 넣지 않는다. 그래서 시험 fold의 데이터가 달라져도 그 fold의 기록
    (후보별 점수, 고른 설정, 다시 학습한 모델의 표준화 상수·온도·가중치 해시)은 같아야 한다(`tests/test_leakage.py`).

    학습은 차례로 한다. 이 PC에서 GPU 하나를 프로세스 여러 개가 나눠 쓰면 빨라지지 않았다(8개 학습: 1개 프로세스
    33.3초, 4개 30.1초, 8개 32.1초).
    가정: 주 결과 행이 고른 후보가 그 계열의 계열별 선택과 같으면 다시 학습하지 않고 같은 예측을 `selected`로도 싣는다.
    가정: 안쪽 선택의 학습은 시드 하나만 쓴다(6장은 다시 학습할 때의 시드 수만 정했다).
    """
    if arm not in data.tables:
        raise ValueError(f"no step table for arm {arm!r}")
    if len(families) == 0:
        raise ValueError("need at least one model family")
    if variant not in RUN_VARIANTS:
        raise ValueError(f"variant must be one of {RUN_VARIANTS}, got {variant!r}")
    main_run = variant == evaluate.VARIANT_BASE
    window_steps = cfg.windows.window_steps
    eval_stride = cfg.windows.eval_stride
    seeds = tuple(cfg.training.seeds)
    device = resolve_device(cfg.training.device)
    base = _base_train_config(cfg, data.feature_spec, variant)
    grid = candidates(cfg.training, families)

    table = data.tables[arm]
    videos = evaluate.video_table(table.quality)
    sources = mix_sources(data, cfg, arm)
    app_table = data.tables.get(ARM_SESSION_BASELINE) if arm == ARM_ALERT_BASELINE else None
    if app_table is not None and not _video_tables_match(videos, evaluate.video_table(app_table.quality)):
        raise ValueError("the session_baseline step table lists different videos than the training arm")

    splits = list(nested_splits(data.manifest))
    timings: dict[str, float] = {}

    started = time.perf_counter()
    jobs: list[_ScoreJob] = []
    for split in splits:
        assert_subject_disjoint(split.train_subjects, split.test_subjects)
        jobs.extend(
            _ScoreJob(
                candidate=candidate,
                splits=split.inner,
                steps=_steps_of(table.steps, split.train_subjects),
                extra_steps=tuple(source.of(split.train_subjects) for source in sources),
                base=base,
                window_steps=window_steps,
                eval_stride=eval_stride,
                seed=seeds[0],
            )
            for candidate in grid
        )
    scores: list[CandidateScore] = []
    for index, job in enumerate(jobs):
        scores.append(_score_job(job))
        _progress(
            f"inner selection {index + 1}/{len(jobs)} ({job.candidate.name}) done, "
            f"{time.perf_counter() - started:.0f} s elapsed"
        )
    timings["inner_selection_s"] = time.perf_counter() - started

    started = time.perf_counter()
    train_windows = make_windows(table.steps, window_steps, 1)
    eval_windows = train_windows if eval_stride == 1 else make_windows(table.steps, window_steps, eval_stride)
    source_windows = tuple(source.windows(window_steps) for source in sources)
    app_windows = make_windows(app_table.steps, window_steps, eval_stride) if app_table is not None else None

    # 15 fps 모사 평가(부가): 평가 조건마다 그 추출의 창. 원본에 창이 있는 시험 fold를 다 덮을 때만 평가한다.
    fps_eval: list[tuple[str, WindowSet, pd.DataFrame]] = []
    fps_skipped: dict[str, str] = {}
    if main_run:
        for feature_arm, native_windows in ((arm, eval_windows), (ARM_SESSION_BASELINE, app_windows)):
            if native_windows is None or feature_arm not in data.extra_tables:
                continue
            fps_steps = data.extra_tables[feature_arm].steps
            fps_windows = make_windows(fps_steps, window_steps, eval_stride)
            uncovered = [
                int(split.test_fold)
                for split in splits
                if len(subset_subjects(native_windows, split.test_subjects)) > 0
                and len(subset_subjects(fps_windows, split.test_subjects)) == 0
            ]
            if uncovered:
                fps_skipped[feature_arm] = f"the 15 fps extraction has no windows for test fold(s) {uncovered}"
                continue
            fps_eval.append((feature_arm, fps_windows, fps_steps))
    int8_wanted = bool(fold_int8) and main_run and KIND_TCN in families
    int8_error: str | None = None
    int8_models = 0
    int8_seconds = 0.0
    int8_float_diff = 0.0  # float ONNX 대 PyTorch 확률의 최대 절대 차이(일치 검증과 같은 양)
    int8_drift = 0.0  # INT8 대 float ONNX 확률의 최대 절대 차이
    fps_seconds = 0.0

    predictions: list[pd.DataFrame] = []
    onnx_predictions: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    for position, split in enumerate(splits):
        fold_scores = scores[position * len(grid) : (position + 1) * len(grid)]
        chosen: dict[str, CandidateScore] = {
            family: choose_candidate([score for score in fold_scores if score.candidate.family == family])
            for family in families
        }
        chosen[evaluate.FAMILY_SELECTED] = choose_candidate(fold_scores)
        refits: list[dict[str, Any]] = []
        records.append(
            {
                "fold": int(split.test_fold),
                "test_subjects": list(split.test_subjects),
                "train_subjects": list(split.train_subjects),
                "inner_val_folds": [int(inner.val_fold) for inner in split.inner],
                "candidates": [_score_record(score) for score in fold_scores],
                "chosen": {label: score.candidate.name for label, score in chosen.items()},
                "refits": refits,
            }
        )

        fold_train = subset_subjects(train_windows, split.train_subjects)
        fold_extra = tuple(item.of(split.train_subjects) for item in source_windows)
        native = evaluate.FEATURE_SOURCE_NATIVE
        evaluated: list[tuple[str, str, WindowSet, pd.DataFrame]] = [
            (arm, native, subset_subjects(eval_windows, split.test_subjects), table.steps)
        ]
        if app_windows is not None and app_table is not None:
            evaluated.append(
                (ARM_SESSION_BASELINE, native, subset_subjects(app_windows, split.test_subjects), app_table.steps)
            )
        for feature_arm, fps_windows, fps_steps in fps_eval:
            evaluated.append(
                (feature_arm, SOURCE_FPS_SIM, subset_subjects(fps_windows, split.test_subjects), fps_steps)
            )
        refit: dict[str, CandidateScore] = {score.candidate.name: score for score in chosen.values()}
        for name, score in refit.items():
            labels = [label for label, picked in chosen.items() if picked.candidate.name == name]
            refit_cfg = dataclasses.replace(
                base,
                kind=score.candidate.family,
                hyper=score.candidate.hyper,
                fixed_units=score.units,
                temperature=score.temperature,
            )
            for seed in seeds:
                result = fit(fold_train, None, refit_cfg, seed, extra_train=fold_extra)
                refits.append(
                    {
                        "name": name,
                        "labels": labels,
                        "seed": int(seed),
                        "units": int(result.units),
                        "updates": int(result.updates),
                        "temperature": float(result.temperature),
                        "mean": [float(value) for value in result.spec.means],
                        "std": [float(value) for value in result.spec.stds],
                        "weights_sha256": weights_sha256(result.net),
                        "sources": [[source_name, float(share)] for source_name, share in result.sources],
                    }
                )
                native_probs: dict[str, np.ndarray] = {}
                for feature_arm, feature_source, windows, steps in evaluated:
                    if len(windows) == 0:
                        continue  # 창이 없는 시험 fold: 영상 표의 영상이 모두 오답으로 남는다(D7)
                    tick = time.perf_counter()
                    probs = predict_proba(result, windows, device)
                    if feature_source == native:
                        native_probs[feature_arm] = probs
                    for label in labels:
                        predictions.append(
                            evaluate.prediction_table(
                                windows,
                                probs,
                                steps,
                                videos,
                                run_id=run_id,
                                family=label,
                                arm=feature_arm,
                                seed=seed,
                                temperature=result.temperature,
                                feature_source=feature_source,
                                variant=variant,
                                supplementary=True if supplementary else None,
                            )
                        )
                    if feature_source != native:
                        fps_seconds += time.perf_counter() - tick
                if int8_wanted and int8_error is None and evaluate.FAMILY_TCN in labels:
                    tick = time.perf_counter()
                    try:
                        with tempfile.TemporaryDirectory() as work_dir:
                            float_path, int8_path = export_fold_model(result, cfg.paths.feature_spec, Path(work_dir))
                            for feature_arm, feature_source, windows, steps in evaluated:
                                if feature_source != native or len(windows) == 0:
                                    continue
                                assert_subject_disjoint(result.train_subjects, window_subjects(windows))
                                float_probs = onnx_predict(float_path, windows.x)
                                int8_probs = onnx_predict(int8_path, windows.x)
                                int8_float_diff = max(
                                    int8_float_diff, float(np.max(np.abs(float_probs - native_probs[feature_arm])))
                                )
                                int8_drift = max(int8_drift, float(np.max(np.abs(int8_probs - float_probs))))
                                for family, onnx_probs in (
                                    (FAMILY_TCN_ONNX, float_probs),
                                    (FAMILY_TCN_ONNX_INT8, int8_probs),
                                ):
                                    onnx_predictions.append(
                                        evaluate.prediction_table(
                                            windows,
                                            onnx_probs,
                                            steps,
                                            videos,
                                            run_id=run_id,
                                            family=family,
                                            arm=feature_arm,
                                            seed=seed,
                                            temperature=result.temperature,
                                            supplementary=True,
                                        )
                                    )
                        int8_models += 1
                    except (ValueError, RuntimeError, OSError) as exc:  # ContractError도 ValueError다
                        int8_error = f"fold {split.test_fold}, seed {seed}: {type(exc).__name__}: {exc}"
                        _progress(f"INT8 comparison of the fold models failed and is dropped ({int8_error})")
                    int8_seconds += time.perf_counter() - tick
        _progress(f"fold {split.test_fold}: refit and test done, {time.perf_counter() - started:.0f} s elapsed")
    timings["refit_and_test_s"] = time.perf_counter() - started
    if fps_eval:
        timings["fps15_eval_s"] = fps_seconds
    if int8_wanted:
        timings["fold_int8_s"] = int8_seconds
    if int8_wanted and int8_error is None:
        predictions.extend(onnx_predictions)
    if notes is not None:
        if not int8_wanted:
            int8_status = "not requested"
        elif int8_error is not None:
            int8_status = "failed"
        else:
            int8_status = "ok" if int8_models > 0 else "no model"
        notes.update(
            {
                "variant": variant,
                "train_sources": source_shares(sources),
                "fps15_evaluation": {
                    "evaluated_arms": [feature_arm for feature_arm, _, _ in fps_eval],
                    "skipped": fps_skipped,
                },
                "fold_int8": {
                    "status": int8_status,
                    "models": int8_models,
                    "error": int8_error,
                    "families": [FAMILY_TCN_ONNX, FAMILY_TCN_ONNX_INT8],
                    "float_onnx_vs_pytorch_max_abs_diff": int8_float_diff if int8_status == "ok" else None,
                    "int8_vs_float_onnx_max_abs_diff": int8_drift if int8_status == "ok" else None,
                },
            }
        )
    if not predictions:
        raise ValueError("cross-validation produced no predictions: no test fold has windows")
    return evaluate.validate_predictions(pd.concat(predictions, ignore_index=True)), records, timings


@dataclass(frozen=True, eq=False)
class CVResult:
    """교차검증 실행의 결과(`run_nested_cv`, `baseline_lgbm.run_lgbm_cv`).

    - run_dir: 리포트 폴더. predictions: 이 실행이 만든 예측 표(`evaluate`의 형식).
    - selection: 바깥 fold별 안쪽 선택 기록. report: `evaluate.write_reports`의 결과. timings: 단계별 소요 시간(초).
    """

    run_dir: Path
    predictions: pd.DataFrame
    selection: list[dict[str, Any]]
    report: evaluate.Report
    timings: dict[str, float]


def run_id_for(arm: str, fast: bool, suffix: str = "") -> str:
    """학습 실행의 이름(예측 표의 run_id): `train-{학습 조건}{접미사}`, 축소 설정이면 `-fast`가 붙는다."""
    return f"train-{arm}{suffix}" + ("-fast" if fast else "")


def rules_for(cfg: ExperimentConfig, spec: FeatureSpec) -> RulesConfig:
    """평가에 쓰는 규칙 설정(`paths.rules`). 시계 설정은 feature_spec의 것으로 맞춘다."""
    return dataclasses.replace(load_rules(cfg.paths.rules), clock=spec.pipeline.clock)


def fold_subjects(manifest: pd.DataFrame) -> dict[int, tuple[int, ...]]:
    """바깥 fold 번호 → 그 fold의 시험 피험자(`evaluate.write_reports`의 fold_subjects)."""
    return {int(split.test_fold): tuple(split.test_subjects) for split in nested_splits(manifest)}


def confound_note(run_dir: Path) -> str:
    """교란 점검 결과(`probes/probes.json`)가 있고 우연 수준을 넘은 것이 있으면 리포트 맨 앞에 적을 문장, 아니면 ""."""
    path = run_dir / PROBES_SUBDIR / PROBES_FILE
    if not path.is_file():
        return ""
    probes = json.loads(path.read_text(encoding="utf-8")).get("probes", [])
    flagged = [str(probe.get("name")) for probe in probes if probe.get("above_chance")]
    if not flagged:
        return ""
    return f"CONFOUND PROBE ABOVE CHANCE ({', '.join(flagged)}; see {PROBES_SUBDIR}/summary.md). "


def step_table_fingerprint(steps: pd.DataFrame) -> str:
    """스텝 표 내용의 지문(sha256 16진수). 리포트에 실린 예측이 같은 스텝 특징에서 나왔는지 대조하는 데 쓴다.

    `STEP_COLUMNS`의 열을 그 순서대로, 열마다 `이름(UTF-8) + b"\\0" + 값 바이트`를 sha256에 넣는다. 값 바이트는
    식별자 문자열 열(video_id, session_id)이 값을 `"\\0"`으로 이은 UTF-8, `full_lookback`이 uint8, 특징 17개가
    little-endian float64, 나머지 정수 열이 little-endian int64다. 행 순서는 표의 순서 그대로다(스텝 표는 목록의 세션
    순서, 그 안에서 k 오름차순으로 만들어진다).
    추출 결과 파일의 크기, 수정 시각, 파일 전체의 sha256으로 만드는 캐시 지문(`dataset._source_fingerprint`)과 달리 표의 값만 본다.
    같은 추출 결과를 다시 복사해도 값이 같고, 추출을 다시 하거나 특징을 만드는 코드가 바뀌어 값이 달라지면 바뀐다.
    가정: 지문은 `STEP_COLUMNS` 전체(식별자, 스텝 번호, 특징 17개)를 표의 행 순서대로 넣는다. 목록의 세션 순서가
    바뀌어도 값이 바뀐다(같은 데이터를 다른 순서로 읽은 것도 "다른 표"로 본다).
    """
    lacking = [name for name in STEP_COLUMNS if name not in steps.columns]
    if lacking:
        raise ValueError(f"steps table lacks columns {lacking}")
    digest = hashlib.sha256()
    for name in STEP_COLUMNS:
        column = steps[name]
        if name in ("video_id", "session_id"):
            payload = "\0".join(str(value) for value in column.tolist()).encode("utf-8")
        elif name == "full_lookback":
            payload = column.to_numpy(dtype=bool).astype(np.uint8).tobytes()
        elif name in FEATURE_NAMES:
            payload = np.ascontiguousarray(column.to_numpy(dtype="<f8")).tobytes()
        else:
            payload = np.ascontiguousarray(column.to_numpy(dtype="<i8")).tobytes()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(payload)
    return digest.hexdigest()


def write_predictions(predictions: pd.DataFrame, path: Path, *, synthetic: bool) -> None:
    """예측 표를 parquet으로 쓴다. 스키마 메타데이터 `drowsy_predictions`에 데이터 종류를 적는다.

    값은 JSON `{"data_note": ..., "synthetic": true|false}`다. 표의 열에는 합성 데이터라는 표시가 없어서(run_id도 실제
    데이터의 것과 같다) 파일만 따로 복사되면 실제 데이터의 예측과 구분할 수 없다. `pd.read_parquet`은 이 메타데이터를
    무시하고 표를 그대로 읽는다. 표지는 `read_predictions_marker`로 읽는다.
    """
    table = pa.Table.from_pandas(predictions, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    marker = {"data_note": data_note(synthetic), "synthetic": bool(synthetic)}
    metadata[PREDICTIONS_METADATA_KEY] = json.dumps(marker).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(metadata), path)


def read_predictions_marker(path: Path) -> dict[str, Any] | None:
    """`write_predictions`가 적은 데이터 종류 표지. 표지가 없는 파일(예전 형식, 다른 곳에서 쓴 파일)이면 None."""
    stored = (pq.read_schema(path).metadata or {}).get(PREDICTIONS_METADATA_KEY)
    if stored is None:
        return None
    marker = json.loads(stored.decode("utf-8"))
    return marker if isinstance(marker, dict) else None


def write_run_files(
    run_dir: Path,
    predictions: pd.DataFrame,
    selection: Sequence[Mapping[str, Any]],
    timings: Mapping[str, float],
    *,
    synthetic: bool,
    description: str,
    command: str,
    pipeline_hash: str,
    step_fingerprints: Mapping[str, str],
    extra: Mapping[str, Any] | None = None,
) -> None:
    """실행 폴더에 예측 표와 기록 파일을 쓴다. JSON 파일은 첫 키가 데이터 종류(`data_note`)다.

    - predictions.parquet: `write_predictions`(데이터 종류를 parquet 메타데이터에 적는다).
    - selection.json, timings.json, run.json. run.json에는 pipeline_hash와 step_fingerprints(정규화 조건 → 그 조건
      스텝 표의 `step_table_fingerprint`), extra의 항목이 들어간다.
    가정: 같은 폴더의 `config.yaml`과 `rules.json`(`evaluate.write_reports`가 쓴다)은 설정 파일의 바이트 사본이라
    데이터 종류 표시가 없다. rules.json은 바이트의 해시가 rules_hash라서 한 줄도 더할 수 없다.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(predictions, run_dir / PREDICTIONS_FILE, synthetic=synthetic)
    header = {"data_note": data_note(synthetic), "synthetic": synthetic, "description": description, "command": command}
    write_json({**header, "folds": list(selection)}, run_dir / SELECTION_FILE)
    write_json({**header, "seconds": {key: float(value) for key, value in timings.items()}}, run_dir / TIMINGS_FILE)
    write_json(
        {
            **header,
            "pipeline_hash": pipeline_hash,
            "step_fingerprints": {str(arm): str(value) for arm, value in step_fingerprints.items()},
            **(dict(extra) if extra else {}),
        },
        run_dir / RUN_INFO_FILE,
    )


def check_run_dir(
    run_dir: Path, identity: Mapping[str, Any], overwrite: bool = False, record_file: str = RUN_INFO_FILE
) -> None:
    """run_dir에 다른 실행의 리포트가 있으면 덮어쓰기 전에 멈춘다 (ValueError). 학습을 시작하기 전에 부른다.

    `run_dir/{record_file}`(기본 run.json)이 있고 identity의 키 가운데 값이 다른 것이 있으면 다른 실행이다. identity는
    그 폴더를 쓰는 실행을 가르는 값들이다: 신경망 교차검증은 `{"run_id", "synthetic"}`(학습 조건과 축소 설정 여부가
    run_id에 들어 있다), LightGBM은 `{"arms", "fast", "closure_w", "synthetic"}`, 최종 모델(`final/final.json`)은
    `{"arm", "fast", "synthetic"}`. 기록을 읽을 수 없어도 다른 실행으로 본다.
    기록 파일이 없으면(새 폴더, `lgbm/`이나 `probes/`만 있는 폴더) 통과다. overwrite가 True면 검사하지 않는다.
    가정: 같은 identity의 실행을 다시 돌리는 것(같은 조건의 재실행)은 덮어쓴다. 추출을 다시 했거나 코드를 고친 뒤의
    재실행이 그런 경우다.
    """
    info_path = Path(run_dir) / record_file
    if overwrite or not info_path.is_file():
        return
    try:
        stored = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        stored = None
    if not isinstance(stored, dict):
        raise ValueError(
            f"{run_dir} holds a run record that cannot be read ({record_file}); "
            "pass --overwrite to replace that report or choose another --run-name"
        )
    differing = [
        f"{key}: {stored.get(key)!r} there, {value!r} now"
        for key, value in identity.items()
        if stored.get(key) != value
    ]
    if differing:
        raise ValueError(
            f"{run_dir} already holds the report of another run ({'; '.join(differing)}); "
            "pass --overwrite to replace it or choose another --run-name"
        )


def _reference_table(data: CVData, cfg: ExperimentConfig) -> StepTable:
    """기준 LightGBM 조건(`lgbm.reference_arm`)의 스텝 표. data에 없으면 같은 데이터에서 읽는다(보통 캐시에 있다)."""
    arm = cfg.lgbm.reference_arm
    if arm in data.tables:
        return data.tables[arm]
    tables = load_step_tables(
        data.manifest,
        data.raw_dir,
        data.feature_spec.pipeline,
        (arm,),
        gate=cfg.evaluation.quality_gate,
        workers=cfg.extraction.workers,
    )
    return tables[arm]


def _reference_lgbm(run_dir: Path, cfg: ExperimentConfig, data: CVData, fast: bool) -> pd.DataFrame | None:
    """같은 실행 폴더의 `lgbm/`에 기준 LightGBM(`lgbm.reference_arm`)의 예측 표가 있으면 읽는다. 없으면 None.

    그 예측이 이 실행과 같은 조건에서 나온 것인지 대조하고, 하나라도 어긋나면 ValueError다(리포트와 D11에 섞지 않는다).
    - `lgbm/run.json`의 pipeline_hash와 합성 데이터 여부, `predictions.parquet`의 데이터 종류 표지
    - 기준 조건 스텝 표의 내용 지문(`step_table_fingerprint`): 추출을 다시 했거나 특징 코드가 바뀐 뒤 LightGBM을
      다시 돌리지 않은 경우를 잡는다
    - 축소 설정 여부: `--fast`로 만든 기준(후보 하나, 짧은 부스팅)을 사전에 고정한 절차의 실행이 쓰지 않게, 그리고 그
      반대도 막는다
    - closure W: `lgbm/`은 feature_spec의 W로 돈 실행이고, 그 값이 `lgbm.reference_w`여야 한다(PLAN 5.3절)
    `lgbm/`의 실행이 기준 조건을 돌리지 않았으면(다른 조건만) None이다.
    가정: 지문이나 데이터 종류 표지가 없는 예전 형식의 `lgbm/`(이 검사들이 생기기 전에 만든 것)은 대조할 수 없으므로
    받지 않는다. `baseline_lgbm cv`를 다시 돌리면 된다.
    """
    folder = run_dir / LGBM_SUBDIR
    predictions_path = folder / PREDICTIONS_FILE
    info_path = folder / RUN_INFO_FILE
    if not predictions_path.is_file() or not info_path.is_file():
        return None
    info = json.loads(info_path.read_text(encoding="utf-8"))
    arm = cfg.lgbm.reference_arm
    arms = info.get("arms")
    if isinstance(arms, list) and arm not in arms:
        return None
    table = _reference_table(data, cfg)
    kind = {True: "synthetic", False: "real"}
    problems: list[str] = []
    if info.get("pipeline_hash") != table.pipeline_hash:
        problems.append(f"other pipeline constants (pipeline_hash {info.get('pipeline_hash')})")
    if bool(info.get("synthetic")) != data.synthetic:
        problems.append(f"{kind[bool(info.get('synthetic'))]} data, this run uses {kind[data.synthetic]} data")
    marker = read_predictions_marker(predictions_path)
    if marker is None:
        problems.append(f"{PREDICTIONS_FILE} carries no data marker")
    elif bool(marker.get("synthetic")) != data.synthetic:
        problems.append(f"{PREDICTIONS_FILE} is marked {kind[bool(marker.get('synthetic'))]} data")
    fingerprints = info.get("step_fingerprints")
    stored_fingerprint = fingerprints.get(arm) if isinstance(fingerprints, dict) else None
    if stored_fingerprint != step_table_fingerprint(table.steps):
        problems.append(f"other step features for arm {arm} (the extraction or the feature code changed)")
    if bool(info.get("fast")) != fast:
        problems.append(f"made {'with' if info.get('fast') else 'without'} --fast, unlike this run")
    if info.get("closure_w") is not None or info.get("effective_closure_w") != cfg.lgbm.reference_w:
        problems.append(
            f"closure W {info.get('effective_closure_w')!r} (override {info.get('closure_w')!r}), "
            f"lgbm.reference_w is {cfg.lgbm.reference_w!r}"
        )
    if problems:
        raise ValueError(
            f"{folder} cannot serve as the LightGBM reference of this run: {'; '.join(problems)}. "
            "Rerun `python -m drowsy.baseline_lgbm cv` with the same --run-name and profile"
        )
    preds = evaluate.validate_predictions(pd.read_parquet(predictions_path))
    reference = preds[
        (preds["family"] == evaluate.FAMILY_LGBM)
        & (preds["arm"] == arm)
        & (preds["run_id"] == run_id_for(arm, fast))
        & ~preds["supplementary"].to_numpy(dtype=bool)  # 부가 묶음(E3)은 기준이 아니다
    ]
    if len(reference) == 0:
        return None
    return reference.reset_index(drop=True)


def report_window_counts(data: CVData, cfg: ExperimentConfig) -> dict[str, dict[int, dict[str, int]]]:
    """리포트 8절에 싣는 조건별·보폭별·클래스별 창 수 (PLAN 4.4절). `evaluate.write_reports(window_counts=...)`의 값.

    원본 추출의 조건마다 한 항목, 두 번째 추출(15 fps 누락 모사)의 표가 있으면 `{조건} [{drop_sim.subdir}]`, 학습에 섞는
    프레임 누락 변형이 있으면 `{조건} [frame_drop_v{번호}]` 항목을 더한다. 보폭은 `windows.train_stride`와
    `windows.eval_stride`다.
    """
    steps = {name: table.steps for name, table in data.tables.items()}
    for name, table in data.extra_tables.items():
        steps[f"{name} [{cfg.extraction.drop_sim.subdir}]"] = table.steps
    for name, variants in data.frame_drop_tables.items():
        for position, table in enumerate(variants):
            steps[f"{name} [{frame_drop_source_name(table, position)}]"] = table.steps
    return evaluate.window_counts(steps, cfg.windows.window_steps, (cfg.windows.train_stride, cfg.windows.eval_stride))


def baseline_summary(data: CVData, cfg: ExperimentConfig) -> dict[str, Any]:
    """run.json에 싣는 기준선·창 요약(`dataset.baseline_window_summary`, 세션 단위. PLAN 4.4절).

    리포트(summary.md 8절, videos.csv)는 논리 영상 단위로 합친 값을 싣고, 이 요약은 나뉜 영상의 부분마다 T_b를 따로
    적는다.
    """
    return baseline_window_summary(
        data.tables,
        cfg.windows.window_steps,
        (cfg.windows.train_stride, cfg.windows.eval_stride),
        data.feature_spec.pipeline.baseline.duration_ms,
    )


def main_run_predictions(
    folder: Path,
    *,
    expected: Mapping[str, Any],
    synthetic: bool,
    pipeline_hash: str,
    fingerprints: Mapping[str, str],
    run_ids: Sequence[str],
) -> pd.DataFrame | None:
    """부가 실행이 짝지어 비교할 주 실행의 예측 표(주 결과 묶음만)를 읽는다. folder에 주 실행이 없으면 None.

    folder는 주 실행의 리포트 폴더다(신경망은 실행 폴더, LightGBM은 `lgbm/`). 주 실행이 이 부가 실행과 같은 데이터,
    같은 절차에서 나온 것인지 대조하고, 하나라도 어긋나면 ValueError다(학습을 시작하기 전에 부른다).
    - `run.json`의 expected 항목(신경망: run_id, LightGBM: 축소 설정 여부와 closure W)
    - 합성 데이터 여부(`run.json`과 `predictions.parquet`의 데이터 종류 표지), pipeline_hash
    - 스텝 표의 내용 지문: fingerprints(조건 → 지문) 가운데 주 실행의 기록에 있는 조건이 모두 같아야 한다
    돌려주는 표는 run_id가 run_ids에 들고 부가 표시가 없는 행이다. 그런 행이 없으면 None이다.
    부가 실행은 이 표를 읽기만 한다. 주 실행의 파일은 고치지 않는다.
    가정: 주 실행의 리포트가 있는데 대조에 실패하면(추출이나 코드가 바뀐 뒤 주 실행을 다시 돌리지 않은 경우) 짝 없이
    진행하지 않고 멈춘다. 부가 실행의 목적이 주 실행과의 비교이기 때문이다. 주 실행이 아직 없으면 짝 없이 돈다.
    """
    folder = Path(folder)
    predictions_path = folder / PREDICTIONS_FILE
    info_path = folder / RUN_INFO_FILE
    if not predictions_path.is_file() or not info_path.is_file():
        return None
    kind = {True: "synthetic", False: "real"}
    problems: list[str] = []
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        info = None
    if not isinstance(info, dict):
        info = {}
        problems.append(f"{RUN_INFO_FILE} cannot be read")
    for key, value in expected.items():
        if info.get(key) != value:
            problems.append(f"{key} is {info.get(key)!r}, this run needs {value!r}")
    if bool(info.get("synthetic")) != synthetic:
        problems.append(f"{kind[bool(info.get('synthetic'))]} data, this run uses {kind[synthetic]} data")
    marker = read_predictions_marker(predictions_path)
    if marker is None:
        problems.append(f"{PREDICTIONS_FILE} carries no data marker")
    elif bool(marker.get("synthetic")) != synthetic:
        problems.append(f"{PREDICTIONS_FILE} is marked {kind[bool(marker.get('synthetic'))]} data")
    if info.get("pipeline_hash") != pipeline_hash:
        problems.append(f"other pipeline constants (pipeline_hash {info.get('pipeline_hash')})")
    stored = info.get("step_fingerprints")
    stored = stored if isinstance(stored, dict) else {}
    differing = sorted(name for name, value in fingerprints.items() if name in stored and stored[name] != value)
    if differing:
        problems.append(f"other step features for {', '.join(differing)} (the extraction or the feature code changed)")
    if problems:
        raise ValueError(
            f"{folder} cannot serve as the main run of this supplementary run: {'; '.join(problems)}. "
            "Rerun the main run with the same --run-name and profile first, or choose another --run-name"
        )
    preds = evaluate.validate_predictions(pd.read_parquet(predictions_path))
    main = preds[preds["run_id"].isin(list(run_ids)) & ~preds["supplementary"].to_numpy(dtype=bool)]
    if len(main) == 0:
        return None
    return main.reset_index(drop=True)


def _group_keys(predictions: pd.DataFrame) -> list[tuple[str, str, str, str, str]]:
    """예측 표의 묶음들 (run_id, family, arm, feature_source, variant), 오름차순."""
    groups = predictions[list(evaluate.GROUP_KEYS)].drop_duplicates()
    return sorted(tuple(str(value) for value in row) for row in groups.itertuples(index=False, name=None))


def supplementary_run_pairs(
    predictions: pd.DataFrame, main: pd.DataFrame, run_ids: Mapping[str, str]
) -> list[tuple[evaluate.GroupKey, evaluate.GroupKey]]:
    """부가 실행의 묶음마다 `(부가 묶음, 주 실행의 같은 계열·같은 평가 조건·같은 출처의 묶음)` 쌍을 만든다.

    run_ids는 부가 실행의 run_id → 주 실행의 run_id다. 주 실행에 짝이 없는 묶음은 쌍을 만들지 않는다.
    run_id가 달라 `evaluate.supplementary_pairs`가 찾지 못하는 쌍이라 여기서 직접 적는다.
    """
    known = {evaluate.group_key(*key) for key in _group_keys(main)}
    pairs: list[tuple[evaluate.GroupKey, evaluate.GroupKey]] = []
    for run_id, family, arm, source, variant in _group_keys(predictions):
        if run_id not in run_ids:
            continue
        base = evaluate.group_key(run_ids[run_id], family, arm, source)
        if base in known:
            pairs.append((evaluate.group_key(run_id, family, arm, source, variant), base))
    return pairs


def variant_description(variant: str, settings: TrainingSettings) -> str:
    """부가 실행의 선택 사항을 리포트 설명에 적는 문장."""
    if variant == evaluate.VARIANT_ORDINAL:
        return (
            "ordinal soft labels instead of uniform label smoothing: each class gives "
            f"{settings.ordinal_neighbor_mass:g} of its target probability to its ordinal neighbours (the middle "
            "class half to each side, the edge classes all to their single neighbour)"
        )
    if variant == evaluate.VARIANT_NO_VELOCITY:
        names = ", ".join(f"#{index} {FEATURE_NAMES[index]}" for index in VELOCITY_FEATURES)
        return f"without the velocity features {names}"
    raise ValueError(f"no supplementary run option {variant!r}, expected one of {SUPPLEMENTARY_VARIANTS}")


def run_nested_cv(
    cfg: ExperimentConfig,
    *,
    arm: str | None = None,
    families: Sequence[str] = KINDS,
    fast: bool = False,
    run_name: str = DEFAULT_CV_RUN_NAME,
    config_path: Path = Path("config/default.yaml"),
    command: str = "",
    overwrite: bool = False,
    allow_partial_fps_sim: bool = False,
    rebuild: bool = False,
    variant: str = evaluate.VARIANT_BASE,
) -> CVResult:
    """중첩 교차검증을 돌리고 리포트를 쓴다 (PLAN 6장, 11장 E2, E3).

    - arm: 학습 조건. None이면 `training.export_arm`(alert 영상 기준선).
    - families: 후보 계열(tcn, gru). fast: 축소 설정(`fast` 블록)으로 돈다. 사전에 고정한 절차가 아니라 연기 시험용이다.
    - run_name: 리포트 폴더 이름(`paths.reports_dir` 아래). config_path, command: 리포트에 남길 설정 사본과 명령.
    - overwrite: 실행 폴더에 다른 실행(다른 학습 조건, 축소 설정 여부, 데이터 종류)의 리포트가 있어도 덮어쓴다.
      False면 학습을 시작하기 전에 ValueError로 멈춘다(`check_run_dir`).
    - allow_partial_fps_sim, rebuild: `load_cv_data`에 그대로 넘긴다.
    - variant: `base`가 주 실행이다. `ordinal`(순서형 소프트 라벨)과 `no_velocity`(속도 특징 #7, #8 제외)는 부가
      실행이다(아래).
    학습 조건이 `training.export_arm`이 아닌 실행(PLAN 11장 E3의 "나머지 정규화 조건")은 같은 절차를 그 조건으로 돈
    것이고, 묶음을 모두 부가 묶음으로 표시한다(리포트 9절에만 실린다). 그런 실행은 D11을 판정하지 않는다: 실행 폴더에
    기준 LightGBM이 있어도 판정 대상을 넘기지 않고, 리포트 설명에 판정하지 않았다고 적는다(D11은 주 결과 묶음으로만
    판정하는데 이 실행의 묶음은 모두 부가 묶음이다).
    가정: 주 결과(E2)는 내보낼 조건(`training.export_arm`, alert 영상 기준선)으로 학습한 실행 하나다. 다른 조건의
    실행은 다른 실행 이름(폴더)에 쓰이고(`check_run_dir`), 그 리포트에는 주 결과 묶음이 없다.

    주 실행(variant = base)
    리포트에는 신경망 묶음(계열별, selected, 앱 조건 평가)과, 같은 실행 폴더에 `lgbm/`이 있으면 기준 LightGBM 묶음을
    싣는다. 그때 D11 판정(후보: selected의 앱 조건 평가, 기준: LightGBM)과 짝지은 비교(TCN 대 GRU, selected 대
    LightGBM)도 함께 싣는다. 합성 데이터의 리포트는 첫 줄에 그 사실을 적고 판정원은 perclos로 둔다.
    `lgbm/`의 예측이 이 실행과 다른 조건에서 나온 것이면(`_reference_lgbm`) 학습을 시작하기 전에 ValueError로 멈춘다.
    리포트의 8절에는 기준선이 연장·대체된 영상과 조건별·클래스별 창 수가 실린다(PLAN 4.4절).
    9절(부가 결과)에는 같은 fold 모델의 15 fps 모사 평가(그 추출이 있을 때)와, `export.int8_if_tcn`이 켜져 있으면 TCN
    fold 모델의 float 대 INT8 비교(D16)가 짝지은 비교와 함께 실린다(`cross_validate`). 이 묶음들은 주 결과 표와
    D11에 쓰이지 않는다.
    가정: 설정 키 `export.int8_if_tcn`(최종 모델의 INT8 파일을 만들지)이 fold 모델의 INT8 비교도 함께 켜고 끈다.
    따로 키를 두지 않았다. fold 모델의 내보내기는 임시 폴더에서 하고 파일을 남기지 않는다.

    부가 실행(variant = ordinal 또는 no_velocity. PLAN 11장 E3)
    같은 데이터, 같은 탐색 범위, 같은 절차를 그 선택 사항으로 한 번 더 돈다. run_id에 `-{variant}`가 붙고, 모든 묶음이
    부가 묶음이며, 리포트는 실행 폴더의 하위 폴더 `{variant}/`에 쓴다. 주 실행의 파일(`summary.md`,
    `predictions.parquet` 등)은 읽기만 하고 고치지 않으며, D11을 판정하지 않는다. 실행 폴더에 주 실행의 리포트가
    있으면(`main_run_predictions`) 그 주 결과 묶음을 이 하위 리포트에 함께 실어 묶음마다 짝지은 비교를 낸다. 주 실행이
    없으면 짝지은 비교 없이 돈다. 평가는 원본 추출 특징으로만 한다(15 fps 평가와 INT8 비교는 주 실행에만 있다).
    가정: 부가 실행끼리 겹친 조합(순서형 라벨 모델의 15 fps 평가 등)은 만들지 않는다. 계획서가 든 부가 분석은 넷이고
    (나머지 정규화 조건, 속도 특징 제외, 순서형 라벨, 15 fps 모사 평가) 각각 주 실행과 하나만 다르게 둔다.
    가정: 계획서 11장 E3의 "설정은 E2 것을 고정"을 "탐색 범위, 시드, 절차를 E2와 같게 둔다"로 읽는다. E2가 fold마다
    고른 후보를 그대로 가져오지 않고, 바깥 fold마다 안쪽 선택을 다시 한다(선택 사항이 바뀌면 맞는 학습 길이와 온도가
    달라지고, 그 값들을 시험 fold 없이 정하는 길은 안쪽 선택뿐이다).
    """
    if fast:
        cfg = apply_fast_profile(cfg)
    if variant not in RUN_VARIANTS:
        raise ValueError(f"variant must be one of {RUN_VARIANTS}, got {variant!r}")
    arm = cfg.training.export_arm if arm is None else arm
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    families = tuple(families)
    main_run = variant == evaluate.VARIANT_BASE
    main_dir = cfg.paths.reports_dir / run_name
    run_dir = main_dir if main_run else main_dir / variant
    total_started = time.perf_counter()

    arms = (arm, ARM_SESSION_BASELINE) if arm == ARM_ALERT_BASELINE else (arm,)
    data = load_cv_data(
        cfg,
        arms,
        mix_arms=(arm,),
        fps_eval_arms=arms if main_run else (),
        allow_partial_fps_sim=allow_partial_fps_sim,
        rebuild=rebuild,
    )
    timings: dict[str, float] = {"load_data_s": time.perf_counter() - total_started}
    main_run_id = run_id_for(arm, fast)
    run_id = main_run_id if main_run else run_id_for(arm, fast, f"-{variant}")
    table = data.tables[arm]
    fingerprints = {name: step_table_fingerprint(item.steps) for name, item in data.tables.items()}
    # 오래 걸리는 학습을 시작하기 전에 멈출 수 있는 것은 여기서 멈춘다: 다른 실행의 리포트, 맞지 않는 기준 LightGBM,
    # 맞지 않는 주 실행.
    check_run_dir(run_dir, {"run_id": run_id, "synthetic": data.synthetic}, overwrite)
    reference: pd.DataFrame | None = None
    main: pd.DataFrame | None = None
    if main_run:
        reference = _reference_lgbm(run_dir, cfg, data, fast)
    else:
        main = main_run_predictions(
            main_dir,
            expected={"run_id": main_run_id},
            synthetic=data.synthetic,
            pipeline_hash=table.pipeline_hash,
            fingerprints=fingerprints,
            run_ids=(main_run_id,),
        )
    notes: dict[str, Any] = {}
    other_arm = arm != cfg.training.export_arm
    predictions, selection, cv_timings = cross_validate(
        data,
        cfg,
        arm,
        families,
        run_id,
        variant=variant,
        fold_int8=main_run and cfg.export.int8_if_tcn,
        notes=notes,
        supplementary=other_arm,
    )
    timings.update(cv_timings)

    videos = evaluate.video_table(table.quality)
    device = resolve_device(cfg.training.device)
    shares = ", ".join(f"{name} {share:g}" for name, share in notes["train_sources"].items())
    description = (
        f"{confound_note(main_dir)}{data.description}. Nested cross-validation (PLAN section 6): training arm {arm}, "
        f"families {', '.join(families)}, refit seeds {list(cfg.training.seeds)}, device {device.type}, "
        f"training sample shares {shares}. {CONSTANTS_NOTE}"
    )
    if other_arm:
        description += (
            f". SUPPLEMENTARY ANALYSIS (PLAN 11 E3): training arm {arm} is not the export arm "
            f"({cfg.training.export_arm}); every group of this run is supplementary (section 9) and is never used "
            "for the headline rows or D11. D11 (the conditions for making the model the default drowsiness source) "
            "is not evaluated for this run"
        )
    if main_run:
        fps_note = notes["fps15_evaluation"]
        int8_note = notes["fold_int8"]
        extras: list[str] = []
        if fps_note["evaluated_arms"]:
            extras.append(
                "the fold models evaluated on the 15 fps simulated extraction of the test fold (feature_source "
                f"{SOURCE_FPS_SIM}, arms {', '.join(fps_note['evaluated_arms'])})"
            )
        for skipped_arm, reason in fps_note["skipped"].items():
            extras.append(f"15 fps evaluation of arm {skipped_arm} skipped ({reason})")
        if int8_note["status"] == "ok":
            extras.append(
                f"the TCN fold models as float and INT8 ONNX files run by onnxruntime (families {FAMILY_TCN_ONNX}, "
                f"{FAMILY_TCN_ONNX_INT8}; D16)"
            )
        elif int8_note["status"] == "failed":
            extras.append(f"INT8 comparison of the fold models FAILED and is not reported ({int8_note['error']})")
        if extras:
            description += ". Supplementary groups (section 9 only, never used for the headline rows or D11): " + (
                "; ".join(extras)
            )
    else:
        description += (
            f". SUPPLEMENTARY RUN (PLAN 11 E3): {variant_description(variant, cfg.training)}. Every group of this "
            "run is supplementary (section 9) and is never used for the headline rows or D11; "
            + (
                f"the main groups listed here are copied from the main run in {main_dir} for the paired comparison"
                if main is not None
                else f"no main run was found in {main_dir}, so there is no paired comparison"
            )
        )
    if fast:
        description += f". {FAST_NOTE}"
    run_info = {
        "arm": arm,
        "run_id": run_id,
        "fast": fast,
        "families": list(families),
        "variant": variant,
        "export_arm": cfg.training.export_arm,
        "train_sources": notes["train_sources"],
        "supplementary": {
            "fps15_evaluation": notes["fps15_evaluation"],
            "fold_int8": notes["fold_int8"],
            "paired_with_main_run": main is not None,
        },
        "baselines": baseline_summary(data, cfg),
        "rotation_check": rotation_record(data.rotation),
    }
    write_run_files(
        run_dir,
        predictions,
        selection,
        timings,
        synthetic=data.synthetic,
        description=description,
        command=command,
        pipeline_hash=table.pipeline_hash,
        step_fingerprints=fingerprints,
        extra=run_info,
    )

    started = time.perf_counter()
    all_predictions = predictions
    paired: list[tuple[evaluate.GroupKey, evaluate.GroupKey]] = []
    gate: tuple[evaluate.GroupKey, evaluate.GroupKey] | None = None
    native = evaluate.FEATURE_SOURCE_NATIVE
    if KIND_TCN in families and KIND_GRU in families:
        paired.append(
            (
                evaluate.group_key(run_id, evaluate.FAMILY_TCN, arm, native, variant),
                evaluate.group_key(run_id, evaluate.FAMILY_GRU, arm, native, variant),
            )
        )
    if reference is not None:
        all_predictions = pd.concat([predictions, reference], ignore_index=True)
        reference_key = (str(reference["run_id"].iloc[0]), evaluate.FAMILY_LGBM, cfg.lgbm.reference_arm)
        if arm == cfg.lgbm.reference_arm:
            paired.append(((run_id, evaluate.FAMILY_SELECTED, arm), reference_key))
        app_key = (run_id, evaluate.FAMILY_SELECTED, ARM_SESSION_BASELINE)
        # D11은 앱 조건 평가의 주 결과 행으로 판정한다. 내보낼 조건이 아닌 조건으로 학습한 실행(other_arm)은 묶음이 모두
        # 부가 묶음이라 판정 대상이 없다(부가 묶음을 넘기면 `write_reports`가 학습이 끝난 뒤에 거부한다).
        if arm == ARM_ALERT_BASELINE and not other_arm and bool((predictions["arm"] == ARM_SESSION_BASELINE).any()):
            gate = (app_key, reference_key)
    if main is not None:
        all_predictions = pd.concat([main, predictions], ignore_index=True)
        paired.extend(supplementary_run_pairs(predictions, main, {run_id: main_run_id}))
    if main_run:
        # float ONNX 묶음과 PyTorch 묶음의 쌍(같은 모델, 다른 실행기). INT8 대 float ONNX 쌍은 아래 함수가 만든다.
        for _, family, feature_arm, source, _ in _group_keys(predictions):
            if family == FAMILY_TCN_ONNX and source == native:
                paired.append(
                    (
                        evaluate.group_key(run_id, FAMILY_TCN_ONNX, feature_arm),
                        evaluate.group_key(run_id, evaluate.FAMILY_TCN, feature_arm),
                    )
                )
        paired.extend(evaluate.supplementary_pairs(all_predictions))
    report = evaluate.write_reports(
        run_dir,
        all_predictions,
        videos,
        fold_subjects(data.manifest),
        rules_for(cfg, data.feature_spec),
        cfg.evaluation,
        synthetic=data.synthetic,
        description=description,
        command=command,
        exclusions=evaluate.exclusion_list(table.quality),
        config_path=config_path,
        gate=gate,
        paired=paired,
        window_counts=report_window_counts(data, cfg),
        nominal_baseline_ms=data.feature_spec.pipeline.baseline.duration_ms,
    )
    timings["report_s"] = time.perf_counter() - started
    timings["total_s"] = time.perf_counter() - total_started
    write_run_files(
        run_dir,
        predictions,
        selection,
        timings,
        synthetic=data.synthetic,
        description=description,
        command=command,
        pipeline_hash=table.pipeline_hash,
        step_fingerprints=fingerprints,
        extra=run_info,
    )
    return CVResult(run_dir=run_dir, predictions=predictions, selection=selection, report=report, timings=timings)


# ---------------------------------------------------------------------------
# 최종 모델
# ---------------------------------------------------------------------------


def final_splits(manifest: pd.DataFrame) -> tuple[InnerSplit, ...]:
    """전체 피험자에 대한 선택용 분할: 공식 fold마다 그 fold를 검증으로, 나머지를 학습으로 둔다 (PLAN 11장 E4).

    60명이면 5개(학습 48명, 검증 12명)다. fold가 2개 미만이면 ValueError다.
    """
    folds: dict[int, set[int]] = {}
    for subject, fold in zip(manifest["subject"].tolist(), manifest["fold"].tolist()):
        folds.setdefault(int(fold), set()).add(int(subject))
    if len(folds) < 2:
        raise ValueError(f"the final selection needs at least 2 folds, the manifest has {len(folds)}")
    assert_subject_disjoint(*folds.values())
    splits: list[InnerSplit] = []
    for val_fold in sorted(folds):
        train = tuple(sorted(s for fold, subjects in folds.items() if fold != val_fold for s in subjects))
        val = tuple(sorted(folds[val_fold]))
        assert_subject_disjoint(train, val)
        splits.append(InnerSplit(val_fold=val_fold, train_subjects=train, val_subjects=val))
    return tuple(splits)


def write_trained_spec(base_spec_path: Path, result: FitResult, out_path: Path) -> str:
    """기준 feature_spec.json에 학습한 평균·표준편차·온도를 넣어 out_path에 쓴다. 그 파일의 sha256을 돌려준다.

    나머지 내용(특징 목록, pipeline 블록, pipeline_hash)은 기준 파일 그대로다(PLAN 4.7절: 재학습으로 평균·표준편차가
    바뀌어도 pipeline_hash는 유효하다). 표준편차는 이미 하한이 적용된 값이다.
    """
    data = read_feature_spec_dict(base_spec_path)
    features = data["features"]
    if len(features) != result.spec.n_features:
        raise ValueError(f"{base_spec_path} lists {len(features)} features, the model has {result.spec.n_features}")
    for feature, trained in zip(features, result.spec.features):
        if feature["name"] != trained.name:
            raise ValueError(f"feature order differs: {feature['name']!r} in the file, {trained.name!r} in the model")
        feature["mean"] = float(trained.mean)
        feature["std"] = float(trained.std)
    data["standardization"]["temperature"] = float(result.temperature)
    return write_feature_spec(data, out_path)


def export_fit(
    result: FitResult,
    base_spec_path: Path,
    out_dir: Path,
    version: int,
    trained_on: str,
    export_cfg: ExportSettings,
    real_windows: np.ndarray | None = None,
) -> dict[str, Any]:
    """학습 결과를 `out_dir/drowsiness_v{version}.onnx`와 `out_dir/feature_spec_v{version}.json`으로 내보낸다.

    순서
    1. git 상태(`export_onnx.git_commit()`)를 어떤 파일도 쓰기 전에 읽는다. out_dir이 저장소 안이면 새 버전의 사본과
       모델은 추적되지 않는 파일이라, 쓴 뒤에 읽으면 깨끗한 작업 트리에서 내보낸 모델에 "-dirty"가 붙는다.
    2. 학습한 평균·표준편차·온도를 넣은 feature_spec 사본을 임시 폴더에 쓰고(`write_trained_spec`), 그 파일을 읽어 만든
       래퍼를 `export_onnx.export`로 내보낸다. out_dir의 사본은 export가 모델과 함께, 계약 검사를 통과한 뒤에 쓴다
       (바이트 단위로 같은 파일). 내보내기가 실패하면 out_dir에 있던 같은 버전의 모델과 사본은 그대로다.
    3. 계약 검사(`verify_contract`), 일치 검증(무작위 입력과, 주면 실제 창), 지연 측정을 한다.
    4. TCN이고 `export.int8_if_tcn`이면 INT8 파일도 만든다(D16). 계약 검사를 통과하지 못한 INT8 파일은 지운다.
       INT8 기록의 `max_abs_diff_vs_float`는 float 모델과의 확률 차이일 뿐 지표 변화가 아니다(`INT8_FINAL_NOTE`).
       최종 모델은 전원으로 학습해 따로 떼어 둔 시험 fold가 없다. 양자화에 따른 지표 변화는 `train cv`가 fold
       모델로 잰 부가 묶음(`tcn_onnx`, `tcn_onnx_int8`)이 그 추정치다.
    일치 검증이 `export.parity_tolerance`를 넘으면 RuntimeError다. 돌려주는 dict는 검증 기록이다.
    """
    commit = export_onnx.git_commit()
    out_dir = Path(out_dir)
    spec_path = out_dir / export_onnx.exported_spec_name(version)
    onnx_path = out_dir / export_onnx.model_file_name(version)
    with tempfile.TemporaryDirectory() as work_dir:
        trained_spec = Path(work_dir) / spec_path.name
        spec_hash = write_trained_spec(base_spec_path, result, trained_spec)
        spec = load_feature_spec(trained_spec)
        # 파일에서 읽은 값으로 래퍼를 만든다. 그래프 상수와 파일의 값이 같다는 것을 export가 검사한다.
        wrapper = ExportWrapper(result.net, spec, spec.standardization.temperature)
        meta = {"model_version": str(version), "trained_on": trained_on, "commit": commit}
        export_onnx.export(wrapper, trained_spec, onnx_path, meta)
    if file_sha256(spec_path) != spec_hash:
        raise RuntimeError(f"{spec_path} is not the feature spec copy this export wrote (sha256 differs)")
    export_onnx.verify_contract(onnx_path, spec_path)

    tolerance = export_cfg.parity_tolerance
    random_inputs = export_onnx.random_raw_inputs(spec, export_cfg.parity_random_inputs, seed=0)
    record: dict[str, Any] = {
        "onnx_path": str(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "onnx_sha256": file_sha256(onnx_path),
        "spec_path": str(spec_path),
        "feature_spec_hash": spec_hash,
        "metadata": export_onnx.read_metadata(onnx_path),
        "operators": list(export_onnx.graph_operators(onnx_path)),
        "verify_contract": "ok",
        "parity_tolerance": tolerance,
        "parity_random_inputs": int(random_inputs.shape[0]),
        "parity_random_max_abs_diff": export_onnx.parity_max_abs_diff(wrapper, onnx_path, random_inputs),
        "parity_real_windows": 0,
        "parity_real_max_abs_diff": None,
    }
    if real_windows is not None and len(real_windows) > 0:
        record["parity_real_windows"] = int(len(real_windows))
        record["parity_real_max_abs_diff"] = export_onnx.parity_max_abs_diff(wrapper, onnx_path, real_windows)
    worst = max(record["parity_random_max_abs_diff"], record["parity_real_max_abs_diff"] or 0.0)
    if not worst < tolerance:
        raise RuntimeError(f"exported model differs from PyTorch by {worst:.3e}, tolerance {tolerance:.0e}")
    record["latency"] = export_onnx.measure_latency(onnx_path, export_cfg.latency_runs, export_cfg.intra_op_threads)
    record["latency_runs"] = export_cfg.latency_runs
    record["int8"] = None
    if result.kind == KIND_TCN and export_cfg.int8_if_tcn:
        int8_path = export_onnx.quantize_int8(
            onnx_path, onnx_path.with_name(export_onnx.model_file_name(version, int8=True))
        )
        try:
            export_onnx.verify_contract(int8_path, spec_path, export_onnx.ALLOWED_OPS_INT8)
        except export_onnx.ContractError:
            int8_path.unlink(missing_ok=True)  # 계약에 맞지 않는 INT8 파일을 모델 옆에 남기지 않는다
            raise
        inputs = real_windows if real_windows is not None and len(real_windows) > 0 else random_inputs
        record["int8"] = {
            "onnx_path": str(int8_path),
            "onnx_bytes": int8_path.stat().st_size,
            "verify_contract": "ok",
            "max_abs_diff_vs_float": export_onnx.parity_max_abs_diff(wrapper, int8_path, inputs),
            "inputs": "real windows" if inputs is real_windows else "random raw inputs",
            "note": INT8_FINAL_NOTE,
        }
    return record


def trained_on_text(data: CVData, arm: str) -> str:
    """ONNX 메타데이터 `trained_on`의 값. 합성 데이터로 학습한 모델은 그 사실을 맨 앞에 적는다.

    합성 데이터의 표지는 `export_onnx.SYNTHETIC_TRAINED_ON_PREFIX`다. `python -m drowsy.export_onnx --checkpoint`가
    같은 표지로 합성 모델을 알아본다.
    """
    n_subjects = int(data.manifest["subject"].nunique())
    if data.synthetic:
        prefix = export_onnx.SYNTHETIC_TRAINED_ON_PREFIX
        return f"{prefix} sessions (drowsy.synth) {n_subjects} subjects, arm={arm}; pipeline check only"
    return f"UTA-RLDD {n_subjects} subjects, arm={arm}"


def fit_final(
    cfg: ExperimentConfig,
    *,
    families: Sequence[str] = KINDS,
    fast: bool = False,
    run_name: str = DEFAULT_CV_RUN_NAME,
    out_dir: Path | None = None,
    version: int = DEFAULT_FINAL_VERSION,
    command: str = "",
    overwrite: bool = False,
    allow_partial_fps_sim: bool = False,
    rebuild: bool = False,
) -> FitResult:
    """최종 모델을 학습해 내보낸다 (PLAN 11장 E4, D10).

    1. 전체 피험자에 같은 선택 절차를 한 번 적용한다: 공식 fold마다 그 fold를 검증으로 두고(`final_splits`) 후보를
       모두 학습해, 계열·설정·학습 길이·온도를 고른다(`choose_candidate`, 시드는 `training.seeds`의 첫 값).
    2. 전원으로 `training.final_seed`(0) 학습을 한다. 표준화 상수는 전원의 학습 창에서 계산한다. 학습 표본의 출처
       (원본 추출, 15 fps 누락 모사 추출, 프레임 누락 변형)와 몫은 교차검증과 같다(`mix_sources`). 선택 단계에서는
       분할마다 그 분할의 학습 피험자의 창만 쓴다.
    3. 학습한 평균·표준편차·온도를 feature_spec 사본에 써서 `export_onnx`로 내보내고 검증한다(`export_fit`).
    4. 내보낸 모델 옆에 규칙 설정 사본 `rules.json`을 쓴다(`spec.export_rules_json`, 앱이 읽는 형식). 그 바이트의
       sha256이 rules_hash이고 `final.json`에 적는다.
       가정: `paths.rules`(config/rules.yaml)의 내용을 그대로 쓴다. D11 판정 결과에 따라 `drowsiness.source`를
       바꾸는 일은 하지 않는다(규칙 파일을 고치는 것은 rules_hash를 바꾸는 결정이라 사람이 한다).
    학습 조건은 `training.export_arm`이다. 내보내는 위치는 out_dir(None이면 `paths.export_dir`), 버전은 version이다.
    `{reports_dir}/{run_name}/final/`에 선택 기록과 내보내기 검증 기록(`final.json`, `summary.md`), 체크포인트를 쓴다.
    summary.md의 머리말에는 교차검증 리포트와 같은 자리에 D 번호의 뜻을 알리는 한 줄(`evaluate.PLAN_DECISION_NOTE`)이 있다.
    체크포인트(v2)에는 내보낸 feature_spec 사본의 sha256, 합성 데이터 여부, 축소 설정 여부(`fast`)가 들어간다.
    `python -m drowsy.export_onnx --checkpoint`는 그 값들로 합성 데이터 모델과 축소 설정 모델을 `paths.export_dir`에
    다시 내보내지 못하게 막는다. `final.json`에는 추출 결과의 회전 점검 기록(`rotation_check`)도 적고, 회전이 확인되지 않은
    영상이 있으면 그 수가 설명(`description`, summary.md의 첫 문단)에 실린다(`rotation_note`).

    내보내기 폴더(`paths.export_dir`, 그리고 설정이 무엇을 가리키든 이 저장소의 `export/`. 앱 팀에 넘기는 폴더다)와 그 아래
    폴더로는 내보내지 않는 것 (ValueError. out_dir을 따로 줘야 한다)
    - 합성 데이터로 학습한 모델(`export_onnx.refuse_synthetic_export`).
    - 축소 설정(fast)으로 학습한 모델(`export_onnx.refuse_fast_export`). 사전에 고정한 절차의 모델이 아닌데 `trained_on`만으로는
      구분되지 않는다.
      가정: `trained_on`에 축소 설정 표시를 넣지 않는다(메타데이터 값의 형식을 바꾸지 않는다). 대신 위치를 막고
      `final.json`의 `fast`에 적는다.
    두 검사는 같은 함수(`export_onnx.protected_export_dir`. `is_export_dir_or_below`가 그 판단을 참·거짓으로 돌려준다)로
    폴더를 견준다. 상대 경로로 적힌 `paths.export_dir`은 실행 위치 기준과 저장소 루트 기준 두 폴더를 모두 가리키는 것으로
    보므로, 저장소 루트가 아닌 곳에서 실행해도 막힌다. 연기 시험용 설정처럼 `paths.export_dir`이 다른 폴더를 가리켜도
    저장소의 `export/`로는 나가지 않는다.
    `{reports_dir}/{run_name}/final/`에 다른 실행(다른 학습 조건, 축소 설정 여부, 데이터 종류)의 기록과 체크포인트가
    있으면 학습 전에 ValueError로 멈춘다(`check_run_dir`). 실제 데이터로 만든 E4의 기록을 연기 시험이나 합성 데이터
    실행이 덮어쓰지 않게 한다. overwrite를 주면 덮어쓴다.
    allow_partial_fps_sim, rebuild는 `load_cv_data`에 그대로 넘긴다.
    이 모델은 전원으로 학습했으므로 보고하는 수치는 교차검증 추정치이고 이 파일을 시험한 값이 아니다(D10).
    """
    if fast:
        cfg = apply_fast_profile(cfg)
    families = tuple(families)
    arm = cfg.training.export_arm
    out_dir = cfg.paths.export_dir if out_dir is None else Path(out_dir)
    run_dir = cfg.paths.reports_dir / run_name / FINAL_SUBDIR
    total_started = time.perf_counter()

    if fast:
        export_onnx.refuse_fast_export(out_dir, cfg.paths.export_dir)
    data = load_cv_data(cfg, (arm,), mix_arms=(arm,), allow_partial_fps_sim=allow_partial_fps_sim, rebuild=rebuild)
    if data.synthetic:
        export_onnx.refuse_synthetic_export(out_dir, cfg.paths.export_dir)
    check_run_dir(run_dir, {"arm": arm, "fast": fast, "synthetic": data.synthetic}, overwrite, FINAL_RECORD_FILE)
    table = data.tables[arm]
    sources = mix_sources(data, cfg, arm)  # 전원의 학습 전용 출처(D10: 최종 모델은 전원으로 학습한다)
    base = _base_train_config(cfg, data.feature_spec)
    grid = candidates(cfg.training, families)
    splits = final_splits(data.manifest)
    window_steps = cfg.windows.window_steps
    timings: dict[str, float] = {"load_data_s": time.perf_counter() - total_started}

    started = time.perf_counter()
    jobs = [
        _ScoreJob(
            candidate=candidate,
            splits=splits,
            steps=table.steps,
            extra_steps=sources,
            base=base,
            window_steps=window_steps,
            eval_stride=cfg.windows.eval_stride,
            seed=cfg.training.seeds[0],
        )
        for candidate in grid
    ]
    scores: list[CandidateScore] = []
    for index, job in enumerate(jobs):
        scores.append(_score_job(job))
        _progress(
            f"selection {index + 1}/{len(jobs)} ({job.candidate.name}) done, "
            f"{time.perf_counter() - started:.0f} s elapsed"
        )
    chosen = choose_candidate(scores)
    timings["selection_s"] = time.perf_counter() - started

    started = time.perf_counter()
    windows = make_windows(table.steps, window_steps, 1)
    extra_windows = tuple(source.windows(window_steps) for source in sources)
    final_cfg = dataclasses.replace(
        base,
        kind=chosen.candidate.family,
        hyper=chosen.candidate.hyper,
        fixed_units=chosen.units,
        temperature=chosen.temperature,
    )
    result = fit(windows, None, final_cfg, cfg.training.final_seed, extra_train=extra_windows)
    timings["final_fit_s"] = time.perf_counter() - started

    started = time.perf_counter()
    trained_on = trained_on_text(data, arm)
    picks = np.random.default_rng(0).permutation(len(windows))[:_PARITY_REAL_WINDOWS]
    record = export_fit(
        result, cfg.paths.feature_spec, out_dir, version, trained_on, cfg.export, real_windows=windows.x[np.sort(picks)]
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / FINAL_CHECKPOINT_FILE
    save_checkpoint(
        checkpoint_path,
        result.net,
        result.kind,
        result.spec.n_features,
        result.hyper,
        result.temperature,
        trained_on,
        record["feature_spec_hash"],
        data.synthetic,
        fast=fast,
    )
    rules_path = Path(out_dir) / FINAL_RULES_FILE
    rules_hash_value = export_rules_json(rules_for(cfg, data.feature_spec), rules_path)
    timings["export_s"] = time.perf_counter() - started
    timings["total_s"] = time.perf_counter() - total_started

    note = data_note(data.synthetic)
    description = (
        f"{data.description}. Final model (PLAN section 11, E4): arm {arm}, families {', '.join(families)}, "
        f"selection over {len(splits)} folds, refit on all subjects with seed {cfg.training.final_seed}"
        + (f". {FAST_NOTE}" if fast else "")
    )
    write_json(
        {
            "data_note": note,
            "synthetic": data.synthetic,
            "description": description,
            "command": command,
            "arm": arm,
            "fast": fast,
            "trained_on": trained_on,
            "train_sources": {name: share for name, share in result.sources},
            "pipeline_hash": table.pipeline_hash,
            "step_fingerprints": {arm: step_table_fingerprint(table.steps)},
            "rotation_check": rotation_record(data.rotation),
            "rules_path": str(rules_path),
            "rules_hash": rules_hash_value,
            "subjects": sorted(int(value) for value in data.manifest["subject"].unique()),
            "selection": {
                "val_folds": [int(split.val_fold) for split in splits],
                "candidates": [_score_record(score) for score in scores],
                "chosen": chosen.candidate.name,
            },
            "final_fit": {
                "kind": result.kind,
                "hyper": result.hyper,
                "n_parameters": result.n_parameters,
                "seed": result.seed,
                "units": result.units,
                "updates": result.updates,
                "temperature": result.temperature,
                "checkpoint": str(checkpoint_path),
            },
            "export": record,
            "seconds": {key: float(value) for key, value in timings.items()},
        },
        run_dir / FINAL_RECORD_FILE,
    )
    int8 = record["int8"]
    lines = [
        f"# {note}",
        "",
        description,
        "",
        (
            "합성 데이터로 학습한 모델이다. 전 과정이 도는지 확인하는 용도이며 이 모델에 대해 보고하는 성능 수치는 없다."
            if data.synthetic
            else "이 모델은 전체 피험자로 학습했다. 보고하는 수치는 같은 절차의 교차검증 추정치이고 "
            "이 파일을 시험한 값이 아니다 (D10)."
        ),
        "",
        evaluate.PLAN_DECISION_NOTE,  # 아래 항목의 D10, D16이 무엇의 번호인지 알린다(교차검증 리포트와 같은 자리)
        "",
        f"- 생성 명령: `{command}`",
        f"- 고른 설정: `{chosen.candidate.name}` (파라미터 {result.n_parameters}개), 학습 길이 {result.units}단위"
        f"({result.updates} 갱신), 온도 {result.temperature:.4f}, 시드 {result.seed}",
        f"- 모델: `{record['onnx_path']}` ({record['onnx_bytes']} bytes, sha256 `{record['onnx_sha256']}`)",
        f"- feature_spec 사본: `{record['spec_path']}` (feature_spec_hash `{record['feature_spec_hash']}`)",
        f"- 규칙 설정 사본: `{rules_path}` (rules_hash `{rules_hash_value}`)",
        f"- trained_on: `{trained_on}`",
        "- 계약 검사(`verify_contract`): 통과",
        f"- 일치 검증: 무작위 입력 {record['parity_random_inputs']}개 최대 절대 오차 "
        f"{record['parity_random_max_abs_diff']:.3e}, 실제 창 {record['parity_real_windows']}개 "
        f"{record['parity_real_max_abs_diff']:.3e} (허용 {record['parity_tolerance']:.0e})",
        f"- 지연(onnxruntime CPU, {record['latency_runs']}회): 중앙값 {record['latency']['median_ms']:.4f} ms, "
        f"95 분위수 {record['latency']['p95_ms']:.4f} ms",
        (
            f"- INT8: `{int8['onnx_path']}` ({int8['onnx_bytes']} bytes), 계약 검사 통과, float 모델과의 확률 최대 절대 "
            f"차이 {int8['max_abs_diff_vs_float']:.3e} ({int8['inputs']}). 이 값은 확률의 차이일 뿐 지표 변화가 아니다. "
            f"지표 변화는 `train cv` 리포트 9절의 부가 묶음 `{FAMILY_TCN_ONNX}` 대 `{FAMILY_TCN_ONNX_INT8}`"
            "(fold 모델의 교차검증 추정치)에 있다"
            if int8 is not None
            else "- INT8: 만들지 않았다(D16: TCN이 최종 모델일 때만 만든다)"
        ),
        f"- 체크포인트: `{checkpoint_path}`, 기록: `{FINAL_RECORD_FILE}`",
        "",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# 명령
# ---------------------------------------------------------------------------


def with_data_root(cfg: ExperimentConfig, data_root: Path | None) -> ExperimentConfig:
    """`--data-root`를 적용한 설정 사본.

    가정: `--data-root`는 설정이 가리키는 데이터 루트를 바꾼다. `paths.rldd_root`가 있으면 그것(영상 폴더)을,
    null이면 `paths.synth_dir`(합성 데이터 폴더)을 바꾼다. 이 옵션으로 합성 데이터 설정을 실제 데이터로 바꿀 수는 없다.
    실제 데이터로 돌리려면 `config/default.yaml`을 git에 넣지 않는 `config/local.yaml`로 복사해 `paths.rldd_root`에
    경로를 넣고 `--config config/local.yaml`을 준다. 추적되는 기본 설정은 고치지 않는다.
    """
    if data_root is None:
        return cfg
    if cfg.paths.rldd_root is not None:
        paths = dataclasses.replace(cfg.paths, rldd_root=Path(data_root))
    else:
        paths = dataclasses.replace(cfg.paths, synth_dir=Path(data_root))
    return dataclasses.replace(cfg, paths=paths)


def add_common_arguments(parser: argparse.ArgumentParser, suppress: bool = False) -> None:
    """`--config`, `--data-root`, `--run-name`, `--rebuild`. 하위 명령 뒤에 적어도 받도록 하위 명령에도 같은 옵션을 둔다."""

    def default(value: Any) -> Any:
        return argparse.SUPPRESS if suppress else value

    parser.add_argument("--config", default=default("config/default.yaml"), help="experiment config YAML")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default(None),
        help="override the data root of the config (paths.rldd_root if set, otherwise paths.synth_dir)",
    )
    parser.add_argument(
        "--run-name", default=default(DEFAULT_CV_RUN_NAME), help="report folder name below paths.reports_dir"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        default=default(False),
        help="ignore the step-table cache and rebuild it (as `python -m drowsy.dataset --rebuild`)",
    )


def command_line(module: str, argv: Sequence[str] | None) -> str:
    """리포트에 남기는 명령 문자열."""
    arguments = sys.argv[1:] if argv is None else list(argv)
    return " ".join([f"python -m {module}", *arguments])


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m drowsy.train`. 종료 코드: 0 성공, 2 시작할 수 없음(실행 이름, 설정, 데이터), 3 피험자 겹침으로 중단.

    `--run-name`은 폴더 이름 하나여야 한다(`config.check_run_name`). 경로 구분자나 `..`이 든 이름은 설정과 데이터를 읽기
    전에 `error: ...` 한 줄과 종료 코드 2로 거부한다(리포트 폴더 밖에 쓰게 되는 이름이다).
    출력의 첫 줄은 데이터 설명, 둘째 줄은 추출 결과의 회전 점검 기록이다(`print_rotation_check`). 회전이 확인되지 않은
    영상이 있어도 종료 코드는 바뀌지 않는다.
    """
    parser = argparse.ArgumentParser(
        prog="python -m drowsy.train", description="Nested cross-validation and the final model (PLAN 5.2, 6, 11)."
    )
    add_common_arguments(parser)
    commands = parser.add_subparsers(dest="command", required=True)
    cv = commands.add_parser("cv", help="nested cross-validation with reports")
    add_common_arguments(cv, suppress=True)
    cv.add_argument("--arm", choices=ARMS, default=None, help="training arm (default: training.export_arm)")
    options = cv.add_mutually_exclusive_group()
    options.add_argument(
        "--ordinal",
        action="store_true",
        help="supplementary run (E3): ordinal soft labels (training.ordinal_neighbor_mass); report in <run>/ordinal/",
    )
    options.add_argument(
        "--no-velocity",
        action="store_true",
        help="supplementary run (E3): velocity features #7 and #8 held at 0; report in <run>/no_velocity/",
    )
    final = commands.add_parser("final", help="selection over all subjects, final fit and ONNX export")
    add_common_arguments(final, suppress=True)
    final.add_argument("--out-dir", type=Path, default=None, help="export folder (default: paths.export_dir)")
    final.add_argument("--version", type=int, default=DEFAULT_FINAL_VERSION, help="model version N")
    for sub in (cv, final):
        sub.add_argument("--families", nargs="+", choices=KINDS, default=list(KINDS), help="model families")
        sub.add_argument("--fast", action="store_true", help="smoke-run profile (config block `fast`)")
        sub.add_argument(
            "--overwrite",
            action="store_true",
            help="replace the report of another run (other arm, profile or kind of data) in the same run folder",
        )
        sub.add_argument(
            "--allow-partial-fps-sim",
            action="store_true",
            help="train although the frame-drop simulated extraction lacks sessions of the native extraction",
        )
    args = parser.parse_args(argv)
    command = command_line("drowsy.train", argv)

    try:
        check_run_name(args.run_name)  # 두 하위 명령(cv, final) 모두 다른 일을 하기 전에 확인한다
        cfg = with_data_root(load_config(args.config), args.data_root)
        source = open_dataset(cfg)
        print(source.description, flush=True)
        print_rotation_check(cfg, source)  # 학습을 시작하기 전에 알린다. 같은 기록이 실행 기록과 리포트 설명에도 남는다
        if args.command == "cv":
            variant = evaluate.VARIANT_BASE
            if args.ordinal:
                variant = evaluate.VARIANT_ORDINAL
            elif args.no_velocity:
                variant = evaluate.VARIANT_NO_VELOCITY
            result = run_nested_cv(
                cfg,
                arm=args.arm,
                families=args.families,
                fast=args.fast,
                run_name=args.run_name,
                config_path=Path(args.config),
                command=command,
                overwrite=args.overwrite,
                allow_partial_fps_sim=args.allow_partial_fps_sim,
                rebuild=args.rebuild,
                variant=variant,
            )
            for record in result.selection:
                picks = ", ".join(f"{label}: {name}" for label, name in record["chosen"].items())
                print(f"fold {record['fold']}: {picks}")
            print("seconds: " + ", ".join(f"{key} {value:.1f}" for key, value in result.timings.items()))
            gate = result.report.gate
            trained_arm = args.arm if args.arm is not None else cfg.training.export_arm
            if variant != evaluate.VARIANT_BASE:
                print(
                    f"supplementary run ({variant}): every group is supplementary; the main report, its headline "
                    "rows and D11 are not touched"
                )
            elif trained_arm != cfg.training.export_arm:
                print(
                    f"supplementary analysis (training arm {trained_arm} is not the export arm "
                    f"{cfg.training.export_arm}): every group is supplementary and D11 is not evaluated"
                )
            elif gate is None:
                print(
                    "D11 gate: not evaluated (it needs the alert_baseline training arm and the reference LightGBM "
                    f"predictions in {result.run_dir / LGBM_SUBDIR}; run `python -m drowsy.baseline_lgbm cv` first)"
                )
            else:
                print(
                    f"D11 gate: {'all conditions met' if gate.passed else 'not met'}; "
                    f"drowsiness.source stays {result.report.drowsiness_source}"
                )
            print(f"reports: {result.report.summary_path}")
        else:
            fitted = fit_final(
                cfg,
                families=args.families,
                fast=args.fast,
                run_name=args.run_name,
                out_dir=args.out_dir,
                version=args.version,
                command=command,
                overwrite=args.overwrite,
                allow_partial_fps_sim=args.allow_partial_fps_sim,
                rebuild=args.rebuild,
            )
            run_dir = cfg.paths.reports_dir / args.run_name / FINAL_SUBDIR
            record = json.loads((run_dir / FINAL_RECORD_FILE).read_text(encoding="utf-8"))
            exported = record["export"]
            print(f"final model: {fitted.kind} {fitted.hyper}, units {fitted.units}, seed {fitted.seed}")
            print(f"wrote {exported['onnx_path']} ({exported['onnx_bytes']} bytes) and {exported['spec_path']}")
            print(f"wrote {record['rules_path']} (rules_hash {record['rules_hash']})")
            print(f"trained_on: {record['trained_on']}")
            print(
                "verify_contract: OK; parity max abs diff "
                f"{exported['parity_random_max_abs_diff']:.3e} (random), "
                f"{exported['parity_real_max_abs_diff']:.3e} (real windows), "
                f"tolerance {exported['parity_tolerance']:.0e}"
            )
            print("seconds: " + ", ".join(f"{key} {value:.1f}" for key, value in record["seconds"].items()))
            print(f"record: {run_dir / FINAL_RECORD_FILE}")
    except SubjectOverlapError as exc:
        print(f"error: subject overlap, run stopped: {exc}", file=sys.stderr)
        return 3
    except (ValueError, OSError, ExtractError) as exc:  # SpecError, ManifestError, ContractError도 ValueError다
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
