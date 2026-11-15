"""LightGBM 비교 기준선과 교란 점검 (PLAN 5.3절, 6장, 11장 E1). 이식 대상이 아니다. 이 모델은 내보내지 않는다.

- `window_stats`: 창 `[N, 120, F]`를 특징별 평균, 표준편차, 최소, 최대, 마지막 값, 선형 기울기 `[N, 6F]`로 줄인다.
- `lgbm_cross_validate` / `run_lgbm_cv`: 신경망과 같은 fold, 같은 창, 같은 안쪽 선택 절차(`drowsy.train.choose`)로
  중첩 교차검증을 하고 같은 형식의 예측 표와 리포트를 만든다. 탐색 범위는 `num_leaves` × `min_data_in_leaf`다.
  원본 프레임률 추출로만 학습한다(`NATIVE_ONLY_NOTE`).
- `run_lgbm_w_grid`: E1의 closure W 비교. `lgbm.w_grid`의 W마다 기준 조건의 교차검증을 돌려 표 하나로 합친다(보고용).
- 부가 실행(`run_lgbm_cv`의 선택 사항. 리포트 9절에만 실리고 `lgbm/`과 D11에는 쓰이지 않는다)
  - `cv --no-velocity`: 속도 특징 #7, #8의 통계 12개를 입력에서 뺀 실행(PLAN 4.6절, 11장 E3).
  - `cv --arms global --foldwise-global`: 전역 상수를 바깥 fold마다 학습 피험자만으로 다시 계산해 전역 조건의 특징을
    다시 만든 민감도 점검(PLAN 4.3절, `foldwise_global_tables`).
- 교란 점검 두 가지(`recording_metadata_probe`, `baseline_offset_probe`): 영상마다 한 행인 표(`probe_table`)의
  열만으로 라벨을 맞혀 본다. 보고만 하고 선택에는 쓰지 않는다. 점검 (1)의 밝기, 해상도, 평균 프레임률, 가변 프레임률
  표시는 추출이 parquet 메타데이터에 적어 둔 값을 쓴다. 그 값이 없는 데이터(합성 세션)에서는 밝기와 가변 프레임률
  표시 없이 돌고 리포트에 그렇게 적는다.

리포트 폴더 `{paths.reports_dir}/{실행명}/`
    lgbm/         교차검증: summary.md 등(`evaluate.write_reports`), predictions.parquet, selection.json, run.json
                  (closure W를 바꾼 실행은 `lgbm_w{W}/`). 비교표와 D11의 기준은 이 폴더의 기준 조건 묶음이다
    lgbm_w_grid/  `cv --all-w`: W 비교. 같은 파일들과 합친 표 `w_grid.md`, `w_grid.csv`
    lgbm_no_velocity/   `cv --no-velocity`: 속도 특징을 뺀 부가 실행. `lgbm/`의 주 실행이 있으면 짝지은 비교를 싣는다
    lgbm_fold_global/   `cv --arms global --foldwise-global`: feature_spec 상수의 묶음(주)과 fold별 상수의 묶음(부가)
    probes/       교란 점검: summary.md, probes.json, probe_table.csv
같은 폴더에 다른 실행(다른 조건 목록, 축소 설정 여부, closure W, 데이터 종류)의 리포트가 있으면 덮어쓰지 않고 멈춘다
(`--overwrite`를 주면 덮어쓴다).

명령 (저장소 루트에서 실행. 공통 옵션은 `drowsy.train`과 같다)
    python -m drowsy.baseline_lgbm --config config/default.yaml [--data-root DIR] [--run-name NAME] [--rebuild] cv
        [--arms alert_baseline ...] [--fast] [--closure-w W | --all-w | --no-velocity | --foldwise-global]
        [--overwrite]
    python -m drowsy.baseline_lgbm --config config/default.yaml [--data-root DIR] [--run-name NAME] [--rebuild] probes
        [--arm alert_baseline] [--workers N]
합성 데이터로 만든 출력은 첫 줄에 합성 데이터임을 적는다. 합성 데이터의 점수는 파이프라인 점검용이다(PLAN 3.4절).
두 명령 모두 데이터 설명 다음 줄에, 학습과 재생을 시작하기 전에, 추출 결과의 회전 점검 기록을 구분별 파일 수로 출력한다
(`train.print_rotation_check`). 회전이 확인되지 않은 영상이 있으면 그 아래에 영상마다 한 줄씩 적고, 실행 기록(`run.json`,
`probes.json`)의 `rotation_check`에 목록을, 리포트 설명에 그 수를 남긴다.

D7과 D11은 계획서(docs/PLAN.md 1.2절 표)의 결정 번호다. D7: 창이 하나도 없는 영상은 영상 단위 정확도에서 오답으로 센다.
D11: 교차검증 결과가 정해 둔 조건을 모두 채울 때만 모델을 기본 졸음 판정원으로 둔다. 그 조건의 하나가 이 모듈의 기준
LightGBM과의 macro-F1 비교다.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import math
import multiprocessing
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from drowsy import calibrate, evaluate, signal_check, train
from drowsy.config import (
    ARM_GLOBAL,
    ARMS,
    ExperimentConfig,
    LgbmSettings,
    apply_fast_profile,
    check_run_name,
    load_config,
)
from drowsy.dataset import (
    StepTable,
    SubjectOverlapError,
    WindowSet,
    assert_subject_disjoint,
    load_step_tables,
    make_windows,
    nested_splits,
    open_dataset,
    spec_pipeline_hash,
)
from drowsy.extract import ExtractError, raw_parquet_path, read_extract_metadata, read_raw_frames
from drowsy.pipeline import DriverPipeline
from drowsy.spec import PipelineSpec, RulesConfig, apply_overrides, load_feature_spec
from drowsy.types import CLASS_ALERT, CLASS_ORDER, FEATURE_NAMES, Baseline, RawFrame

STAT_NAMES: tuple[str, ...] = ("mean", "std", "min", "max", "last", "slope")
W_GRID_SUBDIR = "lgbm_w_grid"  # `cv --all-w`의 리포트 폴더
NO_VELOCITY_SUBDIR = f"{train.LGBM_SUBDIR}_{evaluate.VARIANT_NO_VELOCITY}"  # `cv --no-velocity`: lgbm_no_velocity/
FOLD_GLOBAL_SUBDIR = f"{train.LGBM_SUBDIR}_{evaluate.FEATURE_SOURCE_FOLD_GLOBAL}"  # `cv --foldwise-global`
W_GRID_CSV_FILE = "w_grid.csv"
W_GRID_MD_FILE = "w_grid.md"
W_GRID_METRICS: tuple[str, ...] = evaluate.PAIRED_METRICS  # 합친 표에 싣는 지표(짝지은 비교의 지표와 같다)
# 기준 LightGBM은 원본 프레임률 추출로만 학습한다. 리포트 설명(맨 앞 문단)과 명령의 출력에 이 문장을 싣는다.
# 신경망이 섞는 두 증강(PLAN 5.2절: 15 fps 누락 모사 추출, 프레임 누락 변형)은 여기서 쓰지 않는다.
NATIVE_ONLY_NOTE = (
    "LightGBM trains on the native-rate extraction only; the networks also draw training samples from the "
    "frame-drop simulated (15 fps) extraction when it exists and from frame-drop variants of the native extraction, "
    "so the D11 reference is not trained on those augmentations"
)
LGBM_THREADS = 4  # 표가 작아 스레드를 늘려도 빨라지지 않는다. deterministic 설정이라 결과는 스레드 수와 무관하다

# 교란 점검용 표의 열
# 가정: 가변 프레임률 표시(`variable_fps`)는 계획서 6장의 목록(평균 프레임률, 해상도, 얼굴 크기, 얼굴 검출률, 밝기)에
# 없다. 추출이 함께 기록하는 녹화 조건이고 라벨과 함께 움직일 수 있어 점검 (1)의 입력에 넣는다.
RECORDING_COLUMNS: tuple[str, ...] = (
    "mean_fps", "width", "height", "face_size", "face_ratio", "brightness", "variable_fps",
)  # fmt: skip
# 추출 결과 parquet의 메타데이터에서만 얻을 수 있는 열(`extract.extract_video`가 적는다. 합성 데이터에는 없다)
METADATA_ONLY_COLUMNS: tuple[str, ...] = ("brightness", "variable_fps")
OFFSET_COLUMNS: tuple[str, ...] = ("open_closure_median", "yaw_rel_median", "pitch_rel_median")
PROBE_ID_COLUMNS: tuple[str, ...] = ("video_id", "subject", "fold", "class_index")
PROBE_RECORDING = "recording_metadata"
PROBE_BASELINE_OFFSET = "baseline_offset"
# 가정: 교란 점검의 모델 크기는 계획서에 없다. 영상 180개짜리 표에 맞춰 작게 고정한다(선택에 쓰지 않으므로 탐색하지 않는다).
PROBE_MIN_DATA_IN_LEAF = 5
PROBE_ROUNDS = 100
CHANCE_ACCURACY = 1.0 / len(CLASS_ORDER)

_BASELINE_SESSION = "session"  # 품질 표의 baseline_used 값(`dataset.StepTable`)
_BASELINE_ALERT = "alert"
_BASELINE_GLOBAL = "global"


# ---------------------------------------------------------------------------
# 창 통계
# ---------------------------------------------------------------------------


def window_stats(x: np.ndarray) -> np.ndarray:
    """창 `[N, T, F]`를 특징별 통계 `[N, 6F]`(float64)로 줄인다 (PLAN 5.3절). 120 × 17이면 102개다.

    통계는 `STAT_NAMES` 순서로 블록을 이룬다: 열 `s·F + f`가 특징 f의 s번째 통계다(`window_stat_names`).
    - mean, std(모표준편차), min, max: 창의 T 스텝에 대한 값.
    - last: 가장 최근 스텝의 값.
    - slope: 스텝 번호 0..T−1에 대한 최소제곱 직선의 기울기(스텝당 변화량). T가 1이면 0이다.
    원시 특징을 그대로 쓴다(나무 모델이라 표준화하지 않는다).
    """
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"windows must have shape [N, steps, features], got {list(values.shape)}")
    n_steps = values.shape[1]
    if n_steps == 0:
        raise ValueError("windows need at least one step")
    mean = values.mean(axis=1)
    centred_t = np.arange(n_steps, dtype=np.float64) - (n_steps - 1) / 2.0
    spread = float((centred_t * centred_t).sum())
    if spread > 0.0:
        slope = np.einsum("t,ntf->nf", centred_t, values - mean[:, None, :]) / spread
    else:
        slope = np.zeros_like(mean)
    return np.concatenate(
        [mean, values.std(axis=1), values.min(axis=1), values.max(axis=1), values[:, -1, :], slope], axis=1
    )


def window_stat_names(feature_names: Sequence[str] = FEATURE_NAMES) -> tuple[str, ...]:
    """`window_stats` 열의 이름: `{특징}_{통계}`, 통계별 블록 순서."""
    return tuple(f"{feature}_{stat}" for stat in STAT_NAMES for feature in feature_names)


def stat_columns_without(features: Sequence[int], n_features: int = len(FEATURE_NAMES)) -> np.ndarray:
    """`window_stats` 열 가운데 주어진 특징들의 통계를 뺀 열 번호(오름차순). 특징 하나에 통계 6개가 빠진다.

    속도 특징 #7, #8을 뺀 실행(`cv --no-velocity`)은 102개 중 12개를 빼고 90개 열로 학습하고 평가한다.
    """
    dropped = {int(feature) for feature in features}
    unknown = sorted(feature for feature in dropped if not 0 <= feature < n_features)
    if unknown:
        raise ValueError(f"no feature with index {unknown} among {n_features} features")
    keep = [
        stat * n_features + feature
        for stat in range(len(STAT_NAMES))
        for feature in range(n_features)
        if feature not in dropped
    ]
    return np.array(keep, dtype=np.int64)


def sample_weights(meta: pd.DataFrame) -> np.ndarray:
    """신경망의 표집(피험자 → 영상 → 창 균등)과 같은 비중이 되게 하는 창별 가중치. 평균이 1이다.

    창 하나의 가중치는 `1 / (피험자 수 × 그 피험자의 영상 수 × 그 영상의 창 수)`에 비례한다.
    """
    if len(meta) == 0:
        return np.zeros(0, dtype=np.float64)
    keys = list(evaluate.VIDEO_KEYS)
    per_video = meta.groupby(keys, sort=False)["subject"].transform("size").to_numpy(dtype=np.float64)
    videos = meta[keys].drop_duplicates()
    per_subject = videos.groupby("subject", sort=False)["video_id"].size()
    n_videos = meta["subject"].map(per_subject).to_numpy(dtype=np.float64)
    weights = 1.0 / (n_videos * per_video)
    return weights * (len(weights) / weights.sum())


# ---------------------------------------------------------------------------
# 학습과 안쪽 선택
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LgbmCandidate:
    """탐색 범위의 후보 하나 (PLAN 5.3절)."""

    num_leaves: int
    min_data_in_leaf: int

    @property
    def name(self) -> str:
        return f"lgbm(num_leaves={self.num_leaves},min_data_in_leaf={self.min_data_in_leaf})"


@dataclass(frozen=True, slots=True)
class LgbmScore:
    """후보 하나의 안쪽 검증 결과. rounds는 다시 학습할 때의 부스팅 횟수(fold별 값의 중앙값)다."""

    candidate: LgbmCandidate
    val_folds: tuple[int, ...]
    fold_macro_f1: tuple[float, ...]
    mean_macro_f1: float
    std_macro_f1: float
    fold_rounds: tuple[int, ...]
    rounds: int


def lgbm_candidates(settings: LgbmSettings) -> tuple[LgbmCandidate, ...]:
    return tuple(
        LgbmCandidate(int(leaves), int(min_data))
        for leaves in settings.num_leaves
        for min_data in settings.min_data_in_leaf
    )


def lgbm_params(settings: LgbmSettings, num_leaves: int, min_data_in_leaf: int) -> dict[str, Any]:
    """LightGBM 설정. 무작위 요소(배깅, 열 표집)를 쓰지 않고 deterministic으로 둬서 같은 입력이면 같은 모델이 나온다."""
    return {
        "objective": "multiclass",
        "num_class": len(CLASS_ORDER),
        "learning_rate": settings.learning_rate,
        "num_leaves": num_leaves,
        "min_data_in_leaf": min_data_in_leaf,
        "seed": settings.seed,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": LGBM_THREADS,
        "metric": "None",  # 조기 종료는 아래의 영상 가중 macro-F1만 본다
        "verbosity": -1,
    }


def fit_lgbm(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    params: dict[str, Any],
    rounds: int,
    val: tuple[np.ndarray, pd.DataFrame] | None = None,
    early_stopping_rounds: int = 0,
    feature_names: Sequence[str] | None = None,
) -> tuple[lgb.Booster, int, float]:
    """LightGBM 하나를 학습한다. `(모델, 채택한 부스팅 횟수, 내부 검증 macro-F1)`을 돌려준다.

    val(검증 창의 통계와 메타데이터)을 주면 회차마다 영상 가중 macro-F1(`train.VideoMacroF1`, 신경망의 조기 종료와 같은
    지표)을 계산해 early_stopping_rounds 동안 나아지지 않으면 멈추고, 가장 좋았던 회차를 채택한다(최대 rounds).
    주지 않으면 rounds번을 그대로 학습하고 macro-F1은 NaN이다.
    feature_names는 x의 열 이름이다. None이면 `window_stat_names()`(102개)다. 열을 뺀 실행은 남은 열의 이름을 준다.
    """
    names = list(window_stat_names()) if feature_names is None else [str(name) for name in feature_names]
    if len(names) != np.shape(x)[1]:
        raise ValueError(f"x has {np.shape(x)[1]} columns but {len(names)} feature names were given")
    train_set = lgb.Dataset(x, label=y, weight=weights, feature_name=names, free_raw_data=False)
    if val is None:
        return lgb.train(params, train_set, num_boost_round=rounds), rounds, math.nan
    val_x, val_meta = val
    scorer = train.VideoMacroF1(val_meta)

    def macro_f1(scores: np.ndarray, _: lgb.Dataset) -> tuple[str, float, bool]:
        return "macro_f1", scorer(np.argmax(scores.reshape(len(val_meta), -1), axis=1)), True

    val_set = lgb.Dataset(val_x, label=val_meta["class_index"].to_numpy(), reference=train_set)
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=rounds,
        valid_sets=[val_set],
        valid_names=["val"],
        feval=macro_f1,
        callbacks=[lgb.early_stopping(early_stopping_rounds, first_metric_only=True, verbose=False)],
    )
    return booster, int(booster.best_iteration), float(booster.best_score["val"]["macro_f1"])


@dataclass(frozen=True, eq=False)
class _Stats:
    """창 묶음과 그 통계 행렬."""

    windows: WindowSet
    stats: np.ndarray

    def of(self, subjects: Iterable[int]) -> _Stats:
        wanted = np.array(sorted({int(value) for value in subjects}), dtype=np.int64)
        mask = np.isin(self.windows.meta["subject"].to_numpy(dtype=np.int64), wanted)
        return _Stats(self.windows.subset(mask), self.stats[mask])


def _stats(windows: WindowSet, columns: np.ndarray | None = None) -> _Stats:
    """창 묶음의 통계 행렬. columns를 주면 그 열만 남긴다(`stat_columns_without`)."""
    stats = window_stats(windows.x)
    return _Stats(windows, stats if columns is None else stats[:, columns])


def _score_record(score: LgbmScore) -> dict[str, Any]:
    return {
        "name": score.candidate.name,
        "num_leaves": score.candidate.num_leaves,
        "min_data_in_leaf": score.candidate.min_data_in_leaf,
        "val_folds": list(score.val_folds),
        "fold_macro_f1": list(score.fold_macro_f1),
        "mean_macro_f1": score.mean_macro_f1,
        "std_macro_f1": score.std_macro_f1,
        "fold_rounds": list(score.fold_rounds),
        "rounds": score.rounds,
    }


def choose_lgbm(scores: Sequence[LgbmScore]) -> LgbmScore:
    """신경망과 같은 선택 규칙(`train.choose`).

    가정: LightGBM에는 "파라미터 수"가 없다. 잎이 적을수록, 잎 수가 같으면 `min_data_in_leaf`가 클수록 단순한 것으로 본다.
    """
    return train.choose(
        scores,
        complexity=lambda score: (float(score.candidate.num_leaves), -float(score.candidate.min_data_in_leaf)),
        mean=lambda score: score.mean_macro_f1,
        std=lambda score: score.std_macro_f1,
    )


def model_sha256(booster: lgb.Booster, rounds: int) -> str:
    """LightGBM 모델(처음 rounds회차까지)의 문자열 표현의 sha256 16진수. 두 실행의 모델이 같은지 대조하는 데 쓴다."""
    return hashlib.sha256(booster.model_to_string(num_iteration=rounds).encode("utf-8")).hexdigest()


def lgbm_cross_validate(
    data: train.CVData,
    cfg: ExperimentConfig,
    arm: str,
    run_id: str,
    variant: str = evaluate.VARIANT_BASE,
    *,
    drop_features: Sequence[int] = (),
    fold_tables: Mapping[int, StepTable] | None = None,
    feature_source: str = evaluate.FEATURE_SOURCE_NATIVE,
    extra_eval: Mapping[str, pd.DataFrame] | None = None,
    notes: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, float]]:
    """LightGBM의 중첩 교차검증 (PLAN 5.3절, 6장). `(예측 표, 바깥 fold별 선택 기록, 소요 시간)`.

    variant와 feature_source는 예측 표의 표지(`evaluate.prediction_table`)다. 둘 중 하나라도 기본값이 아니면 그 묶음은
    부가 결과로 표시된다(W 비교에서 기준 W가 아닌 묶음, 속도 특징 제외, fold별 전역 상수).
    - drop_features: 통계를 입력에서 뺄 특징 번호들(`stat_columns_without`). 속도 특징 제외 실행은 #7, #8을 준다.
    - fold_tables: 바깥 fold 번호 → 그 fold의 학습, 검증, 시험에 쓸 스텝 표(`foldwise_global_tables`). 주면 fold마다
      그 표에서 창을 만들고, 주지 않으면 `data.tables[arm]` 하나를 모든 fold가 쓴다. 영상 표(분모와 품질 판정)는 어느
      경우에나 `data.tables[arm]`의 것이다.
    - extra_eval: 출처 이름(예측 표의 `feature_source`) → 다른 추출의 스텝 표. 다시 학습한 fold 모델을 시험 fold의 그
      추출 특징으로도 평가해 부가 묶음으로 싣는다(15 fps 모사 평가, PLAN 11장 E3. 학습에는 쓰지 않는다). 원본
      추출에는 창이 있는데 그 추출에는 창이 하나도 없는 시험 fold가 있으면 그 출처의 평가는 통째로 건너뛴다(평가하지
      않은 fold를 오답으로 세지 않는다). notes에 dict를 주면 평가한 출처와 건너뛴 이유를 `extra_evaluation`에 적는다.

    신경망(`train.cross_validate`)과 같은 fold, 같은 안쪽 분할, 같은 선택 규칙을 쓴다.
    - 학습 창은 학습 보폭(`windows.train_stride`)의 창, 검증·시험 창은 평가 보폭(1초)의 창이다. 창마다
      `sample_weights`를 줘서 피험자·영상이 같은 비중을 갖게 한다.
    - 안쪽 분할마다 조기 종료가 고른 부스팅 횟수의 중앙값이 다시 학습할 때의 횟수다.
    - 고른 후보로 학습 fold 전체를 다시 학습해 시험 fold를 한 번 평가한다. 시드는 `lgbm.seed` 하나다(무작위 요소가
      없어 시드를 바꿔도 모델이 같다). 온도는 쓰지 않는다(1.0).
    - 선택 기록의 `refit`에 다시 학습한 모델의 부스팅 횟수, 입력 열 수, 모델 문자열의 sha256을 적는다. 다시 학습은
      시험 fold에 창이 있든 없든 하므로, 기록은 시험 fold의 데이터에 따라 달라지지 않는다(`tests/test_leakage.py`).
    가정: 신경망은 영상의 보폭 1 창을 균등하게 뽑아 보지만 LightGBM에는 에폭이 없으므로 학습 보폭 격자 위의 창만 쓴다.
    가정: 원본 프레임률 추출의 창으로만 학습한다. 신경망은 15 fps 누락 모사 추출과 프레임 누락 변형이 있으면 학습
    표본의 일부를 거기서 뽑지만(증강 (1), (2)) 여기서는 섞지 않는다. 계획서 5.3절의 "같은 창"과 다른 점이고, 비교표와
    D11의 기준 LightGBM은 그 증강 없이 학습한 것이다. 리포트 설명과 명령의 출력에 적는다(`NATIVE_ONLY_NOTE`).
    """
    if arm not in data.tables:
        raise ValueError(f"no step table for arm {arm!r}")
    settings = cfg.lgbm
    window_steps = cfg.windows.window_steps
    table = data.tables[arm]
    videos = evaluate.video_table(table.quality)
    grid = lgbm_candidates(settings)
    columns = stat_columns_without(drop_features)
    all_names = window_stat_names()
    names = [all_names[index] for index in columns.tolist()]
    timings: dict[str, float] = {"window_stats_s": 0.0}

    def stats_of(steps: pd.DataFrame) -> tuple[_Stats, _Stats]:
        started = time.perf_counter()
        train_part = _stats(make_windows(steps, window_steps, cfg.windows.train_stride), columns)
        eval_part = _stats(make_windows(steps, window_steps, cfg.windows.eval_stride), columns)
        timings["window_stats_s"] += time.perf_counter() - started
        return train_part, eval_part

    shared = stats_of(table.steps) if fold_tables is None else None
    splits = list(nested_splits(data.manifest))

    # 다른 추출에서의 평가(부가): 원본에 창이 있는 시험 fold를 다 덮는 출처만 평가한다.
    extra_stats: dict[str, tuple[_Stats, pd.DataFrame]] = {}
    extra_skipped: dict[str, str] = {}
    if extra_eval:
        if shared is None:
            raise ValueError("extra_eval cannot be combined with fold_tables")
        for name, extra_steps in extra_eval.items():
            candidate_stats = _stats(make_windows(extra_steps, window_steps, cfg.windows.eval_stride), columns)
            uncovered = [
                int(split.test_fold)
                for split in splits
                if len(shared[1].of(split.test_subjects).windows) > 0
                and len(candidate_stats.of(split.test_subjects).windows) == 0
            ]
            if uncovered:
                extra_skipped[str(name)] = f"the extraction has no windows for test fold(s) {uncovered}"
            else:
                extra_stats[str(name)] = (candidate_stats, extra_steps)
    if notes is not None:
        notes["extra_evaluation"] = {"evaluated": list(extra_stats), "skipped": extra_skipped}

    started = time.perf_counter()
    stats_before = timings["window_stats_s"]
    predictions: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    for split in splits:
        assert_subject_disjoint(split.train_subjects, split.test_subjects)
        steps = table.steps
        if shared is not None:
            train_stats, eval_stats = shared
        else:
            assert fold_tables is not None
            if int(split.test_fold) not in fold_tables:
                raise ValueError(f"no step table for outer fold {split.test_fold}")
            fold_table = fold_tables[int(split.test_fold)]
            if not train._video_tables_match(videos, evaluate.video_table(fold_table.quality)):
                raise ValueError(f"the step table of outer fold {split.test_fold} lists other videos than arm {arm}")
            steps = fold_table.steps
            train_stats, eval_stats = stats_of(steps)
        scores: list[LgbmScore] = []
        for candidate in grid:
            params = lgbm_params(settings, candidate.num_leaves, candidate.min_data_in_leaf)
            fold_scores: list[float] = []
            fold_rounds: list[int] = []
            for inner in split.inner:
                assert_subject_disjoint(inner.train_subjects, inner.val_subjects, split.test_subjects)
                fit_part = train_stats.of(inner.train_subjects)
                val_part = eval_stats.of(inner.val_subjects)
                if len(fit_part.windows) == 0 or len(val_part.windows) == 0:
                    raise ValueError(f"inner split with validation fold {inner.val_fold} has no windows")
                _, rounds, score = fit_lgbm(
                    fit_part.stats,
                    fit_part.windows.y,
                    sample_weights(fit_part.windows.meta),
                    params,
                    settings.max_rounds,
                    val=(val_part.stats, val_part.windows.meta),
                    early_stopping_rounds=settings.early_stopping_rounds,
                    feature_names=names,
                )
                fold_scores.append(score)
                fold_rounds.append(rounds)
            scores.append(
                LgbmScore(
                    candidate=candidate,
                    val_folds=tuple(int(inner.val_fold) for inner in split.inner),
                    fold_macro_f1=tuple(fold_scores),
                    mean_macro_f1=float(np.mean(fold_scores)),
                    std_macro_f1=float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,
                    fold_rounds=tuple(fold_rounds),
                    rounds=train.median_half_up(fold_rounds),
                )
            )
        chosen = choose_lgbm(scores)

        fit_part = train_stats.of(split.train_subjects)
        test_part = eval_stats.of(split.test_subjects)
        assert_subject_disjoint(train.window_subjects(fit_part.windows), train.window_subjects(test_part.windows))
        booster, _, _ = fit_lgbm(
            fit_part.stats,
            fit_part.windows.y,
            sample_weights(fit_part.windows.meta),
            lgbm_params(settings, chosen.candidate.num_leaves, chosen.candidate.min_data_in_leaf),
            chosen.rounds,
            feature_names=names,
        )
        records.append(
            {
                "fold": int(split.test_fold),
                "test_subjects": list(split.test_subjects),
                "train_subjects": list(split.train_subjects),
                "inner_val_folds": [int(inner.val_fold) for inner in split.inner],
                "candidates": [_score_record(score) for score in scores],
                "chosen": {evaluate.FAMILY_LGBM: chosen.candidate.name},
                "chosen_mean_macro_f1": chosen.mean_macro_f1,
                "refit": {
                    "rounds": int(chosen.rounds),
                    "n_columns": int(len(columns)),
                    "model_sha256": model_sha256(booster, chosen.rounds),
                },
            }
        )
        if len(test_part.windows) == 0:
            continue  # 창이 없는 시험 fold: 영상 표의 영상이 모두 오답으로 남는다(D7)
        probs = np.asarray(booster.predict(test_part.stats), dtype=np.float64).reshape(len(test_part.windows), -1)
        predictions.append(
            evaluate.prediction_table(
                test_part.windows,
                probs / probs.sum(axis=1, keepdims=True),
                steps,
                videos,
                run_id=run_id,
                family=evaluate.FAMILY_LGBM,
                arm=arm,
                seed=settings.seed,
                feature_source=feature_source,
                variant=variant,
            )
        )
        for name, (source_stats, source_steps) in extra_stats.items():
            other_part = source_stats.of(split.test_subjects)
            if len(other_part.windows) == 0:
                continue
            assert_subject_disjoint(train.window_subjects(fit_part.windows), train.window_subjects(other_part.windows))
            other = np.asarray(booster.predict(other_part.stats), dtype=np.float64).reshape(len(other_part.windows), -1)
            predictions.append(
                evaluate.prediction_table(
                    other_part.windows,
                    other / other.sum(axis=1, keepdims=True),
                    source_steps,
                    videos,
                    run_id=run_id,
                    family=evaluate.FAMILY_LGBM,
                    arm=arm,
                    seed=settings.seed,
                    feature_source=name,
                    variant=variant,
                )
            )
    timings["cross_validation_s"] = time.perf_counter() - started - (timings["window_stats_s"] - stats_before)
    if not predictions:
        raise ValueError("cross-validation produced no predictions: no test fold has windows")
    return evaluate.validate_predictions(pd.concat(predictions, ignore_index=True)), records, timings


def foldwise_global_tables(
    data: train.CVData, cfg: ExperimentConfig, *, rebuild: bool = False
) -> tuple[dict[int, StepTable], dict[int, dict[str, Any]]]:
    """바깥 fold마다 학습 피험자만으로 다시 계산한 전역 상수로 만든 전역 조건의 스텝 표 (PLAN 4.3절의 민감도 점검).

    `(바깥 fold 번호 → 스텝 표, 바깥 fold 번호 → 그 fold의 상수 기록)`을 돌려준다.
    feature_spec의 전역 상수는 E0에서 라벨 없이 전체 영상(시험 피험자의 영상 포함)으로 한 번 정한다. 이 함수는 바깥
    fold k마다 그 fold의 학습 피험자(시험 fold를 뺀 나머지)의 세션만으로 같은 통계(세션별 기준선 E, B, jaw0의 중앙값,
    `signal_check.global_constants`)를 다시 계산하고, 그 상수를 전역 조건의 기준선으로 넣어 모든 세션의 스텝 표를
    다시 만든다(`dataset.load_step_tables`의 global_override. 상수마다 따로 캐시된다). fold k의 학습, 검증, 시험이 모두
    그 표를 쓴다. 시험 fold의 영상은 그 fold의 상수에 들어가지 않는다.
    상수 기록: `{"e", "b", "jaw0", "n_sessions", "train_subjects"}`(n_sessions는 중앙값에 들어간 세션 수).
    읽는 품질 표는 `data.tables["global"].quality`다. 세션 자신의 기준선 열(`base_*`, `baseline_status`)은 정규화
    조건과 무관하게 같은 값이라(`dataset.StepTable`) 자기 세션 기준선 조건의 품질 표를 읽는 것과 같다.
    전역 조건의 표가 feature_spec의 상수로 만든 것이 아니면(closure W를 바꾼 실행) ValueError다.
    """
    if ARM_GLOBAL not in data.tables:
        raise ValueError(f"the fold-wise global constants need the step table of arm {ARM_GLOBAL!r}")
    table = data.tables[ARM_GLOBAL]
    spec = data.feature_spec.pipeline
    if spec_pipeline_hash(spec) != table.pipeline_hash:
        raise ValueError("the global-arm step table was not built with the pipeline constants of the feature spec")
    tables: dict[int, StepTable] = {}
    constants: dict[int, dict[str, Any]] = {}
    for split in nested_splits(data.manifest):
        assert_subject_disjoint(split.train_subjects, split.test_subjects)
        override = signal_check.global_constants(table.quality, spec, subjects=split.train_subjects)
        medians = signal_check.constant_medians(table.quality, split.train_subjects)
        tables[int(split.test_fold)] = load_step_tables(
            data.manifest,
            data.raw_dir,
            spec,
            (ARM_GLOBAL,),
            global_override=override,
            gate=cfg.evaluation.quality_gate,
            workers=cfg.extraction.workers,
            rebuild=rebuild,
        )[ARM_GLOBAL]
        constants[int(split.test_fold)] = {
            "e": float(override.e_l),
            "b": float(override.b_l),
            "jaw0": float(override.jaw0),
            "n_sessions": int(medians.n_sessions),
            "train_subjects": [int(subject) for subject in split.train_subjects],
        }
    return tables, constants


def e1_gate(selection: Sequence[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """E1 통과 기준 (PLAN 11장): 고른 후보의 내부 검증 macro-F1 평균(바깥 fold들의 평균)이 threshold 이상인지.

    가정: "내부 검증 macro-F1 평균"은 바깥 fold마다 고른 후보의 안쪽 fold 평균을 다시 바깥 fold에 대해 평균한 값이다.
    시험 fold의 값은 쓰지 않는다.
    """
    values = [float(record["chosen_mean_macro_f1"]) for record in selection]
    mean = float(np.mean(values)) if values else math.nan
    return {"inner_macro_f1_mean": mean, "threshold": float(threshold), "passed": bool(mean >= threshold)}


def lgbm_subdir(closure_w: float | None) -> str:
    """리포트 하위 폴더 이름. closure W를 바꾼 실행은 기준 실행(`lgbm/`)과 섞이지 않게 `lgbm_w{W}/`에 쓴다."""
    return train.LGBM_SUBDIR if closure_w is None else f"{train.LGBM_SUBDIR}_w{closure_w:g}"


def run_lgbm_cv(
    cfg: ExperimentConfig,
    *,
    arms: Sequence[str] | None = None,
    fast: bool = False,
    run_name: str = train.DEFAULT_CV_RUN_NAME,
    config_path: Path = Path("config/default.yaml"),
    command: str = "",
    closure_w: float | None = None,
    overwrite: bool = False,
    rebuild: bool = False,
    no_velocity: bool = False,
    foldwise_global: bool = False,
) -> train.CVResult:
    """LightGBM 중첩 교차검증을 돌리고 `{reports_dir}/{run_name}/lgbm/`에 리포트를 쓴다 (PLAN 5.3절, 11장 E1).

    - arms: 정규화 조건들. None이면 `lgbm.reference_arm`(비교표와 D11의 기준) 하나다. 조건마다 묶음 하나가 된다
      (run_id `train-{조건}`).
    - closure_w: closure의 W를 이 값으로 바꾼 스텝 표로 돈다(E1의 W 비교, 보고용). run_id에 `-w{W}`가 붙고 폴더는
      `lgbm_w{W}/`다. None이면 feature_spec의 값 그대로이고, 이 실행이 비교표와 D11의 기준이다(PLAN 5.3절: 기준은
      alert 영상 기준선 조건, W = `lgbm.reference_w`). 그래서 None일 때 feature_spec의 closure.w가
      `lgbm.reference_w`와 다르면 ValueError다.
      가정: 기준 실행은 "closure W를 바꾸지 않은 실행"이다. E0에서 feature_spec의 W를 바꾸게 되면 `lgbm.reference_w`를
      어떻게 둘지(계획서의 0.5를 지킬지, 확정한 W를 따를지)는 사람이 정해 설정에 적어야 하고, 그 전에는 시작하지 않는다.
    - fast: 축소 설정으로 돈다(연기 시험용).
    - overwrite: 같은 폴더에 다른 실행(다른 조건 목록, 축소 설정 여부, closure W, 데이터 종류)의 리포트가 있어도
      덮어쓴다. False면 학습 전에 ValueError로 멈춘다(`train.check_run_dir`). rebuild: 스텝 표 캐시를 다시 만든다.
    `run.json`에 E1 통과 기준(`e1_gate`, 조건별 값), 실제로 쓴 closure W(`effective_closure_w`), 조건별 스텝 표의
    내용 지문(`step_fingerprints`), 기준선·창 요약(`baselines`), 추출 결과의 회전 점검 기록(`rotation_check`,
    `train.rotation_record`)을 함께 적는다. 신경망 리포트(`drowsy.train`)는 이 기록으로 기준 LightGBM이 같은 데이터, 같은
    절차에서 나온 것인지 대조한다. 회전이 확인되지 않은 영상이 있으면 그 수가 리포트 설명에 실린다(`train.rotation_note`).
    리포트의 8절에는 기준선이 연장·대체된 영상과 조건별·클래스별 창 수가 실린다(PLAN 4.4절).

    부가 실행 두 가지(PLAN 11장 E3, 4.3절). 둘을 함께 쓰거나 closure_w와 함께 쓸 수 없다(ValueError). 어느 쪽도
    `lgbm/`(비교표와 D11의 기준)을 고치지 않고, E1 통과 기준의 판단에도 쓰이지 않는다.
    - no_velocity: 속도 특징 #7, #8의 통계 12개를 입력에서 뺀다(102개 → 90개). run_id에 `-no_velocity`가 붙고 묶음은
      모두 부가 묶음(variant `no_velocity`)이며 폴더는 `lgbm_no_velocity/`다. 같은 실행 폴더의 `lgbm/`에 같은 데이터,
      같은 축소 설정의 주 실행이 있으면(`train.main_run_predictions`) 그 묶음을 함께 실어 조건마다 짝지은 비교를 낸다.
      `lgbm/`이 있는데 다른 데이터나 절차의 것이면 학습 전에 ValueError다. 없으면 짝지은 비교 없이 돈다.
    - foldwise_global: 전역 조건의 민감도 점검. arms는 `("global",)`이어야 한다. 주 결과 묶음(feature_spec의 전역
      상수, 전체 영상으로 정한 값)과 나란히, 바깥 fold마다 학습 피험자만으로 다시 계산한 상수로 특징을 다시 만들어
      돌린 부가 묶음(feature_source `fold_global`, 같은 run_id)을 싣고 둘의 짝지은 비교를 낸다. 폴더는
      `lgbm_fold_global/`다. fold별 상수는 `run.json`의 `fold_global_constants`와 선택 기록에 적는다
      (`foldwise_global_tables`).
    """
    if fast:
        cfg = apply_fast_profile(cfg)
    arms = (cfg.lgbm.reference_arm,) if arms is None else tuple(arms)
    if len(arms) == 0:
        raise ValueError("need at least one arm")
    if no_velocity and foldwise_global:
        raise ValueError("--no-velocity and --foldwise-global are separate supplementary runs; pass one of them")
    if (no_velocity or foldwise_global) and closure_w is not None:
        raise ValueError("a supplementary run (--no-velocity, --foldwise-global) cannot be combined with --closure-w")
    if foldwise_global and arms != (ARM_GLOBAL,):
        raise ValueError(f"--foldwise-global is the sensitivity check of the global arm: pass --arms {ARM_GLOBAL}")
    if no_velocity:
        subdir = NO_VELOCITY_SUBDIR
    elif foldwise_global:
        subdir = FOLD_GLOBAL_SUBDIR
    else:
        subdir = lgbm_subdir(closure_w)
    main_dir = cfg.paths.reports_dir / run_name / train.LGBM_SUBDIR
    run_dir = cfg.paths.reports_dir / run_name / subdir
    total_started = time.perf_counter()

    spec_w = float(load_feature_spec(cfg.paths.feature_spec).pipeline.closure.w)
    pipeline: PipelineSpec | None = None
    suffix = ""
    if closure_w is None:
        if spec_w != cfg.lgbm.reference_w:
            raise ValueError(
                f"the feature spec has closure.w = {spec_w:g} but lgbm.reference_w is {cfg.lgbm.reference_w:g}. The "
                "run without --closure-w is the reference of the comparison table and D11 (PLAN 5.3), so the two "
                "must agree: set lgbm.reference_w to the W of the feature spec, or pass --closure-w for a "
                "reporting-only run"
            )
    else:
        pipeline = apply_overrides(load_feature_spec(cfg.paths.feature_spec).pipeline, {"closure.w": float(closure_w)})
        suffix = f"-w{closure_w:g}"
    if no_velocity:
        suffix = f"-{evaluate.VARIANT_NO_VELOCITY}"
    effective_w = spec_w if closure_w is None else float(closure_w)
    # 15 fps 모사 평가(부가)는 선택 사항이 없는 실행에서만 한다. 학습에는 그 추출을 쓰지 않으므로 덜 끝난 추출이어도
    # 시작은 한다(없는 영상은 그 부가 묶음에서 오답으로 남고, 덮은 세션 수가 설명에 실린다).
    fps_eval_arms = () if (no_velocity or foldwise_global) else arms
    data = train.load_cv_data(
        cfg, arms, pipeline=pipeline, fps_eval_arms=fps_eval_arms, allow_partial_fps_sim=True, rebuild=rebuild
    )
    identity = {"arms": list(arms), "fast": fast, "closure_w": closure_w, "synthetic": data.synthetic}
    train.check_run_dir(run_dir, identity, overwrite)
    timings: dict[str, float] = {"load_data_s": time.perf_counter() - total_started}
    first = data.tables[arms[0]]
    fingerprints = {arm: train.step_table_fingerprint(data.tables[arm].steps) for arm in arms}
    main: pd.DataFrame | None = None
    if no_velocity:
        main = train.main_run_predictions(
            main_dir,
            expected={"fast": fast, "closure_w": None},
            synthetic=data.synthetic,
            pipeline_hash=first.pipeline_hash,
            fingerprints=fingerprints,
            run_ids=[train.run_id_for(arm, fast) for arm in arms],
        )

    variant = evaluate.VARIANT_NO_VELOCITY if no_velocity else evaluate.VARIANT_BASE
    drop_features = train.VELOCITY_FEATURES if no_velocity else ()
    predictions: list[pd.DataFrame] = []
    selection: list[dict[str, Any]] = []
    gates: dict[str, dict[str, Any]] = {}
    fps_notes: dict[str, dict[str, Any]] = {}
    for arm in arms:
        run_id = train.run_id_for(arm, fast, suffix)
        arm_notes: dict[str, Any] = {}
        extra_eval = {train.SOURCE_FPS_SIM: data.extra_tables[arm].steps} if arm in data.extra_tables else None
        preds, records, arm_timings = lgbm_cross_validate(
            data, cfg, arm, run_id, variant, drop_features=drop_features, extra_eval=extra_eval, notes=arm_notes
        )
        fps_notes[arm] = arm_notes["extra_evaluation"]
        predictions.append(preds)
        selection.extend({"arm": arm, "run_id": run_id, **record} for record in records)
        gates[arm] = e1_gate(records, cfg.evaluation.lgbm_min_macro_f1)
        for key, value in arm_timings.items():
            timings[f"{arm}_{key}"] = value
    fold_constants: dict[int, dict[str, Any]] = {}
    if foldwise_global:
        started = time.perf_counter()
        fold_tables, fold_constants = foldwise_global_tables(data, cfg, rebuild=rebuild)
        timings["fold_global_tables_s"] = time.perf_counter() - started
        run_id = train.run_id_for(ARM_GLOBAL, fast, suffix)
        source = evaluate.FEATURE_SOURCE_FOLD_GLOBAL
        preds, records, fold_timings = lgbm_cross_validate(
            data, cfg, ARM_GLOBAL, run_id, fold_tables=fold_tables, feature_source=source
        )
        predictions.append(preds)
        selection.extend(
            {
                "arm": ARM_GLOBAL,
                "run_id": run_id,
                "feature_source": source,
                "global_constants": fold_constants[int(record["fold"])],
                **record,
            }
            for record in records
        )
        for key, value in fold_timings.items():
            timings[f"{source}_{key}"] = value
        for fold, fold_table in fold_tables.items():
            fingerprints[f"{ARM_GLOBAL} [fold {fold} constants]"] = train.step_table_fingerprint(fold_table.steps)
    run_predictions = pd.concat(predictions, ignore_index=True)

    description = (
        f"{data.description}. LightGBM baseline, nested cross-validation (PLAN 5.3, 6): arms {', '.join(arms)}, "
        + (
            f"closure W {closure_w:g} (override: a reporting-only comparison, not the D11 reference)"
            if closure_w is not None
            else f"closure W {effective_w:g} (feature spec)"
        )
        + f". {NATIVE_ONLY_NOTE}. {train.CONSTANTS_NOTE}"
    )
    all_predictions = run_predictions
    paired = list(evaluate.supplementary_pairs(run_predictions))
    evaluated_arms = [arm for arm in arms if fps_notes[arm]["evaluated"]]
    if evaluated_arms:
        description += (
            ". Supplementary groups (section 9 only, never used for the comparison table, the E1 criterion or D11): "
            f"the fold models evaluated on the 15 fps simulated extraction of the test fold (feature_source "
            f"{train.SOURCE_FPS_SIM}, arms {', '.join(evaluated_arms)}); training does not use that extraction"
        )
    for arm in arms:
        for reason in fps_notes[arm]["skipped"].values():
            description += f". 15 fps evaluation of arm {arm} skipped ({reason})"
    if no_velocity:
        names = ", ".join(f"#{index} {FEATURE_NAMES[index]}" for index in train.VELOCITY_FEATURES)
        description += (
            f". SUPPLEMENTARY RUN (PLAN 11 E3): without the velocity features {names}; their "
            f"{len(STAT_NAMES) * len(train.VELOCITY_FEATURES)} window statistics are dropped from the "
            f"{len(STAT_NAMES) * len(FEATURE_NAMES)} inputs. Every group of this run is supplementary (section 9) and "
            "is never used for the comparison table, the E1 criterion or D11; "
            + (
                f"the main groups listed here are copied from the main run in {main_dir} for the paired comparison"
                if main is not None
                else f"no main run was found in {main_dir}, so there is no paired comparison"
            )
        )
        if main is not None:
            all_predictions = pd.concat([main, run_predictions], ignore_index=True)
            run_ids = {train.run_id_for(arm, fast, suffix): train.run_id_for(arm, fast) for arm in arms}
            paired = train.supplementary_run_pairs(run_predictions, main, run_ids)
    if foldwise_global:
        description += (
            ". SENSITIVITY CHECK of PLAN 4.3 (supplementary): the group with feature_source "
            f"{evaluate.FEATURE_SOURCE_FOLD_GLOBAL} uses global constants recomputed for every outer fold from the "
            "training subjects only (medians over their sessions' personal baselines); the features of that fold's "
            "training, validation and test sessions are rebuilt with them. The main group uses the constants of the "
            "feature spec, which are fixed once on all videos. This folder is not the D11 reference"
        )
    if fast:
        description += f". {train.FAST_NOTE}"
    started = time.perf_counter()
    report = evaluate.write_reports(
        run_dir,
        all_predictions,
        evaluate.video_table(first.quality),
        train.fold_subjects(data.manifest),
        train.rules_for(cfg, data.feature_spec),
        cfg.evaluation,
        synthetic=data.synthetic,
        description=description,
        command=command,
        exclusions=evaluate.exclusion_list(first.quality),
        config_path=config_path,
        paired=paired,
        window_counts=train.report_window_counts(data, cfg),
        nominal_baseline_ms=data.feature_spec.pipeline.baseline.duration_ms,
    )
    timings["report_s"] = time.perf_counter() - started
    timings["total_s"] = time.perf_counter() - total_started
    extra: dict[str, Any] = {
        "arms": list(arms),
        "fast": fast,
        "closure_w": closure_w,
        "effective_closure_w": effective_w,
        "variant": variant,
        "foldwise_global": foldwise_global,
        "fps15_evaluation": fps_notes,
        "e1_gate": gates,
        "baselines": train.baseline_summary(data, cfg),
        "rotation_check": train.rotation_record(data.rotation),
    }
    if no_velocity:
        extra["paired_with_main_run"] = main is not None
    if foldwise_global:
        extra["fold_global_constants"] = {str(fold): record for fold, record in fold_constants.items()}
    train.write_run_files(
        run_dir,
        run_predictions,
        selection,
        timings,
        synthetic=data.synthetic,
        description=description,
        command=command,
        pipeline_hash=first.pipeline_hash,
        step_fingerprints=fingerprints,
        extra=extra,
    )
    return train.CVResult(
        run_dir=run_dir, predictions=run_predictions, selection=selection, report=report, timings=timings
    )


def closure_w_variant(closure_w: float) -> str:
    """W 비교에서 closure W를 바꾼 묶음의 `variant` 표지: `closure_w{W}`(예: `closure_w1`, `closure_w0`)."""
    return f"closure_w{closure_w:g}"


def _fmt(value: float, digits: int = 3) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"


def _json_value(value: Any) -> Any:
    """JSON에 쓸 값: 유한하지 않은 실수(값이 없는 칸)는 None으로 바꾼다."""
    return None if isinstance(value, float) and not math.isfinite(value) else value


def run_lgbm_w_grid(
    cfg: ExperimentConfig,
    *,
    fast: bool = False,
    run_name: str = train.DEFAULT_CV_RUN_NAME,
    config_path: Path = Path("config/default.yaml"),
    command: str = "",
    overwrite: bool = False,
    rebuild: bool = False,
) -> train.CVResult:
    """E1의 closure W 비교: `lgbm.w_grid`의 W마다 기준 조건의 교차검증을 돌려 한 리포트로 합친다 (PLAN 11장 E1, 보고용).

    W마다 closure의 W만 바꾼 스텝 표를 만들고(`dataset.load_step_tables`, W마다 다른 pipeline_hash로 캐시된다)
    `lgbm.reference_arm`에서 `lgbm_cross_validate`를 돈다. 결과를 `{reports_dir}/{run_name}/lgbm_w_grid/`에 쓴다.
    - summary.md 등(`evaluate.write_reports`): W = `lgbm.reference_w`의 묶음이 주 결과, 나머지 W의 묶음은 부가 결과
      (9절)이고 variant가 `closure_w{W}`다. 부가 묶음마다 기준 W 묶음과의 짝지은 비교가 실린다.
    - w_grid.md, w_grid.csv: W마다 한 행인 합친 표. 열은 closure_w, role(reference 또는 supplementary), variant,
      pipeline_hash, inner_macro_f1_mean(E1 기준의 값), 그리고 `W_GRID_METRICS`의 지표마다 fold 평균, fold 표준편차,
      fold를 합친 값과 그 부트스트랩 구간, 기준 W와의 차이와 그 짝지은 부트스트랩 구간, differs다. w_grid.md의 머리말에는
      교차검증 리포트와 같은 자리에 D 번호의 뜻을 알리는 한 줄(`evaluate.PLAN_DECISION_NOTE`)이 있다.
    - predictions.parquet, selection.json, timings.json, run.json(`train.write_run_files`). run.json에는 추출 결과의 회전
      점검 기록(`rotation_check`)도 적는다.
    비교표와 D11의 기준은 이 폴더가 아니라 `lgbm/`의 실행(`run_lgbm_cv`, closure_w 없이)이다. 이 명령은 `lgbm/`을
    읽지도 쓰지도 않는다. 기준 W의 묶음은 같은 절차를 한 번 더 돈 것이라 같은 설정이면 `lgbm/`의 기준 묶음과 값이 같다.
    가정: 계획서는 "W 세 가지 비교는 보고용"이라고만 했다. 기준 조건 하나에서만 돌리고, 기준 W가 아닌 묶음을 부가로
    표시하며, E1 통과 기준(종료 코드 4)은 기준 실행에만 적용한다. 이 명령은 기준 미달이어도 멈추지 않는다.
    가정: 영상 표(분모와 품질 판정)는 기준 W의 스텝 표에서 만든다. closure W는 기준선 완성 시각과 얼굴 검출 비율을
    바꾸지 않으므로 W마다 같다(다르면 예측 표의 검사가 gate_pass 불일치로 멈춘다).
    """
    if fast:
        cfg = apply_fast_profile(cfg)
    arm = cfg.lgbm.reference_arm
    grid = tuple(float(value) for value in cfg.lgbm.w_grid)
    reference_w = float(cfg.lgbm.reference_w)
    if reference_w not in grid:
        raise ValueError(f"lgbm.w_grid {list(grid)} must contain lgbm.reference_w {reference_w:g}")
    run_dir = cfg.paths.reports_dir / run_name / W_GRID_SUBDIR
    run_id = train.run_id_for(arm, fast)
    base_pipeline = load_feature_spec(cfg.paths.feature_spec).pipeline
    total_started = time.perf_counter()

    timings: dict[str, float] = {}
    predictions: list[pd.DataFrame] = []
    selection: list[dict[str, Any]] = []
    gates: dict[float, dict[str, Any]] = {}
    tables: dict[float, StepTable] = {}
    reference_data: train.CVData | None = None
    for w in grid:
        started = time.perf_counter()
        data = train.load_cv_data(
            cfg, (arm,), pipeline=apply_overrides(base_pipeline, {"closure.w": w}), rebuild=rebuild
        )
        timings[f"w{w:g}_load_data_s"] = time.perf_counter() - started
        if w == grid[0]:  # 첫 스텝 표를 읽은 뒤, 학습 전에 한 번 검사한다
            identity = {"arm": arm, "fast": fast, "w_grid": list(grid), "synthetic": data.synthetic}
            train.check_run_dir(run_dir, identity, overwrite)
        variant = evaluate.VARIANT_BASE if w == reference_w else closure_w_variant(w)
        preds, records, w_timings = lgbm_cross_validate(data, cfg, arm, run_id, variant=variant)
        predictions.append(preds)
        selection.extend({"arm": arm, "run_id": run_id, "closure_w": w, "variant": variant, **r} for r in records)
        gates[w] = e1_gate(records, cfg.evaluation.lgbm_min_macro_f1)
        tables[w] = data.tables[arm]
        for key, value in w_timings.items():
            timings[f"w{w:g}_{key}"] = value
        if w == reference_w:
            reference_data = data
    assert reference_data is not None
    all_predictions = evaluate.validate_predictions(pd.concat(predictions, ignore_index=True))

    others = [w for w in grid if w != reference_w]
    reference_key = evaluate.group_key(run_id, evaluate.FAMILY_LGBM, arm)
    pairs = [
        (
            evaluate.group_key(
                run_id, evaluate.FAMILY_LGBM, arm, evaluate.FEATURE_SOURCE_NATIVE, closure_w_variant(w)
            ),
            reference_key,
        )
        for w in others
    ]
    quality = tables[reference_w].quality
    strides = (cfg.windows.train_stride, cfg.windows.eval_stride)
    description = (
        f"{reference_data.description}. LightGBM closure-W comparison (PLAN 11 E1, supplementary, for reporting "
        f"only): arm {arm}, W grid {', '.join(f'{w:g}' for w in grid)}; the group at the reference W "
        f"{reference_w:g} is the main group, the other W are supplementary groups (variant closure_w<W>). The "
        f"reference of the comparison table and D11 is the run in {train.LGBM_SUBDIR}/, not this folder. "
        f"{NATIVE_ONLY_NOTE}. {train.CONSTANTS_NOTE}" + (f". {train.FAST_NOTE}" if fast else "")
    )
    started = time.perf_counter()
    report = evaluate.write_reports(
        run_dir,
        all_predictions,
        evaluate.video_table(quality),
        train.fold_subjects(reference_data.manifest),
        train.rules_for(cfg, reference_data.feature_spec),
        cfg.evaluation,
        synthetic=reference_data.synthetic,
        description=description,
        command=command,
        exclusions=evaluate.exclusion_list(quality),
        config_path=config_path,
        paired=pairs,
        window_counts=evaluate.window_counts(
            {f"{arm} [closure W {w:g}]": tables[w].steps for w in grid}, cfg.windows.window_steps, strides
        ),
        nominal_baseline_ms=reference_data.feature_spec.pipeline.baseline.duration_ms,
    )
    timings["report_s"] = time.perf_counter() - started

    # 합친 표: W마다 한 행. 짝지은 비교는 요청한 쌍의 순서(others)대로, 쌍마다 PAIRED_METRICS 순서로 실려 있다.
    by_variant = {result.variant: result for result in (*report.results, *report.supplementary)}
    paired = report.supplementary_paired
    if len(paired) != len(others) * len(evaluate.PAIRED_METRICS):
        raise RuntimeError("the paired comparison table does not hold one block per supplementary W")
    rows: list[dict[str, Any]] = []
    for w in grid:
        variant = evaluate.VARIANT_BASE if w == reference_w else closure_w_variant(w)
        result = by_variant[variant]
        row: dict[str, Any] = {
            "closure_w": w,
            "role": "reference" if w == reference_w else "supplementary",
            "variant": variant,
            "pipeline_hash": tables[w].pipeline_hash,
            "inner_macro_f1_mean": gates[w]["inner_macro_f1_mean"],
        }
        block = None
        if w != reference_w:
            position = others.index(w) * len(evaluate.PAIRED_METRICS)
            block = paired.iloc[position : position + len(evaluate.PAIRED_METRICS)].set_index("metric")
        for metric in W_GRID_METRICS:
            for column in ("fold_mean", "fold_std", "pooled", "ci_low", "ci_high"):
                row[f"{metric}_{column}"] = result.value(metric, column)
            row[f"{metric}_minus_reference"] = math.nan if block is None else float(block.loc[metric, "difference"])
            row[f"{metric}_minus_reference_ci_low"] = math.nan if block is None else float(block.loc[metric, "ci_low"])
            row[f"{metric}_minus_reference_ci_high"] = (
                math.nan if block is None else float(block.loc[metric, "ci_high"])
            )
            row[f"{metric}_differs"] = False if block is None else bool(block.loc[metric, "differs"])
        rows.append(row)
    table = pd.DataFrame(rows)
    note = train.data_note(reference_data.synthetic)
    with (run_dir / W_GRID_CSV_FILE).open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# {note}\n")
        table.to_csv(handle, index=False, lineterminator="\n")
    lines = [
        f"# {note}",
        "",
        description,
        "",
        (
            "합성 데이터로 만든 표다. 아래 숫자는 파이프라인 점검용이며 성능 수치가 아니다."
            if reference_data.synthetic
            else "아래 숫자는 교차검증 추정치다. W 비교는 보고용 부가 분석이고 모델 선택과 D11 판정에 쓰지 않는다."
        ),
        "",
        evaluate.PLAN_DECISION_NOTE,  # 설명 문단의 D11이 무엇의 번호인지 알린다(교차검증 리포트와 같은 자리)
        "",
        f"- 생성 명령: `{command}`",
        f"- 정규화 조건 `{arm}`, 기준 W {reference_w:g}. 지표는 fold 평균 ± 표준편차, 대괄호는 fold를 합친 값의 피험자 "
        "부트스트랩 95% 구간이다",
        "- \"기준 W와의 차이\"는 fold를 합친 값의 차이(이 W − 기준 W)와 짝지은 피험자 부트스트랩 95% 구간이다. 구간이 "
        "0을 벗어날 때만 차이가 있다고 쓴다",
        f"- fold별 표와 혼동 행렬, 오경보 표는 같은 폴더의 `{evaluate.SUMMARY_FILE}`에 있다. 값 전체는 `{W_GRID_CSV_FILE}`",
        "",
        "| closure W | 구분 | 내부 검증 macro-F1 평균 | "
        + " | ".join(f"`{metric}`" for metric in W_GRID_METRICS)
        + " | "
        + " | ".join(f"`{metric}` 기준 W와의 차이" for metric in W_GRID_METRICS)
        + " |",
        "| --- | --- | --- | " + " | ".join("---" for _ in range(2 * len(W_GRID_METRICS))) + " |",
    ]
    for row in rows:
        cells = [f"{row['closure_w']:g}", "기준" if row["role"] == "reference" else "부가"]
        cells.append(_fmt(row["inner_macro_f1_mean"]))
        for metric in W_GRID_METRICS:
            cells.append(
                f"{_fmt(row[f'{metric}_fold_mean'])} ± {_fmt(row[f'{metric}_fold_std'])} "
                f"[{_fmt(row[f'{metric}_ci_low'])}, {_fmt(row[f'{metric}_ci_high'])}]"
            )
        for metric in W_GRID_METRICS:
            if row["role"] == "reference":
                cells.append("-")
            else:
                cells.append(
                    f"{_fmt(row[f'{metric}_minus_reference'])} [{_fmt(row[f'{metric}_minus_reference_ci_low'])}, "
                    f"{_fmt(row[f'{metric}_minus_reference_ci_high'])}]"
                    + (" 차이 있음" if row[f"{metric}_differs"] else "")
                )
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    (run_dir / W_GRID_MD_FILE).write_text("\n".join(lines), encoding="utf-8")

    timings["total_s"] = time.perf_counter() - total_started
    train.write_run_files(
        run_dir,
        all_predictions,
        selection,
        timings,
        synthetic=reference_data.synthetic,
        description=description,
        command=command,
        pipeline_hash=tables[reference_w].pipeline_hash,
        step_fingerprints={
            f"{arm} [closure W {w:g}]": train.step_table_fingerprint(tables[w].steps) for w in grid
        },
        extra={
            "arm": arm,
            "fast": fast,
            "w_grid": list(grid),
            "reference_w": reference_w,
            "pipeline_hashes": {f"{w:g}": tables[w].pipeline_hash for w in grid},
            "e1_gate": {f"{w:g}": gates[w] for w in grid},
            "rotation_check": train.rotation_record(reference_data.rotation),
            "table": [{key: _json_value(value) for key, value in row.items()} for row in rows],
        },
    )
    return train.CVResult(
        run_dir=run_dir, predictions=all_predictions, selection=selection, report=report, timings=timings
    )


# ---------------------------------------------------------------------------
# 교란 점검 (PLAN 6장)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ProbeSession:
    row: int  # 품질 표에서의 행 위치
    class_index: int
    part: int
    raw_path: str
    baseline_used: str
    t_b_ms: int


@dataclass(frozen=True, slots=True)
class _ProbeJob:
    sessions: tuple[_ProbeSession, ...]  # 피험자 한 명의 세션들
    spec: PipelineSpec
    rules: RulesConfig


def _median(values: Sequence[float]) -> float:
    return float(np.median(values)) if len(values) > 0 else math.nan


def _state_medians(
    frames: Sequence[RawFrame],
    spec: PipelineSpec,
    rules: RulesConfig,
    baseline: Baseline | None,
    feature_start_ms: int | None,
) -> tuple[dict[str, float], Baseline | None]:
    """세션을 `DriverPipeline`에 넣어 특징 누적 구간의 프레임 상태에서 중앙값을 구한다. `(값, 파이프라인의 기준선)`.

    baseline이 None이면 앱처럼 세션 앞부분에서 보정하고 보정이 끝난 뒤의 프레임을, 주면 세션 시간 feature_start_ms
    이후의 프레임을 본다(스텝 특징이 만들어진 구간과 같다).
    - open_closure_median: closure가 유효하고 특징용 진입 임계값(0.5) 미만인 프레임의 closure 중앙값.
    - yaw_rel_median, pitch_rel_median: 얼굴이 있는 프레임의 기준선 대비 yaw, pitch 중앙값(도).
    closure와 각도는 파이프라인이 낸 `FrameState`의 값이다. 여기서 다시 계산하지 않는다.
    가정: "뜬 눈 closure"는 특징용 감김 구간에 들지 않은(진입 임계값 미만) 프레임의 closure다.
    """
    pipe = DriverPipeline(spec, rules, baseline, feature_start_ms)
    enter_thr = spec.run_detector.enter_thr
    closures: list[float] = []
    yaws: list[float] = []
    pitches: list[float] = []
    origin: int | None = None
    for frame in frames:
        out = pipe.update(frame)
        if out is None:
            continue
        state = out.state
        if origin is None:
            origin = state.t_ms
        if state.provisional or (feature_start_ms is not None and state.t_ms - origin < feature_start_ms):
            continue
        if state.closure_valid and state.closure < enter_thr:
            closures.append(state.closure)
        if state.face:
            yaws.append(state.yaw_rel)
            pitches.append(state.pitch_rel)
    values = {
        "open_closure_median": _median(closures),
        "yaw_rel_median": _median(yaws),
        "pitch_rel_median": _median(pitches),
    }
    return values, pipe.baseline


def _metadata_number(metadata: Mapping[str, Any], key: str) -> float:
    """추출 메타데이터의 값(수 또는 bool)을 실수로. 키가 없거나 수가 아니면 NaN이다. bool은 1.0 / 0.0이다."""
    value = metadata.get(key)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return math.nan


def _recording_values(
    frames: Sequence[RawFrame], spec: PipelineSpec, metadata: Mapping[str, Any] | None = None
) -> dict[str, float]:
    """세션의 녹화 메타데이터: 평균 프레임률, 해상도, 얼굴 크기, 밝기, 가변 프레임률 표시.

    metadata는 추출 결과 parquet의 파일 메타데이터(`extract.read_extract_metadata`)다. 추출이 적어 둔 값이 있으면 그것을
    쓴다(`extract.extract_video`의 설명).
    - `brightness`: `brightness_mean`(MediaPipe에 넘긴 프레임의 밝기 평균). 메타데이터에만 있다. 없으면 NaN.
    - `variable_fps`: `variable_intervals`(원본 영상의 프레임 간격이 변하는지)를 1.0 / 0.0으로. 없으면 NaN.
    - `mean_fps`, `width`, `height`: 메타데이터의 값(원본 영상의 평균 프레임률, 세운 뒤의 해상도). 메타데이터에 없으면
      저장된 프레임에서 구한다(프레임률은 (프레임 수 − 1)·1000 / 길이, 해상도는 첫 프레임의 w, h).
    - `face_size`: 얼굴 폭 / 프레임 긴 변의 중앙값(저장된 프레임에서 구한다).
    가정: 누락 모사 추출에서 메타데이터의 평균 프레임률은 모사가 버리기 전 원본 영상의 것이다. 교란 점검은 원본
    프레임률 추출을 읽으므로 두 정의의 차이는 디코딩에 실패한 프레임뿐이다.
    """
    metadata = {} if metadata is None else metadata
    values = {name: math.nan for name in ("mean_fps", "width", "height", "face_size", "brightness", "variable_fps")}
    values["brightness"] = _metadata_number(metadata, "brightness_mean")
    values["variable_fps"] = _metadata_number(metadata, "variable_intervals")
    for name in ("mean_fps", "width", "height"):
        values[name] = _metadata_number(metadata, name)
    if len(frames) == 0:
        return values
    if math.isnan(values["width"]):
        values["width"] = float(frames[0].w)
    if math.isnan(values["height"]):
        values["height"] = float(frames[0].h)
    duration_ms = frames[-1].t_ms - frames[0].t_ms
    if math.isnan(values["mean_fps"]) and duration_ms > 0:
        values["mean_fps"] = (len(frames) - 1) * 1000.0 / duration_ms
    left, right = spec.landmarks.face_width
    sizes: list[float] = []
    for frame in frames:
        if not frame.face:
            continue
        (ax, ay), (bx, by) = frame.lm[left], frame.lm[right]
        sizes.append(math.hypot((bx - ax) * frame.w, (by - ay) * frame.h) / max(frame.w, frame.h))
    values["face_size"] = _median(sizes)
    return values


def _probe_subject(job: _ProbeJob) -> list[tuple[int, dict[str, float]]]:
    """피험자 한 명의 세션마다 (품질 표의 행 위치, 녹화 메타데이터와 기준선 편차)를 낸다."""
    alert = [ref for ref in job.sessions if ref.class_index == CLASS_ALERT]
    # 피험자의 alert 기준선은 alert 영상(나뉘어 있으면 첫 부분)의 것이다(`dataset.build_step_table`의 규칙).
    source = min(alert, key=lambda ref: ref.part) if alert else None
    ordered = ([source] if source is not None else []) + [ref for ref in job.sessions if ref is not source]
    alert_baseline: Baseline | None = None
    out: list[tuple[int, dict[str, float]]] = []
    for ref in ordered:
        values = dict.fromkeys(OFFSET_COLUMNS, math.nan)
        path = Path(ref.raw_path)
        if not path.is_file():
            out.append((ref.row, {**_recording_values([], job.spec), **values}))
            continue
        frames = list(read_raw_frames(path))
        metadata = read_extract_metadata(path)
        own: tuple[dict[str, float], Baseline | None] | None = None
        if ref is source or ref.baseline_used == _BASELINE_SESSION:
            own = _state_medians(frames, job.spec, job.rules, None, None)
            if ref is source:
                alert_baseline = own[1]
        if ref.baseline_used == _BASELINE_SESSION or (ref.baseline_used == _BASELINE_ALERT and ref is source):
            if own is not None:
                values = own[0]
        elif ref.baseline_used == _BASELINE_ALERT:
            if alert_baseline is None:
                raise RuntimeError("quality table says the alert baseline was used but the alert session has none")
            values = _state_medians(frames, job.spec, job.rules, alert_baseline, ref.t_b_ms)[0]
        elif ref.baseline_used == _BASELINE_GLOBAL:
            values = _state_medians(frames, job.spec, job.rules, calibrate.global_baseline(job.spec), ref.t_b_ms)[0]
        out.append((ref.row, {**_recording_values(frames, job.spec, metadata), **values}))
    return out


def probe_table(
    quality: pd.DataFrame,
    manifest: pd.DataFrame,
    raw_dir: Path,
    spec: PipelineSpec,
    rules: RulesConfig,
    workers: int = 1,
) -> pd.DataFrame:
    """교란 점검용 표: 논리 영상마다 한 행 (PLAN 6장 "교란 점검").

    열: video_id, subject, fold, class_index와
    - 녹화 메타데이터 `mean_fps`, `width`, `height`, `face_size`(얼굴 폭 / 프레임 긴 변), `face_ratio`(얼굴 검출 비율),
      `brightness`(밝기), `variable_fps`(가변 프레임률 표시). 밝기와 가변 프레임률 표시는 추출 결과 parquet의
      메타데이터에서 읽는다(`_recording_values`). 그 메타데이터가 없는 파일(합성 데이터, 예전 추출)의 값은 NaN이다.
    - 기준선 편차 `open_closure_median`, `yaw_rel_median`, `pitch_rel_median`: quality(정규화 조건 하나의 품질 표)가
      말하는 그 조건의 기준선(`baseline_used`)으로 세션을 다시 재생해 얻은, 특징 누적 구간의 중앙값. alert 영상
      기준선 조건이면 alert 영상은 자기 기준선, 나머지 영상은 alert 영상의 기준선 대비 값이다.
    추출 결과를 세션마다 한 번 더 읽어 `DriverPipeline`에 넣는다(피험자 단위로 workers개 프로세스). 나뉜 영상은 부분별
    값을 평균한다(가변 프레임률 표시는 부분 중 그런 것의 비율이 된다). 추출 결과가 없거나 스텝이 없는 세션의 값은
    NaN이다. 모든 영상에서 NaN인 열은 `recording_metadata_probe`가 쓰지 않고 "없는 열"로 적는다.
    """
    needed = ("session_id", "rel_path", "subject")
    lacking = [name for name in needed if name not in manifest.columns]
    if lacking:
        raise ValueError(f"manifest lacks columns {lacking}")
    paths = dict(zip(manifest["session_id"].tolist(), manifest["rel_path"].tolist()))
    by_subject: dict[int, list[_ProbeSession]] = {}
    for row in range(len(quality)):
        session_id = str(quality["session_id"].iloc[row])
        if session_id not in paths:
            raise ValueError(f"quality table lists session {session_id!r} that the manifest does not have")
        ref = _ProbeSession(
            row=row,
            class_index=int(quality["class_index"].iloc[row]),
            part=int(quality["part"].iloc[row]),
            raw_path=str(raw_parquet_path(Path(raw_dir), Path(paths[session_id]))),
            baseline_used=str(quality["baseline_used"].iloc[row]),
            t_b_ms=int(quality["t_b_ms"].iloc[row]),
        )
        by_subject.setdefault(int(quality["subject"].iloc[row]), []).append(ref)
    jobs = [_ProbeJob(tuple(refs), spec, rules) for refs in by_subject.values()]
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if workers == 1 or len(jobs) <= 1:
        results = [_probe_subject(job) for job in jobs]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=context) as pool:
            results = list(pool.map(_probe_subject, jobs))

    measured = [name for name in RECORDING_COLUMNS if name != "face_ratio"] + list(OFFSET_COLUMNS)
    sessions = quality[list(PROBE_ID_COLUMNS)].reset_index(drop=True).copy()
    for name in measured:
        sessions[name] = math.nan
    for result in results:
        for row, values in result:
            for name in measured:
                sessions.loc[row, name] = values[name]
    sessions["face_ratio"] = quality["video_face_ratio"].to_numpy(dtype=np.float64)
    columns = list(RECORDING_COLUMNS) + list(OFFSET_COLUMNS)
    table = sessions.groupby(list(PROBE_ID_COLUMNS), sort=False)[columns].mean().reset_index()
    return table.sort_values(["subject", "class_index"], kind="stable").reset_index(drop=True)


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """교란 점검 하나의 결과.

    - columns: 입력으로 쓴 열. missing_columns: 그 점검의 열 중 표에 없어 쓰지 못한 것.
    - accuracy: 영상 단위 3클래스 정확도(피험자 분리 fold로 예측). chance는 1/3.
    - ci_low, ci_high: 피험자 부트스트랩 95% 구간. above_chance: 구간의 아래 끝이 우연 수준보다 높은지.
    - recall: 클래스별 재현율. confusion: 정답 × 예측. fold_accuracy: fold별 정확도.
    """

    name: str
    columns: tuple[str, ...]
    missing_columns: tuple[str, ...]
    n_videos: int
    n_subjects: int
    accuracy: float
    chance: float
    ci_low: float
    ci_high: float
    above_chance: bool
    recall: tuple[float, ...]
    confusion: tuple[tuple[int, ...], ...]
    fold_accuracy: dict[int, float]


def confound_probe(
    table: pd.DataFrame, columns: Sequence[str], name: str, cfg: ExperimentConfig
) -> ProbeResult:
    """표의 주어진 열만으로 LightGBM을 학습해 영상의 라벨을 맞혀 본다 (PLAN 6장 "교란 점검").

    table은 영상마다 한 행이고 `video_id`, `subject`, `fold`, `class_index`와 입력 열이 있어야 한다. fold마다 그 fold를
    빼고 학습해 그 fold의 영상을 예측한다(피험자 분리, `assert_subject_disjoint`). 모델은 고정된 작은 설정이다
    (`lgbm.num_leaves`의 가장 작은 값, `PROBE_MIN_DATA_IN_LEAF`, `PROBE_ROUNDS`회). 값이 없는 칸(NaN)은 그대로 넣는다.
    정확도의 피험자 부트스트랩 구간(`evaluation.bootstrap_n`회)의 아래 끝이 우연 수준(1/3)보다 높으면
    `above_chance`다. 보고만 하고 선택에는 쓰지 않는다.
    가정: "우연 수준을 크게 넘는다"를 95% 구간의 아래 끝이 1/3을 넘는 것으로 정한다.
    """
    columns = tuple(columns)
    lacking = [column for column in PROBE_ID_COLUMNS + columns if column not in table.columns]
    if lacking:
        raise ValueError(f"probe table lacks columns {lacking}")
    if len(columns) == 0:
        raise ValueError("a confound probe needs at least one input column")
    if table["video_id"].duplicated().any():
        raise ValueError("probe table lists a video more than once")
    folds = sorted(int(value) for value in table["fold"].unique())
    if len(folds) < 2:
        raise ValueError("a confound probe needs at least 2 folds")
    x = table[list(columns)].to_numpy(dtype=np.float64)
    y = table["class_index"].to_numpy(dtype=np.int64)
    fold = table["fold"].to_numpy(dtype=np.int64)
    subject = table["subject"].to_numpy(dtype=np.int64)
    params = lgbm_params(cfg.lgbm, min(cfg.lgbm.num_leaves), PROBE_MIN_DATA_IN_LEAF)
    params["min_data_in_bin"] = 1  # 영상이 적어도 값이 다른 칸을 나눌 수 있게 한다
    predicted = np.zeros(len(table), dtype=np.int64)
    for held_out in folds:
        test = fold == held_out
        assert_subject_disjoint(subject[~test], subject[test])
        booster = lgb.train(
            params, lgb.Dataset(x[~test], label=y[~test], feature_name=list(columns)), num_boost_round=PROBE_ROUNDS
        )
        scores = np.asarray(booster.predict(x[test]), dtype=np.float64).reshape(int(test.sum()), -1)
        predicted[test] = np.argmax(scores, axis=1)

    votes = pd.DataFrame(
        {"subject": subject, "fold": fold, "class_index": y, "pred_class": predicted, "correct": predicted == y}
    )
    metrics = evaluate.video_metrics(votes)
    low, high = evaluate.subject_bootstrap(
        votes,
        lambda sample: float(sample["correct"].mean()),
        n=cfg.evaluation.bootstrap_n,
        seed=cfg.evaluation.bootstrap_seed,
    )
    return ProbeResult(
        name=name,
        columns=columns,
        missing_columns=(),
        n_videos=len(table),
        n_subjects=int(len(np.unique(subject))),
        accuracy=metrics.accuracy,
        chance=CHANCE_ACCURACY,
        ci_low=low,
        ci_high=high,
        above_chance=bool(low > CHANCE_ACCURACY),
        recall=metrics.recall,
        confusion=tuple(tuple(int(value) for value in row[: len(CLASS_ORDER)]) for row in metrics.confusion),
        fold_accuracy={
            held_out: float(votes.loc[votes["fold"] == held_out, "correct"].mean()) for held_out in folds
        },
    )


def _probe_with(table: pd.DataFrame, wanted: Sequence[str], name: str, cfg: ExperimentConfig) -> ProbeResult:
    present = tuple(column for column in wanted if column in table.columns and bool(table[column].notna().any()))
    missing = tuple(column for column in wanted if column not in present)
    if not present:
        raise ValueError(f"probe table has none of the columns {list(wanted)}")
    return dataclasses.replace(confound_probe(table, present, name, cfg), missing_columns=missing)


def recording_metadata_probe(table: pd.DataFrame, cfg: ExperimentConfig) -> ProbeResult:
    """교란 점검 (1): 녹화 메타데이터(평균 프레임률, 해상도, 얼굴 크기, 얼굴 검출률, 밝기, 가변 프레임률 표시)만으로
    라벨을 맞혀 본다.

    table은 `probe_table`의 표(또는 같은 열을 가진 품질·메타데이터 표)다. `RECORDING_COLUMNS` 중 표에 있고 값이 하나라도
    있는 열만 쓰고, 없는 열(추출 메타데이터가 없어 모두 NaN인 밝기와 가변 프레임률 표시 등)은 `missing_columns`에 적는다.
    """
    return _probe_with(table, RECORDING_COLUMNS, PROBE_RECORDING, cfg)


def baseline_offset_probe(table: pd.DataFrame, cfg: ExperimentConfig) -> ProbeResult:
    """교란 점검 (2): 영상별 기준선 편차(뜬 눈 closure 중앙값, 상대 yaw·pitch 중앙값)만으로 라벨을 맞혀 본다.

    table은 `probe_table`의 표다. alert 영상 기준선 조건에서는 alert 영상의 편차가 0에 가깝고 나머지 영상만 다른 세션의
    기준선 대비 값을 갖는다. 이 점검이 우연 수준을 넘으면 그 차이만으로 라벨이 드러난다는 뜻이다.
    """
    return _probe_with(table, OFFSET_COLUMNS, PROBE_BASELINE_OFFSET, cfg)


def metadata_note(table: pd.DataFrame) -> str:
    """교란 점검 리포트에 적는 문장: 밝기와 가변 프레임률 표시가 추출 메타데이터에서 왔는지, 없어서 빠졌는지."""
    counts = {
        name: int(table[name].notna().sum()) if name in table.columns else 0 for name in METADATA_ONLY_COLUMNS
    }
    source = (
        "밝기(`brightness`)와 가변 프레임률 표시(`variable_fps`)는 추출 결과 parquet의 메타데이터(`brightness_mean`, "
        "`variable_intervals`)에서 읽는다. 평균 프레임률과 해상도도 그 메타데이터에 있으면 그 값을 쓴다."
    )
    if all(count == 0 for count in counts.values()):
        return (
            f"{source} 이 데이터의 추출 결과에는 그 메타데이터가 없다(합성 데이터 `drowsy.synth`의 parquet이나 이 값을 "
            "적기 전의 추출). 그래서 점검 (1)은 밝기와 가변 프레임률 표시 없이 돌았고 두 열은 \"없는 열\"에 적혀 있다."
        )
    found = ", ".join(f"`{name}` {count}개" for name, count in counts.items())
    return f"{source} 값이 있는 논리 영상 수(전체 {len(table)}개 중): {found}."


def run_probes(
    cfg: ExperimentConfig,
    *,
    arm: str | None = None,
    run_name: str = train.DEFAULT_CV_RUN_NAME,
    command: str = "",
    workers: int | None = None,
    rebuild: bool = False,
) -> list[ProbeResult]:
    """교란 점검 두 가지를 돌리고 `{reports_dir}/{run_name}/probes/`에 결과를 쓴다 (PLAN 6장, 11장 E0).

    arm: 기준선 편차를 볼 정규화 조건. None이면 `training.export_arm`. workers: 재생 프로세스 수(None이면
    `extraction.workers`). rebuild: 스텝 표 캐시를 다시 만든다. 쓰는 파일: summary.md(첫 줄이 데이터 종류, 우연 수준을
    넘은 점검이 있으면 그다음 줄에 적는다), probes.json, probe_table.csv. `drowsy.train`의 리포트는 이 폴더의 결과를
    읽어 맨 앞에 적는다. probes.json에는 추출 결과의 회전 점검 기록(`rotation_check`)도 적고, 회전이 확인되지 않은 영상이
    있으면 그 수가 설명에 실린다(`train.rotation_note`).
    """
    arm = cfg.training.export_arm if arm is None else arm
    data = train.load_cv_data(cfg, (arm,), rebuild=rebuild)
    rules = train.rules_for(cfg, data.feature_spec)
    quality = data.tables[arm].quality
    table = probe_table(
        quality,
        data.manifest,
        data.raw_dir,
        data.feature_spec.pipeline,
        rules,
        workers=cfg.extraction.workers if workers is None else workers,
    )
    results = [recording_metadata_probe(table, cfg), baseline_offset_probe(table, cfg)]

    run_dir = cfg.paths.reports_dir / run_name / train.PROBES_SUBDIR
    run_dir.mkdir(parents=True, exist_ok=True)
    note = train.data_note(data.synthetic)
    description = f"{data.description}. Confound probes (PLAN section 6), offsets under arm {arm}"
    with (run_dir / "probe_table.csv").open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# {note}\n")
        table.to_csv(handle, index=False, lineterminator="\n")
    train.write_json(
        {
            "data_note": note,
            "synthetic": data.synthetic,
            "description": description,
            "command": command,
            "arm": arm,
            "rotation_check": train.rotation_record(data.rotation),
            "metadata_columns": {
                name: {"videos_with_value": int(table[name].notna().sum()), "videos": int(len(table))}
                for name in METADATA_ONLY_COLUMNS
            },
            "metadata_note": metadata_note(table),
            "bootstrap": {"n": cfg.evaluation.bootstrap_n, "seed": cfg.evaluation.bootstrap_seed},
            "probes": [
                {**dataclasses.asdict(result), "fold_accuracy": {str(k): v for k, v in result.fold_accuracy.items()}}
                for result in results
            ],
        },
        run_dir / train.PROBES_FILE,
    )
    flagged = [result.name for result in results if result.above_chance]
    lines = [f"# {note}", ""]
    if flagged:
        lines += [f"**교란 점검이 우연 수준을 넘었다: {', '.join(flagged)}.** 아래 표를 먼저 볼 것.", ""]
    lines += [
        description,
        "",
        "입력 열만으로 LightGBM을 학습해 영상의 라벨(3클래스)을 맞혀 본 결과다. 보고만 하고 모델 선택에는 쓰지 않는다. "
        "우연 수준은 1/3이고, 피험자 부트스트랩 95% 구간의 아래 끝이 그보다 높으면 \"우연 수준을 넘음\"으로 적는다.",
        "",
        f"- 생성 명령: `{command}`",
        f"- 논리 영상 {len(table)}개, 피험자 {table['subject'].nunique()}명. 기준선 편차는 정규화 조건 `{arm}`의 값이다",
        "",
        "| 점검 | 입력 열 | 없는 열 | 영상 정확도 [95% 구간] | 재현율 alert / low_vigilant / drowsy | 우연 수준을 넘음 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            f"| `{result.name}` | {', '.join(result.columns)} | {', '.join(result.missing_columns) or '-'} | "
            f"{result.accuracy:.3f} [{result.ci_low:.3f}, {result.ci_high:.3f}] | "
            f"{' / '.join(f'{value:.2f}' for value in result.recall)} | {'예' if result.above_chance else '아니오'} |"
        )
    lines += ["", metadata_note(table), ""]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return results


# ---------------------------------------------------------------------------
# 명령
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m drowsy.baseline_lgbm`.

    종료 코드: 0 성공, 2 시작할 수 없음(실행 이름, 설정, 데이터, 다른 실행의 리포트가 있는 폴더), 3 피험자 겹침으로 중단,
    4 실제 데이터에서 E1 통과 기준 미달(신경망으로 가기 전에 멈추고 보고한다. 리포트는 써 둔다).
    `--run-name`은 폴더 이름 하나여야 한다(`config.check_run_name`). 경로 구분자나 `..`이 든 이름은 설정과 데이터를 읽기
    전에 `error: ...` 한 줄과 종료 코드 2로 거부한다.
    `cv --all-w`(W 비교, 보고용)와 부가 실행(`cv --no-velocity`, `cv --foldwise-global`)은 E1 기준에 못 미쳐도 4로
    끝나지 않는다. 그 판단은 기준 실행(`cv`)의 몫이다.
    출력의 첫 줄은 데이터 설명, 둘째 줄은 추출 결과의 회전 점검 기록이다(`train.print_rotation_check`: 구분별 파일 수와,
    회전이 확인되지 않은 영상이 있으면 그 아래에 영상마다 한 줄). 학습과 재생을 시작하기 전에 출력하고 종료 코드는 바꾸지 않는다.
    """
    parser = argparse.ArgumentParser(
        prog="python -m drowsy.baseline_lgbm", description="LightGBM baseline and confound probes (PLAN 5.3, 6)."
    )
    train.add_common_arguments(parser)
    commands = parser.add_subparsers(dest="command", required=True)
    cv = commands.add_parser("cv", help="nested cross-validation with reports")
    train.add_common_arguments(cv, suppress=True)
    cv.add_argument("--arms", nargs="+", choices=ARMS, default=None, help="default: lgbm.reference_arm")
    cv.add_argument("--fast", action="store_true", help="smoke-run profile (config block `fast`)")
    cv.add_argument("--closure-w", type=float, default=None, help="rebuild the step features with this closure W")
    cv.add_argument(
        "--all-w",
        action="store_true",
        help="closure-W comparison (E1, reporting only): run lgbm.w_grid for the reference arm into one report",
    )
    cv.add_argument(
        "--no-velocity",
        action="store_true",
        help="supplementary run (E3): drop the statistics of the velocity features #7 and #8; "
        f"report in <run>/{NO_VELOCITY_SUBDIR}/",
    )
    cv.add_argument(
        "--foldwise-global",
        action="store_true",
        help="sensitivity check (PLAN 4.3, needs --arms global): global constants recomputed per outer fold from "
        f"the training subjects; report in <run>/{FOLD_GLOBAL_SUBDIR}/",
    )
    cv.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the report of another run (other arms, profile, closure W or kind of data) in the same folder",
    )
    probes = commands.add_parser("probes", help="the two confound probes")
    train.add_common_arguments(probes, suppress=True)
    probes.add_argument("--arm", choices=ARMS, default=None, help="arm of the baseline offsets (default: export arm)")
    probes.add_argument("--workers", type=int, default=None, help="replay processes (default: extraction.workers)")
    args = parser.parse_args(argv)
    command = train.command_line("drowsy.baseline_lgbm", argv)

    try:
        check_run_name(args.run_name)  # 두 하위 명령(cv와 그 선택 사항들, probes) 모두 다른 일을 하기 전에 확인한다
        cfg = train.with_data_root(load_config(args.config), args.data_root)
        source = open_dataset(cfg)
        print(source.description, flush=True)
        train.print_rotation_check(cfg, source)  # 학습과 재생을 시작하기 전에 알린다(실행 기록과 리포트 설명에도 남는다)
        if args.command == "cv" and args.all_w:
            if args.arms is not None or args.closure_w is not None or args.no_velocity or args.foldwise_global:
                raise ValueError(
                    "--all-w runs lgbm.w_grid for lgbm.reference_arm; do not pass --arms, --closure-w, "
                    "--no-velocity or --foldwise-global"
                )
            print(NATIVE_ONLY_NOTE)
            result = run_lgbm_w_grid(
                cfg,
                fast=args.fast,
                run_name=args.run_name,
                config_path=Path(args.config),
                command=command,
                overwrite=args.overwrite,
                rebuild=args.rebuild,
            )
            for record in result.selection:
                print(f"[W {record['closure_w']:g}] fold {record['fold']}: {record['chosen'][evaluate.FAMILY_LGBM]}")
            print("seconds: " + ", ".join(f"{key} {value:.1f}" for key, value in result.timings.items()))
            print(
                f"closure-W comparison (supplementary, reporting only; the D11 reference is {train.LGBM_SUBDIR}/): "
                f"{result.run_dir / W_GRID_MD_FILE}"
            )
            print(f"reports: {result.report.summary_path}")
        elif args.command == "cv":
            print(NATIVE_ONLY_NOTE)
            result = run_lgbm_cv(
                cfg,
                arms=args.arms,
                fast=args.fast,
                run_name=args.run_name,
                config_path=Path(args.config),
                command=command,
                closure_w=args.closure_w,
                overwrite=args.overwrite,
                rebuild=args.rebuild,
                no_velocity=args.no_velocity,
                foldwise_global=args.foldwise_global,
            )
            for record in result.selection:
                tag = f", {record['feature_source']}" if "feature_source" in record else ""
                print(f"[{record['arm']}{tag}] fold {record['fold']}: {record['chosen'][evaluate.FAMILY_LGBM]}")
            print("seconds: " + ", ".join(f"{key} {value:.1f}" for key, value in result.timings.items()))
            print(f"reports: {result.report.summary_path}")
            reference = cfg.lgbm.reference_arm
            records = [record for record in result.selection if record["arm"] == reference]
            if args.no_velocity or args.foldwise_global:
                # 부가 실행은 E1 통과 기준을 판단하지 않는다. 그 판단은 기준 실행(`cv`)의 몫이다.
                print(
                    f"supplementary run: {train.LGBM_SUBDIR}/ (the reference of the comparison table and D11) is "
                    "not touched and the E1 criterion is not judged here"
                )
            elif records:
                gate = e1_gate(records, cfg.evaluation.lgbm_min_macro_f1)
                met = "met" if gate["passed"] else "NOT met"
                check_only = " (synthetic data: pipeline check only)" if source.synthetic else ""
                print(
                    f"E1 criterion (inner-validation macro-F1 mean >= {gate['threshold']:g}, arm {reference}): "
                    f"{met}{check_only}"
                )
                if not gate["passed"] and not source.synthetic:
                    print("stop before the neural networks and report (PLAN section 11, E1)", file=sys.stderr)
                    return 4
        else:
            results = run_probes(
                cfg,
                arm=args.arm,
                run_name=args.run_name,
                command=command,
                workers=args.workers,
                rebuild=args.rebuild,
            )
            for probe in results:
                print(
                    f"{probe.name}: columns {list(probe.columns)}, missing {list(probe.missing_columns)}, "
                    f"above chance: {'yes' if probe.above_chance else 'no'}"
                )
            print(f"reports: {cfg.paths.reports_dir / args.run_name / train.PROBES_SUBDIR / 'summary.md'}")
    except SubjectOverlapError as exc:
        print(f"error: subject overlap, run stopped: {exc}", file=sys.stderr)
        return 3
    except (ValueError, OSError, ExtractError) as exc:  # SpecError, ManifestError도 ValueError다
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    # `python -m`으로 실행하면 이 파일이 `__main__`으로 한 번 더 올라온다. 재생 작업을 다른 프로세스로 넘길 때
    # 클래스가 두 모듈에 따로 있지 않도록, 패키지에서 import한 쪽의 main을 부른다.
    from drowsy.baseline_lgbm import main as _main

    sys.exit(_main())
