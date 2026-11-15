"""E0 신호 점검·상수 확정 도구와 15 fps 깜빡임 길이 모사 (PLAN 4.2절, 4.3절, 4.5절, 11장 E0). 이식 대상이 아니다.

명령 (저장소 루트에서 실행)
    python -m drowsy.signal_check e0 --config config/default.yaml [--run-name N]
        [--override closure.rho=0.30 ...] [--workers W] [--fps-sim]
    python -m drowsy.signal_check blink-sim --config config/default.yaml [--run-name N] [--seed S]

e0 (PLAN 11장 E0)
    라벨을 쓰지 않는 신호 점검이다. 목록에서 `session_id`, `subject`, `rel_path` 세 열만 꺼내 쓰고 라벨·클래스·fold 열은
    읽지 않는다(`label_free`). 세션(파일)마다 저장된 프레임을 `DriverPipeline`에 넣어 앱처럼 자기 앞부분으로 보정하고
    (`collect_session`), 기준선 통계와 깜빡임 통계를 모아 통과 기준 셋을 판정한다.
      1. 기준선이 60초에 대체 없이 완성된 세션이 `evaluation.signal_gate.min_baseline_ok_ratio`(0.9) 이상
      2. 깜빡임이 분당 5~40회인 세션이 `min_blink_rate_ok_ratio`(0.8) 이상
      3. 깜빡임 정점 closure의 중앙값이 0.8 이상인 세션이 `min_peak_closure_ok_ratio`(0.8) 이상
    전역 상수(E, B, jaw0)는 세션별 기준선 통계의 중앙값으로 제안한다(`global_constants`, PLAN 4.3절).
    `--override`는 후보 상수로 기준을 다시 판정해 보는 용도다(`spec.apply_overrides`). 이 도구는 `config/feature_spec.json`을
    고치지 않는다. `reports/<실행명>/e0/`에 summary.md, per_video.csv, criteria.json, config.yaml(설정 사본)과
    제안 파일(지금 spec에 덮어쓴 값과 제안 전역 상수를 넣고 spec_version을 올리고 pipeline_hash를 다시 계산한 것)을 쓴다.
    추출이 남긴 회전 점검 기록도 읽는다(`dataset.read_rotation_checks`): 자동 점검이 각도를 가르지 못했는데 수동 회전
    보정이 없는 영상의 수를 한 줄로 출력하고, 그 목록을 summary.md의 "회전 점검" 절과 criteria.json의 `rotation_check`에
    싣는다. 그런 영상의 신호는 틀린 각도에서 나왔을 수 있으므로 상수를 확정하기 전에 영상을 직접 확인한다.
    `--fps-sim`은 기기 프레임 누락을 모사한 추출 결과로 같은 점검을 하고 `e0_fps_sim/`에 쓴다.
    종료 코드: 0 성공, 2 시작할 수 없음, 4 실제 데이터에서 통과 기준 미달(리포트는 써 둔다).
    `--run-name`은 폴더 이름 하나여야 한다(`config.check_run_name`. 경로 구분자나 `..`이 있으면 두 명령 모두 2로 끝난다).

blink-sim (PLAN 4.5절 마지막 항목)
    합성 깜빡임 파형을 30 fps, 15 fps, 프레임이 무작위로 빠진 15 fps로 표본화해 `ClosureRunDetector`(특징용 임계값)에
    넣고, 지속 시간 오차(보간한 시각과 프레임 시각), 길이별 검출률, 연속 파형 대비 진폭과 속도의 비를 잰다.
    `reports/<실행명>/blink_sim/`에 summary.md와 blink_sim.csv를 쓴다. 눈을 잰 값이 아니라 모사다.

가정 (계획서가 정하지 않아서 정한 것. 함수별 가정은 각 docstring에 있다)
가정: E0의 판정 단위는 세션(파일)이다. 나뉜 영상의 부분(`10_1`, `10_2`)은 앱의 세션처럼 각자 자기 기준선으로 따로
  재생되므로 따로 센다. RLDD라면 논리 영상 180개가 아니라 파일 182개가 분모다. 출력 파일 이름은 지시대로
  per_video.csv이고 한 행이 세션 하나다.
가정: `dataset.replay_video`는 프레임 상태를 돌려주지 않아 closure 유효 시간을 셀 수 없다. 그래서 같은
  `DriverPipeline`을 프레임 하나씩 직접 부르는 `collect_session`을 둔다. 신호, closure, 감김 검출은 파이프라인이 낸
  값만 쓴다(PLAN 3.1절). 겹치는 값(기준선, 감김 구간, 프레임 수, 얼굴 시간)이 `replay_video`와 같은지는 테스트가 본다.
가정: 실제 데이터에서 통과 기준에 미달하면 종료 코드 4로 끝낸다(E1의 `baseline_lgbm cv`와 같은 방식. PLAN 12장:
  미달 시 학습 전에 멈춘다). 합성 데이터에서는 파이프라인 점검일 뿐이므로 0이다.
가정: 합성 데이터로 만든 제안 파일의 이름은 `feature_spec_proposed.synthetic.json`이고 "받아들인 뒤 실행할 것" 목록을
  쓰지 않는다. spec 로더가 모르는 키를 거부해서 파일 안에 합성 표시를 넣을 수 없기 때문이다. 합성 세션에서 나온
  상수가 `config/feature_spec.json`에 들어가면 안 된다.
가정: `--fps-sim` 실행은 제안 파일을 쓰지 않는다. 전역 상수는 원본 프레임률 추출에서 정한다(PLAN 4.3절).
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import multiprocessing
import random
import shutil
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from drowsy import calibrate, signals
from drowsy.blink import ClosureRunDetector
from drowsy.calibrate import BaselineDiagnostics
from drowsy.config import ExperimentConfig, SignalGate, check_run_name, load_config
from drowsy.dataset import (
    BASELINE_NONE,
    RAW_MISSING,
    RAW_OK,
    SYNTHETIC_NOTE,
    DatasetSource,
    ManifestError,
    RotationSummary,
    dataset_rotation_checks,
    default_rules,
    open_dataset,
    rotation_check_line,
    rotation_check_section,
    spec_pipeline_hash,
)
from drowsy.extract import OVERRIDES_KEY, ExtractError, raw_parquet_path, read_raw_frames
from drowsy.pipeline import DriverPipeline
from drowsy.spec import (
    FeatureSpec,
    PipelineSpec,
    RulesConfig,
    RunDetectorSpec,
    SpecError,
    apply_overrides,
    compute_pipeline_hash,
    feature_spec_from_dict,
    load_feature_spec,
    load_rules,
    read_feature_spec_dict,
    write_feature_spec,
)
from drowsy.types import (
    BASELINE_FALLBACK_GLOBAL,
    BASELINE_OFF_AXIS,
    BASELINE_OK,
    Baseline,
    ClosureRun,
    RawFrame,
)

DEFAULT_RUN_NAME = "signal_check"
E0_SUBDIR = "e0"
E0_FPS_SIM_SUBDIR = "e0_fps_sim"
BLINK_SIM_SUBDIR = "blink_sim"

SUMMARY_FILE = "summary.md"
PER_VIDEO_FILE = "per_video.csv"
CRITERIA_FILE = "criteria.json"
PROPOSED_SPEC_FILE = "feature_spec_proposed.json"
SYNTHETIC_PROPOSED_SPEC_FILE = "feature_spec_proposed.synthetic.json"
CONFIG_COPY_FILE = "config.yaml"
BLINK_SIM_CSV_FILE = "blink_sim.csv"

E0_REAL_NOTE = "REAL data: label-free signal check (PLAN section 11, E0); no model performance in this report"
BLINK_SIM_NOTE = "SIMULATION on synthetic blink waveforms: not a measurement on eyes, not a performance result"

# 목록에서 읽는 열은 이 셋뿐이다. 라벨·클래스·fold 열(`label`, `class_index`, `fold`)과 그것으로 만든 `video_id`는 읽지 않는다.
LABEL_FREE_COLUMNS: tuple[str, ...] = ("session_id", "subject", "rel_path")

# 개인 통계가 있는 기준선 상태. 전역 상수로 대체된 기준선에는 개인 통계가 없다.
USABLE_BASELINE_STATUSES: tuple[str, ...] = (BASELINE_OK, BASELINE_OFF_AXIS)
BASELINE_COLUMNS: tuple[str, ...] = ("base_e_l", "base_e_r", "base_b_l", "base_b_r", "base_jaw0")
CONSTANT_DECIMALS = 4  # 제안 전역 상수의 소수 자릿수(가정은 `global_constants`에 적었다)

CRITERION_BASELINE = "baseline_complete"
CRITERION_BLINK_RATE = "blink_rate"
CRITERION_PEAK_CLOSURE = "peak_closure"

# 계획서와 SPEC.md 19장이 잠정 또는 E0에서 바꾸는 값으로 표시한 `pipeline` 블록의 경로(앞부분 일치).
PROVISIONAL_PATHS: tuple[str, ...] = (
    "closure.rho",
    "closure.b_closed",
    "run_detector.enter_thr",
    "run_detector.exit_thr",
    "step_features.perclos_thr",
    "global_constants.e",
    "global_constants.b",
    "global_constants.jaw0",
    "yawn.",
    "nod.",
)

_MS_PER_MINUTE = 60000.0
_MAX_LISTED = 20  # 요약에 적는 세션 수의 상한(전체는 per_video.csv)


# ---------------------------------------------------------------------------
# 세션 재생
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SessionSignals:
    """세션 하나를 자기 기준선으로 재생해 모은 것 (`collect_session`).

    - baseline, diagnostics: 이 세션에서 보정한 기준선과 판정 통계. 보정이 끝나기 전에 프레임이 끝났으면 None.
    - runs: 특징용 감김 검출기(진입 0.5 / 이탈 0.3)가 낸 구간. 특징 구간(T_b 이후)에서만 나온다.
    - n_frames, n_dropped, duration_ms, face_ms: `dataset.ReplayResult`의 같은 이름과 같은 정의다.
    - feature_ms: 특징 구간의 길이 = (마지막 프레임의 세션 시간) − T_b. 기준선이 없으면 0.
    - closure_valid_ms: 특징 구간에서 closure가 유효한 프레임이 대표한 시간. 다음 프레임까지의 간격이고 최대
      `clock.max_hold_ms`다(PLAN 3.3절). 스텝 특징의 "유효 시간"과 같은 규칙이다.
    """

    baseline: Baseline | None
    diagnostics: BaselineDiagnostics | None
    runs: tuple[ClosureRun, ...]
    n_frames: int
    n_dropped: int
    duration_ms: int
    face_ms: int
    feature_ms: int
    closure_valid_ms: int


def collect_session(
    frames: Iterable[RawFrame], spec: PipelineSpec, rules: RulesConfig | None = None
) -> SessionSignals:
    """저장된 프레임을 `DriverPipeline`에 차례로 넣고(앱처럼 세션 앞부분에서 보정한다) E0에 필요한 것을 모은다.

    rules가 None이면 `dataset.default_rules(spec)`. 여기서 모으는 값은 규칙과 무관하다.
    특징 구간에 속하는 프레임은 파이프라인과 같은 조건으로 가린다: 기준선이 있고 세션 시간 ≥ T_b.
    """
    pipe = DriverPipeline(spec, rules if rules is not None else default_rules(spec))
    max_hold_ms = spec.clock.max_hold_ms
    runs: list[ClosureRun] = []
    first_t_ms = 0
    last_t_ms = 0
    last_face = False
    last_valid = False  # 직전 프레임이 특징 구간에 있고 closure가 유효했는지
    face_ms = 0
    closure_valid_ms = 0
    for frame in frames:
        out = pipe.update(frame)
        if out is None:
            continue
        state = out.state
        t_ms = state.t_ms
        if pipe.frames_accepted == 1:
            first_t_ms = t_ms
        else:
            held_ms = min(t_ms - last_t_ms, max_hold_ms)
            if last_face:
                face_ms += held_ms
            if last_valid:
                closure_valid_ms += held_ms
        baseline = pipe.baseline
        in_feature_period = baseline is not None and t_ms - first_t_ms >= baseline.t_complete_ms
        last_t_ms = t_ms
        last_face = state.face
        last_valid = in_feature_period and state.closure_valid
        runs.extend(out.runs)

    baseline = pipe.baseline
    duration_ms = last_t_ms - first_t_ms
    return SessionSignals(
        baseline=baseline,
        diagnostics=pipe.baseline_diagnostics,
        runs=tuple(runs),
        n_frames=pipe.frames_accepted,
        n_dropped=pipe.dropped_frames,
        duration_ms=duration_ms,
        face_ms=face_ms,
        feature_ms=0 if baseline is None else duration_ms - baseline.t_complete_ms,
        closure_valid_ms=closure_valid_ms,
    )


def _quartiles(values: Sequence[float]) -> tuple[float, float, float]:
    """(25 분위수, 중앙값, 75 분위수). 분위수는 최근접 순위, 중앙값은 짝수 개일 때 가운데 두 값의 평균이다(PLAN 4.2절)."""
    if len(values) == 0:
        return (math.nan, math.nan, math.nan)
    return (
        calibrate.nearest_rank_quantile(values, 0.25),
        calibrate.median(values),
        calibrate.nearest_rank_quantile(values, 0.75),
    )


def _empty_record(raw_status: str) -> dict[str, Any]:
    """추출 결과가 없는 세션의 행. 값이 없는 실수는 NaN, 개수는 0이다."""
    return {
        "raw_status": raw_status,
        "n_frames": 0,
        "n_dropped": 0,
        "duration_ms": 0,
        "face_ratio": 0.0,
        "baseline_status": BASELINE_NONE,
        "t_b_ms": -1,
        "calib_face_frames": 0,
        "calib_face_ratio": 0.0,
        "base_e_l": math.nan,
        "base_e_r": math.nan,
        "base_b_l": math.nan,
        "base_b_r": math.nan,
        "base_jaw0": math.nan,
        "base_yaw0": math.nan,
        "base_pitch0": math.nan,
        "base_roll0": math.nan,
        "feature_ms": 0,
        "closure_valid_ms": 0,
        "closure_valid_ratio": math.nan,
        "n_runs": 0,
        "n_truncated": 0,
        "n_blinks": 0,
        "blink_rate_per_min": math.nan,
        "peak_closure_q25": math.nan,
        "peak_closure_median": math.nan,
        "peak_closure_q75": math.nan,
        "n_peak_ge_perclos_thr": 0,
        "n_peak_ge_rule_enter": 0,
        "blink_dur_q25_ms": math.nan,
        "blink_dur_median_ms": math.nan,
        "blink_dur_q75_ms": math.nan,
        "blink_dur_mean_ms": math.nan,
        "blink_dur_max_ms": math.nan,
    }


def session_record(collected: SessionSignals, spec: PipelineSpec, rule_enter_thr: float) -> dict[str, Any]:
    """세션 하나의 E0 통계(per_video.csv의 한 행에서 식별자와 판정 열을 뺀 것).

    - `baseline_status`, `t_b_ms`, `base_*`: 이 세션 자신의 기준선(`dataset`의 품질 표와 같은 이름, 같은 뜻).
      `base_yaw0`, `base_pitch0`, `base_roll0`은 기준 회전 R0의 오일러 각(도)이다. 전역 상수로 대체된 세션의
      `base_*`는 개인 통계가 아니라 전역 상수 그대로다. 보정이 안 끝났으면 status는 `none`, `t_b_ms`는 −1이다.
    - `feature_ms`, `closure_valid_ms`, `closure_valid_ratio`: 특징 구간(T_b 이후)의 길이, 그 안의 closure 유효 시간, 둘의 비.
    - `n_runs`, `n_truncated`, `n_blinks`: 특징용 감김 구간 전체, 그중 잘린 것, 잘리지 않은 것(= 깜빡임).
    - `blink_rate_per_min` = n_blinks × 60000 / closure_valid_ms (스텝 특징 #4와 같은 식을 특징 구간 전체에 적용).
    - `peak_closure_*`: 깜빡임의 정점 closure(`ClosureRun.amplitude`)의 25 분위수, 중앙값, 75 분위수.
    - `n_peak_ge_perclos_thr`, `n_peak_ge_rule_enter`: 정점이 `step_features.perclos_thr`(0.8) 이상, 규칙용 검출기의
      진입 임계(0.7) 이상인 깜빡임 수. 임계값을 확정할 때 보는 값이고 통과 기준에는 쓰지 않는다.
    - `blink_dur_*_ms`: 깜빡임 지속 시간(`duration_ms`, 상한 없이)의 사분위수, 평균, 최댓값.

    가정: 깜빡임은 특징용 검출기가 낸 잘리지 않은 구간 전부다(PLAN 4.6절의 "이벤트"). 길이로 거르지 않으므로
      긴 감김도 들어간다.
    가정: closure 유효 시간이 0이면 분당 횟수는 0이다(PLAN 3.3절: 0으로 나누게 되는 비율은 0). 기준선이 없는 세션은
      특징 구간이 없으므로 분당 횟수와 깜빡임 통계가 NaN이다. 깜빡임이 없는 세션의 정점·길이 통계도 NaN이다.
    가정: 사분위수는 최근접 순위, 중앙값은 PLAN 4.2절의 정의(짝수 개면 가운데 두 값의 평균)를 쓴다.
    """
    record = _empty_record(RAW_OK)
    record.update(
        n_frames=collected.n_frames,
        n_dropped=collected.n_dropped,
        duration_ms=collected.duration_ms,
        face_ratio=collected.face_ms / collected.duration_ms if collected.duration_ms > 0 else 0.0,
    )
    if collected.diagnostics is not None:
        record.update(
            calib_face_frames=collected.diagnostics.n_face_frames,
            calib_face_ratio=collected.diagnostics.face_ratio,
        )
    baseline = collected.baseline
    if baseline is None:
        return record

    yaw0, pitch0, roll0 = signals.euler_from_rotation(baseline.r0)
    blinks = [run for run in collected.runs if not run.truncated]
    peaks = [run.amplitude for run in blinks]
    durations = [float(run.duration_ms) for run in blinks]
    peak_q25, peak_median, peak_q75 = _quartiles(peaks)
    dur_q25, dur_median, dur_q75 = _quartiles(durations)
    valid_ms = collected.closure_valid_ms
    record.update(
        baseline_status=baseline.status,
        t_b_ms=baseline.t_complete_ms,
        base_e_l=baseline.e_l,
        base_e_r=baseline.e_r,
        base_b_l=baseline.b_l,
        base_b_r=baseline.b_r,
        base_jaw0=baseline.jaw0,
        base_yaw0=yaw0,
        base_pitch0=pitch0,
        base_roll0=roll0,
        feature_ms=collected.feature_ms,
        closure_valid_ms=valid_ms,
        closure_valid_ratio=valid_ms / collected.feature_ms if collected.feature_ms > 0 else 0.0,
        n_runs=len(collected.runs),
        n_truncated=len(collected.runs) - len(blinks),
        n_blinks=len(blinks),
        blink_rate_per_min=len(blinks) * _MS_PER_MINUTE / valid_ms if valid_ms > 0 else 0.0,
        peak_closure_q25=peak_q25,
        peak_closure_median=peak_median,
        peak_closure_q75=peak_q75,
        n_peak_ge_perclos_thr=sum(1 for peak in peaks if peak >= spec.step_features.perclos_thr),
        n_peak_ge_rule_enter=sum(1 for peak in peaks if peak >= rule_enter_thr),
        blink_dur_q25_ms=dur_q25,
        blink_dur_median_ms=dur_median,
        blink_dur_q75_ms=dur_q75,
        blink_dur_mean_ms=sum(durations) / len(durations) if durations else math.nan,
        blink_dur_max_ms=max(durations) if durations else math.nan,
    )
    return record


# ---------------------------------------------------------------------------
# 세션 표
# ---------------------------------------------------------------------------


def label_free(manifest: pd.DataFrame) -> pd.DataFrame:
    """목록에서 E0이 읽는 세 열(`session_id`, `subject`, `rel_path`)만 남기고 (피험자, 세션) 순으로 정렬한 사본.

    라벨·클래스·fold 열과 `video_id`는 여기서 버려진다. 뒤의 계산은 이 표만 받으므로 라벨을 읽을 길이 없다.
    행 순서도 목록의 순서(라벨로 정렬돼 있다)를 따르지 않고 다시 정한다.
    """
    missing = [name for name in LABEL_FREE_COLUMNS if name not in manifest.columns]
    if missing:
        raise ManifestError(f"manifest lacks columns {missing}; build it with dataset.build_manifest")
    table = manifest[list(LABEL_FREE_COLUMNS)].copy()
    return table.sort_values(["subject", "session_id"], kind="stable").reset_index(drop=True)


@dataclass(frozen=True, slots=True)
class _SessionJob:
    raw_path: str
    spec: PipelineSpec
    rules: RulesConfig


def _run_session_job(job: _SessionJob) -> dict[str, Any]:
    path = Path(job.raw_path)
    if not path.is_file():
        return _empty_record(RAW_MISSING)
    collected = collect_session(read_raw_frames(path), job.spec, job.rules)
    return session_record(collected, job.spec, job.rules.closure.detector.enter_thr)


def signal_table(
    manifest: pd.DataFrame,
    raw_dir: Path,
    spec: PipelineSpec,
    rules: RulesConfig | None = None,
    *,
    workers: int = 1,
) -> pd.DataFrame:
    """목록의 모든 세션을 자기 기준선으로 재생해 세션마다 한 행인 표를 만든다. 열은 식별자 셋과 `session_record`의 값이다.

    추출 결과는 `extract.raw_parquet_path(raw_dir, rel_path)`에서 읽는다. 없는 세션은 `raw_status = missing`으로 남는다.
    workers는 세션 단위 병렬 프로세스 수이고 결과는 workers와 무관하다.
    가정: 추출 결과가 하나도 없으면 raw_dir이 틀린 것으로 보고 `ManifestError`를 낸다(`dataset.build_step_table`과 같다).
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    ids = label_free(manifest)
    rules = default_rules(spec) if rules is None else rules
    raw_paths = [raw_parquet_path(Path(raw_dir), Path(rel_path)) for rel_path in ids["rel_path"].tolist()]
    if not any(path.is_file() for path in raw_paths):
        raise ManifestError(
            f"none of the {len(raw_paths)} raw parquet files of the manifest exists below {raw_dir} "
            f"(first: {raw_paths[0] if raw_paths else '-'}); run the extraction or fix raw_dir"
        )
    jobs = [_SessionJob(str(path), spec, rules) for path in raw_paths]
    if workers == 1:
        records = [_run_session_job(job) for job in jobs]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            records = list(pool.map(_run_session_job, jobs))
    return pd.concat([ids, pd.DataFrame(records)], axis=1)


# ---------------------------------------------------------------------------
# 통과 기준
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Criterion:
    """통과 기준 하나의 판정. count는 조건을 만족한 세션 수, total은 전체 세션 수다."""

    name: str
    description: str
    count: int
    total: int
    ratio: float
    required_ratio: float
    passed: bool


def criterion_flags(table: pd.DataFrame, gate: SignalGate, spec: PipelineSpec) -> pd.DataFrame:
    """세션마다 세 기준의 조건을 만족하는지(`baseline_ok`, `blink_rate_ok`, `peak_closure_ok`)를 붙인 사본.

    - `baseline_ok`: 기준선이 60초에 대체 없이 완성됐다. status가 ok 또는 off_axis이고 T_b = `baseline.duration_ms`.
    - `blink_rate_ok`: `blink_rate_min_per_min` ≤ 분당 깜빡임 ≤ `blink_rate_max_per_min`.
    - `peak_closure_ok`: 깜빡임 정점 closure의 중앙값 ≥ `min_peak_closure_median`.
    읽는 열은 `baseline_status`, `t_b_ms`, `blink_rate_per_min`, `peak_closure_median`뿐이다.

    가정: off_axis(거치 각도가 15도를 넘음)는 대체가 아니므로 완성으로 센다. 연장된 기준선(T_b > 60초)은 세지 않는다.
    가정: 경계는 포함한다(정확히 5회, 40회, 0.8이면 만족).
    가정: 값이 없는 세션(추출 결과 없음, 보정이 끝나기 전에 끝남, 깜빡임 없음)은 해당 조건을 만족하지 못한 것이다.
    """
    out = table.copy()
    status = out["baseline_status"].astype(str)
    t_b_ms = out["t_b_ms"].to_numpy(dtype=np.int64)
    rate = out["blink_rate_per_min"].to_numpy(dtype=np.float64)
    peak = out["peak_closure_median"].to_numpy(dtype=np.float64)
    usable = status.isin(list(USABLE_BASELINE_STATUSES)).to_numpy(dtype=bool)
    out["baseline_ok"] = usable & (t_b_ms == spec.baseline.duration_ms)
    out["blink_rate_ok"] = (rate >= gate.blink_rate_min_per_min) & (rate <= gate.blink_rate_max_per_min)  # NaN은 False
    out["peak_closure_ok"] = peak >= gate.min_peak_closure_median
    return out


def evaluate_criteria(table: pd.DataFrame, gate: SignalGate, spec: PipelineSpec) -> tuple[Criterion, ...]:
    """PLAN 11장 E0의 통과 기준 셋을 판정한다. 분모는 표의 모든 행(세션)이다.

    기준마다 조건을 만족한 세션의 비율이 설정의 비율 이상이면 통과다(`criterion_flags`가 세션별 조건을 정한다).
    가정: 세션이 하나도 없으면 비율은 0이고 미달이다.
    """
    flags = criterion_flags(table, gate, spec)
    total = len(flags)
    seconds = spec.baseline.duration_ms // 1000
    plan = (
        (
            CRITERION_BASELINE,
            f"baseline complete at {seconds} s without fallback",
            "baseline_ok",
            gate.min_baseline_ok_ratio,
        ),
        (
            CRITERION_BLINK_RATE,
            f"{gate.blink_rate_min_per_min:g} to {gate.blink_rate_max_per_min:g} blinks per minute",
            "blink_rate_ok",
            gate.min_blink_rate_ok_ratio,
        ),
        (
            CRITERION_PEAK_CLOSURE,
            f"median blink peak closure >= {gate.min_peak_closure_median:g}",
            "peak_closure_ok",
            gate.min_peak_closure_ok_ratio,
        ),
    )
    criteria: list[Criterion] = []
    for name, description, column, required in plan:
        count = int(flags[column].to_numpy(dtype=bool).sum())
        ratio = count / total if total > 0 else 0.0
        criteria.append(Criterion(name, description, count, total, ratio, required, total > 0 and ratio >= required))
    return tuple(criteria)


# ---------------------------------------------------------------------------
# 전역 상수
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConstantMedians:
    """세션별 기준선 통계의 중앙값(반올림 전)과 거기에 들어간 세션 수."""

    e: float
    b: float
    jaw0: float
    n_sessions: int


def constant_medians(quality: pd.DataFrame, subjects: Iterable[int] | None = None) -> ConstantMedians:
    """세션별 기준선 통계(E, B, jaw0)의 중앙값 (PLAN 4.3절: "전체 영상의 영상별 통계 중앙값").

    quality는 세션마다 한 행인 표다. `dataset.StepTable.quality`와 이 모듈의 세션 표(per_video.csv) 둘 다 받는다.
    읽는 열은 `subject`, `baseline_status`, `base_e_l`, `base_e_r`, `base_b_l`, `base_b_r`, `base_jaw0`뿐이다
    (라벨·클래스·fold 열이 있어도 읽지 않는다). subjects를 주면 그 피험자의 세션만 쓴다.
    쓸 세션이 없으면 ValueError다.

    가정: 세션별 통계는 E = (E_l + E_r) / 2, B = (B_l + B_r) / 2, jaw0이다(전역 상수는 두 눈에 같은 값을 쓴다).
    가정: 개인 기준선이 완성된 세션(status ok 또는 off_axis. 연장된 것 포함)만 쓴다. 전역 상수로 대체된 세션의 값은
      전역 상수 자신이라 넣으면 순환이 되고, 보정이 끝나지 않은 세션에는 값이 없다.
    가정: 중앙값은 PLAN 4.2절의 정의다(짝수 개면 가운데 두 값의 평균).
    """
    needed = ("subject", "baseline_status") + BASELINE_COLUMNS
    missing = [name for name in needed if name not in quality.columns]
    if missing:
        raise ValueError(f"quality table lacks columns {missing}")
    rows = quality[list(needed)]
    keep = rows["baseline_status"].astype(str).isin(list(USABLE_BASELINE_STATUSES)).to_numpy(dtype=bool)
    if subjects is not None:
        wanted = np.array(sorted({int(value) for value in subjects}), dtype=np.int64)
        keep = keep & np.isin(rows["subject"].to_numpy(dtype=np.int64), wanted)  # pandas가 준 배열은 읽기 전용이다
    values = rows.loc[keep, list(BASELINE_COLUMNS)].to_numpy(dtype=np.float64)
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) == 0:
        raise ValueError("no session with a completed personal baseline (status ok or off_axis) to take medians from")
    e = ((values[:, 0] + values[:, 1]) / 2.0).tolist()
    b = ((values[:, 2] + values[:, 3]) / 2.0).tolist()
    jaw0 = values[:, 4].tolist()
    return ConstantMedians(
        e=calibrate.median(e), b=calibrate.median(b), jaw0=calibrate.median(jaw0), n_sessions=len(values)
    )


def _round_constant(value: float) -> float:
    scale = 10.0**CONSTANT_DECIMALS
    return math.floor(value * scale + 0.5) / scale


def global_constants(quality: pd.DataFrame, spec: PipelineSpec, subjects: Iterable[int] | None = None) -> Baseline:
    """세션별 기준선 통계의 중앙값으로 만든 전역 상수 기준선 (PLAN 4.3절).

    quality와 subjects는 `constant_medians`와 같다. subjects를 주지 않으면 표의 모든 세션을 쓴다(E0의 제안 상수.
    시험 피험자의 영상도 들어간다). 학습 피험자만 주면 fold별 상수다(전역 조건의 민감도 점검, PLAN 4.3절).
    돌려주는 값은 `calibrate.global_baseline(spec)`과 같은 모양이다: 두 눈에 같은 E와 B, R0·gaze_h0·gaze_v0은 spec의
    전역 상수 그대로, `t_complete_ms` = 0, status = fallback_global. `dataset.build_step_table`과 `load_step_tables`의
    `global_override`에 그대로 넘길 수 있다.

    가정: 상수는 소수 넷째 자리로 반올림한다(`floor(x·10⁴ + 0.5) / 10⁴`). 설정 파일과 SPEC.md의 표에 같은 값을
      그대로 적기 위해서다. 세션 사이의 차이에 비하면 무시할 크기다. 반올림 전 값은 `constant_medians`가 준다.
    """
    medians = constant_medians(quality, subjects)
    constants = spec.global_constants
    e = _round_constant(medians.e)
    b = _round_constant(medians.b)
    return Baseline(
        e_l=e,
        e_r=e,
        b_l=b,
        b_r=b,
        r0=constants.r0,
        jaw0=_round_constant(medians.jaw0),
        gaze_h0=constants.gaze_h0,
        gaze_v0=constants.gaze_v0,
        t_complete_ms=0,
        status=BASELINE_FALLBACK_GLOBAL,
    )


# ---------------------------------------------------------------------------
# 덮어쓰기와 제안 spec
# ---------------------------------------------------------------------------


def parse_overrides(items: Sequence[str]) -> dict[str, Any]:
    """`--override`의 `경로=값` 목록을 `spec.apply_overrides`가 받는 dict로 바꾼다.

    경로는 `pipeline` 블록 안의 점으로 이은 경로(`closure.rho`)이고 값은 JSON으로 읽는다(`0.30`, `60`, `true`,
    `[0.5, 0.9]`). JSON이 아닌 값은 문자열로 둔다. 실수 자리에는 소수점을 붙여 적어야 한다(spec 로더가 정수를 거부한다).
    `=`가 없거나 같은 경로가 두 번 나오면 ValueError다.
    """
    overrides: dict[str, Any] = {}
    for item in items:
        path, separator, text = item.partition("=")
        path = path.strip()
        if separator == "" or path == "" or text.strip() == "":
            raise ValueError(f"--override expects PATH=VALUE (for example closure.rho=0.30), got {item!r}")
        if path in overrides:
            raise ValueError(f"--override names {path!r} more than once")
        try:
            overrides[path] = json.loads(text)
        except json.JSONDecodeError:
            overrides[path] = text.strip()
    return overrides


def non_provisional_paths(overrides: Mapping[str, Any]) -> list[str]:
    """덮어쓴 경로 가운데 계획서가 잠정으로 표시하지 않은 것(`PROVISIONAL_PATHS`에 없는 것)."""
    return [path for path in overrides if not any(path.startswith(prefix) for prefix in PROVISIONAL_PATHS)]


def _set_path(block: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node: Any = block
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            raise SpecError(f"override path not found: {dotted!r}")
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        raise SpecError(f"override path not found: {dotted!r}")
    node[parts[-1]] = value


def build_proposed_spec(
    current: Mapping[str, Any], overrides: Mapping[str, Any], constants: Baseline | None
) -> dict[str, Any]:
    """제안 feature_spec의 dict: 지금 파일의 내용에 덮어쓴 값과 제안 전역 상수를 넣은 것.

    current는 `spec.read_feature_spec_dict`로 읽은 지금 파일이고 고치지 않는다. 바뀌는 것은 `pipeline` 블록의 해당 값,
    `spec_version`, `pipeline_hash`뿐이다(특징 목록과 표준화 상수, 키 순서는 그대로다). 결과는 spec 로더의 검증을
    통과한 것이다(아니면 SpecError). constants가 None이면 전역 상수를 바꾸지 않는다.

    가정: `global_constants.e`, `.b`, `.jaw0`을 덮어썼어도 제안 파일에는 constants(세션별 통계의 중앙값)가 들어간다.
      덮어쓴 전역 상수는 평가에만 쓰인다.
    가정: `spec_version`은 `pipeline` 블록이 바뀔 때만 1 올린다. 바뀐 것이 없으면 그대로다.
    """
    data = copy.deepcopy(dict(current))
    pipeline = data["pipeline"]
    changes = dict(overrides)
    if constants is not None:
        changes["global_constants.e"] = constants.e_l
        changes["global_constants.b"] = constants.b_l
        changes["global_constants.jaw0"] = constants.jaw0
    for dotted, value in changes.items():
        _set_path(pipeline, dotted, value)
    new_hash = compute_pipeline_hash(pipeline)
    if new_hash != compute_pipeline_hash(current["pipeline"]):
        data["spec_version"] = int(current["spec_version"]) + 1
    data["pipeline_hash"] = new_hash
    feature_spec_from_dict(data)
    return data


# ---------------------------------------------------------------------------
# E0 리포트
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class E0Result:
    """`run_e0`의 결과.

    - run_dir: 리포트 폴더. table: 세션 표(per_video.csv와 같은 내용). criteria, passed: 통과 기준 판정.
    - constants: 제안 전역 상수(쓸 세션이 없으면 None). proposed_spec_path: 쓴 제안 파일(쓰지 않았으면 None).
    - rotation: 추출 결과의 회전 점검 기록을 모은 것(`dataset.read_rotation_checks`). 확인이 필요한 영상이 들어 있다.
    """

    run_dir: Path
    table: pd.DataFrame
    criteria: tuple[Criterion, ...]
    passed: bool
    synthetic: bool
    constants: Baseline | None
    proposed_spec_path: Path | None
    rotation: RotationSummary


def data_note(synthetic: bool) -> str:
    """E0 출력의 첫 줄: 합성 데이터이면 `dataset.SYNTHETIC_NOTE`, 아니면 `E0_REAL_NOTE`."""
    return SYNTHETIC_NOTE if synthetic else E0_REAL_NOTE


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _fmt(value: float, digits: int = 3) -> str:
    """표에 적는 실수. 값이 없으면(NaN, 무한대) `-`이고, 반올림해서 0이 되는 음수는 부호 없이 적는다."""
    if not math.isfinite(float(value)):
        return "-"
    text = f"{value:.{digits}f}"
    return text[1:] if text.startswith("-") and float(text) == 0.0 else text


def _md_table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines


def _listed(items: Sequence[str]) -> str:
    if len(items) == 0:
        return "없음"
    shown = ", ".join(f"`{item}`" for item in items[:_MAX_LISTED])
    return shown if len(items) <= _MAX_LISTED else f"{shown} 외 {len(items) - _MAX_LISTED}개"


def _distribution_row(name: str, values: np.ndarray, digits: int) -> list[str]:
    finite = [float(value) for value in values.tolist() if math.isfinite(float(value))]
    if len(finite) == 0:
        return [name, "0", "-", "-", "-", "-", "-"]
    q25, median, q75 = _quartiles(finite)
    return [
        name,
        str(len(finite)),
        _fmt(min(finite), digits),
        _fmt(q25, digits),
        _fmt(median, digits),
        _fmt(q75, digits),
        _fmt(max(finite), digits),
    ]


def follow_up_steps(
    proposed_path: Path, spec_path: Path, config_arg: str, new_hash: str, changed: Sequence[str]
) -> list[str]:
    """제안 파일을 받아들인 뒤 소유자가 차례로 할 일(명령과 손으로 고칠 곳). 이 도구는 아무것도 대신 하지 않는다."""
    changed_text = ", ".join(changed)
    return [
        f"cp {proposed_path.as_posix()} {spec_path.as_posix()}",
        f"python -m drowsy.spec check --spec {spec_path.as_posix()}  # expect: pipeline_hash OK: {new_hash}",
        f"edit tests/test_spec.py: set PIPELINE_HASH_V0 to {new_hash} and update the assertions that pin the changed "
        f"constants ({changed_text})",
        f"python -m drowsy.export_onnx --config {config_arg} --random-init --kind tcn --version 0  "
        "# re-export v0: its metadata and spec copy carry pipeline_hash and feature_spec_hash",
        "python -m drowsy.vectors generate --out tests/vectors",
        "python -m drowsy.vectors verify --dir tests/vectors",
        "edit docs/SPEC.md (the check:hashes and check:pipeline tables, inline quotes of the changed constants, "
        "section 17 and section 19) and the hash table of export/README.md",
        "python -m pytest -q",
        f"python -m drowsy.dataset --config {config_arg}  "
        "# the step-table cache is keyed by pipeline_hash and is rebuilt",
        f"tell the app team the new pipeline_hash {new_hash} (PLAN section 11, E0)",
    ]


def run_e0(
    cfg: ExperimentConfig,
    *,
    overrides: Mapping[str, Any] | None = None,
    run_name: str = DEFAULT_RUN_NAME,
    fps_sim: bool = False,
    workers: int | None = None,
    command: str = "",
    config_path: Path | None = None,
    source: DatasetSource | None = None,
) -> E0Result:
    """E0 신호 점검을 돌리고 `{reports_dir}/{run_name}/e0/`(fps_sim이면 `e0_fps_sim/`)에 결과를 쓴다 (PLAN 11장 E0).

    - overrides: `pipeline` 블록의 경로 → 값(`parse_overrides`). 지금 spec에 이 값을 덮어쓴 상수로 재생하고 판정한다.
    - workers: 세션 단위 병렬 프로세스 수. None이면 `extraction.workers`.
    - command, config_path: 리포트에 남길 명령과 설정 사본(PLAN 3.4절). config_path는 명령에 적은 경로 그대로다.
    - source: 데이터. None이면 `dataset.open_dataset(cfg, fps_sim)`.
    run_name이 폴더 이름 하나가 아니면 ValueError다(`config.check_run_name`. 리포트 폴더 밖에 쓰지 않게 한다).

    쓰는 파일(모두 첫 줄 또는 첫 키가 데이터 종류다. 제안 파일은 예외이고 합성 데이터이면 파일 이름에 적는다)
    - summary.md: 통과 기준 판정, 기준선 상태, 세션별 값의 분포, 전역 상수 제안, 받아들인 뒤 할 일, 회전 점검
      (자동 점검이 각도를 가르지 못했는데 수동 회전 보정이 없는 영상의 목록. `dataset.rotation_check_section`).
    - per_video.csv: 세션 표(`signal_table` + `criterion_flags`). 첫 줄은 `#`으로 시작하는 데이터 종류다.
    - criteria.json: 기준별 수치와 판정, 덮어쓴 값, 제안 상수, 받아들인 뒤 할 일, 회전 점검 기록(`rotation_check`).
    - feature_spec_proposed.json(합성 데이터이면 feature_spec_proposed.synthetic.json): `build_proposed_spec`의 결과.
      fps_sim 실행에서는 쓰지 않는다. `config/feature_spec.json`은 어느 경우에도 고치지 않는다.
    - config.yaml: 설정 사본(config_path를 줬을 때).
    실행 시간은 파일에 적지 않는다(같은 입력이면 파일이 바이트 단위로 같다).

    가정: 통과 기준은 "지금 spec + 덮어쓴 값"으로 판정한다. 제안 파일은 여기에 전역 상수까지 바꾼 것이다. 전역 상수는
      기준선 완성 전의 임시 closure와 전역 상수로 대체된 세션에만 쓰이므로, 대체된 세션이 없으면 판정이 같다.
      대체된 세션이 있으면 제안 상수를 `--override`로 넣어 한 번 더 돌려 확인하라고 요약에 적는다.
    가정: 회전이 확인되지 않은 영상이 있어도 통과 기준의 판정과 종료 코드는 바꾸지 않는다(그 영상을 빼지도 않는다).
      리포트와 출력으로 알리기만 한다. 영상을 볼지, 더 엄격한 점검을 넣을지는 소유자의 결정을 기다리는 항목이다.
      회전 점검 기록은 라벨과 무관하다: 목록의 `rel_path`와 추출 결과 파일, 설정의 수동 회전 보정표만 읽는다.
    """
    run_name = check_run_name(run_name)
    overrides = dict(overrides) if overrides else {}
    current = read_feature_spec_dict(cfg.paths.feature_spec)
    base_spec = feature_spec_from_dict(current)
    spec = apply_overrides(base_spec.pipeline, overrides) if overrides else base_spec.pipeline
    if source is None:
        source = open_dataset(cfg, fps_sim=fps_sim)
    rules = dataclasses.replace(load_rules(cfg.paths.rules), clock=spec.clock)
    gate = cfg.evaluation.signal_gate

    table = signal_table(
        source.manifest, source.raw_dir, spec, rules, workers=cfg.extraction.workers if workers is None else workers
    )
    table = criterion_flags(table, gate, spec)
    criteria = evaluate_criteria(table, gate, spec)
    passed = all(criterion.passed for criterion in criteria)
    rotation = dataset_rotation_checks(cfg, source, table)  # 세션 표의 rel_path만 쓴다(라벨 열은 이미 없다)

    medians: ConstantMedians | None = None
    constants: Baseline | None = None
    try:
        medians = constant_medians(table)
        constants = global_constants(table, spec)
    except ValueError:
        pass  # 개인 기준선이 완성된 세션이 없다. 전역 상수를 제안하지 않는다.

    note = data_note(source.synthetic)
    run_dir = cfg.paths.reports_dir / run_name / (E0_FPS_SIM_SUBDIR if fps_sim else E0_SUBDIR)
    run_dir.mkdir(parents=True, exist_ok=True)
    for stale in (PROPOSED_SPEC_FILE, SYNTHETIC_PROPOSED_SPEC_FILE):  # 앞 실행의 제안 파일이 남아 섞이지 않게 한다
        (run_dir / stale).unlink(missing_ok=True)

    proposed: dict[str, Any] | None = None
    proposed_path: Path | None = None
    proposed_file_hash: str | None = None
    proposal_error: str | None = None
    steps: list[str] = []
    if not fps_sim:
        try:
            proposed = build_proposed_spec(current, overrides, constants)
        except SpecError as exc:
            proposal_error = str(exc)
    if proposed is not None:
        proposed_path = run_dir / (SYNTHETIC_PROPOSED_SPEC_FILE if source.synthetic else PROPOSED_SPEC_FILE)
        proposed_file_hash = write_feature_spec(proposed, proposed_path)
        changed = set(overrides)
        if constants is not None:
            changed |= {"global_constants.e", "global_constants.b", "global_constants.jaw0"}
        if not source.synthetic and proposed["pipeline_hash"] != current["pipeline_hash"]:
            steps = follow_up_steps(
                proposed_path,
                cfg.paths.feature_spec,
                config_path.as_posix() if config_path is not None else "config/default.yaml",
                proposed["pipeline_hash"],
                sorted(changed),
            )

    with (run_dir / PER_VIDEO_FILE).open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# {note}\n")
        table.to_csv(handle, index=False, lineterminator="\n")
    if config_path is not None:
        shutil.copyfile(config_path, run_dir / CONFIG_COPY_FILE)

    status = table["baseline_status"].astype(str)
    extended = (status.isin(list(USABLE_BASELINE_STATUSES))) & (table["t_b_ms"] > spec.baseline.duration_ms)
    fallback = status == BASELINE_FALLBACK_GLOBAL
    unfinished = (status == BASELINE_NONE) & (table["raw_status"] == RAW_OK)
    missing = table["raw_status"] == RAW_MISSING
    n_blinks = int(table["n_blinks"].sum())
    current_constants = base_spec.pipeline.global_constants
    evaluated_hash = spec_pipeline_hash(spec)
    payload: dict[str, Any] = {
        "data_note": note,
        "synthetic": source.synthetic,
        "description": source.description,
        "command": command,
        "label_free": True,
        "manifest_columns_read": list(LABEL_FREE_COLUMNS),
        "unit": "session (one file; the parts of a split video count separately)",
        "fps_sim": fps_sim,
        "feature_spec": cfg.paths.feature_spec.as_posix(),
        "spec_version": base_spec.spec_version,
        "pipeline_hash_current": base_spec.pipeline_hash,
        "pipeline_hash_evaluated": evaluated_hash,
        "overrides": overrides,
        "overrides_not_marked_provisional": non_provisional_paths(overrides),
        "n_sessions": len(table),
        "n_subjects": int(table["subject"].nunique()),
        "passed": passed,
        "criteria": [
            {
                "name": criterion.name,
                "description": criterion.description,
                "count": criterion.count,
                "total": criterion.total,
                "ratio": criterion.ratio,
                "required_ratio": criterion.required_ratio,
                "passed": criterion.passed,
            }
            for criterion in criteria
        ],
        "signal_gate": dataclasses.asdict(gate),
        "baseline": {
            "ok": int((status == BASELINE_OK).sum()),
            "off_axis": int((status == BASELINE_OFF_AXIS).sum()),
            "fallback_global": int(fallback.sum()),
            "not_completed": int(unfinished.sum()),
            "raw_missing": int(missing.sum()),
            "extended": int(extended.sum()),
            "extended_sessions": table.loc[extended, "session_id"].tolist(),
            "fallback_sessions": table.loc[fallback, "session_id"].tolist(),
            "not_completed_sessions": table.loc[unfinished, "session_id"].tolist(),
            "raw_missing_sessions": table.loc[missing, "session_id"].tolist(),
        },
        "rotation_check": {"overrides_key": OVERRIDES_KEY, **rotation.as_dict()},
        "blinks": {
            "total": n_blinks,
            "peak_ge_perclos_thr": int(table["n_peak_ge_perclos_thr"].sum()),
            "perclos_thr": spec.step_features.perclos_thr,
            "peak_ge_rule_enter": int(table["n_peak_ge_rule_enter"].sum()),
            "rule_enter_thr": rules.closure.detector.enter_thr,
        },
        "global_constants": {
            "current": {"e": current_constants.e, "b": current_constants.b, "jaw0": current_constants.jaw0},
            "evaluated": {
                "e": spec.global_constants.e,
                "b": spec.global_constants.b,
                "jaw0": spec.global_constants.jaw0,
            },
            "proposed": None
            if constants is None
            else {"e": constants.e_l, "b": constants.b_l, "jaw0": constants.jaw0},
            "median_before_rounding": None
            if medians is None
            else {"e": medians.e, "b": medians.b, "jaw0": medians.jaw0},
            "decimals": CONSTANT_DECIMALS,
            "n_sessions_used": 0 if medians is None else medians.n_sessions,
            "includes_test_subjects": True,
        },
        "proposed_spec": None
        if proposed is None or proposed_path is None
        else {
            "file": proposed_path.name,
            "spec_version": proposed["spec_version"],
            "pipeline_hash": proposed["pipeline_hash"],
            "feature_spec_hash": proposed_file_hash,
        },
        "proposal_error": proposal_error,
        "follow_up": steps,
    }
    _write_json(payload, run_dir / CRITERIA_FILE)

    lines = _e0_summary(
        note=note,
        description=source.description,
        command=command,
        cfg=cfg,
        spec=spec,
        base_spec=base_spec,
        overrides=overrides,
        table=table,
        criteria=criteria,
        passed=passed,
        synthetic=source.synthetic,
        fps_sim=fps_sim,
        medians=medians,
        constants=constants,
        proposed=proposed,
        proposed_path=proposed_path,
        proposal_error=proposal_error,
        steps=steps,
        rule_enter_thr=rules.closure.detector.enter_thr,
        config_arg=config_path.as_posix() if config_path is not None else "config/default.yaml",
        run_name=run_name,
        rotation=rotation,
    )
    (run_dir / SUMMARY_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return E0Result(
        run_dir=run_dir,
        table=table,
        criteria=criteria,
        passed=passed,
        synthetic=source.synthetic,
        constants=constants,
        proposed_spec_path=proposed_path,
        rotation=rotation,
    )


def _e0_summary(
    *,
    note: str,
    description: str,
    command: str,
    cfg: ExperimentConfig,
    spec: PipelineSpec,
    base_spec: FeatureSpec,
    overrides: Mapping[str, Any],
    table: pd.DataFrame,
    criteria: Sequence[Criterion],
    passed: bool,
    synthetic: bool,
    fps_sim: bool,
    medians: ConstantMedians | None,
    constants: Baseline | None,
    proposed: Mapping[str, Any] | None,
    proposed_path: Path | None,
    proposal_error: str | None,
    steps: Sequence[str],
    rule_enter_thr: float,
    config_arg: str,
    run_name: str,
    rotation: RotationSummary,
) -> list[str]:
    """summary.md의 줄 목록."""
    gate = cfg.evaluation.signal_gate
    base_hash = base_spec.pipeline_hash
    base_version = base_spec.spec_version
    status = table["baseline_status"].astype(str)
    usable = status.isin(list(USABLE_BASELINE_STATUSES))
    extended = usable & (table["t_b_ms"] > spec.baseline.duration_ms)
    fallback = status == BASELINE_FALLBACK_GLOBAL
    unfinished = (status == BASELINE_NONE) & (table["raw_status"] == RAW_OK)
    missing = table["raw_status"] == RAW_MISSING
    seconds = spec.baseline.duration_ms // 1000
    override_text = ", ".join(f"`{path}` = {json.dumps(value)}" for path, value in overrides.items()) or "없음"
    evaluated_hash = spec_pipeline_hash(spec)

    lines = [
        f"# {note}",
        "",
        "E0 신호 점검 (PLAN 11장 E0, 4.2절, 4.3절). 라벨을 쓰지 않는다: 목록에서 "
        + ", ".join(f"`{name}`" for name in LABEL_FREE_COLUMNS)
        + " 열만 읽고 라벨·클래스·fold 열은 읽지 않는다.",
        "",
        description,
        "",
        f"- 생성 명령: `{command}`",
        f"- 판정에 쓴 상수: `{cfg.paths.feature_spec.as_posix()}`(spec_version {base_version}, pipeline_hash "
        f"`{base_hash}`)에 덮어쓴 값: {override_text}. 판정에 쓴 pipeline_hash는 `{evaluated_hash}`다",
        f"- 단위: 세션(파일) {len(table)}개, 피험자 {table['subject'].nunique()}명. 나뉜 영상의 부분은 각자 자기 "
        "기준선으로 따로 재생하므로 따로 센다(가정)",
        "- 재생: 세션마다 저장된 프레임을 `DriverPipeline`에 넣어 앱처럼 자기 앞부분으로 보정했다. 깜빡임은 특징용 "
        f"검출기(진입 {spec.run_detector.enter_thr:g} / 이탈 {spec.run_detector.exit_thr:g})가 기준선 완성 뒤에 낸, "
        "잘리지 않은 감김 구간이다",
    ]
    if fps_sim:
        lines.append(
            "- 이 실행은 기기 프레임 누락을 모사한 추출 결과의 점검이다. 전역 상수 제안과 제안 파일은 원본 프레임률 "
            "추출의 실행에서만 쓴다"
        )
    if rotation.flagged:
        lines += [
            "",
            f"**주의: 회전이 확인되지 않은 영상이 {len(rotation.flagged)}개 있다.** 추출의 회전 자동 점검이 각도를 가르지 "
            "못했는데 수동 회전 보정이 없거나, 보정표의 각도가 추출에 적용되지 않았다. 그 영상의 신호는 틀린 각도에서 "
            "나왔을 수 있다. 목록은 맨 아래 \"회전 점검\" 절에 있다. 상수를 확정하기 전에 영상을 직접 확인한다.",
        ]
    flagged = non_provisional_paths(overrides)
    if flagged:
        lines += [
            "",
            "**주의: 덮어쓴 값 가운데 계획서가 잠정으로 표시하지 않은 상수가 있다: "
            + ", ".join(f"`{path}`" for path in flagged)
            + ".** E0에서는 잠정 상수만 바꾼다(PLAN 4.6절). 이 값을 바꾸려면 사전에 확인을 받는다.",
        ]
    if base_version >= 1:
        lines += [
            "",
            f"**주의: 지금 spec_version이 {base_version}이다.** E0의 상수 확정은 한 번만 한다(PLAN 4.6절, 11장). "
            "다시 바꾸려면 사전에 확인을 받는다.",
        ]

    korean = {
        CRITERION_BASELINE: f"기준선이 {seconds}초에 대체 없이 완성",
        CRITERION_BLINK_RATE: f"깜빡임 분당 {gate.blink_rate_min_per_min:g}~{gate.blink_rate_max_per_min:g}회",
        CRITERION_PEAK_CLOSURE: f"깜빡임 정점 closure의 중앙값 {gate.min_peak_closure_median:g} 이상",
    }
    lines += ["", "## 통과 기준", ""]
    lines += _md_table(
        ["기준", "만족한 세션", "비율", "필요한 비율", "판정"],
        [
            [
                f"`{criterion.name}`: {korean[criterion.name]}",
                f"{criterion.count} / {criterion.total}",
                f"{100.0 * criterion.ratio:.1f}%",
                f"{100.0 * criterion.required_ratio:g}% 이상",
                "통과" if criterion.passed else "미달",
            ]
            for criterion in criteria
        ],
    )
    verdict = "통과" if passed else "미달"
    if synthetic:
        verdict += " (합성 데이터다. 파이프라인이 도는지만 보여 주며 실제 눈에 대한 판정이 아니다)"
    lines += [
        "",
        f"종합: **{verdict}**",
        "",
        f"- `{CRITERION_BASELINE}`: 기준선 status가 ok 또는 off_axis이고 T_b = {seconds}초. 연장된 기준선과 전역 상수 "
        "대체는 만족하지 못한 것이다",
        f"- `{CRITERION_BLINK_RATE}`: 깜빡임 수 × 60000 / 특징 구간의 closure 유효 시간(ms)이 "
        f"{gate.blink_rate_min_per_min:g} 이상 {gate.blink_rate_max_per_min:g} 이하",
        f"- `{CRITERION_PEAK_CLOSURE}`: 깜빡임 정점 closure(`ClosureRun.amplitude`)의 세션 중앙값이 "
        f"{gate.min_peak_closure_median:g} 이상",
        "- 분모는 모든 세션이다. 추출 결과가 없거나 보정이 끝나기 전에 끝난 세션, 깜빡임이 없는 세션은 해당 기준을 "
        "만족하지 못한 것으로 센다",
        "- 미달이면 잠정 상수를 `--override`로 바꿔 다시 판정해 볼 수 있다. 식을 바꿔야 하면 spec v2이고 사전 확인이 "
        "필요하다(PLAN 11장)",
    ]

    lines += ["", "## 기준선", ""]
    lines += _md_table(
        ["상태", "세션 수", "세션"],
        [
            [f"ok, T_b = {seconds}초", int(((status == BASELINE_OK) & ~extended).sum()), ""],
            [
                f"off_axis(거치 각도 {spec.baseline.off_axis_yaw_deg:g}도 초과), T_b = {seconds}초",
                int(((status == BASELINE_OFF_AXIS) & ~extended).sum()),
                _listed(table.loc[(status == BASELINE_OFF_AXIS) & ~extended, "session_id"].tolist()),
            ],
            ["연장됨(T_b > 기본 시간)", int(extended.sum()), _listed(table.loc[extended, "session_id"].tolist())],
            [
                "전역 상수로 대체(fallback_global)",
                int(fallback.sum()),
                _listed(table.loc[fallback, "session_id"].tolist()),
            ],
            ["보정이 끝나기 전에 끝남", int(unfinished.sum()), _listed(table.loc[unfinished, "session_id"].tolist())],
            ["추출 결과 없음", int(missing.sum()), _listed(table.loc[missing, "session_id"].tolist())],
        ],
    )

    lines += [
        "",
        "## 세션별 값의 분포",
        "",
        "분위수는 최근접 순위, 중앙값은 PLAN 4.2절의 정의다. n은 값이 있는 세션 수다. 기준선 통계(E, B, jaw0, 거치 각도)는 "
        "개인 기준선이 완성된 세션만 넣었다.",
        "",
    ]
    personal = table.loc[usable]
    e_mean = ((personal["base_e_l"] + personal["base_e_r"]) / 2.0).to_numpy(dtype=np.float64)
    b_mean = ((personal["base_b_l"] + personal["base_b_r"]) / 2.0).to_numpy(dtype=np.float64)
    lines += _md_table(
        ["값", "n", "최소", "25%", "중앙값", "75%", "최대"],
        [
            _distribution_row("E = (E_l + E_r) / 2", e_mean, 4),
            _distribution_row("B = (B_l + B_r) / 2", b_mean, 4),
            _distribution_row("jaw0", personal["base_jaw0"].to_numpy(dtype=np.float64), 4),
            _distribution_row("abs(yaw0) (도)", np.abs(personal["base_yaw0"].to_numpy(dtype=np.float64)), 1),
            _distribution_row("pitch0 (도)", personal["base_pitch0"].to_numpy(dtype=np.float64), 1),
            _distribution_row("얼굴 검출 비율(시간 기준, 영상 전체)", table["face_ratio"].to_numpy(dtype=np.float64), 3),
            _distribution_row(
                "closure 유효 시간 비율(특징 구간)", table["closure_valid_ratio"].to_numpy(dtype=np.float64), 3
            ),
            _distribution_row("분당 깜빡임", table["blink_rate_per_min"].to_numpy(dtype=np.float64), 2),
            _distribution_row("깜빡임 정점 closure의 중앙값", table["peak_closure_median"].to_numpy(dtype=np.float64), 3),
            _distribution_row(
                "깜빡임 지속 시간의 중앙값 (ms)", table["blink_dur_median_ms"].to_numpy(dtype=np.float64), 1
            ),
        ],
    )
    n_blinks = int(table["n_blinks"].sum())
    if n_blinks > 0:
        ge_perclos = int(table["n_peak_ge_perclos_thr"].sum())
        ge_rule = int(table["n_peak_ge_rule_enter"].sum())
        lines += [
            "",
            f"모든 세션의 깜빡임 {n_blinks}개 가운데 정점이 `step_features.perclos_thr`({spec.step_features.perclos_thr:g}) "
            f"이상인 것은 {ge_perclos}개({100.0 * ge_perclos / n_blinks:.1f}%), 규칙용 검출기의 진입 임계"
            f"({rule_enter_thr:g}, `config/rules.yaml`) 이상인 것은 {ge_rule}개({100.0 * ge_rule / n_blinks:.1f}%)다. "
            "규칙 설정의 값은 `--override`로 바꿀 수 없다(이 도구는 `pipeline` 블록만 다룬다).",
        ]

    lines += ["", "## 전역 상수(참고)" if fps_sim else "## 전역 상수 제안", ""]
    current = base_spec.pipeline.global_constants
    if constants is None or medians is None:
        lines.append("개인 기준선이 완성된 세션이 없어 전역 상수를 제안하지 못했다. 지금 값을 그대로 둔다.")
    else:
        lines += _md_table(
            ["상수", "지금 파일의 값", "이 추출의 값(참고)" if fps_sim else "제안 값", "반올림 전 중앙값"],
            [
                ["`global_constants.e`", f"{current.e:g}", f"{constants.e_l:g}", f"{medians.e:.6f}"],
                ["`global_constants.b`", f"{current.b:g}", f"{constants.b_l:g}", f"{medians.b:.6f}"],
                ["`global_constants.jaw0`", f"{current.jaw0:g}", f"{constants.jaw0:g}", f"{medians.jaw0:.6f}"],
            ],
        )
        lines += [
            "",
            f"- 이 값은 개인 기준선이 완성된 세션 {medians.n_sessions}개의 세션별 통계(E와 B는 두 눈의 평균)의 중앙값을 "
            f"소수 {CONSTANT_DECIMALS}자리로 반올림한 것이다(PLAN 4.3절)",
            "- **이 상수는 라벨을 쓰지 않지만 시험 피험자의 영상도 들어간 값이다**(PLAN 4.3절, 6장). 전역 정규화 조건의 "
            "민감도 점검은 바깥 fold마다 학습 피험자만으로 다시 계산한 상수로 한다"
            "(`signal_check.global_constants(quality, spec, subjects=학습 피험자)`)",
        ]
        n_fallback = int(fallback.sum())
        if n_fallback > 0 and not fps_sim:
            confirm = " ".join(
                [
                    f"python -m drowsy.signal_check e0 --config {config_arg} --run-name {run_name}_confirm --override",
                    *[
                        f"{path}={json.dumps(value)}"
                        for path, value in overrides.items()
                        if not path.startswith("global_constants.")
                    ],
                    f"global_constants.e={constants.e_l!r}",
                    f"global_constants.b={constants.b_l!r}",
                    f"global_constants.jaw0={constants.jaw0!r}",
                ]
            )
            lines.append(
                f"- 전역 상수로 대체된 세션이 {n_fallback}개 있다. 그 세션의 깜빡임 통계는 판정에 쓴 전역 상수로 계산한 "
                f"것이다. 제안 상수에서의 판정은 `{confirm}`로 확인한다"
            )

    lines += ["", "## 제안 파일", ""]
    if fps_sim:
        lines.append("프레임 누락 모사 추출의 점검에서는 제안 파일을 쓰지 않는다.")
    elif proposed is None or proposed_path is None:
        lines.append(f"제안 파일을 쓰지 못했다. 덮어쓴 값과 제안 상수가 spec 검증을 통과하지 못한다: {proposal_error}")
    else:
        lines.append(
            f"`{proposed_path.name}`: 지금 파일에 덮어쓴 값과 제안 전역 상수를 넣은 것이다. spec_version "
            f"{base_version} → {proposed['spec_version']}, pipeline_hash `{proposed['pipeline_hash']}`. "
            f"**이 도구는 `{cfg.paths.feature_spec.as_posix()}`을 고치지 않는다.**"
        )
        if synthetic:
            lines += [
                "",
                "합성 세션에서 나온 상수다. 파일 쓰기가 도는지만 보여 준다. **`config/feature_spec.json`에 쓰지 않는다.**",
            ]
        elif len(steps) == 0:
            lines += ["", "지금 파일과 `pipeline` 블록이 같다. 바꿀 것이 없다."]
        else:
            lines += ["", "받아들인 뒤 차례로 할 일(이 도구는 대신 하지 않는다):", ""]
            for index, step in enumerate(steps, start=1):
                is_command = step.startswith(("cp ", "python "))
                lines.append(f"{index}. `{step}`" if is_command else f"{index}. {step}")

    failing = table.loc[~(table["baseline_ok"] & table["blink_rate_ok"] & table["peak_closure_ok"])]
    lines += ["", "## 조건을 만족하지 못한 세션", ""]
    if len(failing) == 0:
        lines.append("없음.")
    else:
        shown = failing.head(_MAX_LISTED)
        lines += _md_table(
            ["세션", "기준선", "T_b (ms)", "분당 깜빡임", "정점 closure 중앙값", "만족하지 못한 기준"],
            [
                [
                    f"`{row.session_id}`",
                    row.baseline_status,
                    row.t_b_ms,
                    _fmt(row.blink_rate_per_min, 2),
                    _fmt(row.peak_closure_median, 3),
                    ", ".join(
                        name
                        for name, ok in (
                            (CRITERION_BASELINE, row.baseline_ok),
                            (CRITERION_BLINK_RATE, row.blink_rate_ok),
                            (CRITERION_PEAK_CLOSURE, row.peak_closure_ok),
                        )
                        if not ok
                    ),
                ]
                for row in shown.itertuples(index=False)
            ],
        )
        if len(failing) > len(shown):
            lines += ["", f"{len(failing)}개 가운데 {len(shown)}개만 적었다. 전체는 `{PER_VIDEO_FILE}`에 있다."]
    lines += ["", "## 회전 점검", ""]
    lines += rotation_check_section(rotation, OVERRIDES_KEY)
    return lines


# ---------------------------------------------------------------------------
# 깜빡임 길이 모사 (PLAN 4.5절)
# ---------------------------------------------------------------------------

SIM_DURATIONS_MS: tuple[float, ...] = (60.0, 80.0, 100.0, 150.0, 200.0, 300.0, 400.0)
SIM_AMPLITUDES: tuple[float, ...] = (0.7, 0.85, 1.0)
SIM_CLOSING_FRACTION = 0.35
SIM_PHASES = 50
SIM_DROP_REPEATS = 4
SIM_BLINK_START_MS = 1000.0
SIM_TAIL_MS = 1500.0
FRAME_STAMP_MAX_DT_MS = 1

STAMPS_INTERPOLATED = "interpolated"
STAMPS_FRAME_TIME = "frame_time"
STAMPS: tuple[str, ...] = (STAMPS_INTERPOLATED, STAMPS_FRAME_TIME)

LEVEL_CELL = "cell"  # (조건, 시각 방식, 길이, 진폭)
LEVEL_DURATION = "duration"  # 진폭을 합친 것
LEVEL_CONDITION = "condition"  # 길이와 진폭을 모두 합친 것


@dataclass(frozen=True, slots=True)
class SimCondition:
    """표본화 조건. drop_prob는 프레임 하나가 빠질 확률(프레임마다 독립)이다."""

    name: str
    fps: float
    drop_prob: float


SIM_CONDITIONS: tuple[SimCondition, ...] = (
    SimCondition("30fps", 30.0, 0.0),
    SimCondition("15fps", 15.0, 0.0),
    SimCondition("15fps_drop20", 15.0, 0.2),
)


@dataclass(frozen=True, slots=True)
class BlinkShape:
    """합성 깜빡임 하나의 연속 파형.

    closure(t) = amplitude · 0.5 · (1 − cos(π·x / close_ms))                 0 ≤ x < close_ms   (감는 구간)
               = amplitude · 0.5 · (1 + cos(π·(x − close_ms) / open_ms))     close_ms ≤ x ≤ total_ms (뜨는 구간)
    x = t − start_ms, close_ms = closing_fraction · total_ms, open_ms = total_ms − close_ms. 그 밖에서는 0이다.

    가정: 실제 눈꺼풀 궤적의 측정값이 아니다. 감는 쪽이 뜨는 쪽보다 빠르고(전체의 35%) 양 끝과 정점에서 속도가 0인
      매끄러운 모양으로 정했다. 머무는 구간은 없다. 뜬 눈의 closure는 0이고 잡음과 MediaPipe의 시간 평활은 넣지 않았다.
    """

    start_ms: float
    total_ms: float
    amplitude: float
    closing_fraction: float

    @property
    def close_ms(self) -> float:
        return self.closing_fraction * self.total_ms

    @property
    def open_ms(self) -> float:
        return self.total_ms - self.close_ms

    def closure(self, t_ms: float) -> float:
        x = t_ms - self.start_ms
        if x <= 0.0 or x >= self.total_ms:
            return 0.0
        if x < self.close_ms:
            return self.amplitude * 0.5 * (1.0 - math.cos(math.pi * x / self.close_ms))
        return self.amplitude * 0.5 * (1.0 + math.cos(math.pi * (x - self.close_ms) / self.open_ms))

    def peak_closing_velocity(self) -> float:
        """연속 파형의 최대 감는 속도(1/s)."""
        return self.amplitude * math.pi / (2.0 * self.close_ms) * 1000.0

    def peak_opening_velocity(self) -> float:
        """연속 파형의 최대 뜨는 속도(1/s)."""
        return self.amplitude * math.pi / (2.0 * self.open_ms) * 1000.0


def blink_shape(
    duration_ms: float,
    amplitude: float,
    enter_thr: float,
    start_ms: float = SIM_BLINK_START_MS,
    closing_fraction: float = SIM_CLOSING_FRACTION,
) -> BlinkShape:
    """closure가 enter_thr 이상인 시간이 정확히 duration_ms인 깜빡임 파형.

    duration_ms는 검출기가 재는 길이("enter 넘게 감긴 시간", PLAN 4.5절)의 참값이다. 파형 전체의 길이는
    total_ms = duration_ms / (1 − φ), φ = acos(1 − 2·enter_thr / amplitude) / π 다(진폭이 1이고 임계가 0.5이면 2배).
    amplitude가 enter_thr 이하이면 ValueError다.
    """
    if not amplitude > enter_thr > 0.0:
        raise ValueError("blink amplitude must be above the enter threshold")
    if not duration_ms > 0.0 or not 0.0 < closing_fraction < 1.0:
        raise ValueError("need duration_ms > 0 and 0 < closing_fraction < 1")
    phi = math.acos(1.0 - 2.0 * enter_thr / amplitude) / math.pi
    return BlinkShape(start_ms, duration_ms / (1.0 - phi), amplitude, closing_fraction)


def sample_times(fps: float, phase_ms: float, end_ms: float, drop_prob: float, rng: random.Random) -> list[int]:
    """프레임 시각(정수 ms, 엄격히 증가). 프레임 k의 시각은 floor(phase_ms + k·1000/fps + 0.5)이고 end_ms 이하까지 낸다.

    drop_prob > 0이면 프레임마다 독립으로 그 확률로 뺀다(난수는 빠지는 프레임에서도 뽑는다).
    """
    period_ms = 1000.0 / fps
    times: list[int] = []
    k = 0
    while True:
        t_ms = math.floor(phase_ms + k * period_ms + 0.5)
        if t_ms > end_ms:
            break
        k += 1
        if drop_prob > 0.0 and rng.random() < drop_prob:
            continue
        if not times or t_ms > times[-1]:
            times.append(t_ms)
    return times


def detect_blink(shape: BlinkShape, times: Sequence[int], settings: RunDetectorSpec) -> tuple[ClosureRun | None, bool]:
    """파형을 times에서 표본화해 `ClosureRunDetector`에 넣는다. (잘리지 않은 첫 구간 또는 None, 잘린 구간이 있었는지)."""
    detector = ClosureRunDetector(settings)
    found: ClosureRun | None = None
    truncated = False
    for t_ms in times:
        run = detector.update(t_ms, shape.closure(float(t_ms)), True)
        if run is None:
            continue
        if run.truncated:
            truncated = True
        elif found is None:
            found = run
    return found, truncated


def frame_stamp_settings(settings: RunDetectorSpec) -> RunDetectorSpec:
    """보간을 끈 검출기 설정: `max_dt_ms`를 1로 둔다.

    검출기는 두 유효 표본의 간격이 `max_dt_ms` 이하일 때만 보간한다(PLAN 4.5절). 모사의 프레임 간격은 33 ms 이상이라
    어느 쌍도 보간되지 않고, `start_ms`는 enter 이상을 처음 본 프레임의 시각, `end_ms`는 enter 이상이었던 마지막 프레임의
    시각이 된다. 검출기 코드를 따로 두지 않고 설정만 바꾼 것이다. 이 설정에서는 속도도 재지 않는다
    (`velocity_valid = False`). 그래서 속도의 비는 보간하는 쪽에서만 낸다.
    """
    return dataclasses.replace(settings, max_dt_ms=FRAME_STAMP_MAX_DT_MS)


def _error_stats(errors: Sequence[float]) -> tuple[float, float, float]:
    """(치우침 = 평균 오차, 절대 오차의 중앙값, 절대 오차의 95 분위수). 분위수는 최근접 순위다."""
    if len(errors) == 0:
        return (math.nan, math.nan, math.nan)
    absolute = [abs(error) for error in errors]
    return (
        sum(errors) / len(errors),
        calibrate.median(absolute),
        calibrate.nearest_rank_quantile(absolute, 0.95),
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if len(values) > 0 else math.nan


def simulate_blinks(
    settings: RunDetectorSpec,
    *,
    seed: int = 0,
    durations_ms: Sequence[float] = SIM_DURATIONS_MS,
    amplitudes: Sequence[float] = SIM_AMPLITUDES,
    conditions: Sequence[SimCondition] = SIM_CONDITIONS,
    phases: int = SIM_PHASES,
    drop_repeats: int = SIM_DROP_REPEATS,
) -> pd.DataFrame:
    """깜빡임 길이 오차 모사 (PLAN 4.5절 마지막 항목). 같은 seed면 결과가 같다.

    조건(프레임률, 누락 확률)마다, 길이와 진폭의 조합마다 `blink_shape`의 파형을 위상 phases개(프레임 주기를 고르게
    나눈 값)로 표본화한다. 누락이 있는 조건은 위상마다 누락 패턴을 drop_repeats번 새로 뽑는다. 표본열 하나를 두 검출기에
    넣는다: settings 그대로(보간, `interpolated`)와 `frame_stamp_settings`(프레임 시각, `frame_time`).

    돌려주는 표의 열
    - `level`: `cell`(조건·시각 방식·길이·진폭), `duration`(진폭을 합침), `condition`(길이와 진폭을 모두 합침).
      합친 행의 `duration_ms`, `amplitude`, `total_ms`는 NaN이다.
    - `condition`, `fps`, `drop_prob`, `stamps`, `duration_ms`(enter 넘게 감긴 시간의 참값), `amplitude`,
      `total_ms`(파형 전체 길이)
    - `n_trials`, `n_detected`, `detection_rate`, `n_truncated`
    - `bias_ms`, `median_abs_error_ms`, `p95_abs_error_ms`: 검출된 깜빡임의 (검출기의 `duration_ms` − 참값)의 평균,
      절대값의 중앙값, 절대값의 95 분위수(최근접 순위)
    - `amplitude_ratio_mean`: 검출된 깜빡임의 `amplitude` / 파형의 진폭
    - `n_velocity_valid`, `closing_velocity_ratio_mean`, `opening_velocity_ratio_mean`: `velocity_valid`인 깜빡임의
      `peak_closing_velocity`, `peak_opening_velocity`를 연속 파형의 최대 속도로 나눈 값의 평균. `frame_time`에서는 NaN이다.

    가정: 검출은 "잘리지 않은 구간이 하나 나왔다"로 센다. 프레임이 500 ms 넘게 끊겨 잘린 구간만 나온 시행은 검출되지
      않은 것으로 세고 `n_truncated`에 적는다.
    가정: 길이 격자는 enter 넘게 감긴 시간으로 60~400 ms, 진폭은 0.7, 0.85, 1.0, 위상은 50개, 누락 확률은 0.2
      (PLAN 5.2절의 10~30%의 가운데), 누락 패턴은 위상마다 4번이다. 합친 행은 격자의 모든 조합에 같은 비중을 준다.
      실제 깜빡임 길이의 분포가 아니므로 합친 값은 격자에 따라 달라진다.
    """
    if phases < 1 or drop_repeats < 1:
        raise ValueError("phases and drop_repeats must be >= 1")
    detectors = {STAMPS_INTERPOLATED: settings, STAMPS_FRAME_TIME: frame_stamp_settings(settings)}
    trials: list[dict[str, Any]] = []
    for condition in conditions:
        period_ms = 1000.0 / condition.fps
        if math.floor(period_ms) <= FRAME_STAMP_MAX_DT_MS:
            raise ValueError("frame interval too short for the frame-time emulation")
        repeats = drop_repeats if condition.drop_prob > 0.0 else 1
        for duration_ms in durations_ms:
            for amplitude in amplitudes:
                shape = blink_shape(duration_ms, amplitude, settings.enter_thr)
                end_ms = shape.start_ms + shape.total_ms + SIM_TAIL_MS
                for phase in range(phases):
                    for repeat in range(repeats):
                        rng = random.Random(f"{seed}:{condition.name}:{duration_ms!r}:{amplitude!r}:{phase}:{repeat}")
                        times = sample_times(
                            condition.fps, period_ms * phase / phases, end_ms, condition.drop_prob, rng
                        )
                        for stamps, detector_settings in detectors.items():
                            run, truncated = detect_blink(shape, times, detector_settings)
                            trial: dict[str, Any] = {
                                "condition": condition.name,
                                "stamps": stamps,
                                "duration_ms": float(duration_ms),
                                "amplitude": float(amplitude),
                                "detected": run is not None,
                                "truncated": truncated,
                                "error_ms": math.nan,
                                "amplitude_ratio": math.nan,
                                "velocity_valid": False,
                                "closing_ratio": math.nan,
                                "opening_ratio": math.nan,
                            }
                            if run is not None:
                                trial["error_ms"] = run.duration_ms - float(duration_ms)
                                trial["amplitude_ratio"] = run.amplitude / amplitude
                                if run.velocity_valid:
                                    trial["velocity_valid"] = True
                                    trial["closing_ratio"] = run.peak_closing_velocity / shape.peak_closing_velocity()
                                    trial["opening_ratio"] = run.peak_opening_velocity / shape.peak_opening_velocity()
                            trials.append(trial)

    by_condition = {condition.name: condition for condition in conditions}
    rows: list[dict[str, Any]] = []

    def aggregate(level: str, selected: Sequence[dict[str, Any]], duration_ms: float, amplitude: float) -> None:
        first = selected[0]
        condition = by_condition[first["condition"]]
        detected = [trial for trial in selected if trial["detected"]]
        bias, median_abs, p95_abs = _error_stats([trial["error_ms"] for trial in detected])
        with_velocity = [trial for trial in detected if trial["velocity_valid"]]
        total_ms = math.nan
        if level == LEVEL_CELL:
            total_ms = blink_shape(duration_ms, amplitude, settings.enter_thr).total_ms
        rows.append(
            {
                "level": level,
                "condition": condition.name,
                "fps": condition.fps,
                "drop_prob": condition.drop_prob,
                "stamps": first["stamps"],
                "duration_ms": duration_ms,
                "amplitude": amplitude,
                "total_ms": total_ms,
                "n_trials": len(selected),
                "n_detected": len(detected),
                "detection_rate": len(detected) / len(selected),
                "n_truncated": sum(1 for trial in selected if trial["truncated"]),
                "bias_ms": bias,
                "median_abs_error_ms": median_abs,
                "p95_abs_error_ms": p95_abs,
                "amplitude_ratio_mean": _mean([trial["amplitude_ratio"] for trial in detected]),
                "n_velocity_valid": len(with_velocity),
                "closing_velocity_ratio_mean": _mean([trial["closing_ratio"] for trial in with_velocity]),
                "opening_velocity_ratio_mean": _mean([trial["opening_ratio"] for trial in with_velocity]),
            }
        )

    for condition in conditions:
        for stamps in STAMPS:
            group = [trial for trial in trials if trial["condition"] == condition.name and trial["stamps"] == stamps]
            for duration_ms in durations_ms:
                of_duration = [trial for trial in group if trial["duration_ms"] == float(duration_ms)]
                for amplitude in amplitudes:
                    cell = [trial for trial in of_duration if trial["amplitude"] == float(amplitude)]
                    aggregate(LEVEL_CELL, cell, float(duration_ms), float(amplitude))
                aggregate(LEVEL_DURATION, of_duration, float(duration_ms), math.nan)
            aggregate(LEVEL_CONDITION, group, math.nan, math.nan)
    return pd.DataFrame(rows)


def run_blink_sim(
    cfg: ExperimentConfig, *, run_name: str = DEFAULT_RUN_NAME, seed: int = 0, command: str = ""
) -> tuple[Path, pd.DataFrame]:
    """깜빡임 길이 모사를 돌리고 `{reports_dir}/{run_name}/blink_sim/`에 summary.md와 blink_sim.csv를 쓴다.

    검출기 설정은 `paths.feature_spec`의 `pipeline.run_detector`(특징용 임계값)다. 두 파일의 첫 줄은 모사임을 적는다.
    (리포트 폴더, `simulate_blinks`의 표)를 돌려준다. 같은 seed와 같은 spec이면 파일이 바이트 단위로 같다.
    run_name이 폴더 이름 하나가 아니면 ValueError다(`config.check_run_name`).
    """
    run_name = check_run_name(run_name)
    settings = load_feature_spec(cfg.paths.feature_spec).pipeline.run_detector
    table = simulate_blinks(settings, seed=seed)
    run_dir = cfg.paths.reports_dir / run_name / BLINK_SIM_SUBDIR
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / BLINK_SIM_CSV_FILE).open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# {BLINK_SIM_NOTE}\n")
        table.to_csv(handle, index=False, lineterminator="\n")
    (run_dir / SUMMARY_FILE).write_text(
        "\n".join(_blink_sim_summary(table, settings, seed, command)) + "\n", encoding="utf-8"
    )
    return run_dir, table


def _sim_row(table: pd.DataFrame, level: str, condition: str, stamps: str, duration_ms: float | None = None) -> Any:
    rows = table[(table["level"] == level) & (table["condition"] == condition) & (table["stamps"] == stamps)]
    if duration_ms is not None:
        rows = rows[rows["duration_ms"] == duration_ms]
    if len(rows) != 1:
        raise ValueError(f"expected one {level} row for {condition}/{stamps}/{duration_ms}, found {len(rows)}")
    return rows.iloc[0]


def _blink_sim_summary(table: pd.DataFrame, settings: RunDetectorSpec, seed: int, command: str) -> list[str]:
    """blink-sim summary.md의 줄 목록."""
    conditions = list(dict.fromkeys(table["condition"].tolist()))
    durations = sorted({float(value) for value in table.loc[table["level"] == LEVEL_DURATION, "duration_ms"].tolist()})
    amplitudes = sorted({float(value) for value in table.loc[table["level"] == LEVEL_CELL, "amplitude"].tolist()})
    cells = table[table["level"] == LEVEL_CELL]
    length_ratio = (cells["total_ms"] / cells["duration_ms"]).to_numpy(dtype=np.float64)
    per_condition = {
        name: int(cells[(cells["condition"] == name) & (cells["stamps"] == STAMPS_INTERPOLATED)]["n_trials"].sum())
        for name in conditions
    }
    lines = [
        f"# {BLINK_SIM_NOTE}",
        "",
        "깜빡임 길이 오차 모사 (PLAN 4.5절 마지막 항목). 합성 깜빡임 파형을 프레임률과 위상을 바꿔 표본화하고 "
        f"`ClosureRunDetector`(특징용: 진입 {settings.enter_thr:g} / 이탈 {settings.exit_thr:g}, 최소 길이 "
        f"{settings.min_duration_ms} ms, `max_dt_ms` {settings.max_dt_ms})에 넣어 잰 값이다. 실제 눈을 잰 값이 아니고 "
        "MediaPipe의 시간 평활과 신호 잡음은 들어 있지 않다.",
        "",
        f"- 생성 명령: `{command}`",
        f"- 시드: {seed}(프레임 누락 패턴에만 쓰인다)",
        f"- 파형: 감는 구간이 전체의 {100.0 * SIM_CLOSING_FRACTION:g}%인 비대칭 코사인 모양(`signal_check.BlinkShape`). "
        "머무는 구간 없음, 뜬 눈의 closure는 0",
        "- 길이: closure가 진입 임계 이상인 시간(검출기가 재는 `duration_ms`의 참값) "
        + ", ".join(f"{value:g}" for value in durations)
        + " ms. 진폭: "
        + ", ".join(f"{value:g}" for value in amplitudes)
        + f". 파형 전체의 길이는 이 길이의 {length_ratio.min():.2f}~{length_ratio.max():.2f}배다(진폭이 낮을수록 길다)",
        "- 조건별 시행 수(길이 × 진폭 × 위상 × 누락 패턴): "
        + ", ".join(f"`{name}` {count}" for name, count in per_condition.items()),
        f"- 시각 방식: `{STAMPS_INTERPOLATED}`는 검출기 그대로(앞뒤 표본의 선형 보간), `{STAMPS_FRAME_TIME}`은 같은 "
        f"검출기에 `max_dt_ms = {FRAME_STAMP_MAX_DT_MS}`을 줘서 보간을 끈 것이다(시작은 진입 임계 이상을 처음 본 프레임, "
        "끝은 진입 임계 이상이었던 마지막 프레임의 시각. 검출기가 간격이 `max_dt_ms`를 넘는 쌍에서 하는 것과 같다)",
        "- 오차 = 검출기의 `duration_ms` − 참값. 치우침은 평균, 중앙 절대 오차는 절대값의 중앙값, 95%는 절대값의 "
        "95 분위수(최근접 순위)이고 검출된 깜빡임만으로 계산한다",
        "- 합친 행은 길이와 진폭의 모든 조합에 같은 비중을 준 값이다. 실제 깜빡임 길이의 분포가 아니다",
        "",
        "## 조건별 (길이와 진폭을 모두 합침)",
        "",
    ]
    rows = []
    for name in conditions:
        for stamps in STAMPS:
            row = _sim_row(table, LEVEL_CONDITION, name, stamps)
            rows.append(
                [
                    f"`{name}`",
                    f"`{stamps}`",
                    f"{int(row['n_detected'])} / {int(row['n_trials'])}",
                    f"{100.0 * row['detection_rate']:.1f}%",
                    _fmt(row["bias_ms"], 1),
                    _fmt(row["median_abs_error_ms"], 1),
                    _fmt(row["p95_abs_error_ms"], 1),
                ]
            )
    lines += _md_table(
        ["조건", "시각 방식", "검출", "검출률", "치우침 (ms)", "중앙 절대 오차 (ms)", "절대 오차 95% (ms)"], rows
    )

    lines += ["", "## 길이별 (진폭을 합침)", ""]
    rows = []
    for name in conditions:
        for duration_ms in durations:
            interpolated = _sim_row(table, LEVEL_DURATION, name, STAMPS_INTERPOLATED, duration_ms)
            frame_time = _sim_row(table, LEVEL_DURATION, name, STAMPS_FRAME_TIME, duration_ms)
            rows.append(
                [
                    f"`{name}`",
                    f"{duration_ms:g}",
                    f"{100.0 * interpolated['detection_rate']:.1f}%",
                    _fmt(interpolated["bias_ms"], 1),
                    _fmt(interpolated["median_abs_error_ms"], 1),
                    _fmt(interpolated["p95_abs_error_ms"], 1),
                    f"{100.0 * frame_time['detection_rate']:.1f}%",
                    _fmt(frame_time["bias_ms"], 1),
                    _fmt(frame_time["median_abs_error_ms"], 1),
                    _fmt(frame_time["p95_abs_error_ms"], 1),
                ]
            )
    lines += _md_table(
        [
            "조건",
            "길이 (ms)",
            "보간: 검출률",
            "보간: 치우침",
            "보간: 중앙 절대 오차",
            "보간: 95%",
            "프레임 시각: 검출률",
            "프레임 시각: 치우침",
            "프레임 시각: 중앙 절대 오차",
            "프레임 시각: 95%",
        ],
        rows,
    )

    lines += [
        "",
        "## 진폭과 속도 (연속 파형 대비, 보간하는 검출기)",
        "",
        "진폭의 비 = 검출기의 `amplitude` / 파형의 진폭. 속도의 비 = `peak_closing_velocity`, `peak_opening_velocity` / "
        "연속 파형의 최대 속도. 속도의 비는 `velocity_valid`인 깜빡임만으로 계산한다.",
        "",
    ]
    rows = []
    for name in conditions:
        for duration_ms in [None, *durations]:
            level = LEVEL_CONDITION if duration_ms is None else LEVEL_DURATION
            row = _sim_row(table, level, name, STAMPS_INTERPOLATED, duration_ms)
            rows.append(
                [
                    f"`{name}`",
                    "전체" if duration_ms is None else f"{duration_ms:g}",
                    _fmt(row["amplitude_ratio_mean"], 3),
                    _fmt(row["closing_velocity_ratio_mean"], 3),
                    _fmt(row["opening_velocity_ratio_mean"], 3),
                    f"{int(row['n_velocity_valid'])} / {int(row['n_detected'])}",
                ]
            )
    lines += _md_table(
        ["조건", "길이 (ms)", "진폭의 비", "감는 속도의 비", "뜨는 속도의 비", "속도가 유효한 깜빡임 / 검출된 깜빡임"],
        rows,
    )
    truncated = int(table.loc[table["level"] == LEVEL_CONDITION, "n_truncated"].sum())
    lines += [
        "",
        f"프레임이 500 ms 넘게 끊겨 잘린 구간이 나온 시행은 두 시각 방식을 합쳐 {truncated}번이다(검출되지 않은 것으로 셌다). "
        f"길이·진폭별 값은 `{BLINK_SIM_CSV_FILE}`에 있다.",
    ]
    return lines


# ---------------------------------------------------------------------------
# 명령
# ---------------------------------------------------------------------------


def _command_line(argv: Sequence[str] | None) -> str:
    arguments = sys.argv[1:] if argv is None else list(argv)
    return " ".join(["python -m drowsy.signal_check", *arguments])


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m drowsy.signal_check`.

    종료 코드: 0 성공, 2 시작할 수 없음(설정, 데이터, 덮어쓰기 값, 폴더 이름이 아닌 `--run-name`), 4 실제 데이터에서
    E0 통과 기준 미달(리포트는 써 둔다).
    두 하위 명령 모두 다른 일을 하기 전에 `--run-name`을 검사한다(`config.check_run_name`): 경로 구분자나 `..`이 든
    이름은 리포트를 `paths.reports_dir` 밖에 쓰게 되므로 한 줄 오류와 함께 2로 끝난다.
    가정: `--fps-sim` 실행은 참고용 점검이라 기준에 미달해도 0으로 끝낸다(판정은 원본 프레임률 추출로 한다).
    가정: 회전이 확인되지 않은 영상이 있어도 종료 코드는 바꾸지 않는다. 한 줄로 알리고 목록은 리포트에 싣는다.
    """
    parser = argparse.ArgumentParser(
        prog="python -m drowsy.signal_check",
        description="E0 label-free signal check with proposed constants, and the blink-duration simulation "
        "(PLAN 4.5, 11).",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    e0 = commands.add_parser("e0", help="label-free signal check, pass criteria and proposed global constants")
    sim = commands.add_parser("blink-sim", help="blink-duration error simulation at 30 and 15 fps (synthetic blinks)")
    for command in (e0, sim):
        command.add_argument("--config", default="config/default.yaml", help="experiment config YAML")
        command.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="report folder name below paths.reports_dir")
    e0.add_argument(
        "--override",
        nargs="+",
        action="extend",
        default=None,
        metavar="PATH=VALUE",
        help="evaluate with these pipeline constants instead of the spec file's, e.g. closure.rho=0.30",
    )
    e0.add_argument("--workers", type=int, default=None, help="parallel sessions (default: extraction.workers)")
    e0.add_argument("--fps-sim", action="store_true", help="check the frame-drop simulated extraction instead")
    sim.add_argument("--seed", type=int, default=0, help="seed of the random frame drops")
    args = parser.parse_args(argv)
    command = _command_line(argv)

    started = time.perf_counter()
    try:
        check_run_name(args.run_name)
        cfg = load_config(args.config)
        if args.command == "blink-sim":
            run_dir, table = run_blink_sim(cfg, run_name=args.run_name, seed=args.seed, command=command)
            print(BLINK_SIM_NOTE)
            for name in dict.fromkeys(table["condition"].tolist()):
                for stamps in STAMPS:
                    row = _sim_row(table, LEVEL_CONDITION, name, stamps)
                    print(
                        f"[{name}, {stamps}] detected {int(row['n_detected'])}/{int(row['n_trials'])}, duration error: "
                        f"bias {row['bias_ms']:.1f} ms, median abs {row['median_abs_error_ms']:.1f} ms, "
                        f"p95 abs {row['p95_abs_error_ms']:.1f} ms"
                    )
            print(f"reports: {run_dir / SUMMARY_FILE}")
            print(f"elapsed: {time.perf_counter() - started:.1f} s")
            return 0

        overrides = parse_overrides(args.override or [])
        source = open_dataset(cfg, fps_sim=args.fps_sim)
        note = data_note(source.synthetic)
        if not source.description.startswith(note):  # 합성 데이터의 설명은 이미 같은 문장으로 시작한다
            print(note)
        print(source.description, flush=True)
        result = run_e0(
            cfg,
            overrides=overrides,
            run_name=args.run_name,
            fps_sim=args.fps_sim,
            workers=args.workers,
            command=command,
            config_path=Path(args.config),
            source=source,
        )
    except (ValueError, OSError, ExtractError) as exc:  # SpecError와 ManifestError도 ValueError다
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for criterion in result.criteria:
        print(
            f"[{criterion.name}] {criterion.description}: {criterion.count}/{criterion.total} sessions = "
            f"{100.0 * criterion.ratio:.1f}% (need >= {100.0 * criterion.required_ratio:g}%): "
            f"{'met' if criterion.passed else 'NOT met'}"
        )
    check_only = " (synthetic data: pipeline check only)" if result.synthetic else ""
    verdict = "met" if result.passed else "NOT met"
    print(f"E0 criteria (label-free, {len(result.table)} sessions): {verdict}{check_only}")
    print(rotation_check_line(result.rotation, f"section '회전 점검' of {result.run_dir / SUMMARY_FILE}"))
    if result.constants is not None:
        constants = result.constants
        label = "global constants of this extraction (reference only)" if args.fps_sim else "proposed global constants"
        print(
            f"{label} (median over sessions, test subjects included): e={constants.e_l:g}, "
            f"b={constants.b_l:g}, jaw0={constants.jaw0:g}"
        )
    if result.proposed_spec_path is not None:
        print(f"proposed spec: {result.proposed_spec_path} (config/feature_spec.json is not modified)")
    print(f"reports: {result.run_dir / SUMMARY_FILE}")
    print(f"elapsed: {time.perf_counter() - started:.1f} s")
    if not result.passed and not result.synthetic and not args.fps_sim:
        print("E0 criteria not met: stop before training and report (PLAN sections 11 and 12)", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    # `python -m`으로 실행하면 이 파일이 `__main__`으로 한 번 더 올라온다. 재생 작업을 다른 프로세스로 넘길 때
    # 클래스가 두 모듈에 따로 있지 않도록, 패키지에서 import한 쪽의 main을 부른다.
    from drowsy.signal_check import main as _main

    sys.exit(_main())
