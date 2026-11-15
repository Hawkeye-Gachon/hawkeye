"""실험 설정(config/default.yaml)의 자료형과 로더. 이식 대상이 아니다.

이식 코드의 상수는 여기에 두지 않는다. 그것은 `drowsy.spec`의 PipelineSpec과 RulesConfig에 있다.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import yaml

from drowsy.spec import SpecError, build_dataclass

ARM_ALERT_BASELINE = "alert_baseline"
ARM_GLOBAL = "global"
ARM_SESSION_BASELINE = "session_baseline"
ARMS: tuple[str, ...] = (ARM_ALERT_BASELINE, ARM_GLOBAL, ARM_SESSION_BASELINE)


@dataclass(frozen=True, slots=True)
class PathSettings:
    """경로. 상대 경로는 명령을 실행한 위치(저장소 루트) 기준이다."""

    rldd_root: Path | None  # UTA-RLDD 사본(Q8). 없으면 None
    unlabeled_root: Path | None  # 팀 수집 차량 영상. 없으면 None
    raw_dir: Path
    synth_dir: Path  # 합성 데이터셋(drowsy.synth가 쓴 RawFrame parquet). rldd_root가 None일 때 쓴다
    model_path: Path
    feature_spec: Path
    rules: Path
    export_dir: Path
    reports_dir: Path


@dataclass(frozen=True, slots=True)
class DropSimConfig:
    """기기 프레임 누락 모사(흐름 제한기) 설정 (PLAN 5.2절 증강 (1))."""

    subdir: str  # raw_dir 아래 모사 추출 결과 폴더
    latency_ms_mean: float  # 프레임당 처리 지연 평균. 66.7이면 약 15 fps
    latency_ms_std: float
    seed: int


@dataclass(frozen=True, slots=True)
class ExtractionSettings:
    workers: int
    pilot_videos: int
    native_subdir: str
    rotation_overrides: dict[str, int]  # 영상 상대 경로 -> 회전 각도(0, 90, 180, 270)
    rotation_check_frames: int  # 영상별 회전 자동 점검(PLAN 12장)에 쓰는 표본 프레임 수
    rotation_check_interval_ms: int  # 표본 프레임 사이의 간격
    rotation_roll_margin_deg: float  # 회전 점검: |roll 중앙값|의 차이가 이 값 미만인 각도들은 가르지 못한 것으로 본다
    unlabeled_rotation_overrides: dict[str, int]  # `--unlabeled` 추출의 수동 회전 보정표(unlabeled_root 기준 상대 경로)
    drop_sim: DropSimConfig


@dataclass(frozen=True, slots=True)
class WindowSettings:
    window_steps: int
    train_stride: int
    eval_stride: int


@dataclass(frozen=True, slots=True)
class TcnGrid:
    channels: tuple[int, ...]
    dropout: tuple[float, ...]
    kernel_size: int
    dilations: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GruGrid:
    hidden: tuple[int, ...]
    dropout: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    """학습 설정 (PLAN 5.2절)."""

    arms: tuple[str, ...]
    export_arm: str
    optimizer: str
    lr: float
    weight_decay: float
    batch_size: int
    updates_per_unit: int
    max_units: int
    patience_units: int
    label_smoothing: float
    ordinal_neighbor_mass: float  # 부가 실험(순서형 소프트 라벨)에서 정답 클래스가 이웃 클래스에 나눠 주는 확률
    noise_sigma: float
    frame_drop_min: float
    frame_drop_max: float
    fps_sim_share: float  # 누락 모사 추출의 스텝 표가 있을 때 학습 표본 중 거기서 뽑는 비율(증강 (1))
    frame_drop_share: float  # 프레임 누락 변형의 스텝 표에서 뽑는 비율(증강 (2)). 변형들이 똑같이 나눈다
    frame_drop_variants: int  # 프레임 누락 변형의 수(0이면 만들지 않는다)
    seeds: tuple[int, ...]
    final_seed: int
    device: str
    tcn: TcnGrid
    gru: GruGrid


@dataclass(frozen=True, slots=True)
class LgbmSettings:
    """LightGBM 비교 기준선 (PLAN 5.3절)."""

    num_leaves: tuple[int, ...]
    min_data_in_leaf: tuple[int, ...]
    learning_rate: float
    max_rounds: int
    early_stopping_rounds: int
    seed: int
    reference_arm: str
    reference_w: float
    w_grid: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class QualityGate:
    """영상 품질 기준 (D7)."""

    min_face_ratio: float
    min_windows: int


@dataclass(frozen=True, slots=True)
class ModelGate:
    """모델을 기본 졸음 판정원으로 둘 조건 (D11)."""

    max_macro_f1_deficit: float
    min_two_class_video_accuracy: float
    max_false_alarms_per_hour: float
    max_alert_video_alarm_ratio: float
    min_drowsy_video_alarm_ratio: float


@dataclass(frozen=True, slots=True)
class SignalGate:
    """E0 통과 기준 (PLAN 11장)."""

    min_baseline_ok_ratio: float
    blink_rate_min_per_min: float
    blink_rate_max_per_min: float
    min_blink_rate_ok_ratio: float
    min_peak_closure_median: float
    min_peak_closure_ok_ratio: float


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    """평가 설정 (PLAN 6장)."""

    n_folds: int
    subjects_per_fold: int
    bootstrap_n: int
    bootstrap_seed: int
    alarm_threshold_sweep: tuple[float, ...]
    lgbm_min_macro_f1: float
    quality_gate: QualityGate
    model_gate: ModelGate
    signal_gate: SignalGate


@dataclass(frozen=True, slots=True)
class ExportSettings:
    """ONNX 내보내기 설정 (PLAN 7장)."""

    model_version: int
    opset: int
    ir_version: int
    input_name: str
    output_name: str
    max_file_bytes: int
    parity_tolerance: float
    parity_random_inputs: int
    latency_runs: int
    intra_op_threads: int
    int8_if_tcn: bool


@dataclass(frozen=True, slots=True)
class DemoSettings:
    model_path: Path
    out_dir: Path
    figure_dpi: int


@dataclass(frozen=True, slots=True)
class FastProfile:
    """연기 시험용 축소 설정(`--fast`). `apply_fast_profile`이 해당 설정 값을 이것으로 바꾼다.

    사전에 고정한 절차(PLAN 5.2절, 5.3절, 6장)가 아니다. 전 과정이 도는지만 빨리 확인하는 용도이고,
    이 설정으로 만든 수치는 보고하지 않는다.
    """

    max_units: int
    patience_units: int
    updates_per_unit: int
    seeds: tuple[int, ...]
    tcn_channels: tuple[int, ...]
    tcn_dropout: tuple[float, ...]
    gru_hidden: tuple[int, ...]
    gru_dropout: tuple[float, ...]
    lgbm_num_leaves: tuple[int, ...]
    lgbm_min_data_in_leaf: tuple[int, ...]
    lgbm_max_rounds: int
    bootstrap_n: int


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """config/default.yaml 전체."""

    paths: PathSettings
    extraction: ExtractionSettings
    windows: WindowSettings
    training: TrainingSettings
    lgbm: LgbmSettings
    evaluation: EvaluationSettings
    export: ExportSettings
    demo: DemoSettings
    fast: FastProfile


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SpecError(message)


def check_run_name(run_name: str) -> str:
    """`--run-name` 값이 폴더 이름 하나인지 확인하고 그대로 돌려준다.

    리포트는 `paths.reports_dir/<run name>/` 아래에 쓰므로, 경로 구분자나 `..`이 들어간 이름은
    리포트 폴더 밖(예: `../export`)에 파일을 쓰게 된다. 그런 값은 ValueError로 거부한다.
    `--run-name`을 받는 모든 명령이 다른 일을 하기 전에 이 함수를 부른다.
    """
    if run_name in ("", ".", "..") or "/" in run_name or "\\" in run_name:
        raise ValueError(f"--run-name must be a plain folder name, got {run_name!r}")
    return run_name


def _validate(cfg: ExperimentConfig) -> None:
    ext = cfg.extraction
    _check(ext.workers >= 1, "extraction.workers must be >= 1")
    _check(ext.pilot_videos >= 0, "extraction.pilot_videos must be >= 0")
    _check(
        all(angle in (0, 90, 180, 270) for angle in ext.rotation_overrides.values()),
        "extraction.rotation_overrides values must be 0, 90, 180 or 270",
    )
    _check(
        ext.rotation_check_frames >= 1 and ext.rotation_check_interval_ms >= 1,
        "extraction: rotation_check_frames and rotation_check_interval_ms must be >= 1",
    )
    _check(
        0.0 < ext.rotation_roll_margin_deg < 90.0,
        "extraction.rotation_roll_margin_deg must be within (0, 90) degrees",
    )
    _check(
        all(angle in (0, 90, 180, 270) for angle in ext.unlabeled_rotation_overrides.values()),
        "extraction.unlabeled_rotation_overrides values must be 0, 90, 180 or 270",
    )
    _check(ext.drop_sim.latency_ms_mean > 0.0, "extraction.drop_sim.latency_ms_mean must be > 0")
    _check(ext.drop_sim.latency_ms_std >= 0.0, "extraction.drop_sim.latency_ms_std must be >= 0")

    win = cfg.windows
    _check(win.window_steps >= 1 and win.train_stride >= 1 and win.eval_stride >= 1, "windows: values must be >= 1")

    tr = cfg.training
    _check(len(tr.arms) > 0 and all(arm in ARMS for arm in tr.arms), f"training.arms must be drawn from {ARMS}")
    _check(tr.export_arm in tr.arms, "training.export_arm must be one of training.arms")
    _check(tr.lr > 0.0 and tr.weight_decay >= 0.0, "training: need lr > 0 and weight_decay >= 0")
    _check(
        tr.batch_size >= 1 and tr.updates_per_unit >= 1 and tr.max_units >= 1 and tr.patience_units >= 1,
        "training: batch_size, updates_per_unit, max_units and patience_units must be >= 1",
    )
    _check(0.0 <= tr.label_smoothing < 1.0, "training.label_smoothing must be within [0, 1)")
    _check(tr.noise_sigma >= 0.0, "training.noise_sigma must be >= 0")
    _check(
        0.0 <= tr.frame_drop_min <= tr.frame_drop_max < 1.0,
        "training: need 0 <= frame_drop_min <= frame_drop_max < 1",
    )
    _check(0.0 <= tr.fps_sim_share < 1.0, "training.fps_sim_share must be within [0, 1)")
    _check(0.0 <= tr.frame_drop_share < 1.0, "training.frame_drop_share must be within [0, 1)")
    # 가정: 학습 표본에서 두 증강의 몫을 뺀 나머지가 원본 추출의 몫이다. 합이 1을 채우면 원본에서 뽑을 것이 없다.
    _check(
        tr.fps_sim_share + tr.frame_drop_share < 1.0,
        "training: fps_sim_share + frame_drop_share must be below 1 (the rest is the share of the native extraction)",
    )
    _check(tr.frame_drop_variants >= 0, "training.frame_drop_variants must be >= 0")
    # 가정: 이웃에 주는 확률이 0.5 이상이면 정답 클래스가 더는 가장 큰 목표가 아니다. 그 아래로만 받는다.
    _check(0.0 <= tr.ordinal_neighbor_mass < 0.5, "training.ordinal_neighbor_mass must be within [0, 0.5)")
    _check(len(tr.seeds) > 0, "training.seeds must not be empty")
    _check(tr.device in ("auto", "cpu", "cuda"), "training.device must be auto, cpu or cuda")
    for name, grid in (
        ("tcn.channels", tr.tcn.channels),
        ("tcn.dropout", tr.tcn.dropout),
        ("tcn.dilations", tr.tcn.dilations),
        ("gru.hidden", tr.gru.hidden),
        ("gru.dropout", tr.gru.dropout),
    ):
        _check(len(grid) > 0, f"training.{name} must not be empty")

    lg = cfg.lgbm
    _check(len(lg.num_leaves) > 0 and len(lg.min_data_in_leaf) > 0, "lgbm grids must not be empty")
    _check(
        lg.learning_rate > 0.0 and lg.max_rounds >= 1 and lg.early_stopping_rounds >= 1,
        "lgbm: need learning_rate > 0, max_rounds >= 1 and early_stopping_rounds >= 1",
    )
    _check(lg.reference_arm in ARMS, f"lgbm.reference_arm must be one of {ARMS}")
    _check(
        len(lg.w_grid) > 0 and len(set(lg.w_grid)) == len(lg.w_grid) and all(0.0 <= w <= 1.0 for w in lg.w_grid),
        "lgbm.w_grid must list distinct closure W values within 0..1",
    )
    # 가정: W 비교(`baseline_lgbm cv --all-w`)는 기준 W의 묶음을 주 결과로, 나머지를 그에 대한 부가 결과로 싣는다.
    # 그래서 값 목록에 기준 W가 들어 있어야 한다.
    _check(lg.reference_w in lg.w_grid, "lgbm.w_grid must contain lgbm.reference_w")

    ev = cfg.evaluation
    _check(ev.n_folds >= 2 and ev.subjects_per_fold >= 1, "evaluation: need n_folds >= 2 and subjects_per_fold >= 1")
    _check(ev.bootstrap_n >= 1, "evaluation.bootstrap_n must be >= 1")
    _check(0.0 <= ev.quality_gate.min_face_ratio <= 1.0, "evaluation.quality_gate.min_face_ratio must be within 0..1")
    _check(ev.quality_gate.min_windows >= 0, "evaluation.quality_gate.min_windows must be >= 0")

    ex = cfg.export
    _check(ex.model_version >= 0, "export.model_version must be >= 0")
    _check(
        ex.max_file_bytes > 0 and ex.parity_tolerance > 0.0,
        "export: need max_file_bytes > 0 and parity_tolerance > 0",
    )
    _check(
        ex.parity_random_inputs >= 1 and ex.latency_runs >= 1 and ex.intra_op_threads >= 1,
        "export: parity_random_inputs, latency_runs and intra_op_threads must be >= 1",
    )

    fast = cfg.fast
    _check(
        fast.max_units >= 1 and fast.patience_units >= 1 and fast.updates_per_unit >= 1,
        "fast: max_units, patience_units and updates_per_unit must be >= 1",
    )
    for name in (
        "seeds",
        "tcn_channels",
        "tcn_dropout",
        "gru_hidden",
        "gru_dropout",
        "lgbm_num_leaves",
        "lgbm_min_data_in_leaf",
    ):
        _check(len(getattr(fast, name)) > 0, f"fast.{name} must not be empty")
    _check(fast.lgbm_max_rounds >= 1 and fast.bootstrap_n >= 1, "fast: lgbm_max_rounds and bootstrap_n must be >= 1")


def apply_fast_profile(cfg: ExperimentConfig) -> ExperimentConfig:
    """`fast` 블록의 값으로 학습 길이, 시드, 탐색 범위, 부트스트랩 횟수를 줄인 설정 사본을 돌려준다.

    바뀌는 것: training의 max_units·patience_units·updates_per_unit·seeds와 tcn·gru의 탐색 범위,
    lgbm의 탐색 범위와 max_rounds(early_stopping_rounds는 max_rounds를 넘지 않게 줄인다), evaluation.bootstrap_n.
    나머지(창, 보폭, fold, 규칙, 내보내기 계약)는 그대로다.
    """
    fast = cfg.fast
    training = dataclasses.replace(
        cfg.training,
        max_units=fast.max_units,
        patience_units=fast.patience_units,
        updates_per_unit=fast.updates_per_unit,
        seeds=fast.seeds,
        tcn=dataclasses.replace(cfg.training.tcn, channels=fast.tcn_channels, dropout=fast.tcn_dropout),
        gru=dataclasses.replace(cfg.training.gru, hidden=fast.gru_hidden, dropout=fast.gru_dropout),
    )
    lgbm = dataclasses.replace(
        cfg.lgbm,
        num_leaves=fast.lgbm_num_leaves,
        min_data_in_leaf=fast.lgbm_min_data_in_leaf,
        max_rounds=fast.lgbm_max_rounds,
        early_stopping_rounds=min(cfg.lgbm.early_stopping_rounds, fast.lgbm_max_rounds),
    )
    evaluation = dataclasses.replace(cfg.evaluation, bootstrap_n=fast.bootstrap_n)
    fast_cfg = dataclasses.replace(cfg, training=training, lgbm=lgbm, evaluation=evaluation)
    _validate(fast_cfg)
    return fast_cfg


def load_config(path: str | Path) -> ExperimentConfig:
    """실험 설정 YAML을 읽어 검증한다. 모르는 키, 빠진 키, 틀린 자료형은 SpecError다."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg = build_dataclass(ExperimentConfig, data, "config")
    _validate(cfg)
    return cfg
