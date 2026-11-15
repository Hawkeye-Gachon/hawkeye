"""영상 → MediaPipe FaceLandmarker(VIDEO 모드) → RawFrame parquet, 프레임 누락 모사 (PLAN 2.3, 4.1, 5.2절).

이식 대상이 아니다. mediapipe는 랜드마커를 만들 때만 import한다(parquet 읽기와 누락 모사만 쓰는 코드가
MediaPipe 없이도 이 모듈을 import할 수 있게 하기 위해서다).

parquet 형식 (MediaPipe에 넣은 프레임마다 한 행, 시각순)
- `t_ms` int64, `face` bool, `w` int32, `h` int32 (MediaPipe에 넘긴, 세운 영상의 크기)
- `lm{인덱스}_x`, `lm{인덱스}_y`: 랜드마크 부분집합 44개의 정규화 좌표
- `bs_{이름}`: 블렌드셰이프 52개 전부(규약을 바꿔도 MediaPipe를 다시 돌리지 않기 위해서다)
- `m0`..`m15`: 변환 행렬의 열 우선 16개 값(`M[i][j] = m[4*j + i]`)
- `face_w_px`: 얼굴 폭(픽셀). 품질 통계용
- 실수 열은 float64다(MediaPipe의 float32 값을 그대로 담는다). 얼굴이 없는 행의 실수 열은 null이다.
- 파일 메타데이터 `drowsy_extract`(JSON): 원본 경로, 모델 sha256, mediapipe 버전, 누락 모사 설정, 회전,
  프레임 계수, 랜드마크 인덱스 목록, 블렌드셰이프 이름 목록. `extract_video`가 만든 파일에는 녹화 요약도 있다
  (`extract_version`, `width`, `height`, `duration_ms`, `mean_fps`, `min_interval_ms`, `max_interval_ms`,
  `variable_intervals`, `brightness_mean`, `brightness_median`, `display_flip`. 뜻은 `extract_video`의 설명 참고).
  원본 영상은 세 키로 적는다: `source`(추출할 때 받은 경로 그대로), `source_name`(파일 이름), `source_size_bytes`
  (파일 크기, 바이트). 뒤의 둘은 이어 하기에서 이름이 같은 다른 영상의 결과를 가려내는 데 쓴다(`output_is_current`).
  이 두 키가 없는 파일(합성 데이터, 이 키를 적기 전의 추출 결과)도 그대로 읽힌다.
  회전에 관한 키는 셋이다. `rotation`은 적용한 각도, `rotation_override`는 영상의 회전 메타데이터 대신 쓴 각도(수동 값
  이거나 자동 점검이 고른 값. 회전 메타데이터를 그대로 썼으면 null), `rotation_check`는 자동 점검으로 각도를 정한
  결과에만 있는 점검 기록이다(`rotation_check_record`: 후보 각도별 검출 비율과 roll, 가르지 못했는지 `ambiguous`,
  사유 `reason`). 그래서 `rotation_check`가 있으면 자동 점검으로, 없고 `rotation_override`가 null이 아니면 수동 값으로,
  둘 다 없으면 회전 메타데이터만으로(이 판의 명령은 그렇게 추출하지 않는다) 만든 결과다. 자동 점검이 고른 각도를
  사람이 나중에 보정표에 같은 값으로 적어 확인하면 다시 추출하지 않으므로 점검 기록이 그대로 남는다.
가정: 계획서는 저장할 내용만 정했고 열 이름, 자료형, 메타데이터 키는 정하지 않았다. 위 형식으로 고정하고
형식을 바꾸면 SCHEMA_VERSION을 올린다. 읽기는 `read_raw_frames`만 쓰면 되므로 다른 모듈은 열 이름에 의존하지 않는다.
가정: 녹화 요약은 열이 아니라 메타데이터 키로 적는다. 표의 열은 그대로이므로 SCHEMA_VERSION을 올리지 않고
(합성 데이터와 `read_raw_frames`가 그대로 유효하다) 메타데이터 내용의 판은 `extract_version`으로 따로 센다.

명령 (저장소 루트에서 실행)
    python -m drowsy.extract --config config/default.yaml --video path.mp4 [--rotation 90]
    python -m drowsy.extract --config config/default.yaml --input-dir videos/ --workers 8
    python -m drowsy.extract --config config/default.yaml --input-dir videos/ --fps-sim 15
    python -m drowsy.extract --config config/default.yaml --dataset --pilot 6     # 시험 추출(M3)
    python -m drowsy.extract --config config/default.yaml --dataset [--fps-sim 15] [--run-name NAME]
    python -m drowsy.extract --config config/default.yaml --unlabeled [--fps-sim 15]   # 팀 수집 영상(M4)

`--dataset`은 설정의 `paths.rldd_root`에서 목록(`dataset.build_manifest`)을 만들어 그 영상들을 추출한다.
결과는 목록의 배치를 그대로 따른다(`raw_dir/<native 또는 누락 모사 폴더>/<영상 상대 경로>.parquet`).
`--unlabeled`는 `paths.unlabeled_root`의 영상(`dataset.scan_unlabeled`)을 `unlabeled_raw_dir`에 추출한다.
네 입력(`--video`, `--input-dir`, `--dataset`, `--unlabeled`) 모두 영상마다 회전을 자동으로 점검한다(PLAN 12장).
설정의 수동 회전 보정표에 있는 영상과 `--video`에 `--rotation`으로 각도를 준 영상은 점검 없이 그 값을 쓴다.
`--video`를 뺀 모든 실행은 `reports_dir/<실행명>/extraction/`에 영상별 표(CSV)와 요약(Markdown)을 쓴다
(`write_extraction_report`). 얼굴 화소는 어디에도 저장하지 않는다(PLAN 1.2절의 결정 D13).
이미 결과가 있는 영상은 건너뛴다(`skip` 줄). 결과에 적힌 원본 파일 크기가 지금 받은 영상과 다르면 다른 영상으로 보고
다시 추출하며(`replace` 줄), 건너뛴 결과에 적힌 원본 경로가 지금 받은 영상의 경로와 다르면 `skip` 줄이 그 경로를 알려 준다.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import difflib
import importlib.metadata
import json
import math
import multiprocessing
import os
import shlex
import shutil
import statistics
import sys
import time
from array import array
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from drowsy import signals
from drowsy.config import DropSimConfig, ExperimentConfig, check_run_name, load_config
from drowsy.spec import DEFAULT_FEATURE_SPEC_PATH, PipelineSpec, SpecError, file_sha256, load_feature_spec
from drowsy.types import RawFrame
from drowsy.video_io import (
    CONSTANT_INTERVAL_TOLERANCE_MS,
    DecodeCounts,
    VideoError,
    iter_video_frames,
    normalize_rotation,
    probe_video,
    rotate_upright,
)

if TYPE_CHECKING:
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerResult

METADATA_KEY = b"drowsy_extract"
SCHEMA_VERSION = 1
# `extract_video`가 적는 메타데이터 내용의 판(표의 열 형식인 SCHEMA_VERSION과 별개다).
# 2: 녹화 요약(크기, 길이, 평균 프레임률, 프레임 간격, 밝기, 표시 행렬 뒤집기)을 적는다.
# 판이 다른 결과는 이어 하기에서 최신으로 보지 않고 다시 만든다.
EXTRACT_VERSION = 2
# 원본 영상의 파일 이름과 크기(바이트)를 적는 메타데이터 키. 이 키를 더한 것으로는 EXTRACT_VERSION을 올리지 않는다:
# 키가 없는 전의 결과는 전과 같이 다루기로 했기 때문이다(`output_is_current`의 설명 참고).
SOURCE_NAME_KEY = "source_name"
SOURCE_SIZE_KEY = "source_size_bytes"
# 밝기에 쓰는 luma 가중치(BT.601): Y = 0.299 R + 0.587 G + 0.114 B
LUMA_WEIGHTS: tuple[float, float, float] = (0.299, 0.587, 0.114)
MATRIX_VALUES = 16
READ_BATCH_ROWS = 2048
# PLAN 2.4절: RLDD의 확장자는 .mp4, .MP4, .mov, .MOV, .m4v가 섞여 있다. 대소문자를 구분하지 않고 찾는다.
VIDEO_SUFFIXES: tuple[str, ...] = (".mp4", ".mov", ".m4v")

_MODEL_FIX = (
    "Fix: put the official face_landmarker.task (float16, version 1) at that path. "
    "If the app really ships a different file (open question Q2 in section 1.1 of docs/PLAN.md), "
    "update pipeline.mediapipe.model_sha256 in "
    "config/feature_spec.json and run `python -m drowsy.spec update-hash`."
)
_NATIVE_FIX = (
    "Fix: install the pinned environment (`uv sync` gives mediapipe==0.10.32, which needs no extra system "
    "library). Newer mediapipe releases need libGLESv2.so.2 and libEGL.so.1: `sudo apt install libgles2 libegl1`."
)

T = TypeVar("T")


class ExtractError(RuntimeError):
    """추출을 진행할 수 없을 때(모델 파일 불일치, MediaPipe 로드 실패, 빈 영상, 형식이 다른 parquet)."""


@dataclass(frozen=True, slots=True)
class ExtractStats:
    """영상 하나의 추출 결과 요약. n_decoded = n_skipped + n_dropped + n_processed다."""

    source: str  # 원본 영상 경로
    out_parquet: str
    width: int  # 처음 처리한 프레임의 세운 뒤 크기
    height: int
    rotation: int  # 적용한 회전(반시계, 0·90·180·270)
    n_decoded: int  # 디코더가 내준 프레임 수
    n_skipped: int  # 타임스탬프가 증가하지 않아 버린 프레임 수
    n_dropped: int  # 프레임 누락 모사가 버린 프레임 수(모사를 안 하면 0)
    n_processed: int  # MediaPipe에 넣은 프레임 수 = parquet 행 수
    n_face: int  # 얼굴이 검출된 프레임 수
    first_t_ms: int  # 처리한 첫 프레임과 마지막 프레임의 t_ms
    last_t_ms: int
    elapsed_s: float  # 디코딩, 추론, 저장을 합친 실제 소요 시간


# ---------------------------------------------------------------------------
# spec, 모델 파일, 랜드마커
# ---------------------------------------------------------------------------


def default_pipeline_spec() -> PipelineSpec:
    """저장소의 config/feature_spec.json에서 읽은 PipelineSpec."""
    repo_root = Path(__file__).resolve().parents[1]
    return load_feature_spec(repo_root / DEFAULT_FEATURE_SPEC_PATH).pipeline


def verify_model_file(model_path: Path, spec: PipelineSpec) -> str:
    """모델 파일의 sha256이 spec의 값과 같은지 검사하고 그 값을 돌려준다. 다르면 ExtractError.

    앱이 쓰는 모델 파일이 공개 표준 파일과 같은지는 아직 답을 받지 못한 질문이다(PLAN 1.1절의 Q2). 그래서 추출을
    시작할 때마다 검사한다.
    """
    if not Path(model_path).is_file():
        raise ExtractError(f"MediaPipe model file not found: {model_path}. {_MODEL_FIX}")
    actual = file_sha256(model_path)
    expected = spec.mediapipe.model_sha256
    if actual != expected:
        raise ExtractError(
            f"MediaPipe model sha256 mismatch for {model_path}: expected {expected}, got {actual}. {_MODEL_FIX}"
        )
    return actual


def _import_mediapipe() -> Any:
    try:
        import mediapipe
    except (ImportError, OSError) as exc:
        raise ExtractError(f"MediaPipe could not be imported ({exc}). {_NATIVE_FIX}") from exc
    return mediapipe


def mediapipe_version() -> str:
    """설치된 mediapipe 배포판의 버전. 패키지를 import하지 않는다."""
    try:
        return importlib.metadata.version("mediapipe")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ExtractError(f"mediapipe is not installed. {_NATIVE_FIX}") from exc


def make_landmarker(model_path: Path, spec: PipelineSpec | None = None) -> FaceLandmarker:
    """모델 파일의 sha256을 spec과 대조한 뒤 VIDEO 모드 FaceLandmarker를 만든다.

    설정은 앱이 MediaPipe에 프레임을 넘길 때의 전제(PLAN 1.1절의 질문 Q3)와 같다: 얼굴 1개(1이 아니면 시간 평활이
    꺼진다), 블렌드셰이프와 변환 행렬 출력.
    IMAGE 모드는 결과가 달라서 쓰지 않는다. 추적과 평활 상태가 있으므로 영상마다 새로 만들고, 다 쓰면 close()한다.
    spec이 None이면 저장소의 config/feature_spec.json을 읽는다.
    """
    if spec is None:
        spec = default_pipeline_spec()
    verify_model_file(model_path, spec)
    mp = _import_mediapipe()
    try:
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=spec.mediapipe.num_faces,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        return mp.tasks.vision.FaceLandmarker.create_from_options(options)
    except (ImportError, OSError) as exc:
        raise ExtractError(f"MediaPipe native library could not be loaded ({exc}). {_NATIVE_FIX}") from exc


def result_to_raw_frame(t_ms: int, result: FaceLandmarkerResult, w: int, h: int, spec: PipelineSpec) -> RawFrame:
    """FaceLandmarker 결과를 앱이 받는 것과 같은 내용의 RawFrame으로 옮긴다.

    - 얼굴 여부는 결과 리스트가 비었는지로만 판정한다. 얼굴이 없으면 lm, bs는 빈 dict, m은 빈 튜플이다.
    - lm: `spec.landmarks.required_indices()`의 정규화 좌표 (x, y).
    - bs: 결과에 들어 있는 블렌드셰이프 전부를 category_name으로 적는다(52개).
    - m: 4x4 행렬을 Android의 float[16]과 같은 열 우선 순서로 편 것. `M.T.reshape(-1)`을 쓴다
      (`M.reshape(-1)`은 행 우선이어서 전치 행렬로 읽히고 각도 부호가 뒤집힌다).
    """
    if len(result.face_landmarks) == 0:
        return RawFrame(t_ms=t_ms, face=False, w=w, h=h, lm={}, bs={}, m=())
    if len(result.face_blendshapes) == 0 or len(result.facial_transformation_matrixes) == 0:
        raise ExtractError(
            "FaceLandmarker result has landmarks but no blendshapes or no transformation matrix. "
            "Fix: create the landmarker with make_landmarker (both outputs enabled)."
        )
    landmarks = result.face_landmarks[0]
    required = spec.landmarks.required_indices()
    if len(landmarks) <= required[-1]:
        raise ExtractError(
            f"FaceLandmarker returned {len(landmarks)} landmarks but index {required[-1]} is required "
            "(the iris points 468..477 come only from the 478-landmark model)"
        )
    lm = {index: (float(landmarks[index].x), float(landmarks[index].y)) for index in required}
    bs = {category.category_name: float(category.score) for category in result.face_blendshapes[0]}
    matrix = np.asarray(result.facial_transformation_matrixes[0])
    if matrix.shape != (4, 4):
        raise ExtractError(f"transformation matrix must be 4x4, got shape {matrix.shape}")
    m = tuple(float(value) for value in matrix.T.reshape(-1))
    return RawFrame(t_ms=t_ms, face=True, w=w, h=h, lm=lm, bs=bs, m=m)


def detect_frame(landmarker: FaceLandmarker, t_ms: int, rgb: np.ndarray, spec: PipelineSpec) -> RawFrame:
    """세운 RGB 프레임 [H, W, 3] 하나를 VIDEO 모드로 추론해 RawFrame으로 돌려준다.

    t_ms는 양수이고 같은 랜드마커에 넣은 이전 값보다 커야 한다(아니면 MediaPipe가 ValueError를 낸다).
    """
    mp = _import_mediapipe()
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = landmarker.detect_for_video(image, t_ms)
    return result_to_raw_frame(t_ms, result, rgb.shape[1], rgb.shape[0], spec)


# ---------------------------------------------------------------------------
# 프레임 누락 모사 (LIVE_STREAM의 흐름 제한기)
# ---------------------------------------------------------------------------


class FlowLimiter(Generic[T]):
    """LIVE_STREAM 흐름 제한기(처리 중 1개, 대기 1개)를 한 프레임씩 흉내 낸다 (PLAN 5.2절 증강 (1)).

    - 처리 중인 프레임이 없으면 도착한 프레임을 바로 처리한다. 처리는 도착 시각에 시작해 지연만큼 걸린다.
    - 처리 중이면 도착한 프레임을 대기 칸에 둔다. 대기 칸에 이미 있던 프레임은 새 프레임으로 바뀌고 버려진다.
    - 처리가 끝나는 순간 대기 중인 프레임이 있으면 그 프레임의 처리를 바로 시작한다.

    처리된 프레임은 자기 원래 타임스탬프를 그대로 갖는다(LIVE_STREAM 콜백이 입력 타임스탬프를 돌려주는 것과 같다).
    VIDEO와 LIVE_STREAM은 같은 프레임과 타임스탬프에서 결과가 같으므로, 고른 프레임만 VIDEO 모드에 넣으면
    그 누락 양상에서의 기기 동작이 된다.
    가정: 프레임은 자기 타임스탬프 시각에 도착하고, 처리 시간은 호출자가 주는 지연 값이다(실제 추론 시간이 아니다).
    가정: 처리가 끝나는 시각과 프레임 도착 시각이 정확히 같으면 처리가 먼저 끝난 것으로 본다.
    가정: 입력이 끝났을 때 대기 중인 프레임은 처리된다(`flush`).
    """

    def __init__(self) -> None:
        self._busy = False
        self._busy_until = 0.0  # 처리 중인 프레임이 끝나는 시각(ms)
        self._queued: tuple[T, float] | None = None  # 대기 중인 (프레임, 그 프레임의 처리 지연)

    def offer(self, t_ms: int, latency_ms: float, item: T) -> list[T]:
        """프레임 `item`이 t_ms에 도착했다. 이 도착으로 처리가 확정된 프레임을 처리 순서대로 돌려준다.

        latency_ms는 이 프레임을 처리하게 될 경우에 걸리는 시간이다. 도착 시각은 엄격히 증가해야 한다.
        결과는 0~2개다(직전까지 대기하던 프레임, 그리고 이 프레임).
        """
        if latency_ms < 0.0:
            raise ValueError(f"latency_ms must be >= 0, got {latency_ms}")
        now = float(t_ms)
        started: list[T] = []
        while self._busy and self._busy_until <= now:
            self._busy = False
            if self._queued is not None:
                queued_item, queued_latency = self._queued
                self._queued = None
                started.append(queued_item)
                self._busy = True
                self._busy_until += queued_latency  # 앞 프레임이 끝난 시각에 바로 시작한다
        if self._busy:
            self._queued = (item, latency_ms)
        else:
            started.append(item)
            self._busy = True
            self._busy_until = now + latency_ms
        return started

    def flush(self) -> list[T]:
        """입력이 끝났다. 대기 중이던 프레임이 있으면 돌려준다."""
        if self._queued is None:
            return []
        queued_item = self._queued[0]
        self._queued = None
        return [queued_item]


def flow_limiter_select(t_ms: Sequence[int], latency_ms: Sequence[float]) -> list[int]:
    """LIVE_STREAM 흐름 제한기가 처리했을 프레임의 인덱스(오름차순).

    t_ms는 프레임 도착 시각(엄격히 증가), latency_ms[i]는 프레임 i를 처리할 때 걸리는 시간(ms)이다.
    """
    if len(t_ms) != len(latency_ms):
        raise ValueError(f"t_ms and latency_ms must have the same length, got {len(t_ms)} and {len(latency_ms)}")
    limiter: FlowLimiter[int] = FlowLimiter()
    selected: list[int] = []
    for index, (arrival, latency) in enumerate(zip(t_ms, latency_ms)):
        selected.extend(limiter.offer(arrival, latency, index))
    selected.extend(limiter.flush())
    return selected


def drop_sim_latencies(drop_sim: DropSimConfig) -> Iterator[float]:
    """도착 프레임마다 하나씩 쓰는 처리 지연(ms)의 끝없는 수열.

    가정: 평균 latency_ms_mean, 표준편차 latency_ms_std인 정규분포에서 뽑고 음수는 0으로 자른다.
    난수열은 drop_sim.seed만으로 정해지므로 모든 영상이 같은 지연 수열을 쓴다.
    """
    rng = np.random.default_rng(drop_sim.seed)
    while True:
        yield max(0.0, float(rng.normal(drop_sim.latency_ms_mean, drop_sim.latency_ms_std)))


def simulate_frame_drops(
    frames: Iterable[tuple[int, np.ndarray]], drop_sim: DropSimConfig
) -> Iterator[tuple[int, np.ndarray]]:
    """(t_ms, 그림) 수열에서 흐름 제한기가 처리했을 프레임만 시각순으로 낸다. 고른 프레임의 t_ms는 그대로다.

    `flow_limiter_select(시각들, drop_sim_latencies의 앞부분)`이 고르는 프레임과 같다. 그림은 대기 칸의 한 장만 들고 있는다.
    """
    limiter: FlowLimiter[tuple[int, np.ndarray]] = FlowLimiter()
    latencies = drop_sim_latencies(drop_sim)
    for frame in frames:
        yield from limiter.offer(frame[0], next(latencies), frame)
    yield from limiter.flush()


def _drop_sim_metadata(drop_sim: DropSimConfig | None) -> dict[str, Any] | None:
    """누락 모사 설정 중 결과를 바꾸는 값. 저장 폴더 이름(subdir)은 결과를 바꾸지 않으므로 뺀다."""
    if drop_sim is None:
        return None
    return {
        "latency_ms_mean": drop_sim.latency_ms_mean,
        "latency_ms_std": drop_sim.latency_ms_std,
        "seed": drop_sim.seed,
    }


# ---------------------------------------------------------------------------
# parquet 쓰기와 읽기
# ---------------------------------------------------------------------------


class _TableBuilder:
    """RawFrame을 열 버퍼에 쌓았다가 parquet 한 파일로 쓴다. 행마다 dict를 들고 있지 않아 메모리가 작다."""

    def __init__(self, spec: PipelineSpec) -> None:
        self._landmark_indices = spec.landmarks.required_indices()
        self._blendshape_names = spec.blendshapes.order
        self._face_width = spec.landmarks.face_width
        self._t_ms = array("q")
        self._face = array("b")
        self._w = array("i")
        self._h = array("i")
        self._lm = array("d")  # 행마다 2·(랜드마크 수)개: x, y, x, y, ...
        self._bs = array("d")  # 행마다 블렌드셰이프 수만큼
        self._bs_missing = array("b")  # 같은 모양. 1이면 그 칸은 null
        self._m = array("d")  # 행마다 16개
        self._face_w_px = array("d")
        self.n_rows = 0
        self.n_face = 0

    def add(self, frame: RawFrame) -> None:
        if not frame.face:
            # 얼굴 없는 행의 실수 칸은 쓸 때 null로 가린다. 여기 넣는 0.0은 자리만 채운다.
            self._lm.extend([0.0] * (2 * len(self._landmark_indices)))
            self._bs.extend([0.0] * len(self._blendshape_names))
            self._bs_missing.extend([1] * len(self._blendshape_names))
            self._m.extend([0.0] * MATRIX_VALUES)
            self._face_w_px.append(0.0)
        else:
            missing = [index for index in self._landmark_indices if index not in frame.lm]
            if missing:
                raise ExtractError(f"frame t_ms={frame.t_ms} lacks required landmark indices {missing}")
            unknown = [name for name in frame.bs if name not in self._blendshape_names]
            if unknown:
                raise ExtractError(f"frame t_ms={frame.t_ms} has blendshape names outside the spec order: {unknown}")
            if len(frame.m) != MATRIX_VALUES:
                raise ExtractError(f"frame t_ms={frame.t_ms} needs {MATRIX_VALUES} matrix values, got {len(frame.m)}")
            for index in self._landmark_indices:
                x, y = frame.lm[index]
                self._lm.append(x)
                self._lm.append(y)
            for name in self._blendshape_names:
                present = name in frame.bs
                self._bs.append(frame.bs[name] if present else 0.0)
                self._bs_missing.append(0 if present else 1)
            self._m.extend(frame.m)
            ax, ay = frame.lm[self._face_width[0]]
            bx, by = frame.lm[self._face_width[1]]
            self._face_w_px.append(math.hypot((bx - ax) * frame.w, (by - ay) * frame.h))
            self.n_face += 1
        self._t_ms.append(frame.t_ms)
        self._face.append(1 if frame.face else 0)
        self._w.append(frame.w)
        self._h.append(frame.h)
        self.n_rows += 1

    def write(self, out_parquet: Path, metadata: Mapping[str, Any]) -> None:
        """parquet을 쓴다. 임시 파일에 쓴 뒤 이름을 바꾸므로 중간에 죽어도 반쪽 파일이 남지 않는다."""
        n = self.n_rows
        n_bs = len(self._blendshape_names)
        no_face = np.array(self._face, dtype=np.int8) == 0
        lm = np.array(self._lm, dtype=np.float64).reshape(n, 2 * len(self._landmark_indices))
        bs = np.array(self._bs, dtype=np.float64).reshape(n, n_bs)
        bs_missing = np.array(self._bs_missing, dtype=np.int8).reshape(n, n_bs) != 0
        m = np.array(self._m, dtype=np.float64).reshape(n, MATRIX_VALUES)
        face_w_px = np.array(self._face_w_px, dtype=np.float64)

        names = ["t_ms", "face", "w", "h"]
        columns = [
            pa.array(np.array(self._t_ms, dtype=np.int64)),
            pa.array(~no_face),
            pa.array(np.array(self._w, dtype=np.int32)),
            pa.array(np.array(self._h, dtype=np.int32)),
        ]
        for j, index in enumerate(self._landmark_indices):
            names.append(f"lm{index}_x")
            columns.append(pa.array(lm[:, 2 * j], mask=no_face))
            names.append(f"lm{index}_y")
            columns.append(pa.array(lm[:, 2 * j + 1], mask=no_face))
        for j, name in enumerate(self._blendshape_names):
            names.append(f"bs_{name}")
            columns.append(pa.array(bs[:, j], mask=bs_missing[:, j]))
        for j in range(MATRIX_VALUES):
            names.append(f"m{j}")
            columns.append(pa.array(m[:, j], mask=no_face))
        names.append("face_w_px")
        columns.append(pa.array(face_w_px, mask=no_face))

        full_metadata = dict(metadata)
        full_metadata["schema_version"] = SCHEMA_VERSION
        full_metadata["landmark_indices"] = list(self._landmark_indices)
        full_metadata["blendshape_names"] = list(self._blendshape_names)
        encoded = json.dumps(full_metadata, sort_keys=True, allow_nan=False).encode("utf-8")
        table = pa.Table.from_arrays(columns, names=names).replace_schema_metadata({METADATA_KEY: encoded})

        out_parquet = Path(out_parquet)
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        temp_path = out_parquet.with_name(out_parquet.name + ".tmp")
        # 실수 열은 값이 거의 다 달라서 사전 부호화가 파일만 키운다(실측 약 20%). 끄고 zstd만 쓴다.
        pq.write_table(table, temp_path, compression="zstd", use_dictionary=False)
        os.replace(temp_path, out_parquet)


def write_raw_frames(
    frames: Iterable[RawFrame],
    out_parquet: Path,
    spec: PipelineSpec,
    metadata: Mapping[str, Any] | None = None,
) -> int:
    """RawFrame들을 추출 결과와 같은 형식의 parquet으로 쓰고 행 수를 돌려준다(합성 세션 저장과 테스트용).

    얼굴이 있는 프레임은 `spec.landmarks.required_indices()`의 랜드마크 44개와 행렬 16개 값을 모두 가져야 한다.
    블렌드셰이프는 `spec.blendshapes.order`에 있는 이름만 받고, 프레임에 없는 이름은 null로 저장한다
    (다시 읽으면 그 이름은 dict에 없다). 그 밖의 랜드마크 인덱스는 저장하지 않는다.
    metadata는 JSON으로 바꿀 수 있는 값이어야 하며 파일 메타데이터에 함께 적힌다.
    """
    builder = _TableBuilder(spec)
    for frame in frames:
        builder.add(frame)
    builder.write(out_parquet, metadata if metadata is not None else {})
    return builder.n_rows


def read_extract_metadata(parquet_path: Path) -> dict[str, Any]:
    """추출 parquet의 파일 메타데이터(`drowsy_extract`)를 dict로 읽는다. 없거나 형식 판이 다르면 ExtractError."""
    schema_metadata = pq.read_schema(parquet_path).metadata
    if schema_metadata is None or METADATA_KEY not in schema_metadata:
        raise ExtractError(f"{parquet_path} has no '{METADATA_KEY.decode()}' metadata; not an extraction parquet")
    metadata = json.loads(schema_metadata[METADATA_KEY].decode("utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ExtractError(
            f"{parquet_path} has schema_version {metadata.get('schema_version')}, expected {SCHEMA_VERSION}"
        )
    return metadata


def read_raw_frames(parquet_path: Path) -> Iterator[RawFrame]:
    """추출 parquet을 저장된 순서대로 RawFrame으로 되돌린다. 추출 때 MediaPipe가 낸 값과 정확히 같다.

    랜드마크 인덱스와 블렌드셰이프 이름은 현재 spec이 아니라 파일에 적힌 목록을 따른다.
    파일을 조금씩 읽으므로 긴 영상도 메모리를 많이 쓰지 않는다.
    """
    metadata = read_extract_metadata(parquet_path)
    landmark_indices: list[int] = metadata["landmark_indices"]
    blendshape_names: list[str] = metadata["blendshape_names"]
    with pq.ParquetFile(parquet_path) as parquet_file:
        for batch in parquet_file.iter_batches(batch_size=READ_BATCH_ROWS):
            columns = batch.to_pydict()
            t_ms = columns["t_ms"]
            face = columns["face"]
            width = columns["w"]
            height = columns["h"]
            lm_columns = [(index, columns[f"lm{index}_x"], columns[f"lm{index}_y"]) for index in landmark_indices]
            bs_columns = [(name, columns[f"bs_{name}"]) for name in blendshape_names]
            m_columns = [columns[f"m{j}"] for j in range(MATRIX_VALUES)]
            for row in range(batch.num_rows):
                if not face[row]:
                    yield RawFrame(t_ms=t_ms[row], face=False, w=width[row], h=height[row], lm={}, bs={}, m=())
                    continue
                lm = {index: (xs[row], ys[row]) for index, xs, ys in lm_columns}
                bs = {name: values[row] for name, values in bs_columns if values[row] is not None}
                m = tuple(values[row] for values in m_columns)
                yield RawFrame(t_ms=t_ms[row], face=True, w=width[row], h=height[row], lm=lm, bs=bs, m=m)


# ---------------------------------------------------------------------------
# 영상 하나 추출
# ---------------------------------------------------------------------------


def frame_brightness(rgb: np.ndarray) -> float:
    """세운 RGB 프레임 [H, W, 3](uint8) 하나의 밝기: 프레임 전체 화소의 luma 평균(0~255).

    가정: 밝기 = (0.299·ΣR + 0.587·ΣG + 0.114·ΣB) / (H·W). BT.601 가중치를 디코더가 낸 8비트 RGB 값(감마가
    적용된 값)에 그대로 쓴다. 선형 광량으로 바꾸지 않고, 얼굴 영역이 아니라 MediaPipe에 넘긴 프레임 전체를 본다.
    채널별 합을 정수로 구하므로 같은 프레임이면 값이 언제나 같다. 화소는 저장하지 않는다(PLAN 1.2절의 결정 D13).
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] == 0 or rgb.shape[1] == 0:
        raise ValueError(f"frame_brightness needs an RGB image [H, W, 3], got shape {rgb.shape}")
    weighted = 0.0
    for channel, weight in enumerate(LUMA_WEIGHTS):
        weighted += weight * int(rgb[..., channel].sum(dtype=np.uint64))
    return weighted / (rgb.shape[0] * rgb.shape[1])


@dataclass(slots=True)
class _SourceTiming:
    """디코더가 낸(누락 모사를 하기 전의) 프레임 시각의 요약. 정의는 `video_io.probe_video`와 같다."""

    n_frames: int = 0
    first_t_ms: int = 0
    last_t_ms: int = 0
    min_interval_ms: int = 0  # 프레임이 2개 미만이면 0
    max_interval_ms: int = 0

    def add(self, t_ms: int) -> None:
        if self.n_frames == 0:
            self.first_t_ms = t_ms
        else:
            interval = t_ms - self.last_t_ms
            if self.n_frames == 1 or interval < self.min_interval_ms:
                self.min_interval_ms = interval
            if self.n_frames == 1 or interval > self.max_interval_ms:
                self.max_interval_ms = interval
        self.last_t_ms = t_ms
        self.n_frames += 1

    def summary(self) -> dict[str, Any]:
        duration_ms = self.last_t_ms - self.first_t_ms
        return {
            "duration_ms": duration_ms,
            "mean_fps": (self.n_frames - 1) * 1000.0 / duration_ms if duration_ms > 0 else 0.0,
            "min_interval_ms": self.min_interval_ms,
            "max_interval_ms": self.max_interval_ms,
            "variable_intervals": self.max_interval_ms - self.min_interval_ms > CONSTANT_INTERVAL_TOLERANCE_MS,
        }


def _observe_times(
    frames: Iterable[tuple[int, np.ndarray]], timing: _SourceTiming
) -> Iterator[tuple[int, np.ndarray]]:
    for t_ms, rgb in frames:
        timing.add(t_ms)
        yield t_ms, rgb


def source_file_size(path: Path) -> int | None:
    """원본 영상 파일의 크기(바이트). 읽을 수 없으면(파일이 없다, 권한이 없다) None. 심볼릭 링크는 가리키는 파일의 크기다."""
    try:
        return int(Path(path).stat().st_size)
    except OSError:
        return None


def extract_video(
    path: Path,
    model_path: Path,
    out_parquet: Path,
    drop_sim: DropSimConfig | None = None,
    *,
    spec: PipelineSpec | None = None,
    rotation: int | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> ExtractStats:
    """영상 하나를 VIDEO 모드 FaceLandmarker에 통과시켜 처리한 프레임마다 한 행인 parquet을 쓴다.

    - drop_sim을 주면 흐름 제한기를 흉내 내 고른 프레임만 MediaPipe에 넣는다(`simulate_frame_drops`).
    - spec: None이면 저장소의 config/feature_spec.json을 읽는다.
    - rotation: None이면 영상의 회전 메타데이터를 쓴다. 값을 주면 메타데이터 대신 그 각도(반시계, 90의 배수)로
      돌려 세운다(설정의 수동 회전 보정표, 회전 자동 점검의 결과).
    - extra_metadata: 파일 메타데이터에 함께 적을 값(JSON으로 바꿀 수 있어야 한다). 이 함수가 적는 키와 겹치면 ValueError.
    랜드마커는 이 호출 안에서 만들고 닫는다(영상마다 하나). 프레임이 하나도 없거나 도중에 실패하면 파일을 쓰지 않는다.
    표시 행렬에 뒤집기가 든 영상은 rotation을 주지 않으면 VideoError다(`video_io.iter_video_frames`).

    파일 메타데이터에 적는 녹화 요약(교란 점검과 추출 리포트가 쓴다):
    - `width`, `height`: MediaPipe에 넘긴 첫 프레임의 크기(세운 뒤).
    - `duration_ms`, `mean_fps`, `min_interval_ms`, `max_interval_ms`, `variable_intervals`: 디코더가 낸 프레임
      시각(t_ms)으로 구한 길이, 평균 프레임률 (프레임 수 − 1)·1000 / 길이, 프레임 간격의 최소·최대, 간격이 변하는지
      (최대 − 최소 > 1 ms). `video_io.probe_video`와 같은 정의다.
    - `brightness_mean`, `brightness_median`: MediaPipe에 넘긴 프레임마다 구한 `frame_brightness`의 평균과 중앙값.
    - `display_flip`: 표시 행렬에 뒤집기가 든 영상을 수동 회전 값으로 추출했으면 true(거울상은 되돌리지 않았다).
    원본 영상을 가리키는 키: `source`(받은 경로 그대로), `source_name`(파일 이름), `source_size_bytes`(디코딩을 시작하기
    전에 읽은 파일 크기, 바이트. 읽지 못했으면 null). 이어 하기가 뒤의 값으로 다른 영상의 결과를 가려낸다(`output_is_current`).
    가정: 수동 회전 보정 값은 `frame.rotation`과 같은 규약(세우려면 반시계로 돌려야 하는 각도)이다.
    가정: 길이·프레임률·간격은 원본 영상의 것이다. 누락 모사 추출에서도 모사가 버리기 전의 프레임 시각으로 구한다
    (저장된 행의 간격은 `t_ms` 열에서 다시 구할 수 있다). 밝기는 실제로 처리한 프레임만 본다.
    가정: 처리 시간은 메타데이터에 적지 않는다(같은 영상을 다시 추출하면 파일이 바이트 단위로 같아야 한다).
    """
    path = Path(path)
    out_parquet = Path(out_parquet)
    started = time.perf_counter()
    if spec is None:
        spec = default_pipeline_spec()
    rotation_override = None if rotation is None else normalize_rotation(rotation)
    source_size = source_file_size(path)  # 디코딩 전에 읽는다
    landmarker = make_landmarker(model_path, spec)

    counts = DecodeCounts()
    timing = _SourceTiming()
    brightness = array("d")  # 처리한 프레임마다 하나
    builder = _TableBuilder(spec)
    frames = _observe_times(iter_video_frames(path, rotation=rotation_override, counts=counts), timing)
    if drop_sim is not None:
        frames = simulate_frame_drops(frames, drop_sim)
    first: RawFrame | None = None
    last: RawFrame | None = None
    try:
        for t_ms, rgb in frames:
            brightness.append(frame_brightness(rgb))
            last = detect_frame(landmarker, t_ms, rgb, spec)
            if first is None:
                first = last
            builder.add(last)
    finally:
        landmarker.close()
    if first is None or last is None:
        raise ExtractError(f"no decodable video frame in {path}")

    n_dropped = counts.decoded - counts.skipped - builder.n_rows
    metadata = {
        "source": str(path),
        SOURCE_NAME_KEY: path.name,
        SOURCE_SIZE_KEY: source_size,
        "model_sha256": spec.mediapipe.model_sha256,  # make_landmarker가 파일과 같다고 확인한 값
        "mediapipe_version": mediapipe_version(),
        "drop_sim": _drop_sim_metadata(drop_sim),
        "rotation": counts.rotation,
        "rotation_override": rotation_override,
        "n_decoded": counts.decoded,
        "n_skipped": counts.skipped,
        "n_dropped": n_dropped,
        "n_processed": builder.n_rows,
        "n_face": builder.n_face,
        "extract_version": EXTRACT_VERSION,
        "width": first.w,
        "height": first.h,
        **timing.summary(),
        "brightness_mean": statistics.fmean(brightness),
        "brightness_median": statistics.median(brightness),
        "display_flip": counts.display_flip,
    }
    if extra_metadata is not None:
        clashing = sorted(set(extra_metadata) & set(metadata))
        if clashing:
            raise ValueError(f"extra_metadata must not redefine {clashing}")
        metadata.update(extra_metadata)
    builder.write(out_parquet, metadata)
    return ExtractStats(
        source=str(path),
        out_parquet=str(out_parquet),
        width=first.w,
        height=first.h,
        rotation=counts.rotation,
        n_decoded=counts.decoded,
        n_skipped=counts.skipped,
        n_dropped=n_dropped,
        n_processed=builder.n_rows,
        n_face=builder.n_face,
        first_t_ms=first.t_ms,
        last_t_ms=last.t_ms,
        elapsed_s=time.perf_counter() - started,
    )


def output_is_current(
    out_parquet: Path,
    spec: PipelineSpec,
    drop_sim: DropSimConfig | None,
    rotation: int | None,
    source: Path | None = None,
) -> bool:
    """이미 있는 추출 결과를 그대로 써도 되는지(이어 하기용). source는 지금 추출하려는 영상이다(네 입력 모두 준다).

    가정: 파일이 있고, 같은 모델 파일과 mediapipe 버전, 같은 누락 모사 설정, 같은 수동 회전 보정, 같은 랜드마크·
    블렌드셰이프 목록, 같은 메타데이터 판(`EXTRACT_VERSION`)으로 만들었고, 결과에 적힌 원본 파일 크기
    (`source_size_bytes`)가 source의 지금 크기와 같으면 최신으로 본다. 원본 경로는 비교하지 않는다(데이터 폴더를 옮겨도
    다시 추출하지 않는다). 크기가 다르면 이름만 같은 다른 영상이거나 바뀐 파일이므로 다시 추출한다. 크기가 적혀 있는데
    source를 읽을 수 없으면(파일이 없다) 최신으로 보지 않는다(추출을 시도해 그 오류가 드러나게 한다).
    가정: 원본의 내용은 크기로만 비교한다(해시를 구하지 않는다. 이어 하기마다 모든 영상을 끝까지 읽지 않기 위해서다).
    그래서 크기가 바이트 단위로 같은 다른 영상은 가려내지 못한다(예: 같은 영상에 회전 메타데이터만 바꿔 단 파일).
    그런 영상으로 바꿨다면 결과 파일을 지우고 다시 실행한다.
    가정: 원본 파일 크기가 적혀 있지 않은 결과(이 키를 적기 전의 추출 결과)는 전과 같이 다룬다: 크기를 비교하지 않고
    나머지 조건만으로 최신 여부를 정한다. source를 주지 않아도 크기를 비교하지 않는다.
    """
    return _metadata_if_current(out_parquet, spec, drop_sim, rotation, source) is not None


def _metadata_if_current(
    out_parquet: Path, spec: PipelineSpec, drop_sim: DropSimConfig | None, rotation: int | None, source: Path | None
) -> dict[str, Any] | None:
    """`output_is_current`가 참인 결과의 메타데이터, 아니면 None(건너뛰는 줄에 적힌 원본 경로를 쓰려고 돌려준다)."""
    metadata = _current_metadata(out_parquet, spec, drop_sim, source)
    if metadata is None:
        return None
    if metadata.get("rotation_override") != (None if rotation is None else normalize_rotation(rotation)):
        return None
    return metadata


def _current_metadata(
    out_parquet: Path, spec: PipelineSpec, drop_sim: DropSimConfig | None, source: Path | None = None
) -> dict[str, Any] | None:
    """추출 결과가 있고 회전을 뺀 조건이 같으면 그 메타데이터, 아니면 None.

    조건: 모델, mediapipe 버전, 누락 모사, 랜드마크·블렌드셰이프 목록, 메타데이터 판, 그리고 source를 줬고 결과에 원본
    파일 크기가 적혀 있으면 그 크기가 source의 지금 크기와 같을 것(`output_is_current`의 설명 참고).
    """
    if not Path(out_parquet).is_file():
        return None
    try:
        metadata = read_extract_metadata(out_parquet)
    except (ExtractError, OSError, ValueError, pa.ArrowException):
        return None
    same = (
        metadata.get("extract_version") == EXTRACT_VERSION
        and metadata.get("model_sha256") == spec.mediapipe.model_sha256
        and metadata.get("mediapipe_version") == mediapipe_version()
        and metadata.get("drop_sim") == _drop_sim_metadata(drop_sim)
        and metadata.get("landmark_indices") == list(spec.landmarks.required_indices())
        and metadata.get("blendshape_names") == list(spec.blendshapes.order)
    )
    if not same:
        return None
    recorded_size = _recorded_source_size(metadata)
    if source is not None and recorded_size is not None and source_file_size(source) != recorded_size:
        return None
    return metadata


def _recorded_source_size(metadata: Mapping[str, Any]) -> int | None:
    """결과의 메타데이터에 적힌 원본 파일 크기(바이트). 적혀 있지 않으면(전의 결과, 합성 데이터) None."""
    value = metadata.get(SOURCE_SIZE_KEY)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


# ---------------------------------------------------------------------------
# 영상별 회전 점검 (PLAN 12장)
# ---------------------------------------------------------------------------

ROTATION_CANDIDATES: tuple[int, ...] = (0, 90, 180, 270)
ROTATION_CHECK_KEY = "rotation_check"  # 자동 점검으로 회전을 정한 추출 결과의 메타데이터 키
# 점검 규칙의 판. 2: 얼굴 검출 비율이 비슷한 각도들을 머리 roll로 가르고, 가르지 못하면 표시한다.
# 판이나 점검 설정이 다른 결과는 이어 하기에서 다시 만든다.
ROTATION_CHECK_VERSION = 2
# `RotationCheck.reason`의 값
REASON_BEST_RATE = "best_rate"  # 얼굴 검출 비율만으로 정했다
REASON_SMALLEST_ROLL = "smallest_roll"  # 비율이 비슷한 각도 중 |roll 중앙값|이 뚜렷이 가장 작은 각도
REASON_KEPT_METADATA = "ambiguous_kept_metadata"  # roll로 가르지 못한 각도들 안에 회전 메타데이터가 있어 그것을 유지
REASON_AMBIGUOUS_ROLL = "ambiguous_smallest_roll"  # 가르지 못했고 회전 메타데이터는 그 안에 없다. |roll|이 가장 작은 각도
REASON_NO_FACE = "no_face"  # 어느 각도에서도 얼굴을 못 찾아 회전 메타데이터를 유지
_RATE_EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class RotationChoice:
    """`choose_rotation`의 결과."""

    rotation: int  # 고른 회전
    ambiguous: bool  # 점검이 각도를 가르지 못했다. 영상을 보고 수동 회전 보정표에 적어야 한다
    reason: str  # REASON_* 중 하나


@dataclass(frozen=True, slots=True)
class RotationCheck:
    """영상 하나의 회전 점검 결과. 각도는 `frame.rotation`과 같은 규약(세우려면 반시계로 돌려야 하는 각도)이다."""

    rotation: int  # 고른 회전
    metadata_rotation: int  # 영상의 회전 메타데이터
    rates: dict[int, float]  # 후보 회전 → 표본 프레임 중 얼굴이 검출된 비율
    rolls: dict[int, float | None]  # 후보 회전 → 얼굴이 검출된 표본의 머리 roll 중앙값(도). 검출이 없으면 None
    n_frames: int  # 표본 프레임 수
    ambiguous: bool  # 점검이 각도를 가르지 못했다(`RotationChoice.ambiguous`)
    reason: str  # REASON_* 중 하나


def sample_frames(
    frames: Iterable[tuple[int, np.ndarray]], n_frames: int, interval_ms: int
) -> list[tuple[int, np.ndarray]]:
    """(t_ms, 그림) 수열에서 interval_ms마다 한 장씩, 최대 n_frames장을 고른다.

    가정: 첫 프레임을 고르고, 그 뒤로는 직전에 고른 프레임보다 interval_ms 이상 늦은 첫 프레임을 고른다.
    필요한 만큼 고르면 수열을 더 읽지 않는다(영상을 끝까지 디코딩하지 않는다).
    """
    if n_frames < 1 or interval_ms < 1:
        raise ValueError("n_frames and interval_ms must be >= 1")
    samples: list[tuple[int, np.ndarray]] = []
    for t_ms, rgb in frames:
        if not samples or t_ms - samples[-1][0] >= interval_ms:
            samples.append((t_ms, rgb))
            if len(samples) == n_frames:
                break
    return samples


def median_angle(angles: Sequence[float]) -> float:
    """각도(도)들의 중앙값을 (−180, 180] 범위로 돌려준다. ±180 근처에 걸친 값도 바르게 다룬다.

    값들의 평균 방향을 기준으로 편차(−180 이상 180 미만)의 중앙값을 구해 기준에 더한다. 값들이 180도 경계를
    넘지 않으면 보통의 중앙값과 같다(예: 179와 −179의 중앙값은 0이 아니라 180이다).
    """
    if len(angles) == 0:
        raise ValueError("median_angle needs at least one angle")
    sin_sum = sum(math.sin(math.radians(angle)) for angle in angles)
    cos_sum = sum(math.cos(math.radians(angle)) for angle in angles)
    center = math.degrees(math.atan2(sin_sum, cos_sum)) if math.hypot(sin_sum, cos_sum) > 1e-9 else angles[0]
    deviations = [(angle - center + 180.0) % 360.0 - 180.0 for angle in angles]
    wrapped = (center + statistics.median(deviations) + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def rotation_rates_and_rolls(
    samples: Sequence[tuple[int, np.ndarray]], detect: Callable[[int, int, np.ndarray], float | None]
) -> tuple[dict[int, float], dict[int, float | None]]:
    """후보 회전(0, 90, 180, 270)마다 표본 프레임 중 얼굴이 검출된 비율과, 검출된 표본의 머리 roll 중앙값(도).

    samples는 돌리지 않은 원본 프레임이다. `detect(rotation, t_ms, 그 각도로 돌린 그림)`은 얼굴을 찾았으면 그 얼굴의
    roll(도)을, 못 찾았으면 None을 돌려준다. 검출기를 인자로 받으므로 MediaPipe 없이 시험할 수 있다.
    표본이 없으면 모든 비율이 0.0이고, 검출이 하나도 없는 후보의 roll은 None이다.
    """
    rates: dict[int, float] = {}
    rolls: dict[int, float | None] = {}
    for rotation in ROTATION_CANDIDATES:
        found: list[float] = []
        for t_ms, rgb in samples:
            roll = detect(rotation, t_ms, rotate_upright(rgb, rotation))
            if roll is not None:
                found.append(float(roll))
        rates[rotation] = len(found) / len(samples) if samples else 0.0
        rolls[rotation] = median_angle(found) if found else None
    return rates, rolls


def choose_rotation(
    rates: Mapping[int, float],
    rolls: Mapping[int, float | None],
    metadata_rotation: int,
    n_frames: int,
    roll_margin_deg: float,
) -> RotationChoice:
    """후보별 얼굴 검출 비율과 머리 roll로 회전을 고른다 (PLAN 12장).

    1. 얼굴 검출 비율이 가장 높은 각도가 첫 기준이다. 최고 비율과의 차이가 표본 한 장(1 / n_frames) 이내이고
       검출이 한 번이라도 있는 각도를 동률로 본다(표본 한 장의 검출 실패로 결과가 바뀌지 않게 한다).
    2. 동률이 여럿이면 |roll 중앙값|이 가장 작은 각도를 고른다. 똑바로 선 얼굴의 roll은 0 근처이고 옆으로 누운
       얼굴은 ±90 근처다.
    3. 가장 작은 |roll|과의 차이가 roll_margin_deg 미만인 동률 각도가 또 있으면 가르지 못한 것이다(`ambiguous`).
       그 각도들 안에 회전 메타데이터가 있으면 회전 메타데이터를 유지하고, 없으면 |roll|이 가장 작은 각도를
       고른다(|roll|까지 같으면 작은 각도). 어느 쪽이든 영상을 보고 수동 회전 보정표에 적어야 한다.
    4. 어느 각도에서도 얼굴을 못 찾았으면 회전 메타데이터를 유지하고 `ambiguous`로 표시한다.

    가정: 실제 랜드마커는 옆으로 누운 얼굴과 거꾸로 선 얼굴도 검출하므로(공개 시험 이미지 2장으로 잰 것) 검출
    비율만으로는 방향을 가를 수 없다. 거꾸로 선 얼굴이 검출되면 똑바로 선 얼굴로 잘못 맞춰져 roll이 정방향과 몇 도
    안에서 같게 나온다. 그런 영상에서 이 점검은 고른 각도와 그 반대 각도(0도와 180도)를 가르지 못한다. 그때는
    가른 것처럼 다루지 않고 `ambiguous`로 표시해 사람이 확인하게 한다.
    가정: 3번에서 회전 메타데이터가 가르지 못한 각도들 안에 없을 때(메타데이터의 각도는 roll이 ±90 근처여서 이미
    틀린 것으로 드러난 경우) 메타데이터를 따르지 않고 |roll|이 가장 작은 각도를 쓴다. 추출하려면 각도 하나를 정해야
    해서 고른 것일 뿐 근거가 약하다(맞는 각도이거나 그 반대 각도다). 그래서 `ambiguous`로 표시되고 리포트의 맨 앞
    목록에 오른다.
    가정: 어느 각도에서도 얼굴을 못 찾은 영상(4번)도 `ambiguous`로 표시한다. 점검이 아무것도 확인하지 못했기 때문이다.
    """
    if n_frames < 1 or max(rates[rotation] for rotation in ROTATION_CANDIDATES) <= 0.0:
        return RotationChoice(metadata_rotation, True, REASON_NO_FACE)
    best = max(rates[rotation] for rotation in ROTATION_CANDIDATES)
    one_sample = 1.0 / n_frames
    tied = [
        rotation
        for rotation in ROTATION_CANDIDATES
        if rolls[rotation] is not None and rates[rotation] >= best - one_sample - _RATE_EPSILON
    ]
    if not tied:  # 비율은 있는데 roll이 없다(호출자가 준 값이 서로 맞지 않는다)
        return RotationChoice(metadata_rotation, True, REASON_NO_FACE)
    if len(tied) == 1:
        return RotationChoice(tied[0], False, REASON_BEST_RATE)
    abs_roll = {rotation: abs(roll) for rotation in tied if (roll := rolls[rotation]) is not None}
    smallest = min(abs_roll.values())
    close = [rotation for rotation in tied if abs_roll[rotation] - smallest < roll_margin_deg]
    if len(close) == 1:
        return RotationChoice(close[0], False, REASON_SMALLEST_ROLL)
    if metadata_rotation in close:
        return RotationChoice(metadata_rotation, True, REASON_KEPT_METADATA)
    return RotationChoice(min(close, key=lambda rotation: (abs_roll[rotation], rotation)), True, REASON_AMBIGUOUS_ROLL)


def check_rotation(
    path: Path, model_path: Path, spec: PipelineSpec, n_frames: int, interval_ms: int, roll_margin_deg: float
) -> RotationCheck:
    """영상 앞부분의 표본 프레임을 네 방향으로 돌려 얼굴 검출 비율과 머리 roll을 재고 회전을 고른다 (PLAN 12장).

    후보마다 VIDEO 모드 랜드마커를 하나씩 만들어 표본을 시각순으로 넣는다(추출과 같은 모드와 설정).
    roll은 변환 행렬에서 `signals.euler_from_colmajor`로 구한다(추출 결과가 쓰는 것과 같은 식).
    고르는 규칙은 `choose_rotation`에 있다. 표시 행렬에 뒤집기가 든 영상은 VideoError다(회전 메타데이터를
    믿을 수 없으므로 수동 회전 보정표에 적어야 한다).
    가정: 표본은 영상 앞부분에서만 뽑는다(interval_ms 간격으로 n_frames장). 영상 전체를 디코딩하지 않기 위해서다.
    """
    path = Path(path)
    counts = DecodeCounts()
    with contextlib.closing(iter_video_frames(path, counts=counts)) as upright:
        if next(upright, None) is None:
            raise ExtractError(f"no decodable video frame in {path}")
    metadata_rotation = counts.rotation
    with contextlib.closing(iter_video_frames(path, rotation=0)) as raw_frames:
        samples = sample_frames(raw_frames, n_frames, interval_ms)

    landmarkers: dict[int, Any] = {}

    def detect(rotation: int, t_ms: int, rgb: np.ndarray) -> float | None:
        if rotation not in landmarkers:
            landmarkers[rotation] = make_landmarker(model_path, spec)
        frame = detect_frame(landmarkers[rotation], t_ms, rgb, spec)
        return signals.euler_from_colmajor(frame.m)[2] if frame.face else None

    try:
        rates, rolls = rotation_rates_and_rolls(samples, detect)
    finally:
        for landmarker in landmarkers.values():
            landmarker.close()
    choice = choose_rotation(rates, rolls, metadata_rotation, len(samples), roll_margin_deg)
    return RotationCheck(
        rotation=choice.rotation,
        metadata_rotation=metadata_rotation,
        rates=rates,
        rolls=rolls,
        n_frames=len(samples),
        ambiguous=choice.ambiguous,
        reason=choice.reason,
    )


# ---------------------------------------------------------------------------
# 명령: 작업, 입력 목록, 수동 회전 보정표
# ---------------------------------------------------------------------------

UNLABELED_PREFIX = "unlabeled_"  # raw_dir 아래 라벨 없는 영상의 추출 결과 폴더 이름 앞에 붙인다
OVERRIDES_KEY = "extraction.rotation_overrides"
UNLABELED_OVERRIDES_KEY = "extraction.unlabeled_rotation_overrides"


@dataclass(frozen=True, slots=True)
class RotationCheckSettings:
    """회전 자동 점검의 설정(`extraction.rotation_check_frames`, `rotation_check_interval_ms`, `rotation_roll_margin_deg`)."""

    frames: int  # 표본 프레임 수
    interval_ms: int  # 표본 프레임 사이의 간격
    roll_margin_deg: float  # |roll 중앙값|의 차이가 이 값 미만인 각도들은 가르지 못한 것으로 본다

    def as_metadata(self) -> dict[str, Any]:
        return {"frames": self.frames, "interval_ms": self.interval_ms, "roll_margin_deg": self.roll_margin_deg}


@dataclass(frozen=True, slots=True)
class _Job:
    label: str  # 보고에 쓰는 영상 상대 경로
    video: Path
    out_parquet: Path
    model_path: Path
    spec: PipelineSpec
    drop_sim: DropSimConfig | None
    rotation: int | None  # 수동 회전 보정 값. None이면 회전 메타데이터(또는 아래의 자동 점검 결과)를 쓴다
    rotation_check: RotationCheckSettings | None = None  # rotation이 None일 때 이 설정으로 자동 점검을 한다
    rotation_from_command: bool = False  # rotation이 보정표가 아니라 `--video`의 `--rotation`에서 온 값인지


def rotation_check_record(check: RotationCheck, settings: RotationCheckSettings) -> dict[str, Any]:
    """자동 점검 결과를 추출 결과의 메타데이터(`rotation_check` 키)에 적는 꼴로 바꾼다. 데모의 이벤트 JSON도 같은 꼴을 쓴다.

    키: `version`(점검 규칙의 판), `settings`(표본 수, 간격, roll 여유), `rates`와 `rolls`(후보 각도 "0", "90", "180",
    "270"마다 얼굴 검출 비율과 roll 중앙값. 검출이 없으면 roll은 null), `metadata_rotation`(영상의 회전 메타데이터),
    `n_frames`(표본 수), `ambiguous`(가르지 못했는지), `reason`(REASON_* 중 하나). 고른 각도는 이 기록이 아니라
    메타데이터의 `rotation`과 `rotation_override`에 있다.
    """
    return {
        "version": ROTATION_CHECK_VERSION,
        "settings": settings.as_metadata(),
        "rates": {str(rotation): check.rates[rotation] for rotation in ROTATION_CANDIDATES},
        "rolls": {str(rotation): check.rolls[rotation] for rotation in ROTATION_CANDIDATES},
        "metadata_rotation": check.metadata_rotation,
        "n_frames": check.n_frames,
        "ambiguous": check.ambiguous,
        "reason": check.reason,
    }


def _run_job(job: _Job) -> tuple[ExtractStats, RotationCheck | None]:
    """영상 하나를 추출한다. 회전 자동 점검을 했으면 그 결과도 돌려주고 파일 메타데이터에 적는다."""
    if job.rotation is not None or job.rotation_check is None:
        stats = extract_video(
            job.video, job.model_path, job.out_parquet, job.drop_sim, spec=job.spec, rotation=job.rotation
        )
        return stats, None
    settings = job.rotation_check
    check = check_rotation(
        job.video, job.model_path, job.spec, settings.frames, settings.interval_ms, settings.roll_margin_deg
    )
    stats = extract_video(
        job.video,
        job.model_path,
        job.out_parquet,
        job.drop_sim,
        spec=job.spec,
        rotation=check.rotation,
        extra_metadata={ROTATION_CHECK_KEY: rotation_check_record(check, settings)},
    )
    return stats, check


def _job_is_current(job: _Job) -> bool:
    """이 작업의 결과가 이미 있는지(이어 하기용). 규칙은 `_job_current_metadata`에 있다."""
    return _job_current_metadata(job) is not None


def _job_current_metadata(job: _Job) -> dict[str, Any] | None:
    """이 작업의 결과가 이미 있으면(이어 하기에서 건너뛸 수 있으면) 그 결과의 메타데이터, 아니면 None.

    어느 갈래든 결과에 적힌 원본 파일 크기가 이 작업의 영상과 다르면 최신이 아니다(`output_is_current`).
    가정: 회전을 자동으로 점검하는 작업은, 같은 점검 규칙 판(`ROTATION_CHECK_VERSION`)과 같은 점검 설정으로 자동
    점검을 거쳐 만든 결과가 있으면 최신으로 본다(점검을 다시 돌려 각도를 비교하지 않는다. 같은 영상의 점검 결과는 같다).
    점검 규칙이나 설정이 바뀌면 다시 추출한다. 점검 기록이 없는 결과(회전 메타데이터만 써서 만든 것, 수동 값으로 만든
    것)도 다시 추출한다.
    가정: 각도를 `--rotation`으로 준 작업은 수동 값으로 만든 결과(점검 기록이 없고 각도가 같은 것)만 최신으로 본다.
    자동 점검으로 만든 결과는 각도가 같아도 다시 추출한다. `--rotation`은 그 실행의 명령줄에만 있으므로, 사람이
    각도를 확인했다는 사실을 결과 파일에 남기는 길이 그것뿐이기 때문이다. 보정표로 준 각도는 설정 파일에 남으므로
    각도만 같으면 최신으로 보고, 그때 파일의 점검 기록은 그대로 남는다(`output_is_current`).
    """
    if job.rotation is not None or job.rotation_check is None:
        metadata = _metadata_if_current(job.out_parquet, job.spec, job.drop_sim, job.rotation, job.video)
        if metadata is None or (job.rotation_from_command and ROTATION_CHECK_KEY in metadata):
            return None
        return metadata
    metadata = _current_metadata(job.out_parquet, job.spec, job.drop_sim, job.video)
    if metadata is None:
        return None
    recorded = metadata.get(ROTATION_CHECK_KEY)
    current = (
        isinstance(recorded, dict)
        and recorded.get("version") == ROTATION_CHECK_VERSION
        and recorded.get("settings") == job.rotation_check.as_metadata()
    )
    return metadata if current else None


def _same_path(first: str, second: Path) -> bool:
    """두 경로가 같은 곳을 가리키는지. 절대 경로로 바꿔 글자로 비교한다(심볼릭 링크는 풀지 않는다)."""
    return os.path.abspath(first) == os.path.abspath(second)


def _skip_note(job: _Job, metadata: Mapping[str, Any]) -> str:
    """건너뛴 결과에 적힌 원본 경로가 이 작업의 영상과 다를 때 `skip` 줄 끝에 붙이는 말(영어). 같으면 빈 문자열.

    가정: "다르다"는 절대 경로로 바꾼 글자열이 다르다는 뜻이다. 결과에 상대 경로가 적혀 있으면 지금의 실행 위치를
    기준으로 푼다(명령은 저장소 루트에서 실행한다고 본다). 심볼릭 링크는 풀지 않으므로, 같은 파일을 가리키는 다른
    경로도 다르다고 알린다. 데이터 폴더를 옮긴 뒤의 이어 하기에서는 건너뛴 영상마다 옛 경로가 붙는다.
    """
    recorded = metadata.get("source")
    if not isinstance(recorded, str) or recorded == "" or _same_path(recorded, job.video):
        return ""
    size = _recorded_source_size(metadata)
    if size is None:
        return f" (extracted from {recorded}; the result records no source size, so the two files were not compared)"
    return f" (extracted from {recorded}, a file of the same size as this video: {size} bytes)"


def _replace_note(job: _Job) -> str | None:
    """있는 결과를 다시 추출하는 까닭이 원본 파일 크기일 때 알리는 한 줄(영어). 그런 경우가 아니면 None.

    결과 파일이 있고, 거기 적힌 원본 파일 크기가 이 작업의 영상과 다를 때다(이름만 같은 다른 영상이거나 바뀐 파일).
    결과 파일은 하나뿐이므로 다시 추출하면 전의 영상의 결과는 없어진다. 그 사실을 사용자가 알 수 있게 한다.
    """
    if not job.out_parquet.is_file():
        return None
    try:
        metadata = read_extract_metadata(job.out_parquet)
    except (ExtractError, OSError, ValueError, pa.ArrowException):
        return None
    recorded_size = _recorded_source_size(metadata)
    size = source_file_size(job.video)
    if recorded_size is None or size is None or size == recorded_size:
        return None
    return (
        f"replace {job.label}: {job.out_parquet} was extracted from {metadata.get('source')} "
        f"({recorded_size} bytes); this video has {size} bytes, so it is extracted again and that result is overwritten"
    )


def find_videos(input_dir: Path) -> list[Path]:
    """폴더 아래의 영상 파일(VIDEO_SUFFIXES, 대소문자 무시)을 경로순으로 찾는다."""
    found = [p for p in Path(input_dir).rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES]
    return sorted(found)


def raw_parquet_path(out_root: Path, relative_video: Path) -> Path:
    """영상의 상대 경로에 대응하는 추출 결과 경로: 같은 폴더 구조에 확장자만 `.parquet`으로 바꾼다.

    가정: 계획서는 raw_dir 아래의 배치를 정하지 않았다. 확장자만 다른 두 영상(`a.mp4`, `a.mov`)은 같은 경로가
    되므로 명령이 시작 전에 오류로 알린다.
    """
    return Path(out_root) / Path(relative_video).with_suffix(".parquet")


def unlabeled_raw_dir(cfg: ExperimentConfig, fps_sim: bool = False) -> Path:
    """`--unlabeled` 추출 결과의 루트: `<paths.raw_dir>/unlabeled_<라벨 있는 추출과 같은 하위 폴더 이름>`.

    하위 폴더 이름은 원본 프레임률이면 `extraction.native_subdir`, fps_sim이면 `extraction.drop_sim.subdir`다
    (기본 설정에서 `data/raw/unlabeled_native`, `data/raw/unlabeled_fps15`). 영상별 파일은
    `raw_parquet_path(unlabeled_raw_dir(cfg), 영상 상대 경로)`에 있다.
    가정: 라벨 없는 영상의 결과를 라벨 있는 데이터셋의 결과와 같은 폴더에 섞지 않는다. `--fps-sim`에 설정과 다른
    프레임률을 주면 추출은 `unlabeled_fps{FPS}`에 쓰므로 이 함수가 가리키는 폴더와 달라진다(`drop_sim_for_fps`).
    """
    subdir = cfg.extraction.drop_sim.subdir if fps_sim else cfg.extraction.native_subdir
    return cfg.paths.raw_dir / f"{UNLABELED_PREFIX}{subdir}"


def drop_sim_for_fps(base: DropSimConfig, fps: int) -> DropSimConfig:
    """`--fps-sim FPS`에 쓸 누락 모사 설정.

    가정: FPS가 설정의 지연 평균이 뜻하는 프레임률(1000 / latency_ms_mean을 반올림한 값)과 같으면 설정을
    그대로 쓴다. 다르면 지연 평균을 1000 / FPS로, 저장 폴더를 `fps{FPS}`로 바꾸고 표준편차와 시드는 유지한다.
    """
    if fps <= 0:
        raise ExtractError(f"--fps-sim must be positive, got {fps}")
    if math.floor(1000.0 / base.latency_ms_mean + 0.5) == fps:
        return base
    return replace(base, subdir=f"fps{fps}", latency_ms_mean=1000.0 / fps)


def _closest_labels(key: str, labels: Sequence[str], limit: int = 3) -> list[str]:
    """보정표의 키와 가장 비슷한 영상 경로: 대소문자만 다른 것과 폴더 깊이만 다른 것. 그런 것이 없으면 철자가 비슷한 것."""
    lowered = key.lower()
    found: list[str] = []
    for label in labels:
        if label.lower() == lowered and label not in found:
            found.append(label)
    for label in labels:
        other = label.lower()
        if (other.endswith("/" + lowered) or lowered.endswith("/" + other)) and label not in found:
            found.append(label)
    if not found:
        found = difflib.get_close_matches(key, labels, n=limit, cutoff=0.6)
    return found[:limit]


def check_override_keys(overrides: Mapping[str, int], labels: Sequence[str], table: str) -> None:
    """수동 회전 보정표의 키가 모두 이번 입력의 영상 경로와 정확히 맞는지 검사한다. 맞지 않는 키가 있으면 ExtractError.

    labels는 이번 입력의 모든 영상 경로다(`--dataset --pilot`에서도 목록 전체). 키마다 가장 비슷한 경로를 함께 알려 준다.
    가정: 맞지 않는 키를 조용히 무시하지 않는다. 보정표는 회전이 틀린 영상을 고치는 유일한 수단이어서, 대소문자나
    폴더 깊이를 틀리게 적은 키가 무시되면 그 영상은 틀린 각도로 추출되고 실행은 성공으로 끝난다.
    """
    known = set(labels)
    unmatched = [key for key in overrides if key not in known]
    if not unmatched:
        return
    described: list[str] = []
    for key in unmatched:
        closest = _closest_labels(key, labels)
        hint = "closest: " + ", ".join(repr(label) for label in closest) if closest else "no similar video"
        described.append(f"{key!r} ({hint})")
    raise ExtractError(
        f"{table} has {len(unmatched)} key(s) that match no video of this run: {'; '.join(described)}. "
        "Fix: keys are video paths relative to the source root of the run (slash separated, exact case and "
        "extension; the file name for --video). Correct the keys, or use a config whose table lists only the "
        "videos of this source"
    )


def _format_stats(stats: ExtractStats) -> str:
    face_ratio = stats.n_face / stats.n_processed
    ms_per_frame = 1000.0 * stats.elapsed_s / stats.n_processed
    return (
        f"{stats.width}x{stats.height} rot={stats.rotation} frames={stats.n_processed} "
        f"face={face_ratio:.3f} skipped={stats.n_skipped} dropped={stats.n_dropped} "
        f"t_ms={stats.first_t_ms}..{stats.last_t_ms} {ms_per_frame:.1f} ms/frame"
    )


def rotation_check_details(check: RotationCheck) -> str:
    """자동 점검이 잰 값 한 줄(명령의 출력용, 영어): 회전 메타데이터, 사유, 후보별 얼굴 검출 비율과 roll 중앙값."""
    rates = " ".join(f"{rotation}:{check.rates[rotation]:.2f}" for rotation in ROTATION_CANDIDATES)
    rolls = " ".join(
        f"{rotation}:{'-' if check.rolls[rotation] is None else format(check.rolls[rotation], '.1f')}"
        for rotation in ROTATION_CANDIDATES
    )
    return (
        f"metadata {check.metadata_rotation}; {check.reason}; "
        f"face rate over {check.n_frames} frames {rates}; median roll {rolls}"
    )


def _format_rotation(job: _Job, check: RotationCheck | None, ambiguous_fix: str) -> str:
    """회전을 어떻게 정했는지. 자동 점검을 했으면 후보별 얼굴 검출 비율과 roll 중앙값, 가르지 못했는지를 함께 적는다.

    ambiguous_fix는 점검이 가르지 못했을 때 붙이는, 사용자가 할 일(영어 문장)이다.
    """
    if check is None:
        if job.rotation is None:
            return "rotation: video metadata"
        return "rotation: manual override (--rotation)" if job.rotation_from_command else "rotation: manual override"
    text = f"rotation: auto check chose {check.rotation} ({rotation_check_details(check)})"
    if check.ambiguous:
        text += f" WARNING rotation ambiguous: {ambiguous_fix}"
    elif check.rotation != check.metadata_rotation:
        text += " WARNING rotation differs from the video metadata"
    return text


def _dataset_videos(cfg: ExperimentConfig) -> tuple[Path, Any]:
    """`--dataset`: 설정의 rldd_root에서 목록을 만들어 (영상 루트, 목록 DataFrame)을 돌려준다."""
    # dataset 모듈이 이 모듈을 import하므로 여기서 import한다.
    from drowsy.dataset import RAW_SUFFIX, ManifestError, build_manifest

    root = cfg.paths.rldd_root
    if root is None:
        raise ExtractError(
            "paths.rldd_root is null in the config. Fix: set it to the UTA-RLDD copy, "
            "or use --input-dir / --video for other videos"
        )
    try:
        manifest = build_manifest(root)
    except ManifestError as exc:
        raise ExtractError(f"cannot build the dataset manifest: {exc}") from exc
    if Path(manifest["rel_path"].iloc[0]).suffix.lower() == RAW_SUFFIX:
        raise ExtractError(f"paths.rldd_root ({root}) holds raw parquet files, not videos; nothing to extract")
    return Path(root), manifest


def _unlabeled_videos(cfg: ExperimentConfig) -> tuple[Path, list[Path]]:
    """`--unlabeled`: 설정의 unlabeled_root를 `dataset.scan_unlabeled`로 훑어 (영상 루트, 영상 상대 경로들)을 돌려준다."""
    from drowsy.dataset import RAW_SUFFIX, ManifestError, scan_unlabeled

    root = cfg.paths.unlabeled_root
    if root is None:
        raise ExtractError(
            "paths.unlabeled_root is null in the config. Fix: set it to the folder of unlabeled videos, "
            "or use --input-dir / --video"
        )
    try:
        listed = scan_unlabeled(root)
    except ManifestError as exc:
        raise ExtractError(f"cannot list the unlabeled videos: {exc}") from exc
    relative_paths = [Path(rel_path) for rel_path in listed["rel_path"].tolist()]
    if relative_paths[0].suffix.lower() == RAW_SUFFIX:
        raise ExtractError(f"paths.unlabeled_root ({root}) holds raw parquet files, not videos; nothing to extract")
    return Path(root), relative_paths


# ---------------------------------------------------------------------------
# 추출 리포트 (PLAN 11장 M3)
# ---------------------------------------------------------------------------

REPORT_SUBDIR = "extraction"  # reports_dir/<실행명>/ 아래의 폴더
DEFAULT_RUN_NAME = "extraction"
STATUS_DONE = "done"  # 이번 실행에서 추출했다
STATUS_UP_TO_DATE = "up_to_date"  # 같은 설정으로 만든 결과가 이미 있어 건너뛰었다
STATUS_FAILED = "failed"  # 이번 실행에서 실패했다(결과 파일 없음)
STATUS_PENDING = "pending"  # 실행이 중간에 끊겨 처리되지 않았다
SOURCE_MANUAL = "manual_override"  # 리포트의 rotation_source 값
SOURCE_AUTO = "auto_check"
SOURCE_METADATA = "video_metadata"
WARN_AMBIGUOUS = "rotation_ambiguous"  # 리포트의 warnings 값
WARN_DIFFERS = "rotation_differs_from_metadata"
WARN_FLIP = "display_flip"
WARN_LOW_FACE = "low_face_rate"
WARN_TIMESTAMPS = "timestamps_not_increasing"
# 기준값: UTA-RLDD 전체 사본의 파일 수와 논리 영상 수(PLAN 2.4절. 피험자 32, 49의 10이 두 부분으로 나뉘어 있다)
RLDD_FILES = 182
RLDD_LOGICAL_VIDEOS = 180

REPORT_COLUMNS: tuple[str, ...] = (
    "video",  # 입력 루트 기준 영상 상대 경로
    "status",  # done | up_to_date | failed | pending
    "reason",  # 실패(제외) 사유: 예외 종류와 메시지
    "warnings",  # 세미콜론으로 이은 주의 표시(WARN_*)
    "video_id",  # 아래 다섯 열은 --dataset에서만: 논리 영상, 세션, 피험자, 라벨, 부분 번호
    "session_id",
    "subject",
    "label",
    "part",
    "out_parquet",
    "frames_decoded",  # 디코더가 내준 프레임 수
    "frames_skipped",  # 타임스탬프가 증가하지 않아 버린 수
    "frames_dropped",  # 프레임 누락 모사가 버린 수
    "frames_kept",  # MediaPipe에 넣은 프레임 수 = parquet 행 수
    "frames_face",  # 얼굴이 검출된 프레임 수
    "face_rate",  # frames_face / frames_kept
    "duration_ms",  # 원본 영상의 마지막 프레임 t_ms − 첫 프레임 t_ms
    "mean_fps",  # 원본 영상의 평균 프레임률
    "min_interval_ms",
    "max_interval_ms",
    "variable_intervals",  # 프레임 간격이 변하는지(최대 − 최소 > 1 ms)
    "width",  # MediaPipe에 넘긴(세운 뒤) 크기
    "height",
    "rotation",  # 적용한 회전(반시계)
    "rotation_source",  # manual_override | auto_check | video_metadata
    "rotation_reason",  # 자동 점검의 사유(REASON_*)
    "metadata_rotation",  # 영상의 회전 메타데이터(자동 점검을 한 영상만)
    "rotation_ambiguous",  # 자동 점검이 각도를 가르지 못했는지
    "check_frames",  # 자동 점검의 표본 프레임 수
    "rate_0",  # 후보 각도별 얼굴 검출 비율
    "rate_90",
    "rate_180",
    "rate_270",
    "roll_0",  # 후보 각도별 머리 roll 중앙값(도)
    "roll_90",
    "roll_180",
    "roll_270",
    "display_flip",  # 표시 행렬에 뒤집기가 든 영상인지
    "elapsed_s",  # 영상 하나의 처리 시간(랜드마커 생성 + 디코딩 + 추론 + 저장, 회전 점검 제외)
    "ms_per_frame",  # 1000 · elapsed_s / frames_kept
    "brightness_mean",  # 처리한 프레임의 밝기(`frame_brightness`) 평균과 중앙값
    "brightness_median",
    "t_ms_ok",  # 저장된 t_ms가 양수이고 엄격히 증가하는지
    "first_t_ms",
    "last_t_ms",
    "model_sha256",  # MediaPipe 모델 파일의 sha256
    "mediapipe_version",
)


@dataclass(frozen=True, slots=True)
class ReportContext:
    """추출 리포트의 머리말에 쓰는 실행 정보."""

    description: str  # 첫 줄: 데이터 출처
    command: str
    config_path: str
    out_root: str
    drop_sim: DropSimConfig | None
    model_sha256: str
    mediapipe_version: str
    rotation_check: RotationCheckSettings | None  # 자동 점검을 한 실행이면 그 설정
    overrides_key: str  # 수동 회전 보정표의 설정 키
    min_face_ratio: float  # 영상 품질 기준(PLAN 1.2절의 결정 D7)의 얼굴 검출 비율. 주의 표시에만 쓴다
    workers: int  # 이번 실행이 동시에 처리한 영상 수의 상한. 1이면 차례로 처리했다
    pilot_of: int | None = None  # 시험 추출이면 목록 전체의 파일 수
    manifest_totals: tuple[int, int, int] | None = None  # --dataset: 목록 전체의 (파일 수, 논리 영상 수, 피험자 수)


@dataclass(slots=True)
class _Entry:
    """입력 영상 하나의 진행 상태."""

    job: _Job
    status: str
    reason: str = ""
    stats: ExtractStats | None = None


def _blank_row(label: str, out_parquet: Path) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(REPORT_COLUMNS)
    row["video"] = label
    row["out_parquet"] = str(out_parquet)
    return row


def _fill_from_result(row: dict[str, Any], out_parquet: Path, manual_rotation: bool) -> None:
    """추출 결과 파일의 메타데이터와 t_ms 열로 리포트 행을 채운다(이번에 만든 결과와 전에 만든 결과를 같은 방법으로 읽는다)."""
    metadata = read_extract_metadata(out_parquet)
    t_ms = pq.read_table(out_parquet, columns=["t_ms"]).column("t_ms").to_numpy()
    kept = int(metadata["n_processed"])
    row["frames_decoded"] = metadata.get("n_decoded")
    row["frames_skipped"] = metadata.get("n_skipped")
    row["frames_dropped"] = metadata.get("n_dropped")
    row["frames_kept"] = kept
    row["frames_face"] = metadata.get("n_face")
    row["face_rate"] = int(metadata["n_face"]) / kept if kept > 0 else None
    for name in (
        "duration_ms",
        "mean_fps",
        "min_interval_ms",
        "max_interval_ms",
        "variable_intervals",
        "width",
        "height",
        "rotation",
        "display_flip",
        "brightness_mean",
        "brightness_median",
        "model_sha256",
        "mediapipe_version",
    ):
        row[name] = metadata.get(name)
    recorded = metadata.get(ROTATION_CHECK_KEY)
    if isinstance(recorded, dict):  # 전에 자동 점검으로 만든 결과를 수동 값으로 확인한 경우에도 점검 수치는 남긴다
        row["metadata_rotation"] = recorded.get("metadata_rotation")
        row["check_frames"] = recorded.get("n_frames")
        row["rotation_reason"] = recorded.get("reason")
        for rotation in ROTATION_CANDIDATES:
            row[f"rate_{rotation}"] = (recorded.get("rates") or {}).get(str(rotation))
            row[f"roll_{rotation}"] = (recorded.get("rolls") or {}).get(str(rotation))
    if manual_rotation:
        row["rotation_source"] = SOURCE_MANUAL
        row["rotation_ambiguous"] = False  # 사람이 정한 값이다
    elif isinstance(recorded, dict):
        row["rotation_source"] = SOURCE_AUTO
        row["rotation_ambiguous"] = bool(recorded.get("ambiguous"))
    else:
        row["rotation_source"] = SOURCE_METADATA
        row["rotation_ambiguous"] = False
    row["t_ms_ok"] = bool(len(t_ms) > 0 and t_ms[0] >= 1 and np.all(np.diff(t_ms) > 0))
    row["first_t_ms"] = int(t_ms[0]) if len(t_ms) > 0 else None
    row["last_t_ms"] = int(t_ms[-1]) if len(t_ms) > 0 else None


def _fill_from_probe(row: dict[str, Any], video: Path) -> None:
    """실패한 영상의 길이, 프레임률, 크기를 `probe_video`로 읽을 수 있으면 채운다(읽지 못하면 비워 둔다)."""
    try:
        info = probe_video(video)
    except Exception:  # 디코딩 자체가 안 되는 파일이다. 사유는 이미 reason에 있다
        return
    row["duration_ms"] = info.duration_ms
    row["mean_fps"] = info.mean_fps
    row["min_interval_ms"] = info.min_interval_ms
    row["max_interval_ms"] = info.max_interval_ms
    row["variable_intervals"] = info.variable_intervals
    row["display_flip"] = info.display_flip
    if not info.display_flip:  # 뒤집기가 있으면 libav가 알려 주는 회전과 그에 따른 크기를 믿을 수 없다
        row["width"] = info.width
        row["height"] = info.height
        row["metadata_rotation"] = info.rotation


def _row_warnings(row: Mapping[str, Any], min_face_ratio: float) -> str:
    """리포트 행의 주의 표시(WARN_*)를 세미콜론으로 이어 돌려준다.

    가정: `low_face_rate`는 프레임 기준 얼굴 검출 비율이 영상 품질 기준(PLAN 1.2절의 결정 D7: 얼굴 검출 비율과 창의
    수로 평가 대상 영상을 거른다)의 값(`evaluation.quality_gate.min_face_ratio`) 미만일 때 붙인다. 그 기준 자체는
    시간 기준 비율이고 평가 때 계산하므로, 이 표시는 미리 알려 주는 용도일 뿐 영상을 제외하지 않는다.
    """
    warnings: list[str] = []
    if row["rotation_ambiguous"]:
        warnings.append(WARN_AMBIGUOUS)
    elif row["rotation_source"] == SOURCE_AUTO and row["rotation"] != row["metadata_rotation"]:
        warnings.append(WARN_DIFFERS)
    if row["display_flip"]:
        warnings.append(WARN_FLIP)
    if row["face_rate"] is not None and row["face_rate"] < min_face_ratio:
        warnings.append(WARN_LOW_FACE)
    if row["t_ms_ok"] is False:
        warnings.append(WARN_TIMESTAMPS)
    return ";".join(warnings)


def _previous_timings(csv_path: Path) -> dict[str, tuple[str, str, str]]:
    """전에 쓴 리포트 하나의 영상별 (frames_kept, elapsed_s, ms_per_frame). 처리 시간이 적힌 행만. 파일이 없으면 빈 dict."""
    if not csv_path.is_file():
        return {}
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            return {
                row["video"]: (row["frames_kept"], row["elapsed_s"], row["ms_per_frame"])
                for row in csv.DictReader(handle)
                if row.get("video") and row.get("elapsed_s") and row.get("ms_per_frame")
            }
    except (OSError, KeyError, csv.Error, UnicodeDecodeError):
        return {}


# 리포트 행의 `_timing_carried` 값(CSV의 열이 아니다): 이어 온 처리 시간이 어느 리포트에 적혀 있던 값인지
CARRIED_FROM_OWN = "own_report"  # 같은 이름으로 전에 쓴 리포트(이 리포트의 전 판)
CARRIED_FROM_PILOT = "pilot_report"  # 같은 폴더의 시험 추출 리포트(`--dataset`의 리포트만)


@dataclass(frozen=True, slots=True)
class PreviousTimings:
    """이어 하기에서 건너뛴 영상의 처리 시간을 이어 올 수 있는, 전에 쓴 리포트들의 값 (`read_previous_timings`).

    - own: 같은 이름으로 전에 쓴 리포트(이 리포트의 전 판)의 영상 → (frames_kept, elapsed_s, ms_per_frame). CSV의 글자 그대로다.
    - pilot: 같은 폴더의 시험 추출 리포트의 값. `--dataset`의 리포트가 아니면 빈 dict다.
    - pilot_is_newer: 시험 추출 리포트 파일이 이 리포트의 전 판보다 나중에 쓰였는지(전 판이 없으면 True).
    """

    own: Mapping[str, tuple[str, str, str]]
    pilot: Mapping[str, tuple[str, str, str]]
    pilot_is_newer: bool

    def lookup(self, video: str, frames_kept: int) -> tuple[float, float, str] | None:
        """건너뛴 영상 하나의 (elapsed_s, ms_per_frame, 출처 CARRIED_FROM_*). 이어 올 값이 없으면 None.

        frames_kept는 지금 있는 결과 파일의 행 수다. 프레임 수가 다른 행은 다른 추출의 값이므로 쓰지 않는다.
        가정: 두 리포트에 그 영상의 값이 다 있으면 나중에 쓴 리포트의 값을 쓴다. "나중"은 이번 실행이 리포트를 다시
        쓰기 전에 읽은 두 CSV 파일의 수정 시각(ns)으로 정한다(같은 폴더의 두 파일이므로 같은 시계다). 시험 추출은
        결과가 있어도 다시 추출하므로, 전체 추출 뒤에 시험 추출을 다시 했다면 지금 있는 결과 파일은 시험 추출이 만든
        것이고 그 처리 시간은 시험 추출 리포트에 있다.
        가정: 두 리포트의 값이 글자까지 같으면 시험 추출에서 잰 값으로 본다(전체 추출의 리포트는 건너뛴 영상의 값을
        시험 추출 리포트에서 옮겨 적으므로). 이어 하기는 이 리포트를 다시 써서 수정 시각을 새로 만들지만 옮겨 적은 값은
        그대로이므로, 이어 하기를 몇 번 거듭해도 값과 출처가 바뀌지 않는다. 따로 잰 두 값이 유효 숫자 6자리까지 우연히
        같으면 출처만 시험 추출로 적힌다(값은 같다).
        """
        wanted = str(frames_kept)
        own = self.own.get(video)
        pilot = self.pilot.get(video)
        if own is not None and own[0] != wanted:
            own = None
        if pilot is not None and pilot[0] != wanted:
            pilot = None
        if pilot is not None and (own is None or own == pilot or self.pilot_is_newer):
            return float(pilot[1]), float(pilot[2]), CARRIED_FROM_PILOT
        if own is not None:
            return float(own[1]), float(own[2]), CARRIED_FROM_OWN
        return None


def _modified_ns(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def read_previous_timings(own_csv: Path, pilot_csv: Path | None = None) -> PreviousTimings:
    """전에 쓴 리포트(같은 이름의 것, 그리고 pilot_csv를 주면 시험 추출의 것)에서 영상별 처리 시간을 읽는다.

    이번 실행이 own_csv를 다시 쓰기 전에 불러야 한다(두 파일의 수정 시각을 비교한다. `PreviousTimings.lookup`).
    두 파일의 수정 시각이 같으면 시험 추출 리포트를 더 나중으로 보지 않는다.
    """
    own = _previous_timings(own_csv)
    pilot = {} if pilot_csv is None else _previous_timings(pilot_csv)
    own_ns = _modified_ns(own_csv)
    pilot_ns = None if pilot_csv is None else _modified_ns(pilot_csv)
    pilot_is_newer = pilot_ns is not None and (own_ns is None or pilot_ns > own_ns)
    return PreviousTimings(own, pilot, pilot_is_newer)


def _report_rows(
    entries: Sequence[_Entry],
    dataset_info: Mapping[str, Mapping[str, Any]],
    previous: PreviousTimings,
    min_face_ratio: float,
) -> list[dict[str, Any]]:
    """입력 영상마다 한 행. 추출 결과가 있는 행(done, up_to_date)은 결과 파일에서 읽어 채운다.

    가정: 처리 시간은 결과 파일에 없으므로, 이번에 추출한 영상은 이번 실행의 값을, 건너뛴 영상(up_to_date)은 전에 쓴
    리포트(같은 이름의 리포트, `--dataset`이면 같은 폴더의 시험 추출 리포트도)에 그 영상의 값이 있고 프레임 수가 같을 때
    그 값을 쓴다(둘 다 있으면 `PreviousTimings.lookup`의 규칙). 없으면 비워 둔다. 이번 실행이 추출했는데 처리 시간을
    받지 못한 영상(실행이 끊김)에는 전의 값을 잇지 않는다: 전 리포트의 값은 지금의 결과 파일을 만든 시간이 아니다.
    """
    rows: list[dict[str, Any]] = []
    version = mediapipe_version()
    for entry in entries:
        job = entry.job
        row = _blank_row(job.label, job.out_parquet)
        row.update(dataset_info.get(job.label, {}))
        row["status"] = entry.status
        row["reason"] = entry.reason
        # 결과가 없는 행(실패, 미처리)에는 이번 실행이 쓰려던 모델과 버전을 적는다. 결과가 있는 행은 파일에 적힌 값으로 바뀐다.
        row["model_sha256"] = job.spec.mediapipe.model_sha256
        row["mediapipe_version"] = version
        if entry.status in (STATUS_DONE, STATUS_UP_TO_DATE):
            try:
                _fill_from_result(row, job.out_parquet, job.rotation is not None)
            except (ExtractError, OSError, KeyError, ValueError, pa.ArrowException) as exc:
                row["reason"] = f"cannot read the result file: {type(exc).__name__}: {exc}"
            if entry.stats is not None:
                row["elapsed_s"] = entry.stats.elapsed_s
                row["ms_per_frame"] = 1000.0 * entry.stats.elapsed_s / entry.stats.n_processed
            elif entry.status == STATUS_UP_TO_DATE and row["frames_kept"] is not None:
                carried = previous.lookup(job.label, row["frames_kept"])
                if carried is not None:
                    # `_timing_carried`는 CSV의 열이 아니다. 요약이 이번에 잰 값과 구분하고 출처별로 나누는 데만 쓴다
                    row["elapsed_s"], row["ms_per_frame"], row["_timing_carried"] = carried
        elif entry.status == STATUS_FAILED:
            _fill_from_probe(row, job.video)
        row["warnings"] = _row_warnings(row, min_face_ratio)
        rows.append(row)
    return rows


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".6g")
    return str(value)


def _cell(value: Any, digits: int = 3) -> str:
    """요약 표의 칸: 없는 값은 `-`, 참·거짓은 예/아니오, 실수는 소수 digits자리."""
    if value is None or value == "":
        return "-"
    if isinstance(value, (bool, np.bool_)):
        return "예" if value else "아니오"
    if isinstance(value, float):
        return format(value, f".{digits}f")
    return str(value).replace("|", "\\|")


def _table(header: Sequence[str], body: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(cells) + " |" for cells in body)
    return lines


def _per_angle(row: Mapping[str, Any], prefix: str, digits: int) -> str:
    return " / ".join(_cell(row[f"{prefix}_{rotation}"], digits) for rotation in ROTATION_CANDIDATES)


def _rotation_table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    body = [
        [
            f"`{row['video']}`",
            _cell(row["rotation"]),
            _cell(row["metadata_rotation"]),
            _cell(row["rotation_reason"]),
            _per_angle(row, "rate", 2),
            _per_angle(row, "roll", 1),
        ]
        for row in rows
    ]
    return _table(
        ["영상", "고른 회전", "회전 메타데이터", "사유", "검출 비율 0 / 90 / 180 / 270", "roll 중앙값 0 / 90 / 180 / 270"],
        body,
    )


def _range_text(values: Sequence[float], digits: int) -> str:
    if not values:
        return "-"
    return (
        f"최소 {min(values):.{digits}f}, 중앙값 {statistics.median(values):.{digits}f}, "
        f"최대 {max(values):.{digits}f}"
    )


def _timing_text(rows: Sequence[Mapping[str, Any]]) -> str:
    """처리 시간이 있는 행들의 요약: 영상 수, 프레임 수, 프레임당 시간(합친 시간 / 합친 프레임 수), 영상별 값의 범위."""
    frames = sum(int(row["frames_kept"]) for row in rows)
    elapsed_ms = 1000.0 * sum(float(row["elapsed_s"]) for row in rows)
    per_video = [float(row["ms_per_frame"]) for row in rows]
    return (
        f"영상 {len(rows)}개, 프레임 {frames}개, 프레임당 {elapsed_ms / frames:.2f} ms"
        f" (영상별 {_range_text(per_video, 2)})"
    )


def _summary_lines(tag: str, rows: Sequence[Mapping[str, Any]], context: ReportContext) -> list[str]:
    by_status = {
        status: [row for row in rows if row["status"] == status]
        for status in (STATUS_DONE, STATUS_UP_TO_DATE, STATUS_FAILED, STATUS_PENDING)
    }
    extracted = [row for row in rows if row["status"] in (STATUS_DONE, STATUS_UP_TO_DATE) and row["frames_kept"]]
    drop_sim = context.drop_sim
    mode = (
        "원본 프레임률(영상의 모든 프레임)"
        if drop_sim is None
        else f"기기 프레임 누락 모사(지연 평균 {drop_sim.latency_ms_mean:g} ms, 표준편차 "
        f"{drop_sim.latency_ms_std:g} ms, 시드 {drop_sim.seed})"
    )
    check = context.rotation_check
    rotation_text = (
        "영상의 회전 메타데이터(자동 점검 없음)"
        if check is None
        else f"영상별 자동 점검(표본 {check.frames}장, 간격 {check.interval_ms} ms, roll 여유 "
        f"{check.roll_margin_deg:g}도)"
    )
    lines = [
        f"# {context.description}",
        "",
        "영상을 MediaPipe FaceLandmarker(VIDEO 모드)에 통과시켜 RawFrame parquet을 만든 결과를 영상마다 적은 리포트다"
        " (계획서 docs/PLAN.md 11장의 M3 단계). 얼굴 화소는 어디에도 저장하지 않는다(계획서 1.2절의 결정 D13)."
        " 아래 값은 모두 숫자와 경로다.",
        "",
        f"- 생성 명령: `{context.command}`",
        f"- 설정: `{context.config_path}` (사본 `{tag}_config.yaml`). 추출 결과 폴더: `{context.out_root}`",
        f"- 추출 방식: {mode}",
        f"- MediaPipe {context.mediapipe_version}, 모델 파일 sha256 `{context.model_sha256}`",
        f"- 회전: {rotation_text}. 수동 회전 보정표 `{context.overrides_key}`에 적은 영상은 그 값을 쓴다",
        f"- 영상별 표: `{tag}_videos.csv` (입력 영상마다 한 행. 열의 뜻은 7절)",
    ]
    if context.pilot_of is not None:
        lines.append(
            f"- 시험 추출이다: 목록의 파일 {context.pilot_of}개 중 앞 {len(rows)}개만, 이미 결과가 있어도 다시, 한 번에"
            " 하나씩 추출했다"
        )

    lines += ["", "## 1. 합계", ""]
    totals = [
        ["이 리포트의 입력 영상(파일) 수", str(len(rows))],
        ["이번 실행에서 추출 (`done`)", str(len(by_status[STATUS_DONE]))],
        ["이미 있던 결과를 그대로 씀 (`up_to_date`)", str(len(by_status[STATUS_UP_TO_DATE]))],
        ["실패 (`failed`)", str(len(by_status[STATUS_FAILED]))],
        ["처리되지 않음 (`pending`, 실행이 끊김)", str(len(by_status[STATUS_PENDING]))],
    ]
    if context.manifest_totals is not None:
        n_files, n_logical, n_subjects = context.manifest_totals
        match = "같다" if (n_files, n_logical) == (RLDD_FILES, RLDD_LOGICAL_VIDEOS) else "다르다"
        totals += [
            ["데이터셋 목록 전체의 파일 수", f"{n_files} (UTA-RLDD 전체 사본의 기준값 {RLDD_FILES})"],
            ["데이터셋 목록 전체의 논리 영상 수", f"{n_logical} (기준값 {RLDD_LOGICAL_VIDEOS})"],
            ["데이터셋 목록 전체의 피험자 수", str(n_subjects)],
            ["기준값과 비교", match],
        ]
        logical: dict[str, list[str]] = {}
        for row in rows:
            logical.setdefault(str(row["video_id"]), []).append(str(row["status"]))
        have_result = (STATUS_DONE, STATUS_UP_TO_DATE)
        complete = [v for v, statuses in logical.items() if all(s in have_result for s in statuses)]
        failed = sorted(v for v, statuses in logical.items() if STATUS_FAILED in statuses)
        pending = [
            v for v, statuses in logical.items() if STATUS_PENDING in statuses and STATUS_FAILED not in statuses
        ]
        totals += [
            ["이 리포트의 논리 영상 수", str(len(logical))],
            ["모든 부분이 추출된 논리 영상", str(len(complete))],
            ["실패한 부분이 있는 논리 영상", f"{len(failed)}" + (f" ({', '.join(failed)})" if failed else "")],
            ["처리되지 않은 부분이 있는 논리 영상", str(len(pending))],
        ]
    lines += _table(["항목", "값"], totals)
    if context.manifest_totals is not None:
        lines += [
            "",
            f"기준값 {RLDD_FILES}와 {RLDD_LOGICAL_VIDEOS}은 계획서 2.4절의 값이다(피험자 60명 x 라벨 3개, 피험자 32와 49의"
            " `10`이 두 파일로 나뉨). 목록은 받은 사본을 그대로 센 것이며 기준값과 다르면 사본이 전체가 아니거나 배치가 다른 것이다.",
        ]

    lines += ["", "## 2. 타임스탬프 검사", ""]
    checked = [row for row in rows if row["t_ms_ok"] is not None]
    bad_times = [row for row in checked if not row["t_ms_ok"]]
    lines.append(
        f"추출 결과가 있는 영상 {len(checked)}개의 `t_ms` 열을 파일에서 다시 읽어, 첫 값이 1 이상이고 엄격히 증가하는지"
        f" 검사했다. 통과 {len(checked) - len(bad_times)}개, 위반 {len(bad_times)}개."
    )
    if bad_times:
        lines.append("위반: " + ", ".join(f"`{row['video']}`" for row in bad_times))

    lines += ["", "## 3. 실패한 영상과 사유", ""]
    problems = [row for row in rows if row["status"] == STATUS_FAILED or row["reason"]]
    if problems:
        lines.append(
            "실패한 영상은 추출 결과가 없어서 학습과 평가에서 빠진다(평가의 영상 단위 정확도에서는 분모에 남아 오답으로"
            " 센다. 계획서 1.2절의 결정 D7)."
            " 길이와 프레임률은 파일을 읽을 수 있을 때만 CSV에 적었다."
        )
        lines.append("")
        lines += _table(
            ["영상", "상태", "사유"],
            [[f"`{row['video']}`", str(row["status"]), _cell(row["reason"])] for row in problems],
        )
    else:
        lines.append("없음.")

    lines += ["", "## 4. 회전", ""]
    auto = [row for row in extracted if row["rotation_source"] == SOURCE_AUTO]
    manual = [row for row in extracted if row["rotation_source"] == SOURCE_MANUAL]
    applied: dict[int, int] = {}
    for row in extracted:
        applied[int(row["rotation"])] = applied.get(int(row["rotation"]), 0) + 1
    applied_text = ", ".join(f"{rotation}도 {count}개" for rotation, count in sorted(applied.items())) or "-"
    lines.append(
        f"추출된 영상 {len(extracted)}개 중 자동 점검 {len(auto)}개, 수동 보정 {len(manual)}개, 회전 메타데이터 그대로"
        f" {len(extracted) - len(auto) - len(manual)}개. 적용한 회전별 영상 수: {applied_text}."
    )
    if check is not None:
        lines += [
            "",
            "자동 점검은 영상 앞부분의 표본을 0·90·180·270도로 돌려 각도마다 얼굴 검출 비율과 머리 roll의 중앙값을 잰다."
            " 검출 비율이 가장 높은 각도(표본 한 장 차이까지 동률)를 먼저 고르고, 동률이면 |roll|이 가장 작은 각도를 고른다."
            f" |roll|의 차이가 {check.roll_margin_deg:g}도 미만인 각도가 또 있으면 가르지 못한 것으로 표시한다.",
            "",
            "**이 점검은 고른 각도와 그 반대 각도(예: 0도와 180도)를 가르지 못할 수 있다.** 실제 랜드마커는 거꾸로 선"
            " 얼굴도 검출하고, 그때 roll이 정방향과 거의 같게 나오기 때문이다. 아래 영상은 직접 보고 수동 회전 보정표"
            f" `{context.overrides_key}`에 `영상 경로: 각도`로 적은 뒤 다시 실행한다(그 영상만 다시 추출된다).",
        ]
        groups = [
            (
                "가르지 못했고 회전 메타데이터가 후보에 없음 (`ambiguous_smallest_roll`): 메타데이터가 틀린 것으로 보인다."
                " 고른 각도는 맞거나 정반대다. 먼저 확인할 것",
                [row for row in auto if row["rotation_reason"] == REASON_AMBIGUOUS_ROLL],
            ),
            (
                "점검이 회전 메타데이터와 다른 각도를 골랐음 (가르지 못한 경우는 위에 있다)",
                [row for row in auto if not row["rotation_ambiguous"] and row["rotation"] != row["metadata_rotation"]],
            ),
            (
                "어느 각도에서도 얼굴을 못 찾음 (`no_face`): 회전 메타데이터를 유지했다",
                [row for row in auto if row["rotation_reason"] == REASON_NO_FACE],
            ),
            (
                "가르지 못해 회전 메타데이터를 유지했음 (`ambiguous_kept_metadata`): 메타데이터가 맞으면 문제가 없다."
                " 정반대(180도 차이)일 가능성만 남는다",
                [row for row in auto if row["rotation_reason"] == REASON_KEPT_METADATA],
            ),
        ]
        for title, group in groups:
            lines += ["", f"### {title}", "", f"{len(group)}개."]
            if group:
                lines += [""] + _rotation_table(group)
    flipped = [row for row in rows if row["display_flip"]]
    lines += ["", f"표시 행렬에 뒤집기(거울상)가 든 영상: {len(flipped)}개."]
    if flipped:
        lines.append(
            "이런 영상은 회전 메타데이터를 믿을 수 없어 수동 보정표에 없으면 실패로 처리하고, 있으면 그 각도로만 돌려"
            " 거울상인 채로 추출한다: " + ", ".join(f"`{row['video']}` ({row['status']})" for row in flipped)
        )

    lines += ["", "## 5. 녹화 통계", ""]
    if extracted:
        kept = sum(int(row["frames_kept"]) for row in extracted)
        face = sum(int(row["frames_face"]) for row in extracted)
        lowest = min(extracted, key=lambda row: row["face_rate"])
        low_face = [row for row in extracted if row["face_rate"] < context.min_face_ratio]
        durations = [int(row["duration_ms"]) for row in extracted if row["duration_ms"] is not None]
        fps = [float(row["mean_fps"]) for row in extracted if row["mean_fps"] is not None]
        variable = [row for row in extracted if row["variable_intervals"]]
        brightness = [float(row["brightness_mean"]) for row in extracted if row["brightness_mean"] is not None]
        sizes: dict[str, int] = {}
        for row in extracted:
            size = f"{row['width']}x{row['height']}"
            sizes[size] = sizes.get(size, 0) + 1
        low_face_text = f"{len(low_face)}"
        if low_face:
            low_face_text += " (" + ", ".join(f"`{row['video']}`" for row in low_face) + ")"
        stats_rows = [
            ["처리한 프레임(= parquet 행) 합계", str(kept)],
            ["디코딩한 프레임 합계", str(sum(int(row["frames_decoded"]) for row in extracted))],
            ["타임스탬프 때문에 버린 프레임 합계", str(sum(int(row["frames_skipped"]) for row in extracted))],
            ["누락 모사가 버린 프레임 합계", str(sum(int(row["frames_dropped"]) for row in extracted))],
            ["얼굴 검출 비율(프레임 기준, 전체)", f"{face / kept:.3f}"],
            ["얼굴 검출 비율이 가장 낮은 영상", f"`{lowest['video']}` {lowest['face_rate']:.3f}"],
            [f"얼굴 검출 비율이 {context.min_face_ratio:g} 미만인 영상 (`low_face_rate`)", low_face_text],
            ["원본 영상 길이 합계", f"{sum(durations) / 3_600_000:.3f}시간 ({sum(durations) / 1000:.1f}초)"],
            ["원본 평균 프레임률", _range_text(fps, 2)],
            ["프레임 간격이 변하는 영상 (최대 − 최소 > 1 ms)", f"{len(variable)} / {len(extracted)}"],
            ["해상도(세운 뒤)별 영상 수", ", ".join(f"{size} {count}개" for size, count in sorted(sizes.items()))],
            ["밝기(영상별 평균, 0~255)", _range_text(brightness, 1)],
        ]
        lines += _table(["항목", "값"], stats_rows)
        lines += [
            "",
            "길이, 프레임률, 간격은 원본 영상의 프레임 시각으로 구한 값이다(누락 모사 추출에서도 같다). 밝기는 MediaPipe에"
            " 넘긴 세운 RGB 프레임 전체의 luma 평균 (0.299·R + 0.587·G + 0.114·B)를 영상의 처리한 프레임에 대해 요약한 값이다."
            f" `low_face_rate`는 프레임 기준 비율로 붙인 주의 표시다. 평가의 영상 품질 기준(계획서 1.2절의 결정 D7, 얼굴"
            f" 검출 비율 {context.min_face_ratio:g} 이상)은 시간 기준 비율이며 평가 때 따로 계산한다.",
        ]
    else:
        lines.append("추출 결과가 있는 영상이 없다.")

    lines += ["", "## 6. 처리 시간", ""]
    timed = [row for row in extracted if row["elapsed_s"] is not None]
    measured = [row for row in timed if not row.get("_timing_carried")]
    carried = [row for row in timed if row.get("_timing_carried")]
    is_dataset = tag.startswith("dataset_")  # 전체 추출의 리포트: 시험 추출 리포트의 값도 이어 온다
    pilot_csv = f"pilot_{tag.removeprefix('dataset_')}_videos.csv"
    intro = (
        "영상 하나의 랜드마커 생성, 디코딩, 추론, 저장을 합친 시간이며 회전 점검은 뺀 값이다. 이번 실행에서 잰 값과 전에 쓴"
        " 리포트에서 이어 온 값은 잰 조건(동시에 처리한 영상 수)이 다를 수 있어서 합치지 않고 따로 적는다. 이어 온 값은"
        " 그 값이 적혀 있던 리포트별로 적는다."
    )
    if is_dataset:
        intro += (
            f" 한 영상의 값이 시험 추출 리포트(`{pilot_csv}`)와 이 리포트의 전 판에 다 있으면 나중에 쓴 리포트의 값을"
            " 쓰고, 두 값이 같으면 시험 추출에서 잰 값으로 적는다(이 리포트의 전 판이 시험 추출의 값을 옮겨 적은 것이다)."
        )
    lines += [intro, ""]
    if measured:
        how = (
            "이번 실행은 영상을 한 번에 하나씩 처리했다"
            if context.workers == 1
            else f"이번 실행은 영상을 최대 {context.workers}개씩 동시에 처리했다(CPU를 나눠 쓴 값이다)"
        )
        lines.append(f"- 이번 실행에서 잰 것: {_timing_text(measured)}. {how}.")
    else:
        lines.append("- 이번 실행에서 잰 것: 없음.")
    from_pilot = [row for row in carried if row.get("_timing_carried") == CARRIED_FROM_PILOT]
    from_own = [row for row in carried if row.get("_timing_carried") != CARRIED_FROM_PILOT]
    if from_pilot:
        lines.append(
            f"- 전 리포트에서 이어 온 것(시험 추출 리포트 `{pilot_csv}`): {_timing_text(from_pilot)}. 이번 실행이 건너뛴"
            " 영상이고, 값은 시험 추출(`--dataset --pilot`)이 영상을 한 번에 하나씩 처리하며 잰 것이다."
        )
    if from_own:
        if tag.startswith("pilot_"):
            where = "값은 전의 시험 추출이 영상을 한 번에 하나씩 처리하며 잰 것이다."
        else:
            where = (
                "값은 그 영상을 추출한 전의 실행에서 잰 것이다. 그 실행이 영상을 몇 개씩 동시에 처리했는지는 이 리포트에"
                " 남아 있지 않다."
            )
        lines.append(
            f"- 전 리포트에서 이어 온 것(이 리포트의 전 판 `{tag}_videos.csv`): {_timing_text(from_own)}. 이번 실행이"
            f" 건너뛴 영상이고, {where}"
        )
    if not carried:
        lines.append("- 전 리포트에서 이어 온 것: 없음.")
    if len(timed) < len(extracted):
        lines.append(
            f"- 처리 시간을 모르는 것: 영상 {len(extracted) - len(timed)}개. 전에 추출한 결과를 그대로 썼는데 전 리포트에 값이"
            " 없거나, 실행이 끊겨 처리 시간을 받지 못한 영상이다."
        )

    lines += ["", "## 7. 영상별 표", ""]
    lines += _table(
        ["영상", "상태", "프레임", "얼굴 비율", "길이(초)", "평균 fps", "간격 변함", "크기", "회전", "회전 출처", "가르지 못함", "ms/프레임", "밝기", "주의"],
        [
            [
                f"`{row['video']}`",
                str(row["status"]),
                _cell(row["frames_kept"]),
                _cell(row["face_rate"]),
                _cell(None if row["duration_ms"] is None else row["duration_ms"] / 1000.0, 1),
                _cell(row["mean_fps"], 2),
                _cell(row["variable_intervals"]),
                "-" if row["width"] is None else f"{row['width']}x{row['height']}",
                _cell(row["rotation"]),
                _cell(row["rotation_source"]),
                _cell(row["rotation_ambiguous"]),
                _cell(row["ms_per_frame"], 2),
                _cell(row["brightness_mean"], 1),
                _cell(row["warnings"]),
            ]
            for row in rows
        ],
    )
    lines += [
        "",
        f"`{tag}_videos.csv`의 열: `status`(done 이번에 추출, up_to_date 전 결과 사용, failed 실패, pending 미처리),"
        " `reason`(실패 사유), `warnings`(주의 표시), `frames_decoded`/`frames_skipped`/`frames_dropped`/`frames_kept`/"
        "`frames_face`(디코딩, 타임스탬프로 버림, 누락 모사로 버림, 처리, 얼굴 검출 프레임 수), `face_rate`,"
        " `duration_ms`, `mean_fps`, `min_interval_ms`, `max_interval_ms`, `variable_intervals`, `width`, `height`,"
        " `rotation`(적용한 회전, 반시계), `rotation_source`(manual_override, auto_check, video_metadata),"
        " `rotation_reason`, `metadata_rotation`, `rotation_ambiguous`, `check_frames`, `rate_0`..`rate_270`(각도별 얼굴"
        " 검출 비율), `roll_0`..`roll_270`(각도별 roll 중앙값, 도), `display_flip`, `elapsed_s`, `ms_per_frame`,"
        " `brightness_mean`, `brightness_median`, `t_ms_ok`, `first_t_ms`, `last_t_ms`, `model_sha256`,"
        " `mediapipe_version`. `--dataset`에서는 `video_id`, `session_id`, `subject`, `label`, `part`도 있다.",
    ]
    return lines


def _write_text_atomic(path: Path, text: str) -> None:
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(text, encoding="utf-8", newline="")
    os.replace(temp_path, path)


def write_extraction_report(
    report_dir: Path, tag: str, rows: Sequence[Mapping[str, Any]], context: ReportContext
) -> tuple[Path, Path]:
    """추출 리포트를 쓰고 (CSV 경로, Markdown 경로)를 돌려준다 (PLAN 11장 M3).

    - `<tag>_videos.csv`: 입력 영상마다 한 행, 열은 `REPORT_COLUMNS`. 빈 칸은 값이 없다는 뜻이고 참·거짓은 true/false다.
    - `<tag>_summary.md`: 첫 줄에 데이터 출처, 이어서 합계, 타임스탬프 검사, 실패 사유, 회전, 녹화 통계, 처리 시간, 영상별 표.
      처리 시간은 이번 실행에서 잰 행과 전 리포트에서 이어 온 행(행의 `_timing_carried`에 출처 `CARRIED_FROM_PILOT`이나
      `CARRIED_FROM_OWN`이 있다. CSV에는 쓰지 않는 표시다)을 따로 요약하고, 이어 온 행은 그 값이 적혀 있던 리포트별로
      (시험 추출 리포트, 이 리포트의 전 판) 한 줄씩 적어 어디서 잰 값인지 밝힌다. "한 번에 몇 개씩 처리했다"는 문장
      (`context.workers`)은 이번 실행에서 잰 값에만 붙인다. CSV의 열은 그대로다(`REPORT_COLUMNS`, 47개).
    숫자와 경로만 쓴다. 얼굴 화소는 쓰지 않는다(PLAN 1.2절의 결정 D13).
    가정: 파일 이름의 tag는 `<입력 종류>_<추출 폴더 이름>`이다(dataset_native, pilot_native, dataset_fps15,
    input_dir_native, unlabeled_native 등). 원본 추출, 누락 모사 추출, 시험 추출의 리포트가 한 실행명 아래에서
    서로 덮어쓰지 않게 하기 위해서다.
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / f"{tag}_videos.csv"
    md_path = report_dir / f"{tag}_summary.md"
    temp_csv = csv_path.with_name(csv_path.name + ".tmp")
    with temp_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(REPORT_COLUMNS)
        for row in rows:
            writer.writerow([_csv_value(row.get(name)) for name in REPORT_COLUMNS])
    os.replace(temp_csv, csv_path)
    _write_text_atomic(md_path, "\n".join(_summary_lines(tag, rows, context)) + "\n")
    return csv_path, md_path


# ---------------------------------------------------------------------------
# 명령 실행
# ---------------------------------------------------------------------------


def _mode_text(drop_sim: DropSimConfig | None, subdir: str) -> str:
    if drop_sim is None:
        return "native frame rate extraction"
    return f"device frame-drop simulation '{subdir}' (latency mean {drop_sim.latency_ms_mean:g} ms)"


def _extract_all(args: argparse.Namespace, command: str, run_name: str) -> int:
    cfg = load_config(args.config)
    spec = load_feature_spec(cfg.paths.feature_spec).pipeline
    verify_model_file(cfg.paths.model_path, spec)
    drop_sim = None if args.fps_sim is None else drop_sim_for_fps(cfg.extraction.drop_sim, args.fps_sim)
    subdir = cfg.extraction.native_subdir if drop_sim is None else drop_sim.subdir
    out_root = cfg.paths.raw_dir / subdir
    workers = cfg.extraction.workers if args.workers is None else args.workers
    if workers < 1:
        raise ExtractError(f"--workers must be >= 1, got {workers}")
    pilot: int | None = None
    if args.pilot is not None:
        if not args.dataset:
            raise ExtractError("--pilot needs --dataset")
        pilot = cfg.extraction.pilot_videos if args.pilot < 0 else args.pilot
        if pilot < 1:
            raise ExtractError(f"--pilot must be >= 1, got {pilot}")
    if args.rotation is not None and args.video is None:
        raise ExtractError(
            "--rotation needs --video. Fix: for --input-dir, --dataset and --unlabeled list the video in the "
            "manual rotation table of the config"
        )

    # 회전 자동 점검은 모든 입력에서 한다. 수동 값(보정표, `--video`의 `--rotation`)이 있는 영상만 건너뛴다.
    rotation_check = RotationCheckSettings(
        cfg.extraction.rotation_check_frames,
        cfg.extraction.rotation_check_interval_ms,
        cfg.extraction.rotation_roll_margin_deg,
    )
    overrides: Mapping[str, int] = cfg.extraction.rotation_overrides
    overrides_key = OVERRIDES_KEY
    dataset_info: dict[str, dict[str, Any]] = {}
    manifest_totals: tuple[int, int, int] | None = None
    mode = _mode_text(drop_sim, subdir)
    if args.dataset:
        video_root, manifest = _dataset_videos(cfg)
        all_paths = [Path(rel_path) for rel_path in manifest["rel_path"].tolist()]
        relative_paths = all_paths if pilot is None else all_paths[:pilot]
        kind = "dataset" if pilot is None else "pilot"
        for name in ("video_id", "session_id", "subject", "label", "part"):
            for rel_path, value in zip(manifest["rel_path"].tolist(), manifest[name].tolist()):
                dataset_info.setdefault(rel_path, {})[name] = value
        manifest_totals = (len(all_paths), int(manifest["video_id"].nunique()), int(manifest["subject"].nunique()))
        description = (
            f"Data source: labelled video dataset at {video_root} (paths.rldd_root, RLDD folder layout): "
            f"{manifest_totals[0]} files, {manifest_totals[1]} logical videos, {manifest_totals[2]} subjects; {mode}"
        )
        if pilot is not None:
            description += f"; pilot extraction of the first {len(relative_paths)} file(s)"
    elif args.unlabeled:
        video_root, all_paths = _unlabeled_videos(cfg)
        relative_paths = all_paths
        overrides = cfg.extraction.unlabeled_rotation_overrides
        overrides_key = UNLABELED_OVERRIDES_KEY
        out_root = cfg.paths.raw_dir / f"{UNLABELED_PREFIX}{subdir}"
        kind = "unlabeled"
        description = (
            f"Data source: unlabeled videos at {video_root} (paths.unlabeled_root, no labels): "
            f"{len(all_paths)} files; {mode}"
        )
    elif args.video is not None:
        video_root = args.video.parent
        all_paths = relative_paths = [Path(args.video.name)]
        kind = "video"
        description = f"Data source: single video {args.video}; {mode}"
    else:
        video_root = args.input_dir
        all_paths = relative_paths = [p.relative_to(video_root) for p in find_videos(video_root)]
        if not relative_paths:
            raise ExtractError(f"no video files {VIDEO_SUFFIXES} below {video_root}")
        kind = "input_dir"
        description = f"Data source: videos below {video_root} (--input-dir): {len(all_paths)} files; {mode}"
    check_override_keys(overrides, [relative.as_posix() for relative in all_paths], overrides_key)
    ambiguous_fix = f"look at the video and add it to the manual rotation table ({overrides_key})"
    if kind == "video":
        ambiguous_fix = f"look at the video and pass --rotation (or add it to {overrides_key})"

    entries: list[_Entry] = []
    owners: dict[Path, str] = {}
    for relative in relative_paths:
        label = relative.as_posix()
        out_parquet = raw_parquet_path(out_root, relative)
        if out_parquet in owners:
            raise ExtractError(f"{label} and {owners[out_parquet]} would both be written to {out_parquet}")
        owners[out_parquet] = label
        manual_rotation = overrides.get(label)
        if args.rotation is not None:  # `--video`만 받는다. 명령에 직접 준 값이 보정표보다 우선한다
            manual_rotation = args.rotation
        job = _Job(
            label,
            video_root / relative,
            out_parquet,
            cfg.paths.model_path,
            spec,
            drop_sim,
            manual_rotation,
            rotation_check,
            rotation_from_command=args.rotation is not None,
        )
        # 시험 추출은 프레임당 시간을 재는 것이 목적이므로 이미 있는 결과도 다시 만든다.
        current = None if pilot is not None else _job_current_metadata(job)
        if current is not None:
            entries.append(_Entry(job, STATUS_UP_TO_DATE))
            print(f"skip {label}: {out_parquet} is up to date{_skip_note(job, current)}")
        else:
            entries.append(_Entry(job, STATUS_PENDING))
            replaced = _replace_note(job)  # 있는 결과가 이름만 같은 다른 영상(크기가 다른 파일)의 것이면 알린다
            if replaced is not None:
                print(replaced)
    to_run = [entry for entry in entries if entry.status == STATUS_PENDING]
    n_current = len(entries) - len(to_run)
    sequential = args.video is not None or pilot is not None

    def finish(
        entry: _Entry, result: tuple[ExtractStats, RotationCheck | None] | None, error: BaseException | None
    ) -> None:
        if result is not None:
            stats, check = result
            entry.status, entry.stats = STATUS_DONE, stats
            rotation_text = _format_rotation(entry.job, check, ambiguous_fix)
            print(f"done {entry.job.label}: {_format_stats(stats)}; {rotation_text}")
        else:
            entry.status, entry.reason = STATUS_FAILED, f"{type(error).__name__}: {error}"
            print(f"FAILED {entry.job.label}: {entry.reason}", file=sys.stderr)

    report_paths: tuple[Path, Path] | None = None
    try:
        if sequential:
            # 이 프로세스에서 차례로 처리한다. 시험 추출은 영상끼리 CPU를 다투지 않아야 프레임당 시간이 의미가 있다.
            for entry in to_run:
                try:
                    finish(entry, _run_job(entry.job), None)
                except Exception as exc:  # 영상 하나의 실패를 보고하고 종료 코드로 알린다
                    finish(entry, None, exc)
        elif to_run:
            # max_tasks_per_child=1: 영상마다 새 프로세스. 랜드마커의 추적·평활 상태와 타임스탬프 규칙이 영상 사이에 섞이지 않는다.
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context, max_tasks_per_child=1) as pool:
                futures = {pool.submit(_run_job, entry.job): entry for entry in to_run}
                for future in as_completed(futures):
                    error = future.exception()
                    finish(futures[future], None if error is not None else future.result(), error)
    finally:
        # 영상이 실패했거나 실행이 중간에 끊겨도 그때까지의 결과로 리포트를 쓴다.
        if kind != "video":
            for entry in entries:
                if entry.status == STATUS_PENDING and _job_is_current(entry.job):
                    # 실행이 끊겨 보고를 받지 못한 영상이다. 시험 추출이 아니면 시작할 때 결과가 없던 영상이므로 이번
                    # 실행이 쓴 결과다(처리 시간은 모른다). 시험 추출이면 전에 있던 결과다.
                    entry.status = STATUS_DONE if pilot is None else STATUS_UP_TO_DATE
            report_dir = cfg.paths.reports_dir / run_name / REPORT_SUBDIR
            tag = f"{kind}_{subdir}"
            # 시험 추출에서 잰 영상의 처리 시간도 잇는다(전체 추출의 리포트만).
            pilot_csv = report_dir / f"pilot_{subdir}_videos.csv" if kind == "dataset" else None
            previous = read_previous_timings(report_dir / f"{tag}_videos.csv", pilot_csv)
            rows = _report_rows(entries, dataset_info, previous, cfg.evaluation.quality_gate.min_face_ratio)
            report_context = ReportContext(
                description=description,
                command=command,
                config_path=str(args.config),
                out_root=str(out_root),
                drop_sim=drop_sim,
                model_sha256=spec.mediapipe.model_sha256,
                mediapipe_version=mediapipe_version(),
                rotation_check=rotation_check,
                overrides_key=overrides_key,
                min_face_ratio=cfg.evaluation.quality_gate.min_face_ratio,
                workers=1 if sequential else min(workers, max(len(to_run), 1)),
                pilot_of=len(all_paths) if pilot is not None else None,
                manifest_totals=manifest_totals,
            )
            report_paths = write_extraction_report(report_dir, tag, rows, report_context)
            shutil.copyfile(args.config, report_dir / f"{tag}_config.yaml")

    finished = [entry.stats for entry in entries if entry.stats is not None]
    failures = [entry for entry in entries if entry.status == STATUS_FAILED]
    if pilot is not None and finished:
        n_frames = sum(stats.n_processed for stats in finished)
        n_face = sum(stats.n_face for stats in finished)
        elapsed_ms = 1000.0 * sum(stats.elapsed_s for stats in finished)
        lowest = min(stats.n_face / stats.n_processed for stats in finished)
        rotations = sorted({stats.rotation for stats in finished})
        print(
            f"pilot: {len(finished)} videos, {n_frames} frames, {elapsed_ms / n_frames:.1f} ms/frame "
            f"(decode + inference + write, without the rotation check), face rate {n_face / n_frames:.3f} "
            f"(lowest video {lowest:.3f}), rotations applied {rotations}"
        )
    print(f"extracted {len(to_run) - len(failures)}, up to date {n_current}, failed {len(failures)} -> {out_root}")
    if report_paths is not None:
        print(f"report: {report_paths[1]} (table: {report_paths[0].name})")
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m drowsy.extract`: 영상 하나, 폴더 전체, 데이터셋 목록, 또는 라벨 없는 영상 폴더를 추출한다.

    - 결과는 `raw_dir/<native_subdir 또는 누락 모사 subdir>/<영상 상대 경로>.parquet`에 쓴다(`raw_parquet_path`).
      `--unlabeled`는 `raw_dir/unlabeled_<같은 하위 폴더 이름>/` 아래에 쓴다(`unlabeled_raw_dir`).
    - 이미 같은 설정으로 만든 결과가 있으면 건너뛴다(중간에 끊겨도 다시 실행하면 이어서 한다). 결과에 적힌 원본 파일
      크기가 지금의 영상과 다르면(이름만 같은 다른 영상, 바뀐 파일) 건너뛰지 않고 다시 추출하며 `replace` 줄로 알린다.
      건너뛴 결과가 다른 경로의 영상에서 나온 것이면 `skip` 줄이 그 경로를 적는다(`output_is_current`, `_skip_note`).
    - `--input-dir`, `--dataset`, `--unlabeled`는 영상마다 새 프로세스와 새 랜드마커를 쓴다. `--video`는 이 프로세스에서 바로 처리한다.
    - 회전: 네 입력 모두, 수동 값이 없는 영상은 회전을 자동으로 점검한다(`check_rotation`, `choose_rotation`: 표본
      프레임을 0·90·180·270도로 돌려 얼굴 검출 비율이 가장 높은 각도, 동률이면 머리 roll이 가장 작은 각도. 가르지
      못하면 표시한다). 수동 값은 설정의 수동 회전 보정표(`--video`, `--input-dir`, `--dataset`은
      `extraction.rotation_overrides`, `--unlabeled`는 `extraction.unlabeled_rotation_overrides`)에 적거나,
      `--video`에서는 `--rotation {0,90,180,270}`으로 준다. 영상마다 어떻게 정했는지를 `done` 줄에 출력한다.
    - `--dataset`: 설정의 `paths.rldd_root`에서 만든 목록의 영상을 추출한다.
    - `--dataset --pilot [N]`: 목록의 앞 N개(생략하면 설정의 `extraction.pilot_videos`)만, 이 프로세스에서 차례로,
      이미 결과가 있어도 다시 추출하고 영상별 프레임당 시간·얼굴 검출률·회전과 전체 요약을 출력한다(M3의 시험 추출).
      가정: "앞 N개"는 목록 순서(피험자, 라벨, 부분)다. 결과는 전체 추출과 같은 자리에 쓰므로 전체 추출이 이어받는다.
    - `--unlabeled`: 설정의 `paths.unlabeled_root`를 `dataset.scan_unlabeled`로 훑어 그 영상들을 추출한다(PLAN 11장 M4의
      시간당 경고 수 계산용). 보정표는 `extraction.unlabeled_rotation_overrides`를 쓴다.
    - `--run-name NAME`(기본 `extraction`): `--video`를 뺀 실행은 `reports_dir/NAME/extraction/`에
      `<입력 종류>_<추출 폴더>_videos.csv`와 `_summary.md`, 설정 사본을 쓴다(`write_extraction_report`). 실패한 영상이
      있어도, 모든 결과가 이미 있어도, 실행이 중간에 끊겨도 쓴다.
    - 종료 코드: 0 성공, 1 실패한 영상이 있음, 2 시작할 수 없음(설정, 모델 파일, 입력 폴더, 목록, 보정표의 키,
      실행명, 옵션 조합 문제).
    가정: 설정의 rotation_overrides 키는 `--input-dir`이나 데이터셋 루트 기준 상대 경로(슬래시 구분)이고, `--video`에서는
    파일 이름이다. 값은 `frame.rotation`과 같은 규약(세우려면 반시계로 돌려야 하는 각도)이며 영상의 회전 메타데이터와
    자동 점검 결과를 대신한다. `--video`의 `--rotation`도 같은 규약이고, 보정표에 그 영상이 있어도 `--rotation`이
    이긴다(명령에 직접 준 값이 설정 파일보다 우선한다). `--rotation`은 그 실행에만 적용된다: 다음에 `--rotation` 없이
    실행하면 자동 점검으로 다시 추출한다. 이번 입력의 어느 영상과도 맞지 않는 보정표 키가 있으면 시작하지 않는다
    (`check_override_keys`).
    가정: 회전 자동 점검은 네 입력(`--video`, `--input-dir`, `--dataset`, `--unlabeled`) 모두에서 한다. 휴대전화로 찍은
    영상은 회전 메타데이터가 없거나 틀릴 수 있고, 실제 랜드마커는 옆으로 누운 얼굴도 검출하므로(얼굴 검출 비율 1.0)
    회전 메타데이터만 믿고 추출하면 틀린 방향의 신호가 아무 표시 없이 나온다. 회전 메타데이터만 써서 만든 전의 결과
    (자동 점검 기록이 없는 `--video`, `--input-dir` 결과)는 최신으로 보지 않고 다시 추출한다.
    가정: `--unlabeled`의 보정표를 `rotation_overrides`와 따로 둔다. 한 설정 파일에 데이터셋과 라벨 없는 영상의 보정
    값을 함께 적어도 서로를 "맞지 않는 키"로 거부하지 않게 하기 위해서다.
    """
    parser = argparse.ArgumentParser(
        prog="python -m drowsy.extract",
        description="Run MediaPipe FaceLandmarker (VIDEO mode) over videos and write RawFrame parquet files.",
    )
    parser.add_argument("--config", default="config/default.yaml", help="experiment config YAML")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="extract a single video (no extraction report)")
    source.add_argument("--input-dir", type=Path, help="extract every video below this directory")
    source.add_argument(
        "--dataset",
        action="store_true",
        help="extract the videos of the dataset manifest built from paths.rldd_root",
    )
    source.add_argument(
        "--unlabeled",
        action="store_true",
        help="extract the unlabeled videos below paths.unlabeled_root into raw_dir/unlabeled_<subdir>",
    )
    parser.add_argument(
        "--rotation",
        type=int,
        choices=ROTATION_CANDIDATES,
        default=None,
        help="with --video: turn the picture counter-clockwise by this angle to make it upright (same convention as "
        "the manual rotation table of the config) instead of running the automatic rotation check. Every input runs "
        "that check for videos without a manual rotation",
    )
    parser.add_argument(
        "--pilot",
        type=int,
        nargs="?",
        const=-1,
        default=None,
        metavar="N",
        help="with --dataset: extract only the first N videos (default extraction.pilot_videos) sequentially "
        "and print ms per frame, face rate and rotation",
    )
    parser.add_argument(
        "--fps-sim",
        type=int,
        default=None,
        metavar="FPS",
        help="simulate device frame drops (flow limiter) at about this frame rate instead of native extraction",
    )
    parser.add_argument("--workers", type=int, default=None, help="parallel videos (default: extraction.workers)")
    parser.add_argument(
        "--run-name",
        default=DEFAULT_RUN_NAME,
        help="report folder name below paths.reports_dir; the extraction report goes to <run name>/extraction/",
    )
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    command = "python -m drowsy.extract " + shlex.join(str(argument) for argument in arguments)
    try:
        run_name = check_run_name(args.run_name)  # 다른 일을 하기 전에 확인한다(리포트 폴더 밖에 쓰지 않게)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        return _extract_all(args, command, run_name)
    except (ExtractError, SpecError, VideoError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
