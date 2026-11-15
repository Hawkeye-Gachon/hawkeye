"""RLDD 로더, 무라벨 차량 영상 로더, 재생, 스텝 표, 창, fold (PLAN 2.4절, 4.4절, 6장, 1.2절의 결정 D7). 이식 대상이 아니다.

D7은 계획서(`docs/PLAN.md`) 1.2절의 표에서 승인된 결정이다: 영상 단위 정확도의 분모는 논리 영상 전체이고 창이 하나도
없는 영상은 오답으로 세며, 품질 기준(얼굴 검출 비율 0.7 이상이고 창 1개 이상)을 통과한 영상만의 값을 보조 행으로 싣는다.
이 파일의 "D7 품질 기준"은 그 품질 기준을 가리킨다.

흐름
    manifest = build_manifest(root)                    # 영상 폴더 또는 RawFrame parquet 폴더(합성 데이터 배치)
    tables = load_step_tables(manifest, raw_dir, spec) # 정규화 조건별 스텝 표(없으면 만들고 parquet으로 캐시)
    windows = make_windows(tables["alert_baseline"].steps, 120, stride=5)
    for split in nested_splits(manifest): ...
    # 학습 전용 증강 (2) (PLAN 5.2절): 저장된 프레임을 무작위로 솎아 다시 재생한 스텝 표 변형
    thinned = load_frame_drop_tables(manifest, raw_dir, spec, ["alert_baseline"],
                                     variant=0, seed=0, drop_min=0.10, drop_max=0.30)

스텝 특징은 저장된 `RawFrame`을 `DriverPipeline`에 프레임 하나씩 넣어서 얻는다(`replay_video`). 신호, closure,
감김 검출, 스텝 특징을 여기서 따로 계산하지 않는다(PLAN 3.1절). 프레임 누락 변형도 같은 재생 경로를 쓴다.

용어
- 논리 영상(video): 피험자 한 명의 한 라벨. `video_id`는 `S{피험자}_{라벨}`(예: `S01_0`, `S32_10`).
- 세션(session): 파일 하나. 나뉜 영상(`10_1`, `10_2`)은 부분마다 세션이고 `session_id`는 `S32_10_1`처럼 적는다.
  나뉘지 않은 영상은 `session_id == video_id`, `part == 0`이다. 세션마다 파이프라인을 새로 시작한다.
가정: 계획서는 식별자의 모양을 정하지 않았다. 위 형식으로 고정한다(피험자 번호는 두 자리 이상으로 0을 채운다).

명령 (저장소 루트에서 실행): 스텝 표를 만들어 캐시에 두고 회전 점검 기록, 조건별 창 수, 품질 요약, 기준선 요약을 출력한다.
    python -m drowsy.dataset --config config/default.yaml [--arms alert_baseline global session_baseline] [--workers N]
        [--fps-sim] [--rebuild] [--frame-drop-variants V [V ...]] [--frame-drop-seed S]
- `--rebuild`: 캐시가 유효해도 무시하고 다시 만든다. 캐시는 추출 결과 파일의 크기, 수정 시각, 파일 전체의 sha256
  가운데 하나라도 바뀌면 스스로 다시 만든다(명령을 실행할 때마다 파일을 모두 읽어 확인한다. `raw_content_digest`).
  파이썬·넘파이 판이 바뀐 것은 알아채지 못하므로 그럴 때 이 옵션을 쓴다.
- `--frame-drop-variants`: 프레임 누락 변형(증강 (2))의 스텝 표도 만들어 캐시에 둔다. 비율 범위는
  `training.frame_drop_min`~`training.frame_drop_max`, 시드는 `--frame-drop-seed`(기본 0)다.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import multiprocessing
import operator
import os
import random
import re
import sys
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from drowsy import calibrate
from drowsy.calibrate import BaselineDiagnostics
from drowsy.config import (
    ARM_ALERT_BASELINE,
    ARM_GLOBAL,
    ARM_SESSION_BASELINE,
    ARMS,
    ExperimentConfig,
    QualityGate,
    load_config,
)
from drowsy.extract import (
    OVERRIDES_KEY,
    ROTATION_CHECK_KEY,
    VIDEO_SUFFIXES,
    ExtractError,
    raw_parquet_path,
    read_extract_metadata,
    read_raw_frames,
)
from drowsy.pipeline import DriverPipeline
from drowsy.spec import (
    DEFAULT_RULES_PATH,
    PipelineSpec,
    RulesConfig,
    compute_pipeline_hash,
    load_feature_spec,
    load_rules,
    pipeline_to_dict,
)
from drowsy.types import (
    BASELINE_FALLBACK_GLOBAL,
    CLASS_ALERT,
    CLASS_DROWSY,
    CLASS_LOW_VIGILANT,
    CLASS_ORDER,
    FEATURE_NAMES,
    N_FEATURES,
    Baseline,
    ClosureRun,
    DriverStateEvent,
    RawFrame,
    StepFeatures,
)

# RLDD 파일 이름의 라벨(0 = alert, 5 = low vigilant, 10 = drowsy) → 클래스 인덱스
LABEL_TO_CLASS: dict[int, int] = {0: CLASS_ALERT, 5: CLASS_LOW_VIGILANT, 10: CLASS_DROWSY}
SUBJECTS_PER_FOLD = 12  # 공식 5-fold: fold k는 피험자 12(k−1)+1 ~ 12k (PLAN 2.4절)
RAW_SUFFIX = ".parquet"
# PLAN 1.2절의 결정 D7이 고정한 품질 기준: 얼굴 검출 비율(시간 기준) 0.7 이상이고 창 1개 이상
DEFAULT_QUALITY_GATE = QualityGate(min_face_ratio=0.7, min_windows=1)

ID_COLUMNS: tuple[str, ...] = ("video_id", "session_id", "subject", "fold", "label", "class_index", "part")
MANIFEST_COLUMNS: tuple[str, ...] = ID_COLUMNS + ("rel_path", "path")
STEP_COLUMNS: tuple[str, ...] = ID_COLUMNS + ("k", "t_end_ms", "full_lookback") + FEATURE_NAMES
UNLABELED_COLUMNS: tuple[str, ...] = ("video_id", "rel_path", "path")

BASELINE_NONE = "none"  # 품질 표의 baseline_status: 보정이 끝나기 전에 영상이 끝남
RAW_OK = "ok"
RAW_MISSING = "missing"  # 품질 표의 raw_status: 추출 결과 parquet이 없음

_STEM_RE = re.compile(r"^(0|5|10)(?:_(\d+))?$")
_FOLD_RE = re.compile(r"^fold(\d+)(?:_part\d+)?$", re.IGNORECASE)
_SUBJECT_RE = re.compile(r"^\d+$")
_MAX_LISTED = 10  # 오류 메시지에 적는 항목 수의 상한


class ManifestError(ValueError):
    """데이터 폴더의 내용이 기대한 배치와 다를 때(빠진 클래스, fold 불일치, 해석할 수 없는 파일 등)."""


class SubjectOverlapError(AssertionError):
    """피험자 집합들이 서로 겹칠 때(`assert_subject_disjoint`)."""


# ---------------------------------------------------------------------------
# fold와 피험자 분리
# ---------------------------------------------------------------------------


def official_fold(subject_id: int) -> int:
    """RLDD 공식 fold 번호(1부터): `(subject_id − 1) // 12 + 1`.

    가정: 60번을 넘는 피험자 번호(합성 데이터)에도 같은 식을 쓴다. 1 미만이면 ValueError.
    """
    if subject_id < 1:
        raise ValueError(f"subject id must be >= 1, got {subject_id}")
    return (subject_id - 1) // SUBJECTS_PER_FOLD + 1


def assert_subject_disjoint(*subject_sets: Iterable[int]) -> None:
    """피험자 집합들이 서로 겹치지 않는지 검사한다. 한 명이라도 두 집합에 들어 있으면 `SubjectOverlapError`.

    데이터 구성, fold 구성, 리포트 작성 시점에 부른다(PLAN 6장). 한 집합 안의 중복은 겹침이 아니다.
    """
    owner: dict[int, int] = {}
    overlaps: dict[int, list[int]] = {}
    for position, subjects in enumerate(subject_sets):
        for subject in sorted({int(value) for value in subjects}):
            if subject in owner:
                overlaps.setdefault(subject, [owner[subject]]).append(position)
            else:
                owner[subject] = position
    if overlaps:
        listed = ", ".join(f"subject {s} in sets {sets}" for s, sets in sorted(overlaps.items())[:_MAX_LISTED])
        raise SubjectOverlapError(f"subject sets overlap ({len(overlaps)} subjects): {listed}")


# ---------------------------------------------------------------------------
# 목록
# ---------------------------------------------------------------------------


def _scan(root: Path) -> tuple[list[Path], bool]:
    """root 아래의 영상 파일 또는 RawFrame parquet 파일의 상대 경로(정렬됨)와, parquet 폴더인지 여부.

    숨김 항목(이름이 점으로 시작)과 `__MACOSX` 폴더는 건너뛴다(zip을 풀 때 생기는 부산물이다).
    """
    root = Path(root)
    if not root.is_dir():
        raise ManifestError(f"dataset root is not a directory: {root}")
    videos: list[Path] = []
    raws: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part.startswith(".") or part == "__MACOSX" for part in relative.parts):
            continue
        suffix = path.suffix.lower()
        if suffix in VIDEO_SUFFIXES:
            videos.append(relative)
        elif suffix == RAW_SUFFIX:
            raws.append(relative)
    if videos and raws:
        raise ManifestError(
            f"{root} holds both video files ({len(videos)}) and raw parquet files ({len(raws)}); "
            "point at the video tree or at the raw parquet tree, not at a mix"
        )
    if not videos and not raws:
        raise ManifestError(f"no video files {VIDEO_SUFFIXES} and no raw parquet files below {root}")
    return (raws, True) if raws else (videos, False)


def _listed(items: Sequence[str]) -> str:
    shown = ", ".join(items[:_MAX_LISTED])
    return shown if len(items) <= _MAX_LISTED else f"{shown}, ... ({len(items)} in total)"


def build_manifest(root: Path) -> pd.DataFrame:
    """RLDD 영상 폴더(또는 같은 배치의 RawFrame parquet 폴더)를 훑어 세션 목록을 만든다 (PLAN 2.4절).

    받는 배치: `.../{피험자 폴더}/{0|5|10}[_N].{확장자}`.
    - 피험자 폴더는 숫자 이름이다. 0을 채운 것(`01`)과 채우지 않은 것(`1`)을 모두 받는다.
    - 그 위의 폴더 깊이와 이름은 자유다(`Fold1_part1/Fold1_part1/01/...`처럼 겹쳐 있어도 된다).
      경로에 `FoldK` 또는 `FoldK_partP` 폴더가 있으면 K가 공식 fold `(피험자 − 1) // 12 + 1`과 같은지 검사한다.
    - 확장자는 `.mp4`, `.mov`, `.m4v`(대소문자 무시)이고 parquet 폴더에서는 `.parquet`이다.
    - 파일 이름 `10_1`, `10_2`는 나뉜 영상의 부분이다. 부분 번호는 1부터 빠짐없이 있어야 하고 부분은 둘 이상이어야 한다.
      가정: `_N`은 세 라벨 모두에서 받는다(RLDD에는 10에만 있다).
      가정: `10_1`만 있고 `10_2`가 없으면 복사가 덜 된 것으로 보고 `ManifestError`를 낸다(영상의 절반만으로 논리 영상
      하나를 만들지 않는다). 부분이 셋 이상일 때 마지막 부분이 빠진 것은 파일 이름만으로는 알 수 없어 검사하지 못한다.

    열: `video_id`, `session_id`, `subject`, `fold`, `label`(0, 5, 10), `class_index`(0, 1, 2), `part`
    (나뉘지 않았으면 0), `rel_path`(root 기준, 슬래시 구분), `path`. 한 행이 세션(파일) 하나이고
    (subject, class_index, part) 순으로 정렬돼 있다.

    조용히 넘어가지 않고 `ManifestError`를 내는 경우: 해석할 수 없는 영상 파일, FoldK 폴더와 피험자 번호의 불일치,
    같은 (피험자, 라벨, 부분)의 파일이 둘 이상, `10`과 `10_N`이 함께 있음, 부분 번호가 빠짐, 부분이 `_1` 하나뿐임,
    세 라벨 중 빠진 것이 있는 피험자, 영상과 parquet이 섞인 폴더, 파일이 하나도 없음.
    가정: 피험자 수는 검사하지 않는다(합성 데이터나 부분 사본도 받는다). 파일은 열어 보지 않는다.
    """
    root = Path(root)
    relatives, _ = _scan(root)

    rows: list[dict[str, Any]] = []
    unparsed: list[str] = []
    fold_mismatch: list[str] = []
    for relative in relatives:
        stem = _STEM_RE.match(relative.stem)
        if stem is None or len(relative.parts) < 2 or _SUBJECT_RE.match(relative.parts[-2]) is None:
            unparsed.append(relative.as_posix())
            continue
        subject = int(relative.parts[-2])
        if subject < 1:
            unparsed.append(relative.as_posix())
            continue
        label = int(stem.group(1))
        part = 0 if stem.group(2) is None else int(stem.group(2))
        if stem.group(2) is not None and part < 1:
            unparsed.append(relative.as_posix())
            continue
        fold = official_fold(subject)
        for folder in relative.parts[:-2]:
            named = _FOLD_RE.match(folder)
            if named is not None and int(named.group(1)) != fold:
                fold_mismatch.append(f"{relative.as_posix()} (subject {subject} belongs to fold {fold})")
                break
        video_id = f"S{subject:02d}_{label}"
        rows.append(
            {
                "video_id": video_id,
                "session_id": video_id if part == 0 else f"{video_id}_{part}",
                "subject": subject,
                "fold": fold,
                "label": label,
                "class_index": LABEL_TO_CLASS[label],
                "part": part,
                "rel_path": relative.as_posix(),
                "path": str(root / relative),
            }
        )

    if unparsed:
        raise ManifestError(
            f"cannot tell subject and label of {len(unparsed)} file(s) below {root}: {_listed(unparsed)}. "
            "Expected .../<subject number>/<0|5|10>[_N].<ext>; rename or remove these files"
        )
    if fold_mismatch:
        raise ManifestError(
            f"FoldK folder name contradicts the official fold (subject-1)//{SUBJECTS_PER_FOLD}+1: "
            f"{_listed(fold_mismatch)}"
        )

    by_session: dict[str, list[str]] = {}
    for row in rows:
        by_session.setdefault(row["session_id"], []).append(row["rel_path"])
    duplicates = [f"{session}: {paths}" for session, paths in sorted(by_session.items()) if len(paths) > 1]
    if duplicates:
        raise ManifestError(f"more than one file for the same subject, label and part: {_listed(duplicates)}")

    parts_by_video: dict[str, list[int]] = {}
    labels_by_subject: dict[int, set[int]] = {}
    for row in rows:
        parts_by_video.setdefault(row["video_id"], []).append(row["part"])
        labels_by_subject.setdefault(row["subject"], set()).add(row["label"])
    bad_parts = [
        f"{video}: parts {sorted(parts)}"
        for video, parts in sorted(parts_by_video.items())
        # 나뉘지 않은 영상([0])이 아니면 부분이 둘 이상이고 1..N이 빠짐없이 있어야 한다. [1] 하나뿐인 것도 오류다.
        if sorted(parts) != [0] and (len(parts) < 2 or sorted(parts) != list(range(1, len(parts) + 1)))
    ]
    if bad_parts:
        raise ManifestError(
            "split videos need at least two parts, numbered _1.._N without gaps, and must not coexist with the "
            f"unsplit file (a lone _1 means the copy is incomplete): {_listed(bad_parts)}"
        )
    missing = [
        f"subject {subject} lacks label(s) {sorted(set(LABEL_TO_CLASS) - labels)}"
        for subject, labels in sorted(labels_by_subject.items())
        if labels != set(LABEL_TO_CLASS)
    ]
    if missing:
        raise ManifestError(f"every subject needs the three labels 0, 5 and 10: {_listed(missing)}")

    manifest = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))
    manifest = manifest.sort_values(["subject", "class_index", "part"], kind="stable").reset_index(drop=True)
    _check_folds(manifest)
    return manifest


def _subjects_by_fold(manifest: pd.DataFrame) -> dict[int, tuple[int, ...]]:
    folds: dict[int, set[int]] = {}
    for subject, fold in zip(manifest["subject"].tolist(), manifest["fold"].tolist()):
        folds.setdefault(int(fold), set()).add(int(subject))
    return {fold: tuple(sorted(folds[fold])) for fold in sorted(folds)}


def _check_folds(manifest: pd.DataFrame) -> dict[int, tuple[int, ...]]:
    """fold별 피험자 집합을 돌려준다. 한 피험자가 두 fold에 있으면 `SubjectOverlapError`."""
    folds = _subjects_by_fold(manifest)
    assert_subject_disjoint(*folds.values())
    return folds


def scan_unlabeled(root: Path) -> pd.DataFrame:
    """팀이 수집한 차량 영상처럼 라벨이 없는 영상 폴더를 훑는다. 시간당 경고 수를 재는 데만 쓴다 (PLAN 6장, 11장 M4).

    폴더 구조는 자유다. 영상 파일(`.mp4`, `.mov`, `.m4v`, 대소문자 무시) 또는 그 추출 결과(`.parquet`)를 경로순으로 찾는다.
    열: `video_id`(확장자를 뺀 상대 경로), `rel_path`, `path`. 추출 결과의 위치는 `extract.raw_parquet_path(raw_dir, rel_path)`다.
    파일이 없거나 영상과 parquet이 섞여 있으면 `ManifestError`. 확장자만 다른 두 파일도 `ManifestError`다(추출 결과가 겹친다).
    """
    root = Path(root)
    relatives, _ = _scan(root)
    rows = [
        {"video_id": relative.with_suffix("").as_posix(), "rel_path": relative.as_posix(), "path": str(root / relative)}
        for relative in relatives
    ]
    ids = [row["video_id"] for row in rows]
    clashes = sorted({video_id for video_id in ids if ids.count(video_id) > 1})
    if clashes:
        raise ManifestError(f"files that differ only by extension would share one raw parquet: {_listed(clashes)}")
    return pd.DataFrame(rows, columns=list(UNLABELED_COLUMNS))


# ---------------------------------------------------------------------------
# 재생
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """`replay_video`의 결과. 시각은 모두 원래 t_ms 시계다.

    - steps, runs, events: `DriverPipeline`이 낸 스텝, 특징용 감김 구간, 상태 이벤트(발행 순서).
    - baseline: 재생이 끝났을 때 파이프라인이 쓰던 개인 기준선. 넣어 준 기준선이 있으면 그것이고,
      없으면 이 세션에서 보정한 것이다. 보정이 끝나기 전에 프레임이 끝났으면 None이다.
    - diagnostics: 이 세션에서 보정했을 때의 판정 통계(기준선을 넣어 줬거나 보정이 안 끝났으면 None).
    - n_frames, n_dropped: 받아들인 프레임 수와, t_ms가 증가하지 않아 버린 프레임 수.
    - duration_ms: 받아들인 마지막 프레임과 첫 프레임의 t_ms 차.
    - face_ms: 얼굴 프레임이 대표한 시간. 다음 프레임까지의 간격이고 최대 `clock.max_hold_ms`다(PLAN 3.3절).

    가정: 계획서는 재생 결과로 스텝, 감김 구간, 이벤트, 기준선, 진단만 적었다. 품질 표(D7)에 필요한 프레임 계수와
    얼굴 시간을 같은 재생에서 함께 센다(파이프라인이 낸 `FrameState.face`와 `t_ms`만 쓴다).
    가정: 얼굴 검출 비율(시간 기준)의 분모는 받아들인 첫 프레임과 마지막 프레임 사이의 시간이다
    (프레임이 끊긴 시간도 분모에 들어간다. 마지막 프레임은 뒤 프레임이 없어 시간을 대표하지 않는다).
    """

    steps: tuple[StepFeatures, ...]
    runs: tuple[ClosureRun, ...]
    events: tuple[DriverStateEvent, ...]
    baseline: Baseline | None
    diagnostics: BaselineDiagnostics | None
    n_frames: int
    n_dropped: int
    duration_ms: int
    face_ms: int

    @property
    def face_ratio(self) -> float:
        """얼굴 검출 비율(시간 기준) = face_ms / duration_ms. 길이가 0이면 0.0이다."""
        if self.duration_ms <= 0:
            return 0.0
        return self.face_ms / self.duration_ms


def default_rules(spec: PipelineSpec) -> RulesConfig:
    """저장소의 config/rules.yaml. 시계 설정만 spec의 것으로 맞춘다(`DriverPipeline`이 둘이 같기를 요구한다)."""
    repo_root = Path(__file__).resolve().parents[1]
    return dataclasses.replace(load_rules(repo_root / DEFAULT_RULES_PATH), clock=spec.clock)


def replay_video(
    frames: Iterable[RawFrame],
    spec: PipelineSpec,
    baseline: Baseline | None,
    feature_start_ms: int | None,
    *,
    rules: RulesConfig | None = None,
) -> ReplayResult:
    """저장된 `RawFrame`을 `DriverPipeline`에 차례로 넣고 나온 것을 모은다.

    - baseline이 None이면 앱처럼 세션 앞부분에서 보정한다(feature_start_ms도 None이어야 한다).
    - baseline을 주면 보정을 건너뛰고 그 기준선을 쓰며, 특징 누적은 세션 시간 feature_start_ms부터 시작한다
      (None이면 `baseline.t_complete_ms`).
    - rules: 이벤트 판정에 쓰는 규칙. None이면 `default_rules(spec)`. 스텝 특징은 규칙과 무관하다.
    모델은 넣지 않는다(확률이 필요한 평가는 스텝 표의 창으로 따로 추론한다). 차량 속도도 넣지 않는다.
    가정: 계획서의 시그니처에는 규칙 인자가 없지만 `DriverPipeline`은 규칙 설정을 요구한다. 선택 인자로 받고
    주지 않으면 저장소의 기본 규칙을 쓴다.
    """
    pipe = DriverPipeline(spec, rules if rules is not None else default_rules(spec), baseline, feature_start_ms)
    max_hold_ms = spec.clock.max_hold_ms
    steps: list[StepFeatures] = []
    runs: list[ClosureRun] = []
    events: list[DriverStateEvent] = []
    first_t_ms = 0
    last_t_ms = 0
    last_face = False
    face_ms = 0
    for frame in frames:
        out = pipe.update(frame)
        if out is None:
            continue
        t_ms = out.state.t_ms
        if pipe.frames_accepted == 1:
            first_t_ms = t_ms
        elif last_face:
            face_ms += min(t_ms - last_t_ms, max_hold_ms)
        last_t_ms = t_ms
        last_face = out.state.face
        steps.extend(out.steps)
        runs.extend(out.runs)
        events.extend(out.events)
    return ReplayResult(
        steps=tuple(steps),
        runs=tuple(runs),
        events=tuple(events),
        baseline=pipe.baseline,
        diagnostics=pipe.baseline_diagnostics,
        n_frames=pipe.frames_accepted,
        n_dropped=pipe.dropped_frames,
        duration_ms=last_t_ms - first_t_ms,
        face_ms=face_ms,
    )


# ---------------------------------------------------------------------------
# 스텝 표
# ---------------------------------------------------------------------------


FRAME_DROP_METHOD = "bernoulli-v1"  # 프레임 누락 변형의 표집 절차 이름(캐시 열쇠에 들어간다). 절차를 바꾸면 이름도 바꾼다
_FRAME_DROP_SEED_PREFIX = "hawkeye.frame_drop"
FRAME_DROP_QUALITY_COLUMNS: tuple[str, ...] = ("drop_rate", "n_source_frames", "n_thinned")


@dataclass(frozen=True, slots=True)
class FrameDropSpec:
    """프레임 누락 변형 하나의 설정 (PLAN 5.2절 증강 (2)).

    - variant: 변형 번호(0 이상). 같은 시드에서 번호가 다르면 다른 누락 무늬가 나온다.
    - seed: 시드(정수).
    - drop_min, drop_max: 세션마다 뽑는 누락 비율의 범위(`training.frame_drop_min`, `training.frame_drop_max`).
    `0 ≤ drop_min ≤ drop_max < 1`이 아니거나 variant가 음수이면 ValueError.
    """

    variant: int
    seed: int
    drop_min: float
    drop_max: float

    def __post_init__(self) -> None:
        if isinstance(self.variant, bool) or not isinstance(self.variant, int) or self.variant < 0:
            raise ValueError(f"frame-drop variant must be an integer >= 0, got {self.variant!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"frame-drop seed must be an integer, got {self.seed!r}")
        if not 0.0 <= self.drop_min <= self.drop_max < 1.0:
            raise ValueError(
                f"frame-drop rates need 0 <= drop_min <= drop_max < 1, got {self.drop_min!r} and {self.drop_max!r}"
            )


def frame_drop_plan(session_id: str, n_frames: int, drop: FrameDropSpec) -> tuple[float, list[bool]]:
    """세션 하나의 프레임 누락 계획: (누락 비율, 저장된 프레임마다 남기는지 여부). (PLAN 5.2절 증강 (2))

    절차(`FRAME_DROP_METHOD = "bernoulli-v1"`)
    1. 난수 생성기는 `random.Random(n)`이고 n은
       `sha256("hawkeye.frame_drop|{seed}|{variant}|{session_id}")`의 앞 8바이트를 빅 엔디언 정수로 읽은 값이다.
    2. 첫 난수 u0로 비율을 정한다: `rate = drop_min + (drop_max − drop_min) · u0`
       (부동소수 반올림으로 범위를 벗어나지 않게 `[drop_min, drop_max]`로 자른다).
    3. 저장된 순서대로 두 번째 프레임부터 프레임마다 난수 u를 하나씩 뽑아 `u < rate`이면 버린다.
       첫 프레임은 항상 남긴다(세션 원점, 따라서 스텝 격자가 원본과 같게 유지된다).
    같은 (seed, variant, session_id)이면 프로세스, 실행 순서, `PYTHONHASHSEED`, workers 수와 무관하게 같은 계획이 나온다.
    가정: 계획서는 "프레임을 무작위로 10~30% 버린다"만 적었다. 비율은 세션마다 한 번 균등 분포에서 뽑고,
    프레임은 서로 독립으로 버린다(연속 누락을 따로 흉내 내지 않는다. 기기 동작의 모사는 증강 (1)이 맡는다).
    가정: 난수는 넘파이가 아니라 표준 라이브러리의 메르센 트위스터를 쓴다(정수 시드의 `random()` 수열은 파이썬
    판이 바뀌어도 유지된다고 문서가 보장한다).
    """
    if n_frames < 0:
        raise ValueError(f"n_frames must be >= 0, got {n_frames}")
    text = f"{_FRAME_DROP_SEED_PREFIX}|{drop.seed}|{drop.variant}|{session_id}"
    rng = random.Random(int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big"))
    rate = drop.drop_min + (drop.drop_max - drop.drop_min) * rng.random()
    rate = min(max(rate, drop.drop_min), drop.drop_max)
    keep = [True] * n_frames
    for index in range(1, n_frames):
        keep[index] = not rng.random() < rate
    return rate, keep


@dataclass(frozen=True, slots=True, eq=False)
class StepTable:
    """정규화 조건 하나의 스텝 표와 영상 품질 표.

    steps: 스텝마다 한 행. 열은 `STEP_COLUMNS`(식별자 7개, `k`, `t_end_ms`, `full_lookback`, 원시 특징 17개).
      세션 안에서는 k 오름차순이고 세션 순서는 목록과 같다.
    quality: 세션(파일)마다 한 행, 목록과 같은 순서. 열:
      - 식별자 7개
      - `raw_status`: `ok` 또는 `missing`(추출 결과 parquet이 없음)
      - `n_frames`, `n_dropped`, `duration_ms`, `face_ms`, `face_ratio`(얼굴 검출 비율, 시간 기준)
      - `baseline_status`: 이 세션 자신의 보정 결과(`ok`, `off_axis`, `fallback_global`, 보정이 안 끝났으면 `none`)
      - `t_b_ms`: 이 세션 자신의 기준선 완성 시각 T_b(세션 시간). 보정이 안 끝났으면 −1
      - `base_e_l`, `base_e_r`, `base_b_l`, `base_b_r`, `base_jaw0`: 이 세션 자신의 기준선 값(없으면 NaN)
      - `calib_face_frames`, `calib_face_ratio`: 보정 구간의 얼굴 프레임 수와 얼굴 검출 비율(없으면 0)
      - `baseline_used`: 이 조건에서 특징에 쓴 기준선(`session`, `alert`, `global`, 스텝이 없으면 `none`)
      - `alert_fallback`: alert 영상 기준선 조건에서 피험자의 alert 기준선을 쓸 수 없어 전역 상수로 대체했으면 True.
        그 피험자의 모든 세션에 표시한다(스텝이 없는 세션 포함). 다른 조건에서는 항상 False다
      - `n_steps`, `n_full_steps`, `n_windows`(보폭 1에서 이 세션이 내는 창 수)
      - `video_face_ratio`, `video_n_windows`: 논리 영상 단위로 합친 값(나뉜 영상의 부분들을 합친다)
      - `gate_pass`: D7 품질 기준(`video_face_ratio ≥ 0.7`이고 `video_n_windows ≥ 1`)을 통과했는지
    frame_drop: 프레임 누락 변형이면 그 설정, 원본이면 None.

    프레임 누락 변형(`load_frame_drop_tables`)의 품질 표는 열이 셋 더 있고 뜻이 조금 다르다.
      - `drop_rate`: 이 세션에 적용한 누락 비율. 솎지 않은 세션(추출 결과가 없거나 보정이 안 끝나 스텝이 없는 세션)은 NaN
      - `n_source_frames`: 솎기 전 저장된 프레임 수. `n_thinned`: 솎아 낸 프레임 수.
        솎은 세션에서는 `n_frames + n_dropped = n_source_frames − n_thinned`다
      - `n_frames`, `n_dropped`, `duration_ms`, `face_ms`, `face_ratio`는 솎은 프레임을 재생한 값이다.
        기준선에 관한 열(`baseline_status`, `t_b_ms`, `base_*`, `calib_*`)은 솎지 않은 원본 재생의 값 그대로다.
    """

    arm: str
    pipeline_hash: str
    steps: pd.DataFrame
    quality: pd.DataFrame
    frame_drop: FrameDropSpec | None = None


@dataclass(frozen=True, slots=True)
class _SessionRef:
    row: int  # 목록에서의 행 위치
    session_id: str
    class_index: int
    part: int
    raw_path: str


@dataclass(frozen=True, slots=True)
class _SubjectJob:
    sessions: tuple[_SessionRef, ...]
    spec: PipelineSpec
    rules: RulesConfig
    arms: tuple[str, ...]
    global_baseline: Baseline  # 전역 조건에 넣는 기준선
    frame_drop: FrameDropSpec | None = None  # 주면 특징 재생에 솎은 프레임을 쓴다


@dataclass(slots=True, eq=False)
class _SessionOutput:
    row: int
    info: dict[str, Any]  # 조건과 무관한 품질 통계
    arm_info: dict[str, dict[str, Any]]  # 조건 → baseline_used, alert_fallback
    k: np.ndarray  # 스텝 번호. 세 조건이 같다
    t_end_ms: np.ndarray
    full: np.ndarray
    values: dict[str, np.ndarray]  # 조건 → [스텝 수, 17] float64


def spec_pipeline_hash(spec: PipelineSpec) -> str:
    """`PipelineSpec`의 pipeline_hash(feature_spec.json의 `pipeline_hash`와 같은 값)."""
    return compute_pipeline_hash(pipeline_to_dict(spec))


def build_step_table(
    manifest: pd.DataFrame,
    raw_dir: Path,
    spec: PipelineSpec,
    arm: str,
    global_override: Baseline | None = None,
    *,
    gate: QualityGate | None = None,
    workers: int = 1,
    frame_drop: FrameDropSpec | None = None,
) -> StepTable:
    """목록의 모든 세션을 재생해 정규화 조건 하나의 스텝 표를 만든다 (PLAN 4.4절, 1.2절의 결정 D1과 D7).

    D1은 정규화 비교에 세 번째 조건(영상 자신의 앞 60초 기준선, 앱의 동작)을 더한 결정이고, D7은 품질 기준이다.

    추출 결과는 `raw_dir / rel_path`(확장자만 `.parquet`으로 바꾼 경로)에서 읽는다.
    세션마다 먼저 앱처럼 자기 앞부분으로 보정해 본다. 그 기준선이 완성된 시각 T_b가 세 조건 모두에서 그 세션의
    특징 누적 시작 시각이다. 그래서 보정에 쓰인 프레임은 어느 조건에서도 특징에 들어가지 않고, 세 조건이 같은
    스텝(따라서 같은 창)을 낸다. 조건은 특징을 계산할 때 쓰는 기준선만 다르다.
    - `session_baseline`: 그 세션 자신의 기준선(앱의 동작).
    - `alert_baseline`: 피험자의 alert 영상 기준선. alert 영상은 자기 기준선이므로 위 조건과 같고,
      나머지 영상에는 alert 영상의 기준선을 넣는다. alert 영상의 보정이 전역 상수로 대체됐거나 끝나지 않았으면
      그 피험자의 모든 세션에 spec의 전역 상수를 넣고 `alert_fallback`을 True로 표시한다
      (다른 피험자의 기준선이나 다른 라벨 영상의 기준선은 쓰지 않는다).
    - `global`: 전역 상수 기준선. `global_override`를 주면 spec의 상수 대신 그것을 넣는다(fold별 상수 민감도 점검용).
    자기 보정이 끝나지 않은 세션(60초보다 짧은 영상 등)은 어느 조건에서도 스텝이 없다.
    추출 결과가 없는 세션은 스텝 없이 `raw_status = missing`으로 품질 표에 남는다(D7: 창이 없는 영상은 오답으로 센다).

    - gate: D7 품질 기준. None이면 `DEFAULT_QUALITY_GATE`(얼굴 검출 비율 0.7 이상, 창 1개 이상).
    - workers: 피험자 단위 병렬 프로세스 수. 결과는 workers와 무관하다.
    - frame_drop: 주면 프레임 누락 변형을 만든다(캐시 없이. 뜻과 가정은 `load_frame_drop_tables`의 설명에 있다).
    가정: 계획서의 시그니처는 표 하나를 돌려주지만, 스텝 표와 영상 품질 표를 `StepTable`로 묶어 돌려준다.
    가정: "쓸 수 있는 alert 기준선"은 alert 영상의 보정이 대체 없이 완성된 것(status가 ok 또는 off_axis)이다.
    가정: alert 영상이 나뉘어 있으면 첫 부분의 기준선을 피험자의 alert 기준선으로 쓴다.
    가정: 품질 기준의 "창 수"는 보폭 1에서 세고, 얼굴 검출 비율은 보정 구간을 포함한 영상 전체에서 잰다.
    가정: 추출 결과가 하나도 없으면 raw_dir이 틀린 것으로 보고 `ManifestError`를 낸다.
    """
    return _build_step_tables(manifest, Path(raw_dir), spec, (arm,), global_override, gate, workers, frame_drop)[arm]


def _build_step_tables(
    manifest: pd.DataFrame,
    raw_dir: Path,
    spec: PipelineSpec,
    arms: Sequence[str],
    global_override: Baseline | None,
    gate: QualityGate | None,
    workers: int,
    frame_drop: FrameDropSpec | None = None,
) -> dict[str, StepTable]:
    """여러 조건의 스텝 표를 한 번에 만든다. 파일 읽기와 자기 보정 재생을 조건들이 함께 쓴다.

    frame_drop을 주면 프레임 누락 변형을 만든다(`load_frame_drop_tables`의 설명 참고).
    """
    for arm in arms:
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    missing_columns = [name for name in MANIFEST_COLUMNS if name not in manifest.columns]
    if missing_columns:
        raise ManifestError(f"manifest lacks columns {missing_columns}; build it with build_manifest")
    _check_folds(manifest)  # 데이터 구성 시점의 피험자 분리 검사(PLAN 6장)

    raw_paths = _raw_paths(manifest, raw_dir)
    if not any(path.is_file() for path in raw_paths):
        raise ManifestError(
            f"none of the {len(raw_paths)} raw parquet files of the manifest exists below {raw_dir} "
            f"(first: {raw_paths[0] if raw_paths else '-'}); run the extraction or fix raw_dir"
        )

    rules = default_rules(spec)
    global_baseline = global_override if global_override is not None else calibrate.global_baseline(spec)
    by_subject: dict[int, list[_SessionRef]] = {}
    for row in range(len(manifest)):
        ref = _SessionRef(
            row=row,
            session_id=str(manifest["session_id"].iloc[row]),
            class_index=int(manifest["class_index"].iloc[row]),
            part=int(manifest["part"].iloc[row]),
            raw_path=str(raw_paths[row]),
        )
        by_subject.setdefault(int(manifest["subject"].iloc[row]), []).append(ref)
    jobs = [
        _SubjectJob(tuple(refs), spec, rules, tuple(arms), global_baseline, frame_drop) for refs in by_subject.values()
    ]

    outputs: list[_SessionOutput] = []
    if workers == 1:
        for job in jobs:
            outputs.extend(_process_subject(job))
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            for result in pool.map(_process_subject, jobs):
                outputs.extend(result)
    outputs.sort(key=lambda output: output.row)

    hash_value = spec_pipeline_hash(spec)
    window_steps = spec.step_features.window_steps
    tables: dict[str, StepTable] = {}
    for arm in arms:
        steps = _steps_frame(manifest, outputs, arm)
        quality = apply_quality_gate(_quality_frame(manifest, outputs, arm, window_steps), gate)
        tables[arm] = StepTable(
            arm=arm, pipeline_hash=hash_value, steps=steps, quality=quality, frame_drop=frame_drop
        )
    return tables


def _raw_paths(manifest: pd.DataFrame, raw_dir: Path) -> list[Path]:
    return [raw_parquet_path(raw_dir, Path(rel_path)) for rel_path in manifest["rel_path"].tolist()]


def _process_subject(job: _SubjectJob) -> list[_SessionOutput]:
    """피험자 한 명의 세션을 모두 재생한다. alert 영상을 먼저 처리해 그 기준선을 나머지 세션에 넣는다.

    `job.frame_drop`이 있으면(프레임 누락 변형) 기준선과 T_b는 원본과 똑같이 솎지 않은 프레임의 자기 보정 재생에서
    얻고, 특징을 내는 재생에만 솎은 프레임을 쓴다. 자기 기준선 조건에도 그 (솎지 않은 프레임으로 보정한) 자기
    기준선을 넣어 준다. 그래서 원본 표와 다른 것은 특징에 들어가는 프레임뿐이다.
    """
    alert_refs = [ref for ref in job.sessions if ref.class_index == CLASS_ALERT]
    source = min(alert_refs, key=lambda ref: ref.part) if alert_refs else None
    ordered = ([source] if source is not None else []) + [ref for ref in job.sessions if ref is not source]

    spec_global = calibrate.global_baseline(job.spec)
    alert_baseline: Baseline | None = None
    outputs: list[_SessionOutput] = []
    for ref in ordered:
        if not Path(ref.raw_path).is_file():
            outputs.append(_empty_output(ref, job, RAW_MISSING, None, alert_baseline is None, 0))
            continue
        frames = list(read_raw_frames(Path(ref.raw_path)))  # 같은 프레임을 조건마다 다시 재생한다
        own = replay_video(frames, job.spec, None, None, rules=job.rules)
        if ref is source and own.baseline is not None and own.baseline.status != BASELINE_FALLBACK_GLOBAL:
            alert_baseline = own.baseline
        if own.baseline is None:
            outputs.append(_empty_output(ref, job, RAW_OK, own, alert_baseline is None, len(frames)))
            continue

        t_b_ms = own.baseline.t_complete_ms
        info = _session_info(RAW_OK, own)
        feature_frames = frames
        # 스텝 번호의 기준이 되는 재생. 원본은 자기 보정 재생이고, 솎은 변형은 처음 돌린 조건의 재생이다.
        reference: ReplayResult | None = own
        drop_rate = math.nan
        if job.frame_drop is not None:
            drop_rate, keep = frame_drop_plan(ref.session_id, len(frames), job.frame_drop)
            feature_frames = [frame for frame, kept in zip(frames, keep) if kept]
            reference = None

        replays: list[tuple[Baseline, ReplayResult]] = []  # (넣은 기준선, 재생). 같은 기준선을 쓰는 조건은 한 번만 재생한다
        values: dict[str, np.ndarray] = {}
        arm_used: dict[str, tuple[str, bool]] = {}
        for arm in job.arms:
            used, injected, fallback = _arm_baseline(
                arm, ref is source, alert_baseline, job.global_baseline, spec_global
            )
            if injected is None and job.frame_drop is None:
                result = own
            else:
                baseline = own.baseline if injected is None else injected
                result = next((done for used_baseline, done in replays if used_baseline == baseline), None)
                if result is None:
                    result = replay_video(feature_frames, job.spec, baseline, t_b_ms, rules=job.rules)
                    replays.append((baseline, result))
                if reference is None:
                    reference = result
                same = len(result.steps) == len(reference.steps) and all(
                    a.k == b.k and a.full_lookback == b.full_lookback for a, b in zip(result.steps, reference.steps)
                )
                if not same:
                    raise RuntimeError(
                        f"{ref.session_id}: arm {arm} emitted different steps than the session's reference replay"
                    )
            values[arm] = _values_array(result.steps)
            arm_used[arm] = (used, fallback)
        if reference is None:  # 조건이 하나도 없을 때(부를 일은 없다)
            reference = own
        if job.frame_drop is not None:
            # 솎은 프레임을 재생한 계수로 바꾼다. 얼굴 검출 여부는 기준선과 무관하므로 어느 조건의 재생이든 같다.
            info.update(
                n_frames=reference.n_frames,
                n_dropped=reference.n_dropped,
                duration_ms=reference.duration_ms,
                face_ms=reference.face_ms,
                face_ratio=reference.face_ratio,
                drop_rate=drop_rate,
                n_source_frames=len(frames),
                n_thinned=len(frames) - len(feature_frames),
            )
        outputs.append(
            _SessionOutput(
                row=ref.row,
                info=info,
                arm_info={
                    arm: {
                        "baseline_used": used if len(reference.steps) > 0 else BASELINE_NONE,
                        "alert_fallback": fallback,
                    }
                    for arm, (used, fallback) in arm_used.items()
                },
                k=np.array([step.k for step in reference.steps], dtype=np.int64),
                t_end_ms=np.array([step.t_end_ms for step in reference.steps], dtype=np.int64),
                full=np.array([step.full_lookback for step in reference.steps], dtype=bool),
                values=values,
            )
        )
    return outputs


def _arm_baseline(
    arm: str,
    is_alert_source: bool,
    alert_baseline: Baseline | None,
    global_baseline: Baseline,
    spec_global: Baseline,
) -> tuple[str, Baseline | None, bool]:
    """조건 arm에서 세션에 쓸 기준선: (`baseline_used`, 넣을 기준선, `alert_fallback`).

    넣을 기준선이 None이면 그 세션 자신의 보정 재생을 그대로 쓴다(프레임 누락 변형에서는 그 재생이 낸 자기
    기준선을 솎은 프레임의 재생에 넣는다).
    """
    if arm == ARM_SESSION_BASELINE:
        return "session", None, False
    if arm == ARM_GLOBAL:
        return "global", global_baseline, False
    if arm != ARM_ALERT_BASELINE:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    if alert_baseline is None:  # 쓸 수 있는 alert 기준선이 없는 피험자: spec의 전역 상수로 대체하고 표시한다
        return "global", spec_global, True
    if is_alert_source:  # 자기 기준선이 곧 alert 영상 기준선이다
        return "alert", None, False
    return "alert", alert_baseline, False


def _values_array(steps: Sequence[StepFeatures]) -> np.ndarray:
    if len(steps) == 0:
        return np.zeros((0, N_FEATURES), dtype=np.float64)
    return np.array([step.values for step in steps], dtype=np.float64)


def _session_info(raw_status: str, own: ReplayResult | None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "raw_status": raw_status,
        "n_frames": 0,
        "n_dropped": 0,
        "duration_ms": 0,
        "face_ms": 0,
        "face_ratio": 0.0,
        "baseline_status": BASELINE_NONE,
        "t_b_ms": -1,
        "base_e_l": math.nan,
        "base_e_r": math.nan,
        "base_b_l": math.nan,
        "base_b_r": math.nan,
        "base_jaw0": math.nan,
        "calib_face_frames": 0,
        "calib_face_ratio": 0.0,
    }
    if own is None:
        return info
    info.update(
        n_frames=own.n_frames,
        n_dropped=own.n_dropped,
        duration_ms=own.duration_ms,
        face_ms=own.face_ms,
        face_ratio=own.face_ratio,
    )
    if own.baseline is not None:
        info.update(
            baseline_status=own.baseline.status,
            t_b_ms=own.baseline.t_complete_ms,
            base_e_l=own.baseline.e_l,
            base_e_r=own.baseline.e_r,
            base_b_l=own.baseline.b_l,
            base_b_r=own.baseline.b_r,
            base_jaw0=own.baseline.jaw0,
        )
    if own.diagnostics is not None:
        info.update(calib_face_frames=own.diagnostics.n_face_frames, calib_face_ratio=own.diagnostics.face_ratio)
    return info


def _empty_output(
    ref: _SessionRef,
    job: _SubjectJob,
    raw_status: str,
    own: ReplayResult | None,
    no_alert_baseline: bool,
    n_source_frames: int,
) -> _SessionOutput:
    """스텝이 없는 세션(추출 결과가 없거나 보정이 끝나기 전에 끝난 영상). 프레임 누락 변형에서도 솎지 않는다."""
    info = _session_info(raw_status, own)
    if job.frame_drop is not None:
        info.update(drop_rate=math.nan, n_source_frames=n_source_frames, n_thinned=0)
    return _SessionOutput(
        row=ref.row,
        info=info,
        arm_info={
            arm: {"baseline_used": BASELINE_NONE, "alert_fallback": no_alert_baseline and arm == ARM_ALERT_BASELINE}
            for arm in job.arms
        },
        k=np.zeros(0, dtype=np.int64),
        t_end_ms=np.zeros(0, dtype=np.int64),
        full=np.zeros(0, dtype=bool),
        values={arm: np.zeros((0, N_FEATURES), dtype=np.float64) for arm in job.arms},
    )


def _steps_frame(manifest: pd.DataFrame, outputs: Sequence[_SessionOutput], arm: str) -> pd.DataFrame:
    counts = np.array([len(output.k) for output in outputs], dtype=np.int64)
    rows = np.repeat(np.array([output.row for output in outputs], dtype=np.int64), counts)
    data: dict[str, Any] = {name: manifest[name].to_numpy()[rows] for name in ID_COLUMNS}
    data["k"] = np.concatenate([output.k for output in outputs]) if len(outputs) else np.zeros(0, dtype=np.int64)
    data["t_end_ms"] = (
        np.concatenate([output.t_end_ms for output in outputs]) if len(outputs) else np.zeros(0, dtype=np.int64)
    )
    data["full_lookback"] = (
        np.concatenate([output.full for output in outputs]) if len(outputs) else np.zeros(0, dtype=bool)
    )
    values = (
        np.concatenate([output.values[arm] for output in outputs])
        if len(outputs)
        else np.zeros((0, N_FEATURES), dtype=np.float64)
    )
    for index, name in enumerate(FEATURE_NAMES):
        data[name] = values[:, index]
    return pd.DataFrame(data, columns=list(STEP_COLUMNS))


def _quality_frame(
    manifest: pd.DataFrame, outputs: Sequence[_SessionOutput], arm: str, window_steps: int
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for output in outputs:
        record: dict[str, Any] = {name: manifest[name].iloc[output.row] for name in ID_COLUMNS}
        record.update(output.info)
        record.update(output.arm_info[arm])
        record["n_steps"] = int(len(output.k))
        record["n_full_steps"] = int(output.full.sum())
        record["n_windows"] = sum(
            stop - start - window_steps + 1 for start, stop in _full_runs(output.k, output.full, window_steps)
        )
        records.append(record)
    return pd.DataFrame(records)


def apply_quality_gate(quality: pd.DataFrame, gate: QualityGate | None = None) -> pd.DataFrame:
    """품질 표에 논리 영상 단위의 값과 D7 판정을 붙인 사본을 돌려준다.

    `video_face_ratio`는 부분들의 얼굴 시간 합 / 길이 합, `video_n_windows`는 부분들의 창 수 합이다.
    `gate_pass = video_face_ratio ≥ gate.min_face_ratio and video_n_windows ≥ gate.min_windows`.
    라벨을 쓰지 않는다. gate가 None이면 `DEFAULT_QUALITY_GATE`.
    """
    if gate is None:
        gate = DEFAULT_QUALITY_GATE
    out = quality.drop(columns=["video_face_ratio", "video_n_windows", "gate_pass"], errors="ignore").copy()
    grouped = out.groupby("video_id", sort=False)
    face_ms = grouped["face_ms"].transform("sum").to_numpy(dtype=np.float64)
    duration_ms = grouped["duration_ms"].transform("sum").to_numpy(dtype=np.float64)
    ratio = np.zeros(len(out), dtype=np.float64)
    np.divide(face_ms, duration_ms, out=ratio, where=duration_ms > 0)
    windows = grouped["n_windows"].transform("sum").to_numpy(dtype=np.int64)
    out["video_face_ratio"] = ratio
    out["video_n_windows"] = windows
    out["gate_pass"] = (ratio >= gate.min_face_ratio) & (windows >= gate.min_windows)
    return out


# ---------------------------------------------------------------------------
# 창
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class WindowSet:
    """모델 입력 창의 묶음.

    - x: float32 `[N, window_steps, 17]`. 표준화하지 않은 원시 스텝 특징이고 `x[i, 0]`이 가장 오래된 스텝이다
      (ONNX 입력 `features`와 같은 배치).
    - y: int64 `[N]`. 클래스 인덱스(0 alert, 1 low_vigilant, 2 drowsy).
    - meta: 창마다 한 행. 식별자 7개와 `k_end`(창의 마지막 스텝 번호), `t_end_ms`(그 스텝의 끝 시각).
    """

    x: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def subset(self, selector: Any) -> WindowSet:
        """창을 골라낸 새 묶음. selector는 길이 N의 bool 배열이나 정수 위치 배열이다."""
        index = np.asarray(selector)
        if index.dtype == bool:
            if index.shape != (len(self),):
                raise ValueError(f"boolean selector must have shape ({len(self)},), got {index.shape}")
            index = np.flatnonzero(index)
        return WindowSet(x=self.x[index], y=self.y[index], meta=self.meta.iloc[index].reset_index(drop=True))


def _full_runs(k: np.ndarray, full: np.ndarray, min_length: int) -> list[tuple[int, int]]:
    """스텝 번호가 이어지고 모두 full_lookback인 구간 [start, stop) 중 길이가 min_length 이상인 것."""
    runs: list[tuple[int, int]] = []
    n = len(k)
    start = 0
    while start < n:
        if not full[start]:
            start += 1
            continue
        stop = start + 1
        while stop < n and full[stop] and k[stop] == k[stop - 1] + 1:
            stop += 1
        if stop - start >= min_length:
            runs.append((start, stop))
        start = stop
    return runs


def make_windows(steps: pd.DataFrame, window_steps: int = 120, stride: int = 5) -> WindowSet:
    """스텝 표에서 모델 입력 창을 만든다 (PLAN 4.4절).

    창은 한 세션의, 스텝 번호가 이어진 `full_lookback` 스텝 window_steps개다. 조회 구간이 덜 찬 스텝은 어느 창에도
    들어가지 않고, 창은 세션(나뉜 영상의 부분)의 경계나 10초 넘는 끊김(스텝 번호가 건너뛴다)을 넘지 않는다.
    600초 영상(T_b = 60초)은 보폭 1에서 362개, 보폭 10에서 37개가 나온다.
    창의 순서는 스텝 표의 세션 순서, 그 안에서는 시간순이다.
    가정: 이어진 구간마다 첫 창은 그 구간의 첫 window_steps개이고(앱이 처음 확률을 내는 시점과 같다) 그 뒤로 stride
    스텝씩 옮긴다. 학습의 "시작 위치 무작위 이동"은 보폭 1로 만든 창에서 골라 쓰면 된다(`WindowSet.subset`).
    """
    if window_steps < 1 or stride < 1:
        raise ValueError("window_steps and stride must be >= 1")
    missing_columns = [name for name in STEP_COLUMNS if name not in steps.columns]
    if missing_columns:
        raise ValueError(f"steps table lacks columns {missing_columns}")

    blocks: list[np.ndarray] = []
    meta_rows: list[np.ndarray] = []  # 창마다 그 마지막 스텝의 행 위치
    session_ids = steps["session_id"].to_numpy()
    k_all = steps["k"].to_numpy(dtype=np.int64)
    full_all = steps["full_lookback"].to_numpy(dtype=bool)
    values_all = steps[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    positions: dict[Any, list[int]] = {}
    for position, session_id in enumerate(session_ids.tolist()):
        positions.setdefault(session_id, []).append(position)
    for session_rows in positions.values():
        rows = np.array(session_rows, dtype=np.int64)
        rows = rows[np.argsort(k_all[rows], kind="stable")]
        k = k_all[rows]
        if len(k) > 1 and np.any(np.diff(k) <= 0):
            raise ValueError(f"session {session_ids[rows[0]]!r} has duplicate step numbers")
        for start, stop in _full_runs(k, full_all[rows], window_steps):
            values = values_all[rows[start:stop]]
            # [길이 − W + 1, F, W] → [.., W, F]
            view = np.lib.stride_tricks.sliding_window_view(values, window_steps, axis=0)[::stride]
            blocks.append(np.ascontiguousarray(view.transpose(0, 2, 1), dtype=np.float32))
            meta_rows.append(rows[np.arange(start + window_steps - 1, stop, stride)])

    if blocks:
        x = np.concatenate(blocks)
        last_rows = np.concatenate(meta_rows)
    else:
        x = np.zeros((0, window_steps, N_FEATURES), dtype=np.float32)
        last_rows = np.zeros(0, dtype=np.int64)
    meta = steps[list(ID_COLUMNS)].iloc[last_rows].reset_index(drop=True)
    meta["k_end"] = k_all[last_rows]
    meta["t_end_ms"] = steps["t_end_ms"].to_numpy(dtype=np.int64)[last_rows]
    y = meta["class_index"].to_numpy(dtype=np.int64)
    return WindowSet(x=x, y=y, meta=meta)


# ---------------------------------------------------------------------------
# 중첩 교차검증 분할
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InnerSplit:
    """바깥 학습 fold들 안에서 검증 fold 하나를 뺀 분할."""

    val_fold: int
    train_subjects: tuple[int, ...]
    val_subjects: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FoldSplit:
    """바깥 시험 fold 하나와 그 안쪽 분할들 (PLAN 6장).

    train_subjects는 시험 fold를 뺀 전체(선택이 끝난 뒤 다시 학습할 때 쓰는 집합)다.
    60명이면 시험 12명, train_subjects 48명, 안쪽 분할 4개(학습 36명, 검증 12명)다.
    """

    test_fold: int
    test_subjects: tuple[int, ...]
    train_subjects: tuple[int, ...]
    inner: tuple[InnerSplit, ...]


def nested_splits(manifest: pd.DataFrame) -> Iterator[FoldSplit]:
    """목록의 fold마다, 그 fold를 시험으로 두고 나머지 fold에서 검증 fold를 하나씩 돌린 분할을 낸다 (PLAN 6장).

    fold는 목록의 `fold` 열(공식 대응)이고 번호순으로 낸다. 모든 분할에서 `assert_subject_disjoint`를 부른다.
    가정: fold가 3개 미만이면 안쪽 분할의 학습 집합이 비므로 ValueError다.
    """
    folds = _check_folds(manifest)
    if len(folds) < 3:
        raise ValueError(f"nested splits need at least 3 folds, the manifest has {len(folds)}")
    for test_fold, test_subjects in folds.items():
        outer_train = tuple(sorted(s for fold, subjects in folds.items() if fold != test_fold for s in subjects))
        assert_subject_disjoint(outer_train, test_subjects)
        inner: list[InnerSplit] = []
        for val_fold, val_subjects in folds.items():
            if val_fold == test_fold:
                continue
            train = tuple(
                sorted(s for fold, subjects in folds.items() if fold not in (test_fold, val_fold) for s in subjects)
            )
            assert_subject_disjoint(train, val_subjects, test_subjects)
            inner.append(InnerSplit(val_fold=val_fold, train_subjects=train, val_subjects=val_subjects))
        yield FoldSplit(
            test_fold=test_fold, test_subjects=test_subjects, train_subjects=outer_train, inner=tuple(inner)
        )


# ---------------------------------------------------------------------------
# 스텝 표 캐시
# ---------------------------------------------------------------------------

CACHE_METADATA_KEY = b"drowsy_steps"
# 2: 열쇠에 코드 지문(`code`)과 프레임 누락 변형(`frame_drop`)이 들어간다. 3: 추출 결과의 지문(`source`)에 파일마다
# 꼬리(footer)의 sha256이 더 들어간다. 4: 꼬리 대신 파일 전체의 sha256이 들어간다(`raw_content_digest`. 꼬리가 같은
# 다른 내용, 예를 들어 두 프레임의 값을 맞바꾼 파일을 꼬리 해시는 구별하지 못했다). 번호가 다른 캐시는 다시 만든다
CACHE_FORMAT = 4

# 스텝 표의 값을 정하는 코드(이식 모듈과 그것을 부르는 쪽). 이 파일들의 바이트가 바뀌면 캐시를 다시 만든다.
FINGERPRINT_MODULES: tuple[str, ...] = (
    "types",
    "signals",
    "calibrate",
    "blink",
    "step_features",
    "rules",
    "pipeline",
    "spec",
    "extract",
    "dataset",
)


def fingerprint_files() -> tuple[Path, ...]:
    """코드 지문에 들어가는 파일들(`drowsy/{FINGERPRINT_MODULES}.py`, 고정된 순서)."""
    package_dir = Path(__file__).resolve().parent
    return tuple(package_dir / f"{name}.py" for name in FINGERPRINT_MODULES)


def code_fingerprint(files: Sequence[Path] | None = None) -> str:
    """스텝 표를 만드는 코드의 지문(sha256 16진수). 스텝 표 캐시의 열쇠에 들어간다.

    files(None이면 `fingerprint_files()`)를 주어진 순서대로, 파일마다
    `이름(UTF-8) + b"\\0" + 바이트 수(10진 ASCII) + b"\\0" + 파일 바이트`를 sha256에 넣는다.
    가정: 지문은 소스 파일의 바이트다. 주석이나 문서 문자열만 바꿔도 캐시를 다시 만든다(안전한 쪽으로 틀린다).
    특징을 바꾸지 않는 모듈(`rules`, `spec`, `extract`, `dataset`)도 재생 경로에 있어서 넣었다. 실행 중에 코드를
    바꿔치기한 것(monkeypatch)이나 파이썬·넘파이 판이 바뀐 것은 알아채지 못한다. 그럴 때는 `--rebuild`를 쓴다.
    """
    digest = hashlib.sha256()
    for path in fingerprint_files() if files is None else files:
        data = Path(path).read_bytes()
        digest.update(Path(path).name.encode("utf-8") + b"\0" + str(len(data)).encode("ascii") + b"\0")
        digest.update(data)
    return digest.hexdigest()


def step_cache_dir(raw_dir: Path) -> Path:
    """스텝 표 캐시의 기본 위치: raw_dir 옆의 `{raw_dir 이름}_steps` 폴더.

    가정: 계획서는 캐시의 위치를 정하지 않았다. raw_dir 안에 두지 않는 이유는 raw_dir이 `build_manifest`가
    훑는 폴더일 수 있기 때문이다(합성 데이터).
    """
    raw_dir = Path(raw_dir)
    return raw_dir.parent / f"{raw_dir.name}_steps"


def _baseline_hash(baseline: Baseline) -> str:
    encoded = json.dumps(dataclasses.asdict(baseline), sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# 파일 내용 지문의 기억(이 프로세스 안에서만): (심볼릭 링크를 푼 절대 경로, 크기, 수정 시각 ns) → 파일 전체의 sha256
_CONTENT_DIGESTS: dict[tuple[str, int, int], str] = {}
_HASH_BLOCK_BYTES = 1 << 20


def _read_file_sha256(path: Path) -> tuple[str, tuple[int, int] | None]:
    """파일을 처음부터 끝까지 읽어 sha256(16진수)과, 읽은 파일의 (크기, 수정 시각 ns)를 돌려준다.

    크기와 수정 시각은 연 파일에서 읽기 전과 후에 잰다. 읽는 동안 둘 중 하나라도 바뀌었으면 둘째 값은 None이다.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        before = os.fstat(handle.fileno())
        for block in iter(lambda: handle.read(_HASH_BLOCK_BYTES), b""):
            digest.update(block)
        after = os.fstat(handle.fileno())
    stamp = (before.st_size, before.st_mtime_ns)
    return digest.hexdigest(), stamp if stamp == (after.st_size, after.st_mtime_ns) else None


def raw_content_digest(path: Path, *, refresh: bool = False) -> str:
    """추출 결과 parquet 한 파일의 내용 지문: 파일 전체 바이트의 sha256(16진수).

    파일을 처음부터 끝까지 읽으므로 열 데이터, 스키마, 파일 메타데이터 가운데 한 바이트만 달라도 값이 달라진다.
    parquet인지는 보지 않는다(어떤 파일이든 같은 방식이다). 파일이 없으면 OSError다.

    같은 프로세스 안에서는 값을 `(심볼릭 링크를 푼 절대 경로, 크기, 수정 시각 ns)`별로 기억해 두고, 셋이 그대로인
    파일은 다시 읽지 않는다(학습 한 번이 `load_step_tables`를 여러 번 부르기 때문이다). 기억은 디스크에 남기지 않고
    프로세스가 끝나면 사라진다. 읽는 동안 크기나 수정 시각이 바뀐 파일의 값은 기억하지 않는다.
    - refresh: True면 기억을 보지 않고 다시 읽어 기억을 새 값으로 바꾼다(`load_step_tables(rebuild=True)`가 쓴다).

    알아채는 것: 내용이 바뀐 모든 경우. 크기가 같고 수정 시각까지 되돌려 놓은 파일(`cp -p`, `rsync -t`로 덮어쓴 것,
    프레임 몇 개의 값만 다른 것)도 다음 프로세스, 곧 다음 명령 실행에서는 파일을 다시 읽으므로 알아챈다.
    알아채지 못하는 것: 같은 프로세스가 도는 동안, 이미 읽은 파일의 내용을 크기와 수정 시각이 그대로인 채로 바꾼 경우
    (수정 시각을 되돌렸거나, 파일 시스템의 시각 해상도 안에서 같은 크기로 다시 쓴 경우). 그 프로세스는 기억해 둔 옛
    값을 돌려준다.
    가정: 한 프로세스가 도는 동안에는 추출 결과를 그런 식으로 바꾸지 않는다고 본다(추출과 학습은 서로 다른 명령이다).
      바꿨다면 그 프로세스에서는 `rebuild=True`로 다시 만든다. 새로 실행한 명령은 스스로 알아챈다.
    가정: 해시는 `code_fingerprint`와 같은 sha256이다(표준 라이브러리). 비용은 파일 크기에 비례한다. 합성 데이터
      (`data/synth`, 182개 파일 443 MB)에서 2026-11-14에 이 PC로 잰 값: 지문 전체(`_source_fingerprint`)의 프로세스 첫
      호출이 약 0.2초(파일들을 운영체제의 페이지 캐시에서 내보낸 뒤에는 약 0.4초), 그 뒤의 호출이 약 6 ms.
    """
    path = Path(path)
    resolved = os.path.realpath(path)
    if not refresh:
        stat = path.stat()
        known = _CONTENT_DIGESTS.get((resolved, stat.st_size, stat.st_mtime_ns))
        if known is not None:
            return known
    value, stamp = _read_file_sha256(path)
    if stamp is not None:
        _CONTENT_DIGESTS[(resolved, *stamp)] = value
    return value


def _source_fingerprint(manifest: pd.DataFrame, raw_dir: Path, refresh: bool = False) -> str:
    """목록과 추출 결과 파일들의 지문. 파일이 바뀌면 캐시를 다시 만든다.

    세션마다 `[session_id, rel_path, 크기, 수정 시각(ns), raw_content_digest]`를 모아 sha256에 넣는다(없는 파일은
    `[session_id, rel_path, -1, -1, ""]`). 다섯째 값은 파일 전체의 sha256이다. 크기와 수정 시각이 그대로여도 내용이
    한 바이트라도 다르면 지문이 달라진다. 한 프로세스에서 처음 부를 때 파일을 모두 읽고, 그 뒤로는 크기와 수정
    시각이 그대로인 파일을 다시 읽지 않는다(알아채는 것과 못 알아채는 것은 `raw_content_digest`의 설명에 있다).
    refresh가 True면 기억해 둔 값을 쓰지 않고 모두 다시 읽는다.
    """
    entries: list[list[Any]] = []
    for session_id, rel_path, path in zip(
        manifest["session_id"].tolist(), manifest["rel_path"].tolist(), _raw_paths(manifest, raw_dir)
    ):
        if path.is_file():
            stat = path.stat()
            digest = raw_content_digest(path, refresh=refresh)
            entries.append([session_id, rel_path, stat.st_size, stat.st_mtime_ns, digest])
        else:
            entries.append([session_id, rel_path, -1, -1, ""])
    return hashlib.sha256(json.dumps(entries).encode("utf-8")).hexdigest()


def _cache_paths(
    cache_dir: Path,
    arm: str,
    hash_value: str,
    override_hash: str | None,
    frame_drop: FrameDropSpec | None = None,
) -> tuple[Path, Path]:
    name = f"{arm}_{hash_value[:16]}"
    if override_hash is not None:
        name += f"_g{override_hash[:12]}"
    if frame_drop is not None:
        name += (
            f"_fd{frame_drop.variant}_s{frame_drop.seed}"
            f"_r{float(frame_drop.drop_min)!r}-{float(frame_drop.drop_max)!r}"
        )
    return cache_dir / f"{name}.steps.parquet", cache_dir / f"{name}.quality.parquet"


def _write_cached(frame: pd.DataFrame, path: Path, metadata: dict[str, Any]) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    merged = dict(table.schema.metadata or {})
    merged[CACHE_METADATA_KEY] = json.dumps(metadata, sort_keys=True).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    pq.write_table(table.replace_schema_metadata(merged), temp_path, compression="zstd")
    os.replace(temp_path, path)


def _read_cached(path: Path, metadata: dict[str, Any]) -> pd.DataFrame | None:
    """캐시 파일이 있고 메타데이터가 같으면 표를, 아니면 None을 돌려준다."""
    if not path.is_file():
        return None
    try:
        table = pq.read_table(path)
    except (OSError, pa.ArrowException):
        return None
    stored = (table.schema.metadata or {}).get(CACHE_METADATA_KEY)
    if stored is None or json.loads(stored.decode("utf-8")) != metadata:
        return None
    return table.to_pandas()


def load_step_tables(
    manifest: pd.DataFrame,
    raw_dir: Path,
    spec: PipelineSpec,
    arms: Sequence[str] = ARMS,
    *,
    cache_dir: Path | None = None,
    global_override: Baseline | None = None,
    gate: QualityGate | None = None,
    workers: int = 1,
    rebuild: bool = False,
) -> dict[str, StepTable]:
    """조건별 스텝 표를 캐시에서 읽고, 없는 조건은 한 번에 만들어 캐시에 쓴다. 조건 이름 → `StepTable`.

    캐시 파일은 `{cache_dir}/{조건}_{pipeline_hash 앞 16자}.steps.parquet`과 `.quality.parquet`이다
    (`global_override`를 준 전역 조건은 이름에 `_g{그 기준선 해시 앞 12자}`가 붙어 기준선마다 따로 캐시된다.
    다른 조건은 `global_override`와 무관하므로 이름도 내용도 같다). cache_dir이 None이면 `step_cache_dir(raw_dir)`.
    파일 메타데이터에 형식 번호, 조건, pipeline_hash 전체, 코드 지문(`code_fingerprint()`), 목록과 추출 결과 파일의
    지문(파일마다 크기, 수정 시각, 파일 전체의 sha256. `raw_content_digest`), 넣은 전역 기준선의 해시 전체를 적어 두고,
    하나라도 다르면 그 조건을 다시 만든다.
    - rebuild: True면 캐시를 읽지 않고 요청한 조건을 모두 다시 만들어 덮어쓴다(`python -m drowsy.dataset --rebuild`).
      추출 결과 파일의 sha256도 기억해 둔 값을 쓰지 않고 다시 계산한다.
    품질 판정(`gate_pass`)은 캐시에 넣지 않고 읽을 때마다 gate로 다시 계산한다.
    가정: 캐시의 열쇠는 pipeline_hash와 조건이다(지시 사항). 여기에 데이터가 바뀐 것을 알아채기 위한 지문과,
    특징을 만드는 코드가 바뀐 것을 알아채기 위한 코드 지문을 더했다(상수와 추출 결과가 그대로여도 코드를 고치면
    옛 특징으로 학습하게 되는 것을 막는다). 추출 결과는 프로세스마다 한 번 파일 전체를 읽어 sha256을 구한다. 같은
    프로세스가 도는 동안 크기와 수정 시각을 그대로 둔 채 내용을 바꾼 파일만 알아채지 못하고, 그럴 때는 `rebuild`를 쓴다.
    """
    return _load_tables(manifest, raw_dir, spec, arms, cache_dir, global_override, gate, workers, rebuild, None)


def load_frame_drop_tables(
    manifest: pd.DataFrame,
    raw_dir: Path,
    spec: PipelineSpec,
    arms: Sequence[str] = ARMS,
    *,
    variant: int,
    seed: int,
    drop_min: float,
    drop_max: float,
    cache_dir: Path | None = None,
    gate: QualityGate | None = None,
    workers: int = 1,
    rebuild: bool = False,
) -> dict[str, StepTable]:
    """프레임 누락 변형의 조건별 스텝 표(학습 전용 증강, PLAN 5.2절 증강 (2)). 캐시에서 읽고 없으면 만들어 쓴다.

    세션마다 `frame_drop_plan(session_id, 프레임 수, FrameDropSpec(variant, seed, drop_min, drop_max))`로 누락
    비율을 `[drop_min, drop_max]`에서 하나 뽑고, 저장된 프레임을 첫 프레임만 빼고 그 비율로 서로 독립으로 버린 뒤,
    남은 프레임을 원본과 같은 경로(`replay_video` → `DriverPipeline`)로 재생한다. 조건별 기준선 규칙은
    `build_step_table`과 같다(자기 기준선, 피험자의 alert 영상 기준선 또는 그 대체, 전역 상수).
    돌려주는 `StepTable`의 `frame_drop`에 설정이 들어 있고, 품질 표에 `drop_rate`, `n_source_frames`, `n_thinned`
    열이 더 있다(`StepTable` 설명 참고).

    가정(근사): 실제 기기에서 프레임이 빠지면 MediaPipe의 추적 결과(랜드마크와 블렌드셰이프 값)도 조금 달라지지만,
    여기서는 남은 프레임의 저장된 값을 그대로 쓴다. 누락도 프레임마다 독립이라 기기의 누락 무늬와 다르다.
    기기 동작을 그대로 재현하는 것은 증강 (1)(흐름 제한기를 흉내 내 다시 추출한 것)이고, 이것은 그 보조다.

    가정: 기준선과 T_b는 솎지 않은 원본 프레임에서 구한다. 조건마다 기준선을 대는 세션(자기 세션 또는 alert 영상)을
    솎지 않고 앱처럼 보정해 얻은 기준선을, 솎은 프레임의 재생에 넣고 그 세션 자신의 (솎지 않은) T_b부터 특징을 누적한다.
    그래서 원본 표와 다른 것은 특징에 들어가는 프레임뿐이고, 기준선 열과 누적 시작 시각은 원본 표와 같다.
    가정: 스텝 번호는 원본과 같은 격자지만 스텝의 집합은 조금 다를 수 있다. 마지막 프레임들이 버려지면 끝 스텝이
    하나 덜 나오고, 원래 10초에 가깝던 끊김이 10초를 넘으면 누적이 초기화된다. 원본 표와 행을 맞추지 않는다.
    가정: 전역 조건은 spec의 전역 상수를 쓴다(`global_override`는 받지 않는다. fold별 상수 점검은 원본 표만 쓴다).
    가정: 누락 모사 추출(15 fps)의 표를 다시 솎는 것도 막지 않는다. raw_dir이 가리키는 추출 결과를 그대로 솎는다.
    가정: 변형의 품질 표에서 프레임 수와 얼굴 시간은 솎은 재생의 값이다(그 표가 실제로 재생한 것을 적는다).
    스텝이 없는 세션(추출 결과 없음, 보정 미완)은 솎지 않고 `drop_rate`를 NaN으로 둔다.

    캐시 파일은 `{cache_dir}/{조건}_{pipeline_hash 앞 16자}_fd{variant}_s{seed}_r{drop_min}-{drop_max}.steps.parquet`과
    `.quality.parquet`이다. 열쇠는 `load_step_tables`의 것에 variant, seed, 비율 범위, 표집 절차 이름
    (`FRAME_DROP_METHOD`)을 더한 것이다. 하나라도 다르면 다시 만든다.
    시험 fold의 피험자가 섞이지 않게 고르는 것은 부르는 쪽의 몫이다(표에는 목록의 모든 세션이 들어 있다).
    """
    frame_drop = FrameDropSpec(
        variant=operator.index(variant), seed=operator.index(seed), drop_min=float(drop_min), drop_max=float(drop_max)
    )
    return _load_tables(manifest, raw_dir, spec, arms, cache_dir, None, gate, workers, rebuild, frame_drop)


def _load_tables(
    manifest: pd.DataFrame,
    raw_dir: Path,
    spec: PipelineSpec,
    arms: Sequence[str],
    cache_dir: Path | None,
    global_override: Baseline | None,
    gate: QualityGate | None,
    workers: int,
    rebuild: bool,
    frame_drop: FrameDropSpec | None,
) -> dict[str, StepTable]:
    """`load_step_tables`와 `load_frame_drop_tables`의 공통 부분: 캐시 읽기, 없는 조건 만들기, 캐시 쓰기."""
    raw_dir = Path(raw_dir)
    cache_dir = step_cache_dir(raw_dir) if cache_dir is None else Path(cache_dir)
    hash_value = spec_pipeline_hash(spec)
    code = code_fingerprint()
    fingerprint = _source_fingerprint(manifest, raw_dir, refresh=rebuild)
    override_hash = None if global_override is None else _baseline_hash(global_override)
    drop_key = (
        None
        if frame_drop is None
        else {
            "method": FRAME_DROP_METHOD,
            "variant": frame_drop.variant,
            "seed": frame_drop.seed,
            "drop_min": frame_drop.drop_min,
            "drop_max": frame_drop.drop_max,
        }
    )

    def metadata_for(arm: str) -> dict[str, Any]:
        return {
            "format": CACHE_FORMAT,
            "arm": arm,
            "pipeline_hash": hash_value,
            "code": code,
            "source": fingerprint,
            "global_override": override_hash if arm == ARM_GLOBAL else None,
            "frame_drop": drop_key,
        }

    def paths_for(arm: str) -> tuple[Path, Path]:
        return _cache_paths(cache_dir, arm, hash_value, override_hash if arm == ARM_GLOBAL else None, frame_drop)

    tables: dict[str, StepTable] = {}
    to_build: list[str] = []
    for arm in arms:
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
        steps = quality = None
        if not rebuild:
            steps_path, quality_path = paths_for(arm)
            steps = _read_cached(steps_path, metadata_for(arm))
            quality = _read_cached(quality_path, metadata_for(arm))
        if steps is None or quality is None:
            if arm not in to_build:
                to_build.append(arm)
            continue
        tables[arm] = StepTable(
            arm=arm,
            pipeline_hash=hash_value,
            steps=steps,
            quality=apply_quality_gate(quality, gate),
            frame_drop=frame_drop,
        )
    if to_build:
        if frame_drop is None:
            built = _build_step_tables(manifest, raw_dir, spec, to_build, global_override, gate, workers)
        else:
            built = _build_step_tables(manifest, raw_dir, spec, to_build, None, gate, workers, frame_drop)
        for arm, table in built.items():
            steps_path, quality_path = paths_for(arm)
            stored_quality = table.quality.drop(columns=["video_face_ratio", "video_n_windows", "gate_pass"])
            _write_cached(table.steps, steps_path, metadata_for(arm))
            _write_cached(stored_quality, quality_path, metadata_for(arm))
            tables[arm] = table
    return {arm: tables[arm] for arm in arms}


# ---------------------------------------------------------------------------
# 회전 점검 기록 (추출이 parquet 메타데이터에 남긴 것)
# ---------------------------------------------------------------------------

ROTATION_AMBIGUOUS = "ambiguous"  # 자동 점검이 각도를 가르지 못했고, 사람이 정한 각도(수동 회전 보정표)가 없다
ROTATION_OVERRIDE_NOT_APPLIED = "override_not_applied"  # 보정표에는 있는데 파일은 다른 각도로 추출돼 있다


@dataclass(frozen=True, slots=True)
class RotationFlag:
    """사람이 영상을 직접 보고 확인해야 하는 추출 결과 하나 (`read_rotation_checks`).

    - video: 영상의 상대 경로(수동 회전 보정표에 적는 키와 같은 꼴).
    - kind: `ambiguous`(자동 점검이 각도를 가르지 못했고 수동 회전 보정이 없다) 또는 `override_not_applied`(보정표에
      각도가 있는데 파일은 다른 각도로 추출돼 있다. 보정표를 고친 뒤 추출을 다시 돌리지 않은 경우다).
    - rotation: 추출에 적용된 회전(반시계, 도). 파일에 적혀 있지 않으면 None.
    - metadata_rotation: 영상의 회전 메타데이터(점검 기록의 값). 기록이 없으면 None.
    - reason: 점검 기록의 사유(`extract.REASON_*`. 예: `ambiguous_kept_metadata`). 기록이 없으면 빈 문자열.
    - override: 보정표의 각도. 보정표에 없으면 None.
    """

    video: str
    kind: str
    rotation: int | None
    metadata_rotation: int | None
    reason: str
    override: int | None


@dataclass(frozen=True, slots=True)
class RotationSummary:
    """추출 결과 파일들의 회전 점검 기록을 모은 것 (`read_rotation_checks`).

    - n_files: 있는 추출 결과 파일 수. `n_decided + n_manual + n_no_record + len(flagged)`와 같다.
    - n_missing: 없는 파일 수(추출 전이거나 추출에 실패한 영상). n_files에 들어가지 않는다.
    - n_decided: 자동 점검이 각도를 정한 파일(점검 기록이 있고 `ambiguous`가 거짓).
    - n_manual: 사람이 정한 각도로 추출된 파일. 수동 회전 보정표의 각도와 같은 각도로 추출된 것(점검 기록이
      `ambiguous`였어도 여기에 센다)과, 점검 기록 없이 수동 값으로 추출된 것(보정표나 `--video`의 `--rotation`)이다.
    - n_no_record: 점검 기록도 수동 값도 없는 파일(합성 데이터, 영상의 회전 메타데이터만 믿고 추출한 예전 결과,
      메타데이터를 읽을 수 없는 파일).
    - flagged: 확인이 필요한 영상(`RotationFlag`), 받은 순서.
    """

    n_files: int
    n_missing: int
    n_decided: int
    n_manual: int
    n_no_record: int
    flagged: tuple[RotationFlag, ...]

    @property
    def n_ambiguous(self) -> int:
        """자동 점검이 각도를 가르지 못했고 수동 회전 보정이 없는 파일 수."""
        return sum(1 for flag in self.flagged if flag.kind == ROTATION_AMBIGUOUS)

    @property
    def n_override_not_applied(self) -> int:
        """보정표에 각도가 있는데 다른 각도로 추출돼 있는 파일 수."""
        return sum(1 for flag in self.flagged if flag.kind == ROTATION_OVERRIDE_NOT_APPLIED)

    def as_dict(self) -> dict[str, Any]:
        """JSON 리포트에 그대로 넣을 수 있는 dict(값은 모두 파이썬 기본 자료형)."""
        return {
            "n_files": self.n_files,
            "n_missing": self.n_missing,
            "decided_by_check": self.n_decided,
            "manual_override": self.n_manual,
            "ambiguous_without_override": self.n_ambiguous,
            "override_not_applied": self.n_override_not_applied,
            "no_record": self.n_no_record,
            "flagged": [dataclasses.asdict(flag) for flag in self.flagged],
        }


def _without_suffix(name: str) -> str:
    """상대 경로에서 확장자 하나를 뗀 것(슬래시 구분). 영상 경로와 그 추출 결과 경로가 같은 값이 된다."""
    return Path(name).with_suffix("").as_posix()


def _angle(value: Any) -> int | None:
    return int(value) % 360 if isinstance(value, int) and not isinstance(value, bool) else None


def read_rotation_checks(
    files: Iterable[tuple[str, Path]], overrides: Mapping[str, int] | None = None
) -> RotationSummary:
    """추출 결과 파일마다 회전 점검 기록을 읽어, 회전이 틀렸을 수 있는데 사람이 확인하지 않은 영상을 모은다.

    추출(`python -m drowsy.extract`)은 회전을 자동으로 점검한 영상의 parquet 메타데이터(`drowsy_extract`)에
    `rotation_check`(`extract.ROTATION_CHECK_KEY`) 기록을 남긴다. 여기서 읽는 것은 그 기록의 `ambiguous`(점검이 각도를
    가르지 못했는지), `reason`, `metadata_rotation`과, 메타데이터의 `rotation`(추출에 적용된 회전), `rotation_override`
    (영상의 회전 메타데이터 대신 쓴 각도. 점검 기록이 없는데 이 값이 있으면 사람이 준 수동 값이다)이다.
    자동 점검은 얼굴이 여러 각도에서 비슷하게 검출되는 영상(거꾸로 선 얼굴도 검출되는 영상이 대표적이다)에서 맞는
    각도를 가르지 못한다. 그런 영상은 추출이 끝나도 표시만 남으므로, 여러 추출 결과를 읽는 명령(`python -m
    drowsy.dataset`, `signal_check e0`, `evaluate unlabeled`, `python -m drowsy.train`의 cv와 final,
    `python -m drowsy.baseline_lgbm`의 cv와 probes)이 모두 이 함수로 표시를 다시 읽어 알린다. 추출 결과 파일 하나를 재생하는
    `python -m drowsy.demo --raw`는 이 함수를 쓰지 않고 그 파일의 메타데이터를 직접 읽어(`demo.read_raw_rotation`) 같은
    표시를 알린다.

    - files: (이름, 추출 결과 parquet 경로)들. 이름은 영상의 상대 경로(`manifest["rel_path"]`)나 추출 결과의 상대 경로
      (`scan_unlabeled(raw_dir)["rel_path"]`)다. 확장자를 떼고 보정표의 키와 맞춘다.
    - overrides: 그 추출에 쓴 수동 회전 보정표(영상 상대 경로 → 각도. 설정의 `extraction.rotation_overrides` 또는
      `extraction.unlabeled_rotation_overrides`). None이면 빈 표다.

    파일 하나의 분류(위에서부터 처음 맞는 것)
    1. 파일이 없다: `n_missing`(다른 수에는 들어가지 않는다).
    2. 메타데이터를 읽을 수 없거나 합성 표시(`synthetic: true`)가 있다: 점검 기록 없음(`n_no_record`).
    3. 보정표에 그 영상이 있고 파일에 적용된 회전이 보정표의 각도와 같다: 수동 보정(`n_manual`).
    4. 보정표에 그 영상이 있는데 적용된 회전이 다르거나 적혀 있지 않다: 확인 필요(`override_not_applied`).
    5. 점검 기록이 있고 `ambiguous`가 참이다: 확인 필요(`ambiguous`).
    6. 점검 기록이 있다: 자동 점검이 정함(`n_decided`).
    7. 점검 기록이 없고 `rotation_override`에 각도가 있다: 수동 값으로 추출됐다(`n_manual`. 추출할 때 보정표에 있었거나
       `--video`에 `--rotation`을 준 영상).
    8. 그 밖: 점검 기록 없음(`n_no_record`).
    예외를 내지 않는다(읽을 수 없는 파일을 어떻게 다룰지는 부르는 명령이 따로 정한다).

    가정: "수동 보정이 있다"는 것은 보정표에 키가 있고 파일이 그 각도로 추출돼 있다는 뜻이다. 추출은 자동 점검이 고른
      각도와 같은 각도를 보정표에 적으면 파일을 다시 만들지 않으므로(점검 기록이 그대로 남는다) 파일만으로는 사람이
      확인했는지 알 수 없다. 그래서 보정표를 함께 받는다. 반대로 보정표의 각도와 파일의 각도가 다르면 사람이 정한
      각도가 아직 적용되지 않은 것이므로 확인 필요로 올린다.
    가정: 보정표의 키와 파일은 확장자를 뗀 상대 경로로 맞춘다(추출 결과 폴더에는 영상의 확장자가 남지 않는다. 확장자만
      다른 두 영상은 추출이 이미 거부한다). 다른 폴더의 추출 결과를 넘기면 키가 맞지 않아 보정이 없는 것으로 본다.
    가정: 읽을 수 없는 파일은 "기록 없음"에 센다(회전이 점검됐는지 알 수 없다는 뜻에서 같다).
    """
    table = {_without_suffix(key): (key, int(angle) % 360) for key, angle in (overrides or {}).items()}
    n_files = n_missing = n_decided = n_manual = n_no_record = 0
    flagged: list[RotationFlag] = []
    for name, raw_path in files:
        raw_path = Path(raw_path)
        if not raw_path.is_file():
            n_missing += 1
            continue
        n_files += 1
        try:
            metadata = read_extract_metadata(raw_path)
        except (ExtractError, OSError, ValueError, pa.ArrowException):
            n_no_record += 1
            continue
        if metadata.get("synthetic") is True:  # 영상이 없는 합성 세션이다. 보정표와 대조하지 않는다
            n_no_record += 1
            continue
        found = metadata.get(ROTATION_CHECK_KEY)
        record: Mapping[str, Any] = found if isinstance(found, dict) else {}
        applied = _angle(metadata.get("rotation"))
        metadata_rotation = _angle(record.get("metadata_rotation"))
        reason = str(record.get("reason") or "")
        entry = table.get(_without_suffix(name))
        if entry is not None:
            key, angle = entry
            if applied == angle:
                n_manual += 1
            else:
                flagged.append(
                    RotationFlag(key, ROTATION_OVERRIDE_NOT_APPLIED, applied, metadata_rotation, reason, angle)
                )
        elif record.get("ambiguous"):
            video = name
            if Path(name).suffix.lower() not in VIDEO_SUFFIXES:  # 추출 결과의 경로를 받았다: 영상의 확장자를 되살린다
                suffix = Path(str(metadata.get("source") or "")).suffix
                video = _without_suffix(name) + (suffix if suffix.lower() in VIDEO_SUFFIXES else "")
            flagged.append(RotationFlag(video, ROTATION_AMBIGUOUS, applied, metadata_rotation, reason, None))
        elif record:
            n_decided += 1
        elif _angle(metadata.get("rotation_override")) is not None:
            n_manual += 1
        else:
            n_no_record += 1
    return RotationSummary(n_files, n_missing, n_decided, n_manual, n_no_record, tuple(flagged))


def rotation_check_line(summary: RotationSummary, listed_in: str) -> str:
    """명령이 출력하는 한 줄: 회전 점검 기록의 구분별 파일 수. listed_in은 확인이 필요한 영상의 목록이 있는 곳이다.

    점검 기록도 수동 값도 없는 파일(합성 데이터, 영상의 회전 메타데이터만 믿고 추출한 예전 결과, 읽을 수 없는 파일)은
    "without a rotation-check record"로 센다. 그 뜻은 리포트의 절(`rotation_check_section`)이 풀어 적는다.
    """
    line = (
        f"rotation check of {summary.n_files} raw file(s): {summary.n_decided} decided by the automatic check, "
        f"{summary.n_manual} extracted with a manual override, "
        f"{summary.n_ambiguous} ambiguous without a manual override, "
        f"{summary.n_override_not_applied} with an override that was not applied, "
        f"{summary.n_no_record} without a rotation-check record"
    )
    if summary.flagged:
        line += f". LOOK AT THE {len(summary.flagged)} FLAGGED VIDEO(S) before using this data: {listed_in}"
    return line


def rotation_flag_lines(summary: RotationSummary, overrides_key: str) -> list[str]:
    """확인이 필요한 영상마다 한 줄(리포트를 쓰지 않는 명령이 출력한다). overrides_key는 보정표의 설정 키다."""
    lines: list[str] = []
    for flag in summary.flagged:
        applied = "an unrecorded angle" if flag.rotation is None else f"{flag.rotation} deg"
        if flag.kind == ROTATION_AMBIGUOUS:
            lines.append(
                f"  rotation ambiguous: {flag.video}: extracted at {applied} (video metadata "
                f"{'-' if flag.metadata_rotation is None else flag.metadata_rotation}, {flag.reason or 'no reason'}); "
                f"look at the video, add its angle to {overrides_key} and run the extraction again"
            )
        else:
            lines.append(
                f"  rotation override not applied: {flag.video}: {overrides_key} says {flag.override} deg but the "
                f"raw file was extracted at {applied}; run the extraction again"
            )
    return lines


def rotation_check_section(summary: RotationSummary, overrides_key: str) -> list[str]:
    """리포트(summary.md)에 싣는 회전 점검 절의 본문(제목은 부르는 쪽이 단다). 마크다운 줄 목록이다.

    구분별 파일 수의 표, 확인이 필요한 영상의 목록(전부), 점검 기록이 없는 파일에 대한 설명을 적는다.
    overrides_key는 그 추출의 수동 회전 보정표의 설정 키다(`extraction.rotation_overrides` 등).
    """
    lines = [
        "추출이 영상마다 남긴 회전 자동 점검 기록(추출 결과 parquet의 메타데이터 `rotation_check`)과 설정의 수동 회전 "
        f"보정표(`{overrides_key}`)를 읽은 것이다. 추출 결과 파일 {summary.n_files}개"
        + (f"(없는 파일 {summary.n_missing}개는 세지 않았다)" if summary.n_missing else "")
        + ":",
        "",
        "| 구분 | 파일 수 |",
        "| --- | --- |",
        f"| 자동 점검이 각도를 정함 | {summary.n_decided} |",
        f"| 사람이 정한 각도(수동 회전 보정)로 추출됨 | {summary.n_manual} |",
        f"| 자동 점검이 각도를 가르지 못했고 수동 보정이 없음(확인 필요) | {summary.n_ambiguous} |",
        f"| 보정표에 각도가 있는데 다른 각도로 추출돼 있음(확인 필요) | {summary.n_override_not_applied} |",
        f"| 점검 기록 없음 | {summary.n_no_record} |",
        "",
    ]
    if summary.flagged:
        lines += [
            f"**확인이 필요한 영상 {len(summary.flagged)}개.** 자동 점검은 얼굴이 여러 각도에서 비슷하게 검출되는 영상"
            "(거꾸로 선 얼굴도 검출되는 영상이 대표적이다)에서 맞는 각도를 가르지 못한다. 아래 영상은 회전이 틀린 채 "
            "추출됐을 수 있고, 이 리포트의 수치에는 그 영상들의 값이 그대로 들어 있다. 영상을 직접 보고 맞는 각도를 "
            f"`{overrides_key}`에 적은 뒤 추출을 다시 실행한다(점검이 고른 각도가 맞았다면 그 각도를 적는다. 그러면 이 "
            "목록에서 빠진다).",
            "",
            "| 영상 | 구분 | 추출에 적용된 회전 | 영상의 회전 메타데이터 | 점검 사유 | 보정표의 각도 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        kinds = {
            ROTATION_AMBIGUOUS: "가르지 못함, 수동 보정 없음",
            ROTATION_OVERRIDE_NOT_APPLIED: "보정표의 각도가 적용되지 않음(추출을 다시 실행)",
        }
        for flag in summary.flagged:
            cells = [
                f"`{flag.video}`",
                kinds[flag.kind],
                "-" if flag.rotation is None else str(flag.rotation),
                "-" if flag.metadata_rotation is None else str(flag.metadata_rotation),
                f"`{flag.reason}`" if flag.reason else "-",
                "-" if flag.override is None else str(flag.override),
            ]
            lines.append("| " + " | ".join(cells) + " |")
    else:
        lines.append("확인이 필요한 영상이 없다.")
    if summary.n_no_record > 0:
        lines += [
            "",
            f"점검 기록이 없는 파일 {summary.n_no_record}개는 회전이 점검됐는지 이 기록으로 알 수 없다. 합성 데이터, 자동 "
            "점검도 수동 값도 없이(영상의 회전 메타데이터만 믿고) 추출한 예전 결과, 메타데이터를 읽을 수 없는 파일이 "
            "여기에 든다.",
        ]
    return lines


# ---------------------------------------------------------------------------
# 설정에서 데이터 찾기, 명령
# ---------------------------------------------------------------------------

SYNTHETIC_NOTE = "SYNTHETIC data (drowsy.synth): pipeline check only, not a performance result"


@dataclass(frozen=True, slots=True, eq=False)
class DatasetSource:
    """설정이 가리키는 데이터.

    - manifest: 세션 목록(`build_manifest`).
    - raw_dir: 추출 결과(RawFrame parquet)의 루트. `build_step_table`과 `load_step_tables`에 그대로 넘긴다.
    - synthetic: 합성 데이터이면 True. 이 데이터로 만든 리포트는 첫 줄에 합성 데이터임을 적어야 한다.
    - description: 리포트 첫 줄에 쓸 수 있는 한 줄 설명.
    """

    manifest: pd.DataFrame
    raw_dir: Path
    synthetic: bool
    description: str


def _refuse_synthetic_raw(paths: Iterable[Path]) -> None:
    """실제 데이터로 쓰려는 추출 결과 가운데 합성 표시(`synthetic: true`)가 붙은 파일이 있으면 `ManifestError`.

    없는 파일은 건너뛴다(추출 전이거나 추출에 실패한 영상). 파일의 메타데이터(스키마)만 읽는다.
    """
    flagged: list[str] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            metadata = read_extract_metadata(path)
        except (ExtractError, OSError, pa.ArrowException) as exc:
            raise ManifestError(f"{path} is not a readable extraction parquet: {exc}") from exc
        if metadata.get("synthetic"):
            flagged.append(str(path))
    if flagged:
        raise ManifestError(
            f"paths.rldd_root is set (real data) but {len(flagged)} raw parquet file(s) are marked synthetic "
            f"(written by drowsy.synth): {_listed(flagged)}. Synthetic sessions are only used with "
            "paths.rldd_root: null and paths.synth_dir; their scores must never be reported as real-data results"
        )


def open_dataset(cfg: ExperimentConfig, fps_sim: bool = False) -> DatasetSource:
    """설정에서 학습·평가에 쓸 데이터를 찾는다.

    - `paths.rldd_root`가 있으면 실제 데이터다. 목록은 그 영상 폴더에서 만들고, 추출 결과는
      `raw_dir/native`(fps_sim이면 `raw_dir/{drop_sim.subdir}`)에서 읽는다.
    - null이면 합성 데이터(`paths.synth_dir`, `python -m drowsy.synth`가 쓴 것)를 쓴다.
      fps_sim이면 그 옆의 `{synth_dir 이름}_{drop_sim.subdir}` 폴더(`python -m drowsy.synth --fps-sim 15`가 쓴 것)다.
    그래서 `rldd_root`에 경로를 넣고 추출을 돌리면 같은 코드가 실제 데이터로 돈다.
    폴더가 없으면 만드는 명령을 적은 `ManifestError`다. 합성 폴더의 파일이 합성 표시(`synthetic: true`)를 달고
    있지 않아도 `ManifestError`다(실제 추출 결과를 합성으로 잘못 적는 일을 막는다).
    반대로 실제 데이터 쪽에서는, 목록의 세션 가운데 추출 결과가 있는 것을 모두 열어(파일 끝의 메타데이터만 읽는다)
    합성 표시가 붙은 것이 하나라도 있으면 `ManifestError`다. `rldd_root` 자체가 parquet 폴더일 때는 그 파일들도 본다.
    합성 세션이 실제 데이터(`synthetic = False`, "UTA-RLDD")로 표시되어 합성 점수가 성능처럼 보고되는 일을 막는다.
    가정: parquet 폴더를 `rldd_root`로 주는 것 자체는 막지 않는다(실제 추출 결과만 복사해 둔 배치일 수 있다).
    가정: 있는데 추출 결과로 읽을 수 없는 파일(메타데이터 없음, 깨진 파일)도 여기서 `ManifestError`로 알린다.
    """
    if cfg.paths.rldd_root is not None:
        subdir = cfg.extraction.drop_sim.subdir if fps_sim else cfg.extraction.native_subdir
        manifest = build_manifest(cfg.paths.rldd_root)
        raw_dir = cfg.paths.raw_dir / subdir
        candidates = _raw_paths(manifest, raw_dir)
        candidates += [Path(path) for path in manifest["path"].tolist() if Path(path).suffix.lower() == RAW_SUFFIX]
        _refuse_synthetic_raw(candidates)
        mode = f"device frame-drop simulation '{subdir}'" if fps_sim else "native frame rate"
        description = (
            f"UTA-RLDD at {cfg.paths.rldd_root}: {manifest['subject'].nunique()} subjects, "
            f"{manifest['video_id'].nunique()} logical videos, {mode}"
        )
        return DatasetSource(manifest, raw_dir, False, description)

    synth_dir = cfg.paths.synth_dir
    command = f"python -m drowsy.synth --config config/default.yaml --out {synth_dir} --subjects 60"
    if fps_sim:
        subdir = cfg.extraction.drop_sim.subdir
        synth_dir = synth_dir.with_name(f"{synth_dir.name}_{subdir}")
        fps = math.floor(1000.0 / cfg.extraction.drop_sim.latency_ms_mean + 0.5)
        command = f"python -m drowsy.synth --config config/default.yaml --out {synth_dir} --subjects 60 --fps-sim {fps}"
    if not synth_dir.is_dir():
        raise ManifestError(
            f"paths.rldd_root is null and the synthetic dataset {synth_dir} does not exist. Create it with `{command}`"
        )
    manifest = build_manifest(synth_dir)
    try:
        marked = read_extract_metadata(Path(manifest["path"].iloc[0])).get("synthetic") is True
    except (ExtractError, OSError, pa.ArrowException) as exc:
        raise ManifestError(f"{synth_dir} does not hold raw parquet files written by drowsy.synth: {exc}") from exc
    if not marked:
        raise ManifestError(f"{synth_dir} holds raw parquet files that are not marked synthetic")
    description = (
        f"{SYNTHETIC_NOTE}; {manifest['subject'].nunique()} subjects, "
        f"{manifest['video_id'].nunique()} logical videos at {synth_dir}"
        + (", device frame drops simulated" if fps_sim else "")
    )
    return DatasetSource(manifest, synth_dir, True, description)


def dataset_rotation_checks(
    cfg: ExperimentConfig, source: DatasetSource, manifest: pd.DataFrame | None = None
) -> RotationSummary:
    """`open_dataset`이 찾은 데이터의 추출 결과에서 회전 점검 기록을 읽는다 (`read_rotation_checks`).

    보정표는 데이터셋 추출(`python -m drowsy.extract --dataset`)이 쓰는 `extraction.rotation_overrides`다.
    manifest를 주면 `source.manifest` 대신 그 표의 `rel_path` 열을 쓴다(E0은 라벨 열을 버린 표를 넘긴다).
    """
    table = source.manifest if manifest is None else manifest
    files = zip(table["rel_path"].tolist(), _raw_paths(table, source.raw_dir))
    return read_rotation_checks(files, cfg.extraction.rotation_overrides)


def windows_per_class(steps: pd.DataFrame, window_steps: int, stride: int) -> dict[str, int]:
    """클래스 이름 → 창 수(창을 만들지 않고 센다). 리포트의 조건별·클래스별 창 수에 쓴다 (PLAN 4.4절).

    `make_windows(steps, window_steps, stride)`가 만드는 창의 수와 같다: 이어진 full_lookback 구간(길이 L)마다
    `(L − window_steps) // stride + 1`개. 세 정규화 조건은 같은 스텝을 내므로 조건이 달라도 값이 같다.
    """
    counts = dict.fromkeys(CLASS_ORDER, 0)
    for _, group in steps.groupby("session_id", sort=False):
        group = group.sort_values("k")
        k = group["k"].to_numpy(dtype=np.int64)
        runs = _full_runs(k, group["full_lookback"].to_numpy(dtype=bool), window_steps)
        n = sum((stop - start - window_steps) // stride + 1 for start, stop in runs)
        counts[CLASS_ORDER[int(group["class_index"].iloc[0])]] += n
    return counts


BASELINE_RECORD_COLUMNS: tuple[str, ...] = (
    "video_id",
    "session_id",
    "subject",
    "fold",
    "class_index",
    "t_b_ms",
    "baseline_status",
)


def baseline_window_summary(
    tables: Mapping[str, StepTable], window_steps: int, strides: Sequence[int], normal_t_b_ms: int
) -> dict[str, Any]:
    """리포트에 싣는 기준선·창 요약 (PLAN 4.4절 마지막 항목). 학습 리포트와 LightGBM 리포트가 같이 쓴다.

    - tables: 정규화 조건 → `StepTable`(같은 데이터의 것). window_steps: 창 길이(120).
    - strides: 창 수를 셀 보폭들(보통 `windows.train_stride`, `windows.eval_stride`). 같은 값은 한 번만 센다.
    - normal_t_b_ms: 연장 없이 끝난 보정의 T_b(`spec.baseline.duration_ms`, 60000).

    돌려주는 dict(값은 모두 파이썬 기본 자료형이라 `json.dumps`에 그대로 넣을 수 있다):
    - `normal_t_b_ms`, `n_sessions`, `n_videos`
    - `extended_baselines`: 자기 보정이 연장된 세션(`t_b_ms > normal_t_b_ms`. 전역 상수로 대체된 것도 들어간다).
      목록 순서이고 항목은 `BASELINE_RECORD_COLUMNS`(video_id, session_id, subject, fold, class_index, t_b_ms,
      baseline_status)다.
    - `fallback_baselines`: 자기 보정이 전역 상수로 대체된 세션(`baseline_status == fallback_global`). 같은 항목.
    - `unfinished_baselines`: 추출 결과는 있는데 보정이 끝나기 전에 끝난 세션(스텝이 없다, `t_b_ms`는 −1). 같은 항목.
    - `missing_raw`: 추출 결과가 없는 세션의 `session_id` 목록.
    - `alert_fallback_subjects`, `alert_fallback_sessions`: alert 영상 기준선 조건에서 alert 기준선을 쓸 수 없어 전역
      상수로 대체한 피험자 번호와 그 세션들. tables에 `alert_baseline` 조건이 없으면 둘 다 None이다.
    - `window_counts`: 조건(tables 순서) × 보폭마다 한 행. `{"arm", "stride", "alert", "low_vigilant", "drowsy",
      "total"}`. 창 수는 `windows_per_class`로 센다(학습의 시작 위치 무작위 이동과 무관한 명목 창 수다).

    기준선에 관한 값은 세션 자신의 보정 결과라 조건과 무관하다. 첫 표에서 읽고, 표들이 서로 다른 데이터의 것이면
    (세션 목록이나 T_b가 다르면) ValueError다.
    가정: 계획서는 "기준선이 연장된 영상과 조건별·클래스별 창 수를 리포트에 싣는다"만 적었다. 항목은 세션(파일)
    단위로 싣는다. 나뉜 영상은 부분마다 T_b가 따로이기 때문이다(`video_id`로 묶을 수 있다).
    """
    if len(tables) == 0:
        raise ValueError("baseline_window_summary needs at least one step table")
    if window_steps < 1 or any(stride < 1 for stride in strides):
        raise ValueError("window_steps and strides must be >= 1")
    first = next(iter(tables.values())).quality
    for arm, table in tables.items():
        same = table.quality["session_id"].tolist() == first["session_id"].tolist() and (
            table.quality["t_b_ms"].tolist() == first["t_b_ms"].tolist()
            and table.quality["baseline_status"].tolist() == first["baseline_status"].tolist()
        )
        if not same:
            raise ValueError(f"step table {arm!r} describes other sessions or baselines than the first table")

    def records(mask: pd.Series) -> list[dict[str, Any]]:
        rows = first.loc[mask, list(BASELINE_RECORD_COLUMNS)]
        return [
            {
                name: (str(value) if name in ("video_id", "session_id", "baseline_status") else int(value))
                for name, value in zip(BASELINE_RECORD_COLUMNS, row)
            }
            for row in rows.itertuples(index=False, name=None)
        ]

    alert_subjects: list[int] | None = None
    alert_sessions: list[str] | None = None
    if ARM_ALERT_BASELINE in tables:
        flagged = tables[ARM_ALERT_BASELINE].quality
        flagged = flagged[flagged["alert_fallback"].to_numpy(dtype=bool)]
        alert_subjects = sorted({int(subject) for subject in flagged["subject"].tolist()})
        alert_sessions = [str(session) for session in flagged["session_id"].tolist()]

    window_counts: list[dict[str, Any]] = []
    for arm, table in tables.items():
        for stride in dict.fromkeys(int(stride) for stride in strides):
            counts = windows_per_class(table.steps, window_steps, stride)
            window_counts.append({"arm": arm, "stride": stride, **counts, "total": sum(counts.values())})

    return {
        "normal_t_b_ms": int(normal_t_b_ms),
        "n_sessions": int(len(first)),
        "n_videos": int(first["video_id"].nunique()),
        "extended_baselines": records(first["t_b_ms"] > normal_t_b_ms),
        "fallback_baselines": records(first["baseline_status"] == BASELINE_FALLBACK_GLOBAL),
        "unfinished_baselines": records((first["raw_status"] == RAW_OK) & (first["baseline_status"] == BASELINE_NONE)),
        "missing_raw": [str(session) for session in first.loc[first["raw_status"] == RAW_MISSING, "session_id"]],
        "alert_fallback_subjects": alert_subjects,
        "alert_fallback_sessions": alert_sessions,
        "window_counts": window_counts,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m drowsy.dataset`: 설정이 가리키는 데이터의 스텝 표를 만들어 캐시에 두고 요약을 출력한다.

    `--rebuild`는 유효한 캐시도 무시하고 다시 만든다. `--frame-drop-variants`는 프레임 누락 변형(증강 (2))의 표도 만든다.
    목록 다음 줄에 추출 결과의 회전 점검 기록을 구분별 파일 수로 출력하고(`rotation_check_line`), 자동 점검이 각도를
    가르지 못했는데 수동 회전 보정이 없는 영상이 있으면 그 아래에 영상마다 한 줄씩 모두 적는다(`rotation_flag_lines`).
    이 명령은 리포트 파일을 쓰지 않으므로 목록 전체를 출력한다.
    마지막에 기준선 요약(`baseline_window_summary`: 보정이 연장·대체된 세션, alert 기준선 대체 피험자)을 출력한다.
    가정: 회전이 확인되지 않은 영상이 있어도 그 영상을 빼지 않고 스텝 표를 만들며 종료 코드는 0이다. 알리기만 한다
    (영상을 모두 직접 볼지, 더 엄격한 자동 점검을 넣을지는 소유자의 결정을 기다리는 항목이다).
    가정: `--frame-drop-seed`의 기본값은 0이다. 캐시의 열쇠에 시드가 들어가므로, 여기서 미리 만든 변형을 학습이
    쓰려면 학습 코드가 같은 시드와 변형 번호로 `load_frame_drop_tables`를 불러야 한다.
    종료 코드: 0 성공, 2 시작할 수 없음(설정, 데이터 폴더, 추출 결과 문제).
    """
    parser = argparse.ArgumentParser(
        prog="python -m drowsy.dataset",
        description="Build (and cache) the step tables of the configured dataset and print window counts.",
    )
    parser.add_argument("--config", default="config/default.yaml", help="experiment config YAML")
    parser.add_argument("--arms", nargs="+", default=None, choices=ARMS, help="default: training.arms")
    parser.add_argument("--fps-sim", action="store_true", help="use the frame-drop simulated extraction")
    parser.add_argument("--workers", type=int, default=None, help="parallel subjects (default: extraction.workers)")
    parser.add_argument(
        "--rebuild", action="store_true", help="ignore existing step-table caches and build the requested tables again"
    )
    parser.add_argument(
        "--frame-drop-variants",
        nargs="+",
        type=int,
        default=(),
        metavar="V",
        help="also build these frame-drop variants (training augmentation 2; rates from training.frame_drop_min/max)",
    )
    parser.add_argument("--frame-drop-seed", type=int, default=0, help="seed of the frame-drop variants (default 0)")
    args = parser.parse_args(argv)
    variants = tuple(dict.fromkeys(args.frame_drop_variants))
    thinned: dict[int, dict[str, StepTable]] = {}
    try:
        cfg = load_config(args.config)
        spec = load_feature_spec(cfg.paths.feature_spec).pipeline
        source = open_dataset(cfg, fps_sim=args.fps_sim)
        print(source.description)
        manifest = source.manifest
        print(
            f"manifest: {manifest['subject'].nunique()} subjects, {manifest['video_id'].nunique()} logical videos, "
            f"{len(manifest)} sessions, folds {sorted(int(f) for f in manifest['fold'].unique())}"
        )
        rotation = dataset_rotation_checks(cfg, source)
        print(rotation_check_line(rotation, "the lines below"))
        for line in rotation_flag_lines(rotation, OVERRIDES_KEY):
            print(line)
        sys.stdout.flush()
        arms = tuple(cfg.training.arms if args.arms is None else args.arms)
        workers = cfg.extraction.workers if args.workers is None else args.workers
        gate = cfg.evaluation.quality_gate
        started = time.perf_counter()
        tables = load_step_tables(
            manifest, source.raw_dir, spec, arms, gate=gate, workers=workers, rebuild=args.rebuild
        )
        elapsed = time.perf_counter() - started
        print(
            f"step tables for {list(arms)} ready in {elapsed:.1f} s (cache: {step_cache_dir(source.raw_dir)}"
            + (", rebuilt" if args.rebuild else "")
            + ")",
            flush=True,
        )
        for variant in variants:
            started = time.perf_counter()
            thinned[variant] = load_frame_drop_tables(
                manifest,
                source.raw_dir,
                spec,
                arms,
                variant=variant,
                seed=args.frame_drop_seed,
                drop_min=cfg.training.frame_drop_min,
                drop_max=cfg.training.frame_drop_max,
                gate=gate,
                workers=workers,
                rebuild=args.rebuild,
            )
            print(
                f"frame-drop variant {variant} (seed {args.frame_drop_seed}, rates "
                f"{cfg.training.frame_drop_min:g}..{cfg.training.frame_drop_max:g}) for {list(arms)} "
                f"ready in {time.perf_counter() - started:.1f} s",
                flush=True,
            )
    except (ValueError, OSError, ExtractError) as exc:  # SpecError와 ManifestError도 ValueError다
        print(f"error: {exc}", file=sys.stderr)
        return 2
    windows = cfg.windows
    for arm in arms:
        table = tables[arm]
        quality = table.quality
        videos = quality.drop_duplicates("video_id")
        statuses = quality["baseline_status"].value_counts().to_dict()
        print(
            f"[{arm}] steps {len(table.steps)}, "
            f"windows stride {windows.train_stride}: "
            f"{windows_per_class(table.steps, windows.window_steps, windows.train_stride)}, "
            f"stride {windows.eval_stride}: "
            f"{windows_per_class(table.steps, windows.window_steps, windows.eval_stride)}"
        )
        print(
            f"[{arm}] quality gate passed by {int(videos['gate_pass'].sum())} of {len(videos)} logical videos; "
            f"own baseline status {statuses}; alert-baseline fallback sessions {int(quality['alert_fallback'].sum())}; "
            f"missing raw files {int((quality['raw_status'] == RAW_MISSING).sum())}"
        )

    summary = baseline_window_summary(
        tables, windows.window_steps, (windows.train_stride, windows.eval_stride), spec.baseline.duration_ms
    )
    extended = [
        f"{record['session_id']} ({record['t_b_ms']} ms, {record['baseline_status']})"
        for record in summary["extended_baselines"]
    ]
    print(
        f"baselines: {len(extended)} of {summary['n_sessions']} sessions finished calibration later than "
        f"{summary['normal_t_b_ms']} ms"
        + (f" [{_listed(extended)}]" if extended else "")
        + f"; {len(summary['fallback_baselines'])} fell back to the global constants; "
        f"{len(summary['unfinished_baselines'])} ended before calibration finished"
    )
    if summary["alert_fallback_subjects"] is not None:
        print(f"alert-baseline fallback subjects: {summary['alert_fallback_subjects']}")

    for variant, variant_tables in thinned.items():
        for arm in arms:
            table = variant_tables[arm]
            rates = table.quality["drop_rate"].dropna().to_numpy(dtype=np.float64)
            spread = (
                f"min {rates.min():.3f}, median {float(np.median(rates)):.3f}, max {rates.max():.3f}"
                if len(rates)
                else "none"
            )
            print(
                f"[{arm}] frame-drop variant {variant}: steps {len(table.steps)}, drop rates of "
                f"{len(rates)} thinned sessions: {spread}; frames removed "
                f"{int(table.quality['n_thinned'].sum())} of {int(table.quality['n_source_frames'].sum())}; "
                f"windows stride {windows.train_stride}: "
                f"{windows_per_class(table.steps, windows.window_steps, windows.train_stride)}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
