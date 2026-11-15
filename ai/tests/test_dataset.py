"""`drowsy.dataset` 테스트: fold, 목록, 피험자 분리, 재생, 세 정규화 조건, 창, 중첩 분할, 캐시, 프레임 누락 변형
(PLAN 2.4, 4.4절, 5.2절, 6장, 1.2절의 결정 D7: 영상 단위 지표의 분모와 품질 기준).

영상 폴더는 빈 파일로 흉내 내고(목록은 파일을 열지 않는다), 추출 결과는 `drowsy.synth`의 각본 시나리오를
`extract.write_raw_frames`로 써서 만든다. 프레임은 10 fps다(세션을 길게 써도 빠르게 돌리기 위해서다).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import av
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import yaml

from drowsy import dataset, extract, synth
from drowsy.config import ARM_ALERT_BASELINE, ARM_GLOBAL, ARM_SESSION_BASELINE, ARMS, QualityGate, load_config
from drowsy.dataset import (
    BASELINE_RECORD_COLUMNS,
    FINGERPRINT_MODULES,
    FRAME_DROP_QUALITY_COLUMNS,
    ID_COLUMNS,
    MANIFEST_COLUMNS,
    ROTATION_AMBIGUOUS,
    ROTATION_OVERRIDE_NOT_APPLIED,
    STEP_COLUMNS,
    FrameDropSpec,
    ManifestError,
    RotationFlag,
    RotationSummary,
    StepTable,
    SubjectOverlapError,
    WindowSet,
    apply_quality_gate,
    assert_subject_disjoint,
    baseline_window_summary,
    build_manifest,
    build_step_table,
    code_fingerprint,
    fingerprint_files,
    frame_drop_plan,
    load_frame_drop_tables,
    load_step_tables,
    make_windows,
    nested_splits,
    official_fold,
    open_dataset,
    raw_content_digest,
    read_rotation_checks,
    replay_video,
    rotation_check_line,
    rotation_check_section,
    rotation_flag_lines,
    scan_unlabeled,
    spec_pipeline_hash,
    step_cache_dir,
    windows_per_class,
)
from drowsy.extract import read_raw_frames, write_raw_frames
from drowsy.spec import (
    FeatureSpec,
    PipelineSpec,
    RulesConfig,
    apply_overrides,
    compute_pipeline_hash,
    file_sha256,
    read_feature_spec_dict,
    write_feature_spec,
)
from drowsy.synth import Closure, FaceMissing, Linger, Scenario, Yawn, default_profile, render
from drowsy.types import (
    BASELINE_FALLBACK_GLOBAL,
    BASELINE_OK,
    CLASS_ORDER,
    EVENT_CALIBRATING,
    F_CLOSURE_MEAN_60,
    F_LONG_CLOSURE_COUNT_60,
    F_MAX_CLOSURE_MS_60,
    F_YAWN_COUNT_60,
    FEATURE_NAMES,
    IDENTITY_R0,
    N_FEATURES,
    Baseline,
    RawFrame,
)

FPS = 10.0
FEATURES = list(FEATURE_NAMES)

# ---------------------------------------------------------------------------
# 도우미
# ---------------------------------------------------------------------------


def touch_tree(root: Path, names: list[str]) -> Path:
    """빈 파일로 된 폴더 구조를 만든다."""
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    return root


def rldd_like_names(n_subjects: int = 60, extension: str = "mp4") -> list[str]:
    """RLDD 미러와 같은 배치: `Fold{k}_part{p}/{NN}/{라벨}.{확장자}`, 32번과 49번의 drowsy는 두 부분."""
    names: list[str] = []
    for subject in range(1, n_subjects + 1):
        fold = (subject - 1) // 12 + 1
        part = 1 if (subject - 1) % 12 < 6 else 2
        folder = f"Fold{fold}_part{part}/{subject:02d}"
        names += [f"{folder}/0.{extension}", f"{folder}/5.{extension}"]
        if subject in (32, 49):
            names += [f"{folder}/10_1.{extension}", f"{folder}/10_2.{extension}"]
        else:
            names.append(f"{folder}/10.{extension}")
    return names


def closures(hold_ms: int, starts_s: list[int]) -> tuple[Closure, ...]:
    return tuple(Closure(hold_ms=hold_ms, start_ms=1000 * start) for start in starts_s)


def write_session(root: Path, rel_path: str, scenario: Scenario, spec: PipelineSpec) -> None:
    write_raw_frames(render(scenario, spec), root / rel_path, spec, {"synthetic": True})


def step_values(table: StepTable, session_id: str) -> np.ndarray:
    rows = table.steps[table.steps["session_id"] == session_id]
    return rows[FEATURES].to_numpy()


def reference_plan(session_id: str, n_frames: int, drop: FrameDropSpec) -> tuple[float, list[bool]]:
    """`frame_drop_plan`의 설명만 보고 따로 적은 구현(코드를 부르지 않는다).

    sha256("hawkeye.frame_drop|seed|variant|session_id")의 앞 8바이트(빅 엔디언) → `random.Random`,
    첫 난수로 비율, 그 뒤로 두 번째 프레임부터 프레임마다 난수 하나(`u < 비율`이면 버림).
    """
    text = f"hawkeye.frame_drop|{drop.seed}|{drop.variant}|{session_id}"
    rng = random.Random(int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big"))
    rate = drop.drop_min + (drop.drop_max - drop.drop_min) * rng.random()
    return rate, [True] + [rng.random() >= rate for _ in range(n_frames - 1)]


def spy_on_builder(monkeypatch: pytest.MonkeyPatch) -> list[tuple[tuple[str, ...], FrameDropSpec | None]]:
    """`dataset._build_step_tables`가 불릴 때마다 (조건들, 프레임 누락 설정)을 적는 목록을 돌려준다."""
    builds: list[tuple[tuple[str, ...], FrameDropSpec | None]] = []
    real_build = dataset._build_step_tables

    def counting_build(manifest, raw_dir, spec, arms, global_override, gate, workers, frame_drop=None):
        builds.append((tuple(arms), frame_drop))
        return real_build(manifest, raw_dir, spec, arms, global_override, gate, workers, frame_drop)

    monkeypatch.setattr(dataset, "_build_step_tables", counting_build)
    return builds


def cache_metadata(path: Path) -> dict:
    return json.loads(pq.read_schema(path).metadata[dataset.CACHE_METADATA_KEY].decode("utf-8"))


def assert_windows_are_whole(windows: WindowSet, steps: pd.DataFrame, window_steps: int) -> None:
    """모든 창이 한 세션의, 번호가 이어진 full_lookback 스텝으로만 되어 있고 값이 스텝 표와 같은지 검사한다."""
    index = {(s, k): row for row, (s, k) in enumerate(zip(steps["session_id"].tolist(), steps["k"].tolist()))}
    full = steps["full_lookback"].to_numpy()
    values = steps[FEATURES].to_numpy()
    for i in range(len(windows)):
        session_id = windows.meta["session_id"].iloc[i]
        k_end = int(windows.meta["k_end"].iloc[i])
        rows = [index[(session_id, k)] for k in range(k_end - window_steps + 1, k_end + 1)]  # 없으면 KeyError
        assert full[rows].all()
        np.testing.assert_array_equal(windows.x[i], values[rows].astype(np.float32))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory, pipeline_spec: PipelineSpec) -> SimpleNamespace:
    """피험자 3명의 추출 결과 폴더와 세 조건의 스텝 표.

    - 1번: alert 영상의 보정 구간(앞 60초)에 긴 감김과 하품이 있다. 5 영상은 245~260초에 프레임이 끊긴다.
      alert 영상과 나머지 영상의 얼굴(뜬 눈 수준, 거치 각도)이 다르다.
    - 2번: drowsy 영상이 `10_1`(250초), `10_2`(245초)로 나뉘어 있다.
    - 3번: alert 영상의 앞 170초 동안 얼굴이 없다(보정이 전역 상수로 대체된다).
    """
    spec = pipeline_spec
    root = tmp_path_factory.mktemp("tiny") / "raw"
    base = default_profile(spec)
    face_a = dataclasses.replace(base, e_open_l=0.30, e_open_r=0.31, b_open_l=0.10, b_open_r=0.12, jaw_rest=0.05)
    face_b = dataclasses.replace(base, e_open_l=0.25, e_open_r=0.26, b_open_l=0.20, b_open_r=0.18, yaw=5.0, pitch=3.0)

    def scenario(duration_s: int, segments: tuple, profile=base, **kwargs) -> Scenario:
        return Scenario(segments=segments, duration_ms=1000 * duration_s, fps=FPS, profile=profile, **kwargs)

    alert_1 = (Closure(hold_ms=2_000, start_ms=20_000), Yawn(start_ms=30_000)) + closures(600, [70, 100, 150, 200])
    write_session(root, "Fold1_part1/01/0.parquet", scenario(250, alert_1, face_a), spec)
    write_session(
        root,
        "Fold1_part1/01/5.parquet",
        scenario(445, closures(800, list(range(70, 440, 30))), face_b, gaps=((245_000, 260_000),)),
        spec,
    )
    drowsy_1 = closures(1_500, list(range(70, 240, 20))) + (Linger(8_000, 0.6, start_ms=125_000),)
    write_session(root, "Fold1_part1/01/10.parquet", scenario(250, drowsy_1, face_b), spec)

    write_session(root, "Fold1_part1/02/0.parquet", scenario(250, closures(600, [80, 160])), spec)
    write_session(root, "Fold1_part1/02/5.parquet", scenario(250, closures(700, [75, 130, 190])), spec)
    write_session(root, "Fold1_part1/02/10_1.parquet", scenario(250, closures(1_200, [70, 110, 150, 210])), spec)
    write_session(root, "Fold1_part1/02/10_2.parquet", scenario(245, closures(1_200, [65, 95, 175])), spec)

    write_session(root, "Fold1_part1/03/0.parquet", scenario(200, (FaceMissing(170_000),), face_a), spec)
    write_session(root, "Fold1_part1/03/5.parquet", scenario(250, closures(700, [90, 180]), face_a), spec)
    write_session(root, "Fold1_part1/03/10.parquet", scenario(250, closures(1_500, [70, 140, 220]), face_a), spec)

    manifest = build_manifest(root)
    tables = {arm: build_step_table(manifest, root, spec, arm) for arm in ARMS}
    return SimpleNamespace(root=root, manifest=manifest, tables=tables, face_a=face_a, face_b=face_b)


@pytest.fixture
def small_tree(tmp_path: Path, pipeline_spec: PipelineSpec) -> SimpleNamespace:
    """피험자 2명, 세션마다 70초인 작은 추출 결과 폴더(스텝은 나오고 창은 나오지 않는다)."""
    root = tmp_path / "small" / "raw"
    for subject in (1, 2):
        for label in (0, 5, 10):
            segments = closures(600, [10 + subject, 62 + label // 5])
            scenario = Scenario(segments=segments, duration_ms=70_000, fps=FPS)
            write_session(root, f"{subject:02d}/{label}.parquet", scenario, pipeline_spec)
    return SimpleNamespace(root=root, manifest=build_manifest(root))


TINY_DROP = FrameDropSpec(variant=0, seed=0, drop_min=0.10, drop_max=0.30)


@pytest.fixture(scope="module")
def tiny_thinned(tiny: SimpleNamespace, pipeline_spec: PipelineSpec) -> dict[str, StepTable]:
    """`tiny`의 프레임 누락 변형(변형 0, 시드 0, 비율 10~30%) 세 조건."""
    return dataset._build_step_tables(tiny.manifest, tiny.root, pipeline_spec, ARMS, None, None, 1, TINY_DROP)


# ---------------------------------------------------------------------------
# fold와 피험자 분리
# ---------------------------------------------------------------------------


def test_official_fold_mapping() -> None:
    assert [official_fold(s) for s in (1, 12, 13, 24, 25, 36, 37, 48, 49, 60)] == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    for fold in range(1, 6):  # fold k는 피험자 12(k−1)+1 ~ 12k
        assert [s for s in range(1, 61) if official_fold(s) == fold] == list(range(12 * (fold - 1) + 1, 12 * fold + 1))
    assert official_fold(61) == 6
    with pytest.raises(ValueError):
        official_fold(0)


def test_assert_subject_disjoint() -> None:
    assert_subject_disjoint([1, 2, 3], {4, 5}, (6,), [])
    assert_subject_disjoint([1, 1, 2], [3])  # 한 집합 안의 중복은 겹침이 아니다
    assert_subject_disjoint(np.array([1, 2]), pd.Series([3, 4]))
    with pytest.raises(SubjectOverlapError, match="subject 3 in sets \\[0, 1\\]"):
        assert_subject_disjoint([1, 2, 3], [3, 4])
    with pytest.raises(AssertionError):  # SubjectOverlapError는 AssertionError다
        assert_subject_disjoint(range(1, 37), range(37, 49), [48, 49])
    with pytest.raises(SubjectOverlapError, match="2 subjects"):
        assert_subject_disjoint([1, 2], [5], [2, 1])


# ---------------------------------------------------------------------------
# 목록
# ---------------------------------------------------------------------------


def test_manifest_of_an_rldd_like_tree(tmp_path: Path) -> None:
    root = touch_tree(tmp_path / "rldd", rldd_like_names())
    manifest = build_manifest(root)

    assert list(manifest.columns) == list(MANIFEST_COLUMNS)
    assert len(manifest) == 182 and manifest["video_id"].nunique() == 180 and manifest["subject"].nunique() == 60
    assert manifest["session_id"].is_unique
    assert manifest["fold"].tolist() == [official_fold(s) for s in manifest["subject"]]
    assert manifest.groupby("fold")["subject"].nunique().tolist() == [12] * 5
    assert manifest.groupby("class_index")["video_id"].nunique().tolist() == [60, 60, 60]
    assert set(zip(manifest["label"], manifest["class_index"])) == {(0, 0), (5, 1), (10, 2)}
    assert manifest[["subject", "class_index", "part"]].apply(tuple, axis=1).is_monotonic_increasing

    first = manifest.iloc[0]
    assert (first["video_id"], first["session_id"], first["subject"], first["fold"]) == ("S01_0", "S01_0", 1, 1)
    assert (first["label"], first["class_index"], first["part"]) == (0, 0, 0)
    assert first["rel_path"] == "Fold1_part1/01/0.mp4" and Path(first["path"]) == root / "Fold1_part1/01/0.mp4"

    split = manifest[manifest["subject"] == 32]
    assert split["session_id"].tolist() == ["S32_0", "S32_5", "S32_10_1", "S32_10_2"]
    assert split["video_id"].tolist() == ["S32_0", "S32_5", "S32_10", "S32_10"]
    assert split["part"].tolist() == [0, 0, 1, 2] and split["fold"].tolist() == [3] * 4
    assert manifest[manifest["part"] > 0]["subject"].unique().tolist() == [32, 49]


def test_manifest_accepts_nesting_mixed_case_and_unpadded_folders(tmp_path: Path) -> None:
    names = [
        "Fold1_part1/Fold1_part1/01/0.mov",  # 미러처럼 한 번 더 겹친 폴더
        "Fold1_part1/Fold1_part1/01/5.mov",
        "Fold1_part1/Fold1_part1/01/10.MOV",  # 한 폴더 안에서 대소문자가 섞임
        "Fold1_part1/Fold1_part1/2/0.MP4",  # 0을 채우지 않은 폴더
        "Fold1_part1/Fold1_part1/2/5.m4v",
        "Fold1_part1/Fold1_part1/2/10.Mp4",
        "fold2_PART2/20/0.mp4",  # 폴더 이름의 대소문자
        "fold2_PART2/20/10.mp4",
        "fold2_PART2/20/5.MOV",
        "loose/copy/40/0.mp4",  # FoldK 폴더가 없으면 공식 대응만 쓴다
        "loose/copy/40/5.mp4",
        "loose/copy/40/10_1.mp4",
        "loose/copy/40/10_2.mp4",
        "loose/copy/40/10_3.mp4",
        "Fold1_part1/Fold1_part1/01/notes.txt",  # 영상이 아닌 파일
        "Fold1_part1/Fold1_part1/01/._0.mov",  # 숨김 파일
        "__MACOSX/Fold1_part1/01/0.mov",
        "Fold1_part1/Fold1_part1/01/0.parquet.tmp",
    ]
    manifest = build_manifest(touch_tree(tmp_path / "copy", names))
    assert manifest["subject"].tolist() == [1, 1, 1, 2, 2, 2, 20, 20, 20, 40, 40, 40, 40, 40]
    assert manifest["fold"].tolist() == [1] * 6 + [2] * 3 + [4] * 5
    assert manifest["label"].tolist() == [0, 5, 10] * 3 + [0, 5, 10, 10, 10]
    assert manifest["rel_path"].tolist()[2:5] == [
        "Fold1_part1/Fold1_part1/01/10.MOV",
        "Fold1_part1/Fold1_part1/2/0.MP4",
        "Fold1_part1/Fold1_part1/2/5.m4v",
    ]
    assert manifest["video_id"].tolist()[3] == "S02_0"
    assert manifest["session_id"].tolist()[-3:] == ["S40_10_1", "S40_10_2", "S40_10_3"]
    assert manifest["part"].tolist()[-3:] == [1, 2, 3]


def test_manifest_fails_loudly(tmp_path: Path) -> None:
    good = ["Fold1_part1/01/0.mp4", "Fold1_part1/01/5.mp4", "Fold1_part1/01/10.mp4"]

    def build(name: str, names: list[str]) -> pd.DataFrame:
        return build_manifest(touch_tree(tmp_path / name, names))

    assert len(build("good", good)) == 3
    with pytest.raises(ManifestError, match=r"subject 2 lacks label\(s\) \[5\]"):
        build("missing_class", good + ["Fold1_part1/02/0.mp4", "Fold1_part1/02/10.mp4"])
    with pytest.raises(ManifestError, match="subject 13 belongs to fold 2"):
        build("wrong_fold", good + ["Fold1_part2/13/0.mp4", "Fold1_part2/13/5.mp4", "Fold1_part2/13/10.mp4"])
    with pytest.raises(ManifestError, match="contradicts the official fold"):
        build("wrong_outer_fold", ["Fold3_part1/Fold1_part1/01/0.mp4", "Fold3_part1/Fold1_part1/01/5.mp4",
                                   "Fold3_part1/Fold1_part1/01/10.mp4"])  # fmt: skip
    with pytest.raises(ManifestError, match="more than one file"):
        build("duplicate", good + ["Fold1_part1/01/0.MOV"])
    with pytest.raises(ManifestError, match="more than one file"):
        build("two_copies", good + ["backup/1/5.mp4"])
    with pytest.raises(ManifestError, match="S01_10: parts \\[0, 1\\]"):
        build("split_and_whole", good + ["Fold1_part1/01/10_1.mp4"])
    with pytest.raises(ManifestError, match="without gaps"):
        build("missing_part", good[:2] + ["Fold1_part1/01/10_2.mp4"])
    with pytest.raises(ManifestError, match=r"S01_10: parts \[1, 3\]"):
        build("gap_in_parts", good[:2] + ["Fold1_part1/01/10_1.mp4", "Fold1_part1/01/10_3.mp4"])
    # `10_1`만 있고 `10_2`가 없는 사본(복사가 덜 된 것): 반쪽짜리 영상을 논리 영상 하나로 받지 않는다.
    with pytest.raises(ManifestError, match=r"at least two parts.*S01_10: parts \[1\]"):
        build("lone_first_part", good[:2] + ["Fold1_part1/01/10_1.mp4"])
    with pytest.raises(ManifestError, match=r"S01_0: parts \[1\]"):  # `_N`은 세 라벨 모두에서 같은 규칙이다
        build("lone_first_part_alert", ["Fold1_part1/01/0_1.mp4"] + good[1:])
    incomplete = [name for name in rldd_like_names() if name != "Fold3_part2/32/10_2.mp4"]
    assert len(incomplete) == 181
    with pytest.raises(ManifestError, match=r"S32_10: parts \[1\]") as lone:
        build("rldd_without_32_10_2", incomplete)
    assert "S49_10" not in str(lone.value)  # 두 부분이 다 있는 49번은 문제 삼지 않는다
    both_parts = build("two_parts", good[:2] + ["Fold1_part1/01/10_1.mp4", "Fold1_part1/01/10_2.mp4"])
    assert both_parts["session_id"].tolist() == ["S01_0", "S01_5", "S01_10_1", "S01_10_2"]
    with pytest.raises(ManifestError, match="cannot tell subject and label.*extra.mp4"):
        build("stray_video", good + ["Fold1_part1/01/extra.mp4"])
    with pytest.raises(ManifestError, match="cannot tell subject and label"):
        build("no_subject_folder", good + ["Fold1_part1/sub1/0.mp4"])
    with pytest.raises(ManifestError, match="cannot tell subject and label"):
        build("top_level_file", good + ["0.mp4"])
    with pytest.raises(ManifestError, match="cannot tell subject and label"):
        build("subject_zero", good + ["00/0.mp4", "00/5.mp4", "00/10.mp4"])
    with pytest.raises(ManifestError, match="both video files"):
        build("mixed", good + ["Fold1_part1/01/0.parquet"])
    with pytest.raises(ManifestError, match="no video files"):
        build("empty", ["readme.txt"])
    with pytest.raises(ManifestError, match="not a directory"):
        build_manifest(tmp_path / "does_not_exist")


def test_manifest_of_a_raw_parquet_tree_and_unlabeled_scan(tmp_path: Path) -> None:
    raw = touch_tree(tmp_path / "raw", rldd_like_names(24, "parquet"))
    manifest = build_manifest(raw)
    assert len(manifest) == 72 and manifest["fold"].unique().tolist() == [1, 2]
    assert manifest["rel_path"].iloc[0] == "Fold1_part1/01/0.parquet"

    cars = touch_tree(tmp_path / "cars", ["day1/drive_a.mp4", "day1/drive_b.MOV", "night/x/drive_c.m4v", "log.csv"])
    unlabeled = scan_unlabeled(cars)
    assert list(unlabeled.columns) == ["video_id", "rel_path", "path"]
    assert unlabeled["video_id"].tolist() == ["day1/drive_a", "day1/drive_b", "night/x/drive_c"]
    assert unlabeled["rel_path"].tolist() == ["day1/drive_a.mp4", "day1/drive_b.MOV", "night/x/drive_c.m4v"]
    assert Path(unlabeled["path"].iloc[2]) == cars / "night/x/drive_c.m4v"
    assert scan_unlabeled(touch_tree(tmp_path / "car_raw", ["a/b.parquet"]))["video_id"].tolist() == ["a/b"]
    with pytest.raises(ManifestError, match="differ only by extension"):
        scan_unlabeled(touch_tree(tmp_path / "clash", ["a.mp4", "a.mov"]))
    with pytest.raises(ManifestError, match="no video files"):
        scan_unlabeled(touch_tree(tmp_path / "nothing", ["a.txt"]))


# ---------------------------------------------------------------------------
# 재생
# ---------------------------------------------------------------------------


def test_replay_with_own_calibration(pipeline_spec: PipelineSpec, rules_config: RulesConfig) -> None:
    scenario = Scenario(segments=closures(1_500, [20, 70]), duration_ms=75_000, fps=FPS)
    frames = render(scenario, pipeline_spec)
    result = replay_video(frames, pipeline_spec, None, None)

    profile = default_profile(pipeline_spec)
    assert result.baseline is not None and result.baseline.status == BASELINE_OK
    assert result.baseline.t_complete_ms == 60_000
    assert result.baseline.e_l == pytest.approx(profile.e_open_l, abs=1e-9) and result.baseline.b_l == profile.b_open_l
    assert result.diagnostics is not None and result.diagnostics.n_face_frames == 600
    assert (result.n_frames, result.n_dropped, result.duration_ms) == (750, 0, 74_900)
    assert result.face_ms == 74_900 and result.face_ratio == 1.0
    assert [step.k for step in result.steps] == list(range(61, 75))
    assert not any(step.full_lookback for step in result.steps)
    assert len(result.runs) == 1 and result.runs[0].start_ms == pytest.approx(1 + 70_050, abs=1)  # 보정 뒤의 감김만
    assert {event.type for event in result.events} >= {EVENT_CALIBRATING}

    # 규칙을 주지 않으면 저장소의 기본 규칙을 쓴다.
    explicit = replay_video(frames, pipeline_spec, None, None, rules=rules_config)
    assert explicit.steps == result.steps and explicit.events == result.events


def test_replay_with_injected_baseline_and_quality_counters(pipeline_spec: PipelineSpec) -> None:
    profile = default_profile(pipeline_spec)
    baseline = Baseline(
        e_l=profile.e_open_l, e_r=profile.e_open_r, b_l=profile.b_open_l, b_r=profile.b_open_r, r0=IDENTITY_R0,
        jaw0=profile.jaw_rest, gaze_h0=0.0, gaze_v0=0.0, t_complete_ms=0, status=BASELINE_OK,
    )  # fmt: skip
    segments = (FaceMissing(2_000, start_ms=3_000), Closure(hold_ms=1_000, start_ms=33_000))
    scenario = Scenario(segments=segments, duration_ms=40_001, fps=FPS, gaps=((10_000, 10_500),))
    frames = render(scenario, pipeline_spec)
    frames.insert(50, frames[49])  # 중복 타임스탬프 하나

    result = replay_video(frames, pipeline_spec, baseline, 30_000)
    assert result.baseline == baseline and result.diagnostics is None
    assert [step.k for step in result.steps] == list(range(31, 41))
    assert len(result.runs) == 1 and not any(event.type == EVENT_CALIBRATING for event in result.events)
    assert (result.n_frames, result.n_dropped, result.duration_ms) == (396, 1, 40_000)
    # 얼굴 시간: 얼굴 프레임 376개 중 마지막 프레임은 시간을 대표하지 않고, 끊김 앞의 프레임은 200 ms까지만 대표한다.
    assert result.face_ms == 374 * 100 + 200
    assert result.face_ratio == pytest.approx(37_600 / 40_000)

    with pytest.raises(ValueError, match="feature_start_ms"):
        replay_video(frames, pipeline_spec, None, 30_000)
    empty = replay_video([], pipeline_spec, None, None)
    assert (empty.n_frames, empty.duration_ms, empty.face_ratio, empty.baseline, empty.steps) == (0, 0, 0.0, None, ())


def test_calibration_frames_never_reach_the_features(pipeline_spec: PipelineSpec) -> None:
    """보정 구간(앞 60초)에 긴 감김과 하품이 있든 없든 그 뒤의 특징은 같다."""
    later = closures(700, [75, 110])
    busy = (Closure(hold_ms=2_500, start_ms=20_000), Yawn(start_ms=30_000), Closure(hold_ms=1_500, start_ms=57_000))
    quiet = replay_video(render(Scenario(later, 130_000, fps=FPS), pipeline_spec), pipeline_spec, None, None)
    noisy = replay_video(render(Scenario(busy + later, 130_000, fps=FPS), pipeline_spec), pipeline_spec, None, None)
    assert noisy.baseline == quiet.baseline and noisy.steps == quiet.steps and noisy.runs == quiet.runs
    first = noisy.steps[0]
    assert first.k == 61
    assert first.values[F_MAX_CLOSURE_MS_60] == 0.0 and first.values[F_LONG_CLOSURE_COUNT_60] == 0.0
    assert all(step.values[F_YAWN_COUNT_60] == 0.0 for step in noisy.steps)
    assert noisy.steps[-1].values[F_LONG_CLOSURE_COUNT_60] == 2.0  # 보정 뒤의 두 감김만 센다


# ---------------------------------------------------------------------------
# 스텝 표와 세 조건
# ---------------------------------------------------------------------------


def test_step_table_shape_and_identifiers(tiny: SimpleNamespace, feature_spec: FeatureSpec) -> None:
    for arm in ARMS:
        table = tiny.tables[arm]
        steps = table.steps
        assert table.arm == arm and table.pipeline_hash == feature_spec.pipeline_hash
        assert list(steps.columns) == list(STEP_COLUMNS) and len(STEP_COLUMNS) == 10 + N_FEATURES
        assert np.isfinite(steps[FEATURES].to_numpy()).all()
        assert list(table.quality["session_id"]) == list(tiny.manifest["session_id"])
        merged = steps[list(ID_COLUMNS)].drop_duplicates().reset_index(drop=True)
        expected = tiny.manifest[list(ID_COLUMNS)].reset_index(drop=True)
        pd.testing.assert_frame_equal(merged, expected)  # 세션 순서와 식별자가 목록과 같다
        for _, group in steps.groupby("session_id", sort=False):
            assert group["k"].is_monotonic_increasing and group["k"].is_unique
            origin = group["t_end_ms"].to_numpy() - 1000 * group["k"].to_numpy()
            assert (origin == 1).all()  # t_end_ms = 첫 프레임의 t_ms + 1000·k


def test_every_arm_starts_features_at_the_sessions_own_baseline_time(tiny: SimpleNamespace) -> None:
    expected_t_b = {session: 60_000 for session in tiny.manifest["session_id"]}
    expected_t_b["S03_0"] = 180_000  # 얼굴이 없어 연장 끝에 전역 상수로 대체된 영상
    for arm in ARMS:
        quality = tiny.tables[arm].quality.set_index("session_id")
        assert quality["t_b_ms"].to_dict() == expected_t_b
        first_k = tiny.tables[arm].steps.groupby("session_id")["k"].min().to_dict()
        assert first_k == {session: t_b // 1000 + 1 for session, t_b in expected_t_b.items()}
        # 1번 alert 영상의 보정 구간에 있던 2초 감김과 하품은 어느 조건의 특징에도 들어가지 않는다.
        first_alert = tiny.tables[arm].steps[tiny.tables[arm].steps["session_id"] == "S01_0"]
        assert (first_alert["yawn_count_60"] == 0.0).all()
        assert (first_alert.loc[first_alert["k"] <= 70, "max_closure_ms_60"] == 0.0).all()
        assert first_alert["max_closure_ms_60"].max() < 1_000.0  # 보정 뒤의 감김(약 0.7초)만 보인다
        # 조회 구간이 다 찬 스텝은 스텝 끝 ≥ T_b + 60초부터다.
        full = tiny.tables[arm].steps[tiny.tables[arm].steps["full_lookback"]]
        assert full.groupby("session_id")["k"].min().to_dict() == {
            session: t_b // 1000 + 60 for session, t_b in expected_t_b.items() if session != "S03_0"
        }


def test_three_arms_give_identical_steps_and_windows(tiny: SimpleNamespace) -> None:
    reference = tiny.tables[ARM_SESSION_BASELINE]
    keys = list(ID_COLUMNS) + ["k", "t_end_ms", "full_lookback"]
    for stride in (1, 5):
        reference_windows = make_windows(reference.steps, 120, stride)
        assert len(reference_windows) > 0
        for arm in (ARM_ALERT_BASELINE, ARM_GLOBAL):
            table = tiny.tables[arm]
            pd.testing.assert_frame_equal(table.steps[keys], reference.steps[keys])
            windows = make_windows(table.steps, 120, stride)
            pd.testing.assert_frame_equal(windows.meta, reference_windows.meta)
            np.testing.assert_array_equal(windows.y, reference_windows.y)
            assert windows.x.shape == reference_windows.x.shape
    # 창 수는 같고 값은 조건마다 다르다.
    session_x = make_windows(reference.steps, 120, 1).x
    assert not np.array_equal(session_x, make_windows(tiny.tables[ARM_GLOBAL].steps, 120, 1).x)
    assert not np.array_equal(session_x, make_windows(tiny.tables[ARM_ALERT_BASELINE].steps, 120, 1).x)


def test_arm_baselines(tiny: SimpleNamespace, pipeline_spec: PipelineSpec, rules_config: RulesConfig) -> None:
    session, alert, global_ = (tiny.tables[arm] for arm in (ARM_SESSION_BASELINE, ARM_ALERT_BASELINE, ARM_GLOBAL))

    # alert 영상은 자기 기준선이 곧 alert 영상 기준선이다: 두 조건에서 같다.
    for session_id in ("S01_0", "S02_0"):
        np.testing.assert_array_equal(step_values(alert, session_id), step_values(session, session_id))
    # 1번의 나머지 영상은 얼굴이 달라서(다른 세션) alert 기준선으로 계산한 값이 자기 기준선 값과 다르다.
    for session_id in ("S01_5", "S01_10"):
        assert not np.array_equal(step_values(alert, session_id), step_values(session, session_id))
        assert not np.array_equal(step_values(global_, session_id), step_values(session, session_id))

    # alert 조건의 값은 "alert 영상의 기준선을 넣고 자기 T_b부터 누적한" 재생과 같다.
    own_alert = replay_video(read_raw_frames(tiny.root / "Fold1_part1/01/0.parquet"), pipeline_spec, None, None)
    assert own_alert.baseline is not None and own_alert.baseline.e_l == pytest.approx(0.30, abs=1e-9)
    injected = replay_video(
        read_raw_frames(tiny.root / "Fold1_part1/01/10.parquet"),
        pipeline_spec, own_alert.baseline, 60_000, rules=rules_config,
    )  # fmt: skip
    np.testing.assert_array_equal(step_values(alert, "S01_10"), np.array([step.values for step in injected.steps]))
    # 1번의 다른 세션은 뜬 눈의 EAR이 더 작으므로(0.25 대 0.30) alert 기준선으로는 눈이 더 감긴 것으로 읽힌다.
    mean_alert = step_values(alert, "S01_10")[:, F_CLOSURE_MEAN_60].mean()
    mean_session = step_values(session, "S01_10")[:, F_CLOSURE_MEAN_60].mean()
    assert mean_alert > mean_session + 0.05

    quality = {arm: tiny.tables[arm].quality.set_index("session_id") for arm in ARMS}
    assert set(quality[ARM_SESSION_BASELINE]["baseline_used"]) == {"session"}
    assert set(quality[ARM_GLOBAL]["baseline_used"]) == {"global"}
    assert not quality[ARM_SESSION_BASELINE]["alert_fallback"].any() and not quality[ARM_GLOBAL]["alert_fallback"].any()
    used = quality[ARM_ALERT_BASELINE]["baseline_used"].to_dict()
    assert {used[s] for s in used if not s.startswith("S03")} == {"alert"}


def test_subject_without_usable_alert_baseline_falls_back_to_global(tiny: SimpleNamespace) -> None:
    alert = tiny.tables[ARM_ALERT_BASELINE]
    quality = alert.quality.set_index("session_id")
    assert quality.loc["S03_0", "baseline_status"] == BASELINE_FALLBACK_GLOBAL
    assert quality["baseline_status"].drop("S03_0").unique().tolist() == [BASELINE_OK]
    flagged = quality[quality["alert_fallback"]]
    assert flagged.index.tolist() == ["S03_0", "S03_5", "S03_10"]  # 피험자 3의 모든 세션에 표시된다
    assert flagged["baseline_used"].unique().tolist() == ["global"]
    # 값은 전역 조건과 같고(같은 상수, 같은 시작 시각) 자기 기준선 조건과는 다르다.
    for session_id in ("S03_0", "S03_5", "S03_10"):
        np.testing.assert_array_equal(step_values(alert, session_id), step_values(tiny.tables[ARM_GLOBAL], session_id))
    assert not np.array_equal(
        step_values(alert, "S03_5"), step_values(tiny.tables[ARM_SESSION_BASELINE], "S03_5")
    )
    assert quality.loc["S03_0", "base_e_l"] == 0.27  # 대체된 기준선의 값(전역 상수)이 적힌다
    assert quality.loc["S01_0", "base_e_l"] == pytest.approx(0.30, abs=1e-9)
    assert quality.loc["S01_0", "calib_face_frames"] == 600 and quality.loc["S03_0", "calib_face_frames"] == 100


def test_global_override_replaces_the_global_constants(
    small_tree: SimpleNamespace, pipeline_spec: PipelineSpec
) -> None:
    default = build_step_table(small_tree.manifest, small_tree.root, pipeline_spec, ARM_GLOBAL)
    session = build_step_table(small_tree.manifest, small_tree.root, pipeline_spec, ARM_SESSION_BASELINE)
    # 기본 얼굴은 전역 상수와 같으므로 전역 조건의 closure가 자기 기준선 조건과 같다(특징은 반올림 오차 안에서 같다).
    np.testing.assert_allclose(default.steps[FEATURES].to_numpy(), session.steps[FEATURES].to_numpy(), atol=1e-9)

    override = dataclasses.replace(dataset.calibrate.global_baseline(pipeline_spec), e_l=0.33, e_r=0.33)
    shifted = build_step_table(small_tree.manifest, small_tree.root, pipeline_spec, ARM_GLOBAL, override)
    assert shifted.steps["closure_mean_60"].mean() > default.steps["closure_mean_60"].mean() + 0.05
    with pytest.raises(ValueError, match="arm must be one of"):
        build_step_table(small_tree.manifest, small_tree.root, pipeline_spec, "per_video")
    with pytest.raises(ManifestError, match="lacks columns"):
        build_step_table(small_tree.manifest.drop(columns=["rel_path"]), small_tree.root, pipeline_spec, ARM_GLOBAL)


# ---------------------------------------------------------------------------
# 창
# ---------------------------------------------------------------------------


def test_window_counts_of_the_tiny_dataset(tiny: SimpleNamespace) -> None:
    steps = tiny.tables[ARM_ALERT_BASELINE].steps
    windows = make_windows(steps, 120, 1)
    counts = windows.meta.groupby("session_id", sort=False).size().to_dict()
    # 250초: 스텝 61..249, 다 찬 스텝 120..249(130개) → 11개. 245초: 6개. 끊긴 영상: 양쪽에서 6개씩.
    assert counts == {
        "S01_0": 11, "S01_5": 12, "S01_10": 11,
        "S02_0": 11, "S02_5": 11, "S02_10_1": 11, "S02_10_2": 6,
        "S03_5": 11, "S03_10": 11,
    }  # fmt: skip
    assert windows.x.shape == (95, 120, N_FEATURES) and windows.x.dtype == np.float32
    assert windows.y.dtype == np.int64 and windows.y.tolist() == windows.meta["class_index"].tolist()
    assert list(windows.meta.columns) == list(ID_COLUMNS) + ["k_end", "t_end_ms"]
    assert windows_per_class(steps, 120, 1) == {"alert": 22, "low_vigilant": 34, "drowsy": 39}

    strided = make_windows(steps, 120, 5)
    assert strided.meta.groupby("session_id", sort=False).size().to_dict() == {
        "S01_0": 3, "S01_5": 4, "S01_10": 3,
        "S02_0": 3, "S02_5": 3, "S02_10_1": 3, "S02_10_2": 2,
        "S03_5": 3, "S03_10": 3,
    }  # fmt: skip
    assert windows_per_class(steps, 120, 5) == {"alert": 6, "low_vigilant": 10, "drowsy": 11}
    # 이어진 구간의 첫 창은 그 구간의 첫 120스텝이다(앱이 처음 확률을 내는 시점).
    assert strided.meta.loc[strided.meta["session_id"] == "S01_0", "k_end"].tolist() == [239, 244, 249]
    assert strided.meta["t_end_ms"].tolist() == (1 + 1000 * strided.meta["k_end"]).tolist()
    # 리포트에 싣는 창 수(`windows_per_class`, 창을 만들지 않고 센다)는 어느 보폭에서나 실제로 만든 창의 수와 같다.
    for stride in (1, 2, 5, 7, 10, 11, 200):
        made = make_windows(steps, 120, stride).meta.groupby("class_index").size().to_dict()
        counted = windows_per_class(steps, 120, stride)
        assert counted == {name: made.get(index, 0) for index, name in enumerate(CLASS_ORDER)}, stride
    assert windows_per_class(steps, 120, 200) == {"alert": 2, "low_vigilant": 4, "drowsy": 4}  # 이어진 구간마다 하나


def test_windows_hold_only_full_lookback_steps_of_one_session(tiny: SimpleNamespace) -> None:
    for arm in ARMS:
        steps = tiny.tables[arm].steps
        assert not steps["full_lookback"].all()  # 덜 찬 스텝이 표에는 있다
        for stride in (1, 5):
            assert_windows_are_whole(make_windows(steps, 120, stride), steps, 120)

    windows = make_windows(tiny.tables[ARM_SESSION_BASELINE].steps, 120, 1)
    meta = windows.meta
    # 나뉜 영상: 두 부분이 같은 논리 영상이지만 창은 부분마다 따로 나온다.
    parts = meta[meta["video_id"] == "S02_10"]
    assert parts.groupby("session_id", sort=False)["k_end"].agg(["min", "max"]).to_dict("index") == {
        "S02_10_1": {"min": 239, "max": 249},
        "S02_10_2": {"min": 239, "max": 244},
    }
    assert parts["part"].unique().tolist() == [1, 2]
    # 10초 넘게 끊긴 영상: 끊기기 전(스텝 ~244)과 다시 60초가 찬 뒤(스텝 320~)의 창만 있다.
    gapped = meta.loc[meta["session_id"] == "S01_5", "k_end"].tolist()
    assert gapped == [239, 240, 241, 242, 243, 244, 439, 440, 441, 442, 443, 444]
    gap_steps = tiny.tables[ARM_SESSION_BASELINE].steps
    gap_steps = gap_steps[gap_steps["session_id"] == "S01_5"]
    assert gap_steps["k"].tolist() == list(range(61, 245)) + list(range(261, 445))
    assert gap_steps.loc[gap_steps["k"].between(261, 319), "full_lookback"].sum() == 0


def test_make_windows_on_hand_built_steps() -> None:
    def table(session_id: str, ks: list[int], full_from: int, video_id: str = "S01_0", part: int = 0) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "video_id": video_id, "session_id": session_id, "subject": 1, "fold": 1, "label": 0,
                "class_index": 0, "part": part, "k": ks, "t_end_ms": [1 + 1000 * k for k in ks],
                "full_lookback": [k >= full_from for k in ks],
            }
        )  # fmt: skip
        for index, name in enumerate(FEATURE_NAMES):
            frame[name] = [k + index / 100.0 for k in ks]
        return frame

    # PLAN 4.4절: 600초 영상은 보폭 1에서 362개, 보폭 10에서 37개.
    ten_minutes = table("S01_0", list(range(61, 601)), 120)
    assert len(make_windows(ten_minutes, 120, 1)) == 362 and len(make_windows(ten_minutes, 120, 10)) == 37
    windows = make_windows(ten_minutes, 120, 5)
    assert len(windows) == 73 and windows.meta["k_end"].tolist() == list(range(239, 601, 5))
    assert windows.x[0, 0, 0] == 120.0 and windows.x[0, -1, 0] == 239.0  # 가장 오래된 스텝이 앞이다
    assert windows.x[3, 0, 5] == np.float32(135.05)

    # 240초보다 짧으면 창이 없다.
    short = make_windows(table("S01_0", list(range(61, 239)), 120), 120, 1)
    assert len(short) == 0 and short.x.shape == (0, 120, N_FEATURES)
    assert list(short.meta.columns)[-2:] == ["k_end", "t_end_ms"]

    # 세션 경계: 스텝 번호가 이어져 보여도 다른 세션(나뉜 부분)과 합쳐지지 않는다.
    two_parts = pd.concat(
        [
            table("S01_10_1", list(range(1, 131)), 1, "S01_10", 1),
            table("S01_10_2", list(range(131, 261)), 131, "S01_10", 2),
        ],
        ignore_index=True,
    )
    windows = make_windows(two_parts, 120, 1)
    assert windows.meta.groupby("session_id", sort=False).size().to_dict() == {"S01_10_1": 11, "S01_10_2": 11}
    assert_windows_are_whole(windows, two_parts, 120)

    # 번호가 건너뛰거나 중간에 덜 찬 스텝이 있으면 그 지점을 넘는 창이 없다.
    jump = table("S01_0", list(range(1, 126)) + list(range(127, 300)), 1)
    assert make_windows(jump, 120, 1).meta["k_end"].tolist() == list(range(120, 126)) + list(range(246, 300))
    hole = table("S01_0", list(range(1, 300)), 1)
    hole.loc[hole["k"] == 150, "full_lookback"] = False
    assert make_windows(hole, 120, 1).meta["k_end"].tolist() == list(range(120, 150)) + list(range(270, 300))
    # 행 순서가 섞여 있어도 세션 안에서 k 순으로 만든다.
    shuffled = ten_minutes.sample(frac=1.0, random_state=0).reset_index(drop=True)
    np.testing.assert_array_equal(make_windows(shuffled, 120, 10).x, make_windows(ten_minutes, 120, 10).x)

    subset = windows.subset(windows.meta["session_id"] == "S01_10_2")
    assert len(subset) == 11 and subset.meta["session_id"].unique().tolist() == ["S01_10_2"]
    assert subset.subset([0, 2]).meta["k_end"].tolist() == [250, 252]
    with pytest.raises(ValueError):
        windows.subset(np.array([True, False]))
    with pytest.raises(ValueError):
        make_windows(ten_minutes, 120, 0)
    with pytest.raises(ValueError, match="lacks columns"):
        make_windows(ten_minutes.drop(columns=["full_lookback"]), 120, 1)
    with pytest.raises(ValueError, match="duplicate step numbers"):
        make_windows(pd.concat([ten_minutes, ten_minutes.iloc[:1]], ignore_index=True), 120, 1)


# ---------------------------------------------------------------------------
# 품질 표 (PLAN 1.2절의 결정 D7: 얼굴 검출 비율 0.7 이상이고 창 1개 이상이면 품질 기준 통과)
# ---------------------------------------------------------------------------


def test_quality_table_and_gate(tiny: SimpleNamespace, pipeline_spec: PipelineSpec) -> None:
    quality = tiny.tables[ARM_ALERT_BASELINE].quality.set_index("session_id")
    assert quality["raw_status"].unique().tolist() == ["ok"]
    assert quality.loc["S01_0", ["n_frames", "n_dropped", "duration_ms"]].tolist() == [2500, 0, 249_900]
    assert quality.loc["S01_0", "face_ratio"] == 1.0
    assert quality.loc["S01_5", "n_frames"] == 4450 - 150
    # 끊긴 15초 중 끊기기 직전 프레임이 대표하는 200 ms만 얼굴 시간이다.
    assert quality.loc["S01_5", "face_ratio"] == pytest.approx((444_900 - 15_100 + 200) / 444_900)
    assert quality.loc["S03_0", "face_ratio"] == pytest.approx(29_900 / 199_900)
    assert quality.loc["S01_0", ["n_steps", "n_full_steps", "n_windows"]].tolist() == [189, 130, 11]
    assert quality.loc["S01_5", ["n_steps", "n_full_steps", "n_windows"]].tolist() == [368, 250, 12]
    assert quality.loc["S03_0", ["n_steps", "n_full_steps", "n_windows"]].tolist() == [19, 0, 0]

    # 논리 영상 단위: 나뉜 영상은 부분을 합친다.
    assert quality.loc[["S02_10_1", "S02_10_2"], "video_n_windows"].tolist() == [17, 17]
    assert quality.loc["S02_10_1", "video_face_ratio"] == quality.loc["S02_10_2", "video_face_ratio"] == 1.0
    assert quality["gate_pass"].to_dict() == {session: session != "S03_0" for session in quality.index}

    # 기준을 바꾸면 판정만 달라진다. 라벨은 쓰지 않는다.
    strict_gate = QualityGate(min_face_ratio=0.99, min_windows=12)
    strict = apply_quality_gate(tiny.tables[ARM_ALERT_BASELINE].quality, strict_gate)
    passed = strict.loc[strict["gate_pass"], "session_id"].tolist()
    assert passed == ["S02_10_1", "S02_10_2"]  # S01_5는 얼굴 비율 0.97로 탈락, 나머지는 창이 11개
    loose = build_step_table(
        tiny.manifest[tiny.manifest["subject"] == 3].reset_index(drop=True), tiny.root, pipeline_spec,
        ARM_SESSION_BASELINE, gate=QualityGate(min_face_ratio=0.1, min_windows=0),
    )  # fmt: skip
    assert loose.quality["gate_pass"].all()


def test_short_and_missing_sessions_stay_in_the_quality_table(
    small_tree: SimpleNamespace, pipeline_spec: PipelineSpec
) -> None:
    (small_tree.root / "02/5.parquet").unlink()  # 추출에 실패한 영상
    short = Scenario(segments=(), duration_ms=40_000, fps=FPS)  # 보정이 끝나기 전에 끝나는 영상
    write_session(small_tree.root, "02/0.parquet", short, pipeline_spec)

    for arm in ARMS:
        table = build_step_table(small_tree.manifest, small_tree.root, pipeline_spec, arm)
        quality = table.quality.set_index("session_id")
        assert len(quality) == 6 and not quality["gate_pass"].any()  # 창이 없으면 통과하지 못한다
        assert quality.loc["S02_5", ["raw_status", "n_frames", "baseline_status", "t_b_ms"]].tolist() == [
            "missing", 0, "none", -1,
        ]  # fmt: skip
        assert quality.loc["S02_0", ["raw_status", "n_frames", "baseline_status", "t_b_ms"]].tolist() == [
            "ok", 400, "none", -1,
        ]  # fmt: skip
        assert quality.loc[["S02_0", "S02_5"], "baseline_used"].tolist() == ["none", "none"]
        assert set(table.steps["session_id"]) == {"S01_0", "S01_5", "S01_10", "S02_10"}
        assert np.isnan(quality.loc["S02_0", "base_e_l"])
    # alert 영상의 보정이 끝나지 않은 피험자: 남은 영상은 전역 상수로 계산하고, 그 피험자의 세션을 모두 표시한다.
    alert = build_step_table(small_tree.manifest, small_tree.root, pipeline_spec, ARM_ALERT_BASELINE).quality
    assert alert.set_index("session_id")["alert_fallback"].to_dict() == {
        "S01_0": False, "S01_5": False, "S01_10": False, "S02_0": True, "S02_5": True, "S02_10": True,
    }  # fmt: skip
    assert alert.set_index("session_id").loc["S02_10", "baseline_used"] == "global"
    for arm in (ARM_SESSION_BASELINE, ARM_GLOBAL):
        other = build_step_table(small_tree.manifest, small_tree.root, pipeline_spec, arm)
        assert not other.quality["alert_fallback"].any()

    with pytest.raises(ManifestError, match="none of the 6 raw parquet files"):
        build_step_table(small_tree.manifest, small_tree.root.parent / "elsewhere", pipeline_spec, ARM_GLOBAL)


# ---------------------------------------------------------------------------
# 리포트용 기준선·창 요약 (PLAN 4.4절 마지막 항목)
# ---------------------------------------------------------------------------


def test_baseline_window_summary_of_the_tiny_dataset(
    tiny: SimpleNamespace, small_tree: SimpleNamespace, pipeline_spec: PipelineSpec
) -> None:
    summary = baseline_window_summary(tiny.tables, 120, (5, 1), 60_000)
    assert json.loads(json.dumps(summary)) == summary  # 그대로 JSON에 넣을 수 있다
    # 3번 피험자의 alert 영상만 보정이 연장됐고(얼굴이 170초 동안 없어 180초에 전역 상수로 대체), 나머지는 60초다.
    replaced = {
        "video_id": "S03_0", "session_id": "S03_0", "subject": 3, "fold": 1, "class_index": 0,
        "t_b_ms": 180_000, "baseline_status": "fallback_global",
    }  # fmt: skip
    assert list(replaced) == list(BASELINE_RECORD_COLUMNS)
    # 창 수는 test_window_counts_of_the_tiny_dataset에서 손으로 센 값이다. 세 조건은 같은 창을 쓴다.
    stride_5 = {"alert": 6, "low_vigilant": 10, "drowsy": 11, "total": 27}
    stride_1 = {"alert": 22, "low_vigilant": 34, "drowsy": 39, "total": 95}
    assert summary == {
        "normal_t_b_ms": 60_000,
        "n_sessions": 10,
        "n_videos": 9,
        "extended_baselines": [replaced],
        "fallback_baselines": [replaced],
        "unfinished_baselines": [],
        "missing_raw": [],
        "alert_fallback_subjects": [3],
        "alert_fallback_sessions": ["S03_0", "S03_5", "S03_10"],
        "window_counts": [
            {"arm": arm, "stride": stride, **counts}
            for arm in ARMS
            for stride, counts in ((5, stride_5), (1, stride_1))
        ],
    }
    assert type(summary["extended_baselines"][0]["t_b_ms"]) is int
    assert type(summary["window_counts"][0]["total"]) is int

    # alert 영상 기준선 조건의 표가 없으면 그 조건의 대체 여부는 알 수 없다(None). 같은 보폭은 한 번만 센다.
    only_global = baseline_window_summary({ARM_GLOBAL: tiny.tables[ARM_GLOBAL]}, 120, [1, 1, 5], 60_000)
    assert only_global["alert_fallback_subjects"] is None and only_global["alert_fallback_sessions"] is None
    assert [(row["arm"], row["stride"], row["total"]) for row in only_global["window_counts"]] == [
        (ARM_GLOBAL, 1, 95), (ARM_GLOBAL, 5, 27),
    ]  # fmt: skip
    assert only_global["extended_baselines"] == [replaced]
    # 기준을 180초로 주면 연장된 것은 없고, 대체된 것은 그대로 대체로 남는다.
    late = baseline_window_summary(tiny.tables, 120, (1,), 180_000)
    assert late["extended_baselines"] == [] and late["fallback_baselines"] == [replaced]

    other = build_step_table(small_tree.manifest, small_tree.root, pipeline_spec, ARM_SESSION_BASELINE)
    with pytest.raises(ValueError, match="other sessions or baselines"):
        baseline_window_summary({ARM_GLOBAL: tiny.tables[ARM_GLOBAL], ARM_SESSION_BASELINE: other}, 120, (1,), 60_000)
    with pytest.raises(ValueError, match="at least one"):
        baseline_window_summary({}, 120, (1,), 60_000)
    with pytest.raises(ValueError, match=">= 1"):
        baseline_window_summary(tiny.tables, 120, (0,), 60_000)


def test_baseline_window_summary_lists_extended_unfinished_and_missing_sessions(
    small_tree: SimpleNamespace, pipeline_spec: PipelineSpec
) -> None:
    root = small_tree.root
    # 1번의 5 영상: 앞 40초 동안 얼굴이 없다. 60초에는 얼굴 프레임이 200개뿐이라 30초 연장되고 90초에 대체 없이 완성된다.
    write_session(root, "01/5.parquet", Scenario((FaceMissing(40_000),), 100_000, fps=FPS), pipeline_spec)
    write_session(root, "02/0.parquet", Scenario((), 40_000, fps=FPS), pipeline_spec)  # 보정이 끝나기 전에 끝남
    (root / "02/5.parquet").unlink()  # 추출 결과 없음
    arms = (ARM_SESSION_BASELINE, ARM_ALERT_BASELINE)
    tables = dataset._build_step_tables(small_tree.manifest, root, pipeline_spec, arms, None, None, 1)
    summary = baseline_window_summary(tables, 120, (5, 1), pipeline_spec.baseline.duration_ms)
    assert pipeline_spec.baseline.duration_ms == 60_000 and summary["normal_t_b_ms"] == 60_000
    assert (summary["n_sessions"], summary["n_videos"]) == (6, 6)
    assert summary["extended_baselines"] == [
        {"video_id": "S01_5", "session_id": "S01_5", "subject": 1, "fold": 1, "class_index": 1,
         "t_b_ms": 90_000, "baseline_status": "ok"},
    ]  # fmt: skip
    assert summary["fallback_baselines"] == []  # 연장됐지만 대체되지는 않았다
    assert summary["unfinished_baselines"] == [
        {"video_id": "S02_0", "session_id": "S02_0", "subject": 2, "fold": 1, "class_index": 0,
         "t_b_ms": -1, "baseline_status": "none"},
    ]  # fmt: skip
    assert summary["missing_raw"] == ["S02_5"]
    # 2번은 alert 영상의 보정이 끝나지 않아 alert 기준선이 없다: 그 피험자의 모든 세션이 대체로 표시된다.
    assert summary["alert_fallback_subjects"] == [2]
    assert summary["alert_fallback_sessions"] == ["S02_0", "S02_5", "S02_10"]
    assert [(row["arm"], row["stride"], row["total"]) for row in summary["window_counts"]] == [
        (ARM_SESSION_BASELINE, 5, 0), (ARM_SESSION_BASELINE, 1, 0),
        (ARM_ALERT_BASELINE, 5, 0), (ARM_ALERT_BASELINE, 1, 0),
    ]  # fmt: skip


# ---------------------------------------------------------------------------
# 중첩 분할
# ---------------------------------------------------------------------------


def test_nested_splits_on_sixty_subjects(tmp_path: Path) -> None:
    manifest = build_manifest(touch_tree(tmp_path / "raw", rldd_like_names(60, "parquet")))
    splits = list(nested_splits(manifest))
    assert [split.test_fold for split in splits] == [1, 2, 3, 4, 5]
    for split in splits:
        assert split.test_subjects == tuple(range(12 * (split.test_fold - 1) + 1, 12 * split.test_fold + 1))
        assert len(split.train_subjects) == 48 and not set(split.train_subjects) & set(split.test_subjects)
        assert sorted(split.train_subjects + split.test_subjects) == list(range(1, 61))
        assert [inner.val_fold for inner in split.inner] == [f for f in range(1, 6) if f != split.test_fold]
        for inner in split.inner:
            assert (len(inner.train_subjects), len(inner.val_subjects), len(split.test_subjects)) == (36, 12, 12)
            assert inner.val_subjects == tuple(range(12 * (inner.val_fold - 1) + 1, 12 * inner.val_fold + 1))
            assert sorted(inner.train_subjects + inner.val_subjects) == sorted(split.train_subjects)
            assert_subject_disjoint(inner.train_subjects, inner.val_subjects, split.test_subjects)
    assert sum(len(split.inner) for split in splits) == 20


def test_splits_and_step_tables_reject_a_subject_in_two_folds(
    tmp_path: Path, small_tree: SimpleNamespace, pipeline_spec: PipelineSpec
) -> None:
    manifest = build_manifest(touch_tree(tmp_path / "raw", rldd_like_names(60, "parquet")))
    leaky = manifest.copy()
    leaky.loc[(leaky["subject"] == 7) & (leaky["label"] == 10), "fold"] = 4  # 피험자 7이 fold 1과 4에 걸친다
    with pytest.raises(SubjectOverlapError, match="subject 7"):
        list(nested_splits(leaky))
    with pytest.raises(ValueError, match="at least 3 folds"):
        list(nested_splits(manifest[manifest["fold"] <= 2]))

    leaky_small = small_tree.manifest.copy()
    leaky_small.loc[(leaky_small["subject"] == 2) & (leaky_small["label"] == 5), "fold"] = 2
    with pytest.raises(SubjectOverlapError, match="subject 2"):
        build_step_table(leaky_small, small_tree.root, pipeline_spec, ARM_SESSION_BASELINE)


# ---------------------------------------------------------------------------
# 캐시, 병렬 처리
# ---------------------------------------------------------------------------


def test_step_table_cache(
    small_tree: SimpleNamespace, pipeline_spec: PipelineSpec, feature_spec: FeatureSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert spec_pipeline_hash(pipeline_spec) == feature_spec.pipeline_hash
    cache = step_cache_dir(small_tree.root)
    assert cache == small_tree.root.parent / "raw_steps" and not cache.exists()

    builds: list[tuple[str, ...]] = []
    real_build = dataset._build_step_tables

    def counting_build(manifest, raw_dir, spec, arms, *args, **kwargs):
        builds.append(tuple(arms))
        return real_build(manifest, raw_dir, spec, arms, *args, **kwargs)

    monkeypatch.setattr(dataset, "_build_step_tables", counting_build)

    first = load_step_tables(small_tree.manifest, small_tree.root, pipeline_spec)
    assert list(first) == list(ARMS) and builds == [ARMS]  # 세 조건을 한 번에 만든다
    prefix = feature_spec.pipeline_hash[:16]
    assert sorted(path.name for path in cache.iterdir()) == sorted(
        f"{arm}_{prefix}.{kind}.parquet" for arm in ARMS for kind in ("steps", "quality")
    )
    for arm in ARMS:
        direct = real_build(small_tree.manifest, small_tree.root, pipeline_spec, (arm,), None, None, 1)[arm]
        pd.testing.assert_frame_equal(first[arm].steps, direct.steps)
        pd.testing.assert_frame_equal(first[arm].quality, direct.quality)

    # 두 번째에는 캐시에서 읽는다. 값과 자료형이 같다.
    again = load_step_tables(small_tree.manifest, small_tree.root, pipeline_spec, [ARM_GLOBAL, ARM_SESSION_BASELINE])
    assert builds == [ARMS] and list(again) == [ARM_GLOBAL, ARM_SESSION_BASELINE]
    for arm in again:
        pd.testing.assert_frame_equal(again[arm].steps, first[arm].steps)
        pd.testing.assert_frame_equal(again[arm].quality, first[arm].quality)
        assert again[arm].pipeline_hash == feature_spec.pipeline_hash

    # 품질 기준은 캐시에 묶이지 않는다.
    loose_gate = QualityGate(min_face_ratio=0.5, min_windows=0)
    gated = load_step_tables(small_tree.manifest, small_tree.root, pipeline_spec, [ARM_GLOBAL], gate=loose_gate)
    assert builds == [ARMS] and gated[ARM_GLOBAL].quality["gate_pass"].all()
    assert not first[ARM_GLOBAL].quality["gate_pass"].any()

    # spec이 바뀌면(pipeline_hash) 다른 파일로 다시 만든다.
    other_spec = apply_overrides(pipeline_spec, {"closure.w": 1.0})
    other = load_step_tables(small_tree.manifest, small_tree.root, other_spec, [ARM_SESSION_BASELINE])
    assert builds[-1] == (ARM_SESSION_BASELINE,) and len(builds) == 2
    assert other[ARM_SESSION_BASELINE].pipeline_hash != feature_spec.pipeline_hash
    assert len(list(cache.iterdir())) == 8

    # 전역 기준선을 바꿔 넣으면 전역 조건만 따로 캐시한다.
    override = dataclasses.replace(dataset.calibrate.global_baseline(pipeline_spec), e_l=0.33, e_r=0.33)
    shifted = load_step_tables(small_tree.manifest, small_tree.root, pipeline_spec, ARMS, global_override=override)
    assert builds[-1] == (ARM_GLOBAL,) and len(builds) == 3
    assert not shifted[ARM_GLOBAL].steps.equals(first[ARM_GLOBAL].steps)
    load_step_tables(small_tree.manifest, small_tree.root, pipeline_spec, ARMS, global_override=override)
    assert len(builds) == 3

    # 추출 결과 파일이 바뀌면 다시 만든다.
    write_session(small_tree.root, "01/10.parquet", Scenario(closures(900, [63, 66]), 70_000, fps=FPS), pipeline_spec)
    rebuilt = load_step_tables(small_tree.manifest, small_tree.root, pipeline_spec, [ARM_SESSION_BASELINE])
    assert builds[-1] == (ARM_SESSION_BASELINE,) and len(builds) == 4
    assert not rebuilt[ARM_SESSION_BASELINE].steps.equals(first[ARM_SESSION_BASELINE].steps)

    # 다른 위치를 줄 수 있다.
    elsewhere = small_tree.root.parent / "my_cache"
    load_step_tables(small_tree.manifest, small_tree.root, pipeline_spec, [ARM_GLOBAL], cache_dir=elsewhere)
    assert len(builds) == 5 and len(list(elsewhere.iterdir())) == 2
    with pytest.raises(ValueError, match="arm must be one of"):
        load_step_tables(small_tree.manifest, small_tree.root, pipeline_spec, ["nope"])


def test_code_fingerprint_follows_the_bytes_of_the_listed_files(tmp_path: Path, repo_root: Path) -> None:
    # 특징을 만드는 이식 모듈과 그것을 부르는 쪽이 모두 지문에 들어 있다.
    assert set(FINGERPRINT_MODULES) >= {"types", "signals", "calibrate", "blink", "step_features", "pipeline"}
    assert set(FINGERPRINT_MODULES) >= {"rules", "spec", "extract", "dataset"}
    files = fingerprint_files()
    assert [path.name for path in files] == [f"{name}.py" for name in FINGERPRINT_MODULES]
    assert all(path.is_file() and path.parent == repo_root / "drowsy" for path in files)

    def expected(paths: list[Path]) -> str:  # 설명대로 따로 계산: 이름, 0, 바이트 수, 0, 바이트
        digest = hashlib.sha256()
        for path in paths:
            data = path.read_bytes()
            digest.update(path.name.encode("utf-8") + b"\0" + str(len(data)).encode("ascii") + b"\0" + data)
        return digest.hexdigest()

    assert code_fingerprint() == expected(list(files)) and len(code_fingerprint()) == 64

    first = tmp_path / "blink.py"
    second = tmp_path / "step_features.py"
    first.write_bytes(b"x = 1\n")
    second.write_bytes(b"y = 2\n")
    base = code_fingerprint([first, second])
    assert base == expected([first, second]) and base == code_fingerprint([first, second])
    second.write_bytes(b"y = 3\n")  # 한 파일의 한 바이트
    changed = code_fingerprint([first, second])
    assert changed != base and changed == expected([first, second])
    second.write_bytes(b"y = 2\n")
    assert code_fingerprint([first, second]) == base  # 바이트가 돌아오면 지문도 돌아온다
    second.write_bytes(b"y = 2\n# note\n")  # 주석만 바꿔도 달라진다(안전한 쪽)
    assert code_fingerprint([first, second]) != base
    second.write_bytes(b"y = 2\n")
    assert code_fingerprint([second, first]) != base  # 순서
    assert code_fingerprint([first]) != base  # 파일이 빠짐
    # 파일 경계: 앞 파일의 끝 바이트를 뒤 파일 앞으로 옮기면 이어 붙인 바이트는 같아도 지문은 다르다.
    first.write_bytes(b"x = 1")
    second.write_bytes(b"\ny = 2\n")
    assert code_fingerprint([first, second]) != base
    renamed = tmp_path / "other.py"
    renamed.write_bytes(b"\ny = 2\n")
    assert code_fingerprint([first, renamed]) != code_fingerprint([first, second])  # 이름도 들어간다
    with pytest.raises(OSError):
        code_fingerprint([tmp_path / "missing.py"])


def test_cache_is_rebuilt_when_the_code_fingerprint_changes(
    small_tree: SimpleNamespace, pipeline_spec: PipelineSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    builds = spy_on_builder(monkeypatch)
    cache = step_cache_dir(small_tree.root)
    arms = [ARM_ALERT_BASELINE, ARM_SESSION_BASELINE]
    first = load_step_tables(small_tree.manifest, small_tree.root, pipeline_spec, arms)
    assert builds == [(tuple(arms), None)]
    steps_file = cache / f"{ARM_ALERT_BASELINE}_{spec_pipeline_hash(pipeline_spec)[:16]}.steps.parquet"
    stored = cache_metadata(steps_file)
    assert stored["format"] == dataset.CACHE_FORMAT == 4 and stored["code"] == code_fingerprint()
    assert stored["frame_drop"] is None and stored["pipeline_hash"] == spec_pipeline_hash(pipeline_spec)
    load_step_tables(small_tree.manifest, small_tree.root, pipeline_spec, arms)
    assert len(builds) == 1  # 코드가 그대로면 캐시에서 읽는다

    # 캐시가 정말 읽히는지 보이기 위해, 열쇠는 그대로 두고 내용만 바꾼 파일을 넣는다.
    doctored = first[ARM_ALERT_BASELINE].steps.copy()
    doctored["perclos_60"] = 0.123
    dataset._write_cached(doctored, steps_file, stored)
    stale = load_step_tables(small_tree.manifest, small_tree.root, pipeline_spec, arms)
    assert len(builds) == 1 and (stale[ARM_ALERT_BASELINE].steps["perclos_60"] == 0.123).all()

    # 특징 코드가 바뀐 것처럼 지문만 바꾼다: 상수(pipeline_hash)와 추출 결과가 같아도 요청한 조건을 모두 다시 만든다.
    monkeypatch.setattr(dataset, "code_fingerprint", lambda files=None: "f" * 64)
    rebuilt = load_step_tables(small_tree.manifest, small_tree.root, pipeline_spec, arms)
    assert builds[1:] == [(tuple(arms), None)]
    pd.testing.assert_frame_equal(rebuilt[ARM_ALERT_BASELINE].steps, first[ARM_ALERT_BASELINE].steps)
    assert cache_metadata(steps_file)["code"] == "f" * 64
    load_step_tables(small_tree.manifest, small_tree.root, pipeline_spec, arms)
    assert len(builds) == 2  # 새 지문으로 쓴 캐시는 그 지문에서 다시 읽힌다
    monkeypatch.setattr(dataset, "code_fingerprint", code_fingerprint)
    load_step_tables(small_tree.manifest, small_tree.root, pipeline_spec, [ARM_SESSION_BASELINE])
    assert builds[2:] == [((ARM_SESSION_BASELINE,), None)]  # 지문이 다시 달라졌으므로(원래 코드) 또 만든다

    # 형식 1의 캐시(코드 지문이 없던 때의 파일)는 받지 않는다.
    session_file = steps_file.with_name(steps_file.name.replace(ARM_ALERT_BASELINE, ARM_SESSION_BASELINE))
    old_metadata = cache_metadata(session_file)
    del old_metadata["code"], old_metadata["frame_drop"]
    old_metadata["format"] = 1
    old_steps = first[ARM_SESSION_BASELINE].steps.copy()
    old_steps["perclos_60"] = 0.123
    dataset._write_cached(old_steps, session_file, old_metadata)
    fresh = load_step_tables(small_tree.manifest, small_tree.root, pipeline_spec, [ARM_SESSION_BASELINE])
    assert len(builds) == 4
    pd.testing.assert_frame_equal(fresh[ARM_SESSION_BASELINE].steps, first[ARM_SESSION_BASELINE].steps)
    assert cache_metadata(session_file)["format"] == 4

    # 형식 2(추출 결과의 지문이 파일마다 크기와 수정 시각뿐)와 형식 3(거기에 parquet 꼬리의 sha256을 더한 것)의 캐시도
    # 받지 않고 한 번씩 다시 만든다. 두 지문은 그때의 설명대로 여기서 따로 계산한다. 지문은 지금 것 그대로이고 형식
    # 번호만 3인 캐시도 받지 않는다.
    current = cache_metadata(session_file)
    entries_2: list[list[Any]] = []
    entries_3: list[list[Any]] = []
    for session_id, rel_path in zip(small_tree.manifest["session_id"], small_tree.manifest["rel_path"]):
        raw_file = small_tree.root / rel_path
        stat = raw_file.stat()
        entries_2.append([session_id, rel_path, stat.st_size, stat.st_mtime_ns])
        footer_digest = hashlib.sha256(parquet_footer(raw_file)).hexdigest()
        entries_3.append([session_id, rel_path, stat.st_size, stat.st_mtime_ns, footer_digest])
    old_caches = [
        dict(current, format=2, source=hashlib.sha256(json.dumps(entries_2).encode("utf-8")).hexdigest()),
        dict(current, format=3, source=hashlib.sha256(json.dumps(entries_3).encode("utf-8")).hexdigest()),
        dict(current, format=3),
    ]
    assert len({old["source"] for old in old_caches}) == 3  # 지금의 지문에는 파일 전체의 sha256이 들어 있다
    for count, old in enumerate(old_caches, start=5):
        dataset._write_cached(old_steps, session_file, old)
        fresh = load_step_tables(small_tree.manifest, small_tree.root, pipeline_spec, [ARM_SESSION_BASELINE])
        assert len(builds) == count, old["format"]
        pd.testing.assert_frame_equal(fresh[ARM_SESSION_BASELINE].steps, first[ARM_SESSION_BASELINE].steps)
        assert cache_metadata(session_file) == current


def parquet_footer(path: Path) -> bytes:
    """parquet 파일의 꼬리와 마지막 8바이트(꼬리 길이 4바이트 리틀 엔디언 + "PAR1"). 캐시 형식 3이 해시하던 부분이다.

    꼬리에는 스키마, 파일 메타데이터, 행 수, 열마다의 위치·압축 크기·최솟값·최댓값이 있고 열 데이터 자체는 없다.
    """
    data = path.read_bytes()
    assert data[-4:] == b"PAR1"
    length = int.from_bytes(data[-8:-4], "little")
    assert length == pq.ParquetFile(path).metadata.serialized_size  # pyarrow가 말하는 꼬리 크기와 같다
    assert 0 < length < len(data) - 8  # 꼬리는 파일의 일부다
    return data[-(length + 8) :]


def noisy_session(spec: PipelineSpec) -> list[RawFrame]:
    """잡음이 있는 합성 세션(drowsy, 80초, 12 fps 안팎, 약 980프레임). 프레임마다 값이 다르다.

    각본 시나리오(`render`)의 프레임은 대부분 서로 같아서 두 프레임을 맞바꿔도 달라지는 것이 없다. 이 세션은 독립
    검증이 쓴 것과 같은 생성기(`synth.generate_session`)로 만든다.
    """
    profile = dataclasses.replace(synth.make_subject(1, 0), fps=12.0)
    return list(synth.generate_session(profile, 2, 80_000, 0, spec))


def exchanges_with_the_same_footer(
    write: Callable[[list[RawFrame]], Path], frames: list[RawFrame], start_ms: int = 62_000
) -> Iterator[list[RawFrame]]:
    """독립 검증의 재현(캐시 형식 3이 놓친 경우)에 쓰는 프레임 목록들을 차례로 낸다.

    7프레임 떨어진 두 프레임의 내용을 시각(`t_ms`)만 남기고 맞바꾼 목록 가운데, 파일로 썼을 때 크기와 parquet 꼬리가
    원본과 바이트 단위로 같고 파일 바이트는 다른 것만 낸다. `write(frames)`는 파일을 쓰고 그 경로를 돌려준다. 내놓는
    시점에 파일에는 그 목록이 쓰여 있다. start_ms 뒤의 프레임만 바꾼다(기준선 수집이 끝난 뒤라 스텝 특징이 달라진다).
    """
    path = write(frames)
    size, footer, data = path.stat().st_size, parquet_footer(path), path.read_bytes()
    first = next(index for index, frame in enumerate(frames) if frame.t_ms > start_ms)
    for index in range(first, len(frames) - 7, 5):
        swapped = list(frames)
        swapped[index] = dataclasses.replace(frames[index + 7], t_ms=frames[index].t_ms)
        swapped[index + 7] = dataclasses.replace(frames[index], t_ms=frames[index + 7].t_ms)
        path = write(swapped)
        other = path.read_bytes()
        if len(other) == size and other != data and parquet_footer(path) == footer:
            yield swapped


def test_raw_content_digest_is_the_sha256_of_the_whole_file(
    tmp_path: Path, small_tree: SimpleNamespace, pipeline_spec: PipelineSpec
) -> None:
    raw = small_tree.root / "01/0.parquet"
    data = raw.read_bytes()
    assert raw_content_digest(raw) == hashlib.sha256(data).hexdigest()  # 따로 계산한 값과 같다
    assert raw_content_digest(raw) != hashlib.sha256(parquet_footer(raw)).hexdigest()  # 꼬리만의 해시가 아니다

    # 같은 프레임을 다시 쓰면 같고, 프레임 값 하나가 달라지거나 파일 메타데이터만 달라져도 달라진다.
    # 경우마다 다른 경로에 쓴다(같은 경로에 같은 크기로 곧바로 다시 쓰면 수정 시각까지 같을 수 있다).
    frames = list(read_raw_frames(raw))
    write_raw_frames(frames, tmp_path / "copy.parquet", pipeline_spec, {"synthetic": True})
    assert (tmp_path / "copy.parquet").read_bytes() == data
    assert raw_content_digest(tmp_path / "copy.parquet") == raw_content_digest(raw)
    nudged = list(frames)
    nudged[5] = dataclasses.replace(frames[5], w=frames[5].w + 1)
    write_raw_frames(nudged, tmp_path / "nudged.parquet", pipeline_spec, {"synthetic": True})
    assert raw_content_digest(tmp_path / "nudged.parquet") != raw_content_digest(raw)
    write_raw_frames(frames, tmp_path / "noted.parquet", pipeline_spec, {"synthetic": True, "note": "x"})
    assert raw_content_digest(tmp_path / "noted.parquet") != raw_content_digest(raw)

    # 독립 검증이 찾은 경우: 두 프레임의 값을 맞바꾼 파일. 크기와 꼬리는 원본과 바이트 단위로 같아서 꼬리의 해시(형식 3)는
    # 같았다. 파일 전체의 해시는 다르다.
    noisy = noisy_session(pipeline_spec)
    assert 900 < len(noisy) < 1100 and noisy[-1].t_ms > 79_000

    def write(frames_to_write: list[RawFrame]) -> Path:
        target = tmp_path / f"noisy_{len(list(tmp_path.glob('noisy_*')))}.parquet"
        write_raw_frames(frames_to_write, target, pipeline_spec, {"synthetic": True})
        return target

    exchanged = next(exchanges_with_the_same_footer(write, noisy), None)
    assert exchanged is not None, "no exchange of two frames keeps the file size and the parquet footer"
    assert exchanged != noisy and [frame.t_ms for frame in exchanged] == [frame.t_ms for frame in noisy]
    original, changed = tmp_path / "noisy_0.parquet", write(exchanged)
    assert original.stat().st_size == changed.stat().st_size and original.read_bytes() != changed.read_bytes()
    assert hashlib.sha256(parquet_footer(original)).digest() == hashlib.sha256(parquet_footer(changed)).digest()
    assert raw_content_digest(original) != raw_content_digest(changed)
    assert raw_content_digest(changed) == hashlib.sha256(changed.read_bytes()).hexdigest()

    # parquet인지는 보지 않는다. 어떤 파일이든 파일 전체의 sha256이다(빈 파일, parquet처럼 끝나는 파일 포함)
    contents = (b"", b"short", b"this is not a parquet file", b"x" * 20 + (10_000).to_bytes(4, "little") + b"PAR1")
    for number, content in enumerate(contents):
        other = tmp_path / f"other_{number}.bin"
        other.write_bytes(content)
        assert raw_content_digest(other) == hashlib.sha256(content).hexdigest(), content
    for refresh in (False, True):
        with pytest.raises(OSError):
            raw_content_digest(tmp_path / "missing.parquet", refresh=refresh)

    # 지문 전체: 세션마다 [session_id, rel_path, 크기, 수정 시각, 파일 전체의 sha256], 없는 파일은 [.., -1, -1, ""]
    manifest = small_tree.manifest
    entries: list[list[Any]] = []
    for session_id, rel_path in zip(manifest["session_id"].tolist(), manifest["rel_path"].tolist()):
        raw_file = small_tree.root / rel_path
        stat = raw_file.stat()
        whole = hashlib.sha256(raw_file.read_bytes()).hexdigest()
        entries.append([session_id, rel_path, stat.st_size, stat.st_mtime_ns, whole])
    expected = hashlib.sha256(json.dumps(entries).encode("utf-8")).hexdigest()
    assert dataset._source_fingerprint(manifest, small_tree.root) == expected
    assert dataset._source_fingerprint(manifest, small_tree.root, refresh=True) == expected
    (small_tree.root / "02/5.parquet").unlink()
    entries[4] = [entries[4][0], entries[4][1], -1, -1, ""]
    assert manifest["rel_path"].iloc[4] == "02/5.parquet"
    expected = hashlib.sha256(json.dumps(entries).encode("utf-8")).hexdigest()
    assert dataset._source_fingerprint(manifest, small_tree.root) == expected


def test_content_digests_are_memoised_by_resolved_path_size_and_mtime(
    tmp_path: Path, small_tree: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """파일 전체의 해시는 프로세스 안에서 (심볼릭 링크를 푼 경로, 크기, 수정 시각 ns)별로 한 번만 계산한다.

    파일을 읽는 함수(`dataset._read_file_sha256`)가 불린 횟수를 센다.
    """
    manifest, root = small_tree.manifest, small_tree.root
    files = [root / rel_path for rel_path in manifest["rel_path"].tolist()]
    assert len(files) == 6
    reads: list[Path] = []
    real_read = dataset._read_file_sha256

    def counting_read(path: Path) -> tuple[str, tuple[int, int] | None]:
        reads.append(Path(path))
        return real_read(path)

    monkeypatch.setattr(dataset, "_read_file_sha256", counting_read)
    monkeypatch.setattr(dataset, "_CONTENT_DIGESTS", {})  # 새 프로세스처럼 빈 기억에서 시작한다

    first = dataset._source_fingerprint(manifest, root)
    assert reads == files  # 처음에는 파일마다 한 번 읽는다
    for _ in range(3):
        assert dataset._source_fingerprint(manifest, root) == first
    assert [raw_content_digest(path) for path in files] == [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    assert reads == files  # 그 뒤로는 읽지 않는다
    # 기억의 열쇠는 (푼 경로, 크기, 수정 시각 ns)다
    stats = [path.stat() for path in files]
    assert set(dataset._CONTENT_DIGESTS) == {
        (os.path.realpath(path), stat.st_size, stat.st_mtime_ns) for path, stat in zip(files, stats)
    }

    # 한 파일의 수정 시각이 1 ns 달라지면 그 파일만 다시 읽는다(내용이 같으므로 해시는 같고, 지문은 수정 시각 때문에 다르다)
    victim, stamp = files[2], stats[2]
    os.utime(victim, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1))
    touched = dataset._source_fingerprint(manifest, root)
    assert touched != first and reads == [*files, victim]
    assert dataset._source_fingerprint(manifest, root) == touched and len(reads) == 7

    # 같은 파일을 가리키는 다른 경로(폴더의 심볼릭 링크)는 기억을 함께 쓴다
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)
    assert dataset._source_fingerprint(manifest, link) == touched and len(reads) == 7
    # 크기와 수정 시각이 같은 사본(`cp -p`)이라도 경로가 다르면 따로 읽는다
    copy = tmp_path / "copy.parquet"
    shutil.copy2(victim, copy)
    assert (copy.stat().st_size, copy.stat().st_mtime_ns) == (stamp.st_size, stamp.st_mtime_ns + 1)
    assert raw_content_digest(copy) == raw_content_digest(victim) and reads[7:] == [copy]
    # 수정 시각을 되돌려도 크기가 달라졌으면 다시 읽는다(크기도 열쇠다)
    grown = tmp_path / "grown.parquet"
    shutil.copy2(victim, grown)
    assert raw_content_digest(grown) == raw_content_digest(victim) and reads[8:] == [grown]
    with grown.open("ab") as handle:
        handle.write(b"x")
    os.utime(grown, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1))
    assert (grown.stat().st_size, grown.stat().st_mtime_ns) == (stamp.st_size + 1, stamp.st_mtime_ns + 1)
    assert raw_content_digest(grown) == hashlib.sha256(victim.read_bytes() + b"x").hexdigest() and reads[9:] == [grown]
    del reads[7:]
    # refresh=True는 기억이 있어도 다시 읽는다
    assert raw_content_digest(victim, refresh=True) == raw_content_digest(victim) and reads[7:] == [victim]
    assert dataset._source_fingerprint(manifest, root, refresh=True) == touched and reads[8:] == files

    # 알아채지 못하는 경우(문서에 적은 한계): 같은 프로세스에서 크기와 수정 시각을 그대로 둔 채 내용을 바꾼 파일.
    # 기억해 둔 옛 값이 나온다. 기억을 비우면(다음 프로세스) 새 값이 나온다. refresh=True도 새 값을 낸다.
    data = victim.read_bytes()
    flipped = data[:100] + bytes([data[100] ^ 0xFF]) + data[101:]
    old_digest, new_digest = hashlib.sha256(data).hexdigest(), hashlib.sha256(flipped).hexdigest()
    victim.write_bytes(flipped)
    os.utime(victim, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1))
    count = len(reads)
    assert raw_content_digest(victim) == old_digest != new_digest and len(reads) == count
    kept = dict(dataset._CONTENT_DIGESTS)
    dataset._CONTENT_DIGESTS.clear()
    assert raw_content_digest(victim) == new_digest and len(reads) == count + 1
    dataset._CONTENT_DIGESTS.clear()
    dataset._CONTENT_DIGESTS.update(kept)  # 옛 값을 기억하던 상태로 되돌린다
    assert raw_content_digest(victim) == old_digest
    assert raw_content_digest(victim, refresh=True) == new_digest == raw_content_digest(victim)


def test_a_file_that_changes_while_it_is_read_is_not_memoised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """읽는 동안 크기나 수정 시각이 바뀐 파일의 해시는 기억하지 않는다(다음 호출이 다시 읽는다)."""
    target = tmp_path / "growing.bin"
    target.write_bytes(b"a" * 5000)
    monkeypatch.setattr(dataset, "_CONTENT_DIGESTS", {})
    real_sha256 = hashlib.sha256

    class GrowingHash:
        """첫 블록을 받을 때 읽고 있는 파일 끝에 한 바이트를 붙이는 sha256."""

        def __init__(self) -> None:
            self.inner = real_sha256()
            self.grown = False

        def update(self, block: bytes) -> None:
            if not self.grown:
                self.grown = True
                with target.open("ab") as handle:
                    handle.write(b"b")
            self.inner.update(block)

        def hexdigest(self) -> str:
            return self.inner.hexdigest()

    with monkeypatch.context() as patch:
        patch.setattr(dataset.hashlib, "sha256", GrowingHash)
        value, stamp = dataset._read_file_sha256(target)
    assert stamp is None and target.stat().st_size == 5001
    assert value == real_sha256(b"a" * 5000 + b"b").hexdigest()  # 읽은 바이트의 해시이긴 하다
    # 그대로인 파일은 (크기, 수정 시각)을 함께 돌려준다
    value, stamp = dataset._read_file_sha256(target)
    assert stamp == (5001, target.stat().st_mtime_ns) and value == real_sha256(target.read_bytes()).hexdigest()

    # `raw_content_digest`는 그런 값을 기억하지 않는다: 다음 호출이 파일을 다시 읽는다
    reads: list[Path] = []
    answers = iter([("0" * 64, None), ("1" * 64, (5001, target.stat().st_mtime_ns))])

    def scripted_read(path: Path) -> tuple[str, tuple[int, int] | None]:
        reads.append(Path(path))
        return next(answers)

    monkeypatch.setattr(dataset, "_read_file_sha256", scripted_read)
    assert raw_content_digest(target) == "0" * 64 and dataset._CONTENT_DIGESTS == {}
    assert raw_content_digest(target) == "1" * 64 and len(reads) == 2
    assert raw_content_digest(target) == "1" * 64 and len(reads) == 2  # 이번에는 기억했다
    assert list(dataset._CONTENT_DIGESTS) == [(os.path.realpath(target), 5001, target.stat().st_mtime_ns)]


FINGERPRINT_SCRIPT = (
    "import sys\n"
    "from pathlib import Path\n"
    "from drowsy import dataset\n"
    "root = Path(sys.argv[1])\n"
    "print(dataset._source_fingerprint(dataset.build_manifest(root), root))\n"
)


def fingerprint_in_another_process(root: Path, repo_root: Path) -> str:
    """새 파이썬 프로세스(다음 명령 실행과 같다)가 계산한 추출 결과 지문."""
    done = subprocess.run(
        [sys.executable, "-c", FINGERPRINT_SCRIPT, str(root)],
        cwd=repo_root, capture_output=True, text=True, check=False, timeout=300,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    return done.stdout.strip().splitlines()[-1]


def test_cache_is_rebuilt_when_two_frames_are_exchanged_with_the_same_size_footer_and_mtime(
    small_tree: SimpleNamespace, pipeline_spec: PipelineSpec, monkeypatch: pytest.MonkeyPatch, repo_root: Path
) -> None:
    """독립 검증의 재현: 추출 결과 하나에서 두 프레임의 값을 맞바꾸고(파일 크기와 parquet 꼬리가 그대로다) 수정 시각을
    되돌린다(`cp -p`, `rsync -t`로 덮어쓴 것과 같다).

    지문이 크기와 수정 시각뿐이던 때(형식 2)에도, 꼬리의 sha256을 더한 때(형식 3)에도 캐시가 다시 만들어지지 않아 옛
    내용의 스텝 표가 그대로 읽혔다. 지금은 파일 전체의 sha256이 지문에 들어 있어 다음 프로세스가 다시 만든다. 같은
    프로세스 안에서는 (경로, 크기, 수정 시각)이 그대로인 파일을 다시 읽지 않으므로 알아채지 못한다(문서에 적은 한계.
    `rebuild=True`가 그때의 방법이다). 손대지 않은 캐시는 다음 프로세스에서도 다시 만들지 않는다.
    """
    real_build = dataset._build_step_tables
    builds = spy_on_builder(monkeypatch)
    monkeypatch.setattr(dataset, "_CONTENT_DIGESTS", {})  # 새 프로세스처럼 빈 기억에서 시작한다
    manifest, root, spec = small_tree.manifest, small_tree.root, pipeline_spec
    victim = root / "01/10.parquet"
    content_a = noisy_session(spec)

    def write(frames: list[RawFrame]) -> Path:
        write_raw_frames(frames, victim, spec, {"synthetic": True})
        return victim

    def load(rebuild: bool = False) -> StepTable:
        return load_step_tables(manifest, root, spec, [ARM_SESSION_BASELINE], rebuild=rebuild)[ARM_SESSION_BASELINE]

    def uncached() -> StepTable:
        return real_build(manifest, root, spec, (ARM_SESSION_BASELINE,), None, None, 1)[ARM_SESSION_BASELINE]

    def source_key() -> str:
        cache_file = step_cache_dir(root) / f"{ARM_SESSION_BASELINE}_{spec_pipeline_hash(spec)[:16]}.steps.parquet"
        return cache_metadata(cache_file)["source"]

    # 맞바꿀 두 프레임을 고른다: 크기와 꼬리가 그대로이고 스텝 표가 달라지는 첫 번째 것
    write(content_a)
    truth_a = uncached()
    candidates = exchanges_with_the_same_footer(write, content_a)  # 내놓을 때마다 파일에 그 목록이 쓰여 있다
    content_b = next((frames for frames in candidates if not uncached().steps.equals(truth_a.steps)), None)
    assert content_b is not None, "no exchange of two frames keeps size and footer and changes the step table"

    # 상태 A: 캐시를 만든다. 그대로 두면 이 프로세스도 다른 프로세스도 다시 만들지 않는다(지문이 같다).
    write(content_a)
    stamp, bytes_a, footer_a = victim.stat(), victim.read_bytes(), parquet_footer(victim)
    table_a = load()
    load()
    assert builds == [((ARM_SESSION_BASELINE,), None)]
    pd.testing.assert_frame_equal(table_a.steps, truth_a.steps)
    fingerprint_a = dataset._source_fingerprint(manifest, root)
    assert source_key() == fingerprint_a == fingerprint_in_another_process(root, repo_root)

    # 상태 B: 두 프레임을 맞바꾼 내용. 크기, 꼬리, (되돌린) 수정 시각이 A와 같고 파일 바이트와 스텝 표는 다르다.
    write(content_b)
    os.utime(victim, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    after = victim.stat()
    assert (after.st_size, after.st_mtime_ns) == (stamp.st_size, stamp.st_mtime_ns)
    assert parquet_footer(victim) == footer_a and victim.read_bytes() != bytes_a
    truth_b = uncached()
    assert not truth_b.steps.equals(table_a.steps)

    # 다음 프로세스는 알아챈다.
    fingerprint_b = fingerprint_in_another_process(root, repo_root)
    assert fingerprint_b != fingerprint_a
    # 같은 프로세스(A를 읽은 기억이 남아 있다)는 알아채지 못하고 옛 표를 읽는다: 문서에 적은 한계
    assert dataset._source_fingerprint(manifest, root) == fingerprint_a
    pd.testing.assert_frame_equal(load().steps, table_a.steps)
    assert len(builds) == 1
    # 기억을 비운 뒤(다음 프로세스와 같다)에는 다시 만들고, 표는 지금 디스크의 내용과 같다
    dataset._CONTENT_DIGESTS.clear()
    assert dataset._source_fingerprint(manifest, root) == fingerprint_b
    table_b = load()
    assert len(builds) == 2 and source_key() == fingerprint_b
    pd.testing.assert_frame_equal(table_b.steps, truth_b.steps)
    pd.testing.assert_frame_equal(table_b.quality, truth_b.quality)
    assert not table_b.steps.equals(table_a.steps)  # 옛 내용의 표가 아니다
    load()
    dataset._CONTENT_DIGESTS.clear()
    load()
    assert len(builds) == 2  # 손대지 않으면 캐시에서 읽는다(기억이 있든 없든)

    # 내용과 크기는 그대로 두고 수정 시각만 1 ns 바꿔도 다시 만든다(전과 같다).
    os.utime(victim, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1))
    pd.testing.assert_frame_equal(load().steps, truth_b.steps)
    assert len(builds) == 3

    # 같은 프로세스에서 내용을 A로 되돌리고(크기와 수정 시각은 그대로) rebuild=True로 부른다: 파일을 다시 읽으므로 표도
    # 캐시에 적는 지문도 지금의 내용과 맞고, 다음 프로세스가 그 캐시를 그대로 읽는다.
    write(content_a)
    os.utime(victim, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1))
    pd.testing.assert_frame_equal(load().steps, truth_b.steps)  # rebuild 없이는 옛 표(한계)
    assert len(builds) == 3
    rebuilt = load(rebuild=True)
    assert len(builds) == 4
    pd.testing.assert_frame_equal(rebuilt.steps, truth_a.steps)
    assert source_key() == fingerprint_in_another_process(root, repo_root)
    assert source_key() not in (fingerprint_a, fingerprint_b)  # 내용은 A지만 수정 시각이 1 ns 다르다
    dataset._CONTENT_DIGESTS.clear()
    pd.testing.assert_frame_equal(load().steps, truth_a.steps)
    assert len(builds) == 4


def test_rebuild_ignores_a_valid_cache(
    small_tree: SimpleNamespace, pipeline_spec: PipelineSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    builds = spy_on_builder(monkeypatch)
    cache = step_cache_dir(small_tree.root)
    first = load_step_tables(small_tree.manifest, small_tree.root, pipeline_spec, [ARM_GLOBAL, ARM_SESSION_BASELINE])
    steps_file = cache / f"{ARM_GLOBAL}_{spec_pipeline_hash(pipeline_spec)[:16]}.steps.parquet"
    doctored = first[ARM_GLOBAL].steps.copy()
    doctored["closure_mean_60"] = 0.5
    dataset._write_cached(doctored, steps_file, cache_metadata(steps_file))  # 열쇠가 맞는(유효한) 캐시
    cached = load_step_tables(small_tree.manifest, small_tree.root, pipeline_spec, [ARM_GLOBAL])
    assert len(builds) == 1 and (cached[ARM_GLOBAL].steps["closure_mean_60"] == 0.5).all()

    forced = load_step_tables(small_tree.manifest, small_tree.root, pipeline_spec, [ARM_GLOBAL], rebuild=True)
    assert builds[1:] == [((ARM_GLOBAL,), None)]  # 요청한 조건만 다시 만든다
    pd.testing.assert_frame_equal(forced[ARM_GLOBAL].steps, first[ARM_GLOBAL].steps)
    after = load_step_tables(small_tree.manifest, small_tree.root, pipeline_spec, [ARM_GLOBAL, ARM_SESSION_BASELINE])
    assert len(builds) == 2  # 다시 만든 것이 캐시에 덮어써졌고, 요청하지 않은 조건의 캐시는 그대로다
    pd.testing.assert_frame_equal(after[ARM_GLOBAL].steps, first[ARM_GLOBAL].steps)
    pd.testing.assert_frame_equal(after[ARM_GLOBAL].quality, first[ARM_GLOBAL].quality)


def test_each_global_override_has_its_own_cache(
    small_tree: SimpleNamespace, pipeline_spec: PipelineSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fold별 전역 상수(민감도 점검)는 fold마다 다른 기준선을 넣는다. 서로의 캐시를 덮어쓰거나 섞어 읽으면 안 된다."""
    builds = spy_on_builder(monkeypatch)
    manifest, root, spec = small_tree.manifest, small_tree.root, pipeline_spec
    cache = step_cache_dir(root)
    spec_global = dataset.calibrate.global_baseline(spec)
    wide = dataclasses.replace(spec_global, e_l=0.33, e_r=0.33)
    narrow = dataclasses.replace(spec_global, e_l=0.24, e_r=0.24)
    blink = dataclasses.replace(spec_global, b_l=0.10, b_r=0.10)  # EAR이 아닌 다른 필드만 다른 기준선
    overrides = {"wide": wide, "narrow": narrow, "blink": blink}
    direct = {
        name: build_step_table(manifest, root, spec, ARM_GLOBAL, override) for name, override in overrides.items()
    }
    direct["default"] = build_step_table(manifest, root, spec, ARM_GLOBAL)
    means = {name: float(table.steps["closure_mean_60"].mean()) for name, table in direct.items()}
    assert means["wide"] > means["default"] + 0.05  # 뜬 눈 기준을 키우면 같은 눈이 더 감긴 것으로 읽힌다
    names = list(direct)
    for i, a in enumerate(names):  # 네 표가 서로 다 다르다(그래야 섞어 읽은 것을 알아챌 수 있다)
        for b in names[i + 1 :]:
            assert not direct[a].steps.equals(direct[b].steps), (a, b)
    builds.clear()

    default = load_step_tables(manifest, root, spec, ARMS)
    assert builds == [(ARMS, None)]
    for name, override in overrides.items():
        loaded = load_step_tables(manifest, root, spec, ARMS, global_override=override)
        assert builds[-1] == ((ARM_GLOBAL,), None)  # 전역 조건만 새로 만든다
        pd.testing.assert_frame_equal(loaded[ARM_GLOBAL].steps, direct[name].steps)
        for arm in (ARM_ALERT_BASELINE, ARM_SESSION_BASELINE):  # 나머지 조건은 넣은 상수와 무관하다
            pd.testing.assert_frame_equal(loaded[arm].steps, default[arm].steps)
    assert len(builds) == 4

    # 파일 이름: 기준선 전체를 JSON으로 적은 것의 sha256 앞 12자가 붙는다.
    prefix = spec_pipeline_hash(spec)[:16]
    tags = {
        name: hashlib.sha256(json.dumps(dataclasses.asdict(override), sort_keys=True).encode("utf-8")).hexdigest()
        for name, override in overrides.items()
    }
    assert len(set(tags.values())) == 3
    global_files = sorted(path.name for path in cache.iterdir() if path.name.startswith("global_"))
    assert global_files == sorted(
        f"global_{prefix}{suffix}.{kind}.parquet"
        for suffix in [""] + [f"_g{tag[:12]}" for tag in tags.values()]
        for kind in ("steps", "quality")
    )
    for name, tag in tags.items():
        stored = cache_metadata(cache / f"global_{prefix}_g{tag[:12]}.steps.parquet")
        assert stored["global_override"] == tag and stored["arm"] == ARM_GLOBAL
    assert cache_metadata(cache / f"global_{prefix}.steps.parquet")["global_override"] is None
    assert cache_metadata(cache / f"alert_baseline_{prefix}.steps.parquet")["global_override"] is None

    # 다른 순서로 다시 읽어도 각자 자기 표가 나오고 새로 만들지 않는다.
    for name in ("narrow", "blink", "default", "wide", "narrow"):
        override = overrides.get(name)
        again = load_step_tables(manifest, root, spec, [ARM_GLOBAL], global_override=override)
        pd.testing.assert_frame_equal(again[ARM_GLOBAL].steps, direct[name].steps)
        pd.testing.assert_frame_equal(again[ARM_GLOBAL].quality, direct[name].quality)
    assert len(builds) == 4
    # 값이 spec의 전역 상수와 같은 기준선을 직접 넣으면 캐시는 따로지만 표는 기본 전역 조건과 같다.
    same = load_step_tables(manifest, root, spec, [ARM_GLOBAL], global_override=spec_global)
    assert len(builds) == 5
    pd.testing.assert_frame_equal(same[ARM_GLOBAL].steps, direct["default"].steps)


def test_parallel_build_matches_serial(small_tree: SimpleNamespace, pipeline_spec: PipelineSpec) -> None:
    serial = dataset._build_step_tables(small_tree.manifest, small_tree.root, pipeline_spec, ARMS, None, None, 1)
    parallel = dataset._build_step_tables(small_tree.manifest, small_tree.root, pipeline_spec, ARMS, None, None, 2)
    for arm in ARMS:
        pd.testing.assert_frame_equal(parallel[arm].steps, serial[arm].steps)
        pd.testing.assert_frame_equal(parallel[arm].quality, serial[arm].quality)
    with pytest.raises(ValueError, match="workers"):
        build_step_table(small_tree.manifest, small_tree.root, pipeline_spec, ARM_GLOBAL, workers=0)
    # 프레임 누락 변형도 workers와 무관하다(작업 프로세스마다 해시 무작위화가 달라도 같은 프레임이 버려진다).
    drop = FrameDropSpec(variant=1, seed=5, drop_min=0.10, drop_max=0.30)
    manifest, root = small_tree.manifest, small_tree.root
    serial = dataset._build_step_tables(manifest, root, pipeline_spec, ARMS, None, None, 1, drop)
    parallel = dataset._build_step_tables(manifest, root, pipeline_spec, ARMS, None, None, 2, drop)
    for arm in ARMS:
        pd.testing.assert_frame_equal(parallel[arm].steps, serial[arm].steps)
        pd.testing.assert_frame_equal(parallel[arm].quality, serial[arm].quality)
        assert parallel[arm].frame_drop == drop


# ---------------------------------------------------------------------------
# 프레임 누락 변형 (PLAN 5.2절 증강 (2))
# ---------------------------------------------------------------------------


def mask_text(keep: list[bool]) -> str:
    return "".join("1" if kept else "0" for kept in keep)


def test_frame_drop_plan_is_seeded_from_seed_variant_and_session() -> None:
    base = FrameDropSpec(variant=0, seed=0, drop_min=0.10, drop_max=0.30)
    # 기대값은 dataset 모듈을 쓰지 않는 스크립트(hashlib + random.Random)로 따로 계산해 적은 것이다.
    golden = [
        ("S01_0", base, 0.10031072089011894, "111111111111"),
        ("S01_0", dataclasses.replace(base, variant=1), 0.1992230134099841, "101110011011"),
        ("S01_0", dataclasses.replace(base, seed=1), 0.16556905943513903, "111110111111"),
        ("S32_10_2", base, 0.23382877248201078, "111101111111"),
        ("S01_0", dataclasses.replace(base, drop_min=0.5, drop_max=0.5), 0.5, "101010101111"),
    ]
    for session_id, drop, rate, mask in golden:
        got_rate, keep = frame_drop_plan(session_id, 12, drop)
        assert got_rate == rate and mask_text(keep) == mask, (session_id, drop)
        assert mask_text(frame_drop_plan(session_id, 5, drop)[1]) == mask[:5]  # 프레임 수와 무관한 같은 수열

    # 첫 프레임은 비율이 아무리 커도 남는다. 비율은 언제나 범위 안이고 범위 전체에 퍼진다.
    harsh = FrameDropSpec(variant=0, seed=0, drop_min=0.999, drop_max=0.999)
    rates: list[float] = []
    for subject in range(1, 201):
        session_id = f"S{subject:02d}_{(0, 5, 10)[subject % 3]}"
        keep = frame_drop_plan(session_id, 50, harsh)[1]
        assert keep[0] is True and sum(keep) <= 3
        rate, keep = frame_drop_plan(session_id, 50, base)
        assert 0.10 <= rate <= 0.30 and keep[0] is True and len(keep) == 50
        assert (rate, keep) == reference_plan(session_id, 50, base)
        rates.append(rate)
    assert len(set(rates)) == 200  # 세션마다 다른 비율
    assert min(rates) < 0.12 and max(rates) > 0.28
    assert abs(float(np.mean(rates)) - 0.20) < 0.021  # 균등 분포 평균 0.2, 표준오차 0.0041의 5배
    # 프레임마다 독립으로 버린다: 비율 0.2로 20,000개 → 4,000 ± 5·56.6.
    rate, keep = frame_drop_plan("S07_5", 20_001, FrameDropSpec(variant=2, seed=9, drop_min=0.2, drop_max=0.2))
    assert rate == 0.2 and abs((20_001 - sum(keep)) - 4_000) < 283
    pairs = sum(1 for a, b in zip(keep[1:], keep[2:]) if not a and not b)  # 연속 누락은 0.2² = 4%
    assert abs(pairs - 0.04 * 19_999) < 5 * (19_999 * 0.04 * 0.96) ** 0.5 * 1.5

    assert frame_drop_plan("S01_0", 0, base) == (0.10031072089011894, [])
    assert frame_drop_plan("S01_0", 1, base)[1] == [True]
    assert frame_drop_plan("S01_0", 30, FrameDropSpec(0, 0, 0.0, 0.0)) == (0.0, [True] * 30)
    for bad in (
        {"variant": -1}, {"variant": 1.0}, {"variant": True}, {"seed": 0.5}, {"drop_min": 0.4}, {"drop_max": 1.0},
        {"drop_min": -0.1},
    ):  # fmt: skip
        with pytest.raises(ValueError, match="frame-drop"):
            dataclasses.replace(base, **bad)
    with pytest.raises(ValueError, match="n_frames"):
        frame_drop_plan("S01_0", -1, base)


def test_frame_drop_builder_is_identical_in_other_processes(
    small_tree: SimpleNamespace, pipeline_spec: PipelineSpec, repo_root: Path
) -> None:
    """다른 프로세스(해시 무작위화 시드가 다르다)에서 만들어도 같은 프레임이 버려지고 같은 표가 나온다."""
    drop = FrameDropSpec(variant=2, seed=11, drop_min=0.10, drop_max=0.30)

    def digest(tables: dict[str, StepTable]) -> dict[str, object]:
        out: dict[str, object] = {}
        for arm, table in tables.items():
            values = np.ascontiguousarray(table.steps[FEATURES].to_numpy(dtype=np.float64))
            out[arm] = {
                "features": hashlib.sha256(values.tobytes()).hexdigest(),
                "k": table.steps["k"].tolist(),
                "rates": [repr(float(rate)) for rate in table.quality["drop_rate"]],
                "thinned": table.quality["n_thinned"].tolist(),
            }
        return out

    here = dataset._build_step_tables(small_tree.manifest, small_tree.root, pipeline_spec, ARMS, None, None, 1, drop)
    expected = json.loads(json.dumps(digest(here)))
    plan_here = [repr(frame_drop_plan(f"S{n:02d}_10_1", 40, drop)) for n in range(1, 6)]
    script = (
        "import hashlib, json, sys\n"
        "import numpy as np\n"
        "from drowsy import dataset\n"
        "from drowsy.config import ARMS\n"
        "from drowsy.spec import load_feature_spec\n"
        "spec = load_feature_spec('config/feature_spec.json').pipeline\n"
        "drop = dataset.FrameDropSpec(variant=2, seed=11, drop_min=0.10, drop_max=0.30)\n"
        "from pathlib import Path\n"
        "root = Path(sys.argv[1])\n"
        "manifest = dataset.build_manifest(root)\n"
        "tables = dataset._build_step_tables(manifest, root, spec, ARMS, None, None, 1, drop)\n"
        "out = {}\n"
        "for arm, table in tables.items():\n"
        "    values = np.ascontiguousarray(table.steps[list(dataset.FEATURE_NAMES)].to_numpy(dtype=np.float64))\n"
        "    out[arm] = {'features': hashlib.sha256(values.tobytes()).hexdigest(), 'k': table.steps['k'].tolist(),\n"
        "                'rates': [repr(float(rate)) for rate in table.quality['drop_rate']],\n"
        "                'thinned': table.quality['n_thinned'].tolist()}\n"
        "plan = [repr(dataset.frame_drop_plan(f'S{n:02d}_10_1', 40, drop)) for n in range(1, 6)]\n"
        "print(json.dumps({'tables': out, 'plan': plan, 'hash_of_str': hash('S01_0')}))\n"
    )
    seen_hashes = set()
    for hash_seed in ("1", "2024"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        done = subprocess.run(
            [sys.executable, "-c", script, str(small_tree.root)],
            cwd=repo_root, env=env, capture_output=True, text=True, check=False, timeout=300,
        )  # fmt: skip
        assert done.returncode == 0, done.stderr
        other = json.loads(done.stdout.strip().splitlines()[-1])
        assert other["tables"] == expected and other["plan"] == plan_here
        seen_hashes.add(other["hash_of_str"])
    assert len(seen_hashes) == 2  # 두 프로세스의 문자열 해시는 실제로 달랐다
    assert all(rate != "nan" for rate in expected[ARM_GLOBAL]["rates"]) and sum(expected[ARM_GLOBAL]["thinned"]) > 0


def test_frame_drop_tables_differ_from_the_original_only_in_the_feature_frames(
    tiny: SimpleNamespace, tiny_thinned: dict[str, StepTable], pipeline_spec: PipelineSpec, rules_config: RulesConfig
) -> None:
    drop = TINY_DROP
    sessions = tiny.manifest["session_id"].tolist()
    baseline_columns = [
        "raw_status", "baseline_status", "t_b_ms", "base_e_l", "base_e_r", "base_b_l", "base_b_r", "base_jaw0",
        "calib_face_frames", "calib_face_ratio", "baseline_used", "alert_fallback",
    ]  # fmt: skip
    for arm in ARMS:
        plain, thinned = tiny.tables[arm], tiny_thinned[arm]
        assert thinned.arm == arm and thinned.frame_drop == drop and plain.frame_drop is None
        assert thinned.pipeline_hash == plain.pipeline_hash
        assert list(thinned.steps.columns) == list(STEP_COLUMNS)
        quality = thinned.quality.set_index("session_id")
        original = plain.quality.set_index("session_id")
        assert list(quality.index) == sessions
        assert [name for name in thinned.quality.columns if name not in plain.quality.columns] == list(
            FRAME_DROP_QUALITY_COLUMNS
        )
        # 기준선과 T_b는 솎지 않은 원본에서 온 그대로다.
        pd.testing.assert_frame_equal(quality[baseline_columns], original[baseline_columns])

        for session_id in sessions:
            source_frames = int(original.loc[session_id, "n_frames"] + original.loc[session_id, "n_dropped"])
            rate, keep = reference_plan(session_id, source_frames, drop)
            row = quality.loc[session_id]
            assert 0.10 <= row["drop_rate"] <= 0.30 and row["drop_rate"] == rate
            assert row["n_source_frames"] == source_frames and row["n_thinned"] == keep.count(False)
            assert row["n_frames"] + row["n_dropped"] == source_frames - row["n_thinned"]
            assert row["n_thinned"] > 0.05 * source_frames  # 실제로 프레임이 빠졌다
        # 10 fps에서는 두 장이 연달아 빠지면 200 ms를 넘는 빈 시간이 생겨 얼굴 시간이 준다. 계수는 솎은 재생의 값이다.
        assert quality["face_ms"].sum() < original["face_ms"].sum() - 10_000
        assert (quality["n_frames"] < original["n_frames"]).all()

        # 첫 프레임이 남아 세션 원점이 그대로다: 스텝 격자(t_end_ms = 1 + 1000·k)와 첫 스텝(T_b 직후)이 원본과 같다.
        assert ((thinned.steps["t_end_ms"] - 1000 * thinned.steps["k"]) == 1).all()
        first_k = thinned.steps.groupby("session_id")["k"].min().to_dict()
        assert first_k == plain.steps.groupby("session_id")["k"].min().to_dict()
        # 같은 (세션, 스텝)의 특징은 원본과 다르다. 세션마다 다르다.
        merged = thinned.steps.merge(plain.steps, on=["session_id", "k"], suffixes=("", "_plain"))
        assert len(merged) >= len(plain.steps) - len(sessions)  # 끝 스텝 하나쯤은 빠질 수 있다
        for session_id, group in merged.groupby("session_id"):
            a = group[FEATURES].to_numpy()
            b = group[[f"{name}_plain" for name in FEATURES]].to_numpy()
            assert not np.array_equal(a, b), session_id
        assert (merged["face_missing_ratio_60"] >= merged["face_missing_ratio_60_plain"] - 1e-12).all()
        assert merged["face_missing_ratio_60"].mean() > merged["face_missing_ratio_60_plain"].mean() + 0.005

    # 세 조건은 변형 안에서도 같은 스텝을 낸다.
    keys = list(ID_COLUMNS) + ["k", "t_end_ms", "full_lookback"]
    for arm in (ARM_ALERT_BASELINE, ARM_GLOBAL):
        pd.testing.assert_frame_equal(tiny_thinned[arm].steps[keys], tiny_thinned[ARM_SESSION_BASELINE].steps[keys])

    # 값은 "설명대로 따로 솎은 프레임을, 솎지 않은 프레임으로 구한 기준선과 T_b로 재생한 것"과 같다.
    def thinned_frames(rel_path: str, session_id: str) -> list:
        frames = list(read_raw_frames(tiny.root / rel_path))
        keep = reference_plan(session_id, len(frames), drop)[1]
        return [frame for frame, kept in zip(frames, keep) if kept]

    def values_of(result) -> np.ndarray:
        return np.array([step.values for step in result.steps])

    own_alert = replay_video(read_raw_frames(tiny.root / "Fold1_part1/01/0.parquet"), pipeline_spec, None, None)
    own_drowsy = replay_video(read_raw_frames(tiny.root / "Fold1_part1/01/10.parquet"), pipeline_spec, None, None)
    assert own_alert.baseline is not None and own_drowsy.baseline is not None
    assert own_alert.baseline != own_drowsy.baseline and own_drowsy.baseline.t_complete_ms == 60_000
    frames = thinned_frames("Fold1_part1/01/10.parquet", "S01_10")
    for arm, baseline in (
        (ARM_ALERT_BASELINE, own_alert.baseline),
        (ARM_SESSION_BASELINE, own_drowsy.baseline),
        (ARM_GLOBAL, dataset.calibrate.global_baseline(pipeline_spec)),
    ):
        expected = replay_video(frames, pipeline_spec, baseline, 60_000, rules=rules_config)
        np.testing.assert_array_equal(step_values(tiny_thinned[arm], "S01_10"), values_of(expected))
    # alert 영상 자신도 솎지 않은 프레임으로 구한 자기 기준선으로 계산한다(자기 기준선 조건과 같다).
    alert_frames = thinned_frames("Fold1_part1/01/0.parquet", "S01_0")
    expected = replay_video(alert_frames, pipeline_spec, own_alert.baseline, 60_000, rules=rules_config)
    for arm in (ARM_ALERT_BASELINE, ARM_SESSION_BASELINE):
        np.testing.assert_array_equal(step_values(tiny_thinned[arm], "S01_0"), values_of(expected))
    # 솎은 프레임으로 다시 보정했다면 다른 값이 나왔을 것이다(이 검사가 구별하는 대상).
    recalibrated = replay_video(alert_frames, pipeline_spec, None, None)
    assert recalibrated.diagnostics is not None and recalibrated.diagnostics.n_face_frames < 600
    # alert 기준선을 쓸 수 없는 3번 피험자: 변형에서도 전역 상수로 대체되고, T_b는 그 세션 자신의 것(180초)이다.
    for session_id in ("S03_0", "S03_5", "S03_10"):
        np.testing.assert_array_equal(
            step_values(tiny_thinned[ARM_ALERT_BASELINE], session_id), step_values(tiny_thinned[ARM_GLOBAL], session_id)
        )
    assert tiny_thinned[ARM_GLOBAL].steps.loc[tiny_thinned[ARM_GLOBAL].steps["session_id"] == "S03_0", "k"].min() == 181
    # 창도 그대로 만들 수 있다(학습에서 원본 창과 섞어 쓴다).
    windows = make_windows(tiny_thinned[ARM_ALERT_BASELINE].steps, 120, 5)
    assert len(windows) >= 20
    assert_windows_are_whole(windows, tiny_thinned[ARM_ALERT_BASELINE].steps, 120)


def test_frame_drop_with_zero_rate_reproduces_the_original_table(
    small_tree: SimpleNamespace, tiny: SimpleNamespace, pipeline_spec: PipelineSpec
) -> None:
    """비율 0이면 아무것도 버리지 않는다. 기준선을 넣어 재생하는 경로가 원본 표와 비트 단위로 같아야 한다."""
    # 보정이 전역 상수로 대체된 alert 영상(T_b = 180초)을 가진 피험자도 마찬가지다.
    third = tiny.manifest[tiny.manifest["subject"] == 3].reset_index(drop=True)
    no_drop = FrameDropSpec(0, 0, 0.0, 0.0)
    unthinned = dataset._build_step_tables(third, tiny.root, pipeline_spec, ARMS, None, None, 1, no_drop)
    for arm in ARMS:
        expected = tiny.tables[arm].steps
        expected = expected[expected["subject"] == 3].reset_index(drop=True)
        assert len(expected) > 300
        pd.testing.assert_frame_equal(unthinned[arm].steps, expected)

    (small_tree.root / "02/5.parquet").unlink()  # 추출 결과가 없는 세션
    write_session(small_tree.root, "02/0.parquet", Scenario((), 40_000, fps=FPS), pipeline_spec)  # 보정 미완
    manifest, root = small_tree.manifest, small_tree.root
    plain = dataset._build_step_tables(manifest, root, pipeline_spec, ARMS, None, None, 1)
    zero = dataset._build_step_tables(manifest, root, pipeline_spec, ARMS, None, None, 1, FrameDropSpec(3, 7, 0.0, 0.0))
    some = dataset._build_step_tables(manifest, root, pipeline_spec, ARMS, None, None, 1, FrameDropSpec(3, 7, 0.2, 0.2))
    for arm in ARMS:
        pd.testing.assert_frame_equal(zero[arm].steps, plain[arm].steps)
        pd.testing.assert_frame_equal(
            zero[arm].quality.drop(columns=list(FRAME_DROP_QUALITY_COLUMNS)), plain[arm].quality
        )
        assert not some[arm].steps[FEATURES].equals(plain[arm].steps[FEATURES])
        for table, expected_rate in ((zero[arm], 0.0), (some[arm], 0.2)):
            quality = table.quality.set_index("session_id")
            # 스텝이 없는 세션은 솎지 않는다: 비율 NaN, 솎은 수 0.
            assert np.isnan(quality.loc["S02_5", "drop_rate"]) and np.isnan(quality.loc["S02_0", "drop_rate"])
            assert quality.loc[["S02_5", "S02_0"], "n_source_frames"].tolist() == [0, 400]
            assert quality.loc[["S02_5", "S02_0"], "n_thinned"].tolist() == [0, 0]
            assert quality.loc["S02_0", "n_frames"] == 400 and quality.loc["S02_5", "raw_status"] == "missing"
            others = quality.drop(index=["S02_5", "S02_0"])
            assert (others["drop_rate"] == expected_rate).all() and (others["n_source_frames"] == 700).all()
            assert (others["n_thinned"] > 100).all() if expected_rate else (others["n_thinned"] == 0).all()
    assert some[ARM_ALERT_BASELINE].quality.set_index("session_id")["alert_fallback"].to_dict() == {
        "S01_0": False, "S01_5": False, "S01_10": False, "S02_0": True, "S02_5": True, "S02_10": True,
    }  # fmt: skip
    # 캐시 없이 표 하나만 만드는 공개 함수로도 같은 변형이 나온다.
    single = build_step_table(manifest, root, pipeline_spec, ARM_GLOBAL, frame_drop=FrameDropSpec(3, 7, 0.2, 0.2))
    pd.testing.assert_frame_equal(single.steps, some[ARM_GLOBAL].steps)
    pd.testing.assert_frame_equal(single.quality, some[ARM_GLOBAL].quality)
    assert single.frame_drop == FrameDropSpec(3, 7, 0.2, 0.2)


def test_frame_drop_cache_key_holds_variant_seed_and_rate_range(
    small_tree: SimpleNamespace, pipeline_spec: PipelineSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    builds = spy_on_builder(monkeypatch)
    manifest, root, spec = small_tree.manifest, small_tree.root, pipeline_spec
    cache = step_cache_dir(root)
    arms = [ARM_ALERT_BASELINE, ARM_SESSION_BASELINE]
    settings = {"variant": 0, "seed": 0, "drop_min": 0.10, "drop_max": 0.30}
    drop = FrameDropSpec(**settings)

    first = load_frame_drop_tables(manifest, root, spec, arms, **settings)
    assert builds == [(tuple(arms), drop)] and list(first) == arms
    prefix = spec_pipeline_hash(spec)[:16]
    assert sorted(path.name for path in cache.iterdir()) == sorted(
        f"{arm}_{prefix}_fd0_s0_r0.1-0.3.{kind}.parquet" for arm in arms for kind in ("steps", "quality")
    )  # 원본 표의 캐시는 건드리지 않는다
    stored = cache_metadata(cache / f"{ARM_ALERT_BASELINE}_{prefix}_fd0_s0_r0.1-0.3.steps.parquet")
    assert stored["frame_drop"] == {"method": "bernoulli-v1", "variant": 0, "seed": 0, "drop_min": 0.1, "drop_max": 0.3}
    assert stored["code"] == code_fingerprint() and stored["arm"] == ARM_ALERT_BASELINE and stored["format"] == 4

    # 캐시 적중: 다시 만들지 않고 값, 자료형, 설정이 같다.
    again = load_frame_drop_tables(manifest, root, spec, arms, **settings)
    assert len(builds) == 1
    for arm in arms:
        pd.testing.assert_frame_equal(again[arm].steps, first[arm].steps)
        pd.testing.assert_frame_equal(again[arm].quality, first[arm].quality)
        assert again[arm].frame_drop == drop and again[arm].arm == arm
    direct = dataset._build_step_tables(manifest, root, spec, arms, None, None, 1, drop)
    del builds[1:]  # 방금 직접 부른 것은 세지 않는다
    pd.testing.assert_frame_equal(again[ARM_ALERT_BASELINE].steps, direct[ARM_ALERT_BASELINE].steps)

    # 시드, 변형 번호, 비율 범위 가운데 하나만 달라도 캐시가 빗나가고 다른 표가 나온다.
    tables = {"base": first}
    for name, change in (
        ("seed", {"seed": 1}), ("variant", {"variant": 1}), ("max", {"drop_max": 0.25}), ("min", {"drop_min": 0.15}),
    ):  # fmt: skip
        changed = {**settings, **change}
        tables[name] = load_frame_drop_tables(manifest, root, spec, arms, **changed)
        assert builds[-1] == (tuple(arms), FrameDropSpec(**changed)), name
        load_frame_drop_tables(manifest, root, spec, arms, **changed)
    assert len(builds) == 5
    names = list(tables)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            table_a, table_b = tables[a][ARM_ALERT_BASELINE], tables[b][ARM_ALERT_BASELINE]
            assert not table_a.steps[FEATURES].equals(table_b.steps[FEATURES]), (a, b)
            assert table_a.quality["drop_rate"].tolist() != table_b.quality["drop_rate"].tolist(), (a, b)
    assert len(list(cache.iterdir())) == 5 * 4
    assert (cache / f"{ARM_SESSION_BASELINE}_{prefix}_fd1_s0_r0.1-0.3.quality.parquet").is_file()
    assert (cache / f"{ARM_SESSION_BASELINE}_{prefix}_fd0_s1_r0.1-0.3.quality.parquet").is_file()
    assert (cache / f"{ARM_SESSION_BASELINE}_{prefix}_fd0_s0_r0.15-0.3.quality.parquet").is_file()

    # 조건을 더 요청하면 없는 조건만 만든다. 원본 표는 따로 캐시된다.
    load_frame_drop_tables(manifest, root, spec, ARMS, **settings)
    assert builds[-1] == ((ARM_GLOBAL,), drop) and len(builds) == 6
    plain = load_step_tables(manifest, root, spec, arms)
    assert builds[-1] == (tuple(arms), None) and len(builds) == 7
    assert not plain[ARM_ALERT_BASELINE].steps[FEATURES].equals(first[ARM_ALERT_BASELINE].steps[FEATURES])
    load_frame_drop_tables(manifest, root, spec, arms, **settings)
    load_step_tables(manifest, root, spec, arms)
    assert len(builds) == 7

    # 추출 결과가 바뀌거나, 코드 지문이 바뀌거나, rebuild를 주면 변형도 다시 만든다.
    forced = load_frame_drop_tables(manifest, root, spec, arms, rebuild=True, **settings)
    assert builds[-1] == (tuple(arms), drop) and len(builds) == 8
    pd.testing.assert_frame_equal(forced[ARM_ALERT_BASELINE].steps, first[ARM_ALERT_BASELINE].steps)
    monkeypatch.setattr(dataset, "code_fingerprint", lambda files=None: "0" * 64)
    load_frame_drop_tables(manifest, root, spec, arms, **settings)
    assert builds[-1] == (tuple(arms), drop) and len(builds) == 9
    load_frame_drop_tables(manifest, root, spec, arms, **settings)
    assert len(builds) == 9
    monkeypatch.setattr(dataset, "code_fingerprint", code_fingerprint)
    load_frame_drop_tables(manifest, root, spec, arms, **settings)
    assert len(builds) == 10
    # 정수이기만 하면 넘파이 정수로 줘도 같은 열쇠다.
    load_frame_drop_tables(
        manifest, root, spec, arms, variant=np.int64(0), seed=np.int64(0), drop_min=0.10, drop_max=0.30
    )
    assert len(builds) == 10
    write_session(root, "01/10.parquet", Scenario(closures(900, [63, 66]), 70_000, fps=FPS), spec)
    rebuilt = load_frame_drop_tables(manifest, root, spec, arms, **settings)
    assert len(builds) == 11
    assert not rebuilt[ARM_ALERT_BASELINE].steps[FEATURES].equals(first[ARM_ALERT_BASELINE].steps[FEATURES])

    elsewhere = root.parent / "my_cache"
    load_frame_drop_tables(manifest, root, spec, [ARM_GLOBAL], cache_dir=elsewhere, **settings)
    assert sorted(path.name for path in elsewhere.iterdir()) == [
        f"global_{prefix}_fd0_s0_r0.1-0.3.quality.parquet", f"global_{prefix}_fd0_s0_r0.1-0.3.steps.parquet",
    ]  # fmt: skip
    with pytest.raises(ValueError, match="arm must be one of"):
        load_frame_drop_tables(manifest, root, spec, ["nope"], **settings)
    with pytest.raises(ValueError, match="frame-drop rates"):
        load_frame_drop_tables(manifest, root, spec, arms, variant=0, seed=0, drop_min=0.3, drop_max=0.1)
    with pytest.raises(TypeError):
        load_frame_drop_tables(manifest, root, spec, arms)  # variant, seed, 범위는 꼭 줘야 한다


# ---------------------------------------------------------------------------
# 설정에서 데이터 찾기, 명령
# ---------------------------------------------------------------------------


def write_config(
    tmp_path: Path,
    config_path: Path,
    name: str,
    extraction: dict | None = None,
    training: dict | None = None,
    **paths: str | None,
) -> Path:
    """기본 설정의 사본. `paths.rldd_root`는 주지 않으면 null로 적는다(기본 설정 파일의 값에 기대지 않는다)."""
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["paths"].update({"rldd_root": None, **paths})
    data["extraction"].update(extraction or {})
    data["training"].update(training or {})
    out = tmp_path / name
    out.write_text(yaml.safe_dump(data), encoding="utf-8")
    return out


def test_open_dataset_synthetic_and_real(tmp_path: Path, config_path: Path, pipeline_spec: PipelineSpec) -> None:
    synth_dir = tmp_path / "data" / "synth"
    synth.write_dataset(synth_dir, 1, 0.1, 0, pipeline_spec)
    cfg = load_config(write_config(tmp_path, config_path, "synth.yaml", synth_dir=str(synth_dir)))
    source = open_dataset(cfg)
    assert source.synthetic is True and source.raw_dir == synth_dir and len(source.manifest) == 4
    assert source.description.startswith("SYNTHETIC data") and "not a performance result" in source.description
    # 누락 모사 합성 데이터는 옆 폴더(`synth_fps15`)에서 찾는다. 없으면 만드는 명령을 알려 준다.
    with pytest.raises(ManifestError, match=r"--out .*synth_fps15 --subjects 60 --fps-sim 15"):
        open_dataset(cfg, fps_sim=True)
    synth.write_dataset(tmp_path / "data" / "synth_fps15", 1, 0.1, 0, pipeline_spec, drop_sim=cfg.extraction.drop_sim)
    thinned = open_dataset(cfg, fps_sim=True)
    assert thinned.synthetic is True and thinned.raw_dir == tmp_path / "data" / "synth_fps15"
    assert thinned.description.startswith("SYNTHETIC data") and "frame drops simulated" in thinned.description
    assert thinned.manifest["rel_path"].tolist() == source.manifest["rel_path"].tolist()

    missing = load_config(write_config(tmp_path, config_path, "missing.yaml", synth_dir=str(tmp_path / "nowhere")))
    with pytest.raises(ManifestError, match="python -m drowsy.synth"):
        open_dataset(missing)

    # 합성 표시가 없는 parquet 폴더를 합성 데이터로 쓰지 않는다.
    unmarked = tmp_path / "unmarked"
    for label in (0, 5, 10):
        frames = render(Scenario((), 1_000, fps=FPS), pipeline_spec)
        write_raw_frames(frames, unmarked / f"01/{label}.parquet", pipeline_spec)
    with pytest.raises(ManifestError, match="not marked synthetic"):
        open_dataset(load_config(write_config(tmp_path, config_path, "unmarked.yaml", synth_dir=str(unmarked))))

    # rldd_root를 넣으면 같은 호출이 실제 데이터를 가리킨다.
    videos = touch_tree(tmp_path / "rldd", rldd_like_names(12))
    real = load_config(
        write_config(tmp_path, config_path, "real.yaml", rldd_root=str(videos), raw_dir=str(tmp_path / "raw"))
    )
    source = open_dataset(real)
    assert source.synthetic is False and source.raw_dir == tmp_path / "raw" / "native" and len(source.manifest) == 36
    assert "UTA-RLDD" in source.description and "SYNTHETIC" not in source.description
    assert open_dataset(real, fps_sim=True).raw_dir == tmp_path / "raw" / "fps15"


def test_real_data_branch_refuses_raw_files_marked_synthetic(
    tmp_path: Path, config_path: Path, pipeline_spec: PipelineSpec
) -> None:
    """합성 세션이 실제 데이터(REAL, UTA-RLDD)로 표시되면 합성 점수가 성능처럼 보고된다. 그 길을 막는다."""

    def write_raw(path: Path, metadata: dict | None) -> None:
        write_raw_frames(render(Scenario((), 1_000, fps=FPS), pipeline_spec), path, pipeline_spec, metadata)

    # (1) rldd_root와 raw_dir/native가 모두 합성 parquet 폴더를 가리키는 설정.
    synth_dir = tmp_path / "data" / "synth"
    synth.write_dataset(synth_dir, 1, 0.1, 0, pipeline_spec)
    as_real = load_config(
        write_config(
            tmp_path, config_path, "synth_as_real.yaml", extraction={"native_subdir": "synth"},
            rldd_root=str(synth_dir), raw_dir=str(synth_dir.parent),
        )  # fmt: skip
    )
    with pytest.raises(ManifestError, match="marked synthetic") as refused:
        open_dataset(as_real)
    assert "4 raw parquet file(s)" in str(refused.value)  # 같은 파일을 두 번 세지 않는다
    assert str(synth_dir / build_manifest(synth_dir)["rel_path"].iloc[0]) in str(refused.value)
    # (2) rldd_root만 합성 parquet 폴더이고 raw_dir/native는 아직 없을 때도 그 폴더의 파일을 보고 거부한다.
    root_only = load_config(
        write_config(tmp_path, config_path, "root_only.yaml", rldd_root=str(synth_dir), raw_dir=str(tmp_path / "none"))
    )
    with pytest.raises(ManifestError, match="marked synthetic"):
        open_dataset(root_only)

    # (3) 영상 폴더 + 추출 결과 폴더. 표시가 없는(실제 추출) 파일은 받는다. 없는 파일은 건너뛴다.
    videos = touch_tree(tmp_path / "rldd", ["01/0.mp4", "01/5.mp4", "01/10_1.mp4", "01/10_2.mp4"])
    raw = tmp_path / "raw"
    write_raw(raw / "native/01/0.parquet", None)
    write_raw(raw / "native/01/5.parquet", {"synthetic": False})
    write_raw(raw / "native/01/10_1.parquet", {"source": "phone"})
    real = load_config(write_config(tmp_path, config_path, "real.yaml", rldd_root=str(videos), raw_dir=str(raw)))
    source = open_dataset(real)
    assert source.synthetic is False and "UTA-RLDD" in source.description and len(source.manifest) == 4
    # 누락 모사 추출 폴더는 그 폴더를 쓸 때만 본다.
    write_raw(raw / "fps15/01/5.parquet", {"synthetic": True})
    assert open_dataset(real).synthetic is False
    with pytest.raises(ManifestError, match=r"1 raw parquet file\(s\) are marked synthetic.*fps15/01/5.parquet"):
        open_dataset(real, fps_sim=True)
    # 합성 표시가 붙은 파일이 하나라도 섞여 있으면 거부하고 그 파일을 알려 준다.
    write_raw(raw / "native/01/10_2.parquet", {"synthetic": True})
    with pytest.raises(ManifestError, match=r"1 raw parquet file\(s\) are marked synthetic") as mixed:
        open_dataset(real)
    assert str(raw / "native/01/10_2.parquet") in str(mixed.value) and "10_1.parquet" not in str(mixed.value)
    assert "rldd_root: null" in str(mixed.value)
    # 추출 결과로 읽을 수 없는 파일도 여기서 알린다.
    (raw / "native/01/10_2.parquet").write_bytes(b"")
    with pytest.raises(ManifestError, match="not a readable extraction parquet"):
        open_dataset(real)
    write_raw(raw / "native/01/10_2.parquet", None)
    assert open_dataset(real).synthetic is False

    # (4) 표시가 없는 parquet 폴더를 rldd_root로 주는 것은 막지 않는다(실제 추출 결과만 복사해 둔 배치).
    copy_only = load_config(
        write_config(
            tmp_path, config_path, "copy_only.yaml", extraction={"native_subdir": "01_copy"},
            rldd_root=str(raw / "native"), raw_dir=str(tmp_path / "elsewhere"),
        )  # fmt: skip
    )
    source = open_dataset(copy_only)
    assert source.synthetic is False and len(source.manifest) == 4 and "SYNTHETIC" not in source.description


def test_cli_builds_and_reuses_the_cache(
    tmp_path: Path, config_path: Path, pipeline_spec: PipelineSpec, capsys: pytest.CaptureFixture[str]
) -> None:
    synth_dir = tmp_path / "data" / "synth"
    synth.write_dataset(synth_dir, 2, 1.2, 0, pipeline_spec)
    config = write_config(tmp_path, config_path, "synth.yaml", synth_dir=str(synth_dir))

    assert dataset.main(["--config", str(config), "--workers", "1"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("SYNTHETIC data")  # 합성 데이터로 만든 출력은 첫 줄에 그렇게 적는다
    assert lines[1] == "manifest: 2 subjects, 6 logical videos, 8 sessions, folds [1]"
    # 합성 데이터에는 회전 점검 기록이 없다. 그 사실을 한 줄로 적고, 확인할 영상의 목록은 없다
    assert lines[2] == (
        "rotation check of 8 raw file(s): 0 decided by the automatic check, 0 extracted with a manual override, "
        "0 ambiguous without a manual override, 0 with an override that was not applied, "
        "8 without a rotation-check record"
    )
    assert sum(line.startswith("rotation check of") for line in lines) == 1
    assert not any(line.startswith("  rotation") for line in lines) and lines[3].startswith("step tables for")
    assert sum(line.startswith("[alert_baseline]") for line in lines) == 2 and "[global] steps" in "\n".join(lines)
    assert len(list((tmp_path / "data" / "synth_steps").iterdir())) == 6

    stamps = {path.name: path.stat().st_mtime_ns for path in (tmp_path / "data" / "synth_steps").iterdir()}
    assert dataset.main(["--config", str(config), "--arms", "global", "--workers", "1"]) == 0
    out = capsys.readouterr().out
    assert "[global]" in out and "[alert_baseline]" not in out
    assert stamps == {path.name: path.stat().st_mtime_ns for path in (tmp_path / "data" / "synth_steps").iterdir()}

    shutil.rmtree(synth_dir)
    assert dataset.main(["--config", str(config)]) == 2
    assert "python -m drowsy.synth" in capsys.readouterr().err
    assert dataset.main(["--config", str(tmp_path / "none.yaml")]) == 2
    capsys.readouterr()


def test_cli_rebuild_and_frame_drop_variants(
    tmp_path: Path, config_path: Path, pipeline_spec: PipelineSpec, capsys: pytest.CaptureFixture[str]
) -> None:
    synth_dir = tmp_path / "data" / "synth"
    synth.write_dataset(synth_dir, 2, 1.2, 0, pipeline_spec)
    rates = {"frame_drop_min": 0.10, "frame_drop_max": 0.30}
    config = write_config(tmp_path, config_path, "synth.yaml", training=rates, synth_dir=str(synth_dir))
    cache = tmp_path / "data" / "synth_steps"
    prefix = spec_pipeline_hash(pipeline_spec)[:16]

    def stamps() -> dict[str, int]:
        return {path.name: path.stat().st_mtime_ns for path in cache.iterdir()}

    base = ["--config", str(config), "--workers", "1", "--arms", "global", "session_baseline"]
    assert dataset.main(base) == 0
    out = capsys.readouterr().out
    assert "rebuilt" not in out and "frame-drop" not in out
    assert sum(line.startswith("baselines: ") for line in out.splitlines()) == 1
    assert "of 8 sessions finished calibration later than 60000 ms" in out
    assert "alert-baseline fallback subjects" not in out  # alert 영상 기준선 조건을 만들지 않았다
    first = stamps()
    assert sorted(first) == sorted(
        f"{arm}_{prefix}.{kind}.parquet" for arm in ("global", "session_baseline") for kind in ("steps", "quality")
    )

    # --rebuild: 유효한 캐시가 있어도 요청한 조건을 다시 만들어 덮어쓴다. 요청하지 않은 조건은 그대로 둔다.
    assert dataset.main(base[:-1] + ["--rebuild"]) == 0
    out = capsys.readouterr().out
    assert "step tables for ['global'] ready" in out and ", rebuilt)" in out
    second = stamps()
    assert sorted(second) == sorted(first)
    for name in first:
        assert (second[name] > first[name]) == name.startswith("global_"), name
    assert dataset.main(base) == 0  # --rebuild가 없으면 그대로 읽는다
    capsys.readouterr()
    assert stamps() == second

    # --frame-drop-variants: 변형 표를 설정의 비율 범위(0.10~0.30)와 준 시드로 만들어 캐시에 둔다.
    variants = base + ["--frame-drop-variants", "0", "2", "0", "--frame-drop-seed", "3"]
    assert dataset.main(variants) == 0
    out = capsys.readouterr().out
    assert out.count("frame-drop variant 0 (seed 3, rates 0.1..0.3)") == 1  # 같은 번호는 한 번만 만든다
    assert "frame-drop variant 2 (seed 3, rates 0.1..0.3)" in out
    assert sum(line.startswith("[global] frame-drop variant") for line in out.splitlines()) == 2
    third = stamps()
    assert sorted(set(third) - set(second)) == sorted(
        f"{arm}_{prefix}_fd{variant}_s3_r0.1-0.3.{kind}.parquet"
        for arm in ("global", "session_baseline") for variant in (0, 2) for kind in ("steps", "quality")
    )  # fmt: skip
    assert {name: third[name] for name in second} == second  # 원본 표는 다시 만들지 않았다
    quality = pd.read_parquet(cache / f"global_{prefix}_fd2_s3_r0.1-0.3.quality.parquet")
    thinned = quality[quality["n_thinned"] > 0]
    assert len(thinned) > 0 and thinned["drop_rate"].between(0.10, 0.30).all()
    expected = [
        reference_plan(session_id, int(n), FrameDropSpec(variant=2, seed=3, drop_min=0.10, drop_max=0.30))[0]
        for session_id, n in zip(thinned["session_id"], thinned["n_source_frames"])
    ]
    assert thinned["drop_rate"].tolist() == expected
    assert dataset.main(variants) == 0
    capsys.readouterr()
    assert stamps() == third  # 두 번째에는 변형도 캐시에서 읽는다

    assert dataset.main(base + ["--frame-drop-variants", "-1"]) == 2
    assert "frame-drop variant" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 회전 점검 기록 (추출이 parquet 메타데이터에 남긴 것)
# ---------------------------------------------------------------------------


def rotation_record(ambiguous: bool, reason: str, metadata_rotation: int = 0) -> dict[str, Any]:
    """추출이 `rotation_check` 키에 적는 기록과 같은 모양의 dict(값은 이 테스트가 정한다)."""
    return {
        "version": extract.ROTATION_CHECK_VERSION,
        "settings": {"frames": 12, "interval_ms": 2000, "roll_margin_deg": 15.0},
        "rates": {"0": 1.0, "90": 0.0, "180": 1.0, "270": 0.0},
        "rolls": {"0": -2.0, "90": None, "180": -0.6, "270": None},
        "metadata_rotation": metadata_rotation,
        "n_frames": 12,
        "ambiguous": ambiguous,
        "reason": reason,
    }


def write_rotation_tree(
    root: Path, spec: PipelineSpec, extra: dict[str, Any] | None = None
) -> list[tuple[str, Path]]:
    """회전 점검의 경우를 하나씩 담은 추출 결과 폴더. (추출 결과의 상대 경로, 경로)의 목록을 돌려준다.

    extra를 주면 그 키들을 모든 파일의 메타데이터에 더한다(회전 점검과 무관한 키. 아래 표의 값은 그대로다).

    파일              메타데이터                                                           보정표            분류
    decided           기록(가르는 데 성공, best_rate), 적용 회전 0                           -                자동 점검이 정함
    turned            기록(성공, smallest_roll, 메타데이터 0), 적용 회전 90                  -                자동 점검이 정함
    ambiguous         기록(가르지 못함, ambiguous_kept_metadata), 적용 0, 원본 .MOV          -                확인 필요
    weak              기록(가르지 못함, ambiguous_smallest_roll, 메타데이터 90), 적용 180    -                확인 필요
    confirmed         기록(가르지 못함), 적용 0                                             0 (같은 각도)     수동 보정
    stale             기록(가르지 못함), 적용 0                                             180 (다른 각도)   확인 필요(미적용)
    manual            기록 없음, 적용 180, 수동 값 180                                       180              수동 보정
    command           기록 없음, 적용 270, 수동 값 270(`--video --rotation 270`)             -                수동 보정
    plain             기록 없음, 적용 90, 수동 값 없음(회전 메타데이터만 믿은 예전 추출)       -                기록 없음
    synthetic         합성 표시                                                            90 (대조 안 함)   기록 없음
    broken            parquet이 아님                                                       -                기록 없음
    missing           파일 없음                                                            -                없는 파일
    """
    frames = list(render(Scenario((), 1_000, fps=FPS), spec))

    def write(name: str, metadata: dict[str, Any]) -> None:
        write_raw_frames(frames, root / f"cam/{name}.parquet", spec, {**metadata, **(extra or {})})

    kept = extract.REASON_KEPT_METADATA
    write("decided", {"rotation": 0, extract.ROTATION_CHECK_KEY: rotation_record(False, extract.REASON_BEST_RATE)})
    write("turned", {"rotation": 90, extract.ROTATION_CHECK_KEY: rotation_record(False, extract.REASON_SMALLEST_ROLL)})
    write(
        "ambiguous",
        {"rotation": 0, "source": "/phone/cam/ambiguous.MOV", extract.ROTATION_CHECK_KEY: rotation_record(True, kept)},
    )
    weak = rotation_record(True, extract.REASON_AMBIGUOUS_ROLL, metadata_rotation=90)
    write("weak", {"rotation": 180, "source": "/phone/cam/weak.mp4", extract.ROTATION_CHECK_KEY: weak})
    write("confirmed", {"rotation": 0, extract.ROTATION_CHECK_KEY: rotation_record(True, kept)})
    write("stale", {"rotation": 0, extract.ROTATION_CHECK_KEY: rotation_record(True, kept)})
    write("manual", {"rotation": 180, "rotation_override": 180})
    write("command", {"rotation": 270, "rotation_override": 270})
    write("plain", {"rotation": 90, "rotation_override": None})
    write("synthetic", {"synthetic": True})
    (root / "cam/broken.parquet").write_bytes(b"not a parquet file")
    names = ["decided", "turned", "ambiguous", "weak", "confirmed", "stale", "manual", "command", "plain", "synthetic"]
    names.append("broken")
    return [(f"cam/{name}.parquet", root / f"cam/{name}.parquet") for name in [*names, "missing"]]


ROTATION_TABLE = {"cam/confirmed.mp4": 0, "cam/stale.mp4": 180, "cam/manual.MOV": 180, "cam/synthetic.mp4": 90}


def test_read_rotation_checks_sorts_every_file_into_one_group(tmp_path: Path, pipeline_spec: PipelineSpec) -> None:
    files = write_rotation_tree(tmp_path / "raw", pipeline_spec)
    summary = read_rotation_checks(files, ROTATION_TABLE)
    # 있는 파일 11개: 자동 점검 2(decided, turned), 수동 보정 3(confirmed, manual, command), 기록 없음 3(plain,
    # synthetic, broken), 확인 필요 3(ambiguous, weak, stale). 없는 파일 1개는 따로 센다.
    assert (summary.n_files, summary.n_missing) == (11, 1)
    assert (summary.n_decided, summary.n_manual, summary.n_no_record) == (2, 3, 3)
    assert (summary.n_ambiguous, summary.n_override_not_applied) == (2, 1)
    kept = "ambiguous_kept_metadata"
    assert summary.flagged == (
        # 영상의 이름은 보정표에 적을 키다: 추출 결과의 경로에 원본 영상의 확장자를 되살린다
        RotationFlag("cam/ambiguous.MOV", ROTATION_AMBIGUOUS, 0, 0, kept, None),
        RotationFlag("cam/weak.mp4", ROTATION_AMBIGUOUS, 180, 90, "ambiguous_smallest_roll", None),
        RotationFlag("cam/stale.mp4", ROTATION_OVERRIDE_NOT_APPLIED, 0, 0, kept, 180),
    )
    assert summary.as_dict() == {
        "n_files": 11,
        "n_missing": 1,
        "decided_by_check": 2,
        "manual_override": 3,
        "ambiguous_without_override": 2,
        "override_not_applied": 1,
        "no_record": 3,
        "flagged": [
            {"video": "cam/ambiguous.MOV", "kind": "ambiguous", "rotation": 0, "metadata_rotation": 0, "reason": kept,
             "override": None},
            {"video": "cam/weak.mp4", "kind": "ambiguous", "rotation": 180, "metadata_rotation": 90,
             "reason": "ambiguous_smallest_roll", "override": None},
            {"video": "cam/stale.mp4", "kind": "override_not_applied", "rotation": 0, "metadata_rotation": 0,
             "reason": kept, "override": 180},
        ],
    }  # fmt: skip
    json.dumps(summary.as_dict(), allow_nan=False)  # JSON 리포트에 그대로 들어간다

    # 보정표가 없으면 기록이 가르지 못했다고 한 넷이 모두 확인 필요다. 수동 보정은 파일에 수동 값이 적힌 둘(manual,
    # command)뿐이다. 영상의 상대 경로(확장자 포함)로 불러도 같은 파일을 가리키고, 그 이름이 그대로 목록에 실린다.
    by_video = [(name.replace(".parquet", ".mp4"), path) for name, path in files]
    bare = read_rotation_checks(by_video)
    assert (bare.n_files, bare.n_decided, bare.n_manual, bare.n_no_record) == (11, 2, 2, 3)
    assert [(flag.video, flag.kind) for flag in bare.flagged] == [
        ("cam/ambiguous.mp4", "ambiguous"),
        ("cam/weak.mp4", "ambiguous"),
        ("cam/confirmed.mp4", "ambiguous"),
        ("cam/stale.mp4", "ambiguous"),
    ]
    assert bare.n_ambiguous == 4 and bare.n_override_not_applied == 0
    # 각도가 360을 넘게 적힌 보정표도 같은 각도로 본다. 다른 폴더의 키는 아무 파일과도 맞지 않는다.
    wrapped = read_rotation_checks(files, {"cam/manual.mp4": 540, "cam/command.mp4": 90, "other/confirmed.mp4": 0})
    assert wrapped.n_manual == 1  # manual(540 = 180도)만. command는 보정표의 90도와 달라 확인 필요로 올라간다
    assert [(flag.video, flag.kind, flag.rotation, flag.override) for flag in wrapped.flagged][-1] == (
        "cam/command.mp4", "override_not_applied", 270, 90,
    )  # fmt: skip
    empty = read_rotation_checks([])
    assert empty == RotationSummary(0, 0, 0, 0, 0, ()) and empty.as_dict()["flagged"] == []


def test_rotation_reader_does_not_depend_on_other_metadata_keys(tmp_path: Path, pipeline_spec: PipelineSpec) -> None:
    """추출이 메타데이터에 다른 키를 더해도 회전 점검 기록의 분류는 그대로다.

    추출은 원본 영상의 파일 이름과 크기를 `source_name`, `source_size_bytes`로 적는다(크기를 읽지 못했으면 null). 그 키가
    없는 파일(합성 데이터, 그 키를 적기 전의 추출 결과), 있는 파일, 값이 null인 파일이 모두 같은 분류를 받아야 한다.
    """
    without = read_rotation_checks(write_rotation_tree(tmp_path / "without", pipeline_spec), ROTATION_TABLE)
    bare = read_rotation_checks(write_rotation_tree(tmp_path / "bare", pipeline_spec))  # 보정표 없이 읽은 것
    assert (without.n_files, without.n_missing, without.n_decided, without.n_manual) == (11, 1, 2, 3)
    assert (without.n_no_record, without.n_ambiguous, without.n_override_not_applied) == (3, 2, 1)

    added_keys = [
        {"source_name": "clip.mp4", "source_size_bytes": 123_456},
        {"source_name": "clip.avi", "source_size_bytes": None},
        {"source_name": "clip.mp4", "source_size_bytes": 123_456, "a_key_added_later": {"nested": [1, 2, 3]}},
    ]
    for number, extra in enumerate(added_keys):
        files = write_rotation_tree(tmp_path / f"with_{number}", pipeline_spec, extra)
        stored = extract.read_extract_metadata(files[0][1])
        assert all(stored[key] == value for key, value in extra.items())  # 키가 실제로 파일에 들어 있다
        summary = read_rotation_checks(files, ROTATION_TABLE)
        # 분류와 목록이 같다. 영상 이름의 확장자는 여전히 `source`에서 온다(`cam/ambiguous.MOV`): `source_name`이 아니다
        assert summary == without, extra
        assert summary.flagged[0].video == "cam/ambiguous.MOV"
        assert read_rotation_checks(files) == bare and bare.n_ambiguous == 4
        # 스텝 표 캐시의 내용 지문은 파일 전체를 보므로, 메타데이터에 키가 더해진 파일은 다른 지문을 낸다
        assert raw_content_digest(files[0][1]) != raw_content_digest(tmp_path / "without" / files[0][0])


def test_rotation_output_lines_and_report_section(tmp_path: Path, pipeline_spec: PipelineSpec) -> None:
    summary = read_rotation_checks(write_rotation_tree(tmp_path / "raw", pipeline_spec), ROTATION_TABLE)
    key = "extraction.rotation_overrides"
    line = rotation_check_line(summary, "section 4 of summary.md")
    assert line == (
        "rotation check of 11 raw file(s): 2 decided by the automatic check, 3 extracted with a manual override, "
        "2 ambiguous without a manual override, 1 with an override that was not applied, "
        "3 without a rotation-check record. "
        "LOOK AT THE 3 FLAGGED VIDEO(S) before using this data: section 4 of summary.md"
    )
    assert "\n" not in line
    assert rotation_flag_lines(summary, key) == [
        "  rotation ambiguous: cam/ambiguous.MOV: extracted at 0 deg (video metadata 0, ambiguous_kept_metadata); "
        "look at the video, add its angle to extraction.rotation_overrides and run the extraction again",
        "  rotation ambiguous: cam/weak.mp4: extracted at 180 deg (video metadata 90, ambiguous_smallest_roll); "
        "look at the video, add its angle to extraction.rotation_overrides and run the extraction again",
        "  rotation override not applied: cam/stale.mp4: extraction.rotation_overrides says 180 deg but the raw file "
        "was extracted at 0 deg; run the extraction again",
    ]
    section = rotation_check_section(summary, key)
    assert section[0].startswith("추출이 영상마다 남긴 회전 자동 점검 기록") and "`extraction.rotation_overrides`" in section[0]
    assert "추출 결과 파일 11개(없는 파일 1개는 세지 않았다):" in section[0]
    for row in (
        "| 자동 점검이 각도를 정함 | 2 |",
        "| 사람이 정한 각도(수동 회전 보정)로 추출됨 | 3 |",
        "| 자동 점검이 각도를 가르지 못했고 수동 보정이 없음(확인 필요) | 2 |",
        "| 보정표에 각도가 있는데 다른 각도로 추출돼 있음(확인 필요) | 1 |",
        "| 점검 기록 없음 | 3 |",
        "| `cam/ambiguous.MOV` | 가르지 못함, 수동 보정 없음 | 0 | 0 | `ambiguous_kept_metadata` | - |",
        "| `cam/weak.mp4` | 가르지 못함, 수동 보정 없음 | 180 | 90 | `ambiguous_smallest_roll` | - |",
        "| `cam/stale.mp4` | 보정표의 각도가 적용되지 않음(추출을 다시 실행) | 0 | 0 | `ambiguous_kept_metadata` | 180 |",
    ):
        assert row in section, row
    assert any(line.startswith("**확인이 필요한 영상 3개.**") for line in section)
    assert section[-1].startswith("점검 기록이 없는 파일 3개는")
    assert all("\n" not in line for line in section)

    # 확인할 영상도, 기록이 없는 파일도 없을 때
    clean = RotationSummary(n_files=4, n_missing=0, n_decided=3, n_manual=1, n_no_record=0, flagged=())
    assert rotation_check_line(clean, "nowhere") == (
        "rotation check of 4 raw file(s): 3 decided by the automatic check, 1 extracted with a manual override, "
        "0 ambiguous without a manual override, 0 with an override that was not applied, "
        "0 without a rotation-check record"
    )
    assert rotation_flag_lines(clean, key) == []
    section = rotation_check_section(clean, key)
    assert section[-1] == "확인이 필요한 영상이 없다." and "추출 결과 파일 4개:" in section[0]


# 가짜 랜드마커가 알아보는 그림: 위쪽 절반이 밝고 아래쪽 절반이 어두우면 "바로 선 얼굴", 고르게 회색이면
# "어느 쪽으로 돌려도 검출되는 얼굴"(실제 랜드마커가 거꾸로 선 얼굴도 찾는 경우를 흉내 낸다)이다.
VIDEO_W, VIDEO_H = 160, 120


def marker_image(top_bright: bool) -> np.ndarray:
    image = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
    half = slice(0, VIDEO_H // 2) if top_bright else slice(VIDEO_H // 2, VIDEO_H)
    image[half] = 235
    return image


def write_still_video(path: Path, image: np.ndarray, n_frames: int = 12) -> None:
    """같은 그림 n_frames장을 30 fps의 H.264 영상으로 쓴다(회전 메타데이터 없음)."""
    time_base = Fraction(1, 1000)
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=30)
    stream.width, stream.height, stream.pix_fmt = image.shape[1], image.shape[0], "yuv420p"
    stream.codec_context.time_base = time_base
    stream.time_base = time_base
    stream.options = {"bf": "0", "crf": "20"}
    for index in range(n_frames):
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts = round(index * 1000 / 30)
        frame.time_base = time_base
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def fake_face(spec: PipelineSpec, t_ms: int, w: int, h: int) -> RawFrame:
    """필수 랜드마크와 블렌드셰이프만 가진 얼굴 프레임. 변환 행렬은 회전이 없다(머리 roll 0도)."""
    lm = {index: (0.3 + index / 2000.0, 0.4 + index / 3000.0) for index in spec.landmarks.required_indices()}
    bs = {name: 0.01 * (k + 1) for k, name in enumerate(spec.blendshapes.required_names())}
    m = tuple(float(v) for v in (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.0, -2.0, -30.0, 1))
    return RawFrame(t_ms=t_ms, face=True, w=w, h=h, lm=lm, bs=bs, m=m)


def use_fake_landmarker(monkeypatch: pytest.MonkeyPatch) -> None:
    """`extract`의 랜드마커와 추론을 그림의 밝기 무늬만 보는 가짜로 바꾼다(MediaPipe와 모델 파일 없이 추출이 돈다)."""

    class Landmarker:
        def close(self) -> None:
            pass

    def detect_frame(landmarker: Landmarker, t_ms: int, rgb: np.ndarray, spec: PipelineSpec) -> RawFrame:
        height, width = rgb.shape[0], rgb.shape[1]
        gray = rgb.astype(np.float64).mean(axis=2)
        top, bottom = gray[: height // 2].mean(), gray[height // 2 :].mean()
        left, right = gray[:, : width // 2].mean(), gray[:, width // 2 :].mean()
        uniform = abs(top - bottom) < 20 and abs(left - right) < 20
        upright = top - bottom > 100
        if uniform or upright:
            return fake_face(spec, t_ms, width, height)
        return RawFrame(t_ms=t_ms, face=False, w=width, h=height, lm={}, bs={}, m=())

    monkeypatch.setattr(extract, "make_landmarker", lambda model_path, spec=None: Landmarker())
    monkeypatch.setattr(extract, "detect_frame", detect_frame)
    monkeypatch.setattr(extract, "mediapipe_version", lambda: "0.0-fake")


def test_rotation_record_written_by_the_extraction_reaches_the_dataset_command(
    tmp_path: Path,
    config_path: Path,
    feature_spec_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """추출 명령이 쓴 파일(가짜 랜드마커, 테스트가 만든 영상)의 회전 점검 기록을 `python -m drowsy.dataset`이 읽는다.

    추출의 기록은 영상과 랜드마커가 있어야 만들어지므로(영상 없이 부를 수 있는 것은 `extract.write_raw_frames`뿐이다)
    작은 영상 셋을 만들어 `python -m drowsy.extract --dataset --pilot`(이 프로세스에서 차례로 추출한다)을 그대로 돌린다.
    - 01/0.mp4: 위가 밝은 그림. 가짜 랜드마커는 0도에서만 얼굴을 찾는다 → 자동 점검이 0도로 정한다.
    - 01/5.mp4: 고른 회색. 네 각도 모두에서 roll 0도의 얼굴이 나온다 → 가르지 못하고 회전 메타데이터(0도)를 유지한다.
    - 01/10.mp4: 아래가 밝은 그림(거꾸로 찍힌 영상). 180도에서만 얼굴을 찾는다 → 자동 점검이 180도로 정한다.
    """
    use_fake_landmarker(monkeypatch)
    videos = tmp_path / "rldd"
    write_still_video(videos / "01/0.mp4", marker_image(top_bright=True))
    write_still_video(videos / "01/5.mp4", np.full((VIDEO_H, VIDEO_W, 3), 128, dtype=np.uint8))
    write_still_video(videos / "01/10.mp4", marker_image(top_bright=False))
    model_path = tmp_path / "fake_model.task"
    model_path.write_bytes(b"not a real model")
    spec_dict = read_feature_spec_dict(feature_spec_path)
    spec_dict["pipeline"]["mediapipe"]["model_sha256"] = file_sha256(model_path)
    spec_dict["pipeline_hash"] = compute_pipeline_hash(spec_dict["pipeline"])
    spec_path = tmp_path / "feature_spec.json"
    write_feature_spec(spec_dict, spec_path)
    raw = tmp_path / "raw"

    def config(name: str, overrides: dict[str, int]) -> Path:
        check = {"rotation_check_frames": 3, "rotation_check_interval_ms": 100, "rotation_overrides": overrides}
        return write_config(
            tmp_path, config_path, name, extraction=check, rldd_root=str(videos), raw_dir=str(raw),
            reports_dir=str(tmp_path / "reports"), model_path=str(model_path), feature_spec=str(spec_path),
        )  # fmt: skip

    def extract_all(config_file: Path) -> None:
        assert extract.main(["--config", str(config_file), "--dataset", "--pilot", "3", "--run-name", "fix_rot"]) == 0
        capsys.readouterr()

    def dataset_lines(config_file: Path) -> list[str]:
        assert dataset.main(["--config", str(config_file), "--workers", "1"]) == 0
        return capsys.readouterr().out.splitlines()

    plain = config("plain.yaml", {})
    extract_all(plain)
    files = [(name, raw / "native" / name) for name in ("01/0.parquet", "01/5.parquet", "01/10.parquet")]
    # 추출이 실제로 적은 기록: 키와 필드 이름이 읽는 쪽과 같다
    recorded = {name: extract.read_extract_metadata(path) for name, path in files}
    assert recorded["01/5.parquet"][extract.ROTATION_CHECK_KEY]["ambiguous"] is True
    assert recorded["01/5.parquet"][extract.ROTATION_CHECK_KEY]["reason"] == extract.REASON_KEPT_METADATA
    assert recorded["01/0.parquet"][extract.ROTATION_CHECK_KEY]["ambiguous"] is False
    assert [recorded[name]["rotation"] for name, _ in files] == [0, 0, 180]
    summary = read_rotation_checks(files)
    assert (summary.n_files, summary.n_decided, summary.n_manual, summary.n_no_record) == (3, 2, 0, 0)
    assert summary.flagged == (RotationFlag("01/5.mp4", ROTATION_AMBIGUOUS, 0, 0, "ambiguous_kept_metadata", None),)

    # dataset 명령: 목록 다음 줄에 수를, 그 아래에 확인할 영상을 적는다. 스텝 표는 그대로 만든다(종료 코드 0).
    lines = dataset_lines(plain)
    assert lines[1] == "manifest: 1 subjects, 3 logical videos, 3 sessions, folds [1]"
    assert lines[2] == (
        "rotation check of 3 raw file(s): 2 decided by the automatic check, 0 extracted with a manual override, "
        "1 ambiguous without a manual override, 0 with an override that was not applied, "
        "0 without a rotation-check record. LOOK AT THE 1 FLAGGED VIDEO(S) before using this data: the lines below"
    )
    assert lines[3] == (
        "  rotation ambiguous: 01/5.mp4: extracted at 0 deg (video metadata 0, ambiguous_kept_metadata); "
        "look at the video, add its angle to extraction.rotation_overrides and run the extraction again"
    )
    assert lines[4].startswith("step tables for")

    # 영상을 보고 점검이 고른 각도(0도)가 맞다고 보정표에 적는다. 추출은 그 파일을 다시 만들지 않으므로(기록이 남아
    # 있다) 보정표를 함께 읽어야 "확인됨"을 안다.
    confirmed = config("confirmed.yaml", {"01/5.mp4": 0})
    lines = dataset_lines(confirmed)
    assert lines[2] == (
        "rotation check of 3 raw file(s): 2 decided by the automatic check, 1 extracted with a manual override, "
        "0 ambiguous without a manual override, 0 with an override that was not applied, "
        "0 without a rotation-check record"
    )
    assert not any(line.startswith("  rotation") for line in lines)

    # 180도가 맞다고 적고 추출을 다시 돌리지 않았다: 파일은 아직 0도로 추출돼 있다.
    turned = config("turned.yaml", {"01/5.mp4": 180})
    lines = dataset_lines(turned)
    assert "0 ambiguous without a manual override, 1 with an override that was not applied" in lines[2]
    assert lines[3] == (
        "  rotation override not applied: 01/5.mp4: extraction.rotation_overrides says 180 deg but the raw file "
        "was extracted at 0 deg; run the extraction again"
    )
    # 추출을 다시 돌리면 그 각도로 추출되고(자동 점검을 하지 않으므로 기록이 없다) 수동 보정으로 센다.
    extract_all(turned)
    again = extract.read_extract_metadata(raw / "native/01/5.parquet")
    assert again["rotation"] == 180 and extract.ROTATION_CHECK_KEY not in again
    lines = dataset_lines(turned)
    assert "2 decided by the automatic check, 1 extracted with a manual override, 0 ambiguous" in lines[2]
    assert not any(line.startswith("  rotation") for line in lines)

