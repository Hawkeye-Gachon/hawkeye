"""`drowsy.signal_check` 테스트: E0 통과 기준, 라벨 독립성, 전역 상수, 제안 spec, 깜빡임 길이 모사 (PLAN 4.2, 4.3, 4.5절, 11장 E0).

E0 쪽은 `drowsy.synth`의 각본 시나리오로 만든 피험자 3명(세션 10개)의 추출 결과를 쓴다. 세션마다 기준선 상태와 깜빡임의
수·깊이를 손으로 정해 두었으므로 기준별 세션 수와 전역 상수를 손으로 계산한 값과 비교한다.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from drowsy import extract, signal_check, synth
from drowsy.blink import ClosureRunDetector
from drowsy.config import ARM_GLOBAL, ExperimentConfig, SignalGate, load_config
from drowsy.dataset import (
    ROTATION_AMBIGUOUS,
    DatasetSource,
    ManifestError,
    RotationFlag,
    build_step_table,
    open_dataset,
    replay_video,
)
from drowsy.extract import read_raw_frames, write_raw_frames
from drowsy.signal_check import (
    BLINK_SIM_NOTE,
    E0_REAL_NOTE,
    LEVEL_CELL,
    LEVEL_CONDITION,
    LEVEL_DURATION,
    STAMPS_FRAME_TIME,
    STAMPS_INTERPOLATED,
    blink_shape,
    build_proposed_spec,
    collect_session,
    constant_medians,
    criterion_flags,
    evaluate_criteria,
    frame_stamp_settings,
    global_constants,
    label_free,
    non_provisional_paths,
    parse_overrides,
    run_blink_sim,
    run_e0,
    sample_times,
    session_record,
    signal_table,
    simulate_blinks,
)
from drowsy.spec import (
    PipelineSpec,
    SpecError,
    apply_overrides,
    compute_pipeline_hash,
    feature_spec_from_dict,
    load_feature_spec,
    read_feature_spec_dict,
)
from drowsy.synth import Closure, FaceMissing, FaceProfile, Scenario, default_profile, render
from drowsy.types import (
    BASELINE_FALLBACK_GLOBAL,
    BASELINE_OFF_AXIS,
    BASELINE_OK,
    IDENTITY_R0,
)

GATE = SignalGate(
    min_baseline_ok_ratio=0.9,
    blink_rate_min_per_min=5.0,
    blink_rate_max_per_min=40.0,
    min_blink_rate_ok_ratio=0.8,
    min_peak_closure_median=0.8,
    min_peak_closure_ok_ratio=0.8,
)
SYNTHETIC_FIRST_LINE = "SYNTHETIC data (drowsy.synth): pipeline check only, not a performance result"
# 계획서(`docs/PLAN.md`) 1.2절의 결정 번호(D1~D16) 모양의 표기. E0과 깜빡임 모사의 리포트는 이 번호를 쓰지 않는다(절 번호로만
# 가리킨다). 그래서 `drowsy.evaluate`의 리포트와 달리 "D 번호는 계획서 1.2절의 결정 번호다"라는 머리말 줄이 없다. 이
# 리포트에 결정 번호를 쓰게 되면 그 줄(`evaluate.PLAN_DECISION_NOTE`)도 함께 넣는다.
PLAN_DECISION_NUMBER = re.compile(r"\bD\d{1,2}\b")

# ---------------------------------------------------------------------------
# 도우미
# ---------------------------------------------------------------------------


def hand_table(rows: list[tuple[str, int, float, float]]) -> pd.DataFrame:
    """(baseline_status, t_b_ms, blink_rate_per_min, peak_closure_median)의 목록으로 만든 세션 표."""
    return pd.DataFrame(rows, columns=["baseline_status", "t_b_ms", "blink_rate_per_min", "peak_closure_median"])


def by_name(criteria: tuple[signal_check.Criterion, ...]) -> dict[str, signal_check.Criterion]:
    return {criterion.name: criterion for criterion in criteria}


def closures(depth: float, starts_s: list[int]) -> tuple[Closure, ...]:
    """close 100 ms, hold 300 ms, open 150 ms인 감김들. 정점은 depth이고 0.5를 넘는 시간은 550 − 125 / depth ms다."""
    return tuple(Closure(hold_ms=300, depth=depth, start_ms=1000 * start) for start in starts_s)


def face(base: FaceProfile, e: tuple[float, float], b: tuple[float, float], jaw: float) -> FaceProfile:
    return dataclasses.replace(base, e_open_l=e[0], e_open_r=e[1], b_open_l=b[0], b_open_r=b[1], jaw_rest=jaw)


S1_DEPTHS = [0.6, 0.72, 0.81, 0.9, 1.0, 1.0, 0.9, 0.75]  # 임계(0.7, 0.8)에 딱 걸리는 값은 피한다
S1_STARTS_S = [70, 75, 80, 85, 90, 95, 100, 105]


def s1_scenario(base: FaceProfile) -> Scenario:
    """130초, 30 fps. 보정 구간(30초)에 감김 하나, 그 뒤에 깊이가 다른 감김 8개, 120~122초에 얼굴 없음."""
    segments = (
        Closure(hold_ms=300, depth=1.0, start_ms=30_000),
        *(Closure(hold_ms=300, depth=d, start_ms=1000 * s) for d, s in zip(S1_DEPTHS, S1_STARTS_S)),
        FaceMissing(2_000, start_ms=120_000),
    )
    return Scenario(segments, 130_000, fps=30.0, profile=face(base, (0.30, 0.32), (0.10, 0.14), 0.05))


def scripted_sessions(spec: PipelineSpec) -> dict[str, Scenario]:
    """상대 경로 → 각본. 피험자 3명, 세션 10개(2번의 drowsy 영상은 두 부분).

    세션         기준선                    깜빡임(기준선 완성 뒤)        기준 1 / 2 / 3
    01/0         ok, 60초                 8개, 정점 중앙값 0.855        O / O / O
    01/5         ok, 60초                 6개, 정점 0.85               O / O / O
    01/10        ok, 90초로 연장           6개, 정점 0.9                X / O / O
    02/0         ok, 60초                 없음                         O / X / X
    02/5         50초짜리(보정 미완)        -                            X / X / X
    02/10_1      ok, 60초                 6개, 정점 0.6                O / O / X
    02/10_2      ok, 60초                 6개, 정점 0.85               O / O / O
    03/0         전역 상수 대체(180초)      3개, 정점 0.9                X / O / O
    03/5, 03/10  ok, 60초                 2개, 정점 1.0                O / O / O
    합계                                                              7 / 8 / 7 (10개 중)
    """
    base = default_profile(spec)

    def scene(seconds: int, segments: tuple, profile: FaceProfile = base) -> Scenario:
        return Scenario(segments, 1000 * seconds, fps=15.0, profile=profile)

    late = [65, 70, 75, 80, 85, 90]
    return {
        "Fold1_part1/01/0.parquet": s1_scenario(base),
        "Fold1_part1/01/5.parquet": scene(100, closures(0.85, late), face(base, (0.29, 0.31), (0.16, 0.18), 0.06)),
        "Fold1_part1/01/10.parquet": scene(
            130,
            (FaceMissing(40_000, start_ms=0), *closures(0.9, [95, 100, 105, 110, 115, 120])),
            face(base, (0.33, 0.35), (0.08, 0.10), 0.04),
        ),
        "Fold1_part1/02/0.parquet": scene(100, (), face(base, (0.24, 0.26), (0.20, 0.24), 0.09)),
        "Fold1_part1/02/5.parquet": scene(50, closures(1.0, [20]), face(base, (0.40, 0.40), (0.30, 0.30), 0.02)),
        "Fold1_part1/02/10_1.parquet": scene(100, closures(0.6, late), face(base, (0.22, 0.24), (0.25, 0.27), 0.10)),
        "Fold1_part1/02/10_2.parquet": scene(
            100, closures(0.85, late), face(base, (0.275, 0.285), (0.19, 0.21), 0.08)
        ),
        "Fold1_part1/03/0.parquet": scene(200, (FaceMissing(175_000, start_ms=0), *closures(0.9, [185, 190, 195]))),
        "Fold1_part1/03/5.parquet": scene(70, closures(1.0, [62, 66])),
        "Fold1_part1/03/10.parquet": scene(70, closures(1.0, [62, 66])),
    }


EXPECTED_COUNTS = {"baseline_complete": 7, "blink_rate": 8, "peak_closure": 7}
# 개인 기준선이 완성된 세션 8개의 E, B, jaw0 (01/0, 01/5, 01/10, 02/0, 02/10_1, 02/10_2, 03/5, 03/10)
#   E:    0.31, 0.30, 0.34, 0.25, 0.23, 0.28, 0.27, 0.27  → 정렬하면 가운데 둘이 0.27, 0.28 → 0.275
#   B:    0.12, 0.17, 0.09, 0.22, 0.26, 0.20, 0.22, 0.22  → 가운데 둘이 0.20, 0.22 → 0.21
#   jaw0: 0.05, 0.06, 0.04, 0.09, 0.10, 0.08, 0.08, 0.08  → 가운데 둘이 0.08, 0.08 → 0.08
EXPECTED_CONSTANTS = {"e": 0.275, "b": 0.21, "jaw0": 0.08}


def write_config(folder: Path, config_path: Path, repo_root: Path, name: str, **paths: str) -> Path:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["paths"].update(
        feature_spec=str(repo_root / "config" / "feature_spec.json"),
        rules=str(repo_root / "config" / "rules.yaml"),
        **paths,
    )
    data["extraction"]["workers"] = 1
    out = folder / name
    out.write_text(yaml.safe_dump(data), encoding="utf-8")
    return out


@pytest.fixture(scope="module")
def scripted(
    tmp_path_factory: pytest.TempPathFactory, config_path: Path, repo_root: Path, pipeline_spec: PipelineSpec
) -> SimpleNamespace:
    """각본 세션 10개의 합성 데이터 폴더, 그것을 가리키는 설정, 기본 실행 결과."""
    tmp = tmp_path_factory.mktemp("scripted_e0")
    synth_dir = tmp / "data" / "synth"
    for rel_path, scenario in scripted_sessions(pipeline_spec).items():
        write_raw_frames(render(scenario, pipeline_spec), synth_dir / rel_path, pipeline_spec, {"synthetic": True})
    config = write_config(
        tmp, config_path, repo_root, "config.yaml", synth_dir=str(synth_dir), reports_dir=str(tmp / "reports")
    )
    cfg = load_config(config)
    source = open_dataset(cfg)
    result = run_e0(cfg, run_name="base", command="test command", config_path=config, source=source)
    return SimpleNamespace(tmp=tmp, synth_dir=synth_dir, config=config, cfg=cfg, source=source, result=result)


def read_outputs(run_dir: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(run_dir.iterdir())}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 통과 기준의 산술 (손으로 만든 표)
# ---------------------------------------------------------------------------


def test_default_config_holds_the_plan_criteria(experiment_config: ExperimentConfig) -> None:
    # PLAN 11장 E0: 90% 이상 / 80% 이상이 분당 5~40회 / 80% 이상이 중앙값 0.8 이상
    assert experiment_config.evaluation.signal_gate == GATE


def test_criteria_counts_on_a_hand_made_table(pipeline_spec: PipelineSpec) -> None:
    nan = math.nan
    table = hand_table(
        [
            # 기준선          T_b     분당    정점      기준 1  2  3
            (BASELINE_OK, 60000, 12.0, 0.95),  #     O  O  O
            (BASELINE_OK, 60000, 5.0, 0.80),  #      O  O  O   경계값은 만족
            (BASELINE_OK, 60000, 40.0, 1.00),  #     O  O  O   경계값은 만족
            (BASELINE_OFF_AXIS, 60000, 20.0, 0.9),  # O  O  O   off_axis는 대체가 아니다
            (BASELINE_OK, 90000, 20.0, 0.9),  #      X  O  O   연장된 기준선
            (BASELINE_FALLBACK_GLOBAL, 180000, 20.0, 0.9),  # X  O  O
            (BASELINE_OK, 60000, 4.99, 0.85),  #     O  X  O
            (BASELINE_OK, 60000, 40.01, 0.79),  #    O  X  X
            (BASELINE_OK, 60000, 0.0, nan),  #       O  X  X   깜빡임 없음
            ("none", -1, nan, nan),  #               X  X  X   보정이 끝나기 전에 끝남
        ]
    )
    criteria = by_name(evaluate_criteria(table, GATE, pipeline_spec))
    assert list(criteria) == ["baseline_complete", "blink_rate", "peak_closure"]
    assert [(c.count, c.total) for c in criteria.values()] == [(7, 10), (6, 10), (7, 10)]
    assert [c.ratio for c in criteria.values()] == [0.7, 0.6, 0.7]
    assert [c.required_ratio for c in criteria.values()] == [0.9, 0.8, 0.8]
    assert [c.passed for c in criteria.values()] == [False, False, False]

    flags = criterion_flags(table, GATE, pipeline_spec)
    assert flags["baseline_ok"].tolist() == [True, True, True, True, False, False, True, True, True, False]
    assert flags["blink_rate_ok"].tolist() == [True, True, True, True, True, True, False, False, False, False]
    assert flags["peak_closure_ok"].tolist() == [True, True, True, True, True, True, True, False, False, False]
    assert "baseline_ok" not in table.columns  # 받은 표는 고치지 않는다


def test_criteria_pass_exactly_at_the_required_ratio(pipeline_spec: PipelineSpec) -> None:
    good = (BASELINE_OK, 60000, 15.0, 0.9)
    bad = ("none", -1, math.nan, math.nan)
    # 10개 중 9개: 기준선 90%(통과), 깜빡임과 정점 90%(통과)
    criteria = by_name(evaluate_criteria(hand_table([good] * 9 + [bad]), GATE, pipeline_spec))
    assert [c.passed for c in criteria.values()] == [True, True, True]
    # 10개 중 8개: 기준선 80% < 90%(미달), 나머지 둘은 정확히 80%(통과)
    criteria = by_name(evaluate_criteria(hand_table([good] * 8 + [bad] * 2), GATE, pipeline_spec))
    assert [c.passed for c in criteria.values()] == [False, True, True]
    # 180개 중 161개(89.4%)는 미달, 162개(90.0%)는 통과
    assert not by_name(evaluate_criteria(hand_table([good] * 161 + [bad] * 19), GATE, pipeline_spec))[
        "baseline_complete"
    ].passed
    assert by_name(evaluate_criteria(hand_table([good] * 162 + [bad] * 18), GATE, pipeline_spec))[
        "baseline_complete"
    ].passed
    # 세션이 없으면 미달
    empty = evaluate_criteria(hand_table([]), GATE, pipeline_spec)
    assert [(c.count, c.total, c.ratio, c.passed) for c in empty] == [(0, 0, 0.0, False)] * 3


def test_criteria_follow_the_gate_settings(pipeline_spec: PipelineSpec) -> None:
    table = hand_table([(BASELINE_OK, 60000, 3.0, 0.75), (BASELINE_OK, 60000, 50.0, 0.9)])
    assert [c.count for c in evaluate_criteria(table, GATE, pipeline_spec)] == [2, 0, 1]
    loose = dataclasses.replace(
        GATE, blink_rate_min_per_min=2.0, blink_rate_max_per_min=60.0, min_peak_closure_median=0.7
    )
    assert [c.count for c in evaluate_criteria(table, loose, pipeline_spec)] == [2, 2, 2]
    # "60초"는 spec의 baseline.duration_ms다
    longer = apply_overrides(pipeline_spec, {"baseline.duration_ms": 90000})
    assert [c.count for c in evaluate_criteria(table, GATE, longer)] == [0, 0, 1]


# ---------------------------------------------------------------------------
# 전역 상수 (손으로 만든 표)
# ---------------------------------------------------------------------------


def quality_table() -> pd.DataFrame:
    nan = math.nan
    rows = [
        # subject, status, e_l, e_r, b_l, b_r, jaw0          세션별 E, B
        (1, BASELINE_OK, 0.30, 0.34, 0.10, 0.14, 0.05),  #        0.32, 0.12
        (1, BASELINE_OFF_AXIS, 0.20, 0.24, 0.20, 0.20, 0.07),  #  0.22, 0.20
        (2, BASELINE_OK, 0.26, 0.26, 0.30, 0.34, 0.01),  #        0.26, 0.32
        (2, BASELINE_OK, 0.28, 0.30, 0.24, 0.24, 0.11),  #        0.29, 0.24
        (3, BASELINE_FALLBACK_GLOBAL, 0.27, 0.27, 0.22, 0.22, 0.08),  # 전역 상수 자신: 쓰지 않는다
        (3, "none", nan, nan, nan, nan, nan),  #                   값 없음: 쓰지 않는다
        (4, BASELINE_OK, 0.40, 0.40, 0.02, 0.02, 0.09),  #        0.40, 0.02
    ]
    table = pd.DataFrame(
        rows, columns=["subject", "baseline_status", "base_e_l", "base_e_r", "base_b_l", "base_b_r", "base_jaw0"]
    )
    table["label"] = "not a label"  # 라벨 열은 읽지 않는다
    return table


def test_constant_medians_on_a_hand_made_table() -> None:
    table = quality_table()
    # 쓸 세션 5개. E: 0.22, 0.26, 0.29, 0.32, 0.40 → 0.29 / B: 0.02, 0.12, 0.20, 0.24, 0.32 → 0.20 /
    # jaw0: 0.01, 0.05, 0.07, 0.09, 0.11 → 0.07
    medians = constant_medians(table)
    assert medians.n_sessions == 5
    assert medians.e == pytest.approx(0.29, abs=1e-12)
    assert medians.b == pytest.approx(0.20, abs=1e-12)
    assert medians.jaw0 == pytest.approx(0.07, abs=1e-12)

    # 피험자 1, 2만(4개, 짝수): 중앙값은 가운데 두 값의 평균이다(PLAN 4.2절).
    # E: 0.22, 0.26, 0.29, 0.32 → 0.275 / B: 0.12, 0.20, 0.24, 0.32 → 0.22 / jaw0: 0.01, 0.05, 0.07, 0.11 → 0.06
    medians = constant_medians(table, subjects=[2, 1, 1])
    assert medians.n_sessions == 4
    assert (medians.e, medians.b, medians.jaw0) == pytest.approx((0.275, 0.22, 0.06), abs=1e-12)

    # 피험자 3은 개인 기준선이 없다
    with pytest.raises(ValueError, match="no session with a completed personal baseline"):
        constant_medians(table, subjects=[3])
    with pytest.raises(ValueError, match="lacks columns"):
        constant_medians(table.drop(columns=["base_jaw0"]))
    # 라벨 열이 없어도 같다
    assert constant_medians(table.drop(columns=["label"])) == constant_medians(table)


def test_global_constants_returns_a_global_baseline(pipeline_spec: PipelineSpec) -> None:
    table = quality_table()
    baseline = global_constants(table, pipeline_spec)
    assert (baseline.e_l, baseline.e_r, baseline.b_l, baseline.b_r, baseline.jaw0) == (0.29, 0.29, 0.2, 0.2, 0.07)
    assert baseline.r0 == IDENTITY_R0 and baseline.gaze_h0 == 0.0 and baseline.gaze_v0 == 0.0
    assert baseline.t_complete_ms == 0 and baseline.status == BASELINE_FALLBACK_GLOBAL

    fold = global_constants(table, pipeline_spec, subjects=(1, 2))
    assert (fold.e_l, fold.b_l, fold.jaw0) == (0.275, 0.22, 0.06)

    # 소수 넷째 자리로 반올림한다: 세션 하나의 E = (0.31234 + 0.31244) / 2 = 0.31239 → 0.3124,
    # B = (0.12346 + 0.12348) / 2 = 0.12347 → 0.1235(올림), jaw0 = 0.05554 → 0.0555(내림)
    one = pd.DataFrame(
        {
            "subject": [9],
            "baseline_status": [BASELINE_OK],
            "base_e_l": [0.31234],
            "base_e_r": [0.31244],
            "base_b_l": [0.12346],
            "base_b_r": [0.12348],
            "base_jaw0": [0.05554],
        }
    )
    rounded = global_constants(one, pipeline_spec)
    assert (rounded.e_l, rounded.b_l, rounded.jaw0) == (0.3124, 0.1235, 0.0555)
    assert constant_medians(one).e == pytest.approx(0.31239, abs=1e-12)  # 반올림 전 값은 그대로 남는다


# ---------------------------------------------------------------------------
# 덮어쓰기와 제안 spec
# ---------------------------------------------------------------------------


def test_parse_overrides() -> None:
    parsed = parse_overrides(
        [
            "closure.rho=0.30",
            "run_detector.min_duration_ms=60",
            "pose.order=Ry*Rx*Rz",
            "baseline.blink_record_quantiles=[0.5, 0.9]",
        ]
    )
    assert parsed == {
        "closure.rho": 0.3,
        "run_detector.min_duration_ms": 60,
        "pose.order": "Ry*Rx*Rz",
        "baseline.blink_record_quantiles": [0.5, 0.9],
    }
    assert type(parsed["run_detector.min_duration_ms"]) is int and type(parsed["closure.rho"]) is float
    assert parse_overrides([]) == {}
    for bad in ("closure.rho", "=0.3", "closure.rho="):
        with pytest.raises(ValueError, match="PATH=VALUE"):
            parse_overrides([bad])
    with pytest.raises(ValueError, match="more than once"):
        parse_overrides(["closure.rho=0.3", "closure.rho=0.4"])


def test_non_provisional_paths() -> None:
    overrides = {
        "closure.rho": 0.3,
        "closure.b_closed": 0.7,
        "run_detector.enter_thr": 0.55,
        "step_features.perclos_thr": 0.75,
        "yawn.enter_thr": 0.5,
        "nod.enter_deg": 12.0,
        "global_constants.e": 0.3,
        "closure.w": 0.6,
        "baseline.min_face_frames": 200,
        "run_detector.hold_ms": 400,
    }
    assert non_provisional_paths(overrides) == ["closure.w", "baseline.min_face_frames", "run_detector.hold_ms"]


def test_build_proposed_spec(feature_spec_path: Path, pipeline_spec: PipelineSpec) -> None:
    current = read_feature_spec_dict(feature_spec_path)
    untouched = json.dumps(current, sort_keys=True)
    constants = global_constants(quality_table(), pipeline_spec)  # e 0.29, b 0.2, jaw0 0.07
    proposed = build_proposed_spec(current, {"closure.rho": 0.3, "run_detector.enter_thr": 0.55}, constants)
    assert json.dumps(current, sort_keys=True) == untouched  # 받은 dict는 고치지 않는다

    # drowsy.spec으로 읽히고 pipeline_hash가 맞는다
    loaded = feature_spec_from_dict(proposed)
    assert proposed["pipeline_hash"] == compute_pipeline_hash(proposed["pipeline"]) != current["pipeline_hash"]
    assert proposed["spec_version"] == current["spec_version"] + 1
    assert loaded.pipeline.closure.rho == 0.3 and loaded.pipeline.run_detector.enter_thr == 0.55
    assert loaded.pipeline.global_constants.e == 0.29
    assert loaded.pipeline.global_constants.b == 0.2
    assert loaded.pipeline.global_constants.jaw0 == 0.07
    # spec.apply_overrides로 만든 것과 같은 상수다
    assert loaded.pipeline == apply_overrides(
        pipeline_spec,
        {
            "closure.rho": 0.3,
            "run_detector.enter_thr": 0.55,
            "global_constants.e": 0.29,
            "global_constants.b": 0.2,
            "global_constants.jaw0": 0.07,
        },
    )
    # 바뀐 곳은 그 다섯 값과 spec_version, pipeline_hash뿐이다
    restored = json.loads(json.dumps(proposed))
    restored["spec_version"] = current["spec_version"]
    restored["pipeline_hash"] = current["pipeline_hash"]
    restored["pipeline"]["closure"]["rho"] = current["pipeline"]["closure"]["rho"]
    restored["pipeline"]["run_detector"]["enter_thr"] = current["pipeline"]["run_detector"]["enter_thr"]
    restored["pipeline"]["global_constants"] = current["pipeline"]["global_constants"]
    assert restored == current and list(proposed) == list(current)

    # 덮어쓴 전역 상수는 제안 파일에 들어가지 않는다(중앙값이 들어간다)
    forced = build_proposed_spec(current, {"global_constants.e": 0.4}, constants)
    assert forced["pipeline"]["global_constants"]["e"] == 0.29
    # 바뀐 것이 없으면 spec_version과 해시가 그대로다
    same = build_proposed_spec(current, {}, None)
    assert same == current
    # 검증을 통과하지 못하는 조합과 없는 경로는 SpecError다
    with pytest.raises(SpecError):
        build_proposed_spec(current, {"closure.b_closed": 0.1}, constants)  # b(0.2) >= b_closed
    with pytest.raises(SpecError, match="override path not found"):
        build_proposed_spec(current, {"closure.nothing": 1.0}, None)
    with pytest.raises(SpecError, match="expected float"):
        build_proposed_spec(current, {"closure.rho": 1}, None)


# ---------------------------------------------------------------------------
# 세션 하나
# ---------------------------------------------------------------------------


def test_collect_session_agrees_with_replay_video(pipeline_spec: PipelineSpec) -> None:
    """`collect_session`은 `dataset.replay_video`와 같은 파이프라인을 돌린다: 겹치는 값이 모두 같아야 한다."""
    profile = synth.make_subject(7, 3)
    frames = list(synth.generate_session(profile, 2, 150_000, 3, pipeline_spec))
    frames.insert(500, frames[499])  # 중복 타임스탬프 하나(버려진다)
    collected = collect_session(frames, pipeline_spec)
    replayed = replay_video(frames, pipeline_spec, None, None)
    assert collected.baseline == replayed.baseline and collected.baseline is not None
    assert collected.diagnostics == replayed.diagnostics
    assert collected.runs == replayed.runs and len(collected.runs) > 5
    assert (collected.n_frames, collected.n_dropped) == (replayed.n_frames, replayed.n_dropped) == (len(frames) - 1, 1)
    assert (collected.duration_ms, collected.face_ms) == (replayed.duration_ms, replayed.face_ms)
    assert collected.feature_ms == replayed.duration_ms - replayed.baseline.t_complete_ms
    assert 0 < collected.closure_valid_ms <= collected.feature_ms


def test_session_record_matches_hand_computed_values(pipeline_spec: PipelineSpec) -> None:
    scenario = s1_scenario(default_profile(pipeline_spec))
    frames = render(scenario, pipeline_spec)
    record = session_record(collect_session(frames, pipeline_spec), pipeline_spec, rule_enter_thr=0.7)

    # 기준선: 보정 구간의 눈은 거의 내내 떠 있으므로 E, B, jaw0은 얼굴의 뜬 눈 값이다. 자세는 정면.
    assert (record["baseline_status"], record["t_b_ms"]) == (BASELINE_OK, 60_000)
    expected = {"base_e_l": 0.30, "base_e_r": 0.32, "base_b_l": 0.10, "base_b_r": 0.14, "base_jaw0": 0.05}
    for name, value in expected.items():
        assert record[name] == pytest.approx(value, abs=1e-9), name
    pose = (record["base_yaw0"], record["base_pitch0"], record["base_roll0"])
    assert pose == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    assert record["calib_face_frames"] == 1800 and record["calib_face_ratio"] == pytest.approx(1.0)

    # 시간: 30 fps의 프레임 시각은 floor(k·1000/30 + 0.5). 마지막 프레임은 k = 3899 → 129967 ms.
    # 특징 구간은 60000~129967 ms(69967 ms)이고 그중 120000~122000 ms(2000 ms)에 얼굴이 없다.
    times = synth.scenario_times(scenario)
    assert times[-1] == 129_967 and record["n_frames"] == len(times) == 3900 and record["n_dropped"] == 0
    assert record["duration_ms"] == 129_967
    assert record["feature_ms"] == 69_967
    assert record["closure_valid_ms"] == 67_967
    assert record["closure_valid_ratio"] == pytest.approx(67_967 / 69_967)
    assert record["face_ratio"] == pytest.approx((129_967 - 2_000) / 129_967)

    # 깜빡임: 보정 구간(30초)의 감김은 세지 않는다. 8개, 잘린 것 없음.
    assert (record["n_runs"], record["n_truncated"], record["n_blinks"]) == (8, 0, 8)
    assert record["blink_rate_per_min"] == pytest.approx(8 * 60_000 / 67_967)  # 7.06
    # 정점 0.6, 0.72, 0.75, 0.81, 0.9, 0.9, 1.0, 1.0 → 25%는 2번째(0.72), 중앙값은 (0.81 + 0.9) / 2, 75%는 6번째(0.9)
    assert record["peak_closure_q25"] == pytest.approx(0.72, abs=1e-9)
    assert record["peak_closure_median"] == pytest.approx(0.855, abs=1e-9)
    assert record["peak_closure_q75"] == pytest.approx(0.9, abs=1e-9)
    assert record["n_peak_ge_perclos_thr"] == 5  # 0.8 이상: 0.81, 0.9, 0.9, 1.0, 1.0
    assert record["n_peak_ge_rule_enter"] == 7  # 0.7 이상: 0.6만 빠진다
    # 길이 = 550 − 125 / depth: 341.7, 376.4, 383.3, 395.7, 411.1, 411.1, 425, 425 (시각 반올림으로 ±1 ms)
    durations = sorted(550.0 - 125.0 / depth for depth in S1_DEPTHS)
    assert record["blink_dur_q25_ms"] == pytest.approx(durations[1], abs=1.01)
    assert record["blink_dur_median_ms"] == pytest.approx((durations[3] + durations[4]) / 2.0, abs=1.01)
    assert record["blink_dur_q75_ms"] == pytest.approx(durations[5], abs=1.01)
    assert record["blink_dur_mean_ms"] == pytest.approx(sum(durations) / 8.0, abs=1.01)
    assert record["blink_dur_max_ms"] == pytest.approx(425.0, abs=1.01)


def test_truncated_runs_are_not_blinks(pipeline_spec: PipelineSpec) -> None:
    """감긴 채로 얼굴이 1초 사라지면 그 감김은 잘린 구간과 그 뒤의 새 구간으로 나뉜다. 잘린 것은 깜빡임으로 세지 않는다."""
    segments = (
        Closure(hold_ms=2_000, depth=1.0, start_ms=70_000),  # 70.0~72.25초
        FaceMissing(1_000, start_ms=70_500),  # 그 안의 70.5~71.5초
        *closures(1.0, [80, 90]),
    )
    frames = render(Scenario(segments, 100_000, fps=30.0), pipeline_spec)
    record = session_record(collect_session(frames, pipeline_spec), pipeline_spec, rule_enter_thr=0.7)
    assert (record["n_runs"], record["n_truncated"], record["n_blinks"]) == (4, 1, 3)
    # 얼굴이 돌아온 71500 ms부터(보간할 앞 표본이 없다) 0.5 아래로 내려가는 72175 ms까지 675 ms, 나머지 둘은 425 ms
    assert record["blink_dur_max_ms"] == pytest.approx(675.0, abs=1.01)
    assert record["blink_dur_median_ms"] == pytest.approx(425.0, abs=1.01)
    # 특징 구간 60000~99967 ms에서 얼굴 없는 1000 ms를 뺀 38967 ms가 유효 시간이다
    assert record["closure_valid_ms"] == 38_967
    assert record["blink_rate_per_min"] == pytest.approx(3 * 60_000 / 38_967)
    assert record["peak_closure_median"] == pytest.approx(1.0, abs=1e-9)


def test_session_record_without_a_baseline(pipeline_spec: PipelineSpec) -> None:
    frames = render(Scenario(closures(1.0, [20]), 50_000, fps=15.0), pipeline_spec)
    record = session_record(collect_session(frames, pipeline_spec), pipeline_spec, rule_enter_thr=0.7)
    assert (record["baseline_status"], record["t_b_ms"], record["raw_status"]) == ("none", -1, "ok")
    assert record["n_frames"] == 750 and record["feature_ms"] == 0 and record["n_blinks"] == 0
    assert math.isnan(record["blink_rate_per_min"]) and math.isnan(record["peak_closure_median"])
    assert math.isnan(record["base_e_l"])


# ---------------------------------------------------------------------------
# E0 전체 (각본 데이터)
# ---------------------------------------------------------------------------


def test_run_e0_counts_and_constants_match_the_script(scripted: SimpleNamespace, feature_spec_path: Path) -> None:
    result = scripted.result
    table = result.table.set_index("session_id")
    assert len(table) == 10 and result.synthetic is True
    assert {c.name: (c.count, c.total) for c in result.criteria} == {
        name: (count, 10) for name, count in EXPECTED_COUNTS.items()
    }
    # 7/10 < 0.9 미달, 8/10 = 0.8 통과(경계), 7/10 < 0.8 미달
    assert [c.passed for c in result.criteria] == [False, True, False] and result.passed is False

    assert table["baseline_status"].to_dict() == {
        "S01_0": "ok",
        "S01_10": "ok",
        "S01_5": "ok",
        "S02_0": "ok",
        "S02_10_1": "ok",
        "S02_10_2": "ok",
        "S02_5": "none",
        "S03_0": "fallback_global",
        "S03_10": "ok",
        "S03_5": "ok",
    }
    assert table.loc["S01_10", "t_b_ms"] == 90_000 and table.loc["S03_0", "t_b_ms"] == 180_000
    assert not table.loc["S01_10", "baseline_ok"] and not table.loc["S03_0", "baseline_ok"]
    assert table["n_blinks"].to_dict() == {
        "S01_0": 8, "S01_10": 6, "S01_5": 6, "S02_0": 0, "S02_10_1": 6, "S02_10_2": 6, "S02_5": 0, "S03_0": 3,
        "S03_10": 2, "S03_5": 2,
    }  # fmt: skip
    peaks = {
        "S01_0": 0.855, "S01_10": 0.9, "S01_5": 0.85, "S02_10_1": 0.6, "S02_10_2": 0.85, "S03_0": 0.9, "S03_10": 1.0,
    }  # fmt: skip
    for session_id, peak in peaks.items():
        assert table.loc[session_id, "peak_closure_median"] == pytest.approx(peak, abs=1e-9), session_id
    assert math.isnan(table.loc["S02_0", "peak_closure_median"]) and table.loc["S02_0", "blink_rate_per_min"] == 0.0
    # 02/10_1: 15 fps의 마지막 프레임은 99933 ms. 유효 시간 39933 ms에 6개 → 분당 9.015
    assert table.loc["S02_10_1", "blink_rate_per_min"] == pytest.approx(6 * 60_000 / 39_933)
    # 03/0: 전역 상수로 대체된 뒤(180초~)의 20초에 3개
    assert table.loc["S03_0", "blink_rate_per_min"] == pytest.approx(3 * 60_000 / 19_933)

    # 전역 상수: 개인 기준선이 완성된 8개 세션의 중앙값(대체된 03/0과 보정이 안 끝난 02/5는 빠진다)
    constants = result.constants
    assert constants is not None
    assert (constants.e_l, constants.b_l, constants.jaw0) == (
        EXPECTED_CONSTANTS["e"],
        EXPECTED_CONSTANTS["b"],
        EXPECTED_CONSTANTS["jaw0"],
    )
    fold = global_constants(result.table, load_feature_spec(feature_spec_path).pipeline, subjects=[1])
    assert (fold.e_l, fold.b_l, fold.jaw0) == (0.31, 0.12, 0.05)  # 피험자 1: E 0.30, 0.31, 0.34 / B 0.09, 0.12, 0.17


def test_global_constants_accepts_the_dataset_quality_table(
    scripted: SimpleNamespace, pipeline_spec: PipelineSpec
) -> None:
    """fold별 상수(PLAN 4.3절의 민감도 점검)는 `dataset`의 품질 표에서 바로 계산할 수 있어야 한다."""
    quality = build_step_table(scripted.source.manifest, scripted.source.raw_dir, pipeline_spec, ARM_GLOBAL).quality
    assert global_constants(quality, pipeline_spec) == scripted.result.constants
    assert constant_medians(quality) == constant_medians(scripted.result.table)
    fold = global_constants(quality, pipeline_spec, subjects=[2, 3])
    # 피험자 2, 3: E 0.23, 0.25, 0.27, 0.27, 0.28 → 0.27 / B 0.20, 0.22, 0.22, 0.22, 0.26 → 0.22 / jaw0 → 0.08
    assert (fold.e_l, fold.b_l, fold.jaw0) == (0.27, 0.22, 0.08)
    # 그 기준선을 전역 조건의 덮어쓰기로 그대로 넘길 수 있다
    table = build_step_table(
        scripted.source.manifest, scripted.source.raw_dir, pipeline_spec, ARM_GLOBAL, global_override=fold
    )
    assert set(table.quality["baseline_used"]) == {"global", "none"} and len(table.steps) > 0


def test_run_e0_writes_the_report_files(scripted: SimpleNamespace, feature_spec_path: Path) -> None:
    run_dir = scripted.result.run_dir
    assert run_dir == scripted.tmp / "reports" / "base" / "e0"
    assert sorted(path.name for path in run_dir.iterdir()) == [
        "config.yaml",
        "criteria.json",
        "feature_spec_proposed.synthetic.json",  # 합성 데이터의 제안은 이름으로 구분한다
        "per_video.csv",
        "summary.md",
    ]
    # 첫 줄(JSON은 첫 키)이 데이터 종류다
    summary = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert summary.splitlines()[0] == f"# {SYNTHETIC_FIRST_LINE}"
    assert PLAN_DECISION_NUMBER.findall(summary) == []  # 결정 번호를 쓰지 않는 리포트다(`PLAN_DECISION_NUMBER`의 설명)
    assert (run_dir / "per_video.csv").read_text(encoding="utf-8").splitlines()[0] == f"# {SYNTHETIC_FIRST_LINE}"
    criteria = json.loads((run_dir / "criteria.json").read_text(encoding="utf-8"))
    assert next(iter(criteria)) == "data_note" and criteria["data_note"] == SYNTHETIC_FIRST_LINE
    assert criteria["synthetic"] is True and criteria["label_free"] is True
    assert criteria["manifest_columns_read"] == ["session_id", "subject", "rel_path"]
    assert criteria["command"] == "test command" and criteria["n_sessions"] == 10 and criteria["n_subjects"] == 3
    assert criteria["passed"] is False
    assert [(c["name"], c["count"], c["total"], c["passed"]) for c in criteria["criteria"]] == [
        ("baseline_complete", 7, 10, False),
        ("blink_rate", 8, 10, True),
        ("peak_closure", 7, 10, False),
    ]
    assert criteria["baseline"]["extended_sessions"] == ["S01_10"]
    assert criteria["baseline"]["fallback_sessions"] == ["S03_0"]
    assert criteria["baseline"]["not_completed_sessions"] == ["S02_5"]
    # 정점이 0.8 이상인 깜빡임: 5 + 6 + 6 + 0 + 0 + 6 + 0 + 3 + 2 + 2 = 30, 0.7 이상은 깊이 0.6인 7개를 뺀 32
    assert criteria["blinks"]["total"] == 39 and criteria["blinks"]["peak_ge_perclos_thr"] == 30
    assert criteria["blinks"]["peak_ge_rule_enter"] == 32
    assert criteria["global_constants"]["proposed"] == EXPECTED_CONSTANTS
    assert criteria["global_constants"]["current"] == {"e": 0.27, "b": 0.22, "jaw0": 0.08}
    assert criteria["global_constants"]["n_sessions_used"] == 8
    assert criteria["global_constants"]["includes_test_subjects"] is True
    assert criteria["follow_up"] == []  # 합성 데이터의 제안에는 "받아들인 뒤 할 일"이 없다
    # 회전 점검: 합성 세션에는 추출의 점검 기록이 없다. 그 수를 적고, 확인할 영상은 없다
    assert criteria["rotation_check"] == {
        "overrides_key": "extraction.rotation_overrides",
        "n_files": 10,
        "n_missing": 0,
        "decided_by_check": 0,
        "manual_override": 0,
        "ambiguous_without_override": 0,
        "override_not_applied": 0,
        "no_record": 10,
        "flagged": [],
    }
    assert scripted.result.rotation.n_no_record == 10 and scripted.result.rotation.flagged == ()
    section = summary.splitlines()[summary.splitlines().index("## 회전 점검") :]
    assert "| 점검 기록 없음 | 10 |" in section and "확인이 필요한 영상이 없다." in section
    assert section[-1].startswith("점검 기록이 없는 파일 10개는") and "합성 데이터" in section[-1]
    assert "회전이 확인되지 않은 영상" not in summary

    # per_video.csv: 세션마다 한 행, 라벨·클래스·fold 열이 없다
    table = pd.read_csv(run_dir / "per_video.csv", comment="#")
    assert len(table) == 10 and table["session_id"].tolist() == sorted(table["session_id"].tolist())
    assert not {"label", "class_index", "fold", "video_id"} & set(table.columns)
    assert int(table["baseline_ok"].sum()) == 7 and int(table["peak_closure_ok"].sum()) == 7

    # 제안 파일: drowsy.spec으로 읽히고 pipeline_hash가 맞고, 전역 상수만 바뀌었다
    proposed_path = scripted.result.proposed_spec_path
    assert proposed_path == run_dir / "feature_spec_proposed.synthetic.json"
    proposed = load_feature_spec(proposed_path)
    current = load_feature_spec(feature_spec_path)
    raw = json.loads(proposed_path.read_text(encoding="utf-8"))
    assert raw["pipeline_hash"] == compute_pipeline_hash(raw["pipeline"]) == proposed.pipeline_hash
    assert proposed.pipeline_hash != current.pipeline_hash and proposed.spec_version == current.spec_version + 1
    constants = proposed.pipeline.global_constants
    assert (constants.e, constants.b, constants.jaw0) == (0.275, 0.21, 0.08)
    assert proposed.pipeline == apply_overrides(
        current.pipeline, {"global_constants.e": 0.275, "global_constants.b": 0.21, "global_constants.jaw0": 0.08}
    )
    assert proposed.features == current.features and proposed.standardization == current.standardization
    assert criteria["proposed_spec"] == {
        "file": "feature_spec_proposed.synthetic.json",
        "spec_version": current.spec_version + 1,
        "pipeline_hash": proposed.pipeline_hash,
        "feature_spec_hash": sha256(proposed_path),
    }

    # 요약: 시험 피험자가 들어간 상수라는 것, 대체된 세션이 있을 때의 확인 명령, 합성 제안을 쓰지 말라는 것
    assert "시험 피험자의 영상도 들어간 값" in summary
    assert "global_constants.e=0.275 global_constants.b=0.21 global_constants.jaw0=0.08" in summary
    assert "`config/feature_spec.json`에 쓰지 않는다" in summary
    assert "| `baseline_complete`: 기준선이 60초에 대체 없이 완성 | 7 / 10 | 70.0% | 90% 이상 | 미달 |" in summary
    assert "| `blink_rate`: 깜빡임 분당 5~40회 | 8 / 10 | 80.0% | 80% 이상 | 통과 |" in summary
    assert (run_dir / "config.yaml").read_bytes() == scripted.config.read_bytes()


def test_label_permutation_leaves_every_output_byte_identical(scripted: SimpleNamespace) -> None:
    """라벨·클래스·fold 열을 뒤섞고, 지우고, 행 순서를 바꿔도 모든 출력 파일이 바이트 단위로 같다."""
    cfg, source = scripted.cfg, scripted.source

    def outputs(manifest: pd.DataFrame) -> dict[str, bytes]:
        permuted = dataclasses.replace(source, manifest=manifest)
        result = run_e0(cfg, run_name="labels", command="test command", config_path=scripted.config, source=permuted)
        return read_outputs(result.run_dir)

    reference = outputs(source.manifest)
    assert len(reference) == 5

    rng = np.random.default_rng(0)
    shuffled = source.manifest.copy()
    for column in ("label", "class_index", "fold"):
        shuffled[column] = shuffled[column].to_numpy()[rng.permutation(len(shuffled))]
    assert (shuffled["label"].to_numpy() != source.manifest["label"].to_numpy()).any()  # 실제로 섞였다
    shuffled["label"] = shuffled["label"] * 7 + 1  # 값 자체도 바꾼다
    shuffled["fold"] = 99
    assert outputs(shuffled) == reference

    reordered = shuffled.iloc[rng.permutation(len(shuffled))].reset_index(drop=True)
    assert outputs(reordered) == reference
    assert outputs(source.manifest.drop(columns=["label", "class_index", "fold", "video_id"])) == reference

    kept = label_free(shuffled)
    assert list(kept.columns) == ["session_id", "subject", "rel_path"]
    with pytest.raises(ManifestError, match="lacks columns"):
        label_free(source.manifest.drop(columns=["rel_path"]))


def test_relabelled_files_give_the_same_criteria_and_constants(
    scripted: SimpleNamespace, tmp_path: Path, config_path: Path, repo_root: Path
) -> None:
    """파일의 라벨 이름 자체를 바꿔도(세션의 내용은 그대로) 기준별 세션 수와 제안 상수가 같다."""
    swap = {
        "01/0": "01/5", "01/5": "01/10", "01/10": "01/0",
        "02/0": "02/5", "02/5": "02/0", "02/10_1": "02/10_2", "02/10_2": "02/10_1",
        "03/0": "03/10", "03/5": "03/0", "03/10": "03/5",
    }  # fmt: skip
    relabelled = tmp_path / "data" / "synth"
    for old, new in swap.items():
        target = relabelled / "Fold1_part1" / f"{new}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(scripted.synth_dir / "Fold1_part1" / f"{old}.parquet", target)
    config = write_config(
        tmp_path,
        config_path,
        repo_root,
        "relabelled.yaml",
        synth_dir=str(relabelled),
        reports_dir=str(tmp_path / "reports"),
    )
    result = run_e0(load_config(config), run_name="relabelled", command="test command")
    assert [(c.name, c.count, c.total, c.passed) for c in result.criteria] == [
        (c.name, c.count, c.total, c.passed) for c in scripted.result.criteria
    ]
    assert result.constants == scripted.result.constants

    # 세션별 값은 파일을 따라간다: 식별자를 뺀 행들의 모음이 같다
    def rows(table: pd.DataFrame) -> list[str]:
        values = table.drop(columns=["session_id", "rel_path"])
        return sorted(values.to_csv(index=False, header=False).splitlines())

    assert rows(result.table) == rows(scripted.result.table)
    relabelled_table = result.table.set_index("session_id")
    assert relabelled_table.loc["S01_0", "t_b_ms"] == 90_000  # 원래의 01/10(연장된 기준선)
    assert relabelled_table.loc["S03_10", "baseline_status"] == "fallback_global"  # 원래의 03/0


def test_override_changes_the_result(
    scripted: SimpleNamespace, capsys: pytest.CaptureFixture[str], feature_spec_path: Path
) -> None:
    """`--override closure.rho=0.05`: EAR 쪽 감김이 0.65·d / 0.95로 줄어 정점이 d·(0.5 + 0.5·0.65/0.95) = 0.8421·d가 된다."""
    before = sha256(feature_spec_path)
    arguments = ["e0", "--config", str(scripted.config), "--run-name", "rho", "--override", "closure.rho=0.05"]
    assert signal_check.main(arguments) == 0  # 합성 데이터에서는 미달이어도 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith(SYNTHETIC_FIRST_LINE)
    # 정점 중앙값 0.8 이상은 깊이 1.0인 두 세션(0.842)뿐이다: 7 → 2. 나머지 두 기준은 그대로다.
    baseline_line = "[baseline_complete] baseline complete at 60 s without fallback: 7/10 sessions = 70.0%"
    assert f"{baseline_line} (need >= 90%): NOT met" in out
    assert "[blink_rate] 5 to 40 blinks per minute: 8/10 sessions = 80.0% (need >= 80%): met" in out
    assert "[peak_closure] median blink peak closure >= 0.8: 2/10 sessions = 20.0% (need >= 80%): NOT met" in out
    assert "E0 criteria (label-free, 10 sessions): NOT met (synthetic data: pipeline check only)" in out
    # 합성 데이터의 출력에서 달라진 것은 이 한 줄이다: 회전 점검 기록이 없는 파일 수
    assert (
        "rotation check of 10 raw file(s): 0 decided by the automatic check, 0 extracted with a manual override, "
        "0 ambiguous without a manual override, 0 with an override that was not applied, "
        "10 without a rotation-check record"
    ) in out
    assert sum(line.startswith("rotation check of") for line in out) == 1

    run_dir = scripted.tmp / "reports" / "rho" / "e0"
    table = pd.read_csv(run_dir / "per_video.csv", comment="#").set_index("session_id")
    factor = 0.5 + 0.5 * 0.65 / 0.95
    assert table.loc["S01_5", "peak_closure_median"] == pytest.approx(0.85 * factor, abs=1e-9)  # 0.7158
    assert table.loc["S03_10", "peak_closure_median"] == pytest.approx(1.0 * factor, abs=1e-9)  # 0.8421
    assert table.loc["S02_10_1", "peak_closure_median"] == pytest.approx(0.6 * factor, abs=1e-9)  # 0.5053
    assert table["n_blinks"].tolist() == scripted.result.table["n_blinks"].tolist()  # 검출된 수는 같다
    assert int(table["peak_closure_ok"].sum()) == 2

    criteria = json.loads((run_dir / "criteria.json").read_text(encoding="utf-8"))
    assert criteria["overrides"] == {"closure.rho": 0.05} and criteria["overrides_not_marked_provisional"] == []
    assert criteria["pipeline_hash_evaluated"] != criteria["pipeline_hash_current"]
    assert criteria["command"] == "python -m drowsy.signal_check " + " ".join(arguments)
    # 제안 파일에는 덮어쓴 값과 제안 전역 상수가 함께 들어간다
    proposed = load_feature_spec(run_dir / "feature_spec_proposed.synthetic.json")
    assert proposed.pipeline == apply_overrides(
        load_feature_spec(feature_spec_path).pipeline,
        {"closure.rho": 0.05, "global_constants.e": 0.275, "global_constants.b": 0.21, "global_constants.jaw0": 0.08},
    )
    assert "`closure.rho` = 0.05" in (run_dir / "summary.md").read_text(encoding="utf-8")
    assert sha256(feature_spec_path) == before  # config/feature_spec.json은 그대로다

    # 잠정으로 표시되지 않은 상수를 덮어쓰면 요약에 경고가 실린다
    arguments = ["e0", "--config", str(scripted.config), "--run-name", "w"]
    arguments += ["--override", "closure.w=0.4", "closure.rho=0.3"]
    assert signal_check.main(arguments) == 0
    capsys.readouterr()
    summary = (scripted.tmp / "reports" / "w" / "e0" / "summary.md").read_text(encoding="utf-8")
    assert "계획서가 잠정으로 표시하지 않은 상수가 있다: `closure.w`" in summary

    # 틀린 덮어쓰기는 시작하지 못한다(종료 코드 2). 리포트 폴더를 만들지 않는다.
    for bad, message in (
        ("closure.rho=1", "expected float"),
        ("closure.nothing=1.0", "override path not found"),
        ("closure.rho", "PATH=VALUE"),
    ):
        assert signal_check.main(["e0", "--config", str(scripted.config), "--run-name", "bad", "--override", bad]) == 2
        assert message in capsys.readouterr().err
    assert not (scripted.tmp / "reports" / "bad").exists()


def test_real_data_gets_the_proposal_and_the_follow_up_steps(
    scripted: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    feature_spec_path: Path,
) -> None:
    real_open = signal_check.open_dataset

    def as_real(cfg: ExperimentConfig, fps_sim: bool = False) -> DatasetSource:
        return dataclasses.replace(real_open(cfg, fps_sim), synthetic=False, description="stand-in for real data")

    monkeypatch.setattr(signal_check, "open_dataset", as_real)
    before = sha256(feature_spec_path)
    # 실제 데이터에서 기준에 미달하면 종료 코드 4(리포트는 써 둔다)
    assert signal_check.main(["e0", "--config", str(scripted.config), "--run-name", "real"]) == 4
    captured = capsys.readouterr()
    assert captured.out.splitlines()[:2] == [E0_REAL_NOTE, "stand-in for real data"]
    assert "E0 criteria (label-free, 10 sessions): NOT met" in captured.out
    assert "synthetic" not in captured.out and "stop before training" in captured.err

    run_dir = scripted.tmp / "reports" / "real" / "e0"
    assert sorted(path.name for path in run_dir.iterdir()) == [
        "config.yaml",
        "criteria.json",
        "feature_spec_proposed.json",
        "per_video.csv",
        "summary.md",
    ]
    summary = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert summary.splitlines()[0] == f"# {E0_REAL_NOTE}"
    assert PLAN_DECISION_NUMBER.findall(summary) == []  # 실제 데이터의 리포트도 결정 번호를 쓰지 않는다
    assert (run_dir / "per_video.csv").read_text(encoding="utf-8").splitlines()[0] == f"# {E0_REAL_NOTE}"
    criteria = json.loads((run_dir / "criteria.json").read_text(encoding="utf-8"))
    assert criteria["data_note"] == E0_REAL_NOTE and criteria["synthetic"] is False
    proposed = load_feature_spec(run_dir / "feature_spec_proposed.json")
    new_hash = proposed.pipeline_hash
    steps = criteria["follow_up"]
    assert steps[0] == f"cp {(run_dir / 'feature_spec_proposed.json').as_posix()} {feature_spec_path.as_posix()}"
    assert steps[1].startswith(f"python -m drowsy.spec check --spec {feature_spec_path.as_posix()}")
    assert new_hash in steps[1]
    joined = "\n".join(steps)
    for needle in (
        "PIPELINE_HASH_V0",
        "global_constants.b, global_constants.e, global_constants.jaw0",
        "python -m drowsy.export_onnx --config",
        "--random-init --kind tcn --version 0",
        "python -m drowsy.vectors generate --out tests/vectors",
        "python -m drowsy.vectors verify --dir tests/vectors",
        "docs/SPEC.md",
        "python -m pytest -q",
        f"new pipeline_hash {new_hash}",
    ):
        assert needle in joined, needle
    for step in steps:
        assert step in summary
    assert "**이 도구는 `" in summary and "을 고치지 않는다.**" in summary
    assert sha256(feature_spec_path) == before

    # 기준을 모두 만족하면 0이다: 기준을 느슨하게 한 설정으로 다시 돌린다
    data = yaml.safe_load(scripted.config.read_text(encoding="utf-8"))
    data["evaluation"]["signal_gate"].update(
        min_baseline_ok_ratio=0.7, min_blink_rate_ok_ratio=0.8, min_peak_closure_ok_ratio=0.7
    )
    loose = scripted.tmp / "loose.yaml"
    loose.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert signal_check.main(["e0", "--config", str(loose), "--run-name", "loose"]) == 0
    assert "E0 criteria (label-free, 10 sessions): met" in capsys.readouterr().out


def rotation_record(ambiguous: bool, reason: str) -> dict:
    """추출이 `rotation_check` 키에 적는 기록과 같은 모양(값은 이 테스트가 정한다)."""
    return {
        "version": extract.ROTATION_CHECK_VERSION,
        "settings": {"frames": 12, "interval_ms": 2000, "roll_margin_deg": 15.0},
        "rates": {"0": 1.0, "90": 0.0, "180": 1.0, "270": 0.0},
        "rolls": {"0": -2.0, "90": None, "180": -0.6, "270": None},
        "metadata_rotation": 0,
        "n_frames": 12,
        "ambiguous": ambiguous,
        "reason": reason,
    }


def test_e0_reports_videos_whose_rotation_is_ambiguous_without_an_override(
    tmp_path: Path,
    config_path: Path,
    repo_root: Path,
    pipeline_spec: PipelineSpec,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """실제 데이터 꼴(합성 표시 없음)의 추출 결과 셋. 메타데이터의 회전 점검 기록은 이 테스트가 적는다.

    01/0.mp4: 자동 점검이 0도로 정했다.  01/5.mov: 가르지 못했고 보정표에 없다.  01/10.mp4: 가르지 못했다.
    세션은 모두 70초, 60초에 기준선 완성, 그 뒤 깊이 1.0의 깜빡임 둘(통과 기준 셋을 모두 만족한다: 3/3).
    """
    videos = tmp_path / "rldd"
    for name in ("01/0.mp4", "01/5.mov", "01/10.mp4"):
        (videos / name).parent.mkdir(parents=True, exist_ok=True)
        (videos / name).write_bytes(b"")
    raw = tmp_path / "raw"
    frames = render(Scenario(closures(1.0, [62, 66]), 70_000, fps=15.0), pipeline_spec)
    key = extract.ROTATION_CHECK_KEY
    kept = extract.REASON_KEPT_METADATA
    metadata = {
        "01/0.parquet": {"rotation": 0, key: rotation_record(False, extract.REASON_BEST_RATE)},
        "01/5.parquet": {"rotation": 0, "source": "/data/rldd/01/5.mov", key: rotation_record(True, kept)},
        "01/10.parquet": {"rotation": 0, "source": "/data/rldd/01/10.mp4", key: rotation_record(True, kept)},
    }
    for name, values in metadata.items():
        write_raw_frames(frames, raw / "native" / name, pipeline_spec, values)

    def config(name: str, overrides: dict[str, int]) -> Path:
        base = write_config(
            tmp_path, config_path, repo_root, name, rldd_root=str(videos), raw_dir=str(raw),
            reports_dir=str(tmp_path / "reports"),
        )  # fmt: skip
        data = yaml.safe_load(base.read_text(encoding="utf-8"))
        data["extraction"]["rotation_overrides"] = overrides
        base.write_text(yaml.safe_dump(data), encoding="utf-8")
        return base

    def run(name: str, overrides: dict[str, int]) -> tuple[int, list[str], Path]:
        code = signal_check.main(["e0", "--config", str(config(f"{name}.yaml", overrides)), "--run-name", name])
        return code, capsys.readouterr().out.splitlines(), tmp_path / "reports" / name / "e0"

    # 보정표에 01/10.mp4만 있다(점검이 고른 0도를 사람이 확인했다): 01/5.mov 하나가 확인 필요로 남는다
    code, out, run_dir = run("one_flag", {"01/10.mp4": 0})
    assert out[0] == E0_REAL_NOTE and "E0 criteria (label-free, 3 sessions): met" in out
    assert (
        "rotation check of 3 raw file(s): 1 decided by the automatic check, 1 extracted with a manual override, "
        "1 ambiguous without a manual override, 0 with an override that was not applied, "
        "0 without a rotation-check record. LOOK AT THE 1 FLAGGED VIDEO(S) before using this data: "
        f"section '회전 점검' of {run_dir / 'summary.md'}"
    ) in out
    criteria = json.loads((run_dir / "criteria.json").read_text(encoding="utf-8"))
    assert criteria["rotation_check"] == {
        "overrides_key": "extraction.rotation_overrides",
        "n_files": 3,
        "n_missing": 0,
        "decided_by_check": 1,
        "manual_override": 1,
        "ambiguous_without_override": 1,
        "override_not_applied": 0,
        "no_record": 0,
        "flagged": [
            {"video": "01/5.mov", "kind": "ambiguous", "rotation": 0, "metadata_rotation": 0, "reason": kept,
             "override": None},
        ],
    }  # fmt: skip
    summary = (run_dir / "summary.md").read_text(encoding="utf-8").splitlines()
    assert any(line.startswith("**주의: 회전이 확인되지 않은 영상이 1개 있다.**") for line in summary)
    section = summary[summary.index("## 회전 점검") :]
    assert "| 자동 점검이 각도를 가르지 못했고 수동 보정이 없음(확인 필요) | 1 |" in section
    assert "| `01/5.mov` | 가르지 못함, 수동 보정 없음 | 0 | 0 | `ambiguous_kept_metadata` | - |" in section
    assert not any("01/10" in line or "01/0." in line for line in section)  # 확인된 영상과 가른 영상은 목록에 없다
    flagged_counts = [(c["count"], c["total"], c["passed"]) for c in criteria["criteria"]]
    assert flagged_counts == [(3, 3, True)] * 3

    # 01/5.mov도 보정표에 적으면 확인할 영상이 없다. 통과 기준의 판정과 종료 코드는 표시가 있을 때와 같다(가정)
    clean_code, out, run_dir = run("no_flag", {"01/10.mp4": 0, "01/5.mov": 0})
    assert code == clean_code == 0
    assert (
        "rotation check of 3 raw file(s): 1 decided by the automatic check, 2 extracted with a manual override, "
        "0 ambiguous without a manual override, 0 with an override that was not applied, "
        "0 without a rotation-check record"
    ) in out
    criteria = json.loads((run_dir / "criteria.json").read_text(encoding="utf-8"))
    assert criteria["rotation_check"]["flagged"] == [] and criteria["rotation_check"]["manual_override"] == 2
    assert [(c["count"], c["total"], c["passed"]) for c in criteria["criteria"]] == flagged_counts
    summary = (run_dir / "summary.md").read_text(encoding="utf-8").splitlines()
    assert "확인이 필요한 영상이 없다." in summary and not any("회전이 확인되지 않은" in line for line in summary)

    # 보정표가 비어 있으면 기록이 가르지 못했다고 한 둘이 모두 남는다. 결과 객체에도 같은 목록이 있다.
    # 순서는 세션 표의 순서(피험자, session_id의 문자열 순: S01_0, S01_10, S01_5)이고 라벨과 무관하다
    cfg = load_config(config("api.yaml", {}))
    result = run_e0(cfg, run_name="api", command="test command")
    assert result.rotation.flagged == (
        RotationFlag("01/10.mp4", ROTATION_AMBIGUOUS, 0, 0, kept, None),
        RotationFlag("01/5.mov", ROTATION_AMBIGUOUS, 0, 0, kept, None),
    )


@pytest.mark.parametrize("command", ["e0", "blink-sim"])
def test_cli_refuses_a_run_name_that_is_not_a_plain_folder_name(
    command: str, scripted: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--run-name ../escape`는 리포트를 리포트 폴더 밖에 쓰게 된다. 두 하위 명령 모두 아무것도 하기 전에 거부한다."""
    for bad in ("../escape", "a/b", "..", ".", "", "a\\b"):
        assert signal_check.main([command, "--config", str(scripted.config), "--run-name", bad]) == 2
        captured = capsys.readouterr()
        assert captured.err == f"error: --run-name must be a plain folder name, got {bad!r}\n"
        assert captured.out == ""
    # 리포트 폴더(scripted.tmp / "reports") 밖에 생긴 것이 없다
    assert not (scripted.tmp / "escape").exists() and not (scripted.tmp / "reports" / "a").exists()
    # 다른 일보다 먼저 검사한다: 설정 파일이 없어도 이름의 오류가 나온다
    arguments = [command, "--config", str(scripted.tmp / "missing.yaml"), "--run-name", "../escape"]
    assert signal_check.main(arguments) == 2
    assert capsys.readouterr().err == "error: --run-name must be a plain folder name, got '../escape'\n"
    # 함수로 불러도 같다
    with pytest.raises(ValueError, match="--run-name must be a plain folder name, got '../escape'"):
        if command == "e0":
            run_e0(scripted.cfg, run_name="../escape", source=scripted.source)
        else:
            run_blink_sim(scripted.cfg, run_name="../escape")
    assert not (scripted.tmp / "escape").exists()


def test_workers_do_not_change_the_table(scripted: SimpleNamespace, pipeline_spec: PipelineSpec) -> None:
    source = scripted.source
    parallel = signal_table(source.manifest, source.raw_dir, pipeline_spec, workers=2)
    serial = signal_table(source.manifest, source.raw_dir, pipeline_spec, workers=1)
    pd.testing.assert_frame_equal(parallel, serial)
    pd.testing.assert_frame_equal(
        criterion_flags(serial, GATE, pipeline_spec).reset_index(drop=True),
        scripted.result.table.reset_index(drop=True),
    )
    with pytest.raises(ValueError, match="workers"):
        signal_table(source.manifest, source.raw_dir, pipeline_spec, workers=0)


def test_missing_extraction_counts_against_every_criterion(
    scripted: SimpleNamespace, tmp_path: Path, pipeline_spec: PipelineSpec
) -> None:
    raw_dir = tmp_path / "raw"
    shutil.copytree(scripted.synth_dir, raw_dir)
    (raw_dir / "Fold1_part1" / "01" / "5.parquet").unlink()
    table = criterion_flags(signal_table(scripted.source.manifest, raw_dir, pipeline_spec), GATE, pipeline_spec)
    row = table.set_index("session_id").loc["S01_5"]
    assert row["raw_status"] == "missing" and row["baseline_status"] == "none" and row["n_frames"] == 0
    assert not row["baseline_ok"] and not row["blink_rate_ok"] and not row["peak_closure_ok"]
    assert [(c.count, c.total) for c in evaluate_criteria(table, GATE, pipeline_spec)] == [(6, 10), (7, 10), (6, 10)]
    with pytest.raises(ManifestError, match="none of the 10 raw parquet files"):
        signal_table(scripted.source.manifest, tmp_path / "nowhere", pipeline_spec)


def test_fps_sim_run_writes_no_proposal(
    scripted: SimpleNamespace, capsys: pytest.CaptureFixture[str], pipeline_spec: PipelineSpec
) -> None:
    # 누락 모사 합성 데이터가 없으면 만드는 명령을 알려 주고 2로 끝난다
    assert signal_check.main(["e0", "--config", str(scripted.config), "--run-name", "sim", "--fps-sim"]) == 2
    assert "--fps-sim 15" in capsys.readouterr().err

    thinned = scripted.synth_dir.with_name("synth_fps15")
    for rel_path in ("Fold1_part1/03/0.parquet", "Fold1_part1/03/5.parquet", "Fold1_part1/03/10.parquet"):
        frames = list(read_raw_frames(scripted.synth_dir / rel_path))
        write_raw_frames(frames[::2], thinned / rel_path, pipeline_spec, {"synthetic": True})  # 한 장 걸러 한 장
    assert signal_check.main(["e0", "--config", str(scripted.config), "--run-name", "sim", "--fps-sim"]) == 0
    out = capsys.readouterr().out
    assert "[blink_rate] 5 to 40 blinks per minute: 3/3 sessions" in out
    assert "reference only" in out and "proposed spec" not in out
    run_dir = scripted.tmp / "reports" / "sim" / "e0_fps_sim"
    names = sorted(path.name for path in run_dir.iterdir())
    assert names == ["config.yaml", "criteria.json", "per_video.csv", "summary.md"]
    criteria = json.loads((run_dir / "criteria.json").read_text(encoding="utf-8"))
    assert criteria["fps_sim"] is True and criteria["proposed_spec"] is None and criteria["follow_up"] == []
    assert "제안 파일을 쓰지 않는다" in (run_dir / "summary.md").read_text(encoding="utf-8")
    shutil.rmtree(thinned)


def test_cli_cannot_start_without_data(
    tmp_path: Path, config_path: Path, repo_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(
        tmp_path,
        config_path,
        repo_root,
        "none.yaml",
        synth_dir=str(tmp_path / "nowhere"),
        reports_dir=str(tmp_path / "reports"),
    )
    assert signal_check.main(["e0", "--config", str(config)]) == 2
    assert "python -m drowsy.synth" in capsys.readouterr().err
    assert signal_check.main(["e0", "--config", str(tmp_path / "missing.yaml")]) == 2
    capsys.readouterr()
    assert not (tmp_path / "reports").exists()


# ---------------------------------------------------------------------------
# 깜빡임 길이 모사
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sim_table(pipeline_spec: PipelineSpec) -> pd.DataFrame:
    return simulate_blinks(pipeline_spec.run_detector, seed=0)


def sim_row(
    table: pd.DataFrame, level: str, condition: str, stamps: str, duration_ms: float | None = None
) -> pd.Series:
    rows = table[(table["level"] == level) & (table["condition"] == condition) & (table["stamps"] == stamps)]
    if duration_ms is not None:
        rows = rows[rows["duration_ms"] == duration_ms]
    assert len(rows) == 1
    return rows.iloc[0]


@pytest.mark.parametrize(("duration_ms", "amplitude"), [(60.0, 1.0), (150.0, 0.7), (300.0, 0.85), (400.0, 0.55)])
def test_blink_shape_has_the_requested_time_above_the_threshold(duration_ms: float, amplitude: float) -> None:
    shape = blink_shape(duration_ms, amplitude, enter_thr=0.5)
    step = 0.01
    grid = np.arange(shape.start_ms - 10.0, shape.start_ms + shape.total_ms + 10.0, step)
    values = np.array([shape.closure(float(t)) for t in grid])
    assert float((values >= 0.5).sum()) * step == pytest.approx(duration_ms, abs=0.05)  # 조밀하게 세어 본 값
    assert values.max() == pytest.approx(amplitude, abs=1e-6) and values.min() == 0.0
    assert shape.closure(shape.start_ms) == 0.0 and shape.closure(shape.start_ms + shape.total_ms) == 0.0
    # 감는 구간은 전체의 35%이고 정점은 그 끝에 있다
    assert grid[int(values.argmax())] == pytest.approx(shape.start_ms + 0.35 * shape.total_ms, abs=0.02)
    # 연속 파형의 최대 속도(1/s): 수치 미분의 최댓값과 같다
    slopes = np.diff(values) / step * 1000.0
    assert shape.peak_closing_velocity() == pytest.approx(float(slopes.max()), rel=1e-3)
    assert shape.peak_opening_velocity() == pytest.approx(float(-slopes.min()), rel=1e-3)
    if amplitude == 1.0:
        assert shape.total_ms == pytest.approx(2.0 * duration_ms)  # 진폭 1, 임계 0.5이면 전체 길이는 2배
    with pytest.raises(ValueError, match="above the enter threshold"):
        blink_shape(100.0, 0.5, enter_thr=0.5)


def test_sample_times() -> None:
    rng = random.Random(0)
    assert sample_times(30.0, 0.0, 200.0, 0.0, rng) == [0, 33, 67, 100, 133, 167, 200]
    assert sample_times(15.0, 10.0, 220.0, 0.0, rng) == [10, 77, 143, 210]
    regular = sample_times(15.0, 0.0, 60_000.0, 0.0, rng)
    dropped = sample_times(15.0, 0.0, 60_000.0, 0.2, random.Random(5))
    assert set(dropped) < set(regular) and dropped == sorted(dropped)
    assert 0.15 < 1.0 - len(dropped) / len(regular) < 0.25  # 약 20%가 빠진다
    assert dropped == sample_times(15.0, 0.0, 60_000.0, 0.2, random.Random(5))  # 같은 시드면 같다


def test_frame_time_emulation_is_a_valid_detector_setting(pipeline_spec: PipelineSpec) -> None:
    settings = frame_stamp_settings(pipeline_spec.run_detector)
    assert settings == dataclasses.replace(pipeline_spec.run_detector, max_dt_ms=1)
    # spec 검증을 통과하는 설정이다(검출기 코드를 따로 두지 않는다)
    assert apply_overrides(pipeline_spec, {"run_detector.max_dt_ms": 1}).run_detector == settings

    # 보간을 끄면 시작과 끝이 프레임 시각이다: 0.5 이상을 처음 본 프레임과 마지막으로 본 프레임
    shape = blink_shape(200.0, 1.0, enter_thr=0.5)
    times = sample_times(15.0, 20.0, 3000.0, 0.0, random.Random(0))
    closed = [t for t in times if shape.closure(float(t)) >= 0.5]
    run, truncated = signal_check.detect_blink(shape, times, settings)
    assert run is not None and not truncated
    assert (run.start_ms, run.end_ms, run.duration_ms) == (closed[0], closed[-1], closed[-1] - closed[0])
    assert run.velocity_valid is False


def test_single_trial_matches_an_independent_computation(pipeline_spec: PipelineSpec) -> None:
    """보간하는 쪽의 길이를 검출기 없이 따로 계산해 본다: 임계를 지나는 두 쌍의 선형 보간."""
    settings = pipeline_spec.run_detector
    shape = blink_shape(150.0, 0.85, settings.enter_thr)
    times = sample_times(15.0, 23.0, shape.start_ms + shape.total_ms + 1500.0, 0.0, random.Random(0))
    values = [shape.closure(float(t)) for t in times]
    above = [i for i, value in enumerate(values) if value >= settings.enter_thr]
    first, last = above[0], above[-1]
    assert above == list(range(first, last + 1)) and first > 0

    def crossing(i: int, j: int) -> float:  # 표본 i와 j 사이에서 임계를 지나는 시각
        fraction = (settings.enter_thr - values[i]) / (values[j] - values[i])
        return times[i] + fraction * (times[j] - times[i])

    expected = crossing(last, last + 1) - crossing(first - 1, first)
    run, _ = signal_check.detect_blink(shape, times, settings)
    assert run is not None
    assert run.duration_ms == pytest.approx(expected, abs=1.01)  # 시작과 끝을 각각 정수 ms로 반올림한다
    assert run.amplitude == max(values)
    # 참값(150 ms)과의 차는 표본 간격(67 ms)보다 훨씬 작다
    assert abs(run.duration_ms - 150.0) < 15.0


def test_blink_sim_is_deterministic(sim_table: pd.DataFrame, pipeline_spec: PipelineSpec) -> None:
    again = simulate_blinks(pipeline_spec.run_detector, seed=0)
    pd.testing.assert_frame_equal(again, sim_table)
    other = simulate_blinks(pipeline_spec.run_detector, seed=1)
    regular = sim_table["drop_prob"] == 0.0
    pd.testing.assert_frame_equal(other[regular], sim_table[regular])  # 누락이 없는 조건은 시드와 무관하다
    assert not other[~regular].equals(sim_table[~regular])  # 누락 패턴은 시드를 따른다


def test_blink_sim_table_layout(sim_table: pd.DataFrame) -> None:
    # 조건 3 × 시각 방식 2 × (길이 7 × 진폭 3 + 길이 7 + 1)
    assert len(sim_table) == 3 * 2 * (7 * 3 + 7 + 1)
    assert sim_table["level"].value_counts().to_dict() == {LEVEL_CELL: 126, LEVEL_DURATION: 42, LEVEL_CONDITION: 6}
    cells = sim_table[sim_table["level"] == LEVEL_CELL]
    assert sorted(cells["duration_ms"].unique()) == [60.0, 80.0, 100.0, 150.0, 200.0, 300.0, 400.0]
    assert sorted(cells["amplitude"].unique()) == [0.7, 0.85, 1.0]
    # 시행 수: 위상 50개, 누락 조건은 위상마다 패턴 4개
    assert set(cells.loc[cells["drop_prob"] == 0.0, "n_trials"]) == {50}
    assert set(cells.loc[cells["drop_prob"] > 0.0, "n_trials"]) == {200}
    for name, trials in (("30fps", 1050), ("15fps", 1050), ("15fps_drop20", 4200)):
        assert sim_row(sim_table, LEVEL_CONDITION, name, STAMPS_INTERPOLATED)["n_trials"] == trials
    # 합친 행은 칸들의 합이다
    for column in ("n_trials", "n_detected", "n_truncated", "n_velocity_valid"):
        merged = sim_table[sim_table["level"] == LEVEL_CONDITION][column].sum()
        assert merged == cells[column].sum() == sim_table[sim_table["level"] == LEVEL_DURATION][column].sum()
    assert (sim_table["detection_rate"] == sim_table["n_detected"] / sim_table["n_trials"]).all()
    assert np.isfinite(sim_table.loc[sim_table["n_detected"] > 0, "median_abs_error_ms"]).all()


def test_blink_sim_sanity(sim_table: pd.DataFrame) -> None:
    def condition(name: str, stamps: str = STAMPS_INTERPOLATED) -> pd.Series:
        return sim_row(sim_table, LEVEL_CONDITION, name, stamps)

    fast, slow, dropped = condition("30fps"), condition("15fps"), condition("15fps_drop20")
    # 30 fps의 오차가 15 fps보다 작고, 프레임이 빠지면 더 커진다
    assert fast["median_abs_error_ms"] < slow["median_abs_error_ms"] < dropped["median_abs_error_ms"]
    assert fast["p95_abs_error_ms"] < slow["p95_abs_error_ms"] < dropped["p95_abs_error_ms"]
    assert fast["median_abs_error_ms"] <= 2.0 and fast["p95_abs_error_ms"] <= 8.0  # 30 fps에서는 몇 ms 안
    # 30 fps에서는 50 ms보다 긴 깜빡임을 놓치지 않는다
    assert fast["detection_rate"] == 1.0
    # 300 ms 깜빡임은 규칙적인 30 fps와 15 fps에서 두 방식 모두 항상 검출된다. 400 ms도 같다.
    for name in ("30fps", "15fps"):
        for stamps in (STAMPS_INTERPOLATED, STAMPS_FRAME_TIME):
            for duration_ms in (300.0, 400.0):
                row = sim_row(sim_table, LEVEL_DURATION, name, stamps, duration_ms)
                assert row["n_detected"] == row["n_trials"] == 150, (name, stamps, duration_ms)
    # 프레임이 20%씩 빠져도 300 ms 깜빡임은 거의 다 잡힌다
    assert sim_row(sim_table, LEVEL_DURATION, "15fps_drop20", STAMPS_INTERPOLATED, 300.0)["detection_rate"] > 0.97
    # 15 fps에서 60 ms 깜빡임은 위상에 따라 놓친다(PLAN 4.5절: 약 70 ms보다 짧으면 놓칠 수 있다). 80 ms부터는 다 잡힌다.
    assert 0.1 < sim_row(sim_table, LEVEL_DURATION, "15fps", STAMPS_INTERPOLATED, 60.0)["detection_rate"] < 0.9
    assert sim_row(sim_table, LEVEL_DURATION, "15fps", STAMPS_INTERPOLATED, 80.0)["detection_rate"] == 1.0

    # 보간이 프레임 시각보다 정확하다. 프레임 시각은 길이를 표본 간격만큼 짧게 읽는다(치우침이 약 −1 간격).
    for name, period_ms in (("30fps", 1000.0 / 30.0), ("15fps", 1000.0 / 15.0)):
        interpolated, frame_time = condition(name), condition(name, STAMPS_FRAME_TIME)
        assert interpolated["median_abs_error_ms"] < frame_time["median_abs_error_ms"] / 5.0
        assert abs(interpolated["bias_ms"]) < 5.0
        assert frame_time["detection_rate"] < interpolated["detection_rate"]
        for duration_ms in (200.0, 300.0, 400.0):
            row = sim_row(sim_table, LEVEL_DURATION, name, STAMPS_FRAME_TIME, duration_ms)
            assert row["bias_ms"] == pytest.approx(-period_ms, abs=1.0), (name, duration_ms)

    # 진폭과 속도는 연속 파형보다 크게 읽힐 수 없고(표본은 파형 위의 점이다), 15 fps에서 더 작게 읽힌다
    rows = sim_table[(sim_table["stamps"] == STAMPS_INTERPOLATED) & (sim_table["n_velocity_valid"] > 0)]
    assert len(rows) >= 80
    for column in ("amplitude_ratio_mean", "closing_velocity_ratio_mean", "opening_velocity_ratio_mean"):
        assert (rows[column] <= 1.0 + 1e-9).all() and (rows[column] > 0.0).all(), column
        assert slow[column] < fast[column], column
    assert slow["closing_velocity_ratio_mean"] < slow["opening_velocity_ratio_mean"]  # 감는 쪽이 더 빠르고 더 깎인다
    # 프레임 시각 쪽에서는 속도를 재지 않는다
    frame_rows = sim_table[sim_table["stamps"] == STAMPS_FRAME_TIME]
    assert (frame_rows["n_velocity_valid"] == 0).all() and frame_rows["closing_velocity_ratio_mean"].isna().all()
    # 한 표본짜리 검출은 프레임 시각으로는 길이가 0이라 버려진다: 60 ms 깜빡임은 두 프레임률 모두에서 잡히지 않는다
    for name in ("30fps", "15fps"):
        assert sim_row(sim_table, LEVEL_DURATION, name, STAMPS_FRAME_TIME, 60.0)["n_detected"] == 0


def test_blink_sim_command_writes_identical_reports(
    tmp_path: Path, config_path: Path, repo_root: Path, capsys: pytest.CaptureFixture[str], sim_table: pd.DataFrame
) -> None:
    config = write_config(tmp_path, config_path, repo_root, "sim.yaml", reports_dir=str(tmp_path / "reports"))
    arguments = ["blink-sim", "--config", str(config), "--run-name", "sim"]
    assert signal_check.main(arguments) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == BLINK_SIM_NOTE
    assert "SIMULATION on synthetic blink" in out[0] and "not a measurement on eyes" in out[0]
    assert sum(line.startswith("[") for line in out) == 6

    run_dir = tmp_path / "reports" / "sim" / "blink_sim"
    first = read_outputs(run_dir)
    assert sorted(first) == ["blink_sim.csv", "summary.md"]
    summary = first["summary.md"].decode("utf-8")
    assert summary.splitlines()[0] == f"# {BLINK_SIM_NOTE}"
    assert PLAN_DECISION_NUMBER.findall(summary) == []  # 결정 번호를 쓰지 않는 리포트다(`PLAN_DECISION_NUMBER`의 설명)
    assert first["blink_sim.csv"].decode("utf-8").splitlines()[0] == f"# {BLINK_SIM_NOTE}"
    assert "python -m drowsy.signal_check " + " ".join(arguments) in summary
    written = pd.read_csv(run_dir / "blink_sim.csv", comment="#")
    pd.testing.assert_frame_equal(written, sim_table, check_exact=False, rtol=1e-12, check_dtype=False)

    # 같은 시드면 파일이 바이트 단위로 같다. 시드를 바꾸면 달라진다.
    assert signal_check.main(arguments) == 0
    assert read_outputs(run_dir) == first
    command = "python -m drowsy.signal_check " + " ".join(arguments)
    table = run_blink_sim(load_config(config), run_name="sim", seed=0, command=command)[1]
    assert read_outputs(run_dir) == first and len(table) == len(sim_table)
    assert signal_check.main([*arguments, "--seed", "3"]) == 0
    assert read_outputs(run_dir)["blink_sim.csv"] != first["blink_sim.csv"]
    capsys.readouterr()


def test_detector_used_by_the_simulation_is_the_ported_one() -> None:
    # 모사는 검출기를 따로 구현하지 않는다
    assert signal_check.ClosureRunDetector is ClosureRunDetector
