"""`drowsy.vectors`와 `tests/vectors/*.json` 테스트 (PLAN 10.2절, 10.1절의 규칙 경계값).

- 다시 만들면 저장된 파일과 바이트 단위로 같다.
- 저장된 expected를 프레임에서 다시 계산하면 파일에 적힌 허용 오차 안에서 같다.
- 계획서가 이름을 든 경우가 모두 있다.
- 파일은 지금의 pipeline_hash, rules_hash, 모델 파일 sha256을 담고 있다.
- 검토에서 나온 경계와 2026-11-13 소유자 결정 네 가지를 고정하는 벡터가 손으로 계산한 값을 낸다(파일 뒤쪽).
- 네 번째 검증 뒤에 더한 벡터 열셋(신호, 기준선, 스텝 특징)도 손으로 계산한 값을 낸다(파일 끝).
입력은 모두 합성 값이다(D14). 여기 나오는 수치는 파이프라인 점검용이고 성능이 아니다.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import pytest

from drowsy import blink, calibrate, signals, step_features, vectors
from drowsy.blink import ClosureRunDetector
from drowsy.calibrate import BaselineEstimator
from drowsy.spec import apply_overrides, file_sha256
from drowsy.types import (
    BASELINE_FALLBACK_GLOBAL,
    BASELINE_OFF_AXIS,
    BASELINE_OK,
    EVENT_DROWSY,
    EVENT_GAZE_AWAY_LONG,
    EVENT_MICROSLEEP,
    EVENT_SLEEP,
    EVENT_VATS,
    FEATURE_NAMES,
    PHASE_ONSET,
)
from drowsy.vectors import (
    BUNDLE_BASELINE,
    BUNDLE_E2E,
    BUNDLE_MODEL,
    BUNDLE_RULES,
    BUNDLE_RUNS,
    BUNDLE_SIGNALS,
    BUNDLE_STEPS,
    LARGE_FIRST_T_MS,
    OFFICIAL_MATRIX,
    SMALL_FIRST_T_MS,
    TOLERANCES,
    VECTOR_NAMES,
    Context,
    VectorError,
)

# 계획서가 이름을 든 경우 -> (벡터 이름, 묶음). 10.2절 표의 묶음별 내용과 10.1절의 규칙 경계값이다.
PLAN_CASES: dict[str, tuple[str, str]] = {
    # 신호
    "정사각 프레임의 EAR": ("signals_ear_square", BUNDLE_SIGNALS),
    "비정사각 프레임의 EAR": ("signals_ear_non_square", BUNDLE_SIGNALS),
    "공식 행렬의 각도": ("signals_official_matrix", BUNDLE_SIGNALS),
    "반전 행렬의 각도": ("signals_mirrored_matrix", BUNDLE_SIGNALS),
    "무작위 행렬의 각도": ("signals_random_matrices", BUNDLE_SIGNALS),
    "한쪽 눈만 감은 경우(좌우 확인)": ("signals_one_eye_closed", BUNDLE_SIGNALS),
    "폭이 0인 눈": ("signals_zero_width_eye", BUNDLE_SIGNALS),
    # 기준선과 closure
    "기준선 정상": ("baseline_normal", BUNDLE_BASELINE),
    "기준선 연장": ("baseline_extended", BUNDLE_BASELINE),
    "기준선 대체": ("baseline_fallback", BUNDLE_BASELINE),
    "yaw에 따른 눈 가중": ("closure_yaw_weighting", BUNDLE_BASELINE),
    "얼굴 없음": ("closure_no_face", BUNDLE_BASELINE),
    # 감김 검출
    "30 fps": ("runs_blinks_30fps", BUNDLE_RUNS),
    "15 fps": ("runs_blinks_15fps", BUNDLE_RUNS),
    "프레임 누락 포함": ("runs_dropped_frames", BUNDLE_RUNS),
    "한 표본짜리 깜빡임": ("runs_one_sample_blink", BUNDLE_RUNS),
    "600 ms 감김": ("runs_closure_600ms", BUNDLE_RUNS),
    "끊겨서 잘린 감김": ("runs_truncated_by_gap", BUNDLE_RUNS),
    "중복 타임스탬프": ("runs_duplicate_timestamp", BUNDLE_RUNS),
    "임계값 사이에 머무는 눈꺼풀": ("runs_lingering_lids", BUNDLE_RUNS),
    # 스텝 특징
    "5분 세션(초기 구간, 깜빡임 없는 구간, 얼굴 소실, 긴 끊김)": ("steps_five_minutes", BUNDLE_STEPS),
    "첫 t_ms가 큰 값인 세션": ("steps_five_minutes_large_t0", BUNDLE_STEPS),
    "10초 넘는 끊김 뒤 초기화": ("steps_gap_over_10s", BUNDLE_STEPS),
    # 규칙 (10.1절의 경계값)
    "감김 0.9초": ("rules_closure_900ms", BUNDLE_RULES),
    "감김 1.1초": ("rules_closure_1100ms", BUNDLE_RULES),
    "감김 2.5초": ("rules_closure_2500ms", BUNDLE_RULES),
    "감김 3.0초": ("rules_closure_3000ms", BUNDLE_RULES),
    "감김 6초": ("rules_closure_6000ms", BUNDLE_RULES),
    "시선 2.9초": ("rules_gaze_2900ms", BUNDLE_RULES),
    "시선 3.1초": ("rules_gaze_3100ms", BUNDLE_RULES),
    "얼굴이 사라지는 고개 돌림": ("rules_gaze_face_lost_in_turn", BUNDLE_RULES),
    "VATS 누적": ("rules_vats", BUNDLE_RULES),
    "모델 졸음 판정의 시작과 종료": ("rules_drowsy_model_stub", BUNDLE_RULES),
    "PERCLOS 판정": ("rules_drowsy_perclos", BUNDLE_RULES),
    "보정 중과 임시 판정": ("rules_calibrating_provisional", BUNDLE_RULES),
    "속도 기준 위아래": ("rules_speed_gates", BUNDLE_RULES),
    "재경고 대기": ("rules_cooldown", BUNDLE_RULES),
    "sleep 경고의 되풀이": ("rules_sleep_repeat", BUNDLE_RULES),
    "분석 불가": ("rules_unable_to_analyze", BUNDLE_RULES),
    # 모델, 전 과정
    "features [1,120,17] 세 개와 probs, 모델 파일 해시": ("model_v0_inputs", BUNDLE_MODEL),
    "전 과정 약 250초": ("e2e_250s", BUNDLE_E2E),
}

# 계획서의 경우에 더해 넣은 벡터 -> 묶음. 이름이 바뀌거나 빠지면 테스트가 알려 준다.
EXTRA_CASES: dict[str, str] = {
    "signals_no_face": BUNDLE_SIGNALS,
    "rules_gaze_eye_only": BUNDLE_RULES,
    # 검토 결과: 감는 속도의 거슬러 보기, 반올림과 원점, 0 나눗셈, 끊김 사이의 스텝, 규칙의 경계
    "runs_velocity_after_gap": BUNDLE_RUNS,
    "runs_rounding_large_t0": BUNDLE_RUNS,
    "steps_closure_across_gap": BUNDLE_STEPS,
    "steps_zero_division": BUNDLE_STEPS,
    "rules_gap_exactly_10s": BUNDLE_RULES,
    "rules_gap_unable_boundary": BUNDLE_RULES,
    "rules_unable_sparse_frames": BUNDLE_RULES,
    "rules_gaze_credit_window_missed": BUNDLE_RULES,
    "rules_gaze_credit_boundary": BUNDLE_RULES,
    "rules_gaze_debounce_250ms": BUNDLE_RULES,
    "rules_forward_zone_edges": BUNDLE_RULES,
    "rules_vats_reset_enabled": BUNDLE_RULES,
    "rules_alert_priority": BUNDLE_RULES,
    # 2026-11-13 결정 1: 감김의 길이는 진입 임계 이상이던 시간의 합
    "runs_multi_span": BUNDLE_RUNS,
    "runs_multi_span_truncated": BUNDLE_RUNS,
    "steps_multi_span_runs": BUNDLE_STEPS,
    "rules_closure_dwell_below_1s": BUNDLE_RULES,
    "rules_closure_dwell_reaches_1s": BUNDLE_RULES,
    "rules_closure_three_spans": BUNDLE_RULES,
    # 결정 2: 막힌 경고는 막힘이 풀렸을 때 상태가 이어지고 있으면 한 번 낸다
    "rules_alert_deferred_cooldown": BUNDLE_RULES,
    "rules_alert_blocked_microsleep_sleep": BUNDLE_RULES,
    "rules_drowsy_perclos_exact": BUNDLE_RULES,
    # 결정 3: 밀린 스텝마다 모델을 부른다
    "rules_drowsy_model_stall": BUNDLE_RULES,
    # 결정 4: 10초 넘는 끊김 뒤에는 규칙 엔진이 다시 시작한다
    "rules_gap_over_10s": BUNDLE_RULES,
    "rules_gap_over_10s_drowsy": BUNDLE_RULES,
    # 2차 검증: 단위 테스트만 잡던 동작과, 벡터도 단위 테스트도 잡지 못하던 동작
    "runs_two_humps": BUNDLE_RUNS,
    "runs_one_sample_truncated": BUNDLE_RUNS,
    "steps_gap_exactly_10s": BUNDLE_STEPS,
    "rules_gap_over_10s_candidate": BUNDLE_RULES,
    "rules_gaze_candidate_dropped": BUNDLE_RULES,
    "rules_forward_zone_eye_edges": BUNDLE_RULES,
    "rules_gaze_unable_pauses_timer": BUNDLE_RULES,
    "rules_drowsy_either": BUNDLE_RULES,
    "rules_gap_over_10s_drowsy_model": BUNDLE_RULES,
    "rules_alert_pending_survives_end": BUNDLE_RULES,
    "rules_closure_after_dropped_run": BUNDLE_RULES,
    "rules_gaze_glance_and_credit_times": BUNDLE_RULES,
    "rules_calibrating_large_t0": BUNDLE_RULES,
    # 3차 검증: 단위 테스트만 잡던 한 줄짜리 이식 오류 여섯, 벡터도 단위 테스트도 잡지 못하던 작은 첫 t_ms,
    # 그리고 그 정비에서 찾은 자리 하나(스텝 경계에서 벗어난 프레임이 내는 drowsy의 발행 시각)
    "runs_two_rises_before_peak": BUNDLE_RUNS,
    "rules_vats_window_edge": BUNDLE_RULES,
    "rules_vats_confidence_face_lost": BUNDLE_RULES,
    "rules_gaze_one_frame_glances": BUNDLE_RULES,
    "rules_microsleep_confidence_face_lost": BUNDLE_RULES,
    "rules_alert_pending_survives_onset": BUNDLE_RULES,
    "rules_unable_small_t0": BUNDLE_RULES,
    "rules_drowsy_off_step_boundary": BUNDLE_RULES,
    # 4차 검증: 신호, 기준선, 스텝 특징 모듈의 한 줄짜리 이식 오류 가운데 벡터 89개를 모두 통과하던 것
    "signals_iris_tilted_eyes": BUNDLE_SIGNALS,
    "baseline_odd_sample_count": BUNDLE_BASELINE,
    "baseline_burst_frames": BUNDLE_BASELINE,
    "baseline_exactly_300_frames": BUNDLE_BASELINE,
    "closure_blink_span_floor": BUNDLE_BASELINE,
    "steps_yawn_edges": BUNDLE_STEPS,
    "steps_nod_edges": BUNDLE_STEPS,
    "steps_nod_slow_mean": BUNDLE_STEPS,
    "steps_nod_after_long_gap": BUNDLE_STEPS,
    "steps_two_closures_in_one_step": BUNDLE_STEPS,
    "steps_closure_above_cap": BUNDLE_STEPS,
    "steps_closure_exactly_500ms": BUNDLE_STEPS,
    "steps_pitch_std_at_5000ms": BUNDLE_STEPS,
}

# 묶음마다 expected에 반드시 있어야 하는 항목
REQUIRED_SECTIONS: dict[str, tuple[str, ...]] = {
    BUNDLE_SIGNALS: ("signals",),
    BUNDLE_BASELINE: ("closure",),
    BUNDLE_RUNS: ("closure", "runs"),
    BUNDLE_STEPS: ("steps", "runs"),
    BUNDLE_RULES: ("events", "alerts"),
    BUNDLE_MODEL: ("probs",),
    BUNDLE_E2E: ("baseline", "steps", "model_inputs", "probs", "events", "alerts"),
}

GATED_TYPES: tuple[str, ...] = (EVENT_GAZE_AWAY_LONG, EVENT_VATS, EVENT_DROWSY, EVENT_MICROSLEEP, EVENT_SLEEP)


@pytest.fixture(scope="module")
def vector_dir(repo_root: Path) -> Path:
    return repo_root / "tests" / "vectors"


@pytest.fixture(scope="module")
def ctx(repo_root: Path) -> Context:
    return vectors.load_context(
        repo_root / "config" / "feature_spec.json", repo_root / "config" / "rules.yaml", repo_root / "export"
    )


@pytest.fixture(scope="module")
def stored(vector_dir: Path) -> dict[str, dict[str, Any]]:
    """저장된 벡터 전부. 이름 -> 내용. 테스트는 이 값을 고치지 않는다(고칠 때는 깊은 사본을 만든다)."""
    return {path.stem: vectors.read_vector(path) for path in vectors.vector_files(vector_dir)}


@pytest.fixture(scope="module")
def regenerated(repo_root: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """명령으로 벡터를 임시 폴더에 다시 만든다."""
    out_dir = tmp_path_factory.mktemp("vectors")
    code = vectors.main(
        [
            "generate",
            "--out",
            str(out_dir),
            "--spec",
            str(repo_root / "config" / "feature_spec.json"),
            "--rules",
            str(repo_root / "config" / "rules.yaml"),
            "--model",
            str(repo_root / "export" / "drowsiness_v0.onnx"),
        ]
    )
    assert code == 0
    return out_dir


def _speed_at(vector: dict[str, Any], t_ms: int) -> float | None:
    speed: float | None = None
    for start_ms, kmh in vector["speed"]:
        if start_ms <= t_ms:
            speed = kmh
    return speed


def _reject_constant(token: str) -> float:
    raise AssertionError(f"non-finite JSON number: {token}")


def _kinds(items: list[dict[str, Any]]) -> list[str]:
    return [f"{item['type']}:{item['phase']}" for item in items]


# ---------------------------------------------------------------------------
# 있어야 할 벡터
# ---------------------------------------------------------------------------


def test_the_directory_holds_exactly_the_named_vectors(stored: dict[str, dict[str, Any]]) -> None:
    assert sorted(stored) == sorted(VECTOR_NAMES)
    assert len(set(VECTOR_NAMES)) == len(VECTOR_NAMES)
    for name, vector in stored.items():
        assert vector["name"] == name


@pytest.mark.parametrize("case", sorted(PLAN_CASES))
def test_every_plan_case_is_present(case: str, stored: dict[str, dict[str, Any]]) -> None:
    name, bundle = PLAN_CASES[case]
    assert name in VECTOR_NAMES
    assert name in stored, f"plan case {case!r} needs tests/vectors/{name}.json"
    assert stored[name]["bundle"] == bundle
    assert len(stored[name]["description"]) > 20


def test_every_vector_belongs_to_a_plan_case_or_is_listed_as_extra(stored: dict[str, dict[str, Any]]) -> None:
    plan_names = {name for name, _ in PLAN_CASES.values()}
    assert plan_names | set(EXTRA_CASES) == set(VECTOR_NAMES)
    assert plan_names & set(EXTRA_CASES) == set()
    for name, bundle in EXTRA_CASES.items():
        assert stored[name]["bundle"] == bundle, name
        assert len(stored[name]["description"]) > 20, name


# ---------------------------------------------------------------------------
# 다시 만들기와 다시 계산하기
# ---------------------------------------------------------------------------


def test_regenerating_gives_byte_identical_files(regenerated: Path, vector_dir: Path) -> None:
    assert sorted(path.name for path in regenerated.glob("*.json")) == sorted(f"{name}.json" for name in VECTOR_NAMES)
    different = [
        name
        for name in VECTOR_NAMES
        if (regenerated / f"{name}.json").read_bytes() != (vector_dir / f"{name}.json").read_bytes()
    ]
    assert different == [], (
        "stored vectors differ from the reference implementation; "
        "if the change is intended run `python -m drowsy.vectors generate --out tests/vectors`"
    )


@pytest.mark.parametrize("name", VECTOR_NAMES)
def test_verify_recomputes_every_expected_value(name: str, stored: dict[str, dict[str, Any]], ctx: Context) -> None:
    assert vectors.verify_vector(stored[name], ctx) == []


def test_verify_dir_and_the_command_agree(regenerated: Path, ctx: Context, capsys: pytest.CaptureFixture[str]) -> None:
    results = vectors.verify_dir(regenerated, ctx)
    assert sorted(results) == sorted(VECTOR_NAMES)
    assert all(problems == [] for problems in results.values())

    code = vectors.main(["verify", "--dir", str(regenerated), "--model-dir", str(ctx.model_dir)])
    out = capsys.readouterr().out
    assert code == 0
    assert out.count("ok    ") == len(VECTOR_NAMES)
    assert f"{len(VECTOR_NAMES)} of {len(VECTOR_NAMES)} vectors verified" in out


def test_stored_files_are_in_canonical_form(vector_dir: Path, stored: dict[str, dict[str, Any]]) -> None:
    # 읽어서 다시 쓰면 같은 바이트다: 실수가 정확히 왕복하고(repr), 줄 배치가 정해져 있다.
    for name in VECTOR_NAMES:
        text = (vector_dir / f"{name}.json").read_bytes().decode("utf-8")
        assert vectors.dumps_vector(stored[name]) == text
        assert "\r" not in text and text.endswith("}\n")
        json.loads(text, parse_constant=_reject_constant)  # NaN, Infinity는 숫자로 쓰이지 않는다


# ---------------------------------------------------------------------------
# 해시, 형식, 허용 오차
# ---------------------------------------------------------------------------


def test_vectors_carry_the_current_hashes(stored: dict[str, dict[str, Any]], ctx: Context, repo_root: Path) -> None:
    model_sha256 = file_sha256(repo_root / "export" / "drowsiness_v0.onnx")
    for name, vector in stored.items():
        assert vector["pipeline_hash"] == ctx.pipeline_hash, name
        assert vector["rules_hash"] == ctx.rules_hash, name
        if vector["model"] is not None and vector["model"]["kind"] == vectors.MODEL_KIND_ONNX:
            assert vector["model"] == {"kind": "onnx", "file": "drowsiness_v0.onnx", "sha256": model_sha256}, name
    assert stored["model_v0_inputs"]["model"]["kind"] == vectors.MODEL_KIND_ONNX
    assert stored["e2e_250s"]["model"]["kind"] == vectors.MODEL_KIND_ONNX


def test_file_format(stored: dict[str, dict[str, Any]], ctx: Context) -> None:
    keys = [
        "name",
        "bundle",
        "description",
        "pipeline_hash",
        "rules_hash",
        "tolerances",
        "baseline",
        "config_overrides",
        "model",
        "frames",
        "speed",
    ]
    landmark_keys = {str(index) for index in ctx.spec.landmarks.required_indices()}
    blendshape_names = set(ctx.spec.blendshapes.required_names())
    for name, vector in stored.items():
        expected_keys = keys + (["model_inputs"] if vector["bundle"] == BUNDLE_MODEL else []) + ["expected"]
        assert list(vector) == expected_keys, name
        assert vector["bundle"] in vectors.BUNDLES
        if vector["baseline"] is not None:
            assert list(vector["baseline"]) == [field.name for field in dataclasses.fields(vectors.Baseline)]
            assert len(vector["baseline"]["r0"]) == 9
        for frame in vector["frames"]:
            assert list(frame) == ["t_ms", "face", "w", "h", "lm", "bs", "m"]
            assert isinstance(frame["t_ms"], int) and frame["t_ms"] > 0, name
            if frame["face"]:
                assert set(frame["lm"]) == landmark_keys
                assert set(frame["bs"]) == blendshape_names
                assert len(frame["m"]) == 16
                assert all(len(xy) == 2 for xy in frame["lm"].values())
            else:
                assert frame["lm"] == {} and frame["bs"] == {} and frame["m"] == []
        for start_ms, kmh in vector["speed"]:
            assert isinstance(start_ms, int) and isinstance(kmh, float)


def test_only_rules_vectors_override_the_configuration(stored: dict[str, dict[str, Any]], ctx: Context) -> None:
    overridden = {name: vector["config_overrides"] for name, vector in stored.items() if vector["config_overrides"]}
    assert overridden == {
        "rules_drowsy_model_stub": {"drowsiness.source": "model"},
        "rules_drowsy_model_stall": {"drowsiness.source": "model"},
        "rules_vats_reset_enabled": {"vats.reset_enabled": True},
        "rules_drowsy_either": {"drowsiness.source": "either"},
        "rules_gap_over_10s_drowsy_model": {
            "drowsiness.source": "model",
            "drowsiness.model.refractory_ms": 600000,
        },
    }
    assert all(stored[name]["bundle"] == BUNDLE_RULES for name in overridden)
    assert apply_overrides(ctx.rules, overridden["rules_vats_reset_enabled"]).vats.reset_enabled is True
    assert ctx.rules.vats.reset_enabled is False
    # 덮어쓴 값은 규칙 설정의 경로다. rules_hash는 덮어쓰기 전의 설정을 가리킨다.
    assert apply_overrides(ctx.rules, overridden["rules_drowsy_model_stub"]).drowsiness.source == "model"
    assert ctx.rules.drowsiness.source == "perclos"
    longer = apply_overrides(ctx.rules, overridden["rules_gap_over_10s_drowsy_model"]).drowsiness.model
    assert longer.refractory_ms == 600000 and ctx.rules.drowsiness.model.refractory_ms == 60000


def test_tolerances_follow_the_plan(stored: dict[str, dict[str, Any]]) -> None:
    assert TOLERANCES["signals"] == {"abs": 1e-6, "angle_abs_deg": 1e-3}
    assert TOLERANCES["closure"] == {"abs": 1e-6} and TOLERANCES["baseline"] == {"abs": 1e-6}
    assert TOLERANCES["runs"] == {"time_abs_ms": 1, "rel": 1e-4}
    assert TOLERANCES["steps"] == {"abs": 1e-5, "rel": 1e-5}
    assert TOLERANCES["probs"] == {"abs": 1e-4} and TOLERANCES["step_probs"] == {"abs": 1e-4}
    assert TOLERANCES["events"]["time_abs_ms"] == 1 and TOLERANCES["alerts"]["time_abs_ms"] == 1
    for name, vector in stored.items():
        sections = [section for section in vector["expected"] if section in TOLERANCES]
        assert list(vector["tolerances"]) == sections, name
        for section in sections:
            assert vector["tolerances"][section] == TOLERANCES[section], name


def test_expected_sections_per_bundle(stored: dict[str, dict[str, Any]]) -> None:
    for name, vector in stored.items():
        for section in REQUIRED_SECTIONS[vector["bundle"]]:
            assert section in vector["expected"], f"{name} needs expected.{section}"
    in_session = ("baseline_normal", "baseline_extended", "baseline_fallback", "rules_calibrating_provisional")
    hand_built = ("baseline_odd_sample_count", "baseline_burst_frames", "baseline_exactly_300_frames")
    for name in (*in_session, "rules_calibrating_large_t0", *hand_built):
        assert stored[name]["baseline"] is None and stored[name]["expected"]["baseline"] is not None
    # 긴 세션은 프레임별 신호를 저장하지 않는다.
    for name in ("steps_five_minutes", "steps_five_minutes_large_t0", "steps_gap_over_10s", "e2e_250s"):
        assert "signals" not in stored[name]["expected"] and "closure" not in stored[name]["expected"]


# ---------------------------------------------------------------------------
# verify가 차이를 잡아내는지
# ---------------------------------------------------------------------------


def test_verify_reports_wrong_hashes_and_missing_keys(stored: dict[str, dict[str, Any]], ctx: Context) -> None:
    vector = copy.deepcopy(stored["signals_no_face"])
    vector["pipeline_hash"] = "0" * 64
    vector["rules_hash"] = "1" * 64
    problems = vectors.verify_vector(vector, ctx)
    assert len(problems) == 2
    assert "pipeline_hash" in problems[0] and "rules_hash" in problems[1]

    del vector["speed"]
    assert vectors.verify_vector(vector, ctx) == ["missing keys: speed"]


def test_verify_reports_a_changed_model_file(stored: dict[str, dict[str, Any]], ctx: Context, tmp_path: Path) -> None:
    vector = copy.deepcopy(stored["model_v0_inputs"])
    vector["model"]["sha256"] = "0" * 64
    assert any("model sha256" in problem for problem in vectors.verify_vector(vector, ctx))

    elsewhere = dataclasses.replace(ctx, model_dir=tmp_path)
    assert vectors.verify_vector(stored["model_v0_inputs"], elsewhere) == [
        f"model file not found: {tmp_path / 'drowsiness_v0.onnx'}"
    ]


def test_verify_reports_changed_inputs_and_outputs(stored: dict[str, dict[str, Any]], ctx: Context) -> None:
    original = stored["rules_closure_3000ms"]

    cut_short = copy.deepcopy(original)
    del cut_short["frames"][-20:]  # 눈을 뜨는 프레임이 없어져서 end 이벤트가 나오지 않는다
    assert any("expected.events" in problem for problem in vectors.verify_vector(cut_short, ctx))

    wrong_type = copy.deepcopy(original)
    wrong_type["expected"]["events"][1]["type"] = "microsleep"
    assert any("expected.events[1].type" in problem for problem in vectors.verify_vector(wrong_type, ctx))

    missing_alert = copy.deepcopy(original)
    del missing_alert["expected"]["alerts"][-1]
    assert vectors.verify_vector(missing_alert, ctx) == ["expected.alerts: expected 1 items, got 2"]

    unknown_section = copy.deepcopy(stored["signals_no_face"])
    unknown_section["expected"]["events"] = []
    assert any("expected.events" in problem for problem in vectors.verify_vector(unknown_section, ctx))

    no_tolerance = copy.deepcopy(original)
    no_tolerance["tolerances"] = {}
    assert any("tolerances.events" in problem for problem in vectors.verify_vector(no_tolerance, ctx))

    empty = copy.deepcopy(original)
    empty["expected"] = {}
    assert vectors.verify_vector(empty, ctx) == ["expected is empty"]


@pytest.mark.parametrize(
    ("name", "path", "inside", "outside"),
    [
        # (벡터, expected 안의 경로, 허용 오차 안의 변화, 허용 오차 밖의 변화). 변화는 더하는 값이다.
        ("signals_ear_square", ("signals", 0, "ear_l"), 5e-7, 2e-6),
        ("signals_ear_square", ("signals", 0, "yaw"), 5e-4, 2e-3),
        ("signals_ear_square", ("signals", 0, "eye_look", 3), 5e-7, 2e-6),
        ("closure_no_face", ("closure", 0, "closure"), 5e-7, 2e-6),
        ("baseline_normal", ("baseline", "e_l"), 5e-7, 2e-6),
        ("runs_closure_600ms", ("runs", 0, "start_ms"), 1, 2),
        ("runs_closure_600ms", ("runs", 0, "duration_ms"), -1, -2),
        ("runs_closure_600ms", ("runs", 0, "peak_closing_velocity"), 5e-4, 2e-3),  # 값이 10이므로 상대 5e-5, 2e-4
        ("steps_gap_over_10s", ("steps", 70, "values", 5), 1e-3, 1e-2),  # 값이 약 194: 1e-5 + 1e-5·194 = 1.95e-3
        ("steps_gap_over_10s", ("steps", 70, "values", 0), 5e-6, 2e-5),
        ("model_v0_inputs", ("probs", 1, "probs", 2), 5e-5, 2e-4),
        ("rules_closure_3000ms", ("events", 1, "t_start_ms"), 1, 2),
        ("rules_closure_3000ms", ("events", 1, "duration_ms"), -1, -2),
        ("rules_closure_3000ms", ("alerts", 0, "confidence"), -5e-5, -2e-4),
    ],
)
def test_tolerances_are_applied_per_field(
    name: str,
    path: tuple[str | int, ...],
    inside: float,
    outside: float,
    stored: dict[str, dict[str, Any]],
    ctx: Context,
) -> None:
    for change, accepted in ((inside, True), (outside, False)):
        vector = copy.deepcopy(stored[name])
        node: Any = vector["expected"]
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] += change
        problems = vectors.verify_vector(vector, ctx)
        assert (problems == []) == accepted, problems


def test_exact_fields_have_no_tolerance(stored: dict[str, dict[str, Any]], ctx: Context) -> None:
    for name, path in (
        ("runs_closure_600ms", ("runs", 0, "t_ms")),
        ("steps_gap_over_10s", ("steps", 3, "k")),
        ("steps_gap_over_10s", ("steps", 3, "t_end_ms")),
        ("rules_closure_3000ms", ("events", 0, "t_ms")),
        ("rules_closure_3000ms", ("events", 0, "severity")),
        ("rules_closure_3000ms", ("counters", "frames_accepted")),
    ):
        vector = copy.deepcopy(stored[name])
        node: Any = vector["expected"]
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] += 1
        assert vectors.verify_vector(vector, ctx) != [], (name, path)

    flipped = copy.deepcopy(stored["runs_truncated_by_gap"])
    flipped["expected"]["runs"][1]["truncated"] = False
    assert vectors.verify_vector(flipped, ctx) != []
    # 불리언 자리에 정수, 실수 자리에 불리언이 와도 같은 것으로 치지 않는다.
    as_int = copy.deepcopy(stored["runs_truncated_by_gap"])
    as_int["expected"]["runs"][1]["truncated"] = 1
    assert vectors.verify_vector(as_int, ctx) != []


def test_the_command_fails_on_a_changed_or_incomplete_folder(
    stored: dict[str, dict[str, Any]], ctx: Context, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = ["verify", "--dir", str(tmp_path), "--model-dir", str(ctx.model_dir)]
    assert vectors.main(arguments) == 1
    assert "no vector files" in capsys.readouterr().err

    changed = copy.deepcopy(stored["signals_official_matrix"])
    changed["expected"]["signals"][0]["pitch"] += 0.5
    vectors.write_vectors([changed, stored["signals_no_face"]], tmp_path)
    assert vectors.main(arguments) == 1
    out = capsys.readouterr().out
    assert "FAIL  signals_official_matrix" in out and "expected.signals[0].pitch" in out
    assert "ok    signals_no_face" in out
    assert "FAIL  missing vectors:" in out

    (tmp_path / "broken.json").write_text("[1, 2", encoding="utf-8")
    renamed = tmp_path / "renamed.json"
    renamed.write_bytes((tmp_path / "signals_no_face.json").read_bytes())
    results = vectors.verify_dir(tmp_path, ctx)
    assert results["broken"][0].startswith("could not read")
    assert results["renamed"] == ["name is 'signals_no_face', file is renamed.json"]


# ---------------------------------------------------------------------------
# 벡터의 내용: 계획서가 정한 값과 경계
# ---------------------------------------------------------------------------


def test_official_matrix_angles(stored: dict[str, dict[str, Any]]) -> None:
    vector = stored["signals_official_matrix"]
    assert vector["frames"][0]["m"] == list(OFFICIAL_MATRIX)
    row = vector["expected"]["signals"][0]
    assert (row["yaw"], row["pitch"], row["roll"]) == pytest.approx((2.24, 6.32, 0.99), abs=0.005)

    plain, mirrored = stored["signals_mirrored_matrix"]["expected"]["signals"]
    assert (mirrored["yaw"], mirrored["pitch"], mirrored["roll"]) == (-plain["yaw"], plain["pitch"], -plain["roll"])
    assert (mirrored["blink_l"], mirrored["blink_r"]) == (plain["blink_r"], plain["blink_l"])


def test_non_square_frames_need_pixel_coordinates(stored: dict[str, dict[str, Any]]) -> None:
    vector = stored["signals_ear_non_square"]
    sizes = [(frame["w"], frame["h"]) for frame in vector["frames"]]
    assert sizes == [(1280, 720), (720, 1280), (568, 320)]
    for row in vector["expected"]["signals"]:
        assert (row["ear_l"], row["ear_r"]) == pytest.approx((0.31, 0.27), abs=1e-9)
    # 정규화 좌표로 구한 EAR은 프레임마다 달라진다. 벡터가 그 실수를 드러낸다.
    frame = vector["frames"][0]
    points = [frame["lm"][str(index)] for index in (362, 385, 387, 263, 373, 380)]
    assert signals.eye_aspect_ratio([(x, y) for x, y in points]) != pytest.approx(0.31, abs=0.05)


def test_baseline_vectors_cover_normal_extended_and_fallback(stored: dict[str, dict[str, Any]]) -> None:
    decided = {
        name: (stored[name]["expected"]["baseline"]["status"], stored[name]["expected"]["baseline"]["t_complete_ms"])
        for name in ("baseline_normal", "baseline_extended", "baseline_fallback")
    }
    assert decided == {
        "baseline_normal": (BASELINE_OK, 60000),
        "baseline_extended": (BASELINE_OFF_AXIS, 90000),
        "baseline_fallback": (BASELINE_FALLBACK_GLOBAL, 180000),
    }
    normal = stored["baseline_normal"]["expected"]["baseline"]
    assert normal["e_l"] != normal["e_r"] and normal["b_l"] != normal["b_r"]
    assert normal["r0"] != [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


@pytest.mark.parametrize(
    "name",
    [
        "baseline_normal",
        "baseline_extended",
        "baseline_fallback",
        "baseline_odd_sample_count",
        "baseline_burst_frames",
        "baseline_exactly_300_frames",
    ],
)
def test_baseline_vectors_work_with_the_modules_alone(
    name: str, stored: dict[str, dict[str, Any]], ctx: Context
) -> None:
    # Android 팀이 파이프라인 없이 BaselineEstimator와 frame_state만으로 이 묶음을 확인할 수 있어야 한다.
    vector = stored[name]
    estimator = BaselineEstimator(ctx.spec)
    fallback = calibrate.global_baseline(ctx.spec)
    baseline = None
    rows = vector["expected"]["closure"]
    assert len(rows) == len(vector["frames"])
    for item, row in zip(vector["frames"], rows):
        frame_signals = signals.compute(vectors.frame_from_json(item), ctx.spec)
        if baseline is None:
            baseline = estimator.update(frame_signals)
            if baseline is not None:
                assert row["t_ms"] == vector["expected"]["baseline"]["t_ms"]
        state = calibrate.frame_state(frame_signals, baseline or fallback, ctx.spec, baseline is None)
        assert dataclasses.asdict(state) == row
    assert baseline is not None
    assert {"t_ms": vector["expected"]["baseline"]["t_ms"], **vectors.baseline_to_json(baseline)} == (
        vector["expected"]["baseline"]
    )


@pytest.mark.parametrize("name", [name for name in VECTOR_NAMES if name.startswith("runs_")])
def test_run_vectors_work_with_a_detector_alone(name: str, stored: dict[str, dict[str, Any]], ctx: Context) -> None:
    # expected.closure의 (t_ms, closure, closure_valid)를 특징용 검출기에 넣으면 expected.runs가 나온다.
    vector = stored[name]
    detector = ClosureRunDetector(ctx.spec.run_detector)
    runs = []
    for row in vector["expected"]["closure"]:
        run = detector.update(row["t_ms"], row["closure"], row["closure_valid"])
        if run is not None:
            runs.append({"t_ms": row["t_ms"], **dataclasses.asdict(run)})
    assert runs == vector["expected"]["runs"]
    counters = vector["expected"]["counters"]
    assert counters["frames_accepted"] == len(vector["expected"]["closure"])
    assert counters["frames_accepted"] + counters["dropped_frames"] == len(vector["frames"])


def test_run_vectors_cover_the_named_situations(stored: dict[str, dict[str, Any]]) -> None:
    def runs(name: str) -> list[dict[str, Any]]:
        return stored[name]["expected"]["runs"]

    def frame_gaps(name: str) -> list[int]:
        times = [frame["t_ms"] for frame in stored[name]["frames"]]
        return [later - earlier for earlier, later in zip(times, times[1:])]

    assert set(frame_gaps("runs_blinks_30fps")) == {33, 34}
    assert set(frame_gaps("runs_blinks_15fps")) == {66, 67}
    assert max(frame_gaps("runs_dropped_frames")) > 100
    assert len(runs("runs_blinks_30fps")) == 3 and len(runs("runs_blinks_15fps")) == 2

    one_sample = stored["runs_one_sample_blink"]
    assert len(runs("runs_one_sample_blink")) == 1
    closed = [row for row in one_sample["expected"]["closure"] if row["closure"] >= 0.5]
    assert len(closed) == 2  # 한 표본짜리 깜빡임 둘 가운데 하나만 구간이 된다

    assert runs("runs_closure_600ms")[0]["duration_ms"] == 600
    assert [run["truncated"] for run in runs("runs_truncated_by_gap")] == [False, True, False, True, False]
    assert all(not run["velocity_valid"] for run in runs("runs_truncated_by_gap") if run["truncated"])

    assert min(frame_gaps("runs_duplicate_timestamp")) < 0 and 0 in frame_gaps("runs_duplicate_timestamp")
    assert stored["runs_duplicate_timestamp"]["expected"]["counters"]["dropped_frames"] == 2

    lingering = [row["closure"] for row in stored["runs_lingering_lids"]["expected"]["closure"]]
    assert sum(1 for value in lingering if 0.3 < value < 0.5) > 40
    assert len(runs("runs_lingering_lids")) == 2
    # 첫 구간은 0.4에 머문 구간을 건너 이어진다. 길이는 0.5 이상이던 두 토막의 합이고 머문 시간은 들어가지 않는다.
    # 각본(세션 시간): 첫 감김은 1500에 감기 시작해 100 ms 만에 닫히고 1900부터 150 ms에 걸쳐 뜬다. 0.5를 1550에 위로,
    # 1975에 아래로 지난다(425 ms). 둘째 감김(2400~)은 2450에 위로, 2775에 아래로 지난다(325 ms). 그 사이는 0.4다.
    # 검출기의 값은 0.4로 꺾이는 자리를 건너는 두 쌍의 선형 보간 오차만큼(각각 4 ms 이내) 다르다.
    across = runs("runs_lingering_lids")[0]
    assert (across["start_ms"], across["end_ms"]) == (1 + 1550, 1 + 2775)
    assert across["end_ms"] - across["start_ms"] == 1225
    assert abs(across["duration_ms"] - (425 + 325)) <= 8 and across["duration_ms"] == 757


def test_five_minute_session_and_its_shifted_copy(stored: dict[str, dict[str, Any]]) -> None:
    plain = stored["steps_five_minutes"]
    shifted = stored["steps_five_minutes_large_t0"]
    offset = LARGE_FIRST_T_MS - 1
    assert LARGE_FIRST_T_MS > 2**31
    assert plain["frames"][0]["t_ms"] == 1 and shifted["frames"][0]["t_ms"] == LARGE_FIRST_T_MS
    assert plain["frames"][-1]["t_ms"] - plain["frames"][0]["t_ms"] > 299_000

    steps = plain["expected"]["steps"]
    assert [step["k"] for step in steps] == list(range(1, 300))
    assert [step["full_lookback"] for step in steps] == [False] * 59 + [True] * 240
    assert all(len(step["values"]) == len(FEATURE_NAMES) for step in steps)
    assert all(math.isfinite(value) for step in steps for value in step["values"])

    # 큰 t_ms: 스텝 번호와 값은 같고 시각만 밀린다.
    for step, moved in zip(steps, shifted["expected"]["steps"], strict=True):
        assert moved == {**step, "t_ms": step["t_ms"] + offset, "t_end_ms": step["t_end_ms"] + offset}
    for run, moved in zip(plain["expected"]["runs"], shifted["expected"]["runs"], strict=True):
        assert moved["start_ms"] == run["start_ms"] + offset and moved["end_ms"] == run["end_ms"] + offset
        assert moved["amplitude"] == run["amplitude"] and moved["duration_ms"] == run["duration_ms"]

    # 긴 끊김: 프레임 하나가 스텝 여러 개를 발행한다. 얼굴 소실: 잘린 감김.
    emitted_by: dict[int, int] = {}
    for step in steps:
        emitted_by[step["t_ms"]] = emitted_by.get(step["t_ms"], 0) + 1
    assert max(emitted_by.values()) >= 7
    assert any(run["truncated"] for run in plain["expected"]["runs"])
    face_frames = [frame["face"] for frame in plain["frames"]]
    assert 50 < face_frames.count(False) < 90

    # 얼굴이 10초 넘게 없으면 10초 구간의 비율은 정의 불가 값 0이다(0으로 나누지 않는다. PLAN 3.3절, 4.6절).
    names = list(FEATURE_NAMES)
    no_face = [step["values"] for step in steps if step["values"][names.index("face_missing_ratio_10")] == 1.0]
    assert len(no_face) >= 2
    for values in no_face:
        assert values[names.index("perclos_10")] == 0.0 and values[names.index("closure_mean_10")] == 0.0
    # 유효 시간이 2초에 못 미치면 perclos_10은 0이고, closure_mean_10은 남은 유효 시간으로 구한다.
    assert any(
        values[names.index("perclos_10")] == 0.0 and values[names.index("closure_mean_10")] > 0.9
        for values in (step["values"] for step in steps)
    )


def test_gap_over_ten_seconds_restarts_the_accumulation(stored: dict[str, dict[str, Any]]) -> None:
    vector = stored["steps_gap_over_10s"]
    times = [frame["t_ms"] for frame in vector["frames"]]
    assert max(later - earlier for earlier, later in zip(times, times[1:])) > 10_000
    steps = vector["expected"]["steps"]
    assert [step["k"] for step in steps] == list(range(1, 64)) + list(range(77, 87))
    assert [step["full_lookback"] for step in steps] == [False] * 59 + [True] * 4 + [False] * 10


def test_constant_pitch_exercises_the_variance_clip(
    stored: dict[str, dict[str, Any]], ctx: Context, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 고개를 숙인 채 가만히 있으면 pitch의 분산이 반올림 오차로 음수가 된다. 0 이상으로 자르지 않으면 NaN이 나온다.
    vector = stored["steps_gap_over_10s"]
    variances: list[float] = []
    original = step_features._std_from_moments

    def recording(mean: float, mean_sq: float) -> float:
        variances.append(mean_sq - mean * mean)
        return original(mean, mean_sq)

    monkeypatch.setattr(step_features, "_std_from_moments", recording)
    steps = vectors.compute_expected(vector, ctx)["steps"]
    assert steps == vector["expected"]["steps"]
    assert sum(1 for variance in variances if variance < 0.0) > 50
    index = list(FEATURE_NAMES).index("pitch_std_60")
    assert {step["values"][index] for step in steps} == {0.0}
    assert all(frame["m"][9] != 0.0 for frame in vector["frames"])  # 행렬의 pitch 성분이 0이 아니다


def test_closure_rule_boundaries(stored: dict[str, dict[str, Any]]) -> None:
    def events(name: str) -> list[dict[str, Any]]:
        return stored[name]["expected"]["events"]

    def alerts(name: str) -> list[dict[str, Any]]:
        return stored[name]["expected"]["alerts"]

    assert events("rules_closure_900ms") == [] and alerts("rules_closure_900ms") == []
    for name in ("rules_closure_1100ms", "rules_closure_2500ms"):
        assert _kinds(events(name)) == ["microsleep:onset", "microsleep:end"]
        assert 1000 <= events(name)[0]["duration_ms"] < 1040
    assert 1080 < events("rules_closure_1100ms")[1]["duration_ms"] < 1250
    assert 2480 < events("rules_closure_2500ms")[1]["duration_ms"] < 2650

    at_three = events("rules_closure_3000ms")
    assert _kinds(at_three) == ["microsleep:onset", "sleep:onset", "sleep:end"]
    assert at_three[1]["duration_ms"] == 3000  # "3.0초 이상"의 경계 (D2)
    assert at_three[1]["t_start_ms"] == at_three[0]["t_start_ms"] == at_three[2]["t_start_ms"]
    # 감김의 마지막 감긴 프레임이 바로 그 프레임이다: "초과"로 구현하면 sleep이 나지 않는다.
    assert at_three[2]["t_ms"] - at_three[1]["t_ms"] == 35

    six = alerts("rules_closure_6000ms")
    assert _kinds(six) == ["microsleep:onset", "sleep:onset", "sleep:onset"]
    assert six[1]["duration_ms"] == 3000 and six[2]["duration_ms"] == 6010
    assert _kinds(events("rules_closure_6000ms")) == ["microsleep:onset", "sleep:onset", "sleep:end"]

    repeated = [alert for alert in alerts("rules_sleep_repeat") if alert["type"] == EVENT_SLEEP]
    assert [alert["duration_ms"] - repeated[0]["duration_ms"] for alert in repeated] == [0, 3000, 6000]
    assert _kinds(alerts("rules_cooldown")) == ["microsleep:onset", "microsleep:onset"]
    assert _kinds(events("rules_cooldown")) == ["microsleep:onset", "microsleep:end"] * 3


def test_gaze_rule_boundaries(stored: dict[str, dict[str, Any]]) -> None:
    assert stored["rules_gaze_2900ms"]["expected"]["events"] == []
    onset, end = stored["rules_gaze_3100ms"]["expected"]["events"]
    assert (onset["type"], onset["phase"], onset["source"]) == (EVENT_GAZE_AWAY_LONG, PHASE_ONSET, "head_pose")
    assert onset["duration_ms"] == 3000 and 3100 < end["duration_ms"] < 3500

    assert {event["source"] for event in stored["rules_gaze_eye_only"]["expected"]["events"]} == {"eye"}

    lost = stored["rules_gaze_face_lost_in_turn"]
    onset = lost["expected"]["events"][0]
    frame_at_onset = next(frame for frame in lost["frames"] if frame["t_ms"] == onset["t_ms"])
    assert not frame_at_onset["face"]  # 얼굴이 없는 동안 임계 시간에 도달한다
    assert onset["duration_ms"] == 3000 and onset["confidence"] < 0.5

    vats = stored["rules_vats"]["expected"]
    assert _kinds(vats["events"]) == ["vats:onset", "vats:end"]
    assert _kinds(vats["alerts"]) == ["vats:onset"]


def test_drowsy_rule_vectors(stored: dict[str, dict[str, Any]], ctx: Context) -> None:
    stub = stored["rules_drowsy_model_stub"]
    assert stub["model"]["kind"] == vectors.MODEL_KIND_STUB
    probs = stub["expected"]["probs"]
    assert probs[0]["k"] == 179 and [entry["k"] for entry in probs] == list(range(179, 267))
    # 스텝마다 하나씩: 모델을 부르지 않은 스텝은 null, 부른 스텝은 probs 항목과 같은 값이다(이 벡터는 프레임마다 스텝이 하나다).
    step_probs = stub["expected"]["step_probs"]
    assert [entry["k"] for entry in step_probs] == list(range(1, 267))
    assert all(entry["probs"] is None for entry in step_probs[:178])
    assert step_probs[178:] == probs
    scripted = stub["model"]["probs"]
    for index, entry in enumerate(probs):  # i번째 호출에 probs[i], 목록이 끝나면 마지막 값
        assert entry["probs"] == scripted[min(index, len(scripted) - 1)]
    first_on, end, second_on = stub["expected"]["events"]
    assert (first_on["source"], first_on["confidence"]) == ("model", 0.7)
    assert first_on["t_start_ms"] == 1 + 182_000 and first_on["duration_ms"] == 9000
    assert end["confidence"] == 0.3
    assert second_on["t_ms"] - end["t_ms"] == ctx.rules.drowsiness.model.refractory_ms
    assert _kinds(stub["expected"]["alerts"]) == ["drowsy:onset"]

    perclos = stored["rules_drowsy_perclos"]["expected"]
    assert _kinds(perclos["events"]) == ["drowsy:onset", "drowsy:end"]
    assert {event["source"] for event in perclos["events"]} == {"perclos"}
    assert perclos["alerts"] == []  # 40 km/h: 졸음 경고의 기준(50 km/h) 미만


def test_calibrating_and_unable_to_analyze_vectors(stored: dict[str, dict[str, Any]]) -> None:
    calibrating = stored["rules_calibrating_provisional"]["expected"]
    events = calibrating["events"]
    assert _kinds(events)[:4] == ["calibrating:onset", "microsleep:onset", "microsleep:end", "calibrating:end"]
    assert [event["provisional"] for event in events[:4]] == [True, True, True, False]
    assert events[3]["duration_ms"] == 60000 and calibrating["baseline"]["status"] == BASELINE_OK
    assert all(not event["provisional"] for event in events[4:])
    assert [event["type"] for event in events[4:]] == [EVENT_MICROSLEEP] * 2 + [EVENT_GAZE_AWAY_LONG] * 2
    assert calibrating["alerts"][0]["provisional"]

    unable = stored["rules_unable_to_analyze"]["expected"]
    assert _kinds(unable["events"]) == [
        "unable_to_analyze:onset",
        "unable_to_analyze:end",
        "microsleep:onset",
        "microsleep:end",
    ]
    assert _kinds(unable["alerts"]) == ["unable_to_analyze:onset", "microsleep:onset"]


def test_speed_gates_are_covered_below_and_above_for_every_gated_type(
    stored: dict[str, dict[str, Any]], ctx: Context
) -> None:
    gates = {event_type: ctx.rules.events.get(event_type).min_speed_kmh for event_type in GATED_TYPES}
    assert all(gate > 0.0 for gate in gates.values())
    suppressed_below: set[str] = set()
    alerted_above: set[str] = set()
    alerted_without_speed: set[str] = set()
    for vector in stored.values():
        if "events" not in vector["expected"]:
            continue
        alerts = {(alert["t_ms"], alert["type"], alert["t_start_ms"]) for alert in vector["expected"]["alerts"]}
        for event in vector["expected"]["events"]:
            if event["phase"] != PHASE_ONSET or event["type"] not in gates:
                continue
            speed = _speed_at(vector, event["t_ms"])
            alerted = (event["t_ms"], event["type"], event["t_start_ms"]) in alerts
            if speed is None and alerted:
                alerted_without_speed.add(event["type"])
            elif speed is not None and speed < gates[event["type"]] and not alerted:
                suppressed_below.add(event["type"])
            elif speed is not None and speed >= gates[event["type"]] and alerted:
                alerted_above.add(event["type"])
    assert suppressed_below == set(GATED_TYPES)
    assert alerted_above == set(GATED_TYPES)
    assert {EVENT_MICROSLEEP, EVENT_SLEEP, EVENT_GAZE_AWAY_LONG} <= alerted_without_speed  # 속도 입력이 없으면 억제하지 않는다

    # 20과 50 km/h 사이의 속도는 졸음 경고만 억제한다.
    assert _speed_at(stored["rules_drowsy_perclos"], 1) == 40.0
    late_sleep = stored["rules_speed_gates"]["expected"]["alerts"][1]
    assert late_sleep["type"] == EVENT_SLEEP and _speed_at(stored["rules_speed_gates"], late_sleep["t_ms"] - 1) == 10.0


def test_model_bundle(stored: dict[str, dict[str, Any]], ctx: Context) -> None:
    vector = stored["model_v0_inputs"]
    inputs = vector["model_inputs"]
    assert len(inputs) == 3
    assert all(len(window) == 120 and all(len(row) == 17 for row in window) for window in inputs)
    assert vector["frames"] == []
    rows = vector["expected"]["probs"]
    assert [row["input"] for row in rows] == [0, 1, 2]
    for row in rows:
        assert sum(row["probs"]) == pytest.approx(1.0, abs=1e-5)
    # 입력 1은 전 과정 벡터의 첫 창이고 같은 확률을 낸다.
    e2e = stored["e2e_250s"]["expected"]
    assert inputs[1] == e2e["model_inputs"][0]["features"]
    assert rows[1]["probs"] == pytest.approx(e2e["probs"][0]["probs"], abs=1e-6)
    # 스텝 순서를 뒤집으면 다른 확률이 나온다: 순서를 틀리면 드러난다.
    reversed_probs = vectors.compute_expected({**vector, "model_inputs": [inputs[2][::-1]]}, ctx)["probs"][0]["probs"]
    assert max(abs(a - b) for a, b in zip(reversed_probs, rows[2]["probs"])) > 1e-3


def test_end_to_end_vector(stored: dict[str, dict[str, Any]], ctx: Context) -> None:
    vector = stored["e2e_250s"]
    expected = vector["expected"]
    times = [frame["t_ms"] for frame in vector["frames"]]
    assert 249_000 < times[-1] - times[0] < 251_000
    assert vector["baseline"] is None and expected["baseline"]["t_complete_ms"] == 60000

    steps = {step["k"]: step for step in expected["steps"]}
    assert sorted(steps) == list(range(61, 251))
    assert min(k for k, step in steps.items() if step["full_lookback"]) == 120

    # 120스텝 창: features[0]이 가장 오래된 스텝(k − 119), features[119]가 가장 최근 스텝(k)이다.
    window = expected["model_inputs"][0]
    assert window["k"] == 239 and len(window["features"]) == ctx.window_steps
    assert window["features"] == [steps[k]["values"] for k in range(120, 240)]
    assert window["features"][0] != window["features"][-1]
    assert [entry["k"] for entry in expected["probs"]] == list(range(239, 251))
    assert expected["probs"][0]["t_ms"] == window["t_ms"] == 1 + 239_000
    assert [entry["k"] for entry in expected["step_probs"]] == sorted(steps)
    assert [entry for entry in expected["step_probs"] if entry["probs"] is not None] == expected["probs"]

    assert _kinds(expected["events"]) == [
        "calibrating:onset",
        "calibrating:end",
        "microsleep:onset",
        "microsleep:end",
        "drowsy:onset",
        "drowsy:end",
    ]
    drowsy_on, drowsy_end = expected["events"][4:]
    assert drowsy_on["source"] == drowsy_end["source"] == "perclos"  # v0은 무작위 가중치라 모델로는 판정이 나지 않는다
    assert drowsy_on["t_start_ms"] == steps[120]["t_end_ms"] and drowsy_on["t_ms"] == steps[129]["t_end_ms"]
    assert _kinds(expected["alerts"]) == ["microsleep:onset", "drowsy:onset"]
    assert expected["counters"]["model_calls"] == 12 and expected["counters"]["nonfinite_windows"] == 0


# ---------------------------------------------------------------------------
# 다른 기기에서도 같은 결과가 나오게 하는 장치
# ---------------------------------------------------------------------------


def test_interpolated_times_stay_clear_of_the_rounding_boundary(ctx: Context, monkeypatch: pytest.MonkeyPatch) -> None:
    # 보간한 시각이 x.5 ms에 붙어 있으면 수학 함수의 마지막 자리 차이로 반올림이 갈린다. 그런 벡터는 두지 않는다.
    distances: list[float] = []
    original = blink._round_half_up

    def recording(x: float) -> int:
        distances.append(abs(x - math.floor(x) - 0.5))
        return original(x)

    monkeypatch.setattr(blink, "_round_half_up", recording)
    built = vectors.build_vectors(ctx)
    assert len(built) == len(VECTOR_NAMES)
    assert len(distances) > 200
    assert min(distances) > 1e-6


def test_samples_on_a_threshold_are_rejected(ctx: Context) -> None:
    row = {
        "t_ms": 1,
        "face": True,
        "closure": 0.1,
        "closure_valid": True,
        "provisional": False,
        "yaw_rel": 0.0,
        "pitch_rel": 0.0,
        "pose_valid": True,
        "gaze_h_rel": 0.0,
        "gaze_v_rel": 0.0,
        "jaw_rel": 0.0,
    }
    assert vectors._threshold_problem([row], ctx) is None
    for field, value in (
        ("closure", 0.5 - 1e-12),
        ("closure", 0.3),
        ("closure", 0.7 + 1e-9),
        ("closure", 0.8),
        ("yaw_rel", -40.0),
        ("yaw_rel", 23.0 + 1e-8),
        ("pitch_rel", 18.0),
        ("pitch_rel", -25.0),
        ("gaze_h_rel", -0.25),
        ("gaze_v_rel", 0.35),
        ("jaw_rel", 0.4),
        ("jaw_rel", 0.25),
    ):
        problem = vectors._threshold_problem([{**row, field: value}], ctx)
        assert problem is not None and "threshold" in problem, (field, value)
    # 유효하지 않은 값은 판정에 쓰이지 않으므로 보지 않는다.
    assert vectors._threshold_problem([{**row, "closure": 0.5, "closure_valid": False}], ctx) is None
    assert vectors._threshold_problem([{**row, "pitch_rel": 18.0, "pose_valid": False}], ctx) is None

    # 저장된 벡터는 모두 이 검사를 지난다(만들 때 확인한다). 임계값에 붙은 표본이 있으면 생성이 실패한다.
    lids = vectors.synth.lid_signals(0.5, 0.27, 0.22, ctx.spec)
    half_closed = vectors._face_signals(
        1, ear_l=lids[0], ear_r=lids[0], blink_l=lids[1], blink_r=lids[1], yaw=0.0, pitch=0.0, roll=0.0
    )
    on_threshold = vectors.synth.frame_from_signals(half_closed, 640, 480, ctx.spec)
    case = vectors._Case(
        name="on_threshold",
        bundle=BUNDLE_BASELINE,
        description="closure 0.5",
        sections=("closure",),
        frames=(on_threshold,),
        baseline=vectors._personal_baseline(ctx.spec),
    )
    with pytest.raises(VectorError, match="sits on the threshold 0.5"):
        vectors._build_vector(case, ctx)


def test_generation_fails_when_a_script_misses_its_intent(ctx: Context) -> None:
    case = vectors._Case(
        name="no_closure",
        bundle=BUNDLE_RULES,
        description="눈을 뜬 프레임뿐인데 microsleep을 기대한다",
        sections=("events", "alerts"),
        frames=vectors._render(ctx.spec, [], 500),
        baseline=vectors._personal_baseline(ctx.spec),
        events=("microsleep:onset", "microsleep:end"),
    )
    with pytest.raises(VectorError, match="no_closure: events are"):
        vectors._build_vector(case, ctx)

    with pytest.raises(VectorError, match="frame times must be positive"):
        vectors._build_vector(
            dataclasses.replace(case, events=None, frames=vectors._render(ctx.spec, [], 500, first_t_ms=0)), ctx
        )


def test_frame_json_round_trip(stored: dict[str, dict[str, Any]]) -> None:
    for name in ("signals_one_eye_closed", "runs_truncated_by_gap"):
        for item in stored[name]["frames"]:
            frame = vectors.frame_from_json(item)
            assert vectors.frame_to_json(frame) == item
            assert json.loads(json.dumps(item)) == item
    baseline = stored["closure_yaw_weighting"]["baseline"]
    assert vectors.baseline_to_json(vectors.baseline_from_json(baseline)) == baseline


def test_unknown_model_kind_and_bundle_are_rejected(stored: dict[str, dict[str, Any]], ctx: Context) -> None:
    vector = copy.deepcopy(stored["rules_closure_900ms"])
    vector["model"] = {"kind": "tflite"}
    assert any("unknown model kind" in problem for problem in vectors.verify_vector(vector, ctx))
    vector["model"] = {"kind": "stub", "probs": []}
    assert any("stub model needs" in problem for problem in vectors.verify_vector(vector, ctx))
    vector["model"] = None
    vector["bundle"] = "other"
    assert any("unknown bundle" in problem for problem in vectors.verify_vector(vector, ctx))
    vector["bundle"] = BUNDLE_RULES
    vector["speed"] = [[500, 30.0], [100, 10.0]]
    assert any("sorted by t_ms" in problem for problem in vectors.verify_vector(vector, ctx))


# ---------------------------------------------------------------------------
# 검토에서 나온 경계와 2026-11-13 소유자 결정: 손으로 계산한 값
# ---------------------------------------------------------------------------
#
# 아래의 벡터는 프레임 시각과 값을 손으로 놓았다(t_ms = 세션 시간 + 1, 값은 계단 모양). 기대값은 규칙을 따라 손으로
# 계산한 것이고 저장된 expected와 비교한다. 감김 0.0 -> 0.9의 계단에서 임계를 지나는 자리(간격 100 ms 기준):
#   특징용(진입 0.5): 0.0 -> 0.9는 56 ms(0.5 / 0.9), 0.9 -> 0.0은 44 ms, 0.4 -> 0.9는 20 ms, 0.9 -> 0.4는 80 ms
#   규칙용(진입 0.7): 0.0 -> 0.9는 78 ms(0.7 / 0.9), 0.9 -> 0.0은 22 ms, 0.6 -> 0.9는 33 ms, 0.9 -> 0.6은 67 ms


def _timed(items: list[dict[str, Any]]) -> list[tuple[str, str, int]]:
    """(종류, 단계, 세션 시간). 손으로 놓은 벡터의 첫 t_ms는 1이다."""
    return [(item["type"], item["phase"], item["t_ms"] - 1) for item in items]


def _run_fields(run: dict[str, Any]) -> tuple[int, int, int, bool, bool]:
    return (run["start_ms"], run["end_ms"], run["duration_ms"], run["truncated"], run["velocity_valid"])


def test_step_probs_hold_one_entry_per_emitted_step(stored: dict[str, dict[str, Any]], ctx: Context) -> None:
    with_step_probs = sorted(name for name, vector in stored.items() if "step_probs" in vector["expected"])
    assert with_step_probs == [
        "e2e_250s",
        "rules_drowsy_either",
        "rules_drowsy_model_stall",
        "rules_drowsy_model_stub",
        "rules_gap_over_10s_drowsy_model",
    ]
    for name in with_step_probs:
        expected = stored[name]["expected"]
        steps = vectors.compute_expected(stored[name], ctx)["steps"]
        assert [(entry["t_ms"], entry["k"]) for entry in expected["step_probs"]] == [
            (step["t_ms"], step["k"]) for step in steps
        ], name
        # probs는 프레임마다 마지막 스텝의 확률이다: 같은 t_ms의 step_probs 가운데 마지막 것이 null이 아닐 때만 있다.
        last_of_frame: dict[int, dict[str, Any]] = {}
        for entry in expected["step_probs"]:
            last_of_frame[entry["t_ms"]] = entry
        assert expected["probs"] == [entry for entry in last_of_frame.values() if entry["probs"] is not None], name
        assert expected["counters"]["model_calls"] == sum(
            1 for entry in expected["step_probs"] if entry["probs"] is not None
        )


def test_velocity_lookback_stops_at_an_unusable_pair(stored: dict[str, dict[str, Any]]) -> None:
    runs = stored["runs_velocity_after_gap"]["expected"]["runs"]
    assert [_run_fields(run) for run in runs] == [
        (33, 121, 88, True, False),  # 0.1 -> 0.6의 40 ms 중 32 ms에 0.5를 지난다. 640 ms 끊겨 잘린다(끝은 120의 프레임)
        (761, 821, 60, False, True),  # 끊김 뒤 첫 표본에서 시작. 0.92 -> 0.1의 40 ms 중 20 ms(0.42 / 0.82)에 내려간다
        (11001, 11061, 60, False, False),  # 9초 끊김 뒤 첫 표본이 감긴 눈
        (13301, 13361, 60, False, False),  # 무효 프레임 여덟을 사이에 둔 267 ms 간격
        (15041, 15161, 120, False, True),  # 0.45 -> 0.9의 67 ms 중 7 ms(0.05 / 0.45)에 0.5를 지난다
    ]
    velocities = [(run["peak_closing_velocity"], run["peak_opening_velocity"]) for run in runs]
    assert velocities[0] == (0.0, 0.0) and velocities[2] == (0.0, 0.0) and velocities[3] == (0.0, 0.0)
    # 끊김 뒤의 구간: 구간 안의 쌍 0.90 -> 0.92(40 ms)뿐이다. 잘린 앞 구간의 증가율(0.6 -> 0.82: 5.5/s)이 아니다.
    assert velocities[1] == pytest.approx((0.02 / 0.040, 0.82 / 0.040), rel=1e-9)
    # 무효 프레임 하나를 건넌 쌍(67 ms)은 쓸 수 있는 쌍이라 진입 두 표본 전의 증가율(0.35 / 33 ms)까지 거슬러 본다.
    assert velocities[4] == pytest.approx((0.35 / 0.033, 0.8 / 0.040), rel=1e-9)


def test_interpolated_times_do_not_depend_on_the_time_origin(stored: dict[str, dict[str, Any]], ctx: Context) -> None:
    vector = stored["runs_rounding_large_t0"]
    (run,) = vector["expected"]["runs"]
    origin = LARGE_FIRST_T_MS
    assert vector["frames"][0]["t_ms"] == origin
    # 0.05 -> 0.95의 33 ms 중 16.49993 ms에 0.5를 지나고(반올림 16), 0.95 -> 0.4045의 40 ms 중 33 ms에 내려간다.
    assert (run["start_ms"], run["end_ms"], run["duration_ms"]) == (origin + 100 + 16, origin + 133 + 33, 50)
    rows = {row["t_ms"] - origin: row["closure"] for row in vector["expected"]["closure"]}
    assert (rows[100], rows[133], rows[173]) == pytest.approx((0.05, 0.95, 0.4045), abs=1e-4)
    offset = (0.5 - rows[100]) / (rows[133] - rows[100]) * 33
    assert 16.4999 < offset < 16.49995 and math.floor(offset + 0.5) == 16
    # 절대 시각을 더한 뒤 반올림하면 이 크기의 Double에서는 합이 x.5가 되어 17이 나온다(구간이 49 ms로 버려진다).
    absolute = math.floor(float(origin + 100) + offset + 0.5)
    assert absolute - (origin + 100) == 17
    # closure가 허용 오차(1e-6)만큼 달라도 오프셋은 16.5에 닿지 않는다(0.45 / 0.81 x 33 x 1e-6 x 2 = 3.7e-5 ms).
    worst = (0.5 - (rows[100] + 1e-6)) / ((rows[133] - 1e-6) - (rows[100] + 1e-6)) * 33
    assert max(offset, worst) < 16.5 - 1e-5 and math.floor(worst + 0.5) == 16
    # 같은 표본을 원점 1에서 넣으면 같은 구간이 나온다.
    detector = ClosureRunDetector(ctx.spec.run_detector)
    shifted = [
        detector.update(row["t_ms"] - origin + 1, row["closure"], row["closure_valid"])
        for row in vector["expected"]["closure"]
    ]
    (moved,) = [item for item in shifted if item is not None]
    assert (moved.start_ms, moved.end_ms, moved.duration_ms) == (1 + 116, 1 + 166, 50)


def test_multi_span_run_vectors(stored: dict[str, dict[str, Any]]) -> None:
    three, held = stored["runs_multi_span"]["expected"]["runs"]
    # 토막 셋: 456~780, 1120~1380, 1920~2244 (세션 시간). 0.4에 머문 340 ms와 540 ms는 들어가지 않는다.
    assert _run_fields(three) == (457, 2245, 324 + 260 + 324, False, True)
    assert three["end_ms"] - three["start_ms"] == 1788
    # 토막 둘: 2956~3680(얼굴이 400 ms 사라진 끊김은 앞뒤가 모두 감긴 눈이라 토막 안이다), 4100~4244(HOLD에서
    # 돌아온 프레임은 직전 유효 표본과 400 ms 떨어져 보간하지 못하므로 그 프레임의 시각에서 시작한다)
    assert _run_fields(held) == (2957, 4245, 724 + 144, False, True)

    cut = stored["runs_multi_span_truncated"]["expected"]["runs"]
    assert [(*_run_fields(run), run["t_ms"]) for run in cut] == [
        # 토막 456~680 뒤 920에서 다시 감겼고, 마지막 감긴 프레임은 1100이다. 1100에서 500 ms 넘게 지난 1700에서 잘린다
        (457, 1101, 224 + 180, True, False, 1701),
        # 둘째 토막 3320~ 뒤 0.4로 내려간 표본(3600, 3700)을 봤어도 끝은 마지막 감긴 프레임 3500이다
        (2957, 3501, 224 + 180, True, False, 4301),
        # 5300의 감긴 프레임 뒤 무효 프레임 하나, 5500에서 눈을 떴다
        (4957, 5301, 224 + 80, True, False, 5501),
    ]


def test_step_features_use_the_sum_of_the_closed_spans(stored: dict[str, dict[str, Any]]) -> None:
    expected = stored["steps_multi_span_runs"]["expected"]
    # 구간 A 956~1944(토막 224 + 224), B 3956~5144(324 + 260 + 324), C 6956~9744(424 + 224)
    assert [_run_fields(run)[:3] for run in expected["runs"]] == [
        (957, 1945, 448),
        (3957, 5145, 908),
        (6957, 9745, 648),
    ]
    names = list(FEATURE_NAMES)
    steps = {step["k"]: dict(zip(names, step["values"])) for step in expected["steps"]}
    assert sorted(steps) == list(range(1, 13))
    # 스텝 3: 구간 A가 이벤트 하나. 448 ms는 긴 감김(500 ms 이상)이 아니다
    assert (steps[3]["blink_dur_mean_60"], steps[3]["long_closure_count_60"], steps[3]["max_closure_ms_60"]) == (
        448.0,
        0.0,
        448.0,
    )
    # 스텝 5: 구간 B가 진행 중. 4900의 프레임까지 324 + 260 + (4900 − 4820) = 664 ms
    assert (steps[5]["long_closure_count_60"], steps[5]["max_closure_ms_60"], steps[5]["max_closure_ms_10"]) == (
        1.0,
        664.0,
        664.0,
    )
    assert steps[5]["blink_rate_60"] == 1 * 60000.0 / 5000.0 and steps[5]["blink_dur_mean_60"] == 448.0
    # 스텝 6: 이벤트 둘(448, 908). 평균 678, 표준편차 230(= (908 − 448) / 2)
    assert steps[6]["blink_rate_60"] == 2 * 60000.0 / 6000.0
    assert (steps[6]["blink_dur_mean_60"], steps[6]["long_closure_count_60"]) == (678.0, 1.0)
    assert steps[6]["blink_dur_std_60"] == pytest.approx(230.0, abs=1e-9)
    # 스텝 8, 9: 구간 C가 0.4에 머무는 중. 진행 중인 길이는 344 ms(7300 − 6956)에서 늘지 않으므로 긴 감김은 B 하나다
    for k in (8, 9):
        assert (steps[k]["long_closure_count_60"], steps[k]["max_closure_ms_60"]) == (1.0, 908.0)
    # 스텝 10: 구간 C가 끝났다(648 ms). 이벤트 셋의 평균 (448 + 908 + 648) / 3 = 668
    assert (steps[10]["blink_dur_mean_60"], steps[10]["long_closure_count_60"]) == (668.0, 2.0)
    assert steps[12]["max_closure_ms_60"] == 908.0 and steps[12]["blink_rate_60"] == 3 * 60000.0 / 12000.0
    # 처음부터 끝까지의 시간(988, 1188, 2788 ms)은 어느 특징에도 나오지 않는다.
    assert all(value not in (988.0, 1188.0, 2788.0) for step in steps.values() for value in step.values())


def test_steps_emitted_during_a_frame_gap_reuse_the_last_ongoing_closure(stored: dict[str, dict[str, Any]]) -> None:
    expected = stored["steps_closure_across_gap"]["expected"]
    names = list(FEATURE_NAMES)
    steps = {step["k"]: step for step in expected["steps"]}
    values = {k: dict(zip(names, step["values"])) for k, step in steps.items()}
    assert sorted(steps) == list(range(1, 33))
    # 12800의 프레임 다음 프레임이 20600에 온다. 그 프레임이 스텝 13~20을 한꺼번에 발행한다.
    assert [k for k, step in steps.items() if step["t_ms"] == 20601] == list(range(13, 21))
    # 감김은 11800 -> 12000의 200 ms 중 111 ms(0.5 / 0.9)에 시작했다: 12800 − 11911 = 889 ms. 500 ms 넘게 끊겨 잘린다.
    (run,) = expected["runs"]
    assert (run["t_ms"], *_run_fields(run)) == (20601, 11912, 12801, 889, True, False)
    for k in range(13, 21):
        assert (values[k]["long_closure_count_60"], values[k]["max_closure_ms_60"]) == (1.0, 889.0)
        assert values[k]["max_closure_ms_10"] == 889.0
    for k in range(21, 31):  # 잘린 구간은 스텝 21의 통에 들어간다. 두 번 세지 않고 10초 구간에 스텝 30까지 남는다
        assert (values[k]["long_closure_count_60"], values[k]["max_closure_ms_10"]) == (1.0, 889.0)
    assert values[31]["max_closure_ms_10"] == 0.0 and values[31]["max_closure_ms_60"] == 889.0
    # perclos_10: 감긴 시간 1000 ms는 그대로이고 유효 시간이 10000에서 3000으로 줄어든다(스텝 13~20).
    assert [values[k]["perclos_10"] for k in range(13, 21)] == [
        1000.0 / valid for valid in (10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000)
    ]
    # 스텝 21, 22: 유효 시간 2000 + 400, 1000 + 1400. 스텝 23에서는 감긴 시간이 10초 구간을 벗어났다
    assert values[21]["perclos_10"] == 1000.0 / 2400.0 and values[22]["perclos_10"] == 1000.0 / 2400.0
    assert values[23]["perclos_10"] == 0.0


def test_zero_division_vector(stored: dict[str, dict[str, Any]]) -> None:
    expected = stored["steps_zero_division"]["expected"]
    names = list(FEATURE_NAMES)
    steps = {step["k"]: dict(zip(names, step["values"])) for step in expected["steps"]}
    assert sorted(steps) == list(range(1, 21))
    ratios = [name for name in names if not name.startswith("face_missing_ratio")]
    for k in range(1, 7):  # 얼굴 없음: 분모가 모두 0이다
        assert all(steps[k][name] == 0.0 for name in ratios), k
        assert steps[k]["face_missing_ratio_60"] == 1.0 and steps[k]["face_missing_ratio_10"] == 1.0
    for k in range(7, 13):  # 얼굴은 있고 closure는 무효(yaw 50도): 유효 시간 0, 얼굴 시간은 스텝마다 1000 ms
        assert all(steps[k][name] == 0.0 for name in ratios), k
        assert steps[k]["face_missing_ratio_60"] == (60000 - 1000 * (k - 6)) / 60000
    # 유효 시간: 스텝 13에 1000, 14에 2000 ... 16에 4000, 17에 4400(16.4초부터 얼굴이 없다), 18에 5200 ms
    assert steps[13]["perclos_10"] == 0.0  # 2000 ms 미만
    assert steps[14]["perclos_10"] == 400.0 / 2000.0  # 13.0~13.4초의 감김
    assert steps[14]["closure_mean_10"] == pytest.approx(0.9 * 400.0 / 2000.0, abs=1e-12)
    for k in range(13, 18):
        assert steps[k]["perclos_60"] == 0.0 and steps[k]["blink_rate_60"] == 0.0, k
    assert steps[18]["perclos_60"] == 800.0 / 5200.0 and steps[18]["blink_rate_60"] == 60000.0 / 5200.0
    # 이벤트가 하나뿐이면 표준편차는 0이다. 잘린 감김은 이벤트 수에 들어가지 않는다.
    assert [run["truncated"] for run in expected["runs"]] == [False, True]
    assert steps[20]["blink_dur_mean_60"] == float(expected["runs"][0]["duration_ms"])
    assert steps[20]["blink_dur_std_60"] == 0.0
    assert all(math.isfinite(value) for step in steps.values() for value in step.values())


def test_closure_length_counts_only_the_time_at_or_above_enter(stored: dict[str, dict[str, Any]], ctx: Context) -> None:
    below = stored["rules_closure_dwell_below_1s"]
    assert below["expected"]["events"] == [] and below["expected"]["alerts"] == []
    # 규칙용 검출기(진입 0.7)로 보면 한 감김이다: 토막 978~1367(389 ms)과 6333~6722(389 ms). 머문 5초는 들어가지 않는다.
    detector = ClosureRunDetector(ctx.rules.closure.detector)
    rule_runs = []
    for row in vectors.compute_expected(below, ctx)["closure"]:
        run = detector.update(row["t_ms"], row["closure"], row["closure_valid"])
        if run is not None:
            rule_runs.append((run.start_ms, run.end_ms, run.duration_ms, run.truncated))
    assert rule_runs == [(979, 6723, 389 + 389, False)]

    reaches = stored["rules_closure_dwell_reaches_1s"]["expected"]
    assert _timed(reaches["events"]) == [("microsleep", "onset", 6944), ("microsleep", "end", 7300)]
    assert _timed(reaches["alerts"]) == [("microsleep", "onset", 6944)]
    onset, end = reaches["events"]
    # 누적 감김 시간은 389 + (6944 − 6333) = 1000 ms. duration_ms는 감김의 시작부터라 머문 5초를 포함한다.
    assert (onset["t_start_ms"], onset["duration_ms"], onset["confidence"]) == (979, 6944 - 978, 1.0)
    assert (end["t_start_ms"], end["duration_ms"]) == (979, 7300 - 978)
    times = [frame["t_ms"] for frame in stored["rules_closure_dwell_reaches_1s"]["frames"]]
    assert 6901 in times and 7001 in times  # 6900에서는 956 ms라 아직이고, '초과'로 구현하면 7000에서 난다

    three = stored["rules_closure_three_spans"]["expected"]
    assert _timed(three["events"]) == [("microsleep", "onset", 2510), ("sleep", "onset", 4510), ("sleep", "end", 4700)]
    assert _timed(three["alerts"]) == [("microsleep", "onset", 2510), ("sleep", "onset", 4510)]
    # 토막 978~1467(489), 1733~2067(334), 2333~: 823 + (2510 − 2333) = 1000, 823 + (4510 − 2333) = 3000
    assert [event["t_start_ms"] for event in three["events"]] == [979, 979, 979]
    assert [event["duration_ms"] for event in three["events"]] == [2510 - 978, 4510 - 978, 4700 - 978]


def test_blocked_alerts_are_issued_once_when_the_block_lifts(stored: dict[str, dict[str, Any]]) -> None:
    cooldown = stored["rules_alert_deferred_cooldown"]["expected"]
    assert _timed(cooldown["events"]) == [
        ("microsleep", "onset", 2000),
        ("microsleep", "end", 2200),
        ("microsleep", "onset", 4000),  # 대기 중. 대기가 풀리기 전(4300)에 끝난다: 경고 없음
        ("microsleep", "end", 4300),
        ("microsleep", "onset", 6000),  # 대기 중. 7000에도 이어진다
        ("microsleep", "end", 7600),
        ("microsleep", "onset", 9000),  # 미뤘다 낸 경고(7000)의 대기 중. 9300에 끝난다: 경고 없음
        ("microsleep", "end", 9300),
    ]
    assert _timed(cooldown["alerts"]) == [("microsleep", "onset", 2000), ("microsleep", "onset", 7000)]
    deferred = cooldown["alerts"][1]
    # 셋째 onset(감김의 시작 4978)의 사본이고 duration_ms만 경고한 시각 기준이다
    assert deferred == {**cooldown["events"][4], "t_ms": 7001, "duration_ms": 7000 - 4978}
    assert cooldown["events"][4]["duration_ms"] == 6000 - 4978

    blocked = stored["rules_alert_blocked_microsleep_sleep"]
    assert _timed(blocked["expected"]["events"]) == [
        ("microsleep", "onset", 2000),
        ("sleep", "onset", 4000),
        ("sleep", "end", 4500),
        ("microsleep", "onset", 8000),
        ("sleep", "onset", 10000),
        ("sleep", "end", 11500),
    ]
    assert blocked["speed"] == [[1, 10.0], [5001, 25.0], [6001, 10.0], [10501, 20.0]]
    # 첫 감김은 막힌 채 끝났으므로 속도가 오른 뒤(5000)에도 경고가 없다. 둘째 감김은 정확히 20.0 km/h가 된
    # 프레임에서 sleep 경고 하나만 낸다(microsleep은 sleep onset에서 지워졌다).
    assert _timed(blocked["expected"]["alerts"]) == [("sleep", "onset", 10500)]
    (alert,) = blocked["expected"]["alerts"]
    assert alert == {**blocked["expected"]["events"][4], "t_ms": 10501, "duration_ms": 10500 - 6978}

    exact = stored["rules_drowsy_perclos_exact"]["expected"]
    names = list(FEATURE_NAMES)
    perclos = {step["k"]: step["values"][names.index("perclos_60")] for step in exact["steps"]}
    # 3 fps: 60초에 프레임 180장 x 200 ms = 36000 ms. 감긴 프레임 27장(5400 ms)이면 0.15, 18장이면 0.10
    assert [perclos[k] for k in range(60, 72)] == [0.15] * 12
    assert [perclos[k] for k in (72, 75, 78, 101, 102)] == [4800 / 36000, 4200 / 36000, 0.1, 0.1, 3000 / 36000]
    assert _timed(exact["events"]) == [("drowsy", "onset", 69000), ("drowsy", "end", 111000)]
    onset, end = exact["events"]
    assert (onset["t_start_ms"], onset["duration_ms"], onset["source"]) == (60001, 9000, "perclos")
    assert (end["t_start_ms"], end["duration_ms"]) == (60001, 111000 - 60000)
    assert stored["rules_drowsy_perclos_exact"]["speed"] == [[1, 45.0], [75001, 50.0]]
    # 45 km/h에서 막힌 onset은 정확히 50.0 km/h가 된 첫 프레임에서 한 번 나간다
    assert exact["alerts"] == [{**onset, "t_ms": 75001, "duration_ms": 15000}]


def test_model_runs_for_every_caught_up_step(stored: dict[str, dict[str, Any]]) -> None:
    vector = stored["rules_drowsy_model_stall"]
    expected = vector["expected"]
    times = [frame["t_ms"] - 1 for frame in vector["frames"]]
    stalls = [(earlier, later) for earlier, later in zip(times, times[1:]) if later - earlier > 1000]
    assert stalls == [(184400, 187000), (197400, 200000)]

    step_probs = expected["step_probs"]
    assert [entry["k"] for entry in step_probs] == list(range(1, 205))
    assert all(entry["probs"] is None for entry in step_probs if entry["k"] < 179)
    # 스텝 179부터 스텝마다 한 번, 호출 순서대로 스텁의 값을 받는다
    scripted = vector["model"]["probs"]
    assert len(scripted) == 26
    assert [entry["probs"] for entry in step_probs if entry["k"] >= 179] == scripted
    assert [p[2] for p in scripted] == [0.5] * 3 + [0.6] * 10 + [0.5] * 3 + [0.3] * 10
    # 끊김 뒤의 프레임은 스텝 셋을 발행하고 모델을 세 번 부른다
    assert [entry["k"] for entry in step_probs if entry["t_ms"] == 187001] == [185, 186, 187]
    assert [entry["k"] for entry in step_probs if entry["t_ms"] == 200001] == [198, 199, 200]
    assert expected["counters"] == {
        "frames_accepted": len(times),
        "dropped_frames": 0,
        "steps_emitted": 204,
        "model_calls": 26,
        "nonfinite_windows": 0,
    }
    # probs는 프레임마다 마지막 스텝 하나다: 끊김 뒤의 프레임 둘에서 둘씩 빠진다
    assert [entry["k"] for entry in expected["probs"]] == [
        k for k in range(179, 205) if k not in (185, 186, 198, 199)
    ]
    # 0.6(시작 임계와 같은 값)이 스텝 182~191의 열 번, 0.3이 195~204의 열 번: 끊김이 연속 수를 끊지 않는다
    assert _timed(expected["events"]) == [("drowsy", "onset", 191000), ("drowsy", "end", 204000)]
    onset, end = expected["events"]
    assert (onset["t_start_ms"], onset["duration_ms"], onset["confidence"], onset["source"]) == (
        1 + 182000,
        9000,
        0.6,
        "model",
    )
    assert (end["t_start_ms"], end["duration_ms"], end["confidence"]) == (1 + 182000, 22000, 0.3)
    assert _timed(expected["alerts"]) == [("drowsy", "onset", 191000)]


def test_gap_over_ten_seconds_restarts_the_rule_engine(stored: dict[str, dict[str, Any]]) -> None:
    over = stored["rules_gap_over_10s"]
    times = [frame["t_ms"] - 1 for frame in over["frames"]]
    assert max(later - earlier for earlier, later in zip(times, times[1:])) == 10001
    events = over["expected"]["events"]
    assert _timed(events) == [
        ("gaze_away_long", "onset", 4000),  # 1000부터 영역 밖
        ("vats", "onset", 11000),  # 누적 정확히 10000 ms
        ("microsleep", "onset", 12400),  # 감김의 시작 11356(200 ms 중 156 ms). 12400 − 11356 = 1044
        ("sleep", "onset", 14400),
        ("gaze_away_long", "end", 24601),  # 초기화가 내는 end: gaze_away_long, 감김, 그리고 비워진 합 때문에 vats
        ("sleep", "end", 24601),
        ("vats", "end", 24601),
        ("microsleep", "onset", 25601),  # 끊김 뒤 첫 프레임부터 다시 센다: 정확히 1000 ms
        ("gaze_away_long", "onset", 27601),  # 정확히 3000 ms
        ("sleep", "onset", 27601),
        ("sleep", "end", 28001),
        ("gaze_away_long", "end", 28401),
    ]
    by_key = {key: event for key, event in zip(_timed(events), events)}
    gaze_end = by_key[("gaze_away_long", "end", 24601)]
    # pose_valid 시간: 1000~14600의 프레임과 마지막 프레임의 200 ms = 13800 ms. 구간은 24601 − 1000
    assert (gaze_end["t_start_ms"], gaze_end["duration_ms"]) == (1001, 23601)
    assert gaze_end["confidence"] == pytest.approx(13800 / 23601, abs=1e-12)
    vats_end = by_key[("vats", "end", 24601)]
    assert (vats_end["t_start_ms"], vats_end["duration_ms"]) == (1001, 23601)
    assert vats_end["confidence"] == pytest.approx(13800 / 23601, abs=1e-12)
    sleep_end = by_key[("sleep", "end", 24601)]
    # 감김을 처음 본 프레임 11400부터: closure_valid 시간 3400 ms, 구간 24601 − 11400
    assert (sleep_end["t_start_ms"], sleep_end["duration_ms"]) == (1 + 11356, 24601 - 11356)
    assert sleep_end["confidence"] == pytest.approx(3400 / 13201, abs=1e-12)
    for key in (("microsleep", "onset", 25601), ("gaze_away_long", "onset", 27601), ("sleep", "onset", 27601)):
        assert by_key[key]["t_start_ms"] == 24602 and by_key[key]["confidence"] == 1.0
    # 속도: 15 km/h로 시작해 4200에 정확히 20.0. 첫 gaze_away_long 경고는 4200에 한 번 나가고(미뤄 둔 사본),
    # 그 경고의 대기(10초)가 풀리는 14200 뒤에도 상태가 이어지지만 다시 나가지 않는다.
    assert over["speed"] == [[1, 15.0], [4201, 20.0]]
    assert over["expected"]["alerts"][0] == {**events[0], "t_ms": 4201, "duration_ms": 3200}
    assert _timed(over["expected"]["alerts"]) == [
        ("gaze_away_long", "onset", 4200),
        ("vats", "onset", 11000),
        ("microsleep", "onset", 12400),
        ("sleep", "onset", 14400),
        ("microsleep", "onset", 25601),
        ("sleep", "onset", 27601),  # 같은 프레임의 경고 둘은 우선순위 순서다(이벤트는 gaze_away_long이 먼저다)
        ("gaze_away_long", "onset", 27601),
    ]

    exact = stored["rules_gap_exactly_10s"]
    times = [frame["t_ms"] - 1 for frame in exact["frames"]]
    assert max(later - earlier for earlier, later in zip(times, times[1:])) == 10000
    assert over["frames"][:74] == exact["frames"][:74]  # 끊기기 전의 프레임은 두 벡터가 같다
    events = exact["expected"]["events"]
    assert _timed(events) == [
        ("gaze_away_long", "onset", 4000),
        ("vats", "onset", 11000),
        ("microsleep", "onset", 12400),
        ("sleep", "onset", 14400),
        ("unable_to_analyze", "onset", 24600),  # 초기화하지 않는다: 최근 10초의 얼굴 시간이 200 ms
        ("sleep", "end", 24600),  # 감김은 500 ms 넘게 끊겨 잘린다
        ("unable_to_analyze", "end", 31600),  # 26600부터의 얼굴 프레임 25장 = 5000 ms
        ("gaze_away_long", "end", 32000),  # 31600에 정면 후보, 400 ms 뒤 확정
        ("vats", "end", 38000),  # 영역 밖 1000~17800(얼굴 소실 뒤 3초까지). 창이 7800 ms 넘게 밀려 합이 10초 아래
    ]
    by_key = {key: event for key, event in zip(_timed(events), events)}
    unable_end = by_key[("unable_to_analyze", "end", 31600)]
    assert (unable_end["duration_ms"], unable_end["confidence"]) == (7000, pytest.approx(5000 / 7000, abs=1e-12))
    gaze_end = by_key[("gaze_away_long", "end", 32000)]
    # pose_valid: 13800 ms(끊김 전) + 26600~32000의 5400 ms
    assert (gaze_end["duration_ms"], gaze_end["confidence"]) == (31000, pytest.approx(19200 / 31000, abs=1e-12))
    vats_end = by_key[("vats", "end", 38000)]
    # 30초보다 긴 vats: 구간 전체의 pose_valid 시간 13800 + 11400 = 25200 ms를 37000 ms로 나눈다(30000 / 37000이 아니다)
    assert (vats_end["t_start_ms"], vats_end["duration_ms"]) == (1001, 37000)
    assert vats_end["confidence"] == pytest.approx(25200 / 37000, abs=1e-12)
    assert _timed(exact["expected"]["alerts"])[-1] == ("unable_to_analyze", "onset", 24600)

    drowsy = stored["rules_gap_over_10s_drowsy"]
    assert _timed(drowsy["expected"]["events"]) == [
        ("drowsy", "onset", 69000),
        ("gaze_away_long", "onset", 69000),
        ("drowsy", "end", 80001),
        ("gaze_away_long", "end", 80001),
    ]
    onset, _, end, _ = drowsy["expected"]["events"]
    # 끊김 초기화의 drowsy end: 발행 시각은 그 프레임의 t_ms, confidence 0.0, onset과 같은 t_start_ms와 source
    assert (end["t_start_ms"], end["duration_ms"], end["confidence"], end["source"]) == (60001, 20001, 0.0, "perclos")
    assert (onset["t_start_ms"], onset["source"]) == (60001, "perclos")
    # 45 km/h에서 막힌 drowsy 경고는 끊김 뒤 80 km/h가 되어도 나가지 않는다. 끊김 뒤의 스텝에서 다시 시작하지도 않는다.
    assert drowsy["speed"] == [[1, 45.0], [80002, 80.0]]
    assert _timed(drowsy["expected"]["alerts"]) == [("gaze_away_long", "onset", 69000)]
    assert [step["k"] for step in drowsy["expected"]["steps"]][-3:] == [70, 81, 82]


def test_gaps_up_to_ten_seconds_count_as_time_without_a_face(stored: dict[str, dict[str, Any]]) -> None:
    boundary = stored["rules_gap_unable_boundary"]
    times = [frame["t_ms"] - 1 for frame in boundary["frames"]]
    gaps = [(earlier, later - earlier) for earlier, later in zip(times, times[1:]) if later - earlier > 200]
    assert gaps == [(10000, 5200), (25400, 5201)]
    assert all(frame["face"] for frame in boundary["frames"])
    # 둘째 끊김 직전의 네 프레임(24800~25400)은 고개를 돌리고 있다(변환 행렬의 yaw 성분이 0이 아니다).
    turned = [frame["t_ms"] - 1 for frame in boundary["frames"] if abs(frame["m"][8]) > 0.1]
    assert turned == [24800, 25000, 25200, 25400]
    # 5200 ms: 얼굴 시간 4800 + 200 = 5000 ms(0.5, 시작하지 않음). 5201 ms: 4999 ms(시작). 35601에서 다시 5000 ms(종료)
    assert _timed(boundary["expected"]["events"]) == [
        ("unable_to_analyze", "onset", 30601),
        ("unable_to_analyze", "end", 35601),
    ]
    # 끊긴 동안 영역 밖으로 친 시간(25600까지 본 것 + 3000 ms)이 3초를 넘어도, 분석 불가 중인 프레임에서는
    # gaze_away_long이 나지 않는다.
    assert _timed(boundary["expected"]["alerts"]) == [("unable_to_analyze", "onset", 30601)]

    sparse = stored["rules_unable_sparse_frames"]
    times = [frame["t_ms"] - 1 for frame in sparse["frames"]]
    assert sorted({later - earlier for earlier, later in zip(times, times[1:])}) == [500, 10500]
    assert all(frame["face"] for frame in sparse["frames"])
    # 500 ms 간격: 프레임마다 200 ms만 인정되어 얼굴 시간 비율이 0.4다. 처음 10초는 판정하지 않는다.
    assert _timed(sparse["expected"]["events"]) == [
        ("unable_to_analyze", "onset", 10000),
        ("unable_to_analyze", "end", 22500),  # 10초 넘는 끊김 뒤의 첫 프레임
        ("unable_to_analyze", "onset", 32500),  # 그 프레임부터 다시 10초
    ]
    end = sparse["expected"]["events"][1]
    # 10000~22500의 얼굴 시간: 10000, 10500, 11000, 11500, 12000의 프레임이 200 ms씩
    assert (end["duration_ms"], end["confidence"]) == (12500, pytest.approx(1000 / 12500, abs=1e-12))
    # 둘째 onset은 재경고 대기(30초)에 걸리고, 첫 경고에서 정확히 30000 ms 뒤의 프레임에 나간다
    assert _timed(sparse["expected"]["alerts"]) == [
        ("unable_to_analyze", "onset", 10000),
        ("unable_to_analyze", "onset", 40000),
    ]
    assert sparse["expected"]["alerts"][1] == {**sparse["expected"]["events"][2], "t_ms": 40001, "duration_ms": 7500}


def test_forward_zone_boundary_vectors(stored: dict[str, dict[str, Any]]) -> None:
    missed = stored["rules_gaze_credit_window_missed"]
    times = [frame["t_ms"] - 1 for frame in missed["frames"]]
    assert 3900 in times and 4300 in times and not any(3900 < t < 4300 for t in times)
    with_face = [frame["t_ms"] - 1 for frame in missed["frames"] if frame["face"]]
    assert with_face[:13] == list(range(0, 1300, 100)) and with_face[13] == 5100  # 1300~5000은 얼굴이 없다
    assert _timed(missed["expected"]["events"]) == [("gaze_away_long", "onset", 4300), ("gaze_away_long", "end", 5400)]
    onset = missed["expected"]["events"][0]
    # 영역 밖을 처음 본 프레임 1000부터. pose_valid는 1000, 1100, 1200의 프레임 셋(300 ms)뿐이다
    assert (onset["t_start_ms"], onset["duration_ms"]) == (1001, 3300)
    assert onset["confidence"] == pytest.approx(300 / 3300, abs=1e-12)

    credit = stored["rules_gaze_credit_boundary"]["expected"]
    assert _timed(credit["events"]) == [
        ("gaze_away_long", "onset", 11000),  # 둘째 장면: 8000부터 영역 밖
        ("gaze_away_long", "end", 15200),  # 14700의 프레임은 이어받지 않는다. 14900에 정면 후보, 15200에 확정
        ("gaze_away_long", "onset", 21000),
        ("gaze_away_long", "end", 24650),  # 인정 구간이 끝나기 50 ms 전의 프레임: 이어받은 표지로 확정
    ]
    # 첫 장면(1000~1300의 이탈, 1400의 정면 프레임, 4600의 얼굴 없는 프레임)에서는 이벤트가 없다.
    assert _timed(credit["alerts"]) == [("gaze_away_long", "onset", 11000), ("gaze_away_long", "onset", 21000)]

    debounce = stored["rules_gaze_debounce_250ms"]
    times = [frame["t_ms"] - 1 for frame in debounce["frames"]]
    assert {4500, 4550, 4600, 4650, 4700, 4750, 4800} <= set(times)
    assert _timed(debounce["expected"]["events"]) == [
        ("gaze_away_long", "onset", 3000),
        ("gaze_away_long", "end", 4750),  # 3500~3650의 정면(200 ms)은 확정되지 않았고, 4500의 후보가 정확히 250 ms 뒤에 확정됐다
    ]

    edges = stored["rules_forward_zone_edges"]["expected"]
    gaze = [event for event in edges["events"] if event["type"] == EVENT_GAZE_AWAY_LONG]
    starts = [1000 + 3800 * index for index in range(6)]
    assert _timed(gaze[::2]) == [("gaze_away_long", "onset", start + 3000) for start in starts]
    assert _timed(gaze[1::2]) == [("gaze_away_long", "end", start + 3600) for start in starts]
    assert [event["source"] for event in gaze[::2]] == ["head_pose"] * 3 + ["eye", "eye", "head_pose"]
    assert all(event["duration_ms"] == 3000 for event in gaze[::2])
    # 이탈마다 3200 ms(바깥 값의 프레임 16장)가 영역 밖이다. 세 번에 9600 ms, 네 번째 이탈(눈동자)의 400 ms에서 10초
    (vats,) = [event for event in edges["events"] if event["type"] == EVENT_VATS]
    assert (vats["t_ms"] - 1, vats["phase"]) == (starts[3] + 400, "onset")
    assert (vats["t_start_ms"], vats["source"]) == (1001, "eye")
    # 경고: 첫 onset, vats, 그리고 첫 경고에서 10초 넘게 지난 넷째 onset. 나머지는 대기에 걸린 채 상태가 끝난다
    assert _timed(edges["alerts"]) == [
        ("gaze_away_long", "onset", starts[0] + 3000),
        ("vats", "onset", starts[3] + 400),
        ("gaze_away_long", "onset", starts[3] + 3000),
    ]

    reset = stored["rules_vats_reset_enabled"]["expected"]
    # 이탈 1000~3800, 6000~8800, 10600~13400, 15200~18000, 19800~. 5800에서 합을 지웠으므로 둘째 이탈부터 센다:
    # 2800 x 3 = 8400, 다섯 번째 이탈의 1600 ms에서 10000 ms
    assert _timed(reset["events"]) == [("vats", "onset", 21400)]
    assert (reset["events"][0]["t_start_ms"], reset["events"][0]["duration_ms"]) == (6001, 15400)

    priority = stored["rules_alert_priority"]["expected"]
    assert _timed(priority["events"])[:2] == [("gaze_away_long", "onset", 4000), ("microsleep", "onset", 4000)]
    assert _timed(priority["alerts"]) == [("microsleep", "onset", 4000), ("gaze_away_long", "onset", 4000)]


def test_calibrating_end_confidence_is_the_face_time_ratio(stored: dict[str, dict[str, Any]]) -> None:
    vector = stored["rules_calibrating_provisional"]
    missing = [frame["t_ms"] - 1 for frame in vector["frames"] if not frame["face"]]
    assert len(missing) == 12 and 40000 <= min(missing) and max(missing) < 42000
    end = vector["expected"]["events"][3]
    # 60초 중 2초 동안 얼굴이 없다(6 fps의 프레임 간격은 200 ms 이하라 빠지는 시간이 없다)
    assert (end["type"], end["phase"], end["duration_ms"]) == ("calibrating", "end", 60000)
    assert end["confidence"] == pytest.approx(58000 / 60000, abs=1e-12)


# ---------------------------------------------------------------------------
# 2차 검증에서 더한 벡터 (손으로 계산한 기대값)
# ---------------------------------------------------------------------------
#
# 단위 테스트만 잡던 동작(정점이 옮겨지는 구간의 속도, 감긴 표본 하나로 잘린 구간, 분석 불가 동안 멈추는 시선
# 타이머, 모델 판정기의 초기화, 다른 종류의 end와 미뤄 둔 경고)과 어느 쪽도 잡지 못하던 동작(특징 쪽의 10초 경계,
# 끊김에서 버리는 후보, 버려진 후보 구간, 아래를 보는 눈동자, either의 source, 빈 영역 밖 구간)을 벡터로 고정한다.


def _turned(vector: dict[str, Any]) -> list[int]:
    """고개를 돌린 프레임의 세션 시간(변환 행렬의 yaw 성분이 0이 아니다)."""
    return [frame["t_ms"] - 1 for frame in vector["frames"] if frame["face"] and abs(frame["m"][8]) > 0.1]


def test_velocities_of_a_run_whose_peak_moves_to_a_later_sample(stored: dict[str, dict[str, Any]]) -> None:
    expected = stored["runs_two_humps"]["expected"]
    rows = {row["t_ms"] - 1: row["closure"] for row in expected["closure"]}
    shape = [rows[t] for t in (200, 400, 433, 466, 566, 666, 766)]
    assert shape == pytest.approx([0.05, 0.55, 0.42, 0.54, 0.82, 0.45, 0.1], abs=1e-9)
    moved, tied, before, after = expected["runs"]

    # (1) 진입: 0.05 -> 0.55(200 ms)의 180 ms에 0.5를 지난다(380). 0.55 -> 0.42(33 ms)의 13 ms에 내려가고(413),
    # 0.42 -> 0.54(33 ms)의 22 ms에 다시 올라온다(455). 0.82 -> 0.45(100 ms)의 86 ms에 내려간다(652).
    assert _run_fields(moved) == (381, 653, (413 - 380) + (652 - 455), False, True)
    assert moved["amplitude"] == pytest.approx(0.82, abs=1e-9)
    # 감는 속도: 정점(0.82의 표본)까지의 쌍은 2.5(진입), 3.64(0.42 -> 0.54), 2.8(0.54 -> 0.82)다. 가장 큰 3.64.
    assert moved["peak_closing_velocity"] == pytest.approx(0.12 / 0.033, rel=1e-9)
    # 뜨는 속도: 정점 뒤의 쌍은 3.7(0.82 -> 0.45), 3.5(0.45 -> 0.1)다. 앞 봉우리에서 내려간 3.94는 들어가지 않는다.
    assert moved["peak_opening_velocity"] == pytest.approx(0.37 / 0.1, rel=1e-9)
    assert moved["peak_opening_velocity"] < 0.13 / 0.033

    # (2) 0.1 -> 0.9(100 ms)의 50 ms에 올라오고(2150), 0.9 -> 0.1(200 ms)의 100 ms에 내려간다(2433).
    assert [rows[t] for t in (2200, 2233, 2333, 2533)] == pytest.approx([0.9, 0.6, 0.9, 0.1], abs=1e-9)
    assert rows[2200] == rows[2333]  # 같은 최댓값이 두 번: 정점은 처음 닿은 표본(2200)이다
    assert _run_fields(tied) == (2151, 2434, 283, False, True)
    assert tied["peak_closing_velocity"] == pytest.approx(0.8 / 0.1, rel=1e-9)
    # 정점이 2333으로 옮겨졌다면 그 뒤의 쌍은 0.9 -> 0.1(200 ms)의 4.0/s뿐이다.
    assert tied["peak_opening_velocity"] == pytest.approx(0.3 / 0.033, rel=1e-9)

    # (3) 0.9 -> 0.55 -> 0.86 -> 0.1. 0.55 -> 0.86(33 ms)의 9.39/s는 정점을 넘지 못한 증가라 감는 속도가 아니다.
    assert _run_fields(before) == (4151, 4381, 230, False, True)
    assert before["peak_closing_velocity"] == pytest.approx(0.8 / 0.1, rel=1e-9)
    assert before["peak_opening_velocity"] == pytest.approx(0.76 / 0.1, rel=1e-9)
    # (4) 0.1 -> 0.55(200 ms, 2.25/s) -> 0.82(100 ms, 2.7/s) -> 0.1. 앞 구간에 남은 9.39/s가 넘어오지 않는다.
    assert _run_fields(after) == (5379, 5545, 166, False, True)
    assert after["peak_closing_velocity"] == pytest.approx(0.27 / 0.1, rel=1e-9)
    assert after["peak_opening_velocity"] == pytest.approx(0.72 / 0.1, rel=1e-9)


def test_truncated_runs_with_a_single_closed_sample(stored: dict[str, dict[str, Any]]) -> None:
    vector = stored["runs_one_sample_truncated"]
    rows = vector["expected"]["closure"]
    closed = [row["t_ms"] - 1 for row in rows if row["closure"] >= 0.5]
    assert closed == [1100, 3200, 5200]
    times = [frame["t_ms"] - 1 for frame in vector["frames"]]
    assert [(a, b - a) for a, b in zip(times, times[1:]) if b - a > 100] == [(1100, 600), (5200, 600)]
    assert [frame["t_ms"] - 1 for frame in vector["frames"] if not frame["face"]] == list(range(3300, 3900, 100))
    runs = vector["expected"]["runs"]
    # 0.42 -> 0.9(100 ms)의 17 ms에 0.5를 지난다. 끝은 감긴 표본의 시각이고 길이는 83 ms다.
    assert [(run["t_ms"], *_run_fields(run)) for run in runs] == [
        (1701, 1018, 1101, 83, True, False),  # 프레임이 600 ms 끊긴 뒤의 프레임에서
        (3801, 3118, 3201, 83, True, False),  # 얼굴이 사라지고 500 ms를 넘긴 첫 무효 프레임에서
    ]
    for run in runs:
        assert (run["peak_closing_velocity"], run["peak_opening_velocity"]) == (0.0, 0.0)
        assert run["amplitude"] == pytest.approx(0.9, abs=1e-9)
    # 5200의 표본은 0.05 -> 0.9의 53 ms에 0.5를 지나 길이가 47 ms다. 잘린 구간도 50 ms 미만이면 버린다.
    by_time = {row["t_ms"] - 1: row["closure"] for row in rows}
    assert (by_time[5100], by_time[5200]) == pytest.approx((0.05, 0.9), abs=1e-9)
    assert all(run["t_ms"] < 5000 for run in runs)


def test_feature_side_reset_boundary_is_strictly_above_ten_seconds(stored: dict[str, dict[str, Any]]) -> None:
    vector = stored["steps_gap_exactly_10s"]
    expected = vector["expected"]
    times = [frame["t_ms"] - 1 for frame in vector["frames"]]
    assert [(a, b - a) for a, b in zip(times, times[1:]) if b - a > 200] == [(5400, 10000), (20400, 10001)]
    names = list(FEATURE_NAMES)
    steps = {step["k"]: step for step in expected["steps"]}
    values = {k: dict(zip(names, step["values"])) for k, step in steps.items()}
    assert sorted(steps) == [*range(1, 21), 31, 32, 33]
    assert not any(step["full_lookback"] for step in steps.values())

    # 10000 ms: 초기화하지 않는다. 감김은 4600(0.0) -> 4800(0.9)의 111 ms에 시작했고 5400까지 봤다(689 ms).
    (run,) = expected["runs"]
    assert (run["t_ms"], *_run_fields(run)) == (15401, 4712, 5401, 689, True, False)
    assert [k for k, step in steps.items() if step["t_ms"] == 15401] == list(range(6, 16))
    for k in range(6, 16):
        assert (values[k]["long_closure_count_60"], values[k]["max_closure_ms_60"]) == (1.0, 689.0), k
        assert values[k]["max_closure_ms_10"] == 689.0, k
    # 끊긴 동안 10초 구간의 유효 시간이 줄어든다: 감긴 시간 800 ms(프레임 넷)는 그대로다.
    assert values[6]["perclos_10"] == 800.0 / 5600.0  # 유효 시간은 0~5600
    assert values[11]["perclos_10"] == 800.0 / 4600.0  # 창이 1000에서 시작한다
    assert values[14]["perclos_10"] == 0.0  # 유효 시간 1600 ms: 하한(2000 ms) 미만
    for k in range(16, 21):  # 잘린 구간은 그 프레임이 속한 스텝 16의 통에 들어간다
        assert (values[k]["long_closure_count_60"], values[k]["max_closure_ms_10"]) == (1.0, 689.0), k
    assert values[20]["face_missing_ratio_60"] == pytest.approx((60000 - 10200) / 60000, abs=1e-12)

    # 10001 ms: 초기화한다. 둘째 감김(19800~20400)은 구간이 되지 못하고, 스텝 21~30은 없다.
    assert steps[31]["t_ms"] == 31002 and steps[33]["t_ms"] == 33002
    for k in (31, 32, 33):
        assert (values[k]["long_closure_count_60"], values[k]["max_closure_ms_60"]) == (0.0, 0.0), k
        assert (values[k]["perclos_60"], values[k]["blink_rate_60"]) == (0.0, 0.0), k
    # 새 누적은 스텝 31의 시작(30000)부터다. 첫 프레임 30401부터 31000까지의 599 ms만 얼굴 시간이다.
    assert values[31]["face_missing_ratio_60"] == pytest.approx((60000 - 599) / 60000, abs=1e-12)
    assert values[33]["face_missing_ratio_60"] == pytest.approx((60000 - 2599) / 60000, abs=1e-12)
    assert expected["counters"]["steps_emitted"] == 23


def test_forward_zone_candidates_at_gaps(stored: dict[str, dict[str, Any]]) -> None:
    candidate = stored["rules_gap_over_10s_candidate"]
    times = [frame["t_ms"] - 1 for frame in candidate["frames"]]
    assert [(a, b - a) for a, b in zip(times, times[1:]) if b - a > 200] == [(2000, 10001)]
    # 끊기기 전에 고개를 돌린 프레임은 2000 하나뿐이다. 끊김 뒤에는 12001~15201.
    assert _turned(candidate) == [2000, *range(12001, 15401, 200)]
    events = candidate["expected"]["events"]
    assert _timed(events) == [("gaze_away_long", "onset", 15001), ("gaze_away_long", "end", 15801)]
    onset, end = events
    # 영역 밖은 끊김 뒤의 첫 프레임부터다(2000으로 소급하지 않는다). 정확히 3000 ms 뒤에 onset.
    assert (onset["t_start_ms"], onset["duration_ms"], onset["confidence"]) == (12002, 3000, 1.0)
    assert (end["t_start_ms"], end["duration_ms"], end["confidence"]) == (12002, 3800, 1.0)
    assert candidate["expected"]["alerts"] == [onset]

    dropped = stored["rules_gaze_candidate_dropped"]
    times = [frame["t_ms"] - 1 for frame in dropped["frames"]]
    assert times == [*range(0, 3100, 100), *range(6300, 8400, 100)]
    assert _turned(dropped) == [*range(1000, 3000, 100), *range(6300, 8000, 100)]
    events = dropped["expected"]["events"]
    assert _timed(events) == [("gaze_away_long", "onset", 7300), ("gaze_away_long", "end", 8300)]
    onset, end = events
    # 연속 영역 밖 시간: 1000~3000의 2000 ms + 6300~7300의 1000 ms. 버려진 후보 구간(3000~6300)은 세지 않는다.
    assert (onset["t_start_ms"], onset["duration_ms"], onset["source"]) == (1001, 6300, "head_pose")
    # pose_valid 시간: 1000~2900의 2000 ms, 3000의 프레임이 유지된 200 ms, 6300~7200의 1000 ms
    assert onset["confidence"] == pytest.approx(3200 / 6300, abs=1e-12)
    assert (end["t_start_ms"], end["duration_ms"]) == (1001, 7300)
    assert end["confidence"] == pytest.approx(4200 / 7300, abs=1e-12)
    assert dropped["expected"]["alerts"] == [onset]


def test_eye_gaze_edges_in_the_other_two_directions(stored: dict[str, dict[str, Any]], ctx: Context) -> None:
    vector = stored["rules_forward_zone_eye_edges"]
    rows = {row["t_ms"] - 1: row for row in vectors.compute_expected(vector, ctx)["closure"]}
    assert [rows[t]["gaze_h_rel"] for t in (1000, 4000, 4200, 4600)] == pytest.approx([0.251, 0.251, 0.249, 0.249])
    assert [rows[t]["gaze_v_rel"] for t in (4800, 7800, 8000, 8400)] == pytest.approx([-0.351, -0.351, -0.349, -0.349])
    assert all(row["yaw_rel"] == pytest.approx(0.0, abs=1e-9) for row in rows.values())  # 머리는 정면이다
    events = vector["expected"]["events"]
    assert _timed(events) == [
        ("gaze_away_long", "onset", 4000),
        ("gaze_away_long", "end", 4600),
        ("gaze_away_long", "onset", 7800),
        ("gaze_away_long", "end", 8400),
    ]
    assert [event["source"] for event in events] == ["eye"] * 4
    assert [event["t_start_ms"] for event in events] == [1001, 1001, 4801, 4801]
    assert [event["duration_ms"] for event in events] == [3000, 3600, 3000, 3600]
    # 둘째 onset은 첫 경고(4000)의 재경고 대기 10초 안이고, 대기가 풀리기 전에 상태가 끝난다.
    assert vector["expected"]["alerts"] == [events[0]]
    # rules_forward_zone_edges가 지나는 방향은 반대쪽이다(gaze_h 음수, gaze_v 양수).
    other = vectors.compute_expected(stored["rules_forward_zone_edges"], ctx)["closure"]
    assert min(row["gaze_h_rel"] for row in other) == pytest.approx(-0.251)
    assert max(row["gaze_v_rel"] for row in other) == pytest.approx(0.351)
    assert max(row["gaze_h_rel"] for row in other) < 0.01 and min(row["gaze_v_rel"] for row in other) > -0.01


def test_unable_to_analyze_pauses_the_away_timer(stored: dict[str, dict[str, Any]]) -> None:
    vector = stored["rules_gaze_unable_pauses_timer"]
    times = [frame["t_ms"] - 1 for frame in vector["frames"]]
    assert times == list(range(0, 57000, 200))
    missing = [frame["t_ms"] - 1 for frame in vector["frames"] if not frame["face"]]
    assert missing == [*range(12000, 16000, 200), *range(16600, 40000, 200)]
    assert _turned(vector) == [16000, 16200, 16400, *range(40000, 56000, 200)]
    events = vector["expected"]["events"]
    assert _timed(events) == [
        # 최근 10초의 얼굴 시간: 17600에서 4400 + 600 = 5000(아직), 17800에서 4200 + 600 = 4800
        ("unable_to_analyze", "onset", 17800),
        # 40000부터의 얼굴 프레임 25장 = 5000 ms
        ("unable_to_analyze", "end", 45000),
        # 분석 불가 전 16000~17800의 1800 ms + 뒤 45000~46200의 1200 ms = 3000 ms
        ("gaze_away_long", "onset", 46200),
        # 영역 밖 구간 [45000, 55000) = 10000 ms. [16000, 17800)은 47800에 30초 창을 벗어났다
        ("vats", "onset", 55000),
        # 56000의 정면 후보가 400 ms 뒤에 확정된다
        ("gaze_away_long", "end", 56400),
    ]
    unable_on, unable_end, gaze, vats, gaze_end = events
    assert (unable_end["t_start_ms"], unable_end["duration_ms"]) == (17801, 27200)
    assert unable_end["confidence"] == pytest.approx(5000 / 27200, abs=1e-12)
    assert (gaze["t_start_ms"], gaze["duration_ms"], gaze["source"]) == (16001, 30200, "head_pose")
    # pose_valid 시간: 16000~16400의 프레임 셋(600 ms) + 40000~46000의 프레임 서른하나(6200 ms)
    assert gaze["confidence"] == pytest.approx(6800 / 30200, abs=1e-12)
    # vats의 시작은 그때 30초 창 안의 첫 영역 밖 구간의 시작이다. 분석 불가 동안(17800~45000)의 프레임 시각이 아니다.
    assert (vats["t_start_ms"], vats["duration_ms"], vats["confidence"]) == (45001, 10000, 1.0)
    assert (gaze_end["t_start_ms"], gaze_end["duration_ms"]) == (16001, 40400)
    assert gaze_end["confidence"] == pytest.approx(17000 / 40400, abs=1e-12)
    assert vector["expected"]["alerts"] == [unable_on, gaze, vats]


def test_either_source_takes_the_source_of_the_judge_that_opened_the_state(stored: dict[str, dict[str, Any]]) -> None:
    vector = stored["rules_drowsy_either"]
    assert vector["config_overrides"] == {"drowsiness.source": "either"}
    expected = vector["expected"]
    names = list(FEATURE_NAMES)
    steps = {step["k"]: dict(zip(names, step["values"])) for step in expected["steps"]}
    assert sorted(steps) == list(range(1, 201))
    # 3 fps: 60초의 유효 시간 36000 ms. 감긴 프레임 세 장(600 ms)이 든 초가 165, 168, ..., 189의 아홉.
    assert [steps[k]["perclos_60"] for k in (164, 166, 189)] == [0.0, 600 / 36000, 4800 / 36000]
    assert [steps[k]["perclos_60"] for k in range(190, 201)] == [0.15] * 11
    assert steps[199]["face_missing_ratio_60"] == pytest.approx(0.4, abs=1e-12)
    scripted = [entry["probs"] for entry in expected["step_probs"] if entry["probs"] is not None]
    assert [p[2] for p in scripted] == [0.7] * 10 + [0.3] * 10 + [0.5] * 2  # 스텝 179~200
    assert [entry["k"] for entry in expected["step_probs"] if entry["probs"] is not None] == list(range(179, 201))

    events = expected["events"]
    assert _timed(events) == [("drowsy", "onset", 188000), ("drowsy", "end", 198000), ("drowsy", "onset", 199000)]
    by_model, model_end, by_perclos = events
    assert (by_model["source"], by_model["t_start_ms"], by_model["duration_ms"]) == ("model", 179001, 9000)
    assert by_model["confidence"] == 0.7
    assert (model_end["source"], model_end["t_start_ms"], model_end["duration_ms"]) == ("model", 179001, 19000)
    assert model_end["confidence"] == 0.3
    # 둘째 상태는 PERCLOS가 열었다: perclos_60이 0.15인 스텝 190~199의 열 번째. source가 model로 남으면 안 된다.
    assert (by_perclos["source"], by_perclos["t_start_ms"], by_perclos["duration_ms"]) == ("perclos", 190001, 9000)
    assert by_perclos["confidence"] == pytest.approx(0.6, abs=1e-12)  # 1 − face_missing_ratio_60
    # 60 km/h. 둘째 onset은 첫 경고의 재경고 대기(300초) 안이다.
    assert expected["alerts"] == [by_model]
    assert expected["counters"]["model_calls"] == 22


def test_gap_over_ten_seconds_resets_the_model_judge(stored: dict[str, dict[str, Any]], ctx: Context) -> None:
    vector = stored["rules_gap_over_10s_drowsy_model"]
    assert vector["config_overrides"] == {"drowsiness.source": "model", "drowsiness.model.refractory_ms": 600000}
    assert ctx.rules.drowsiness.model.refractory_ms == 60000  # 기본값으로는 끊김 뒤 179초를 기다리는 동안 이미 지난다
    times = [frame["t_ms"] - 1 for frame in vector["frames"]]
    assert [(a, b - a) for a, b in zip(times, times[1:]) if b - a > 334] == [(199000, 10001)]
    expected = vector["expected"]
    step_probs = expected["step_probs"]
    # 스텝은 199까지, 그리고 끊김 뒤의 프레임(209001)이 속한 스텝 210부터. 끊긴 동안의 스텝은 없다.
    assert [entry["k"] for entry in step_probs] == [*range(1, 200), *range(210, 399)]
    # 모델 호출: 179~199(21번), 끊김 뒤에는 다 찬 스텝(269~)이 120개 쌓인 388부터(11번).
    called = [entry["k"] for entry in step_probs if entry["probs"] is not None]
    assert called == [*range(179, 200), *range(388, 399)]
    assert [entry["probs"][2] for entry in step_probs if entry["probs"] is not None] == (
        [0.7] * 10 + [0.3] * 10 + [0.5] + [0.7] * 11
    )
    assert expected["counters"] == {
        "frames_accepted": 598 + 570,
        "dropped_frames": 0,
        "steps_emitted": 199 + 189,
        "model_calls": 32,
        "nonfinite_windows": 0,
    }
    events = expected["events"]
    assert _timed(events) == [("drowsy", "onset", 188000), ("drowsy", "end", 198000), ("drowsy", "onset", 397001)]
    first_on, end, second_on = events
    assert (first_on["t_start_ms"], first_on["duration_ms"], first_on["confidence"]) == (179001, 9000, 0.7)
    assert (end["t_start_ms"], end["duration_ms"], end["confidence"]) == (179001, 19000, 0.3)
    # 새 onset: 0.7이 스텝 388~397의 열 번. 앞 상태가 끝난 지 199초로 금지 시간(600초) 안이지만 판정기가 처음
    # 상태로 돌아갔으므로 난다. 스텝의 끝 시각은 세션 시간 그대로다(388000, 397000).
    assert (second_on["t_start_ms"], second_on["duration_ms"], second_on["confidence"]) == (388001, 9000, 0.7)
    assert all(event["source"] == "model" for event in events)
    assert (second_on["t_ms"] - 1) - 198000 < 600000 and second_on["t_ms"] < 600000
    # 끊김의 프레임에는 이벤트가 없다(진행 중인 상태가 없었다). 경고는 첫 onset 하나(둘째는 재경고 대기 300초 안).
    assert expected["alerts"] == [first_on]


def test_pending_microsleep_alert_survives_the_end_of_another_state(stored: dict[str, dict[str, Any]]) -> None:
    vector = stored["rules_alert_pending_survives_end"]
    assert vector["speed"] == [[1, 10.0], [6801, 25.0]]
    events = vector["expected"]["events"]
    assert _timed(events) == [
        ("gaze_away_long", "onset", 4000),  # 1000부터 고개를 돌렸다
        ("microsleep", "onset", 6000),  # 감김의 시작 4956(4800 -> 5000의 200 ms 중 156 ms). 6000 − 4956 = 1044
        ("gaze_away_long", "end", 6600),  # 6200의 정면 후보가 400 ms 뒤에 확정된다(눈은 감은 채다)
        ("microsleep", "end", 7200),
    ]
    gaze, microsleep, gaze_end, microsleep_end = events
    assert (microsleep["t_start_ms"], microsleep["duration_ms"]) == (4957, 1044)
    assert (gaze_end["t_start_ms"], gaze_end["duration_ms"]) == (1001, 5600)
    assert (microsleep_end["t_start_ms"], microsleep_end["duration_ms"]) == (4957, 2244)
    # 10 km/h에서 두 onset 모두 막힌다. gaze_away_long은 자기 end(6600)에서 지워지고, microsleep은 남아 있다가
    # 25 km/h가 된 프레임에서 한 번 나간다(onset의 사본, duration_ms만 그 시각 기준).
    assert vector["expected"]["alerts"] == [{**microsleep, "t_ms": 6801, "duration_ms": 6800 - 4956}]


def test_new_closure_right_after_a_dropped_short_run(stored: dict[str, dict[str, Any]], ctx: Context) -> None:
    vector = stored["rules_closure_after_dropped_run"]
    times = [frame["t_ms"] - 1 for frame in vector["frames"]]
    assert times == [*range(0, 1100, 100), *range(1700, 3300, 100)]
    rows = vectors.compute_expected(vector, ctx)["closure"]
    assert [row["t_ms"] - 1 for row in rows if row["closure"] >= 0.7] == [1000, *range(1700, 3000, 100)]
    # 규칙용 검출기(진입 0.7)만 따로 돌리면 구간은 하나다. 1000의 감김(978~1000, 22 ms)은 1700에서 잘리고 50 ms
    # 미만이라 버려진다. 새 감김은 1700에서 시작한다(700 ms 떨어진 쌍은 보간하지 않는다). 끝은 2900 + 22.
    detector = ClosureRunDetector(ctx.rules.closure.detector)
    runs = [detector.update(row["t_ms"], row["closure"], row["closure_valid"]) for row in rows]
    assert [(r.start_ms, r.end_ms, r.duration_ms, r.truncated) for r in runs if r is not None] == [
        (1701, 2923, 1222, False)
    ]
    events = vector["expected"]["events"]
    assert _timed(events) == [("microsleep", "onset", 2700), ("microsleep", "end", 3000)]
    onset, end = events
    # 새 감김의 것이다: 시작 1700, 처음 본 프레임도 1700(그 뒤의 프레임은 모두 유효하다).
    assert (onset["t_start_ms"], onset["duration_ms"], onset["confidence"]) == (1701, 1000, 1.0)
    assert (end["t_start_ms"], end["duration_ms"], end["confidence"]) == (1701, 1300, 1.0)
    assert vector["expected"]["alerts"] == [onset]


def test_short_forward_glance_and_the_two_ends_of_the_face_loss_credit(stored: dict[str, dict[str, Any]]) -> None:
    vector = stored["rules_gaze_glance_and_credit_times"]
    times = [frame["t_ms"] - 1 for frame in vector["frames"]]
    assert times == [
        *range(0, 12200, 200),
        *(12100, 15150, 15250, 15350, 15450, 15550, 15650),
        *range(15800, 21200, 200),
        24200,
    ]
    assert [frame["t_ms"] - 1 for frame in vector["frames"] if not frame["face"]] == [12100, 15150]
    # 고개를 돌린 프레임: 1000~11800에서 2600만 빼고, 그리고 21000과 24200.
    assert _turned(vector) == [*(t for t in range(1000, 12000, 200) if t != 2600), 21000, 24200]
    events = vector["expected"]["events"]
    assert _timed(events) == [
        ("gaze_away_long", "onset", 4000),  # 2600의 정면 프레임 하나는 연속을 끊지 않는다
        ("vats", "onset", 11000),  # [1000, 11000) 전체가 영역 밖 시간이다(2600~2800의 200 ms 포함)
        # 12000의 관측은 12100까지 유지됐다(다음 프레임이 100 ms 뒤에 왔다). 이어받는 구간은 15100에 끝나고 15150은
        # 그 뒤다. 15250의 정면 후보가 300 ms 뒤에 확정된다.
        ("gaze_away_long", "end", 15550),
        # 21000의 영역 밖 후보는 관측이 유지된 끝(21200) + 3000 = 24200까지 살아 있다. 정확히 그 시각의 프레임에서
        # 21000으로 소급해 확정되고, 연속 영역 밖 시간이 3200 ms라 바로 onset이 난다.
        ("gaze_away_long", "onset", 24200),
    ]
    gaze, vats, gaze_end, credited = events
    assert (credited["t_start_ms"], credited["duration_ms"]) == (21001, 3200)
    assert credited["confidence"] == pytest.approx(200 / 3200, abs=1e-12)  # 21000의 프레임이 유지된 200 ms
    assert (gaze["t_start_ms"], gaze["duration_ms"], gaze["confidence"]) == (1001, 3000, 1.0)
    assert (vats["t_start_ms"], vats["duration_ms"], vats["confidence"]) == (1001, 10000, 1.0)
    # pose_valid 시간: 1000~12000의 프레임이 12100까지 11100 ms, 15250~15450의 프레임 셋이 300 ms
    assert (gaze_end["t_start_ms"], gaze_end["duration_ms"]) == (1001, 14550)
    assert gaze_end["confidence"] == pytest.approx(11400 / 14550, abs=1e-12)
    # 셋째 경고는 첫 gaze_away_long 경고(4000)의 재경고 대기 10초가 지난 뒤다.
    assert vector["expected"]["alerts"] == [gaze, vats, credited]


def test_rules_and_alerts_with_a_large_first_timestamp(stored: dict[str, dict[str, Any]]) -> None:
    vector = stored["rules_calibrating_large_t0"]
    origin = LARGE_FIRST_T_MS
    assert vector["baseline"] is None and vector["frames"][0]["t_ms"] == origin
    times = [frame["t_ms"] - origin for frame in vector["frames"]]
    # 6 fps(프레임 k의 시각은 floor(k x 1000 / 6 + 0.5)). 59700~61200의 프레임 아홉이 없다.
    assert len(times) == 399 and all(frame["face"] for frame in vector["frames"])
    assert [(a, b - a) for a, b in zip(times, times[1:]) if b - a > 167] == [(59667, 1666)]
    expected = vector["expected"]
    baseline = expected["baseline"]
    assert (baseline["t_ms"] - origin, baseline["t_complete_ms"], baseline["status"]) == (61333, 60000, "ok")

    def timed(items: list[dict[str, Any]]) -> list[tuple[str, str, int]]:
        return [(item["type"], item["phase"], item["t_ms"] - origin) for item in items]

    events = expected["events"]
    assert timed(events) == [
        ("calibrating", "onset", 0),
        # 감김은 10000(0.0) -> 10167(0.97)의 167 ms 중 121 ms(0.7 / 0.97)에 시작했다: 11167 − 10121 = 1046
        ("microsleep", "onset", 11167),
        ("microsleep", "end", 11833),
        ("calibrating", "end", 61333),  # 기준선이 완성된 프레임(T_b 뒤의 첫 프레임)
        ("gaze_away_long", "onset", 66333),  # 영역 밖을 처음 본 프레임 63333에서 정확히 3000 ms 뒤
        ("gaze_away_long", "end", 67167),  # 66833의 정면 후보가 334 ms 뒤에 확정된다
    ]
    calib_on, micro_on, micro_end, calib_end, gaze, gaze_end = events
    assert (calib_on["t_start_ms"], calib_on["duration_ms"], calib_on["provisional"]) == (origin, 0, True)
    assert (micro_on["t_start_ms"] - origin, micro_on["duration_ms"], micro_on["provisional"]) == (10121, 1046, True)
    assert (micro_end["t_start_ms"] - origin, micro_end["duration_ms"], micro_end["provisional"]) == (10121, 1712, True)
    # calibrating end: 구간은 첫 프레임부터 61333 ms. 얼굴 시간은 59667까지 이어진 뒤 마지막 프레임이 200 ms 유지된다.
    assert (calib_end["t_start_ms"], calib_end["duration_ms"], calib_end["provisional"]) == (origin, 61333, False)
    assert calib_end["confidence"] == pytest.approx(59867 / 61333, abs=1e-12)
    assert (gaze["t_start_ms"] - origin, gaze["duration_ms"], gaze["confidence"]) == (63333, 3000, 1.0)
    assert (gaze_end["t_start_ms"] - origin, gaze_end["duration_ms"]) == (63333, 3834)
    assert timed(expected["alerts"]) == [("microsleep", "onset", 11167), ("gaze_away_long", "onset", 66333)]
    assert min(event["t_start_ms"] for event in events) > 2**31

    # 누적은 T_b = 60000부터다. 61333의 프레임이 빈 스텝 61을 발행한다(스텝의 끝 시각은 세션 시간 61000).
    names = list(FEATURE_NAMES)
    steps = expected["steps"]
    assert [(step["t_ms"] - origin, step["k"], step["t_end_ms"] - origin) for step in steps] == [
        (61333, 61, 61000),
        *[(1000 * k, k, 1000 * k) for k in range(62, 68)],
    ]
    missing = [dict(zip(names, step["values"]))["face_missing_ratio_60"] for step in steps]
    assert missing[0] == 1.0
    # 스텝 62: 61333~62000의 667 ms. 그 뒤로는 스텝마다 1000 ms씩 는다.
    assert missing[1:] == pytest.approx([(60000 - 667 - 1000 * i) / 60000 for i in range(6)], abs=1e-12)
    assert expected["counters"]["steps_emitted"] == 7


# ---------------------------------------------------------------------------
# 3차 검증 뒤에 더한 벡터: 손으로 계산한 값
# ---------------------------------------------------------------------------
#
# 참조 구현의 한 줄을 일부러 틀리게 바꿔도 벡터 81개를 모두 통과하던 자리 여덟을 고정한다. 여섯은 단위 테스트만 잡던
# 것(30초 창의 가장자리, 정점이 옮겨지기 전의 두 증가, pose_valid가 끊긴 vats의 confidence, 한 프레임짜리 곁눈질,
# 얼굴이 사라진 감김의 confidence, 다른 종류의 onset과 미뤄 둔 경고)이고, 하나는 단위 테스트도 잡지 못하던 것(첫 t_ms가
# 작은 세션의 "처음 10초")이며, 하나는 그 정비에서 찾은 것(스텝 경계에서 벗어난 프레임이 내는 drowsy의 발행 시각과 바로
# 낸 경고)이다. 기대값은 규칙을 따라 손으로 계산했다. 벡터를 틀린 코드로 다시 만들면 여기서 드러난다.


def test_closing_velocity_keeps_the_largest_rise_until_the_peak_moves(stored: dict[str, dict[str, Any]]) -> None:
    vector = stored["runs_two_rises_before_peak"]
    expected = vector["expected"]
    times = [frame["t_ms"] - 1 for frame in vector["frames"]]
    assert times == [0, 200, 400, 433, 466, 499, 532, 632, 732, 832, 1000]
    rows = {row["t_ms"] - 1: row["closure"] for row in expected["closure"]}
    shape = [rows[t] for t in times]
    assert shape == pytest.approx([0.05, 0.05, 0.55, 0.42, 0.54, 0.45, 0.48, 0.82, 0.45, 0.1, 0.1], abs=1e-9)
    (run,) = expected["runs"]
    # 토막 셋. 0.05 -> 0.55(200 ms)의 180 ms에 0.5를 위로 지나고(380), 0.55 -> 0.42(33 ms)의 12.7 ms에 내려간다(413).
    # 0.42 -> 0.54(33 ms)의 22 ms에 올라오고(455), 0.54 -> 0.45(33 ms)의 14.7 ms에 내려간다(481).
    # 0.48 -> 0.82(100 ms)의 5.9 ms에 올라오고(538), 0.82 -> 0.45(100 ms)의 86.5 ms에 내려간다(718).
    assert (run["t_ms"], *_run_fields(run)) == (833, 381, 719, (413 - 380) + (481 - 455) + (718 - 538), False, True)
    assert run["duration_ms"] == 239 and run["end_ms"] - run["start_ms"] == 338
    assert run["amplitude"] == pytest.approx(0.82, abs=1e-9)
    # 감는 속도: 정점(632의 표본)까지의 쌍은 2.5(진입), 3.64(0.42 -> 0.54), 0.91(0.45 -> 0.48), 3.4(0.48 -> 0.82)다.
    # 정점을 넘지 못한 두 증가 가운데 큰 것(3.64)이 정점이 옮겨질 때까지 남아 있어야 한다.
    assert run["peak_closing_velocity"] == pytest.approx(0.12 / 0.033, rel=1e-9)
    # 마지막 증가율(0.91)만 들고 있는 구현은 3.4를 낸다. 벡터의 허용 오차(상대 1e-4)로 갈린다.
    assert 0.12 / 0.033 - 0.34 / 0.1 > 0.2
    # 뜨는 속도: 새 정점 뒤의 쌍은 3.7(0.82 -> 0.45), 3.5(0.45 -> 0.1)다. 앞에서 내려간 3.94와 2.73은 버려진다.
    assert run["peak_opening_velocity"] == pytest.approx(0.37 / 0.1, rel=1e-9)
    # runs_two_humps의 첫 구간에는 정점이 옮겨지기 전의 증가가 한 번뿐이다(0.42 -> 0.54 뒤 바로 0.82).
    humps = {row["t_ms"] - 1: row["closure"] for row in stored["runs_two_humps"]["expected"]["closure"]}
    assert [humps[t] for t in (400, 433, 466, 566)] == pytest.approx([0.55, 0.42, 0.54, 0.82], abs=1e-9)


def test_vats_drops_an_interval_that_ends_exactly_at_the_start_of_the_window(stored: dict[str, dict[str, Any]]) -> None:
    vector = stored["rules_vats_window_edge"]
    times = [frame["t_ms"] - 1 for frame in vector["frames"]]
    assert times == [*range(0, 29200, 200), *range(32000, 33200, 200)]
    assert all(frame["face"] for frame in vector["frames"])
    assert _turned(vector) == [*range(1000, 2000, 200), *range(21000, 29200, 200), *range(32000, 33200, 200)]
    events = vector["expected"]["events"]
    assert _timed(events) == [("gaze_away_long", "onset", 24000), ("vats", "onset", 32000)]
    gaze, vats = events
    assert (gaze["t_start_ms"], gaze["duration_ms"], gaze["confidence"]) == (21001, 3000, 1.0)
    # 영역 밖 구간은 [1000, 2000)과 [21000, ...)이다. 29000의 프레임에서 합은 1000 + 8000 = 9000 ms.
    # 다음 프레임(32000)까지 끊긴 3000 ms는 29000 + 200 + 3000 = 32200 전이라 모두 영역 밖으로 친다: 둘째 구간이
    # [21000, 32000) = 11000 ms. 30초 창의 시작은 32000 - 30000 = 2000이고 첫 구간은 정확히 거기서 끝나므로 버린다.
    # 남은 첫 구간의 시작 21000이 t_start_ms다. 버리지 않으면 길이 0인 구간의 시작 2000이 되고 duration_ms가 30000이다.
    assert (vats["t_start_ms"], vats["duration_ms"], vats["source"]) == (21001, 11000, "head_pose")
    # pose_valid 시간: 21000~28800의 프레임 마흔이 200 ms씩, 29000의 프레임이 유지된 200 ms(그 뒤 2800 ms는 데이터 없음)
    assert vats["confidence"] == pytest.approx(8200 / 11000, abs=1e-12)
    assert vector["expected"]["alerts"] == [gaze, vats]


def test_vats_onset_confidence_with_pose_validity_interrupted(stored: dict[str, dict[str, Any]]) -> None:
    vector = stored["rules_vats_confidence_face_lost"]
    times = [frame["t_ms"] - 1 for frame in vector["frames"]]
    assert times == list(range(0, 12800, 200))
    missing = [frame["t_ms"] - 1 for frame in vector["frames"] if not frame["face"]]
    assert missing == [400, 600, *range(5000, 7000, 200)]
    assert _turned(vector) == [*range(1000, 5000, 200), *range(7000, 12000, 200)]
    events = vector["expected"]["events"]
    assert _timed(events) == [
        ("gaze_away_long", "onset", 4000),
        # 영역 밖 [1000, 11000): 얼굴이 없던 5000~7000도 마지막 관측 + 200 + 3000 = 8000 전이라 영역 밖으로 친다
        ("vats", "onset", 11000),
        ("gaze_away_long", "end", 12400),  # 12000의 정면 후보가 400 ms 뒤에 확정된다
    ]
    gaze, vats, gaze_end = events
    assert (gaze["t_start_ms"], gaze["duration_ms"], gaze["confidence"]) == (1001, 3000, 1.0)
    assert (vats["t_start_ms"], vats["duration_ms"]) == (1001, 10000)
    # pose_valid 구간은 [0, 400), [800, 5000), [7000, 11000)이다. t_start_ms(1000) 뒤의 부분만 구간마다 더한다:
    # 첫 구간은 0, 둘째는 4000, 셋째는 4000. 셋째 구간을 1000부터 세면 1.4, 첫 구간을 400 - 1000으로 더하면 0.74다.
    assert vats["confidence"] == pytest.approx(8000 / 10000, abs=1e-12)
    assert (gaze_end["t_start_ms"], gaze_end["duration_ms"]) == (1001, 11400)
    assert gaze_end["confidence"] == pytest.approx(9400 / 11400, abs=1e-12)  # 4000 + (12400 - 7000)
    assert vector["expected"]["alerts"] == [gaze, vats]


def test_one_frame_glances_never_add_up_to_away_time(stored: dict[str, dict[str, Any]]) -> None:
    vector = stored["rules_gaze_one_frame_glances"]
    times = [frame["t_ms"] - 1 for frame in vector["frames"]]
    assert times == list(range(0, 33200, 200))
    assert all(frame["face"] for frame in vector["frames"])
    turned = _turned(vector)
    # 400 ms마다 프레임 하나: 다음 프레임(200 ms 뒤)은 정면이라 영역 밖 후보가 250 ms를 채우지 못한다.
    assert turned == list(range(1000, 32400, 400)) and len(turned) == 79
    # 고개를 돌린 프레임의 시간은 79 x 200 = 15800 ms(10초 초과), 곁눈질은 31.4초 동안(30초 창보다 길게) 이어진다.
    assert 200 * len(turned) == 15800 and turned[-1] + 200 - turned[0] == 31400
    # 영역 밖이 한 번도 확정되지 않으므로 영역 밖 시간이 없다. 버려진 후보의 구간을 더하면 20800에 vats가 난다.
    assert vector["expected"]["events"] == [] and vector["expected"]["alerts"] == []
    assert vector["expected"]["counters"]["frames_accepted"] == 166


def test_microsleep_confidence_with_the_face_lost_inside_the_closure(
    stored: dict[str, dict[str, Any]], ctx: Context
) -> None:
    vector = stored["rules_microsleep_confidence_face_lost"]
    times = [frame["t_ms"] - 1 for frame in vector["frames"]]
    assert times == list(range(0, 3100, 100))
    assert [frame["t_ms"] - 1 for frame in vector["frames"] if not frame["face"]] == [1300, 1400, 1500]
    rows = vectors.compute_expected(vector, ctx)["closure"]
    closed = [row["t_ms"] - 1 for row in rows if row["closure_valid"] and row["closure"] >= 0.7]
    assert closed == [1000, 1100, 1200, *range(1600, 2600, 100)]
    events = vector["expected"]["events"]
    assert _timed(events) == [("microsleep", "onset", 2000), ("microsleep", "end", 2600)]
    onset, end = events
    # 감김의 시작 978(900 -> 1000의 78 ms). 얼굴이 없던 400 ms(1200 -> 1600)는 hold_ms 500 이하이고 앞뒤 표본이 모두
    # 0.7 이상이라 길이에 들어간다: 2000의 프레임에서 2000 - 978 = 1022 ms(1900에서는 922).
    assert (onset["t_start_ms"], onset["duration_ms"]) == (979, 1022)
    # closure_valid 시간은 감김을 처음 본 프레임(1000)부터 센다: 1000~1200의 프레임 셋 300 ms + 1600~1900의 넷 400 ms.
    # 얼굴 없는 프레임 셋(1300~1500)이 유지된 300 ms는 유효 시간이 아니다.
    assert onset["confidence"] == pytest.approx(700 / 1000, abs=1e-12)
    assert (end["t_start_ms"], end["duration_ms"]) == (979, 1622)
    assert end["confidence"] == pytest.approx(1300 / 1600, abs=1e-12)  # 300 + 1000 ms of [1000, 2600]
    assert vector["expected"]["alerts"] == [onset]


def test_pending_microsleep_alert_survives_the_onset_of_another_state(stored: dict[str, dict[str, Any]]) -> None:
    vector = stored["rules_alert_pending_survives_onset"]
    assert vector["speed"] == [[1, 10.0], [4401, 25.0]]
    times = [frame["t_ms"] - 1 for frame in vector["frames"]]
    assert times == list(range(0, 6200, 200))
    assert _turned(vector) == list(range(1000, 5000, 200))
    events = vector["expected"]["events"]
    assert _timed(events) == [
        ("microsleep", "onset", 3800),  # 감김의 시작 2756(2600 -> 2800의 200 ms 중 156 ms). 3800 - 2756 = 1044
        ("gaze_away_long", "onset", 4000),  # 1000부터 고개를 돌렸다
        ("microsleep", "end", 5000),
        ("gaze_away_long", "end", 5400),  # 5000의 정면 후보가 400 ms 뒤에 확정된다
    ]
    microsleep, gaze, microsleep_end, gaze_end = events
    assert (microsleep["t_start_ms"], microsleep["duration_ms"]) == (2757, 1044)
    assert (gaze["t_start_ms"], gaze["duration_ms"]) == (1001, 3000)
    assert (microsleep_end["duration_ms"], gaze_end["duration_ms"]) == (2244, 4400)
    # 10 km/h에서 두 onset 모두 막혀 미뤄진다. gaze_away_long의 onset(4000)은 미뤄 둔 microsleep 경고를 지우지 않는다.
    # 25 km/h가 된 프레임(4400)에서 둘 다 나간다: 우선순위 순서(microsleep 먼저), onset의 사본, duration_ms만 그 시각 기준.
    assert vector["expected"]["alerts"] == [
        {**microsleep, "t_ms": 4401, "duration_ms": 4400 - 2756},
        {**gaze, "t_ms": 4401, "duration_ms": 4400 - 1000},
    ]


def test_first_ten_seconds_are_measured_from_a_small_first_timestamp(stored: dict[str, dict[str, Any]]) -> None:
    vector = stored["rules_unable_small_t0"]
    origin = SMALL_FIRST_T_MS
    assert origin == 5000 and vector["frames"][0]["t_ms"] == 5000
    assert vector["baseline"] is not None  # 기준선을 넣어 줬다: calibrating 이벤트가 없다
    times = [frame["t_ms"] - origin for frame in vector["frames"]]
    assert times == list(range(0, 17400, 200))
    assert [t for t, frame in zip(times, vector["frames"]) if not frame["face"]] == list(range(0, 12000, 200))
    events = vector["expected"]["events"]
    onset, end = events
    assert [(event["type"], event["phase"]) for event in events] == [
        ("unable_to_analyze", "onset"),
        ("unable_to_analyze", "end"),
    ]
    # "처음 10초"는 첫 프레임(t_ms 5000)부터다: 정확히 10000 ms 뒤의 프레임 t_ms 15000에서 처음 판정하고, 얼굴 시간이
    # 0이라 바로 시작한다. t_ms 0부터 재면 t_ms 10000의 프레임에서 시작한다.
    assert (onset["t_ms"], onset["t_start_ms"], onset["duration_ms"], onset["confidence"]) == (15000, 15000, 0, 0.0)
    # 세션 12000부터의 얼굴 프레임 스물다섯(12000~16800)이 5000 ms: 세션 17000(t_ms 22000)의 프레임에서 비율 0.5로 끝난다.
    assert (end["t_ms"], end["t_start_ms"], end["duration_ms"]) == (22000, 15000, 7000)
    assert end["confidence"] == pytest.approx(5000 / 7000, abs=1e-12)
    assert vector["expected"]["alerts"] == [onset]
    assert vector["speed"] == []  # 분석 불가 경고는 속도와 무관하다


def test_drowsy_emit_time_is_the_step_end_when_frames_are_off_the_step_boundary(
    stored: dict[str, dict[str, Any]],
) -> None:
    vector = stored["rules_drowsy_off_step_boundary"]
    times = [frame["t_ms"] - 1 for frame in vector["frames"]]
    # 3 fps, 72초. 첫 프레임을 뺀 모든 프레임이 50 ms 늦다: 초마다 50, 383, 717.
    assert times == [0, 383, 717, *(1000 * second + offset for second in range(1, 72) for offset in (50, 383, 717))]
    assert all(frame["face"] for frame in vector["frames"])
    expected = vector["expected"]
    names = list(FEATURE_NAMES)
    steps = {step["k"]: step for step in expected["steps"]}
    assert sorted(steps) == list(range(1, 72))
    # 스텝 k는 세션 1000k + 50의 프레임이 발행하고, 스텝의 끝 시각은 1000k다.
    assert all((steps[k]["t_ms"], steps[k]["t_end_ms"]) == (1000 * k + 51, 1000 * k + 1) for k in steps)
    values = {k: dict(zip(names, step["values"])) for k, step in steps.items()}
    # 감긴 프레임 세 장(600 ms)이 든 초가 아홉(11, 14, 17, 41, 44, ..., 56): 60초의 유효 시간 36000 ms 중 5400 ms.
    assert [values[k]["perclos_60"] for k in range(60, 72)] == [0.15] * 12
    assert not steps[59]["full_lookback"] and steps[60]["full_lookback"]
    (onset,) = expected["events"]
    assert (onset["type"], onset["phase"], onset["source"]) == ("drowsy", "onset", "perclos")
    assert onset["provisional"] is False
    # 스텝 60~69의 열 번째에 onset. 발행한 프레임은 세션 69050이지만 이벤트의 발행 시각은 스텝 69의 끝(69000)이다:
    # t_start_ms는 스텝 60의 끝, duration_ms는 69000 - 60000. 프레임 시각에서 재면 9050이다.
    assert (onset["t_ms"], onset["t_start_ms"], onset["duration_ms"]) == (69051, 60001, 9000)
    assert onset["confidence"] == pytest.approx(0.6, abs=1e-12)  # 1 - face_missing_ratio_60: 얼굴 시간 36000 / 60000
    # 60 km/h(기준 50 이상)라 경고가 그 프레임에서 바로 나간다. 바로 낸 경고는 onset 이벤트 그대로다. duration_ms를
    # 프레임 시각 기준으로 고치는 것은 미뤘다 낸 경고와 되풀이 경고뿐이다.
    assert vector["speed"] == [[1, 60.0]]
    assert expected["alerts"] == [onset]

    # 다른 벡터에서는 drowsy 이벤트를 낸 프레임의 시각과 이벤트의 발행 시각(t_start_ms + duration_ms)이 같거나 1 ms
    # 차이다(허용 오차 안). 발행 시각을 프레임에서 재는 구현은 이 벡터에서만 드러난다.
    def frame_minus_emit(name: str) -> list[int]:
        events = stored[name]["expected"].get("events", []) + stored[name]["expected"].get("alerts", [])
        return [e["t_ms"] - e["t_start_ms"] - e["duration_ms"] for e in events if e["type"] == "drowsy"]

    others = [offset for name in stored if name != vector["name"] for offset in frame_minus_emit(name)]
    assert len(others) >= 20 and set(others) <= {0, 1}
    assert frame_minus_emit(vector["name"]) == [50, 50]


# ---------------------------------------------------------------------------
# 4차 검증 뒤에 더한 벡터: 손으로 계산한 값
# ---------------------------------------------------------------------------
#
# 신호, 기준선, 스텝 특징 모듈의 한 줄을 일부러 틀리게 바꿔도 벡터 89개를 모두 통과하던 자리를 고정하는 벡터 열셋이다.
# 기대값은 규칙을 따라 손으로 계산해 여기에 수로 적었다(참조 구현에서 읽어 온 값이 아니다). 벡터를 틀린 코드로 다시
# 만들면 여기서 드러난다. 스텝 특징 벡터는 `_personal_baseline`을 넣었다: E = 0.27, B = 0.22, jaw0 = 0.08, R0는 단위
# 행렬이라 pitch_rel은 머리의 pitch, jaw_rel은 jawOpen − 0.08, closure는 (eyeBlink − 0.22) / 0.38이다.


def _pitch(frame: dict[str, Any]) -> float:
    """프레임의 pitch(도). yaw와 roll이 0인 행렬에서 M[1][2] = m[9] = −sin(pitch)다."""
    return round(math.degrees(math.asin(-frame["m"][9])), 6) + 0.0


def _jaw(frame: dict[str, Any]) -> float:
    return round(frame["bs"]["jawOpen"] - 0.08, 6)


def _lids(frame: dict[str, Any]) -> float:
    return round((frame["bs"]["eyeBlinkLeft"] - 0.22) / 0.38, 6)


def _held(frames: list[dict[str, Any]], value: Any, of: Any) -> list[int]:
    """`of(frame)`이 value인 얼굴 프레임의 세션 시간."""
    return [frame["t_ms"] - 1 for frame in frames if frame["face"] and of(frame) == value]


def _feature(vector: dict[str, Any], name: str) -> list[float]:
    index = list(FEATURE_NAMES).index(name)
    return [step["values"][index] for step in vector["expected"]["steps"]]


def _no_face(vector: dict[str, Any]) -> list[int]:
    return [frame["t_ms"] - 1 for frame in vector["frames"] if not frame["face"]]


def test_iris_ratio_does_not_depend_on_the_tilt_of_the_eye_axis(stored: dict[str, dict[str, Any]]) -> None:
    vector = stored["signals_iris_tilted_eyes"]
    assert [(frame["w"], frame["h"]) for frame in vector["frames"]] == [(640, 480), (1280, 720)]
    # 두 눈 꼬리(Left 263과 362, Right 33과 133)의 픽셀 좌표 차. 눈 폭 52 px과 48 px을 3:4:5 삼각형의 각도로 돌렸다:
    # 첫 프레임 Left는 (41.6, 31.2), Right는 (28.8, 38.4), 둘째 프레임 Left는 (31.2, 41.6), Right는 (38.4, 28.8).
    spans = []
    for frame in vector["frames"]:
        for outer, inner in (("263", "362"), ("33", "133")):
            dx = (frame["lm"][inner][0] - frame["lm"][outer][0]) * frame["w"]
            dy = (frame["lm"][inner][1] - frame["lm"][outer][1]) * frame["h"]
            spans.append((abs(dx), abs(dy)))
    assert spans == [pytest.approx(pair, abs=1e-9) for pair in ((41.6, 31.2), (28.8, 38.4), (31.2, 41.6), (38.4, 28.8))]
    for row in vector["expected"]["signals"]:
        assert (row["eye_w_l"], row["eye_w_r"]) == pytest.approx((52.0, 48.0), abs=1e-9)
        assert (row["ear_l"], row["ear_r"]) == pytest.approx((0.31, 0.27), abs=1e-9)
        # 홍채 중심은 바깥 꼬리에서 눈 축을 따라 h·|a|, 축에 수직으로 v·|a|만큼 떨어져 있다. 축을 돌려도 (h, v)는 같다.
        assert (row["iris_h_l"], row["iris_v_l"]) == pytest.approx((0.46, 0.02), abs=1e-9)
        assert (row["iris_h_r"], row["iris_v_r"]) == pytest.approx((0.55, -0.03), abs=1e-9)
    # 식에서 a의 세로 성분이 든 항의 부호를 틀리면: 첫 프레임 Left 눈은 a = (−41.6, −31.2)·(±1)이고 |a|² = 2704다.
    # |a|²를 ax² − ay²(= 757.12)로 구하면 h가 0.46·2704/757.12 = 1.643이 된다. 벡터의 허용 오차(1e-6)로 갈린다.
    assert 0.46 * 2704 / (41.6**2 - 31.2**2) == pytest.approx(1.6429, abs=1e-4)
    # 다른 신호 벡터에서는 두 눈 꼬리의 높이가 같다(그래서 그 항이 0이다).
    for name in ("signals_ear_square", "signals_ear_non_square", "signals_one_eye_closed"):
        for frame in stored[name]["frames"]:
            assert frame["lm"]["263"][1] == frame["lm"]["362"][1] and frame["lm"]["33"][1] == frame["lm"]["133"][1]


def test_baseline_with_an_odd_number_of_samples_takes_the_middle_sample(stored: dict[str, dict[str, Any]]) -> None:
    vector = stored["baseline_odd_sample_count"]
    frames = vector["frames"]
    assert vector["baseline"] is None and all(frame["face"] for frame in frames)
    times = [frame["t_ms"] - 1 for frame in frames]
    collected = [*range(0, 60000, 200), 65000, 72000, 79000, *range(90000, 118000, 1000)]
    assert times == [*collected, 120000, 120200, 120400] and len(collected) == 331
    # 수집한 331개: 앞의 165개는 pitch 0·턱 0.05, 뒤의 166개는 pitch 4·턱 0.09
    assert [_pitch(frame) for frame in frames[:331]] == [0.0] * 165 + [4.0] * 166
    assert [round(frame["bs"]["jawOpen"], 6) for frame in frames[:331]] == [0.05] * 165 + [0.09] * 166
    signals_of = [signals.compute(vectors.frame_from_json(frame), vectors.load_context().spec) for frame in frames[:331]]
    # Right 눈: 처음 300개는 폭 3 px(쓸 수 없음), 다음 셋은 EAR 0.50, 나머지 28개는 0.28. Left 눈은 늘 0.30.
    assert [round(s.eye_w_r, 6) for s in signals_of] == [3.0] * 300 + [44.0] * 31
    assert [round(s.ear_r, 6) for s in signals_of[300:]] == [0.5] * 3 + [0.28] * 28
    assert {round(s.ear_l, 6) for s in signals_of} == {0.3}
    assert {(round(s.blink_l, 6), round(s.blink_r, 6)) for s in signals_of} == {(0.2, 0.24)}

    baseline = vector["expected"]["baseline"]
    # 60초 마감: 프레임 300개(≥ 300), 얼굴 시간 300 x 200 = 60000 ms(비율 1.0). Right 눈의 표본이 없어 연장.
    # 90초 마감: Right 눈 표본 셋이 모두 0.50 > e_max 0.45라 연장.
    # 120초 마감: Right 눈 표본 31개의 90 분위수는 ceil(0.9 x 31) = 28번째 = 0.28. 프레임 331개, 얼굴 시간
    # 60000 + 3 x 200 + 28 x 200 = 66200 ms(비율 0.5517). 완성.
    assert (baseline["t_complete_ms"], baseline["status"], baseline["t_ms"]) == (120000, BASELINE_OK, 120001)
    assert (baseline["e_l"], baseline["e_r"]) == pytest.approx((0.30, 0.28), abs=1e-9)
    assert (baseline["b_l"], baseline["b_r"]) == pytest.approx((0.20, 0.24), abs=1e-9)
    # 중앙값: 331개의 가운데는 166번째 표본이다. 작은 값이 165개뿐이므로 큰 값이 중앙값이다(pitch 4, 턱 0.09,
    # 눈동자 합성값 0.1과 0.06). 가운데 두 표본의 평균을 내는 구현은 pitch 2, 턱 0.07, 0.05와 0.03을 낸다.
    assert baseline["jaw0"] == pytest.approx(0.09, abs=1e-9)
    assert (baseline["gaze_h0"], baseline["gaze_v0"]) == pytest.approx((0.1, 0.06), abs=1e-9)
    cos4, sin4 = math.cos(math.radians(4.0)), math.sin(math.radians(4.0))
    assert baseline["r0"] == pytest.approx([1.0, 0.0, 0.0, 0.0, cos4, -sin4, 0.0, sin4, cos4], abs=1e-9)  # Rx(4도)

    rows = vector["expected"]["closure"]
    assert [row["provisional"] for row in rows] == [True] * 331 + [False] * 3
    same, low, shut = rows[-3:]
    # 완성 뒤: 기준선과 같은 얼굴은 모두 0. 앞쪽 자세의 얼굴은 pitch 0 − 4, 턱 0.05 − 0.09, 눈동자 0 − 0.1과 0 − 0.06.
    assert (same["pitch_rel"], same["jaw_rel"], same["closure"]) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    assert (low["pitch_rel"], low["jaw_rel"]) == pytest.approx((-4.0, -0.04), abs=1e-9)
    assert (low["gaze_h_rel"], low["gaze_v_rel"]) == pytest.approx((-0.1, -0.06), abs=1e-9)
    # 셋째 프레임: Left EAR 0.30·(1 − 0.65 x 0.6) = 0.183, eyeBlink 0.20 + 0.40 x 0.6 = 0.44. Right EAR 0.28·0.61 = 0.1708,
    # eyeBlink 0.24 + 0.36 x 0.6 = 0.456. 두 눈 모두 c = 0.6이므로 closure 0.6. E_r이 0.50이었다면 a_r = 1이 된다.
    last = frames[-1]["bs"]
    assert (last["eyeBlinkLeft"], last["eyeBlinkRight"]) == pytest.approx((0.44, 0.456), abs=1e-9)
    assert shut["closure"] == pytest.approx(0.6, abs=1e-9) and shut["closure_valid"]


def test_face_time_ratio_of_the_baseline_when_frames_arrive_in_bursts(stored: dict[str, dict[str, Any]]) -> None:
    vector = stored["baseline_burst_frames"]
    frames = vector["frames"]
    assert vector["baseline"] is None
    face = [frame["t_ms"] - 1 for frame in frames if frame["face"]]
    bursts = [1000 * second + offset for second in range(59) for offset in (0, 75, 150, 225, 300)]
    assert face == [
        *bursts,
        *(59000, 59025, 59050, 59075, 59100, 59900),
        *range(60100, 75100, 200),
        *(90000, 90200, 90400),
    ]
    assert _no_face(vector) == [500, 1500, 2500]
    every = [frame["t_ms"] - 1 for frame in frames]

    def face_time(deadline: int, cap: int | None = 200, clip: bool = True, count_missing: bool = False) -> int:
        total = 0
        for index, frame in enumerate(frames):
            t = every[index]
            if t >= deadline or not (frame["face"] or count_missing):
                continue
            end = every[index + 1]
            if cap is not None:
                end = min(end, t + cap)
            if clip:
                end = min(end, deadline)
            total += end - t
        return total

    # 60초 마감. 초마다의 다섯 장: 75 x 4 + 200 = 500 ms, 59번. 마지막 1초: 25 x 4 + 200 = 300 ms, 59900의 프레임은
    # 마감까지 100 ms. 합 59 x 500 + 300 + 100 = 29900 ms, 프레임 59 x 5 + 6 = 301개: 수는 충분하고 비율이 0.5에 못 미친다.
    assert sum(1 for t in face if t < 60000) == 301
    assert face_time(60000) == 59 * 500 + 300 + 100 == 29900 and 29900 / 60000 < 0.5
    # 틀린 구현 셋은 60초에 완성한다: 상한 없이 간격을 다 더하면 58500 ms(얼굴 없는 프레임 셋 뒤의 500 ms씩만 빠진다),
    # 59900의 프레임이 대표한 200 ms를 마감에서 자르지 않으면 30000 ms, 얼굴 없는 프레임 셋의 200 ms를 더하면 30500 ms.
    assert face_time(60000, cap=None) == 60000 - 3 * 500 == 58500
    assert face_time(60000, clip=False) == 30000
    assert face_time(60000, count_missing=True) == 30500
    # 90초 마감. 59900의 프레임의 나머지 100 ms와 60100~74900의 75장 x 200 ms를 더해 정확히 45000 ms: 비율이 정확히 0.5다.
    assert face_time(90000) == 29900 + 100 + 75 * 200 == 45000 and sum(1 for t in face if t < 90000) == 376
    baseline = vector["expected"]["baseline"]
    assert (baseline["t_complete_ms"], baseline["status"], baseline["t_ms"]) == (90000, BASELINE_OK, 90001)
    assert (baseline["e_l"], baseline["e_r"], baseline["b_l"], baseline["b_r"], baseline["jaw0"]) == pytest.approx(
        (0.30, 0.30, 0.20, 0.20, 0.10), abs=1e-9
    )
    provisional = [row["provisional"] for row in vector["expected"]["closure"]]
    assert provisional == [True] * (len(frames) - 3) + [False] * 3


def test_baseline_completes_with_exactly_300_face_frames(stored: dict[str, dict[str, Any]], ctx: Context) -> None:
    vector = stored["baseline_exactly_300_frames"]
    frames = vector["frames"]
    assert vector["baseline"] is None and all(frame["face"] for frame in frames)
    times = [frame["t_ms"] - 1 for frame in frames]
    assert times == [*range(0, 59800, 200), 60000, 90000, 90200, 90400] and len(times) == 303
    assert (ctx.spec.baseline.min_face_frames, ctx.spec.baseline.min_face_ratio) == (300, 0.5)
    collected = [signals.compute(vectors.frame_from_json(frame), ctx.spec) for frame in frames[:300]]
    # Left 눈: 처음 299개는 폭 3 px(쓸 수 없음, EAR 0.0), 300번째만 폭 44 px에 EAR 0.30. Right 눈은 늘 0.28.
    assert [round(s.eye_w_l, 6) for s in collected] == [3.0] * 299 + [44.0]
    assert [s.ear_l for s in collected[:299]] == [0.0] * 299 and collected[299].ear_l == pytest.approx(0.30, abs=1e-9)
    assert {round(s.ear_r, 6) for s in collected} == {0.28} and {round(s.eye_w_r, 6) for s in collected} == {44.0}
    assert {(round(s.blink_l, 6), round(s.blink_r, 6), round(s.jaw_open, 6)) for s in collected} == {(0.2, 0.24, 0.05)}
    assert {_pitch(frame) for frame in frames} == {0.0}

    baseline = vector["expected"]["baseline"]
    # 60초 마감(60000의 프레임이 오면 그 프레임을 넣기 전에 검사한다): 얼굴 프레임 299개(< 300), Left 눈의 표본 0개. 연장.
    assert sum(1 for t in times if t < 60000) == 299
    # 90초 마감: 얼굴 프레임이 정확히 300개('이상'이라 통과). 얼굴 시간: 0~59400의 298장 x 200 ms, 59600의 프레임은
    # 다음 프레임(60000)까지 400 ms 가운데 200 ms, 60000의 프레임은 다음 프레임(90000)까지 가운데 200 ms.
    # 298 x 200 + 200 + 200 = 60000 ms, 비율 60000 / 90000 = 0.6667 ≥ 0.5.
    assert sum(1 for t in times if t < 90000) == 300
    held = [min(later - earlier, 200) for earlier, later in zip(times[:301], times[1:301])]
    assert held == [200] * 300 and sum(held) == 60000 and 60000 / 90000 >= 0.5
    assert (baseline["t_complete_ms"], baseline["status"], baseline["t_ms"]) == (90000, BASELINE_OK, 90001)
    # E_l: 표본 하나(0.30)의 90 분위수는 ceil(0.9 x 1) = 1번째 = 0.30. E_r: 300개 모두 0.28.
    assert (baseline["e_l"], baseline["e_r"]) == pytest.approx((0.30, 0.28), abs=1e-9)
    assert (baseline["b_l"], baseline["b_r"], baseline["jaw0"]) == pytest.approx((0.20, 0.24, 0.05), abs=1e-9)
    assert (baseline["gaze_h0"], baseline["gaze_v0"]) == pytest.approx((0.0, 0.0), abs=1e-9)
    assert baseline["r0"] == pytest.approx([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], abs=1e-9)
    # 틀린 구현: 프레임 수를 '초과'로 세면 90초에도 완성되지 않는다(다음 마감 120초는 세션 밖이다). 쓸 수 없는 Left 눈의
    # EAR 0.0을 표본에 넣으면 표본 300개의 270번째(ceil(0.9 x 300))가 0.0이라 E_l = 0.0 < e_min 0.12: 완성되지 않는다.
    assert math.ceil(0.9 * 300) == 270 and 270 <= 299 and ctx.spec.baseline.e_min == 0.12

    rows = vector["expected"]["closure"]
    assert [row["provisional"] for row in rows] == [True] * 300 + [False] * 3
    same, again, shut = rows[-3:]
    for row in (same, again):
        assert (row["closure"], row["pitch_rel"], row["jaw_rel"]) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    # 마지막 프레임: Left EAR 0.30·(1 − 0.65 x 0.6) = 0.183, eyeBlink 0.20 + 0.40 x 0.6 = 0.44. Right EAR 0.28·0.61 = 0.1708,
    # eyeBlink 0.24 + 0.36 x 0.6 = 0.456. 두 눈 모두 c = 0.6이므로 closure 0.6. E_l이 0.30이 아니면 달라진다.
    last = frames[-1]["bs"]
    assert (last["eyeBlinkLeft"], last["eyeBlinkRight"]) == pytest.approx((0.44, 0.456), abs=1e-9)
    assert shut["closure"] == pytest.approx(0.6, abs=1e-9) and shut["closure_valid"]


def test_blink_span_floor_in_the_closure(stored: dict[str, dict[str, Any]], ctx: Context) -> None:
    vector = stored["closure_blink_span_floor"]
    baseline = vector["baseline"]
    assert (baseline["e_l"], baseline["e_r"], baseline["b_l"], baseline["b_r"]) == (0.30, 0.30, 0.50, 0.30)
    assert (ctx.spec.closure.b_closed, ctx.spec.closure.b_span_min, ctx.spec.closure.w) == (0.6, 0.2, 0.5)
    blinks = [(frame["bs"]["eyeBlinkLeft"], frame["bs"]["eyeBlinkRight"]) for frame in vector["frames"]]
    assert blinks == [(0.60, 0.30), (0.66, 0.45), (0.75, 0.54)]
    # EAR은 E와 같아 a_e = 0. Left: 분모 max(0.6 − 0.5, 0.2) = 0.2라 s = 0.5, 0.8, 1.25 -> 1. Right: 분모 0.3이라
    # s = 0, 0.5, 0.8. c_e = 0.5·s_e, closure는 두 눈 평균(yaw_rel 0).
    closures = [row["closure"] for row in vector["expected"]["closure"]]
    assert closures == pytest.approx([(0.25 + 0.0) / 2, (0.4 + 0.25) / 2, (0.5 + 0.4) / 2], abs=1e-9)
    assert closures == pytest.approx([0.125, 0.325, 0.45], abs=1e-9)
    # 하한이 없으면 Left의 분모가 0.1이라 s = 1, 1, 1: closure 0.25, 0.375, 0.45. 앞의 두 프레임에서 드러난다.


def test_yawn_counter_edges(stored: dict[str, dict[str, Any]]) -> None:
    vector = stored["steps_yawn_edges"]
    frames = vector["frames"]
    assert [frame["t_ms"] - 1 for frame in frames] == sorted([*range(0, 15200, 200), 6500])
    wide = _held(frames, 0.5, _jaw)  # 시작 임계 0.40 이상
    ajar = _held(frames, 0.3, _jaw)  # 끝 임계 0.25 이상, 시작 임계 미만
    assert ajar == [*range(0, 2400, 200), *range(10000, 11200, 200)]
    assert wide == [
        *range(2600, 4200, 200),  # B: 1400 ms
        *range(4800, 5800, 200),  # B: 800 ms 뒤에 돌아와 0.8초 더
        *(6000, 6200, 6400, 6500),  # C
        *range(7000, 8400, 200),  # C: 정확히 500 ms 뒤에 돌아온다
        *range(9000, 10000, 200),  # D
        *range(12000, 14000, 200),  # E
    ]
    assert _no_face(vector) == [4200, 4400, 4600, 6600, 6800]
    assert all(_jaw(frame) in (0.0, 0.3, 0.5) for frame in frames if frame["face"])
    # A: 첫 프레임부터 0.3. 0.40에 닿지 않았으므로 2200 ms를 머물러도 하품이 아니다.
    # B: 2600에 시작. 4000 다음의 유효 프레임은 4800(800 ms > 500): 취소하고 4800에 새로 시작, 5600까지 800 ms.
    #    취소하지 않으면 4800 − 2600 = 2200 ≥ 2000으로 그 프레임에서 센다.
    # C: 6000에 시작. 6500 다음의 유효 프레임은 7000(정확히 500 ms): 이어진다. 8000 − 6000 = 2000: 8000의 프레임(스텝 9).
    # D: 9000에 시작, 10000부터 0.3(끝 임계 이상). 11000 − 9000 = 2000: 11000의 프레임(스텝 12). 11200에 닫는다.
    # E: 12000에 시작, 14000 − 12000 = 2000이지만 그 프레임의 값이 0이라 세지 않는다.
    assert _feature(vector, "yawn_count_60") == [0.0] * 8 + [1.0] * 3 + [2.0] * 4
    assert [step["k"] for step in vector["expected"]["steps"]] == list(range(1, 16))
    assert _feature(vector, "nod_count_60") == [0.0] * 15 and vector["expected"]["runs"] == []


def test_nod_counter_edges(stored: dict[str, dict[str, Any]]) -> None:
    vector = stored["steps_nod_edges"]
    frames = vector["frames"]
    assert [frame["t_ms"] - 1 for frame in frames] == sorted([*range(0, 8200, 200), 7700])
    assert all(_pitch(frame) in (0.0, 20.0) for frame in frames if frame["face"])
    assert _held(frames, 20.0, _pitch) == [200, *range(1000, 4000, 200), 5000, 5200, 5400, 7000, 7200, 7700]
    assert _no_face(vector) == [5600, 5800, 6000, 6200, 7400, 7600]
    # 느린 평균 m의 범위: 숙인 표본은 모두 200 ms 간격(α = 1/51)이고 가장 긴 숙임이 15표본이다.
    # m ≤ 20·(1 − (50/51)^15) + (앞에서 남은 값) < 6.3이므로 숙인 프레임의 d는 13.7 이상(시작), 든 프레임의 d는 0 이하(끝).
    assert 20 * (1 - (50 / 51) ** 15) == pytest.approx(5.14, abs=0.01)
    # A: 200에 시작, 400에 끝. 길이 200 ms = min_duration_ms: 센다(스텝 1).
    # B: 1000에 시작, 4000에 끝. 길이 3000 ms = max_duration_ms: 센다(4000의 프레임은 스텝 5).
    # C: 5000에 시작. 5400 뒤로 얼굴이 없다. 6000의 프레임에서 600 ms > 500: 취소. 세지 않는다.
    #    얼굴 없는 프레임을 pitch 0의 표본으로 넣는 구현은 5600에서 길이 600 ms로 끝내 스텝 6에서 하나를 더 센다.
    # D: 7000에 시작. 7200 다음의 유효 프레임은 7700(정확히 500 ms): 이어진다. 7800에 끝, 길이 800 ms: 센다(스텝 8).
    assert _feature(vector, "nod_count_60") == [1.0] * 4 + [2.0] * 3 + [3.0]
    assert [step["k"] for step in vector["expected"]["steps"]] == list(range(1, 9))
    assert _feature(vector, "yawn_count_60") == [0.0] * 8


def test_nod_slow_mean_sign_update_and_rate(stored: dict[str, dict[str, Any]]) -> None:
    vector = stored["steps_nod_slow_mean"]
    frames = vector["frames"]
    times = [frame["t_ms"] - 1 for frame in frames]
    assert times == [0, 100, 200, 300, *range(400, 1400, 100), *range(3000, 8200, 200)]
    assert all(frame["face"] for frame in frames)
    assert [_pitch(frame) for frame in frames] == [12.0] * 4 + [20.0] * 31 + [25.25] * 2 + [20.0] * 3
    # m의 초기값은 첫 표본 12. 20도의 표본 31개: 400~1300의 열 개는 dt = 100(α = 100/10100 = 1/101), 3000의 하나는
    # 간격 1700을 200으로 잘라 α = 200/10200 = 1/51, 3200~7000의 스무 개는 dt = 200(α = 1/51).
    # 표본마다 (20 − m)이 (1 − α)배가 된다.
    m = 20 - 8 * (100 / 101) ** 10 * (50 / 51) ** 21
    assert m == pytest.approx(15.2217, abs=5e-5)
    d_start = 25.25 - m  # 7200의 프레임
    assert d_start == pytest.approx(10.0283, abs=5e-5) and d_start >= 10.0
    # 25.25도의 두 표본 뒤 m = 25.25 − d_start·(50/51)², 7600의 프레임에서 d = 20 − m < 5: 끝, 길이 400 ms.
    d_end = 20 - (25.25 - d_start * (50 / 51) ** 2)
    assert d_end == pytest.approx(4.3889, abs=5e-5) and d_end < 5.0
    # 그 전에는 d가 10에 닿지 않는다: 12도의 표본은 d = 0, 20도의 첫 표본은 d = 20 − 12 = 8이고 그 뒤로 줄어든다.
    assert _feature(vector, "nod_count_60") == [0.0] * 7 + [1.0]
    assert [step["k"] for step in vector["expected"]["steps"]] == list(range(1, 9))
    # 이 벡터가 가르는 틀린 구현(모두 스텝 8의 값이 0이 된다). 25.25도에서의 d가 시작 임계 10에 못 미친다:
    # α = dt / 10000: 0.99^10·0.98^21
    assert 25.25 - (20 - 8 * 0.99**10 * 0.98**21) == pytest.approx(9.984, abs=1e-3)
    # α = dt / (10000 − dt): (98/99)^10·(48/49)^21
    assert 25.25 - (20 - 8 * (98 / 99) ** 10 * (48 / 49) ** 21) == pytest.approx(9.938, abs=1e-3)
    # dt를 자르지 않음: 3000의 표본에서 α = 1700 / 11700
    assert 25.25 - (20 - 8 * (100 / 101) ** 10 * (1 - 1700 / 11700) * (50 / 51) ** 20) == pytest.approx(9.416, abs=1e-3)
    # dt를 늘 200으로 둠(직전 표본의 시각을 갱신하지 않거나 빼기를 더하기로 쓴 경우): (50/51)^31
    assert 25.25 - (20 - 8 * (50 / 51) ** 31) == pytest.approx(9.580, abs=1e-3)
    # m을 갱신하지 않으면 m = 12: 20도에서 d = 8(시작 안 함), 25.25도에서 13.25(시작), 다시 20도에서 8 ≥ 5(끝나지 않음).
    # d = pitch_rel + m이면 둘째 프레임에서 24로 시작해 끝나지 않는다. m을 0에서 시작하면 첫 프레임에서 d = 12로
    # 시작하고 m이 0 근처에 머물러 20도에서도 d가 5 아래로 내려가지 않는다(스텝 1의 값은 0으로 같지만 스텝 8이 0이다).


def test_nod_detector_restarts_after_a_gap_over_ten_seconds(stored: dict[str, dict[str, Any]]) -> None:
    vector = stored["steps_nod_after_long_gap"]
    frames = vector["frames"]
    times = [frame["t_ms"] - 1 for frame in frames]
    assert times == [*range(0, 1200, 200), *range(11400, 14200, 200)]
    assert max(later - earlier for earlier, later in zip(times, times[1:])) == 10400  # > reset_gap_ms 10000
    assert all(frame["face"] for frame in frames)
    assert [_pitch(frame) for frame in frames] == [0.0] * 6 + [12.0] * 5 + [0.0] * 9
    steps = vector["expected"]["steps"]
    # 끊김 뒤에는 새 프레임이 속한 스텝(12)부터 다시 쌓는다. 스텝 2~11은 발행되지 않는다.
    assert [step["k"] for step in steps] == [1, 12, 13, 14]
    assert [step["t_ms"] - 1 for step in steps] == [1000, 12000, 13000, 14000]
    # 끊김 뒤 첫 표본(12도)이 m의 새 초기값이라 d = 0. 12400에 0도로 돌아오면 d = 0 − m ≈ −11.1: 끄덕임이 없다.
    # m을 끊김 앞의 0으로 두면 11400에 d = 12로 시작해 12400에 끝난다(1000 ms): 스텝 13과 14의 값이 1이 된다.
    assert _feature(vector, "nod_count_60") == [0.0, 0.0, 0.0, 0.0]
    names = list(FEATURE_NAMES)
    # 끊김 뒤의 스텝 12: 누적을 11000에서 다시 시작했고 얼굴 시간은 11400~12000의 600 ms다.
    assert steps[1]["values"][names.index("face_missing_ratio_60")] == (60000 - 600) / 60000
    assert not steps[1]["full_lookback"]


def test_step_keeps_the_longest_of_two_closures(stored: dict[str, dict[str, Any]]) -> None:
    vector = stored["steps_two_closures_in_one_step"]
    frames = vector["frames"]
    assert [frame["t_ms"] - 1 for frame in frames] == list(range(0, 2100, 100))
    assert _held(frames, 0.9, _lids) == [200, 300, 400, 500, 800, 1100, 1400, 1500, 1600, 1700, 1800]
    assert all(_lids(frame) in (0.0, 0.9) for frame in frames)
    # 0.0 -> 0.9(100 ms)는 56 ms(0.5 / 0.9)에, 0.9 -> 0.0은 44 ms(0.4 / 0.9)에 0.5를 지난다.
    assert [(run["t_ms"], *_run_fields(run)[:3]) for run in vector["expected"]["runs"]] == [
        (601, 157, 545, 388),  # 156~544, 600의 프레임에서 끝(스텝 1)
        (901, 757, 845, 88),  # 756~844, 900의 프레임에서 끝(스텝 1)
        (1201, 1057, 1145, 88),  # 스텝 2
        (1901, 1357, 1845, 488),  # 스텝 2
    ]
    # 스텝의 값은 그 스텝에 귀속된 감김 가운데 가장 긴 것. 나중 것으로 덮어쓰면 [88, 488], 먼저 것을 남기면 [388, 388].
    assert _feature(vector, "max_closure_ms_60") == [388.0, 488.0]
    assert _feature(vector, "max_closure_ms_10") == [388.0, 488.0]
    assert _feature(vector, "blink_dur_mean_60") == [(388 + 88) / 2, (388 + 88 + 88 + 488) / 4] == [238.0, 263.0]
    # 표준편차: sqrt((388² + 88²)/2 − 238²) = sqrt(22500) = 150, sqrt((388² + 88² + 88² + 488²)/4 − 263²) = sqrt(31875)
    assert _feature(vector, "blink_dur_std_60") == pytest.approx([150.0, math.sqrt(31875.0)], abs=1e-9)
    assert _feature(vector, "long_closure_count_60") == [0.0, 0.0]  # 500 ms 이상인 감김이 없다
    assert _feature(vector, "blink_rate_60") == [0.0, 0.0]  # 유효 시간이 5초에 못 미친다


def test_closure_above_the_cap(stored: dict[str, dict[str, Any]]) -> None:
    vector = stored["steps_closure_above_cap"]
    frames = vector["frames"]
    assert [frame["t_ms"] - 1 for frame in frames] == list(range(0, 13200, 200))
    assert _held(frames, 0.9, _lids) == list(range(400, 12000, 200))
    (run,) = vector["expected"]["runs"]
    # 200 -> 400(200 ms)의 111 ms(200 x 0.5 / 0.9)에 시작: 311. 11800 -> 12000의 89 ms(200 x 0.4 / 0.9)에 끝: 11889.
    assert (run["t_ms"], *_run_fields(run)[:4]) == (12001, 312, 11890, 11889 - 311, False)
    assert run["duration_ms"] == 11578
    # 스텝 k(1~12)의 마지막 프레임은 1000k − 200이고 그때 진행 중인 길이는 1000k − 200 − 311.
    ongoing = [1000 * k - 511 for k in range(1, 13)]
    assert ongoing[9:] == [9489, 10489, 11489]
    capped = [489.0, 1489.0, 2489.0, 3489.0, 4489.0, 5489.0, 6489.0, 7489.0, 8489.0, 9489.0, 10000.0, 10000.0]
    # 스텝 13: 끝난 감김(11578 ms)이 그 스텝의 통에 들어갔다. 역시 10000.
    assert _feature(vector, "max_closure_ms_60") == [*capped, 10000.0]
    assert _feature(vector, "max_closure_ms_10") == [*capped, 10000.0]
    assert _feature(vector, "long_closure_count_60") == [0.0] + [1.0] * 12  # 489 ms는 긴 감김이 아니다
    assert _feature(vector, "blink_dur_mean_60") == [0.0] * 12 + [2000.0]  # 깜빡임 길이의 상한 2000 ms
    assert _feature(vector, "blink_dur_std_60") == [0.0] * 13


def test_closure_of_exactly_500_ms_is_a_long_closure(stored: dict[str, dict[str, Any]]) -> None:
    vector = stored["steps_closure_exactly_500ms"]
    frames = vector["frames"]
    assert [frame["t_ms"] - 1 for frame in frames] == sorted([*range(0, 3100, 100), 1950])
    assert all(_lids(frame) in (0.1, 0.9) for frame in frames)
    assert _held(frames, 0.9, _lids) == [200, 300, 400, 500, 600, 1500, 1600, 1700, 1800, 1900, 1950, 2000]
    # 0.1 <-> 0.9의 100 ms 쌍에서 0.5는 정확히 가운데(50 ms)다: 150~650(500 ms), 1450~2050(600 ms).
    assert [(run["t_ms"], *_run_fields(run)[:3]) for run in vector["expected"]["runs"]] == [
        (701, 151, 651, 500),
        (2101, 1451, 2051, 600),
    ]
    # 스텝 1: 끝난 감김이 정확히 500 ms. 스텝 2: 마지막 프레임 1950에서 진행 중인 길이 1950 − 1450 = 500 ms가 하나 더.
    # 스텝 3: 둘째 감김이 600 ms로 끝났다. 끝난 감김을 '초과'로 세면 [0, 1, 1], 진행 중인 감김을 '초과'로 세면 [1, 1, 2]다.
    assert _feature(vector, "long_closure_count_60") == [1.0, 2.0, 2.0]
    assert _feature(vector, "max_closure_ms_60") == [500.0, 500.0, 600.0]
    assert _feature(vector, "blink_dur_mean_60") == [500.0, 500.0, (500 + 600) / 2]


def test_pitch_std_starts_at_exactly_5000_ms_of_face_time(stored: dict[str, dict[str, Any]]) -> None:
    vector = stored["steps_pitch_std_at_5000ms"]
    frames = vector["frames"]
    assert [frame["t_ms"] - 1 for frame in frames] == list(range(0, 6200, 200))
    assert all(frame["face"] for frame in frames)
    assert [_pitch(frame) for frame in frames] == [0.0] * 10 + [5.0] * 21
    # 프레임마다 200 ms를 대표하므로 스텝 k가 끝날 때 얼굴 시간은 정확히 1000k ms다.
    assert _feature(vector, "face_missing_ratio_60") == [(60000 - 1000 * k) / 60000 for k in range(1, 7)]
    spread = _feature(vector, "pitch_std_60")
    # 스텝 1~4: 얼굴 시간이 5000 ms 미만이라 0(스텝 3, 4에서는 pitch가 이미 바뀌었다).
    assert spread[:4] == [0.0, 0.0, 0.0, 0.0]
    # 스텝 5: 0도 2000 ms + 5도 3000 ms. 평균 15000/5000 = 3, 제곱 평균 75000/5000 = 15, 분산 15 − 9 = 6.
    assert spread[4] == pytest.approx(math.sqrt(6.0), abs=1e-9) and spread[4] == pytest.approx(2.449490, abs=1e-6)
    # 스텝 6: 0도 2000 ms + 5도 4000 ms. 평균 10/3, 제곱 평균 50/3, 분산 50/3 − 100/9 = 50/9.
    assert spread[5] == pytest.approx(math.sqrt(50.0 / 9.0), abs=1e-9) and spread[5] == pytest.approx(2.357023, abs=1e-6)
