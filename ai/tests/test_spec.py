"""공용 계약(자료형, feature_spec, 규칙 설정, 실험 설정)의 테스트.

상수의 기댓값은 docs/PLAN.md에서 손으로 옮겨 적은 것이다.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from drowsy import spec as S
from drowsy import types as T
from drowsy.config import DropSimConfig, ExperimentConfig, ModelGate, QualityGate, load_config

# pipeline 블록(이식 코드의 상수)이 바뀌면 이 값이 달라진다. 의도한 변경이면
# `python -m drowsy.spec update-hash`를 실행하고 이 값을 고친 뒤, 테스트 벡터를 다시 만들고
# 앱 팀에 새 해시를 알린다(PLAN 4.7절, 11장 E0).
PIPELINE_HASH_V0 = "650495e5c8a158db9a071457713f852af846e54e32d80352376592768f9c31a6"

MEDIAPIPE_MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"

# (이름, 구간 초, 단위, 변환) — PLAN 4.6절 표의 순서
FEATURE_TABLE: tuple[tuple[str, int, str, str], ...] = (
    ("perclos_60", 60, "ratio", "none"),
    ("perclos_10", 10, "ratio", "none"),
    ("closure_mean_60", 60, "0-1", "none"),
    ("closure_mean_10", 10, "0-1", "none"),
    ("blink_rate_60", 60, "1/min", "none"),
    ("blink_dur_mean_60", 60, "ms", "none"),
    ("blink_dur_std_60", 60, "ms", "none"),
    ("closing_vel_mean_60", 60, "1/s", "none"),
    ("opening_vel_mean_60", 60, "1/s", "none"),
    ("long_closure_count_60", 60, "count", "log1p"),
    ("max_closure_ms_60", 60, "ms", "log1p"),
    ("max_closure_ms_10", 10, "ms", "log1p"),
    ("yawn_count_60", 60, "count", "log1p"),
    ("pitch_std_60", 60, "deg", "none"),
    ("nod_count_60", 60, "count", "log1p"),
    ("face_missing_ratio_60", 60, "ratio", "none"),
    ("face_missing_ratio_10", 10, "ratio", "none"),
)


def _raw_spec(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rehash(raw: dict[str, Any]) -> dict[str, Any]:
    raw["pipeline_hash"] = S.compute_pipeline_hash(raw["pipeline"])
    return raw


# ---------------------------------------------------------------------------
# 해시
# ---------------------------------------------------------------------------


def test_pipeline_hash_is_sha256_of_canonical_json() -> None:
    # 키 정렬, 공백 없는 구분자: 손으로 쓴 정규 문자열과 같아야 한다.
    block = {"b": 1, "a": [1.5, "x", True], "c": {"z": 0.25, "y": None}}
    canonical = b'{"a":[1.5,"x",true],"b":1,"c":{"y":null,"z":0.25}}'
    assert S.compute_pipeline_hash(block) == hashlib.sha256(canonical).hexdigest()


def test_pipeline_hash_ignores_key_order() -> None:
    first = {"clock": {"step_ms": 1000, "max_hold_ms": 200}, "w": 0.5}
    second = {"w": 0.5, "clock": {"max_hold_ms": 200, "step_ms": 1000}}
    assert S.compute_pipeline_hash(first) == S.compute_pipeline_hash(second)


def test_pipeline_hash_escapes_non_ascii() -> None:
    expected = hashlib.sha256(b'{"k":"\\uac00"}').hexdigest()
    assert S.compute_pipeline_hash({"k": "가"}) == expected


def test_pipeline_hash_distinguishes_int_and_float() -> None:
    assert S.compute_pipeline_hash({"x": 25}) != S.compute_pipeline_hash({"x": 25.0})


def test_pipeline_hash_rejects_nan() -> None:
    with pytest.raises(ValueError):
        S.compute_pipeline_hash({"x": float("nan")})


def test_stored_pipeline_hash_matches_block(feature_spec: S.FeatureSpec, feature_spec_path: Path) -> None:
    raw = _raw_spec(feature_spec_path)
    assert raw["pipeline_hash"] == S.compute_pipeline_hash(raw["pipeline"])
    assert feature_spec.pipeline_hash == raw["pipeline_hash"]
    assert feature_spec.pipeline_hash == PIPELINE_HASH_V0, (
        "the pipeline block changed: update PIPELINE_HASH_V0, regenerate the test vectors "
        "and tell the app team the new pipeline_hash"
    )


def test_pipeline_hash_changes_with_any_constant(feature_spec_path: Path) -> None:
    raw = _raw_spec(feature_spec_path)
    original = S.compute_pipeline_hash(raw["pipeline"])
    raw["pipeline"]["closure"]["rho"] = 0.36
    assert S.compute_pipeline_hash(raw["pipeline"]) != original


def test_pipeline_hash_is_unaffected_by_training_statistics(feature_spec_path: Path) -> None:
    # 재학습으로 평균·표준편차·온도가 바뀌어도 pipeline_hash는 유효하다(PLAN 4.7절).
    raw = _raw_spec(feature_spec_path)
    raw["spec_version"] = 1
    raw["features"][0]["mean"] = 0.031
    raw["features"][0]["std"] = 0.07
    raw["standardization"]["temperature"] = 1.7
    spec = S.feature_spec_from_dict(raw)
    assert spec.pipeline_hash == PIPELINE_HASH_V0
    assert spec.means[0] == 0.031 and spec.stds[0] == 0.07


def test_file_sha256(tmp_path: Path) -> None:
    target = tmp_path / "abc.bin"
    target.write_bytes(b"abc")
    assert S.file_sha256(target) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


# ---------------------------------------------------------------------------
# feature_spec.json 왕복과 파일 형식
# ---------------------------------------------------------------------------


def test_feature_spec_round_trip(feature_spec: S.FeatureSpec, feature_spec_path: Path) -> None:
    raw = _raw_spec(feature_spec_path)
    assert S.feature_spec_to_dict(feature_spec) == raw
    assert S.pipeline_to_dict(feature_spec.pipeline) == raw["pipeline"]
    assert S.compute_pipeline_hash(S.pipeline_to_dict(feature_spec.pipeline)) == feature_spec.pipeline_hash
    assert S.feature_spec_from_dict(S.feature_spec_to_dict(feature_spec)) == feature_spec
    assert S.pipeline_spec_from_dict(raw["pipeline"]) == feature_spec.pipeline


def test_write_feature_spec_reproduces_file_bytes(feature_spec_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "nested" / "feature_spec.json"
    digest = S.write_feature_spec(_raw_spec(feature_spec_path), out)
    assert out.read_bytes() == feature_spec_path.read_bytes()
    assert digest == S.file_sha256(out) == S.file_sha256(feature_spec_path)


def test_feature_spec_file_format(feature_spec_path: Path) -> None:
    data = feature_spec_path.read_bytes()
    assert data.endswith(b"}\n") and not data.endswith(b"\n\n")
    assert b"\r" not in data and b"\t" not in data
    text = data.decode("utf-8")
    assert text.startswith('{\n  "spec_version": 0,\n')
    assert list(json.loads(text)) == [
        "spec_version",
        "step_sec",
        "window_steps",
        "class_order",
        "features",
        "standardization",
        "pipeline",
        "pipeline_hash",
    ]


def test_write_feature_spec_refuses_stale_hash(feature_spec_path: Path, tmp_path: Path) -> None:
    raw = _raw_spec(feature_spec_path)
    raw["pipeline"]["closure"]["rho"] = 0.4
    out = tmp_path / "feature_spec.json"
    with pytest.raises(S.SpecError, match="pipeline_hash mismatch"):
        S.write_feature_spec(raw, out)
    assert not out.exists()


# ---------------------------------------------------------------------------
# 특징 목록 (PLAN 4.6절, 5.1절)
# ---------------------------------------------------------------------------


def test_feature_names_order_windows_units_transforms(feature_spec: S.FeatureSpec) -> None:
    assert len(feature_spec.features) == 17
    for i, (name, window_sec, unit, transform) in enumerate(FEATURE_TABLE):
        feat = feature_spec.features[i]
        assert (feat.index, feat.name, feat.window_sec) == (i, name, window_sec)
        assert (feat.unit, feat.transform) == (unit, transform)
        assert feat.formula.strip() == feat.formula and len(feat.formula) > 10 and "\n" not in feat.formula
    assert feature_spec.names == tuple(row[0] for row in FEATURE_TABLE)
    assert feature_spec.names == T.FEATURE_NAMES == feature_spec.pipeline.step_features.names
    assert feature_spec.n_features == T.N_FEATURES == 17


def test_log1p_applies_to_count_and_duration_features_only(feature_spec: S.FeatureSpec) -> None:
    log1p_indices = [i for i, flag in enumerate(feature_spec.log1p_mask) if flag]
    assert log1p_indices == [9, 10, 11, 12, 14]


def test_v0_placeholders(feature_spec: S.FeatureSpec) -> None:
    assert feature_spec.spec_version == 0
    assert feature_spec.means == (0.0,) * 17
    assert feature_spec.stds == (1.0,) * 17
    assert all(feat.undefined_value == 0.0 for feat in feature_spec.features)
    assert feature_spec.standardization == S.Standardization(
        location="in_graph", z_clip=5.0, std_floor=0.001, temperature=1.0
    )


def test_top_level_contract(feature_spec: S.FeatureSpec) -> None:
    assert feature_spec.step_sec == 1
    assert feature_spec.window_steps == 120
    assert feature_spec.class_order == ("alert", "low_vigilant", "drowsy") == T.CLASS_ORDER
    assert (T.CLASS_ALERT, T.CLASS_LOW_VIGILANT, T.CLASS_DROWSY) == (0, 1, 2)


def test_feature_index_constants_match_names() -> None:
    for index, name in enumerate(T.FEATURE_NAMES):
        assert getattr(T, "F_" + name.upper()) == index
    assert T.F_PERCLOS_60 == 0 and T.F_FACE_MISSING_RATIO_10 == 16


# ---------------------------------------------------------------------------
# pipeline 블록의 상수 (PLAN 2.1, 2.2, 3.3, 4.1~4.6절)
# ---------------------------------------------------------------------------


def test_landmark_sets(pipeline_spec: S.PipelineSpec) -> None:
    lm = pipeline_spec.landmarks
    assert lm.ear_r == (33, 160, 158, 133, 153, 144)
    assert lm.ear_l == (362, 385, 387, 263, 373, 380)
    assert lm.iris_r == (468, 469, 470, 471, 472)
    assert lm.iris_l == (473, 474, 475, 476, 477)
    assert (lm.corner_outer_r, lm.corner_inner_r) == (33, 133)
    assert (lm.corner_outer_l, lm.corner_inner_l) == (263, 362)
    # 윗눈꺼풀 사슬 + 아랫눈꺼풀 사슬(바깥 꼬리에서 안쪽 꼬리까지), 조사 문서 landmarks.md의 값
    assert set(lm.contour_r) == {33, 246, 161, 160, 159, 158, 157, 173, 133, 7, 163, 144, 145, 153, 154, 155}
    assert set(lm.contour_l) == {263, 466, 388, 387, 386, 385, 384, 398, 362, 249, 390, 373, 374, 380, 381, 382}
    assert lm.contour_r[:9] == (33, 246, 161, 160, 159, 158, 157, 173, 133)
    assert lm.contour_l[:9] == (263, 466, 388, 387, 386, 385, 384, 398, 362)
    assert lm.contour_r[9:] == (155, 154, 153, 145, 144, 163, 7)
    assert lm.contour_l[9:] == (382, 381, 380, 374, 373, 390, 249)
    assert lm.face_width == (234, 454)
    required = lm.required_indices()
    # 눈 윤곽 32 + 홍채 10 + 얼굴 폭 2
    assert len(required) == 44 and list(required) == sorted(set(required))
    assert set(lm.ear_r) | set(lm.ear_l) <= set(required)


def test_blendshape_names_and_indices(pipeline_spec: S.PipelineSpec) -> None:
    bs = pipeline_spec.blendshapes
    assert (bs.eye_blink_l, bs.eye_blink_r, bs.jaw_open) == ("eyeBlinkLeft", "eyeBlinkRight", "jawOpen")
    assert bs.eye_look == (
        "eyeLookInLeft",
        "eyeLookOutLeft",
        "eyeLookUpLeft",
        "eyeLookDownLeft",
        "eyeLookInRight",
        "eyeLookOutRight",
        "eyeLookUpRight",
        "eyeLookDownRight",
    )
    # eye_look의 순서는 FrameSignals.eye_look의 고정 순서와 같다.
    for position, key in enumerate(T.EYE_LOOK_ORDER):
        direction, side = key.split("_")
        expected = "eyeLook" + direction.capitalize() + {"l": "Left", "r": "Right"}[side]
        assert bs.eye_look[position] == expected
    assert (T.EL_IN_L, T.EL_OUT_L, T.EL_UP_L, T.EL_DOWN_L) == (0, 1, 2, 3)
    assert (T.EL_IN_R, T.EL_OUT_R, T.EL_UP_R, T.EL_DOWN_R) == (4, 5, 6, 7)
    # PLAN 2.1절의 인덱스
    assert len(bs.order) == 52
    assert bs.index_of("_neutral") == 0
    assert bs.index_of("eyeBlinkLeft") == 9
    assert bs.index_of("eyeBlinkRight") == 10
    assert bs.order[11:19] == (
        "eyeLookDownLeft",
        "eyeLookDownRight",
        "eyeLookInLeft",
        "eyeLookInRight",
        "eyeLookOutLeft",
        "eyeLookOutRight",
        "eyeLookUpLeft",
        "eyeLookUpRight",
    )
    assert bs.index_of("jawOpen") == 25
    assert bs.order[51] == "noseSneerRight"
    assert len(bs.required_names()) == 11 and bs.required_names()[0] == "eyeBlinkLeft"


def test_pose_clock_signal_constants(pipeline_spec: S.PipelineSpec) -> None:
    assert pipeline_spec.pose == S.PoseSpec(
        order="Ry*Rx*Rz",
        yaw_pos="subject_left",
        pitch_pos="chin_down",
        roll_pos="ccw_in_image",
        matrix_layout="column_major",
        angle_unit="deg",
    )
    assert pipeline_spec.clock == S.ClockSpec(step_ms=1000, max_hold_ms=200, reset_gap_ms=10000)
    assert pipeline_spec.signals == S.SignalSpec(min_eye_width_px=6.0)


def test_baseline_and_closure_constants(pipeline_spec: S.PipelineSpec) -> None:
    assert pipeline_spec.baseline == S.BaselineSpec(
        duration_ms=60000,
        extend_step_ms=30000,
        max_duration_ms=180000,
        min_face_frames=300,
        min_face_ratio=0.5,
        e_min=0.12,
        e_max=0.45,
        ear_quantile=0.9,
        blink_quantile=0.1,
        blink_record_quantiles=(0.5, 0.9),
        off_axis_yaw_deg=15.0,
    )
    assert pipeline_spec.closure == S.ClosureSpec(
        rho=0.35,
        b_closed=0.6,
        b_span_min=0.2,
        w=0.5,
        far_eye_zero_yaw_deg=25.0,
        far_eye_ramp_deg=10.0,
        valid_max_abs_yaw_deg=40.0,
    )
    assert pipeline_spec.global_constants == S.GlobalConstants(
        e=0.27, b=0.22, jaw0=0.08, gaze_h0=0.0, gaze_v0=0.0, r0=T.IDENTITY_R0
    )
    assert T.IDENTITY_R0 == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def test_run_detector_and_step_feature_constants(pipeline_spec: S.PipelineSpec) -> None:
    assert pipeline_spec.run_detector == S.RunDetectorSpec(
        enter_thr=0.5, exit_thr=0.3, min_duration_ms=50, max_dt_ms=200, hold_ms=500, velocity_lookback_samples=2
    )
    assert pipeline_spec.step_features == S.StepFeatureSpec(
        names=T.FEATURE_NAMES,
        window_steps=120,
        lookback_long_ms=60000,
        lookback_short_ms=10000,
        perclos_thr=0.8,
        perclos_min_valid_ms_60=5000,
        perclos_min_valid_ms_10=2000,
        blink_rate_min_valid_ms=5000,
        blink_duration_cap_ms=2000,
        long_closure_min_ms=500,
        max_closure_cap_ms=10000,
        pitch_std_min_face_ms=5000,
    )


def test_yawn_nod_mediapipe_constants(pipeline_spec: S.PipelineSpec) -> None:
    assert pipeline_spec.yawn == S.YawnSpec(enter_thr=0.4, exit_thr=0.25, count_at_ms=2000, cancel_gap_ms=500)
    assert pipeline_spec.nod == S.NodSpec(
        enter_deg=10.0,
        exit_deg=5.0,
        min_duration_ms=200,
        max_duration_ms=3000,
        ema_tau_ms=10000,
        ema_max_dt_ms=200,
        cancel_gap_ms=500,
    )
    assert pipeline_spec.mediapipe == S.MediaPipeSpec(model_sha256=MEDIAPIPE_MODEL_SHA256, num_faces=1)


def test_timeline_follows_from_constants(feature_spec: S.FeatureSpec) -> None:
    # PLAN 4.4절: T_b = 60초면 첫 모델 출력은 239초(스텝 끝 기준 240초: 60 + 60 + 120).
    pipe = feature_spec.pipeline
    first_full_lookback_end_ms = pipe.baseline.duration_ms + pipe.step_features.lookback_long_ms
    first_model_step_end_ms = first_full_lookback_end_ms + (feature_spec.window_steps - 1) * pipe.clock.step_ms
    assert first_full_lookback_end_ms == 120000
    assert first_model_step_end_ms == 239000


# ---------------------------------------------------------------------------
# feature_spec 검증 오류
# ---------------------------------------------------------------------------


def _mutate_stale_hash(raw: dict[str, Any]) -> None:
    raw["pipeline"]["closure"]["rho"] = 0.4


def _mutate_int_for_float(raw: dict[str, Any]) -> None:
    raw["pipeline"]["closure"]["w"] = 1
    _rehash(raw)


def _mutate_bool_for_int(raw: dict[str, Any]) -> None:
    raw["pipeline"]["clock"]["step_ms"] = True
    _rehash(raw)


def _mutate_unknown_key(raw: dict[str, Any]) -> None:
    raw["pipeline"]["clock"]["extra"] = 1
    _rehash(raw)


def _mutate_missing_key(raw: dict[str, Any]) -> None:
    del raw["pipeline"]["nod"]["ema_tau_ms"]
    _rehash(raw)


def _mutate_feature_order(raw: dict[str, Any]) -> None:
    raw["features"][0], raw["features"][1] = raw["features"][1], raw["features"][0]


def _mutate_feature_name(raw: dict[str, Any]) -> None:
    raw["features"][3]["name"] = "closure_avg_10"


def _mutate_transform(raw: dict[str, Any]) -> None:
    raw["features"][9]["transform"] = "sqrt"


def _mutate_std_below_floor(raw: dict[str, Any]) -> None:
    raw["features"][0]["std"] = 0.0001


def _mutate_nan_mean(raw: dict[str, Any]) -> None:
    raw["features"][0]["mean"] = float("nan")


def _mutate_hysteresis(raw: dict[str, Any]) -> None:
    raw["pipeline"]["run_detector"]["exit_thr"] = 0.6
    _rehash(raw)


def _mutate_hold_at_reset_gap(raw: dict[str, Any]) -> None:
    raw["pipeline"]["run_detector"]["hold_ms"] = raw["pipeline"]["clock"]["reset_gap_ms"]
    _rehash(raw)


def _mutate_yawn_cancel_at_reset_gap(raw: dict[str, Any]) -> None:
    raw["pipeline"]["yawn"]["cancel_gap_ms"] = raw["pipeline"]["clock"]["reset_gap_ms"]
    _rehash(raw)


def _mutate_nod_cancel_at_reset_gap(raw: dict[str, Any]) -> None:
    raw["pipeline"]["nod"]["cancel_gap_ms"] = raw["pipeline"]["clock"]["reset_gap_ms"]
    _rehash(raw)


def _mutate_window_steps(raw: dict[str, Any]) -> None:
    raw["window_steps"] = 60


def _mutate_class_order(raw: dict[str, Any]) -> None:
    raw["class_order"] = ["alert", "drowsy", "low_vigilant"]


def _mutate_location(raw: dict[str, Any]) -> None:
    raw["standardization"]["location"] = "app"


def _mutate_contour(raw: dict[str, Any]) -> None:
    raw["pipeline"]["landmarks"]["contour_r"][1] = 33
    _rehash(raw)


def _mutate_pipeline_feature_names(raw: dict[str, Any]) -> None:
    names = raw["pipeline"]["step_features"]["names"]
    names[0], names[1] = names[1], names[0]
    _rehash(raw)


def _mutate_model_sha(raw: dict[str, Any]) -> None:
    raw["pipeline"]["mediapipe"]["model_sha256"] = "64184E22"
    _rehash(raw)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_mutate_stale_hash, "pipeline_hash mismatch"),
        (_mutate_int_for_float, r"closure\.w: expected float, got integer 1 \(write 1\.0\)"),
        (_mutate_bool_for_int, r"clock\.step_ms: expected int"),
        (_mutate_unknown_key, r"pipeline\.clock: unknown keys \['extra'\]"),
        (_mutate_missing_key, r"pipeline\.nod: missing keys \['ema_tau_ms'\]"),
        (_mutate_feature_order, r"features\[0\]\.index must be 0"),
        (_mutate_feature_name, r"features\[3\]\.name must be 'closure_mean_10'"),
        (_mutate_transform, r"features\[9\]\.transform"),
        (_mutate_std_below_floor, r"features\[0\]\.std must be >= standardization\.std_floor"),
        (_mutate_nan_mean, "finite"),
        (_mutate_hysteresis, "exit_thr < enter_thr"),
        (_mutate_hold_at_reset_gap, r"pipeline\.run_detector: need hold_ms < clock\.reset_gap_ms"),
        (_mutate_yawn_cancel_at_reset_gap, r"pipeline\.yawn: need cancel_gap_ms < clock\.reset_gap_ms"),
        (_mutate_nod_cancel_at_reset_gap, r"pipeline\.nod: need cancel_gap_ms < clock\.reset_gap_ms"),
        (_mutate_window_steps, "window_steps must match"),
        (_mutate_class_order, "class_order"),
        (_mutate_location, "standardization.location"),
        (_mutate_contour, "contour_r: need 16 distinct points"),
        (_mutate_pipeline_feature_names, "FEATURE_NAMES"),
        (_mutate_model_sha, "model_sha256"),
    ],
)
def test_invalid_feature_spec_is_rejected(feature_spec_path: Path, mutate: Any, message: str) -> None:
    raw = _raw_spec(feature_spec_path)
    mutate(raw)
    with pytest.raises(S.SpecError, match=message):
        S.feature_spec_from_dict(raw)


# ---------------------------------------------------------------------------
# 규칙 설정 (PLAN 8장, D2, D3, D6, D11, D12)
# ---------------------------------------------------------------------------


def test_rules_clock_equals_pipeline_clock(rules_config: S.RulesConfig, pipeline_spec: S.PipelineSpec) -> None:
    # 규칙 엔진은 RulesConfig만 받는다. 시간 규칙은 스텝 특징 쪽과 같은 값이어야 한다(PLAN 3.3절).
    assert rules_config.clock == S.ClockSpec(step_ms=1000, max_hold_ms=200, reset_gap_ms=10000)
    assert rules_config.clock == pipeline_spec.clock


def test_forward_zone_and_gaze_rules(rules_config: S.RulesConfig) -> None:
    assert rules_config.forward_zone == S.ForwardZoneRules(
        yaw_abs_max_deg=23.0,
        pitch_up_max_deg=25.0,
        pitch_down_max_deg=18.0,
        eye_aux_enabled=True,
        gaze_h_abs_max=0.25,
        gaze_v_abs_max=0.35,
        debounce_ms=250,
        face_loss_credit_ms=3000,
    )
    assert rules_config.gaze_away_long == S.GazeAwayLongRules(onset_ms=3000)
    # D6: 초기화는 꺼져 있고, 켤 때 쓰는 값만 둔다.
    assert rules_config.vats == S.VatsRules(
        window_ms=30000, cumulative_ms=10000, reset_enabled=False, reset_forward_ms=2000
    )


def test_closure_rules(rules_config: S.RulesConfig, pipeline_spec: S.PipelineSpec) -> None:
    closure = rules_config.closure
    assert (closure.detector.enter_thr, closure.detector.exit_thr) == (0.7, 0.5)
    assert (closure.microsleep_ms, closure.sleep_ms) == (1000, 3000)
    # 같은 검출기를 임계값만 바꿔 두 번 쓴다(PLAN 4.5절): 나머지 설정은 특징용과 같아야 한다.
    assert dataclasses.replace(closure.detector, enter_thr=0.5, exit_thr=0.3) == pipeline_spec.run_detector
    # 규칙용 임계값이 특징용보다 높다(PLAN 8.1절).
    assert closure.detector.enter_thr > pipeline_spec.run_detector.enter_thr
    assert closure.detector.exit_thr > pipeline_spec.run_detector.exit_thr


def test_drowsiness_rules(rules_config: S.RulesConfig) -> None:
    drowsy = rules_config.drowsiness
    # D11: 교차검증 기준을 통과하기 전의 기본 판정원은 PERCLOS 규칙이다.
    assert drowsy.source == "perclos" == T.DROWSY_SOURCE_PERCLOS
    assert drowsy.model == S.ModelDrowsyRules(onset_p=0.6, onset_steps=10, end_p=0.4, end_steps=10, refractory_ms=60000)
    assert drowsy.perclos == S.PerclosDrowsyRules(
        onset_ratio=0.15, onset_steps=10, end_ratio=0.10, end_steps=10, refractory_ms=0
    )
    assert rules_config.unable_to_analyze == S.UnableToAnalyzeRules(window_ms=10000, min_face_ratio=0.5)


def test_event_policies(rules_config: S.RulesConfig) -> None:
    # (종류, 심각도, 재경고 대기 ms, 반복 ms, 최소 속도 km/h) — PLAN 8.2절 표와 D3
    table = (
        ("calibrating", 0, 0, 0, 0.0),
        ("unable_to_analyze", 1, 30000, 0, 0.0),
        ("gaze_away_long", 2, 10000, 0, 20.0),
        ("vats", 2, 30000, 0, 20.0),
        ("drowsy", 2, 300000, 0, 50.0),
        ("microsleep", 3, 5000, 0, 20.0),
        ("sleep", 3, 0, 3000, 20.0),
    )
    assert tuple(row[0] for row in table) == T.EVENT_TYPES
    for event_type, severity, cooldown_ms, repeat_ms, min_speed_kmh in table:
        expected = S.EventPolicy(
            severity=severity, cooldown_ms=cooldown_ms, repeat_ms=repeat_ms, min_speed_kmh=min_speed_kmh
        )
        assert rules_config.events.get(event_type) == expected
        assert getattr(rules_config.events, event_type) == expected
    with pytest.raises(KeyError):
        rules_config.events.get("yawn")
    assert rules_config.alert_priority[:5] == ("sleep", "microsleep", "gaze_away_long", "vats", "drowsy")
    assert sorted(rules_config.alert_priority) == sorted(T.EVENT_TYPES)


def test_rules_export_hash(rules_config: S.RulesConfig, tmp_path: Path) -> None:
    out = tmp_path / "export" / "rules.json"
    digest = S.export_rules_json(rules_config, out)
    data = out.read_bytes()
    assert digest == hashlib.sha256(data).hexdigest()
    assert digest == S.file_sha256(out) == S.rules_hash(rules_config)
    assert data == S.rules_to_json(rules_config).encode("utf-8")
    assert data.endswith(b"}\n") and b"\r" not in data
    # 같은 설정은 항상 같은 바이트, 값이 하나라도 다르면 다른 해시
    assert S.export_rules_json(rules_config, tmp_path / "again.json") == digest
    changed = S.apply_overrides(rules_config, {"vats.reset_enabled": True})
    assert S.rules_hash(changed) != digest


def test_rules_json_round_trip(rules_config: S.RulesConfig, tmp_path: Path) -> None:
    out = tmp_path / "rules.json"
    S.export_rules_json(rules_config, out)
    assert S.load_rules(out) == rules_config
    assert S.rules_from_dict(S.rules_to_dict(rules_config)) == rules_config
    exported = json.loads(out.read_text(encoding="utf-8"))
    assert list(exported) == [
        "clock",
        "forward_zone",
        "gaze_away_long",
        "vats",
        "closure",
        "drowsiness",
        "unable_to_analyze",
        "events",
        "alert_priority",
    ]
    assert exported["closure"]["detector"]["enter_thr"] == 0.7
    assert exported["drowsiness"]["source"] == "perclos"
    assert exported["events"]["sleep"] == {"severity": 3, "cooldown_ms": 0, "repeat_ms": 3000, "min_speed_kmh": 20.0}


def test_rules_accept_integers_for_float_fields(rules_config: S.RulesConfig) -> None:
    # YAML에서는 23과 23.0을 구분해 쓰기 어렵다. 규칙 설정은 정수를 float로 받아들인다.
    plain = S.rules_to_dict(rules_config)
    plain["forward_zone"]["yaw_abs_max_deg"] = 23
    loaded = S.rules_from_dict(plain)
    assert loaded == rules_config
    assert type(loaded.forward_zone.yaw_abs_max_deg) is float


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"drowsiness.source": "both"}, "drowsiness.source"),
        ({"closure.detector.exit_thr": 0.8}, "exit_thr < enter_thr"),
        ({"closure.sleep_ms": 500}, "microsleep_ms <= sleep_ms"),
        ({"vats.cumulative_ms": 40000}, "cumulative_ms <= window_ms"),
        ({"drowsiness.model.end_p": 0.7}, "end_p < onset_p"),
        ({"drowsiness.perclos.end_ratio": 0.2}, "end_ratio < onset_ratio"),
        ({"events.sleep.severity": 4}, "severity"),
        ({"alert_priority": ["sleep", "microsleep"]}, "alert_priority"),
        ({"forward_zone.debounce_ms": 250.0}, "expected int"),
        ({"vats.reset_enabled": 1}, "expected bool"),
        ({"clock.max_hold_ms": 20000}, "max_hold_ms < reset_gap_ms"),
        ({"closure.detector.hold_ms": 10000}, r"closure\.detector: need hold_ms < clock\.reset_gap_ms"),
        ({"vats.no_such_key": 1}, "override path not found"),
    ],
)
def test_invalid_rules_are_rejected(rules_config: S.RulesConfig, overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(S.SpecError, match=message):
        S.apply_overrides(rules_config, overrides)


def test_run_detector_hold_must_stay_below_the_reset_gap(
    rules_config: S.RulesConfig, pipeline_spec: S.PipelineSpec
) -> None:
    # 초기화를 일으키는 끊김(reset_gap_ms 초과)은 언제나 진행 중인 감김을 자르는 끊김(hold_ms 초과)이기도 해야 한다.
    # 그래야 끊김 앞뒤의 감김이 검출기의 reset() 호출과 무관하게 서로 다른 감김이다. 두 검출기 모두에 요구한다.
    # 기본값(500과 10000)은 통과한다.
    for detector, clock in (
        (pipeline_spec.run_detector, pipeline_spec.clock),
        (rules_config.closure.detector, rules_config.clock),
    ):
        assert (detector.hold_ms, clock.reset_gap_ms) == (500, 10000)

    # 경계: reset_gap_ms보다 1 작은 값은 통과하고, 같은 값과 그보다 큰 값은 거부한다.
    assert S.apply_overrides(rules_config, {"closure.detector.hold_ms": 9999}).closure.detector.hold_ms == 9999
    assert S.apply_overrides(pipeline_spec, {"run_detector.hold_ms": 9999}).run_detector.hold_ms == 9999
    for hold_ms in (10000, 10001, 20000):
        told = rf"need hold_ms < clock\.reset_gap_ms \(hold_ms is {hold_ms}, reset_gap_ms is 10000\)"
        with pytest.raises(S.SpecError, match=r"^closure\.detector: " + told):
            S.apply_overrides(rules_config, {"closure.detector.hold_ms": hold_ms})
        with pytest.raises(S.SpecError, match=r"^pipeline\.run_detector: " + told):
            S.apply_overrides(pipeline_spec, {"run_detector.hold_ms": hold_ms})

    # reset_gap_ms를 내려도 같은 검사에 걸린다(max_hold_ms 200 < 500이라 clock 자체의 검사는 통과한다).
    told = r"need hold_ms < clock\.reset_gap_ms \(hold_ms is 500, reset_gap_ms is 500\)"
    with pytest.raises(S.SpecError, match=r"^closure\.detector: " + told):
        S.apply_overrides(rules_config, {"clock.reset_gap_ms": 500})
    with pytest.raises(S.SpecError, match=r"^pipeline\.run_detector: " + told):
        S.apply_overrides(pipeline_spec, {"clock.reset_gap_ms": 500})
    assert S.apply_overrides(rules_config, {"clock.reset_gap_ms": 501}).clock.reset_gap_ms == 501

    # 파일로 읽을 때도 같다: 내보낸 rules.json의 값을 바꾼 내용은 읽히지 않는다.
    plain = S.rules_to_dict(rules_config)
    plain["closure"]["detector"]["hold_ms"] = 10000
    with pytest.raises(S.SpecError, match="hold_ms < clock.reset_gap_ms"):
        S.rules_from_dict(plain)


def test_yawn_and_nod_cancel_gaps_must_stay_below_the_reset_gap(pipeline_spec: S.PipelineSpec) -> None:
    # 초기화를 일으키는 끊김(reset_gap_ms 초과)은 언제나 진행 중인 하품·끄덕임 판정을 취소하는 끊김(cancel_gap_ms 초과)
    # 이기도 해야 한다. 그래야 끊김 앞뒤의 하품(끄덕임)이 계수기의 reset() 호출과 무관하게 서로 다른 이벤트다.
    # 기본값(500, 500, 10000)은 통과한다.
    assert (pipeline_spec.yawn.cancel_gap_ms, pipeline_spec.nod.cancel_gap_ms) == (500, 500)
    assert pipeline_spec.clock.reset_gap_ms == 10000

    for block in ("yawn", "nod"):
        path = f"{block}.cancel_gap_ms"
        # 경계: reset_gap_ms보다 1 작은 값은 통과하고, 같은 값과 그보다 큰 값은 거부한다. 오류 문장에 두 값이 적힌다.
        accepted = S.apply_overrides(pipeline_spec, {path: 9999})
        assert getattr(accepted, block).cancel_gap_ms == 9999
        for cancel_gap_ms in (10000, 10001, 20000):
            told = (
                rf"^pipeline\.{block}: need cancel_gap_ms < clock\.reset_gap_ms "
                rf"\(cancel_gap_ms is {cancel_gap_ms}, reset_gap_ms is 10000\)$"
            )
            with pytest.raises(S.SpecError, match=told):
                S.apply_overrides(pipeline_spec, {path: cancel_gap_ms})
        # 다른 쪽의 값은 이 블록의 검사에 영향을 주지 않는다.
        other = "nod" if block == "yawn" else "yawn"
        with pytest.raises(S.SpecError, match=rf"^pipeline\.{other}: need cancel_gap_ms"):
            S.apply_overrides(pipeline_spec, {path: 9999, f"{other}.cancel_gap_ms": 10000})

    # reset_gap_ms를 내려도 걸린다. 501이면 감김 검출기의 hold_ms(500)와 두 cancel_gap_ms(500)가 모두 통과한다.
    assert S.apply_overrides(pipeline_spec, {"clock.reset_gap_ms": 501}).clock.reset_gap_ms == 501
    lowered = {"clock.reset_gap_ms": 450, "run_detector.hold_ms": 300, "yawn.cancel_gap_ms": 300}
    told = r"^pipeline\.nod: need cancel_gap_ms < clock\.reset_gap_ms \(cancel_gap_ms is 500, reset_gap_ms is 450\)$"
    with pytest.raises(S.SpecError, match=told):
        S.apply_overrides(pipeline_spec, lowered)
    lowered = {"clock.reset_gap_ms": 450, "run_detector.hold_ms": 300, "nod.cancel_gap_ms": 300}
    told = r"^pipeline\.yawn: need cancel_gap_ms < clock\.reset_gap_ms \(cancel_gap_ms is 500, reset_gap_ms is 450\)$"
    with pytest.raises(S.SpecError, match=told):
        S.apply_overrides(pipeline_spec, lowered)

    # 느린 평균의 dt 상한에는 이 조건이 없다(초기화와 얽히지 않는다. drowsy/spec.py의 주석): reset_gap_ms 이상도 읽힌다.
    assert S.apply_overrides(pipeline_spec, {"nod.ema_max_dt_ms": 20000}).nod.ema_max_dt_ms == 20000


def test_apply_overrides_returns_modified_copy(rules_config: S.RulesConfig, pipeline_spec: S.PipelineSpec) -> None:
    changed = S.apply_overrides(rules_config, {"drowsiness.source": "either", "vats.reset_enabled": True})
    assert changed.drowsiness.source == "either" and changed.vats.reset_enabled is True
    assert rules_config.drowsiness.source == "perclos" and rules_config.vats.reset_enabled is False
    assert dataclasses.replace(
        changed,
        drowsiness=dataclasses.replace(changed.drowsiness, source="perclos"),
        vats=dataclasses.replace(changed.vats, reset_enabled=False),
    ) == rules_config

    wide = S.apply_overrides(pipeline_spec, {"closure.w": 1.0})
    assert wide.closure.w == 1.0 and pipeline_spec.closure.w == 0.5
    assert S.compute_pipeline_hash(S.pipeline_to_dict(wide)) != PIPELINE_HASH_V0
    with pytest.raises(S.SpecError, match="expected float"):
        S.apply_overrides(pipeline_spec, {"closure.w": 1})
    with pytest.raises(TypeError):
        S.apply_overrides({"a": 1}, {"a": 2})


# ---------------------------------------------------------------------------
# 명령줄
# ---------------------------------------------------------------------------


def test_cli_check_passes_on_repository_spec(feature_spec_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert S.main(["check", "--spec", str(feature_spec_path)]) == 0
    out = capsys.readouterr().out
    assert f"pipeline_hash OK: {PIPELINE_HASH_V0}" in out
    assert f"feature_spec_hash: {S.file_sha256(feature_spec_path)}" in out


def test_cli_check_fails_then_update_hash_repairs(
    feature_spec_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = _raw_spec(feature_spec_path)
    raw["pipeline"]["closure"]["rho"] = 0.4
    edited = tmp_path / "feature_spec.json"
    edited.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")

    assert S.main(["check", "--spec", str(edited)]) == 1
    assert "pipeline_hash MISMATCH" in capsys.readouterr().err

    assert S.main(["update-hash", "--spec", str(edited)]) == 0
    assert "pipeline_hash updated" in capsys.readouterr().out
    assert S.main(["check", "--spec", str(edited)]) == 0
    repaired = S.load_feature_spec(edited)
    assert repaired.pipeline.closure.rho == 0.4
    assert repaired.pipeline_hash != PIPELINE_HASH_V0

    # 이미 맞는 파일은 바이트가 그대로다.
    before = edited.read_bytes()
    assert S.main(["update-hash", "--spec", str(edited)]) == 0
    assert "pipeline_hash unchanged" in capsys.readouterr().out
    assert edited.read_bytes() == before


def test_cli_update_hash_refuses_invalid_spec(feature_spec_path: Path, tmp_path: Path) -> None:
    raw = _raw_spec(feature_spec_path)
    raw["pipeline"]["run_detector"]["exit_thr"] = 0.9
    edited = tmp_path / "feature_spec.json"
    original = json.dumps(raw, indent=2) + "\n"
    edited.write_text(original, encoding="utf-8")
    assert S.main(["update-hash", "--spec", str(edited)]) == 1
    assert edited.read_text(encoding="utf-8") == original


def test_cli_export_rules(
    rules_path: Path, rules_config: S.RulesConfig, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "export" / "rules.json"
    assert S.main(["export-rules", "--rules", str(rules_path), "--out", str(out)]) == 0
    assert f"rules_hash: {S.file_sha256(out)}" in capsys.readouterr().out
    assert S.load_rules(out) == rules_config


def test_cli_runs_as_module_from_repository_root(repo_root: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "drowsy.spec", "check"], cwd=repo_root, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert f"pipeline_hash OK: {PIPELINE_HASH_V0}" in result.stdout


# ---------------------------------------------------------------------------
# 이식 대상이 쓰는 모듈은 numpy와 yaml 없이 import된다
# ---------------------------------------------------------------------------


def _module_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            found.add(node.module.split(".")[0])
    return found


def test_types_module_imports_standard_library_only(repo_root: Path) -> None:
    assert _module_level_imports(repo_root / "drowsy" / "types.py") == {"__future__", "dataclasses"}


def test_spec_module_imports_standard_library_only_at_import_time(repo_root: Path) -> None:
    imports = _module_level_imports(repo_root / "drowsy" / "spec.py")
    assert imports <= set(sys.stdlib_module_names) | {"__future__", "drowsy"}
    assert not imports & {"numpy", "yaml", "pandas", "torch"}


def test_types_and_spec_import_without_numpy_or_yaml(repo_root: Path) -> None:
    # 다른 테스트가 numpy를 이미 올렸을 수 있으므로 새 인터프리터에서 확인한다.
    code = (
        "import sys\n"
        "import drowsy, drowsy.types, drowsy.spec\n"
        "spec = drowsy.spec.load_feature_spec('config/feature_spec.json')\n"
        "assert spec.n_features == 17\n"
        "loaded = [name for name in ('numpy', 'yaml', 'pandas', 'torch') if name in sys.modules]\n"
        "assert not loaded, loaded\n"
        "drowsy.spec.load_rules('config/rules.yaml')\n"
        "assert 'yaml' in sys.modules and 'numpy' not in sys.modules\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=repo_root, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_package_version() -> None:
    import drowsy

    assert drowsy.__version__ == "0.1.0"


# ---------------------------------------------------------------------------
# 자료형 (PLAN 9장)
# ---------------------------------------------------------------------------

_FRAME_STATE = T.FrameState(
    t_ms=1,
    face=False,
    closure=0.0,
    closure_valid=False,
    provisional=True,
    yaw_rel=0.0,
    pitch_rel=0.0,
    pose_valid=False,
    gaze_h_rel=0.0,
    gaze_v_rel=0.0,
    jaw_rel=0.0,
)

_TYPE_SAMPLES: tuple[tuple[Any, tuple[str, ...]], ...] = (
    (
        T.RawFrame(t_ms=1, face=False, w=640, h=480, lm={}, bs={}, m=()),
        ("t_ms", "face", "w", "h", "lm", "bs", "m"),
    ),
    (
        T.FrameSignals(
            t_ms=1,
            face=False,
            ear_l=0.0,
            ear_r=0.0,
            eye_w_l=0.0,
            eye_w_r=0.0,
            blink_l=0.0,
            blink_r=0.0,
            eye_look=(0.0,) * 8,
            jaw_open=0.0,
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
        ),
        (
            "t_ms",
            "face",
            "ear_l",
            "ear_r",
            "eye_w_l",
            "eye_w_r",
            "blink_l",
            "blink_r",
            "eye_look",
            "jaw_open",
            "yaw",
            "pitch",
            "roll",
            "iris_h_l",
            "iris_h_r",
            "iris_v_l",
            "iris_v_r",
        ),
    ),
    (
        T.Baseline(
            e_l=0.27,
            e_r=0.27,
            b_l=0.22,
            b_r=0.22,
            r0=T.IDENTITY_R0,
            jaw0=0.08,
            gaze_h0=0.0,
            gaze_v0=0.0,
            t_complete_ms=60000,
            status=T.BASELINE_OK,
        ),
        ("e_l", "e_r", "b_l", "b_r", "r0", "jaw0", "gaze_h0", "gaze_v0", "t_complete_ms", "status"),
    ),
    (
        _FRAME_STATE,
        (
            "t_ms",
            "face",
            "closure",
            "closure_valid",
            "provisional",
            "yaw_rel",
            "pitch_rel",
            "pose_valid",
            "gaze_h_rel",
            "gaze_v_rel",
            "jaw_rel",
        ),
    ),
    (
        T.ClosureRun(
            start_ms=100,
            end_ms=250,
            duration_ms=150,
            amplitude=0.9,
            peak_closing_velocity=12.0,
            peak_opening_velocity=6.0,
            velocity_valid=True,
            truncated=False,
        ),
        (
            "start_ms",
            "end_ms",
            "duration_ms",
            "amplitude",
            "peak_closing_velocity",
            "peak_opening_velocity",
            "velocity_valid",
            "truncated",
        ),
    ),
    (
        T.StepFeatures(k=61, t_end_ms=61001, full_lookback=False, values=(0.0,) * 17),
        ("k", "t_end_ms", "full_lookback", "values"),
    ),
    (
        T.DriverStateEvent(
            type=T.EVENT_MICROSLEEP,
            severity=3,
            t_start_ms=5000,
            duration_ms=1000,
            confidence=1.0,
            source=T.SOURCE_CLOSURE,
            phase=T.PHASE_ONSET,
            provisional=False,
        ),
        ("type", "severity", "t_start_ms", "duration_ms", "confidence", "source", "phase", "provisional"),
    ),
    (
        T.PipelineOutput(state=_FRAME_STATE, runs=(), steps=(), step_probs=(), probs=None, events=(), alerts=()),
        ("state", "runs", "steps", "step_probs", "probs", "events", "alerts"),
    ),
)


@pytest.mark.parametrize(("sample", "field_names"), _TYPE_SAMPLES, ids=lambda value: type(value).__name__)
def test_value_types_are_frozen_slotted_and_match_the_plan(sample: Any, field_names: tuple[str, ...]) -> None:
    assert tuple(f.name for f in dataclasses.fields(sample)) == field_names
    assert not hasattr(sample, "__dict__")
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(sample, field_names[0], 0)
    assert copy.deepcopy(sample) == sample
    # 영상 단위 다중 프로세스 추출에서 프로세스 사이를 오간다.
    assert pickle.loads(pickle.dumps(sample)) == sample


def test_value_type_set_is_exactly_the_plan_list() -> None:
    declared = {name for name, value in vars(T).items() if dataclasses.is_dataclass(value)}
    assert declared == {
        "RawFrame",
        "FrameSignals",
        "Baseline",
        "FrameState",
        "ClosureRun",
        "StepFeatures",
        "DriverStateEvent",
        "PipelineOutput",
    }


def test_optional_iris_ratios_default_to_none() -> None:
    signals = _TYPE_SAMPLES[1][0]
    assert (signals.iris_h_l, signals.iris_h_r, signals.iris_v_l, signals.iris_v_r) == (None, None, None, None)
    with_iris = dataclasses.replace(signals, face=True, iris_h_l=0.5, iris_v_l=-0.02)
    assert with_iris.iris_h_l == 0.5 and with_iris.iris_h_r is None


def test_string_constants() -> None:
    assert T.EVENT_TYPES == (
        "calibrating",
        "unable_to_analyze",
        "gaze_away_long",
        "vats",
        "drowsy",
        "microsleep",
        "sleep",
    )
    assert T.PHASES == ("onset", "end")
    assert T.SOURCES == ("face", "head_pose", "eye", "model", "perclos", "closure")
    assert T.DROWSY_SOURCES == ("model", "perclos", "either")
    assert T.BASELINE_STATUSES == ("ok", "off_axis", "fallback_global")
    assert T.EYE_LOOK_ORDER == ("in_l", "out_l", "up_l", "down_l", "in_r", "out_r", "up_r", "down_r")


# ---------------------------------------------------------------------------
# 실험 설정 (config/default.yaml)
# ---------------------------------------------------------------------------


def test_default_config_paths(experiment_config: ExperimentConfig, repo_root: Path) -> None:
    paths = experiment_config.paths
    assert paths.rldd_root is None and paths.unlabeled_root is None
    assert paths.raw_dir == Path("data/raw")
    assert paths.model_path == Path("models/face_landmarker.task")
    assert paths.feature_spec == Path("config/feature_spec.json")
    assert paths.rules == Path("config/rules.yaml")
    assert paths.export_dir == Path("export")
    assert paths.reports_dir == Path("reports")
    assert (repo_root / paths.feature_spec).is_file() and (repo_root / paths.rules).is_file()


def test_default_config_extraction_and_windows(
    experiment_config: ExperimentConfig, feature_spec: S.FeatureSpec
) -> None:
    extraction = experiment_config.extraction
    assert extraction.workers >= 1
    assert extraction.rotation_overrides == {}
    assert extraction.drop_sim == DropSimConfig(subdir="fps15", latency_ms_mean=66.7, latency_ms_std=8.0, seed=0)
    # 지연이 프레임 간격보다 길면 흐름 제한기의 출력률은 1000 / 지연: 약 15 fps
    assert 1000.0 / extraction.drop_sim.latency_ms_mean == pytest.approx(15.0, abs=0.01)
    windows = experiment_config.windows
    assert (windows.window_steps, windows.train_stride, windows.eval_stride) == (120, 5, 1)
    assert windows.window_steps == feature_spec.window_steps


def test_default_config_training(experiment_config: ExperimentConfig) -> None:
    training = experiment_config.training
    assert training.arms == ("alert_baseline", "global", "session_baseline")
    assert training.export_arm == "alert_baseline"
    assert (training.optimizer, training.lr, training.weight_decay) == ("adamw", 1e-3, 1e-2)
    assert (training.batch_size, training.updates_per_unit) == (64, 200)
    assert (training.max_units, training.patience_units) == (60, 10)
    assert training.label_smoothing == 0.1
    assert training.noise_sigma == 0.05
    assert (training.frame_drop_min, training.frame_drop_max) == (0.10, 0.30)
    assert training.seeds == (0, 1, 2) and training.final_seed == 0
    assert training.tcn.channels == (16, 32) and training.tcn.dropout == (0.1, 0.3)
    assert training.tcn.kernel_size == 3 and training.tcn.dilations == (1, 2, 4, 8, 16)
    assert training.gru.hidden == (32, 64) and training.gru.dropout == (0.1, 0.3)
    # 수용 영역 = 1 + 2·(k−1)·Σ확장 = 1 + 2·2·31 = 125스텝 ≥ 창 120스텝 (PLAN 5.1절)
    receptive_field = 1 + 2 * (training.tcn.kernel_size - 1) * sum(training.tcn.dilations)
    assert receptive_field == 125


def test_default_config_lgbm_evaluation_export(experiment_config: ExperimentConfig) -> None:
    lgbm = experiment_config.lgbm
    assert lgbm.num_leaves == (7, 15) and lgbm.min_data_in_leaf == (20, 60)
    assert lgbm.reference_arm == "alert_baseline" and lgbm.reference_w == 0.5
    assert lgbm.w_grid == (0.0, 0.5, 1.0)

    evaluation = experiment_config.evaluation
    assert (evaluation.n_folds, evaluation.subjects_per_fold) == (5, 12)
    assert evaluation.bootstrap_n == 10000
    assert evaluation.lgbm_min_macro_f1 == 0.40
    assert evaluation.quality_gate == QualityGate(min_face_ratio=0.7, min_windows=1)
    assert evaluation.model_gate == ModelGate(
        max_macro_f1_deficit=0.02,
        min_two_class_video_accuracy=0.70,
        max_false_alarms_per_hour=2.0,
        max_alert_video_alarm_ratio=0.10,
        min_drowsy_video_alarm_ratio=0.50,
    )
    assert evaluation.signal_gate.min_baseline_ok_ratio == 0.9
    assert (evaluation.signal_gate.blink_rate_min_per_min, evaluation.signal_gate.blink_rate_max_per_min) == (5.0, 40.0)
    assert evaluation.signal_gate.min_peak_closure_median == 0.8

    export = experiment_config.export
    assert (export.opset, export.ir_version) == (17, 8)
    assert (export.input_name, export.output_name) == ("features", "probs")
    assert export.model_version == 0
    assert export.max_file_bytes == 1_000_000
    assert export.parity_tolerance == 1e-4
    assert (export.latency_runs, export.intra_op_threads) == (1000, 1)
    assert export.int8_if_tcn is True

    assert experiment_config.demo.model_path == Path("export/drowsiness_v1.onnx")


def test_config_rejects_unknown_and_missing_keys(config_path: Path, tmp_path: Path) -> None:
    text = config_path.read_text(encoding="utf-8")
    extra = tmp_path / "extra.yaml"
    extra.write_text(text + "\nsurprise: 1\n", encoding="utf-8")
    with pytest.raises(S.SpecError, match=r"unknown keys \['surprise'\]"):
        load_config(extra)

    missing = tmp_path / "missing.yaml"
    missing.write_text(text.replace("  eval_stride: 1", "  # eval_stride removed"), encoding="utf-8")
    with pytest.raises(S.SpecError, match=r"config\.windows: missing keys \['eval_stride'\]"):
        load_config(missing)

    bad_arm = tmp_path / "bad_arm.yaml"
    bad_arm.write_text(text.replace("export_arm: alert_baseline", "export_arm: per_video"), encoding="utf-8")
    with pytest.raises(S.SpecError, match="export_arm"):
        load_config(bad_arm)


def test_model_file_matches_spec_hash(
    experiment_config: ExperimentConfig, pipeline_spec: S.PipelineSpec, repo_root: Path
) -> None:
    model_path = repo_root / experiment_config.paths.model_path
    if not model_path.is_file():
        pytest.skip("face_landmarker.task is not present (it is never committed)")
    assert S.file_sha256(model_path) == pipeline_spec.mediapipe.model_sha256 == MEDIAPIPE_MODEL_SHA256
