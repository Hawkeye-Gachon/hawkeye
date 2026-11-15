"""demo: 모델 메타데이터 대조, 파이프라인 실행, 타임라인 PNG와 이벤트 JSON, 영상의 회전, 명령 (PLAN 9장 데모 항목).

얼굴 이미지는 쓰지 않는다(데모는 얼굴 프레임을 저장하지 않는다: PLAN 1.2절의 결정 D13). 입력은 `drowsy.synth`의 각본
시나리오를 추출 결과와 같은 형식의 parquet으로 쓴 것이고, 영상 입력의 회전은 가짜 랜드마커와 표식 그림으로 시험한다.
`mediapipe` 표시가 붙은 테스트만 실제 랜드마커를 얼굴 없는 합성 영상에 돌린다. 실제 얼굴로 도는 테스트는 하나이고
환경 변수 `DROWSY_TEST_FACE_IMAGE`가 쓸 수 있는 얼굴 사진을 가리킬 때만 돈다(없으면 건너뛴다).
"""

from __future__ import annotations

import ast
import dataclasses
import json
import math
import os
import shutil
import struct
import sys
import tomllib
import warnings
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import av
import matplotlib.image
import numpy as np
import onnx
import pytest
import yaml

from drowsy import demo, extract, synth
from drowsy.config import ExperimentConfig
from drowsy.dataset import SYNTHETIC_NOTE
from drowsy.demo import ALERT_DEFERRED, ALERT_IMMEDIATE, ALERT_REPEAT, DemoError, EventRecord, run_demo_raw
from drowsy.export_onnx import METADATA_KEYS, TRAINED_ON_NONE, exported_spec_path
from drowsy.rules import ZONE_IN, ZONE_OUT, ZONE_UNKNOWN
from drowsy.spec import (
    FeatureSpec,
    PipelineSpec,
    RulesConfig,
    apply_overrides,
    compute_pipeline_hash,
    file_sha256,
    read_feature_spec_dict,
    rules_hash,
    write_feature_spec,
)
from drowsy.types import (
    BASELINE_OK,
    EVENT_CALIBRATING,
    EVENT_DROWSY,
    EVENT_GAZE_AWAY_LONG,
    EVENT_MICROSLEEP,
    EVENT_SLEEP,
    EVENT_TYPES,
    PHASE_END,
    PHASE_ONSET,
    SOURCE_CLOSURE,
    SOURCE_HEAD_POSE,
    SOURCE_PERCLOS,
    DriverStateEvent,
)

SESSION_MS = 300_000
FIRST_OUTPUT_MS = 239_000  # 기준선 60초 + 조회 구간 60초 + 120스텝(PLAN 4.4절)

# 각본: 세션 시간(ms)에 시작하는 감김과 고개 돌림
CALIBRATION_CLOSURE_MS = 20_000  # 기준선 수집 중의 1.5초 감김 → 임시 microsleep
MICROSLEEP_MS = 100_000  # 1.5초 감김 → microsleep
SLEEP_MS = 150_000  # 6.5초 감김 → microsleep, sleep, sleep 되풀이 경고
SHORT_CLOSURES_MS = tuple(range(165_000, 195_000, 5_000))  # 0.8초 감김 6번 → PERCLOS가 0.15를 넘는다
GLANCE_MS = 250_000  # 4.5초 고개 돌림(35도) → gaze_away_long


# ---------------------------------------------------------------------------
# 도우미와 픽스처
# ---------------------------------------------------------------------------


def scripted_frames(spec: PipelineSpec, duration_ms: int) -> list[Any]:
    """각본 세션의 프레임. 길이를 줄이면 그 시각까지의 구간만 들어간다."""
    scripted = [
        synth.Closure(1500, start_ms=CALIBRATION_CLOSURE_MS),
        synth.Closure(1500, start_ms=MICROSLEEP_MS),
        synth.Closure(6500, start_ms=SLEEP_MS),
        *(synth.Closure(800, start_ms=start) for start in SHORT_CLOSURES_MS),
        synth.HeadTurn(4500, yaw=35.0, start_ms=GLANCE_MS),
    ]
    busy = [(segment.start_ms - 1500, segment.start_ms + segment.duration_ms + 1500) for segment in scripted]
    blinks = [
        synth.Blink(start_ms=start)
        for start in range(3_000, duration_ms, 4_000)
        if not any(low <= start < high for low, high in busy)
    ]
    scenario = synth.Scenario(segments=(*scripted, *blinks), duration_ms=duration_ms, fps=30.0)
    return synth.render(scenario, spec)


def write_session(path: Path, spec: PipelineSpec, duration_ms: int) -> Path:
    """각본 세션을 `drowsy.synth`의 세션과 같은 표시(`synthetic: true`)를 단 RawFrame parquet으로 쓴다."""
    metadata = {"source": "synthetic (drowsy.synth)", "synthetic": True}
    extract.write_raw_frames(scripted_frames(spec, duration_ms), path, spec, metadata)
    return path


def png_size(path: Path) -> tuple[int, int]:
    """PNG 머리말에서 (너비, 높이)를 읽는다."""
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR"
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def expected_png_size(cfg: ExperimentConfig) -> tuple[int, int]:
    dpi = cfg.demo.figure_dpi
    return int(demo.FIGURE_SIZE_IN[0] * dpi), int(demo.FIGURE_SIZE_IN[1] * dpi)


def assert_real_picture(path: Path, cfg: ExperimentConfig) -> None:
    """크기가 맞고 내용이 있는 PNG인지(한 가지 색으로 찬 빈 그림이 아닌지)."""
    assert png_size(path) == expected_png_size(cfg)
    image = matplotlib.image.imread(path)
    assert image.shape[:2] == expected_png_size(cfg)[::-1]
    assert len(np.unique(image.reshape(-1, image.shape[2]), axis=0)) > 20


def kinds(events: list[dict[str, Any]]) -> list[str]:
    return [f"{event['type']}:{event['phase']}" for event in events]


def find(events: list[dict[str, Any]], kind: str, index: int = 0) -> dict[str, Any]:
    """`종류:단계`가 kind인 이벤트 중 index번째."""
    return [event for event in events if f"{event['type']}:{event['phase']}" == kind][index]


def glyph_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    messages = [str(item.message) for item in caught]
    return [message for message in messages if "glyph" in message.lower() or "font" in message.lower()]


@pytest.fixture(scope="module")
def demo_cfg(experiment_config: ExperimentConfig, repo_root: Path) -> ExperimentConfig:
    """기본 설정에서 경로만 절대 경로로 바꾼 것(실행 위치와 무관하게 돌도록)."""
    paths = dataclasses.replace(
        experiment_config.paths,
        feature_spec=repo_root / "config" / "feature_spec.json",
        rules=repo_root / "config" / "rules.yaml",
        model_path=repo_root / "models" / "face_landmarker.task",
        export_dir=repo_root / "export",
    )
    return dataclasses.replace(experiment_config, paths=paths)


@pytest.fixture(scope="module")
def model_v0(repo_root: Path) -> Path:
    """저장소의 무작위 가중치 모델. 옆에 `feature_spec_v0.json`이 있다."""
    return repo_root / "export" / "drowsiness_v0.onnx"


@pytest.fixture(scope="module")
def session_parquet(tmp_path_factory: pytest.TempPathFactory, pipeline_spec: PipelineSpec) -> Path:
    """5분 각본 세션."""
    return write_session(tmp_path_factory.mktemp("demo_session") / "scripted.parquet", pipeline_spec, SESSION_MS)


@pytest.fixture(scope="module")
def session_outputs(
    tmp_path_factory: pytest.TempPathFactory, session_parquet: Path, model_v0: Path, demo_cfg: ExperimentConfig
) -> tuple[Path, Path, list[str]]:
    """5분 세션에 데모를 한 번 돌린 결과: (PNG, JSON, 글꼴 관련 경고)."""
    out_dir = tmp_path_factory.mktemp("demo_out")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        png_path, json_path = run_demo_raw(session_parquet, model_v0, out_dir, cfg=demo_cfg)
    return png_path, json_path, glyph_warnings(caught)


def write_cli_config(tmp_path: Path, config_path: Path, repo_root: Path, **demo_block: str) -> Path:
    """기본 설정에서 경로를 절대 경로로, demo 블록을 주어진 값으로 바꾼 설정 파일을 쓴다."""
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["paths"]["feature_spec"] = str(repo_root / "config" / "feature_spec.json")
    data["paths"]["rules"] = str(repo_root / "config" / "rules.yaml")
    data["paths"]["model_path"] = str(repo_root / "models" / "face_landmarker.task")
    data["paths"]["export_dir"] = str(repo_root / "export")
    data["demo"].update(demo_block)
    out = tmp_path / "config.yaml"
    out.write_text(yaml.safe_dump(data), encoding="utf-8")
    return out


def copy_model(model: Path, out_dir: Path, **metadata: str) -> Path:
    """모델과 feature_spec 사본을 out_dir에 복사한다. metadata에 준 키는 모델 메타데이터에서 그 값으로 바꾼다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    proto = onnx.load(str(model))
    for prop in proto.metadata_props:
        if prop.key in metadata:
            prop.value = metadata[prop.key]
    copied = out_dir / model.name
    onnx.save(proto, str(copied))
    shutil.copy(exported_spec_path(model), out_dir)
    return copied


# ---------------------------------------------------------------------------
# 5분 각본 세션: 두 파일, 이벤트, 메타데이터
# ---------------------------------------------------------------------------


def test_demo_writes_png_and_json(
    session_outputs: tuple[Path, Path, list[str]], session_parquet: Path, demo_cfg: ExperimentConfig
) -> None:
    png_path, json_path, font_warnings = session_outputs
    assert png_path.name == "scripted_timeline.png" and json_path.name == "scripted_events.json"
    assert png_path.parent == json_path.parent
    # 쓰는 파일은 둘뿐이다(얼굴 프레임이나 중간 파일을 남기지 않는다. PLAN 1.2절의 결정 D13)
    assert sorted(path.name for path in png_path.parent.iterdir()) == [json_path.name, png_path.name]
    assert sorted(path.name for path in session_parquet.parent.iterdir()) == [session_parquet.name]

    assert_real_picture(png_path, demo_cfg)
    assert font_warnings == []  # 없는 글리프 경고가 없다

    text = json_path.read_text(encoding="utf-8")
    report = json.loads(text)
    assert text.endswith("\n")
    assert list(report) == [
        "note",
        "source",
        "model",
        "spec",
        "drowsiness_source",
        "speed_kmh",
        "session",
        "baseline",
        "counters",
        "probabilities",
        "events",
        "alerts",
    ]
    # 합성 입력으로 만든 출력은 첫 항목에 합성 데이터임을 적는다
    assert report["note"] == SYNTHETIC_NOTE
    assert report["source"]["kind"] == demo.SOURCE_RAW and report["source"]["synthetic"] is True
    assert report["source"]["path"] == str(session_parquet)


def test_events_json_holds_the_scripted_events(session_outputs: tuple[Path, Path, list[str]]) -> None:
    report = json.loads(session_outputs[1].read_text(encoding="utf-8"))
    events = report["events"]
    first_t_ms = report["session"]["first_t_ms"]
    assert first_t_ms == 1

    assert kinds(events) == [
        "calibrating:onset",
        "microsleep:onset",  # 기준선 수집 중의 감김(임시 판정)
        "microsleep:end",
        "calibrating:end",
        "microsleep:onset",  # 1.5초 감김
        "microsleep:end",
        "microsleep:onset",  # 6.5초 감김
        "sleep:onset",
        "sleep:end",
        "drowsy:onset",  # PERCLOS 판정
        "drowsy:end",
        "gaze_away_long:onset",  # 고개 돌림
        "gaze_away_long:end",
    ]
    for event in events:  # DriverStateEvent의 여덟 필드 + 발행 시각 + 경고 여부
        assert set(event) == {field.name for field in dataclasses.fields(DriverStateEvent)} | {"t_ms", "alert"}
        assert event["t_ms"] >= event["t_start_ms"]

    def session_ms(event: dict[str, Any]) -> int:
        return event["t_start_ms"] - first_t_ms

    calibrating_end = find(events, "calibrating:end")
    assert (calibrating_end["t_start_ms"], calibrating_end["duration_ms"]) == (first_t_ms, 60_000)
    assert find(events, "calibrating:onset")["alert"] is False  # calibrating은 경고하지 않는다

    provisional = find(events, "microsleep:onset", 0)
    assert provisional["provisional"] is True and provisional["alert"] is True
    assert provisional["source"] == SOURCE_CLOSURE and provisional["severity"] == 3
    assert 0 <= session_ms(provisional) - CALIBRATION_CLOSURE_MS <= 150
    assert 1000 <= provisional["duration_ms"] <= 1100  # 1.0초에 도달한 프레임에서 시작 이벤트가 나간다

    microsleep = find(events, "microsleep:onset", 1)
    assert microsleep["provisional"] is False and microsleep["alert"] is True
    assert 0 <= session_ms(microsleep) - MICROSLEEP_MS <= 150
    assert find(events, "microsleep:end", 1)["t_start_ms"] == microsleep["t_start_ms"]

    long_closure = find(events, "microsleep:onset", 2)
    sleep_onset = find(events, "sleep:onset")
    sleep_end = find(events, "sleep:end")
    assert 0 <= session_ms(long_closure) - SLEEP_MS <= 150
    assert sleep_onset["t_start_ms"] == sleep_end["t_start_ms"] == long_closure["t_start_ms"]
    assert 3000 <= sleep_onset["duration_ms"] <= 3100 and 6500 <= sleep_end["duration_ms"] <= 6800
    assert sleep_end["alert"] is False  # 경고가 되는 것은 onset뿐이다

    drowsy_onset = find(events, "drowsy:onset")
    drowsy_end = find(events, "drowsy:end")
    assert drowsy_onset["source"] == drowsy_end["source"] == SOURCE_PERCLOS  # 기본 판정원은 PERCLOS다(PLAN 1.2절의 결정 D11)
    assert drowsy_onset["alert"] is True and drowsy_onset["provisional"] is False
    assert SLEEP_MS < session_ms(drowsy_onset) < SHORT_CLOSURES_MS[-1] + 10_000
    assert drowsy_end["t_start_ms"] == drowsy_onset["t_start_ms"]
    assert drowsy_end["t_start_ms"] + drowsy_end["duration_ms"] < first_t_ms + GLANCE_MS

    gaze = find(events, "gaze_away_long:onset")
    assert gaze["source"] == SOURCE_HEAD_POSE and gaze["alert"] is True
    assert 0 <= session_ms(gaze) - GLANCE_MS <= 400 and gaze["duration_ms"] == 3000

    # 경고 목록: 시작한 프레임에서 바로 나간 경고 여섯 개와, 감김이 이어지는 동안 3초 뒤에 되풀이된 sleep 경고 하나.
    # 속도를 주지 않았고 재경고 대기에 걸린 시작도 없으므로 미뤘다 낸 경고는 없다
    alerts = report["alerts"]
    assert [(alert["type"], alert["kind"]) for alert in alerts] == [
        (EVENT_MICROSLEEP, ALERT_IMMEDIATE),
        (EVENT_MICROSLEEP, ALERT_IMMEDIATE),
        (EVENT_MICROSLEEP, ALERT_IMMEDIATE),
        (EVENT_SLEEP, ALERT_IMMEDIATE),
        (EVENT_SLEEP, ALERT_REPEAT),
        (EVENT_DROWSY, ALERT_IMMEDIATE),
        (EVENT_GAZE_AWAY_LONG, ALERT_IMMEDIATE),
    ]
    for alert in alerts:
        assert set(alert) == {field.name for field in dataclasses.fields(DriverStateEvent)} | {"t_ms", "kind"}
        assert alert["phase"] == PHASE_ONSET
    repeat = alerts[4]
    assert repeat["t_start_ms"] == sleep_onset["t_start_ms"] and repeat["t_ms"] - alerts[3]["t_ms"] == 3000
    flagged = [event for event in events if event["alert"]]
    assert [(e["type"], e["t_ms"]) for e in flagged] == [
        (a["type"], a["t_ms"]) for a in alerts if a["kind"] == ALERT_IMMEDIATE
    ]
    counters = report["counters"]
    assert counters["events"] == len(events) and counters["alerts"] == len(alerts)
    assert (counters["alerts_immediate"], counters["alerts_deferred"], counters["alerts_repeat"]) == (6, 0, 1)


def test_events_json_holds_the_run_metadata(
    session_outputs: tuple[Path, Path, list[str]],
    model_v0: Path,
    demo_cfg: ExperimentConfig,
    feature_spec: FeatureSpec,
    rules_config: RulesConfig,
) -> None:
    report = json.loads(session_outputs[1].read_text(encoding="utf-8"))

    model = report["model"]
    assert model["file"] == str(model_v0) and model["sha256"] == file_sha256(model_v0)
    assert sorted(model["metadata"]) == sorted(METADATA_KEYS)
    assert model["metadata"]["trained_on"] == TRAINED_ON_NONE and model["metadata"]["model_version"] == "0"
    spec_copy = model_v0.with_name("feature_spec_v0.json")
    assert model["feature_spec_copy"] == str(spec_copy)
    assert model["feature_spec_hash"] == model["metadata"]["feature_spec_hash"] == file_sha256(spec_copy)

    spec = report["spec"]
    assert spec["feature_spec"] == str(demo_cfg.paths.feature_spec)
    assert spec["feature_spec_sha256"] == file_sha256(demo_cfg.paths.feature_spec)
    assert spec["pipeline_hash"] == feature_spec.pipeline_hash == model["metadata"]["pipeline_hash"]
    assert spec["rules"] == str(demo_cfg.paths.rules) and spec["rules_hash"] == rules_hash(rules_config)

    assert report["drowsiness_source"] == rules_config.drowsiness.source == "perclos"
    assert report["speed_kmh"] is None
    assert report["session"]["duration_ms"] == report["session"]["last_t_ms"] - report["session"]["first_t_ms"]
    assert SESSION_MS - 100 < report["session"]["duration_ms"] < SESSION_MS

    baseline = report["baseline"]
    assert baseline["status"] == BASELINE_OK and baseline["t_complete_ms"] == 60_000
    constants = feature_spec.pipeline.global_constants  # 각본의 얼굴은 뜬 눈 값이 전역 상수와 같다
    assert baseline["e_l"] == pytest.approx(constants.e, abs=1e-6)
    assert baseline["b_r"] == pytest.approx(constants.b, abs=1e-6)
    assert len(baseline["r0"]) == 9 and baseline["yaw0_deg"] == pytest.approx(0.0, abs=1e-6)
    assert baseline["diagnostics"]["n_face_frames"] > 1700 and baseline["diagnostics"]["face_ratio"] == 1.0

    counters = report["counters"]
    assert counters["frames_accepted"] == counters["face_frames"] == 9000 and counters["dropped_frames"] == 0
    assert counters["steps_emitted"] == 239  # 스텝 61..299
    assert counters["model_calls"] == 61 and counters["nonfinite_windows"] == 0  # 239초부터 매초

    probabilities = report["probabilities"]
    assert probabilities["n_outputs"] == 61 and probabilities["empty_reason"] is None
    assert probabilities["first_t_end_ms"] == report["session"]["first_t_ms"] + FIRST_OUTPUT_MS
    assert probabilities["last_t_end_ms"] == probabilities["first_t_end_ms"] + 60_000


def test_speed_gates_change_the_alert_flags_only(
    tmp_path: Path, session_parquet: Path, session_outputs: tuple[Path, Path, list[str]], model_v0: Path,
    demo_cfg: ExperimentConfig,
) -> None:  # fmt: skip
    reference = json.loads(session_outputs[1].read_text(encoding="utf-8"))
    _, json_path = run_demo_raw(session_parquet, model_v0, tmp_path, 30.0, cfg=demo_cfg)
    report = json.loads(json_path.read_text(encoding="utf-8"))

    assert report["speed_kmh"] == 30.0
    # 상태 이벤트는 속도와 무관하다. 30 km/h에서는 졸음 경고(50 km/h부터)만 억제된다(PLAN 1.2절의 결정 D3: 속도 억제)
    strip = [{key: value for key, value in event.items() if key != "alert"} for event in report["events"]]
    assert strip == [{key: value for key, value in event.items() if key != "alert"} for event in reference["events"]]
    assert find(report["events"], "drowsy:onset")["alert"] is False
    assert find(report["events"], "sleep:onset")["alert"] is True
    assert [alert["type"] for alert in report["alerts"]] == [
        alert["type"] for alert in reference["alerts"] if alert["type"] != EVENT_DROWSY
    ]

    _, json_path = run_demo_raw(session_parquet, model_v0, tmp_path / "slow", 10.0, cfg=demo_cfg)
    assert json.loads(json_path.read_text(encoding="utf-8"))["alerts"] == []  # 20 km/h 미만에서는 모두 억제된다

    with pytest.raises(DemoError, match="speed"):
        run_demo_raw(session_parquet, model_v0, tmp_path / "bad", -1.0, cfg=demo_cfg)


# ---------------------------------------------------------------------------
# 그림의 내용
# ---------------------------------------------------------------------------


def render(parquet: Path, model: Path, cfg: ExperimentConfig, feature_spec: FeatureSpec, rules: RulesConfig) -> Any:
    """데모와 같은 순서로 파이프라인을 돌리고 그림을 만든다. (그림, 실행 결과, 확률이 없는 이유)를 돌려준다."""
    spec = feature_spec.pipeline
    info = demo.verify_model(model, feature_spec)
    model_fn = demo.make_model_fn(info.path, spec.step_features.window_steps, len(spec.step_features.names))
    run = demo.run_pipeline(extract.read_raw_frames(parquet), spec, rules, model_fn, None)
    reason = demo.probability_gap_reason(run, spec)
    figure = demo.render_timeline(run, spec, rules, ["title"], ["note"], "Model output", reason)
    return figure, run, reason


def test_timeline_has_four_aligned_tracks(
    session_parquet: Path,
    model_v0: Path,
    demo_cfg: ExperimentConfig,
    feature_spec: FeatureSpec,
    rules_config: RulesConfig,
) -> None:
    figure, run, reason = render(session_parquet, model_v0, demo_cfg, feature_spec, rules_config)
    assert reason is None
    ax_closure, ax_gaze, ax_probs, ax_events = figure.axes
    assert tuple(figure.get_size_inches()) == demo.FIGURE_SIZE_IN

    # 네 줄의 가로축이 같다(세션 시간, 초)
    limits = [ax.get_xlim() for ax in figure.axes]
    assert all(limit == limits[0] for limit in limits)
    assert limits[0] == (0.0, pytest.approx(run.duration_ms / 1000.0))

    # 1. closure: 임계값 0.8(PERCLOS), 0.7·0.5(규칙), 0.5·0.3(특징)이 선과 이름으로 그려져 있다
    spec = feature_spec.pipeline
    detector = rules_config.closure.detector
    wanted = {spec.step_features.perclos_thr, detector.enter_thr, detector.exit_thr, spec.run_detector.exit_thr}
    horizontal = {float(line.get_ydata()[0]) for line in ax_closure.lines if len(line.get_xdata()) == 2}
    assert wanted <= horizontal and spec.run_detector.enter_thr in horizontal
    labels = [text.get_text() for text in ax_closure.texts]
    assert labels == ["0.80 PERCLOS", "0.70 rule enter", "0.50 rule exit / blink enter", "0.30 blink exit"]
    drawn = [line for line in ax_closure.lines if len(line.get_xdata()) > 2]
    assert len(drawn) == 2  # 임시 closure와 확정 closure
    provisional_y, final_y = (np.ma.asarray(line.get_ydata()) for line in drawn)
    assert provisional_y.count() + final_y.count() == sum(run.closure_valid)
    assert provisional_y.count() == sum(run.provisional) and final_y.max() == pytest.approx(1.0)

    # 2. 전방 주시: 영역 밖 구간 하나(고개 돌림), 알 수 없음 구간 하나(기준선 수집 중)
    away, unknown = (collection.get_paths() for collection in ax_gaze.collections)
    assert len(away) == 1 and len(unknown) == 1
    away_x = away[0].vertices[:, 0]
    # 표시되는 것은 확정된 상태여서 디바운스(250 ms)만큼 늦게 시작하고 늦게 끝난다
    debounce_s = rules_config.forward_zone.debounce_ms / 1000.0
    assert GLANCE_MS / 1000.0 + debounce_s < away_x.min() < GLANCE_MS / 1000.0 + 1.0
    assert (GLANCE_MS + 3500) / 1000.0 < away_x.max() < (GLANCE_MS + 4500) / 1000.0 + debounce_s
    assert unknown[0].vertices[:, 0].max() == pytest.approx(60.0, abs=0.1)
    assert [label.get_text() for label in ax_gaze.get_yticklabels()] == ["away", "unknown / no data"]

    # 3. 확률: p_low_vigilant와 p_drowsy 선, 판정 임계값 두 개
    curves = [line for line in ax_probs.lines if len(line.get_xdata()) > 2]
    assert len(curves) == 2 and all(len(curve.get_xdata()) == len(run.probs) == 61 for curve in curves)
    assert curves[0].get_xdata()[0] == pytest.approx(FIRST_OUTPUT_MS / 1000.0)
    assert [text.get_text() for text in ax_probs.texts] == ["0.60 drowsy onset", "0.40 drowsy end"]
    assert [text.get_text() for text in ax_probs.get_legend().get_texts()] == ["p_drowsy", "p_low_vigilant"]
    assert curves[1].get_color() == ax_probs.get_legend().get_lines()[0].get_color()  # p_drowsy가 진한 쪽

    # 4. 이벤트: 종류마다 한 줄, 구간 일곱 개, 경고 표식 일곱 개(바로 낸 경고 여섯은 세모, 되풀이 하나는 세로 금.
    #    미뤘다 낸 경고는 없으므로 그 표식은 범례에만 있다)
    assert [label.get_text() for label in ax_events.get_yticklabels()] == list(EVENT_TYPES)
    assert len(ax_events.collections) == len(demo.event_spans(run.events, run.t_ms[-1])) == 7
    assert [(line.get_marker(), len(line.get_xdata())) for line in ax_events.lines] == [("v", 6), ("|", 1)]
    assert len(run.alerts) == 7
    legend = [text.get_text() for text in ax_events.get_legend().get_texts()]
    assert legend[:5] == ["severity 0", "severity 1", "severity 2", "severity 3", "provisional"]
    assert legend[5:] == ["alert", "deferred alert", "repeat alert"]
    # 심각도가 높을수록 어두운 색이고, 임시 판정만 빗금이다
    hatched = [collection.get_hatch() is not None for collection in ax_events.collections]
    assert hatched == [span.provisional for span in demo.event_spans(run.events, run.t_ms[-1])]


def test_short_input_has_an_empty_probability_track(
    tmp_path: Path,
    pipeline_spec: PipelineSpec,
    model_v0: Path,
    demo_cfg: ExperimentConfig,
    feature_spec: FeatureSpec,
    rules_config: RulesConfig,
) -> None:
    # 기준선은 완성되지만 첫 출력(239초)에 못 미치는 입력
    short = write_session(tmp_path / "short.parquet", pipeline_spec, 120_000)
    png_path, json_path = run_demo_raw(short, model_v0, tmp_path / "out", cfg=demo_cfg)
    assert_real_picture(png_path, demo_cfg)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    probabilities = report["probabilities"]
    assert probabilities["n_outputs"] == 0 and report["counters"]["model_calls"] == 0
    assert probabilities["first_t_end_ms"] is None and probabilities["last_t_end_ms"] is None
    assert "120 s" in probabilities["empty_reason"] and "needs 239 s" in probabilities["empty_reason"]
    assert report["baseline"]["status"] == BASELINE_OK and report["counters"]["steps_emitted"] == 59
    assert "microsleep:onset" in kinds(report["events"])  # 규칙은 모델을 기다리지 않는다

    figure, run, reason = render(short, model_v0, demo_cfg, feature_spec, rules_config)
    ax_probs = figure.axes[2]
    assert reason == probabilities["empty_reason"]
    assert all(len(line.get_xdata()) == 2 for line in ax_probs.lines)  # 임계값 선뿐이고 확률 선은 없다
    written = " ".join(text.get_text().replace("\n", " ") for text in ax_probs.texts)
    assert "No model output" in written and "needs 239 s" in written

    # 기준선이 완성되기 전에 끝나는 입력
    tiny = write_session(tmp_path / "tiny.parquet", pipeline_spec, 30_000)
    _, json_path = run_demo_raw(tiny, model_v0, tmp_path / "out", cfg=demo_cfg)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["baseline"] is None and report["counters"]["steps_emitted"] == 0
    assert "baseline was not completed" in report["probabilities"]["empty_reason"]
    assert kinds(report["events"]) == ["calibrating:onset", "microsleep:onset", "microsleep:end"]
    assert all(event["provisional"] for event in report["events"])


def test_first_output_time_and_gap_reasons(
    pipeline_spec: PipelineSpec, rules_config: RulesConfig, session_parquet: Path
) -> None:
    run = demo.run_pipeline(extract.read_raw_frames(session_parquet), pipeline_spec, rules_config, None, None)
    assert run.baseline is not None
    assert demo.first_output_session_ms(run.baseline, pipeline_spec) == FIRST_OUTPUT_MS
    # 모델을 주지 않으면 출력이 없다. 입력은 충분히 길므로 이유는 창이 이어지지 않았다는 쪽이다
    assert run.probs == () and "consecutive full-lookback steps" in str(demo.probability_gap_reason(run, pipeline_spec))
    with pytest.raises(DemoError, match="no frame"):
        demo.run_pipeline([], pipeline_spec, rules_config, None, None)


def test_event_spans_pair_onsets_with_ends() -> None:
    def event(kind: str, phase: str, t_start_ms: int, duration_ms: int, provisional: bool = False) -> EventRecord:
        made = DriverStateEvent(kind, 3, t_start_ms, duration_ms, 1.0, SOURCE_CLOSURE, phase, provisional)
        return EventRecord(t_start_ms + duration_ms, made, False)

    records = [
        event(EVENT_CALIBRATING, PHASE_ONSET, 1, 0, True),
        event(EVENT_MICROSLEEP, PHASE_ONSET, 1000, 1030),
        event(EVENT_MICROSLEEP, PHASE_END, 1000, 1500),
        event(EVENT_MICROSLEEP, PHASE_ONSET, 5000, 1030),
        event(EVENT_SLEEP, PHASE_ONSET, 5000, 3030),
        event(EVENT_SLEEP, PHASE_END, 5000, 4000),  # 같은 감김의 microsleep 구간도 닫는다
        event(EVENT_DROWSY, PHASE_ONSET, 20_000, 9000),  # 입력이 끝날 때까지 end가 없다
    ]
    spans = demo.event_spans(records, 40_000)
    assert [(s.type, s.start_ms, s.end_ms, s.provisional, s.ended) for s in spans] == [
        (EVENT_MICROSLEEP, 1000, 2500, False, True),
        (EVENT_SLEEP, 5000, 9000, False, True),
        (EVENT_MICROSLEEP, 5000, 9000, False, True),
        (EVENT_CALIBRATING, 1, 40_000, True, False),
        (EVENT_DROWSY, 20_000, 40_000, False, False),
    ]


def test_zone_spans_hold_each_frame_up_to_max_hold() -> None:
    t_ms = [0, 100, 200, 1000, 1100, 1200]
    zones = [ZONE_IN, ZONE_OUT, ZONE_OUT, ZONE_OUT, ZONE_UNKNOWN, ZONE_IN]
    spans = demo.zone_spans(t_ms, zones, 200)
    assert spans[ZONE_IN] == [(0, 100)]  # 마지막 프레임은 시간을 대표하지 않는다
    # 200에서 본 표지는 400까지만 유지되고, 그 뒤 1000까지는 자료가 없는 시간이다
    assert spans[ZONE_OUT] == [(100, 400), (1000, 1100)]
    assert spans[ZONE_UNKNOWN] == [(400, 1000), (1100, 1200)]


def test_non_ascii_file_name_renders_without_missing_glyphs(
    tmp_path: Path, pipeline_spec: PipelineSpec, model_v0: Path, demo_cfg: ExperimentConfig
) -> None:
    parquet = write_session(tmp_path / "주행_세션.parquet", pipeline_spec, 20_000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        png_path, json_path = run_demo_raw(parquet, model_v0, tmp_path / "out", cfg=demo_cfg)
    assert glyph_warnings(caught) == []
    assert png_path.name == "주행_세션_timeline.png"
    assert_real_picture(png_path, demo_cfg)
    assert json.loads(json_path.read_text(encoding="utf-8"))["source"]["path"] == str(parquet)  # JSON에는 원래 경로


# ---------------------------------------------------------------------------
# 경고의 종류(바로 낸 것, 미뤘다 낸 것, 되풀이)와 스텝마다의 모델 출력. 2026-11-13에 소유자가 정한 두 가지
# ("막힌 경고는 막힘이 풀렸을 때 상태가 계속 중이면 한 번 낸다", "밀린 스텝마다 모델을 호출한다")가 데모에 드러나는지 본다.
# ---------------------------------------------------------------------------

KINDS_SESSION_MS = 310_000


def kinds_frames(spec: PipelineSpec) -> list[Any]:
    """경고의 세 종류가 모두 나오는 각본 세션(310초, 10 fps, 첫 프레임의 t_ms는 1. 프레임은 세션 시간 100 ms마다).

    세션 시간(초)                          손 계산(10 fps라 판정은 100 ms 격자에서 난다)
      72.0  고개 35도 9초                  yaw가 23도를 넘는 첫 프레임 72.2 -> 3초 뒤 75.2에 gaze_away_long 시작,
                                           영역 안으로 돌아온 것이 확정되는 81.2에 종료
     105.0  고개 35도 6초                  108.2에 시작, 111.2에 종료
     140.0  감김(머무는 시간 1.5초)        감긴 첫 프레임 140.1, 감김의 시작은 보간한 140.07 -> 1초가 차는 141.1에
                                           microsleep 시작(길이 1030 ms), 141.7에 종료
     143.5  감김(2.7초)                    144.6에 microsleep 시작, 146.4에 종료(3초가 안 돼 sleep은 아니다)
     160.0  감김(10.5초)                   161.1에 microsleep, 163.1에 sleep 시작, 170.7에 sleep 종료
     181.0  감김(12초)                     182.1에 microsleep, 184.1에 sleep 시작, 193.2에 sleep 종료
    """
    segments = [
        synth.HeadTurn(9000, yaw=35.0, start_ms=72_000),
        synth.HeadTurn(6000, yaw=35.0, start_ms=105_000),
        synth.Closure(1500, start_ms=140_000),
        synth.Closure(2700, start_ms=143_500),
        synth.Closure(10_500, start_ms=160_000),
        synth.Closure(12_000, start_ms=181_000),
    ]
    return synth.render(synth.Scenario(segments=tuple(segments), duration_ms=KINDS_SESSION_MS, fps=10.0), spec)


def changing_speed(t_ms: int) -> float:
    """세션 시간에 따라 바뀌는 차량 속도(km/h). 경고 기준은 시선·감김 20, 졸음 50이다(PLAN 1.2절의 결정 D3)."""
    t = t_ms - 1
    if t < 80_000:
        return 10.0
    if t < 100_000:
        return 60.0
    if t < 130_000:
        return 10.0
    if t < 180_000:
        return 60.0
    if t < 186_000:
        return 10.0
    if t < 230_000:
        return 60.0
    if t < 300_000:
        return 45.0
    return 100.0


def test_alert_kinds_with_a_speed_gate_that_opens_during_a_state(
    pipeline_spec: PipelineSpec, rules_config: RulesConfig
) -> None:
    """속도가 바뀌는 각본 세션에서 경고마다 종류가 맞게 붙는다. 기댓값은 모두 손으로 센 것이다(`kinds_frames`).

    졸음 판정원은 model이고 모델은 늘 p_drowsy 0.95를 낸다: 첫 출력은 스텝 239, 열 개째인 스텝 248에서 drowsy 시작
    (t_start는 스텝 239의 끝 239 001). PERCLOS 판정은 쓰지 않으므로 긴 감김이 drowsy를 더 만들지 않는다.

      t_ms     경고             종류        까닭
      80 001   gaze_away_long   deferred    75.2초의 시작은 10 km/h라 막혔고, 60 km/h가 된 첫 프레임에서 나간다
      (없음)   gaze_away_long   -           108.2초의 시작은 10 km/h에서 막힌 채 111.2초에 끝났다
     141 101   microsleep       immediate
     146 101   microsleep       deferred    144.6초의 시작은 141.1초의 경고 뒤 5초 대기에 걸렸고, 대기가 끝난 프레임에서
     161 101   microsleep       immediate
     163 101   sleep            immediate
     166 101   sleep            repeat      3초마다. 다음(172.1초)은 감김이 끝난 뒤다
     169 101   sleep            repeat
     (없음)    microsleep       -           182.1초의 시작은 10 km/h에서 막혔고 184.1초의 sleep 시작이 지웠다
     186 001   sleep            deferred    184.1초의 시작은 10 km/h라 막혔고, 60 km/h가 된 첫 프레임에서 나간다
     189 001   sleep            repeat      그 경고부터 3초마다. 다음(195.0초)은 감김이 끝난 뒤다
     192 001   sleep            repeat
     300 001   drowsy           deferred    248초의 시작은 45 km/h라 막혔고, 100 km/h가 된 첫 프레임에서 나간다
    """
    rules = apply_overrides(rules_config, {"drowsiness.source": "model"})
    frames = kinds_frames(pipeline_spec)
    run = demo.run_pipeline(frames, pipeline_spec, rules, lambda window: (0.02, 0.03, 0.95), changing_speed)

    assert [(a.t_ms, a.event.type, a.kind, a.event.t_start_ms, a.event.duration_ms) for a in run.alerts] == [
        (80_001, EVENT_GAZE_AWAY_LONG, ALERT_DEFERRED, 72_201, 7_800),
        (141_101, EVENT_MICROSLEEP, ALERT_IMMEDIATE, 140_071, 1_030),
        (146_101, EVENT_MICROSLEEP, ALERT_DEFERRED, 143_571, 2_530),
        (161_101, EVENT_MICROSLEEP, ALERT_IMMEDIATE, 160_071, 1_030),
        (163_101, EVENT_SLEEP, ALERT_IMMEDIATE, 160_071, 3_030),
        (166_101, EVENT_SLEEP, ALERT_REPEAT, 160_071, 6_030),
        (169_101, EVENT_SLEEP, ALERT_REPEAT, 160_071, 9_030),
        (186_001, EVENT_SLEEP, ALERT_DEFERRED, 181_071, 4_930),
        (189_001, EVENT_SLEEP, ALERT_REPEAT, 181_071, 7_930),
        (192_001, EVENT_SLEEP, ALERT_REPEAT, 181_071, 10_930),
        (300_001, EVENT_DROWSY, ALERT_DEFERRED, 239_001, 61_000),
    ]
    assert all(alert.event.phase == PHASE_ONSET for alert in run.alerts)
    counts = {kind: sum(1 for alert in run.alerts if alert.kind == kind) for kind in demo.ALERT_KINDS}
    assert counts == {ALERT_IMMEDIATE: 3, ALERT_DEFERRED: 4, ALERT_REPEAT: 4}

    # 상태 이벤트는 속도와 무관하다. 시작 일곱 개의 발행 프레임과, 그 프레임에서 경고로 나갔는지(바로 낸 경고만 True)
    onsets = [(r.t_ms, r.event.type, r.alert) for r in run.events if r.event.phase == PHASE_ONSET]
    assert onsets == [
        (1, EVENT_CALIBRATING, False),
        (75_201, EVENT_GAZE_AWAY_LONG, False),
        (108_201, EVENT_GAZE_AWAY_LONG, False),
        (141_101, EVENT_MICROSLEEP, True),
        (144_601, EVENT_MICROSLEEP, False),
        (161_101, EVENT_MICROSLEEP, True),
        (163_101, EVENT_SLEEP, True),
        (182_101, EVENT_MICROSLEEP, False),
        (184_101, EVENT_SLEEP, False),
        (248_001, EVENT_DROWSY, False),
    ]
    ends = [(r.t_ms, r.event.type) for r in run.events if r.event.phase == PHASE_END]
    assert ends == [
        (60_001, EVENT_CALIBRATING),
        (81_201, EVENT_GAZE_AWAY_LONG),
        (111_201, EVENT_GAZE_AWAY_LONG),
        (141_701, EVENT_MICROSLEEP),
        (146_401, EVENT_MICROSLEEP),
        (170_701, EVENT_SLEEP),
        (193_201, EVENT_SLEEP),
    ]

    # 그림: 종류마다 표식 모양이 다르고(채운 세모, 빈 세모, 세로 금) 범례에 세 종류가 모두 있다
    figure = demo.render_timeline(run, pipeline_spec, rules, ["title"], ["note"], "Model output", None)
    ax_events = figure.axes[3]
    series = [(line.get_marker(), len(line.get_xdata())) for line in ax_events.lines]
    assert series == [("v", 3), ("v", 4), ("|", 4)]
    immediate, deferred, repeat = ax_events.lines
    assert immediate.get_markerfacecolor() != deferred.get_markerfacecolor()  # 채운 세모와 빈 세모
    assert deferred.get_markeredgecolor() == immediate.get_markerfacecolor() == repeat.get_markeredgecolor()
    assert [float(x) for x in deferred.get_xdata()] == pytest.approx([80.0, 146.1, 186.0, 300.0])
    assert [float(x) for x in repeat.get_xdata()] == pytest.approx([166.1, 169.1, 189.0, 192.0])
    rows = {name: row for row, name in enumerate(EVENT_TYPES)}
    wanted_rows = [EVENT_GAZE_AWAY_LONG, EVENT_MICROSLEEP, EVENT_SLEEP, EVENT_DROWSY]
    assert [round(float(y) + 0.36) for y in deferred.get_ydata()] == [rows[name] for name in wanted_rows]
    legend = ax_events.get_legend()
    assert [text.get_text() for text in legend.get_texts()][-3:] == ["alert", "deferred alert", "repeat alert"]
    handles = legend.legend_handles[-3:]
    assert [handle.get_marker() for handle in handles] == ["v", "v", "|"]
    assert handles[0].get_markerfacecolor() == immediate.get_markerfacecolor()
    assert handles[1].get_markerfacecolor() == deferred.get_markerfacecolor()


def test_alert_kinds_reach_the_events_json(
    tmp_path: Path, pipeline_spec: PipelineSpec, model_v0: Path, demo_cfg: ExperimentConfig
) -> None:
    """같은 각본을 명령과 같은 경로(`run_demo_raw`, 속도는 세션 내내 30 km/h)로 돌린 이벤트 JSON의 `kind`와 계수.

    30 km/h에서는 시선·감김 경고(20 km/h부터)가 모두 나가고 졸음 경고(50 km/h부터)는 나가지 않는다. 속도가 바뀌지
    않으므로 미뤘다 낸 경고는 재경고 대기가 풀린 하나뿐이다(146.1초의 microsleep). 손 계산:
      gaze_away_long  75.2, 108.2초에 바로(둘 사이가 대기 10초보다 길다)
      microsleep      141.1초 바로, 144.6초의 시작은 대기 5초에 걸려 146.1초에 미뤘다 냄, 161.1초와 182.1초 바로
      sleep           163.1초 바로 + 166.1, 169.1초 되풀이. 184.1초 바로 + 187.1, 190.1, 193.1초 되풀이(193.2초에 종료)
    바로 7, 미뤘다 1, 되풀이 5. 기본 판정원(perclos)의 drowsy 상태는 생기지만 경고는 없다.
    """
    parquet = tmp_path / "kinds.parquet"
    metadata = {"source": "synthetic (drowsy.synth)", "synthetic": True}
    extract.write_raw_frames(kinds_frames(pipeline_spec), parquet, pipeline_spec, metadata)
    _, json_path = run_demo_raw(parquet, model_v0, tmp_path / "out", 30.0, cfg=demo_cfg)
    report = json.loads(json_path.read_text(encoding="utf-8"))

    alerts = [(a["t_ms"], a["type"], a["kind"]) for a in report["alerts"] if a["type"] != EVENT_DROWSY]
    assert alerts == [
        (75_201, EVENT_GAZE_AWAY_LONG, ALERT_IMMEDIATE),
        (108_201, EVENT_GAZE_AWAY_LONG, ALERT_IMMEDIATE),
        (141_101, EVENT_MICROSLEEP, ALERT_IMMEDIATE),
        (146_101, EVENT_MICROSLEEP, ALERT_DEFERRED),
        (161_101, EVENT_MICROSLEEP, ALERT_IMMEDIATE),
        (163_101, EVENT_SLEEP, ALERT_IMMEDIATE),
        (166_101, EVENT_SLEEP, ALERT_REPEAT),
        (169_101, EVENT_SLEEP, ALERT_REPEAT),
        (182_101, EVENT_MICROSLEEP, ALERT_IMMEDIATE),
        (184_101, EVENT_SLEEP, ALERT_IMMEDIATE),
        (187_101, EVENT_SLEEP, ALERT_REPEAT),
        (190_101, EVENT_SLEEP, ALERT_REPEAT),
        (193_101, EVENT_SLEEP, ALERT_REPEAT),
    ]
    assert len(report["alerts"]) == 13  # 졸음 경고는 없다
    assert "drowsy:onset" in kinds(report["events"]) and find(report["events"], "drowsy:onset")["alert"] is False
    for alert in report["alerts"]:
        assert set(alert) == {field.name for field in dataclasses.fields(DriverStateEvent)} | {"t_ms", "kind"}
    counters = report["counters"]
    assert counters["alerts"] == 13
    assert (counters["alerts_immediate"], counters["alerts_deferred"], counters["alerts_repeat"]) == (7, 1, 5)
    # 미뤘다 낸 경고는 그 시작 이벤트의 `alert`를 바꾸지 않는다(그 프레임에서는 나가지 않았다)
    blocked = find(report["events"], "microsleep:onset", 1)
    assert (blocked["t_ms"], blocked["alert"]) == (144_601, False)
    deferred = report["alerts"][3]
    assert (deferred["t_start_ms"], deferred["duration_ms"]) == (blocked["t_start_ms"], 146_101 - 143_571)


def test_alert_kinds_classifier_follows_the_state_of_each_type() -> None:
    def event(kind: str, phase: str, t_start_ms: int, duration_ms: int) -> DriverStateEvent:
        return DriverStateEvent(kind, 3, t_start_ms, duration_ms, 1.0, SOURCE_CLOSURE, phase, False)

    kinds_of = demo.AlertKinds()
    sleep_onset = event(EVENT_SLEEP, PHASE_ONSET, 1000, 3000)
    # 시작한 프레임에서 나간 경고는 immediate, 같은 상태의 그다음 사본은 repeat
    assert kinds_of.classify([sleep_onset], [sleep_onset]) == [ALERT_IMMEDIATE]
    assert kinds_of.classify([], [event(EVENT_SLEEP, PHASE_ONSET, 1000, 6000)]) == [ALERT_REPEAT]
    assert kinds_of.classify([], []) == []
    # 다른 종류는 따로 센다: 막혔던 drowsy의 첫 사본은 deferred이고, sleep의 사본은 여전히 repeat다
    drowsy_onset = event(EVENT_DROWSY, PHASE_ONSET, 5000, 9000)
    assert kinds_of.classify([drowsy_onset], []) == []
    late = [event(EVENT_SLEEP, PHASE_ONSET, 1000, 9000), event(EVENT_DROWSY, PHASE_ONSET, 5000, 30_000)]
    assert kinds_of.classify([], late) == [ALERT_REPEAT, ALERT_DEFERRED]
    # 상태가 끝나고 같은 종류가 새로 시작하면 처음부터 다시 가린다(막혔다가 나간 첫 경고는 deferred, 그 뒤는 repeat)
    assert kinds_of.classify([event(EVENT_SLEEP, PHASE_END, 1000, 12_000)], []) == []
    assert kinds_of.classify([event(EVENT_SLEEP, PHASE_ONSET, 20_000, 3000)], []) == []
    assert kinds_of.classify([], [event(EVENT_SLEEP, PHASE_ONSET, 20_000, 5000)]) == [ALERT_DEFERRED]
    assert kinds_of.classify([], [event(EVENT_SLEEP, PHASE_ONSET, 20_000, 8000)]) == [ALERT_REPEAT]
    # 같은 프레임에 끝과 새 시작이 함께 오고 그 시작이 바로 나가면 immediate다
    again = event(EVENT_SLEEP, PHASE_ONSET, 40_000, 3000)
    assert kinds_of.classify([event(EVENT_SLEEP, PHASE_END, 20_000, 9000), again], [again]) == [ALERT_IMMEDIATE]
    # 가정: 시작을 본 적 없는 종류의 사본은 그 상태의 첫 경고로 본다
    assert demo.AlertKinds().classify([], [event(EVENT_MICROSLEEP, PHASE_ONSET, 0, 2000)]) == [ALERT_DEFERRED]
    assert demo.ALERT_KINDS == ("immediate", "deferred", "repeat")


def test_probability_track_has_one_point_per_step_after_a_stall(
    pipeline_spec: PipelineSpec, rules_config: RulesConfig
) -> None:
    """프레임이 2초 넘게 끊겼다가 오면 그 프레임이 스텝을 여럿 발행한다. 모델 출력은 스텝마다 하나이고 점도 그렇다.

    세션: 300초, 10 fps, 세션 시간 [250.05초, 252.35초)의 프레임이 없다. 끊기기 전 마지막 프레임은 250.0초(스텝 250을
    발행), 다음 프레임은 252.4초(간격 2400 ms)이고 스텝 251과 252를 함께 발행한다. 모델은 부른 순서 i(0부터)에 대해
    p_drowsy = i / 100을 낸다.
    손 계산: 첫 출력은 스텝 239(i = 0), 마지막 프레임 299.9초가 발행하는 마지막 스텝은 299 -> 출력 61개(i = 0..60).
    스텝 251은 i = 12, 스텝 252는 i = 13이고 두 점의 발행 프레임은 같다(t_ms 252 401).
    결정 전(프레임마다 점 하나)에는 스텝 251의 점이 없어 60개였고 선이 250과 252 사이에서 끊겼다.
    """
    scenario = synth.Scenario(segments=(), duration_ms=300_000, fps=10.0, gaps=((250_050, 252_350),))
    frames = synth.render(scenario, pipeline_spec)
    assert max(b.t_ms - a.t_ms for a, b in zip(frames, frames[1:])) == 2400
    calls = {"n": 0}

    def model(window: Any) -> tuple[float, float, float]:
        p = calls["n"] / 100.0
        calls["n"] += 1
        return (1.0 - p, 0.0, p)

    run = demo.run_pipeline(frames, pipeline_spec, rules_config, model, None)
    assert [point.k for point in run.probs] == list(range(239, 300))
    assert len(run.probs) == run.counters["model_calls"] == 61
    assert [point.probs[2] for point in run.probs] == [i / 100.0 for i in range(61)]
    assert [point.t_end_ms for point in run.probs] == [1 + 1000 * k for k in range(239, 300)]
    stalled = [point for point in run.probs if point.t_ms == 252_401]
    assert [(point.k, point.t_end_ms, point.probs[2]) for point in stalled] == [
        (251, 251_001, 0.12),
        (252, 252_001, 0.13),
    ]
    assert all(point.t_ms == point.t_end_ms for point in run.probs if point.k not in (251, 252))

    # 그림: 스텝 번호가 이어지므로 선은 하나이고(끊기지 않는다) 점 61개가 스텝의 끝 시각에 있다
    figure = demo.render_timeline(run, pipeline_spec, rules_config, ["title"], ["note"], "Model output", None)
    curves = [line for line in figure.axes[2].lines if len(line.get_xdata()) > 2]
    assert len(curves) == 2 and all(len(curve.get_xdata()) == 61 for curve in curves)
    assert [float(x) for x in curves[1].get_xdata()] == pytest.approx([float(k) for k in range(239, 300)])
    assert [float(y) for y in curves[1].get_ydata()] == pytest.approx([i / 100.0 for i in range(61)])
    # JSON의 출력 수도 모델을 부른 횟수와 같다
    assert demo.probability_gap_reason(run, pipeline_spec) is None


# ---------------------------------------------------------------------------
# 모델 메타데이터 대조
# ---------------------------------------------------------------------------


def test_verify_model_accepts_the_exported_model(model_v0: Path, feature_spec: FeatureSpec) -> None:
    info = demo.verify_model(model_v0, feature_spec)
    assert info.path == model_v0 and info.sha256 == file_sha256(model_v0)
    assert info.spec_copy == model_v0.with_name("feature_spec_v0.json")
    assert info.feature_spec_hash == info.metadata["feature_spec_hash"]
    assert info.metadata["pipeline_hash"] == feature_spec.pipeline_hash


def test_model_with_mismatching_metadata_is_refused(
    tmp_path: Path,
    session_parquet: Path,
    model_v0: Path,
    demo_cfg: ExperimentConfig,
    feature_spec: FeatureSpec,
    feature_spec_path: Path,
) -> None:
    out_dir = tmp_path / "out"

    # (1) 모델 메타데이터의 feature_spec_hash가 옆의 사본과 다르다
    wrong_spec_hash = copy_model(model_v0, tmp_path / "wrong_spec_hash", feature_spec_hash="0" * 64)
    with pytest.raises(DemoError, match="feature_spec_hash mismatch"):
        run_demo_raw(session_parquet, wrong_spec_hash, out_dir, cfg=demo_cfg)

    # (2) 사본의 바이트가 바뀌었다(줄바꿈 하나)
    edited_copy = copy_model(model_v0, tmp_path / "edited_copy")
    with exported_spec_path(edited_copy).open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(DemoError, match="feature_spec_hash mismatch"):
        run_demo_raw(session_parquet, edited_copy, out_dir, cfg=demo_cfg)

    # (3) 모델 메타데이터의 pipeline_hash가 다르다
    wrong_pipeline = copy_model(model_v0, tmp_path / "wrong_pipeline", pipeline_hash="f" * 64)
    with pytest.raises(DemoError, match="pipeline_hash mismatch"):
        run_demo_raw(session_parquet, wrong_pipeline, out_dir, cfg=demo_cfg)

    # (4) 파이프라인을 돌릴 feature_spec의 상수가 모델이 학습한 것과 다르다
    spec_dict = read_feature_spec_dict(feature_spec_path)
    spec_dict["pipeline"]["closure"]["rho"] = 0.4
    spec_dict["pipeline_hash"] = compute_pipeline_hash(spec_dict["pipeline"])
    other_spec = tmp_path / "feature_spec.json"
    write_feature_spec(spec_dict, other_spec)
    other_cfg = dataclasses.replace(demo_cfg, paths=dataclasses.replace(demo_cfg.paths, feature_spec=other_spec))
    with pytest.raises(DemoError, match="pipeline_hash mismatch") as caught:
        run_demo_raw(session_parquet, model_v0, out_dir, cfg=other_cfg)
    assert feature_spec.pipeline_hash in str(caught.value) and spec_dict["pipeline_hash"] in str(caught.value)

    # (5) 사본이 없다, (6) 내보낸 모델이 아니다(메타데이터 없음), (7) 모델 파일이 없거나 ONNX가 아니다
    no_copy = copy_model(model_v0, tmp_path / "no_copy")
    exported_spec_path(no_copy).unlink()
    with pytest.raises(DemoError, match="feature spec copy shipped with the model is missing"):
        run_demo_raw(session_parquet, no_copy, out_dir, cfg=demo_cfg)
    bare = onnx.load(str(model_v0))
    del bare.metadata_props[:]
    onnx.save(bare, str(tmp_path / "bare.onnx"))
    with pytest.raises(DemoError, match="metadata lacks"):
        run_demo_raw(session_parquet, tmp_path / "bare.onnx", out_dir, cfg=demo_cfg)
    with pytest.raises(DemoError, match="ONNX model not found"):
        run_demo_raw(session_parquet, tmp_path / "missing.onnx", out_dir, cfg=demo_cfg)
    (tmp_path / "garbage.onnx").write_bytes(b"not an onnx file")
    with pytest.raises(DemoError, match="not a readable ONNX file"):
        run_demo_raw(session_parquet, tmp_path / "garbage.onnx", out_dir, cfg=demo_cfg)

    assert not out_dir.exists()  # 거부하면 아무것도 쓰지 않는다


def test_raw_parquet_from_another_mediapipe_model_is_refused(
    tmp_path: Path, pipeline_spec: PipelineSpec, model_v0: Path, demo_cfg: ExperimentConfig
) -> None:
    parquet = tmp_path / "other_model.parquet"
    frames = scripted_frames(pipeline_spec, 5_000)
    extract.write_raw_frames(frames, parquet, pipeline_spec, {"source": "x.mp4", "model_sha256": "0" * 64})
    with pytest.raises(DemoError, match="extracted with MediaPipe model"):
        run_demo_raw(parquet, model_v0, tmp_path / "out", cfg=demo_cfg)
    with pytest.raises(DemoError, match="raw parquet not found"):
        run_demo_raw(tmp_path / "missing.parquet", model_v0, tmp_path / "out", cfg=demo_cfg)

    # 같은 MediaPipe 모델로 추출한 것은 받고, 합성 표시가 없으면 합성 데이터라고 적지 않는다
    extract.write_raw_frames(
        frames, parquet, pipeline_spec, {"source": "x.mp4", "model_sha256": pipeline_spec.mediapipe.model_sha256}
    )
    _, json_path = run_demo_raw(parquet, model_v0, tmp_path / "out", cfg=demo_cfg)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["note"] == demo.NOTE_RECORDING and report["source"]["synthetic"] is False
    assert report["source"]["extract_source"] == "x.mp4"


# ---------------------------------------------------------------------------
# 명령
# ---------------------------------------------------------------------------


def test_cli_raw_with_defaults(
    tmp_path: Path,
    config_path: Path,
    repo_root: Path,
    session_parquet: Path,
    model_v0: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # demo.model_path의 파일이 아직 없으면 export_dir의 무작위 가중치 모델을 쓰고 알린다. 출력은 <demo.out_dir>/demo
    config = write_cli_config(
        tmp_path, config_path, repo_root, model_path=str(tmp_path / "drowsiness_v1.onnx"), out_dir=str(tmp_path / "out")
    )
    assert demo.main(["--config", str(config), "--raw", str(session_parquet), "--speed-kmh", "60"]) == 0
    captured = capsys.readouterr()
    assert "does not exist yet" in captured.err and str(model_v0) in captured.err
    lines = captured.out.splitlines()
    assert lines[0] == SYNTHETIC_NOTE  # 합성 입력이면 출력의 첫 줄에 적는다
    # 둘째 줄은 parquet의 회전 기록이다. 합성 세션에는 영상이 없으므로 기록이 없다고 적는다.
    assert lines[1] == "rotation: no rotation-check record (synthetic session: there was no video to turn upright)"
    assert [line for line in lines if line.startswith("rotation:")] == [lines[1]] and "WARNING" not in captured.out
    out_dir = tmp_path / "out" / "demo"
    assert f"wrote {out_dir / 'scripted_timeline.png'}" in captured.out
    assert f"wrote {out_dir / 'scripted_events.json'}" in captured.out
    assert "events 13, alerts 7, model outputs 61, drowsiness.source perclos" in captured.out
    report = json.loads((out_dir / "scripted_events.json").read_text(encoding="utf-8"))
    assert report["model"]["file"] == str(model_v0) and report["speed_kmh"] == 60.0
    source = report["source"]
    assert source["rotation_note"] == lines[1]  # 출력한 줄 그대로 JSON에 적는다
    assert (source["rotation"], source["rotation_source"], source["rotation_check"]) == (None, None, None)
    assert source["rotation_ambiguous"] is False

    # demo.model_path의 파일이 있으면 그것을 쓴다. --model과 --out은 설정보다 우선한다
    configured = copy_model(model_v0, tmp_path / "configured")
    config = write_cli_config(tmp_path, config_path, repo_root, model_path=str(configured), out_dir=str(tmp_path / "o"))
    assert demo.main(["--config", str(config), "--raw", str(session_parquet)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    report = json.loads((tmp_path / "o" / "demo" / "scripted_events.json").read_text(encoding="utf-8"))
    assert report["model"]["file"] == str(configured)
    argv = ["--config", str(config), "--raw", str(session_parquet), "--model", str(model_v0), "--out"]
    assert demo.main([*argv, str(tmp_path / "explicit")]) == 0
    capsys.readouterr()
    report = json.loads((tmp_path / "explicit" / "scripted_events.json").read_text(encoding="utf-8"))
    assert report["model"]["file"] == str(model_v0)


def test_cli_reports_errors_with_exit_code_2(
    tmp_path: Path,
    config_path: Path,
    repo_root: Path,
    session_parquet: Path,
    model_v0: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_cli_config(tmp_path, config_path, repo_root, out_dir=str(tmp_path / "out"))
    base = ["--config", str(config), "--out", str(tmp_path / "out")]

    wrong = copy_model(model_v0, tmp_path / "wrong", feature_spec_hash="0" * 64)
    assert demo.main([*base, "--raw", str(session_parquet), "--model", str(wrong)]) == 2
    assert "error: feature_spec_hash mismatch" in capsys.readouterr().err

    not_parquet = tmp_path / "notes.parquet"
    not_parquet.write_text("not a parquet file", encoding="utf-8")
    assert demo.main([*base, "--raw", str(not_parquet), "--model", str(model_v0)]) == 2
    assert capsys.readouterr().err.startswith("error:")

    assert demo.main([*base, "--video", str(tmp_path / "missing.mp4"), "--model", str(model_v0)]) == 2
    assert "video not found" in capsys.readouterr().err
    assert demo.main(["--config", str(tmp_path / "missing.yaml"), "--raw", str(session_parquet)]) == 2
    assert capsys.readouterr().err.startswith("error:")
    assert not (tmp_path / "out").exists()

    for argv in ([*base], [*base, "--raw", str(session_parquet), "--video", "x.mp4"]):
        with pytest.raises(SystemExit):  # --video와 --raw 중 정확히 하나가 필요하다
            demo.main(argv)
    capsys.readouterr()


# ---------------------------------------------------------------------------
# 영상 입력의 회전 (가짜 랜드마커와 표식 그림. face_landmarker.task가 없어도 돈다)
# ---------------------------------------------------------------------------

MARKER_W, MARKER_H = 160, 120
# 실제 랜드마커가 정면 얼굴 한 장(MediaPipe 공개 시험 이미지)에서 낸 roll(도). 키는 그림이 시계 방향으로 돌아간 횟수다:
# 똑바로 −0.6, 시계 방향 90도로 누우면 −90.6, 거꾸로 서면 −1.9(똑바로 선 얼굴로 잘못 맞춰진다), 반시계 90도로 누우면 89.4.
MARKER_ROLL = {0: -0.6, 1: -90.6, 2: -1.9, 3: 89.4}
IDENTITY_DISPLAY_MATRIX = struct.pack(">9i", 0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000)


def marker_image(index: int) -> np.ndarray:
    """똑바로 섰을 때 왼쪽 위에 흰 사각형이 있는 그림(얼굴이 아니다). 가짜 랜드마커가 이 표식으로 방향을 알아본다."""
    image = np.full((MARKER_H, MARKER_W, 3), 40, dtype=np.uint8)
    image[:40, :40] = 255
    shift = 2 * (index % 20)
    image[60:80, 60 + shift : 80 + shift] = (90, 160, 90)
    return image


def marker_turns_cw(rgb: np.ndarray) -> int:
    """흰 사각형이 있는 모서리로, 그림이 똑바로 선 상태에서 시계 방향으로 몇 번(90도 단위) 돌아가 있는지 알아낸다."""
    corners = [rgb[:20, :20], rgb[:20, -20:], rgb[-20:, -20:], rgb[-20:, :20]]  # 왼쪽 위, 오른쪽 위, 오른쪽 아래, 왼쪽 아래
    return int(np.argmax([corner.mean() for corner in corners]))


def write_marker_clip(path: Path, n_frames: int, quarter_turns_cw: int, flip_tag: bool = False) -> None:
    """표식 그림 n_frames장(10 fps)을 시계 방향으로 quarter_turns_cw번 돌려 회전 메타데이터 없이 저장한다.

    flip_tag가 참이면 mp4의 표시 행렬을 좌우 뒤집기로 바꿔 쓴다(그림 자체는 그대로다).
    """
    time_base = Fraction(1, 1000)
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), "w", format="mp4")
    stream = container.add_stream("libx264", rate=30)
    first = np.rot90(marker_image(0), -quarter_turns_cw)
    stream.width, stream.height, stream.pix_fmt = first.shape[1], first.shape[0], "yuv420p"
    stream.codec_context.time_base = time_base
    stream.time_base = time_base
    stream.options = {"bf": "0", "crf": "20"}
    for index in range(n_frames):
        image = np.ascontiguousarray(np.rot90(marker_image(index), -quarter_turns_cw))
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts = 100 * index
        frame.time_base = time_base
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    if flip_tag:
        data = bytearray(path.read_bytes())
        tkhd = data.find(b"tkhd")
        at = data.find(IDENTITY_DISPLAY_MATRIX, tkhd)
        assert tkhd > 0 and 0 < at - tkhd < 80, "the file does not hold an identity display matrix"
        data[at : at + 36] = struct.pack(">9i", -(1 << 16), 0, 0, 0, 1 << 16, 0, 0, 0, 0x40000000)
        path.write_bytes(bytes(data))


class MarkerLandmarker:
    def __init__(self) -> None:
        self.closed = False
        self.seen: list[tuple[int, int, int]] = []  # (t_ms, w, h)
        self.turns: list[int] = []  # 받은 그림이 시계 방향으로 돌아가 있던 횟수

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def marker_landmarker(monkeypatch: pytest.MonkeyPatch, pipeline_spec: PipelineSpec) -> SimpleNamespace:
    """`extract`의 랜드마커와 추론을 가짜로 바꾼다: 어느 방향에서나 얼굴을 찾고 방향에 따라 다른 roll을 낸다.

    `upside_down_found`를 False로 바꾸면 거꾸로 선 그림에서는 얼굴을 못 찾는다(실제 랜드마커가 다른 시험 이미지에서 보인
    동작). 돌려주는 프레임은 합성 세션의 얼굴 프레임에 변환 행렬만 그 roll의 z축 회전으로 바꾼 것이다.
    """
    state = SimpleNamespace(created=[], upside_down_found=True)
    template = next(frame for frame in scripted_frames(pipeline_spec, 1_000) if frame.face)

    def make_landmarker(model_path: Path, spec: PipelineSpec | None = None) -> MarkerLandmarker:
        landmarker = MarkerLandmarker()
        state.created.append(landmarker)
        return landmarker

    def detect_frame(landmarker: MarkerLandmarker, t_ms: int, rgb: np.ndarray, spec: PipelineSpec) -> Any:
        assert not landmarker.closed
        assert all(t_ms > seen_t for seen_t, _, _ in landmarker.seen)  # VIDEO 모드는 엄격히 증가하는 값만 받는다
        height, width = rgb.shape[0], rgb.shape[1]
        turns = marker_turns_cw(rgb)
        landmarker.seen.append((t_ms, width, height))
        landmarker.turns.append(turns)
        if turns == 2 and not state.upside_down_found:
            return dataclasses.replace(template, t_ms=t_ms, face=False, w=width, h=height, lm={}, bs={}, m=())
        cos, sin = math.cos(math.radians(MARKER_ROLL[turns])), math.sin(math.radians(MARKER_ROLL[turns]))
        m = (cos, sin, 0.0, 0.0, -sin, cos, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -2.0, -30.0, 1.0)
        return dataclasses.replace(template, t_ms=t_ms, w=width, h=height, m=m)

    monkeypatch.setattr(extract, "make_landmarker", make_landmarker)
    monkeypatch.setattr(extract, "detect_frame", detect_frame)
    monkeypatch.setattr(extract, "mediapipe_version", lambda: "0.0-fake")
    return state


def test_cli_video_checks_the_rotation_once_and_records_it(
    tmp_path: Path,
    config_path: Path,
    repo_root: Path,
    model_v0: Path,
    marker_landmarker: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--rotation` 없이: 추출과 같은 자동 점검을 한 번 돌려 고른 각도로 세우고, 각도와 출처를 출력과 JSON에 남긴다."""
    marker_landmarker.upside_down_found = False  # 거꾸로 선 얼굴을 못 찾는 얼굴: 점검이 끝까지 가른다
    video = tmp_path / "videos" / "lying.mp4"
    write_marker_clip(video, 30, quarter_turns_cw=1)  # 3초, 회전 메타데이터 없이 시계 방향 90도로 누움: 맞는 각도는 90
    config = write_cli_config(tmp_path, config_path, repo_root)
    out_dir = tmp_path / "out"
    argv = ["--config", str(config), "--video", str(video), "--model", str(model_v0), "--out", str(out_dir)]
    assert demo.main(argv) == 0
    lines = capsys.readouterr().out.splitlines()

    # 고른 각도와 가르지 못했는지를 프레임을 돌리기 전에 먼저 출력한다. 기본 설정(2000 ms 간격)에서 3초 영상의 표본은
    # 2장이다. 후보 0은 시계 방향 90도로 누운 채(roll −90.6), 90은 똑바로(−0.6), 180은 반시계로 누운 채(89.4),
    # 270은 거꾸로(검출 없음)다.
    assert lines[0] == (
        "rotation: auto check chose 90, ambiguous: no (metadata 0; smallest_roll; face rate over 2 frames "
        "0:1.00 90:1.00 180:1.00 270:0.00; median roll 0:-90.6 90:-0.6 180:89.4 270:-) "
        "WARNING rotation differs from the video metadata"
    )
    assert lines[1] == demo.NOTE_RECORDING and lines[2].startswith("wrote ")

    # 점검은 한 번이다: 후보 각도마다 랜드마커 하나가 표본 2장을 보고, 그 뒤 실행용 랜드마커 하나가 30장을 모두 본다.
    created = marker_landmarker.created
    assert [len(landmarker.seen) for landmarker in created] == [2, 2, 2, 2, 30]
    assert all(landmarker.closed for landmarker in created)
    assert [landmarker.turns[0] for landmarker in created[:4]] == [1, 0, 3, 2]  # 후보 0, 90, 180, 270으로 돌린 그림
    # 파이프라인에 들어간 프레임은 모두 세워진 그림이다.
    assert set(created[4].turns) == {0}
    assert {(width, height) for _, width, height in created[4].seen} == {(MARKER_W, MARKER_H)}

    report = json.loads((out_dir / "lying_events.json").read_text(encoding="utf-8"))
    source = report["source"]
    assert (source["kind"], source["rotation"], source["n_decoded"]) == (demo.SOURCE_VIDEO, 90, 30)
    assert (source["rotation_source"], source["rotation_ambiguous"]) == ("auto_check", False)
    assert source["display_flip"] is False
    check = source["rotation_check"]  # 추출 결과의 메타데이터에 적는 기록과 같은 꼴이다
    expected_rolls = {"0": pytest.approx(-90.6), "90": pytest.approx(-0.6), "180": pytest.approx(89.4), "270": None}
    assert check["rolls"] == expected_rolls
    assert {key: value for key, value in check.items() if key != "rolls"} == {
        "version": 2,
        "settings": {"frames": 12, "interval_ms": 2000, "roll_margin_deg": 15.0},
        "rates": {"0": 1.0, "90": 1.0, "180": 1.0, "270": 0.0},
        "metadata_rotation": 0,
        "n_frames": 2,
        "ambiguous": False,
        "reason": "smallest_roll",
    }
    assert report["counters"]["frames_accepted"] == report["counters"]["face_frames"] == 30
    assert sorted(path.name for path in out_dir.iterdir()) == ["lying_events.json", "lying_timeline.png"]


def test_cli_video_flags_an_ambiguous_rotation_and_takes_a_manual_one(
    tmp_path: Path,
    config_path: Path,
    repo_root: Path,
    model_v0: Path,
    pipeline_spec: PipelineSpec,
    marker_landmarker: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """거꾸로 저장된 영상: 점검은 0도와 180도를 가르지 못해 틀린 각도(0)로 돌지만 표시한다. `--rotation`으로 바로잡는다."""
    video = tmp_path / "upside_down.mp4"
    write_marker_clip(video, 25, quarter_turns_cw=2)
    config = write_cli_config(tmp_path, config_path, repo_root)
    argv = ["--config", str(config), "--video", str(video), "--model", str(model_v0)]

    assert demo.main([*argv, "--out", str(tmp_path / "auto")]) == 0
    first = capsys.readouterr().out.splitlines()[0]
    assert first.startswith("rotation: auto check chose 0, ambiguous: yes (metadata 0; ambiguous_kept_metadata;")
    assert "WARNING rotation ambiguous" in first and first.endswith("run again with --rotation {0,90,180,270}")
    source = json.loads((tmp_path / "auto" / "upside_down_events.json").read_text(encoding="utf-8"))["source"]
    assert (source["rotation"], source["rotation_source"], source["rotation_ambiguous"]) == (0, "auto_check", True)
    assert source["rotation_check"]["reason"] == "ambiguous_kept_metadata"
    assert set(marker_landmarker.created[4].turns) == {2}  # 거꾸로 선 채로 돌았다(그래서 표시가 필요하다)

    # 사람이 본 각도를 준다: 점검을 하지 않고(랜드마커는 실행용 하나) 그 각도로 세운다.
    before = len(marker_landmarker.created)
    assert demo.main([*argv, "--out", str(tmp_path / "manual"), "--rotation", "180"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "rotation: 180 (given with --rotation; the automatic check was not run)"
    assert len(marker_landmarker.created) == before + 1 and set(marker_landmarker.created[-1].turns) == {0}
    source = json.loads((tmp_path / "manual" / "upside_down_events.json").read_text(encoding="utf-8"))["source"]
    assert (source["rotation"], source["rotation_source"]) == (180, "manual_override")
    assert source["rotation_ambiguous"] is False
    assert source["rotation_check"] is None and source["display_flip"] is False

    # --rotation은 영상 입력에만 쓴다. 값은 0, 90, 180, 270뿐이다.
    parquet = write_session(tmp_path / "short.parquet", pipeline_spec, 2_000)
    raw_argv = ["--config", str(config), "--raw", str(parquet), "--model", str(model_v0)]
    raw_argv += ["--out", str(tmp_path / "raw")]
    assert demo.main([*raw_argv, "--rotation", "90"]) == 2
    assert "error: --rotation needs --video" in capsys.readouterr().err
    assert not (tmp_path / "raw").exists()
    with pytest.raises(SystemExit):
        demo.main([*argv, "--rotation", "45"])
    capsys.readouterr()


def test_cli_video_with_a_flip_tag_is_refused_with_a_message_that_names_rotation(
    tmp_path: Path,
    config_path: Path,
    repo_root: Path,
    model_v0: Path,
    marker_landmarker: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
) -> None:
    video = tmp_path / "flipped.mp4"
    write_marker_clip(video, 12, quarter_turns_cw=0, flip_tag=True)
    with av.open(str(video)) as container:
        assert next(container.decode(video=0)).rotation == -180  # 좌우 뒤집기만 있는데 libav는 회전 −180으로 읽는다
    config = write_cli_config(tmp_path, config_path, repo_root)
    out_dir = tmp_path / "out"
    argv = ["--config", str(config), "--video", str(video), "--model", str(model_v0), "--out", str(out_dir)]

    assert demo.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith(f"error: {video}: the display matrix contains a flip")
    assert "--rotation" in captured.err and "python -m drowsy.demo --video" in captured.err
    assert captured.out == "" and not out_dir.exists() and marker_landmarker.created == []

    # 각도를 주면 그 각도로만 돌려 읽고(거울상은 되돌리지 않는다) 그 사실을 JSON에 남긴다.
    assert demo.main([*argv, "--rotation", "0"]) == 0
    capsys.readouterr()
    source = json.loads((out_dir / "flipped_events.json").read_text(encoding="utf-8"))["source"]
    assert (source["rotation"], source["rotation_source"], source["display_flip"]) == (0, "manual_override", True)
    assert set(marker_landmarker.created[-1].turns) == {0}  # 저장된 그림 그대로다


def test_run_demo_rotation_arguments(
    tmp_path: Path, model_v0: Path, demo_cfg: ExperimentConfig, marker_landmarker: SimpleNamespace
) -> None:
    video = tmp_path / "lying.mp4"
    write_marker_clip(video, 12, quarter_turns_cw=3)  # 반시계 90도로 누움: 맞는 각도는 270(= −90)
    told: list[demo.VideoRotation] = []

    # 모델을 먼저 검증한다: 모델이 틀리면 회전 점검(랜드마커)을 시작하지 않는다.
    with pytest.raises(DemoError, match="ONNX model not found"):
        demo.run_demo(video, tmp_path / "missing.onnx", tmp_path / "out", cfg=demo_cfg, on_rotation=told.append)
    assert marker_landmarker.created == [] and told == []

    # 90의 배수가 아닌 각도는 아무것도 하기 전에 거부한다. 음수 각도는 같은 방향의 0~270으로 바꾼다.
    with pytest.raises(ValueError, match="multiple of 90"):
        demo.run_demo(video, model_v0, tmp_path / "out", cfg=demo_cfg, rotation=45)
    assert marker_landmarker.created == [] and not (tmp_path / "out").exists()
    _, json_path = demo.run_demo(video, model_v0, tmp_path / "out", cfg=demo_cfg, rotation=-90, on_rotation=told.append)
    assert [(chosen.rotation, chosen.source, chosen.ambiguous, chosen.check) for chosen in told] == [
        (270, "manual_override", False, None)
    ]
    assert json.loads(json_path.read_text(encoding="utf-8"))["source"]["rotation"] == 270
    assert set(marker_landmarker.created[-1].turns) == {0}

    # 자동 점검: 이 가짜 검출기는 거꾸로 선 얼굴도 찾으므로 270과 90을 가르지 못하고 |roll|이 작은 270을 고른 뒤 표시한다.
    told.clear()
    demo.run_demo(video, model_v0, tmp_path / "auto", cfg=demo_cfg, on_rotation=told.append)
    assert len(told) == 1
    assert (told[0].rotation, told[0].source, told[0].ambiguous) == (270, "auto_check", True)
    assert told[0].check is not None and told[0].check.reason == "ambiguous_smallest_roll"
    line = demo.describe_video_rotation(told[0])
    assert line.startswith("rotation: auto check chose 270, ambiguous: yes (metadata 0;")


def test_timeline_header_names_the_video_rotation(
    model_v0: Path, feature_spec: FeatureSpec, rules_config: RulesConfig, session_parquet: Path
) -> None:
    spec = feature_spec.pipeline
    run = demo.run_pipeline(scripted_frames(spec, 2_000), spec, rules_config, None, None)
    model = demo.verify_model(model_v0, feature_spec)

    def notes(**details: Any) -> list[str]:
        source = demo.SourceInfo(kind=demo.SOURCE_VIDEO, path=Path("clip.mp4"), synthetic=False, details=details)
        return demo._header_lines(run, source, model, rules_config, None)[1]

    auto = notes(rotation=270, rotation_source="auto_check", rotation_ambiguous=False)
    assert len(auto) == 4 and auto[3] == "video rotation: 270 degrees counter-clockwise (automatic check)"
    flagged = notes(rotation=0, rotation_source="auto_check", rotation_ambiguous=True)
    assert flagged[3] == (
        "video rotation: 0 degrees counter-clockwise "
        "(automatic check, AMBIGUOUS: look at the video and pass --rotation)"
    )
    assert len(flagged[3]) < demo._NOTE_CHARS  # 그림에서 잘리지 않는다
    manual = notes(rotation=90, rotation_source="manual_override", rotation_ambiguous=False)
    assert manual[3] == "video rotation: 90 degrees counter-clockwise (given with --rotation)"

    # parquet 입력에는 회전 줄이 없다(셋 그대로).
    raw = demo.SourceInfo(kind=demo.SOURCE_RAW, path=session_parquet, synthetic=True, details={})
    raw_titles, raw_notes = demo._header_lines(run, raw, model, rules_config, None)
    assert len(raw_titles) == 2 and len(raw_notes) == 3

    # 추출 결과(합성이 아닌 parquet)의 점검 기록이 "가르지 못했다"일 때만 넷째 줄이 붙는다. 할 일은 다시 추출하는 것이다.
    def raw_notes_for(synthetic: bool = False, **details: Any) -> list[str]:
        source = demo.SourceInfo(kind=demo.SOURCE_RAW, path=Path("clip.parquet"), synthetic=synthetic, details=details)
        return demo._header_lines(run, source, model, rules_config, None)[1]

    raw_flagged = raw_notes_for(rotation=0, rotation_source="auto_check", rotation_ambiguous=True)
    assert len(raw_flagged) == 4 and raw_flagged[3] == (
        "video rotation: 0 degrees counter-clockwise (automatic check at extraction, AMBIGUOUS: look at the video "
        "and extract it again with a manual rotation)"
    )
    assert len(raw_flagged[3]) < demo._NOTE_CHARS and "--rotation" not in raw_flagged[3]
    assert len(raw_notes_for(rotation=90, rotation_source="auto_check", rotation_ambiguous=False)) == 3
    assert len(raw_notes_for(rotation=180, rotation_source="manual_override", rotation_ambiguous=False)) == 3
    assert len(raw_notes_for(rotation=None, rotation_source=None, rotation_ambiguous=False)) == 3
    # 합성 입력의 머리말은 이미 다섯 줄이라 더 붙이지 않는다(합성 세션에는 점검 기록이 없다).
    assert len(raw_notes_for(synthetic=True, rotation=0, rotation_source="auto_check", rotation_ambiguous=True)) == 3

    # 영상 입력의 머리말(굵은 줄 하나, 작은 줄 넷)은 합성 입력의 머리말(굵은 줄 둘, 작은 줄 셋)보다 아래로 내려가지 않는다.
    def lowest_header_line(titles: list[str], small: list[str]) -> float:
        figure = demo.render_timeline(run, spec, rules_config, titles, small, "Model output", None)
        header = [text for text in figure.texts if text.get_position()[0] < 0.5]  # 왼쪽 위의 머리말(오른쪽 아래 문구 제외)
        assert len(header) == len(titles) + len(small)
        return min(text.get_position()[1] for text in header)

    assert lowest_header_line(["Driver state timeline: clip.mp4"], flagged) >= lowest_header_line(raw_titles, raw_notes)
    # parquet 입력에 넷째 줄이 붙어도(굵은 줄 하나, 작은 줄 넷) 마찬가지다.
    raw_title = ["Driver state timeline: clip.parquet"]
    assert lowest_header_line(raw_title, raw_flagged) >= lowest_header_line(raw_titles, raw_notes)


# ---------------------------------------------------------------------------
# parquet 입력(`--raw`)의 회전 기록: 추출이 메타데이터에 남긴 것을 읽어 알린다
# ---------------------------------------------------------------------------

# 점검이 가르지 못했을 때 `--raw`가 알려 주는 할 일. 구현과 따로 손으로 적었다.
RAW_AMBIGUOUS_ADVICE = (
    "Look at the video and extract it again with the angle given by hand (python -m drowsy.extract --video ... "
    "--rotation {0,90,180,270}, or the manual rotation table of the config for the other inputs)"
)
VIDEO_MANUAL_LINE = "rotation: 180 (given with --rotation; the automatic check was not run)"  # `--video --rotation 180`의 줄


def check_record(
    rotation: int,
    metadata_rotation: int,
    rates: tuple[float, float, float, float],
    rolls: tuple[float | None, float | None, float | None, float | None],
    ambiguous: bool,
    reason: str,
) -> dict[str, Any]:
    """추출이 parquet 메타데이터의 `rotation_check`에 적는 것과 같은 기록(`extract.rotation_check_record`). 값은 손으로 준다."""
    angles = (0, 90, 180, 270)
    check = extract.RotationCheck(
        rotation, metadata_rotation, dict(zip(angles, rates)), dict(zip(angles, rolls)), 2, ambiguous, reason
    )
    return extract.rotation_check_record(check, extract.RotationCheckSettings(12, 2000, 15.0))


# 거꾸로 저장된 영상: 얼굴이 네 각도에서 다 검출되고 0도와 180도의 roll이 비슷해서 점검이 가르지 못하고 메타데이터(0)를 유지했다.
UPSIDE_DOWN_RECORD = check_record(
    0, 0, (1.0, 1.0, 1.0, 1.0), (-1.9, 89.4, -0.6, -90.6), True, "ambiguous_kept_metadata"
)
# 시계 방향 90도로 누워 저장된 영상: 거꾸로 선 얼굴은 검출되지 않아 점검이 90도로 정했다(메타데이터는 0).
LYING_RECORD = check_record(90, 0, (1.0, 1.0, 1.0, 0.0), (-90.6, -0.6, 89.4, None), False, "smallest_roll")


def test_read_raw_rotation_tells_what_the_extraction_recorded() -> None:
    flagged = demo.read_raw_rotation({"rotation": 0, "rotation_override": 0, "rotation_check": UPSIDE_DOWN_RECORD})
    assert (flagged.rotation, flagged.source, flagged.ambiguous) == (0, "auto_check", True)
    assert flagged.record == UPSIDE_DOWN_RECORD
    assert flagged.note == (
        "rotation: the extraction's auto check chose 0, ambiguous: yes (metadata 0; ambiguous_kept_metadata; "
        "face rate over 2 frames 0:1.00 90:1.00 180:1.00 270:1.00; median roll 0:-1.9 90:89.4 180:-0.6 270:-90.6) "
        "WARNING rotation ambiguous: the check could not tell this angle from another one (usually the opposite "
        f"one). {RAW_AMBIGUOUS_ADVICE}"
    )

    # 어느 각도에서도 얼굴을 못 찾은 영상도 가르지 못한 것이다. 까닭을 다르게 적는다.
    no_face = check_record(0, 0, (0.0, 0.0, 0.0, 0.0), (None, None, None, None), True, "no_face")
    unseen = demo.read_raw_rotation({"rotation": 0, "rotation_override": 0, "rotation_check": no_face})
    assert unseen.ambiguous is True and unseen.note == (
        "rotation: the extraction's auto check chose 0, ambiguous: yes (metadata 0; no_face; face rate over 2 frames "
        "0:0.00 90:0.00 180:0.00 270:0.00; median roll 0:- 90:- 180:- 270:-) WARNING rotation ambiguous: no face was "
        f"found at any angle in the sampled frames, so the rotation metadata was used unverified. {RAW_AMBIGUOUS_ADVICE}"
    )

    # 점검이 각도를 정했다: 고른 각도가 영상의 회전 메타데이터와 다르면 `--video` 입력처럼 그것만 알린다.
    decided = demo.read_raw_rotation({"rotation": 90, "rotation_override": 90, "rotation_check": LYING_RECORD})
    assert (decided.rotation, decided.source, decided.ambiguous) == (90, "auto_check", False)
    assert decided.note == (
        "rotation: the extraction's auto check chose 90, ambiguous: no (metadata 0; smallest_roll; face rate over "
        "2 frames 0:1.00 90:1.00 180:1.00 270:0.00; median roll 0:-90.6 90:-0.6 180:89.4 270:-) "
        "WARNING rotation differs from the video metadata"
    )
    upright = check_record(0, 0, (1.0, 1.0, 1.0, 0.0), (-0.6, 89.4, -90.6, None), False, "smallest_roll")
    plain = demo.read_raw_rotation({"rotation": 0, "rotation_override": 0, "rotation_check": upright})
    assert plain.ambiguous is False and "WARNING" not in plain.note and plain.note.endswith("median roll 0:-0.6 90:89.4 180:-90.6 270:-)")

    # 점검 기록 없이 수동 값으로 추출됐다: `--video --rotation`의 줄과 같은 꼴이고 그 밖에는 아무것도 붙지 않는다.
    manual = demo.read_raw_rotation({"rotation": 180, "rotation_override": 180})
    assert (manual.rotation, manual.source, manual.ambiguous, manual.record) == (180, "manual_override", False, None)
    assert manual.note == "rotation: 180 (manual value recorded by the extraction; the automatic check was not run)"
    assert manual.note.replace("manual value recorded by the extraction", "given with --rotation") == VIDEO_MANUAL_LINE

    # 점검 기록이 없다: 영상의 회전 메타데이터만 믿고 추출한 전의 결과, 회전이 적혀 있지 않은 파일, 합성 세션.
    legacy = demo.read_raw_rotation({"rotation": 90, "rotation_override": None})
    assert (legacy.rotation, legacy.source, legacy.ambiguous, legacy.record) == (90, "video_metadata", False, None)
    assert legacy.note == (
        "rotation: 90 from the video's rotation metadata, no rotation-check record (the extraction did not check "
        "the angle; extract the video again to run the check)"
    )
    bare = demo.read_raw_rotation({"source": "x.mp4"})
    assert (bare.rotation, bare.source, bare.ambiguous, bare.record) == (None, None, False, None)
    assert bare.note == "rotation: no rotation-check record (the file does not say how the picture was turned upright)"
    synthetic = demo.read_raw_rotation({"source": "synthetic (drowsy.synth)", "synthetic": True})
    assert (synthetic.rotation, synthetic.source, synthetic.ambiguous, synthetic.record) == (None, None, False, None)
    assert synthetic.note == (
        "rotation: no rotation-check record (synthetic session: there was no video to turn upright)"
    )

    # 값이 빠진 점검 기록(손으로 고친 파일): 가르지 못했다는 표시는 그대로 알리고, 잰 값 대신 사유만 적는다.
    partial = demo.read_raw_rotation({"rotation": 0, "rotation_check": {"ambiguous": True, "reason": "no_face"}})
    assert partial.ambiguous is True
    assert partial.note.startswith(
        "rotation: the extraction's auto check chose 0, ambiguous: yes (reason no_face) WARNING rotation ambiguous: "
    )


def test_cli_raw_prints_the_rotation_record_and_writes_it_to_the_events_json(
    tmp_path: Path,
    config_path: Path,
    repo_root: Path,
    model_v0: Path,
    pipeline_spec: PipelineSpec,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--raw`: 추출이 parquet에 남긴 회전 기록을 한 줄로 출력하고 JSON에 적는다(가르지 못했으면 WARNING, 기록이 없으면 그 사실)."""
    frames = scripted_frames(pipeline_spec, 2_000)
    config = write_cli_config(tmp_path, config_path, repo_root)

    def run(name: str, **metadata: Any) -> tuple[list[str], dict[str, Any]]:
        """추출 결과처럼 메타데이터를 단 parquet에 명령을 돌려 (출력 줄들, JSON의 source)를 돌려준다."""
        parquet = tmp_path / f"{name}.parquet"
        extract.write_raw_frames(frames, parquet, pipeline_spec, {"source": f"videos/{name}.mp4", **metadata})
        out_dir = tmp_path / "out" / name
        argv = ["--config", str(config), "--raw", str(parquet), "--model", str(model_v0), "--out", str(out_dir)]
        assert demo.main(argv) == 0
        captured = capsys.readouterr()
        assert captured.err == ""
        lines = captured.out.splitlines()
        # 회전 줄은 정확히 하나이고, 데이터 표시 줄 다음(둘째 줄)이다.
        assert lines[0] == demo.NOTE_RECORDING and lines[2].startswith("wrote ")
        assert [index for index, line in enumerate(lines) if line.startswith("rotation:")] == [1]
        source = json.loads((out_dir / f"{name}_events.json").read_text(encoding="utf-8"))["source"]
        assert source["rotation_note"] == lines[1]  # 출력한 줄 그대로다
        assert sorted(path.name for path in out_dir.iterdir()) == [f"{name}_events.json", f"{name}_timeline.png"]
        return lines, source

    # 1. 자동 점검이 가르지 못한 영상의 추출 결과: WARNING과 할 일.
    lines, source = run("upside_down", rotation=0, rotation_override=0, rotation_check=UPSIDE_DOWN_RECORD)
    assert lines[1].startswith("rotation: the extraction's auto check chose 0, ambiguous: yes (metadata 0; ")
    assert lines[1].count("WARNING rotation ambiguous") == 1 and lines[1].endswith(RAW_AMBIGUOUS_ADVICE)
    assert (source["rotation"], source["rotation_source"], source["rotation_ambiguous"]) == (0, "auto_check", True)
    assert source["rotation_check"] == UPSIDE_DOWN_RECORD and source["extract_source"] == "videos/upside_down.mp4"

    # 2. 점검이 각도를 정한 결과: 가르지 못했다는 말이 없다.
    lines, source = run("lying", rotation=90, rotation_override=90, rotation_check=LYING_RECORD)
    assert lines[1].startswith("rotation: the extraction's auto check chose 90, ambiguous: no (metadata 0; smallest_roll;")
    assert "rotation ambiguous" not in "\n".join(lines)
    assert (source["rotation"], source["rotation_source"], source["rotation_ambiguous"]) == (90, "auto_check", False)
    assert source["rotation_check"] == LYING_RECORD

    # 3. 수동 값으로 추출한 결과: `--video --rotation 180`이 출력하는 줄과 같은 꼴의 한 줄뿐이고 WARNING이 없다.
    lines, source = run("by_hand", rotation=180, rotation_override=180)
    assert lines[1] == "rotation: 180 (manual value recorded by the extraction; the automatic check was not run)"
    assert lines[1].replace("manual value recorded by the extraction", "given with --rotation") == VIDEO_MANUAL_LINE
    assert "WARNING" not in "\n".join(lines)
    assert (source["rotation"], source["rotation_source"], source["rotation_ambiguous"]) == (180, "manual_override", False)
    assert source["rotation_check"] is None

    # 4. 점검 기록이 없는 결과(영상의 회전 메타데이터만 믿고 추출한 전의 결과): 기록이 없다고 적는다.
    lines, source = run("unchecked", rotation=90, rotation_override=None)
    assert lines[1] == (
        "rotation: 90 from the video's rotation metadata, no rotation-check record (the extraction did not check "
        "the angle; extract the video again to run the check)"
    )
    assert (source["rotation"], source["rotation_source"], source["rotation_ambiguous"]) == (90, "video_metadata", False)
    assert source["rotation_check"] is None

    # 5. 회전이 적혀 있지 않은 파일.
    lines, source = run("bare")
    assert lines[1] == "rotation: no rotation-check record (the file does not say how the picture was turned upright)"
    assert (source["rotation"], source["rotation_source"], source["rotation_ambiguous"]) == (None, None, False)


def test_run_demo_raw_marks_an_ambiguous_rotation_in_the_figure_header(
    tmp_path: Path, model_v0: Path, demo_cfg: ExperimentConfig, pipeline_spec: PipelineSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    """그림의 머리말: 가르지 못한 추출 결과에만 회전 줄이 붙는다(`render_timeline`이 받는 작은 줄을 가로채서 본다)."""
    frames = scripted_frames(pipeline_spec, 2_000)
    seen: list[list[str]] = []
    render = demo.render_timeline

    def spy(run: Any, spec: Any, rules: Any, titles: Any, notes: Any, *rest: Any) -> Any:
        seen.append(list(notes))
        return render(run, spec, rules, titles, notes, *rest)

    monkeypatch.setattr(demo, "render_timeline", spy)
    flagged = tmp_path / "flagged.parquet"
    metadata = {"source": "x.mp4", "rotation": 0, "rotation_override": 0, "rotation_check": UPSIDE_DOWN_RECORD}
    extract.write_raw_frames(frames, flagged, pipeline_spec, metadata)
    run_demo_raw(flagged, model_v0, tmp_path / "out", cfg=demo_cfg)
    decided = tmp_path / "decided.parquet"
    metadata = {"source": "x.mp4", "rotation": 90, "rotation_override": 90, "rotation_check": LYING_RECORD}
    extract.write_raw_frames(frames, decided, pipeline_spec, metadata)
    run_demo_raw(decided, model_v0, tmp_path / "out", cfg=demo_cfg)

    assert [len(notes) for notes in seen] == [4, 3]
    assert seen[0][3] == (
        "video rotation: 0 degrees counter-clockwise (automatic check at extraction, AMBIGUOUS: look at the video "
        "and extract it again with a manual rotation)"
    )
    assert not any("rotation" in note for note in seen[1])


# ---------------------------------------------------------------------------
# 실제 MediaPipe (모델 파일이나 라이브러리가 없으면 건너뛴다)
# ---------------------------------------------------------------------------

VIDEO_W, VIDEO_H = 160, 120
VIDEO_PTS_MS = [0, 33, 67, 100, 133, 167, 200, 233, 267, 300, 333, 367]


def write_plain_video(path: Path) -> None:
    """얼굴이 없는 합성 그림(검은 화면, 잡음, 움직이는 사각형) 12장을 H.264 영상으로 쓴다."""
    time_base = Fraction(1, 1000)
    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=30)
    stream.width, stream.height, stream.pix_fmt = VIDEO_W, VIDEO_H, "yuv420p"
    stream.codec_context.time_base = time_base
    stream.time_base = time_base
    stream.options = {"bf": "0", "crf": "20"}
    for index, pts in enumerate(VIDEO_PTS_MS):
        if index % 3 == 0:
            image = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
        elif index % 3 == 1:
            image = np.random.default_rng(index).integers(0, 256, size=(VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
        else:
            image = np.full((VIDEO_H, VIDEO_W, 3), 128, dtype=np.uint8)
            image[30:60, 20 + 4 * index : 60 + 4 * index] = (200, 60, 60)
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


@pytest.fixture(scope="module")
def real_landmarker_model(repo_root: Path, pipeline_spec: PipelineSpec) -> Path:
    model = repo_root / "models" / "face_landmarker.task"
    if not model.is_file():
        pytest.skip("models/face_landmarker.task is absent")
    try:
        extract.make_landmarker(model, pipeline_spec).close()
    except extract.ExtractError as exc:
        pytest.skip(f"MediaPipe cannot be used here: {exc}")
    return model


@pytest.mark.mediapipe
def test_cli_video_on_a_non_face_video(
    tmp_path: Path,
    config_path: Path,
    repo_root: Path,
    model_v0: Path,
    real_landmarker_model: Path,
    experiment_config: ExperimentConfig,
    capsys: pytest.CaptureFixture[str],
) -> None:
    video = tmp_path / "videos" / "plain.mp4"
    video.parent.mkdir()
    write_plain_video(video)
    config = write_cli_config(tmp_path, config_path, repo_root)
    out_dir = tmp_path / "out"
    argv = ["--config", str(config), "--video", str(video), "--model", str(model_v0), "--out", str(out_dir)]
    assert demo.main(argv) == 0
    captured = capsys.readouterr()
    assert "No model output: the baseline was not completed" in captured.out
    # 회전 자동 점검을 한 번 돌리고 결과를 먼저 출력한다. 얼굴이 없는 영상이라 아무 각도도 확인하지 못했다.
    assert captured.out.startswith("rotation: auto check chose 0, ambiguous: yes (metadata 0; no_face;")

    # 쓰는 것은 PNG와 JSON뿐이다. 영상 폴더에도 아무것도 남기지 않는다(프레임을 저장하지 않는다. PLAN 1.2절의 결정 D13)
    assert sorted(path.name for path in out_dir.iterdir()) == ["plain_events.json", "plain_timeline.png"]
    assert sorted(path.name for path in video.parent.iterdir()) == ["plain.mp4"]
    assert_real_picture(out_dir / "plain_timeline.png", experiment_config)

    report = json.loads((out_dir / "plain_events.json").read_text(encoding="utf-8"))
    assert report["note"] == demo.NOTE_RECORDING
    source = report["source"]
    assert source["kind"] == demo.SOURCE_VIDEO and source["path"] == str(video) and source["synthetic"] is False
    assert (source["n_decoded"], source["n_skipped"], source["rotation"]) == (12, 0, 0)
    assert (source["rotation_source"], source["rotation_ambiguous"]) == ("auto_check", True)
    assert source["display_flip"] is False and source["rotation_check"]["reason"] == "no_face"
    assert source["rotation_check"]["rates"] == {"0": 0.0, "90": 0.0, "180": 0.0, "270": 0.0}
    assert source["mediapipe_version"] == extract.mediapipe_version()
    assert source["mediapipe_model_sha256"] == file_sha256(real_landmarker_model)
    assert report["session"] == {"first_t_ms": 1, "last_t_ms": 368, "duration_ms": 367}
    assert report["counters"]["frames_accepted"] == 12 and report["counters"]["face_frames"] == 0
    assert report["baseline"] is None and report["probabilities"]["n_outputs"] == 0
    # 보정은 시작만 하고 끝나지 않는다
    calibrating = [event["phase"] for event in report["events"] if event["type"] == EVENT_CALIBRATING]
    assert kinds(report["events"])[0] == "calibrating:onset" and calibrating == [PHASE_ONSET]


FACE_IMAGE_ENV = "DROWSY_TEST_FACE_IMAGE"


@pytest.mark.mediapipe
def test_cli_video_turns_a_sideways_face_clip_upright(
    tmp_path: Path,
    config_path: Path,
    repo_root: Path,
    model_v0: Path,
    real_landmarker_model: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """실제 랜드마커와 실제 얼굴: 회전 메타데이터 없이 누워 저장된 영상에서 데모가 회전을 점검해 세우고 그 각도를 알린다.

    `DROWSY_TEST_FACE_IMAGE=/path/to/face.png`(똑바로 선 정면 얼굴 하나)를 주고 돌린다. 저장소에는 얼굴 사진이 없다.
    """
    image_path = os.environ.get(FACE_IMAGE_ENV)
    if not image_path:
        pytest.skip(f"{FACE_IMAGE_ENV} is not set (path of an upright frontal face image you may use)")
    import cv2

    image = cv2.imread(image_path)
    if image is None:
        pytest.fail(f"{FACE_IMAGE_ENV}={image_path} cannot be read as an image")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    scale = min(640 / rgb.shape[1], 480 / rgb.shape[0])
    size = (int(rgb.shape[1] * scale) // 2 * 2, int(rgb.shape[0] * scale) // 2 * 2)
    small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    canvas = np.full((480, 640, 3), 70, dtype=np.uint8)
    x0, y0 = (640 - small.shape[1]) // 2, (480 - small.shape[0]) // 2
    canvas[y0 : y0 + small.shape[0], x0 : x0 + small.shape[1]] = small

    video = tmp_path / "videos" / "lying.mp4"  # 시계 방향 90도로 누워 저장, 회전 메타데이터 없음: 맞는 각도는 90
    video.parent.mkdir()
    time_base = Fraction(1, 1000)
    container = av.open(str(video), "w")
    stream = container.add_stream("libx264", rate=30)
    stream.width, stream.height, stream.pix_fmt = 480, 640, "yuv420p"
    stream.codec_context.time_base = time_base
    stream.time_base = time_base
    stream.options = {"bf": "0", "crf": "16"}
    for index in range(8):
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(np.rot90(canvas, -1)), format="rgb24")
        frame.pts = 100 * index
        frame.time_base = time_base
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    config = write_cli_config(tmp_path, config_path, repo_root)
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    data["extraction"].update(rotation_check_frames=4, rotation_check_interval_ms=200)
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    out_dir = tmp_path / "out"
    argv = ["--config", str(config), "--video", str(video), "--model", str(model_v0), "--out", str(out_dir)]

    assert demo.main(argv) == 0
    first = capsys.readouterr().out.splitlines()[0]
    assert first.startswith("rotation: auto check chose "), first
    report = json.loads((out_dir / "lying_events.json").read_text(encoding="utf-8"))
    source = report["source"]
    check = source["rotation_check"]
    assert (source["rotation_source"], check["n_frames"], check["metadata_rotation"]) == ("auto_check", 4, 0), first
    # 옆으로 누운 채로는 돌리지 않는다: 맞는 각도(90)이거나, 거꾸로 선 얼굴도 검출되는 얼굴이라 가르지 못해 표시한 270이다.
    assert source["rotation"] in (90, 270) and abs(check["rolls"][str(source["rotation"])]) < 45.0, first
    if source["rotation"] != 90 or check["rolls"]["270"] is not None:
        assert source["rotation_ambiguous"] is True and "ambiguous: yes" in first and "WARNING" in first, first
    else:
        assert source["rotation_ambiguous"] is False and "ambiguous: no" in first, first
    assert report["counters"]["frames_accepted"] == report["counters"]["face_frames"] == 8
    # 쓰는 것은 PNG와 JSON뿐이다(얼굴 프레임을 저장하지 않는다).
    assert sorted(path.name for path in out_dir.iterdir()) == ["lying_events.json", "lying_timeline.png"]

    assert demo.main([*argv, "--rotation", "90"]) == 0
    assert capsys.readouterr().out.startswith("rotation: 90 (given with --rotation")
    report = json.loads((out_dir / "lying_events.json").read_text(encoding="utf-8"))
    assert (report["source"]["rotation"], report["source"]["rotation_source"]) == (90, "manual_override")
    assert report["counters"]["face_frames"] == 8


def test_video_path_names_the_fix_when_the_landmarker_model_is_absent(
    tmp_path: Path, model_v0: Path, demo_cfg: ExperimentConfig
) -> None:
    video = tmp_path / "plain.mp4"
    write_plain_video(video)
    cfg = dataclasses.replace(
        demo_cfg, paths=dataclasses.replace(demo_cfg.paths, model_path=tmp_path / "face_landmarker.task")
    )
    with pytest.raises(extract.ExtractError, match="MediaPipe model file not found"):
        demo.run_demo(video, model_v0, tmp_path / "out", cfg=cfg)
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------------------
# 의존성: 선언하지 않은 패키지를 직접 import하지 않는다
# (demo.py가 pyproject.toml에 없는 `google.protobuf`를 직접 import하던 것을 고친 뒤의 회귀 검사다)
# ---------------------------------------------------------------------------

# pyproject.toml의 배포 이름 → import하는 최상위 모듈 이름(이름이 다른 것만)
_IMPORT_NAMES = {"opencv-python-headless": "cv2", "scikit-learn": "sklearn", "pyyaml": "yaml"}


def undeclared_imports(source: str, declared: set[str]) -> list[str]:
    """소스가 import하는 최상위 모듈 중 표준 라이브러리도, 이 패키지도, 선언한 의존성도 아닌 것(오름차순)."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
            found.add(node.module.split(".")[0])
    return sorted(found - set(sys.stdlib_module_names) - declared - {"drowsy", "__future__"})


def test_demo_imports_only_declared_dependencies(repo_root: Path) -> None:
    project = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    names = [requirement.split("==")[0].strip().lower() for requirement in project["dependencies"]]
    declared = {_IMPORT_NAMES.get(name, name) for name in names}
    assert "protobuf" not in names and {"onnx", "onnxruntime", "matplotlib", "av", "numpy"} <= declared

    # 검사가 실제로 잡아내는지: protobuf는 onnx를 따라 설치될 뿐 선언한 의존성이 아니다.
    assert undeclared_imports("from google.protobuf.message import DecodeError\nimport numpy\n", declared) == ["google"]
    assert undeclared_imports("import os.path\nfrom drowsy import spec\nimport onnx.helper\n", declared) == []

    source = (repo_root / "drowsy" / "demo.py").read_text(encoding="utf-8")
    assert undeclared_imports(source, declared) == []
    assert "google.protobuf" not in source
