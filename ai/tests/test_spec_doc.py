"""docs/SPEC.md가 코드와 설정에서 벗어나지 않게 붙잡는 가벼운 일치 검사 (PLAN 10.3절).

문서가 인용하는 값 가운데 기계로 읽을 수 있는 것만 뽑아 `config/feature_spec.json`, `config/rules.yaml`,
`drowsy.types`, 내보내기 계약(`drowsy.export_onnx`), 테스트 벡터, 깜빡임 길이 모사(`drowsy.signal_check`)와 비교한다.
문장과 의사 코드는 검사하지 않는다(그쪽은 tests/vectors가 참조 구현 자체를 고정한다).

문서 쪽 약속 (SPEC.md 0.3절)
- 검사 대상 표는 `<!-- check:이름 -->` 주석 다음에 오는 Markdown 표다. 같은 이름의 표가 여럿이면 행을 합친다.
- 본문에서 코드 조각 바로 뒤 괄호에 숫자를 적은 것(`` `clock.max_hold_ms`(200) ``)은 그 경로의 설정값과 비교한다.

가정: 상수 표의 첫 칸은 경로, 둘째 칸은 값이다. 값은 설정의 자료형에 맞춰 읽는다(정수는 소수점 없이, 실수는
  소수점을 붙여서, 불리언은 true/false, 목록은 쉼표로 나눠서 적는다). 표의 칸 안에는 `|`를 쓰지 않는다.
가정: 본문의 괄호 인용에서 점이 없는 이름(`hold_ms`)은 두 설정에서 같은 이름의 값을 모두 찾아 전부와 비교한다.
  값이 서로 다른 이름을 점 없이 인용하면 실패하므로, 그런 이름은 문서에 전체 경로로 적어야 한다.
가정: `pipeline` 블록과 규칙 설정의 값은 빠짐없이 표에 있어야 한다(랜드마크·블렌드셰이프·특징 이름·이벤트 정책은
  각자의 표가 맡는다). 설정에 값을 더하고 문서를 고치지 않으면 여기서 드러난다.
가정: 17.3절의 모사 표는 `signal_check.simulate_blinks`를 시드 0으로 다시 돌린 값과 비교한다. 문서는 반올림해 적으므로
  적힌 자릿수의 절반까지 허용한다. 검출된 깜빡임이 없어 값이 없는 칸은 `-`로 적는다.
가정: 본문에서 코드 조각으로 인용한 이름 가운데 `signals_`, `baseline_`, `runs_`, `steps_`, `rules_`, `e2e_`로 시작하는
  것은 벡터 이름으로 본다(벡터가 아닌 식별자 둘은 테스트에 적어 둔다). 그런 이름은 모두 실제 벡터여야 한다.
가정: 20장(소유자의 결정을 기다리는 동작)은 "## 20."으로 시작하는 줄부터 문서 끝까지이고, 그 안의 질문은 줄 맨 앞의
  "N. **"으로 시작한다. 본문의 "20장 N번"은 그 번호를 가리킨다. 벡터의 수와 합계 크기는 20장에
  "JSON 파일 N개, 합계 B bytes" 모양으로 적는다(B는 쉼표로 끊어 적는다). 결정된 질문의 번호는 비워 두고 다른 번호는
  바꾸지 않는다(`RETIRED_SECTION_20_ITEMS`). 결정 기록은 20장 머리에 "<날짜>에 결정되어 이 목록에서 빠진 것"으로 적는다.
가정: 20장의 항목 하나는 "N. **"으로 시작하는 줄부터 다음 항목 앞까지이고(마지막 항목은 맺는 문단 앞의 빈 줄까지), 대안은
  "제안된 대안"이라는 말로 적는다. 9번은 대안을 넣은 참조 구현으로 돌려 본 벡터의 수를 "N개가 모두 그대로 통과"로 적는다.
가정: README.md가 docs/SPEC.md를 이식 명세로 가리키는지는 "한 줄에 `docs/SPEC.md`와 '이식 명세'가 함께 있는가"로 본다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from drowsy import spec as S
from drowsy import types as T

Table = tuple[list[str], list[list[str]]]  # (머리 행, 본문 행들). 칸은 앞뒤 공백을 뗀 원문이다

_MARKER = re.compile(r"^<!-- check:([a-z0-9-]+) -->$")
_SEPARATOR_CELL = re.compile(r"^:?-{3,}:?$")
_INLINE_QUOTE = re.compile(r"`([A-Za-z_][A-Za-z0-9_.]*)`\((-?\d+(?:\.\d+)?)")
_WINDOWED_NAME = re.compile(r"`([a-z][a-z0-9_]*_(?:60|10))`")
_VECTOR_NAME = re.compile(r"`((?:signals|baseline|runs|steps|rules|e2e)_[a-z0-9_]+)`")
# 기준선 묶음에는 `closure_`로 시작하는 벡터가 셋 있다. 이 접두어는 필드 이름(`closure_valid`, `closure_sum` 등)과 겹치므로
# 위의 정규식에 넣지 않고, 벡터 목록에 실제로 있는 이름일 때만 벡터로 센다(`_quoted_vector_names`).
_CLOSURE_NAME = re.compile(r"`(closure_[a-z0-9_]+)`")
_VECTOR_COUNT = re.compile(r"지금 (\d+)개다")
_SECTION_20 = re.compile(r"^## 20\. ", re.MULTILINE)
_SECTION_20_ITEM = re.compile(r"^(\d+)\. \*\*", re.MULTILINE)
_SECTION_20_REFERENCE = re.compile(r"20장 (\d+)번")
_VECTOR_SIZE = re.compile(r"JSON 파일 (\d+)개, 합계 ([\d,]+) bytes")
_OVERRIDE_SENTENCE = re.compile(r"`config_overrides`가 있는 벡터는 (\S+)이다: (.+?)\n\n", re.DOTALL)
_KOREAN_COUNTS = {"하나": 1, "둘": 2, "셋": 3, "넷": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10}

# 20장에서 결정되어 비워 둔 번호. 1번(반쯤 뜬 눈꺼풀이 감김을 끝내지 않는 것)은 2026-11-14에 소유자가 "지금대로 둠"으로
# 결정했다. 다른 항목은 번호를 바꾸지 않는다.
RETIRED_SECTION_20_ITEMS: frozenset[int] = frozenset({1})
# 그 결정으로 확정된 동작을 고정하는 벡터(문서가 "결정을 기다리는 질문에 걸린 벡터"로 적던 것)
HOVER_DECISION_VECTORS: tuple[str, ...] = (
    "runs_lingering_lids",
    "runs_multi_span",
    "runs_multi_span_truncated",
    "steps_multi_span_runs",
    "rules_closure_dwell_below_1s",
    "rules_closure_dwell_reaches_1s",
)

# 2026-11-14의 첫 개정에서 더한 벡터 25개와, 같은 날 두 번째 검증 뒤에 더한 벡터 13개. 문서가 모두 가리켜야 한다.
FIRST_REVISION_VECTORS: tuple[str, ...] = (
    "runs_velocity_after_gap",
    "runs_rounding_large_t0",
    "runs_multi_span",
    "runs_multi_span_truncated",
    "steps_multi_span_runs",
    "steps_closure_across_gap",
    "steps_zero_division",
    "rules_closure_dwell_below_1s",
    "rules_closure_dwell_reaches_1s",
    "rules_closure_three_spans",
    "rules_alert_deferred_cooldown",
    "rules_alert_blocked_microsleep_sleep",
    "rules_drowsy_perclos_exact",
    "rules_drowsy_model_stall",
    "rules_gap_over_10s",
    "rules_gap_exactly_10s",
    "rules_gap_over_10s_drowsy",
    "rules_gap_unable_boundary",
    "rules_unable_sparse_frames",
    "rules_gaze_credit_window_missed",
    "rules_gaze_credit_boundary",
    "rules_gaze_debounce_250ms",
    "rules_forward_zone_edges",
    "rules_vats_reset_enabled",
    "rules_alert_priority",
)
SECOND_VERIFICATION_VECTORS: tuple[str, ...] = (
    "runs_two_humps",
    "runs_one_sample_truncated",
    "steps_gap_exactly_10s",
    "rules_gap_over_10s_candidate",
    "rules_gaze_candidate_dropped",
    "rules_forward_zone_eye_edges",
    "rules_gaze_unable_pauses_timer",
    "rules_drowsy_either",
    "rules_gap_over_10s_drowsy_model",
    "rules_alert_pending_survives_end",
    "rules_closure_after_dropped_run",
    "rules_gaze_glance_and_credit_times",
    "rules_calibrating_large_t0",
)
# 세 번째 검증 뒤에 더한 벡터 8개(검증이 짚은 일곱과 그 정비에서 찾은 하나).
THIRD_VERIFICATION_VECTORS: tuple[str, ...] = (
    "runs_two_rises_before_peak",
    "rules_vats_window_edge",
    "rules_vats_confidence_face_lost",
    "rules_gaze_one_frame_glances",
    "rules_microsleep_confidence_face_lost",
    "rules_alert_pending_survives_onset",
    "rules_unable_small_t0",
    "rules_drowsy_off_step_boundary",
)
# 네 번째 검증 뒤에 더한 벡터 13개(신호, 기준선, 스텝 특징 모듈의 한 줄짜리 오류 가운데 벡터 89개를 통과하던 것).
FOURTH_VERIFICATION_VECTORS: tuple[str, ...] = (
    "signals_iris_tilted_eyes",
    "baseline_odd_sample_count",
    "baseline_burst_frames",
    "baseline_exactly_300_frames",
    "closure_blink_span_floor",
    "steps_yawn_edges",
    "steps_nod_edges",
    "steps_nod_slow_mean",
    "steps_nod_after_long_gap",
    "steps_two_closures_in_one_step",
    "steps_closure_above_cap",
    "steps_closure_exactly_500ms",
    "steps_pitch_std_at_5000ms",
)
# 16.1절의 한 줄 바꾸기 점검 표가 싣는 모듈(이식 대상 가운데 계산이 있는 여섯)과 묶음
MUTATION_CHECK_MODULES: tuple[str, ...] = ("signals.py", "calibrate.py", "blink.py", "step_features.py", "rules.py", "pipeline.py")
MUTATION_CHECK_SETS: tuple[str, ...] = ("기본", "확장")

EXPECTED_MARKERS: frozenset[str] = frozenset(
    {
        "hashes",
        "enums",
        "landmarks",
        "blendshape-names",
        "blendshapes",
        "pipeline",
        "rules",
        "features",
        "events",
        "alert-priority",
        "onnx",
        "feature-spec",
        "metadata",
        "operators",
        "tolerances",
        "blink-sim-conditions",
        "blink-sim-durations",
        "blink-sim-ratios",
        "mutation-check",
    }
)

ENUM_NAMES: tuple[str, ...] = (
    "EVENT_TYPES",
    "PHASES",
    "SOURCES",
    "BASELINE_STATUSES",
    "CLASS_ORDER",
    "DROWSY_SOURCES",
    "EYE_LOOK_ORDER",
)

POLICY_COLUMNS: tuple[str, ...] = ("severity", "cooldown_ms", "repeat_ms", "min_speed_kmh")


# ---------------------------------------------------------------------------
# 문서 읽기
# ---------------------------------------------------------------------------


def _split_row(line: str) -> list[str]:
    body = line.strip()
    assert body.startswith("|") and body.endswith("|"), f"not a table row: {line!r}"
    return [cell.strip() for cell in body[1:-1].split("|")]


def _parse_tables(text: str) -> dict[str, list[Table]]:
    """`<!-- check:이름 -->` 주석마다 그 다음의 표를 읽는다."""
    lines = text.splitlines()
    tables: dict[str, list[Table]] = {}
    for index, line in enumerate(lines):
        match = _MARKER.match(line.strip())
        if match is None:
            continue
        cursor = index + 1
        while cursor < len(lines) and lines[cursor].strip() == "":
            cursor += 1
        rows: list[list[str]] = []
        while cursor < len(lines) and lines[cursor].lstrip().startswith("|"):
            rows.append(_split_row(lines[cursor]))
            cursor += 1
        assert len(rows) >= 3, f"marker {match.group(1)!r} at line {index + 1} is not followed by a table"
        header, separator, body = rows[0], rows[1], rows[2:]
        assert all(_SEPARATOR_CELL.match(cell) for cell in separator), f"bad separator row after line {index + 1}"
        for row in body:
            assert len(row) == len(header), f"row has {len(row)} cells, header has {len(header)}: {row}"
        tables.setdefault(match.group(1), []).append((header, body))
    return tables


def _plain(cell: str) -> str:
    """칸에서 코드 표시(백틱)와 공백을 뗀 값."""
    return cell.replace("`", "").strip()


def _items(cell: str) -> list[str]:
    """쉼표로 나눈 목록."""
    return [_plain(part) for part in cell.split(",") if _plain(part) != ""]


def _column(header: list[str], name: str) -> int:
    names = [_plain(cell) for cell in header]
    assert name in names, f"column {name!r} not in {names}"
    return names.index(name)


def _key_value_rows(tables: list[Table]) -> dict[str, str]:
    """첫 칸이 열쇠, 둘째 칸이 값인 표들을 하나로 합친다. 같은 열쇠가 두 번 나오면 실패한다."""
    merged: dict[str, str] = {}
    for _header, body in tables:
        for row in body:
            key = _plain(row[0])
            assert key not in merged, f"{key!r} appears twice in the marked tables"
            merged[key] = row[1]
    return merged


def _leaves(node: Any, prefix: str = "") -> dict[str, Any]:
    """중첩 dict를 `a.b.c` -> 값으로 편다. 목록은 값으로 둔다."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            out.update(_leaves(value, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: node}


def _matches(cell: str, value: Any) -> bool:
    """표의 칸이 설정값과 같은가. 자료형까지 맞아야 한다."""
    if isinstance(value, (list, tuple)):
        parts = _items(cell)
        return len(parts) == len(value) and all(_matches(part, item) for part, item in zip(parts, value))
    text = _plain(cell)
    if isinstance(value, bool):
        return text in ("true", "false") and (text == "true") == value
    if isinstance(value, int):
        return re.fullmatch(r"-?\d+", text) is not None and int(text) == value
    if isinstance(value, float):
        return re.fullmatch(r"-?\d+\.\d+", text) is not None and float(text) == value
    return text == value


def _check_constants(rows: dict[str, str], expected: dict[str, Any]) -> None:
    assert sorted(set(rows) - set(expected)) == [], "the document quotes paths that are not in the config"
    assert sorted(set(expected) - set(rows)) == [], "config values missing from the document's tables"
    wrong = {path: (rows[path], expected[path]) for path in expected if not _matches(rows[path], expected[path])}
    assert wrong == {}, f"document value differs from config (document, config): {wrong}"


@pytest.fixture(scope="module")
def spec_text(repo_root: Path) -> str:
    return (repo_root / "docs" / "SPEC.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tables(spec_text: str) -> dict[str, list[Table]]:
    return _parse_tables(spec_text)


@pytest.fixture(scope="module")
def pipeline_leaves(pipeline_spec: S.PipelineSpec) -> dict[str, Any]:
    return _leaves(S.pipeline_to_dict(pipeline_spec))


@pytest.fixture(scope="module")
def rules_leaves(rules_config: S.RulesConfig) -> dict[str, Any]:
    return _leaves(S.rules_to_dict(rules_config))


@pytest.fixture(scope="module")
def root_leaves(feature_spec: S.FeatureSpec) -> dict[str, Any]:
    """feature_spec의 맨 위 값들(`pipeline` 블록과 특징 목록은 따로 본다)."""
    raw = S.feature_spec_to_dict(feature_spec)
    return _leaves({key: value for key, value in raw.items() if key not in ("pipeline", "features")})


# ---------------------------------------------------------------------------
# 표
# ---------------------------------------------------------------------------


def test_document_has_exactly_the_expected_marked_tables(tables: dict[str, list[Table]]) -> None:
    # 주석 이름을 잘못 적으면 그 표는 조용히 검사에서 빠진다. 이름 집합을 고정해 막는다.
    assert set(tables) == EXPECTED_MARKERS


def test_hashes_match_the_loaded_spec_and_rules(
    tables: dict[str, list[Table]], feature_spec: S.FeatureSpec, rules_config: S.RulesConfig, repo_root: Path
) -> None:
    rows = _key_value_rows(tables["hashes"])
    expected = {
        "spec_version": feature_spec.spec_version,
        "pipeline_hash": feature_spec.pipeline_hash,
        "rules_hash": S.rules_hash(rules_config),
        "mediapipe_model_sha256": feature_spec.pipeline.mediapipe.model_sha256,
    }
    _check_constants(rows, expected)
    # 문서는 rules_hash를 "export/rules.json 바이트의 sha256"이라고 적는다. 내보낸 파일이 있으면 그 말이 맞는지 본다.
    exported_rules = repo_root / "export" / "rules.json"
    if exported_rules.is_file():
        assert S.file_sha256(exported_rules) == _plain(rows["rules_hash"])


def test_string_constants_match_types(tables: dict[str, list[Table]]) -> None:
    rows = _key_value_rows(tables["enums"])
    assert tuple(rows) == ENUM_NAMES
    for name in ENUM_NAMES:
        assert tuple(_items(rows[name])) == getattr(T, name), name


def test_ear_index_order_matches_spec(tables: dict[str, list[Table]], pipeline_spec: S.PipelineSpec) -> None:
    rows = _key_value_rows(tables["landmarks"])
    # p1..p6의 순서가 곧 EAR 식의 점 순서다. 집합이 아니라 순서를 비교한다.
    assert [int(item) for item in _items(rows["ear_r"])] == list(pipeline_spec.landmarks.ear_r)
    assert [int(item) for item in _items(rows["ear_l"])] == list(pipeline_spec.landmarks.ear_l)


def test_landmark_table_matches_and_covers_spec(tables: dict[str, list[Table]], pipeline_leaves: dict[str, Any]) -> None:
    rows = _key_value_rows(tables["landmarks"])
    prefix = "landmarks."
    expected = {path[len(prefix) :]: value for path, value in pipeline_leaves.items() if path.startswith(prefix)}
    _check_constants(rows, expected)


def test_blendshape_tables_match_spec(tables: dict[str, list[Table]], pipeline_spec: S.PipelineSpec) -> None:
    names = pipeline_spec.blendshapes
    rows = _key_value_rows(tables["blendshape-names"])
    expected = {
        "eye_blink_l": names.eye_blink_l,
        "eye_blink_r": names.eye_blink_r,
        "eye_look": names.eye_look,
        "jaw_open": names.jaw_open,
    }
    _check_constants(rows, expected)

    # 52개 표는 한 행에 (인덱스, 이름)이 네 쌍씩 있다.
    by_index: dict[int, str] = {}
    for _header, body in tables["blendshapes"]:
        for row in body:
            assert len(row) % 2 == 0
            for position in range(0, len(row), 2):
                index = int(_plain(row[position]))
                assert index not in by_index, f"blendshape index {index} appears twice"
                by_index[index] = _plain(row[position + 1])
    assert sorted(by_index) == list(range(len(names.order)))
    assert tuple(by_index[index] for index in sorted(by_index)) == names.order


def test_pipeline_constant_tables_match_and_cover_spec(
    tables: dict[str, list[Table]], pipeline_leaves: dict[str, Any]
) -> None:
    rows = _key_value_rows(tables["pipeline"])
    expected = {
        path: value
        for path, value in pipeline_leaves.items()
        if not path.startswith(("landmarks.", "blendshapes.")) and path != "step_features.names"
    }
    _check_constants(rows, expected)


def test_rules_constant_tables_match_and_cover_rules(
    tables: dict[str, list[Table]], rules_leaves: dict[str, Any]
) -> None:
    rows = _key_value_rows(tables["rules"])
    expected = {
        path: value
        for path, value in rules_leaves.items()
        if not path.startswith("events.") and path != "alert_priority"
    }
    _check_constants(rows, expected)


def test_feature_table_matches_spec(tables: dict[str, list[Table]], feature_spec: S.FeatureSpec) -> None:
    assert len(tables["features"]) == 1
    header, body = tables["features"][0]
    columns = [_column(header, name) for name in ("#", "이름", "구간(s)", "단위", "변환", "정의 불가 값")]
    assert len(body) == T.N_FEATURES == len(feature_spec.features)
    for position, (row, feature) in enumerate(zip(body, feature_spec.features)):
        index, name, window, unit, transform, undefined = (_plain(row[column]) for column in columns)
        assert int(index) == position == feature.index
        assert name == feature.name == T.FEATURE_NAMES[position]
        assert int(window) == feature.window_sec
        assert unit == feature.unit
        assert transform == feature.transform
        assert float(undefined) == feature.undefined_value


def test_every_quoted_windowed_name_is_a_feature_or_spec_constant(
    spec_text: str, pipeline_spec: S.PipelineSpec
) -> None:
    # `_60`, `_10`으로 끝나는 이름을 코드 조각으로 인용했다면 특징 이름이거나 step_features의 상수 이름이어야 한다.
    allowed = set(T.FEATURE_NAMES) | set(S.to_plain(pipeline_spec.step_features))
    quoted = set(_WINDOWED_NAME.findall(spec_text))
    assert set(T.FEATURE_NAMES) <= quoted, "every feature name must be quoted in the document"
    assert sorted(quoted - allowed) == []


def test_event_tables_match_types_and_policies(tables: dict[str, list[Table]], rules_config: S.RulesConfig) -> None:
    assert len(tables["events"]) == 2  # 12.8절의 이벤트 표와 13장의 정책 표
    checked_columns: set[str] = set()
    for header, body in tables["events"]:
        names = [_plain(cell) for cell in header]
        type_column = names.index("type")
        # 두 표 모두 모든 종류를 한 번씩, 정해진 순서로 싣는다.
        assert tuple(_plain(row[type_column]) for row in body) == T.EVENT_TYPES
        for row in body:
            policy = rules_config.events.get(_plain(row[type_column]))
            for column in POLICY_COLUMNS:
                if column in names:
                    assert _matches(row[names.index(column)], getattr(policy, column)), (row[type_column], column)
                    checked_columns.add(column)
            if "source" in names:
                sources = _items(row[names.index("source")])
                assert sources and set(sources) <= set(T.SOURCES), row
            if "phase" in names:
                assert tuple(_items(row[names.index("phase")])) == T.PHASES, row
    assert checked_columns == set(POLICY_COLUMNS)


def test_alert_priority_table_matches_rules(tables: dict[str, list[Table]], rules_config: S.RulesConfig) -> None:
    assert len(tables["alert-priority"]) == 1
    _header, body = tables["alert-priority"][0]
    assert [int(_plain(row[0])) for row in body] == list(range(1, len(body) + 1))
    assert tuple(_plain(row[1]) for row in body) == rules_config.alert_priority


def test_onnx_contract_tables_match_export_module(
    tables: dict[str, list[Table]], feature_spec: S.FeatureSpec, root_leaves: dict[str, Any]
) -> None:
    from drowsy import export_onnx  # torch와 onnx를 올리므로 이 테스트 안에서만 import한다

    rows = _key_value_rows(tables["onnx"])
    expected: dict[str, Any] = {
        "opset": export_onnx.OPSET,
        "ir_version": export_onnx.IR_VERSION,
        "input_name": export_onnx.INPUT_NAME,
        "input_shape": [1, feature_spec.window_steps, feature_spec.n_features],
        "output_name": export_onnx.OUTPUT_NAME,
        "output_shape": [1, len(feature_spec.class_order)],
        "max_file_bytes": export_onnx.MAX_FILE_BYTES,
    }
    _check_constants(rows, expected)

    # feature_spec 맨 위 값 가운데 문서가 인용한 것(버전마다 바뀌는 온도와 spec_version은 이 표에 없다).
    quoted = _key_value_rows(tables["feature-spec"])
    assert set(quoted) <= set(root_leaves)
    assert {"step_sec", "window_steps", "class_order", "standardization.z_clip"} <= set(quoted)
    for path, cell in quoted.items():
        assert _matches(cell, root_leaves[path]), path


def test_metadata_table_matches_export_module(tables: dict[str, list[Table]], feature_spec: S.FeatureSpec) -> None:
    from drowsy import export_onnx

    rows = _key_value_rows(tables["metadata"])
    assert tuple(rows) == export_onnx.METADATA_KEYS
    built = export_onnx.build_metadata(feature_spec, "hash", "0", export_onnx.TRAINED_ON_NONE, "commit")
    # spec에서 나오는 고정 값 넷은 문서에 값 그대로 적혀 있다.
    for key in ("step_sec", "window_steps", "class_order", "standardization"):
        assert _plain(rows[key]) == built[key], key
    assert export_onnx.TRAINED_ON_NONE in _plain(rows["trained_on"])


def test_operator_table_matches_export_module(tables: dict[str, list[Table]], repo_root: Path) -> None:
    from drowsy import export_onnx

    rows = _key_value_rows(tables["operators"])
    assert set(rows) == {"default", "int8", "dynamo", "v0"}
    assert set(_items(rows["default"])) == set(export_onnx.ALLOWED_OPS)
    assert set(_items(rows["int8"])) == set(export_onnx.ALLOWED_OPS_INT8)
    assert set(_items(rows["dynamo"])) == set(export_onnx.ALLOWED_OPS_DYNAMO)
    v0_ops = set(_items(rows["v0"]))
    assert v0_ops <= set(export_onnx.ALLOWED_OPS)
    v0_path = repo_root / "export" / "drowsiness_v0.onnx"
    if v0_path.is_file():
        assert v0_ops == set(export_onnx.graph_operators(v0_path))


def test_tolerance_table_matches_vector_module(tables: dict[str, list[Table]]) -> None:
    from drowsy import vectors

    assert len(tables["tolerances"]) == 1
    header, body = tables["tolerances"][0]
    keys = [_plain(cell) for cell in header[1:]]
    quoted: dict[str, dict[str, float]] = {}
    for row in body:
        quoted[_plain(row[0])] = {key: float(_plain(cell)) for key, cell in zip(keys, row[1:]) if _plain(cell) != ""}
    assert quoted == {section: dict(values) for section, values in vectors.TOLERANCES.items()}


def _same_rounded(cell: str, value: float) -> bool:
    """문서의 칸(반올림한 십진 표기, 값이 없으면 `-`)이 계산값과 같은가."""
    text = _plain(cell)
    if text == "-":
        return value != value  # NaN
    assert re.fullmatch(r"-?\d+\.\d+", text) is not None, f"not a decimal: {cell!r}"
    decimals = len(text.split(".")[1])
    return abs(float(text) - value) <= 0.5 * 10.0 ** (-decimals) + 1e-9


def test_blink_simulation_tables_match_a_rerun(tables: dict[str, list[Table]], pipeline_spec: S.PipelineSpec) -> None:
    from drowsy import signal_check  # pandas를 올리므로 이 테스트 안에서만 import한다

    table = signal_check.simulate_blinks(pipeline_spec.run_detector, seed=0)
    errors = (
        ("치우침 (ms)", "bias_ms"),
        ("중앙 절대 오차 (ms)", "median_abs_error_ms"),
        ("절대 오차 95% (ms)", "p95_abs_error_ms"),
    )

    def check_error_rows(marker: str, level: str, with_duration: bool) -> set[tuple[Any, ...]]:
        assert len(tables[marker]) == 1
        header, body = tables[marker][0]
        names = [_plain(cell) for cell in header]
        seen: set[tuple[Any, ...]] = set()
        for row in body:
            condition = _plain(row[names.index("조건")])
            stamps = _plain(row[names.index("시각 방식")])
            duration = float(_plain(row[names.index("길이 (ms)")])) if with_duration else None
            computed = signal_check._sim_row(table, level, condition, stamps, duration)
            key = (condition, stamps, duration)
            assert key not in seen, key
            seen.add(key)
            assert int(_plain(row[names.index("시행")])) == int(computed["n_trials"]), key
            assert int(_plain(row[names.index("검출")])) == int(computed["n_detected"]), key
            for column, field in errors:
                assert _same_rounded(row[names.index(column)], float(computed[field])), (key, column)
        return seen

    # 조건별 표는 조건 셋과 시각 방식 둘을 빠짐없이 싣는다.
    conditions = {condition.name for condition in signal_check.SIM_CONDITIONS}
    stamps = {signal_check.STAMPS_INTERPOLATED, signal_check.STAMPS_FRAME_TIME}
    seen = check_error_rows("blink-sim-conditions", signal_check.LEVEL_CONDITION, with_duration=False)
    assert seen == {(condition, stamp, None) for condition in conditions for stamp in stamps}

    # 길이별 표는 15 fps의 모든 길이를 싣는다.
    seen = check_error_rows("blink-sim-durations", signal_check.LEVEL_DURATION, with_duration=True)
    assert seen == {("15fps", stamp, duration) for stamp in stamps for duration in signal_check.SIM_DURATIONS_MS}

    assert len(tables["blink-sim-ratios"]) == 1
    header, body = tables["blink-sim-ratios"][0]
    names = [_plain(cell) for cell in header]
    ratios = (
        ("진폭의 비", "amplitude_ratio_mean"),
        ("감는 속도의 비", "closing_velocity_ratio_mean"),
        ("뜨는 속도의 비", "opening_velocity_ratio_mean"),
    )
    assert {_plain(row[names.index("조건")]) for row in body} == conditions and len(body) == len(conditions)
    for row in body:
        condition = _plain(row[names.index("조건")])
        computed = signal_check._sim_row(
            table, signal_check.LEVEL_CONDITION, condition, signal_check.STAMPS_INTERPOLATED
        )
        for column, field in ratios:
            assert _same_rounded(row[names.index(column)], float(computed[field])), (condition, column)
        assert int(_plain(row[names.index("속도가 유효한 깜빡임")])) == int(computed["n_velocity_valid"]), condition
        assert int(_plain(row[names.index("검출된 깜빡임")])) == int(computed["n_detected"]), condition


def test_blink_simulation_check_catches_a_wrong_value() -> None:
    assert _same_rounded("-3.6", -3.6421) and not _same_rounded("-3.6", -3.66)
    assert _same_rounded("0.752", 0.75249) and not _same_rounded("0.752", 0.7526)
    assert _same_rounded("-", float("nan")) and not _same_rounded("-", 0.0) and not _same_rounded("1.0", float("nan"))


def _quoted_vector_names(text: str) -> set[str]:
    """글이 백틱으로 인용한 벡터 이름. `closure_`로 시작하는 이름은 벡터 목록에 있는 것만 센다."""
    from drowsy import vectors

    names = set(_VECTOR_NAME.findall(text))
    return names | (set(_CLOSURE_NAME.findall(text)) & set(vectors.VECTOR_NAMES))


def test_quoted_vector_names_exist_and_the_count_is_current(spec_text: str) -> None:
    from drowsy import vectors

    quoted = _quoted_vector_names(spec_text)
    # 같은 접두어로 시작하지만 벡터가 아닌 식별자(세는 값, 해시 이름)는 여기 적는다.
    not_vectors = {"steps_emitted", "rules_hash"}
    unknown = sorted(quoted - set(vectors.VECTOR_NAMES) - not_vectors)
    assert unknown == [], f"the document quotes vector names that do not exist: {unknown}"
    assert len(quoted & set(vectors.VECTOR_NAMES)) >= 40  # 정규식이 문서의 표기와 어긋나 아무것도 못 찾는 일을 막는다
    # 2026-11-14의 개정과 세 번의 검증 뒤에 더한 벡터는 모두 본문에서 가리킨다. 네 목록은 지금의 벡터 목록에 들어 있어야 한다.
    assert len(FIRST_REVISION_VECTORS) == 25 and len(SECOND_VERIFICATION_VECTORS) == 13
    assert len(THIRD_VERIFICATION_VECTORS) == 8 and len(FOURTH_VERIFICATION_VECTORS) == 13
    added = set(FIRST_REVISION_VECTORS) | set(SECOND_VERIFICATION_VECTORS) | set(THIRD_VERIFICATION_VECTORS)
    added |= set(FOURTH_VERIFICATION_VECTORS)
    assert len(added) == 59 and added <= set(vectors.VECTOR_NAMES)
    assert sorted(added - quoted) == [], "the document must point at every vector added in the four revisions"
    # 문서 16.1절: 처음 43개, 첫 개정 뒤 68개, 두 번째 검증 뒤 81개, 세 번째 검증 뒤 89개, 네 번째 검증 뒤 102개
    assert len(vectors.VECTOR_NAMES) == 43 + len(added) == 102
    # 16.1절의 세 번째 표는 그 여덟을, 네 번째 표는 그 열셋을 모두 싣는다(다른 벡터는 싣지 않는다).
    fourth_start = spec_text.index("네 번째 검증 뒤에 더한 벡터 13개")
    third_table = spec_text[spec_text.index("세 번째 검증 뒤에 더한 벡터 8개") : fourth_start]
    assert set(_VECTOR_NAME.findall(third_table)) == set(THIRD_VERIFICATION_VECTORS)
    fourth_table = spec_text[fourth_start : spec_text.index("`config_overrides`가 있는 벡터는")]
    assert _quoted_vector_names(fourth_table) == set(FOURTH_VERIFICATION_VECTORS)
    rows = [line for line in fourth_table.splitlines() if line.startswith("| ") and not line.startswith("| 고정하는 것")]
    for name in FOURTH_VERIFICATION_VECTORS:  # 벡터마다 줄이 하나 이상 있고, 줄마다 틀린 구현의 값과 옳은 값을 적는다
        assert any(row.rstrip().endswith(f"`{name}` |") for row in rows), name
    assert all(len(_split_row(row)) == 3 and "(" in _split_row(row)[1] for row in rows)
    counts = {int(number) for number in _VECTOR_COUNT.findall(spec_text)}
    assert counts == {len(vectors.VECTOR_NAMES)}


def test_vector_count_and_total_size_quoted_in_section_20_are_current(spec_text: str, repo_root: Path) -> None:
    # 20장은 벡터의 수와 합계 크기를 "다시 만들었을 때 잰 값"으로 적는다. 벡터를 다시 만들면 여기서 드러난다.
    section = spec_text[_section_20_start(spec_text) :]
    quoted = _VECTOR_SIZE.findall(section)
    assert len(quoted) >= 1, "section 20 must state the number of vector files and their total size"
    files = sorted((repo_root / "tests" / "vectors").glob("*.json"))
    current = (len(files), sum(path.stat().st_size for path in files))
    # 첫 인용이 지금의 값이다. 그 뒤의 인용은 지난 개정의 값(달라야 한다)이다.
    count, size = quoted[0]
    assert (int(count), int(size.replace(",", ""))) == current
    assert re.fullmatch(r"\d{1,3}(,\d{3})*", size) is not None, "write the size with thousands separators"
    for older_count, older_size in quoted[1:]:
        assert int(older_count) < current[0] and int(older_size.replace(",", "")) < current[1]


def test_config_override_sentence_names_the_vectors_that_override(spec_text: str, repo_root: Path) -> None:
    from drowsy import vectors

    match = _OVERRIDE_SENTENCE.search(spec_text)
    assert match is not None, "the document must list the vectors that carry config_overrides"
    named = set(_VECTOR_NAME.findall(match.group(2)))
    stored = {
        path.stem
        for path in vectors.vector_files(repo_root / "tests" / "vectors")
        if vectors.read_vector(path)["config_overrides"]
    }
    assert named == stored
    assert _KOREAN_COUNTS[match.group(1)] == len(stored)


# ---------------------------------------------------------------------------
# 20장의 번호와 README
# ---------------------------------------------------------------------------


def _section_20_start(spec_text: str) -> int:
    matches = list(_SECTION_20.finditer(spec_text))
    assert len(matches) == 1, "the document must have exactly one section 20"
    return matches[0].start()


def _first_section_20_item(section: str) -> int:
    """20장(`section`)에서 첫 번호 항목이 시작하는 위치. 그 앞이 장의 머리(결정 기록과 정비에서 정한 것)다."""
    match = _SECTION_20_ITEM.search(section)
    assert match is not None, "section 20 must have numbered items"
    return match.start()


def test_section_20_items_are_numbered_and_every_reference_points_at_one(spec_text: str) -> None:
    start = _section_20_start(spec_text)
    body, section = spec_text[:start], spec_text[start:]
    assert "## 21." not in section  # 20장이 문서의 마지막 장이라는 가정을 붙잡는다
    numbers = [int(number) for number in _SECTION_20_ITEM.findall(section)]
    # 결정된 번호는 비워 두고 나머지는 번호를 바꾸지 않는다: 1부터 마지막 번호까지에서 비운 번호만 빠진다.
    assert numbers == [n for n in range(1, max(numbers) + 1) if n not in RETIRED_SECTION_20_ITEMS]
    assert len(numbers) >= 10
    referenced = {int(number) for number in _SECTION_20_REFERENCE.findall(spec_text)}
    assert len(_SECTION_20_REFERENCE.findall(body)) >= 5  # 정규식이 표기와 어긋나 아무것도 못 찾는 일을 막는다
    assert sorted(referenced - set(numbers)) == [], "a '20장 N번' reference points at a missing or retired item"
    # 정비에서 정한 것으로 옮긴 동작(끊긴 동안 발행되는 스텝)은 번호가 붙은 질문이 아니다.
    first_item = _first_section_20_item(section)
    decided = section.index("정비에서 정한 것")
    assert decided < first_item
    assert "steps_closure_across_gap" in section[decided:first_item]
    # 이식 대상이 아닌 질문은 README의 절을 가리킨다.
    assert 'README의 "소유자 결정 대기" 절' in section


def test_section_20_check_catches_a_gap_in_the_numbering() -> None:
    text = "## 20. 제목\n\n1. **가.** 설명\n2. **나.** 설명\n4. **라.** 설명\n"
    numbers = [int(number) for number in _SECTION_20_ITEM.findall(text)]
    assert numbers == [1, 2, 4] and numbers != list(range(1, len(numbers) + 1))
    assert _SECTION_20_REFERENCE.findall("판정에서 뺄지는 20장 6번, 이름은 20장 10번).") == ["6", "10"]
    assert _VECTOR_SIZE.findall("지금은 JSON 파일 81개, 합계 26,908,941 bytes다") == [("81", "26,908,941")]


def _section_20_items(spec_text: str) -> dict[int, str]:
    """20장의 번호 붙은 질문 -> 그 항목의 글. 마지막 항목은 맺는 문단 앞의 빈 줄에서 끝난다."""
    section = spec_text[_section_20_start(spec_text) :]
    matches = list(_SECTION_20_ITEM.finditer(section))
    items: dict[int, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else section.index("\n\n", match.start())
        items[int(match.group(1))] = section[match.start() : end]
    return items


def test_section_20_states_the_alternatives_and_keeps_decided_items_out(spec_text: str) -> None:
    items = _section_20_items(spec_text)
    assert sorted(items) == list(range(2, 13))  # 1번은 2026-11-14에 결정되어 비워 두었다
    section = spec_text[_section_20_start(spec_text) :]
    intro = section[: _first_section_20_item(section)]
    # 동작을 바꾸는 질문(2~9번)에는 README의 같은 번호 항목과 같은 대안이 적혀 있다. 추천은 README에만 적는다.
    for number in range(2, 10):
        assert "제안된 대안" in items[number], f"item {number} must state the proposed alternative"
    assert "추천" in intro and all("추천" not in text for text in items.values())
    # 거슬러 보기를 끊는 기준은 구현하면서 정한 것이다: 3번의 질문이 아니라 "정비에서 정한 것"에 있다.
    decided = intro[intro.index("정비에서 정한 것") :]
    assert "함께 확인할 것" not in items[3]
    assert "runs_velocity_after_gap" in decided and "`max_dt_ms`" in decided
    # 12번은 표시용 상태를 벡터에 담을지를 함께 묻는다.
    assert "`gaze_zone`" in items[12]
    # 계획서의 결정 번호(D11 등)를 뜻 없이 쓰지 않는다.
    assert re.search(r"\bD\d+\b", section) is None


def test_section_20_item_9_quotes_the_current_vector_count(spec_text: str) -> None:
    from drowsy import vectors

    # 9번은 "대안을 넣어도 지금의 벡터가 모두 통과한다"고 적는다. 벡터가 늘면 다시 돌려 보고 숫자를 고친다.
    item = _section_20_items(spec_text)[9]
    assert re.findall(r"(\d+)개가 모두 그대로 통과", item) == [str(len(vectors.VECTOR_NAMES))]
    assert "`rules_gaze_credit_boundary`" in item and "`expected`는 같다" in item


def test_hover_decision_is_recorded_and_its_number_is_retired(spec_text: str, repo_root: Path) -> None:
    section = spec_text[_section_20_start(spec_text) :]
    intro = section[: _first_section_20_item(section)]
    # 결정 기록은 2026-11-13의 결정 목록 옆(20장 머리)에 있는 한 문단이다.
    assert intro.index("2026-11-13에 결정되어") < intro.index("2026-11-14에 결정되어 이 목록에서 빠진 것")
    start = intro.index("2026-11-14에 결정되어 이 목록에서 빠진 것")
    record = intro[start : intro.index("\n\n", start)]
    assert "1번이던 질문" in record and "지금대로 둠" in record and "**반쯤 뜬 눈꺼풀은 감김을 끝내지" in record
    # 결정된 동작이자 알려진 한계이고, 잰 사례를 적는다(tests/test_rules.py의 단위 테스트가 같은 값을 고정한다).
    assert "알려진 한계" in record and "재현해야 한다" in record
    for measured in ("24.1초", "72.0초", "33 ms 간격", "68.1초", "30 fps", "3초마다", "36번"):
        assert measured in record, measured
    test_name = "test_decided_case_of_lids_hovering_between_the_two_thresholds"
    assert f"`{test_name}`" in record
    assert f"def {test_name}(" in (repo_root / "tests" / "test_rules.py").read_text(encoding="utf-8")
    # 고르지 않은 대안은 한 번만, 대안으로 적는다(아직 기다리는 질문의 "제안된 대안"이 아니다).
    assert record.count("고르지 않은 대안") == 1 and "`hold_ms`(500)" in record and "제안된 대안" not in record
    assert set(_VECTOR_NAME.findall(record)) == set(HOVER_DECISION_VECTORS)
    # 번호는 비워 두었다고 한 문장으로 말하고 결정 기록을 가리킨다. 남은 질문의 수도 맞다.
    lead = intro[intro.index("기다리는 질문은 아래") :]
    assert "열하나(2~12번)" in lead and "1번은 2026-11-14에" in lead and "결정되어 이 목록에서 빠진 것" in lead
    assert len(_section_20_items(spec_text)) == 11 and RETIRED_SECTION_20_ITEMS == {1}
    # 비운 번호를 가리키는 곳이 없다(이 문서와 벡터 README).
    vectors_readme = (repo_root / "tests" / "vectors" / "README.md").read_text(encoding="utf-8")
    for text in (spec_text, vectors_readme):
        retired = {int(n) for n in _SECTION_20_REFERENCE.findall(text)} & RETIRED_SECTION_20_ITEMS
        assert retired == set(), "a reference still points at a retired item of section 20"
    # 감김 규칙의 본문(10.5절 7번, 12.5절)은 그 동작을 결정된 동작이자 알려진 한계로 적고 결정 기록을 가리킨다.
    item_7 = spec_text[spec_text.index("7. `ongoing_ms()`는") : spec_text.index("8. 두 표본이 모두")]
    closure_rule = spec_text[spec_text.index("### 12.5 감김") : spec_text.index("### 12.6 ")]
    for text in (item_7, closure_rule):
        assert "2026-11-14 소유자 결정" in text and "알려진 한계" in text and "20장 머리의 결정 기록" in text
    # 그 동작을 고정하는 벡터는 16.1절과 벡터 README에서 결정된 동작을 고정하는 벡터로 적는다.
    section_16_1 = spec_text[spec_text.index("### 16.1 ") : spec_text.index("## 17. ")]
    paragraph = section_16_1[section_16_1.index("이탈 임계 위에 머무는 장면이 든 벡터 여섯") :]
    paragraph = paragraph[: paragraph.index("\n\n")]
    assert set(_VECTOR_NAME.findall(paragraph)) == set(HOVER_DECISION_VECTORS)
    assert "2026-11-14 소유자 결정으로" in paragraph and "기다리는 질문에 걸린 벡터가 아니다" in paragraph
    part = vectors_readme[vectors_readme.index("감김 벡터 여섯은 2026-11-14의 소유자 결정을 고정한다") :]
    part = part[: part.index("소유자의 결정을 기다리는 질문 가운데 여기서는")]
    assert set(HOVER_DECISION_VECTORS) <= set(_VECTOR_NAME.findall(part))
    assert "지금대로 둠" in part and "20장 머리" in part and "기다리" not in part


def test_vectors_readme_does_not_claim_a_complete_list_of_waiting_questions(repo_root: Path) -> None:
    # 벡터 README는 기다리는 질문 가운데 전방 주시 쪽의 둘만 설명하고, 나머지는 PLAN 14.2절 표의 "다시 만들 것" 칸을 가리킨다.
    # "벡터에 걸린 것은 ... 둘이다"처럼 전부인 듯 읽히는 문장은 두지 않는다(다른 질문도 벡터를 바꾼다).
    vectors_readme = (repo_root / "tests" / "vectors" / "README.md").read_text(encoding="utf-8")
    assert "벡터에 걸린 것은 전방 주시 쪽의 둘이다" not in vectors_readme
    start = vectors_readme.index("소유자의 결정을 기다리는 질문 가운데 여기서는 전방 주시 쪽의 둘만 설명한다")
    paragraph = vectors_readme[start : vectors_readme.index("\n\n", start)] if "\n\n" in vectors_readme[start:] else vectors_readme[start:]
    assert "docs/PLAN.md 14.2절" in paragraph and '"다시 만들 것"' in paragraph
    assert "20장 5번" in paragraph and "20장 9번" in paragraph
    plan = (repo_root / "docs" / "PLAN.md").read_text(encoding="utf-8")
    section = plan[plan.index("14.2") :]
    assert "| 다시 만들 것 |" in section, "docs/PLAN.md 14.2 must keep the column the vectors README points at"


def test_run_detector_constraints_are_stated_and_hold(
    spec_text: str, pipeline_spec: S.PipelineSpec, rules_config: S.RulesConfig
) -> None:
    # 10.1절은 설정 검증이 두 검출기에 요구하는 두 조건을 적는다. 지금 상수는 둘 다 만족한다.
    contract = spec_text[spec_text.index("### 10.1 호출 계약") : spec_text.index("### 10.2 알고리즘")]
    assert "`max_dt_ms <= hold_ms`" in contract and "`hold_ms < clock.reset_gap_ms`" in contract
    assert "`0 < max_dt_ms <= hold_ms < clock.reset_gap_ms`" in contract
    for detector, clock in (
        (pipeline_spec.run_detector, pipeline_spec.clock),
        (rules_config.closure.detector, rules_config.clock),
    ):
        assert 0 < detector.max_dt_ms <= detector.hold_ms < clock.reset_gap_ms
    # 문서가 적은 조건을 어긴 설정은 읽히지 않는다.
    with pytest.raises(S.SpecError, match="hold_ms < clock.reset_gap_ms"):
        S.apply_overrides(rules_config, {"closure.detector.hold_ms": rules_config.clock.reset_gap_ms})


def test_cancel_gap_constraints_are_stated_and_hold(spec_text: str, pipeline_spec: S.PipelineSpec) -> None:
    # 11.5절은 설정 검증이 하품과 끄덕임의 취소 간격에 요구하는 조건을 적고, 10.1절은 그것을 한 문장으로 가리킨다.
    section = spec_text[spec_text.index("### 11.5 하품과 끄덕임") : spec_text.index("### 11.6 ")]
    assert "`yawn.cancel_gap_ms < clock.reset_gap_ms`" in section
    assert "`nod.cancel_gap_ms < clock.reset_gap_ms`" in section
    assert "지금 값은 500,\n500, 10000" in section or "지금 값은 500, 500, 10000" in section
    assert "`nod.ema_max_dt_ms`에는 이런 조건이 없다" in section
    contract = spec_text[spec_text.index("### 10.1 호출 계약") : spec_text.index("### 10.2 알고리즘")]
    assert "`yawn.cancel_gap_ms`와 `nod.cancel_gap_ms`도 `clock.reset_gap_ms`보다 작아야 한다(11.5절)" in contract
    reset_gap_ms = pipeline_spec.clock.reset_gap_ms
    assert (pipeline_spec.yawn.cancel_gap_ms, pipeline_spec.nod.cancel_gap_ms, reset_gap_ms) == (500, 500, 10000)
    # 문서가 적은 조건을 어긴 설정은 읽히지 않는다. 느린 평균의 간격 상한에는 조건이 없다.
    for block in ("yawn", "nod"):
        with pytest.raises(S.SpecError, match=rf"pipeline\.{block}: need cancel_gap_ms < clock\.reset_gap_ms"):
            S.apply_overrides(pipeline_spec, {f"{block}.cancel_gap_ms": reset_gap_ms})
    assert S.apply_overrides(pipeline_spec, {"nod.ema_max_dt_ms": reset_gap_ms}).nod.ema_max_dt_ms == reset_gap_ms


def test_mutation_check_table_of_section_16_1(spec_text: str, tables: dict[str, list[Table]]) -> None:
    # 16.1절의 한 줄 바꾸기 점검 표: 모듈 여섯, 묶음 둘. 줄마다 (벡터가 잡음) + (단위 테스트만 잡음) + (둘 다 통과)
    # + (문법에 맞지 않아 뺀 사본)이 바꾼 사본의 수와 같다. 수 자체는 점검을 다시 돌려야 확인된다(저장소 밖의 작업이다).
    (header, body), = tables["mutation-check"]
    assert header == ["모듈", "묶음", "바꾼 사본", "벡터가 잡음", "단위 테스트만 잡음", "둘 다 통과"]
    assert [(row[0].strip("`"), row[1]) for row in body] == [
        (f"drowsy/{module}", kind) for module in MUTATION_CHECK_MODULES for kind in MUTATION_CHECK_SETS
    ]
    for row in body:
        total, caught, tests_only = int(row[2]), int(row[3]), int(row[4])
        survived = re.fullmatch(r"(\d+)(?: \(그 밖에 (\d+)개는 문법에 맞지 않는다\))?", row[5])
        assert survived is not None, row
        assert caught + tests_only + int(survived.group(1)) + int(survived.group(2) or 0) == total, row
    section = spec_text[spec_text.index("### 16.1 ") : spec_text.index("## 17. ")]
    # 점검이 세 모듈만 보았다는 옛 문장은 남아 있지 않다. 벡터의 수는 지금의 수다.
    assert "점검의 대상은 `drowsy/blink.py`, `drowsy/rules.py`, `drowsy/pipeline.py`이고" not in section
    assert "벡터 102개를 모두 통과해도 아래는 확인된 것이 아니다" in section
    # `ClosureRunDetector.reset()`의 항목은 시각 순서 검사가 다시 시작되는 것도 적는다(tests/test_blink.py가 고정한다).
    bullet = section[section.index("- `ClosureRunDetector.reset()` 뒤의 첫 표본이") :]
    bullet = bullet[: bullet.index("\n- ")]
    assert "시각 순서 검사" in bullet and "`test_interpolation_needs_gap_within_max_dt`" in bullet
    # 홍채 비율: 무엇이 고정하는지와, 다른 이식 대상 모듈이 그 값을 읽지 않는다는 것을 적는다.
    assert "`signals_iris_tilted_eyes`" in section and "`iris_h`, `iris_v`를 읽지 않는다" in section


def test_readme_names_the_spec_as_the_porting_specification(repo_root: Path) -> None:
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    lines = [line for line in readme.splitlines() if "docs/SPEC.md" in line and "이식 명세" in line]
    assert len(lines) >= 1, "README.md must name docs/SPEC.md as the porting specification (이식 명세)"
    # 그 절이 README에 있어야 20장의 마지막 문장이 가리키는 곳이 있다.
    assert re.search(r"^## .*소유자 결정 대기", readme, re.MULTILINE) is not None


# ---------------------------------------------------------------------------
# 본문의 괄호 인용
# ---------------------------------------------------------------------------


def test_inline_quoted_constants_match_config(
    spec_text: str, pipeline_leaves: dict[str, Any], rules_leaves: dict[str, Any], root_leaves: dict[str, Any]
) -> None:
    sources = (pipeline_leaves, rules_leaves, root_leaves)
    top_level = {path.split(".")[0] for leaves in sources for path in leaves}
    checked = 0
    for token, number in _INLINE_QUOTE.findall(spec_text):
        candidates = [leaves[token] for leaves in sources if token in leaves]
        if not candidates and "." not in token:
            candidates = [
                value for leaves in sources for path, value in leaves.items() if path.split(".")[-1] == token
            ]
        if not candidates:
            # 설정 블록 이름으로 시작하는 경로인데 찾지 못했다면 오타다. 그 밖의 이름(함수 등)은 지나간다.
            assert not ("." in token and token.split(".")[0] in top_level), f"unknown config path quoted: {token}"
            continue
        for value in candidates:
            assert isinstance(value, (int, float)) and not isinstance(value, bool), f"{token} is not numeric"
            assert float(number) == float(value), f"`{token}`({number}) but the config has {value}"
        checked += 1
    # 정규식이 문서의 표기와 어긋나서 아무것도 검사하지 못하는 일을 막는다.
    assert checked >= 10


def test_inline_quote_check_catches_a_wrong_value(pipeline_leaves: dict[str, Any]) -> None:
    # 검사 자체가 동작하는지: 틀린 값을 인용한 문장에서 정규식이 경로와 숫자를 뽑아내야 한다.
    found = _INLINE_QUOTE.findall("간격이 `clock.max_hold_ms`(250)를 넘으면")
    assert found == [("clock.max_hold_ms", "250")]
    assert float(found[0][1]) != float(pipeline_leaves["clock.max_hold_ms"])
    assert not _matches("250", pipeline_leaves["clock.max_hold_ms"])
    assert not _matches("200.0", pipeline_leaves["clock.max_hold_ms"])  # 정수 자리에 실수 표기
    assert _matches("200", pipeline_leaves["clock.max_hold_ms"])
