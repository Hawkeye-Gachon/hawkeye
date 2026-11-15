"""`drowsy.evaluate` 테스트 (PLAN 6장, 1.2절의 결정 D7, D8, D9, D11, 4.4절의 리포트 항목, 11장 E3·M4).

`D숫자`는 계획서(`docs/PLAN.md`) 1.2절의 표에서 승인된 결정의 번호다. D3: 경고의 속도 억제(시선·감김 경고는 20 km/h
미만, 졸음 경고는 50 km/h 미만에서 내지 않는다). D7: 영상 단위 정확도의 분모는 논리 영상 전체이고 창이 없는 영상은
오답이다. D8: 2클래스(alert 대 drowsy) 보고. D9: 시간당 졸음 오판정 수(판정 시작 횟수와 창 단위 값). D11: 교차검증
결과가 기준을 만족할 때만 학습 모델을 기본 졸음 판정원으로 둔다. D13: 얼굴 화소를 저장하지 않는다. D16: INT8 양자화
비교. 계획서 14장의 소유자 결정(2026-11-13, 2026-11-14)은 이 번호와 다르고, "2026-11-13 소유자 결정 2"처럼 날짜를
붙여 적는다.

예측 표를 손으로 만들고 기댓값도 손으로 계산한다(계산은 각 테스트의 주석에 있다). 지표와 리포트의 테스트는 모델도
영상도 쓰지 않는다. 라벨 없는 영상의 경고 수 테스트만 `drowsy.synth`의 각본 세션을 추출 결과 형식으로 써서 저장소의
무작위 가중치 모델(`export/drowsiness_v0.onnx`)과 함께 재생하고, 각본에서 나와야 하는 이벤트 수와 비교한다.
여기 나오는 점수는 구현을 확인하려는 숫자일 뿐이고 성능과 무관하다.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import shutil
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import f1_score

from drowsy import evaluate, extract, synth
from drowsy.config import ExperimentConfig, ModelGate
from drowsy.dataset import (
    ROTATION_AMBIGUOUS,
    STEP_COLUMNS,
    SYNTHETIC_NOTE,
    RotationFlag,
    SubjectOverlapError,
    make_windows,
)
from drowsy.evaluate import (
    ALARM_COLUMNS,
    LABEL_COLUMNS,
    LABEL_DEFAULTS,
    LITERATURE,
    LITERATURE_LABEL,
    METRIC_NAMES,
    NO_PREDICTION,
    PLAN_DECISION_NOTE,
    PREDICTION_COLUMNS,
    PREDICTION_DTYPES,
    REAL_NOTE,
    REQUIRED_PREDICTION_COLUMNS,
    SWEEP_COLUMNS,
    TWO_CLASSES,
    VOTE_COLUMNS,
    GateInputs,
    alarm_report,
    alarm_summary,
    alarm_sweep,
    alarm_videos,
    calibration_error,
    evaluate_group,
    exclusion_list,
    expected_calibration_error,
    gate_inputs,
    group_key,
    iter_runs,
    model_gate,
    paired_subject_bootstrap,
    prediction_table,
    subject_bootstrap,
    supplementary_pairs,
    two_class,
    untemper,
    validate_predictions,
    video_metrics,
    video_table,
    video_vote,
    vote_window_metrics,
    window_metrics,
    write_reports,
)
from drowsy.pipeline import DriverPipeline
from drowsy.rules import RuleEngine
from drowsy.spec import (
    FeatureSpec,
    ModelDrowsyRules,
    PerclosDrowsyRules,
    PipelineSpec,
    RulesConfig,
    apply_overrides,
    file_sha256,
    rules_hash,
)
from drowsy.types import (
    CLASS_ALERT,
    CLASS_DROWSY,
    CLASS_LOW_VIGILANT,
    DROWSY_SOURCE_EITHER,
    DROWSY_SOURCE_MODEL,
    DROWSY_SOURCE_PERCLOS,
    DROWSY_SOURCES,
    EVENT_DROWSY,
    EVENT_TYPES,
    FEATURE_NAMES,
    N_FEATURES,
    PHASE_END,
    PHASE_ONSET,
    StepFeatures,
)

LABELS = (0, 5, 10)
# 창 하나의 확률. 가장 큰 값이 그 창의 예측이다.
A = (0.7, 0.2, 0.1)
L = (0.2, 0.7, 0.1)
D = (0.1, 0.2, 0.7)


# ---------------------------------------------------------------------------
# 표 만들기
# ---------------------------------------------------------------------------


def window_rows(
    subject: int,
    class_index: int,
    probs: Sequence[tuple[float, float, float]],
    *,
    part: int = 0,
    k_start: int = 240,
    ks: Sequence[int] | None = None,
    perclos: float | Sequence[float] = 0.0,
    gate: bool = True,
    seed: int = 0,
    family: str = "tcn",
    arm: str = "alert_baseline",
    run_id: str = "run",
    temperature: float = 1.0,
    fold: int | None = None,
    labels: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """한 세션(영상의 부분)의 창 행들. fold는 따로 주지 않으면 공식 fold다.

    labels를 주면 그 표지 열(feature_source, variant, supplementary)을 행에 더한다. 주지 않으면 표지 열이 없는
    예전 형식의 행이다.
    """
    steps = list(range(k_start, k_start + len(probs))) if ks is None else list(ks)
    ratios = [float(perclos)] * len(probs) if isinstance(perclos, (int, float)) else list(perclos)
    extra = {} if labels is None else dict(labels)
    return [
        {
            "run_id": run_id,
            "family": family,
            "arm": arm,
            "fold": (subject - 1) // 12 + 1 if fold is None else fold,
            "seed": seed,
            "subject": subject,
            "video_id": f"S{subject:02d}_{LABELS[class_index]}",
            "part": part,
            "class_index": class_index,
            "k": k,
            "t_end_ms": 1 + 1000 * k,
            "p_alert": p[0],
            "p_low_vigilant": p[1],
            "p_drowsy": p[2],
            "perclos_60": ratio,
            "gate_pass": gate,
            "temperature": temperature,
            **extra,
        }
        for k, p, ratio in zip(steps, probs, ratios)
    ]


def sequence_rows(
    subject: int, class_index: int, p_drowsy: Sequence[float], **kwargs: Any
) -> list[dict[str, Any]]:
    """p_drowsy 수열로 만든 세션. 나머지 확률은 모두 p_alert에 준다(p_drowsy > 0.5일 때만 drowsy로 분류된다)."""
    return window_rows(subject, class_index, [(1.0 - p, 0.0, p) for p in p_drowsy], **kwargs)


def table(*sessions: list[dict[str, Any]]) -> pd.DataFrame:
    return validate_predictions(pd.DataFrame([row for session in sessions for row in session]))


def video_rows(*entries: tuple[int, int, bool]) -> pd.DataFrame:
    """영상 표. 항목은 (subject, class_index, gate_pass)."""
    return pd.DataFrame(
        [
            {
                "video_id": f"S{subject:02d}_{LABELS[class_index]}",
                "subject": subject,
                "fold": (subject - 1) // 12 + 1,
                "class_index": class_index,
                "gate_pass": gate,
            }
            for subject, class_index, gate in entries
        ]
    )


def vote_row(votes: pd.DataFrame, video_id: str) -> pd.Series:
    rows = votes[votes["video_id"] == video_id]
    assert len(rows) == 1
    return rows.iloc[0]


@pytest.fixture()
def small_table() -> pd.DataFrame:
    """영상 넷: S01_0(창 4개), S01_5(2개), S01_10(1개), 나뉜 영상 S32_10(부분 1에 3개, 부분 2에 2개)."""
    return table(
        window_rows(1, CLASS_ALERT, [A, A, A, D]),
        window_rows(1, CLASS_LOW_VIGILANT, [L, D]),
        window_rows(1, CLASS_DROWSY, [D]),
        window_rows(32, CLASS_DROWSY, [D, D, A], part=1),
        window_rows(32, CLASS_DROWSY, [A, A], part=2),
    )


# ---------------------------------------------------------------------------
# 예측 표의 형식
# ---------------------------------------------------------------------------


def test_schema_columns_and_dtypes() -> None:
    assert LABEL_COLUMNS == ("feature_source", "variant", "supplementary")
    assert LABEL_DEFAULTS == {"feature_source": "native", "variant": "base", "supplementary": False}
    assert PREDICTION_COLUMNS == REQUIRED_PREDICTION_COLUMNS + LABEL_COLUMNS
    assert PREDICTION_DTYPES["feature_source"] == PREDICTION_DTYPES["variant"] == "str"
    assert PREDICTION_DTYPES["supplementary"] == "bool"
    assert REQUIRED_PREDICTION_COLUMNS == (
        "run_id",
        "family",
        "arm",
        "fold",
        "seed",
        "subject",
        "video_id",
        "part",
        "class_index",
        "k",
        "t_end_ms",
        "p_alert",
        "p_low_vigilant",
        "p_drowsy",
        "perclos_60",
        "gate_pass",
        "temperature",
    )
    assert PREDICTION_DTYPES["run_id"] == PREDICTION_DTYPES["family"] == PREDICTION_DTYPES["arm"] == "str"
    assert PREDICTION_DTYPES["video_id"] == "str"
    for name in ("fold", "seed", "subject", "part", "class_index", "k", "t_end_ms"):
        assert PREDICTION_DTYPES[name] == "int64"
    for name in ("p_alert", "p_low_vigilant", "p_drowsy", "perclos_60", "temperature"):
        assert PREDICTION_DTYPES[name] == "float64"
    assert PREDICTION_DTYPES["gate_pass"] == "bool"


def test_validate_predictions_normalises_order_and_dtypes() -> None:
    raw = pd.DataFrame(window_rows(1, CLASS_ALERT, [A, D]))
    raw["note"] = "kept"
    raw = raw[list(reversed(raw.columns))].astype({"fold": "int32", "p_alert": "float32", "perclos_60": "int64"})
    checked = validate_predictions(raw)
    assert list(checked.columns) == list(PREDICTION_COLUMNS) + ["note"]
    for name, dtype in PREDICTION_DTYPES.items():
        assert str(checked[name].dtype) == dtype
    assert len(validate_predictions(checked.iloc[:0])) == 0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda t: t.drop(columns=["temperature"]), "lacks columns"),
        (lambda t: t.assign(p_drowsy=[0.5, 0.5]), "sum to 1"),
        (lambda t: t.assign(p_alert=[1.2, 0.7], p_drowsy=[-0.4, 0.1]), "within \\[0, 1\\]"),
        (lambda t: t.assign(class_index=3), "class_index"),
        (lambda t: t.assign(k=[240, 240]), "more than once"),
        (lambda t: t.assign(class_index=[0, 1]), "one subject, class_index and gate_pass"),
        (lambda t: t.assign(gate_pass=[True, False]), "one subject, class_index and gate_pass"),
        (lambda t: t.assign(perclos_60=[0.1, math.nan]), "missing values"),
        (lambda t: t.assign(perclos_60=[0.1, 1.5]), "perclos_60"),
        (lambda t: t.assign(temperature=0.0), "temperature"),
        (lambda t: t.assign(k=[0, 1]), "k "),
        (lambda t: t.assign(fold=[1.0, 1.0]), "must be int64"),
        (lambda t: t.assign(gate_pass=[1, 1]), "must be bool"),
        (lambda t: t.assign(video_id=[7, 7]), "must be str"),
    ],
)
def test_validate_predictions_rejects(change: Any, message: str) -> None:
    good = pd.DataFrame(window_rows(1, CLASS_ALERT, [A, D]))
    with pytest.raises(ValueError, match=message):
        validate_predictions(change(good))


def test_iter_runs_splits_by_run_family_arm_seed() -> None:
    preds = table(
        window_rows(1, CLASS_ALERT, [A], seed=1),
        window_rows(1, CLASS_ALERT, [A], seed=0),
        window_rows(1, CLASS_ALERT, [A, A], family="gru"),
    )
    runs = list(iter_runs(preds))
    assert [key for key, _ in runs] == [
        ("run", "gru", "alert_baseline", 0),
        ("run", "tcn", "alert_baseline", 0),
        ("run", "tcn", "alert_baseline", 1),
    ]
    assert [len(rows) for _, rows in runs] == [2, 1, 1]


# ---------------------------------------------------------------------------
# 표지 열: 특징 출처, 변형, 부가 표시
# ---------------------------------------------------------------------------

FPS15 = {"feature_source": "fps15", "variant": "base", "supplementary": True}
ORDINAL = {"feature_source": "native", "variant": "ordinal", "supplementary": True}
INT8 = {"feature_source": "native", "variant": "base", "supplementary": True}


def test_group_key_is_short_for_default_labels() -> None:
    assert group_key("run", "tcn", "alert_baseline") == ("run", "tcn", "alert_baseline")
    assert group_key("run", "tcn", "alert_baseline", "native", "base") == ("run", "tcn", "alert_baseline")
    assert group_key("run", "tcn", "alert_baseline", "fps15") == ("run", "tcn", "alert_baseline", "fps15", "base")
    assert group_key("run", "tcn", "global", variant="ordinal") == ("run", "tcn", "global", "native", "ordinal")
    assert evaluate.GROUP_KEYS == ("run_id", "family", "arm", "feature_source", "variant")
    assert evaluate.RUN_KEYS == evaluate.GROUP_KEYS + ("seed",)
    assert evaluate.FAMILY_TCN_INT8 == "tcn_int8" and evaluate.INT8_SUFFIX == "_int8"


def test_validate_predictions_fills_label_defaults_for_tables_without_them() -> None:
    """표지 열이 없는 예전 형식의 표: 기본값(native, base)으로 채우고, 부가 표시는 그 두 표지에서 정해 돌려준다."""
    old = pd.DataFrame(window_rows(1, CLASS_ALERT, [A, D]))
    assert not set(LABEL_COLUMNS) & set(old.columns)
    checked = validate_predictions(old)
    assert list(checked.columns) == list(PREDICTION_COLUMNS)
    assert checked["feature_source"].tolist() == ["native", "native"]
    assert checked["variant"].tolist() == ["base", "base"]
    assert checked["supplementary"].tolist() == [False, False]
    for name, dtype in PREDICTION_DTYPES.items():
        assert str(checked[name].dtype) == dtype
    # 한 열만 있어도 된다. 없는 표지는 기본값이고, 부가 표시는 표지에서 정한다: 변형이 기본값이 아니므로 부가 결과다
    # (전에는 상수 False로 채워져 순서형 라벨 묶음이 주 결과로 들어갔다)
    partial = validate_predictions(old.assign(variant="ordinal"))
    assert partial["variant"].tolist() == ["ordinal", "ordinal"]
    assert partial["feature_source"].tolist() == ["native", "native"]
    assert partial["supplementary"].tolist() == [True, True] and str(partial["supplementary"].dtype) == "bool"


def test_validate_predictions_accepts_old_and_supplementary_tables_concatenated() -> None:
    """예전 형식의 표와 부가 묶음의 표를 이어 붙이면 예전 쪽 행의 표지가 빈 값이 된다. 그 행만 기본값이 된다."""
    old = pd.DataFrame(window_rows(1, CLASS_ALERT, [A, D]))
    extra = pd.DataFrame(window_rows(1, CLASS_ALERT, [A, A, D], labels=FPS15))
    joined = pd.concat([old, extra], ignore_index=True)
    assert joined["supplementary"].isna().sum() == 2
    checked = validate_predictions(joined)
    assert checked["feature_source"].tolist() == ["native", "native", "fps15", "fps15", "fps15"]
    assert checked["variant"].tolist() == ["base"] * 5
    assert checked["supplementary"].tolist() == [False, False, True, True, True]
    assert str(checked["supplementary"].dtype) == "bool" and str(checked["feature_source"].dtype) == "str"
    # 같은 창 (video, part, k)이 두 묶음에 있어도 겹침이 아니다: 특징 출처가 다른 실행이다
    assert len(checked) == 5
    # 검사를 통과한 표끼리 이어 붙인 것도 그대로 통과한다
    again = validate_predictions(pd.concat([validate_predictions(old), validate_predictions(extra)]))
    pd.testing.assert_frame_equal(again, checked)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda t: t.assign(supplementary=[True, False]), "supplementary in all of its rows or in none"),
        (lambda t: t.assign(feature_source=["fps15", " "]), "feature_source must not be empty"),
        (lambda t: t.assign(variant=""), "variant must not be empty"),
        (lambda t: t.assign(supplementary=["yes", "no"]), "must be bool"),
        (lambda t: t.assign(feature_source=[1, 2]), "must be str"),
        # 같은 특징 출처 안에서는 한 영상의 gate_pass가 하나여야 한다
        (lambda t: t.assign(gate_pass=[True, False]), "one subject, class_index and gate_pass"),
    ],
)
def test_validate_predictions_rejects_bad_labels(change: Any, message: str) -> None:
    good = pd.DataFrame(window_rows(1, CLASS_ALERT, [A, D], labels=FPS15))
    with pytest.raises(ValueError, match=message):
        validate_predictions(change(good))


@pytest.mark.parametrize(
    ("labels", "key"),
    [
        ({"feature_source": "fps15"}, ("run", "tcn", "alert_baseline", "fps15", "base")),
        ({"variant": "ordinal"}, ("run", "tcn", "alert_baseline", "native", "ordinal")),
        (
            {"feature_source": "fold_global", "variant": "no_velocity"},
            ("run", "tcn", "alert_baseline", "fold_global", "no_velocity"),
        ),
    ],
)
def test_validate_predictions_refuses_a_labelled_group_that_is_not_supplementary(
    labels: dict[str, str], key: tuple[str, ...]
) -> None:
    """특징 출처가 native가 아니거나 변형이 base가 아닌 묶음은 주 결과가 될 수 없다. False로 적혀 있으면 거부한다."""
    rows = pd.DataFrame(window_rows(1, CLASS_ALERT, [A, D]))
    with pytest.raises(ValueError, match="is never a main result and must have supplementary = True") as refused:
        validate_predictions(rows.assign(**labels, supplementary=False))
    assert f"not flagged: [{key!r}]" in str(refused.value)  # 어느 묶음인지 알려 준다
    # 부가 표시가 없으면(열이 없거나 값이 비어 있으면) 표지에서 정한다: 부가 결과다
    assert validate_predictions(rows.assign(**labels))["supplementary"].tolist() == [True, True]
    blank = rows.assign(**labels, supplementary=pd.Series([None, None], dtype=object))
    assert validate_predictions(blank)["supplementary"].tolist() == [True, True]
    half = rows.assign(**labels, supplementary=pd.Series([True, None], dtype=object))
    assert validate_predictions(half)["supplementary"].tolist() == [True, True]
    assert validate_predictions(rows.assign(**labels, supplementary=True))["supplementary"].tolist() == [True, True]


def test_validate_predictions_keeps_the_explicit_flag_of_default_labelled_families() -> None:
    """표지가 기본값인 묶음(ONNX로 다시 돌린 예측, INT8 모델)은 적혀 있는 부가 표시를 그대로 따른다.

    계열 이름으로 추측하지 않는다: 표시가 없으면 표지(기본값)를 따라 False다.
    """
    for family in ("tcn", "tcn_onnx", "tcn_onnx_int8", evaluate.FAMILY_TCN_INT8):
        rows = pd.DataFrame(window_rows(1, CLASS_ALERT, [A, D], family=family))
        assert validate_predictions(rows.assign(supplementary=True))["supplementary"].tolist() == [True, True]
        assert validate_predictions(rows.assign(supplementary=False))["supplementary"].tolist() == [False, False]
        assert validate_predictions(rows)["supplementary"].tolist() == [False, False]
    # 한 표 안에서 행마다 따로 정한다: 표시가 적힌 주 결과, 표시가 적힌 INT8, 표시가 없는 15 fps 묶음
    main = pd.DataFrame(window_rows(1, CLASS_ALERT, [A], labels={"supplementary": False}))
    int8 = pd.DataFrame(window_rows(1, CLASS_ALERT, [A], family="tcn_int8", labels={"supplementary": True}))
    fps15 = pd.DataFrame(window_rows(1, CLASS_ALERT, [A], labels={"feature_source": "fps15"}))
    checked = validate_predictions(pd.concat([main, int8, fps15], ignore_index=True))
    assert checked["feature_source"].tolist() == ["native", "native", "fps15"]
    assert checked["supplementary"].tolist() == [False, True, True]


def test_gate_pass_may_differ_between_feature_sources() -> None:
    """다른 추출의 품질 표에서 온 gate_pass는 원본 추출의 값과 달라도 된다(영상의 다른 값은 여전히 하나여야 한다)."""
    native = window_rows(1, CLASS_ALERT, [A, D], gate=True)
    other = window_rows(1, CLASS_ALERT, [A, D], gate=False, labels=FPS15)
    checked = validate_predictions(pd.DataFrame(native + other))
    assert checked.groupby("feature_source")["gate_pass"].first().to_dict() == {"fps15": False, "native": True}
    wrong_class = [dict(row, class_index=CLASS_DROWSY) for row in other]
    with pytest.raises(ValueError, match="one subject, class_index and gate_pass"):
        validate_predictions(pd.DataFrame(native + wrong_class))


def test_iter_runs_keeps_labelled_groups_apart() -> None:
    preds = table(
        window_rows(1, CLASS_ALERT, [A]),
        window_rows(1, CLASS_ALERT, [A, A], labels=FPS15),
        window_rows(1, CLASS_ALERT, [A, A, A], labels=ORDINAL, seed=1),
        window_rows(1, CLASS_ALERT, [A, A, A, A], family="tcn_int8", labels=INT8),
    )
    runs = list(iter_runs(preds))
    assert [key for key, _ in runs] == [
        ("run", "tcn", "alert_baseline", "fps15", "base", 0),
        ("run", "tcn", "alert_baseline", 0),
        ("run", "tcn", "alert_baseline", "native", "ordinal", 1),
        ("run", "tcn_int8", "alert_baseline", 0),
    ]
    assert [len(rows) for _, rows in runs] == [2, 1, 3, 4]
    # 표지가 다른 실행을 한 표로 넘기면 지표 함수가 거부한다
    with pytest.raises(ValueError, match="mixes several values of 'feature_source'"):
        video_vote(preds[preds["family"] == "tcn"])
    # 검사를 거치지 않은 예전 형식의 표도 지표 함수와 iter_runs가 받는다
    old = pd.DataFrame(window_rows(1, CLASS_ALERT, [A, A, D]))
    assert [key for key, _ in iter_runs(old)] == [("run", "tcn", "alert_baseline", 0)]
    assert vote_row(video_vote(old), "S01_0")["votes_alert"] == 2


def test_metric_functions_refuse_mixed_runs(rules_config: RulesConfig) -> None:
    preds = table(window_rows(1, CLASS_ALERT, [A], seed=0), window_rows(1, CLASS_ALERT, [D], seed=1))
    for call in (
        lambda: window_metrics(preds),
        lambda: video_vote(preds),
        lambda: two_class(preds),
        lambda: alarm_report(preds, rules_config, DROWSY_SOURCE_MODEL),
    ):
        with pytest.raises(ValueError, match="mixes several values of 'seed'"):
            call()


def test_metric_functions_do_not_modify_their_inputs(small_table: pd.DataFrame, rules_config: RulesConfig) -> None:
    videos = video_rows(
        (1, CLASS_ALERT, True), (1, CLASS_LOW_VIGILANT, True), (1, CLASS_DROWSY, True), (32, CLASS_DROWSY, True)
    )
    preds_before, videos_before = small_table.copy(deep=True), videos.copy(deep=True)
    votes = video_vote(small_table, videos)
    votes_before = votes.copy(deep=True)
    window_metrics(small_table)
    two_class(small_table, videos)
    video_metrics(votes)
    alarm_report(small_table, rules_config, DROWSY_SOURCE_EITHER, videos, sweep=(0.5,))
    calibration_error(small_table)
    subject_bootstrap(votes, lambda sample: float(sample["correct"].mean()), n=20)
    pd.testing.assert_frame_equal(small_table, preds_before)
    pd.testing.assert_frame_equal(videos, videos_before)
    pd.testing.assert_frame_equal(votes, votes_before)


def _steps_frame(n_steps: int, perclos: np.ndarray) -> pd.DataFrame:
    """세션 하나(S03_5)의 스텝 표. 모두 조회 구간이 찬 스텝이다."""
    data: dict[str, Any] = {
        "video_id": "S03_5",
        "session_id": "S03_5",
        "subject": 3,
        "fold": 1,
        "label": 5,
        "class_index": CLASS_LOW_VIGILANT,
        "part": 0,
        "k": np.arange(121, 121 + n_steps),
        "t_end_ms": 1 + 1000 * np.arange(121, 121 + n_steps),
        "full_lookback": True,
    }
    for name in FEATURE_NAMES:
        data[name] = np.zeros(n_steps)
    data["perclos_60"] = perclos
    return pd.DataFrame(data, columns=list(STEP_COLUMNS))


def test_prediction_table_from_windows() -> None:
    perclos = 0.1 + 0.001 * np.arange(123)  # float32로는 정확히 표현되지 않는 값들
    steps = _steps_frame(123, perclos)
    windows = make_windows(steps, window_steps=120, stride=1)
    assert len(windows) == 4
    probs = np.array([A, L, D, A], dtype=np.float32)
    videos = video_rows((3, CLASS_LOW_VIGILANT, False))
    preds = prediction_table(
        windows, probs, steps, videos, run_id="e2", family="gru", arm="session_baseline", seed=2, temperature=1.5
    )

    assert list(preds.columns) == list(PREDICTION_COLUMNS)
    for name, dtype in PREDICTION_DTYPES.items():
        assert str(preds[name].dtype) == dtype
    assert preds["k"].tolist() == [240, 241, 242, 243]
    assert preds["t_end_ms"].tolist() == [240001, 241001, 242001, 243001]
    # 창 텐서의 float32가 아니라 스텝 표의 float64 값이다
    assert preds["perclos_60"].tolist() == perclos[119:].tolist()
    assert preds["perclos_60"].tolist() != windows.x[:, -1, 0].astype(np.float64).tolist()
    assert np.allclose(preds[["p_alert", "p_low_vigilant", "p_drowsy"]].to_numpy(), probs)
    first = preds.iloc[0]
    assert (first["run_id"], first["family"], first["arm"], first["seed"]) == ("e2", "gru", "session_baseline", 2)
    assert (first["subject"], first["video_id"], first["part"], first["fold"]) == (3, "S03_5", 0, 1)
    assert first["class_index"] == CLASS_LOW_VIGILANT and not first["gate_pass"] and first["temperature"] == 1.5

    with pytest.raises(ValueError, match="probs must have shape"):
        prediction_table(windows, probs[:3], steps, videos, run_id="e2", family="gru", arm="a", seed=0)
    other = steps.assign(perclos_60=perclos + 0.01)
    with pytest.raises(ValueError, match="not the one these windows were made from"):
        prediction_table(windows, probs, other, videos, run_id="e2", family="gru", arm="a", seed=0)
    with pytest.raises(ValueError, match="lacks the last step"):
        prediction_table(windows, probs, steps.iloc[:-1], videos, run_id="e2", family="gru", arm="a", seed=0)
    with pytest.raises(ValueError, match="does not list the videos"):
        prediction_table(
            windows, probs, steps, video_rows((4, CLASS_LOW_VIGILANT, True)), run_id="e2", family="gru", arm="a", seed=0
        )


def test_prediction_table_labels_and_default_supplementary_flag() -> None:
    steps = _steps_frame(121, np.full(121, 0.25))
    windows = make_windows(steps, window_steps=120, stride=1)
    probs = np.array([A, D], dtype=np.float32)
    videos = video_rows((3, CLASS_LOW_VIGILANT, True))

    def build(**labels: Any) -> pd.Series:
        made = prediction_table(windows, probs, steps, videos, run_id="e3", family="tcn", arm="a", seed=0, **labels)
        assert list(made.columns) == list(PREDICTION_COLUMNS) and len(made) == 2
        return made.iloc[0]

    def labels_of(row: pd.Series) -> tuple[str, str, bool]:
        return (row["feature_source"], row["variant"], bool(row["supplementary"]))

    # 주 결과: 표지를 주지 않는다
    assert labels_of(build()) == ("native", "base", False)
    # 표지가 기본값이 아니면 부가 표시를 따로 주지 않아도 부가다(주 결과에 섞이지 않게)
    assert labels_of(build(feature_source=evaluate.FEATURE_SOURCE_FPS15)) == ("fps15", "base", True)
    assert labels_of(build(variant=evaluate.VARIANT_ORDINAL)) == ("native", "ordinal", True)
    both = build(feature_source=evaluate.FEATURE_SOURCE_FOLD_GLOBAL, variant=evaluate.VARIANT_NO_VELOCITY)
    assert labels_of(both) == ("fold_global", "no_velocity", True)
    # 계열 이름만 다른 부가 묶음(INT8)은 직접 표시한다
    assert labels_of(build(supplementary=True)) == ("native", "base", True)
    assert labels_of(build(supplementary=False)) == ("native", "base", False)
    # 표지가 기본값이 아닌 묶음을 주 결과로 표시할 수는 없다
    for labels in ({"variant": "ordinal"}, {"feature_source": "fps15"}):
        with pytest.raises(ValueError, match="must have supplementary = True"):
            build(**labels, supplementary=False)
        assert labels_of(build(**labels, supplementary=True))[2] is True


def _quality_frame() -> pd.DataFrame:
    rows = [
        # video_id, session_id, subject, class, part, raw, baseline, gate, video_n_windows, video_face_ratio,
        # t_b_ms, baseline_used, alert_fallback
        ("S01_0", "S01_0", 1, 0, 0, "ok", "ok", True, 362, 0.98, 60_000, "alert", False),
        ("S01_5", "S01_5", 1, 1, 0, "ok", "none", False, 0, 0.91, -1, "none", False),
        ("S32_10", "S32_10_1", 32, 2, 1, "ok", "ok", False, 12, 0.5, 90_000, "global", True),
        ("S32_10", "S32_10_2", 32, 2, 2, "missing", "none", False, 12, 0.5, -1, "none", False),
        ("S33_10", "S33_10_1", 33, 2, 1, "ok", "ok", True, 300, 0.9, 60_000, "alert", False),
        ("S33_10", "S33_10_2", 33, 2, 2, "ok", "ok", True, 300, 0.9, 120_000, "alert", False),
    ]
    return pd.DataFrame(
        [
            {
                "video_id": video_id,
                "session_id": session_id,
                "subject": subject,
                "fold": (subject - 1) // 12 + 1,
                "label": LABELS[class_index],
                "class_index": class_index,
                "part": part,
                "raw_status": raw,
                "baseline_status": baseline,
                "gate_pass": gate,
                "video_n_windows": n_windows,
                "video_face_ratio": ratio,
                "t_b_ms": t_b_ms,
                "baseline_used": used,
                "alert_fallback": fallback,
            }
            for (
                video_id, session_id, subject, class_index, part, raw, baseline, gate, n_windows, ratio,
                t_b_ms, used, fallback,
            ) in rows
        ]
    )  # fmt: skip


def test_video_table_merges_split_parts() -> None:
    videos = video_table(_quality_frame())
    assert videos["video_id"].tolist() == ["S01_0", "S01_5", "S32_10", "S33_10"]
    assert videos["gate_pass"].tolist() == [True, False, False, True]
    assert videos["n_windows"].tolist() == [362, 0, 12, 300]
    assert videos["fold"].tolist() == [1, 1, 3, 3]
    assert list(videos.columns) == list(evaluate.VIDEO_COLUMNS) + ["n_windows", "face_ratio"] + list(
        evaluate.BASELINE_COLUMNS
    )
    assert evaluate.VIDEO_COLUMNS == ("video_id", "subject", "fold", "class_index", "gate_pass")
    assert evaluate.BASELINE_COLUMNS == ("baseline_status", "t_b_ms", "baseline_used", "alert_fallback")


def test_video_table_carries_baseline_columns_per_logical_video() -> None:
    """PLAN 4.4절: 영상마다 기준선 상태, T_b, 쓴 기준선, alert_fallback. 나뉜 영상은 한 행으로 합친다.

    S32_10: 부분 1은 ok(T_b 90초), 부분 2는 추출 결과가 없어 none(−1) → 상태 "ok/none", T_b는 늦은 쪽 90초,
            쓴 기준선 "global/none", alert_fallback은 하나라도 True이면 True(여기서는 부분 1만 True로 적었다).
    S33_10: 두 부분 모두 ok → "ok" 하나로 적고, T_b는 60초와 120초 중 늦은 120초.
    """
    videos = video_table(_quality_frame()).set_index("video_id")
    assert videos["baseline_status"].to_dict() == {"S01_0": "ok", "S01_5": "none", "S32_10": "ok/none", "S33_10": "ok"}
    assert videos["t_b_ms"].to_dict() == {"S01_0": 60_000, "S01_5": -1, "S32_10": 90_000, "S33_10": 120_000}
    assert videos["baseline_used"].to_dict() == {
        "S01_0": "alert",
        "S01_5": "none",
        "S32_10": "global/none",
        "S33_10": "alert",
    }
    assert videos["alert_fallback"].to_dict() == {"S01_0": False, "S01_5": False, "S32_10": True, "S33_10": False}
    assert str(videos["t_b_ms"].dtype) == "int64" and str(videos["alert_fallback"].dtype) == "bool"

    # 기준선 열이 없는 품질 표(손으로 만든 표)도 받는다. 있는 열만 옮긴다
    bare = _quality_frame().drop(columns=["t_b_ms", "baseline_used", "alert_fallback"])
    assert list(video_table(bare).columns) == [*evaluate.VIDEO_COLUMNS, "n_windows", "face_ratio", "baseline_status"]
    none = bare.drop(columns=["baseline_status"])
    assert list(video_table(none).columns) == list(evaluate.VIDEO_COLUMNS) + ["n_windows", "face_ratio"]


def test_exclusion_list_reasons() -> None:
    listed = exclusion_list(_quality_frame().iloc[:4])
    assert listed["video_id"].tolist() == ["S01_5", "S32_10"]
    assert listed["reason"].tolist() == ["no_window", "quality_gate"]
    assert listed["detail"].tolist() == [
        "face_ratio=0.910, windows=0, raw=ok, baseline=none",
        "face_ratio=0.500, windows=12, raw=ok/missing, baseline=ok/none",
    ]
    assert len(exclusion_list(_quality_frame().iloc[:1])) == 0


# ---------------------------------------------------------------------------
# 창 단위 지표
# ---------------------------------------------------------------------------


def test_window_metrics_hand_computed() -> None:
    """S01_0: A A A D, S01_5: L D, S01_10: D.

    창 합산 혼동 행렬 [[3,0,1],[0,1,1],[0,0,1]]
      재현율 3/4, 1/2, 1. 정밀도 1, 1, 1/3. F1 = 6/7, 2/3, 1/2. 정확도 5/7.
    영상 가중 혼동 행렬 [[.75,0,.25],[0,.5,.5],[0,0,1]]
      재현율은 같다. 정밀도 1, 1, 1/1.75 = 4/7. F1 = 6/7, 2/3, 2·(4/7)/(11/7) = 8/11. 정확도 2.25/3.
    """
    preds = table(
        window_rows(1, CLASS_ALERT, [A, A, A, D]),
        window_rows(1, CLASS_LOW_VIGILANT, [L, D]),
        window_rows(1, CLASS_DROWSY, [D]),
    )
    metrics = window_metrics(preds)

    assert metrics.n_windows == 7 and metrics.n_videos == 3
    assert metrics.pooled.confusion == ((3.0, 0.0, 1.0), (0.0, 1.0, 1.0), (0.0, 0.0, 1.0))
    assert metrics.pooled.recall == pytest.approx((0.75, 0.5, 1.0))
    assert metrics.pooled.f1 == pytest.approx((6 / 7, 2 / 3, 1 / 2))
    assert metrics.pooled.macro_f1 == pytest.approx((6 / 7 + 2 / 3 + 1 / 2) / 3)
    assert metrics.pooled.accuracy == pytest.approx(5 / 7)

    assert metrics.video_weighted.confusion == ((0.75, 0.0, 0.25), (0.0, 0.5, 0.5), (0.0, 0.0, 1.0))
    assert metrics.video_weighted.recall == pytest.approx((0.75, 0.5, 1.0))
    assert metrics.video_weighted.f1 == pytest.approx((6 / 7, 2 / 3, 8 / 11))
    assert metrics.video_weighted.macro_f1 == pytest.approx((6 / 7 + 2 / 3 + 8 / 11) / 3)
    assert metrics.video_weighted.accuracy == pytest.approx(0.75)

    # 독립 구현과 비교: 창마다 1/(그 영상의 창 수)를 가중치로 준 sklearn의 macro-F1
    truth = [0, 0, 0, 0, 1, 1, 2]
    predicted = [0, 0, 0, 2, 1, 2, 2]
    weights = [0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 1.0]
    assert metrics.pooled.macro_f1 == pytest.approx(f1_score(truth, predicted, average="macro"))
    assert metrics.video_weighted.macro_f1 == pytest.approx(
        f1_score(truth, predicted, average="macro", sample_weight=weights)
    )
    assert vote_window_metrics(video_vote(preds)) == metrics


def test_window_metrics_absent_class_has_zero_f1_and_tie_goes_to_first_class() -> None:
    # 확률이 같으면 앞선 클래스(alert). 한 번도 나오지 않은 클래스의 F1은 0이다 → macro-F1 = (1 + 0 + 0) / 3
    preds = table(window_rows(1, CLASS_ALERT, [(0.4, 0.4, 0.2), (0.35, 0.3, 0.35)]))
    metrics = window_metrics(preds)
    assert metrics.pooled.confusion == ((2.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert metrics.pooled.f1 == (1.0, 0.0, 0.0)
    assert metrics.pooled.macro_f1 == pytest.approx(1 / 3)
    assert metrics.video_weighted.macro_f1 == pytest.approx(1 / 3)


def test_window_metrics_pool_split_parts_into_one_video(small_table: pd.DataFrame) -> None:
    # S32_10은 부분 둘의 창 5개(D D A / A A)가 한 영상으로 1을 나눠 갖는다: drowsy 행에 [0.6, 0, 0.4]
    metrics = window_metrics(small_table[small_table["subject"] == 32])
    assert metrics.n_videos == 1 and metrics.n_windows == 5
    assert metrics.video_weighted.confusion[CLASS_DROWSY] == pytest.approx((0.6, 0.0, 0.4))


# ---------------------------------------------------------------------------
# 영상 표결 (D7)
# ---------------------------------------------------------------------------


def test_video_vote_majority_and_columns(small_table: pd.DataFrame) -> None:
    votes = video_vote(small_table)
    assert list(votes.columns) == list(VOTE_COLUMNS)
    assert votes["video_id"].tolist() == ["S01_0", "S01_5", "S01_10", "S32_10"]

    alert = vote_row(votes, "S01_0")
    assert alert["n_windows"] == 4
    assert (alert["votes_alert"], alert["votes_low_vigilant"], alert["votes_drowsy"]) == (3, 0, 1)
    assert alert["mean_p_alert"] == pytest.approx((0.7 * 3 + 0.1) / 4)
    assert alert["mean_p_low_vigilant"] == pytest.approx(0.2)
    assert alert["mean_p_drowsy"] == pytest.approx((0.1 * 3 + 0.7) / 4)
    assert alert["pred_class"] == CLASS_ALERT and alert["correct"]


def test_video_vote_tie_is_broken_by_mean_probability(small_table: pd.DataFrame) -> None:
    # S01_5: L, D → 표는 low_vigilant 1, drowsy 1. 평균 확률은 low_vigilant (0.7 + 0.2)/2 = 0.45, drowsy 0.40
    row = vote_row(video_vote(small_table), "S01_5")
    assert (row["votes_low_vigilant"], row["votes_drowsy"]) == (1, 1)
    assert row["pred_class"] == CLASS_LOW_VIGILANT and row["correct"]

    # 평균 확률이 반대면 판정도 반대다: drowsy 평균 (0.8 + 0.1)/2 = 0.45 > low_vigilant (0.15 + 0.6)/2 = 0.375
    preds = table(window_rows(1, CLASS_LOW_VIGILANT, [(0.05, 0.15, 0.8), (0.3, 0.6, 0.1)]))
    row = vote_row(video_vote(preds), "S01_5")
    assert row["pred_class"] == CLASS_DROWSY and not row["correct"]


def test_video_vote_tie_looks_only_at_tied_classes_then_class_order() -> None:
    # 표: alert 1, drowsy 1, low_vigilant 0. 평균 확률은 low_vigilant가 0.44로 가장 크지만 표가 없으므로 후보가 아니다.
    # alert와 drowsy의 평균은 둘 다 (0.45 + 0.11)/2로 정확히 같다 → 순서가 앞선 alert
    preds = table(window_rows(1, CLASS_DROWSY, [(0.45, 0.44, 0.11), (0.11, 0.44, 0.45)]))
    row = vote_row(video_vote(preds), "S01_10")
    assert row["mean_p_alert"] == row["mean_p_drowsy"]
    assert row["mean_p_low_vigilant"] > row["mean_p_alert"]
    assert row["pred_class"] == CLASS_ALERT and not row["correct"]


def test_video_vote_pools_split_parts(small_table: pd.DataFrame) -> None:
    # 부분 1: D D A, 부분 2: A A → 합치면 alert 3, drowsy 2. 부분별로 따로 세면 부분 1은 drowsy였을 것이다
    votes = video_vote(small_table)
    row = vote_row(votes, "S32_10")
    assert (row["n_windows"], row["votes_alert"], row["votes_drowsy"]) == (5, 3, 2)
    assert row["pred_class"] == CLASS_ALERT and not row["correct"]
    assert (votes["video_id"] == "S32_10").sum() == 1


def test_video_vote_counts_a_video_without_windows_as_wrong(small_table: pd.DataFrame) -> None:
    videos = video_rows(
        (1, CLASS_ALERT, True),
        (1, CLASS_LOW_VIGILANT, True),
        (1, CLASS_DROWSY, True),
        (2, CLASS_ALERT, False),  # 예측 표에 창이 없다
        (32, CLASS_DROWSY, False),
    )
    votes = video_vote(small_table, videos)
    assert votes["video_id"].tolist() == ["S01_0", "S01_5", "S01_10", "S02_0", "S32_10"]
    missing = vote_row(votes, "S02_0")
    assert missing["n_windows"] == 0 and missing["pred_class"] == NO_PREDICTION and not missing["correct"]
    assert (missing["votes_alert"], missing["votes_low_vigilant"], missing["votes_drowsy"]) == (0, 0, 0)
    assert math.isnan(missing["mean_p_alert"])
    # gate_pass는 영상 표의 값이다(예측 표에서는 S32_10이 True였다)
    assert votes["gate_pass"].tolist() == [True, True, True, False, False]
    assert str(votes["pred_class"].dtype) == "int64" and str(votes["correct"].dtype) == "bool"

    # 맞은 영상: S01_0, S01_5, S01_10. 틀린 영상: S02_0(창 없음), S32_10(alert로 판정)
    metrics = video_metrics(votes)
    assert metrics.n_videos == 5 and metrics.n_no_window == 1
    assert metrics.accuracy == pytest.approx(3 / 5)
    assert metrics.recall == pytest.approx((1 / 2, 1.0, 1 / 2))
    assert metrics.confusion == ((1, 0, 0, 1), (0, 1, 0, 0), (1, 0, 1, 0))

    # 보조 행: 품질 기준을 통과한 영상만
    gated = video_metrics(votes[votes["gate_pass"]])
    assert gated.n_videos == 3 and gated.accuracy == 1.0 and gated.n_no_window == 0

    # 영상 표가 없으면 예측에 나온 영상만 분모다
    assert video_metrics(video_vote(small_table)).accuracy == pytest.approx(3 / 4)


def test_video_vote_rejects_inconsistent_video_table(small_table: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="does not list"):
        video_vote(small_table, video_rows((1, CLASS_ALERT, True)))
    full = video_rows(
        (1, CLASS_ALERT, True), (1, CLASS_LOW_VIGILANT, True), (1, CLASS_DROWSY, True), (32, CLASS_DROWSY, True)
    )
    with pytest.raises(ValueError, match="fold of a video differs"):
        video_vote(small_table, full.assign(fold=9))
    with pytest.raises(ValueError, match="more than once"):
        video_vote(small_table, pd.concat([full, full]))


def test_video_metrics_of_empty_table_is_zero() -> None:
    metrics = video_metrics(video_vote(table(window_rows(1, CLASS_ALERT, [A])).iloc[:0]))
    assert metrics.n_videos == 0 and metrics.accuracy == 0.0 and metrics.recall == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# 2클래스 (D8)
# ---------------------------------------------------------------------------


def test_two_class_hand_computed() -> None:
    """alert·drowsy 영상만, 창마다 p_alert와 p_drowsy만 비교한다.

    S01_0  : (0.3, 0.6, 0.1) ×2 → 3클래스로는 low_vigilant지만 여기서는 alert          → 맞음
    S01_5  : low_vigilant 영상이라 대상이 아니다
    S01_10 : (0.2,0.5,0.3) drowsy, (0.35,0.5,0.15) alert, (0.1,0.6,0.3) drowsy → 2 대 1  → 맞음
    S02_0  : 창 없음                                                                  → 틀림
    S02_10 : (0.4,0.2,0.4) 같으면 alert, (0.5,0.3,0.2) alert → alert                   → 틀림
    정확도 2/4, 재현율 alert 1/2, drowsy 1/2.
    """
    preds = table(
        window_rows(1, CLASS_ALERT, [(0.3, 0.6, 0.1), (0.3, 0.6, 0.1)]),
        window_rows(1, CLASS_LOW_VIGILANT, [D, D]),
        window_rows(1, CLASS_DROWSY, [(0.2, 0.5, 0.3), (0.35, 0.5, 0.15), (0.1, 0.6, 0.3)]),
        window_rows(2, CLASS_DROWSY, [(0.4, 0.2, 0.4), (0.5, 0.3, 0.2)]),
    )
    videos = video_rows(
        (1, CLASS_ALERT, True),
        (1, CLASS_LOW_VIGILANT, True),
        (1, CLASS_DROWSY, True),
        (2, CLASS_ALERT, True),
        (2, CLASS_LOW_VIGILANT, True),
        (2, CLASS_DROWSY, True),
    )
    votes = two_class(preds, videos)
    assert list(votes.columns) == list(VOTE_COLUMNS)
    assert votes["video_id"].tolist() == ["S01_0", "S01_10", "S02_0", "S02_10"]
    assert votes["pred_class"].tolist() == [CLASS_ALERT, CLASS_DROWSY, NO_PREDICTION, CLASS_ALERT]
    assert votes["correct"].tolist() == [True, True, False, False]
    assert (votes["votes_low_vigilant"] == 0).all()
    drowsy = vote_row(votes, "S01_10")
    assert (drowsy["votes_alert"], drowsy["votes_drowsy"]) == (1, 2)

    metrics = video_metrics(votes, TWO_CLASSES)
    assert metrics.classes == (CLASS_ALERT, CLASS_DROWSY)
    assert metrics.n_videos == 4 and metrics.n_no_window == 1
    assert metrics.accuracy == pytest.approx(2 / 4)
    assert metrics.recall == pytest.approx((1 / 2, 1 / 2))
    assert metrics.confusion == ((1, 0, 1), (1, 1, 0))

    # 같은 창을 3클래스로 표결하면 S01_0은 low_vigilant다
    assert vote_row(video_vote(preds), "S01_0")["pred_class"] == CLASS_LOW_VIGILANT
    with pytest.raises(ValueError, match="outside classes"):
        video_metrics(video_vote(preds), TWO_CLASSES)


def test_two_class_video_tie_uses_mean_of_the_two_probabilities() -> None:
    # 표 1 대 1. 평균 p_alert = (0.5 + 0.1)/2 = 0.30, 평균 p_drowsy = (0.2 + 0.6)/2 = 0.40 → drowsy
    preds = table(window_rows(1, CLASS_DROWSY, [(0.5, 0.3, 0.2), (0.1, 0.3, 0.6)]))
    assert vote_row(two_class(preds), "S01_10")["pred_class"] == CLASS_DROWSY


# ---------------------------------------------------------------------------
# 오경보 (PLAN 6장, D9)
# ---------------------------------------------------------------------------

HIGH = 0.6  # 시작 조건의 경계값(p_drowsy ≥ 0.6)
LOW = 0.4  # 종료 조건의 경계값(p_drowsy ≤ 0.4)
MID = 0.5  # 어느 쪽도 아님


def test_rule_thresholds_these_tests_assume(rules_config: RulesConfig) -> None:
    assert rules_config.drowsiness.model == ModelDrowsyRules(0.6, 10, 0.4, 10, 60000)
    assert rules_config.drowsiness.perclos == PerclosDrowsyRules(0.15, 10, 0.10, 10, 0)
    assert rules_config.clock.step_ms == 1000


def only_row(frame: pd.DataFrame) -> pd.Series:
    assert len(frame) == 1
    return frame.iloc[0]


def test_alarm_needs_ten_consecutive_steps(rules_config: RulesConfig) -> None:
    nine = table(sequence_rows(1, CLASS_ALERT, [MID] * 5 + [HIGH] * 9 + [MID] * 5))
    assert only_row(alarm_videos(nine, rules_config, DROWSY_SOURCE_MODEL))["onsets"] == 0

    ten = table(sequence_rows(1, CLASS_ALERT, [MID] * 5 + [HIGH] * 10 + [MID] * 5))
    row = only_row(alarm_videos(ten, rules_config, DROWSY_SOURCE_MODEL))
    # 10번째 높은 스텝에서 시작하고, 종료 조건(≤ 0.4)이 없으므로 끝까지 이어진다: 판정 상태 스텝 1 + 5
    assert (row["onsets"], row["alarm_steps"], row["n_steps"], row["evaluated_ms"]) == (1, 6, 20, 20000)

    # 0.6 바로 아래는 시작 조건이 아니다
    below = table(sequence_rows(1, CLASS_ALERT, [0.59] * 20))
    assert only_row(alarm_videos(below, rules_config, DROWSY_SOURCE_MODEL))["onsets"] == 0


def test_alarm_latches_one_continuously_flagged_video_is_one_onset(rules_config: RulesConfig) -> None:
    preds = table(sequence_rows(1, CLASS_ALERT, [0.9] * 100))
    report = alarm_report(preds, rules_config, DROWSY_SOURCE_MODEL)
    # 100스텝 내내 조건을 만족해도 시작은 한 번. 10번째 스텝부터 판정 상태: 91스텝
    assert report.alert_onsets == 1 and report.alert_videos_with_onset == 1
    assert report.alarm_time_share == pytest.approx(91 / 100)
    assert report.alert_hours == pytest.approx(100 / 3600)
    assert report.onsets_per_hour == pytest.approx(36.0)  # 1 / (100 s / 3600)
    # D9: 100개 창이 모두 drowsy로 분류됐다 → 100 / (100/3600 h) = 3600
    assert report.alert_flagged_windows == 100 and report.flagged_windows_per_hour == pytest.approx(3600.0)
    assert report.source == DROWSY_SOURCE_MODEL and report.onset_p == 0.6
    assert report.rules_hash == rules_hash(rules_config)
    assert list(report.videos.columns) == list(ALARM_COLUMNS)
    assert len(report.sweep) == 0 and list(report.sweep.columns) == list(SWEEP_COLUMNS)


def test_alarm_end_and_refractory_period(rules_config: RulesConfig) -> None:
    """높음 10(스텝 1~10) → 낮음 10(11~20) → 높음.

    스텝 10에서 시작, 스텝 20에서 종료. 그 뒤 높은 스텝이 스텝 30에서 10개 연속이 되지만 종료 후 60초가 지나지
    않아 시작하지 않는다. 종료 60초 뒤인 스텝 80이 처음으로 다시 시작할 수 있는 스텝이다.
    """
    until_79 = table(sequence_rows(1, CLASS_ALERT, [HIGH] * 10 + [LOW] * 10 + [HIGH] * 59, k_start=1))
    row = only_row(alarm_videos(until_79, rules_config, DROWSY_SOURCE_MODEL))
    assert (row["onsets"], row["alarm_steps"]) == (1, 10)  # 판정 상태: 스텝 10~19

    until_80 = table(sequence_rows(1, CLASS_ALERT, [HIGH] * 10 + [LOW] * 10 + [HIGH] * 60, k_start=1))
    row = only_row(alarm_videos(until_80, rules_config, DROWSY_SOURCE_MODEL))
    assert (row["onsets"], row["alarm_steps"]) == (2, 11)  # 스텝 10~19와 스텝 80

    # 0.4 바로 위는 종료 조건이 아니다: 판정이 끝나지 않으므로 시작은 한 번이고 끝까지 판정 상태다
    never_ends = table(sequence_rows(1, CLASS_ALERT, [HIGH] * 10 + [0.41] * 10 + [HIGH] * 60, k_start=1))
    row = only_row(alarm_videos(never_ends, rules_config, DROWSY_SOURCE_MODEL))
    assert (row["onsets"], row["alarm_steps"]) == (1, 71)


def test_alarm_perclos_source(rules_config: RulesConfig) -> None:
    """perclos_60 ≥ 0.15가 10스텝이면 시작, < 0.10이 10스텝이면 종료, 재시작 금지 시간은 없다.

    0.15 ×10 (스텝 10에서 시작) → 0.10 ×10 (0.10은 "미만"이 아니라 끝나지 않음) → 0.09 ×10 (스텝 30에서 종료)
    → 0.15 ×10 (스텝 40에서 다시 시작). 판정 상태: 스텝 10~29와 스텝 40 = 21스텝.
    """
    ratios = [0.15] * 10 + [0.10] * 10 + [0.09] * 10 + [0.15] * 10
    preds = table(sequence_rows(1, CLASS_ALERT, [0.0] * 40, perclos=ratios, k_start=1))
    row = only_row(alarm_videos(preds, rules_config, DROWSY_SOURCE_PERCLOS))
    assert (row["onsets"], row["alarm_steps"]) == (2, 21)
    # D9의 perclos 쪽: perclos_60 ≥ 0.15인 창 20개
    assert row["flagged_windows"] == 20
    # 모델 쪽에서는 아무 일도 없다
    model = only_row(alarm_videos(preds, rules_config, DROWSY_SOURCE_MODEL))
    assert (model["onsets"], model["alarm_steps"], model["flagged_windows"]) == (0, 0, 0)

    just_below = table(sequence_rows(1, CLASS_ALERT, [0.0] * 20, perclos=[0.149] * 20))
    assert only_row(alarm_videos(just_below, rules_config, DROWSY_SOURCE_PERCLOS))["onsets"] == 0


def test_alarm_either_source(rules_config: RulesConfig) -> None:
    """either는 두 판정기 중 하나라도 판정 중인 상태다.

    떨어진 경우: 모델 높음 10(스텝 10 시작) → 낮음 30(스텝 20 종료). perclos는 스텝 26~40이 0.2(스텝 35 시작).
      model 1번, perclos 1번, either 2번(스텝 10~19와 35~40: 16스텝).
    겹친 경우: 모델은 40스텝 내내 높고 perclos는 스텝 6부터 0.2 → either는 스텝 10에 한 번 시작해 이어진다.
    """
    apart = table(
        sequence_rows(1, CLASS_ALERT, [HIGH] * 10 + [LOW] * 30, perclos=[0.0] * 25 + [0.2] * 15, k_start=1)
    )
    onsets = {s: only_row(alarm_videos(apart, rules_config, s)) for s in DROWSY_SOURCES}
    assert onsets[DROWSY_SOURCE_MODEL]["onsets"] == 1 and onsets[DROWSY_SOURCE_MODEL]["alarm_steps"] == 10
    assert onsets[DROWSY_SOURCE_PERCLOS]["onsets"] == 1 and onsets[DROWSY_SOURCE_PERCLOS]["alarm_steps"] == 6
    assert onsets[DROWSY_SOURCE_EITHER]["onsets"] == 2 and onsets[DROWSY_SOURCE_EITHER]["alarm_steps"] == 16
    # D9: 모델은 p_drowsy 0.6인 창 10개, perclos는 0.2인 창 15개, either는 둘의 합집합 25개
    assert [int(onsets[s]["flagged_windows"]) for s in DROWSY_SOURCES] == [10, 15, 25]

    overlapping = table(
        sequence_rows(1, CLASS_ALERT, [HIGH] * 40, perclos=[0.0] * 5 + [0.2] * 35, k_start=1)
    )
    row = only_row(alarm_videos(overlapping, rules_config, DROWSY_SOURCE_EITHER))
    assert (row["onsets"], row["alarm_steps"]) == (1, 31)


def test_alarm_report_pools_videos_and_uses_video_table_denominators(rules_config: RulesConfig) -> None:
    """alert 영상 셋(하나는 창 없음), drowsy 영상 둘, low_vigilant 영상 하나.

    S01_0 : 높음 10 + 중간 20 → 시작 1번, 판정 상태 21스텝, drowsy로 분류된 창 10개
    S02_0 : 중간 30            → 시작 없음
    S03_0 : 창 없음            → 분모(영상 수)에만 들어간다
    S01_10: 높음 30 → 판정 남. S02_10: 중간 30 → 판정 없음. S01_5: 높음 30(어느 수치에도 들어가지 않는다)
    alert 평가 시간 60 s = 1/60 h → 시간당 60번. 판정이 난 alert 영상 1/3. 판정 상태 시간 비율 21/60.
    창 단위 값 10 / (1/60) = 600. drowsy 영상 중 판정이 난 비율 1/2.
    """
    preds = table(
        sequence_rows(1, CLASS_ALERT, [HIGH] * 10 + [MID] * 20),
        sequence_rows(2, CLASS_ALERT, [MID] * 30),
        sequence_rows(1, CLASS_DROWSY, [HIGH] * 30),
        sequence_rows(2, CLASS_DROWSY, [MID] * 30),
        sequence_rows(1, CLASS_LOW_VIGILANT, [HIGH] * 30),
    )
    videos = video_rows(
        (1, CLASS_ALERT, True),
        (2, CLASS_ALERT, True),
        (3, CLASS_ALERT, False),
        (1, CLASS_DROWSY, True),
        (2, CLASS_DROWSY, True),
        (1, CLASS_LOW_VIGILANT, True),
    )
    report = alarm_report(preds, rules_config, DROWSY_SOURCE_MODEL, videos)
    assert report.n_alert_videos == 3 and report.n_drowsy_videos == 2
    assert report.alert_onsets == 1
    assert report.alert_hours == pytest.approx(1 / 60)
    assert report.onsets_per_hour == pytest.approx(60.0)
    assert report.alert_videos_with_onset == 1
    assert report.alert_video_onset_ratio == pytest.approx(1 / 3)
    assert report.alarm_time_share == pytest.approx(21 / 60)
    assert report.alert_flagged_windows == 10
    assert report.flagged_windows_per_hour == pytest.approx(600.0)
    assert report.drowsy_videos_with_onset == 1
    assert report.drowsy_video_onset_ratio == pytest.approx(1 / 2)

    empty = report.videos[report.videos["video_id"] == "S03_0"].iloc[0]
    assert (empty["n_steps"], empty["evaluated_ms"], empty["onsets"], empty["alarm_steps"]) == (0, 0, 0, 0)
    assert not empty["gate_pass"]

    # 영상 표가 없으면 창이 있는 alert 영상 둘만 분모다. fold 하나만 떼어 낸 값은 alarm_summary로 구한다
    assert alarm_report(preds, rules_config, DROWSY_SOURCE_MODEL).alert_video_onset_ratio == pytest.approx(1 / 2)
    summary = alarm_summary(report.videos[report.videos["subject"] == 1])
    assert summary["onsets"] == 1.0 and summary["alert_hours"] == pytest.approx(30 / 3600)
    assert summary["onsets_per_hour"] == pytest.approx(120.0) and summary["drowsy_video_onset_ratio"] == 1.0


def test_alarm_detector_restarts_for_each_part_of_a_split_video(rules_config: RulesConfig) -> None:
    # 부분 1의 끝 6스텝과 부분 2의 처음 6스텝이 높다. 세션마다 판정기가 새로 시작하므로 12개 연속이 아니다
    split = table(
        sequence_rows(32, CLASS_DROWSY, [MID] * 20 + [HIGH] * 6, part=1),
        sequence_rows(32, CLASS_DROWSY, [HIGH] * 6 + [MID] * 20, part=2),
    )
    row = only_row(alarm_videos(split, rules_config, DROWSY_SOURCE_MODEL))
    assert (row["onsets"], row["n_steps"]) == (0, 52)

    # 같은 수열이 한 세션이면 판정이 난다
    whole = table(sequence_rows(32, CLASS_DROWSY, [MID] * 20 + [HIGH] * 12 + [MID] * 20))
    assert only_row(alarm_videos(whole, rules_config, DROWSY_SOURCE_MODEL))["onsets"] == 1

    # 부분마다 판정이 나면 논리 영상의 시작 횟수는 그 합이다
    both = table(
        sequence_rows(32, CLASS_DROWSY, [HIGH] * 12, part=1), sequence_rows(32, CLASS_DROWSY, [HIGH] * 12, part=2)
    )
    assert only_row(alarm_videos(both, rules_config, DROWSY_SOURCE_MODEL))["onsets"] == 2


def test_alarm_step_gap_resets_the_consecutive_count(rules_config: RulesConfig) -> None:
    # 스텝 번호가 건너뛰면(10초 넘는 끊김) 연속 수가 0으로 돌아간다: 6개 + 6개는 판정이 아니다
    ks = list(range(240, 246)) + list(range(260, 266))
    gap = table(sequence_rows(1, CLASS_ALERT, [HIGH] * 12, ks=ks))
    assert only_row(alarm_videos(gap, rules_config, DROWSY_SOURCE_MODEL))["onsets"] == 0
    # 행의 순서는 상관없다(k 순으로 넣는다)
    shuffled = table(sequence_rows(1, CLASS_ALERT, [MID] * 5 + [HIGH] * 10)).sample(frac=1.0, random_state=0)
    assert only_row(alarm_videos(shuffled, rules_config, DROWSY_SOURCE_MODEL))["onsets"] == 1


def test_alarm_restarts_where_the_step_number_jumps_inside_a_session(rules_config: RulesConfig) -> None:
    """세션 안에서 k가 건너뛰면(그 사이에 10초 넘는 끊김) 판정이 거기서 끝나고 두 판정기가 처음부터 다시 센다.

    앱의 규칙 엔진이 그 끊김 뒤 첫 프레임에서 하는 일과 같다(2026-11-13 소유자 결정 4). 기댓값은 모두 손으로 센 것이고,
    괄호 안은 판정기를 되돌리지 않고 연속 수만 0으로 하던 예전 동작의 값이다(이 테스트가 그 동작에서는 실패한다).
    """
    jump = list(range(240, 252)) + list(range(500, 512))

    # (1) 끊기는 순간 판정 중: 높음 12(스텝 249에서 시작, 249~251의 3스텝) | 건너뜀 | 높음 12(스텝 509에서 다시 시작,
    #     509~511의 3스텝). 시작 2번, 판정 상태 6스텝. (예전: 끊김 뒤 12행이 모두 판정 상태 -> 1번, 15스텝)
    still_high = table(sequence_rows(1, CLASS_ALERT, [HIGH] * 24, ks=jump))
    row = only_row(alarm_videos(still_high, rules_config, DROWSY_SOURCE_MODEL))
    assert (row["onsets"], row["alarm_steps"], row["n_steps"]) == (2, 6, 24)

    # (2) 끊김 뒤에는 낮음 5행뿐: 판정은 끊김에서 끝났으므로 그 5행은 판정 상태가 아니다. 1번, 3스텝. (예전: 1번, 8스텝)
    then_low = table(sequence_rows(1, CLASS_ALERT, [HIGH] * 12 + [0.05] * 5, ks=jump[:17]))
    row = only_row(alarm_videos(then_low, rules_config, DROWSY_SOURCE_MODEL))
    assert (row["onsets"], row["alarm_steps"]) == (1, 3)

    # (3) 재시작 금지 기억도 지워진다: 높음 10(스텝 10 시작) → 낮음 10(스텝 20 종료, 판정 상태 10~19) | 건너뜀 |
    #     높음 10(스텝 31~40). 종료 뒤 20초밖에 지나지 않았지만 새로 시작한 판정기라 스텝 40에서 시작한다.
    #     2번, 11스텝. (예전: 60초 금지에 걸려 1번, 10스텝)
    ks = list(range(1, 21)) + list(range(31, 41))
    refractory = table(sequence_rows(1, CLASS_ALERT, [HIGH] * 10 + [LOW] * 10 + [HIGH] * 10, ks=ks))
    row = only_row(alarm_videos(refractory, rules_config, DROWSY_SOURCE_MODEL))
    assert (row["onsets"], row["alarm_steps"]) == (2, 11)
    # 건너뛰지 않은 같은 수열(스텝 21~30)은 금지에 걸린다: 되돌리는 것은 건너뜀이지 수열이 아니다
    contiguous = table(sequence_rows(1, CLASS_ALERT, [HIGH] * 10 + [LOW] * 10 + [HIGH] * 10, k_start=1))
    row = only_row(alarm_videos(contiguous, rules_config, DROWSY_SOURCE_MODEL))
    assert (row["onsets"], row["alarm_steps"]) == (1, 10)

    # (4) PERCLOS 판정기도 같이 되돌린다: perclos_60 0.2가 12행(스텝 249 시작, 3스텝) | 건너뜀 | 0.2가 5행(10행이 안 돼
    #     시작하지 않는다). perclos와 either 모두 1번, 3스텝. (예전: 1번, 8스텝)
    ratios = table(sequence_rows(1, CLASS_ALERT, [MID] * 17, perclos=[0.2] * 17, ks=jump[:17]))
    for source in (DROWSY_SOURCE_PERCLOS, DROWSY_SOURCE_EITHER):
        row = only_row(alarm_videos(ratios, rules_config, source))
        assert (row["onsets"], row["alarm_steps"]) == (1, 3)
    # either에서 한쪽만 판정 중이어도 끊김에서 끝난다: 모델이 스텝 249부터 판정 중, 끊김 뒤 perclos 0.2가 12행
    # (스텝 509에서 시작, 3스텝). 2번, 6스텝. (예전: 모델 판정이 이어져 1번, 15스텝)
    mixed = table(
        sequence_rows(1, CLASS_ALERT, [HIGH] * 12 + [MID] * 12, perclos=[0.0] * 12 + [0.2] * 12, ks=jump)
    )
    row = only_row(alarm_videos(mixed, rules_config, DROWSY_SOURCE_EITHER))
    assert (row["onsets"], row["alarm_steps"]) == (2, 6)

    # (5) 건너뛴 크기는 보지 않는다(가정): 스텝 하나만 빠져도(251 → 253) 같은 다시 시작이다
    one_missing = table(sequence_rows(1, CLASS_ALERT, [HIGH] * 24, ks=list(range(240, 252)) + list(range(253, 265))))
    row = only_row(alarm_videos(one_missing, rules_config, DROWSY_SOURCE_MODEL))
    assert (row["onsets"], row["alarm_steps"]) == (2, 6)

    # (6) 끊김 뒤 첫 행에서 곧바로 다시 시작해도 새 시작으로 센다(앱은 끊김에서 `end`를 내고 새 `onset`을 낸다).
    #     onset_steps를 1로 바꾸면 높음 24행이 모두 판정 상태이고 시작은 건너뜀 앞뒤로 한 번씩이다. 2번, 24스텝.
    one_step = apply_overrides(rules_config, {"drowsiness.model.onset_steps": 1})
    row = only_row(alarm_videos(still_high, one_step, DROWSY_SOURCE_MODEL))
    assert (row["onsets"], row["alarm_steps"]) == (2, 24)

    # (7) 다른 세션의 건너뜀은 이 세션의 판정을 건드리지 않는다: 영상 둘, 건너뜀은 피험자 2에만 있다
    two = table(
        sequence_rows(1, CLASS_ALERT, [HIGH] * 24),
        sequence_rows(2, CLASS_ALERT, [HIGH] * 24, ks=jump),
    )
    result = alarm_videos(two, rules_config, DROWSY_SOURCE_MODEL).set_index("subject")
    assert (result.loc[1, "onsets"], result.loc[1, "alarm_steps"]) == (1, 15)  # 스텝 249~263
    assert (result.loc[2, "onsets"], result.loc[2, "alarm_steps"]) == (2, 6)
    # 시간당 시작 횟수: 3번 / (48스텝 = 48/3600 h) = 225
    summary = alarm_summary(alarm_videos(two, rules_config, DROWSY_SOURCE_MODEL))
    assert summary["onsets_per_hour"] == pytest.approx(225.0)


def test_alarm_rejects_tables_that_cannot_be_evaluated(rules_config: RulesConfig) -> None:
    stride_5 = table(sequence_rows(1, CLASS_ALERT, [HIGH] * 20, ks=range(240, 340, 5)))
    with pytest.raises(ValueError, match="evaluation stride 1"):
        alarm_videos(stride_5, rules_config, DROWSY_SOURCE_MODEL)
    preds = table(sequence_rows(1, CLASS_ALERT, [HIGH] * 20))
    with pytest.raises(ValueError, match="source must be one of"):
        alarm_videos(preds, rules_config, "rule")
    with pytest.raises(ValueError, match="onset_p must be within"):
        alarm_videos(preds, rules_config, DROWSY_SOURCE_MODEL, onset_p=0.4)
    with pytest.raises(ValueError, match="does not apply to source 'perclos'"):
        alarm_report(preds, rules_config, DROWSY_SOURCE_PERCLOS, sweep=(0.5,))


def test_alarm_threshold_sweep(rules_config: RulesConfig) -> None:
    # alert 영상의 p_drowsy가 0.65로 30스텝: 임계값 0.5와 0.6에서는 판정이 나고 0.7에서는 나지 않는다
    preds = table(sequence_rows(1, CLASS_ALERT, [0.65] * 30), sequence_rows(1, CLASS_DROWSY, [0.95] * 30))
    sweep = alarm_sweep(preds, rules_config, (0.5, 0.6, 0.7))
    assert list(sweep.columns) == list(SWEEP_COLUMNS)
    assert sweep["onset_p"].tolist() == [0.5, 0.6, 0.7]
    assert sweep["onsets"].tolist() == [1.0, 1.0, 0.0]
    assert sweep["onsets_per_hour"].tolist() == pytest.approx([120.0, 120.0, 0.0])
    assert sweep["alarm_time_share"].tolist() == pytest.approx([21 / 30, 21 / 30, 0.0])
    assert sweep["drowsy_video_onset_ratio"].tolist() == [1.0, 1.0, 1.0]
    # 창 단위 값은 임계값과 무관하다(분류 결과로 센다)
    assert sweep["flagged_windows_per_hour"].nunique() == 1

    report = alarm_report(preds, rules_config, DROWSY_SOURCE_MODEL, sweep=(0.5, 0.7))
    pd.testing.assert_frame_equal(report.sweep, sweep.iloc[[0, 2]].reset_index(drop=True))
    assert report.alert_onsets == 1  # 주 수치는 설정의 임계값 0.6이다
    # either에서는 임계값이 모델 쪽 판정기만 바꾼다. 임계값이 없으면 빈 표다(perclos도 받는다)
    assert alarm_sweep(preds, rules_config, (0.5, 0.7), DROWSY_SOURCE_EITHER)["onsets"].tolist() == [1.0, 0.0]
    assert len(alarm_sweep(preds, rules_config, (), DROWSY_SOURCE_PERCLOS)) == 0


def test_alarm_matches_rule_engine_on_step(rules_config: RulesConfig) -> None:
    """세 판정원 모두에서, 같은 수열을 규칙 엔진의 `on_step`에 넣었을 때와 시작 횟수·판정 상태 스텝 수가 같다."""
    rng = np.random.default_rng(7)
    n_steps = 600
    k = np.arange(240, 240 + n_steps)
    sessions = []
    for subject in (1, 2, 3):
        phase = rng.uniform(0.0, 2.0 * math.pi, size=2)
        p = np.clip(0.5 + 0.45 * np.sin(2.0 * math.pi * k / 97.0 + phase[0]) + rng.normal(0, 0.03, n_steps), 0, 1)
        ratio = np.clip(0.125 + 0.1 * np.sin(2.0 * math.pi * k / 61.0 + phase[1]), 0, 1)
        sessions.append(sequence_rows(subject, CLASS_ALERT, p.tolist(), perclos=ratio.tolist()))
    preds = table(*sessions)

    for source in DROWSY_SOURCES:
        expected: dict[int, tuple[int, int]] = {}
        for session in sessions:
            engine = RuleEngine(apply_overrides(rules_config, {"drowsiness.source": source}))
            onsets = active_steps = 0
            active = False
            for row in session:
                values = (row["perclos_60"],) + (0.0,) * (N_FEATURES - 1)
                step = StepFeatures(k=row["k"], t_end_ms=row["t_end_ms"], full_lookback=True, values=values)
                for event in engine.on_step(step, (row["p_alert"], row["p_low_vigilant"], row["p_drowsy"])):
                    if event.phase == PHASE_ONSET:
                        onsets += 1
                        active = True
                    elif event.phase == PHASE_END:
                        active = False
                active_steps += int(active)
            expected[session[0]["subject"]] = (onsets, active_steps)

        result = alarm_videos(preds, rules_config, source)
        got = {
            int(row["subject"]): (int(row["onsets"]), int(row["alarm_steps"])) for _, row in result.iterrows()
        }
        assert got == expected
        # 수열이 실제로 판정을 여러 번 냈는지(시작, 종료, 재시작 금지 시간이 모두 걸렸는지) 확인한다
        assert sum(onsets for onsets, _ in expected.values()) >= 6
        assert all(0 < steps < n_steps for _, steps in expected.values())


@pytest.mark.parametrize("stalled", [False, True])
def test_alarm_matches_the_app_when_frames_stall_across_step_boundaries(
    pipeline_spec: PipelineSpec, rules_config: RulesConfig, stalled: bool
) -> None:
    """앱(`DriverPipeline` + 고정 확률 모델)과 오프라인 경보 평가가 같은 프레임에서 같은 졸음 시작을 낸다.

    2026-11-13 소유자 결정(밀린 스텝마다 모델을 부른다)의 회귀 테스트다. 결정 전에는 프레임 하나가 스텝을 둘 이상
    발행하면 마지막 스텝에만 확률이 있어서, 아래의 끊김이 있는 세션에서 앱은 시작 0번, 이 평가는 1번이었다.

    세션: 480초, 30 fps, 눈을 뜬 채. 끊김이 있는 쪽은 210초부터 9초마다 [t + 0.7초, t + 2.2초)의 프레임이 없다
    (1.5초, 스텝 경계 t + 1초와 t + 2초를 넘는다. 30번). 모델은 언제나 p_drowsy 0.95를 낸다.
    손 계산: T_b = 60초 -> 스텝 61~479, 조회 구간이 찬 스텝 120~479, 창이 찬 스텝 239~479(241개).
    0.95가 스텝 239~248에서 열 개 연속 -> 스텝 248에서 시작하고 끝까지 이어진다 -> 시작 1번, 판정 상태 248~479의 232스텝.
    """
    rules = apply_overrides(rules_config, {"drowsiness.source": DROWSY_SOURCE_MODEL})
    gaps = tuple((1000 * t + 700, 1000 * t + 2200) for t in range(210, 480, 9)) if stalled else ()
    frames = synth.render(synth.Scenario(segments=(), duration_ms=480_000, fps=30.0, gaps=gaps), pipeline_spec)
    probs = (0.02, 0.03, 0.95)
    pipe = DriverPipeline(pipeline_spec, rules, model=lambda window: probs)

    steps: list[StepFeatures] = []
    rows_with_probs: list[int] = []
    app_onset_steps: list[int] = []
    app_alarm_steps = 0
    active = False
    multi_step_frames = 0
    for frame in frames:
        out = pipe.update(frame)
        assert out is not None
        multi_step_frames += len(out.steps) > 1
        drowsy = [event for event in out.events if event.type == EVENT_DROWSY]
        for step, step_probs in zip(out.steps, out.step_probs):
            # 스텝 이벤트의 발행 시각(t_start_ms + duration_ms)은 그 스텝의 끝이다.
            for event in drowsy:
                if event.t_start_ms + event.duration_ms == step.t_end_ms:
                    active = event.phase == PHASE_ONSET
                    if active:
                        app_onset_steps.append(step.k)
            if step_probs is not None:
                rows_with_probs.append(step.k)
                app_alarm_steps += active
            steps.append(step)
    assert multi_step_frames == (30 if stalled else 0)
    assert [step.k for step in steps] == list(range(61, 480))
    assert rows_with_probs == list(range(239, 480))
    assert pipe.model_calls == 241
    assert (app_onset_steps, app_alarm_steps) == ([248], 232)

    data: dict[str, Any] = {
        "video_id": "S01_0",
        "session_id": "S01_0",
        "subject": 1,
        "fold": 1,
        "label": 0,
        "class_index": CLASS_ALERT,
        "part": 0,
        "k": [step.k for step in steps],
        "t_end_ms": [step.t_end_ms for step in steps],
        "full_lookback": [step.full_lookback for step in steps],
    }
    for index, name in enumerate(FEATURE_NAMES):
        data[name] = [step.values[index] for step in steps]
    step_table = pd.DataFrame(data, columns=list(STEP_COLUMNS))
    windows = make_windows(step_table, window_steps=120, stride=1)
    videos = video_rows((1, CLASS_ALERT, True))
    preds = prediction_table(
        windows, np.tile(probs, (len(windows), 1)), step_table, videos, run_id="r", family="tcn", arm="a", seed=0
    )
    assert preds["k"].tolist() == rows_with_probs  # 평가의 행 = 앱이 모델을 부른 스텝
    offline = only_row(alarm_videos(preds, rules, DROWSY_SOURCE_MODEL, videos))
    assert (offline["n_steps"], offline["onsets"], offline["alarm_steps"]) == (241, 1, 232)
    assert (offline["onsets"], offline["alarm_steps"]) == (len(app_onset_steps), app_alarm_steps)


def _app_drowsy_run(
    frames: Sequence[Any], spec: PipelineSpec, rules: RulesConfig, p_drowsy_at: Any
) -> tuple[list[StepFeatures], dict[int, tuple[float, float, float]], list[tuple[str, str, int]], int]:
    """프레임을 `DriverPipeline`에 넣고 (스텝, 스텝 번호 → 모델 확률, drowsy 이벤트, 판정 상태 스텝 수)를 돌려준다.

    모델은 `p_drowsy_at(그 스텝을 발행시킨 프레임의 t_ms)`를 p_drowsy로 낸다. drowsy 이벤트는 (단계, 어디서, 발행
    시각)이고 어디서는 `step`(스텝의 판정) 또는 `reset`(10초 넘는 끊김 뒤 첫 프레임의 초기화)이다. 판정 상태 스텝 수는
    모델 확률을 받은 스텝 가운데 그 스텝을 처리한 뒤 졸음 상태였던 것의 수다(`alarm_videos`의 alarm_steps와 같은 정의).
    """
    now = {"t_ms": 0}

    def model(window: Any) -> tuple[float, float, float]:
        p = p_drowsy_at(now["t_ms"])
        return ((1.0 - p) / 2.0, (1.0 - p) / 2.0, p)

    pipe = DriverPipeline(spec, rules, model=model)
    steps: list[StepFeatures] = []
    probs: dict[int, tuple[float, float, float]] = {}
    events: list[tuple[str, str, int]] = []
    active = False
    alarm_steps = 0
    for frame in frames:
        now["t_ms"] = frame.t_ms
        out = pipe.update(frame)
        assert out is not None
        drowsy = [event for event in out.events if event.type == EVENT_DROWSY]
        at_step = [False] * len(drowsy)
        for step, step_probs in zip(out.steps, out.step_probs):
            for index, event in enumerate(drowsy):  # 스텝 이벤트의 발행 시각은 그 스텝의 끝이다
                if event.t_start_ms + event.duration_ms == step.t_end_ms:
                    at_step[index] = True
                    active = event.phase == PHASE_ONSET
                    events.append((event.phase, "step", step.t_end_ms))
            if step_probs is not None:
                probs[step.k] = step_probs
                alarm_steps += active
            steps.append(step)
        for index, event in enumerate(drowsy):  # 초기화가 끝낸 drowsy: 발행 시각이 그 프레임의 t_ms다
            if not at_step[index]:
                assert event.phase == PHASE_END and event.t_start_ms + event.duration_ms == frame.t_ms
                active = False
                events.append((event.phase, "reset", frame.t_ms))
    assert pipe.model_calls == len(probs) and pipe.nonfinite_windows == 0
    return steps, probs, events, alarm_steps


def _offline_alarm(
    steps: Sequence[StepFeatures], probs: dict[int, tuple[float, float, float]], rules: RulesConfig, source: str
) -> tuple[pd.Series, list[int]]:
    """같은 스텝으로 스텝 표 → 보폭 1의 창 → 예측 표(앱이 그 스텝에 쓴 확률)를 만들어 `alarm_videos`에 넣는다."""
    data: dict[str, Any] = {
        "video_id": "S01_0",
        "session_id": "S01_0",
        "subject": 1,
        "fold": 1,
        "label": 0,
        "class_index": CLASS_ALERT,
        "part": 0,
        "k": [step.k for step in steps],
        "t_end_ms": [step.t_end_ms for step in steps],
        "full_lookback": [step.full_lookback for step in steps],
    }
    for index, name in enumerate(FEATURE_NAMES):
        data[name] = [step.values[index] for step in steps]
    step_table = pd.DataFrame(data, columns=list(STEP_COLUMNS))
    windows = make_windows(step_table, window_steps=120, stride=1)
    rows = windows.meta["k_end"].tolist()
    videos = video_rows((1, CLASS_ALERT, True))
    preds = prediction_table(
        windows, np.array([probs[k] for k in rows]), step_table, videos, run_id="r", family="tcn", arm="a", seed=0
    )
    return only_row(alarm_videos(preds, rules, source, videos)), rows


@pytest.mark.parametrize(
    ("p_after", "app_events", "expected"),
    [
        # 끊김 뒤에도 0.95: 스텝 491~500에서 다시 열 개 -> 스텝 500에서 두 번째 시작, 500~699의 200스텝
        (0.95, [("onset", "step", 248_001), ("end", "reset", 312_001), ("onset", "step", 500_001)], (2, 252)),
        # 끊김 뒤에는 0.05: 다시 시작하지 않는다. 끊김 뒤의 209행은 판정 상태가 아니다
        (0.05, [("onset", "step", 248_001), ("end", "reset", 312_001)], (1, 52)),
    ],
)
def test_alarm_matches_the_app_across_a_gap_longer_than_the_reset_gap(
    pipeline_spec: PipelineSpec,
    rules_config: RulesConfig,
    p_after: float,
    app_events: list[tuple[str, str, int]],
    expected: tuple[int, int],
) -> None:
    """10초 넘는 끊김이 있는 세션에서 앱(`DriverPipeline` + 고정 확률 모델)과 오프라인 경보 평가가 같다.

    2026-11-13 소유자 결정 4(10초 넘는 끊김 뒤 규칙 엔진이 다시 시작한다)의 후속 회귀 테스트다. 고치기 전의
    `alarm_videos`는 이 세션에서 (시작 1번, 판정 상태 261스텝)과 (1번, 61스텝)을 냈다.

    세션: 700초, 30 fps, 눈을 뜬 채, 세션 시간 [300초, 312초)의 프레임이 없다. 첫 프레임의 t_ms는 1이다.
    손 계산
    - 끊김 전: T_b = 60초 -> 스텝 61~299. 조회 구간이 찬 스텝 120~299, 창이 찬 스텝 239~299(61개).
      p_drowsy 0.95가 스텝 239~248에서 열 개 연속 -> 스텝 248(끝 시각 248 001)에서 시작, 248~299의 52스텝이 판정 상태.
    - 끊김: 마지막 프레임은 세션 시간 299 967 ms, 다음 프레임은 312 000 ms(t_ms 312 001). 간격 12 033 ms > 10 000 ms.
      그 프레임에서 규칙 엔진이 drowsy를 끝내고(발행 시각 312 001) 판정기를 되돌린다. 누적기는 스텝 313부터 다시 센다.
    - 끊김 뒤: 스텝 313~699. 조회 구간이 찬 스텝 372~699(312초 + 60초부터), 창이 찬 스텝 491~699(209개).
    - 평가의 행: 61 + 209 = 270개. 행의 k는 299에서 491로 건너뛴다.
    """
    rules = apply_overrides(rules_config, {"drowsiness.source": DROWSY_SOURCE_MODEL})
    scenario = synth.Scenario(segments=(), duration_ms=700_000, fps=30.0, gaps=((300_000, 312_000),))
    frames = synth.render(scenario, pipeline_spec)
    gap_ms = max(b.t_ms - a.t_ms for a, b in zip(frames, frames[1:]))
    assert gap_ms == 12_033 and gap_ms > pipeline_spec.clock.reset_gap_ms

    steps, probs, events, app_alarm_steps = _app_drowsy_run(
        frames, pipeline_spec, rules, lambda t_ms: 0.95 if t_ms <= 300_000 else p_after
    )
    assert [step.k for step in steps] == list(range(61, 300)) + list(range(313, 700))
    assert sorted(probs) == list(range(239, 300)) + list(range(491, 700))
    assert events == app_events
    app_onsets = sum(1 for phase, _, _ in events if phase == PHASE_ONSET)
    assert (app_onsets, app_alarm_steps) == expected

    offline, rows = _offline_alarm(steps, probs, rules, DROWSY_SOURCE_MODEL)
    assert rows == sorted(probs)  # 평가의 행 = 앱이 모델을 부른 스텝
    assert (offline["n_steps"], offline["onsets"], offline["alarm_steps"]) == (270, *expected)


def test_alarm_either_matches_the_app_when_a_perclos_state_meets_a_long_gap(
    pipeline_spec: PipelineSpec, rules_config: RulesConfig
) -> None:
    """either 판정원: PERCLOS로 열린 졸음 상태가 20분 끊김을 만나는 세션에서 앱과 오프라인 평가가 같다.

    세션(10 fps, 첫 프레임 t_ms 1): 세션 시간 0~520초, 20분 끊김(520~1720초), 1720~2300초.
    - 모델: 세션 시간 (300초, 340초]의 프레임이 발행한 스텝 301~340에서 p_drowsy 0.95, 그 밖에는 0.05.
      손 계산: 스텝 310에서 시작(301~310이 열 개), 341~350의 낮음 열 개로 스텝 350에서 종료.
    - 감김 묶음 둘(1.1초 감김을 4초마다, 감긴 비율 약 0.29): 330~520초(끊김까지 이어진다)와 1990~2100초.
      둘 다 그 구간의 첫 행(스텝 239, 끊김 뒤에는 1721 + 178 = 1899)보다 90초 넘게 뒤에 시작하므로, 앱의 PERCLOS
      판정기가 첫 행 전에 값을 받기 시작하는 것이 결과를 바꾸지 않는다(`alarm_videos`의 설명 "앱과의 관계").
    - 앱의 drowsy 이벤트: 모델 시작(310) · 종료(350), PERCLOS 시작, 끊김 뒤 첫 프레임의 초기화 종료(t_ms 1 720 001),
      PERCLOS 시작, 스텝 종료. PERCLOS의 스텝 번호는 손으로 세지 않고 앱과 평가가 같은지만 본다.
    고치기 전의 `alarm_videos`는 끊김 뒤의 행을 곧바로 판정 상태로 세어 이 세션에서 앱과 달랐다.
    """
    rules = apply_overrides(rules_config, {"drowsiness.source": DROWSY_SOURCE_EITHER})
    closures = [synth.Closure(1100, start_ms=start) for start in range(330_000, 520_000, 4_000)]
    closures += [synth.Closure(1100, start_ms=start) for start in range(1_990_000, 2_100_000, 4_000)]
    scenario = synth.Scenario(
        segments=tuple(closures), duration_ms=2_300_000, fps=10.0, gaps=((520_000, 1_720_000),)
    )
    frames = synth.render(scenario, pipeline_spec)
    steps, probs, events, app_alarm_steps = _app_drowsy_run(
        frames, pipeline_spec, rules, lambda t_ms: 0.95 if 300_001 < t_ms <= 340_001 else 0.05
    )
    assert sorted(probs) == list(range(239, 520)) + list(range(1899, 2300))
    assert [(phase, where) for phase, where, _ in events] == [
        ("onset", "step"), ("end", "step"), ("onset", "step"), ("end", "reset"), ("onset", "step"), ("end", "step"),
    ]  # fmt: skip
    assert events[0] == ("onset", "step", 310_001) and events[1] == ("end", "step", 350_001)
    assert events[3] == ("end", "reset", 1_720_001)
    assert 360_001 < events[2][2] < 400_001 and 2_020_001 < events[4][2] < 2_060_001  # PERCLOS 시작(묶음 시작 + 30~70초)
    assert app_alarm_steps > 40 + 100 + 60  # 모델 40스텝, 끊김 전 PERCLOS 100스텝 이상, 끊김 뒤 60스텝 이상

    offline, rows = _offline_alarm(steps, probs, rules, DROWSY_SOURCE_EITHER)
    assert rows == sorted(probs)
    assert (offline["onsets"], offline["alarm_steps"]) == (3, app_alarm_steps)
    # 모델 판정원만 보면 끊김과 무관한 한 번이다: 스텝 310~349의 40스텝
    model_only, _ = _offline_alarm(steps, probs, rules, DROWSY_SOURCE_MODEL)
    assert (model_only["onsets"], model_only["alarm_steps"]) == (1, 40)


# ---------------------------------------------------------------------------
# 부트스트랩
# ---------------------------------------------------------------------------


def _value_table(values: dict[int, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"subject": subject, "value": value} for subject, items in values.items() for value in items]
    )


def _mean_value(frame: pd.DataFrame) -> float:
    return float(frame["value"].mean())


def test_subject_bootstrap_is_deterministic() -> None:
    rng = np.random.default_rng(3)
    frame = _value_table({subject: rng.random(3).tolist() for subject in range(1, 21)})
    first = subject_bootstrap(frame, _mean_value, n=300, seed=11)
    assert first == subject_bootstrap(frame, _mean_value, n=300, seed=11)
    assert first != subject_bootstrap(frame, _mean_value, n=300, seed=12)
    assert first[0] < _mean_value(frame) < first[1]
    # 행의 순서를 바꿔도 같은 구간이다(피험자 번호순으로 추출한다)
    low, high = subject_bootstrap(frame.sample(frac=1.0, random_state=0), _mean_value, n=300, seed=11)
    assert (low, high) == pytest.approx(first)
    # 구간이 좁아지면 수준이 낮은 것이다
    narrow = subject_bootstrap(frame, _mean_value, n=300, seed=11, level=0.5)
    assert first[0] < narrow[0] < narrow[1] < first[1]


def test_subject_bootstrap_hand_computed_interval() -> None:
    # 피험자 둘의 값이 0과 1이면 추출한 평균은 0, 0.5, 1이 1/4, 1/2, 1/4의 확률로 나온다 → 95% 구간은 (0, 1)
    frame = _value_table({1: [0.0], 2: [1.0]})
    assert subject_bootstrap(frame, _mean_value, n=2000, seed=0) == (0.0, 1.0)
    # 50% 구간은 25, 75 백분위수: (0.0~0.5, 0.5~1.0) 사이
    low, high = subject_bootstrap(frame, _mean_value, n=2000, seed=0, level=0.5)
    assert 0.0 <= low <= 0.5 <= high <= 1.0
    # 모두 같은 값이면 구간의 폭이 0이다
    assert subject_bootstrap(_value_table({1: [0.3], 2: [0.3], 3: [0.3]}), _mean_value, n=50) == (0.3, 0.3)


def test_subject_bootstrap_resamples_whole_subjects_and_relabels_them() -> None:
    # 피험자 1은 1행, 피험자 2는 3행 → 추출한 표의 행 수는 2, 4, 6 중 하나다(피험자의 행은 함께 움직인다)
    frame = pd.DataFrame(
        {"subject": [2, 1, 2, 2], "video_id": ["S02_0", "S01_0", "S02_5", "S02_10"], "value": [1.0, 0.0, 1.0, 1.0]}
    )
    seen: list[tuple[int, list[int], int]] = []

    def metric(sample: pd.DataFrame) -> float:
        seen.append(
            (len(sample), sorted(sample["subject"].unique().tolist()), sample.groupby(["subject", "video_id"]).ngroups)
        )
        return float(len(sample))

    subject_bootstrap(frame, metric, n=200, seed=5)
    assert {length for length, _, _ in seen} == {2, 4, 6}
    # subject는 추출 순번으로 바뀐다: 같은 피험자가 두 번 뽑혀도 피험자 둘, 영상은 행 수만큼이다
    assert all(subjects == [0, 1] for _, subjects, _ in seen)
    assert all(videos == length for length, _, videos in seen)


def test_subject_bootstrap_accepts_a_prediction_table(small_table: pd.DataFrame) -> None:
    def accuracy(sample: pd.DataFrame) -> float:
        return video_metrics(video_vote(sample)).accuracy

    interval = subject_bootstrap(small_table, accuracy, n=100, seed=1)
    # 피험자 1은 영상 셋을 모두 맞혔고 피험자 32는 하나를 틀렸다: 추출한 정확도는 0, 0.75, 1 중 하나다
    assert interval == (0.0, 1.0)
    assert interval == subject_bootstrap(small_table, accuracy, n=100, seed=1)


def test_paired_subject_bootstrap() -> None:
    rng = np.random.default_rng(4)
    a = _value_table({subject: rng.random(2).tolist() for subject in range(1, 16)})
    # 같은 표끼리는 차이가 정확히 0이다
    assert paired_subject_bootstrap(a, a, _mean_value, n=100, seed=2) == (0.0, 0.0)
    # 모든 행이 0.25씩 크면 어떤 추출에서도 차이가 0.25다(같은 추출을 두 표에 쓴다는 뜻이다)
    b = a.assign(value=a["value"] + 0.25)
    low, high = paired_subject_bootstrap(b, a, _mean_value, n=100, seed=2)
    assert low == pytest.approx(0.25) and high == pytest.approx(0.25)
    # 짝짓지 않으면 구간에 폭이 생긴다
    unpaired = subject_bootstrap(a, _mean_value, n=100, seed=2)
    assert unpaired[1] - unpaired[0] > 0.01
    assert paired_subject_bootstrap(b, a, _mean_value, n=100, seed=2) == paired_subject_bootstrap(
        b, a, _mean_value, n=100, seed=2
    )
    # 표의 행 수는 달라도 되지만 피험자 집합은 같아야 한다
    doubled = pd.concat([a, a], ignore_index=True)
    assert paired_subject_bootstrap(doubled, a, _mean_value, n=50, seed=2) == pytest.approx((0.0, 0.0))
    with pytest.raises(ValueError, match="same subjects"):
        paired_subject_bootstrap(a, a[a["subject"] != 3], _mean_value, n=10)


def test_subject_bootstrap_rejects_bad_input() -> None:
    frame = _value_table({1: [0.0], 2: [1.0]})
    with pytest.raises(ValueError, match="replicates"):
        subject_bootstrap(frame, _mean_value, n=0)
    with pytest.raises(ValueError, match="level"):
        subject_bootstrap(frame, _mean_value, n=10, level=1.0)
    with pytest.raises(ValueError, match="non-finite"):
        subject_bootstrap(frame, lambda sample: math.nan, n=10)
    with pytest.raises(ValueError, match="lacks columns"):
        subject_bootstrap(frame.drop(columns=["subject"]), _mean_value, n=10)
    with pytest.raises(ValueError, match="no subjects"):
        subject_bootstrap(frame.iloc[:0], _mean_value, n=10)


# ---------------------------------------------------------------------------
# 확률 보정
# ---------------------------------------------------------------------------


def test_expected_calibration_error_hand_computed() -> None:
    """구간 5개(폭 0.2).

    신뢰도 0.90 맞음, 0.85 틀림 → 구간 [0.8, 1.0): 정확도 0.5, 평균 신뢰도 0.875 → 차 0.375, 비중 2/4
    신뢰도 0.50 맞음, 0.55 맞음 → 구간 [0.4, 0.6): 정확도 1.0, 평균 신뢰도 0.525 → 차 0.475, 비중 2/4
    ECE = 0.5·0.375 + 0.5·0.475 = 0.425
    """
    probs = np.array([[0.9, 0.05, 0.05], [0.85, 0.1, 0.05], [0.5, 0.3, 0.2], [0.25, 0.55, 0.2]])
    labels = np.array([0, 1, 0, 1])
    assert expected_calibration_error(probs, labels, n_bins=5) == pytest.approx(0.425)
    # 구간이 하나면 |전체 정확도 − 평균 신뢰도| = |0.75 − 0.7|
    assert expected_calibration_error(probs, labels, n_bins=1) == pytest.approx(0.05)


def test_expected_calibration_error_edge_cases() -> None:
    # 신뢰도 0.7인 창 10개 중 7개가 맞으면 보정 오차가 없다
    probs = np.tile([0.7, 0.2, 0.1], (10, 1))
    labels = np.array([0] * 7 + [1] * 3)
    assert expected_calibration_error(probs, labels) == pytest.approx(0.0, abs=1e-12)
    # 신뢰도 1.0은 마지막 구간에 들어간다
    assert expected_calibration_error(np.array([[1.0, 0.0, 0.0]]), np.array([1])) == pytest.approx(1.0)
    assert expected_calibration_error(np.zeros((0, 3)), np.zeros(0, dtype=np.int64)) == 0.0
    with pytest.raises(ValueError, match="probs must be"):
        expected_calibration_error(np.zeros((2, 3)), np.zeros(3))
    with pytest.raises(ValueError, match="n_bins"):
        expected_calibration_error(probs, labels, n_bins=0)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    return shifted / shifted.sum(axis=1, keepdims=True)


def test_untemper_recovers_the_uncalibrated_probabilities() -> None:
    logits = np.random.default_rng(0).normal(0.0, 2.0, size=(50, 3))
    assert np.allclose(untemper(_softmax(logits / 2.5), 2.5), _softmax(logits), atol=1e-12)
    temperatures = np.linspace(0.5, 3.0, 50)
    assert np.allclose(untemper(_softmax(logits / temperatures[:, None]), temperatures), _softmax(logits), atol=1e-12)
    assert np.allclose(untemper(_softmax(logits), 1.0), _softmax(logits), atol=1e-15)
    # 확률이 0인 클래스는 0으로 남는다
    assert untemper(np.array([[0.75, 0.25, 0.0]]), 2.0)[0].tolist() == pytest.approx([0.9, 0.1, 0.0])


def test_calibration_error_before_and_after_temperature() -> None:
    """로짓 (2, 0, 0)에 온도 2를 적용한 창 두 개. 하나는 맞고 하나는 틀리다(정확도 0.5).

    적용 전 신뢰도 e²/(e² + 2) = 0.7870, 적용 후 e/(e + 2) = 0.5761.
    ECE는 적용 전 |0.5 − 0.7870|, 적용 후 |0.5 − 0.5761|.
    """
    after = _softmax(np.array([[2.0, 0.0, 0.0]]) / 2.0)[0]
    probs = (float(after[0]), float(after[1]), float(after[2]))
    preds = table(
        window_rows(1, CLASS_ALERT, [probs], temperature=2.0),
        window_rows(1, CLASS_LOW_VIGILANT, [probs], temperature=2.0),
    )
    before_ece, after_ece = calibration_error(preds)
    assert before_ece == pytest.approx(abs(0.5 - math.e**2 / (math.e**2 + 2.0)))
    assert after_ece == pytest.approx(abs(0.5 - math.e / (math.e + 2.0)))
    assert after_ece < before_ece

    unscaled = preds.assign(temperature=1.0)
    assert calibration_error(unscaled) == (pytest.approx(after_ece), pytest.approx(after_ece))
    assert calibration_error(preds.iloc[:0]) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# D11
# ---------------------------------------------------------------------------


@pytest.fixture()
def gate_cfg(experiment_config: ExperimentConfig) -> ModelGate:
    cfg = experiment_config.evaluation.model_gate
    assert cfg == ModelGate(0.02, 0.70, 2.0, 0.10, 0.50)  # D11의 값
    return cfg


def _boundary_inputs(cfg: ModelGate) -> GateInputs:
    lgbm = 0.55
    return GateInputs(
        macro_f1=lgbm - cfg.max_macro_f1_deficit,
        lgbm_macro_f1=lgbm,
        two_class_video_accuracy=84 / 120,
        false_alarms_per_hour=2.0,
        alert_video_alarm_ratio=6 / 60,
        drowsy_video_alarm_ratio=30 / 60,
    )


def test_model_gate_passes_exactly_at_the_boundaries(gate_cfg: ModelGate) -> None:
    result = model_gate(_boundary_inputs(gate_cfg), gate_cfg)
    assert result.passed and result.source == DROWSY_SOURCE_MODEL
    assert [c.name for c in result.conditions] == [
        "macro_f1",
        "two_class_video_accuracy",
        "false_alarms_per_hour",
        "alert_video_alarm_ratio",
        "drowsy_video_alarm_ratio",
    ]
    assert [c.comparison for c in result.conditions] == [">=", ">=", "<=", "<=", ">="]
    assert [c.threshold for c in result.conditions] == [0.55 - 0.02, 0.70, 2.0, 0.10, 0.50]
    assert [c.value for c in result.conditions] == [0.55 - 0.02, 84 / 120, 2.0, 6 / 60, 30 / 60]
    assert all(c.passed for c in result.conditions)


@pytest.mark.parametrize(
    ("field", "direction"),
    [
        ("macro_f1", -1.0),
        ("two_class_video_accuracy", -1.0),
        ("false_alarms_per_hour", 1.0),
        ("alert_video_alarm_ratio", 1.0),
        ("drowsy_video_alarm_ratio", -1.0),
    ],
)
def test_model_gate_fails_just_past_each_boundary(gate_cfg: ModelGate, field: str, direction: float) -> None:
    boundary = _boundary_inputs(gate_cfg)
    value = getattr(boundary, field)
    past = dataclasses.replace(boundary, **{field: float(np.nextafter(value, value + direction))})
    result = model_gate(past, gate_cfg)
    assert not result.passed and result.source == DROWSY_SOURCE_PERCLOS
    assert [c.name for c in result.conditions if not c.passed] == [field]
    # 반대 방향으로 조금 움직이면 통과다
    inside = dataclasses.replace(boundary, **{field: float(np.nextafter(value, value - direction))})
    assert model_gate(inside, gate_cfg).passed


def test_model_gate_typical_values_and_bad_input(gate_cfg: ModelGate) -> None:
    good = GateInputs(0.60, 0.55, 0.85, 0.5, 0.05, 0.8)
    assert model_gate(good, gate_cfg).passed
    # 조건 하나만 어겨도 판정원은 perclos다. 어긴 조건의 값과 기준이 그대로 실린다
    result = model_gate(dataclasses.replace(good, false_alarms_per_hour=2.5), gate_cfg)
    failed = [c for c in result.conditions if not c.passed]
    assert not result.passed and len(failed) == 1
    assert (failed[0].name, failed[0].value, failed[0].comparison, failed[0].threshold) == (
        "false_alarms_per_hour",
        2.5,
        "<=",
        2.0,
    )
    with pytest.raises(ValueError, match="not finite"):
        model_gate(dataclasses.replace(good, macro_f1=math.nan), gate_cfg)


# ---------------------------------------------------------------------------
# 묶음 평가와 리포트
# ---------------------------------------------------------------------------

FOLD_SUBJECTS = {1: (1, 2), 2: (13, 14)}
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
TCN: tuple[str, str, str] = ("run", "tcn", "alert_baseline")
GRU: tuple[str, str, str] = ("run", "gru", "alert_baseline")
ROOT_HALF = 1.0 / math.sqrt(2.0)  # 값 둘의 표본 표준편차 = |차| / √2


def _group_rows(family: str, seed: int, wrong_alert: bool) -> list[list[dict[str, Any]]]:
    """피험자 넷(fold 1: 1, 2. fold 2: 13, 14), 영상마다 창 12개. S14_5는 창이 없다.

    모든 창이 정답 클래스를 고른다. wrong_alert이면 S02_0의 창 12개만 p_drowsy 0.9다(영상은 drowsy로 판정되고
    모델 졸음 판정이 10번째 스텝에서 시작한다). drowsy 영상은 p_drowsy 0.7, perclos_60 0.2다.
    """
    sessions = []
    for subject in (1, 2, 13, 14):
        alert = [(0.05, 0.05, 0.9)] * 12 if wrong_alert and subject == 2 else [A] * 12
        sessions.append(window_rows(subject, CLASS_ALERT, alert, seed=seed, family=family))
        if subject != 14:
            sessions.append(window_rows(subject, CLASS_LOW_VIGILANT, [L] * 12, seed=seed, family=family))
        sessions.append(window_rows(subject, CLASS_DROWSY, [D] * 12, perclos=0.2, seed=seed, family=family))
    return sessions


@pytest.fixture(scope="module")
def group_videos() -> pd.DataFrame:
    entries = [(s, c, not (s == 14 and c == CLASS_LOW_VIGILANT)) for s in (1, 2, 13, 14) for c in range(3)]
    return video_rows(*entries)


@pytest.fixture(scope="module")
def tcn_preds() -> pd.DataFrame:
    """시드 0에서는 S02_0을 틀리고 시드 1에서는 모두 맞힌다."""
    return table(*_group_rows("tcn", 0, True), *_group_rows("tcn", 1, False))


@pytest.fixture(scope="module")
def gru_preds() -> pd.DataFrame:
    return table(*_group_rows("gru", 0, False), *_group_rows("gru", 1, False))


@pytest.fixture(scope="module")
def evaluation(experiment_config: ExperimentConfig) -> Any:
    return dataclasses.replace(experiment_config.evaluation, bootstrap_n=200, bootstrap_seed=3)


@pytest.fixture(scope="module")
def tcn_result(tcn_preds: pd.DataFrame, group_videos: pd.DataFrame, rules_config: RulesConfig, evaluation: Any) -> Any:
    return evaluate_group(tcn_preds, group_videos, rules_config, evaluation)


def test_evaluate_group_fold_values_are_seed_means(tcn_result: Any) -> None:
    """영상 정확도(분모: fold마다 영상 6개. S14_5는 창이 없어 항상 오답).

    fold 1: 시드 0은 5/6(S02_0 오답), 시드 1은 6/6 → 시드 평균 11/12
    fold 2: 두 시드 모두 5/6
    fold 평균 (11/12 + 10/12)/2 = 0.875, fold 표준편차 (1/12)/√2
    시드별 fold 평균: 시드 0은 5/6, 시드 1은 11/12 → 시드 표준편차 (1/12)/√2
    fold를 합친 값: 시드 0은 10/12, 시드 1은 11/12 → 0.875
    """
    result = tcn_result
    assert result.key == TCN and result.seeds == (0, 1) and result.folds == (1, 2)
    assert result.n_videos == 12 and result.n_no_window == 1.0
    assert list(result.per_fold.columns) == ["fold"] + list(METRIC_NAMES)
    assert result.per_fold["video_accuracy"].tolist() == pytest.approx([11 / 12, 5 / 6])
    seed_values = result.per_fold_seed.set_index(["fold", "seed"])["video_accuracy"]
    assert seed_values.to_dict() == pytest.approx({(1, 0): 5 / 6, (1, 1): 1.0, (2, 0): 5 / 6, (2, 1): 5 / 6})

    assert result.value("video_accuracy", "fold_mean") == pytest.approx(0.875)
    assert result.value("video_accuracy", "fold_std") == pytest.approx(ROOT_HALF / 12)
    assert result.value("video_accuracy", "seed_std") == pytest.approx(ROOT_HALF / 12)
    assert result.value("video_accuracy") == pytest.approx(0.875)
    with pytest.raises(KeyError):
        result.value("no_such_metric")


def test_evaluate_group_classification_metrics(tcn_result: Any) -> None:
    """품질 기준 통과 영상만(S14_5 제외, 11개): 시드 0은 10/11, 시드 1은 11/11.
    2클래스(alert·drowsy 8개): 시드 0은 S02_0이 drowsy로 판정돼 7/8, 시드 1은 8/8. fold 1의 시드 평균 7/8.
    창 macro-F1(영상 가중), 시드 0, fold 합침: alert 행 [3, 0, 1], low_vigilant [0, 3, 0], drowsy [0, 0, 4]
      F1 = 6/7, 1, 8/9 → 173/189. 시드 1은 1.
    영상 재현율 alert: 시드 0은 3/4, 시드 1은 4/4. low_vigilant: 3/4(S14_5).
    """
    result = tcn_result
    assert result.value("video_accuracy_gated") == pytest.approx((10 / 11 + 1.0) / 2)
    assert result.value("two_class_video_accuracy") == pytest.approx(15 / 16)
    assert result.per_fold["two_class_video_accuracy"].tolist() == pytest.approx([7 / 8, 1.0])
    assert result.value("window_macro_f1") == pytest.approx((173 / 189 + 1.0) / 2)
    assert result.value("window_macro_f1_pooled") == pytest.approx((173 / 189 + 1.0) / 2)
    assert result.value("video_recall_alert") == pytest.approx(7 / 8)
    assert result.value("video_recall_low_vigilant") == pytest.approx(3 / 4)
    assert result.value("video_recall_drowsy") == 1.0
    assert result.value("window_recall_alert") == pytest.approx(7 / 8)
    # fold 2의 창 지표: 창이 있는 영상은 모두 맞았다
    assert result.per_fold["window_macro_f1"].tolist()[1] == pytest.approx(1.0)

    assert result.confusion["video"] == [[3.5, 0.0, 0.5, 0.0], [0.0, 3.0, 0.0, 1.0], [0.0, 0.0, 4.0, 0.0]]
    assert result.confusion["two_class"] == [[3.5, 0.5, 0.0], [0.0, 4.0, 0.0]]
    assert result.confusion["window_pooled"] == [[42.0, 0.0, 6.0], [0.0, 36.0, 0.0], [0.0, 0.0, 48.0]]
    assert result.confusion["window_video_weighted"] == [[3.5, 0.0, 0.5], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]
    assert result.fold_confusion[1]["video"] == [[1.5, 0.0, 0.5, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]]
    assert result.fold_confusion[2]["video"] == [[2.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 2.0, 0.0]]


def test_evaluate_group_calibration(tcn_result: Any) -> None:
    """시드 1: 창 132개가 모두 신뢰도 0.7로 맞음 → ECE 0.3.
    시드 0: 120개는 신뢰도 0.7로 맞고 12개는 신뢰도 0.9로 틀림 → (120·0.3 + 12·0.9)/132.
    온도가 1이므로 적용 전후가 같다.
    """
    expected = (0.3 + (120 * 0.3 + 12 * 0.9) / 132) / 2
    assert tcn_result.value("ece_after") == pytest.approx(expected)
    assert tcn_result.value("ece_before") == pytest.approx(expected)


def test_evaluate_group_alarm_metrics(tcn_result: Any) -> None:
    """alert 영상 4개 × 12스텝 = 48 s = 1/75 h.

    model: 시드 0은 S02_0에서 시작 1번(스텝 10, 11, 12가 판정 상태), 시드 1은 0번.
      시작 횟수 0.5, 시간당 (75 + 0)/2 = 37.5, 판정이 난 alert 영상 비율 (1/4 + 0)/2, 판정 상태 시간 비율 (3/48)/2,
      창 단위 값 (12·75 + 0)/2 = 450. drowsy 영상(p_drowsy 0.7 ×12)은 모두 판정이 난다.
    perclos: drowsy 영상(0.2 ×12)만 판정이 난다. either: alert 쪽은 model과 같다.
    fold별: fold 1의 alert 평가 시간은 24 s = 1/150 h, model 시작 횟수의 시드 평균 0.5.
    """
    result = tcn_result
    assert result.value("model_onsets") == pytest.approx(0.5)
    assert result.value("model_alert_hours") == pytest.approx(1 / 75)
    assert result.value("model_onsets_per_hour") == pytest.approx(37.5)
    assert result.value("model_alert_videos_with_onset") == pytest.approx(0.5)
    assert result.value("model_alert_video_onset_ratio") == pytest.approx(0.125)
    assert result.value("model_alarm_time_share") == pytest.approx(3 / 96)
    assert result.value("model_flagged_windows_per_hour") == pytest.approx(450.0)
    assert result.value("model_drowsy_video_onset_ratio") == 1.0
    assert result.value("perclos_onsets") == 0.0 and result.value("perclos_drowsy_video_onset_ratio") == 1.0
    assert result.value("perclos_flagged_windows_per_hour") == 0.0
    assert result.value("either_onsets_per_hour") == pytest.approx(37.5)
    assert result.per_fold["model_onsets"].tolist() == pytest.approx([0.5, 0.0])
    assert result.per_fold["model_alert_hours"].tolist() == pytest.approx([1 / 150, 1 / 150])
    assert result.per_fold["model_onsets_per_hour"].tolist() == pytest.approx([75.0, 0.0])

    # 임계값 곡선: p_drowsy 0.9인 S02_0(시드 0)은 모든 임계값에서 판정이 나고, drowsy 영상(0.7)은 0.7까지만 난다
    sweep = result.alarm_sweep
    assert sweep["onset_p"].tolist() == [0.5, 0.6, 0.7, 0.8, 0.9]
    assert sweep["onsets"].tolist() == pytest.approx([0.5] * 5)
    assert sweep["drowsy_video_onset_ratio"].tolist() == pytest.approx([1.0, 1.0, 1.0, 0.0, 0.0])


def test_evaluate_group_agrees_with_the_metric_functions(
    tcn_result: Any, tcn_preds: pd.DataFrame, group_videos: pd.DataFrame, rules_config: RulesConfig
) -> None:
    """fold 1, 시드 0의 값이 공개 함수로 직접 계산한 값과 같다."""
    run = tcn_preds[(tcn_preds["fold"] == 1) & (tcn_preds["seed"] == 0)]
    videos = group_videos[group_videos["fold"] == 1]
    row = tcn_result.per_fold_seed.set_index(["fold", "seed"]).loc[(1, 0)]

    votes = video_vote(run, videos)
    assert row["video_accuracy"] == pytest.approx(video_metrics(votes).accuracy)
    assert row["video_accuracy_gated"] == pytest.approx(video_metrics(votes[votes["gate_pass"]]).accuracy)
    window = window_metrics(run)
    assert row["window_macro_f1"] == pytest.approx(window.video_weighted.macro_f1)
    assert row["window_macro_f1_pooled"] == pytest.approx(window.pooled.macro_f1)
    assert row["window_recall_alert"] == pytest.approx(window.video_weighted.recall[CLASS_ALERT])
    assert row["two_class_video_accuracy"] == pytest.approx(
        video_metrics(two_class(run, videos), TWO_CLASSES).accuracy
    )
    assert (row["ece_before"], row["ece_after"]) == pytest.approx(calibration_error(run))
    for source in DROWSY_SOURCES:
        report = alarm_report(run, rules_config, source, videos)
        assert row[f"{source}_onsets"] == report.alert_onsets
        assert row[f"{source}_alert_hours"] == pytest.approx(report.alert_hours)
        assert row[f"{source}_onsets_per_hour"] == pytest.approx(report.onsets_per_hour)
        assert row[f"{source}_alert_video_onset_ratio"] == pytest.approx(report.alert_video_onset_ratio)
        assert row[f"{source}_alarm_time_share"] == pytest.approx(report.alarm_time_share)
        assert row[f"{source}_flagged_windows_per_hour"] == pytest.approx(report.flagged_windows_per_hour)
        assert row[f"{source}_drowsy_video_onset_ratio"] == pytest.approx(report.drowsy_video_onset_ratio)


def test_evaluate_group_bootstrap_intervals(
    tcn_result: Any, tcn_preds: pd.DataFrame, group_videos: pd.DataFrame, rules_config: RulesConfig, evaluation: Any
) -> None:
    result = tcn_result
    # 통계 표는 (시드, 영상)마다 한 행이다. 그 표를 공개 부트스트랩에 넣으면 같은 구간이 나온다
    assert len(result.stats) == 2 * 12
    expected = subject_bootstrap(result.stats, lambda t: float(t["correct"].mean()), n=200, seed=3)
    interval = (result.value("video_accuracy", "ci_low"), result.value("video_accuracy", "ci_high"))
    assert interval == pytest.approx(expected)
    assert interval[0] < result.value("video_accuracy") <= interval[1]
    assert 0.0 <= result.value("model_onsets_per_hour", "ci_low") <= result.value("model_onsets_per_hour", "ci_high")
    # 구간을 구하지 않는 지표는 NaN이다
    assert math.isnan(result.value("ece_after", "ci_low"))

    again = evaluate_group(tcn_preds, group_videos, rules_config, evaluation)
    pd.testing.assert_frame_equal(again.summary, result.summary)
    without = evaluate_group(tcn_preds, group_videos, rules_config, evaluation, bootstrap=False)
    assert without.summary["ci_low"].isna().all()
    pd.testing.assert_series_equal(without.summary["pooled"], result.summary["pooled"])


def test_evaluate_group_rejects_mixed_groups_and_empty_tables(
    tcn_preds: pd.DataFrame,
    gru_preds: pd.DataFrame,
    group_videos: pd.DataFrame,
    rules_config: RulesConfig,
    evaluation: Any,
) -> None:
    with pytest.raises(ValueError, match="mixes several values of 'family'"):
        evaluate_group(pd.concat([tcn_preds, gru_preds]), group_videos, rules_config, evaluation)
    with pytest.raises(ValueError, match="empty"):
        evaluate_group(tcn_preds.iloc[:0], group_videos, rules_config, evaluation)
    # 시드 1에 fold 2의 예측이 없으면 시드 평균이 서로 다른 fold를 섞게 된다
    uneven = tcn_preds[(tcn_preds["seed"] == 0) | (tcn_preds["fold"] == 1)]
    with pytest.raises(ValueError, match="different folds"):
        evaluate_group(uneven, group_videos, rules_config, evaluation)


def test_gate_inputs_use_fold_means_and_pooled_alarm_numbers(
    tcn_result: Any, gru_preds: pd.DataFrame, group_videos: pd.DataFrame, rules_config: RulesConfig, evaluation: Any
) -> None:
    gru_result = evaluate_group(gru_preds, group_videos, rules_config, evaluation, bootstrap=False)
    inputs = gate_inputs(tcn_result, gru_result)
    # tcn의 창 macro-F1(fold 평균): fold 1은 시드 0이 (2/3 + 1 + 4/5)/3, 시드 1이 1. fold 2는 1
    fold_1 = ((2 / 3 + 1.0 + 4 / 5) / 3 + 1.0) / 2
    assert inputs.macro_f1 == pytest.approx((fold_1 + 1.0) / 2)
    assert inputs.lgbm_macro_f1 == pytest.approx(1.0)
    assert inputs.two_class_video_accuracy == pytest.approx((7 / 8 + 1.0) / 2)
    assert inputs.false_alarms_per_hour == pytest.approx(37.5)
    assert inputs.alert_video_alarm_ratio == pytest.approx(0.125)
    assert inputs.drowsy_video_alarm_ratio == 1.0


def _report(
    tmp_path: Path,
    preds: pd.DataFrame,
    videos: pd.DataFrame,
    rules_config: RulesConfig,
    evaluation: Any,
    **overrides: Any,
) -> evaluate.Report:
    arguments: dict[str, Any] = {
        "synthetic": True,
        "description": "hand-made tables",
        "command": "pytest tests/test_evaluate.py",
        "config_path": CONFIG_PATH,
        "exclusions": pd.DataFrame(
            [
                {
                    "video_id": "S14_5",
                    "subject": 14,
                    "fold": 2,
                    "class_index": 1,
                    "reason": "no_window",
                    "detail": "face_ratio=0.950, windows=0, raw=ok, baseline=none",
                }
            ]
        ),
    }
    fold_subjects = overrides.pop("fold_subjects", FOLD_SUBJECTS)
    arguments.update(overrides)
    return write_reports(tmp_path / "report", preds, videos, fold_subjects, rules_config, evaluation, **arguments)


@pytest.fixture(scope="module")
def both_preds(tcn_preds: pd.DataFrame, gru_preds: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([tcn_preds, gru_preds], ignore_index=True)


def test_write_reports_files_and_first_lines(
    tmp_path: Path, both_preds: pd.DataFrame, group_videos: pd.DataFrame, rules_config: RulesConfig, evaluation: Any
) -> None:
    report = _report(tmp_path, both_preds, group_videos, rules_config, evaluation)
    assert report.run_dir == tmp_path / "report" and report.summary_path == report.run_dir / "summary.md"
    assert sorted(path.name for path in report.run_dir.iterdir()) == [
        "alarm.csv",
        "alarm_sweep.csv",
        "config.yaml",
        "exclusions.csv",
        "per_fold.csv",
        "per_fold.json",
        "rules.json",
        "summary.json",
        "summary.md",
        "supplementary_alarm.csv",
        "supplementary_alarm_sweep.csv",
        "supplementary_per_fold.csv",
        "videos.csv",
    ]
    # 합성 데이터로 만든 파일은 모두 첫 줄(JSON은 첫 키)에 그 사실을 적는다
    summary = report.summary_path.read_text(encoding="utf-8")
    assert summary.splitlines()[0] == f"# {SYNTHETIC_NOTE}"
    assert "SYNTHETIC" in SYNTHETIC_NOTE
    # 머리말: 설명, 데이터 종류 문장, D 번호가 계획서 1.2절의 결정 번호라는 한 줄, 그다음이 항목들이다
    head = summary.splitlines()[:9]
    assert head[2] == "hand-made tables" and head[4].startswith("합성 데이터로 만든 리포트다.")
    assert head[6] == PLAN_DECISION_NOTE and head[5] == head[7] == "" and head[8].startswith("- 생성 명령: ")
    csv_names = sorted(path.name for path in report.run_dir.iterdir() if path.suffix == ".csv")
    assert len(csv_names) == 8
    for name in csv_names:
        assert (report.run_dir / name).read_text(encoding="utf-8").splitlines()[0] == f"# {SYNTHETIC_NOTE}"
    for name in ("per_fold.json", "summary.json"):
        payload = json.loads((report.run_dir / name).read_text(encoding="utf-8"))
        assert next(iter(payload)) == "data_note" and payload["data_note"] == SYNTHETIC_NOTE
        assert payload["command"] == "pytest tests/test_evaluate.py"

    # 설정 사본과 규칙 사본(PLAN 3.4절). 없는 설정 파일을 주면 아무것도 쓰기 전에 실패한다
    assert (report.run_dir / "config.yaml").read_bytes() == CONFIG_PATH.read_bytes()
    with pytest.raises(ValueError, match="does not exist"):
        _report(tmp_path / "x", both_preds, group_videos, rules_config, evaluation, config_path=tmp_path / "no.yaml")
    assert not (tmp_path / "x").exists()
    rules_copy = (report.run_dir / "rules.json").read_bytes()
    payload = json.loads((report.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["rules_hash"] == rules_hash(rules_config)
    assert hashlib.sha256(rules_copy).hexdigest() == payload["rules_hash"]
    assert payload["synthetic"] is True and payload["fold_subjects"] == {"1": [1, 2], "2": [13, 14]}
    assert payload["bootstrap"] == {"n": 200, "seed": 3, "level": 0.95}
    assert [row["method"] for row in payload["literature"]] == [row.method for row in LITERATURE]

    assert [result.key for result in report.results] == [GRU, TCN]
    assert report.gate is None and report.drowsiness_source == DROWSY_SOURCE_PERCLOS and len(report.paired) == 0


def test_write_reports_real_data_first_line(
    tmp_path: Path, tcn_preds: pd.DataFrame, group_videos: pd.DataFrame, rules_config: RulesConfig, evaluation: Any
) -> None:
    report = _report(
        tmp_path, tcn_preds, group_videos, rules_config, evaluation, synthetic=False, description="UTA-RLDD"
    )
    lines = report.summary_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == f"# {REAL_NOTE}"
    assert "SYNTHETIC" not in lines[0] and "REAL" in lines[0]
    assert lines[2] == "UTA-RLDD"
    assert lines[4].endswith("내보낸 모델 파일 자체를 시험한 값이 아니다 (D10).")
    assert lines[6] == PLAN_DECISION_NOTE and lines[5] == lines[7] == "" and lines[8].startswith("- 생성 명령: ")
    assert not any("합성" in line for line in lines)
    assert (report.run_dir / "per_fold.csv").read_text(encoding="utf-8").splitlines()[0] == f"# {REAL_NOTE}"
    assert json.loads((report.run_dir / "summary.json").read_text(encoding="utf-8"))["data_note"] == REAL_NOTE


def _plan_decision_numbers(repo_root: Path) -> set[str]:
    """계획서 1.2절의 표에 있는 결정 번호(첫 칸이 `D숫자`인 행). 문서를 직접 읽는다."""
    plan = (repo_root / "docs" / "PLAN.md").read_text(encoding="utf-8")
    section = plan[plan.index("\n### 1.2 ") : plan.index("\n### 1.3 ")]
    return set(re.findall(r"^\| (D\d+) \|", section, flags=re.MULTILINE))


def test_reports_say_what_the_plan_decision_numbers_are(
    tmp_path: Path,
    tcn_preds: pd.DataFrame,
    supplementary_report: evaluate.Report,
    group_videos: pd.DataFrame,
    rules_config: RulesConfig,
    evaluation: Any,
    repo_root: Path,
) -> None:
    """리포트 문구의 `D숫자`는 계획서 1.2절의 결정 번호이고, 리포트가 머리말에서 그렇게 말한다.

    계획서 14장의 소유자 결정(2026-11-13의 1~4번, 2026-11-14의 5번)과 혼동되지 않게 하려는 것이다(그 결정은 리포트에서
    번호만으로 부르지 않는다). 같은 줄을 쓰는 다른 두 리포트는 그것을 만드는 명령의 시험이 같은 규칙으로 본다:
    `train final`의 final/summary.md는 `test_train.test_final_summary_says_what_the_plan_decision_numbers_are`,
    `baseline_lgbm cv --all-w`의 w_grid.md는 `test_baseline_lgbm.test_cli_all_w_writes_one_combined_closure_w_table`.
    """
    # 한 줄의 내용: 어느 문서의 어느 절인지, 승인된 결정이라는 것, 소유자 결정과 다른 번호라는 것. 실제 데이터 리포트에도
    # 들어가므로 "합성"이라는 말이 없어야 한다(`test_write_reports_real_data_first_line`). 소유자 결정의 수("네 가지")를 적지
    # 않으므로 결정이 늘어도 문장이 그대로 맞다.
    assert "\n" not in PLAN_DECISION_NOTE and "합성" not in PLAN_DECISION_NOTE
    for phrase in (
        "D 번호", "`docs/PLAN.md` 1.2절", "승인된 결정의 번호", "14장에 실린 소유자 결정", "2026-11-13", "2026-11-14",
        "번호와는 다르다",
    ):  # fmt: skip
        assert phrase in PLAN_DECISION_NOTE, phrase
    assert "네 가지" not in PLAN_DECISION_NOTE
    # 그 문장이 가리키는 곳: 계획서 14장에 소유자 결정의 표가 있고 두 날짜가 적혀 있다(문서에서 읽는다).
    plan = (repo_root / "docs" / "PLAN.md").read_text(encoding="utf-8")
    chapter = plan[plan.index("\n## 14. ") :]
    owner_heading = next(line for line in chapter.splitlines() if line.startswith("### ") and "소유자 결정" in line)
    assert "2026-11-13" in owner_heading and "2026-11-14" in owner_heading

    # 계획서 1.2절의 표에는 D1~D16이 있다(문서에서 읽는다). 리포트에 나오는 번호는 모두 그 표의 번호다
    plan_numbers = _plan_decision_numbers(repo_root)
    assert plan_numbers == {f"D{number}" for number in range(1, 17)}

    def numbers_used(lines: list[str]) -> set[str]:
        return set(re.findall(r"\bD\d+\b", "\n".join(line for line in lines if line != PLAN_DECISION_NOTE)))

    real = _report(tmp_path, tcn_preds, group_videos, rules_config, evaluation, synthetic=False, gate=(TCN, TCN))
    cases = {
        "real": (real.summary_path, {"D7", "D8", "D9", "D10", "D11"}),  # D10: 교차검증 추정치라는 문장(실제 데이터만)
        "supplementary": (supplementary_report.summary_path, {"D7", "D8", "D9", "D11", "D16"}),  # D16: 부가 묶음의 설명
    }
    for name, (summary_path, expected) in cases.items():
        lines = summary_path.read_text(encoding="utf-8").splitlines()
        assert numbers_used(lines) == expected and expected <= plan_numbers, name
        # 한 줄은 한 번, 첫 절보다 앞에 있다. 그보다 앞에서 번호를 쓰는 줄은 바로 앞 문단(데이터 종류 문장)뿐이다
        assert lines.count(PLAN_DECISION_NOTE) == 1, name
        position = lines.index(PLAN_DECISION_NOTE)
        assert position == 6 and position < lines.index("## 1. fold별 결과"), name
        assert numbers_used(lines[:4]) == set(), name
        assert numbers_used(lines[4:6]) == ({"D10"} if name == "real" else set()), name


def test_write_reports_summary_contents(
    tmp_path: Path, both_preds: pd.DataFrame, group_videos: pd.DataFrame, rules_config: RulesConfig, evaluation: Any
) -> None:
    report = _report(tmp_path, both_preds, group_videos, rules_config, evaluation)
    summary = report.summary_path.read_text(encoding="utf-8")

    # fold별 표: fold 1, fold 2, fold 평균 ± 표준편차, 시드 표준편차, fold 합침과 구간
    row = next(line for line in summary.splitlines() if "`video_accuracy`" in line and "| 0.917 |" in line)
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    assert cells[1:5] == ["0.917", "0.833", "0.875 ± 0.059", "0.059"]
    assert cells[5].startswith("0.875 [") and cells[5].endswith("]")
    assert "### run / tcn / features=alert_baseline" in summary
    assert "### run / gru / features=alert_baseline" in summary
    assert "시드 0, 1. 논리 영상 12개 중 창이 없어 오답으로 센 영상 1개." in summary
    # 혼동 행렬(영상 단위, tcn): alert 행은 시드 평균 3.5, 0, 0.5, 창 없음 0
    assert "| alert | 3.50 | 0.00 | 0.50 | 0.00 |" in summary
    assert "| low_vigilant | 0.00 | 3.00 | 0.00 | 1.00 |" in summary
    # 오경보 표: model, perclos, either가 나란히 있다
    assert "| 항목 | model | perclos | either |" in summary
    onsets = next(line for line in summary.splitlines() if line.startswith("| alert 영상에서 판정이 시작된 횟수"))
    assert [cell.strip() for cell in onsets.strip("|").split("|")][1:] == ["0", "0", "0"]  # gru가 먼저 나온다
    assert "| 0.50 | 0 | 0.50 |" in summary  # tcn: model 0.5번, perclos 0번, either 0.5번
    assert "| onset_p |" in summary
    # 제외 목록
    assert "| S14_5 | 14 | 2 | 1 | no_window | face_ratio=0.950, windows=0, raw=ok, baseline=none |" in summary
    # 피험자 분리 검사와 부트스트랩 설정이 적혀 있다
    assert "assert_subject_disjoint" in summary and "200회(시드 3)" in summary
    assert "`pytest tests/test_evaluate.py`" in summary


def test_write_reports_comparison_table_labels_literature_rows(
    tmp_path: Path, tcn_preds: pd.DataFrame, group_videos: pd.DataFrame, rules_config: RulesConfig, evaluation: Any
) -> None:
    report = _report(tmp_path / "real", tcn_preds, group_videos, rules_config, evaluation, synthetic=False)
    lines = report.summary_path.read_text(encoding="utf-8").splitlines()
    start = lines.index("## 6. 비교표")
    rows = [line for line in lines[start : lines.index("## 7. 제외 목록")] if line.startswith("| ")]
    assert len(rows) == 2 + 1 + len(LITERATURE)  # 머리글 두 줄, 이 저장소의 묶음 하나, 문헌 행

    literature = {
        "HM-LSTM": ("65.2%", "-", "174개 영상만", "검증 분할이 없다", "0.81 / 0.32 / 0.82"),
        "사람 판정": ("57.8%", "-"),
        "Wolter 2025": ("52.5%", "80.4%"),
        "Magán 2022": ("-", "55~63%"),
        "Xu 2025": ("86.5%", "-", "관찰자 판정과 라벨이 다른 구간을 버린 뒤의 값"),
    }
    assert len(LITERATURE) == len(literature)
    for method, expected in literature.items():
        row = next(line for line in rows if line.startswith(f"| {method}"))
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        # 문헌 행은 모두 "다른 연구자가 보고한 값"으로 표시된다
        assert cells[1] == LITERATURE_LABEL and "다른 연구자가 보고한 값" in LITERATURE_LABEL
        assert (cells[2], cells[3]) == expected[:2]
        for caveat in expected[2:]:
            assert caveat in cells[4]

    ours = next(line for line in rows if line.startswith("| run / tcn"))
    cells = [cell.strip() for cell in ours.strip("|").split("|")]
    assert LITERATURE_LABEL not in ours and cells[1] == "이 저장소, 교차검증 추정치"
    assert cells[2] == "87.5% ± 5.9%"  # 영상 정확도의 fold 평균 ± 표준편차
    assert cells[3].startswith("93.8% ±")  # 2클래스: (7/8 + 1)/2

    # 합성 데이터의 리포트에서는 이 저장소의 점수를 문헌 값 옆에 두지 않는다. 문헌 행은 그대로 있다
    synthetic = _report(tmp_path / "synthetic", tcn_preds, group_videos, rules_config, evaluation)
    lines = synthetic.summary_path.read_text(encoding="utf-8").splitlines()
    section = lines[lines.index("## 6. 비교표") : lines.index("## 7. 제외 목록")]
    rows = [line for line in section if line.startswith("| ")]
    assert len(rows) == 2 + len(LITERATURE) and not any(line.startswith("| run /") for line in rows)
    assert all(LITERATURE_LABEL in line for line in rows[2:])
    assert "합성 데이터 리포트이므로 이 저장소의 결과는 이 표에 싣지 않는다(성능 수치가 아니다)." in section


def test_write_reports_csv_and_json_values(
    tmp_path: Path, both_preds: pd.DataFrame, group_videos: pd.DataFrame, rules_config: RulesConfig, evaluation: Any
) -> None:
    report = _report(tmp_path, both_preds, group_videos, rules_config, evaluation)
    per_fold = pd.read_csv(report.run_dir / "per_fold.csv", skiprows=1)
    group_columns = ["run_id", "family", "arm", "feature_source", "variant"]
    assert list(per_fold.columns) == group_columns + ["fold", "n_seeds"] + list(METRIC_NAMES)
    assert per_fold["feature_source"].unique().tolist() == ["native"]
    assert per_fold["variant"].unique().tolist() == ["base"]
    assert per_fold[["family", "fold"]].values.tolist() == [["gru", 1], ["gru", 2], ["tcn", 1], ["tcn", 2]]
    assert per_fold["n_seeds"].tolist() == [2, 2, 2, 2]
    assert per_fold["video_accuracy"].tolist() == pytest.approx([1.0, 5 / 6, 11 / 12, 5 / 6])
    assert per_fold["model_onsets"].tolist() == pytest.approx([0.0, 0.0, 0.5, 0.0])

    alarm = pd.read_csv(report.run_dir / "alarm.csv", skiprows=1)
    assert alarm[["family", "source"]].values.tolist() == [[f, s] for f in ("gru", "tcn") for s in DROWSY_SOURCES]
    tcn_model = alarm[(alarm["family"] == "tcn") & (alarm["source"] == "model")].iloc[0]
    assert tcn_model["onsets_per_hour"] == pytest.approx(37.5) and tcn_model["onsets"] == pytest.approx(0.5)
    assert tcn_model["onsets_per_hour_ci_low"] <= 37.5 <= tcn_model["onsets_per_hour_ci_high"]
    sweep = pd.read_csv(report.run_dir / "alarm_sweep.csv", skiprows=1)
    assert len(sweep) == 2 * 5 and sweep["onset_p"].tolist()[:5] == [0.5, 0.6, 0.7, 0.8, 0.9]
    assert len(pd.read_csv(report.run_dir / "videos.csv", skiprows=1)) == 12
    assert pd.read_csv(report.run_dir / "exclusions.csv", skiprows=1)["video_id"].tolist() == ["S14_5"]

    folds = json.loads((report.run_dir / "per_fold.json").read_text(encoding="utf-8"))["groups"][1]
    assert (folds["run_id"], folds["family"], folds["arm"], folds["seeds"]) == ("run", "tcn", "alert_baseline", [0, 1])
    fold_1 = folds["folds"][0]
    assert fold_1["fold"] == 1 and fold_1["metrics"]["video_accuracy"] == pytest.approx(11 / 12)
    assert fold_1["per_seed"]["0"]["video_accuracy"] == pytest.approx(5 / 6)
    assert fold_1["per_seed"]["1"]["video_accuracy"] == pytest.approx(1.0)
    assert fold_1["confusion"]["video"] == [[1.5, 0.0, 0.5, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]]

    summary = json.loads((report.run_dir / "summary.json").read_text(encoding="utf-8"))["groups"][1]
    accuracy = summary["summary"]["video_accuracy"]
    assert accuracy["fold_mean"] == pytest.approx(0.875) and accuracy["fold_std"] == pytest.approx(ROOT_HALF / 12)
    assert accuracy["pooled"] == pytest.approx(0.875) and accuracy["ci_low"] <= 0.875 <= accuracy["ci_high"]
    assert summary["summary"]["ece_after"]["ci_low"] is None  # NaN은 null로 쓴다
    assert summary["n_videos"] == 12 and summary["n_no_window"] == 1.0


def test_write_reports_checks_the_fold_assignment_it_is_given(
    tmp_path: Path,
    tcn_preds: pd.DataFrame,
    group_videos: pd.DataFrame,
    rules_config: RulesConfig,
    evaluation: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[int, ...], ...]] = []
    original = evaluate.assert_subject_disjoint

    def recording(*subject_sets: Any) -> None:
        calls.append(tuple(tuple(subjects) for subjects in subject_sets))
        original(*subject_sets)

    monkeypatch.setattr(evaluate, "assert_subject_disjoint", recording)
    _report(tmp_path, tcn_preds, group_videos, rules_config, evaluation, fold_subjects={2: [14, 13], 1: [2, 1]})
    # 받은 fold 배정의 fold별 시험 피험자 집합으로 한 번 부른다
    assert calls == [((1, 2), (13, 14))]


def test_write_reports_rejects_overlapping_or_violated_fold_assignment(
    tmp_path: Path, tcn_preds: pd.DataFrame, group_videos: pd.DataFrame, rules_config: RulesConfig, evaluation: Any
) -> None:
    # 한 피험자가 두 fold의 시험 집합에 있다
    with pytest.raises(SubjectOverlapError, match="subject 2"):
        _report(tmp_path, tcn_preds, group_videos, rules_config, evaluation, fold_subjects={1: [1, 2], 2: [2, 13, 14]})
    assert not (tmp_path / "report").exists()  # 검사에 실패하면 아무것도 쓰지 않는다

    # 예측 표가 배정을 어긴다: 피험자 13(fold 2의 시험 피험자)을 fold 1에서 평가했다
    leaked = tcn_preds.assign(fold=np.where(tcn_preds["subject"] == 13, 1, tcn_preds["fold"]))
    with pytest.raises(SubjectOverlapError, match="subject 13 in fold 1"):
        _report(tmp_path, leaked, group_videos, rules_config, evaluation)

    with pytest.raises(ValueError, match="folds \\[2\\] that the fold assignment does not list"):
        _report(tmp_path, tcn_preds, group_videos, rules_config, evaluation, fold_subjects={1: [1, 2]})
    # 시험 피험자의 영상이 영상 표에 없으면 분모에서 조용히 빠지게 된다
    with pytest.raises(ValueError, match="no video of test subjects \\[15\\]"):
        _report(
            tmp_path, tcn_preds, group_videos, rules_config, evaluation, fold_subjects={1: [1, 2], 2: [13, 14, 15]}
        )
    with pytest.raises(ValueError, match="fold assignment is empty"):
        _report(tmp_path, tcn_preds, group_videos, rules_config, evaluation, fold_subjects={})


def test_write_reports_without_exclusions_and_without_sweep(
    tmp_path: Path, tcn_preds: pd.DataFrame, group_videos: pd.DataFrame, rules_config: RulesConfig, evaluation: Any
) -> None:
    report = _report(
        tmp_path,
        tcn_preds,
        group_videos,
        rules_config,
        dataclasses.replace(evaluation, alarm_threshold_sweep=()),
        exclusions=pd.DataFrame(columns=["video_id", "reason"]),
    )
    summary = report.summary_path.read_text(encoding="utf-8")
    assert "제외하거나 따로 표시한 영상이 없다." in summary and "| onset_p |" not in summary
    assert len(report.results[0].alarm_sweep) == 0
    assert len(pd.read_csv(report.run_dir / "alarm_sweep.csv", skiprows=1)) == 0
    with pytest.raises(ValueError, match="exclusion list lacks columns"):
        _report(tmp_path / "x", tcn_preds, group_videos, rules_config, evaluation, exclusions=pd.DataFrame())


def test_write_reports_drops_videos_of_folds_that_are_not_evaluated(
    tmp_path: Path, tcn_preds: pd.DataFrame, group_videos: pd.DataFrame, rules_config: RulesConfig, evaluation: Any
) -> None:
    extra = pd.concat([group_videos, video_rows((25, 0, True), (25, 1, True), (25, 2, True))], ignore_index=True)
    report = _report(tmp_path, tcn_preds, extra, rules_config, evaluation)
    assert report.results[0].n_videos == 12 and report.results[0].folds == (1, 2)
    assert report.results[0].value("video_accuracy") == pytest.approx(0.875)


def test_write_reports_gate_and_default_source(
    tmp_path: Path, both_preds: pd.DataFrame, group_videos: pd.DataFrame, rules_config: RulesConfig, evaluation: Any
) -> None:
    # gru는 다섯 조건을 모두 만족한다(오경보 없음, drowsy 영상 모두 판정). 그래도 합성 데이터면 판정원은 perclos다
    synthetic = _report(tmp_path / "a", both_preds, group_videos, rules_config, evaluation, gate=(GRU, TCN))
    assert synthetic.gate is not None and synthetic.gate.passed
    assert synthetic.drowsiness_source == DROWSY_SOURCE_PERCLOS
    text = synthetic.summary_path.read_text(encoding="utf-8")
    assert "합성 데이터의 수치이므로 이 판정은 쓰지 않는다" in text and "| `false_alarms_per_hour` | 0.000 | <= 2.000 | 예 |" in text

    real = _report(tmp_path / "b", both_preds, group_videos, rules_config, evaluation, gate=(GRU, TCN), synthetic=False)
    assert real.gate is not None and real.gate.passed and real.drowsiness_source == DROWSY_SOURCE_MODEL
    payload = json.loads((real.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["gate"]["passed"] is True and payload["drowsiness_source"] == "model"
    assert payload["gate"]["candidate"] == list(GRU) and payload["gate"]["reference"] == list(TCN)
    assert [c["name"] for c in payload["gate"]["conditions"]][2] == "false_alarms_per_hour"

    # tcn은 alert 영상에서 시간당 37.5번 판정이 시작돼 조건을 어긴다
    failing = _report(
        tmp_path / "c", both_preds, group_videos, rules_config, evaluation, gate=(TCN, GRU), synthetic=False
    )
    assert failing.gate is not None and not failing.gate.passed
    assert failing.drowsiness_source == DROWSY_SOURCE_PERCLOS
    assert "| `false_alarms_per_hour` | 37.500 | <= 2.000 | 아니오 |" in failing.summary_path.read_text(encoding="utf-8")
    assert "실험용" in failing.summary_path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="names a group without predictions"):
        _report(tmp_path / "d", both_preds, group_videos, rules_config, evaluation, gate=(GRU, ("run", "lgbm", "x")))


def test_write_reports_paired_comparison(
    tmp_path: Path, both_preds: pd.DataFrame, group_videos: pd.DataFrame, rules_config: RulesConfig, evaluation: Any
) -> None:
    report = _report(tmp_path, both_preds, group_videos, rules_config, evaluation, paired=[(TCN, GRU), (GRU, GRU)])
    paired = report.paired
    assert paired["metric"].tolist() == ["video_accuracy", "window_macro_f1", "two_class_video_accuracy"] * 2
    first = paired.iloc[0]
    # 영상 정확도(fold 합침): tcn 0.875, gru 11/12(S14_5만 오답)
    assert (first["a"], first["b"]) == ("run / tcn / features=alert_baseline", "run / gru / features=alert_baseline")
    assert first["value_a"] == pytest.approx(0.875) and first["value_b"] == pytest.approx(11 / 12)
    assert first["difference"] == pytest.approx(-1 / 24)
    by_key = {result.key: result for result in report.results}
    expected = paired_subject_bootstrap(
        by_key[TCN].stats, by_key[GRU].stats, lambda t: float(t["correct"].mean()), n=200, seed=3
    )
    assert (first["ci_low"], first["ci_high"]) == pytest.approx(expected)
    # 차이가 가장 큰 추출에서도 0을 넘지 않는다(tcn이 더 맞힌 영상이 없다)
    assert first["ci_high"] <= 0.0
    assert bool(first["differs"]) == (first["ci_high"] < 0.0 or first["ci_low"] > 0.0)
    # 같은 묶음끼리는 차이가 없다
    same = paired.iloc[3:]
    assert same["difference"].tolist() == [0.0, 0.0, 0.0] and not same["differs"].any()
    assert (same["ci_low"] == 0.0).all() and (same["ci_high"] == 0.0).all()
    assert "## 4. 짝지은 비교" in report.summary_path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="paired comparison names a group"):
        _report(
            tmp_path / "x", both_preds, group_videos, rules_config, evaluation, paired=[(TCN, ("run", "lgbm", "x"))]
        )


# ---------------------------------------------------------------------------
# 예측이 없는 fold: 평가하지 않은 fold를 0점으로 싣지 않는다 (독립 검토에서 나온 지적을 고친 것)
# ---------------------------------------------------------------------------


def test_evaluate_group_rejects_a_fold_without_predictions(
    tmp_path: Path,
    tcn_preds: pd.DataFrame,
    gru_preds: pd.DataFrame,
    group_videos: pd.DataFrame,
    rules_config: RulesConfig,
    evaluation: Any,
) -> None:
    """fold 2를 평가하지 않은 묶음을 0점으로 싣지 않는다(전에는 fold 2가 0.000, fold 평균이 0.458로 나왔다)."""
    partial = tcn_preds[tcn_preds["fold"] == 1]
    with pytest.raises(ValueError, match=r"no predictions for fold\(s\) \[2\]"):
        evaluate_group(partial, group_videos, rules_config, evaluation, bootstrap=False)
    # 다른 묶음(gru)이 그 fold를 평가했더라도 리포트 전체를 거부하고 아무것도 쓰지 않는다
    with pytest.raises(ValueError, match=r"\('run', 'tcn', 'alert_baseline'\) has no predictions for fold\(s\) \[2\]"):
        _report(tmp_path, pd.concat([partial, gru_preds], ignore_index=True), group_videos, rules_config, evaluation)
    assert not (tmp_path / "report").exists()

    # fold 배정을 평가한 fold로 줄이면 받는다(일부 fold만 평가한 리포트). fold 1: 시드 0은 5/6, 시드 1은 6/6
    report = _report(tmp_path, partial, group_videos, rules_config, evaluation, fold_subjects={1: (1, 2)})
    assert report.results[0].folds == (1,) and report.results[0].n_videos == 6
    assert report.results[0].value("video_accuracy") == pytest.approx(11 / 12)


def test_evaluate_group_keeps_the_d7_rule_inside_evaluated_folds(
    tcn_preds: pd.DataFrame, group_videos: pd.DataFrame, rules_config: RulesConfig, evaluation: Any
) -> None:
    """예측이 있는 fold 안에서 창이 없는 영상은 여전히 오답이다. 영상 표가 "이 fold에는 창이 없다"고 하면 fold도 받는다."""
    # fold 2에서 S13의 영상 셋의 창을 모두 뺀다: fold 2는 평가됐고(S14의 창이 있다) S13의 영상은 오답이 된다
    thinned = tcn_preds[tcn_preds["subject"] != 13]
    result = evaluate_group(thinned, group_videos, rules_config, evaluation, bootstrap=False)
    # fold 2의 영상 6개 중 맞은 것은 S14_0, S14_10뿐이다(S13 셋과 S14_5는 창이 없다)
    assert result.per_fold["video_accuracy"].tolist() == pytest.approx([11 / 12, 2 / 6])
    assert result.n_no_window == 4.0

    # 영상 표의 n_windows가 fold 2의 영상이 모두 0이라고 하면, 예측이 없는 것은 평가하지 않은 것이 아니다
    partial = tcn_preds[tcn_preds["fold"] == 1]
    counted = group_videos.assign(n_windows=np.where(group_videos["fold"] == 1, 12, 0))
    result = evaluate_group(partial, counted, rules_config, evaluation, bootstrap=False)
    assert result.folds == (1, 2) and result.n_no_window == 6.0
    assert result.per_fold["video_accuracy"].tolist() == pytest.approx([11 / 12, 0.0])
    # fold 2에 창이 있는 영상이 하나라도 있으면 평가하지 않은 fold다
    counted.loc[counted["video_id"] == "S13_0", "n_windows"] = 5
    with pytest.raises(ValueError, match=r"no predictions for fold\(s\) \[2\]"):
        evaluate_group(partial, counted, rules_config, evaluation, bootstrap=False)


# ---------------------------------------------------------------------------
# 부가 묶음 (PLAN 11장 E3, D16)
# ---------------------------------------------------------------------------

FPS15_KEY = ("run", "tcn", "alert_baseline", "fps15", "base")
ORDINAL_KEY = ("run", "tcn", "alert_baseline", "native", "ordinal")
INT8_KEY = ("run", "tcn_int8", "alert_baseline")


def _labelled(sessions: list[list[dict[str, Any]]], labels: dict[str, Any], drop: str | None = None) -> list[Any]:
    """세션들의 행에 표지를 단다. drop에 준 영상의 창은 뺀다(그 추출에서 창이 나오지 않은 영상)."""
    kept = [session for session in sessions if drop is None or session[0]["video_id"] != drop]
    return [[dict(row, **labels) for row in session] for session in kept]


@pytest.fixture(scope="module")
def supplementary_preds(tcn_preds: pd.DataFrame) -> pd.DataFrame:
    """주 결과 tcn에 부가 묶음 셋을 더한 표.

    tcn(주 결과):  시드 0은 S02_0을 틀린다. S14_5는 창이 없다 → fold를 합친 영상 정확도 (10/12 + 11/12)/2 = 21/24
    tcn_int8:      모든 창이 맞는다 → 11/12 (두 시드)
    tcn [fps15]:   tcn과 같은 예측이지만 S13_10의 창이 없다 → (9/12 + 10/12)/2 = 19/24, 창 없는 영상 2개
    tcn ordinal:   모든 창이 맞는다 → 11/12
    """
    sessions: list[list[dict[str, Any]]] = []
    for seed in (0, 1):
        sessions += _labelled(_group_rows("tcn_int8", seed, False), INT8)
        sessions += _labelled(_group_rows("tcn", seed, seed == 0), FPS15, drop="S13_10")
        sessions += _labelled(_group_rows("tcn", seed, False), ORDINAL)
    extra = pd.DataFrame([row for session in sessions for row in session])
    # 주 결과의 표는 표지 열이 없던 때 그대로 이어 붙인다(학습 코드가 하는 방식)
    return pd.concat([tcn_preds.drop(columns=list(LABEL_COLUMNS)), extra], ignore_index=True)


@pytest.fixture(scope="module")
def supplementary_report(
    tmp_path_factory: pytest.TempPathFactory,
    supplementary_preds: pd.DataFrame,
    group_videos: pd.DataFrame,
    rules_config: RulesConfig,
    evaluation: Any,
) -> evaluate.Report:
    return _report(
        tmp_path_factory.mktemp("supplementary"),
        supplementary_preds,
        group_videos,
        rules_config,
        evaluation,
        paired=supplementary_pairs(supplementary_preds),
    )


def test_supplementary_pairs_name_the_base_group(supplementary_preds: pd.DataFrame, tcn_preds: pd.DataFrame) -> None:
    # 부가 묶음의 키 순서. INT8은 접미사를 뗀 계열, 15 fps 모사는 native, 순서형 라벨은 base가 기준이다
    assert supplementary_pairs(supplementary_preds) == [(FPS15_KEY, TCN), (ORDINAL_KEY, TCN), (INT8_KEY, TCN)]
    # 부가 묶음이 없는 표, 표지 열이 없는 표
    assert supplementary_pairs(tcn_preds) == []
    assert supplementary_pairs(tcn_preds.drop(columns=list(LABEL_COLUMNS))) == []
    # 기준 묶음이 표에 없으면 쌍을 만들지 않는다
    only_extra = supplementary_preds[supplementary_preds["supplementary"].fillna(False).astype(bool)]
    assert supplementary_pairs(only_extra) == []
    # INT8이면서 15 fps 모사인 묶음의 기준은 같은 추출의 float 묶음이다(다른 점 하나만 되돌린다)
    quantised = window_rows(1, CLASS_ALERT, [A], family="tcn_int8", labels=FPS15)
    both = pd.DataFrame(quantised + window_rows(1, CLASS_ALERT, [A], labels=FPS15))
    assert supplementary_pairs(both) == [(("run", "tcn_int8", "alert_baseline", "fps15", "base"), FPS15_KEY)]


def test_write_reports_keeps_supplementary_groups_out_of_the_main_result(
    supplementary_report: evaluate.Report,
) -> None:
    report = supplementary_report
    assert [result.key for result in report.results] == [TCN]
    assert [result.key for result in report.supplementary] == [FPS15_KEY, ORDINAL_KEY, INT8_KEY]
    assert all(result.supplementary for result in report.supplementary) and not report.results[0].supplementary
    by_key = {result.key: result for result in report.supplementary}
    assert (by_key[FPS15_KEY].feature_source, by_key[FPS15_KEY].variant) == ("fps15", "base")
    assert (by_key[ORDINAL_KEY].feature_source, by_key[ORDINAL_KEY].variant) == ("native", "ordinal")
    assert by_key[INT8_KEY].family == "tcn_int8"

    # 값은 주 결과와 같은 방식, 같은 분모(영상 표의 12개)로 계산한다
    assert report.results[0].value("video_accuracy") == pytest.approx(21 / 24)
    assert by_key[INT8_KEY].value("video_accuracy") == pytest.approx(11 / 12)
    assert by_key[ORDINAL_KEY].value("video_accuracy") == pytest.approx(11 / 12)
    assert by_key[FPS15_KEY].value("video_accuracy") == pytest.approx(19 / 24)
    assert by_key[FPS15_KEY].n_videos == 12 and by_key[FPS15_KEY].n_no_window == 2.0
    # fold 2: S13_10과 S14_5에 창이 없다 → 4/6
    assert by_key[FPS15_KEY].per_fold["video_accuracy"].tolist() == pytest.approx([11 / 12, 4 / 6])

    summary = report.summary_path.read_text(encoding="utf-8")
    lines = summary.splitlines()
    main = "\n".join(lines[: lines.index("## 7. 제외 목록")])
    extra = "\n".join(lines[lines.index("## 9. 부가 결과") :])
    assert lines.index("## 8. 기준선과 창 수 (PLAN 4.4절)") < lines.index("## 9. 부가 결과")
    # 1~6절에는 부가 묶음이 없다
    for word in ("tcn_int8", "fps15", "ordinal"):
        assert word not in main and word in extra
    assert main.count("### run / tcn / features=alert_baseline") == 3  # fold별 표, 혼동 행렬, 오경보
    assert "부가 묶음 3개는 9절" in main
    # 9절: 부가 묶음마다 "부가" 제목과 fold별 표, 오경보 표
    assert [line for line in lines if line.startswith("### 부가")] == [
        "### 부가: run / tcn / features=alert_baseline [fps15]",
        "### 부가: run / tcn / features=alert_baseline / variant=ordinal",
        "### 부가: run / tcn_int8 / features=alert_baseline",
        "### 부가 묶음의 짝지은 비교",
    ]
    assert "주 결과(1~6절), 비교표, D11 판정에는 쓰지 않는다" in extra
    assert extra.count("| 항목 | model | perclos | either |") == 3
    assert "논리 영상 12개 중 창이 없어 오답으로 센 영상 2개." in extra


def test_write_reports_supplementary_files(supplementary_report: evaluate.Report) -> None:
    run_dir = supplementary_report.run_dir
    assert sorted(path.name for path in run_dir.iterdir() if path.name.startswith("supplementary_")) == [
        "supplementary_alarm.csv",
        "supplementary_alarm_sweep.csv",
        "supplementary_per_fold.csv",
    ]
    # 주 결과의 표 파일에는 주 결과 묶음만 있다
    per_fold = pd.read_csv(run_dir / "per_fold.csv", skiprows=1)
    assert per_fold[["family", "feature_source", "variant", "fold"]].values.tolist() == [
        ["tcn", "native", "base", 1],
        ["tcn", "native", "base", 2],
    ]
    assert pd.read_csv(run_dir / "alarm.csv", skiprows=1)["family"].unique().tolist() == ["tcn"]
    assert pd.read_csv(run_dir / "alarm_sweep.csv", skiprows=1)["family"].unique().tolist() == ["tcn"]

    extra = pd.read_csv(run_dir / "supplementary_per_fold.csv", skiprows=1)
    assert (run_dir / "supplementary_per_fold.csv").read_text(encoding="utf-8").splitlines()[0] == f"# {SYNTHETIC_NOTE}"
    assert extra[["family", "feature_source", "variant", "fold"]].values.tolist() == [
        ["tcn", "fps15", "base", 1],
        ["tcn", "fps15", "base", 2],
        ["tcn", "native", "ordinal", 1],
        ["tcn", "native", "ordinal", 2],
        ["tcn_int8", "native", "base", 1],
        ["tcn_int8", "native", "base", 2],
    ]
    assert extra["video_accuracy"].tolist() == pytest.approx([11 / 12, 4 / 6, 1.0, 5 / 6, 1.0, 5 / 6])
    alarm = pd.read_csv(run_dir / "supplementary_alarm.csv", skiprows=1)
    assert len(alarm) == 3 * len(DROWSY_SOURCES) and set(alarm["family"]) == {"tcn", "tcn_int8"}
    # 15 fps 묶음의 모델 오경보는 tcn과 같다(S02_0, 시드 0): 시간당 37.5
    fps = alarm[(alarm["feature_source"] == "fps15") & (alarm["source"] == "model")].iloc[0]
    assert fps["onsets_per_hour"] == pytest.approx(37.5)

    def labels_of(groups: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
        return [(g["family"], g["feature_source"], g["variant"], g["supplementary"]) for g in groups]

    payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert labels_of(payload["groups"]) == [("tcn", "native", "base", False)]
    assert payload["paired"] == []
    assert labels_of(payload["supplementary"]["groups"]) == [
        ("tcn", "fps15", "base", True),
        ("tcn", "native", "ordinal", True),
        ("tcn_int8", "native", "base", True),
    ]
    assert len(payload["supplementary"]["paired"]) == 9
    folds = json.loads((run_dir / "per_fold.json").read_text(encoding="utf-8"))
    assert [group["family"] for group in folds["groups"]] == ["tcn"]
    assert [group["family"] for group in folds["supplementary_groups"]] == ["tcn", "tcn", "tcn_int8"]


def test_write_reports_supplementary_paired_comparison(supplementary_report: evaluate.Report) -> None:
    """부가 묶음 − 기준 묶음(영상 정확도, fold 합침): 15 fps 19/24 − 21/24, 순서형 22/24 − 21/24, INT8 22/24 − 21/24."""
    report = supplementary_report
    assert len(report.paired) == 0  # 주 결과끼리의 비교는 요청하지 않았다
    paired = report.supplementary_paired
    assert list(paired.columns) == list(evaluate.PAIRED_COLUMNS) and paired["supplementary"].all()
    accuracy = paired[paired["metric"] == "video_accuracy"].reset_index(drop=True)
    assert accuracy["a"].tolist() == [
        "run / tcn / features=alert_baseline [fps15]",
        "run / tcn / features=alert_baseline / variant=ordinal",
        "run / tcn_int8 / features=alert_baseline",
    ]
    assert accuracy["b"].unique().tolist() == ["run / tcn / features=alert_baseline"]
    assert accuracy["difference"].tolist() == pytest.approx([-1 / 12, 1 / 24, 1 / 24])
    assert accuracy["value_b"].tolist() == pytest.approx([21 / 24] * 3)

    by_key = {result.key: result for result in (*report.results, *report.supplementary)}
    expected = paired_subject_bootstrap(
        by_key[FPS15_KEY].stats, by_key[TCN].stats, lambda t: float(t["correct"].mean()), n=200, seed=3
    )
    assert (accuracy.loc[0, "ci_low"], accuracy.loc[0, "ci_high"]) == pytest.approx(expected)
    # 15 fps 묶음이 더 맞힌 영상은 없으므로 차이의 구간이 0을 넘지 않는다
    assert accuracy.loc[0, "ci_high"] <= 0.0

    summary = report.summary_path.read_text(encoding="utf-8").splitlines()
    section_4 = summary[summary.index("## 4. 짝지은 비교") : summary.index("## 5. D11 판정")]
    assert "요청된 비교가 없다." in section_4
    tail = summary[summary.index("### 부가 묶음의 짝지은 비교") :]
    row = next(line for line in tail if line.startswith("| run / tcn_int8") and "`video_accuracy`" in line)
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    assert cells[:2] == ["run / tcn_int8 / features=alert_baseline", "run / tcn / features=alert_baseline"]
    assert cells[2:5] == ["`video_accuracy`", "0.917", "0.875"]
    assert cells[5].startswith("0.042 [")


def test_write_reports_mixed_and_explicit_keys(
    tmp_path: Path,
    supplementary_preds: pd.DataFrame,
    gru_preds: pd.DataFrame,
    group_videos: pd.DataFrame,
    rules_config: RulesConfig,
    evaluation: Any,
) -> None:
    """주 결과의 쌍과 부가 쌍을 함께 요청한다. 표지를 모두 적은 키도 기본값이면 세 값짜리 키와 같은 묶음이다."""
    preds = pd.concat([supplementary_preds, gru_preds], ignore_index=True)
    report = _report(
        tmp_path,
        preds,
        group_videos,
        rules_config,
        evaluation,
        paired=[(("run", "tcn", "alert_baseline", "native", "base"), GRU), (INT8_KEY, TCN)],
        gate=(GRU, ("run", "tcn", "alert_baseline", "native", "base")),
    )
    assert [result.key for result in report.results] == [GRU, TCN]
    assert report.paired["a"].unique().tolist() == ["run / tcn / features=alert_baseline"]
    assert report.paired["b"].unique().tolist() == ["run / gru / features=alert_baseline"]
    assert not report.paired["supplementary"].any() and len(report.paired) == 3
    assert report.supplementary_paired["a"].unique().tolist() == ["run / tcn_int8 / features=alert_baseline"]
    payload = json.loads((report.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["gate"]["candidate"] == list(GRU) and payload["gate"]["reference"] == list(TCN)
    assert len(payload["paired"]) == 3 and len(payload["supplementary"]["paired"]) == 3
    with pytest.raises(ValueError, match="a group key is"):
        _report(tmp_path / "x", preds, group_videos, rules_config, evaluation, paired=[(("run", "tcn"), GRU)])


def test_write_reports_gate_and_headline_never_use_supplementary_groups(
    tmp_path: Path,
    supplementary_preds: pd.DataFrame,
    gru_preds: pd.DataFrame,
    both_preds: pd.DataFrame,
    group_videos: pd.DataFrame,
    rules_config: RulesConfig,
    evaluation: Any,
) -> None:
    preds = pd.concat([supplementary_preds, gru_preds], ignore_index=True)
    for key in (INT8_KEY, FPS15_KEY, ORDINAL_KEY):
        with pytest.raises(ValueError, match="must not use a supplementary group"):
            _report(tmp_path / "bad", preds, group_videos, rules_config, evaluation, gate=(key, GRU))
        with pytest.raises(ValueError, match="must not use a supplementary group"):
            _report(tmp_path / "bad", preds, group_videos, rules_config, evaluation, gate=(GRU, key))
    assert not (tmp_path / "bad").exists()

    # 부가 묶음이 표에 있어도 D11의 값과 판정, 비교표의 행은 부가 묶음이 없을 때와 같다
    with_extra = _report(
        tmp_path / "with", preds, group_videos, rules_config, evaluation, gate=(TCN, GRU), synthetic=False
    )
    without = _report(
        tmp_path / "without", both_preds, group_videos, rules_config, evaluation, gate=(TCN, GRU), synthetic=False
    )
    assert with_extra.gate == without.gate and with_extra.gate is not None and not with_extra.gate.passed
    assert with_extra.drowsiness_source == without.drowsiness_source == DROWSY_SOURCE_PERCLOS

    def comparison_rows(report: evaluate.Report) -> list[str]:
        lines = report.summary_path.read_text(encoding="utf-8").splitlines()
        section = lines[lines.index("## 6. 비교표") : lines.index("## 7. 제외 목록")]
        return [line for line in section if line.startswith("| run /")]

    assert comparison_rows(with_extra) == comparison_rows(without)
    assert [row.split(" | ")[0] for row in comparison_rows(with_extra)] == [
        "| run / gru / features=alert_baseline",
        "| run / tcn / features=alert_baseline",
    ]


FAB_RUN = "train-alert_baseline"
FAB_SUBJECTS = {1: (1, 2, 3), 2: (13, 14, 15), 3: (25, 26, 27)}
FAB_CANDIDATE = (FAB_RUN, "selected", "session_baseline")
FAB_REFERENCE = (FAB_RUN, "lgbm", "alert_baseline")
FAB_FPS15 = (*FAB_CANDIDATE, "fps15", "base")


def _fabricated_group(family: str, arm: str, quality: str, **labels: Any) -> pd.DataFrame:
    """세 fold의 피험자 아홉 명, 영상마다 창 40개인 묶음(표지 열은 labels에 준 것만 있다).

    quality: `perfect`는 모든 창이 정답 클래스에 0.98, `poor`는 모든 창이 alert에 0.98, `fair`는 창 10개마다 앞 7개만
    정답 클래스이고 나머지 3개는 다음 클래스다.
    """
    sessions = []
    for subjects in FAB_SUBJECTS.values():
        for subject in subjects:
            for class_index in range(3):
                picks = {
                    "perfect": [class_index] * 40,
                    "poor": [CLASS_ALERT] * 40,
                    "fair": [class_index if i % 10 < 7 else (class_index + 1) % 3 for i in range(40)],
                }[quality]
                probs = [tuple(0.98 if c == pick else 0.01 for c in range(3)) for pick in picks]
                sessions.append(
                    window_rows(
                        subject, class_index, probs, k_start=180, perclos=0.02, family=family, arm=arm,
                        run_id=FAB_RUN, labels=labels,
                    )  # fmt: skip
                )
    return pd.DataFrame([row for session in sessions for row in session])


def test_write_reports_refuses_a_perfect_labelled_group_flagged_as_a_main_result(
    tmp_path: Path, rules_config: RulesConfig, evaluation: Any
) -> None:
    """독립 검증에서 만든 표. 주 결과 묶음 셋은 나쁘고, 15 fps 모사 추출의 묶음 하나는 완벽한데 `supplementary = False`다.

    고치기 전에는 이 표가 그대로 통과했다: 주 결과 묶음이 넷이 되고, 그 묶음을 후보로 준 판정(학습 모델을 기본 졸음
    판정원으로 둘지)이 통과해 판정원이 model이 됐다. 지금은 표를 받지 않는다.
    """
    videos = video_rows(*[(s, c, True) for subjects in FAB_SUBJECTS.values() for s in subjects for c in range(3)])
    main = pd.concat(
        [
            _fabricated_group("selected", "session_baseline", "poor"),
            _fabricated_group("selected", "alert_baseline", "poor"),
            _fabricated_group("lgbm", "alert_baseline", "fair"),
        ],
        ignore_index=True,
    )
    sneaky_group = _fabricated_group(
        "selected", "session_baseline", "perfect", feature_source="fps15", supplementary=False
    )
    sneaky = pd.concat([main, sneaky_group], ignore_index=True)
    common: dict[str, Any] = {
        "fold_subjects": FAB_SUBJECTS,
        "synthetic": False,
        "exclusions": pd.DataFrame(columns=["video_id", "reason"]),
    }

    def report(folder: str, preds: pd.DataFrame, gate: tuple[tuple[str, ...], tuple[str, ...]]) -> evaluate.Report:
        return _report(tmp_path / folder, preds, videos, rules_config, evaluation, gate=gate, **common)

    for gate in ((FAB_FPS15, FAB_REFERENCE), (FAB_CANDIDATE, FAB_REFERENCE)):  # 판정 키와 무관하게 표를 거부한다
        with pytest.raises(ValueError, match="is never a main result and must have supplementary = True") as refused:
            report("sneaky", sneaky, gate)
        assert f"not flagged: [{FAB_FPS15!r}]" in str(refused.value)
    assert not (tmp_path / "sneaky").exists()

    # 표지 열은 있는데 supplementary 열만 없는 예측 파일: 그 묶음은 부가 결과가 되고(전에는 주 결과였다), 판정의 후보가
    # 될 수 없다. 주 결과 후보(나쁜 묶음)의 판정은 통과하지 못한다.
    derived = pd.concat([main, sneaky_group.drop(columns=["supplementary"])], ignore_index=True)
    assert "feature_source" in derived.columns and "supplementary" not in derived.columns
    with pytest.raises(ValueError, match="must not use a supplementary group"):
        report("derived_bad", derived, (FAB_FPS15, FAB_REFERENCE))
    accepted = report("derived", derived, (FAB_CANDIDATE, FAB_REFERENCE))
    assert sorted(result.key for result in accepted.results) == sorted(
        [FAB_CANDIDATE, (FAB_RUN, "selected", "alert_baseline"), FAB_REFERENCE]
    )
    assert [result.key for result in accepted.supplementary] == [FAB_FPS15]
    assert accepted.gate is not None and not accepted.gate.passed
    assert accepted.drowsiness_source == DROWSY_SOURCE_PERCLOS
    by_key = {result.key: result for result in (*accepted.results, *accepted.supplementary)}
    assert by_key[FAB_FPS15].value("video_accuracy") == 1.0  # 완벽한 묶음은 부가 결과에만 있다
    assert by_key[FAB_CANDIDATE].value("video_accuracy") == pytest.approx(1 / 3)  # alert 영상만 맞는다

    # 대조: 완벽한 묶음이 정말 주 결과(원본 추출, 기본 절차)라면 같은 판정이 통과한다. 위의 거부가 막은 것이 이것이다.
    honest = pd.concat(
        [main[main["arm"] != "session_baseline"], _fabricated_group("selected", "session_baseline", "perfect")],
        ignore_index=True,
    )
    passed = report("honest", honest, (FAB_CANDIDATE, FAB_REFERENCE))
    assert passed.gate is not None and passed.gate.passed and passed.drowsiness_source == DROWSY_SOURCE_MODEL
    assert len(passed.results) == 3 and passed.supplementary == ()


def test_write_reports_checks_keys_and_fold_coverage_before_evaluating(
    tmp_path: Path,
    supplementary_preds: pd.DataFrame,
    gru_preds: pd.DataFrame,
    group_videos: pd.DataFrame,
    rules_config: RulesConfig,
    evaluation: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """잘못된 요청은 묶음을 하나도 평가하기 전에(부트스트랩 전에) 거부한다."""
    calls: list[Any] = []
    original = evaluate.evaluate_group

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(args[0]["family"].iloc[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(evaluate, "evaluate_group", counting)
    preds = pd.concat([supplementary_preds, gru_preds], ignore_index=True)
    missing = ("run", "lgbm", "alert_baseline")
    for message, arguments in (
        ("paired comparison names a group without predictions", {"paired": [(TCN, missing)]}),
        ("the D11 gate names a group without predictions", {"gate": (missing, TCN)}),
        ("must not use a supplementary group", {"gate": (INT8_KEY, TCN)}),
    ):
        with pytest.raises(ValueError, match=message):
            _report(tmp_path, preds, group_videos, rules_config, evaluation, **arguments)
    # fold 2의 예측이 없는 묶음이 키 순서로 맨 뒤(tcn_int8)여도 앞의 묶음들을 평가하기 전에 거부한다
    partial = preds[(preds["family"] != "tcn_int8") | (preds["fold"] == 1)]
    with pytest.raises(ValueError, match=r"\('run', 'tcn_int8', 'alert_baseline'\) has no predictions for fold"):
        _report(tmp_path, partial, group_videos, rules_config, evaluation)
    assert calls == [] and not (tmp_path / "report").exists()

    _report(tmp_path, preds, group_videos, rules_config, evaluation, paired=[(TCN, GRU)])
    assert calls == ["gru", "tcn", "tcn", "tcn", "tcn_int8"]  # 묶음 다섯 개를 한 번씩


def test_write_reports_with_supplementary_groups_only(
    tmp_path: Path,
    supplementary_preds: pd.DataFrame,
    group_videos: pd.DataFrame,
    rules_config: RulesConfig,
    evaluation: Any,
) -> None:
    """부가 분석만 담은 리포트(E3 실행을 따로 돌린 경우): 주 결과 절은 비어 있다고 적는다."""
    only_extra = supplementary_preds[supplementary_preds["family"] == "tcn_int8"]
    report = _report(tmp_path, only_extra, group_videos, rules_config, evaluation)
    assert report.results == () and [result.key for result in report.supplementary] == [INT8_KEY]
    summary = report.summary_path.read_text(encoding="utf-8")
    assert summary.count("주 결과 묶음이 없다. 이 리포트의 묶음은 모두 부가 결과다(9절).") == 3
    assert "### 부가: run / tcn_int8 / features=alert_baseline" in summary
    per_fold = pd.read_csv(report.run_dir / "per_fold.csv", skiprows=1)
    assert len(per_fold) == 0 and "video_accuracy" in per_fold.columns
    assert len(pd.read_csv(report.run_dir / "supplementary_per_fold.csv", skiprows=1)) == 2


def test_write_reports_without_supplementary_groups_says_so(
    tmp_path: Path, tcn_preds: pd.DataFrame, group_videos: pd.DataFrame, rules_config: RulesConfig, evaluation: Any
) -> None:
    report = _report(tmp_path, tcn_preds, group_videos, rules_config, evaluation)
    lines = report.summary_path.read_text(encoding="utf-8").splitlines()
    assert lines[lines.index("## 9. 부가 결과") + 2] == "부가 묶음이 없다."
    # 부가 표 파일은 머리글만 있다
    for name in ("supplementary_per_fold.csv", "supplementary_alarm.csv", "supplementary_alarm_sweep.csv"):
        frame = pd.read_csv(report.run_dir / name, skiprows=1)
        assert len(frame) == 0 and list(frame.columns)[:5] == list(evaluate.GROUP_KEYS)
    assert report.supplementary == () and len(report.supplementary_paired) == 0
    payload = json.loads((report.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["supplementary"] == {"groups": [], "paired": []}


# ---------------------------------------------------------------------------
# 기준선과 창 수 (PLAN 4.4절의 리포트 항목)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def baseline_videos(group_videos: pd.DataFrame) -> pd.DataFrame:
    """영상 12개에 기준선 열과 창 수를 붙인 영상 표.

    S02_5: T_b 90초(연장). S13_10: 180초까지 연장한 뒤 전역 상수로 대체. S01_10: off_axis(60초에 완성).
    S14_5: 보정이 끝나지 않음(창 0개). 피험자 14는 alert 영상 기준선을 쓰지 못해 세 영상 모두 alert_fallback.
    """
    videos = group_videos.copy()
    ids = videos["video_id"]
    videos["n_windows"] = np.where(ids == "S14_5", 0, 12)
    videos["face_ratio"] = 0.95
    videos["baseline_status"] = "ok"
    videos.loc[ids == "S13_10", "baseline_status"] = "fallback_global"
    videos.loc[ids == "S01_10", "baseline_status"] = "off_axis"
    videos.loc[ids == "S14_5", "baseline_status"] = "none"
    videos["t_b_ms"] = 60_000
    videos.loc[ids == "S02_5", "t_b_ms"] = 90_000
    videos.loc[ids == "S13_10", "t_b_ms"] = 180_000
    videos.loc[ids == "S14_5", "t_b_ms"] = -1
    videos["baseline_used"] = np.where(videos["subject"] == 14, "global", "alert")
    videos["alert_fallback"] = videos["subject"] == 14
    return videos


WINDOW_COUNTS = {
    "alert_baseline": {
        5: {"alert": 12, "low_vigilant": 9, "drowsy": 12},
        1: {"alert": 48, "low_vigilant": 36, "drowsy": 48},
    },
    "session_baseline": {1: {"alert": 48, "low_vigilant": 36, "drowsy": 48}},
}


def test_write_reports_lists_extended_and_fallback_baselines(
    tmp_path: Path, tcn_preds: pd.DataFrame, baseline_videos: pd.DataFrame, rules_config: RulesConfig, evaluation: Any
) -> None:
    report = _report(tmp_path, tcn_preds, baseline_videos, rules_config, evaluation, window_counts=WINDOW_COUNTS)
    lines = report.summary_path.read_text(encoding="utf-8").splitlines()
    section = lines[lines.index("## 8. 기준선과 창 수 (PLAN 4.4절)") : lines.index("## 9. 부가 결과")]
    assert "- 영상 자신의 기준선 상태별 논리 영상 수: fallback_global 1, none 1, off_axis 1, ok 9" in section
    assert "- 기준선이 연장된 영상(T_b > 60초): 2개" in section
    assert "- 자기 보정이 전역 상수로 대체된 영상(`fallback_global`): 1개" in section
    assert "- 보정이 끝나기 전에 끝난 영상(`none`. 창이 없다): 1개" in section
    assert "- alert 영상 기준선을 쓸 수 없어 전역 상수를 넣은 영상(`alert_fallback`): 3개, 피험자 1명 (14)" in section

    # 목록: 연장 2개(S02_5, S13_10. S13_10은 대체이기도 하다)와 피험자 14의 영상 3개. off_axis만인 영상은 없다
    header = (
        "| video_id | subject | fold | class_index | baseline_status | t_b_ms | baseline_used | alert_fallback "
        "| flags |"
    )
    start = section.index(header)
    assert section[start + 2 : start + 7] == [
        "| S02_5 | 2 | 1 | 1 | ok | 90000 | alert | False | extended |",
        "| S13_10 | 13 | 2 | 2 | fallback_global | 180000 | alert | False | extended, fallback_global |",
        "| S14_0 | 14 | 2 | 0 | ok | 60000 | global | True | alert_fallback |",
        "| S14_5 | 14 | 2 | 1 | none | -1 | global | True | alert_fallback |",
        "| S14_10 | 14 | 2 | 2 | ok | 60000 | global | True | alert_fallback |",
    ]
    assert section[start + 7] == ""

    # 조건별·클래스별 창 수: 영상 표의 값(영상마다 12개, S14_5는 0개)과 넘겨준 조건별 값(보폭 오름차순)
    assert "### 조건별·클래스별 창 수" in section
    table_start = section.index("| 조건 | 보폭(스텝) | alert | low_vigilant | drowsy | 합 |")
    assert section[table_start + 2 : table_start + 6] == [
        "| 영상 표(평가 대상 영상) | 1 | 48 | 36 | 48 | 132 |",
        "| alert_baseline | 1 | 48 | 36 | 48 | 132 |",
        "| alert_baseline | 5 | 12 | 9 | 12 | 33 |",
        "| session_baseline | 1 | 48 | 36 | 48 | 132 |",
    ]

    payload = json.loads((report.run_dir / "summary.json").read_text(encoding="utf-8"))
    baselines = payload["baselines"]
    assert baselines["nominal_ms"] == 60_000 and baselines["columns"] == list(evaluate.BASELINE_COLUMNS)
    assert baselines["status_counts"] == {"fallback_global": 1, "none": 1, "off_axis": 1, "ok": 9}
    assert baselines["extended"] == ["S02_5", "S13_10"]
    assert baselines["fallback_global"] == ["S13_10"] and baselines["not_completed"] == ["S14_5"]
    assert baselines["alert_fallback"] == ["S14_0", "S14_5", "S14_10"] and baselines["alert_fallback_subjects"] == [14]
    assert [row["video_id"] for row in baselines["listed"]] == ["S02_5", "S13_10", "S14_0", "S14_5", "S14_10"]
    assert payload["window_counts"]["alert_baseline"]["5"] == {"alert": 12, "low_vigilant": 9, "drowsy": 12}
    assert list(payload["window_counts"]) == ["alert_baseline", "session_baseline"]

    # videos.csv에 영상마다 기준선 열이 있다
    videos = pd.read_csv(report.run_dir / "videos.csv", skiprows=1)
    assert list(videos.columns) == list(evaluate.VIDEO_COLUMNS) + ["n_windows", "face_ratio"] + list(
        evaluate.BASELINE_COLUMNS
    )
    replaced = videos.set_index("video_id").loc["S13_10"]
    assert (replaced["baseline_status"], replaced["t_b_ms"]) == ("fallback_global", 180_000)
    assert int(videos["alert_fallback"].sum()) == 3


def test_write_reports_baseline_section_follows_the_nominal_duration_and_split_parts(
    tmp_path: Path, tcn_preds: pd.DataFrame, baseline_videos: pd.DataFrame, rules_config: RulesConfig, evaluation: Any
) -> None:
    # 기본 수집 시간을 90초로 주면 90초에 완성된 S02_5는 연장이 아니다
    report = _report(
        tmp_path / "a", tcn_preds, baseline_videos, rules_config, evaluation, nominal_baseline_ms=90_000
    )
    payload = json.loads((report.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["baselines"]["extended"] == ["S13_10"] and payload["window_counts"] is None
    summary = report.summary_path.read_text(encoding="utf-8")
    assert "- 기준선이 연장된 영상(T_b > 90초): 1개" in summary
    assert "조건별 창 수는 전달되지 않았다(`write_reports`의 window_counts). 영상 표의 창 수만 싣는다." in summary

    # 나뉜 영상의 상태 "ok/fallback_global": 부분 하나라도 대체됐으면 대체된 영상으로 센다
    split = baseline_videos.copy()
    split.loc[split["video_id"] == "S01_0", "baseline_status"] = "ok/fallback_global"
    report = _report(tmp_path / "b", tcn_preds, split, rules_config, evaluation)
    payload = json.loads((report.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["baselines"]["fallback_global"] == ["S01_0", "S13_10"]
    assert payload["baselines"]["status_counts"]["ok/fallback_global"] == 1

    # 연장도 대체도 없으면 그렇게 적는다
    plain = baseline_videos.assign(baseline_status="ok", t_b_ms=60_000, alert_fallback=False)
    report = _report(tmp_path / "c", tcn_preds, plain, rules_config, evaluation)
    assert "기준선이 연장되거나 전역 상수로 대체된 영상이 없다." in report.summary_path.read_text(encoding="utf-8")


def test_write_reports_without_baseline_columns_says_what_is_missing(
    tmp_path: Path, tcn_preds: pd.DataFrame, group_videos: pd.DataFrame, rules_config: RulesConfig, evaluation: Any
) -> None:
    """기준선 열도 창 수도 없는 영상 표(손으로 만든 표): 8절에 없다고 적고 나머지는 그대로 쓴다."""
    report = _report(tmp_path, tcn_preds, group_videos, rules_config, evaluation)
    summary = report.summary_path.read_text(encoding="utf-8")
    assert "영상 표에 기준선 열(baseline_status, t_b_ms, baseline_used, alert_fallback)이 없어" in summary
    assert "창 수가 전달되지 않았다(`write_reports`의 window_counts)." in summary
    payload = json.loads((report.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["baselines"] is None and payload["window_counts"] is None


def _session_steps(video_id: str, class_index: int, n_steps: int, full_from: int = 0) -> pd.DataFrame:
    """세션 하나의 스텝 표. 앞의 full_from개는 조회 구간이 덜 찬 스텝이다."""
    data: dict[str, Any] = {
        "video_id": video_id,
        "session_id": video_id,
        "subject": 3,
        "fold": 1,
        "label": LABELS[class_index],
        "class_index": class_index,
        "part": 0,
        "k": np.arange(61, 61 + n_steps),
        "t_end_ms": 1 + 1000 * np.arange(61, 61 + n_steps),
        "full_lookback": np.arange(n_steps) >= full_from,
    }
    for name in FEATURE_NAMES:
        data[name] = np.zeros(n_steps)
    return pd.DataFrame(data, columns=list(STEP_COLUMNS))


def test_window_counts_per_label_stride_and_class() -> None:
    """창 수 = 이어진 다 찬 스텝 n개에서 보폭 1이면 n − 119, 보폭 5이면 (n − 120) // 5 + 1.

    alert 121개 → 2, 1. low_vigilant 130개 → 11, 3. drowsy는 190개 중 앞 60개가 덜 찬 스텝 → 다 찬 130개 → 11, 3.
    두 번째 표는 drowsy가 100개뿐이라 창이 없다.
    """
    steps = pd.concat(
        [
            _session_steps("S03_0", CLASS_ALERT, 121),
            _session_steps("S03_5", CLASS_LOW_VIGILANT, 130),
            _session_steps("S03_10", CLASS_DROWSY, 190, full_from=60),
        ],
        ignore_index=True,
    )
    short = pd.concat(
        [_session_steps("S03_0", CLASS_ALERT, 121), _session_steps("S03_10", CLASS_DROWSY, 100)], ignore_index=True
    )
    counts = evaluate.window_counts({"alert_baseline": steps, "alert_baseline [fps15]": short}, 120, (5, 1))
    assert counts == {
        "alert_baseline": {
            5: {"alert": 1, "low_vigilant": 3, "drowsy": 3},
            1: {"alert": 2, "low_vigilant": 11, "drowsy": 11},
        },
        "alert_baseline [fps15]": {
            5: {"alert": 1, "low_vigilant": 0, "drowsy": 0},
            1: {"alert": 2, "low_vigilant": 0, "drowsy": 0},
        },
    }
    # 실제로 만든 창의 수와 같다
    assert len(make_windows(steps, 120, 1)) == 2 + 11 + 11 and len(make_windows(steps, 120, 5)) == 1 + 3 + 3


# ---------------------------------------------------------------------------
# 라벨이 없는 영상의 시간당 경고 수 (PLAN 11장 M4)
# ---------------------------------------------------------------------------

DRIVE_A_MS = 245_000
DRIVE_B_MS = 75_000
SYNTHETIC_MARK = {"source": "synthetic (drowsy.synth)", "synthetic": True}
UNLABELED_FIRST_LINE = "# UNLABELED videos (no ground truth): the numbers below are alert counts, not accuracy"


def _scripted_drive(spec: PipelineSpec, segments: Sequence[Any], duration_ms: int) -> list[Any]:
    """각본 세션(30 fps). 4초마다 0.2초 깜빡임을 넣되 각본 구간의 앞뒤 1.5초에는 넣지 않는다."""
    busy = [(segment.start_ms - 1500, segment.start_ms + segment.duration_ms + 1500) for segment in segments]
    blinks = [
        synth.Blink(start_ms=start)
        for start in range(3_000, duration_ms, 4_000)
        if not any(low <= start < high for low, high in busy)
    ]
    return synth.render(synth.Scenario(segments=(*segments, *blinks), duration_ms=duration_ms, fps=30.0), spec)


def _drive_a(spec: PipelineSpec) -> list[Any]:
    """245초, 얼굴이 항상 있다.

    20초: 1.5초 감김(기준선 수집 중이라 임시 microsleep)  70초: 1.5초 감김(microsleep)
    80초: 4.5초 동안 고개 35도(3초가 되는 순간 gaze_away_long)
    상태 시작: calibrating 1, microsleep 2, gaze_away_long 1.  경고: microsleep 2, gaze_away_long 1.
    모델 출력: 기준선 60초 + 조회 구간 60초 + 120스텝 = 239초부터 매초. 마지막 프레임이 244.967초이므로 스텝
    239~244의 6개.
    """
    segments = [
        synth.Closure(1500, start_ms=20_000),
        synth.Closure(1500, start_ms=70_000),
        synth.HeadTurn(4500, yaw=35.0, start_ms=80_000),
    ]
    return _scripted_drive(spec, segments, DRIVE_A_MS)


def _drive_b(spec: PipelineSpec) -> list[Any]:
    """75초. 30~40초에는 얼굴이 없다.

    최근 10초의 얼굴 비율이 절반 아래로 내려가는 35초쯤에 unable_to_analyze가 시작해 45초쯤에 끝난다.
    65초: 6.5초 감김(1초에 microsleep, 3초에 sleep, 그 3초 뒤에 sleep 되풀이 경고 한 번. 감김은 71.7초쯤 끝난다).
    상태 시작: calibrating 1, unable_to_analyze 1, microsleep 1, sleep 1.
    상태 종료: calibrating 1, unable_to_analyze 1, sleep 1(한 감김은 가장 높은 종류의 end 하나만 낸다).
    경고: unable_to_analyze 1, microsleep 1, sleep 2(그중 되풀이 1).  모델 출력은 없다(239초보다 짧다).
    """
    segments = [synth.FaceMissing(10_000, start_ms=30_000), synth.Closure(6500, start_ms=65_000)]
    return _scripted_drive(spec, segments, DRIVE_B_MS)


# 프레임 시각은 floor(k·1000/30 + 0.5): 마지막 프레임은 A가 244,967 ms(k = 7349), B가 74,967 ms(k = 2249)다.
# 프레임 간격(33~34 ms)이 max_hold_ms(200)보다 짧으므로 얼굴 프레임이 대표한 시간은 간격의 합이다.
DRIVE_A_ANALYSABLE_MS = 244_967  # 얼굴이 항상 있다: 첫 프레임부터 마지막 프레임까지
DRIVE_B_ANALYSABLE_MS = 74_967 - 10_000  # 얼굴 없는 프레임(30,000 ~ 39,967 ms)이 대표한 10,000 ms를 뺀다
# 경고 수 = 바로 낸 경고 + 미뤘다 낸 경고 + 되풀이 경고. 두 각본에는 속도가 없고 재경고 대기에 걸리는 시작도 없어서
# 미뤘다 낸 경고는 없다(미뤘다 낸 경고가 나오는 각본은 아래 "경고의 종류"에 있다).
DRIVE_A_COUNTS = {
    "state_onsets": {"calibrating": 1, "microsleep": 2, "gaze_away_long": 1},
    "state_ends": {"calibrating": 1, "microsleep": 2, "gaze_away_long": 1},
    "alerts": {"microsleep": 2, "gaze_away_long": 1},
    "immediate_alerts": {"microsleep": 2, "gaze_away_long": 1},
    "deferred_alerts": {},
    "repeat_alerts": {},
}
DRIVE_B_COUNTS = {
    "state_onsets": {"calibrating": 1, "unable_to_analyze": 1, "microsleep": 1, "sleep": 1},
    "state_ends": {"calibrating": 1, "unable_to_analyze": 1, "sleep": 1},
    "alerts": {"unable_to_analyze": 1, "microsleep": 1, "sleep": 2},  # 되풀이 경고도 경고 수에 들어간다
    "immediate_alerts": {"unable_to_analyze": 1, "microsleep": 1, "sleep": 1},
    "deferred_alerts": {},
    "repeat_alerts": {"sleep": 1},
}


def _counts(partial: dict[str, int]) -> dict[str, int]:
    return {name: partial.get(name, 0) for name in EVENT_TYPES}


def _write_unlabeled_config(folder: Path, config_path: Path, repo_root: Path) -> Path:
    """기본 설정에서 경로만 바꾼 설정 파일: spec·규칙·내보내기 폴더는 저장소의 것, 쓰는 폴더는 모두 folder 아래."""
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["paths"]["feature_spec"] = str(repo_root / "config" / "feature_spec.json")
    data["paths"]["rules"] = str(repo_root / "config" / "rules.yaml")
    data["paths"]["export_dir"] = str(repo_root / "export")
    data["paths"]["raw_dir"] = str(folder / "raw")
    data["paths"]["reports_dir"] = str(folder / "reports")
    data["demo"]["model_path"] = str(folder / "not_exported_yet.onnx")  # 없으면 export/drowsiness_v0.onnx를 쓴다
    out = folder / "config.yaml"
    out.write_text(yaml.safe_dump(data), encoding="utf-8")
    return out


@pytest.fixture(scope="module")
def unlabeled_inputs(
    tmp_path_factory: pytest.TempPathFactory, pipeline_spec: PipelineSpec, config_path: Path, repo_root: Path
) -> SimpleNamespace:
    """각본 세션 둘을 추출 결과 형식으로 쓴 폴더와 설정 파일."""
    folder = tmp_path_factory.mktemp("unlabeled")
    raw_dir = folder / "car_raw"
    drives = {"day1/drive_a.parquet": _drive_a(pipeline_spec), "night/drive_b.parquet": _drive_b(pipeline_spec)}
    for name, frames in drives.items():
        extract.write_raw_frames(frames, raw_dir / name, pipeline_spec, SYNTHETIC_MARK)
    return SimpleNamespace(
        folder=folder,
        raw_dir=raw_dir,
        config=_write_unlabeled_config(folder, config_path, repo_root),
        reports=folder / "reports",
        model=repo_root / "export" / "drowsiness_v0.onnx",
    )


@pytest.fixture(scope="module")
def unlabeled_run(unlabeled_inputs: SimpleNamespace) -> SimpleNamespace:
    """명령을 한 번 돌린 결과(속도를 주지 않는다. 모델은 설정의 기본값)."""
    arguments = ["unlabeled", "--config", str(unlabeled_inputs.config), "--raw-dir", str(unlabeled_inputs.raw_dir)]
    arguments += ["--run-name", "fix_m4"]
    code = evaluate.main(arguments)
    run_dir = unlabeled_inputs.reports / "fix_m4" / "unlabeled"
    return SimpleNamespace(code=code, run_dir=run_dir, arguments=arguments)


def _alert_rows(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, skiprows=1, keep_default_na=False)


def test_analysable_ms_hand_computed() -> None:
    # 간격 100(얼굴) + 50(얼굴 없음) + 350(얼굴, 200까지만 인정) + 20(얼굴). 마지막 프레임은 시간을 대표하지 않는다
    t_ms = [1000, 1100, 1150, 1500, 1520]
    assert evaluate.analysable_ms(t_ms, [True, False, True, True, True], 200) == 100 + 200 + 20
    assert evaluate.analysable_ms(t_ms, [False] * 5, 200) == 0
    assert evaluate.analysable_ms(t_ms, [True] * 5, 1000) == 520
    assert evaluate.analysable_ms([7], [True], 200) == 0 and evaluate.analysable_ms([], [], 200) == 0
    with pytest.raises(ValueError, match="same length"):
        evaluate.analysable_ms(t_ms, [True], 200)


def test_unlabeled_command_writes_alerts_csv_and_summary(
    unlabeled_run: SimpleNamespace, unlabeled_inputs: SimpleNamespace
) -> None:
    assert unlabeled_run.code == 0
    names = sorted(path.name for path in unlabeled_run.run_dir.iterdir())
    assert names == ["alerts.csv", "config.yaml", "rules.json", "summary.md"]
    # 설정 사본과 규칙 사본(PLAN 3.4절). 규칙 사본 바이트의 sha256이 리포트에 적힌 rules_hash다
    assert (unlabeled_run.run_dir / "config.yaml").read_bytes() == unlabeled_inputs.config.read_bytes()
    copied_hash = hashlib.sha256((unlabeled_run.run_dir / "rules.json").read_bytes()).hexdigest()
    assert f"rules_hash `{copied_hash}`" in (unlabeled_run.run_dir / "summary.md").read_text(encoding="utf-8")
    # 리포트 폴더 밖에는 아무것도 쓰지 않는다(얼굴 프레임도, 추출 폴더의 부산물도 없다. D13)
    assert sorted(path.name for path in unlabeled_run.run_dir.parent.iterdir()) == ["unlabeled"]
    raw_dir = unlabeled_inputs.raw_dir
    assert sorted(path.relative_to(raw_dir).as_posix() for path in raw_dir.rglob("*.*")) == [
        "day1/drive_a.parquet",
        "night/drive_b.parquet",
    ]

    # 첫 줄: 라벨이 없는 영상이고 숫자는 경고 수이지 정확도가 아니다. 입력이 합성이므로 그 사실도 첫 줄에 있다
    for name in ("summary.md", "alerts.csv"):
        first = (unlabeled_run.run_dir / name).read_text(encoding="utf-8").splitlines()[0]
        assert first == f"# {SYNTHETIC_NOTE}. {UNLABELED_FIRST_LINE[2:]}"
        assert "UNLABELED" in first and "alert counts, not accuracy" in first and first.startswith("# SYNTHETIC data")
    assert f"# {evaluate.UNLABELED_NOTE}" == UNLABELED_FIRST_LINE


def test_unlabeled_counts_match_the_scripted_events(unlabeled_run: SimpleNamespace) -> None:
    rows = _alert_rows(unlabeled_run.run_dir / "alerts.csv")
    assert list(rows.columns) == list(evaluate.UNLABELED_COLUMNS)
    # 영상 둘과 합계: 각각 이벤트 종류 7개 + 전체
    assert rows["video_id"].tolist() == ["day1/drive_a"] * 8 + ["night/drive_b"] * 8 + ["TOTAL"] * 8
    assert rows["event_type"].tolist() == [*EVENT_TYPES, "all"] * 3
    assert (rows["status"] == "ok").all() and rows["synthetic"].all()

    def column(video_id: str, name: str) -> dict[str, Any]:
        own = rows[rows["video_id"] == video_id]
        return dict(zip(own["event_type"], own[name]))

    for video_id, expected, analysable, duration, outputs in (
        ("day1/drive_a", DRIVE_A_COUNTS, DRIVE_A_ANALYSABLE_MS, 244_967, 6),
        ("night/drive_b", DRIVE_B_COUNTS, DRIVE_B_ANALYSABLE_MS, 74_967, 0),
    ):
        assert evaluate.UNLABELED_COUNT_COLUMNS == tuple(expected)
        for name in evaluate.UNLABELED_COUNT_COLUMNS:
            got = column(video_id, name)
            assert {key: got[key] for key in EVENT_TYPES} == _counts(expected[name]), (video_id, name)
            assert got["all"] == sum(expected[name].values())
        own = rows[rows["video_id"] == video_id]
        assert own["analysable_h"].tolist() == pytest.approx([analysable / 3_600_000] * 8)
        assert own["duration_h"].tolist() == pytest.approx([duration / 3_600_000] * 8)
        assert own["face_ratio"].tolist() == pytest.approx([analysable / duration] * 8)
        assert own["baseline_status"].unique().tolist() == ["ok"] and own["t_b_ms"].unique().tolist() == [60_000]
        assert own["model_outputs"].unique().tolist() == [outputs]
        per_hour = column(video_id, "alerts_per_hour")
        n_alerts = sum(expected["alerts"].values())
        for event_type, count in {**_counts(expected["alerts"]), "all": n_alerts}.items():
            assert per_hour[event_type] == pytest.approx(count * 3_600_000 / analysable)

    # 합계: 시간과 횟수를 더한 뒤 나눈다. 분석 가능한 시간 309,934 ms, 경고 3 + 4 = 7번(그중 되풀이 1번)
    total_ms = DRIVE_A_ANALYSABLE_MS + DRIVE_B_ANALYSABLE_MS
    assert total_ms == 309_934
    alerts = column("TOTAL", "alerts")
    assert {key: alerts[key] for key in EVENT_TYPES} == _counts(
        {"microsleep": 3, "sleep": 2, "gaze_away_long": 1, "unable_to_analyze": 1}
    )
    assert alerts["all"] == 7 and column("TOTAL", "repeat_alerts")["all"] == 1
    assert column("TOTAL", "repeat_alerts")["sleep"] == 1
    assert column("TOTAL", "immediate_alerts")["all"] == 6 and column("TOTAL", "deferred_alerts")["all"] == 0
    # 모든 행에서 경고 수는 세 종류의 합이다
    assert (rows["alerts"] == rows["immediate_alerts"] + rows["deferred_alerts"] + rows["repeat_alerts"]).all()
    assert column("TOTAL", "state_onsets")["all"] == 8 and column("TOTAL", "state_ends")["all"] == 7
    per_hour = column("TOTAL", "alerts_per_hour")
    assert per_hour["all"] == pytest.approx(7 * 3_600_000 / 309_934)  # 약 81.3
    assert per_hour["microsleep"] == pytest.approx(3 * 3_600_000 / 309_934)
    assert per_hour["drowsy"] == 0.0
    total = rows[rows["video_id"] == "TOTAL"]
    assert total["analysable_h"].tolist() == pytest.approx([309_934 / 3_600_000] * 8)
    assert total["n_frames"].unique().tolist() == [7350 + 2250] and total["model_outputs"].unique().tolist() == [6]


def test_unlabeled_summary_states_model_hashes_source_and_assumptions(
    unlabeled_run: SimpleNamespace,
    unlabeled_inputs: SimpleNamespace,
    feature_spec: FeatureSpec,
    rules_config: RulesConfig,
) -> None:
    summary = (unlabeled_run.run_dir / "summary.md").read_text(encoding="utf-8")
    lines = summary.splitlines()
    assert "정답이 없으므로 정확도도 오경보율도 아니다." in lines[2]
    assert f"- 생성 명령: `python -m drowsy.evaluate {' '.join(unlabeled_run.arguments)}`" in lines
    assert f"- 입력: `{unlabeled_inputs.raw_dir}`의 추출 결과 2개 (재생 2개, 실패 0개)" in lines
    # 모델: 설정의 demo.model_path가 없어 저장소의 v0(무작위 가중치)을 썼다는 사실까지 적는다
    model_line = next(line for line in lines if line.startswith("- 모델: "))
    assert f"`{unlabeled_inputs.model}`" in model_line and "model_version 0" in model_line
    assert f"sha256 `{file_sha256(unlabeled_inputs.model)}`" in model_line
    assert "trained_on: none (random weights)" in model_line
    assert any("not_exported_yet.onnx" in line and "무작위 가중치 계약 모델" in line for line in lines)
    assert "- 이 모델은 무작위 가중치다. 모델 출력에는 의미가 없다" in lines
    # 해시와 적용된 졸음 판정원
    assert any(f"pipeline_hash `{feature_spec.pipeline_hash}`" in line for line in lines)
    assert any(f"rules_hash `{rules_hash(rules_config)}`" in line for line in lines)
    assert rules_config.drowsiness.source == DROWSY_SOURCE_PERCLOS
    assert any(line.startswith("- 적용된 졸음 판정원(`drowsiness.source`): perclos.") for line in lines)
    # 가정: 속도를 주지 않았으므로 속도 억제가 없다. 분석 가능한 시간의 정의
    speed = next(line for line in lines if line.startswith("- 차량 속도"))
    assert "주지 않았다" in speed and "가정" in speed
    assert "속도 억제(PLAN 1.2절 결정 D3)를 적용하지 않았다" in speed
    # D 번호는 계획서 1.2절의 결정 번호라는 한 줄이 설명 문단 바로 다음, 항목들 앞에 있다. 이 리포트에 나오는 번호는
    # D3뿐이고 나올 때마다 "PLAN 1.2절 결정"이 앞에 붙는다(2026-11-13 소유자 결정 3번으로 읽히지 않게)
    assert lines[4] == PLAN_DECISION_NOTE and lines[3] == lines[5] == "" and lines[6].startswith("- 생성 명령: ")
    assert lines.count(PLAN_DECISION_NOTE) == 1
    body = "\n".join(line for line in lines if line != PLAN_DECISION_NOTE)
    assert re.findall(r"\bD\d+\b", body) == ["D3"] and "(D3)" not in body
    assert re.findall(r".{13}\bD\d+\b", body) == ["PLAN 1.2절 결정 D3"]
    assert any(line.startswith("- 분석 가능한 시간(가정): 얼굴이 검출된 프레임이 대표한 시간의 합") for line in lines)
    # 짧은 영상에서는 drowsy가 판정되지 않는다는 주의(PLAN 4.4절: 기준선 + 조회 구간 60초, 모델은 120스텝 더)
    assert any(line.startswith("- `drowsy`는 기준선이 완성되고 조회 구간 60초가 찬 스텝부터") and "120개" in line for line in lines)

    # 전체 표: 분석 가능한 시간 309,934 ms = 0.0861 h, 경고 7번 → 시간당 7 / 0.086093 = 81.31
    assert "재생한 영상 2개. 분석 가능한 시간 0.0861 h (영상 길이의 합 0.0889 h)." in lines
    # 열: 상태 시작, 상태 종료, 경고 수, 바로 낸 경고, 미뤘다 낸 경고, 되풀이 경고, 시간당 경고 수
    header = ["이벤트", "상태 시작(onset) 수", "상태 종료(end) 수", "경고 수", "바로 낸 경고", "미뤘다 낸 경고", "되풀이 경고"]
    assert "| " + " | ".join([*header, "시간당 경고 수"]) + " |" in lines
    assert "| `microsleep` | 3 | 2 | 3 | 3 | 0 | 0 | 34.85 |" in lines
    assert "| `sleep` | 1 | 1 | 2 | 1 | 0 | 1 | 23.23 |" in lines
    assert "| `drowsy` | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 |" in lines
    assert "| 전체 | 8 | 7 | 7 | 6 | 0 | 1 | 81.31 |" in lines
    assert any(line.startswith("- 경고 수 = 바로 낸 경고(") and "미뤘다 낸 경고(" in line for line in lines)
    # 영상별 표: 길이, 분석 가능한 시간, 얼굴 비율, 기준선, 모델 출력 수, 경고 수와 시간당 값
    row_a = next(line for line in lines if line.startswith("| day1/drive_a |"))
    cells = [cell.strip() for cell in row_a.strip("|").split("|")]
    assert cells[:10] == ["day1/drive_a", "0.0680", "0.0680", "1.000", "ok", "60", "6", "4", "3", "44.09"]
    row_b = next(line for line in lines if line.startswith("| night/drive_b |"))
    cells = [cell.strip() for cell in row_b.strip("|").split("|")]
    # B: 경고 4번 / (64,967 ms = 0.018046 h) = 221.65
    assert cells[:10] == ["night/drive_b", "0.0208", "0.0180", "0.867", "ok", "60", "0", "4", "4", "221.65"]
    assert lines[lines.index("## 3. 재생하지 못한 파일") + 2] == "없다."
    # 4절: 합성 세션에는 추출의 회전 점검 기록이 없다. 그 수를 적고, 확인할 영상은 없다
    section = lines[lines.index("## 4. 회전 점검") :]
    assert "`extraction.unlabeled_rotation_overrides`" in section[2] and "추출 결과 파일 2개:" in section[2]
    assert "| 점검 기록 없음 | 2 |" in section and "확인이 필요한 영상이 없다." in section
    assert any(line.startswith("점검 기록이 없는 파일 2개는") for line in section)
    assert not any("회전이 확인되지 않은 영상" in line for line in lines)


def test_unlabeled_speed_gate(unlabeled_inputs: SimpleNamespace) -> None:
    """속도를 주면 D3의 속도 억제가 걸린다. 상태 이벤트는 그대로이고 경고만 줄어든다.

    10 km/h: microsleep·sleep·gaze_away_long(20 km/h 미만에서 억제)의 경고가 없고(sleep의 되풀이 경고도 없다),
    속도 억제가 없는 unable_to_analyze 하나만 남는다. 30 km/h: 일곱 경고가 모두 나간다(졸음 경고의 50 km/h 기준은
    이 각본에 해당 없음).
    """
    from drowsy.config import load_config

    cfg = load_config(unlabeled_inputs.config)
    common = {"config_path": unlabeled_inputs.config, "raw_dir": unlabeled_inputs.raw_dir}
    slow = evaluate.run_unlabeled(cfg, speed_kmh=10.0, run_name="fix_slow", **common)
    assert slow.run_dir == unlabeled_inputs.reports / "fix_slow" / "unlabeled"
    by_id = {video.video_id: video for video in slow.videos}
    assert by_id["day1/drive_a"].alerts == _counts({})
    assert by_id["night/drive_b"].alerts == _counts({"unable_to_analyze": 1})
    assert by_id["day1/drive_a"].state_onsets == _counts(DRIVE_A_COUNTS["state_onsets"])
    assert by_id["night/drive_b"].state_onsets == _counts(DRIVE_B_COUNTS["state_onsets"])
    total = slow.table[(slow.table["video_id"] == "TOTAL") & (slow.table["event_type"] == "all")].iloc[0]
    assert total["alerts"] == 1 and total["alerts_per_hour"] == pytest.approx(3_600_000 / 309_934)
    summary = slow.summary_path.read_text(encoding="utf-8")
    lines = summary.splitlines()
    assert "- 차량 속도: 10 km/h. 가정: 세션 내내 같은 값으로 두고 경고의 속도 억제(PLAN 1.2절 결정 D3)를 적용했다" in lines
    assert "주지 않았다" not in summary and "(D3)" not in summary and lines[4] == PLAN_DECISION_NOTE
    assert re.findall(r".{13}\bD\d+\b", summary.replace(PLAN_DECISION_NOTE, "")) == ["PLAN 1.2절 결정 D3"]

    fast = evaluate.run_unlabeled(cfg, speed_kmh=30.0, run_name="fix_fast", **common)
    by_id = {video.video_id: video for video in fast.videos}
    assert by_id["day1/drive_a"].alerts == _counts(DRIVE_A_COUNTS["alerts"])
    assert by_id["night/drive_b"].alerts == _counts(DRIVE_B_COUNTS["alerts"])
    assert by_id["night/drive_b"].repeat_alerts == _counts({"sleep": 1})

    for bad in (-1.0, math.nan, math.inf):
        with pytest.raises(evaluate.UnlabeledError, match="speed must be"):
            evaluate.run_unlabeled(cfg, speed_kmh=bad, run_name="fix_bad", **common)
    with pytest.raises(evaluate.UnlabeledError, match="config file to copy"):
        evaluate.run_unlabeled(
            cfg, config_path=unlabeled_inputs.folder / "no.yaml", raw_dir=unlabeled_inputs.raw_dir, run_name="fix_bad"
        )
    assert not (unlabeled_inputs.reports / "fix_bad").exists()


def test_unlabeled_lists_files_it_cannot_replay(
    tmp_path: Path, unlabeled_inputs: SimpleNamespace, pipeline_spec: PipelineSpec, capsys: pytest.CaptureFixture[str]
) -> None:
    """읽을 수 없는 파일과 다른 MediaPipe 모델로 추출한 파일은 이유와 함께 적고 합계에서 뺀다. 실행은 계속한다."""
    raw_dir = tmp_path / "raw"
    frames = _drive_b(pipeline_spec)
    # 합성 표시가 없는 추출 결과(실제 영상의 것과 같은 꼴)
    extract.write_raw_frames(frames, raw_dir / "b.parquet", pipeline_spec, {"source": "car video"})
    other_model = {"source": "car video", "model_sha256": "0" * 64}
    extract.write_raw_frames(frames, raw_dir / "other_model.parquet", pipeline_spec, other_model)
    (raw_dir / "broken.parquet").write_bytes(b"this is not a parquet file")

    code = evaluate.main(
        ["unlabeled", "--config", str(unlabeled_inputs.config), "--raw-dir", str(raw_dir), "--run-name", "fix_partial"]
        + ["--model", str(unlabeled_inputs.model)]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "warning: broken not replayed" in captured.err and "warning: other_model not replayed" in captured.err
    assert "videos replayed 1 of 3" in captured.out and "alerts 4" in captured.out
    # 회전 점검 기록: 셋 다 기록이 없다(추출 메타데이터에 기록이 없는 둘, 읽을 수 없는 파일 하나)
    assert captured.out.splitlines()[-1] == (
        "rotation check of 3 raw file(s): 0 decided by the automatic check, 0 extracted with a manual override, "
        "0 ambiguous without a manual override, 0 with an override that was not applied, "
        "3 without a rotation-check record"
    )

    run_dir = unlabeled_inputs.reports / "fix_partial" / "unlabeled"
    summary = (run_dir / "summary.md").read_text(encoding="utf-8").splitlines()
    # 합성 표시가 없는 입력: 첫 줄은 라벨 없음 문장뿐이다
    assert summary[0] == UNLABELED_FIRST_LINE and "SYNTHETIC" not in summary[0]
    assert (run_dir / "alerts.csv").read_text(encoding="utf-8").splitlines()[0] == UNLABELED_FIRST_LINE
    assert f"- 입력: `{raw_dir}`의 추출 결과 3개 (재생 1개, 실패 2개)" in summary
    assert not any("not_exported_yet" in line for line in summary)  # --model을 줬으므로 대체 모델 안내가 없다
    failed = summary[summary.index("## 3. 재생하지 못한 파일") :]
    assert any(line.startswith("| broken |") for line in failed)
    reason = next(line for line in failed if line.startswith("| other_model |"))
    assert "extracted with MediaPipe model " + "0" * 64 in reason

    rows = _alert_rows(run_dir / "alerts.csv")
    assert rows["video_id"].tolist() == ["b"] * 8 + ["broken", "other_model"] + ["TOTAL"] * 8
    failed_rows = rows[rows["status"] == "failed"]
    assert failed_rows["event_type"].tolist() == ["all", "all"] and (failed_rows["alerts"] == 0).all()
    assert (failed_rows["detail"] != "").all() and not rows["synthetic"].any()
    total = rows[(rows["video_id"] == "TOTAL") & (rows["event_type"] == "all")].iloc[0]
    assert total["alerts"] == 4 and total["analysable_h"] == pytest.approx(DRIVE_B_ANALYSABLE_MS / 3_600_000)
    assert total["detail"] == "1 of 3 videos replayed"


def test_unlabeled_reports_videos_whose_rotation_is_ambiguous_without_an_override(
    tmp_path: Path, unlabeled_inputs: SimpleNamespace, pipeline_spec: PipelineSpec, capsys: pytest.CaptureFixture[str]
) -> None:
    """추출이 남긴 회전 점검 기록(메타데이터는 이 테스트가 적는다)을 읽어 확인이 필요한 영상을 알린다.

    clear: 자동 점검이 정했다.  trip/amb: 가르지 못했고 보정표에 없다(원본은 `.MP4`).
    trip/seen: 가르지 못했지만 보정표에 같은 각도가 있다(사람이 확인했다).  old: 기록이 없다.
    """
    raw_dir = tmp_path / "raw"
    frames = _scripted_drive(pipeline_spec, [], 5_000)  # 5초, 3초에 깜빡임 하나. 경고는 없다
    key = extract.ROTATION_CHECK_KEY

    def record(ambiguous: bool, reason: str) -> dict[str, Any]:
        version = extract.ROTATION_CHECK_VERSION
        return {"version": version, "metadata_rotation": 0, "ambiguous": ambiguous, "reason": reason}

    files = {
        "clear.parquet": {"source": "/car/clear.mp4", "rotation": 0, key: record(False, extract.REASON_BEST_RATE)},
        "trip/amb.parquet": {
            "source": "/car/trip/amb.MP4", "rotation": 90, key: record(True, extract.REASON_AMBIGUOUS_ROLL)
        },
        "trip/seen.parquet": {
            "source": "/car/trip/seen.mov", "rotation": 0, key: record(True, extract.REASON_KEPT_METADATA)
        },
        "old.parquet": {"source": "/car/old.mp4"},
    }  # fmt: skip
    for name, metadata in files.items():
        extract.write_raw_frames(frames, raw_dir / name, pipeline_spec, metadata)
    data = yaml.safe_load(unlabeled_inputs.config.read_text(encoding="utf-8"))
    data["extraction"]["unlabeled_rotation_overrides"] = {"trip/seen.mov": 0}
    data["extraction"]["rotation_overrides"] = {"trip/amb.MP4": 90}  # 데이터셋 추출의 보정표는 여기서 쓰지 않는다
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")

    arguments = ["unlabeled", "--config", str(config), "--raw-dir", str(raw_dir), "--run-name", "fix_rotation"]
    assert evaluate.main(arguments + ["--model", str(unlabeled_inputs.model)]) == 0
    out = capsys.readouterr().out.splitlines()
    run_dir = unlabeled_inputs.reports / "fix_rotation" / "unlabeled"
    assert "videos replayed 4 of 4" in out[-2]  # 표시된 영상도 재생하고 합계에 넣는다
    assert out[-1] == (
        "rotation check of 4 raw file(s): 1 decided by the automatic check, 1 extracted with a manual override, "
        "1 ambiguous without a manual override, 0 with an override that was not applied, "
        "1 without a rotation-check record. LOOK AT THE 1 FLAGGED VIDEO(S) before using this data: "
        f"section 4 of {run_dir / 'summary.md'}"
    )
    lines = (run_dir / "summary.md").read_text(encoding="utf-8").splitlines()
    warning = next(line for line in lines if line.startswith("- **주의: 회전이 확인되지 않은 영상이 1개 있다**"))
    assert "목록은 4절에 있다" in warning
    section = lines[lines.index("## 4. 회전 점검") :]
    for row in (
        "| 자동 점검이 각도를 정함 | 1 |",
        "| 사람이 정한 각도(수동 회전 보정)로 추출됨 | 1 |",
        "| 자동 점검이 각도를 가르지 못했고 수동 보정이 없음(확인 필요) | 1 |",
        "| 보정표에 각도가 있는데 다른 각도로 추출돼 있음(확인 필요) | 0 |",
        "| 점검 기록 없음 | 1 |",
        # 보정표에 적을 키: 추출 결과의 상대 경로에 원본 영상의 확장자를 되살린 것
        "| `trip/amb.MP4` | 가르지 못함, 수동 보정 없음 | 90 | 0 | `ambiguous_smallest_roll` | - |",
    ):
        assert row in section, row
    assert not any("trip/seen" in line or "clear" in line or "`old" in line for line in section)
    rows = _alert_rows(run_dir / "alerts.csv")
    assert rows["video_id"].unique().tolist() == ["clear", "old", "trip/amb", "trip/seen", "TOTAL"]
    assert (rows["status"] == "ok").all()

    from drowsy.config import load_config

    report = evaluate.run_unlabeled(
        load_config(config), config_path=config, raw_dir=raw_dir, model_path=unlabeled_inputs.model, run_name="fix_api"
    )
    assert report.rotation.flagged == (
        RotationFlag("trip/amb.MP4", ROTATION_AMBIGUOUS, 90, 0, "ambiguous_smallest_roll", None),
    )
    assert (report.rotation.n_files, report.rotation.n_decided, report.rotation.n_manual) == (4, 1, 1)


def test_unlabeled_refuses_a_run_name_that_is_not_a_plain_folder_name(
    unlabeled_inputs: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--run-name ../escape_test`는 리포트(설정 사본, 규칙 사본 포함)를 리포트 폴더 밖에 쓰게 된다. 시작 전에 거부한다."""
    base = ["unlabeled", "--config", str(unlabeled_inputs.config), "--raw-dir", str(unlabeled_inputs.raw_dir)]
    for bad in ("../escape_test", "a/b", "..", ".", "", "a\\b"):
        assert evaluate.main([*base, "--run-name", bad]) == 2
        captured = capsys.readouterr()
        assert captured.err == f"error: --run-name must be a plain folder name, got {bad!r}\n"
        assert captured.out == ""
    # 리포트 폴더(folder / "reports") 밖에도 안에도 생긴 것이 없다
    assert not (unlabeled_inputs.folder / "escape_test").exists() and not (unlabeled_inputs.reports / "a").exists()
    # 다른 일보다 먼저 검사한다: 설정 파일이 없어도 이름의 오류가 나온다
    missing = ["unlabeled", "--config", str(unlabeled_inputs.folder / "no.yaml"), "--run-name", "../escape_test"]
    assert evaluate.main(missing) == 2
    assert capsys.readouterr().err == "error: --run-name must be a plain folder name, got '../escape_test'\n"
    # 함수로 불러도 같다
    from drowsy.config import load_config

    with pytest.raises(ValueError, match="--run-name must be a plain folder name, got '../escape_test'"):
        evaluate.run_unlabeled(
            load_config(unlabeled_inputs.config),
            config_path=unlabeled_inputs.config,
            raw_dir=unlabeled_inputs.raw_dir,
            run_name="../escape_test",
        )
    assert not (unlabeled_inputs.folder / "escape_test").exists()


def test_unlabeled_cannot_start(
    tmp_path: Path, unlabeled_inputs: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """추출 결과가 없거나 모델이 spec과 맞지 않으면 종료 코드 2이고 아무것도 쓰지 않는다."""
    import onnx

    from drowsy.export_onnx import exported_spec_path

    base = ["unlabeled", "--config", str(unlabeled_inputs.config), "--run-name", "fix_none"]

    def run(*extra: str) -> str:
        assert evaluate.main([*base, *extra]) == 2
        return capsys.readouterr().err

    empty = tmp_path / "empty"
    empty.mkdir()
    assert "no video files" in run("--raw-dir", str(empty))
    missing = run("--raw-dir", str(tmp_path / "missing"))
    assert "does not exist" in missing and "python -m drowsy.extract --unlabeled" in missing
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "drive.mp4").write_bytes(b"")
    assert "holds video files, not raw parquet files" in run("--raw-dir", str(videos))
    # 모든 파일이 재생되지 않는 폴더
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "x.parquet").write_bytes(b"junk")
    assert "none of the 1 raw parquet files" in run("--raw-dir", str(broken))
    (broken / "TOTAL.parquet").write_bytes(b"junk")  # 합계 행의 이름과 겹치는 파일
    assert "reserves for the totals row" in run("--raw-dir", str(broken))

    # 모델: 없는 파일, 그리고 pipeline_hash가 다른 모델(데모와 같은 대조: `demo.verify_model`)
    raw = ["--raw-dir", str(unlabeled_inputs.raw_dir)]
    assert "ONNX model not found" in run(*raw, "--model", str(tmp_path / "no.onnx"))
    proto = onnx.load(str(unlabeled_inputs.model))
    for prop in proto.metadata_props:
        if prop.key == "pipeline_hash":
            prop.value = "0" * 64
    copied = tmp_path / "model" / unlabeled_inputs.model.name
    copied.parent.mkdir()
    onnx.save(proto, str(copied))
    shutil.copy(exported_spec_path(unlabeled_inputs.model), copied.parent)
    assert "pipeline_hash mismatch" in run(*raw, "--model", str(copied))
    assert "speed must be" in run(*raw, "--speed-kmh", "-5")
    assert "error:" in run(*raw, "--config", str(tmp_path / "no_config.yaml"))

    assert not (unlabeled_inputs.reports / "fix_none").exists()


def test_unlabeled_default_raw_dir_and_options_before_the_command(
    unlabeled_inputs: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--raw-dir을 주지 않으면 추출 모듈이 정한 폴더(`extract.unlabeled_raw_dir(cfg)`)를 읽는다."""
    asked: list[Any] = []

    def raw_dir_of(cfg: Any) -> Path:
        asked.append(cfg.paths.raw_dir)
        return unlabeled_inputs.raw_dir / "night"

    monkeypatch.setattr(extract, "unlabeled_raw_dir", raw_dir_of, raising=False)
    # --config는 하위 명령 앞에 적어도 된다. 실행 이름을 주지 않으면 교차검증·최종 학습과 같은 폴더다
    assert evaluate.main(["--config", str(unlabeled_inputs.config), "unlabeled"]) == 0
    assert asked == [unlabeled_inputs.folder / "raw"]
    out = capsys.readouterr().out
    assert "videos replayed 1 of 1" in out and "alerts 4" in out
    run_dir = unlabeled_inputs.reports / evaluate.DEFAULT_RUN_NAME / "unlabeled"
    rows = _alert_rows(run_dir / "alerts.csv")
    assert rows["video_id"].unique().tolist() == ["drive_b", "TOTAL"]


def test_unlabeled_default_run_name_is_the_cross_validation_folder() -> None:
    from drowsy import train

    assert evaluate.DEFAULT_RUN_NAME == train.DEFAULT_CV_RUN_NAME


def test_unlabeled_table_totals_are_sums_not_means() -> None:
    """합계의 시간당 값 = 경고 수의 합 / 시간의 합. 영상별 값의 평균(여기서는 (60 + 6)/2 = 33)이 아니다."""

    def video(video_id: str, analysable_ms: int, alerts: int, status: str = "ok") -> evaluate.UnlabeledVideo:
        zero = dict.fromkeys(EVENT_TYPES, 0)
        return evaluate.UnlabeledVideo(
            video_id=video_id,
            path=Path(f"{video_id}.parquet"),
            status=status,
            detail="" if status == "ok" else "ValueError: unreadable",
            synthetic=False,
            n_frames=10,
            duration_ms=2 * analysable_ms,
            analysable_ms=analysable_ms,
            baseline_status="ok",
            t_b_ms=60_000,
            model_outputs=0,
            state_onsets={**zero, "microsleep": alerts + 1},
            state_ends={**zero, "microsleep": alerts + 1},
            alerts={**zero, "microsleep": alerts},
            immediate_alerts={**zero, "microsleep": alerts - 2},
            deferred_alerts={**zero, "microsleep": 2},
            repeat_alerts=zero,
        )

    # 짧은 영상: 6분에 경고 6번(시간당 60). 긴 영상: 1시간에 6번(시간당 6). 실패한 영상은 들어가지 않는다
    table = evaluate.unlabeled_table(
        [video("short", 360_000, 6), video("long", 3_600_000, 6), video("bad", 3_600_000, 99, status="failed")]
    )
    rows = table[table["event_type"] == "all"].set_index("video_id")
    expected = {"short": 60.0, "long": 6.0, "bad": 0.0, "TOTAL": 12 / 1.1}
    assert rows["alerts_per_hour"].to_dict() == pytest.approx(expected)
    assert rows.loc["TOTAL", "alerts"] == 12 and rows.loc["TOTAL", "analysable_h"] == pytest.approx(1.1)
    assert rows.loc["TOTAL", "state_onsets"] == 14 and rows.loc["TOTAL", "face_ratio"] == pytest.approx(0.5)
    # 종류별 수도 더한다: 바로 낸 경고 4 + 4, 미뤘다 낸 경고 2 + 2
    assert (rows.loc["TOTAL", "immediate_alerts"], rows.loc["TOTAL", "deferred_alerts"]) == (8, 4)
    assert (rows.loc["bad", "immediate_alerts"], rows.loc["bad", "deferred_alerts"]) == (0, 0)
    assert rows.loc["bad", "status"] == "failed" and rows.loc["bad", "alerts"] == 0
    assert len(table[table["video_id"] == "bad"]) == 1 and len(table[table["video_id"] == "short"]) == 8


# ---------------------------------------------------------------------------
# 경고의 종류: 바로 낸 것, 미뤘다 낸 것, 되풀이 (2026-11-13 소유자 결정 2 "막힌 경고는 막힘이 풀렸을 때 상태가
# 이어지고 있으면 한 번 낸다"에 따른 구분)
# ---------------------------------------------------------------------------

KINDS_DRIVE_MS = 310_000
KINDS_ANALYSABLE_MS = 309_900  # 10 fps, 얼굴이 항상 있다: 첫 프레임(0 ms)부터 마지막 프레임(309,900 ms)까지


def _kinds_drive(spec: PipelineSpec) -> list[Any]:
    """경고의 세 종류가 모두 나오는 각본(310초, 10 fps, 첫 프레임의 t_ms는 1). 판정은 100 ms 격자에서 난다.

    세션 시간(초)                    손 계산
      72.0  고개 35도 9초            72.2에 영역 밖 -> 75.2에 gaze_away_long 시작, 81.2에 종료
     105.0  고개 35도 6초            108.2에 시작, 111.2에 종료
     140.0  감김(머무는 시간 1.5초)  141.1에 microsleep 시작(감김의 시작은 보간한 140.07), 141.7에 종료
     143.5  감김(2.7초)              144.6에 microsleep 시작, 146.4에 종료(sleep은 아니다)
     160.0  감김(10.5초)             161.1에 microsleep, 163.1에 sleep 시작, 170.7에 sleep 종료
     181.0  감김(12초)               182.1에 microsleep, 184.1에 sleep 시작, 193.2에 sleep 종료
    상태 시작: calibrating 1, gaze_away_long 2, microsleep 4, sleep 2.  종료: calibrating 1, gaze_away_long 2,
    microsleep 2(sleep까지 간 감김은 sleep의 end만 낸다), sleep 2.  (tests/test_demo.py의 `kinds_frames`와 같은 각본이다.)
    """
    segments = [
        synth.HeadTurn(9000, yaw=35.0, start_ms=72_000),
        synth.HeadTurn(6000, yaw=35.0, start_ms=105_000),
        synth.Closure(1500, start_ms=140_000),
        synth.Closure(2700, start_ms=143_500),
        synth.Closure(10_500, start_ms=160_000),
        synth.Closure(12_000, start_ms=181_000),
    ]
    return synth.render(synth.Scenario(segments=tuple(segments), duration_ms=KINDS_DRIVE_MS, fps=10.0), spec)


def _changing_speed(t_ms: int) -> float:
    """세션 시간에 따라 바뀌는 속도(km/h): 80초까지 10, 100초까지 60, 130초까지 10, 180초까지 60, 186초까지 10,
    230초까지 60, 300초까지 45, 그 뒤 100. 경고 기준은 시선·감김 20, 졸음 50이다(D3)."""
    t = t_ms - 1
    for until, speed in ((80_000, 10.0), (100_000, 60.0), (130_000, 10.0), (180_000, 60.0), (186_000, 10.0),
                         (230_000, 60.0), (300_000, 45.0)):  # fmt: skip
        if t < until:
            return speed
    return 100.0


@pytest.fixture(scope="module")
def kinds_parquet(tmp_path_factory: pytest.TempPathFactory, pipeline_spec: PipelineSpec) -> Path:
    path = tmp_path_factory.mktemp("kinds") / "car_raw" / "kinds.parquet"
    extract.write_raw_frames(_kinds_drive(pipeline_spec), path, pipeline_spec, SYNTHETIC_MARK)
    return path


def test_unlabeled_counts_alert_kinds_when_the_speed_gate_opens_during_a_state(
    kinds_parquet: Path, feature_spec: FeatureSpec, rules_config: RulesConfig
) -> None:
    """속도가 바뀌는 재생: 막혔다가 상태가 이어지는 동안 나간 첫 경고는 되풀이가 아니라 미뤘다 낸 경고로 센다.

    졸음 판정원은 model, 모델은 늘 p_drowsy 0.95(스텝 248에서 drowsy 시작). 손으로 센 경고(t_ms):
      gaze_away_long  75.2초의 시작은 10 km/h라 막힘 -> 60 km/h가 된 80 001에 미뤘다 냄. 108.2초의 시작은 10 km/h에서
                      막힌 채 끝나 경고 없음.                                             바로 0, 미뤘다 1, 되풀이 0
      microsleep      141 101 바로. 144.6초의 시작은 대기 5초에 걸려 146 101에 미뤘다 냄. 161 101 바로. 182.1초의 시작은
                      10 km/h에서 막혔고 sleep 시작이 지웠다.                              바로 2, 미뤘다 1, 되풀이 0
      sleep           163 101 바로 + 166 101, 169 101 되풀이. 184.1초의 시작은 10 km/h라 막힘 -> 186 001에 미뤘다 냄
                      + 189 001, 192 001 되풀이.                                          바로 1, 미뤘다 1, 되풀이 4
      drowsy          248초의 시작은 45 km/h라 막힘 -> 100 km/h가 된 300 001에 미뤘다 냄.  바로 0, 미뤘다 1, 되풀이 0
    고치기 전에는 미뤘다 낸 넷이 모두 "되풀이"로 세어졌다(되풀이 8).
    """
    rules = apply_overrides(rules_config, {"drowsiness.source": DROWSY_SOURCE_MODEL})
    video = evaluate.replay_unlabeled(
        kinds_parquet, "kinds", feature_spec, rules, lambda window: (0.02, 0.03, 0.95), _changing_speed
    )
    assert video.status == "ok" and video.detail == "" and video.synthetic
    assert video.analysable_ms == KINDS_ANALYSABLE_MS and video.model_outputs == 71  # 스텝 239~309
    assert video.state_onsets == _counts(
        {"calibrating": 1, "gaze_away_long": 2, "microsleep": 4, "sleep": 2, "drowsy": 1}
    )
    assert video.state_ends == _counts({"calibrating": 1, "gaze_away_long": 2, "microsleep": 2, "sleep": 2})
    assert video.alerts == _counts({"gaze_away_long": 1, "microsleep": 3, "sleep": 6, "drowsy": 1})
    assert video.immediate_alerts == _counts({"microsleep": 2, "sleep": 1})
    assert video.deferred_alerts == _counts({"gaze_away_long": 1, "microsleep": 1, "sleep": 1, "drowsy": 1})
    assert video.repeat_alerts == _counts({"sleep": 4})

    rows = evaluate.unlabeled_table([video])
    assert list(rows.columns) == list(evaluate.UNLABELED_COLUMNS)
    total = rows[(rows["video_id"] == "TOTAL")].set_index("event_type")
    assert tuple(int(total.loc["all", name]) for name in evaluate.UNLABELED_COUNT_COLUMNS) == (10, 7, 11, 3, 4, 4)
    assert tuple(int(total.loc["sleep", name]) for name in evaluate.UNLABELED_COUNT_COLUMNS) == (2, 2, 6, 1, 1, 4)
    assert (rows["alerts"] == rows["immediate_alerts"] + rows["deferred_alerts"] + rows["repeat_alerts"]).all()
    assert total.loc["all", "alerts_per_hour"] == pytest.approx(11 * 3_600_000 / KINDS_ANALYSABLE_MS)


def test_unlabeled_report_counts_a_cooldown_deferred_alert(
    kinds_parquet: Path, unlabeled_inputs: SimpleNamespace
) -> None:
    """명령과 같은 경로(`run_unlabeled`, 속도는 세션 내내 30 km/h)의 리포트. 미뤘다 낸 경고는 대기가 풀린 하나다.

    30 km/h에서는 시선·감김 경고가 모두 나가고 졸음 경고(50 km/h부터)는 나가지 않는다. 손으로 센 경고:
      gaze_away_long  75.2초, 108.2초 바로(둘 사이가 대기 10초보다 길다)                    2 = 2 + 0 + 0
      microsleep      141.1초 바로, 144.6초의 시작은 대기 5초에 걸려 146.1초에 미뤘다 냄,
                      161.1초와 182.1초 바로                                              4 = 3 + 1 + 0
      sleep           163.1초 바로 + 166.1, 169.1초 되풀이. 184.1초 바로 + 187.1, 190.1,
                      193.1초 되풀이(193.2초에 종료)                                       7 = 2 + 0 + 5
    합계 13 = 7 + 1 + 5. 분석 가능한 시간 309,900 ms -> 시간당 13 × 3,600,000 / 309,900 = 151.02.
    """
    from drowsy.config import load_config

    cfg = load_config(unlabeled_inputs.config)
    report = evaluate.run_unlabeled(
        cfg, config_path=unlabeled_inputs.config, raw_dir=kinds_parquet.parent, speed_kmh=30.0, run_name="fix_kinds"
    )
    (video,) = report.videos
    assert video.alerts == _counts({"gaze_away_long": 2, "microsleep": 4, "sleep": 7})
    assert video.immediate_alerts == _counts({"gaze_away_long": 2, "microsleep": 3, "sleep": 2})
    assert video.deferred_alerts == _counts({"microsleep": 1})
    assert video.repeat_alerts == _counts({"sleep": 5})
    assert video.state_onsets["microsleep"] == 4 and video.state_ends["microsleep"] == 2
    assert video.state_onsets["drowsy"] >= 1  # 기본 판정원(perclos)의 상태는 생기지만 30 km/h에서는 경고가 없다

    rows = _alert_rows(report.alerts_path)
    assert list(rows.columns) == list(evaluate.UNLABELED_COLUMNS)
    total = rows[rows["video_id"] == "TOTAL"].set_index("event_type")
    kind_columns = ("alerts", "immediate_alerts", "deferred_alerts", "repeat_alerts")
    assert tuple(int(total.loc["all", name]) for name in kind_columns) == (13, 7, 1, 5)
    lines = report.summary_path.read_text(encoding="utf-8").splitlines()
    assert "| `gaze_away_long` | 2 | 2 | 2 | 2 | 0 | 0 | 23.23 |" in lines
    assert "| `microsleep` | 4 | 2 | 4 | 3 | 1 | 0 | 46.47 |" in lines
    assert "| `sleep` | 2 | 2 | 7 | 2 | 0 | 5 | 81.32 |" in lines
    overall = next(line for line in lines if line.startswith("| 전체 |"))
    assert [cell.strip() for cell in overall.strip("|").split("|")][3:] == ["13", "7", "1", "5", "151.02"]

