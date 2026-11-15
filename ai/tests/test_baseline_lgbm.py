"""drowsy.baseline_lgbm 테스트 (PLAN 5.3절, 6장 "교란 점검", 11장 E1). 합성 표와 합성 세션만 쓴다.

손으로 만든 스텝 표(`make_steps`)의 특징 값은 난수이고 파이프라인의 출력이 아니다. 교차검증의 부기를 보는 데만 쓴다.
교란 점검용 표(`probe_table`)와 명령은 합성 세션을 `DriverPipeline`에 넣어 만든 데이터로 본다.
리포트 문구에 나오는 D11은 계획서(docs/PLAN.md 1.2절 표)의 결정 번호다: 교차검증 결과가 정해 둔 조건을 모두 채울 때만 모델을
기본 졸음 판정원으로 두고, 그 조건의 하나가 기준 LightGBM과의 비교다.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import yaml

from drowsy import baseline_lgbm, evaluate, synth, train
from drowsy.config import ExperimentConfig, load_config
from drowsy.dataset import (
    ID_COLUMNS,
    LABEL_TO_CLASS,
    STEP_COLUMNS,
    DatasetSource,
    StepTable,
    SubjectOverlapError,
    apply_quality_gate,
    build_manifest,
    build_step_table,
    make_windows,
    nested_splits,
    official_fold,
    spec_pipeline_hash,
)
from drowsy.extract import read_extract_metadata, read_raw_frames, write_raw_frames
from drowsy.spec import FeatureSpec, PipelineSpec, RulesConfig, apply_overrides
from drowsy.synth import Blink, FaceMissing, Scenario, default_profile, render
from drowsy.types import BASELINE_FALLBACK_GLOBAL, FEATURE_NAMES, N_FEATURES, Baseline

ARM = "alert_baseline"
FOUR_FOLDS = (1, 2, 13, 14, 25, 26, 37, 38)  # 공식 fold 1~4에 두 명씩


# ---------------------------------------------------------------------------
# 도우미
# ---------------------------------------------------------------------------


def make_steps(subjects: tuple[int, ...], n_steps: int = 134, seed: int = 0) -> pd.DataFrame:
    """피험자마다 세 라벨의 세션 하나씩. 모든 스텝이 full_lookback이다(보폭 1에서 창 15개, 보폭 5에서 3개)."""
    rng = np.random.default_rng(seed)
    frames = []
    for subject in subjects:
        for label, class_index in LABEL_TO_CLASS.items():
            video_id = f"S{subject:02d}_{label}"
            k = np.arange(121, 121 + n_steps)
            data: dict[str, Any] = {
                "video_id": video_id, "session_id": video_id, "subject": subject, "fold": official_fold(subject),
                "label": label, "class_index": class_index, "part": 0, "k": k, "t_end_ms": 1 + 1000 * k,
                "full_lookback": True,
            }  # fmt: skip
            values = np.clip(0.1 + 0.25 * class_index + 0.03 * rng.standard_normal((n_steps, N_FEATURES)), 0.0, 1.0)
            for index, name in enumerate(FEATURE_NAMES):
                data[name] = values[:, index]
            frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True)[list(STEP_COLUMNS)]


def make_data(feature_spec: FeatureSpec, subjects: tuple[int, ...]) -> train.CVData:
    steps = make_steps(subjects)
    rows = steps.drop_duplicates("session_id")[list(ID_COLUMNS)].reset_index(drop=True)
    rows["raw_status"] = "ok"
    rows["baseline_status"] = "ok"
    rows["face_ms"] = 95_000
    rows["duration_ms"] = 100_000
    rows["n_windows"] = steps.groupby("session_id", sort=False).size().to_numpy() - 119
    quality = apply_quality_gate(rows)
    return train.CVData(
        feature_spec=feature_spec,
        manifest=quality[list(ID_COLUMNS)].copy(),
        raw_dir=Path("unused"),
        tables={ARM: StepTable(ARM, feature_spec.pipeline_hash, steps, quality)},
        extra_tables={},
        synthetic=True,
        description="SYNTHETIC data (hand-built table)",
    )


def tiny_config(cfg: ExperimentConfig) -> ExperimentConfig:
    return dataclasses.replace(
        cfg,
        lgbm=dataclasses.replace(
            cfg.lgbm, num_leaves=(3, 5), min_data_in_leaf=(2, 4), max_rounds=15, early_stopping_rounds=5
        ),
        evaluation=dataclasses.replace(cfg.evaluation, bootstrap_n=200),
    )


@pytest.fixture(scope="module")
def cfg(experiment_config: ExperimentConfig) -> ExperimentConfig:
    return tiny_config(experiment_config)


# ---------------------------------------------------------------------------
# 창 통계
# ---------------------------------------------------------------------------


def test_window_stats_against_hand_computed_values() -> None:
    # 창 하나, 스텝 4개, 특징 2개: 특징 0은 0, 1, 2, 3 (직선), 특징 1은 10, 10, 10, 13.
    x = np.array([[[0.0, 10.0], [1.0, 10.0], [2.0, 10.0], [3.0, 13.0]]], dtype=np.float32)
    stats = baseline_lgbm.window_stats(x)
    assert stats.shape == (1, 12) and stats.dtype == np.float64
    expected = [
        1.5, 10.75,  # mean
        math.sqrt(1.25), math.sqrt(1.6875),  # std (모표준편차)
        0.0, 10.0,  # min
        3.0, 13.0,  # max
        3.0, 13.0,  # last
        1.0, 0.9,  # slope: 특징 1은 Σ(t − 1.5)(x − 10.75) / Σ(t − 1.5)² = 4.5 / 5
    ]  # fmt: skip
    np.testing.assert_allclose(stats[0], expected, rtol=0.0, atol=1e-12)
    assert baseline_lgbm.window_stat_names(("a", "b")) == (
        "a_mean", "b_mean", "a_std", "b_std", "a_min", "b_min", "a_max", "b_max", "a_last", "b_last",
        "a_slope", "b_slope",
    )  # fmt: skip

    # 120 × 17 창은 102개 입력이 된다. 기울기는 최소제곱 직선의 것이다(numpy.polyfit과 비교).
    rng = np.random.default_rng(0)
    windows = rng.normal(0.0, 1.0, size=(5, 120, 17)).astype(np.float32)
    stats = baseline_lgbm.window_stats(windows)
    assert stats.shape == (5, 102) and len(baseline_lgbm.window_stat_names()) == 102
    assert baseline_lgbm.window_stat_names()[:2] == ("perclos_60_mean", "perclos_10_mean")
    assert baseline_lgbm.window_stat_names()[-1] == "face_missing_ratio_10_slope"
    steps = np.arange(120)
    for n, feature in ((0, 0), (3, 16), (4, 7)):
        slope = np.polyfit(steps, windows[n, :, feature].astype(np.float64), 1)[0]
        assert stats[n, 5 * 17 + feature] == pytest.approx(slope, abs=1e-12)
        assert stats[n, 4 * 17 + feature] == windows[n, -1, feature]
        assert stats[n, 0 * 17 + feature] == pytest.approx(windows[n, :, feature].astype(np.float64).mean())

    constant = baseline_lgbm.window_stats(np.full((2, 7, 3), 4.0))
    np.testing.assert_array_equal(constant[:, :3], 4.0)  # mean
    np.testing.assert_array_equal(constant[:, 3:6], 0.0)  # std
    np.testing.assert_array_equal(constant[:, 15:], 0.0)  # slope
    one_step = baseline_lgbm.window_stats(np.array([[[2.0, 5.0]]]))
    np.testing.assert_array_equal(one_step[0], [2.0, 5.0, 0.0, 0.0, 2.0, 5.0, 2.0, 5.0, 2.0, 5.0, 0.0, 0.0])
    assert baseline_lgbm.window_stats(np.zeros((0, 120, 17))).shape == (0, 102)
    with pytest.raises(ValueError, match="shape"):
        baseline_lgbm.window_stats(np.zeros((120, 17)))
    with pytest.raises(ValueError, match="at least one step"):
        baseline_lgbm.window_stats(np.zeros((2, 0, 17)))


def test_sample_weights_mirror_the_subject_video_window_sampler() -> None:
    rows = []
    counts = {(1, "S01_0"): 10, (1, "S01_5"): 2, (1, "S01_10"): 4, (2, "S02_0"): 3, (2, "S02_10"): 1}
    for (subject, video), n in counts.items():
        rows.append(pd.DataFrame({"subject": subject, "video_id": video, "k_end": np.arange(n)}))
    meta = pd.concat(rows, ignore_index=True)
    weights = baseline_lgbm.sample_weights(meta)
    assert weights.mean() == pytest.approx(1.0)
    meta["w"] = weights
    by_subject = meta.groupby("subject")["w"].sum()
    assert by_subject[1] == pytest.approx(by_subject[2])  # 피험자마다 같은 비중
    by_video = meta.groupby(["subject", "video_id"])["w"].sum()
    assert by_video[(1, "S01_0")] == pytest.approx(by_video[(1, "S01_5")]) == pytest.approx(by_video[(1, "S01_10")])
    # 영상이 둘뿐인 피험자는 영상마다 1/2, 셋인 피험자는 1/3을 갖는다.
    assert by_video[(2, "S02_0")] == pytest.approx(by_video[(2, "S02_10")])
    assert by_video[(2, "S02_0")] == pytest.approx(1.5 * by_video[(1, "S01_0")])
    assert meta.groupby("video_id")["w"].nunique().max() == 1  # 한 영상 안에서는 창마다 같다
    assert len(baseline_lgbm.sample_weights(meta.iloc[:0])) == 0


# ---------------------------------------------------------------------------
# 후보, 선택, 학습
# ---------------------------------------------------------------------------


def test_candidates_params_and_tie_rule(experiment_config: ExperimentConfig) -> None:
    grid = baseline_lgbm.lgbm_candidates(experiment_config.lgbm)
    assert [(c.num_leaves, c.min_data_in_leaf) for c in grid] == [(7, 20), (7, 60), (15, 20), (15, 60)]
    assert grid[1].name == "lgbm(num_leaves=7,min_data_in_leaf=60)"
    params = baseline_lgbm.lgbm_params(experiment_config.lgbm, 7, 20)
    assert params["objective"] == "multiclass" and params["num_class"] == 3
    assert (params["num_leaves"], params["min_data_in_leaf"]) == (7, 20)
    assert params["learning_rate"] == experiment_config.lgbm.learning_rate and params["seed"] == 0
    assert params["deterministic"] is True and params["metric"] == "None"

    def score(leaves: int, min_data: int, mean: float, std: float) -> baseline_lgbm.LgbmScore:
        candidate = baseline_lgbm.LgbmCandidate(leaves, min_data)
        return baseline_lgbm.LgbmScore(candidate, (1, 2), (mean, mean), mean, std, (10, 20), 15)

    # 평균이 가장 높은 후보의 표준편차 안이면 잎이 적은 쪽, 잎 수가 같으면 min_data_in_leaf가 큰 쪽이다.
    scores = [score(7, 20, 0.70, 0.0), score(7, 60, 0.69, 0.0), score(15, 20, 0.72, 0.035), score(15, 60, 0.71, 0.0)]
    assert baseline_lgbm.choose_lgbm(scores).candidate == baseline_lgbm.LgbmCandidate(7, 60)
    scores[2] = score(15, 20, 0.72, 0.015)  # 표준편차 안에 드는 것은 (15, 60)뿐이다
    assert baseline_lgbm.choose_lgbm(scores).candidate == baseline_lgbm.LgbmCandidate(15, 60)
    scores[2] = score(15, 20, 0.72, 0.0)
    assert baseline_lgbm.choose_lgbm(scores).candidate == baseline_lgbm.LgbmCandidate(15, 20)


def test_fit_lgbm_early_stopping_uses_the_video_weighted_macro_f1(
    cfg: ExperimentConfig, feature_spec: FeatureSpec
) -> None:
    data = make_data(feature_spec, FOUR_FOLDS)
    steps = data.tables[ARM].steps
    train_windows = train.subset_subjects(make_windows(steps, 120, 5), (1, 2, 13, 14, 25, 26))
    val_windows = train.subset_subjects(make_windows(steps, 120, 1), (37, 38))
    x, val_x = baseline_lgbm.window_stats(train_windows.x), baseline_lgbm.window_stats(val_windows.x)
    weights = baseline_lgbm.sample_weights(train_windows.meta)
    params = baseline_lgbm.lgbm_params(cfg.lgbm, 3, 2)

    booster, rounds, macro_f1 = baseline_lgbm.fit_lgbm(
        x, train_windows.y, weights, params, 15, val=(val_x, val_windows.meta), early_stopping_rounds=5
    )
    assert 1 <= rounds <= 15 and booster.num_feature() == 102
    # 돌려준 값은 채택한 회차의 모델로 다시 계산한 영상 가중 macro-F1(신경망의 조기 종료와 같은 지표)이다.
    probs = booster.predict(val_x, num_iteration=rounds)
    assert macro_f1 == pytest.approx(train.VideoMacroF1(val_windows.meta)(np.argmax(probs, axis=1)))
    assert macro_f1 > 0.9  # 클래스마다 수준이 다른 표라 쉽게 갈린다
    again = baseline_lgbm.fit_lgbm(
        x, train_windows.y, weights, params, 15, val=(val_x, val_windows.meta), early_stopping_rounds=5
    )
    assert again[1:] == (rounds, macro_f1)
    np.testing.assert_array_equal(again[0].predict(val_x, num_iteration=rounds), probs)  # 결정적

    # 검증 없이: 정한 횟수를 그대로 학습한다.
    fixed, fixed_rounds, score = baseline_lgbm.fit_lgbm(x, train_windows.y, weights, params, 4)
    assert fixed_rounds == 4 and fixed.current_iteration() == 4 and math.isnan(score)


# ---------------------------------------------------------------------------
# 중첩 교차검증
# ---------------------------------------------------------------------------


def test_lgbm_nested_cv_uses_the_same_folds_and_never_touches_the_test_fold(
    cfg: ExperimentConfig, feature_spec: FeatureSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = make_data(feature_spec, FOUR_FOLDS)
    fits: list[dict[str, Any]] = []
    weighed: list[set[int]] = []
    real_fit, real_weights = baseline_lgbm.fit_lgbm, baseline_lgbm.sample_weights

    def spy_weights(meta: pd.DataFrame) -> np.ndarray:
        weighed.append({int(value) for value in meta["subject"].unique()})
        return real_weights(meta)

    def spy_fit(x: Any, y: Any, weights: Any, params: Any, rounds: int, val: Any = None, **kwargs: Any) -> Any:
        entry = {
            "train": weighed[-1],
            "n_train": len(x),
            "val": {int(value) for value in val[1]["subject"].unique()} if val is not None else None,
            "params": (params["num_leaves"], params["min_data_in_leaf"]),
            "rounds": rounds,
        }
        fits.append(entry)
        result = real_fit(x, y, weights, params, rounds, val=val, **kwargs)
        entry["best"] = result[1]
        return result

    monkeypatch.setattr(baseline_lgbm, "sample_weights", spy_weights)
    monkeypatch.setattr(baseline_lgbm, "fit_lgbm", spy_fit)
    preds, records, timings = baseline_lgbm.lgbm_cross_validate(data, cfg, ARM, "train-alert_baseline")
    assert set(timings) == {"window_stats_s", "cross_validation_s"}

    everyone = set(FOUR_FOLDS)
    folds = {fold: {s for s in FOUR_FOLDS if official_fold(s) == fold} for fold in (1, 2, 3, 4)}
    assert len(fits) == 4 * (4 * 3 + 1)  # 바깥 fold 4 × (후보 4 × 안쪽 분할 3 + 다시 학습 1)
    for position, test_fold in enumerate((1, 2, 3, 4)):
        block = fits[position * 13 : (position + 1) * 13]
        inner, refit = block[:12], block[12]
        for index, entry in enumerate(inner):
            val_fold = [fold for fold in (1, 2, 3, 4) if fold != test_fold][index % 3]
            assert entry["val"] == folds[val_fold]
            assert entry["train"] == everyone - folds[test_fold] - folds[val_fold]
            assert entry["rounds"] == cfg.lgbm.max_rounds
            assert entry["n_train"] == 4 * 3 * 3  # 학습 보폭 5초: 피험자 4명 × 영상 3개 × 창 3개
        assert [entry["params"] for entry in inner[::3]] == [(3, 2), (3, 4), (5, 2), (5, 4)]
        record = records[position]
        assert record["fold"] == test_fold and record["test_subjects"] == sorted(folds[test_fold])
        chosen = next(item for item in record["candidates"] if item["name"] == record["chosen"]["lgbm"])
        # 다시 학습: 학습 fold 전체, 고른 후보, 안쪽 분할이 고른 횟수의 중앙값.
        assert refit["val"] is None and refit["train"] == everyone - folds[test_fold]
        assert refit["params"] == (chosen["num_leaves"], chosen["min_data_in_leaf"])
        picked = [entry["best"] for entry in inner if entry["params"] == refit["params"]]
        assert refit["rounds"] == chosen["rounds"] == train.median_half_up(picked) and chosen["fold_rounds"] == picked
        assert record["chosen_mean_macro_f1"] == chosen["mean_macro_f1"] == pytest.approx(
            float(np.mean(chosen["fold_macro_f1"]))
        )

    # 예측 표: 평가 보폭 1초의 시험 창마다 한 행, 시드 하나, 온도 1.0.
    preds = evaluate.validate_predictions(preds)
    assert list(preds.columns) == list(evaluate.PREDICTION_COLUMNS)
    assert set(preds["family"]) == {"lgbm"} and set(preds["arm"]) == {ARM}
    assert set(preds["run_id"]) == {"train-alert_baseline"}
    assert set(preds["seed"]) == {cfg.lgbm.seed} and set(preds["temperature"]) == {1.0}
    assert len(preds) == len(FOUR_FOLDS) * 3 * 15
    assert (preds["fold"] == preds["subject"].map(official_fold)).all()
    assert np.allclose(preds[list(evaluate.PROB_COLUMNS)].sum(axis=1), 1.0)

    assert set(preds["variant"]) == {"base"} and not preds["supplementary"].any()

    gate = baseline_lgbm.e1_gate(records, 0.40)
    assert gate["passed"] is True and gate["threshold"] == 0.40
    assert gate["inner_macro_f1_mean"] == pytest.approx(np.mean([r["chosen_mean_macro_f1"] for r in records]))
    assert baseline_lgbm.e1_gate(records, 1.01)["passed"] is False


def test_lgbm_cv_stops_on_injected_subject_overlap(
    cfg: ExperimentConfig, feature_spec: FeatureSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = make_data(feature_spec, FOUR_FOLDS)
    calls: list[Any] = []
    monkeypatch.setattr(baseline_lgbm, "fit_lgbm", lambda *args, **kwargs: calls.append(args))
    manifest = data.manifest.copy()
    manifest.loc[manifest["video_id"] == "S13_10", "fold"] = 4
    with pytest.raises(SubjectOverlapError, match="subject 13"):
        baseline_lgbm.lgbm_cross_validate(dataclasses.replace(data, manifest=manifest), cfg, ARM, "r")

    def leaky_splits(_: pd.DataFrame) -> Any:
        for split in nested_splits(data.manifest):
            yield dataclasses.replace(split, train_subjects=split.train_subjects + split.test_subjects[:1])

    monkeypatch.setattr(baseline_lgbm, "nested_splits", leaky_splits)
    with pytest.raises(SubjectOverlapError):
        baseline_lgbm.lgbm_cross_validate(data, cfg, ARM, "r")
    assert calls == []
    with pytest.raises(ValueError, match="no step table"):
        baseline_lgbm.lgbm_cross_validate(data, cfg, "global", "r")


def test_lgbm_cross_validate_labels_a_variant_as_supplementary(
    cfg: ExperimentConfig, feature_spec: FeatureSpec
) -> None:
    # W 비교에서 기준 W가 아닌 묶음: 같은 run_id, 다른 variant, 부가 결과로 표시된다. 확률은 같은 표라 기본 묶음과 같다.
    data = make_data(feature_spec, FOUR_FOLDS)
    base, _, _ = baseline_lgbm.lgbm_cross_validate(data, cfg, ARM, "train-alert_baseline")
    other, _, _ = baseline_lgbm.lgbm_cross_validate(data, cfg, ARM, "train-alert_baseline", variant="closure_w1")
    assert baseline_lgbm.closure_w_variant(1.0) == "closure_w1" and baseline_lgbm.closure_w_variant(0.0) == "closure_w0"
    assert baseline_lgbm.closure_w_variant(0.25) == "closure_w0.25"
    assert set(other["variant"]) == {"closure_w1"} and other["supplementary"].all()
    probs = list(evaluate.PROB_COLUMNS)
    np.testing.assert_array_equal(other[probs].to_numpy(), base[probs].to_numpy())
    both = evaluate.validate_predictions(pd.concat([base, other], ignore_index=True))
    assert evaluate.supplementary_pairs(both) == [
        (("train-alert_baseline", "lgbm", ARM, "native", "closure_w1"), ("train-alert_baseline", "lgbm", ARM))
    ]


# ---------------------------------------------------------------------------
# 교란 점검: 표를 받는 함수
# ---------------------------------------------------------------------------


def probe_frame(n_folds: int = 5, per_fold: int = 6, seed: int = 0) -> pd.DataFrame:
    """영상마다 한 행인 표. `leak`은 라벨을 그대로 드러내고 `noise`는 라벨과 무관하다."""
    rng = np.random.default_rng(seed)
    rows = []
    for fold in range(1, n_folds + 1):
        for index in range(per_fold):
            subject = 12 * (fold - 1) + index + 1
            for label, class_index in LABEL_TO_CLASS.items():
                rows.append(
                    {
                        "video_id": f"S{subject:02d}_{label}", "subject": subject, "fold": fold,
                        "class_index": class_index, "leak": class_index + 0.1 * rng.standard_normal(),
                        "noise": rng.standard_normal(),
                    }
                )  # fmt: skip
    return pd.DataFrame(rows)


def test_confound_probe_flags_a_leaking_column_and_not_noise(
    cfg: ExperimentConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = probe_frame()
    checked: list[tuple[set[int], set[int]]] = []
    real_check = baseline_lgbm.assert_subject_disjoint

    def spy_check(train_subjects: Any, test_subjects: Any) -> None:
        checked.append(({int(s) for s in train_subjects}, {int(s) for s in test_subjects}))
        real_check(train_subjects, test_subjects)

    monkeypatch.setattr(baseline_lgbm, "assert_subject_disjoint", spy_check)
    leak = baseline_lgbm.confound_probe(table, ["leak"], "leak_only", cfg)
    assert leak.name == "leak_only" and leak.columns == ("leak",) and leak.missing_columns == ()
    assert (leak.n_videos, leak.n_subjects, leak.chance) == (90, 30, pytest.approx(1.0 / 3.0))
    assert leak.accuracy > 0.95 and leak.ci_low > 0.9 and leak.above_chance is True
    assert leak.ci_low <= leak.accuracy <= leak.ci_high
    assert sum(sum(row) for row in leak.confusion) == 90 and len(leak.confusion) == 3
    assert set(leak.fold_accuracy) == {1, 2, 3, 4, 5} and min(leak.recall) > 0.9
    # fold마다 그 fold의 피험자를 빼고 학습했다(피험자 분리).
    assert len(checked) == 5
    for train_subjects, test_subjects in checked:
        assert len(test_subjects) == 6 and not train_subjects & test_subjects
        assert train_subjects | test_subjects == set(table["subject"])

    # 라벨과 무관한 열은 우연 수준(1/3) 근처에 머문다. 95% 구간으로 판정하므로 표 하나는 우연히 넘을 수 있어
    # (시드 40개에서 잰 정확도는 평균 0.325, 표준편차 0.065, 넘은 비율 5%였다) 표 여러 개로 본다.
    noise = [baseline_lgbm.confound_probe(probe_frame(seed=seed), ["noise"], "noise_only", cfg) for seed in range(8)]
    assert abs(float(np.mean([result.accuracy for result in noise])) - 1.0 / 3.0) < 0.08
    assert max(result.accuracy for result in noise) < 0.6
    assert sum(result.above_chance for result in noise) <= 2
    assert all(result.above_chance == (result.ci_low > 1.0 / 3.0) for result in noise)

    with pytest.raises(ValueError, match="lacks columns"):
        baseline_lgbm.confound_probe(table, ["nope"], "x", cfg)
    with pytest.raises(ValueError, match="at least one input column"):
        baseline_lgbm.confound_probe(table, [], "x", cfg)
    with pytest.raises(ValueError, match="more than once"):
        baseline_lgbm.confound_probe(pd.concat([table, table.iloc[:1]]), ["leak"], "x", cfg)
    with pytest.raises(ValueError, match="at least 2 folds"):
        baseline_lgbm.confound_probe(table[table["fold"] == 1], ["leak"], "x", cfg)


def test_the_two_plan_probes_take_the_metadata_table(cfg: ExperimentConfig) -> None:
    table = probe_frame(seed=1)
    rng = np.random.default_rng(2)
    for name in ("mean_fps", "width", "height", "face_size", "face_ratio"):
        table[name] = rng.standard_normal(len(table))
    table["open_closure_median"] = table["leak"]  # 기준선 편차가 라벨을 드러내는 경우
    table["yaw_rel_median"] = rng.standard_normal(len(table))
    table["pitch_rel_median"] = math.nan  # 값이 하나도 없는 열은 없는 열로 친다

    recording = baseline_lgbm.recording_metadata_probe(table, cfg)
    assert recording.name == "recording_metadata"
    assert recording.columns == ("mean_fps", "width", "height", "face_size", "face_ratio")
    assert recording.missing_columns == ("brightness", "variable_fps") and recording.above_chance is False
    offset = baseline_lgbm.baseline_offset_probe(table, cfg)
    assert offset.name == "baseline_offset" and offset.columns == ("open_closure_median", "yaw_rel_median")
    assert offset.missing_columns == ("pitch_rel_median",) and offset.above_chance is True

    # 밝기 열이 있으면(추출 단계가 기록해 주면) 녹화 메타데이터 점검이 그것도 쓴다.
    table["brightness"] = table["leak"]
    with_brightness = baseline_lgbm.recording_metadata_probe(table, cfg)
    assert with_brightness.columns[-1] == "brightness" and with_brightness.missing_columns == ("variable_fps",)
    assert with_brightness.above_chance is True
    # 가변 프레임률 표시만 라벨을 드러내는 경우도 잡는다. 값이 모두 비어 있는 열은 없는 열이다.
    table["brightness"] = math.nan
    table["variable_fps"] = (table["class_index"] == 2).astype(float)
    with_flag = baseline_lgbm.recording_metadata_probe(table, cfg)
    assert with_flag.columns[-1] == "variable_fps" and with_flag.missing_columns == ("brightness",)
    assert baseline_lgbm.RECORDING_COLUMNS == (
        "mean_fps", "width", "height", "face_size", "face_ratio", "brightness", "variable_fps",
    )  # fmt: skip
    with pytest.raises(ValueError, match="none of the columns"):
        baseline_lgbm.baseline_offset_probe(table.drop(columns=["open_closure_median", "yaw_rel_median"]), cfg)


# ---------------------------------------------------------------------------
# 교란 점검용 표: 합성 세션을 DriverPipeline에 넣어 만든다
# ---------------------------------------------------------------------------

FPS = 10.0


@pytest.fixture(scope="module")
def scripted(
    tmp_path_factory: pytest.TempPathFactory, pipeline_spec: PipelineSpec, rules_config: RulesConfig
) -> SimpleNamespace:
    """피험자 2명의 각본 세션(130초, 10 fps).

    - 1번: alert 영상은 기본 얼굴(정면). 5 영상은 뜬 눈이 좁고 고개가 yaw 6도, pitch 4도 돌아가 있다.
      drowsy 영상은 `10_1`(같은 얼굴 B)과 `10_2`(yaw −4도)로 나뉘어 있다.
    - 13번: alert 영상의 앞 170초 동안 얼굴이 없어 보정이 전역 상수로 대체된다(모든 세션이 전역 상수 기준선을 쓴다).
    """
    spec = pipeline_spec
    root = tmp_path_factory.mktemp("scripted") / "raw"
    base = default_profile(spec)
    face_b = dataclasses.replace(base, e_open_l=0.25, e_open_r=0.25, yaw=6.0, pitch=4.0)
    face_c = dataclasses.replace(base, yaw=-4.0)
    blinks = tuple(Blink(start_ms=start) for start in range(65_000, 125_000, 5_000))

    def write(rel_path: str, profile: Any, duration_s: int = 130, segments: tuple = blinks, **kwargs: Any) -> None:
        scenario = Scenario(segments=segments, duration_ms=1000 * duration_s, fps=FPS, profile=profile, **kwargs)
        write_raw_frames(render(scenario, spec), root / rel_path, spec, {"synthetic": True})

    write("Fold1_part1/01/0.parquet", base)
    write("Fold1_part1/01/5.parquet", face_b)
    write("Fold1_part1/01/10_1.parquet", face_b)
    write("Fold1_part1/01/10_2.parquet", face_c, w=1280, h=720)
    write("Fold2_part1/13/0.parquet", base, duration_s=200, segments=(FaceMissing(170_000),))
    write("Fold2_part1/13/5.parquet", face_b)
    write("Fold2_part1/13/10.parquet", face_c)
    manifest = build_manifest(root)
    rules = dataclasses.replace(rules_config, clock=spec.clock)
    quality = {arm: build_step_table(manifest, root, spec, arm).quality for arm in (ARM, "session_baseline")}
    tables = {arm: baseline_lgbm.probe_table(quality[arm], manifest, root, spec, rules) for arm in quality}
    return SimpleNamespace(root=root, manifest=manifest, quality=quality, tables=tables, rules=rules, spec=spec)


def test_probe_table_offsets_follow_the_arm_baseline(scripted: SimpleNamespace) -> None:
    quality = scripted.quality[ARM]
    assert quality.set_index("session_id")["baseline_used"].to_dict() == {
        "S01_0": "alert", "S01_5": "alert", "S01_10_1": "alert", "S01_10_2": "alert",
        "S13_0": "global", "S13_5": "global", "S13_10": "global",
    }  # fmt: skip
    table = scripted.tables[ARM]
    assert list(table.columns) == [
        "video_id", "subject", "fold", "class_index", "mean_fps", "width", "height", "face_size", "face_ratio",
        "brightness", "variable_fps", "open_closure_median", "yaw_rel_median", "pitch_rel_median",
    ]  # fmt: skip
    # 합성 세션의 parquet에는 추출이 적는 밝기와 가변 프레임률 표시가 없다. 값이 비어 있을 뿐 표는 만들어진다.
    assert table[["brightness", "variable_fps"]].isna().all().all()
    assert "그 메타데이터가 없다" in baseline_lgbm.metadata_note(table)
    assert table["video_id"].tolist() == ["S01_0", "S01_5", "S01_10", "S13_0", "S13_5", "S13_10"]  # 논리 영상마다 한 행
    row = table.set_index("video_id")

    # alert 영상은 자기 기준선 대비라 편차가 0이다.
    assert row.loc["S01_0", ["open_closure_median", "yaw_rel_median", "pitch_rel_median"]].abs().max() < 1e-6
    # 5 영상은 alert 영상의 기준선 대비: 고개 6도·4도, 뜬 눈이 좁은 만큼의 closure.
    assert row.loc["S01_5", "yaw_rel_median"] == pytest.approx(6.0, abs=0.05)
    assert row.loc["S01_5", "pitch_rel_median"] == pytest.approx(4.0, abs=0.05)
    # EAR 쪽만 (0.27 − 0.25) / (0.27 · 0.65)만큼 좁아지고 블렌드셰이프 쪽은 그대로다: closure = 0.5 · 0.114.
    assert row.loc["S01_5", "open_closure_median"] == pytest.approx(0.5 * 0.02 / (0.27 * 0.65), abs=2e-3)
    # 나뉜 영상은 부분별 값의 평균이다: yaw (6 + (−4)) / 2, 해상도 (640 + 1280) / 2.
    assert row.loc["S01_10", "yaw_rel_median"] == pytest.approx(1.0, abs=0.05)
    assert row.loc["S01_10", "width"] == 960.0 and row.loc["S01_10", "height"] == 600.0
    # 13번은 전역 상수 기준선(정면, 단위 행렬) 대비다.
    assert row.loc["S13_5", "yaw_rel_median"] == pytest.approx(6.0, abs=0.05)
    assert row.loc["S13_10", "yaw_rel_median"] == pytest.approx(-4.0, abs=0.05)
    assert abs(row.loc["S13_0", "yaw_rel_median"]) < 1e-6

    # 녹화 메타데이터.
    assert np.allclose(table["mean_fps"], FPS, atol=0.05)
    assert row.loc["S01_0", "width"] == 640.0 and row.loc["S01_0", "height"] == 480.0
    assert (table["face_size"] > 0.05).all() and (table["face_size"] < 1.0).all()
    assert row.loc["S01_0", "face_ratio"] > 0.99 and row.loc["S13_0", "face_ratio"] < 0.2

    # 자기 세션 기준선 조건에서는 모든 영상의 편차가 0이다(전역 상수로 대체된 13번의 alert 영상도 정면이다).
    own = scripted.tables["session_baseline"].set_index("video_id")
    assert own[["yaw_rel_median", "pitch_rel_median"]].abs().to_numpy().max() < 1e-6
    assert own["open_closure_median"].abs().max() < 1e-6
    pd.testing.assert_frame_equal(
        scripted.tables["session_baseline"][["mean_fps", "width", "height", "face_size", "face_ratio"]],
        table[["mean_fps", "width", "height", "face_size", "face_ratio"]],
    )


def test_probe_table_parallel_missing_files_and_bad_input(scripted: SimpleNamespace, tmp_path: Path) -> None:
    quality = scripted.quality[ARM]
    parallel = baseline_lgbm.probe_table(
        quality, scripted.manifest, scripted.root, scripted.spec, scripted.rules, workers=2
    )
    pd.testing.assert_frame_equal(parallel, scripted.tables[ARM])

    # 추출 결과가 없는 세션은 값이 NaN이다(다른 폴더를 가리키면 전부 없다).
    empty = baseline_lgbm.probe_table(quality, scripted.manifest, tmp_path, scripted.spec, scripted.rules)
    assert len(empty) == 6 and empty[["mean_fps", "open_closure_median", "yaw_rel_median"]].isna().all().all()
    assert empty["face_ratio"].notna().all()  # 얼굴 검출 비율은 품질 표의 값이다

    with pytest.raises(ValueError, match="manifest does not have"):
        baseline_lgbm.probe_table(
            quality, scripted.manifest[scripted.manifest["subject"] == 1], scripted.root, scripted.spec, scripted.rules
        )
    with pytest.raises(ValueError, match="workers"):
        baseline_lgbm.probe_table(quality, scripted.manifest, scripted.root, scripted.spec, scripted.rules, workers=0)


def test_probe_table_reads_brightness_and_timing_from_the_extraction_metadata(
    scripted: SimpleNamespace, tmp_path: Path
) -> None:
    """교란 점검 (1): 추출이 parquet 메타데이터에 적은 밝기, 해상도, 평균 프레임률, 가변 프레임률 표시를 쓴다."""
    # 각본 세션을 다시 쓰면서 추출이 적는 키를 넣는다(`extract.extract_video`의 메타데이터 키). 13번의 두 영상에는 넣지 않는다.
    recorded = {
        "Fold1_part1/01/0.parquet": {"brightness_mean": 80.0, "variable_intervals": False, "mean_fps": 29.97,
                                     "width": 1920, "height": 1080},
        "Fold1_part1/01/5.parquet": {"brightness_mean": 120.5, "variable_intervals": True, "mean_fps": 24.0,
                                     "width": 1280, "height": 720},
        "Fold1_part1/01/10_1.parquet": {"brightness_mean": 100.0, "variable_intervals": True, "mean_fps": 30.0,
                                        "width": 640, "height": 480},
        "Fold1_part1/01/10_2.parquet": {"brightness_mean": 60.0, "variable_intervals": False, "mean_fps": 20.0,
                                        "width": 1280, "height": 720},
        "Fold2_part1/13/0.parquet": {"brightness_mean": 33.0, "variable_intervals": False},
    }  # fmt: skip
    root = tmp_path / "raw"
    for source in sorted(scripted.root.rglob("*.parquet")):
        rel_path = source.relative_to(scripted.root).as_posix()
        metadata = {"synthetic": True, **recorded.get(rel_path, {})}
        write_raw_frames(read_raw_frames(source), root / rel_path, scripted.spec, metadata)
    assert read_extract_metadata(root / "Fold1_part1/01/5.parquet")["brightness_mean"] == 120.5
    table = baseline_lgbm.probe_table(scripted.quality[ARM], scripted.manifest, root, scripted.spec, scripted.rules)
    row = table.set_index("video_id")
    # 밝기와 가변 프레임률 표시는 메타데이터의 값이다. 나뉜 영상은 부분의 평균(표시는 그런 부분의 비율)이다.
    assert row["brightness"].to_dict() == pytest.approx(
        {"S01_0": 80.0, "S01_5": 120.5, "S01_10": 80.0, "S13_0": 33.0, "S13_5": math.nan, "S13_10": math.nan},
        nan_ok=True,
    )
    assert row["variable_fps"].to_dict() == pytest.approx(
        {"S01_0": 0.0, "S01_5": 1.0, "S01_10": 0.5, "S13_0": 0.0, "S13_5": math.nan, "S13_10": math.nan}, nan_ok=True
    )
    # 평균 프레임률과 해상도는 메타데이터에 있으면 그 값(원본 영상의 것), 없으면 저장된 프레임에서 구한 값이다.
    assert row.loc["S01_0", ["mean_fps", "width", "height"]].tolist() == [29.97, 1920.0, 1080.0]
    assert row.loc["S01_10", ["mean_fps", "width", "height"]].tolist() == [25.0, 960.0, 600.0]
    assert row.loc["S13_0", "mean_fps"] == pytest.approx(FPS, abs=0.05) and row.loc["S13_0", "width"] == 640.0
    # 다른 열은 메타데이터와 무관하다.
    same = ["face_size", "face_ratio", "open_closure_median", "yaw_rel_median", "pitch_rel_median"]
    pd.testing.assert_frame_equal(table[same], scripted.tables[ARM][same])
    note = baseline_lgbm.metadata_note(table)
    assert "`brightness` 4개" in note and "`variable_fps` 4개" in note and "전체 6개" in note

    # 세션 하나의 값: 메타데이터가 없으면 저장된 프레임에서, 수가 아닌 값은 없는 것으로.
    frames = list(read_raw_frames(root / "Fold1_part1/01/0.parquet"))
    plain = baseline_lgbm._recording_values(frames, scripted.spec)
    assert math.isnan(plain["brightness"]) and math.isnan(plain["variable_fps"])
    assert plain["mean_fps"] == pytest.approx(FPS, abs=0.05) and (plain["width"], plain["height"]) == (640.0, 480.0)
    odd = baseline_lgbm._recording_values(
        frames, scripted.spec, {"brightness_mean": "bright", "variable_intervals": None, "mean_fps": math.inf}
    )
    assert math.isnan(odd["brightness"]) and math.isnan(odd["variable_fps"])
    assert odd["mean_fps"] == pytest.approx(FPS, abs=0.05)
    empty = baseline_lgbm._recording_values([], scripted.spec, {"brightness_mean": 7, "variable_intervals": True})
    assert (empty["brightness"], empty["variable_fps"]) == (7.0, 1.0) and math.isnan(empty["face_size"])


# ---------------------------------------------------------------------------
# 명령: 합성 세션 → 스텝 표 → cv, probes
# ---------------------------------------------------------------------------

TINY_SUBJECTS = (1, 13, 25)  # 공식 fold 1, 2, 3에 한 명씩


# 합성 데이터(세션 파일 10개)에서 명령이 둘째 줄에 출력하는 회전 점검 기록: 합성 세션에는 기록이 없다.
# 회전이 확인되지 않은 영상이 있는 데이터에서 네 명령이 무엇을 출력하고 남기는지는 tests/test_train.py의 `flagged_run`이 본다.
SYNTHETIC_ROTATION_LINE = (
    "rotation check of 10 raw file(s): 0 decided by the automatic check, 0 extracted with a manual override, "
    "0 ambiguous without a manual override, 0 with an override that was not applied, "
    "10 without a rotation-check record"
)
SYNTHETIC_ROTATION_RECORD = {
    "overrides_key": "extraction.rotation_overrides", "n_files": 10, "n_missing": 0, "decided_by_check": 0,
    "manual_override": 0, "ambiguous_without_override": 0, "override_not_applied": 0, "no_record": 10, "flagged": [],
}  # fmt: skip


@pytest.fixture(scope="module")
def tiny_cli(
    tmp_path_factory: pytest.TempPathFactory, config_path: Path, repo_root: Path, pipeline_spec: PipelineSpec
) -> SimpleNamespace:
    tmp = tmp_path_factory.mktemp("tiny_cli")
    synth_dir = tmp / "data" / "synth"
    for job in synth.session_jobs(max(TINY_SUBJECTS), 4.1, 0):
        if job.subject in TINY_SUBJECTS:
            synth.write_session(job, synth_dir, pipeline_spec)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    # 합성 데이터로 도는 설정이므로 rldd_root를 직접 null로 적는다(기본 설정의 값에 기대지 않는다).
    data["paths"].update(
        rldd_root=None,
        synth_dir=str(synth_dir),
        feature_spec=str(repo_root / "config" / "feature_spec.json"),
        rules=str(repo_root / "config" / "rules.yaml"),
        reports_dir=str(tmp / "reports"),
    )
    data["extraction"]["workers"] = 1
    data["fast"].update(lgbm_num_leaves=[3], lgbm_min_data_in_leaf=[1], lgbm_max_rounds=5, bootstrap_n=20)
    config = tmp / "config.yaml"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    return SimpleNamespace(tmp=tmp, config=config, common=["--config", str(config), "--run-name", "tiny"], data=data)


def test_cli_cv_writes_reports_and_the_e1_criterion(
    tiny_cli: SimpleNamespace, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert baseline_lgbm.main(tiny_cli.common + ["cv", "--fast"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("SYNTHETIC data (drowsy.synth): pipeline check only")
    # 둘째 줄은 추출 결과의 회전 점검 기록이다. 합성 세션에는 그 기록이 없으므로 열 파일 모두 "기록 없음"이다.
    assert lines[1] == SYNTHETIC_ROTATION_LINE
    # 원본 프레임률 추출로만 학습한다는 것을 출력과 리포트 맨 앞 문단에 적는다.
    assert lines[2] == baseline_lgbm.NATIVE_ONLY_NOTE and "native-rate extraction only" in lines[2]
    assert [line for line in lines if line.startswith("[alert_baseline] fold")] == [
        f"[alert_baseline] fold {fold}: lgbm(num_leaves=3,min_data_in_leaf=1)" for fold in (1, 2, 3)
    ]
    assert any(line.startswith("E1 criterion") and "synthetic data: pipeline check only" in line for line in lines)

    run_dir = tiny_cli.tmp / "reports" / "tiny" / "lgbm"
    summary = (run_dir / "summary.md").read_text(encoding="utf-8").splitlines()
    assert summary[0].startswith("# SYNTHETIC data")
    assert baseline_lgbm.NATIVE_ONLY_NOTE in summary[2] and "closure W 0.5 (feature spec)" in summary[2]
    assert "not trained on those augmentations" in summary[2] and train.CONSTANTS_NOTE in summary[2]
    assert "frame-drop variants of the native extraction" in baseline_lgbm.NATIVE_ONLY_NOTE
    preds = evaluate.validate_predictions(pd.read_parquet(run_dir / "predictions.parquet"))
    marker = json.loads(pq.read_schema(run_dir / "predictions.parquet").metadata[b"drowsy_predictions"])
    assert marker == {"data_note": evaluate.SYNTHETIC_NOTE, "synthetic": True}
    assert set(preds["run_id"]) == {"train-alert_baseline-fast"} and set(preds["family"]) == {"lgbm"}
    assert preds["video_id"].nunique() == 9 and sorted(preds["fold"].unique()) == [1, 2, 3]
    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert next(iter(info)) == "data_note" and info["synthetic"] is True and info["fast"] is True
    assert info["arms"] == ["alert_baseline"] and info["closure_w"] is None and info["effective_closure_w"] == 0.5
    # 추출 결과의 회전 점검 기록도 실행 기록에 남는다. 합성 데이터에는 확인할 영상이 없고 설명에도 아무 말이 붙지 않는다.
    assert info["rotation_check"] == SYNTHETIC_ROTATION_RECORD and "ROTATION" not in info["description"]
    assert "ROTATION" not in summary[2]
    cfg = load_config(tiny_cli.config)
    table = train.load_cv_data(cfg, (ARM,)).tables[ARM]
    assert info["step_fingerprints"] == {ARM: train.step_table_fingerprint(table.steps)}
    assert info["pipeline_hash"] == table.pipeline_hash
    # PLAN 4.4절: 조건별·클래스별 창 수(학습 보폭 5와 평가 보폭 1). 창을 실제로 만들어 센 값과 같다.
    report = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    for stride in (5, 1):
        windows = make_windows(table.steps, 120, stride)
        names = ("alert", "low_vigilant", "drowsy")
        counted = {name: int((windows.y == index).sum()) for index, name in enumerate(names)}
        assert report["window_counts"][ARM][str(stride)] == counted and sum(counted.values()) > 0
    assert info["baselines"]["window_counts"][0]["arm"] == ARM and info["baselines"]["n_videos"] == 9
    assert set(info["e1_gate"]["alert_baseline"]) == {"inner_macro_f1_mean", "threshold", "passed"}
    assert info["e1_gate"]["alert_baseline"]["threshold"] == 0.40
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    assert [record["fold"] for record in selection["folds"]] == [1, 2, 3]
    assert selection["command"].startswith("python -m drowsy.baseline_lgbm --config")

    # 실제 데이터에서 E1 기준에 못 미치면 종료 코드 4로 멈춘다(리포트는 써 둔다). 합성 데이터에서는 멈추지 않는다.
    strict = dict(tiny_cli.data)
    strict["evaluation"] = dict(strict["evaluation"], lgbm_min_macro_f1=1.01)
    strict_config = tiny_cli.tmp / "strict.yaml"
    strict_config.write_text(yaml.safe_dump(strict), encoding="utf-8")
    arguments = ["--config", str(strict_config), "--run-name", "strict", "cv", "--fast"]
    assert baseline_lgbm.main(arguments) == 0
    assert "NOT met (synthetic data: pipeline check only)" in capsys.readouterr().out
    real_open = baseline_lgbm.open_dataset

    def as_real(cfg: ExperimentConfig, fps_sim: bool = False) -> DatasetSource:
        return dataclasses.replace(real_open(cfg, fps_sim), synthetic=False)

    monkeypatch.setattr(baseline_lgbm, "open_dataset", as_real)
    assert baseline_lgbm.main(arguments) == 4
    captured = capsys.readouterr()
    assert "NOT met" in captured.out and "stop before the neural networks" in captured.err
    assert (tiny_cli.tmp / "reports" / "strict" / "lgbm" / "summary.md").is_file()


def test_cli_cv_with_another_closure_w_goes_to_its_own_folder(
    tiny_cli: SimpleNamespace, capsys: pytest.CaptureFixture[str], feature_spec: FeatureSpec
) -> None:
    assert baseline_lgbm.main(tiny_cli.common + ["cv", "--fast", "--closure-w", "1"]) == 0
    capsys.readouterr()
    run_dir = tiny_cli.tmp / "reports" / "tiny" / "lgbm_w1"
    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert info["closure_w"] == 1.0 and info["pipeline_hash"] != feature_spec.pipeline_hash  # 다른 상수의 스텝 표
    assert info["effective_closure_w"] == 1.0
    preds = pd.read_parquet(run_dir / "predictions.parquet")
    assert set(preds["run_id"]) == {"train-alert_baseline-w1-fast"}
    description = (run_dir / "summary.md").read_text(encoding="utf-8").splitlines()[2]
    assert "closure W 1 (override: a reporting-only comparison, not the D11 reference)" in description


def test_default_run_must_use_the_reference_closure_w(
    tiny_cli: SimpleNamespace, capsys: pytest.CaptureFixture[str], feature_spec: FeatureSpec, tmp_path: Path
) -> None:
    """`lgbm.reference_w`는 기준 실행(closure W를 바꾸지 않은 실행)의 W와 대조된다(PLAN 5.3절)."""
    assert feature_spec.pipeline.closure.w == 0.5 and load_config(tiny_cli.config).lgbm.reference_w == 0.5
    data = dict(tiny_cli.data)
    data["lgbm"] = dict(data["lgbm"], reference_w=1.0)
    config = tmp_path / "other_w.yaml"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    arguments = ["--config", str(config), "--run-name", "other_w", "cv", "--fast"]
    assert baseline_lgbm.main(arguments) == 2
    error = capsys.readouterr().err
    assert "closure.w = 0.5 but lgbm.reference_w is 1" in error and "--closure-w" in error
    assert not (tiny_cli.tmp / "reports" / "other_w").exists()
    # W를 직접 준 실행은 기준이 아니라서 이 대조를 받지 않는다.
    with pytest.raises(ValueError, match="closure.w = 0.5 but lgbm.reference_w is 1"):
        baseline_lgbm.run_lgbm_cv(load_config(config), fast=True, run_name="other_w")

    # W 비교의 값 목록은 기준 W를 담고 있어야 하고, 0~1 안의 서로 다른 값이어야 한다.
    cases = (([0.0, 1.0], "must contain lgbm.reference_w"), ([0.5, 0.5], "distinct"), ([0.5, 1.5], "0..1"))
    for grid, message in cases:
        data["lgbm"] = dict(tiny_cli.data["lgbm"], w_grid=grid)
        config.write_text(yaml.safe_dump(data), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_config(config)


def test_cli_does_not_overwrite_the_report_of_another_lgbm_run(
    tiny_cli: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """같은 폴더에 다른 조건 목록이나 다른 설정의 리포트가 있으면 멈춘다. 기준 LightGBM을 지키는 장치다."""
    common = ["--config", str(tiny_cli.config), "--run-name", "guard"]
    run_dir = tiny_cli.tmp / "reports" / "guard" / "lgbm"
    assert baseline_lgbm.main(common + ["cv", "--fast"]) == 0
    kept = ("run.json", "predictions.parquet", "summary.md", "selection.json")
    before = {name: (run_dir / name).read_bytes() for name in kept}
    capsys.readouterr()

    for arguments, differing in (
        (["cv", "--fast", "--arms", "global"], "arms: ['alert_baseline'] there, ['global'] now"),
        (["cv"], "fast: True there, False now"),
    ):
        assert baseline_lgbm.main(common + arguments) == 2
        error = capsys.readouterr().err
        assert "already holds the report of another run" in error and differing in error and "--overwrite" in error
    assert {name: (run_dir / name).read_bytes() for name in kept} == before

    # 같은 실행의 재실행은 그대로 되고(결정적이라 예측 표의 내용도 같다), 다른 실행은 --overwrite를 줘야 한다.
    assert baseline_lgbm.main(common + ["cv", "--fast"]) == 0
    assert (run_dir / "predictions.parquet").read_bytes() == before["predictions.parquet"]
    assert baseline_lgbm.main(common + ["cv", "--fast", "--arms", "global", "--overwrite"]) == 0
    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert info["arms"] == ["global"] and set(info["step_fingerprints"]) == {"global"}
    capsys.readouterr()


def test_cli_all_w_writes_one_combined_closure_w_table(
    tiny_cli: SimpleNamespace, capsys: pytest.CaptureFixture[str], feature_spec: FeatureSpec
) -> None:
    """E1의 W 비교: `cv --all-w`가 `lgbm.w_grid`의 W마다 기준 조건을 돌려 표 하나로 합친다. 기준 실행은 건드리지 않는다."""
    common = ["--config", str(tiny_cli.config), "--run-name", "grid"]
    reports = tiny_cli.tmp / "reports" / "grid"
    assert baseline_lgbm.main(common + ["cv", "--fast"]) == 0  # 기준 실행(lgbm/, W = 0.5)
    reference_files = {path.name: path.read_bytes() for path in sorted((reports / "lgbm").iterdir())}
    capsys.readouterr()

    assert baseline_lgbm.main(common + ["cv", "--fast", "--all-w"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("SYNTHETIC data") and lines[1] == SYNTHETIC_ROTATION_LINE
    assert lines[2] == baseline_lgbm.NATIVE_ONLY_NOTE
    assert [line.split(" fold ")[0] for line in lines if line.startswith("[W ")] == (
        ["[W 0]"] * 3 + ["[W 0.5]"] * 3 + ["[W 1]"] * 3
    )
    assert any("supplementary, reporting only; the D11 reference is lgbm/" in line for line in lines)
    run_dir = reports / "lgbm_w_grid"
    assert sorted(path.name for path in reports.iterdir()) == ["lgbm", "lgbm_w_grid"]
    # 기준 실행의 폴더는 읽지도 쓰지도 않았다.
    assert {path.name: path.read_bytes() for path in sorted((reports / "lgbm").iterdir())} == reference_files

    # 합친 표: W마다 한 행, 첫 줄은 데이터 종류.
    assert (run_dir / "w_grid.csv").read_text(encoding="utf-8").startswith("# SYNTHETIC data")
    table = pd.read_csv(run_dir / "w_grid.csv", skiprows=1)
    assert table["closure_w"].tolist() == [0.0, 0.5, 1.0]
    assert table["role"].tolist() == ["supplementary", "reference", "supplementary"]
    assert table["variant"].tolist() == ["closure_w0", "base", "closure_w1"]
    # W마다 closure.w만 바꾼 상수의 스텝 표로 돌았다(pipeline_hash를 따로 계산해 대조한다).
    hashes = [spec_pipeline_hash(apply_overrides(feature_spec.pipeline, {"closure.w": w})) for w in (0.0, 0.5, 1.0)]
    assert table["pipeline_hash"].tolist() == hashes and hashes[1] == feature_spec.pipeline_hash
    assert len(set(hashes)) == 3

    # 기준 W의 행은 기준 실행(lgbm/)과 같은 절차를 한 번 더 돈 것이다: 예측 확률과 지표가 그 실행의 것과 같다.
    preds = evaluate.validate_predictions(pd.read_parquet(run_dir / "predictions.parquet"))
    assert set(preds["run_id"]) == {"train-alert_baseline-fast"} and set(preds["family"]) == {"lgbm"}
    assert preds.groupby("variant")["supplementary"].first().to_dict() == {
        "base": False, "closure_w0": True, "closure_w1": True,
    }  # fmt: skip
    keys = ["fold", "subject", "video_id", "part", "k"]
    columns = keys + list(evaluate.PROB_COLUMNS)
    base = preds[preds["variant"] == "base"].sort_values(keys)[columns].reset_index(drop=True)
    reference = pd.read_parquet(reports / "lgbm" / "predictions.parquet").sort_values(keys)[columns]
    pd.testing.assert_frame_equal(base, reference.reset_index(drop=True))
    reference_report = json.loads((reports / "lgbm" / "summary.json").read_text(encoding="utf-8"))
    reference_summary = reference_report["groups"][0]["summary"]
    row = table.iloc[1]
    for metric in ("video_accuracy", "window_macro_f1", "two_class_video_accuracy"):
        for column in ("fold_mean", "fold_std", "pooled", "ci_low", "ci_high"):
            expected = reference_summary[metric][column]
            assert row[f"{metric}_{column}"] == pytest.approx(expected, abs=1e-12, nan_ok=True)
        assert np.isnan(row[f"{metric}_minus_reference"]) and not row[f"{metric}_differs"]
    info = json.loads((reports / "lgbm" / "run.json").read_text(encoding="utf-8"))
    assert row["inner_macro_f1_mean"] == pytest.approx(info["e1_gate"][ARM]["inner_macro_f1_mean"], abs=1e-12)
    # 다른 W의 확률은 기준 W의 것과 다르다(다른 특징에서 나왔다). 차이 열은 fold를 합친 값의 차이다.
    other = preds[preds["variant"] == "closure_w1"].sort_values(keys)[columns].reset_index(drop=True)
    assert len(other) == len(base) and not np.allclose(other["p_drowsy"].to_numpy(), base["p_drowsy"].to_numpy())
    for position in (0, 2):
        for metric in ("video_accuracy", "window_macro_f1", "two_class_video_accuracy"):
            difference = table.loc[position, f"{metric}_pooled"] - table.loc[1, f"{metric}_pooled"]
            assert table.loc[position, f"{metric}_minus_reference"] == pytest.approx(difference, abs=1e-12)
            low = table.loc[position, f"{metric}_minus_reference_ci_low"]
            high = table.loc[position, f"{metric}_minus_reference_ci_high"]
            assert low <= high and table.loc[position, f"{metric}_differs"] == bool(low > 0.0 or high < 0.0)

    # 리포트: 기준 W의 묶음이 주 결과, 나머지는 부가 결과(9절)와 기준 W와의 짝지은 비교.
    report = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert [(g["variant"], g["supplementary"]) for g in report["groups"]] == [("base", False)]
    assert [g["variant"] for g in report["supplementary"]["groups"]] == ["closure_w0", "closure_w1"]
    assert len(report["supplementary"]["paired"]) == 6 and report["paired"] == [] and report["gate"] is None
    assert all(pair["b"].endswith("features=alert_baseline") for pair in report["supplementary"]["paired"])
    assert set(report["window_counts"]) == {f"{ARM} [closure W {w}]" for w in ("0", "0.5", "1")}
    summary = (run_dir / "summary.md").read_text(encoding="utf-8").splitlines()
    assert summary[0].startswith("# SYNTHETIC data") and "supplementary, for reporting only" in summary[2]
    assert "W grid 0, 0.5, 1" in summary[2] and "not this folder" in summary[2]
    text = (run_dir / "w_grid.md").read_text(encoding="utf-8")
    assert text.startswith("# SYNTHETIC data") and "성능 수치가 아니다" in text
    # 머리말: 설명 문단(D11이 나온다)과 데이터 종류 문장 다음에, 교차검증 리포트와 같은 자리(7번째 줄)에 D 번호가 계획서
    # 1.2절의 결정 번호라는 한 줄이 한 번 있다(`test_evaluate.test_reports_say_what_the_plan_decision_numbers_are`와 같은 규칙).
    head = text.splitlines()
    assert head.count(evaluate.PLAN_DECISION_NOTE) == 1 and head[6] == evaluate.PLAN_DECISION_NOTE
    assert head[5] == head[7] == "" and head[4].startswith("합성 데이터로 만든 표다.") and head[8].startswith("- 생성 명령: ")
    body = "\n".join(line for line in head if line != evaluate.PLAN_DECISION_NOTE)
    assert set(re.findall(r"\bD\d+\b", body)) == {"D11"} and "D11" in head[2]
    rows = [line for line in text.splitlines() if line.startswith("| ") and line[2] in "01"]  # 머리글과 구분선을 뺀다
    assert [line.split(" | ")[0:2] for line in rows] == [["| 0", "부가"], ["| 0.5", "기준"], ["| 1", "부가"]]
    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert next(iter(info)) == "data_note" and info["w_grid"] == [0.0, 0.5, 1.0] and info["reference_w"] == 0.5
    assert info["rotation_check"] == SYNTHETIC_ROTATION_RECORD
    assert info["pipeline_hashes"] == dict(zip(("0", "0.5", "1"), hashes)) and len(info["table"]) == 3
    assert set(info["step_fingerprints"]) == set(report["window_counts"])
    assert len(set(info["step_fingerprints"].values())) == 3

    # 다른 실행의 리포트를 덮어쓰지 않는다. --arms, --closure-w와 함께 쓸 수 없다.
    assert baseline_lgbm.main(common + ["cv", "--all-w"]) == 2
    assert "already holds the report of another run" in capsys.readouterr().err
    for extra in (["--arms", "global"], ["--closure-w", "1"]):
        assert baseline_lgbm.main(common + ["cv", "--fast", "--all-w", *extra]) == 2
        assert "do not pass --arms, --closure-w, --no-velocity or --foldwise-global" in capsys.readouterr().err


def test_cli_passes_rebuild_to_the_data_loader(
    tiny_cli: SimpleNamespace, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[bool] = []

    def fake_load(cfg: ExperimentConfig, arms: Any, **kwargs: Any) -> Any:
        seen.append(kwargs["rebuild"])
        raise ValueError("stop here")

    monkeypatch.setattr(train, "load_cv_data", fake_load)
    for arguments in (["cv", "--fast"], ["cv", "--fast", "--all-w"], ["probes"]):
        assert baseline_lgbm.main(tiny_cli.common + arguments) == 2
        assert baseline_lgbm.main(tiny_cli.common + ["--rebuild", *arguments]) == 2
        assert baseline_lgbm.main(tiny_cli.common + [*arguments, "--rebuild"]) == 2  # 하위 명령 뒤에 적어도 된다
    assert seen == [False, True, True] * 3
    assert "stop here" in capsys.readouterr().err


def test_cli_probes_and_failures(tiny_cli: SimpleNamespace, capsys: pytest.CaptureFixture[str]) -> None:
    assert baseline_lgbm.main(tiny_cli.common + ["probes", "--workers", "1"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("SYNTHETIC data (drowsy.synth): pipeline check only")
    assert lines[1] == SYNTHETIC_ROTATION_LINE
    assert lines[2].startswith("recording_metadata: columns ['mean_fps', 'width', 'height', 'face_size', 'face_ratio']")
    assert "missing ['brightness', 'variable_fps']" in lines[2] and lines[3].startswith("baseline_offset: columns")
    run_dir = tiny_cli.tmp / "reports" / "tiny" / "probes"
    summary = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert summary.splitlines()[0] == "# SYNTHETIC data (drowsy.synth): pipeline check only, not a performance result"
    # 합성 parquet에는 밝기 메타데이터가 없다. 점검은 실패하지 않고 그 사실을 리포트에 적는다.
    assert "추출 결과 parquet의 메타데이터(`brightness_mean`, `variable_intervals`)에서 읽는다" in summary
    assert "이 데이터의 추출 결과에는 그 메타데이터가 없다" in summary and "밝기와 가변 프레임률 표시 없이 돌았고" in summary
    assert "잴 수 없다" not in summary
    record = json.loads((run_dir / "probes.json").read_text(encoding="utf-8"))
    assert next(iter(record)) == "data_note" and record["arm"] == "alert_baseline"
    assert record["rotation_check"] == SYNTHETIC_ROTATION_RECORD and "ROTATION" not in summary
    assert record["metadata_columns"] == {
        "brightness": {"videos_with_value": 0, "videos": 9}, "variable_fps": {"videos_with_value": 0, "videos": 9},
    }  # fmt: skip
    assert record["metadata_note"] in summary
    assert record["probes"][0]["missing_columns"] == ["brightness", "variable_fps"]
    assert [probe["name"] for probe in record["probes"]] == ["recording_metadata", "baseline_offset"]
    assert all(probe["n_videos"] == 9 and probe["n_subjects"] == 3 for probe in record["probes"])
    table_text = (run_dir / "probe_table.csv").read_text(encoding="utf-8")
    assert table_text.startswith("# SYNTHETIC data")
    table = pd.read_csv(run_dir / "probe_table.csv", skiprows=1)
    assert len(table) == 9 and table["open_closure_median"].notna().all()
    # 신경망 리포트는 이 기록을 읽어, 우연 수준을 넘은 점검이 있으면 맨 앞에 적는다.
    flagged = [probe["name"] for probe in record["probes"] if probe["above_chance"]]
    note = train.confound_note(tiny_cli.tmp / "reports" / "tiny")
    assert (note != "") == bool(flagged) and all(name in note for name in flagged)
    assert ("교란 점검이 우연 수준을 넘었다" in summary) == bool(flagged)

    # 데이터가 없으면 종료 코드 2.
    missing = ["--config", str(tiny_cli.config), "--data-root", str(tiny_cli.tmp / "none"), "probes"]
    assert baseline_lgbm.main(missing) == 2
    assert "python -m drowsy.synth" in capsys.readouterr().err


BAD_RUN_NAMES = ("../export", "a/b", "/tmp/x", "..", ".", "", "a\\b")


@pytest.mark.parametrize(
    "arguments",
    [["cv", "--fast"], ["cv", "--fast", "--all-w"], ["cv", "--fast", "--no-velocity"], ["probes"]],
    ids=["cv", "cv_all_w", "cv_no_velocity", "probes"],
)
def test_cli_rejects_a_run_name_that_is_not_a_plain_folder_name(
    arguments: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--run-name`은 폴더 이름 하나여야 한다(리포트 폴더 밖에 쓰지 않게). 다른 일을 하기 전에 한 줄 오류, 종료 코드 2로 끝난다."""
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "missing.yaml"  # 설정 파일을 읽으려 했다면 이 이름이 오류에 나온다
    for name in BAD_RUN_NAMES:
        assert baseline_lgbm.main(["--config", str(config), "--run-name", name, *arguments]) == 2, name
        captured = capsys.readouterr()
        assert captured.err == f"error: --run-name must be a plain folder name, got {name!r}\n"
        assert captured.out == ""
    # 공통 옵션을 하위 명령 뒤에 적어도 같다.
    assert baseline_lgbm.main([*arguments, "--run-name", "x/y", "--config", str(config)]) == 2
    assert capsys.readouterr().err == "error: --run-name must be a plain folder name, got 'x/y'\n"
    # 폴더 이름 하나면 이 검사를 지나 다음 단계(없는 설정 파일)에서 멈춘다.
    assert baseline_lgbm.main(["--config", str(config), "--run-name", "plain-name_1", *arguments]) == 2
    error = capsys.readouterr().err
    assert "missing.yaml" in error and "--run-name" not in error
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# 부가 실행: 속도 특징 제외, fold별 전역 상수
# ---------------------------------------------------------------------------


def test_stat_columns_without_drops_six_statistics_per_feature() -> None:
    assert N_FEATURES == 17 and baseline_lgbm.STAT_NAMES == ("mean", "std", "min", "max", "last", "slope")
    keep = baseline_lgbm.stat_columns_without((7, 8))
    # 통계 블록마다 열 7, 8이 빠진다: 102개 중 12개.
    dropped = sorted(set(range(102)) - set(keep.tolist()))
    assert dropped == [7, 8, 24, 25, 41, 42, 58, 59, 75, 76, 92, 93] and len(keep) == 90
    assert keep.tolist() == sorted(keep.tolist())
    names = baseline_lgbm.window_stat_names()
    velocity = ("closing_vel_mean_60", "opening_vel_mean_60")
    assert sorted(names[index] for index in dropped) == sorted(
        f"{feature}_{stat}" for feature in velocity for stat in baseline_lgbm.STAT_NAMES
    )
    assert baseline_lgbm.stat_columns_without(()).tolist() == list(range(102))
    assert baseline_lgbm.stat_columns_without((0,), n_features=2).tolist() == [1, 3, 5, 7, 9, 11]
    with pytest.raises(ValueError, match="no feature with index"):
        baseline_lgbm.stat_columns_without((17,))


def test_lgbm_cv_without_velocity_never_reads_the_two_features(
    cfg: ExperimentConfig, feature_spec: FeatureSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = make_data(feature_spec, FOUR_FOLDS)
    shapes: list[tuple[int, tuple[str, ...]]] = []
    real_fit = baseline_lgbm.fit_lgbm

    def spy_fit(x: Any, *args: Any, **kwargs: Any) -> Any:
        shapes.append((x.shape[1], tuple(kwargs["feature_names"])))
        return real_fit(x, *args, **kwargs)

    monkeypatch.setattr(baseline_lgbm, "fit_lgbm", spy_fit)
    arguments = (cfg, ARM, "train-alert_baseline-no_velocity", "no_velocity")
    preds, records, _ = baseline_lgbm.lgbm_cross_validate(data, *arguments, drop_features=train.VELOCITY_FEATURES)
    assert {width for width, _ in shapes} == {90}
    assert all(not any("_vel_" in name for name in names) and len(names) == 90 for _, names in shapes)
    assert preds["supplementary"].all() and set(preds["variant"]) == {"no_velocity"}
    assert all(record["refit"]["n_columns"] == 90 for record in records)

    # 두 특징의 값을 통째로 바꿔도 모델과 예측이 같다(입력에 없다). 다른 특징을 바꾸면 달라진다.
    def changed(columns: list[str]) -> train.CVData:
        steps = data.tables[ARM].steps.copy()
        rng = np.random.default_rng(5)
        for name in columns:
            steps[name] = rng.random(len(steps))
        return dataclasses.replace(data, tables={ARM: dataclasses.replace(data.tables[ARM], steps=steps)})

    velocity = [FEATURE_NAMES[index] for index in train.VELOCITY_FEATURES]
    assert velocity == ["closing_vel_mean_60", "opening_vel_mean_60"]
    scrambled, scrambled_records, _ = baseline_lgbm.lgbm_cross_validate(
        changed(velocity), *arguments, drop_features=train.VELOCITY_FEATURES
    )
    probs = list(evaluate.PROB_COLUMNS)
    np.testing.assert_array_equal(scrambled[probs].to_numpy(), preds[probs].to_numpy())
    assert [r["refit"]["model_sha256"] for r in scrambled_records] == [r["refit"]["model_sha256"] for r in records]
    other, other_records, _ = baseline_lgbm.lgbm_cross_validate(
        changed(["closure_mean_60"]), *arguments, drop_features=train.VELOCITY_FEATURES
    )
    assert not np.array_equal(other[probs].to_numpy(), preds[probs].to_numpy())
    assert [r["refit"]["model_sha256"] for r in other_records] != [r["refit"]["model_sha256"] for r in records]
    # 주 실행(102개 열)은 속도 특징을 읽는다.
    shapes.clear()
    base, base_records, _ = baseline_lgbm.lgbm_cross_validate(data, cfg, ARM, "train-alert_baseline")
    base_scrambled, _, _ = baseline_lgbm.lgbm_cross_validate(changed(velocity), cfg, ARM, "train-alert_baseline")
    assert {width for width, _ in shapes} == {102} and all(r["refit"]["n_columns"] == 102 for r in base_records)
    assert not np.array_equal(base_scrambled[probs].to_numpy(), base[probs].to_numpy())
    # 기록의 모델 해시는 다시 학습한 모델 문자열의 sha256이다.
    assert all(len(record["refit"]["model_sha256"]) == 64 for record in base_records)
    with pytest.raises(ValueError, match="feature names were given"):
        real_fit(np.zeros((4, 3)), np.zeros(4), np.ones(4), {}, 1, feature_names=["a", "b"])


def _rounded(value: float) -> float:
    return math.floor(value * 10_000 + 0.5) / 10_000


def test_foldwise_global_constants_come_from_the_training_subjects_only(
    tiny_cli: SimpleNamespace, pipeline_spec: PipelineSpec
) -> None:
    cfg = load_config(tiny_cli.config)
    data = train.load_cv_data(cfg, ("global", "session_baseline"))
    quality = data.tables["global"].quality
    # 세션 자신의 기준선 열은 정규화 조건과 무관하다(자기 세션 기준선 조건의 품질 표와 같은 값이다).
    base_columns = ["session_id", "subject", "baseline_status"]
    base_columns += ["base_e_l", "base_e_r", "base_b_l", "base_b_r", "base_jaw0"]
    pd.testing.assert_frame_equal(quality[base_columns], data.tables["session_baseline"].quality[base_columns])
    assert set(quality["baseline_status"]) == {"ok"} and len(quality) == 10

    tables, constants = baseline_lgbm.foldwise_global_tables(data, cfg)
    assert sorted(tables) == sorted(constants) == [1, 2, 3]
    held_out = {1: 1, 2: 13, 3: 25}
    seen: set[tuple[float, float, float]] = set()
    for fold, subject in held_out.items():
        record = constants[fold]
        assert record["train_subjects"] == sorted(set(TINY_SUBJECTS) - {subject})
        rows = quality[quality["subject"] != subject]  # 시험 피험자의 세션을 뺀 것(독립 계산)
        assert record["n_sessions"] == len(rows)
        assert record["e"] == _rounded(float(np.median((rows["base_e_l"] + rows["base_e_r"]) / 2.0)))
        assert record["b"] == _rounded(float(np.median((rows["base_b_l"] + rows["base_b_r"]) / 2.0)))
        assert record["jaw0"] == _rounded(float(np.median(rows["base_jaw0"])))
        seen.add((record["e"], record["b"], record["jaw0"]))
        # 그 fold의 표는 그 상수를 전역 기준선으로 넣어 모든 세션을 다시 재생한 것이다(직접 만든 표와 같다).
        table = tables[fold]
        assert table.arm == "global" and table.pipeline_hash == data.tables["global"].pipeline_hash
        assert set(table.quality["baseline_used"]) == {"global"}
        assert sorted(table.steps["subject"].unique()) == list(TINY_SUBJECTS)
    assert len(seen) == 3  # 피험자가 다르면 중앙값이 다르다(시험 피험자를 빼는 것이 값에 닿는다)
    spec_constants = pipeline_spec.global_constants
    assert (spec_constants.e, spec_constants.b, spec_constants.jaw0) not in seen
    override = Baseline(
        e_l=constants[3]["e"], e_r=constants[3]["e"], b_l=constants[3]["b"], b_r=constants[3]["b"],
        r0=spec_constants.r0, jaw0=constants[3]["jaw0"], gaze_h0=spec_constants.gaze_h0,
        gaze_v0=spec_constants.gaze_v0, t_complete_ms=0, status=BASELINE_FALLBACK_GLOBAL,
    )  # fmt: skip
    direct = build_step_table(data.manifest, data.raw_dir, pipeline_spec, "global", global_override=override)
    pd.testing.assert_frame_equal(tables[3].steps, direct.steps)
    # 상수가 달라 특징 값이 feature_spec 상수의 표와 다르고, 스텝(따라서 창)은 같다.
    main_steps = data.tables["global"].steps
    assert tables[3].steps[["session_id", "k"]].equals(main_steps[["session_id", "k"]])
    assert train.step_table_fingerprint(tables[3].steps) != train.step_table_fingerprint(main_steps)

    # 전역 조건의 표가 없거나 다른 상수(closure W)로 만든 것이면 거부한다.
    with pytest.raises(ValueError, match="need the step table of arm 'global'"):
        baseline_lgbm.foldwise_global_tables(dataclasses.replace(data, tables={}), cfg)
    other = dataclasses.replace(data.tables["global"], pipeline_hash="0" * 64)
    with pytest.raises(ValueError, match="not built with the pipeline constants of the feature spec"):
        baseline_lgbm.foldwise_global_tables(dataclasses.replace(data, tables={"global": other}), cfg)


def test_cli_foldwise_global_reports_the_sensitivity_row_next_to_the_spec_constants(
    tiny_cli: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    common = ["--config", str(tiny_cli.config), "--run-name", "sens"]
    assert baseline_lgbm.main(common + ["cv", "--fast"]) == 0  # 기준 실행(lgbm/)을 먼저 둔다
    capsys.readouterr()
    reference_dir = tiny_cli.tmp / "reports" / "sens" / "lgbm"
    before = {path.name: path.read_bytes() for path in sorted(reference_dir.iterdir()) if path.is_file()}

    assert baseline_lgbm.main(common + ["cv", "--fast", "--arms", "global", "--foldwise-global"]) == 0
    out = capsys.readouterr().out
    assert "[global] fold 3:" in out and "[global, fold_global] fold 3:" in out
    assert "supplementary run: lgbm/" in out and "E1 criterion (inner-validation" not in out
    run_dir = tiny_cli.tmp / "reports" / "sens" / "lgbm_fold_global"
    assert baseline_lgbm.FOLD_GLOBAL_SUBDIR == "lgbm_fold_global" and f"reports: {run_dir / 'summary.md'}" in out
    assert {path.name: path.read_bytes() for path in sorted(reference_dir.iterdir()) if path.is_file()} == before

    preds = evaluate.validate_predictions(pd.read_parquet(run_dir / "predictions.parquet"))
    groups = preds[["run_id", "family", "arm", "feature_source", "variant", "supplementary"]].drop_duplicates()
    assert sorted(groups.itertuples(index=False, name=None)) == [
        ("train-global-fast", "lgbm", "global", "fold_global", "base", True),
        ("train-global-fast", "lgbm", "global", "native", "base", False),
    ]
    main_rows = preds[~preds["supplementary"]].sort_values(["subject", "video_id", "part", "k"])
    fold_rows = preds[preds["supplementary"]].sort_values(["subject", "video_id", "part", "k"])
    assert len(main_rows) == len(fold_rows) > 0  # 같은 창을 평가한다(상수는 창의 집합을 바꾸지 않는다)
    assert not np.allclose(main_rows["perclos_60"].to_numpy(), fold_rows["perclos_60"].to_numpy())

    report = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert [(g["feature_source"], g["supplementary"]) for g in report["groups"]] == [("native", False)]
    assert [(g["feature_source"], g["supplementary"]) for g in report["supplementary"]["groups"]] == [
        ("fold_global", True)
    ]
    assert len(report["supplementary"]["paired"]) == len(evaluate.PAIRED_METRICS) and report["gate"] is None
    summary = (run_dir / "summary.md").read_text(encoding="utf-8").splitlines()
    assert summary[0].startswith("# SYNTHETIC data") and "SENSITIVITY CHECK of PLAN 4.3 (supplementary)" in summary[2]
    assert "training subjects only" in summary[2] and "not the D11 reference" in summary[2]

    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert info["arms"] == ["global"] and info["foldwise_global"] is True and info["variant"] == "base"
    cfg = load_config(tiny_cli.config)
    data = train.load_cv_data(cfg, ("global",))
    _, constants = baseline_lgbm.foldwise_global_tables(data, cfg)
    assert info["fold_global_constants"] == {str(fold): record for fold, record in constants.items()}
    assert set(info["step_fingerprints"]) == {"global", *(f"global [fold {fold} constants]" for fold in (1, 2, 3))}
    assert len(set(info["step_fingerprints"].values())) == 4
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))["folds"]
    assert [(r["fold"], r.get("feature_source")) for r in selection] == [
        (1, None), (2, None), (3, None), (1, "fold_global"), (2, "fold_global"), (3, "fold_global"),
    ]  # fmt: skip
    for record in selection[3:]:
        assert record["global_constants"] == constants[record["fold"]]
        assert set(record["global_constants"]["train_subjects"]) == set(record["train_subjects"])
        assert not set(record["global_constants"]["train_subjects"]) & set(record["test_subjects"])

    # 전역 조건의 점검이다: 다른 조건, closure W 변경, 속도 특징 제외와 함께 쓸 수 없다.
    for arguments, message in (
        (["cv", "--fast", "--foldwise-global"], "pass --arms global"),
        (["cv", "--fast", "--arms", "global", "alert_baseline", "--foldwise-global"], "pass --arms global"),
        (["cv", "--fast", "--arms", "global", "--foldwise-global", "--closure-w", "1"], "cannot be combined"),
        (["cv", "--fast", "--arms", "global", "--foldwise-global", "--no-velocity"], "separate supplementary runs"),
        (["cv", "--fast", "--no-velocity", "--closure-w", "1"], "cannot be combined"),
    ):
        assert baseline_lgbm.main(common + arguments) == 2, arguments
        assert message in capsys.readouterr().err


def test_cli_no_velocity_run_goes_to_its_own_folder_and_pairs_with_the_main_run(
    tiny_cli: SimpleNamespace, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    common = ["--config", str(tiny_cli.config), "--run-name", "novel"]
    reports = tiny_cli.tmp / "reports" / "novel"
    # (1) 주 실행이 없으면 짝지은 비교 없이 돈다.
    assert baseline_lgbm.main(common + ["cv", "--fast", "--no-velocity"]) == 0
    out = capsys.readouterr().out
    assert "supplementary run: lgbm/" in out and "E1 criterion (inner-validation" not in out
    run_dir = reports / "lgbm_no_velocity"
    assert baseline_lgbm.NO_VELOCITY_SUBDIR == "lgbm_no_velocity" and sorted(p.name for p in reports.iterdir()) == [
        "lgbm_no_velocity"
    ]
    report = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert report["groups"] == [] and report["supplementary"]["paired"] == []
    assert json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["paired_with_main_run"] is False

    # (2) 주 실행(lgbm/)을 돌린 뒤에는 그 묶음을 함께 싣고 짝지은 비교를 낸다. lgbm/은 바뀌지 않는다.
    assert baseline_lgbm.main(common + ["cv", "--fast"]) == 0
    before = {path.name: path.read_bytes() for path in sorted((reports / "lgbm").iterdir()) if path.is_file()}
    assert baseline_lgbm.main(common + ["cv", "--fast", "--no-velocity"]) == 0
    capsys.readouterr()
    assert {path.name: path.read_bytes() for path in sorted((reports / "lgbm").iterdir()) if path.is_file()} == before
    preds = evaluate.validate_predictions(pd.read_parquet(run_dir / "predictions.parquet"))
    assert set(preds["run_id"]) == {"train-alert_baseline-no_velocity-fast"}
    assert preds["supplementary"].all() and set(preds["variant"]) == {"no_velocity"}
    report = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert [(g["run_id"], g["variant"]) for g in report["groups"]] == [("train-alert_baseline-fast", "base")]
    assert [(g["run_id"], g["variant"]) for g in report["supplementary"]["groups"]] == [
        ("train-alert_baseline-no_velocity-fast", "no_velocity")
    ]
    assert len(report["supplementary"]["paired"]) == len(evaluate.PAIRED_METRICS)
    assert all("train-alert_baseline-fast /" in row["b"] for row in report["supplementary"]["paired"])
    summary = (run_dir / "summary.md").read_text(encoding="utf-8").splitlines()
    assert "SUPPLEMENTARY RUN (PLAN 11 E3): without the velocity features #7 closing_vel_mean_60" in summary[2]
    assert "12 window statistics are dropped from the 102 inputs" in summary[2]
    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert info["variant"] == "no_velocity" and info["paired_with_main_run"] is True and info["closure_w"] is None
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))["folds"]
    assert all(record["refit"]["n_columns"] == 90 for record in selection)

    # (3) lgbm/이 다른 스텝 특징에서 나온 것이면 학습 전에 멈춘다.
    record = json.loads((reports / "lgbm" / "run.json").read_text(encoding="utf-8"))
    record["step_fingerprints"][ARM] = hashlib.sha256(b"other").hexdigest()
    (reports / "lgbm" / "run.json").write_text(json.dumps(record), encoding="utf-8")
    calls: list[Any] = []
    monkeypatch.setattr(baseline_lgbm, "fit_lgbm", lambda *args, **kwargs: calls.append(args))
    assert baseline_lgbm.main(common + ["cv", "--fast", "--no-velocity"]) == 2
    assert "cannot serve as the main run of this supplementary run" in capsys.readouterr().err
    assert calls == []
    shutil.rmtree(reports)


def test_lgbm_fold_models_are_also_evaluated_on_another_extraction_without_touching_selection(
    cfg: ExperimentConfig, feature_spec: FeatureSpec
) -> None:
    """다시 학습한 fold 모델을 시험 fold의 다른 추출(15 fps 모사) 특징으로도 평가한다. 학습과 선택은 그대로다."""
    data = make_data(feature_spec, FOUR_FOLDS)
    other_steps = make_steps(FOUR_FOLDS, n_steps=130, seed=9)  # 다른 추출: 값이 다르고 창 수도 다르다(세션마다 11개)
    plain, plain_records, _ = baseline_lgbm.lgbm_cross_validate(data, cfg, ARM, "train-alert_baseline")
    notes: dict[str, Any] = {}
    both, records, _ = baseline_lgbm.lgbm_cross_validate(
        data, cfg, ARM, "train-alert_baseline", extra_eval={"fps15": other_steps}, notes=notes
    )
    assert notes == {"extra_evaluation": {"evaluated": ["fps15"], "skipped": {}}}
    assert records == plain_records  # 선택 기록과 다시 학습한 모델(해시)은 평가할 추출과 무관하다
    native = both[both["feature_source"] == "native"].reset_index(drop=True)
    pd.testing.assert_frame_equal(native, plain)
    extra = both[both["feature_source"] == "fps15"]
    assert extra["supplementary"].all() and not native["supplementary"].any()
    assert len(extra) == len(FOUR_FOLDS) * 3 * 11 and len(native) == len(FOUR_FOLDS) * 3 * 15
    assert (extra["fold"] == extra["subject"].map(official_fold)).all()
    # 예측은 그 fold의 모델을 그 추출의 창에 돌린 것이다: perclos_60이 그 추출의 스텝 표에서 온다.
    row = extra.iloc[0]
    step = other_steps[(other_steps["session_id"] == row["video_id"]) & (other_steps["k"] == row["k"])]
    assert len(step) == 1 and step["perclos_60"].iloc[0] == row["perclos_60"]
    assert evaluate.supplementary_pairs(both) == [
        (("train-alert_baseline", "lgbm", ARM, "fps15", "base"), ("train-alert_baseline", "lgbm", ARM))
    ]
    # 확률은 그 추출의 시험 창을 그 fold의 모델에 넣은 값이다: 같은 표를 "다른 추출"로 주면 원본 묶음과 확률이 같고,
    # 다른 표를 주면 다르다.
    same, _, _ = baseline_lgbm.lgbm_cross_validate(
        data, cfg, ARM, "train-alert_baseline", extra_eval={"fps15": data.tables[ARM].steps}
    )
    probs = list(evaluate.PROB_COLUMNS)
    keys = ["subject", "video_id", "k"]
    twin = same[same["feature_source"] == "fps15"].sort_values(keys)
    np.testing.assert_array_equal(twin[probs].to_numpy(), native.sort_values(keys)[probs].to_numpy())
    merged = extra.merge(native, on=keys, suffixes=("_extra", ""))
    assert len(merged) == len(extra) and not np.allclose(merged["p_alert_extra"], merged["p_alert"])

    # 그 추출이 시험 fold 하나를 통째로 빠뜨리면 평가를 건너뛰고 이유를 적는다(평가하지 않은 fold를 오답으로 세지 않는다).
    partial = other_steps[other_steps["fold"] != 3].reset_index(drop=True)
    notes = {}
    skipped, _, _ = baseline_lgbm.lgbm_cross_validate(
        data, cfg, ARM, "train-alert_baseline", extra_eval={"fps15": partial}, notes=notes
    )
    assert set(skipped["feature_source"]) == {"native"}
    assert notes["extra_evaluation"] == {
        "evaluated": [], "skipped": {"fps15": "the extraction has no windows for test fold(s) [3]"},
    }  # fmt: skip
    with pytest.raises(ValueError, match="cannot be combined with fold_tables"):
        baseline_lgbm.lgbm_cross_validate(
            data, cfg, ARM, "r", fold_tables={1: data.tables[ARM]}, extra_eval={"fps15": other_steps}
        )
