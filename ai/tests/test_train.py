"""drowsy.train 테스트 (PLAN 5.2절, 6장, 11장 E2~E4). 합성 표와 합성 세션만 쓴다. 학습은 모두 CPU에서 한다.

손으로 만든 스텝 표(`make_steps`)는 학습 절차의 부기(어느 피험자가 어느 호출에 들어가는지)를 보는 데 쓴다. 특징 값은
난수이고 파이프라인의 출력이 아니다. 실제 파이프라인이 만든 특징으로 도는 전 과정은 `tiny_run`(합성 세션 →
`DriverPipeline` → 스텝 표 → 명령)이 본다.

D11과 D16은 계획서(docs/PLAN.md 1.2절 표)의 결정 번호다. D11: 교차검증 결과가 정해 둔 조건을 모두 채울 때만 모델을 기본
졸음 판정원으로 둔다. D16: TCN이 최종 모델이면 INT8 파일도 만든다.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import json
import math
import os
import re
import shutil
import struct
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
import pyarrow.parquet as pq
import pytest
import torch
import yaml

from drowsy import baseline_lgbm, evaluate, export_onnx, extract, synth, train
from drowsy.config import ExperimentConfig, apply_fast_profile, load_config
from drowsy.dataset import (
    ID_COLUMNS,
    LABEL_TO_CLASS,
    ROTATION_AMBIGUOUS,
    ROTATION_OVERRIDE_NOT_APPLIED,
    STEP_COLUMNS,
    FrameDropSpec,
    InnerSplit,
    RotationFlag,
    RotationSummary,
    StepTable,
    SubjectOverlapError,
    WindowSet,
    apply_quality_gate,
    make_windows,
    nested_splits,
    official_fold,
)
from drowsy.export_onnx import (
    ALLOWED_OPS_INT8,
    ContractError,
    exported_spec_path,
    parity_max_abs_diff,
    random_raw_inputs,
    read_metadata,
    verify_contract,
)
from drowsy.models import KIND_GRU, KIND_TCN, KINDS, ExportWrapper, build_model, count_parameters, load_checkpoint
from drowsy.spec import FeatureSpec, PipelineSpec, file_sha256, load_feature_spec, load_rules
from drowsy.types import CLASS_ORDER, FEATURE_NAMES, N_FEATURES

LOG1P_COLUMNS = [9, 10, 11, 12, 14]
FOUR_FOLDS = (1, 2, 13, 14, 25, 26, 37, 38)  # 공식 fold 1~4에 두 명씩
THREE_FOLDS = (1, 2, 13, 14, 25, 26)
ARM = "alert_baseline"
APP_ARM = "session_baseline"


# ---------------------------------------------------------------------------
# 도우미: 손으로 만든 스텝 표
# ---------------------------------------------------------------------------


def make_steps(subjects: tuple[int, ...], n_steps: int = 123, seed: int = 0, offset: float = 0.0) -> pd.DataFrame:
    """피험자마다 세 라벨의 세션 하나씩. 모든 스텝이 full_lookback이고 세션마다 창이 `n_steps − 119`개 나온다.

    특징 값은 클래스마다 수준이 다른 난수다(0~1 안이라 perclos_60의 범위 검사를 지난다).
    """
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
            values = 0.1 + 0.25 * class_index + offset + 0.03 * rng.standard_normal((n_steps, N_FEATURES))
            values = np.clip(values, 0.0, 1.0)
            for index, name in enumerate(FEATURE_NAMES):
                data[name] = values[:, index]
            frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True)[list(STEP_COLUMNS)]


def make_quality(steps: pd.DataFrame) -> pd.DataFrame:
    rows = steps.drop_duplicates("session_id")[list(ID_COLUMNS)].reset_index(drop=True)
    rows["raw_status"] = "ok"
    rows["baseline_status"] = "ok"
    rows["face_ms"] = 95_000
    rows["duration_ms"] = 100_000
    rows["n_windows"] = steps.groupby("session_id", sort=False).size().to_numpy() - 119
    return apply_quality_gate(rows)


def make_data(
    feature_spec: FeatureSpec,
    subjects: tuple[int, ...],
    with_app: bool = True,
    with_extra: bool = False,
    with_frame_drop: bool = False,
) -> train.CVData:
    steps = make_steps(subjects, seed=0)
    quality = make_quality(steps)
    tables = {ARM: StepTable(ARM, feature_spec.pipeline_hash, steps, quality)}
    if with_app:
        app_steps = make_steps(subjects, seed=1, offset=0.05)
        tables[APP_ARM] = StepTable(APP_ARM, feature_spec.pipeline_hash, app_steps, make_quality(app_steps))
    extra_tables = {}
    if with_extra:
        extra_steps = make_steps(subjects, seed=2, offset=0.02)
        extra_tables[ARM] = StepTable(ARM, feature_spec.pipeline_hash, extra_steps, make_quality(extra_steps))
    frame_drop_tables = {}
    if with_frame_drop:
        drop_steps = make_steps(subjects, seed=3, offset=0.04)
        frame_drop_tables[ARM] = (StepTable(ARM, feature_spec.pipeline_hash, drop_steps, make_quality(drop_steps)),)
    return train.CVData(
        feature_spec=feature_spec,
        manifest=quality[list(ID_COLUMNS)].copy(),
        raw_dir=Path("unused"),
        tables=tables,
        extra_tables=extra_tables,
        synthetic=True,
        description="SYNTHETIC data (hand-built table)",
        frame_drop_tables=frame_drop_tables,
    )


def source(name: str, share: float, windows: WindowSet) -> train.TrainSource:
    return train.TrainSource(name, share, windows)


def joined(*sets: WindowSet) -> WindowSet:
    """창 묶음들을 하나로 이어 붙인 것."""
    return WindowSet(
        x=np.concatenate([item.x for item in sets]),
        y=np.concatenate([item.y for item in sets]),
        meta=pd.concat([item.meta for item in sets], ignore_index=True),
    )


def tiny_config(cfg: ExperimentConfig, **training: Any) -> ExperimentConfig:
    """CPU에서 몇 초 안에 도는 학습 설정."""
    values: dict[str, Any] = {
        "device": "cpu", "batch_size": 16, "updates_per_unit": 4, "max_units": 3, "patience_units": 2,
        "seeds": (0, 1),
        "tcn": dataclasses.replace(cfg.training.tcn, channels=(8,), dropout=(0.1,)),
        "gru": dataclasses.replace(cfg.training.gru, hidden=(8,), dropout=(0.1,)),
    }  # fmt: skip
    values.update(training)
    return dataclasses.replace(
        cfg,
        training=dataclasses.replace(cfg.training, **values),
        evaluation=dataclasses.replace(cfg.evaluation, bootstrap_n=30),
    )


def train_config(cfg: ExperimentConfig, spec: FeatureSpec, kind: str = KIND_GRU, **overrides: Any) -> train.TrainConfig:
    hyper = {"hidden": 8, "dropout": 0.1} if kind == KIND_GRU else {"channels": 8, "dropout": 0.1}
    base = train.TrainConfig(kind=kind, hyper=hyper, spec=spec, settings=cfg.training)
    return dataclasses.replace(base, **overrides)


def windows_of(steps: pd.DataFrame, subjects: tuple[int, ...]) -> WindowSet:
    return train.subset_subjects(make_windows(steps, 120, 1), subjects)


@pytest.fixture(scope="module")
def cfg(experiment_config: ExperimentConfig) -> ExperimentConfig:
    return tiny_config(experiment_config)


@pytest.fixture(scope="module")
def steps() -> pd.DataFrame:
    return make_steps(THREE_FOLDS)


def states_equal(a: torch.nn.Module, b: torch.nn.Module) -> bool:
    return all(torch.equal(value, b.state_dict()[key]) for key, value in a.state_dict().items())


# ---------------------------------------------------------------------------
# 표집
# ---------------------------------------------------------------------------


def sampler_meta(
    counts: dict[tuple[int, str], int], gap_after: dict[tuple[int, str], int] | None = None
) -> pd.DataFrame:
    """(피험자, 영상) → 창 수. gap_after에 적은 영상은 그 창 뒤에서 스텝 번호가 건너뛴다(구간이 둘이 된다)."""
    rows = []
    for (subject, video), n in counts.items():
        k_end = np.arange(240, 240 + n)
        if gap_after and (subject, video) in gap_after:
            k_end[gap_after[(subject, video)] :] += 50
        rows.append(
            pd.DataFrame({"subject": subject, "video_id": video, "session_id": video, "k_end": k_end})
        )
    return pd.concat(rows, ignore_index=True)


def test_sampler_draws_subject_then_video_then_window_uniformly() -> None:
    # 영상마다 창 수가 크게 다르다. 창을 그대로 뽑으면 긴 영상이 지배하지만, 표집은 피험자와 영상을 먼저 고른다.
    counts = {
        (1, "S01_0"): 50, (1, "S01_5"): 5, (1, "S01_10"): 20,
        (2, "S02_0"): 10, (2, "S02_5"): 10, (2, "S02_10"): 200,
        (3, "S03_0"): 30, (3, "S03_10"): 3,  # 영상이 둘뿐인 피험자
    }  # fmt: skip
    meta = sampler_meta(counts)
    sampler = train.BalancedSampler([meta], seed=0)
    n = 120_000
    drawn = sampler.draw(n)
    assert drawn.dtype == np.int64 and drawn.min() >= 0 and drawn.max() < len(meta)
    picked = meta.iloc[drawn]

    by_subject = picked.groupby("subject").size() / n
    assert np.allclose(by_subject.to_numpy(), 1.0 / 3.0, atol=0.01)
    by_video = picked.groupby(["subject", "video_id"]).size() / n
    for (subject, video), share in by_video.items():
        n_videos = sum(1 for key in counts if key[0] == subject)
        assert share == pytest.approx(1.0 / (3 * n_videos), abs=0.01), (subject, video)
    # 한 영상 안에서는 창이 균등하다.
    long_video = picked[picked["video_id"] == "S02_10"].groupby("k_end").size()
    assert len(long_video) == 200
    assert long_video.min() > 0.5 * long_video.mean() and long_video.max() < 1.5 * long_video.mean()
    # 창을 그대로 합쳐 뽑았다면 S02_10이 200 / 328을 차지했을 것이다.
    assert by_video[(2, "S02_10")] < 0.15

    # 같은 시드면 같은 수열, 다른 시드면 다른 수열이다.
    assert np.array_equal(train.BalancedSampler([meta], seed=0).draw(n), drawn)
    assert not np.array_equal(train.BalancedSampler([meta], seed=1).draw(n), drawn)


def test_sampler_balances_classes_when_every_subject_has_one_video_per_class() -> None:
    counts = {(s, f"S{s:02d}_{label}"): 5 + 40 * (label == 10) for s in (1, 2, 3, 4) for label in (0, 5, 10)}
    meta = sampler_meta(counts)
    meta["class_index"] = meta["video_id"].str.split("_").str[1].map({"0": 0, "5": 1, "10": 2})
    picked = meta.iloc[train.BalancedSampler([meta], seed=3).draw(60_000)]
    shares = picked.groupby("class_index").size() / 60_000
    assert np.allclose(shares.to_numpy(), 1.0 / 3.0, atol=0.01)  # drowsy 영상이 9배 길어도 클래스가 균형을 이룬다


def test_sampler_is_uniform_over_the_stride_1_windows_of_a_video() -> None:
    # PLAN 5.2절: 창을 균등하게. 23개 창이 이어진 영상에서 창마다 1/23이다. 보폭 5 격자의 칸을 먼저 고르던 구현은
    # 마지막 세 창을 1/15씩(균등의 1.5배), 나머지를 1/25씩 뽑았다.
    meta = sampler_meta({(1, "S01_0"): 23})
    n = 115_000
    hits = np.bincount(train.BalancedSampler([meta], seed=0).draw(n), minlength=23)
    assert hits.sum() == n and (hits > 0).all()
    assert np.allclose(hits / n, 1.0 / 23.0, rtol=0.06)  # 창마다 기대 5,000번, 표준편차 약 69번
    assert hits[20:].sum() / n == pytest.approx(3.0 / 23.0, rel=0.03)  # 예전 구현은 3/15 = 0.2

    # 계획서 4.4절의 600초 영상(창 362개): 끝의 두 창이 균등의 2.5배로 뽑히던 치우침이 없다.
    long_meta = sampler_meta({(1, "S01_0"): 362})
    hits = np.bincount(train.BalancedSampler([long_meta], seed=1).draw(362_000), minlength=362)
    assert hits.min() > 800 and hits.max() < 1200  # 창마다 기대 1,000번, 표준편차 약 32번

    # 스텝 번호가 건너뛰어 구간이 둘([0..6], [7..11])인 영상도 창 12개가 똑같이 1/12이다.
    gapped = sampler_meta({(1, "S01_0"): 12}, gap_after={(1, "S01_0"): 7})
    hits = np.bincount(train.BalancedSampler([gapped], seed=1).draw(60_000), minlength=12)
    assert np.allclose(hits / 60_000, 1.0 / 12.0, rtol=0.06)
    assert hits[5] + hits[6] == pytest.approx(10_000, rel=0.04)  # 예전 구현은 이 두 창에 1/3(20,000번)을 줬다

    # 나뉜 영상(세션 둘)은 논리 영상 하나다: 부분들의 창 16개를 합쳐 균등하게, 피험자의 다른 영상과 같은 비중으로.
    parts = pd.concat(
        [
            pd.DataFrame({"subject": 1, "video_id": "S01_10", "session_id": "S01_10_1", "k_end": np.arange(4)}),
            pd.DataFrame({"subject": 1, "video_id": "S01_0", "session_id": "S01_0", "k_end": np.arange(8)}),
            pd.DataFrame({"subject": 1, "video_id": "S01_10", "session_id": "S01_10_2", "k_end": np.arange(12)}),
        ],
        ignore_index=True,
    )
    drawn = train.BalancedSampler([parts], seed=2).draw(96_000)
    hits = np.bincount(drawn, minlength=24)
    split_rows = np.flatnonzero(parts["video_id"].to_numpy() == "S01_10")
    assert len(split_rows) == 16 and hits[split_rows].sum() == pytest.approx(48_000, rel=0.02)  # 영상마다 1/2
    assert np.allclose(hits[split_rows] / 96_000, 1.0 / 32.0, rtol=0.08)  # 1/2 × 1/16
    assert np.allclose(hits[4:12] / 96_000, 1.0 / 16.0, rtol=0.06)  # 1/2 × 1/8

    # 정렬되지 않은 묶음을 줘도 번호는 받은 묶음의 행 위치다.
    shuffled = meta.sample(frac=1.0, random_state=0).reset_index(drop=True)
    drawn = train.BalancedSampler([shuffled], seed=0).draw(20_000)
    assert set(np.unique(drawn)) == set(range(23))


def test_sampler_mixes_a_second_extraction_by_share() -> None:
    native = sampler_meta({(1, "S01_0"): 10, (2, "S02_0"): 10})
    extra = sampler_meta({(1, "S01_0"): 7, (2, "S02_0"): 7})
    sampler = train.BalancedSampler([native, extra], seed=0, extra_shares=[0.25])
    drawn = sampler.draw(40_000)
    assert drawn.min() >= 0 and drawn.max() < 34  # 두 묶음을 이어 붙인 번호
    assert (drawn >= 20).mean() == pytest.approx(0.25, abs=0.01)
    assert set(np.unique(drawn)) == set(range(34))
    # 묶음이 하나면 비율은 없다. 비율이 0이면 두 번째 묶음에서 뽑지 않는다.
    assert train.BalancedSampler([native], 0).draw(1000).max() < 20
    assert train.BalancedSampler([native, extra], 0, [0.0]).draw(5000).max() < 20

    # 묶음 셋: 나머지 묶음마다 자기 비율을 갖는다(원본 0.5, 둘째 0.3, 셋째 0.2).
    third = sampler_meta({(1, "S01_0"): 5, (2, "S02_0"): 5})
    drawn = train.BalancedSampler([native, extra, third], 0, [0.3, 0.2]).draw(60_000)
    shares = [(drawn < 20).mean(), ((drawn >= 20) & (drawn < 34)).mean(), (drawn >= 34).mean()]
    assert shares == pytest.approx([0.5, 0.3, 0.2], abs=0.01) and drawn.max() < 44

    with pytest.raises(ValueError, match="one extra share per extra window set"):
        train.BalancedSampler([native, extra], 0)
    with pytest.raises(ValueError, match="one extra share per extra window set"):
        train.BalancedSampler([native], 0, [0.5])
    with pytest.raises(ValueError, match="extra shares"):
        train.BalancedSampler([native, extra], 0, [1.0])
    with pytest.raises(ValueError, match="extra shares"):
        train.BalancedSampler([native, extra, third], 0, [0.6, 0.4])
    with pytest.raises(ValueError, match="extra shares"):
        train.BalancedSampler([native, extra], 0, [-0.1])
    with pytest.raises(ValueError, match="at least one window"):
        train.BalancedSampler([native.iloc[:0], native], 0, [0.5])


def test_sampler_keeps_subjects_and_videos_uniform_when_the_second_extraction_is_partial() -> None:
    # 원본: 피험자 4명 × 영상 3개. 두 번째 추출(덜 끝난 폴더): 1번의 영상 3개, 2번의 영상 1개, 그리고 원본에 없는 9번.
    native = sampler_meta({(s, f"S{s:02d}_{label}"): 10 for s in (1, 2, 3, 4) for label in (0, 5, 10)})
    extra = sampler_meta(
        {(1, "S01_0"): 7, (1, "S01_5"): 7, (1, "S01_10"): 7, (2, "S02_5"): 4, (9, "S09_0"): 6}
    )
    n = 240_000
    drawn = train.BalancedSampler([native, extra], seed=0, extra_shares=[0.5]).draw(n)
    from_extra = drawn >= len(native)
    both = pd.concat([native, extra], ignore_index=True)
    picked = both.iloc[drawn]

    # 피험자마다 1/4, 영상마다 1/12. 추출을 먼저 고르던 구현에서는 1번이 0.5 · 1/4 + 0.5 · 1/3 = 0.29였고
    # 두 번째 추출이 표본의 절반을 댔다.
    by_subject = picked.groupby("subject").size() / n
    assert sorted(by_subject.index) == [1, 2, 3, 4]  # 원본에 없는 9번은 뽑히지 않는다
    assert np.allclose(by_subject.to_numpy(), 0.25, atol=0.005)
    by_video = picked.groupby("video_id").size() / n
    assert len(by_video) == 12 and np.allclose(by_video.to_numpy(), 1.0 / 12.0, atol=0.004)
    # 두 번째 추출에서 나온 표본: 그 추출이 있는 영상 4개에서만, 각 영상 표본의 절반 = 전체의 4/12 × 0.5 = 1/6.
    assert from_extra.mean() == pytest.approx(1.0 / 6.0, abs=0.005)
    extra_videos = picked[from_extra].groupby("video_id").size() / n
    assert sorted(extra_videos.index) == ["S01_0", "S01_10", "S01_5", "S02_5"]
    assert np.allclose(extra_videos.to_numpy(), 1.0 / 24.0, atol=0.003)
    # 그 추출이 없는 영상의 표본은 모두 원본에서 나온다(몫이 원본으로 간다): 원본 표본은 전체의 5/6.
    native_videos = picked[~from_extra].groupby("video_id").size() / n
    assert native_videos["S03_0"] == pytest.approx(1.0 / 12.0, abs=0.004)
    assert native_videos["S01_0"] == pytest.approx(1.0 / 24.0, abs=0.003)
    # 영상 안에서는 고른 묶음의 창이 균등하다(S02_5의 두 번째 추출 창 4개).
    rows = np.flatnonzero((both["video_id"].to_numpy() == "S02_5") & (np.arange(len(both)) >= len(native)))
    assert len(rows) == 4 and np.allclose(np.bincount(drawn, minlength=len(both))[rows] / n, 1.0 / 96.0, rtol=0.1)


# ---------------------------------------------------------------------------
# 표준화 상수, 온도, 내부 검증 지표
# ---------------------------------------------------------------------------


def hand_windows(values_by_video: dict[str, np.ndarray], subject: int = 1) -> WindowSet:
    """영상 → [창 수, 스텝 수, 17] 배열."""
    xs, metas = [], []
    for video, values in values_by_video.items():
        xs.append(values.astype(np.float32))
        metas.append(
            pd.DataFrame(
                {
                    "video_id": video, "session_id": video, "subject": subject, "fold": 1, "label": 0, "class_index": 0,
                    "part": 0, "k_end": 240 + np.arange(len(values)), "t_end_ms": 0,
                }
            )  # fmt: skip
        )
    x = np.concatenate(xs)
    meta = pd.concat(metas, ignore_index=True)
    return WindowSet(x=x, y=meta["class_index"].to_numpy(dtype=np.int64), meta=meta)


def test_standardization_is_video_weighted_after_log1p_and_floored(feature_spec: FeatureSpec) -> None:
    assert np.flatnonzero(feature_spec.log1p_mask).tolist() == LOG1P_COLUMNS
    # 영상 A: 창 1개, 모든 값 1. 영상 B: 창 3개, 모든 값 3. 12번 열(하품 수, log1p)은 두 영상 모두 0이라 분산이 0이다.
    a = np.full((1, 2, N_FEATURES), 1.0)
    b = np.full((3, 2, N_FEATURES), 3.0)
    a[..., 12] = 0.0
    b[..., 12] = 0.0
    b[..., 9] = -4.0  # log1p 열의 음수는 0으로 자른 뒤 변환한다
    windows = hand_windows({"S01_0": a, "S01_5": b})
    mean, std = train.standardization_stats(windows, feature_spec)

    # 변환하지 않는 열(0번): 영상 평균 1과 3의 평균 = 2(창을 그대로 합치면 2.5), 표준편차 1.
    assert mean[0] == pytest.approx(2.0) and std[0] == pytest.approx(1.0)
    # log1p 열(10번): 영상 평균 log 2와 log 4.
    assert mean[10] == pytest.approx((math.log(2.0) + math.log(4.0)) / 2.0)
    assert std[10] == pytest.approx((math.log(4.0) - math.log(2.0)) / 2.0)
    # 9번 열: 영상 A는 log 2, 영상 B는 log1p(max(−4, 0)) = 0.
    assert mean[9] == pytest.approx(math.log(2.0) / 2.0)
    # 분산이 0인 열은 하한 0.001로 올린다.
    assert mean[12] == 0.0 and std[12] == feature_spec.standardization.std_floor == 0.001
    assert all(isinstance(value, float) for value in mean + std)

    # 묶음을 둘 주면 각 묶음의 영상을 따로 센다(영상 넷의 평균).
    c = np.full((2, 2, N_FEATURES), 5.0)
    mean2, _ = train.standardization_stats([windows, hand_windows({"S01_0": c, "S01_5": c})], feature_spec)
    assert mean2[0] == pytest.approx((1.0 + 3.0 + 5.0 + 5.0) / 4.0)
    # 빈 묶음은 건너뛴다. 창이 하나도 없으면 오류다.
    assert train.standardization_stats([windows, windows.subset(np.zeros(4, dtype=bool))], feature_spec)[0] == mean
    with pytest.raises(ValueError, match="without training windows"):
        train.standardization_stats(windows.subset(np.zeros(4, dtype=bool)), feature_spec)

    # 출처별 몫(원본 0.5, 15 fps 0.25, 프레임 누락 변형 0.25). 변형에는 영상 A만 있고 B의 그 몫은 원본으로 간다.
    # 영상 A: 0.5·1 + 0.25·5 + 0.25·9 = 4, 영상 B: 0.75·3 + 0.25·5 = 3.5. 평균 3.75.
    # 제곱 평균: A 0.5·1 + 0.25·25 + 0.25·81 = 27, B 0.75·9 + 0.25·25 = 13. 평균 20. 분산 20 − 3.75² = 5.9375.
    fps = hand_windows({"S01_0": c, "S01_5": c})
    drop = hand_windows({"S01_0": np.full((4, 2, N_FEATURES), 9.0), "S01_10": np.full((1, 2, N_FEATURES), 100.0)})
    mean3, std3 = train.standardization_stats([windows, fps, drop], feature_spec, [0.25, 0.25])
    assert mean3[0] == pytest.approx(3.75) and std3[0] == pytest.approx(math.sqrt(5.9375))
    # 원본에 없는 영상(S01_10)은 표집되지 않으므로 상수에도 들어가지 않는다(값 100이 보이지 않는다).
    # 같은 묶음과 몫으로 표집기가 뽑은 표본의 평균과 같다(독립 확인: 표본 12만 개의 평균, 표준오차 약 0.007).
    sampler = train.BalancedSampler([windows.meta, fps.meta, drop.meta], seed=0, extra_shares=[0.25, 0.25])
    drawn = np.concatenate([windows.x, fps.x, drop.x])[sampler.draw(120_000)][:, 0, 0]
    assert float(drawn.mean()) == pytest.approx(3.75, abs=0.04) and float(drawn.max()) == 9.0
    assert float(drawn.std()) == pytest.approx(math.sqrt(5.9375), abs=0.04)
    # 몫이 0.5씩이던 예전 가중(묶음마다 같은 비중)과는 다른 값이다.
    assert train.standardization_stats([windows, fps], feature_spec, [0.25])[0][0] == pytest.approx(
        (0.75 * 1.0 + 0.25 * 5.0 + 0.75 * 3.0 + 0.25 * 5.0) / 2.0
    )
    with pytest.raises(ValueError, match="one extra share per extra window set"):
        train.standardization_stats([windows, fps, drop], feature_spec, [0.25])
    with pytest.raises(ValueError, match="sum to less than 1"):
        train.standardization_stats([windows, fps, drop], feature_spec, [0.5, 0.5])

    spec = train.with_standardization(feature_spec, mean, std, 1.7)
    assert spec.means == mean and spec.stds == std and spec.standardization.temperature == 1.7
    assert spec.pipeline_hash == feature_spec.pipeline_hash and spec.names == feature_spec.names
    with pytest.raises(ValueError, match="means and stds"):
        train.with_standardization(feature_spec, mean[:3], std, 1.0)


def test_fit_temperature_recovers_a_known_scale() -> None:
    rng = np.random.default_rng(0)
    true_logits = rng.normal(0.0, 1.5, size=(20_000, 3))
    probs = np.exp(true_logits) / np.exp(true_logits).sum(axis=1, keepdims=True)
    labels = (rng.random((20_000, 1)) > np.cumsum(probs, axis=1)).sum(axis=1)
    # 로짓을 2.5배 키운(지나치게 확신하는) 모델은 온도 2.5로 나눠야 보정된다.
    assert train.fit_temperature(2.5 * true_logits, labels) == pytest.approx(2.5, rel=0.05)
    assert train.fit_temperature(0.5 * true_logits, labels) == pytest.approx(0.5, rel=0.05)
    assert train.fit_temperature(true_logits, labels) == train.fit_temperature(true_logits, labels)  # 결정적

    # 모두 맞히는 로짓은 온도가 작을수록 가능도가 커진다: 아래 끝. 모두 틀리는 로짓은 위 끝.
    sure = np.eye(3)[labels] * 4.0
    assert train.fit_temperature(sure, labels) == pytest.approx(train.TEMPERATURE_MIN, rel=1e-3)
    assert train.fit_temperature(-sure, labels) == pytest.approx(train.TEMPERATURE_MAX, rel=1e-3)

    with pytest.raises(ValueError, match="logits"):
        train.fit_temperature(np.zeros((0, 3)), np.zeros(0, dtype=np.int64))
    with pytest.raises(ValueError, match="logits"):
        train.fit_temperature(np.zeros((4, 3)), np.zeros(3, dtype=np.int64))
    with pytest.raises(ValueError, match="finite"):
        train.fit_temperature(np.full((2, 3), np.nan), np.zeros(2, dtype=np.int64))


def test_video_macro_f1_is_the_report_metric(feature_spec: FeatureSpec, steps: pd.DataFrame) -> None:
    windows = make_windows(steps, 120, 1)
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3), size=len(windows))
    preds = evaluate.prediction_table(
        windows, probs, steps, evaluate.video_table(make_quality(steps)), run_id="r", family="tcn", arm=ARM, seed=0
    )
    expected = evaluate.window_metrics(preds).video_weighted.macro_f1
    scorer = train.VideoMacroF1(windows.meta)
    assert scorer(np.argmax(probs, axis=1)) == pytest.approx(expected, abs=1e-12)
    assert scorer(windows.y) == 1.0
    with pytest.raises(ValueError, match="predictions"):
        scorer(windows.y[:-1])


def test_small_helpers(experiment_config: ExperimentConfig) -> None:
    assert train.median_half_up([3]) == 3 and train.median_half_up([5, 1, 3]) == 3
    assert train.median_half_up([1, 2]) == 2 and train.median_half_up([2, 4, 6, 9]) == 5  # 가운데 두 값의 평균을 반올림
    assert train.median_half_up([4, 4, 5, 9]) == 5
    with pytest.raises(ValueError):
        train.median_half_up([])
    assert train.run_id_for("alert_baseline", False) == "train-alert_baseline"
    assert train.run_id_for("global", True, "-w1") == "train-global-w1-fast"
    assert train.data_note(True).startswith("SYNTHETIC data") and "pipeline check only" in train.data_note(True)
    assert train.data_note(False) == evaluate.REAL_NOTE
    assert train.resolve_device("cpu").type == "cpu"
    assert train.resolve_device("auto").type == ("cuda" if torch.cuda.is_available() else "cpu")

    # --data-root는 설정이 가리키는 데이터 루트를 바꾼다: rldd_root가 null이면 synth_dir, 있으면 rldd_root.
    # 기본 설정의 rldd_root 값에 기대지 않도록 두 경우를 모두 직접 만든다.
    assert train.with_data_root(experiment_config, None) is experiment_config
    synthetic = dataclasses.replace(
        experiment_config, paths=dataclasses.replace(experiment_config.paths, rldd_root=None)
    )
    moved = train.with_data_root(synthetic, Path("/data/elsewhere"))
    assert moved.paths.synth_dir == Path("/data/elsewhere") and moved.paths.rldd_root is None
    real = dataclasses.replace(
        experiment_config, paths=dataclasses.replace(experiment_config.paths, rldd_root=Path("/data/rldd"))
    )
    moved = train.with_data_root(real, Path("/mnt/rldd"))
    assert moved.paths.rldd_root == Path("/mnt/rldd") and moved.paths.synth_dir == experiment_config.paths.synth_dir


def test_step_table_fingerprint_is_a_content_hash(tmp_path: Path) -> None:
    table = make_steps((1,), n_steps=2, seed=4)  # 세션 3개 × 스텝 2개
    assert len(table) == 6
    # 설명에 적힌 정의를 struct로 따로 계산한다: 열마다 이름 + NUL + 값 바이트.
    digest = hashlib.sha256()
    for name in STEP_COLUMNS:
        values = table[name].tolist()
        if name in ("video_id", "session_id"):
            payload = "\0".join(values).encode("utf-8")
        elif name == "full_lookback":
            payload = bytes(int(value) for value in values)
        elif name in FEATURE_NAMES:
            payload = struct.pack("<6d", *values)
        else:
            payload = struct.pack("<6q", *values)
        digest.update(name.encode("utf-8") + b"\0" + payload)
    fingerprint = train.step_table_fingerprint(table)
    assert fingerprint == digest.hexdigest() and re.fullmatch(r"[0-9a-f]{64}", fingerprint)

    # 값 하나, 행 순서, 식별자가 바뀌면 달라진다.
    changed = table.copy()
    changed.loc[3, "perclos_60"] = float(np.nextafter(changed.loc[3, "perclos_60"], 1.0))
    assert train.step_table_fingerprint(changed) != fingerprint
    assert train.step_table_fingerprint(table.iloc[::-1].reset_index(drop=True)) != fingerprint
    renamed = table.copy()
    renamed.loc[0, "session_id"] = "S01_0_1"
    assert train.step_table_fingerprint(renamed) != fingerprint
    flagged = table.copy()
    flagged.loc[5, "full_lookback"] = False
    assert train.step_table_fingerprint(flagged) != fingerprint
    # 내용이 같으면 같다: 다른 정수 폭, 열 순서, 더 있는 열, parquet에 썼다 읽은 표(캐시에서 읽은 스텝 표).
    narrow = table.astype({"k": np.int32, "subject": np.int16})
    assert train.step_table_fingerprint(narrow) == fingerprint
    reordered = table[list(reversed(STEP_COLUMNS))].assign(extra=1.0)
    assert train.step_table_fingerprint(reordered) == fingerprint
    table.to_parquet(tmp_path / "steps.parquet", index=False)
    assert train.step_table_fingerprint(pd.read_parquet(tmp_path / "steps.parquet")) == fingerprint
    with pytest.raises(ValueError, match="lacks columns"):
        train.step_table_fingerprint(table.drop(columns=["k"]))


def test_predictions_file_carries_the_kind_of_data(tmp_path: Path, steps: pd.DataFrame) -> None:
    windows = make_windows(steps, 120, 1)
    probs = np.random.default_rng(0).dirichlet(np.ones(3), size=len(windows))
    preds = evaluate.prediction_table(
        windows, probs, steps, evaluate.video_table(make_quality(steps)), run_id="r", family="tcn", arm=ARM, seed=0
    )
    for synthetic, note in ((True, "SYNTHETIC data (drowsy.synth): pipeline check only"), (False, "REAL data:")):
        path = tmp_path / f"{synthetic}.parquet"
        train.write_predictions(preds, path, synthetic=synthetic)
        stored = json.loads(pq.read_schema(path).metadata[b"drowsy_predictions"])
        assert stored["synthetic"] is synthetic and stored["data_note"].startswith(note)
        assert train.read_predictions_marker(path) == stored
        # 표 자체는 그대로 읽힌다(표지는 parquet 메타데이터에만 있다).
        pd.testing.assert_frame_equal(evaluate.validate_predictions(pd.read_parquet(path)), preds)
    # 표지 없이 쓴 파일(예전 형식, 다른 도구가 쓴 파일)은 None이다.
    preds.to_parquet(tmp_path / "plain.parquet", index=False)
    assert train.read_predictions_marker(tmp_path / "plain.parquet") is None


def test_check_run_dir_refuses_to_replace_the_report_of_another_run(tmp_path: Path) -> None:
    identity = {"run_id": "train-alert_baseline", "synthetic": False}
    train.check_run_dir(tmp_path / "missing", identity)  # 폴더가 없다
    (tmp_path / "lgbm").mkdir()
    train.check_run_dir(tmp_path, identity)  # run.json이 없다(lgbm/만 있는 폴더)

    stored = {"data_note": "x", "run_id": "train-alert_baseline", "synthetic": False, "fast": False}
    train.write_json(stored, tmp_path / "run.json")
    train.check_run_dir(tmp_path, identity)  # 같은 실행의 재실행은 덮어쓴다
    # 다른 학습 조건, 축소 설정(run_id에 들어 있다), 다른 종류의 데이터.
    with pytest.raises(ValueError, match="run_id: 'train-alert_baseline' there, 'train-global' now") as caught:
        train.check_run_dir(tmp_path, {**identity, "run_id": "train-global"})
    assert "--overwrite" in str(caught.value) and str(tmp_path) in str(caught.value)
    with pytest.raises(ValueError, match="'train-alert_baseline-fast' now"):
        train.check_run_dir(tmp_path, {**identity, "run_id": "train-alert_baseline-fast"})
    with pytest.raises(ValueError, match="synthetic: False there, True now"):
        train.check_run_dir(tmp_path, {**identity, "synthetic": True})
    # 기록에 없는 키는 다른 값이다(LightGBM 폴더의 identity를 신경망 기록에 대 본 경우).
    with pytest.raises(ValueError, match="arms: None there"):
        train.check_run_dir(tmp_path, {"arms": ["alert_baseline"], "synthetic": False})
    train.check_run_dir(tmp_path, {**identity, "run_id": "train-global"}, overwrite=True)

    (tmp_path / "run.json").write_text("{ not json", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be read"):
        train.check_run_dir(tmp_path, identity)
    (tmp_path / "run.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be read"):
        train.check_run_dir(tmp_path, identity)
    train.check_run_dir(tmp_path, identity, overwrite=True)

    # 다른 기록 파일 이름(최종 모델의 final.json)도 같은 방식으로 본다.
    final_identity = {"arm": "alert_baseline", "fast": False, "synthetic": False}
    train.check_run_dir(tmp_path, final_identity, record_file="final.json")  # final.json이 없다
    train.write_json({"data_note": "x", **final_identity}, tmp_path / "final.json")
    train.check_run_dir(tmp_path, final_identity, record_file="final.json")
    with pytest.raises(ValueError, match="fast: False there, True now; synthetic: False there, True now"):
        train.check_run_dir(tmp_path, {**final_identity, "fast": True, "synthetic": True}, record_file="final.json")


def test_fast_profile_and_mixing_share_in_the_default_config(
    experiment_config: ExperimentConfig, config_path: Path, tmp_path: Path
) -> None:
    # 학습 표본의 몫(가정): 원본 0.5, 15 fps 누락 모사 0.25, 프레임 누락 변형 0.25(변형 하나).
    training = experiment_config.training
    assert (training.fps_sim_share, training.frame_drop_share, training.frame_drop_variants) == (0.25, 0.25, 1)
    assert (training.frame_drop_min, training.frame_drop_max) == (0.10, 0.30)
    assert training.ordinal_neighbor_mass == 0.2 and training.label_smoothing == 0.1
    fast = apply_fast_profile(experiment_config)
    profile = experiment_config.fast
    # 줄어드는 것: 학습 길이, 시드, 탐색 범위, LightGBM 횟수, 부트스트랩 횟수.
    assert (fast.training.max_units, fast.training.patience_units, fast.training.updates_per_unit) == (
        profile.max_units, profile.patience_units, profile.updates_per_unit,
    ) == (6, 2, 50)  # fmt: skip
    assert fast.training.seeds == (0,) and fast.training.tcn.channels == (16,) and fast.training.gru.hidden == (32,)
    assert fast.lgbm.num_leaves == (7,) and fast.lgbm.min_data_in_leaf == (20,)
    assert fast.lgbm.max_rounds == 40 and fast.lgbm.early_stopping_rounds == 30
    assert fast.evaluation.bootstrap_n == 200
    # 그대로인 것: 사전에 고정한 나머지 설정과 계약.
    assert fast.windows == experiment_config.windows and fast.export == experiment_config.export
    assert fast.paths == experiment_config.paths and fast.extraction == experiment_config.extraction
    kept = (
        "lr", "weight_decay", "batch_size", "label_smoothing", "noise_sigma", "fps_sim_share", "final_seed",
        "frame_drop_share", "frame_drop_variants", "ordinal_neighbor_mass",
    )  # fmt: skip
    assert all(getattr(fast.training, name) == getattr(experiment_config.training, name) for name in kept)
    assert fast.training.tcn.dilations == (1, 2, 4, 8, 16)
    assert fast.evaluation.model_gate == experiment_config.evaluation.model_gate
    # 기본 설정 자체는 계획서의 값 그대로다(축소 설정은 --fast를 줄 때만 적용된다).
    assert experiment_config.training.max_units == 60 and experiment_config.evaluation.bootstrap_n == 10000

    text = config_path.read_text(encoding="utf-8")
    for position, (old, new, message) in enumerate(
        (
            ("  fps_sim_share: 0.25", "  fps_sim_share: 1.0", "fps_sim_share must be within"),
            ("  fps_sim_share: 0.25", "  fps_sim_share: 0.75", r"fps_sim_share \+ frame_drop_share must be below 1"),
            ("  frame_drop_share: 0.25", "  frame_drop_share: -0.1", "frame_drop_share must be within"),
            ("  frame_drop_variants: 1", "  frame_drop_variants: -1", "frame_drop_variants must be >= 0"),
            ("  ordinal_neighbor_mass: 0.2", "  ordinal_neighbor_mass: 0.5", "ordinal_neighbor_mass must be within"),
            ("  max_units: 6 ", "  max_units: 0 ", "fast"),
            ("  seeds: [0]  ", "  seeds: []   ", "fast.seeds"),
        )
    ):
        assert text.count(old) == 1, old
        broken = tmp_path / f"broken_{position}.yaml"
        broken.write_text(text.replace(old, new), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_config(broken)
    # 설정 파일의 주석이 이 키들을 "아직 읽지 않는다"고 적고 있지 않다(학습이 읽는다).
    for key in ("ordinal_neighbor_mass", "frame_drop_share", "frame_drop_variants", "fps_sim_share"):
        line = next(item for item in text.splitlines() if item.startswith(f"  {key}:"))
        assert "아직" not in line and "11.4" not in line, key


def test_candidates_are_the_preregistered_grids(experiment_config: ExperimentConfig) -> None:
    grid = train.candidates(experiment_config.training)
    assert [candidate.name for candidate in grid] == [
        "tcn(channels=16,dropout=0.1)", "tcn(channels=16,dropout=0.3)",
        "tcn(channels=32,dropout=0.1)", "tcn(channels=32,dropout=0.3)",
        "gru(hidden=32,dropout=0.1)", "gru(hidden=32,dropout=0.3)",
        "gru(hidden=64,dropout=0.1)", "gru(hidden=64,dropout=0.3)",
    ]  # fmt: skip
    sizes = [count_parameters(build_model(c.family, N_FEATURES, **c.hyper)) for c in grid]
    assert sizes[0] == sizes[1] == 8547 and sizes[2] == 30915 and max(sizes) < 100_000
    assert grid[0].hyper == {"channels": 16, "kernel_size": 3, "dilations": [1, 2, 4, 8, 16], "dropout": 0.1}
    assert [c.family for c in train.candidates(experiment_config.training, ["gru"])] == ["gru"] * 4
    with pytest.raises(ValueError, match="unknown model family"):
        train.candidates(experiment_config.training, ["lstm"])
    # 축소 설정은 후보를 계열마다 하나로 줄인다.
    fast = train.candidates(apply_fast_profile(experiment_config).training)
    assert [candidate.name for candidate in fast] == ["tcn(channels=16,dropout=0.1)", "gru(hidden=32,dropout=0.1)"]


def score(name: str, mean: float, std: float, n_parameters: int) -> train.CandidateScore:
    candidate = train.Candidate(name.split("-")[0], {}, name)
    return train.CandidateScore(candidate, n_parameters, (1, 2), (mean - std, mean + std), mean, std, (3, 5), 4, 1.0)


def test_tie_rule_prefers_fewer_parameters_within_the_inner_fold_std() -> None:
    best = score("tcn-big", 0.80, 0.05, 30_000)
    close = score("gru-mid", 0.77, 0.01, 16_000)  # 차이 0.03 ≤ 0.05
    closer_small = score("gru-small", 0.76, 0.20, 5_000)  # 차이 0.04 ≤ 0.05
    far = score("tcn-tiny", 0.70, 0.30, 1_000)  # 차이 0.10 > 0.05: 가장 작아도 고르지 않는다
    assert train.choose_candidate([best, close, closer_small, far]).candidate.name == "gru-small"
    assert train.choose_candidate([far, closer_small, close, best]).candidate.name == "gru-small"  # 순서와 무관
    # 표준편차는 평균이 가장 높은 후보의 것이다. 0이면 가장 높은 후보를 그대로 고른다.
    assert train.choose_candidate([score("tcn-big", 0.80, 0.0, 30_000), close]).candidate.name == "tcn-big"
    # 파라미터 수가 같으면 평균이 높은 쪽, 그것도 같으면 목록에서 앞선 쪽.
    a, b = score("gru-a", 0.78, 0.05, 5_000), score("gru-b", 0.79, 0.05, 5_000)
    assert train.choose_candidate([a, b]).candidate.name == "gru-b"
    assert train.choose_candidate([score("gru-a", 0.79, 0.05, 5_000), b]).candidate.name == "gru-a"
    with pytest.raises(ValueError, match="nothing to choose"):
        train.choose_candidate([])
    with pytest.raises(ValueError, match="finite"):
        train.choose_candidate([score("gru-a", math.nan, 0.0, 1)])


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_fit_is_deterministic_for_a_fixed_seed_on_cpu(
    kind: str, cfg: ExperimentConfig, feature_spec: FeatureSpec, steps: pd.DataFrame
) -> None:
    train_windows = windows_of(steps, (1, 2, 13, 14))
    val_windows = windows_of(steps, (25, 26))
    config = train_config(cfg, feature_spec, kind)
    first = train.fit(train_windows, val_windows, config, seed=7)
    second = train.fit(train_windows, val_windows, config, seed=7)
    assert states_equal(first.net, second.net)
    assert first.history == second.history and first.units == second.units
    assert first.temperature == second.temperature and first.spec == second.spec
    assert np.array_equal(first.val_logits, second.val_logits)
    other = train.fit(train_windows, val_windows, config, seed=8)
    assert not states_equal(first.net, other.net)

    assert first.kind == kind and first.seed == 7 and first.n_parameters == count_parameters(first.net)
    assert not first.net.training and next(first.net.parameters()).device.type == "cpu"
    assert 1 <= first.units <= first.units_run <= cfg.training.max_units
    assert first.updates == first.units * cfg.training.updates_per_unit
    assert len(first.history) == first.units_run and first.best_macro_f1 == max(first.history)
    assert first.train_subjects == (1, 2, 13, 14) and first.val_subjects == (25, 26)


def test_fit_keeps_the_best_unit_and_stops_after_the_patience(
    cfg: ExperimentConfig, feature_spec: FeatureSpec, steps: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_windows = windows_of(steps, (1, 2, 13, 14))
    val_windows = windows_of(steps, (25, 26))
    long_cfg = tiny_config(cfg, max_units=9, patience_units=2)
    scripted = iter([0.50, 0.70, 0.70, 0.60, 0.99, 0.99, 0.99, 0.99, 0.99])
    monkeypatch.setattr(train.VideoMacroF1, "__call__", lambda self, pred: next(scripted))
    result = train.fit(train_windows, val_windows, train_config(long_cfg, feature_spec), seed=0)
    # 2단위가 가장 좋았고(같은 값의 3단위는 새 기록이 아니다) 그 뒤 2단위 동안 나아지지 않아 4단위에서 멈춘다.
    assert result.history == (0.50, 0.70, 0.70, 0.60)
    assert (result.units, result.units_run, result.best_macro_f1) == (2, 4, 0.70)
    assert result.updates == 2 * long_cfg.training.updates_per_unit
    monkeypatch.undo()

    # 남긴 가중치는 2단위까지만 학습한 것과 같다(검증은 난수를 쓰지 않는다).
    fixed = train.fit(train_windows, None, train_config(long_cfg, feature_spec, fixed_units=2), seed=0)
    assert states_equal(result.net, fixed.net)
    assert (fixed.units, fixed.units_run, fixed.history) == (2, 2, ())
    assert math.isnan(fixed.best_macro_f1) and fixed.val_logits is None and fixed.val_subjects == ()
    assert fixed.temperature == 1.0
    # 다시 학습할 때의 온도는 설정으로 받는다.
    refit = train.fit(train_windows, None, train_config(long_cfg, feature_spec, fixed_units=1, temperature=1.8), 0)
    assert refit.temperature == 1.8 and refit.spec.standardization.temperature == 1.8


def test_fit_rejects_inconsistent_arguments(
    cfg: ExperimentConfig, feature_spec: FeatureSpec, steps: pd.DataFrame
) -> None:
    train_windows = windows_of(steps, (1, 2))
    val_windows = windows_of(steps, (25,))
    config = train_config(cfg, feature_spec)
    with pytest.raises(ValueError, match="fixed_units"):
        train.fit(train_windows, None, config, 0)
    with pytest.raises(ValueError, match="refits without validation"):
        train.fit(train_windows, val_windows, dataclasses.replace(config, fixed_units=2), 0)
    with pytest.raises(ValueError, match="at least one training window"):
        train.fit(train_windows.subset(np.zeros(len(train_windows), dtype=bool)), val_windows, config, 0)
    with pytest.raises(ValueError, match="validation set has no windows"):
        train.fit(train_windows, val_windows.subset(np.zeros(len(val_windows), dtype=bool)), config, 0)
    sgd = tiny_config(cfg, optimizer="sgd")
    with pytest.raises(ValueError, match="optimizer"):
        train.fit(train_windows, val_windows, train_config(sgd, feature_spec), 0)


def test_fit_refuses_to_start_when_subjects_overlap(
    cfg: ExperimentConfig, feature_spec: FeatureSpec, steps: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[int] = []
    monkeypatch.setattr(train, "seed_everything", lambda seed: started.append(seed))
    config = train_config(cfg, feature_spec)
    with pytest.raises(SubjectOverlapError, match="subject 13"):
        train.fit(windows_of(steps, (1, 2, 13)), windows_of(steps, (13, 14)), config, 0)
    # 두 번째 추출의 창이 검증 피험자를 담고 있어도 시작하지 않는다.
    with pytest.raises(SubjectOverlapError, match="subject 25"):
        train.fit(
            windows_of(steps, (1, 2)),
            windows_of(steps, (25, 26)),
            config,
            0,
            extra_train=[source("fps15", 0.25, windows_of(steps, (1, 25)))],
        )
    # 프레임 누락 변형의 창도 마찬가지다(출처가 여럿이어도 어느 하나에 검증 피험자가 있으면 멈춘다).
    with pytest.raises(SubjectOverlapError, match="subject 26"):
        train.fit(
            windows_of(steps, (1, 2)),
            windows_of(steps, (25, 26)),
            config,
            0,
            extra_train=[
                source("fps15", 0.25, windows_of(steps, (1, 2))),
                source("frame_drop_v0", 0.25, windows_of(steps, (2, 26))),
            ],
        )
    assert started == []  # 시드 고정(학습의 첫 단계)에도 이르지 못했다

    # 평가도 마찬가지다: 학습이나 검증에 쓴 피험자의 창은 평가하지 않는다.
    monkeypatch.undo()
    result = train.fit(windows_of(steps, (1, 2)), windows_of(steps, (13,)), config, 0)
    for subjects in ((1,), (13, 25)):
        with pytest.raises(SubjectOverlapError):
            train.predict_proba(result, windows_of(steps, subjects))
    probs = train.predict_proba(result, windows_of(steps, (25, 26)))
    assert probs.shape == (len(windows_of(steps, (25, 26))), 3) and probs.dtype == np.float64
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_standardization_and_temperature_never_see_validation_data_for_fitting(
    cfg: ExperimentConfig, feature_spec: FeatureSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 검증 피험자의 특징을 크게 옮겨 둔다. 표준화 상수에 섞였다면 평균이 달라진다.
    train_steps = make_steps((1, 2, 13, 14), seed=0)
    val_steps = make_steps((25, 26), seed=1, offset=0.4)
    train_windows = make_windows(train_steps, 120, 1)
    val_windows = make_windows(val_steps, 120, 1)

    seen_by_stats: list[tuple[int, ...]] = []
    seen_by_temperature: list[tuple[int, np.ndarray]] = []
    real_stats, real_temperature = train.standardization_stats, train.fit_temperature

    def spy_stats(windows: Any, spec: FeatureSpec, extra_shares: Any = None) -> Any:
        sets = [windows] if isinstance(windows, WindowSet) else list(windows)
        seen_by_stats.append(tuple(sorted({s for item in sets for s in train.window_subjects(item)})))
        return real_stats(windows, spec, extra_shares)

    def spy_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
        seen_by_temperature.append((len(labels), np.array(labels)))
        return real_temperature(logits, labels)

    monkeypatch.setattr(train, "standardization_stats", spy_stats)
    monkeypatch.setattr(train, "fit_temperature", spy_temperature)
    result = train.fit(train_windows, val_windows, train_config(cfg, feature_spec), seed=0)

    assert seen_by_stats == [(1, 2, 13, 14)]  # 표준화 상수는 학습 피험자의 창만 봤다
    mean, std = real_stats(train_windows, feature_spec)
    assert result.spec.means == mean and result.spec.stds == std
    pooled_mean, _ = real_stats(joined(train_windows, val_windows), feature_spec)
    assert abs(pooled_mean[0] - mean[0]) > 0.05  # 검증 창이 섞였다면 이만큼 달랐을 것이다
    # 온도는 검증 창의 로짓으로만 맞췄다(창 수와 정답이 검증 묶음의 것이다).
    assert len(seen_by_temperature) == 1
    assert seen_by_temperature[0][0] == len(val_windows) and np.array_equal(seen_by_temperature[0][1], val_windows.y)
    assert result.temperature == real_temperature(result.val_logits, val_windows.y)
    assert result.spec.standardization.temperature == result.temperature
    assert train.TEMPERATURE_MIN <= result.temperature <= train.TEMPERATURE_MAX


def test_fit_mixes_the_second_extraction_into_training_only(
    cfg: ExperimentConfig, feature_spec: FeatureSpec, steps: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_windows = windows_of(steps, (1, 2, 13, 14))
    fps_windows = windows_of(make_steps(THREE_FOLDS, seed=5, offset=0.03), (1, 2, 13, 14))
    drop_windows = windows_of(make_steps(THREE_FOLDS, seed=6, offset=0.06), (1, 2, 13, 14))
    val_windows = windows_of(steps, (25, 26))
    built: list[tuple[list[int], int, list[float]]] = []
    real_sampler = train.BalancedSampler

    def spy_sampler(metas: Any, seed: int, extra_shares: Any = ()) -> Any:
        built.append(([len(meta) for meta in metas], seed, list(extra_shares)))
        return real_sampler(metas, seed, extra_shares)

    monkeypatch.setattr(train, "BalancedSampler", spy_sampler)
    config = train_config(cfg, feature_spec)
    extras = [source("fps15", 0.25, fps_windows), source("frame_drop_v0", 0.25, drop_windows)]
    mixed = train.fit(train_windows, val_windows, config, 0, extra_train=extras)
    # 표집기는 세 출처를 원본 0.5, 15 fps 0.25, 프레임 누락 변형 0.25의 몫으로 받는다.
    assert built == [([len(train_windows), len(fps_windows), len(drop_windows)], 0, [0.25, 0.25])]
    assert mixed.sources == (("native", 0.5), ("fps15", 0.25), ("frame_drop_v0", 0.25))
    # 표준화 상수는 세 출처의 학습 창에서, 표집과 같은 몫으로 나온다(같은 몫이 아니면 값이 다르다).
    weighted = train.standardization_stats([train_windows, fps_windows, drop_windows], feature_spec, [0.25, 0.25])
    assert (mixed.spec.means, mixed.spec.stds) == weighted
    assert weighted != train.standardization_stats([train_windows, fps_windows, drop_windows], feature_spec)
    assert mixed.train_subjects == (1, 2, 13, 14) and len(mixed.val_logits) == len(val_windows)
    assert mixed.held_features == ()

    # 없는 출처의 몫은 원본으로 간다: 창이 없는 출처, 몫이 0인 출처는 쓰지 않는다.
    built.clear()
    empty = drop_windows.subset(np.zeros(len(drop_windows), dtype=bool))
    only_fps = train.fit(
        train_windows, val_windows, config, 0,
        extra_train=[source("fps15", 0.25, fps_windows), source("frame_drop_v0", 0.25, empty)],
    )  # fmt: skip
    assert built == [([len(train_windows), len(fps_windows)], 0, [0.25])]
    assert only_fps.sources == (("native", 0.75), ("fps15", 0.25))
    assert only_fps.spec.means == train.standardization_stats([train_windows, fps_windows], feature_spec, [0.25])[0]
    built.clear()
    alone = train.fit(
        train_windows, val_windows, config, 0,
        extra_train=[source("fps15", 0.0, fps_windows), source("frame_drop_v0", 0.25, empty)],
    )  # fmt: skip
    assert built == [([len(train_windows)], 0, [])] and alone.sources == (("native", 1.0),)
    assert alone.spec.means == train.standardization_stats(train_windows, feature_spec)[0]
    assert states_equal(alone.net, train.fit(train_windows, val_windows, config, 0).net)

    # 잘못된 출처 목록은 학습 전에 거부한다.
    with pytest.raises(ValueError, match="distinct names"):
        train.fit(train_windows, val_windows, config, 0, extra_train=[extras[0], source("fps15", 0.1, drop_windows)])
    with pytest.raises(ValueError, match="sum to less than 1"):
        train.fit(
            train_windows, val_windows, config, 0,
            extra_train=[source("fps15", 0.5, fps_windows), source("frame_drop_v0", 0.5, drop_windows)],
        )  # fmt: skip
    with pytest.raises(TypeError, match="TrainSource"):
        train.fit(train_windows, val_windows, config, 0, extra_train=fps_windows)  # type: ignore[arg-type]


def test_fit_with_a_partial_second_extraction_keeps_every_training_subject(
    cfg: ExperimentConfig, feature_spec: FeatureSpec, steps: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 두 번째 추출이 학습 피험자 넷 중 한 명(1번)의 창만 담고 있다. 표본은 여전히 네 피험자에서 고르게 나온다.
    extra_steps = make_steps((1,), seed=5, offset=0.03)
    train_windows = windows_of(steps, (1, 2, 13, 14))
    extra_windows = windows_of(extra_steps, (1,))
    val_windows = windows_of(steps, (25, 26))
    drawn: list[np.ndarray] = []
    real_draw = train.BalancedSampler.draw

    def spy_draw(self: Any, n: int) -> np.ndarray:
        out = real_draw(self, n)
        drawn.append(out)
        return out

    monkeypatch.setattr(train.BalancedSampler, "draw", spy_draw)
    half = [source("fps15", 0.5, extra_windows)]
    result = train.fit(train_windows, val_windows, train_config(cfg, feature_spec), 0, extra_train=half)
    assert result.train_subjects == (1, 2, 13, 14)
    picks = np.concatenate(drawn)
    subjects = np.concatenate([train_windows.meta["subject"].to_numpy(), extra_windows.meta["subject"].to_numpy()])
    shares = pd.Series(subjects[picks]).value_counts(normalize=True)
    # 표본 수가 적어(단위당 64개) 넉넉히 본다. 추출을 먼저 고르던 구현에서는 1번이 0.5 + 0.5 / 4 = 0.625였다.
    assert sorted(shares.index) == [1, 2, 13, 14] and shares.max() < 0.45 and shares.min() > 0.1
    # 두 번째 추출에서 나온 표본은 1번의 것뿐이고 전체의 1/4 × 0.5 = 1/8 근처다.
    from_extra = picks >= len(train_windows)
    assert set(subjects[picks[from_extra]]) == {1} and 0.03 < from_extra.mean() < 0.25


# ---------------------------------------------------------------------------
# 부가 실행의 선택 사항: 순서형 소프트 라벨, 속도 특징 제외
# ---------------------------------------------------------------------------


def test_ordinal_targets_give_the_neighbor_mass_to_the_ordinal_neighbors() -> None:
    # 결정된 정의: 가운데 클래스는 양쪽에 0.1씩, 끝 클래스는 하나뿐인 이웃에 0.2.
    assert train.ordinal_targets(3, 0.2).tolist() == [[0.8, 0.2, 0.0], [0.1, 0.8, 0.1], [0.0, 0.2, 0.8]]
    four = train.ordinal_targets(4, 0.3)
    assert np.allclose(four, [[0.7, 0.3, 0, 0], [0.15, 0.7, 0.15, 0], [0, 0.15, 0.7, 0.15], [0, 0, 0.3, 0.7]])
    assert np.allclose(four.sum(axis=1), 1.0) and np.array_equal(train.ordinal_targets(3, 0.0), np.eye(3))
    for classes, mass in ((1, 0.2), (3, 0.5), (3, -0.1)):
        with pytest.raises(ValueError):
            train.ordinal_targets(classes, mass)


def test_fit_with_ordinal_labels_replaces_label_smoothing(
    cfg: ExperimentConfig, feature_spec: FeatureSpec, steps: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_windows = windows_of(steps, (1, 2, 13, 14))
    val_windows = windows_of(steps, (25, 26))
    made: list[dict[str, Any]] = []
    targets: list[torch.Tensor] = []

    class Recorder(torch.nn.CrossEntropyLoss):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            made.append(dict(kwargs))
            super().__init__(*args, **kwargs)

        def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # noqa: A002
            targets.append(target.detach().clone())
            return super().forward(input, target)

    monkeypatch.setattr(train.nn, "CrossEntropyLoss", Recorder)
    assert cfg.training.ordinal_neighbor_mass == 0.2 and cfg.training.label_smoothing == 0.1
    ordinal = train.fit(train_windows, val_windows, train_config(cfg, feature_spec, variant="ordinal"), 0)
    # 손실은 라벨 스무딩 없이 만들어지고, 목표는 클래스 확률 [배치, 3]이며 세 분포 가운데 하나다.
    assert made == [{}]
    rows = torch.cat(targets)
    assert rows.dtype == torch.float32 and rows.shape[1] == 3 and len(targets) == ordinal.units_run * 4
    expected = torch.tensor([[0.8, 0.2, 0.0], [0.1, 0.8, 0.1], [0.0, 0.2, 0.8]])
    matches = (rows[:, None, :] == expected[None, :, :]).all(dim=2)
    assert bool(matches.any(dim=1).all()) and bool(matches.any(dim=0).all())
    # 조기 종료의 지표와 온도는 원래 라벨로 계산한다.
    assert np.array_equal(ordinal.val_labels, val_windows.y) and ordinal.held_features == ()

    made.clear()
    targets.clear()
    base = train.fit(train_windows, val_windows, train_config(cfg, feature_spec), 0)
    assert made == [{"label_smoothing": 0.1}]
    assert all(target.dtype == torch.int64 and target.ndim == 1 for target in targets)
    assert not states_equal(base.net, ordinal.net)  # 같은 시드, 같은 표본이어도 목표가 달라 가중치가 다르다
    with pytest.raises(ValueError, match="variant must be one of"):
        train.fit(train_windows, val_windows, train_config(cfg, feature_spec, variant="focal"), 0)


def test_fit_without_velocity_holds_the_two_standardised_inputs_at_zero(
    cfg: ExperimentConfig, feature_spec: FeatureSpec, steps: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert train.VELOCITY_FEATURES == (7, 8)
    assert [FEATURE_NAMES[index] for index in (7, 8)] == ["closing_vel_mean_60", "opening_vel_mean_60"]
    train_windows = windows_of(steps, (1, 2, 13, 14))
    val_windows = windows_of(steps, (25, 26))
    seen: list[tuple[bool, float, float]] = []  # (학습 모드인지, 속도 열의 최대 절댓값, 다른 열의 최대 절댓값)
    real_build = train.build_model

    def build(kind: str, n_features: int, **hyper: Any) -> torch.nn.Module:
        net = real_build(kind, n_features, **hyper)

        def record(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            z = inputs[0]
            others = [index for index in range(N_FEATURES) if index not in (7, 8)]
            seen.append((module.training, float(z[..., [7, 8]].abs().max()), float(z[..., others].abs().max())))

        net.register_forward_pre_hook(record)
        return net

    monkeypatch.setattr(train, "build_model", build)
    assert cfg.training.noise_sigma > 0.0  # 잡음 증강이 켜져 있어도 두 입력은 0이다
    held = train.fit(train_windows, val_windows, train_config(cfg, feature_spec, variant="no_velocity"), 0)
    assert held.held_features == (7, 8)
    assert any(training for training, _, _ in seen) and any(not training for training, _, _ in seen)
    assert all(velocity == 0.0 and others > 0.0 for _, velocity, others in seen)
    # 입력 수 F와 표준화 상수는 그대로다(두 열의 평균·표준편차도 계산해 spec에 넣는다).
    assert held.spec.means == train.standardization_stats(train_windows, feature_spec)[0]
    assert next(held.net.parameters()).shape[1] == N_FEATURES or held.kind == KIND_TCN

    # 평가도 같은 입력을 0으로 둔다: 시험 창의 속도 특징을 아무 값으로 바꿔도 확률이 같다.
    test_windows = windows_of(steps, (25, 26)).subset(np.arange(6))
    refit_config = train_config(cfg, feature_spec, variant="no_velocity", fixed_units=1)
    other_model = train.fit(train_windows, None, refit_config, 0)
    scrambled = WindowSet(x=test_windows.x.copy(), y=test_windows.y, meta=test_windows.meta)
    scrambled.x[..., 7] = 123.0
    scrambled.x[..., 8] = -45.0
    seen.clear()
    assert np.array_equal(train.predict_proba(other_model, test_windows), train.predict_proba(other_model, scrambled))
    assert seen and all(velocity == 0.0 for _, velocity, _ in seen)
    # 주 실행의 모델은 그 두 입력을 읽는다(같은 바꿔치기에 확률이 달라진다).
    base = train.fit(train_windows, None, train_config(cfg, feature_spec, fixed_units=1), 0)
    assert base.held_features == ()
    assert not np.allclose(train.predict_proba(base, test_windows), train.predict_proba(base, scrambled))
    # log1p 열은 평균으로 바꿔 넣는 방법을 쓸 수 없다.
    wrapper = train.export_wrapper(base)
    with pytest.raises(ValueError, match="log1p column"):
        train._hold_at_mean(wrapper, torch.zeros(1, 120, N_FEATURES), [9])


def test_mix_sources_split_the_shares_and_give_missing_ones_to_native(
    experiment_config: ExperimentConfig, feature_spec: FeatureSpec
) -> None:
    cfg = tiny_config(experiment_config)
    data = make_data(feature_spec, THREE_FOLDS, with_extra=True, with_frame_drop=True)
    sources = train.mix_sources(data, cfg, ARM)
    assert [(item.name, item.share) for item in sources] == [("fps15", 0.25), ("frame_drop_v0", 0.25)]
    assert train.source_shares(sources) == {"native": 0.5, "fps15": 0.25, "frame_drop_v0": 0.25}
    assert sources[0].steps is data.extra_tables[ARM].steps and sources[1].steps is data.frame_drop_tables[ARM][0].steps
    # 변형이 둘이면 몫을 똑같이 나눈다.
    table = data.frame_drop_tables[ARM][0]
    second = dataclasses.replace(table, frame_drop=FrameDropSpec(variant=4, seed=0, drop_min=0.1, drop_max=0.3))
    two = dataclasses.replace(data, frame_drop_tables={ARM: (table, second)})
    assert train.source_shares(train.mix_sources(two, cfg, ARM)) == {
        "native": 0.5, "fps15": 0.25, "frame_drop_v0": 0.125, "frame_drop_v4": 0.125,
    }  # fmt: skip
    # 없는 출처(표가 없거나 몫이 0)의 몫은 원본으로 간다. 학습 조건이 아닌 조건에는 출처가 없다.
    without_fps = dataclasses.replace(data, extra_tables={})
    assert train.source_shares(train.mix_sources(without_fps, cfg, ARM)) == {"native": 0.75, "frame_drop_v0": 0.25}
    without_drop = dataclasses.replace(data, frame_drop_tables={})
    assert train.source_shares(train.mix_sources(without_drop, cfg, ARM)) == {"native": 0.75, "fps15": 0.25}
    zero = tiny_config(experiment_config, fps_sim_share=0.0, frame_drop_share=0.0)
    assert train.mix_sources(data, zero, ARM) == () and train.source_shares(()) == {"native": 1.0}
    assert train.mix_sources(data, cfg, APP_ARM) == ()
    # 피험자를 골라낸 사본: 어느 출처든 주어진 피험자의 스텝과 창만 남는다.
    part = sources[1].of((1, 13))
    assert sorted(part.steps["subject"].unique()) == [1, 13] and (part.name, part.share) == ("frame_drop_v0", 0.25)
    windows = part.windows(120)
    assert train.window_subjects(windows.windows) == (1, 13) and len(windows.windows) == 2 * 3 * 4
    assert train.window_subjects(windows.of((13,)).windows) == (13,)


def test_weights_sha256_is_a_hash_of_names_shapes_and_values() -> None:
    net = torch.nn.Linear(2, 1)
    with torch.no_grad():
        net.weight.copy_(torch.tensor([[1.0, 2.0]]))
        net.bias.copy_(torch.tensor([0.5]))
    expected = hashlib.sha256(
        b"bias\0float32\0" + b"1\0" + struct.pack("<f", 0.5)
        + b"weight\0float32\0" + b"1,2\0" + struct.pack("<2f", 1.0, 2.0)
    ).hexdigest()  # fmt: skip
    assert train.weights_sha256(net) == expected
    with torch.no_grad():
        net.bias.copy_(torch.tensor([0.5000001]))
    assert train.weights_sha256(net) != expected


# ---------------------------------------------------------------------------
# 중첩 교차검증의 부기 (fit을 대역으로 바꿔서)
# ---------------------------------------------------------------------------


class StubFit:
    """`train.fit`의 대역. 호출을 기록하고, 후보별로 정해 둔 내부 검증 값과 학습 길이를 돌려준다."""

    # 후보 이름 → 안쪽 분할 순서대로의 (macro-F1, 단위 수)
    SCRIPT: dict[str, list[tuple[float, int]]] = {
        "tcn(channels=8,dropout=0.1)": [(0.70, 2), (0.70, 2), (0.70, 2)],
        "tcn(channels=16,dropout=0.1)": [(0.74, 3), (0.78, 5), (0.76, 4)],  # 평균 0.76, 표준편차 0.02
        "gru(hidden=8,dropout=0.1)": [(0.76, 2), (0.76, 2), (0.76, 7)],
        "gru(hidden=16,dropout=0.1)": [(0.75, 6), (0.79, 6), (0.77, 6)],  # 평균 0.77, 표준편차 0.02
    }

    def __init__(self, spec: FeatureSpec, settings: Any) -> None:
        self.spec = spec
        self.calls: list[dict[str, Any]] = []
        self._inner_count: dict[str, int] = {}
        self._names = {
            json.dumps(candidate.hyper, sort_keys=True) + candidate.family: candidate.name
            for candidate in train.candidates(settings)
        }

    def __call__(
        self, train_ws: WindowSet, val_ws: WindowSet | None, cfg: train.TrainConfig, seed: int, extra_train: Any = None
    ) -> train.FitResult:
        name = self._names[json.dumps(dict(cfg.hyper), sort_keys=True) + cfg.kind]
        call: dict[str, Any] = {
            "name": name,
            "train": train.window_subjects(train_ws),
            "val": train.window_subjects(val_ws) if val_ws is not None else None,
            "extra": tuple(
                (item.name, item.share, train.window_subjects(item.windows)) for item in (extra_train or ())
            ),
            "seed": seed,
            "fixed_units": cfg.fixed_units,
            "temperature": cfg.temperature,
            "variant": cfg.variant,
        }
        self.calls.append(call)
        torch.manual_seed(len(self.calls))
        net = build_model(cfg.kind, N_FEATURES, **cfg.hyper).eval()
        rng = np.random.default_rng(len(self.calls))
        if val_ws is None:
            macro_f1, units, temperature = math.nan, int(cfg.fixed_units or 0), cfg.temperature
            logits, labels = None, None
        else:
            position = self._inner_count.get(name, 0)
            self._inner_count[name] = position + 1
            macro_f1, units = self.SCRIPT[name][position % 3]
            logits, labels = rng.normal(0.0, 2.0, size=(len(val_ws), 3)), np.asarray(val_ws.y)
            temperature = train.fit_temperature(logits, labels)
            call["val_logits"], call["val_labels"] = logits, labels
        return train.FitResult(
            net=net, kind=cfg.kind, hyper=dict(cfg.hyper), spec=train.with_standardization(
                self.spec, self.spec.means, self.spec.stds, float(temperature or 1.0)
            ),
            temperature=float(temperature or 1.0), seed=seed, units=units, units_run=units, updates=units * 4,
            history=(), best_macro_f1=macro_f1, val_logits=logits, val_labels=labels,
            n_parameters=count_parameters(net), train_subjects=call["train"], val_subjects=call["val"] or (),
        )  # fmt: skip


def test_nested_cv_bookkeeping_with_a_stubbed_fit(
    experiment_config: ExperimentConfig, feature_spec: FeatureSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tiny_config(
        experiment_config,
        tcn=dataclasses.replace(experiment_config.training.tcn, channels=(8, 16), dropout=(0.1,)),
        gru=dataclasses.replace(experiment_config.training.gru, hidden=(8, 16), dropout=(0.1,)),
        seeds=(0, 1, 2),
    )
    data = make_data(feature_spec, FOUR_FOLDS)
    stub = StubFit(feature_spec, cfg.training)
    monkeypatch.setattr(train, "fit", stub)
    preds, records, timings = train.cross_validate(data, cfg, ARM, KINDS, "train-alert_baseline")
    assert set(timings) == {"inner_selection_s", "refit_and_test_s"}

    folds = {fold: tuple(s for s in FOUR_FOLDS if official_fold(s) == fold) for fold in (1, 2, 3, 4)}
    inner_calls = [call for call in stub.calls if call["val"] is not None]
    refit_calls = [call for call in stub.calls if call["val"] is None]
    # 안쪽 선택: 바깥 fold 4 × 후보 4 × 안쪽 분할 3. 다시 학습: 바깥 fold 4 × 고른 후보 2 × 시드 3.
    assert len(inner_calls) == 4 * 4 * 3 and len(refit_calls) == 4 * 2 * 3
    # 안쪽 선택이 모두 끝난 뒤에 다시 학습이 시작된다. 안쪽 학습의 시드는 첫 시드 하나다.
    assert all(call["val"] is not None for call in stub.calls[:48])
    assert {call["seed"] for call in inner_calls} == {0}
    assert {call["extra"] for call in stub.calls} == {()}  # 학습 전용 출처가 없는 표

    for position, test_fold in enumerate((1, 2, 3, 4)):
        test_subjects = set(folds[test_fold])
        fold_inner = inner_calls[position * 12 : (position + 1) * 12]
        # 후보마다 검증 fold를 하나씩 돌린다: 학습 2개 fold(4명), 검증 1개 fold(2명), 시험 fold는 어디에도 없다.
        for index, call in enumerate(fold_inner):
            val_fold = [fold for fold in (1, 2, 3, 4) if fold != test_fold][index % 3]
            assert call["val"] == folds[val_fold]
            assert set(call["train"]) == set(FOUR_FOLDS) - test_subjects - set(folds[val_fold])
            assert call["fixed_units"] is None and call["temperature"] is None
        assert [call["name"] for call in fold_inner[::3]] == list(StubFit.SCRIPT)

        record = records[position]
        assert record["fold"] == test_fold and record["test_subjects"] == list(folds[test_fold])
        assert record["train_subjects"] == sorted(set(FOUR_FOLDS) - test_subjects)
        assert record["inner_val_folds"] == [fold for fold in (1, 2, 3, 4) if fold != test_fold]
        # 동률 규칙: TCN은 16채널(8채널은 표준편차 밖), GRU는 은닉 8(차이 0.01이 표준편차 0.02 안이라 작은 쪽),
        # 주 결과 행은 전체에서 가장 높은 GRU 16의 표준편차 안에 드는 후보 중 가장 작은 GRU 8이다.
        assert record["chosen"] == {
            "tcn": "tcn(channels=16,dropout=0.1)",
            "gru": "gru(hidden=8,dropout=0.1)",
            "selected": "gru(hidden=8,dropout=0.1)",
        }
        by_name = {item["name"]: item for item in record["candidates"]}
        assert by_name["tcn(channels=16,dropout=0.1)"]["mean_macro_f1"] == pytest.approx(0.76)
        assert by_name["tcn(channels=16,dropout=0.1)"]["std_macro_f1"] == pytest.approx(0.02)
        assert by_name["tcn(channels=16,dropout=0.1)"]["units"] == 4  # 3, 5, 4의 중앙값
        assert by_name["gru(hidden=8,dropout=0.1)"]["fold_units"] == [2, 2, 7]

        # 다시 학습: 학습 fold 전체(6명), 고른 학습 길이, 안쪽 검증 로짓을 모아 맞춘 온도, 시드 3개.
        fold_refit = refit_calls[position * 6 : (position + 1) * 6]
        assert [call["name"] for call in fold_refit] == ["tcn(channels=16,dropout=0.1)"] * 3 + [
            "gru(hidden=8,dropout=0.1)"
        ] * 3
        assert [call["seed"] for call in fold_refit] == [0, 1, 2, 0, 1, 2]
        for call in fold_refit:
            assert set(call["train"]) == set(FOUR_FOLDS) - test_subjects
            inner = [item for item in fold_inner if item["name"] == call["name"]]
            expected_units = {"tcn(channels=16,dropout=0.1)": 4, "gru(hidden=8,dropout=0.1)": 2}[call["name"]]
            assert call["fixed_units"] == by_name[call["name"]]["units"] == expected_units
            pooled = train.fit_temperature(
                np.concatenate([item["val_logits"] for item in inner]),
                np.concatenate([item["val_labels"] for item in inner]),
            )
            assert call["temperature"] == pooled == by_name[call["name"]]["temperature"]

    # 예측 표: 계열별 행과 주 결과 행, 각각 학습 조건의 특징과 앱 조건(자기 세션 기준선)의 특징.
    preds = evaluate.validate_predictions(preds)
    groups = preds.groupby(["run_id", "family", "arm"]).size()
    n_windows = len(make_windows(data.tables[ARM].steps, 120, 1))
    assert n_windows == len(FOUR_FOLDS) * 3 * 4
    assert groups.to_dict() == {
        ("train-alert_baseline", family, arm): 3 * n_windows
        for family in ("tcn", "gru", "selected")
        for arm in (ARM, APP_ARM)
    }
    assert sorted(preds["seed"].unique()) == [0, 1, 2]
    # 시험 fold의 피험자만, 자기 fold에서만 평가된다.
    assert (preds["fold"] == preds["subject"].map(official_fold)).all()
    keys = ["seed", "subject", "video_id", "k"]
    for arm in (ARM, APP_ARM):
        gru = preds[(preds["family"] == "gru") & (preds["arm"] == arm)].sort_values(keys)
        selected = preds[(preds["family"] == "selected") & (preds["arm"] == arm)].sort_values(keys)
        tcn = preds[(preds["family"] == "tcn") & (preds["arm"] == arm)].sort_values(keys)
        columns = list(evaluate.PROB_COLUMNS) + ["temperature"]
        # 주 결과 행은 고른 계열(GRU)의 예측 그대로다.
        assert np.array_equal(selected[columns].to_numpy(), gru[columns].to_numpy())
        probs = list(evaluate.PROB_COLUMNS)
        assert not np.allclose(selected[probs].to_numpy(), tcn[probs].to_numpy())
    # 앱 조건 평가는 같은 모델을 다른 특징에 돌린 것이다(재학습 없음: 호출 수는 위에서 셌다). perclos도 그 특징의 값이다.
    native = preds[(preds["family"] == "gru") & (preds["arm"] == ARM)].sort_values(keys)
    app = preds[(preds["family"] == "gru") & (preds["arm"] == APP_ARM)].sort_values(keys)
    assert np.array_equal(native["temperature"].to_numpy(), app["temperature"].to_numpy())
    assert not np.allclose(native["p_alert"].to_numpy(), app["p_alert"].to_numpy())
    assert not np.allclose(native["perclos_60"].to_numpy(), app["perclos_60"].to_numpy())


def test_nested_cv_other_arms_have_no_app_condition_rows(
    experiment_config: ExperimentConfig, feature_spec: FeatureSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tiny_config(experiment_config, seeds=(0,))
    data = make_data(feature_spec, THREE_FOLDS)
    monkeypatch.setattr(train, "fit", StubFit(feature_spec, cfg.training))
    # 자기 세션 기준선으로 학습하면 앱 조건이 곧 학습 조건이다. 한 계열만 주면 주 결과 행은 그 계열이다.
    preds, records, _ = train.cross_validate(data, cfg, APP_ARM, ["gru"], "train-session_baseline")
    assert preds.groupby(["family", "arm"]).size().index.tolist() == [("gru", APP_ARM), ("selected", APP_ARM)]
    assert all(set(record["chosen"]) == {"gru", "selected"} for record in records)
    # alert 영상 기준선으로 학습했는데 자기 세션 기준선 스텝 표가 없으면 앱 조건 행이 없다.
    without_app = make_data(feature_spec, THREE_FOLDS, with_app=False)
    preds, _, _ = train.cross_validate(without_app, cfg, ARM, ["gru"], "r")
    assert set(preds["arm"]) == {ARM}
    with pytest.raises(ValueError, match="no step table"):
        train.cross_validate(without_app, cfg, "global", ["gru"], "r")
    with pytest.raises(ValueError, match="at least one model family"):
        train.cross_validate(data, cfg, ARM, [], "r")


def test_supplementary_variant_labels_every_group_and_skips_the_other_supplementary_evaluations(
    experiment_config: ExperimentConfig, feature_spec: FeatureSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tiny_config(experiment_config, seeds=(0,))
    data = make_data(feature_spec, THREE_FOLDS, with_extra=True, with_frame_drop=True)
    stub = StubFit(feature_spec, cfg.training)
    monkeypatch.setattr(train, "fit", stub)
    notes: dict[str, Any] = {}
    preds, records, timings = train.cross_validate(
        data, cfg, ARM, KINDS, "train-alert_baseline-ordinal", variant="ordinal", fold_int8=True, notes=notes
    )
    # 선택 사항은 모든 학습 호출(안쪽 선택과 다시 학습)에 전달되고, 학습 전용 출처는 주 실행과 같다.
    assert {call["variant"] for call in stub.calls} == {"ordinal"}
    assert {tuple(name for name, _, _ in call["extra"]) for call in stub.calls} == {("fps15", "frame_drop_v0")}
    # 모든 묶음이 그 표지를 단 부가 묶음이다. 15 fps 평가와 INT8 비교는 하지 않는다.
    assert preds["supplementary"].all() and set(preds["variant"]) == {"ordinal"}
    assert set(preds["feature_source"]) == {"native"} and set(preds["family"]) == {"tcn", "gru", "selected"}
    assert set(preds["arm"]) == {ARM, APP_ARM} and set(timings) == {"inner_selection_s", "refit_and_test_s"}
    assert notes["variant"] == "ordinal" and notes["fold_int8"]["status"] == "not requested"
    assert notes["fps15_evaluation"] == {"evaluated_arms": [], "skipped": {}}
    assert notes["train_sources"] == {"native": 0.5, "fps15": 0.25, "frame_drop_v0": 0.25}
    assert all(len(record["refits"]) == 2 for record in records)
    with pytest.raises(ValueError, match="variant must be one of"):
        train.cross_validate(data, cfg, ARM, KINDS, "r", variant="focal")

    # 주 실행: 15 fps 추출이 시험 fold 하나를 통째로 빠뜨리면 그 평가는 건너뛴다(평가하지 않은 fold를 오답으로 세지 않는다).
    extra_steps = data.extra_tables[ARM].steps
    partial_steps = extra_steps[extra_steps["subject"].isin([1, 2, 13, 14])].reset_index(drop=True)
    partial = dataclasses.replace(
        data, extra_tables={ARM: StepTable(ARM, feature_spec.pipeline_hash, partial_steps, make_quality(partial_steps))}
    )
    notes = {}
    preds, _, timings = train.cross_validate(partial, cfg, ARM, ["gru"], "train-alert_baseline", notes=notes)
    assert set(preds["feature_source"]) == {"native"} and not preds["supplementary"].any()
    assert notes["fps15_evaluation"]["evaluated_arms"] == []
    assert notes["fps15_evaluation"]["skipped"] == {ARM: "the 15 fps extraction has no windows for test fold(s) [3]"}
    assert "fps15_eval_s" not in timings and notes["fold_int8"]["status"] == "not requested"
    # 추출이 모든 시험 fold를 덮으면 평가한다.
    notes = {}
    preds, _, timings = train.cross_validate(data, cfg, ARM, ["gru"], "train-alert_baseline", notes=notes)
    assert notes["fps15_evaluation"] == {"evaluated_arms": [ARM], "skipped": {}} and "fps15_eval_s" in timings
    assert set(preds.loc[preds["supplementary"], "feature_source"]) == {"fps15"}


def test_failed_int8_step_is_recorded_and_does_not_lose_the_main_groups(
    experiment_config: ExperimentConfig,
    feature_spec: FeatureSpec,
    feature_spec_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = tiny_config(experiment_config, seeds=(0,), max_units=1, patience_units=1)
    cfg = dataclasses.replace(cfg, paths=dataclasses.replace(cfg.paths, feature_spec=feature_spec_path))
    data = make_data(feature_spec, THREE_FOLDS, with_app=False)

    def broken_quantize(onnx_path: Path, out_path: Path) -> Path:
        raise ContractError("quantizer produced an operator outside the INT8 list")

    monkeypatch.setattr(train.export_onnx, "quantize_int8", broken_quantize)
    notes: dict[str, Any] = {}
    preds, records, timings = train.cross_validate(data, cfg, ARM, ["tcn"], "r", fold_int8=True, notes=notes)
    int8 = notes["fold_int8"]
    assert int8["status"] == "failed" and int8["models"] == 0
    assert "fold 1, seed 0: ContractError: quantizer produced an operator outside the INT8 list" == int8["error"]
    assert int8["float_onnx_vs_pytorch_max_abs_diff"] is None and "fold_int8_s" in timings
    # ONNX 묶음은 하나도 남지 않고(일부 fold만 있는 묶음을 싣지 않는다) 주 결과 묶음은 모든 fold에 있다.
    assert set(preds["family"]) == {"tcn", "selected"} and not preds["supplementary"].any()
    assert sorted(preds["fold"].unique()) == [1, 2, 3] and len(records) == 3
    monkeypatch.undo()

    # 고장 없이는 두 묶음이 모든 fold에 실린다. GRU만 돌리면 INT8 비교를 하지 않는다.
    notes = {}
    preds, _, _ = train.cross_validate(data, cfg, ARM, ["tcn"], "r", fold_int8=True, notes=notes)
    assert notes["fold_int8"]["status"] == "ok" and notes["fold_int8"]["models"] == 3
    onnx_rows = preds[preds["family"].isin(["tcn_onnx", "tcn_onnx_int8"])]
    assert onnx_rows["supplementary"].all() and set(onnx_rows["family"]) == {"tcn_onnx", "tcn_onnx_int8"}
    assert sorted(onnx_rows["fold"].unique()) == [1, 2, 3]
    assert evaluate.supplementary_pairs(preds) == [(("r", "tcn_onnx_int8", ARM), ("r", "tcn_onnx", ARM))]
    notes = {}
    train.cross_validate(data, cfg, ARM, ["gru"], "r", fold_int8=True, notes=notes)
    assert notes["fold_int8"]["status"] == "not requested"


# ---------------------------------------------------------------------------
# 시험 fold가 선택, 조기 종료, 표준화, 온도에 들어가지 않는다 (실제 fit, 피험자 번호로 확인)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def traced_cv(experiment_config: ExperimentConfig, feature_spec: FeatureSpec) -> SimpleNamespace:
    """작은 표에서 실제 학습으로 중첩 교차검증을 돌리고, 함수마다 어떤 피험자가 들어왔는지 기록한다."""
    cfg = tiny_config(experiment_config)
    data = make_data(feature_spec, THREE_FOLDS, with_extra=True, with_frame_drop=True)
    log: dict[str, list[Any]] = {"fit": [], "stats": [], "shares": [], "pooled": [], "predict": [], "scorer": []}
    patch = pytest.MonkeyPatch()
    real: dict[str, Callable[..., Any]] = {
        name: getattr(train, name) for name in ("fit", "standardization_stats", "pooled_temperature", "predict_proba")
    }
    real_scorer_init = train.VideoMacroF1.__init__

    def spy_fit(train_ws: WindowSet, val_ws: Any, config: Any, seed: int, extra_train: Any = None) -> Any:
        entry = {
            "train": set(train.window_subjects(train_ws)),
            "val": set(train.window_subjects(val_ws)) if val_ws is not None else None,
            "extra": {item.name: set(train.window_subjects(item.windows)) for item in (extra_train or ())},
            "shares": {item.name: item.share for item in (extra_train or ())},
            "seed": seed,
        }
        log["fit"].append(entry)
        result = real["fit"](train_ws, val_ws, config, seed, extra_train=extra_train)
        entry["units"] = result.units
        return result

    def spy_stats(windows: Any, spec: FeatureSpec, extra_shares: Any = None) -> Any:
        sets = [windows] if isinstance(windows, WindowSet) else list(windows)
        log["stats"].append({s for item in sets for s in train.window_subjects(item)})
        log["shares"].append(None if extra_shares is None else list(extra_shares))
        return real["standardization_stats"](windows, spec, extra_shares)

    def spy_pooled(results: Any) -> float:
        log["pooled"].append([(set(r.train_subjects), set(r.val_subjects), len(r.val_labels)) for r in results])
        return real["pooled_temperature"](results)

    def spy_predict(result: Any, windows: WindowSet, device: Any = "cpu") -> Any:
        log["predict"].append((set(result.train_subjects), set(train.window_subjects(windows))))
        log.setdefault("predict_windows", []).append(windows)
        return real["predict_proba"](result, windows, device)

    def spy_scorer_init(self: Any, meta: pd.DataFrame) -> None:
        log["scorer"].append({int(value) for value in meta["subject"].unique()})
        real_scorer_init(self, meta)

    patch.setattr(train, "fit", spy_fit)
    patch.setattr(train, "standardization_stats", spy_stats)
    patch.setattr(train, "pooled_temperature", spy_pooled)
    patch.setattr(train, "predict_proba", spy_predict)
    patch.setattr(train.VideoMacroF1, "__init__", spy_scorer_init)
    try:
        preds, records, _ = train.cross_validate(data, cfg, ARM, KINDS, "train-alert_baseline")
    finally:
        patch.undo()
    return SimpleNamespace(cfg=cfg, data=data, log=log, preds=preds, records=records)


def test_test_fold_never_reaches_fit_standardization_early_stopping_or_temperature(traced_cv: SimpleNamespace) -> None:
    log = traced_cv.log
    everyone = set(THREE_FOLDS)
    folds = {fold: {s for s in THREE_FOLDS if official_fold(s) == fold} for fold in (1, 2, 3)}

    def held_out(used: set[int]) -> int:
        """쓰인 피험자들이 통째로 빼놓은 fold. 정확히 한 fold가 통째로 빠져 있어야 한다."""
        missing = everyone - used
        matches = [fold for fold, subjects in folds.items() if subjects == missing]
        assert len(matches) == 1, f"used subjects {sorted(used)} do not leave out exactly one fold"
        return matches[0]

    # 모든 학습 호출(안쪽 선택과 다시 학습)은 fold 하나를 통째로 빼놓는다. 그 fold가 그 호출의 시험 fold다.
    inner = [entry for entry in log["fit"] if entry["val"] is not None]
    refit = [entry for entry in log["fit"] if entry["val"] is None]
    assert len(inner) == 3 * 2 * 2 and len(refit) == 3 * 2 * 2  # fold 3 × 후보 2 × (안쪽 분할 2 | 시드 2)
    for entry in inner:
        assert not entry["train"] & entry["val"]
        test_fold = held_out(entry["train"] | entry["val"])
        # 학습 전용 출처(15 fps 누락 모사, 프레임 누락 변형)도 같은 학습 피험자의 것만 들어간다.
        assert entry["extra"] == {"fps15": entry["train"], "frame_drop_v0": entry["train"]}
        assert not set().union(*entry["extra"].values()) & (folds[test_fold] | entry["val"])
        assert entry["shares"] == {"fps15": 0.25, "frame_drop_v0": 0.25}
    for entry in refit:
        test_fold = held_out(entry["train"])
        assert entry["extra"] == {"fps15": entry["train"], "frame_drop_v0": entry["train"]}
        assert not set().union(*entry["extra"].values()) & folds[test_fold]
    assert sorted(held_out(e["train"] | e["val"]) for e in inner) == [1] * 4 + [2] * 4 + [3] * 4
    assert sorted(held_out(e["train"]) for e in refit) == [1] * 4 + [2] * 4 + [3] * 4

    # 표준화 상수: 호출마다 그 학습의 학습 피험자(검증·시험 피험자 없이)만 봤다.
    assert log["stats"] == [entry["train"] for entry in log["fit"]]
    assert log["shares"] == [[0.25, 0.25]] * len(log["fit"])  # 표집과 같은 몫으로 계산한다
    # 조기 종료의 지표는 그 학습의 검증 피험자만 봤다.
    assert log["scorer"] == [entry["val"] for entry in inner]
    # 온도: 후보마다 안쪽 검증 로짓을 모은다. 모인 검증 피험자는 그 바깥 fold의 학습 피험자 전부이고 시험 fold는 없다.
    assert len(log["pooled"]) == 3 * 2
    for results in log["pooled"]:
        validated = set().union(*(val for _, val, _ in results))
        test_fold = held_out(validated)
        for train_subjects, val_subjects, _ in results:
            assert not (train_subjects | val_subjects) & folds[test_fold]
    # 평가: 다시 학습한 모델이 빼놓았던 바로 그 fold만, 학습 조건과 앱 조건 특징으로 한 번씩, 그리고 학습 조건의
    # 15 fps 추출 특징으로 한 번(부가. 손으로 만든 표에는 그 추출이 학습 조건에만 있다).
    assert len(log["predict"]) == len(refit) * 3
    for train_subjects, evaluated in log["predict"]:
        assert evaluated == folds[held_out(train_subjects)]
    # 프레임 누락 변형의 창은 평가에 쓰이지 않는다(학습 전용).
    drop_windows = make_windows(traced_cv.data.frame_drop_tables[ARM][0].steps, 120, 1)
    for windows in log["predict_windows"]:
        sampled = drop_windows.x[:: len(drop_windows) // 9]
        assert not any(np.array_equal(windows.x[0], candidate) for candidate in sampled)

    # 다시 학습의 길이는 안쪽 분할이 고른 길이의 중앙값이다.
    for record in traced_cv.records:
        chosen = {item["name"]: item for item in record["candidates"]}
        for item in chosen.values():
            assert item["units"] == train.median_half_up(item["fold_units"])
            assert train.TEMPERATURE_MIN <= item["temperature"] <= train.TEMPERATURE_MAX


def test_short_training_run_gives_a_prediction_table_in_the_evaluate_schema(
    traced_cv: SimpleNamespace, rules_config: Any
) -> None:
    preds = traced_cv.preds
    assert list(preds.columns) == list(evaluate.PREDICTION_COLUMNS)
    validated = evaluate.validate_predictions(preds)
    assert {name: str(dtype) for name, dtype in validated.dtypes.items()} == evaluate.PREDICTION_DTYPES
    assert set(preds["family"]) == {"tcn", "gru", "selected"} and set(preds["arm"]) == {ARM, APP_ARM}
    assert sorted(preds["seed"].unique()) == [0, 1] and sorted(preds["fold"].unique()) == [1, 2, 3]
    # 15 fps 평가 묶음은 부가 묶음이고 나머지는 주 결과 묶음이다(PLAN 11장 E3).
    labels = preds[["arm", "feature_source", "variant", "supplementary"]].drop_duplicates()
    assert sorted(labels.itertuples(index=False, name=None)) == [
        (ARM, "fps15", "base", True), (ARM, "native", "base", False), (APP_ARM, "native", "base", False),
    ]  # fmt: skip
    fps = preds[preds["feature_source"] == "fps15"]
    native = preds[(preds["feature_source"] == "native") & (preds["arm"] == ARM)]
    assert len(fps) == len(native) and set(fps["family"]) == {"tcn", "gru", "selected"}
    # 15 fps 묶음의 perclos_60은 그 추출의 스텝 표에서 온 값이다(원본 추출의 값이 아니다).
    keys = ["family", "seed", "subject", "video_id", "k"]
    assert not np.allclose(
        fps.sort_values(keys)["perclos_60"].to_numpy(), native.sort_values(keys)["perclos_60"].to_numpy()
    )
    assert evaluate.supplementary_pairs(preds) == [
        (("train-alert_baseline", family, ARM, "fps15", "base"), ("train-alert_baseline", family, ARM))
        for family in ("gru", "selected", "tcn")
    ]
    preds = preds[~preds["supplementary"]]
    # 온도는 그 fold에서 고른 값이 그대로 실려 있다.
    for record in traced_cv.records:
        name = record["chosen"]["selected"]
        temperature = next(item["temperature"] for item in record["candidates"] if item["name"] == name)
        rows = preds[(preds["family"] == "selected") & (preds["fold"] == record["fold"])]
        assert (rows["temperature"] == temperature).all()

    # 평가 모듈이 그대로 받는다.
    videos = evaluate.video_table(traced_cv.data.tables[ARM].quality)
    group = preds[(preds["family"] == "selected") & (preds["arm"] == APP_ARM)]
    rules = dataclasses.replace(rules_config, clock=traced_cv.data.feature_spec.pipeline.clock)
    result = evaluate.evaluate_group(group, videos, rules, traced_cv.cfg.evaluation, bootstrap=False)
    assert result.seeds == (0, 1) and result.folds == (1, 2, 3) and result.n_videos == 18
    assert 0.0 <= result.value("window_macro_f1", "fold_mean") <= 1.0

    # 선택 기록: 다시 학습한 모델마다 표준화 상수, 온도, 가중치 해시, 출처별 몫이 적혀 있다.
    for record in traced_cv.records:
        chosen = set(record["chosen"].values())
        assert len(record["refits"]) == len(chosen) * 2 and {item["name"] for item in record["refits"]} == chosen
        for item in record["refits"]:
            assert item["seed"] in (0, 1) and len(item["mean"]) == len(item["std"]) == N_FEATURES
            assert re.fullmatch(r"[0-9a-f]{64}", item["weights_sha256"])
            assert item["sources"] == [["native", 0.5], ["fps15", 0.25], ["frame_drop_v0", 0.25]]
            candidate = next(c for c in record["candidates"] if c["name"] == item["name"])
            assert (item["units"], item["temperature"]) == (candidate["units"], candidate["temperature"])
            assert set(item["labels"]) == {label for label, name in record["chosen"].items() if name == item["name"]}
        # 시드가 다르면 가중치가 다르다(해시가 가중치를 본다).
        assert len({item["weights_sha256"] for item in record["refits"]}) == len(record["refits"])


def test_injected_subject_overlap_stops_the_run_before_any_training(
    experiment_config: ExperimentConfig, feature_spec: FeatureSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tiny_config(experiment_config)
    data = make_data(feature_spec, THREE_FOLDS)
    calls: list[Any] = []
    monkeypatch.setattr(train, "fit", lambda *args, **kwargs: calls.append(args))

    # (1) 한 피험자가 두 fold에 있는 목록.
    manifest = data.manifest.copy()
    manifest.loc[manifest["video_id"] == "S01_5", "fold"] = 2
    with pytest.raises(SubjectOverlapError, match="subject 1"):
        train.cross_validate(dataclasses.replace(data, manifest=manifest), cfg, ARM, KINDS, "r")
    with pytest.raises(SubjectOverlapError):
        train.final_splits(manifest)
    with pytest.raises(SubjectOverlapError):
        train.fold_subjects(manifest)

    # (2) 시험 피험자가 학습 집합에 섞인 분할.
    def leaky_splits(_: pd.DataFrame) -> Any:
        for split in nested_splits(data.manifest):
            yield dataclasses.replace(split, train_subjects=split.train_subjects + split.test_subjects[:1])

    monkeypatch.setattr(train, "nested_splits", leaky_splits)
    with pytest.raises(SubjectOverlapError):
        train.cross_validate(data, cfg, ARM, KINDS, "r")
    monkeypatch.setattr(train, "nested_splits", nested_splits)

    # (3) 검증 피험자가 안쪽 학습 집합에 섞인 분할.
    windows = make_windows(data.tables[ARM].steps, 120, 1)
    leaky_inner = InnerSplit(val_fold=2, train_subjects=(1, 2, 13), val_subjects=(13, 14))
    candidate = train.candidates(cfg.training)[0]
    with pytest.raises(SubjectOverlapError, match="subject 13"):
        train.score_candidate(
            candidate, windows, windows, None, [leaky_inner], train._base_train_config(cfg, feature_spec), 0
        )
    assert calls == []  # 어느 경우에도 학습은 시작되지 않았다


def test_final_splits_validate_every_fold_once(feature_spec: FeatureSpec) -> None:
    data = make_data(feature_spec, FOUR_FOLDS)
    splits = train.final_splits(data.manifest)
    assert [split.val_fold for split in splits] == [1, 2, 3, 4]
    for split in splits:
        assert split.val_subjects == tuple(s for s in FOUR_FOLDS if official_fold(s) == split.val_fold)
        assert set(split.train_subjects) == set(FOUR_FOLDS) - set(split.val_subjects)
    assert train.fold_subjects(data.manifest) == {split.val_fold: split.val_subjects for split in splits}
    one_fold = data.manifest[data.manifest["fold"] == 1]
    with pytest.raises(ValueError, match="at least 2 folds"):
        train.final_splits(one_fold)


# ---------------------------------------------------------------------------
# 학습한 모델의 내보내기
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_trained_model_export_passes_contract_and_parity(
    kind: str,
    cfg: ExperimentConfig,
    feature_spec: FeatureSpec,
    feature_spec_path: Path,
    steps: pd.DataFrame,
    tmp_path: Path,
) -> None:
    windows = make_windows(steps, 120, 1)
    config = train_config(cfg, feature_spec, kind, fixed_units=2, temperature=1.6)
    result = train.fit(windows, None, config, seed=0)
    export_cfg = dataclasses.replace(cfg.export, latency_runs=20, parity_random_inputs=20)
    trained_on = "SYNTHETIC sessions (test), arm=alert_baseline"
    record = train.export_fit(
        result, feature_spec_path, tmp_path / "out", 5, trained_on, export_cfg, real_windows=windows.x[:25]
    )

    onnx_path = tmp_path / "out" / "drowsiness_v5.onnx"
    spec_copy = tmp_path / "out" / "feature_spec_v5.json"
    assert Path(record["onnx_path"]) == onnx_path and Path(record["spec_path"]) == spec_copy
    assert exported_spec_path(onnx_path) == spec_copy
    verify_contract(onnx_path, spec_copy)

    # 사본에는 학습한 평균·표준편차·온도가 들어 있고 나머지는 기준 파일 그대로다.
    exported = load_feature_spec(spec_copy)
    assert exported.means == result.spec.means and exported.stds == result.spec.stds
    assert exported.standardization.temperature == 1.6
    assert exported.means != feature_spec.means and exported.pipeline_hash == feature_spec.pipeline_hash
    assert dataclasses.replace(
        exported, features=feature_spec.features, standardization=feature_spec.standardization
    ) == feature_spec
    assert all(std >= 0.001 for std in exported.stds)

    metadata = read_metadata(onnx_path)
    assert metadata["model_version"] == "5" and metadata["trained_on"] == trained_on
    assert metadata["feature_spec_hash"] == file_sha256(spec_copy) == record["feature_spec_hash"]
    assert metadata["pipeline_hash"] == feature_spec.pipeline_hash

    # 일치 검증: 기록에 적힌 값과, 따로 다시 잰 값 모두 허용 오차 안이다(실제 창과 무작위 입력).
    assert record["verify_contract"] == "ok"
    assert record["parity_random_inputs"] == 20 and record["parity_real_windows"] == 25
    assert record["parity_random_max_abs_diff"] < 1e-4 and record["parity_real_max_abs_diff"] < 1e-4
    wrapper = train.export_wrapper(result)
    assert parity_max_abs_diff(wrapper, onnx_path, windows.x[25:60]) < 1e-4
    assert parity_max_abs_diff(wrapper, onnx_path, random_raw_inputs(exported, 20, seed=3)) < 1e-4
    # 내보낸 모델의 출력은 교차검증이 쓰는 추론(predict_proba)과 같다.
    held_out = make_windows(make_steps((37,), seed=9), 120, 1)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    produced = np.concatenate([session.run(["probs"], {"features": held_out.x[i : i + 1]})[0] for i in range(4)])
    assert np.allclose(produced, train.predict_proba(result, held_out)[:4], atol=1e-5)
    assert 0.0 < record["latency"]["median_ms"] <= record["latency"]["p95_ms"]

    if kind == KIND_TCN:  # D16: TCN이면 INT8 파일도 만든다
        int8_path = tmp_path / "out" / "drowsiness_v5_int8.onnx"
        assert Path(record["int8"]["onnx_path"]) == int8_path and record["int8"]["inputs"] == "real windows"
        verify_contract(int8_path, spec_copy, ALLOWED_OPS_INT8)
        assert record["int8"]["max_abs_diff_vs_float"] < 0.2
        assert sorted(path.name for path in (tmp_path / "out").iterdir()) == [
            "drowsiness_v5.onnx", "drowsiness_v5_int8.onnx", "feature_spec_v5.json",
        ]  # fmt: skip
    else:
        assert record["int8"] is None
        written = sorted(path.name for path in (tmp_path / "out").iterdir())
        assert written == ["drowsiness_v5.onnx", "feature_spec_v5.json"]

    # 허용 오차를 넘으면 내보내기가 실패로 끝난다(허용 오차 0은 어떤 차이도 받지 않는다).
    strict = dataclasses.replace(export_cfg, parity_tolerance=0.0)
    with pytest.raises(RuntimeError, match="differs from PyTorch"):
        train.export_fit(result, feature_spec_path, tmp_path / "strict", 5, trained_on, strict)


def test_failed_export_leaves_the_previous_model_and_spec_copy_untouched(
    cfg: ExperimentConfig, feature_spec: FeatureSpec, feature_spec_path: Path, steps: pd.DataFrame, tmp_path: Path
) -> None:
    windows = make_windows(steps, 120, 1)
    export_cfg = dataclasses.replace(cfg.export, latency_runs=5, parity_random_inputs=5)
    out_dir = tmp_path / "out"
    first = train.fit(windows, None, train_config(cfg, feature_spec, fixed_units=1, temperature=1.6), seed=0)
    record = train.export_fit(first, feature_spec_path, out_dir, 5, "SYNTHETIC sessions (test)", export_cfg)
    before = {path.name: path.read_bytes() for path in sorted(out_dir.iterdir())}
    assert sorted(before) == ["drowsiness_v5.onnx", "feature_spec_v5.json"]
    assert hashlib.sha256(before["feature_spec_v5.json"]).hexdigest() == record["feature_spec_hash"]

    # 다른 가중치와 온도의 모델을 같은 버전으로 내보내다 실패한다(빈 trained_on은 export가 받지 않는다).
    # 학습한 사본을 제자리에 먼저 쓰던 구현은 이때 사본만 새 값으로 바뀌어, 남은 모델과 해시가 어긋났다.
    second = train.fit(windows, None, train_config(cfg, feature_spec, fixed_units=1, temperature=2.5), seed=1)
    assert second.spec.means == first.spec.means and second.temperature != first.temperature
    with pytest.raises(ValueError, match="trained_on"):
        train.export_fit(second, feature_spec_path, out_dir, 5, "", export_cfg)
    assert {path.name: path.read_bytes() for path in sorted(out_dir.iterdir())} == before
    verify_contract(out_dir / "drowsiness_v5.onnx", out_dir / "feature_spec_v5.json")
    # 실패한 내보내기는 없던 폴더도 남기지 않는다.
    with pytest.raises(ValueError, match="trained_on"):
        train.export_fit(second, feature_spec_path, tmp_path / "fresh", 5, "", export_cfg)
    assert not (tmp_path / "fresh").exists()


def test_int8_file_that_fails_the_contract_check_is_removed(
    cfg: ExperimentConfig,
    feature_spec: FeatureSpec,
    feature_spec_path: Path,
    steps: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    windows = make_windows(steps, 120, 1)
    export_cfg = dataclasses.replace(cfg.export, latency_runs=5, parity_random_inputs=5)
    result = train.fit(windows, None, train_config(cfg, feature_spec, KIND_TCN, fixed_units=1), seed=0)
    real_verify = export_onnx.verify_contract
    seen: list[str] = []

    def strict_verify(onnx_path: Path, spec_path: Path, allowed_ops: Any = export_onnx.ALLOWED_OPS) -> None:
        seen.append(Path(onnx_path).name)
        if allowed_ops is ALLOWED_OPS_INT8:
            assert Path(onnx_path).is_file()  # 양자화한 파일이 실제로 쓰인 뒤에 검사한다
            raise ContractError(f"{onnx_path}: operator not allowed (injected)")
        real_verify(onnx_path, spec_path, allowed_ops)

    monkeypatch.setattr(export_onnx, "verify_contract", strict_verify)
    with pytest.raises(ContractError, match="injected"):
        train.export_fit(result, feature_spec_path, tmp_path / "out", 5, "SYNTHETIC sessions (test)", export_cfg)
    assert seen[-1] == "drowsiness_v5_int8.onnx"
    # float 모델과 사본은 남고, 계약에 맞지 않는 INT8 파일은 남지 않는다.
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == ["drowsiness_v5.onnx", "feature_spec_v5.json"]


@pytest.fixture
def committed_repo(tmp_path: Path, feature_spec_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """커밋 하나가 있고 작업 트리가 깨끗한 임시 git 저장소. feature_spec.json 하나를 추적한다.

    사용자와 시스템의 git 설정(전역 무시 목록, 훅, 서명 등)이 결과를 바꾸지 않도록 이 테스트 동안 끈다.
    """
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    for name in [key for key in os.environ if key.startswith("GIT_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    repo = tmp_path / "repo"
    repo.mkdir()
    shutil.copyfile(feature_spec_path, repo / "feature_spec.json")
    identity = ["-c", "user.name=test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false"]
    for args in (["init", "-q"], ["add", "-A"], [*identity, "commit", "-q", "-m", "initial"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    return repo


def test_export_fit_from_a_clean_tree_is_not_stamped_dirty(
    committed_repo: Path,
    cfg: ExperimentConfig,
    feature_spec: FeatureSpec,
    steps: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = committed_repo
    done = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True)
    head = done.stdout.strip()
    assert re.fullmatch(r"[0-9a-f]{40}", head)
    real_commit = export_onnx.git_commit
    assert real_commit(repo) == head  # 깨끗한 작업 트리

    # export_fit과 export가 이 임시 저장소의 상태를 읽게 하고, 읽은 시점에 저장소에 무엇이 있었는지 적어 둔다.
    states: list[tuple[str, list[str]]] = []

    def repo_commit(repo_dir: Any = None) -> str:
        value = real_commit(repo)
        states.append((value, sorted(path.name for path in repo.iterdir() if path.name != ".git")))
        return value

    handed: list[dict[str, str]] = []
    real_export = export_onnx.export

    def spy_export(model: Any, spec_path: Path, out_path: Path, meta: Any, dynamo: bool = False) -> Path:
        handed.append(dict(meta))
        return real_export(model, spec_path, out_path, meta, dynamo)

    monkeypatch.setattr(export_onnx, "git_commit", repo_commit)
    monkeypatch.setattr(export_onnx, "export", spy_export)
    windows = make_windows(steps, 120, 1)
    result = train.fit(windows, None, train_config(cfg, feature_spec, fixed_units=1, temperature=1.3), seed=0)
    export_cfg = dataclasses.replace(cfg.export, latency_runs=5, parity_random_inputs=5)
    record = train.export_fit(result, repo / "feature_spec.json", repo / "export", 7, "x", export_cfg)

    # 새 버전의 첫 내보내기: 추적되지 않는 새 파일 둘을 쓰지만 모델에는 깨끗한 커밋이 찍힌다.
    assert record["metadata"]["commit"] == head == read_metadata(repo / "export" / "drowsiness_v7.onnx")["commit"]
    # git 상태는 어떤 파일도 쓰기 전에 한 번 읽었고(그때 저장소에는 추적되는 파일뿐이었다) 그 값을 export에 넘겼다.
    assert states == [(head, ["feature_spec.json"])]
    assert handed == [{"model_version": "7", "trained_on": "x", "commit": head}]
    # 이 시험이 순서를 실제로 가른다: 내보낸 뒤의 작업 트리는 그 두 파일 때문에 깨끗하지 않다.
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, check=True, capture_output=True, text=True)
    assert status.stdout.split() == ["??", "export/"]
    assert real_commit(repo) == head + "-dirty"
    assert sorted(path.name for path in (repo / "export").iterdir()) == ["drowsiness_v7.onnx", "feature_spec_v7.json"]


# ---------------------------------------------------------------------------
# 전 과정: 합성 세션 → DriverPipeline → 스텝 표 → 명령 (LightGBM, 중첩 교차검증, 최종 모델)
# ---------------------------------------------------------------------------

TINY_SUBJECTS = (1, 13, 25)  # 공식 fold 1, 2, 3에 한 명씩
TINY_MINUTES = 4.1  # 세션 246초: 창이 세션마다 7개


def write_tiny_dataset(root: Path, spec: PipelineSpec, drop_sim: Any = None) -> None:
    for job in synth.session_jobs(max(TINY_SUBJECTS), TINY_MINUTES, 0, drop_sim=drop_sim):
        if job.subject in TINY_SUBJECTS:
            synth.write_session(job, root, spec)


def write_tiny_config(config_path: Path, repo_root: Path, tmp: Path) -> Path:
    """기본 설정에서 경로와 축소 설정만 바꾼 설정 파일. 데이터 경로는 일부러 없는 폴더로 둔다(--data-root로 준다).

    합성 데이터로 도는 설정이므로 `rldd_root`를 직접 null로 적는다. 받은 설정의 값에 기대지 않는다(실제 데이터 경로를
    적은 설정 사본에서 만들어도 같은 결과가 나와야 한다).
    """
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["paths"].update(
        rldd_root=None,
        synth_dir=str(tmp / "nowhere"),
        feature_spec=str(repo_root / "config" / "feature_spec.json"),
        rules=str(repo_root / "config" / "rules.yaml"),
        reports_dir=str(tmp / "reports"),
        export_dir=str(tmp / "export"),
    )
    data["extraction"]["workers"] = 1
    data["training"].update(device="cpu", batch_size=16)
    data["export"].update(latency_runs=20, parity_random_inputs=10)
    data["fast"].update(
        max_units=2, patience_units=1, updates_per_unit=3, seeds=[0, 1], tcn_channels=[8], gru_hidden=[8],
        lgbm_num_leaves=[3], lgbm_min_data_in_leaf=[1], lgbm_max_rounds=5, bootstrap_n=20,
    )  # fmt: skip
    out = tmp / "config.yaml"
    out.write_text(yaml.safe_dump(data), encoding="utf-8")
    return out


@pytest.fixture(scope="module")
def tiny_run(
    tmp_path_factory: pytest.TempPathFactory, config_path: Path, repo_root: Path, pipeline_spec: PipelineSpec
) -> SimpleNamespace:
    """피험자 3명의 합성 세션(원본과 15 fps 누락 모사)으로 명령 네 개를 차례로 돌린 결과."""
    tmp = tmp_path_factory.mktemp("tiny_run")
    config = write_tiny_config(config_path, repo_root, tmp)
    synth_dir = tmp / "data" / "synth"
    write_tiny_dataset(synth_dir, pipeline_spec)
    write_tiny_dataset(tmp / "data" / "synth_fps15", pipeline_spec, drop_sim=load_config(config).extraction.drop_sim)
    common = ["--config", str(config), "--data-root", str(synth_dir), "--run-name", "tiny"]
    outputs: dict[str, str] = {}
    codes: dict[str, int] = {}

    def run(name: str, main: Callable[[list[str]], int], arguments: list[str]) -> None:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            codes[name] = main(common + arguments)
        outputs[name] = buffer.getvalue()

    run("lgbm", baseline_lgbm.main, ["cv", "--fast"])
    run("cv", train.main, ["cv", "--fast"])
    run("final", train.main, ["final", "--fast", "--out-dir", str(tmp / "out")])
    return SimpleNamespace(
        tmp=tmp, config=config, synth_dir=synth_dir, common=common, codes=codes, outputs=outputs,
        run_dir=tmp / "reports" / "tiny", out_dir=tmp / "out",
    )  # fmt: skip


def test_cli_cross_validation_end_to_end_on_synthetic_sessions(tiny_run: SimpleNamespace) -> None:
    assert tiny_run.codes == {"lgbm": 0, "cv": 0, "final": 0}
    # 합성 데이터로 만든 출력과 리포트는 첫 줄에 그 사실을 적는다.
    for name in ("lgbm", "cv", "final"):
        assert tiny_run.outputs[name].splitlines()[0].startswith("SYNTHETIC data (drowsy.synth): pipeline check only")
    run_dir = tiny_run.run_dir
    summary = (run_dir / "summary.md").read_text(encoding="utf-8").splitlines()
    assert summary[0] == "# SYNTHETIC data (drowsy.synth): pipeline check only, not a performance result"
    assert "FAST profile" in summary[2] and "training mixes in the frame-drop simulated extraction" in summary[2]
    assert "windows for 10 of 10 sessions" in summary[2] and "PARTIAL" not in summary[2]  # 25번의 drowsy 영상은 둘로 나뉨
    # 학습 표본의 출처와 몫(PLAN 5.2절 증강 (1), (2)): 원본 0.5, 15 fps 누락 모사 0.25, 프레임 누락 변형 0.25.
    assert "(share 0.25, windows for 10 of 10 sessions)" in summary[2]
    assert "training mixes in 1 frame-drop variant(s) of the native extraction" in summary[2]
    assert "dropped at a per-session rate within 0.1-0.3, seed 0, share 0.25 in total" in summary[2]
    assert "training sample shares native 0.5, fps15 0.25, frame_drop_v0 0.25" in summary[2]
    # 시험 fold 격리의 예외(E0에서 라벨 없이 정하는 상수)를 리포트가 밝힌다(PLAN 6장).
    assert train.CONSTANTS_NOTE in summary[2]
    assert "closure constants and global constants of the feature spec are fixed once at E0" in train.CONSTANTS_NOTE
    assert "성능 수치가 아니다" in summary[4]
    for name in ("selection.json", "timings.json", "run.json", "summary.json", "per_fold.json"):
        payload = json.loads((run_dir / name).read_text(encoding="utf-8"))
        assert next(iter(payload)) == "data_note" and payload["data_note"].startswith("SYNTHETIC data"), name
        assert payload["synthetic"] is True
    for name in ("per_fold.csv", "alarm.csv", "videos.csv", "exclusions.csv"):
        assert (run_dir / name).read_text(encoding="utf-8").startswith("# SYNTHETIC data"), name
    assert (run_dir / "config.yaml").read_bytes() == tiny_run.config.read_bytes()

    everything = evaluate.validate_predictions(pd.read_parquet(run_dir / "predictions.parquet"))
    assert list(everything.columns) == list(evaluate.PREDICTION_COLUMNS)
    assert set(everything["run_id"]) == {"train-alert_baseline-fast"}
    # 주 결과 묶음은 계열별 행과 selected, 학습 조건과 앱 조건의 원본 추출 특징이다. 나머지는 모두 부가 묶음이다.
    preds = everything[~everything["supplementary"]]
    assert set(preds["feature_source"]) == {"native"} and set(preds["variant"]) == {"base"}
    assert set(preds["family"]) == {"tcn", "gru", "selected"} and set(preds["arm"]) == {ARM, APP_ARM}
    assert sorted(preds["seed"].unique()) == [0, 1] and sorted(preds["fold"].unique()) == [1, 2, 3]
    assert (preds["fold"] == preds["subject"].map(official_fold)).all()
    # 논리 영상 9개가 모두 평가됐다. 평가 보폭은 1초다(세션 246초에서 창이 몇 개씩 나오고 끝 스텝이 1씩 이어진다).
    run = preds[(preds["family"] == "tcn") & (preds["arm"] == ARM) & (preds["seed"] == 0)]
    assert run["video_id"].nunique() == 9
    per_session = run.groupby(["video_id", "part"])["k"].agg(["size", "min", "max"])
    assert (per_session["size"] >= 2).all() and (per_session["size"] <= 7).all()
    assert (per_session["max"] - per_session["min"] == per_session["size"] - 1).all()

    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    assert [record["fold"] for record in selection["folds"]] == [1, 2, 3]
    for record in selection["folds"]:
        assert record["test_subjects"] == [s for s in TINY_SUBJECTS if official_fold(s) == record["fold"]]
        assert set(record["chosen"]) == {"tcn", "gru", "selected"}
        assert len(record["candidates"]) == 2 and all(len(c["fold_units"]) == 2 for c in record["candidates"])
    assert "command" in selection and selection["command"].startswith("python -m drowsy.train --config")
    timings = json.loads((run_dir / "timings.json").read_text(encoding="utf-8"))["seconds"]
    assert set(timings) == {
        "load_data_s", "inner_selection_s", "refit_and_test_s", "fps15_eval_s", "fold_int8_s", "report_s", "total_s",
    }  # fmt: skip
    for record in selection["folds"]:
        assert f"fold {record['fold']}:" in tiny_run.outputs["cv"]

    # 같은 실행 폴더의 LightGBM 묶음이 리포트에 함께 실리고 D11이 판정된다. 합성 데이터라 판정원은 perclos로 남는다.
    report = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    groups = {(group["run_id"], group["family"], group["arm"]) for group in report["groups"]}
    assert ("train-alert_baseline-fast", "lgbm", ARM) in groups and len(groups) == 7
    assert report["gate"]["candidate"] == ["train-alert_baseline-fast", "selected", APP_ARM]
    assert report["gate"]["reference"] == ["train-alert_baseline-fast", "lgbm", ARM]
    assert report["drowsiness_source"] == "perclos"
    assert {(row["a"].split(" / ")[1], row["b"].split(" / ")[1]) for row in report["paired"]} == {
        ("tcn", "gru"), ("selected", "lgbm"),
    }  # fmt: skip
    assert "D11 gate:" in tiny_run.outputs["cv"] and "drowsiness.source stays perclos" in tiny_run.outputs["cv"]

    lgbm_dir = run_dir / "lgbm"
    lgbm_summary = (lgbm_dir / "summary.md").read_text(encoding="utf-8").splitlines()
    assert lgbm_summary[0].startswith("# SYNTHETIC data")
    # 기준 LightGBM은 원본 프레임률 추출로만 학습한다는 것을 리포트 맨 앞 문단과 명령의 출력에 적는다.
    assert "LightGBM trains on the native-rate extraction only" in lgbm_summary[2]
    assert "closure W 0.5 (feature spec)" in lgbm_summary[2] and "fixed once at E0" in lgbm_summary[2]
    assert tiny_run.outputs["lgbm"].splitlines()[2] == baseline_lgbm.NATIVE_ONLY_NOTE
    lgbm_preds = evaluate.validate_predictions(pd.read_parquet(lgbm_dir / "predictions.parquet"))
    assert set(lgbm_preds["family"]) == {"lgbm"} and set(lgbm_preds["temperature"]) == {1.0}
    # LightGBM fold 모델도 시험 fold의 15 fps 추출 특징으로 한 번 더 평가된다(부가 묶음. 학습에는 쓰지 않는다).
    lgbm_groups = lgbm_preds[["arm", "feature_source", "supplementary"]].drop_duplicates()
    assert sorted(lgbm_groups.itertuples(index=False, name=None)) == [(ARM, "fps15", True), (ARM, "native", False)]
    assert "the fold models evaluated on the 15 fps simulated extraction of the test fold" in lgbm_summary[2]
    assert "training does not use that extraction" in lgbm_summary[2]
    lgbm_report = json.loads((lgbm_dir / "summary.json").read_text(encoding="utf-8"))
    assert [group["feature_source"] for group in lgbm_report["groups"]] == ["native"]
    assert [group["feature_source"] for group in lgbm_report["supplementary"]["groups"]] == ["fps15"]
    assert len(lgbm_report["supplementary"]["paired"]) == len(evaluate.PAIRED_METRICS)
    lgbm_info = json.loads((lgbm_dir / "run.json").read_text(encoding="utf-8"))
    assert lgbm_info["fps15_evaluation"] == {ARM: {"evaluated": ["fps15"], "skipped": {}}}

    # 예측 표 파일 안에도 데이터 종류가 적혀 있다. 표의 열과 run_id만으로는 실제 데이터의 것과 같다.
    for path in (run_dir / "predictions.parquet", lgbm_dir / "predictions.parquet"):
        marker = json.loads(pq.read_schema(path).metadata[b"drowsy_predictions"])
        assert marker == {"data_note": evaluate.SYNTHETIC_NOTE, "synthetic": True}, path
    # 신경망 리포트에 함께 실리는 기준 LightGBM은 그 주 결과 묶음 하나뿐이다(부가 묶음은 읽어 오지 않는다).
    assert [g["feature_source"] for g in report["groups"] if g["family"] == "lgbm"] == ["native"]
    assert not any(g["family"] == "lgbm" for g in report["supplementary"]["groups"])


def test_cli_cross_validation_adds_the_15_fps_and_int8_groups_as_supplementary(tiny_run: SimpleNamespace) -> None:
    """같은 fold 모델의 15 fps 모사 평가와 TCN fold 모델의 float 대 INT8 비교가 부가 묶음으로 실린다."""
    run_dir = tiny_run.run_dir
    preds = evaluate.validate_predictions(pd.read_parquet(run_dir / "predictions.parquet"))
    extra = preds[preds["supplementary"]]
    groups = set(extra[["family", "arm", "feature_source", "variant"]].itertuples(index=False, name=None))
    assert groups == {
        *((family, arm, "fps15", "base") for family in ("tcn", "gru", "selected") for arm in (ARM, APP_ARM)),
        *((family, arm, "native", "base") for family in ("tcn_onnx", "tcn_onnx_int8") for arm in (ARM, APP_ARM)),
    }
    # 부가 묶음도 모든 시험 fold와 두 시드를 덮는다(다시 학습한 같은 모델을 한 번 더 평가한 것이다).
    for _, rows in extra.groupby(["family", "arm", "feature_source"]):
        assert sorted(rows["fold"].unique()) == [1, 2, 3] and sorted(rows["seed"].unique()) == [0, 1]
    assert (extra["fold"] == extra["subject"].map(official_fold)).all()

    def group(family: str, arm: str, source: str = "native") -> pd.DataFrame:
        rows = preds[(preds["family"] == family) & (preds["arm"] == arm) & (preds["feature_source"] == source)]
        return rows.sort_values(["seed", "subject", "video_id", "part", "k"]).reset_index(drop=True)

    probs = list(evaluate.PROB_COLUMNS)
    for arm in (ARM, APP_ARM):
        torch_rows, onnx_rows, int8_rows = group("tcn", arm), group("tcn_onnx", arm), group("tcn_onnx_int8", arm)
        assert len(torch_rows) == len(onnx_rows) == len(int8_rows) > 0
        # float ONNX 파일을 onnxruntime으로 돌린 확률은 PyTorch의 것과 일치 검증의 허용 오차 안이다(PLAN 7장).
        float_diff = np.abs(onnx_rows[probs].to_numpy() - torch_rows[probs].to_numpy()).max()
        assert float_diff < 1e-4
        # INT8 파일은 다른 가중치(양자화)라 확률이 조금 다르지만 같은 창, 같은 온도의 예측이다.
        drift = np.abs(int8_rows[probs].to_numpy() - onnx_rows[probs].to_numpy()).max()
        assert 0.0 < drift < 0.5 and drift > float_diff
        assert np.array_equal(int8_rows["temperature"].to_numpy(), torch_rows["temperature"].to_numpy())
        assert np.array_equal(int8_rows[["subject", "k"]].to_numpy(), torch_rows[["subject", "k"]].to_numpy())
        # 15 fps 묶음은 그 추출에서 나온 창이라 스텝 수와 perclos_60이 원본 추출의 것과 다르다.
        fps_rows = group("tcn", arm, "fps15")
        assert len(fps_rows) > 0 and set(fps_rows["video_id"]) == set(torch_rows["video_id"])
        merged = fps_rows.merge(torch_rows, on=["seed", "subject", "video_id", "part", "k"], suffixes=("_fps", ""))
        assert len(merged) > 0 and not np.allclose(merged["perclos_60_fps"], merged["perclos_60"])
    cfg = train.with_data_root(load_config(tiny_run.config), tiny_run.synth_dir)
    fps_steps = train.load_cv_data(cfg, (ARM,), fps_eval_arms=(ARM,)).extra_tables[ARM].steps
    fps_windows = make_windows(fps_steps, 120, 1)
    assert len(group("tcn", ARM, "fps15")) == 2 * len(fps_windows)  # 시드 둘

    # 리포트: 부가 묶음은 9절과 JSON의 supplementary에만 있고, 주 결과 표와 D11은 그대로다.
    report = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert all(not item["supplementary"] for item in report["groups"]) and len(report["groups"]) == 7
    extra_groups = {
        (item["family"], item["arm"], item["feature_source"]) for item in report["supplementary"]["groups"]
    }
    assert extra_groups == {(family, arm, source) for family, arm, source, _ in groups}
    assert report["gate"]["candidate"] == ["train-alert_baseline-fast", "selected", APP_ARM]
    pairs = {(row["a"], row["b"]) for row in report["supplementary"]["paired"]}
    assert len(pairs) == 6 + 2 + 2 and len(report["supplementary"]["paired"]) == 10 * len(evaluate.PAIRED_METRICS)
    names = {(a.split(" / ")[1], b.split(" / ")[1]) for a, b in pairs}
    assert names == {
        ("tcn", "tcn"), ("gru", "gru"), ("selected", "selected"),  # 15 fps 대 원본 추출
        ("tcn_onnx_int8", "tcn_onnx"), ("tcn_onnx", "tcn"),  # INT8 대 float ONNX, float ONNX 대 PyTorch
    }  # fmt: skip
    summary = (run_dir / "summary.md").read_text(encoding="utf-8")
    # 1~6절(fold별 표, 혼동 행렬, 오경보, 주 결과끼리의 짝지은 비교, D11, 비교표)에는 부가 묶음이 나오지 않는다.
    main_sections = summary.split("## 1. fold별 결과")[1].split("## 7. 제외 목록")[0]
    assert "tcn_onnx" not in main_sections and "fps15" not in main_sections
    tail = summary.split("## 9. 부가 결과")[1]
    assert "tcn_onnx_int8" in tail and "fps15" in tail
    assert "Supplementary groups (section 9 only, never used for the headline rows or D11)" in summary.splitlines()[2]
    assert "tcn_onnx_int8" in (run_dir / "supplementary_per_fold.csv").read_text(encoding="utf-8")
    assert "tcn_onnx" not in (run_dir / "per_fold.csv").read_text(encoding="utf-8")

    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert info["variant"] == "base"
    assert info["train_sources"] == {"native": 0.5, "fps15": 0.25, "frame_drop_v0": 0.25}
    notes = info["supplementary"]
    assert notes["fps15_evaluation"] == {"evaluated_arms": [ARM, APP_ARM], "skipped": {}}
    int8 = notes["fold_int8"]
    assert (int8["status"], int8["models"], int8["error"]) == ("ok", 3 * 2, None)  # fold 3 × 시드 2
    assert int8["families"] == ["tcn_onnx", "tcn_onnx_int8"]
    assert int8["float_onnx_vs_pytorch_max_abs_diff"] < 1e-4 < int8["int8_vs_float_onnx_max_abs_diff"]
    # 선택 기록의 다시 학습 항목: 표준화 상수, 온도, 가중치 해시, 출처별 몫.
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    for record in selection["folds"]:
        assert {item["name"] for item in record["refits"]} == set(record["chosen"].values())
        shares = [["native", 0.5], ["fps15", 0.25], ["frame_drop_v0", 0.25]]
        assert all(item["sources"] == shares for item in record["refits"])


def test_cli_reports_list_baselines_and_window_counts(tiny_run: SimpleNamespace) -> None:
    """PLAN 4.4절: 기준선이 연장·대체된 영상과 조건별·클래스별 창 수가 리포트에 실린다(신경망과 LightGBM 모두)."""
    run_dir = tiny_run.run_dir
    cfg = train.with_data_root(load_config(tiny_run.config), tiny_run.synth_dir)
    data = train.load_cv_data(cfg, (ARM, APP_ARM), mix_arms=(ARM,), fps_eval_arms=(ARM, APP_ARM))
    tables = {name: table.steps for name, table in data.tables.items()}
    tables[f"{ARM} [fps15]"] = data.extra_tables[ARM].steps
    tables[f"{APP_ARM} [fps15]"] = data.extra_tables[APP_ARM].steps
    tables[f"{ARM} [frame_drop_v0]"] = data.frame_drop_tables[ARM][0].steps
    assert set(tables) == {
        ARM, APP_ARM, "alert_baseline [fps15]", "session_baseline [fps15]", "alert_baseline [frame_drop_v0]",
    }  # fmt: skip
    # 프레임 누락 변형(PLAN 5.2절 증강 (2)): 설정의 비율 범위, 시드 0, 변형 번호 0. 세션마다 다른 비율이고 특징 값이 원본과 다르다.
    variant = data.frame_drop_tables[ARM][0]
    assert variant.frame_drop == FrameDropSpec(variant=0, seed=train.FRAME_DROP_SEED, drop_min=0.10, drop_max=0.30)
    assert train.FRAME_DROP_SEED == 0 and set(data.frame_drop_tables) == {ARM}
    rates = variant.quality["drop_rate"]
    assert rates.between(0.10, 0.30).all() and rates.nunique() == 10
    assert sorted(variant.steps["session_id"].unique()) == sorted(data.tables[ARM].steps["session_id"].unique())
    assert train.step_table_fingerprint(variant.steps) != train.step_table_fingerprint(data.tables[ARM].steps)

    def counted(steps: pd.DataFrame, stride: int) -> dict[str, int]:
        """창을 실제로 만들어 센다(리포트는 `dataset.windows_per_class`로 창을 만들지 않고 센다)."""
        windows = make_windows(steps, 120, stride)
        return {name: int((windows.y == index).sum()) for index, name in enumerate(CLASS_ORDER)}

    report = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert set(report["window_counts"]) == set(tables)
    for label, steps in tables.items():
        assert report["window_counts"][label] == {"5": counted(steps, 5), "1": counted(steps, 1)}, label
    native = report["window_counts"][ARM]
    assert native == report["window_counts"][APP_ARM]  # 정규화 조건은 같은 창 집합을 쓴다
    # 세션 246초: 스텝 61~245 중 조회 구간이 찬 것이 120~245(126개)라 보폭 1에서 세션마다 7개, 보폭 5에서 2개다.
    # 세션은 10개다(피험자 3명 × 영상 3개, 25번의 drowsy 영상이 두 부분으로 나뉘어 drowsy만 세션이 4개).
    assert native["1"] == {"alert": 21, "low_vigilant": 21, "drowsy": 28}
    assert native["5"] == {"alert": 6, "low_vigilant": 6, "drowsy": 8}
    summary = (run_dir / "summary.md").read_text(encoding="utf-8")
    section = summary[summary.index("## 8. 기준선과 창 수 (PLAN 4.4절)") : summary.index("## 9. 부가 결과")]
    assert "| alert_baseline [fps15] | 5 |" in section and "| alert_baseline | 1 | 21 | 21 | 28 | 70 |" in section
    assert "기준선이 연장된 영상(T_b > 60초): 0개" in section
    assert report["baselines"]["nominal_ms"] == 60_000 and report["baselines"]["extended"] == []

    videos = pd.read_csv(run_dir / "videos.csv", skiprows=1)
    assert {"baseline_status", "t_b_ms", "baseline_used", "alert_fallback"} <= set(videos.columns)
    assert len(videos) == 9 and set(videos["t_b_ms"]) == {60_000} and set(videos["baseline_used"]) == {"alert"}

    # run.json: 세션 단위의 기준선·창 요약과 스텝 표의 내용 지문.
    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert info["arm"] == ARM and info["run_id"] == "train-alert_baseline-fast" and info["fast"] is True
    assert info["step_fingerprints"] == {
        name: train.step_table_fingerprint(data.tables[name].steps) for name in (ARM, APP_ARM)
    }
    baselines = info["baselines"]
    assert (baselines["n_sessions"], baselines["n_videos"], baselines["normal_t_b_ms"]) == (10, 9, 60_000)
    assert baselines["extended_baselines"] == [] and baselines["alert_fallback_subjects"] == []
    rows = {(row["arm"], row["stride"]): row for row in baselines["window_counts"]}
    assert set(rows) == {(ARM, 5), (ARM, 1), (APP_ARM, 5), (APP_ARM, 1)}
    assert rows[(ARM, 1)]["total"] == 70 and rows[(ARM, 5)]["drowsy"] == 8

    # LightGBM 리포트도 같은 항목을 싣는다. 15 fps 추출은 학습에 섞지 않고 부가 평가에만 쓰므로 그 행이 하나 있다.
    lgbm = json.loads((run_dir / "lgbm" / "summary.json").read_text(encoding="utf-8"))
    assert lgbm["window_counts"] == {ARM: native, f"{ARM} [fps15]": report["window_counts"][f"{ARM} [fps15]"]}
    assert lgbm["baselines"]["extended"] == []
    lgbm_info = json.loads((run_dir / "lgbm" / "run.json").read_text(encoding="utf-8"))
    assert lgbm_info["closure_w"] is None and lgbm_info["effective_closure_w"] == 0.5
    assert lgbm_info["step_fingerprints"] == {ARM: info["step_fingerprints"][ARM]}
    assert lgbm_info["baselines"]["n_sessions"] == 10
    assert "## 8. 기준선과 창 수 (PLAN 4.4절)" in (run_dir / "lgbm" / "summary.md").read_text(encoding="utf-8")


def test_cli_final_model_end_to_end_on_synthetic_sessions(tiny_run: SimpleNamespace, feature_spec: FeatureSpec) -> None:
    record = json.loads((tiny_run.run_dir / "final" / "final.json").read_text(encoding="utf-8"))
    assert next(iter(record)) == "data_note" and record["data_note"].startswith("SYNTHETIC data")
    summary = (tiny_run.run_dir / "final" / "summary.md").read_text(encoding="utf-8")
    assert summary.startswith("# SYNTHETIC data (drowsy.synth): pipeline check only")
    assert record["subjects"] == list(TINY_SUBJECTS) and record["selection"]["val_folds"] == [1, 2, 3]
    assert record["final_fit"]["seed"] == 0  # 최종 모델의 시드는 0으로 고정한다
    chosen = next(c for c in record["selection"]["candidates"] if c["name"] == record["selection"]["chosen"])
    assert record["final_fit"]["units"] == chosen["units"] == train.median_half_up(chosen["fold_units"])
    assert record["final_fit"]["temperature"] == chosen["temperature"]

    onnx_path = tiny_run.out_dir / "drowsiness_v1.onnx"
    spec_copy = tiny_run.out_dir / "feature_spec_v1.json"
    assert Path(record["export"]["onnx_path"]) == onnx_path and exported_spec_path(onnx_path) == spec_copy
    verify_contract(onnx_path, spec_copy)
    metadata = read_metadata(onnx_path)
    assert metadata["model_version"] == "1"
    assert metadata["trained_on"] == record["trained_on"]
    assert metadata["trained_on"].startswith("SYNTHETIC sessions (drowsy.synth) 3 subjects, arm=alert_baseline")
    exported = load_feature_spec(spec_copy)
    assert exported.standardization.temperature == record["final_fit"]["temperature"]
    assert exported.means != feature_spec.means and exported.pipeline_hash == feature_spec.pipeline_hash
    assert record["export"]["parity_real_windows"] > 0
    assert record["export"]["parity_real_max_abs_diff"] < 1e-4 and record["export"]["parity_random_max_abs_diff"] < 1e-4

    # 체크포인트와 사본으로 다시 만든 래퍼가 내보낸 파일과 일치한다(실제 파이프라인이 만든 창과 무작위 입력).
    checkpoint = load_checkpoint(tiny_run.run_dir / "final" / "final_model.pt")
    assert checkpoint.kind == record["final_fit"]["kind"] and checkpoint.trained_on == metadata["trained_on"]
    # 체크포인트 v2: 학습이 쓴 feature_spec 사본의 해시와 합성 데이터 여부가 들어 있다.
    assert checkpoint.synthetic is True
    assert checkpoint.feature_spec_hash == file_sha256(spec_copy) == metadata["feature_spec_hash"]
    assert checkpoint.feature_spec_hash == record["export"]["feature_spec_hash"]
    wrapper = ExportWrapper(checkpoint.net, exported, checkpoint.temperature)
    assert parity_max_abs_diff(wrapper, onnx_path, random_raw_inputs(exported, 20, seed=1)) < 1e-4
    cfg = train.with_data_root(load_config(tiny_run.config), tiny_run.synth_dir)
    data = train.load_cv_data(cfg, (ARM,), mix_arms=(ARM,))
    assert set(data.extra_tables) == {ARM} and "frame-drop simulated extraction" in data.description
    assert set(data.frame_drop_tables) == {ARM} and len(data.frame_drop_tables[ARM]) == 1
    windows = make_windows(data.tables[ARM].steps, 120, 1)
    assert parity_max_abs_diff(wrapper, onnx_path, windows.x[:30]) < 1e-4
    # 사본의 평균·표준편차는 전체 피험자의 학습 창(원본, 15 fps 누락 모사, 프레임 누락 변형)에서 표집과 같은 몫으로
    # 계산한 값이다.
    extra = make_windows(data.extra_tables[ARM].steps, 120, 1)
    thinned = make_windows(data.frame_drop_tables[ARM][0].steps, 120, 1)
    mean, std = train.standardization_stats([windows, extra, thinned], feature_spec, [0.25, 0.25])
    assert exported.means == mean and exported.stds == std
    assert record["train_sources"] == {"native": 0.5, "fps15": 0.25, "frame_drop_v0": 0.25}
    # INT8 파일의 기록은 확률 차이일 뿐 지표 변화가 아니라고 적는다(전원으로 학습해 시험 fold가 없다).
    if record["export"]["int8"] is not None:
        assert record["export"]["int8"]["note"] == train.INT8_FINAL_NOTE
        assert "지표 변화가 아니다" in summary and "tcn_onnx_int8" in summary
    assert train.INT8_FINAL_NOTE.startswith("probability drift only")

    assert f"wrote {onnx_path}" in tiny_run.outputs["final"] and "verify_contract: OK" in tiny_run.outputs["final"]
    assert "trained_on: SYNTHETIC" in tiny_run.outputs["final"]
    assert record["fast"] is True and record["pipeline_hash"] == feature_spec.pipeline_hash
    assert record["step_fingerprints"] == {ARM: train.step_table_fingerprint(data.tables[ARM].steps)}


def test_final_checkpoint_carries_the_fast_flag_of_the_run(
    tiny_run: SimpleNamespace, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """체크포인트의 `fast`는 그 모델을 학습한 실행의 축소 설정 여부다(`export_onnx --checkpoint`가 이 값으로 막는다)."""
    assert load_checkpoint(tiny_run.run_dir / "final" / "final_model.pt").fast is True  # `final --fast`의 것
    # 축소 설정 없이 돌린 실행의 체크포인트는 False다. 빨리 돌도록 학습 설정 자체를 작게 적은 설정 파일을 쓴다.
    data = yaml.safe_load(tiny_run.config.read_text(encoding="utf-8"))
    data["training"].update(max_units=1, patience_units=1, updates_per_unit=2, seeds=[0])
    data["training"]["gru"].update(hidden=[8], dropout=[0.1])
    config = tmp_path / "small.yaml"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    saved: list[dict[str, Any]] = []
    real_save = train.save_checkpoint

    def spy_save(*args: Any, **kwargs: Any) -> None:
        saved.append(dict(kwargs))
        real_save(*args, **kwargs)

    monkeypatch.setattr(train, "save_checkpoint", spy_save)
    common = ["--config", str(config), "--data-root", str(tiny_run.synth_dir), "--run-name", "notfast"]
    out_dir = tmp_path / "out"
    assert train.main(common + ["final", "--families", "gru", "--out-dir", str(out_dir)]) == 0
    capsys.readouterr()
    assert saved == [{"fast": False}]
    final_dir = tiny_run.tmp / "reports" / "notfast" / "final"
    checkpoint = load_checkpoint(final_dir / "final_model.pt")
    assert checkpoint.fast is False and checkpoint.synthetic is True
    assert json.loads((final_dir / "final.json").read_text(encoding="utf-8"))["fast"] is False
    # 같은 설정을 --fast로 돌리면 True가 실린다.
    saved.clear()
    assert train.main(common + ["final", "--fast", "--families", "gru", "--out-dir", str(out_dir), "--overwrite"]) == 0
    capsys.readouterr()
    assert saved == [{"fast": True}] and load_checkpoint(final_dir / "final_model.pt").fast is True


def test_extra_source_windows_of_other_subjects_never_reach_the_model(
    cfg: ExperimentConfig, feature_spec: FeatureSpec, steps: pd.DataFrame
) -> None:
    """학습 전용 출처에 원본 학습 묶음에 없는 피험자의 창이 섞여 있어도 표본과 표준화 상수에 들어가지 않는다.

    표집기와 표준화 상수는 원본 묶음의 영상만 대상으로 삼는다. 부르는 쪽이 학습 피험자만 골라 넘기는 것과 겹치는
    두 번째 방어선이다. 그래도 그 피험자는 학습 피험자로 기록되어 평가의 겹침 검사가 그 피험자의 창을 거부한다.
    """
    subjects = (1, 2, 13, 14)
    train_windows = windows_of(steps, subjects)
    fps_steps = make_steps(THREE_FOLDS, seed=5, offset=0.03)
    drop_steps = make_steps(THREE_FOLDS, seed=6, offset=0.3)  # 25, 26번의 값은 눈에 띄게 다르다
    config = train_config(cfg, feature_spec, fixed_units=2)
    filtered = train.fit(
        train_windows, None, config, 0,
        extra_train=[source("fps15", 0.25, windows_of(fps_steps, subjects)),
                     source("frame_drop_v0", 0.25, windows_of(drop_steps, subjects))],
    )  # fmt: skip
    unfiltered = train.fit(
        train_windows, None, config, 0,
        extra_train=[source("fps15", 0.25, windows_of(fps_steps, THREE_FOLDS)),
                     source("frame_drop_v0", 0.25, windows_of(drop_steps, THREE_FOLDS))],
    )  # fmt: skip
    assert states_equal(filtered.net, unfiltered.net) and filtered.spec == unfiltered.spec
    assert filtered.train_subjects == subjects and unfiltered.train_subjects == THREE_FOLDS
    with pytest.raises(SubjectOverlapError, match="subject 25"):
        train.predict_proba(unfiltered, windows_of(steps, (25,)))
    # 대조: 원본 묶음에 그 피험자가 들어 있으면 값이 달라진다(위의 같음이 "아무것도 보지 않아서"가 아니다).
    leaked = train.fit(
        windows_of(steps, THREE_FOLDS), None, config, 0,
        extra_train=[source("fps15", 0.25, windows_of(fps_steps, THREE_FOLDS)),
                     source("frame_drop_v0", 0.25, windows_of(drop_steps, THREE_FOLDS))],
    )  # fmt: skip
    assert leaked.spec.means != filtered.spec.means and not states_equal(leaked.net, filtered.net)


def test_cli_final_writes_rules_json_next_to_the_model(
    tiny_run: SimpleNamespace, repo_root: Path, rules_config: Any, feature_spec: FeatureSpec
) -> None:
    """PLAN 11장 E4: 내보낸 모델 옆에 앱이 읽는 `rules.json`이 있고 그 해시가 기록에 있다."""
    record = json.loads((tiny_run.run_dir / "final" / "final.json").read_text(encoding="utf-8"))
    rules_path = tiny_run.out_dir / "rules.json"
    assert Path(record["rules_path"]) == rules_path and rules_path.is_file()
    # 저장소의 규칙 설정에서 만든 파일이다: 앱 팀에 넘긴 export/rules.json과 바이트 단위로 같고 해시가 9c2d05f2...다.
    assert rules_path.read_bytes() == (repo_root / "export" / "rules.json").read_bytes()
    assert record["rules_hash"] == file_sha256(rules_path)
    assert record["rules_hash"] == "9c2d05f29c4c4339d4759425f996c38137677d7216ad73c27091e75fb6cec7b4"
    assert load_rules(rules_path) == dataclasses.replace(rules_config, clock=feature_spec.pipeline.clock)
    summary = (tiny_run.run_dir / "final" / "summary.md").read_text(encoding="utf-8")
    assert f"- 규칙 설정 사본: `{rules_path}` (rules_hash `{record['rules_hash']}`)" in summary
    assert f"wrote {rules_path} (rules_hash {record['rules_hash']})" in tiny_run.outputs["final"]
    # 내보낸 폴더에는 모델, 사본, 규칙 파일(과 TCN이면 INT8 파일)만 있다.
    names = sorted(path.name for path in tiny_run.out_dir.iterdir())
    expected = ["drowsiness_v1.onnx", "feature_spec_v1.json", "rules.json"]
    if record["final_fit"]["kind"] == KIND_TCN:
        expected.insert(1, "drowsiness_v1_int8.onnx")
    assert names == expected


def test_final_checkpoint_re_exports_to_the_same_bytes(
    tiny_run: SimpleNamespace, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`train final`이 쓴 체크포인트와 사본으로 `export_onnx --checkpoint`를 돌리면 같은 모델 파일이 나온다."""
    checkpoint = tiny_run.run_dir / "final" / "final_model.pt"
    spec_copy = tiny_run.out_dir / "feature_spec_v1.json"
    arguments = ["--config", str(tiny_run.config), "--checkpoint", str(checkpoint), "--version", "1"]
    out_dir = tmp_path / "again"
    assert export_onnx.main([*arguments, "--spec", str(spec_copy), "--out-dir", str(out_dir)]) == 0
    capsys.readouterr()
    assert (out_dir / "drowsiness_v1.onnx").read_bytes() == (tiny_run.out_dir / "drowsiness_v1.onnx").read_bytes()
    assert (out_dir / "feature_spec_v1.json").read_bytes() == spec_copy.read_bytes()
    # 합성 데이터로 학습한 체크포인트는 이 명령으로도 paths.export_dir 아래에 쓰지 못한다(체크포인트의 synthetic 값).
    below = tiny_run.tmp / "export" / "candidate"
    assert export_onnx.main([*arguments, "--spec", str(spec_copy), "--out-dir", str(below)]) == 2
    assert "must not be exported to paths.export_dir" in capsys.readouterr().err
    assert not (tiny_run.tmp / "export").exists()


def test_cli_refuses_synthetic_export_dir_and_missing_data(
    tiny_run: SimpleNamespace, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fits: list[Any] = []
    monkeypatch.setattr(train, "fit", lambda *args, **kwargs: fits.append(args))
    below = str(tiny_run.tmp / "export" / "candidate")
    # 합성 데이터로 학습한 모델은 paths.export_dir과 그 아래 폴더로 내보내지 않는다.
    for arguments in (["final"], ["final", "--out-dir", below], ["final", "--out-dir", below + "/../candidate/v1"]):
        assert train.main(tiny_run.common + arguments) == 2, arguments
        captured = capsys.readouterr()
        assert "a model trained on synthetic data must not be exported to paths.export_dir" in captured.err
        assert "or a folder below it" in captured.err
        assert captured.out.splitlines()[0].startswith("SYNTHETIC data")
    # 축소 설정(--fast)으로 학습한 모델도 같은 폴더들로 내보내지 않는다. 데이터 종류와 무관하게 막는다.
    for arguments in (["final", "--fast"], ["final", "--fast", "--out-dir", below]):
        assert train.main(tiny_run.common + arguments) == 2, arguments
        captured = capsys.readouterr()
        assert "trained with the fast profile (smoke run) must not be exported to paths.export_dir" in captured.err
    real = dataclasses.replace(load_config(tiny_run.config), paths=dataclasses.replace(
        load_config(tiny_run.config).paths, rldd_root=tiny_run.tmp / "no_such_rldd"
    ))  # fmt: skip
    with pytest.raises(ValueError, match="fast profile"):  # 데이터를 읽기 전에 멈춘다(없는 폴더를 가리켜도 같은 오류)
        train.fit_final(real, fast=True, out_dir=None)
    # 같은 실행 폴더의 final/에는 축소 설정으로 만든 기록과 체크포인트가 있다. 다른 설정의 실행은 그것을 덮어쓰지
    # 않고 멈춘다(실제 데이터의 E4 기록을 연기 시험이 덮어쓰는 일을 막는 장치).
    kept = {name: (tiny_run.run_dir / "final" / name).read_bytes() for name in ("final.json", "final_model.pt")}
    assert train.main(tiny_run.common + ["final", "--out-dir", str(tiny_run.tmp / "elsewhere")]) == 2
    error = capsys.readouterr().err
    assert "already holds the report of another run (fast: True there, False now)" in error and "--overwrite" in error
    assert {name: (tiny_run.run_dir / "final" / name).read_bytes() for name in kept} == kept
    assert not (tiny_run.tmp / "elsewhere").exists()
    # 어느 경우에도 학습은 시작되지 않았고 아무것도 쓰지 않았다. 이름만 비슷한 이웃 폴더는 막지 않는다.
    assert fits == [] and not (tiny_run.tmp / "export").exists()
    assert not export_onnx.is_export_dir_or_below(tiny_run.tmp / "export2", tiny_run.tmp / "export")
    assert export_onnx.is_export_dir_or_below(tiny_run.tmp / "export" / "a" / "b", tiny_run.tmp / "export")
    assert not hasattr(train, "_inside")  # 학습 쪽에 따로 둔 사본은 없다. 두 모듈이 같은 함수로 판단한다
    monkeypatch.undo()

    # --data-root 없이는 설정의 (없는) 합성 데이터 폴더를 찾다가 멈춘다. 종료 코드 2.
    without_root = ["--config", str(tiny_run.config), "--run-name", "none"]
    assert train.main(without_root + ["cv", "--fast"]) == 2
    assert "python -m drowsy.synth" in capsys.readouterr().err
    assert not (tiny_run.tmp / "reports" / "none").exists()
    # 공통 옵션은 하위 명령 뒤에 적어도 된다.
    assert train.main(["cv", "--fast", "--config", str(tiny_run.config), "--run-name", "none"]) == 2
    capsys.readouterr()


def _relative_export_dir_setup(
    tiny_run: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> SimpleNamespace:
    """설정의 `paths.export_dir`을 기본값처럼 상대 경로 `export`로 두고, 저장소 루트가 아닌 폴더에서 실행하는 상황을 만든다.

    저장소 루트는 임시 폴더 `repo`로 바꿔 둔다(`export_onnx.REPO_ROOT`). 검사가 뚫려도 저장소의 진짜 export/에는 아무것도
    쓰이지 않는다. 진짜 루트가 그 상수의 값이라는 것은 tests/test_export_onnx.py가 따로 확인한다.
    실행 위치는 `repo/work`다. 그래서 `export`는 실행 위치 기준으로 `repo/work/export`, 저장소 루트 기준으로 `repo/export`다.
    """
    data = yaml.safe_load(tiny_run.config.read_text(encoding="utf-8"))
    data["paths"]["export_dir"] = "export"
    config = tmp_path / "relative_export.yaml"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert load_config(config).paths.export_dir == Path("export")
    repo = tmp_path / "repo"
    work = repo / "work"
    work.mkdir(parents=True)
    monkeypatch.setattr(export_onnx, "REPO_ROOT", repo)
    monkeypatch.chdir(work)
    refused: list[str | None] = [
        None,  # --out-dir 없음: 설정의 `export`를 실행 위치 기준으로 쓴다(repo/work/export)
        "export",
        "export/",
        "export/candidate",
        "sub/../export/candidate",
        str(work / "export"),
        str(repo / "export"),  # 저장소 루트의 export/를 절대 경로로
        str(repo / "export") + "/",
        str(repo / "export" / "candidate"),
        "../export",  # 저장소 루트의 export/를 상대 경로로
        "../export/candidate",
    ]
    link = repo / "link_to_export"
    try:
        link.symlink_to(repo / "export", target_is_directory=True)
    except OSError:  # 심볼릭 링크를 만들 수 없는 파일 시스템
        pass
    else:
        refused += [str(link), "../link_to_export/v1"]
    common = ["--config", str(config), "--data-root", str(tiny_run.synth_dir)]
    return SimpleNamespace(config=config, repo=repo, work=work, refused=refused, common=common)


def test_cli_final_refuses_the_export_dir_from_another_working_directory(
    tiny_run: SimpleNamespace, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """합성 모델과 축소 설정 모델의 두 울타리는 실행 위치와 무관하다.

    전에는 상대 경로 `export`를 실행 위치 기준으로만 풀어서, 다른 폴더에서 실행하면서 `--out-dir`로 저장소의 export/를
    가리키면 두 검사를 모두 지나쳤다.
    """
    setup = _relative_export_dir_setup(tiny_run, tmp_path, monkeypatch)
    fits: list[Any] = []
    monkeypatch.setattr(train, "fit", lambda *args, **kwargs: fits.append(args))
    for fast, message in (
        (False, "a model trained on synthetic data must not be exported to paths.export_dir (export)"),
        (True, "a model trained with the fast profile (smoke run) must not be exported to paths.export_dir (export)"),
    ):
        for out_dir in setup.refused:
            arguments = ["--run-name", "fence", "final", "--families", "gru"]
            arguments += ["--fast"] if fast else []
            arguments += [] if out_dir is None else ["--out-dir", out_dir]
            assert train.main(setup.common + arguments) == 2, (fast, out_dir)
            error = capsys.readouterr().err
            assert error.startswith("error: ") and error.count("\n") == 1 and message in error, (fast, out_dir)
            assert f"got {Path('export' if out_dir is None else out_dir)};" in error  # 끝의 `/`는 Path가 떼어 낸다
    # 어느 경우에도 학습은 시작되지 않았고, 두 export 폴더와 리포트 폴더에 아무것도 쓰이지 않았다.
    assert fits == []
    assert not (setup.repo / "export").exists() and not (setup.work / "export").exists()
    assert not (tiny_run.tmp / "reports" / "fence").exists()


def test_cli_final_still_exports_elsewhere_from_another_working_directory(
    tiny_run: SimpleNamespace, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """울타리 밖의 폴더로는 다른 실행 위치에서도 그대로 나간다: 상대 경로(실행 위치 기준)와 절대 경로, 이름이 비슷한 이웃 폴더."""
    setup = _relative_export_dir_setup(tiny_run, tmp_path, monkeypatch)
    expected = ["drowsiness_v1.onnx", "feature_spec_v1.json", "rules.json"]
    for run_name, out_dir, written in (
        ("fence_relative", "out", setup.work / "out"),
        ("fence_absolute", str(setup.repo / "export_smoke"), setup.repo / "export_smoke"),
    ):
        arguments = ["--run-name", run_name, "final", "--fast", "--families", "gru", "--out-dir", out_dir]
        assert train.main(setup.common + arguments) == 0, out_dir
        capsys.readouterr()
        assert sorted(path.name for path in written.iterdir()) == expected
        verify_contract(written / "drowsiness_v1.onnx", written / "feature_spec_v1.json")
    assert not (setup.repo / "export").exists() and not (setup.work / "export").exists()
    assert sorted(path.name for path in setup.work.iterdir()) == ["out"]  # 실행 위치에 다른 것은 쓰지 않았다


@pytest.mark.parametrize("configured", ["absolute", "relative"])
def test_cli_final_refuses_the_repository_export_folder_under_another_config(
    configured: str, tiny_run: SimpleNamespace, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    """`train final`: 설정의 `paths.export_dir`이 다른 폴더(절대 경로, 다른 상대 이름)여도 저장소의 export/로는 내보내지 않는다.

    이 파일의 연기 시험 설정이 바로 그런 설정이다(`export_dir`이 임시 폴더). 전에는 그 설정으로 저장소 루트에서
    `--out-dir export`를 주면 합성 데이터 모델과 축소 설정 모델이 커밋하는 폴더에 쓰였다. 저장소 루트는 임시 폴더 `repo`로
    바꿔 둔다(`export_onnx.REPO_ROOT`). 진짜 export/에는 어떤 경우에도 쓰지 않는다.
    """
    value = str(tmp_path / "smoke_export") if configured == "absolute" else "release"
    data = yaml.safe_load(tiny_run.config.read_text(encoding="utf-8"))
    data["paths"]["export_dir"] = value
    config = tmp_path / "other_export_dir.yaml"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert load_config(config).paths.export_dir == Path(value)
    repo = tmp_path / "repo"
    work = repo / "work"
    work.mkdir(parents=True)
    monkeypatch.setattr(export_onnx, "REPO_ROOT", repo)
    shipped = repo / "export"
    always = f"the export folder of this repository ({shipped}; it is protected whatever paths.export_dir names, "
    always += f"here {value})"
    common = ["--config", str(config), "--data-root", str(tiny_run.synth_dir)]
    with monkeypatch.context() as patch:
        fits: list[Any] = []
        patch.setattr(train, "fit", lambda *args, **kwargs: fits.append(args))
        for cwd, out_dirs in (
            (repo, ["export", "export/", "export/candidate", str(shipped)]),
            (work, ["../export", str(shipped / "v1")]),
        ):
            patch.chdir(cwd)
            for fast, start in (
                (False, "a model trained on synthetic data must not be exported to "),
                (True, "a model trained with the fast profile (smoke run) must not be exported to "),
            ):
                for out_dir in out_dirs:
                    arguments = ["--run-name", "fence_other", "final", "--families", "gru", "--out-dir", out_dir]
                    assert train.main(common + arguments + (["--fast"] if fast else [])) == 2, (fast, out_dir)
                    assert capsys.readouterr().err == (
                        f"error: {start}{always} or a folder below it, got {Path(out_dir)}; "
                        "pass another output folder\n"
                    ), (cwd, fast, out_dir)
                # 설정이 가리키는 폴더(--out-dir 없이)는 전과 같은 말로 막힌다.
                arguments = ["--run-name", "fence_other", "final", "--families", "gru"]
                assert train.main(common + arguments + (["--fast"] if fast else [])) == 2
                assert f"{start}paths.export_dir ({value}) or a folder below it" in capsys.readouterr().err
        # 어느 경우에도 학습은 시작되지 않았고 아무것도 쓰이지 않았다.
        assert fits == []
    assert sorted(path.name for path in repo.iterdir()) == ["work"] and list(work.iterdir()) == []
    assert not (tiny_run.tmp / "reports" / "fence_other").exists() and not (tmp_path / "smoke_export").exists()

    # 다른 폴더로는 그대로 나간다: 저장소 안의 이름이 비슷한 이웃 폴더(상대 경로, 저장소 루트에서 실행).
    monkeypatch.chdir(repo)
    arguments = ["--run-name", f"fence_other_{configured}", "final", "--fast", "--families", "gru"]
    assert train.main(common + arguments + ["--out-dir", "export_smoke"]) == 0
    capsys.readouterr()
    written = repo / "export_smoke"
    files = sorted(path.name for path in written.iterdir())
    assert files == ["drowsiness_v1.onnx", "feature_spec_v1.json", "rules.json"]
    verify_contract(written / "drowsiness_v1.onnx", written / "feature_spec_v1.json")
    assert not shipped.exists()


def test_cli_run_on_the_alert_baseline_arm_is_reported_when_another_arm_is_the_export_arm(
    tiny_run: SimpleNamespace, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`training.export_arm`이 다른 조건인 설정에서 alert 영상 기준선 조건으로 학습한 실행은 리포트까지 쓴다.

    그 실행의 묶음은 모두 부가 묶음이라 D11을 판정하지 않는다. 전에는 기준 LightGBM이 실행 폴더에 있으면 학습을 다 마친 뒤
    D11에 부가 묶음을 넘겨, 리포트 없이 종료 코드 2로 끝났다.
    """
    data = yaml.safe_load(tiny_run.config.read_text(encoding="utf-8"))
    assert data["training"]["export_arm"] == ARM and data["lgbm"]["reference_arm"] == ARM
    data["training"]["export_arm"] = "global"
    config = tmp_path / "export_arm_global.yaml"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    common = ["--config", str(config), "--data-root", str(tiny_run.synth_dir), "--run-name", "other_export_arm"]
    run_dir = tiny_run.tmp / "reports" / "other_export_arm"

    # 기준 LightGBM(조건 alert_baseline)이 같은 실행 폴더에 있다. 내보낼 조건의 실행이었다면 D11을 판정하는 상황이다.
    assert baseline_lgbm.main(common + ["cv", "--fast"]) == 0
    capsys.readouterr()
    assert train.main(common + ["cv", "--fast", "--arm", ARM, "--families", "gru"]) == 0
    captured = capsys.readouterr()
    assert "error:" not in captured.err
    assert (
        "supplementary analysis (training arm alert_baseline is not the export arm global): every group is "
        "supplementary and D11 is not evaluated"
    ) in captured.out
    assert "D11 gate:" not in captured.out and f"reports: {run_dir / 'summary.md'}" in captured.out

    summary = (run_dir / "summary.md").read_text(encoding="utf-8").splitlines()
    assert "SUPPLEMENTARY ANALYSIS (PLAN 11 E3): training arm alert_baseline is not the export arm" in summary[2]
    assert "(global); every group of this run is supplementary" in summary[2]
    assert "is not evaluated for this run" in summary[2]
    report = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert report["gate"] is None and report["drowsiness_source"] == "perclos"
    # 주 결과 자리에는 함께 실은 기준 LightGBM만 있고, 이 실행의 묶음(학습 조건과 앱 조건)은 모두 부가 묶음이다.
    assert [(group["family"], group["arm"]) for group in report["groups"]] == [("lgbm", ARM)]
    assert {(group["family"], group["arm"]) for group in report["supplementary"]["groups"]} >= {
        ("gru", ARM), ("gru", APP_ARM), ("selected", ARM), ("selected", APP_ARM),
    }  # fmt: skip
    preds = evaluate.validate_predictions(pd.read_parquet(run_dir / "predictions.parquet"))
    assert preds["supplementary"].all() and set(preds["run_id"]) == {"train-alert_baseline-fast"}
    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert (info["arm"], info["export_arm"]) == (ARM, "global")


BAD_RUN_NAMES = ("../export", "a/b", "/tmp/x", "..", ".", "", "a\\b")


@pytest.mark.parametrize("arguments", [["cv", "--fast"], ["final", "--fast"]], ids=["cv", "final"])
def test_cli_rejects_a_run_name_that_is_not_a_plain_folder_name(
    arguments: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--run-name`은 폴더 이름 하나여야 한다(리포트 폴더 밖에 쓰지 않게). 다른 일을 하기 전에 한 줄 오류, 종료 코드 2로 끝난다."""
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "missing.yaml"  # 설정 파일을 읽으려 했다면 이 이름이 오류에 나온다
    for name in BAD_RUN_NAMES:
        assert train.main(["--config", str(config), "--run-name", name, *arguments]) == 2, name
        captured = capsys.readouterr()
        assert captured.err == f"error: --run-name must be a plain folder name, got {name!r}\n"
        assert captured.out == ""
    # 공통 옵션을 하위 명령 뒤에 적어도 같다.
    assert train.main([*arguments, "--run-name", "x/y", "--config", str(config)]) == 2
    assert capsys.readouterr().err == "error: --run-name must be a plain folder name, got 'x/y'\n"
    # 폴더 이름 하나면 이 검사를 지나 다음 단계(없는 설정 파일)에서 멈춘다.
    assert train.main(["--config", str(config), "--run-name", "plain-name_1", *arguments]) == 2
    error = capsys.readouterr().err
    assert "missing.yaml" in error and "--run-name" not in error
    assert list(tmp_path.iterdir()) == []


def test_tiny_config_does_not_depend_on_the_data_root_of_the_base_config(
    config_path: Path, repo_root: Path, tmp_path: Path
) -> None:
    # 실제 데이터 경로를 적은 설정 사본(config/local.yaml)에서 만들어도 테스트 설정은 합성 데이터로 돈다.
    base = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base["paths"]["rldd_root"] = "/data/UTA-RLDD"
    local = tmp_path / "local.yaml"
    local.write_text(yaml.safe_dump(base), encoding="utf-8")
    assert load_config(local).paths.rldd_root == Path("/data/UTA-RLDD")
    written = load_config(write_tiny_config(local, repo_root, tmp_path))
    assert written.paths.rldd_root is None and written.paths.synth_dir == tmp_path / "nowhere"
    assert written == load_config(write_tiny_config(config_path, repo_root, tmp_path))


def test_partial_frame_drop_extraction_is_refused_unless_allowed(
    tiny_run: SimpleNamespace, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # 원본 추출은 그대로(캐시도 함께 복사한다), 15 fps 추출은 1번 피험자의 세션 3개만 있는 덜 끝난 폴더.
    root = tmp_path / "data"
    shutil.copytree(tiny_run.synth_dir, root / "synth")
    shutil.copytree(tiny_run.synth_dir.with_name("synth_steps"), root / "synth_steps")
    full = tiny_run.synth_dir.with_name("synth_fps15")
    for path in sorted(full.rglob("*.parquet")):
        if path.parent.name == "01":
            target = root / "synth_fps15" / path.relative_to(full)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    assert len(list((root / "synth_fps15").rglob("*.parquet"))) == 3
    cfg = train.with_data_root(load_config(tiny_run.config), root / "synth")

    with pytest.raises(ValueError, match="has windows for only 3 of the 10 sessions") as caught:
        train.load_cv_data(cfg, (ARM,), mix_arms=(ARM,))
    message = str(caught.value)
    assert "--allow-partial-fps-sim" in message and "S13_0" in message and "S01_0" not in message
    # 섞지 않는 호출(LightGBM, 교란 점검)과 비율 0은 그 폴더를 보지 않는다. 프레임 누락 변형도 만들지 않는다.
    plain = train.load_cv_data(cfg, (ARM,))
    assert plain.extra_tables == {} and plain.frame_drop_tables == {} and "mixes" not in plain.description
    no_mix = dataclasses.replace(
        cfg, training=dataclasses.replace(cfg.training, fps_sim_share=0.0, frame_drop_share=0.0)
    )
    unmixed = train.load_cv_data(no_mix, (ARM,), mix_arms=(ARM,))
    assert unmixed.extra_tables == {} and unmixed.frame_drop_tables == {}
    assert train.mix_sources(unmixed, no_mix, ARM) == ()
    # 15 fps 평가용으로 읽을 때도 덜 끝난 폴더는 거부한다.
    with pytest.raises(ValueError, match="has windows for only 3 of the 10 sessions"):
        train.load_cv_data(no_mix, (ARM,), fps_eval_arms=(ARM,))
    with pytest.raises(ValueError, match="fps_eval_arms must be among arms"):
        train.load_cv_data(cfg, (ARM,), fps_eval_arms=("global",))
    with pytest.raises(ValueError, match="mix_arms must be among arms"):
        train.load_cv_data(cfg, (ARM,), mix_arms=("global",))

    # 명시적으로 받아들이면 읽고, 덮은 세션 수를 설명에 적는다.
    data = train.load_cv_data(cfg, (ARM,), mix_arms=(ARM,), allow_partial_fps_sim=True)
    assert sorted(data.extra_tables[ARM].steps["subject"].unique()) == [1]
    assert "windows for 3 of 10 sessions, PARTIAL" in data.description

    # 명령: 옵션 없이는 학습을 시작하지 않고 리포트도 쓰지 않는다. 옵션을 주면 끝까지 돈다.
    common = ["--config", str(tiny_run.config), "--data-root", str(root / "synth"), "--run-name", "partial"]
    for arguments in (["cv", "--fast", "--families", "gru"], ["final", "--fast", "--out-dir", str(tmp_path / "out")]):
        assert train.main(common + arguments) == 2
        assert "--allow-partial-fps-sim" in capsys.readouterr().err
    assert not (tiny_run.tmp / "reports" / "partial").exists() and not (tmp_path / "out").exists()

    seen: list[tuple[set[int], set[int] | None]] = []
    real_fit = train.fit

    def spy_fit(train_ws: WindowSet, val_ws: Any, config: Any, seed: int, extra_train: Any = None) -> Any:
        by_name = {item.name: set(train.window_subjects(item.windows)) for item in extra_train}
        assert set(by_name) == {"fps15", "frame_drop_v0"}
        # 프레임 누락 변형은 원본 추출에서 만들므로 학습 피험자를 모두 덮는다. 덜 끝난 것은 15 fps 추출뿐이다.
        assert by_name["frame_drop_v0"] == set(train.window_subjects(train_ws))
        seen.append((set(train.window_subjects(train_ws)), by_name["fps15"]))
        return real_fit(train_ws, val_ws, config, seed, extra_train=extra_train)

    monkeypatch.setattr(train, "fit", spy_fit)
    assert train.main(common + ["cv", "--fast", "--families", "gru", "--allow-partial-fps-sim"]) == 0
    capsys.readouterr()
    summary = (tiny_run.tmp / "reports" / "partial" / "summary.md").read_text(encoding="utf-8").splitlines()
    assert "windows for 3 of 10 sessions, PARTIAL" in summary[2]
    # 1번이 학습 피험자인 학습에만 15 fps 창이 들어가고, 그때도 1번의 것뿐이다(시험 fold일 때는 들어가지 않는다).
    assert len(seen) > 0 and all(extra in (set(), {1}) for _, extra in seen)
    assert all((extra == {1}) == (1 in subjects) for subjects, extra in seen)


def test_config_fold_keys_must_match_the_data(
    tiny_run: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`evaluation.subjects_per_fold`와 `evaluation.n_folds`는 데이터와 대조된다."""
    cfg = train.with_data_root(load_config(tiny_run.config), tiny_run.synth_dir)
    assert (cfg.evaluation.subjects_per_fold, cfg.evaluation.n_folds) == (12, 5)
    assert train.load_cv_data(cfg, (ARM,)).synthetic is True  # 합성 데이터는 fold 수(여기서는 3)를 보지 않는다

    other = dataclasses.replace(cfg, evaluation=dataclasses.replace(cfg.evaluation, subjects_per_fold=10))
    with pytest.raises(ValueError, match="subjects_per_fold is 10 .* fixed at 12 subjects per fold"):
        train.load_cv_data(other, (ARM,))

    # 실제 데이터: 목록의 fold 수가 설정과 달라서는 안 된다. 같은 세션을 실제 데이터로 표시해 그 경로를 지난다.
    real_open = train.open_dataset

    def as_real(config: ExperimentConfig, fps_sim: bool = False) -> Any:
        return dataclasses.replace(real_open(config, fps_sim), synthetic=False)

    monkeypatch.setattr(train, "open_dataset", as_real)
    with pytest.raises(ValueError, match=r"the dataset has 3 fold\(s\) but evaluation.n_folds is 5"):
        train.load_cv_data(cfg, (ARM,))
    three = dataclasses.replace(cfg, evaluation=dataclasses.replace(cfg.evaluation, n_folds=3))
    data = train.load_cv_data(three, (ARM,))
    assert data.synthetic is False and sorted(data.manifest["fold"].unique()) == [1, 2, 3]


def test_rebuild_reaches_the_step_table_cache(tiny_run: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = train.with_data_root(load_config(tiny_run.config), tiny_run.synth_dir)
    seen: list[tuple[str, bool]] = []
    real_load = train.load_step_tables

    def spy_load(manifest: Any, raw_dir: Path, *args: Any, **kwargs: Any) -> Any:
        seen.append((Path(raw_dir).name, kwargs["rebuild"]))
        return real_load(manifest, raw_dir, *args, **kwargs)

    variants: list[dict[str, Any]] = []
    real_variants = train.load_frame_drop_tables

    def spy_variants(manifest: Any, raw_dir: Path, *args: Any, **kwargs: Any) -> Any:
        variants.append({"raw_dir": Path(raw_dir).name, "arms": args[1], **kwargs})
        return real_variants(manifest, raw_dir, *args, **kwargs)

    monkeypatch.setattr(train, "load_frame_drop_tables", spy_variants)
    monkeypatch.setattr(train, "load_step_tables", spy_load)
    cached = train.load_cv_data(cfg, (ARM,), mix_arms=(ARM,))
    cache_file = next(tiny_run.synth_dir.with_name("synth_steps").glob(f"{ARM}_*.steps.parquet"))
    before = cache_file.stat().st_mtime_ns
    rebuilt = train.load_cv_data(cfg, (ARM,), mix_arms=(ARM,), rebuild=True)
    # 원본 추출과 15 fps 추출의 표 모두에 전달된다. 다시 만든 표는 캐시의 것과 같고 캐시 파일은 새로 쓰였다.
    assert seen == [("synth", False), ("synth_fps15", False), ("synth", True), ("synth_fps15", True)]
    # 프레임 누락 변형은 원본 추출 폴더에서, 설정의 비율 범위와 고정 시드로, 학습 조건에 대해서만 만든다.
    assert [(call["raw_dir"], call["arms"], call["rebuild"]) for call in variants] == [
        ("synth", (ARM,), False), ("synth", (ARM,), True),
    ]  # fmt: skip
    assert all(
        (call["variant"], call["seed"], call["drop_min"], call["drop_max"]) == (0, 0, 0.10, 0.30) for call in variants
    )
    assert train.step_table_fingerprint(rebuilt.frame_drop_tables[ARM][0].steps) == train.step_table_fingerprint(
        cached.frame_drop_tables[ARM][0].steps
    )
    assert cache_file.stat().st_mtime_ns > before
    pd.testing.assert_frame_equal(rebuilt.tables[ARM].steps, cached.tables[ARM].steps)
    assert train.step_table_fingerprint(rebuilt.extra_tables[ARM].steps) == train.step_table_fingerprint(
        cached.extra_tables[ARM].steps
    )


def test_cli_does_not_overwrite_the_report_of_another_run(
    tiny_run: SimpleNamespace, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """같은 실행 폴더에 다른 학습 조건이나 다른 설정의 리포트가 있으면 학습 전에 멈춘다."""
    kept = ("run.json", "predictions.parquet", "summary.md", "selection.json", "summary.json")
    before = {name: (tiny_run.run_dir / name).read_bytes() for name in kept}
    fits: list[Any] = []
    monkeypatch.setattr(train, "fit", lambda *args, **kwargs: fits.append(args))

    # tiny 폴더에는 alert 영상 기준선 조건, 축소 설정의 리포트가 있다. 다른 조건(E3)을 같은 이름으로 돌린다.
    assert train.main(tiny_run.common + ["cv", "--fast", "--arm", "global", "--families", "gru"]) == 2
    error = capsys.readouterr().err
    assert "already holds the report of another run" in error and "--overwrite" in error
    assert "run_id: 'train-alert_baseline-fast' there, 'train-global-fast' now" in error
    # 같은 조건이어도 축소 설정이 아닌 실행은 다른 실행이다.
    assert train.main(tiny_run.common + ["cv"]) == 2
    assert "'train-alert_baseline-fast' there, 'train-alert_baseline' now" in capsys.readouterr().err
    assert fits == [] and {name: (tiny_run.run_dir / name).read_bytes() for name in kept} == before
    monkeypatch.undo()

    # 덮어쓰기는 명시적으로만: 새 실행 이름에서 전역 상수 조건을 돌린 뒤 alert 영상 기준선 조건으로 바꾼다.
    common = ["--config", str(tiny_run.config), "--data-root", str(tiny_run.synth_dir), "--run-name", "ow"]
    run_dir = tiny_run.tmp / "reports" / "ow"
    assert train.main(common + ["cv", "--fast", "--arm", "global", "--families", "gru"]) == 0
    assert json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["run_id"] == "train-global-fast"
    # 내보낼 조건이 아닌 조건으로 학습한 실행(E3의 "나머지 정규화 조건")은 묶음이 모두 부가 묶음이다.
    other = evaluate.validate_predictions(pd.read_parquet(run_dir / "predictions.parquet"))
    assert other["supplementary"].all() and set(other["arm"]) == {"global"} and set(other["variant"]) == {"base"}
    other_report = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert other_report["groups"] == [] and other_report["gate"] is None
    assert {group["family"] for group in other_report["supplementary"]["groups"]} == {"gru", "selected"}
    other_summary = (run_dir / "summary.md").read_text(encoding="utf-8").splitlines()[2]
    assert "SUPPLEMENTARY ANALYSIS (PLAN 11 E3): training arm global is not the export arm" in other_summary
    assert "(alert_baseline); every group of this run is supplementary" in other_summary
    assert train.main(common + ["cv", "--fast", "--families", "gru"]) == 2
    assert json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["run_id"] == "train-global-fast"
    assert train.main(common + ["cv", "--fast", "--families", "gru", "--overwrite"]) == 0
    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert info["run_id"] == "train-alert_baseline-fast" and info["arm"] == ARM == info["export_arm"]
    assert set(pd.read_parquet(run_dir / "predictions.parquet")["run_id"]) == {"train-alert_baseline-fast"}
    report = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert {group["family"] for group in report["groups"]} == {"gru", "selected"}  # 내보낼 조건의 묶음은 주 결과다
    assert "SUPPLEMENTARY ANALYSIS" not in (run_dir / "summary.md").read_text(encoding="utf-8")
    # 같은 실행을 다시 돌리는 것은 옵션 없이도 된다.
    assert train.main(common + ["cv", "--fast", "--families", "gru"]) == 0
    capsys.readouterr()


def test_cli_checks_the_lgbm_reference_before_training(
    tiny_run: SimpleNamespace, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # --fast로 만든 LightGBM 기준만 있는 실행 폴더에서 사전에 고정한 절차(축소 설정 없이)를 돌린다.
    run_dir = tiny_run.tmp / "reports" / "ref"
    shutil.copytree(tiny_run.run_dir / "lgbm", run_dir / "lgbm")
    fits: list[Any] = []
    monkeypatch.setattr(train, "fit", lambda *args, **kwargs: fits.append(args))
    common = ["--config", str(tiny_run.config), "--data-root", str(tiny_run.synth_dir), "--run-name", "ref"]
    assert train.main(common + ["cv", "--families", "gru"]) == 2
    error = capsys.readouterr().err
    assert "cannot serve as the LightGBM reference" in error and "made with --fast, unlike this run" in error
    # 몇 시간짜리 학습을 한 뒤가 아니라 시작하기 전에 멈춘다. 리포트는 쓰지 않았다.
    assert fits == [] and sorted(path.name for path in run_dir.iterdir()) == ["lgbm"]


def test_cli_passes_the_run_options_on(tiny_run: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_cv(cfg: ExperimentConfig, **kwargs: Any) -> Any:
        calls.append(("cv", kwargs))
        raise ValueError("stop here")

    def fake_final(cfg: ExperimentConfig, **kwargs: Any) -> Any:
        calls.append(("final", kwargs))
        raise ValueError("stop here")

    monkeypatch.setattr(train, "run_nested_cv", fake_cv)
    monkeypatch.setattr(train, "fit_final", fake_final)
    assert train.main(tiny_run.common + ["cv", "--fast"]) == 2
    assert train.main(tiny_run.common + ["--rebuild", "cv", "--overwrite", "--allow-partial-fps-sim"]) == 2
    assert train.main(tiny_run.common + ["final", "--rebuild", "--allow-partial-fps-sim", "--overwrite"]) == 2
    assert train.main(tiny_run.common + ["final"]) == 2
    flags = [
        (name, {key: kwargs.get(key) for key in ("overwrite", "allow_partial_fps_sim", "rebuild")})
        for name, kwargs in calls
    ]
    assert flags == [
        ("cv", {"overwrite": False, "allow_partial_fps_sim": False, "rebuild": False}),
        ("cv", {"overwrite": True, "allow_partial_fps_sim": True, "rebuild": True}),
        ("final", {"overwrite": True, "allow_partial_fps_sim": True, "rebuild": True}),
        ("final", {"overwrite": False, "allow_partial_fps_sim": False, "rebuild": False}),
    ]
    # 부가 실행의 선택 사항(E3): 옵션이 없으면 주 실행(base)이다.
    assert [kwargs.get("variant") for name, kwargs in calls if name == "cv"] == ["base", "base"]
    calls.clear()
    assert train.main(tiny_run.common + ["cv", "--fast", "--ordinal"]) == 2
    assert train.main(tiny_run.common + ["cv", "--no-velocity", "--arm", "global"]) == 2
    assert [(kwargs["variant"], kwargs["fast"], kwargs["arm"]) for _, kwargs in calls] == [
        ("ordinal", True, None), ("no_velocity", False, "global"),
    ]  # fmt: skip


def test_cli_reports_subject_overlap_as_a_failed_run(
    tiny_run: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def leaky_splits(manifest: pd.DataFrame) -> Any:
        for split in nested_splits(manifest):
            yield dataclasses.replace(split, train_subjects=split.train_subjects + split.test_subjects[:1])

    monkeypatch.setattr(train, "nested_splits", leaky_splits)
    code = train.main(["--config", str(tiny_run.config), "--data-root", str(tiny_run.synth_dir), "--run-name", "leak",
                       "cv", "--fast"])  # fmt: skip
    assert code == 3
    assert "subject overlap, run stopped" in capsys.readouterr().err
    assert not (tiny_run.tmp / "reports" / "leak").exists()  # 리포트를 쓰지 않았다


def test_reference_lgbm_from_other_data_is_not_mixed_into_the_report(
    tiny_run: SimpleNamespace, tmp_path: Path
) -> None:
    cfg = apply_fast_profile(train.with_data_root(load_config(tiny_run.config), tiny_run.synth_dir))
    data = train.load_cv_data(cfg, (ARM,))
    reference = train._reference_lgbm(tiny_run.run_dir, cfg, data, True)
    assert reference is not None and set(reference["family"]) == {"lgbm"}
    assert set(reference["run_id"]) == {"train-alert_baseline-fast"} and not reference["supplementary"].any()
    assert train._reference_lgbm(tmp_path, cfg, data, True) is None  # lgbm/ 폴더가 없으면 싣지 않는다

    def tampered(name: str, **changes: Any) -> Path:
        """`lgbm/`의 사본에서 run.json의 값만 바꾼 실행 폴더."""
        copied = tmp_path / name
        shutil.copytree(tiny_run.run_dir / "lgbm", copied / "lgbm")
        info_path = copied / "lgbm" / "run.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info.update(changes)
        info_path.write_text(json.dumps(info), encoding="utf-8")
        return copied

    assert train._reference_lgbm(tampered("same"), cfg, data, True) is not None  # 사본 그대로는 받는다
    # 다른 pipeline 상수.
    with pytest.raises(ValueError, match="other pipeline constants"):
        train._reference_lgbm(tampered("hash", pipeline_hash="0" * 64), cfg, data, True)
    # 축소 설정 여부가 다르다: --fast로 만든 기준을 사전에 고정한 절차의 실행이 쓰지 않는다. 그 반대도.
    with pytest.raises(ValueError, match="made with --fast, unlike this run"):
        train._reference_lgbm(tiny_run.run_dir, cfg, data, False)
    with pytest.raises(ValueError, match="made without --fast, unlike this run"):
        train._reference_lgbm(tampered("full", fast=False), cfg, data, True)
    # 데이터 종류가 다르다: run.json의 값과 predictions.parquet 안의 표지를 따로 본다.
    with pytest.raises(ValueError, match="real data, this run uses synthetic data"):
        train._reference_lgbm(tampered("real", synthetic=False), cfg, data, True)
    unmarked = tampered("unmarked")
    pd.read_parquet(unmarked / "lgbm" / "predictions.parquet").to_parquet(
        unmarked / "lgbm" / "predictions.parquet", index=False
    )
    with pytest.raises(ValueError, match="predictions.parquet carries no data marker"):
        train._reference_lgbm(unmarked, cfg, data, True)
    relabelled = tampered("relabelled")
    preds = pd.read_parquet(relabelled / "lgbm" / "predictions.parquet")
    train.write_predictions(preds, relabelled / "lgbm" / "predictions.parquet", synthetic=False)
    with pytest.raises(ValueError, match="predictions.parquet is marked real data"):
        train._reference_lgbm(relabelled, cfg, data, True)
    # 스텝 표의 내용 지문이 다르다(training_verify-01). 기록이 없는 예전 run.json도 받지 않는다.
    with pytest.raises(ValueError, match="other step features for arm alert_baseline"):
        train._reference_lgbm(tampered("steps", step_fingerprints={ARM: "0" * 64}), cfg, data, True)
    with pytest.raises(ValueError, match="other step features"):
        train._reference_lgbm(tampered("old", step_fingerprints=None), cfg, data, True)
    # closure W가 기준 W가 아니다.
    with pytest.raises(ValueError, match=r"closure W 1\.0 .*lgbm\.reference_w is 0\.5"):
        train._reference_lgbm(tampered("w", effective_closure_w=1.0), cfg, data, True)
    with pytest.raises(ValueError, match="closure W"):
        train._reference_lgbm(tampered("override", closure_w=0.5), cfg, data, True)
    other_w = dataclasses.replace(cfg, lgbm=dataclasses.replace(cfg.lgbm, reference_w=1.0))
    with pytest.raises(ValueError, match=r"lgbm\.reference_w is 1\.0"):
        train._reference_lgbm(tiny_run.run_dir, other_w, data, True)
    # 기준 조건을 돌리지 않은 실행은 기준이 아니다(오류가 아니라 없는 것으로 본다).
    assert train._reference_lgbm(tampered("arms", arms=["global"]), cfg, data, True) is None
    # 여러 가지가 어긋나면 모두 적는다.
    with pytest.raises(ValueError, match="other pipeline constants.*made without --fast") as caught:
        train._reference_lgbm(tampered("two", pipeline_hash="0" * 64, fast=False), cfg, data, True)
    assert "cannot serve as the LightGBM reference" in str(caught.value) and "Rerun" in str(caught.value)


def test_reference_lgbm_from_another_dataset_with_the_same_constants_is_refused(
    tiny_run: SimpleNamespace, pipeline_spec: PipelineSpec, tmp_path: Path
) -> None:
    # 같은 pipeline 상수, 같은 피험자와 파일 이름, 둘 다 합성 데이터지만 다른 시드로 만든 세션(다시 한 추출을 흉내 낸다).
    other_dir = tmp_path / "data" / "synth"
    for job in synth.session_jobs(max(TINY_SUBJECTS), TINY_MINUTES, 1):
        if job.subject in TINY_SUBJECTS:
            synth.write_session(job, other_dir, pipeline_spec)
    cfg = apply_fast_profile(train.with_data_root(load_config(tiny_run.config), other_dir))
    other = train.load_cv_data(cfg, (ARM,))
    same = train.load_cv_data(apply_fast_profile(train.with_data_root(cfg, tiny_run.synth_dir)), (ARM,))
    assert other.tables[ARM].pipeline_hash == same.tables[ARM].pipeline_hash and other.synthetic and same.synthetic
    assert other.tables[ARM].quality["session_id"].tolist() == same.tables[ARM].quality["session_id"].tolist()
    assert train.step_table_fingerprint(other.tables[ARM].steps) != train.step_table_fingerprint(same.tables[ARM].steps)

    info = json.loads((tiny_run.run_dir / "lgbm" / "run.json").read_text(encoding="utf-8"))
    assert info["step_fingerprints"] == {ARM: train.step_table_fingerprint(same.tables[ARM].steps)}
    assert train._reference_lgbm(tiny_run.run_dir, cfg, same, True) is not None
    with pytest.raises(ValueError, match="other step features for arm alert_baseline"):
        train._reference_lgbm(tiny_run.run_dir, cfg, other, True)

    # 학습 조건이 기준 조건이 아니어도(전역 상수 조건) 기준 조건의 스텝 표를 읽어 같은 대조를 한다.
    global_only = train.load_cv_data(cfg, ("global",))
    assert ARM not in global_only.tables
    with pytest.raises(ValueError, match="other step features for arm alert_baseline"):
        train._reference_lgbm(tiny_run.run_dir, cfg, global_only, True)


def test_confound_note_reads_the_probe_record(tmp_path: Path) -> None:
    assert train.confound_note(tmp_path) == ""
    path = tmp_path / "probes" / "probes.json"
    train.write_json({"probes": [{"name": "recording_metadata", "above_chance": False}]}, path)
    assert train.confound_note(tmp_path) == ""
    train.write_json(
        {
            "probes": [
                {"name": "recording_metadata", "above_chance": False},
                {"name": "baseline_offset", "above_chance": True},
            ]
        },
        path,
    )
    assert train.confound_note(tmp_path).startswith("CONFOUND PROBE ABOVE CHANCE (baseline_offset;")


# ---------------------------------------------------------------------------
# 부가 실행의 명령: cv --ordinal, cv --no-velocity (PLAN 11장 E3)
# ---------------------------------------------------------------------------


def _files(folder: Path, skip: tuple[str, ...]) -> dict[str, bytes]:
    """folder 아래 파일들의 바이트(하위 폴더 skip은 뺀다)."""
    return {
        str(path.relative_to(folder)): path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.relative_to(folder).parts[0] not in skip
    }


def test_cli_supplementary_runs_write_their_own_folder_and_leave_the_main_report_alone(
    tiny_run: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tiny_run.run_dir
    variants = ("ordinal", "no_velocity")
    before = _files(run_dir, variants)
    assert "summary.md" in before and "predictions.parquet" in before and "lgbm/predictions.parquet" in before
    main_report = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    main_run_id = "train-alert_baseline-fast"

    for flag, variant in (("--ordinal", "ordinal"), ("--no-velocity", "no_velocity")):
        assert train.main(tiny_run.common + ["cv", "--fast", flag]) == 0
        out = capsys.readouterr().out
        assert f"supplementary run ({variant}): every group is supplementary" in out and "D11 gate" not in out
        sub = run_dir / variant
        assert f"reports: {sub / 'summary.md'}" in out
        run_id = f"train-alert_baseline-{variant}-fast"

        # 이 실행의 예측 표: run_id 접미사, variant 표지, 전부 부가 묶음, 원본 추출 특징만.
        preds = evaluate.validate_predictions(pd.read_parquet(sub / "predictions.parquet"))
        assert set(preds["run_id"]) == {run_id} and set(preds["variant"]) == {variant}
        assert preds["supplementary"].all() and set(preds["feature_source"]) == {"native"}
        assert set(preds["family"]) == {"tcn", "gru", "selected"} and set(preds["arm"]) == {ARM, APP_ARM}
        assert sorted(preds["fold"].unique()) == [1, 2, 3] and sorted(preds["seed"].unique()) == [0, 1]

        # 리포트: 주 결과 자리에는 주 실행에서 읽어 온 묶음만 있고(값도 주 리포트와 같다) D11은 판정하지 않는다.
        report = json.loads((sub / "summary.json").read_text(encoding="utf-8"))
        main_groups = {(group["run_id"], group["family"], group["arm"]): group for group in report["groups"]}
        assert set(main_groups) == {(main_run_id, f, a) for f in ("tcn", "gru", "selected") for a in (ARM, APP_ARM)}
        for group in main_report["groups"]:
            key = (group["run_id"], group["family"], group["arm"])
            if key in main_groups:
                assert main_groups[key]["summary"] == group["summary"], key
        assert report["gate"] is None and report["drowsiness_source"] == "perclos" and report["paired"] == []
        extra = report["supplementary"]["groups"]
        assert {(g["run_id"], g["family"], g["arm"], g["variant"]) for g in extra} == {
            (run_id, f, a, variant) for f in ("tcn", "gru", "selected") for a in (ARM, APP_ARM)
        }
        # 짝지은 비교: 묶음마다 주 실행의 같은 계열·같은 평가 조건 묶음과 한 쌍(6개), 그리고 이 실행 안의 TCN 대 GRU.
        pairs = {(row["a"], row["b"]) for row in report["supplementary"]["paired"]}
        assert len(pairs) == 7 and len(report["supplementary"]["paired"]) == 7 * len(evaluate.PAIRED_METRICS)
        across = [(a, b) for a, b in pairs if main_run_id + " /" in b and run_id not in b]
        assert len(across) == 6 and all(run_id in a and a.split(" / ")[1] == b.split(" / ")[1] for a, b in across)

        summary = (sub / "summary.md").read_text(encoding="utf-8").splitlines()
        assert summary[0].startswith("# SYNTHETIC data") and "SUPPLEMENTARY RUN (PLAN 11 E3)" in summary[2]
        assert "never used for the headline rows or D11" in summary[2] and "copied from the main run" in summary[2]
        info = json.loads((sub / "run.json").read_text(encoding="utf-8"))
        assert (info["run_id"], info["variant"], info["arm"], info["fast"]) == (run_id, variant, ARM, True)
        notes = info["supplementary"]
        assert notes["paired_with_main_run"] is True and notes["fold_int8"]["status"] == "not requested"
        assert notes["fps15_evaluation"] == {"evaluated_arms": [], "skipped": {}}
        assert info["train_sources"] == {"native": 0.5, "fps15": 0.25, "frame_drop_v0": 0.25}
        if variant == "ordinal":
            assert "ordinal soft labels instead of uniform label smoothing: each class gives 0.2" in summary[2]
        else:
            assert "without the velocity features #7 closing_vel_mean_60, #8 opening_vel_mean_60" in summary[2]

    # 주 실행의 파일은 하나도 바뀌지 않았다(리포트, 예측 표, 선택 기록, lgbm/, final/ 모두).
    assert _files(run_dir, variants) == before
    # 두 부가 실행의 모델은 서로 다르고 주 실행과도 다르다(선택 사항이 학습에 닿았다).
    hashes = {
        name: {
            (record["fold"], item["name"], item["seed"]): item["weights_sha256"]
            for record in json.loads((folder / "selection.json").read_text(encoding="utf-8"))["folds"]
            for item in record["refits"]
        }
        for name, folder in (
            ("base", run_dir), ("ordinal", run_dir / "ordinal"), ("no_velocity", run_dir / "no_velocity"),
        )  # fmt: skip
    }
    for a, b in (("base", "ordinal"), ("base", "no_velocity"), ("ordinal", "no_velocity")):
        shared = set(hashes[a]) & set(hashes[b])
        assert shared and all(hashes[a][key] != hashes[b][key] for key in shared), (a, b)
    # 두 선택 사항을 한 번에 줄 수는 없다.
    with pytest.raises(SystemExit):
        train.main(tiny_run.common + ["cv", "--fast", "--ordinal", "--no-velocity"])
    capsys.readouterr()


def test_cli_supplementary_run_without_or_with_a_stale_main_run(
    tiny_run: SimpleNamespace, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # (1) 주 실행이 없는 실행 폴더: 짝지은 비교 없이 돌고, 주 결과 묶음이 없는 리포트가 된다.
    common = ["--config", str(tiny_run.config), "--data-root", str(tiny_run.synth_dir), "--run-name", "solo"]
    assert train.main(common + ["cv", "--fast", "--ordinal", "--families", "gru"]) == 0
    capsys.readouterr()
    solo = tiny_run.tmp / "reports" / "solo"
    assert sorted(path.name for path in solo.iterdir()) == ["ordinal"]  # 주 리포트 자리에는 아무것도 쓰지 않는다
    report = json.loads((solo / "ordinal" / "summary.json").read_text(encoding="utf-8"))
    assert report["groups"] == [] and len(report["supplementary"]["groups"]) == 4
    assert report["supplementary"]["paired"] == [] and report["gate"] is None
    info = json.loads((solo / "ordinal" / "run.json").read_text(encoding="utf-8"))
    assert info["supplementary"]["paired_with_main_run"] is False
    assert "no main run was found" in (solo / "ordinal" / "summary.md").read_text(encoding="utf-8").splitlines()[2]

    # (2) 주 실행의 리포트가 다른 스텝 특징에서 나온 것이면(추출이나 코드가 바뀐 뒤 다시 돌리지 않음) 학습 전에 멈춘다.
    stale = tiny_run.tmp / "reports" / "stale"
    stale.mkdir()
    for name in ("run.json", "predictions.parquet"):
        shutil.copy2(tiny_run.run_dir / name, stale / name)
    record = json.loads((stale / "run.json").read_text(encoding="utf-8"))
    record["step_fingerprints"][ARM] = "0" * 64
    (stale / "run.json").write_text(json.dumps(record), encoding="utf-8")
    fits: list[Any] = []
    monkeypatch.setattr(train, "fit", lambda *args, **kwargs: fits.append(args))
    common = ["--config", str(tiny_run.config), "--data-root", str(tiny_run.synth_dir), "--run-name", "stale"]
    assert train.main(common + ["cv", "--fast", "--no-velocity"]) == 2
    error = capsys.readouterr().err
    assert "cannot serve as the main run of this supplementary run" in error
    assert "other step features for alert_baseline" in error and "Rerun the main run" in error
    assert fits == [] and not (stale / "no_velocity").exists()
    # 축소 설정이 다른 주 실행(run_id가 다르다)도 받지 않는다.
    assert train.main(common + ["cv", "--no-velocity"]) == 2
    assert "run_id is 'train-alert_baseline-fast', this run needs 'train-alert_baseline'" in capsys.readouterr().err
    assert fits == []


def test_main_run_predictions_keeps_only_the_main_groups_of_the_named_runs(tiny_run: SimpleNamespace) -> None:
    run_dir = tiny_run.run_dir
    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    arguments = {
        "expected": {"run_id": "train-alert_baseline-fast"},
        "synthetic": True,
        "pipeline_hash": info["pipeline_hash"],
        "fingerprints": info["step_fingerprints"],
        "run_ids": ("train-alert_baseline-fast",),
    }
    main = train.main_run_predictions(run_dir, **arguments)
    stored = evaluate.validate_predictions(pd.read_parquet(run_dir / "predictions.parquet"))
    assert stored["supplementary"].any()  # 주 실행의 파일에는 15 fps, INT8 부가 묶음도 들어 있다
    assert main is not None and not main["supplementary"].any() and len(main) == int((~stored["supplementary"]).sum())
    assert set(main["family"]) == {"tcn", "gru", "selected"}
    # 폴더에 주 실행이 없거나 그 run_id의 행이 없으면 None이다.
    assert train.main_run_predictions(run_dir / "nowhere", **arguments) is None
    assert train.main_run_predictions(run_dir, **{**arguments, "run_ids": ("train-global-fast",)}) is None
    # 어긋나는 것은 이유와 함께 거부한다.
    for change, message in (
        ({"synthetic": False}, "synthetic data, this run uses real data"),
        ({"pipeline_hash": "0" * 64}, "other pipeline constants"),
        ({"expected": {"run_id": "train-global-fast"}}, "run_id is 'train-alert_baseline-fast'"),
        ({"fingerprints": {APP_ARM: "f" * 64}}, "other step features for session_baseline"),
    ):
        with pytest.raises(ValueError, match=message):
            train.main_run_predictions(run_dir, **{**arguments, **change})
    # 주 실행의 기록에 없는 조건의 지문은 대조하지 않는다(그 조건의 묶음이 없을 뿐이다).
    assert train.main_run_predictions(run_dir, **{**arguments, "fingerprints": {"global": "f" * 64}}) is not None
    # 부가 실행의 묶음과 주 실행의 묶음을 짝짓는다: 같은 계열, 같은 평가 조건, 같은 출처, run_id만 대응표로 바꾼다.
    ordinal = main.assign(run_id="train-alert_baseline-ordinal-fast", variant="ordinal", supplementary=True)
    run_ids = {"train-alert_baseline-ordinal-fast": "train-alert_baseline-fast"}
    pairs = train.supplementary_run_pairs(ordinal, main, run_ids)
    assert len(pairs) == 6
    for a, b in pairs:
        assert a == ("train-alert_baseline-ordinal-fast", b[1], b[2], "native", "ordinal")
        assert b[0] == "train-alert_baseline-fast" and len(b) == 3
    assert train.supplementary_run_pairs(ordinal, main, {"other-run": "train-alert_baseline-fast"}) == []
    # 주 실행에 같은 출처의 주 결과 묶음이 없으면 쌍이 없다(15 fps 묶음은 주 실행에서도 부가 묶음이라 읽어 오지 않는다).
    fps = stored[stored["feature_source"] == "fps15"]
    assert train.supplementary_run_pairs(fps, main, {"train-alert_baseline-fast": "train-alert_baseline-fast"}) == []


# ---------------------------------------------------------------------------
# 회전 점검 기록: 학습 명령이 시작할 때 알리고 실행 기록과 리포트 설명에 남긴다
# ---------------------------------------------------------------------------

OVERRIDES_KEY = "extraction.rotation_overrides"
# 합성 데이터(세션 파일 10개)에서 명령이 둘째 줄에 출력하는 것: 합성 세션에는 회전 점검 기록이 없다.
SYNTHETIC_ROTATION_LINE = (
    "rotation check of 10 raw file(s): 0 decided by the automatic check, 0 extracted with a manual override, "
    "0 ambiguous without a manual override, 0 with an override that was not applied, "
    "10 without a rotation-check record"
)


def test_cli_prints_the_rotation_line_and_records_it_on_synthetic_data(tiny_run: SimpleNamespace) -> None:
    """합성 데이터에서는 한 줄이 더 나오고 실행 기록에 `rotation_check`가 생길 뿐이다. 리포트의 설명은 그대로다."""
    for name in ("lgbm", "cv", "final"):
        lines = tiny_run.outputs[name].splitlines()
        assert lines[0].startswith("SYNTHETIC data") and lines[1] == SYNTHETIC_ROTATION_LINE, name
        assert sum("rotation" in line for line in lines) == 1, name  # 확인할 영상이 없으므로 목록 줄은 없다
    record = {
        "overrides_key": OVERRIDES_KEY, "n_files": 10, "n_missing": 0, "decided_by_check": 0, "manual_override": 0,
        "ambiguous_without_override": 0, "override_not_applied": 0, "no_record": 10, "flagged": [],
    }  # fmt: skip
    run_dir = tiny_run.run_dir
    for path in (run_dir / "run.json", run_dir / "lgbm" / "run.json", run_dir / "final" / "final.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["rotation_check"] == record, path
        assert "ROTATION" not in payload["description"]
    for path in (run_dir / "summary.md", run_dir / "lgbm" / "summary.md", run_dir / "final" / "summary.md"):
        assert "ROTATION" not in path.read_text(encoding="utf-8"), path


def test_rotation_note_and_record_of_the_run_files() -> None:
    assert train.rotation_note(None) == "" and train.rotation_record(None) is None
    clean = RotationSummary(n_files=4, n_missing=1, n_decided=3, n_manual=1, n_no_record=0, flagged=())
    assert train.rotation_note(clean) == ""
    assert train.rotation_record(clean) == {
        "overrides_key": OVERRIDES_KEY, "n_files": 4, "n_missing": 1, "decided_by_check": 3, "manual_override": 1,
        "ambiguous_without_override": 0, "override_not_applied": 0, "no_record": 0, "flagged": [],
    }  # fmt: skip
    flags = (
        RotationFlag("01/5.mp4", ROTATION_AMBIGUOUS, 0, 0, "ambiguous_kept_metadata", None),
        RotationFlag("02/0.mp4", ROTATION_AMBIGUOUS, 180, 90, "ambiguous_smallest_roll", None),
        RotationFlag("02/10.mp4", ROTATION_OVERRIDE_NOT_APPLIED, 0, 0, "ambiguous_kept_metadata", 180),
    )
    flagged = RotationSummary(n_files=9, n_missing=0, n_decided=5, n_manual=1, n_no_record=0, flagged=flags)
    assert train.rotation_note(flagged) == (
        "; ROTATION NOT CONFIRMED for 3 of 9 raw file(s) (2 where the automatic rotation check of the extraction "
        "could not tell the angle and no manual override exists, 1 extracted at another angle than their manual "
        "override): these videos may be rotated wrongly and they are included in every number below as they were "
        "extracted; look at them and fill `extraction.rotation_overrides` (list: key `rotation_check` of the run "
        "record in this folder, or `python -m drowsy.dataset`)"
    )
    record = train.rotation_record(flagged)
    assert record is not None and record["ambiguous_without_override"] == 2 and record["override_not_applied"] == 1
    assert [item["video"] for item in record["flagged"]] == ["01/5.mp4", "02/0.mp4", "02/10.mp4"]
    json.dumps(record, allow_nan=False)


AMBIGUOUS_VIDEO = "Fold2_part1/13/5.mp4"  # 대역 데이터에서 자동 점검이 각도를 가르지 못한 것으로 적는 영상
FLAGGED_ROTATION_LINE = (
    "rotation check of 10 raw file(s): 9 decided by the automatic check, 0 extracted with a manual override, "
    "1 ambiguous without a manual override, 0 with an override that was not applied, "
    "0 without a rotation-check record. LOOK AT THE 1 FLAGGED VIDEO(S) before using this data: the lines below"
)
FLAGGED_VIDEO_LINE = (
    f"  rotation ambiguous: {AMBIGUOUS_VIDEO}: extracted at 0 deg (video metadata 0, ambiguous_kept_metadata); "
    "look at the video, add its angle to extraction.rotation_overrides and run the extraction again"
)
FLAGGED_NOTE = (
    "; ROTATION NOT CONFIRMED for 1 of 10 raw file(s) (1 where the automatic rotation check of the extraction "
    "could not tell the angle and no manual override exists, 0 extracted at another angle than their manual override)"
)
FLAGGED_RECORD = {
    "overrides_key": OVERRIDES_KEY, "n_files": 10, "n_missing": 0, "decided_by_check": 9, "manual_override": 0,
    "ambiguous_without_override": 1, "override_not_applied": 0, "no_record": 0,
    "flagged": [
        {"video": AMBIGUOUS_VIDEO, "kind": "ambiguous", "rotation": 0, "metadata_rotation": 0,
         "reason": "ambiguous_kept_metadata", "override": None},
    ],
}  # fmt: skip


def rotation_check_record(ambiguous: bool) -> dict[str, Any]:
    """추출이 parquet 메타데이터의 `rotation_check` 키에 적는 기록과 같은 모양의 dict(값은 이 테스트가 정한다)."""
    return {
        "version": extract.ROTATION_CHECK_VERSION,
        "settings": {"frames": 12, "interval_ms": 2000, "roll_margin_deg": 15.0},
        "rates": {"0": 1.0, "90": 0.0, "180": 1.0 if ambiguous else 0.0, "270": 0.0},
        "rolls": {"0": -2.0, "90": None, "180": -0.6 if ambiguous else None, "270": None},
        "metadata_rotation": 0,
        "n_frames": 12,
        "ambiguous": ambiguous,
        "reason": extract.REASON_KEPT_METADATA if ambiguous else extract.REASON_BEST_RATE,
    }


@pytest.fixture(scope="module")
def flagged_run(
    tmp_path_factory: pytest.TempPathFactory, config_path: Path, repo_root: Path, pipeline_spec: PipelineSpec
) -> SimpleNamespace:
    """회전이 확인되지 않은 영상이 하나 있는 데이터로 네 명령(`baseline_lgbm cv`, `probes`, `train cv`, `final`)을 돌린 결과.

    회전 점검 기록은 실제 영상의 추출 결과에만 있으므로 실제 데이터의 경로(`paths.rldd_root`가 있는 설정)를 지나는 대역
    데이터를 쓴다: 합성 세션의 프레임(명목 18 fps)을 합성 표시 없이, 추출이 적는 것과 같은 메타데이터(`source`, `rotation`,
    `rotation_check`)로 쓴다(`extract.write_raw_frames`). 영상 폴더에는 이름만 있는 빈 파일을 둔다(세션 목록을 만드는 코드는
    영상을 열지 않는다). 파일 열 개 가운데 `AMBIGUOUS_VIDEO` 하나만 점검이 각도를 가르지 못한 것으로 적혀 있다.
    이 데이터의 리포트는 임시 폴더에만 쓰이고 숫자에는 아무 뜻이 없다(보는 것은 회전에 관한 줄과 기록뿐이다).
    `started`는 명령마다, 데이터를 읽기 시작한 시점(`train.load_cv_data`. 학습과 재생은 그 뒤다)까지 출력된 글이다.
    """
    tmp = tmp_path_factory.mktemp("flagged_run")
    videos, raw = tmp / "videos", tmp / "raw"
    for job in synth.session_jobs(max(TINY_SUBJECTS), TINY_MINUTES, 0):
        if job.subject not in TINY_SUBJECTS:
            continue
        video = Path(job.rel_path).with_suffix(".mp4")
        (videos / video).parent.mkdir(parents=True, exist_ok=True)
        (videos / video).touch()
        metadata = {
            "source": f"/camera/{video.as_posix()}",
            "rotation": 0,
            "rotation_override": None,
            extract.ROTATION_CHECK_KEY: rotation_check_record(video.as_posix() == AMBIGUOUS_VIDEO),
        }
        profile = dataclasses.replace(synth.make_subject(job.subject, job.seed), fps=18.0)
        frames = synth.generate_session(profile, job.class_index, job.duration_ms, job.seed, pipeline_spec, job.part)
        extract.write_raw_frames(frames, raw / "native" / job.rel_path, pipeline_spec, metadata)

    data = yaml.safe_load(write_tiny_config(config_path, repo_root, tmp).read_text(encoding="utf-8"))
    data["paths"].update(rldd_root=str(videos), raw_dir=str(raw))
    data["evaluation"].update(n_folds=3, lgbm_min_macro_f1=0.0)  # fold 셋짜리 데이터다. E1 기준은 이 시험의 대상이 아니다
    data["training"].update(frame_drop_variants=0)  # 프레임 누락 변형을 만들지 않는다(시간을 줄인다)
    data["fast"].update(seeds=[0])
    config = tmp / "flagged.yaml"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    common = ["--config", str(config), "--run-name", "flagged"]
    commands: dict[str, tuple[Callable[[list[str]], int], list[str]]] = {
        "lgbm": (baseline_lgbm.main, ["cv", "--fast"]),
        "probes": (baseline_lgbm.main, ["probes", "--workers", "1"]),
        "cv": (train.main, ["cv", "--fast", "--families", "gru"]),
        "final": (train.main, ["final", "--fast", "--families", "gru", "--out-dir", str(tmp / "out")]),
    }
    outputs: dict[str, str] = {}
    started: dict[str, str] = {}
    codes: dict[str, int] = {}
    real_load = train.load_cv_data
    with pytest.MonkeyPatch.context() as patch:
        for name, (main, arguments) in commands.items():
            buffer = io.StringIO()

            def spy_load(*args: Any, _name: str = name, _buffer: io.StringIO = buffer, **kwargs: Any) -> Any:
                started.setdefault(_name, _buffer.getvalue())
                return real_load(*args, **kwargs)

            patch.setattr(train, "load_cv_data", spy_load)
            with contextlib.redirect_stdout(buffer):
                codes[name] = main(common + arguments)
            outputs[name] = buffer.getvalue()
    return SimpleNamespace(
        tmp=tmp, config=config, data=data, codes=codes, outputs=outputs, started=started,
        run_dir=tmp / "reports" / "flagged",
    )  # fmt: skip


def test_cli_reports_a_video_with_an_unconfirmed_rotation_before_training_starts(flagged_run: SimpleNamespace) -> None:
    """네 명령 모두: 데이터 설명 다음 줄에 구분별 파일 수를, 그 아래에 확인이 필요한 영상을 적는다. 학습 전에 출력한다."""
    assert flagged_run.codes == {"lgbm": 0, "probes": 0, "cv": 0, "final": 0}
    for name in ("lgbm", "probes", "cv", "final"):
        lines = flagged_run.outputs[name].splitlines()
        assert lines[0].startswith("UTA-RLDD at ") and "3 subjects, 9 logical videos, native frame rate" in lines[0]
        assert lines[1] == FLAGGED_ROTATION_LINE, name
        assert lines[2] == FLAGGED_VIDEO_LINE, name
        assert not lines[3].startswith("  rotation"), name  # 확인할 영상은 하나뿐이다
        # 데이터를 읽기 시작할 때(학습과 재생은 그 뒤다) 그 줄들은 이미 출력돼 있었다.
        assert flagged_run.started[name].splitlines()[:3] == lines[:3], name


def test_cli_records_the_unconfirmed_rotation_in_the_run_record_and_the_report_description(
    flagged_run: SimpleNamespace,
) -> None:
    run_dir = flagged_run.run_dir
    records = {
        "lgbm": run_dir / "lgbm" / "run.json",
        "probes": run_dir / "probes" / "probes.json",
        "cv": run_dir / "run.json",
        "final": run_dir / "final" / "final.json",
    }
    summaries = {
        "lgbm": run_dir / "lgbm" / "summary.md",
        "probes": run_dir / "probes" / "summary.md",
        "cv": run_dir / "summary.md",
        "final": run_dir / "final" / "summary.md",
    }
    for name in records:
        payload = json.loads(records[name].read_text(encoding="utf-8"))
        assert payload["synthetic"] is False and payload["rotation_check"] == FLAGGED_RECORD, name
        # 설명은 데이터 설명으로 시작하고, 그 바로 뒤에 회전이 확인되지 않은 파일의 수가 붙는다.
        description = payload["description"]
        assert description.startswith("UTA-RLDD at ") and FLAGGED_NOTE in description, name
        assert description.index("native frame rate" + FLAGGED_NOTE) > 0, name
        assert "`rotation_check` of the run record in this folder, or `python -m drowsy.dataset`" in description
        # 리포트(summary.md)의 설명 문단이 그 설명 그대로다.
        summary = summaries[name].read_text(encoding="utf-8").splitlines()
        assert summary[0] == "# " + evaluate.REAL_NOTE and description in summary, name
    # 같은 설명이 교차검증의 선택 기록에도 실린다(`write_run_files`의 머리 부분).
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    assert FLAGGED_NOTE in selection["description"]


def test_final_summary_says_what_the_plan_decision_numbers_are(
    tiny_run: SimpleNamespace, flagged_run: SimpleNamespace
) -> None:
    """`train final`의 final/summary.md도 D 번호(D10, D16)를 쓰므로, 교차검증 리포트와 같은 자리에 그 번호가 계획서 1.2절의
    결정 번호라는 한 줄(`evaluate.PLAN_DECISION_NOTE`)을 한 번 싣는다(`test_evaluate.test_reports_say_what_the_plan_decision_
    numbers_are`와 같은 규칙: 설명 문단과 데이터 종류 문장 다음, 첫 항목 앞).

    합성 데이터(`tiny_run`)와 실제 데이터 경로를 지나는 대역 데이터(`flagged_run`, GRU만 학습해 INT8 파일이 없다) 둘을 본다.
    """
    note = evaluate.PLAN_DECISION_NOTE
    cases = {"synthetic": tiny_run.run_dir / "final", "real": flagged_run.run_dir / "final"}
    for name, final_dir in cases.items():
        record = json.loads((final_dir / "final.json").read_text(encoding="utf-8"))
        lines = (final_dir / "summary.md").read_text(encoding="utf-8").splitlines()
        assert lines.count(note) == 1 and lines[6] == note, name
        assert lines[3] == lines[5] == lines[7] == "" and lines[8].startswith("- 생성 명령: "), name
        assert lines[4].startswith("합성 데이터로 학습한 모델이다." if name == "synthetic" else "이 모델은 전체 피험자로"), name
        used = set(re.findall(r"\bD\d+\b", "\n".join(line for line in lines if line != note)))
        # D10: 실제 데이터의 데이터 종류 문장. D16: INT8 파일을 만들지 않았다는 항목(TCN이 최종 모델일 때만 만든다)
        expected = ({"D10"} if name == "real" else set()) | ({"D16"} if record["export"]["int8"] is None else set())
        assert used == expected, name
    assert json.loads((cases["real"] / "final.json").read_text(encoding="utf-8"))["export"]["int8"] is None


def test_rotation_record_follows_the_override_table_of_the_config(
    flagged_run: SimpleNamespace, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """영상을 보고 보정표에 각도를 적으면 그 영상은 목록에서 빠지고, 다른 각도를 적고 추출을 다시 하지 않았으면 남는다."""
    base = load_config(flagged_run.config)

    def loaded(overrides: dict[str, int]) -> tuple[ExperimentConfig, train.CVData]:
        cfg = dataclasses.replace(base, extraction=dataclasses.replace(base.extraction, rotation_overrides=overrides))
        return cfg, train.load_cv_data(cfg, (ARM,))  # 스텝 표는 명령이 만든 캐시에서 읽는다

    _, data = loaded({})
    assert data.synthetic is False and train.rotation_record(data.rotation) == FLAGGED_RECORD
    assert data.description.endswith("native frame rate" + train.rotation_note(data.rotation))

    # 점검이 고른 각도(0도)가 맞다고 적었다: 확인이 필요한 영상이 없고 설명에 아무 말도 붙지 않는다.
    cfg, data = loaded({AMBIGUOUS_VIDEO: 0})
    assert data.rotation is not None and data.rotation.flagged == ()
    assert (data.rotation.n_decided, data.rotation.n_manual) == (9, 1)
    assert "ROTATION" not in data.description and data.description.endswith("native frame rate")
    printed = train.print_rotation_check(cfg, train.open_dataset(cfg))
    assert printed == data.rotation
    assert capsys.readouterr().out == (
        "rotation check of 10 raw file(s): 9 decided by the automatic check, 1 extracted with a manual override, "
        "0 ambiguous without a manual override, 0 with an override that was not applied, "
        "0 without a rotation-check record\n"
    )

    # 180도가 맞다고 적고 추출을 다시 돌리지 않았다: 파일은 아직 0도로 추출돼 있다.
    cfg, data = loaded({AMBIGUOUS_VIDEO: 180})
    assert data.rotation is not None and data.rotation.flagged == (
        RotationFlag(AMBIGUOUS_VIDEO, ROTATION_OVERRIDE_NOT_APPLIED, 0, 0, "ambiguous_kept_metadata", 180),
    )
    assert "ROTATION NOT CONFIRMED for 1 of 10 raw file(s) (0 where" in data.description
    assert "1 extracted at another angle than their manual override)" in data.description
    train.print_rotation_check(cfg, train.open_dataset(cfg))
    lines = capsys.readouterr().out.splitlines()
    assert "0 ambiguous without a manual override, 1 with an override that was not applied" in lines[0]
    assert lines[1] == (
        f"  rotation override not applied: {AMBIGUOUS_VIDEO}: extraction.rotation_overrides says 180 deg but the "
        "raw file was extracted at 0 deg; run the extraction again"
    )
