"""drowsy.export_onnx 테스트 (PLAN 7장, 10.1절 "내보내기"). 무작위 가중치와 합성 입력만 쓴다."""

from __future__ import annotations

import ast
import dataclasses
import functools
import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from onnx import numpy_helper
from torch import nn

from drowsy import export_onnx
from drowsy.config import load_config
from drowsy.export_onnx import (
    ALLOWED_OPS,
    ALLOWED_OPS_DYNAMO,
    ALLOWED_OPS_INT8,
    METADATA_KEYS,
    SYNTHETIC_TRAINED_ON_PREFIX,
    TRAINED_ON_NONE,
    ContractError,
    export,
    exported_spec_name,
    exported_spec_path,
    git_commit,
    graph_operators,
    is_export_dir_or_below,
    is_synthetic_trained_on,
    measure_latency,
    model_file_name,
    parity_max_abs_diff,
    quantize_int8,
    random_raw_inputs,
    read_metadata,
    refuse_fast_export,
    refuse_synthetic_export,
    verify_contract,
    verify_file_name,
)
from drowsy.models import KIND_GRU, KIND_TCN, KINDS, ExportWrapper, build_model, save_checkpoint
from drowsy.spec import (
    FeatureSpec,
    SpecError,
    file_sha256,
    load_feature_spec,
    read_feature_spec_dict,
    write_feature_spec,
)

LOG1P_COLUMNS = [9, 10, 11, 12, 14]
META = {"model_version": "3", "trained_on": "synthetic sessions (test)", "commit": "0123abc-dirty"}

# 학습 뒤의 spec을 흉내 낸 값(변환 뒤 척도). 특징마다 척도가 크게 다르다.
MEANS = [0.06, 0.07, 0.18, 0.19, 17.0, 210.0, 95.0, 6.5, 5.0, 0.4, 5.9, 4.8, 0.1, 2.5, 0.2, 0.05, 0.06]
STDS = [0.08, 0.11, 0.09, 0.12, 9.0, 120.0, 80.0, 3.0, 2.5, 0.6, 1.1, 1.6, 0.3, 1.8, 0.4, 0.1, 0.15]
TEMPERATURE = 1.7


@dataclasses.dataclass(frozen=True)
class Exported:
    kind: str
    wrapper: ExportWrapper
    spec: FeatureSpec
    onnx_path: Path
    spec_path: Path  # 모델 옆의 사본(feature_spec_v3.json)


def _write_spec(source: Path, out: Path, means: list[float], stds: list[float], temperature: float) -> Path:
    data = read_feature_spec_dict(source)
    for feature, mean, std in zip(data["features"], means, stds, strict=True):
        feature["mean"] = mean
        feature["std"] = std
    data["standardization"]["temperature"] = temperature
    write_feature_spec(data, out)
    return out


def _make_wrapper(kind: str, spec: FeatureSpec, seed: int = 0, **hyper: object) -> ExportWrapper:
    torch.manual_seed(seed)
    net = build_model(kind, spec.n_features, **hyper)
    # 배치 정규화가 항등 변환이면 합성곱에 합쳐졌는지 일치 검증으로 드러나지 않는다.
    with torch.no_grad():
        for module in net.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.running_mean.normal_(0.0, 0.5)
                module.running_var.uniform_(0.5, 2.0)
                module.weight.uniform_(0.5, 1.5)
                module.bias.normal_(0.0, 0.3)
    return ExportWrapper(net, spec, spec.standardization.temperature)


def _run(onnx_path: Path, window: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return session.run(["probs"], {"features": window.astype(np.float32)})[0]


@pytest.fixture(scope="module")
def trained_spec_path(tmp_path_factory: pytest.TempPathFactory, feature_spec_path: Path) -> Path:
    out = tmp_path_factory.mktemp("trained_spec") / "feature_spec.json"
    return _write_spec(feature_spec_path, out, MEANS, STDS, TEMPERATURE)


@pytest.fixture(scope="module", params=KINDS)
def exported(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory, trained_spec_path: Path
) -> Exported:
    kind = request.param
    spec = load_feature_spec(trained_spec_path)
    wrapper = _make_wrapper(kind, spec)
    out_dir = tmp_path_factory.mktemp(f"export_{kind}")
    onnx_path = export(wrapper, trained_spec_path, out_dir / "drowsiness_v3.onnx", META)
    return Exported(kind, wrapper, spec, onnx_path, out_dir / "feature_spec_v3.json")


# ---------------------------------------------------------------------------
# 계약과 메타데이터
# ---------------------------------------------------------------------------


def test_export_satisfies_contract(exported: Exported) -> None:
    verify_contract(exported.onnx_path, exported.spec_path)

    model = onnx.load(str(exported.onnx_path))
    assert [(entry.domain, entry.version) for entry in model.opset_import] == [("", 17)]
    assert model.ir_version == 8
    assert len(model.graph.value_info) == 0
    assert model.doc_string == "" and model.graph.doc_string == ""
    assert all(node.doc_string == "" for node in model.graph.node)
    # 단일 파일: 폴더에는 모델과 spec 사본뿐이다.
    written = sorted(path.name for path in exported.onnx_path.parent.iterdir())
    assert written == ["drowsiness_v3.onnx", "feature_spec_v3.json"]
    assert exported_spec_path(exported.onnx_path) == exported.spec_path

    session = ort.InferenceSession(str(exported.onnx_path), providers=["CPUExecutionProvider"])
    (graph_input,) = session.get_inputs()
    (graph_output,) = session.get_outputs()
    assert (graph_input.name, graph_input.type, graph_input.shape) == ("features", "tensor(float)", [1, 120, 17])
    assert (graph_output.name, graph_output.type, graph_output.shape) == ("probs", "tensor(float)", [1, 3])


def test_metadata_round_trips_through_onnxruntime(exported: Exported, trained_spec_path: Path) -> None:
    session = ort.InferenceSession(str(exported.onnx_path), providers=["CPUExecutionProvider"])
    metadata = session.get_modelmeta().custom_metadata_map
    assert metadata == {
        "model_version": "3",
        "feature_spec_hash": file_sha256(exported.spec_path),
        "step_sec": "1",
        "window_steps": "120",
        "class_order": "alert,low_vigilant,drowsy",
        "trained_on": "synthetic sessions (test)",
        "commit": "0123abc-dirty",
        "standardization": "in_graph",
        "pipeline_hash": exported.spec.pipeline_hash,
        "mediapipe_model_sha256": exported.spec.pipeline.mediapipe.model_sha256,
    }
    assert len(metadata) == 10 and all(isinstance(value, str) for value in metadata.values())
    assert list(read_metadata(exported.onnx_path)) == list(METADATA_KEYS)
    # 사본은 write_feature_spec이 쓴 원본과 바이트 단위로 같고, 해시는 그 바이트의 sha256이다.
    assert exported.spec_path.read_bytes() == trained_spec_path.read_bytes()
    for key in ("feature_spec_hash", "pipeline_hash", "mediapipe_model_sha256"):
        assert re.fullmatch(r"[0-9a-f]{64}", metadata[key]), key


def test_spec_copy_is_written_in_the_canonical_format(tmp_path: Path, trained_spec_path: Path) -> None:
    # 내용은 같고 형식만 다른 원본(들여쓰기 없음). 사본은 write_feature_spec의 형식으로 쓰이고 해시는 사본의 것이다.
    compact = tmp_path / "compact.json"
    compact.write_text(json.dumps(read_feature_spec_dict(trained_spec_path), ensure_ascii=False), encoding="utf-8")
    assert compact.read_bytes() != trained_spec_path.read_bytes()
    spec = load_feature_spec(compact)
    path = export(_make_wrapper(KIND_GRU, spec), compact, tmp_path / "out" / "m.onnx", META)
    spec_copy = tmp_path / "out" / "feature_spec_v3.json"  # 이름은 모델 파일 이름이 아니라 model_version을 따른다
    assert spec_copy == exported_spec_path(path)
    assert spec_copy.read_bytes() == trained_spec_path.read_bytes()
    assert read_metadata(path)["feature_spec_hash"] == file_sha256(spec_copy) != file_sha256(compact)
    verify_contract(path, spec_copy)


def test_exported_spec_name_follows_the_model_version() -> None:
    assert exported_spec_name(0) == "feature_spec_v0.json"
    assert exported_spec_name("12") == "feature_spec_v12.json"
    for bad in (-1, "v1", "1.0", "", "１"):
        with pytest.raises(ValueError, match="model_version"):
            exported_spec_name(bad)


def test_versions_in_one_folder_keep_their_own_spec_copies(
    tmp_path: Path, feature_spec_path: Path, trained_spec_path: Path, feature_spec: FeatureSpec
) -> None:
    # v0(무작위 가중치, 평균 0·표준편차 1)과 재학습한 v1(다른 평균·표준편차·온도)을 같은 폴더에 내보낸다.
    out_dir = tmp_path / "export"
    v0 = export(
        _make_wrapper(KIND_TCN, feature_spec),
        feature_spec_path,
        out_dir / "drowsiness_v0.onnx",
        {**META, "model_version": "0"},
    )
    trained = load_feature_spec(trained_spec_path)
    v1 = export(
        _make_wrapper(KIND_GRU, trained),
        trained_spec_path,
        out_dir / "drowsiness_v1.onnx",
        {**META, "model_version": "1"},
    )
    assert sorted(path.name for path in out_dir.iterdir()) == [
        "drowsiness_v0.onnx",
        "drowsiness_v1.onnx",
        "feature_spec_v0.json",
        "feature_spec_v1.json",
    ]
    # 뒤에 내보낸 모델이 앞 모델의 사본과 해시를 건드리지 않는다.
    assert exported_spec_path(v0).read_bytes() == feature_spec_path.read_bytes()
    assert exported_spec_path(v1).read_bytes() == trained_spec_path.read_bytes()
    assert read_metadata(v0)["feature_spec_hash"] == file_sha256(out_dir / "feature_spec_v0.json")
    assert read_metadata(v1)["feature_spec_hash"] == file_sha256(out_dir / "feature_spec_v1.json")
    assert read_metadata(v0)["feature_spec_hash"] != read_metadata(v1)["feature_spec_hash"]
    verify_contract(v0, exported_spec_path(v0))
    verify_contract(v1, exported_spec_path(v1))
    # 다른 버전의 사본으로는 계약 검사를 통과하지 못한다.
    with pytest.raises(ContractError, match="metadata feature_spec_hash"):
        verify_contract(v1, exported_spec_path(v0))


def test_exported_spec_path_needs_a_valid_model_version(exported: Exported, tmp_path: Path) -> None:
    model = onnx.load(str(exported.onnx_path))
    kept = [prop for prop in model.metadata_props if prop.key != "model_version"]
    del model.metadata_props[:]
    model.metadata_props.extend(kept)
    broken = tmp_path / "broken.onnx"
    broken.write_bytes(model.SerializeToString())
    with pytest.raises(ContractError, match="model_version"):
        exported_spec_path(broken)


def test_repository_v0_model_ships_with_its_versioned_spec_copy(repo_root: Path, feature_spec_path: Path) -> None:
    export_dir = repo_root / "export"
    model_path = export_dir / "drowsiness_v0.onnx"
    spec_copy = export_dir / "feature_spec_v0.json"
    assert exported_spec_path(model_path) == spec_copy
    verify_contract(model_path, spec_copy)
    verify_file_name(model_path)
    metadata = read_metadata(model_path)
    assert metadata["model_version"] == "0" and metadata["trained_on"] == TRAINED_ON_NONE
    assert metadata["feature_spec_hash"] == file_sha256(spec_copy)
    assert spec_copy.read_bytes() == feature_spec_path.read_bytes()  # v0은 설계값 spec 그대로다
    assert not (export_dir / "feature_spec.json").exists()  # 버전 없는 사본은 더 쓰지 않는다


def test_file_size_is_under_1_mb(exported: Exported) -> None:
    size = exported.onnx_path.stat().st_size
    assert 10_000 < size < 1_000_000
    assert size < 200_000  # 계획서 5.1절의 예상(TCN 130~153 KB, GRU 72 KB)에서 크게 벗어나지 않는다


def test_operator_whitelist_is_exactly_what_the_two_kinds_use(
    tmp_path: Path, trained_spec_path: Path
) -> None:
    spec = load_feature_spec(trained_spec_path)
    used: dict[str, set[str]] = {}
    for kind in KINDS:
        path = export(_make_wrapper(kind, spec), trained_spec_path, tmp_path / kind / "m.onnx", META)
        used[kind] = set(graph_operators(path))
    assert used[KIND_TCN] == {
        "Add", "Clip", "Concat", "Constant", "Conv", "Div", "Gather", "Gemm", "Log", "Mul", "Relu", "Softmax", "Sub",
        "Transpose",
    }  # fmt: skip
    assert used[KIND_GRU] == {
        "Add", "Clip", "Concat", "Constant", "Div", "Expand", "GRU", "Gather", "Gemm", "Log", "Mul", "Relu", "Shape",
        "Softmax", "Sub", "Transpose", "Unsqueeze",
    }  # fmt: skip
    assert used[KIND_TCN] | used[KIND_GRU] == ALLOWED_OPS
    # 배치 정규화는 합성곱에 합쳐지고 드롭아웃은 eval 모드라 그래프에 없다.
    assert not {"BatchNormalization", "Dropout"} & (used[KIND_TCN] | used[KIND_GRU])


def test_graph_constants_equal_spec_values(exported: Exported) -> None:
    model = onnx.load(str(exported.onnx_path))
    constants = {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer}
    assert np.array_equal(constants["mean"], np.asarray(MEANS, dtype=np.float32))
    assert np.array_equal(constants["std"], np.asarray(STDS, dtype=np.float32))
    assert np.flatnonzero(constants["log1p_mask"]).tolist() == LOG1P_COLUMNS
    assert constants["temperature"] == np.float32(TEMPERATURE)


def test_default_commit_comes_from_git(tmp_path: Path, trained_spec_path: Path) -> None:
    spec = load_feature_spec(trained_spec_path)
    meta = {"model_version": "0", "trained_on": TRAINED_ON_NONE}
    path = export(_make_wrapper(KIND_GRU, spec), trained_spec_path, tmp_path / "m.onnx", meta)
    commit = read_metadata(path)["commit"]
    assert commit == git_commit()
    assert re.fullmatch(r"([0-9a-f]{40}|uncommitted)(-dirty)?", commit)


# ---------------------------------------------------------------------------
# 일치 검증, 재현성
# ---------------------------------------------------------------------------


def test_parity_on_random_raw_scale_inputs(exported: Exported) -> None:
    inputs = random_raw_inputs(exported.spec, 200, seed=0)
    assert inputs.shape == (200, 120, 17) and inputs.dtype == np.float32
    assert np.array_equal(inputs, random_raw_inputs(exported.spec, 200, seed=0))
    # 원시 척도: 특징마다 척도가 다르고, 자르기 범위 밖의 값과 log1p 열의 음수가 섞여 있다.
    assert float(inputs[..., 5].std()) > 50.0 * float(inputs[..., 0].std())
    assert bool((np.abs((inputs[..., 0] - MEANS[0]) / STDS[0]) > 5.0).any())
    assert bool((inputs[..., LOG1P_COLUMNS] < 0.0).any())

    assert parity_max_abs_diff(exported.wrapper, exported.onnx_path, inputs) < 1e-4

    probs = np.concatenate([_run(exported.onnx_path, inputs[i : i + 1]) for i in range(20)])
    assert np.all(np.isfinite(probs)) and np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert float(probs.std(axis=0).max()) > 1e-3  # 입력에 따라 출력이 달라진다


def test_parity_helper_rejects_bad_inputs(exported: Exported) -> None:
    with pytest.raises(ValueError, match="shape"):
        parity_max_abs_diff(exported.wrapper, exported.onnx_path, np.zeros((120, 17), dtype=np.float32))
    bad = np.zeros((1, 120, 17), dtype=np.float32)
    bad[0, 3, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        parity_max_abs_diff(exported.wrapper, exported.onnx_path, bad)


def test_reexport_of_the_same_weights_is_byte_identical(
    exported: Exported, tmp_path: Path, trained_spec_path: Path
) -> None:
    again = export(exported.wrapper, trained_spec_path, tmp_path / "a" / "drowsiness_v3.onnx", META)
    assert again.read_bytes() == exported.onnx_path.read_bytes()
    # 같은 시드로 새로 만든 신경망도 같은 파일이 된다.
    rebuilt = _make_wrapper(exported.kind, exported.spec)
    fresh = export(rebuilt, trained_spec_path, tmp_path / "b" / "drowsiness_v3.onnx", META)
    assert fresh.read_bytes() == exported.onnx_path.read_bytes()
    assert exported_spec_path(fresh).read_bytes() == exported.spec_path.read_bytes()
    # 가중치가 다르면 파일도 다르다.
    reseeded = _make_wrapper(exported.kind, exported.spec, seed=1)
    other = export(reseeded, trained_spec_path, tmp_path / "c" / "m.onnx", META)
    assert other.read_bytes() != exported.onnx_path.read_bytes()


def test_export_leaves_the_wrapper_in_training_mode(tmp_path: Path, trained_spec_path: Path) -> None:
    spec = load_feature_spec(trained_spec_path)
    wrapper = _make_wrapper(KIND_TCN, spec, dropout=0.5)
    wrapper.train()
    path = export(wrapper, trained_spec_path, tmp_path / "m.onnx", META)
    assert wrapper.training and wrapper.net.training
    # 내보낸 그래프는 eval 모드의 것이다.
    inputs = random_raw_inputs(spec, 10, seed=3)
    assert parity_max_abs_diff(wrapper, path, inputs) < 1e-4


# ---------------------------------------------------------------------------
# 전처리: log1p 열, 음수, 표준편차 하한
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_log1p_columns_in_the_graph_and_negative_raw_values(kind: str, tmp_path: Path, feature_spec_path: Path) -> None:
    spec = load_feature_spec(feature_spec_path)  # v0: 평균 0, 표준편차 1이라 아래 값들이 자르기 범위 안에 있다
    wrapper = _make_wrapper(kind, spec)
    path = export(wrapper, feature_spec_path, tmp_path / "m.onnx", META)

    base = _run(path, np.zeros((1, 120, 17)))
    unchanged = []
    for column in range(17):
        window = np.zeros((1, 120, 17))
        window[0, :, column] = -3.0
        probs = _run(path, window)
        assert np.all(np.isfinite(probs))
        if np.array_equal(probs, base):
            unchanged.append(column)
    # log1p(max(x, 0))가 걸린 열만 음수가 0과 같은 결과를 낸다.
    assert unchanged == LOG1P_COLUMNS

    extreme = np.full((1, 120, 17), -1.0e6)
    probs = _run(path, extreme)
    assert np.all(np.isfinite(probs)) and abs(float(probs.sum()) - 1.0) < 1e-5
    assert parity_max_abs_diff(wrapper, path, extreme) < 1e-4


@pytest.mark.parametrize("kind", KINDS)
def test_std_zero_feature_exports_with_the_floor_and_gives_finite_probabilities(
    kind: str, tmp_path: Path, feature_spec_path: Path, feature_spec: FeatureSpec
) -> None:
    floor = feature_spec.standardization.std_floor
    assert floor == 0.001
    stds = list(STDS)
    stds[12] = 0.0  # 학습 데이터에서 분산이 0인 특징
    # 메모리의 spec은 std 0을 그대로 갖고, 래퍼가 하한을 적용한다.
    in_memory = dataclasses.replace(
        feature_spec,
        features=tuple(
            dataclasses.replace(feat, mean=mean, std=std) for feat, mean, std in zip(feature_spec.features, MEANS, stds)
        ),
    )
    wrapper = _make_wrapper(kind, in_memory)
    assert wrapper.std[12].item() == np.float32(floor)

    # 파일에는 하한을 적용한 값을 쓴다(spec 검증은 하한 아래의 std를 받지 않는다).
    with pytest.raises(SpecError, match="std_floor"):
        _write_spec(feature_spec_path, tmp_path / "bad.json", MEANS, stds, 1.0)
    spec_path = _write_spec(feature_spec_path, tmp_path / "spec.json", MEANS, [max(s, floor) for s in stds], 1.0)

    path = export(wrapper, spec_path, tmp_path / "out" / "m.onnx", META)
    model = onnx.load(str(path))
    std = next(numpy_helper.to_array(tensor) for tensor in model.graph.initializer if tensor.name == "std")
    assert std[12] == np.float32(floor) and bool(np.all(std >= np.float32(floor)))

    spec = load_feature_spec(spec_path)
    inputs = random_raw_inputs(spec, 20, seed=5)
    inputs[0, :, 12] = MEANS[12]  # 하한이 없으면 0 / 0
    inputs[1, :, 12] = 1.0e6
    inputs[2, :, 12] = -1.0e6
    for index in range(inputs.shape[0]):
        probs = _run(path, inputs[index : index + 1])
        assert np.all(np.isfinite(probs)) and abs(float(probs.sum()) - 1.0) < 1e-5
    assert parity_max_abs_diff(wrapper, path, inputs) < 1e-4


# ---------------------------------------------------------------------------
# export()의 입력 검사
# ---------------------------------------------------------------------------


def test_export_rejects_a_wrapper_that_does_not_match_the_spec_file(
    tmp_path: Path, trained_spec_path: Path, feature_spec: FeatureSpec
) -> None:
    trained = load_feature_spec(trained_spec_path)
    out = tmp_path / "out" / "m.onnx"
    with pytest.raises(ValueError, match="ExportWrapper.mean does not match"):
        export(_make_wrapper(KIND_GRU, feature_spec), trained_spec_path, out, META)
    wrong_temperature = ExportWrapper(build_model(KIND_GRU, 17), trained, 1.0)
    with pytest.raises(ValueError, match="temperature"):
        export(wrong_temperature, trained_spec_path, out, META)
    assert not out.parent.exists()  # 아무것도 쓰지 않았다


@pytest.mark.parametrize(
    "meta",
    [
        {"model_version": "1"},
        {"trained_on": "x"},
        {"model_version": "1", "trained_on": "x", "step_sec": "1"},
        {"model_version": "v1", "trained_on": "x"},
        {"model_version": "-1", "trained_on": "x"},
        {"model_version": "1", "trained_on": ""},
    ],
)
def test_export_rejects_bad_meta(
    meta: dict[str, str], tmp_path: Path, feature_spec_path: Path, feature_spec: FeatureSpec
) -> None:
    wrapper = ExportWrapper(build_model(KIND_GRU, 17), feature_spec, 1.0)
    with pytest.raises(ValueError, match="meta"):
        export(wrapper, feature_spec_path, tmp_path / "out" / "m.onnx", meta)
    assert not (tmp_path / "out").exists()


def test_export_removes_a_file_that_fails_the_contract(
    tmp_path: Path, trained_spec_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = load_feature_spec(trained_spec_path)
    monkeypatch.setattr(export_onnx, "MAX_FILE_BYTES", 1000)
    out = tmp_path / "m.onnx"
    with pytest.raises(ContractError, match="must be under 1000"):
        export(_make_wrapper(KIND_GRU, spec), trained_spec_path, out, META)
    assert not out.exists()
    # 모델도, 짝이 없는 spec 사본도, 임시 파일도 남지 않는다.
    assert list(tmp_path.iterdir()) == []
    # 이 호출이 만든 폴더도 남기지 않는다.
    with pytest.raises(ContractError, match="must be under 1000"):
        export(_make_wrapper(KIND_GRU, spec), trained_spec_path, tmp_path / "new" / "m.onnx", META)
    assert list(tmp_path.iterdir()) == []


class _TanhNet(nn.Module):
    """허용 목록에 없는 연산자(Tanh)를 쓰는 신경망. 내보내면 계약 검사에서 걸린다."""

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(17, 3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.head(z[:, -1, :]))


def _folder_bytes(folder: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(folder.iterdir())}


def test_failed_reexport_leaves_the_previous_pair_byte_identical(
    tmp_path: Path, trained_spec_path: Path, feature_spec_path: Path, feature_spec: FeatureSpec
) -> None:
    # 버전 3의 유효한 쌍(학습한 spec, TCN)과 그 INT8 파일이 이미 폴더에 있다.
    out_dir = tmp_path / "export"
    trained = load_feature_spec(trained_spec_path)
    model_path = export(_make_wrapper(KIND_TCN, trained), trained_spec_path, out_dir / "drowsiness_v3.onnx", META)
    quantize_int8(model_path, out_dir / "drowsiness_v3_int8.onnx")
    before = _folder_bytes(out_dir)
    assert sorted(before) == ["drowsiness_v3.onnx", "drowsiness_v3_int8.onnx", "feature_spec_v3.json"]

    # 같은 버전을 다른 spec(v0 설계값)과 계약을 어기는 신경망으로 다시 내보낸다.
    assert feature_spec_path.read_bytes() != before["feature_spec_v3.json"]
    bad = ExportWrapper(_TanhNet(), feature_spec, feature_spec.standardization.temperature)
    with pytest.raises(ContractError, match=r"operators not in the allowed list: \['Tanh'\]"):
        export(bad, feature_spec_path, out_dir / "drowsiness_v3.onnx", META)

    # 모델, spec 사본, INT8 파일 모두 바이트 단위로 그대로이고 임시 파일은 남지 않았다.
    assert _folder_bytes(out_dir) == before
    verify_contract(model_path, exported_spec_path(model_path))
    verify_contract(out_dir / "drowsiness_v3_int8.onnx", out_dir / "feature_spec_v3.json", ALLOWED_OPS_INT8)


def test_export_interrupted_before_the_swap_leaves_the_previous_pair(
    tmp_path: Path, trained_spec_path: Path, feature_spec_path: Path, feature_spec: FeatureSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    out_dir = tmp_path / "export"
    trained = load_feature_spec(trained_spec_path)
    export(_make_wrapper(KIND_GRU, trained), trained_spec_path, out_dir / "drowsiness_v3.onnx", META)
    before = _folder_bytes(out_dir)

    seen: list[list[str]] = []

    def interrupted(onnx_path: Path, spec_path: Path, allowed_ops: frozenset[str]) -> None:
        # 검사 시점에 새 파일은 임시 이름으로만 있고, 제자리의 쌍은 아직 예전 것이다.
        seen.append(sorted(path.name for path in out_dir.iterdir()))
        assert onnx_path.name not in before and spec_path.name not in before
        assert spec_path.read_bytes() == feature_spec_path.read_bytes()
        assert _folder_bytes(out_dir)["feature_spec_v3.json"] == before["feature_spec_v3.json"]
        raise KeyboardInterrupt

    monkeypatch.setattr(export_onnx, "verify_contract", interrupted)
    with pytest.raises(KeyboardInterrupt):
        export(_make_wrapper(KIND_GRU, feature_spec), feature_spec_path, out_dir / "drowsiness_v3.onnx", META)
    assert len(seen) == 1 and len(seen[0]) == 4  # 예전 쌍과 임시 파일 둘
    assert _folder_bytes(out_dir) == before


def test_successful_reexport_replaces_the_pair_and_removes_the_stale_int8_file(
    tmp_path: Path, trained_spec_path: Path, feature_spec_path: Path, feature_spec: FeatureSpec
) -> None:
    out_dir = tmp_path / "export"
    trained = load_feature_spec(trained_spec_path)
    first = export(_make_wrapper(KIND_TCN, trained), trained_spec_path, out_dir / "drowsiness_v3.onnx", META)
    int8_path = quantize_int8(first, out_dir / "drowsiness_v3_int8.onnx")
    other_version = export(
        _make_wrapper(KIND_TCN, trained), trained_spec_path, out_dir / "drowsiness_v4.onnx",
        {**META, "model_version": "4"},
    )  # fmt: skip
    other_int8 = quantize_int8(other_version, out_dir / "drowsiness_v4_int8.onnx")
    old_int8_bytes = int8_path.read_bytes()

    # 같은 버전을 GRU와 다른 spec으로 다시 내보낸다. 예전 INT8 파일은 예전 가중치와 예전 사본의 것이다.
    second = export(_make_wrapper(KIND_GRU, feature_spec), feature_spec_path, out_dir / "drowsiness_v3.onnx", META)
    assert sorted(path.name for path in out_dir.iterdir()) == [
        "drowsiness_v3.onnx",
        "drowsiness_v4.onnx",
        "drowsiness_v4_int8.onnx",
        "feature_spec_v3.json",
        "feature_spec_v4.json",
    ]
    assert (out_dir / "feature_spec_v3.json").read_bytes() == feature_spec_path.read_bytes()
    assert "GRU" in graph_operators(second)
    verify_contract(second, exported_spec_path(second))
    # 지운 파일은 새 사본과 맞지 않는 것이었다(되살려 놓으면 계약 검사가 해시에서 걸린다).
    int8_path.write_bytes(old_int8_bytes)
    with pytest.raises(ContractError, match="metadata feature_spec_hash"):
        verify_contract(int8_path, out_dir / "feature_spec_v3.json", ALLOWED_OPS_INT8)
    # 다른 버전의 INT8 파일은 건드리지 않는다.
    verify_contract(other_int8, out_dir / "feature_spec_v4.json", ALLOWED_OPS_INT8)


def test_export_reads_the_git_state_before_it_writes_anything(
    tmp_path: Path, trained_spec_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "export"
    listings: list[bool] = []

    def recording_git_commit() -> str:
        listings.append(out_dir.exists())
        return "0" * 40

    monkeypatch.setattr(export_onnx, "git_commit", recording_git_commit)
    spec = load_feature_spec(trained_spec_path)
    meta = {"model_version": "5", "trained_on": "x"}
    path = export(_make_wrapper(KIND_GRU, spec), trained_spec_path, out_dir / "drowsiness_v5.onnx", meta)
    # git 상태를 읽을 때 출력 폴더는 아직 없었다(사본도 모델도 임시 파일도 쓰기 전이다).
    assert listings == [False]
    assert read_metadata(path)["commit"] == "0" * 40
    # commit을 주면 git을 부르지 않는다.
    export(_make_wrapper(KIND_GRU, spec), trained_spec_path, out_dir / "drowsiness_v5.onnx", {**meta, "commit": "c"})
    assert listings == [False]


# ---------------------------------------------------------------------------
# verify_contract가 위반을 잡는지
# ---------------------------------------------------------------------------


def _set_opset_18(model: onnx.ModelProto) -> None:
    model.opset_import[0].version = 18


def _set_ir_9(model: onnx.ModelProto) -> None:
    model.ir_version = 9


def _drop_commit(model: onnx.ModelProto) -> None:
    kept = [prop for prop in model.metadata_props if prop.key != "commit"]
    del model.metadata_props[:]
    model.metadata_props.extend(kept)


def _add_metadata_key(model: onnx.ModelProto) -> None:
    model.metadata_props.add(key="onnx.infer", value="onnxruntime.quant")


def _set_metadata(key: str, value: str) -> Callable[[onnx.ModelProto], None]:
    def mutate(model: onnx.ModelProto) -> None:
        for prop in model.metadata_props:
            if prop.key == key:
                prop.value = value

    return mutate


def _rename_input(model: onnx.ModelProto) -> None:
    model.graph.input[0].name = "x"
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == "features":
                node.input[index] = "x"


def _dynamic_batch(model: onnx.ModelProto) -> None:
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch"


def _append_identity(model: onnx.ModelProto) -> None:
    producer = next(node for node in model.graph.node if "probs" in node.output)
    producer.output[list(producer.output).index("probs")] = "probs_before_identity"
    model.graph.node.append(onnx.helper.make_node("Identity", ["probs_before_identity"], ["probs"]))


def _scale_initializer(name: str, index: int | None, factor: float) -> Callable[[onnx.ModelProto], None]:
    def mutate(model: onnx.ModelProto) -> None:
        tensor = next(item for item in model.graph.initializer if item.name == name)
        values = numpy_helper.to_array(tensor).copy()
        if index is None:
            values = values * np.float32(factor)
        else:
            values[index] *= np.float32(factor)
        tensor.CopyFrom(numpy_helper.from_array(values.astype(np.float32), name))

    return mutate


def _add_big_initializer(model: onnx.ModelProto) -> None:
    model.graph.initializer.append(numpy_helper.from_array(np.zeros(250_000, dtype=np.float32), "padding"))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_set_opset_18, "opset_import must be exactly"),
        (_set_ir_9, "ir_version must be 8"),
        (_drop_commit, "metadata keys must be"),
        (_add_metadata_key, "metadata keys must be"),
        (_set_metadata("pipeline_hash", "0" * 64), "metadata pipeline_hash"),
        (_set_metadata("class_order", "alert,drowsy,low_vigilant"), "metadata class_order"),
        (_set_metadata("window_steps", "60"), "metadata window_steps"),
        (_set_metadata("standardization", "external"), "metadata standardization"),
        (_set_metadata("mediapipe_model_sha256", "f" * 64), "metadata mediapipe_model_sha256"),
        (_set_metadata("model_version", "v3"), "model_version must be a non-negative integer"),
        (_set_metadata("trained_on", ""), "trained_on and commit must be set"),
        (_rename_input, "graph input must be named 'features'"),
        (_dynamic_batch, "features: shape must be"),
        (_append_identity, r"operators not in the allowed list: \['Identity'\]"),
        (_scale_initializer("mean", 4, 1.01), "mean in the graph does not match"),
        (_scale_initializer("std", 3, 0.001), "std in the graph does not match"),
        (_scale_initializer("log1p_mask", 9, 0.0), "log1p mask in the graph does not match"),
        (_scale_initializer("temperature", None, 2.0), "temperature in the graph does not match"),
        (_add_big_initializer, "must be under 1000000"),
    ],
    ids=[
        "opset18", "ir9", "missing_key", "extra_key", "pipeline_hash", "class_order", "window_steps", "standardization",
        "mediapipe_sha", "model_version", "trained_on", "input_name", "dynamic_batch", "extra_operator", "mean", "std",
        "log1p_mask", "temperature", "file_size",
    ],  # fmt: skip
)
def test_verify_contract_detects_violations(
    mutate: Callable[[onnx.ModelProto], None], message: str, exported: Exported, tmp_path: Path
) -> None:
    model = onnx.load(str(exported.onnx_path))
    mutate(model)
    broken = tmp_path / "broken.onnx"
    broken.write_bytes(model.SerializeToString())
    spec_copy = tmp_path / exported_spec_name(3)
    shutil.copyfile(exported.spec_path, spec_copy)
    with pytest.raises(ContractError, match=message):
        verify_contract(broken, spec_copy)


def test_verify_contract_detects_spec_file_changes(exported: Exported, tmp_path: Path, feature_spec_path: Path) -> None:
    model_copy = tmp_path / "m.onnx"
    shutil.copyfile(exported.onnx_path, model_copy)
    # 내용은 같고 바이트만 다른 spec(끝에 줄바꿈 하나 추가)
    spec_copy = tmp_path / exported_spec_name(3)
    spec_copy.write_bytes(exported.spec_path.read_bytes() + b"\n")
    with pytest.raises(ContractError, match="metadata feature_spec_hash"):
        verify_contract(model_copy, spec_copy)
    # 다른 spec(v0)
    with pytest.raises(ContractError, match="metadata feature_spec_hash"):
        verify_contract(model_copy, feature_spec_path)


def test_verify_contract_detects_external_data(exported: Exported, tmp_path: Path) -> None:
    model = onnx.load(str(exported.onnx_path))
    broken = tmp_path / "broken.onnx"
    onnx.save_model(
        model, str(broken), save_as_external_data=True, all_tensors_to_one_file=True, location="broken.onnx.data",
        size_threshold=0,
    )  # fmt: skip
    assert (tmp_path / "broken.onnx.data").exists()
    spec_copy = tmp_path / exported_spec_name(3)
    shutil.copyfile(exported.spec_path, spec_copy)
    with pytest.raises(ContractError, match="external data"):
        verify_contract(broken, spec_copy)


# ---------------------------------------------------------------------------
# verify_contract: 전처리·후처리의 연결 (연산자 목록과 상수만으로는 잡히지 않는 변형)
# ---------------------------------------------------------------------------


def _node(model: onnx.ModelProto, op_type: str, reads: str) -> onnx.NodeProto:
    """`reads`를 입력으로 읽는 op_type 노드. 정확히 하나여야 한다."""
    found = [node for node in model.graph.node if node.op_type == op_type and reads in node.input]
    assert len(found) == 1, (op_type, reads, len(found))
    return found[0]


def _rewire(model: onnx.ModelProto, old: str, new: str, keep: tuple[str, ...] = ()) -> int:
    """`old`를 읽는 입력을 `new`로 바꾼다. 출력 이름이 keep에 있는 노드는 그대로 둔다. 바꾼 자리 수를 돌려준다."""
    changed = 0
    for node in model.graph.node:
        if node.output[0] in keep:
            continue
        for index, name in enumerate(node.input):
            if name == old:
                node.input[index] = new
                changed += 1
    return changed


def _drop_max_before_log(model: onnx.ModelProto) -> None:
    # log1p(max(x, 0)) -> log1p(x): Relu를 없애고 Add가 입력을 바로 읽게 한다.
    relu = _node(model, "Relu", "features")
    assert _rewire(model, relu.output[0], "features") == 1
    model.graph.node.remove(relu)


def _log_of_x_plus_two(model: onnx.ModelProto) -> None:
    # log(max(x, 0) + 1) -> log(max(x, 0) + 2): 상수만 바꾼다(TorchScript는 Constant 노드, dynamo는 initializer).
    relu = _node(model, "Relu", "features")
    add = _node(model, "Add", relu.output[0])
    constant_name = next(name for name in add.input if name != relu.output[0])
    two = np.asarray(2.0, dtype=np.float32)
    replaced = 0
    for node in model.graph.node:
        if node.op_type == "Constant" and node.output[0] == constant_name:
            for attribute in node.attribute:
                if attribute.name == "value":
                    attribute.t.CopyFrom(numpy_helper.from_array(two))
                    replaced += 1
    for tensor in model.graph.initializer:
        if tensor.name == constant_name:
            tensor.CopyFrom(numpy_helper.from_array(two, constant_name))
            replaced += 1
    assert replaced == 1


def _reverse_time_at_input(model: onnx.ModelProto) -> None:
    # 입력의 시간축을 뒤집는 Gather를 맨 앞에 끼운다(Gather는 허용 연산자다).
    steps = numpy_helper.from_array(np.arange(119, -1, -1, dtype=np.int64), "reversed_steps")
    model.graph.initializer.append(steps)
    assert _rewire(model, "features", "features_reversed") == 2
    gather = onnx.helper.make_node("Gather", ["features", "reversed_steps"], ["features_reversed"], axis=1)
    model.graph.node.insert(0, gather)


def _permute_classes_after_softmax(model: onnx.ModelProto) -> None:
    # 출력 클래스 순서를 drowsy, low_vigilant, alert로 바꾼다.
    softmax = next(node for node in model.graph.node if node.op_type == "Softmax")
    softmax.output[0] = "probs_before_permutation"
    model.graph.initializer.append(numpy_helper.from_array(np.array([2, 1, 0], dtype=np.int64), "class_permutation"))
    model.graph.node.append(
        onnx.helper.make_node("Gather", ["probs_before_permutation", "class_permutation"], ["probs"], axis=1)
    )


def _softmax_over_the_batch_axis(model: onnx.ModelProto) -> None:
    softmax = next(node for node in model.graph.node if node.op_type == "Softmax")
    del softmax.attribute[:]
    softmax.attribute.append(onnx.helper.make_attribute("axis", 0))


def _second_softmax(model: onnx.ModelProto) -> None:
    softmax = next(node for node in model.graph.node if node.op_type == "Softmax")
    softmax.output[0] = "probs_once"
    model.graph.node.append(onnx.helper.make_node("Softmax", ["probs_once"], ["probs"], axis=-1))


def _raw_features_bypass_the_preprocessing(model: onnx.ModelProto) -> None:
    # 신경망이 표준화한 값에 원시 입력을 더한 것을 읽게 한다.
    clip = next(node for node in model.graph.node if node.op_type == "Clip")
    position = [node.output[0] for node in model.graph.node].index(clip.output[0])
    assert _rewire(model, clip.output[0], "mixed") >= 1
    model.graph.node.insert(position + 1, onnx.helper.make_node("Add", [clip.output[0], "features"], ["mixed"]))


def _network_reads_the_unclipped_features(model: onnx.ModelProto) -> None:
    clip = next(node for node in model.graph.node if node.op_type == "Clip")
    assert _rewire(model, clip.output[0], clip.input[0]) >= 1


def _drop_log(model: onnx.ModelProto) -> None:
    # log1p(max(x, 0)) -> max(x, 0) + 1
    log = next(node for node in model.graph.node if node.op_type == "Log")
    assert _rewire(model, log.output[0], log.input[0]) == 1
    model.graph.node.remove(log)


def _drop_the_log1p_branch(model: onnx.ModelProto) -> None:
    # 평균을 빼는 Sub가 두 곱의 합이 아니라 keep 마스크 쪽 곱만 읽는다(log1p 열이 0이 된다).
    _node(model, "Sub", "mean").input[0] = _node(model, "Mul", "keep_mask").output[0]


def _keep_mask_on_rectified_features(model: onnx.ModelProto) -> None:
    # log1p가 아닌 열에도 max(x, 0)이 걸린다.
    _node(model, "Mul", "keep_mask").input[0] = _node(model, "Relu", "features").output[0]


# (이름, 변형, verify_contract가 내야 하는 메시지). 앞의 넷은 검토에서 계약 검사를 통과했던 변형이다.
GRAPH_MUTATIONS: list[tuple[str, Callable[[onnx.ModelProto], None], str]] = [
    ("no_max_before_log", _drop_max_before_log, "log1p constant is added to must be the output of one Relu node"),
    ("log_of_x_plus_two", _log_of_x_plus_two, "constant added before the log1p Log must be float32 1.0"),
    ("time_reversed", _reverse_time_at_input, "keep-mask Mul must read the graph input 'features'"),
    ("classes_permuted", _permute_classes_after_softmax, "Softmax output must be the graph output 'probs'"),
    ("softmax_axis_0", _softmax_over_the_batch_axis, r"Softmax must run over the class axis \(-1\), got axis 0"),
    ("second_softmax", _second_softmax, "expected exactly one Softmax node, got 2"),
    ("raw_bypass", _raw_features_bypass_the_preprocessing, "graph input 'features' must be read only by"),
    ("unclipped", _network_reads_the_unclipped_features, "the standardized features must feed only the next step"),
    ("no_log", _drop_log, "multiplied by the log1p mask must be the output of one Log node"),
    ("no_log1p_branch", _drop_the_log1p_branch, "mean is subtracted from must be the output of one Add node"),
    ("keep_mask_on_relu", _keep_mask_on_rectified_features, "keep-mask Mul must read the graph input 'features'"),
]


def _assert_mutation_changes_behaviour_and_is_rejected(
    model_path: Path, spec_path: Path, allowed_ops: frozenset[str], mutate: Callable[[onnx.ModelProto], None],
    message: str, work_dir: Path,
) -> None:  # fmt: skip
    spec = load_feature_spec(spec_path)
    model = onnx.load(str(model_path))
    mutate(model)
    broken = work_dir / "broken.onnx"
    broken.write_bytes(model.SerializeToString())
    spec_copy = work_dir / spec_path.name
    shutil.copyfile(spec_path, spec_copy)

    # 변형한 그래프는 onnxruntime이 그대로 실행하는 모델이고 출력이 원본과 다르다(NaN이거나 값이 다르다).
    windows = random_raw_inputs(spec, 3, seed=11)  # 자르기 범위 밖의 값과 음수가 섞여 있다
    assert bool((np.abs((windows - np.asarray(spec.means)) / np.asarray(spec.stds)) > 5.0).any())
    assert bool((windows < 0.0).any())
    original = np.concatenate([_run(model_path, windows[i : i + 1]) for i in range(len(windows))])
    changed = np.concatenate([_run(broken, windows[i : i + 1]) for i in range(len(windows))])
    assert np.all(np.isfinite(original))
    assert bool(np.isnan(changed).any()) or float(np.max(np.abs(changed - original))) > 1e-4

    with pytest.raises(ContractError, match=message):
        verify_contract(broken, spec_copy, allowed_ops)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [(mutate, message) for _, mutate, message in GRAPH_MUTATIONS],
    ids=[name for name, _, _ in GRAPH_MUTATIONS],
)
def test_verify_contract_detects_changed_pre_and_post_processing(
    mutate: Callable[[onnx.ModelProto], None], message: str, exported: Exported, tmp_path: Path
) -> None:
    _assert_mutation_changes_behaviour_and_is_rejected(
        exported.onnx_path, exported.spec_path, ALLOWED_OPS, mutate, message, tmp_path
    )


@pytest.fixture(scope="module")
def other_path_models(
    tmp_path_factory: pytest.TempPathFactory, trained_spec_path: Path
) -> dict[str, tuple[Path, Path, frozenset[str]]]:
    """기본 경로 말고 나머지 내보내기 경로의 파일: dynamo(TCN, GRU)와 INT8(TCN). 이름 -> (모델, spec 사본, 허용 목록)."""
    pytest.importorskip("onnxscript")
    spec = load_feature_spec(trained_spec_path)
    out = tmp_path_factory.mktemp("other_paths")
    models: dict[str, tuple[Path, Path, frozenset[str]]] = {}
    for kind in KINDS:
        out_path = out / f"dynamo_{kind}" / "m.onnx"
        path = export(_make_wrapper(kind, spec), trained_spec_path, out_path, META, dynamo=True)
        models[f"dynamo_{kind}"] = (path, exported_spec_path(path), ALLOWED_OPS_DYNAMO)
    float_path = export(_make_wrapper(KIND_TCN, spec), trained_spec_path, out / "int8" / "drowsiness_v3.onnx", META)
    int8_path = quantize_int8(float_path, out / "int8" / "drowsiness_v3_int8.onnx")
    models["int8_tcn"] = (int8_path, exported_spec_path(int8_path), ALLOWED_OPS_INT8)
    return models


@pytest.mark.parametrize("path_name", ["dynamo_tcn", "dynamo_gru", "int8_tcn"])
def test_pre_and_post_processing_checks_hold_on_the_dynamo_and_int8_files(
    path_name: str, other_path_models: dict[str, tuple[Path, Path, frozenset[str]]], tmp_path: Path
) -> None:
    model_path, spec_path, allowed_ops = other_path_models[path_name]
    verify_contract(model_path, spec_path, allowed_ops)  # 변형하지 않은 파일은 통과한다
    for name, mutate, message in GRAPH_MUTATIONS:
        work_dir = tmp_path / name
        work_dir.mkdir()
        _assert_mutation_changes_behaviour_and_is_rejected(
            model_path, spec_path, allowed_ops, mutate, message, work_dir
        )


# ---------------------------------------------------------------------------
# verify_contract: 온도 (로짓을 `temperature`가 아닌 상수로 나누거나 곱하는 그래프)
# ---------------------------------------------------------------------------


def _rename_temperature_and_change_it(model: onnx.ModelProto) -> None:
    # `temperature` 상수의 이름을 `tau`로 바꾸고 값을 5.0으로 둔다: 로짓을 spec에 없는 값으로 나누는 그래프가 된다.
    tensor = next(item for item in model.graph.initializer if item.name == "temperature")
    assert _rewire(model, "temperature", "tau") == 1
    model.graph.initializer.remove(tensor)
    model.graph.initializer.append(numpy_helper.from_array(np.asarray(5.0, dtype=np.float32), "tau"))


def _scale_logits(op_type: str, value: float) -> Callable[[onnx.ModelProto], None]:
    # Softmax 바로 앞에 새 상수 `tau`로 나누거나 곱하는 노드를 끼운다(Div와 Mul은 허용 연산자다).
    def mutate(model: onnx.ModelProto) -> None:
        softmax = next(node for node in model.graph.node if node.op_type == "Softmax")
        model.graph.initializer.append(numpy_helper.from_array(np.asarray(value, dtype=np.float32), "tau"))
        logits = softmax.input[0]
        softmax.input[0] = "scaled_logits"
        position = [node.output[0] for node in model.graph.node].index(softmax.output[0])
        model.graph.node.insert(position, onnx.helper.make_node(op_type, [logits, "tau"], ["scaled_logits"]))

    return mutate


def _scale_logits_before_the_temperature(op_type: str, value: float) -> Callable[[onnx.ModelProto], None]:
    # 온도로 나누는 Div 앞에 새 상수 `tau`로 나누거나 곱하는 노드를 끼운다. 온도 Div와 Softmax는 그대로다.
    def mutate(model: onnx.ModelProto) -> None:
        division = _node(model, "Div", "temperature")
        model.graph.initializer.append(numpy_helper.from_array(np.asarray(value, dtype=np.float32), "tau"))
        logits = division.input[0]
        division.input[0] = "scaled_logits"
        position = [node.output[0] for node in model.graph.node].index(division.output[0])
        model.graph.node.insert(position, onnx.helper.make_node(op_type, [logits, "tau"], ["scaled_logits"]))

    return mutate


def _drop_the_temperature_division(model: onnx.ModelProto) -> None:
    # 온도로 나누는 Div와 그 상수를 지우고 Softmax가 로짓을 바로 읽게 한다. 온도가 1.0이면 출력은 그대로다.
    division = _node(model, "Div", "temperature")
    assert _rewire(model, division.output[0], division.input[0]) == 1
    model.graph.node.remove(division)
    model.graph.initializer.remove(next(item for item in model.graph.initializer if item.name == "temperature"))


NO_TEMPERATURE_CONSTANT = "the graph has no constant named 'temperature'"
SCALED_LOGITS = "the logits are the output of a {} node: only the division by the 'temperature' constant may scale them"


@pytest.fixture(scope="module")
def temperature_one_models(
    tmp_path_factory: pytest.TempPathFactory, feature_spec_path: Path
) -> dict[str, tuple[Path, Path, frozenset[str]]]:
    """spec의 온도가 정확히 1.0인 파일들: 기본 경로(TCN, GRU), INT8(TCN), dynamo(TCN, GRU). 이름 -> (모델, 사본, 허용 목록).

    v0과, 맞춘 온도가 1.0으로 나온 학습 모델이 이 경우다.
    """
    pytest.importorskip("onnxscript")
    out = tmp_path_factory.mktemp("temperature_one")
    spec_path = _write_spec(feature_spec_path, out / "trained.json", MEANS, STDS, 1.0)
    spec = load_feature_spec(spec_path)
    assert spec.standardization.temperature == 1.0
    models: dict[str, tuple[Path, Path, frozenset[str]]] = {}
    for kind in KINDS:
        path = export(_make_wrapper(kind, spec), spec_path, out / f"default_{kind}" / "drowsiness_v3.onnx", META)
        models[f"default_{kind}"] = (path, exported_spec_path(path), ALLOWED_OPS)
        path = export(_make_wrapper(kind, spec), spec_path, out / f"dynamo_{kind}" / "m.onnx", META, dynamo=True)
        models[f"dynamo_{kind}"] = (path, exported_spec_path(path), ALLOWED_OPS_DYNAMO)
    float_path = models[f"default_{KIND_TCN}"][0]
    int8_path = quantize_int8(float_path, out / "int8" / "drowsiness_v3_int8.onnx")
    models["int8_tcn"] = (int8_path, exported_spec_path(float_path), ALLOWED_OPS_INT8)
    return models


def _initializer_names(model_path: Path) -> set[str]:
    return {tensor.name for tensor in onnx.load(str(model_path)).graph.initializer}


@pytest.mark.parametrize("name", ["default_tcn", "default_gru", "int8_tcn"])
def test_renamed_temperature_constant_is_rejected_when_the_spec_temperature_is_one(
    name: str, temperature_one_models: dict[str, tuple[Path, Path, frozenset[str]]], tmp_path: Path
) -> None:
    """기본 경로와 INT8 파일은 온도가 1.0이어도 `temperature` 상수로 나눈다. 그 상수가 없는 그래프는 받지 않는다.

    전에는 온도 상수가 없으면 spec의 온도가 1.0인지만 봐서, 상수 이름을 바꿔 5.0으로 나누게 고친 파일이 통과했다.
    """
    model_path, spec_path, allowed_ops = temperature_one_models[name]
    assert "temperature" in _initializer_names(model_path)  # 이 경로의 내보내기는 1.0으로 나누는 노드를 지우지 않는다
    verify_contract(model_path, spec_path, allowed_ops)  # 고치지 않은 파일은 통과한다

    renamed = tmp_path / "renamed"
    renamed.mkdir()
    _assert_mutation_changes_behaviour_and_is_rejected(
        model_path, spec_path, allowed_ops, _rename_temperature_and_change_it, NO_TEMPERATURE_CONSTANT, renamed
    )
    # 이름을 지키고 값만 바꾼 것과, 온도 Div와 Softmax 사이에 다른 상수를 끼운 것은 전부터 잡혔다. 온도 Div 앞에 끼운 것도
    # 잡는다(온도 상수는 그대로 1.0인데 로짓은 다른 상수로 나뉘거나 곱해진다).
    for folder, mutate, message in (
        ("value", _scale_initializer("temperature", None, 5.0), "temperature in the graph does not match"),
        ("extra_div", _scale_logits("Div", 5.0), "the temperature division must feed the Softmax node"),
        ("extra_mul", _scale_logits("Mul", 0.2), "the temperature division must feed the Softmax node"),
        ("div_before", _scale_logits_before_the_temperature("Div", 5.0), SCALED_LOGITS.format("Div")),
        ("mul_before", _scale_logits_before_the_temperature("Mul", 0.2), SCALED_LOGITS.format("Mul")),
    ):
        (tmp_path / folder).mkdir()
        _assert_mutation_changes_behaviour_and_is_rejected(
            model_path, spec_path, allowed_ops, mutate, message, tmp_path / folder
        )

    # 1.0으로 나누는 노드를 지운 그래프는 출력이 같다. 그래도 이 경로의 내보내기가 만드는 모양이 아니므로 받지 않는다
    # (상수가 없어도 되는 것은 그 노드를 실제로 지우는 dynamo 경로뿐이다).
    model = onnx.load(str(model_path))
    _drop_the_temperature_division(model)
    dropped = tmp_path / "dropped.onnx"
    dropped.write_bytes(model.SerializeToString())
    window = random_raw_inputs(load_feature_spec(spec_path), 1, seed=5)
    assert np.array_equal(_run(dropped, window), _run(model_path, window))
    with pytest.raises(ContractError, match=NO_TEMPERATURE_CONSTANT):
        verify_contract(dropped, spec_path, allowed_ops)


@pytest.mark.parametrize("kind", KINDS)
def test_dynamo_file_without_a_temperature_constant_must_not_scale_the_logits(
    kind: str, temperature_one_models: dict[str, tuple[Path, Path, frozenset[str]]], tmp_path: Path
) -> None:
    """dynamo 경로는 온도가 1.0이면 나눗셈과 상수를 지운다. 그 파일만 상수 없이 통과하고, 로짓을 다른 상수로 나누거나 곱하면 걸린다."""
    model_path, spec_path, allowed_ops = temperature_one_models[f"dynamo_{kind}"]
    assert allowed_ops == ALLOWED_OPS_DYNAMO
    assert "temperature" not in _initializer_names(model_path)  # 이름을 바꿀 상수가 없다
    verify_contract(model_path, spec_path, allowed_ops)
    for op_type, value in (("Div", 5.0), ("Mul", 0.2)):
        work_dir = tmp_path / op_type
        work_dir.mkdir()
        _assert_mutation_changes_behaviour_and_is_rejected(
            model_path, spec_path, allowed_ops, _scale_logits(op_type, value), SCALED_LOGITS.format(op_type), work_dir
        )


def test_renamed_temperature_constant_is_rejected_for_any_other_temperature(exported: Exported, tmp_path: Path) -> None:
    """spec의 온도가 1.0이 아니면(여기서는 1.7) 이름을 바꾼 상수는 전부터 걸렸다. 기본 경로의 두 계열."""
    assert exported.spec.standardization.temperature == TEMPERATURE != 1.0
    _assert_mutation_changes_behaviour_and_is_rejected(
        exported.onnx_path, exported.spec_path, ALLOWED_OPS, _rename_temperature_and_change_it,
        NO_TEMPERATURE_CONSTANT, tmp_path,
    )  # fmt: skip


@pytest.mark.parametrize("path_name", ["dynamo_tcn", "dynamo_gru", "int8_tcn"])
def test_renamed_temperature_constant_is_rejected_on_the_dynamo_and_int8_files(
    path_name: str, other_path_models: dict[str, tuple[Path, Path, frozenset[str]]], tmp_path: Path
) -> None:
    """온도 1.7의 dynamo 파일과 INT8 파일. dynamo 경로도 온도가 1.0이 아니면 `temperature` 상수가 있어야 한다."""
    model_path, spec_path, allowed_ops = other_path_models[path_name]
    assert load_feature_spec(spec_path).standardization.temperature == TEMPERATURE != 1.0
    assert "temperature" in _initializer_names(model_path)
    message = (
        "no temperature constant but standardization.temperature is not 1.0"
        if allowed_ops == ALLOWED_OPS_DYNAMO
        else NO_TEMPERATURE_CONSTANT
    )
    _assert_mutation_changes_behaviour_and_is_rejected(
        model_path, spec_path, allowed_ops, _rename_temperature_and_change_it, message, tmp_path
    )
    # 온도 Div 앞에 다른 상수로 나누는 노드를 끼운 것도 어느 경로의 파일에서나 걸린다.
    work_dir = tmp_path / "div_before"
    work_dir.mkdir()
    _assert_mutation_changes_behaviour_and_is_rejected(
        model_path, spec_path, allowed_ops, _scale_logits_before_the_temperature("Div", 5.0),
        SCALED_LOGITS.format("Div"), work_dir,
    )  # fmt: skip


def test_shipped_v0_with_a_renamed_temperature_constant_is_rejected(repo_root: Path, tmp_path: Path) -> None:
    """저장소의 v0(온도 1.0)을 임시 폴더에 복사해 같은 변형을 한다. 저장소의 export/는 읽기만 한다."""
    shipped = repo_root / "export" / "drowsiness_v0.onnx"
    before = shipped.read_bytes()
    model_path = tmp_path / "drowsiness_v0.onnx"
    spec_path = tmp_path / "feature_spec_v0.json"
    shutil.copyfile(shipped, model_path)
    shutil.copyfile(repo_root / "export" / "feature_spec_v0.json", spec_path)
    assert load_feature_spec(spec_path).standardization.temperature == 1.0
    verify_contract(model_path, spec_path)  # 고치지 않은 v0은 통과한다

    work_dir = tmp_path / "mutated"
    work_dir.mkdir()
    _assert_mutation_changes_behaviour_and_is_rejected(
        model_path, spec_path, ALLOWED_OPS, _rename_temperature_and_change_it, NO_TEMPERATURE_CONSTANT, work_dir
    )
    # 파일 이름 검사는 메타데이터의 버전만 보므로 이 변형을 알아보지 못한다. 계약 검사가 잡아야 하는 이유다.
    shutil.copyfile(work_dir / "broken.onnx", work_dir / "drowsiness_v0.onnx")
    verify_file_name(work_dir / "drowsiness_v0.onnx")
    assert shipped.read_bytes() == before


# ---------------------------------------------------------------------------
# verify_contract: 로짓을 내는 노드 (배율을 값이 바뀌지 않는 노드 뒤나 마지막 선형 층에 숨긴 그래프)
# ---------------------------------------------------------------------------


def _logits_reader(model: onnx.ModelProto) -> onnx.NodeProto:
    """로짓을 읽는 노드: 온도로 나누는 Div가 있으면 그 Div, 없으면(dynamo 경로, 온도 1.0) Softmax."""
    softmax = next(node for node in model.graph.node if node.op_type == "Softmax")
    for node in model.graph.node:
        if softmax.input[0] in node.output and node.op_type == "Div" and "temperature" in node.input:
            return node
    return softmax


def _insert_before(model: onnx.ModelProto, anchor: onnx.NodeProto, nodes: list[onnx.NodeProto]) -> None:
    position = [node.output[0] for node in model.graph.node].index(anchor.output[0])
    for offset, node in enumerate(nodes):
        model.graph.node.insert(position + offset, node)


def _scale_logits_then(op_after: str) -> Callable[[onnx.ModelProto], None]:
    # 로짓에 0.2를 곱한 뒤 값을 바꾸지 않는 노드 하나(0을 더하거나 빼기, 입력이 하나인 Concat, 순서를 바꾸지 않는 Transpose)를
    # 끼운다. 로짓을 읽는 노드 바로 앞의 노드는 Mul이 아니게 되어 "로짓은 Div나 Mul의 출력이 아니어야 한다"만으로는 통과했다.
    def mutate(model: onnx.ModelProto) -> None:
        reader = _logits_reader(model)
        graph = model.graph
        graph.initializer.append(numpy_helper.from_array(np.asarray(0.2, dtype=np.float32), "tau"))
        nodes = [onnx.helper.make_node("Mul", [reader.input[0], "tau"], ["scaled_logits"])]
        if op_after in ("Add", "Sub"):
            graph.initializer.append(numpy_helper.from_array(np.zeros(3, dtype=np.float32), "zero3"))
            nodes.append(onnx.helper.make_node(op_after, ["scaled_logits", "zero3"], ["same_logits"]))
        elif op_after == "Concat":
            nodes.append(onnx.helper.make_node("Concat", ["scaled_logits"], ["same_logits"], axis=1))
        else:
            assert op_after == "Transpose"
            nodes.append(onnx.helper.make_node("Transpose", ["scaled_logits"], ["same_logits"], perm=[0, 1]))
        reader.input[0] = "same_logits"
        _insert_before(model, reader, nodes)

    return mutate


def _second_gemm_scales_the_logits(model: onnx.ModelProto) -> None:
    # 로짓 뒤에 Gemm(로짓, 0.2 · I)를 하나 더 둔다. 새 Gemm의 alpha와 beta는 기본값 1이고 편향은 없다.
    reader = _logits_reader(model)
    model.graph.initializer.append(numpy_helper.from_array((0.2 * np.eye(3)).astype(np.float32), "scaled_identity"))
    node = onnx.helper.make_node("Gemm", [reader.input[0], "scaled_identity"], ["scaled_logits"])
    reader.input[0] = "scaled_logits"
    _insert_before(model, reader, [node])


def _last_gemm(model: onnx.ModelProto) -> onnx.NodeProto:
    reader = _logits_reader(model)
    return next(node for node in model.graph.node if reader.input[0] in node.output and node.op_type == "Gemm")


def _scale_in_the_gemm_attributes(model: onnx.ModelProto) -> None:
    # 노드도 상수도 건드리지 않고 마지막 Gemm의 속성만 바꾼다: alpha = beta = 0.2 (로짓 전체에 0.2를 곱한 것과 같다).
    gemm = _last_gemm(model)
    kept = [attribute for attribute in gemm.attribute if attribute.name not in ("alpha", "beta")]
    del gemm.attribute[:]
    gemm.attribute.extend(kept)
    gemm.attribute.extend([onnx.helper.make_attribute("alpha", 0.2), onnx.helper.make_attribute("beta", 0.2)])


def _permute_classes_at_the_logits(model: onnx.ModelProto) -> None:
    # 로짓의 클래스 순서를 뒤집는 Gather를 로짓과 그것을 읽는 노드 사이에 끼운다(Softmax 뒤가 아니라 앞이다).
    reader = _logits_reader(model)
    model.graph.initializer.append(numpy_helper.from_array(np.array([2, 1, 0], dtype=np.int64), "logit_permutation"))
    node = onnx.helper.make_node("Gather", [reader.input[0], "logit_permutation"], ["permuted_logits"], axis=1)
    reader.input[0] = "permuted_logits"
    _insert_before(model, reader, [node])


def _scale_last_layer_constants(model: onnx.ModelProto) -> int:
    """마지막 선형 층이 읽는 float 상수에 모두 0.2를 곱한다(그래프의 모양과 속성은 그대로다). 바꾼 상수의 수를 돌려준다.

    기본 경로와 dynamo 경로는 Gemm의 가중치와 편향, INT8 파일은 편향과 가중치의 양자화 배율(`..._scale`)이다.
    """
    reader = _logits_reader(model)
    last = next(node for node in model.graph.node if reader.input[0] in node.output)
    names = list(last.input)
    if last.op_type == "Add":  # INT8: 로짓 = Mul(Cast(MatMulInteger(...)), Mul(입력 배율, 가중치 배율)) + 편향
        for node in model.graph.node:
            if any(name in node.output for name in last.input) and node.op_type == "Mul":
                for scale in model.graph.node:
                    if any(name in scale.output for name in node.input) and scale.op_type == "Mul":
                        names += list(scale.input)
    changed = 0
    for tensor in model.graph.initializer:
        if tensor.name in names and tensor.data_type == onnx.TensorProto.FLOAT:
            scaled = (numpy_helper.to_array(tensor) * np.float32(0.2)).astype(np.float32)
            tensor.CopyFrom(numpy_helper.from_array(scaled, tensor.name))
            changed += 1
    return changed


def _gemm_beta_only(model: onnx.ModelProto) -> None:
    # 마지막 Gemm의 beta만 0.2로 바꾼다(편향만 줄어든다). alpha는 그대로 1이다.
    gemm = _last_gemm(model)
    kept = [attribute for attribute in gemm.attribute if attribute.name != "beta"]
    del gemm.attribute[:]
    gemm.attribute.extend([*kept, onnx.helper.make_attribute("beta", 0.2)])


def _gemm_weights_from_mul_nodes(model: onnx.ModelProto) -> None:
    # 마지막 Gemm이 가중치와 편향을 상수에서 바로 읽지 않고 Mul(상수, 0.2)의 출력에서 읽는다(상수 자체는 그대로다).
    gemm = _last_gemm(model)
    model.graph.initializer.append(numpy_helper.from_array(np.asarray(0.2, dtype=np.float32), "tau"))
    nodes = [onnx.helper.make_node("Mul", [gemm.input[index], "tau"], [f"scaled_input_{index}"]) for index in (1, 2)]
    gemm.input[1], gemm.input[2] = "scaled_input_1", "scaled_input_2"
    _insert_before(model, gemm, nodes)


def _int8_nodes(model: onnx.ModelProto) -> tuple[onnx.NodeProto, onnx.NodeProto, onnx.NodeProto, onnx.NodeProto]:
    """INT8 파일의 마지막 층: (MatMulInteger, 그 출력을 읽는 Cast, Cast 출력을 곱하는 Mul, 배율 가지의 Mul)."""
    nodes = list(model.graph.node)
    matmul = next(node for node in nodes if node.op_type == "MatMulInteger")
    cast = next(node for node in nodes if node.op_type == "Cast" and list(node.input) == [matmul.output[0]])
    product = next(node for node in nodes if node.op_type == "Mul" and cast.output[0] in node.input)
    other = next(name for name in product.input if name != cast.output[0])
    scale = next(node for node in nodes if other in node.output)
    assert scale.op_type == "Mul"
    return matmul, cast, product, scale


def _int8_cast_reads_a_scaled_integer(model: onnx.ModelProto) -> None:
    # Cast가 MatMulInteger의 출력 대신 Mul(그 출력, int32 3)을 읽는다(로짓의 곱 부분이 세 배가 된다).
    matmul, cast, _, _ = _int8_nodes(model)
    model.graph.initializer.append(numpy_helper.from_array(np.asarray(3, dtype=np.int32), "three"))
    cast.input[0] = "scaled_integer"
    _insert_before(model, cast, [onnx.helper.make_node("Mul", [matmul.output[0], "three"], ["scaled_integer"])])


def _int8_second_mul_on_the_scale_branch(model: onnx.ModelProto) -> None:
    # 배율 가지(양자화기의 배율 × 가중치 배율)의 출력에 0.2를 한 번 더 곱해 곱 Mul에 넘긴다.
    _, _, product, scale = _int8_nodes(model)
    model.graph.initializer.append(numpy_helper.from_array(np.asarray(0.2, dtype=np.float32), "tau"))
    position = list(product.input).index(scale.output[0])
    product.input[position] = "scale_again"
    _insert_before(model, product, [onnx.helper.make_node("Mul", [scale.output[0], "tau"], ["scale_again"])])


def _int8_weight_scale_from_a_node(model: onnx.ModelProto) -> None:
    # 배율 Mul이 가중치 배율 상수를 바로 읽지 않고 Mul(그 상수, 0.2)의 출력을 읽는다.
    _, _, _, scale = _int8_nodes(model)
    constants = {tensor.name for tensor in model.graph.initializer}
    position, name = next((index, item) for index, item in enumerate(scale.input) if item in constants)
    model.graph.initializer.append(numpy_helper.from_array(np.asarray(0.2, dtype=np.float32), "tau"))
    scale.input[position] = "weight_scale_scaled"
    _insert_before(model, scale, [onnx.helper.make_node("Mul", [name, "tau"], ["weight_scale_scaled"])])


def _int8_constant_zero_point(model: onnx.ModelProto) -> None:
    # MatMulInteger가 입력의 영점을 DynamicQuantizeLinear의 출력이 아니라 상수(uint8 128)에서 읽는다. TCN의 마지막 층의
    # 입력에는 음수가 없어서 양자화기의 영점이 0으로 나오므로(무작위 가중치 셋에서 재 봤다) 상수 0으로는 값이 바뀌지 않는다.
    matmul, _, _, _ = _int8_nodes(model)
    model.graph.initializer.append(numpy_helper.from_array(np.asarray(128, dtype=np.uint8), "zero_point_128"))
    matmul.input[2] = "zero_point_128"


NOT_THE_ONLY_GEMM = r"the logits must be the output of the network's last linear layer, which is the only Gemm node"
NOT_THE_QUANTIZED_LAYER = "the logits of an INT8 file must come from the quantized last linear layer"
# (이름, 변형, 기본 경로와 dynamo 경로의 파일에서 나와야 하는 메시지, INT8 파일에서 나와야 하는 메시지. None이면 그 파일에는
# 해당하지 않는 변형이다). 앞의 여섯은 3차 검증에서 계약 검사를 통과했던 변형이고 마지막은 그 검사가 함께 잡게 된 것이다.
LOGITS_PRODUCER_MUTATIONS: list[tuple[str, Callable[[onnx.ModelProto], None], str, str | None]] = [
    (
        "mul_then_add_zero", _scale_logits_then("Add"),
        NOT_THE_ONLY_GEMM + r" of the graph; got producer \['Add'\] and 1 Gemm node",
        NOT_THE_QUANTIZED_LAYER + ".*the nodes behind the bias Add do not have that shape",
    ),
    (
        "mul_then_sub_zero", _scale_logits_then("Sub"),
        NOT_THE_ONLY_GEMM + r" of the graph; got producer \['Sub'\] and 1 Gemm node",
        NOT_THE_QUANTIZED_LAYER + r".*got producer \['Sub'\] and 1 MatMulInteger node",
    ),
    (
        "mul_then_single_concat", _scale_logits_then("Concat"),
        NOT_THE_ONLY_GEMM + r" of the graph; got producer \['Concat'\] and 1 Gemm node",
        NOT_THE_QUANTIZED_LAYER + r".*got producer \['Concat'\] and 1 MatMulInteger node",
    ),
    (
        "mul_then_identity_transpose", _scale_logits_then("Transpose"),
        NOT_THE_ONLY_GEMM + r" of the graph; got producer \['Transpose'\] and 1 Gemm node",
        NOT_THE_QUANTIZED_LAYER + r".*got producer \['Transpose'\] and 1 MatMulInteger node",
    ),
    (
        "second_gemm", _second_gemm_scales_the_logits,
        NOT_THE_ONLY_GEMM + r" of the graph; got producer \['Gemm'\] and 2 Gemm node",
        r"operators not in the allowed list: \['Gemm'\]",  # INT8 파일에는 Gemm이 없다. 허용 연산자 목록에서 먼저 걸린다
    ),
    (
        "gemm_alpha_beta", _scale_in_the_gemm_attributes,
        "the Gemm node that makes the logits must have alpha = 1, beta = 1 and transA = 0, got alpha 0.2, beta 0.2",
        None,
    ),
    (
        "classes_permuted_at_the_logits", _permute_classes_at_the_logits,
        NOT_THE_ONLY_GEMM + r" of the graph; got producer \['Gather'\] and 1 Gemm node",
        NOT_THE_QUANTIZED_LAYER + r".*got producer \['Gather'\] and 1 MatMulInteger node",
    ),
]  # fmt: skip
# 실패 메시지는 이 검사가 무엇을 보고 무엇을 못 보는지를 함께 적는다.
LOGITS_SCOPE_NOTE = (
    "a last layer whose weights and bias were rescaled is not detected here: the parity check against the checkpoint "
    r"\(parity_max_abs_diff\) is the guard for the weights"
)


INT8_CHAIN = NOT_THE_QUANTIZED_LAYER + ".*the nodes behind the bias Add do not have that shape"
# 로짓을 내는 노드의 검사 가운데 조건 하나씩을 따로 고정하는 변형(4차 검증에서 어느 시험도 잡지 않던 조건들). (이름, 변형,
# 메시지, 파일). 파일이 "default"면 기본 경로의 두 계열, "int8_tcn"이면 INT8 파일(TCN)이다. 변형마다 그 조건 하나만 빼면
# 받아들여지고 출력이 달라지는 파일이다.
LOGITS_PRODUCER_CONDITION_MUTATIONS: list[tuple[str, Callable[[onnx.ModelProto], None], str, str]] = [
    (
        "gemm_beta_only", _gemm_beta_only,  # beta = 1
        "the Gemm node that makes the logits must have alpha = 1, beta = 1 and transA = 0, got alpha 1, beta 0.2",
        "default",
    ),
    (
        "gemm_weights_from_mul_nodes", _gemm_weights_from_mul_nodes,  # 가중치와 편향이 그래프의 float32 상수
        "the Gemm node that makes the logits must read its weights and its bias from float32 constants of the graph",
        "default",
    ),
    ("int8_cast_reads_a_scaled_integer", _int8_cast_reads_a_scaled_integer, INT8_CHAIN, "int8_tcn"),  # Cast의 입력
    ("int8_second_mul_on_the_scale_branch", _int8_second_mul_on_the_scale_branch, INT8_CHAIN, "int8_tcn"),  # 배율
    ("int8_weight_scale_from_a_node", _int8_weight_scale_from_a_node, INT8_CHAIN, "int8_tcn"),  # 배율 상수 하나
    ("int8_constant_zero_point", _int8_constant_zero_point, INT8_CHAIN, "int8_tcn"),  # 양자화기의 출력을 읽는다
]  # fmt: skip
_PRODUCER_CASES = [
    *[
        (kind, name, mutate, message, "default")
        for name, mutate, message, _ in LOGITS_PRODUCER_MUTATIONS
        for kind in KINDS
    ],
    *[
        (kind, name, mutate, message, path)
        for name, mutate, message, path in LOGITS_PRODUCER_CONDITION_MUTATIONS
        for kind in (KINDS if path == "default" else (KIND_TCN,))
    ],
]


@pytest.mark.parametrize(
    ("exported", "mutate", "message", "path"),
    [(kind, mutate, message, path) for kind, _, mutate, message, path in _PRODUCER_CASES],
    ids=[f"{kind if path == 'default' else path}-{name}" for kind, name, _, _, path in _PRODUCER_CASES],
    indirect=["exported"],
)
def test_scaling_hidden_behind_a_value_preserving_node_or_in_the_last_layer_is_rejected(
    mutate: Callable[[onnx.ModelProto], None],
    message: str,
    path: str,
    exported: Exported,
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    """로짓을 내는 노드는 신경망의 마지막 선형 층 그대로여야 한다(기본 경로의 두 계열, 온도 1.7. INT8 파일은 TCN).

    전에는 "로짓은 Div나 Mul의 출력이 아니어야 한다"만 봐서, 0.2를 곱한 뒤 값을 바꾸지 않는 노드를 하나 끼우거나 배율을
    마지막 Gemm의 속성(alpha, beta)이나 둘째 Gemm에 넣은 파일이 통과했다(확률이 0.3까지 달라지는 파일이다).
    `LOGITS_PRODUCER_CONDITION_MUTATIONS`는 그 검사의 조건 하나씩을 고정한다: beta만 바꾼 Gemm, Mul 노드에서 가중치를 읽는
    Gemm, INT8 파일에서 배율을 곱한 정수를 읽는 Cast, 배율 가지의 둘째 Mul, 노드에서 읽는 가중치 배율, 상수 영점.
    """
    if path == "default":
        model_path, spec_path, allowed_ops = exported.onnx_path, exported.spec_path, ALLOWED_OPS
    else:  # INT8 파일은 TCN으로만 만든다(`export.int8_if_tcn`). 기본 경로 TCN과 같은 가중치(시드 0)에서 양자화한 파일이다
        model_path, spec_path, allowed_ops = request.getfixturevalue("other_path_models")[path]
    _assert_mutation_changes_behaviour_and_is_rejected(model_path, spec_path, allowed_ops, mutate, message, tmp_path)
    with pytest.raises(ContractError, match=LOGITS_SCOPE_NOTE):
        verify_contract(tmp_path / "broken.onnx", tmp_path / spec_path.name, allowed_ops)


def _assert_logits_producer_is_pinned(
    model_path: Path, spec_path: Path, allowed_ops: frozenset[str], work_dir: Path
) -> int:
    """`LOGITS_PRODUCER_MUTATIONS`를 파일 하나에 모두 해 본다. 해당하는 변형의 수를 돌려준다."""
    verify_contract(model_path, spec_path, allowed_ops)  # 변형하지 않은 파일은 통과한다
    tried = 0
    for name, mutate, float_message, int8_message in LOGITS_PRODUCER_MUTATIONS:
        message = int8_message if allowed_ops == ALLOWED_OPS_INT8 else float_message
        if message is None:
            continue
        folder = work_dir / name
        folder.mkdir()
        _assert_mutation_changes_behaviour_and_is_rejected(model_path, spec_path, allowed_ops, mutate, message, folder)
        tried += 1
    return tried


@pytest.mark.parametrize("name", ["default_tcn", "default_gru", "dynamo_tcn", "dynamo_gru", "int8_tcn"])
def test_logits_producer_is_pinned_when_the_spec_temperature_is_one(
    name: str, temperature_one_models: dict[str, tuple[Path, Path, frozenset[str]]], tmp_path: Path
) -> None:
    """온도 1.0의 파일들. dynamo 경로의 파일에는 온도 Div가 없어 Softmax가 로짓을 바로 읽는다. 거기서도 같은 검사가 걸린다."""
    model_path, spec_path, allowed_ops = temperature_one_models[name]
    has_division = "temperature" in _initializer_names(model_path)
    assert has_division == (not name.startswith("dynamo"))
    tried = _assert_logits_producer_is_pinned(model_path, spec_path, allowed_ops, tmp_path)
    assert tried == (6 if name == "int8_tcn" else 7)  # Gemm의 속성을 바꾸는 변형은 Gemm이 없는 INT8 파일에 해당하지 않는다


@pytest.mark.parametrize("path_name", ["dynamo_tcn", "dynamo_gru", "int8_tcn"])
def test_logits_producer_is_pinned_on_the_dynamo_and_int8_files(
    path_name: str, other_path_models: dict[str, tuple[Path, Path, frozenset[str]]], tmp_path: Path
) -> None:
    """온도 1.7의 dynamo 파일과 INT8 파일."""
    model_path, spec_path, allowed_ops = other_path_models[path_name]
    tried = _assert_logits_producer_is_pinned(model_path, spec_path, allowed_ops, tmp_path)
    assert tried == (6 if path_name == "int8_tcn" else 7)


def test_shipped_v0_with_scaled_logits_behind_a_value_preserving_node_is_rejected(
    repo_root: Path, tmp_path: Path
) -> None:
    """저장소의 v0을 임시 폴더에 복사해 같은 변형을 한다. 고치지 않은 v0은 새 검사를 통과하고, 저장소의 파일은 읽기만 한다."""
    shipped = repo_root / "export" / "drowsiness_v0.onnx"
    before = shipped.read_bytes()
    model_path = tmp_path / "drowsiness_v0.onnx"
    spec_path = tmp_path / "feature_spec_v0.json"
    shutil.copyfile(shipped, model_path)
    shutil.copyfile(repo_root / "export" / "feature_spec_v0.json", spec_path)
    work_dir = tmp_path / "mutated"
    work_dir.mkdir()
    assert _assert_logits_producer_is_pinned(model_path, spec_path, ALLOWED_OPS, work_dir) == 7
    verify_contract(shipped, repo_root / "export" / "feature_spec_v0.json")
    assert shipped.read_bytes() == before


def _raw_attribute(name: str, attribute_type: int, **fields: object) -> onnx.AttributeProto:
    """형과 값을 따로 정한 속성(`onnx.helper.make_attribute`는 값에 맞는 형만 만든다)."""
    attribute = onnx.AttributeProto(name=name, type=attribute_type)
    for field, value in fields.items():
        if field == "t":
            attribute.t.CopyFrom(value)
        else:
            setattr(attribute, field, value)
    return attribute


_AT = onnx.AttributeProto
# (이름, 마지막 Gemm에서 바꿀 속성, 새로 넣을 속성들, 메시지의 "got" 부분)
MALFORMED_GEMM_ATTRIBUTES: list[tuple[str, str, list[onnx.AttributeProto], str]] = [
    ("alpha_undefined", "alpha", [_raw_attribute("alpha", _AT.UNDEFINED, f=0.2)], "UNDEFINED"),
    ("alpha_integer", "alpha", [_raw_attribute("alpha", _AT.INT, i=1)], "INT"),
    ("beta_string", "beta", [_raw_attribute("beta", _AT.STRING, s=b"1")], "STRING"),
    ("alpha_tensor", "alpha", [_raw_attribute("alpha", _AT.TENSOR, t=numpy_helper.from_array(np.float32(1)))],
     "TENSOR"),
    ("beta_without_a_value", "beta", [_raw_attribute("beta", _AT.FLOAT)], "FLOAT without a value"),
    ("trans_a_without_a_value", "transA", [_raw_attribute("transA", _AT.INT)], "INT without a value"),
    (
        "alpha_twice", "alpha",
        [_raw_attribute("alpha", _AT.FLOAT, f=1.0), _raw_attribute("alpha", _AT.FLOAT, f=0.2)], "FLOAT, FLOAT",
    ),
]  # fmt: skip


@pytest.mark.parametrize(
    ("name", "attributes", "got"),
    [(name, attributes, got) for _, name, attributes, got in MALFORMED_GEMM_ATTRIBUTES],
    ids=[case for case, _, _, _ in MALFORMED_GEMM_ATTRIBUTES],
)
def test_malformed_attribute_of_the_last_gemm_is_a_contract_error_with_a_one_line_reason(
    name: str, attributes: list[onnx.AttributeProto], got: str, repo_root: Path, tmp_path: Path
) -> None:
    """v0 사본의 마지막 Gemm의 속성 하나를 형이 틀리거나 값이 없거나 두 번 나오게 바꾼다. 이유가 한 줄인 ContractError다.

    전에는 UNDEFINED 형(값이 None), 문자열, 텐서 형의 alpha·beta에서 메시지를 만들다가 TypeError가 났다. 형이 맞지 않는
    속성은 onnxruntime도 읽지 못하므로 받아들여지는 일은 없었지만, `verify_contract`가 약속한 예외가 아니었다.
    """
    shipped = repo_root / "export" / "drowsiness_v0.onnx"
    before = shipped.read_bytes()
    model = onnx.load(str(shipped))
    gemm = next(node for node in model.graph.node if node.op_type == "Gemm")
    kept = [attribute for attribute in gemm.attribute if attribute.name != name]
    del gemm.attribute[:]
    gemm.attribute.extend([*kept, *attributes])
    model_path = tmp_path / "drowsiness_v0.onnx"
    model_path.write_bytes(model.SerializeToString())
    spec_path = tmp_path / "feature_spec_v0.json"
    shutil.copyfile(repo_root / "export" / "feature_spec_v0.json", spec_path)

    kind = "INT" if name == "transA" else "FLOAT"
    expected = (
        f"attribute '{name}' of the Gemm node that makes the logits must appear at most once, with type {kind} and a "
        f"value; got {got}; "
    )
    with pytest.raises(ContractError) as caught:
        verify_contract(model_path, spec_path)
    reason = str(caught.value)
    assert reason.startswith(expected) and "\n" not in reason
    assert re.search(LOGITS_SCOPE_NOTE, reason)
    assert shipped.read_bytes() == before


def test_rescaled_last_layer_constants_pass_the_contract_and_only_parity_catches_them(
    exported: Exported, other_path_models: dict[str, tuple[Path, Path, frozenset[str]]], tmp_path: Path
) -> None:
    """알려진 한계: 마지막 선형 층의 가중치와 편향에 통째로 0.2를 곱한 파일은 계약 검사를 통과한다.

    그래프의 모양, 속성, 전처리·후처리 상수가 모두 그대로이고 달라진 것은 신경망의 가중치뿐이다. 계약 검사는 가중치가
    맞는지 보지 않는다(`verify_contract`의 설명). 그런 파일은 체크포인트의 PyTorch 모델과 출력을 견주는 일치 검증이 잡는다.
    이 시험은 그 경계를 적어 두는 것이다: 계약 검사가 이 변형을 잡게 되면(또는 일치 검증이 못 잡게 되면) 여기서 드러난다.
    """
    cases = [
        (exported.onnx_path, exported.spec_path, ALLOWED_OPS, 2),  # Gemm의 가중치와 편향
        (*other_path_models["int8_tcn"], 2),  # 편향과 가중치의 양자화 배율
        (*other_path_models[f"dynamo_{exported.kind}"], 2),
    ]
    inputs = random_raw_inputs(exported.spec, 5, seed=3)
    for index, (model_path, spec_path, allowed_ops, n_constants) in enumerate(cases):
        model = onnx.load(str(model_path))
        assert _scale_last_layer_constants(model) == n_constants
        folder = tmp_path / str(index)
        folder.mkdir()
        rescaled = folder / "rescaled.onnx"
        rescaled.write_bytes(model.SerializeToString())
        spec_copy = folder / spec_path.name
        shutil.copyfile(spec_path, spec_copy)
        # 출력은 원본과 크게 다르다. 그래도 계약 검사는 통과한다.
        original = np.concatenate([_run(model_path, inputs[i : i + 1]) for i in range(len(inputs))])
        changed = np.concatenate([_run(rescaled, inputs[i : i + 1]) for i in range(len(inputs))])
        assert float(np.max(np.abs(changed - original))) > 1e-3
        verify_contract(rescaled, spec_copy, allowed_ops)

    # 일치 검증(내보낸 직후에 명령이 하는 검사)은 float 파일의 그 변형을 허용 오차 1e-5의 수백 배 차이로 잡는다.
    assert parity_max_abs_diff(exported.wrapper, exported.onnx_path, inputs) < 1e-5
    assert parity_max_abs_diff(exported.wrapper, tmp_path / "0" / "rescaled.onnx", inputs) > 1e-3
    assert parity_max_abs_diff(exported.wrapper, tmp_path / "2" / "rescaled.onnx", inputs) > 1e-3


def _reference_preprocess(
    raw: np.ndarray, means: list[float], stds: list[float], log_columns: list[int], z_clip: float
) -> np.ndarray:
    """PLAN 5.1절의 전처리를 float64 numpy로 따로 쓴 것: log1p(max(x, 0))(일부 열) → (x − mean) / std → ±z_clip."""
    x = raw.astype(np.float64)
    x[..., log_columns] = np.log1p(np.maximum(x[..., log_columns], 0.0))
    return np.clip((x - np.asarray(means)) / np.asarray(stds), -z_clip, z_clip)


def _reference_probs(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits.astype(np.float64) / temperature
    exp = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def _raw_windows(means: list[float], stds: list[float], seed: int) -> np.ndarray:
    """기준 구현과 비교할 원시 특징 창 [6, 120, 17]. `random_raw_inputs`를 쓰지 않고 여기서 따로 만든다."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 2.5, size=(6, 120, 17))
    raw = z * np.asarray(stds) + np.asarray(means)
    raw[..., LOG1P_COLUMNS] = np.expm1(raw[..., LOG1P_COLUMNS])  # 변환 뒤 척도 -> 원시 척도. 음수도 나온다
    raw[0, :10, :] = -1.0e6
    raw[1, :10, :] = 1.0e6
    raw[2, :10, LOG1P_COLUMNS[0]] = -3.0
    return raw.astype(np.float32)


def _compare_with_reference(
    model_path: Path, raw: np.ndarray, means: list[float], stds: list[float], temperature: float
) -> tuple[float, float]:
    """그래프 안 전처리 출력(Clip의 출력)과 확률을 기준 구현과 비교한 최대 절대 오차 (z, probs)."""
    model = onnx.load(str(model_path))
    clip = next(node for node in model.graph.node if node.op_type == "Clip")
    softmax = next(node for node in model.graph.node if node.op_type == "Softmax")
    scale = next(node for node in model.graph.node if softmax.input[0] in node.output)
    has_division = scale.op_type == "Div" and "temperature" in scale.input
    assert has_division or temperature == 1.0
    logits_name = scale.input[0] if has_division else softmax.input[0]
    for name in (clip.output[0], logits_name):
        model.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])

    expected_z = _reference_preprocess(raw, means, stds, LOG1P_COLUMNS, z_clip=5.0)
    # 입력이 검사할 경우를 모두 담고 있다: 자르기 범위 안팎의 값, log1p 열의 음수.
    assert bool((np.abs(expected_z) == 5.0).any()) and bool((np.abs(expected_z) < 4.0).any())
    assert bool((raw[..., LOG1P_COLUMNS] < 0.0).any())
    worst_z = 0.0
    worst_probs = 0.0
    for index in range(raw.shape[0]):
        probs, z, logits = session.run(["probs", clip.output[0], logits_name], {"features": raw[index : index + 1]})
        assert z.shape == (1, 120, 17) and logits.shape == (1, 3) and probs.shape == (1, 3)
        worst_z = max(worst_z, float(np.max(np.abs(z - expected_z[index : index + 1]))))
        worst_probs = max(worst_probs, float(np.max(np.abs(probs - _reference_probs(logits, temperature)))))
    return worst_z, worst_probs


def test_in_graph_processing_matches_an_independent_numpy_reference(exported: Exported) -> None:
    raw = _raw_windows(MEANS, STDS, seed=123)
    worst_z, worst_probs = _compare_with_reference(exported.onnx_path, raw, MEANS, STDS, TEMPERATURE)
    # float32 그래프와 float64 기준의 차이. 잰 값은 z 1.1e-6, 확률 4e-8 이하다(TCN, GRU 모두).
    assert worst_z < 2e-5
    assert worst_probs < 1e-6
    # 기준이 민감한지: 온도를 빼거나 log1p 열 하나를 빼면 허용 오차를 크게 넘는다.
    assert _compare_with_reference(exported.onnx_path, raw, MEANS, STDS, 1.0)[1] > 1e-3
    shifted = [value + 0.5 * std for value, std in zip(MEANS, STDS)]
    assert _compare_with_reference(exported.onnx_path, raw, shifted, STDS, TEMPERATURE)[0] > 0.4


def test_shipped_v0_processing_matches_an_independent_numpy_reference(repo_root: Path) -> None:
    # v0은 설계값 spec 그대로다: 평균 0, 표준편차 1, 온도 1.
    zeros = [0.0] * 17
    ones = [1.0] * 17
    raw = _raw_windows(zeros, ones, seed=321)
    worst_z, worst_probs = _compare_with_reference(repo_root / "export" / "drowsiness_v0.onnx", raw, zeros, ones, 1.0)
    assert worst_z < 2e-5
    assert worst_probs < 1e-6


# ---------------------------------------------------------------------------
# 파일 이름, 읽을 수 없는 파일, .gitattributes
# ---------------------------------------------------------------------------


def test_model_file_names() -> None:
    assert model_file_name(0) == "drowsiness_v0.onnx"
    assert model_file_name("12", int8=True) == "drowsiness_v12_int8.onnx"
    for bad in (-1, "v1", "1.0", ""):
        with pytest.raises(ValueError, match="model_version"):
            model_file_name(bad)


def test_verify_file_name_ties_the_name_to_the_metadata_version(exported: Exported, tmp_path: Path) -> None:
    verify_file_name(exported.onnx_path)  # drowsiness_v3.onnx, model_version 3
    for name in ("drowsiness_v3.onnx", "drowsiness_v3_int8.onnx"):
        shutil.copyfile(exported.onnx_path, tmp_path / name)
        verify_file_name(tmp_path / name)
    for name in ("drowsiness_v5.onnx", "whatever.onnx", "drowsiness_v03.onnx", "drowsiness_v3.onnx.bak",
                 "Drowsiness_v3.onnx", "drowsiness_v3_INT8.onnx"):  # fmt: skip
        renamed = tmp_path / name
        shutil.copyfile(exported.onnx_path, renamed)
        with pytest.raises(ContractError, match="file name must be 'drowsiness_v3.onnx' or 'drowsiness_v3_int8.onnx'"):
            verify_file_name(renamed)
    # verify_contract는 파일 이름을 보지 않는다(임시 이름으로 쓴 파일도 검사한다). 이름은 verify_file_name의 몫이다.
    shutil.copyfile(exported.spec_path, tmp_path / exported_spec_name(3))
    verify_contract(tmp_path / "whatever.onnx", tmp_path / exported_spec_name(3))


def test_unreadable_model_file_raises_value_error(exported: Exported, tmp_path: Path) -> None:
    garbage = tmp_path / "drowsiness_v3.onnx"
    garbage.write_bytes(b"this is not an onnx file\x00\x01\x02" * 10)
    truncated = tmp_path / "truncated.onnx"
    truncated.write_bytes(exported.onnx_path.read_bytes()[:5000])
    for path in (garbage, truncated):
        for call in (read_metadata, graph_operators, exported_spec_path, verify_file_name):
            with pytest.raises(ValueError, match="not a readable ONNX file") as caught:
                call(path)
            # 부르는 쪽이 protobuf의 예외 형을 알 필요가 없다: 이 패키지의 ValueError다.
            assert type(caught.value) is ContractError and str(path) in str(caught.value)
        with pytest.raises(ContractError, match="not a readable ONNX file"):
            verify_contract(path, exported.spec_path)
    with pytest.raises(FileNotFoundError):
        read_metadata(tmp_path / "missing.onnx")


def test_export_modules_do_not_import_protobuf_directly(repo_root: Path) -> None:
    # protobuf는 onnx가 끌어오는 간접 의존성이다. pyproject.toml에 없으므로 직접 import하지 않는다.
    for name in ("export_onnx.py", "models.py"):
        tree = ast.parse((repo_root / "drowsy" / name).read_text(encoding="utf-8"))
        modules = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        modules |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
        assert modules, name
        assert not [module for module in modules if module.split(".")[0] == "google"], name


def test_gitattributes_marks_model_files_binary(repo_root: Path) -> None:
    rules = [
        line.split()
        for line in (repo_root / ".gitattributes").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert ["*.onnx", "binary"] in rules
    assert ["*.task", "binary"] in rules
    # 해시 대상 JSON은 줄바꿈 변환만 끈다(텍스트 diff는 그대로 쓴다).
    assert ["export/*.json", "-text"] in rules and ["config/feature_spec.json", "-text"] in rules


def test_git_treats_the_shipped_model_as_binary(repo_root: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    done = subprocess.run(
        ["git", "check-attr", "text", "diff", "merge", "--",
         "export/drowsiness_v0.onnx", "export/feature_spec_v0.json"],
        cwd=repo_root, capture_output=True, text=True, check=False,
    )  # fmt: skip
    if done.returncode != 0:
        pytest.skip("not a git work tree")
    lines = done.stdout.splitlines()
    for attribute in ("text", "diff", "merge"):
        assert f"export/drowsiness_v0.onnx: {attribute}: unset" in lines
    assert "export/feature_spec_v0.json: text: unset" in lines
    assert "export/feature_spec_v0.json: diff: unspecified" in lines


# ---------------------------------------------------------------------------
# INT8 (계획서의 결정 D16: TCN이 최종 모델일 때만 만든다)
# ---------------------------------------------------------------------------


def test_int8_tcn_loads_and_stays_close(exported: Exported, tmp_path: Path) -> None:
    if exported.kind != KIND_TCN:
        # GRU 연산자는 양자화되지 않으므로 INT8 파일을 만들지 않는다. INT8 허용 목록에도 없다.
        int8_path = quantize_int8(exported.onnx_path, tmp_path / "gru_int8.onnx")
        with pytest.raises(ContractError, match="operators not in the allowed list"):
            verify_contract(int8_path, exported.spec_path, ALLOWED_OPS_INT8)
        return

    int8_path = quantize_int8(exported.onnx_path, tmp_path / "nested" / "drowsiness_v3_int8.onnx")
    verify_contract(int8_path, exported.spec_path, ALLOWED_OPS_INT8)
    assert set(graph_operators(int8_path)) == ALLOWED_OPS_INT8
    assert int8_path.stat().st_size < exported.onnx_path.stat().st_size

    # 양자화기가 더한 키 없이 원본의 메타데이터 열 개가 그대로 있다.
    session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    assert session.get_modelmeta().custom_metadata_map == read_metadata(exported.onnx_path)
    assert onnx.load(str(int8_path)).ir_version == 8

    # 허용 목록은 파일 종류마다 다르다.
    with pytest.raises(ContractError, match="operators not in the allowed list"):
        verify_contract(int8_path, exported.spec_path)
    with pytest.raises(ContractError, match="operators not in the allowed list"):
        verify_contract(exported.onnx_path, exported.spec_path, ALLOWED_OPS_INT8)

    inputs = random_raw_inputs(exported.spec, 100, seed=1)
    diffs = []
    same_class = 0
    for index in range(inputs.shape[0]):
        window = inputs[index : index + 1]
        float_probs = _run(exported.onnx_path, window)
        int8_probs = _run(int8_path, window)
        assert np.all(np.isfinite(int8_probs))
        diffs.append(float(np.max(np.abs(int8_probs - float_probs))))
        same_class += int(np.argmax(int8_probs) == np.argmax(float_probs))
    # 느슨한 허용 오차. 무작위 가중치에서 잰 값은 최대 약 0.004였다.
    assert max(diffs) < 0.05
    assert float(np.mean(diffs)) < 0.01
    assert same_class >= 90


# ---------------------------------------------------------------------------
# dynamo 대체 경로
# ---------------------------------------------------------------------------


def test_dynamo_fallback_exports_the_same_contract(tmp_path: Path, trained_spec_path: Path) -> None:
    pytest.importorskip("onnxscript")
    spec = load_feature_spec(trained_spec_path)
    inputs = random_raw_inputs(spec, 30, seed=2)
    used: set[str] = set()
    for kind in KINDS:
        wrapper = _make_wrapper(kind, spec)
        path = export(wrapper, trained_spec_path, tmp_path / kind / "drowsiness_v3.onnx", META, dynamo=True)
        verify_contract(path, exported_spec_path(path), ALLOWED_OPS_DYNAMO)
        model = onnx.load(str(path))
        assert [(entry.domain, entry.version) for entry in model.opset_import] == [("", 17)]
        assert model.ir_version == 8
        written = sorted(item.name for item in (tmp_path / kind).iterdir())
        assert written == ["drowsiness_v3.onnx", "feature_spec_v3.json"]
        assert read_metadata(path)["commit"] == META["commit"]
        assert parity_max_abs_diff(wrapper, path, inputs) < 1e-4
        # 두 경로의 출력이 같다.
        reference = export(wrapper, trained_spec_path, tmp_path / f"{kind}_torchscript" / "m.onnx", META)
        for index in range(5):
            window = inputs[index : index + 1]
            assert np.allclose(_run(path, window), _run(reference, window), rtol=0.0, atol=1e-5)
        used |= set(graph_operators(path))
    assert used == ALLOWED_OPS_DYNAMO


def test_dynamo_fallback_with_temperature_one(
    tmp_path: Path, feature_spec_path: Path, feature_spec: FeatureSpec
) -> None:
    pytest.importorskip("onnxscript")
    # 온도가 1이면 dynamo 경로는 나눗셈 노드를 지운다. 그래도 계약 검사를 통과해야 한다.
    wrapper = _make_wrapper(KIND_GRU, feature_spec)
    path = export(wrapper, feature_spec_path, tmp_path / "m.onnx", META, dynamo=True)
    names = {tensor.name for tensor in onnx.load(str(path)).graph.initializer}
    assert "temperature" not in names
    assert parity_max_abs_diff(wrapper, path, random_raw_inputs(feature_spec, 10, seed=4)) < 1e-4


# ---------------------------------------------------------------------------
# git_commit
# ---------------------------------------------------------------------------


HEAD_HASH = "9fceb02d0ae598e95dc970b74767f19372d61af8"


@pytest.mark.parametrize(
    ("head", "status", "expected"),
    [
        (HEAD_HASH, "", HEAD_HASH),
        (HEAD_HASH, " M drowsy/models.py", HEAD_HASH + "-dirty"),
        (HEAD_HASH, "?? notes.txt", HEAD_HASH + "-dirty"),
        (None, "?? drowsy/", "uncommitted-dirty"),
        (None, "", "uncommitted"),
        (None, None, "uncommitted"),
    ],
)
def test_git_commit_string(
    head: str | None, status: str | None, expected: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_git(repo_dir: Path, *args: str) -> str | None:
        assert repo_dir == tmp_path
        return {("rev-parse", "HEAD"): head, ("status", "--porcelain"): status}[args]

    monkeypatch.setattr(export_onnx, "_run_git", fake_git)
    assert git_commit(tmp_path) == expected


def test_git_commit_in_a_repository_without_commits(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    # 커밋이 없으면 `git rev-parse HEAD`는 실패하면서 표준 출력에 "HEAD"를 쓴다. 그 문자열을 해시로 쓰면 안 된다.
    assert git_commit(tmp_path) == "uncommitted"
    (tmp_path / "file.txt").write_text("x\n", encoding="utf-8")
    assert git_commit(tmp_path) == "uncommitted-dirty"


@pytest.fixture
def committed_repo(tmp_path: Path, trained_spec_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
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
    shutil.copyfile(trained_spec_path, repo / "feature_spec.json")
    identity = ["-c", "user.name=test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false"]
    for args in (["init", "-q"], ["add", "-A"], [*identity, "commit", "-q", "-m", "initial"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    return repo


def _head(repo: Path) -> str:
    done = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True)
    return done.stdout.strip()


def test_first_export_of_a_new_version_from_a_clean_tree_is_not_dirty(
    committed_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = committed_repo
    head = _head(repo)
    assert re.fullmatch(r"[0-9a-f]{40}", head)
    assert git_commit(repo) == head  # 깨끗한 작업 트리

    # export()가 이 임시 저장소의 상태를 읽게 한다. 새 버전의 모델과 사본은 저장소 안(추적되지 않는 새 파일)에 쓴다.
    monkeypatch.setattr(export_onnx, "git_commit", functools.partial(git_commit, repo))
    spec_path = repo / "feature_spec.json"
    out_path = repo / "export" / "drowsiness_v7.onnx"
    spec = load_feature_spec(spec_path)
    meta = {"model_version": "7", "trained_on": "x"}
    path = export(_make_wrapper(KIND_GRU, spec), spec_path, out_path, meta)
    assert read_metadata(path)["commit"] == head

    # 이 시험이 순서를 실제로 가른다: 내보낸 뒤의 작업 트리는 그 두 파일 때문에 깨끗하지 않다.
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, check=True, capture_output=True, text=True)
    assert status.stdout.split() == ["??", "export/"]
    assert git_commit(repo) == head + "-dirty"
    # 그래서 같은 버전을 바로 다시 내보내면(커밋하지 않은 채) "-dirty"가 붙는다.
    again = export(_make_wrapper(KIND_GRU, spec), spec_path, out_path, meta)
    assert read_metadata(again)["commit"] == head + "-dirty"


def test_export_from_a_tree_with_other_changes_is_dirty(committed_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = committed_repo
    head = _head(repo)
    (repo / "notes.txt").write_text("not committed\n", encoding="utf-8")
    monkeypatch.setattr(export_onnx, "git_commit", functools.partial(git_commit, repo))
    spec_path = repo / "feature_spec.json"
    spec = load_feature_spec(spec_path)
    meta = {"model_version": "7", "trained_on": "x"}
    path = export(_make_wrapper(KIND_GRU, spec), spec_path, repo / "export" / "drowsiness_v7.onnx", meta)
    assert read_metadata(path)["commit"] == head + "-dirty"


# ---------------------------------------------------------------------------
# 지연, 명령줄
# ---------------------------------------------------------------------------


def test_measure_latency(exported: Exported) -> None:
    latency = measure_latency(exported.onnx_path, runs=50)
    assert set(latency) == {"median_ms", "p95_ms"}
    assert 0.0 < latency["median_ms"] <= latency["p95_ms"] < 1000.0
    with pytest.raises(ValueError, match="runs"):
        measure_latency(exported.onnx_path, runs=0)


@pytest.mark.parametrize("kind", KINDS)
def test_cli_random_init_writes_v0(
    kind: str, tmp_path: Path, config_path: Path, feature_spec_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def run(out_dir: Path) -> int:
        return export_onnx.main(
            ["--config", str(config_path), "--random-init", "--kind", kind, "--version", "0",
             "--spec", str(feature_spec_path), "--out-dir", str(out_dir)]
        )  # fmt: skip

    assert run(tmp_path / "a") == 0
    printed = capsys.readouterr().out
    assert "verify_contract: OK" in printed
    assert re.search(r"parity \(\d+ random raw inputs\): max abs diff \S+ \(tolerance \S+\) OK", printed)
    assert re.search(r"latency \(\d+ runs, \d+ intra-op thread\(s\)\): median \S+ ms, p95 \S+ ms", printed)

    onnx_path = tmp_path / "a" / "drowsiness_v0.onnx"
    spec_copy = tmp_path / "a" / "feature_spec_v0.json"
    assert sorted(path.name for path in (tmp_path / "a").iterdir()) == ["drowsiness_v0.onnx", "feature_spec_v0.json"]
    assert f"wrote {spec_copy}" in printed
    assert spec_copy.read_bytes() == feature_spec_path.read_bytes()
    verify_contract(onnx_path, spec_copy)
    metadata = read_metadata(onnx_path)
    assert metadata["model_version"] == "0" and metadata["trained_on"] == "none (random weights)"
    assert metadata["feature_spec_hash"] == file_sha256(feature_spec_path)

    # 시드가 고정이라 다시 실행해도 같은 파일이다.
    assert run(tmp_path / "b") == 0
    assert (tmp_path / "b" / "drowsiness_v0.onnx").read_bytes() == onnx_path.read_bytes()


def _one_line_error(captured: pytest.CaptureResult[str]) -> str:
    """명령줄 오류의 형식: 표준 오류에 `error: ...` 한 줄이고 예외 추적은 없다. 그 줄을 돌려준다."""
    error = captured.err
    assert error.startswith("error: ") and error.endswith("\n") and error.count("\n") == 1, error
    assert "Traceback" not in error and "Traceback" not in captured.out
    return error


def _config_with_export_dir(config_path: Path, tmp_path: Path) -> tuple[Path, Path]:
    """`paths.export_dir`만 임시 폴더로 바꾼 설정 사본과 그 폴더(만들지는 않는다).

    `--out-dir`을 주지 않는 실행을 시험할 때 쓴다. 저장소의 export/에는 어떤 경우에도 쓰지 않는다.
    """
    export_dir = tmp_path / "configured_export"
    text = config_path.read_text(encoding="utf-8")
    changed, count = re.subn(r"(?m)^([ \t]+export_dir:)[ \t]*\S+", lambda match: f"{match.group(1)} {export_dir}", text)
    assert count == 1
    out = tmp_path / "config.yaml"
    out.write_text(changed, encoding="utf-8")
    assert load_config(out).paths.export_dir == export_dir
    return out, export_dir


REAL_TRAINED_ON = "UTA-RLDD 60 subjects, arm=alert_baseline"
SYNTHETIC_TRAINED_ON = "SYNTHETIC sessions (drowsy.synth) 60 subjects, arm=alert_baseline; pipeline check only"


def _save(
    path: Path, wrapper: ExportWrapper, kind: str, hyper: dict[str, object], spec_path: Path,
    temperature: float = TEMPERATURE, trained_on: str = REAL_TRAINED_ON, synthetic: bool = False,
    fast: bool = False,
) -> Path:  # fmt: skip
    """v2 체크포인트. 해시는 spec_path(`write_feature_spec`이 쓴 파일) 바이트의 sha256이다."""
    spec_hash = file_sha256(spec_path)
    save_checkpoint(path, wrapper.net, kind, 17, hyper, temperature, trained_on, spec_hash, synthetic, fast=fast)
    return path


def test_cli_refuses_a_config_that_changes_the_contract(
    tmp_path: Path, config_path: Path, feature_spec_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text = config_path.read_text(encoding="utf-8")
    assert text.count("opset: 17") == 1
    changed = tmp_path / "config.yaml"
    changed.write_text(text.replace("opset: 17", "opset: 18"), encoding="utf-8")
    out_dir = tmp_path / "out"
    args = ["--config", str(changed), "--random-init", "--spec", str(feature_spec_path), "--out-dir", str(out_dir)]
    assert export_onnx.main(args) == 2
    error = _one_line_error(capsys.readouterr())
    assert error == "error: config export.opset is 18 but the ONNX contract fixes it at 17\n"
    assert not out_dir.exists()

    # 음수 버전과 없는 설정 파일도 같은 형식의 오류다.
    args = ["--config", str(config_path), "--random-init", "--spec", str(feature_spec_path), "--out-dir", str(out_dir)]
    assert export_onnx.main([*args, "--version", "-1"]) == 2
    assert "--version must be a non-negative integer, got -1" in _one_line_error(capsys.readouterr())
    assert export_onnx.main(["--config", str(tmp_path / "missing.yaml"), "--random-init"]) == 2
    assert "missing.yaml" in _one_line_error(capsys.readouterr())
    assert not out_dir.exists()


def test_cli_checkpoint_exports_trained_model_and_int8(
    tmp_path: Path, config_path: Path, trained_spec_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = load_feature_spec(trained_spec_path)
    hyper = {"channels": 16, "dropout": 0.3}
    wrapper = _make_wrapper(KIND_TCN, spec, **hyper)
    checkpoint = _save(tmp_path / "final.pt", wrapper, KIND_TCN, hyper, trained_spec_path)

    out_dir = tmp_path / "export"
    code = export_onnx.main(
        ["--config", str(config_path), "--checkpoint", str(checkpoint), "--version", "7",
         "--spec", str(trained_spec_path), "--out-dir", str(out_dir)]
    )  # fmt: skip
    assert code == 0
    captured = capsys.readouterr()
    printed = captured.out
    assert captured.err == ""
    assert "parameters 8547" in printed and "int8 verify_contract: OK" in printed
    assert "removed" not in printed  # 지울 예전 INT8 파일이 없었다
    assert sorted(path.name for path in out_dir.iterdir()) == [
        "drowsiness_v7.onnx",
        "drowsiness_v7_int8.onnx",
        "feature_spec_v7.json",
    ]
    verify_contract(out_dir / "drowsiness_v7.onnx", out_dir / "feature_spec_v7.json")
    verify_contract(out_dir / "drowsiness_v7_int8.onnx", out_dir / "feature_spec_v7.json", ALLOWED_OPS_INT8)
    verify_file_name(out_dir / "drowsiness_v7.onnx")
    verify_file_name(out_dir / "drowsiness_v7_int8.onnx")
    # INT8 파일은 같은 버전의 사본을 같이 쓴다.
    assert exported_spec_path(out_dir / "drowsiness_v7_int8.onnx") == out_dir / "feature_spec_v7.json"
    metadata = read_metadata(out_dir / "drowsiness_v7.onnx")
    assert metadata["model_version"] == "7"
    assert metadata["trained_on"] == REAL_TRAINED_ON
    # 모델이 가리키는 사본의 해시는 체크포인트에 적힌, 학습이 쓴 spec의 해시다.
    assert metadata["feature_spec_hash"] == file_sha256(trained_spec_path)
    assert metadata["feature_spec_hash"] == file_sha256(out_dir / "feature_spec_v7.json")
    # 내보낸 그래프는 체크포인트의 가중치와 온도를 쓴다: 같은 래퍼를 직접 내보낸 파일과 바이트 단위로 같다.
    assert parity_max_abs_diff(wrapper, out_dir / "drowsiness_v7.onnx", random_raw_inputs(spec, 20, seed=6)) < 1e-4
    direct = export(
        wrapper, trained_spec_path, tmp_path / "direct" / "drowsiness_v7.onnx",
        {"model_version": "7", "trained_on": REAL_TRAINED_ON},
    )  # fmt: skip
    assert direct.read_bytes() == (out_dir / "drowsiness_v7.onnx").read_bytes()


def test_cli_checkpoint_gru_has_no_int8_and_temperature_must_match_spec(
    tmp_path: Path, config_path: Path, trained_spec_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = load_feature_spec(trained_spec_path)
    wrapper = _make_wrapper(KIND_GRU, spec)
    good = _save(tmp_path / "gru.pt", wrapper, KIND_GRU, {"hidden": 64}, trained_spec_path)
    args = ["--config", str(config_path), "--version", "2", "--spec", str(trained_spec_path)]
    assert export_onnx.main([*args, "--checkpoint", str(good), "--out-dir", str(tmp_path / "ok")]) == 0
    assert sorted(path.name for path in (tmp_path / "ok").iterdir()) == ["drowsiness_v2.onnx", "feature_spec_v2.json"]
    capsys.readouterr()

    # 해시는 맞지만 온도가 spec과 다른 체크포인트(손으로 만든 경우). 내보내지 않고, spec을 고치라고 하지도 않는다.
    stale = _save(tmp_path / "stale.pt", wrapper, KIND_GRU, {"hidden": 64}, trained_spec_path, temperature=1.0)
    assert export_onnx.main([*args, "--checkpoint", str(stale), "--out-dir", str(tmp_path / "bad")]) == 2
    error = _one_line_error(capsys.readouterr())
    assert "ExportWrapper temperature 1.0 does not match standardization.temperature 1.7" in error
    assert "write the calibrated temperature into the spec" not in error
    assert not (tmp_path / "bad").exists()


def test_cli_checkpoint_requires_spec_and_version(
    tmp_path: Path, config_path: Path, trained_spec_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config, export_dir = _config_with_export_dir(config_path, tmp_path)
    spec = load_feature_spec(trained_spec_path)
    checkpoint = _save(tmp_path / "gru.pt", _make_wrapper(KIND_GRU, spec), KIND_GRU, {"hidden": 64}, trained_spec_path)
    base = ["--config", str(config), "--checkpoint", str(checkpoint)]

    # --spec이 없으면 설정의 기본 spec(평균 0, 표준편차 1)으로 넘어가지 않고 멈춘다.
    assert export_onnx.main([*base, "--version", "1"]) == 2
    error = _one_line_error(capsys.readouterr())
    assert "--checkpoint needs --spec" in error and "feature_spec_v{N}.json" in error
    # --version이 없으면 설정의 export.model_version(0, 배포한 v0의 번호)으로 넘어가지 않고 멈춘다.
    assert export_onnx.main([*base, "--spec", str(trained_spec_path)]) == 2
    assert "--checkpoint needs --version" in _one_line_error(capsys.readouterr())
    assert export_onnx.main(base) == 2
    _one_line_error(capsys.readouterr())
    assert not export_dir.exists()

    # 둘 다 주면 (합성 모델이 아니므로) 설정의 내보내기 폴더에 쓴다.
    assert export_onnx.main([*base, "--spec", str(trained_spec_path), "--version", "1"]) == 0
    assert sorted(path.name for path in export_dir.iterdir()) == ["drowsiness_v1.onnx", "feature_spec_v1.json"]


def test_cli_checkpoint_rejects_a_spec_that_is_not_the_trained_one(
    tmp_path: Path, config_path: Path, trained_spec_path: Path, feature_spec_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    spec = load_feature_spec(trained_spec_path)
    wrapper = _make_wrapper(KIND_GRU, spec)
    checkpoint = _save(tmp_path / "gru.pt", wrapper, KIND_GRU, {"hidden": 64}, trained_spec_path)
    out_dir = tmp_path / "out"

    def run(checkpoint_path: Path, spec_path: Path) -> int:
        return export_onnx.main(
            ["--config", str(config_path), "--checkpoint", str(checkpoint_path), "--spec", str(spec_path),
             "--version", "1", "--out-dir", str(out_dir)]
        )  # fmt: skip

    # (1) 설계값 spec(평균 0, 표준편차 1)에 체크포인트의 온도만 적어 넣은 파일. 온도 비교만으로는 통과하던 경우다:
    #     래퍼를 그 spec으로 만들기 때문에 계약 검사와 일치 검증도 통과하고, 표준화 상수가 틀린 모델이 나간다.
    hinted = _write_spec(feature_spec_path, tmp_path / "hinted.json", [0.0] * 17, [1.0] * 17, TEMPERATURE)
    assert load_feature_spec(hinted).standardization.temperature == TEMPERATURE
    assert run(checkpoint, hinted) == 2
    error = _one_line_error(capsys.readouterr())
    assert "is not the feature spec this checkpoint was trained with" in error
    assert file_sha256(hinted) in error and file_sha256(trained_spec_path) in error
    assert not out_dir.exists()

    # (2) 온도가 1.0인 체크포인트와 설정의 기본 spec. 온도가 같아서 아무 오류 없이 나가던 경우다.
    trained_t1 = _write_spec(feature_spec_path, tmp_path / "trained_t1.json", MEANS, STDS, 1.0)
    wrapper_t1 = _make_wrapper(KIND_GRU, load_feature_spec(trained_t1))
    checkpoint_t1 = _save(tmp_path / "t1.pt", wrapper_t1, KIND_GRU, {"hidden": 64}, trained_t1, temperature=1.0)
    assert run(checkpoint_t1, feature_spec_path) == 2
    assert "is not the feature spec this checkpoint was trained with" in _one_line_error(capsys.readouterr())
    assert not out_dir.exists()

    # (3) 평균 하나만 조금 다른 spec.
    means = list(MEANS)
    means[4] += 0.01
    nudged = _write_spec(feature_spec_path, tmp_path / "nudged.json", means, STDS, TEMPERATURE)
    assert run(checkpoint, nudged) == 2
    assert "is not the feature spec this checkpoint was trained with" in _one_line_error(capsys.readouterr())
    assert not out_dir.exists()

    # (4) 내용은 학습한 spec과 같고 형식만 다른 파일(들여쓰기 없음)은 받는다. 모델 옆에 쓰는 사본은 고정된 형식이라
    #     학습이 쓴 파일과 바이트 단위로 같고, 그 해시가 체크포인트의 값이다.
    compact = tmp_path / "compact.json"
    compact.write_text(json.dumps(read_feature_spec_dict(trained_spec_path), ensure_ascii=False), encoding="utf-8")
    assert file_sha256(compact) != file_sha256(trained_spec_path)
    assert run(checkpoint, compact) == 0
    capsys.readouterr()
    assert (out_dir / "feature_spec_v1.json").read_bytes() == trained_spec_path.read_bytes()
    assert read_metadata(out_dir / "drowsiness_v1.onnx")["feature_spec_hash"] == file_sha256(trained_spec_path)
    # (5) 학습이 쓴 파일 그대로면 같은 모델이 나온다.
    before = (out_dir / "drowsiness_v1.onnx").read_bytes()
    assert run(checkpoint, trained_spec_path) == 0
    assert (out_dir / "drowsiness_v1.onnx").read_bytes() == before


def test_cli_rejects_v1_and_unreadable_checkpoints(
    tmp_path: Path, config_path: Path, trained_spec_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = load_feature_spec(trained_spec_path)
    net = _make_wrapper(KIND_GRU, spec).net
    out_dir = tmp_path / "out"

    def run(checkpoint_path: Path) -> int:
        return export_onnx.main(
            ["--config", str(config_path), "--checkpoint", str(checkpoint_path), "--spec", str(trained_spec_path),
             "--version", "1", "--out-dir", str(out_dir)]
        )  # fmt: skip

    # v1 체크포인트(해시와 synthetic이 없다): 다시 학습하라는 안내와 함께 거부한다.
    v1 = tmp_path / "v1.pt"
    torch.save(
        {
            "format": "drowsy-checkpoint-v1",
            "kind": KIND_GRU,
            "n_features": 17,
            "hyper": {"hidden": 64},
            "state_dict": net.state_dict(),
            "temperature": TEMPERATURE,
            "trained_on": REAL_TRAINED_ON,
        },
        v1,
    )
    assert run(v1) == 2
    error = _one_line_error(capsys.readouterr())
    assert "drowsy-checkpoint-v1" in error and "re-run `python -m drowsy.train final`" in error

    garbage = tmp_path / "garbage.pt"
    garbage.write_bytes(b"not a checkpoint\x00" * 50)
    assert run(garbage) == 2
    assert "not a readable drowsy-checkpoint-v2 checkpoint" in _one_line_error(capsys.readouterr())

    assert run(tmp_path / "missing.pt") == 2
    assert "missing.pt" in _one_line_error(capsys.readouterr())
    assert not out_dir.exists()


def test_refuse_synthetic_export_covers_the_export_dir_and_everything_below(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export_dir = tmp_path / "export"
    refused = [
        export_dir,
        export_dir / "candidate",
        export_dir / "a" / "b",
        tmp_path / "other" / ".." / "export",
        tmp_path / "other" / ".." / "export" / "x",
    ]
    link = tmp_path / "link_to_export"
    try:
        link.symlink_to(export_dir, target_is_directory=True)
    except OSError:  # 심볼릭 링크를 만들 수 없는 파일 시스템
        pass
    else:
        refused += [link, link / "sub"]
    for out_dir in refused:
        with pytest.raises(ValueError, match=r"must not be exported to paths.export_dir \(.*\) or a folder below it"):
            refuse_synthetic_export(out_dir, export_dir)
    # 이름이 비슷한 이웃 폴더, 상위 폴더, 다른 곳의 같은 이름 폴더는 막지 않는다.
    for out_dir in (tmp_path, tmp_path / "export2", tmp_path / "export_candidate", tmp_path / "out" / "export"):
        refuse_synthetic_export(out_dir, export_dir)

    # 상대 경로는 실행 위치 기준으로 풀어 비교한다(설정의 기본값이 상대 경로 "export"다).
    monkeypatch.chdir(tmp_path)
    for out_dir in ("export", "./export/", "export/sub", "out/../export/sub"):
        with pytest.raises(ValueError, match="must not be exported to paths.export_dir"):
            refuse_synthetic_export(out_dir, Path("export"))
        with pytest.raises(ValueError, match="must not be exported to paths.export_dir"):
            refuse_synthetic_export(out_dir, export_dir)
    refuse_synthetic_export("out", "export")
    refuse_synthetic_export(Path("out") / "synth", Path("export"))


def test_repo_root_is_the_folder_that_holds_the_package(repo_root: Path) -> None:
    assert export_onnx.REPO_ROOT == repo_root
    assert (export_onnx.REPO_ROOT / "drowsy" / "export_onnx.py").samefile(export_onnx.__file__)
    assert (export_onnx.REPO_ROOT / "export" / "drowsiness_v0.onnx").is_file()


def test_relative_export_dir_is_protected_from_any_working_directory(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """상대 경로로 적힌 `paths.export_dir`(기본값 `export`)은 실행 위치 기준 폴더와 저장소 루트 기준 폴더를 모두 가리킨다.

    전에는 실행 위치 기준으로만 풀어서, 저장소 루트가 아닌 곳에서 실행하면 저장소의 export/가 내보내기 폴더로 보이지 않았다.
    """
    # (1) 진짜 저장소 루트로: 다른 폴더에서 실행해도 저장소의 export/와 그 아래는 내보내기 폴더다(경로만 견준다. 쓰지 않는다).
    monkeypatch.chdir(tmp_path)
    shipped = repo_root / "export"
    assert is_export_dir_or_below(shipped, "export") and is_export_dir_or_below(shipped / "candidate", Path("export"))
    assert is_export_dir_or_below(str(shipped) + "/", "export/")
    assert not is_export_dir_or_below(repo_root, "export") and not is_export_dir_or_below(repo_root / "out", "export")
    # 실행 위치 기준 폴더도 그대로 내보내기 폴더다.
    assert is_export_dir_or_below("export", "export") and is_export_dir_or_below(tmp_path / "export" / "x", "export")
    # 절대 경로로 적힌 export_dir은 그 폴더 하나를 가리킨다. 저장소의 export/는 설정이 다른 폴더를 가리켜도 내보내기 폴더다.
    assert is_export_dir_or_below("export/sub", tmp_path / "export")
    assert is_export_dir_or_below(shipped, tmp_path / "export") and is_export_dir_or_below(shipped / "v9", "release")
    assert not is_export_dir_or_below(repo_root / "export2", tmp_path / "export")
    assert not is_export_dir_or_below(repo_root / "out" / "export", tmp_path / "export")

    # (2) 저장소 루트를 임시 폴더로 바꿔서: 그 아래의 깊은 폴더에서 실행한다.
    repo = tmp_path / "repo"
    work = repo / "work" / "deep"
    work.mkdir(parents=True)
    monkeypatch.setattr(export_onnx, "REPO_ROOT", repo)
    monkeypatch.chdir(work)
    refused: list[str | Path] = [
        "export", "./export/sub", work / "export",  # 실행 위치 기준
        "../../export", "../../export/", "../../export/v1", repo / "export", repo / "work" / ".." / "export" / "x",
    ]  # fmt: skip
    link = repo / "link_to_export"
    try:
        link.symlink_to(repo / "export", target_is_directory=True)
    except OSError:  # 심볼릭 링크를 만들 수 없는 파일 시스템
        pass
    else:
        refused += [link, "../../link_to_export/sub"]
    synthetic_message = r"trained on synthetic data must not be exported to paths.export_dir \(export\)"
    fast_message = r"fast profile \(smoke run\) must not be exported to paths.export_dir \(export\)"
    for out_dir in refused:
        assert is_export_dir_or_below(out_dir, "export"), out_dir
        with pytest.raises(ValueError, match=synthetic_message):
            refuse_synthetic_export(out_dir, "export")
        with pytest.raises(ValueError, match=fast_message):
            refuse_fast_export(out_dir, Path("export"))
    # 두 폴더가 아닌 곳: 중간 폴더의 export, 이름이 비슷한 이웃 폴더, 상위 폴더, 저장소 밖의 같은 이름 폴더.
    for out_dir in ("../export", repo / "export2", "../../export2", "out", repo, tmp_path / "export"):
        assert not is_export_dir_or_below(out_dir, "export"), out_dir
        refuse_synthetic_export(out_dir, "export")
        refuse_fast_export(out_dir, "export")
    # 설정에 `./export`나 `x/../export`로 적혀 있어도 같은 폴더다.
    assert is_export_dir_or_below(repo / "export", "./export") and is_export_dir_or_below("export/a", "x/../export")


def _config_with_relative_export_dir(config_path: Path, tmp_path: Path, value: str = "export") -> Path:
    """`paths.export_dir`을 value로 적은 설정 사본. 기본값은 상대 경로 `export`다(저장소의 기본 설정이 그렇게 적혀 있다)."""
    text = config_path.read_text(encoding="utf-8")
    changed, count = re.subn(r"(?m)^([ \t]+export_dir:)[ \t]*\S+", lambda match: f"{match.group(1)} {value}", text)
    assert count == 1
    out = tmp_path / "relative_export.yaml"
    out.write_text(changed, encoding="utf-8")
    assert load_config(out).paths.export_dir == Path(value)
    return out


def test_cli_refuses_synthetic_and_fast_checkpoints_from_another_working_directory(
    tmp_path: Path, config_path: Path, trained_spec_path: Path, capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    """`export_onnx --checkpoint`의 두 울타리는 실행 위치와 무관하다: 상대 경로와 절대 경로의 `--out-dir` 모두.

    저장소 루트는 임시 폴더 `repo`로 바꿔 둔다. 검사가 뚫려도 저장소의 진짜 export/에는 아무것도 쓰이지 않는다.
    """
    config = _config_with_relative_export_dir(config_path, tmp_path)
    repo = tmp_path / "repo"
    work = repo / "work"
    work.mkdir(parents=True)
    monkeypatch.setattr(export_onnx, "REPO_ROOT", repo)
    monkeypatch.chdir(work)
    spec = load_feature_spec(trained_spec_path)
    wrapper = _make_wrapper(KIND_GRU, spec)
    hyper: dict[str, object] = {"hidden": 64}
    synthetic = _save(
        tmp_path / "synthetic.pt", wrapper, KIND_GRU, hyper, trained_spec_path,
        trained_on=SYNTHETIC_TRAINED_ON, synthetic=True,
    )  # fmt: skip
    fast = _save(tmp_path / "fast.pt", wrapper, KIND_GRU, hyper, trained_spec_path, fast=True)
    real = _save(tmp_path / "real.pt", wrapper, KIND_GRU, hyper, trained_spec_path)
    base = ["--config", str(config), "--spec", str(trained_spec_path), "--version", "1"]

    refused: list[str | None] = [
        None,  # --out-dir 없음: 설정의 `export`를 실행 위치 기준으로 쓴다(repo/work/export)
        "export",
        "export/",
        "export/candidate",
        str(work / "export"),
        str(repo / "export"),  # 저장소 루트의 export/를 절대 경로로
        str(repo / "export") + "/",
        str(repo / "export" / "candidate"),
        "../export",  # 저장소 루트의 export/를 상대 경로로
        "../export/candidate",
        "x/../../export",
    ]
    link = repo / "link_to_export"
    try:
        link.symlink_to(repo / "export", target_is_directory=True)
    except OSError:  # 심볼릭 링크를 만들 수 없는 파일 시스템
        pass
    else:
        refused += [str(link), "../link_to_export/v1"]
    for checkpoint, message in (
        (synthetic, "a model trained on synthetic data must not be exported to paths.export_dir (export)"),
        (fast, "a model trained with the fast profile (smoke run) must not be exported to paths.export_dir (export)"),
    ):
        for out_dir in refused:
            extra = [] if out_dir is None else ["--out-dir", out_dir]
            assert export_onnx.main([*base, "--checkpoint", str(checkpoint), *extra]) == 2, out_dir
            assert message in _one_line_error(capsys.readouterr()), out_dir
            assert not (repo / "export").exists() and not (work / "export").exists()

    # 다른 폴더로는 그대로 나간다: 상대 경로(실행 위치 기준)와 절대 경로(이름이 비슷한 이웃 폴더).
    written = ["drowsiness_v1.onnx", "feature_spec_v1.json"]
    assert export_onnx.main([*base, "--checkpoint", str(synthetic), "--out-dir", "out"]) == 0
    assert sorted(path.name for path in (work / "out").iterdir()) == written
    assert export_onnx.main([*base, "--checkpoint", str(fast), "--out-dir", str(repo / "export_smoke")]) == 0
    assert sorted(path.name for path in (repo / "export_smoke").iterdir()) == written
    assert not (repo / "export").exists() and not (work / "export").exists()
    # 실제 데이터와 정식 설정으로 학습한 체크포인트는 여기서도 저장소 루트의 export/로 들어간다.
    assert export_onnx.main([*base, "--checkpoint", str(real), "--out-dir", "../export"]) == 0
    capsys.readouterr()
    assert sorted(path.name for path in (repo / "export").iterdir()) == written


def test_repository_export_folder_is_protected_whatever_the_config_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """설정의 `paths.export_dir`이 다른 폴더(절대 경로든 다른 상대 이름이든)를 가리켜도 저장소의 export/는 내보내기 폴더다."""
    repo = tmp_path / "repo"
    work = repo / "work"
    work.mkdir(parents=True)
    monkeypatch.setattr(export_onnx, "REPO_ROOT", repo)
    assert export_onnx.SHIPPED_EXPORT_DIRNAME == "export"
    elsewhere = tmp_path / "smoke_export"
    shipped = repo / "export"
    for configured in (elsewhere, "release", Path("out") / "export"):
        monkeypatch.chdir(repo)
        named = f"paths.export_dir ({configured})"
        always = f"the export folder of this repository ({shipped}; it is protected whatever paths.export_dir names, "
        always += f"here {configured})"
        # 저장소 루트에서: 상대 경로, 끝의 `/`, 아래 폴더, 절대 경로, `..`을 거친 경로.
        for out_dir in ("export", "export/", "./export/candidate", shipped, shipped / "a" / "b", "work/../export"):
            assert export_onnx.protected_export_dir(out_dir, configured) == always, (configured, out_dir)
            assert is_export_dir_or_below(out_dir, configured)
            for refuse, model in (
                (refuse_synthetic_export, "a model trained on synthetic data"),
                (refuse_fast_export, "a model trained with the fast profile (smoke run)"),
            ):
                with pytest.raises(ValueError) as caught:
                    refuse(out_dir, configured)
                assert str(caught.value) == (
                    f"{model} must not be exported to {always} or a folder below it, got {out_dir}; "
                    "pass another output folder"
                )
        # 설정이 가리키는 폴더 자신은 전과 같은 말로 막힌다.
        assert export_onnx.protected_export_dir(configured, configured) == named
        assert export_onnx.protected_export_dir(Path(configured) / "v1", configured) == named
        # 저장소의 다른 폴더와 이름이 비슷한 폴더는 막지 않는다.
        for out_dir in ("out", "export2", "reports/export", repo, tmp_path / "export", repo / "work" / "export"):
            assert export_onnx.protected_export_dir(out_dir, configured) is None, (configured, out_dir)
            refuse_synthetic_export(out_dir, configured)
            refuse_fast_export(out_dir, configured)
        # 저장소의 하위 폴더에서 실행해도 같다.
        monkeypatch.chdir(work)
        for out_dir in ("../export", "../export/candidate", shipped):
            assert export_onnx.protected_export_dir(out_dir, configured) == always, (configured, out_dir)
        assert export_onnx.protected_export_dir("export", elsewhere) is None  # repo/work/export는 그 폴더가 아니다
    try:
        (repo / "link_to_export").symlink_to(shipped, target_is_directory=True)
    except OSError:  # 심볼릭 링크를 만들 수 없는 파일 시스템
        pass
    else:
        assert is_export_dir_or_below(repo / "link_to_export" / "v1", elsewhere)
    assert not shipped.exists()  # 경로만 견줬다


@pytest.mark.parametrize("configured", ["absolute", "relative"])
def test_cli_refuses_synthetic_and_fast_checkpoints_in_the_repository_export_folder_under_another_config(
    configured: str, tmp_path: Path, config_path: Path, trained_spec_path: Path, capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    """`export_onnx --checkpoint`: 설정의 `paths.export_dir`이 다른 폴더여도 저장소의 export/에는 합성 모델과 축소 설정 모델이 들어가지 못한다.

    연기 시험용 설정은 `paths.export_dir`을 임시 폴더로 적는다. 전에는 그 설정으로 저장소 루트에서 `--out-dir export`를 주면
    두 검사를 모두 지나 커밋하는 폴더에 파일이 쓰였다. 저장소 루트는 임시 폴더 `repo`로 바꿔 둔다(진짜 export/에는 쓰지 않는다).
    """
    value = str(tmp_path / "smoke_export") if configured == "absolute" else "release"
    config = _config_with_relative_export_dir(config_path, tmp_path, value)
    repo = tmp_path / "repo"
    work = repo / "work"
    work.mkdir(parents=True)
    monkeypatch.setattr(export_onnx, "REPO_ROOT", repo)
    spec = load_feature_spec(trained_spec_path)
    wrapper = _make_wrapper(KIND_GRU, spec)
    hyper: dict[str, object] = {"hidden": 64}
    synthetic = _save(
        tmp_path / "synthetic.pt", wrapper, KIND_GRU, hyper, trained_spec_path,
        trained_on=SYNTHETIC_TRAINED_ON, synthetic=True,
    )  # fmt: skip
    fast = _save(tmp_path / "fast.pt", wrapper, KIND_GRU, hyper, trained_spec_path, fast=True)
    real = _save(tmp_path / "real.pt", wrapper, KIND_GRU, hyper, trained_spec_path)
    base = ["--config", str(config), "--spec", str(trained_spec_path), "--version", "9"]
    shipped = repo / "export"
    always = f"the export folder of this repository ({shipped}; it is protected whatever paths.export_dir names, "
    always += f"here {value})"
    cases = (
        (synthetic, "a model trained on synthetic data must not be exported to "),
        (fast, "a model trained with the fast profile (smoke run) must not be exported to "),
    )
    for cwd, out_dirs in (
        (repo, ["export", "export/", "export/candidate", str(shipped), "work/../export"]),
        (work, ["../export", "../export/candidate", str(shipped / "candidate")]),
    ):
        monkeypatch.chdir(cwd)
        for checkpoint, start in cases:
            for out_dir in out_dirs:
                assert export_onnx.main([*base, "--checkpoint", str(checkpoint), "--out-dir", out_dir]) == 2, out_dir
                error = _one_line_error(capsys.readouterr())
                assert error == (
                    f"error: {start}{always} or a folder below it, got {Path(out_dir)}; pass another output folder\n"
                ), (cwd, out_dir)
                assert not shipped.exists()
            # 설정이 가리키는 폴더(--out-dir 없이)는 전과 같이 막힌다.
            assert export_onnx.main([*base, "--checkpoint", str(checkpoint)]) == 2
            assert f"{start}paths.export_dir ({value}) or a folder below it" in _one_line_error(capsys.readouterr())
    assert sorted(path.name for path in repo.iterdir()) == ["work"] and list(work.iterdir()) == []
    assert not (tmp_path / "smoke_export").exists()

    # 다른 폴더로는 그대로 나간다(저장소 안의 다른 이름, 이름이 비슷한 이웃 폴더). 실제 데이터와 정식 설정의 체크포인트는
    # 저장소의 export/로 들어간다.
    monkeypatch.chdir(repo)
    written = ["drowsiness_v9.onnx", "feature_spec_v9.json"]
    assert export_onnx.main([*base, "--checkpoint", str(synthetic), "--out-dir", "out"]) == 0
    assert sorted(path.name for path in (repo / "out").iterdir()) == written
    assert export_onnx.main([*base, "--checkpoint", str(fast), "--out-dir", "export_smoke"]) == 0
    assert sorted(path.name for path in (repo / "export_smoke").iterdir()) == written
    assert not shipped.exists()
    assert export_onnx.main([*base, "--checkpoint", str(real), "--out-dir", "export"]) == 0
    capsys.readouterr()
    assert sorted(path.name for path in shipped.iterdir()) == written


def test_synthetic_trained_on_marker() -> None:
    assert SYNTHETIC_TRAINED_ON_PREFIX == "SYNTHETIC"
    assert is_synthetic_trained_on(SYNTHETIC_TRAINED_ON)
    assert not is_synthetic_trained_on(REAL_TRAINED_ON)
    assert not is_synthetic_trained_on(TRAINED_ON_NONE)
    assert not is_synthetic_trained_on("trained on UTA-RLDD, not SYNTHETIC")  # 맨 앞에 있을 때만이다


def test_cli_refuses_a_synthetic_checkpoint_in_the_export_dir(
    tmp_path: Path, config_path: Path, trained_spec_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config, export_dir = _config_with_export_dir(config_path, tmp_path)
    spec = load_feature_spec(trained_spec_path)
    wrapper = _make_wrapper(KIND_GRU, spec)
    hyper: dict[str, object] = {"hidden": 64}
    synthetic = _save(
        tmp_path / "synthetic.pt", wrapper, KIND_GRU, hyper, trained_spec_path,
        trained_on=SYNTHETIC_TRAINED_ON, synthetic=True,
    )  # fmt: skip
    base = ["--config", str(config), "--spec", str(trained_spec_path), "--version", "1"]

    # --out-dir을 주지 않은 경우(기본값이 paths.export_dir), 그 폴더를 직접 준 경우, 그 아래 폴더를 준 경우.
    refused_out_dirs: list[Path | None] = [
        None,
        export_dir,
        export_dir / "candidate",
        tmp_path / "elsewhere" / ".." / export_dir.name / "deep" / "er",
    ]
    for out_dir in refused_out_dirs:
        extra = [] if out_dir is None else ["--out-dir", str(out_dir)]
        assert export_onnx.main([*base, "--checkpoint", str(synthetic), *extra]) == 2
        error = _one_line_error(capsys.readouterr())
        assert "a model trained on synthetic data must not be exported to paths.export_dir" in error
        assert not export_dir.exists()

    # synthetic 값이 false여도 trained_on이 합성 표지로 시작하면 같은 취급이다(둘 중 하나라도 합성이면 막는다).
    relabelled = _save(
        tmp_path / "relabelled.pt", wrapper, KIND_GRU, hyper, trained_spec_path,
        trained_on=SYNTHETIC_TRAINED_ON, synthetic=False,
    )  # fmt: skip
    flagged = _save(tmp_path / "flagged.pt", wrapper, KIND_GRU, hyper, trained_spec_path, synthetic=True)
    for checkpoint in (relabelled, flagged):
        assert export_onnx.main([*base, "--checkpoint", str(checkpoint)]) == 2
        assert "must not be exported to paths.export_dir" in _one_line_error(capsys.readouterr())
        assert not export_dir.exists()

    # 다른 폴더(이름이 비슷한 이웃 폴더 포함)로는 나간다. trained_on은 그대로 합성임을 밝힌다.
    elsewhere = tmp_path / (export_dir.name + "_synth")
    assert export_onnx.main([*base, "--checkpoint", str(synthetic), "--out-dir", str(elsewhere)]) == 0
    capsys.readouterr()
    assert sorted(path.name for path in elsewhere.iterdir()) == ["drowsiness_v1.onnx", "feature_spec_v1.json"]
    assert read_metadata(elsewhere / "drowsiness_v1.onnx")["trained_on"] == SYNTHETIC_TRAINED_ON
    assert not export_dir.exists()

    # 실제 데이터로 학습한 체크포인트는 같은 명령으로 내보내기 폴더에 들어간다.
    real = _save(tmp_path / "real.pt", wrapper, KIND_GRU, hyper, trained_spec_path)
    assert export_onnx.main([*base, "--checkpoint", str(real)]) == 0
    assert sorted(path.name for path in export_dir.iterdir()) == ["drowsiness_v1.onnx", "feature_spec_v1.json"]


def test_refuse_fast_export_covers_the_export_dir_and_everything_below(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    message = (
        r"trained with the fast profile \(smoke run\) must not be exported to paths.export_dir \(.*\) "
        r"or a folder below it"
    )
    for out_dir in (export_dir, export_dir / "candidate", tmp_path / "other" / ".." / "export" / "x"):
        with pytest.raises(ValueError, match=message):
            refuse_fast_export(out_dir, export_dir)
    # 이름이 비슷한 이웃 폴더, 상위 폴더, 다른 곳의 같은 이름 폴더는 막지 않는다(합성 모델의 규칙과 같다).
    for out_dir in (tmp_path, tmp_path / "export2", tmp_path / "out" / "export"):
        refuse_fast_export(out_dir, export_dir)


def test_cli_refuses_a_fast_checkpoint_in_the_export_dir(
    tmp_path: Path, config_path: Path, trained_spec_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """축소 설정(`train final --fast`)으로 학습한 체크포인트는 실제 데이터의 것이어도 내보내기 폴더에 들어가지 못한다.

    `trained_on`은 정식 실행과 같은 글자열이라(합성 표지가 없다) 체크포인트의 `fast` 값만이 구분한다.
    """
    config, export_dir = _config_with_export_dir(config_path, tmp_path)
    spec = load_feature_spec(trained_spec_path)
    wrapper = _make_wrapper(KIND_GRU, spec)
    hyper: dict[str, object] = {"hidden": 64}
    fast = _save(tmp_path / "fast.pt", wrapper, KIND_GRU, hyper, trained_spec_path, fast=True)
    base = ["--config", str(config), "--spec", str(trained_spec_path), "--version", "1"]

    # --out-dir을 주지 않은 경우(기본값이 paths.export_dir), 그 폴더를 직접 준 경우, 그 아래 폴더를 준 경우.
    refused_out_dirs: list[Path | None] = [
        None,
        export_dir,
        export_dir / "smoke",
        tmp_path / "elsewhere" / ".." / export_dir.name / "deep" / "er",
    ]
    for out_dir in refused_out_dirs:
        extra = [] if out_dir is None else ["--out-dir", str(out_dir)]
        assert export_onnx.main([*base, "--checkpoint", str(fast), *extra]) == 2
        error = _one_line_error(capsys.readouterr())
        shown = export_dir if out_dir is None else out_dir
        assert error == (
            "error: a model trained with the fast profile (smoke run) must not be exported to paths.export_dir "
            f"({export_dir}) or a folder below it, got {shown}; pass another output folder\n"
        )
        assert not export_dir.exists()

    # 다른 폴더로는 나간다. trained_on은 정식 실행과 같은 글자열 그대로다.
    elsewhere = tmp_path / (export_dir.name + "_smoke")
    assert export_onnx.main([*base, "--checkpoint", str(fast), "--out-dir", str(elsewhere)]) == 0
    capsys.readouterr()
    assert sorted(path.name for path in elsewhere.iterdir()) == ["drowsiness_v1.onnx", "feature_spec_v1.json"]
    assert read_metadata(elsewhere / "drowsiness_v1.onnx")["trained_on"] == REAL_TRAINED_ON
    assert not export_dir.exists()

    # 합성이면서 축소 설정인 체크포인트도 막힌다(먼저 걸리는 것은 합성 모델의 규칙이다).
    both = _save(
        tmp_path / "both.pt", wrapper, KIND_GRU, hyper, trained_spec_path,
        trained_on=SYNTHETIC_TRAINED_ON, synthetic=True, fast=True,
    )  # fmt: skip
    assert export_onnx.main([*base, "--checkpoint", str(both)]) == 2
    assert "trained on synthetic data must not be exported" in _one_line_error(capsys.readouterr())
    assert not export_dir.exists()

    # `fast` 키가 없는 v2 체크포인트(이 키가 생기기 전에 쓴 파일)는 축소 설정이 아닌 것으로 읽혀 내보내기 폴더에 들어간다.
    payload = torch.load(fast, map_location="cpu", weights_only=True)
    assert payload.pop("fast") is True
    without_key = tmp_path / "without_key.pt"
    torch.save(payload, without_key)
    assert export_onnx.main([*base, "--checkpoint", str(without_key)]) == 0
    capsys.readouterr()
    assert sorted(path.name for path in export_dir.iterdir()) == ["drowsiness_v1.onnx", "feature_spec_v1.json"]


def test_cli_reexport_of_a_version_removes_the_stale_int8_file(
    tmp_path: Path, config_path: Path, trained_spec_path: Path, feature_spec_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    out_dir = tmp_path / "export"
    trained = load_feature_spec(trained_spec_path)
    hyper: dict[str, object] = {"channels": 16}
    tcn = _save(tmp_path / "tcn.pt", _make_wrapper(KIND_TCN, trained, **hyper), KIND_TCN, hyper, trained_spec_path)
    other_spec = _write_spec(feature_spec_path, tmp_path / "other.json", [m + 1.0 for m in MEANS], STDS, 1.3)
    gru_wrapper = _make_wrapper(KIND_GRU, load_feature_spec(other_spec))
    gru = _save(tmp_path / "gru.pt", gru_wrapper, KIND_GRU, {"hidden": 64}, other_spec, temperature=1.3)

    def run(checkpoint: Path, spec_path: Path) -> int:
        return export_onnx.main(
            ["--config", str(config_path), "--checkpoint", str(checkpoint), "--spec", str(spec_path),
             "--version", "7", "--out-dir", str(out_dir)]
        )  # fmt: skip

    assert run(tcn, trained_spec_path) == 0
    capsys.readouterr()
    int8_path = out_dir / "drowsiness_v7_int8.onnx"
    assert sorted(path.name for path in out_dir.iterdir()) == [
        "drowsiness_v7.onnx", "drowsiness_v7_int8.onnx", "feature_spec_v7.json",
    ]  # fmt: skip

    # 같은 버전을 GRU로 다시 내보낸다. INT8 파일은 만들지 않으므로 예전 것이 남으면 안 된다.
    assert run(gru, other_spec) == 0
    printed = capsys.readouterr().out
    assert sorted(path.name for path in out_dir.iterdir()) == ["drowsiness_v7.onnx", "feature_spec_v7.json"]
    assert f"removed {int8_path}" in printed
    assert (out_dir / "feature_spec_v7.json").read_bytes() == other_spec.read_bytes()

    # TCN으로 다시 내보내면 INT8 파일도 새로 생기고 지금의 사본과 맞는다.
    assert run(tcn, trained_spec_path) == 0
    capsys.readouterr()
    verify_contract(int8_path, out_dir / "feature_spec_v7.json", ALLOWED_OPS_INT8)
    assert read_metadata(int8_path) == read_metadata(out_dir / "drowsiness_v7.onnx")


def test_cli_failed_reexport_keeps_the_previous_pair(
    tmp_path: Path, config_path: Path, trained_spec_path: Path, feature_spec_path: Path,
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    out_dir = tmp_path / "export"
    trained = load_feature_spec(trained_spec_path)
    first = _save(tmp_path / "a.pt", _make_wrapper(KIND_GRU, trained), KIND_GRU, {"hidden": 64}, trained_spec_path)
    other_spec = _write_spec(feature_spec_path, tmp_path / "other.json", [m + 1.0 for m in MEANS], STDS, 1.3)
    second_wrapper = _make_wrapper(KIND_GRU, load_feature_spec(other_spec), seed=1)
    second = _save(tmp_path / "b.pt", second_wrapper, KIND_GRU, {"hidden": 64}, other_spec, temperature=1.3)

    def run(checkpoint: Path, spec_path: Path) -> int:
        return export_onnx.main(
            ["--config", str(config_path), "--checkpoint", str(checkpoint), "--spec", str(spec_path),
             "--version", "2", "--out-dir", str(out_dir)]
        )  # fmt: skip

    assert run(first, trained_spec_path) == 0
    capsys.readouterr()
    before = _folder_bytes(out_dir)
    assert sorted(before) == ["drowsiness_v2.onnx", "feature_spec_v2.json"]

    # 다음 내보내기는 계약 검사에서 걸린다(허용 목록에서 GRU를 뺀다).
    monkeypatch.setattr(export_onnx, "ALLOWED_OPS", ALLOWED_OPS - {"GRU"})
    assert run(second, other_spec) == 2
    error = _one_line_error(capsys.readouterr())
    assert error == "error: operators not in the allowed list: ['GRU']\n"
    assert _folder_bytes(out_dir) == before
