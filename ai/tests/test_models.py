"""drowsy.models 테스트 (PLAN 5.1절, 10.1절 "모델"). 합성 입력만 쓴다."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from drowsy.models import (
    CHECKPOINT_FORMAT,
    KIND_GRU,
    KIND_TCN,
    KINDS,
    CausalConv1d,
    CausalTCN,
    ExportWrapper,
    GRUClassifier,
    ResidualBlock,
    build_model,
    count_parameters,
    load_checkpoint,
    save_checkpoint,
)
from drowsy.spec import FeatureSpec
from drowsy.types import FEATURE_NAMES, N_FEATURES

WINDOW = 120
LOG1P_COLUMNS = [9, 10, 11, 12, 14]


def _randomize_batchnorm(net: nn.Module, seed: int) -> None:
    """배치 정규화가 항등 변환이 아니게 만든다(초기값은 평균 0, 분산 1, scale 1, shift 0)."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in net.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.running_mean.copy_(torch.randn(module.num_features, generator=generator) * 0.5)
                module.running_var.copy_(torch.rand(module.num_features, generator=generator) * 1.5 + 0.5)
                module.weight.copy_(torch.rand(module.num_features, generator=generator) + 0.5)
                module.bias.copy_(torch.randn(module.num_features, generator=generator) * 0.3)


def _with_stats(spec: FeatureSpec, changes: dict[int, tuple[float, float]], **standardization: float) -> FeatureSpec:
    """특징 index -> (mean, std)를 바꾼 spec 사본. 파일 검증을 거치지 않으므로 std 0도 넣을 수 있다."""
    features = tuple(
        dataclasses.replace(feat, mean=changes[i][0], std=changes[i][1]) if i in changes else feat
        for i, feat in enumerate(spec.features)
    )
    return dataclasses.replace(
        spec, features=features, standardization=dataclasses.replace(spec.standardization, **standardization)
    )


class _Probe(nn.Module):
    """받은 z를 기억하고 정해진 로짓을 돌려주는 가짜 신경망."""

    def __init__(self, logits: tuple[float, float, float]) -> None:
        super().__init__()
        self.logits = torch.tensor([logits], dtype=torch.float32)
        self.seen: Tensor | None = None

    def forward(self, z: Tensor) -> Tensor:
        self.seen = z
        return self.logits.expand(z.shape[0], -1)


# ---------------------------------------------------------------------------
# 구조와 파라미터 수
# ---------------------------------------------------------------------------


def test_parameter_counts_match_hand_computation() -> None:
    # TCN, F = 17, 채널 32, 커널 3:
    #   블록 0: conv1 17·32·3+32 = 1664, bn 64, conv2 32·32·3+32 = 3104, bn 64, 1×1 잔차 17·32+32 = 576 -> 5472
    #   블록 1~4: 2·(3104 + 64) = 6336씩 -> 25344
    #   선형: 32·3+3 = 99
    assert count_parameters(build_model(KIND_TCN, 17)) == 5472 + 25344 + 99 == 30915
    # GRU, F = 17, 은닉 64: 3·(17·64 + 64·64 + 64 + 64) = 15936, 선형 64·3+3 = 195
    assert count_parameters(build_model(KIND_GRU, 17)) == 15936 + 195 == 16131
    # 계획서 5.1절의 F = 24 실측값
    assert count_parameters(build_model(KIND_TCN, 24)) == 31811
    assert count_parameters(build_model(KIND_GRU, 24)) == 17475


@pytest.mark.parametrize(
    ("kind", "hyper", "expected"),
    [
        (KIND_TCN, {"channels": 32, "dropout": 0.1}, 30915),
        (KIND_TCN, {"channels": 16, "dropout": 0.3}, 8547),  # 1968 + 4·1632 + 51
        (KIND_GRU, {"hidden": 64, "dropout": 0.1}, 16131),
        (KIND_GRU, {"hidden": 32, "dropout": 0.3}, 4995),  # 3·(17·32 + 32·32 + 64) + 99
    ],
)
def test_search_grid_models_stay_under_100k_parameters(kind: str, hyper: dict[str, float], expected: int) -> None:
    net = build_model(kind, N_FEATURES, **hyper)
    assert count_parameters(net) == expected
    assert count_parameters(net) < 100_000


def test_count_parameters_ignores_buffers(feature_spec: FeatureSpec) -> None:
    net = build_model(KIND_TCN, N_FEATURES)
    assert count_parameters(ExportWrapper(net, feature_spec, 1.0)) == count_parameters(net)


@pytest.mark.parametrize("kind", KINDS)
def test_forward_shape_is_batch_by_three(kind: str) -> None:
    net = build_model(kind, N_FEATURES).eval()
    logits = net(torch.zeros(4, WINDOW, N_FEATURES))
    assert logits.shape == (4, 3)
    assert torch.isfinite(logits).all()


def test_build_model_kinds_and_errors() -> None:
    assert isinstance(build_model(KIND_TCN, N_FEATURES), CausalTCN)
    assert isinstance(build_model(KIND_GRU, N_FEATURES), GRUClassifier)
    with pytest.raises(ValueError, match="unknown model kind"):
        build_model("lstm", N_FEATURES)
    with pytest.raises(TypeError):
        build_model(KIND_GRU, N_FEATURES, channels=32)


def test_tcn_structure() -> None:
    net = CausalTCN(N_FEATURES)
    blocks = list(net.blocks)
    assert len(blocks) == 5
    for block, dilation in zip(blocks, (1, 2, 4, 8, 16)):
        assert isinstance(block, ResidualBlock)
        for conv in (block.conv1, block.conv2):
            # 배치 정규화가 합성곱 바로 뒤에 있어야 내보낼 때 합쳐진다.
            assert [name for name, _ in conv.named_children()] == ["conv", "bn"]
            assert conv.conv.kernel_size == (3,)
            assert conv.conv.dilation == (dilation,)
            assert conv.conv.padding == (0,)
            assert conv.left_pad == 2 * dilation
    assert net.head.in_features == 32 and net.head.out_features == 3


def test_gru_is_one_unidirectional_layer_read_at_the_last_step() -> None:
    torch.manual_seed(0)
    net = GRUClassifier(N_FEATURES).eval()
    assert net.gru.num_layers == 1 and net.gru.bidirectional is False and net.gru.batch_first is True
    z = torch.randn(3, WINDOW, N_FEATURES)
    with torch.no_grad():
        outputs, _ = net.gru(z)
        assert torch.equal(net(z), net.head(outputs[:, -1, :]))


def test_tcn_reads_only_the_last_step() -> None:
    torch.manual_seed(0)
    net = CausalTCN(N_FEATURES).eval()
    z = torch.randn(2, WINDOW, N_FEATURES)
    with torch.no_grad():
        reference = net(z)

        def scramble_all_but_last(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> Tensor:
            changed = output.clone()
            changed[:, :, :-1] = torch.randn_like(changed[:, :, :-1]) * 10.0
            return changed

        handle = net.blocks.register_forward_hook(scramble_all_but_last)
        try:
            assert torch.equal(net(z), reference)
        finally:
            handle.remove()


# ---------------------------------------------------------------------------
# 인과성과 수용 영역
# ---------------------------------------------------------------------------


def _conv_stack_activations(net: CausalTCN, z: Tensor) -> dict[str, Tensor]:
    """합성곱 층(인과 합성곱 10개, 1×1 잔차 합성곱)과 잔차 블록 5개의 출력 [B, C, T]."""
    captured: dict[str, Tensor] = {}
    handles = []
    for name, module in net.named_modules():
        if isinstance(module, (CausalConv1d, ResidualBlock)) or name.endswith("shortcut"):
            handles.append(
                module.register_forward_hook(
                    lambda _m, _i, output, name=name: captured.__setitem__(name, output.detach().clone())
                )
            )
    try:
        with torch.no_grad():
            net(z)
    finally:
        for handle in handles:
            handle.remove()
    return captured


@pytest.mark.parametrize("t", [1, 2, 37, 64, 119])
def test_tcn_conv_stack_is_strictly_causal(t: int) -> None:
    torch.manual_seed(0)
    net = CausalTCN(N_FEATURES)
    _randomize_batchnorm(net, seed=1)
    net.eval()
    z = torch.randn(2, WINDOW, N_FEATURES)
    changed = z.clone()
    changed[:, t:, :] = torch.randn(2, WINDOW - t, N_FEATURES) * 10.0 + 3.0

    before = _conv_stack_activations(net, z)
    after = _conv_stack_activations(net, changed)

    assert len(before) == 5 * 2 + 5 + 5  # 인과 합성곱 10, 잔차 경로 5(첫 블록만 합성곱), 블록 5
    for name, activation in before.items():
        assert activation.shape[0] == 2 and activation.shape[2] == WINDOW, name
        # t 이후의 입력을 바꿔도 t 이전의 활성값은 비트 단위로 같아야 한다.
        assert torch.equal(activation[:, :, :t], after[name][:, :, :t]), name
        # 스텝 t의 활성값은 바뀐다(검사가 공허하지 않다는 확인).
        assert not torch.equal(activation[:, :, t], after[name][:, :, t]), name


def _steps_with_gradient(net: nn.Module, steps: int) -> list[int]:
    """출력(세 로짓 각각)의 입력에 대한 기울기가 0이 아닌 스텝 번호. 배정밀도로 계산한다."""
    generator = torch.Generator().manual_seed(0)
    z = torch.randn(4, steps, N_FEATURES, dtype=torch.float64, generator=generator, requires_grad=True)
    logits = net.double().eval()(z)
    touched = torch.zeros(steps, dtype=torch.bool)
    for class_index in range(logits.shape[1]):
        (grad,) = torch.autograd.grad(logits[:, class_index].sum(), z, retain_graph=True)
        touched |= grad.abs().sum(dim=(0, 2)) > 0.0
    return [int(step) for step in torch.nonzero(touched).flatten()]


@pytest.mark.parametrize("kind", KINDS)
def test_every_one_of_the_120_steps_influences_the_output(kind: str) -> None:
    torch.manual_seed(0)
    net = build_model(kind, N_FEATURES)
    assert _steps_with_gradient(net, WINDOW) == list(range(WINDOW))


def test_tcn_receptive_field_is_125_steps() -> None:
    torch.manual_seed(0)
    # 1 + 2·(3 − 1)·(1 + 2 + 4 + 8 + 16) = 125. 길이 200인 입력에서는 마지막 125스텝(75..199)만 출력에 닿는다.
    assert _steps_with_gradient(CausalTCN(N_FEATURES), 200) == list(range(75, 200))
    # 확장을 하나 줄이면 1 + 2·2·15 = 61이라 창 120스텝 중 앞 59스텝이 출력에 닿지 않는다.
    short = CausalTCN(N_FEATURES, dilations=(1, 2, 4, 8))
    assert _steps_with_gradient(short, WINDOW) == list(range(59, WINDOW))


@pytest.mark.parametrize("kind", KINDS)
def test_every_step_influences_the_exported_probabilities(kind: str, feature_spec: FeatureSpec) -> None:
    torch.manual_seed(0)
    wrapper = ExportWrapper(build_model(kind, N_FEATURES), feature_spec, 1.0).double().eval()
    generator = torch.Generator().manual_seed(1)
    # log1p 열의 기울기가 0이 되지 않게 양수만 쓴다. v0 spec(평균 0, 표준편차 1)에서 ±5 안에 든다.
    raw = torch.rand(1, WINDOW, N_FEATURES, dtype=torch.float64, generator=generator, requires_grad=True)
    probs = wrapper(raw)
    (grad,) = torch.autograd.grad(probs[0, 2], raw)
    assert bool((grad.abs().sum(dim=(0, 2)) > 0.0).all())


# ---------------------------------------------------------------------------
# ExportWrapper: 전처리와 후처리
# ---------------------------------------------------------------------------


def test_log1p_columns_are_exactly_9_10_11_12_14(feature_spec: FeatureSpec) -> None:
    assert [i for i, flag in enumerate(feature_spec.log1p_mask) if flag] == LOG1P_COLUMNS
    assert [FEATURE_NAMES[i] for i in LOG1P_COLUMNS] == [
        "long_closure_count_60",
        "max_closure_ms_60",
        "max_closure_ms_10",
        "yawn_count_60",
        "nod_count_60",
    ]
    wrapper = ExportWrapper(_Probe((0.0, 0.0, 0.0)), feature_spec, 1.0)
    assert torch.nonzero(wrapper.log1p_mask).flatten().tolist() == LOG1P_COLUMNS
    assert torch.equal(wrapper.keep_mask, 1.0 - wrapper.log1p_mask)


def test_preprocess_order_with_hand_computed_values(feature_spec: FeatureSpec) -> None:
    spec = _with_stats(feature_spec, {0: (0.1, 0.05), 9: (0.5, 0.25), 10: (5.0, 1.0)})
    probe = _Probe((0.0, 0.0, 0.0))
    wrapper = ExportWrapper(probe, spec, 1.0).eval()

    raw = torch.zeros(1, 6, N_FEATURES)
    raw[0, 0, 0] = 0.2  # (0.2 − 0.1) / 0.05 = 2
    raw[0, 1, 0] = 0.5  # (0.5 − 0.1) / 0.05 = 8 -> 5로 자름
    raw[0, 2, 0] = -0.4  # (−0.4 − 0.1) / 0.05 = −10 -> −5로 자름
    raw[0, 0, 4] = -3.0  # log1p 열이 아니면 음수를 그대로 둔다: (−3 − 0) / 1 = −3
    raw[0, 0, 9] = math.e - 1.0  # log1p = 1, (1 − 0.5) / 0.25 = 2
    raw[0, 1, 9] = -3.0  # max(x, 0) = 0, log1p = 0, (0 − 0.5) / 0.25 = −2
    raw[0, 0, 10] = math.exp(7.0) - 1.0  # log1p = 7, (7 − 5) / 1 = 2
    raw[0, 1, 10] = 10000.0  # log1p = 9.2104404, (9.2104404 − 5) / 1 = 4.2104404
    raw[0, 2, 10] = 1.0e9  # log1p = 20.72 -> 5로 자름
    with torch.no_grad():
        wrapper(raw)
    z = probe.seen
    assert z is not None and z.shape == raw.shape

    expected = torch.zeros(1, 6, N_FEATURES)
    expected[0, :, 0] = -2.0  # 원시값 0: (0 − 0.1) / 0.05
    expected[0, :, 9] = -2.0  # 원시값 0: (log1p(0) − 0.5) / 0.25
    expected[0, :, 10] = -5.0  # 원시값 0: (0 − 5) / 1
    expected[0, 0, 0] = 2.0
    expected[0, 1, 0] = 5.0
    expected[0, 2, 0] = -5.0
    expected[0, 0, 4] = -3.0
    expected[0, 0, 9] = 2.0
    expected[0, 1, 9] = -2.0
    expected[0, 0, 10] = 2.0
    expected[0, 1, 10] = 4.2104404
    expected[0, 2, 10] = 5.0
    assert torch.allclose(z, expected, rtol=0.0, atol=1e-5)
    # 자르기는 정확히 ±z_clip이다.
    assert z[0, 1, 0].item() == 5.0 and z[0, 2, 0].item() == -5.0


def test_temperature_divides_logits_before_softmax(feature_spec: FeatureSpec) -> None:
    wrapper = ExportWrapper(_Probe((1.0, 2.0, 4.0)), feature_spec, 2.0).eval()
    with torch.no_grad():
        probs = wrapper(torch.zeros(1, WINDOW, N_FEATURES))
    # softmax([0.5, 1, 2]) = [e^0.5, e^1, e^2] / 11.7560592
    expected = torch.tensor([[0.14024438, 0.23122390, 0.62853172]])
    assert probs.shape == (1, 3)
    assert torch.allclose(probs, expected, rtol=0.0, atol=1e-6)
    assert probs.sum().item() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("temperature", [0.0, -1.0, math.inf, math.nan])
def test_wrapper_rejects_non_positive_or_non_finite_temperature(temperature: float, feature_spec: FeatureSpec) -> None:
    with pytest.raises(ValueError, match="temperature"):
        ExportWrapper(_Probe((0.0, 0.0, 0.0)), feature_spec, temperature)


@pytest.mark.parametrize("kind", KINDS)
def test_negative_raw_values_do_not_produce_nan(kind: str, feature_spec: FeatureSpec) -> None:
    torch.manual_seed(0)
    wrapper = ExportWrapper(build_model(kind, N_FEATURES), feature_spec, 1.0).eval()
    raw = torch.full((1, WINDOW, N_FEATURES), -1.0e6)
    with torch.no_grad():
        z = wrapper.preprocess(raw)
        probs = wrapper(raw)
    assert torch.isfinite(z).all()
    # log1p 열은 log1p(max(x, 0)) = 0에서 표준화되고(v0 spec: 0), 나머지는 −z_clip으로 잘린다.
    for column in range(N_FEATURES):
        assert torch.all(z[..., column] == (0.0 if column in LOG1P_COLUMNS else -5.0))
    assert torch.isfinite(probs).all()
    assert probs.sum().item() == pytest.approx(1.0, abs=1e-6)

    # log1p 열의 음수는 0과 같은 결과를 낸다.
    mixed = torch.rand(1, WINDOW, N_FEATURES)
    negative = mixed.clone()
    negative[..., LOG1P_COLUMNS] = -negative[..., LOG1P_COLUMNS] - 0.5
    zeroed = mixed.clone()
    zeroed[..., LOG1P_COLUMNS] = 0.0
    with torch.no_grad():
        assert torch.equal(wrapper(negative), wrapper(zeroed))


@pytest.mark.parametrize("kind", KINDS)
def test_std_zero_feature_still_gives_finite_probabilities(kind: str, feature_spec: FeatureSpec) -> None:
    torch.manual_seed(0)
    # 분산이 0인 특징(예: 하품이 한 번도 없는 학습 데이터의 yawn_count_60). 평균과 같은 값은 0/0이 될 뻔한 경우다.
    spec = _with_stats(feature_spec, {12: (0.0, 0.0), 15: (0.25, 0.0)})
    wrapper = ExportWrapper(build_model(kind, N_FEATURES), spec, 1.0).eval()
    assert wrapper.std[12].item() == pytest.approx(0.001) and wrapper.std[15].item() == pytest.approx(0.001)
    assert bool((wrapper.std >= torch.tensor(0.001)).all())

    raw = torch.rand(3, WINDOW, N_FEATURES)
    raw[0, :, 12] = 0.0
    raw[0, :, 15] = 0.25
    raw[1, :, 12] = 2.0
    raw[1, :, 15] = 1.0
    with torch.no_grad():
        z = wrapper.preprocess(raw)
        probs = wrapper(raw)
    assert torch.isfinite(z).all() and torch.isfinite(probs).all()
    assert torch.all(z[0, :, 12] == 0.0) and torch.all(z[0, :, 15] == 0.0)
    assert torch.all(z[1, :, 12] == 5.0) and torch.all(z[1, :, 15] == 5.0)
    assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-6)


# ---------------------------------------------------------------------------
# 체크포인트
# ---------------------------------------------------------------------------


# 체크포인트에 적는 feature_spec 사본의 해시 자리에 쓰는 값. 형식(sha256, 소문자 16진수 64자리)만 맞으면 된다.
SPEC_HASH = "0123456789abcdef" * 4


@pytest.mark.parametrize(
    ("kind", "hyper", "synthetic"),
    [
        (KIND_TCN, {"channels": 16, "kernel_size": 3, "dilations": (1, 2, 4, 8, 16), "dropout": 0.3}, False),
        (KIND_GRU, {"hidden": 32, "dropout": 0.1}, True),
    ],
)
def test_checkpoint_round_trip(kind: str, hyper: dict[str, object], synthetic: bool, tmp_path: Path) -> None:
    torch.manual_seed(0)
    net = build_model(kind, N_FEATURES, **hyper)
    _randomize_batchnorm(net, seed=2)
    net.train()
    path = tmp_path / "nested" / "model.pt"
    save_checkpoint(path, net, kind, N_FEATURES, hyper, 1.37, "synthetic sessions", SPEC_HASH, synthetic)

    # 숫자, 문자열, 불, 리스트, 텐서만 들어 있어 weights_only로 읽힌다.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert CHECKPOINT_FORMAT == "drowsy-checkpoint-v2" and payload["format"] == CHECKPOINT_FORMAT
    assert set(payload) == {
        "format", "kind", "n_features", "hyper", "state_dict", "temperature", "trained_on", "feature_spec_hash",
        "synthetic", "fast",
    }  # fmt: skip
    assert payload["feature_spec_hash"] == SPEC_HASH and payload["synthetic"] is synthetic
    assert payload["fast"] is False  # fast를 주지 않으면 축소 설정이 아닌 것으로 적는다

    loaded = load_checkpoint(path)
    assert loaded.kind == kind and loaded.n_features == N_FEATURES
    assert loaded.temperature == pytest.approx(1.37) and loaded.trained_on == "synthetic sessions"
    assert loaded.feature_spec_hash == SPEC_HASH and loaded.synthetic is synthetic and loaded.fast is False
    assert loaded.hyper == {key: list(value) if isinstance(value, tuple) else value for key, value in hyper.items()}
    assert loaded.net.training is False
    z = torch.randn(2, WINDOW, N_FEATURES)
    with torch.no_grad():
        assert torch.equal(loaded.net(z), net.eval()(z))


@pytest.mark.parametrize("fast", [True, False])
def test_checkpoint_fast_flag_round_trip(fast: bool, tmp_path: Path) -> None:
    """축소 설정(연기 시험) 표시: 이름을 적어 준 값이 파일에 그대로 들어가고 그대로 읽힌다. synthetic과는 별개의 값이다."""
    net = build_model(KIND_TCN, N_FEATURES, channels=16)
    path = tmp_path / "model.pt"
    save_checkpoint(path, net, KIND_TCN, N_FEATURES, {"channels": 16}, 1.0, "x", SPEC_HASH, not fast, fast=fast)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["fast"] is fast and payload["synthetic"] is (not fast)
    assert payload["format"] == "drowsy-checkpoint-v2"  # 형식 이름은 바뀌지 않는다
    loaded = load_checkpoint(path)
    assert loaded.fast is fast and loaded.synthetic is (not fast)


def test_checkpoint_without_the_fast_key_loads_as_not_fast(tmp_path: Path) -> None:
    """`fast` 키가 생기기 전에 쓴 v2 파일(키가 없다)은 축소 설정이 아닌 것으로 읽는다. 키가 있으면 불이어야 한다."""
    net = build_model(KIND_TCN, N_FEATURES, channels=16)
    old = _payload(net)
    assert "fast" not in old and old["format"] == "drowsy-checkpoint-v2"
    torch.save(old, tmp_path / "old.pt")
    assert load_checkpoint(tmp_path / "old.pt").fast is False

    torch.save(_payload(net, fast=True), tmp_path / "fast.pt")
    assert load_checkpoint(tmp_path / "fast.pt").fast is True
    for index, bad in enumerate(("yes", 1, 0, "false")):
        path = tmp_path / f"bad_{index}.pt"
        torch.save(_payload(net, fast=bad), path)
        with pytest.raises(ValueError, match="checkpoint fast must be true or false"):
            load_checkpoint(path)

    # 쓸 때도 불만 받고, 자리로는 줄 수 없다(예전 호출 `save_checkpoint(..., hash, synthetic)`이 그대로 통한다)
    good = {"channels": 16}
    out = tmp_path / "a.pt"
    bad_flags: list[Any] = [None, 1, "true"]
    for bad_flag in bad_flags:
        with pytest.raises(ValueError, match="fast must be True or False"):
            save_checkpoint(out, net, KIND_TCN, N_FEATURES, good, 1.0, "x", SPEC_HASH, False, fast=bad_flag)
    with pytest.raises(TypeError):
        save_checkpoint(out, net, KIND_TCN, N_FEATURES, good, 1.0, "x", SPEC_HASH, False, True)  # type: ignore[misc]
    assert not out.exists()


def test_checkpoint_errors(tmp_path: Path) -> None:
    net = build_model(KIND_TCN, N_FEATURES, channels=16)
    good = {"channels": 16}
    with pytest.raises(RuntimeError):  # 하이퍼파라미터가 가중치와 맞지 않는다
        save_checkpoint(tmp_path / "a.pt", net, KIND_TCN, N_FEATURES, {"channels": 32}, 1.0, "x", SPEC_HASH, False)
    with pytest.raises(ValueError, match="temperature"):
        save_checkpoint(tmp_path / "a.pt", net, KIND_TCN, N_FEATURES, good, 0.0, "x", SPEC_HASH, False)
    with pytest.raises(ValueError, match="trained_on"):
        save_checkpoint(tmp_path / "a.pt", net, KIND_TCN, N_FEATURES, good, 1.0, "", SPEC_HASH, False)
    # 해시는 sha256 형식이어야 하고(대문자, 짧은 값, 빈 값은 안 된다) synthetic은 불이어야 한다.
    bad_hashes: list[Any] = ["", "abc", SPEC_HASH.upper(), SPEC_HASH + "0", None]
    for bad_hash in bad_hashes:
        with pytest.raises(ValueError, match="feature_spec_hash"):
            save_checkpoint(tmp_path / "a.pt", net, KIND_TCN, N_FEATURES, good, 1.0, "x", bad_hash, False)
    bad_flags: list[Any] = [None, 0, "false"]
    for bad_flag in bad_flags:
        with pytest.raises(ValueError, match="synthetic"):
            save_checkpoint(tmp_path / "a.pt", net, KIND_TCN, N_FEATURES, good, 1.0, "x", SPEC_HASH, bad_flag)
    with pytest.raises(TypeError):  # 두 값은 기본값이 없다. v1 방식의 호출은 실패한다
        save_checkpoint(tmp_path / "a.pt", net, KIND_TCN, N_FEATURES, good, 1.0, "x")  # type: ignore[call-arg]
    assert not (tmp_path / "a.pt").exists()

    torch.save({"state_dict": net.state_dict()}, tmp_path / "other.pt")
    with pytest.raises(ValueError, match="not a drowsy-checkpoint-v2 checkpoint"):
        load_checkpoint(tmp_path / "other.pt")
    torch.save({"format": CHECKPOINT_FORMAT, "kind": KIND_TCN}, tmp_path / "partial.pt")
    with pytest.raises(ValueError, match="missing keys"):
        load_checkpoint(tmp_path / "partial.pt")


def _payload(net: torch.nn.Module, **changes: object) -> dict[str, object]:
    """손으로 만든 v2 체크포인트 내용(TCN, 채널 16). changes로 키를 바꾸고, 값이 None이면 그 키를 뺀다."""
    payload: dict[str, object] = {
        "format": "drowsy-checkpoint-v2",
        "kind": KIND_TCN,
        "n_features": N_FEATURES,
        "hyper": {"channels": 16},
        "state_dict": net.state_dict(),
        "temperature": 1.25,
        "trained_on": "UTA-RLDD 60 subjects, arm=alert_baseline",
        "feature_spec_hash": SPEC_HASH,
        "synthetic": False,
    }
    payload.update(changes)
    return {key: value for key, value in payload.items() if value is not None}


def test_v1_checkpoint_is_rejected_with_instructions(tmp_path: Path) -> None:
    net = build_model(KIND_TCN, N_FEATURES, channels=16)
    # v1 형식 그대로: 해시와 synthetic 키가 없다.
    v1 = _payload(net, format="drowsy-checkpoint-v1", feature_spec_hash=None, synthetic=None)
    assert set(v1) == {"format", "kind", "n_features", "hyper", "state_dict", "temperature", "trained_on"}
    torch.save(v1, tmp_path / "v1.pt")
    with pytest.raises(ValueError) as caught:
        load_checkpoint(tmp_path / "v1.pt")
    message = str(caught.value)
    assert "drowsy-checkpoint-v1" in message and "drowsy-checkpoint-v2" in message
    assert "re-run `python -m drowsy.train final`" in message
    # 해시와 synthetic을 손으로 덧붙여도 형식 이름이 v1이면 받지 않는다.
    torch.save(_payload(net, format="drowsy-checkpoint-v1"), tmp_path / "v1_patched.pt")
    with pytest.raises(ValueError, match="re-run `python -m drowsy.train final`"):
        load_checkpoint(tmp_path / "v1_patched.pt")


def test_load_checkpoint_rejects_incomplete_or_unreadable_files(tmp_path: Path) -> None:
    net = build_model(KIND_TCN, N_FEATURES, channels=16)
    torch.save(_payload(net), tmp_path / "ok.pt")
    loaded = load_checkpoint(tmp_path / "ok.pt")  # 손으로 만든 기준 내용 자체는 읽힌다
    assert loaded.feature_spec_hash == SPEC_HASH and loaded.synthetic is False and loaded.temperature == 1.25

    cases: list[tuple[dict[str, object], str]] = [
        ({"feature_spec_hash": None}, r"missing keys \['feature_spec_hash'\]"),
        ({"synthetic": None}, r"missing keys \['synthetic'\]"),
        ({"feature_spec_hash": "not-a-hash"}, "feature_spec_hash must be 64 lowercase hex digits"),
        ({"feature_spec_hash": 7}, "feature_spec_hash must be 64 lowercase hex digits"),
        ({"synthetic": "no"}, "synthetic must be true or false"),
        ({"synthetic": 1}, "synthetic must be true or false"),
        ({"hyper": {"channels": 32}}, "weights do not fit"),
    ]
    for index, (changes, message) in enumerate(cases):
        path = tmp_path / f"case_{index}.pt"
        torch.save(_payload(net, **changes), path)
        with pytest.raises(ValueError, match=message):
            load_checkpoint(path)

    garbage = tmp_path / "garbage.pt"
    garbage.write_bytes(b"this is not a checkpoint\x00\x01" * 20)
    with pytest.raises(ValueError, match="not a readable drowsy-checkpoint-v2 checkpoint"):
        load_checkpoint(garbage)
    (tmp_path / "empty.pt").write_bytes(b"")
    with pytest.raises(ValueError, match="not a readable drowsy-checkpoint-v2 checkpoint"):
        load_checkpoint(tmp_path / "empty.pt")
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt")
