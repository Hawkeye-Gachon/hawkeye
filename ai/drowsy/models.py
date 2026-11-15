"""졸음 판정 신경망과 내보내기용 래퍼 (PLAN 5.1절). 이식 대상이 아니다.

- `CausalTCN`, `GRUClassifier`: 표준화된 스텝 특징 [B, T, F]를 받아 로짓 [B, 3]을 낸다. T는 120이다.
- `ExportWrapper`: 원시 스텝 특징을 받아 확률을 낸다. ONNX로 나가는 것은 이 모듈이다.
  처리 순서는 두 모델이 같다: log1p(max(x, 0))(일부 열) → (x − mean) / std → ±z_clip 자르기 → 신경망
  → 온도로 나누기 → 소프트맥스.
- 클래스 순서는 `drowsy.types.CLASS_ORDER`(alert, low_vigilant, drowsy)다.

체크포인트 형식 v2 (학습 코드가 쓰고 `python -m drowsy.export_onnx --checkpoint`가 읽는다)
    `torch.save`로 쓴 dict 하나. `save_checkpoint`로 쓰고 `load_checkpoint`로 읽는다.
    {
      "format": "drowsy-checkpoint-v2",
      "kind": "tcn" | "gru",
      "n_features": 17,
      "hyper": {...},            # build_model(kind, n_features, **hyper)에 그대로 넘기는 값(숫자와 리스트만)
      "state_dict": {...},       # 래퍼가 아니라 신경망(CausalTCN 또는 GRUClassifier)의 state_dict
      "temperature": 1.0,        # 내부 검증 예측으로 맞춘 온도. feature_spec 사본의 standardization.temperature와 같아야 한다
      "trained_on": "UTA-RLDD 60 subjects, arm=alert_baseline",   # ONNX 메타데이터 trained_on에 그대로 들어간다
      "feature_spec_hash": "<64자리 16진수>",   # 학습이 쓴 feature_spec 사본(feature_spec_v{N}.json) 바이트의 sha256
      "synthetic": false,        # 합성 데이터(drowsy.synth)로 학습했으면 true
      "fast": false,             # 축소 설정(`--fast`, 연기 시험)으로 학습했으면 true. 이 키가 없는 파일은 false로 읽는다
    }
    평균과 표준편차는 체크포인트가 아니라 feature_spec 사본에 둔다(PLAN 4.7절). 대신 체크포인트가 그 사본의 해시를
    갖고 있어, 내보낼 때 다른 spec(예: 평균 0·표준편차 1인 config/feature_spec.json)을 주면 거부한다.
    `synthetic`이나 `fast`가 true인 체크포인트는 `paths.export_dir`와 그 아래로 내보내지 못한다(`drowsy.export_onnx`).
    `fast`는 v2에 나중에 더한 키다. 형식 이름은 그대로이고, 이 키가 없는 v2 파일도 읽는다(`load_checkpoint`의 가정).
    v1 형식("drowsy-checkpoint-v1": 위의 마지막 두 키가 없다)은 읽지 않는다. `python -m drowsy.train final`을
    다시 실행해 v2 체크포인트를 만든다.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from drowsy.spec import FeatureSpec
from drowsy.types import CLASS_ORDER

N_CLASSES = len(CLASS_ORDER)

KIND_TCN = "tcn"
KIND_GRU = "gru"
KINDS: tuple[str, ...] = (KIND_TCN, KIND_GRU)

CHECKPOINT_FORMAT = "drowsy-checkpoint-v2"
CHECKPOINT_FORMAT_V1 = "drowsy-checkpoint-v1"  # 더 읽지 않는 옛 형식. 알아보고 안내하는 데만 쓴다

_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


class CausalConv1d(nn.Module):
    """왼쪽 0 채움 → Conv1d → BatchNorm1d. 입력과 출력은 [B, C, T]이고 길이가 같다.

    왼쪽에만 (kernel_size − 1) · dilation개를 채우므로 출력 t는 입력 t − (kernel_size − 1) · dilation .. t만 본다.
    배치 정규화를 합성곱 바로 뒤에 두어야 내보낼 때 합성곱에 합쳐진다(PLAN 5.1절).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        # F.pad 대신 0 텐서를 이어 붙인다. TorchScript 내보내기는 F.pad를 Pad 하나가 아니라 상수 계산 노드
        # 여러 개(Cast, ConstantOfShape, Reshape, Slice 등)로 풀어 쓰고, 이어 붙이기는 Concat 하나가 된다.
        zeros = x.new_zeros((x.shape[0], x.shape[1], self.left_pad))
        return self.bn(self.conv(torch.cat((zeros, x), dim=2)))


class ResidualBlock(nn.Module):
    """인과 합성곱 두 개(각각 배치 정규화, ReLU, 드롭아웃)와 잔차 연결. 입력과 출력은 [B, C, T]."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # 채널 수가 달라지는 블록(첫 블록)만 1×1 합성곱으로 잔차 경로의 채널을 맞춘다.
        self.shortcut: nn.Module = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.dropout(self.relu(self.conv1(x)))
        y = self.dropout(self.relu(self.conv2(y)))
        return self.relu(y + self.shortcut(x))


class CausalTCN(nn.Module):
    """인과 TCN (PLAN 5.1절).

    잔차 블록이 확장(dilation)마다 하나씩 있고 마지막 스텝의 활성값만 읽어 선형 층에 넣는다.
    수용 영역은 1 + 2 · (kernel_size − 1) · sum(dilations) 스텝이다. 기본값이면 125로 창 120스텝을 모두 덮는다.
    합성곱은 모두 왼쪽에만 채우므로 스텝 t의 활성값은 t 이후의 입력을 보지 않는다. 다만 학습 모드의 배치 정규화는
    시간축 전체의 통계를 쓰므로, 이 성질이 정확히 성립하는 것은 eval 모드(내보내는 그래프)에서다.
    """

    def __init__(
        self,
        n_features: int,
        channels: int = 32,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4, 8, 16),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = n_features
        for dilation in dilations:
            blocks.append(ResidualBlock(in_channels, channels, kernel_size, dilation, dropout))
            in_channels = channels
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, N_CLASSES)

    def forward(self, z: Tensor) -> Tensor:
        """표준화된 [B, T, F] -> 로짓 [B, 3]. z[:, 0]이 가장 오래된 스텝이다."""
        h = self.blocks(z.transpose(1, 2))
        return self.head(h[:, :, -1])


class GRUClassifier(nn.Module):
    """단방향 GRU 1층, 마지막 은닉 상태, 드롭아웃, 선형 3 (PLAN 5.1절)."""

    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, N_CLASSES)

    def forward(self, z: Tensor) -> Tensor:
        """표준화된 [B, T, F] -> 로짓 [B, 3]. z[:, 0]이 가장 오래된 스텝이다."""
        _, h_n = self.gru(z)
        return self.head(self.dropout(h_n[-1]))


class ExportWrapper(nn.Module):
    """원시 스텝 특징 [B, T, F] -> 확률 [B, 3]. ONNX 그래프에 들어가는 전처리와 후처리를 모두 포함한다 (계획서 1.1절 질문 Q6).

    내보낼 때는 B = 1로 고정한다. 입력은 유한값이어야 한다(PLAN 2.5절: 무한대는 마스크 곱에서 NaN이 된다).

    - log1p: ONNX opset 17에는 Log1p 연산자가 없어 내보내기가 Log(Add(x, 1))을 쓴다. PyTorch 쪽도 같은 식
      `log(relu(x) + 1)`로 계산해 두 실행기가 같은 float32 연산을 하게 한다.
    - 마스크: `x · keep + log(relu(x) + 1) · mask`. mask는 transform이 log1p인 열만 1인 상수다.
    - 표준편차는 spec의 `std_floor` 아래로 내려가지 않게 여기서 한 번 더 자른다.
    가정: 온도는 인자로 받은 값을 쓴다. spec의 `standardization.temperature`와 같은지는 내보낼 때
    `drowsy.export_onnx.export`가 검사한다.
    """

    mean: Tensor
    std: Tensor
    log1p_mask: Tensor
    keep_mask: Tensor
    temperature: Tensor

    def __init__(self, net: nn.Module, spec: FeatureSpec, temperature: float) -> None:
        super().__init__()
        if not (math.isfinite(temperature) and temperature > 0.0):
            raise ValueError(f"temperature must be a finite positive number, got {temperature!r}")
        std_floor = spec.standardization.std_floor
        mask = torch.tensor([1.0 if flag else 0.0 for flag in spec.log1p_mask], dtype=torch.float32)
        self.net = net
        self.register_buffer("log1p_mask", mask)
        self.register_buffer("keep_mask", 1.0 - mask)
        self.register_buffer("mean", torch.tensor(spec.means, dtype=torch.float32))
        self.register_buffer("std", torch.tensor([max(s, std_floor) for s in spec.stds], dtype=torch.float32))
        self.register_buffer("temperature", torch.tensor(temperature, dtype=torch.float32))
        self.z_clip = float(spec.standardization.z_clip)

    def preprocess(self, features: Tensor) -> Tensor:
        """원시 [B, T, F] -> 신경망 입력 z [B, T, F] (log1p, 표준화, ±z_clip 자르기)."""
        x = features * self.keep_mask + torch.log(torch.relu(features) + 1.0) * self.log1p_mask
        z = (x - self.mean) / self.std
        return torch.clamp(z, -self.z_clip, self.z_clip)

    def forward(self, features: Tensor) -> Tensor:
        logits = self.net(self.preprocess(features))
        return torch.softmax(logits / self.temperature, dim=-1)


def build_model(kind: str, n_features: int, **hyper: Any) -> nn.Module:
    """`kind`("tcn" 또는 "gru")와 하이퍼파라미터로 신경망을 만든다. hyper는 해당 클래스 생성자의 인자다."""
    if kind == KIND_TCN:
        return CausalTCN(n_features, **hyper)
    if kind == KIND_GRU:
        return GRUClassifier(n_features, **hyper)
    raise ValueError(f"unknown model kind {kind!r}, expected one of {KINDS}")


def count_parameters(m: nn.Module) -> int:
    """파라미터 수(가중치, 편향, 배치 정규화의 scale과 shift). 이동 평균 같은 버퍼는 세지 않는다."""
    return sum(p.numel() for p in m.parameters())


@dataclass(frozen=True)
class Checkpoint:
    """`load_checkpoint`의 결과. `net`은 가중치를 올린 신경망이고 eval 모드다."""

    net: nn.Module
    kind: str
    n_features: int
    hyper: dict[str, Any]
    temperature: float
    trained_on: str
    feature_spec_hash: str  # 학습이 쓴 feature_spec 사본 바이트의 sha256
    synthetic: bool  # 합성 데이터로 학습했는지
    fast: bool = False  # 축소 설정(연기 시험)으로 학습했는지. 키가 없는 파일은 False


def _plain_hyper(hyper: Mapping[str, Any]) -> dict[str, Any]:
    # torch.load(weights_only=True)로 읽을 수 있게 튜플을 리스트로 바꾼다.
    return {str(key): list(value) if isinstance(value, (list, tuple)) else value for key, value in hyper.items()}


def save_checkpoint(
    path: str | Path,
    net: nn.Module,
    kind: str,
    n_features: int,
    hyper: Mapping[str, Any],
    temperature: float,
    trained_on: str,
    feature_spec_hash: str,
    synthetic: bool,
    *,
    fast: bool = False,
) -> None:
    """모듈 설명의 형식(v2)으로 체크포인트를 쓴다.

    `net`은 `build_model(kind, n_features, **hyper)`로 만든 신경망이어야 한다. 쓰기 전에 같은 인자로
    신경망을 다시 만들어 state_dict가 맞는지 확인하므로, 하이퍼파라미터가 틀리면 여기서 예외가 난다.
    feature_spec_hash는 이 모델의 평균·표준편차·온도를 담은 feature_spec 사본(`feature_spec_v{N}.json`,
    `drowsy.spec.write_feature_spec`이 쓴 파일) 바이트의 sha256이고, synthetic은 합성 데이터로 학습했는지다.
    두 값은 `python -m drowsy.export_onnx --checkpoint`가 spec을 대조하고 합성 모델의 내보내기 위치를 막는 데 쓴다.
    fast는 축소 설정(`python -m drowsy.train final --fast`, 사전에 고정한 절차가 아닌 연기 시험)으로 학습했는지다.
    이름을 적어서만 줄 수 있고 기본값은 False다. true이면 같은 명령이 `paths.export_dir`와 그 아래로 내보내지 않는다.
    """
    if not (math.isfinite(temperature) and temperature > 0.0):
        raise ValueError(f"temperature must be a finite positive number, got {temperature!r}")
    if not trained_on:
        raise ValueError("trained_on must not be empty")
    if not (isinstance(feature_spec_hash, str) and _SHA256_HEX.fullmatch(feature_spec_hash)):
        raise ValueError(f"feature_spec_hash must be 64 lowercase hex digits (sha256), got {feature_spec_hash!r}")
    if not isinstance(synthetic, bool):
        raise ValueError(f"synthetic must be True or False, got {synthetic!r}")
    if not isinstance(fast, bool):
        raise ValueError(f"fast must be True or False, got {fast!r}")
    plain = _plain_hyper(hyper)
    state_dict = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
    build_model(kind, n_features, **plain).load_state_dict(state_dict)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "kind": kind,
        "n_features": int(n_features),
        "hyper": plain,
        "state_dict": state_dict,
        "temperature": float(temperature),
        "trained_on": trained_on,
        "feature_spec_hash": feature_spec_hash,
        "synthetic": synthetic,
        "fast": fast,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)


def load_checkpoint(path: str | Path) -> Checkpoint:
    """`save_checkpoint`가 쓴 파일(v2)을 읽어 신경망을 만들고 가중치를 올린다(CPU, eval 모드).

    읽을 수 없는 파일, 다른 형식, 빠진 키, 틀린 값은 ValueError다(파일이 없으면 OSError). v1 체크포인트는
    받지 않고, `python -m drowsy.train final`을 다시 실행하라는 안내와 함께 ValueError를 낸다.
    가정: v1 체크포인트는 어떤 feature_spec으로 학습했는지와 합성 데이터 여부를 담고 있지 않아 확인할 방법이 없으므로,
    경고만 하고 읽는 대신 거부한다.
    가정: `fast` 키가 없는 v2 체크포인트(이 키를 더하기 전에 쓴 파일)는 축소 설정이 아닌 것으로 읽는다(`fast=False`).
    그 파일이 실제로 축소 설정으로 학습한 것이었는지는 파일만으로 알 수 없다. 키가 있으면 불이어야 한다.
    """
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except OSError:
        raise
    except Exception as error:  # torch.load는 깨진 파일에 UnpicklingError, EOFError, IndexError 등을 낸다
        raise ValueError(f"{path}: not a readable {CHECKPOINT_FORMAT} checkpoint ({type(error).__name__})") from error
    if isinstance(payload, dict) and payload.get("format") == CHECKPOINT_FORMAT_V1:
        raise ValueError(
            f"{path}: this is a {CHECKPOINT_FORMAT_V1} checkpoint, which records neither the feature spec it was "
            f"trained with nor whether the data was synthetic; re-run `python -m drowsy.train final` to write a "
            f"{CHECKPOINT_FORMAT} checkpoint"
        )
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: not a {CHECKPOINT_FORMAT} checkpoint")
    required = (
        "kind", "n_features", "hyper", "state_dict", "temperature", "trained_on", "feature_spec_hash", "synthetic",
    )  # fmt: skip
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{path}: checkpoint is missing keys {missing}")
    spec_hash = payload["feature_spec_hash"]
    if not (isinstance(spec_hash, str) and _SHA256_HEX.fullmatch(spec_hash)):
        raise ValueError(f"{path}: checkpoint feature_spec_hash must be 64 lowercase hex digits, got {spec_hash!r}")
    if not isinstance(payload["synthetic"], bool):
        raise ValueError(f"{path}: checkpoint synthetic must be true or false, got {payload['synthetic']!r}")
    fast = payload.get("fast", False)
    if not isinstance(fast, bool):
        raise ValueError(f"{path}: checkpoint fast must be true or false, got {fast!r}")
    try:
        net = build_model(payload["kind"], payload["n_features"], **payload["hyper"])
        net.load_state_dict(payload["state_dict"])
    except (RuntimeError, TypeError) as error:  # 하이퍼파라미터와 가중치가 맞지 않는 파일
        raise ValueError(f"{path}: checkpoint weights do not fit its kind and hyper-parameters ({error})") from error
    net.eval()
    return Checkpoint(
        net=net,
        kind=payload["kind"],
        n_features=payload["n_features"],
        hyper=dict(payload["hyper"]),
        temperature=float(payload["temperature"]),
        trained_on=payload["trained_on"],
        feature_spec_hash=spec_hash,
        synthetic=payload["synthetic"],
        fast=fast,
    )
