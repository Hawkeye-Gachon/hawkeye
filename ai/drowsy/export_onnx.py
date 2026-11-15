"""ONNX 내보내기, 계약 검사, 일치 검증, 지연 측정, INT8 양자화 (PLAN 7장). 이식 대상이 아니다.

계약
- 파일: `export/drowsiness_v{N}.onnx`. opset 17, IR 8, 단일 파일, 1 MB 미만.
- 입력 `features` float32 [1, 120, F]: 표준화하지 않은 원시 스텝 특징. `features[0][0]`이 가장 오래된 스텝이다.
- 출력 `probs` float32 [1, 3]: alert, low_vigilant, drowsy 순서의 확률.
- 메타데이터(`metadata_props`, 값은 모두 문자열) 열 개: `METADATA_KEYS`.
- 모델과 함께 나가는 feature_spec 사본은 버전마다 따로다: `drowsiness_v{N}.onnx` 옆의 `feature_spec_v{N}.json`
  (`exported_spec_name`). `feature_spec_hash`는 그 파일 바이트의 sha256이다. 재학습으로 평균·표준편차·온도가
  바뀐 모델을 같은 폴더에 내보내도 앞 버전의 사본과 해시는 그대로 남는다. INT8 파일 `drowsiness_v{N}_int8.onnx`는
  같은 버전의 사본을 같이 쓴다.
- 합성 데이터로 학습한 모델은 `paths.export_dir`와 그 아래 폴더로 내보내지 않는다(`refuse_synthetic_export`).
  학습 쪽(`drowsy.train`)과 이 모듈의 명령이 같은 함수와 같은 표지(`SYNTHETIC_TRAINED_ON_PREFIX`)를 쓴다.
- 축소 설정(`train final --fast`, 연기 시험)으로 학습한 모델도 같은 폴더로 내보내지 않는다(`refuse_fast_export`).
  학습 쪽은 실행 옵션으로, 이 모듈의 명령은 체크포인트의 `fast` 값으로 판단한다.
- 두 경우 모두 "그 폴더인지"는 `is_export_dir_or_below` 하나가 판단한다. 상대 경로로 적힌 `paths.export_dir`은 명령을
  실행한 위치 기준의 폴더와 저장소 루트 기준의 폴더 둘 다를 가리키는 것으로 본다. 설정이 다른 폴더를 가리켜도(연기 시험용
  설정이 임시 폴더를 적어 둔 경우) 이 저장소의 `export/`는 늘 함께 지킨다.

내보내기 경로
- 기본: TorchScript 방식(`dynamo=False`). opset 17, IR 8, 단일 파일을 그대로 쓴다.
- 대체: dynamo 방식(`export(..., dynamo=True)`, 명령줄 `--dynamo`). IR 10을 쓰고 opset 18로 남는 경우가 있어
  `finalize`가 고친다. 두 경로 모두 `finalize`와 `verify_contract`를 거친다. 허용 연산자 목록은 경로마다 따로다.

명령 (저장소 루트에서 실행)
    python -m drowsy.export_onnx --random-init --kind tcn --version 0
        무작위 가중치(시드 고정)로 export/drowsiness_v0.onnx와 export/feature_spec_v0.json을 쓴다.
    python -m drowsy.export_onnx --checkpoint <체크포인트> --spec <폴더>/feature_spec_v1.json --version 1 --out-dir <폴더>
        학습한 모델을 내보낸다. 체크포인트 형식(v2)은 `drowsy.models`의 모듈 설명에 있다
        (`drowsy.models.save_checkpoint`로 쓴다). `--spec`과 `--version`은 반드시 준다. `--spec`은 그 학습이 쓴
        feature_spec 사본(`python -m drowsy.train final`이 모델 옆에 쓴 `feature_spec_v{N}.json`)이다. 평균·표준편차·온도를
        거기서 읽고, 그 내용의 sha256이 체크포인트에 적힌 값과 다르면 내보내지 않는다(다른 spec으로는 표준화 상수가
        틀린 모델이 계약 검사와 일치 검증을 모두 통과하기 때문이다). 합성 데이터로 학습한 체크포인트와 축소 설정
        (`--fast`)으로 학습한 체크포인트는 `paths.export_dir`와 이 저장소의 `export/`, 그리고 그 아래로 내보내지 않는다
        (`--out-dir`로 다른 폴더를 준다). TCN이면 INT8 파일 `drowsiness_v{N}_int8.onnx`도 만든다(계획서의 결정 D16.
        `--dynamo`일 때는 만들지 않는다).
    두 명령 모두 계약 검사, 일치 검증, 지연 측정을 하고 결과를 출력한다.
    종료 코드: 0 정상, 1 일치 검증이 허용 오차를 넘음, 2 오류(표준 오류에 `error: ...` 한 줄).
"""

from __future__ import annotations

import argparse
import copy
import io
import os
import subprocess
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic

from drowsy.config import ExportSettings, load_config
from drowsy.models import KIND_TCN, KINDS, ExportWrapper, build_model, count_parameters, load_checkpoint
from drowsy.spec import (
    STANDARDIZATION_IN_GRAPH,
    FeatureSpec,
    feature_spec_from_dict,
    file_sha256,
    load_feature_spec,
    read_feature_spec_dict,
    write_feature_spec,
)

OPSET = 17
IR_VERSION = 8
INPUT_NAME = "features"
OUTPUT_NAME = "probs"
MAX_FILE_BYTES = 1_000_000  # "1 MB 미만"
MODEL_FILE_PATTERN = "drowsiness_v{version}.onnx"
MODEL_INT8_FILE_PATTERN = "drowsiness_v{version}_int8.onnx"
EXPORTED_SPEC_PATTERN = "feature_spec_v{version}.json"  # 모델 옆에 쓰는 feature_spec 사본의 이름
TRAINED_ON_NONE = "none (random weights)"
# 합성 데이터로 학습한 모델의 `trained_on`은 이 말로 시작한다. `drowsy.train.trained_on_text`가 이 상수를 쓴다.
SYNTHETIC_TRAINED_ON_PREFIX = "SYNTHETIC"

# 계약의 일곱 개와 계획서 1.1절 질문 Q6에서 더한 세 개(standardization, pipeline_hash, mediapipe_model_sha256). 이 순서로 쓴다.
METADATA_KEYS: tuple[str, ...] = (
    "model_version",
    "feature_spec_hash",
    "step_sec",
    "window_steps",
    "class_order",
    "trained_on",
    "commit",
    "standardization",
    "pipeline_hash",
    "mediapipe_model_sha256",
)

# 허용 연산자(모두 기본 도메인 ai.onnx). 기본 경로(TorchScript)로 두 모델(F = 17, 기본 하이퍼파라미터)을
# 내보내 얻은 집합의 합이다. 배치 정규화는 합성곱에 합쳐져 나오지 않는다.
#   TCN: Add, Clip, Concat, Constant, Conv, Div, Gather, Gemm, Log, Mul, Relu, Softmax, Sub, Transpose
#   GRU: Add, Clip, Concat, Constant, Div, Expand, GRU, Gather, Gemm, Log, Mul, Relu, Shape, Softmax, Sub,
#        Transpose, Unsqueeze
ALLOWED_OPS: frozenset[str] = frozenset(
    {
        "Add",
        "Clip",
        "Concat",
        "Constant",
        "Conv",
        "Div",
        "Expand",
        "GRU",
        "Gather",
        "Gemm",
        "Log",
        "Mul",
        "Relu",
        "Shape",
        "Softmax",
        "Sub",
        "Transpose",
        "Unsqueeze",
    }
)

# INT8 파일의 허용 연산자. 기본 경로의 TCN을 `quantize_int8`로 양자화해 얻은 집합이다.
# Conv는 ConvInteger로, Gemm은 MatMulInteger + Add로 바뀐다. GRU는 양자화하지 않는다(계획서의 결정 D16).
ALLOWED_OPS_INT8: frozenset[str] = frozenset(
    {
        "Add",
        "Cast",
        "Clip",
        "Concat",
        "Constant",
        "ConvInteger",
        "Div",
        "DynamicQuantizeLinear",
        "Gather",
        "Log",
        "MatMulInteger",
        "Mul",
        "Relu",
        "Reshape",
        "Softmax",
        "Sub",
        "Transpose",
    }
)

# 대체 경로(dynamo)의 허용 연산자. 같은 두 모델을 `dynamo=True`로 내보내 얻은 집합의 합이다.
#   TCN: Add, Clip, Concat, ConstantOfShape, Conv, Div, Gather, Gemm, Log, Mul, Relu, Softmax, Sub, Transpose
#   GRU: Add, Clip, Concat, Div, GRU, Gather, Gemm, Log, Mul, Relu, Slice, Softmax, Sub, Transpose, Unsqueeze
ALLOWED_OPS_DYNAMO: frozenset[str] = frozenset(
    {
        "Add",
        "Clip",
        "Concat",
        "ConstantOfShape",
        "Conv",
        "Div",
        "GRU",
        "Gather",
        "Gemm",
        "Log",
        "Mul",
        "Relu",
        "Slice",
        "Softmax",
        "Sub",
        "Transpose",
        "Unsqueeze",
    }
)

# `ExportWrapper`의 버퍼 이름. 두 내보내기 경로 모두 이 이름을 그래프 상수(initializer) 이름으로 쓴다.
_MEAN = "mean"
_STD = "std"
_LOG1P_MASK = "log1p_mask"
_KEEP_MASK = "keep_mask"
_TEMPERATURE = "temperature"

# `random_raw_inputs`에서 log1p 열을 되돌릴 때의 지수 상한. exp(30)은 float32 범위 안이다.
_MAX_LOG_VALUE = 30.0
# 로짓의 크기를 바꾸는 연산자. 온도로 나누는 Div 말고는 로짓을 내는 노드가 이 연산자면 안 된다(`_check_graph_constants`).
_SCALING = ("Div", "Mul")
# 로짓을 내는 노드의 검사(`_check_logits_producer`)가 실패할 때 메시지 끝에 붙이는 말: 이 검사가 무엇을 보고 무엇을 못 보는지.
_LOGITS_SCOPE = (
    "this check pins only the node that makes the logits (so that no extra node can rescale them); a last layer whose "
    "weights and bias were rescaled is not detected here: the parity check against the checkpoint "
    "(parity_max_abs_diff) is the guard for the weights"
)


class ContractError(ValueError):
    """내보낸 파일이 계약(PLAN 7장)과 맞지 않을 때."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _is_version(text: str) -> bool:
    """`model_version` 값의 형식: ASCII 숫자만으로 된 음이 아닌 정수."""
    return text.isascii() and text.isdigit()


def _version_text(model_version: int | str) -> str:
    text = str(model_version)
    if not _is_version(text):
        raise ValueError(f"model_version must be a non-negative integer, got {model_version!r}")
    return text


def exported_spec_name(model_version: int | str) -> str:
    """모델 버전 N과 함께 나가는 feature_spec 사본의 파일 이름 `feature_spec_v{N}.json`.

    N은 메타데이터 `model_version`의 값(음이 아닌 정수)이다. 형식이 틀리면 ValueError다.
    """
    return EXPORTED_SPEC_PATTERN.format(version=_version_text(model_version))


def model_file_name(model_version: int | str, int8: bool = False) -> str:
    """모델 버전 N의 파일 이름: `drowsiness_v{N}.onnx`, INT8 파일은 `drowsiness_v{N}_int8.onnx`.

    형식이 틀리면 ValueError다.
    """
    pattern = MODEL_INT8_FILE_PATTERN if int8 else MODEL_FILE_PATTERN
    return pattern.format(version=_version_text(model_version))


def exported_spec_path(onnx_path: str | Path) -> Path:
    """내보낸 모델 파일과 함께 나간 feature_spec 사본의 경로: 같은 폴더의 `feature_spec_v{model_version}.json`.

    버전은 파일 이름이 아니라 모델의 메타데이터 `model_version`에서 읽는다(INT8 파일도 같은 사본을 가리킨다).
    메타데이터에 `model_version`이 없거나 형식이 틀리면, 또는 ONNX로 해석할 수 없는 파일이면 ContractError다.
    사본이 실제로 있는지는 보지 않는다.
    """
    version = read_metadata(onnx_path).get("model_version", "")
    if not _is_version(version):
        raise ContractError(f"{onnx_path}: metadata model_version must be a non-negative integer string")
    return Path(onnx_path).parent / exported_spec_name(version)


def verify_file_name(onnx_path: str | Path) -> None:
    """파일 이름이 계약의 이름인지: `drowsiness_v{model_version}.onnx` 또는 `drowsiness_v{model_version}_int8.onnx`.

    버전은 메타데이터 `model_version`에서 읽는다. 다르면 ContractError다. `verify_contract`는 파일 이름을 보지 않는다
    (임시 이름으로 쓴 파일도 검사하기 때문이다). 폴더에 놓인 배포 파일은 이 함수로 따로 확인한다.
    """
    onnx_path = Path(onnx_path)
    version = read_metadata(onnx_path).get("model_version", "")
    _require(_is_version(version), f"{onnx_path}: metadata model_version must be a non-negative integer string")
    allowed = (model_file_name(version), model_file_name(version, int8=True))
    _require(
        onnx_path.name in allowed,
        f"{onnx_path}: file name must be {allowed[0]!r} or {allowed[1]!r} for metadata model_version {version!r}",
    )


def is_synthetic_trained_on(trained_on: str) -> bool:
    """`trained_on` 문자열이 합성 데이터 표지(`SYNTHETIC_TRAINED_ON_PREFIX`)로 시작하는지."""
    return trained_on.startswith(SYNTHETIC_TRAINED_ON_PREFIX)


def refuse_synthetic_export(out_dir: str | Path, export_dir: str | Path) -> None:
    """합성 데이터로 학습한 모델을 out_dir에 써도 되는지. 내보내기 폴더이거나 그 아래 폴더면 ValueError다.

    export_dir은 설정의 `paths.export_dir`(앱 팀에 넘기고 저장소에 커밋하는 폴더)다. 내보내기 폴더는 그 폴더와, 설정이
    무엇을 가리키든 이 저장소의 `export/`다. 경로는 `protected_export_dir`로 견준다(상대 경로, `..`, 심볼릭 링크, 저장소
    루트가 아닌 실행 위치를 모두 다룬다). 합성 모델인지는 부르는 쪽이 판단한다(`drowsy.train.fit_final`은 데이터 종류로,
    이 모듈의 명령은 체크포인트의 `synthetic` 값과 `trained_on`으로).
    가정: "아래 폴더"는 풀어 쓴 경로의 상위 폴더에 export_dir이 있는 경우다. 이름만 비슷한 이웃 폴더(`export2`)나
    다른 곳의 같은 이름 폴더(`out/export`)는 막지 않는다.
    """
    protected = protected_export_dir(out_dir, export_dir)
    if protected is not None:
        raise ValueError(
            f"a model trained on synthetic data must not be exported to {protected} "
            f"or a folder below it, got {out_dir}; pass another output folder"
        )


def refuse_fast_export(out_dir: str | Path, export_dir: str | Path) -> None:
    """축소 설정(fast, 연기 시험)으로 학습한 모델을 out_dir에 써도 되는지. 내보내기 폴더이거나 그 아래면 ValueError다.

    내보내기 폴더, 경로를 견주는 방법, "아래 폴더"의 뜻은 `refuse_synthetic_export`와 같다. `drowsy.train.fit_final`도
    `--fast` 실행을 이 함수로 막는다. 축소 설정 모델인지는 부르는 쪽이 판단한다(학습은 실행 옵션으로, 이 모듈의 명령은
    체크포인트의 `fast` 값으로).
    """
    protected = protected_export_dir(out_dir, export_dir)
    if protected is not None:
        raise ValueError(
            "a model trained with the fast profile (smoke run) must not be exported to "
            f"{protected} or a folder below it, got {out_dir}; pass another output folder"
        )


# 이 패키지가 들어 있는 저장소의 루트(`drowsy` 폴더의 부모). 상대 경로로 적힌 `paths.export_dir`을 여기에도 대 본다.
REPO_ROOT = Path(__file__).resolve().parents[1]
# 저장소에서 앱 팀에 넘기고 커밋하는 폴더의 이름(`config/default.yaml`의 `paths.export_dir` 기본값과 같다). `REPO_ROOT` 아래의
# 이 폴더는 설정의 `paths.export_dir`이 다른 폴더를 가리킬 때도 내보내기 폴더로 지킨다(`protected_export_dir`).
SHIPPED_EXPORT_DIRNAME = "export"


def protected_export_dir(out_dir: str | Path, export_dir: str | Path) -> str | None:
    """out_dir이 내보내기 폴더이거나 그 아래 폴더면 그 폴더를 가리키는 말(오류 메시지에 넣는다), 아니면 None.

    합성 모델과 축소 설정 모델을 막는 검사(`refuse_synthetic_export`, `refuse_fast_export`)가 이 함수 하나로 판단한다.
    `drowsy.train.fit_final`도 그 두 함수를 부른다.
    - out_dir은 파일을 실제로 쓰게 될 곳이다. 상대 경로면 명령을 실행한 위치(현재 작업 폴더) 기준으로 푼다.
    - 설정이 가리키는 폴더(export_dir = `paths.export_dir`): 절대 경로면 그 폴더 하나다. 상대 경로(설정의 기본값
      `export`)면 두 폴더를 모두 본다: 실행 위치 기준으로 푼 것과 저장소 루트(`REPO_ROOT`) 기준으로 푼 것. 그래서 저장소
      루트가 아닌 곳에서 명령을 실행하면서 `--out-dir`로 저장소의 `export/`를 절대 경로나 `../export`로 가리켜도 그 폴더로
      본다. 돌려주는 말은 `paths.export_dir (<설정의 값>)`이다.
    - 이 저장소의 `export/`(`REPO_ROOT / SHIPPED_EXPORT_DIRNAME`): 설정이 무엇을 가리키든 늘 내보내기 폴더다. 연기 시험용
      설정은 `paths.export_dir`을 임시 폴더로 적어 두는데, 그 설정으로 저장소 루트에서 `--out-dir export`를 주면 합성 모델이나
      축소 설정 모델이 커밋하는 폴더에 들어가던 것을 막는다. 이 폴더만으로 걸렸을 때는 돌려주는 말이 그 폴더의 경로와
      설정의 값을 함께 적는다.
    - 경로는 모두 `resolve()`로 풀어 견준다. `..`, 끝의 `/`, 심볼릭 링크로 돌아 들어가는 경우도 같은 폴더다.
    가정: 저장소 루트는 이 모듈 파일의 위치에서 정한다(`drowsy/export_onnx.py`의 두 단계 위 폴더). 패키지를 다른 곳에
    복사해 쓰면 그 사본이 놓인 폴더가 기준이 되고, 원래 저장소의 `export/`는 절대 경로로 적은 설정으로만 지킬 수 있다.
    가정: 늘 지키는 폴더는 이름이 `export`인 그 폴더 하나다. 저장소 안의 다른 폴더(`out/`, `reports/`)는 설정이 가리킬 때만
    지킨다. 실제 데이터와 정식 설정으로 학습한 모델은 이 검사와 무관하게 어느 폴더로든 나간다.
    """
    target = Path(out_dir).resolve()

    def holds(folder: Path) -> bool:
        return target == folder or folder in target.parents

    configured = [Path(export_dir).resolve()]
    if not Path(export_dir).is_absolute():
        configured.append((REPO_ROOT / export_dir).resolve())
    if any(holds(folder) for folder in configured):
        return f"paths.export_dir ({export_dir})"
    shipped = (REPO_ROOT / SHIPPED_EXPORT_DIRNAME).resolve()
    if holds(shipped):
        return (
            f"the export folder of this repository ({shipped}; it is protected whatever paths.export_dir names, "
            f"here {export_dir})"
        )
    return None


def is_export_dir_or_below(out_dir: str | Path, export_dir: str | Path) -> bool:
    """out_dir이 내보내기 폴더이거나 그 아래 폴더인지: 설정의 `paths.export_dir`, 또는 이 저장소의 `export/`.

    판단은 `protected_export_dir`이 한다(무엇을 내보내기 폴더로 보는지와 경로를 견주는 방법은 그 설명에 있다).
    """
    return protected_export_dir(out_dir, export_dir) is not None


# ---------------------------------------------------------------------------
# 메타데이터
# ---------------------------------------------------------------------------


def _run_git(repo_dir: Path, *args: str) -> str | None:
    """git 명령의 표준 출력. 실패하면(종료 코드가 0이 아니거나 git이 없으면) None."""
    try:
        done = subprocess.run(["git", *args], cwd=repo_dir, capture_output=True, text=True, check=False)
    except OSError:
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip()


def git_commit(repo_dir: str | Path | None = None) -> str:
    """메타데이터 `commit` 값: `git rev-parse HEAD`의 출력. 작업 트리에 변경이 있으면 "-dirty"를 붙인다.

    저장소에 커밋이 아직 없으면 해시 자리에 "uncommitted"를 쓴다.
    repo_dir을 주지 않으면 이 패키지가 들어 있는 저장소를 본다.
    가정: 변경은 `git status --porcelain`의 출력이 비어 있지 않은 것으로 판정한다. 추적하지 않는 파일
    (.gitignore에 걸리지 않은 것)도 변경으로 센다.
    가정: git을 실행할 수 없거나 git 저장소가 아니면 "uncommitted"다("-dirty"는 붙지 않는다).
    """
    root = Path(repo_dir) if repo_dir is not None else Path(__file__).resolve().parents[1]
    head = _run_git(root, "rev-parse", "HEAD")
    commit = head if head else "uncommitted"
    if _run_git(root, "status", "--porcelain"):
        commit += "-dirty"
    return commit


def build_metadata(
    spec: FeatureSpec, feature_spec_hash: str, model_version: str, trained_on: str, commit: str
) -> dict[str, str]:
    """`METADATA_KEYS` 순서의 메타데이터. spec에서 나오는 값은 여기서만 만든다."""
    return {
        "model_version": model_version,
        "feature_spec_hash": feature_spec_hash,
        "step_sec": str(spec.step_sec),
        "window_steps": str(spec.window_steps),
        "class_order": ",".join(spec.class_order),
        "trained_on": trained_on,
        "commit": commit,
        "standardization": spec.standardization.location,
        "pipeline_hash": spec.pipeline_hash,
        "mediapipe_model_sha256": spec.pipeline.mediapipe.model_sha256,
    }


def _load_model(onnx_path: str | Path) -> onnx.ModelProto:
    """ONNX 파일을 읽는다(외부 데이터는 읽지 않는다). ONNX로 해석할 수 없는 파일이면 ContractError다.

    파일이 없거나 열 수 없으면 OSError가 그대로 나간다.
    가정: onnx는 해석할 수 없는 바이트에 protobuf의 DecodeError를 낸다. protobuf는 onnx가 끌어오는 패키지이고 이 프로젝트의
    의존성 목록(pyproject.toml)에 없으므로 그 형을 import하지 않고, OSError가 아닌 예외를 모두 "읽을 수 없는 파일"로 바꾼다.
    """
    try:
        return onnx.load(str(onnx_path), load_external_data=False)
    except OSError:
        raise
    except Exception as error:
        raise ContractError(f"{onnx_path}: not a readable ONNX file ({error})") from error


def read_metadata(onnx_path: str | Path) -> dict[str, str]:
    """파일의 `metadata_props`.

    ONNX로 해석할 수 없는 파일이면 ValueError(`ContractError`)이고 메시지에 "not a readable ONNX file"이 들어간다.
    파일이 없으면 OSError다. 부르는 쪽은 protobuf의 예외를 알 필요가 없다.
    """
    model = _load_model(onnx_path)
    return {prop.key: prop.value for prop in model.metadata_props}


def graph_operators(onnx_path: str | Path) -> tuple[str, ...]:
    """그래프에 쓰인 연산자 이름(중복 없이 정렬)."""
    model = _load_model(onnx_path)
    return tuple(sorted({node.op_type for node in model.graph.node}))


# ---------------------------------------------------------------------------
# 내보내기
# ---------------------------------------------------------------------------


def _floored_stds(spec: FeatureSpec) -> list[float]:
    return [max(std, spec.standardization.std_floor) for std in spec.stds]


def _as_float32(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _check_wrapper_matches_spec(model: ExportWrapper, spec: FeatureSpec) -> None:
    """래퍼의 상수가 spec 파일의 값과 같은지(PLAN 4.7절: 내보낼 때 두 값이 같은지 검사한다)."""
    expected = {
        _MEAN: _as_float32(spec.means),
        _STD: _as_float32(_floored_stds(spec)),
        _LOG1P_MASK: _as_float32([1.0 if flag else 0.0 for flag in spec.log1p_mask]),
    }
    for name, want in expected.items():
        have = getattr(model, name).detach().cpu().numpy()
        if have.shape != want.shape or not np.array_equal(have, want):
            raise ValueError(f"ExportWrapper.{name} does not match the feature spec")
    if np.float32(model.z_clip) != np.float32(spec.standardization.z_clip):
        raise ValueError(
            f"ExportWrapper.z_clip {model.z_clip!r} does not match the feature spec {spec.standardization.z_clip!r}"
        )
    temperature = float(model.temperature)
    if np.float32(temperature) != np.float32(spec.standardization.temperature):
        raise ValueError(
            f"ExportWrapper temperature {temperature!r} does not match standardization.temperature "
            f"{spec.standardization.temperature!r} in the feature spec; use the feature spec copy written by the "
            f"training run of this model ({EXPORTED_SPEC_PATTERN.format(version='{N}')} next to its exported model), "
            "not a spec edited by hand"
        )


def _to_onnx(model: ExportWrapper, spec: FeatureSpec, dynamo: bool) -> onnx.ModelProto:
    # 사본을 CPU, eval 모드로 바꿔서 쓴다. 받은 래퍼의 장치와 학습 모드는 건드리지 않는다.
    wrapper = copy.deepcopy(model).cpu().eval()
    dummy = torch.zeros(1, spec.window_steps, spec.n_features, dtype=torch.float32)
    with warnings.catch_warnings():
        # TorchScript 내보내기는 호출마다 폐기 예정 경고와 추적 경고를 낸다. 결과는 verify_contract가 검사한다.
        warnings.simplefilter("ignore")
        if dynamo:
            program = torch.onnx.export(
                wrapper,
                (dummy,),
                dynamo=True,
                opset_version=OPSET,
                input_names=[INPUT_NAME],
                output_names=[OUTPUT_NAME],
                external_data=False,
                verbose=False,
            )
            return program.model_proto
        buffer = io.BytesIO()
        torch.onnx.export(
            wrapper,
            (dummy,),
            buffer,
            dynamo=False,
            opset_version=OPSET,
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
        )
    return onnx.load_model_from_string(buffer.getvalue())


def finalize(model: onnx.ModelProto, metadata: Mapping[str, str]) -> onnx.ModelProto:
    """내보낸 그래프를 계약 형태로 맞춘다. 받은 모델을 고쳐서 돌려준다(opset을 바꿀 때는 새 객체).

    opset이 17이 아니면 17로 변환하고, value_info와 설명 문자열과 노드 메타데이터를 지우고,
    IR 버전을 8로 두고, `metadata_props`를 metadata로 바꿔 쓴 뒤 onnx 검사기를 돌린다.
    """
    if [(entry.domain, entry.version) for entry in model.opset_import] != [("", OPSET)]:
        model = onnx.version_converter.convert_version(model, OPSET)
    graph = model.graph
    del graph.value_info[:]
    del graph.metadata_props[:]
    model.doc_string = ""
    graph.doc_string = ""
    for node in graph.node:
        node.doc_string = ""
        del node.metadata_props[:]
    for tensor in graph.initializer:
        tensor.doc_string = ""
    for value in list(graph.input) + list(graph.output):
        value.doc_string = ""
    model.ir_version = IR_VERSION
    onnx.helper.set_model_props(model, dict(metadata))
    onnx.checker.check_model(model, full_check=True)
    return model


def _temporary_sibling(path: Path) -> Path:
    """같은 폴더의 임시 파일 이름. 같은 파일 시스템이라 `os.replace`로 한 번에 바꿔 놓을 수 있다."""
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def _write_bytes_atomically(path: Path, payload: bytes) -> None:
    """임시 이름으로 쓴 뒤 `os.replace`로 바꿔 놓는다. 중간에 실패하면 원래 파일은 그대로다."""
    temporary = _temporary_sibling(path)
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def export(
    model: ExportWrapper, spec_path: Path, out_path: Path, meta: Mapping[str, str], dynamo: bool = False
) -> Path:
    """래퍼를 ONNX 파일 하나로 내보내고, 같은 폴더에 `feature_spec_v{model_version}.json` 사본을 쓴다.

    out_path를 돌려준다. 사본의 이름은 `exported_spec_name(meta["model_version"])`이다.

    meta에는 `model_version`(음이 아닌 정수 문자열)과 `trained_on`을 준다. `commit`은 주지 않으면
    `git_commit()`으로 구한다. 나머지 일곱 개는 spec과 사본의 해시에서 만들므로 받지 않는다.
    래퍼의 상수(평균, 하한을 적용한 표준편차, log1p 마스크, z_clip, 온도)가 spec_path의 값과 다르면 ValueError다.
    같은 가중치와 같은 meta로 다시 내보내면 바이트 단위로 같은 파일이 나온다.
    사본은 `drowsy.spec.write_feature_spec`으로 쓴다. spec_path의 파일이 그 함수가 쓴 것이면 사본은 바이트 단위로 같다.

    쓰는 순서
    1. git 상태(`commit`)는 어떤 파일도 쓰기 전에 읽는다. 이 함수가 쓰는 사본과 모델은 새 버전이면 추적되지 않는
       파일이라, 쓴 뒤에 읽으면 깨끗한 작업 트리에서 내보낸 모델에 "-dirty"가 붙는다. 이 함수를 부르기 전에 같은
       저장소 안에 파일을 쓰는 쪽(`drowsy.train.export_fit`)은 그보다 먼저 `git_commit()`을 읽어 meta로 넘겨야 한다.
    2. 사본과 모델을 같은 폴더의 임시 이름으로 쓰고 `verify_contract`로 검사한다(PLAN 7장의 "내보낸 뒤 자동 검사").
    3. 통과하면 `os.replace`로 사본, 모델 순서로 제자리에 놓는다. 통과하지 못하거나 도중에 예외가 나면 임시 파일만
       지운다. 그 폴더에 있던 같은 버전의 모델과 사본은 바이트 하나 바뀌지 않는다(ContractError 등은 그대로 나간다).
    4. 같은 폴더의 `drowsiness_v{model_version}_int8.onnx`가 있으면 지운다. 그 파일은 앞서 내보낸 가중치와 사본에서
       만든 것이라 새 쌍과 맞지 않는다. INT8 파일이 필요하면 이 함수 뒤에 `quantize_int8`로 다시 만든다.
    가정: 계획서는 온도를 spec에도 적고 그래프에도 넣는다. 둘이 다르면 문서와 모델이 어긋나므로 내보내지 않는다.
    가정: 사본과 모델을 바꿔 놓는 두 번의 `os.replace` 사이에 프로세스가 죽거나 두 번째가 실패하는 경우는 다루지
    않는다(그때는 새 사본과 예전 모델이 남고, 계약 검사가 해시 불일치로 알려 준다).
    가정: 임시 파일 이름은 `.<파일 이름>.<프로세스 번호>.tmp`다. 프로세스가 강제로 종료되면 남을 수 있고 그때는 손으로
    지운다. 같은 폴더에 두 프로세스가 동시에 내보내는 경우는 다루지 않는다.
    가정: 같은 버전을 다시 내보내면 가중치가 같더라도 예전 INT8 파일을 지운다. 그 파일이 어느 내보내기에서 나온
    것인지 파일만으로는 알 수 없기 때문이다.
    """
    spec_path = Path(spec_path)
    out_path = Path(out_path)
    required = ("model_version", "trained_on")
    missing = [key for key in required if key not in meta]
    unknown = sorted(set(meta) - set(required) - {"commit"})
    if missing or unknown:
        raise ValueError(f"meta: missing keys {missing}, unexpected keys {unknown}")
    if not _is_version(meta["model_version"]):
        raise ValueError(f"meta['model_version'] must be a non-negative integer string, got {meta['model_version']!r}")
    if not meta["trained_on"]:
        raise ValueError("meta['trained_on'] must not be empty")
    # 아래에서 쓰는 파일이 작업 트리를 바꾸기 전에 읽는다(docstring 1번).
    commit = meta.get("commit") or git_commit()

    spec_dict = read_feature_spec_dict(spec_path)
    spec = feature_spec_from_dict(spec_dict)
    _check_wrapper_matches_spec(model, spec)
    # 변환이 실패하면 아무 파일도 쓰지 않도록 그래프를 먼저 만든다.
    proto = _to_onnx(model, spec, dynamo)

    out_dir = out_path.parent
    exported_spec = out_dir / exported_spec_name(meta["model_version"])
    temporary_spec = _temporary_sibling(exported_spec)
    temporary_model = _temporary_sibling(out_path)
    created_dir = not out_dir.exists()
    try:
        metadata = build_metadata(
            spec,
            feature_spec_hash=write_feature_spec(spec_dict, temporary_spec),
            model_version=meta["model_version"],
            trained_on=meta["trained_on"],
            commit=commit,
        )
        temporary_model.write_bytes(finalize(proto, metadata).SerializeToString())
        verify_contract(temporary_model, temporary_spec, ALLOWED_OPS_DYNAMO if dynamo else ALLOWED_OPS)
        os.replace(temporary_spec, exported_spec)
        os.replace(temporary_model, out_path)
    except BaseException:
        temporary_spec.unlink(missing_ok=True)
        temporary_model.unlink(missing_ok=True)
        if created_dir:
            # 이 호출이 만든 빈 폴더는 남기지 않는다. 비어 있지 않으면 rmdir이 실패하고 그대로 둔다.
            try:
                out_dir.rmdir()
            except OSError:
                pass
        raise
    stale_int8 = out_dir / model_file_name(meta["model_version"], int8=True)
    if stale_int8 != out_path:
        stale_int8.unlink(missing_ok=True)
    return out_path


# ---------------------------------------------------------------------------
# 계약 검사
# ---------------------------------------------------------------------------


def _static_shape(value: onnx.ValueInfoProto) -> list[int | None]:
    dims = value.type.tensor_type.shape.dim
    return [dim.dim_value if dim.HasField("dim_value") else None for dim in dims]


def _check_io(values: Sequence[onnx.ValueInfoProto], kind: str, name: str, shape: Sequence[int]) -> None:
    _require(len(values) == 1, f"expected exactly one graph {kind}, got {[value.name for value in values]}")
    value = values[0]
    _require(value.name == name, f"graph {kind} must be named {name!r}, got {value.name!r}")
    elem_type = value.type.tensor_type.elem_type
    _require(elem_type == onnx.TensorProto.FLOAT, f"{name}: element type must be float32, got {elem_type}")
    _require(_static_shape(value) == list(shape), f"{name}: shape must be {list(shape)}, got {_static_shape(value)}")


def _graph_tensors(graph: onnx.GraphProto) -> list[onnx.TensorProto]:
    tensors = list(graph.initializer)
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                tensors.append(attribute.t)
            tensors.extend(attribute.tensors)
    return tensors


def _graph_constants(graph: onnx.GraphProto) -> dict[str, np.ndarray]:
    """이름 -> 값. initializer와 Constant 노드의 출력."""
    constants = {tensor.name: numpy_helper.to_array(tensor) for tensor in graph.initializer}
    for node in graph.node:
        if node.op_type != "Constant":
            continue
        for attribute in node.attribute:
            if attribute.name == "value":
                constants[node.output[0]] = numpy_helper.to_array(attribute.t)
    return constants


def _single_consumer(graph: onnx.GraphProto, name: str, op_type: str) -> onnx.NodeProto:
    """상수 `name`을 둘째 입력으로 쓰는 단 하나의 노드."""
    consumers = [node for node in graph.node if name in node.input]
    _require(
        len(consumers) == 1 and consumers[0].op_type == op_type and list(consumers[0].input).index(name) == 1,
        f"constant {name!r} must be the second input of exactly one {op_type} node",
    )
    return consumers[0]


def _require_constant(constants: Mapping[str, np.ndarray], name: str, expected: np.ndarray, what: str) -> None:
    _require(name in constants, f"{what}: {name!r} is not a constant in the graph")
    value = constants[name]
    _require(
        value.dtype == np.float32 and value.shape == expected.shape and np.array_equal(value, expected),
        f"{what} in the graph does not match the feature spec",
    )


def _producer(graph: onnx.GraphProto, name: str, op_type: str, what: str) -> onnx.NodeProto:
    """값 `name`을 내는 노드. 그런 노드가 하나가 아니거나 연산자가 op_type이 아니면 ContractError다."""
    producers = [node for node in graph.node if name in node.output]
    _require(
        len(producers) == 1 and producers[0].op_type == op_type,
        f"{what} must be the output of one {op_type} node, got {[node.op_type for node in producers]}",
    )
    return producers[0]


def _readers(graph: onnx.GraphProto, name: str) -> list[onnx.NodeProto]:
    """값 `name`을 입력으로 읽는 노드. 한 노드가 두 자리에서 읽으면 두 번 들어간다."""
    return [node for node in graph.node for item in node.input if item == name]


def _same_node(a: onnx.NodeProto, b: onnx.NodeProto) -> bool:
    # 그래프의 값 이름은 한 번만 정의되므로(onnx 검사기가 확인한다) 출력 이름이 같으면 같은 노드다.
    return list(a.output) == list(b.output)


def _require_only_consumer(graph: onnx.GraphProto, name: str, consumer: onnx.NodeProto, what: str) -> None:
    """값 `name`을 쓰는 곳이 consumer 한 노드의 입력 한 자리뿐인지(그래프 출력으로도 나가지 않는지)."""
    uses = _readers(graph, name)
    _require(
        len(uses) == 1 and _same_node(uses[0], consumer) and all(value.name != name for value in graph.output),
        f"{what} must feed only the next step of the in-graph pre/post-processing, "
        f"got consumers {[node.op_type for node in uses]}",
    )


def _attribute_type_name(attribute: onnx.AttributeProto) -> str:
    """속성의 형 이름(`FLOAT`, `UNDEFINED` 등). ONNX에 없는 형 번호면 그 번호."""
    try:
        return onnx.AttributeProto.AttributeType.Name(attribute.type)
    except ValueError:
        return str(attribute.type)


def _attribute(node: onnx.NodeProto, name: str, default: float | int, what: str) -> float | int:
    """노드의 숫자 속성 값. 그 속성이 없으면 default(ONNX가 정한 기본값을 부르는 쪽이 준다).

    default가 float이면 속성은 FLOAT, int이면 INT여야 한다. 같은 이름의 속성이 둘 이상이거나, 형이 다르거나(UNDEFINED,
    INT 자리의 FLOAT, 문자열, 텐서 등), 형에 맞는 값이 비어 있으면 ContractError다(한 줄 이유. what은 그 노드가 무엇인지).
    전에는 값을 그대로 꺼내 견주고 메시지에 숫자 형식으로 넣다가, UNDEFINED 형(값이 None)이나 문자열·텐서에서 TypeError가
    났다. 값을 견주기 전에 형을 확인하므로 메시지에 넣는 값은 언제나 숫자다.
    형이 틀린 속성(UNDEFINED, 문자열, 텐서 등)은 이 확인이 없어도 onnx 검사기나 onnxruntime이 거부하므로, 이 확인은 받는
    파일을 늘리지 않고 무엇이 틀렸는지를 먼저 한 줄로 알린다.
    가정: 값이 비어 있는 속성도 거부한다. onnxruntime은 값이 없는 INT 속성(transA 등)을 기본값으로 읽어 그대로 실행하지만,
    내보내기는 그런 파일을 만들지 않으므로 받지 않는 쪽을 택했다(내보낸 파일과 v0은 모든 속성에 값이 있다).
    """
    found = [attribute for attribute in node.attribute if attribute.name == name]
    if not found:
        return default
    kind, field = (onnx.AttributeProto.FLOAT, "f") if isinstance(default, float) else (onnx.AttributeProto.INT, "i")
    attribute = found[0]
    if len(found) != 1 or attribute.type != kind or not attribute.HasField(field):
        got = [
            _attribute_type_name(item) + ("" if item.type != kind or item.HasField(field) else " without a value")
            for item in found
        ]
        raise ContractError(
            f"attribute {name!r} of {what} must appear at most once, with type "
            f"{onnx.AttributeProto.AttributeType.Name(kind)} and a value; got {', '.join(got)}; {_LOGITS_SCOPE}"
        )
    return attribute.f if kind == onnx.AttributeProto.FLOAT else attribute.i


def _is_float_constant(constants: Mapping[str, np.ndarray], name: str) -> bool:
    return name in constants and constants[name].dtype == np.float32


def _check_logits_producer(
    graph: onnx.GraphProto, constants: Mapping[str, np.ndarray], logits: str, quantized: bool
) -> None:
    """로짓(값 이름 logits)을 내는 노드가 내보내기가 쓰는 신경망의 마지막 선형 층 그대로인지. 아니면 ContractError다.

    로짓과 온도 Div(또는 Softmax) 사이에 값을 바꾸지 않는 노드를 하나 끼우고 그 앞에서 로짓에 상수를 곱하는 변형, 곱하는
    상수를 마지막 층의 속성이나 둘째 선형 층에 넣는 변형을 막는다. 두 계열(TCN, GRU) 모두 선형 층은 맨 끝의 하나뿐이다.

    - 기본 경로와 dynamo 경로(quantized가 False): 로짓은 Gemm 노드 하나의 출력이다. 그 Gemm은 그래프의 유일한 Gemm이고,
      alpha = 1, beta = 1, transA = 0이며(속성이 없으면 ONNX의 기본값 1, 1, 0), 가중치(둘째 입력)와 편향(셋째 입력)이
      그래프의 float32 상수다. 세 속성은 있으면 한 번씩, 맞는 형(alpha와 beta는 FLOAT, transA는 INT)과 값으로 있어야
      한다(`_attribute`. INT8 파일의 Cast의 `to`도 같다).
    - INT8 파일(quantized가 True): 양자화기가 Gemm을 바꿔 쓴 모양이어야 한다.

          hidden ─ DynamicQuantizeLinear ─┬ (양자화한 값, 영점) ─ MatMulInteger(·, 가중치) ─ Cast(float) ─┐
                                          └ (배율) ─ Mul(·, 가중치 배율 상수) ───────────────────────────┴ Mul ─ Add(·, 편향) = 로짓

      로짓은 float32 상수(편향) 하나를 더하는 Add의 출력이고, 더해지는 값은 Mul의 출력이며, 그 Mul의 한 입력은 그래프의
      유일한 MatMulInteger의 출력을 float32로 바꾼 Cast의 출력, 다른 입력은 그 MatMulInteger의 입력을 만든
      DynamicQuantizeLinear의 배율 출력에 float32 상수 하나를 곱한 Mul의 출력이다. MatMulInteger의 가중치와 그 영점은
      그래프의 상수다.

    이 검사가 보지 못하는 것: 마지막 층의 상수 자체(가중치와 편향, INT8 파일의 가중치 배율)를 통째로 줄이거나 키운 파일.
    그래프의 모양과 속성은 그대로이기 때문이다. 그런 파일은 체크포인트와의 일치 검증(`parity_max_abs_diff`)이 잡는다.
    신경망의 안쪽(Clip과 마지막 층 사이)도 여기서 보지 않는다.
    가정: INT8 파일의 모양은 이 프로젝트가 쓰는 onnxruntime의 `quantize_dynamic`이 Gemm에 대해 내는 모양이다. 양자화기가
    다른 모양을 내는 버전으로 바뀌면 `ALLOWED_OPS_INT8`과 함께 이 검사도 다시 맞춰야 한다(그때는 INT8 파일이 계약 검사에서
    걸리므로 조용히 지나가지 않는다).
    """
    producers = [node for node in graph.node if logits in node.output]
    found = [node.op_type for node in producers]
    if not quantized:
        gemms = [node for node in graph.node if node.op_type == "Gemm"]
        _require(
            len(producers) == 1 and found == ["Gemm"] and len(gemms) == 1,
            "the logits must be the output of the network's last linear layer, which is the only Gemm node of the "
            f"graph; got producer {found} and {len(gemms)} Gemm node(s); {_LOGITS_SCOPE}",
        )
        gemm = producers[0]
        what = "the Gemm node that makes the logits"
        alpha, beta = _attribute(gemm, "alpha", 1.0, what), _attribute(gemm, "beta", 1.0, what)
        trans_a = _attribute(gemm, "transA", 0, what)
        _require(
            alpha == 1.0 and beta == 1.0 and trans_a == 0,
            "the Gemm node that makes the logits must have alpha = 1, beta = 1 and transA = 0, "
            f"got alpha {alpha:g}, beta {beta:g}, transA {trans_a}; {_LOGITS_SCOPE}",
        )
        _require(
            len(gemm.input) == 3 and all(_is_float_constant(constants, name) for name in gemm.input[1:]),
            "the Gemm node that makes the logits must read its weights and its bias from float32 constants of the "
            f"graph; {_LOGITS_SCOPE}",
        )
        return

    matmuls = [node for node in graph.node if node.op_type == "MatMulInteger"]
    pattern = (
        "the logits of an INT8 file must come from the quantized last linear layer as the quantizer writes it "
        "(DynamicQuantizeLinear -> the only MatMulInteger -> Cast to float32 -> Mul by the scales -> Add of the "
        "constant bias)"
    )
    _require(
        len(producers) == 1 and found == ["Add"] and len(matmuls) == 1,
        f"{pattern}; got producer {found} and {len(matmuls)} MatMulInteger node(s); {_LOGITS_SCOPE}",
    )
    bias_add, matmul = producers[0], matmuls[0]
    biases = [name for name in bias_add.input if _is_float_constant(constants, name)]
    _require(
        len(bias_add.input) == 2 and len(biases) == 1,
        f"{pattern}; the Add that makes the logits must add exactly one float32 constant (the bias); {_LOGITS_SCOPE}",
    )

    def made_by(name: str, op_type: str) -> onnx.NodeProto | None:
        """값 name을 내는 단 하나의 노드가 op_type이면 그 노드, 아니면 None."""
        nodes = [node for node in graph.node if name in node.output]
        return nodes[0] if len(nodes) == 1 and nodes[0].op_type == op_type else None

    product = made_by(next(name for name in bias_add.input if name != biases[0]), "Mul")
    quantize = made_by(matmul.input[0], "DynamicQuantizeLinear") if len(matmul.input) == 4 else None
    chain_ok = False
    if product is not None and quantize is not None and len(product.input) == 2 and len(quantize.output) == 3:
        casts = [made_by(name, "Cast") for name in product.input]
        scales = [made_by(name, "Mul") for name in product.input]
        cast = next((node for node in casts if node is not None), None)
        scale = next((node for node in scales if node is not None), None)
        chain_ok = (
            cast is not None
            and scale is not None
            and list(cast.input) == [matmul.output[0]]
            and _attribute(cast, "to", 0, "the Cast node of the quantized last linear layer") == onnx.TensorProto.FLOAT
            and matmul.input[0] == quantize.output[0]
            and matmul.input[2] == quantize.output[2]
            and all(name in constants for name in (matmul.input[1], matmul.input[3]))
            and len(scale.input) == 2
            and quantize.output[1] in scale.input
            and sum(1 for name in scale.input if _is_float_constant(constants, name)) == 1
        )
    _require(chain_ok, f"{pattern}; the nodes behind the bias Add do not have that shape; {_LOGITS_SCOPE}")


def _check_graph_constants(
    graph: onnx.GraphProto, spec: FeatureSpec, temperature_may_be_absent: bool = False, quantized: bool = False
) -> None:
    """그래프 안 전처리·후처리의 상수가 spec과 같은지, 그리고 연산이 정해진 순서로 이어져 있는지.

    temperature_may_be_absent는 dynamo 경로의 파일을 검사할 때만 True다(아래 온도 항목). quantized는 INT8 파일을 검사할
    때만 True다(아래 로짓 항목).

    상수: 평균, 하한을 적용한 표준편차, log1p 마스크와 그 여집합, ±z_clip, 온도, log1p의 1.0.
    연결(세 내보내기 경로와 INT8 파일이 모두 이 모양이다):

        features ─ Mul(·, keep_mask) ───────────────────────────┐
        features ─ Relu ─ Add(·, 1.0) ─ Log ─ Mul(·, log1p_mask) ┴ Add ─ Sub(·, mean) ─ Div(·, std)
                 ─ Clip(·, −z_clip, +z_clip) ─ [신경망] ─ Div(·, temperature) ─ Softmax(마지막 축) ─ probs

    - `features`를 읽는 노드는 맨 앞의 Mul과 Relu 둘뿐이다. 그래서 입력에 다른 연산(시간 순서 뒤집기 등)을 끼울 수
      없고, 원시 값이 전처리를 건너뛰어 신경망에 들어갈 수 없다.
    - Clip까지의 중간 값은 각각 다음 단계 하나만 읽는다.
    - Softmax는 하나이고 마지막 축(axis −1, 출력이 2차원이라 1도 같다)에 걸리며 그 출력이 곧 그래프 출력 `probs`다.
      그래서 Softmax 뒤에 다른 연산(클래스 순서 바꾸기 등)을 끼울 수 없다.
    - 온도로 나누는 Div는 Softmax 바로 앞에 있고, 나누는 값은 `temperature`라는 이름의 상수이며 spec의 온도와 같다.
      기본 경로와 INT8 파일은 온도가 1.0이어도 이 상수와 Div를 그대로 갖고 있으므로, 그 상수가 없는 그래프는 받지 않는다
      (상수 이름을 바꿔 다른 값으로 나누게 고친 파일이 온도 1.0인 spec과 함께 통과하던 것을 막는다).
    - dynamo 경로만 온도가 1.0이면 그 Div와 상수를 지운다. 그 경로의 파일은(temperature_may_be_absent) spec의 온도가
      1.0이면 상수가 없어도 받는다.
    - 로짓(온도 Div의 첫 입력. 온도 상수가 없는 dynamo 그래프에서는 Softmax의 입력)은 Div나 Mul의 출력이 아니어야 한다.
      온도 말고 다른 상수로 로짓을 한 번 더 나누거나 곱한 그래프를 막는다.
    - 로짓을 내는 노드는 신경망의 마지막 선형 층 그대로여야 한다(`_check_logits_producer`): 기본 경로와 dynamo 경로는
      그래프의 유일한 Gemm(alpha = beta = 1, transA = 0, 상수 가중치와 편향), INT8 파일은 양자화기가 그 Gemm을 바꿔 쓴
      모양의 끝에 있는 편향 Add다. 로짓에 상수를 곱한 뒤 값을 바꾸지 않는 노드(0을 더하는 Add, 입력이 하나인 Concat,
      순서를 바꾸지 않는 Transpose 등)를 하나 끼워 위 항목을 비켜 가는 변형과, 배율을 Gemm의 속성이나 둘째 Gemm에 넣는
      변형을 막는다.
    Clip과 마지막 층 사이(신경망)는 계열마다 달라서 그 밖에는 보지 않는다. 그 부분은 허용 연산자 목록과 일치 검증이 맡는다.
    가정: log1p의 Add는 두 입력의 순서를 가리지 않고(덧셈이라 결과가 같다), Softmax의 축은 −1과 1을 같은 것으로 받는다.
    가정: 신경망은 두 계열 모두 선형 층 하나로 끝나고 그래프에 선형 층은 그것뿐이다(`drowsy.models`의 `head`). 그래서
    로짓을 내는 노드를 그 층의 모양으로 고정한다. 마지막 층의 상수(가중치와 편향)를 통째로 줄이거나 키운 파일은 그래프의
    모양이 같아서 이 검사로 가릴 수 없다. 그것은 체크포인트와의 일치 검증(`parity_max_abs_diff`)이 잡는다.
    """
    constants = _graph_constants(graph)
    std_floor = np.float32(spec.standardization.std_floor)
    mask = _as_float32([1.0 if flag else 0.0 for flag in spec.log1p_mask])

    _require_constant(constants, _MEAN, _as_float32(spec.means), "mean")
    _require_constant(constants, _STD, _as_float32(_floored_stds(spec)), "std")
    _require(bool(np.all(constants[_STD] >= std_floor)), "std in the graph is below standardization.std_floor")
    _require_constant(constants, _LOG1P_MASK, mask, "log1p mask")
    _require_constant(constants, _KEEP_MASK, np.float32(1.0) - mask, "keep mask")

    # 변환: x · keep + log(max(x, 0) + 1) · mask
    keep_mul = _single_consumer(graph, _KEEP_MASK, "Mul")
    _require(keep_mul.input[0] == INPUT_NAME, f"the keep-mask Mul must read the graph input {INPUT_NAME!r}")
    log_mul = _single_consumer(graph, _LOG1P_MASK, "Mul")
    log = _producer(graph, log_mul.input[0], "Log", "the value multiplied by the log1p mask")
    plus_one = _producer(graph, log.input[0], "Add", "the argument of the log1p Log")
    ones = [name for name in plus_one.input if name in constants]
    _require(
        len(plus_one.input) == 2 and len(ones) == 1,
        "the Add before the log1p Log must add one constant to max(features, 0)",
    )
    one = constants[ones[0]]
    _require(
        one.dtype == np.float32 and one.size == 1 and float(one.reshape(-1)[0]) == 1.0,
        f"the constant added before the log1p Log must be float32 1.0, got {one.dtype} {one.tolist()}",
    )
    relu_output = next(name for name in plus_one.input if name != ones[0])
    relu = _producer(graph, relu_output, "Relu", "the value the log1p constant is added to")
    _require(
        list(relu.input) == [INPUT_NAME], f"the Relu before the log1p Log must read the graph input {INPUT_NAME!r}"
    )
    readers = _readers(graph, INPUT_NAME)
    _require(
        len(readers) == 2
        and any(_same_node(node, keep_mul) for node in readers)
        and any(_same_node(node, relu) for node in readers),
        f"the graph input {INPUT_NAME!r} must be read only by the keep-mask Mul and the log1p Relu, "
        f"got consumers {[node.op_type for node in readers]}",
    )

    # 표준화: (변환한 값 − mean) / std → Clip
    sub = _single_consumer(graph, _MEAN, "Sub")
    merge = _producer(graph, sub.input[0], "Add", "the value the mean is subtracted from")
    _require(
        sorted(merge.input) == sorted([keep_mul.output[0], log_mul.output[0]]),
        "the mean must be subtracted from the sum of the keep-mask product and the log1p-mask product",
    )
    div = _single_consumer(graph, _STD, "Div")
    _require(div.input[0] == sub.output[0], "std must divide the output of the mean subtraction")
    clips = [node for node in graph.node if node.op_type == "Clip"]
    _require(
        len(clips) == 1 and len(clips[0].input) == 3 and clips[0].input[0] == div.output[0],
        "expected exactly one Clip node, applied to the standardized features with constant bounds",
    )
    z_clip = np.float32(spec.standardization.z_clip)
    _require_constant(constants, clips[0].input[1], np.asarray(-z_clip, dtype=np.float32), "lower clip bound")
    _require_constant(constants, clips[0].input[2], np.asarray(z_clip, dtype=np.float32), "upper clip bound")
    chain = (
        (relu, plus_one, "max(features, 0)"),
        (plus_one, log, "max(features, 0) + 1"),
        (log, log_mul, "the log1p value"),
        (keep_mul, merge, "the keep-mask product"),
        (log_mul, merge, "the log1p-mask product"),
        (merge, sub, "the transformed features"),
        (sub, div, "the mean-subtracted features"),
        (div, clips[0], "the standardized features"),
    )
    for node, consumer, what in chain:
        _require_only_consumer(graph, node.output[0], consumer, what)

    # 후처리: 로짓 / 온도 → Softmax(마지막 축) → probs
    softmaxes = [node for node in graph.node if node.op_type == "Softmax"]
    _require(len(softmaxes) == 1, f"expected exactly one Softmax node, got {len(softmaxes)}")
    softmax = softmaxes[0]
    axis = next((attribute.i for attribute in softmax.attribute if attribute.name == "axis"), -1)
    _require(axis in (-1, 1), f"Softmax must run over the class axis (-1), got axis {axis}")
    _require(
        list(softmax.output) == [OUTPUT_NAME],
        f"the Softmax output must be the graph output {OUTPUT_NAME!r}, got {list(softmax.output)}",
    )
    temperature = np.float32(spec.standardization.temperature)
    if _TEMPERATURE in constants:
        scale = _single_consumer(graph, _TEMPERATURE, "Div")
        _require(softmax.input[0] == scale.output[0], "the temperature division must feed the Softmax node")
        _require_only_consumer(graph, scale.output[0], softmax, "the temperature-scaled logits")
        _require_constant(constants, _TEMPERATURE, np.asarray(temperature, dtype=np.float32), "temperature")
        logits = scale.input[0]
    else:
        # dynamo 경로는 1.0으로 나누는 노드를 지운다. 그 경로에서, 온도가 1.0일 때만 온도 상수가 없어도 된다.
        _require(
            temperature_may_be_absent,
            f"the graph has no constant named {_TEMPERATURE!r}: the logits must be divided by it right before the "
            "Softmax node (only a file of the dynamo path may leave it out, and only for temperature 1.0)",
        )
        _require(
            temperature == np.float32(1.0),
            "the graph has no temperature constant but standardization.temperature is not 1.0",
        )
        logits = softmax.input[0]
    scaling = [node.op_type for node in graph.node if logits in node.output and node.op_type in _SCALING]
    _require(
        not scaling,
        f"the logits are the output of a {'/'.join(scaling)} node: only the division by the {_TEMPERATURE!r} constant "
        "may scale them (it is left out by the dynamo path when the temperature is 1.0)",
    )
    _check_logits_producer(graph, constants, logits, quantized)


def _session(onnx_path: str | Path, intra_op_threads: int = 1) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op_threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])


def verify_contract(onnx_path: Path, spec_path: Path, allowed_ops: frozenset[str] = ALLOWED_OPS) -> None:
    """내보낸 파일이 계약을 지키는지 검사한다. 어기면 ContractError다.

    spec_path는 모델 옆의 `feature_spec_v{N}.json` 사본이다(그 바이트의 sha256이 `feature_spec_hash`,
    경로는 `exported_spec_path(onnx_path)`).
    allowed_ops는 파일 종류에 맞는 허용 연산자 목록이다: 기본 경로는 `ALLOWED_OPS`, INT8 파일은
    `ALLOWED_OPS_INT8`, dynamo 경로는 `ALLOWED_OPS_DYNAMO`.
    가정: 파일 종류를 따로 받지 않고 allowed_ops로 정한다. `ALLOWED_OPS_DYNAMO`와 같은 목록을 준 검사만 dynamo 경로의
    파일로 보고, spec의 온도가 1.0일 때 온도 상수가 없는 그래프를 받는다(`_check_graph_constants`). `ALLOWED_OPS_INT8`과
    같은 목록을 준 검사만 INT8 파일로 보고, 로짓을 내는 노드를 양자화한 선형 층의 모양과 견준다. 그 밖의 목록은 Gemm과 견준다.
    검사 항목: 파일 크기 1 MB 미만, opset 정확히 17, IR 8, 외부 데이터 없음, 입출력 이름·형·모양,
    메타데이터 열 개와 그 값, 허용 연산자, 그래프 안 상수와 spec 값의 일치, 표준편차 하한,
    전처리·후처리 연산의 연결(`_check_graph_constants`의 설명: log1p(max(x, 0)) 사슬, 입력을 읽는 노드,
    Softmax의 축과 출력, Softmax 바로 앞에서 `temperature` 상수로 나누는 Div와 그 밖의 로짓 배율 금지, 로짓을 내는 노드가
    신경망의 마지막 선형 층 그대로인지), onnx 검사기, onnxruntime 로드와 0 입력 한 번의 실행.
    파일 이름은 보지 않는다(`verify_file_name`). ONNX로 해석할 수 없는 파일도 ContractError다.

    이 검사가 보장하지 않는 것: 가중치가 맞는지. 신경망 안쪽(Clip과 마지막 선형 층 사이)과 마지막 층의 상수 값은 보지
    않으므로, 가중치와 편향을 통째로 줄이거나 키운 파일(로짓에 상수를 곱한 것과 같다)도 그래프의 모양만 맞으면 통과한다.
    가중치는 체크포인트의 PyTorch 모델과 출력을 견주는 일치 검증(`parity_max_abs_diff`. `export_onnx --checkpoint`와
    `train final`이 내보낸 직후에 한다)이 지킨다. 받은 파일을 의심할 때도 체크포인트의 모델과 출력을 견준다.
    """
    onnx_path = Path(onnx_path)
    spec_path = Path(spec_path)
    spec = load_feature_spec(spec_path)

    size = onnx_path.stat().st_size
    _require(size < MAX_FILE_BYTES, f"file is {size} bytes, must be under {MAX_FILE_BYTES}")

    model = _load_model(onnx_path)
    graph = model.graph
    opsets = [(entry.domain, entry.version) for entry in model.opset_import]
    _require(opsets == [("", OPSET)], f"opset_import must be exactly [('', {OPSET})], got {opsets}")
    _require(model.ir_version == IR_VERSION, f"ir_version must be {IR_VERSION}, got {model.ir_version}")
    _require(len(model.functions) == 0, "the model must not contain local functions")
    for tensor in _graph_tensors(graph):
        _require(
            tensor.data_location != onnx.TensorProto.EXTERNAL and len(tensor.external_data) == 0,
            f"tensor {tensor.name!r} uses external data; the model must be a single file",
        )

    _check_io(graph.input, "input", INPUT_NAME, (1, spec.window_steps, spec.n_features))
    _check_io(graph.output, "output", OUTPUT_NAME, (1, len(spec.class_order)))

    keys = [prop.key for prop in model.metadata_props]
    _require(sorted(keys) == sorted(METADATA_KEYS), f"metadata keys must be {list(METADATA_KEYS)}, got {keys}")
    metadata = {prop.key: prop.value for prop in model.metadata_props}
    _require(_is_version(metadata["model_version"]), "metadata model_version must be a non-negative integer string")
    _require(metadata["trained_on"] != "" and metadata["commit"] != "", "metadata trained_on and commit must be set")
    expected = build_metadata(
        spec, file_sha256(spec_path), metadata["model_version"], metadata["trained_on"], metadata["commit"]
    )
    for key in METADATA_KEYS:
        _require(
            metadata[key] == expected[key], f"metadata {key} is {metadata[key]!r}, expected {expected[key]!r}"
        )
    _require(
        metadata["standardization"] == STANDARDIZATION_IN_GRAPH,
        f"metadata standardization must be {STANDARDIZATION_IN_GRAPH!r}",
    )

    domains = sorted({node.domain for node in graph.node} - {"", "ai.onnx"})
    _require(not domains, f"operators outside the default ONNX domain: {domains}")
    unexpected = sorted({node.op_type for node in graph.node} - allowed_ops)
    _require(not unexpected, f"operators not in the allowed list: {unexpected}")

    _check_graph_constants(
        graph,
        spec,
        temperature_may_be_absent=allowed_ops == ALLOWED_OPS_DYNAMO,
        quantized=allowed_ops == ALLOWED_OPS_INT8,
    )

    try:
        onnx.checker.check_model(model, full_check=True)
    except onnx.checker.ValidationError as error:
        raise ContractError(f"onnx checker failed: {error}") from error

    session = _session(onnx_path)
    _require(
        session.get_modelmeta().custom_metadata_map == metadata,
        "onnxruntime custom_metadata_map differs from metadata_props",
    )
    zeros = np.zeros((1, spec.window_steps, spec.n_features), dtype=np.float32)
    probs = session.run([OUTPUT_NAME], {INPUT_NAME: zeros})[0]
    _require(
        probs.dtype == np.float32 and probs.shape == (1, len(spec.class_order)),
        f"onnxruntime output must be float32 [1, {len(spec.class_order)}], got {probs.dtype} {list(probs.shape)}",
    )
    _require(
        bool(np.all(np.isfinite(probs))) and abs(float(probs.sum()) - 1.0) < 1e-5,
        f"probabilities for a zero input must be finite and sum to 1, got {probs.tolist()}",
    )


# ---------------------------------------------------------------------------
# 일치 검증, 지연, INT8
# ---------------------------------------------------------------------------


def random_raw_inputs(spec: FeatureSpec, n: int, seed: int = 0) -> np.ndarray:
    """일치 검증용 무작위 원시 특징 [n, window_steps, F] float32. 같은 인자면 같은 값이다.

    표준화 공간에서 N(0, 2²)를 뽑아 spec의 평균·표준편차와 log1p의 역변환으로 원시 척도로 되돌린다.
    그래서 특징마다 그 특징의 척도를 갖고, 일부 값은 ±z_clip 밖에 놓이며, log1p 열에는 음수도 섞인다.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 2.0, size=(n, spec.window_steps, spec.n_features))
    x = z * np.asarray(_floored_stds(spec)) + np.asarray(spec.means)
    mask = np.asarray(spec.log1p_mask)
    x[..., mask] = np.expm1(np.minimum(x[..., mask], _MAX_LOG_VALUE))
    return x.astype(np.float32)


def parity_max_abs_diff(model: ExportWrapper, onnx_path: Path, inputs: np.ndarray) -> float:
    """PyTorch 래퍼와 onnxruntime 출력의 최대 절대 오차. inputs는 원시 특징 [N, 120, F]이고 유한값이어야 한다.

    앱과 같이 창 하나씩(배치 1) 넣는다. 래퍼는 사본을 CPU, eval 모드로 바꿔 쓴다.
    """
    batch = np.asarray(inputs, dtype=np.float32)
    if batch.ndim != 3 or batch.shape[0] == 0:
        raise ValueError(f"inputs must have shape [N, steps, features] with N >= 1, got {list(batch.shape)}")
    if not np.all(np.isfinite(batch)):
        raise ValueError("inputs must be finite")
    wrapper = copy.deepcopy(model).cpu().eval()
    session = _session(onnx_path)
    worst = 0.0
    with torch.no_grad():
        for index in range(batch.shape[0]):
            window = batch[index : index + 1]
            reference = wrapper(torch.from_numpy(window)).numpy()
            produced = session.run([OUTPUT_NAME], {INPUT_NAME: window})[0]
            worst = max(worst, float(np.max(np.abs(produced - reference))))
    return worst


def measure_latency(onnx_path: Path, runs: int = 1000, intra_op_threads: int = 1) -> dict[str, float]:
    """onnxruntime CPU 추론 한 번의 지연. `{"median_ms": ..., "p95_ms": ...}`.

    연산 내부 스레드는 기본 1개다. 준비 실행 뒤에 runs번을 한 번씩 재고, 95 분위수는 numpy의 선형 보간 값이다.
    """
    if runs < 1:
        raise ValueError(f"runs must be >= 1, got {runs}")
    session = _session(onnx_path, intra_op_threads)
    shape = session.get_inputs()[0].shape
    window = np.random.default_rng(0).normal(0.0, 1.0, size=shape).astype(np.float32)
    feed = {INPUT_NAME: window}
    for _ in range(min(runs, 100)):
        session.run([OUTPUT_NAME], feed)
    elapsed_ms = np.empty(runs, dtype=np.float64)
    for index in range(runs):
        start = time.perf_counter_ns()
        session.run([OUTPUT_NAME], feed)
        elapsed_ms[index] = (time.perf_counter_ns() - start) / 1e6
    return {"median_ms": float(np.median(elapsed_ms)), "p95_ms": float(np.percentile(elapsed_ms, 95))}


def quantize_int8(onnx_path: Path, out_path: Path) -> Path:
    """동적 INT8 양자화(`quantize_dynamic`, 가중치 형 QUInt8). out_path를 돌려준다 (계획서의 결정 D16).

    양자화 전에 value_info를 지운다(양자화기가 Gemm의 가중치를 같은 이름으로 전치해 모양 추론이 실패할 수 있다).
    양자화기가 메타데이터에 자기 키를 더하고 IR 버전은 입력을 따르므로, 원본의 메타데이터 열 개와 IR 8을 다시 쓴다.
    TCN 전용이다. GRU 연산자는 양자화되지 않는다. 결과는 `verify_contract(out_path, spec_path, ALLOWED_OPS_INT8)`로
    검사한다. spec_path는 원본 모델 옆의 `feature_spec_v{N}.json` 사본이다. 이 함수는 검사하지 않으므로, 부르는 쪽이
    검사하고 통과하지 못한 파일은 지워야 한다. 파일은 임시 이름으로 쓴 뒤 바꿔 놓는다.
    """
    onnx_path = Path(onnx_path)
    out_path = Path(out_path)
    source = _load_model(onnx_path)
    metadata = {prop.key: prop.value for prop in source.metadata_props}
    del source.graph.value_info[:]
    with tempfile.TemporaryDirectory() as work_dir:
        float_path = Path(work_dir) / "float.onnx"
        int8_path = Path(work_dir) / "int8.onnx"
        float_path.write_bytes(source.SerializeToString())
        quantize_dynamic(float_path, int8_path, weight_type=QuantType.QUInt8)
        quantized = onnx.load(str(int8_path))
    quantized = finalize(quantized, metadata)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_bytes_atomically(out_path, quantized.SerializeToString())
    return out_path


# ---------------------------------------------------------------------------
# 명령줄
# ---------------------------------------------------------------------------


def _check_config_matches_contract(export_cfg: ExportSettings) -> None:
    """설정 파일의 계약 값이 이 모듈의 상수와 같은지. 계약은 설정으로 바꿀 수 없다."""
    pairs = (
        ("opset", OPSET),
        ("ir_version", IR_VERSION),
        ("input_name", INPUT_NAME),
        ("output_name", OUTPUT_NAME),
        ("max_file_bytes", MAX_FILE_BYTES),
    )
    for name, fixed in pairs:
        value = getattr(export_cfg, name)
        if value != fixed:
            raise ValueError(f"config export.{name} is {value!r} but the ONNX contract fixes it at {fixed!r}")


def _spec_copy_sha256(spec_dict: Mapping[str, object]) -> str:
    """`export`가 모델 옆에 쓸 feature_spec 사본 바이트의 sha256(파일은 남기지 않는다).

    사본의 형식은 `drowsy.spec.write_feature_spec` 한 곳에만 있으므로, 임시 폴더에 그 함수로 써서 해시를 얻는다.
    """
    with tempfile.TemporaryDirectory() as work_dir:
        return write_feature_spec(spec_dict, Path(work_dir) / "feature_spec.json")


def _run_cli(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    _check_config_matches_contract(cfg.export)
    out_dir = args.out_dir if args.out_dir is not None else cfg.paths.export_dir

    if args.random_init:
        spec_path = args.spec if args.spec is not None else cfg.paths.feature_spec
        version = args.version if args.version is not None else cfg.export.model_version
        spec = load_feature_spec(spec_path)
        torch.manual_seed(args.seed)
        kind = args.kind
        net = build_model(kind, spec.n_features)
        temperature = spec.standardization.temperature
        trained_on = TRAINED_ON_NONE
        origin = f"random init, seed {args.seed}"
    else:
        # 학습한 모델은 설정의 기본값(config/feature_spec.json, export.model_version = 0)으로 내보내지 않는다.
        if args.spec is None:
            raise ValueError(
                "--checkpoint needs --spec: pass the feature spec copy written by the training run of this "
                f"checkpoint ({EXPORTED_SPEC_PATTERN.format(version='{N}')} next to the model that "
                "`python -m drowsy.train final` exported)"
            )
        if args.version is None:
            raise ValueError("--checkpoint needs --version (a trained model does not take export.model_version)")
        spec_path = args.spec
        version = args.version
        checkpoint = load_checkpoint(args.checkpoint)
        spec = load_feature_spec(spec_path)
        if checkpoint.n_features != spec.n_features:
            raise ValueError(
                f"checkpoint has {checkpoint.n_features} features but the feature spec has {spec.n_features}"
            )
        # 가정: spec은 받은 파일의 바이트가 아니라 `export`가 모델 옆에 쓸 사본(고정된 형식)의 바이트로 대조한다.
        # 체크포인트의 해시가 그 사본의 것이기 때문이다. 내용이 같고 들여쓰기만 다른 파일은 받는다.
        spec_hash = _spec_copy_sha256(read_feature_spec_dict(spec_path))
        if spec_hash != checkpoint.feature_spec_hash:
            raise ValueError(
                f"{spec_path} is not the feature spec this checkpoint was trained with: its exported copy would have "
                f"sha256 {spec_hash}, the checkpoint records {checkpoint.feature_spec_hash}. Pass the "
                f"{EXPORTED_SPEC_PATTERN.format(version='{N}')} that `python -m drowsy.train final` wrote next to "
                "the model (it holds the trained means, stds and temperature)"
            )
        # 가정: 체크포인트의 `synthetic`이 false여도 `trained_on`이 합성 표지로 시작하면 합성 모델로 본다.
        if checkpoint.synthetic or is_synthetic_trained_on(checkpoint.trained_on):
            refuse_synthetic_export(out_dir, cfg.paths.export_dir)
        # 축소 설정(연기 시험)으로 학습한 모델은 `trained_on`만으로는 구분되지 않는다. 체크포인트의 `fast`로 막는다.
        if checkpoint.fast:
            refuse_fast_export(out_dir, cfg.paths.export_dir)
        kind = checkpoint.kind
        net = checkpoint.net
        temperature = checkpoint.temperature
        trained_on = checkpoint.trained_on
        origin = f"checkpoint {args.checkpoint}"
    if version < 0:
        raise ValueError(f"--version must be a non-negative integer, got {version}")
    wrapper = ExportWrapper(net, spec, temperature)
    print(f"model: {kind} ({origin}), parameters {count_parameters(net)}")

    int8_path = out_dir / model_file_name(version, int8=True)
    had_int8 = int8_path.exists()
    onnx_path = export(
        wrapper,
        spec_path,
        out_dir / model_file_name(version),
        {"model_version": str(version), "trained_on": trained_on},
        dynamo=args.dynamo,
    )
    exported_spec = exported_spec_path(onnx_path)
    print(f"wrote {onnx_path} ({onnx_path.stat().st_size} bytes, sha256 {file_sha256(onnx_path)})")
    print(f"wrote {exported_spec} ({exported_spec.stat().st_size} bytes)")
    if had_int8:
        print(f"removed {int8_path} (it was made from the previous export of version {version})")
    print("metadata:")
    for key, value in read_metadata(onnx_path).items():
        print(f"  {key} = {value}")
    print(f"operators: {', '.join(graph_operators(onnx_path))}")

    verify_contract(onnx_path, exported_spec, ALLOWED_OPS_DYNAMO if args.dynamo else ALLOWED_OPS)
    verify_file_name(onnx_path)
    print("verify_contract: OK")

    inputs = random_raw_inputs(spec, cfg.export.parity_random_inputs, seed=0)
    diff = parity_max_abs_diff(wrapper, onnx_path, inputs)
    parity_ok = diff < cfg.export.parity_tolerance
    print(
        f"parity ({inputs.shape[0]} random raw inputs): max abs diff {diff:.3e} "
        f"(tolerance {cfg.export.parity_tolerance:.0e}) {'OK' if parity_ok else 'FAILED'}"
    )

    latency = measure_latency(onnx_path, cfg.export.latency_runs, cfg.export.intra_op_threads)
    print(
        f"latency ({cfg.export.latency_runs} runs, {cfg.export.intra_op_threads} intra-op thread(s)): "
        f"median {latency['median_ms']:.4f} ms, p95 {latency['p95_ms']:.4f} ms"
    )

    # INT8 파일은 기본 경로의 그래프로만 만든다(허용 연산자 목록을 그 그래프에서 얻었다).
    if args.checkpoint is not None and kind == KIND_TCN and cfg.export.int8_if_tcn and not args.dynamo:
        quantize_int8(onnx_path, int8_path)
        try:
            verify_contract(int8_path, exported_spec, ALLOWED_OPS_INT8)
            verify_file_name(int8_path)
        except ContractError:
            int8_path.unlink(missing_ok=True)  # 계약을 어긴 INT8 파일을 폴더에 남기지 않는다
            raise
        drift = parity_max_abs_diff(wrapper, int8_path, inputs)
        print(f"wrote {int8_path} ({int8_path.stat().st_size} bytes, sha256 {file_sha256(int8_path)})")
        print(f"int8 operators: {', '.join(graph_operators(int8_path))}")
        print(f"int8 verify_contract: OK, max abs diff vs PyTorch on the parity inputs {drift:.3e}")

    return 0 if parity_ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    """명령줄 진입점. 종료 코드: 0 정상, 1 일치 검증 실패, 2 오류.

    오류(틀린 인자 조합, 읽을 수 없는 파일, 맞지 않는 spec, 합성 모델이나 축소 설정 모델의 내보내기 위치, 계약 위반)는 예외 추적 대신
    표준 오류에 `error: ...` 한 줄로 쓰고 2를 돌려준다. argparse가 잡는 인자 오류도 종료 코드 2다.
    가정: 여러 줄로 된 예외 메시지는 줄바꿈과 연속 공백을 공백 하나로 바꿔 한 줄로 만든다.
    """
    parser = argparse.ArgumentParser(prog="python -m drowsy.export_onnx", description="export the ONNX model")
    parser.add_argument("--config", default="config/default.yaml")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--random-init", action="store_true", help="random weights with a fixed seed (v0)")
    source.add_argument("--checkpoint", type=Path, help="checkpoint written by drowsy.models.save_checkpoint")
    parser.add_argument("--kind", choices=KINDS, default=KIND_TCN, help="model kind for --random-init")
    parser.add_argument("--seed", type=int, default=0, help="seed for --random-init")
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="model version N (required with --checkpoint; default for --random-init: export.model_version)",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=None,
        help="feature spec file (required with --checkpoint: the feature_spec_v{N}.json written by the training "
        "run; default for --random-init: paths.feature_spec)",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="output folder (default: paths.export_dir)")
    parser.add_argument("--dynamo", action="store_true", help="use the dynamo exporter (fallback path)")
    args = parser.parse_args(argv)
    try:
        return _run_cli(args)
    except (ValueError, OSError) as error:  # SpecError, ContractError도 ValueError다
        print(f"error: {' '.join(str(error).split())}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
