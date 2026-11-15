"""이식 대상 모듈이 표준 라이브러리만 쓰는지 검사한다 (PLAN 3.1절, 10.1절).

Kotlin으로 옮길 모듈(types, signals, calibrate, blink, step_features, rules, pipeline)은 numpy, pandas,
yaml, torch 없이 동작해야 한다. 다른 테스트가 이미 올려 둔 모듈에 속지 않도록 새 인터프리터에서 import한다.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from drowsy.spec import RulesConfig, export_rules_json

PORTED_MODULES: tuple[str, ...] = (
    "drowsy.types",
    "drowsy.signals",
    "drowsy.calibrate",
    "drowsy.blink",
    "drowsy.step_features",
    "drowsy.rules",
    "drowsy.pipeline",
)
FORBIDDEN: tuple[str, ...] = ("numpy", "pandas", "yaml", "torch")

# 이식 대상 모듈이 직접 import해도 되는 최상위 이름
ALLOWED_IMPORTS: frozenset[str] = frozenset({"__future__", "math", "collections", "dataclasses", "drowsy"})
# PLAN 3.1절이 허용한 수학 함수(와 각도 변환용 상수 pi)
ALLOWED_MATH: frozenset[str] = frozenset({"sqrt", "atan2", "hypot", "sin", "cos", "floor", "ceil", "pi"})


def _loaded_modules_after_import(repo_root: Path, modules: tuple[str, ...]) -> list[str]:
    """새 인터프리터에서 modules를 import한 뒤의 `sys.modules` 이름 목록."""
    code = (
        "import importlib, json, sys\n"
        f"for name in {list(modules)!r}:\n"
        "    importlib.import_module(name)\n"
        "print(json.dumps(sorted(sys.modules)))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    assert isinstance(loaded, list)
    return loaded


def _forbidden_in(loaded: list[str]) -> list[str]:
    return [name for name in loaded if name.split(".")[0] in FORBIDDEN]


def test_importing_all_ported_modules_loads_no_forbidden_package(repo_root: Path) -> None:
    loaded = _loaded_modules_after_import(repo_root, PORTED_MODULES)
    for module in PORTED_MODULES:
        assert module in loaded
    assert _forbidden_in(loaded) == []


@pytest.mark.parametrize("module", PORTED_MODULES)
def test_each_ported_module_is_stdlib_only_on_its_own(repo_root: Path, module: str) -> None:
    loaded = _loaded_modules_after_import(repo_root, (module,))
    assert module in loaded
    assert _forbidden_in(loaded) == []


def test_the_check_detects_a_forbidden_import(repo_root: Path) -> None:
    # 검사 자체가 동작하는지: 설정 로더는 yaml을 쓰므로 걸려야 한다.
    loaded = _loaded_modules_after_import(repo_root, ("drowsy.config",))
    assert "yaml" in _forbidden_in(loaded)


def test_pipeline_runs_without_forbidden_packages(repo_root: Path, rules_config: RulesConfig, tmp_path: Path) -> None:
    # import뿐 아니라 실제로 프레임을 처리한 뒤에도 올라오지 않아야 한다(지연 import가 없는지).
    # 규칙은 앱처럼 rules.json으로 읽는다(YAML 로더는 yaml을 쓴다).
    rules_json = tmp_path / "rules.json"
    export_rules_json(rules_config, rules_json)
    code = (
        "import json, sys\n"
        "from drowsy import calibrate\n"
        "from drowsy.pipeline import DriverPipeline\n"
        "from drowsy.spec import load_feature_spec, load_rules\n"
        "from drowsy.types import RawFrame\n"
        "spec = load_feature_spec('config/feature_spec.json').pipeline\n"
        f"rules = load_rules({str(rules_json)!r})\n"
        "lm = {i: (0.5, 0.5) for i in spec.landmarks.required_indices()}\n"
        "bs = {name: 0.0 for name in spec.blendshapes.required_names()}\n"
        "m = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)\n"
        "pipe = DriverPipeline(spec, rules, baseline=calibrate.global_baseline(spec), feature_start_ms=0)\n"
        "n_steps = 0\n"
        "for i in range(40):\n"
        "    if i % 4 == 3:\n"
        "        frame = RawFrame(1 + 100 * i, False, 640, 480, {}, {}, ())\n"
        "    else:\n"
        "        frame = RawFrame(1 + 100 * i, True, 640, 480, lm, bs, m)\n"
        "    n_steps += len(pipe.update(frame, 30.0).steps)\n"
        "assert n_steps == 3, n_steps\n"
        "print(json.dumps(sorted(sys.modules)))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=repo_root, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    assert "drowsy.pipeline" in loaded
    assert _forbidden_in(loaded) == []


@pytest.mark.parametrize("module", PORTED_MODULES)
def test_ported_module_source_uses_only_allowed_imports_and_math(repo_root: Path, module: str) -> None:
    path = repo_root / (module.replace(".", "/") + ".py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in ALLOWED_IMPORTS, f"{module} imports {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{module} uses a relative import"
            assert node.module is not None
            assert node.module.split(".")[0] in ALLOWED_IMPORTS, f"{module} imports from {node.module}"
            if node.module == "math":
                for alias in node.names:
                    assert alias.name in ALLOWED_MATH, f"{module} imports math.{alias.name}"
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "math":
                assert node.attr in ALLOWED_MATH, f"{module} uses math.{node.attr}"
