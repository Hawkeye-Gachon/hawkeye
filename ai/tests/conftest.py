"""공용 픽스처. 다른 테스트 파일도 같은 이름으로 받아 쓴다.

모두 세션 범위다. 돌려주는 값은 불변 dataclass이므로 테스트끼리 공유해도 된다.
값을 바꿔 보려면 `dataclasses.replace`나 `drowsy.spec.apply_overrides`로 사본을 만든다.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from drowsy.spec import FeatureSpec, PipelineSpec, RulesConfig, load_feature_spec, load_rules

if TYPE_CHECKING:
    from drowsy.config import ExperimentConfig


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """저장소 루트."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def feature_spec_path(repo_root: Path) -> Path:
    return repo_root / "config" / "feature_spec.json"


@pytest.fixture(scope="session")
def rules_path(repo_root: Path) -> Path:
    return repo_root / "config" / "rules.yaml"


@pytest.fixture(scope="session")
def config_path(repo_root: Path) -> Path:
    return repo_root / "config" / "default.yaml"


@pytest.fixture(scope="session")
def feature_spec(feature_spec_path: Path) -> FeatureSpec:
    """config/feature_spec.json 전체."""
    return load_feature_spec(feature_spec_path)


@pytest.fixture(scope="session")
def pipeline_spec(feature_spec: FeatureSpec) -> PipelineSpec:
    """이식 코드가 받는 상수(`pipeline` 블록)."""
    return feature_spec.pipeline


@pytest.fixture(scope="session")
def rules_config(rules_path: Path) -> RulesConfig:
    """config/rules.yaml."""
    return load_rules(rules_path)


@pytest.fixture(scope="session")
def experiment_config(config_path: Path) -> ExperimentConfig:
    """config/default.yaml."""
    # 이식 대상 모듈만 시험하는 실행에서 설정 모듈까지 올리지 않도록 여기서 import한다.
    from drowsy.config import load_config

    return load_config(config_path)
