from __future__ import annotations

from pathlib import Path

import pytest

from model_lifecycle.lifecycle import promote, train_candidate


@pytest.fixture(scope="session")
def trained_registry(tmp_path_factory: pytest.TempPathFactory) -> Path:
    registry = tmp_path_factory.mktemp("registry")
    train_candidate(registry, "test-v1")
    promote(registry, "test-v1")
    return registry


@pytest.fixture
def valid_payload() -> dict[str, float]:
    return {
        "alcohol": 13.2,
        "malic_acid": 1.78,
        "ash": 2.14,
        "alcalinity_of_ash": 11.2,
        "magnesium": 100.0,
        "total_phenols": 2.65,
        "flavanoids": 2.76,
        "nonflavanoid_phenols": 0.26,
        "proanthocyanins": 1.28,
        "color_intensity": 4.38,
        "hue": 1.05,
        "od280_od315_of_diluted_wines": 3.4,
        "proline": 1050.0,
    }

