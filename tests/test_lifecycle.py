from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from model_lifecycle.lifecycle import (
    ArtifactError,
    GateRejected,
    drift_diagnostic,
    load_active,
    promote,
    rollback,
    train_candidate,
)


def test_candidate_gate_rejection_and_known_model_rollback(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    accepted = train_candidate(registry, "good-v1")
    assert accepted["release_gate"]["passed"] is True
    promote(registry, "good-v1")

    rejected = train_candidate(registry, "bad-v1", model_kind="dummy")
    assert rejected["release_gate"]["passed"] is False
    with pytest.raises(GateRejected):
        promote(registry, "bad-v1")
    assert load_active(registry).metadata["version"] == "good-v1"

    train_candidate(registry, "good-v2", seed=20260908)
    promote(registry, "good-v2")
    assert load_active(registry).metadata["version"] == "good-v2"
    state = rollback(registry, "good-v1")
    assert state["active_version"] == "good-v1"
    assert load_active(registry).metadata["version"] == "good-v1"


def test_artifact_hash_is_verified_before_load(trained_registry: Path, tmp_path: Path) -> None:
    copied_registry = tmp_path / "registry"
    shutil.copytree(trained_registry, copied_registry)
    artifact = copied_registry / "versions" / "test-v1" / "model.skops"
    artifact.write_bytes(artifact.read_bytes() + b"tamper")
    with pytest.raises(ArtifactError, match="hash mismatch"):
        load_active(copied_registry)


def test_simulated_drift_is_explicitly_labeled(trained_registry: Path) -> None:
    result = drift_diagnostic(trained_registry, simulate=True)
    assert result["diagnostic_type"] == "SIMULATED_DRIFT_DIAGNOSTIC"
    assert result["is_real_monitoring"] is False
    assert result["flagged_features"]
