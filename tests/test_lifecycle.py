from __future__ import annotations

import json
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from model_lifecycle.lifecycle import (
    ArtifactError,
    GateRejected,
    RegistryBusy,
    _exclusive_registry_lock,
    drift_diagnostic,
    load_active,
    promote,
    read_registry,
    rollback,
    train_candidate,
)
from model_lifecycle.receipt import atomic_write_json


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


def test_release_gate_uses_validation_and_not_perfect_heldout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import model_lifecycle.lifecycle as lifecycle

    bad_validation = {
        "count": 36,
        "accuracy": 0.0,
        "balanced_accuracy": 0.0,
        "per_class_recall": [0.0, 0.0, 0.0],
        "confusion_matrix": [[0, 0, 0]] * 3,
    }
    baseline_validation = {
        "count": 36,
        "accuracy": 0.4,
        "balanced_accuracy": 0.3,
        "per_class_recall": [0.0, 1.0, 0.0],
        "confusion_matrix": [[0, 0, 0]] * 3,
    }
    perfect_heldout = {
        "count": 36,
        "accuracy": 1.0,
        "balanced_accuracy": 1.0,
        "per_class_recall": [1.0, 1.0, 1.0],
        "confusion_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }
    metric_results = iter(
        [bad_validation, baseline_validation, perfect_heldout, baseline_validation]
    )
    monkeypatch.setattr(lifecycle, "_metrics", lambda *_args: next(metric_results))
    candidate = train_candidate(tmp_path / "registry", "gate-partition-v1")
    assert candidate["release_gate"] == {
        "minimum_balanced_accuracy_gain": 0.1,
        "minimum_per_class_recall": 0.6,
        "partition": "tuning_validation",
        "passed": False,
    }
    assert candidate["evaluation"]["heldout_test"]["balanced_accuracy"] == 1.0


def test_duplicate_version_rejects_before_dataset_or_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import model_lifecycle.lifecycle as lifecycle

    registry = tmp_path / "registry"
    (registry / "versions" / "duplicate-v1").mkdir(parents=True)

    def unexpected_dataset(_seed: int):
        raise AssertionError("duplicate version reached dataset loading")

    monkeypatch.setattr(lifecycle, "load_dataset", unexpected_dataset)
    with pytest.raises(FileExistsError):
        train_candidate(registry, "duplicate-v1")


def test_sigkill_before_atomic_replace_preserves_registry_and_rollback(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    train_candidate(registry, "known-v1")
    promote(registry, "known-v1")
    train_candidate(registry, "current-v2", seed=20260908)
    promote(registry, "current-v2")
    original = (registry / "registry.json").read_bytes()
    payload = json.loads(original)
    payload["active_version"] = "known-v1"
    script = """
import json, os, signal, sys
from pathlib import Path
from model_lifecycle.receipt import atomic_write_json
atomic_write_json(
    Path(sys.argv[1]),
    json.loads(sys.argv[2]),
    before_replace=lambda: os.kill(os.getpid(), signal.SIGKILL),
)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(registry / "registry.json"), json.dumps(payload)],
        check=False,
    )
    assert result.returncode == -signal.SIGKILL
    assert (registry / "registry.json").read_bytes() == original
    assert load_active(registry).metadata["version"] == "current-v2"
    assert rollback(registry, "known-v1")["active_version"] == "known-v1"


def test_rollback_rejects_substituted_approved_files(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    train_candidate(registry, "known-v1")
    promote(registry, "known-v1")
    train_candidate(registry, "current-v2", seed=20260908)
    promote(registry, "current-v2")
    train_candidate(registry, "rejected-dummy", model_kind="dummy")

    known = registry / "versions" / "known-v1"
    rejected = registry / "versions" / "rejected-dummy"
    shutil.copy2(rejected / "model.skops", known / "model.skops")
    shutil.copy2(rejected / "dataset-manifest.json", known / "dataset-manifest.json")
    replacement_metadata = json.loads((rejected / "metadata.json").read_text())
    replacement_metadata["version"] = "known-v1"
    replacement_metadata["artifact_sha256"] = lifecycle_sha256(known / "model.skops")
    replacement_metadata["dataset_manifest_sha256"] = lifecycle_sha256(
        known / "dataset-manifest.json"
    )
    atomic_write_json(known / "metadata.json", replacement_metadata)

    with pytest.raises(ArtifactError, match="approved model identity"):
        rollback(registry, "known-v1")
    assert load_active(registry).metadata["version"] == "current-v2"
    registry_payload = read_registry(registry)
    registry_payload["active_version"] = "known-v1"
    atomic_write_json(registry / "registry.json", registry_payload)
    with pytest.raises(ArtifactError, match="approved model identity"):
        load_active(registry)


def lifecycle_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_concurrent_mutation_fails_fast_before_training(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    with _exclusive_registry_lock(registry), pytest.raises(RegistryBusy):
        train_candidate(registry, "blocked-v1")
    assert not (registry / "versions" / "blocked-v1").exists()
