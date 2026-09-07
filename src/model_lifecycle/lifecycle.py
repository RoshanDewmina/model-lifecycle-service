from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import skops.io as sio
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import load_dataset, manifest
from .features import FEATURE_NAMES
from .receipt import atomic_write_json, git_state, receipt_base, write_json

CLASS_NAMES = ("cultivar_1", "cultivar_2", "cultivar_3")
GATE = {"minimum_balanced_accuracy_gain": 0.10, "minimum_per_class_recall": 0.60}
VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
IDENTITY_KEYS = ("artifact_sha256", "metadata_sha256", "dataset_manifest_sha256")


class GateRejected(RuntimeError):
    pass


class ArtifactError(RuntimeError):
    pass


class RegistryBusy(RuntimeError):
    pass


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    metadata: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_version(version: str) -> None:
    if not VERSION_PATTERN.fullmatch(version):
        raise ValueError("version must use 1-64 letters, digits, dots, underscores, or hyphens")


def _registry_file(registry: Path) -> Path:
    return registry / "registry.json"


@contextmanager
def _exclusive_registry_lock(registry: Path):
    registry.mkdir(parents=True, exist_ok=True)
    lock_path = registry / ".registry.lock"
    with lock_path.open("a+b") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RegistryBusy("another lifecycle mutation holds the registry lock") from exc
        try:
            yield
        finally:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)


def read_registry(registry: Path) -> dict[str, Any]:
    path = _registry_file(registry)
    if not path.exists():
        return {"schema_version": 1, "active_version": None, "versions": {}}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError("registry is unreadable or invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ArtifactError("registry must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ArtifactError("unsupported registry schema")
    active = payload.get("active_version")
    versions = payload.get("versions")
    if active is not None and not isinstance(active, str):
        raise ArtifactError("registry active_version must be a string or null")
    if not isinstance(versions, dict) or not all(
        isinstance(key, str) and isinstance(value, dict) for key, value in versions.items()
    ):
        raise ArtifactError("registry versions must map strings to objects")
    return payload


def _write_registry(registry: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(_registry_file(registry), payload)


def _metrics(model: Any, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    predicted = model.predict(X)
    return {
        "count": int(len(y)),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "per_class_recall": [float(x) for x in recall_score(y, predicted, average=None)],
        "confusion_matrix": confusion_matrix(y, predicted).astype(int).tolist(),
    }


def _identity(version_dir: Path) -> dict[str, str]:
    return {
        "artifact_sha256": _sha256(version_dir / "model.skops"),
        "metadata_sha256": _sha256(version_dir / "metadata.json"),
        "dataset_manifest_sha256": _sha256(version_dir / "dataset-manifest.json"),
    }


def _read_metadata(version_dir: Path, version: str) -> dict[str, Any]:
    metadata_path = version_dir / "metadata.json"
    if not metadata_path.is_file():
        raise ArtifactError(f"metadata missing for version {version}")
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"metadata is unreadable for version {version}") from exc
    if not isinstance(metadata, dict):
        raise ArtifactError("model metadata must be a JSON object")
    required_objects = ("evaluation", "release_gate", "training_stats")
    if any(not isinstance(metadata.get(name), dict) for name in required_objects):
        raise ArtifactError("model metadata object fields are malformed")
    if (
        metadata.get("version") != version
        or metadata.get("feature_names") != list(FEATURE_NAMES)
        or metadata.get("class_names") != list(CLASS_NAMES)
        or metadata.get("artifact") != "model.skops"
        or metadata.get("dataset_manifest") != "dataset-manifest.json"
    ):
        raise ArtifactError("model metadata does not match the serving contract")
    for name in ("artifact_sha256", "dataset_manifest_sha256"):
        value = metadata.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise ArtifactError(f"model metadata {name} is malformed")
    return metadata


def train_candidate(
    registry: Path,
    version: str,
    *,
    seed: int = 20260907,
    model_kind: str = "logistic",
    receipt_path: Path | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    _validate_version(version)
    started = time.perf_counter()
    revision, dirty = git_state()
    versions_dir = registry / "versions"
    version_dir = versions_dir / version

    with _exclusive_registry_lock(registry):
        if version_dir.exists():
            raise FileExistsError(f"immutable version already exists: {version}")
        versions_dir.mkdir(parents=True, exist_ok=True)
        staging_dir: Path | None = Path(
            tempfile.mkdtemp(prefix=f".staging-{version}-", dir=versions_dir)
        )
        try:
            split = load_dataset(seed)
            train_X, train_y = split.X[split.train_idx], split.y[split.train_idx]

            baseline = DummyClassifier(strategy="most_frequent")
            baseline.fit(train_X, train_y)
            if model_kind == "logistic":
                candidate: Any = Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "classifier",
                            LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
                        ),
                    ]
                )
            elif model_kind == "dummy":
                candidate = DummyClassifier(strategy="most_frequent")
            else:
                raise ValueError(f"unsupported model kind: {model_kind}")
            candidate.fit(train_X, train_y)

            validation_metrics = _metrics(
                candidate, split.X[split.validation_idx], split.y[split.validation_idx]
            )
            baseline_validation_metrics = _metrics(
                baseline, split.X[split.validation_idx], split.y[split.validation_idx]
            )
            validation_gain = (
                validation_metrics["balanced_accuracy"]
                - baseline_validation_metrics["balanced_accuracy"]
            )
            gate_passed = validation_gain >= GATE["minimum_balanced_accuracy_gain"] and min(
                validation_metrics["per_class_recall"]
            ) >= GATE["minimum_per_class_recall"]

            # Candidate configuration and gate are fixed before informational test evaluation.
            heldout_metrics = _metrics(
                candidate, split.X[split.heldout_idx], split.y[split.heldout_idx]
            )
            baseline_heldout_metrics = _metrics(
                baseline, split.X[split.heldout_idx], split.y[split.heldout_idx]
            )

            artifact_path = staging_dir / "model.skops"
            sio.dump(candidate, artifact_path)
            unknown_types = sio.get_untrusted_types(file=artifact_path)
            if unknown_types:
                raise ArtifactError(
                    f"artifact contains types outside the skops trusted set: {unknown_types}"
                )

            dataset_manifest = manifest(split, seed)
            write_json(staging_dir / "dataset-manifest.json", dataset_manifest)
            training_stats = {
                name: {"mean": float(train_X[:, i].mean()), "std": float(train_X[:, i].std())}
                for i, name in enumerate(FEATURE_NAMES)
            }
            metadata = {
                "schema_version": 2,
                "version": version,
                "model_kind": model_kind,
                "source_revision": revision,
                "source_dirty_tree": dirty,
                "feature_names": list(FEATURE_NAMES),
                "class_names": list(CLASS_NAMES),
                "artifact": "model.skops",
                "artifact_sha256": _sha256(artifact_path),
                "dataset_manifest": "dataset-manifest.json",
                "dataset_manifest_sha256": _sha256(staging_dir / "dataset-manifest.json"),
                "training_stats": training_stats,
                "evaluation": {
                    "tuning_validation": validation_metrics,
                    "dummy_baseline_tuning_validation": baseline_validation_metrics,
                    "heldout_test": heldout_metrics,
                    "dummy_baseline_heldout_test": baseline_heldout_metrics,
                    "validation_balanced_accuracy_gain_over_dummy": validation_gain,
                },
                "release_gate": {
                    **GATE,
                    "partition": "tuning_validation",
                    "passed": gate_passed,
                },
                "limitations": dataset_manifest["limitations"],
            }
            write_json(staging_dir / "metadata.json", metadata)
            os.replace(staging_dir, version_dir)
            staging_dir = None
        finally:
            if staging_dir is not None and staging_dir.exists():
                shutil.rmtree(staging_dir)

    if receipt_path:
        receipt = receipt_base(command or f"modelctl train --version {version}", revision, dirty)
        receipt.update(
            {
                "exit_status": 0,
                "inputs": {
                    "dataset_id": dataset_manifest["dataset_id"],
                    "counts": dataset_manifest["counts"],
                    "seed": seed,
                    "model_kind": model_kind,
                },
                "measured_results": {
                    "duration_seconds": time.perf_counter() - started,
                    **metadata["evaluation"],
                    "candidate_gate_passed": gate_passed,
                    "gate_partition": "tuning_validation",
                    "artifact_sha256": metadata["artifact_sha256"],
                },
                "limitations": metadata["limitations"],
            }
        )
        write_json(receipt_path, receipt)
    return metadata


def promote(registry: Path, version: str) -> dict[str, Any]:
    _validate_version(version)
    with _exclusive_registry_lock(registry):
        loaded = load_version(registry, version)
        metadata = loaded.metadata
        if not metadata["release_gate"].get("passed"):
            raise GateRejected(f"candidate {version} failed the release gate")
        version_dir = registry / "versions" / version
        identity = _identity(version_dir)
        payload = read_registry(registry)
        previous_entry = payload["versions"].get(version)
        if previous_entry is not None:
            saved = {key: previous_entry.get(key) for key in IDENTITY_KEYS}
            if saved != identity:
                raise GateRejected(f"approved identity changed for version {version}")
        previous = payload["active_version"]
        if previous == version:
            if previous_entry is None:
                raise ArtifactError("active version is missing its approved registry identity")
            if previous_entry.get("status") != "active":
                previous_entry["status"] = "active"
                _write_registry(registry, payload)
            return payload
        payload["versions"][version] = {
            "status": "active",
            "previous_active_version": previous,
            **identity,
        }
        if previous and previous in payload["versions"]:
            payload["versions"][previous]["status"] = "inactive"
        payload["active_version"] = version
        _write_registry(registry, payload)
        return payload


def rollback(registry: Path, version: str) -> dict[str, Any]:
    _validate_version(version)
    with _exclusive_registry_lock(registry):
        payload = read_registry(registry)
        entry = payload["versions"].get(version)
        if not isinstance(entry, dict):
            raise GateRejected(f"rollback target was never promoted: {version}")
        load_version(registry, version, expected_identity=entry)
        current = payload["active_version"]
        if current and current in payload["versions"]:
            payload["versions"][current]["status"] = "inactive"
        payload["versions"][version]["status"] = "active"
        payload["versions"][version]["rollback_from"] = current
        payload["active_version"] = version
        _write_registry(registry, payload)
        return payload


def load_version(
    registry: Path, version: str, *, expected_identity: dict[str, Any] | None = None
) -> LoadedModel:
    _validate_version(version)
    version_dir = registry / "versions" / version
    metadata = _read_metadata(version_dir, version)
    artifact_path = version_dir / "model.skops"
    if not artifact_path.is_file() or _sha256(artifact_path) != metadata["artifact_sha256"]:
        raise ArtifactError("model artifact hash mismatch")
    manifest_path = version_dir / "dataset-manifest.json"
    if (
        not manifest_path.is_file()
        or _sha256(manifest_path) != metadata["dataset_manifest_sha256"]
    ):
        raise ArtifactError("dataset manifest hash mismatch")
    if expected_identity is not None:
        saved = {key: expected_identity.get(key) for key in IDENTITY_KEYS}
        if not all(isinstance(saved[key], str) and len(saved[key]) == 64 for key in IDENTITY_KEYS):
            raise ArtifactError("approved registry identity is incomplete")
        if saved != _identity(version_dir):
            raise ArtifactError("approved model identity does not match stored files")
    try:
        unknown_types = sio.get_untrusted_types(file=artifact_path)
        if unknown_types:
            raise ArtifactError(f"refusing artifact with untrusted types: {unknown_types}")
        model = sio.load(artifact_path, trusted=[])
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactError("model artifact could not be safely loaded") from exc
    if int(getattr(model, "n_features_in_", -1)) != len(FEATURE_NAMES):
        raise ArtifactError("loaded model feature count does not match serving schema")
    return LoadedModel(model, metadata)


def load_active(registry: Path) -> LoadedModel:
    payload = read_registry(registry)
    active = payload["active_version"]
    if not active:
        raise ArtifactError("no active model")
    entry = payload["versions"].get(active)
    if not isinstance(entry, dict):
        raise ArtifactError("active model is missing from registry versions")
    return load_version(registry, active, expected_identity=entry)


def drift_diagnostic(registry: Path, *, simulate: bool, seed: int = 20260907) -> dict[str, Any]:
    if not simulate:
        raise ValueError("only the explicitly simulated local diagnostic is implemented")
    loaded = load_active(registry)
    rng = np.random.default_rng(seed)
    stats = loaded.metadata["training_stats"]
    observed: dict[str, float] = {}
    for index, name in enumerate(FEATURE_NAMES):
        mean = stats[name]["mean"]
        std = stats[name]["std"] or 1.0
        shift = 0.9 if index in (0, 9) else 0.0
        batch = rng.normal(mean + shift * std, std, size=200)
        observed[name] = abs(float(batch.mean()) - mean) / std
    threshold = 0.5
    return {
        "diagnostic_type": "SIMULATED_DRIFT_DIAGNOSTIC",
        "is_real_monitoring": False,
        "sample_count": 200,
        "threshold_standardized_mean_shift": threshold,
        "feature_shifts": observed,
        "flagged_features": sorted(name for name, value in observed.items() if value >= threshold),
        "limitations": "Generated perturbations only; this is not evidence of real-world drift.",
    }
