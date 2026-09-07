from __future__ import annotations

import hashlib
import json
import shutil
import time
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
from .receipt import git_state, receipt_base, write_json

CLASS_NAMES = ("cultivar_1", "cultivar_2", "cultivar_3")
GATE = {"minimum_balanced_accuracy_gain": 0.10, "minimum_per_class_recall": 0.60}


class GateRejected(RuntimeError):
    pass


class ArtifactError(RuntimeError):
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


def _registry_file(registry: Path) -> Path:
    return registry / "registry.json"


def read_registry(registry: Path) -> dict[str, Any]:
    path = _registry_file(registry)
    if not path.exists():
        return {"schema_version": 1, "active_version": None, "versions": {}}
    return json.loads(path.read_text())


def _write_registry(registry: Path, payload: dict[str, Any]) -> None:
    write_json(_registry_file(registry), payload)


def _metrics(model: Any, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    predicted = model.predict(X)
    return {
        "count": int(len(y)),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "per_class_recall": [float(x) for x in recall_score(y, predicted, average=None)],
        "confusion_matrix": confusion_matrix(y, predicted).astype(int).tolist(),
    }


def train_candidate(
    registry: Path,
    version: str,
    *,
    seed: int = 20260907,
    model_kind: str = "logistic",
    receipt_path: Path | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    revision, dirty = git_state()
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
    heldout_metrics = _metrics(candidate, split.X[split.heldout_idx], split.y[split.heldout_idx])
    baseline_metrics = _metrics(baseline, split.X[split.heldout_idx], split.y[split.heldout_idx])
    gain = heldout_metrics["balanced_accuracy"] - baseline_metrics["balanced_accuracy"]
    gate_passed = gain >= GATE["minimum_balanced_accuracy_gain"] and min(
        heldout_metrics["per_class_recall"]
    ) >= GATE["minimum_per_class_recall"]

    version_dir = registry / "versions" / version
    if version_dir.exists():
        raise FileExistsError(f"immutable version already exists: {version}")
    version_dir.mkdir(parents=True)
    artifact_path = version_dir / "model.skops"
    sio.dump(candidate, artifact_path)
    unknown_types = sio.get_untrusted_types(file=artifact_path)
    if unknown_types:
        shutil.rmtree(version_dir)
        raise ArtifactError(
            f"artifact contains types outside the skops trusted set: {unknown_types}"
        )

    dataset_manifest = manifest(split, seed)
    write_json(version_dir / "dataset-manifest.json", dataset_manifest)
    training_stats = {
        name: {"mean": float(train_X[:, i].mean()), "std": float(train_X[:, i].std())}
        for i, name in enumerate(FEATURE_NAMES)
    }
    metadata = {
        "schema_version": 1,
        "version": version,
        "model_kind": model_kind,
        "source_revision": revision,
        "source_dirty_tree": dirty,
        "feature_names": list(FEATURE_NAMES),
        "class_names": list(CLASS_NAMES),
        "artifact": "model.skops",
        "artifact_sha256": _sha256(artifact_path),
        "dataset_manifest": "dataset-manifest.json",
        "dataset_manifest_sha256": _sha256(version_dir / "dataset-manifest.json"),
        "training_stats": training_stats,
        "evaluation": {
            "validation": validation_metrics,
            "heldout_test": heldout_metrics,
            "dummy_baseline_heldout_test": baseline_metrics,
            "balanced_accuracy_gain_over_dummy": gain,
        },
        "release_gate": {**GATE, "passed": gate_passed},
        "limitations": dataset_manifest["limitations"],
    }
    write_json(version_dir / "metadata.json", metadata)

    if receipt_path:
        receipt = receipt_base(
            command or f"modelctl train --version {version}", revision, dirty
        )
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
                    "artifact_sha256": metadata["artifact_sha256"],
                },
                "limitations": metadata["limitations"],
            }
        )
        write_json(receipt_path, receipt)
    return metadata


def promote(registry: Path, version: str) -> dict[str, Any]:
    version_dir = registry / "versions" / version
    metadata_path = version_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"candidate does not exist: {version}")
    metadata = json.loads(metadata_path.read_text())
    if not metadata["release_gate"]["passed"]:
        raise GateRejected(f"candidate {version} failed the release gate")
    load_version(registry, version)
    payload = read_registry(registry)
    previous = payload["active_version"]
    payload["versions"][version] = {
        "status": "active",
        "previous_active_version": previous,
        "artifact_sha256": metadata["artifact_sha256"],
    }
    if previous and previous in payload["versions"]:
        payload["versions"][previous]["status"] = "inactive"
    payload["active_version"] = version
    _write_registry(registry, payload)
    return payload


def rollback(registry: Path, version: str) -> dict[str, Any]:
    payload = read_registry(registry)
    if version not in payload["versions"]:
        raise GateRejected(f"rollback target was never promoted: {version}")
    load_version(registry, version)
    current = payload["active_version"]
    if current and current in payload["versions"]:
        payload["versions"][current]["status"] = "inactive"
    payload["versions"][version]["status"] = "active"
    payload["versions"][version]["rollback_from"] = current
    payload["active_version"] = version
    _write_registry(registry, payload)
    return payload


def load_version(registry: Path, version: str) -> LoadedModel:
    version_dir = registry / "versions" / version
    metadata_path = version_dir / "metadata.json"
    if not metadata_path.exists():
        raise ArtifactError(f"metadata missing for version {version}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("version") != version or metadata.get("feature_names") != list(FEATURE_NAMES):
        raise ArtifactError("model metadata does not match the serving contract")
    artifact_path = version_dir / metadata["artifact"]
    if not artifact_path.is_file() or _sha256(artifact_path) != metadata["artifact_sha256"]:
        raise ArtifactError("model artifact hash mismatch")
    manifest_path = version_dir / metadata["dataset_manifest"]
    if not manifest_path.is_file() or _sha256(manifest_path) != metadata["dataset_manifest_sha256"]:
        raise ArtifactError("dataset manifest hash mismatch")
    unknown_types = sio.get_untrusted_types(file=artifact_path)
    if unknown_types:
        raise ArtifactError(f"refusing artifact with untrusted types: {unknown_types}")
    model = sio.load(artifact_path, trusted=[])
    if int(getattr(model, "n_features_in_", -1)) != len(FEATURE_NAMES):
        raise ArtifactError("loaded model feature count does not match serving schema")
    return LoadedModel(model, metadata)


def load_active(registry: Path) -> LoadedModel:
    payload = read_registry(registry)
    if not payload["active_version"]:
        raise ArtifactError("no active model")
    return load_version(registry, payload["active_version"])


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
