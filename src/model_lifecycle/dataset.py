from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from .features import FEATURE_NAMES

DATASET_ID = "uci-wine-sklearn-copy"
DATASET_VERSION = "1"
DATASET_DOI = "10.24432/C5PC7J"
DATASET_URL = "https://archive.ics.uci.edu/dataset/109/wine"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"


@dataclass(frozen=True)
class DatasetSplit:
    X: np.ndarray
    y: np.ndarray
    sample_ids: np.ndarray
    train_idx: np.ndarray
    validation_idx: np.ndarray
    heldout_idx: np.ndarray


def _sample_id(row: np.ndarray, target: int) -> str:
    payload = json.dumps(
        {"features": [float(x) for x in row], "target": int(target)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def load_dataset(seed: int) -> DatasetSplit:
    bundle = load_wine()
    normalized_names = tuple(name.replace("/", "_") for name in bundle.feature_names)
    if normalized_names != FEATURE_NAMES:
        raise RuntimeError("scikit-learn Wine feature schema changed")
    X = np.asarray(bundle.data, dtype=np.float64)
    y = np.asarray(bundle.target, dtype=np.int64)
    sample_ids = np.asarray([_sample_id(row, target) for row, target in zip(X, y, strict=True)])

    indices = np.arange(len(y))
    train_idx, remainder_idx = train_test_split(
        indices, test_size=0.40, stratify=y, random_state=seed
    )
    validation_idx, heldout_idx = train_test_split(
        remainder_idx, test_size=0.50, stratify=y[remainder_idx], random_state=seed
    )
    return DatasetSplit(X, y, sample_ids, train_idx, validation_idx, heldout_idx)


def dataset_payload_hash(split: DatasetSplit) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(split.X).tobytes())
    digest.update(np.ascontiguousarray(split.y).tobytes())
    return digest.hexdigest()


def manifest(split: DatasetSplit, seed: int) -> dict[str, Any]:
    def ids(indices: np.ndarray) -> list[str]:
        return sorted(str(split.sample_ids[i]) for i in indices)

    return {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "version": DATASET_VERSION,
        "title": "Wine",
        "doi": DATASET_DOI,
        "license": "CC BY 4.0",
        "source_urls": [DATASET_URL, LICENSE_URL],
        "individual_file_hashes": {"sklearn_embedded_numeric_payload": dataset_payload_hash(split)},
        "split_method": "two-stage stratified random split: 60% train, 20% validation, 20% heldout",
        "seed": seed,
        "identities": {
            "train": ids(split.train_idx),
            "tuning_validation": ids(split.validation_idx),
            "heldout_test": ids(split.heldout_idx),
        },
        "counts": {
            "total": len(split.y),
            "train": len(split.train_idx),
            "tuning_validation": len(split.validation_idx),
            "heldout_test": len(split.heldout_idx),
        },
        "limitations": [
            "Small, decades-old teaching dataset; results do not establish production performance.",
            "Cultivar prediction from laboratory measurements is only a local lifecycle "
            "demonstration.",
            "The held-out split is evaluated once per candidate run and is not used for "
            "fitting or tuning.",
        ],
    }
