from __future__ import annotations

from model_lifecycle.dataset import load_dataset, manifest


def test_split_identities_are_frozen_disjoint_and_reproducible() -> None:
    first = manifest(load_dataset(20260907), 20260907)
    second = manifest(load_dataset(20260907), 20260907)
    assert first["identities"] == second["identities"]
    train = set(first["identities"]["train"])
    validation = set(first["identities"]["tuning_validation"])
    heldout = set(first["identities"]["heldout_test"])
    assert len(train | validation | heldout) == 178
    assert not (train & validation or train & heldout or validation & heldout)
    assert first["counts"] == {
        "total": 178,
        "train": 106,
        "tuning_validation": 36,
        "heldout_test": 36,
    }

