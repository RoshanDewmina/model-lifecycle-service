from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient

from model_lifecycle.api import create_app
from model_lifecycle.lifecycle import (
    GateRejected,
    drift_diagnostic,
    promote,
    rollback,
    train_candidate,
)
from model_lifecycle.receipt import git_state, receipt_base, write_json

SAMPLE = {
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


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[min(int(len(values) * fraction), len(values) - 1)]


def main() -> None:
    revision, dirty = git_state()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="model-lifecycle-benchmark-") as temp:
        registry = Path(temp) / "registry"
        v1 = train_candidate(registry, "benchmark-v1")
        promote(registry, "benchmark-v1")
        client = TestClient(create_app(registry))
        health = client.get("/health")
        prediction = client.post("/predict", json=SAMPLE)
        latencies_ms = []
        for _ in range(200):
            request_started = time.perf_counter()
            response = client.post("/predict", json=SAMPLE)
            response.raise_for_status()
            latencies_ms.append((time.perf_counter() - request_started) * 1000)

        train_candidate(registry, "rejected-dummy", model_kind="dummy")
        rejection_observed = False
        try:
            promote(registry, "rejected-dummy")
        except GateRejected:
            rejection_observed = True

        train_candidate(registry, "benchmark-v2")
        promote(registry, "benchmark-v2")
        rollback_state = rollback(registry, "benchmark-v1")
        drift = drift_diagnostic(registry, simulate=True)

    receipt = receipt_base("make benchmark", revision, dirty)
    receipt.update(
        {
            "exit_status": 0,
            "inputs": {
                "seed": 20260907,
                "request_count": 200,
                "workload": (
                    "sequential in-process FastAPI TestClient requests on one fixed valid sample"
                ),
            },
            "measured_results": {
                "total_duration_seconds": time.perf_counter() - started,
                "health_status": health.status_code,
                "prediction_status": prediction.status_code,
                "prediction": prediction.json(),
                "latency_ms": {
                    "min": min(latencies_ms),
                    "median": percentile(latencies_ms, 0.50),
                    "p95": percentile(latencies_ms, 0.95),
                    "max": max(latencies_ms),
                },
                "heldout_balanced_accuracy": v1["evaluation"]["heldout_test"]["balanced_accuracy"],
                "dummy_baseline_balanced_accuracy": v1["evaluation"]
                ["dummy_baseline_heldout_test"]["balanced_accuracy"],
                "tuning_validation_balanced_accuracy": v1["evaluation"]["tuning_validation"]
                ["balanced_accuracy"],
                "dummy_baseline_tuning_validation_balanced_accuracy": v1["evaluation"]
                ["dummy_baseline_tuning_validation"]["balanced_accuracy"],
                "release_gate_partition": v1["release_gate"]["partition"],
                "candidate_rejection_observed": rejection_observed,
                "rollback_target": rollback_state["active_version"],
                "simulated_drift_label": drift["diagnostic_type"],
                "simulated_drift_flagged_features": drift["flagged_features"],
            },
            "limitations": [
                "Latency uses an in-process sequential TestClient workload, "
                "not a network load test.",
                "The drift batch is generated and does not represent observed real-world data.",
                "Small local dataset and single-machine measurements do not establish "
                "production performance.",
            ],
        }
    )
    write_json(Path("evidence/generated/benchmark-receipt.json"), receipt)
    print(json.dumps(receipt["measured_results"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
