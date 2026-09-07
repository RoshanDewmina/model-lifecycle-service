from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from model_lifecycle.api import create_app


def test_health_and_actual_prediction(
    trained_registry: Path, valid_payload: dict[str, float]
) -> None:
    client = TestClient(create_app(trained_registry))
    assert client.get("/health").json() == {
        "status": "ok",
        "service": "model-lifecycle-service",
        "version": "test-v1",
    }
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "test-v1"
    assert body["class_name"] in {"cultivar_1", "cultivar_2", "cultivar_3"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-8


def test_missing_invalid_and_extra_inputs_are_rejected(
    trained_registry: Path, valid_payload: dict[str, float]
) -> None:
    client = TestClient(create_app(trained_registry))
    missing = {key: value for key, value in valid_payload.items() if key != "alcohol"}
    assert client.post("/predict", json=missing).status_code == 422
    assert client.post("/predict", json={**valid_payload, "alcohol": 999}).status_code == 422
    assert client.post("/predict", json={**valid_payload, "unknown": 1}).status_code == 422
    assert client.post("/predict", json={**valid_payload, "ash": "2.14"}).status_code == 422


def test_dependency_failure_is_non_200(tmp_path: Path) -> None:
    response = TestClient(create_app(tmp_path / "missing")).get("/health")
    assert response.status_code == 503
    assert response.json()["detail"]["status"] == "error"


def test_non_finite_raw_json_is_a_safe_422(
    trained_registry: Path, valid_payload: dict[str, float]
) -> None:
    client = TestClient(create_app(trained_registry), raise_server_exceptions=False)
    base = json.dumps(valid_payload)
    for token in ("1e999", "NaN", "Infinity", "-Infinity"):
        raw = base.replace("13.2", token, 1)
        response = client.post(
            "/predict", content=raw, headers={"content-type": "application/json"}
        )
        assert response.status_code == 422
        assert response.headers["content-type"].startswith("application/json")


def test_malformed_active_metadata_is_503_on_all_model_routes(
    trained_registry: Path, valid_payload: dict[str, float], tmp_path: Path
) -> None:
    registry = tmp_path / "registry"
    shutil.copytree(trained_registry, registry)
    (registry / "versions" / "test-v1" / "metadata.json").write_text("[]\n")
    client = TestClient(create_app(registry), raise_server_exceptions=False)
    assert client.get("/health").status_code == 503
    assert client.get("/evidence").status_code == 503
    assert client.post("/predict", json=valid_payload).status_code == 503
