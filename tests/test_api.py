from __future__ import annotations

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

