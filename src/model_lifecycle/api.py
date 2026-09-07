from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import __version__
from .features import WineFeatures
from .lifecycle import CLASS_NAMES, ArtifactError, LoadedModel, load_active

SERVICE_NAME = "model-lifecycle-service"


class PredictionResponse(BaseModel):
    prediction: int
    class_name: str
    probabilities: dict[str, float]
    model_version: str


def create_app(registry: Path | None = None) -> FastAPI:
    bundled_registry = Path(__file__).parent / "bundled_registry"
    registry_path = registry or Path(os.getenv("MODEL_REGISTRY", str(bundled_registry)))
    application = FastAPI(title="Model Lifecycle Service", version=__version__)
    state: dict[str, LoadedModel | str | None] = {"loaded": None, "error": None}

    def refresh() -> LoadedModel:
        try:
            loaded = load_active(registry_path)
            state["loaded"] = loaded
            state["error"] = None
            return loaded
        except (ArtifactError, OSError, ValueError, KeyError) as exc:
            state["loaded"] = None
            state["error"] = str(exc)
            raise

    @application.get("/health")
    def health() -> dict[str, str]:
        try:
            loaded = refresh()
        except (ArtifactError, OSError, ValueError, KeyError) as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "error",
                    "service": SERVICE_NAME,
                    "version": __version__,
                    "reason": str(exc),
                },
            ) from exc
        return {"status": "ok", "service": SERVICE_NAME, "version": loaded.metadata["version"]}

    @application.post("/predict", response_model=PredictionResponse)
    def predict(features: WineFeatures) -> PredictionResponse:
        try:
            loaded = refresh()
        except (ArtifactError, OSError, ValueError, KeyError) as exc:
            raise HTTPException(status_code=503, detail="active model unavailable") from exc
        row = np.asarray([features.as_row()], dtype=np.float64)
        prediction = int(loaded.model.predict(row)[0])
        raw_probabilities = loaded.model.predict_proba(row)[0]
        probabilities = {
            CLASS_NAMES[int(class_id)]: float(probability)
            for class_id, probability in zip(loaded.model.classes_, raw_probabilities, strict=True)
        }
        return PredictionResponse(
            prediction=prediction,
            class_name=CLASS_NAMES[prediction],
            probabilities=probabilities,
            model_version=loaded.metadata["version"],
        )

    @application.get("/evidence")
    def evidence() -> dict[str, Any]:
        try:
            loaded = refresh()
        except (ArtifactError, OSError, ValueError, KeyError) as exc:
            raise HTTPException(status_code=503, detail="active model unavailable") from exc
        evaluation = loaded.metadata["evaluation"]
        return {
            "model_version": loaded.metadata["version"],
            "artifact_sha256": loaded.metadata["artifact_sha256"],
            "heldout_test": evaluation["heldout_test"],
            "dummy_baseline_heldout_test": evaluation["dummy_baseline_heldout_test"],
            "release_gate": loaded.metadata["release_gate"],
            "limitations": loaded.metadata["limitations"],
        }

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    return application


app = create_app()
