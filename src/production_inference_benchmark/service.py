"""FastAPI application exposing the reference model as a small inference service."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .benchmark import DEFAULT_MODEL_PATH
from .model import MODEL_VERSION, load_model, model_metadata


class PredictionRequest(BaseModel):
    """JSON request accepted by the prediction endpoint."""

    features: list[list[float]]


class ModelService:
    """Lazy model loader used by the API and by tests."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, model: object | None = None):
        self.model_path = model_path
        self._model = model

    @property
    def model(self):
        if self._model is None:
            self._model = load_model(self.model_path)
        return self._model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def metadata(self) -> dict[str, object]:
        return model_metadata(self.model)

    def predict(self, features: list[list[float]]) -> tuple[list[int], list[list[float]], float]:
        array = np.asarray(features, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] == 0:
            raise ValueError("features must be a non-empty two-dimensional array")

        expected_features = int(self.model.n_features_in_)
        if array.shape[1] != expected_features:
            raise ValueError(f"expected {expected_features} features per row, got {array.shape[1]}")

        started = time.perf_counter()
        probabilities = self.model.predict_proba(array)
        classes = np.asarray(self.model.classes_)
        predictions = classes[np.argmax(probabilities, axis=1)]
        latency_ms = (time.perf_counter() - started) * 1000
        return predictions.astype(int).tolist(), probabilities.tolist(), latency_ms


def create_app(
    model_path: Path = DEFAULT_MODEL_PATH,
    model: object | None = None,
) -> FastAPI:
    """Create an application instance with an injectable model for testing."""

    service = ModelService(model_path=model_path, model=model)
    app = FastAPI(
        title="Production Inference and Serving Benchmark",
        version="0.1.0",
        description="A small, measurable reference service for ML inference workloads.",
    )

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "model_loaded": service.is_loaded,
            "model_version": MODEL_VERSION,
        }

    @app.get("/metadata")
    def metadata() -> dict[str, object]:
        return service.metadata()

    @app.post("/predict")
    def predict(request: PredictionRequest) -> dict[str, object]:
        try:
            predictions, probabilities, latency_ms = service.predict(request.features)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "latency_ms": latency_ms,
            "model_version": MODEL_VERSION,
        }

    return app


app = create_app()

