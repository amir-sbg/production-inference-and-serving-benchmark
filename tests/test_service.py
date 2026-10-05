import asyncio
from pathlib import Path

import httpx
import numpy as np
import pytest

from production_inference_benchmark.data import load_dataset
from production_inference_benchmark.model import load_model
from production_inference_benchmark.service import (
    ModelService,
    create_app,
    prediction_confidence_summary,
)


def request(app, method: str, path: str, **kwargs) -> httpx.Response:
    async def send_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send_request())


def test_health_and_prediction_endpoints(tmp_path: Path) -> None:
    model = load_model(tmp_path / "model.joblib")
    app = create_app(model=model)
    split = load_dataset()

    health = request(app, "GET", "/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    response = request(
        app,
        "POST",
        "/predict",
        json={"features": split.x_test[:3].tolist()},
    )
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 3
    assert len(response.json()["probabilities"]) == 3
    assert "mean_confidence" in response.json()["confidence"]


def test_prediction_confidence_summary_reports_probability_range() -> None:
    report = prediction_confidence_summary(
        np.array([[0.2, 0.8], [0.6, 0.4]], dtype=np.float64)
    )

    assert report["mean_confidence"] == pytest.approx(0.7)
    assert report["min_confidence"] == pytest.approx(0.6)
    assert report["max_confidence"] == pytest.approx(0.8)
    assert report["mean_entropy"] > 0
    assert 0 < report["mean_normalized_entropy"] < 1


def test_prediction_rejects_wrong_feature_count(tmp_path: Path) -> None:
    model = load_model(tmp_path / "model.joblib")

    response = request(
        create_app(model=model),
        "POST",
        "/predict",
        json={"features": [[0.0, 1.0]]},
    )

    assert response.status_code == 422
    assert "expected 64 features" in response.json()["detail"]


def test_prediction_rejects_non_finite_features(tmp_path: Path) -> None:
    model = load_model(tmp_path / "model.joblib")
    service = ModelService(model=model)

    with pytest.raises(ValueError, match="finite numeric values"):
        service.predict([[float("nan")] * 64])
