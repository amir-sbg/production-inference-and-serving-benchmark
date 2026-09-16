from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from production_inference_benchmark.data import load_dataset
from production_inference_benchmark.model import load_model
from production_inference_benchmark.service import ModelService, create_app


def test_health_and_prediction_endpoints(tmp_path: Path) -> None:
    model = load_model(tmp_path / "model.joblib")
    app = create_app(model=model)
    client = TestClient(app)
    split = load_dataset()

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    response = client.post("/predict", json={"features": split.x_test[:3].tolist()})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 3
    assert len(response.json()["probabilities"]) == 3


def test_prediction_rejects_wrong_feature_count(tmp_path: Path) -> None:
    model = load_model(tmp_path / "model.joblib")
    client = TestClient(create_app(model=model))

    response = client.post("/predict", json={"features": [[0.0, 1.0]]})

    assert response.status_code == 422
    assert "expected 64 features" in response.json()["detail"]


def test_prediction_rejects_non_finite_features(tmp_path: Path) -> None:
    model = load_model(tmp_path / "model.joblib")
    service = ModelService(model=model)

    with pytest.raises(ValueError, match="finite numeric values"):
        service.predict([[float("nan")] * 64])
