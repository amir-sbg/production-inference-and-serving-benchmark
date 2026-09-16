from pathlib import Path

from production_inference_benchmark.data import load_dataset
from production_inference_benchmark.model import load_model, model_metadata


def test_model_is_reproducible_and_persisted(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    split = load_dataset()

    first = load_model(model_path)
    second = load_model(model_path)

    assert model_path.exists()
    assert model_metadata(first) == model_metadata(second)
    assert first.score(split.x_test, split.y_test) > 0.90

