"""Model training and persistence for the reference inference workload."""

from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_dataset

MODEL_VERSION = "digits-logistic-regression-v1"


def build_model(random_state: int = 42) -> Pipeline:
    """Build the reference model used in local and HTTP benchmarks."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=500,
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def train_model(output_path: Path, random_state: int = 42) -> Pipeline:
    """Train and persist the reference model, creating its parent directory."""

    split = load_dataset(random_state=random_state)
    model = build_model(random_state=random_state)
    model.fit(split.x_train, split.y_train)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return model


def load_model(model_path: Path, random_state: int = 42) -> Pipeline:
    """Load a persisted model, training it when the artifact is absent."""

    if not model_path.exists():
        return train_model(model_path, random_state=random_state)
    return joblib.load(model_path)


def model_metadata(model: Pipeline) -> dict[str, object]:
    """Return metadata suitable for benchmark reports and service responses."""

    classifier = model.named_steps["classifier"]
    return {
        "model_version": MODEL_VERSION,
        "model_type": type(classifier).__name__,
        "n_features": int(model.n_features_in_),
        "n_classes": int(len(classifier.classes_)),
    }
