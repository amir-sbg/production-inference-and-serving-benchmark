"""Command-line benchmark for in-process model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import load_dataset
from .metrics import (
    batch_efficiency_rows,
    benchmark_callable,
    capacity_headroom,
    classification_quality,
    environment_metadata,
    latency_slo_report,
    prediction_drift_report,
    select_batch_operating_point,
)
from .model import MODEL_VERSION, load_model, model_metadata

DEFAULT_MODEL_PATH = Path("artifacts/digits-logistic-regression.joblib")


def _batch_from_rows(rows: np.ndarray, batch_size: int) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch sizes must be positive")
    repeats = int(np.ceil(batch_size / len(rows)))
    return np.tile(rows, (repeats, 1))[:batch_size]


def run_benchmark(
    model_path: Path = DEFAULT_MODEL_PATH,
    batch_sizes: list[int] | None = None,
    iterations: int = 50,
    warmup_iterations: int = 10,
    random_state: int = 42,
    p95_target_ms: float = 10.0,
    target_samples_per_second: float = 1000.0,
) -> dict[str, object]:
    """Run the reference model across several batch sizes."""

    batch_sizes = batch_sizes or [1, 8, 32, 128]
    split = load_dataset(random_state=random_state)
    model = load_model(model_path, random_state=random_state)
    test_probabilities = model.predict_proba(split.x_test)
    train_probabilities = model.predict_proba(split.x_train)

    measurements = []
    for batch_size in batch_sizes:
        batch = _batch_from_rows(split.x_test, batch_size)
        measurement = benchmark_callable(
            model.predict_proba,
            batch,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
        )
        measurement["backend"] = "sklearn-in-process"
        measurement["batch_size"] = batch_size
        measurement["latency_slo"] = latency_slo_report(measurement, p95_target_ms)
        measurements.append(measurement)

    return {
        "schema_version": 1,
        "benchmark": {
            "workload": "scikit-learn digits classification",
            "model_version": MODEL_VERSION,
            "measurements": measurements,
            "batch_efficiency": batch_efficiency_rows(measurements),
            "capacity_headroom": capacity_headroom(measurements, target_samples_per_second),
            "recommended_operating_point": select_batch_operating_point(
                measurements,
                p95_target_ms,
            ),
        },
        "model": model_metadata(model),
        "quality": classification_quality(split.y_test, test_probabilities),
        "prediction_drift": prediction_drift_report(train_probabilities, test_probabilities),
        "environment": environment_metadata(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=Path("results/local-benchmark.json"))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--p95-target-ms", type=float, default=10.0)
    parser.add_argument("--target-samples-per-second", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = run_benchmark(
        model_path=args.model_path,
        batch_sizes=args.batch_sizes,
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
        random_state=args.seed,
        p95_target_ms=args.p95_target_ms,
        target_samples_per_second=args.target_samples_per_second,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
