"""Benchmark request latency against a running HTTP inference service."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from .data import load_dataset
from .metrics import environment_metadata, summarize_latencies


def _batch_from_rows(rows: np.ndarray, batch_size: int) -> list[list[float]]:
    if batch_size < 1:
        raise ValueError("batch sizes must be positive")
    repeats = int(np.ceil(batch_size / len(rows)))
    return np.tile(rows, (repeats, 1))[:batch_size].tolist()


def _post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def run_http_benchmark(
    url: str,
    batch_sizes: list[int] | None = None,
    iterations: int = 25,
    warmup_iterations: int = 5,
    random_state: int = 42,
) -> dict[str, object]:
    """Measure end-to-end HTTP request latency, including JSON serialization."""

    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    batch_sizes = batch_sizes or [1, 8, 32]
    split = load_dataset(random_state=random_state)
    measurements = []

    for batch_size in batch_sizes:
        features = _batch_from_rows(split.x_test, batch_size)
        payload = {"features": features}
        for _ in range(warmup_iterations):
            _post_json(url, payload)

        latencies_ms = []
        for _ in range(iterations):
            started = time.perf_counter()
            response = _post_json(url, payload)
            elapsed_ms = (time.perf_counter() - started) * 1000
            if len(response["predictions"]) != batch_size:
                raise RuntimeError("server returned a prediction count different from the request")
            latencies_ms.append(elapsed_ms)

        summary = summarize_latencies(latencies_ms)
        summary.update(
            {
                "backend": "fastapi-http",
                "batch_size": batch_size,
                "iterations": iterations,
                "warmup_iterations": warmup_iterations,
                "throughput_samples_per_second": float(
                    batch_size * iterations / (sum(latencies_ms) / 1000)
                ),
            }
        )
        measurements.append(summary)

    return {
        "schema_version": 1,
        "benchmark": {
            "workload": "scikit-learn digits classification over HTTP",
            "url": url,
            "measurements": measurements,
        },
        "environment": environment_metadata(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--output", type=Path, default=Path("results/http-benchmark.json"))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = run_http_benchmark(
        url=args.url,
        batch_sizes=args.batch_sizes,
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
        random_state=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
