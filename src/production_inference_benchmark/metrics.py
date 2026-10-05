"""Timing and reporting utilities shared by local and HTTP benchmarks."""

from __future__ import annotations

import os
import platform
import time
import tracemalloc
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np


def classification_quality(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Summarize predictive quality without hiding confidence errors."""

    truth = np.asarray(labels)
    scores = np.asarray(probabilities, dtype=np.float64)
    if scores.ndim != 2 or truth.ndim != 1 or len(truth) != len(scores):
        raise ValueError("labels and probabilities have incompatible shapes")
    if len(truth) == 0 or not np.isfinite(scores).all():
        raise ValueError("quality inputs must be non-empty and finite")
    if np.any(scores < 0) or not np.allclose(scores.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("each probability row must be non-negative and sum to one")
    if not np.issubdtype(truth.dtype, np.integer):
        raise ValueError("labels must be integer class indices")
    if np.any(truth < 0) or np.any(truth >= scores.shape[1]):
        raise ValueError("labels contain a class outside the probability columns")

    predictions = np.argmax(scores, axis=1)
    confidence = np.max(scores, axis=1)
    correct = predictions == truth
    likelihood = scores[np.arange(len(truth)), truth]
    return {
        "accuracy": float(np.mean(correct)),
        "mean_confidence": float(np.mean(confidence)),
        "confidence_gap": float(np.mean(confidence) - np.mean(correct)),
        "negative_log_likelihood": float(np.mean(-np.log(np.clip(likelihood, 1e-12, 1.0)))),
    }


def percentile(values: Iterable[float], quantile: float) -> float:
    """Return a percentile as a plain float, with a clear empty-input error."""

    values_array = np.asarray(list(values), dtype=float)
    if values_array.size == 0:
        raise ValueError("cannot calculate a percentile from no measurements")
    return float(np.percentile(values_array, quantile))


def summarize_latencies(latencies_ms: list[float]) -> dict[str, float]:
    """Summarize per-request latencies in milliseconds."""

    if not latencies_ms:
        raise ValueError("no measurements were provided")
    return {
        "p50_ms": percentile(latencies_ms, 50),
        "p95_ms": percentile(latencies_ms, 95),
        "p99_ms": percentile(latencies_ms, 99),
        "min_ms": float(min(latencies_ms)),
        "max_ms": float(max(latencies_ms)),
    }


def latency_slo_report(summary: dict[str, float], p95_target_ms: float) -> dict[str, float | bool]:
    if p95_target_ms <= 0:
        raise ValueError("p95_target_ms must be positive")
    p95_ms = float(summary["p95_ms"])
    p99_ms = float(summary["p99_ms"])
    return {
        "p95_target_ms": float(p95_target_ms),
        "p95_within_target": bool(p95_ms <= p95_target_ms),
        "p95_margin_ms": float(p95_target_ms - p95_ms),
        "p99_to_p95_ratio": float(p99_ms / p95_ms) if p95_ms > 0 else float("inf"),
    }


def batch_efficiency_rows(measurements: list[dict[str, Any]]) -> list[dict[str, float | int]]:
    if not measurements:
        return []
    sorted_rows = sorted(measurements, key=lambda row: int(row["batch_size"]))
    baseline = float(sorted_rows[0]["throughput_samples_per_second"])
    baseline_batch = int(sorted_rows[0]["batch_size"])
    if baseline <= 0 or baseline_batch < 1:
        raise ValueError("baseline throughput and batch size must be positive")
    return [
        {
            "batch_size": int(row["batch_size"]),
            "throughput_samples_per_second": float(row["throughput_samples_per_second"]),
            "throughput_gain_vs_smallest_batch": float(
                float(row["throughput_samples_per_second"]) / baseline
            ),
            "batch_efficiency": float(
                float(row["throughput_samples_per_second"])
                / baseline
                / (int(row["batch_size"]) / baseline_batch)
            ),
        }
        for row in sorted_rows
    ]


def benchmark_callable(
    predict: Callable[[np.ndarray], Any],
    batch: np.ndarray,
    iterations: int,
    warmup_iterations: int,
) -> dict[str, float | int | str]:
    """Measure a callable that accepts one batch of feature rows."""

    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations cannot be negative")
    if batch.ndim != 2 or batch.shape[0] == 0:
        raise ValueError("batch must be a non-empty two-dimensional array")

    for _ in range(warmup_iterations):
        predict(batch)

    latencies_ms: list[float] = []
    tracemalloc.start()
    try:
        for _ in range(iterations):
            started = time.perf_counter()
            predict(batch)
            latencies_ms.append((time.perf_counter() - started) * 1000)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    total_seconds = sum(latencies_ms) / 1000
    summary: dict[str, float | int | str] = {
        "sample_count": int(batch.shape[0]),
        "iterations": iterations,
        "warmup_iterations": warmup_iterations,
        "throughput_samples_per_second": float(batch.shape[0] * iterations / total_seconds),
        "python_peak_memory_mb": float(peak_bytes / (1024 * 1024)),
    }
    summary.update(summarize_latencies(latencies_ms))
    return summary


def environment_metadata() -> dict[str, object]:
    """Capture context needed to interpret machine-dependent measurements."""

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "numpy_version": np.__version__,
    }
