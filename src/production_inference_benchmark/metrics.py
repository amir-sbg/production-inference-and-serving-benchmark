"""Timing and reporting utilities shared by local and HTTP benchmarks."""

from __future__ import annotations

import os
import platform
import time
import tracemalloc
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np


def expected_calibration_error(
    confidence: np.ndarray,
    correct: np.ndarray,
    bins: int = 10,
) -> float:
    if bins < 2:
        raise ValueError("bins must be at least 2")
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if confidence.ndim != 1 or correct.shape != confidence.shape or confidence.size == 0:
        raise ValueError("confidence and correctness must be aligned non-empty vectors")
    if np.any(confidence < 0) or np.any(confidence > 1):
        raise ValueError("confidence must be between zero and one")

    bin_ids = np.minimum((confidence * bins).astype(int), bins - 1)
    error = 0.0
    for bin_id in range(bins):
        active = bin_ids == bin_id
        if np.any(active):
            error += np.mean(active) * abs(np.mean(confidence[active]) - np.mean(correct[active]))
    return float(error)


def multiclass_brier_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    truth = np.asarray(labels)
    scores = np.asarray(probabilities, dtype=np.float64)
    if scores.ndim != 2 or truth.ndim != 1 or len(truth) != len(scores) or len(truth) == 0:
        raise ValueError("labels and probabilities must be aligned non-empty arrays")
    if not np.issubdtype(truth.dtype, np.integer):
        raise ValueError("labels must be integer class indices")
    if np.any(truth < 0) or np.any(truth >= scores.shape[1]):
        raise ValueError("labels contain a class outside the probability columns")
    if not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(scores.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("probability rows must sum to one")
    targets = np.eye(scores.shape[1], dtype=np.float64)[truth]
    return float(np.mean(np.sum((scores - targets) ** 2, axis=1)))


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
        "expected_calibration_error": expected_calibration_error(confidence, correct),
        "multiclass_brier_score": multiclass_brier_score(truth, scores),
    }


def classwise_quality(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, float | int]]:
    """Report support and error rates that aggregate accuracy can hide."""

    truth = np.asarray(labels)
    scores = np.asarray(probabilities, dtype=np.float64)
    classification_quality(truth, scores)
    predictions = np.argmax(scores, axis=1)
    rows = []
    for class_index in range(scores.shape[1]):
        actual = truth == class_index
        predicted = predictions == class_index
        true_positive = int(np.sum(actual & predicted))
        support = int(np.sum(actual))
        predicted_count = int(np.sum(predicted))
        rows.append(
            {
                "class_index": class_index,
                "support": support,
                "predicted_count": predicted_count,
                "precision": true_positive / predicted_count if predicted_count else 0.0,
                "recall": true_positive / support if support else 0.0,
                "mean_true_class_probability": (
                    float(np.mean(scores[actual, class_index])) if support else 0.0
                ),
            }
        )
    return rows


def prediction_drift_report(
    reference_probabilities: np.ndarray,
    observed_probabilities: np.ndarray,
    bins: int = 10,
) -> dict[str, float | int]:
    """Compare class mix and confidence distributions between two traffic samples."""

    reference = np.asarray(reference_probabilities, dtype=np.float64)
    observed = np.asarray(observed_probabilities, dtype=np.float64)
    if reference.ndim != 2 or observed.ndim != 2 or reference.shape[1] != observed.shape[1]:
        raise ValueError("probability matrices must have the same number of classes")
    if len(reference) == 0 or len(observed) == 0 or bins < 2:
        raise ValueError("probability matrices must be non-empty and bins must be at least 2")
    for values in (reference, observed):
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("probabilities must be finite and non-negative")
        if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("probability rows must sum to one")

    epsilon = 1e-8
    reference_mix = np.clip(reference.mean(axis=0), epsilon, 1.0)
    observed_mix = np.clip(observed.mean(axis=0), epsilon, 1.0)
    midpoint = 0.5 * (reference_mix + observed_mix)
    js_divergence = 0.5 * np.sum(reference_mix * np.log(reference_mix / midpoint))
    js_divergence += 0.5 * np.sum(observed_mix * np.log(observed_mix / midpoint))

    edges = np.linspace(0.0, 1.0, bins + 1)
    reference_histogram = np.histogram(reference.max(axis=1), bins=edges)[0] / len(reference)
    observed_histogram = np.histogram(observed.max(axis=1), bins=edges)[0] / len(observed)
    reference_histogram = np.clip(reference_histogram, epsilon, None)
    observed_histogram = np.clip(observed_histogram, epsilon, None)
    confidence_psi = np.sum(
        (observed_histogram - reference_histogram)
        * np.log(observed_histogram / reference_histogram)
    )
    return {
        "reference_examples": int(len(reference)),
        "observed_examples": int(len(observed)),
        "classes": int(reference.shape[1]),
        "confidence_bins": int(bins),
        "class_distribution_js_divergence": float(js_divergence),
        "confidence_population_stability_index": float(confidence_psi),
        "mean_confidence_shift": float(observed.max(axis=1).mean() - reference.max(axis=1).mean()),
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


def latency_stability_report(
    latencies_ms: list[float],
    drift_tolerance: float = 0.10,
    dispersion_tolerance: float = 0.15,
) -> dict[str, float | int | bool]:
    """Flag timing runs whose center or dispersion changes during measurement."""

    values = np.asarray(latencies_ms, dtype=np.float64)
    if values.size < 2 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("latency stability needs at least two finite non-negative measurements")
    if drift_tolerance < 0 or dispersion_tolerance < 0:
        raise ValueError("stability tolerances must not be negative")

    split = len(values) // 2
    first_median = float(np.median(values[:split]))
    second_median = float(np.median(values[split:]))
    median = float(np.median(values))
    scale = max(median, np.finfo(np.float64).eps)
    relative_drift = (second_median - first_median) / scale
    median_absolute_deviation = float(np.median(np.abs(values - median)))
    robust_coefficient_of_variation = 1.4826 * median_absolute_deviation / scale
    sufficient_samples = values.size >= 4
    return {
        "sample_count": int(values.size),
        "sufficient_samples": bool(sufficient_samples),
        "first_half_median_ms": first_median,
        "second_half_median_ms": second_median,
        "relative_median_drift": float(relative_drift),
        "robust_coefficient_of_variation": float(robust_coefficient_of_variation),
        "stable": bool(
            sufficient_samples
            and abs(relative_drift) <= drift_tolerance
            and robust_coefficient_of_variation <= dispersion_tolerance
        ),
    }


def bootstrap_latency_interval(
    latencies_ms: list[float],
    quantile: float = 95.0,
    confidence: float = 0.95,
    samples: int = 500,
    seed: int = 42,
) -> dict[str, float | int]:
    """Estimate uncertainty around a latency percentile by resampling requests."""

    values = np.asarray(latencies_ms, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("latencies must be non-empty, finite, and non-negative")
    if not 0 < quantile <= 100 or not 0 < confidence < 1 or samples < 1:
        raise ValueError("quantile, confidence, and samples are outside their valid ranges")

    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True)
    estimates = np.percentile(draws, quantile, axis=1)
    tail = (1.0 - confidence) / 2.0
    return {
        "quantile": float(quantile),
        "confidence": float(confidence),
        "bootstrap_samples": int(samples),
        "estimate_ms": float(np.percentile(values, quantile)),
        "lower_ms": float(np.quantile(estimates, tail)),
        "upper_ms": float(np.quantile(estimates, 1.0 - tail)),
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


def capacity_headroom(
    measurements: list[dict[str, Any]],
    target_samples_per_second: float,
) -> dict[str, float | int | bool]:
    """Compare a traffic target with measured single-request service capacity."""

    if target_samples_per_second <= 0:
        raise ValueError("target_samples_per_second must be positive")
    single_rows = [row for row in measurements if int(row["batch_size"]) == 1]
    if len(single_rows) != 1:
        raise ValueError("exactly one batch-size-one measurement is required")
    measured_capacity = float(single_rows[0]["throughput_samples_per_second"])
    if measured_capacity <= 0:
        raise ValueError("measured throughput must be positive")
    utilization = target_samples_per_second / measured_capacity
    return {
        "batch_size": 1,
        "target_samples_per_second": float(target_samples_per_second),
        "measured_capacity_samples_per_second": measured_capacity,
        "estimated_utilization": float(utilization),
        "headroom_samples_per_second": float(measured_capacity - target_samples_per_second),
        "headroom_fraction": float(1.0 - utilization),
        "target_within_measured_capacity": bool(utilization <= 1.0),
    }


def mm1_queueing_report(
    service_rate_per_second: float,
    arrival_rate_per_second: float,
    wait_target_ms: float = 10.0,
) -> dict[str, float | bool | None]:
    """Approximate queue delay for a single worker under Poisson traffic."""

    if service_rate_per_second <= 0 or arrival_rate_per_second < 0:
        raise ValueError("service rate must be positive and arrival rate must not be negative")
    if wait_target_ms <= 0:
        raise ValueError("wait_target_ms must be positive")

    utilization = arrival_rate_per_second / service_rate_per_second
    stable = utilization < 1.0
    service_time_ms = 1000.0 / service_rate_per_second
    if not stable:
        return {
            "stable": False,
            "utilization": float(utilization),
            "expected_service_time_ms": float(service_time_ms),
            "expected_queue_wait_ms": None,
            "expected_response_time_ms": None,
            "wait_target_ms": float(wait_target_ms),
            "probability_queue_wait_exceeds_target": None,
        }

    rate_gap = service_rate_per_second - arrival_rate_per_second
    expected_response_ms = 1000.0 / rate_gap
    expected_queue_wait_ms = expected_response_ms - service_time_ms
    wait_probability = utilization * np.exp(-rate_gap * wait_target_ms / 1000.0)
    return {
        "stable": True,
        "utilization": float(utilization),
        "expected_service_time_ms": float(service_time_ms),
        "expected_queue_wait_ms": float(expected_queue_wait_ms),
        "expected_response_time_ms": float(expected_response_ms),
        "wait_target_ms": float(wait_target_ms),
        "probability_queue_wait_exceeds_target": float(wait_probability),
    }


def select_batch_operating_point(
    measurements: list[dict[str, Any]],
    p95_target_ms: float,
) -> dict[str, float | int | bool]:
    """Choose the highest-throughput batch size that satisfies the latency target."""

    if not measurements or p95_target_ms <= 0:
        raise ValueError("measurements must be non-empty and p95_target_ms must be positive")
    candidates = [row for row in measurements if float(row["p95_ms"]) <= p95_target_ms]
    within_target = bool(candidates)
    if candidates:
        selected = max(
            candidates,
            key=lambda row: (
                float(row["throughput_samples_per_second"]),
                -float(row["p95_ms"]),
            ),
        )
    else:
        selected = min(measurements, key=lambda row: float(row["p95_ms"]))
    return {
        "batch_size": int(selected["batch_size"]),
        "p95_ms": float(selected["p95_ms"]),
        "throughput_samples_per_second": float(selected["throughput_samples_per_second"]),
        "p95_target_ms": float(p95_target_ms),
        "within_latency_target": within_target,
    }


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

    first_call_started = time.perf_counter()
    predict(batch)
    first_call_ms = (time.perf_counter() - first_call_started) * 1000

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
        "first_call_ms": float(first_call_ms),
    }
    summary.update(summarize_latencies(latencies_ms))
    summary["p95_confidence_interval"] = bootstrap_latency_interval(latencies_ms)
    summary["latency_stability"] = latency_stability_report(latencies_ms)
    summary["first_call_to_warm_p50_ratio"] = float(
        first_call_ms / max(float(summary["p50_ms"]), 1e-12)
    )
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
