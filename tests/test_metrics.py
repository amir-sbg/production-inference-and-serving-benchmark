import numpy as np
import pytest

from production_inference_benchmark.metrics import (
    benchmark_callable,
    batch_efficiency_rows,
    classification_quality,
    expected_calibration_error,
    latency_slo_report,
    multiclass_brier_score,
    summarize_latencies,
)


def test_latency_summary_contains_percentiles() -> None:
    summary = summarize_latencies([1.0, 2.0, 3.0, 4.0, 5.0])

    assert summary["p50_ms"] == 3.0
    assert summary["p95_ms"] >= summary["p50_ms"]
    assert summary["p99_ms"] >= summary["p95_ms"]


def test_benchmark_callable_reports_throughput() -> None:
    batch = np.ones((4, 2))
    result = benchmark_callable(
        lambda rows: rows.sum(axis=1), batch, iterations=3, warmup_iterations=1
    )

    assert result["sample_count"] == 4
    assert result["iterations"] == 3
    assert result["throughput_samples_per_second"] > 0
    assert result["first_call_ms"] >= 0
    assert result["first_call_to_warm_p50_ratio"] >= 0


def test_first_call_is_excluded_from_warm_latency_samples() -> None:
    calls = 0

    def predict(rows):
        nonlocal calls
        calls += 1
        return rows

    result = benchmark_callable(predict, np.ones((2, 2)), iterations=3, warmup_iterations=2)

    assert calls == 6
    assert result["iterations"] == 3


def test_latency_slo_report_tracks_target_margin() -> None:
    report = latency_slo_report({"p95_ms": 8.0, "p99_ms": 12.0}, p95_target_ms=10.0)

    assert report["p95_within_target"] is True
    assert report["p95_margin_ms"] == 2.0
    assert report["p99_to_p95_ratio"] == 1.5


def test_batch_efficiency_rows_compare_to_smallest_batch() -> None:
    rows = batch_efficiency_rows(
        [
            {"batch_size": 8, "throughput_samples_per_second": 400.0},
            {"batch_size": 1, "throughput_samples_per_second": 100.0},
        ]
    )

    assert rows[0]["batch_size"] == 1
    assert rows[1]["throughput_gain_vs_smallest_batch"] == 4.0
    assert rows[1]["batch_efficiency"] == 0.5


def test_empty_measurements_are_rejected() -> None:
    with pytest.raises(ValueError, match="no measurements"):
        summarize_latencies([])


def test_classification_quality_reports_confidence_gap() -> None:
    quality = classification_quality(
        np.array([0, 1, 1]),
        np.array([[0.8, 0.2], [0.4, 0.6], [0.7, 0.3]]),
    )

    assert quality["accuracy"] == pytest.approx(2 / 3)
    assert quality["mean_confidence"] == pytest.approx(0.7)
    assert quality["confidence_gap"] == pytest.approx(0.7 - 2 / 3)
    assert quality["negative_log_likelihood"] > 0
    assert quality["expected_calibration_error"] >= 0


def test_expected_calibration_error_uses_fixed_probability_bins() -> None:
    error = expected_calibration_error(
        np.array([0.1, 0.2, 0.8, 0.9]),
        np.array([0, 0, 1, 1]),
        bins=2,
    )

    assert error == pytest.approx(0.15)


def test_multiclass_brier_score_rewards_sharp_correct_probabilities() -> None:
    labels = np.array([0, 1])
    sharp = np.array([[0.9, 0.1], [0.1, 0.9]])
    uncertain = np.array([[0.5, 0.5], [0.5, 0.5]])

    assert multiclass_brier_score(labels, sharp) < multiclass_brier_score(labels, uncertain)
    assert multiclass_brier_score(labels, sharp) == pytest.approx(0.02)
