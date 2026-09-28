import numpy as np
import pytest

from production_inference_benchmark.metrics import (
    benchmark_callable,
    latency_slo_report,
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


def test_latency_slo_report_tracks_target_margin() -> None:
    report = latency_slo_report({"p95_ms": 8.0, "p99_ms": 12.0}, p95_target_ms=10.0)

    assert report["p95_within_target"] is True
    assert report["p95_margin_ms"] == 2.0
    assert report["p99_to_p95_ratio"] == 1.5


def test_empty_measurements_are_rejected() -> None:
    with pytest.raises(ValueError, match="no measurements"):
        summarize_latencies([])
