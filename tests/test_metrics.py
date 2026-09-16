import numpy as np
import pytest

from production_inference_benchmark.metrics import benchmark_callable, summarize_latencies


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


def test_empty_measurements_are_rejected() -> None:
    with pytest.raises(ValueError, match="no measurements"):
        summarize_latencies([])
