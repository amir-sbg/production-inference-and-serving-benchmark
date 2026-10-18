from pathlib import Path

from production_inference_benchmark.benchmark import run_benchmark


def test_benchmark_report_has_one_row_per_batch_size(tmp_path: Path) -> None:
    report = run_benchmark(
        model_path=tmp_path / "model.joblib",
        batch_sizes=[1, 4],
        iterations=2,
        warmup_iterations=1,
    )

    measurements = report["benchmark"]["measurements"]
    assert [row["batch_size"] for row in measurements] == [1, 4]
    assert all(row["p95_ms"] >= row["p50_ms"] for row in measurements)
    assert all(row["throughput_samples_per_second"] > 0 for row in measurements)
    assert all("first_call_to_warm_p50_ratio" in row for row in measurements)
    assert all("latency_slo" in row for row in measurements)
    assert [row["batch_size"] for row in report["benchmark"]["batch_efficiency"]] == [1, 4]
    assert report["prediction_drift"]["reference_examples"] > 0
    assert report["benchmark"]["capacity_headroom"]["batch_size"] == 1
    assert report["benchmark"]["recommended_operating_point"]["batch_size"] in {1, 4}
