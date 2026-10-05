from types import SimpleNamespace

import numpy as np

from production_inference_benchmark import http_benchmark


def test_http_benchmark_separates_server_and_transport_cost(monkeypatch) -> None:
    monkeypatch.setattr(
        http_benchmark,
        "load_dataset",
        lambda random_state: SimpleNamespace(x_test=np.ones((3, 4))),
    )
    monkeypatch.setattr(
        http_benchmark,
        "_post_json",
        lambda url, payload: {
            "predictions": [1] * len(payload["features"]),
            "probabilities": [[0.1, 0.9]] * len(payload["features"]),
            "latency_ms": 0.25,
        },
    )

    report = http_benchmark.run_http_benchmark(
        "http://test/predict",
        batch_sizes=[2],
        iterations=2,
        warmup_iterations=1,
    )
    measurement = report["benchmark"]["measurements"][0]

    assert measurement["request_bytes"] > 0
    assert measurement["response_bytes"] > 0
    assert measurement["server_compute_latency"]["p50_ms"] == 0.25
    assert measurement["median_transport_and_serialization_ms"] >= 0
