# Production Inference and Serving Benchmark

Reproducible benchmark for measuring the parts of an ML system that become important after a model works: inference latency, tail latency, batching, throughput, Python memory, and HTTP serving overhead.

The project is intentionally small enough to run on a laptop or in CI. It uses a persisted scikit-learn classifier as a reference workload, then measures it both in-process and behind a FastAPI service. The goal is not to claim that a digits classifier represents a production model. The goal is to make the measurement harness, serving boundary, and engineering tradeoffs easy to inspect and extend to a larger model.

[![CI](https://github.com/amir-sbg/production-inference-and-serving-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/amir-sbg/production-inference-and-serving-benchmark/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Why this repository exists

Model quality is only one part of an ML product. A useful service also needs to answer practical questions:

- How much latency does a single request pay?
- How does batching change throughput and tail latency?
- How much overhead comes from JSON serialization and HTTP?
- Can another engineer reproduce the measurements on a different machine?
- Are invalid inputs rejected clearly before they reach the model?

This repository turns those questions into code, tests, and machine-readable reports.

## What is included

- A deterministic training path for a 64-feature, 10-class reference model with accuracy, NLL, ECE, and multiclass Brier evaluation.
- An in-process benchmark with p50, p95, p99, min/max latency, throughput, and Python peak memory.
- A FastAPI service with `/health`, `/metadata`, and `/predict` endpoints.
- An HTTP benchmark that includes request encoding, network-loopback overhead, response decoding, and model inference.
- A Docker image for running the service.
- Unit tests for the model artifact, benchmark metrics, request validation, and prediction responses.
- GitHub Actions for tests and linting.

## Architecture

```text
scikit-learn digits data
          |
          v
  StandardScaler + LogisticRegression
          |
          +----------------------+
          |                      |
          v                      v
  In-process benchmark      FastAPI service
  p50/p95/p99/throughput    /health /metadata /predict
                                 |
                                 v
                         HTTP benchmark client
```

The reference workload is deliberately CPU-friendly. A future model adapter can replace the classifier while keeping the benchmark and serving contracts unchanged.

## Quick start

Create an environment and install the project with its development and server dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,server]"
```

Run the tests and linter:

```bash
python -m pytest
python -m ruff check .
```

Run the local inference benchmark:

```bash
python -m production_inference_benchmark.benchmark \
  --output results/local-benchmark.json \
  --iterations 50 \
  --warmup-iterations 10 \
  --p95-target-ms 10
```

The model is trained and saved automatically at `artifacts/digits-logistic-regression.joblib`. Artifacts and machine-specific results are ignored by Git so benchmark runs do not create noisy commits.

## Run the service

Start the local API:

```bash
python -m production_inference_benchmark.server
```

In another terminal, check the service:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metadata
```

Send a prediction request with 64 features per row:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features": [[0, 0, 5, 13, 9, 1, 0, 0, 0, 0, 13, 15, 10, 15, 5, 0, 0, 3, 15, 2, 0, 11, 8, 0, 0, 4, 12, 0, 0, 8, 8, 0, 0, 5, 8, 0, 0, 9, 8, 0, 0, 4, 11, 0, 1, 12, 7, 0, 0, 2, 14, 5, 10, 12, 0, 0, 0, 6, 13, 10, 0, 0, 0]]}'
```

Measure end-to-end loopback serving overhead:

```bash
python -m production_inference_benchmark.http_benchmark \
  --url http://127.0.0.1:8000/predict \
  --output results/http-benchmark.json \
  --iterations 25 \
  --warmup-iterations 5
```

The HTTP benchmark checks that the server returns the expected number of predictions for every request. It is intentionally a client-side benchmark: the reported latency includes the complete request/response path, not only the model call.

## Docker

Build and run the service without installing Python dependencies on the host:

```bash
docker build -t production-inference-benchmark .
docker run --rm -p 8000:8000 production-inference-benchmark
```

The image trains the small reference model lazily on the first request. For a real deployment, the model artifact should be built in a controlled release step and loaded during application startup.

## Reading the measurements

The benchmark reports latency in milliseconds, throughput in samples per second, and a simple p95 SLO check. It measures each batch as one model call, so a batch-size-32 result is not directly comparable to a batch-size-1 result as a user-facing request latency.

The following numbers are one local run on an Apple Silicon laptop using Python 3.13, 20 timed in-process iterations, and 10 timed HTTP iterations. They are included to show the shape of the output, not as universal performance claims.

| Path | Batch size | p50 | p95 | Throughput |
| --- | ---: | ---: | ---: | ---: |
| In-process | 1 | 0.28 ms | 0.35 ms | 3,467 samples/s |
| In-process | 32 | 0.28 ms | 0.29 ms | 112,718 samples/s |
| FastAPI over loopback | 1 | 0.80 ms | 0.95 ms | 1,207 samples/s |
| FastAPI over loopback | 32 | 1.09 ms | 1.17 ms | 29,201 samples/s |

The gap between the first and third rows is the cost of crossing the service boundary for this very small model. With a larger model, model execution would dominate more of the request time; with a smaller model, serialization and scheduling overhead become more visible.

## Repository layout

```text
src/production_inference_benchmark/
├── benchmark.py       # in-process benchmark CLI
├── data.py            # deterministic dataset split
├── http_benchmark.py  # client-side HTTP benchmark CLI
├── metrics.py         # timing summaries and environment capture
├── model.py           # model construction and persistence
├── server.py          # Uvicorn entry point
└── service.py         # FastAPI application and input validation
tests/
├── test_benchmark.py
├── test_metrics.py
├── test_model.py
└── test_service.py
```

## Design decisions and limitations

### Why scikit-learn instead of a large language model?

The first version optimizes for reproducibility. A reviewer can run the complete project without a GPU, model credentials, or a multi-gigabyte download. The benchmark is structured so a PyTorch, ONNX, or transformer backend can be added later without changing the report vocabulary.

### What this benchmark does not measure

It does not claim to measure GPU kernel efficiency, distributed serving, cold-start time, concurrency under load, autoscaling, or model quality for a production domain. Those require a model, hardware, traffic distribution, and SLO that match the deployment being studied.

### What should be added for a production model

- A model-specific accuracy and regression suite.
- Cold-start and warm-start measurements.
- Concurrent load tests and queueing behavior.
- CPU/GPU utilization and memory telemetry.
- Error-rate and timeout reporting.
- Canary comparison against the previous model version.
- Cost per thousand requests under a stated cloud configuration.

## License

MIT. See [LICENSE](LICENSE).
