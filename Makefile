.PHONY: install test lint benchmark serve

install:
	python -m pip install -e ".[dev,server]"

test:
	python -m pytest

lint:
	python -m ruff check .

benchmark:
	python -m production_inference_benchmark.benchmark --output results/local-benchmark.json

serve:
	python -m production_inference_benchmark.server

