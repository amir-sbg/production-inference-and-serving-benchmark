FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir ".[server]"

EXPOSE 8000

CMD ["uvicorn", "production_inference_benchmark.service:app", "--host", "0.0.0.0", "--port", "8000"]

