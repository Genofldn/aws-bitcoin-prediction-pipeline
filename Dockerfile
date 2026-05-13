FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

FROM base AS builder

COPY option1-robust-pipeline/lambda-functions/data-processor/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM base AS runtime

COPY --from=builder /install /usr/local

COPY option1-robust-pipeline/lambda-functions/data-processor/ ./data-processor/
COPY option1-robust-pipeline/lambda-functions/shared/ ./shared/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["python", "data-processor/predictor.py"]
