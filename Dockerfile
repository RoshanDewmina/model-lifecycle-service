FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8115 \
    MODEL_REGISTRY=/app/artifacts/registry

WORKDIR /app
RUN pip install --no-cache-dir uv==0.11.8
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev
COPY scripts ./scripts
RUN uv run modelctl ensure-demo --registry /app/artifacts/registry

EXPOSE 8115
CMD ["sh", "-c", "uv run uvicorn model_lifecycle.api:app --host ${HOST} --port ${PORT}"]

