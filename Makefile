.PHONY: setup test demo benchmark lint train

setup:
	uv sync --frozen

test:
	uv run ruff check .
	uv run pytest

train:
	uv run modelctl train --version wine-logreg-v1 --registry artifacts/registry --receipt evidence/generated/training-receipt.json
	uv run modelctl promote --version wine-logreg-v1 --registry artifacts/registry

demo:
	uv run modelctl ensure-demo --registry artifacts/registry
	MODEL_REGISTRY=artifacts/registry uv run uvicorn model_lifecycle.api:app --host "$${HOST:-127.0.0.1}" --port "$${PORT:-8115}"

benchmark:
	uv run python3 scripts/benchmark.py

lint:
	uv run ruff check .
