# Architecture decisions

## Local-first core

The core loop depends only on NumPy and uses an atomic JSON registry. This keeps the principal
claim—automatic detection, retraining, canary evaluation, and rollback—runnable in seconds. The
FastAPI, Prometheus, and Grafana adapters are provided through Docker Compose.

## Production extension seams

- Replace `FileModelRegistry` with an MLflow adapter; keep aliases and release transitions.
- Feed `TelemetryRow` from Kafka/Redpanda and persist Bronze/Silver/Gold with Spark + Delta Lake.
- Invoke `HealingLoop.process` from a Dagster asset/sensor rather than the demo driver.
- Replace the in-process metrics collector only if labels or multiprocess aggregation are needed.

These are replacements at stable seams, not extra wrappers around the local implementation.

## Safety gates

The quality gate runs before drift scoring or training. The offline metric gate compares candidate
and production on the same recent validation slice. A separate cross-batch canary gate compares
shadow errors and either moves the alias atomically or removes the candidate alias and records a
rollback event.

