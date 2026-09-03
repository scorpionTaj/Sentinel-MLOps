# Architecture Decisions & Extension Seams

Sentinel is engineered with strict separation of concerns: a pure, deterministic core loop in Python/NumPy, with clean architectural extension seams for enterprise scale.

---

## 1. Local-First Core Strategy

### Context
MLOps reference architectures frequently suffer from dependency bloat: booting Kubernetes, Kafka, Spark, and MinIO locally takes minutes, consumes gigabytes of memory, and introduces flaky non-deterministic test failures.

### Decision
The core self-healing loop depends solely on **CPython 3.11+ and NumPy**.
- All mathematical operations (ridge regression, standardization, PSI distribution binning, MAE scoring) run in-process.
- Whole-loop end-to-end tests complete in **under 50 milliseconds**.
- Observability (Prometheus exporter and Grafana dashboards) is isolated in a Docker Compose profile, keeping the core testable anywhere.

---

## 2. Production Extension Seams

The system defines stable adapter seams rather than wrapping heavyweight frameworks in artificial abstractions:

### Seam A: Remote Model Registry (MLflow / S3)
- **Local Implementation**: [`FileModelRegistry`](../src/sentinel/registry.py) uses atomic JSON file writes (`os.replace`) to maintain versions, aliases (`production`, `canary`), metrics, and event audit history.
- **Reference Adapter**: [`MLflowModelRegistry`](../src/sentinel/adapters/mlflow.py) implements [`ModelRegistryProtocol`](../src/sentinel/registry.py):
  ```python
  class ModelRegistryProtocol(Protocol):
      def register(self, model: RidgeModel, metrics: dict[str, float], state: str) -> int: ...
      def deploy_initial(self, version: int) -> None: ...
      def start_canary(self, version: int) -> None: ...
      def promote(self) -> int: ...
      def rollback(self) -> int: ...
      def model(self, alias: str) -> RidgeModel: ...
      def version(self, alias: str) -> int | None: ...
  ```

### Seam B: Telemetry Streaming & Medallion Storage (Kafka + Delta Lake)
- **Local Implementation**: In-memory `TelemetryRow` sequence validated into structured feature matrices.
- **Reference Adapter**: [`MedallionLakehouse`](../src/sentinel/adapters/streaming.py) models the raw Bronze stream (Kafka offsets/payloads), typed Silver conformed rows, and validated Gold matrices, paired with [`StreamingBatchConsumer`](../src/sentinel/adapters/streaming.py).

### Seam C: Orchestration & Automation (Dagster / Prefect)
- **Local Implementation**: `HealingLoop.process(batch)` coordinates quality gating, drift detection, retraining, canary deployment, and rollback.
- **Reference Adapter**: [`HealingSensor`](../src/sentinel/adapters/orchestrator.py) implements the sensor pattern to monitor streaming queues or Gold table partitions and trigger healing executions.

---

## 3. Safety Gates & Defense-in-Depth

Sentinel implements four consecutive safety boundaries before any candidate can take 100% production traffic:

```
[ Ingested Telemetry ]
          │
          ▼
┌──────────────────┐
│ Data Quality Gate│ ──(Violations: NaN, null, out-of-range)──> [ Quarantine ]
└──────────────────┘
          │ (Validated Gold)
          ▼
┌──────────────────┐
│  Drift Detector  │ ──(PSI < 0.20)──> [ Keep Serving Production ]
└──────────────────┘
          │ (PSI ≥ 0.20 Threshold Breach)
          ▼
┌──────────────────┐
│   Retrain Loop   │ 
└──────────────────┘
          │
          ▼
┌──────────────────┐
│ Offline Gate     │ ──(Candidate MAE > Production MAE)──> [ Reject Candidate ]
└──────────────────┘
          │ (Candidate Outperforms Baseline)
          ▼
┌──────────────────┐
│ Shadow Canary    │ ──(Shadow MAE Regresses)──> [ Automatic Rollback ]
└──────────────────┘
          │ (Canary Outperforms over Observation Window)
          ▼
   [ Promote to Production ]
```

1. **Pre-Ingest Data Quality Gate**: Validates sensor limits, rejects nulls/NaNs, and guards against malformed inputs before touching memory.
2. **Drift Scoring Gate (PSI)**: Univariate Population Stability Index compares current operational telemetry to baseline reference. Breaches trigger automated retraining.
3. **Offline Validation Gate**: Newly trained candidate must demonstrate measurable error reduction on recent validation slices before receiving candidate deployment.
4. **Online Shadow Canary Gate**: Live inference is split via deterministic SHA-256 bucketing. Candidate model predictions are scored against observed ground truth. If serving regression occurs, the canary alias is revoked immediately.

---

## 4. Deterministic Canary Routing

### Context
Random routing (`random.random() < 0.1`) makes testing non-deterministic and prevents session-consistent canary assignments for the same request ID.

### Decision
Live traffic is partitioned using cryptographic SHA-256 hashing of the `request_id`:
```python
bucket = int.from_bytes(hashlib.sha256(request_id.encode()).digest()[:8], "big") / 2**64
use_canary = bucket < self.weight
```
This guarantees deterministic routing across distributed instances while ensuring uniform distribution across any number of requests.
