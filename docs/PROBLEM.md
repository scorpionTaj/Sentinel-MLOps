# Problem Statement and Drift Strategy

Sentinel estimates remaining useful life (RUL) from multivariate turbofan telemetry. A baseline model trained under the single-condition `FD001` regime is subjected to continuous operational telemetry. When operating regimes shift or fault degradation emerges, the production model's error rises.

Sentinel continuously scores distribution drift, isolates candidate releases in shadow evaluation, and automatically promotes or rolls back models based on live evidence.

---

## Population Stability Index (PSI)

Population Stability Index (PSI) quantifies shift between a baseline reference distribution $Q$ and incoming operational batches $P$ across quantile bins $B$:

$$\text{PSI} = \sum_{b=1}^{B} (P_b - Q_b) \times \ln\left(\frac{P_b}{Q_b}\right)$$

### Decision Thresholds
- **$\text{PSI} < 0.10$**: Stable distribution. Baseline model remains in full production.
- **$0.10 \le \text{PSI} \le 0.25$**: Moderate shift. Warning logged; evidence buffered.
- **$\text{PSI} \ge 0.20$ (Configurable Safety Line)**: Distributional breach. Sentinel initiates automated retraining, validates candidate against production on recent observations, and deploys a canary.

---

## Scenario Catalogue

Sentinel ships with a deterministic generator modeling the NASA C-MAPSS dataset families and sensor stress scenarios. It also provides loaders (`sentinel.datasets.load_cmapps_training` and `load_cmapps_test`) for official 26-column NASA telemetry files placed under `data/reference/`:

| Scenario ID | Regime / Stress Type | Operating Conditions | Fault Modes | Drift Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **`FD001`** *(Baseline)* | Reference condition | Single (Sea Level) | 1 (HPC degradation) | Baseline reference distribution |
| **`FD002`** | Multi-condition drift | 6 operational regimes | 1 (HPC degradation) | Operating condition & sensor mean shift |
| **`FD003`** | Dual-fault drift | Single (Sea Level) | 2 (HPC + Fan degradation) | Degradation trajectory deviation |
| **`FD004`** | Complex compound drift | 6 operational regimes | 2 (HPC + Fan degradation) | Compound operating & fault shift |
| **`SENSOR_BIAS`** | Sensor hardware calibration drift | Single (Sea Level) | 1 (HPC degradation) | Constant sensor measurement offset (+3.5σ) |
| **`SENSOR_DROPOUT`** | Hardware telemetry loss | Single (Sea Level) | 1 (HPC degradation) | Signal attenuation / zeroing on critical channels |
| **`NOISE_BURST`** | Acquisition noise | Single (Sea Level) | 1 (HPC degradation) | High-variance measurements across every channel |
| **`ADVERSARIAL`** | Relationship inversion | Multi-condition | Synthetic inversion | Health-to-sensor relationships reverse direction |

---

## The Shadow Canary & Rollback Proof

A model that passes offline validation may still fail in production due to serving skew, feature drift, or serialized pipeline defects. Sentinel guards against this through **online shadow evaluation**:

1. **Deterministic Canary Routing**: Live inference requests are routed using cryptographic SHA-256 bucketing (`bucket = sha256(request_id) < canary_weight`).
2. **Shadow Scoring**: Predictions from both the active production model and the canary candidate are recorded alongside observed ground truth.
3. **Automated Promotion vs. Rollback**:
   - **Promotion**: When canary MAE is superior or within `regression_tolerance` over `canary_min_observations`, the canary is promoted to production.
   - **Rollback**: If the canary exhibits serving regression (simulated via `/simulate-regression`), the canary alias is immediately revoked, protecting production baseline traffic.

---

## Execution Stack and Architecture Scope

The reference implementation runs entirely in-process using pure **NumPy** for the core loop and **FastAPI** for HTTP serving, with an optional Docker Compose profile for **Prometheus** and **Grafana**.

Distributed infrastructure tools (Kafka/Redpanda, Spark, Delta Lake, Dagster, and MLflow) are not active runtime dependencies in this repository; they are documented architectural extension seams described in [`DECISIONS.md`](DECISIONS.md).
