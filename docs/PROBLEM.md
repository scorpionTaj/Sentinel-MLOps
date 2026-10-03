# Problem Statement and Drift Strategy

Sentinel estimates remaining useful life (RUL) from multivariate turbofan telemetry. A baseline model trained under the single-condition `FD001` regime is subjected to continuous operational telemetry. When operating regimes shift or fault degradation emerges, the production model's error rises.

Sentinel continuously scores distribution drift, isolates candidate releases in shadow evaluation, and automatically promotes or rolls back models based on live evidence.

---

## Population Stability Index (PSI)

Population Stability Index (PSI) quantifies shift between a baseline reference distribution $Q$ and incoming operational batches $P$ across quantile bins $B$:

$$\text{PSI} = \sum_{b=1}^{B} (P_b - Q_b) \times \ln\left(\frac{P_b}{Q_b}\right)$$

PSI is computed per monitored feature on 8 reference-quantile bins with a 0.5 pseudo-count per bin,
and the batch score is the maximum over features. `cycle_scaled` is **not** monitored: it is a time
index, so a batch of shorter engine histories would look like drift without any change in the data.

### Decision rule
- **Alert line** = max(0.20, noise floor). The noise floor is the PSI that sampling noise alone
  exceeds with 1% family-wise probability: χ²₇ quantile × (1/n_batch + 1/n_reference). It only
  binds for small batches (≈0.55 at 50 rows, ≈0.13 at 400 rows against a 360-row reference).
- **Warning** (`drift_warning` event): PSI between half the alert line and the alert line.
- **Breach** (`drift_detected`): Sentinel retrains on the batch (engine-grouped validation),
  applies the offline gate, and starts a canary.
- After a promotion, the reference is re-fitted on the promoted model's training window, so the
  new regime becomes the baseline.

### Second trigger: performance degradation (concept drift)
PSI only sees input distributions. A regime can keep every input distribution and still break the
model; `ADVERSARIAL` inverts the sensor-to-RUL relationship, and after an FD002 promotion it scores
PSI ≈ 0.08. Because each batch arrives with labels, the loop also compares production MAE on the
batch with the model's recorded validation MAE. At ≥ 4× (stable batches measure 1.4–3.0×) it emits
`performance_degraded` and retrains exactly as for a PSI breach.

Each report also carries `feature_shift` (|Δmean| in reference σ), an unbounded effect size that
keeps ranking severity once binned PSI saturates. Measured false-alarm and detection rates are in
[EVALUATION.md](EVALUATION.md) §2.

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
3. **Automated Promotion vs. Rollback** (paired non-inferiority test on shadow absolute errors,
   d = |canary error| − |production error|, margin = `canary_regression_tolerance` × production MAE):
   - **Promotion**: after at least `canary_min_observations`, the 95% upper bound of mean(d) is
     within the margin.
   - **Rollback**: the 95% lower bound exceeds the margin (e.g. the serving regression injected by
     `/simulate-regression`).
   - **Inconclusive**: keep collecting; if still undecided at `canary_max_observations`
     (default 4× the minimum), roll back and keep production.

Targets are capped at 125 cycles (`rul_cap`) for both training and shadow scoring.

---

## Execution Stack and Architecture Scope

The reference implementation runs entirely in-process using pure **NumPy** for the core loop and **FastAPI** for HTTP serving, with an optional Docker Compose profile for **Prometheus** and **Grafana**.

Distributed infrastructure tools (Kafka/Redpanda, Spark, Delta Lake, Dagster, and MLflow) are not active runtime dependencies in this repository; they are documented architectural extension seams described in [`DECISIONS.md`](DECISIONS.md).
