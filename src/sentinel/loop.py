from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sentinel.drift import PSIDriftDetector
from sentinel.evaluation.splits import group_split
from sentinel.features import MONITORED_FEATURES, MONITORED_INDEX, RollingFeaturePipeline
from sentinel.metrics import Metrics
from sentinel.model import RidgeModel, mae
from sentinel.quality import DataQualityError, QualityGate
from sentinel.registry import FileModelRegistry
from sentinel.router import CanaryRouter, Prediction
from sentinel.types import BatchResult, DriftReport, FeatureRow, PipelineEvent, TelemetryRow


@dataclass(frozen=True, slots=True)
class HealingConfig:
    drift_threshold: float = 0.2
    drift_warning_threshold: float | None = None
    offline_regression_tolerance: float = 0.05
    canary_weight: float = 0.1
    canary_min_observations: int = 50
    canary_max_observations: int | None = None
    canary_regression_tolerance: float = 0.03
    minimum_retrain_rows: int = 30
    validation_fraction: float = 0.3
    # Piecewise-linear RUL target: early-life cycles are indistinguishable from healthy ones,
    # so training and shadow evaluation both cap the target (standard C-MAPSS practice).
    rul_cap: float | None = 125.0
    # Concept drift can leave every input distribution unchanged (e.g. an inverted sensor-to-RUL
    # relationship), so batches with labels are also checked against the production model's
    # recorded validation error. Stable synthetic batches measure 1.4-3.0x; 4x is the trigger.
    performance_degradation_ratio: float | None = 4.0


class HealingLoop:
    """The closed-loop interface: bootstrap, process telemetry, and predict."""

    def __init__(self, state_dir: Path, config: HealingConfig | None = None) -> None:
        self.config = config or HealingConfig()
        self.features = RollingFeaturePipeline()
        self.quality = QualityGate()
        self.drift = PSIDriftDetector(
            threshold=self.config.drift_threshold,
            feature_names=MONITORED_FEATURES,
            warning_threshold=self.config.drift_warning_threshold,
        )
        self.registry = FileModelRegistry(state_dir / "registry.json")
        self.metrics = Metrics()
        self.router = CanaryRouter(
            self.registry,
            weight=self.config.canary_weight,
            min_observations=self.config.canary_min_observations,
            regression_tolerance=self.config.canary_regression_tolerance,
            max_observations=self.config.canary_max_observations,
        )
        self._sequence = 0
        self._gold: list[FeatureRow] = []
        self._bootstrapped = False
        self._last_drift: DriftReport | None = None
        self._canary_reference: np.ndarray | None = None
        self._batch = 0
        self._domain = ""
        # Bounded histories surfaced by status() for the control-room UI.
        self._history: deque[dict[str, object]] = deque(maxlen=200)
        self._drift_history: deque[dict[str, object]] = deque(maxlen=60)

    def bootstrap(self, rows: list[TelemetryRow]) -> int:
        if self._bootstrapped:
            raise RuntimeError("loop is already bootstrapped")
        gold = self.features.transform(rows)
        self.quality.validate(gold)
        x, y = self._matrix(gold)
        existing_version = self.registry.version("production")
        if existing_version is None:
            train, holdout = group_split([row.group for row in gold], self.config.validation_fraction)
            holdout_mae = mae(RidgeModel.fit(x[train], y[train]), x[holdout], y[holdout])
            model = RidgeModel.fit(x, y)
            baseline_mae = mae(model, x, y)
            version = self.registry.register(
                model, {"training_mae": baseline_mae, "holdout_mae": holdout_mae}, "candidate"
            )
            self.registry.deploy_initial(version)
            self.metrics.gauge("sentinel_training_mae", baseline_mae)
            self.metrics.gauge("sentinel_holdout_mae", holdout_mae)
            self.metrics.increment("sentinel_bootstrap_total")
            self._event(
                "bootstrapped",
                f"trained on {len(gold)} rows · train MAE {baseline_mae:.3f}, "
                f"engine-holdout MAE {holdout_mae:.3f}",
                holdout_mae,
                version,
            )
        else:
            version = existing_version
            self._event("restored", f"production v{version} restored from registry", None, version)
        self.drift.fit_reference(self._monitored(x))
        self._gold.extend(gold)
        self._bootstrapped = True
        self.metrics.gauge("sentinel_model_version", version)
        return version

    def process(self, rows: list[TelemetryRow]) -> BatchResult:
        if not self._bootstrapped:
            raise RuntimeError("bootstrap must run before process")
        events: list[PipelineEvent] = []
        self._batch += 1
        self.metrics.increment("sentinel_batches_total")
        self._domain = Counter(row.domain for row in rows).most_common(1)[0][0] if rows else ""
        features = self.features.transform(rows)
        try:
            quality = self.quality.validate(features)
        except DataQualityError:
            self.metrics.increment("sentinel_quality_rejections_total")
            raise
        self._gold.extend(features)
        x, _ = self._matrix(features)
        drift = self.drift.score(self._monitored(x))
        self._last_drift = drift
        self.metrics.gauge("sentinel_drift_psi", drift.aggregate_psi)
        alert_line = max(drift.threshold, drift.noise_floor)
        top_feature = max(drift.feature_psi, key=drift.feature_psi.get)
        verdict = "detected" if drift.detected else "warning" if drift.warning else "stable"
        self._drift_history.append(
            {
                "batch": self._batch,
                "domain": self._domain,
                "rows": len(features),
                "psi": drift.aggregate_psi,
                "alert_line": alert_line,
                "verdict": verdict,
                "top_feature": top_feature,
            }
        )
        events.append(
            self._event(
                "drift_scored",
                f"{len(features)} rows · PSI {drift.aggregate_psi:.3f} vs alert line "
                f"{alert_line:.3f} · {verdict}",
                drift.aggregate_psi,
            )
        )
        if drift.warning:
            self.metrics.increment("sentinel_drift_warnings_total")
            events.append(
                self._event(
                    "drift_warning",
                    f"moderate shift on {top_feature}; logged, no retrain",
                    drift.aggregate_psi,
                )
            )

        degradation = self._performance_degradation(features)
        self._drift_history[-1]["error_ratio"] = degradation
        no_canary = self.registry.version("canary") is None
        if drift.detected and no_canary:
            self.metrics.increment("sentinel_drift_breaches_total")
            events.append(
                self._event(
                    "drift_detected",
                    f"PSI {drift.aggregate_psi:.3f} ≥ {alert_line:.3f}; largest shift on "
                    f"{top_feature} ({drift.feature_shift.get(top_feature, 0.0):.2f}σ)",
                    drift.aggregate_psi,
                )
            )
            events.append(self._retrain(features))
        elif (
            no_canary
            and degradation is not None
            and self.config.performance_degradation_ratio is not None
            and degradation >= self.config.performance_degradation_ratio
        ):
            self.metrics.increment("sentinel_performance_breaches_total")
            events.append(
                self._event(
                    "performance_degraded",
                    f"inputs look stable (PSI {drift.aggregate_psi:.3f}) but production MAE is "
                    f"{degradation:.1f}× its validation MAE (trigger "
                    f"{self.config.performance_degradation_ratio:.1f}×): concept drift",
                    degradation,
                    self.registry.version("production"),
                )
            )
            events.append(self._retrain(features))

        for row in features:
            canary_before = self.registry.version("canary")
            prediction = self.router.predict(
                row.values, row.request_id, actual=self._cap(row.target)
            )
            if prediction.decision == "promoted":
                self.metrics.increment("sentinel_promotions_total")
                self.metrics.gauge(
                    "sentinel_model_version", self.registry.version("production") or 0
                )
                # The promoted model defines the new normal: drift is now measured against the
                # data it was trained on, otherwise every later batch would re-trigger retraining.
                if self._canary_reference is not None:
                    self.drift.fit_reference(self._canary_reference)
                    self._canary_reference = None
                events.append(
                    self._event("promoted", self._evidence_detail("non-inferior"), None, canary_before)
                )
            elif prediction.decision == "rolled_back":
                self.metrics.increment("sentinel_rollbacks_total")
                self._canary_reference = None
                events.append(
                    self._event(
                        "rolled_back", self._evidence_detail("regression"), None, canary_before
                    )
                )

        return BatchResult(
            quality=quality,
            drift=drift,
            active_version=self.registry.version("production") or 0,
            canary_version=self.registry.version("canary"),
            events=tuple(events),
        )

    def predict(self, values: tuple[float, ...], request_id: str) -> Prediction:
        return self.router.predict(values, request_id)

    def inject_canary_regression(self, offset: float = 80.0) -> int:
        """Deliberately damage a canary artifact to exercise automatic rollback."""
        canary = self.registry.model("canary")
        damaged = RidgeModel(
            mean=canary.mean,
            scale=canary.scale,
            coefficients=canary.coefficients,
            intercept=canary.intercept + offset,
        )
        version = self.registry.replace_model("canary", damaged, "regression_injected")
        self.metrics.increment("sentinel_fault_injections_total")
        return version

    def status(self) -> dict[str, object]:
        snapshot = self.registry.snapshot()
        versions = [
            {
                "version": record["version"],
                "state": record["state"],
                "metrics": record["metrics"],
            }
            for record in snapshot["versions"]
        ]
        return {
            "bootstrapped": self._bootstrapped,
            "aliases": snapshot["aliases"],
            "events": snapshot["events"],
            "versions": versions,
            "gold_rows": len(self._gold),
            "metrics": self.metrics.snapshot(),
            "last_drift": self._last_drift.to_dict() if self._last_drift is not None else None,
            "last_canary_evidence": self.router.last_evidence,
            "canary_progress": self._canary_progress(),
            "performance_trigger": self.config.performance_degradation_ratio,
            "pipeline_events": list(self._history),
            "drift_history": list(self._drift_history),
        }

    def _performance_degradation(self, features: list[FeatureRow]) -> float | None:
        """Production MAE on this labelled batch divided by its recorded validation MAE."""

        version = self.registry.version("production")
        record = next(
            (v for v in self.registry.snapshot()["versions"] if v["version"] == version), None
        )
        if record is None:
            return None
        expected = record["metrics"].get("validation_mae", record["metrics"].get("holdout_mae"))
        if not expected or expected <= 0:
            return None
        x, y = self._matrix(features)
        return mae(self.registry.model("production"), x, y) / float(expected)

    def _canary_progress(self) -> dict[str, int] | None:
        version = self.registry.version("canary")
        if version is None:
            return None
        return {
            "version": version,
            "observations": self.router.observations,
            "min_observations": self.router.min_observations,
            "max_observations": self.router.max_observations,
            "traffic_weight": self.router.weight,
        }

    def _retrain(self, recent: list[FeatureRow]) -> PipelineEvent:
        if len(recent) < self.config.minimum_retrain_rows:
            return self._event("retrain_skipped", "insufficient recent Gold rows", float(len(recent)))
        x, y = self._matrix(recent)
        train, validation = group_split(
            [row.group for row in recent], self.config.validation_fraction
        )
        if len(train) < 2 or len(validation) < 1:
            return self._event("retrain_skipped", "not enough engines to validate", float(len(recent)))
        train_x, train_y = x[train], y[train]
        validation_x, validation_y = x[validation], y[validation]
        candidate = RidgeModel.fit(train_x, train_y)
        candidate_mae = mae(candidate, validation_x, validation_y)
        production_mae = mae(self.registry.model("production"), validation_x, validation_y)
        version = self.registry.register(
            candidate,
            {"validation_mae": candidate_mae, "production_mae": production_mae},
            "candidate",
        )
        self.metrics.increment("sentinel_retrains_total")
        if candidate_mae > production_mae * (1 + self.config.offline_regression_tolerance):
            self.registry.reject(version)
            self.metrics.increment("sentinel_offline_rejections_total")
            return self._event(
                "candidate_rejected",
                f"validation MAE {candidate_mae:.3f} > production {production_mae:.3f} + "
                f"{self.config.offline_regression_tolerance:.0%}",
                candidate_mae,
                version,
            )
        self.registry.start_canary(version)
        self.router.reset_evidence()
        self._canary_reference = self._monitored(x)
        self.metrics.gauge("sentinel_canary_version", version)
        return self._event(
            "canary_started",
            f"offline gate passed: validation MAE {candidate_mae:.3f} vs production "
            f"{production_mae:.3f} on {len(validation)} held-out rows",
            candidate_mae,
            version,
        )

    def _event(
        self, kind: str, detail: str, value: float | None = None, version: int | None = None
    ) -> PipelineEvent:
        self._sequence += 1
        event = PipelineEvent(self._sequence, kind, detail, value, version)
        self._history.append({**event.to_dict(), "batch": self._batch, "domain": self._domain})
        return event

    def _evidence_detail(self, fallback: str) -> str:
        evidence = self.router.last_evidence
        if not evidence:
            return fallback
        prefix = "inconclusive at the observation cap; " if evidence["reason"] == "inconclusive" else ""
        return (
            f"{prefix}canary MAE {evidence['canary_mae']:.3f} vs production "
            f"{evidence['production_mae']:.3f}, paired diff CI "
            f"[{evidence['ci_lower']:.3f}, {evidence['ci_upper']:.3f}] "
            f"margin {evidence['margin']:.3f}, n={evidence['observations']}"
        )

    def _cap(self, target: float) -> float:
        cap = self.config.rul_cap
        return target if cap is None else min(target, cap)

    @staticmethod
    def _monitored(x: np.ndarray) -> np.ndarray:
        return x[:, MONITORED_INDEX]

    def _matrix(self, rows: list[FeatureRow]) -> tuple[np.ndarray, np.ndarray]:
        target = np.asarray([row.target for row in rows], dtype=float)
        if self.config.rul_cap is not None:
            target = np.minimum(target, self.config.rul_cap)
        return np.asarray([row.values for row in rows], dtype=float), target
