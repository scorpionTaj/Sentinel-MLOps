from __future__ import annotations

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
        else:
            version = existing_version
        self.drift.fit_reference(self._monitored(x))
        self._gold.extend(gold)
        self._bootstrapped = True
        self.metrics.gauge("sentinel_model_version", version)
        return version

    def process(self, rows: list[TelemetryRow]) -> BatchResult:
        if not self._bootstrapped:
            raise RuntimeError("bootstrap must run before process")
        events: list[PipelineEvent] = []
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
        events.append(self._event("drift_scored", "PSI scored for validated batch", drift.aggregate_psi))
        if drift.warning:
            self.metrics.increment("sentinel_drift_warnings_total")
            events.append(self._event("drift_warning", "moderate shift logged", drift.aggregate_psi))

        if drift.detected and self.registry.version("canary") is None:
            self.metrics.increment("sentinel_drift_breaches_total")
            events.append(self._event("drift_detected", "threshold breached", drift.aggregate_psi))
            candidate_event = self._retrain(features)
            events.append(candidate_event)

        for row in features:
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
                events.append(self._event("promoted", self._evidence_detail("non-inferior")))
            elif prediction.decision == "rolled_back":
                self.metrics.increment("sentinel_rollbacks_total")
                self._canary_reference = None
                events.append(self._event("rolled_back", self._evidence_detail("regression")))

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
            return self._event("candidate_rejected", f"version {version} failed offline gate", candidate_mae)
        self.registry.start_canary(version)
        self.router.reset_evidence()
        self._canary_reference = self._monitored(x)
        self.metrics.gauge("sentinel_canary_version", version)
        return self._event("canary_started", f"version {version} passed offline gate", candidate_mae)

    def _event(self, kind: str, detail: str, value: float | None = None) -> PipelineEvent:
        self._sequence += 1
        return PipelineEvent(self._sequence, kind, detail, value)

    def _evidence_detail(self, fallback: str) -> str:
        evidence = self.router.last_evidence
        if not evidence:
            return fallback
        return (
            f"{evidence['reason']}: canary MAE {evidence['canary_mae']:.3f} vs production "
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
