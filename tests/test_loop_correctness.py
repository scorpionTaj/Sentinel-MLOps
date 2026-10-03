from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from sentinel.features import MONITORED_FEATURES, RollingFeaturePipeline
from sentinel.loop import HealingConfig, HealingLoop
from sentinel.model import RidgeModel
from sentinel.registry import FileModelRegistry
from sentinel.router import CanaryRouter
from sentinel.synthetic import generate_telemetry
from sentinel.types import TelemetryRow


class LoopCorrectnessTests(unittest.TestCase):
    def test_shorter_engine_histories_do_not_trigger_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            loop = HealingLoop(Path(temporary))
            loop.bootstrap(generate_telemetry("FD001", engines=8, cycles=45, seed=10))
            result = loop.process(
                generate_telemetry("FD001", engines=3, cycles=20, seed=77, engine_offset=500)
            )
        self.assertNotIn("cycle_scaled", result.drift.feature_psi)
        self.assertFalse(result.drift.detected)

    def test_promotion_moves_the_drift_reference_to_the_new_regime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            loop = HealingLoop(
                Path(temporary), HealingConfig(canary_min_observations=60, canary_weight=0.2)
            )
            loop.bootstrap(generate_telemetry("FD001", engines=8, cycles=45, seed=10))
            first = loop.process(generate_telemetry("FD002", 3, 35, seed=20, engine_offset=100))
            kinds = [event.kind for event in first.events]
            self.assertEqual(kinds[-2:], ["canary_started", "promoted"])
            later = [
                loop.process(generate_telemetry("FD002", 3, 35, seed=seed, engine_offset=offset))
                for seed, offset in ((30, 200), (40, 300))
            ]
        for result in later:
            self.assertFalse(result.drift.detected, result.drift.feature_psi)
            self.assertEqual(result.active_version, 2)

    def test_rolling_state_is_not_shared_across_domains(self) -> None:
        pipeline = RollingFeaturePipeline(window=5)
        first = TelemetryRow(1, 1, 0.0, (100.0, 0.0, 100.0, 0.0, 0.0), 10.0, "FD001")
        second = TelemetryRow(1, 1, 0.0, (1.0, 0.0, 1.0, 0.0, 0.0), 10.0, "FD002")
        features = pipeline.transform([first, second])
        self.assertEqual(features[1].values[-2:], (1.0, 1.0))
        self.assertNotEqual(features[0].group, features[1].group)

    def test_bootstrap_records_engine_holdout_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            loop = HealingLoop(Path(temporary))
            loop.bootstrap(generate_telemetry("FD001", engines=8, cycles=45, seed=10))
            metrics = loop.status()["versions"][0]["metrics"]
        self.assertIn("holdout_mae", metrics)
        self.assertGreater(metrics["holdout_mae"], metrics["training_mae"])

    def test_drift_report_exposes_unbounded_shift_for_monitored_features(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            loop = HealingLoop(Path(temporary))
            loop.bootstrap(generate_telemetry("FD001", engines=8, cycles=45, seed=10))
            report = loop.process(generate_telemetry("ADVERSARIAL", 3, 35, engine_offset=100)).drift
        self.assertEqual(set(report.feature_shift), set(MONITORED_FEATURES))
        self.assertGreater(max(report.feature_shift.values()), 5.0)


    def test_concept_drift_with_stable_inputs_triggers_retraining(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            loop = HealingLoop(Path(temporary), HealingConfig(canary_min_observations=300))
            loop.bootstrap(generate_telemetry("FD001", engines=8, cycles=45, seed=10))
            for seed, offset in ((1099, 100), (2099, 300)):
                loop.process(generate_telemetry("FD002", 6, 35, seed=seed, engine_offset=offset))
            self.assertEqual(loop.registry.version("production"), 2)
            stable = loop.process(generate_telemetry("FD002", 6, 35, seed=7, engine_offset=600))
            inverted = loop.process(
                generate_telemetry("ADVERSARIAL", 6, 35, seed=8, engine_offset=900)
            )
        self.assertNotIn("performance_degraded", [e.kind for e in stable.events])
        self.assertFalse(inverted.drift.detected)
        self.assertEqual(
            [e.kind for e in inverted.events][1:], ["performance_degraded", "canary_started"]
        )

    def test_status_exposes_ui_histories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            loop = HealingLoop(Path(temporary), HealingConfig(canary_min_observations=300))
            loop.bootstrap(generate_telemetry("FD001", engines=8, cycles=45, seed=10))
            loop.process(generate_telemetry("FD002", 6, 35, seed=20, engine_offset=100))
            status = loop.status()
        self.assertEqual(status["pipeline_events"][0]["kind"], "bootstrapped")
        self.assertEqual(status["drift_history"][0]["verdict"], "detected")
        self.assertEqual(status["canary_progress"]["observations"], 210)
        self.assertEqual(status["canary_progress"]["min_observations"], 300)


class CanaryDecisionTests(unittest.TestCase):
    def _router(self, temporary: str, **kwargs: object) -> CanaryRouter:
        registry = FileModelRegistry(Path(temporary) / "registry.json")
        x = np.column_stack([np.arange(10.0), np.ones(10)])
        good = RidgeModel.fit(x, np.arange(10.0))
        production = registry.register(good, {}, "candidate")
        registry.deploy_initial(production)
        noisy = RidgeModel(good.mean, good.scale, good.coefficients, good.intercept + 0.01)
        registry.start_canary(registry.register(noisy, {}, "candidate"))
        return CanaryRouter(registry, weight=0.5, regression_tolerance=0.0, **kwargs)

    def test_inconclusive_evidence_keeps_collecting_then_keeps_production(self) -> None:
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as temporary:
            router = self._router(temporary, min_observations=20, max_observations=60)
            decisions = []
            for index in range(60):
                value = float(index % 10)
                actual = value + rng.normal(0, 3)
                decisions.append(router.predict((value, 1.0), f"r{index}", actual=actual).decision)
            self.assertTrue(all(decision is None for decision in decisions[:59]))
            self.assertEqual(decisions[59], "rolled_back")
            self.assertEqual(router.last_evidence["reason"], "inconclusive")
            self.assertEqual(router.registry.version("production"), 1)


if __name__ == "__main__":
    unittest.main()
