from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from sentinel.cli import run_dataset_matrix
from sentinel.drift import PSIDriftDetector
from sentinel.loop import HealingConfig, HealingLoop
from sentinel.model import RidgeModel
from sentinel.quality import DataQualityError
from sentinel.registry import FileModelRegistry
from sentinel.scenarios import drift_scenario_ids
from sentinel.synthetic import generate_telemetry
from sentinel.types import TelemetryRow


class AdvancedSafetyTests(unittest.TestCase):
    def test_all_supported_drift_regimes_cross_the_threshold(self) -> None:
        matrix = run_dataset_matrix()

        self.assertEqual(set(matrix), set(drift_scenario_ids()))
        for domain, result in matrix.items():
            with self.subTest(domain=domain):
                self.assertTrue(result["detected"])
                self.assertIn("drift_detected", result["event_kinds"])

    def test_every_scenario_is_deterministic_for_a_fixed_seed(self) -> None:
        for domain in ("FD001", *drift_scenario_ids()):
            with self.subTest(domain=domain):
                first = generate_telemetry(domain, engines=2, cycles=12, seed=123)
                second = generate_telemetry(domain, engines=2, cycles=12, seed=123)
                self.assertEqual(first, second)

    def test_small_drift_batch_skips_retraining_without_losing_production(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            loop = HealingLoop(
                Path(temporary),
                HealingConfig(drift_threshold=0.2, minimum_retrain_rows=1000),
            )
            loop.bootstrap(generate_telemetry("FD001", engines=4, cycles=30))
            result = loop.process(
                generate_telemetry("FD004", engines=1, cycles=20, engine_offset=100)
            )

        self.assertEqual(result.active_version, 1)
        self.assertIsNone(result.canary_version)
        self.assertIn("retrain_skipped", [event.kind for event in result.events])

    def test_quality_gate_rejects_impossible_rul(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            loop = HealingLoop(Path(temporary))
            loop.bootstrap(generate_telemetry("FD001", engines=2, cycles=20))
            impossible = TelemetryRow(99, 1, 0.0, (1.0, 1.0, 1.0, 1.0, 1.0), 500, "FD004")
            with self.assertRaises(DataQualityError):
                loop.process([impossible])

    def test_registry_reset_removes_versions_and_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = FileModelRegistry(Path(temporary) / "registry.json")
            model = RidgeModel.fit(np.ones((3, 2)), np.array([1.0, 2.0, 3.0]))
            version = registry.register(model, {"mae": 0.0}, "candidate")
            registry.deploy_initial(version)
            registry.reset()

            snapshot = registry.snapshot()
            self.assertEqual(snapshot["aliases"], {})
            self.assertEqual(snapshot["versions"], [])
            self.assertEqual(snapshot["next_version"], 1)

    def test_model_serialization_preserves_predictions(self) -> None:
        x = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0], [4.0, 8.0]])
        y = np.array([1.0, 2.0, 4.0, 8.0])
        model = RidgeModel.fit(x, y)
        restored = RidgeModel.from_dict(model.to_dict())
        np.testing.assert_allclose(model.predict(x), restored.predict(x))

    def test_psi_rejects_feature_shape_mismatch(self) -> None:
        detector = PSIDriftDetector()
        detector.fit_reference(np.ones((20, 9)))
        with self.assertRaises(ValueError):
            detector.score(np.ones((20, 8)))

    def test_status_exposes_latest_per_feature_drift_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            loop = HealingLoop(Path(temporary))
            loop.bootstrap(generate_telemetry("FD001", engines=4, cycles=30))
            loop.process(generate_telemetry("SENSOR_BIAS", engines=2, cycles=30))
            latest = loop.status()["last_drift"]

        self.assertIsNotNone(latest)
        self.assertTrue(latest["detected"])
        self.assertEqual(set(latest["feature_psi"]), set(detector_feature_names()))


def detector_feature_names() -> tuple[str, ...]:
    from sentinel.features import FEATURE_NAMES

    return FEATURE_NAMES


if __name__ == "__main__":
    unittest.main()
