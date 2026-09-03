from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sentinel.cli import run_demo
from sentinel.loop import HealingLoop
from sentinel.quality import DataQualityError
from sentinel.synthetic import generate_telemetry
from sentinel.types import TelemetryRow


class ClosedLoopTests(unittest.TestCase):
    def test_drift_retrains_and_promotes_a_better_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = run_demo(Path(temporary), "promote")

        self.assertEqual(result["status"]["aliases"], {"production": 2})
        registry_events = [event["kind"] for event in result["status"]["events"]]
        self.assertEqual(
            registry_events,
            ["registered", "initial_deploy", "registered", "canary_started", "promoted"],
        )

    def test_injected_canary_regression_rolls_back(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = run_demo(Path(temporary), "rollback")

        self.assertEqual(result["status"]["aliases"], {"production": 1})
        registry_events = [event["kind"] for event in result["status"]["events"]]
        self.assertIn("regression_injected", registry_events)
        self.assertEqual(registry_events[-1], "rolled_back")

    def test_quality_gate_blocks_non_finite_sensor_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            loop = HealingLoop(Path(temporary))
            loop.bootstrap(generate_telemetry("FD001", engines=2, cycles=20))
            corrupt = TelemetryRow(
                engine_id=99,
                cycle=1,
                operating_condition=0.0,
                sensors=(float("nan"), 1.0, 1.0, 1.0, 1.0),
                rul=20,
                domain="FD002",
            )
            with self.assertRaises(DataQualityError):
                loop.process([corrupt])
            self.assertIn("sentinel_quality_rejections_total 1.0", loop.metrics.render())

    def test_registry_restart_rebuilds_reference_without_new_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state_dir = Path(temporary)
            reference = generate_telemetry("FD001", engines=2, cycles=20)
            first = HealingLoop(state_dir)
            self.assertEqual(first.bootstrap(reference), 1)
            restarted = HealingLoop(state_dir)
            self.assertEqual(restarted.bootstrap(reference), 1)
            self.assertEqual(len(restarted.registry.snapshot()["versions"]), 1)


if __name__ == "__main__":
    unittest.main()

