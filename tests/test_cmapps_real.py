from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from sentinel.datasets import load_cmapps_test, load_cmapps_training
from sentinel.loop import HealingConfig, HealingLoop


class TestRealCMAPSSDataset(unittest.TestCase):
    """Stress-test Sentinel on the large, complex real-world NASA C-MAPSS dataset.

    Verifies scalability and correctness when exposed to:
    - 20,631 real operational cycles in train_FD001.txt
    - 53,759 real operational cycles across 6 flight conditions in train_FD002.txt
    - 13,096 test cycles in test_FD001.txt with RUL_FD001.txt ground truth offsets
    """

    def setUp(self) -> None:
        self.ref_dir = Path("data/reference")
        self.train_fd001 = self.ref_dir / "train_FD001.txt"
        self.train_fd002 = self.ref_dir / "train_FD002.txt"
        self.test_fd001 = self.ref_dir / "test_FD001.txt"
        self.rul_fd001 = self.ref_dir / "RUL_FD001.txt"

        if not self.train_fd001.exists() or not self.train_fd002.exists():
            self.skipTest("NASA C-MAPSS data not found in data/reference/. Skipping real-data stress test.")

    def test_large_scale_training_and_drift_adaptation(self) -> None:
        # 1. Load the full 20,631-row reference dataset
        t0 = time.perf_counter()
        fd001_rows = load_cmapps_training(self.train_fd001, "FD001")
        t_load_fd001 = time.perf_counter() - t0
        self.assertEqual(len(fd001_rows), 20631)
        self.assertLess(t_load_fd001, 2.0, "Loading 20k rows should take under 2 seconds")

        # 2. Load the full 53,759-row multi-condition dataset
        t0 = time.perf_counter()
        fd002_rows = load_cmapps_training(self.train_fd002, "FD002")
        t_load_fd002 = time.perf_counter() - t0
        self.assertEqual(len(fd002_rows), 53759)
        self.assertLess(t_load_fd002, 4.0, "Loading 53k rows should take under 4 seconds")

        with tempfile.TemporaryDirectory() as tmp_dir:
            state_dir = Path(tmp_dir)
            loop = HealingLoop(
                state_dir=state_dir,
                config=HealingConfig(
                    drift_threshold=0.20,
                    canary_min_observations=150,
                    minimum_retrain_rows=100,
                ),
            )

            # 3. Bootstrap baseline production model on 20,631 real rows
            t0 = time.perf_counter()
            initial_ver = loop.bootstrap(fd001_rows)
            t_boot = time.perf_counter() - t0

            self.assertEqual(initial_ver, 1)
            self.assertLess(t_boot, 1.5, "Bootstrapping 20,631 rows in NumPy should be sub-second")
            status = loop.status()
            self.assertEqual(status["aliases"]["production"], 1)
            self.assertGreater(status["gold_rows"], 20000)

            # 4. Inject complex real-world flight regime shift (FD002)
            # 6 operating conditions across altitude, Mach, and throttle
            drift_batch = fd002_rows[:500]
            t0 = time.perf_counter()
            drift_result = loop.process(drift_batch)
            t_proc = time.perf_counter() - t0

            self.assertLess(t_proc, 0.25, "Processing 500 real telemetry cycles should take under 250ms")
            self.assertTrue(drift_result.drift.detected, "FD002 flight regime shift must trigger PSI drift")
            self.assertGreater(drift_result.drift.aggregate_psi, 0.20)

            # Candidate v2 must have been trained and canary started or promoted
            self.assertIn(
                "canary_started",
                [event.kind for event in drift_result.events],
                "Valid candidate must pass offline gate and enter canary evaluation",
            )

            # 5. Subsequent multi-condition observations evaluate and promote canary
            followup_batch = fd002_rows[500:1000]
            followup_result = loop.process(followup_batch)
            self.assertIsNotNone(followup_result.active_version)

            # Check that production version exists and metrics are sound
            latest_status = loop.status()
            active_prod = latest_status["aliases"]["production"]
            self.assertIsNotNone(active_prod)
            self.assertGreaterEqual(active_prod, 1)

    def test_real_test_partition_with_true_rul_offsets(self) -> None:
        if not self.test_fd001.exists() or not self.rul_fd001.exists():
            self.skipTest("test_FD001.txt or RUL_FD001.txt missing")

        test_rows = load_cmapps_test(self.test_fd001, self.rul_fd001, "FD001")
        self.assertEqual(len(test_rows), 13096)

        # Verify RUL offsets derived properly: positive and finite
        for r in test_rows[:100]:
            self.assertGreaterEqual(r.rul, 0.0)
            self.assertLessEqual(r.rul, 400.0)


if __name__ == "__main__":
    unittest.main()
