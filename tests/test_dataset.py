from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sentinel.datasets import load_cmapps_test, load_cmapps_training
from sentinel.scenarios import all_scenarios, drift_scenario_ids


class DatasetTests(unittest.TestCase):
    def test_loader_derives_rul_from_last_engine_cycle(self) -> None:
        first = [str(value) for value in range(1, 27)]
        first[1] = "1"
        second = first.copy()
        second[1] = "2"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train_FD001.txt"
            path.write_text(" ".join(first) + "\n" + " ".join(second) + "\n", encoding="utf-8")
            rows = load_cmapps_training(path, "FD001")

        self.assertEqual([row.rul for row in rows], [1.0, 0.0])
        self.assertEqual(len(rows[0].sensors), 5)

    def test_loader_rejects_malformed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "broken.txt"
            path.write_text("1 2 3\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected 26 columns"):
                load_cmapps_training(path, "FD003")

    def test_test_split_loader_applies_terminal_rul_offsets(self) -> None:
        engine_one = [str(value) for value in range(1, 27)]
        engine_one[0], engine_one[1] = "1", "1"
        engine_one_last = engine_one.copy()
        engine_one_last[1] = "2"
        engine_two = engine_one.copy()
        engine_two[0], engine_two[1] = "2", "1"
        with tempfile.TemporaryDirectory() as temporary:
            test_path = Path(temporary) / "test_FD004.txt"
            rul_path = Path(temporary) / "RUL_FD004.txt"
            test_path.write_text(
                "\n".join(map(" ".join, (engine_one, engine_one_last, engine_two))) + "\n",
                encoding="utf-8",
            )
            rul_path.write_text("10\n20\n", encoding="utf-8")
            rows = load_cmapps_test(test_path, rul_path, "FD004")

        self.assertEqual([row.rul for row in rows], [11.0, 10.0, 20.0])

    def test_catalog_separates_cmapps_domains_from_stress_scenarios(self) -> None:
        catalog = all_scenarios()
        self.assertEqual(catalog[0].id, "FD001")
        self.assertFalse(catalog[0].injectable)
        self.assertIn("FD004", drift_scenario_ids())
        self.assertIn("SENSOR_DROPOUT", drift_scenario_ids())
        self.assertIn("NOISE_BURST", drift_scenario_ids())
        self.assertIn("ADVERSARIAL", drift_scenario_ids())
        self.assertTrue(any(item.category == "stress_test" for item in catalog))

    def test_test_split_loader_validates_terminal_rul_count(self) -> None:
        row = [str(value) for value in range(1, 27)]
        row[0], row[1] = "3", "1"
        with tempfile.TemporaryDirectory() as temporary:
            test_path = Path(temporary) / "test_FD001.txt"
            rul_path = Path(temporary) / "RUL_FD001.txt"
            test_path.write_text(" ".join(row) + "\n", encoding="utf-8")
            rul_path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected 1 terminal RUL"):
                load_cmapps_test(test_path, rul_path, "FD001")

    def test_official_loader_rejects_generated_stress_scenario(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train_SENSOR_BIAS.txt"
            path.write_text(" ".join(str(value) for value in range(1, 27)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a C-MAPSS domain"):
                load_cmapps_training(path, "SENSOR_BIAS")
