from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sentinel.adapters.mlflow import MLflowModelRegistry
from sentinel.adapters.orchestrator import HealingSensor
from sentinel.adapters.streaming import MedallionLakehouse, StreamingBatchConsumer
from sentinel.features import RollingFeaturePipeline
from sentinel.loop import HealingConfig, HealingLoop
from sentinel.model import RidgeModel
from sentinel.quality import QualityGate
from sentinel.registry import FileModelRegistry, ModelRegistryProtocol
from sentinel.synthetic import generate_telemetry


class TestAdapters(unittest.TestCase):
    def test_registry_protocol_conformance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_reg = FileModelRegistry(Path(tmp_dir) / "reg.json")
            mlflow_reg = MLflowModelRegistry(local_store_path=Path(tmp_dir) / "mlflow.json")

            self.assertIsInstance(file_reg, ModelRegistryProtocol)
            self.assertIsInstance(mlflow_reg, ModelRegistryProtocol)

    def test_mlflow_model_registry_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg = MLflowModelRegistry(local_store_path=Path(tmp_dir) / "mlflow.json")
            m1 = RidgeModel(mean=(0.0, 0.0), scale=(1.0, 1.0), coefficients=(0.5, 0.2), intercept=1.0)
            v1 = reg.register(m1, {"mae": 1.5}, "trained")
            self.assertEqual(v1, 1)

            reg.deploy_initial(v1)
            self.assertEqual(reg.version("production"), 1)
            self.assertEqual(reg.model("production").intercept, 1.0)

            m2 = RidgeModel(mean=(0.0, 0.0), scale=(1.0, 1.0), coefficients=(0.4, 0.3), intercept=0.8)
            v2 = reg.register(m2, {"mae": 0.8}, "canary")
            self.assertEqual(v2, 2)

            reg.start_canary(v2)
            self.assertEqual(reg.version("canary"), 2)

            promoted = reg.promote()
            self.assertEqual(promoted, 2)
            self.assertEqual(reg.version("production"), 2)

            reg.reset()
            self.assertIsNone(reg.version("production"))

    def test_medallion_lakehouse(self) -> None:
        lakehouse = MedallionLakehouse()
        payloads = [
            {
                "engine_id": 1,
                "cycle": 1,
                "op_setting_1": 0.0,
                "op_setting_2": 0.0,
                "op_setting_3": 100.0,
                "sensor_1": 518.67,
                "sensor_2": 641.82,
                "sensor_3": 1589.70,
                "sensor_4": 1400.60,
                "sensor_5": 14.62,
                "sensor_6": 21.61,
                "sensor_7": 554.36,
                "sensor_8": 2388.06,
                "sensor_9": 9046.19,
                "sensor_10": 1.30,
                "sensor_11": 47.47,
                "sensor_12": 521.66,
                "sensor_13": 2388.02,
                "sensor_14": 8138.62,
                "sensor_15": 8.4195,
                "sensor_16": 0.03,
                "sensor_17": 392.0,
                "sensor_18": 2388.0,
                "sensor_19": 100.0,
                "sensor_20": 39.06,
                "sensor_21": 23.4190,
                "rul": 112.0,
            }
        ]

        offsets = lakehouse.append_bronze(payloads)
        self.assertEqual(offsets, [0])
        self.assertEqual(len(lakehouse.bronze), 1)

        silver_rows = lakehouse.process_silver()
        self.assertEqual(len(silver_rows), 1)
        self.assertEqual(silver_rows[0].engine_id, 1)

        pipeline = RollingFeaturePipeline(window=1)
        quality = QualityGate()
        gold_rows = lakehouse.materialize_gold(pipeline, quality)
        self.assertEqual(len(gold_rows), 1)
        self.assertEqual(len(gold_rows[0].values), 9)

    def test_orchestrator_sensor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            loop = HealingLoop(state_dir=Path(tmp_dir), config=HealingConfig())
            consumer = StreamingBatchConsumer()
            sensor = HealingSensor(loop=loop, consumer=consumer, min_batch_size=50)

            # Not enough records -> sensor skips
            res_skip = sensor.tick()
            self.assertFalse(res_skip.triggered)

            # Bootstrap loop with reference distribution
            baseline_rows = generate_telemetry("FD001", engines=4, cycles=30, seed=42)
            loop.bootstrap(baseline_rows)

            # Ingest sufficient streaming records
            stream_rows = generate_telemetry("FD002", engines=2, cycles=30, seed=43)
            consumer.publish(stream_rows)

            res_triggered = sensor.tick()
            self.assertTrue(res_triggered.triggered)
            self.assertEqual(res_triggered.batch_size, 50)
            self.assertIsNotNone(res_triggered.active_production_version)


if __name__ == "__main__":
    unittest.main()
