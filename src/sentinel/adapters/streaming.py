from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from sentinel.features import RollingFeaturePipeline
from sentinel.quality import QualityGate
from sentinel.types import FeatureRow, TelemetryRow


@dataclass(slots=True)
class BronzeRecord:
    """Raw ingestion record representing an unvalidated Kafka/stream event."""

    offset: int
    timestamp: float
    payload: dict[str, Any]


class MedallionLakehouse:
    """In-memory Medallion (Bronze/Silver/Gold) Lakehouse simulator.

    Demonstrates the production data architecture seam:
    - Bronze: Raw event stream ingestion directly from Kafka topics.
    - Silver: Conformed, typed, and schema-validated TelemetryRows.
    - Gold: Rolling feature engineered matrix validated by DataQualityGate.
    """

    def __init__(self) -> None:
        self.bronze: list[BronzeRecord] = []
        self.silver: list[TelemetryRow] = []
        self._next_offset = 0

    def append_bronze(self, payloads: list[dict[str, Any]]) -> list[int]:
        """Appends raw event payloads to Bronze layer."""
        offsets: list[int] = []
        now = time.time()
        for payload in payloads:
            record = BronzeRecord(offset=self._next_offset, timestamp=now, payload=payload)
            self.bronze.append(record)
            offsets.append(self._next_offset)
            self._next_offset += 1
        return offsets

    def process_silver(self) -> list[TelemetryRow]:
        """Conforms unprocessed Bronze records into Silver TelemetryRows."""
        conformed: list[TelemetryRow] = []
        processed_count = len(self.silver)
        unprocessed = self.bronze[processed_count:]

        for rec in unprocessed:
            p = rec.payload
            raw_sensors = p.get("sensors")
            if raw_sensors and isinstance(raw_sensors, (list, tuple)) and len(raw_sensors) >= 5:
                sensors = (
                    float(raw_sensors[0]),
                    float(raw_sensors[1]),
                    float(raw_sensors[2]),
                    float(raw_sensors[3]),
                    float(raw_sensors[4]),
                )
            else:
                sensors = (
                    float(p.get("sensor_1", 0.0)),
                    float(p.get("sensor_2", 0.0)),
                    float(p.get("sensor_3", 0.0)),
                    float(p.get("sensor_4", 0.0)),
                    float(p.get("sensor_5", 0.0)),
                )
            row = TelemetryRow(
                engine_id=int(p.get("engine_id", 1)),
                cycle=int(p.get("cycle", 1)),
                operating_condition=float(p.get("operating_condition", p.get("op_setting_1", 0.0))),
                sensors=sensors,
                rul=float(p.get("rul", 0.0)),
                domain=str(p.get("domain", "STREAM")),
            )
            conformed.append(row)
            self.silver.append(row)

        return conformed

    def materialize_gold(
        self, pipeline: RollingFeaturePipeline, quality: QualityGate
    ) -> list[FeatureRow]:
        """Materializes validated Gold feature rows."""
        if not self.silver:
            return []

        feature_rows = pipeline.transform(self.silver)
        quality.validate(feature_rows)
        return feature_rows


@dataclass
class StreamingBatchConsumer:
    """Consumer polling abstraction for batching streaming telemetry rows."""

    queue: list[TelemetryRow] = field(default_factory=list)
    committed_offset: int = 0

    def publish(self, rows: list[TelemetryRow]) -> None:
        self.queue.extend(rows)

    def poll_batch(self, batch_size: int = 50) -> list[TelemetryRow]:
        if not self.queue:
            return []
        batch = self.queue[:batch_size]
        self.queue = self.queue[batch_size:]
        self.committed_offset += len(batch)
        return batch

    @property
    def pending_count(self) -> int:
        return len(self.queue)
