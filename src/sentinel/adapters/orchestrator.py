from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sentinel.adapters.streaming import StreamingBatchConsumer
from sentinel.loop import HealingLoop
from sentinel.types import TelemetryRow


@dataclass(slots=True)
class SensorExecutionResult:
    """Execution report emitted by the orchestrator sensor tick."""

    triggered: bool
    batch_size: int
    drift_score: float | None
    retrained: bool
    promoted: bool
    rolled_back: bool
    active_production_version: int | None
    active_canary_version: int | None
    details: dict[str, Any]


class HealingSensor:
    """Orchestrator sensor adapter (Dagster/Airflow sensor seam).

    Monitors incoming streaming telemetry or feature store partitions.
    When a batch of adequate size is ready, the sensor ticks, feeds
    the batch to the HealingLoop, and reports downstream orchestration actions.
    """

    def __init__(
        self,
        loop: HealingLoop,
        consumer: StreamingBatchConsumer,
        min_batch_size: int = 50,
    ) -> None:
        self.loop = loop
        self.consumer = consumer
        self.min_batch_size = min_batch_size
        self.total_ticks: int = 0
        self.total_triggers: int = 0

    def tick(self) -> SensorExecutionResult:
        """Evaluates sensor condition and triggers self-healing execution if batch is ready."""
        self.total_ticks += 1

        if self.consumer.pending_count < self.min_batch_size:
            status = self.loop.status()
            aliases = status.get("aliases", {})
            return SensorExecutionResult(
                triggered=False,
                batch_size=self.consumer.pending_count,
                drift_score=None,
                retrained=False,
                promoted=False,
                rolled_back=False,
                active_production_version=aliases.get("production"),
                active_canary_version=aliases.get("canary"),
                details={"reason": f"Insufficient batch size ({self.consumer.pending_count}/{self.min_batch_size})"},
            )

        batch: list[TelemetryRow] = self.consumer.poll_batch(self.min_batch_size)
        initial_status = self.loop.status()
        initial_counters = initial_status.get("metrics", {}).get("counters", {})

        # Process through self-healing loop
        self.loop.process(batch)
        self.total_triggers += 1

        updated_status = self.loop.status()
        updated_counters = updated_status.get("metrics", {}).get("counters", {})
        updated_gauges = updated_status.get("metrics", {}).get("gauges", {})
        aliases = updated_status.get("aliases", {})

        retrained = (
            updated_counters.get("sentinel_retrains_total", 0)
            > initial_counters.get("sentinel_retrains_total", 0)
        )
        promoted = (
            updated_counters.get("sentinel_promotions_total", 0)
            > initial_counters.get("sentinel_promotions_total", 0)
        )
        rolled_back = (
            updated_counters.get("sentinel_rollbacks_total", 0)
            > initial_counters.get("sentinel_rollbacks_total", 0)
        )

        return SensorExecutionResult(
            triggered=True,
            batch_size=len(batch),
            drift_score=updated_gauges.get("sentinel_drift_psi"),
            retrained=retrained,
            promoted=promoted,
            rolled_back=rolled_back,
            active_production_version=aliases.get("production"),
            active_canary_version=aliases.get("canary"),
            details={
                "events_count": len(updated_status.get("events", [])),
                "gold_rows": updated_status.get("gold_rows", 0),
            },
        )
