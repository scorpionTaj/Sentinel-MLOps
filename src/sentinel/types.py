from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

ReleaseState = Literal["candidate", "canary", "production", "rejected", "rolled_back"]


@dataclass(frozen=True, slots=True)
class TelemetryRow:
    engine_id: int
    cycle: int
    operating_condition: float
    sensors: tuple[float, float, float, float, float]
    rul: float
    domain: str


@dataclass(frozen=True, slots=True)
class FeatureRow:
    request_id: str
    values: tuple[float, ...]
    target: float
    domain: str
    group: str = ""


@dataclass(frozen=True, slots=True)
class GateReport:
    accepted: bool
    checked_rows: int
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DriftReport:
    detected: bool
    aggregate_psi: float
    feature_psi: dict[str, float]
    threshold: float
    feature_shift: dict[str, float] = field(default_factory=dict)
    warning: bool = False
    noise_floor: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PipelineEvent:
    sequence: int
    kind: str
    detail: str
    value: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BatchResult:
    quality: GateReport
    drift: DriftReport
    active_version: int
    canary_version: int | None
    events: tuple[PipelineEvent, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

