from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from sentinel.registry import FileModelRegistry


@dataclass(frozen=True, slots=True)
class Prediction:
    value: float
    model_version: int
    route: str
    decision: str | None = None


class CanaryRouter:
    """Routes traffic and hides shadow evaluation plus release decisions."""

    def __init__(
        self,
        registry: FileModelRegistry,
        weight: float,
        min_observations: int,
        regression_tolerance: float,
    ) -> None:
        self.registry = registry
        self.weight = weight
        self.min_observations = min_observations
        self.regression_tolerance = regression_tolerance
        self._production_errors: list[float] = []
        self._canary_errors: list[float] = []

    def predict(
        self, values: tuple[float, ...], request_id: str, actual: float | None = None
    ) -> Prediction:
        matrix = np.asarray([values], dtype=float)
        production = self.registry.model("production")
        production_version = self.registry.version("production")
        if production_version is None:
            raise RuntimeError("no production model is deployed")
        production_value = float(production.predict(matrix)[0])
        canary_version = self.registry.version("canary")
        if canary_version is None:
            return Prediction(production_value, production_version, "production")

        canary = self.registry.model("canary")
        canary_value = float(canary.predict(matrix)[0])
        if actual is not None:
            self._production_errors.append(abs(production_value - actual))
            self._canary_errors.append(abs(canary_value - actual))

        bucket = int.from_bytes(hashlib.sha256(request_id.encode()).digest()[:8], "big") / 2**64
        use_canary = bucket < self.weight
        decision = self._decide_if_ready()
        if use_canary:
            return Prediction(canary_value, canary_version, "canary", decision)
        return Prediction(production_value, production_version, "production", decision)

    def reset_evidence(self) -> None:
        self._production_errors.clear()
        self._canary_errors.clear()

    def _decide_if_ready(self) -> str | None:
        if len(self._canary_errors) < self.min_observations:
            return None
        production_mae = float(np.mean(self._production_errors))
        canary_mae = float(np.mean(self._canary_errors))
        self.reset_evidence()
        if canary_mae <= production_mae * (1 + self.regression_tolerance):
            self.registry.promote()
            return "promoted"
        self.registry.rollback()
        return "rolled_back"

