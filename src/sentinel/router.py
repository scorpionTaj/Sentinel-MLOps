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
        max_observations: int | None = None,
        z_score: float = 1.96,
    ) -> None:
        self.registry = registry
        self.weight = weight
        self.min_observations = min_observations
        self.max_observations = max_observations or 4 * min_observations
        self.regression_tolerance = regression_tolerance
        self.z_score = z_score
        self._production_errors: list[float] = []
        self._canary_errors: list[float] = []
        self.last_evidence: dict[str, float | int | str] | None = None

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
        """Paired non-inferiority test on shadow absolute errors.

        With d_i = |canary_i| - |production_i| and margin = tolerance * production MAE:
        promote when the upper confidence bound of mean(d) is within the margin, roll back when
        the lower bound exceeds it, and otherwise keep collecting evidence. If the evidence is
        still inconclusive at `max_observations`, the canary is rolled back (production is kept).
        """
        observations = len(self._canary_errors)
        if observations < self.min_observations:
            return None
        production = np.asarray(self._production_errors)
        differences = np.asarray(self._canary_errors) - production
        mean_difference = float(differences.mean())
        standard_error = float(differences.std(ddof=1) / np.sqrt(observations))
        margin = self.regression_tolerance * float(production.mean())
        upper = mean_difference + self.z_score * standard_error
        lower = mean_difference - self.z_score * standard_error
        if upper <= margin:
            outcome = "promoted"
        elif lower > margin or observations >= self.max_observations:
            outcome = "rolled_back"
        else:
            return None
        self.last_evidence = {
            "observations": observations,
            "production_mae": float(production.mean()),
            "canary_mae": float(np.mean(self._canary_errors)),
            "mean_difference": mean_difference,
            "ci_lower": lower,
            "ci_upper": upper,
            "margin": margin,
            "reason": "inconclusive" if lower <= margin < upper else outcome,
        }
        self.reset_evidence()
        if outcome == "promoted":
            self.registry.promote()
        else:
            self.registry.rollback()
        return outcome
