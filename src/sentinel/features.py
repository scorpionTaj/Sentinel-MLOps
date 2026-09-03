from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from sentinel.types import FeatureRow, TelemetryRow

FEATURE_NAMES = (
    "cycle_scaled",
    "operating_condition",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_1_rolling_mean",
    "sensor_3_rolling_mean",
)


class RollingFeaturePipeline:
    """Turns raw telemetry into model-ready rows while hiding rolling state."""

    def __init__(self, window: int = 5) -> None:
        if window < 1:
            raise ValueError("window must be positive")
        self._window = window
        self._sensor_1: dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self._sensor_3: dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=window))

    def transform(self, rows: list[TelemetryRow]) -> list[FeatureRow]:
        features: list[FeatureRow] = []
        for row in rows:
            self._sensor_1[row.engine_id].append(row.sensors[0])
            self._sensor_3[row.engine_id].append(row.sensors[2])
            values = (
                row.cycle / 150.0,
                row.operating_condition,
                *row.sensors,
                float(np.mean(self._sensor_1[row.engine_id])),
                float(np.mean(self._sensor_3[row.engine_id])),
            )
            features.append(
                FeatureRow(
                    request_id=f"{row.domain}-{row.engine_id}-{row.cycle}",
                    values=tuple(float(value) for value in values),
                    target=row.rul,
                    domain=row.domain,
                )
            )
        return features

