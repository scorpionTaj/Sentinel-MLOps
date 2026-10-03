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

# Drift is monitored on covariates only. `cycle_scaled` is a time index: a batch of shorter
# engine histories shifts its distribution without any change in the data-generating process.
MONITORED_FEATURES = tuple(name for name in FEATURE_NAMES if name != "cycle_scaled")
MONITORED_INDEX = tuple(FEATURE_NAMES.index(name) for name in MONITORED_FEATURES)


class RollingFeaturePipeline:
    """Turns raw telemetry into model-ready rows while hiding rolling state."""

    def __init__(self, window: int = 5) -> None:
        if window < 1:
            raise ValueError("window must be positive")
        self._window = window
        # Keyed by (domain, engine_id): engine numbers restart at 1 in every C-MAPSS file.
        self._sensor_1: dict[tuple[str, int], deque[float]] = defaultdict(
            lambda: deque(maxlen=window)
        )
        self._sensor_3: dict[tuple[str, int], deque[float]] = defaultdict(
            lambda: deque(maxlen=window)
        )

    def transform(self, rows: list[TelemetryRow]) -> list[FeatureRow]:
        features: list[FeatureRow] = []
        for row in rows:
            key = (row.domain, row.engine_id)
            self._sensor_1[key].append(row.sensors[0])
            self._sensor_3[key].append(row.sensors[2])
            values = (
                row.cycle / 150.0,
                row.operating_condition,
                *row.sensors,
                float(np.mean(self._sensor_1[key])),
                float(np.mean(self._sensor_3[key])),
            )
            features.append(
                FeatureRow(
                    request_id=f"{row.domain}-{row.engine_id}-{row.cycle}",
                    values=tuple(float(value) for value in values),
                    target=row.rul,
                    domain=row.domain,
                    group=f"{row.domain}-{row.engine_id}",
                )
            )
        return features

