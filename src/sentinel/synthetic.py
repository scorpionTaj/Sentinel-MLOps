from __future__ import annotations

import numpy as np

from sentinel.types import TelemetryRow


def generate_telemetry(
    domain: str,
    engines: int,
    cycles: int,
    seed: int = 7,
    engine_offset: int = 0,
) -> list[TelemetryRow]:
    """Generate C-MAPSS-shaped telemetry with controllable concept drift."""

    supported = {"FD001", "FD002", "FD003", "FD004", "ADVERSARIAL"}
    if domain not in supported:
        raise ValueError(f"domain must be one of {', '.join(sorted(supported))}")
    rng = np.random.default_rng(seed)
    rows: list[TelemetryRow] = []
    for engine_index in range(engines):
        engine_id = engine_offset + engine_index + 1
        life = cycles + int(rng.integers(-5, 6))
        for cycle in range(1, cycles + 1):
            remaining = max(0.0, float(life - cycle))
            health = remaining / max(1, life)
            if domain == "FD001":
                operating = float(rng.normal(0.0, 0.04))
                sensors = (
                    health + rng.normal(0, 0.018),
                    1.8 * health + rng.normal(0, 0.025),
                    0.45 * health + rng.normal(0, 0.02),
                    0.3 + 0.5 * health + rng.normal(0, 0.02),
                    1.1 - health + rng.normal(0, 0.02),
                )
            elif domain == "FD002":
                operating = float(rng.normal(1.15, 0.08))
                sensors = (
                    0.35 + 0.62 * health + rng.normal(0, 0.018),
                    1.2 * health + 0.3 * operating + rng.normal(0, 0.025),
                    0.2 + 0.9 * health + rng.normal(0, 0.02),
                    0.6 * health + rng.normal(0, 0.02),
                    1.35 - 0.55 * health + rng.normal(0, 0.02),
                )
            elif domain == "FD003":
                operating = float(rng.normal(0.12, 0.05))
                fault = 1.0 - health
                sensors = (
                    0.08 + 0.9 * health + rng.normal(0, 0.02),
                    1.55 * health - 0.2 * fault**2 + rng.normal(0, 0.028),
                    0.18 + 0.65 * health**2 + rng.normal(0, 0.02),
                    0.2 + 0.75 * health - 0.12 * fault + rng.normal(0, 0.022),
                    1.15 - 0.72 * health + 0.15 * fault**2 + rng.normal(0, 0.022),
                )
            elif domain == "FD004":
                regime = (-1.1, 0.25, 1.35)[engine_index % 3]
                operating = float(rng.normal(regime, 0.1))
                fault = 1.0 - health
                condition = operating * 0.12
                sensors = (
                    0.22 + 0.7 * health + condition + rng.normal(0, 0.025),
                    1.35 * health + 0.28 * fault**2 + condition + rng.normal(0, 0.03),
                    0.12 + 0.78 * health**2 - condition + rng.normal(0, 0.024),
                    0.48 * health - 0.18 * fault + condition + rng.normal(0, 0.024),
                    1.28 - 0.62 * health + 0.2 * fault**2 + rng.normal(0, 0.024),
                )
            else:
                operating = float(rng.normal(1.15, 0.08))
                inverted = 1.0 - health
                sensors = (
                    0.35 + 0.62 * inverted + rng.normal(0, 0.018),
                    1.2 * inverted + 0.3 * operating + rng.normal(0, 0.025),
                    0.2 + 0.9 * inverted + rng.normal(0, 0.02),
                    0.6 * inverted + rng.normal(0, 0.02),
                    0.8 + 0.55 * health + rng.normal(0, 0.02),
                )
            rows.append(
                TelemetryRow(
                    engine_id=engine_id,
                    cycle=cycle,
                    operating_condition=operating,
                    sensors=tuple(float(value) for value in sensors),
                    rul=remaining,
                    domain=domain,
                )
            )
    return rows
