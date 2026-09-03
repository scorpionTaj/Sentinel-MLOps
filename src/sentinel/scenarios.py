from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Scenario:
    id: str
    name: str
    category: str
    injectable: bool
    description: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        id="FD001",
        name="FD001 baseline",
        category="cmapps",
        injectable=False,
        description="Sea-level single condition baseline dataset.",
    ),
    Scenario(
        id="FD002",
        name="FD002 multi-condition",
        category="cmapps",
        injectable=True,
        description="Six operational condition regimes introducing operating drift.",
    ),
    Scenario(
        id="FD003",
        name="FD003 dual-fault",
        category="cmapps",
        injectable=True,
        description="HPC and fan degradation modes introducing fault drift.",
    ),
    Scenario(
        id="FD004",
        name="FD004 multi-condition + dual-fault",
        category="cmapps",
        injectable=True,
        description="Combination of operational condition regimes and dual fault modes.",
    ),
    Scenario(
        id="SENSOR_BIAS",
        name="Sensor bias",
        category="stress_test",
        injectable=True,
        description="Systematic sensor measurement offset inducing distributional drift.",
    ),
    Scenario(
        id="SENSOR_DROPOUT",
        name="Sensor dropout",
        category="stress_test",
        injectable=True,
        description="Severe telemetry signal attenuation simulating degraded sensing hardware.",
    ),
)


def all_scenarios() -> tuple[Scenario, ...]:
    return SCENARIOS


def drift_scenario_ids() -> tuple[str, ...]:
    return tuple(scenario.id for scenario in SCENARIOS if scenario.injectable)
