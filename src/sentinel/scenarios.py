from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class Scenario:
    id: str
    name: str
    category: str
    injectable: bool
    seed: int
    description: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        id="FD001",
        name="FD001 baseline",
        category="cmapps",
        injectable=False,
        seed=10,
        description="Sea-level single condition baseline dataset.",
    ),
    Scenario(
        id="FD002",
        name="FD002 multi-condition",
        category="cmapps",
        injectable=True,
        seed=99,
        description="Six operational condition regimes introducing operating drift.",
    ),
    Scenario(
        id="FD003",
        name="FD003 dual-fault",
        category="cmapps",
        injectable=True,
        seed=199,
        description="HPC and fan degradation modes introducing fault drift.",
    ),
    Scenario(
        id="FD004",
        name="FD004 multi-condition + dual-fault",
        category="cmapps",
        injectable=True,
        seed=299,
        description="Combination of operational condition regimes and dual fault modes.",
    ),
    Scenario(
        id="SENSOR_BIAS",
        name="Sensor bias",
        category="stress_test",
        injectable=True,
        seed=399,
        description="Systematic sensor measurement offset inducing distributional drift.",
    ),
    Scenario(
        id="SENSOR_DROPOUT",
        name="Sensor dropout",
        category="stress_test",
        injectable=True,
        seed=499,
        description="Severe telemetry signal attenuation simulating degraded sensing hardware.",
    ),
    Scenario(
        id="NOISE_BURST",
        name="Noise burst",
        category="stress_test",
        injectable=True,
        seed=599,
        description="High-variance measurements simulating a noisy acquisition window.",
    ),
    Scenario(
        id="ADVERSARIAL",
        name="Adversarial inversion",
        category="stress_test",
        injectable=True,
        seed=699,
        description="Inverted health relationships exercise offline and live safety gates.",
    ),
)


def all_scenarios() -> tuple[Scenario, ...]:
    return SCENARIOS


def drift_scenario_ids() -> tuple[str, ...]:
    return tuple(scenario.id for scenario in SCENARIOS if scenario.injectable)


def drift_scenarios() -> tuple[Scenario, ...]:
    return tuple(scenario for scenario in SCENARIOS if scenario.injectable)


def get_scenario(scenario_id: str) -> Scenario:
    for scenario in SCENARIOS:
        if scenario.id == scenario_id:
            return scenario
    supported = ", ".join(scenario.id for scenario in SCENARIOS)
    raise ValueError(f"domain must be one of {supported}")
