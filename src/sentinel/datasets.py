from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from sentinel.types import TelemetryRow


def load_cmapps_training(path: Path, domain: str) -> list[TelemetryRow]:
    """Load a NASA C-MAPSS train_FD00x.txt file and derive per-cycle RUL."""

    parsed: list[tuple[int, int, list[float]]] = []
    maximum_cycle: dict[int, int] = defaultdict(int)
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if len(values) != 26:
            raise ValueError(f"{path}:{line_number}: expected 26 columns, found {len(values)}")
        engine_id, cycle = int(values[0]), int(values[1])
        parsed.append((engine_id, cycle, values))
        maximum_cycle[engine_id] = max(maximum_cycle[engine_id], cycle)

    rows: list[TelemetryRow] = []
    for engine_id, cycle, values in parsed:
        sensor_values = values[5:]
        selected = tuple(sensor_values[index] for index in (1, 2, 3, 6, 10))
        rows.append(
            TelemetryRow(
                engine_id=engine_id,
                cycle=cycle,
                operating_condition=values[2],
                sensors=selected,  # type: ignore[arg-type]
                rul=float(maximum_cycle[engine_id] - cycle),
                domain=domain,
            )
        )
    return rows


def load_cmapps_test(test_path: Path, rul_path: Path, domain: str) -> list[TelemetryRow]:
    """Load a NASA C-MAPSS test_FD00x.txt file with RUL_FD00x.txt offsets."""

    ruls = [
        float(line.strip())
        for line in rul_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    parsed: list[tuple[int, int, list[float]]] = []
    maximum_cycle: dict[int, int] = defaultdict(int)
    for line_number, line in enumerate(
        test_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if len(values) != 26:
            raise ValueError(f"{test_path}:{line_number}: expected 26 columns, found {len(values)}")
        engine_id, cycle = int(values[0]), int(values[1])
        parsed.append((engine_id, cycle, values))
        maximum_cycle[engine_id] = max(maximum_cycle[engine_id], cycle)

    rows: list[TelemetryRow] = []
    for engine_id, cycle, values in parsed:
        sensor_values = values[5:]
        selected = tuple(sensor_values[index] for index in (1, 2, 3, 6, 10))
        terminal_rul = ruls[engine_id - 1]
        rul = float(terminal_rul + (maximum_cycle[engine_id] - cycle))
        rows.append(
            TelemetryRow(
                engine_id=engine_id,
                cycle=cycle,
                operating_condition=values[2],
                sensors=selected,  # type: ignore[arg-type]
                rul=rul,
                domain=domain,
            )
        )
    return rows

