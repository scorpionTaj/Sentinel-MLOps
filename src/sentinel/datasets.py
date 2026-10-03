from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sentinel.scenarios import get_scenario
from sentinel.types import TelemetryRow


def _validate_domain(domain: str) -> None:
    if get_scenario(domain).category != "cmapps":
        raise ValueError(f"{domain} is a generated stress scenario, not a C-MAPSS domain")


def _parse_rows(path: Path) -> tuple[list[tuple[int, int, list[float]]], dict[int, int]]:
    parsed: list[tuple[int, int, list[float]]] = []
    maximum_cycle: dict[int, int] = defaultdict(int)
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if len(values) != 26:
            raise ValueError(f"{path}:{line_number}: expected 26 columns, found {len(values)}")
        engine_id, cycle = int(values[0]), int(values[1])
        if engine_id < 1 or cycle < 1:
            raise ValueError(f"{path}:{line_number}: engine id and cycle must be positive")
        parsed.append((engine_id, cycle, values))
        maximum_cycle[engine_id] = max(maximum_cycle[engine_id], cycle)
    if not parsed:
        raise ValueError(f"{path}: dataset is empty")
    return parsed, maximum_cycle


def _telemetry_rows(
    parsed: list[tuple[int, int, list[float]]],
    domain: str,
    end_of_life: dict[int, float],
) -> list[TelemetryRow]:
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
                rul=float(end_of_life[engine_id] - cycle),
                domain=domain,
            )
        )
    return rows


def load_cmapps_training(path: Path, domain: str) -> list[TelemetryRow]:
    """Load a NASA C-MAPSS train_FD00x.txt file and derive per-cycle RUL."""

    _validate_domain(domain)
    parsed, maximum_cycle = _parse_rows(path)
    return _telemetry_rows(
        parsed, domain, {engine_id: float(cycle) for engine_id, cycle in maximum_cycle.items()}
    )


def load_cmapps_test(test_path: Path, rul_path: Path, domain: str) -> list[TelemetryRow]:
    """Load a NASA C-MAPSS test_FD00x.txt file with RUL_FD00x.txt offsets."""

    _validate_domain(domain)
    ruls = [
        float(line.strip())
        for line in rul_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    parsed, maximum_cycle = _parse_rows(test_path)
    engine_ids = sorted(maximum_cycle)
    if len(ruls) != len(engine_ids):
        raise ValueError(
            f"{rul_path}: expected {len(engine_ids)} terminal RUL values, found {len(ruls)}"
        )
    end_of_life = {
        engine_id: float(maximum_cycle[engine_id]) + terminal_rul
        for engine_id, terminal_rul in zip(engine_ids, ruls, strict=True)
    }
    return _telemetry_rows(parsed, domain, end_of_life)


@dataclass(frozen=True)
class CMAPSSArrays:
    """All 26 C-MAPSS columns as arrays, for offline research (the live loop uses TelemetryRow)."""

    domain: str
    engine: np.ndarray
    cycle: np.ndarray
    settings: np.ndarray
    sensors: np.ndarray
    rul: np.ndarray

    def __len__(self) -> int:
        return len(self.engine)

    def last_cycle_mask(self) -> np.ndarray:
        """True on each engine's final observed row (the official test-set evaluation point)."""

        is_last = np.ones(len(self.engine), dtype=bool)
        is_last[:-1] = self.engine[1:] != self.engine[:-1]
        return is_last

    def subset(self, mask: np.ndarray) -> CMAPSSArrays:
        return CMAPSSArrays(
            self.domain,
            self.engine[mask],
            self.cycle[mask],
            self.settings[mask],
            self.sensors[mask],
            self.rul[mask],
        )


def _load_matrix(path: Path) -> np.ndarray:
    matrix = np.loadtxt(path, ndmin=2)
    if matrix.size == 0:
        raise ValueError(f"{path}: dataset is empty")
    if matrix.shape[1] != 26:
        raise ValueError(f"{path}: expected 26 columns, found {matrix.shape[1]}")
    return matrix


def load_cmapps_arrays(
    path: Path, domain: str, rul_path: Path | None = None
) -> CMAPSSArrays:
    """Load a train (RUL from last cycle) or test (RUL from RUL_FD00x.txt) file as arrays."""

    _validate_domain(domain)
    matrix = _load_matrix(path)
    engine = matrix[:, 0].astype(int)
    cycle = matrix[:, 1].astype(int)
    if np.any(np.diff(engine) < 0):
        raise ValueError(f"{path}: rows must be grouped by ascending engine id")
    engines = np.unique(engine)
    max_cycle = {e: int(cycle[engine == e].max()) for e in engines}
    if rul_path is None:
        offsets = dict.fromkeys(engines, 0.0)
    else:
        terminal = np.loadtxt(rul_path, ndmin=1)
        if len(terminal) != len(engines):
            raise ValueError(
                f"{rul_path}: expected {len(engines)} terminal RUL values, found {len(terminal)}"
            )
        offsets = dict(zip(engines, terminal.astype(float), strict=True))
    end_of_life = np.array([max_cycle[e] + offsets[e] for e in engine], dtype=float)
    return CMAPSSArrays(
        domain=domain,
        engine=engine,
        cycle=cycle,
        settings=matrix[:, 2:5],
        sensors=matrix[:, 5:],
        rul=end_of_life - cycle,
    )
