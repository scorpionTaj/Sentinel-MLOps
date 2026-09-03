from __future__ import annotations

import math

from sentinel.features import FEATURE_NAMES
from sentinel.types import FeatureRow, GateReport


class DataQualityError(ValueError):
    def __init__(self, report: GateReport) -> None:
        super().__init__("; ".join(report.failures))
        self.report = report


class QualityGate:
    """Enforces the Gold-table contract at a single seam."""

    def validate(self, rows: list[FeatureRow]) -> GateReport:
        failures: list[str] = []
        if not rows:
            failures.append("batch is empty")
        for index, row in enumerate(rows):
            if len(row.values) != len(FEATURE_NAMES):
                failures.append(f"row {index}: expected {len(FEATURE_NAMES)} features")
                continue
            if not all(math.isfinite(value) for value in (*row.values, row.target)):
                failures.append(f"row {index}: non-finite value")
            if row.target < 0 or row.target > 200:
                failures.append(f"row {index}: RUL outside [0, 200]")
            if abs(row.values[1]) > 5:
                failures.append(f"row {index}: operating condition outside [-5, 5]")
            if len(failures) >= 20:
                failures.append("additional failures omitted")
                break
        report = GateReport(not failures, len(rows), tuple(failures))
        if failures:
            raise DataQualityError(report)
        return report

