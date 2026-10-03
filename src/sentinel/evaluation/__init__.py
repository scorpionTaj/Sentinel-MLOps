"""Offline evaluation: metrics, leakage-free splits, C-MAPSS protocols and calibration studies.

Everything here is deterministic and NumPy-only so that reported numbers can be regenerated with
`make eval` and `make report`.
"""

from sentinel.evaluation.metrics import bootstrap_ci, mae, nasa_score, regression_report, rmse
from sentinel.evaluation.splits import group_kfold, group_split

__all__ = [
    "bootstrap_ci",
    "group_kfold",
    "group_split",
    "mae",
    "nasa_score",
    "regression_report",
    "rmse",
]
