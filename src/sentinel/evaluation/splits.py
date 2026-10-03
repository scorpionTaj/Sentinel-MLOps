from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import numpy as np


def _ordered_groups(groups: Sequence[object]) -> list[object]:
    return list(dict.fromkeys(groups))


def group_split(
    groups: Sequence[object], validation_fraction: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """Split row indices so that no engine appears on both sides.

    The last `validation_fraction` of groups (in order of first appearance) are held out, which
    keeps the split deterministic and mirrors "train on earlier fleet, validate on later fleet".
    With a single group there is nothing to hold out by engine, so the rows are split
    chronologically instead.
    """

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    indices = np.arange(len(groups))
    ordered = _ordered_groups(groups)
    if len(ordered) < 2:
        cut = max(2, int(len(indices) * (1 - validation_fraction)))
        return indices[:cut], indices[cut:]
    held_out = set(ordered[len(ordered) - max(1, math.ceil(len(ordered) * validation_fraction)) :])
    mask = np.fromiter((group in held_out for group in groups), dtype=bool, count=len(groups))
    return indices[~mask], indices[mask]


def group_kfold(
    groups: Sequence[object], folds: int = 5, seed: int = 0
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train, validation) index pairs with engines shuffled into disjoint folds."""

    ordered = _ordered_groups(groups)
    if folds < 2 or folds > len(ordered):
        raise ValueError("folds must be between 2 and the number of groups")
    rng = np.random.default_rng(seed)
    shuffled = [ordered[i] for i in rng.permutation(len(ordered))]
    assignment = {group: index % folds for index, group in enumerate(shuffled)}
    fold_of_row = np.fromiter((assignment[g] for g in groups), dtype=int, count=len(groups))
    indices = np.arange(len(groups))
    for fold in range(folds):
        yield indices[fold_of_row != fold], indices[fold_of_row == fold]
