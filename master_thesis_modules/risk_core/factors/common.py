"""Shared numeric helpers for risk factor modules."""

from __future__ import annotations

import math


SPATIAL_NORMALIZATION_PARAM = math.sqrt(2.0) * 6.0


def clip01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


def safe_mean(values: list[float]) -> float:
    finite_values = [v for v in values if math.isfinite(v)]
    if not finite_values:
        return 0.0
    return sum(finite_values) / len(finite_values)

