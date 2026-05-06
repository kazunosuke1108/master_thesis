"""Result objects returned by the risk engine."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskResult:
    person_id: str
    time_s: float
    factor_risks: dict[int, float]
    upper_risks: dict[int, float]
    total_risk: float
    explanation: str = ""

