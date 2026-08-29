"""Scenario event dataclass for future extension."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioEvent:
    time_s: float
    target_id: str
    event_type: str
    payload: dict[str, object]
