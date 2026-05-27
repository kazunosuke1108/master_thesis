"""Action labels supported by the scenario encoder."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    label: str
    description: str = ""

