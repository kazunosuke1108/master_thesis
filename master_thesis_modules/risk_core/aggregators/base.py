"""Aggregator interface used by RiskEngine."""

from typing import Protocol


class Aggregator(Protocol):
    def aggregate(self, inputs: dict[int, float]) -> float:
        """Return one 0.0-1.0 risk value from node keyed inputs."""

