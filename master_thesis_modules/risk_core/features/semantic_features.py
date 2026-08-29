"""Semantic feature labels produced by perception or scenarios."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticFeatures:
    is_patient_label: str = "unknown"
    is_patient_confidence: float = 1.0
    age_group_label: str = "unknown"
    age_confidence: float = 1.0
    action_label: str | None = None

