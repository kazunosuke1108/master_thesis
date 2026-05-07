"""Batch evaluation for multiple people and timestamps."""

from master_thesis_modules.risk_core.engine.risk_engine import RiskEngine
from master_thesis_modules.risk_core.engine.risk_result import RiskResult
from master_thesis_modules.risk_core.features.feature_sequence import FeatureFrameSequence
from master_thesis_modules.risk_core.features.dataframe_adapter import (
    results_to_dataframe,
)


class BatchRiskEngine:
    def __init__(self, risk_engine: RiskEngine | None = None) -> None:
        self.risk_engine = risk_engine or RiskEngine()

    def evaluate(
        self,
        sequences: dict[str, FeatureFrameSequence],
    ) -> dict[str, list[RiskResult]]:
        return {
            person_id: [self.risk_engine.evaluate(frame) for frame in sequence.frames]
            for person_id, sequence in sequences.items()
        }

    def evaluate_to_dataframes(
        self,
        sequences: dict[str, FeatureFrameSequence],
        source_dataframes,
    ):
        results = self.evaluate(sequences)
        return {
            person_id: results_to_dataframe(source_dataframes[person_id], person_results)
            for person_id, person_results in results.items()
        }

