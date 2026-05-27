"""Build time-series FeatureFrameSequence objects from scenario events."""

from dataclasses import replace
import math

from master_thesis_modules.risk_core.features.feature_sequence import FeatureFrameSequence
from master_thesis_modules.scenario_sim.domain.world_state import WorldState
from master_thesis_modules.scenario_sim.encoder.feature_encoder import FeatureEncoder
from master_thesis_modules.scenario_sim.events.event_engine import EventEngine


class ScenarioDataFrameBuilder:
    def __init__(
        self,
        feature_encoder: FeatureEncoder | None = None,
        event_engine: EventEngine | None = None,
    ) -> None:
        self.feature_encoder = feature_encoder or FeatureEncoder()
        self.event_engine = event_engine or EventEngine()

    def build_sequences(self, world_state: WorldState) -> dict[str, FeatureFrameSequence]:
        if world_state.duration_s <= 0.0:
            frames = self.feature_encoder.encode(world_state)
            return {
                frame.person_id: FeatureFrameSequence(frame.person_id, [frame])
                for frame in frames
            }

        sequences: dict[str, list] = {patient.person_id: [] for patient in world_state.patients}
        current = replace(world_state, events=())
        step_count = int(math.ceil(world_state.duration_s / world_state.step_s))
        applied_event_ids: set[int] = set()
        previous_time = -float("inf")
        for step in range(step_count + 1):
            time_s = min(round(step * world_state.step_s, 10), world_state.duration_s)
            event_pairs = tuple(
                (event_index, event)
                for event_index, event in enumerate(world_state.events)
                if event_index not in applied_event_ids and previous_time < event.time_s <= time_s
            )
            events = tuple(event for _, event in event_pairs)
            for event_index, _ in event_pairs:
                applied_event_ids.add(event_index)
            current = self.event_engine.apply(current, events)
            current_at_time = replace(current, time_s=time_s)
            for frame in self.feature_encoder.encode(current_at_time):
                sequences[frame.person_id].append(frame)
            previous_time = time_s
        return {
            person_id: FeatureFrameSequence(person_id, frames)
            for person_id, frames in sequences.items()
        }
