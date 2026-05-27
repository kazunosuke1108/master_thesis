from master_thesis_modules.risk_core.features.position import Position2D, Velocity2D
from master_thesis_modules.scenario_sim.domain.patient import Patient
from master_thesis_modules.scenario_sim.domain.staff import Staff
from master_thesis_modules.scenario_sim.domain.world_state import WorldState
from master_thesis_modules.scenario_sim.encoder.dataframe_builder import (
    ScenarioDataFrameBuilder,
)
from master_thesis_modules.scenario_sim.events.scenario_event import ScenarioEvent


def test_event_between_steps_is_applied_at_next_frame():
    world_state = WorldState(
        scenario_name="off_grid_event",
        time_s=0.0,
        duration_s=1.0,
        step_s=0.5,
        patients=(
            Patient(
                person_id="A",
                is_patient_label="yes",
                age_group_label="old",
                position=Position2D(0.0, 0.0),
                action_label="neutral_sitting",
            ),
        ),
        staff=(Staff("S", Position2D(0.0, 1.0), Velocity2D(0.0, -0.1)),),
        objects=(),
        events=(
            ScenarioEvent(
                time_s=0.25,
                target_id="A",
                event_type="move_patient",
                payload={"x": 2.0, "y": 0.0},
            ),
        ),
    )

    sequence = ScenarioDataFrameBuilder().build_sequences(world_state)["A"]

    assert sequence.frames[0].patient_position == Position2D(0.0, 0.0)
    assert sequence.frames[1].patient_position == Position2D(2.0, 0.0)
    assert sequence.frames[2].patient_position == Position2D(2.0, 0.0)

