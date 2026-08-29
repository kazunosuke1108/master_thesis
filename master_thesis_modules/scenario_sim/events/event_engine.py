"""Apply simple scenario events to a WorldState."""

from dataclasses import replace

from master_thesis_modules.risk_core.features.pose_features import PoseFeatures
from master_thesis_modules.risk_core.features.position import Position2D, Velocity2D
from master_thesis_modules.scenario_sim.domain.world_state import WorldState
from master_thesis_modules.scenario_sim.events.scenario_event import ScenarioEvent


class EventEngine:
    def apply(
        self,
        world_state: WorldState,
        events: tuple[ScenarioEvent, ...] = (),
    ) -> WorldState:
        patients = {patient.person_id: patient for patient in world_state.patients}
        staff = {person.staff_id: person for person in world_state.staff}
        objects = {item.object_id: item for item in world_state.objects}

        for event in sorted(events, key=lambda item: item.time_s):
            if event.event_type == "set_action" and event.target_id in patients:
                patients[event.target_id] = replace(
                    patients[event.target_id],
                    action_label=str(event.payload["action_label"]),
                    pose_features=None,
                )
            elif event.event_type == "move_patient" and event.target_id in patients:
                patients[event.target_id] = replace(
                    patients[event.target_id],
                    position=_position(event.payload),
                )
            elif event.event_type == "move_staff" and event.target_id in staff:
                staff[event.target_id] = replace(
                    staff[event.target_id],
                    position=_position(event.payload),
                    velocity=_velocity(event.payload),
                )
            elif event.event_type == "set_pose_features" and event.target_id in patients:
                patients[event.target_id] = replace(
                    patients[event.target_id],
                    pose_features=PoseFeatures(
                        float(event.payload["standing_degree"]),
                        float(event.payload["trunk_tilt"]),
                        float(event.payload["wrist_distance_from_hip"]),
                        float(event.payload["ankle_spread"]),
                    ),
                )
            elif event.event_type == "set_object_position" and event.target_id in objects:
                objects[event.target_id] = replace(
                    objects[event.target_id],
                    position=_position(event.payload),
                )
            elif event.event_type in {
                "set_action",
                "move_patient",
                "move_staff",
                "set_pose_features",
                "set_object_position",
            }:
                raise KeyError(
                    f"event target '{event.target_id}' was not found for {event.event_type}"
                )
            else:
                raise ValueError(f"Unsupported event_type: {event.event_type}")

        return replace(
            world_state,
            patients=tuple(patients.values()),
            staff=tuple(staff.values()),
            objects=tuple(objects.values()),
        )


def _position(payload: dict[str, object]) -> Position2D:
    if "position" in payload:
        payload = payload["position"]  # type: ignore[assignment]
    return Position2D(float(payload["x"]), float(payload["y"]))


def _velocity(payload: dict[str, object]) -> Velocity2D:
    raw = payload.get("velocity", payload)
    return Velocity2D(float(raw.get("vx", 0.0)), float(raw.get("vy", 0.0)))
