"""Encode a WorldState into FeatureFrame objects."""

from master_thesis_modules.risk_core.features.feature_frame import FeatureFrame
from master_thesis_modules.risk_core.features.position import Position2D, Velocity2D
from master_thesis_modules.scenario_sim.domain.object_entity import ObjectEntity
from master_thesis_modules.scenario_sim.domain.patient import Patient
from master_thesis_modules.scenario_sim.domain.staff import Staff
from master_thesis_modules.scenario_sim.domain.world_state import WorldState
from master_thesis_modules.scenario_sim.encoder.pose_preset_encoder import PosePresetEncoder


class FeatureEncoder:
    def __init__(self, pose_encoder: PosePresetEncoder | None = None) -> None:
        self.pose_encoder = pose_encoder or PosePresetEncoder()

    def encode(self, world_state: WorldState) -> list[FeatureFrame]:
        return [
            self._encode_patient(patient, world_state)
            for patient in world_state.patients
        ]

    def _encode_patient(self, patient: Patient, world_state: WorldState) -> FeatureFrame:
        nearest_iv = _nearest_object(patient.position, world_state.objects, "iv_pole")
        nearest_wheelchair = _nearest_object(
            patient.position,
            world_state.objects,
            "wheelchair",
        )
        nearest_handrail = _nearest_object(
            patient.position,
            world_state.objects,
            "handrail",
        )
        nearest_staff = _nearest_staff(patient.position, world_state.staff)
        staff_position: Position2D | None = None
        staff_velocity: Velocity2D | None = None
        if nearest_staff is not None:
            staff_position = nearest_staff.position
            staff_velocity = nearest_staff.velocity

        return FeatureFrame(
            person_id=patient.person_id,
            time_s=world_state.time_s,
            is_patient_label=patient.is_patient_label,
            is_patient_confidence=1.0,
            age_group_label=patient.age_group_label,
            age_confidence=1.0,
            pose_features=patient.pose_features or self.pose_encoder.encode(patient.action_label),
            patient_position=patient.position,
            nearest_iv_position=nearest_iv.position if nearest_iv else None,
            nearest_wheelchair_position=(
                nearest_wheelchair.position if nearest_wheelchair else None
            ),
            nearest_handrail_position=nearest_handrail.position if nearest_handrail else None,
            nearest_staff_position=staff_position,
            nearest_staff_velocity=staff_velocity,
            height_max=patient.height_max,
            action_label=patient.action_label,
        )


def _nearest_object(
    position: Position2D,
    objects: tuple[ObjectEntity, ...],
    object_type: str,
) -> ObjectEntity | None:
    candidates = [item for item in objects if item.object_type == object_type]
    if not candidates:
        return None
    return min(candidates, key=lambda item: position.distance_to(item.position))


def _nearest_staff(position: Position2D, staff: tuple[Staff, ...]) -> Staff | None:
    if not staff:
        return None
    return min(staff, key=lambda item: position.distance_to(item.position))
