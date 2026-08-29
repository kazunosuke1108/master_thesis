"""Converters between semantic dataclasses and legacy node-number dictionaries."""

from master_thesis_modules.risk_core.features.feature_frame import FeatureFrame
from master_thesis_modules.risk_core.schema import node_ids as ids


def feature_frame_to_legacy_nodes(frame: FeatureFrame) -> dict[int, object]:
    nodes: dict[int, object] = {
        ids.IS_PATIENT: frame.is_patient_label,
        ids.AGE_CATEGORY: frame.age_group_label,
        ids.POSE_STANDING_DEGREE: frame.pose_features.standing_degree,
        ids.POSE_TRUNK_TILT: frame.pose_features.trunk_tilt,
        ids.POSE_WRIST_DISTANCE_FROM_HIP: frame.pose_features.wrist_distance_from_hip,
        ids.POSE_ANKLE_SPREAD: frame.pose_features.ankle_spread,
        ids.PERSON_X: frame.patient_position.x,
        ids.PERSON_Y: frame.patient_position.y,
    }
    if frame.height_max is not None:
        nodes[ids.PERSON_HEIGHT_MAX] = frame.height_max
    if frame.nearest_iv_position is not None:
        nodes[ids.IV_POLE_X] = frame.nearest_iv_position.x
        nodes[ids.IV_POLE_Y] = frame.nearest_iv_position.y
    if frame.nearest_wheelchair_position is not None:
        nodes[ids.WHEELCHAIR_X] = frame.nearest_wheelchair_position.x
        nodes[ids.WHEELCHAIR_Y] = frame.nearest_wheelchair_position.y
    if frame.nearest_handrail_position is not None:
        nodes[ids.HANDRAIL_X] = frame.nearest_handrail_position.x
        nodes[ids.HANDRAIL_Y] = frame.nearest_handrail_position.y
    if frame.nearest_staff_position is not None:
        nodes[ids.STAFF_X] = frame.nearest_staff_position.x
        nodes[ids.STAFF_Y] = frame.nearest_staff_position.y
    if frame.nearest_staff_velocity is not None:
        nodes[ids.STAFF_VX] = frame.nearest_staff_velocity.vx
        nodes[ids.STAFF_VY] = frame.nearest_staff_velocity.vy
    return nodes

