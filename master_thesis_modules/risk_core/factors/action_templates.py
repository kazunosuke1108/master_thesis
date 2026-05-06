"""Risky motion templates copied from ``master_v5.py``."""

from dataclasses import dataclass
import math

from master_thesis_modules.risk_core.features.pose_features import PoseFeatures
from master_thesis_modules.risk_core.schema import node_ids as ids


@dataclass(frozen=True)
class RiskyMotionTemplate:
    action_name: str
    node_id: int
    reference_pose: PoseFeatures


STANDING = RiskyMotionTemplate(
    "standUp",
    ids.STANDING_RISK,
    PoseFeatures(1.0, 0.0, 0.0, 1.0),
)
RELEASING_BRAKES = RiskyMotionTemplate(
    "releaseBrake",
    ids.WHEELCHAIR_BRAKE_RELEASE_RISK,
    PoseFeatures(0.0, 0.5, 0.5, 0.5),
)
MOVING_WHEELCHAIR = RiskyMotionTemplate(
    "moveWheelchair",
    ids.WHEELCHAIR_MOVE_RISK,
    PoseFeatures(0.0, 0.5, 0.5, 0.5),
)
LOSING_BALANCE = RiskyMotionTemplate(
    "loseBalance",
    ids.LOSING_BALANCE_RISK,
    PoseFeatures(0.0, 1.0, math.nan, math.nan),
)
RAISING_HANDS = RiskyMotionTemplate(
    "moveHand",
    ids.HAND_MOVEMENT_RISK,
    PoseFeatures(math.nan, 0.0, 1.0, math.nan),
)
COUGHING_UP = RiskyMotionTemplate(
    "coughUp",
    ids.COUGHING_RISK,
    PoseFeatures(math.nan, 0.5, 0.5, math.nan),
)
TOUCHING_FACE = RiskyMotionTemplate(
    "touchFace",
    ids.TOUCHING_FACE_RISK,
    PoseFeatures(math.nan, 0.0, 0.5, math.nan),
)

DEFAULT_ACTION_TEMPLATES = (
    STANDING,
    RELEASING_BRAKES,
    MOVING_WHEELCHAIR,
    LOSING_BALANCE,
    RAISING_HANDS,
    COUGHING_UP,
    TOUCHING_FACE,
)

ACTION_LABEL_TO_POSE = {
    template.action_name: template.reference_pose for template in DEFAULT_ACTION_TEMPLATES
}

