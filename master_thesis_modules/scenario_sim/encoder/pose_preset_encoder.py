"""Map scenario action labels to four-dimensional pose features."""

from master_thesis_modules.risk_core.factors.action_templates import ACTION_LABEL_TO_POSE
from master_thesis_modules.risk_core.features.pose_features import PoseFeatures


POSE_PRESETS = {
    "neutral_sitting": PoseFeatures(0.0, 0.0, 0.0, 0.2),
    "reach_floor": PoseFeatures(0.0, 0.8, 0.8, 0.4),
    "reach_table": PoseFeatures(0.2, 0.2, 0.9, 0.2),
    "reach_iv_pole": PoseFeatures(0.15, 0.35, 0.9, 0.3),
    "standing_up": ACTION_LABEL_TO_POSE["standUp"],
    "release_brake": ACTION_LABEL_TO_POSE["releaseBrake"],
    "move_wheelchair": ACTION_LABEL_TO_POSE["moveWheelchair"],
    "lose_balance": ACTION_LABEL_TO_POSE["loseBalance"],
    "raise_hands": ACTION_LABEL_TO_POSE["moveHand"],
    "cough_up": ACTION_LABEL_TO_POSE["coughUp"],
    "touch_face": ACTION_LABEL_TO_POSE["touchFace"],
}


class PosePresetEncoder:
    def encode(self, action_label: str | None) -> PoseFeatures:
        if action_label is None:
            return POSE_PRESETS["neutral_sitting"]
        try:
            return POSE_PRESETS[action_label]
        except KeyError as exc:
            supported = ", ".join(sorted(POSE_PRESETS))
            raise ValueError(f"Unsupported action_label '{action_label}'. Use: {supported}") from exc

