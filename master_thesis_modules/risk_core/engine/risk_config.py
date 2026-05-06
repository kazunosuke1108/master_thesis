"""Configuration for the risk engine."""

from dataclasses import dataclass, field
import math

from master_thesis_modules.risk_core.schema import node_ids as ids


@dataclass(frozen=True)
class RiskConfig:
    """Weights are explicit and can be swapped for legacy AHP/Fuzzy adapters."""

    model_type: str = "spatial_context"
    room_diagonal_m: float = math.sqrt(2.0) * 6.0
    height_sigmoid_gain: float = 5.0
    height_sigmoid_center: float = 1.0
    use_legacy_like_fuzzy: bool = True
    ahp_profile_name: str | None = None
    fuzzy_profile_name: str | None = None
    fuzzy_rule_results: dict[int, tuple[str | float, ...]] = field(default_factory=dict)
    internal_static_weight: float = 0.15
    internal_dynamic_weight: float = 0.85
    internal_weight: float = 0.45
    external_weight: float = 0.55
    action_weights: dict[int, float] = field(
        default_factory=lambda: {
            ids.STANDING_RISK: 1.1,
            ids.WHEELCHAIR_BRAKE_RELEASE_RISK: 0.8,
            ids.WHEELCHAIR_MOVE_RISK: 0.8,
            ids.LOSING_BALANCE_RISK: 1.2,
            ids.HAND_MOVEMENT_RISK: 0.9,
            ids.COUGHING_RISK: 0.4,
            ids.TOUCHING_FACE_RISK: 0.3,
        }
    )
    object_weights: dict[int, float] = field(
        default_factory=lambda: {
            ids.IV_POLE_RISK: 1.0,
            ids.WHEELCHAIR_RISK: 0.7,
            ids.HANDRAIL_DISTANCE_RISK: 0.8,
        }
    )
    staff_weights: dict[int, float] = field(
        default_factory=lambda: {
            ids.STAFF_DISTANCE_RISK: 0.9,
            ids.STAFF_NOT_WATCHING_RISK: 1.1,
        }
    )
    total_factor_weights: dict[int, float] = field(
        default_factory=lambda: {
            ids.STANDING_RISK: 0.35,
            ids.WHEELCHAIR_BRAKE_RELEASE_RISK: 0.15,
            ids.WHEELCHAIR_MOVE_RISK: 0.15,
            ids.LOSING_BALANCE_RISK: 0.35,
            ids.HAND_MOVEMENT_RISK: 0.35,
            ids.COUGHING_RISK: 0.1,
            ids.TOUCHING_FACE_RISK: 0.1,
            ids.IV_POLE_RISK: 0.55,
            ids.WHEELCHAIR_RISK: 0.25,
            ids.HANDRAIL_DISTANCE_RISK: 0.4,
            ids.STAFF_DISTANCE_RISK: 0.55,
            ids.STAFF_NOT_WATCHING_RISK: 0.65,
        }
    )
