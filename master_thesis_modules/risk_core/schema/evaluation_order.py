"""Evaluation order for the MVP risk engine."""

from dataclasses import dataclass

from master_thesis_modules.risk_core.schema import node_ids as ids


@dataclass(frozen=True)
class EvaluationStep:
    name: str
    input_nodes: tuple[int, ...]
    output_nodes: tuple[int, ...]


LEGACY_EVALUATION_ORDER = (
    EvaluationStep(
        "internal_static_features",
        (ids.IS_PATIENT, ids.AGE_CATEGORY),
        ids.ATTRIBUTE_RISK_NODES,
    ),
    EvaluationStep(
        "internal_dynamic_features",
        (
            ids.POSE_STANDING_DEGREE,
            ids.POSE_TRUNK_TILT,
            ids.POSE_WRIST_DISTANCE_FROM_HIP,
            ids.POSE_ANKLE_SPREAD,
            ids.PERSON_HEIGHT_MAX,
        ),
        ids.ACTION_RISK_NODES,
    ),
    EvaluationStep(
        "external_static_features",
        (
            ids.IV_POLE_X,
            ids.IV_POLE_Y,
            ids.WHEELCHAIR_X,
            ids.WHEELCHAIR_Y,
            ids.HANDRAIL_X,
            ids.HANDRAIL_Y,
            ids.PERSON_X,
            ids.PERSON_Y,
        ),
        ids.OBJECT_RISK_NODES,
    ),
    EvaluationStep(
        "external_dynamic_features",
        (
            ids.STAFF_X,
            ids.STAFF_Y,
            ids.STAFF_VX,
            ids.STAFF_VY,
            ids.PERSON_X,
            ids.PERSON_Y,
        ),
        ids.STAFF_RISK_NODES,
    ),
    EvaluationStep("internal_static_risk", ids.ATTRIBUTE_RISK_NODES, (ids.INTERNAL_STATIC_RISK,)),
    EvaluationStep("internal_dynamic_risk", ids.ACTION_RISK_NODES, (ids.INTERNAL_DYNAMIC_RISK,)),
    EvaluationStep("external_static_risk", ids.OBJECT_RISK_NODES, (ids.EXTERNAL_STATIC_RISK,)),
    EvaluationStep("external_dynamic_risk", ids.STAFF_RISK_NODES, (ids.EXTERNAL_DYNAMIC_RISK,)),
    EvaluationStep(
        "internal_risk",
        (ids.INTERNAL_STATIC_RISK, ids.INTERNAL_DYNAMIC_RISK),
        (ids.INTERNAL_RISK,),
    ),
    EvaluationStep(
        "external_risk",
        (ids.EXTERNAL_STATIC_RISK, ids.EXTERNAL_DYNAMIC_RISK),
        (ids.EXTERNAL_RISK,),
    ),
    EvaluationStep("total_risk", (ids.INTERNAL_RISK, ids.EXTERNAL_RISK), (ids.TOTAL_RISK,)),
)

EVALUATION_ORDER = LEGACY_EVALUATION_ORDER

