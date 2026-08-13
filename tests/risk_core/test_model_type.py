import pytest

from master_thesis_modules.risk_core.engine.risk_config import RiskConfig
from master_thesis_modules.risk_core.engine.risk_engine import RiskEngine
from master_thesis_modules.risk_core.features.feature_frame import FeatureFrame
from master_thesis_modules.risk_core.features.pose_features import PoseFeatures
from master_thesis_modules.risk_core.features.position import Position2D, Velocity2D
from master_thesis_modules.risk_core.schema import node_ids as ids


def test_patient_context_ignores_object_and_staff_context_in_total_risk():
    frame = _spatially_risky_frame()

    result = RiskEngine(RiskConfig(model_type="patient_context")).evaluate(frame)

    assert result.factor_risks[ids.IV_POLE_RISK] == 0.0
    assert result.factor_risks[ids.WHEELCHAIR_RISK] == 0.0
    assert result.factor_risks[ids.HANDRAIL_DISTANCE_RISK] == 0.0
    assert result.factor_risks[ids.STAFF_DISTANCE_RISK] == 0.0
    assert result.factor_risks[ids.STAFF_NOT_WATCHING_RISK] == 0.0
    assert result.upper_risks[ids.EXTERNAL_RISK] == 0.0
    assert result.total_risk == result.upper_risks[ids.INTERNAL_RISK]


def test_patient_context_alias_matches_legacy_action_attribute_name():
    frame = _spatially_risky_frame()

    patient_context = RiskEngine(RiskConfig(model_type="patient_context")).evaluate(frame)
    action_attribute = RiskEngine(RiskConfig(model_type="action_attribute")).evaluate(frame)

    assert patient_context.total_risk == action_attribute.total_risk
    assert patient_context.factor_risks == action_attribute.factor_risks


def test_unknown_model_type_is_rejected():
    with pytest.raises(ValueError, match="Unknown model_type"):
        RiskConfig(model_type="unknown")


def _spatially_risky_frame() -> FeatureFrame:
    return FeatureFrame(
        person_id="A",
        time_s=0.0,
        is_patient_label="yes",
        is_patient_confidence=1.0,
        age_group_label="old",
        age_confidence=1.0,
        pose_features=PoseFeatures(
            standing_degree=0.8,
            trunk_tilt=0.2,
            wrist_distance_from_hip=0.4,
            ankle_spread=0.7,
        ),
        patient_position=Position2D(0.0, 0.0),
        nearest_iv_position=Position2D(0.0, 0.0),
        nearest_wheelchair_position=Position2D(0.0, 0.0),
        nearest_handrail_position=Position2D(6.0, 6.0),
        nearest_staff_position=Position2D(6.0, 6.0),
        nearest_staff_velocity=Velocity2D(1.0, 1.0),
    )
