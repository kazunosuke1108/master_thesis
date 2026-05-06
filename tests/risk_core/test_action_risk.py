import math

from master_thesis_modules.risk_core.factors.action_risk import (
    ActionRiskCalculator,
    pose_similarity,
)
from master_thesis_modules.risk_core.factors.action_templates import (
    RELEASING_BRAKES,
    STANDING,
)
from master_thesis_modules.risk_core.features.pose_features import PoseFeatures
from master_thesis_modules.risk_core.schema import node_ids as ids


def test_standing_template_match_is_high():
    risks = ActionRiskCalculator().calculate(STANDING.reference_pose)

    assert risks[ids.STANDING_RISK] == 1.0


def test_releasing_brake_template_match_is_high():
    risks = ActionRiskCalculator().calculate(RELEASING_BRAKES.reference_pose)

    assert risks[ids.WHEELCHAIR_BRAKE_RELEASE_RISK] == 1.0


def test_far_pose_has_low_action_risks():
    observed = PoseFeatures(1.0, 1.0, 1.0, 0.0)
    risks = ActionRiskCalculator().calculate(observed)

    assert risks[ids.WHEELCHAIR_BRAKE_RELEASE_RISK] < 0.1


def test_similarity_power_suppresses_middle_similarity():
    observed = PoseFeatures(0.5, 0.5, 0.5, 0.5)
    reference = PoseFeatures(1.0, 1.0, 1.0, 1.0)

    assert math.isclose(pose_similarity(observed, reference), 0.5**4)


def test_height_max_overwrites_standing_risk():
    observed = STANDING.reference_pose
    risks = ActionRiskCalculator().calculate(observed, height_max=0.0)

    assert risks[ids.STANDING_RISK] < 0.01


def test_all_action_risks_are_clipped_to_unit_range():
    observed = PoseFeatures(3.0, -2.0, 10.0, -10.0)
    risks = ActionRiskCalculator().calculate(observed, height_max=10.0)

    assert all(0.0 <= value <= 1.0 for value in risks.values())

