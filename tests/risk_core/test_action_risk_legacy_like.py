from master_thesis_modules.risk_core.factors.action_risk import ActionRiskCalculator
from master_thesis_modules.risk_core.factors.action_templates import STANDING
from master_thesis_modules.risk_core.features.pose_features import PoseFeatures
from master_thesis_modules.risk_core.schema import node_ids as ids


def test_template_match_raises_corresponding_action_risk():
    risks = ActionRiskCalculator().calculate(STANDING.reference_pose)

    assert risks[ids.STANDING_RISK] == 1.0


def test_far_pose_keeps_template_risks_low():
    risks = ActionRiskCalculator().calculate(PoseFeatures(1.0, 1.0, 1.0, 0.0))

    assert risks[ids.WHEELCHAIR_BRAKE_RELEASE_RISK] < 0.1


def test_height_max_overwrites_standing_risk():
    risks = ActionRiskCalculator().calculate(STANDING.reference_pose, height_max=0.0)

    assert risks[ids.STANDING_RISK] < 0.01

