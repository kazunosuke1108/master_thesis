from pathlib import Path

from master_thesis_modules.risk_core.engine.risk_engine import RiskEngine
from master_thesis_modules.scenario_sim.encoder.feature_encoder import FeatureEncoder
from master_thesis_modules.scenario_sim.encoder.scenario_loader import ScenarioLoader


SCENARIO = Path(
    "master_thesis_modules/scenario_sim/scenarios/reach_object_context_demo.yaml"
)


def test_reach_object_context_demo_ranking_is_c_a_b():
    world_state = ScenarioLoader().load(SCENARIO)
    frames = FeatureEncoder().encode(world_state)
    results = [RiskEngine().evaluate(frame) for frame in frames]
    ranked_ids = [
        result.person_id
        for result in sorted(results, key=lambda result: result.total_risk, reverse=True)
    ]

    assert ranked_ids == ["C", "A", "B"]

