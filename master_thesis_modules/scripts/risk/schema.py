from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationStep:
    name: str
    method: str
    input_nodes: tuple
    output_nodes: tuple


@dataclass(frozen=True)
class NodeSpec:
    code: int
    name: str
    layer: int
    meaning: str
    value_direction: str
    expected_type: str


EVALUATION_STEPS = (
    EvaluationStep("internal_static_features","fuzzy_logic",(50000000,50000010),(40000000,40000001)),
    EvaluationStep("internal_dynamic_features","pose_similarity",(50000100,50000101,50000102,50000103),(40000010,40000011,40000012,40000013,40000014,40000015,40000016)),
    EvaluationStep("external_static_features","object_risk",(50001000,50001001,50001010,50001011,50001020,50001021,60010000,60010001),(40000100,40000101,40000102)),
    EvaluationStep("external_dynamic_features","staff_risk",(50001100,50001101,50001110,50001111,60010000,60010001),(40000110,40000111)),
    EvaluationStep("internal_static_risk","fuzzy_multiply",(40000000,40000001),(30000000,)),
    EvaluationStep("internal_dynamic_risk","AHP_weight_sum",(40000010,40000011,40000012,40000013,40000014,40000015,40000016),(30000001,)),
    EvaluationStep("external_static_risk","AHP_weight_sum",(40000100,40000101,40000102),(30000010,)),
    EvaluationStep("external_dynamic_risk","fuzzy_reasoning_master",(40000110,40000111),(30000011,)),
    EvaluationStep("internal_risk","simple_weight_sum",(30000000,30000001),(20000000,)),
    EvaluationStep("external_risk","fuzzy_reasoning_master",(30000010,30000011),(20000001,)),
    EvaluationStep("total_risk","fuzzy_reasoning_master",(20000000,20000001),(10000000,)),
)


NODE_SCHEMA = {
    10000000: NodeSpec(10000000,"total_risk",1,"総合危険度","higher_is_riskier","float_0_to_1"),
    20000000: NodeSpec(20000000,"internal_risk",2,"内的危険度","higher_is_riskier","float_0_to_1"),
    20000001: NodeSpec(20000001,"external_risk",2,"外的危険度","higher_is_riskier","float_0_to_1"),
    30000000: NodeSpec(30000000,"internal_static_risk",3,"内的・静的危険度","higher_is_riskier","float_0_to_1"),
    30000001: NodeSpec(30000001,"internal_dynamic_risk",3,"内的・動的危険度","higher_is_riskier","float_0_to_1"),
    30000010: NodeSpec(30000010,"external_static_risk",3,"外的・静的危険度","higher_is_riskier","float_0_to_1"),
    30000011: NodeSpec(30000011,"external_dynamic_risk",3,"外的・動的危険度","higher_is_riskier","float_0_to_1"),
    40000010: NodeSpec(40000010,"stand_up",4,"立ち上がり動作","higher_is_riskier","float_0_to_1"),
    40000100: NodeSpec(40000100,"near_iv_pole",4,"点滴の近くにいる","higher_is_riskier","float_0_to_1"),
    40000101: NodeSpec(40000101,"near_wheelchair",4,"車椅子に乗っている/近い","higher_is_riskier","float_0_to_1"),
    40000102: NodeSpec(40000102,"far_from_handrail",4,"手すりから離れている","higher_is_riskier","float_0_to_1"),
    40000110: NodeSpec(40000110,"staff_distance",4,"スタッフが近くにいない","higher_is_riskier","float_0_to_1"),
    40000111: NodeSpec(40000111,"staff_watch_loss",4,"スタッフが見ていない","higher_is_riskier","float_0_to_1"),
    50001000: NodeSpec(50001000,"iv_pole_x",5,"最近傍点滴 x 座標","coordinate","float"),
    50001001: NodeSpec(50001001,"iv_pole_y",5,"最近傍点滴 y 座標","coordinate","float"),
    50001002: NodeSpec(50001002,"iv_pole_confidence_0",5,"点滴存在ポテンシャル","higher_is_more_confident","float_0_to_1"),
    50001003: NodeSpec(50001003,"iv_pole_confidence_1",5,"点滴存在ポテンシャル","higher_is_more_confident","float_0_to_1"),
    50001010: NodeSpec(50001010,"wheelchair_x",5,"最近傍車椅子 x 座標","coordinate","float"),
    50001011: NodeSpec(50001011,"wheelchair_y",5,"最近傍車椅子 y 座標","coordinate","float"),
    50001012: NodeSpec(50001012,"wheelchair_confidence_0",5,"車椅子存在ポテンシャル","higher_is_more_confident","float_0_to_1"),
    50001013: NodeSpec(50001013,"wheelchair_confidence_1",5,"車椅子存在ポテンシャル","higher_is_more_confident","float_0_to_1"),
    50001020: NodeSpec(50001020,"handrail_x",5,"最近傍手すり x 座標","coordinate","float"),
    50001021: NodeSpec(50001021,"handrail_y",5,"最近傍手すり y 座標","coordinate","float"),
    50001100: NodeSpec(50001100,"staff_x",5,"最近傍スタッフ x 座標","coordinate","float"),
    50001101: NodeSpec(50001101,"staff_y",5,"最近傍スタッフ y 座標","coordinate","float"),
    50001110: NodeSpec(50001110,"staff_vx",5,"最近傍スタッフ x 方向変化","velocity","float"),
    50001111: NodeSpec(50001111,"staff_vy",5,"最近傍スタッフ y 方向変化","velocity","float"),
    60010000: NodeSpec(60010000,"person_x",6,"対象者 x 座標","coordinate","float"),
    60010001: NodeSpec(60010001,"person_y",6,"対象者 y 座標","coordinate","float"),
    60010002: NodeSpec(60010002,"person_zmax",6,"対象者高さ最大値","higher_is_taller","float"),
}
