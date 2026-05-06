"""Text fragments used by the deterministic explanation generator."""

from master_thesis_modules.risk_core.schema import node_ids as ids

REASON_TEMPLATES = {
    ids.STANDING_RISK: "立ち上がり動作の兆候が強い",
    ids.WHEELCHAIR_BRAKE_RELEASE_RISK: "車椅子ブレーキ解除に近い姿勢がある",
    ids.WHEELCHAIR_MOVE_RISK: "車椅子を動かす動作に近い姿勢がある",
    ids.LOSING_BALANCE_RISK: "バランスを崩す姿勢に近い",
    ids.HAND_MOVEMENT_RISK: "手を伸ばす/動かす動作が見られる",
    ids.COUGHING_RISK: "せき込む動作に近い姿勢がある",
    ids.TOUCHING_FACE_RISK: "顔を触る動作に近い姿勢がある",
    ids.IV_POLE_RISK: "点滴付近にいる",
    ids.WHEELCHAIR_RISK: "車椅子付近にいる",
    ids.HANDRAIL_DISTANCE_RISK: "手すりから離れている",
    ids.STAFF_DISTANCE_RISK: "スタッフが離れている",
    ids.STAFF_NOT_WATCHING_RISK: "スタッフの見守り方向が弱い",
}

