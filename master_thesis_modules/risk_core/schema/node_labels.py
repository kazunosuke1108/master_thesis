"""Human-readable labels for risk nodes."""

from master_thesis_modules.risk_core.schema import node_ids as ids

NODE_LABELS = {
    ids.TOTAL_RISK: "総合危険度",
    ids.INTERNAL_RISK: "内的リスク",
    ids.EXTERNAL_RISK: "外的リスク",
    ids.INTERNAL_STATIC_RISK: "内的・静的リスク",
    ids.INTERNAL_DYNAMIC_RISK: "内的・動的リスク",
    ids.EXTERNAL_STATIC_RISK: "外的・静的リスク",
    ids.EXTERNAL_DYNAMIC_RISK: "外的・動的リスク",
    ids.STANDING_RISK: "立ち上がり動作",
    ids.WHEELCHAIR_BRAKE_RELEASE_RISK: "車椅子ブレーキ解除",
    ids.WHEELCHAIR_MOVE_RISK: "車椅子を動かす",
    ids.LOSING_BALANCE_RISK: "バランスを崩す",
    ids.HAND_MOVEMENT_RISK: "手を挙げる/動かす",
    ids.COUGHING_RISK: "せき込む",
    ids.TOUCHING_FACE_RISK: "顔を触る",
    ids.IV_POLE_RISK: "点滴付近",
    ids.WHEELCHAIR_RISK: "車椅子付近",
    ids.HANDRAIL_DISTANCE_RISK: "手すりから離れている",
    ids.STAFF_DISTANCE_RISK: "スタッフが近くにいない",
    ids.STAFF_NOT_WATCHING_RISK: "スタッフが見ていない",
}

