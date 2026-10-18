"""Strict, explicit respondent-level AHP/Fuzzy configuration (REQ-004).

Questionnaire consequents use risk-high/risk-low order HH, HL, LH, LL.
For watching, the two inputs are distance risk and not-watching risk;
being in the staff member's field of view therefore means a low second input.
The engine's existing movement-direction proxy and lower transformations stay
unchanged. Extra top-level provenance metadata is retained by the caller.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
import re

import numpy as np

from master_thesis_modules.risk_core.engine.risk_config import RiskConfig
from master_thesis_modules.risk_core.schema import node_ids as ids


_FUZZY_NODES = {
    "internal": ids.INTERNAL_RISK,
    "watching": ids.EXTERNAL_DYNAMIC_RISK,
    "external": ids.EXTERNAL_RISK,
    "total": ids.TOTAL_RISK,
}


def _validated_matrix(value: object, size: int, field: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{field} must be a {size}x{size} numeric matrix")
    for row in value:
        if not isinstance(row, list) or len(row) != size:
            raise ValueError(f"{field} must be a {size}x{size} numeric matrix")
        for cell in row:
            if isinstance(cell, bool) or not isinstance(cell, Real):
                raise ValueError(f"{field} requires finite numeric comparisons")
            if not 1 / 9 <= cell <= 9 or not math.isfinite(cell):
                raise ValueError(f"{field} comparisons must lie in [1/9, 9]")
    matrix = np.asarray(value, dtype=float)
    if not np.allclose(matrix.diagonal(), 1.0, rtol=0, atol=1e-12):
        raise ValueError(f"{field} must have a unit diagonal")
    # Float-serialized reciprocal matrices are accepted without repairing them.
    if not np.allclose(matrix * matrix.T, 1.0, rtol=1e-6, atol=0):
        raise ValueError(f"{field} must contain reciprocal comparisons")
    return matrix


def validate_personalized_knowledge(knowledge: dict) -> None:
    """Reject incomplete/ambiguous knowledge; never fall back to shared rules."""
    if not isinstance(knowledge, dict):
        raise ValueError("knowledge must be a dictionary")
    respondent_id = knowledge.get("respondent_id")
    if not isinstance(respondent_id, str) or not re.fullmatch(
        r"[A-Za-z][A-Za-z0-9_-]*", respondent_id, flags=re.ASCII
    ):
        raise ValueError("respondent_id must be an anonymous ASCII identifier")
    _validated_matrix(knowledge.get("action_pairwise"), 7, "action_pairwise")
    _validated_matrix(knowledge.get("object_pairwise"), 3, "object_pairwise")
    answers = knowledge.get("fuzzy_answers")
    if not isinstance(answers, dict) or set(answers) != set(_FUZZY_NODES):
        raise ValueError("fuzzy_answers requires exactly internal, watching, external, total")
    for group, scores in answers.items():
        if not isinstance(scores, list) or len(scores) != 4:
            raise ValueError(f"{group} requires 4 integer scores in HH/HL/LH/LL order")
        if any(
            isinstance(score, bool)
            or not isinstance(score, Integral)
            or not 1 <= score <= 5
            for score in scores
        ):
            raise ValueError(f"{group} requires integer scores from 1 to 5")


def _ahp_weights(matrix: list[list[float]]) -> list[float]:
    # Same principal-eigenvector method as legacy_ahp_adapter, without its
    # lower-triangle/diagonal repair (validation above requires valid input).
    eigenvalues, eigenvectors = np.linalg.eig(np.asarray(matrix, dtype=float))
    weights = eigenvectors[:, eigenvalues.real.argmax()].real
    weights = weights / weights.sum()
    if not np.all(np.isfinite(weights)) or not np.all(weights > 0):
        raise ValueError("AHP eigenvector weights must be finite and positive")
    return weights.tolist()


def make_personalized_risk_config(
    knowledge: dict, action_aggregation: str = "weighted_sum"
) -> RiskConfig:
    """Build the full spatial-context model with all four personal Fuzzy groups.

    Matrix order follows ids.ACTION_RISK_NODES and ids.OBJECT_RISK_NODES
    (IV pole, wheelchair, handrail). Questionnaire scores are representative
    values, not triangular uncertainty intervals: center = (score - 1) / 4.
    """
    validate_personalized_knowledge(knowledge)
    if action_aggregation not in {"weighted_sum", "weighted_max"}:
        raise ValueError("action_aggregation must be weighted_sum or weighted_max")
    fuzzy_rules = {}
    for group, node in _FUZZY_NODES.items():
        values = tuple((score - 1) / 4 for score in knowledge["fuzzy_answers"][group])
        if group == "watching":
            values = tuple(values[index] for index in (0, 2, 1, 3))
        fuzzy_rules[node] = values
    return RiskConfig(
        model_type="spatial_context",
        use_legacy_like_fuzzy=True,
        action_aggregation=action_aggregation,
        ahp_profile_name=knowledge["respondent_id"],
        fuzzy_profile_name=knowledge["respondent_id"],
        action_weights=dict(zip(ids.ACTION_RISK_NODES, _ahp_weights(knowledge["action_pairwise"]))),
        object_weights=dict(zip(ids.OBJECT_RISK_NODES, _ahp_weights(knowledge["object_pairwise"]))),
        fuzzy_rule_results=fuzzy_rules,
    )
