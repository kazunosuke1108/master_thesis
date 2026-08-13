"""Legacy-style notification message generation.

This mirrors the notification sentence generation described in thesis section
3.3.6.2 while operating on the renovated risk-core result objects.
"""

from __future__ import annotations

from collections.abc import Iterable
import math

import numpy as np

from master_thesis_modules.risk_core.engine.risk_result import RiskResult
from master_thesis_modules.risk_core.schema import node_ids as ids


LEGACY_NODE_DESCRIPTIONS_JA = {
    ids.PATIENT_ATTRIBUTE_RISK: "患者である",
    ids.AGE_ATTRIBUTE_RISK: "高齢である",
    ids.STANDING_RISK: "立ち上がろうとしている",
    ids.WHEELCHAIR_BRAKE_RELEASE_RISK: "車椅子のブレーキを解除しようとしている",
    ids.WHEELCHAIR_MOVE_RISK: "車椅子を動かそうとしている",
    ids.LOSING_BALANCE_RISK: "バランスを崩している",
    ids.HAND_MOVEMENT_RISK: "手を挙げている",
    ids.COUGHING_RISK: "せき込んでいる",
    ids.TOUCHING_FACE_RISK: "顔を触っている",
    ids.IV_POLE_RISK: "点滴の近くにいる",
    ids.WHEELCHAIR_RISK: "車椅子に乗っている",
    ids.HANDRAIL_DISTANCE_RISK: "手すりから離れている",
    ids.STAFF_DISTANCE_RISK: "スタッフがいない",
    ids.STAFF_NOT_WATCHING_RISK: "スタッフが見ていない",
}

LEGACY_STATIC_NODES = (
    ids.PATIENT_ATTRIBUTE_RISK,
    ids.AGE_ATTRIBUTE_RISK,
    ids.IV_POLE_RISK,
    ids.WHEELCHAIR_RISK,
    ids.HANDRAIL_DISTANCE_RISK,
)

LEGACY_DYNAMIC_NODES = ids.ACTION_RISK_NODES + ids.STAFF_RISK_NODES


class LegacyNotificationMessageGenerator:
    def __init__(self, window_size: int = 20) -> None:
        self.window_size = window_size

    def generate_notice(
        self,
        result: RiskResult,
        results_by_person: dict[str, list[RiskResult]],
    ) -> str:
        person_id = str(result.person_id)
        window_by_person = self._window_by_person(results_by_person, result.time_s)
        patient_window = window_by_person.get(person_id, [result])
        dynamic_node = self._guess_dynamic_factor(patient_window)
        static_node = self._guess_static_factor(window_by_person, person_id)
        return self._format(person_id, static_node, dynamic_node)

    def _window_by_person(
        self,
        results_by_person: dict[str, list[RiskResult]],
        timestamp: float,
    ) -> dict[str, list[RiskResult]]:
        window_by_person = {}
        for person_id, results in results_by_person.items():
            past_results = [
                result for result in results if float(result.time_s) <= float(timestamp)
            ]
            window_by_person[str(person_id)] = past_results[-self.window_size :]
        return window_by_person

    def _guess_dynamic_factor(self, patient_window: list[RiskResult]) -> int:
        correlations = {}
        timestamps = np.array([result.time_s for result in patient_window], dtype=float)
        for node_id in LEGACY_DYNAMIC_NODES:
            values = np.array(
                [self._node_value(result, node_id) for result in patient_window],
                dtype=float,
            )
            correlations[node_id] = self._correlation(timestamps, values)

        finite_correlations = {
            node_id: corr
            for node_id, corr in correlations.items()
            if math.isfinite(corr)
        }
        if finite_correlations:
            return max(finite_correlations, key=finite_correlations.get)

        latest = patient_window[-1]
        return max(
            LEGACY_DYNAMIC_NODES,
            key=lambda node_id: self._node_value(latest, node_id),
        )

    def _guess_static_factor(
        self,
        window_by_person: dict[str, list[RiskResult]],
        target_person_id: str,
    ) -> int:
        best_node = None
        best_significance = -math.inf
        fallback_node = None
        fallback_value = -math.inf
        for node_id in LEGACY_STATIC_NODES:
            averages = {
                person_id: self._mean_node_value(results, node_id)
                for person_id, results in window_by_person.items()
                if results
            }
            averages = {
                person_id: value
                for person_id, value in averages.items()
                if math.isfinite(value)
            }
            if target_person_id not in averages or len(averages) < 2:
                continue
            target_value = averages[target_person_id]
            if target_value > fallback_value:
                fallback_node = node_id
                fallback_value = target_value
            risky_person = max(averages, key=averages.get)
            if risky_person != target_person_id:
                continue
            other_values = [
                value
                for person_id, value in averages.items()
                if person_id != target_person_id
            ]
            significance = abs(averages[target_person_id] - float(np.mean(other_values)))
            if significance <= 1e-12:
                continue
            if significance > best_significance:
                best_node = node_id
                best_significance = significance
        return best_node or fallback_node or ids.AGE_ATTRIBUTE_RISK

    def _format(
        self,
        person_id: str,
        static_node: int,
        dynamic_node: int,
    ) -> str:
        static_text = LEGACY_NODE_DESCRIPTIONS_JA.get(static_node, str(static_node))
        dynamic_text = LEGACY_NODE_DESCRIPTIONS_JA.get(dynamic_node, str(dynamic_node))
        return f"{person_id}さんが，{static_text}のに，{dynamic_text}ので，危険です．"

    def _mean_node_value(self, results: Iterable[RiskResult], node_id: int) -> float:
        values = [self._node_value(result, node_id) for result in results]
        finite_values = [value for value in values if math.isfinite(value)]
        if not finite_values:
            return math.nan
        return float(np.mean(finite_values))

    def _node_value(self, result: RiskResult, node_id: int) -> float:
        if node_id in result.factor_risks:
            return self._as_scalar(result.factor_risks[node_id])
        if node_id in result.upper_risks:
            return self._as_scalar(result.upper_risks[node_id])
        return math.nan

    def _as_scalar(self, value: object) -> float:
        if isinstance(value, (tuple, list)):
            if len(value) >= 2:
                return self._as_scalar(value[1])
            return math.nan
        try:
            return float(value)
        except (TypeError, ValueError):
            return math.nan

    def _correlation(self, xs: np.ndarray, ys: np.ndarray) -> float:
        mask = np.isfinite(xs) & np.isfinite(ys)
        if int(mask.sum()) < 2:
            return math.nan
        x_values = xs[mask]
        y_values = ys[mask]
        if np.allclose(x_values, x_values[0]) or np.allclose(y_values, y_values[0]):
            return math.nan
        return float(np.corrcoef(x_values, y_values)[0, 1])
