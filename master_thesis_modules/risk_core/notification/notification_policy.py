"""Notification and support-request policy."""

from dataclasses import dataclass, field, replace

from master_thesis_modules.risk_core.engine.risk_result import RiskResult
from master_thesis_modules.risk_core.notification.notification_result import (
    NotificationResult,
)
from master_thesis_modules.risk_core.schema import node_ids as ids


@dataclass(frozen=True)
class NotificationConfig:
    threshold: float = 0.30
    rising_delta: float = 0.08
    cooldown_s: float = 2.0
    support_threshold: float = 0.45
    support_patient_count: int = 2
    max_targets: int = 3
    action_threshold: float = 0.8
    staff_context_threshold: float = 0.5
    strong_rising_delta: float = 0.15
    staff_context_rising_delta: float = 0.01
    active_action_nodes: tuple[int, ...] = field(
        default_factory=lambda: tuple(ids.ACTION_RISK_NODES)
    )


class NotificationPolicy:
    def __init__(
        self,
        threshold: float | None = None,
        max_targets: int | None = None,
        config: NotificationConfig | None = None,
    ) -> None:
        base = config or NotificationConfig()
        if threshold is not None:
            base = replace(base, threshold=threshold)
        if max_targets is not None:
            base = replace(base, max_targets=max_targets)
        self.config = base
        self._last_notice_time = -10**9
        self._last_help_time = -10**9
        self._previous_top_patient: str | None = None
        self._previous_risks: dict[str, float] = {}
        self._previous_action_risks: dict[tuple[str, int], float] = {}

    def evaluate(self, results: list[RiskResult]) -> list[NotificationResult]:
        ranked = sorted(results, key=lambda result: result.total_risk, reverse=True)
        notifications = []
        for rank, result in enumerate(ranked, start=1):
            notifications.append(
                NotificationResult(
                    person_id=result.person_id,
                    total_risk=result.total_risk,
                    should_notify=(
                        result.total_risk >= self.config.threshold
                        and rank <= self.config.max_targets
                    ),
                    rank=rank,
                    explanation=result.explanation,
                    timestamp=result.time_s,
                    message=result.explanation,
                )
            )
        return notifications

    def evaluate_timestep(
        self,
        timestamp: float,
        results: list[RiskResult],
        staff_count: int = 1,
    ) -> list[NotificationResult]:
        ranked = sorted(results, key=lambda result: result.total_risk, reverse=True)
        if not ranked:
            return []
        rank_by_person = {result.person_id: idx for idx, result in enumerate(ranked, start=1)}
        notifications: list[NotificationResult] = []
        top = ranked[0]

        cooldown_ok = timestamp - self._last_notice_time >= self.config.cooldown_s
        if self.config.active_action_nodes:
            notice_candidates = [
                result
                for rank, result in enumerate(ranked, start=1)
                if self._should_notify_action_event(result, rank)
            ][: self.config.max_targets]
            if cooldown_ok and notice_candidates:
                for result in notice_candidates:
                    notifications.append(
                        NotificationResult(
                            person_id=result.person_id,
                            total_risk=result.total_risk,
                            should_notify=True,
                            rank=rank_by_person[result.person_id],
                            explanation=result.explanation,
                            timestamp=timestamp,
                            notification_type="notice",
                            message=result.explanation,
                        )
                    )
                self._last_notice_time = timestamp
        else:
            top = ranked[0]
            top_changed = (
                self._previous_top_patient is not None
                and top.person_id != self._previous_top_patient
            )
            previous_top_risk = self._previous_risks.get(top.person_id, top.total_risk)
            top_rising = top.total_risk - previous_top_risk >= self.config.rising_delta
            threshold_crossed = top.total_risk >= self.config.threshold

            if (
                cooldown_ok
                and threshold_crossed
                and (top_changed or top_rising or self._previous_top_patient is None)
            ):
                notifications.append(
                    NotificationResult(
                        person_id=top.person_id,
                        total_risk=top.total_risk,
                        should_notify=True,
                        rank=rank_by_person[top.person_id],
                        explanation=top.explanation,
                        timestamp=timestamp,
                        notification_type="notice",
                        message=top.explanation,
                    )
                )
                self._last_notice_time = timestamp

        risky_patients = [
            result
            for result in ranked
            if result.total_risk >= self.config.support_threshold
        ]
        help_cooldown_ok = timestamp - self._last_help_time >= self.config.cooldown_s
        if (
            help_cooldown_ok
            and len(risky_patients) > max(staff_count, self.config.support_patient_count - 1)
        ):
            message = "デイルームで複数の患者さんの対応が必要です。デイルームに来てください。"
            notifications.append(
                NotificationResult(
                    person_id="",
                    total_risk=max(result.total_risk for result in risky_patients),
                    should_notify=True,
                    rank=0,
                    explanation=message,
                    timestamp=timestamp,
                    notification_type="help",
                    message=message,
                )
            )
            self._last_help_time = timestamp

        self._previous_top_patient = top.person_id
        self._previous_risks = {result.person_id: result.total_risk for result in ranked}
        self._previous_action_risks = {
            (result.person_id, node_id): result.factor_risks.get(node_id, 0.0)
            for result in ranked
            for node_id in self.config.active_action_nodes
        }
        return notifications

    def _should_notify_action_event(self, result: RiskResult, rank: int) -> bool:
        if result.total_risk < self.config.threshold:
            return False
        if not any(
            self._is_new_active_action(result, node_id)
            for node_id in self.config.active_action_nodes
        ):
            return False
        previous_total = self._previous_risks.get(result.person_id, result.total_risk)
        total_delta = result.total_risk - previous_total
        return (
            total_delta >= self.config.strong_rising_delta
            or (
                self._has_insufficient_staff_context(result)
                and total_delta >= self.config.staff_context_rising_delta
            )
        )

    def _is_new_active_action(self, result: RiskResult, node_id: int) -> bool:
        current = result.factor_risks.get(node_id, 0.0)
        previous = self._previous_action_risks.get((result.person_id, node_id), 0.0)
        return (
            current >= self.config.action_threshold
            and previous < self.config.action_threshold
        )

    def _has_insufficient_staff_context(self, result: RiskResult) -> bool:
        return max(
            result.factor_risks.get(ids.STAFF_DISTANCE_RISK, 0.0),
            result.factor_risks.get(ids.STAFF_NOT_WATCHING_RISK, 0.0),
        ) >= self.config.staff_context_threshold
