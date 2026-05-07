"""Notification and support-request policy."""

from dataclasses import dataclass

from master_thesis_modules.risk_core.engine.risk_result import RiskResult
from master_thesis_modules.risk_core.notification.notification_result import (
    NotificationResult,
)


@dataclass(frozen=True)
class NotificationConfig:
    threshold: float = 0.45
    rising_delta: float = 0.08
    cooldown_s: float = 2.0
    support_threshold: float = 0.45
    support_patient_count: int = 2
    max_targets: int = 3


class NotificationPolicy:
    def __init__(
        self,
        threshold: float | None = None,
        max_targets: int | None = None,
        config: NotificationConfig | None = None,
    ) -> None:
        base = config or NotificationConfig()
        if threshold is not None:
            base = NotificationConfig(
                threshold=threshold,
                rising_delta=base.rising_delta,
                cooldown_s=base.cooldown_s,
                support_threshold=base.support_threshold,
                support_patient_count=base.support_patient_count,
                max_targets=base.max_targets,
            )
        if max_targets is not None:
            base = NotificationConfig(
                threshold=base.threshold,
                rising_delta=base.rising_delta,
                cooldown_s=base.cooldown_s,
                support_threshold=base.support_threshold,
                support_patient_count=base.support_patient_count,
                max_targets=max_targets,
            )
        self.config = base
        self._last_notice_time = -10**9
        self._last_help_time = -10**9
        self._previous_top_patient: str | None = None
        self._previous_risks: dict[str, float] = {}

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
        top_changed = self._previous_top_patient is not None and top.person_id != self._previous_top_patient
        previous_top_risk = self._previous_risks.get(top.person_id, top.total_risk)
        top_rising = top.total_risk - previous_top_risk >= self.config.rising_delta
        threshold_crossed = top.total_risk >= self.config.threshold
        cooldown_ok = timestamp - self._last_notice_time >= self.config.cooldown_s

        if cooldown_ok and threshold_crossed and (top_changed or top_rising or self._previous_top_patient is None):
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
        return notifications

