"""Notification result dataclass."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NotificationResult:
    person_id: str
    total_risk: float
    should_notify: bool
    rank: int
    explanation: str
    timestamp: float = 0.0
    notification_type: str = "notice"
    message: str = ""
