"""Notification history serialization."""

from dataclasses import dataclass, field

import pandas as pd

from master_thesis_modules.risk_core.notification.notification_result import (
    NotificationResult,
)


@dataclass
class NotificationHistory:
    records: list[NotificationResult] = field(default_factory=list)

    def append_many(self, records: list[NotificationResult]) -> None:
        self.records.extend(record for record in records if record.should_notify)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for record in self.records:
            rows.append(
                {
                    "timestamp": record.timestamp,
                    "target_patient_id": record.person_id,
                    "total_risk": record.total_risk,
                    "rank": record.rank,
                    "reason": record.explanation,
                    "message": record.message or record.explanation,
                    "notification_type": record.notification_type,
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "timestamp",
                "target_patient_id",
                "total_risk",
                "rank",
                "reason",
                "message",
                "notification_type",
            ],
        )

    def save_csv(self, path) -> None:
        self.to_dataframe().to_csv(path, index=False)
