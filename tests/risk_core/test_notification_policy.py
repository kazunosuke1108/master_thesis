from master_thesis_modules.risk_core.engine.risk_result import RiskResult
from master_thesis_modules.risk_core.notification.notification_policy import NotificationPolicy
from master_thesis_modules.risk_core.schema import node_ids as ids


def _result(
    person_id: str,
    total_risk: float,
    standing: float = 0.0,
    touching_face: float = 0.0,
    staff_distance: float = 0.0,
    staff_not_watching: float = 0.0,
    timestamp: float = 0.0,
) -> RiskResult:
    return RiskResult(
        person_id=person_id,
        time_s=timestamp,
        factor_risks={
            ids.STANDING_RISK: standing,
            ids.TOUCHING_FACE_RISK: touching_face,
            ids.STAFF_DISTANCE_RISK: staff_distance,
            ids.STAFF_NOT_WATCHING_RISK: staff_not_watching,
        },
        upper_risks={},
        total_risk=total_risk,
        explanation=f"{person_id} explanation",
    )


def test_notice_requires_new_action_and_meaningful_risk_change():
    policy = NotificationPolicy()

    initial_notifications = policy.evaluate_timestep(
        0.0,
        [
            _result("A", 0.905, touching_face=0.316, staff_not_watching=0.9),
            _result("B", 0.818, touching_face=0.316),
            _result("C", 0.920, touching_face=0.316, staff_not_watching=0.9),
        ],
    )
    assert [
        notification
        for notification in initial_notifications
        if notification.notification_type == "notice"
    ] == []

    notifications = policy.evaluate_timestep(
        2.0,
        [
            _result("A", 0.923, touching_face=1.0, staff_not_watching=0.9, timestamp=2.0),
            _result("B", 0.853, touching_face=1.0, timestamp=2.0),
        ],
    )

    notice_targets = [
        notification.person_id
        for notification in notifications
        if notification.notification_type == "notice"
    ]
    assert notice_targets == ["A"]


def test_notice_allows_large_risk_rise_even_without_staff_context():
    policy = NotificationPolicy()
    policy.evaluate_timestep(0.0, [_result("B", 0.22)])

    notifications = policy.evaluate_timestep(
        2.0,
        [
            _result("B", 0.46, standing=1.0, timestamp=2.0),
        ],
    )

    notice_targets = [
        notification.person_id
        for notification in notifications
        if notification.notification_type == "notice"
    ]
    assert notice_targets == ["B"]
