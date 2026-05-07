from master_thesis_modules.risk_core.factors.staff_risk import (
    staff_distance_risk,
    staff_not_watching_risk,
)
from master_thesis_modules.risk_core.features.position import Position2D, Velocity2D


def test_staff_distance_is_higher_when_far():
    patient = Position2D(0, 0)

    assert staff_distance_risk(patient, Position2D(5, 0)) > staff_distance_risk(
        patient,
        Position2D(0.1, 0),
    )


def test_staff_watch_loss_low_when_moving_toward_patient():
    assert staff_not_watching_risk(
        Position2D(1, 0),
        Position2D(0, 0),
        Velocity2D(1, 0),
    ) == 0.0


def test_staff_watch_loss_high_when_moving_away():
    assert staff_not_watching_risk(
        Position2D(1, 0),
        Position2D(0, 0),
        Velocity2D(-1, 0),
    ) == 1.0


def test_zero_velocity_uses_neutral_watch_risk():
    value = staff_not_watching_risk(
        Position2D(1, 0),
        Position2D(0, 0),
        Velocity2D(0, 0),
    )

    assert value == 0.5
