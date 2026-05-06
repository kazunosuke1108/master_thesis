from master_thesis_modules.risk_core.factors.object_risk import (
    far_from_object_risk,
    near_object_risk,
)
from master_thesis_modules.risk_core.factors.staff_risk import (
    staff_distance_risk,
    staff_not_watching_risk,
)
from master_thesis_modules.risk_core.features.position import Position2D, Velocity2D


def test_iv_and_wheelchair_near_object_risk_increases_when_close():
    patient = Position2D(0.0, 0.0)

    assert near_object_risk(patient, Position2D(0.1, 0.0)) > near_object_risk(
        patient,
        Position2D(5.0, 0.0),
    )


def test_handrail_risk_increases_when_far():
    patient = Position2D(0.0, 0.0)

    assert far_from_object_risk(patient, Position2D(5.0, 0.0)) > far_from_object_risk(
        patient,
        Position2D(0.1, 0.0),
    )


def test_staff_distance_risk_increases_when_far():
    patient = Position2D(0.0, 0.0)

    assert staff_distance_risk(patient, Position2D(5.0, 0.0)) > staff_distance_risk(
        patient,
        Position2D(0.1, 0.0),
    )


def test_staff_watching_risk_low_when_moving_toward_patient():
    patient = Position2D(1.0, 0.0)
    staff = Position2D(0.0, 0.0)

    assert staff_not_watching_risk(patient, staff, Velocity2D(1.0, 0.0)) == 0.0


def test_staff_watching_risk_high_when_moving_away_from_patient():
    patient = Position2D(1.0, 0.0)
    staff = Position2D(0.0, 0.0)

    assert staff_not_watching_risk(patient, staff, Velocity2D(-1.0, 0.0)) == 1.0

