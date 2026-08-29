from master_thesis_modules.risk_core.factors.object_risk import (
    far_from_object_risk,
    near_object_risk,
)
from master_thesis_modules.risk_core.features.position import Position2D


def test_near_iv_and_wheelchair_are_higher_when_close():
    patient = Position2D(0, 0)

    assert near_object_risk(patient, Position2D(0.1, 0)) > near_object_risk(
        patient,
        Position2D(5, 0),
    )


def test_handrail_is_higher_when_far():
    patient = Position2D(0, 0)

    assert far_from_object_risk(patient, Position2D(5, 0)) > far_from_object_risk(
        patient,
        Position2D(0.1, 0),
    )


def test_object_risks_are_unit_range():
    patient = Position2D(0, 0)

    values = [
        near_object_risk(patient, Position2D(100, 0)),
        far_from_object_risk(patient, Position2D(100, 0)),
    ]
    assert all(0.0 <= value <= 1.0 for value in values)

