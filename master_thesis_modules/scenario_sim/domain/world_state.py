"""World state for one scenario timestamp."""

from dataclasses import dataclass

from master_thesis_modules.scenario_sim.domain.object_entity import ObjectEntity
from master_thesis_modules.scenario_sim.domain.patient import Patient
from master_thesis_modules.scenario_sim.domain.staff import Staff


@dataclass(frozen=True)
class WorldState:
    scenario_name: str
    time_s: float
    patients: tuple[Patient, ...]
    staff: tuple[Staff, ...]
    objects: tuple[ObjectEntity, ...]
    events: tuple[object, ...] = ()
    duration_s: float = 0.0
    step_s: float = 1.0
