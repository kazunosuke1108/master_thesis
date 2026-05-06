"""Load semantic YAML/JSON scenarios without exposing node numbers."""

from __future__ import annotations

from pathlib import Path
import json

import yaml

from master_thesis_modules.risk_core.features.position import Position2D, Velocity2D
from master_thesis_modules.risk_core.features.pose_features import PoseFeatures
from master_thesis_modules.scenario_sim.domain.object_entity import ObjectEntity
from master_thesis_modules.scenario_sim.domain.patient import Patient
from master_thesis_modules.scenario_sim.domain.staff import Staff
from master_thesis_modules.scenario_sim.domain.world_state import WorldState
from master_thesis_modules.scenario_sim.events.scenario_event import ScenarioEvent


class ScenarioLoader:
    def load(self, path: str | Path) -> WorldState:
        path = Path(path)
        raw = self._load_raw(path)
        patients = tuple(self._patient(item) for item in raw.get("patients", []))
        staff = tuple(self._staff(item) for item in raw.get("staff", []))
        objects = tuple(self._object(item) for item in raw.get("objects", []))
        events = tuple(self._event(item) for item in raw.get("events", []))
        time_range = raw.get("time_range", {})
        return WorldState(
            scenario_name=raw.get("scenario_name", path.stem),
            time_s=float(raw.get("time_s", 0.0)),
            patients=patients,
            staff=staff,
            objects=objects,
            events=events,
            duration_s=float(time_range.get("duration_s", raw.get("duration_s", 0.0))),
            step_s=float(time_range.get("step_s", raw.get("step_s", 1.0))),
        )

    def _load_raw(self, path: Path) -> dict[str, object]:
        with path.open("r", encoding="utf-8") as handle:
            if path.suffix.lower() == ".json":
                return json.load(handle)
            return yaml.safe_load(handle)

    def _patient(self, item: dict[str, object]) -> Patient:
        return Patient(
            person_id=str(item["id"]),
            is_patient_label=_label(item.get("is_patient_label", "yes")),
            age_group_label=str(item.get("age_group_label", "middle")),
            position=_position(item["position"]),
            action_label=str(item["action_label"]) if item.get("action_label") else None,
            height_max=(
                float(item["height_max"]) if item.get("height_max") is not None else None
            ),
            pose_features=(
                _pose_features(item["pose_features"])
                if item.get("pose_features") is not None
                else None
            ),
        )

    def _staff(self, item: dict[str, object]) -> Staff:
        return Staff(
            staff_id=str(item["id"]),
            position=_position(item["position"]),
            velocity=_velocity(item.get("velocity", {"vx": 0.0, "vy": 0.0})),
        )

    def _object(self, item: dict[str, object]) -> ObjectEntity:
        return ObjectEntity(
            object_id=str(item["id"]),
            object_type=str(item["type"]),
            position=_position(item["position"]),
        )

    def _event(self, item: dict[str, object]) -> ScenarioEvent:
        payload = dict(item.get("payload", {}))
        for key in (
            "action_label",
            "position",
            "velocity",
            "standing_degree",
            "trunk_tilt",
            "wrist_distance_from_hip",
            "ankle_spread",
        ):
            if key in item:
                payload[key] = item[key]
        return ScenarioEvent(
            time_s=float(item["time_s"]),
            target_id=str(item["target_id"]),
            event_type=str(item["event_type"]),
            payload=payload,
        )


def _position(raw: dict[str, object]) -> Position2D:
    return Position2D(float(raw["x"]), float(raw["y"]))


def _velocity(raw: dict[str, object]) -> Velocity2D:
    return Velocity2D(float(raw.get("vx", 0.0)), float(raw.get("vy", 0.0)))


def _pose_features(raw: dict[str, object]) -> PoseFeatures:
    return PoseFeatures(
        float(raw["standing_degree"]),
        float(raw["trunk_tilt"]),
        float(raw["wrist_distance_from_hip"]),
        float(raw["ankle_spread"]),
    )


def _label(raw: object) -> str:
    if raw is True:
        return "yes"
    if raw is False:
        return "no"
    return str(raw)
