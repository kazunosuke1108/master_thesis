"""Draw comic-strip style scenario snapshots from semantic YAML scenarios."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import pandas as pd

from master_thesis_modules.risk_core.features.position import Position2D
from master_thesis_modules.scenario_sim.domain.world_state import WorldState
from master_thesis_modules.scenario_sim.events.event_engine import EventEngine
from master_thesis_modules.scenario_sim.events.scenario_event import ScenarioEvent


ROOM_MIN_X = 0.0
ROOM_MAX_X = 7.0
STORYBOARD_MAX_X = 8.0
ROOM_MIN_Y = 0.0
WALL_DISPLAY_MARGIN = 0.8


@dataclass(frozen=True)
class ScenarioSnapshot:
    time_s: float
    world_state: WorldState
    events: tuple[ScenarioEvent, ...]
    previous_positions: dict[str, Position2D]


def visualize_scenario_storyboard(
    world_state: WorldState,
    output_png: str | Path,
    output_csv: str | Path | None = None,
    columns: int = 4,
) -> dict[str, Path]:
    """Create a storyboard PNG plus an optional snapshot table."""

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    snapshots = build_scenario_snapshots(world_state)
    _plot_snapshots(world_state, snapshots, output_png, columns=columns)
    paths = {"scenario_storyboard": output_png}
    if output_csv is not None:
        output_csv = Path(output_csv)
        build_snapshot_table(snapshots).to_csv(output_csv, index=False)
        paths["scenario_storyboard_snapshots"] = output_csv
    return paths


def build_scenario_snapshots(world_state: WorldState) -> list[ScenarioSnapshot]:
    """Return world states immediately after each key scenario time."""

    event_engine = EventEngine()
    events = tuple(world_state.events)
    key_times = sorted(
        {
            0.0,
            world_state.duration_s,
            *(event.time_s for event in events if isinstance(event, ScenarioEvent)),
        }
    )
    current = replace(world_state, events=())
    applied_event_ids: set[int] = set()
    previous_positions = _entity_positions(current)
    snapshots: list[ScenarioSnapshot] = []
    previous_time = -float("inf")
    for time_s in key_times:
        event_pairs = tuple(
            (event_index, event)
            for event_index, event in enumerate(events)
            if (
                isinstance(event, ScenarioEvent)
                and event_index not in applied_event_ids
                and previous_time < event.time_s <= time_s
            )
        )
        current = event_engine.apply(current, tuple(event for _, event in event_pairs))
        current_at_time = replace(current, time_s=time_s)
        snapshots.append(
            ScenarioSnapshot(
                time_s=time_s,
                world_state=current_at_time,
                events=tuple(event for _, event in event_pairs),
                previous_positions=previous_positions,
            )
        )
        previous_positions = _entity_positions(current_at_time)
        for event_index, _ in event_pairs:
            applied_event_ids.add(event_index)
        previous_time = time_s
    return snapshots


def build_snapshot_table(snapshots: list[ScenarioSnapshot]) -> pd.DataFrame:
    rows = []
    for snapshot in snapshots:
        event_summary = "; ".join(_event_label(event) for event in snapshot.events)
        for patient in snapshot.world_state.patients:
            rows.append(
                {
                    "timestamp": snapshot.time_s,
                    "entity_type": "patient",
                    "entity_id": patient.person_id,
                    "x": patient.position.x,
                    "y": patient.position.y,
                    "action_label": patient.action_label,
                    "vx": "",
                    "vy": "",
                    "object_type": "",
                    "events": event_summary,
                }
            )
        for staff in snapshot.world_state.staff:
            rows.append(
                {
                    "timestamp": snapshot.time_s,
                    "entity_type": "staff",
                    "entity_id": staff.staff_id,
                    "x": staff.position.x,
                    "y": staff.position.y,
                    "action_label": "",
                    "vx": staff.velocity.vx,
                    "vy": staff.velocity.vy,
                    "object_type": "",
                    "events": event_summary,
                }
            )
        for item in snapshot.world_state.objects:
            rows.append(
                {
                    "timestamp": snapshot.time_s,
                    "entity_type": "object",
                    "entity_id": item.object_id,
                    "x": item.position.x,
                    "y": item.position.y,
                    "action_label": "",
                    "vx": "",
                    "vy": "",
                    "object_type": item.object_type,
                    "events": event_summary,
                }
            )
    return pd.DataFrame(rows)


def _plot_snapshots(
    world_state: WorldState,
    snapshots: list[ScenarioSnapshot],
    output_png: Path,
    columns: int,
) -> None:
    rows = math.ceil(len(snapshots) / columns)
    bounds = _plot_bounds(snapshots)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.6 * columns, 4.8 * rows),
        squeeze=False,
    )
    axes_list = list(axes.ravel())
    for ax, snapshot in zip(axes_list, snapshots):
        _draw_snapshot(ax, snapshot, bounds)
    for ax in axes_list[len(snapshots) :]:
        ax.axis("off")
    fig.suptitle(f"Scenario storyboard: {world_state.scenario_name}", fontsize=15)
    fig.legend(
        handles=_storyboard_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=5,
        frameon=True,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.96), h_pad=2.6)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def _draw_snapshot(
    ax: Axes,
    snapshot: ScenarioSnapshot,
    bounds: tuple[float, float, float, float],
) -> None:
    min_x, max_x, min_y, max_y = bounds
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect("equal", adjustable="box")
    _draw_wall_regions(ax, bounds)
    ax.grid(True, alpha=0.25)
    ax.set_title(f"t = {snapshot.time_s:g}s", fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    object_label_offsets = _object_label_offsets(snapshot.world_state.objects)
    for item, (offset_x, offset_y, horizontal_alignment) in zip(
        snapshot.world_state.objects,
        object_label_offsets,
    ):
        marker, color = _object_style(item.object_type)
        ax.scatter(
            item.position.x,
            item.position.y,
            marker=marker,
            s=90,
            color=color,
            edgecolor="#333333",
            linewidth=0.7,
            zorder=2,
        )
        ax.annotate(
            f"{item.object_type}\n{item.object_id}",
            xy=(item.position.x, item.position.y),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            ha=horizontal_alignment,
            va="top",
            fontsize=7,
            color="#333333",
            bbox={
                "boxstyle": "round,pad=0.12",
                "fc": "#ffffff",
                "ec": "none",
                "alpha": 0.72,
            },
            zorder=3,
        )
        _draw_position_delta(
            ax,
            f"object:{item.object_id}",
            item.position,
            snapshot.previous_positions,
            color=color,
            linestyle=":",
        )

    for patient in snapshot.world_state.patients:
        ax.scatter(
            patient.position.x,
            patient.position.y,
            marker="o",
            s=260,
            color="#f4a259",
            edgecolor="#5f3712",
            linewidth=1.2,
            zorder=4,
        )
        ax.text(
            patient.position.x,
            patient.position.y,
            patient.person_id,
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
            color="#1f1f1f",
            zorder=5,
        )
        _draw_position_delta(
            ax,
            f"patient:{patient.person_id}",
            patient.position,
            snapshot.previous_positions,
            color="#a55a13",
        )
        ax.text(
            patient.position.x,
            patient.position.y + 0.25,
            patient.action_label or "neutral_sitting",
            ha="center",
            va="bottom",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.18", "fc": "#fff4e8", "ec": "#d08439"},
            zorder=6,
        )

    for staff in snapshot.world_state.staff:
        ax.scatter(
            staff.position.x,
            staff.position.y,
            marker="s",
            s=190,
            color="#6aa6d8",
            edgecolor="#1e4f7a",
            linewidth=1.2,
            zorder=4,
        )
        ax.text(
            staff.position.x,
            staff.position.y,
            staff.staff_id,
            ha="center",
            va="center",
            fontsize=9,
            weight="bold",
            color="#0b2239",
            zorder=5,
        )
        if staff.velocity.norm > 0.0:
            ax.arrow(
                staff.position.x,
                staff.position.y,
                staff.velocity.vx,
                staff.velocity.vy,
                length_includes_head=True,
                head_width=0.12,
                head_length=0.18,
                color="#1e4f7a",
                linewidth=1.2,
                zorder=6,
            )
        _draw_position_delta(
            ax,
            f"staff:{staff.staff_id}",
            staff.position,
            snapshot.previous_positions,
            color="#1e4f7a",
            linestyle="--",
        )
        if snapshot.world_state.scenario_name not in {
            "20260807_standup2",
            "20260807_touchface2",
        }:
            ax.text(
                staff.position.x,
                staff.position.y + 0.25,
                f"v=({staff.velocity.vx:g},{staff.velocity.vy:g})",
                ha="center",
                va="bottom",
                fontsize=8,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "fc": "#eef7ff",
                    "ec": "#6aa6d8",
                },
                zorder=6,
            )

    event_text = "\n".join(_event_label(event) for event in snapshot.events)
    if event_text:
        ax.text(
            0.98,
            0.98,
            event_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.25",
                "fc": "#ffffff",
                "ec": "#bbbbbb",
                "alpha": 0.9,
            },
            zorder=7,
        )


def _object_label_offsets(objects: tuple[object, ...]) -> list[tuple[float, float, str]]:
    """Return non-overlapping point offsets for labels at the same position."""

    coordinate_groups: dict[tuple[float, float], list[int]] = {}
    for index, item in enumerate(objects):
        key = (round(item.position.x, 6), round(item.position.y, 6))
        coordinate_groups.setdefault(key, []).append(index)

    offsets: list[tuple[float, float, str]] = [(0.0, -10.0, "center")] * len(objects)
    for indices in coordinate_groups.values():
        if len(indices) == 1:
            offsets[indices[0]] = (0.0, -10.0, "center")
            continue
        for group_index, object_index in enumerate(indices):
            row = group_index // 2
            if group_index % 2 == 0:
                offsets[object_index] = (-7.0, -10.0 - 23.0 * row, "right")
            else:
                offsets[object_index] = (7.0, -10.0 - 23.0 * row, "left")
    return offsets


def _draw_wall_regions(
    ax: Axes,
    bounds: tuple[float, float, float, float],
) -> None:
    """Shade the regions outside the scenario room as walls."""

    min_x, max_x, min_y, max_y = bounds
    wall_style = {
        "facecolor": "#e6e6e6",
        "edgecolor": "#b8b8b8",
        "alpha": 0.55,
        "hatch": "////",
        "linewidth": 0.0,
        "zorder": 0,
    }
    ax.axvspan(min_x, ROOM_MIN_X, **wall_style)
    ax.axvspan(ROOM_MAX_X, max_x, **wall_style)
    ax.axhspan(min_y, ROOM_MIN_Y, **wall_style)
    ax.axvline(ROOM_MIN_X, color="#9a9a9a", linewidth=1.0, zorder=1)
    ax.axvline(ROOM_MAX_X, color="#9a9a9a", linewidth=1.0, zorder=1)
    ax.axhline(ROOM_MIN_Y, color="#9a9a9a", linewidth=1.0, zorder=1)


def _draw_position_delta(
    ax: Axes,
    entity_key: str,
    position: Position2D,
    previous_positions: dict[str, Position2D],
    color: str,
    linestyle: str = "-",
) -> None:
    previous = previous_positions.get(entity_key)
    if previous is None:
        return
    dx = position.x - previous.x
    dy = position.y - previous.y
    if math.hypot(dx, dy) <= 1e-9:
        return
    ax.annotate(
        "",
        xy=(position.x, position.y),
        xytext=(previous.x, previous.y),
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "linewidth": 1.4,
            "linestyle": linestyle,
        },
        zorder=3,
    )


def _plot_bounds(snapshots: list[ScenarioSnapshot]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for snapshot in snapshots:
        for position in _entity_positions(snapshot.world_state).values():
            xs.append(position.x)
            ys.append(position.y)
    if not xs or not ys:
        return (
            ROOM_MIN_X - WALL_DISPLAY_MARGIN,
            STORYBOARD_MAX_X,
            ROOM_MIN_Y - WALL_DISPLAY_MARGIN,
            1.0,
        )
    margin = 0.8
    min_x = min(min(xs) - margin, ROOM_MIN_X - WALL_DISPLAY_MARGIN)
    max_x = max(max(xs) + margin, STORYBOARD_MAX_X)
    min_y = min(min(ys) - margin, ROOM_MIN_Y - WALL_DISPLAY_MARGIN)
    max_y = max(ys) + margin
    if math.isclose(min_x, max_x):
        min_x -= 1.0
        max_x += 1.0
    if math.isclose(min_y, max_y):
        min_y -= 1.0
        max_y += 1.0
    return min_x, max_x, min_y, max_y


def _entity_positions(world_state: WorldState) -> dict[str, Position2D]:
    positions: dict[str, Position2D] = {}
    positions.update(
        {f"patient:{patient.person_id}": patient.position for patient in world_state.patients}
    )
    positions.update({f"staff:{staff.staff_id}": staff.position for staff in world_state.staff})
    positions.update(
        {f"object:{item.object_id}": item.position for item in world_state.objects}
    )
    return positions


def _object_style(object_type: str) -> tuple[str, str]:
    if object_type == "wheelchair":
        return "D", "#9b8acb"
    if object_type == "iv_pole":
        return "^", "#73b37d"
    if object_type == "handrail":
        return "P", "#b7a071"
    return "X", "#a0a0a0"


def _storyboard_legend_handles() -> list[Line2D]:
    styles = (
        ("Patient", "o", "#f4a259", "#5f3712", 10),
        ("Staff", "s", "#6aa6d8", "#1e4f7a", 9),
        ("Wheelchair", "D", "#9b8acb", "#333333", 7),
        ("IV pole", "^", "#73b37d", "#333333", 7),
        ("Handrail", "P", "#b7a071", "#333333", 7),
    )
    return [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=marker,
            markersize=marker_size,
            markerfacecolor=facecolor,
            markeredgecolor=edgecolor,
            label=label,
        )
        for label, marker, facecolor, edgecolor, marker_size in styles
    ]


def _event_label(event: ScenarioEvent) -> str:
    if event.event_type == "set_action":
        return f"{event.target_id}: {event.payload.get('action_label')}"
    if event.event_type in {"move_patient", "move_staff", "set_object_position"}:
        position = event.payload.get("position", event.payload)
        if isinstance(position, dict):
            return f"{event.target_id}: move to ({float(position['x']):g}, {float(position['y']):g})"
    if event.event_type == "set_pose_features":
        return f"{event.target_id}: pose_features"
    return f"{event.target_id}: {event.event_type}"
