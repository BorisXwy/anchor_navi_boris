"""Post-run-only hidden geometry for rgb_only_v1 episodes.

Reads ``<episode_dir>/evaluation_only/evaluation_geometry.json``, which the
evaluation-only Habitat handle writes after the run.  Nothing here may be
imported by the online navigation path; it exists so that post-run auditors
can score rgb-only edges whose trajectory.json carries no pose by contract.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_hidden_geometry(episode_dir):
    """Return ``{target_index: {...}}`` pairing each hop's selected point
    with the pose where point navigation stopped; ``{}`` if the file is
    missing (crashed episodes never write it)."""
    path = Path(episode_dir) / "evaluation_only" / "evaluation_geometry.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    selected = {}
    ended = {}
    for event in payload.get("point_events", []):
        info = event.get("payload") or {}
        index = info.get("target_index")
        if index is None:
            continue
        index = int(index)
        if event.get("event") == "point_selected":
            selected[index] = {
                "start_xyz": event.get("position_xyz"),
                "start_yaw_rad": event.get("yaw_rad"),
                "selected_navmesh_xyz": event.get("selected_point_navmesh_xyz"),
                "selected_world_xyz": event.get("selected_point_world_xyz"),
            }
        elif event.get("event") == "point_navigation_stopped":
            ended[index] = {
                "end_xyz": event.get("position_xyz"),
                "end_yaw_rad": event.get("yaw_rad"),
                "policy_declared_arrival": bool(
                    info.get("policy_declared_arrival")),
                "end_reason": info.get("end_reason"),
                "final_target_geodesic_distance_m": event.get(
                    "final_target_geodesic_distance_m"),
            }
    result = {}
    for index, entry in ended.items():
        if entry.get("end_xyz") is None:
            continue
        result[index] = {**selected.get(index, {}), **entry}
    return result
