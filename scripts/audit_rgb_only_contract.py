#!/usr/bin/env python3
"""Fail-closed audit for navigation-facing artifacts of an RGB-only run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PRIVILEGED_KEY_PARTS = (
    "position_xyz", "base_yaw_rad", "absolute_yaw_rad", "depth_m",
    "world_xyz", "navmesh", "geodesic", "collision", "moved_m",
    "reference_path", "goal_position",
)


def _violations(value: Any, path: str = "root") -> list[str]:
    violations = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            lowered = str(key).lower()
            if child is not None and any(part in lowered
                                         for part in PRIVILEGED_KEY_PARTS):
                # A legacy graph schema retains a distance slot. Zero means
                # no metric value was observed or accumulated.
                if not (key == "traveled_distance_m" and child == 0.0):
                    violations.append(child_path)
            violations.extend(_violations(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            violations.extend(_violations(child, f"{path}[{index}]"))
    return violations


def audit_run(output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    trajectory_path = output_dir / "trajectory.json"
    graph_path = output_dir / "navigation_graph" / "navigation_graph.json"
    if not graph_path.exists():
        graph_path = output_dir / "navigation_graph" / "graph.json"
    if not trajectory_path.exists():
        raise FileNotFoundError(trajectory_path)
    trajectory = json.loads(trajectory_path.read_text())
    navigation_plane = {
        "motion_log": trajectory.get("motion_log", []),
        "targets": trajectory.get("targets", []),
    }
    violations = _violations(navigation_plane, "trajectory.navigation_plane")
    if graph_path.exists():
        graph = json.loads(graph_path.read_text())
        violations.extend(_violations(graph, "navigation_graph"))
        if graph.get("policy_input_contract") != "rgb_only_v1":
            violations.append("navigation_graph.policy_input_contract")
    else:
        violations.append("navigation_graph.missing")
    contract = trajectory.get("policy_input_contract", {})
    if contract.get("name") != "rgb_only_v1":
        violations.append("trajectory.policy_input_contract.name")
    result = {
        "passed": not violations,
        "policy_input_contract": "rgb_only_v1",
        "audited": [str(trajectory_path), str(graph_path)],
        "violations": violations,
        "evaluation_geometry_excluded_from_audit_plane": True,
    }
    (output_dir / "rgb_only_contract_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    result = audit_run(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
