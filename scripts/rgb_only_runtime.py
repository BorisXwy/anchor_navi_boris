#!/usr/bin/env python3
"""Hard runtime boundary between an RGB navigation policy and Habitat.

The wrapped simulator intentionally exposes only RGB observations and the
three discrete embodied actions.  Pose, depth, semantic sensors, pathfinder,
navmesh and collision state remain owned by the evaluation process holding the
raw simulator.  Navigation code receives this facade, so accidental privileged
access fails immediately instead of silently contaminating an experiment.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


RGB_SENSOR_PREFIXES = ("rgb", "pano_rgb_", "completion_rgb_")
ALLOWED_ACTIONS = frozenset({"move_forward", "turn_left", "turn_right"})
FORBIDDEN_ATTRIBUTES = frozenset({
    "pathfinder", "navmesh", "get_agent", "agents", "semantic_scene",
    "get_gravity", "get_physics_contact_points",
})


def _is_rgb_sensor(key: str) -> bool:
    value = str(key)
    return value == "rgb" or value.startswith("pano_rgb_") or value.startswith(
        "completion_rgb_")


def rgb_observations_only(observations: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Copy only RGB arrays from a raw Habitat observation dictionary."""
    result = {
        str(key): np.asarray(value)[..., :3]
        for key, value in observations.items()
        if _is_rgb_sensor(str(key))
    }
    if "rgb" not in result:
        raise RuntimeError("RGB-only policy observation is missing primary 'rgb'")
    return result


@dataclass
class RGBOnlyPolicySimulator:
    """Capability-limited simulator handle supplied to online navigation.

    ``_sim`` is deliberately private and must only be retained by the Habitat
    adapter/evaluator.  Policy modules can observe RGB or issue an action.  No
    API exists for querying whether that action changed pose or collided.
    """

    _sim: Any
    _evaluation_hook: Optional[Callable[[str, Any], None]] = None
    _evaluation_event_hook: Optional[Callable[[str, dict, Any], None]] = None
    action_audit: list[dict] = field(default_factory=list)

    policy_input_contract: str = "rgb_only_v1"

    def get_sensor_observations(self) -> dict[str, np.ndarray]:
        return rgb_observations_only(self._sim.get_sensor_observations())

    def step(self, action: str) -> dict[str, np.ndarray]:
        action = str(action)
        if action not in ALLOWED_ACTIONS:
            raise ValueError(
                f"RGB-only policy may issue only {sorted(ALLOWED_ACTIONS)}, got {action!r}")
        raw = self._sim.step(action)
        if self._evaluation_hook is not None:
            self._evaluation_hook(action, self._sim)
        self.action_audit.append({
            "action_index": len(self.action_audit),
            "action": action,
            "feedback": "rgb_observation_only",
        })
        return rgb_observations_only(raw)

    def emit_evaluation_event(self, event: str, payload: Mapping[str, Any]) -> None:
        """Send a write-only marker to the test harness.

        The callback's return value is intentionally discarded.  This lets a
        benchmark attach hidden geometric labels to an RGB policy decision
        without creating a channel through which the policy can read them.
        """
        if self._evaluation_event_hook is not None:
            self._evaluation_event_hook(str(event), dict(payload), self._sim)

    def __getattr__(self, name: str):
        if name in FORBIDDEN_ATTRIBUTES or any(token in name.lower() for token in (
                "pose", "position", "rotation", "depth", "path", "navmesh",
                "geodesic", "collision")):
            raise RuntimeError(
                f"privileged Habitat attribute {name!r} is forbidden by "
                f"{self.policy_input_contract}")
        raise AttributeError(name)


def require_rgb_only_policy_sim(sim: Any) -> RGBOnlyPolicySimulator:
    if not isinstance(sim, RGBOnlyPolicySimulator):
        raise TypeError(
            "online navigation requires RGBOnlyPolicySimulator; raw Habitat "
            "simulator handles are evaluation-only")
    return sim


@dataclass
class RGBOnlyEvaluationVideoBridge:
    """Write-only bridge from policy frames to evaluator-owned composition."""

    _emit_callback: Callable[..., None]

    def emit(self, obs_bgr: np.ndarray, instruction: str, target_index: int,
             phase: str, *, full_instruction: Optional[str] = None,
             sub_instruction: Optional[str] = None, repeat: int = 1) -> None:
        self._emit_callback(
            np.asarray(obs_bgr, np.uint8), str(instruction),
            int(target_index), str(phase),
            full_instruction=full_instruction,
            sub_instruction=sub_instruction, repeat=int(repeat))

    def __getattr__(self, name: str):
        raise RuntimeError(
            f"RGB-only policy video bridge is write-only; {name!r} is not exposed")
