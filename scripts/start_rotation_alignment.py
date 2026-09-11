#!/usr/bin/env python3
"""Rewrite an R2R episode ``start_rotation`` so that, after executing the
instruction's opening turn, the agent faces the initial direction of the
ground-truth ``reference_path``.

Pure math plus the deterministic instruction taxonomy; no Habitat dependency.
The rule is form-level (project_rulle.md 16.4): every episode of a given
opening form is treated identically, and forms whose turn angle is undefined
keep the official rotation.

Conventions (Habitat / VLN-CE): quaternion coefficients are ``[x, y, z, w]``
rotating about +y only, forward is -z and left is -x, so a heading of ``yaw``
radians (left positive) points along ``(-sin yaw, 0, -cos yaw)``.
"""

import copy
import math

from instruction_taxonomy import decompose_by_definition

OPENING_TURN_OFFSET_DEG = {
    "TURN_LEFT": 90.0,
    "TURN_RIGHT": -90.0,
    "TURN_AROUND": 180.0,
}
# "Turn to face the door" / "Face the grill" fix the heading by a landmark the
# dataset does not localise, so no turn angle can be assigned.
UNDEFINED_OPENING_FORMS = frozenset({"TURN_TO_LANDMARK", "OTHER"})
GT_BEARING_MIN_DISTANCE_M = 1.0
STATUS_REWRITTEN = "rewritten"
STATUS_KEPT_OFFICIAL = "kept_official"


def wrap_angle(value):
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion_coeffs(coeffs):
    x, y, z, w = (float(v) for v in coeffs[:4])
    return wrap_angle(2.0 * math.atan2(y, w))


def quaternion_coeffs_from_yaw(yaw):
    half = 0.5 * wrap_angle(yaw)
    return [0.0, math.sin(half), 0.0, math.cos(half)]


def bearing_between(origin, target):
    dx = float(target[0]) - float(origin[0])
    dz = float(target[2]) - float(origin[2])
    return wrap_angle(math.atan2(-dx, -dz))


def gt_initial_bearing(reference_path, min_distance_m=GT_BEARING_MIN_DISTANCE_M):
    """Bearing from the start to the first waypoint at least ``min_distance_m``
    along the polyline (the last waypoint when the path is shorter).

    Returns ``(bearing_rad, anchor_index, anchor_distance_m)``.
    """
    if len(reference_path) < 2:
        raise ValueError("reference_path needs at least two waypoints")
    start = reference_path[0]
    cumulative = 0.0
    anchor_index = len(reference_path) - 1
    for index in range(1, len(reference_path)):
        previous, current = reference_path[index - 1], reference_path[index]
        cumulative += math.hypot(float(current[0]) - float(previous[0]),
                                 float(current[2]) - float(previous[2]))
        if cumulative >= min_distance_m:
            anchor_index = index
            break
    anchor = reference_path[anchor_index]
    if math.hypot(float(anchor[0]) - float(start[0]),
                  float(anchor[2]) - float(start[2])) <= 1e-9:
        raise ValueError("reference_path anchor coincides with the start")
    return bearing_between(start, anchor), anchor_index, cumulative


def classify_opening_form(instruction_text):
    stages = decompose_by_definition(instruction_text or "")
    return stages[0]["form"] if stages else "OTHER"


def align_episode(episode, min_distance_m=GT_BEARING_MIN_DISTANCE_M):
    """Return ``(aligned_episode, audit_row)``; only ``start_rotation`` may differ."""
    aligned = copy.deepcopy(episode)
    instruction = (episode.get("instruction") or {}).get("instruction_text", "")
    form = classify_opening_form(instruction)
    bearing, anchor_index, anchor_distance = gt_initial_bearing(
        episode["reference_path"], min_distance_m)
    official_yaw = yaw_from_quaternion_coeffs(episode["start_rotation"])

    if form in UNDEFINED_OPENING_FORMS:
        offset_deg = None
        aligned_yaw = official_yaw
        status = STATUS_KEPT_OFFICIAL
    else:
        offset_deg = OPENING_TURN_OFFSET_DEG.get(form, 0.0)
        aligned_yaw = wrap_angle(bearing - math.radians(offset_deg))
        aligned["start_rotation"] = quaternion_coeffs_from_yaw(aligned_yaw)
        status = STATUS_REWRITTEN

    audit_row = {
        "episode_id": episode["episode_id"],
        "instruction": instruction,
        "opening_form": form,
        "turn_offset_deg": offset_deg,
        "gt_anchor_index": anchor_index,
        "gt_anchor_distance_m": round(anchor_distance, 3),
        "gt_bearing_deg": round(math.degrees(bearing), 2),
        "official_yaw_deg": round(math.degrees(official_yaw), 2),
        "aligned_yaw_deg": round(math.degrees(aligned_yaw), 2),
        "delta_deg": round(math.degrees(wrap_angle(aligned_yaw - official_yaw)), 2),
        "status": status,
    }
    return aligned, audit_row
