#!/usr/bin/env python3
"""Runtime random and instruction/VLM point selectors for Habitat."""

from dataclasses import dataclass, field
from typing import Any, Optional
import math
import re

import cv2
import habitat_sim
import numpy as np
from habitat_sim.utils.common import quat_from_angle_axis

from semantic_detector import extract_detection_queries
from semantic_point_strategy import (
    apply_cross_view_semantic_policy, constrain_floor_candidates,
)
from path_projection import draw_reference_path_overlay, project_reference_path

def wrap_angle(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def angle_distance(a, b):
    return abs(wrap_angle(a - b))


def route_backtrack_exclusion(stage, base_exclusion_rad):
    """Return the incoming-lane exclusion for one semantic route clause.

    Straight/pass/follow clauses normally stay in the forward half-plane.  A
    FOLLOW clause beginning with an explicit but unqualified ``turn`` first
    has to expose a side tangent; treating it as straight can remove the real
    +/-135-degree railing/corridor view before RGB analysis.  This parser is
    form/text based and uses neither depth nor a reference trajectory.
    """
    form = str((stage or {}).get("form", "")).upper()
    instruction = str((stage or {}).get(
        "navigation_instruction", "")).strip().lower()
    metadata = (stage or {}).get("metadata", {}) or {}
    bare_turn_route_setup = bool(
        (metadata.get("bare_turn_route_setup", {}) or {}).get("active"))
    if bare_turn_route_setup:
        # The immediately preceding explicit turn already commits the legal
        # outgoing half-line.  Its instruction-locked corridor is applied at
        # pixel-ray level by the VLM harness, so an incoming-direction cone
        # must not erase the tangent doorway at a 90-degree corner.
        return 0.0
    leading_turn_follow = bool(
        form == "FOLLOW_PATH_BOUNDARY" and
        re.match(r"^(?:then\s+)?turn\b", instruction))
    if (form in {"PASS_LANDMARK", "ADVANCE_STRAIGHT",
                 "FOLLOW_PATH_BOUNDARY"} and not leading_turn_follow and
            not bare_turn_route_setup):
        return max(float(base_exclusion_rad), math.radians(90.0))
    return float(base_exclusion_rad)


def set_pose(sim, position, yaw):
    state = sim.get_agent(0).get_state()
    state.position = np.asarray(position, np.float32)
    state.rotation = quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
    sim.get_agent(0).set_state(state)


def observe(sim):
    return sim.get_sensor_observations()["rgb"][..., :3]



def observe_six_rgbd(sim):
    observations = sim.get_sensor_observations()
    rgbs = [observations["rgb"][..., :3]] + [
        observations[f"pano_rgb_{index}"][..., :3] for index in range(1, 6)]
    depths = [observations["depth"]] + [
        observations[f"pano_depth_{index}"] for index in range(1, 6)]
    return rgbs, depths


def observe_six_rgb(sim):
    """Capture six RGB sectors without requesting a depth observation."""
    observations = sim.get_sensor_observations()
    return [observations["rgb"][..., :3]] + [
        observations[f"pano_rgb_{index}"][..., :3] for index in range(1, 6)]


EIGHT_VIEW_YAW_OFFSETS_DEG = (0, 45, 90, 135, 180, 225, 270, 315)


def observe_eight_rgb(sim):
    """Capture the eight 45-degree RGB compass sectors."""
    observations = sim.get_sensor_observations()
    return [observations["rgb"][..., :3]] + [
        observations[f"completion_rgb_{index}"][..., :3]
        for index in range(1, 8)
    ]


def observe_eight_rgbd(sim):
    """Capture eight simultaneous 45-degree RGB-D compass sectors."""
    observations = sim.get_sensor_observations()
    rgbs = [observations["rgb"][..., :3]] + [
        observations[f"completion_rgb_{index}"][..., :3]
        for index in range(1, 8)
    ]
    depths = [observations["depth"]] + [
        observations[f"completion_depth_{index}"]
        for index in range(1, 8)
    ]
    return rgbs, depths


def targetable_ground_mask(mask):
    h, w = mask.shape
    valid = mask.copy().astype(np.uint8)
    valid[: int(0.42 * h)] = 0
    valid[int(0.86 * h) :] = 0
    valid[:, : int(0.10 * w)] = 0
    valid[:, int(0.90 * w) :] = 0
    if not valid.any():
        valid = mask.astype(np.uint8)
    return valid.astype(bool)


def rgb_lower_floor_prior(shape):
    """Conservative RGB-only floor prior used when Grounded-SAM is empty.

    This is deliberately a weak candidate mask, not a semantic segmentation:
    it exposes only the lower, interior image band so the clean-RGB VLM can
    compare a portal/corridor that the floor detector missed.  The prior is
    never used for depth, navmesh, or arrival decisions and is marked in the
    candidate audit for review.
    """
    h, w = shape
    prior = np.zeros((h, w), bool)
    y0, y1 = int(round(0.54 * h)), int(round(0.84 * h))
    x0, x1 = int(round(0.08 * w)), int(round(0.92 * w))
    if y1 > y0 and x1 > x0:
        prior[y0:y1, x0:x1] = True
    return prior


def select_ground_point(mask, preferred_y=0.68):
    """Pick a well-inside ground pixel, avoiding near-camera and image edges."""
    h, w = mask.shape
    valid = targetable_ground_mask(mask).astype(np.uint8)
    if not valid.any():
        return None, 0.0
    distance = cv2.distanceTransform(valid, cv2.DIST_L2, 5)
    yy, xx = np.indices(mask.shape)
    center_prior = np.exp(-((xx - w / 2) / (0.32 * w)) ** 2)
    depth_prior = np.exp(-((yy - preferred_y * h) / (0.22 * h)) ** 2)
    score = distance * (0.35 + 0.65 * center_prior * depth_prior)
    y, x = np.unravel_index(np.argmax(score), score.shape)
    return np.array([float(x), float(y)], np.float32), float(score[y, x])


def pixel_ground_to_world(point, depth, agent_position, camera_yaw, hfov=90):
    """Project one RGB-D pixel to world XYZ with the Habitat camera model."""
    h, w = depth.shape
    x, y = point
    ix = int(np.clip(round(float(x)), 0, w - 1))
    iy = int(np.clip(round(float(y)), 0, h - 1))
    z_depth = float(depth[iy, ix])
    if not math.isfinite(z_depth) or z_depth <= 0:
        return None, z_depth
    focal = w / (2 * math.tan(math.radians(hfov) / 2))
    local_x = (x - (w - 1) / 2) * z_depth / focal
    local_y = -((y - (h - 1) / 2) * z_depth / focal)
    local_z = -z_depth
    cosine, sine = math.cos(camera_yaw), math.sin(camera_yaw)
    world = np.asarray(agent_position, np.float32).copy()
    world[0] += cosine * local_x + sine * local_z
    world[1] += 1.25 + local_y
    world[2] += -sine * local_x + cosine * local_z
    return world, z_depth


def _ground_anchor_options(mask, preferred=None, count=16):
    """Return deterministic interior anchors for post-VLM reachability checks.

    The VLM still receives only the numbered anchors produced by its harness.
    This helper is deliberately called *after* the VLM freezes a pixel, and is
    used only to find a nearby legal pixel when the frozen RGB ground ray does
    not land on the current navigable island.
    """
    valid = np.asarray(mask, bool)
    ys, xs = np.nonzero(valid)
    if not len(xs):
        return []
    distance = cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 5)
    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    if preferred is not None:
        first = int(np.argmin(((coords - np.asarray(preferred)) ** 2).sum(1)))
    else:
        first = int(np.argmax(distance[ys, xs]))
    chosen = [coords[first]]
    while len(chosen) < min(int(count), len(coords)):
        separation = np.min(np.stack([
            ((coords - point) ** 2).sum(1) for point in chosen]), axis=0)
        interior = distance[ys, xs] ** 2
        index = int(np.argmax(separation * (0.25 + interior)))
        if any(np.array_equal(coords[index], point) for point in chosen):
            break
        chosen.append(coords[index])
    return chosen


def _route_corridor_anchor_options(
        mask, candidate_yaw_rad, absolute_route_yaw_rad,
        route_half_width_rad, vertical_samples=16):
    """Sample the centre and sides of a frozen RGB route corridor.

    Farthest-point mask sampling can miss a thin furniture-bypass lane at the
    angular edge of an otherwise broad floor proposal.  These anchors are
    derived only from the already-selected camera yaw, its binary ground mask,
    and the permitted pixel-ray cone.  They do not use depth, navmesh, the R2R
    demonstration, or goal geometry; those remain downstream safety checks.
    """
    valid = np.asarray(mask, bool)
    ys, xs = np.nonzero(valid)
    if (not len(xs) or absolute_route_yaw_rad is None or
            route_half_width_rad is None):
        return []
    height, width = valid.shape
    center_x = (float(width) - 1.0) / 2.0
    half_image_width = max(float(width) / 2.0, 1.0)
    # Stay just inside both hard angular boundaries to avoid floating-point
    # rejection after integer pixel rounding.
    offsets = (0.0, -0.98 * float(route_half_width_rad),
               0.98 * float(route_half_width_rad))
    y_targets = np.quantile(
        ys.astype(np.float32),
        np.linspace(0.0, 1.0, max(2, int(vertical_samples))))
    anchors = []
    for offset in offsets:
        desired_ray = float(absolute_route_yaw_rad) + offset
        relative_bearing = ((float(candidate_yaw_rad) - desired_ray +
                             math.pi) % (2.0 * math.pi) - math.pi)
        # Horizontal pinhole inverse for the same 90-degree HFOV convention
        # used by pixel_ray_within_absolute_corridor.
        desired_x = center_x + half_image_width * math.tan(relative_bearing)
        desired_x = float(np.clip(desired_x, 0.0, width - 1.0))
        for desired_y in y_targets:
            score = (np.abs(xs.astype(np.float32) - desired_x) +
                     0.35 * np.abs(ys.astype(np.float32) - desired_y))
            index = int(np.argmin(score))
            anchor = np.array([float(xs[index]), float(ys[index])],
                              np.float32)
            if not any(np.array_equal(anchor, item) for item in anchors):
                anchors.append(anchor)
    return anchors


def extend_lower_ground_mask(mask, pixels=0, lower_fraction=0.65):
    """Extend only the image-bottom boundary of a detected floor patch."""
    original = np.asarray(mask, bool)
    pixels = max(0, int(pixels))
    if pixels == 0 or not original.any():
        return original.copy()
    kernel_size = 2 * pixels + 1
    dilated = cv2.dilate(
        original.astype(np.uint8),
        np.ones((kernel_size, kernel_size), np.uint8), iterations=1).astype(bool)
    extended = original.copy()
    lower_start = max(0, min(original.shape[0], int(
        round(float(lower_fraction) * original.shape[0]))))
    extended[lower_start:] = dilated[lower_start:]
    return extended


def pixel_ray_within_absolute_corridor(
        candidate_yaw_rad, pixel_x, image_width, absolute_yaw_rad,
        half_width_rad):
    """Test the actual pixel ray, rather than only its camera view center."""
    if absolute_yaw_rad is None or half_width_rad is None:
        return True
    center_x = (float(image_width) - 1.0) / 2.0
    half_image_width = max(float(image_width) / 2.0, 1.0)
    pixel_bearing = math.atan(
        (float(pixel_x) - center_x) / half_image_width)
    absolute_ray = float(candidate_yaw_rad) - pixel_bearing
    error = abs((absolute_ray - float(absolute_yaw_rad) + math.pi) %
                (2.0 * math.pi) - math.pi)
    return bool(error <= float(half_width_rad) + 1e-9)


def local_ray_navmesh_waypoint(
        sim, agent_position, ray_yaw_rad, minimum_distance_m=0.75,
        maximum_distance_m=3.0, step_m=0.20):
    """Return the farthest directly reachable point before a ray obstruction.

    This is an executor safety construction for a VLM-frozen RGB ray.  It
    samples only the current navmesh along that same ray and rejects snaps or
    shortest paths that bend away from it.  It never consults an R2R path,
    goal position, scene id, or semantic label.
    """
    start = np.asarray(agent_position, np.float32)
    direction = np.array([
        -math.sin(float(ray_yaw_rad)), 0.0,
        -math.cos(float(ray_yaw_rad))], np.float32)
    best = None
    distances = np.arange(
        max(float(step_m), float(minimum_distance_m)),
        float(maximum_distance_m) + 0.5 * float(step_m), float(step_m))
    for requested_distance in distances:
        proposed = start + float(requested_distance) * direction
        snapped = np.asarray(sim.pathfinder.snap_point(proposed), np.float32)
        if (not np.isfinite(snapped).all() or
                abs(float(snapped[1] - start[1])) > 0.30 or
                float(np.linalg.norm(
                    (snapped - proposed)[[0, 2]])) > 0.20):
            continue
        path = habitat_sim.ShortestPath()
        path.requested_start = start
        path.requested_end = snapped
        if not sim.pathfinder.find_path(path):
            continue
        geodesic = float(path.geodesic_distance)
        if not (float(minimum_distance_m) <= geodesic <=
                float(requested_distance) + 0.30):
            continue
        first_tangent = None
        for path_point in list(getattr(path, "points", []) or [])[1:]:
            delta = np.asarray(path_point, np.float32) - start
            if float(np.linalg.norm(delta[[0, 2]])) > 0.10:
                first_tangent = math.atan2(-float(delta[0]),
                                           -float(delta[2]))
                break
        if (first_tangent is not None and
                angle_distance(first_tangent, ray_yaw_rad) >
                math.radians(30.0)):
            continue
        best = {
            "position_xyz": snapped,
            "geodesic_m": geodesic,
            "requested_distance_m": float(requested_distance),
            "path_initial_tangent_yaw_rad": (
                float(first_tangent) if first_tangent is not None else None),
        }
    return best


def repair_selected_ground_point(
        sim, candidates, chosen_index, point, agent_position, selection=None,
        max_view_delta_rad=math.radians(45.0), max_geodesic_m=4.0,
        minimum_geodesic_m=0.0, anchor_sample_count=16,
        lower_mask_boundary_extension_px=0,
        absolute_route_yaw_rad=None, route_half_width_rad=None,
        max_path_ray_tangent_error_rad=math.radians(75.0)):
    """Validate/repair a frozen RGB ground point against the navmesh.

    This is a post-selection safety boundary: no depth, navmesh or path data
    is passed to the VLM.  If the selected pixel projects to a disconnected
    surface (a recurrent doorway/raised-surface failure), search only the
    selected view and its legal <=45-degree neighboring views for the nearest
    interior ground anchor with a valid shortest path.  Semantic direction is
    therefore preserved while impossible pixels are not sent to the executor.
    """
    selection = dict(selection or {})
    chosen_index = int(chosen_index)
    original_index = chosen_index
    original_point = np.asarray(point, np.float32)
    allowed = selection.get("allowed_views")
    if not isinstance(allowed, list) or not allowed:
        allowed = [index for index, candidate in enumerate(candidates)
                   if candidate.get("point") is not None and
                   not candidate.get("excluded", False)]
    allowed = [int(index) for index in allowed
               if 0 <= int(index) < len(candidates)]
    # A refined RGB view may be the only view named in the final VLM answer,
    # while the same answer also records the original/gated legal sectors.
    # If its pixel is disconnected, retain those already-audited sectors as
    # bounded geometric fallbacks.  This does not add a new semantic decision:
    # the VLM has already declared these views legal, and the post-selection
    # check only chooses a reachable floor anchor within the 45-degree
    # direction-preservation budget.
    audited_view_lists = [
        selection.get("initial_allowed_views"),
        (selection.get("view_refinement", {}) or {}).get(
            "initial_allowed_views"),
        (selection.get("relation_gate", {}) or {}).get("allowed_after_gate"),
        (selection.get("relation_direction_gate", {}) or {}).get(
            "allowed_after_gate"),
        (selection.get("direction_gate", {}) or {}).get("allowed_after_gate"),
    ]
    for view_list in audited_view_lists:
        if isinstance(view_list, list):
            allowed.extend(
                int(index) for index in view_list
                if 0 <= int(index) < len(candidates))
    allowed = list(dict.fromkeys(allowed))
    # A local RGB refinement is advisory: if its newly sampled floor ray is
    # disconnected, retain the original confirmed sector as a post-selection
    # fallback.  This is especially important for doorway views where a
    # 20–30° center shift can expose a wall while the neighboring sector has a
    # valid connected floor.  The fallback is still limited by the same
    # max_view_delta_rad direction-preservation bound below.
    refinement = selection.get("view_refinement", {}) or {}
    refinement_base = refinement.get("refinement_base_view_index")
    if refinement_base is not None:
        try:
            refinement_base = int(refinement_base)
        except (TypeError, ValueError):
            refinement_base = None
        if (refinement_base is not None and
                0 <= refinement_base < len(candidates) and
                refinement_base not in allowed):
            allowed.append(refinement_base)
    if chosen_index not in allowed:
        allowed.insert(0, chosen_index)
    selected_candidate = candidates[chosen_index]
    selected_rel = float(selected_candidate.get("relative_yaw_rad", 0.0))
    width = int(np.asarray(selected_candidate["target_mask"]).shape[1])
    height = int(np.asarray(selected_candidate["target_mask"]).shape[0])

    def evaluate(index, anchor):
        candidate = candidates[index]
        if not pixel_ray_within_absolute_corridor(
                candidate.get("yaw", 0.0), float(anchor[0]),
                np.asarray(candidate["target_mask"]).shape[1],
                absolute_route_yaw_rad, route_half_width_rad):
            return None
        depth = candidate.get("depth")
        if depth is None:
            return None
        world, selected_depth = pixel_ground_to_world(
            anchor, np.asarray(depth), agent_position,
            float(candidate.get("yaw", 0.0)))
        if world is None:
            return None
        snapped = np.asarray(sim.pathfinder.snap_point(world), np.float32)
        if not np.isfinite(snapped).all():
            return None
        path = habitat_sim.ShortestPath()
        path.requested_start = np.asarray(agent_position, np.float32)
        path.requested_end = snapped
        if not sim.pathfinder.find_path(path):
            return None
        path_initial_tangent_yaw_rad = None
        for path_point in list(getattr(path, "points", []) or [])[1:]:
            path_delta = (np.asarray(path_point, np.float32) -
                          np.asarray(agent_position, np.float32))
            if float(np.linalg.norm(path_delta[[0, 2]])) > 0.10:
                path_initial_tangent_yaw_rad = math.atan2(
                    -float(path_delta[0]), -float(path_delta[2]))
                break
        center_x = (float(np.asarray(candidate["target_mask"]).shape[1]) -
                    1.0) / 2.0
        half_image_width = max(
            float(np.asarray(candidate["target_mask"]).shape[1]) / 2.0, 1.0)
        pixel_bearing = math.atan(
            (float(anchor[0]) - center_x) / half_image_width)
        pixel_ray_yaw_rad = wrap_angle(
            float(candidate.get("yaw", 0.0)) - pixel_bearing)
        path_ray_tangent_error_rad = (
            angle_distance(path_initial_tangent_yaw_rad, pixel_ray_yaw_rad)
            if path_initial_tangent_yaw_rad is not None else 0.0)
        path_ray_consistent = bool(
            path_ray_tangent_error_rad <=
            float(max_path_ray_tangent_error_rad) + 1e-9)
        delta = angle_distance(
            float(candidate.get("relative_yaw_rad", 0.0)), selected_rel)
        pixel_distance = float(np.linalg.norm(
            np.asarray(anchor, np.float32) - original_point)) / max(
                float(max(width, height)), 1.0)
        center_penalty = abs(float(anchor[0]) - (width - 1) / 2.0) / max(
            float(width), 1.0)
        # Preserve the VLM's direction and pixel whenever possible.  A small
        # path-length term avoids selecting a very distant floor patch when a
        # nearby legal anchor exists, without using it as semantic evidence.
        score = (3.0 * delta / max(float(max_view_delta_rad), 1e-6) +
                 0.8 * pixel_distance + 0.1 * center_penalty +
                 0.02 * min(float(path.geodesic_distance), 10.0))
        return {
            "index": int(index), "point": np.asarray(anchor, np.float32),
            "world": world, "snapped": snapped,
            "selected_depth_m": float(selected_depth),
            "geodesic_m": float(path.geodesic_distance),
            "view_delta_deg": math.degrees(delta), "score": float(score),
            "pixel_ray_yaw_rad": float(pixel_ray_yaw_rad),
            "path_initial_tangent_yaw_rad": (
                float(path_initial_tangent_yaw_rad)
                if path_initial_tangent_yaw_rad is not None else None),
            "path_ray_tangent_error_deg": math.degrees(
                path_ray_tangent_error_rad),
            "path_ray_consistent": path_ray_consistent,
        }

    direct = evaluate(chosen_index, original_point)
    if (direct is not None and direct["path_ray_consistent"] and
            float(minimum_geodesic_m) <= float(direct["geodesic_m"]) <=
            float(max_geodesic_m)):
        return chosen_index, original_point, {
            "attempted": True, "status": "selected_point_reachable",
            "original_view_index": original_index,
            "original_point_xy": original_point.tolist(),
            "final_view_index": chosen_index,
            "final_point_xy": original_point.tolist(),
            "view_changed": False, "anchor_changed": False,
            "selected_geodesic_m": direct["geodesic_m"],
        }

    alternatives = []
    repair_search_audit = []
    for index in allowed:
        candidate = candidates[index]
        delta = angle_distance(
            float(candidate.get("relative_yaw_rad", 0.0)), selected_rel)
        if delta > max_view_delta_rad + 1e-6:
            continue
        anchors = []
        if index == chosen_index:
            anchors.append(original_point)
        candidate_point = candidate.get("point")
        if candidate_point is not None:
            anchors.append(np.asarray(candidate_point, np.float32))
        repair_mask = extend_lower_ground_mask(
            candidate.get("target_mask"),
            pixels=lower_mask_boundary_extension_px)
        anchors.extend(_ground_anchor_options(
            repair_mask,
            preferred=(original_point if index == chosen_index else None),
            count=max(
                int(anchor_sample_count),
                64 if float(minimum_geodesic_m) > 0.0 else 16)))
        anchors.extend(_route_corridor_anchor_options(
            repair_mask, float(candidate.get("yaw", 0.0)),
            absolute_route_yaw_rad, route_half_width_rad,
            vertical_samples=16))
        if float(minimum_geodesic_m) > 0.0:
            # The ordinary interior sampler is intentionally near the VLM
            # pixel.  For a required-progress continuation also expose a few
            # deterministic upper support-band anchors, which usually project
            # farther along the same RGB floor ray.  This is a post-selection
            # reachability check only; it does not feed geometry or distance
            # back into the VLM semantic decision.
            mask = np.asarray(candidate.get("target_mask"), bool)
            ys, xs = np.nonzero(mask)
            if len(xs):
                center_x = (width - 1) * 0.5
                for quantile in (0.20, 0.32, 0.44, 0.56, 0.68):
                    target_y = float(np.quantile(ys, quantile))
                    order = np.argsort(
                        np.abs(xs.astype(np.float32) - center_x) +
                        0.15 * np.abs(ys.astype(np.float32) - target_y))
                    if len(order):
                        anchors.append(np.array(
                            [float(xs[order[0]]), float(ys[order[0]])],
                            np.float32))
        seen = set()
        evaluated_records = []
        for anchor in anchors:
            key = tuple(np.round(np.asarray(anchor), 3).tolist())
            if key in seen:
                continue
            seen.add(key)
            evaluated = evaluate(index, anchor)
            if evaluated is not None:
                evaluated_records.append(evaluated)
            if (evaluated is not None and evaluated["path_ray_consistent"] and
                    float(minimum_geodesic_m) <=
                    float(evaluated["geodesic_m"]) <=
                    float(max_geodesic_m)):
                alternatives.append(evaluated)
        repair_search_audit.append({
            "view_index": int(index),
            "candidate_anchor_count": int(len(seen)),
            "reachable_anchor_count": int(len(evaluated_records)),
            "minimum_reachable_geodesic_m": (
                float(min(item["geodesic_m"]
                          for item in evaluated_records))
                if evaluated_records else None),
            "maximum_reachable_geodesic_m": (
                float(max(item["geodesic_m"]
                          for item in evaluated_records))
                if evaluated_records else None),
            "in_range_anchor_count": int(sum(
                float(minimum_geodesic_m) <= item["geodesic_m"] <=
                float(max_geodesic_m)
                for item in evaluated_records)),
            "path_ray_consistent_anchor_count": int(sum(
                bool(item.get("path_ray_consistent"))
                for item in evaluated_records)),
            "in_range_path_ray_consistent_anchor_count": int(sum(
                float(minimum_geodesic_m) <= item["geodesic_m"] <=
                float(max_geodesic_m) and
                bool(item.get("path_ray_consistent"))
                for item in evaluated_records)),
        })
    if not alternatives:
        # A far ground patch can be visible through an opening while its
        # projected endpoint lies behind a non-traversable wall or narrow
        # portal.  Do not execute the long through-wall endpoint and do not
        # replace the VLM's semantic direction.  Instead, build one local
        # point edge on the same RGB ray: track the lowest segmented pixels
        # and use the last directly reachable navmesh point before the
        # obstruction as the physical endpoint.  The ordinary dense
        # half-loss arrival rule remains mandatory in the executor.
        # Path/ray consistency is independent of the executor distance cap.
        # A visible patch can project to endpoints that are all farther than
        # the current form budget *and* whose shortest paths start around a
        # wall in another direction.  Calling that merely "no anchor" skips
        # the runner's one bounded RGB re-review and terminates at a perfectly
        # valid node.  Whenever a searched view has reachable projections but
        # none can be approached along the frozen image ray, classify the
        # semantic ray itself as physically inconsistent.
        path_ray_inconsistent = any(
            item["reachable_anchor_count"] > 0 and
            item["path_ray_consistent_anchor_count"] == 0
            for item in repair_search_audit)
        same_ray_beyond_cap = bool(float(max_geodesic_m) <= 3.0 and any(
            item["path_ray_consistent_anchor_count"] > 0 and
            item["minimum_reachable_geodesic_m"] is not None and
            float(item["minimum_reachable_geodesic_m"]) >
            float(max_geodesic_m)
            for item in repair_search_audit))
        local_boundary = None
        boundary_point = None
        if path_ray_inconsistent or same_ray_beyond_cap:
            boundary_mask = np.asarray(
                candidates[chosen_index].get("target_mask"), bool)
            ys, xs = np.nonzero(boundary_mask)
            legal = [
                index for index, x in enumerate(xs)
                if pixel_ray_within_absolute_corridor(
                    candidates[chosen_index].get("yaw", 0.0), float(x),
                    boundary_mask.shape[1], absolute_route_yaw_rad,
                    route_half_width_rad)]
            if legal:
                legal = np.asarray(legal, np.int64)
                # Prefer the visible bottom boundary and then the VLM's
                # original horizontal ray. This keeps the stop cluster dense
                # and makes its disappearance physically interpretable.
                order = np.lexsort((
                    np.abs(xs[legal].astype(np.float32) - original_point[0]),
                    -ys[legal].astype(np.float32)))
                picked = int(legal[int(order[0])])
                boundary_point = np.array(
                    [float(xs[picked]), float(ys[picked])], np.float32)
                boundary_center_x = (boundary_mask.shape[1] - 1.0) / 2.0
                boundary_bearing = math.atan(
                    (float(boundary_point[0]) - boundary_center_x) /
                    max(boundary_mask.shape[1] / 2.0, 1.0))
                boundary_ray = wrap_angle(
                    float(candidates[chosen_index].get("yaw", 0.0)) -
                    boundary_bearing)
                local_boundary = local_ray_navmesh_waypoint(
                    sim, agent_position, boundary_ray,
                    minimum_distance_m=float(minimum_geodesic_m),
                    maximum_distance_m=min(float(max_geodesic_m), 3.0))
        if local_boundary is not None and boundary_point is not None:
            local_status = (
                "repaired_to_local_occlusion_boundary"
                if path_ray_inconsistent else
                "repaired_to_local_geodesic_cap_boundary")
            return chosen_index, boundary_point, {
                "attempted": True,
                "status": local_status,
                "original_view_index": original_index,
                "original_point_xy": original_point.tolist(),
                "final_view_index": chosen_index,
                "final_point_xy": boundary_point.tolist(),
                "view_changed": False,
                "anchor_changed": bool(np.linalg.norm(
                    boundary_point - original_point) > 1e-3),
                "selected_geodesic_m": float(local_boundary["geodesic_m"]),
                "physical_waypoint_override_xyz": np.asarray(
                    local_boundary["position_xyz"]).tolist(),
                "physical_waypoint_policy": (
                    "last directly reachable navmesh point on the frozen "
                    "RGB ray before an incompatible through-wall endpoint"
                    if path_ray_inconsistent else
                    "local point on the frozen RGB ray at the instruction-"
                    "form geodesic cap"),
                "repair_search_audit": repair_search_audit,
                "max_path_ray_tangent_error_deg": math.degrees(
                    float(max_path_ray_tangent_error_rad)),
            }
        return chosen_index, original_point, {
            "attempted": True,
            "status": (
                "too_near_to_progress"
                if (direct is not None and
                    float(direct["geodesic_m"]) <
                    float(minimum_geodesic_m)) else
                "path_ray_inconsistent"
                if path_ray_inconsistent else
                "no_reachable_ground_anchor"),
            "original_view_index": original_index,
            "original_point_xy": original_point.tolist(),
            "final_view_index": chosen_index,
            "final_point_xy": original_point.tolist(),
            "view_changed": False, "anchor_changed": False,
            "max_geodesic_m": float(max_geodesic_m),
            "minimum_geodesic_m": float(minimum_geodesic_m),
            "allowed_views_searched": list(allowed),
            "absolute_route_yaw_rad": (
                float(absolute_route_yaw_rad)
                if absolute_route_yaw_rad is not None else None),
            "route_half_width_deg": (
                math.degrees(float(route_half_width_rad))
                if route_half_width_rad is not None else None),
            "max_path_ray_tangent_error_deg": math.degrees(
                float(max_path_ray_tangent_error_rad)),
            "repair_search_audit": repair_search_audit,
        }
    best = min(alternatives, key=lambda item: item["score"])
    repaired_point = best["point"]
    return int(best["index"]), repaired_point, {
        "attempted": True, "status": "repaired_to_reachable_ground_anchor",
        "original_view_index": original_index,
        "original_point_xy": original_point.tolist(),
        "final_view_index": int(best["index"]),
        "final_point_xy": repaired_point.tolist(),
        "view_changed": bool(best["index"] != original_index),
        "anchor_changed": bool(np.linalg.norm(repaired_point - original_point) > 1e-3),
        "view_delta_deg": float(best["view_delta_deg"]),
        "selected_geodesic_m": float(best["geodesic_m"]),
        "max_geodesic_m": float(max_geodesic_m),
        "minimum_geodesic_m": float(minimum_geodesic_m),
        "lower_mask_boundary_extension_px": int(
            lower_mask_boundary_extension_px),
        "repair_policy": (
            "nearest legal <=45-degree reachable RGB ground ray inside any "
            "committed absolute route corridor and within "
            "bounded online geodesic and above the required progress "
            "threshold"),
    }


def exploration_ground_points(mask, depth, preferred_y=0.68, count=6):
    """Return spatially diverse deep/interior ground frontier candidates."""
    valid = targetable_ground_mask(mask) & np.isfinite(depth) & (depth > 0.25)
    if not valid.any():
        return []
    distance = cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 5)
    h, w = mask.shape
    yy, xx = np.indices(mask.shape)
    center = np.exp(-((xx - w / 2) / (0.38 * w)) ** 2)
    vertical = np.exp(-((yy - preferred_y * h) / (0.25 * h)) ** 2)
    capped_depth = np.minimum(depth, np.nanpercentile(depth[valid], 90))
    score = distance * (0.25 + 0.45 * center * vertical) * (0.5 + capped_depth)
    score[~valid] = 0
    candidates = []
    work = score.copy()
    suppression_radius = max(18, int(min(h, w) * 0.14))
    for _ in range(count):
        y, x = np.unravel_index(np.argmax(work), work.shape)
        value = float(work[y, x])
        if value <= 0:
            break
        candidates.append((np.array([float(x), float(y)], np.float32), value))
        cv2.circle(work, (int(x), int(y)), suppression_radius, 0.0, -1)
    return candidates


def exploration_ground_point(mask, depth, preferred_y=0.68):
    candidates = exploration_ground_points(mask, depth, preferred_y, count=1)
    return candidates[0] if candidates else (None, 0.0)


def distance_to_position_history(position, history):
    if not history:
        return math.inf
    points = np.asarray(history, np.float32)
    delta = points[:, [0, 2]] - np.asarray(position, np.float32)[None, [0, 2]]
    return float(np.linalg.norm(delta, axis=1).min())


def frontier_key(position, quantization=0.25):
    """Stable XZ key for auditing repeated attempts at the same frontier."""
    position = np.asarray(position, np.float32)
    return tuple(int(round(float(position[axis]) / quantization))
                 for axis in (0, 2))



def path_novelty_statistics(path_points, history, radius, spacing=0.25):
    """Measure how much of a navmesh path stays away from recorded trajectory."""
    points = np.asarray(path_points, np.float32)
    samples = []
    for start, end in zip(points[:-1], points[1:]):
        length = float(np.linalg.norm((end - start)[[0, 2]]))
        count = max(1, int(math.ceil(length / spacing)))
        samples.extend(start + (end - start) * fraction
                       for fraction in np.linspace(0, 1, count, endpoint=False))
    samples.append(points[-1])
    distances = [distance_to_position_history(point, history) for point in samples]
    novel = [distance >= radius for distance in distances]
    return {
        "sample_count": len(samples),
        "novel_fraction": sum(novel) / max(len(novel), 1),
        "overlap_fraction": 1 - sum(novel) / max(len(novel), 1),
        "minimum_history_distance_m": min(distances) if distances else 0.0,
    }



def render_scan_frame(rgb, target_idx, offset_deg, phase="turn"):
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    label = phase.replace("_", " ")
    cv2.putText(frame, f"stage {target_idx + 1} {label} {offset_deg:+.0f} deg",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    return frame


def continuous_turn(sim, position, start_yaw, end_yaw, max_step, rendered,
                    motion_log, target_idx, phase, video_composer,
                    position_history, instruction, reference_path=None,
                    reference_path_index=0,
                    policy_input_contract="legacy_rgbd_geometry"):
    delta = wrap_angle(end_yaw - start_yaw)
    steps = max(1, int(math.ceil(abs(delta) / max_step)))
    yaw = start_yaw
    if policy_input_contract == "rgb_only_v1":
        from rgb_only_runtime import require_rgb_only_policy_sim
        policy_sim = require_rgb_only_policy_sim(sim)
        action = "turn_left" if delta > 0.0 else "turn_right"
        if abs(delta) < 1e-6:
            return start_yaw, observe(policy_sim)
        # Habitat's discrete turn amount is configured to max_step by the
        # evaluation adapter.  Navigation knows only the command it issued;
        # it never reads the resulting simulator rotation.
        steps = max(1, int(round(abs(delta) / max_step)))
        for step in range(1, steps + 1):
            observations = policy_sim.step(action)
            yaw = wrap_angle(
                start_yaw + math.copysign(max_step * step, delta))
            rgb = observations["rgb"][..., :3]
            obs_frame = render_scan_frame(
                rgb, target_idx,
                math.degrees(wrap_angle(yaw - start_yaw)), phase)
            if video_composer is not None:
                if hasattr(video_composer, "emit"):
                    video_composer.emit(
                        obs_frame, instruction, target_idx, phase)
                else:
                    rendered.append(video_composer.compose(
                        obs_frame, None, None, yaw, instruction,
                        target_idx, phase))
            motion_log.append({
                "phase": phase, "target_index": target_idx,
                "turn_frame": step, "action": action,
                "commanded_turn_deg": math.degrees(
                    math.copysign(max_step, delta)),
                "policy_input_contract": "rgb_only_v1",
            })
        return yaw, rgb
    if policy_input_contract != "legacy_rgbd_geometry":
        raise ValueError(
            f"unknown policy_input_contract {policy_input_contract!r}")
    for step in range(1, steps + 1):
        yaw = wrap_angle(start_yaw + delta * step / steps)
        set_pose(sim, position, yaw)
        rgb = observe(sim)
        if reference_path:
            projection = project_reference_path(
                reference_path,
                range(int(reference_path_index), len(reference_path)),
                np.asarray(position, np.float32) +
                np.array([0.0, 1.25, 0.0], np.float32),
                yaw, rgb.shape[1], rgb.shape[0])
            rgb = draw_reference_path_overlay(
                rgb, projection, next_path_index=int(reference_path_index) + 1)
        obs_frame = render_scan_frame(
            rgb, target_idx, math.degrees(wrap_angle(yaw - start_yaw)), phase)
        rendered.append(video_composer.compose(
            obs_frame, position_history, position, yaw, instruction,
            target_idx, phase))
        motion_log.append({
            "phase": phase, "target_index": target_idx, "turn_frame": step,
            "position_xyz": position.tolist(), "yaw_rad": yaw,
        })
    return yaw, rgb


def append_decision_visualization(rendered, candidates, chosen_index, stage,
                                  prior_action_history, video_composer,
                                  position_history, position,
                                  reference_path=None,
                                  reference_path_index=0):
    action_count = len(prior_action_history)
    for index, candidate in enumerate(candidates):
        rgb = candidate["rgb"].copy()
        object_mask = candidate.get("small_seg_object_mask")
        if object_mask is not None:
            blue = np.zeros_like(rgb)
            blue[..., 2] = 255
            rgb[object_mask] = (0.82 * rgb[object_mask] +
                                0.18 * blue[object_mask]).astype(np.uint8)
        mask = candidate["target_mask"]
        green = np.zeros_like(rgb)
        green[..., 1] = 255
        rgb[mask] = (0.55 * rgb[mask] + 0.45 * green[mask]).astype(np.uint8)
        for detection in candidate.get("semantic_detections", []):
            det_mask = detection.mask
            magenta = np.zeros_like(rgb)
            magenta[..., 0] = 255
            magenta[..., 2] = 255
            rgb[det_mask] = (0.72 * rgb[det_mask] + 0.28 * magenta[det_mask]).astype(np.uint8)
        # Diagnostic-only overlay: the hidden R2R reference path is projected
        # after the VLM has frozen its view/pixel.  It is never part of the
        # model-facing candidate dictionary or prompt.
        if reference_path:
            camera_position = (np.asarray(position, np.float32) +
                               np.array([0.0, 1.25, 0.0], np.float32))
            projection = project_reference_path(
                reference_path,
                range(int(reference_path_index), len(reference_path)),
                camera_position, float(candidate["yaw"]),
                rgb.shape[1], rgb.shape[0])
            rgb = draw_reference_path_overlay(
                rgb, projection,
                selected_point=(candidate.get("point")
                                if index == chosen_index else None),
                selected=bool(index == chosen_index and
                              candidate.get("point") is not None),
                next_path_index=int(reference_path_index) + 1)
            # Keep the audit payload attached to each candidate so runtime
            # videos and JSON records share exactly the same projection.
            candidate["reference_path_projection"] = projection
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for detection in candidate.get("semantic_detections", []):
            x0, y0, x1, y1 = np.round(detection.box_xyxy).astype(int)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 255), 2)
            cv2.putText(frame, f"DET {detection.label} {detection.score:.2f}",
                        (max(3, x0), max(92, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (255, 0, 255), 1)
        color = (0, 0, 255) if candidate["excluded"] else (255, 255, 255)
        status = "BACKTRACK BLOCKED" if candidate["excluded"] else "ALLOWED"
        cv2.putText(frame, f"VLM VIEW {index} {status}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1)
        cv2.putText(frame, f"stage: {stage['navigation_instruction'][:38]}", (6, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        cv2.putText(frame, f"target: {stage['semantic_spatial_target'][:42]}", (6, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        cv2.putText(frame, f"previous actions: {action_count}", (6, 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        if index == chosen_index:
            point = tuple(np.round(candidates[index]["point"]).astype(int))
            cv2.drawMarker(frame, point, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
            requested = candidates[index].get("vlm_selection", {}).get("requested_xy")
            if requested is not None:
                requested_point = tuple(np.round(requested).astype(int))
                cv2.circle(frame, requested_point, 5, (255, 255, 0), 1)
            cv2.putText(frame, "VLM ON-GROUND", (6, 96),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        if hasattr(video_composer, "emit"):
            video_composer.emit(
                frame, stage["navigation_instruction"], stage["stage_id"],
                "six_view_decision", repeat=2)
        else:
            composed = video_composer.compose(
                frame, position_history, position, candidate["yaw"],
                stage["navigation_instruction"], stage["stage_id"],
                "six_view_decision")
            rendered.extend([composed, composed.copy()])


def choose_view(sim, position, yaw, back_yaw, segmenter, semantic_detector,
                vlm_harness, stage,
                prior_action_history, rendered, motion_log, target_idx,
                video_composer, position_history, views=6,
                backtrack_exclusion=math.radians(50),
                scan_step=math.radians(10), blocked_yaws=None,
                blocked_direction_exclusion=math.radians(50),
                reference_path=None, reference_path_index=0,
                minimum_initial_geodesic_m=0.0,
                maximum_initial_geodesic_m=None,
                semantic_reference_rgb=None,
                policy_input_contract="legacy_rgbd_geometry"):
    """Read a simultaneous Habitat panorama and optionally refine one yaw."""
    if views not in {6, 8}:
        raise ValueError("Direct Habitat panorama requires --views 6 or 8")
    candidates = []
    reference_yaw = yaw
    # Passing/straight clauses are forward-progress operations.  A generic
    # 50-degree reverse-sector guard is too permissive after a turn: a
    # 70--80-degree rear-side ray can still send the agent into the previous
    # room while looking superficially like a legal floor ray.  Widen this
    # guard only for forms whose semantics require continuing beyond a
    # landmark/along the current lane; turns and endpoint relations retain
    # their ordinary side/rear flexibility.  This is form-level and uses no
    # demo path, depth, episode identity, or fixed pixel.
    stage_form = str(stage.get("form", "")).upper()
    turn_carryover = (stage.get("metadata", {}) or {}).get(
        "turn_carryover", {}) or {}
    if (stage_form in {"PASS_LANDMARK", "ADVANCE_STRAIGHT",
                       "FOLLOW_PATH_BOUNDARY"} and
            not bool(turn_carryover.get("active"))):
        backtrack_exclusion = route_backtrack_exclusion(
            stage, backtrack_exclusion)
    elif bool(turn_carryover.get("active")):
        # The current clause follows a persisted turn edge.  Its legal
        # resulting corridor may be in the previous edge's rear hemisphere;
        # do not reapply the generic incoming-direction exclusion.  Blocked
        # directions remain hard-excluded below and all RGB ground/navmesh
        # gates still apply.
        backtrack_exclusion = 0.0
    rgb_only = policy_input_contract == "rgb_only_v1"
    if rgb_only:
        from rgb_only_runtime import require_rgb_only_policy_sim
        require_rgb_only_policy_sim(sim)
        if position is not None:
            raise ValueError("rgb_only_v1 point selection cannot receive pose")
        if reference_path is not None:
            raise ValueError(
                "rgb_only_v1 point selection cannot receive reference_path")
    elif policy_input_contract != "legacy_rgbd_geometry":
        raise ValueError(
            f"unknown policy_input_contract {policy_input_contract!r}")
    if views == 8 and rgb_only:
        offsets = np.radians(EIGHT_VIEW_YAW_OFFSETS_DEG)
        rgbs = observe_eight_rgb(sim)
        depths = [None] * len(rgbs)
    elif views == 8:
        offsets = np.radians(EIGHT_VIEW_YAW_OFFSETS_DEG)
        rgbs, depths = observe_eight_rgbd(sim)
    elif rgb_only:
        offsets = np.radians([0, 60, 120, 180, 240, 300])
        rgbs = observe_six_rgb(sim)
        depths = [None] * len(rgbs)
    else:
        offsets = np.radians([0, 60, 120, 180, 240, 300])
        rgbs, depths = observe_six_rgbd(sim)
    strict_ground = bool(getattr(segmenter, "strict_ground_mask", False))
    segmented_views = (segmenter.batch(rgbs)
                       if hasattr(segmenter, "batch") else
                       [segmenter(rgb) for rgb in rgbs])
    blocked_yaws = list(blocked_yaws or [])
    detection_queries = extract_detection_queries(stage)

    def build_candidate(index, offset, rgb, depth, refined=False,
                        segmentation=None):
        candidate_yaw = wrap_angle(reference_yaw + float(offset))
        # Semantic point selection is deliberately RGB-only.  Depth remains
        # attached to the candidate solely for post-selection projection.
        mask, ground_detections = (segmentation if segmentation is not None
                                   else segmenter(rgb))
        mask = np.asarray(mask, dtype=bool)
        # Grounded-SAM occasionally returns a high-area ``walkable ground``
        # mask that is actually a wall/door panel (near-uniform support from
        # the top to the bottom of the image).  Such a mask can make an
        # explicit TURN choose a point on the wall while still passing the
        # binary mask gate.  Apply a form-agnostic 2-D shape sanity check to
        # unusually broad detections: a valid floor proposal should have
        # stronger lower-band than upper-band support.  This uses no depth,
        # navmesh, reference path, or episode identity and records exclusions
        # for auditability.
        floor_mask_sanity = []
        if ground_detections and not strict_ground:
            h_ground, w_ground = mask.shape
            sanitized = []
            for detection in ground_detections:
                det_mask = np.asarray(getattr(detection, "mask", None), bool)
                if det_mask.shape != mask.shape:
                    continue
                area_fraction = float(det_mask.mean())
                top_support = float(det_mask[:max(1, int(0.20 * h_ground))].mean())
                lower_support = float(det_mask[int(0.80 * h_ground):].mean())
                broad_uniform = bool(
                    area_fraction >= 0.70 and
                    lower_support <= max(0.05, top_support * 1.15))
                record = {
                    "label": str(getattr(detection, "label", "")),
                    "area_fraction": area_fraction,
                    "top_support": top_support,
                    "lower_support": lower_support,
                    "excluded": broad_uniform,
                    "reason": ("broad_uniform_vertical_support"
                               if broad_uniform else "retained"),
                }
                floor_mask_sanity.append(record)
                if not broad_uniform:
                    sanitized.append(det_mask)
            if sanitized:
                mask = np.logical_or.reduce(sanitized).astype(bool)
            elif any(not item["excluded"] for item in floor_mask_sanity):
                mask = np.zeros_like(mask, dtype=bool)
            elif floor_mask_sanity:
                # All broad proposals were rejected; force a clean empty
                # mask so downstream RGB fallback/alternate-view logic can
                # request another semantic ray instead of targeting a wall.
                mask = np.zeros_like(mask, dtype=bool)
        backtrack_delta = (angle_distance(candidate_yaw, back_yaw)
                           if back_yaw is not None else math.pi)
        blocked_delta = (min(angle_distance(candidate_yaw, blocked)
                             for blocked in blocked_yaws)
                         if blocked_yaws else math.pi)
        hard_excluded = bool(
            blocked_yaws and blocked_delta < blocked_direction_exclusion)
        excluded = bool(
            hard_excluded or
            (back_yaw is not None and backtrack_delta < backtrack_exclusion))
        detections = (semantic_detector.detect(rgb, detection_queries)
                      if semantic_detector is not None and detection_queries else [])
        semantic_backtrack_override = False
        # A semantic endpoint or portal may legitimately lie in a rear-side
        # tangent after reaching the end of a corridor.  The exact incoming
        # ray remains excluded (25-degree core), but hard-removing both
        # adjacent 45-degree sectors erases valid L/U-shaped continuations and
        # leaves the VLM only unrelated forward objects.  Reopen a tangent
        # only when the current target has RGB detector evidence; explicit
        # recovery-blocked directions remain hard.  The VLM must still justify
        # identity/relation and the post-selection navmesh check is unchanged.
        semantic_backtrack_forms = {
            "STOP_WAIT", "EXIT_REGION", "ENTER_REGION", "SELECT_PORTAL",
            "TRAVERSE_PORTAL_REGION",
        }
        semantic_threshold = (0.20 if stage.get("form") == "STOP_WAIT"
                              else 0.28)
        outside_reverse_core = bool(
            back_yaw is not None and
            backtrack_delta >= math.radians(25.0))
        if (stage.get("form") in semantic_backtrack_forms and
                not hard_excluded and excluded and outside_reverse_core and
                detections and any(
                    float(getattr(item, "score", 0.0)) >= semantic_threshold
                    for item in detections)):
            excluded = False
            semantic_backtrack_override = True
        # Grounding-DINO proposals are useful evidence, but low-confidence
        # noun matches are especially prone to turning walls/door panels into
        # a named landmark.  For PASS_LANDMARK the landmark is an execution
        # constraint only when the RGB detector has a reasonably strong
        # proposal.  Weak proposals are retained in the audit/prompt as
        # fallible observations, while the relation mask is built as a
        # floor-only reacquisition target.  This is a form-level confidence
        # policy (no scene, episode, path, depth, or fixed-pixel condition).
        relation_detections = detections
        weak_relation_detections = []
        if stage.get("form") == "PASS_LANDMARK" and detections:
            relation_score_threshold = 0.40
            relation_detections = [
                item for item in detections
                if float(getattr(item, "score", 0.0)) >=
                relation_score_threshold]
            weak_relation_detections = [
                item.rgb_prompt_record() for item in detections
                if float(getattr(item, "score", 0.0)) <
                relation_score_threshold]
        # Convert instruction-related DINO+SAM detections into a conservative
        # forbidden-object mask.  Previously this mask was always empty, so a
        # spurious floor proposal overlapping a couch/door/wall could still be
        # presented as a valid anchor.  Route-bearing labels (hallway,
        # corridor, floor, opening) remain usable; furniture, portals and
        # structural panels are excluded with a small 2-D safety margin.
        object_mask = np.zeros_like(mask)
        route_labels = {
            "floor", "ground", "walkable", "carpet", "rug", "hallway",
            "corridor", "intersection", "opening", "landing",
        }
        for detection in detections:
            det_mask = np.asarray(getattr(detection, "mask", None), bool)
            if det_mask.shape != mask.shape:
                continue
            label_words = set(re.findall(r"[a-z]+",
                                         str(getattr(detection, "label", ""))
                                         .lower()))
            if label_words & route_labels and not label_words & {
                    "door", "doorway", "portal", "wall", "panel"}:
                continue
            object_mask |= det_mask
        base_target_mask = targetable_ground_mask(mask)
        ground_mask_source = (
            "dense_majority_2_of_3" if strict_ground else "grounded_sam")
        excluded_non_route_ground = []
        # Grounded-SAM's open-vocabulary floor union can include stair treads
        # in a non-vertical clause (the detector quite correctly calls them
        # ``stair``).  Treating those pixels as ordinary floor sends a
        # pass/straight instruction into a side staircase and corrupts the
        # subsequent action history.  Remove only stair/step/landing masks
        # when the instruction is not a vertical transition; VERTICAL_UP/DOWN
        # keeps them as the intended route.  This is a form-level RGB mask
        # rule, not an episode or fixed-pixel exception.
        if stage.get("form") not in {"VERTICAL_UP", "VERTICAL_DOWN"}:
            stair_mask = np.zeros_like(base_target_mask, dtype=bool)
            for detection in ground_detections:
                label = str(getattr(detection, "label", "")).lower()
                if re.search(r"\b(?:stair|stairs|staircase|step|landing)\b", label):
                    det_mask = getattr(detection, "mask", None)
                    if det_mask is not None:
                        stair_mask |= np.asarray(det_mask, dtype=bool)
                    excluded_non_route_ground.append(label)
            if stair_mask.any():
                base_target_mask &= ~stair_mask
                ground_mask_source = (
                    "dense_majority_without_non_route_stairs"
                    if strict_ground else
                    "grounded_sam_without_non_route_stairs")
        # Grounded-SAM can return no floor proposal in a doorway view even
        # when the connected corridor is clearly visible in RGB.  Keep a
        # conservative lower-band prior so that view is not hard-excluded;
        # the VLM must still reject furniture/walls from clean RGB and the
        # audit records distinguish this weak fallback from SAM evidence.
        h_mask, w_mask = base_target_mask.shape
        central_lower = base_target_mask[
            int(round(0.52 * h_mask)):int(round(0.86 * h_mask)),
            int(round(0.18 * w_mask)):int(round(0.82 * w_mask))]
        portal_forms_with_rgb_fallback = {
            "EXIT_REGION", "ENTER_REGION", "SELECT_PORTAL",
            "TRAVERSE_PORTAL_REGION",
        }
        # A bare TURN can legitimately face a corridor whose Grounded-SAM
        # floor mask is fragmented/empty (wall-heavy side views are common at
        # intersections).  Keep the explicit side gate, but expose a bounded
        # lower-band RGB prior for every TURN_LEFT/RIGHT, not only compound
        # clauses.  Clean RGB remains the semantic arbiter and post-selection
        # reachability remains the safety boundary; no depth/path/scene cue is
        # used to manufacture a target.
        compound_turn_fallback = bool(
            stage.get("form") in {"TURN_LEFT", "TURN_RIGHT"})
        pass_progress_fallback = bool(
            # PASS_LANDMARK is intrinsically a forward-progress relation.  A
            # missing/fragmented Grounded-SAM floor mask must not terminate
            # the hop before the landmark can be re-observed.  Use the same
            # bounded RGB lower-band prior for every PASS form (not only
            # compound forms); DINO+SAM object masks and the VLM still gate
            # the final anchor, and no depth/navmesh/demo signal is admitted.
            stage.get("form") == "PASS_LANDMARK")
        if (not strict_ground and
                (not base_target_mask.any() or not central_lower.any()) and
                (stage.get("form") in portal_forms_with_rgb_fallback or
                 compound_turn_fallback or pass_progress_fallback)):
            base_target_mask = rgb_lower_floor_prior(mask.shape)
            ground_mask_source = "rgb_lower_floor_prior"
        elif (not strict_ground and compound_turn_fallback and
              float(base_target_mask.mean()) < 0.12):
            # Side-turn views frequently contain a valid corridor whose
            # Grounded-SAM proposal is mostly swallowed by a stair/step
            # proposal.  A tiny residual mask cannot yield a physically
            # advancing ray even when clean RGB clearly shows the corridor.
            # Union a bounded lower-band RGB prior only for this generic
            # low-support TURN case; semantic VLM gating and post-selection
            # navmesh reachability still decide the final point.  This uses
            # no depth, reference path, scene identity, or fixed pixel.
            base_target_mask = np.logical_or(
                base_target_mask, rgb_lower_floor_prior(mask.shape))
            ground_mask_source = "grounded_sam_union_rgb_lower_floor_prior"
        target_mask, strategy_application = constrain_floor_candidates(
            stage, base_target_mask, None, relation_detections, object_mask)
        stage_metadata = stage.get("metadata", {}) or {}
        circumnavigation_partial = (stage_metadata.get(
            "supported_partial_route_continuation", {}) or {})
        if (strict_ground and stage.get("form") == "CIRCUMNAVIGATE" and
                circumnavigation_partial.get("active")):
            # At the far edge of an obstacle its mask can be clipped to a few
            # columns, making the relation corridor too thin to yield an
            # anchor even though the ensemble floor remains valid. Reopen the
            # same strict ground (minus detected objects) for the one bounded
            # continuation. The persisted route cone and RGB review still
            # decide direction; no new pixel or depth is exposed to the VLM.
            continuation_ground = np.asarray(base_target_mask, bool) & ~object_mask
            if continuation_ground.any():
                target_mask = np.logical_or(
                    np.asarray(target_mask, bool), continuation_ground)
                strategy_application = dict(strategy_application)
                strategy_application[
                    "circumnavigation_partial_ground_expansion"] = {
                        "enabled": True,
                        "source": "original_dense_majority_ground_mask",
                        "policy": (
                            "one route-constrained far-edge continuation "
                            "may use strict ground beyond a clipped object"),
                    }
        boundary_lookahead = (stage_metadata.get(
            "following_landmark_boundary_lookahead", {}) or {})
        terminal_portal_residual = (stage_metadata.get(
            "terminal_portal_residual_edge", {}) or {})
        explicit_linear_segment = (stage_metadata.get(
            "explicit_linear_segment_cap", {}) or {})
        if (boundary_lookahead.get("active") or
                terminal_portal_residual.get("active") or
                explicit_linear_segment.get("active")):
            # A bounded A->B->C handoff deliberately asks for a 0.35--1.25 m
            # clearance point.  The ordinary targetable-ground transform
            # removes the lowest image band to avoid generic near-camera
            # targets, which can leave no candidate in that physical range.
            # Restore only central lower pixels from the actual ensemble
            # ground mask, subtract detected objects, and keep all downstream
            # ray/history/navmesh checks.  No depth or reference path enters
            # this mask operation.
            h_near, w_near = mask.shape
            near_ground = np.asarray(mask, bool).copy()
            near_ground[:int(round(0.68 * h_near))] = False
            near_ground[int(round(0.96 * h_near)):] = False
            near_ground[:, :int(round(0.12 * w_near))] = False
            near_ground[:, int(round(0.88 * w_near)):] = False
            near_ground &= ~object_mask
            if near_ground.any():
                target_mask = np.logical_or(target_mask, near_ground)
                strategy_application = dict(strategy_application)
                strategy_application["boundary_near_ground_expansion"] = {
                    "enabled": True,
                    "image_y_fraction": [0.68, 0.96],
                    "image_x_fraction": [0.12, 0.88],
                    "source": "original_ensemble_ground_mask",
                    "reason": (
                        "terminal_portal_residual_edge"
                        if terminal_portal_residual.get("active") else
                        ("explicit_linear_segment_cap"
                         if explicit_linear_segment.get("active") else
                        "following_landmark_boundary_lookahead")),
                }
        if strict_ground and stage.get("form") == "PASS_LANDMARK":
            # Around a central table/sofa, every farther targetable pixel can
            # project onto the obstacle or outside the local navmesh even
            # though the lower edge contains the only connected bypass lane.
            # Restore only central lower pixels from the immutable 2/3 ground
            # ensemble.  Near-full-frame region detections (for example a
            # Grounding-DINO "bathroom" mask) are semantic evidence rather
            # than collision masks; ordinary localized object masks remain
            # subtracted. Reachability and the selected +/-30-degree pixel ray
            # are still enforced after the VLM freezes its semantic view.
            h_pass, w_pass = mask.shape
            pass_lower = np.asarray(mask, bool).copy()
            pass_lower[:int(round(0.72 * h_pass))] = False
            pass_lower[int(round(0.96 * h_pass)):] = False
            pass_lower[:, :int(round(0.10 * w_pass))] = False
            pass_lower[:, int(round(0.90 * w_pass)):] = False
            localized_object_mask = np.asarray(object_mask, bool)
            object_mask_ignored = bool(localized_object_mask.mean() > 0.60)
            if not object_mask_ignored:
                pass_lower &= ~localized_object_mask
            if pass_lower.any():
                target_mask = np.logical_or(target_mask, pass_lower)
                strategy_application = dict(strategy_application)
                strategy_application["pass_raw_lower_ground_expansion"] = {
                    "enabled": True,
                    "image_y_fraction": [0.72, 0.96],
                    "image_x_fraction": [0.10, 0.90],
                    "source": "dense_majority_2_of_3_ground_mask",
                    "near_full_region_object_mask_ignored": (
                        object_mask_ignored),
                    "policy": (
                        "actual ensemble-ground bypass lane; semantic view "
                        "and post-selection ray/navmesh gates remain active"),
                }
        strategy_application["raw_detection_count"] = len(detections)
        strategy_application["relation_detection_count"] = len(
            relation_detections)
        strategy_application["weak_relation_detection_count"] = len(
            weak_relation_detections)
        if weak_relation_detections:
            strategy_application["weak_relation_detections"] = (
                weak_relation_detections)
        if stage.get("form") == "STOP_WAIT":
            # A STOP/WAIT target is a near-landmark relation, not a portal
            # traversal.  Grounded-SAM can label a doorway/wall transition as
            # floor high in the image; using that far/top band makes the
            # executor drive through the opening before it can stop.  Keep a
            # lower support band when it has any valid pixels, so the selected
            # point stays on the camera side of the landmark.  This is a
            # generic RGB/2-D mask rule and does not inspect depth, navmesh,
            # reference paths, or execution results.
            stop_near_mask = target_mask.copy()
            stop_near_mask[:int(round(0.62 * stop_near_mask.shape[0]))] = False
            # Generic STOP relation: the ordinary targetable mask removes the
            # bottom 14% to avoid distorted near-camera rays.  For a semantic
            # stop, that band is precisely where a safe camera-side floor
            # offset may exist.  Restore only the central lower pixels from
            # the original Grounded-SAM floor mask; this is still a 2-D
            # ground constraint and is never used as depth/navmesh evidence.
            h_stop, w_stop = stop_near_mask.shape
            raw_lower = np.asarray(mask, dtype=bool).copy()
            raw_lower[:int(round(0.72 * h_stop))] = False
            raw_lower[:, :int(round(0.10 * w_stop))] = False
            raw_lower[:, int(round(0.90 * w_stop)):] = False
            if raw_lower.any():
                stop_near_mask |= raw_lower
                strategy_application = dict(strategy_application)
                strategy_application["stop_raw_lower_support_expansion"] = {
                    "enabled": True,
                    "minimum_image_y_fraction": 0.72,
                    "policy": "restore central lower Grounded-SAM floor for near-side STOP",
                }
            if stop_near_mask.any():
                target_mask = stop_near_mask
                strategy_application = dict(strategy_application)
                strategy_application["stop_near_floor_band"] = {
                    "enabled": True,
                    "minimum_image_y_fraction": 0.62,
                    "policy": "near-side STOP avoids crossing landmark",
                }
        # Around/backside clauses often have a valid semantic detection but a
        # relation corridor narrower than Grounded-SAM's floor proposal (for
        # example, wood floor beside a detected rug).  Falling back to the
        # one-sided SAM patch lets the anchor ray point back toward the
        # camera instead of continuing around the object.  Expose a bounded
        # lower/interior RGB-only floor prior in that case; the VLM still has
        # to choose the outgoing lane from clean RGB and the prior is never
        # used for depth, navmesh, or arrival decisions.
        if (not strict_ground and stage.get("form") == "CIRCUMNAVIGATE" and
                strategy_application.get("fallback") ==
                "relation_mask_too_small"):
            base_target_mask = rgb_lower_floor_prior(mask.shape)
            ground_mask_source = "rgb_lower_floor_prior_circumnavigate"
            target_mask, strategy_application = constrain_floor_candidates(
                stage, base_target_mask, None, relation_detections, object_mask)
        point, point_score = select_ground_point(target_mask)
        # The detector may return a few side-view floor pixels, but all of
        # them can be outside the targetable interior band.  Treat that as
        # missing ground for explicit turns and use the same bounded RGB
        # lower-band prior as the empty-mask case.  The VLM still sees clean
        # RGB and the final navmesh/point-arrival boundary remains unchanged.
        if (not strict_ground and point is None and stage.get("form") in {
                "TURN_LEFT", "TURN_RIGHT", "TURN_TO_LANDMARK"}):
            fallback_mask = rgb_lower_floor_prior(mask.shape)
            target_mask, fallback_application = constrain_floor_candidates(
                stage, fallback_mask, None, [], object_mask)
            strategy_application = dict(strategy_application)
            strategy_application["turn_rgb_fallback"] = {
                "enabled": True,
                "reason": "no_targetable_side_ground_anchor",
                "prior": "bounded_lower_interior_rgb_floor",
                "physical_validation": "post-selection navmesh ground snap",
            }
            strategy_application["turn_fallback_application"] = (
                fallback_application)
            point, point_score = select_ground_point(target_mask)
        if (not strict_ground and point is None and
                stage.get("form") == "STOP_WAIT"):
            # Grounded-SAM may miss clean wood/tile floor in a view whose
            # landmark is nevertheless visible (especially under glare or
            # wide windows).  Do not collapse the STOP decision to the one
            # view with a noisy rug/carpet proposal.  Retain a bounded,
            # lower-interior RGB floor prior for that view; the VLM still
            # chooses the semantic relation and the post-selection navmesh
            # check remains authoritative.  This is scene-agnostic and uses
            # no depth, reference path, or execution result.
            fallback_mask = rgb_lower_floor_prior(mask.shape)
            if fallback_mask.any():
                target_mask = fallback_mask
                strategy_application = dict(strategy_application)
                strategy_application["stop_rgb_floor_fallback"] = {
                    "enabled": True,
                    "reason": "ground_segmenter_empty_or_non_targetable",
                    "prior": "bounded_lower_interior_rgb_floor",
                }
                point, point_score = select_ground_point(target_mask)
        # Strict dense-majority mode has one immutable physical contract:
        # semantic masks may remove ensemble-ground pixels, never add pixels.
        # Recompute the anchor after this final intersection so every numbered
        # VLM anchor and every downstream point cluster is provably on the
        # original 2/3 majority mask.
        if strict_ground:
            target_mask = np.asarray(target_mask, bool) & mask
            point, point_score = select_ground_point(target_mask)
        score = float(target_mask.mean()) * 8 + point_score / max(rgb.shape[:2])
        if excluded:
            score -= 10
        if hard_excluded:
            score -= 100
        result = {
            "view_index": index,
            "yaw": candidate_yaw, "relative_yaw_rad": float(offset),
            "incoming_back_relative_yaw_rad": (
                float(wrap_angle(back_yaw - reference_yaw))
                if back_yaw is not None else None),
            "backtrack_exclusion_rad": float(backtrack_exclusion),
            "blocked_relative_yaws_rad": [
                float(wrap_angle(blocked - reference_yaw))
                for blocked in blocked_yaws],
            "blocked_direction_exclusion_rad": float(
                blocked_direction_exclusion),
            "rgb": rgb,
            "mask": mask, "target_mask": target_mask,
            "point": point,
            "ground_fraction": float(mask.mean()),
            "ground_mask_source": ground_mask_source,
            "strict_ground_mask": strict_ground,
            "excluded_non_route_ground_labels": excluded_non_route_ground,
            "backtrack_delta_rad": backtrack_delta,
            "blocked_direction_delta_rad": blocked_delta,
            "hard_excluded": hard_excluded,
            "excluded": excluded, "score": score,
            "semantic_backtrack_override": semantic_backtrack_override,
            "soft_incoming_tangent_reopened": semantic_backtrack_override,
            "detection_queries": detection_queries,
            "semantic_detections": detections,
            "detection_records": [
                detection.rgb_prompt_record() for detection in detections],
            "relation_detection_records": [
                detection.rgb_prompt_record()
                for detection in relation_detections],
            "weak_relation_detections": weak_relation_detections,
            "small_seg_object_mask": object_mask,
            "ground_detection_records": [item.rgb_prompt_record()
                                         for item in ground_detections],
            "floor_mask_sanity": floor_mask_sanity,
            "small_seg_objects": [],
            "strategy_application": strategy_application,
            "is_refined_view": bool(refined),
        }
        if not rgb_only:
            result["depth"] = depth
        return result

    for i, (offset, rgb, depth, segmentation) in enumerate(zip(
            offsets, rgbs, depths, segmented_views)):
        candidates.append(build_candidate(
            i, offset, rgb, depth, segmentation=segmentation))
    # STOP_WAIT is an endpoint relation, not an object-passing command.  A
    # single weak detector hit on one panorama sector must not hard-exclude
    # other connected floor sectors: rooms can contain multiple rugs/sofas,
    # and the strongest detector response is not necessarily the instructed
    # instance or the route-consistent side.  Keep all valid floor candidates
    # and expose the detector hit as soft evidence for the VLM.  Other
    # landmark/portal forms retain the configured hard-gate behavior.
    candidate_policy = getattr(
        vlm_harness, "point_selection_candidate_policy",
        "hard_detection_gate")
    if str(stage.get("form", "")).upper() in {
            "STOP_WAIT", "BETWEEN_OBJECTS", "ENTER_REGION",
            "TRAVERSE_PORTAL_REGION"}:
        candidate_policy = "soft_detection_evidence"
    apply_cross_view_semantic_policy(stage, candidates, policy=candidate_policy)

    refinement_attempts = {}

    def refinement_provider(relative_yaw_rad):
        """Acquire a provisional yaw without mutating the final view list.

        The VLM harness can probe several local yaws before accepting one.
        Those failed probes are transactional diagnostics, not panorama views;
        committing them here used to make the outer visualization exceed its
        audited eight-plus-one schema.  Retain the depth-bearing candidate in
        a private transaction table and commit only the returned winner below.
        """
        refined_yaw = wrap_angle(reference_yaw + float(relative_yaw_rad))
        if rgb_only:
            reached_yaw, refined_rgb = continuous_turn(
                sim, None, reference_yaw, refined_yaw, scan_step,
                rendered, motion_log, target_idx, "rgb_refinement_probe",
                video_composer, None, stage["navigation_instruction"],
                policy_input_contract="rgb_only_v1")
            _, _ = continuous_turn(
                sim, None, reached_yaw, reference_yaw, scan_step,
                rendered, motion_log, target_idx,
                "rgb_refinement_probe_return", video_composer, None,
                stage["navigation_instruction"],
                policy_input_contract="rgb_only_v1")
            refined_depth = None
        else:
            set_pose(sim, position, refined_yaw)
            observations = sim.get_sensor_observations()
            refined_rgb = observations["rgb"][..., :3]
            refined_depth = observations["depth"]
            set_pose(sim, position, reference_yaw)
        transaction_id = len(refinement_attempts)
        candidate = build_candidate(
            len(candidates) + transaction_id, float(relative_yaw_rad),
            refined_rgb, refined_depth, refined=True)
        candidate["refinement_transaction_id"] = transaction_id
        refinement_attempts[transaction_id] = candidate
        return {key: value for key, value in candidate.items()
                if key != "depth"}

    # Enforce an architectural boundary: the VLM harness never receives the
    # depth arrays retained by the Habitat interface for the final selected
    # pixel's execution-time backprojection.
    vlm_candidates = [
        {key: value for key, value in candidate.items() if key != "depth"}
        for candidate in candidates]
    chosen_index, vlm_point, vlm_selection = vlm_harness.select_ground_target(
        stage, vlm_candidates, prior_action_history,
        refinement_provider=(refinement_provider if views == 8 else None),
        semantic_reference_rgb=semantic_reference_rgb)
    if chosen_index >= len(candidates):
        if chosen_index >= len(vlm_candidates):
            raise RuntimeError(
                "VLM selected a refined view absent from its candidate list")
        transaction_id = vlm_candidates[chosen_index].get(
            "refinement_transaction_id")
        if transaction_id not in refinement_attempts:
            raise RuntimeError(
                "accepted refined view has no depth-bearing transaction")
        committed = refinement_attempts[transaction_id]
        committed["view_index"] = int(chosen_index)
        committed.pop("refinement_transaction_id", None)
        if chosen_index != len(candidates):
            raise RuntimeError(
                "accepted refined view is not the single ninth candidate")
        candidates.append(committed)
    if (not rgb_only and (maximum_initial_geodesic_m is not None or
        getattr(vlm_harness, "requested_point_selection_prompt_version", None)
        in {"v20_first_step_route_guard",
            "v21_relation_aware_route_review",
            "v22_task30_route_anchor",
            "v23_landmark_turn_stair_ray",
            "v32_first_stage_shallow_route",
            "v25_stage2_route_continuity",
            "v26_stage2_route_consensus",
            "v29_stage3_stop_relation_ray",
            "v30_stage3_relation_portal_ray",
            "v33_stop_relation_near_side",
            "v34_unified_history_route_guard",
            # v31's first-step relation review is also subject to the
            # form-level geodesic safety bound below.  Without this entry a
            # valid-looking RGB ray could send a bounded executor toward a
            # far corridor and exhaust its control budget.
            "v31_circumnavigate_forward_competitor"}
        and (maximum_initial_geodesic_m is not None or
             not prior_action_history or
             getattr(vlm_harness, "requested_point_selection_prompt_version", None)
             in {"v25_stage2_route_continuity",
                 "v26_stage2_route_consensus",
                 "v29_stage3_stop_relation_ray",
                 "v30_stage3_relation_portal_ray",
                 "v33_stop_relation_near_side",
                 "v34_unified_history_route_guard",
                 # v31 is also used after the first node.  Keep the same
                 # post-selection navmesh repair for later-stage relation
                 # targets; RGB-selected pixels beyond a doorway or between
                 # objects can otherwise project just outside the walkable
                 # island and be rejected before the executor starts.
                 "v31_circumnavigate_forward_competitor"}))):
        # Portal/region stages should terminate at the first safe threshold
        # beyond/inside the named opening.  A deep hallway point can be
        # physically reachable yet change the next instruction's reference
        # state.  Bound the post-selection safety search by instruction form;
        # this is a generic spatial relation policy and is not semantic input
        # to the RGB/VLM choice.
        form_max_geodesic = {
            "EXIT_REGION": 3.0,
            "ENTER_REGION": 6.0,
            "SELECT_PORTAL": 3.0,
            "TRAVERSE_PORTAL_REGION": 6.0,
            # These relations need a nearby physical waypoint: a pure
            # orientation command should not become a long approach, and a
            # circumnavigation hop must leave budget for the following
            # endpoint/semantic boundary.
            # TURN_TO_LANDMARK is an orientation clause.  A nearby floor ray
            # may provide a physical node/edge, but a long room traversal
            # silently executes later clauses and makes a weak bearing look
            # complete. Keep the waypoint local and re-observe the landmark.
            "TURN_TO_LANDMARK": 3.0,
            "VERTICAL_UP": 8.0,
            "VERTICAL_DOWN": 8.0,
            "CIRCUMNAVIGATE": 5.0,
            # Directional turns and bracketed-object corridors may require
            # approaching the branch before a safe, ground-projected point
            # exists.  Use the same bounded relation policy rather than
            # rejecting a valid RGB candidate twice at the 4m default.
            "TURN_LEFT": 8.0,
            "TURN_RIGHT": 8.0,
            # A bracket/gap relation should be captured at the local gap, not
            # by a long ray that can silently execute the following clause.
            "BETWEEN_OBJECTS": 3.5,
            "OTHER": 8.0,
        }.get(str(stage.get("form", "")), 4.0)
        if maximum_initial_geodesic_m is not None:
            form_max_geodesic = float(maximum_initial_geodesic_m)
        route_corridor = ((stage.get("metadata", {}) or {}).get(
            "instruction_committed_route_corridor", {}) or {})
        repair_form = str(stage.get("form", "")).upper()
        linear_repair_form = repair_form in {
            "PASS_LANDMARK", "ADVANCE_STRAIGHT", "CROSS_SPACE",
            "FOLLOW_PATH_BOUNDARY", "BETWEEN_OBJECTS"}
        selected_candidate = candidates[chosen_index]
        selected_width = selected_candidate["target_mask"].shape[1]
        selected_center_x = (selected_width - 1.0) / 2.0
        selected_half_width = max(selected_width / 2.0, 1.0)
        selected_pixel_bearing = math.atan(
            (float(vlm_point[0]) - selected_center_x) /
            selected_half_width)
        selected_pixel_route_yaw = wrap_angle(
            float(selected_candidate["yaw"]) - selected_pixel_bearing)
        repair_absolute_route_yaw = (
            float(route_corridor["absolute_yaw_rad"])
            if route_corridor.get("active") and
            route_corridor.get("absolute_yaw_rad") is not None else
            selected_pixel_route_yaw if linear_repair_form else None)
        repair_route_half_width = (
            math.radians(float(route_corridor.get(
                "half_width_deg", 45.0)))
            if route_corridor.get("active") else
            math.radians(30.0) if linear_repair_form else None)
        repaired_index, repaired_point, repair = repair_selected_ground_point(
            sim, candidates, chosen_index, vlm_point, position,
            selection=vlm_selection,
            max_view_delta_rad=(
                math.radians(45.0) if (repair_form in {
                    "PASS_LANDMARK", "ADVANCE_STRAIGHT",
                    "CROSS_SPACE", "FOLLOW_PATH_BOUNDARY",
                    "BETWEEN_OBJECTS"} and
                    not bool((stage.get("metadata", {}) or {}).get(
                        "bare_turn_route_setup", {}).get("active"))) else
                math.radians(float(vlm_selection.get(
                    "postselection_repair_max_view_delta_deg", 45.0)))),
            max_geodesic_m=form_max_geodesic,
            minimum_geodesic_m=float(minimum_initial_geodesic_m),
            # A bracketed gap can leave only thin disconnected-looking floor
            # islands between two object masks.  Search the same frozen
            # Grounded-SAM mask more densely before declaring it physically
            # unusable; no unsegmented pixel, depth, navmesh, or demo signal
            # is exposed to the VLM or used to change its semantic view.
            anchor_sample_count=(
                128 if (str(stage.get("form", "")).upper() in {
                    "BETWEEN_OBJECTS", "TURN_AROUND", "TURN_TO_LANDMARK",
                    "TURN_LEFT", "TURN_RIGHT", "VERTICAL_UP",
                    "VERTICAL_DOWN", "SELECT_PORTAL", "STOP_WAIT",
                    "EXIT_REGION", "ENTER_REGION",
                    "TRAVERSE_PORTAL_REGION"} or linear_repair_form or
                    (maximum_initial_geodesic_m is not None and
                     float(maximum_initial_geodesic_m) <= 3.0)) else 16),
            # The floor immediately under a horizontal camera is invisible,
            # and Grounded-SAM can clip a few columns from a bottom-touching
            # rear patch. Extend only that lower boundary for pure turns;
            # every added pixel remains subject to navmesh/path validation.
            lower_mask_boundary_extension_px=(
                12 if (not strict_ground and
                       str(stage.get("form", "")).upper() ==
                       "TURN_AROUND") else 0),
            absolute_route_yaw_rad=repair_absolute_route_yaw,
            route_half_width_rad=repair_route_half_width)
        if repair.get("status") != "selected_point_reachable":
            vlm_selection = dict(vlm_selection)
            vlm_selection["postselection_repair"] = repair
            chosen_index, vlm_point = repaired_index, repaired_point
    chosen = candidates[chosen_index]
    final_mask = np.asarray(chosen["target_mask"], dtype=bool)
    final_xy = np.round(np.asarray(vlm_point, np.float32)).astype(int)
    final_xy[0] = np.clip(final_xy[0], 0, final_mask.shape[1] - 1)
    final_xy[1] = np.clip(final_xy[1], 0, final_mask.shape[0] - 1)
    if strict_ground and not final_mask[final_xy[1], final_xy[0]]:
        # A replaceable VLM backend may return free-form coordinates instead
        # of a numbered anchor. Snap only within the already-visible strict
        # target mask, with no RGB rectangle, depth, or navmesh expansion.
        ys, xs = np.nonzero(final_mask)
        if not len(xs):
            raise RuntimeError(
                "VLM selected a view without dense-majority ground anchors")
        nearest = np.argmin(
            (xs.astype(np.float32) - float(vlm_point[0])) ** 2 +
            (ys.astype(np.float32) - float(vlm_point[1])) ** 2)
        vlm_point = np.array([float(xs[nearest]), float(ys[nearest])], np.float32)
        vlm_selection = dict(vlm_selection)
        vlm_selection["strict_ground_snap"] = {
            "applied": True,
            "source": "dense_majority_2_of_3_target_mask",
            "final_xy": vlm_point.tolist(),
        }
    chosen["point"] = vlm_point
    chosen["vlm_selection"] = vlm_selection
    if not rgb_only and float(minimum_initial_geodesic_m) > 0.0:
        selected_world, _ = pixel_ground_to_world(
            vlm_point, chosen.get("depth"), position,
            float(chosen.get("yaw", reference_yaw)))
        selected_geodesic = math.inf
        if selected_world is not None:
            snapped = np.asarray(sim.pathfinder.snap_point(selected_world),
                                 np.float32)
            if np.isfinite(snapped).all():
                shortest = habitat_sim.ShortestPath()
                shortest.requested_start = np.asarray(position, np.float32)
                shortest.requested_end = snapped
                if sim.pathfinder.find_path(shortest):
                    selected_geodesic = float(shortest.geodesic_distance)
        if selected_geodesic < float(minimum_initial_geodesic_m):
            chosen["vlm_selection"] = dict(vlm_selection)
            chosen["vlm_selection"]["minimum_progress_reject"] = {
                "status": "too_near_to_progress",
                "initial_geodesic_m": float(selected_geodesic),
                "minimum_progress_distance_m": float(
                    minimum_initial_geodesic_m),
                "policy": (
                    "after partial/on-route unknown, reject repeated near-zero "
                    "point and reacquire a forward legal ground target"),
            }
    append_decision_visualization(
        rendered, candidates, chosen_index, stage, prior_action_history,
        video_composer, position_history, position,
        reference_path=reference_path,
        reference_path_index=reference_path_index)
    current_yaw, chosen_rgb = continuous_turn(
        sim, position, yaw, chosen["yaw"], scan_step,
        rendered, motion_log, target_idx, "turn_to_selected_target",
        video_composer, position_history, stage["navigation_instruction"],
        reference_path=reference_path,
        reference_path_index=reference_path_index,
        policy_input_contract=policy_input_contract,
    )
    chosen["rgb"] = chosen_rgb
    return chosen, candidates


def append_exploration_visualization(rendered, candidates, chosen_index, target_idx,
                                     history_size, video_composer,
                                     position_history, position, instruction,
                                     reference_path=None,
                                     reference_path_index=0):
    for index, candidate in enumerate(candidates):
        rgb = candidate["rgb"].copy()
        mask = candidate["target_mask"]
        green = np.zeros_like(rgb)
        green[..., 1] = 255
        rgb[mask] = (0.55 * rgb[mask] + 0.45 * green[mask]).astype(np.uint8)
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if reference_path:
            camera_position = (np.asarray(position, np.float32) +
                               np.array([0.0, 1.25, 0.0], np.float32))
            projection = project_reference_path(
                reference_path,
                range(int(reference_path_index), len(reference_path)),
                camera_position, float(candidate["yaw"]),
                rgb.shape[1], rgb.shape[0])
            rgb = draw_reference_path_overlay(
                rgb, projection,
                selected_point=(candidate.get("point")
                                if index == chosen_index else None),
                selected=bool(index == chosen_index and
                              candidate.get("point") is not None),
                next_path_index=int(reference_path_index) + 1)
            candidate["reference_path_projection"] = projection
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        status = "NOVEL" if candidate["eligible"] else "VISITED/INVALID"
        color = (255, 255, 255) if candidate["eligible"] else (0, 0, 255)
        cv2.putText(frame, f"EXPLORE VIEW {index} {status}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
        cv2.putText(frame, f"frontier {target_idx + 1} history {history_size}", (6, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
        cv2.putText(frame, f"novelty {candidate['novelty_distance_m']:.2f}m depth "
                    f"{candidate['selected_depth_m']:.2f}m", (6, 57),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
        if index == chosen_index:
            point = tuple(np.round(candidate["point"]).astype(int))
            cv2.drawMarker(frame, point, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(frame, "PURE EXPLORE SELECTED", (6, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 2)
        composed = video_composer.compose(
            frame, position_history, position, candidate["yaw"], instruction,
            target_idx, "six_view_ground_decision")
        rendered.extend([composed, composed.copy()])


def evaluate_frontier_point(sim, point, point_score, depth, position, candidate_yaw,
                            position_history, blocked_frontiers, novelty_radius):
    world_point, selected_depth = pixel_ground_to_world(
        point, depth, position, candidate_yaw)
    snapped = None
    geodesic = math.inf
    path_novelty = {"sample_count": 0, "novel_fraction": 0.0,
                    "overlap_fraction": 1.0, "minimum_history_distance_m": 0.0}
    if world_point is not None:
        snapped_value = np.asarray(sim.pathfinder.snap_point(world_point), np.float32)
        if np.isfinite(snapped_value).all():
            snapped = snapped_value
            path = habitat_sim.ShortestPath()
            path.requested_start = np.asarray(position, np.float32)
            path.requested_end = snapped
            if sim.pathfinder.find_path(path):
                geodesic = float(path.geodesic_distance)
                path_novelty = path_novelty_statistics(
                    path.points, position_history, novelty_radius * 0.6)
    novelty = (distance_to_position_history(snapped, position_history)
               if snapped is not None else 0.0)
    blocked_distance = (distance_to_position_history(snapped, blocked_frontiers)
                        if snapped is not None and blocked_frontiers else math.inf)
    eligible = bool(snapped is not None and math.isfinite(geodesic) and
                    novelty >= novelty_radius and blocked_distance >= novelty_radius)
    depth_score = selected_depth if math.isfinite(selected_depth) else 0.0
    return {
        "point": point, "point_score": point_score, "world_point": snapped,
        "selected_depth_m": selected_depth, "geodesic_distance_m": geodesic,
        "path_novelty": path_novelty,
        "blocked_frontier_distance_m": blocked_distance,
        "novelty_distance_m": novelty, "eligible": eligible,
        "score": novelty + 2.0 * path_novelty["novel_fraction"] +
                 0.15 * min(depth_score, 8.0),
    }


def choose_exploration_view(sim, position, yaw, segmenter, position_history,
                            blocked_frontiers, frontier_memory, frontier_attempts,
                            novelty_radius, rendered, motion_log, target_idx,
                            video_composer, instruction,
                            views=6, scan_step=math.radians(10),
                            max_frontier_transit_attempts=3,
                            reference_path=None, reference_path_index=0):
    """Choose the ground endpoint furthest from all simulator position history."""
    if views != 6:
        raise ValueError("Pure exploration requires six Habitat panorama sensors")
    rgbs, depths = observe_six_rgbd(sim)
    offsets = np.radians([0, 60, 120, 180, 240, 300])
    segmentations = (segmenter.batch(rgbs)
                     if hasattr(segmenter, "batch") else
                     [segmenter(rgb) for rgb in rgbs])
    candidates = []
    for index, (offset, rgb, depth, segmentation) in enumerate(zip(
            offsets, rgbs, depths, segmentations)):
        candidate_yaw = wrap_angle(yaw + float(offset))
        ground, ground_detections = segmentation
        target_mask = targetable_ground_mask(ground)
        point_options = [evaluate_frontier_point(
            sim, point, point_score, depth, position, candidate_yaw,
            position_history, blocked_frontiers, novelty_radius)
            for point, point_score in exploration_ground_points(target_mask, depth)]
        eligible_options = [option for option in point_options if option["eligible"]]
        if eligible_options:
            best = max(eligible_options, key=lambda option: option["score"])
        elif point_options:
            best = max(point_options, key=lambda option: option["score"])
        else:
            best = {
                "point": None, "point_score": 0.0, "world_point": None,
                "selected_depth_m": math.nan, "geodesic_distance_m": math.inf,
                "path_novelty": {"sample_count": 0, "novel_fraction": 0.0,
                                 "overlap_fraction": 1.0,
                                 "minimum_history_distance_m": 0.0},
                "blocked_frontier_distance_m": math.inf,
                "novelty_distance_m": 0.0, "eligible": False, "score": 0.0,
            }
        candidates.append({
            "view_index": index, "yaw": candidate_yaw,
            "relative_yaw_rad": float(offset), "rgb": rgb, "depth": depth,
            "mask": ground, "target_mask": target_mask,
            **best,
            "ground_point_options": [{
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in option.items()
            } for option in point_options],
            "ground_fraction": float(ground.mean()),
            "ground_detection_records": [item.prompt_record()
                                         for item in ground_detections],
        })
    # Remember every observed but unvisited endpoint, not only the winner. This
    # lets exploration return to an earlier junction after a local dead end.
    for candidate in candidates:
        for option in candidate["ground_point_options"]:
            frontier = option.get("world_point")
            if option.get("eligible") and frontier is not None and not any(
                    np.linalg.norm((np.asarray(frontier) - old)[[0, 2]]) < 0.6
                    for old in frontier_memory):
                frontier_memory.append(np.asarray(frontier, np.float32))

    eligible = [item for item in candidates if item["eligible"]]
    selection_method = "max_distance_from_sim_position_history"
    memory_target = None
    if eligible:
        chosen = max(eligible, key=lambda item: item["score"])
    else:
        viable_memory = [frontier for frontier in frontier_memory
                         if distance_to_position_history(frontier, position_history) >= novelty_radius
                         and (not blocked_frontiers or
                              distance_to_position_history(frontier, blocked_frontiers) >= novelty_radius)
                         and frontier_attempts.get(frontier_key(frontier), 0) <
                         max_frontier_transit_attempts]
        routed = []
        for frontier in viable_memory:
            path = habitat_sim.ShortestPath()
            path.requested_start = np.asarray(position, np.float32)
            path.requested_end = frontier
            if sim.pathfinder.find_path(path) and len(path.points) >= 2:
                routed.append((float(path.geodesic_distance), frontier, path.points))
        visible_ground = [item for item in candidates if item["point"] is not None and
                          math.isfinite(item["geodesic_distance_m"])]
        if not routed or not visible_ground:
            return None, candidates
        _, memory_target, route_points = min(routed, key=lambda item: item[0])
        waypoint = np.asarray(route_points[1], np.float32)
        delta = waypoint - np.asarray(position, np.float32)
        desired_yaw = wrap_angle(math.atan2(-float(delta[0]), -float(delta[2])))
        def routing_error(candidate):
            h, w = candidate["rgb"].shape[:2]
            pixel_bearing = math.atan((candidate["point"][0] - w / 2) / (w / 2))
            ray_yaw = wrap_angle(candidate["yaw"] - pixel_bearing)
            return angle_distance(ray_yaw, desired_yaw)
        chosen = min(visible_ground, key=routing_error)
        chosen["eligible"] = True
        chosen["transit_to_frontier"] = memory_target.tolist()
        selection_method = "transit_via_visible_ground_to_remembered_frontier"
    chosen_index = chosen["view_index"]
    append_exploration_visualization(
        rendered, candidates, chosen_index, target_idx, len(position_history),
        video_composer, position_history, position, instruction,
        reference_path=reference_path,
        reference_path_index=reference_path_index)
    current_yaw, chosen_rgb = continuous_turn(
        sim, position, yaw, chosen["yaw"], scan_step, rendered, motion_log,
        target_idx, "turn_to_unvisited_frontier", video_composer,
        position_history, instruction,
        reference_path=reference_path,
        reference_path_index=reference_path_index)
    chosen["rgb"] = chosen_rgb
    chosen["selection"] = {
        "method": selection_method,
        "view_index": chosen_index, "point_xy": chosen["point"].tolist(),
        "frontier_world_xyz": chosen["world_point"].tolist(),
        "novelty_distance_m": chosen["novelty_distance_m"],
        "path_novelty": chosen["path_novelty"],
        "remembered_frontier_world_xyz": (memory_target.tolist()
                                           if memory_target is not None else None),
        "remembered_frontier_attempts_before": (
            frontier_attempts.get(frontier_key(memory_target), 0)
            if memory_target is not None else None),
    }
    return chosen, candidates




@dataclass
class PointSelectionRequest:
    """State exposed by Habitat to an external point-selection strategy."""

    sim: Any
    position: np.ndarray
    yaw: float
    stage: dict
    target_index: int
    rendered: Any
    motion_log: list
    position_history: list
    previous_action_history: list = field(default_factory=list)
    back_yaw: Optional[float] = None
    blocked_yaws: list[float] = field(default_factory=list)
    # Optional hidden R2R diagnostic context.  Selection strategies must not
    # forward this path to a model; it is consumed only by video overlays.
    reference_path: Optional[list] = None
    reference_path_index: int = 0
    # When an active instruction is known to be partial/on-route, reject a
    # repeated near-zero-progress point and let the selector acquire another
    # legal view/anchor.  This is a policy hint, not semantic completion data.
    minimum_progress_distance_m: float = 0.0
    # Optional policy override for a later on-route continuation that needs a
    # deeper target after an initially shallow portal threshold.
    maximum_initial_geodesic_m: Optional[float] = None
    # Clean RGB identity memory from an earlier real node in this same active
    # sub-instruction. It never contains depth, navmesh, or reference path.
    semantic_reference_rgb: Optional[np.ndarray] = None
    # One bounded retry may reopen the incoming/backtracking sectors after
    # the initially selected RGB view proves to be visible only through an
    # incompatible navmesh route.  The rejected direction remains blocked.
    allow_historical_direction_fallback: bool = False
    policy_input_contract: str = "legacy_rgbd_geometry"


@dataclass
class PointSelectionResult:
    """Selected point/view plus candidates retained for audit and rendering."""

    chosen: Optional[dict]
    candidates: list


class PointSelectionStrategy:
    """Common external-selection contract used by the Habitat interface."""

    mode = "base"

    def select(self, request: PointSelectionRequest) -> PointSelectionResult:
        raise NotImplementedError

    def on_navigation_result(
            self, selection: PointSelectionResult, navigation_result,
            position_history, target_record):
        """Update strategy state after the executor returns."""
        return None


class InstructionVLMPointSelector(PointSelectionStrategy):
    """Select one grounded point per instruction stage through a VLM harness."""

    mode = "instruction-vlm"

    # Half the 45-degree eight-view spacing: a blocked view-centre ray excludes
    # exactly that one panorama view instead of its neighbours as well.
    DEFAULT_BLOCKED_DIRECTION_EXCLUSION_DEG = 22.5

    def __init__(self, segmenter, semantic_detector, vlm_harness,
                 video_composer, views=6, scan_step=math.radians(10),
                 policy_input_contract="legacy_rgbd_geometry",
                 blocked_direction_exclusion_deg=(
                     DEFAULT_BLOCKED_DIRECTION_EXCLUSION_DEG)):
        self.segmenter = segmenter
        self.semantic_detector = semantic_detector
        self.vlm_harness = vlm_harness
        self.video_composer = video_composer
        self.views = int(views)
        self.scan_step = float(scan_step)
        self.policy_input_contract = str(policy_input_contract)
        self.blocked_direction_exclusion = math.radians(
            float(blocked_direction_exclusion_deg))

    def select(self, request):
        if request.policy_input_contract != self.policy_input_contract:
            raise ValueError(
                "point-selection request/strategy input contracts differ")
        chosen, candidates = choose_view(
            request.sim, request.position, request.yaw, request.back_yaw,
            self.segmenter, self.semantic_detector, self.vlm_harness,
            request.stage, request.previous_action_history, request.rendered,
            request.motion_log, request.target_index, self.video_composer,
            request.position_history, self.views, scan_step=self.scan_step,
            backtrack_exclusion=(
                0.0 if request.allow_historical_direction_fallback else
                math.radians(50.0)),
            blocked_yaws=request.blocked_yaws,
            blocked_direction_exclusion=self.blocked_direction_exclusion,
            reference_path=request.reference_path,
            reference_path_index=request.reference_path_index,
            minimum_initial_geodesic_m=(
                request.minimum_progress_distance_m),
            maximum_initial_geodesic_m=(
                request.maximum_initial_geodesic_m),
            semantic_reference_rgb=request.semantic_reference_rgb,
            policy_input_contract=self.policy_input_contract)
        return PointSelectionResult(chosen=chosen, candidates=candidates)

class RandomExplorationPointSelector(PointSelectionStrategy):
    """Choose novel reachable floor points without instruction semantics."""

    mode = "random-exploration"

    def __init__(
            self, segmenter, video_composer, novelty_radius=1.0, views=6,
            scan_step=math.radians(10), max_frontier_transit_attempts=3):
        self.segmenter = segmenter
        self.video_composer = video_composer
        self.novelty_radius = float(novelty_radius)
        self.views = int(views)
        self.scan_step = float(scan_step)
        self.max_frontier_transit_attempts = int(
            max_frontier_transit_attempts)
        self.blocked_frontiers = []
        self.frontier_memory = []
        self.frontier_attempts = {}
        self.termination_reason = None

    def select(self, request):
        chosen, candidates = choose_exploration_view(
            request.sim, request.position, request.yaw, self.segmenter,
            request.position_history, self.blocked_frontiers,
            self.frontier_memory, self.frontier_attempts,
            self.novelty_radius, request.rendered, request.motion_log,
            request.target_index, self.video_composer,
            request.stage["navigation_instruction"], self.views,
            self.scan_step, self.max_frontier_transit_attempts,
            reference_path=request.reference_path,
            reference_path_index=request.reference_path_index)
        if chosen is None:
            self.termination_reason = "no_unvisited_ground_frontier"
        return PointSelectionResult(chosen=chosen, candidates=candidates)

    def on_navigation_result(
            self, selection, navigation_result, position_history,
            target_record):
        chosen = selection.chosen
        remembered = chosen["selection"].get(
            "remembered_frontier_world_xyz")
        if remembered is not None:
            remembered_frontier = np.asarray(remembered, np.float32)
            remaining_distance = distance_to_position_history(
                remembered_frontier, position_history)
            key = frontier_key(remembered_frontier)
            visited = remaining_distance < self.novelty_radius
            if visited:
                self.frontier_attempts.pop(key, None)
            else:
                self.frontier_attempts[key] = (
                    self.frontier_attempts.get(key, 0) + 1)
            target_record["remembered_frontier_visit"] = {
                "world_xyz": remembered_frontier.tolist(),
                "visited": visited,
                "remaining_history_distance_m": remaining_distance,
                "attempts_after_segment": self.frontier_attempts.get(key, 0),
            }
            if (not visited and self.frontier_attempts[key] >=
                    self.max_frontier_transit_attempts):
                self.blocked_frontiers.append(remembered_frontier)
                target_record[
                    "frontier_blacklisted_after_repeated_transit"] = True
                target_record["blacklisted_frontier_world_xyz"] = (
                    remembered_frontier.tolist())
        elif not navigation_result.arrived:
            failed_frontier = np.asarray(chosen["world_point"], np.float32)
            self.blocked_frontiers.append(failed_frontier)
            target_record["frontier_blacklisted_after_failure"] = True
            target_record["blacklisted_frontier_world_xyz"] = (
                failed_frontier.tolist())

    def snapshot(self, position_history):
        remaining = sum(
            distance_to_position_history(
                frontier, position_history) >= self.novelty_radius and
            (not self.blocked_frontiers or distance_to_position_history(
                frontier, self.blocked_frontiers) >= self.novelty_radius)
            for frontier in self.frontier_memory)
        return {
            "blocked_frontiers_xyz": [
                point.tolist() for point in self.blocked_frontiers],
            "remembered_frontiers_xyz": [
                point.tolist() for point in self.frontier_memory],
            "frontier_transit_attempts": {
                f"{key[0]},{key[1]}": value
                for key, value in self.frontier_attempts.items()
            },
            "remaining_unvisited_frontiers": remaining,
            "termination_reason": self.termination_reason,
        }
