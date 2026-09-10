#!/usr/bin/env python3
"""Six-view node-to-node point selection and graph backtracking execution."""

from __future__ import annotations

import math
import heapq
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from path_projection import draw_reference_path_overlay, project_reference_path

from instruction_decomposer import SubInstruction
from navigation_graph_memory import CompactVisualEmbedder


PANORAMA_OFFSETS_RAD = np.radians([0, 60, 120, 180, 240, 300])
BACKTRACK_PLANNER_PROFILES = {
    "legacy_direct",
    "breadcrumb_guard_v1",
    "breadcrumb_budget_v2",
    "breadcrumb_budget_v3",
    "breadcrumb_endpoint_v4",
    "breadcrumb_endpoint_v5",
}


def _wrap_angle(value):
    return (value + math.pi) % (2 * math.pi) - math.pi


def navmesh_first_route_waypoint(sim, start_position, target_position,
                                 minimum_progress_m=0.25):
    """Return the first meaningful navmesh waypoint toward a stored node."""
    if sim is None:
        return None
    shortest = __import__("habitat_sim").ShortestPath()
    start = np.asarray(start_position, np.float32)
    target = np.asarray(target_position, np.float32)
    shortest.requested_start = start
    shortest.requested_end = target
    try:
        if not sim.pathfinder.find_path(shortest):
            return None
    except Exception:
        return None
    points = [np.asarray(point, np.float32)
              for point in (getattr(shortest, "points", None) or [])]
    for point in points[1:]:
        if float(np.linalg.norm(point[[0, 2]] - start[[0, 2]])) >= float(
                minimum_progress_m):
            return point
    if points:
        return points[-1]
    return target


def navmesh_route_distance(sim, start_position, target_position):
    """Return current-to-node geodesic distance, or infinity if unavailable."""
    if sim is None:
        return math.inf
    shortest = __import__("habitat_sim").ShortestPath()
    shortest.requested_start = np.asarray(start_position, np.float32)
    shortest.requested_end = np.asarray(target_position, np.float32)
    try:
        if sim.pathfinder.find_path(shortest):
            return float(shortest.geodesic_distance)
    except Exception:
        pass
    return math.inf


def backtrack_segment_travel_budget(
        breadcrumb, controller_step_m=0.22, settling_margin_m=0.10):
    """Budget one reverse breadcrumb plus a discrete endpoint-control step.

    The matcher retains the strict 0.75 m physical and visual gates.  This
    margin only prevents the point executor from stopping on its travel budget
    before it can take the final discrete action that enters that radius.
    """
    route_arc = abs(
        float(breadcrumb["target_route_progress_m"]) -
        float(breadcrumb["current_route_progress_m"]))
    off_trace = max(0.0, float(breadcrumb["off_trace_distance_m"]))
    discretization_margin = max(
        0.15, float(controller_step_m) + float(settling_margin_m))
    return max(0.35, route_arc + off_trace + discretization_margin)


def recovery_executor_endpoint(planner_profile, selection_record,
                               selected_navmesh, selected_geodesic,
                               stored_node_position,
                               stored_node_route_geodesic,
                               short_range_threshold_m=1.5):
    """Choose the executor's final geometry target for node recovery.

    The RGB ground point remains the visual/tracking goal.  At short range a
    floor anchor can lie beyond the stored pose, so v4 supplies the already
    known graph node to the executor's final-approach controller.  This is
    recovery geometry only and is never exposed to semantic point selection.
    """
    profile = str(planner_profile)
    if profile == "breadcrumb_endpoint_v5":
        short_range_threshold_m = max(float(short_range_threshold_m), 3.0)
    use_stored_node = bool(
        profile in {"breadcrumb_endpoint_v4", "breadcrumb_endpoint_v5"} and
        stored_node_position is not None and
        math.isfinite(float(stored_node_route_geodesic)) and
        float(stored_node_route_geodesic) <= float(short_range_threshold_m))
    if use_stored_node:
        return (np.asarray(stored_node_position, np.float32),
                float(stored_node_route_geodesic), "stored_node_short_range")
    return selected_navmesh, selected_geodesic, "selected_floor_anchor"


def choose_recovery_candidate(eligible, target_route_geodesic_m,
                              short_range_threshold_m=1.5):
    """Rank recovery views, prioritizing exact-node vicinity near the node."""
    short_range = bool(
        math.isfinite(float(target_route_geodesic_m)) and
        float(target_route_geodesic_m) <= float(short_range_threshold_m))
    proximity_candidates = [
        item for item in eligible
        if item.get("selected_point_reachable") is True and
        item.get("selected_point_target_planar_distance_m") is not None and
        math.isfinite(float(
            item["selected_point_target_planar_distance_m"]))]
    if short_range and proximity_candidates:
        return min(proximity_candidates, key=lambda item: (
            float(item["selected_point_target_planar_distance_m"]),
            float(item["yaw_error_rad"]),
            -float(item["hybrid_score"]))), True
    return max(eligible, key=lambda item: item["hybrid_score"]), False


def _position_from_record(record):
    value = record.get("position_xyz") if isinstance(record, dict) else None
    if value is None:
        return None
    position = np.asarray(value, np.float32)
    if position.shape != (3,) or not np.isfinite(position).all():
        return None
    return position


def _forward_edge_trace(forward_edge, target_position, source_position):
    """Recover the robot's own forward trace; never consumes the R2R demo."""
    target = np.asarray(target_position, np.float32)
    source = np.asarray(source_position, np.float32)
    action_points = [
        point for point in (
            _position_from_record(item)
            for item in getattr(forward_edge, "action_history", []))
        if point is not None]
    metadata = getattr(forward_edge, "metadata", {}) or {}
    keyframe_points = [
        point for point in (
            _position_from_record(item)
            for item in metadata.get("edge_keyframes", []))
        if point is not None]
    middle = action_points if action_points else keyframe_points
    raw_points = [target, *middle, source]
    points = [raw_points[0]]
    for point in raw_points[1:]:
        if float(np.linalg.norm(point[[0, 2]] - points[-1][[0, 2]])) > 0.025:
            points.append(point)
    if len(points) == 1:
        points.append(source)
    return points, (
        "forward_action_pose_trace" if action_points else
        "forward_edge_keyframe_trace" if keyframe_points else
        "node_endpoint_fallback")


def _point_at_polyline_progress(points, progress_m):
    progress_m = max(0.0, float(progress_m))
    elapsed = 0.0
    for first, second in zip(points[:-1], points[1:]):
        length = float(np.linalg.norm(
            np.asarray(second)[[0, 2]] - np.asarray(first)[[0, 2]]))
        if elapsed + length >= progress_m and length > 1e-6:
            fraction = (progress_m - elapsed) / length
            return np.asarray(first, np.float32) + fraction * (
                np.asarray(second, np.float32) - np.asarray(first, np.float32))
        elapsed += length
    return np.asarray(points[-1], np.float32).copy()


def reverse_route_breadcrumb(
        forward_edge, target_position, source_position, current_position,
        lookahead_m=1.5, maximum_progress_m=None):
    """Choose the next earlier point along a stored *executed* edge trace."""
    points, trace_source = _forward_edge_trace(
        forward_edge, target_position, source_position)
    current = np.asarray(current_position, np.float32)
    cumulative = 0.0
    best_distance = math.inf
    projected_progress = 0.0
    for first, second in zip(points[:-1], points[1:]):
        first = np.asarray(first, np.float32)
        second = np.asarray(second, np.float32)
        vector = second[[0, 2]] - first[[0, 2]]
        length = float(np.linalg.norm(vector))
        if length <= 1e-6:
            continue
        fraction = float(np.clip(
            np.dot(current[[0, 2]] - first[[0, 2]], vector) /
            max(length * length, 1e-8), 0.0, 1.0))
        projected = first[[0, 2]] + fraction * vector
        distance = float(np.linalg.norm(current[[0, 2]] - projected))
        progress = cumulative + fraction * length
        # Prefer later progress when a path crosses itself and distances tie;
        # the monotonic cap below prevents a later attempt from moving forward.
        if (distance < best_distance - 1e-6 or
                abs(distance - best_distance) <= 1e-6 and
                progress > projected_progress):
            best_distance = distance
            projected_progress = progress
        cumulative += length
    if maximum_progress_m is not None:
        projected_progress = min(
            projected_progress, float(maximum_progress_m))
    target_progress = max(0.0, projected_progress - float(lookahead_m))
    breadcrumb = _point_at_polyline_progress(points, target_progress)
    return {
        "position_xyz": breadcrumb,
        "trace_source": trace_source,
        "trace_point_count": len(points),
        "trace_length_m": cumulative,
        "current_route_progress_m": projected_progress,
        "target_route_progress_m": target_progress,
        "off_trace_distance_m": best_distance,
        "lookahead_m": float(lookahead_m),
        "route_direction": "reverse",
    }


def forward_route_breadcrumb(
        forward_edge, source_position, target_position, current_position,
        lookahead_m=1.5, minimum_progress_m=None):
    """Choose the next later point while replaying a stored edge forward."""
    points, trace_source = _forward_edge_trace(
        forward_edge, source_position, target_position)
    current = np.asarray(current_position, np.float32)
    cumulative = 0.0
    best_distance = math.inf
    projected_progress = 0.0
    for first, second in zip(points[:-1], points[1:]):
        first = np.asarray(first, np.float32)
        second = np.asarray(second, np.float32)
        vector = second[[0, 2]] - first[[0, 2]]
        length = float(np.linalg.norm(vector))
        if length <= 1e-6:
            continue
        fraction = float(np.clip(
            np.dot(current[[0, 2]] - first[[0, 2]], vector) /
            max(length * length, 1e-8), 0.0, 1.0))
        projected = first[[0, 2]] + fraction * vector
        distance = float(np.linalg.norm(current[[0, 2]] - projected))
        progress = cumulative + fraction * length
        if (distance < best_distance - 1e-6 or
                abs(distance - best_distance) <= 1e-6 and
                progress < projected_progress):
            best_distance = distance
            projected_progress = progress
        cumulative += length
    if minimum_progress_m is not None:
        projected_progress = max(
            projected_progress, float(minimum_progress_m))
    target_progress = min(
        cumulative, projected_progress + float(lookahead_m))
    breadcrumb = _point_at_polyline_progress(points, target_progress)
    return {
        "position_xyz": breadcrumb,
        "trace_source": trace_source,
        "trace_point_count": len(points),
        "trace_length_m": cumulative,
        "current_route_progress_m": projected_progress,
        "target_route_progress_m": target_progress,
        "off_trace_distance_m": best_distance,
        "lookahead_m": float(lookahead_m),
        "route_direction": "forward",
    }


def _observe_six_rgbd(sim):
    observations = sim.get_sensor_observations()
    rgbs = [observations["rgb"][..., :3]] + [
        observations[f"pano_rgb_{index}"][..., :3] for index in range(1, 6)]
    depths = [observations["depth"]] + [
        observations[f"pano_depth_{index}"] for index in range(1, 6)]
    return rgbs, depths


def _observe_eight_completion_rgb(sim):
    """Best-effort completion panorama; old unit-test simulators may lack it."""
    observations = sim.get_sensor_observations()
    keys = [f"completion_rgb_{index}" for index in range(1, 8)]
    if not all(key in observations for key in keys):
        return None
    return [observations["rgb"][..., :3]] + [
        observations[key][..., :3] for key in keys]


def _targetable_ground_mask(mask):
    height, width = mask.shape
    valid = np.asarray(mask, bool).copy()
    valid[: int(0.42 * height)] = False
    valid[int(0.86 * height):] = False
    valid[:, : int(0.10 * width)] = False
    valid[:, int(0.90 * width):] = False
    return valid if valid.any() else np.asarray(mask, bool)


def _select_ground_point(mask, preferred_y=0.68):
    height, width = mask.shape
    valid = _targetable_ground_mask(mask).astype(np.uint8)
    if not valid.any():
        return None, 0.0
    distance = cv2.distanceTransform(valid, cv2.DIST_L2, 5)
    yy, xx = np.indices(mask.shape)
    center_prior = np.exp(-((xx - width / 2) / (0.32 * width)) ** 2)
    depth_prior = np.exp(-((yy - preferred_y * height) / (0.22 * height)) ** 2)
    score = distance * (0.35 + 0.65 * center_prior * depth_prior)
    y, x = np.unravel_index(np.argmax(score), score.shape)
    return np.array([float(x), float(y)], np.float32), float(score[y, x])


def _select_reachable_ground_point(
        sim, mask, depth, position, yaw, candidate_count=12,
        preferred_world_position=None):
    """Return a reachable floor anchor, preferring the stored-node vicinity.

    Forward semantic selection must remain image driven.  Backtracking is
    different: it already has an explicit stored node/breadcrumb destination.
    Picking the first deep floor maximum can send the executor several metres
    beyond a node that is less than one metre away.  Evaluate a small diverse
    set of image-floor anchors and, when a stored destination is supplied,
    select the reachable projection nearest that destination.  The returned
    pixel and its depth/world projection remain mutually consistent.
    """
    from point_selectors import (
        exploration_ground_points, pixel_ground_to_world)

    reachable = []
    for point, score in exploration_ground_points(
            mask, depth, count=int(candidate_count)):
        world, selected_depth = pixel_ground_to_world(
            point, depth, position, yaw)
        if world is None:
            continue
        snapped = np.asarray(sim.pathfinder.snap_point(world), np.float32)
        if not np.isfinite(snapped).all():
            continue
        shortest = __import__("habitat_sim").ShortestPath()
        shortest.requested_start = np.asarray(position, np.float32)
        shortest.requested_end = snapped
        if not sim.pathfinder.find_path(shortest):
            continue
        preferred_distance = None
        if preferred_world_position is not None:
            preferred = np.asarray(preferred_world_position, np.float32)
            preferred_distance = float(np.linalg.norm(
                snapped[[0, 2]] - preferred[[0, 2]]))
        reachable.append((point, float(score), {
            "selected_point_world_xyz": np.asarray(world).tolist(),
            "selected_point_navmesh_xyz": snapped.tolist(),
            "selected_point_depth_m": float(selected_depth),
            "selected_point_initial_geodesic_m": float(
                shortest.geodesic_distance),
            "selected_point_target_planar_distance_m": preferred_distance,
            "selected_point_reachable": True,
        }))
    if reachable:
        if preferred_world_position is None:
            return max(reachable, key=lambda item: item[1])
        return min(reachable, key=lambda item: (
            item[2]["selected_point_target_planar_distance_m"], -item[1]))
    return None, 0.0, {
        "selected_point_world_xyz": None,
        "selected_point_navmesh_xyz": None,
        "selected_point_depth_m": None,
        "selected_point_initial_geodesic_m": None,
        "selected_point_target_planar_distance_m": None,
        "selected_point_reachable": False,
    }


def _cosine(left, right):
    left = np.asarray(left, np.float32)
    right = np.asarray(right, np.float32)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / max(denominator, 1e-8))


def _view_visual_similarity(current_rgb, target_rgb, embedder):
    """Orientation-aligned appearance score with auditable components."""
    compact = max(0.0, _cosine(
        embedder.embed_view(current_rgb), embedder.embed_view(target_rgb)))
    current_hsv = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2HSV)
    target_hsv = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2HSV)
    current_hist = cv2.calcHist([current_hsv], [0, 1], None, [24, 16],
                                [0, 180, 0, 256])
    target_hist = cv2.calcHist([target_hsv], [0, 1], None, [24, 16],
                               [0, 180, 0, 256])
    cv2.normalize(current_hist, current_hist)
    cv2.normalize(target_hist, target_hist)
    histogram = float(np.clip(
        cv2.compareHist(current_hist, target_hist, cv2.HISTCMP_CORREL), 0.0, 1.0))

    orb = cv2.ORB_create(nfeatures=240)
    current_gray = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2GRAY)
    current_keys, current_desc = orb.detectAndCompute(current_gray, None)
    target_keys, target_desc = orb.detectAndCompute(target_gray, None)
    local_features = 0.0
    if current_desc is not None and target_desc is not None:
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
            current_desc, target_desc, k=2)
        good = [first for pair in pairs if len(pair) == 2
                for first, second in [pair] if first.distance < 0.75 * second.distance]
        local_features = min(
            1.0, len(good) / max(8.0, 0.20 * min(len(current_keys), len(target_keys))))
    combined = 0.45 * compact + 0.25 * histogram + 0.30 * local_features
    return combined, {
        "compact_layout_similarity": compact,
        "hsv_histogram_similarity": histogram,
        "orb_local_feature_similarity": local_features,
    }


@dataclass
class BacktrackPointSelection:
    current_node_id: str
    target_node_id: str
    view_index: int
    point_xy: np.ndarray
    rgb: np.ndarray
    depth: np.ndarray
    target_mask: np.ndarray
    ground_mask: np.ndarray
    yaw: float
    candidates: list[dict]
    record: dict


@dataclass
class NodeRevisitMatchResult:
    reached: bool
    target_node_id: str
    current_node_id: str
    planar_distance_m: float
    visual_similarity: float
    semantic_similarity: float
    score: float
    reach_radius_m: float
    minimum_visual_similarity: float

    def to_dict(self):
        return asdict(self)


@dataclass
class NodeBacktrackResult:
    success: bool
    source_node_id: str
    target_node_id: str
    ancestor_route: list[str]
    completed_hops: int
    attempted_segments: int
    final_yaw: float
    next_global_step: int
    traveled_distance_m: float
    attempts: list[dict]
    end_reason: str

    def to_dict(self):
        return asdict(self)


class NodeRevisitMatcher:
    """Match a newly reached physical node to an older stored node."""

    def __init__(self, reach_radius_m=0.75, minimum_visual_similarity=0.75):
        self.reach_radius_m = float(reach_radius_m)
        self.minimum_visual_similarity = float(minimum_visual_similarity)

    @staticmethod
    def _labels(node):
        semantics = node.environment_semantics or {}
        return {str(label).lower() for label in semantics.get("labels", [])}

    def match(self, current_node, target_node):
        current_position = np.asarray(current_node.position_xyz, np.float32)
        target_position = np.asarray(target_node.position_xyz, np.float32)
        distance = float(np.linalg.norm(
            current_position[[0, 2]] - target_position[[0, 2]]))
        visual = _cosine(
            current_node.visual_embedding, target_node.visual_embedding)
        current_labels, target_labels = self._labels(current_node), self._labels(target_node)
        union = current_labels | target_labels
        semantic = (len(current_labels & target_labels) / len(union)
                    if union else 0.0)
        position_score = math.exp(-distance / max(self.reach_radius_m, 1e-6))
        score = 0.60 * position_score + 0.30 * max(0.0, visual) + 0.10 * semantic
        reached = bool(
            distance <= self.reach_radius_m and
            visual >= self.minimum_visual_similarity)
        return NodeRevisitMatchResult(
            reached=reached,
            target_node_id=target_node.node_id,
            current_node_id=current_node.node_id,
            planar_distance_m=round(distance, 6),
            visual_similarity=round(visual, 6),
            semantic_similarity=round(semantic, 6),
            score=round(score, 6),
            reach_radius_m=self.reach_radius_m,
            minimum_visual_similarity=self.minimum_visual_similarity,
        )


class NodeBacktrackingPointSelector:
    """Select current-view ground using two stored/current six-view panoramas."""

    def __init__(self, graph_memory, segmenter, vlm_harness=None,
                 mode="auto", visual_embedder=None,
                 planner_profile="legacy_direct", sim=None,
                 reference_path=None, reference_path_index=0):
        if mode not in {"auto", "hybrid", "vlm"}:
            raise ValueError("backtrack selector mode must be auto, hybrid, or vlm")
        if mode == "vlm" and vlm_harness is None:
            raise ValueError("vlm backtrack selection requires a VLM harness")
        self.graph_memory = graph_memory
        self.segmenter = segmenter
        self.vlm_harness = vlm_harness
        self.mode = ("vlm" if mode == "auto" and vlm_harness is not None
                     else "hybrid" if mode == "auto" else mode)
        self.visual_embedder = visual_embedder or CompactVisualEmbedder()
        self.sim = sim
        self.reference_path = reference_path
        self.reference_path_index = int(reference_path_index or 0)
        if planner_profile not in BACKTRACK_PLANNER_PROFILES:
            raise ValueError(
                f"unknown backtrack planner profile {planner_profile!r}")
        self.planner_profile = str(planner_profile)

    def select(self, current_node_id, target_node_id, current_rgbs, current_depths,
               current_position, current_yaw, forward_action_history=None,
               forward_edge=None, reference_source_node_id=None,
               maximum_route_progress_m=None, route_direction="reverse"):
        if route_direction not in {"reverse", "forward"}:
            raise ValueError("route_direction must be reverse or forward")
        if len(current_rgbs) != 6 or len(current_depths) != 6:
            raise ValueError("current node backtrack observation must contain six RGB-D views")
        target_node = self.graph_memory.get_node(target_node_id)
        target_views = self.graph_memory.load_node_views(target_node)
        target_view_records = sorted(
            target_node.six_views, key=lambda item: item["view_index"])
        target_position = np.asarray(target_node.position_xyz, np.float32)
        current_position = np.asarray(current_position, np.float32)
        target_route_geodesic = navmesh_route_distance(
            self.sim, current_position, target_position)
        delta = target_position - current_position
        direct_target_yaw = _wrap_angle(
            math.atan2(-float(delta[0]), -float(delta[2])))
        breadcrumb = None
        navmesh_route_waypoint = None
        if (self.planner_profile in {
                "breadcrumb_guard_v1", "breadcrumb_budget_v2",
                "breadcrumb_budget_v3"} and
                forward_edge is not None):
            edge_source = self.graph_memory.get_node(
                forward_edge.source_node_id)
            edge_target = self.graph_memory.get_node(
                forward_edge.target_node_id)
            if route_direction == "reverse":
                breadcrumb = reverse_route_breadcrumb(
                    forward_edge=forward_edge,
                    target_position=edge_source.position_xyz,
                    source_position=edge_target.position_xyz,
                    current_position=current_position,
                    lookahead_m=1.5,
                    maximum_progress_m=maximum_route_progress_m)
            else:
                breadcrumb = forward_route_breadcrumb(
                    forward_edge=forward_edge,
                    source_position=edge_source.position_xyz,
                    target_position=edge_target.position_xyz,
                    current_position=current_position,
                    lookahead_m=1.5,
                    minimum_progress_m=maximum_route_progress_m)
            breadcrumb_position = np.asarray(
                breadcrumb["position_xyz"], np.float32)
            breadcrumb_delta = breadcrumb_position - current_position
            desired_yaw = _wrap_angle(math.atan2(
                -float(breadcrumb_delta[0]), -float(breadcrumb_delta[2])))
        else:
            if self.planner_profile != "legacy_direct":
                navmesh_route_waypoint = navmesh_first_route_waypoint(
                    self.sim, current_position, target_position)
            if navmesh_route_waypoint is not None:
                route_delta = (np.asarray(navmesh_route_waypoint, np.float32) -
                               current_position)
                desired_yaw = _wrap_angle(math.atan2(
                    -float(route_delta[0]), -float(route_delta[2])))
            else:
                desired_yaw = direct_target_yaw

        strict_ground = bool(
            getattr(self.segmenter, "strict_ground_mask", False))
        segmentations = (self.segmenter.batch(current_rgbs)
                         if hasattr(self.segmenter, "batch") else
                         [self.segmenter(rgb) for rgb in current_rgbs])
        candidates = []
        for index, (offset, rgb, depth, segmentation) in enumerate(zip(
                PANORAMA_OFFSETS_RAD, current_rgbs, current_depths,
                segmentations)):
            candidate_yaw = _wrap_angle(float(current_yaw) + float(offset))
            ground_mask, detections = segmentation
            ground_mask = np.asarray(ground_mask, dtype=bool)
            target_mask = _targetable_ground_mask(ground_mask)
            if (not strict_ground and not target_mask.any() and
                    self.planner_profile != "legacy_direct"):
                # Side/rear recovery sectors often have a valid walkable lane
                # but an empty Grounded-SAM floor mask.  Expose only the
                # bounded lower-interior RGB floor prior; reachability and the
                # final live node match remain hard gates.  This keeps the
                # exact-node recovery directional without inventing semantic
                # evidence from a missing mask.
                from point_selectors import rgb_lower_floor_prior
                target_mask = rgb_lower_floor_prior(ground_mask.shape)
            if self.planner_profile != "legacy_direct":
                minimum_depth = 0.25
                if self.planner_profile == "breadcrumb_budget_v2":
                    route_remaining = (
                        float(breadcrumb["current_route_progress_m"]) +
                        float(breadcrumb["off_trace_distance_m"]))
                    if route_remaining > 1.25:
                        minimum_depth = 1.15
                target_mask = (
                    target_mask & np.isfinite(depth) &
                    (depth > minimum_depth))
            if strict_ground:
                target_mask &= ground_mask
            if self.sim is None:
                point, point_score = _select_ground_point(target_mask)
                reachability = {
                    "selected_point_world_xyz": None,
                    "selected_point_navmesh_xyz": None,
                    "selected_point_depth_m": None,
                    "selected_point_initial_geodesic_m": None,
                    "selected_point_target_planar_distance_m": None,
                    "selected_point_reachable": None,
                }
            else:
                preferred_position = (
                    np.asarray(breadcrumb["position_xyz"], np.float32)
                    if breadcrumb is not None else target_position)
                point, point_score, reachability = (
                    _select_reachable_ground_point(
                        self.sim, target_mask, depth, current_position,
                        candidate_yaw,
                        preferred_world_position=preferred_position))
            target_view_index = min(
                range(6), key=lambda target_index: abs(_wrap_angle(
                    float(target_view_records[target_index]["absolute_yaw_rad"]) -
                    candidate_yaw)))
            visual_similarity, visual_components = _view_visual_similarity(
                rgb, target_views[target_view_index], self.visual_embedder)
            yaw_error = abs(_wrap_angle(candidate_yaw - desired_yaw))
            direction_score = 0.5 * (math.cos(yaw_error) + 1.0)
            backtrack_allowed = bool(yaw_error <= math.radians(100.0))
            ground_fraction = float(target_mask.mean())
            viability = min(1.0, point_score / max(rgb.shape[:2])) if point is not None else 0.0
            if self.planner_profile == "legacy_direct":
                score = (0.30 * direction_score +
                         0.50 * max(0.0, visual_similarity) +
                         0.10 * min(1.0, ground_fraction * 8.0) +
                         0.10 * viability)
            else:
                score = (0.62 * direction_score +
                         0.23 * max(0.0, visual_similarity) +
                         0.10 * min(1.0, ground_fraction * 8.0) +
                         0.05 * viability)
            candidates.append({
                "view_index": index, "rgb": rgb, "depth": depth,
                "yaw": candidate_yaw, "relative_yaw_rad": float(offset),
                "target_mask": target_mask, "ground_mask": ground_mask,
                "strict_ground_mask": strict_ground,
                "ground_fraction": ground_fraction,
                "ground_detection_records": [item.prompt_record()
                                             for item in detections],
                "point": point, "point_score": point_score,
                "visual_similarity": visual_similarity,
                "visual_similarity_components": visual_components,
                "aligned_target_view_index": target_view_index,
                "direction_score": direction_score,
                "desired_yaw_rad": desired_yaw, "yaw_error_rad": yaw_error,
                "backtrack_allowed": backtrack_allowed,
                "hybrid_score": score,
                **reachability,
            })
        usable = [candidate for candidate in candidates if candidate["point"] is not None]
        if not usable:
            raise RuntimeError("no segmented ground candidate for node backtracking")
        gated = [candidate for candidate in usable
                 if candidate["backtrack_allowed"]]
        gate_relaxed = not bool(gated)
        eligible = gated or [min(usable, key=lambda item: item["yaw_error_rad"])]
        recommended, short_range_proximity_priority = choose_recovery_candidate(
            eligible, target_route_geodesic)

        if self.mode == "vlm":
            view_index, point, selection_record = (
                self.vlm_harness.select_backtrack_ground_target(
                    target_node_id=target_node_id,
                    target_node_views=target_views,
                    current_node_id=current_node_id,
                    candidates=candidates,
                    forward_action_history=forward_action_history,
                    recommended_view=recommended["view_index"]))
            chosen = candidates[view_index]
            # The VLM returns an image anchor independently of the dense
            # floor-candidate projection used for navmesh validation.  An
            # anchor can therefore land on a visually plausible but
            # disconnected pixel (e.g. a rug edge, threshold, or furniture
            # gap) even though the selected panorama has a reachable floor
            # candidate.  In that generic case, keep the VLM's view decision
            # but replace only the unsafe pixel with the already validated
            # reachable floor anchor from that same view.  This prevents the
            # executor from failing before motion while preserving the
            # two-node visual/directional selection semantics.
            if (chosen.get("selected_point_reachable") is True and
                    chosen.get("point") is not None):
                proposed = np.asarray(point, np.float32)
                validated = np.asarray(chosen["point"], np.float32)
                if proposed.shape != validated.shape or not np.allclose(
                        proposed, validated, atol=1.0):
                    selection_record["vlm_anchor_xy"] = proposed.tolist()
                    selection_record["validated_floor_anchor_xy"] = (
                        validated.tolist())
                    selection_record["vlm_anchor_replaced_unreachable"] = True
                    point = validated
        else:
            chosen = recommended
            point = np.asarray(chosen["point"], np.float32)
            selection_record = {
                "selector": "hybrid_two_node_six_view",
                "target_node_id": str(target_node_id),
                "current_node_id": str(current_node_id),
                "reason": (
                    "orientation-aligned panorama appearance + graph bearing + "
                    "DINO+SAM floor"),
                "recommended_view": int(chosen["view_index"]),
            }
        if self.reference_path:
            camera_position = (current_position +
                               np.array([0.0, 1.25, 0.0], np.float32))
            for candidate in candidates:
                candidate["reference_path_projection"] = project_reference_path(
                    self.reference_path,
                    range(self.reference_path_index, len(self.reference_path)),
                    camera_position, float(candidate["yaw"]),
                    candidate["rgb"].shape[1], candidate["rgb"].shape[0])
        selection_record.update({
            "planner_profile": self.planner_profile,
            "route_direction": route_direction,
            "selected_view": int(chosen["view_index"]),
            "selected_point_xy": np.asarray(point).tolist(),
            "selected_yaw_rad": float(chosen["yaw"]),
            "desired_target_bearing_yaw_rad": direct_target_yaw,
            "desired_breadcrumb_bearing_yaw_rad": desired_yaw,
            "navmesh_first_route_waypoint_xyz": (
                np.asarray(navmesh_route_waypoint).tolist()
                if navmesh_route_waypoint is not None else None),
            "current_target_route_geodesic_m": (
                target_route_geodesic
                if math.isfinite(target_route_geodesic) else None),
            "short_range_node_proximity_priority": (
                short_range_proximity_priority),
            "direction_gate_relaxed": gate_relaxed,
            "selected_direction_allowed": bool(chosen["backtrack_allowed"]),
            "breadcrumb": ({
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in breadcrumb.items()
            } if breadcrumb is not None else None),
            "forward_action_history_reversed_for_context": list(
                reversed(forward_action_history or [])),
            "reference_path_projection_by_view": ({
                str(candidate["view_index"]): candidate[
                    "reference_path_projection"]
                for candidate in candidates
                if "reference_path_projection" in candidate
            } if self.reference_path else None),
            "candidate_scores": [{
                "view_index": candidate["view_index"],
                "yaw_rad": candidate["yaw"],
                "visual_similarity": candidate["visual_similarity"],
                "visual_similarity_components": candidate[
                    "visual_similarity_components"],
                "aligned_target_view_index": candidate[
                    "aligned_target_view_index"],
                "direction_score": candidate["direction_score"],
                "yaw_error_deg": math.degrees(candidate["yaw_error_rad"]),
                "backtrack_allowed": candidate["backtrack_allowed"],
                "ground_fraction": candidate["ground_fraction"],
                "hybrid_score": candidate["hybrid_score"],
                "has_ground_point": candidate["point"] is not None,
                "selected_point_reachable": candidate[
                    "selected_point_reachable"],
                "selected_point_initial_geodesic_m": candidate[
                    "selected_point_initial_geodesic_m"],
                "selected_point_target_planar_distance_m": candidate[
                    "selected_point_target_planar_distance_m"],
            } for candidate in candidates],
        })
        return BacktrackPointSelection(
            current_node_id=str(current_node_id), target_node_id=str(target_node_id),
            view_index=int(chosen["view_index"]), point_xy=np.asarray(point, np.float32),
            rgb=chosen["rgb"], depth=chosen["depth"],
            target_mask=chosen["target_mask"],
            ground_mask=chosen["ground_mask"], yaw=float(chosen["yaw"]),
            candidates=candidates, record=selection_record)


class NodeBacktrackingController:
    """Walk an ancestor route by repeatedly selecting and executing point goals."""

    def __init__(
            self, sim, graph_memory, segmenter, point_navigation_executor,
            position_history, rendered, video_composer, motion_log,
            vlm_harness=None, selector_mode="auto", scan_step_deg=10.0,
            max_attempts_per_hop=2, reach_radius_m=0.75,
            minimum_visual_similarity=0.75, max_hops=5,
            output_dir=None, planner_profile="legacy_direct",
            reference_path=None, reference_path_index=0):
        self.sim = sim
        self.graph_memory = graph_memory
        self.point_navigation_executor = point_navigation_executor
        self.position_history = position_history
        self.rendered = rendered
        self.video_composer = video_composer
        self.motion_log = motion_log
        self.selector = NodeBacktrackingPointSelector(
            graph_memory, segmenter, vlm_harness, selector_mode,
            planner_profile=planner_profile, sim=sim,
            reference_path=reference_path,
            reference_path_index=reference_path_index)
        self.planner_profile = str(planner_profile)
        self.matcher = NodeRevisitMatcher(
            reach_radius_m, minimum_visual_similarity)
        self.scan_step = math.radians(float(scan_step_deg))
        self.max_attempts_per_hop = int(max_attempts_per_hop)
        self.max_hops = int(max_hops)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        if self.output_dir is not None:
            self.backtrack_dir = self.output_dir / "node_backtracking"
            self.backtrack_dir.mkdir(parents=True, exist_ok=True)

    def _visualize_selection(self, selection, position, segment_index):
        target_views = self.graph_memory.load_node_views(selection.target_node_id)
        instruction = f"BACKTRACK to {selection.target_node_id}"
        target_images = []
        for index, rgb in enumerate(target_views):
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(frame, f"TARGET NODE REF VIEW {index}", (7, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2)
            target_images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.rendered.append(self.video_composer.compose(
                frame, self.position_history, position, selection.yaw,
                instruction, segment_index, "backtrack_target_reference"))
        decision_images = []
        for candidate in selection.candidates:
            rgb = candidate["rgb"].copy()
            green = np.zeros_like(rgb); green[..., 1] = 255
            mask = candidate["target_mask"]
            rgb[mask] = (0.55 * rgb[mask] + 0.45 * green[mask]).astype(np.uint8)
            projection = candidate.get("reference_path_projection")
            if projection is not None:
                rgb = draw_reference_path_overlay(
                    rgb, projection,
                    selected_point=(selection.point_xy
                                    if candidate["view_index"] == selection.view_index
                                    else None),
                    selected=bool(candidate["view_index"] == selection.view_index),
                    next_path_index=self.selector.reference_path_index + 1)
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(
                frame,
                f"CURRENT VIEW {candidate['view_index']} vis "
                f"{candidate['visual_similarity']:.2f} dir "
                f"{candidate['direction_score']:.2f}",
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
            if candidate["view_index"] == selection.view_index:
                cv2.drawMarker(
                    frame, tuple(np.round(selection.point_xy).astype(int)),
                    (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
                cv2.putText(frame, "BACKTRACK POINT", (6, 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
            decision_images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.rendered.extend([self.video_composer.compose(
                frame, self.position_history, position, candidate["yaw"],
                instruction, segment_index, "backtrack_current_selection")] * 2)
        if self.output_dir is not None:
            target_sheet = np.concatenate([
                np.concatenate(target_images[:3], axis=1),
                np.concatenate(target_images[3:], axis=1),
            ], axis=0)
            current_sheet = np.concatenate([
                np.concatenate(decision_images[:3], axis=1),
                np.concatenate(decision_images[3:], axis=1),
            ], axis=0)
            Image.fromarray(np.concatenate([target_sheet, current_sheet], axis=0)).save(
                self.backtrack_dir /
                f"segment_{segment_index:03d}_two_node_contact_sheet.jpg")

    def _resolve_target(self, source_node_id, target_node_id):
        if target_node_id == "previous":
            predecessor, _ = self.graph_memory.predecessor(source_node_id)
            if predecessor is None:
                raise ValueError(f"node {source_node_id} has no predecessor")
            return predecessor.node_id
        return str(target_node_id)

    def _shortest_graph_route(self, source_node_id, target_node_id):
        """Return an undirected graph route, preferring zero-cost closures."""
        source_node_id = str(source_node_id)
        target_node_id = str(target_node_id)
        adjacency = {node.node_id: [] for node in self.graph_memory.nodes}
        for edge in self.graph_memory.edges:
            weight = (0.001 if edge.edge_kind == "node_revisit_loop_closure"
                      else max(0.05, float(edge.traveled_distance_m)))
            adjacency.setdefault(edge.source_node_id, []).append(
                (edge.target_node_id, weight, edge.edge_id))
            adjacency.setdefault(edge.target_node_id, []).append(
                (edge.source_node_id, weight, edge.edge_id))
        queue = [(0.0, source_node_id)]
        distance = {source_node_id: 0.0}
        previous = {}
        while queue:
            cost, node_id = heapq.heappop(queue)
            if cost > distance.get(node_id, math.inf) + 1e-9:
                continue
            if node_id == target_node_id:
                break
            for neighbor, weight, edge_id in adjacency.get(node_id, []):
                next_cost = cost + weight
                if next_cost + 1e-9 < distance.get(neighbor, math.inf):
                    distance[neighbor] = next_cost
                    previous[neighbor] = (node_id, edge_id)
                    heapq.heappush(queue, (next_cost, neighbor))
        if target_node_id not in distance:
            raise ValueError(
                f"no graph route {source_node_id!r} -> {target_node_id!r}")
        nodes = [target_node_id]
        edge_ids = []
        while nodes[-1] != source_node_id:
            predecessor, edge_id = previous[nodes[-1]]
            edge_ids.append(edge_id)
            nodes.append(predecessor)
        nodes.reverse()
        edge_ids.reverse()
        if len(nodes) - 1 > self.max_hops:
            raise ValueError(
                f"graph route exceeds max_hops={self.max_hops}: {nodes}")
        return nodes, edge_ids

    def recover_live_to_verified_node(self, target_node_id, current_yaw=0.0,
                                      global_step=0, target_index_offset=0,
                                      max_attempts=None):
        """Physically return a displaced live agent to a verified node.

        A point-navigation attempt can emit a terminal signal after losing its
        cluster while still outside the selected endpoint.  Such an attempt
        is deliberately not persisted as a graph node.  The ordinary
        ``backtrack`` method starts from the graph's latest node and therefore
        cannot represent this unverified live pose.  This helper uses the same
        two-node selector and point executor from the live pose, verifies the
        stored target with planar distance plus panorama embedding, and keeps
        every recovery attempt diagnostic-only.  It is generic recovery, not
        an episode-specific reset or a simulator teleport.
        """
        from point_navigation_executor import (
            PointNavigationRequest, execute_point_navigation)
        from point_selectors import continuous_turn, pixel_ground_to_world

        target_node_id = str(target_node_id)
        target_node = self.graph_memory.get_node(target_node_id)
        attempts = []
        total_distance = 0.0
        yaw = float(current_yaw)
        step = int(global_step)
        maximum = (self.max_attempts_per_hop if max_attempts is None else
                   max(1, int(max_attempts)))

        def live_match(rgbs, position):
            embedding = self.selector.visual_embedder.embed(rgbs)
            distance = float(np.linalg.norm(
                np.asarray(position, np.float32)[[0, 2]] -
                np.asarray(target_node.position_xyz, np.float32)[[0, 2]]))
            visual = max(0.0, _cosine(embedding, target_node.visual_embedding))
            return {
                "reached": bool(
                    distance <= self.matcher.reach_radius_m and
                    visual >= self.matcher.minimum_visual_similarity),
                "target_node_id": target_node_id,
                "planar_distance_m": distance,
                "visual_similarity": visual,
                "reach_radius_m": self.matcher.reach_radius_m,
                "minimum_visual_similarity": self.matcher.minimum_visual_similarity,
            }

        for attempt_index in range(maximum):
            position = np.asarray(
                self.sim.get_agent(0).get_state().position, np.float32)
            current_rgbs, current_depths = _observe_six_rgbd(self.sim)
            revisit = live_match(current_rgbs, position)
            if revisit["reached"]:
                attempts.append({
                    "attempt_index": attempt_index,
                    "reference_target_node_id": target_node_id,
                    "executor_arrived": None,
                    "executor_end_reason": "already_at_verified_node",
                    "executed_point_navigation": False,
                    "action_history": [],
                    "node_created": False,
                    "node_revisit_match": revisit,
                })
                return NodeBacktrackResult(
                    success=True, source_node_id="live_unverified",
                    target_node_id=target_node_id,
                    ancestor_route=[target_node_id], completed_hops=1,
                    attempted_segments=0, final_yaw=yaw,
                    next_global_step=step, traveled_distance_m=total_distance,
                    attempts=attempts, end_reason="target_node_reached")

            try:
                # A live recovery target is an already verified node.  Use the
                # existing geometry+panorama hybrid ranking here so a VLM
                # cannot replace the exact node bearing with a visually
                # similar side branch; forward semantic selection remains VLM
                # driven.  This is still RGB/floor based and does not expose
                # hidden future waypoints.
                original_selector_mode = self.selector.mode
                self.selector.mode = "hybrid"
                try:
                    selection = self.selector.select(
                        current_node_id="live_unverified",
                        target_node_id=target_node_id,
                        current_rgbs=current_rgbs, current_depths=current_depths,
                        current_position=position, current_yaw=yaw,
                        forward_action_history=None, forward_edge=None,
                        route_direction="reverse")
                finally:
                    self.selector.mode = original_selector_mode
                selection.record["live_recovery_selector_mode"] = "hybrid"
            except Exception as exc:
                attempts.append({
                    "attempt_index": attempt_index,
                    "reference_target_node_id": target_node_id,
                    "selection_error": str(exc),
                    "executor_arrived": None,
                    "executor_end_reason": "live_recovery_selection_failed",
                    "executed_point_navigation": False,
                    "action_history": [], "node_created": False,
                    "node_revisit_match": revisit,
                })
                return NodeBacktrackResult(
                    success=False, source_node_id="live_unverified",
                    target_node_id=target_node_id,
                    ancestor_route=[target_node_id], completed_hops=0,
                    attempted_segments=0, final_yaw=yaw,
                    next_global_step=step, traveled_distance_m=total_distance,
                    attempts=attempts,
                    end_reason=f"live_recovery_selection_failed: {exc}")

            segment_index = attempt_index
            self._visualize_selection(selection, position, segment_index)
            yaw, chosen_rgb = continuous_turn(
                self.sim, position, yaw, selection.yaw, self.scan_step,
                self.rendered, self.motion_log,
                target_index_offset + segment_index,
                "live_recovery_turn", self.video_composer,
                self.position_history,
                f"RECOVER to verified node {target_node_id}",
                reference_path=self.selector.reference_path,
                reference_path_index=self.selector.reference_path_index)
            selected_world, selected_depth = pixel_ground_to_world(
                selection.point_xy, selection.depth, position, selection.yaw)
            selected_navmesh = None
            selected_reachable = False
            selected_geodesic = math.inf
            if selected_world is not None:
                snapped = np.asarray(
                    self.sim.pathfinder.snap_point(selected_world), np.float32)
                if np.isfinite(snapped).all():
                    selected_navmesh = snapped
                    shortest = __import__("habitat_sim").ShortestPath()
                    shortest.requested_start = position
                    shortest.requested_end = selected_navmesh
                    if self.sim.pathfinder.find_path(shortest):
                        selected_reachable = True
                        selected_geodesic = float(shortest.geodesic_distance)
            selection.record.update({
                "selected_point_world_xyz": (
                    np.asarray(selected_world).tolist()
                    if selected_world is not None else None),
                "selected_point_navmesh_xyz": (
                    selected_navmesh.tolist()
                    if selected_navmesh is not None else None),
                "selected_point_reachable": selected_reachable,
                "selected_point_initial_geodesic_m": (
                    selected_geodesic if math.isfinite(selected_geodesic)
                    else None),
            })
            # The VLM/ground anchor is a visual breadcrumb and can lie beyond
            # the exact stored node position.  Bound this live recovery
            # segment by the shortest path to the verified node (plus a small
            # controller margin), so a valid return cannot overshoot the node
            # merely because the selected floor pixel was farther along the
            # same corridor.  This geometry is used only by the recovery
            # executor after the RGB selection is frozen; it is not semantic
            # input to the VLM.
            target_route_geodesic = math.inf
            try:
                target_path = __import__("habitat_sim").ShortestPath()
                target_path.requested_start = position
                target_path.requested_end = np.asarray(
                    target_node.position_xyz, np.float32)
                if self.sim.pathfinder.find_path(target_path):
                    target_route_geodesic = float(target_path.geodesic_distance)
            except Exception:
                target_route_geodesic = math.inf
            recovery_travel_budget = (
                max(0.60, target_route_geodesic + 0.35)
                if math.isfinite(target_route_geodesic) else None)
            executor_navmesh, executor_geodesic, endpoint_source = (
                recovery_executor_endpoint(
                    self.planner_profile, selection.record,
                    selected_navmesh, selected_geodesic,
                    target_node.position_xyz, target_route_geodesic))
            selection.record["verified_node_route_geodesic_m"] = (
                target_route_geodesic if math.isfinite(target_route_geodesic)
                else None)
            selection.record["verified_node_recovery_travel_budget_m"] = (
                recovery_travel_budget)
            selection.record["executor_endpoint_source"] = endpoint_source
            selection.record["executor_endpoint_navmesh_xyz"] = (
                np.asarray(executor_navmesh).tolist()
                if executor_navmesh is not None else None)
            selection.record["executor_endpoint_initial_geodesic_m"] = (
                float(executor_geodesic)
                if math.isfinite(float(executor_geodesic)) else None)
            navigation_result = execute_point_navigation(
                self.point_navigation_executor,
                PointNavigationRequest(
                    rgb=chosen_rgb, selected_point_xy=selection.point_xy,
                    selectable_mask=selection.target_mask,
                    ground_mask=selection.ground_mask, yaw=yaw,
                    position_history=self.position_history,
                    instruction=f"recover to verified node {target_node_id}",
                    semantic_target="stored verified node floor",
                    target_index=target_index_offset + segment_index,
                    stage_count=target_index_offset + maximum + 1,
                    global_step=step,
                    selected_point_depth_m=(
                        float(selection.depth[int(round(selection.point_xy[1])),
                                              int(round(selection.point_xy[0]))])
                        if selection.depth is not None else None),
                    selected_point_reachable=bool(
                        executor_navmesh is not None and
                        math.isfinite(float(executor_geodesic))),
                    selected_point_initial_geodesic_m=(
                        float(executor_geodesic)
                        if math.isfinite(float(executor_geodesic)) else None),
                    selected_point_navmesh_xyz=executor_navmesh,
                    max_travel_distance_m=recovery_travel_budget,
                    reference_path=self.selector.reference_path,
                    reference_path_index=self.selector.reference_path_index))
            yaw = navigation_result.final_yaw
            step = navigation_result.next_global_step
            action_history = navigation_result.action_history
            moved = sum(float(item.get("moved_m", 0.0) or 0.0)
                        for item in action_history)
            total_distance += moved
            stop_position = np.asarray(
                self.sim.get_agent(0).get_state().position, np.float32)
            stop_rgbs, _ = _observe_six_rgbd(self.sim)
            final_geodesic = math.inf
            if executor_navmesh is not None:
                shortest = __import__("habitat_sim").ShortestPath()
                shortest.requested_start = stop_position
                shortest.requested_end = executor_navmesh
                if self.sim.pathfinder.find_path(shortest):
                    final_geodesic = float(shortest.geodesic_distance)
            physical_point_arrival = bool(
                navigation_result.arrived and final_geodesic <= 0.75)
            revisit = live_match(stop_rgbs, stop_position)
            attempt_record = {
                "attempt_index": attempt_index,
                "reference_target_node_id": target_node_id,
                "selection": selection.record,
                "executor_arrived": navigation_result.arrived,
                "executor_signal": navigation_result.signal,
                "executor_end_reason": navigation_result.end_reason,
                "executed_point_navigation": True,
                "action_history": action_history,
                "selected_point_final_geodesic_m": (
                    final_geodesic if math.isfinite(final_geodesic) else None),
                "physical_point_arrival": physical_point_arrival,
                "node_created": False,
                "node_revisit_match": revisit,
            }
            attempts.append(attempt_record)
            if revisit["reached"]:
                return NodeBacktrackResult(
                    success=True, source_node_id="live_unverified",
                    target_node_id=target_node_id,
                    ancestor_route=[target_node_id], completed_hops=1,
                    attempted_segments=attempt_index + 1,
                    final_yaw=yaw, next_global_step=step,
                    traveled_distance_m=total_distance,
                    attempts=attempts, end_reason="target_node_reached")

        return NodeBacktrackResult(
            success=False, source_node_id="live_unverified",
            target_node_id=target_node_id, ancestor_route=[target_node_id],
            completed_hops=0, attempted_segments=len(attempts),
            final_yaw=yaw, next_global_step=step,
            traveled_distance_m=total_distance, attempts=attempts,
            end_reason=f"live_recovery_failed_to_revisit_{target_node_id}")

    def backtrack(self, target_node_id="previous", source_node_id=None,
                  current_yaw=0.0, global_step=0, target_index_offset=0):
        # Lazy imports keep the selector/matcher usable in lightweight graph
        # tests that do not install Habitat-Sim.
        from point_navigation_executor import (
            PointNavigationRequest, execute_point_navigation)
        from point_selectors import continuous_turn, pixel_ground_to_world

        source_node_id = (str(source_node_id) if source_node_id is not None
                          else self.graph_memory.nodes[-1].node_id)
        target_node_id = self._resolve_target(source_node_id, str(target_node_id))
        current_node = self.graph_memory.get_node(source_node_id)
        target_node = self.graph_memory.get_node(target_node_id)
        initial_match = self.matcher.match(current_node, target_node)
        if initial_match.reached and source_node_id != target_node_id:
            closure = self.graph_memory.add_loop_closure_edge(
                target_node_id, source_node_id,
                metadata={
                    "node_revisit_match": initial_match.to_dict(),
                    "zero_motion_graph_route": True,
                })
            return NodeBacktrackResult(
                success=True, source_node_id=source_node_id,
                target_node_id=target_node_id,
                ancestor_route=[source_node_id, target_node_id],
                completed_hops=1, attempted_segments=0,
                final_yaw=float(current_yaw),
                next_global_step=int(global_step), traveled_distance_m=0.0,
                attempts=[{
                    "segment_index": None, "hop_index": 0,
                    "hop_attempt": -1,
                    "reference_source_node_id": source_node_id,
                    "reference_target_node_id": target_node_id,
                    "reference_edge_id": closure.edge_id,
                    "route_direction": "loop_closure",
                    "created_node_id": source_node_id,
                    "created_edge_id": None,
                    "loop_closure_edge_id": closure.edge_id,
                    "selection": None, "executor_arrived": None,
                    "executor_end_reason": "already_at_target_node",
                    "action_history": [],
                    "node_revisit_match": initial_match.to_dict(),
                    "executed_point_navigation": False,
                }], end_reason="target_node_reached")
        route, route_edge_ids = self._shortest_graph_route(
            source_node_id, target_node_id)
        attempts, completed_hops = [], 0
        executed_segments = 0
        total_distance = 0.0
        yaw = float(current_yaw)
        step = int(global_step)
        segment_index = 0
        end_reason = "target_node_reached"

        edge_by_id = {edge.edge_id: edge for edge in self.graph_memory.edges}
        for reference_source_id, reference_target_id, reference_edge_id in zip(
                route[:-1], route[1:], route_edge_ids):
            forward_edge = edge_by_id[reference_edge_id]
            if (forward_edge.source_node_id == reference_source_id and
                    forward_edge.target_node_id == reference_target_id):
                route_direction = "forward"
            elif (forward_edge.target_node_id == reference_source_id and
                  forward_edge.source_node_id == reference_target_id):
                route_direction = "reverse"
            else:
                raise RuntimeError(
                    f"edge {reference_edge_id} is not on route hop "
                    f"{reference_source_id}->{reference_target_id}")
            current_node = self.graph_memory.nodes[-1]
            pre_revisit = self.matcher.match(
                current_node, self.graph_memory.get_node(reference_target_id))
            if pre_revisit.reached:
                closure = self.graph_memory.add_loop_closure_edge(
                    reference_target_id, current_node.node_id,
                    metadata={
                        "node_revisit_match": pre_revisit.to_dict(),
                        "zero_motion_backtrack_hop": True,
                    })
                self.graph_memory.set_node_metadata(current_node.node_id, {
                    "node_revisit_match": pre_revisit.to_dict(),
                    "loop_closure_edge_id": closure.edge_id,
                })
                attempts.append({
                    "segment_index": None,
                    "hop_index": completed_hops,
                    "hop_attempt": -1,
                    "reference_source_node_id": reference_source_id,
                    "reference_target_node_id": reference_target_id,
                    "reference_edge_id": reference_edge_id,
                    "route_direction": route_direction,
                    "created_node_id": current_node.node_id,
                    "created_edge_id": None,
                    "loop_closure_edge_id": closure.edge_id,
                    "selection": None,
                    "executor_arrived": None,
                    "executor_end_reason": "already_at_reference_node",
                    "action_history": [],
                    "node_revisit_match": pre_revisit.to_dict(),
                    "executed_point_navigation": False,
                })
                completed_hops += 1
                continue
            hop_reached = False
            maximum_route_progress_m = None
            for hop_attempt in range(self.max_attempts_per_hop):
                current_node = self.graph_memory.nodes[-1]
                position = np.asarray(
                    self.sim.get_agent(0).get_state().position, np.float32)
                current_rgbs, current_depths = _observe_six_rgbd(self.sim)
                try:
                    selection = self.selector.select(
                        current_node_id=current_node.node_id,
                        target_node_id=reference_target_id,
                        current_rgbs=current_rgbs, current_depths=current_depths,
                        current_position=position, current_yaw=yaw,
                        forward_action_history=forward_edge.action_history,
                        forward_edge=forward_edge,
                        reference_source_node_id=reference_source_id,
                        maximum_route_progress_m=maximum_route_progress_m,
                        route_direction=route_direction)
                except RuntimeError as exc:
                    # A malformed/exhausted VLM response is a recoverable
                    # backtrack failure, not a reason to lose the episode's
                    # trajectory, graph, metrics, and diagnostic video.
                    attempts.append({
                        "segment_index": segment_index,
                        "hop_index": completed_hops,
                        "hop_attempt": hop_attempt,
                        "reference_source_node_id": reference_source_id,
                        "reference_target_node_id": reference_target_id,
                        "reference_edge_id": reference_edge_id,
                        "route_direction": route_direction,
                        "created_node_id": None,
                        "created_edge_id": None,
                        "loop_closure_edge_id": None,
                        "selection": None,
                        "selection_error": str(exc),
                        "executor_arrived": None,
                        "executor_end_reason": "backtrack_selection_failed",
                        "executed_point_navigation": False,
                        "action_history": [],
                        "node_revisit_match": None,
                    })
                    return NodeBacktrackResult(
                        success=False, source_node_id=source_node_id,
                        target_node_id=target_node_id, ancestor_route=route,
                        completed_hops=completed_hops,
                        attempted_segments=executed_segments,
                        final_yaw=yaw, next_global_step=step,
                        traveled_distance_m=total_distance, attempts=attempts,
                        end_reason=f"backtrack_selection_failed: {exc}")
                self._visualize_selection(selection, position, segment_index)
                breadcrumb = selection.record.get("breadcrumb") or {}
                if breadcrumb.get("current_route_progress_m") is not None:
                    maximum_route_progress_m = float(
                        breadcrumb["current_route_progress_m"])
                travel_budget_m = None
                if (self.planner_profile in {
                        "breadcrumb_budget_v2", "breadcrumb_budget_v3",
                        "breadcrumb_endpoint_v4", "breadcrumb_endpoint_v5"} and
                        breadcrumb):
                    travel_budget_m = backtrack_segment_travel_budget(
                        breadcrumb)
                    selection.record["route_segment_travel_budget_m"] = (
                        travel_budget_m)
                    selection.record[
                        "route_segment_controller_margin_m"] = 0.32
                yaw, chosen_rgb = continuous_turn(
                    self.sim, position, yaw, selection.yaw, self.scan_step,
                    self.rendered, self.motion_log,
                    target_index_offset + segment_index,
                    "turn_to_backtrack_point", self.video_composer,
                    self.position_history,
                    f"BACKTRACK to {reference_target_id}",
                    reference_path=self.selector.reference_path,
                    reference_path_index=self.selector.reference_path_index)

                selected_world, selected_depth = pixel_ground_to_world(
                    selection.point_xy, selection.depth, position,
                    selection.yaw)
                selected_navmesh = None
                selected_reachable = False
                selected_geodesic = math.inf
                if selected_world is not None:
                    snapped = np.asarray(
                        self.sim.pathfinder.snap_point(selected_world),
                        np.float32)
                    if np.isfinite(snapped).all():
                        selected_navmesh = snapped
                        shortest = __import__("habitat_sim").ShortestPath()
                        shortest.requested_start = position
                        shortest.requested_end = selected_navmesh
                        if self.sim.pathfinder.find_path(shortest):
                            selected_reachable = True
                            selected_geodesic = float(
                                shortest.geodesic_distance)
                selection.record.update({
                    "selected_point_depth_m": (
                        float(selected_depth)
                        if math.isfinite(float(selected_depth)) else None),
                    "selected_point_world_xyz": (
                        np.asarray(selected_world).tolist()
                        if selected_world is not None else None),
                    "selected_point_navmesh_xyz": (
                        selected_navmesh.tolist()
                        if selected_navmesh is not None else None),
                    "selected_point_reachable": selected_reachable,
                    "selected_point_initial_geodesic_m": (
                        selected_geodesic if math.isfinite(selected_geodesic)
                        else None),
                })
                executor_navmesh, executor_geodesic, endpoint_source = (
                    recovery_executor_endpoint(
                        self.planner_profile, selection.record,
                        selected_navmesh, selected_geodesic,
                        self.graph_memory.get_node(
                            reference_target_id).position_xyz,
                        selection.record.get(
                            "current_target_route_geodesic_m", math.inf)))
                selection.record.update({
                    "executor_endpoint_source": endpoint_source,
                    "executor_endpoint_navmesh_xyz": (
                        np.asarray(executor_navmesh).tolist()
                        if executor_navmesh is not None else None),
                    "executor_endpoint_initial_geodesic_m": (
                        float(executor_geodesic)
                        if math.isfinite(float(executor_geodesic)) else None),
                })

                sub_instruction = SubInstruction.from_mapping({
                    "sub_instruction_id": 900000 + segment_index,
                    "navigation_instruction": (
                        f"backtrack to stored node {reference_target_id}"),
                    "landmark": f"stored panorama of {reference_target_id}",
                    "completion_cue": "current node matches the stored target node",
                    "semantic_spatial_target": (
                        f"walkable floor returning to {reference_target_id}"),
                    "spatial_relation": "at the previously visited node",
                    "visual_arrival_evidence": (
                        "six-view panorama matches the stored target panorama"),
                    "forbidden_target": "new branches not present on the reverse route",
                    "form": "NODE_BACKTRACK",
                    "metadata": {
                        "reference_source_node_id": reference_source_id,
                        "reference_target_node_id": reference_target_id,
                        "reference_edge_id": reference_edge_id,
                        "route_direction": route_direction,
                    },
                })
                navigation_result = execute_point_navigation(
                    self.point_navigation_executor,
                    PointNavigationRequest(
                        rgb=chosen_rgb,
                        selected_point_xy=selection.point_xy,
                        selectable_mask=selection.target_mask,
                        ground_mask=selection.ground_mask,
                        yaw=yaw, position_history=self.position_history,
                        instruction=sub_instruction.navigation_instruction,
                        semantic_target=sub_instruction.semantic_spatial_target,
                        target_index=target_index_offset + segment_index,
                        stage_count=target_index_offset + len(route) + 1,
                        global_step=step,
                        selected_point_depth_m=(
                            float(selected_depth)
                            if math.isfinite(float(selected_depth)) else None),
                        selected_point_reachable=bool(
                            executor_navmesh is not None and
                            math.isfinite(float(executor_geodesic))),
                        selected_point_initial_geodesic_m=(
                            float(executor_geodesic)
                            if math.isfinite(float(executor_geodesic)) else None),
                        selected_point_navmesh_xyz=executor_navmesh,
                        max_travel_distance_m=travel_budget_m,
                        allow_initial_near_field_arrival=(
                            self.planner_profile not in {
                                "breadcrumb_budget_v3",
                                "breadcrumb_endpoint_v4",
                                "breadcrumb_endpoint_v5"}),
                        reference_path=self.selector.reference_path,
                        reference_path_index=self.selector.reference_path_index,
                    ))
                yaw = navigation_result.final_yaw
                step = navigation_result.next_global_step
                segment_distance = sum(
                    float(action.get("moved_m", 0.0))
                    for action in navigation_result.action_history)
                total_distance += segment_distance
                stop_position = np.asarray(
                    self.sim.get_agent(0).get_state().position, np.float32)
                stop_rgbs, stop_depths = _observe_six_rgbd(self.sim)
                stop_completion_views = _observe_eight_completion_rgb(self.sim)
                stop_node, stop_edge = self.graph_memory.add_navigation_stop_node(
                    position_xyz=stop_position, base_yaw_rad=yaw,
                    global_step=step, six_views=stop_rgbs,
                    six_depths=stop_depths, sub_instruction=sub_instruction,
                    action_history=navigation_result.action_history,
                    arrival_signal=navigation_result.signal,
                    completion_views=stop_completion_views,
                    metadata={
                        "executor_end_reason": navigation_result.end_reason,
                        "backtrack_reference_node_id": reference_target_id,
                        "backtrack_hop_attempt": hop_attempt,
                    },
                    edge_kind="node_backtrack_attempt",
                    edge_metadata={
                        "reference_forward_edge_id": forward_edge.edge_id,
                        "reference_source_node_id": reference_source_id,
                        "reference_target_node_id": reference_target_id,
                        "route_direction": route_direction,
                        "selection": selection.record,
                    })
                revisit = self.matcher.match(
                    stop_node, self.graph_memory.get_node(reference_target_id))
                loop_closure_edge = None
                if revisit.reached:
                    loop_closure_edge = self.graph_memory.add_loop_closure_edge(
                        reference_target_id, stop_node.node_id,
                        metadata={
                            "node_revisit_match": revisit.to_dict(),
                            "backtrack_attempt_edge_id": stop_edge.edge_id,
                        })
                self.graph_memory.set_node_metadata(stop_node.node_id, {
                    "node_revisit_match": revisit.to_dict(),
                    "loop_closure_edge_id": (
                        loop_closure_edge.edge_id if loop_closure_edge else None),
                })
                attempt_record = {
                    "segment_index": segment_index,
                    "hop_index": completed_hops,
                    "hop_attempt": hop_attempt,
                    "reference_source_node_id": reference_source_id,
                    "reference_target_node_id": reference_target_id,
                    "reference_edge_id": reference_edge_id,
                    "route_direction": route_direction,
                    "created_node_id": stop_node.node_id,
                    "created_edge_id": stop_edge.edge_id,
                    "loop_closure_edge_id": (
                        loop_closure_edge.edge_id if loop_closure_edge else None),
                    "selection": selection.record,
                    "executor_arrived": navigation_result.arrived,
                    "executor_end_reason": navigation_result.end_reason,
                    "executed_point_navigation": True,
                    "action_history": navigation_result.action_history,
                    "node_revisit_match": revisit.to_dict(),
                }
                attempts.append(attempt_record)
                segment_index += 1
                executed_segments += 1
                if revisit.reached:
                    hop_reached = True
                    completed_hops += 1
                    break
            if not hop_reached:
                end_reason = f"failed_to_revisit_{reference_target_id}"
                break

        success = completed_hops == len(route) - 1
        return NodeBacktrackResult(
            success=success, source_node_id=source_node_id,
            target_node_id=target_node_id, ancestor_route=route,
            completed_hops=completed_hops, attempted_segments=executed_segments,
            final_yaw=yaw, next_global_step=step,
            traveled_distance_m=total_distance, attempts=attempts,
            end_reason=end_reason if not success else "target_node_reached")


def backtrack_to_node(controller, target_node_id="previous", **kwargs):
    """Functional wrapper around :class:`NodeBacktrackingController`."""
    return controller.backtrack(target_node_id=target_node_id, **kwargs)
