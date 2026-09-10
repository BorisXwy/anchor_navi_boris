#!/usr/bin/env python3
"""FIFO graph-frontier exploration with point navigation and node revisits."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from path_projection import draw_reference_path_overlay, project_reference_path

from instruction_decomposer import SubInstruction
from node_backtracking import NodeBacktrackingController
from point_navigation_executor import PointNavigationRequest, execute_point_navigation
from point_selectors import (
    angle_distance, continuous_turn, exploration_ground_points,
    observe_six_rgbd, pixel_ground_to_world, targetable_ground_mask,
    wrap_angle,
)


PANORAMA_OFFSETS_RAD = np.radians([0, 60, 120, 180, 240, 300])


def _xz_distance(first, second):
    first = np.asarray(first, np.float32)
    second = np.asarray(second, np.float32)
    return float(np.linalg.norm(first[[0, 2]] - second[[0, 2]]))


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class BreadthFirstFrontierMemory:
    """Deterministic FIFO frontier queue with global spatial deduplication."""

    def __init__(self, dedup_radius_m=0.8, max_queue_size=512):
        self.dedup_radius_m = float(dedup_radius_m)
        self.max_queue_size = int(max_queue_size)
        self.queue = deque()
        self.records = []
        self.pop_events = []
        self._next_id = 0

    def _insert_level_order(self, record):
        """Keep retries behind peers at the same depth, before deeper work."""
        items = list(self.queue)
        insert_at = len(items)
        for index, queued in enumerate(items):
            if queued["source_depth"] > record["source_depth"]:
                insert_at = index
                break
        items.insert(insert_at, record)
        self.queue = deque(items)

    def enqueue(self, source_node_id, source_depth, world_xyz, metadata=None):
        world = np.asarray(world_xyz, np.float32)
        if any(_xz_distance(world, item["world_xyz"]) < self.dedup_radius_m
               for item in self.records):
            return None
        if len(self.queue) >= self.max_queue_size:
            return None
        record = {
            "frontier_id": f"frontier_{self._next_id:05d}",
            "source_node_id": str(source_node_id),
            "origin_source_node_id": str(source_node_id),
            "source_depth": int(source_depth),
            "world_xyz": world.tolist(),
            "status": "queued", "attempts": 0,
            "discovery_order": self._next_id,
            "metadata": _jsonable(metadata or {}),
        }
        self._next_id += 1
        self.records.append(record)
        self.queue.append(record)
        return record

    def pop(self):
        if not self.queue:
            return None
        head_id = self.queue[0]["frontier_id"]
        record = self.queue.popleft()
        record["status"] = "active"
        self.pop_events.append({
            "pop_order": len(self.pop_events),
            "frontier_id": record["frontier_id"],
            "source_depth": record["source_depth"],
            "was_queue_head": record["frontier_id"] == head_id,
        })
        return record

    def requeue(self, record):
        record["status"] = "queued"
        self._insert_level_order(record)

    def mark(self, record, status, **values):
        record["status"] = str(status)
        record.update(_jsonable(values))

    def snapshot(self):
        counts = {}
        for record in self.records:
            counts[record["status"]] = counts.get(record["status"], 0) + 1
        return {
            "queue_length": len(self.queue),
            "frontier_count": len(self.records),
            "frontier_status_counts": counts,
            "pop_events": _jsonable(self.pop_events),
            "frontiers": _jsonable(self.records),
        }


@dataclass
class BreadthFirstExplorationResult:
    success: bool
    end_reason: str
    expansions: int
    forward_attempts: int
    backtrack_requests: int
    backtrack_successes: int
    skipped_covered_frontiers: int
    blocked_frontiers: int
    final_yaw: float
    next_global_step: int
    traveled_distance_m: float
    selected_yaws_rad: list
    records: list
    backtrack_records: list
    state: dict

    def to_dict(self):
        return _jsonable(asdict(self))


class BreadthFirstExplorationStrategy:
    """Expand non-semantic grounded frontiers in strict discovery FIFO order."""

    mode = "breadth-first-frontier"

    def __init__(
            self, sim, segmenter, point_navigation_executor, graph_memory,
            position_history, rendered, video_composer, motion_log, output_dir,
            max_forward_attempts=120, novelty_radius_m=1.0,
            max_frontier_attempts=3, max_frontiers_per_node=6,
            scan_step_deg=10.0, backtrack_attempts_per_hop=4,
            backtrack_reach_radius_m=0.75,
            backtrack_minimum_visual_similarity=0.75,
            backtrack_planner_profile="breadcrumb_budget_v3",
            reference_path=None, reference_path_index=0):
        self.sim = sim
        self.segmenter = segmenter
        self.executor = point_navigation_executor
        self.graph_memory = graph_memory
        self.position_history = position_history
        self.rendered = rendered
        self.video_composer = video_composer
        self.motion_log = motion_log
        self.reference_path = reference_path
        self.reference_path_index = int(reference_path_index or 0)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bfs_dir = self.output_dir / "breadth_first_exploration"
        self.bfs_dir.mkdir(exist_ok=True)
        self.max_forward_attempts = int(max_forward_attempts)
        self.novelty_radius_m = float(novelty_radius_m)
        self.max_frontier_attempts = int(max_frontier_attempts)
        self.max_frontiers_per_node = int(max_frontiers_per_node)
        self.scan_step = math.radians(float(scan_step_deg))
        self.memory = BreadthFirstFrontierMemory(
            dedup_radius_m=max(0.60, 0.8 * self.novelty_radius_m))
        self.expanded_node_ids = []
        self.expanded_positions = []
        self.source_backtrack_failures = {}
        self.backtracker = NodeBacktrackingController(
            sim=sim, graph_memory=graph_memory, segmenter=segmenter,
            point_navigation_executor=point_navigation_executor,
            position_history=position_history, rendered=rendered,
            video_composer=video_composer, motion_log=motion_log,
            vlm_harness=None, selector_mode="hybrid",
            scan_step_deg=scan_step_deg,
            max_attempts_per_hop=backtrack_attempts_per_hop,
            reach_radius_m=backtrack_reach_radius_m,
            minimum_visual_similarity=backtrack_minimum_visual_similarity,
            max_hops=max(256, self.max_forward_attempts * 8),
            output_dir=output_dir,
            planner_profile=backtrack_planner_profile,
            reference_path=self.reference_path,
            reference_path_index=self.reference_path_index)

    def _path_to(self, start, end):
        shortest = __import__("habitat_sim").ShortestPath()
        shortest.requested_start = np.asarray(start, np.float32)
        shortest.requested_end = np.asarray(end, np.float32)
        if not self.sim.pathfinder.find_path(shortest):
            return None
        return shortest

    def _ground_candidates(self, position, yaw, desired_frontier=None):
        rgbs, depths = observe_six_rgbd(self.sim)
        desired_yaw = None
        route_waypoint = None
        if desired_frontier is not None:
            path = self._path_to(position, desired_frontier)
            if path is None or len(path.points) < 2:
                return [], rgbs, depths
            route_waypoint = np.asarray(path.points[1], np.float32)
            delta = route_waypoint - np.asarray(position, np.float32)
            desired_yaw = wrap_angle(math.atan2(
                -float(delta[0]), -float(delta[2])))
        segmentations = (self.segmenter.batch(rgbs)
                         if hasattr(self.segmenter, "batch") else
                         [self.segmenter(rgb) for rgb in rgbs])
        candidates = []
        for view_index, (offset, rgb, depth, segmentation) in enumerate(zip(
                PANORAMA_OFFSETS_RAD, rgbs, depths, segmentations)):
            candidate_yaw = wrap_angle(float(yaw) + float(offset))
            ground, detections = segmentation
            target_mask = targetable_ground_mask(ground)
            options = []
            for point, point_score in exploration_ground_points(
                    target_mask, depth, count=4):
                world, selected_depth = pixel_ground_to_world(
                    point, depth, position, candidate_yaw)
                snapped = None
                geodesic = math.inf
                if world is not None:
                    value = np.asarray(
                        self.sim.pathfinder.snap_point(world), np.float32)
                    if np.isfinite(value).all():
                        route = self._path_to(position, value)
                        if route is not None:
                            snapped = value
                            geodesic = float(route.geodesic_distance)
                if snapped is None:
                    continue
                h, w = rgb.shape[:2]
                pixel_bearing = math.atan(
                    (float(point[0]) - w / 2) / (w / 2))
                ray_yaw = wrap_angle(candidate_yaw - pixel_bearing)
                if desired_yaw is None:
                    route_error = 0.0
                    target_distance = 0.0
                else:
                    route_error = angle_distance(ray_yaw, desired_yaw)
                    target_distance = _xz_distance(
                        snapped, desired_frontier)
                options.append({
                    "point": np.asarray(point, np.float32),
                    "point_score": float(point_score),
                    "world_point": snapped,
                    "selected_depth_m": float(selected_depth),
                    "geodesic_distance_m": geodesic,
                    "ray_yaw_rad": ray_yaw,
                    "route_error_rad": route_error,
                    "desired_frontier_distance_m": target_distance,
                    "route_score": (
                        route_error + 0.12 * target_distance -
                        0.01 * min(float(point_score), 30.0)),
                })
            if desired_yaw is None:
                best = (max(options, key=lambda item: (
                    item["geodesic_distance_m"], item["point_score"]))
                        if options else None)
            else:
                best = (min(options, key=lambda item: item["route_score"])
                        if options else None)
            candidates.append({
                "view_index": view_index,
                "relative_yaw_rad": float(offset), "yaw": candidate_yaw,
                "rgb": rgb, "depth": depth, "mask": ground,
                "target_mask": target_mask,
                "ground_fraction": float(ground.mean()),
                "ground_detection_records": [
                    item.prompt_record() for item in detections],
                "options": options, "best": best,
                "route_waypoint_xyz": (
                    route_waypoint.tolist()
                    if route_waypoint is not None else None),
            })
        return candidates, rgbs, depths

    def _visualize(self, candidates, selected_view, phase, index, yaw):
        images = []
        position = np.asarray(
            self.sim.get_agent(0).get_state().position, np.float32)
        for candidate in candidates:
            rgb = candidate["rgb"].copy()
            green = np.zeros_like(rgb); green[..., 1] = 255
            mask = candidate["target_mask"]
            rgb[mask] = (0.55 * rgb[mask] + 0.45 * green[mask]).astype(
                np.uint8)
            if self.reference_path:
                projection = project_reference_path(
                    self.reference_path,
                    range(self.reference_path_index, len(self.reference_path)),
                    position + np.array([0.0, 1.25, 0.0], np.float32),
                    float(candidate["yaw"]), rgb.shape[1], rgb.shape[0])
                rgb = draw_reference_path_overlay(
                    rgb, projection,
                    selected_point=(candidate["best"]["point"]
                                    if candidate["view_index"] == selected_view and
                                    candidate.get("best") is not None else None),
                    selected=bool(candidate["view_index"] == selected_view and
                                  candidate.get("best") is not None),
                    next_path_index=self.reference_path_index + 1)
                candidate["reference_path_projection"] = projection
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            best = candidate.get("best")
            label = "GROUND" if best is not None else "NO CANDIDATE"
            cv2.putText(
                frame, f"BFS {phase} VIEW {candidate['view_index']} {label}",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (255, 255, 255), 1)
            if candidate["view_index"] == selected_view and best is not None:
                cv2.drawMarker(
                    frame, tuple(np.round(best["point"]).astype(int)),
                    (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
                cv2.putText(frame, "FIFO FRONTIER", (6, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                            (0, 255, 255), 2)
            images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            composed = self.video_composer.compose(
                frame, self.position_history, position,
                candidate["yaw"], "BFS non-semantic exploration",
                index, f"bfs_{phase}")
            self.rendered.extend([composed, composed.copy()])
        if len(images) == 6:
            sheet = np.concatenate([
                np.concatenate(images[:3], axis=1),
                np.concatenate(images[3:], axis=1)], axis=0)
            Image.fromarray(sheet).save(
                self.bfs_dir / f"{index:04d}_{phase}_six_views.jpg")

    def _discover(self, node_id, depth, yaw, discovery_index):
        position = np.asarray(
            self.sim.get_agent(0).get_state().position, np.float32)
        if any(_xz_distance(position, old) < self.novelty_radius_m
               for old in self.expanded_positions):
            return 0
        self.expanded_node_ids.append(str(node_id))
        self.expanded_positions.append(position.copy())
        self.graph_memory.set_node_metadata(node_id, {
            "bfs_expanded": True, "bfs_depth": int(depth),
            "bfs_expansion_order": len(self.expanded_node_ids) - 1,
        })
        candidates, _, _ = self._ground_candidates(position, yaw)
        added = 0
        for candidate in candidates:
            options = sorted(
                candidate["options"],
                key=lambda item: (
                    -item["geodesic_distance_m"], -item["point_score"]))
            for option in options[:1]:
                endpoint = option["world_point"]
                if option["geodesic_distance_m"] < 0.65:
                    continue
                if any(_xz_distance(endpoint, old) < self.novelty_radius_m
                       for old in self.expanded_positions):
                    continue
                record = self.memory.enqueue(
                    source_node_id=node_id, source_depth=depth,
                    world_xyz=endpoint,
                    metadata={
                        "view_index": candidate["view_index"],
                        "source_yaw_rad": yaw,
                        "candidate_yaw_rad": candidate["yaw"],
                        "point_xy": option["point"].tolist(),
                        "selected_depth_m": option["selected_depth_m"],
                        "geodesic_distance_m": option[
                            "geodesic_distance_m"],
                    })
                if record is not None:
                    added += 1
                if added >= self.max_frontiers_per_node:
                    break
            if added >= self.max_frontiers_per_node:
                break
        self._visualize(candidates, -1, "discover", discovery_index, yaw)
        return added

    def _write_checkpoint(self, **values):
        payload = {**self.memory.snapshot(), **_jsonable(values)}
        (self.output_dir / "bfs_exploration_checkpoint.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    def run(self, initial_yaw=0.0, initial_global_step=0):
        yaw = float(initial_yaw)
        step = int(initial_global_step)
        total_distance = 0.0
        records = []
        backtrack_records = []
        selected_yaws = []
        backtrack_requests = 0
        backtrack_successes = 0
        skipped = 0
        blocked = 0
        origin = self.graph_memory.nodes[-1]
        self._discover(origin.node_id, 0, yaw, 0)

        while self.memory.queue and len(records) < self.max_forward_attempts:
            frontier = self.memory.pop()
            target = np.asarray(frontier["world_xyz"], np.float32)
            if any(_xz_distance(target, old) < self.novelty_radius_m
                   for old in self.expanded_positions):
                self.memory.mark(frontier, "skipped_covered")
                skipped += 1
                continue

            source_node_id = frontier["source_node_id"]
            current_position = np.asarray(
                self.sim.get_agent(0).get_state().position, np.float32)
            source_position = np.asarray(
                self.graph_memory.get_node(source_node_id).position_xyz,
                np.float32)
            backtrack_payload = None
            if _xz_distance(current_position, source_position) > 0.75:
                backtrack_requests += 1
                backtrack = self.backtracker.backtrack(
                    target_node_id=source_node_id, current_yaw=yaw,
                    global_step=step,
                    target_index_offset=self.max_forward_attempts +
                    len(backtrack_records) * 100)
                backtrack_payload = backtrack.to_dict()
                backtrack_records.append({
                    "frontier_id": frontier["frontier_id"],
                    "source_node_id": source_node_id,
                    "result": backtrack_payload,
                })
                yaw = backtrack.final_yaw
                step = backtrack.next_global_step
                total_distance += backtrack.traveled_distance_m
                if not backtrack.success:
                    frontier["attempts"] += 1
                    source_failures = (
                        self.source_backtrack_failures.get(
                            source_node_id, 0) + 1)
                    self.source_backtrack_failures[source_node_id] = (
                        source_failures)
                    if source_failures < self.max_frontier_attempts:
                        self.memory.requeue(frontier)
                    else:
                        affected = [frontier]
                        retained = deque()
                        for queued in self.memory.queue:
                            if queued["source_node_id"] == source_node_id:
                                affected.append(queued)
                            else:
                                retained.append(queued)
                        self.memory.queue = retained
                        for item in affected:
                            self.memory.mark(
                                item, "blocked_source_backtrack",
                                source_backtrack_failure_count=(
                                    source_failures),
                                last_backtrack=backtrack_payload)
                        blocked += len(affected)
                    continue
                self.source_backtrack_failures.pop(source_node_id, None)
                backtrack_successes += 1

            current_position = np.asarray(
                self.sim.get_agent(0).get_state().position, np.float32)
            frontier_distance_before = _xz_distance(
                current_position, target)
            candidates, _, _ = self._ground_candidates(
                current_position, yaw, target)
            usable = [candidate for candidate in candidates
                      if candidate.get("best") is not None]
            if not usable:
                frontier["attempts"] += 1
                if frontier["attempts"] < self.max_frontier_attempts:
                    self.memory.requeue(frontier)
                else:
                    self.memory.mark(frontier, "blocked_no_ground")
                    blocked += 1
                continue
            chosen = min(
                usable, key=lambda candidate: candidate["best"]["route_score"])
            self._visualize(
                candidates, chosen["view_index"], "execute",
                len(records), yaw)
            yaw, chosen_rgb = continuous_turn(
                self.sim, current_position, yaw, chosen["yaw"],
                self.scan_step, self.rendered, self.motion_log,
                len(records), "bfs_turn_to_frontier", self.video_composer,
                self.position_history, "BFS non-semantic exploration",
                reference_path=self.reference_path,
                reference_path_index=self.reference_path_index)
            option = chosen["best"]
            selected_yaws.append(yaw)
            navigation = execute_point_navigation(
                self.executor, PointNavigationRequest(
                    rgb=chosen_rgb, selected_point_xy=option["point"],
                    selectable_mask=chosen["target_mask"],
                    ground_mask=chosen["mask"], yaw=yaw,
                    position_history=self.position_history,
                    instruction="BFS non-semantic frontier expansion",
                    semantic_target="unvisited grounded floor frontier",
                    target_index=len(records),
                    stage_count=self.max_forward_attempts,
                    global_step=step,
                    selected_point_depth_m=option["selected_depth_m"],
                    selected_point_reachable=True,
                    reference_path=self.reference_path,
                    reference_path_index=self.reference_path_index))
            yaw = navigation.final_yaw
            step = navigation.next_global_step
            segment_distance = sum(float(item.get("moved_m", 0.0))
                                   for item in navigation.action_history)
            total_distance += segment_distance
            stop_position = np.asarray(
                self.sim.get_agent(0).get_state().position, np.float32)
            stop_rgbs, stop_depths = observe_six_rgbd(self.sim)
            sub_instruction = SubInstruction.from_mapping({
                "sub_instruction_id": 800000 + len(records),
                "navigation_instruction": "BFS non-semantic frontier expansion",
                "landmark": "none",
                "completion_cue": "frontier point reached or tracker stops",
                "semantic_spatial_target": "unvisited grounded floor frontier",
                "spatial_relation": "FIFO breadth-first child frontier",
                "visual_arrival_evidence": "point cluster disappearance",
                "forbidden_target": "already expanded spatial node",
                "form": "BFS_PURE_EXPLORATION",
            })
            selected_route = self._path_to(stop_position, option["world_point"])
            selected_final_geodesic = (
                float(selected_route.geodesic_distance)
                if selected_route is not None else math.inf)
            physical_arrival = bool(
                navigation.arrived and selected_final_geodesic <= 0.75)
            stop_node = None
            stop_edge = None
            if physical_arrival:
                stop_node, stop_edge = self.graph_memory.add_navigation_stop_node(
                    position_xyz=stop_position, base_yaw_rad=yaw,
                    global_step=step, six_views=stop_rgbs,
                    six_depths=stop_depths, sub_instruction=sub_instruction,
                    action_history=navigation.action_history,
                    arrival_signal=navigation.signal,
                    metadata={
                        "bfs_frontier_id": frontier["frontier_id"],
                        "bfs_source_node_id": source_node_id,
                        "bfs_parent_depth": frontier["source_depth"],
                    }, edge_kind="bfs_frontier_expansion",
                    edge_metadata={
                        "frontier": frontier,
                        "edge_keyframes": navigation.record.get(
                            "edge_keyframes", []),
                    })
            frontier_distance = _xz_distance(stop_position, target)
            new_spatial_node = not any(
                _xz_distance(stop_position, old) < self.novelty_radius_m
                for old in self.expanded_positions)
            record = {
                "target_index": len(records),
                "strategy": self.mode,
                "frontier_id": frontier["frontier_id"],
                "source_node_id": source_node_id,
                "source_depth": frontier["source_depth"],
                "queue_length_before_execution": len(self.memory.queue),
                "selection": {
                    "method": "fifo_frontier_route_ground",
                    "view_index": chosen["view_index"],
                    "point_xy": option["point"].tolist(),
                    "frontier_world_xyz": target.tolist(),
                    "selected_navmesh_world_xyz": option[
                        "world_point"].tolist(),
                    "route_error_rad": option["route_error_rad"],
                },
                "arrived": navigation.arrived,
                "navigation_physical_arrival": physical_arrival,
                "navigation_physical_failure_type": (
                    None if physical_arrival else (
                        "premature_offscreen_stop"
                        if navigation.arrived else navigation.end_reason)),
                **navigation.record,
                "action_history": navigation.action_history,
                "navigation_graph_node_id": (
                    stop_node.node_id if stop_node is not None else None),
                "navigation_graph_edge_id": (
                    stop_edge.edge_id if stop_edge is not None else None),
                "frontier_final_planar_distance_m": frontier_distance,
                "frontier_initial_planar_distance_m": (
                    frontier_distance_before),
                "frontier_progress_m": (
                    frontier_distance_before - frontier_distance),
                "selected_point_final_geodesic_m": (
                    selected_final_geodesic
                    if math.isfinite(selected_final_geodesic) else None),
                "new_spatial_node": new_spatial_node,
                "backtrack_to_source": backtrack_payload,
            }
            records.append(record)
            frontier["attempts"] += 1
            if not physical_arrival:
                if frontier["attempts"] < self.max_frontier_attempts:
                    self.memory.requeue(frontier)
                else:
                    self.memory.mark(
                        frontier, "blocked_navigation",
                        final_planar_distance_m=frontier_distance)
                    blocked += 1
                self._write_checkpoint(
                    forward_attempts=len(records), expansions=len(
                        self.expanded_node_ids), global_step=step,
                    latest_node_id=(
                        stop_node.node_id if stop_node is not None else source_node_id),
                    latest_frontier_id=frontier["frontier_id"])
                continue
            reached = bool(frontier_distance <= self.novelty_radius_m)
            if reached:
                self.memory.mark(
                    frontier, "visited", reached_node_id=stop_node.node_id,
                    final_planar_distance_m=frontier_distance)
                if new_spatial_node:
                    self._discover(
                        stop_node.node_id, frontier["source_depth"] + 1,
                        yaw, len(records))
            elif (new_spatial_node and
                  frontier_distance < frontier_distance_before - 0.20 and
                  frontier["attempts"] < self.max_frontier_attempts):
                next_depth = frontier["source_depth"] + 1
                self._discover(stop_node.node_id, next_depth, yaw, len(records))
                frontier["source_node_id"] = stop_node.node_id
                frontier["source_depth"] = next_depth
                frontier.setdefault("transit_node_ids", []).append(
                    stop_node.node_id)
                self.memory.requeue(frontier)
            elif frontier["attempts"] < self.max_frontier_attempts:
                self.memory.requeue(frontier)
            else:
                self.memory.mark(
                    frontier, "blocked_navigation",
                    final_planar_distance_m=frontier_distance)
                blocked += 1
            self._write_checkpoint(
                forward_attempts=len(records), expansions=len(
                    self.expanded_node_ids), global_step=step,
                latest_node_id=stop_node.node_id,
                latest_frontier_id=frontier["frontier_id"])

        if not self.memory.queue:
            end_reason = "frontier_queue_exhausted"
            success = True
        else:
            end_reason = "max_forward_attempts"
            success = False
        state = {
            **self.memory.snapshot(),
            "termination_reason": end_reason,
            "frontier_queue_exhausted": not self.memory.queue,
            "safety_cap_reached": bool(self.memory.queue),
            "expansions": len(self.expanded_node_ids),
            "forward_attempts": len(records),
            "backtrack_requests": backtrack_requests,
            "backtrack_successes": backtrack_successes,
            "skipped_covered_frontiers": skipped,
            "blocked_frontiers": blocked,
            "source_backtrack_failures": dict(
                self.source_backtrack_failures),
            "expanded_node_ids": list(self.expanded_node_ids),
            "expanded_positions_xyz": [
                position.tolist() for position in self.expanded_positions],
            "fifo_order_verified": all(
                event["was_queue_head"] for event in self.memory.pop_events),
            "level_order_verified": all(
                left["source_depth"] <= right["source_depth"]
                for left, right in zip(
                    self.memory.pop_events[:-1],
                    self.memory.pop_events[1:])),
        }
        self._write_checkpoint(
            status="complete", end_reason=end_reason,
            forward_attempts=len(records),
            expansions=len(self.expanded_node_ids), global_step=step)
        return BreadthFirstExplorationResult(
            success=success, end_reason=end_reason,
            expansions=len(self.expanded_node_ids),
            forward_attempts=len(records),
            backtrack_requests=backtrack_requests,
            backtrack_successes=backtrack_successes,
            skipped_covered_frontiers=skipped,
            blocked_frontiers=blocked, final_yaw=yaw,
            next_global_step=step, traveled_distance_m=total_distance,
            selected_yaws_rad=selected_yaws, records=records,
            backtrack_records=backtrack_records, state=state)
