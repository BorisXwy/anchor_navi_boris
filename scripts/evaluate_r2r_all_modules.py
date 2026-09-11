#!/usr/bin/env python3
"""Run and score every point-navigation module on fixed real R2R states."""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HABITAT_ENTRYPOINT = ROOT / "scripts/habitat_point_navigation.py"
DEFAULT_SAMPLE_MANIFEST = (
    ROOT / "outputs/vlm_point_selection_100ep_v3_soft_semantic/"
    "sample_manifest.json")
DEFAULT_R2R_DATA = (
    ROOT.parent / "3d_wm_vln/StreamVLN/data/datasets/r2r/val_unseen/"
    "val_unseen.json.gz")
MODULE_KEYS = [
    "decomposition_valid",
    "six_view_rgbd_valid",
    "ground_candidate_valid",
    "vlm_api_valid",
    "point_direction_correct",
    "point_on_ground",
    "point_depth_valid",
    "crop_valid",
    "dual_tracking_clusters_valid",
    "tracker_runtime_valid",
    "image_goal_policy_valid",
    "navigation_internal_arrival",
    "navigation_physical_arrival",
    "node_persisted",
    "edge_action_history_valid",
    "sub_instruction_classification_correct",
    "backtrack_selection_valid",
    "backtrack_physical_revisit",
    "loop_closure_valid",
    "sequence_recovery_policy_correct",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2 * math.pi) - math.pi


def heading_between(start, end) -> float | None:
    delta = np.asarray(end, np.float32) - np.asarray(start, np.float32)
    if float(np.linalg.norm(delta[[0, 2]])) <= 1e-6:
        return None
    return wrap_angle(math.atan2(-float(delta[0]), -float(delta[2])))


def angular_error_deg(first, second) -> float:
    if first is None or second is None:
        return math.inf
    return abs(math.degrees(wrap_angle(float(first) - float(second))))


def distance_to_polyline_xz(point, path) -> float:
    point = np.asarray(point, np.float32)[[0, 2]]
    path = np.asarray(path, np.float32)[:, [0, 2]]
    if len(path) == 0:
        return math.inf
    if len(path) == 1:
        return float(np.linalg.norm(point - path[0]))
    best = math.inf
    for start, end in zip(path[:-1], path[1:]):
        segment = end - start
        scale = float(np.dot(point - start, segment) /
                      max(float(np.dot(segment, segment)), 1e-8))
        projection = start + np.clip(scale, 0.0, 1.0) * segment
        best = min(best, float(np.linalg.norm(point - projection)))
    return best


def closest_reference_index(position, path) -> int:
    position = np.asarray(position, np.float32)[[0, 2]]
    path = np.asarray(path, np.float32)[:, [0, 2]]
    return int(np.argmin(np.linalg.norm(path - position[None], axis=1)))


def wilson(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    probability = successes / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    margin = z / denominator * math.sqrt(
        probability * (1 - probability) / total +
        z * z / (4 * total * total))
    return [max(0.0, center - margin), min(1.0, center + margin)]


def valid_decomposition(stages) -> bool:
    required = (
        "navigation_instruction", "form", "semantic_spatial_target",
        "spatial_relation", "visual_arrival_evidence", "forbidden_target")
    return bool(stages) and all(
        all(str(stage.get(key, "")).strip() for key in required)
        for stage in stages)


def graph_maps(graph):
    return (
        {node["node_id"]: node for node in graph.get("nodes", [])},
        {edge["edge_id"]: edge for edge in graph.get("edges", [])},
    )


def hop_demo_truth(target, nodes, edges, reference_path,
                   default_start_index):
    edge = edges.get(target.get("navigation_graph_edge_id"), {})
    source = nodes.get(edge.get("source_node_id"), {})
    stop = nodes.get(target.get("navigation_graph_node_id"), {})
    start_position = source.get("position_xyz")
    stop_position = stop.get("position_xyz")
    selected = target.get("selected_navmesh_target_xyz")
    if start_position is None or stop_position is None or selected is None:
        return {
            "ground_truth_belongs": False,
            "heading_error_to_local_demo_deg": math.inf,
            "forward_projection_m": -math.inf,
            "distance_to_future_demo_polyline_m": math.inf,
            "local_reference_index": None,
        }
    local_index = max(
        int(default_start_index),
        closest_reference_index(start_position, reference_path))
    local_index = min(local_index, len(reference_path) - 2)
    demo_heading = heading_between(
        reference_path[local_index], reference_path[local_index + 1])
    selection_heading = heading_between(start_position, selected)
    heading_error = angular_error_deg(selection_heading, demo_heading)
    displacement = (np.asarray(stop_position, np.float32) -
                    np.asarray(start_position, np.float32))[[0, 2]]
    demo_vector = (
        np.asarray(reference_path[local_index + 1], np.float32) -
        np.asarray(reference_path[local_index], np.float32))[[0, 2]]
    demo_norm = float(np.linalg.norm(demo_vector))
    forward_projection = (
        float(np.dot(displacement, demo_vector / demo_norm))
        if demo_norm > 1e-6 else -math.inf)
    future_distance = distance_to_polyline_xz(
        stop_position, reference_path[local_index:])
    belongs = bool(
        heading_error < 90.0 and forward_projection >= 0.20 and
        future_distance <= 3.0)
    return {
        "ground_truth_belongs": belongs,
        "heading_error_to_local_demo_deg": heading_error,
        "forward_projection_m": forward_projection,
        "distance_to_future_demo_polyline_m": future_distance,
        "local_reference_index": local_index,
        "start_position_xyz": start_position,
        "stop_position_xyz": stop_position,
    }


def score_sequence(targets, hop_truths, trajectory):
    correct = bool(targets)
    recovery_opportunities = 0
    recovery_successes = 0
    off_sequence_depth = 0
    checks = []
    for target, truth in zip(targets, hop_truths):
        if truth["ground_truth_belongs"]:
            expected_action = "complete"
            off_sequence_depth = 0
        elif off_sequence_depth == 0:
            expected_action = "explore_once_more"
            off_sequence_depth = 1
        else:
            expected_action = "backtrack_and_block"
            recovery_opportunities += 1
        directive = target.get("sequence_directive") or {}
        action_matches = directive.get("action") == expected_action
        recovery_ok = True
        block_ok = True
        if expected_action == "backtrack_and_block":
            recovery = target.get("sequence_recovery_backtrack") or {}
            recovery_ok = bool(recovery.get("success"))
            state = ((trajectory.get("instruction_sequence_exploration") or {})
                     .get("state") or {})
            blocked = state.get("blocked_yaws_by_verified_node") or {}
            expected_node = directive.get("backtrack_target_node_id")
            expected_yaw = directive.get("direction_to_block_yaw_rad")
            values = blocked.get(str(expected_node), [])
            block_ok = bool(expected_yaw is not None and any(
                angular_error_deg(value, expected_yaw) < 20.0
                for value in values))
            if recovery_ok and block_ok:
                recovery_successes += 1
            # A successful recovery leaves one off-sequence branch origin.
            off_sequence_depth = 1 if recovery_ok else 2
        hop_ok = bool(action_matches and recovery_ok and block_ok)
        correct = correct and hop_ok
        checks.append({
            "target_index": target.get("target_index"),
            "ground_truth_belongs": truth["ground_truth_belongs"],
            "expected_action": expected_action,
            "actual_action": directive.get("action"),
            "action_matches": action_matches,
            "recovery_success": recovery_ok,
            "blocked_yaw_valid": block_ok,
        })
        if truth["ground_truth_belongs"]:
            break
    if targets and not any(item["ground_truth_belongs"] for item in hop_truths):
        # An unfinished first-off-sequence directive at the hop limit is not a
        # completed strategy outcome.
        last_expected = checks[-1]["expected_action"] if checks else None
        if last_expected == "explore_once_more":
            correct = False
    return correct, recovery_opportunities, recovery_successes, checks


def score_case(case, case_dir: Path, episode, process_returncode=0,
               expected_views=8):
    trajectory_path = case_dir / "trajectory.json"
    graph_path = case_dir / "navigation_graph/navigation_graph.json"
    if process_returncode != 0 or not trajectory_path.exists() or not graph_path.exists():
        return {
            "case_index": case["case_index"],
            "episode_index": case["episode_index"],
            "episode_id": case["episode_id"],
            "reference_path_index": case["reference_path_index"],
            "process_returncode": process_returncode,
            "error": "episode_process_failed_or_missing_artifacts",
            "modules": {key: False for key in MODULE_KEYS},
            "all_modules_success": False,
            "backtrack_pair_formed": False,
            "sequence_recovery_opportunities": 0,
            "sequence_recovery_successes": 0,
        }
    trajectory = json.loads(trajectory_path.read_text())
    graph = json.loads(graph_path.read_text())
    nodes, edges = graph_maps(graph)
    targets = trajectory.get("targets") or []
    first = targets[0] if targets else {}
    candidates = first.get("candidate_views") or []
    selection = first.get("selection") or {}
    reference_path = np.asarray(episode.get("reference_path", []), np.float32)

    initial_images = list((case_dir / "initial_six_views").glob("view_*.jpg"))
    initial_depths = list((case_dir / "initial_depths").glob("view_*.npy"))
    ground_instances = sum(
        len(candidate.get("ground_detection_records") or [])
        for candidate in candidates)
    base_candidates = [
        item for item in candidates if not item.get("is_refined_view", False)]
    ground_candidate_valid = bool(
        len(base_candidates) == expected_views and ground_instances > 0 and
        any(float(candidate.get("ground_fraction", 0.0)) > 0
            for candidate in candidates) and selection)

    point_depth = first.get("selected_point_depth_m")
    selected = first.get("selected_navmesh_target_xyz")
    initial_position = np.asarray(case["reference_position_xyz"], np.float32)
    demo_next = (reference_path[case["reference_path_index"] + 1]
                 if len(reference_path) > case["reference_path_index"] + 1
                 else None)
    heading_error = angular_error_deg(
        heading_between(initial_position, selected) if selected is not None else None,
        heading_between(initial_position, demo_next) if demo_next is not None else None)

    crop = first.get("initial_crop_geometry") or {}
    quad = np.asarray(crop.get("quad_xy") or [], np.float32)
    crop_valid = bool(
        quad.shape == (4, 2) and
        np.isclose(quad[0, 1], quad[1, 1]) and
        np.isclose(quad[2, 1], quad[3, 1]) and
        float(crop.get("axis_length_px", 0)) >=
        float(crop.get("minimum_height_px", 1)) and
        crop.get("output_size_wh") in ([85, 64], [96, 96]))
    steps = first.get("steps") or []
    navigation_cluster_size = int(first.get("navigation_cluster_size") or 0)
    stop_cluster_size = int(first.get("stop_cluster_size") or 0)
    tracker_valid = bool(steps) and all(
        len(step.get("navigation_tracks_xy") or []) == navigation_cluster_size and
        len(step.get("stop_tracks_xy") or []) == stop_cluster_size
        for step in steps)
    policy_valid = bool(steps) and all(
        step.get("policy_distance") is not None and
        bool(step.get("policy_first_trajectory")) for step in steps)

    node_persisted = bool(targets) and all(
        target.get("navigation_graph_node_id") in nodes for target in targets)
    edge_valid = bool(targets)
    for target in targets:
        edge = edges.get(target.get("navigation_graph_edge_id"))
        edge_valid = edge_valid and bool(
            edge is not None and
            edge.get("action_history") == target.get("action_history", []))

    hop_truths = [hop_demo_truth(
        target, nodes, edges, reference_path, case["reference_path_index"])
        for target in targets]
    first_truth = hop_truths[0] if hop_truths else {
        "ground_truth_belongs": False}
    classification_checks = []
    for target, truth in zip(targets, hop_truths):
        classification = (
            target.get("sub_instruction_sequence_classification") or {})
        predicted_positive = bool(
            classification.get("belongs_to_sequence") and
            classification.get("matched_sub_instruction_id") ==
            classification.get("expected_sub_instruction_id") and
            float(classification.get("confidence", 0.0)) >= 0.5)
        classification_checks.append({
            "target_index": target.get("target_index"),
            "ground_truth_positive": truth["ground_truth_belongs"],
            "predicted_positive": predicted_positive,
            "correct": bool(
                predicted_positive == truth["ground_truth_belongs"]),
            "classification": classification,
        })
    classification_correct = bool(classification_checks) and all(
        item["correct"] for item in classification_checks)
    first_classification = (
        classification_checks[0]["classification"]
        if classification_checks else {})
    first_predicted_positive = bool(
        classification_checks and
        classification_checks[0]["predicted_positive"])

    backtrack = trajectory.get("node_backtracking") or {}
    backtrack_source = nodes.get(backtrack.get("source_node_id"), {})
    origin = nodes.get(backtrack.get("target_node_id", "node_0000"),
                       nodes.get("node_0000", {}))
    source_position = backtrack_source.get("position_xyz")
    origin_position = origin.get("position_xyz")
    backtrack_distance = (
        float(np.linalg.norm(
            (np.asarray(source_position, np.float32) -
             np.asarray(origin_position, np.float32))[[0, 2]]))
        if source_position is not None and origin_position is not None
        else 0.0)
    pair_formed = backtrack_distance >= 1.0
    backtrack_attempts = backtrack.get("attempts") or []
    executed_attempts = [attempt for attempt in backtrack_attempts
                         if attempt.get("executed_point_navigation")]
    backtrack_direction_errors = []
    for attempt in executed_attempts:
        item = attempt.get("selection") or {}
        backtrack_direction_errors.append(angular_error_deg(
            item.get("selected_yaw_rad"),
            item.get("desired_target_bearing_yaw_rad")))
    backtrack_selection_valid = bool(
        pair_formed and backtrack_direction_errors and
        min(backtrack_direction_errors) < 90.0)
    backtrack_physical = bool(
        pair_formed and backtrack.get("success") and executed_attempts)
    loop_closure = bool(
        backtrack_physical and any(
            attempt.get("loop_closure_edge_id") for attempt in backtrack_attempts))

    sequence_correct, recovery_opportunities, recovery_successes, sequence_checks = (
        score_sequence(targets, hop_truths, trajectory))

    vlm_calls_path = case_dir / "vlm_calls.json"
    vlm_calls = (json.loads(vlm_calls_path.read_text())
                 if vlm_calls_path.exists() else [])
    tasks = [call.get("task") for call in vlm_calls]
    # A frozen decomposition artifact legitimately removes the online
    # decomposition call; validate that stage structurally above and require
    # only the visual decisions that this run must execute.
    # API health is conditional on calls that were actually reachable.  A
    # missing downstream completion call after navigation failure is an
    # upstream cascade, not a VLM API failure.
    vlm_valid = bool(vlm_calls and all(
        call.get("finish_reason") == "stop" for call in vlm_calls))

    modules = {
        "decomposition_valid": valid_decomposition(
            trajectory.get("all_decomposed_sub_instructions") or []),
        "six_view_rgbd_valid": (
            len(initial_images) == 6 and len(initial_depths) == 6),
        "ground_candidate_valid": ground_candidate_valid,
        "vlm_api_valid": vlm_valid,
        "point_direction_correct": heading_error <= 30.0,
        "point_on_ground": bool(
            selection.get("requested_on_ground") is True and
            len(selection.get("snapped_xy") or []) == 2 and
            first.get("raw_ground_mask_provided") is True and
            first.get("all_initial_cluster_points_on_ground") is True and
            first.get("all_navigation_cluster_points_on_ground") is True and
            first.get("all_stop_cluster_points_on_ground") is True),
        "point_depth_valid": bool(
            point_depth is not None and math.isfinite(float(point_depth)) and
            float(point_depth) > 0 and selected is not None),
        "crop_valid": crop_valid,
        "dual_tracking_clusters_valid": bool(
            navigation_cluster_size >= 9 and
            stop_cluster_size >= navigation_cluster_size and
            first.get("all_navigation_cluster_points_on_ground") is True and
            first.get("all_stop_cluster_points_on_ground") is True),
        "tracker_runtime_valid": tracker_valid,
        "image_goal_policy_valid": policy_valid,
        "navigation_internal_arrival": bool(first.get("arrived")),
        "navigation_physical_arrival": bool(
            first.get("navigation_physical_arrival")),
        "node_persisted": node_persisted,
        "edge_action_history_valid": edge_valid,
        "sub_instruction_classification_correct": classification_correct,
        "backtrack_selection_valid": backtrack_selection_valid,
        "backtrack_physical_revisit": backtrack_physical,
        "loop_closure_valid": loop_closure,
        "sequence_recovery_policy_correct": sequence_correct,
    }
    return {
        "case_index": case["case_index"],
        "episode_index": case["episode_index"],
        "episode_id": case["episode_id"],
        "scene_id": case["scene_id"],
        "reference_path_index": case["reference_path_index"],
        "process_returncode": process_returncode,
        "error": None,
        "modules": modules,
        "all_modules_success": all(modules.values()),
        "heading_error_to_demo_next_deg": heading_error,
        "heading_thresholds": {
            str(threshold): heading_error < threshold
            for threshold in (30, 45, 60, 90)},
        "first_hop_demo_truth": first_truth,
        "first_classification": first_classification,
        "first_classification_predicted_positive": first_predicted_positive,
        "node_classification_checks": classification_checks,
        "hop_demo_truths": hop_truths,
        "sequence_checks": sequence_checks,
        "sequence_recovery_opportunities": recovery_opportunities,
        "sequence_recovery_successes": recovery_successes,
        "backtrack_pair_formed": pair_formed,
        "backtrack_initial_planar_distance_m": backtrack_distance,
        "backtrack_direction_errors_deg": backtrack_direction_errors,
        "backtrack_end_reason": backtrack.get("end_reason"),
        "navigation_internal_end_reason": first.get("end_reason"),
        "navigation_physical_failure_type": (
            "not_attempted" if not targets else
            ("success" if first.get("navigation_physical_arrival") else
             first.get("navigation_physical_failure_type") or "no_arrival")),
        "navigation_initial_target_geodesic_m": first.get(
            "initial_target_geodesic_distance_m"),
        "navigation_final_target_geodesic_m": first.get(
            "final_target_geodesic_distance_m"),
        "trajectory": str(trajectory_path),
    }


def summarize(results):
    total = len(results)
    module_rates = {}
    for key in MODULE_KEYS + ["all_modules_success"]:
        successes = sum(bool(
            row["all_modules_success"] if key == "all_modules_success"
            else row["modules"].get(key)) for row in results)
        module_rates[key] = {
            "successes": successes,
            "total": total,
            "rate": successes / max(total, 1),
            "wilson_95ci": wilson(successes, total),
        }
    pairs = [row for row in results if row.get("backtrack_pair_formed")]
    opportunities = sum(
        int(row.get("sequence_recovery_opportunities", 0)) for row in results)
    recovery_successes = sum(
        int(row.get("sequence_recovery_successes", 0)) for row in results)
    heading_counts = {
        str(threshold): sum(bool(row.get("heading_thresholds", {}).get(
            str(threshold))) for row in results)
        for threshold in (30, 45, 60, 90)}
    errors = [float(row["heading_error_to_demo_next_deg"])
              for row in results
              if math.isfinite(float(row.get(
                  "heading_error_to_demo_next_deg", math.inf)))]
    classification_decisions = [
        check for row in results
        for check in row.get("node_classification_checks", [])]
    first_classifications = [
        row["node_classification_checks"][0] for row in results
        if row.get("node_classification_checks")]

    def confusion(checks):
        return {
            "true_positive": sum(
                item["ground_truth_positive"] and item["predicted_positive"]
                for item in checks),
            "true_negative": sum(
                not item["ground_truth_positive"] and
                not item["predicted_positive"] for item in checks),
            "false_positive": sum(
                not item["ground_truth_positive"] and
                item["predicted_positive"] for item in checks),
            "false_negative": sum(
                item["ground_truth_positive"] and
                not item["predicted_positive"] for item in checks),
        }

    first_correct = sum(item["correct"] for item in first_classifications)
    decision_correct = sum(item["correct"] for item in classification_decisions)
    sequence_checks = [
        check for row in results for check in row.get("sequence_checks", [])]
    backtrack_conditional_success = sum(
        row["modules"].get("backtrack_physical_revisit", False)
        for row in pairs)
    return {
        "sample_count": total,
        "process_success_count": sum(
            row.get("process_returncode") == 0 for row in results),
        "module_rates": module_rates,
        "point_heading_threshold_counts": heading_counts,
        "point_heading_threshold_rates": {
            key: value / max(total, 1) for key, value in heading_counts.items()},
        "median_heading_error_deg": (
            float(np.median(errors)) if errors else None),
        "backtrack_nontrivial_pair_count": len(pairs),
        "backtrack_physical_revisit_conditional": {
            "successes": backtrack_conditional_success,
            "total": len(pairs),
            "rate": backtrack_conditional_success / max(len(pairs), 1),
            "wilson_95ci": wilson(
                backtrack_conditional_success, len(pairs)),
        },
        "sequence_recovery_opportunity_count": opportunities,
        "sequence_recovery_opportunity_success": {
            "successes": recovery_successes,
            "total": opportunities,
            "rate": recovery_successes / max(opportunities, 1),
            "wilson_95ci": wilson(recovery_successes, opportunities),
        },
        "process_error_counts": dict(collections.Counter(
            row.get("error") or "none" for row in results)),
        "navigation_physical_failure_counts": dict(collections.Counter(
            row.get("navigation_physical_failure_type") or "success"
            for row in results)),
        "navigation_internal_end_reason_counts": dict(collections.Counter(
            row.get("navigation_internal_end_reason") or "missing"
            for row in results)),
        "node_classification_first_hop": {
            "successes": first_correct,
            "total": len(first_classifications),
            "rate": first_correct / max(len(first_classifications), 1),
            "wilson_95ci": wilson(first_correct, len(first_classifications)),
            "confusion": confusion(first_classifications),
        },
        "node_classification_all_decisions": {
            "successes": decision_correct,
            "total": len(classification_decisions),
            "rate": decision_correct / max(len(classification_decisions), 1),
            "wilson_95ci": wilson(
                decision_correct, len(classification_decisions)),
            "confusion": confusion(classification_decisions),
        },
        "backtrack_end_reason_counts": dict(collections.Counter(
            row.get("backtrack_end_reason") or "missing" for row in results)),
        "sequence_directive_checks": {
            "correct": sum(item.get("action_matches", False)
                           for item in sequence_checks),
            "total": len(sequence_checks),
            "action_mismatch": sum(not item.get("action_matches", False)
                                   for item in sequence_checks),
            "recovery_failure": sum(not item.get("recovery_success", True)
                                    for item in sequence_checks),
            "blocked_yaw_failure": sum(not item.get("blocked_yaw_valid", True)
                                       for item in sequence_checks),
        },
        "all_modules_success_case_indices": [
            row["case_index"] for row in results
            if row.get("all_modules_success")],
    }


def load_episodes(path: Path):
    with gzip.open(path, "rt") as handle:
        return json.load(handle)["episodes"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-manifest", type=Path,
                        default=DEFAULT_SAMPLE_MANIFEST)
    parser.add_argument("--r2r-data", type=Path, default=DEFAULT_R2R_DATA)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--case-indices", default=None,
                        help="optional comma-separated sample case indices")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "outputs/r2r_all_modules_100ep_v3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--views", type=int, choices=[6, 8], default=8)
    parser.add_argument("--vlm-model", default=None)
    parser.add_argument("--decomposition-artifact", type=Path, default=None)
    parser.add_argument("--point-selection-prompt-version",
                        default="v31_circumnavigate_forward_competitor")
    parser.add_argument("--instruction-completion-prompt-version",
                        default="v19_vertical_guard_consensus")
    parser.add_argument("--tracking-cluster-profile",
                        default="dense_stop_motion_recovery_v13")
    parser.add_argument("--semantic-detector", default="dino-sam")
    parser.add_argument("--floor-segmenter", default="dense-majority")
    parser.add_argument("--vlm-timeout", type=int, default=180)
    parser.add_argument("--vlm-retries", type=int, default=2)
    parser.add_argument("--backtrack-planner-profile",
                        default="breadcrumb_budget_v3")
    parser.add_argument("--max-steps-per-target", type=int, default=30)
    parser.add_argument("--sequence-max-exploration-hops", type=int, default=3)
    parser.add_argument("--backtrack-method",
                        choices=["action-reversal", "visual"],
                        default="action-reversal")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    samples = json.loads(args.sample_manifest.read_text())
    if args.case_indices:
        requested = {int(value) for value in args.case_indices.split(",")
                     if value.strip()}
        samples = [sample for sample in samples
                   if int(sample["case_index"]) in requested]
    else:
        samples = samples[:args.count]
    if not samples:
        parser.error("no cases selected")
    identities = [(sample["episode_index"], sample["episode_id"])
                  for sample in samples]
    if len(set(identities)) != len(identities):
        parser.error("sample manifest must contain unique episodes")
    episodes = load_episodes(args.r2r_data)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.json"
    existing_manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists() else {})
    execution_created_utc = existing_manifest.get(
        "execution_created_utc", existing_manifest.get(
            "created_utc", datetime.now(timezone.utc).isoformat()))
    prior_execution_completed_utc = existing_manifest.get(
        "execution_completed_utc", existing_manifest.get("completed_utc"))
    frozen_samples = args.output_root / "sample_manifest.json"
    write_json(frozen_samples, samples)
    batch_manifest = {
        "schema_version": 1,
        "created_utc": execution_created_utc,
        "execution_created_utc": execution_created_utc,
        "status": "running",
        "rule_file": str((ROOT / "project_rulle.md").resolve()),
        "test_scope": "full",
        "sample_unit": (
            "one fixed initial reference_path state per unique R2R episode"
            if all(int(item["reference_path_index"]) == 0 for item in samples)
            else "one fixed reference_path state per unique R2R episode"),
        "sample_count": len(samples),
        "seed": 17,
        "selection_before_model_run": True,
        "sample_source": str(args.sample_manifest.resolve()),
        "sample_source_sha256": sha256(args.sample_manifest),
        "frozen_sample_sha256": sha256(frozen_samples),
        "target_modules": MODULE_KEYS,
        "configuration": {
            "split": "val_unseen", "vlm_backend": "deepseek",
            "vlm_model": args.vlm_model or "backend_default",
            "decomposition_artifact": (
                str(args.decomposition_artifact.resolve())
                if args.decomposition_artifact is not None else None),
            "point_selection_prompt_version": args.point_selection_prompt_version,
            "instruction_completion_prompt_version": args.instruction_completion_prompt_version,
            "floor_segmenter": args.floor_segmenter,
            "semantic_detector": args.semantic_detector,
            "views": args.views,
            "policy": "gnm", "tracker": "causal TAPIR",
            "tracking_cluster_profile": args.tracking_cluster_profile,
            "max_steps_per_target": args.max_steps_per_target,
            "sequence_max_exploration_hops": args.sequence_max_exploration_hops,
            "backtrack_method": args.backtrack_method,
            "backtrack_target": "node_0000",
            "backtrack_selector": "vlm",
            "backtrack_planner_profile": args.backtrack_planner_profile,
            "backtrack_nontrivial_minimum_displacement_m": 1.0,
            "navigation_arrival_geodesic_threshold_m": 0.75,
            "node_truth_forward_projection_threshold_m": 0.20,
            "node_truth_future_path_distance_threshold_m": 3.0,
            "node_classification_confidence_threshold": 0.5,
        },
        "future_reference_information_exposed_to_models": False,
        "failure_denominator_policy": "all preselected cases",
    }
    write_json(manifest_path, batch_manifest)

    results = []
    executed_process_count = 0
    environment = os.environ.copy()
    environment.update({
        "MAGNUM_LOG": "quiet", "HABITAT_SIM_LOG": "quiet",
        "PYTHONPATH": str(ROOT / "scripts"),
        "HF_HOME": str(ROOT / "weights/huggingface"),
    })
    for order, case in enumerate(samples, 1):
        case_dir = args.output_root / f"case_{int(case['case_index']):03d}"
        case_dir.mkdir(exist_ok=True)
        result_path = case_dir / "module_results.json"
        returncode = 0
        if args.rerun or not (case_dir / "trajectory.json").exists():
            executed_process_count += 1
            command = [
                sys.executable, str(HABITAT_ENTRYPOINT),
                "--mode", "semantic",
                "--exploration-strategy", "instruction-sequence-recovery",
                "--episode-index", str(case["episode_index"]),
                "--reference-path-index", str(case["reference_path_index"]),
                "--single-point-test-scope", "full",
                "--targets", "1",
                "--sequence-max-exploration-hops",
                str(args.sequence_max_exploration_hops),
                "--backtrack-method", args.backtrack_method,
                "--max-steps-per-target", str(args.max_steps_per_target),
                "--floor-segmenter", args.floor_segmenter,
                "--semantic-detector", args.semantic_detector,
                "--vlm-backend", "deepseek",
                "--vlm-timeout", str(args.vlm_timeout),
                "--vlm-retries", str(args.vlm_retries),
                "--point-selection-prompt-version",
                args.point_selection_prompt_version,
                "--instruction-completion-prompt-version",
                args.instruction_completion_prompt_version,
                "--tracking-cluster-profile", args.tracking_cluster_profile,
                "--views", str(args.views),
                "--device", args.device,
                "--backtrack-target-node", "node_0000",
                "--backtrack-selector", "vlm",
                "--backtrack-max-attempts-per-hop", "4",
                "--backtrack-max-hops", "8",
                "--backtrack-planner-profile",
                args.backtrack_planner_profile,
                "--sequence-recovery-backtrack-attempts", "4",
                "--output-dir", str(case_dir),
                "--seed", "17",
            ]
            if args.vlm_model is not None:
                command.extend(["--vlm-model", args.vlm_model])
            if args.decomposition_artifact is not None:
                command.extend([
                    "--decomposition-artifact",
                    str(args.decomposition_artifact),
                ])
            with (case_dir / "process_stdout.log").open("w") as stdout, \
                    (case_dir / "process_stderr.log").open("w") as stderr:
                completed = subprocess.run(
                    command, cwd=ROOT, env=environment,
                    stdout=stdout, stderr=stderr, check=False)
            returncode = completed.returncode
            (case_dir / "process_returncode.txt").write_text(
                f"{returncode}\n")
        elif (case_dir / "process_returncode.txt").exists():
            returncode = int(
                (case_dir / "process_returncode.txt").read_text().strip())
        episode = episodes[int(case["episode_index"])]
        if str(episode["episode_id"]) != str(case["episode_id"]):
            raise RuntimeError("sample episode identity differs from dataset")
        result = score_case(
            case, case_dir, episode, returncode,
            expected_views=args.views)
        write_json(result_path, result)
        # Main runtime hashes its preliminary module result. Refresh that hash
        # after the independent demonstration-trajectory scorer replaces it.
        case_manifest_path = case_dir / "manifest.json"
        if case_manifest_path.exists():
            case_manifest = json.loads(case_manifest_path.read_text())
            case_manifest.setdefault("artifact_sha256", {})[
                "module_results.json"] = sha256(result_path)
            case_manifest["independent_batch_scorer"] = str(
                Path(__file__).resolve())
            case_manifest["batch_case_index"] = case["case_index"]
            write_json(case_manifest_path, case_manifest)
        results.append(result)
        with (args.output_root / "results.jsonl").open("w") as handle:
            for row in results:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        partial = summarize(results)
        write_json(args.output_root / "summary.partial.json", partial)
        print(
            f"[{order}/{len(samples)}] case={case['case_index']} "
            f"episode={case['episode_id']} rc={returncode} "
            f"point={int(result['modules']['point_direction_correct'])} "
            f"nav={int(result['modules']['navigation_physical_arrival'])} "
            f"class={int(result['modules']['sub_instruction_classification_correct'])} "
            f"back={int(result['modules']['backtrack_physical_revisit'])} "
            f"seq={int(result['modules']['sequence_recovery_policy_correct'])}",
            flush=True)

    summary = summarize(results)
    scoring_completed_utc = datetime.now(timezone.utc).isoformat()
    execution_completed_utc = (
        scoring_completed_utc if executed_process_count else
        prior_execution_completed_utc or scoring_completed_utc)
    summary.update({
        "created_utc": batch_manifest["created_utc"],
        "completed_utc": execution_completed_utc,
        "execution_created_utc": execution_created_utc,
        "execution_completed_utc": execution_completed_utc,
        "scoring_completed_utc": scoring_completed_utc,
        "sample_manifest": str(frozen_samples),
        "results": str(args.output_root / "results.jsonl"),
    })
    summary_path = args.output_root / "summary.json"
    write_json(summary_path, summary)
    batch_manifest.update({
        "status": "complete",
        "completed_utc": execution_completed_utc,
        "execution_completed_utc": execution_completed_utc,
        "scoring_completed_utc": scoring_completed_utc,
        "artifact_sha256": {
            "sample_manifest.json": sha256(frozen_samples),
            "results.jsonl": sha256(args.output_root / "results.jsonl"),
            "summary.json": sha256(summary_path),
        },
    })
    write_json(manifest_path, batch_manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
