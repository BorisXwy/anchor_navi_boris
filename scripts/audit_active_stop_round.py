#!/usr/bin/env python3
"""Post-run audit for a fixed ten-episode Active-STOP curriculum round.

The reference trajectory is loaded only after an episode has finished.  It is
used for scoring selected rays and executed edges; none of the values produced
here are available to the online selector, executor, graph, or completion
judge.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path

import numpy as np


FIXED_EPISODES = (0, 3, 6, 9, 18, 27, 45, 126, 204, 219)
CONFIG_INVARIANTS = (
    "mode", "exploration_strategy", "policy", "device", "vlm_backend",
    "semantic_detector", "floor_segmenter", "views",
    "point_selection_prompt_version",
    "instruction_completion_prompt_version", "tracking_cluster_profile",
    "seed",
)


def _load_dataset(path: Path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        payload = json.load(handle)
    return payload["episodes"] if isinstance(payload, dict) else payload


def _polyline(points):
    points = np.asarray(points, np.float64)
    segments = points[1:] - points[:-1]
    lengths = np.linalg.norm(segments, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    return points, segments, lengths, cumulative


def _project_to_path(position, geometry):
    points, segments, lengths, cumulative = geometry
    position = np.asarray(position, np.float64)
    best = None
    for index, (start, vector, length) in enumerate(
            zip(points[:-1], segments, lengths)):
        if length <= 1e-8:
            fraction = 0.0
        else:
            fraction = float(np.clip(
                np.dot(position - start, vector) / (length * length), 0, 1))
        projected = start + fraction * vector
        distance = float(np.linalg.norm(position - projected))
        item = (distance, float(cumulative[index] + fraction * length),
                index, fraction, projected)
        if best is None or item[0] < best[0]:
            best = item
    if best is None:
        return {"distance_m": math.inf, "progress_m": 0.0,
                "segment_index": 0, "fraction": 0.0}
    return {"distance_m": best[0], "progress_m": best[1],
            "segment_index": best[2], "fraction": best[3]}


def _forward_tangent(position, geometry):
    projection = _project_to_path(position, geometry)
    _, segments, lengths, _ = geometry
    index = int(projection["segment_index"])
    while index < len(segments) and lengths[index] <= 1e-8:
        index += 1
    if index >= len(segments):
        return None
    tangent = np.asarray(segments[index], np.float64)[[0, 2]]
    norm = float(np.linalg.norm(tangent))
    return tangent / norm if norm > 1e-8 else None


def _point_at_progress(progress_m, geometry):
    points, segments, lengths, cumulative = geometry
    if not len(points):
        return None
    progress_m = float(np.clip(progress_m, 0.0, cumulative[-1]))
    for index, length in enumerate(lengths):
        if progress_m <= cumulative[index + 1] or index == len(lengths) - 1:
            if length <= 1e-8:
                return np.asarray(points[index + 1], np.float64)
            fraction = (progress_m - cumulative[index]) / length
            return np.asarray(
                points[index] + np.clip(fraction, 0.0, 1.0) * segments[index],
                np.float64)
    return np.asarray(points[-1], np.float64)


def _forward_reference_heading(position, geometry, lookahead_m):
    """Reference heading over the same horizon as one selected/executed edge.

    Point navigation targets often span an R2R polyline corner. Comparing the
    resulting chord to only the infinitesimal incoming segment labels a correct
    instructed turn as a 90-degree error. Use the demonstrated point the same
    travel horizon ahead, while retaining the final tangent when no reference
    path remains. This is post-run scoring only.
    """
    projection = _project_to_path(position, geometry)
    reference = _point_at_progress(
        projection["progress_m"] + max(float(lookahead_m), 0.05), geometry)
    if reference is not None:
        vector = (reference - np.asarray(position, np.float64))[[0, 2]]
        norm = float(np.linalg.norm(vector))
        if norm > 0.05:
            return vector / norm
    return _forward_tangent(position, geometry)


def _heading_error(vector_xyz, tangent):
    vector = np.asarray(vector_xyz, np.float64)[[0, 2]]
    norm = float(np.linalg.norm(vector))
    if tangent is None or norm <= 1e-8:
        return None
    cosine = float(np.clip(np.dot(vector / norm, tangent), -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def _target_end_position(target, start):
    steps = target.get("steps") or []
    return np.asarray(
        steps[-1].get("position_xyz", start) if steps else start,
        np.float64)


def audit_episode(episode_index, trajectory, dataset_episode,
                  trajectory_path=None):
    reference_path = dataset_episode.get("reference_path") or []
    geometry = _polyline(reference_path)
    start = np.asarray(trajectory["reference_state"]["position_xyz"], np.float64)
    targets = []
    for target in trajectory.get("targets", []):
        end = _target_end_position(target, start)
        selected = target.get("selected_navmesh_target_xyz")
        start_projection = _project_to_path(start, geometry)
        end_projection = _project_to_path(end, geometry)
        selected_horizon = (float(np.linalg.norm(
            (np.asarray(selected, np.float64) - start)[[0, 2]]))
            if selected is not None else 0.05)
        movement_horizon = float(np.linalg.norm((end - start)[[0, 2]]))
        selection_heading = _forward_reference_heading(
            start, geometry, selected_horizon)
        movement_heading = _forward_reference_heading(
            start, geometry, movement_horizon)
        selection_error = (_heading_error(
            np.asarray(selected, np.float64) - start, selection_heading)
            if selected is not None else None)
        movement_error = _heading_error(end - start, movement_heading)
        progress = float(end_projection["progress_m"] -
                         start_projection["progress_m"])
        net_motion = float(np.linalg.norm((end - start)[[0, 2]]))
        selection_forward = bool(
            selection_error is not None and selection_error <= 30.0)
        movement_forward = bool(
            movement_error is not None and movement_error <= 30.0 and
            progress > 0.05 and net_motion > 0.05)
        completion = target.get("instruction_completion") or {}
        targets.append({
            "target_index": int(target.get("target_index", len(targets))),
            "sub_instruction_id": int((target.get("sub_instruction") or {}).get(
                "sub_instruction_id", -1)),
            "form": str((target.get("sub_instruction") or {}).get(
                "form", "")),
            "selection_heading_error_to_gt_deg": selection_error,
            "executed_heading_error_to_gt_deg": movement_error,
            "gt_path_progress_delta_m": progress,
            "start_distance_to_gt_path_m": start_projection["distance_m"],
            "end_distance_to_gt_path_m": end_projection["distance_m"],
            "selection_within_30deg": selection_forward,
            "executed_edge_within_30deg_and_forward": movement_forward,
            "point_navigation_arrived": bool(target.get(
                "point_target_arrived", False)),
            "physical_point_reached": bool(target.get(
                "navigation_physical_arrival", False)),
            "node_created_after_arrival": bool(target.get(
                "node_created_after_point_arrival", False)),
            "all_navigation_cluster_points_on_ground": bool(target.get(
                "all_navigation_cluster_points_on_ground", False)),
            "all_stop_cluster_points_on_ground": bool(target.get(
                "all_stop_cluster_points_on_ground", False)),
            "system_completion": str(completion.get("status", "not_run")),
            "end_reason": str(target.get("end_reason", "")),
        })
        start = end

    sequence = trajectory.get("instruction_sequence_exploration") or {}
    metrics = trajectory.get("r2r_metrics") or {}
    stop = trajectory.get("task_stop") or {}
    video_record = trajectory.get("video") or {}
    video_path = (video_record.get("path") if isinstance(video_record, dict)
                  else video_record)
    stop_protocol_ok = bool(
        not stop.get("issued") or
        (stop.get("issued_after_final_sub_instruction_completion") and
         sequence.get("success") and
         metrics.get("complete_instruction_was_evaluated")))
    node_contract_ok = all(
        (not item["physical_point_reached"]) or
        item["node_created_after_arrival"] for item in targets)
    ground_contract_ok = all(
        item["all_navigation_cluster_points_on_ground"] and
        item["all_stop_cluster_points_on_ground"] for item in targets)
    stage_ids = [int(item.get(
        "stage_id", item.get("sub_instruction_id", index)))
        for index, item in enumerate(
            trajectory.get("instruction_stages") or [])]
    completed_stage_ids = set()
    aligned_completed_stage_ids = set()
    for raw_target, scored_target in zip(
            trajectory.get("targets", []), targets):
        completions = [raw_target.get("instruction_completion") or {}]
        chained = raw_target.get("chained_stop_wait_completion")
        if isinstance(chained, dict):
            completions.append(chained)
        for completion in completions:
            if not (completion.get("status") == "completed" and
                    completion.get("instruction_completed")):
                continue
            try:
                stage_id = int(completion["expected_sub_instruction_id"])
            except (KeyError, TypeError, ValueError):
                continue
            completed_stage_ids.add(stage_id)
            if (scored_target["selection_within_30deg"] and
                    scored_target[
                        "executed_edge_within_30deg_and_forward"]):
                aligned_completed_stage_ids.add(stage_id)
    verification_path = (
        Path(trajectory_path).parent / "stage_completion_verification.json"
        if trajectory_path is not None else None)
    independently_verified_stage_ids = set()
    if verification_path is not None and verification_path.is_file():
        try:
            verification = json.loads(verification_path.read_text())
            for item in verification.get("stages", []):
                if (item.get("semantic_completion_verified") is True and
                        item.get("ordered_stage_boundary_verified") is True and
                        item.get("verification_source") in {
                            "frozen_manual_edge_audit",
                            "independent_rgb_action_trajectory_audit",
                        } and item.get("evidence_artifacts")):
                    independently_verified_stage_ids.add(int(
                        item["sub_instruction_id"]))
        except (OSError, ValueError, TypeError, KeyError,
                json.JSONDecodeError):
            independently_verified_stage_ids = set()
    independent_stage_verification_valid = bool(
        stage_ids and set(stage_ids).issubset(
            independently_verified_stage_ids))
    completed_stage_edges_gt_aligned = bool(
        stage_ids and set(stage_ids).issubset(aligned_completed_stage_ids))
    instruction_validated_aligned_success = bool(
        metrics.get("simulator_reported_success", False) and
        sequence.get("success") and
        independent_stage_verification_valid and
        completed_stage_edges_gt_aligned)
    return {
        "episode_index": int(episode_index),
        "episode_id": trajectory.get("episode_id"),
        "sub_instructions_completed_online": int(
            sequence.get("completed_sub_instructions", 0)),
        "sub_instruction_count": len(trajectory.get("instruction_stages") or []),
        "exploration_hops": int(sequence.get("exploration_hops", 0)),
        "termination_reason": sequence.get("end_reason"),
        "stop_action_issued": bool(stop.get("issued", False)),
        "goal_radius_hit": bool(metrics.get("goal_radius_hit_diagnostic", False)),
        "simulator_reported_success": bool(metrics.get(
            "simulator_reported_success", False)),
        "final_geodesic_distance_m": metrics.get("final_geodesic_distance_m"),
        "video_exists": bool(video_path and Path(video_path).is_file()),
        "video_path": str(video_path or ""),
        "stop_protocol_ok": stop_protocol_ok,
        "node_contract_ok": node_contract_ok,
        "strict_ground_cluster_contract_ok": ground_contract_ok,
        "stage_completion_verification_path": (
            str(verification_path) if verification_path is not None else ""),
        "independently_verified_stage_ids": sorted(
            independently_verified_stage_ids),
        "independent_stage_verification_valid": (
            independent_stage_verification_valid),
        "completed_stage_ids": sorted(completed_stage_ids),
        "gt_aligned_completed_stage_ids": sorted(
            aligned_completed_stage_ids),
        "completed_stage_edges_gt_aligned": (
            completed_stage_edges_gt_aligned),
        "instruction_validated_aligned_success": (
            instruction_validated_aligned_success),
        "selected_targets_within_30deg": sum(
            item["selection_within_30deg"] for item in targets),
        "executed_edges_within_30deg_and_forward": sum(
            item["executed_edge_within_30deg_and_forward"]
            for item in targets),
        "target_count": len(targets),
        "targets": targets,
    }


def _failure_taxonomy(item):
    """Return round-level, protocol-aware failure labels for one episode."""
    if item.get("instruction_validated_aligned_success"):
        return []
    labels = []
    if item["stop_action_issued"] and not item["goal_radius_hit"]:
        labels.append("active_stop_outside_goal_radius")
    elif item["goal_radius_hit"] and not item["stop_action_issued"]:
        labels.append("goal_radius_hit_without_active_stop")
    elif not item["stop_action_issued"]:
        labels.append("instruction_sequence_not_completed")
    if item["sub_instructions_completed_online"] < item["sub_instruction_count"]:
        labels.append("online_subinstruction_prefix_incomplete")
    if (item["simulator_reported_success"] and not
            item.get("independent_stage_verification_valid")):
        labels.append("missing_or_failed_independent_stage_verification")
    if (item["simulator_reported_success"] and not
            item.get("completed_stage_edges_gt_aligned")):
        labels.append("completed_stage_edge_not_gt_aligned")
    termination = str(item.get("termination_reason") or "unknown")
    labels.append(f"termination:{termination}")
    if item["target_count"] and (
            item["selected_targets_within_30deg"] < item["target_count"]):
        labels.append("postrun_gt_selection_heading_misalignment")
    if item["target_count"] and (
            item["executed_edges_within_30deg_and_forward"] <
            item["target_count"]):
        labels.append("postrun_gt_executed_edge_misalignment_or_no_progress")
    return labels


def audit_round(round_root: Path, dataset_path: Path):
    trajectories = {}
    for path in round_root.rglob("trajectory.json"):
        trajectory = json.loads(path.read_text())
        reference = trajectory.get("reference_state") or {}
        if reference.get("path_index") is not None:
            continue
        try:
            index = int((trajectory.get("config") or {}).get(
                "episode_index", trajectory.get("episode_index")))
        except (TypeError, ValueError):
            continue
        if index in FIXED_EPISODES and index not in trajectories:
            trajectories[index] = (path, trajectory)
    dataset = _load_dataset(dataset_path)
    missing = [index for index in FIXED_EPISODES if index not in trajectories]
    results = []
    configs = []
    for index in FIXED_EPISODES:
        if index not in trajectories:
            continue
        trajectory_path, trajectory = trajectories[index]
        configs.append({key: (trajectory.get("config") or {}).get(key)
                        for key in CONFIG_INVARIANTS})
        result = audit_episode(
            index, trajectory, dataset[index], trajectory_path)
        result["trajectory_path"] = str(trajectory_path)
        results.append(result)
    frozen_config = configs[0] if configs else {}
    same_config = bool(configs and all(item == frozen_config for item in configs))
    cuda_ok = bool(configs and all(
        str(item.get("device", "")).startswith("cuda:") for item in configs))
    all_target_rows = [target for item in results for target in item["targets"]]
    audit = {
        "schema_version": 1,
        "audit_policy": {
            "reference_path_usage": "post_run_scoring_only",
            "reference_path_exposed_to_online_models": False,
            "selection_heading_threshold_deg": 30.0,
            "executed_edge_requires_positive_gt_progress": True,
            "success_definition": (
                "online final-stage completion -> active STOP -> STOP pose "
                "inside Habitat goal radius, with every completed stage "
                "independently verified and its executed edge within 30 "
                "degrees of forward GT progress"),
        },
        "round_root": str(round_root),
        "dataset": str(dataset_path),
        "fixed_episode_indices": list(FIXED_EPISODES),
        "missing_episode_indices": missing,
        "frozen_config": frozen_config,
        "protocol": {
            "all_ten_present": not missing and len(results) == 10,
            "same_frozen_config": same_config,
            "local_models_on_cuda": cuda_ok,
            "all_videos_present": bool(results and all(
                item["video_exists"] for item in results)),
            "stop_semantics_valid": bool(results and all(
                item["stop_protocol_ok"] for item in results)),
            "arrival_node_contract_valid": bool(results and all(
                item["node_contract_ok"] for item in results)),
            "strict_ground_cluster_contract_valid": bool(results and all(
                item["strict_ground_cluster_contract_ok"] for item in results)),
            "success_claims_have_independent_stage_verification": bool(
                results and all(
                    not item["simulator_reported_success"] or
                    item["independent_stage_verification_valid"]
                    for item in results)),
            "success_claims_have_gt_aligned_completed_edges": bool(
                results and all(
                    not item["simulator_reported_success"] or
                    item["completed_stage_edges_gt_aligned"]
                    for item in results)),
        },
        "totals": {
            "episodes": len(results),
            "simulator_reported_success": sum(
                item["simulator_reported_success"] for item in results),
            "instruction_validated_aligned_success": sum(
                item["instruction_validated_aligned_success"]
                for item in results),
            "stop_actions": sum(item["stop_action_issued"] for item in results),
            "goal_radius_hits": sum(item["goal_radius_hit"] for item in results),
            "targets": len(all_target_rows),
            "selection_within_30deg": sum(
                item["selection_within_30deg"] for item in all_target_rows),
            "executed_edge_within_30deg_and_forward": sum(
                item["executed_edge_within_30deg_and_forward"]
                for item in all_target_rows),
        },
        "episodes": results,
    }
    audit["final_gate_passed"] = bool(
        all(audit["protocol"].values()) and
        audit["totals"]["simulator_reported_success"] == 10 and
        audit["totals"]["instruction_validated_aligned_success"] == 10)
    return audit


def _summary(audit):
    return {
        "schema_version": 1,
        "round_root": audit["round_root"],
        "fixed_episode_indices": audit["fixed_episode_indices"],
        "missing_episode_indices": audit["missing_episode_indices"],
        "success_definition": audit["audit_policy"]["success_definition"],
        "totals": audit["totals"],
        "protocol": audit["protocol"],
        "final_gate_passed": audit["final_gate_passed"],
        "episodes": [{
            "episode_index": item["episode_index"],
            "completed_stages": item["sub_instructions_completed_online"],
            "total_stages": item["sub_instruction_count"],
            "hops": item["exploration_hops"],
            "stop_action_issued": item["stop_action_issued"],
            "goal_radius_hit": item["goal_radius_hit"],
            "simulator_reported_success": item["simulator_reported_success"],
            "instruction_validated_aligned_success": item[
                "instruction_validated_aligned_success"],
            "final_geodesic_distance_m": item["final_geodesic_distance_m"],
            "termination_reason": item["termination_reason"],
            "trajectory_path": item["trajectory_path"],
            "video_path": item["video_path"],
        } for item in audit["episodes"]],
    }


def _failure_cases(audit):
    cases = []
    for item in audit["episodes"]:
        labels = _failure_taxonomy(item)
        if not labels:
            continue
        cases.append({
            "episode_index": item["episode_index"],
            "failure_labels": labels,
            "completed_stages": item["sub_instructions_completed_online"],
            "total_stages": item["sub_instruction_count"],
            "hops": item["exploration_hops"],
            "final_geodesic_distance_m": item["final_geodesic_distance_m"],
            "trajectory_path": item["trajectory_path"],
            "video_path": item["video_path"],
        })
    return {
        "schema_version": 1,
        "postrun_gt_diagnostics_only": True,
        "failure_count": len(cases),
        "cases": cases,
    }


def _failure_markdown(failures):
    lines = ["# Active-STOP failure cases", "",
             "GT alignment labels below are post-run diagnostics only.", "",
             "| EP | stages | hops | final distance | labels |",
             "|---:|---:|---:|---:|---|"]
    for item in failures["cases"]:
        lines.append(
            f"| {item['episode_index']} | {item['completed_stages']}/"
            f"{item['total_stages']} | {item['hops']} | "
            f"{item['final_geodesic_distance_m']} | "
            f"{', '.join(item['failure_labels'])} |")
    lines.append("")
    return "\n".join(lines)


def _markdown(audit):
    totals = audit["totals"]
    lines = [
        "# Active-STOP fixed-ten round audit", "",
        (f"Simulator active-STOP success: "
         f"{totals['simulator_reported_success']}/{totals['episodes']}."),
        (f"Instruction-verified and GT-aligned success: "
         f"{totals['instruction_validated_aligned_success']}/"
         f"{totals['episodes']}."),
        (f"Post-run GT alignment: selection {totals['selection_within_30deg']}"
         f"/{totals['targets']}; executed forward edges "
         f"{totals['executed_edge_within_30deg_and_forward']}"
         f"/{totals['targets']}."), "",
        "| EP | stages | hops | STOP | goal | sim success | select <=30 | "
        "edge <=30 + progress | verified+aligned | termination |", 
        "|---:|---:|---:|:---:|:---:|:---:|---:|---:|:---:|---|",
    ]
    for item in audit["episodes"]:
        lines.append(
            f"| {item['episode_index']} | "
            f"{item['sub_instructions_completed_online']}/"
            f"{item['sub_instruction_count']} | {item['exploration_hops']} | "
            f"{'Y' if item['stop_action_issued'] else 'N'} | "
            f"{'Y' if item['goal_radius_hit'] else 'N'} | "
            f"{'Y' if item['simulator_reported_success'] else 'N'} | "
            f"{item['selected_targets_within_30deg']}/{item['target_count']} | "
            f"{item['executed_edges_within_30deg_and_forward']}/"
            f"{item['target_count']} | "
            f"{'Y' if item['instruction_validated_aligned_success'] else 'N'} | "
            f"{item['termination_reason']} |")
    lines.extend(["", "## Protocol invariants", ""])
    for key, value in audit["protocol"].items():
        lines.append(f"- {key}: {'PASS' if value else 'FAIL'}")
    lines.extend(["", f"Final 10/10 gate: "
                  f"{'PASS' if audit['final_gate_passed'] else 'FAIL'}", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("round_root", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    audit = audit_round(args.round_root, args.dataset)
    json_path = args.json_output or args.round_root / "protocol_audit.json"
    markdown_path = (args.markdown_output or
                     args.round_root / "round_report.md")
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    markdown_path.write_text(_markdown(audit))
    summary_path = args.round_root / "summary.json"
    failures_path = args.round_root / "failure_cases.json"
    failures_md_path = args.round_root / "failure_cases.md"
    summary_path.write_text(json.dumps(
        _summary(audit), ensure_ascii=False, indent=2) + "\n")
    failures = _failure_cases(audit)
    failures_path.write_text(json.dumps(
        failures, ensure_ascii=False, indent=2) + "\n")
    failures_md_path.write_text(_failure_markdown(failures))
    with (args.round_root / "process.log").open("a") as handle:
        handle.write(
            f"{datetime.now(timezone.utc).isoformat()} post_run_audit "
            f"episodes={audit['totals']['episodes']} "
            f"strict_success={audit['totals']['simulator_reported_success']} "
            f"final_gate_passed={audit['final_gate_passed']}\n")
    print(json.dumps(audit["totals"], ensure_ascii=False))
    print(json_path)
    print(markdown_path)
    print(summary_path)
    print(failures_path)


if __name__ == "__main__":
    main()
