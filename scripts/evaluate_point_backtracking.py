#!/usr/bin/env python3
"""Evaluate physical node backtracking on frozen real R2R single-point states."""

from __future__ import annotations

import argparse
import collections
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
HABITAT_ENTRYPOINT = ROOT / "scripts/habitat_point_navigation.py"
DEFAULT_R2R_DATA = (
    ROOT.parent / "3d_wm_vln/StreamVLN/data/datasets/r2r/val_unseen/"
    "val_unseen.json.gz")
DEFAULT_SAMPLES = ROOT / "data/backtrack_development_samples_v1.json"


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def angular_error_deg(first, second) -> float:
    if first is None or second is None:
        return math.inf
    difference = (float(first) - float(second) + math.pi) % (2 * math.pi) - math.pi
    return abs(math.degrees(difference))


def load_episodes(path: Path):
    with gzip.open(path, "rt") as handle:
        return json.load(handle)["episodes"]


def graph_maps(graph):
    return (
        {node["node_id"]: node for node in graph.get("nodes", [])},
        {edge["edge_id"]: edge for edge in graph.get("edges", [])},
    )


def failure_result(case, case_dir, returncode, reason):
    return {
        "case_index": case["case_index"],
        "episode_index": case["episode_index"],
        "episode_id": case["episode_id"],
        "scene_id": case["scene_id"],
        "reference_path_index": case["reference_path_index"],
        "process_returncode": returncode,
        "error": reason,
        "pair_formed": False,
        "backtrack_selection_valid": False,
        "backtrack_physical_revisit": False,
        "loop_closure_valid": False,
        "functional_graph_invariants": False,
        "failure_category": "process_or_artifact_failure",
        "trajectory": str(case_dir / "trajectory.json"),
    }


def score_case(case, case_dir: Path, returncode=0):
    trajectory_path = case_dir / "trajectory.json"
    graph_path = case_dir / "navigation_graph/navigation_graph.json"
    if returncode != 0 or not trajectory_path.exists() or not graph_path.exists():
        return failure_result(
            case, case_dir, returncode,
            "episode_process_failed_or_required_artifact_missing")

    trajectory = json.loads(trajectory_path.read_text())
    graph = json.loads(graph_path.read_text())
    nodes, edges = graph_maps(graph)
    targets = trajectory.get("targets") or []
    forward = targets[0] if targets else {}
    backtrack = trajectory.get("node_backtracking") or {}
    source_node = nodes.get(backtrack.get("source_node_id"))
    target_node = nodes.get(backtrack.get("target_node_id"))
    if source_node is not None and target_node is not None:
        source_position = source_node["position_xyz"]
        target_position = target_node["position_xyz"]
        initial_planar_distance = math.hypot(
            float(source_position[0]) - float(target_position[0]),
            float(source_position[2]) - float(target_position[2]))
    else:
        initial_planar_distance = 0.0
    pair_formed = initial_planar_distance >= 1.0

    attempts = backtrack.get("attempts") or []
    executed = [item for item in attempts
                if item.get("executed_point_navigation")]
    direction_errors = []
    for attempt in executed:
        selection = attempt.get("selection") or {}
        desired = selection.get(
            "desired_breadcrumb_bearing_yaw_rad",
            selection.get("desired_target_bearing_yaw_rad"))
        direction_errors.append(angular_error_deg(
            selection.get("selected_yaw_rad"), desired))
    first_direction_error = (
        direction_errors[0] if direction_errors else math.inf)
    selection_valid = bool(
        pair_formed and executed and first_direction_error < 90.0)

    revisit_attempts = [
        item for item in attempts if item.get("node_revisit_match")]
    successful_revisit = next((
        item for item in revisit_attempts
        if (item.get("node_revisit_match") or {}).get("reached")), None)
    loop_edge_id = (
        successful_revisit.get("loop_closure_edge_id")
        if successful_revisit else None)
    loop_closure_valid = bool(
        pair_formed and loop_edge_id in edges and
        edges[loop_edge_id].get("edge_kind") == "node_revisit_loop_closure")
    physical_revisit = bool(
        pair_formed and backtrack.get("success") and executed and
        successful_revisit is not None and loop_closure_valid)

    graph_invariants = bool(forward)
    forward_edge = edges.get(forward.get("navigation_graph_edge_id"))
    graph_invariants = graph_invariants and bool(
        forward_edge is not None and
        forward_edge.get("action_history") == forward.get("action_history", []))
    for attempt in executed:
        edge = edges.get(attempt.get("created_edge_id"))
        graph_invariants = graph_invariants and bool(
            edge is not None and
            edge.get("action_history") == attempt.get("action_history", []))

    last_match = (
        revisit_attempts[-1].get("node_revisit_match")
        if revisit_attempts else {}) or {}
    if not pair_formed:
        failure_category = "upstream_pair_not_formed"
    elif physical_revisit:
        failure_category = "success"
    elif not attempts or all(item.get("selection") is None for item in attempts):
        failure_category = "backtrack_selection_failed"
    elif direction_errors and first_direction_error >= 90.0:
        failure_category = "initial_wrong_direction"
    elif (last_match.get("planar_distance_m") is not None and
          float(last_match["planar_distance_m"]) <= 0.75 and
          float(last_match.get("visual_similarity", -1)) < 0.75):
        failure_category = "visual_revisit_mismatch"
    elif executed:
        failure_category = "physical_revisit_not_reached"
    else:
        failure_category = "backtrack_not_executed"

    result = {
        "case_index": case["case_index"],
        "episode_index": case["episode_index"],
        "episode_id": case["episode_id"],
        "scene_id": case["scene_id"],
        "trajectory_id": case.get("trajectory_id"),
        "reference_path_index": case["reference_path_index"],
        "process_returncode": returncode,
        "error": None,
        "pair_formed": pair_formed,
        "pair_initial_planar_distance_m": initial_planar_distance,
        "forward_executor_arrived": bool(forward.get("arrived")),
        "forward_executor_end_reason": forward.get("end_reason"),
        "forward_action_count": len(forward.get("action_history") or []),
        "backtrack_selection_valid": selection_valid,
        "backtrack_first_direction_error_deg": first_direction_error,
        "backtrack_direction_errors_deg": direction_errors,
        "backtrack_physical_revisit": physical_revisit,
        "loop_closure_valid": loop_closure_valid,
        "functional_graph_invariants": graph_invariants,
        "attempt_count": len(attempts),
        "executed_attempt_count": len(executed),
        "executor_end_reasons": [
            item.get("executor_end_reason") for item in attempts],
        "final_revisit_match": last_match,
        "backtrack_end_reason": backtrack.get("end_reason"),
        "failure_category": failure_category,
        "trajectory": str(trajectory_path),
        "video": str(case_dir / Path((trajectory.get("video") or {}).get(
            "path", "exploration.mp4")).name),
    }
    return result


def rate(successes, total):
    return {
        "successes": successes,
        "total": total,
        "rate": successes / max(total, 1),
        "wilson_95ci": wilson(successes, total),
    }


def summarize(results, configuration):
    total = len(results)
    pairs = [item for item in results if item.get("pair_formed")]
    physical = sum(item.get("backtrack_physical_revisit", False)
                   for item in results)
    physical_conditional = sum(
        item.get("backtrack_physical_revisit", False) for item in pairs)
    selection = sum(item.get("backtrack_selection_valid", False)
                    for item in results)
    closure = sum(item.get("loop_closure_valid", False) for item in results)
    graph_valid = sum(item.get("functional_graph_invariants", False)
                      for item in results)
    attempt_direction_errors = [
        error for item in results
        for error in item.get("backtrack_direction_errors_deg", [])
        if math.isfinite(error)]
    return {
        "sample_count": total,
        "configuration": configuration,
        "process_success": rate(sum(
            item.get("process_returncode") == 0 for item in results), total),
        "upstream_pair_formation": rate(len(pairs), total),
        "backtrack_selection_valid": rate(selection, total),
        "backtrack_physical_revisit": rate(physical, total),
        "backtrack_physical_revisit_given_pair": rate(
            physical_conditional, len(pairs)),
        "loop_closure_valid": rate(closure, total),
        "functional_graph_invariants": rate(graph_valid, total),
        "attempt_direction_lt_90": rate(sum(
            error < 90.0 for error in attempt_direction_errors),
            len(attempt_direction_errors)),
        "failure_category_counts": dict(collections.Counter(
            item.get("failure_category", "missing") for item in results)),
        "backtrack_end_reason_counts": dict(collections.Counter(
            item.get("backtrack_end_reason") or "missing" for item in results)),
        "executor_end_reason_counts": dict(collections.Counter(
            reason for item in results
            for reason in item.get("executor_end_reasons", []))),
        "total_backtrack_attempts": sum(
            item.get("attempt_count", 0) for item in results),
        "results": results,
        "metric_definitions": {
            "pair_formed": "real forward stop is at least 1.0 m in XZ from origin",
            "backtrack_selection_valid": (
                "pair formed and first executed backtrack selection is <90 deg "
                "from its online graph/breadcrumb bearing"),
            "backtrack_physical_revisit": (
                "all preselected cases denominator; requires pair, <=0.75 m "
                "planar distance, >=0.75 visual similarity, physical execution, "
                "and valid loop closure"),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-manifest", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--r2r-data", type=Path, default=DEFAULT_R2R_DATA)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--selector-mode", choices=["vlm", "hybrid"], default="vlm")
    parser.add_argument("--planner-profile", default="legacy_direct")
    parser.add_argument("--max-steps-per-target", type=int, default=32)
    parser.add_argument("--max-attempts-per-hop", type=int, default=4)
    parser.add_argument("--vlm-model", default="deepseek-v4-flash-vision-exp")
    parser.add_argument("--case-indices", default=None)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    samples = json.loads(args.sample_manifest.read_text())
    if args.case_indices:
        requested = {int(item) for item in args.case_indices.split(",") if item}
        samples = [item for item in samples if int(item["case_index"]) in requested]
    if not samples:
        parser.error("no frozen samples selected")
    episodes = load_episodes(args.r2r_data)
    for sample in samples:
        episode = episodes[int(sample["episode_index"])]
        if (str(episode.get("episode_id")) != str(sample.get("episode_id")) or
                str(episode.get("trajectory_id")) !=
                str(sample.get("trajectory_id"))):
            raise RuntimeError(f"frozen identity mismatch for case {sample['case_index']}")

    configuration = {
        "test_scope": "module",
        "target_modules": ["physical_node_backtracking", "loop_closure"],
        "required_upstream_modules": [
            "instruction_decomposition", "six_view_ground_perception",
            "vlm_point_selection", "point_navigation_executor",
            "navigation_graph_node_and_edge",
        ],
        "not_run_modules": [
            "instruction_completion", "instruction_sequence_recovery"],
        "selector_mode": args.selector_mode,
        "planner_profile": args.planner_profile,
        "policy": "gnm",
        "tracker": "causal TAPIR dense_stop_motion_guard_v7",
        "floor_segmenter": "Dense 2/3 majority (OneFormer + Mask2Former + SegFormer)",
        "vlm_backend": "deepseek",
        "vlm_model": args.vlm_model,
        "max_steps_per_target": args.max_steps_per_target,
        "max_attempts_per_hop": args.max_attempts_per_hop,
        "reach_radius_m": 0.75,
        "minimum_visual_similarity": 0.75,
        "seed": 17,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    frozen_samples = args.output_root / "sample_manifest.json"
    if frozen_samples.exists():
        if json.loads(frozen_samples.read_text()) != samples:
            raise RuntimeError("refusing to change frozen samples in an existing run")
    else:
        write_json(frozen_samples, samples)
    manifest_path = args.output_root / "manifest.json"
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "rule_file": str((ROOT / "project_rulle.md").resolve()),
        "selection_fixed_before_model_run": True,
        "future_reference_information_exposed_to_models": False,
        "failure_denominator_policy": "all preselected cases",
        "sample_source": str(args.sample_manifest.resolve()),
        "sample_source_sha256": sha256(args.sample_manifest),
        "frozen_sample_sha256": sha256(frozen_samples),
        "r2r_dataset_sha256": sha256(args.r2r_data),
        "sample_count": len(samples),
        "configuration": configuration,
    }
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if (old.get("frozen_sample_sha256") != manifest["frozen_sample_sha256"] or
                old.get("configuration") != configuration):
            raise RuntimeError("refusing to change frozen run configuration")
        manifest["created_utc"] = old["created_utc"]
    write_json(manifest_path, manifest)

    environment = os.environ.copy()
    environment.update({
        "MAGNUM_LOG": "quiet", "HABITAT_SIM_LOG": "quiet",
        "HF_HOME": str(ROOT / "weights/huggingface"),
        "PYTHONPATH": str(ROOT / "scripts"),
    })
    results = []
    for order, case in enumerate(samples, 1):
        case_dir = args.output_root / f"case_{int(case['case_index']):03d}"
        case_dir.mkdir(exist_ok=True)
        trajectory_path = case_dir / "trajectory.json"
        returncode = 0
        if args.rerun or not trajectory_path.exists():
            command = [
                sys.executable, str(HABITAT_ENTRYPOINT),
                "--mode", "semantic",
                "--exploration-strategy", "standard",
                "--episode-index", str(case["episode_index"]),
                "--reference-path-index", str(case["reference_path_index"]),
                "--single-point-test-scope", "module",
                "--single-point-target-modules",
                "physical_node_backtracking,loop_closure",
                "--skip-instruction-completion",
                "--targets", "1",
                "--max-steps-per-target", str(args.max_steps_per_target),
                "--floor-segmenter", "dense-majority",
                "--tracking-cluster-profile", "dense_stop_motion_guard_v7",
                "--vlm-backend", "deepseek",
                "--vlm-model", args.vlm_model,
                "--point-selection-prompt-version", "v3_orientation_soft_semantic",
                "--instruction-completion-prompt-version",
                "v7_structured_partial_extent",
                "--device", args.device,
                "--backtrack-target-node", "node_0000",
                "--backtrack-selector", args.selector_mode,
                "--backtrack-max-attempts-per-hop",
                str(args.max_attempts_per_hop),
                "--backtrack-max-hops", "1",
                "--output-dir", str(case_dir),
                "--seed", "17",
            ]
            # Profiles beyond legacy are introduced by the backtracking module;
            # keeping the legacy command compatible makes the baseline auditable.
            if args.planner_profile != "legacy_direct":
                command.extend([
                    "--backtrack-planner-profile", args.planner_profile])
            with (case_dir / "process.log").open("w") as process_log:
                completed = subprocess.run(
                    command, cwd=ROOT, env=environment, check=False,
                    stdout=process_log, stderr=subprocess.STDOUT)
            returncode = completed.returncode
            (case_dir / "process_returncode.txt").write_text(
                f"{returncode}\n")
        elif (case_dir / "process_returncode.txt").exists():
            returncode = int(
                (case_dir / "process_returncode.txt").read_text().strip())

        result = score_case(case, case_dir, returncode)
        write_json(case_dir / "module_results.json", result)
        case_manifest_path = case_dir / "manifest.json"
        if case_manifest_path.exists():
            case_manifest = json.loads(case_manifest_path.read_text())
            case_manifest.update({
                "batch_evaluator": str(Path(__file__).resolve()),
                "batch_manifest": str(manifest_path.resolve()),
                "module_results_sha256": sha256(
                    case_dir / "module_results.json"),
            })
            write_json(case_manifest_path, case_manifest)
        results.append(result)
        summary = summarize(results, configuration)
        write_json(args.output_root / "summary.partial.json", summary)
        print(
            f"[{order}/{len(samples)}] case={case['case_index']} "
            f"pair={int(result.get('pair_formed', False))} "
            f"select={int(result.get('backtrack_selection_valid', False))} "
            f"revisit={int(result.get('backtrack_physical_revisit', False))} "
            f"reason={result.get('failure_category')}", flush=True)

    summary = summarize(results, configuration)
    summary["completed_utc"] = datetime.now(timezone.utc).isoformat()
    summary["sample_manifest"] = str(frozen_samples)
    summary_path = args.output_root / "summary.json"
    write_json(summary_path, summary)
    manifest.update({
        "status": "complete",
        "completed_utc": summary["completed_utc"],
        "artifact_sha256": {
            "sample_manifest.json": sha256(frozen_samples),
            "summary.json": sha256(summary_path),
        },
    })
    write_json(manifest_path, manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
