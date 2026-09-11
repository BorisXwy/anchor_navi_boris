#!/usr/bin/env python3
"""Evaluate the modular Habitat point-navigation pipeline on N R2R episodes."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from node_backtracking import BACKTRACK_PLANNER_PROFILES
from r2r_stop_success import simulator_stop_success
from audit_rgb_only_contract import audit_run


ROOT = Path(__file__).resolve().parents[1]
HABITAT_ENTRYPOINT = ROOT / "scripts/habitat_point_navigation.py"
DEFAULT_R2R_DATA = (
    ROOT.parent / "3d_wm_vln/StreamVLN/data/datasets/r2r/val_unseen/"
    "val_unseen.json.gz")


def provider_fatal_output(output):
    """Detect a child-process VLM provider failure that invalidates a shard."""
    text = str(output or "")
    explicit_marker = "VLMProviderFatalError"
    deepseek_fatal_codes = (
        "DeepSeek HTTP 401", "DeepSeek HTTP 402", "DeepSeek HTTP 403")
    missing_key = "DeepSeek API key is missing"
    exhausted_network_provider = bool(
        "VLM harness exhausted retries" in text and
        "DeepSeek request failed" in text and
        any(marker in text.lower() for marker in (
            "connection refused", "proxyerror", "timed out",
            "temporary failure in name resolution", "network is unreachable")))
    return bool(
        explicit_marker in text or missing_key in text or
        any(marker in text for marker in deepseek_fatal_codes) or
        exhausted_network_provider)


def instruction_validated_goal_metrics(
        trajectory, metrics, sequence, verified_sequence_complete=False):
    """Return strict task success; geometric goal hits are diagnostic only."""
    goal_radius_hit = bool(metrics.get(
        "goal_radius_hit_diagnostic", metrics.get("success", False)))
    all_sub_instructions = trajectory.get(
        "all_decomposed_sub_instructions") or []
    evaluated_sub_instructions = trajectory.get("sub_instructions") or []
    from_episode_start = (
        (trajectory.get("reference_state") or {}).get("path_index") is None)
    complete_instruction_was_evaluated = bool(
        from_episode_start and all_sub_instructions and
        len(evaluated_sub_instructions) == len(all_sub_instructions))
    sequence_completed_in_order = bool(sequence.get("success"))
    stop_action_issued = bool(metrics.get(
        "stop_action_issued",
        (trajectory.get("task_stop") or {}).get("issued", False)))
    strict_success = bool(
        goal_radius_hit and complete_instruction_was_evaluated and
        sequence_completed_in_order and stop_action_issued and
        verified_sequence_complete)
    return {
        "goal_radius_hit_diagnostic": goal_radius_hit,
        "stop_action_issued": stop_action_issued,
        "invalid_goal_radius_hit": bool(goal_radius_hit and not strict_success),
        "instruction_validated_r2r_success": strict_success,
        "complete_instruction_was_evaluated": complete_instruction_was_evaluated,
        "sequence_completed_in_order": sequence_completed_in_order,
        "independent_semantic_verification_complete": bool(
            verified_sequence_complete),
    }


def simulator_stop_goal_metrics(trajectory, metrics):
    """Read Habitat-style success: active STOP must occur inside goal radius."""
    task_stop = trajectory.get("task_stop") or {}
    stop_issued = bool(metrics.get(
        "stop_action_issued", task_stop.get("issued", False)))
    goal_radius_hit = bool(metrics.get(
        "goal_radius_hit_diagnostic", metrics.get("success", False)))
    return {
        "stop_action_issued": stop_issued,
        "simulator_reported_success": simulator_stop_success(
            stop_action_issued=stop_issued,
            goal_radius_hit=goal_radius_hit),
    }


def instruction_validated_spl(metrics, strict_success):
    """Compute SPL from the frozen path metrics after strict verification.

    The trajectory-level SPL may have been written before a post-run semantic
    audit existed, so reusing it would leave a verified success with SPL zero.
    """
    if not strict_success:
        return 0.0
    shortest = float(metrics.get("initial_geodesic_distance_m") or 0.0)
    traveled = float(metrics.get("path_length_m") or 0.0)
    if shortest <= 0.0:
        return 0.0
    return shortest / max(shortest, traveled)


def strict_stage_completion_metrics(trajectory, targets, audit_path):
    """Intersect online node decisions with independent semantic audits.

    Missing, malformed, or incomplete audit evidence fails closed.  The audit
    is post-run scoring evidence and is never exposed to navigation models.
    """
    evaluated = trajectory.get("sub_instructions") or []
    stage_ids = [int(item.get(
        "stage_id", item.get("sub_instruction_id", index)))
                 for index, item in enumerate(evaluated)]
    system_completed = set()
    for target in targets:
        node_was_built = bool(
            target.get("node_created_after_point_arrival") and
            target.get("navigation_graph_node_id") and
            target.get("navigation_graph_edge_id"))
        physically_arrived = bool(target.get(
            "point_target_arrived", target.get("arrived", False)))
        completions = [target.get("instruction_completion") or {}]
        chained = target.get("chained_stop_wait_completion")
        if isinstance(chained, dict):
            completions.append(chained)
        for completion in completions:
            node_judgment_completed = bool(
                completion.get("status") == "completed" and
                completion.get("instruction_completed") and
                completion.get("current_node_id") and
                completion.get("incoming_edge_id"))
            if (node_judgment_completed and node_was_built and
                    physically_arrived):
                try:
                    system_completed.add(int(
                        completion["expected_sub_instruction_id"]))
                except (KeyError, TypeError, ValueError):
                    continue

    verified = set()
    audit_status = "missing"
    if audit_path.exists():
        try:
            audit = json.loads(audit_path.read_text())
            audit_status = "loaded"
            for item in audit.get("stages", []):
                evidence = item.get("evidence_artifacts") or []
                if (item.get("semantic_completion_verified") is True and
                        item.get("ordered_stage_boundary_verified") is True and
                        item.get("verification_source") in {
                            "frozen_manual_edge_audit",
                            "independent_rgb_action_trajectory_audit",
                        } and evidence):
                    verified.add(int(item["sub_instruction_id"]))
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            audit_status = "invalid"

    jointly_passed = system_completed & verified
    ordered_prefix = 0
    for stage_id in stage_ids:
        if stage_id not in jointly_passed:
            break
        ordered_prefix += 1
    return {
        "audit_path": str(audit_path),
        "audit_status": audit_status,
        "system_node_completed_stage_ids": sorted(system_completed),
        "independently_verified_stage_ids": sorted(verified),
        "jointly_passed_stage_ids": sorted(jointly_passed),
        "verified_ordered_prefix_count": ordered_prefix,
        "all_evaluated_stages_jointly_passed": bool(
            stage_ids and ordered_prefix == len(stage_ids)),
    }


def episode_indices(explicit, start, count):
    if explicit:
        return [int(value) for value in explicit.split(",") if value.strip()]
    if count <= 0:
        raise ValueError("--num-episodes must be positive")
    return list(range(start, start + count))


def termination_category(reason):
    reason = str(reason or "")
    if reason == "instruction_sequence_complete":
        return "instruction_sequence_complete"
    if reason == "sequence_recovery_backtrack_failed":
        return "sequence_recovery_backtrack_failed"
    if reason.startswith("vlm_selection_failed:"):
        if "No floor-bearing candidate" in reason:
            return "no_floor_bearing_candidate"
        if "did not return JSON" in reason:
            return "vlm_invalid_json"
        return "vlm_selection_failed_other"
    return reason or "unknown"


def freeze_episode_manifest(output_root, dataset_path, indices, configuration):
    """Freeze the episode set before any model call or simulator execution."""
    dataset_path = Path(dataset_path).resolve()
    with gzip.open(dataset_path, "rt") as handle:
        episodes = json.load(handle)["episodes"]
    selected = []
    for index in indices:
        if not 0 <= index < len(episodes):
            raise IndexError(
                f"episode index {index} is outside [0, {len(episodes) - 1}]")
        episode = episodes[index]
        if not episode.get("reference_path"):
            raise ValueError(f"episode index {index} has no reference_path")
        selected.append({
            "episode_index": index,
            "episode_id": episode.get("episode_id"),
            "trajectory_id": episode.get("trajectory_id"),
            "scene_id": episode.get("scene_id"),
            "start_position": episode.get("start_position"),
            "start_rotation": episode.get("start_rotation"),
            "reference_path_length": len(episode["reference_path"]),
            "instruction": (episode.get("instruction") or {}).get(
                "instruction_text", ""),
        })
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "R2R",
        "split": dataset_path.stem.replace(".json", ""),
        "test_scope": (
            f"{len(selected)}_episode_end_to_end_from_dataset_start"),
        "selection_rule": (
            "indices fixed from CLI before model calls; each run uses the "
            "dataset start_position and start_rotation"),
        "model_output_used_for_sampling": False,
        "dataset_path": str(dataset_path),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "episode_count": len(selected),
        "episode_indices": list(indices),
        "episodes": selected,
        "configuration": configuration,
    }
    path = output_root / "run_manifest.json"
    if path.exists():
        previous = json.loads(path.read_text())
        frozen_keys = ("dataset_sha256", "episode_indices", "configuration")
        if any(previous.get(key) != payload.get(key) for key in frozen_keys):
            raise RuntimeError(
                f"refusing to change frozen evaluation manifest: {path}")
        return previous
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return payload


def episode_artifact_stem(episode_id):
    """Name episode directories and videos by dataset episode_id, not row index."""
    return f"episode_{int(episode_id):04d}"


def failed_episode_result(index, returncode, output_dir, episode_id=None):
    return {
        "episode_index": index,
        "episode_id": episode_id,
        "mode": None,
        "selection_strategy": None,
        "exploration_strategy": None,
        "navigation_executor": None,
        "targets_attempted": 0,
        "targets_arrived": 0,
        "point_arrival_rate": 0.0,
        "sub_instruction_count": 0,
        "system_sub_instructions_completed": 0,
        "sub_instructions_completed": 0,
        "sub_instruction_completion_rate": 0.0,
        "sequence_classifications": 0,
        "sequence_classifications_accepted": 0,
        "sequence_exploration_hops": 0,
        "sequence_recovery_backtracks": 0,
        "sequence_recovery_attempts": 0,
        "sequence_recovery_successes": 0,
        "blocked_direction_count": 0,
        "control_steps": 0,
        "path_length_m": 0.0,
        "initial_geodesic_distance_m": None,
        "final_geodesic_distance_m": None,
        "navigation_progress_m": 0.0,
        "r2r_success": False,
        "stop_action_issued": False,
        "simulator_reported_success": False,
        "goal_radius_hit_diagnostic": False,
        "invalid_goal_radius_hit": False,
        "instruction_validated_r2r_success": False,
        "r2r_spl": 0.0,
        "termination_reason": "episode_process_failed",
        "frontier_queue_exhausted": False,
        "full_space_complete": False,
        "estimated_coverage_ratio": 0.0,
        "reachable_island_coverage_ratio": 0.0,
        "visited_area_estimate_m2": 0.0,
        "reachable_island_area_m2": 0.0,
        "bfs_expansions": 0,
        "bfs_forward_attempts": 0,
        "bfs_backtrack_requests": 0,
        "bfs_backtrack_successes": 0,
        "bfs_blocked_frontiers": 0,
        "backtrack_success": None,
        "instruction_sequence_success": False,
        "instruction_sequence_end_reason": "episode_process_failed",
        "termination_category": "episode_process_failed",
        "process_returncode": returncode,
        "process_log": str(output_dir / "process.log"),
        "trajectory": str(output_dir / "trajectory.json"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=5,
                        help="number of consecutive R2R episodes to run")
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--episodes", default=None,
                        help="optional comma-separated indices; overrides count/start")
    parser.add_argument("--mode", choices=["semantic", "pure-exploration"],
                        default="semantic")
    parser.add_argument(
        "--exploration-strategy",
        choices=["standard", "breadth-first",
                 "instruction-sequence-recovery"],
        default="instruction-sequence-recovery")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "outputs/point_navigation_eval")
    parser.add_argument("--r2r-data", type=Path, default=DEFAULT_R2R_DATA)
    parser.add_argument(
        "--decomposition-artifact", type=Path, default=None,
        help="frozen per-episode decomposition or ten-EP benchmark manifest")
    parser.add_argument("--max-steps-per-target", type=int, default=40)
    parser.add_argument("--max-exploration-targets", type=int, default=120)
    parser.add_argument("--full-space-coverage-threshold", type=float,
                        default=0.90)
    parser.add_argument("--coverage-samples", type=int, default=12000)
    parser.add_argument("--targets", type=int, default=None,
                        help="optional per-episode target/stage cap")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--policy", choices=["gnm", "vint", "nomad"],
                        default="gnm")
    parser.add_argument("--vlm-backend", choices=["deepseek", "ollama", "heuristic"],
                        default="deepseek")
    parser.add_argument("--vlm-model", default=None)
    parser.add_argument(
        "--semantic-detector", choices=["dino-sam", "grounded-sam", "none"],
        default="dino-sam",
        help="open-vocabulary detector passed to the Habitat task entrypoint")
    parser.add_argument(
        "--floor-segmenter",
        choices=["dense-majority", "grounded-sam", "dino-sam"],
        default="dense-majority",
        help="ground mask backend passed to the Habitat task entrypoint")
    parser.add_argument("--detector-box-threshold", type=float, default=0.28)
    parser.add_argument("--detector-text-threshold", type=float, default=0.22)
    parser.add_argument(
        "--adaptive-floor-threshold", action="store_true",
        help="retain weak Grounded-SAM floor proposals for relation/continuous stages")
    parser.add_argument("--views", type=int, choices=[6, 8], default=8)
    parser.add_argument(
        "--skip-instruction-completion", action="store_true",
        help="run task-condition point selection/navigation without the completion VLM call")
    parser.add_argument(
        "--point-selection-prompt-version",
        default="v10_approach_relation_router")
    parser.add_argument(
        "--instruction-completion-prompt-version",
        default="v13_structured_node_edge_binary")
    parser.add_argument(
        "--tracking-cluster-profile", default="rgb_only_dense_stop_v1")
    parser.add_argument("--deepseek-env", type=Path, default=ROOT / ".env.deepseek")
    parser.add_argument("--deepseek-base-url", default=None)
    parser.add_argument("--vlm-timeout", type=int, default=180)
    parser.add_argument("--vlm-retries", type=int, default=2)
    parser.add_argument("--sequence-max-exploration-hops", type=int, default=30)
    parser.add_argument("--sequence-max-blocked-directions", type=int, default=5)
    parser.add_argument(
        "--sequence-min-classification-confidence", type=float, default=0.5)
    parser.add_argument(
        "--sequence-recovery-backtrack-attempts", type=int, default=4)
    parser.add_argument("--backtrack-target-node", default=None)
    parser.add_argument("--backtrack-selector", choices=["auto", "hybrid", "vlm"],
                        default="auto")
    parser.add_argument(
        "--backtrack-planner-profile", default="legacy_direct",
        choices=sorted(BACKTRACK_PLANNER_PROFILES))
    parser.add_argument("--backtrack-max-attempts-per-hop", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    if args.mode != "semantic" or args.exploration_strategy != (
            "instruction-sequence-recovery"):
        parser.error(
            "production evaluation is RGB-only and currently requires "
            "semantic instruction-sequence-recovery")
    if args.tracking_cluster_profile != "rgb_only_dense_stop_v1":
        parser.error(
            "production evaluation requires rgb_only_dense_stop_v1")

    indices = episode_indices(
        args.episodes, args.start_episode, args.num_episodes)
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_configuration = {
        "mode": args.mode,
        "exploration_strategy": args.exploration_strategy,
        "policy": args.policy,
        "device": args.device,
        "vlm_backend": args.vlm_backend,
        "vlm_model": args.vlm_model or "backend_default",
        "semantic_detector": args.semantic_detector,
        "floor_segmenter": args.floor_segmenter,
        "decomposition_artifact": (
            str(args.decomposition_artifact.resolve())
            if args.decomposition_artifact is not None else None),
        "detector_box_threshold": args.detector_box_threshold,
        "detector_text_threshold": args.detector_text_threshold,
        "adaptive_floor_threshold": args.adaptive_floor_threshold,
        "views": args.views,
        "skip_instruction_completion": args.skip_instruction_completion,
        "point_selection_prompt_version": (
            args.point_selection_prompt_version),
        "instruction_completion_prompt_version": (
            args.instruction_completion_prompt_version),
        "tracking_cluster_profile": args.tracking_cluster_profile,
        "policy_input_contract": "rgb_only_v1",
        "policy_observations": ["rgb", "commanded_action_history"],
        "privileged_geometry_scope": "postrun_evaluation_only",
        "vlm_timeout": args.vlm_timeout,
        "vlm_retries": args.vlm_retries,
        "seed": args.seed,
        "max_steps_per_target": args.max_steps_per_target,
        "max_exploration_targets": args.max_exploration_targets,
        "full_space_coverage_threshold": (
            args.full_space_coverage_threshold),
        "coverage_samples": args.coverage_samples,
        "targets": args.targets,
        "sequence_max_exploration_hops": args.sequence_max_exploration_hops,
        "sequence_max_blocked_directions": (
            args.sequence_max_blocked_directions),
        "sequence_min_classification_confidence": (
            args.sequence_min_classification_confidence),
        "sequence_recovery_backtrack_attempts": (
            args.sequence_recovery_backtrack_attempts),
        "backtrack_planner_profile": args.backtrack_planner_profile,
        "backtrack_max_attempts_per_hop": (
            args.backtrack_max_attempts_per_hop),
    }
    manifest = freeze_episode_manifest(
        args.output_root, args.r2r_data, indices, run_configuration)
    episode_id_by_index = {
        int(item["episode_index"]): item["episode_id"]
        for item in manifest["episodes"]}
    results = []
    for index in indices:
        episode_id = episode_id_by_index[index]
        output_dir = args.output_root / episode_artifact_stem(episode_id)
        trajectory_path = output_dir / "trajectory.json"
        if args.rerun or not trajectory_path.exists():
            command = [
                sys.executable, str(HABITAT_ENTRYPOINT),
                "--mode", args.mode,
                "--policy-input-contract", "rgb-only-v1",
                "--exploration-strategy", args.exploration_strategy,
                "--episode-index", str(index),
                "--r2r-data", str(args.r2r_data),
                *((["--decomposition-artifact", str(args.decomposition_artifact)]
                   if args.decomposition_artifact is not None else [])),
                "--policy", args.policy,
                "--device", args.device,
                "--max-steps-per-target", str(args.max_steps_per_target),
                "--max-exploration-targets", str(args.max_exploration_targets),
                "--full-space-coverage-threshold",
                str(args.full_space_coverage_threshold),
                "--coverage-samples", str(args.coverage_samples),
                "--vlm-backend", args.vlm_backend,
                "--vlm-timeout", str(args.vlm_timeout),
                "--vlm-retries", str(args.vlm_retries),
                "--semantic-detector", args.semantic_detector,
                "--floor-segmenter", args.floor_segmenter,
                "--detector-box-threshold", str(args.detector_box_threshold),
                "--detector-text-threshold", str(args.detector_text_threshold),
                "--views", str(args.views),
                "--point-selection-prompt-version",
                args.point_selection_prompt_version,
                "--instruction-completion-prompt-version",
                args.instruction_completion_prompt_version,
                "--tracking-cluster-profile", args.tracking_cluster_profile,
                "--seed", str(args.seed),
                "--sequence-max-exploration-hops",
                str(args.sequence_max_exploration_hops),
                "--sequence-max-blocked-directions",
                str(args.sequence_max_blocked_directions),
                "--sequence-min-classification-confidence",
                str(args.sequence_min_classification_confidence),
                "--sequence-recovery-backtrack-attempts",
                str(args.sequence_recovery_backtrack_attempts),
                "--backtrack-planner-profile",
                args.backtrack_planner_profile,
                "--backtrack-max-attempts-per-hop",
                str(args.backtrack_max_attempts_per_hop),
                "--output-dir", str(output_dir),
            ]
            if args.vlm_model is not None:
                command.extend(["--vlm-model", args.vlm_model])
            command.extend(["--deepseek-env", str(args.deepseek_env)])
            if args.deepseek_base_url is not None:
                command.extend(["--deepseek-base-url", args.deepseek_base_url])
            if args.targets is not None:
                command.extend(["--targets", str(args.targets)])
            if args.adaptive_floor_threshold:
                command.append("--adaptive-floor-threshold")
            if args.skip_instruction_completion:
                command.append("--skip-instruction-completion")
            if args.backtrack_target_node is not None:
                command.extend([
                    "--backtrack-target-node", args.backtrack_target_node,
                    "--backtrack-selector", args.backtrack_selector,
                    "--backtrack-max-attempts-per-hop",
                    str(args.backtrack_max_attempts_per_hop),
                ])
            print(f"starting episode id {episode_id} (index {index}): "
                  f"{output_dir}", flush=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            with (output_dir / "process.log").open("w") as process_log:
                completed = subprocess.run(
                    command, cwd=ROOT, check=False, stdout=process_log,
                    stderr=subprocess.STDOUT)
            print(
                f"finished episode id {episode_id} (index {index}): "
                f"returncode={completed.returncode}", flush=True)
            process_log_path = output_dir / "process.log"
            process_output = process_log_path.read_text(
                errors="replace") if process_log_path.exists() else ""
            if (completed.returncode != 0 and
                    provider_fatal_output(process_output)):
                interruption = {
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "invalid_external_vlm_provider_interruption",
                    "provider": args.vlm_backend,
                    "episode_index": index,
                    "episode_id": episode_id,
                    "process_returncode": completed.returncode,
                    "process_log": str(process_log_path),
                    "completed_episode_indices_before_interruption": [
                        item["episode_index"] for item in results],
                    "unrun_episode_indices": indices[
                        indices.index(index) + 1:],
                    "formal_score": None,
                    "reason": (
                        "Non-retryable VLM provider/authentication/account "
                        "failure; shard stopped to prevent navigation-score "
                        "contamination."),
                }
                interruption_path = args.output_root / (
                    f"provider_interruption_{episode_artifact_stem(episode_id)}.json")
                interruption_path.write_text(
                    json.dumps(interruption, indent=2) + "\n")
                print(
                    f"fatal VLM provider interruption; shard stopped: "
                    f"{interruption_path}", file=sys.stderr, flush=True)
                raise SystemExit(86)
            if completed.returncode != 0 and not trajectory_path.exists():
                results.append(failed_episode_result(
                    index, completed.returncode, output_dir, episode_id))
                continue

        if not trajectory_path.exists():
            results.append(failed_episode_result(
                index, None, output_dir, episode_id))
            continue

        trajectory = json.loads(trajectory_path.read_text())
        contract_audit = audit_run(output_dir)
        if not contract_audit["passed"]:
            raise RuntimeError(
                "RGB-only contract audit failed for episode "
                f"{index}: {contract_audit['violations']}")
        evaluation_geometry_path = (
            output_dir / "evaluation_only" / "evaluation_geometry.json")
        evaluation_geometry = (
            json.loads(evaluation_geometry_path.read_text())
            if evaluation_geometry_path.exists() else {})
        point_validation_by_target = {}
        for event in evaluation_geometry.get("point_events", []):
            if event.get("event") != "point_navigation_stopped":
                continue
            target_index = int((event.get("payload") or {}).get(
                "target_index", -1))
            distance = event.get("final_target_geodesic_distance_m")
            point_validation_by_target[target_index] = {
                "point_target_reference_reached": bool(
                    distance is not None and float(distance) <= 0.75),
                "final_target_geodesic_distance_m": distance,
                "source": "postrun_hidden_evaluator",
            }
        sequence_end_reason = str((trajectory.get(
            "instruction_sequence_exploration") or {}).get(
                "end_reason", ""))
        # The Habitat policy intentionally converts ordinary selection
        # failures into a valid trajectory, but an exhausted external-provider
        # call is not a navigation outcome. Detect that structured end reason
        # even when the child returned zero, invalidate the shard, and require
        # a clean rerun from this episode's dataset start.
        if provider_fatal_output(sequence_end_reason):
            interruption = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "invalid_external_vlm_provider_interruption",
                "provider": args.vlm_backend,
                "episode_index": index,
                "episode_id": episode_id,
                "process_returncode": 0,
                "trajectory": str(trajectory_path),
                "completed_episode_indices_before_interruption": [
                    item["episode_index"] for item in results],
                "unrun_episode_indices": indices[
                    indices.index(index) + 1:],
                "formal_score": None,
                "reason": sequence_end_reason,
            }
            interruption_path = args.output_root / (
                f"provider_interruption_{episode_artifact_stem(episode_id)}.json")
            interruption_path.write_text(
                json.dumps(interruption, indent=2) + "\n")
            print(
                "fatal VLM provider interruption encoded in trajectory; "
                f"shard stopped: {interruption_path}",
                file=sys.stderr, flush=True)
            raise SystemExit(86)
        metrics = trajectory["r2r_metrics"]
        navigation_targets = trajectory.get("targets") or []
        # Enrich a private evaluator copy only after the complete online run;
        # never write hidden labels back into the policy trajectory.
        targets = []
        for item in navigation_targets:
            enriched = dict(item)
            enriched.update(point_validation_by_target.get(
                int(item.get("target_index", -1)), {
                    "point_target_reference_reached": False,
                    "final_target_geodesic_distance_m": None,
                    "source": "postrun_hidden_evaluator_missing_label",
                }))
            targets.append(enriched)
        # In sequence mode ``targets_completed`` is tied to strategy cursor
        # bookkeeping and can differ from the actual per-target executor
        # signals.  Score the auditable target records directly.
        attempted = len(targets)
        # The Habitat task entrypoint uses ``arrived`` as its canonical
        # executor signal; older sequence artifacts called the same field
        # ``point_target_arrived``.  Accept both so task-condition runs are
        # not reported as zero-arrival merely because of the schema alias.
        def target_arrived(item):
            return bool(item.get("point_target_arrived",
                                 item.get("arrived", False)))

        arrived = sum(
            target_arrived(item) for item in targets)
        arrival_tp = sum(
            target_arrived(item) and
            bool(item.get("point_target_reference_reached"))
            for item in targets)
        arrival_fp = sum(
            target_arrived(item) and
            not bool(item.get("point_target_reference_reached"))
            for item in targets)
        arrival_fn = sum(
            not target_arrived(item) and
            bool(item.get("point_target_reference_reached"))
            for item in targets)
        arrival_tn = sum(
            not target_arrived(item) and
            not bool(item.get("point_target_reference_reached"))
            for item in targets)
        reference_reached = arrival_tp + arrival_fn
        completion_outputs = [
            item for item in targets
            if bool((item.get("instruction_completion") or {}).get(
                "instruction_completed"))]
        completion_after_reference_reach = sum(
            bool(item.get("point_target_reference_reached"))
            for item in completion_outputs)
        sequence = trajectory.get("instruction_sequence_exploration") or {}
        stage_verification = strict_stage_completion_metrics(
            trajectory, targets,
            output_dir / "stage_completion_verification.json")
        strict_goal = instruction_validated_goal_metrics(
            trajectory, metrics, sequence,
            verified_sequence_complete=stage_verification[
                "all_evaluated_stages_jointly_passed"])
        simulator_goal = simulator_stop_goal_metrics(trajectory, metrics)
        sequence_state = sequence.get("state") or {}
        classifications = sequence_state.get("classification_history") or []
        recovery_records = sequence.get("recovery_records") or []
        blocked_yaws = sequence_state.get("blocked_yaws_by_verified_node") or {}
        sub_instruction_count = len(trajectory.get("sub_instructions") or [])
        completed_sub_instructions = int(
            sequence.get("completed_sub_instructions") or 0)
        pure = trajectory.get("pure_exploration") or {}
        bfs = trajectory.get("breadth_first_exploration") or {}
        results.append({
            "episode_index": index,
            "episode_id": trajectory["episode_id"],
            "mode": trajectory["mode"],
            "selection_strategy": trajectory["selection_strategy"],
            "exploration_strategy": trajectory.get("exploration_strategy", "standard"),
            "navigation_executor": trajectory["navigation_executor"],
            "rgb_only_contract_audit_passed": True,
            "targets_attempted": attempted,
            "targets_arrived": arrived,
            "point_arrival_rate": arrived / max(attempted, 1),
            "executor_arrival_signal_rate": arrived / max(attempted, 1),
            "reference_point_reaches": reference_reached,
            "physical_point_reach_rate": reference_reached / max(attempted, 1),
            "arrival_confusion": {
                "tp": arrival_tp, "fp": arrival_fp,
                "fn": arrival_fn, "tn": arrival_tn,
            },
            "arrival_decision_accuracy": (
                (arrival_tp + arrival_tn) / max(attempted, 1)),
            "instruction_completed_outputs": len(completion_outputs),
            "instruction_completed_after_reference_reach": (
                completion_after_reference_reach),
            "sub_instruction_count": sub_instruction_count,
            "system_sub_instructions_completed": completed_sub_instructions,
            "sub_instructions_completed": stage_verification[
                "verified_ordered_prefix_count"],
            "sub_instruction_completion_rate": (
                stage_verification["verified_ordered_prefix_count"] /
                max(sub_instruction_count, 1)),
            "stage_completion_verification": stage_verification,
            "sequence_classifications": len(classifications),
            "sequence_classifications_accepted": sum(
                bool(item.get("correct_sequence_position"))
                for item in classifications),
            "sequence_exploration_hops": int(
                sequence.get("exploration_hops") or 0),
            "sequence_recovery_backtracks": int(
                sequence.get("recovery_backtracks") or 0),
            "sequence_recovery_attempts": len(recovery_records),
            "sequence_recovery_successes": sum(
                bool(item.get("success")) for item in recovery_records),
            "blocked_direction_count": sum(
                len(values) for values in blocked_yaws.values()),
            "control_steps": trajectory["total_control_steps"],
            "path_length_m": metrics["path_length_m"],
            "initial_geodesic_distance_m": metrics[
                "initial_geodesic_distance_m"],
            "final_geodesic_distance_m": metrics[
                "final_geodesic_distance_m"],
            "navigation_progress_m": (
                metrics["initial_geodesic_distance_m"]
                - metrics["final_geodesic_distance_m"]),
            "r2r_success": strict_goal["instruction_validated_r2r_success"],
            **simulator_goal,
            **strict_goal,
            "r2r_spl": instruction_validated_spl(
                metrics,
                strict_goal["instruction_validated_r2r_success"]),
            "termination_reason": (
                pure.get("termination_reason")),
            "frontier_queue_exhausted": bool(
                pure.get("frontier_queue_exhausted")),
            "full_space_complete": bool(pure.get("full_space_complete")),
            "estimated_coverage_ratio": float(
                pure.get("estimated_coverage_ratio") or 0.0),
            "reachable_island_coverage_ratio": float(
                pure.get("reachable_island_coverage_ratio") or 0.0),
            "visited_area_estimate_m2": float(
                pure.get("visited_area_estimate_m2") or 0.0),
            "reachable_island_area_m2": float(
                pure.get("reachable_island_area_m2") or 0.0),
            "bfs_expansions": int(bfs.get("expansions") or 0),
            "bfs_forward_attempts": int(bfs.get("forward_attempts") or 0),
            "bfs_backtrack_requests": int(
                bfs.get("backtrack_requests") or 0),
            "bfs_backtrack_successes": int(
                bfs.get("backtrack_successes") or 0),
            "bfs_blocked_frontiers": int(
                bfs.get("blocked_frontiers") or 0),
            "backtrack_success": (
                trajectory.get("node_backtracking") or {}).get("success"),
            "system_instruction_sequence_success": bool(sequence.get("success")),
            "instruction_sequence_success": bool(
                sequence.get("success") and stage_verification[
                    "all_evaluated_stages_jointly_passed"]),
            "instruction_sequence_end_reason": sequence.get("end_reason"),
            "termination_category": termination_category(
                sequence.get("end_reason")),
            "trajectory": str(trajectory_path),
        })

    summary = summarize_results(results, indices, args.mode, {
        "exploration_strategy": args.exploration_strategy,
        "policy": args.policy,
        "device": args.device,
        "vlm_backend": args.vlm_backend,
        "vlm_model": args.vlm_model or "backend_default",
        "semantic_detector": args.semantic_detector,
        "floor_segmenter": args.floor_segmenter,
        "detector_box_threshold": args.detector_box_threshold,
        "detector_text_threshold": args.detector_text_threshold,
        "adaptive_floor_threshold": args.adaptive_floor_threshold,
        "views": args.views,
        "skip_instruction_completion": args.skip_instruction_completion,
        "point_selection_prompt_version": (
            args.point_selection_prompt_version),
        "instruction_completion_prompt_version": (
            args.instruction_completion_prompt_version),
        "tracking_cluster_profile": args.tracking_cluster_profile,
        "seed": args.seed,
        "vlm_timeout": args.vlm_timeout,
        "vlm_retries": args.vlm_retries,
        "max_steps_per_target": args.max_steps_per_target,
        "max_exploration_targets": args.max_exploration_targets,
        "full_space_coverage_threshold": (
            args.full_space_coverage_threshold),
        "coverage_samples": args.coverage_samples,
        "sequence_max_exploration_hops": (
            args.sequence_max_exploration_hops),
        "sequence_min_classification_confidence": (
            args.sequence_min_classification_confidence),
        "sequence_recovery_backtrack_attempts": (
            args.sequence_recovery_backtrack_attempts),
        "backtrack_planner_profile": args.backtrack_planner_profile,
        "backtrack_max_attempts_per_hop": (
            args.backtrack_max_attempts_per_hop),
    })
    summary_path = args.output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"summary: {summary_path}")


def summarize_results(results, indices, mode, configuration):
    """Aggregate per-episode result rows into the frozen summary schema."""
    total_attempted = sum(item["targets_attempted"] for item in results)
    total_arrived = sum(item["targets_arrived"] for item in results)
    total_reference_reached = sum(
        item.get("reference_point_reaches", 0) for item in results)
    arrival_tp = sum(
        item.get("arrival_confusion", {}).get("tp", 0) for item in results)
    arrival_fp = sum(
        item.get("arrival_confusion", {}).get("fp", 0) for item in results)
    arrival_fn = sum(
        item.get("arrival_confusion", {}).get("fn", 0) for item in results)
    arrival_tn = sum(
        item.get("arrival_confusion", {}).get("tn", 0) for item in results)
    total_completion_outputs = sum(
        item.get("instruction_completed_outputs", 0) for item in results)
    total_completion_after_reference_reach = sum(
        item.get("instruction_completed_after_reference_reach", 0)
        for item in results)
    total_sub_instructions = sum(
        item["sub_instruction_count"] for item in results)
    total_sub_instructions_completed = sum(
        item["sub_instructions_completed"] for item in results)
    total_system_sub_instructions_completed = sum(
        item.get("system_sub_instructions_completed", 0) for item in results)
    total_recovery_attempts = sum(
        item["sequence_recovery_attempts"] for item in results)
    total_recovery_successes = sum(
        item["sequence_recovery_successes"] for item in results)
    total_classifications = sum(
        item["sequence_classifications"] for item in results)
    total_classifications_accepted = sum(
        item["sequence_classifications_accepted"] for item in results)
    return {
        "episode_count": len(results),
        "episode_process_success_rate": sum(
            item.get("process_returncode", 0) in (None, 0) and
            Path(item["trajectory"]).exists() for item in results
        ) / len(results),
        "episode_indices": list(indices),
        "mode": mode,
        "configuration": dict(configuration),
        "totals": {
            "targets_attempted": total_attempted,
            "targets_arrived": total_arrived,
            "reference_point_reaches": total_reference_reached,
            "arrival_confusion": {
                "tp": arrival_tp, "fp": arrival_fp,
                "fn": arrival_fn, "tn": arrival_tn,
            },
            "instruction_completed_outputs": total_completion_outputs,
            "instruction_completed_after_reference_reach": (
                total_completion_after_reference_reach),
            "sub_instructions": total_sub_instructions,
            "system_sub_instructions_completed": (
                total_system_sub_instructions_completed),
            "sub_instructions_completed": total_sub_instructions_completed,
            "sequence_classifications": total_classifications,
            "sequence_classifications_accepted": (
                total_classifications_accepted),
            "sequence_recovery_attempts": total_recovery_attempts,
            "sequence_recovery_successes": total_recovery_successes,
        },
        "navigation_start_rate": sum(
            item["targets_attempted"] > 0 for item in results) / len(results),
        "point_arrival_rate": total_arrived / max(total_attempted, 1),
        "frontier_queue_exhaustion_rate": sum(
            item["frontier_queue_exhausted"] for item in results
        ) / len(results),
        "full_space_completion_rate": sum(
            item["full_space_complete"] for item in results
        ) / len(results),
        "mean_estimated_coverage_ratio": sum(
            item["estimated_coverage_ratio"] for item in results
        ) / len(results),
        "mean_reachable_island_coverage_ratio": sum(
            item["reachable_island_coverage_ratio"] for item in results
        ) / len(results),
        "total_bfs_expansions": sum(
            item["bfs_expansions"] for item in results),
        "total_bfs_blocked_frontiers": sum(
            item["bfs_blocked_frontiers"] for item in results),
        "bfs_backtrack_success_rate": (
            sum(item["bfs_backtrack_successes"] for item in results) /
            max(sum(item["bfs_backtrack_requests"] for item in results), 1)),
        "executor_arrival_signal_rate": (
            total_arrived / max(total_attempted, 1)),
        "physical_point_reach_rate": (
            total_reference_reached / max(total_attempted, 1)),
        "arrival_decision_accuracy": (
            (arrival_tp + arrival_tn) / max(total_attempted, 1)),
        "arrival_signal_precision": arrival_tp / max(arrival_tp + arrival_fp, 1),
        "arrival_signal_recall": arrival_tp / max(arrival_tp + arrival_fn, 1),
        "sub_instruction_completion_rate": (
            total_sub_instructions_completed / max(total_sub_instructions, 1)),
        "instruction_sequence_success_rate": sum(
            bool(item["instruction_sequence_success"]) for item in results
        ) / len(results),
        "sequence_recovery_success_rate": (
            total_recovery_successes / max(total_recovery_attempts, 1)),
        "sequence_classification_acceptance_rate": (
            total_classifications_accepted / max(total_classifications, 1)),
        "blocked_direction_count": sum(
            item["blocked_direction_count"] for item in results),
        "termination_category_counts": dict(Counter(
            item["termination_category"] for item in results)),
        "r2r_success_rate": sum(
            item["r2r_success"] for item in results) / len(results),
        "stop_action_count": sum(
            item["stop_action_issued"] for item in results),
        "simulator_reported_success_count": sum(
            item["simulator_reported_success"] for item in results),
        "simulator_reported_success_rate": sum(
            item["simulator_reported_success"] for item in results
        ) / len(results),
        "goal_radius_hit_diagnostic_count": sum(
            item["goal_radius_hit_diagnostic"] for item in results),
        "invalid_goal_radius_hit_count": sum(
            item["invalid_goal_radius_hit"] for item in results),
        "instruction_validated_r2r_success_count": sum(
            item["instruction_validated_r2r_success"] for item in results),
        "mean_r2r_spl": sum(
            item["r2r_spl"] for item in results) / len(results),
        "mean_navigation_progress_m": sum(
            item["navigation_progress_m"] for item in results) / len(results),
        "total_control_steps": sum(
            item["control_steps"] for item in results),
        "metric_definitions": {
            "executor_arrival_signal_rate": (
                "PointNavigationExecutor arrived signals / forward attempts"),
            "physical_point_reach_rate": (
                "hidden final geodesic <= 0.75 m / forward attempts"),
            "arrival_decision_accuracy": (
                "(TP + TN) / forward attempts against hidden 0.75 m label"),
            "sub_instruction_completion_rate": (
                "ordered stages where the navigation system judged completed "
                "at an arrived node AND an independent post-run semantic audit "
                "verified actual completion / evaluated sub-instructions; "
                "missing audit evidence counts as not completed"),
            "r2r_success_rate": (
                "complete instruction sequence executed and judged completed "
                "in order from episode start AND final goal radius reached; "
                "geometric-only hits are invalid and excluded"),
            "simulator_reported_success_rate": (
                "explicit task-level STOP actions issued immediately after "
                "the final sub-instruction is judged completed, with the STOP "
                "pose inside the episode goal radius / episodes"),
            "goal_radius_hit_diagnostic_count": (
                "diagnostic only; never reported as task success"),
            "full_space_completion_rate": (
                "fraction with natural FIFO frontier-queue exhaustion and "
                "deterministic 3-D swept coverage of the full start-reachable "
                "navmesh island >= frozen threshold"),
        },
        "results": results,
    }


if __name__ == "__main__":
    main()
