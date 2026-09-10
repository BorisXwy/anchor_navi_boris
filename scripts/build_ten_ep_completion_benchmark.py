#!/usr/bin/env python3
"""Build the fixed ten-EP compositional completion benchmark.

The benchmark is deliberately a contract, not a model result.  It freezes the
episodes, their deterministic form-level sub-instruction scaffold, and the
stage/episode success predicates.  Human point baselines and runtime artifacts
are filled in by later real Habitat rounds; they are never synthesized here.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from instruction_taxonomy import decompose_by_definition


DEFAULT_EPISODES = (0, 3, 6, 9, 18, 27, 45, 126, 204, 219)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_R2R_DATA = (
    ROOT.parent / "3d_wm_vln/StreamVLN/data/datasets/r2r/val_unseen/"
    "val_unseen.json.gz"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/r2r_curriculum_10ep_20260904/benchmark/"
    "ten_ep_completion_benchmark_v1.json"
)
DEFAULT_MANUAL_BASELINE = (
    ROOT / "outputs/r2r_curriculum_10ep_20260904/round_70_strict_start_baseline/"
    "manual_baseline.json"
)
DEFAULT_DECOMPOSITION_MANIFEST = (
    ROOT / "outputs/r2r_curriculum_10ep_20260904/round_70_strict_start_baseline/"
    "start_baseline_manifest.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def instruction_text(episode: dict) -> str:
    value = episode.get("instruction", "")
    if isinstance(value, dict):
        value = value.get("instruction_text", "")
    return str(value).strip()


def stage_record(stage: dict, manual_baseline_artifact: str | None = None) -> dict:
    """Keep only the frozen, model-independent decomposition contract."""
    stage_id = int(stage.get("stage_id", stage.get("sub_instruction_id", 0)))
    return {
        "sub_instruction_id": stage_id,
        "source_clause": stage.get("source_clause", ""),
        "form": stage.get("form", "OTHER"),
        "secondary_forms": list(stage.get("secondary_forms", []) or []),
        "definition": stage.get("definition", ""),
        "navigation_instruction": stage.get("navigation_instruction", ""),
        "landmark": stage.get("landmark", "unspecified"),
        "completion_cue": stage.get("completion_cue", ""),
        "semantic_spatial_target": stage.get("semantic_spatial_target", ""),
        "spatial_relation": stage.get("spatial_relation", ""),
        "visual_arrival_evidence": stage.get("visual_arrival_evidence", ""),
        "forbidden_target": stage.get("forbidden_target", ""),
        "point_selection_strategy": dict(stage.get(
            "point_selection_strategy", {}) or {}),
        "manual_baseline_artifact": manual_baseline_artifact,
        "runtime_artifact": None,
        "status": "decomposition_scaffold_only",
    }


def build_benchmark(
        r2r_data: Path, episode_indices: tuple[int, ...],
        manual_baseline: Path | None = None,
        decomposition_manifest: Path | None = None) -> dict:
    with gzip.open(r2r_data, "rt") as handle:
        episodes = json.load(handle)["episodes"]
    if len(set(episode_indices)) != len(episode_indices):
        raise ValueError("episode indices must be unique")
    if any(index < 0 or index >= len(episodes) for index in episode_indices):
        raise IndexError("episode index is outside the R2R dataset")

    baseline_rows = {}
    if manual_baseline is not None and manual_baseline.exists():
        raw_baseline = json.loads(manual_baseline.read_text())
        for row in raw_baseline.get("episodes", []):
            baseline_rows[int(row["episode_index"])] = row

    decomposition_rows = {}
    if decomposition_manifest is not None and decomposition_manifest.exists():
        raw_decomposition = json.loads(decomposition_manifest.read_text())
        for row in raw_decomposition.get("episodes", []):
            stages = row.get("decomposition") or row.get("stages") or []
            if stages:
                decomposition_rows[int(row["episode_index"])] = stages

    episode_records = []
    for index in episode_indices:
        episode = episodes[index]
        text = instruction_text(episode)
        prior_stages = decomposition_rows.get(index)
        stages = prior_stages or decompose_by_definition(text)
        if not stages:
            raise ValueError(f"episode {index} produced no sub-instructions")
        baseline_row = baseline_rows.get(index)
        baseline_artifact = (
            str(manual_baseline.resolve())
            if baseline_row is not None and manual_baseline is not None else None)
        episode_records.append({
            "episode_index": index,
            "episode_id": str(episode["episode_id"]),
            "trajectory_id": episode.get("trajectory_id"),
            "scene_id": episode.get("scene_id"),
            "instruction": text,
            "start_position": episode.get("start_position"),
            "start_rotation": episode.get("start_rotation"),
            "goal_specs": episode.get("goals", []),
            "reference_path_length": len(episode.get("reference_path", [])),
            "decomposition": {
                "status": (
                    "frozen_prior_vlm_audit"
                    if prior_stages else
                    "frozen_form_scaffold_pending_vlm_audit"),
                "source": (
                    str(decomposition_manifest.resolve())
                    if prior_stages and decomposition_manifest is not None else
                    "instruction_taxonomy.decompose_by_definition"),
                "stages": [stage_record(
                    stage, baseline_artifact if int(stage.get(
                        "stage_id", stage.get("sub_instruction_id", 0))) == 0
                    else None)
                    for stage in stages],
            },
            "human_baseline_status": (
                "stage0_available_remaining_stages_pending"
                if baseline_row is not None else
                "pending_for_each_applicable_stage"),
            "human_baseline_stage0": baseline_row,
            "runtime_status": "not_run",
        })

    return {
        "schema_version": 1,
        "benchmark_id": "r2r_ten_ep_compositional_completion_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "split": "val_unseen",
            "path": str(r2r_data.resolve()),
            "sha256": sha256(r2r_data),
        },
        "decomposition_source": {
            "path": (str(decomposition_manifest.resolve())
                     if decomposition_manifest is not None and
                     decomposition_manifest.exists() else None),
            "sha256": (sha256(decomposition_manifest)
                       if decomposition_manifest is not None and
                       decomposition_manifest.exists() else None),
            "fallback": "instruction_taxonomy.decompose_by_definition",
        },
        "fixed_episode_indices": list(episode_indices),
        "model_input_boundary": {
            "demonstration_future_path_exposed": False,
            "human_baseline_exposed": False,
            "reference_path_exposed": False,
            "depth_used_by_vlm_selection": False,
            "local_visual_models_device": "cuda:0",
        },
        "stage_contract": {
            "selection": {
                "id": "A_selection",
                "requires_human_baseline": True,
                "requires_grounded_sam_floor_or_stair_mask": True,
                "requires_navmesh_projection": True,
                "max_heading_error_deg": 30.0,
                "must_save": [
                    "initial_six_or_eight_views", "human_target_and_rejections",
                    "vlm_response", "candidate_masks", "selected_pixel",
                    "selected_navmesh_target", "path_projection_overlay",
                ],
            },
            "physical_arrival": {
                "id": "B_physical_arrival",
                "requires_executor_signal": "point_navigation_arrived",
                "max_final_selected_point_geodesic_m": 0.75,
                "requires_human_rgb_action_pose_audit": True,
                "rejects_early_offscreen_or_cluster_loss": True,
            },
            "node_edge": {
                "id": "N_node_edge_invariant",
                "functional_not_accuracy_metric": True,
                "required_fields": [
                    "position", "yaw", "six_views", "semantic_state",
                    "visual_embedding", "incoming_action_history", "keyframes",
                ],
            },
            "completion": {
                "id": "C_instruction_completion",
                "allowed_outputs": ["completed", "unknown"],
                "human_label_frozen_before_model_call": True,
                "unknown_is_not_auto_failure": True,
                "required_metrics": [
                    "confusion_matrix", "precision", "recall", "f1",
                ],
            },
            "completion_verification": {
                "id": "V_independent_semantic_verification",
                "requires_system_node_completion": True,
                "requires_post_run_frozen_manual_edge_audit": True,
                "missing_audit_policy": "stage_not_completed",
                "audit_exposed_to_online_models": False,
                "required_evidence": [
                    "previous_node_views", "current_node_views",
                    "incoming_action_history", "edge_keyframes",
                    "executed_trajectory", "demonstration_trajectory",
                ],
            },
            "backtracking": {
                "id": "R_real_backtracking",
                "teleport_forbidden": True,
                "requires": [
                    "point_navigation_arrived", "position_revisit",
                    "visual_loop_closure", "graph_edge_consistency",
                ],
            },
        },
        "ep_success_predicate": {
            "definition": (
                "An EP succeeds iff every applicable sub-instruction, in order, "
                "passes A_selection, B_physical_arrival, N_node_edge and "
                "C_instruction_completion=completed, then passes independent "
                "V_semantic_verification; all unknown branches are "
                "resolved by the system without manual takeover; any required "
                "backtrack passes R_real_backtracking; and the final Habitat goal "
                "is reached and judged completed."
            ),
            "required_for_each_applicable_stage": [
                "selection_pass", "physical_arrival_pass", "node_edge_complete",
                "completion_status_completed", "semantic_completion_verified",
            ],
            "forbidden": [
                "unresolved_unknown", "manual_takeover", "teleport_recovery",
                "failed_backtrack", "missing_stage_artifact",
                "invalid_goal_radius_hit",
            ],
            "goal_metric_policy": {
                "diagnostic_only": "goal_radius_hit_diagnostic",
                "counted_success": "instruction_validated_r2r_success",
                "rule": (
                    "A goal-radius hit is counted only after the complete "
                    "decomposed instruction sequence has been executed and "
                    "judged completed in order. Accidental, out-of-order, "
                    "wrong-branch, or truncated-prefix hits are failures."
                ),
            },
        },
        "ten_ep_success_predicate": {
            "definition": "All ten fixed EPs must satisfy ep_success_predicate.",
            "not_applicable_policy": (
                "An EP with no stage at a curriculum round is recorded as "
                "not_applicable for that round, but cannot be removed from the "
                "fixed ten-EP manifest or final EP denominator."
            ),
        },
        "episodes": episode_records,
        "artifact_layout": {
            "round_root": "outputs/r2r_curriculum_10ep_20260904/round_<id>/",
            "required_files": [
                "manifest.json", "summary.json", "failure_cases.md",
                "per_episode/<episode_index>/per_sub_instruction/<id>.json",
                "videos/", "human_baselines/",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2r-data", type=Path, default=DEFAULT_R2R_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manual-baseline", type=Path,
                        default=DEFAULT_MANUAL_BASELINE,
                        help="optional pre-model human baseline JSON")
    parser.add_argument("--decomposition-manifest", type=Path,
                        default=DEFAULT_DECOMPOSITION_MANIFEST,
                        help="optional frozen per-EP decomposition manifest")
    parser.add_argument(
        "--episode-indices", default=",".join(map(str, DEFAULT_EPISODES)),
        help="comma-separated fixed R2R episode indices")
    args = parser.parse_args()
    indices = tuple(int(item.strip()) for item in args.episode_indices.split(",")
                    if item.strip())
    benchmark = build_benchmark(
        args.r2r_data, indices, args.manual_baseline,
        args.decomposition_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "benchmark_id": benchmark["benchmark_id"],
        "fixed_episode_indices": benchmark["fixed_episode_indices"],
        "episode_stage_counts": {
            str(item["episode_index"]): len(item["decomposition"]["stages"])
            for item in benchmark["episodes"]
        },
        "human_baselines": {
            "stage0_available": sum(
                item["human_baseline_stage0"] is not None
                for item in benchmark["episodes"]),
            "all_applicable_stages": "pending",
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
