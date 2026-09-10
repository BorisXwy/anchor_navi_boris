#!/usr/bin/env python3
"""Build and evaluate 20 trajectory-disjoint R2R completion judgments.

The set is frozen before any VLM call.  Ten new val_unseen trajectories each
contribute one full demonstrated edge (completed) and one negative edge.  The
negative is alternately a full reversed replay or the first internal segment.
All states, actions, panoramas, keyframes, semantics, and embeddings are created
from real Habitat replays using the same online graph contract as navigation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

import evaluate_instruction_completion_triplets as construction
from instruction_decomposer import SubInstruction
from instruction_taxonomy import FORM_DEFINITIONS
from navigation_graph_memory import (
    DinoSamEnvironmentSemanticExtractor, NavigationGraphMemory)
from semantic_detector import DinoSamDetector
from vlm_harness import NavigationVLMHarness, build_vlm_backend, default_vlm_model


ROOT = Path(__file__).resolve().parents[1]
PROMPT_VERSION = "v13_structured_node_edge_binary"
DEFAULT_OUTPUT_ROOT = (
    ROOT / "outputs/instruction_completion_holdout20_v13_20260901")
DEVELOPMENT_EPISODE_INDICES = {
    34, 133, 321, 632, 226, 166, 780, 570, 1207, 1631}

# Frozen before VLM evaluation.  These are distinct episodes and trajectory IDs
# from the earlier 30-case development set.  Groundings describe the full R2R
# instruction endpoint rather than an intermediate action.
HOLDOUT_BASE_CASES = (
    {
        "episode_index": 7, "form": "CROSS_SPACE",
        "landmark": "room and far end of the bar",
        "target": "the far side of the room at the far end of the bar",
        "arrival": "the room is crossed and the far end of the bar is reached",
    }, {
        "episode_index": 15, "form": "PASS_LANDMARK",
        "landmark": "billiard table and window",
        "target": "the stopping area by the window beyond the billiard table",
        "arrival": "the billiard table is behind and the window-side stop is reached",
    }, {
        "episode_index": 32, "form": "ENTER_REGION",
        "landmark": "hallway, left room, and massage table",
        "target": "inside the room on the left next to the massage table",
        "arrival": "the hallway doorway is behind and the massage table is nearby",
    }, {
        "episode_index": 36, "form": "TURN_RIGHT",
        "landmark": "right turn and bedroom doorway",
        "target": "the bedroom doorway reached after turning right and walking straight",
        "arrival": "the right turn is complete and the agent stands in the bedroom doorway",
    }, {
        "episode_index": 59, "form": "VERTICAL_UP",
        "landmark": "first staircase and the base of the next staircase",
        "target": "the base of the second staircase after climbing the first",
        "arrival": "the first stairs are below or behind and the next stair base is reached",
    }, {
        "episode_index": 84, "form": "APPROACH_LANDMARK",
        "landmark": "display case, eye-chart hallway, and beige couch",
        "target": "a safe stopping area in front of the beige six-pillow couch",
        "arrival": "the waiting area is reached and the beige couch is close in front",
    }, {
        "episode_index": 108, "form": "VERTICAL_DOWN",
        "landmark": "descending stairs, doorways, and glass-painting room",
        "target": "the lower entrance to the room with the glass painting",
        "arrival": "the stairs have been descended and the glass-painting room entrance is reached",
    }, {
        "episode_index": 119, "form": "ENTER_REGION",
        "landmark": "current bedroom exit and next-door bedroom",
        "target": "inside the bedroom next door after leaving the current bedroom",
        "arrival": "the source bedroom is behind and the next-door bedroom is entered",
    }, {
        "episode_index": 242, "form": "BETWEEN_OBJECTS",
        "landmark": "two couches, open area, and stairway",
        "target": "the bottom of the stairs beyond the gap between the couches",
        "arrival": "the couch gap is behind and the bottom of the stairs is reached",
    }, {
        "episode_index": 342, "form": "CROSS_SPACE",
        "landmark": "room and far doorway",
        "target": "the far side of the room by the doorway",
        "arrival": "the room is crossed and the doorway-side stop is reached",
    },
)
CATEGORY_ORDER = ("correct", "wrong", "halfway")
HIDDEN_LABEL_BY_CATEGORY = {
    "correct": "completed", "wrong": "unknown", "halfway": "unknown"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sub_instruction_mapping(base, episode):
    definition = FORM_DEFINITIONS[base["form"]]
    text = episode["instruction"]["instruction_text"].strip()
    return {
        "sub_instruction_id": 0,
        "navigation_instruction": text,
        "landmark": base["landmark"],
        "form": base["form"],
        "secondary_forms": [],
        "definition": definition["definition"],
        "semantic_spatial_target": base["target"],
        "spatial_relation": base["form"].lower(),
        "completion_cue": base["arrival"],
        "visual_arrival_evidence": base["arrival"],
        "forbidden_target": definition["forbidden"],
        "source_clause": text,
        "metadata": {
            "construction": "frozen_full_instruction_single_stage_holdout20"},
    }


def build_frozen_cases(episodes):
    if len(HOLDOUT_BASE_CASES) != 10:
        raise RuntimeError("holdout protocol requires exactly 10 base trajectories")
    cases = []
    trajectory_ids = set()
    for base_index, base in enumerate(HOLDOUT_BASE_CASES):
        episode_index = int(base["episode_index"])
        if episode_index in DEVELOPMENT_EPISODE_INDICES:
            raise RuntimeError(f"holdout episode overlaps development: {episode_index}")
        episode = episodes[episode_index]
        trajectory_id = str(episode.get("trajectory_id"))
        if trajectory_id in trajectory_ids:
            raise RuntimeError(f"duplicate holdout trajectory {trajectory_id}")
        trajectory_ids.add(trajectory_id)
        path = episode.get("reference_path", [])
        if len(path) < 5:
            raise RuntimeError(f"holdout episode {episode_index} has fewer than 5 states")
        sub_instruction = sub_instruction_mapping(base, episode)
        negative_category = "wrong" if base_index % 2 == 0 else "halfway"
        for category in ("correct", negative_category):
            previous_index, current_index, label = construction.category_indices(
                len(path), category)
            previous_yaw, yaw_source = construction.reference_yaw(
                episode, path, previous_index)
            cases.append({
                "case_index": len(cases),
                "base_case_index": base_index,
                "category": category,
                "manual_edge_event_label": label,
                "manual_label_rationale": {
                    "correct": (
                        "full official reference trajectory from start to its "
                        "demonstrated instruction endpoint"),
                    "wrong": (
                        "the full official trajectory replayed endpoint-to-start, "
                        "opposite to the active instruction"),
                    "halfway": (
                        "only the first internal reference segment, before the "
                        "demonstrated instruction endpoint"),
                }[category],
                "episode_index": episode_index,
                "episode_id": episode.get("episode_id"),
                "trajectory_id": episode.get("trajectory_id"),
                "scene_id": episode.get("scene_id"),
                "reference_path_length": len(path),
                "previous_path_index": previous_index,
                "current_path_index": current_index,
                "previous_position_xyz": path[previous_index],
                "current_position_xyz": path[current_index],
                "previous_yaw_rad": previous_yaw,
                "previous_yaw_source": yaw_source,
                "full_instruction": episode["instruction"]["instruction_text"],
                "sub_instruction": sub_instruction,
                "alignment": {
                    "method": "frozen_full_instruction_single_stage_holdout20",
                    "stage_id": 0, "stage_interval": [0, len(path) - 1],
                    "alignment_uncertain": False,
                },
            })
    counts = Counter(case["category"] for case in cases)
    labels = Counter(case["manual_edge_event_label"] for case in cases)
    if counts != {"correct": 10, "wrong": 5, "halfway": 5}:
        raise RuntimeError(f"unexpected holdout category balance: {counts}")
    if labels != {"completed": 10, "unknown": 10}:
        raise RuntimeError(f"unexpected holdout label balance: {labels}")
    return cases


def initialize_manifest(output_root, dataset_path, episodes, cases, device,
                        backend_name, prompt_version):
    development_trajectory_ids = {
        str(episodes[index].get("trajectory_id"))
        for index in DEVELOPMENT_EPISODE_INDICES}
    holdout_trajectory_ids = {str(case["trajectory_id"]) for case in cases}
    overlap = sorted(development_trajectory_ids & holdout_trajectory_ids)
    if overlap:
        raise RuntimeError(f"holdout trajectories overlap development: {overlap}")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "trajectory_disjoint_instruction_completion_holdout20",
        "benchmark": "R2R", "split": "val_unseen",
        "dataset_path": str(Path(dataset_path).resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "selection_frozen_before_any_vlm_call": True,
        "selection_used_completion_model_outputs": False,
        "development_episode_overlap": [],
        "development_trajectory_overlap": overlap,
        "prompt_version": prompt_version,
        "backend": backend_name,
        "model": default_vlm_model(backend_name),
        "case_count": 20,
        "category_counts": {"correct": 10, "wrong": 5, "halfway": 5},
        "label_counts": {"completed": 10, "unknown": 10},
        "sample_design": {
            "base_trajectory_count": 10,
            "completed_per_trajectory": 1,
            "negative_per_trajectory": 1,
            "negative_assignment": "alternating reversed and halfway by frozen base order",
            "completed_edge": "reference_path[0] -> reference_path[-1]",
            "reversed_edge": "reference_path[-1] -> reference_path[0]",
            "halfway_edge": "reference_path[0] -> reference_path[1]",
        },
        "edge_construction": {
            "controller": "deterministic Habitat navmesh reference-edge replay",
            "teleport_between_edge_endpoints": False,
            "linear_step_m": 0.22, "turn_step_deg": 15.0,
            "keyframe_count": 5,
        },
        "model_input_contract": [
            "active_sub_instruction_without_expected_label",
            "previous_and_current_eight_view_rgb",
            "chronological_edge_keyframes",
            "structured_actual_action_history_and_keyframe_poses",
            "previous_and_current_dino_sam_semantics",
            "previous_and_current_online_pose_and_visual_embedding",
        ],
        "forbidden_model_inputs": [
            "case_category", "expected_status", "manual_label_rationale",
            "episode_id", "trajectory_id", "reference_path_indices",
            "future_reference_path", "future_waypoint",
            "demonstration_heading", "earlier_model_result",
        ],
        "device": device,
        "code_hashes": {
            "evaluator": sha256_file(Path(__file__)),
            "constructor": sha256_file(
                ROOT / "scripts/evaluate_instruction_completion_triplets.py"),
            "vlm_harness": sha256_file(ROOT / "scripts/vlm_harness.py"),
            "structured_evidence": sha256_file(
                ROOT / "scripts/instruction_completion_evidence.py"),
            "project_rules": sha256_file(ROOT / "project_rulle.md"),
        },
        "cases": cases,
    }
    path = Path(output_root) / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text())
        for key in ("dataset_sha256", "sample_design", "cases", "prompt_version"):
            if existing.get(key) != manifest.get(key):
                raise RuntimeError(f"refusing to alter frozen manifest {path}")
        return existing
    construction.write_json(path, manifest)
    return manifest


def load_images(root, records):
    return [np.asarray(Image.open(Path(root) / record["image_path"]).convert("RGB"))
            for record in sorted(records, key=lambda item: int(item["view_index"]))]


def evaluate_case(case, case_dir, backend_name, deepseek_env, timeout,
                  prompt_version):
    result_path = case_dir / "instruction_completion.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    graph = json.loads(
        (case_dir / "navigation_graph/navigation_graph.json").read_text())
    previous, current = graph["nodes"][:2]
    edge = graph["edges"][0]
    graph_root = case_dir / "navigation_graph"
    previous_views = load_images(
        graph_root,
        previous["metadata"]["instruction_completion_panorama"]["views"])
    current_views = load_images(
        graph_root,
        current["metadata"]["instruction_completion_panorama"]["views"])
    keyframes = [np.asarray(Image.open(
        case_dir / item["image_path"]).convert("RGB"))
        for item in edge["metadata"]["edge_keyframes"]]
    backend = build_vlm_backend(
        backend_name, timeout=timeout, deepseek_env_file=deepseek_env)
    harness = NavigationVLMHarness(
        backend, case_dir / "vlm_calls.json", retries=2,
        instruction_completion_prompt_version=prompt_version)
    try:
        prediction = harness.judge_edge_instruction_completion(
            sub_instruction=SubInstruction.from_mapping(case["sub_instruction"]),
            previous_node_id=previous["node_id"],
            current_node_id=current["node_id"],
            previous_position_xyz=previous["position_xyz"],
            current_position_xyz=current["position_xyz"],
            previous_six_views=previous_views,
            current_six_views=current_views,
            previous_environment_semantics=previous["environment_semantics"],
            current_environment_semantics=current["environment_semantics"],
            edge_action_history=edge["action_history"], edge_keyframes=keyframes,
            previous_base_yaw_rad=previous.get("base_yaw_rad"),
            current_base_yaw_rad=current.get("base_yaw_rad"),
            previous_visual_embedding=previous.get("visual_embedding"),
            current_visual_embedding=current.get("visual_embedding"),
            edge_keyframe_records=edge.get("metadata", {}).get("edge_keyframes", []))
    except RuntimeError as exc:
        prediction = {
            "status": "error", "confidence": 0.0,
            "reason": str(exc), "visual_evidence": "",
            "failure_counts_in_frozen_denominator": True,
        }

    # Hidden scoring starts only after the VLM response is complete.
    expected = HIDDEN_LABEL_BY_CATEGORY[case["category"]]
    record = {
        "case_index": int(case["case_index"]),
        "category": case["category"],
        "expected_status": expected,
        "predicted_status": prediction["status"],
        "correct": prediction["status"] == expected,
        "confidence": float(prediction.get("confidence", 0.0)),
        "result": prediction,
        "label_exposed_to_vlm": False,
        "future_demonstration_exposed_to_vlm": False,
    }
    construction.write_json(result_path, record)
    construction.write_json(case_dir / "vlm_response.json", prediction)
    prompt_record = harness.calls[-1] if harness.calls else harness.attempts[0]
    (case_dir / "vlm_prompt.txt").write_text(prompt_record["prompt"] + "\n")
    Image.fromarray(harness._completion_contact_sheet(previous_views)).save(
        case_dir / "vlm_previous_contact_sheet.jpg", quality=95)
    Image.fromarray(harness._completion_contact_sheet(current_views)).save(
        case_dir / "vlm_current_contact_sheet.jpg", quality=95)
    construction.write_json(case_dir / "model_input_audit.json", {
        "prompt_sha256": hashlib.sha256(
            prompt_record["prompt"].encode("utf-8")).hexdigest(),
        "prompt_version": prompt_version,
        "allowed_runtime_fields_only": True,
        "label_exposed_to_vlm": False,
        "category_exposed_to_vlm": False,
        "episode_or_trajectory_identity_exposed_to_vlm": False,
        "reference_path_or_future_waypoint_exposed_to_vlm": False,
        "earlier_model_result_exposed_to_vlm": False,
        "expected_status_applied_after_model_call": True,
    })
    construction.write_json(case_dir / "module_results.json", {
        "test_scope": "single_point_module_test",
        "target_module": "binary_edge_instruction_completion",
        "module_success": record["correct"], "prediction": record,
        "not_run_modules": [
            "point_selection", "point_navigation_executor",
            "point_target_arrival", "physical_backtracking"],
        "all_modules_success": "not_applicable",
    })
    return record


def wilson(successes, total, z=1.959963984540054):
    if total == 0:
        return [0.0, 0.0]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize(results):
    categories = {}
    for category in CATEGORY_ORDER:
        values = [item for item in results if item["category"] == category]
        correct = sum(bool(item["correct"]) for item in values)
        categories[category] = {
            "count": len(values), "correct": correct,
            "accuracy": correct / max(len(values), 1),
            "predicted_completed": sum(
                item["predicted_status"] == "completed" for item in values),
            "predicted_unknown": sum(
                item["predicted_status"] == "unknown" for item in values),
            "prediction_errors": sum(
                item["predicted_status"] == "error" for item in values),
        }
    positives = [item for item in results if item["expected_status"] == "completed"]
    negatives = [item for item in results if item["expected_status"] == "unknown"]
    tp = sum(item["predicted_status"] == "completed" for item in positives)
    fn = sum(item["predicted_status"] != "completed" for item in positives)
    fp = sum(item["predicted_status"] == "completed" for item in negatives)
    tn = sum(item["predicted_status"] == "unknown" for item in negatives)
    completed_precision = tp / max(tp + fp, 1)
    completed_recall = tp / max(tp + fn, 1)
    completed_f1 = (2 * completed_precision * completed_recall /
                    max(completed_precision + completed_recall, 1e-12))
    unknown_precision = tn / max(tn + fn, 1)
    unknown_recall = tn / max(tn + fp, 1)
    unknown_f1 = (2 * unknown_precision * unknown_recall /
                  max(unknown_precision + unknown_recall, 1e-12))
    correct = sum(bool(item["correct"]) for item in results)
    return {
        "count": len(results), "correct": correct,
        "accuracy": correct / max(len(results), 1),
        "wilson_95_ci": wilson(correct, len(results)),
        "categories": categories,
        "confusion": {"true_positive": tp, "false_negative": fn,
                      "false_positive": fp, "true_negative": tn},
        "completed_precision": completed_precision,
        "completed_recall": completed_recall,
        "completed_f1": completed_f1,
        "unknown_precision": unknown_precision,
        "unknown_recall": unknown_recall,
        "unknown_f1": unknown_f1,
        "macro_f1": (completed_f1 + unknown_f1) / 2,
        "weighted_f1": (
            len(positives) * completed_f1 + len(negatives) * unknown_f1) /
            max(len(results), 1),
        "prediction_errors": sum(
            item["predicted_status"] == "error" for item in results),
        "holdout_not_used_for_prompt_tuning_before_evaluation": True,
    }


def audit_prepared(output_root, cases):
    problems = []
    for case in cases:
        case_dir = output_root / f"case_{case['case_index']:03d}_{case['category']}"
        graph_path = case_dir / "navigation_graph/navigation_graph.json"
        result_path = case_dir / "preparation_result.json"
        if not graph_path.exists() or not result_path.exists():
            problems.append(f"missing prepared artifacts for case {case['case_index']}")
            continue
        graph = json.loads(graph_path.read_text())
        prepared = json.loads(result_path.read_text())
        if len(graph.get("nodes", [])) != 2 or len(graph.get("edges", [])) != 1:
            problems.append(f"invalid graph cardinality case {case['case_index']}")
        if not prepared.get("exact_current_reference_position"):
            problems.append(f"endpoint mismatch case {case['case_index']}")
        if (case_dir / "vlm_calls.json").exists():
            problems.append(f"VLM call existed before audit case {case['case_index']}")
    audit = {
        "status": "accepted" if not problems else "rejected",
        "case_count": len(cases),
        "labels_frozen_before_vlm_calls": True,
        "vlm_calls_at_audit": 0 if not problems else None,
        "real_habitat_graphs_complete": not problems,
        "episode_overlap_with_development30": [],
        "trajectory_overlap_with_development30": [],
        "balanced_binary_labels": True,
        "same_prompt_schema_validator_for_all_cases": True,
        "case_level_result_rules": False,
        "problems": problems,
    }
    construction.write_json(output_root / "pre_vlm_leakage_audit.json", audit)
    if problems:
        raise RuntimeError("holdout preparation audit failed: " + "; ".join(problems))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["prepare", "evaluate", "all"],
                        default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--r2r-data", type=Path,
                        default=construction.DEFAULT_R2R_DATA)
    parser.add_argument("--mp3d-root", type=Path,
                        default=construction.DEFAULT_MP3D_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vlm-backend",
                        choices=["deepseek", "ollama", "heuristic"],
                        default="deepseek")
    parser.add_argument("--deepseek-env", type=Path,
                        default=ROOT / ".env.deepseek")
    parser.add_argument("--vlm-timeout", type=int, default=240)
    parser.add_argument(
        "--prompt-version", default=PROMPT_VERSION,
        choices=sorted(
            NavigationVLMHarness.INSTRUCTION_COMPLETION_PROMPT_VERSIONS))
    parser.add_argument(
        "--reuse-prepared-root", type=Path, default=None,
        help=("reuse only label-frozen Habitat replay artifacts from a prior "
              "holdout root; prior VLM outputs are never linked"))
    args = parser.parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    episodes = construction.load_dataset(args.r2r_data)
    cases = build_frozen_cases(episodes)
    initialize_manifest(
        args.output_root, args.r2r_data, episodes, cases,
        args.device, args.vlm_backend, args.prompt_version)

    if args.phase in {"prepare", "all"}:
        if args.reuse_prepared_root is not None:
            reusable = (
                "arrival_depths", "arrival_six_views", "edge_keyframes",
                "exploration.mp4", "forward_action_history.json",
                "initial_depths", "initial_six_views", "manifest.json",
                "navigation_graph", "preparation_result.json",
                "semantic_detections.json", "trajectory.json")
            for case in cases:
                name = f"case_{case['case_index']:03d}_{case['category']}"
                source_dir = args.reuse_prepared_root / name
                case_dir = args.output_root / name
                case_dir.mkdir(parents=True, exist_ok=True)
                for artifact in reusable:
                    source = source_dir / artifact
                    if not source.exists():
                        raise RuntimeError(
                            f"missing reusable artifact: {source}")
                    target = case_dir / artifact
                    if target.exists() or target.is_symlink():
                        raise RuntimeError(
                            f"refusing to replace reusable artifact: {target}")
                    target.symlink_to(
                        source.resolve(), target_is_directory=source.is_dir())
                print(f"reused frozen preparation case={case['case_index']:02d}",
                      flush=True)
        else:
            detector = DinoSamDetector(args.device)
            semantics = DinoSamEnvironmentSemanticExtractor(detector)
            for case in cases:
                case_dir = construction.prepare_case(
                    case, episodes[case["episode_index"]], args.output_root,
                    args.mp3d_root, semantics)
                print(
                    f"prepared case={case['case_index']:02d} "
                    f"category={case['category']} episode={case['episode_index']} "
                    f"dir={case_dir}", flush=True)
        audit_prepared(args.output_root, cases)
        construction.write_json(args.output_root / "preparation_complete.json", {
            "prepared_case_count": len(cases), "vlm_called": False,
            "labels_frozen": True, "pre_vlm_audit": "accepted"})

    if args.phase in {"evaluate", "all"}:
        audit_path = args.output_root / "pre_vlm_leakage_audit.json"
        if not audit_path.exists() or json.loads(
                audit_path.read_text()).get("status") != "accepted":
            raise RuntimeError("accepted pre-VLM audit is required before evaluation")
        results = []
        for case in cases:
            case_dir = (args.output_root /
                        f"case_{case['case_index']:03d}_{case['category']}")
            result = evaluate_case(
                case, case_dir, args.vlm_backend,
                args.deepseek_env, args.vlm_timeout, args.prompt_version)
            results.append(result)
            print(
                f"evaluated case={case['case_index']:02d} "
                f"category={case['category']} expected={result['expected_status']} "
                f"predicted={result['predicted_status']} "
                f"correct={int(result['correct'])}", flush=True)
        summary = summarize(results)
        construction.write_json(args.output_root / "results.json", results)
        construction.write_json(args.output_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
