#!/usr/bin/env python3
"""Completed/unknown prompt evaluation on the frozen 30 real R2R edges."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

import evaluate_instruction_progress_triplets as source_tools
from instruction_decomposer import SubInstruction
from vlm_harness import NavigationVLMHarness, build_vlm_backend, default_vlm_model


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_VERSION = "v13_structured_node_edge_binary"
DEFAULT_SOURCE_ROOT = source_tools.DEFAULT_SOURCE_ROOT
DEFAULT_COLLAPSED_BASELINE_ROOT = (
    ROOT / "outputs/instruction_progress_triplets_30_v9_frozen_20260831")
CATEGORY_ORDER = ("correct", "wrong", "halfway")
HIDDEN_LABEL_BY_CATEGORY = {
    "correct": "completed", "wrong": "unknown", "halfway": "unknown"}


def output_case_dir(output_root, case):
    return Path(output_root) / (
        f"case_{int(case['case_index']):03d}_{case['category']}")


def collapsed_v9_baseline(primary_root, cases):
    records = []
    for case in cases:
        path = (Path(primary_root) /
                f"case_{int(case['case_index']):03d}_{case['category']}" /
                "vlm_response.json")
        raw = json.loads(path.read_text())
        if any(key in raw for key in (
                "expected_status", "correct", "category", "case_index")):
            raise RuntimeError(f"collapsed baseline is not label-free: {path}")
        predicted = "completed" if raw.get("status") == "arrived" else "unknown"
        expected = HIDDEN_LABEL_BY_CATEGORY[case["category"]]
        records.append({
            "case_index": int(case["case_index"]),
            "category": case["category"], "expected_status": expected,
            "source_three_way_status": raw.get("status"),
            "predicted_status": predicted, "correct": predicted == expected,
            "source_vlm_response_sha256": source_tools.sha256_file(path),
        })
    return summarize(records)


def prepare(output_root, source_root, baseline_root, backend_name, deepseek_env,
            prompt_version):
    output_root = Path(output_root)
    source_root = Path(source_root).resolve()
    _, cases = source_tools.load_source_cases(source_root)
    output_root.mkdir(parents=True, exist_ok=True)
    root_manifest = {
        "experiment": "binary_edge_instruction_completion_30",
        "test_scope": "single_point_module_test",
        "target_modules": ["binary_edge_instruction_completion"],
        "required_upstream_modules": [
            "real_reference_state_pair", "actual_habitat_edge_replay",
            "six_and_eight_view_capture", "dino_sam_node_semantics"],
        "upstream_artifact_source": str(source_root),
        "upstream_manifest_sha256": source_tools.sha256_file(
            source_root / "manifest.json"),
        "prompt_version": prompt_version,
        "backend": backend_name,
        "model": default_vlm_model(backend_name),
        "deepseek_env_path": str(Path(deepseek_env).resolve()),
        "hidden_scoring_mapping": HIDDEN_LABEL_BY_CATEGORY,
        "unknown_semantics": (
            "includes on-route, partial, wrong, reverse, ambiguous, and unsupported"),
        "label_exposed_to_vlm": False,
        "future_demonstration_exposed_to_vlm": False,
        "case_level_rules": False,
        "case_count": 30,
        "category_counts": {category: 10 for category in CATEGORY_ORDER},
        "code_hashes": {
            "evaluator": source_tools.sha256_file(Path(__file__)),
            "vlm_harness": source_tools.sha256_file(
                ROOT / "scripts/vlm_harness.py"),
            "edge_judge": source_tools.sha256_file(
                ROOT / "scripts/instruction_completion_judge.py"),
            "project_rules": source_tools.sha256_file(ROOT / "project_rulle.md"),
        },
        "model_input_contract": [
            "active_sub_instruction_without_expected_label",
            "previous_and_current_eight_view_rgb",
            "previous_and_current_dino_sam_semantics",
            "previous_and_current_online_xyz",
            "actual_edge_action_history", "chronological_edge_keyframes"],
        "forbidden_model_inputs": [
            "category", "expected_status", "manual_label_rationale",
            "episode_id", "trajectory_id", "reference_path_index",
            "future_reference_path", "future_waypoint", "demonstration_heading",
            "earlier_model_result"],
        "cases": [],
    }
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        keys = (
            "experiment", "upstream_manifest_sha256", "prompt_version",
            "hidden_scoring_mapping", "code_hashes")
        if any(existing.get(key) != root_manifest.get(key) for key in keys):
            raise RuntimeError(f"refusing to alter frozen manifest {manifest_path}")
        return cases

    for case in cases:
        source_dir = source_tools.source_case_dir(source_root, case)
        target_dir = output_case_dir(output_root, case)
        target_dir.mkdir(parents=True, exist_ok=True)
        graph = json.loads(
            (source_dir / "navigation_graph/navigation_graph.json").read_text())
        artifacts, aggregate = source_tools.hash_records(source_dir, graph)
        for name in (
                "exploration.mp4", "forward_action_history.json",
                "semantic_detections.json", "trajectory.json",
                "navigation_graph", "edge_keyframes", "initial_six_views",
                "initial_depths", "arrival_six_views", "arrival_depths"):
            source = source_dir / name
            if source.exists():
                source_tools.safe_symlink(source, target_dir / name)
        case_manifest = {
            "case_index": int(case["case_index"]),
            "category": case["category"],
            "test_scope": "single_point_module_test",
            "target_modules": ["binary_edge_instruction_completion"],
            "required_upstream_modules": root_manifest[
                "required_upstream_modules"],
            "upstream_artifact_source": str(source_dir),
            "upstream_artifact_aggregate_sha256": aggregate,
            "upstream_artifacts": artifacts,
            "state_identity": {
                key: case.get(key) for key in (
                    "episode_index", "episode_id", "scene_id", "trajectory_id",
                    "previous_path_index", "current_path_index",
                    "previous_position_xyz", "current_position_xyz")},
            "sub_instruction": case["sub_instruction"],
            "hidden_expected_status": HIDDEN_LABEL_BY_CATEGORY[case["category"]],
            "label_frozen_before_vlm_call": True,
            "label_exposed_to_vlm": False,
            "not_run_modules": [
                "point_selection", "point_navigation_executor",
                "point_target_arrival", "physical_backtracking"],
        }
        source_tools.write_json(target_dir / "manifest.json", case_manifest)
        root_manifest["cases"].append({
            "case_index": int(case["case_index"]),
            "category": case["category"],
            "hidden_expected_status": HIDDEN_LABEL_BY_CATEGORY[case["category"]],
            "upstream_artifact_aggregate_sha256": aggregate})
    source_tools.write_json(manifest_path, root_manifest)
    source_tools.write_json(
        output_root / "collapsed_v9_binary_baseline.json",
        collapsed_v9_baseline(baseline_root, cases))
    source_tools.write_json(output_root / "pre_vlm_leakage_audit.json", {
        "status": "accepted_before_binary_vlm_calls",
        "binary_vlm_calls_at_audit": 0,
        "case_count": 30,
        "labels_frozen": True,
        "labels_exposed_to_vlm": False,
        "future_demonstration_exposed_to_vlm": False,
        "earlier_model_result_exposed_to_binary_prompt": False,
        "same_prompt_schema_validator_for_all_cases": True,
        "case_level_rules": False,
        "source_artifacts_hashed_and_state_identity_recorded": True,
    })
    return cases


def evaluate_case(case, source_root, output_root, backend_name,
                  deepseek_env, timeout, prompt_version):
    source_dir = source_tools.source_case_dir(source_root, case)
    target_dir = output_case_dir(output_root, case)
    result_path = target_dir / "instruction_completion.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    graph = json.loads(
        (source_dir / "navigation_graph/navigation_graph.json").read_text())
    previous, current = graph["nodes"][:2]
    edge = graph["edges"][0]
    graph_root = source_dir / "navigation_graph"
    previous_views = source_tools.load_images(
        graph_root,
        previous["metadata"]["instruction_completion_panorama"]["views"])
    current_views = source_tools.load_images(
        graph_root,
        current["metadata"]["instruction_completion_panorama"]["views"])
    keyframes = [
        np.asarray(Image.open(source_dir / item["image_path"]).convert("RGB"))
        for item in edge["metadata"]["edge_keyframes"]]
    backend = build_vlm_backend(
        backend_name, timeout=timeout, deepseek_env_file=deepseek_env)
    harness = NavigationVLMHarness(
        backend, target_dir / "vlm_calls.json", retries=2,
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
            edge_keyframe_records=edge.get("metadata", {}).get(
                "edge_keyframes", []))
    except RuntimeError as exc:
        prediction = {
            "status": "error", "confidence": 0.0, "reason": str(exc),
            "visual_evidence": "", "failure_counts_in_frozen_denominator": True}
    expected = HIDDEN_LABEL_BY_CATEGORY[case["category"]]
    record = {
        "case_index": int(case["case_index"]), "category": case["category"],
        "expected_status": expected, "predicted_status": prediction["status"],
        "correct": prediction["status"] == expected,
        "confidence": float(prediction.get("confidence", 0.0)),
        "result": prediction, "label_exposed_to_vlm": False,
        "future_demonstration_exposed_to_vlm": False}
    source_tools.write_json(result_path, record)
    source_tools.write_json(target_dir / "vlm_response.json", prediction)
    prompt_record = harness.calls[-1] if harness.calls else harness.attempts[0]
    (target_dir / "vlm_prompt.txt").write_text(prompt_record["prompt"] + "\n")
    previous_sheet = harness._completion_contact_sheet(previous_views)
    current_sheet = harness._completion_contact_sheet(current_views)
    Image.fromarray(previous_sheet).save(
        target_dir / "vlm_previous_contact_sheet.jpg", quality=95)
    Image.fromarray(current_sheet).save(
        target_dir / "vlm_current_contact_sheet.jpg", quality=95)
    Image.fromarray(np.concatenate([previous_sheet, current_sheet], axis=0)).save(
        target_dir / "vlm_contact_sheet.jpg", quality=95)
    source_tools.write_json(target_dir / "model_input_audit.json", {
        "prompt_sha256": hashlib.sha256(
            prompt_record["prompt"].encode("utf-8")).hexdigest(),
        "prompt_version": prompt_version,
        "label_exposed_to_vlm": False, "category_exposed_to_vlm": False,
        "future_reference_or_waypoint_exposed_to_vlm": False,
        "earlier_model_result_exposed_to_vlm": False,
        "hidden_score_applied_after_model_call": True})
    source_tools.write_json(target_dir / "module_results.json", {
        "test_scope": "single_point_module_test",
        "target_module": "binary_edge_instruction_completion",
        "module_success": record["correct"], "prediction": record,
        "not_run_modules": [
            "point_selection", "point_navigation_executor",
            "point_target_arrival", "physical_backtracking"],
        "all_modules_success": "not_applicable"})
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
            "expected_status": HIDDEN_LABEL_BY_CATEGORY[category],
            "count": len(values), "correct": correct,
            "accuracy": correct / max(len(values), 1),
            "wilson_95_ci": wilson(correct, len(values)),
            "predicted_completed": sum(
                item["predicted_status"] == "completed" for item in values),
            "predicted_unknown": sum(
                item["predicted_status"] == "unknown" for item in values),
            "prediction_errors": sum(
                item["predicted_status"] == "error" for item in values)}
    positives = [item for item in results if item["expected_status"] == "completed"]
    negatives = [item for item in results if item["expected_status"] == "unknown"]
    tp = sum(item["predicted_status"] == "completed" for item in positives)
    fn = sum(item["predicted_status"] == "unknown" for item in positives)
    fp = sum(item["predicted_status"] == "completed" for item in negatives)
    tn = sum(item["predicted_status"] == "unknown" for item in negatives)
    errors = sum(item["predicted_status"] == "error" for item in results)
    correct = sum(bool(item["correct"]) for item in results)
    return {
        "count": len(results), "correct": correct,
        "accuracy": correct / max(len(results), 1),
        "wilson_95_ci": wilson(correct, len(results)),
        "categories": categories,
        "confusion": {"true_positive": tp, "false_negative": fn,
                      "false_positive": fp, "true_negative": tn},
        "completed_precision": tp / max(tp + fp, 1),
        "completed_recall": tp / max(len(positives), 1),
        "unknown_recall": tn / max(len(negatives), 1),
        "prediction_errors": errors,
        "label_semantics": {
            "correct": "completed", "wrong": "unknown",
            "halfway": "unknown", "unknown_includes_on_route": True},
        "main_metric_uses_all_frozen_cases": True}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["prepare", "evaluate", "all"],
                        default="all")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--collapsed-baseline-root", type=Path,
                        default=DEFAULT_COLLAPSED_BASELINE_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--vlm-backend",
                        choices=["deepseek", "ollama", "heuristic"],
                        default="deepseek")
    parser.add_argument("--deepseek-env", type=Path,
                        default=ROOT / ".env.deepseek")
    parser.add_argument("--vlm-timeout", type=int, default=180)
    parser.add_argument(
        "--prompt-version", default=DEFAULT_PROMPT_VERSION,
        choices=sorted(NavigationVLMHarness.INSTRUCTION_COMPLETION_PROMPT_VERSIONS))
    args = parser.parse_args(argv)
    if args.phase in {"prepare", "all"}:
        cases = prepare(
            args.output_root, args.source_root, args.collapsed_baseline_root,
            args.vlm_backend, args.deepseek_env, args.prompt_version)
        print(f"prepared {len(cases)} frozen binary cases", flush=True)
    else:
        _, cases = source_tools.load_source_cases(args.source_root)
        if not (args.output_root / "pre_vlm_leakage_audit.json").exists():
            raise RuntimeError("prepare phase and leakage audit must run first")
    if args.phase in {"evaluate", "all"}:
        results = []
        for case in cases:
            result = evaluate_case(
                case, args.source_root, args.output_root, args.vlm_backend,
                args.deepseek_env, args.vlm_timeout, args.prompt_version)
            results.append(result)
            print(
                f"evaluated case={int(case['case_index']):02d} "
                f"category={case['category']} "
                f"expected={result['expected_status']} "
                f"predicted={result['predicted_status']} "
                f"correct={int(result['correct'])}", flush=True)
        summary = summarize(results)
        source_tools.write_json(args.output_root / "results.json", results)
        source_tools.write_json(args.output_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
