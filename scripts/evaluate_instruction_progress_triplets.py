#!/usr/bin/env python3
"""Evaluate three-way edge instruction state on frozen real R2R artifacts.

The VLM sees only online-available previous/current node observations, semantics,
positions, actual edge actions, keyframes, and the active sub-instruction.  The
frozen category-to-label mapping is applied only after each model call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from instruction_decomposer import SubInstruction
from vlm_harness import (
    NavigationVLMHarness, build_vlm_backend, default_vlm_model,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = (
    ROOT / "outputs/instruction_completion_triplets_30_v8_v3_final_20260831")
DEFAULT_PROMPT_VERSION = "v9_three_way_edge_progress"
CATEGORY_ORDER = ("correct", "wrong", "halfway")
STATUS_ORDER = ("arrived", "on_route", "unknown")
HIDDEN_LABEL_BY_CATEGORY = {
    "correct": "arrived",
    "wrong": "unknown",
    "halfway": "on_route",
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_records(case_dir, graph):
    relative_paths = {
        Path("manifest.json"),
        Path("navigation_graph/navigation_graph.json"),
        Path("forward_action_history.json"),
        Path("semantic_detections.json"),
        Path("trajectory.json"),
        Path("exploration.mp4"),
    }
    for node in graph["nodes"][:2]:
        panorama = node["metadata"]["instruction_completion_panorama"]
        relative_paths.update(
            Path("navigation_graph") / item["image_path"]
            for item in panorama["views"])
    relative_paths.update(
        Path(item["image_path"])
        for item in graph["edges"][0]["metadata"]["edge_keyframes"])
    records = []
    aggregate = hashlib.sha256()
    for relative in sorted(relative_paths, key=str):
        path = case_dir / relative
        if not path.is_file():
            raise RuntimeError(f"missing frozen upstream artifact {path}")
        digest = sha256_file(path)
        records.append({"path": str(relative), "sha256": digest})
        aggregate.update(str(relative).encode("utf-8"))
        aggregate.update(digest.encode("ascii"))
    return records, aggregate.hexdigest()


def safe_symlink(source, target):
    source = Path(source).resolve()
    target = Path(target)
    if target.is_symlink():
        if target.resolve() != source:
            raise RuntimeError(f"refusing to replace mismatched link {target}")
        return
    if target.exists():
        raise RuntimeError(f"refusing to replace existing artifact {target}")
    target.symlink_to(source, target_is_directory=source.is_dir())


def source_case_dir(source_root, case):
    return Path(source_root) / (
        f"case_{int(case['case_index']):03d}_{case['category']}")


def output_case_dir(output_root, case):
    return Path(output_root) / (
        f"case_{int(case['case_index']):03d}_{case['category']}")


def load_source_cases(source_root):
    manifest = json.loads((Path(source_root) / "manifest.json").read_text())
    cases = sorted(manifest["cases"], key=lambda item: int(item["case_index"]))
    if len(cases) != 30:
        raise RuntimeError(f"expected 30 frozen source cases, found {len(cases)}")
    counts = {category: 0 for category in CATEGORY_ORDER}
    for case in cases:
        category = str(case["category"])
        if category not in counts:
            raise RuntimeError(f"unexpected frozen category {category!r}")
        counts[category] += 1
    if any(counts[category] != 10 for category in CATEGORY_ORDER):
        raise RuntimeError(f"expected 10 cases per category, found {counts}")
    return manifest, cases


def prepare(output_root, source_root, backend_name, deepseek_env,
            prompt_version):
    output_root = Path(output_root)
    source_root = Path(source_root).resolve()
    source_manifest, cases = load_source_cases(source_root)
    output_root.mkdir(parents=True, exist_ok=True)
    root_manifest_path = output_root / "manifest.json"
    frozen_contract = {
        "experiment": "three_way_edge_instruction_progress_triplets",
        "test_scope": "single_point_module_test",
        "target_modules": ["three_way_edge_instruction_progress"],
        "required_upstream_modules": [
            "real_reference_state_pair", "actual_habitat_edge_replay",
            "six_and_eight_view_capture", "dino_sam_node_semantics",
        ],
        "upstream_artifact_source": str(source_root),
        "upstream_manifest_sha256": sha256_file(source_root / "manifest.json"),
        "prompt_version": prompt_version,
        "backend": backend_name,
        "model": default_vlm_model(backend_name),
        "deepseek_env_path": str(Path(deepseek_env).resolve()),
        "hidden_scoring_mapping": HIDDEN_LABEL_BY_CATEGORY,
        "model_input_contract": [
            "active_sub_instruction_without_expected_label",
            "previous_and_current_eight_view_rgb",
            "previous_and_current_dino_sam_semantics",
            "previous_and_current_online_xyz",
            "actual_edge_action_history",
            "chronological_edge_keyframes",
        ],
        "forbidden_model_inputs": [
            "case_category", "expected_status", "manual_label_rationale",
            "episode_id", "trajectory_id", "reference_path_index",
            "future_reference_path", "future_waypoint", "demonstration_heading",
        ],
        "label_exposed_to_vlm": False,
        "optimization_scope": (
            "one frozen general prompt/schema/evidence policy; no case-level rules"),
        "case_count": len(cases),
        "category_counts": {category: 10 for category in CATEGORY_ORDER},
        "code_hashes": {
            "evaluator": sha256_file(Path(__file__)),
            "vlm_harness": sha256_file(ROOT / "scripts/vlm_harness.py"),
            "edge_judge": sha256_file(
                ROOT / "scripts/instruction_completion_judge.py"),
            "project_rules": sha256_file(ROOT / "project_rulle.md"),
        },
        "cases": [],
    }
    if root_manifest_path.exists():
        existing = json.loads(root_manifest_path.read_text())
        comparable = (
            "experiment", "upstream_manifest_sha256", "prompt_version",
            "hidden_scoring_mapping", "case_count", "code_hashes",
        )
        if any(existing.get(key) != frozen_contract.get(key) for key in comparable):
            raise RuntimeError(
                f"refusing to alter frozen evaluation manifest {root_manifest_path}")
        return cases

    for case in cases:
        source_dir = source_case_dir(source_root, case)
        graph = json.loads(
            (source_dir / "navigation_graph/navigation_graph.json").read_text())
        artifacts, aggregate_hash = hash_records(source_dir, graph)
        target_dir = output_case_dir(output_root, case)
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in (
                "exploration.mp4", "forward_action_history.json",
                "semantic_detections.json", "trajectory.json",
                "navigation_graph", "edge_keyframes", "initial_six_views",
                "initial_depths", "arrival_six_views", "arrival_depths"):
            source = source_dir / name
            if source.exists():
                safe_symlink(source, target_dir / name)
        case_manifest = {
            "case_index": int(case["case_index"]),
            "category": case["category"],
            "test_scope": "single_point_module_test",
            "target_modules": ["three_way_edge_instruction_progress"],
            "required_upstream_modules": frozen_contract[
                "required_upstream_modules"],
            "upstream_artifact_source": str(source_dir),
            "upstream_artifact_aggregate_sha256": aggregate_hash,
            "upstream_artifacts": artifacts,
            "state_identity": {
                key: case.get(key) for key in (
                    "episode_index", "episode_id", "scene_id", "trajectory_id",
                    "previous_path_index", "current_path_index",
                    "previous_position_xyz", "current_position_xyz")
            },
            "sub_instruction": case["sub_instruction"],
            "hidden_expected_status": HIDDEN_LABEL_BY_CATEGORY[case["category"]],
            "label_frozen_before_vlm_call": True,
            "label_exposed_to_vlm": False,
            "not_run_modules": [
                "point_selection", "point_navigation_executor",
                "point_target_arrival", "physical_backtracking"],
        }
        write_json(target_dir / "manifest.json", case_manifest)
        frozen_contract["cases"].append({
            "case_index": int(case["case_index"]),
            "category": case["category"],
            "hidden_expected_status": HIDDEN_LABEL_BY_CATEGORY[case["category"]],
            "upstream_artifact_aggregate_sha256": aggregate_hash,
        })
    write_json(root_manifest_path, frozen_contract)
    write_json(output_root / "pre_vlm_leakage_audit.json", {
        "status": "accepted_before_vlm_evaluation",
        "vlm_calls_at_audit": 0,
        "case_count": len(cases),
        "labels_frozen": True,
        "labels_exposed_to_vlm": False,
        "future_demonstration_exposed_to_vlm": False,
        "case_level_prompt_or_code_rules": False,
        "same_prompt_schema_and_validator_for_all_cases": True,
        "source_artifacts_hashed_and_state_identity_recorded": True,
    })
    return cases


def load_images(root, records):
    return [
        np.asarray(Image.open(Path(root) / item["image_path"]).convert("RGB"))
        for item in sorted(records, key=lambda value: int(value["view_index"]))
    ]


def evaluate_case(case, source_root, output_root, backend_name,
                  deepseek_env, timeout, prompt_version):
    source_dir = source_case_dir(source_root, case)
    target_dir = output_case_dir(output_root, case)
    result_path = target_dir / "instruction_edge_state.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    graph = json.loads(
        (source_dir / "navigation_graph/navigation_graph.json").read_text())
    previous, current = graph["nodes"][:2]
    edge = graph["edges"][0]
    graph_root = source_dir / "navigation_graph"
    previous_views = load_images(
        graph_root,
        previous["metadata"]["instruction_completion_panorama"]["views"])
    current_views = load_images(
        graph_root,
        current["metadata"]["instruction_completion_panorama"]["views"])
    keyframes = [
        np.asarray(Image.open(source_dir / item["image_path"]).convert("RGB"))
        for item in edge["metadata"]["edge_keyframes"]
    ]
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
            edge_action_history=edge["action_history"],
            edge_keyframes=keyframes)
    except RuntimeError as exc:
        prediction = {
            "status": "error", "confidence": 0.0,
            "reason": str(exc), "visual_evidence": "",
            "failure_counts_in_frozen_denominator": True,
        }

    # Hidden scoring begins only after the VLM call has returned.
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
    write_json(result_path, record)
    write_json(target_dir / "vlm_response.json", prediction)
    prompt_record = (
        harness.calls[-1] if harness.calls else harness.attempts[0])
    (target_dir / "vlm_prompt.txt").write_text(prompt_record["prompt"] + "\n")
    Image.fromarray(harness._completion_contact_sheet(previous_views)).save(
        target_dir / "vlm_previous_contact_sheet.jpg", quality=95)
    Image.fromarray(harness._completion_contact_sheet(current_views)).save(
        target_dir / "vlm_current_contact_sheet.jpg", quality=95)
    Image.fromarray(np.concatenate([
        harness._completion_contact_sheet(previous_views),
        harness._completion_contact_sheet(current_views)], axis=0)).save(
            target_dir / "vlm_contact_sheet.jpg", quality=95)
    write_json(target_dir / "model_input_audit.json", {
        "prompt_sha256": hashlib.sha256(
            prompt_record["prompt"].encode("utf-8")).hexdigest(),
        "prompt_version": prompt_version,
        "allowed_runtime_fields_only": True,
        "label_exposed_to_vlm": False,
        "category_exposed_to_vlm": False,
        "episode_or_trajectory_identity_exposed_to_vlm": False,
        "reference_path_indices_or_future_waypoints_exposed_to_vlm": False,
        "expected_status_applied_after_model_call": True,
    })
    write_json(target_dir / "module_results.json", {
        "test_scope": "single_point_module_test",
        "target_module": "three_way_edge_instruction_progress",
        "module_success": record["correct"],
        "prediction": record,
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
    confusion = {
        expected: {predicted: 0 for predicted in (*STATUS_ORDER, "error")}
        for expected in STATUS_ORDER
    }
    for item in results:
        confusion[item["expected_status"]][item["predicted_status"]] += 1
    categories = {}
    for category in CATEGORY_ORDER:
        values = [item for item in results if item["category"] == category]
        correct = sum(bool(item["correct"]) for item in values)
        categories[category] = {
            "expected_status": HIDDEN_LABEL_BY_CATEGORY[category],
            "count": len(values), "correct": correct,
            "accuracy": correct / max(len(values), 1),
            "wilson_95_ci": wilson(correct, len(values)),
            "predictions": {
                status: sum(item["predicted_status"] == status for item in values)
                for status in (*STATUS_ORDER, "error")
            },
        }
    correct = sum(bool(item["correct"]) for item in results)
    recalls = {
        status: confusion[status][status] / max(
            sum(confusion[status].values()), 1)
        for status in STATUS_ORDER
    }
    return {
        "count": len(results), "correct": correct,
        "accuracy": correct / max(len(results), 1),
        "wilson_95_ci": wilson(correct, len(results)),
        "categories": categories,
        "confusion_matrix": confusion,
        "per_status_recall": recalls,
        "macro_recall": sum(recalls.values()) / len(recalls),
        "prediction_errors": sum(
            item["predicted_status"] == "error" for item in results),
        "status_normalizations": sum(bool(
            item.get("result", {}).get("status_normalized", False))
            for item in results),
        "label_semantics": {
            "correct": "arrived", "halfway": "on_route", "wrong": "unknown",
            "unknown_is_not_a_supervised_off_route_claim": True,
        },
        "main_metric_uses_all_frozen_cases": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["prepare", "evaluate", "all"],
                        default="all")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--vlm-backend",
                        choices=["deepseek", "ollama", "heuristic"],
                        default="deepseek")
    parser.add_argument("--deepseek-env", type=Path,
                        default=ROOT / ".env.deepseek")
    parser.add_argument("--vlm-timeout", type=int, default=180)
    parser.add_argument(
        "--prompt-version",
        choices=["v9_three_way_edge_progress", "v10_bidirectional_three_way"],
        default=DEFAULT_PROMPT_VERSION)
    args = parser.parse_args(argv)
    if args.output_root.resolve() == args.source_root.resolve():
        raise RuntimeError("three-way evaluation must not modify the source root")

    if args.phase in {"prepare", "all"}:
        cases = prepare(
            args.output_root, args.source_root, args.vlm_backend,
            args.deepseek_env, args.prompt_version)
        print(f"prepared {len(cases)} frozen three-way cases", flush=True)
    else:
        _, cases = load_source_cases(args.source_root)
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
        write_json(args.output_root / "results.json", results)
        write_json(args.output_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
