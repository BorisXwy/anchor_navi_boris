#!/usr/bin/env python3
"""Fuse a frozen label-free v9 edge result with unordered endpoint roles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

import evaluate_instruction_progress_triplets as base
from instruction_decomposer import SubInstruction
from vlm_harness import NavigationVLMHarness, build_vlm_backend


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_VERSION = "v11_v9_plus_unordered_endpoint_roles"
DEFAULT_SOURCE_ROOT = base.DEFAULT_SOURCE_ROOT
DEFAULT_PRIMARY_ROOT = (
    ROOT / "outputs/instruction_progress_triplets_30_v9_frozen_20260831")
MIN_ROLE_CONFIDENCE = 0.55


def primary_case_dir(primary_root, case):
    return Path(primary_root) / (
        f"case_{int(case['case_index']):03d}_{case['category']}")


def prepare(output_root, source_root, primary_root, backend_name, deepseek_env):
    cases = base.prepare(
        output_root, source_root, backend_name, deepseek_env, PIPELINE_VERSION)
    output_root = Path(output_root)
    primary_root = Path(primary_root).resolve()
    protocol_path = output_root / "endpoint_role_protocol.json"
    protocol = {
        "pipeline_version": PIPELINE_VERSION,
        "pipeline_code_sha256": base.sha256_file(Path(__file__)),
        "primary_prediction_source": str(primary_root),
        "primary_prompt_version": "v9_three_way_edge_progress",
        "primary_results_used_by_fusion": "vlm_response.json only",
        "primary_scoring_fields_read": False,
        "endpoint_role_model_inputs_include_primary_prediction": False,
        "shuffle_rule": (
            "swap X/Y iff the frozen upstream artifact aggregate SHA256 is odd"),
        "minimum_role_confidence": MIN_ROLE_CONFIDENCE,
        "fusion_rule": {
            "reverse_order": "unknown",
            "observed_order_and_completion_target": (
                "arrived if primary has positive temporal progress"),
            "observed_order_and_intermediate_progress": (
                "on_route if primary has positive temporal progress"),
            "observed_order_and_source_context": "unknown",
            "neither_or_low_confidence": "retain primary temporal result",
            "primary_unknown": "never promoted by endpoint roles",
        },
        "case_level_rules": False,
        "cases": [],
    }
    forbidden_scoring_keys = {
        "category", "expected_status", "correct", "manual_label_rationale",
        "case_index", "episode_id", "reference_path_index",
    }
    for case in cases:
        target_dir = base.output_case_dir(output_root, case)
        case_manifest = json.loads((target_dir / "manifest.json").read_text())
        primary_path = primary_case_dir(primary_root, case) / "vlm_response.json"
        if not primary_path.is_file():
            raise RuntimeError(f"missing frozen label-free primary output {primary_path}")
        primary = json.loads(primary_path.read_text())
        leaked = sorted(forbidden_scoring_keys & set(primary))
        if leaked:
            raise RuntimeError(
                f"primary model output unexpectedly contains scoring fields {leaked}")
        aggregate = case_manifest["upstream_artifact_aggregate_sha256"]
        swap_xy = bool(int(aggregate, 16) & 1)
        record = {
            "case_index": int(case["case_index"]),
            "primary_vlm_response_path": str(primary_path),
            "primary_vlm_response_sha256": base.sha256_file(primary_path),
            "primary_contains_scoring_fields": False,
            "swap_xy": swap_xy,
        }
        protocol["cases"].append(record)
        base.write_json(target_dir / "endpoint_role_freeze.json", record)
    protocol["shuffle_counts"] = {
        "swapped": sum(item["swap_xy"] for item in protocol["cases"]),
        "not_swapped": sum(not item["swap_xy"] for item in protocol["cases"]),
    }
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text())
        if existing != protocol:
            raise RuntimeError(f"refusing to alter frozen protocol {protocol_path}")
    else:
        base.write_json(protocol_path, protocol)
    base.write_json(output_root / "pre_vlm_endpoint_role_audit.json", {
        "status": "accepted_before_endpoint_role_vlm_calls",
        "endpoint_role_vlm_calls_at_audit": 0,
        "case_count": len(cases),
        "same_role_prompt_schema_and_fusion_for_all_cases": True,
        "shuffle_is_label_independent": True,
        "endpoint_role_prompt_receives_chronology": False,
        "endpoint_role_prompt_receives_primary_prediction": False,
        "endpoint_role_prompt_receives_hidden_labels_or_scores": False,
        "primary_prediction_files_are_raw_label_free_vlm_responses": True,
        "case_level_rules": False,
    })
    return cases


def fuse(primary, roles, swap_xy):
    primary_status = str(primary.get("status", "error"))
    if primary_status not in {*base.STATUS_ORDER, "error"}:
        raise RuntimeError(f"unexpected primary status {primary_status!r}")
    if swap_xy:
        previous_role = roles["node_y_role"]
        current_role = roles["node_x_role"]
        observed_order_token = "y_to_x"
        reverse_order_token = "x_to_y"
    else:
        previous_role = roles["node_x_role"]
        current_role = roles["node_y_role"]
        observed_order_token = "x_to_y"
        reverse_order_token = "y_to_x"
    preferred = roles["preferred_order"]
    confident = float(roles["confidence"]) >= MIN_ROLE_CONFIDENCE
    final_status = primary_status
    override = None
    positive_primary = primary_status in {"arrived", "on_route"}
    if confident and preferred == reverse_order_token:
        final_status = "unknown"
        override = "unordered_roles_prefer_reverse_order"
    elif confident and preferred == observed_order_token and positive_primary:
        if current_role == "completion_target":
            final_status = "arrived"
            override = "current_endpoint_is_completion_target"
        elif current_role == "intermediate_progress":
            final_status = "on_route"
            override = "current_endpoint_is_intermediate_progress"
        elif current_role == "source_context":
            final_status = "unknown"
            override = "observed_edge_ends_at_source_context"
    return {
        "status": final_status,
        "primary_status": primary_status,
        "primary_confidence": float(primary.get("confidence", 0.0)),
        "previous_endpoint_role": previous_role,
        "current_endpoint_role": current_role,
        "preferred_order_mapped": (
            "observed" if preferred == observed_order_token else
            "reversed" if preferred == reverse_order_token else "neither"),
        "endpoint_role_confidence": float(roles["confidence"]),
        "fusion_override": override,
        "reason": roles.get("reason", ""),
        "visual_evidence": roles.get("visual_evidence", ""),
        "primary_result": primary,
        "endpoint_role_result": roles,
    }


def evaluate_case(case, source_root, primary_root, output_root,
                  backend_name, deepseek_env, timeout):
    source_dir = base.source_case_dir(source_root, case)
    target_dir = base.output_case_dir(output_root, case)
    result_path = target_dir / "instruction_edge_state.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    freeze = json.loads((target_dir / "endpoint_role_freeze.json").read_text())
    primary_path = primary_case_dir(primary_root, case) / "vlm_response.json"
    if base.sha256_file(primary_path) != freeze["primary_vlm_response_sha256"]:
        raise RuntimeError(f"frozen primary output changed: {primary_path}")
    primary = json.loads(primary_path.read_text())

    graph = json.loads(
        (source_dir / "navigation_graph/navigation_graph.json").read_text())
    previous, current = graph["nodes"][:2]
    graph_root = source_dir / "navigation_graph"
    previous_views = base.load_images(
        graph_root,
        previous["metadata"]["instruction_completion_panorama"]["views"])
    current_views = base.load_images(
        graph_root,
        current["metadata"]["instruction_completion_panorama"]["views"])
    swap_xy = bool(freeze["swap_xy"])
    if swap_xy:
        x_views, y_views = current_views, previous_views
        x_semantics = current["environment_semantics"]
        y_semantics = previous["environment_semantics"]
    else:
        x_views, y_views = previous_views, current_views
        x_semantics = previous["environment_semantics"]
        y_semantics = current["environment_semantics"]

    backend = build_vlm_backend(
        backend_name, timeout=timeout, deepseek_env_file=deepseek_env)
    harness = NavigationVLMHarness(
        backend, target_dir / "endpoint_role_vlm_calls.json", retries=2)
    try:
        roles = harness.classify_unordered_instruction_endpoint_roles(
            SubInstruction.from_mapping(case["sub_instruction"]),
            x_views, y_views, x_semantics, y_semantics)
        prediction = fuse(primary, roles, swap_xy)
    except RuntimeError as exc:
        prediction = {
            "status": "error", "confidence": 0.0,
            "reason": str(exc), "failure_counts_in_frozen_denominator": True,
        }

    # Hidden scoring begins after the endpoint-role call and uniform fusion.
    expected = base.HIDDEN_LABEL_BY_CATEGORY[case["category"]]
    record = {
        "case_index": int(case["case_index"]),
        "category": case["category"],
        "expected_status": expected,
        "predicted_status": prediction["status"],
        "correct": prediction["status"] == expected,
        "confidence": float(prediction.get(
            "endpoint_role_confidence", prediction.get("confidence", 0.0))),
        "result": prediction,
        "label_exposed_to_vlm": False,
        "future_demonstration_exposed_to_vlm": False,
    }
    base.write_json(result_path, record)
    base.write_json(target_dir / "vlm_response.json", prediction)
    prompt_record = (
        harness.calls[-1] if harness.calls else harness.attempts[0])
    (target_dir / "vlm_prompt.txt").write_text(prompt_record["prompt"] + "\n")
    Image.fromarray(harness._completion_contact_sheet(x_views)).save(
        target_dir / "vlm_node_x_contact_sheet.jpg", quality=95)
    Image.fromarray(harness._completion_contact_sheet(y_views)).save(
        target_dir / "vlm_node_y_contact_sheet.jpg", quality=95)
    Image.fromarray(np.concatenate([
        harness._completion_contact_sheet(x_views),
        harness._completion_contact_sheet(y_views)], axis=0)).save(
            target_dir / "vlm_contact_sheet.jpg", quality=95)
    base.write_json(target_dir / "model_input_audit.json", {
        "pipeline_version": PIPELINE_VERSION,
        "endpoint_role_prompt_receives_chronology": False,
        "endpoint_role_prompt_receives_primary_prediction": False,
        "label_exposed_to_vlm": False,
        "category_exposed_to_vlm": False,
        "reference_path_or_future_waypoint_exposed_to_vlm": False,
        "swap_xy": swap_xy,
        "swap_source": "frozen_upstream_artifact_hash_parity",
        "hidden_score_applied_after_model_call_and_fusion": True,
    })
    base.write_json(target_dir / "module_results.json", {
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


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["prepare", "evaluate", "all"],
                        default="all")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--primary-root", type=Path, default=DEFAULT_PRIMARY_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--vlm-backend",
                        choices=["deepseek", "ollama", "heuristic"],
                        default="deepseek")
    parser.add_argument("--deepseek-env", type=Path,
                        default=ROOT / ".env.deepseek")
    parser.add_argument("--vlm-timeout", type=int, default=180)
    args = parser.parse_args(argv)
    if args.phase in {"prepare", "all"}:
        cases = prepare(
            args.output_root, args.source_root, args.primary_root,
            args.vlm_backend, args.deepseek_env)
        print(f"prepared {len(cases)} frozen endpoint-role cases", flush=True)
    else:
        _, cases = base.load_source_cases(args.source_root)
        if not (args.output_root / "pre_vlm_endpoint_role_audit.json").exists():
            raise RuntimeError("prepare phase and endpoint-role audit must run first")
    if args.phase in {"evaluate", "all"}:
        results = []
        for case in cases:
            result = evaluate_case(
                case, args.source_root, args.primary_root, args.output_root,
                args.vlm_backend, args.deepseek_env, args.vlm_timeout)
            results.append(result)
            print(
                f"evaluated case={int(case['case_index']):02d} "
                f"category={case['category']} "
                f"expected={result['expected_status']} "
                f"predicted={result['predicted_status']} "
                f"correct={int(result['correct'])}", flush=True)
        summary = base.summarize(results)
        base.write_json(args.output_root / "results.json", results)
        base.write_json(args.output_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
