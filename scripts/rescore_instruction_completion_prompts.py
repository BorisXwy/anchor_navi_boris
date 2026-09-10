#!/usr/bin/env python3
"""Re-run completion prompts on frozen node/edge/keyframe evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from instruction_decomposer import SubInstruction
from vlm_harness import NavigationVLMHarness, build_vlm_backend


ROOT = Path(__file__).resolve().parents[1]


def load_images(case_dir, records):
    return [np.asarray(Image.open(
        case_dir / record["image_path"]).convert("RGB"))
        for record in sorted(records, key=lambda item: item["view_index"])]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profile", default="dense_stop_motion_guard_v7")
    parser.add_argument("--cases", default=None)
    parser.add_argument("--prompt-version",
                        choices=sorted(NavigationVLMHarness.
                                       INSTRUCTION_COMPLETION_PROMPT_VERSIONS),
                        default="v2_transition_gates")
    parser.add_argument("--vlm-backend", choices=["deepseek", "ollama", "heuristic"],
                        default="deepseek")
    parser.add_argument("--deepseek-env", type=Path, default=ROOT / ".env.deepseek")
    parser.add_argument("--vlm-timeout", type=int, default=180)
    parser.add_argument(
        "--manual-labels", type=Path, default=None,
        help=("JSON label file keyed by case_index. When provided, score only "
              "manually audited completed/unknown transition labels instead "
              "of the coarse demonstration-endpoint proxy."))
    args = parser.parse_args(argv)
    manual_labels = None
    if args.manual_labels is not None:
        label_document = json.loads(args.manual_labels.read_text())
        label_items = (label_document.get("labels", [])
                       if isinstance(label_document, dict)
                       else label_document)
        manual_labels = {int(item["case_index"]): item for item in label_items}
    source_profile = args.source_root / args.profile
    case_dirs = sorted(source_profile.glob("case_*"))
    if args.cases:
        wanted = {int(value) for value in args.cases.split(",") if value}
        case_dirs = [path for path in case_dirs
                     if int(path.name.split("_")[-1]) in wanted]
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for source_case in case_dirs:
        source_result = json.loads((source_case / "result.json").read_text())
        if not source_result.get("instruction_completion_evaluated"):
            continue
        case_index = int(source_result["case_index"])
        if manual_labels is not None and case_index not in manual_labels:
            continue
        graph = json.loads((source_case / "navigation_graph" /
                            "navigation_graph.json").read_text())
        previous, current = graph["nodes"][0], graph["nodes"][1]
        edge = graph["edges"][0]
        if args.prompt_version == "v8_eight_view_spatial_relations":
            try:
                previous_records = previous["metadata"][
                    "instruction_completion_panorama"]["views"]
                current_records = current["metadata"][
                    "instruction_completion_panorama"]["views"]
            except KeyError as exc:
                raise RuntimeError(
                    f"case {source_case} predates completion-only eight-view "
                    "capture and cannot be rescored with v8") from exc
        else:
            previous_records = previous["six_views"]
            current_records = current["six_views"]
        previous_views = load_images(
            source_case / "navigation_graph", previous_records)
        current_views = load_images(
            source_case / "navigation_graph", current_records)
        keyframes = [np.asarray(Image.open(
            source_case / item["image_path"]).convert("RGB"))
            for item in edge["metadata"]["edge_keyframes"]]
        output_case = args.output_root / source_case.name
        output_case.mkdir(parents=True, exist_ok=True)
        backend = build_vlm_backend(
            args.vlm_backend, timeout=args.vlm_timeout,
            deepseek_env_file=args.deepseek_env)
        harness = NavigationVLMHarness(
            backend, output_case / "vlm_calls.json", retries=2,
            instruction_completion_prompt_version=args.prompt_version)
        original_case = Path(source_result["source_case_dir"])
        sub_instruction = json.loads(
            (original_case / "point_selection.json").read_text())[
                "sub_instruction"]
        raw = harness.judge_edge_instruction_completion(
            sub_instruction=SubInstruction.from_mapping(sub_instruction),
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
        prediction = bool(raw["status"] == "completed" and
                          float(raw["confidence"]) >= 0.5)
        if manual_labels is not None:
            manual = manual_labels[case_index]
            reference_status = str(manual["status"]).strip().lower()
            if reference_status not in {"completed", "unknown"}:
                raise ValueError(
                    f"manual case {case_index} has invalid status "
                    f"{reference_status!r}")
            reference = reference_status == "completed"
            reference_source = "manual_edge_semantic_audit"
            reference_rationale = str(manual.get("rationale", ""))
        else:
            reference = bool(
                source_result["instruction_completion_reference"]["completed"])
            reference_source = "demonstration_endpoint_proxy"
            reference_rationale = ""
        item = {
            "case_index": case_index,
            "source_case": str(source_case), "prompt_version": args.prompt_version,
            "prediction": prediction, "reference": reference,
            "reference_source": reference_source,
            "reference_rationale": reference_rationale,
            "correct": prediction == reference, "result": raw,
        }
        (output_case / "result.json").write_text(
            json.dumps(item, ensure_ascii=False, indent=2) + "\n")
        results.append(item)
        print(f"case={item['case_index']:03d} pred={int(prediction)} "
              f"ref={int(reference)} correct={int(item['correct'])}", flush=True)
    summary = {
        "prompt_version": args.prompt_version, "count": len(results),
        "reference_source": (
            "manual_edge_semantic_audit" if manual_labels is not None
            else "demonstration_endpoint_proxy"),
        "correct": sum(item["correct"] for item in results),
        "accuracy": (sum(item["correct"] for item in results) /
                     max(len(results), 1)),
        "predicted_completed": sum(item["prediction"] for item in results),
        "reference_completed": sum(item["reference"] for item in results),
    }
    (args.output_root / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
