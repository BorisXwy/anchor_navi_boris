#!/usr/bin/env python3
"""Run only the edge-level completion judge on frozen curriculum artifacts.

The point selector and executor are deliberately not called here.  Each case
must already contain the two persisted nodes, the real edge action history,
keyframes, and six-view panoramas produced by a preceding curriculum round.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from instruction_decomposer import SubInstruction
from vlm_harness import NavigationVLMHarness, build_vlm_backend


ROOT = Path(__file__).resolve().parents[1]


def load_records(base: Path, records):
    return [np.asarray(Image.open(base / item["image_path"]).convert("RGB"))
            for item in sorted(records, key=lambda item: item["view_index"])]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cases", default=None,
                        help="comma-separated episode indices (default: all)")
    parser.add_argument("--prompt-version", default="v21_stage_endpoint_recovery_structured",
                        choices=sorted(NavigationVLMHarness.
                                       INSTRUCTION_COMPLETION_PROMPT_VERSIONS))
    parser.add_argument("--vlm-backend", choices=["deepseek", "ollama", "heuristic"],
                        default="deepseek")
    parser.add_argument("--deepseek-env", type=Path,
                        default=ROOT / ".env.deepseek")
    parser.add_argument("--vlm-timeout", type=int, default=180)
    args = parser.parse_args(argv)

    wanted = None
    if args.cases:
        wanted = {int(value) for value in args.cases.split(",") if value}
    case_dirs = sorted(args.source_root.glob("episode_*"))
    if wanted is not None:
        case_dirs = [path for path in case_dirs
                     if int(path.name.split("_")[-1]) in wanted]
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    backend = build_vlm_backend(args.vlm_backend, timeout=args.vlm_timeout,
                                deepseek_env_file=args.deepseek_env)
    for source_case in case_dirs:
        episode_index = int(source_case.name.split("_")[-1])
        graph_dir = source_case / "navigation_graph"
        graph = json.loads((graph_dir / "navigation_graph.json").read_text())
        if len(graph.get("nodes", [])) < 2 or not graph.get("edges"):
            raise RuntimeError(f"{source_case} lacks a complete node/edge artifact")
        previous, current = graph["nodes"][0], graph["nodes"][1]
        edge = graph["edges"][0]
        target = json.loads((source_case / "trajectory.json").read_text())["targets"][0]
        sub = SubInstruction.from_mapping(target["sub_instruction"])
        completion_previous = previous.get("metadata", {}).get(
            "instruction_completion_panorama", {}).get("views")
        completion_current = current.get("metadata", {}).get(
            "instruction_completion_panorama", {}).get("views")
        if not completion_previous or not completion_current:
            raise RuntimeError(
                f"{source_case} lacks the required eight-view completion panorama")
        previous_views = load_records(graph_dir, completion_previous)
        current_views = load_records(graph_dir, completion_current)
        keyframes = [np.asarray(Image.open(
            source_case / item["image_path"]).convert("RGB"))
                     for item in edge["metadata"]["edge_keyframes"]]
        out_case = args.output_root / source_case.name
        out_case.mkdir(parents=True, exist_ok=True)
        harness = NavigationVLMHarness(
            backend, out_case / "vlm_calls.json", retries=2,
            instruction_completion_prompt_version=args.prompt_version)
        raw = harness.judge_edge_instruction_completion(
            sub_instruction=sub,
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
        item = {
            "episode_index": episode_index,
            "source_case": str(source_case),
            "prompt_version": args.prompt_version,
            "prediction": (raw.get("status") == "completed" and
                            float(raw.get("confidence", 0.0)) >= 0.5),
            "result": raw,
            "edge_id": edge["edge_id"],
            "action_count": len(edge["action_history"]),
        }
        (out_case / "result.json").write_text(
            json.dumps(item, ensure_ascii=False, indent=2) + "\n")
        results.append(item)
        print(f"episode={episode_index:04d} prediction={int(item['prediction'])}",
              flush=True)
    summary = {
        "prompt_version": args.prompt_version,
        "count": len(results),
        "predicted_completed": sum(item["prediction"] for item in results),
    }
    (args.output_root / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
