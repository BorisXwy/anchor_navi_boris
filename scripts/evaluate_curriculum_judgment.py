#!/usr/bin/env python3
"""Stage-C judge evaluation on the frozen, real 10-EP curriculum edges.

The input edges are the actual node/edge artifacts from the accepted Stage-00
selection+navigation run.  Labels are frozen in a manifest before any VLM
request and are never copied into the harness prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from instruction_decomposer import SubInstruction
from vlm_harness import NavigationVLMHarness, build_vlm_backend


EPISODES = (0, 3, 6, 9, 18, 27, 45, 126, 204, 219)
# Frozen before VLM calls from the semantic completion cue and the actual
# endpoint panorama.  This is test metadata, not model input.
EXPECTED = {
    0: "completed", 3: "unknown", 6: "completed", 9: "completed",
    18: "unknown", 27: "completed", 45: "unknown", 126: "completed",
    204: "completed", 219: "unknown",
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_views(root, records):
    return [
        np.asarray(Image.open(root / item["image_path"]).convert("RGB"))
        for item in sorted(records, key=lambda item: item["view_index"])
    ]


def score(records):
    labels = [str(item["expected_status"]) for item in records]
    preds = [str(item["predicted_status"]) for item in records]
    completed_tp = sum(p == e == "completed" for p, e in zip(preds, labels))
    completed_fp = sum(p == "completed" and e != "completed"
                       for p, e in zip(preds, labels))
    completed_fn = sum(p != "completed" and e == "completed"
                       for p, e in zip(preds, labels))
    unknown_tp = sum(p == e == "unknown" for p, e in zip(preds, labels))
    unknown_fp = sum(p == "unknown" and e != "unknown"
                     for p, e in zip(preds, labels))
    unknown_fn = sum(p != "unknown" and e == "unknown"
                     for p, e in zip(preds, labels))

    def f1(tp, fp, fn):
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return precision, recall, (2 * precision * recall /
                                   (precision + recall)
                                   if precision + recall else 0.0)

    cp, cr, cf = f1(completed_tp, completed_fp, completed_fn)
    up, ur, uf = f1(unknown_tp, unknown_fp, unknown_fn)
    return {
        "count": len(records),
        "correct": sum(p == e for p, e in zip(preds, labels)),
        "accuracy": sum(p == e for p, e in zip(preds, labels)) / len(records),
        "completed": {"tp": completed_tp, "fp": completed_fp,
                      "fn": completed_fn, "precision": cp, "recall": cr,
                      "f1": cf},
        "unknown": {"tp": unknown_tp, "fp": unknown_fp,
                     "fn": unknown_fn, "precision": up, "recall": ur,
                     "f1": uf},
        "macro_f1": (cf + uf) / 2.0,
    }


def evaluate(input_root, output_root, backend_name, deepseek_env, timeout,
             expected_statuses=None, sub_instruction_index=0,
             prompt_version="v19_vertical_guard_consensus"):
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    input_summary = input_root / "summary.json"
    if not input_summary.exists():
        # Stage-A selection runs intentionally do not produce a root summary;
        # the per-episode trajectories are the immutable upstream artifact.
        input_summary = next(
            (input_root / f"episode_{index:04d}" / "trajectory.json"
             for index in EPISODES
             if (input_root / f"episode_{index:04d}" / "trajectory.json").exists()),
            None)
        if input_summary is None:
            raise FileNotFoundError(
                f"no summary.json or episode trajectory under {input_root}")
    expected = dict(EXPECTED if expected_statuses is None else expected_statuses)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_scope": "stage_c_real_curriculum_edge_judgment",
        "benchmark": "R2R",
        "split": "val_unseen",
        "input_run": str(input_root.resolve()),
        "input_summary_sha256": sha256(input_summary),
        "episodes": list(EPISODES),
        "label_freeze": {
            "frozen_before_vlm_call": True,
            "label_exposed_to_vlm": False,
            "source": "semantic completion cue plus RGB endpoint review",
            "statuses": expected,
        },
        "upstream_modules_frozen": [
            "ground_point_selection", "point_navigation_executor",
            "point_target_arrival", "navigation_graph_node_creation",
        ],
        "prompt_version": prompt_version,
        "backend": backend_name,
        "sub_instruction_index": int(sub_instruction_index),
    }
    write_json(output_root / "manifest.json", manifest)
    records = []
    for episode_index in EPISODES:
        case_dir = input_root / f"episode_{episode_index:04d}"
        graph = json.loads((case_dir / "navigation_graph" /
                            "navigation_graph.json").read_text())
        trajectory = json.loads((case_dir / "trajectory.json").read_text())
        previous, current = graph["nodes"][:2]
        edge = graph["edges"][0]
        graph_root = case_dir / "navigation_graph"
        previous_meta = previous["metadata"]["instruction_completion_panorama"]
        current_meta = current["metadata"]["instruction_completion_panorama"]
        previous_views = load_views(graph_root, previous_meta["views"])
        current_views = load_views(graph_root, current_meta["views"])
        edge_records = edge["metadata"]["edge_keyframes"]
        keyframes = [np.asarray(Image.open(case_dir / item["image_path"])
                                .convert("RGB")) for item in edge_records]
        sub = SubInstruction.from_mapping(
            trajectory["sub_instructions"][int(sub_instruction_index)])
        log_path = output_root / f"episode_{episode_index:04d}" / "vlm_calls.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        backend = build_vlm_backend(
            backend_name, timeout=timeout, deepseek_env_file=deepseek_env)
        harness = NavigationVLMHarness(
            backend, log_path, retries=2,
            instruction_completion_prompt_version=prompt_version)
        try:
            prediction = harness.judge_edge_instruction_completion(
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
                edge_keyframes=keyframes,
                previous_base_yaw_rad=previous.get("base_yaw_rad"),
                current_base_yaw_rad=current.get("base_yaw_rad"),
                previous_visual_embedding=previous.get("visual_embedding"),
                current_visual_embedding=current.get("visual_embedding"),
                edge_keyframe_records=edge_records,
            )
            status = prediction["status"]
            error = None
        except Exception as exc:  # retain a failed VLM call in denominator
            prediction = {"status": "error", "confidence": 0.0,
                          "reason": str(exc), "visual_evidence": ""}
            status = "error"
            error = str(exc)
        record = {
            "episode_index": episode_index,
            "expected_status": expected[episode_index],
            "predicted_status": status,
            "correct": status == expected[episode_index],
            "error": error,
            "result": prediction,
            "label_exposed_to_vlm": False,
            "real_state_artifacts": {
                "trajectory": str((case_dir / "trajectory.json").resolve()),
                "graph": str((case_dir / "navigation_graph" /
                              "navigation_graph.json").resolve()),
                "previous_node_id": previous["node_id"],
                "current_node_id": current["node_id"],
                "edge_id": edge["edge_id"],
            },
        }
        write_json(output_root / f"episode_{episode_index:04d}" /
                   "instruction_completion.json", record)
        records.append(record)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_version": prompt_version,
        "records": records,
        "metrics": score(records),
    }
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--backend", default="deepseek")
    parser.add_argument("--deepseek-env", default=".env.deepseek")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument(
        "--expected-status-file",
        help="JSON object mapping episode indices to completed/unknown; labels are frozen before VLM calls")
    parser.add_argument("--sub-instruction-index", type=int, default=0)
    parser.add_argument("--prompt-version",
                        default="v19_vertical_guard_consensus")
    args = parser.parse_args()
    expected = None
    if args.expected_status_file:
        raw = json.loads(Path(args.expected_status_file).read_text())
        expected = {int(key): str(value) for key, value in raw.items()}
        if set(expected) != set(EPISODES) or not set(expected.values()) <= {
                "completed", "unknown"}:
            parser.error(
                "expected-status-file must contain exactly the ten fixed EPs "
                "with completed/unknown values")
    evaluate(args.input_root, args.output_root, args.backend,
             args.deepseek_env, args.timeout, expected,
             args.sub_instruction_index, args.prompt_version)


if __name__ == "__main__":
    main()
