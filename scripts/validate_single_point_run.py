#!/usr/bin/env python3
"""Audit one real point-navigation output against every runtime module contract."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def validate(output_dir):
    output_dir = Path(output_dir)
    trajectory_path = output_dir / "trajectory.json"
    vlm_path = output_dir / "vlm_calls.json"
    trajectory = json.loads(trajectory_path.read_text())
    vlm_calls = json.loads(vlm_path.read_text())
    target = trajectory["targets"][0]
    checks = {}

    def check(module, condition, details):
        checks[module] = {
            "status": "pass" if bool(condition) else "fail",
            "details": details,
        }

    sub_instructions = trajectory.get("sub_instructions", [])
    check(
        "instruction_decomposer",
        bool(sub_instructions) and all(
            stage.get("semantic_spatial_target") and stage.get("form")
            for stage in sub_instructions),
        {"count": len(sub_instructions),
         "forms": [stage.get("form") for stage in sub_instructions]},
    )

    tasks = [call.get("task") for call in vlm_calls]
    finish_reasons = [call.get("finish_reason") for call in vlm_calls]
    check(
        "deepseek_vlm",
        trajectory.get("vlm", {}).get("backend") == "deepseek" and
        {"decompose_instruction", "select_ground_target"}.issubset(tasks) and
        all(reason == "stop" for reason in finish_reasons),
        {"model": trajectory.get("vlm", {}).get("model"), "tasks": tasks,
         "finish_reasons": finish_reasons,
         "total_tokens": sum(
             int(call.get("usage", {}).get("total_tokens", 0))
             for call in vlm_calls)},
    )

    candidates = target.get("candidate_views", [])
    ground_detections = sum(
        len(candidate.get("ground_detection_records", [])) for candidate in candidates)
    semantic_detections = sum(
        len(candidate.get("detection_records", [])) for candidate in candidates)
    check(
        "dino_sam_floor_segmentation",
        ground_detections > 0 and any(
            candidate.get("ground_fraction", 0) > 0 for candidate in candidates),
        {"candidate_views": len(candidates),
         "ground_detection_instances": ground_detections},
    )
    check(
        "semantic_detection_and_floor_constraint",
        semantic_detections > 0 and all(
            candidate.get("strategy_application", {}).get("mode")
            for candidate in candidates),
        {"semantic_detection_instances": semantic_detections,
         "constraint_modes": sorted({
             candidate.get("strategy_application", {}).get("mode")
             for candidate in candidates})},
    )

    selection = target.get("selection", {})
    check(
        "external_point_selector",
        selection.get("requested_on_ground") is True and
        len(selection.get("snapped_xy", [])) == 2,
        {"snapped_xy": selection.get("snapped_xy"),
         "anchor_index": selection.get("anchor_index"),
         "reason": selection.get("reason")},
    )

    crop = target.get("initial_crop_geometry", {})
    quad = np.asarray(crop.get("quad_xy", []), np.float32)
    horizontal = (
        quad.shape == (4, 2) and np.isclose(quad[0, 1], quad[1, 1]) and
        np.isclose(quad[2, 1], quad[3, 1]))
    check(
        "horizontal_symmetric_goal_crop",
        horizontal and crop.get("axis_length_px", 0) >= crop.get("minimum_height_px", 1) and
        crop.get("output_size_wh") in ([85, 64], [96, 96]),
        {"top_midpoint_xy": crop.get("top_midpoint_xy"),
         "bottom_midpoint_xy": crop.get("bottom_midpoint_xy"),
         "axis_length_px": crop.get("axis_length_px"),
         "minimum_height_px": crop.get("minimum_height_px"),
         "output_size_wh": crop.get("output_size_wh")},
    )

    check(
        "dual_tracking_clusters",
        target.get("navigation_cluster_size") == 9 and
        target.get("stop_cluster_size") == 9 and
        target.get("all_stop_cluster_points_on_ground") is True,
        {"navigation_cluster_size": target.get("navigation_cluster_size"),
         "stop_cluster_size": target.get("stop_cluster_size"),
         "navigation_ground_fallback": target.get(
             "dual_cluster_geometry", {}).get("navigation_region_ground_fallback"),
         "stop_points_on_ground": target.get("all_stop_cluster_points_on_ground")},
    )

    steps = target.get("steps", [])
    tracker_contract = bool(steps) and all(
        len(step.get("navigation_tracks_xy", [])) == 9 and
        len(step.get("stop_tracks_xy", [])) == 9 for step in steps)
    check(
        "causal_tapir_tracker",
        tracker_contract,
        {"tracked_steps": len(steps),
         "terminal_navigation_visible_fraction": target.get(
             "terminal_navigation_visible_fraction"),
         "terminal_stop_visible_fraction": target.get(
             "terminal_stop_visible_fraction")},
    )

    policy_contract = bool(steps) and all(
        step.get("policy_distance") is not None and
        step.get("policy_first_trajectory") for step in steps)
    check(
        "image_goal_navigation_policy",
        policy_contract and trajectory.get("policy") in {"gnm", "vint", "nomad"},
        {"policy": trajectory.get("policy"),
         "inference_calls": len(steps),
         "mean_inference_sec": (
             float(np.mean([step["policy_inference_sec"] for step in steps]))
             if steps else None)},
    )

    check(
        "point_navigation_executor",
        target.get("arrived") is True and
        target.get("arrival_signal") == "point_navigation_arrived" and
        target.get("end_reason") == "all_stop_cluster_points_disappeared",
        {"arrived": target.get("arrived"),
         "arrival_signal": target.get("arrival_signal"),
         "end_reason": target.get("end_reason"),
         "action_count": len(target.get("action_history", []))},
    )

    graph = trajectory.get("navigation_graph_memory", {})
    check(
        "navigation_graph_memory",
        graph.get("node_count", 0) >= 2 and graph.get("edge_count", 0) >= 1,
        {"node_count": graph.get("node_count"),
         "edge_count": graph.get("edge_count"),
         "latest_node_id": graph.get("latest_node_id")},
    )
    match = target.get("sub_instruction_match", {})
    check(
        "sub_instruction_node_matcher",
        isinstance(match.get("belongs"), bool) and match.get("score") is not None,
        {"belongs": match.get("belongs"), "score": match.get("score"),
         "threshold": match.get("threshold")},
    )

    video_record = trajectory.get("video", {})
    video_value = (video_record.get("path", "exploration.mp4")
                   if isinstance(video_record, dict) else video_record)
    video_path = Path(video_value)
    if not video_path.is_absolute() and not video_path.exists():
        video_path = output_dir / video_path
    capture = cv2.VideoCapture(str(video_path))
    video_ok = capture.isOpened()
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if video_ok else 0
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if video_ok else 0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if video_ok else 0
    capture.release()
    check(
        "habitat_three_panel_video",
        video_ok and (width, height) == (960, 480) and frame_count > 0,
        {"path": str(video_path), "width": width, "height": height,
         "frame_count": frame_count},
    )

    r2r = trajectory.get("r2r_metrics", {})
    check(
        "r2r_metrics",
        all(key in r2r for key in (
            "initial_geodesic_distance_m", "final_geodesic_distance_m",
            "path_length_m", "success", "spl")),
        r2r,
    )

    passed = sum(item["status"] == "pass" for item in checks.values())
    report = {
        "output_dir": str(output_dir.resolve()),
        "passed": passed,
        "failed": len(checks) - passed,
        "checks": checks,
        "not_exercised_in_forward_single_point": [
            "node_backtracking", "instruction_sequence_recovery",
            "pure_random_exploration"],
        "note": (
            "The three branch-only modules are covered by deterministic unit tests; "
            "they cannot all execute during one forward semantic point hop."),
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = validate(args.output_dir)
    output = args.output or args.output_dir / "module_validation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
