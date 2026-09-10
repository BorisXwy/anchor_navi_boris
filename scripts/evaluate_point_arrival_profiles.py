#!/usr/bin/env python3
"""Replay fixed real R2R point selections under arrival-cluster profiles."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from habitat_point_navigation import (
    CausalTapirCluster, ExplorationVideoComposer, VideoFrameSink,
    load_r2r_episode, make_sim, resolve_mp3d_scene, set_pose,
)
from image_goal_policy import build_policy, predict
from instruction_completion_judge import (
    NodeTransitionInstructionCompletionJudge,
)
from instruction_decomposer import SubInstruction
from navigation_graph_memory import (
    DinoSamEnvironmentSemanticExtractor, NavigationGraphMemory,
)
from point_navigation_executor import (
    TRACKING_CLUSTER_PROFILES, PointNavigationExecutor,
    PointNavigationRequest, execute_point_navigation,
)
from point_selectors import observe_six_rgbd
from semantic_detector import GroundedSamDetector
from vlm_harness import NavigationVLMHarness, build_vlm_backend


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "outputs/r2r_all_modules_100ep_v3"
DEFAULT_R2R_DATA = (
    ROOT.parent / "3d_wm_vln/StreamVLN/data/datasets/r2r/val_unseen/"
    "val_unseen.json.gz")
DEFAULT_MP3D_ROOT = ROOT.parent / "3d_wm_vln/StreamVLN/data/scene_datasets/mp3d"
DEFAULT_CASES = "0,1,3,4,6,7,10,12,17,20,28,31,35,37,38,41,47,54,69,73"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def geodesic(sim, start, end):
    shortest = __import__("habitat_sim").ShortestPath()
    shortest.requested_start = np.asarray(start, np.float32)
    shortest.requested_end = np.asarray(end, np.float32)
    if not sim.pathfinder.find_path(shortest):
        return None
    return float(shortest.geodesic_distance)


def selected_view_index(target):
    selected_yaw = float(target["selected_yaw_rad"])
    candidates = target.get("candidate_views") or []
    if not candidates:
        raise ValueError("trajectory target has no candidate views")
    return int(min(candidates, key=lambda item: abs(math.atan2(
        math.sin(float(item["yaw"]) - selected_yaw),
        math.cos(float(item["yaw"]) - selected_yaw))))["view_index"])


def load_cases(source_root, case_indices):
    cases = []
    for index in case_indices:
        case_dir = source_root / f"case_{index:03d}"
        trajectory = json.loads((case_dir / "trajectory.json").read_text())
        manifest = json.loads((case_dir / "manifest.json").read_text())
        target = trajectory["targets"][0]
        cases.append({
            "case_index": index,
            "source_case_dir": case_dir,
            "episode_index": int(manifest["episode_index"]),
            "episode_id": str(trajectory["episode_id"]),
            "reference_path_index": int(
                trajectory["reference_state"]["path_index"]),
            "start_position_xyz": trajectory["reference_state"]["position_xyz"],
            "target": target,
            "all_decomposed_sub_instructions": (
                trajectory["all_decomposed_sub_instructions"]),
            "selected_view_index": selected_view_index(target),
        })
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "outputs/point_arrival_profile_iterations")
    parser.add_argument("--r2r-data", type=Path, default=DEFAULT_R2R_DATA)
    parser.add_argument("--mp3d-root", type=Path, default=DEFAULT_MP3D_ROOT)
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--profiles", default="legacy_3x3,dense_bottom_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--judge-completion", action="store_true")
    parser.add_argument("--completion-threshold-m", type=float, default=1.5)
    parser.add_argument("--vlm-backend", choices=["deepseek", "ollama", "heuristic"],
                        default="deepseek")
    parser.add_argument("--deepseek-env", type=Path, default=ROOT / ".env.deepseek")
    parser.add_argument("--vlm-timeout", type=int, default=180)
    parser.add_argument(
        "--completion-prompt-version",
        choices=sorted(
            NavigationVLMHarness.INSTRUCTION_COMPLETION_PROMPT_VERSIONS),
        default="v1_edge_evidence")
    args = parser.parse_args(argv)

    case_indices = [int(value) for value in args.cases.split(",") if value]
    profiles = [value for value in args.profiles.split(",") if value]
    unknown = set(profiles) - set(TRACKING_CLUSTER_PROFILES)
    if unknown:
        parser.error(f"unknown profiles: {sorted(unknown)}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.source_root, case_indices)

    torch.manual_seed(17)
    tracker = CausalTapirCluster(args.device)
    arrival_tracker = CausalTapirCluster(
        args.device, shared_model=tracker.model)
    policy, policy_config = build_policy("gnm", torch.device(args.device))
    dino_sam = (GroundedSamDetector(args.device)
                    if args.judge_completion else None)
    results = []
    for profile in profiles:
        for case in cases:
            target = case["target"]
            episode = load_r2r_episode(
                args.r2r_data, case["episode_index"], case["episode_id"])
            scene = resolve_mp3d_scene(episode["scene_id"], args.mp3d_root)
            sim = make_sim(scene, 320, 240)
            start = np.asarray(case["start_position_xyz"], np.float32)
            yaw = float(target["selected_yaw_rad"])
            set_pose(sim, start, yaw)
            rgb = sim.get_sensor_observations()["rgb"][..., :3]
            view_index = case["selected_view_index"]
            mask_path = (case["source_case_dir"] / "ground_masks" /
                         f"hop_000_view_{view_index}.png")
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
            output_dir = args.output_root / profile / f"case_{case['case_index']:03d}"
            output_dir.mkdir(parents=True, exist_ok=True)
            video_sink = None
            if not args.no_video:
                composer = ExplorationVideoComposer(
                    sim, start, obs_width=320, obs_height=240)
                video_sink = VideoFrameSink(
                    output_dir / "point_arrival.mp4", composer.frame_size, fps=5)
            else:
                composer = None
            position_history = [start.copy()]
            graph_memory = None
            origin_rgbs = origin_depths = None
            if args.judge_completion:
                origin_rgbs, origin_depths = observe_six_rgbd(sim)
                graph_memory = NavigationGraphMemory(
                    output_dir / "navigation_graph",
                    DinoSamEnvironmentSemanticExtractor(dino_sam))
                graph_memory.add_origin_node(
                    start, yaw, 0, origin_rgbs, origin_depths,
                    metadata={"real_reference_state": True})
            executor = PointNavigationExecutor(
                sim=sim, tracker=tracker, policy=policy,
                arrival_tracker=(None if profile == "legacy_3x3" else
                                 arrival_tracker),
                policy_config=policy_config, policy_name="gnm",
                predict_fn=predict, device=args.device,
                output_dir=output_dir, video_sink=video_sink,
                video_composer=composer, max_steps=args.max_steps,
                forward_step=0.22, turn_step_deg=15.0, seed=17,
                tracking_cluster_profile=profile, edge_keyframe_count=5)
            selected_target = np.asarray(
                target["selected_navmesh_target_xyz"], np.float32)
            initial_distance = geodesic(sim, start, selected_target)
            write_json(output_dir / "manifest.json", {
                "test_scope": "module",
                "target_modules": [
                    "point_target_arrival",
                    *( ["arrival_node_edge_invariant",
                         "instruction_completion"]
                       if args.judge_completion else []),
                ],
                "required_upstream_modules": [
                    "frozen_real_r2r_point_selection",
                    "grounded_sam_floor_mask",
                ],
                "source_case_dir": str(case["source_case_dir"]),
                "episode_index": case["episode_index"],
                "episode_id": case["episode_id"],
                "reference_path_index": case["reference_path_index"],
                "reference_position_xyz": start.tolist(),
                "selected_point_xy": target["selected_point_xy"],
                "selected_navmesh_target_xyz": selected_target.tolist(),
                "tracking_cluster_profile": profile,
                "completion_prompt_version": (
                    args.completion_prompt_version
                    if args.judge_completion else None),
                "future_reference_information_exposed_to_models": False,
            })
            result = execute_point_navigation(executor, PointNavigationRequest(
                rgb=rgb,
                selected_point_xy=np.asarray(target["selected_point_xy"], np.float32),
                selectable_mask=mask, yaw=yaw,
                position_history=position_history,
                instruction=target["sub_instruction"]["navigation_instruction"],
                semantic_target=target["sub_instruction"]["semantic_spatial_target"],
                target_index=0, stage_count=1, global_step=0,
                selected_point_depth_m=target["selected_point_depth_m"],
                selected_point_reachable=initial_distance is not None,
                selected_point_initial_geodesic_m=initial_distance,
                reference_path=[np.asarray(point, np.float32)
                                for point in episode.get("reference_path", [])],
                reference_path_index=int(case["reference_path_index"])))
            stop = np.asarray(sim.get_agent(0).get_state().position, np.float32)
            final_distance = geodesic(sim, stop, selected_target)
            reference_reached = bool(
                final_distance is not None and final_distance <= 0.75)
            outcome = (
                "true_arrival" if result.arrived and reference_reached else
                "premature_arrival" if result.arrived else
                "missed_arrival" if reference_reached else
                "not_reached")
            record = {
                "case_index": case["case_index"], "profile": profile,
                "episode_index": case["episode_index"],
                "episode_id": case["episode_id"],
                "reference_path_index": case["reference_path_index"],
                "source_case_dir": str(case["source_case_dir"]),
                "selected_view_index": view_index,
                "selected_point_xy": target["selected_point_xy"],
                "selected_navmesh_target_xyz": selected_target.tolist(),
                "initial_geodesic_m": initial_distance,
                "final_geodesic_m": final_distance,
                "executor_arrived": result.arrived,
                "reference_reached_0_75m": reference_reached,
                "outcome": outcome, "end_reason": result.end_reason,
                "action_history": result.action_history,
                "executor_record": result.record,
            }
            if args.judge_completion:
                current_rgbs, current_depths = observe_six_rgbd(sim)
                sub_instruction = SubInstruction.from_mapping(
                    target["sub_instruction"])
                node, edge = graph_memory.add_navigation_stop_node(
                    position_xyz=stop, base_yaw_rad=result.final_yaw,
                    global_step=result.next_global_step,
                    six_views=current_rgbs, six_depths=current_depths,
                    sub_instruction=sub_instruction,
                    action_history=result.action_history,
                    arrival_signal=result.signal,
                    edge_metadata={
                        "edge_keyframes": result.record["edge_keyframes"]})
                record["node_construction"] = {
                    "node_id": node.node_id, "edge_id": edge.edge_id,
                    "node_created": True,
                    "arrival_requires_node_satisfied": bool(
                        not result.arrived or node is not None),
                }
                all_stages = case["all_decomposed_sub_instructions"]
                stage_index = next(
                    index for index, item in enumerate(all_stages)
                    if int(item["sub_instruction_id"]) ==
                    int(sub_instruction.sub_instruction_id))
                reference_path = episode["reference_path"]
                endpoint_index = round(
                    (stage_index + 1) * (len(reference_path) - 1) /
                    len(all_stages))
                endpoint = np.asarray(reference_path[endpoint_index], np.float32)
                completion_distance = float(np.linalg.norm(
                    (stop - endpoint)[[0, 2]]))
                completion_truth = bool(
                    completion_distance <= args.completion_threshold_m)
                record["instruction_completion_reference"] = {
                    "completed": completion_truth,
                    "source": (
                        "auxiliary_uniform_demonstration_endpoint_proxy_not_"
                        "semantic_ground_truth"),
                    "endpoint_index": endpoint_index,
                    "endpoint_position_xyz": endpoint.tolist(),
                    "planar_distance_m": completion_distance,
                    "threshold_m": args.completion_threshold_m,
                    "future_reference_exposed_to_model": False,
                }
                if result.arrived:
                    backend = build_vlm_backend(
                        args.vlm_backend, timeout=args.vlm_timeout,
                        deepseek_env_file=args.deepseek_env)
                    harness = NavigationVLMHarness(
                        backend, output_dir / "vlm_calls.json", retries=2,
                        instruction_completion_prompt_version=(
                            args.completion_prompt_version))
                    completion = NodeTransitionInstructionCompletionJudge(
                        harness, graph_memory, [sub_instruction]).judge(
                            node, sub_instruction.sub_instruction_id,
                            current_rgbs, result.edge_keyframes).to_dict()
                    graph_memory.set_node_metadata(
                        node.node_id, {"instruction_completion": completion})
                    completion_endpoint_proxy_match = bool(
                        completion["instruction_completed"] == completion_truth)
                else:
                    completion = {
                        "status": "unknown", "instruction_completed": False,
                        "confidence": 1.0,
                        "reason": "point target was not declared arrived",
                    }
                    completion_endpoint_proxy_match = None
                record["instruction_completion"] = completion
                write_json(
                    output_dir / "instruction_completion.json", completion)
                record["instruction_completion_evaluated"] = bool(result.arrived)
                record["instruction_completion_endpoint_proxy_match"] = (
                    completion_endpoint_proxy_match)
                if video_sink is not None and result.arrived:
                    completion_frame = cv2.cvtColor(
                        current_rgbs[0], cv2.COLOR_RGB2BGR)
                    status_text = str(completion["status"]).upper()
                    status_color = ((0, 255, 0) if status_text == "COMPLETED"
                                    else (0, 215, 255))
                    cv2.putText(
                        completion_frame,
                        f"INSTRUCTION: {status_text}", (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_color, 2)
                    cv2.putText(
                        completion_frame,
                        f"EDGE {edge.source_node_id} -> {node.node_id}",
                        (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1)
                    cv2.putText(
                        completion_frame,
                        str(completion.get("reason", ""))[:68],
                        (8, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                        (255, 255, 255), 1)
                    composed = composer.compose(
                        completion_frame, position_history, stop,
                        result.final_yaw,
                        sub_instruction.navigation_instruction, 0,
                        f"instruction_completion_{completion['status']}")
                    for _ in range(5):
                        video_sink.append(composed.copy())
            write_json(output_dir / "result.json", record)
            write_json(output_dir / "module_results.json", record)
            results.append(record)
            if video_sink is not None:
                video_sink.close()
            sim.close()
            print(
                f"profile={profile} case={case['case_index']:03d} "
                f"outcome={outcome} final={final_distance} "
                f"steps={len(result.action_history)}", flush=True)

    summary = {
        "cases": case_indices,
        "floor_segmenter": "Grounded-SAM (Grounding-DINO Swin-T + SAM ViT-H)",
        "path_projection_visualization": {
            "enabled": True,
            "reference": "R2R reference_path projected in executor frames",
            "ground_truth_color_rgb": [0, 230, 255],
            "selected_point_color_rgb": [255, 30, 30],
            "path_is_not_model_input": True,
        },
        "profiles": {},
    }
    for profile in profiles:
        values = [item for item in results if item["profile"] == profile]
        counts = {}
        for item in values:
            counts[item["outcome"]] = counts.get(item["outcome"], 0) + 1
        true_positive = counts.get("true_arrival", 0)
        false_positive = counts.get("premature_arrival", 0)
        false_negative = counts.get("missed_arrival", 0)
        true_negative = counts.get("not_reached", 0)
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        summary["profiles"][profile] = {
            "count": len(values), "outcomes": counts,
            "strict_point_arrival_successes": true_positive,
            "strict_point_arrival_rate": true_positive / max(len(values), 1),
            "arrival_decision_confusion": {
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_negative": true_negative,
            },
            "arrival_decision_accuracy": (
                true_positive + true_negative) / max(len(values), 1),
            "arrival_decision_precision": precision,
            "arrival_decision_recall": recall,
            "arrival_decision_f1": (
                2 * precision * recall / max(precision + recall, 1e-12)),
            "initially_unreachable_rejections": sum(
                item["end_reason"] == "selected_point_unreachable"
                for item in values),
        }
        judged = [item for item in values
                  if item.get("instruction_completion_evaluated")]
        if judged:
            summary["profiles"][profile][
                "instruction_completion_auxiliary_endpoint_proxy"] = {
                "count": len(judged),
                "matches": sum(
                    item["instruction_completion_endpoint_proxy_match"]
                    for item in judged),
                "match_rate_not_semantic_accuracy": sum(
                    item["instruction_completion_endpoint_proxy_match"]
                    for item in judged) /
                    len(judged),
                "predicted_completed": sum(
                    item["instruction_completion"]["instruction_completed"]
                    for item in judged),
                "reference_completed": sum(
                    item["instruction_completion_reference"]["completed"]
                    for item in judged),
            }
    write_json(args.output_root / "summary.json", summary)
    write_json(args.output_root / "results.json", results)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
