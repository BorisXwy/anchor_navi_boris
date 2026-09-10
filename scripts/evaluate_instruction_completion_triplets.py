#!/usr/bin/env python3
"""Frozen 10x3 real-R2R test for binary edge instruction completion.

Each base instruction contributes three actual Habitat edge replays over its
demonstration reference states: start->end (completed), start->first internal
state (halfway/unknown), and end->start (wrong-direction/unknown).  Preparation
and VLM evaluation are separate phases so labels and visual evidence are frozen
before any completion-model call.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from habitat_point_navigation import (
    ExplorationVideoComposer, VideoFrameSink, make_sim, resolve_mp3d_scene,
    set_pose, yaw_from_coeffs,
)
from instruction_decomposer import SubInstruction
from instruction_taxonomy import FORM_DEFINITIONS
from navigation_graph_memory import (
    DinoSamEnvironmentSemanticExtractor, NavigationGraphMemory,
)
from point_selectors import observe_eight_rgb, observe_six_rgbd, wrap_angle
from semantic_detector import DinoSamDetector
from vlm_harness import NavigationVLMHarness, build_vlm_backend


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_R2R_DATA = Path(
    "/sharedata/datasets/R2R/R2R_VLNCE_v1-3/val_unseen/val_unseen.json.gz")
DEFAULT_MP3D_ROOT = ROOT.parent / "3d_wm_vln/StreamVLN/data/scene_datasets/mp3d"

# Manually selected before model calls for clear single-goal spatial semantics,
# form diversity, and >=5 real reference states.  The list is part of the
# frozen protocol; evaluation never replaces a case after seeing VLM output.
BASE_CASES = (
    {
        "episode_index": 34, "form": "PASS_LANDMARK",
        "landmark": "potted plant and master bedroom sunburst mirror",
        "target": "inside the master bedroom beyond the hallway potted plant",
        "arrival": "the potted plant is behind and the master bedroom is reached",
    }, {
        "episode_index": 133, "form": "PASS_LANDMARK",
        "landmark": "two wicker baskets and orange couch",
        "target": "the sitting area beyond both wicker baskets",
        "arrival": "both baskets are behind and the orange couch is ahead",
    }, {
        "episode_index": 321, "form": "CROSS_SPACE",
        "landmark": "red-carpeted room and next doors",
        "target": "the far side of the red-carpeted room at the next doors",
        "arrival": "the room has been crossed and the agent waits at the next doors",
    }, {
        "episode_index": 632, "form": "FOLLOW_PATH_BOUNDARY",
        "landmark": "sauna path, exit door, and pool room",
        "target": "inside the pool room after following the sauna path and exiting",
        "arrival": "the sauna exit is behind and the agent is in the pool room",
    }, {
        "episode_index": 226, "form": "TRAVERSE_PORTAL_REGION",
        "landmark": "doorway, hallway, and bedroom with gray throw",
        "target": "the hall end after the left turn, looking into the gray-throw bedroom",
        "arrival": "the doorway and hall are traversed, the left turn is made, and the bedroom is ahead",
    }, {
        "episode_index": 166, "form": "TRAVERSE_PORTAL_REGION",
        "landmark": "dining table, two doorways, and large lobby",
        "target": "inside the large lobby after passing the table and both doorways",
        "arrival": "the dining table and second doorway are behind and the agent waits in the large lobby",
    }, {
        "episode_index": 780, "form": "BETWEEN_OBJECTS",
        "landmark": "bar, table, and chair",
        "target": "the chair beyond the gap between the bar and table",
        "arrival": "the bar-table gap has been cleared and the chair is reached",
    }, {
        "episode_index": 570, "form": "ENTER_REGION",
        "landmark": "kitchen exit and second bedroom",
        "target": "inside the second bedroom after exiting the kitchen and veering right",
        "arrival": "the kitchen is behind and the agent has entered the second bedroom on the left",
    }, {
        "episode_index": 1207, "form": "TURN_RIGHT",
        "landmark": "hall end and butler's pantry",
        "target": "inside the butler's pantry after the hall-end right turn",
        "arrival": "the hall-end right turn is complete and the agent is in the pantry",
    }, {
        "episode_index": 1631, "form": "ADVANCE_STRAIGHT",
        "landmark": "stairs",
        "target": "walkable floor near the stairs after advancing straight",
        "arrival": "the agent has advanced to the stairs without taking a side branch",
    },
)
CATEGORY_ORDER = ("correct", "wrong", "halfway")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def heading_between(start, end):
    delta = np.asarray(end, np.float32) - np.asarray(start, np.float32)
    return wrap_angle(math.atan2(-float(delta[0]), -float(delta[2])))


def reference_yaw(episode, path, index):
    if index == 0:
        return yaw_from_coeffs(episode["start_rotation"]), "episode_start_rotation"
    return heading_between(path[index - 1], path[index]), (
        "derived_from_reference_path_previous_to_current")


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
        "semantic_spatial_target": base.get(
            "target", definition["spatial_target"]),
        "spatial_relation": base["form"].lower(),
        "completion_cue": base.get("arrival", definition["arrival"]),
        "visual_arrival_evidence": base.get(
            "arrival", definition["arrival"]),
        "forbidden_target": definition["forbidden"],
        "source_clause": text,
        "metadata": {"construction": "frozen_full_instruction_single_stage"},
    }


def category_indices(path_length, category):
    end = path_length - 1
    if category == "correct":
        return 0, end, "completed"
    if category == "wrong":
        return end, 0, "unknown"
    if category == "halfway":
        return 0, 1, "unknown"
    raise KeyError(category)


def load_dataset(path):
    with gzip.open(path, "rt") as handle:
        return json.load(handle)["episodes"]


def build_frozen_cases(episodes):
    cases = []
    case_index = 0
    for base_index, base in enumerate(BASE_CASES):
        episode = episodes[base["episode_index"]]
        path = episode.get("reference_path", [])
        if len(path) < 5:
            raise RuntimeError(
                f"frozen episode {base['episode_index']} has fewer than 5 states")
        instruction = sub_instruction_mapping(base, episode)
        for category in CATEGORY_ORDER:
            previous_index, current_index, label = category_indices(
                len(path), category)
            previous_yaw, previous_yaw_source = reference_yaw(
                episode, path, previous_index)
            cases.append({
                "case_index": case_index,
                "base_case_index": base_index,
                "category": category,
                "manual_edge_event_label": label,
                "manual_label_rationale": {
                    "correct": (
                        "forward replay covers the complete official R2R "
                        "reference path and ends at its demonstrated goal"),
                    "wrong": (
                        "the same official path is replayed in the exact reverse "
                        "direction, contradicting the active instruction"),
                    "halfway": (
                        "forward replay stops at the first internal reference "
                        "state, before the demonstrated goal"),
                }[category],
                "episode_index": int(base["episode_index"]),
                "episode_id": episode.get("episode_id"),
                "trajectory_id": episode.get("trajectory_id"),
                "scene_id": episode.get("scene_id"),
                "reference_path_length": len(path),
                "previous_path_index": previous_index,
                "current_path_index": current_index,
                "previous_position_xyz": path[previous_index],
                "current_position_xyz": path[current_index],
                "previous_yaw_rad": previous_yaw,
                "previous_yaw_source": previous_yaw_source,
                "full_instruction": episode["instruction"]["instruction_text"],
                "sub_instruction": instruction,
                "alignment": {
                    "method": "frozen_full_instruction_single_stage",
                    "stage_id": 0,
                    "stage_interval": [0, len(path) - 1],
                    "alignment_uncertain": False,
                },
            })
            case_index += 1
    return cases


def initialize_manifest(output_root, dataset_path, cases, device):
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_scope": "single_point_module_test_instruction_completion",
        "target_modules": ["binary_edge_instruction_completion"],
        "not_run_modules": [
            "instruction_decomposition_vlm", "ground_point_selection",
            "point_navigation_executor", "point_target_arrival",
            "sequence_recovery_policy", "physical_node_backtracking",
        ],
        "benchmark": "R2R", "split": "val_unseen",
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": sha256(dataset_path),
        "selection_frozen_before_any_model_call": True,
        "future_reference_information_exposed_to_vlm": False,
        "sample_design": {
            "base_instruction_count": 10,
            "category_counts": {name: 10 for name in CATEGORY_ORDER},
            "paired_design": (
                "each base instruction contributes correct, completely wrong, "
                "and halfway edges from official reference states"),
            "correct_edge": "reference_path[0] -> reference_path[-1]",
            "wrong_edge": "reference_path[-1] -> reference_path[0]",
            "halfway_edge": "reference_path[0] -> reference_path[1]",
            "case_order": "base instruction order, then correct/wrong/halfway",
        },
        "edge_construction": {
            "controller": "deterministic Habitat navmesh reference-edge replay",
            "route": (
                "chronologically visits every official reference_path state "
                "between the frozen endpoints"),
            "linear_step_m": 0.22, "turn_step_deg": 15.0,
            "keyframe_count": 5,
            "teleport_between_edge_endpoints": False,
        },
        "vlm_configuration": {
            "backend": "deepseek", "model": "backend_default",
            "prompt_version": "v8_eight_view_spatial_relations",
            "temperature": 0, "thinking": "disabled", "retries": 2,
        },
        "perception": {
            "environment_semantics": "Grounding DINO + SAM",
            "six_view_offsets_deg": [0, 60, 120, 180, 240, 300],
            "completion_view_offsets_deg": [0, 45, 90, 135, 180, 225, 270, 315],
        },
        "device": device, "seed": 17,
        "code_hashes": {
            name: sha256(ROOT / "scripts" / name) for name in (
                "evaluate_instruction_completion_triplets.py",
                "vlm_harness.py", "instruction_completion_judge.py",
                "navigation_graph_memory.py", "habitat_point_navigation.py")
        },
        "cases": cases,
    }
    path = output_root / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text())
        frozen_keys = (
            "dataset_sha256", "sample_design", "edge_construction", "cases")
        if any(existing[key] != manifest[key] for key in frozen_keys):
            raise RuntimeError(f"refusing to alter frozen manifest {path}")
        return existing
    write_json(path, manifest)
    return manifest


def save_rgbd(directory, prefix, rgbs, depths):
    rgb_dir = directory / f"{prefix}_six_views"
    depth_dir = directory / f"{prefix}_depths"
    rgb_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)
    for index, (rgb, depth) in enumerate(zip(rgbs, depths)):
        Image.fromarray(np.asarray(rgb, np.uint8)).save(
            rgb_dir / f"view_{index}.jpg", quality=95)
        np.save(depth_dir / f"view_{index}.npy", np.asarray(depth, np.float32))


def append_video_frame(sink, composer, rgb, history, position, yaw,
                       instruction, category, repeat=1):
    if sink is None:
        return
    frame = composer.compose(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), history, position, yaw,
        instruction, target_idx=0, phase=f"completion_test_{category}")
    for _ in range(repeat):
        sink.append(frame.copy())


def replay_reference_edge(sim, path, previous_index, current_index,
                          initial_yaw, instruction, category, sink, composer):
    direction = 1 if current_index > previous_index else -1
    indices = list(range(previous_index, current_index, direction))
    position = np.asarray(path[previous_index], np.float32)
    yaw = float(initial_yaw)
    set_pose(sim, position, yaw)
    history_positions = [position.copy()]
    front_frames = [sim.get_sensor_observations()["rgb"][..., :3].copy()]
    action_history = []
    append_video_frame(
        sink, composer, front_frames[-1], history_positions, position, yaw,
        instruction, category, repeat=3)

    def record(action, moved_m, turn_deg):
        rgb = sim.get_sensor_observations()["rgb"][..., :3].copy()
        front_frames.append(rgb)
        action_history.append({
            "step": len(action_history), "action": action,
            "moved_m": float(moved_m), "turn_deg": float(turn_deg),
            "position_xyz": position.tolist(), "yaw_rad": float(yaw),
        })
        append_video_frame(
            sink, composer, rgb, history_positions, position, yaw,
            instruction, category)

    for index in indices:
        requested_start = np.asarray(path[index], np.float32)
        requested_end = np.asarray(path[index + direction], np.float32)
        shortest = __import__("habitat_sim").ShortestPath()
        shortest.requested_start = requested_start
        shortest.requested_end = requested_end
        if not sim.pathfinder.find_path(shortest):
            raise RuntimeError(
                f"no navmesh path between reference states {index} and "
                f"{index + direction}")
        points = [np.asarray(value, np.float32) for value in shortest.points]
        for segment_start, segment_end in zip(points[:-1], points[1:]):
            target_yaw = heading_between(segment_start, segment_end)
            delta = wrap_angle(target_yaw - yaw)
            turn_count = int(math.ceil(abs(delta) / math.radians(15.0)))
            for _ in range(turn_count):
                remaining = wrap_angle(target_yaw - yaw)
                turn = float(np.clip(
                    remaining, -math.radians(15.0), math.radians(15.0)))
                yaw = wrap_angle(yaw + turn)
                set_pose(sim, position, yaw)
                record("turn_left" if turn > 0 else "turn_right", 0.0,
                       math.degrees(turn))
            distance = float(np.linalg.norm(segment_end - segment_start))
            move_count = max(1, int(math.ceil(distance / 0.22)))
            move_origin = position.copy()
            for move_index in range(1, move_count + 1):
                next_position = move_origin + (
                    segment_end - move_origin) * (move_index / move_count)
                moved = float(np.linalg.norm(next_position - position))
                position = np.asarray(next_position, np.float32)
                set_pose(sim, position, yaw)
                history_positions.append(position.copy())
                record("forward", moved, 0.0)
        position = requested_end.copy()
        set_pose(sim, position, yaw)
        history_positions[-1] = position.copy()

    sample_indices = np.linspace(
        0, len(front_frames) - 1, min(5, len(front_frames))).round().astype(int)
    keyframes = [front_frames[int(index)] for index in sample_indices]
    return {
        "position": position, "yaw": yaw,
        "action_history": action_history,
        "position_history": history_positions,
        "front_frames": front_frames, "keyframes": keyframes,
        "keyframe_source_indices": sample_indices.tolist(),
    }


def prepare_case(case, episode, output_root, mp3d_root, semantic_extractor):
    case_dir = output_root / f"case_{case['case_index']:03d}_{case['category']}"
    case_dir.mkdir(parents=True, exist_ok=True)
    case_manifest = {
        **case,
        "test_scope": "single_point_module_test",
        "target_modules": ["binary_edge_instruction_completion"],
        "required_upstream": [
            "real_reference_state_pair", "actual_habitat_edge_replay",
            "six_and_eight_view_capture", "dino_sam_node_semantics"],
        "label_frozen_before_vlm_call": True,
        "label_exposed_to_vlm": False,
        "not_run_modules": [
            "point_selection", "point_navigation_executor",
            "point_target_arrival", "backtracking"],
    }
    manifest_path = case_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if any(existing.get(key) != case_manifest.get(key) for key in (
                "case_index", "category", "manual_edge_event_label",
                "episode_index", "previous_path_index", "current_path_index")):
            raise RuntimeError(f"refusing to alter frozen case {manifest_path}")
    else:
        write_json(manifest_path, case_manifest)
    graph_path = case_dir / "navigation_graph/navigation_graph.json"
    if graph_path.exists():
        return case_dir

    scene = resolve_mp3d_scene(episode["scene_id"], mp3d_root)
    sim = make_sim(scene, 320, 240)
    path = np.asarray(episode["reference_path"], np.float32)
    previous_index = int(case["previous_path_index"])
    current_index = int(case["current_path_index"])
    previous = path[previous_index]
    if not sim.pathfinder.is_navigable(previous):
        raise RuntimeError(f"reference state is not navigable: {previous.tolist()}")
    set_pose(sim, previous, float(case["previous_yaw_rad"]))
    origin_rgbs, origin_depths = observe_six_rgbd(sim)
    origin_completion = observe_eight_rgb(sim)
    save_rgbd(case_dir, "initial", origin_rgbs, origin_depths)

    composer = ExplorationVideoComposer(
        sim, previous, obs_width=320, obs_height=240)
    sink = VideoFrameSink(case_dir / "exploration.mp4", composer.frame_size, fps=5)
    replay = replay_reference_edge(
        sim, path, previous_index, current_index,
        float(case["previous_yaw_rad"]),
        case["sub_instruction"]["navigation_instruction"],
        case["category"], sink, composer)
    current_rgbs, current_depths = observe_six_rgbd(sim)
    current_completion = observe_eight_rgb(sim)
    save_rgbd(case_dir, "arrival", current_rgbs, current_depths)
    append_video_frame(
        sink, composer, current_rgbs[0], replay["position_history"],
        replay["position"], replay["yaw"],
        case["sub_instruction"]["navigation_instruction"],
        f"{case['category']}_edge_ready", repeat=5)
    sink.close()

    keyframe_dir = case_dir / "edge_keyframes"
    keyframe_dir.mkdir(exist_ok=True)
    keyframe_records = []
    for index, (rgb, source_index) in enumerate(zip(
            replay["keyframes"], replay["keyframe_source_indices"])):
        path_out = keyframe_dir / f"keyframe_{index:02d}.jpg"
        Image.fromarray(rgb).save(path_out, quality=95)
        keyframe_records.append({
            "keyframe_index": index,
            "source_frame_index": int(source_index),
            "image_path": str(path_out.relative_to(case_dir)),
        })

    sub_instruction = SubInstruction.from_mapping(case["sub_instruction"])
    memory = NavigationGraphMemory(
        case_dir / "navigation_graph", semantic_extractor)
    memory.add_origin_node(
        previous, float(case["previous_yaw_rad"]), 0,
        origin_rgbs, origin_depths,
        metadata={
            "real_r2r_reference_state": True,
            "reference_path_index": previous_index,
            "hidden_label_not_for_model": case["manual_edge_event_label"],
        }, completion_views=origin_completion)
    node, edge = memory.add_navigation_stop_node(
        position_xyz=replay["position"], base_yaw_rad=replay["yaw"],
        global_step=len(replay["action_history"]),
        six_views=current_rgbs, six_depths=current_depths,
        sub_instruction=sub_instruction,
        action_history=replay["action_history"],
        arrival_signal="reference_edge_replay_arrived",
        metadata={
            "real_r2r_reference_state": True,
            "reference_path_index": current_index,
        }, edge_kind="frozen_reference_edge_replay",
        edge_metadata={"edge_keyframes": keyframe_records},
        completion_views=current_completion)
    write_json(case_dir / "forward_action_history.json", replay["action_history"])
    write_json(case_dir / "semantic_detections.json", {
        "previous": memory.nodes[0].environment_semantics,
        "current": node.environment_semantics,
    })
    write_json(case_dir / "trajectory.json", {
        "case_index": case["case_index"], "category": case["category"],
        "previous_node_id": memory.nodes[0].node_id,
        "current_node_id": node.node_id, "edge_id": edge.edge_id,
        "previous_position_xyz": previous.tolist(),
        "current_position_xyz": replay["position"].tolist(),
        "final_yaw_rad": replay["yaw"],
        "control_steps": len(replay["action_history"]),
        "traveled_distance_m": edge.traveled_distance_m,
        "keyframes": keyframe_records,
    })
    write_json(case_dir / "preparation_result.json", {
        "prepared": True, "vlm_called": False,
        "node_count": 2, "edge_count": 1,
        "exact_current_reference_position": bool(np.allclose(
            replay["position"], path[current_index], atol=1e-4)),
        "six_view_counts": [6, 6], "completion_view_counts": [8, 8],
    })
    sim.close()
    return case_dir


def load_images(root, records):
    return [np.asarray(Image.open(root / record["image_path"]).convert("RGB"))
            for record in sorted(records, key=lambda item: item["view_index"])]


def evaluate_case(case, case_dir, backend_name, deepseek_env, timeout):
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
        instruction_completion_prompt_version=(
            "v8_eight_view_spatial_relations"))
    expected = case["manual_edge_event_label"]
    try:
        prediction = harness.judge_edge_instruction_completion(
            sub_instruction=SubInstruction.from_mapping(case["sub_instruction"]),
            previous_node_id=previous["node_id"],
            current_node_id=current["node_id"],
            previous_position_xyz=previous["position_xyz"],
            current_position_xyz=current["position_xyz"],
            previous_six_views=previous_views, current_six_views=current_views,
            previous_environment_semantics=previous["environment_semantics"],
            current_environment_semantics=current["environment_semantics"],
            edge_action_history=edge["action_history"], edge_keyframes=keyframes)
    except RuntimeError as exc:
        prediction = {
            "status": "error", "confidence": 0.0,
            "reason": str(exc), "visual_evidence": "",
            "failure_counts_in_frozen_denominator": True,
        }
    record = {
        "case_index": case["case_index"], "category": case["category"],
        "expected_status": expected, "predicted_status": prediction["status"],
        "correct": prediction["status"] == expected,
        "confidence": prediction["confidence"], "result": prediction,
        "manual_label_rationale": case["manual_label_rationale"],
        "label_exposed_to_vlm": False,
    }
    write_json(result_path, record)
    write_json(case_dir / "vlm_response.json", prediction)
    prompt_record = (harness.calls[-1] if harness.calls else harness.attempts[0])
    (case_dir / "vlm_prompt.txt").write_text(prompt_record["prompt"] + "\n")
    Image.fromarray(harness._completion_contact_sheet(previous_views)).save(
        case_dir / "vlm_previous_contact_sheet.jpg", quality=95)
    Image.fromarray(harness._completion_contact_sheet(current_views)).save(
        case_dir / "vlm_current_contact_sheet.jpg", quality=95)
    combined = np.concatenate([
        harness._completion_contact_sheet(previous_views),
        harness._completion_contact_sheet(current_views)], axis=0)
    Image.fromarray(combined).save(
        case_dir / "vlm_contact_sheet.jpg", quality=95)
    write_json(case_dir / "module_results.json", {
        "test_scope": "single_point_module_test",
        "target_module": "binary_edge_instruction_completion",
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
    summary = {"count": len(results), "categories": {}}
    for category in CATEGORY_ORDER:
        values = [item for item in results if item["category"] == category]
        successes = sum(item["correct"] for item in values)
        completed = sum(
            item["predicted_status"] == "completed" for item in values)
        summary["categories"][category] = {
            "count": len(values), "correct": successes,
            "accuracy": successes / max(len(values), 1),
            "wilson_95_ci": wilson(successes, len(values)),
            "predicted_completed": completed,
            "predicted_unknown": sum(
                item["predicted_status"] == "unknown" for item in values),
            "prediction_errors": sum(
                item["predicted_status"] == "error" for item in values),
        }
    successes = sum(item["correct"] for item in results)
    correct_values = [item for item in results if item["category"] == "correct"]
    negative_values = [item for item in results if item["category"] != "correct"]
    true_positive = sum(
        item["predicted_status"] == "completed" for item in correct_values)
    false_negative = sum(
        item["predicted_status"] == "unknown" for item in correct_values)
    false_positive = sum(
        item["predicted_status"] == "completed" for item in negative_values)
    true_negative = sum(
        item["predicted_status"] == "unknown" for item in negative_values)
    prediction_errors = sum(
        item["predicted_status"] == "error" for item in results)
    summary.update({
        "correct": successes, "accuracy": successes / max(len(results), 1),
        "wilson_95_ci": wilson(successes, len(results)),
        "confusion": {
            "true_positive": true_positive, "false_negative": false_negative,
            "false_positive": false_positive, "true_negative": true_negative,
        },
        "completed_precision": true_positive / max(
            true_positive + false_positive, 1),
        "completed_recall": true_positive / max(
            len(correct_values), 1),
        "prediction_errors": prediction_errors,
        "label_semantics": {
            "correct": "completed", "wrong": "unknown", "halfway": "unknown",
            "unknown_does_not_classify_on_route_or_off_route": True,
        },
    })
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["prepare", "evaluate", "all"],
                        default="all")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--r2r-data", type=Path, default=DEFAULT_R2R_DATA)
    parser.add_argument("--mp3d-root", type=Path, default=DEFAULT_MP3D_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vlm-backend", choices=["deepseek", "ollama", "heuristic"],
                        default="deepseek")
    parser.add_argument("--deepseek-env", type=Path, default=ROOT / ".env.deepseek")
    parser.add_argument("--vlm-timeout", type=int, default=180)
    args = parser.parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    episodes = load_dataset(args.r2r_data)
    cases = build_frozen_cases(episodes)
    initialize_manifest(args.output_root, args.r2r_data, cases, args.device)

    if args.phase in {"prepare", "all"}:
        detector = DinoSamDetector(args.device)
        semantic_extractor = DinoSamEnvironmentSemanticExtractor(detector)
        for case in cases:
            case_dir = prepare_case(
                case, episodes[case["episode_index"]], args.output_root,
                args.mp3d_root, semantic_extractor)
            print(
                f"prepared case={case['case_index']:02d} "
                f"category={case['category']} dir={case_dir}", flush=True)
        write_json(args.output_root / "preparation_complete.json", {
            "prepared_case_count": len(cases), "vlm_called": False,
            "labels_frozen": True,
        })

    if args.phase in {"evaluate", "all"}:
        missing = [case["case_index"] for case in cases if not (
            args.output_root /
            f"case_{case['case_index']:03d}_{case['category']}" /
            "navigation_graph/navigation_graph.json").exists()]
        if missing:
            raise RuntimeError(
                f"prepare phase is incomplete; missing cases {missing}")
        results = []
        for case in cases:
            case_dir = (args.output_root /
                        f"case_{case['case_index']:03d}_{case['category']}")
            result = evaluate_case(
                case, case_dir, args.vlm_backend,
                args.deepseek_env, args.vlm_timeout)
            results.append(result)
            print(
                f"evaluated case={case['case_index']:02d} "
                f"category={case['category']} expected={result['expected_status']} "
                f"predicted={result['predicted_status']} "
                f"correct={int(result['correct'])}", flush=True)
        summary = summarize(results)
        write_json(args.output_root / "results.json", results)
        write_json(args.output_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
