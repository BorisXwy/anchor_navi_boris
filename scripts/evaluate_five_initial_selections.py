#!/usr/bin/env python3
"""Five-scene R2R protocol: decomposition and first-stage point selection only."""

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from instruction_taxonomy import decompose_by_definition
from habitat_point_navigation import (
    DEFAULT_MP3D_ROOT, make_sim, resolve_mp3d_scene, set_pose, yaw_from_coeffs,
)
from point_selectors import (
    observe_six_rgbd, select_ground_point, targetable_ground_mask, wrap_angle,
)
from semantic_detector import (DinoSamDetector, DinoSamFloorSegmenter,
                               GroundedSamDetector, extract_detection_queries)
from ground_segmentation_backends import (
    build_ground_segmenter, ground_segmenter_display_name,
)
from semantic_point_strategy import DETECTION_GROUNDED_FORMS, constrain_floor_candidates
from path_projection import draw_reference_path_overlay, project_reference_path
from vlm_harness import NavigationVLMHarness, build_vlm_backend


DEFAULT_DATA = ROOT.parent / "3d_wm_vln/StreamVLN/data/datasets/r2r/val_unseen/val_unseen.json.gz"
OFFSETS = np.radians([0, 60, 120, 180, 240, 300])


def five_unique_scenes(dataset):
    with gzip.open(dataset, "rt") as handle:
        episodes = json.load(handle)["episodes"]
    selected, seen = [], set()
    for episode_index, episode in enumerate(episodes):
        if episode["scene_id"] in seen:
            continue
        seen.add(episode["scene_id"])
        selected.append((episode_index, episode))
        if len(selected) == 5:
            return selected
    raise RuntimeError("dataset contains fewer than five scenes")


def enrich_stage(stage):
    typed = decompose_by_definition(stage["navigation_instruction"])
    if typed:
        stage.update({key: typed[0][key] for key in (
            "source_clause", "form", "secondary_forms", "definition",
            "point_selection_strategy")})
    return stage


def save_scene_visuals(scene_dir, candidates, chosen_index, chosen_point,
                       reference_path, position, base_yaw):
    raw = np.concatenate([candidate["rgb"] for candidate in candidates], axis=1)
    Image.fromarray(raw).save(scene_dir / "initial_six_views.jpg")
    selection_views = []
    initial_path_views = []
    camera_position = np.asarray(position, np.float32) + np.array(
        [0.0, 1.25, 0.0], np.float32)
    for index, candidate in enumerate(candidates):
        rgb = candidate["rgb"]
        ground = candidate["mask"]
        target = candidate["target_mask"]
        objects = candidate["small_seg_object_mask"]
        overlay = rgb.copy()
        overlay[objects] = (0.82 * overlay[objects] +
                            0.18 * np.array([0, 0, 255])).astype(np.uint8)
        overlay[ground] = (0.72 * overlay[ground] +
                           0.28 * np.array([0, 180, 0])).astype(np.uint8)
        overlay[target] = (0.55 * overlay[target] +
                           0.45 * np.array([0, 255, 0])).astype(np.uint8)
        for detection in candidate["semantic_detections"]:
            x0, y0, x1, y1 = np.round(detection.box_xyxy).astype(int)
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 255, 0), 2)
        projection = project_reference_path(
            reference_path, range(len(reference_path)), camera_position,
            candidate["yaw"], rgb.shape[1], rgb.shape[0])
        overlay = draw_reference_path_overlay(
            overlay, projection,
            selected_point=(chosen_point if index == chosen_index else None),
            selected=bool(index == chosen_index),
            next_path_index=1)
        initial_path_views.append(
            draw_reference_path_overlay(
                rgb.copy(), projection, next_path_index=1))
        cv2.putText(overlay, f"VIEW {index}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2)
        if index == chosen_index:
            cv2.drawMarker(overlay, tuple(np.round(chosen_point).astype(int)),
                           (255, 0, 0), cv2.MARKER_CROSS, 22, 2)
            cv2.putText(overlay, "SELECTED", (6, 42), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (255, 255, 0), 2)
        binary = np.repeat((ground.astype(np.uint8) * 255)[..., None], 3, axis=2)
        Image.fromarray(np.concatenate([rgb, overlay, binary], axis=1)).save(
            scene_dir / f"view_{index}_ground_segmentation.jpg")
        selection_views.append(overlay)
    Image.fromarray(np.concatenate(initial_path_views, axis=1)).save(
        scene_dir / "initial_six_views_path_overlay.jpg")
    sheet = np.concatenate([
        np.concatenate(selection_views[:3], axis=1),
        np.concatenate(selection_views[3:], axis=1),
    ], axis=0)
    Image.fromarray(sheet).save(scene_dir / "final_point_selection.jpg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2r-data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--mp3d-root", type=Path, default=DEFAULT_MP3D_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vlm-backend", choices=["deepseek", "ollama", "heuristic"],
                        default="deepseek")
    parser.add_argument("--vlm-model", default=None)
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--deepseek-env", type=Path, default=ROOT / ".env.deepseek")
    parser.add_argument("--deepseek-base-url", default=None)
    parser.add_argument("--fixed-decomposition", type=Path,
                        help="reuse stages from an earlier five-scene summary")
    parser.add_argument(
        "--floor-segmenter",
        choices=["dense-majority", "grounded-sam", "dino-sam"],
        default="dense-majority",
        help="ground mask backend (default: 2/3 dense ADE20K majority)")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/five_initial_selections")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(17); torch.manual_seed(17)

    backend = build_vlm_backend(
        args.vlm_backend, args.vlm_model, args.ollama_host, 180,
        deepseek_env_file=args.deepseek_env,
        deepseek_base_url=args.deepseek_base_url)
    args.vlm_model = getattr(backend, "model", args.vlm_model or args.vlm_backend)
    segmenter, floor_detector = build_ground_segmenter(
        args.floor_segmenter, args.device)
    detector = floor_detector or GroundedSamDetector(args.device)
    fixed = {}
    if args.fixed_decomposition:
        prior = json.loads(args.fixed_decomposition.read_text())
        fixed = {str(item["episode_id"]): item["decomposed_stages"]
                 for item in prior["scenes"]}
    results = []

    for scene_index, (episode_index, episode) in enumerate(five_unique_scenes(args.r2r_data)):
        scene_dir = args.output_dir / f"scene_{scene_index:02d}_episode_{episode['episode_id']}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        harness = NavigationVLMHarness(
            backend, scene_dir / "vlm_calls.json", retries=2)
        instruction = episode["instruction"]["instruction_text"].strip()
        if str(episode["episode_id"]) in fixed:
            stages = fixed[str(episode["episode_id"])]
        else:
            stages = [enrich_stage(stage) for stage in
                      harness.decompose_instruction(instruction)]
        first_stage = stages[0]

        sim = make_sim(resolve_mp3d_scene(episode["scene_id"], args.mp3d_root), 320, 240)
        position = np.asarray(episode["start_position"], np.float32)
        yaw = yaw_from_coeffs(episode["start_rotation"])
        set_pose(sim, position, yaw)
        rgbs, depths = observe_six_rgbd(sim)
        segmentations = segmenter.batch(rgbs, depths)
        queries = extract_detection_queries(first_stage)
        candidates = []
        for offset, rgb, depth, (ground, ground_detections) in zip(
                OFFSETS, rgbs, depths, segmentations):
            objects = np.zeros_like(ground)
            detections = detector.detect(rgb, queries, depth) if queries else []
            target, audit = constrain_floor_candidates(
                first_stage, targetable_ground_mask(ground), depth, detections, objects)
            if getattr(segmenter, "strict_ground_mask", False):
                target = np.asarray(target, bool) & np.asarray(ground, bool)
            point, point_score = select_ground_point(target)
            candidates.append({
                "yaw": wrap_angle(yaw + float(offset)), "relative_yaw_rad": float(offset),
                "rgb": rgb, "mask": ground, "target_mask": target, "point": point,
                "strict_ground_mask": bool(
                    getattr(segmenter, "strict_ground_mask", False)),
                "excluded": False, "ground_fraction": float(ground.mean()),
                "point_score": point_score, "semantic_detections": detections,
                "detection_records": [item.prompt_record() for item in detections],
                "small_seg_object_mask": objects,
                "small_seg_objects": [],
                "ground_detection_records": [item.prompt_record()
                                             for item in ground_detections],
                "strategy_application": audit,
            })
        if first_stage.get("form") in DETECTION_GROUNDED_FORMS:
            grounded = [item for item in candidates if item["strategy_application"]["mode"]
                        not in {"floor_only", "floor_only_fallback"}]
            if grounded:
                for item in candidates:
                    if item["strategy_application"]["mode"] in {
                            "floor_only", "floor_only_fallback"}:
                        item["point"] = None
        chosen_index, chosen_point, selection = harness.select_ground_target(
            first_stage, candidates, [])
        if getattr(segmenter, "strict_ground_mask", False):
            strict_mask = np.asarray(
                candidates[chosen_index]["target_mask"], bool)
            xy = np.round(np.asarray(chosen_point, np.float32)).astype(int)
            xy[0] = np.clip(xy[0], 0, strict_mask.shape[1] - 1)
            xy[1] = np.clip(xy[1], 0, strict_mask.shape[0] - 1)
            if not strict_mask[xy[1], xy[0]]:
                ys, xs = np.nonzero(strict_mask)
                if not len(xs):
                    raise RuntimeError(
                        "selected view has no dense-majority ground anchor")
                nearest = np.argmin(
                    (xs.astype(np.float32) - float(chosen_point[0])) ** 2 +
                    (ys.astype(np.float32) - float(chosen_point[1])) ** 2)
                chosen_point = np.array(
                    [float(xs[nearest]), float(ys[nearest])], np.float32)
                selection = dict(selection or {})
                selection["strict_ground_snap"] = {
                    "applied": True, "final_xy": chosen_point.tolist()}
        save_scene_visuals(
            scene_dir, candidates, chosen_index, chosen_point,
            episode.get("reference_path", []), position, yaw)
        sim.close()

        record = {
            "scene_index": scene_index, "episode_index": episode_index,
            "episode_id": episode["episode_id"], "scene_id": episode["scene_id"],
            "instruction": instruction, "decomposed_stages": stages,
            "first_stage": first_stage,
            "point_selection_logic": first_stage["point_selection_strategy"],
            "detection_queries": queries,
            "views": [{
                "view_index": index, "yaw_offset_deg": index * 60,
                "ground_fraction": item["ground_fraction"],
                "small_seg_objects": item["small_seg_objects"],
                "ground_detections": item["ground_detection_records"],
                "detections": item["detection_records"],
                "strategy_application": item["strategy_application"],
                "reference_path_projection": project_reference_path(
                    episode.get("reference_path", []),
                    range(len(episode.get("reference_path", []))),
                    position + np.array([0.0, 1.25, 0.0], np.float32),
                    item["yaw"], 320, 240),
                "eligible": item["point"] is not None,
                "segmentation_image": f"view_{index}_ground_segmentation.jpg",
            } for index, item in enumerate(candidates)],
            "final_selection": {
                "view_index": chosen_index, "point_xy": chosen_point.tolist(),
                "relative_yaw_deg": chosen_index * 60, **selection,
            },
            "files": {"six_views": "initial_six_views.jpg",
                      "six_views_path_overlay": "initial_six_views_path_overlay.jpg",
                      "selection": "final_point_selection.jpg"},
        }
        (scene_dir / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        results.append(record)
        print(f"[{scene_index + 1}/5] episode={episode['episode_id']} "
              f"form={first_stage.get('form')} view={chosen_index} "
              f"point={chosen_point.tolist()}", flush=True)

    summary = {"protocol": "fixed_decomposition_then_first_stage_initial_selection_only",
               "floor_segmenter": ground_segmenter_display_name(
                   args.floor_segmenter),
               "strict_ground_selection": bool(
                   getattr(segmenter, "strict_ground_mask", False)),
               "ground_queries": getattr(segmenter, "queries", None),
               "object_segmentation": False,
               "path_projection_visualization": {
                   "enabled": True,
                   "reference": "R2R reference_path projected after selection",
                   "ground_truth_color_rgb": [0, 230, 255],
                   "selected_point_color_rgb": [255, 30, 30],
                   "path_is_not_model_input": True,
               },
               "fixed_decomposition": str(args.fixed_decomposition) if args.fixed_decomposition else None,
               "scenes": results}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Five-scene initial point-selection protocol", ""]
    for item in results:
        lines += [f"## Scene {item['scene_index']} / episode {item['episode_id']}", "",
                  f"- Instruction: {item['instruction']}",
                  f"- First stage: {item['first_stage']['navigation_instruction']}",
                  f"- Form: {item['first_stage'].get('form')}",
                  f"- Selected view/point: {item['final_selection']['view_index']} / "
                  f"{item['final_selection']['point_xy']}",
                  f"- Artifacts: `{item['files']['six_views']}`, "
                  f"`{item['files']['six_views_path_overlay']}`, "
                  f"`{item['files']['selection']}`", ""]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
