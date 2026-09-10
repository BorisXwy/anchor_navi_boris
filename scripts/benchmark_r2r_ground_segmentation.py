#!/usr/bin/env python3
"""Compare public ground segmenters on frozen real R2R demonstration states."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_r2r_point_selection import heading_between
from ground_segmentation_backends import (
    EXPANDED_GROUND_QUERIES,
    SAFE_STAIRS_GROUND_QUERIES,
    GroundedSamGroundBackend,
    build_dense_ground_backend,
)
from habitat_point_navigation import (
    make_sim,
    resolve_mp3d_scene,
    set_pose,
    yaw_from_coeffs,
)
from path_projection import draw_reference_path_overlay, project_reference_path
from point_selectors import (
    EIGHT_VIEW_YAW_OFFSETS_DEG,
    observe_eight_rgbd,
    select_ground_point,
    targetable_ground_mask,
    wrap_angle,
)
from semantic_detector import GROUND_QUERIES, GroundedSamDetector
from vlm_harness import NavigationVLMHarness


DEFAULT_DATA = Path(
    "/sharedata/datasets/R2R/R2R_VLNCE_v1-3/val_unseen/val_unseen.json.gz")
DEFAULT_SCENES = Path("/sharedata/datasets/mp3d/v1/tasks/mp3d")
DEFAULT_EPISODES = (0, 3, 6, 9, 18, 27, 45, 126, 204, 219)
DEFAULT_MODELS = (
    "grounded_sam_current",
    "grounded_sam_safe_stairs",
    "grounded_sam_expanded",
    "oneformer_ade20k",
    "mask2former_ade20k",
    "segformer_ade20k",
    "dense_majority",
)
MODEL_IDS = {
    "grounded_sam_current": (
        "IDEA-Research Grounding-DINO Swin-T + Meta SAM ViT-H"),
    "grounded_sam_safe_stairs": (
        "IDEA-Research Grounding-DINO Swin-T + Meta SAM ViT-H"),
    "grounded_sam_expanded": (
        "IDEA-Research Grounding-DINO Swin-T + Meta SAM ViT-H"),
    "oneformer_ade20k": "shi-labs/oneformer_ade20k_swin_tiny",
    "mask2former_ade20k": "facebook/mask2former-swin-small-ade-semantic",
    "segformer_ade20k": "nvidia/segformer-b5-finetuned-ade-640-640",
    "dense_majority": (
        "pixelwise majority of OneFormer, Mask2Former, and SegFormer ADE20K"),
}


def _angle_distance(a, b):
    return abs(wrap_angle(float(a) - float(b)))


def _state_indices(path_length, states_per_episode):
    if path_length < 2:
        return []
    candidates = [0]
    if states_per_episode >= 2 and path_length > 2:
        candidates.append(max(1, min(path_length - 2, (path_length - 1) // 2)))
    if states_per_episode >= 3 and path_length > 3:
        candidates.append(path_length - 2)
    return list(dict.fromkeys(candidates))[:states_per_episode]


def _path_raster(projection, shape, next_path_index, thickness=13):
    """Rasterize only the next demonstration waypoint as a hidden proxy.

    Projecting every future polyline segment through an RGB image draws
    occluded room-to-room edges across walls.  Those are useful in the audit
    overlay, but invalid as pixel ground supervision.  The immediate next
    waypoint is the least ambiguous support-surface proxy available in R2R.
    """
    raster = np.zeros(shape, np.uint8)
    for vertex in projection.get("vertices", []):
        pixel = vertex.get("pixel_xy")
        if (pixel is not None and
                int(vertex.get("path_index", -1)) == int(next_path_index) and
                bool(vertex.get("visible_in_camera", False))):
            cv2.circle(raster, tuple(int(round(value)) for value in pixel),
                       max(2, thickness), 1, -1)
    return raster.astype(bool)


def _render_model_panel(rgb, mask, projection, point, title, metrics):
    rendered = np.asarray(rgb, np.uint8).copy()
    green = np.zeros_like(rendered)
    green[..., 1] = 255
    rendered[mask] = (
        0.52 * rendered[mask] + 0.48 * green[mask]).astype(np.uint8)
    rendered = draw_reference_path_overlay(
        rendered, projection, selected_point=point,
        selected=point is not None, selected_label="MASK POINT",
        legend_point_label="MASK POINT")
    for anchor_index, anchor in enumerate(metrics.get("ground_anchors_xy", [])):
        xy = tuple(np.round(anchor).astype(int))
        cv2.circle(rendered, xy, 5, (255, 255, 255), -1)
        cv2.circle(rendered, xy, 5, (20, 20, 20), 1)
        cv2.putText(rendered, str(anchor_index), (xy[0] - 3, xy[1] + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 0), 1)
    cv2.rectangle(rendered, (0, 0), (rendered.shape[1], 43), (0, 0, 0), -1)
    cv2.putText(rendered, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.40, (255, 255, 255), 1, cv2.LINE_AA)
    path_support = metrics["path_support"]
    path_text = f"{path_support:.2f}" if path_support is not None else "N/A"
    status = (f"mask={metrics['mask_fraction']:.2f} "
              f"path={path_text} "
              f"inside={int(metrics['point_on_raw_mask'])}")
    cv2.putText(rendered, status, (6, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.34, (255, 255, 255), 1, cv2.LINE_AA)
    return rendered


def _render_mask_panel(mask, title, point=None, anchors=()):
    rendered = np.zeros((*mask.shape, 3), np.uint8)
    rendered[mask] = (45, 220, 80)
    if point is not None:
        xy = tuple(np.round(point).astype(int))
        cv2.drawMarker(rendered, xy, (255, 35, 35), cv2.MARKER_CROSS, 20, 2)
    for anchor_index, anchor in enumerate(anchors):
        xy = tuple(np.round(anchor).astype(int))
        cv2.circle(rendered, xy, 4, (255, 255, 255), -1)
        cv2.putText(rendered, str(anchor_index), (xy[0] - 3, xy[1] + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.26, (0, 0, 0), 1)
    cv2.rectangle(rendered, (0, 0), (rendered.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(rendered, f"RAW MASK: {title}", (6, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1,
                cv2.LINE_AA)
    return rendered


def _metrics(mask, path_mask, point):
    mask = np.asarray(mask, bool)
    h, w = mask.shape
    targetable = targetable_ground_mask(mask)
    anchors = NavigationVLMHarness._task30_ground_anchors(
        targetable, count=6)
    point_on_raw = False
    point_on_targetable = False
    margin = 0.0
    if point is not None:
        x = int(np.clip(round(float(point[0])), 0, w - 1))
        y = int(np.clip(round(float(point[1])), 0, h - 1))
        point_on_raw = bool(mask[y, x])
        point_on_targetable = bool(targetable[y, x])
        margin = float(cv2.distanceTransform(
            mask.astype(np.uint8), cv2.DIST_L2, 5)[y, x])
    path_count = int(path_mask.sum())
    return {
        "mask_fraction": float(mask.mean()),
        "targetable_fraction": float(targetable.mean()),
        "path_pixel_count": path_count,
        "path_support": (float((mask & path_mask).sum() / path_count)
                         if path_count else None),
        "upper_mask_share": float(mask[:max(1, int(0.35 * h))].sum() /
                                  max(int(mask.sum()), 1)),
        "lower_band_support": float(mask[int(0.60 * h):].mean()),
        "point_xy": point.tolist() if point is not None else None,
        "point_on_raw_mask": point_on_raw,
        "point_on_targetable_mask": point_on_targetable,
        "point_boundary_margin_px": margin,
        "ground_anchors_xy": [
            np.asarray(anchor, np.float32).tolist() for anchor in anchors],
        "ground_anchor_count": len(anchors),
        "all_ground_anchors_on_raw_mask": bool(
            anchors and all(mask[int(round(float(anchor[1]))),
                                 int(round(float(anchor[0])))]
                            for anchor in anchors)),
        "all_ground_anchors_on_targetable_mask": bool(
            anchors and all(targetable[int(round(float(anchor[1]))),
                                       int(round(float(anchor[0])))]
                            for anchor in anchors)),
    }


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--mp3d-root", type=Path, default=DEFAULT_SCENES)
    parser.add_argument("--episode-indices", default=",".join(
        str(value) for value in DEFAULT_EPISODES))
    parser.add_argument("--states-per-episode", type=int, default=2,
                        choices=(1, 2, 3))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-inputs", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("all segmentation model inference must run on GPU")
    episodes_requested = tuple(int(value) for value in
                               args.episode_indices.split(",") if value.strip())
    model_names = tuple(value.strip() for value in args.models.split(",")
                        if value.strip())
    unknown = sorted(set(model_names) - set(DEFAULT_MODELS))
    if unknown:
        raise ValueError(f"unsupported model names: {unknown}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = args.output_dir / "frozen_inputs"
    cases_dir = args.output_dir / "cases"
    inputs_dir.mkdir(exist_ok=True)
    cases_dir.mkdir(exist_ok=True)

    previous_manifest_path = args.output_dir / "manifest.json"
    if args.reuse_inputs and previous_manifest_path.exists():
        samples = json.loads(previous_manifest_path.read_text())["samples"]
        print(f"reusing {len(samples)} frozen R2R inputs", flush=True)
    else:
        with gzip.open(args.data, "rt") as handle:
            all_episodes = json.load(handle)["episodes"]
        samples = []
        active_scene = None
        sim = None
        offsets = np.radians(EIGHT_VIEW_YAW_OFFSETS_DEG)
        try:
            for episode_index in episodes_requested:
                episode = all_episodes[episode_index]
                path = np.asarray(episode.get("reference_path", []), np.float32)
                scene = resolve_mp3d_scene(episode["scene_id"], args.mp3d_root)
                if scene != active_scene:
                    if sim is not None:
                        sim.close()
                    sim = make_sim(scene, args.width, args.height)
                    active_scene = scene
                for state_index in _state_indices(
                        len(path), args.states_per_episode):
                    position = path[state_index]
                    yaw = (yaw_from_coeffs(episode["start_rotation"])
                           if state_index == 0 else
                           heading_between(path[state_index - 1], position))
                    next_heading = heading_between(position, path[state_index + 1])
                    view_index = int(np.argmin([
                        _angle_distance(wrap_angle(yaw + offset), next_heading)
                        for offset in offsets]))
                    set_pose(sim, position, yaw)
                    rgbs, _ = observe_eight_rgbd(sim)
                    case_index = len(samples)
                    case_dir = cases_dir / f"case_{case_index:03d}"
                    all_views_dir = case_dir / "eight_views"
                    all_views_dir.mkdir(parents=True, exist_ok=True)
                    for index, view in enumerate(rgbs):
                        Image.fromarray(view).save(all_views_dir / f"view_{index}.jpg")
                    rgb = np.asarray(rgbs[view_index], np.uint8)
                    image_path = inputs_dir / f"case_{case_index:03d}.png"
                    Image.fromarray(rgb).save(image_path)
                    camera_yaw = wrap_angle(yaw + float(offsets[view_index]))
                    camera_position = position + np.array(
                        [0.0, 1.25, 0.0], np.float32)
                    projection = project_reference_path(
                        path, range(state_index, len(path)), camera_position,
                        camera_yaw, args.width, args.height)
                    sample = {
                    "case_index": case_index,
                    "split": "val_unseen",
                    "episode_index": episode_index,
                    "episode_id": str(episode["episode_id"]),
                    "scene_id": episode["scene_id"],
                    "instruction": episode["instruction"]["instruction_text"],
                    "reference_path_index": state_index,
                    "reference_path_length": len(path),
                    "reference_position_xyz": position.tolist(),
                    "reference_yaw_rad": float(yaw),
                    "yaw_source": ("episode_start_rotation" if state_index == 0
                                   else "previous_to_current_reference_state"),
                    "evaluation_view_index": view_index,
                    "evaluation_view_offset_deg": float(
                        EIGHT_VIEW_YAW_OFFSETS_DEG[view_index]),
                    "evaluation_camera_yaw_rad": float(camera_yaw),
                    "evaluation_view_selection": (
                        "frozen nearest 45-degree sector to next reference edge; "
                        "used only to define the evaluation image"),
                    "future_reference_exposed_to_segmenter": False,
                    "input_path": str(image_path.resolve()),
                    "input_sha256": _sha256(image_path),
                    "reference_path_projection": projection,
                    }
                    (case_dir / "sample.json").write_text(
                        json.dumps(sample, ensure_ascii=False, indent=2) + "\n")
                    samples.append(sample)
        finally:
            if sim is not None:
                sim.close()

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rule_file": str((ROOT / "project_rulle.md").resolve()),
        "test_scope": "module",
        "target_modules": ["ground_segmentation", "mask_constrained_point_sampling"],
        "required_upstream_modules": [
            "real_r2r_reference_path_state_initialization", "eight_view_rgb"],
        "not_run_modules": [
            "instruction_decomposition", "vlm_point_selection",
            "point_navigation_executor", "node_completion_judge", "backtracking"],
        "ablation_reason": (
            "user-requested public-model comparison replacing the fixed "
            "Grounded-SAM rule only inside this diagnostic benchmark"),
        "rgb_only_model_inference": True,
        "device": args.device,
        "data": str(args.data.resolve()),
        "models": list(model_names),
        "grounded_sam_current_queries": list(GROUND_QUERIES),
        "grounded_sam_safe_stairs_queries": list(SAFE_STAIRS_GROUND_QUERIES),
        "grounded_sam_expanded_queries": list(EXPANDED_GROUND_QUERIES),
        "samples": samples,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    results = {sample["case_index"]: {} for sample in samples}
    shared_grounded_detector = None
    for model_index, model_name in enumerate(model_names):
        model_dir = args.output_dir / "model_masks" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        if model_name == "dense_majority":
            dense_names = (
                "oneformer_ade20k", "mask2former_ade20k",
                "segformer_ade20k")
            missing = [name for name in dense_names
                       if any(name not in results[sample["case_index"]]
                              for sample in samples)]
            if missing:
                raise RuntimeError(
                    "dense_majority must follow all dense backends: " +
                    ", ".join(missing))
            print(f"[{model_index + 1}/{len(model_names)}] deriving "
                  "dense_majority", flush=True)
            for sample in samples:
                dense_masks = [
                    np.asarray(Image.open(
                        results[sample["case_index"]][name]["mask_path"])) > 0
                    for name in dense_names]
                mask = np.sum(dense_masks, axis=0) >= 2
                point, _ = select_ground_point(mask)
                path_mask = _path_raster(
                    sample["reference_path_projection"], mask.shape,
                    sample["reference_path_index"] + 1)
                mask_path = model_dir / f"case_{sample['case_index']:03d}.png"
                Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
                results[sample["case_index"]][model_name] = {
                    "model": model_name,
                    "model_id": MODEL_IDS[model_name],
                    "mask_path": str(mask_path.resolve()),
                    "mask_sha256": _sha256(mask_path),
                    "labels": ["ground-like dense semantic majority"],
                    "class_records": [],
                    "derived_from": list(dense_names),
                    **_metrics(mask, path_mask, point),
                }
            continue
        cached_paths = [
            model_dir / f"case_{sample['case_index']:03d}.png"
            for sample in samples]
        if args.reuse_inputs and all(path.exists() for path in cached_paths):
            print(f"[{model_index + 1}/{len(model_names)}] reusing {model_name}",
                  flush=True)
            for sample, mask_path in zip(samples, cached_paths):
                rgb = np.asarray(Image.open(sample["input_path"]).convert("RGB"))
                mask = np.asarray(Image.open(mask_path)) > 0
                point, _ = select_ground_point(mask)
                path_mask = _path_raster(
                    sample["reference_path_projection"], mask.shape,
                    sample["reference_path_index"] + 1)
                results[sample["case_index"]][model_name] = {
                    "model": model_name,
                    "model_id": MODEL_IDS[model_name],
                    "mask_path": str(mask_path.resolve()),
                    "mask_sha256": _sha256(mask_path),
                    "labels": [],
                    "class_records": [],
                    "cache_reused_after_completed_inference": True,
                    **_metrics(mask, path_mask, point),
                }
            continue
        print(f"[{model_index + 1}/{len(model_names)}] loading {model_name}",
              flush=True)
        if model_name.startswith("grounded_sam_"):
            if shared_grounded_detector is None:
                shared_grounded_detector = GroundedSamDetector(device=args.device)
            backend = GroundedSamGroundBackend(
                device=args.device,
                queries=(
                    GROUND_QUERIES
                    if model_name == "grounded_sam_current" else
                    SAFE_STAIRS_GROUND_QUERIES
                    if model_name == "grounded_sam_safe_stairs" else
                    EXPANDED_GROUND_QUERIES),
                detector=shared_grounded_detector, name=model_name)
        else:
            if shared_grounded_detector is not None:
                # Dense models get the full GPU; no CPU inference is performed.
                del shared_grounded_detector
                shared_grounded_detector = None
                torch.cuda.empty_cache()
            backend = build_dense_ground_backend(model_name, args.device)
        try:
            for sample_index, sample in enumerate(samples):
                rgb = np.asarray(Image.open(sample["input_path"]).convert("RGB"))
                output = backend.predict(rgb)
                mask = np.asarray(output.mask, bool)
                if mask.shape != rgb.shape[:2]:
                    raise RuntimeError(
                        f"{model_name} mask shape {mask.shape} != RGB {rgb.shape[:2]}")
                point, _ = select_ground_point(mask)
                path_mask = _path_raster(
                    sample["reference_path_projection"], mask.shape,
                    sample["reference_path_index"] + 1)
                metrics = _metrics(mask, path_mask, point)
                mask_path = model_dir / f"case_{sample['case_index']:03d}.png"
                Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
                record = {
                    "model": model_name,
                    "model_id": backend.model_id,
                    "mask_path": str(mask_path.resolve()),
                    "mask_sha256": _sha256(mask_path),
                    "labels": output.labels,
                    "class_records": output.records,
                    **metrics,
                }
                results[sample["case_index"]][model_name] = record
                print(f"  [{sample_index + 1:02d}/{len(samples)}] "
                      f"case={sample['case_index']:03d} "
                      f"mask={metrics['mask_fraction']:.3f} "
                      f"path={metrics['path_support']}", flush=True)
        finally:
            backend.close()
            del backend
    if shared_grounded_detector is not None:
        del shared_grounded_detector
        torch.cuda.empty_cache()

    per_case_records = []
    pairwise = {}
    for sample in samples:
        case_index = sample["case_index"]
        case_dir = cases_dir / f"case_{case_index:03d}"
        rgb = np.asarray(Image.open(sample["input_path"]).convert("RGB"))
        projection = sample["reference_path_projection"]
        original = draw_reference_path_overlay(
            rgb, projection, legend_point_label="MASK POINT")
        cv2.rectangle(original, (0, 0), (original.shape[1], 43), (0, 0, 0), -1)
        cv2.putText(original,
                    f"RGB ep={sample['episode_index']} state={sample['reference_path_index']}",
                    (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(original, "cyan=hidden demo path", (6, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                    (255, 255, 255), 1, cv2.LINE_AA)
        overlay_row = [original]
        mask_row = [np.zeros_like(original)]
        cv2.putText(mask_row[0], "MODEL RAW MASKS / RED POINT",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (255, 255, 255), 1, cv2.LINE_AA)
        masks = {}
        for model_name in model_names:
            record = results[case_index][model_name]
            mask = np.asarray(Image.open(record["mask_path"])) > 0
            masks[model_name] = mask
            point = (np.asarray(record["point_xy"], np.float32)
                     if record["point_xy"] is not None else None)
            overlay_row.append(_render_model_panel(
                rgb, mask, projection, point, model_name, record))
            mask_row.append(_render_mask_panel(
                mask, model_name, point, record.get("ground_anchors_xy", [])))
        for first_index, first in enumerate(model_names):
            for second in model_names[first_index + 1:]:
                intersection = int((masks[first] & masks[second]).sum())
                union = int((masks[first] | masks[second]).sum())
                value = float(intersection / union) if union else 1.0
                pairwise.setdefault(f"{first}__{second}", []).append(value)
        contact_sheet = np.concatenate([
            np.concatenate(overlay_row, axis=1),
            np.concatenate(mask_row, axis=1),
        ], axis=0)
        contact_path = case_dir / "model_comparison.png"
        Image.fromarray(contact_sheet).save(contact_path)
        case_record = {
            **sample,
            "models": results[case_index],
            "contact_sheet": str(contact_path.resolve()),
        }
        (case_dir / "result.json").write_text(
            json.dumps(case_record, ensure_ascii=False, indent=2) + "\n")
        per_case_records.append(case_record)

    model_summary = {}
    for model_name in model_names:
        records = [results[sample["case_index"]][model_name]
                   for sample in samples]
        supported = [item["path_support"] for item in records
                     if item["path_support"] is not None]
        model_summary[model_name] = {
            "cases": len(records),
            "nonempty_rate": float(np.mean([
                item["mask_fraction"] > 0 for item in records])),
            "point_available_rate": float(np.mean([
                item["point_xy"] is not None for item in records])),
            "point_on_raw_mask_rate": float(np.mean([
                item["point_on_raw_mask"] for item in records])),
            "point_on_targetable_mask_rate": float(np.mean([
                item["point_on_targetable_mask"] for item in records])),
            "all_ground_anchors_on_raw_mask_rate": float(np.mean([
                item["all_ground_anchors_on_raw_mask"] for item in records])),
            "all_ground_anchors_on_targetable_mask_rate": float(np.mean([
                item["all_ground_anchors_on_targetable_mask"]
                for item in records])),
            "mean_mask_fraction": float(np.mean([
                item["mask_fraction"] for item in records])),
            "mean_path_support": (float(np.mean(supported))
                                  if supported else None),
            "mean_upper_mask_share": float(np.mean([
                item["upper_mask_share"] for item in records])),
            "mean_point_boundary_margin_px": float(np.mean([
                item["point_boundary_margin_px"] for item in records])),
        }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "case_count": len(samples),
        "model_summary": model_summary,
        "mean_pairwise_mask_iou": {
            key: float(np.mean(values)) for key, values in pairwise.items()},
        "metric_caveat": (
            "R2R has no pixel ground labels. Path support is a hidden diagnostic "
            "proxy; final quality requires visual/manual review of contact sheets."),
        "coordinate_contract": (
            "all masks and points remain in native 256x256 RGB coordinates; "
            "point_on_raw_mask is checked after integer rounding"),
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(per_case_records, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    review_lines = [
        "# R2R ground segmentation manual review",
        "",
        ("Green is the raw model/ensemble ground-like mask; the red cross is "
         "the deterministic selected point; numbered white dots are up to six "
         "native-resolution mask anchors exposed to the VLM. Cyan is the "
         "hidden R2R reference path rendered only after inference."),
        "",
        "| Case | EP | State | View | Comparison | Review |",
        "|---:|---:|---:|---:|---|---|",
    ]
    for sample in samples:
        index = sample["case_index"]
        review_lines.append(
            f"| {index:03d} | {sample['episode_index']} | "
            f"{sample['reference_path_index']} | "
            f"{sample['evaluation_view_index']} | "
            f"[image](cases/case_{index:03d}/model_comparison.png) | "
            "[ ] correct / [ ] wrong |")
    (args.output_dir / "MANUAL_REVIEW.md").write_text(
        "\n".join(review_lines) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
