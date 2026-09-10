#!/usr/bin/env python3
"""Evaluate VLM point selection at one real demonstration state per R2R episode."""

import argparse
import collections
import gzip
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import habitat_sim
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from instruction_taxonomy import FORM_DEFINITIONS, decompose_by_definition
from habitat_point_navigation import (
    DEFAULT_MP3D_ROOT, make_sim, resolve_mp3d_scene, set_pose, yaw_from_coeffs,
)
from point_selectors import (
    angle_distance, observe_eight_rgbd, observe_six_rgbd, select_ground_point,
    targetable_ground_mask, wrap_angle, repair_selected_ground_point,
    rgb_lower_floor_prior,
)
from semantic_detector import (
    DinoSamDetector, GroundedSamDetector, DinoSamFloorSegmenter,
    extract_detection_queries,
)
from ground_segmentation_backends import (
    build_ground_segmenter, ground_segmenter_display_name,
)
from semantic_point_strategy import (
    apply_cross_view_semantic_policy, constrain_floor_candidates,
)
from path_projection import draw_reference_path_overlay, project_reference_path
from vlm_harness import NavigationVLMHarness, build_vlm_backend


DEFAULT_DATA = ROOT.parent / "3d_wm_vln/StreamVLN/data/datasets/r2r"
SIX_VIEW_OFFSETS = np.radians([0, 60, 120, 180, 240, 300])
EIGHT_VIEW_OFFSETS = np.radians([0, 45, 90, 135, 180, 225, 270, 315])


def _load_decomposition_stage(decomposition_root, episode_index, instruction):
    """Load the already-produced typed decomposition for an episode.

    First-point evaluation must not silently change the upstream instruction
    interpretation between repeats.  When an artifact is supplied, use its
    ordered ``selected_sub_instructions`` (the same contract consumed by the
    navigation runner), falling back to ``all_sub_instructions`` only when a
    selection list is absent.  The deterministic taxonomy remains the generic
    fallback for ordinary module sampling.
    """
    if decomposition_root is None:
        return None, None
    root = Path(decomposition_root)
    if root.is_file():
        # A frozen ten-EP manifest is a valid decomposition source.  This
        # keeps the benchmark's per-EP stage contract in one immutable file
        # while retaining the older episode-directory layout.
        payload = json.loads(root.read_text())
        rows = {
            int(row["episode_index"]): row
            for row in payload.get("episodes", [])
            if "episode_index" in row
        }
        row = rows.get(int(episode_index))
        if row is None:
            raise FileNotFoundError(
                f"decomposition manifest has no episode index {episode_index}: {root}")
        path = root
        decomposition = row.get("decomposition")
        if isinstance(decomposition, dict):
            decomposition = decomposition.get("stages")
        payload = {"stages": decomposition or row.get("stages") or []}
    else:
        path = root / f"episode_{int(episode_index):04d}" / \
            "instruction_decomposition.json"
        if not path.exists():
            raise FileNotFoundError(
                f"decomposition artifact missing for episode index {episode_index}: {path}")
        payload = json.loads(path.read_text())
    stages = (payload.get("selected_sub_instructions") or
              payload.get("all_sub_instructions") or
              payload.get("stages") or [])
    if not stages:
        raise ValueError(f"decomposition artifact has no stages: {path}")
    # Preserve the upstream VLM fields, while making the source explicit in
    # the audit trail.  Do not expose any path/state information to selection.
    normalized = []
    for index, value in enumerate(stages):
        stage = dict(value)
        stage["stage_id"] = index
        stage["sub_instruction_id"] = index
        stage.setdefault("source_clause", stage.get("navigation_instruction", instruction))
        stage.setdefault("landmark", "unspecified")
        stage["decomposition_artifact"] = str(path)
        normalized.append(stage)
    return normalized, str(path)


def load_scenarios(data_root, splits, seed, state_sampling="internal",
                   excluded_episode_indices=(), episode_indices=None,
                   decomposition_root=None):
    """Choose one real reference-path state per episode before model calls."""
    if state_sampling not in {"internal", "initial"}:
        raise ValueError(
            "state_sampling must be either 'internal' or 'initial'")
    rng = np.random.default_rng(seed)
    excluded_episode_indices = set(int(value) for value in
                                   excluded_episode_indices)
    requested_episode_indices = (None if episode_indices is None else
                                 set(int(value) for value in episode_indices))
    scenarios = []
    for split in splits:
        with gzip.open(data_root / split / f"{split}.json.gz", "rt") as handle:
            episodes = json.load(handle)["episodes"]
        for episode_index, episode in enumerate(episodes):
            if episode_index in excluded_episode_indices:
                continue
            if (requested_episode_indices is not None and
                    episode_index not in requested_episode_indices):
                continue
            path = episode.get("reference_path", [])
            instruction = episode["instruction"]["instruction_text"]
            stages, decomposition_source = _load_decomposition_stage(
                decomposition_root, episode_index, instruction)
            if stages is None:
                stages = decompose_by_definition(instruction)
                decomposition_source = "deterministic_definition_taxonomy"
            minimum_path_length = 3 if state_sampling == "internal" else 2
            if len(path) < minimum_path_length or not stages:
                continue
            if state_sampling == "initial":
                path_index = 0
                stage_index = 0
            else:
                path_index = int(rng.integers(1, len(path) - 1))
                stage_index = min(
                    len(stages) - 1,
                    int(math.floor(
                        path_index * len(stages) / (len(path) - 1))))
            scenarios.append({
                "split": split, "episode_index": episode_index,
                "episode_id": episode["episode_id"], "scene_id": episode["scene_id"],
                "start_rotation": episode["start_rotation"],
                "instruction": instruction,
                "reference_path": path, "stage": stages[stage_index],
                "stage_index": stage_index, "path_index": path_index,
                "all_stage_count": len(stages),
                "decomposition_source": decomposition_source,
            })
    return scenarios


def stratified_sample(scenarios, count, seed, max_scenes=5):
    rng = np.random.default_rng(seed)
    by_scene = collections.defaultdict(list)
    for scenario in scenarios:
        by_scene[scenario["scene_id"]].append(scenario)
    ranked_scenes = sorted(
        by_scene, key=lambda scene: (
            -len({item["stage"]["form"] for item in by_scene[scene]}),
            -len(by_scene[scene]), scene))
    selected_scenes = set(ranked_scenes[:max_scenes])
    scenarios = [scenario for scenario in scenarios if scenario["scene_id"] in selected_scenes]
    groups = collections.defaultdict(list)
    for scenario in scenarios:
        groups[scenario["stage"]["form"]].append(scenario)
    for values in groups.values():
        rng.shuffle(values)
    selected, forms = [], sorted(groups, key=lambda key: (-len(groups[key]), key))
    while len(selected) < count and forms:
        remaining = []
        for form in forms:
            if groups[form] and len(selected) < count:
                selected.append(groups[form].pop())
            if groups[form]:
                remaining.append(form)
        forms = remaining
    if len(selected) != count:
        raise RuntimeError(
            f"only {len(selected)} unique episodes available for requested {count}")
    return selected


def heading_between(start, end):
    delta = np.asarray(end) - np.asarray(start)
    return wrap_angle(math.atan2(-float(delta[0]), -float(delta[2])))


def pixel_to_world(point, depth, camera_position, camera_yaw, width, height, hfov=90):
    x, y = point
    z_depth = float(depth[int(np.clip(round(y), 0, height - 1)),
                          int(np.clip(round(x), 0, width - 1))])
    if not math.isfinite(z_depth) or z_depth <= 0:
        return None, z_depth
    focal = width / (2 * math.tan(math.radians(hfov) / 2))
    local_x = (x - (width - 1) / 2) * z_depth / focal
    local_y = -((y - (height - 1) / 2) * z_depth / focal)
    local_z = -z_depth
    cosine, sine = math.cos(camera_yaw), math.sin(camera_yaw)
    world = np.asarray(camera_position, np.float32).copy()
    world[0] += cosine * local_x + sine * local_z
    world[1] += local_y
    world[2] += -sine * local_x + cosine * local_z
    return world, z_depth


def distance_to_polyline_xz(point, path):
    p = np.asarray(point, np.float32)[[0, 2]]
    path = np.asarray(path, np.float32)[:, [0, 2]]
    best = math.inf
    for start, end in zip(path[:-1], path[1:]):
        segment = end - start
        scale = float(np.dot(p - start, segment) / max(np.dot(segment, segment), 1e-8))
        projection = start + np.clip(scale, 0, 1) * segment
        best = min(best, float(np.linalg.norm(p - projection)))
    return best


def angular_error(a, b):
    return abs(math.degrees(wrap_angle(a - b)))


def wilson_interval(successes, total, z=1.96):
    """Binomial Wilson score interval used for the primary direction accuracy."""
    if total <= 0:
        return [0.0, 0.0]
    probability = successes / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    margin = z / denominator * math.sqrt(
        probability * (1 - probability) / total + z * z / (4 * total * total))
    return [max(0.0, center - margin), min(1.0, center + margin)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--mp3d-root", type=Path, default=DEFAULT_MP3D_ROOT)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument(
        "--episode-indices", default=None,
        help=("optional explicit comma-separated episode indices; when set, "
              "sampling is bypassed and those real R2R states are used"))
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="repeat each explicit episode/state this many independent VLM calls")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--splits", default="val_unseen",
                        help="comma-separated R2R splits; default isolates val_unseen")
    parser.add_argument("--max-scenes", type=int, default=5,
                        help="limit scene reload cost while retaining form-stratified sampling")
    parser.add_argument(
        "--state-sampling", choices=["internal", "initial"],
        default="internal",
        help=("internal samples one non-endpoint demonstration state; initial "
              "tests the first instruction decision at the dataset start"))
    parser.add_argument(
        "--exclude-episode-indices", default="",
        help=("comma-separated dataset indices frozen out of sampling, e.g. "
              "a later end-to-end holdout"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vlm-backend", choices=["deepseek", "ollama", "heuristic"],
                        default="deepseek")
    parser.add_argument("--vlm-model", default=None)
    parser.add_argument(
        "--point-selection-prompt-version",
        choices=sorted(NavigationVLMHarness.POINT_SELECTION_PROMPT_VERSIONS),
        default="v19_pixel_ray_history_and_reverse_override")
    parser.add_argument(
        "--views", type=int, choices=[6, 8], default=8,
        help=("simultaneous RGB compass sectors used for point selection; "
              "hidden depth is queried only after the final pixel is frozen"))
    parser.add_argument(
        "--primary-heading-threshold-deg", type=float, default=45.0,
        help="selected 3D heading error must not exceed this frozen threshold")
    parser.add_argument(
        "--comparison-baseline", type=Path, default=None,
        help="require an identical pre-model sample manifest to this prior run")
    parser.add_argument(
        "--decomposition-root", type=Path, default=None,
        help=("optional root containing episode_%%04d/instruction_decomposition.json; "
              "use this to evaluate a previously fixed five-EP decomposition"))
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--deepseek-env", type=Path, default=ROOT / ".env.deepseek")
    parser.add_argument("--deepseek-base-url", default=None)
    parser.add_argument(
        "--semantic-detector", choices=["dino-sam", "grounded-sam", "none"],
        default="dino-sam",
        help=("instruction-level semantic detector; use none for the VLM-only "
              "ablation, while floor segmentation is controlled separately"))
    parser.add_argument(
        "--floor-segmenter",
        choices=["dense-majority", "grounded-sam", "dino-sam"],
        default="dense-majority",
        help=("ground mask backend; default is pixelwise >=2/3 dense ADE20K "
              "model agreement"))
    parser.add_argument(
        "--floor-box-threshold", type=float, default=0.28,
        help=("Grounded-SAM floor box confidence threshold; lower values "
              "recover weak floor views but may add noisy masks"))
    parser.add_argument(
        "--floor-text-threshold", type=float, default=0.22,
        help="Grounded-SAM floor token threshold")
    parser.add_argument(
        "--adaptive-floor-threshold", action="store_true",
        help=("with a low Grounded-SAM detector threshold, retain weak floor "
              "only for side/continuous/stair route forms; use the default "
              "threshold for ordinary/pass/portal stages"))
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/r2r_point_selection_100")
    args = parser.parse_args()
    splits = tuple(value.strip() for value in args.splits.split(",")
                   if value.strip())
    if not splits:
        parser.error("--splits must contain at least one split")
    if any(value not in {"train", "val_seen", "val_unseen", "test"}
           for value in splits):
        parser.error("--splits contains an unknown R2R split")
    if args.count < 1:
        parser.error("--count must be positive")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not 0.0 < args.floor_box_threshold <= 1.0:
        parser.error("--floor-box-threshold must be in (0,1]")
    if not 0.0 < args.floor_text_threshold <= 1.0:
        parser.error("--floor-text-threshold must be in (0,1]")
    explicit_episode_indices = None
    if args.episode_indices is not None:
        try:
            explicit_episode_indices = tuple(
                int(value.strip()) for value in args.episode_indices.split(",")
                if value.strip())
        except ValueError as exc:
            parser.error(f"--episode-indices must be comma-separated integers: {exc}")
        if not explicit_episode_indices:
            parser.error("--episode-indices must contain at least one index")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    visuals = args.output_dir / "visuals"; visuals.mkdir(exist_ok=True)
    cases_dir = args.output_dir / "cases"; cases_dir.mkdir(exist_ok=True)

    excluded_episode_indices = tuple(
        int(value) for value in args.exclude_episode_indices.split(",")
        if value.strip())
    loaded_scenarios = load_scenarios(
        args.data_root, splits, args.seed, args.state_sampling,
        excluded_episode_indices, explicit_episode_indices,
        args.decomposition_root)
    if explicit_episode_indices is not None:
        scenarios = loaded_scenarios
        missing = sorted(set(explicit_episode_indices) - {
            item["episode_index"] for item in scenarios})
        if missing:
            raise RuntimeError(
                f"explicit episode indices are not present in selected splits: {missing}")
        scenarios.sort(key=lambda scenario: scenario["episode_index"])
        scenarios = [dict(scenario, repeat_index=repeat)
                     for scenario in scenarios for repeat in range(args.repeats)]
        if len(scenarios) != len(explicit_episode_indices) * args.repeats:
            raise RuntimeError("explicit episode repeat expansion produced an unexpected count")
    else:
        if args.repeats != 1:
            parser.error("--repeats requires --episode-indices")
        scenarios = stratified_sample(
            loaded_scenarios, args.count, args.seed, args.max_scenes)
    scenarios.sort(key=lambda scenario: (scenario["scene_id"], scenario["episode_id"],
                                         scenario["stage_index"]))
    sample_manifest = [{
        "case_index": index,
        "repeat_index": int(scenario.get("repeat_index", 0)),
        "split": scenario["split"],
        "episode_index": scenario["episode_index"],
        "episode_id": scenario["episode_id"],
        "scene_id": scenario["scene_id"],
        "reference_path_index": scenario["path_index"],
        "reference_path_length": len(scenario["reference_path"]),
        "reference_position_xyz": scenario["reference_path"][
            scenario["path_index"]],
        "yaw_source": (
            "episode_start_rotation" if scenario["path_index"] == 0 else
            "derived_from_previous_reference_position"),
        "stage_index": scenario["stage_index"],
        "stage_form": scenario["stage"]["form"],
        "sub_instruction": scenario["stage"],
        "decomposition_source": scenario.get("decomposition_source"),
        "alignment_method": (
            "first_stage_at_dataset_start"
            if scenario["path_index"] == 0 else
            "fixed_monotonic_path_progress_floor"),
        "alignment_uncertain": bool(scenario["path_index"] != 0),
    } for index, scenario in enumerate(scenarios)]
    comparison_record = None
    if args.comparison_baseline is not None:
        baseline_dir = args.comparison_baseline.resolve()
        baseline_samples_path = baseline_dir / "sample_manifest.json"
        baseline_summary_path = baseline_dir / "summary.json"
        if not baseline_samples_path.exists() or not baseline_summary_path.exists():
            raise FileNotFoundError(
                f"comparison baseline lacks sample manifest/summary: {baseline_dir}")
        baseline_samples = json.loads(baseline_samples_path.read_text())
        identity_keys = (
            "case_index", "split", "episode_index", "episode_id", "scene_id",
            "reference_path_index", "stage_index", "stage_form")
        current_identities = [tuple(item[key] for key in identity_keys)
                              for item in sample_manifest]
        baseline_identities = [tuple(item[key] for key in identity_keys)
                               for item in baseline_samples]
        if current_identities != baseline_identities:
            raise RuntimeError(
                "comparison baseline uses a different episode/state/stage sample set")
        baseline_summary = json.loads(baseline_summary_path.read_text())
        comparison_record = {
            "baseline_output_dir": str(baseline_dir),
            "sample_manifest_identical": True,
            "sample_manifest_sha256": hashlib.sha256(
                baseline_samples_path.read_bytes()).hexdigest(),
            "baseline_prompt_version": baseline_summary.get(
                "point_selection_prompt_version", "v1_baseline"),
            "baseline_primary_accuracy": (
                baseline_summary.get("point_accuracy_within_45deg")
                if args.primary_heading_threshold_deg == 45.0 else
                baseline_summary.get("point_accuracy_primary",
                                     baseline_summary.get(
                                         "point_accuracy_forward_hemisphere"))),
            "primary_heading_threshold_deg": (
                args.primary_heading_threshold_deg),
        }
    manifest = {
        "schema_version": 1,
        "rule_file": str((ROOT / "project_rulle.md").resolve()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_scope": "module",
        "target_modules": ["vlm_point_selection"],
        "required_upstream_modules": [
            "real_r2r_state_initialization",
            ("eight_view_rgb" if args.views == 8 else "six_view_rgb"),
            ("dense_majority_ground_segmentation"
             if args.floor_segmenter == "dense-majority" else
             f"{args.floor_segmenter.replace('-', '_')}_ground_segmentation"),
            "fixed_definition_sub_instruction_alignment",
        ],
        "upstream_artifact_source": (
            "fixed per-episode instruction_decomposition.json"
            if args.decomposition_root is not None else None),
        "decomposition_root": (
            str(args.decomposition_root.resolve())
            if args.decomposition_root is not None else None),
        "not_run_modules": [
            "point_navigation_executor", "navigation_graph_node_and_edge",
            "sub_instruction_node_classification", "physical_node_backtracking",
            "loop_closure",
        ],
        "selection_fixed_before_model_run": True,
        "future_reference_information_exposed_to_models": False,
        "fake_demonstration_action_history_exposed_to_models": False,
        "point_selection_input_policy": (
            "rgb_only_depth_prohibited"),
        "depth_usage": (
            "only after final pixel selection for camera backprojection, "
            "navmesh projection, and hidden scoring"),
        "sample_unit": (
            ("one initial reference_path state per explicit R2R episode, repeated"
             if args.state_sampling == "initial" else
             "one deterministic internal reference_path state per explicit R2R episode, repeated")
            if explicit_episode_indices is not None else
            "one unique R2R episode"),
        "requested_episode_count": (
            len(explicit_episode_indices)
            if explicit_episode_indices is not None else args.count),
        "explicit_episode_indices": list(explicit_episode_indices)
        if explicit_episode_indices is not None else None,
        "repeats_per_episode": args.repeats,
        "splits": list(splits),
        "seed": args.seed,
        "max_scenes": args.max_scenes,
        "state_sampling": (
            "dataset reference_path start state"
            if args.state_sampling == "initial" else
            "uniform internal reference_path index per episode"),
        "excluded_episode_indices": list(excluded_episode_indices),
        "stage_alignment": (
            "floor(path_index * num_sub_instructions / "
            "(reference_path_length - 1))"),
        "primary_metric": (
            "selected navigable 3D point heading error <= "
            f"{args.primary_heading_threshold_deg:g} degrees"),
        "failure_denominator_policy": "all preselected episodes",
        "samples": sample_manifest,
        "point_selection_prompt_version": args.point_selection_prompt_version,
        "comparison": comparison_record,
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "sample_manifest.json").write_text(
        json.dumps(sample_manifest, ensure_ascii=False, indent=2) + "\n")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    backend = build_vlm_backend(
        args.vlm_backend, args.vlm_model, args.ollama_host, 180,
        deepseek_env_file=args.deepseek_env,
        deepseek_base_url=args.deepseek_base_url)
    args.vlm_model = getattr(backend, "model", args.vlm_model or args.vlm_backend)
    harness = NavigationVLMHarness(
        backend, args.output_dir / "vlm_calls.json", retries=2,
        point_selection_prompt_version=args.point_selection_prompt_version)
    segmenter, floor_detector = build_ground_segmenter(
        args.floor_segmenter, args.device,
        box_threshold=args.floor_box_threshold,
        text_threshold=args.floor_text_threshold)
    if args.semantic_detector == "grounded-sam":
        semantic_detector = (floor_detector
                             if isinstance(floor_detector, GroundedSamDetector)
                             else GroundedSamDetector(args.device))
    elif args.semantic_detector == "dino-sam":
        semantic_detector = (floor_detector
                             if isinstance(floor_detector, DinoSamDetector)
                             else DinoSamDetector(args.device))
    else:
        semantic_detector = None
    width, height = 320, 240
    results, sim, active_scene = [], None, None
    video_path = args.output_dir / "point_selection_cases.mp4"
    video_columns = 4 if args.views == 8 else 3
    video_rows = 3 if args.views == 8 else 2
    video = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0,
        (width * video_columns, height * video_rows))
    if not video.isOpened():
        raise RuntimeError(f"failed to open video writer: {video_path}")

    for scenario_index, scenario in enumerate(scenarios):
        case_dir = cases_dir / f"case_{scenario_index:03d}"
        views_dir = case_dir / (
            "initial_eight_views" if args.views == 8 else
            "initial_six_views")
        depths_dir = case_dir / "initial_depths"
        masks_dir = case_dir / "ground_masks"
        views_dir.mkdir(parents=True, exist_ok=True)
        depths_dir.mkdir(exist_ok=True)
        masks_dir.mkdir(exist_ok=True)
        scene = resolve_mp3d_scene(scenario["scene_id"], args.mp3d_root)
        if scene != active_scene:
            if sim is not None:
                sim.close()
            sim = make_sim(scene, width, height)
            active_scene = scene
        path = np.asarray(scenario["reference_path"], np.float32)
        path_index = scenario["path_index"]
        position, next_position = path[path_index], path[path_index + 1]
        yaw = (heading_between(path[path_index - 1], position) if path_index > 0
               else yaw_from_coeffs(scenario["start_rotation"]))
        if not sim.pathfinder.is_navigable(position):
            result = {
                "repeat_index": int(scenario.get("repeat_index", 0)),
                **{key: scenario[key] for key in (
                    "split", "episode_index", "episode_id", "instruction",
                    "stage_index", "path_index")},
                "stage": scenario["stage"], "scene_id": scenario["scene_id"],
                "reference_position_xyz": position.tolist(),
                "reference_yaw_rad": yaw,
                "reference_yaw_source": (
                    "episode_start_rotation" if path_index == 0 else
                    "derived_from_previous_reference_position"),
                "chosen_view": None, "selected_pixel": None,
                "selected_world": None, "selected_depth_m": None,
                "selection": None, "distance_to_future_demo_path_m": math.inf,
                "heading_error_to_demo_deg": math.inf,
                "backtracking_selection": False,
                "point_on_ground": False, "valid_navmesh_projection": False,
                "selected_point_reachable": False,
                "selected_point_initial_geodesic_m": None,
                "error": "reference_path state is not navigable",
                "candidate_audit": [],
            }
            results.append(result)
            (case_dir / "case_result.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            failure_frame = np.zeros(
                (height * video_rows, width * video_columns, 3), np.uint8)
            cv2.putText(failure_frame, f"case {scenario_index:03d} INVALID STATE",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)
            video.write(failure_frame)
            print(f"[{scenario_index + 1}/{len(scenarios)}] INVALID STATE", flush=True)
            continue
        set_pose(sim, position, yaw)
        if args.views == 8:
            offsets = EIGHT_VIEW_OFFSETS
            rgbs, depths = observe_eight_rgbd(sim)
        else:
            offsets = SIX_VIEW_OFFSETS
            rgbs, depths = observe_six_rgbd(sim)
        for view_index, (rgb, depth) in enumerate(zip(rgbs, depths)):
            Image.fromarray(rgb).save(views_dir / f"view_{view_index}.jpg")
            np.save(depths_dir / f"view_{view_index}.npy",
                    np.asarray(depth, np.float32))
        camera_position = position + np.array([0.0, 1.25, 0.0], np.float32)
        # RGB-only point selection: depth is retained only for projection after
        # the VLM freezes its final pixel.  With the optional adaptive policy,
        # weak Grounded-SAM floor proposals are retained only for route forms
        # that genuinely need side/rear/continuous support; ordinary/pass
        # stages keep the original high-confidence floor mask.
        if (args.adaptive_floor_threshold and
                args.floor_segmenter == "grounded-sam"):
            low_floor_forms = {
                "CIRCUMNAVIGATE", "APPROACH_LANDMARK",
                "TRAVERSE_PORTAL_REGION", "FOLLOW_PATH_BOUNDARY",
                "VERTICAL_DOWN",
            }
            segmenter.min_detection_score = (
                max(0.28, args.floor_box_threshold)
                if scenario["stage"].get("form") not in low_floor_forms
                else 0.0)
        segmentations = segmenter.batch(rgbs)
        strict_ground = bool(getattr(segmenter, "strict_ground_mask", False))
        detection_queries = extract_detection_queries(scenario["stage"])
        back_yaw = heading_between(position, path[path_index - 1]) if path_index > 0 else None
        candidates = []
        projection_depths = {}

        def build_candidate(view_index, offset, rgb, depth, segmentation,
                            refined=False):
            # This side table is outside the VLM candidate object. It is read
            # only after select_ground_target has frozen a pixel and view.
            projection_depths[int(view_index)] = np.asarray(depth, np.float32)
            mask, ground_detections = segmentation
            Image.fromarray(mask.astype(np.uint8) * 255).save(
                masks_dir / f"view_{view_index}.png")
            # Generic object segmentation remains disabled. Instruction-specific
            # portal/landmark detection below is still allowed for spatial grounding.
            object_mask = np.zeros_like(mask)
            # V21 uses Grounded-SAM object evidence only for the three
            # relation forms where object identity disambiguates a route
            # (around/behind/through).  Other forms keep the proven RGB-only
            # floor candidate, preventing noisy whole-panorama labels from
            # hijacking stairs, exits, or plain forward motion.
            targeted_detection_forms = {
                "CIRCUMNAVIGATE", "APPROACH_LANDMARK", "TURN_TO_LANDMARK",
                "TRAVERSE_PORTAL_REGION",
            }
            detection_enabled = (
                semantic_detector is not None and detection_queries and
                args.point_selection_prompt_version in {
                    "v21_relation_aware_route_review",
                    "v22_task30_route_anchor",
                    "v23_landmark_turn_stair_ray",
                    "v31_circumnavigate_forward_competitor"} and
                scenario["stage"].get("form") in targeted_detection_forms)
            detections = (semantic_detector.detect(rgb, detection_queries)
                          if detection_enabled else [])
            target_mask, strategy_application = constrain_floor_candidates(
                scenario["stage"], targetable_ground_mask(mask), None,
                detections, object_mask)
            # A relation route can be visible in RGB while Grounded-SAM misses
            # every floor pixel in one compass sector (typically a side/rear
            # transition).  Expose only the conservative lower/interior RGB
            # prior in that empty sector; it is still marked in the audit and
            # never supplies depth or semantic direction to the VLM.
            if (not strict_ground and not target_mask.any() and
                    scenario["stage"].get("form") == "CIRCUMNAVIGATE"):
                target_mask = targetable_ground_mask(
                    rgb_lower_floor_prior(mask.shape))
                strategy_application = dict(strategy_application)
                strategy_application["rgb_floor_prior_fallback"] = True
                strategy_application["fallback"] = "empty_ground_mask_rgb_prior"
            if strict_ground:
                target_mask = np.asarray(target_mask, bool) & np.asarray(mask, bool)
            point, point_score = select_ground_point(target_mask)
            view_yaw = wrap_angle(yaw + float(offset))
            back_delta = angle_distance(view_yaw, back_yaw) if back_yaw is not None else math.pi
            return {
                "view_index": view_index,
                "yaw": view_yaw, "relative_yaw_rad": float(offset), "rgb": rgb,
                "mask": mask, "target_mask": target_mask, "point": point,
                "strict_ground_mask": strict_ground,
                "ground_fraction": float(mask.mean()), "point_score": point_score,
                "backtrack_delta_rad": back_delta,
                "excluded": bool(back_yaw is not None and back_delta < math.radians(50)),
                "hard_excluded": False,
                "incoming_back_relative_yaw_rad": (
                    float(wrap_angle(back_yaw - yaw))
                    if back_yaw is not None else None),
                "backtrack_exclusion_rad": float(math.radians(50)),
                "blocked_relative_yaws_rad": [],
                "blocked_direction_exclusion_rad": float(math.radians(50)),
                "detection_queries": detection_queries,
                "semantic_detections": detections,
                "detection_records": [
                    item.rgb_prompt_record() for item in detections],
                "small_seg_object_mask": object_mask,
                "small_seg_objects": [],
                "ground_detection_records": [
                    item.rgb_prompt_record() for item in ground_detections],
                "strategy_application": strategy_application,
                "is_refined_view": bool(refined),
            }

        for view_index, (offset, rgb, depth, segmentation) in enumerate(
                zip(offsets, rgbs, depths, segmentations)):
            candidates.append(build_candidate(
                view_index, offset, rgb, depth, segmentation))
        cross_view_policy = apply_cross_view_semantic_policy(
            scenario["stage"], candidates,
            policy=harness.point_selection_candidate_policy)

        def refinement_provider(relative_yaw_rad):
            refined_index = len(candidates)
            set_pose(sim, position, wrap_angle(yaw + relative_yaw_rad))
            observations = sim.get_sensor_observations()
            refined_rgb = observations["rgb"][..., :3]
            refined_depth = observations["depth"]
            set_pose(sim, position, yaw)
            Image.fromarray(refined_rgb).save(
                views_dir / f"view_{refined_index}_refined.jpg")
            np.save(
                depths_dir / f"view_{refined_index}_refined.npy",
                np.asarray(refined_depth, np.float32))
            return build_candidate(
                refined_index, relative_yaw_rad, refined_rgb, refined_depth,
                segmenter(refined_rgb), refined=True)

        vlm_call_index = len(harness.calls)
        try:
            chosen_index, point, selection = harness.select_ground_target(
                scenario["stage"], candidates, [],
                refinement_provider=(
                    refinement_provider if args.views == 8 else None))
            if args.point_selection_prompt_version in {
                "v20_first_step_route_guard",
                "v21_relation_aware_route_review",
                "v22_task30_route_anchor",
                    "v23_landmark_turn_stair_ray",
                    "v31_circumnavigate_forward_competitor",
                    "v32_first_stage_shallow_route",
                    "v33_stop_relation_near_side"}:
                # Depth is reattached only after the VLM has frozen its RGB
                # view/pixel.  The repair boundary therefore remains outside
                # the model-facing candidate objects.
                for candidate_index, candidate in enumerate(candidates):
                    if candidate_index in projection_depths:
                        candidate["depth"] = projection_depths[candidate_index]
                repaired_index, repaired_point, repair = (
                    repair_selected_ground_point(
                        sim, candidates, chosen_index, point, position,
                        selection=selection,
                        max_view_delta_rad=math.radians(float(
                            selection.get(
                                "postselection_repair_max_view_delta_deg",
                                45.0)))))
                if repair.get("status") != "selected_point_reachable":
                    selection = dict(selection or {})
                    selection["postselection_repair"] = repair
                    chosen_index, point = repaired_index, repaired_point
            candidate = candidates[chosen_index]
            if strict_ground:
                selection_mask = np.asarray(candidate["target_mask"], bool)
                xy = np.round(np.asarray(point, np.float32)).astype(int)
                xy[0] = np.clip(xy[0], 0, selection_mask.shape[1] - 1)
                xy[1] = np.clip(xy[1], 0, selection_mask.shape[0] - 1)
                if not selection_mask[xy[1], xy[0]]:
                    ys, xs = np.nonzero(selection_mask)
                    if not len(xs):
                        raise RuntimeError(
                            "selected view has no dense-majority ground anchor")
                    nearest = np.argmin(
                        (xs.astype(np.float32) - float(point[0])) ** 2 +
                        (ys.astype(np.float32) - float(point[1])) ** 2)
                    point = np.array(
                        [float(xs[nearest]), float(ys[nearest])], np.float32)
                    selection = dict(selection or {})
                    selection["strict_ground_snap"] = {
                        "applied": True, "final_xy": point.tolist()}
            world_point, selected_depth = pixel_to_world(
                point, projection_depths[chosen_index], camera_position,
                candidate["yaw"], width, height)
            if world_point is None:
                raise RuntimeError("selected pixel has invalid depth")
            snapped = np.asarray(sim.pathfinder.snap_point(world_point), np.float32)
            selected_world = snapped if np.isfinite(snapped).all() else world_point
            future_path = path[path_index:]
            distance = distance_to_polyline_xz(selected_world, future_path)
            selected_heading = heading_between(position, selected_world)
            demo_heading = heading_between(position, next_position)
            heading_error = angular_error(selected_heading, demo_heading)
            backtracking = (back_yaw is not None and
                            angle_distance(selected_heading, back_yaw) < math.radians(50))
            point_on_ground = bool(selection.get("requested_on_ground"))
            valid_navmesh_projection = bool(np.isfinite(snapped).all())
            selected_point_reachable = False
            selected_point_initial_geodesic_m = None
            if valid_navmesh_projection:
                reach_path = habitat_sim.ShortestPath()
                reach_path.requested_start = np.asarray(position, np.float32)
                reach_path.requested_end = selected_world
                selected_point_reachable = bool(
                    sim.pathfinder.find_path(reach_path))
                if selected_point_reachable:
                    selected_point_initial_geodesic_m = float(
                        reach_path.geodesic_distance)
            error = None
        except Exception as exc:
            chosen_index, point, selection = None, None, None
            selected_world, selected_depth = None, None
            distance, heading_error, backtracking = math.inf, math.inf, False
            point_on_ground, valid_navmesh_projection = False, False
            selected_point_reachable = False
            selected_point_initial_geodesic_m = None
            error = str(exc)

        # This is an evaluation-only overlay.  The reference path is not
        # included in any VLM prompt or candidate audit used for selection.
        path_projection_by_view = {
            int(index): project_reference_path(
                path, range(path_index, len(path)), camera_position,
                candidate["yaw"], width, height)
            for index, candidate in enumerate(candidates)
        }

        result = {
            "repeat_index": int(scenario.get("repeat_index", 0)),
            **{key: scenario[key] for key in (
                "split", "episode_index", "episode_id", "instruction",
                "stage_index", "path_index")},
            "stage": scenario["stage"], "scene_id": scenario["scene_id"],
            "reference_position_xyz": position.tolist(),
            "reference_yaw_rad": yaw,
            "reference_yaw_source": (
                "episode_start_rotation" if path_index == 0 else
                "derived_from_previous_reference_position"),
            "incoming_reference_position_xyz": (
                path[path_index - 1].tolist() if path_index > 0 else None),
            "stage_alignment_method": (
                "first_stage_at_dataset_start" if path_index == 0 else
                "fixed_monotonic_path_progress_floor"),
            "stage_alignment_uncertain": bool(path_index != 0),
            "chosen_view": chosen_index, "selected_pixel": point.tolist() if point is not None else None,
            "selected_world": selected_world.tolist() if selected_world is not None else None,
            "selected_depth_m": selected_depth, "selection": selection,
            "distance_to_future_demo_path_m": distance,
            "heading_error_to_demo_deg": heading_error,
            "backtracking_selection": backtracking,
            "point_on_ground": point_on_ground,
            "valid_navmesh_projection": valid_navmesh_projection,
            "selected_point_reachable": selected_point_reachable,
            "selected_point_initial_geodesic_m": (
                selected_point_initial_geodesic_m),
            "vlm_call_index": vlm_call_index if error is None else None,
            "model_action_history": [],
            "future_reference_information_exposed_to_model": False,
                "cross_view_semantic_policy": cross_view_policy,
                "error": error,
                "reference_path_projection": path_projection_by_view,
                "candidate_audit": [{
                    "detection_queries": item["detection_queries"],
                    "detections": item["detection_records"],
                    "ground_detections": item["ground_detection_records"],
                    "small_seg_objects": item["small_seg_objects"],
                    "strategy_application": item["strategy_application"],
                    "reference_path_projection": path_projection_by_view.get(
                        int(item["view_index"]), {"vertices": [], "segments": []}),
                } for item in candidates],
            }
        results.append(result)
        (case_dir / "case_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        decision_views = []
        for index, candidate in enumerate(candidates):
            rendered_view = harness._overlay(
                candidate["rgb"], candidate["target_mask"], index,
                bool(candidate["excluded"]),
                candidate.get("semantic_detections"),
                candidate.get("small_seg_object_mask"),
                candidate.get("ground_anchors", []))
            rendered_view = draw_reference_path_overlay(
                rendered_view,
                path_projection_by_view.get(index, {"vertices": [], "segments": []}),
                selected_point=(point if index == chosen_index and point is not None else None),
                selected=bool(index == chosen_index and point is not None),
                next_path_index=path_index + 1)
            if index == chosen_index and point is not None:
                cv2.drawMarker(
                    rendered_view, tuple(np.round(point).astype(int)),
                    (255, 0, 0), cv2.MARKER_CROSS, 22, 3)
                cv2.putText(rendered_view, "FINAL", (8, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
            decision_views.append(rendered_view)
        decision_sheet = harness._point_selection_contact_sheet(decision_views)
        expected_height = height * video_rows
        expected_width = width * video_columns
        if decision_sheet.shape[0] < expected_height:
            decision_sheet = np.concatenate([
                decision_sheet,
                np.zeros((expected_height - decision_sheet.shape[0],
                          decision_sheet.shape[1], 3), np.uint8),
            ], axis=0)
        if decision_sheet.shape[:2] != (expected_height, expected_width):
            raise RuntimeError(
                "decision sheet does not match the frozen video layout: "
                f"{decision_sheet.shape[:2]} vs "
                f"{(expected_height, expected_width)}")
        status = (f"case {scenario_index:03d} ep {scenario['episode_id']} "
                  f"{scenario['stage']['form']} error={heading_error:.1f}deg"
                  if error is None else
                  f"case {scenario_index:03d} ep {scenario['episode_id']} FAILED")
        cv2.rectangle(decision_sheet, (0, 0), (decision_sheet.shape[1], 28),
                      (0, 0, 0), -1)
        cv2.putText(decision_sheet, status, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        Image.fromarray(decision_sheet).save(
            visuals / f"case_{scenario_index:03d}.jpg")
        Image.fromarray(decision_sheet).save(case_dir / "decision.jpg")
        for _ in range(3):
            video.write(cv2.cvtColor(decision_sheet, cv2.COLOR_RGB2BGR))

        if error is None and vlm_call_index < len(harness.calls):
            call = harness.calls[vlm_call_index]
            (case_dir / "vlm_prompt.txt").write_text(call["prompt"] + "\n")
            (case_dir / "vlm_response.json").write_text(
                json.dumps(call, ensure_ascii=False, indent=2) + "\n")
            if call.get("image_paths"):
                source = args.output_dir / call["image_paths"][0]
                if source.exists():
                    shutil.copy2(source, case_dir / "vlm_contact_sheet.jpg")
        print(f"[{scenario_index + 1}/{len(scenarios)}] {scenario['stage']['form']} "
              f"distance={distance:.2f} heading={heading_error:.1f} error={error}", flush=True)

    if sim is not None:
        sim.close()
    video.release()
    valid = [result for result in results if result["error"] is None]
    distances = [result["distance_to_future_demo_path_m"] for result in valid]
    headings = [result["heading_error_to_demo_deg"] for result in valid]
    evaluated = len(results)
    heading_correct = {
        threshold: sum(
            result["error"] is None and
            result.get("selected_point_reachable", False) and
            result["heading_error_to_demo_deg"] <= threshold
            for result in results)
        for threshold in (30, 45, 60)
    }
    primary_correct = sum(
        result["error"] is None and
        result.get("selected_point_reachable", False) and
        result["heading_error_to_demo_deg"] <=
        args.primary_heading_threshold_deg
        for result in results)
    forward_correct = sum(
        result["error"] is None and
        result.get("selected_point_reachable", False) and
        result["heading_error_to_demo_deg"] < 90
        for result in results)
    summary = {
        "test_scope": "module", "target_module": "vlm_point_selection",
        "sample_unit": (
            ("one initial reference_path state per explicit R2R episode, repeated"
             if args.state_sampling == "initial" else
             "one deterministic internal reference_path state per explicit R2R episode, repeated")
            if explicit_episode_indices is not None else
            "one unique R2R episode"),
        "requested_episodes": (
            len(explicit_episode_indices)
            if explicit_episode_indices is not None else args.count),
        "evaluated_episodes": len(results),
        "unique_episode_count": len({
            (result["split"], str(result["episode_id"])) for result in results}),
        "valid_selections": len(valid), "failed_selections": len(results) - len(valid),
        "selection_valid_rate": len(valid) / max(evaluated, 1),
        "primary_accuracy_definition": (
            "selected 3D ground point must have a valid shortest path and "
            "heading error <= "
            f"{args.primary_heading_threshold_deg:g} degrees from the next R2R "
            "demonstration segment; failed selections count as incorrect"),
        "primary_heading_threshold_deg": args.primary_heading_threshold_deg,
        "point_accuracy_primary": primary_correct / max(evaluated, 1),
        "point_accuracy_primary_count": primary_correct,
        "point_accuracy_primary_wilson_95ci": wilson_interval(
            primary_correct, evaluated),
        "point_accuracy_forward_hemisphere": forward_correct / max(evaluated, 1),
        "point_accuracy_forward_hemisphere_count": forward_correct,
        "point_accuracy_forward_hemisphere_wilson_95ci": wilson_interval(
            forward_correct, evaluated),
        "point_accuracy_within_30deg": heading_correct[30] / max(evaluated, 1),
        "point_accuracy_within_30deg_count": heading_correct[30],
        "point_accuracy_within_30deg_wilson_95ci": wilson_interval(
            heading_correct[30], evaluated),
        "point_accuracy_within_45deg": heading_correct[45] / max(evaluated, 1),
        "point_accuracy_within_45deg_count": heading_correct[45],
        "point_accuracy_within_45deg_wilson_95ci": wilson_interval(
            heading_correct[45], evaluated),
        "point_accuracy_within_60deg": heading_correct[60] / max(evaluated, 1),
        "point_accuracy_within_60deg_count": heading_correct[60],
        "point_accuracy_within_60deg_wilson_95ci": wilson_interval(
            heading_correct[60], evaluated),
        "within_0_5m": sum(value <= 0.5 for value in distances) / max(len(valid), 1),
        "within_1m": sum(value <= 1.0 for value in distances) / max(len(valid), 1),
        "within_2m": sum(value <= 2.0 for value in distances) / max(len(valid), 1),
        "median_distance_m": float(np.median(distances)) if distances else math.inf,
        "mean_distance_m": float(np.mean(distances)) if distances else math.inf,
        "heading_within_30deg_valid_only": sum(value <= 30 for value in headings) / max(len(valid), 1),
        "heading_within_45deg_valid_only": sum(value <= 45 for value in headings) / max(len(valid), 1),
        "heading_within_60deg_valid_only": sum(value <= 60 for value in headings) / max(len(valid), 1),
        "heading_in_forward_hemisphere": sum(value < 90 for value in headings) /
        max(len(valid), 1),
        "median_heading_error_deg": float(np.median(headings)) if headings else math.inf,
        "backtracking_rate": sum(result["backtracking_selection"] for result in valid) / max(len(valid), 1),
        "point_on_ground_count": sum(
            result.get("point_on_ground", False) for result in results),
        "valid_navmesh_projection_count": sum(
            result.get("valid_navmesh_projection", False) for result in results),
        "selected_point_reachable_count": sum(
            result.get("selected_point_reachable", False)
            for result in results),
        "selected_point_reach_rate": sum(
            result.get("selected_point_reachable", False)
            for result in results) / max(evaluated, 1),
        "wrong_direction_count": evaluated - forward_correct,
        "primary_direction_error_count": evaluated - primary_correct,
        "failure_reasons": dict(collections.Counter(
            result["error"] for result in results if result["error"])),
        "prompt_leakage_audit": {
            "demonstration_progress_absent": all(
                "demonstration_progress" not in call.get("prompt", "")
                for call in harness.calls),
            "path_index_absent": all(
                "path_index" not in call.get("prompt", "")
                for call in harness.calls),
            "model_action_history_always_empty": all(
                result.get("model_action_history") == [] for result in results),
        },
        "form_counts": dict(collections.Counter(result["stage"]["form"] for result in results)),
        "definitions": FORM_DEFINITIONS,
        "floor_segmenter": ground_segmenter_display_name(
            args.floor_segmenter),
        "strict_ground_selection": bool(
            getattr(segmenter, "strict_ground_mask", False)),
        "ground_queries": getattr(segmenter, "queries", None),
        "generic_object_segmentation": False,
        "vlm_backend": args.vlm_backend,
        "vlm_model": args.vlm_model,
        "point_selection_prompt_version": args.point_selection_prompt_version,
        "explicit_episode_indices": list(explicit_episode_indices)
        if explicit_episode_indices is not None else None,
        "repeats_per_episode": args.repeats,
        "comparison": comparison_record,
        "video": {
            "path": str(video_path),
            "frame_size_wh": [width * video_columns, height * video_rows],
            "fps": 5, "frames_per_case": 3,
        },
        "path_projection_visualization": {
            "enabled": True,
            "reference": "reference_path[path_index:] projected after VLM selection",
            "ground_truth_color_rgb": [0, 230, 255],
            "selected_point_color_rgb": [255, 30, 30],
            "path_is_not_model_input": True,
        },
    }
    results_path = args.output_dir / "results.jsonl"
    summary_path = args.output_dir / "summary.json"
    results_path.write_text("".join(
        json.dumps(result, ensure_ascii=False) + "\n" for result in results))
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    by_form = {}
    for form in sorted({result["stage"]["form"] for result in results}):
        rows = [result for result in results if result["stage"]["form"] == form]
        correct = sum(
            row["error"] is None and
            row.get("selected_point_reachable", False) and
            row["heading_error_to_demo_deg"] < 90
            for row in rows)
        primary_form_correct = sum(
            row["error"] is None and
            row.get("selected_point_reachable", False) and
            row["heading_error_to_demo_deg"] <=
            args.primary_heading_threshold_deg
            for row in rows)
        by_form[form] = {
            "episodes": len(rows),
            "valid": sum(row["error"] is None for row in rows),
            "forward_hemisphere_correct": correct,
            "forward_hemisphere_accuracy_all_preselected": correct / len(rows),
            "primary_heading_threshold_deg": args.primary_heading_threshold_deg,
            "primary_direction_correct": primary_form_correct,
            "primary_direction_accuracy_all_preselected": (
                primary_form_correct / len(rows)),
            "median_heading_error_valid_deg": float(np.median([
                row["heading_error_to_demo_deg"] for row in rows
                if row["error"] is None])) if any(
                    row["error"] is None for row in rows) else math.inf,
        }
    form_summary = {
        "denominator_policy": "all preselected episodes within each form",
        "by_form": by_form,
    }
    (args.output_dir / "form_summary.json").write_text(
        json.dumps(form_summary, ensure_ascii=False, indent=2) + "\n")

    module_results = {
        "test_scope": "module",
        "target_module": "vlm_point_selection",
        "episode_count": evaluated,
        "unique_episode_count": summary["unique_episode_count"],
        "ground_candidate_valid_count": sum(
            any(item.get("ground_detections") for item in row["candidate_audit"])
            for row in results),
        "point_on_ground_count": summary["point_on_ground_count"],
        "point_depth_and_navmesh_valid_count": summary[
            "valid_navmesh_projection_count"],
        "selected_point_reachable_count": summary[
            "selected_point_reachable_count"],
        "primary_heading_threshold_deg": args.primary_heading_threshold_deg,
        "point_direction_correct_count": primary_correct,
        "point_direction_accuracy": summary["point_accuracy_primary"],
        "point_direction_wilson_95ci": summary[
            "point_accuracy_primary_wilson_95ci"],
        "within_30deg_count": heading_correct[30],
        "within_45deg_count": heading_correct[45],
        "within_60deg_count": heading_correct[60],
        "failed_selection_count": evaluated - len(valid),
        "not_run_modules": manifest["not_run_modules"],
    }
    (args.output_dir / "module_results.json").write_text(
        json.dumps(module_results, ensure_ascii=False, indent=2) + "\n")

    artifact_hashes = {}
    for relative in (
            "sample_manifest.json", "vlm_calls.json", "vlm_calls_attempts.json",
            "results.jsonl", "summary.json", "form_summary.json",
            "module_results.json", "point_selection_cases.mp4"):
        path = args.output_dir / relative
        if path.exists():
            artifact_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.update({
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "evaluated_episode_count": evaluated,
        "unique_episode_count": summary["unique_episode_count"],
        "vlm_backend": args.vlm_backend,
        "vlm_model": args.vlm_model,
        "prompt_leakage_audit": summary["prompt_leakage_audit"],
        "artifact_sha256": artifact_hashes,
    })
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
