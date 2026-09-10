#!/usr/bin/env python3
"""Habitat integration for external selection and point-navigation execution.

Ground semantics propose a point cluster, causal TAPIR maintains it, a changing
crop around the tracked cluster is the image-goal, and a lightweight visual-nav
policy plus pixel-space visual servoing produces safe navmesh motion.
"""
# ruff: noqa: E402 -- local TAPIR/scripts paths must be registered before imports.

import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import habitat_sim
import numpy as np
import torch
import tree
from habitat_sim.utils.common import quat_from_angle_axis, quat_from_coeffs, quat_rotate_vector
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models/tapnet"))
sys.path.insert(0, str(ROOT / "scripts"))

from tapnet.torch import tapir_model
from image_goal_policy import build_policy, predict
from semantic_detector import (
    DinoSamDetector, DinoSamFloorSegmenter, GroundedSamDetector,
)
from ground_segmentation_backends import (
    build_ground_segmenter, ground_segmenter_display_name,
)
from instruction_decomposer import InstructionDecomposer, SubInstruction
from instruction_completion_judge import (
    NodeTransitionInstructionCompletionJudge, UNKNOWN,
)
from breadth_first_exploration import BreadthFirstExplorationStrategy
from instruction_sequence_exploration import InstructionSequenceExplorationStrategy
from rgb_only_instruction_sequence import (
    RGBOnlyInstructionSequenceExplorationStrategy,
)
from rgb_only_runtime import (
    RGBOnlyEvaluationVideoBridge, RGBOnlyPolicySimulator,
)
from navigation_graph_memory import (
    DinoSamEnvironmentSemanticExtractor, NavigationGraphMemory,
)
from node_backtracking import (
    BACKTRACK_PLANNER_PROFILES, NodeBacktrackingController,
)
from point_navigation_executor import (
    POINT_NAVIGATION_ARRIVED, STOP_ARRIVAL_REASON,
    TRACKING_CLUSTER_PROFILES, PointNavigationExecutor, PointNavigationRequest,
    execute_point_navigation,
)
from point_selectors import (
    InstructionVLMPointSelector, PointSelectionRequest,
    RandomExplorationPointSelector, observe_eight_rgb, observe_six_rgb,
    observe_six_rgbd,
    pixel_ground_to_world, wrap_angle,
)
from path_projection import draw_reference_path_overlay, project_reference_path
from r2r_stop_success import build_stop_action_record, simulator_stop_success
from sub_instruction_node_matcher import SubInstructionNodeMatcher
from vlm_harness import NavigationVLMHarness, build_vlm_backend
from audit_rgb_only_contract import audit_run


DEFAULT_R2R_DATA = ROOT.parent / "3d_wm_vln/StreamVLN/data/datasets/r2r/val_unseen/val_unseen.json.gz"
DEFAULT_MP3D_ROOT = ROOT.parent / "3d_wm_vln/StreamVLN/data/scene_datasets/mp3d"
TAPIR_CHECKPOINT = ROOT / "models/tapnet/tapnet/checkpoints/causal_bootstapir_checkpoint.pt"


def yaw_from_coeffs(coeffs):
    rotation = quat_from_coeffs(np.asarray(coeffs, np.float32))
    return yaw_from_quaternion(rotation)


def yaw_from_quaternion(rotation):
    forward = quat_rotate_vector(rotation, np.array([0.0, 0.0, -1.0]))
    return wrap_angle(math.atan2(-float(forward[0]), -float(forward[2])))


def load_r2r_episode(dataset_path, episode_index=0, episode_id=None):
    with gzip.open(dataset_path, "rt") as handle:
        episodes = json.load(handle)["episodes"]
    if episode_id is not None:
        matches = [episode for episode in episodes if str(episode["episode_id"]) == str(episode_id)]
        if not matches:
            raise ValueError(f"R2R episode_id={episode_id} is absent from {dataset_path}")
        return matches[0]
    if not 0 <= episode_index < len(episodes):
        raise IndexError(f"episode-index must be in [0, {len(episodes) - 1}]")
    return episodes[episode_index]


def resolve_mp3d_scene(scene_id, mp3d_root):
    scene_name = Path(scene_id).stem
    scene = mp3d_root / scene_name / f"{scene_name}.glb"
    if not scene.exists():
        raise FileNotFoundError(f"Missing MP3D scene for R2R episode: {scene}")
    return scene


class CausalTapirCluster:
    def __init__(self, device="cuda:0", resolution=256, shared_model=None):
        self.device = torch.device(device)
        self.resolution = resolution
        if shared_model is None:
            self.model = tapir_model.TAPIR(
                pyramid_level=1, use_casual_conv=True)
            self.model.load_state_dict(torch.load(
                TAPIR_CHECKPOINT, map_location="cpu"))
            self.model.to(self.device).eval()
        else:
            self.model = shared_model
        self.features = None
        self.causal = None

    def _frame(self, rgb):
        resized = cv2.resize(rgb, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(resized).to(self.device).float() / 255 * 2 - 1

    @torch.inference_mode()
    def reset(self, rgb, points_xy):
        h, w = rgb.shape[:2]
        frame = self._frame(rgb)[None, None]
        q = np.concatenate([
            np.zeros((len(points_xy), 1), np.float32),
            points_xy[:, 1:2] * self.resolution / h,
            points_xy[:, 0:1] * self.resolution / w,
        ], 1)
        q = torch.from_numpy(q)[None].to(self.device)
        grids = self.model.get_feature_grids(frame, is_training=False)
        self.features = self.model.get_query_features(
            frame, is_training=False, query_points=q, feature_grids=grids
        )
        self.causal = self.model.construct_initial_causal_state(
            len(points_xy), len(self.features.resolutions) - 1
        )
        self.causal = tree.map_structure(lambda x: x.to(self.device), self.causal)
        return self.step(rgb)

    @torch.inference_mode()
    def step(self, rgb):
        h, w = rgb.shape[:2]
        frame = self._frame(rgb)[None, None]
        grids = self.model.get_feature_grids(frame, is_training=False)
        out = self.model.estimate_trajectories(
            frame.shape[-3:-1], is_training=False, feature_grids=grids,
            query_features=self.features, query_points_in_video=None,
            query_chunk_size=64, causal_context=self.causal,
            get_causal_context=True,
        )
        self.causal = out["causal_context"]
        tracks = out["tracks"][-1][0, :, 0].cpu().numpy()
        tracks[:, 0] *= w / self.resolution
        tracks[:, 1] *= h / self.resolution
        occ = out["occlusion"][-1][0, :, 0]
        dist = out["expected_dist"][-1][0, :, 0]
        visible = (((1 - torch.sigmoid(occ)) * (1 - torch.sigmoid(dist))) > 0.5)
        return tracks, visible.cpu().numpy().astype(bool)


def make_sim(scene, width, height, turn_step_deg=15.0, forward_step=0.25):
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = str(scene)
    cfg.enable_physics = False
    agent = habitat_sim.agent.AgentConfiguration()
    # Online policies issue these discrete actions through an RGB-only facade.
    # Collision resolution remains an internal simulator transition and no
    # pose/collision result is returned to the policy.
    agent.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(
                amount=float(forward_step))),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(
                amount=float(turn_step_deg))),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(
                amount=float(turn_step_deg))),
    }
    sensors = []
    for index, offset in enumerate([0, 60, 120, 180, 240, 300]):
        sensor = habitat_sim.CameraSensorSpec()
        sensor.uuid = "rgb" if index == 0 else f"pano_rgb_{index}"
        sensor.sensor_type = habitat_sim.SensorType.COLOR
        sensor.resolution = [height, width]
        sensor.position = [0.0, 1.25, 0.0]
        sensor.orientation = [0.0, math.radians(offset), 0.0]
        sensors.append(sensor)
        depth = habitat_sim.CameraSensorSpec()
        depth.uuid = "depth" if index == 0 else f"pano_depth_{index}"
        depth.sensor_type = habitat_sim.SensorType.DEPTH
        depth.resolution = [height, width]
        depth.position = [0.0, 1.25, 0.0]
        depth.orientation = [0.0, math.radians(offset), 0.0]
        sensors.append(depth)
    # Eight 45-degree RGB-D compass sectors are used by strict semantic point
    # selection and by edge-completion analysis.  The 0-degree sector reuses
    # the primary rgb/depth pair; the legacy six-view sensors above remain for
    # graph memory and backtracking compatibility.
    for index, offset in enumerate([45, 90, 135, 180, 225, 270, 315], 1):
        sensor = habitat_sim.CameraSensorSpec()
        sensor.uuid = f"completion_rgb_{index}"
        sensor.sensor_type = habitat_sim.SensorType.COLOR
        sensor.resolution = [height, width]
        sensor.position = [0.0, 1.25, 0.0]
        sensor.orientation = [0.0, math.radians(offset), 0.0]
        sensors.append(sensor)
        depth = habitat_sim.CameraSensorSpec()
        depth.uuid = f"completion_depth_{index}"
        depth.sensor_type = habitat_sim.SensorType.DEPTH
        depth.resolution = [height, width]
        depth.position = [0.0, 1.25, 0.0]
        depth.orientation = [0.0, math.radians(offset), 0.0]
        sensors.append(depth)
    agent.sensor_specifications = sensors
    return habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent]))


def set_pose(sim, position, yaw):
    state = sim.get_agent(0).get_state()
    state.position = np.asarray(position, np.float32)
    state.rotation = quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
    sim.get_agent(0).set_state(state)


def visited_area_estimate(history, radius, resolution=0.25):
    """Grid-union estimate of the area swept by recorded simulator positions."""
    if not history:
        return 0.0
    cells = set()
    reach = int(math.ceil(radius / resolution))
    for position in history:
        cx, cz = np.floor(np.asarray(position)[[0, 2]] / resolution).astype(int)
        for dx in range(-reach, reach + 1):
            for dz in range(-reach, reach + 1):
                if (dx * resolution) ** 2 + (dz * resolution) ** 2 <= radius ** 2:
                    cells.add((int(cx + dx), int(cz + dz)))
    return len(cells) * resolution * resolution


def navmesh_coverage(sim, history, radius, resolution=0.10):
    """Exact visited fraction on the current-floor Habitat navigable raster."""
    if not history:
        return 0.0, 0.0, 0.0
    height = float(np.median(np.asarray(history)[:, 1]))
    navigable = sim.pathfinder.get_topdown_view(resolution, height).astype(bool)
    visited = np.zeros(navigable.shape, np.uint8)
    lower, _ = sim.pathfinder.get_bounds()
    pixel_radius = max(1, int(round(radius / resolution)))
    for position in history:
        col = int(round((float(position[0]) - float(lower[0])) / resolution))
        row = int(round((float(position[2]) - float(lower[2])) / resolution))
        cv2.circle(visited, (col, row), pixel_radius, 1, -1)
    visited &= navigable.astype(np.uint8)
    visited_area = float(visited.sum()) * resolution * resolution
    navigable_area = float(navigable.sum()) * resolution * resolution
    return visited_area, navigable_area, visited_area / max(navigable_area, 1e-6)


def reachable_island_coverage(sim, history, radius, island_index,
                              sample_count=12000, seed=17):
    """Deterministic 3-D Monte-Carlo coverage of the full reachable island."""
    if not history or sample_count <= 0:
        return 0.0, 0, 0
    sim.pathfinder.seed(int(seed))
    samples = np.asarray([
        sim.pathfinder.get_random_navigable_point(
            max_tries=100, island_index=int(island_index))
        for _ in range(int(sample_count))
    ], np.float32)
    samples = samples[np.isfinite(samples).all(axis=1)]
    history_points = np.asarray(history, np.float32)
    covered = 0
    radius_squared = float(radius) ** 2
    for start in range(0, len(samples), 256):
        batch = samples[start:start + 256]
        squared = np.sum(
            (batch[:, None, :] - history_points[None, :, :]) ** 2,
            axis=2)
        covered += int(np.any(squared <= radius_squared, axis=1).sum())
    return covered / max(len(samples), 1), covered, len(samples)


def save_exploration_topdown(sim, history, records, output_path, resolution=0.08):
    """Render navmesh, simulator position history, and selected frontiers."""
    if not history:
        return
    height = float(np.median(np.asarray(history)[:, 1]))
    navigable = sim.pathfinder.get_topdown_view(resolution, height)
    canvas = np.zeros((*navigable.shape, 3), np.uint8)
    canvas[navigable] = (225, 225, 225)
    lower, _ = sim.pathfinder.get_bounds()

    def pixel(position):
        col = int(round((float(position[0]) - float(lower[0])) / resolution))
        row = int(round((float(position[2]) - float(lower[2])) / resolution))
        return (int(np.clip(col, 0, canvas.shape[1] - 1)),
                int(np.clip(row, 0, canvas.shape[0] - 1)))

    route = np.asarray([pixel(position) for position in history], np.int32)
    if len(route) > 1:
        cv2.polylines(canvas, [route], False, (255, 80, 20), 2)
    cv2.circle(canvas, pixel(history[0]), 4, (0, 200, 0), -1)
    cv2.circle(canvas, pixel(history[-1]), 4, (0, 0, 255), -1)
    for record in records:
        frontier = record.get("selection", {}).get("frontier_world_xyz")
        if frontier is not None:
            color = ((0, 180, 255) if record["end_reason"] == STOP_ARRIVAL_REASON
                     else (160, 0, 255))
            cv2.circle(canvas, pixel(frontier), 3, color, -1)
    cv2.imwrite(str(output_path), canvas)


class ExplorationVideoComposer:
    """Compose every saved frame from live obs, instruction, and top-down map."""

    def __init__(self, sim, start_position, obs_width=320, obs_height=240,
                 map_resolution=0.08):
        # Evaluator-owned raw simulator.  Policy modules call ``compose`` with
        # position=None under rgb_only_v1; geometry is read only inside this
        # output sink and is never returned to the caller.
        self._evaluation_sim = sim
        self._evaluation_position_history = [
            np.asarray(start_position, np.float32).copy()]
        self.obs_width = int(obs_width)
        self.obs_height = int(obs_height)
        self.left_size = (self.obs_width * 2, self.obs_height * 2)
        self.right_size = (self.obs_width, self.obs_height)
        self.frame_size = (self.left_size[0] + self.right_size[0],
                           self.left_size[1])
        self.map_resolution = float(map_resolution)
        height = float(np.asarray(start_position)[1])
        navigable = sim.pathfinder.get_topdown_view(
            self.map_resolution, height).astype(bool)
        self.map_base = np.zeros((*navigable.shape, 3), np.uint8)
        self.map_base[navigable] = (225, 225, 225)
        self.map_lower, _ = sim.pathfinder.get_bounds()

    def _map_pixel(self, position):
        col = int(round((float(position[0]) - float(self.map_lower[0])) /
                        self.map_resolution))
        row = int(round((float(position[2]) - float(self.map_lower[2])) /
                        self.map_resolution))
        return (int(np.clip(col, 0, self.map_base.shape[1] - 1)),
                int(np.clip(row, 0, self.map_base.shape[0] - 1)))

    @staticmethod
    def _letterbox(image, size):
        target_w, target_h = size
        scale = min(target_w / image.shape[1], target_h / image.shape[0])
        resized_w = max(1, int(round(image.shape[1] * scale)))
        resized_h = max(1, int(round(image.shape[0] * scale)))
        resized = cv2.resize(image, (resized_w, resized_h),
                             interpolation=cv2.INTER_NEAREST)
        panel = np.zeros((target_h, target_w, 3), np.uint8)
        x0 = (target_w - resized_w) // 2
        y0 = (target_h - resized_h) // 2
        panel[y0:y0 + resized_h, x0:x0 + resized_w] = resized
        return panel

    def render_topdown(self, position_history, position, yaw):
        topdown = self.map_base.copy()
        route = np.asarray([self._map_pixel(point) for point in position_history],
                           np.int32)
        if len(route) > 1:
            cv2.polylines(topdown, [route], False, (255, 90, 20), 2)
        if len(route):
            cv2.circle(topdown, tuple(route[0]), 4, (0, 180, 0), -1)
        current = np.asarray(position, np.float32)
        forward = np.array([-math.sin(yaw), 0.0, -math.cos(yaw)], np.float32)
        current_px = self._map_pixel(current)
        heading_px = self._map_pixel(current + 0.8 * forward)
        cv2.arrowedLine(topdown, current_px, heading_px, (0, 0, 255), 3,
                        tipLength=0.45)
        panel = self._letterbox(topdown, self.right_size)
        cv2.rectangle(panel, (0, 0), (self.right_size[0] - 1,
                                     self.right_size[1] - 1), (255, 255, 255), 1)
        cv2.putText(panel, "LIVE TOP-DOWN", (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 1, cv2.LINE_AA)
        return panel

    @staticmethod
    def _wrapped_lines(text, max_width, font, scale, thickness):
        words = str(text).split()
        lines = []
        current = ""
        for word in words:
            proposed = word if not current else f"{current} {word}"
            width = cv2.getTextSize(proposed, font, scale, thickness)[0][0]
            if current and width > max_width:
                lines.append(current)
                current = word
            else:
                current = proposed
        if current:
            lines.append(current)
        return lines or [""]

    def render_instruction(self, full_instruction, sub_instruction,
                           target_idx, phase):
        """Render text in its own video tile.

        The observation and top-down tiles intentionally contain no
        instruction text.  Keeping the complete R2R instruction and the
        currently executed sub-instruction together in this dedicated tile
        makes it possible to audit stage selection without obscuring either
        visual stream.
        """
        panel = np.zeros((self.right_size[1], self.right_size[0], 3), np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(panel, "INSTRUCTION", (12, 24), font, 0.58,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, f"stage {target_idx + 1} | {phase.replace('_', ' ')}",
                    (12, 44), font, 0.34, (190, 190, 190), 1, cv2.LINE_AA)
        cv2.putText(panel, "FULL INSTRUCTION", (12, 65), font, 0.37,
                    (80, 220, 255), 1, cv2.LINE_AA)
        y = 82
        full_lines = self._wrapped_lines(
            full_instruction or "(missing full instruction)",
            self.right_size[0] - 24, font, 0.35, 1)
        # Reserve the lower half of this independent tile for the active
        # sub-instruction.  Five lines fit the longest instruction in the
        # five-episode run while retaining the complete text.
        full_lines = full_lines[:5]
        for line in full_lines:
            cv2.putText(panel, line, (12, y), font, 0.35,
                        (235, 235, 235), 1, cv2.LINE_AA)
            y += 13
        cv2.line(panel, (10, 140), (self.right_size[0] - 10, 140),
                 (90, 90, 90), 1)
        cv2.putText(panel, "CURRENT SUB-INSTRUCTION", (12, 158), font, 0.34,
                    (120, 255, 140), 1, cv2.LINE_AA)
        y = 174
        sub_lines = self._wrapped_lines(
            sub_instruction or "(missing sub-instruction)",
            self.right_size[0] - 24, font, 0.39, 1)
        if len(sub_lines) > 4:
            sub_lines = sub_lines[:4]
            sub_lines[-1] = sub_lines[-1][:max(1, len(sub_lines[-1]) - 3)] + "..."
        for line in sub_lines:
            if y > self.right_size[1] - 8:
                break
            cv2.putText(panel, line, (12, y), font, 0.39,
                        (235, 255, 235), 1, cv2.LINE_AA)
            y += 17
        cv2.rectangle(panel, (0, 0), (self.right_size[0] - 1,
                                     self.right_size[1] - 1), (255, 255, 255), 1)
        return panel

    def compose(self, obs_bgr, position_history, position, yaw, instruction,
                target_idx, phase, *, full_instruction=None,
                sub_instruction=None):
        if position is None:
            state = self._evaluation_sim.get_agent(0).get_state()
            evaluation_position = np.asarray(state.position, np.float32)
            evaluation_yaw = yaw_from_quaternion(state.rotation)
            if (not self._evaluation_position_history or
                    not np.allclose(
                        self._evaluation_position_history[-1],
                        evaluation_position, atol=1e-6)):
                self._evaluation_position_history.append(
                    evaluation_position.copy())
            position = evaluation_position
            yaw = evaluation_yaw
            position_history = self._evaluation_position_history
        left = cv2.resize(obs_bgr, self.left_size, interpolation=cv2.INTER_LINEAR)
        instruction_panel = self.render_instruction(
            full_instruction if full_instruction is not None else instruction,
            sub_instruction if sub_instruction is not None else instruction,
            target_idx, phase)
        topdown_panel = self.render_topdown(position_history, position, yaw)
        right = np.concatenate([instruction_panel, topdown_panel], axis=0)
        return np.concatenate([left, right], axis=1)


def resolve_h264_ffmpeg():
    """Return an ffmpeg binary that has libx264, or None."""
    candidates = [os.environ.get("NAVI_FFMPEG_BIN"), shutil.which("ffmpeg"),
                  "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]
    for candidate in dict.fromkeys(c for c in candidates if c):
        try:
            probe = subprocess.run(
                [candidate, "-hide_banner", "-encoders"], capture_output=True,
                text=True, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if probe.returncode == 0 and "libx264" in probe.stdout:
            return candidate
    return None


class VideoFrameSink:
    """List-like streaming MP4 sink used by all exploration visualizations.

    Frames are piped to ffmpeg/libx264 (yuv420p, avc1, faststart) so the file
    plays in browser-based viewers such as VS Code; OpenCV's ``mp4v`` output
    is MPEG-4 Part 2, which those players cannot decode.  ffmpeg is only used
    as a CPU encoder and never touches the GPU.
    """

    def __init__(self, output_path, frame_size, fps=5, crf=20,
                 preset="veryfast"):
        self.output_path = Path(output_path)
        self.frame_size = tuple(map(int, frame_size))
        self.fps = float(fps)
        self.frame_count = 0
        self.temp_path = self.output_path.with_name(
            self.output_path.stem + ".tmp" + self.output_path.suffix)
        self.error_log_path = self.output_path.with_name(
            self.output_path.stem + ".tmp.ffmpeg.log")
        self.ffmpeg = resolve_h264_ffmpeg()
        self.codec = "h264" if self.ffmpeg else "mp4v"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.process = None
        self.writer = None
        if self.ffmpeg is None:
            print(f"warning: ffmpeg with libx264 not found; {self.output_path.name} "
                  "falls back to OpenCV mp4v and will not play in VS Code",
                  file=sys.stderr, flush=True)
            self.writer = cv2.VideoWriter(
                str(self.output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps, self.frame_size)
            if not self.writer.isOpened():
                raise RuntimeError(
                    f"Failed to open video writer: {self.output_path}")
            return
        width, height = self.frame_size
        # yuv420p needs even dimensions; the last row/column is duplicated.
        self.encoded_size = (width + width % 2, height + height % 2)
        self._error_handle = self.error_log_path.open("wb")
        self.process = subprocess.Popen([
            self.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s:v", "{}x{}".format(*self.encoded_size),
            "-r", f"{self.fps:.8g}",
            "-i", "pipe:0", "-map_metadata", "-1", "-an",
            "-c:v", "libx264", "-preset", preset, "-crf", str(int(crf)),
            "-pix_fmt", "yuv420p", "-tag:v", "avc1",
            "-movflags", "+faststart", str(self.temp_path),
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=self._error_handle)

    def append(self, frame):
        expected_w, expected_h = self.frame_size
        if frame.shape[:2] != (expected_h, expected_w):
            raise ValueError(
                f"video frame is {frame.shape[1]}x{frame.shape[0]}, expected "
                f"{expected_w}x{expected_h}")
        array = np.ascontiguousarray(np.asarray(frame, np.uint8)[..., :3])
        if self.process is not None:
            pad_h, pad_w = expected_h % 2, expected_w % 2
            if pad_h or pad_w:
                array = np.ascontiguousarray(np.pad(
                    array, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge"))
            try:
                self.process.stdin.write(array.tobytes())
            except BrokenPipeError as error:
                raise RuntimeError(
                    f"ffmpeg stopped accepting frames: {self._ffmpeg_error()}"
                ) from error
        else:
            self.writer.write(array)
        self.frame_count += 1

    def extend(self, frames):
        for frame in frames:
            self.append(frame)

    def _ffmpeg_error(self):
        try:
            return self.error_log_path.read_text(errors="replace")[-2000:]
        except OSError:
            return ""

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            return
        if self.process is None:
            return
        process, self.process = self.process, None
        process.stdin.close()
        try:
            returncode = process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            returncode = process.wait()
        self._error_handle.close()
        if returncode != 0:
            raise RuntimeError(
                f"ffmpeg exited with {returncode}: {self._ffmpeg_error()}")
        self.temp_path.replace(self.output_path)
        self.error_log_path.unlink(missing_ok=True)

    def __len__(self):
        return self.frame_count


def _run_habitat_episode(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["semantic", "pure-exploration"],
                   default="semantic")
    p.add_argument(
        "--policy-input-contract", choices=["rgb-only-v1"],
        default="rgb-only-v1",
        help=("mandatory online contract: navigation receives RGB and its own "
              "commanded action history only; geometry is evaluator-only"))
    p.add_argument(
        "--exploration-strategy",
        choices=["standard", "breadth-first",
                 "instruction-sequence-recovery"],
        default="standard",
        help=("standard greedy pure exploration, FIFO graph breadth-first "
              "pure exploration, or ordered semantic recovery"))
    p.add_argument("--r2r-data", type=Path, default=DEFAULT_R2R_DATA)
    p.add_argument("--mp3d-root", type=Path, default=DEFAULT_MP3D_ROOT)
    p.add_argument("--episode-index", type=int, default=0)
    p.add_argument("--episode-id", default=None)
    p.add_argument(
        "--reference-path-index", type=int, default=None,
        help="initialize directly at this real R2R reference_path state")
    p.add_argument(
        "--initial-node-artifact", type=Path, default=None,
        help=("initialize at the exact node position/yaw persisted by a prior "
              "real run; used for sequential curriculum stages"))
    p.add_argument(
        "--initial-node-id", default=None,
        help="node id inside --initial-node-artifact (default: last node)")
    p.add_argument(
        "--start-sub-instruction-index", type=int, default=None,
        help=("start semantic execution at this frozen sub-instruction index "
              "when using --initial-node-artifact"))
    p.add_argument(
        "--decomposition-artifact", type=Path, default=None,
        help=("reuse a prior trajectory/instruction_decomposition JSON so the "
              "curriculum does not re-decompose the instruction"))
    p.add_argument(
        "--single-point-test-scope", choices=["module", "full"], default=None,
        help="emit the project_rulle.md manifest and strict module results")
    p.add_argument(
        "--single-point-target-modules", default=None,
        help=("comma-separated modules fixed before a module-scope single-point "
              "test; used to make project_rulle.md target/not-run boundaries explicit"))
    p.add_argument(
        "--skip-instruction-completion", action="store_true",
        help=("module-test isolation: do not query semantic completion after the "
              "forward pair-forming navigation"))
    p.add_argument("--policy", choices=["gnm", "vint", "nomad"], default="gnm")
    p.add_argument("--vlm-backend", choices=["deepseek", "ollama", "heuristic"],
                   default="deepseek")
    p.add_argument("--vlm-model", default=None,
                   help="defaults to the selected backend's project model")
    p.add_argument(
        "--point-selection-prompt-version",
        choices=sorted(NavigationVLMHarness.POINT_SELECTION_PROMPT_VERSIONS),
        default="v10_approach_relation_router",
        help=("versioned and reversible semantic point-selection prompt; "
              "v1_baseline remains available for rollback"))
    p.add_argument(
        "--instruction-completion-prompt-version",
        choices=sorted(
            NavigationVLMHarness.INSTRUCTION_COMPLETION_PROMPT_VERSIONS),
        default="v13_structured_node_edge_binary",
        help=("versioned edge-completion harness; runtime output is always "
              "completed/unknown, while v9/v10 may use latent progress evidence"))
    p.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    p.add_argument("--deepseek-env", type=Path, default=ROOT / ".env.deepseek",
                   help="local file containing DEEPSEEK_API_KEY")
    p.add_argument("--deepseek-base-url", default=None,
                   help="optional DeepSeek-compatible API base URL override")
    p.add_argument("--vlm-timeout", type=int, default=180)
    p.add_argument("--vlm-retries", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--semantic-detector", choices=["dino-sam", "grounded-sam", "none"],
        default="dino-sam",
        help=("open-vocabulary semantic backend; grounded-sam is a deprecated "
              "compatibility spelling for dino-sam"))
    p.add_argument(
        "--floor-segmenter",
        choices=["dense-majority", "grounded-sam", "dino-sam"],
        default="dense-majority",
        help=("ground mask backend; default is pixelwise >=2/3 agreement of "
              "OneFormer, Mask2Former and SegFormer ADE20K models"))
    p.add_argument("--detector-box-threshold", type=float, default=0.28)
    p.add_argument("--detector-text-threshold", type=float, default=0.22)
    p.add_argument(
        "--adaptive-floor-threshold", action="store_true",
        help=("retain weak Grounded-SAM floor proposals for relation/continuous "
              "stages while keeping the default confidence for ordinary stages"))
    p.add_argument("--targets", type=int, default=None,
                   help="optional stage cap; default executes every decomposed stage")
    p.add_argument("--max-exploration-targets", type=int, default=120,
                   help="safety cap; pure exploration normally ends when its frontier set is empty")
    p.add_argument("--exploration-novelty-radius", type=float, default=1.0,
                   help="minimum XZ distance from every recorded simulator position")
    p.add_argument("--exploration-visit-radius", type=float, default=0.75,
                   help="radius used for swept-area coverage estimation")
    p.add_argument(
        "--full-space-coverage-threshold", type=float, default=0.90,
        help=("minimum swept current-floor coverage required in addition to "
              "natural BFS frontier exhaustion"))
    p.add_argument(
        "--coverage-samples", type=int, default=12000,
        help="deterministic samples for full reachable-island coverage")
    p.add_argument("--max-frontier-transit-attempts", type=int, default=3,
                   help="blacklist a remembered frontier after this many transit segments fail to visit it")
    p.add_argument("--max-steps-per-target", type=int, default=32)
    p.add_argument(
        "--tracking-cluster-profile",
        choices=sorted(TRACKING_CLUSTER_PROFILES),
        default="rgb_only_dense_stop_v1",
        help=("point-executor navigation/arrival cluster density and "
              "visibility confirmation profile"))
    p.add_argument("--edge-keyframe-count", type=int, default=5)
    p.add_argument("--views", type=int, choices=[6, 8], default=8)
    p.add_argument("--forward-step", type=float, default=0.22)
    p.add_argument("--turn-step-deg", type=float, default=15.0)
    p.add_argument("--scan-step-deg", type=float, default=15.0)
    p.add_argument("--backtrack-target-node", default=None,
                   help="after forward execution, backtrack to 'previous' or node_XXXX")
    p.add_argument(
        "--backtrack-only", action="store_true",
        help=("skip forward semantic execution and only run the physical "
              "backtrack from the imported latest node to --backtrack-target-node"))
    p.add_argument("--backtrack-selector", choices=["auto", "hybrid", "vlm"],
                   default="auto",
                   help="auto uses the VLM when available, otherwise visual/geometric scoring")
    p.add_argument(
        "--backtrack-planner-profile",
        choices=sorted(BACKTRACK_PLANNER_PROFILES),
        default="legacy_direct",
        help=("legacy direct node bearing, or online reverse breadcrumb + "
              "direction/depth/reachability guards"))
    p.add_argument("--backtrack-max-attempts-per-hop", type=int, default=2)
    p.add_argument("--backtrack-max-hops", type=int, default=5)
    p.add_argument("--backtrack-reach-radius", type=float, default=0.75)
    p.add_argument("--backtrack-min-visual-similarity", type=float, default=0.75)
    p.add_argument("--sequence-max-exploration-hops", type=int, default=30)
    p.add_argument("--sequence-max-blocked-directions", type=int, default=5)
    p.add_argument("--sequence-min-classification-confidence", type=float, default=0.5)
    p.add_argument("--sequence-recovery-backtrack-attempts", type=int, default=4)
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/random_exploration")
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args(argv)
    if args.mode != "semantic":
        p.error("rgb-only-v1 currently supports semantic R2R navigation only")
    if args.exploration_strategy != "instruction-sequence-recovery":
        p.error(
            "rgb-only-v1 requires --exploration-strategy "
            "instruction-sequence-recovery; legacy standard/BFS use geometry")
    if args.backtrack_only or args.backtrack_target_node is not None:
        p.error(
            "legacy standalone backtracking is disabled; RGB-only recovery is "
            "integrated into instruction-sequence navigation")
    if args.initial_node_artifact is not None:
        p.error(
            "legacy node artifacts contain pose; RGB-only continuation needs a "
            "new sanitized checkpoint format")
    if args.tracking_cluster_profile != "rgb_only_dense_stop_v1":
        p.error(
            "rgb-only-v1 requires --tracking-cluster-profile "
            "rgb_only_dense_stop_v1")
    if abs(float(args.scan_step_deg) - float(args.turn_step_deg)) > 1e-6:
        p.error(
            "rgb-only discrete turning requires --scan-step-deg to equal "
            "--turn-step-deg")
    if args.backtrack_only and args.backtrack_target_node is None:
        p.error("--backtrack-only requires --backtrack-target-node")
    if args.backtrack_only and args.initial_node_artifact is None:
        p.error("--backtrack-only requires --initial-node-artifact")
    if (args.exploration_strategy == "instruction-sequence-recovery" and
            args.mode != "semantic"):
        p.error("instruction-sequence-recovery requires --mode semantic")
    if (args.exploration_strategy == "breadth-first" and
            args.mode != "pure-exploration"):
        p.error("breadth-first requires --mode pure-exploration")
    # Production navigation must not silently place intermediate models on
    # CPU. Explicit CPU remains available for isolated unit tests, while the
    # normal CUDA path fails fast if the requested accelerator is unavailable.
    requested_device = torch.device(args.device)
    if requested_device.type == "cuda":
        if not torch.cuda.is_available():
            p.error(
                f"{args.device} was requested, but CUDA is unavailable; "
                "refusing a CPU fallback for intermediate models")
        device_index = (requested_device.index
                        if requested_device.index is not None
                        else torch.cuda.current_device())
        if device_index >= torch.cuda.device_count():
            p.error(
                f"requested CUDA device {device_index}, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible")
        torch.cuda.set_device(device_index)
    if not 0.0 <= args.sequence_min_classification_confidence <= 1.0:
        p.error("--sequence-min-classification-confidence must be in [0,1]")
    if args.sequence_recovery_backtrack_attempts < 1:
        p.error("--sequence-recovery-backtrack-attempts must be positive")
    if args.sequence_max_exploration_hops < 1:
        p.error("--sequence-max-exploration-hops must be positive")
    if not 0.0 <= args.full_space_coverage_threshold <= 1.0:
        p.error("--full-space-coverage-threshold must be in [0,1]")
    if args.coverage_samples < 1:
        p.error("--coverage-samples must be positive")
    if args.reference_path_index is not None and args.mode != "semantic":
        p.error("--reference-path-index currently requires --mode semantic")
    # Ordered semantic recovery must replay the actually executed edge trace
    # when it returns from an off-sequence node.  Keep ``legacy_direct`` as
    # the explicit/default behavior for standalone backtrack tests, but make
    # the sequence strategy use the generic breadcrumb planner unless the
    # caller selected another planner explicitly.  This preserves the
    # previous single-point module boundary while preventing a recovery hop
    # from steering only toward a direct bearing and overshooting a stored
    # node.
    if (args.exploration_strategy == "instruction-sequence-recovery" and
            args.backtrack_planner_profile == "legacy_direct"):
        args.backtrack_planner_profile = "breadcrumb_budget_v3"
    if args.initial_node_artifact is not None:
        if args.mode != "semantic":
            p.error("--initial-node-artifact requires --mode semantic")
        if args.reference_path_index is not None:
            p.error("do not combine --initial-node-artifact with --reference-path-index")
        if args.start_sub_instruction_index is None:
            p.error("--initial-node-artifact requires --start-sub-instruction-index")
    if args.start_sub_instruction_index is not None and args.start_sub_instruction_index < 0:
        p.error("--start-sub-instruction-index must be non-negative")
    if args.single_point_test_scope == "full":
        if args.reference_path_index is None:
            p.error("full single-point tests require --reference-path-index")
        if args.exploration_strategy != "instruction-sequence-recovery":
            p.error("full single-point tests require instruction-sequence-recovery")
        if args.targets != 1:
            p.error("full single-point tests require --targets 1")
        if args.backtrack_target_node is None:
            p.error("full single-point tests require an explicit physical backtrack")
        if args.skip_instruction_completion:
            p.error("full single-point tests cannot skip instruction completion")
    if (args.single_point_target_modules is not None and
            args.single_point_test_scope != "module"):
        p.error("--single-point-target-modules requires module test scope")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    views_dir = args.output_dir / "view_selections"
    views_dir.mkdir(exist_ok=True)
    width, height = 320, 240
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    episode = load_r2r_episode(args.r2r_data, args.episode_index, args.episode_id)
    instruction = episode.get("instruction", {}).get("instruction_text", "")
    reference_path = [np.asarray(point, np.float32)
                      for point in episode.get("reference_path", [])]
    reference_index = args.reference_path_index
    incoming_reference_origin = None
    reference_yaw_source = "episode_start_rotation"
    initial_node_artifact = None
    initial_node_id = None
    initial_graph_path = None
    initial_incoming_edge = None
    initial_incoming_source_node = None
    if args.initial_node_artifact is not None:
        artifact_path = Path(args.initial_node_artifact)
        if artifact_path.name == "navigation_graph.json":
            graph_payload = json.loads(artifact_path.read_text())
            initial_graph_path = artifact_path
        else:
            trajectory_payload = json.loads(artifact_path.read_text())
            graph_path = artifact_path.parent / "navigation_graph" / "navigation_graph.json"
            if not graph_path.exists():
                raise FileNotFoundError(
                    "trajectory artifact must sit beside navigation_graph/ "
                    f"or point directly to navigation_graph.json: {artifact_path}")
            graph_payload = json.loads(graph_path.read_text())
            initial_graph_path = graph_path
        nodes = list(graph_payload.get("nodes", []))
        if not nodes:
            raise RuntimeError(f"initial node artifact has no nodes: {artifact_path}")
        wanted = args.initial_node_id
        initial_node_artifact = next(
            (node for node in nodes if wanted is not None and
             str(node.get("node_id")) == str(wanted)), nodes[-1])
        initial_node_id = str(initial_node_artifact["node_id"])
        for edge in graph_payload.get("edges", []):
            if str(edge.get("target_node_id")) == initial_node_id:
                initial_incoming_edge = edge
                break
        if initial_incoming_edge is not None:
            source_id = str(initial_incoming_edge.get("source_node_id"))
            initial_incoming_source_node = next(
                (node for node in nodes if str(node.get("node_id")) == source_id),
                None)
        start_position = np.asarray(initial_node_artifact["position_xyz"], np.float32)
        yaw = float(initial_node_artifact["base_yaw_rad"])
        reference_index = None
        reference_yaw_source = "frozen_prior_real_node"
    elif reference_index is not None:
        if not reference_path:
            raise RuntimeError("selected R2R episode has no reference_path")
        if not 0 <= reference_index < len(reference_path):
            raise IndexError(
                f"reference-path-index must be in [0, {len(reference_path) - 1}]")
        start_position = reference_path[reference_index].copy()
        if reference_index == 0:
            yaw = yaw_from_coeffs(episode["start_rotation"])
        else:
            incoming_reference_origin = reference_path[reference_index - 1].copy()
            delta = start_position - incoming_reference_origin
            if np.linalg.norm(delta[[0, 2]]) <= 1e-6:
                raise RuntimeError("reference_path incoming segment has zero XZ length")
            yaw = wrap_angle(math.atan2(-float(delta[0]), -float(delta[2])))
            reference_yaw_source = "derived_from_previous_reference_position"
    else:
        start_position = np.asarray(episode["start_position"], np.float32)
        yaw = yaw_from_coeffs(episode["start_rotation"])

    manifest_path = args.output_dir / "manifest.json"
    all_single_point_modules = [
        "instruction_decomposition", "six_view_ground_perception",
        "vlm_point_selection", "point_navigation_executor",
        "navigation_graph_node_and_edge", "instruction_completion",
        "physical_node_backtracking", "loop_closure",
        "instruction_sequence_recovery",
    ]
    if args.single_point_test_scope == "full":
        target_modules = list(all_single_point_modules)
        required_upstream_modules = []
        not_run_modules = []
    elif args.single_point_test_scope == "module":
        target_modules = [
            item.strip() for item in
            str(args.single_point_target_modules or "").split(",")
            if item.strip()]
        unknown_modules = sorted(set(target_modules) - set(all_single_point_modules))
        if unknown_modules:
            p.error(f"unknown single-point target modules: {unknown_modules}")
        if "physical_node_backtracking" in target_modules:
            required_upstream_modules = [
                "instruction_decomposition", "six_view_ground_perception",
                "vlm_point_selection", "point_navigation_executor",
                "navigation_graph_node_and_edge",
            ]
        else:
            required_upstream_modules = []
        actually_run = set(target_modules) | set(required_upstream_modules)
        not_run_modules = [
            item for item in all_single_point_modules if item not in actually_run]
    else:
        target_modules, required_upstream_modules, not_run_modules = [], [], []

    manifest = {
        "schema_version": 1,
        "rule_file": str((ROOT / "project_rulle.md").resolve()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_scope": args.single_point_test_scope,
        "target_modules": target_modules,
        "required_upstream_modules": required_upstream_modules,
        "upstream_artifact_source": None,
        "not_run_modules": not_run_modules,
        "selection_fixed_before_model_run": True,
        "seed": args.seed,
        "split": args.r2r_data.stem.replace(".json", ""),
        "r2r_dataset": str(args.r2r_data.resolve()),
        "episode_index": args.episode_index,
        "episode_id": episode.get("episode_id"),
        "scene_id": episode.get("scene_id"),
        "trajectory_id": episode.get("trajectory_id"),
        "reference_path_index": reference_index,
        "reference_path_length": len(reference_path),
        "reference_position_xyz": start_position.tolist(),
        "reference_yaw_rad": float(yaw),
        "reference_yaw_source": reference_yaw_source,
        "initial_node_artifact": (
            str(args.initial_node_artifact.resolve())
            if args.initial_node_artifact is not None else None),
        "initial_node_id": initial_node_id,
        "decomposition_artifact": (
            str(args.decomposition_artifact.resolve())
            if args.decomposition_artifact is not None else None),
        "incoming_context_source": (
            "reference_initialization_context"
            if incoming_reference_origin is not None else None),
        "incoming_reference_origin_xyz": (
            incoming_reference_origin.tolist()
            if incoming_reference_origin is not None else None),
        "future_reference_information_exposed_to_models": False,
        "instruction": instruction,
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
    }
    if args.single_point_test_scope is not None:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    vlm_harness = None
    if args.mode == "semantic":
        backend = build_vlm_backend(
            args.vlm_backend, args.vlm_model, args.ollama_host, args.vlm_timeout,
            deepseek_env_file=args.deepseek_env,
            deepseek_base_url=args.deepseek_base_url)
        args.vlm_model = getattr(backend, "model", args.vlm_model or args.vlm_backend)
        vlm_harness = NavigationVLMHarness(
            backend, args.output_dir / "vlm_calls.json", args.vlm_retries,
            point_selection_prompt_version=(
                args.point_selection_prompt_version),
            instruction_completion_prompt_version=(
                args.instruction_completion_prompt_version))
        instruction_decomposer = InstructionDecomposer(vlm_harness)
        if args.decomposition_artifact is not None:
            decomposition_path = Path(args.decomposition_artifact)
            decomposition_payload = json.loads(decomposition_path.read_text())
            # The compositional ten-EP benchmark stores every frozen
            # decomposition in one manifest. Select only this episode's
            # stages; no benchmark metadata or reference path is forwarded to
            # the VLM. Older per-episode artifacts keep the original shape.
            if isinstance(decomposition_payload.get("episodes"), list):
                benchmark_row = next(
                    (row for row in decomposition_payload["episodes"]
                     if int(row.get("episode_index", -1)) == int(args.episode_index)),
                    None)
                if benchmark_row is None:
                    raise RuntimeError(
                        "decomposition benchmark has no episode index "
                        f"{args.episode_index}: {decomposition_path}")
                decomposition_payload = benchmark_row.get("decomposition") or benchmark_row
            raw_stages = (decomposition_payload.get(
                "all_decomposed_sub_instructions") or
                decomposition_payload.get("all_sub_instructions") or
                decomposition_payload.get("selected_sub_instructions") or
                decomposition_payload.get("stages"))
            if not raw_stages:
                raise RuntimeError(
                    "decomposition artifact has no all_decomposed_sub_instructions: "
                    f"{args.decomposition_artifact}")
            all_sub_instructions = [SubInstruction.from_mapping(value, index)
                                    for index, value in enumerate(raw_stages)]
        else:
            all_sub_instructions = instruction_decomposer.decompose(
                instruction, limit=None)
        aligned_stage_index = 0
        if args.start_sub_instruction_index is not None:
            aligned_stage_index = min(
                len(all_sub_instructions) - 1,
                int(args.start_sub_instruction_index))
            if args.backtrack_only:
                # Keep one descriptive stage for the video overlay and
                # metadata, but never execute it in backtrack-only mode.
                sub_instructions = all_sub_instructions[aligned_stage_index:aligned_stage_index + 1]
            elif args.targets is None:
                sub_instructions = all_sub_instructions[aligned_stage_index:]
            else:
                sub_instructions = all_sub_instructions[
                    aligned_stage_index:aligned_stage_index + args.targets]
        elif reference_index is not None:
            route_segments = max(len(reference_path) - 1, 1)
            aligned_stage_index = min(
                len(all_sub_instructions) - 1,
                int(math.floor(
                    reference_index * len(all_sub_instructions) /
                    route_segments)))
            if args.targets is None:
                sub_instructions = all_sub_instructions[aligned_stage_index:]
            else:
                sub_instructions = all_sub_instructions[
                    aligned_stage_index:aligned_stage_index + args.targets]
        else:
            sub_instructions = (
                all_sub_instructions if args.targets is None else
                all_sub_instructions[:args.targets])
    else:
        aligned_stage_index = 0
        frontier_limit = args.targets or args.max_exploration_targets
        sub_instructions = [SubInstruction.from_mapping({
            "sub_instruction_id": index,
            "navigation_instruction": "explore unvisited ground",
            "landmark": "none",
            "completion_cue": "all crop-bottom stopping points disappear",
            "semantic_spatial_target": "furthest ground frontier from position history",
            "spatial_relation": "unvisited", "visual_arrival_evidence": "tracker exhausted",
            "forbidden_target": "previously visited trajectory", "form": "PURE_EXPLORATION",
            "point_selection_strategy": {
                "perception": ["DINO+SAM ground", "depth", "sim position history"],
                "rank": "maximum XZ distance from every recorded simulator position",
                "semantic_information": "ignored",
            },
        }, index) for index in range(frontier_limit)]
        all_sub_instructions = list(sub_instructions)
    if not sub_instructions:
        raise RuntimeError("Instruction decomposition produced no executable sub-instructions")
    # Existing point selectors consume dictionaries containing ``stage_id``.
    # That name is now only a compatibility alias at this boundary.
    stages = [sub_instruction.to_stage_dict()
              for sub_instruction in sub_instructions]
    stage_count = len(stages)
    decomposition_record = {
        "instruction": instruction,
        "all_sub_instructions": [item.to_dict()
                                 for item in all_sub_instructions],
        "selected_sub_instructions": [item.to_dict()
                                      for item in sub_instructions],
        "alignment": {
            "method": "fixed_monotonic_path_progress_floor",
            "formula": (
                "floor(path_index * num_sub_instructions / "
                "max(reference_path_length - 1, 1))"),
            "reference_path_index": reference_index,
            "aligned_stage_index": aligned_stage_index,
            "alignment_uncertain": reference_index is not None,
        },
    }
    (args.output_dir / "instruction_decomposition.json").write_text(
        json.dumps(decomposition_record, ensure_ascii=False, indent=2) + "\n")
    if args.single_point_test_scope is not None:
        manifest.update({
            "aligned_stage_index": aligned_stage_index,
            "aligned_sub_instruction_id": (
                sub_instructions[0].sub_instruction_id),
            "alignment_method": "fixed_monotonic_path_progress_floor",
            "alignment_uncertain": reference_index is not None,
            "vlm_backend": args.vlm_backend,
            "vlm_model": args.vlm_model,
        })
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    scene = resolve_mp3d_scene(episode["scene_id"], args.mp3d_root)
    sim = make_sim(
        scene, width, height, turn_step_deg=args.turn_step_deg,
        forward_step=args.forward_step)
    if not sim.pathfinder.is_navigable(start_position):
        if reference_index is not None:
            sim.close()
            raise RuntimeError(
                "selected reference_path state is not navigable; strict tests "
                "must select another real state instead of snapping it")
        snapped = np.asarray(sim.pathfinder.snap_point(start_position), np.float32)
        if not np.isfinite(snapped).all():
            raise RuntimeError(f"R2R start is not navigable and cannot be snapped: {start_position}")
        start_position = snapped
    set_pose(sim, start_position, yaw)
    # This is the only simulator handle supplied to online navigation.  The
    # raw handle remains local to this Habitat adapter for initialization,
    # video diagnostics and post-run hidden geometric scoring.
    evaluation_position_history = [start_position.copy()]
    evaluation_action_trace = []
    evaluation_point_events = []
    frozen_evaluation_targets = {}

    def record_evaluation_action(action, raw_sim):
        state = raw_sim.get_agent(0).get_state()
        point = np.asarray(state.position, np.float32)
        evaluation_position_history.append(point.copy())
        evaluation_action_trace.append({
            "action_index": len(evaluation_action_trace),
            "action": str(action),
            "position_xyz": point.tolist(),
            "yaw_rad": float(yaw_from_quaternion(state.rotation)),
        })

    def record_evaluation_event(event, payload, raw_sim):
        state = raw_sim.get_agent(0).get_state()
        record = {
            "event": str(event),
            "payload": dict(payload),
            "position_xyz": np.asarray(state.position, np.float32).tolist(),
            "yaw_rad": float(yaw_from_quaternion(state.rotation)),
        }
        target_index = int(payload.get("target_index", -1))
        if event == "point_selected":
            obs = raw_sim.get_sensor_observations()
            depth = np.asarray(obs["depth"], np.float32)
            point_xy = np.asarray(payload["selected_point_xy"], np.float32)
            target_world, point_depth = pixel_ground_to_world(
                point_xy, depth, np.asarray(state.position, np.float32),
                yaw_from_quaternion(state.rotation))
            target_navmesh = None
            if target_world is not None:
                snapped = np.asarray(raw_sim.pathfinder.snap_point(target_world), np.float32)
                if np.isfinite(snapped).all():
                    target_navmesh = snapped.tolist()
            frozen_evaluation_targets[target_index] = target_navmesh
            record.update({
                "selected_point_depth_m": (
                    None if point_depth is None else float(point_depth)),
                "selected_point_world_xyz": (
                    None if target_world is None else
                    np.asarray(target_world, np.float32).tolist()),
                "selected_point_navmesh_xyz": target_navmesh,
            })
        elif event == "point_navigation_stopped":
            target = frozen_evaluation_targets.get(target_index)
            distance = None
            if target is not None:
                shortest = habitat_sim.ShortestPath()
                shortest.requested_start = np.asarray(state.position, np.float32)
                shortest.requested_end = np.asarray(target, np.float32)
                if raw_sim.pathfinder.find_path(shortest):
                    distance = float(shortest.geodesic_distance)
            record["final_target_geodesic_distance_m"] = distance
        evaluation_point_events.append(record)

    policy_sim = RGBOnlyPolicySimulator(
        sim, _evaluation_hook=record_evaluation_action,
        _evaluation_event_hook=record_evaluation_event)
    segmenter, floor_detector = build_ground_segmenter(
        args.floor_segmenter, args.device,
        box_threshold=args.detector_box_threshold,
        text_threshold=args.detector_text_threshold)
    semantic_detector = None
    if args.mode == "semantic" and args.semantic_detector in {
            "dino-sam", "grounded-sam"}:
        matching_floor_detector = (
            floor_detector is not None and
            ((args.semantic_detector == "grounded-sam" and
              isinstance(floor_detector, GroundedSamDetector)) or
             (args.semantic_detector == "dino-sam" and
              isinstance(floor_detector, DinoSamDetector))))
        semantic_detector = floor_detector if matching_floor_detector else (
            GroundedSamDetector(
                args.device, box_threshold=args.detector_box_threshold,
                text_threshold=args.detector_text_threshold)
            if args.semantic_detector == "grounded-sam" else
            DinoSamDetector(
                args.device, box_threshold=args.detector_box_threshold,
                text_threshold=args.detector_text_threshold))
    # Graph node semantics still need an open-vocabulary instance detector;
    # dense-majority is deliberately floor-only and is never reused here.
    dino_sam = (semantic_detector or floor_detector or
                DinoSamDetector(
                    args.device, box_threshold=args.detector_box_threshold,
                    text_threshold=args.detector_text_threshold))
    tracker = CausalTapirCluster(args.device)
    arrival_tracker = (
        None if args.tracking_cluster_profile == "legacy_3x3" else
        CausalTapirCluster(args.device, shared_model=tracker.model))
    policy, policy_cfg = build_policy(args.policy, torch.device(args.device))
    selected_yaws, records, motion_log = [], [], []
    total_step = 0
    previous_action_history = []
    previous_stage_origin = None
    # Evaluator-owned trace.  It is never passed to the strict strategy.
    position_history = evaluation_position_history
    if initial_incoming_edge is not None:
        # Restore the real incoming edge context so the next-stage selector and
        # executor receive the same action/history boundary as the prior run.
        previous_action_history = list(
            initial_incoming_edge.get("action_history", []))
        if initial_incoming_source_node is not None:
            previous_stage_origin = np.asarray(
                initial_incoming_source_node["position_xyz"], np.float32)
            position_history = [previous_stage_origin.copy()]
        for action in previous_action_history:
            value = action.get("position_xyz")
            if value is not None:
                position_history.append(np.asarray(value, np.float32))
        if not np.allclose(position_history[-1], start_position, atol=1e-4):
            position_history.append(start_position.copy())
    video_composer = ExplorationVideoComposer(
        sim, start_position, obs_width=width, obs_height=height)
    video_path = args.output_dir / f"episode_{int(episode['episode_id']):04d}.mp4"
    rendered = VideoFrameSink(video_path, video_composer.frame_size, fps=5)

    def emit_evaluation_video(
            obs_bgr, navigation_instruction, target_index, phase, *,
            full_instruction=None, sub_instruction=None, repeat=1):
        composed = video_composer.compose(
            obs_bgr, None, None, None, navigation_instruction,
            target_index, phase, full_instruction=full_instruction,
            sub_instruction=sub_instruction)
        for _ in range(max(1, int(repeat))):
            rendered.append(composed.copy())

    policy_video_bridge = RGBOnlyEvaluationVideoBridge(
        emit_evaluation_video)
    point_navigation_executor = PointNavigationExecutor(
        sim=policy_sim, tracker=tracker, arrival_tracker=arrival_tracker,
        policy=policy, policy_config=policy_cfg,
        policy_name=args.policy, predict_fn=predict, device=args.device,
        output_dir=args.output_dir, video_sink=None,
        video_composer=policy_video_bridge,
        max_steps=args.max_steps_per_target, forward_step=args.forward_step,
        turn_step_deg=args.turn_step_deg, seed=args.seed,
        tracking_cluster_profile=args.tracking_cluster_profile,
        edge_keyframe_count=args.edge_keyframe_count)
    if args.mode == "pure-exploration":
        selection_strategy = RandomExplorationPointSelector(
            segmenter=segmenter, video_composer=video_composer,
            novelty_radius=args.exploration_novelty_radius,
            views=args.views, scan_step=math.radians(args.scan_step_deg),
            max_frontier_transit_attempts=args.max_frontier_transit_attempts)
    else:
        selection_strategy = InstructionVLMPointSelector(
            segmenter=segmenter, semantic_detector=semantic_detector,
            vlm_harness=vlm_harness, video_composer=policy_video_bridge,
            views=args.views, scan_step=math.radians(args.scan_step_deg),
            policy_input_contract="rgb_only_v1")

    graph_memory = NavigationGraphMemory(
        args.output_dir / "navigation_graph",
        semantic_extractor=DinoSamEnvironmentSemanticExtractor(dino_sam),
        policy_input_contract="rgb_only_v1",
    )
    if initial_graph_path is not None:
        # Resume from the actual persisted graph rather than manufacturing a
        # fresh origin at the resumed pose.  The latter loses node identity and
        # can make the sequence judge accept an unexecuted hop.
        imported_latest = graph_memory.import_graph(initial_graph_path)
        if initial_node_id is not None and imported_latest.node_id != initial_node_id:
            raise RuntimeError(
                "initial-node-id must name the latest node in the imported "
                f"graph (latest={imported_latest.node_id}, requested={initial_node_id})")
    node_matcher = SubInstructionNodeMatcher()
    # Completion is instantiated inside the strict RGB-only sequence strategy;
    # do not construct the legacy pose-aware judge in production runs.
    completion_judge = None
    origin_rgbs = observe_six_rgb(policy_sim)
    # Evaluator-only depth snapshot.  It is persisted beside the test result
    # but never passed to graph memory, perception, VLM, selector or executor.
    _, origin_depths = observe_six_rgbd(sim)
    origin_completion_views = (
        observe_eight_rgb(sim)
        if args.instruction_completion_prompt_version in
        NavigationVLMHarness.EIGHT_VIEW_COMPLETION_PROMPT_VERSIONS else None)
    initial_views_dir = args.output_dir / "initial_six_views"
    evaluation_only_dir = args.output_dir / "evaluation_only"
    initial_depths_dir = evaluation_only_dir / "initial_depths"
    initial_views_dir.mkdir(exist_ok=True)
    initial_depths_dir.mkdir(parents=True, exist_ok=True)
    for view_index, (rgb, depth) in enumerate(
            zip(origin_rgbs, origin_depths)):
        Image.fromarray(rgb).save(
            initial_views_dir / f"view_{view_index}.jpg")
        np.save(initial_depths_dir / f"view_{view_index}.npy",
                np.asarray(depth, np.float32))
    if initial_graph_path is None:
        graph_memory.add_origin_node(
            position_xyz=None, base_yaw_rad=None, global_step=total_step,
            six_views=origin_rgbs, six_depths=None,
            completion_views=origin_completion_views,
            metadata={
                "role": "edge_source_root",
                "policy_input_contract": "rgb_only_v1",
                "privileged_inputs_used": [],
            })
    # Always emit a valid decision video, including episodes where the first
    # VLM selection fails before any scan or control action can add a frame.
    initial_rgb = origin_rgbs[0].copy()
    if reference_path:
        initial_projection = project_reference_path(
            reference_path,
            range(reference_index or 0, len(reference_path)),
            start_position + np.array([0.0, 1.25, 0.0], np.float32),
            yaw, initial_rgb.shape[1], initial_rgb.shape[0])
        initial_rgb = draw_reference_path_overlay(
            initial_rgb, initial_projection,
            next_path_index=(reference_index or 0) + 1)
    initial_video_frame = video_composer.compose(
        cv2.cvtColor(initial_rgb, cv2.COLOR_RGB2BGR),
        position_history, start_position, yaw,
        sub_instructions[0].navigation_instruction,
        target_idx=0, phase="initial_observation",
        full_instruction=instruction,
        sub_instruction=sub_instructions[0].navigation_instruction)
    for _ in range(5):
        rendered.append(initial_video_frame)

    sequence_exploration_result = None
    breadth_first_result = None
    if args.backtrack_only:
        stages_to_execute = []
    elif args.exploration_strategy == "instruction-sequence-recovery":
        sequence_strategy = RGBOnlyInstructionSequenceExplorationStrategy(
            sim=policy_sim, sub_instructions=sub_instructions,
            point_selector=selection_strategy,
            point_navigation_executor=point_navigation_executor,
            graph_memory=graph_memory, vlm_harness=vlm_harness,
            segmenter=segmenter, output_dir=args.output_dir,
            rendered=[], motion_log=motion_log,
            video_composer=policy_video_bridge,
            scan_step_deg=args.scan_step_deg,
            max_exploration_hops=args.sequence_max_exploration_hops,
            max_blocked_directions_per_node=(
                args.sequence_max_blocked_directions),
            minimum_classification_confidence=(
                args.sequence_min_classification_confidence),
            recovery_backtrack_attempts_per_hop=(
                args.sequence_recovery_backtrack_attempts),
            recovery_minimum_visual_similarity=(
                args.backtrack_min_visual_similarity),
            full_instruction=instruction,
            views=args.views)
        sequence_exploration_result = sequence_strategy.run(
            initial_action_heading=0.0, initial_global_step=total_step)
        yaw = sequence_exploration_result.final_yaw
        total_step = sequence_exploration_result.next_global_step
        selected_yaws.extend(sequence_exploration_result.selected_yaws_rad)
        records.extend(sequence_exploration_result.records)
        stages_to_execute = []
    elif args.exploration_strategy == "breadth-first":
        breadth_first_strategy = BreadthFirstExplorationStrategy(
            sim=sim, segmenter=segmenter,
            point_navigation_executor=point_navigation_executor,
            graph_memory=graph_memory, position_history=position_history,
            rendered=rendered, video_composer=video_composer,
            motion_log=motion_log, output_dir=args.output_dir,
            max_forward_attempts=args.max_exploration_targets,
            novelty_radius_m=args.exploration_novelty_radius,
            max_frontier_attempts=args.max_frontier_transit_attempts,
            max_frontiers_per_node=args.views,
            scan_step_deg=args.scan_step_deg,
            backtrack_attempts_per_hop=(
                args.backtrack_max_attempts_per_hop),
            backtrack_reach_radius_m=args.backtrack_reach_radius,
            backtrack_minimum_visual_similarity=(
                args.backtrack_min_visual_similarity),
            backtrack_planner_profile=args.backtrack_planner_profile,
            reference_path=reference_path,
            reference_path_index=(reference_index
                                  if reference_index is not None else 0))
        breadth_first_result = breadth_first_strategy.run(
            initial_yaw=yaw, initial_global_step=total_step)
        yaw = breadth_first_result.final_yaw
        total_step = breadth_first_result.next_global_step
        selected_yaws.extend(breadth_first_result.selected_yaws_rad)
        records.extend(breadth_first_result.records)
        stages_to_execute = []
    else:
        stages_to_execute = stages

    for target_idx, stage in enumerate(stages_to_execute):
        sub_instruction = sub_instructions[target_idx]
        if (args.adaptive_floor_threshold and
                args.floor_segmenter == "grounded-sam"):
            low_floor_forms = {
                "CIRCUMNAVIGATE", "APPROACH_LANDMARK",
                "TRAVERSE_PORTAL_REGION", "FOLLOW_PATH_BOUNDARY",
                "VERTICAL_DOWN",
            }
            segmenter.min_detection_score = (
                0.0 if str(stage.get("form", "")) in low_floor_forms else
                max(0.28, float(args.detector_box_threshold)))
        position = np.asarray(sim.get_agent(0).get_state().position, np.float32)
        stage_origin = position.copy()
        back_yaw = None
        if previous_stage_origin is not None:
            back_vector = previous_stage_origin - position
            if np.linalg.norm(back_vector[[0, 2]]) > 0.05:
                # Invert the executor's camera-forward mapping
                # [-sin(yaw), -cos(yaw)] for the current->previous ray.
                back_yaw = wrap_angle(math.atan2(-float(back_vector[0]),
                                                 -float(back_vector[2])))
        selection_result = selection_strategy.select(PointSelectionRequest(
            sim=sim, position=position, yaw=yaw, stage=stage,
            target_index=target_idx, rendered=rendered,
            motion_log=motion_log, position_history=position_history,
            previous_action_history=previous_action_history,
            back_yaw=back_yaw,
            reference_path=reference_path,
            reference_path_index=(reference_index
                                  if reference_index is not None else 0)))
        chosen, candidates = selection_result.chosen, selection_result.candidates
        if chosen is None:
            break
        selected_yaws.append(chosen["yaw"])
        decision_views = []
        for candidate in candidates:
            view = candidate["rgb"].copy()
            object_mask = candidate.get("small_seg_object_mask")
            if object_mask is not None:
                blue = np.zeros_like(view)
                blue[..., 2] = 255
                view[object_mask] = (0.82 * view[object_mask] +
                                     0.18 * blue[object_mask]).astype(np.uint8)
            green = np.zeros_like(view)
            green[..., 1] = 255
            mask = candidate["target_mask"]
            view[mask] = (0.55 * view[mask] + 0.45 * green[mask]).astype(np.uint8)
            projection = candidate.get("reference_path_projection")
            if projection is None and reference_path:
                path_start = reference_index or 0
                projection = project_reference_path(
                    reference_path, range(path_start, len(reference_path)),
                    position + np.array([0.0, 1.25, 0.0], np.float32),
                    candidate["yaw"], view.shape[1], view.shape[0])
            if projection is not None:
                view = draw_reference_path_overlay(
                    view, projection,
                    selected_point=(chosen["point"]
                                    if candidate is chosen else None),
                    selected=bool(candidate is chosen),
                    next_path_index=(reference_index or 0) + 1)
            decision_views.append(view)
        view_strip = np.concatenate(decision_views, axis=1)
        Image.fromarray(view_strip).save(
            views_dir / f"target_{target_idx:02d}.jpg")

        selected_world, selected_depth = pixel_ground_to_world(
            chosen["point"], chosen["depth"], position, chosen["yaw"])
        selected_navmesh = None
        selected_reachable = False
        selected_initial_geodesic = None
        if selected_world is not None:
            selected_navmesh = np.asarray(
                sim.pathfinder.snap_point(selected_world), np.float32)
            if np.isfinite(selected_navmesh).all():
                selected_path = habitat_sim.ShortestPath()
                selected_path.requested_start = position
                selected_path.requested_end = selected_navmesh
                selected_reachable = bool(
                    sim.pathfinder.find_path(selected_path))
                if selected_reachable:
                    selected_initial_geodesic = float(
                        selected_path.geodesic_distance)
        navigation_request = PointNavigationRequest(
            rgb=chosen["rgb"],
            selected_point_xy=chosen["point"],
            selectable_mask=chosen["target_mask"],
            ground_mask=chosen["mask"],
            yaw=chosen["yaw"],
            position_history=position_history,
            instruction=stage["navigation_instruction"],
            full_instruction=instruction,
            sub_instruction=stage["navigation_instruction"],
            semantic_target=stage["semantic_spatial_target"],
            target_index=target_idx,
            stage_count=stage_count,
            global_step=total_step,
            selected_point_depth_m=float(selected_depth),
            selected_point_reachable=selected_reachable,
            selected_point_initial_geodesic_m=selected_initial_geodesic,
            selected_point_navmesh_xyz=selected_navmesh,
            reference_path=reference_path,
            reference_path_index=(reference_index
                                  if reference_index is not None else 0),
        )
        navigation_result = execute_point_navigation(
            point_navigation_executor, navigation_request)
        arrival_signal_received = (
            navigation_result.signal == POINT_NAVIGATION_ARRIVED)
        yaw = navigation_result.final_yaw
        total_step = navigation_result.next_global_step
        action_history = navigation_result.action_history
        target_record = {
            "target_index": target_idx,
            "selection_reference_yaw_rad": wrap_angle(
                chosen["yaw"] - chosen["relative_yaw_rad"]),
            "selected_yaw_rad": chosen["yaw"],
            "selected_relative_yaw_rad": chosen["relative_yaw_rad"],
            "sub_instruction": sub_instruction.to_dict(),
            "instruction_stage": stage,
            "prior_stage_action_history": previous_action_history,
            "incoming_backtrack_yaw_rad": back_yaw,
            "selection": (
                chosen["selection"] if args.mode == "pure-exploration"
                else chosen["vlm_selection"]),
            "ground_fraction": chosen["ground_fraction"],
            "candidate_views": [{
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in candidate.items()
                if key not in {
                    "rgb", "depth", "mask", "target_mask", "point",
                    "semantic_detections", "small_seg_object_mask",
                    "ground_anchors",
                }
            } for candidate in candidates],
            "executor": "PointNavigationExecutor",
            "arrived": arrival_signal_received,
            **navigation_result.record,
        }
        records.append(target_record)
        target_record["action_history"] = action_history
        stop_position = np.asarray(
            sim.get_agent(0).get_state().position, np.float32)
        # Hidden post-run audit of physical arrival.  The selected navmesh
        # target is computed before control and is never passed to the policy;
        # this score is only persisted after the executor has finished so the
        # curriculum can distinguish a true point arrival from an off-screen
        # stop without changing behavior.
        final_target_geodesic = math.inf
        if selected_navmesh is not None:
            final_path = habitat_sim.ShortestPath()
            final_path.requested_start = stop_position
            final_path.requested_end = selected_navmesh
            if sim.pathfinder.find_path(final_path):
                final_target_geodesic = float(final_path.geodesic_distance)
        point_target_reference_reached = bool(
            math.isfinite(final_target_geodesic) and
            final_target_geodesic <= 0.75)
        physical_arrival = bool(
            arrival_signal_received and point_target_reference_reached)
        if arrival_signal_received and not physical_arrival:
            physical_failure_type = "premature_offscreen_stop"
        elif (not arrival_signal_received and
              point_target_reference_reached):
            physical_failure_type = "missed_point_arrival_signal"
        elif not arrival_signal_received:
            physical_failure_type = str(navigation_result.end_reason)
        else:
            physical_failure_type = None
        target_record.update({
            # Persist the post-selection projection only after the VLM has
            # frozen the RGB view/pixel.  These fields are audit artifacts for
            # hidden heading/physical-arrival scoring; they are never exposed
            # to the VLM, point policy, or completion judge.
            "selected_point_world_xyz": (
                selected_world.tolist() if selected_world is not None else None),
            "selected_navmesh_target_xyz": (
                selected_navmesh.tolist() if selected_navmesh is not None else None),
            "final_target_geodesic_distance_m": (
                final_target_geodesic
                if math.isfinite(final_target_geodesic) else None),
            "point_target_reference_reached": point_target_reference_reached,
            "navigation_physical_arrival": physical_arrival,
            "navigation_physical_failure_type": physical_failure_type,
            "point_arrival_judgment": {
                "owner": "PointNavigationExecutor",
                "signal": navigation_result.signal,
                "executor_declared_arrival": bool(arrival_signal_received),
                "reference_final_geodesic_within_threshold": (
                    point_target_reference_reached),
                "reference_threshold_m": 0.75,
            },
        })
        stop_rgbs, stop_depths = observe_six_rgbd(sim)
        stop_completion_views = (
            observe_eight_rgb(sim)
            if args.instruction_completion_prompt_version in
            NavigationVLMHarness.EIGHT_VIEW_COMPLETION_PROMPT_VERSIONS else None)
        # Do not let an off-screen/cluster arrival signal create a graph node.
        # The node boundary is the verified physical arrival, i.e. the
        # executor signal together with the post-run selected-navmesh endpoint
        # check above.  Failed attempts remain in target_record only and are
        # retryable; they must not become semantic completion evidence.
        stop_node = None
        stop_edge = None
        match_result = None
        if physical_arrival:
            stop_node, stop_edge = graph_memory.add_navigation_stop_node(
                position_xyz=stop_position, base_yaw_rad=yaw,
                global_step=total_step, six_views=stop_rgbs,
                six_depths=stop_depths, sub_instruction=sub_instruction,
                action_history=action_history,
                arrival_signal=navigation_result.signal,
                completion_views=stop_completion_views,
                metadata={"target_index": target_idx,
                          "executor_end_reason": navigation_result.end_reason},
                edge_metadata={
                    "edge_keyframes": navigation_result.record.get(
                        "edge_keyframes", [])})
            match_result = node_matcher.match(stop_node, sub_instruction)
            graph_memory.set_sub_instruction_match(
                stop_node.node_id, match_result.to_dict())
        if (args.mode == "semantic" and physical_arrival and
                not args.skip_instruction_completion):
            instruction_completion = completion_judge.judge(
                stop_node, sub_instruction.sub_instruction_id,
                (stop_completion_views
                 if stop_completion_views is not None else stop_rgbs),
                navigation_result.edge_keyframes).to_dict()
        elif args.mode == "semantic" and args.skip_instruction_completion:
            instruction_completion = {
                "status": "not_run",
                "instruction_completed": False,
                "reason": "disabled for isolated point-backtracking module test",
            }
        elif args.mode == "semantic":
            instruction_completion = {
                "status": UNKNOWN,
                "instruction_completed": False,
                "expected_sub_instruction_id": int(
                    sub_instruction.sub_instruction_id),
                "confidence": 1.0,
                "reason": (
                    "point executor did not declare arrival; semantic "
                    "completion was not queried"),
                "visual_evidence": "",
                "previous_node_id": (
                    stop_edge.source_node_id if stop_edge is not None else
                    (graph_memory.nodes[-1].node_id
                     if graph_memory.nodes else None)),
                "current_node_id": stop_node.node_id if stop_node is not None else None,
                "incoming_edge_id": stop_edge.edge_id if stop_edge is not None else None,
            }
        else:
            instruction_completion = None
        target_record.update({
            "navigation_graph_node_id": (
                stop_node.node_id if stop_node is not None else None),
            "navigation_graph_edge_id": (
                stop_edge.edge_id if stop_edge is not None else None),
            "sub_instruction_match": (
                match_result.to_dict() if match_result is not None else None),
            "instruction_completion": instruction_completion,
            # The legacy current-node matcher is diagnostic only. Semantic
            # completion comes exclusively from the previous-node -> edge ->
            # current-node binary judge after point arrival.
            "sub_instruction_satisfied": bool(
                instruction_completion is not None and
                instruction_completion["instruction_completed"]),
        })
        previous_action_history = action_history
        previous_stage_origin = stage_origin
        selection_strategy.on_navigation_result(
            selection_result, navigation_result, position_history,
            target_record)
        if (args.mode == "semantic" and
                not target_record["sub_instruction_satisfied"]):
            break
        if args.mode == "pure-exploration":
            exploration_state = selection_strategy.snapshot(position_history)
            checkpoint = {
                "mode": args.mode,
                "targets_attempted": len(records),
                "targets_completed": sum(
                    target["end_reason"] == STOP_ARRIVAL_REASON
                    for target in records),
                "total_control_steps": total_step,
                "position_history_xyz": [position.tolist()
                                         for position in position_history],
                **exploration_state,
                "last_target": {
                    "target_index": target_record["target_index"],
                    "end_reason": target_record["end_reason"],
                    "selection": target_record["selection"],
                },
            }
            (args.output_dir / "exploration_checkpoint.json").write_text(
                json.dumps(checkpoint, indent=2) + "\n")
    if (args.mode == "pure-exploration" and
            args.exploration_strategy == "standard" and
            selection_strategy.termination_reason is None):
        selection_strategy.termination_reason = "max_exploration_targets"

    backtrack_result = None
    if args.backtrack_target_node is not None:
        backtrack_vlm_harness = vlm_harness
        if args.backtrack_selector == "vlm" and backtrack_vlm_harness is None:
            backtrack_backend = build_vlm_backend(
                args.vlm_backend, args.vlm_model,
                args.ollama_host, args.vlm_timeout,
                deepseek_env_file=args.deepseek_env,
                deepseek_base_url=args.deepseek_base_url)
            backtrack_vlm_harness = NavigationVLMHarness(
                backtrack_backend,
                args.output_dir / "backtrack_vlm_calls.json",
                args.vlm_retries,
                point_selection_prompt_version=(
                    args.point_selection_prompt_version))
        backtracker = NodeBacktrackingController(
            sim=sim, graph_memory=graph_memory, segmenter=segmenter,
            point_navigation_executor=point_navigation_executor,
            position_history=position_history, rendered=rendered,
            video_composer=video_composer, motion_log=motion_log,
            vlm_harness=backtrack_vlm_harness,
            selector_mode=args.backtrack_selector,
            scan_step_deg=args.scan_step_deg,
            max_attempts_per_hop=args.backtrack_max_attempts_per_hop,
            reach_radius_m=args.backtrack_reach_radius,
            minimum_visual_similarity=args.backtrack_min_visual_similarity,
            max_hops=args.backtrack_max_hops,
            output_dir=args.output_dir,
            planner_profile=args.backtrack_planner_profile,
            reference_path=reference_path,
            reference_path_index=(reference_index
                                  if reference_index is not None else 0),
        )
        backtrack_result = backtracker.backtrack(
            target_node_id=args.backtrack_target_node,
            current_yaw=yaw, global_step=total_step,
            target_index_offset=stage_count)
        yaw = backtrack_result.final_yaw
        total_step = backtrack_result.next_global_step

    final_position = np.asarray(sim.get_agent(0).get_state().position, np.float32)
    goal = episode.get("goals", [{}])[0]
    goal_position = np.asarray(goal.get("position", final_position), np.float32)
    shortest = habitat_sim.ShortestPath()
    shortest.requested_start = final_position
    shortest.requested_end = goal_position
    final_geodesic = float(shortest.geodesic_distance) if sim.pathfinder.find_path(shortest) else math.inf
    start_shortest = habitat_sim.ShortestPath()
    start_shortest.requested_start = start_position
    start_shortest.requested_end = goal_position
    initial_geodesic = (float(start_shortest.geodesic_distance)
                        if sim.pathfinder.find_path(start_shortest) else math.inf)
    traveled_distance = (
        ((sum(float(np.linalg.norm(current - previous))
              for previous, current in zip(
                  evaluation_position_history,
                  evaluation_position_history[1:]))
          if sequence_exploration_result is not None else
          breadth_first_result.traveled_distance_m
          if breadth_first_result is not None else
          sum(step["moved_m"] for target in records
              for step in target["steps"]))) +
        (backtrack_result.traveled_distance_m if backtrack_result is not None else 0.0))
    success_radius = float(goal.get("radius", 3.0))
    goal_radius_hit = final_geodesic <= success_radius
    sequence_completed_in_order = bool(
        sequence_exploration_result is not None and
        sequence_exploration_result.success)
    complete_instruction_was_evaluated = bool(
        sequence_exploration_result is not None and
        reference_index is None and
        len(sub_instructions) == len(all_sub_instructions))
    task_stop = build_stop_action_record(
        sequence_completed_in_order=sequence_completed_in_order,
        complete_instruction_was_evaluated=complete_instruction_was_evaluated,
        global_step=total_step, position_xyz=final_position)
    simulator_reported_success = simulator_stop_success(
        stop_action_issued=task_stop["issued"],
        goal_radius_hit=goal_radius_hit)
    system_provisional_goal_success = simulator_reported_success
    # Independent semantic verification is a post-run scoring step.  The live
    # navigator cannot certify its own completion decision, so task success
    # fails closed here and can only be promoted by the evaluator after a
    # stage_completion_verification.json audit is present.
    instruction_validated_success = False
    invalid_goal_radius_hit = bool(goal_radius_hit)
    spl = 0.0
    visited_area, floor_navigable_area, coverage_ratio = navmesh_coverage(
        sim, position_history, args.exploration_visit_radius)
    navigable_area_value = sim.pathfinder.navigable_area
    navigable_area = float(navigable_area_value() if callable(navigable_area_value)
                           else navigable_area_value)
    start_island = int(sim.pathfinder.get_island(start_position))
    reachable_island_area = float(sim.pathfinder.island_area(start_island))
    (reachable_coverage_ratio, reachable_samples_covered,
     reachable_samples_total) = reachable_island_coverage(
        sim, position_history, args.exploration_visit_radius, start_island,
        sample_count=args.coverage_samples, seed=args.seed + 100003)
    if args.mode == "pure-exploration":
        save_exploration_topdown(
            sim, position_history, records, args.output_dir / "topdown_trajectory.png")
        exploration_state = (
            breadth_first_result.state
            if breadth_first_result is not None else
            selection_strategy.snapshot(position_history))
    else:
        exploration_state = None
    evaluation_geometry_path = evaluation_only_dir / "evaluation_geometry.json"
    evaluation_geometry_path.write_text(json.dumps({
        "scope": "postrun_test_validation_only",
        "visible_to_navigation_policy": False,
        "policy_input_contract": "rgb_only_v1",
        "action_trace": evaluation_action_trace,
        "point_events": evaluation_point_events,
        "final_position_xyz": final_position.tolist(),
        "final_geodesic_distance_m": (
            final_geodesic if math.isfinite(final_geodesic) else None),
    }, ensure_ascii=False, indent=2) + "\n")
    sim.close()
    video_frame_count = len(rendered)
    rendered.close()
    summary = {
        "benchmark": "R2R", "split": args.r2r_data.stem.replace(".json", ""),
        "mode": args.mode,
        "r2r_dataset": str(args.r2r_data.resolve()),
        "episode_id": episode["episode_id"], "trajectory_id": episode.get("trajectory_id"),
        "instruction": instruction,
        "reference_state": {
            "path_index": reference_index,
            "path_length": len(reference_path),
            "position_xyz": start_position.tolist(),
            "yaw_rad": float(manifest["reference_yaw_rad"]),
            "yaw_source": reference_yaw_source,
            "incoming_context_source": manifest["incoming_context_source"],
            "future_reference_information_exposed_to_models": False,
        },
        "all_decomposed_sub_instructions": [
            item.to_dict() for item in all_sub_instructions],
        "stage_alignment": decomposition_record["alignment"],
        "sub_instructions": [item.to_dict() for item in sub_instructions],
        "instruction_stages": stages,
        "task_stop": task_stop,
        "r2r_goal_positions": [goal["position"] for goal in episode.get("goals", [])],
        "r2r_metrics": {
                        "success": instruction_validated_success,
                        "instruction_validated_r2r_success": (
                            instruction_validated_success),
                        "goal_radius_hit_diagnostic": goal_radius_hit,
                        "stop_action_issued": task_stop["issued"],
                        "simulator_reported_success": (
                            simulator_reported_success),
                        "invalid_goal_radius_hit": invalid_goal_radius_hit,
                        "system_provisional_goal_success": (
                            system_provisional_goal_success),
                        "independent_semantic_verification_complete": False,
                        "sequence_completed_in_order": sequence_completed_in_order,
                        "complete_instruction_was_evaluated": (
                            complete_instruction_was_evaluated),
                        "success_definition": (
                            "strict success: active task-level STOP after the "
                            "complete decomposed instruction sequence, goal "
                            "radius hit at STOP, and independent post-run "
                            "semantic verification; simulator_reported_success "
                            "requires STOP and goal radius only"),
                        "success_radius_m": success_radius,
                        "final_geodesic_distance_m": final_geodesic,
                        "initial_geodesic_distance_m": initial_geodesic,
                        "path_length_m": traveled_distance, "spl": spl},
        "scene": str(scene.resolve()), "policy": args.policy,
        "selection_strategy": (
            BreadthFirstExplorationStrategy.mode
            if breadth_first_result is not None else selection_strategy.mode),
        "exploration_strategy": args.exploration_strategy,
        "navigation_executor": "PointNavigationExecutor",
        "policy_input_contract": {
            "name": "rgb_only_v1",
            "online_observations": [
                "rgb", "instruction", "commanded_action_history",
                "rgb_derived_masks_tracks_embeddings"],
            "forbidden_online_inputs": [
                "pose", "depth", "navmesh", "pathfinder", "geodesic",
                "world_coordinates", "collision_feedback",
                "reference_path"],
            "runtime_boundary": "RGBOnlyPolicySimulator",
            "hidden_validation_artifact": str(evaluation_geometry_path),
        },
        "floor_segmenter": ground_segmenter_display_name(
            args.floor_segmenter),
        "strict_ground_selection": bool(
            getattr(segmenter, "strict_ground_mask", False)),
        "path_projection_visualization": {
            "enabled": bool(reference_path),
            "reference": "R2R reference_path projected after selection",
            "ground_truth_color_rgb": [0, 230, 255],
            "selected_point_color_rgb": [255, 30, 30],
            "path_is_not_model_input": True,
        },
        "instruction_decomposer": "InstructionDecomposer",
        "navigation_graph_memory": graph_memory.summary(),
        "sub_instruction_node_matcher": {
            "class": "SubInstructionNodeMatcher",
            "threshold": node_matcher.threshold,
            "role": "legacy_diagnostic_only",
        },
        "instruction_completion_judge": {
            "class": "RGBOnlyNodeTransitionInstructionCompletionJudge",
            "outcomes": ["completed", "unknown"],
            "prompt_version": args.instruction_completion_prompt_version,
            "route_membership_output": False,
            "latent_progress_evidence": bool(
                args.instruction_completion_prompt_version in {
                    "v9_three_way_edge_progress",
                    "v10_bidirectional_three_way"}),
        },
        "node_backtracking": (
            backtrack_result.to_dict() if backtrack_result is not None else None),
        "instruction_sequence_exploration": ({
            key: value for key, value in
            sequence_exploration_result.to_dict().items() if key != "records"
        } if sequence_exploration_result is not None else None),
        "breadth_first_exploration": ({
            key: value for key, value in
            breadth_first_result.to_dict().items() if key != "records"
        } if breadth_first_result is not None else None),
        "targets_requested": stage_count,
        "targets_attempted": len(records),
        "targets_completed": sum(
            bool(target.get("arrived")) for target in records),
        "arrival_rule": "rgb_only_dense_stop_cluster_arrival",
        "tracking_cluster_roles": {
            "navigation": "3x3 points constrained to the crop's central one-ninth area",
            "stopping": "3x3 points along the crop's bottom-edge band, ground-snapped when available",
            "navigation_signal": "navigation-cluster centroid plus image-goal policy waypoint",
            "stop_signal": (
                "at least half of dense stopping-cluster points invisible for "
                "three RGB frames, with commanded-forward and RGB-motion guards"),
            "dynamic_crop_anchor": "currently visible stopping-cluster points",
        },
        "vlm": ({"backend": args.vlm_backend, "model": args.vlm_model,
                 "harness_log": "vlm_calls.json"}
                if args.mode == "semantic" else None),
        "pure_exploration": ({
            "semantic_instruction_ignored": True,
            "position_history_xyz": [position.tolist() for position in position_history],
            "position_history_count": len(position_history),
            **exploration_state,
            "novelty_radius_m": args.exploration_novelty_radius,
            "visit_radius_m": args.exploration_visit_radius,
            "visited_area_estimate_m2": visited_area,
            "current_floor_navigable_area_m2": floor_navigable_area,
            "scene_navigable_area_m2": navigable_area,
            "reachable_island_area_m2": reachable_island_area,
            "estimated_coverage_ratio": coverage_ratio,
            "reachable_island_coverage_ratio": reachable_coverage_ratio,
            "reachable_island_samples_covered": reachable_samples_covered,
            "reachable_island_samples_total": reachable_samples_total,
            "reachable_island_visited_area_estimate_m2": (
                reachable_coverage_ratio * reachable_island_area),
            "full_space_coverage_threshold": (
                args.full_space_coverage_threshold),
            "full_space_complete": bool(
                breadth_first_result is not None and
                breadth_first_result.success and
                reachable_coverage_ratio >=
                args.full_space_coverage_threshold),
            "full_space_completion_definition": (
                "BFS frontier queue naturally exhausted AND deterministic "
                "3-D swept coverage of the full start-reachable navmesh "
                "island meets the frozen threshold"),
        } if args.mode == "pure-exploration" else None),
        "total_control_steps": total_step, "selected_yaws_rad": selected_yaws,
        "continuous_scan_frames": len(motion_log),
        "video": {
            "path": str(video_path), "frame_count": video_frame_count,
            "codec": rendered.codec,
            "fps": 5, "frame_size_wh": list(video_composer.frame_size),
            "layout": "left_live_obs__upper_right_instruction__lower_right_topdown",
        },
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
        "motion_log": motion_log, "targets": records,
    }
    trajectory_path = args.output_dir / "trajectory.json"
    trajectory_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    contract_audit = audit_run(args.output_dir)
    if not contract_audit["passed"]:
        raise RuntimeError(
            "RGB-only artifact contract failed: " +
            ", ".join(contract_audit["violations"]))

    if args.single_point_test_scope == "full":
        def write_json(name, value):
            path = args.output_dir / name
            path.write_text(
                json.dumps(value, ensure_ascii=False, indent=2) + "\n")
            return path

        first_target = records[0] if records else {}
        candidate_views = first_target.get("candidate_views", [])
        selection = first_target.get("selection", {})
        classification = first_target.get(
            "sub_instruction_sequence_classification", {})
        backtrack_payload = (
            backtrack_result.to_dict() if backtrack_result is not None else {})
        backtrack_attempts = backtrack_payload.get("attempts", [])

        write_json("semantic_detections.json", {
            "views": [{
                "view_index": item.get("view_index", index),
                "detection_queries": item.get("detection_queries", []),
                "detections": item.get("detection_records", []),
                "ground_detections": item.get("ground_detection_records", []),
            } for index, item in enumerate(candidate_views)],
        })
        write_json("point_selection.json", {
            "sub_instruction": first_target.get("sub_instruction"),
            "selected_yaw_rad": first_target.get("selected_yaw_rad"),
            "selected_point_xy": first_target.get("selected_point_xy"),
            "selected_point_depth_m": first_target.get(
                "selected_point_depth_m"),
            "selected_point_world_xyz": first_target.get(
                "selected_point_world_xyz"),
            "selected_navmesh_target_xyz": first_target.get(
                "selected_navmesh_target_xyz"),
            "selection": selection,
            "candidate_views": candidate_views,
        })
        write_json("forward_action_history.json", {
            "navigation_graph_edge_id": first_target.get(
                "navigation_graph_edge_id"),
            "actions": first_target.get("action_history", []),
        })
        write_json("node_classification.json", classification)
        write_json("backtrack_selection.json", {
            "selector": args.backtrack_selector,
            "attempts": [{
                "segment_index": attempt.get("segment_index"),
                "selection": attempt.get("selection"),
                "node_revisit_match": attempt.get("node_revisit_match"),
                "executed_point_navigation": attempt.get(
                    "executed_point_navigation"),
            } for attempt in backtrack_attempts],
        })
        write_json("backtrack_action_history.json", {
            "attempts": [{
                "segment_index": attempt.get("segment_index"),
                "created_edge_id": attempt.get("created_edge_id"),
                "actions": attempt.get("action_history", []),
            } for attempt in backtrack_attempts],
        })

        point_call = next(
            (call for call in (vlm_harness.calls if vlm_harness else [])
             if call.get("task") == "select_ground_target"), None)
        if point_call is not None:
            (args.output_dir / "vlm_prompt.txt").write_text(
                point_call.get("prompt", "") + "\n")
            write_json("vlm_response.json", point_call)
            image_paths = point_call.get("image_paths", [])
            if image_paths:
                source = args.output_dir / image_paths[0]
                if source.exists():
                    shutil.copy2(source, args.output_dir / "vlm_contact_sheet.jpg")

        heading_error_deg = None
        point_direction_correct = False
        selected_target = first_target.get("selected_navmesh_target_xyz")
        if (reference_index is not None and
                reference_index + 1 < len(reference_path) and
                selected_target is not None):
            demo_vector = (reference_path[reference_index + 1] -
                           start_position)[[0, 2]]
            selected_vector = (np.asarray(selected_target, np.float32) -
                               start_position)[[0, 2]]
            demo_norm = float(np.linalg.norm(demo_vector))
            selected_norm = float(np.linalg.norm(selected_vector))
            if demo_norm > 1e-6 and selected_norm > 1e-6:
                cosine = float(np.clip(
                    np.dot(demo_vector, selected_vector) /
                    (demo_norm * selected_norm), -1.0, 1.0))
                heading_error_deg = float(math.degrees(math.acos(cosine)))
                point_direction_correct = heading_error_deg < 90.0

        graph_payload = json.loads(graph_memory.graph_path.read_text())
        forward_node_id = first_target.get("navigation_graph_node_id")
        forward_edge_id = first_target.get("navigation_graph_edge_id")
        persisted_node = next(
            (node for node in graph_payload.get("nodes", [])
             if node.get("node_id") == forward_node_id), None)
        persisted_edge = next(
            (edge for edge in graph_payload.get("edges", [])
             if edge.get("edge_id") == forward_edge_id), None)
        edge_history_valid = bool(
            persisted_edge is not None and
            persisted_edge.get("action_history") ==
            first_target.get("action_history", []))
        executed_backtrack = any(
            bool(attempt.get("executed_point_navigation"))
            for attempt in backtrack_attempts)
        backtrack_selection_valid = any(
            attempt.get("selection") is not None and
            bool(attempt.get("executed_point_navigation"))
            for attempt in backtrack_attempts)
        loop_closure_valid = any(
            attempt.get("loop_closure_edge_id") is not None
            for attempt in backtrack_attempts)
        decomposition_valid = bool(
            sub_instructions and
            all(item.semantic_spatial_target and item.form
                for item in sub_instructions))
        ground_candidate_valid = bool(
            len(candidate_views) == 6 and
            any(float(item.get("ground_fraction", 0.0)) > 0
                for item in candidate_views))
        point_depth = first_target.get("selected_point_depth_m")
        point_depth_valid = bool(
            point_depth is not None and math.isfinite(float(point_depth)) and
            float(point_depth) > 0)
        point_on_ground = bool(
            selection.get("requested_on_ground") is True and
            len(selection.get("snapped_xy", [])) == 2)
        navigation_internal_arrival = bool(first_target.get("arrived"))
        navigation_physical_arrival = bool(
            first_target.get("navigation_physical_arrival"))
        classifier_correct = bool(
            classification.get("expected_sequence_position") and
            float(classification.get("confidence", 0.0)) >=
            args.sequence_min_classification_confidence)
        backtrack_physical_revisit = bool(
            backtrack_payload.get("success") and executed_backtrack)

        result_values = {
            "decomposition_valid": decomposition_valid,
            "ground_candidate_valid": ground_candidate_valid,
            "point_direction_correct": point_direction_correct,
            "point_on_ground": point_on_ground,
            "point_depth_valid": point_depth_valid,
            "navigation_internal_arrival": navigation_internal_arrival,
            "navigation_physical_arrival": navigation_physical_arrival,
            "node_persisted": persisted_node is not None,
            "edge_action_history_valid": edge_history_valid,
            "sub_instruction_classification_correct": classifier_correct,
            "backtrack_selection_valid": backtrack_selection_valid,
            "backtrack_physical_revisit": backtrack_physical_revisit,
            "loop_closure_valid": loop_closure_valid,
        }
        module_results = {
            "test_scope": "full",
            "case_count": 1,
            "accuracy_claimed": False,
            "reference_state": summary["reference_state"],
            "aligned_sub_instruction_id": (
                sub_instructions[0].sub_instruction_id),
            "heading_error_to_demo_next_deg": heading_error_deg,
            "navigation_target_distance": {
                "initial_geodesic_m": first_target.get(
                    "initial_target_geodesic_distance_m"),
                "final_geodesic_m": first_target.get(
                    "final_target_geodesic_distance_m"),
                "threshold_m": first_target.get(
                    "navigation_physical_arrival_threshold_m", 0.75),
            },
            "navigation_failure_type": first_target.get(
                "navigation_physical_failure_type"),
            "sequence_directive": first_target.get("sequence_directive"),
            "backtrack_end_reason": backtrack_payload.get("end_reason"),
            "modules": result_values,
            "all_modules_success": all(result_values.values()),
        }
        write_json("module_results.json", module_results)

        hash_paths = [
            "instruction_decomposition.json", "vlm_calls.json",
            "point_selection.json", "forward_action_history.json",
            "node_classification.json", "backtrack_selection.json",
            "backtrack_action_history.json", "trajectory.json",
            "module_results.json", "navigation_graph/navigation_graph.json",
            video_path.name,
        ]
        artifact_hashes = {}
        for relative in hash_paths:
            path = args.output_dir / relative
            if path.exists():
                artifact_hashes[relative] = hashlib.sha256(
                    path.read_bytes()).hexdigest()
        manifest.update({
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_sha256": artifact_hashes,
            "all_modules_executed": bool(
                records and classification and backtrack_result is not None and
                executed_backtrack),
            "all_modules_success": module_results["all_modules_success"],
        })
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        **{k: summary[k] for k in [
            "policy", "targets_completed", "total_control_steps",
            "selected_yaws_rad"]},
        "backtrack_success": (backtrack_result.success
                              if backtrack_result is not None else None),
    }, indent=2))
    print(f"video: {video_path}\ntrajectory: {args.output_dir / 'trajectory.json'}")
    return summary


class HabitatPointNavigationInterface:
    """Main Habitat adapter connecting a selector to PointNavigationExecutor."""

    def __init__(self, argv=None):
        self.argv = argv

    def run_episode(self):
        return _run_habitat_episode(self.argv)


def run_habitat_episode(argv=None):
    """Functional interface for one R2R/Habitat episode."""
    return HabitatPointNavigationInterface(argv).run_episode()


def main(argv=None):
    return run_habitat_episode(argv)


if __name__ == "__main__":
    main()
