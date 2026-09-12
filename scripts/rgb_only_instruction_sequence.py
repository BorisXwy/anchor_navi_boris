#!/usr/bin/env python3
"""Strict RGB-only ordered instruction exploration for Habitat R2R.

This module intentionally does not import habitat_sim.  It receives a
capability-limited simulator facade, consumes RGB panoramas, segmentation,
VLM results, point tracks and commanded action history, and emits embodied
actions.  Pose/depth/navmesh validation belongs to the caller after the run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from direction_gate import (
    IN_PLACE_TURN_PHASE, TURN_GATE_CENTER_DEG, DirectionGateEmptyError,
    plan_in_place_turn,
)
from instruction_completion_judge import (
    RGBOnlyNodeTransitionInstructionCompletionJudge,
)
from instruction_sequence_exploration import (
    InstructionSequenceExplorationResult, InstructionSequenceStateMachine,
)
from navigation_graph_memory import CompactVisualEmbedder
from point_navigation_executor import (
    REVERSAL_SKIP_KEY, PointNavigationRequest, execute_point_navigation,
)
from point_selectors import (
    PointSelectionRequest, continuous_turn, observe, observe_eight_rgb,
    observe_six_rgb, select_ground_point, targetable_ground_mask, wrap_angle,
)
from rgb_only_runtime import require_rgb_only_policy_sim

TURN_TO_SELECTED_TARGET_PHASE = "turn_to_selected_target"
TURN_START_KEYFRAME_PHASE = "turn_start"


def _cosine(left, right):
    left = np.asarray(left, np.float32)
    right = np.asarray(right, np.float32)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-8 else 0.0


def turn_to_selected_target_actions(motion_log, start_index, hop_index):
    """Edge action records for the in-place turn the point selector executed.

    ``continuous_turn`` logs the discrete ``turn_left``/``turn_right`` commands
    it issues only to ``motion_log``; the executor's action history starts
    after the turn.  Reconstruct those commands as RGB-only action records so
    the completion judge sees the same turn the simulator received.  Eight-view
    refinement probes turn out and straight back, so they are not part of the
    executed edge and are skipped.  Steps are negative so they sort before the
    executor's own ``0..N`` steps.
    """
    entries = [
        entry for entry in list(motion_log)[start_index:]
        if entry.get("phase") == TURN_TO_SELECTED_TARGET_PHASE
        and int(entry.get("target_index", -1)) == int(hop_index)
        and entry.get("action") in {"turn_left", "turn_right"}
    ]
    count = len(entries)
    return [{
        "step": index - count,
        "action": str(entry["action"]),
        "commanded_turn_deg": float(entry.get("commanded_turn_deg", 0.0)),
        "forward_commanded": False,
        "orientation_only": True,
        "phase": TURN_TO_SELECTED_TARGET_PHASE,
        "policy_input_contract": "rgb_only_v1",
    } for index, entry in enumerate(entries)]


def prepend_turn_start_keyframe(output_dir, hop_index, turn_start_rgb,
                                turn_step_count, keyframe_records):
    """Persist the pre-turn forward frame as keyframe 0 of this edge.

    The executor numbers and saves its own keyframes before this frame is
    known, so its records are shifted by one while their image paths stay.
    """
    output_dir = Path(output_dir)
    keyframes_dir = output_dir / "edge_keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    path = keyframes_dir / f"target_{int(hop_index):02d}_turn_start.jpg"
    Image.fromarray(np.asarray(turn_start_rgb, np.uint8)[..., :3]).save(
        path, quality=92)
    turn_start_record = {
        "step": -(int(turn_step_count) + 1),
        "phase": TURN_START_KEYFRAME_PHASE,
        "action": None,
        "policy_input_contract": "rgb_only_v1",
        "keyframe_index": 0,
        "source_frame_index": -1,
        "image_path": str(path.relative_to(output_dir)),
    }
    shifted = []
    for record in keyframe_records:
        item = dict(record)
        item["keyframe_index"] = int(item.get("keyframe_index", 0)) + 1
        shifted.append(item)
    return [turn_start_record] + shifted


def in_place_turn_keyframes(output_dir, hop_index, frames, action_records,
                            count=5):
    """Persist an even subsample of an in-place turn as edge keyframes.

    ``frames[0]`` is the view before the first turn command and
    ``frames[i]`` the view after ``action_records[i - 1]``, so the judge gets
    the same chronological storyboard a forward hop gets from the executor.
    """
    output_dir = Path(output_dir)
    keyframes_dir = output_dir / "edge_keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    total = len(frames)
    indices = sorted(set(
        int(round(value)) for value in
        np.linspace(0, total - 1, max(2, min(int(count), total))).tolist()))
    selected = []
    records = []
    for order, index in enumerate(indices):
        frame = np.asarray(frames[index], np.uint8)[..., :3]
        selected.append(frame)
        path = keyframes_dir / (
            f"target_{int(hop_index):02d}_in_place_turn_{order:02d}.jpg")
        Image.fromarray(frame).save(path, quality=92)
        records.append({
            "step": index - 1,
            "phase": IN_PLACE_TURN_PHASE,
            "action": (str(action_records[index - 1]["action"])
                       if index >= 1 else None),
            "policy_input_contract": "rgb_only_v1",
            "keyframe_index": order,
            "source_frame_index": index,
            "image_path": str(path.relative_to(output_dir)),
        })
    return selected, records


@dataclass
class RGBOnlyBacktrackResult:
    success: bool
    target_node_id: str
    recovered_node_id: str | None
    final_action_heading_rad: float
    next_global_step: int
    attempts: list[dict]
    end_reason: str

    def to_dict(self):
        return asdict(self)


class RGBOnlyGraphBacktracker:
    """Return to a stored node using panorama correspondence and point nav."""

    def __init__(self, sim, graph_memory, segmenter, vlm_harness,
                 point_navigation_executor, views=6, scan_step_deg=15.0,
                 max_attempts=2, minimum_visual_similarity=0.75):
        self.sim = require_rgb_only_policy_sim(sim)
        self.graph_memory = graph_memory
        self.segmenter = segmenter
        self.vlm_harness = vlm_harness
        self.executor = point_navigation_executor
        self.views = int(views)
        self.scan_step = math.radians(float(scan_step_deg))
        self.max_attempts = int(max_attempts)
        self.minimum_visual_similarity = float(minimum_visual_similarity)
        self.embedder = CompactVisualEmbedder()

    def _candidates(self, rgbs, action_heading):
        offsets = np.radians([0, 60, 120, 180, 240, 300])
        segmentations = (self.segmenter.batch(rgbs)
                         if hasattr(self.segmenter, "batch") else
                         [self.segmenter(rgb) for rgb in rgbs])
        result = []
        for index, (offset, rgb, segmentation) in enumerate(zip(
                offsets, rgbs, segmentations)):
            ground, _ = segmentation
            ground = np.asarray(ground, bool)
            target = targetable_ground_mask(ground)
            point, score = select_ground_point(target)
            result.append({
                "view_index": index,
                "relative_yaw_rad": float(offset),
                "rgb": np.asarray(rgb, np.uint8)[..., :3],
                "mask": ground,
                "target_mask": target,
                "point": point,
                "point_score": float(score),
                "ground_fraction": float(ground.mean()),
                # Action-frame heading, never simulator/world yaw.
                "action_heading_rad": wrap_angle(
                    float(action_heading) + float(offset)),
            })
        return result

    def recover(self, target_node_id, action_heading, global_step,
                target_index, forward_action_history):
        target_node = self.graph_memory.get_node(target_node_id)
        target_views = self.graph_memory.load_node_views(target_node_id)
        attempts = []
        heading = float(action_heading)
        step = int(global_step)
        for attempt_index in range(self.max_attempts):
            current = observe_six_rgb(self.sim)
            similarity_before = _cosine(
                self.embedder.embed(current), target_node.visual_embedding)
            if similarity_before >= self.minimum_visual_similarity:
                return RGBOnlyBacktrackResult(
                    True, str(target_node_id), None, heading, step, attempts,
                    "target_panorama_reobserved")
            candidates = self._candidates(current, heading)
            try:
                chosen_index, point, selection = (
                    self.vlm_harness.select_backtrack_ground_target_rgb_only(
                        target_node_id=str(target_node_id),
                        target_node_views=target_views,
                        current_node_id="live_rgb_observation",
                        candidates=candidates,
                        forward_action_history=forward_action_history))
            except Exception as exc:
                attempts.append({
                    "attempt_index": attempt_index,
                    "selection_error": str(exc),
                    "policy_input_contract": "rgb_only_v1",
                })
                break
            chosen = candidates[chosen_index]
            target_heading = wrap_angle(
                heading + float(chosen["relative_yaw_rad"]))
            heading, chosen_rgb = continuous_turn(
                self.sim, None, heading, target_heading, self.scan_step,
                [], [], target_index + attempt_index, "rgb_only_backtrack_turn",
                None, None, f"return to node {target_node_id}",
                policy_input_contract="rgb_only_v1")
            navigation = execute_point_navigation(
                self.executor, PointNavigationRequest(
                    rgb=chosen_rgb, selected_point_xy=point,
                    selectable_mask=chosen["target_mask"],
                    ground_mask=chosen["mask"], yaw=heading,
                    position_history=[], instruction=(
                        f"visually return to stored node {target_node_id}"),
                    semantic_target="stored node panorama",
                    target_index=target_index + attempt_index,
                    stage_count=1, global_step=step,
                    policy_input_contract="rgb_only_v1",
                    allow_initial_near_field_arrival=False))
            heading = navigation.final_yaw
            step = navigation.next_global_step
            current_after = observe_six_rgb(self.sim)
            similarity_after = _cosine(
                self.embedder.embed(current_after), target_node.visual_embedding)
            attempt = {
                "attempt_index": attempt_index,
                "selection": selection,
                "selected_view_index": chosen_index,
                "selected_point_xy": np.asarray(point).tolist(),
                "executor_arrived": navigation.arrived,
                "executor_end_reason": navigation.end_reason,
                "action_history": navigation.action_history,
                "target_panorama_similarity_before": similarity_before,
                "target_panorama_similarity_after": similarity_after,
                "policy_input_contract": "rgb_only_v1",
                "privileged_inputs_used": [],
            }
            attempts.append(attempt)
            if navigation.arrived and similarity_after >= self.minimum_visual_similarity:
                completion_views = observe_eight_rgb(self.sim)
                node, edge = self.graph_memory.add_navigation_stop_node(
                    position_xyz=None, base_yaw_rad=None,
                    global_step=step, six_views=current_after,
                    six_depths=None, sub_instruction={
                        "navigation_instruction": (
                            f"visually return to stored node {target_node_id}"),
                        "semantic_spatial_target": "stored node panorama",
                    }, action_history=navigation.action_history,
                    arrival_signal=navigation.signal,
                    completion_views=completion_views,
                    metadata={
                        "policy_input_contract": "rgb_only_v1",
                        "rgb_revisit_target_node_id": str(target_node_id),
                        "rgb_panorama_similarity": similarity_after,
                    }, edge_kind="rgb_only_node_backtrack",
                    edge_metadata={
                        "edge_keyframes": navigation.record.get(
                            "edge_keyframes", []),
                        "policy_input_contract": "rgb_only_v1",
                    })
                self.graph_memory.add_loop_closure_edge(
                    str(target_node_id), node.node_id,
                    metadata={
                        "verification": "rgb_panorama_similarity_only",
                        "similarity": similarity_after,
                        "policy_input_contract": "rgb_only_v1",
                    })
                attempt["recovered_node_id"] = node.node_id
                attempt["recovery_edge_id"] = edge.edge_id
                return RGBOnlyBacktrackResult(
                    True, str(target_node_id), node.node_id, heading, step,
                    attempts, "target_panorama_reobserved_after_point_navigation")
        return RGBOnlyBacktrackResult(
            False, str(target_node_id), None, heading, step, attempts,
            "rgb_only_backtrack_failed")


BACKTRACK_METHODS = ("action-reversal", "visual")
ACTION_REVERSAL_TURN_AROUND_PHASE = "action_reversal_turn_around"
ACTION_REVERSAL_REPLAY_PHASE = "action_reversal_replay"
ACTION_REVERSAL_RESTORE_HEADING_PHASE = "action_reversal_restore_heading"
_REVERSED_ACTION = {
    "move_forward": "move_forward",
    "turn_left": "turn_right",
    "turn_right": "turn_left",
}


def turn_around_action_count(turn_step_deg):
    """Number of discrete turns in a half rotation; 180 must divide evenly."""
    turns = 180.0 / float(turn_step_deg)
    if abs(turns - round(turns)) > 1e-6:
        raise ValueError(
            "action-reversal backtracking needs 180 to be an integer multiple "
            f"of the turn step, got turn_step_deg={turn_step_deg}")
    return int(round(turns))


def plan_action_reversal(action_history, turn_step_deg):
    """Action records that physically undo ``action_history``.

    Turn a half rotation in place, replay the hop backwards with left and
    right swapped, then turn another half rotation so the action-frame
    heading ends where the hop started.  Only the commanded discrete actions
    are used; no pose is read or assumed.
    """
    turn_step_deg = float(turn_step_deg)
    turn_count = turn_around_action_count(turn_step_deg)
    plan = []

    def append(action, phase):
        commanded = (turn_step_deg if action == "turn_left" else
                     -turn_step_deg if action == "turn_right" else 0.0)
        plan.append({
            "step": len(plan),
            "action": action,
            "commanded_turn_deg": commanded,
            "forward_commanded": action == "move_forward",
            "orientation_only": action != "move_forward",
            "phase": phase,
            "policy_input_contract": "rgb_only_v1",
        })

    for _ in range(turn_count):
        append("turn_left", ACTION_REVERSAL_TURN_AROUND_PHASE)
    for record in reversed(list(action_history)):
        action = str(record.get("action"))
        if action not in _REVERSED_ACTION:
            raise ValueError(
                f"cannot reverse unknown action {action!r} in action history")
        # A forward the executor tagged as producing no RGB motion never
        # moved the agent; replaying it would walk the return leg too far.
        if record.get(REVERSAL_SKIP_KEY):
            continue
        append(_REVERSED_ACTION[action], ACTION_REVERSAL_REPLAY_PHASE)
    for _ in range(turn_count):
        append("turn_left", ACTION_REVERSAL_RESTORE_HEADING_PHASE)
    return plan


def count_reversal_skips(action_history):
    return sum(1 for record in action_history
               if dict(record).get(REVERSAL_SKIP_KEY))


class RGBOnlyActionReversalBacktracker:
    """Return to the hop's start node by replaying its actions in reverse.

    The forward hop's commanded action history is deterministic, so undoing
    it is a fixed action sequence rather than a fresh navigation problem.
    Whether the agent really stands at the stored node again is decided by
    the VLM comparing the stored and the current six-view panoramas; the
    panorama embedding similarity is recorded for audits but never gates.
    """

    method = "action-reversal"

    def __init__(self, sim, graph_memory, vlm_harness, turn_step_deg=15.0,
                 video_composer=None, full_instruction=None):
        self.sim = require_rgb_only_policy_sim(sim)
        self.graph_memory = graph_memory
        self.vlm_harness = vlm_harness
        self.turn_step_deg = float(turn_step_deg)
        self.turn_around_actions = turn_around_action_count(self.turn_step_deg)
        self.video_composer = video_composer
        self.full_instruction = str(full_instruction or "")
        self.embedder = CompactVisualEmbedder()

    def _emit_video(self, rgb, instruction, target_index, phase):
        if self.video_composer is None:
            return
        bgr = np.ascontiguousarray(np.asarray(rgb, np.uint8)[..., :3][..., ::-1])
        self.video_composer.emit(
            bgr, instruction, target_index, phase,
            full_instruction=self.full_instruction or instruction,
            sub_instruction=instruction)

    def recover(self, target_node_id, action_heading, global_step,
                target_index, forward_action_history, *, create_revisit_node):
        target_node_id = str(target_node_id)
        target_node = self.graph_memory.get_node(target_node_id)
        target_views = self.graph_memory.load_node_views(target_node_id)
        instruction = f"reverse action history back to node {target_node_id}"
        plan = plan_action_reversal(forward_action_history, self.turn_step_deg)
        self.sim.emit_evaluation_event("action_reversal_backtrack_started", {
            "target_node_id": target_node_id,
            "target_index": int(target_index),
            "forward_step_count": len(list(forward_action_history)),
            "reversal_step_count": len(plan),
        })
        heading = float(action_heading)
        step = int(global_step)
        for record in plan:
            observations = self.sim.step(record["action"])
            heading = wrap_angle(
                heading + math.radians(record["commanded_turn_deg"]))
            step += 1
            self._emit_video(observations["rgb"], instruction, target_index,
                             record["phase"])

        current_views = observe_six_rgb(self.sim)
        similarity = _cosine(
            self.embedder.embed(current_views), target_node.visual_embedding)
        attempt = {
            "attempt_index": 0,
            "backtrack_method": self.method,
            "turn_around_actions": self.turn_around_actions,
            "forward_step_count": len(list(forward_action_history)),
            "skipped_no_motion_forward_count": count_reversal_skips(
                forward_action_history),
            "action_history": plan,
            "executor_arrived": None,
            "target_panorama_similarity_after": similarity,
            "vlm_node_revisit": None,
            "policy_input_contract": "rgb_only_v1",
            "privileged_inputs_used": [],
        }
        try:
            verdict = self.vlm_harness.judge_node_revisit_rgb_only(
                target_node_id=target_node_id,
                target_node_views=target_views,
                current_views=current_views,
                reversal_action_history=plan)
        except RuntimeError as exc:
            attempt["selection_error"] = str(exc)
            self.sim.emit_evaluation_event(
                "action_reversal_backtrack_finished", {
                    "target_node_id": target_node_id,
                    "same_place": None, "similarity": similarity,
                    "error": str(exc)})
            return RGBOnlyBacktrackResult(
                False, target_node_id, None, heading, step, [attempt],
                "action_reversal_vlm_error")
        attempt["vlm_node_revisit"] = verdict
        same_place = bool(verdict.get("same_place"))
        self.sim.emit_evaluation_event("action_reversal_backtrack_finished", {
            "target_node_id": target_node_id,
            "same_place": same_place,
            "confidence": float(verdict.get("confidence", 0.0)),
            "similarity": similarity,
        })
        if not same_place:
            return RGBOnlyBacktrackResult(
                False, target_node_id, None, heading, step, [attempt],
                "action_reversal_return_rejected_by_vlm")
        if not create_revisit_node:
            return RGBOnlyBacktrackResult(
                True, target_node_id, target_node_id, heading, step, [attempt],
                "action_reversal_return_confirmed")

        completion_views = observe_eight_rgb(self.sim)
        node, edge = self.graph_memory.add_navigation_stop_node(
            position_xyz=None, base_yaw_rad=None, global_step=step,
            six_views=current_views, six_depths=None,
            sub_instruction={
                "navigation_instruction": instruction,
                "semantic_spatial_target": "stored node panorama",
            },
            action_history=plan,
            arrival_signal="action_reversal_node_revisit_confirmed",
            completion_views=completion_views,
            metadata={
                "policy_input_contract": "rgb_only_v1",
                "rgb_revisit_target_node_id": target_node_id,
                "rgb_panorama_similarity": similarity,
                "backtrack_method": self.method,
                "vlm_node_revisit": verdict,
            },
            edge_kind="rgb_only_action_reversal_backtrack",
            edge_metadata={
                "policy_input_contract": "rgb_only_v1",
                "backtrack_method": self.method,
            })
        self.graph_memory.add_loop_closure_edge(
            target_node_id, node.node_id, metadata={
                "verification": "vlm_node_revisit_confirmation",
                "similarity": similarity,
                "vlm_confidence": float(verdict.get("confidence", 0.0)),
                "policy_input_contract": "rgb_only_v1",
            })
        attempt["recovered_node_id"] = node.node_id
        attempt["recovery_edge_id"] = edge.edge_id
        return RGBOnlyBacktrackResult(
            True, target_node_id, node.node_id, heading, step, [attempt],
            "action_reversal_return_confirmed_node_recorded")


class RGBOnlyInstructionSequenceExplorationStrategy:
    """Ordered semantic navigation with an enforced RGB-only data plane."""

    mode = "instruction-sequence-recovery-rgb-only"

    def __init__(
            self, sim, sub_instructions, point_selector,
            point_navigation_executor, graph_memory, vlm_harness,
            segmenter, output_dir, rendered=None, motion_log=None,
            video_composer=None, max_exploration_hops=30,
            max_blocked_directions_per_node=5,
            minimum_classification_confidence=0.5,
            recovery_backtrack_attempts_per_hop=2,
            recovery_minimum_visual_similarity=0.75,
            full_instruction=None, views=8, scan_step_deg=15.0,
            backtrack_method="action-reversal", turn_step_deg=None,
            in_place_turn_on_empty_gate=True):
        if backtrack_method not in BACKTRACK_METHODS:
            raise ValueError(
                f"unknown backtrack_method {backtrack_method!r}; "
                f"expected one of {BACKTRACK_METHODS}")
        self.sim = require_rgb_only_policy_sim(sim)
        self.sub_instructions = list(sub_instructions)
        self.point_selector = point_selector
        self.executor = point_navigation_executor
        self.graph_memory = graph_memory
        self.vlm_harness = vlm_harness
        self.segmenter = segmenter
        self.output_dir = Path(output_dir)
        self.rendered = rendered if rendered is not None else []
        self.motion_log = motion_log if motion_log is not None else []
        self.video_composer = video_composer
        self.full_instruction = str(full_instruction or "")
        self.views = int(views)
        self.scan_step = math.radians(float(scan_step_deg))
        self.max_exploration_hops = int(max_exploration_hops)
        self.completion_judge = RGBOnlyNodeTransitionInstructionCompletionJudge(
            vlm_harness, graph_memory, self.sub_instructions,
            minimum_confidence=minimum_classification_confidence)
        self.state = InstructionSequenceStateMachine(
            self.sub_instructions, graph_memory.nodes[-1].node_id,
            max_blocked_directions_per_node,
            minimum_classification_confidence,
            max_consecutive_on_route_unknowns=0)
        self.backtracker = RGBOnlyGraphBacktracker(
            sim, graph_memory, segmenter, vlm_harness,
            point_navigation_executor, views=6,
            scan_step_deg=scan_step_deg,
            max_attempts=recovery_backtrack_attempts_per_hop,
            minimum_visual_similarity=recovery_minimum_visual_similarity)
        self.backtrack_method = str(backtrack_method)
        # Discrete turning keeps the scan step equal to the simulator turn
        # step (the CLI enforces it), so the scan step is the default here.
        self.turn_step_deg = float(
            scan_step_deg if turn_step_deg is None else turn_step_deg)
        self.action_reversal_backtracker = RGBOnlyActionReversalBacktracker(
            sim, graph_memory, vlm_harness,
            turn_step_deg=self.turn_step_deg,
            video_composer=video_composer,
            full_instruction=self.full_instruction)
        self.in_place_turn_on_empty_gate = bool(in_place_turn_on_empty_gate)
        if self.in_place_turn_on_empty_gate:
            for sector in TURN_GATE_CENTER_DEG:
                plan_in_place_turn(sector, self.turn_step_deg)

    def _emit_video(self, rgb, instruction, target_index, phase):
        if self.video_composer is None:
            return
        bgr = np.ascontiguousarray(np.asarray(rgb, np.uint8)[..., :3][..., ::-1])
        self.video_composer.emit(
            bgr, instruction, target_index, phase,
            full_instruction=self.full_instruction or instruction,
            sub_instruction=instruction)

    def _turn_in_place_to_gate_center(self, sector, hop_index, heading,
                                      global_step, turn_start_rgb,
                                      sub_instruction):
        """Rotate to the empty gate's centre; return the plan, frames, pose."""
        plan = plan_in_place_turn(sector, self.turn_step_deg)
        instruction = (
            f"in-place {sector} turn for "
            f"{sub_instruction.navigation_instruction!r}")
        self.sim.emit_evaluation_event("direction_gate_in_place_turn_started", {
            "sub_instruction_id": int(sub_instruction.sub_instruction_id),
            "sector": sector, "turn_command_count": len(plan),
            "target_index": int(hop_index),
        })
        frames = [turn_start_rgb]
        for record in plan:
            observations = self.sim.step(record["action"])
            heading = wrap_angle(
                heading + math.radians(record["commanded_turn_deg"]))
            global_step += 1
            frames.append(observations["rgb"])
            self._emit_video(observations["rgb"], instruction, hop_index,
                             record["phase"])
        self.sim.emit_evaluation_event("direction_gate_in_place_turn_finished", {
            "sub_instruction_id": int(sub_instruction.sub_instruction_id),
            "sector": sector, "target_index": int(hop_index),
        })
        return plan, frames, heading, global_step

    def _recover(self, target_node_id, heading, global_step, target_index,
                 action_history, *, create_revisit_node):
        if self.backtrack_method == "action-reversal":
            return self.action_reversal_backtracker.recover(
                target_node_id, heading, global_step, target_index,
                action_history, create_revisit_node=create_revisit_node)
        return self.backtracker.recover(
            target_node_id, heading, global_step, target_index,
            action_history)

    def _finish_hop_at_stop_node(
            self, *, hop_index, sub_instruction, stage, action_history,
            arrival_signal, edge_kind, node_metadata, edge_metadata,
            selected_heading, edge_keyframes, edge_keyframe_records, record,
            heading, global_step, recoveries):
        """Record the node the agent stands at, judge it and advance state.

        Shared by a forward point-navigation arrival and the direction-gate
        in-place turn.  ``record`` and ``recoveries`` are updated in place.
        Returns the loop variables the caller must adopt plus ``end_reason``
        (``None`` unless the episode must stop here).
        """
        stop_rgbs = observe_six_rgb(self.sim)
        completion_views = observe_eight_rgb(self.sim)
        stop_node, stop_edge = self.graph_memory.add_navigation_stop_node(
            position_xyz=None, base_yaw_rad=None,
            global_step=global_step, six_views=stop_rgbs,
            six_depths=None, sub_instruction=sub_instruction,
            action_history=action_history,
            arrival_signal=arrival_signal,
            completion_views=completion_views,
            metadata={"policy_input_contract": "rgb_only_v1", **node_metadata},
            edge_kind=edge_kind,
            edge_metadata={
                **edge_metadata,
                "edge_keyframes": edge_keyframe_records,
                "policy_input_contract": "rgb_only_v1",
            })
        completion = self.completion_judge.judge(
            stop_node, self.state.expected_sub_instruction_id,
            completion_views, edge_keyframes).to_dict()
        classification = {
            "node_id": stop_node.node_id,
            "belongs_to_sequence": bool(
                completion["instruction_completed"]),
            "matched_sub_instruction_id": (
                self.state.expected_sub_instruction_id
                if completion["instruction_completed"] else -1),
            "expected_sub_instruction_id": (
                self.state.expected_sub_instruction_id),
            "confidence": float(completion["confidence"]),
            "reason": completion["reason"],
            "visual_evidence": completion["visual_evidence"],
            "unknown_disposition": None,
            "active_form": str(stage.get("form", "")).upper(),
        }
        directive = self.state.observe(
            stop_node.node_id, classification, selected_heading)
        record.update({
            "navigation_graph_node_id": stop_node.node_id,
            "navigation_graph_edge_id": stop_edge.edge_id,
            "node_created_after_point_arrival": True,
            "instruction_completion": completion,
            "sub_instruction_sequence_classification": classification,
            "sequence_directive": directive.to_dict(),
            "sub_instruction_satisfied": bool(
                completion["instruction_completed"]),
        })
        outcome = {
            "heading": heading, "global_step": global_step,
            "previous_action_history": action_history,
            "backtracked": False, "end_reason": None,
        }
        if directive.action == "backtrack_and_block":
            outcome["backtracked"] = True
            backtrack_target = str(directive.backtrack_target_node_id)
            if self.backtrack_method == "action-reversal":
                # Reversal only undoes the edge that just ended at the
                # current node, so the target must be its predecessor.
                predecessor, _ = self.graph_memory.predecessor(
                    stop_node.node_id)
                if predecessor.node_id != backtrack_target:
                    raise RuntimeError(
                        "action reversal can only return along the last "
                        f"edge: target {backtrack_target} is not the "
                        f"predecessor {predecessor.node_id} of "
                        f"{stop_node.node_id}")
            recovery = self._recover(
                backtrack_target, heading, global_step,
                len(self.sub_instructions) + hop_index,
                action_history, create_revisit_node=True)
            record["sequence_recovery_backtrack"] = recovery.to_dict()
            record["backtrack_method"] = self.backtrack_method
            recoveries.append(dict(
                recovery.to_dict(), backtrack_method=self.backtrack_method,
                trigger="backtrack_and_block"))
            if not recovery.success:
                self.state.on_backtrack(False)
                outcome["end_reason"] = "rgb_only_sequence_recovery_failed"
                return outcome
            outcome["heading"] = recovery.final_action_heading_rad
            outcome["global_step"] = recovery.next_global_step
            self.state.on_backtrack(
                True, current_node_id=recovery.recovered_node_id)
            outcome["previous_action_history"] = (
                recovery.attempts[-1].get("action_history", [])
                if recovery.attempts else [])
        elif directive.action == "complete":
            outcome["end_reason"] = "instruction_sequence_complete"
        return outcome

    def run(self, initial_action_heading=0.0, initial_global_step=0):
        heading = float(initial_action_heading)
        global_step = int(initial_global_step)
        previous_action_history = []
        has_incoming_edge = False
        # Heading at the end of the last forward edge.  An in-place turn
        # changes ``heading`` but not where the agent came from, so the
        # incoming-direction exclusion keeps using this value.
        incoming_heading = None
        records = []
        recoveries = []
        selected_headings = []
        end_reason = "max_sequence_exploration_hops"

        for hop_index in range(self.max_exploration_hops):
            if self.state.complete:
                end_reason = "instruction_sequence_complete"
                break
            if self.state.terminated_reason:
                end_reason = self.state.terminated_reason
                break
            sub_instruction = self.state.active_sub_instruction
            stage = sub_instruction.to_stage_dict()
            in_place_turn = self.state.in_place_turn_record(
                sub_instruction.sub_instruction_id)
            if in_place_turn is not None:
                stage["metadata"] = dict(
                    stage.get("metadata", {}) or {},
                    in_place_turn_executed=dict(in_place_turn, active=True))
            back_heading = (
                wrap_angle(incoming_heading + math.pi)
                if has_incoming_edge and incoming_heading is not None
                else None)
            # The selector turns the agent toward the chosen view before
            # returning; remember where its commands start in motion_log and
            # what the agent saw before turning so the edge can include them.
            motion_log_start = len(self.motion_log)
            turn_start_rgb = observe(self.sim)
            hop_start_heading = heading
            hop_start_node_id = self.graph_memory.nodes[-1].node_id
            hop_start_previous_action_history = previous_action_history
            hop_start_has_incoming_edge = has_incoming_edge
            hop_start_incoming_heading = incoming_heading
            try:
                selection = self.point_selector.select(PointSelectionRequest(
                    sim=self.sim, position=None, yaw=heading, stage=stage,
                    target_index=hop_index, rendered=self.rendered,
                    motion_log=self.motion_log,
                    position_history=[],
                    previous_action_history=previous_action_history,
                    back_yaw=back_heading,
                    blocked_yaws=self.state.blocked_yaws(),
                    reference_path=None, reference_path_index=0,
                    minimum_progress_distance_m=0.0,
                    maximum_initial_geodesic_m=None,
                    semantic_reference_rgb=None,
                    policy_input_contract="rgb_only_v1"))
            except DirectionGateEmptyError as exc:
                # An explicit turn clause has no floor-bearing view on its
                # commanded side.  Face that side once, let the judge decide
                # whether the turn is done, and continue; a second empty gate
                # for the same clause (including the post-turn forward gate)
                # ends the episode exactly like any other empty candidate set.
                if (not self.in_place_turn_on_empty_gate or
                        in_place_turn is not None or
                        exc.sector not in TURN_GATE_CENTER_DEG):
                    end_reason = f"vlm_selection_failed: {exc}"
                    break
                turn_plan, turn_frames, heading, global_step = (
                    self._turn_in_place_to_gate_center(
                        exc.sector, hop_index, heading, global_step,
                        turn_start_rgb, sub_instruction))
                edge_keyframes, edge_keyframe_records = in_place_turn_keyframes(
                    self.output_dir, hop_index, turn_frames, turn_plan)
                in_place_turn = {
                    "sector": exc.sector,
                    "target_index": hop_index,
                    "turn_command_count": len(turn_plan),
                    "pre_turn_heading_rad": hop_start_heading,
                    "post_turn_heading_rad": heading,
                    "trigger": str(exc),
                }
                self.state.record_in_place_turn(
                    sub_instruction.sub_instruction_id, in_place_turn)
                selected_headings.append(heading)
                record = {
                    "target_index": hop_index,
                    "strategy": self.mode,
                    "sub_instruction": sub_instruction.to_dict(),
                    "in_place_turn": in_place_turn,
                    "selected_action_heading_rad": heading,
                    "action_history": turn_plan,
                    "point_target_arrived": False,
                    "arrived": False,
                    "signal": IN_PLACE_TURN_PHASE,
                    "end_reason": IN_PLACE_TURN_PHASE,
                    "steps": turn_plan,
                    "policy_input_contract": "rgb_only_v1",
                    "privileged_inputs_used": [],
                }
                records.append(record)
                outcome = self._finish_hop_at_stop_node(
                    hop_index=hop_index, sub_instruction=sub_instruction,
                    stage=stage, action_history=turn_plan,
                    arrival_signal=IN_PLACE_TURN_PHASE,
                    edge_kind="instruction_sequence_rgb_only_in_place_turn",
                    node_metadata={
                        "point_target_arrived": False,
                        "in_place_turn": in_place_turn,
                    },
                    edge_metadata={"in_place_turn": in_place_turn},
                    selected_heading=heading, edge_keyframes=edge_keyframes,
                    edge_keyframe_records=edge_keyframe_records,
                    record=record, heading=heading, global_step=global_step,
                    recoveries=recoveries)
                heading = outcome["heading"]
                global_step = outcome["global_step"]
                previous_action_history = outcome["previous_action_history"]
                has_incoming_edge = True
                if outcome["backtracked"]:
                    incoming_heading = heading
                if outcome["end_reason"] is not None:
                    end_reason = outcome["end_reason"]
                    break
                continue
            except RuntimeError as exc:
                # Same end_reason prefix as the legacy strategy: the evaluator
                # categorises it (no_floor_bearing_candidate, ...) and still
                # detects provider-exhaustion text inside it. Fatal provider
                # errors are not RuntimeError and keep propagating.
                end_reason = f"vlm_selection_failed: {exc}"
                break
            chosen = selection.chosen
            if chosen is None:
                end_reason = "rgb_vlm_selection_returned_no_ground_point"
                break
            selected_heading = float(chosen["yaw"])
            selected_headings.append(selected_heading)
            turn_actions = turn_to_selected_target_actions(
                self.motion_log, motion_log_start, hop_index)
            self.sim.emit_evaluation_event("point_selected", {
                "target_index": hop_index,
                "selected_point_xy": np.asarray(chosen["point"]).tolist(),
                "selected_rgb_shape": list(np.asarray(chosen["rgb"]).shape[:2]),
            })
            navigation = execute_point_navigation(
                self.executor, PointNavigationRequest(
                    rgb=chosen["rgb"],
                    selected_point_xy=chosen["point"],
                    selectable_mask=chosen["target_mask"],
                    ground_mask=chosen["mask"], yaw=selected_heading,
                    position_history=[],
                    instruction=sub_instruction.navigation_instruction,
                    full_instruction=self.full_instruction,
                    sub_instruction=sub_instruction.navigation_instruction,
                    instruction_form=str(stage.get("form", "")).upper() or None,
                    semantic_target=sub_instruction.semantic_spatial_target,
                    target_index=hop_index,
                    stage_count=len(self.sub_instructions),
                    global_step=global_step,
                    allow_initial_near_field_arrival=False,
                    policy_input_contract="rgb_only_v1"))
            heading = navigation.final_yaw
            global_step = navigation.next_global_step
            # One merged list for the hop record, the graph edge and the next
            # selection: the post-run edge integrity check compares them.
            action_history = turn_actions + list(navigation.action_history)
            edge_keyframes = list(navigation.edge_keyframes)
            edge_keyframe_records = list(
                navigation.record.get("edge_keyframes", []))
            if turn_actions:
                edge_keyframes.insert(0, turn_start_rgb)
                edge_keyframe_records = prepend_turn_start_keyframe(
                    self.output_dir, hop_index, turn_start_rgb,
                    len(turn_actions), edge_keyframe_records)
            record: dict[str, Any] = {
                "target_index": hop_index,
                "strategy": self.mode,
                "sub_instruction": sub_instruction.to_dict(),
                "selected_action_heading_rad": selected_heading,
                "selected_relative_view_yaw_rad": float(
                    chosen.get("relative_yaw_rad", 0.0)),
                "selected_point_xy": np.asarray(chosen["point"]).tolist(),
                "selection": chosen.get("vlm_selection", {}),
                "executor": navigation.record,
                "action_history": action_history,
                "point_target_arrived": navigation.arrived,
                "arrived": navigation.arrived,
                "signal": navigation.signal,
                "end_reason": navigation.end_reason,
                "steps": navigation.record.get("steps", action_history),
                "policy_input_contract": "rgb_only_v1",
                "privileged_inputs_used": [],
            }
            records.append(record)
            self.sim.emit_evaluation_event("point_navigation_stopped", {
                "target_index": hop_index,
                "policy_declared_arrival": bool(navigation.arrived),
                "end_reason": navigation.end_reason,
            })

            if not navigation.arrived:
                action_reversal = self.backtrack_method == "action-reversal"
                # Action reversal undoes exactly this hop, so it returns to
                # the node the hop launched from; the visual method searches
                # for the last verified node instead.  A failed hop never
                # becomes a node, so neither path records a revisit node.
                recovery = self._recover(
                    (hop_start_node_id if action_reversal
                     else self.state.last_verified_node_id),
                    heading, global_step,
                    len(self.sub_instructions) + hop_index,
                    action_history, create_revisit_node=False)
                record["physical_failure_recovery"] = recovery.to_dict()
                record["backtrack_method"] = self.backtrack_method
                recoveries.append(dict(
                    recovery.to_dict(), backtrack_method=self.backtrack_method,
                    trigger="physical_failure"))
                if not recovery.success:
                    end_reason = "rgb_only_physical_failure_recovery_failed"
                    break
                heading = recovery.final_action_heading_rad
                global_step = recovery.next_global_step
                self.state.block_failed_physical_direction(selected_heading)
                if action_reversal:
                    # The agent stands where the hop began with the same
                    # heading; the only new knowledge is the blocked ray.
                    if abs(wrap_angle(heading - hop_start_heading)) > 1e-6:
                        raise RuntimeError(
                            "action reversal did not restore the hop start "
                            f"heading: {heading} vs {hop_start_heading}")
                    previous_action_history = hop_start_previous_action_history
                    has_incoming_edge = hop_start_has_incoming_edge
                    incoming_heading = hop_start_incoming_heading
                else:
                    previous_action_history = (
                        recovery.attempts[-1].get("action_history", [])
                        if recovery.attempts else [])
                    has_incoming_edge = True
                    incoming_heading = heading
                continue

            outcome = self._finish_hop_at_stop_node(
                hop_index=hop_index, sub_instruction=sub_instruction,
                stage=stage, action_history=action_history,
                arrival_signal=navigation.signal,
                edge_kind="instruction_sequence_rgb_only",
                node_metadata={"point_target_arrived": True},
                edge_metadata={
                    "selected_action_heading_rad": selected_heading,
                    "point_selection_review": chosen.get("vlm_selection", {}),
                },
                selected_heading=selected_heading,
                edge_keyframes=edge_keyframes,
                edge_keyframe_records=edge_keyframe_records, record=record,
                heading=heading, global_step=global_step,
                recoveries=recoveries)
            heading = outcome["heading"]
            global_step = outcome["global_step"]
            previous_action_history = outcome["previous_action_history"]
            has_incoming_edge = True
            incoming_heading = heading
            if outcome["end_reason"] is not None:
                end_reason = outcome["end_reason"]
                break

        return InstructionSequenceExplorationResult(
            success=self.state.complete, end_reason=end_reason,
            completed_sub_instructions=self.state.cursor,
            expected_sub_instruction_id=self.state.expected_sub_instruction_id,
            exploration_hops=len(records),
            recovery_backtracks=self.state.recovery_backtracks,
            final_yaw=heading, next_global_step=global_step,
            traveled_distance_m=0.0,
            selected_yaws_rad=selected_headings,
            records=records, recovery_records=recoveries,
            state=self.state.snapshot())
