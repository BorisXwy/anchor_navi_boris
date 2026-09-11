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

from instruction_completion_judge import (
    RGBOnlyNodeTransitionInstructionCompletionJudge,
)
from instruction_sequence_exploration import (
    InstructionSequenceExplorationResult, InstructionSequenceStateMachine,
)
from navigation_graph_memory import CompactVisualEmbedder
from point_navigation_executor import (
    PointNavigationRequest, execute_point_navigation,
)
from point_selectors import (
    PointSelectionRequest, continuous_turn, observe_eight_rgb,
    observe_six_rgb, select_ground_point, targetable_ground_mask, wrap_angle,
)
from rgb_only_runtime import require_rgb_only_policy_sim


def _cosine(left, right):
    left = np.asarray(left, np.float32)
    right = np.asarray(right, np.float32)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-8 else 0.0


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
            full_instruction=None, views=8, scan_step_deg=15.0):
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

    def run(self, initial_action_heading=0.0, initial_global_step=0):
        heading = float(initial_action_heading)
        global_step = int(initial_global_step)
        previous_action_history = []
        has_incoming_edge = False
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
            back_heading = wrap_angle(heading + math.pi) if has_incoming_edge else None
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
                    semantic_target=sub_instruction.semantic_spatial_target,
                    target_index=hop_index,
                    stage_count=len(self.sub_instructions),
                    global_step=global_step,
                    allow_initial_near_field_arrival=False,
                    policy_input_contract="rgb_only_v1"))
            heading = navigation.final_yaw
            global_step = navigation.next_global_step
            action_history = navigation.action_history
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
                recovery = self.backtracker.recover(
                    self.state.last_verified_node_id, heading, global_step,
                    len(self.sub_instructions) + hop_index,
                    action_history)
                record["physical_failure_recovery"] = recovery.to_dict()
                recoveries.append(recovery.to_dict())
                if not recovery.success:
                    end_reason = "rgb_only_physical_failure_recovery_failed"
                    break
                heading = recovery.final_action_heading_rad
                global_step = recovery.next_global_step
                self.state.block_failed_physical_direction(selected_heading)
                previous_action_history = (
                    recovery.attempts[-1].get("action_history", [])
                    if recovery.attempts else [])
                has_incoming_edge = True
                continue

            stop_rgbs = observe_six_rgb(self.sim)
            completion_views = observe_eight_rgb(self.sim)
            stop_node, stop_edge = self.graph_memory.add_navigation_stop_node(
                position_xyz=None, base_yaw_rad=None,
                global_step=global_step, six_views=stop_rgbs,
                six_depths=None, sub_instruction=sub_instruction,
                action_history=action_history,
                arrival_signal=navigation.signal,
                completion_views=completion_views,
                metadata={
                    "policy_input_contract": "rgb_only_v1",
                    "point_target_arrived": True,
                }, edge_kind="instruction_sequence_rgb_only",
                edge_metadata={
                    "selected_action_heading_rad": selected_heading,
                    "point_selection_review": chosen.get("vlm_selection", {}),
                    "edge_keyframes": navigation.record.get(
                        "edge_keyframes", []),
                    "policy_input_contract": "rgb_only_v1",
                })
            completion = self.completion_judge.judge(
                stop_node, self.state.expected_sub_instruction_id,
                completion_views, navigation.edge_keyframes).to_dict()
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
            previous_action_history = action_history
            has_incoming_edge = True

            if directive.action == "backtrack_and_block":
                recovery = self.backtracker.recover(
                    directive.backtrack_target_node_id, heading, global_step,
                    len(self.sub_instructions) + hop_index,
                    action_history)
                record["sequence_recovery_backtrack"] = recovery.to_dict()
                recoveries.append(recovery.to_dict())
                if not recovery.success:
                    self.state.on_backtrack(False)
                    end_reason = "rgb_only_sequence_recovery_failed"
                    break
                heading = recovery.final_action_heading_rad
                global_step = recovery.next_global_step
                self.state.on_backtrack(
                    True, current_node_id=recovery.recovered_node_id)
                previous_action_history = (
                    recovery.attempts[-1].get("action_history", [])
                    if recovery.attempts else [])
            elif directive.action == "complete":
                end_reason = "instruction_sequence_complete"
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
