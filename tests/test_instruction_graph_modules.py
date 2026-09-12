#!/usr/bin/env python3

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from instruction_decomposer import InstructionDecomposer, SubInstruction
from instruction_completion_judge import (
    NodeTransitionInstructionCompletionJudge,
    current_rgb_turn_landmark_alignment_supported,
    generic_near_stop_override_eligible,
    landmark_detection_aliases,
)
from instruction_sequence_exploration import (
    LINEAR_SAME_STAGE_CONTINUATION_HALF_WIDTH_DEG,
    MAX_PORTAL_THRESHOLD_ROUTE_RETRIES,
    InstructionSequenceStateMachine, bare_turn_delta_rad,
    bare_turn_preview_yaw_delta_rad,
    bare_turn_route_setup_stage,
    chained_stop_wait_eligible,
    chained_near_stop_evidence_supported,
    compound_turn_continuation_required,
    directional_towards_route_setup_cap_m,
    compound_terminal_extent_route,
    circumnavigation_continuation_cap_m,
    exit_threshold_portal_continuation_required,
    fixed_recovery_corridor_eligible,
    incoming_route_yaw_rad,
    inherits_incoming_route_corridor,
    rejected_ray_retry_reopens_incoming,
    explicit_linear_segment_cap_m,
    local_portal_endpoint_cap_m,
    local_landmark_stop_relation,
    portal_recovery_hemisphere_eligible,
    portal_recovery_corridor_active,
    selection_geodesic_cap_m,
    selected_point_ray_yaw_rad,
    form_requires_physical_progress, infer_unknown_disposition,
    named_landmark_route_setup_stage,
    named_landmark_route_setup_cap_m,
    selection_contact_sheet,
    terminal_facing_landmark,
)
from navigation_graph_memory import NavigationGraphMemory
from semantic_detector import extract_detection_queries
from node_backtracking import (
    backtrack_segment_travel_budget,
    NodeBacktrackingPointSelector, NodeRevisitMatcher,
    recovery_executor_endpoint, reverse_route_breadcrumb,
)
from sub_instruction_node_matcher import SubInstructionNodeMatcher
from vlm_harness import (
    ambiguous_landmark_executable_view,
    ambiguous_landmark_spelling_aliases,
    between_stage_route_integrity_supported,
    between_gap_lateral_bracket_supported,
    circumnavigation_stage_route_integrity_supported,
    compound_terminal_extent_portal_stage,
    compound_turn_portal_threshold_supported,
    compound_turn_requires_semantic_endpoint,
    explicit_turn_selection_supported,
    following_landmark_route_commitment_supported,
    landmark_center_refined_yaw_deg,
    HeuristicBackend, NavigationVLMHarness,
    terminal_extent_rgb_completion_supported,
    terminal_extent_rgb_rear_clear_supported,
    terminal_relation_detection_supported,
    terminal_relation_rgb_consensus_supported,
    turn_direction_motion_supported,
    under_relation_multiview_supported,
)


class FakeHarness:
    def decompose_instruction(self, instruction):
        return [{
            "stage_id": 9,
            "navigation_instruction": "exit the room through the doorway",
            "landmark": "doorway",
            "completion_cue": "outside the room",
            "semantic_spatial_target": "floor beyond the doorway",
            "spatial_relation": "beyond doorway",
            "visual_arrival_evidence": "door frame is behind the camera",
            "forbidden_target": "floor inside the room",
        }]


class CompoundActionHarness:
    def decompose_instruction(self, instruction):
        return [{
            "stage_id": 4,
            "navigation_instruction": (
                "Walk into the kitchen and walk along the counter"),
            "landmark": "kitchen and counter",
            "completion_cue": "inside the kitchen beside the counter",
            "semantic_spatial_target": "kitchen floor along the counter",
            "spatial_relation": "inside and along",
            "visual_arrival_evidence": "the combined route is complete",
            "forbidden_target": "floor outside the kitchen",
        }]


class FakeSemanticExtractor:
    name = "fake_semantics"

    def extract(self, six_views, six_depths=None, sub_instruction=None):
        return {
            # Queries must not count as observations in the matcher.
            "queries": ["doorway", "stairs"],
            "labels": ["doorway"],
            "views": [{"view_index": 0, "detections": [{"label": "doorway"}]}],
        }


class FakeFloorSegmenter:
    def __call__(self, rgb, depth=None):
        return np.ones(rgb.shape[:2], bool), []


class UnknownCompletionEvidenceBackend:
    def generate_json(self, prompt, images, schema):
        result = {
            "status": "unknown", "confidence": 0.6,
            "reason": "completion is not established",
            "visual_evidence": "ambiguous edge",
        }
        if "partial_extent_evidence" in schema.get("properties", {}):
            result["partial_extent_evidence"] = {
                "starts_at_stair_base": False,
                "continuous_ascent": False,
                "current_steps_below": False,
                "current_steps_above": False,
            }
        return result


class RearPassClaimBackend:
    def generate_json(self, prompt, images, schema):
        return {
            "status": "completed", "confidence": 0.85,
            "reason": "target claimed in rear", "visual_evidence": "rear",
            "directional_evidence": {
                "reference_previous_sectors": ["front"],
                "reference_current_sectors": ["rear"],
                "reference_previous_view_indices": [0],
                "reference_current_view_indices": [4],
                "same_instance_confident": True,
                "multiple_similar_instances": False,
                "rear_sector_evidence": True,
                "front_sector_contradiction": False,
                "temporal_relation_change": True,
            },
        }


class ThreeWayOnRouteBackend:
    def generate_json(self, prompt, images, schema):
        return {
            "status": "on_route", "confidence": 0.82,
            "reason": "the portal grows nearer but is not crossed",
            "visual_evidence": "portal front-to-near transition across keyframes",
            "progress_evidence": {
                "completion_boundary_observed": False,
                "instruction_consistent_progress": True,
                "contradiction_observed": False,
            },
        }


class ThreeWayContradictoryClaimBackend:
    def generate_json(self, prompt, images, schema):
        return {
            "status": "arrived", "confidence": 0.9,
            "reason": "model claim conflicts with extracted reverse evidence",
            "visual_evidence": "destination moves behind-to-front in reverse order",
            "progress_evidence": {
                "completion_boundary_observed": True,
                "instruction_consistent_progress": True,
                "contradiction_observed": True,
            },
        }


class BidirectionalReverseBackend:
    def generate_json(self, prompt, images, schema):
        return {
            "observed_status": "on_route",
            "reversed_status": "arrived",
            "direction_fit": "reversed",
            "confidence": 0.88,
            "reason": "only the reverse temporal order approaches the target",
            "visual_evidence": "the observed order moves away from the landmark",
        }


class BinaryPartialClaimBackend:
    def generate_json(self, prompt, images, schema):
        return {
            "status": "completed", "confidence": 0.9,
            "reason": "the agent is moving toward the doorway",
            "visual_evidence": "doorway is closer but remains ahead",
            "completion_evidence": {
                "completion_boundary_observed": False,
                "temporal_relation_change_observed": True,
                "contradiction_observed": False,
            },
        }


class StructuredReverseClaimBackend:
    def generate_json(self, prompt, images, schema):
        return {
            "status": "completed", "confidence": 0.9,
            "endpoint_evidence": {
                "previous_target_state": "unsatisfied",
                "current_target_state": "satisfied",
                "current_completion_cue": "satisfied",
                "reference_previous_sectors": ["rear"],
                "reference_current_sectors": ["front"],
            },
            "temporal_evidence": {
                "semantic_order": "reversed",
                "boundary_event": "observed", "keyframe_support": True,
                "same_reference_instance": "yes",
                "reverse_transition_observed": True,
            },
            "motion_evidence": {
                "instruction_motion_fit": "contradicts",
                "stationary_or_blocked": False,
            },
            "reason": "an intentionally contradictory model claim",
            "visual_evidence": "the temporal order is reversed",
        }


class StructuredCompletionThenReverseRolesBackend:
    def generate_json(self, prompt, images, schema):
        if "node_x_role" in schema.get("properties", {}):
            return {
                "node_x_role": "completion_target",
                "node_y_role": "source_context",
                "confidence": 0.9,
                "reason": "one endpoint is the target and the other is source",
                "visual_evidence": "chronology-hidden endpoint evidence",
            }
        return {
            "status": "completed", "confidence": 0.9,
            "endpoint_evidence": {
                "previous_target_state": "unsatisfied",
                "current_target_state": "satisfied",
                "current_completion_cue": "satisfied",
                "reference_previous_sectors": ["front"],
                "reference_current_sectors": ["rear"],
            },
            "temporal_evidence": {
                "semantic_order": "instructed",
                "boundary_event": "observed",
                "keyframe_support": True,
                "same_reference_instance": "yes",
                "reverse_transition_observed": False,
            },
            "motion_evidence": {
                "instruction_motion_fit": "supports",
                "stationary_or_blocked": False,
            },
            "reason": "primary pass claims completion",
            "visual_evidence": "primary temporal evidence",
        }


class StructuredAlternatingConsensusBackend(
        StructuredCompletionThenReverseRolesBackend):
    def __init__(self):
        self.primary_call_count = 0

    def generate_json(self, prompt, images, schema):
        if "node_x_role" in schema.get("properties", {}):
            return super().generate_json(prompt, images, schema)
        result = super().generate_json(prompt, images, schema)
        call_index = self.primary_call_count
        self.primary_call_count += 1
        if call_index == 1:
            result.update({
                "status": "unknown", "confidence": 0.6,
                "endpoint_evidence": {
                    **result["endpoint_evidence"],
                    "current_target_state": "partial",
                    "current_completion_cue": "partial",
                },
                "temporal_evidence": {
                    **result["temporal_evidence"],
                    "semantic_order": "ambiguous",
                    "boundary_event": "not_observed",
                    "keyframe_support": False,
                    "same_reference_instance": "ambiguous",
                },
            })
        return result


class StructuredPartialTurnLandmarkBackend:
    def generate_json(self, prompt, images, schema):
        return {
            "status": "unknown", "confidence": 0.6,
            "endpoint_evidence": {
                "previous_target_state": "unsatisfied",
                "current_target_state": "partial",
                "current_completion_cue": "partial",
                "reference_previous_sectors": [],
                "reference_current_sectors": ["front", "rear_left"],
            },
            "temporal_evidence": {
                "semantic_order": "ambiguous",
                "boundary_event": "not_observed",
                "keyframe_support": True,
                "same_reference_instance": "ambiguous",
                "reverse_transition_observed": False,
            },
            "motion_evidence": {
                "instruction_motion_fit": "neutral",
                "stationary_or_blocked": False,
            },
            "reason": "landmark is not centered and remains ambiguous",
            "visual_evidence": "landmark appears in both front and rear-side sectors",
        }


class StructuredRearOnlyCircumnavigationBackend:
    def generate_json(self, prompt, images, schema):
        return {
            "status": "unknown", "confidence": 0.6,
            "endpoint_evidence": {
                "previous_target_state": "unsatisfied",
                "current_target_state": "satisfied",
                "current_completion_cue": "satisfied",
                "reference_previous_sectors": ["rear", "rear_left"],
                # A long couch can span one forward overlap view even after
                # the backside endpoint. Rear presence plus real motion is
                # the project-level eight-view criterion.
                "reference_current_sectors": [
                    "rear", "rear_left", "rear_right", "front_right"],
            },
            "temporal_evidence": {
                "semantic_order": "instructed",
                "boundary_event": "not_observed",
                "keyframe_support": True,
                "same_reference_instance": "yes",
                "reverse_transition_observed": False,
            },
            "motion_evidence": {
                "instruction_motion_fit": "neutral",
                "stationary_or_blocked": False,
            },
            "reason": "couch is rear-visible with one overlap sector",
            "visual_evidence": "rear couch endpoint with front-right overlap",
        }


class StructuredPartialExitCrossingBackend:
    def generate_json(self, prompt, images, schema):
        return {
            "status": "unknown", "confidence": 0.6,
            "endpoint_evidence": {
                "previous_target_state": "unsatisfied",
                "current_target_state": "partial",
                "current_completion_cue": "partial",
                "reference_previous_sectors": ["front"],
                "reference_current_sectors": ["rear", "front_left"],
            },
            "temporal_evidence": {
                "semantic_order": "instructed",
                "boundary_event": "not_observed",
                "keyframe_support": True,
                "same_reference_instance": "yes",
                "reverse_transition_observed": False,
            },
            "motion_evidence": {
                "instruction_motion_fit": "supports",
                "stationary_or_blocked": False,
            },
            "reason": "ordered doorway crossing but overlapping endpoint views",
            "visual_evidence": "door is rear-visible after supported keyframes",
        }


class StructuredPartialEnterRegionBackend:
    def generate_json(self, prompt, images, schema):
        return {
            "status": "unknown", "confidence": 0.6,
            "endpoint_evidence": {
                "previous_target_state": "unsatisfied",
                "current_target_state": "partial",
                "current_completion_cue": "partial",
                "reference_previous_sectors": [],
                "reference_current_sectors": ["front", "front_left"],
            },
            "temporal_evidence": {
                "semantic_order": "instructed",
                "boundary_event": "not_observed",
                "keyframe_support": True,
                "same_reference_instance": "yes",
                "reverse_transition_observed": False,
            },
            "motion_evidence": {
                "instruction_motion_fit": "supports",
                "stationary_or_blocked": False,
            },
            "reason": "destination is ahead but the requested relation is partial",
            "visual_evidence": "the camera is not yet inside the named region",
        }


class StructuredSatisfiedEnterRegionBackend:
    def generate_json(self, prompt, images, schema):
        return {
            "status": "completed", "confidence": 0.9,
            "endpoint_evidence": {
                "previous_target_state": "unsatisfied",
                "current_target_state": "satisfied",
                "current_completion_cue": "satisfied",
                "reference_previous_sectors": ["front"],
                "reference_current_sectors": ["rear", "front"],
            },
            "temporal_evidence": {
                "semantic_order": "instructed",
                "boundary_event": "observed",
                "keyframe_support": True,
                "same_reference_instance": "yes",
                "reverse_transition_observed": False,
            },
            "motion_evidence": {
                "instruction_motion_fit": "supports",
                "stationary_or_blocked": False,
            },
            "reason": "the model claims entry into the qualified room",
            "visual_evidence": "the model claims the qualifier is visible",
        }


class StructuredRearAndFrontPassBackend:
    def generate_json(self, prompt, images, schema):
        return {
            "status": "unknown", "confidence": 0.65,
            "endpoint_evidence": {
                "previous_target_state": "unsatisfied",
                "current_target_state": "satisfied",
                "current_completion_cue": "satisfied",
                "reference_previous_sectors": ["front"],
                "reference_current_sectors": ["rear", "front"],
            },
            "temporal_evidence": {
                "semantic_order": "instructed",
                "boundary_event": "observed",
                "keyframe_support": True,
                "same_reference_instance": "yes",
                "reverse_transition_observed": False,
            },
            "motion_evidence": {
                "instruction_motion_fit": "supports",
                "stationary_or_blocked": False,
            },
            "reason": "a long landmark spans front and rear sectors",
            "visual_evidence": "ordered keyframes and a rear endpoint sector",
        }


class StructuredSatisfiedBetweenBackend:
    def generate_json(self, prompt, images, schema):
        return {
            "status": "completed", "confidence": 0.9,
            "endpoint_evidence": {
                "previous_target_state": "unsatisfied",
                "current_target_state": "satisfied",
                "current_completion_cue": "satisfied",
                "reference_previous_sectors": [
                    "front", "front_left", "front_right"],
                "reference_current_sectors": [
                    "rear", "rear_left", "rear_right"],
            },
            "temporal_evidence": {
                "semantic_order": "instructed",
                "boundary_event": "endpoint_inferred",
                "keyframe_support": True,
                "same_reference_instance": "yes",
                "reverse_transition_observed": False,
            },
            "motion_evidence": {
                "instruction_motion_fit": "supports",
                "stationary_or_blocked": False,
            },
            "reason": "model mistakes a shallow side bracket for the gap",
            "visual_evidence": "bar and chairs appear in rear-side views",
        }


class StructuredFalseTopVerticalBackend:
    def generate_json(self, prompt, images, schema):
        return {
            "status": "completed", "confidence": 0.9,
            "endpoint_evidence": {
                "previous_target_state": "unsatisfied",
                "current_target_state": "satisfied",
                "current_completion_cue": "satisfied",
                "reference_previous_sectors": ["front"],
                "reference_current_sectors": ["rear"],
            },
            "temporal_evidence": {
                "semantic_order": "instructed",
                "boundary_event": "observed",
                "keyframe_support": True,
                "same_reference_instance": "yes",
                "reverse_transition_observed": False,
            },
            "motion_evidence": {
                "instruction_motion_fit": "supports",
                "stationary_or_blocked": False,
            },
            "vertical_endpoint_evidence": {
                "current_level_landing": True,
                "remaining_stair_flight_visible": False,
                "endpoint_evidence_stable": True,
            },
            "reason": "model mistakes a middle landing for the top",
            "visual_evidence": "landing-like floor",
        }


class InstructionGraphModuleTest(unittest.TestCase):
    def test_landmark_spelling_ambiguity_is_explicit_not_silent(self):
        self.assertEqual(ambiguous_landmark_spelling_aliases(
            "large wooden manel", [
                "large wooden mantel", "large wooden panel", "fireplace"]),
            {"manel": ["mantel", "panel"]})
        self.assertEqual(ambiguous_landmark_spelling_aliases(
            "wooden mantel", ["wooden mantel", "fireplace mantel"]), {})

    def test_ambiguous_landmark_prefers_substantially_more_executable_view(self):
        candidates = [
            {"ground_fraction": 0.106},
            {"ground_fraction": 0.222},
            {"ground_fraction": 0.14},
        ]
        self.assertEqual(ambiguous_landmark_executable_view(
            0, [0, 1, 2], candidates), 1)
        self.assertEqual(ambiguous_landmark_executable_view(
            2, [0, 1, 2], candidates), 1)
        self.assertEqual(ambiguous_landmark_executable_view(
            1, [0, 1, 2], candidates), 1)

    def test_between_stage_requires_real_cumulative_gap_entry(self):
        self.assertFalse(between_stage_route_integrity_supported(
            current_edge_traveled_m=1.86, current_selected_yaw_rad=0.2))
        self.assertTrue(between_stage_route_integrity_supported(
            "Reach the gap between the bar and chairs",
            current_edge_traveled_m=1.86, current_selected_yaw_rad=0.2))
        self.assertFalse(between_stage_route_integrity_supported(
            "Walk through and clear the gap between the bar and chairs",
            current_edge_traveled_m=1.86, current_selected_yaw_rad=0.2))
        self.assertTrue(between_stage_route_integrity_supported(
            current_edge_traveled_m=2.3, current_selected_yaw_rad=0.2))
        context = {
            "active": True,
            "prior_edge_traveled_distance_m": 1.4,
            "prior_selected_yaw_rad": 0.2,
            "prior_motion_evidence": {
                "instruction_motion_fit": "supports",
                "stationary_or_blocked": False,
            },
        }
        self.assertTrue(between_stage_route_integrity_supported(
            current_edge_traveled_m=1.0,
            stage_progress_context=context,
            current_selected_yaw_rad=0.7))
        self.assertFalse(between_stage_route_integrity_supported(
            current_edge_traveled_m=1.0,
            stage_progress_context=context,
            current_selected_yaw_rad=2.0))

    def test_between_adapter_cannot_override_route_integrity_veto(self):
        six_views = [np.zeros((24, 32, 3), np.uint8) for _ in range(6)]
        eight_views = [np.zeros((24, 32, 3), np.uint8) for _ in range(8)]
        between = SubInstruction.from_mapping({
            "sub_instruction_id": 0,
            "navigation_instruction": (
                "walk through and clear the gap between the bar and chairs"),
            "landmark": "bar and chairs", "form": "BETWEEN_OBJECTS",
            "semantic_spatial_target": "floor beyond the gap",
            "completion_cue": "clear the gap",
        })
        following = SubInstruction.from_mapping({
            "sub_instruction_id": 1,
            "navigation_instruction": "stop at the corner of the bar",
            "landmark": "corner of the bar", "form": "STOP_WAIT",
        })
        harness = NavigationVLMHarness(
            StructuredSatisfiedBetweenBackend(),
            instruction_completion_prompt_version=(
                "v24_multireference_threshold_calibration"))
        with tempfile.TemporaryDirectory() as temporary:
            memory = NavigationGraphMemory(
                temporary, FakeSemanticExtractor())
            memory.add_origin_node(
                [0, 0, 0], 0.0, 0, six_views,
                completion_views=eight_views)
            memory.get_node("node_0000").environment_semantics = {
                "views": []}
            node, _ = memory.add_navigation_stop_node(
                [0, 0, -1.76], 0.0, 1, six_views, between,
                [{"action": "forward", "turn_deg": 0.0,
                  "moved_m": 1.76}],
                arrival_signal="point_navigation_arrived",
                completion_views=eight_views,
                edge_metadata={"selected_yaw_rad": 0.0})
            node.environment_semantics = {"views": [{
                "view_index": 0,
                "detections": [{
                    "label": "corner of the bar", "score": 0.6,
                    "area_fraction": 0.1,
                }],
            }]}
            result = NodeTransitionInstructionCompletionJudge(
                harness, memory, [between, following]).judge(
                    node, 0, eight_views,
                    [eight_views[0], eight_views[1]])
        self.assertEqual(result.status, "unknown")
        self.assertFalse(result.instruction_completed)
        self.assertTrue(result.validation_overrides[
            "between_stage_route_integrity_veto"])
        self.assertFalse(result.decision_gates[
            "between_stage_route_integrity"])
        self.assertEqual(result.model_status, "completed")
        self.assertNotIn("between_following_landmark_override",
                         result.validation_overrides)
        self.assertEqual(infer_unknown_disposition(
            result.to_dict(), [{"action": "forward", "moved_m": 1.76}]),
            "on_route")

    def test_far_side_circumnavigation_requires_coherent_stage_route(self):
        text = "Walk around the backside of the couches"
        self.assertFalse(circumnavigation_stage_route_integrity_supported(
            text, current_edge_traveled_m=2.9,
            current_selected_yaw_rad=1.4))
        self.assertTrue(circumnavigation_stage_route_integrity_supported(
            text, current_edge_traveled_m=4.1,
            current_selected_yaw_rad=1.4))
        context = {
            "active": True,
            "prior_edge_traveled_distance_m": 2.9,
            "prior_selected_yaw_rad": -1.96,
            "prior_motion_evidence": {
                "instruction_motion_fit": "supports",
                "stationary_or_blocked": False,
            },
        }
        self.assertTrue(circumnavigation_stage_route_integrity_supported(
            text, current_edge_traveled_m=2.4,
            stage_progress_context=context,
            current_selected_yaw_rad=-2.64))
        self.assertFalse(circumnavigation_stage_route_integrity_supported(
            text, current_edge_traveled_m=2.4,
            stage_progress_context=context,
            current_selected_yaw_rad=1.44))
        self.assertTrue(circumnavigation_stage_route_integrity_supported(
            "walk beside the couch", current_edge_traveled_m=1.0))

    def test_vertical_recovery_does_not_lock_previous_flight_bearing(self):
        self.assertFalse(fixed_recovery_corridor_eligible("VERTICAL_UP"))
        self.assertFalse(fixed_recovery_corridor_eligible("VERTICAL_DOWN"))
        self.assertTrue(fixed_recovery_corridor_eligible("BETWEEN_OBJECTS"))
        self.assertTrue(fixed_recovery_corridor_eligible("ADVANCE_STRAIGHT"))
        self.assertFalse(fixed_recovery_corridor_eligible(
            "FOLLOW_PATH_BOUNDARY"))
        self.assertFalse(fixed_recovery_corridor_eligible("CIRCUMNAVIGATE"))
        self.assertFalse(fixed_recovery_corridor_eligible("TURN_LEFT"))

    def test_only_endpoint_vetoed_portal_consumes_initial_portal_direction(self):
        progress = {
            "active": True,
            "prior_selected_yaw_rad": -1.2,
            "prior_exit_endpoint_side_veto_applied": True,
        }
        for form in ("EXIT_REGION", "ENTER_REGION", "SELECT_PORTAL",
                     "TRAVERSE_PORTAL_REGION"):
            self.assertTrue(exit_threshold_portal_continuation_required(
                {"form": form}, progress))
        self.assertFalse(exit_threshold_portal_continuation_required(
            {"form": "EXIT_REGION"}, {**progress,
                    "prior_exit_endpoint_side_veto_applied": False}))
        self.assertFalse(exit_threshold_portal_continuation_required(
            {"form": "PASS_LANDMARK"}, progress))

    def test_portal_recovery_keeps_only_supported_forward_hemisphere(self):
        for form in ("EXIT_REGION", "ENTER_REGION", "SELECT_PORTAL",
                     "TRAVERSE_PORTAL_REGION"):
            self.assertTrue(portal_recovery_hemisphere_eligible(form))
            self.assertFalse(portal_recovery_corridor_active(form, False))
            self.assertTrue(portal_recovery_corridor_active(form, True))
        for form in ("PASS_LANDMARK", "TURN_AROUND", "BETWEEN_OBJECTS"):
            self.assertFalse(portal_recovery_hemisphere_eligible(form))
            self.assertFalse(portal_recovery_corridor_active(form, True))

    def test_linear_continuation_reserves_half_of_alignment_budget(self):
        self.assertEqual(
            LINEAR_SAME_STAGE_CONTINUATION_HALF_WIDTH_DEG, 15.0)

    def test_selected_point_route_yaw_uses_pixel_not_view_center(self):
        candidate = {
            "yaw": -2.0,
            "target_mask": np.ones((20, 320), bool),
            "point": np.array([80.0, 15.0]),
        }
        expected = -2.0 - math.atan((80.0 - 159.5) / 160.0)
        self.assertAlmostEqual(
            selected_point_ray_yaw_rad(candidate), expected)

    def test_operational_stop_override_excludes_precise_endpoint_relations(self):
        self.assertTrue(generic_near_stop_override_eligible(
            "stop near the rug", "floor near the rug", "near the rug"))
        for instruction in (
                "stop near the corner of the bar",
                "wait near the end of the counter",
                "stand near the entrance to the bedroom",
                "stop in front of the couch"):
            with self.subTest(instruction=instruction):
                self.assertFalse(generic_near_stop_override_eligible(
                    instruction, instruction, instruction))

    def test_floor_covering_landmark_aliases_are_stable_across_views(self):
        self.assertEqual(
            landmark_detection_aliases("gray rug"),
            {"gray", "rug", "carpet", "mat"})
        self.assertEqual(landmark_detection_aliases("wooden table"),
                         {"wooden", "table"})

    def test_turn_direction_uses_signed_online_motion(self):
        self.assertTrue(turn_direction_motion_supported("TURN_LEFT", {
            "endpoint_heading_delta_deg": 90.0,
            "signed_cumulative_turn_deg": 30.0,
        }))
        self.assertFalse(turn_direction_motion_supported("TURN_LEFT", {
            "endpoint_heading_delta_deg": -90.0,
            "signed_cumulative_turn_deg": -30.0,
        }))
        self.assertFalse(turn_direction_motion_supported("TURN_RIGHT", {
            "endpoint_heading_delta_deg": 80.0,
            "signed_cumulative_turn_deg": -60.0,
        }))
        self.assertTrue(turn_direction_motion_supported("TURN_AROUND", {
            "endpoint_heading_delta_deg": -150.0,
            "signed_cumulative_turn_deg": -120.0,
        }))

    def test_v23_terminal_and_under_relation_detector_vetoes(self):
        closet = [{"view_index": 0, "matched_tokens": ["closet"],
                   "score": 0.35, "area_fraction": 0.02}]
        self.assertFalse(terminal_relation_detection_supported(
            "APPROACH_LANDMARK", []))
        self.assertTrue(terminal_relation_detection_supported(
            "APPROACH_LANDMARK", closet))
        one_balcony = [{"view_index": 1, "matched_tokens": ["balcony"],
                        "score": 0.38, "area_fraction": 0.1}]
        two_balcony = one_balcony + [
            {"view_index": 3, "matched_tokens": ["balcony"],
             "score": 0.34, "area_fraction": 0.08}]
        instruction = "enter the free floor under the balcony"
        self.assertFalse(under_relation_multiview_supported(
            instruction, one_balcony))
        self.assertTrue(under_relation_multiview_supported(
            instruction, two_balcony))
        self.assertTrue(under_relation_multiview_supported(
            "enter the living room", []))

    def test_terminal_relation_rgb_fallback_requires_full_consensus(self):
        evidence = dict(
            same_instance="yes", current_target_state="satisfied",
            current_completion_cue="satisfied",
            current_reference_sectors={"front_left"})
        self.assertTrue(terminal_relation_rgb_consensus_supported(**evidence))
        for key in evidence:
            broken = dict(evidence)
            broken[key] = (set() if key == "current_reference_sectors" else
                           "partial" if key != "same_instance" else "no")
            self.assertFalse(
                terminal_relation_rgb_consensus_supported(**broken))

    def test_v34_combines_all_online_route_guards(self):
        harness = NavigationVLMHarness(
            HeuristicBackend(), point_selection_prompt_version=(
                "v34_unified_history_route_guard"))
        self.assertTrue(harness._first_step_route_guard)
        self.assertTrue(harness._relation_route_guard)
        self.assertTrue(harness._task30_route_anchor)
        self.assertTrue(harness._stage2_route_guard)
        self.assertTrue(harness._stage2_geometry_v27)
        self.assertTrue(harness._stage3_stop_relation_v29)
        self.assertTrue(harness._stage3_relation_portal_v30)
        self.assertEqual(harness.point_selection_candidate_policy,
                         "soft_detection_evidence")

    def test_active_v31_includes_independent_later_stage_route_review(self):
        harness = NavigationVLMHarness(
            HeuristicBackend(), point_selection_prompt_version=(
                "v31_circumnavigate_forward_competitor"))
        self.assertTrue(harness._history_safe_refinement)
        self.assertTrue(harness._relation_route_guard)
        self.assertTrue(harness._task30_route_anchor)
        self.assertTrue(harness._stage2_route_guard)
        self.assertTrue(harness._stage2_geometry_v27)
        self.assertTrue(harness._qualified_region_detector_grounding)

    def test_form_level_selection_caps_match_execution_budget(self):
        self.assertEqual(selection_geodesic_cap_m("EXIT_REGION"), 3.0)
        self.assertEqual(selection_geodesic_cap_m("SELECT_PORTAL"), 3.0)
        self.assertEqual(selection_geodesic_cap_m(
            "EXIT_REGION", retry_after_unknown=True), 3.0)
        self.assertEqual(selection_geodesic_cap_m("TURN_AROUND"), 2.5)
        self.assertEqual(selection_geodesic_cap_m("TURN_TO_LANDMARK"), 3.0)
        self.assertEqual(selection_geodesic_cap_m("VERTICAL_UP"), 8.0)
        self.assertEqual(selection_geodesic_cap_m("PASS_LANDMARK"), 8.0)
        self.assertEqual(selection_geodesic_cap_m("ADVANCE_STRAIGHT"), 6.0)
        self.assertEqual(selection_geodesic_cap_m("FOLLOW_PATH_BOUNDARY"), 6.0)
        self.assertEqual(selection_geodesic_cap_m(
            "TURN_AROUND", retry_after_unknown=True), 8.0)
        self.assertEqual(selection_geodesic_cap_m(
            "TURN_LEFT", retry_after_unknown=True, bare_turn=True), 4.0)
        self.assertEqual(selection_geodesic_cap_m(
            "TURN_RIGHT", bare_turn=True), 4.0)
        self.assertEqual(selection_geodesic_cap_m(
            "TURN_AROUND", bare_turn=True), 2.5)
        self.assertIsNone(selection_geodesic_cap_m(
            "TURN_LEFT", bare_turn=False))
        self.assertEqual(selection_geodesic_cap_m("STOP_WAIT"), 3.0)

    def test_only_explicitly_local_portal_targets_get_short_cap(self):
        self.assertEqual(local_portal_endpoint_cap_m({
            "form": "TRAVERSE_PORTAL_REGION",
            "semantic_spatial_target": "free floor just inside the loft",
        }, None), 4.0)
        self.assertEqual(local_portal_endpoint_cap_m({
            "form": "EXIT_REGION",
            "completion_cue": "immediately outside the room",
        }, 3.0), 3.0)
        self.assertEqual(local_portal_endpoint_cap_m({
            "form": "SELECT_PORTAL",
            "spatial_relation": "at the threshold",
        }, 1.5), 1.5)
        self.assertIsNone(local_portal_endpoint_cap_m({
            "form": "ENTER_REGION",
            "semantic_spatial_target": "free floor just inside the room",
        }, None))

    def test_directional_towards_turn_builds_local_route_node(self):
        stage = {
            "form": "TURN_LEFT",
            "navigation_instruction": (
                "Turn left and walk towards the far side of the loft"),
        }
        self.assertEqual(directional_towards_route_setup_cap_m(
            stage, None), 3.0)
        self.assertEqual(directional_towards_route_setup_cap_m(
            stage, 8.0), 3.0)
        self.assertIsNone(directional_towards_route_setup_cap_m({
            "form": "TURN_LEFT",
            "navigation_instruction": "Turn left and walk through the door",
        }, None))
        self.assertIsNone(local_portal_endpoint_cap_m({
            "form": "TRAVERSE_PORTAL_REGION",
            "semantic_spatial_target": "floor through the long hallway",
        }, None))
        self.assertEqual(local_portal_endpoint_cap_m({
            "form": "ADVANCE_STRAIGHT",
            "semantic_spatial_target": "floor just beyond the sofa",
        }, 6.0), 6.0)

    def test_local_landmark_stop_relation_excludes_region_entry(self):
        self.assertTrue(local_landmark_stop_relation({
            "form": "STOP_WAIT", "landmark": "toilet",
            "navigation_instruction": "Stop in front of the toilet",
            "spatial_relation": "in front of the toilet",
        }))
        self.assertTrue(local_landmark_stop_relation({
            "form": "STOP_WAIT", "landmark": "bar",
            "navigation_instruction": "Stop at the corner of the bar",
            "spatial_relation": "at the corner of the bar",
        }))
        self.assertFalse(local_landmark_stop_relation({
            "form": "STOP_WAIT", "landmark": "workout room",
            "navigation_instruction": "Stop inside the workout room",
            "spatial_relation": "inside the room",
        }))


    def test_unknown_disposition_accepts_real_non_reversing_partial_motion(self):
        completion = {
            "status": "unknown",
            "endpoint_evidence": {
                "current_target_state": "partial",
                "current_completion_cue": "partial",
            },
            "temporal_evidence": {
                "semantic_order": "instructed",
                "keyframe_support": True,
                "same_reference_instance": "yes",
                "reverse_transition_observed": False,
            },
            "motion_evidence": {
                "instruction_motion_fit": "supports",
                "stationary_or_blocked": False,
            },
        }
        actions = [{"action": "forward", "moved_m": 1.0}]
        self.assertEqual(
            infer_unknown_disposition(completion, actions), "on_route")
        # Missing/ambiguous semantic annotations alone must not discard real
        # gradual progress; the state machine bounds how long it may continue.
        for section, key, value in (
                ("temporal_evidence", "semantic_order", "ambiguous"),
                ("temporal_evidence", "keyframe_support", False),
                ("temporal_evidence", "same_reference_instance", "ambiguous"),
                ("motion_evidence", "instruction_motion_fit", "neutral"),
                ("endpoint_evidence", "current_target_state", "unsatisfied")):
            with self.subTest(section=section, key=key):
                changed = {
                    outer: dict(inner) if isinstance(inner, dict) else inner
                    for outer, inner in completion.items()
                }
                changed[section][key] = value
                self.assertEqual(
                    infer_unknown_disposition(changed, actions), "on_route")
        for section, key, value in (
                ("temporal_evidence", "reverse_transition_observed", True),
                ("motion_evidence", "stationary_or_blocked", True)):
            changed = {
                outer: dict(inner) if isinstance(inner, dict) else inner
                for outer, inner in completion.items()
            }
            changed[section][key] = value
            self.assertEqual(
                infer_unknown_disposition(changed, actions), "wrong")
        self.assertEqual(
            infer_unknown_disposition(completion, [
                {"action": "forward", "moved_m": 0.49}]), "wrong")
        shallow_between = {
            **completion,
            "endpoint_evidence": {
                **completion["endpoint_evidence"],
                "current_target_state": "satisfied",
                "current_completion_cue": "satisfied",
            },
            "model_status": "completed",
            "decision_gates": {"between_stage_route_integrity": False},
        }
        self.assertEqual(infer_unknown_disposition(
            shallow_between, actions), "on_route")
        shallow_circumnavigation = {
            **shallow_between,
            "decision_gates": {
                "circumnavigation_stage_route_integrity": False},
        }
        self.assertEqual(infer_unknown_disposition(
            shallow_circumnavigation, actions), "on_route")
        exit_threshold = {
            **completion,
            "endpoint_evidence": {
                **completion["endpoint_evidence"],
                "current_target_state": "satisfied",
                "current_completion_cue": "satisfied",
            },
            "model_status": "completed",
            "validation_overrides": {
                "exit_endpoint_side_veto_applied": True},
            "exit_endpoint_side_audit": {
                "endpoint_side": "source_or_threshold",
                "confidence": 0.9,
            },
        }
        self.assertEqual(infer_unknown_disposition(
            exit_threshold, actions), "on_route")
        exit_threshold["exit_endpoint_side_audit"] = {
            "endpoint_side": "source_or_threshold", "confidence": 0.69}
        self.assertEqual(infer_unknown_disposition(
            exit_threshold, actions), "on_route")
        exit_threshold["exit_endpoint_side_audit"] = {
            "endpoint_side": "invalid", "confidence": 0.99}
        self.assertEqual(infer_unknown_disposition(
            exit_threshold, actions), "wrong")
        local_obstruction_boundary = {
            **completion,
            "endpoint_evidence": {
                **completion["endpoint_evidence"],
                "current_target_state": "satisfied",
                "current_completion_cue": "satisfied",
            },
            "validation_overrides": {
                "local_occlusion_boundary_intermediate_veto": True},
        }
        self.assertEqual(infer_unknown_disposition(
            local_obstruction_boundary, actions), "on_route")
        shallow_between["motion_evidence"] = {
            **completion["motion_evidence"], "stationary_or_blocked": True}
        self.assertEqual(infer_unknown_disposition(
            shallow_between, actions), "wrong")

    def test_sequence_bounds_weak_unknown_progress_before_recovery(self):
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 0,
            "navigation_instruction": "walk along the bar to its far end",
            "form": "FOLLOW_PATH_BOUNDARY",
        })
        state = InstructionSequenceStateMachine(
            [sub_instruction], initial_node_id="node_0000",
            max_consecutive_on_route_unknowns=2)
        on_route = {
            "belongs_to_sequence": False,
            "matched_sub_instruction_id": -1,
            "confidence": 0.8,
            "unknown_disposition": "on_route",
        }
        self.assertEqual(state.observe(
            "node_0001", on_route, selected_yaw=0.1).action,
            "continue_current_instruction")
        second = state.observe(
            "node_0002", on_route, selected_yaw=0.2)
        self.assertEqual(second.action, "backtrack_and_block")
        self.assertEqual(second.backtrack_target_node_id, "node_0001")
        state.on_backtrack(success=True, current_node_id="node_0005")
        self.assertEqual(state.consecutive_on_route_unknowns, 1)
        self.assertEqual(state.off_sequence_nodes, ["node_0005"])
        self.assertAlmostEqual(state.branch_route_yaw_rad, 0.1)
        self.assertAlmostEqual(state.snapshot()["branch_route_yaw_rad"], 0.1)
        retry = state.observe(
            "node_0006", dict(on_route, unknown_disposition="wrong"),
            selected_yaw=-0.5)
        self.assertEqual(retry.action, "backtrack_and_block")
        self.assertEqual(retry.backtrack_target_node_id, "node_0005")

    def test_only_immediately_following_stop_wait_can_share_arrival_edge(self):
        stop = SubInstruction.from_mapping({
            "sub_instruction_id": 1,
            "navigation_instruction": "stop beside the rug",
            "form": "STOP_WAIT",
        })
        advance = type("Directive", (), {"action": "advance_sequence"})()
        retry = type("Directive", (), {"action": "explore_once_more"})()
        self.assertTrue(chained_stop_wait_eligible(advance, stop))
        self.assertFalse(chained_stop_wait_eligible(retry, stop))
        self.assertFalse(chained_stop_wait_eligible(advance, SubInstruction.from_mapping({
            "sub_instruction_id": 1,
            "navigation_instruction": "enter the room",
            "form": "ENTER_REGION",
        })))
        final_portal_stop = SubInstruction.from_mapping({
            "sub_instruction_id": 1,
            "navigation_instruction": "reach the doorway facing the bed and stop",
            "form": "SELECT_PORTAL",
        })
        self.assertFalse(chained_stop_wait_eligible(
            advance, final_portal_stop, is_final_sub_instruction=True))
        self.assertFalse(chained_stop_wait_eligible(
            advance, final_portal_stop, is_final_sub_instruction=False))

    def test_chained_plain_near_stop_accepts_ordered_landmark_growth_only(self):
        stop = SubInstruction.from_mapping({
            "sub_instruction_id": 1,
            "navigation_instruction": "stop near the rug",
            "landmark": "rug",
            "semantic_spatial_target": "floor near the rug",
            "spatial_relation": "near the rug",
            "form": "STOP_WAIT",
        })
        completion = {
            "status": "unknown",
            "endpoint_evidence": {
                "current_target_state": "partial",
                "current_completion_cue": "partial",
            },
            "temporal_evidence": {
                "semantic_order": "instructed",
                "keyframe_support": True,
                "reverse_transition_observed": False,
            },
            "motion_evidence": {
                "instruction_motion_fit": "supports",
                "stationary_or_blocked": False,
            },
        }
        previous = {"views": [{"detections": [{
            "label": "rug carpet", "mask_area_fraction": 0.03,
        }]}]}
        current = {"views": [{"detections": [{
            "label": "carpet", "mask_area_fraction": 0.12,
        }]}]}
        actions = [{"action": "forward", "moved_m": 0.8}]
        self.assertTrue(chained_near_stop_evidence_supported(
            stop, completion, previous, current, actions))
        corner = SubInstruction.from_mapping({
            **stop.to_dict(),
            "navigation_instruction": "stop at the corner of the rug",
            "semantic_spatial_target": "floor at the corner of the rug",
            "spatial_relation": "at the corner of the rug",
        })
        self.assertFalse(chained_near_stop_evidence_supported(
            corner, completion, previous, current, actions))
        reversed_completion = dict(completion, temporal_evidence={
            **completion["temporal_evidence"],
            "reverse_transition_observed": True,
        })
        self.assertFalse(chained_near_stop_evidence_supported(
            stop, reversed_completion, previous, current, actions))

    def test_terminal_extent_rgb_override_requires_complete_rear_chronology(self):
        evidence = dict(
            current_target_state="satisfied",
            current_completion_cue="satisfied",
            same_instance="yes",
            semantic_order="instructed",
            boundary_event="endpoint_inferred",
            keyframe_support=True,
            motion_fit="supports",
            stationary_or_blocked=False,
            reverse_transition_observed=False,
            current_reference_sectors={"rear_left", "rear", "rear_right"},
        )
        self.assertTrue(terminal_extent_rgb_rear_clear_supported(**evidence))
        self.assertFalse(terminal_extent_rgb_rear_clear_supported(
            **dict(evidence, current_reference_sectors={"rear", "front"})))
        self.assertFalse(terminal_extent_rgb_rear_clear_supported(
            **dict(evidence, same_instance="ambiguous")))
        self.assertFalse(terminal_extent_rgb_rear_clear_supported(
            **dict(evidence, keyframe_support=False)))

    def test_terminal_extent_partial_can_complete_after_coherent_multiedge_span(self):
        evidence = dict(
            current_target_state="partial",
            current_completion_cue="partial",
            same_instance="yes",
            semantic_order="instructed",
            keyframe_support=True,
            motion_fit="supports",
            stationary_or_blocked=False,
            reverse_transition_observed=False,
            current_reference_sectors={"rear_left", "rear", "rear_right"},
            cumulative_stage_travel_m=8.7,
        )
        text = "enter the kitchen and walk along the counter to the end"
        self.assertTrue(terminal_extent_rgb_completion_supported(
            text, **evidence))
        self.assertFalse(terminal_extent_rgb_completion_supported(
            text, **dict(evidence, cumulative_stage_travel_m=3.3)))
        self.assertFalse(terminal_extent_rgb_completion_supported(
            text, **dict(evidence,
                         current_reference_sectors={"rear", "front"})))
        self.assertFalse(terminal_extent_rgb_completion_supported(
            "enter the kitchen", **evidence))

    def test_only_explicitly_continuing_portal_inherits_route(self):
        self.assertTrue(inherits_incoming_route_corridor({
            "form": "TRAVERSE_PORTAL_REGION",
            "navigation_instruction": "continue forward through the next doorway",
        }))
        self.assertTrue(inherits_incoming_route_corridor({
            "form": "SELECT_PORTAL",
            "navigation_instruction": "until you reach the next doorway",
        }))
        self.assertFalse(inherits_incoming_route_corridor({
            "form": "SELECT_PORTAL",
            "navigation_instruction": "reach the next doorway",
        }))
        self.assertFalse(inherits_incoming_route_corridor({
            "form": "SELECT_PORTAL",
            "navigation_instruction": "take the doorway on your right",
        }))
        self.assertTrue(inherits_incoming_route_corridor({
            "form": "PASS_LANDMARK",
            "navigation_instruction": "Walk straight passing the gray couch",
        }))
        self.assertTrue(inherits_incoming_route_corridor({
            "form": "ADVANCE_STRAIGHT",
            "navigation_instruction": "Keep walking straight past the room",
        }))

    def test_safety_retry_only_reopens_history_for_explicit_turnaround(self):
        self.assertFalse(rejected_ray_retry_reopens_incoming({
            "form": "ENTER_REGION",
            "navigation_instruction": "Enter the room with boxes",
        }))
        self.assertFalse(rejected_ray_retry_reopens_incoming({
            "form": "PASS_LANDMARK",
            "navigation_instruction": "Walk past the stairs",
        }))
        self.assertTrue(rejected_ray_retry_reopens_incoming({
            "form": "TURN_AROUND",
            "navigation_instruction": "Turn around",
        }))

    def test_compound_terminal_extent_route_is_lexical_and_form_limited(self):
        stage = {
            "form": "ENTER_REGION",
            "navigation_instruction": (
                "enter the kitchen and walk along the countertop"),
            "completion_cue": "reach the end of the countertop area",
        }
        self.assertTrue(compound_terminal_extent_route(stage))
        self.assertFalse(compound_terminal_extent_route({
            **stage, "completion_cue": "enter the kitchen",
        }))
        self.assertFalse(compound_terminal_extent_route({
            **stage, "form": "STOP_WAIT",
        }))
        self.assertFalse(inherits_incoming_route_corridor({
            "form": "PASS_LANDMARK",
            "navigation_instruction": "Pass the couch",
        }))
        self.assertFalse(inherits_incoming_route_corridor({
            "form": "FOLLOW_PATH_BOUNDARY",
            "navigation_instruction": "Turn left and follow the railing",
        }))

    def test_explicit_linear_cap_needs_committed_straight_corridor(self):
        straight_pass = {
            "form": "PASS_LANDMARK",
            "navigation_instruction": "Walk straight past the gray couch",
        }
        self.assertEqual(explicit_linear_segment_cap_m(
            straight_pass, None, corridor_active=True), 2.5)
        self.assertEqual(explicit_linear_segment_cap_m(
            straight_pass, 2.0, corridor_active=True), 2.0)
        self.assertIsNone(explicit_linear_segment_cap_m(
            straight_pass, None, corridor_active=False))
        self.assertIsNone(explicit_linear_segment_cap_m({
            "form": "PASS_LANDMARK",
            "navigation_instruction": "Pass the gray couch",
        }, None, corridor_active=True))
        self.assertIsNone(explicit_linear_segment_cap_m({
            "form": "SELECT_PORTAL",
            "navigation_instruction": "Continue straight to the doorway",
        }, None, corridor_active=True))
        self.assertEqual(circumnavigation_continuation_cap_m(
            8.0, stage_progress_active=True), 3.5)
        self.assertEqual(circumnavigation_continuation_cap_m(
            3.0, stage_progress_active=True), 3.0)
        self.assertIsNone(circumnavigation_continuation_cap_m(
            None, stage_progress_active=False))

    def test_sequence_revisit_inherits_all_prior_blocked_directions(self):
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 0,
            "navigation_instruction": "enter the next room",
            "form": "ENTER_REGION",
        })
        state = InstructionSequenceStateMachine(
            [sub_instruction], initial_node_id="node_0000",
            max_blocked_directions_per_node=5)
        wrong = {
            "belongs_to_sequence": False,
            "matched_sub_instruction_id": -1,
            "confidence": 0.8,
            "unknown_disposition": "wrong",
        }
        state.observe("node_0001", wrong, selected_yaw=0.1)
        state.observe("node_0002", wrong, selected_yaw=0.8)
        state.on_backtrack(success=True, current_node_id="node_0003")
        self.assertEqual(len(state.blocked_yaws()), 1)

        retry = state.observe("node_0004", wrong, selected_yaw=-0.9)
        self.assertEqual(retry.action, "backtrack_and_block")
        state.on_backtrack(success=True, current_node_id="node_0006")
        inherited = state.blocked_yaws()
        self.assertEqual(len(inherited), 2)
        self.assertTrue(any(abs(value - 0.8) < 1e-6 for value in inherited))
        self.assertTrue(any(abs(value + 0.9) < 1e-6 for value in inherited))

    def test_failed_physical_direction_is_blocked_at_logical_branch_origin(self):
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 0,
            "navigation_instruction": "approach the doorway",
            "form": "APPROACH_LANDMARK",
        })
        state = InstructionSequenceStateMachine(
            [sub_instruction], initial_node_id="node_0000")
        state.branch_origin_node_id = "node_0003"
        state.off_sequence_nodes = ["node_0003"]
        blocked_at = state.block_failed_physical_direction(1.2)
        self.assertEqual(blocked_at, "node_0003")
        self.assertEqual(state.blocked_yaws(), [1.2])
        # Controller jitter around the same failed ray does not consume a new
        # branch direction.
        state.block_failed_physical_direction(1.2 + np.deg2rad(5))
        self.assertEqual(len(state.blocked_yaws()), 1)

    def test_physical_failure_blocks_do_not_consume_the_judge_block_cap(self):
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 0,
            "navigation_instruction": "enter the next room",
            "form": "ENTER_REGION",
        })
        state = InstructionSequenceStateMachine(
            [sub_instruction], initial_node_id="node_0000",
            max_blocked_directions_per_node=2)
        wrong = {
            "belongs_to_sequence": False,
            "matched_sub_instruction_id": -1,
            "confidence": 0.8,
            "unknown_disposition": "wrong",
        }
        state.observe("node_0001", wrong, selected_yaw=-0.5)
        state.block_failed_physical_direction(0.0)
        state.block_failed_physical_direction(math.pi / 2)
        state.block_failed_physical_direction(math.pi)
        self.assertEqual(len(state.blocked_yaws()), 3)
        self.assertIsNone(state.terminated_reason)
        self.assertEqual(
            state.judge_blocked_yaws_by_verified_node.get("node_0001", []), [])

        state.observe("node_0002", wrong, selected_yaw=-0.5)
        state.on_backtrack(success=True, current_node_id="node_0003")
        self.assertIsNone(state.terminated_reason)
        # The physical blocks are inherited by the revisit node but still do
        # not count; only the second judge-confirmed block trips the cap.
        self.assertEqual(len(state.blocked_yaws()), 4)
        self.assertEqual(
            len(state.judge_blocked_yaws_by_verified_node["node_0003"]), 1)
        state.observe("node_0004", wrong, selected_yaw=-1.5)
        state.on_backtrack(success=True, current_node_id="node_0005")
        self.assertEqual(
            state.terminated_reason,
            "all_candidate_directions_blocked_at_verified_node")
        snapshot = state.snapshot()
        self.assertIn("judge_blocked_yaws_by_verified_node", snapshot)
        self.assertEqual(snapshot["in_place_turn_records_by_sub_instruction_id"], {})

    def test_in_place_turn_record_is_kept_per_sub_instruction(self):
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 3,
            "navigation_instruction": "turn left",
            "form": "TURN_LEFT",
        })
        state = InstructionSequenceStateMachine(
            [sub_instruction], initial_node_id="node_0000")
        self.assertIsNone(state.in_place_turn_record(3))
        state.record_in_place_turn(3, {"sector": "left", "hop_index": 1})
        self.assertEqual(state.in_place_turn_record(3)["sector"], "left")
        self.assertIsNone(state.in_place_turn_record(4))
        self.assertEqual(
            state.snapshot()["in_place_turn_records_by_sub_instruction_id"],
            {3: {"sector": "left", "hop_index": 1}})

    def test_bare_turn_alignment_excludes_compound_destination_clauses(self):
        self.assertAlmostEqual(bare_turn_delta_rad({
            "form": "TURN_LEFT", "navigation_instruction": "Turn left."
        }), np.pi / 2.0)
        self.assertAlmostEqual(bare_turn_delta_rad({
            "form": "TURN_RIGHT",
            "navigation_instruction": "Then make a right turn",
        }), -np.pi / 2.0)
        self.assertAlmostEqual(bare_turn_delta_rad({
            "form": "TURN_LEFT",
            "navigation_instruction": "Take a hard left.",
        }), np.pi / 2.0)
        self.assertAlmostEqual(bare_turn_delta_rad({
            "form": "TURN_AROUND",
            "navigation_instruction": "Turn around 180 degrees",
        }), np.pi)
        self.assertIsNone(bare_turn_delta_rad({
            "form": "TURN_RIGHT",
            "navigation_instruction": (
                "Turn right and walk across the room to the doorway"),
        }))
        self.assertIsNone(bare_turn_delta_rad({
            "form": "TURN_LEFT",
            "navigation_instruction": "Turn left toward the hallway",
        }))

    def test_bare_turn_preview_records_real_selector_rotation(self):
        alignment = {
            "status": "deferred_until_physical_arrival",
            "source_node_yaw_rad": math.radians(100.0),
        }
        self.assertAlmostEqual(math.degrees(
            bare_turn_preview_yaw_delta_rad(
                alignment, math.radians(167.5))), 67.5)
        self.assertIsNone(bare_turn_preview_yaw_delta_rad(
            {**alignment, "status": "already_aligned"}, 0.0))

    def test_incoming_route_frame_uses_real_xz_displacement(self):
        origin = np.array([0.0, 1.5, 0.0])
        self.assertAlmostEqual(
            incoming_route_yaw_rad(origin, [0.0, 1.5, -2.0]), 0.0)
        self.assertAlmostEqual(
            incoming_route_yaw_rad(origin, [-2.0, 1.5, 0.0]),
            np.pi / 2.0)
        self.assertAlmostEqual(
            incoming_route_yaw_rad(origin, [2.0, 1.5, 0.0]),
            -np.pi / 2.0)
        self.assertIsNone(incoming_route_yaw_rad(
            origin, [0.1, 1.5, -0.1]))
        with self.assertRaises(ValueError):
            incoming_route_yaw_rad([0.0, 0.0], [0.0, 0.0, 1.0])

    def test_named_landmark_turn_translates_on_next_instruction_route(self):
        current = {
            "sub_instruction_id": 0,
            "navigation_instruction": "Turn toward the wooden mantel",
            "landmark": "wooden mantel",
            "semantic_spatial_target": "floor in front of the mantel",
            "form": "TURN_TO_LANDMARK",
        }
        following = SubInstruction.from_mapping({
            "sub_instruction_id": 1,
            "navigation_instruction": "go forward through the doorway",
            "landmark": "doorway",
            "semantic_spatial_target": "floor beyond the doorway",
            "form": "TRAVERSE_PORTAL_REGION",
        })
        route_stage, alignment_stage = named_landmark_route_setup_stage(
            current, following)
        self.assertEqual(route_stage["form"], "TURN_TO_LANDMARK")
        self.assertEqual(alignment_stage["form"], "TURN_TO_LANDMARK")
        self.assertTrue(route_stage["metadata"][
            "turn_to_landmark_route_setup"]["active"])
        unchanged, alignment = named_landmark_route_setup_stage(
            current, following, progress_lookahead_active=True)
        self.assertEqual(unchanged, current)
        self.assertIsNone(alignment)
        self.assertEqual(named_landmark_route_setup_cap_m(None, True), 4.0)
        self.assertEqual(named_landmark_route_setup_cap_m(3.0, True), 3.0)
        self.assertIsNone(named_landmark_route_setup_cap_m(None, False))

    def test_bare_turn_translates_on_following_route_before_exact_alignment(self):
        current = {
            "sub_instruction_id": 1,
            "navigation_instruction": "Turn left.",
            "form": "TURN_LEFT",
        }
        following = SubInstruction.from_mapping({
            "sub_instruction_id": 2,
            "navigation_instruction": "walk through the doorway",
            "landmark": "doorway",
            "semantic_spatial_target": "floor beyond the doorway",
            "form": "TRAVERSE_PORTAL_REGION",
        })
        route_stage, active = bare_turn_route_setup_stage(current, following)
        self.assertTrue(active)
        self.assertEqual(route_stage["form"], "TRAVERSE_PORTAL_REGION")
        self.assertTrue(route_stage["metadata"]["bare_turn_route_setup"]["active"])
        unchanged, active = bare_turn_route_setup_stage(
            current, following, progress_lookahead_active=True)
        self.assertFalse(active)
        self.assertEqual(unchanged, current)

    def test_route_relation_forms_require_distinct_physical_progress(self):
        for form in ("BETWEEN_OBJECTS", "FOLLOW_PATH_BOUNDARY",
                     "CIRCUMNAVIGATE"):
            with self.subTest(form=form):
                self.assertTrue(form_requires_physical_progress(form))
        self.assertFalse(form_requires_physical_progress("STOP_WAIT"))

    def test_relational_landmark_queries_keep_visual_heads(self):
        stage = {
            "source_clause": "Stop by the doorway on the left that leads to the living room.",
            "form": "STOP_WAIT",
            "landmark": "doorway on the left leading to living room",
            "completion_cue": "living room visible through the doorway",
            "semantic_spatial_target": "free floor near the doorway",
            "visual_arrival_evidence": "doorway visible ahead",
        }
        queries = extract_detection_queries(stage)
        self.assertIn("doorway", queries)
        self.assertIn("living room", queries)
    @staticmethod
    def stair_semantics(view_count):
        return {
            "views": [
                {"view_index": index,
                 "detections": ([{"label": "stairs stair"}]
                                if index < view_count else [])}
                for index in range(6)
            ]
        }

    def test_decomposition_returns_named_sub_instructions(self):
        result = InstructionDecomposer(FakeHarness()).decompose("exit the room")
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], SubInstruction)
        self.assertEqual(result[0].sub_instruction_id, 0)
        self.assertEqual(result[0].form, "EXIT_REGION")
        self.assertEqual(result[0].to_stage_dict()["stage_id"], 0)

    def test_vlm_compound_actions_become_independently_verifiable_stages(self):
        result = InstructionDecomposer(CompoundActionHarness()).decompose(
            "Walk into the kitchen and walk along the counter")
        self.assertEqual([item.form for item in result], [
            "ENTER_REGION", "FOLLOW_PATH_BOUNDARY"])
        self.assertEqual([item.sub_instruction_id for item in result], [0, 1])
        self.assertEqual(result[0].navigation_instruction,
                         "Walk into the kitchen")
        self.assertEqual(result[1].navigation_instruction,
                         "walk along the counter")
        self.assertEqual(result[1].metadata["compound_part_index"], 1)
        self.assertEqual(result[1].metadata["compound_action_count"], 2)

    def test_selection_contact_sheet_supports_six_and_eight_views(self):
        six = [np.full((12, 20, 3), index, np.uint8) for index in range(6)]
        eight = [np.full((12, 20, 3), index, np.uint8) for index in range(8)]
        nine = [np.full((12, 20, 3), index, np.uint8) for index in range(9)]
        self.assertEqual(selection_contact_sheet(six).shape, (24, 60, 3))
        self.assertEqual(selection_contact_sheet(eight).shape, (24, 80, 3))
        self.assertEqual(selection_contact_sheet(nine).shape, (36, 80, 3))

    def test_arrival_node_persists_six_views_embedding_and_edge_actions(self):
        views = [np.full((32, 48, 3), index * 30, np.uint8) for index in range(6)]
        sub_instruction = InstructionDecomposer(FakeHarness()).decompose("exit")[0]
        actions = [
            {"step": 0, "action": "forward", "moved_m": 0.22},
            {"step": 1, "action": "turn_left", "moved_m": 0.0},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            memory = NavigationGraphMemory(temporary, FakeSemanticExtractor())
            memory.add_origin_node([0, 0, 0], 0.0, 0, views)
            node, edge = memory.add_arrival_node(
                [0, 0, -0.22], 0.1, 2, views, sub_instruction, actions,
                "point_navigation_arrived")
            match = SubInstructionNodeMatcher().match(node, sub_instruction)
            memory.set_sub_instruction_match(node.node_id, match.to_dict())

            payload = json.loads((Path(temporary) / "navigation_graph.json").read_text())
            self.assertEqual(len(payload["nodes"]), 2)
            self.assertEqual(len(payload["edges"]), 1)
            self.assertEqual(len(payload["nodes"][1]["six_views"]), 6)
            self.assertEqual(len(payload["nodes"][1]["visual_embedding"]), 256)
            self.assertEqual(payload["edges"][0]["action_history"], actions)
            self.assertAlmostEqual(payload["edges"][0]["traveled_distance_m"], 0.22)
            self.assertTrue(match.belongs)

    def test_queries_alone_are_not_observed_semantics(self):
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 4,
            "navigation_instruction": "climb the stairs",
            "landmark": "stairs",
            "form": "VERTICAL_UP",
        })
        node = {
            "departure_purpose_sub_instruction": {
                "sub_instruction_id": 99,
                "navigation_instruction": "walk to the sofa",
                "form": "APPROACH_LANDMARK",
            },
            "environment_semantics": {
                "queries": ["stairs"], "labels": [], "views": [],
            },
            "six_views": [], "visual_embedding": [],
        }
        result = SubInstructionNodeMatcher().match(node, sub_instruction)
        self.assertFalse(result.belongs)
        self.assertEqual(result.environment_semantic_score, 0.0)

    def test_failed_executor_stop_is_a_node_but_not_an_arrival(self):
        views = [np.zeros((16, 24, 3), np.uint8) for _ in range(6)]
        sub_instruction = InstructionDecomposer(FakeHarness()).decompose("exit")[0]
        with tempfile.TemporaryDirectory() as temporary:
            memory = NavigationGraphMemory(temporary, FakeSemanticExtractor())
            memory.add_origin_node([0, 0, 0], 0.0, 0, views)
            node, edge = memory.add_navigation_stop_node(
                [0, 0, 0], 0.0, 1, views, sub_instruction,
                [{"step": 0, "action": "blocked_forward", "moved_m": 0.0}],
                arrival_signal=None, metadata={"executor_end_reason": "max_steps"})
            self.assertEqual(node.node_kind, "point_navigation_stop")
            self.assertIsNone(node.arrival_signal)
            self.assertEqual(edge.target_node_id, node.node_id)
            self.assertEqual(memory.summary()["navigation_stop_node_count"], 1)

    def test_completion_judge_consumes_previous_node_edge_keyframes_and_current_node(self):
        views = [np.full((24, 32, 3), index * 20, np.uint8)
                 for index in range(6)]
        current_views = [np.full((24, 32, 3), 120 + index * 10, np.uint8)
                         for index in range(6)]
        keyframes = [views[0], np.full((24, 32, 3), 80, np.uint8),
                     current_views[0]]
        sub_instruction = InstructionDecomposer(FakeHarness()).decompose("exit")[0]
        with tempfile.TemporaryDirectory() as temporary:
            memory = NavigationGraphMemory(temporary, FakeSemanticExtractor())
            memory.add_origin_node([0, 0, 0], 0.0, 0, views)
            node, edge = memory.add_navigation_stop_node(
                [0, 0, -0.5], 0.0, 2, current_views, sub_instruction,
                [{"step": 0, "action": "forward", "moved_m": 0.5}],
                arrival_signal="point_navigation_arrived",
                edge_metadata={"edge_keyframes": [{"source_frame_index": 0}]})
            harness = NavigationVLMHarness(HeuristicBackend())
            result = NodeTransitionInstructionCompletionJudge(
                harness, memory, [sub_instruction]).judge(
                    node, sub_instruction.sub_instruction_id,
                    current_views, keyframes)
            self.assertTrue(result.instruction_completed)
            self.assertEqual(result.previous_node_id, "node_0000")
            self.assertEqual(result.current_node_id, node.node_id)
            self.assertEqual(result.incoming_edge_id, edge.edge_id)
            self.assertEqual(
                harness.calls[-1]["task"], "judge_edge_instruction_completion")
            self.assertEqual(len(harness.calls[-1]["image_paths"]), 0)

    def test_completion_judge_carries_prior_partial_real_edge_context(self):
        six = [np.full((24, 32, 3), index * 12, np.uint8)
               for index in range(6)]
        eight = [np.full((24, 32, 3), index * 12, np.uint8)
                 for index in range(8)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 2,
            "navigation_instruction": "walk past the couch",
            "landmark": "couch",
            "semantic_spatial_target": "floor beyond the couch",
            "form": "PASS_LANDMARK",
        })
        with tempfile.TemporaryDirectory() as temporary:
            memory = NavigationGraphMemory(temporary, FakeSemanticExtractor())
            memory.add_origin_node(
                [0, 0, 0], 0.0, 0, six, completion_views=eight)
            keyframe_dir = Path(temporary) / "edge_keyframes"
            keyframe_dir.mkdir()
            prior_keyframe_records = []
            for index, image in enumerate(eight[:2]):
                path = keyframe_dir / f"prior_{index}.jpg"
                Image.fromarray(image).save(path)
                prior_keyframe_records.append({
                    "keyframe_index": index,
                    "image_path": f"edge_keyframes/prior_{index}.jpg",
                })
            partial_node, prior_edge = memory.add_navigation_stop_node(
                [0, 0, -1], 0.0, 2, six, sub_instruction,
                [{"action": "forward", "moved_m": 1.0}],
                arrival_signal="point_navigation_arrived",
                completion_views=eight,
                edge_metadata={"edge_keyframes": prior_keyframe_records})
            carryover = {
                "active": True, "stage_id": 2,
                "source_node_id": "node_0000",
                "partial_node_id": partial_node.node_id,
                "prior_edge_id": prior_edge.edge_id,
                "prior_edge_traveled_distance_m": 2.1,
                "prior_edge_displacement_xz_m": [0.0, -2.0],
                "prior_endpoint_evidence": {
                    "current_target_state": "partial"},
            }
            node, edge = memory.add_navigation_stop_node(
                [0, 0, -2], 0.0, 4, six, sub_instruction,
                [{"action": "forward", "moved_m": 1.0}],
                arrival_signal="point_navigation_arrived",
                completion_views=eight,
                edge_metadata={"stage_progress_carryover": carryover})
            harness = NavigationVLMHarness(
                HeuristicBackend(),
                log_path=Path(temporary) / "vlm_calls.json",
                instruction_completion_prompt_version=(
                    "v23_relation_geometry_guard"))
            NodeTransitionInstructionCompletionJudge(
                harness, memory, [sub_instruction]).judge(
                    node, 2, eight, [eight[0], eight[1]])
            prompt = harness.calls[-1]["prompt"]
            self.assertIn("PRECEDING EDGE -> CURRENT EDGE", prompt)
            self.assertIn("prior_edge_traveled_distance_m", prompt)
            self.assertIn("STAGE-START", prompt)
            self.assertEqual(len(harness.calls[-1]["image_paths"]), 5)
            self.assertEqual(edge.metadata[
                "stage_progress_carryover"], carryover)

    def test_completion_harness_has_only_completed_unknown_semantics(self):
        views = [np.zeros((24, 32, 3), np.uint8) for _ in range(6)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 4,
            "navigation_instruction": "walk toward the sofa",
            "form": "APPROACH_LANDMARK",
            "semantic_spatial_target": "floor near sofa",
        })
        harness = NavigationVLMHarness(
            UnknownCompletionEvidenceBackend(),
            instruction_completion_prompt_version=(
                "v7_structured_partial_extent"))
        result = harness.judge_edge_instruction_completion(
            sub_instruction=sub_instruction,
            previous_node_id="node_0000", current_node_id="node_0001",
            previous_position_xyz=[0, 0, 0],
            current_position_xyz=[0, 0, -1],
            previous_six_views=views, current_six_views=views,
            previous_environment_semantics={"views": []},
            current_environment_semantics={"views": []},
            edge_action_history=[{"action": "forward", "moved_m": 1.0}],
            edge_keyframes=[views[0], views[1]])
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(any(key in result for key in (
            "on_route", "off_route", "belongs_to_sequence")))

    def test_eight_view_completion_storage_and_pass_prompt(self):
        six_views = [np.full((24, 32, 3), index * 20, np.uint8)
                     for index in range(6)]
        eight_views = [np.full((24, 32, 3), index * 15, np.uint8)
                       for index in range(8)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 1,
            "navigation_instruction": "walk straight passing the gray couch",
            "landmark": "gray couch",
            "form": "PASS_LANDMARK",
            "semantic_spatial_target": "floor beyond the gray couch",
        })
        harness = NavigationVLMHarness(
            HeuristicBackend(),
            instruction_completion_prompt_version=(
                "v8_eight_view_spatial_relations"))
        with tempfile.TemporaryDirectory() as temporary:
            memory = NavigationGraphMemory(temporary, FakeSemanticExtractor())
            memory.add_origin_node(
                [0, 0, 0], 0.0, 0, six_views,
                completion_views=eight_views)
            node, _ = memory.add_navigation_stop_node(
                [0, 0, -1], 0.0, 2, six_views, sub_instruction,
                [{"action": "forward", "moved_m": 1.0}],
                arrival_signal="point_navigation_arrived",
                completion_views=eight_views)
            stored = memory.load_node_completion_views(node.node_id)
            self.assertEqual(len(stored), 8)
            result = NodeTransitionInstructionCompletionJudge(
                harness, memory, [sub_instruction]).judge(
                    node, sub_instruction.sub_instruction_id,
                    eight_views, [eight_views[0], eight_views[1]])
            call = harness.calls[-1]
            self.assertTrue(result.instruction_completed)
            self.assertEqual(
                harness._completion_contact_sheet(eight_views).shape[:2],
                (48, 128))
            self.assertIn("VIEW 3 REAR_LEFT", call["prompt"])
            self.assertIn("need not disappear", call["prompt"])
            self.assertIn(
                "directional_evidence", call["schema"]["properties"])

    def test_v9_three_way_progress_returns_on_route_without_redundant_sectors(self):
        views = [np.zeros((24, 32, 3), np.uint8) for _ in range(8)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 1,
            "navigation_instruction": "walk through the doorway into the hall",
            "landmark": "doorway and hall", "form": "ENTER_REGION",
            "semantic_spatial_target": "inside the hall",
            "completion_cue": "doorway behind and hall surrounding the camera",
        })
        harness = NavigationVLMHarness(
            ThreeWayOnRouteBackend(),
            instruction_completion_prompt_version="v9_three_way_edge_progress")
        result = harness.judge_edge_instruction_completion(
            sub_instruction=sub_instruction,
            previous_node_id="node_0000", current_node_id="node_0001",
            previous_position_xyz=[0, 0, 0], current_position_xyz=[0, 0, -1],
            previous_six_views=views, current_six_views=views,
            previous_environment_semantics={"views": []},
            current_environment_semantics={"views": []},
            edge_action_history=[{"action": "forward", "moved_m": 1.0}],
            edge_keyframes=[views[0], views[1]])
        self.assertEqual(result["status"], "on_route")
        self.assertFalse(result["status_normalized"])
        call = harness.calls[-1]
        self.assertIn("ARRIVED", call["prompt"])
        self.assertIn("ON_ROUTE", call["prompt"])
        self.assertIn("UNKNOWN", call["prompt"])
        self.assertIn("progress_evidence", call["schema"]["properties"])
        self.assertNotIn("directional_evidence", call["schema"]["properties"])

    def test_v9_three_way_normalizes_contradictory_arrival_to_unknown(self):
        views = [np.zeros((24, 32, 3), np.uint8) for _ in range(8)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 2,
            "navigation_instruction": "walk forward toward the stairs",
            "landmark": "stairs", "form": "ADVANCE_STRAIGHT",
        })
        harness = NavigationVLMHarness(
            ThreeWayContradictoryClaimBackend(),
            instruction_completion_prompt_version="v9_three_way_edge_progress")
        result = harness.judge_edge_instruction_completion(
            sub_instruction=sub_instruction,
            previous_node_id="node_0000", current_node_id="node_0001",
            previous_position_xyz=[0, 0, 0], current_position_xyz=[0, 0, 1],
            previous_six_views=views, current_six_views=views,
            previous_environment_semantics={"views": []},
            current_environment_semantics={"views": []},
            edge_action_history=[{"action": "forward", "moved_m": 1.0}],
            edge_keyframes=[views[0], views[1]])
        self.assertEqual(result["model_status"], "arrived")
        self.assertEqual(result["status"], "unknown")
        self.assertTrue(result["status_normalized"])
        self.assertLessEqual(result["confidence"], 0.65)

    def test_v10_bidirectional_gate_rejects_reverse_fitting_edge(self):
        views = [np.zeros((24, 32, 3), np.uint8) for _ in range(8)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 2,
            "navigation_instruction": "walk forward toward the stairs",
            "landmark": "stairs", "form": "ADVANCE_STRAIGHT",
        })
        harness = NavigationVLMHarness(
            BidirectionalReverseBackend(),
            instruction_completion_prompt_version="v10_bidirectional_three_way")
        result = harness.judge_edge_instruction_completion(
            sub_instruction=sub_instruction,
            previous_node_id="node_0000", current_node_id="node_0001",
            previous_position_xyz=[0, 0, 0], current_position_xyz=[0, 0, 1],
            previous_six_views=views, current_six_views=views,
            previous_environment_semantics={"views": []},
            current_environment_semantics={"views": []},
            edge_action_history=[{"action": "forward", "moved_m": 1.0}],
            edge_keyframes=[views[0], views[1], views[2]])
        self.assertEqual(result["observed_status"], "on_route")
        self.assertEqual(result["reversed_status"], "arrived")
        self.assertEqual(result["direction_fit"], "reversed")
        self.assertEqual(result["status"], "unknown")
        self.assertTrue(result["bidirectional_unknown_gate"])
        call = harness.calls[-1]
        self.assertEqual(len(call["image_paths"]), 0)
        self.assertEqual(len(call["schema"]["properties"]), 6)
        self.assertIn("Image 3", call["prompt"])
        self.assertIn("No demonstration waypoint", call["prompt"])

    def test_unordered_endpoint_roles_do_not_receive_chronology(self):
        views = [np.zeros((24, 32, 3), np.uint8) for _ in range(8)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 3,
            "navigation_instruction": "go through the doorway into the lobby",
            "landmark": "doorway and lobby", "form": "ENTER_REGION",
            "semantic_spatial_target": "inside the lobby",
            "completion_cue": "doorway behind and lobby surrounding the camera",
        })
        harness = NavigationVLMHarness(HeuristicBackend())
        result = harness.classify_unordered_instruction_endpoint_roles(
            sub_instruction, views, views, {"views": []}, {"views": []})
        self.assertEqual(result["node_x_role"], "source_context")
        self.assertEqual(result["node_y_role"], "completion_target")
        self.assertEqual(result["preferred_order"], "x_to_y")
        prompt = harness.calls[-1]["prompt"]
        self.assertIn("deliberately shuffled", prompt)
        self.assertIn("Do not assume X precedes Y", prompt)
        self.assertIn("No chronology", prompt)

    def test_v12_binary_maps_partial_progress_to_unknown(self):
        views = [np.zeros((24, 32, 3), np.uint8) for _ in range(8)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 5,
            "navigation_instruction": "walk through the doorway into the hall",
            "landmark": "doorway and hall", "form": "ENTER_REGION",
            "semantic_spatial_target": "inside the hall",
            "completion_cue": "doorway behind and hall surrounding the camera",
        })
        harness = NavigationVLMHarness(
            BinaryPartialClaimBackend(),
            instruction_completion_prompt_version="v12_binary_completion")
        result = harness.judge_edge_instruction_completion(
            sub_instruction=sub_instruction,
            previous_node_id="node_0000", current_node_id="node_0001",
            previous_position_xyz=[0, 0, 0], current_position_xyz=[0, 0, -1],
            previous_six_views=views, current_six_views=views,
            previous_environment_semantics={"views": []},
            current_environment_semantics={"views": []},
            edge_action_history=[{"action": "forward", "moved_m": 1.0}],
            edge_keyframes=[views[0], views[1]])
        self.assertEqual(result["model_status"], "completed")
        self.assertEqual(result["status"], "unknown")
        self.assertTrue(result["status_normalized"])
        prompt = harness.calls[-1]["prompt"]
        self.assertIn("progress is always `unknown`", prompt)
        self.assertNotIn("on_route", harness.calls[-1]["schema"]["properties"])

    def test_v13_structures_node_edge_evidence_and_gates_reverse_claim(self):
        views = [np.full((24, 32, 3), index * 12, np.uint8)
                 for index in range(8)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 3,
            "navigation_instruction": "pass the couch into the hall",
            "landmark": "couch and hall", "form": "PASS_LANDMARK",
            "semantic_spatial_target": "hall beyond the couch",
            "completion_cue": "couch behind and hall reached",
        })
        harness = NavigationVLMHarness(
            StructuredReverseClaimBackend(),
            instruction_completion_prompt_version=(
                "v13_structured_node_edge_binary"))
        actions = [
            {"action": "turn_left", "turn_deg": 15.0, "moved_m": 0.0,
             "position_xyz": [0, 0, 0], "yaw_rad": 0.26},
            {"action": "forward", "turn_deg": 0.0, "moved_m": 1.0,
             "position_xyz": [0, 0, -1], "yaw_rad": 0.26},
        ]
        result = harness.judge_edge_instruction_completion(
            sub_instruction=sub_instruction,
            previous_node_id="node_0000", current_node_id="node_0001",
            previous_position_xyz=[0, 0, 0], current_position_xyz=[0, 0, -1],
            previous_six_views=views, current_six_views=list(reversed(views)),
            previous_environment_semantics={"views": []},
            current_environment_semantics={"views": []},
            edge_action_history=actions, edge_keyframes=[views[0], views[1]],
            previous_base_yaw_rad=0.0, current_base_yaw_rad=0.26,
            previous_visual_embedding=[1.0, 0.0],
            current_visual_embedding=[0.8, 0.2],
            edge_keyframe_records=[
                {"keyframe_index": 0, "source_frame_index": 0},
                {"keyframe_index": 1, "source_frame_index": 2}])
        self.assertEqual(result["status"], "unknown")
        self.assertTrue(result["status_normalized"])
        self.assertFalse(result["decision_gates"]["no_reverse_transition"])
        self.assertEqual(
            result["structured_motion_summary"]["trajectory_phases"][1]
            ["action"], "forward")
        self.assertIn("STRUCTURED_MOVEMENT_HISTORY", harness.calls[-1]["prompt"])
        self.assertEqual(harness.calls[-1]["schema"],
                         harness.STRUCTURED_BINARY_COMPLETION_SCHEMA)

    def test_v14_unordered_endpoint_roles_can_only_veto_reverse_completion(self):
        views = [np.full((24, 32, 3), index * 12, np.uint8)
                 for index in range(8)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 3,
            "navigation_instruction": "pass the couch into the hall",
            "landmark": "couch and hall", "form": "PASS_LANDMARK",
            "semantic_spatial_target": "hall beyond the couch",
            "completion_cue": "couch behind and hall reached",
        })
        harness = NavigationVLMHarness(
            StructuredCompletionThenReverseRolesBackend(),
            instruction_completion_prompt_version=(
                "v14_structured_with_unordered_reverse_veto"))
        result = harness.judge_edge_instruction_completion(
            sub_instruction=sub_instruction,
            previous_node_id="node_0000", current_node_id="node_0001",
            previous_position_xyz=[0, 0, 0], current_position_xyz=[0, 0, -1],
            previous_six_views=views, current_six_views=list(reversed(views)),
            previous_environment_semantics={"views": []},
            current_environment_semantics={"views": []},
            edge_action_history=[{"action": "forward", "moved_m": 1.0}],
            edge_keyframes=[views[0], views[1]])
        self.assertEqual(result["status"], "unknown")
        self.assertTrue(result["unordered_reverse_veto_applied"])
        self.assertTrue(
            result["unordered_endpoint_role_guard"]["reverse_completion_veto"])
        self.assertEqual(len(harness.calls), 2)
        self.assertIn(
            "Prompt version: v13_structured_node_edge_binary",
            harness.calls[0]["prompt"])
        self.assertIn("No chronology", harness.calls[1]["prompt"])

    def test_v15_requires_role_agreement_across_both_xy_assignments(self):
        views = [np.full((24, 32, 3), index * 12, np.uint8)
                 for index in range(8)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 3,
            "navigation_instruction": "pass the couch into the hall",
            "landmark": "couch and hall", "form": "PASS_LANDMARK",
            "semantic_spatial_target": "hall beyond the couch",
            "completion_cue": "couch behind and hall reached",
        })
        # This backend always labels X as the target. It therefore exposes the
        # X/Y positional bias observed in the real held-out failures: after the
        # physical nodes are swapped, the mapped endpoint roles must disagree.
        harness = NavigationVLMHarness(
            StructuredCompletionThenReverseRolesBackend(),
            instruction_completion_prompt_version=(
                "v15_structured_with_swap_consistent_reverse_veto"))
        result = harness.judge_edge_instruction_completion(
            sub_instruction=sub_instruction,
            previous_node_id="node_0000", current_node_id="node_0001",
            previous_position_xyz=[0, 0, 0], current_position_xyz=[0, 0, -1],
            previous_six_views=views, current_six_views=list(reversed(views)),
            previous_environment_semantics={"views": []},
            current_environment_semantics={"views": []},
            edge_action_history=[{"action": "forward", "moved_m": 1.0}],
            edge_keyframes=[views[0], views[1]])
        self.assertEqual(result["status"], "completed")
        guard = result["unordered_endpoint_role_guard"]
        self.assertTrue(guard["swap_consistency_required"])
        self.assertFalse(guard["permutation_consistent"])
        self.assertFalse(guard["reverse_completion_veto"])
        self.assertEqual(len(harness.calls), 3)

    def test_v17_uses_third_primary_call_only_to_break_disagreement(self):
        views = [np.full((24, 32, 3), index * 12, np.uint8)
                 for index in range(8)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 3,
            "navigation_instruction": "pass the couch into the hall",
            "landmark": "couch and hall", "form": "PASS_LANDMARK",
            "semantic_spatial_target": "hall beyond the couch",
            "completion_cue": "couch behind and hall reached",
        })
        harness = NavigationVLMHarness(
            StructuredAlternatingConsensusBackend(),
            instruction_completion_prompt_version=(
                "v17_primary_consensus_swap_veto"))
        result = harness.judge_edge_instruction_completion(
            sub_instruction=sub_instruction,
            previous_node_id="node_0000", current_node_id="node_0001",
            previous_position_xyz=[0, 0, 0], current_position_xyz=[0, 0, -1],
            previous_six_views=views, current_six_views=list(reversed(views)),
            previous_environment_semantics={"views": []},
            current_environment_semantics={"views": []},
            edge_action_history=[{"action": "forward", "moved_m": 1.0}],
            edge_keyframes=[views[0], views[1]])
        self.assertEqual(result["status"], "completed")
        consensus = result["primary_completion_consensus"]
        self.assertEqual(consensus["call_count"], 3)
        self.assertEqual(
            consensus["statuses"], ["completed", "unknown", "completed"])
        self.assertEqual(len(harness.calls), 5)

    def test_partial_turn_to_landmark_is_not_promoted_by_heading_alone(self):
        views = [np.zeros((24, 32, 3), np.uint8) for _ in range(8)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 0,
            "navigation_instruction": "turn towards the fireplace mantel",
            "landmark": "fireplace mantel",
            "form": "TURN_TO_LANDMARK",
            "semantic_spatial_target": "floor facing the mantel",
            "completion_cue": "mantel centered ahead",
        })
        harness = NavigationVLMHarness(
            StructuredPartialTurnLandmarkBackend(),
            instruction_completion_prompt_version="v19_vertical_guard_consensus")
        result = harness.judge_edge_instruction_completion(
            sub_instruction=sub_instruction,
            previous_node_id="node_0000", current_node_id="node_0001",
            previous_position_xyz=[0, 0, 0], current_position_xyz=[1, 0, -1],
            previous_base_yaw_rad=0.0, current_base_yaw_rad=1.0,
            previous_six_views=views, current_six_views=views,
            previous_environment_semantics={"views": []},
            current_environment_semantics={"views": []},
            edge_action_history=[
                {"action": "turn_left", "turn_deg": 57.0, "moved_m": 0.0},
                {"action": "forward", "turn_deg": 0.0, "moved_m": 1.4},
            ],
            edge_keyframes=[views[0], views[1]])
        self.assertEqual(result["status"], "unknown")

    def test_ambiguous_bare_turn_is_not_promoted_by_heading_alone(self):
        views = [np.zeros((24, 32, 3), np.uint8) for _ in range(8)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 0,
            "navigation_instruction": "turn right and cross the room",
            "landmark": "bed and doorway", "form": "TURN_RIGHT",
            "semantic_spatial_target": "doorway on the right",
            "completion_cue": "reach the doorway",
        })
        harness = NavigationVLMHarness(
            StructuredPartialTurnLandmarkBackend(),
            instruction_completion_prompt_version=(
                "v21_stage_endpoint_recovery_structured"))
        result = harness.judge_edge_instruction_completion(
            sub_instruction=sub_instruction,
            previous_node_id="node_0000", current_node_id="node_0001",
            previous_position_xyz=[0, 0, 0], current_position_xyz=[1, 0, -1],
            previous_base_yaw_rad=0.0, current_base_yaw_rad=-1.0,
            previous_six_views=views, current_six_views=views,
            previous_environment_semantics={"views": []},
            current_environment_semantics={"views": []},
            edge_action_history=[
                {"action": "forward", "turn_deg": 0.0, "moved_m": 1.4}],
            edge_keyframes=[views[0], views[1]])
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(result["endpoint_recovery"])

    def test_rear_present_eight_view_endpoint_completes_circumnavigation(self):
        views = [np.zeros((24, 32, 3), np.uint8) for _ in range(8)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 0,
            "navigation_instruction": "walk around the backside of the couches",
            "landmark": "couches", "form": "CIRCUMNAVIGATE",
            "semantic_spatial_target": "free floor behind the couches",
            "completion_cue": "couches are behind",
        })
        harness = NavigationVLMHarness(
            StructuredRearOnlyCircumnavigationBackend(),
            instruction_completion_prompt_version="v19_vertical_guard_consensus")
        result = harness.judge_edge_instruction_completion(
            sub_instruction=sub_instruction,
            previous_node_id="node_0000", current_node_id="node_0001",
            previous_position_xyz=[0, 0, 0], current_position_xyz=[0, 0, -2],
            previous_base_yaw_rad=0.0, current_base_yaw_rad=0.5,
            previous_six_views=views, current_six_views=views,
            previous_environment_semantics={"views": []},
            current_environment_semantics={"views": []},
            edge_action_history=[
                {"action": "forward", "turn_deg": 0.0, "moved_m": 2.0}],
            edge_keyframes=[views[0], views[1]])
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["circumnavigation_rear_endpoint_override"])

    def test_rear_only_circumnavigation_without_keyframe_stays_unknown(self):
        class MissingKeyframeBackend(StructuredRearOnlyCircumnavigationBackend):
            def generate_json(self, prompt, images, schema):
                result = super().generate_json(prompt, images, schema)
                result["temporal_evidence"]["keyframe_support"] = False
                return result

        views = [np.zeros((24, 32, 3), np.uint8) for _ in range(8)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 0,
            "navigation_instruction": "walk around the backside of the couches",
            "landmark": "couches", "form": "CIRCUMNAVIGATE",
            "semantic_spatial_target": "free floor behind the couches",
            "completion_cue": "couches are behind",
        })
        harness = NavigationVLMHarness(
            MissingKeyframeBackend(),
            instruction_completion_prompt_version=(
                "v21_stage_endpoint_recovery_structured"))
        result = harness.judge_edge_instruction_completion(
            sub_instruction=sub_instruction,
            previous_node_id="node_0000", current_node_id="node_0001",
            previous_position_xyz=[0, 0, 0],
            current_position_xyz=[0, 0, -2],
            previous_base_yaw_rad=0.0, current_base_yaw_rad=0.5,
            previous_six_views=views, current_six_views=views,
            previous_environment_semantics={"views": []},
            current_environment_semantics={"views": []},
            edge_action_history=[{"action": "forward", "moved_m": 2.0}],
            edge_keyframes=[views[0], views[1]])
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(result["circumnavigation_rear_endpoint_override"])

    def test_partial_ambiguous_rear_circumnavigation_stays_unknown(self):
        class PartialBackend(StructuredRearOnlyCircumnavigationBackend):
            def generate_json(self, prompt, images, schema):
                result = super().generate_json(prompt, images, schema)
                result["endpoint_evidence"]["current_target_state"] = "partial"
                result["endpoint_evidence"]["current_completion_cue"] = "partial"
                result["temporal_evidence"]["semantic_order"] = "ambiguous"
                result["temporal_evidence"][
                    "same_reference_instance"] = "ambiguous"
                return result

        views = [np.zeros((24, 32, 3), np.uint8) for _ in range(8)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 0,
            "navigation_instruction": "walk around the backside of the couches",
            "landmark": "couches", "form": "CIRCUMNAVIGATE",
            "semantic_spatial_target": "free floor behind the couches",
            "completion_cue": "couches are behind",
        })
        harness = NavigationVLMHarness(
            PartialBackend(), instruction_completion_prompt_version=(
                "v21_stage_endpoint_recovery_structured"))
        result = harness.judge_edge_instruction_completion(
            sub_instruction=sub_instruction,
            previous_node_id="node_0000", current_node_id="node_0001",
            previous_position_xyz=[0, 0, 0], current_position_xyz=[0, 0, -2],
            previous_base_yaw_rad=0.0, current_base_yaw_rad=0.5,
            previous_six_views=views, current_six_views=views,
            previous_environment_semantics={"views": []},
            current_environment_semantics={"views": []},
            edge_action_history=[{"action": "forward", "moved_m": 2.0}],
            edge_keyframes=[views[0], views[1]])
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(result["circumnavigation_rear_endpoint_override"])

    def test_v23_partial_rear_circumnavigation_cannot_override_consensus(self):
        class PartialBackend(StructuredRearOnlyCircumnavigationBackend):
            def generate_json(self, prompt, images, schema):
                result = super().generate_json(prompt, images, schema)
                result["endpoint_evidence"]["current_target_state"] = "partial"
                result["endpoint_evidence"]["current_completion_cue"] = "partial"
                result["temporal_evidence"]["semantic_order"] = "ambiguous"
                return result

        views = [np.zeros((24, 32, 3), np.uint8) for _ in range(8)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 0,
            "navigation_instruction": "walk around the backside of the couches",
            "landmark": "couches", "form": "CIRCUMNAVIGATE",
            "semantic_spatial_target": "free floor behind the couches",
            "completion_cue": "couches are behind",
        })
        result = NavigationVLMHarness(
            PartialBackend(), instruction_completion_prompt_version=(
                "v23_relation_geometry_guard")).judge_edge_instruction_completion(
                    sub_instruction=sub_instruction,
                    previous_node_id="node_0000",
                    current_node_id="node_0001",
                    previous_position_xyz=[0, 0, 0],
                    current_position_xyz=[0, 0, -2],
                    previous_base_yaw_rad=0.0,
                    current_base_yaw_rad=0.5,
                    previous_six_views=views, current_six_views=views,
                    previous_environment_semantics={"views": []},
                    current_environment_semantics={"views": []},
                    edge_action_history=[{
                        "action": "forward", "moved_m": 2.0}],
                    edge_keyframes=[views[0], views[1]])
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(result["circumnavigation_rear_endpoint_override"])

    def test_ordered_rear_visible_partial_exit_completes_crossing(self):
        views = [np.zeros((24, 32, 3), np.uint8) for _ in range(8)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 0,
            "navigation_instruction": "head through the door to outside",
            "landmark": "door to outside", "form": "EXIT_REGION",
            "semantic_spatial_target": "free floor beyond the doorway",
            "completion_cue": "cross the door frame",
        })
        harness = NavigationVLMHarness(
            StructuredPartialExitCrossingBackend(),
            instruction_completion_prompt_version="v19_vertical_guard_consensus")
        result = harness.judge_edge_instruction_completion(
            sub_instruction=sub_instruction,
            previous_node_id="node_0000", current_node_id="node_0001",
            previous_position_xyz=[0, 0, 0], current_position_xyz=[0, 0, -2],
            previous_base_yaw_rad=0.0, current_base_yaw_rad=0.0,
            previous_six_views=views, current_six_views=views,
            previous_environment_semantics={"views": []},
            current_environment_semantics={"views": []},
            edge_action_history=[
                {"action": "forward", "turn_deg": 0.0, "moved_m": 2.0}],
            edge_keyframes=[views[0], views[1]])
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["exit_temporal_crossing_override"])

    def test_v24_exit_requires_two_endpoint_side_audits_to_agree(self):
        class DisagreeingExitEndpointBackend(
                StructuredPartialExitCrossingBackend):
            def __init__(self):
                self.exit_audit_calls = 0

            def generate_json(self, prompt, images, schema):
                if prompt.startswith("EXIT_ENDPOINT_SIDE_AUDIT"):
                    self.exit_audit_calls += 1
                    if self.exit_audit_calls == 1:
                        return {
                            "endpoint_side": "destination_side",
                            "confidence": 0.95,
                            "reason": "first audit overcalls crossing",
                        }
                    return {
                        "endpoint_side": "source_or_threshold",
                        "confidence": 0.9,
                        "reason": "second audit still sees source room",
                    }
                result = super().generate_json(prompt, images, schema)
                result["status"] = "completed"
                result["confidence"] = 0.9
                result["endpoint_evidence"][
                    "current_target_state"] = "satisfied"
                result["endpoint_evidence"][
                    "current_completion_cue"] = "satisfied"
                result["temporal_evidence"][
                    "boundary_event"] = "observed"
                return result

        views = [np.zeros((24, 32, 3), np.uint8) for _ in range(8)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 0,
            "navigation_instruction": "exit through the door on the right",
            "landmark": "door on the right", "form": "EXIT_REGION",
            "semantic_spatial_target": "floor beyond the doorway",
            "completion_cue": "camera crosses the door frame",
        })
        backend = DisagreeingExitEndpointBackend()
        result = NavigationVLMHarness(
            backend, instruction_completion_prompt_version=(
                "v24_multireference_threshold_calibration"
            )).judge_edge_instruction_completion(
                sub_instruction=sub_instruction,
                previous_node_id="node_0000", current_node_id="node_0001",
                previous_position_xyz=[0, 0, 0],
                current_position_xyz=[0, 0, -2],
                previous_base_yaw_rad=0.0, current_base_yaw_rad=0.0,
                previous_six_views=views, current_six_views=views,
                previous_environment_semantics={"views": []},
                current_environment_semantics={"views": []},
                edge_action_history=[{
                    "action": "forward", "turn_deg": 0.0,
                    "moved_m": 2.0}],
                edge_keyframes=[views[0], views[1]])
        self.assertEqual(backend.exit_audit_calls, 2)
        self.assertEqual(result["status"], "unknown")
        self.assertTrue(result["exit_endpoint_side_veto_applied"])
        audit = result["exit_endpoint_side_audit"]
        self.assertEqual(audit["endpoint_side"], "source_or_threshold")
        self.assertFalse(audit["endpoint_side_consensus"][
            "destination_unanimous"])
        self.assertEqual(len(audit["endpoint_side_consensus"]["results"]), 2)

    def test_v24_all_portal_forms_require_endpoint_side_consensus(self):
        class DisagreeingPortalEndpointBackend(
                StructuredPartialExitCrossingBackend):
            def __init__(self):
                self.portal_audit_calls = 0

            def generate_json(self, prompt, images, schema):
                if prompt.startswith("PORTAL_ENDPOINT_SIDE_AUDIT"):
                    self.portal_audit_calls += 1
                    return {
                        "endpoint_side": (
                            "destination_side"
                            if self.portal_audit_calls % 2 else
                            "source_or_threshold"),
                        "confidence": 0.93,
                        "reason": "one raw-RGB audit still sees source side",
                    }
                result = super().generate_json(prompt, images, schema)
                result["status"] = "completed"
                result["confidence"] = 0.9
                result["endpoint_evidence"][
                    "current_target_state"] = "satisfied"
                result["endpoint_evidence"][
                    "current_completion_cue"] = "satisfied"
                result["temporal_evidence"]["boundary_event"] = "observed"
                return result

        views = [np.zeros((24, 32, 3), np.uint8) for _ in range(8)]
        for form in ("ENTER_REGION", "SELECT_PORTAL",
                     "TRAVERSE_PORTAL_REGION"):
            with self.subTest(form=form):
                sub_instruction = SubInstruction.from_mapping({
                    "sub_instruction_id": 0,
                    "navigation_instruction": "go through the selected door",
                    "landmark": "selected door", "form": form,
                    "semantic_spatial_target": "floor beyond the doorway",
                    "completion_cue": "camera crosses the door frame",
                })
                backend = DisagreeingPortalEndpointBackend()
                result = NavigationVLMHarness(
                    backend, instruction_completion_prompt_version=(
                        "v24_multireference_threshold_calibration"
                    )).judge_edge_instruction_completion(
                        sub_instruction=sub_instruction,
                        previous_node_id="node_0000",
                        current_node_id="node_0001",
                        previous_position_xyz=[0, 0, 0],
                        current_position_xyz=[0, 0, -2],
                        previous_base_yaw_rad=0.0,
                        current_base_yaw_rad=0.0,
                        previous_six_views=views,
                        current_six_views=views,
                        previous_environment_semantics={"views": []},
                        current_environment_semantics={"views": []},
                        edge_action_history=[{
                            "action": "forward", "turn_deg": 0.0,
                            "moved_m": 2.0}],
                        edge_keyframes=[views[0], views[1]])
                self.assertEqual(backend.portal_audit_calls, 2)
                self.assertEqual(result["status"], "unknown")
                self.assertTrue(result[
                    "portal_endpoint_side_veto_applied"])
                self.assertIs(result["portal_endpoint_side_audit"],
                              result["exit_endpoint_side_audit"])

    def test_v24_compound_terminal_extent_skips_portal_side_audit(self):
        class PortalAuditForbiddenBackend(StructuredSatisfiedEnterRegionBackend):
            def generate_json(self, prompt, images, schema):
                if prompt.startswith("PORTAL_ENDPOINT_SIDE_AUDIT"):
                    raise AssertionError(
                        "compound terminal extent must not run portal audit")
                return super().generate_json(prompt, images, schema)

        stage = {
            "sub_instruction_id": 1,
            "navigation_instruction": (
                "Walk into the kitchen and walk along the barstools and "
                "countertop until the arched entryway"),
            "landmark": "barstools and countertop",
            "form": "ENTER_REGION",
            "semantic_spatial_target": (
                "free floor at the far end of the countertop"),
            "spatial_relation": "along the countertop",
            "completion_cue": "reach the end of the countertop area",
            "visual_arrival_evidence": (
                "camera has passed the length of the counter"),
        }
        self.assertTrue(compound_terminal_extent_portal_stage(stage))
        views = [np.zeros((24, 32, 3), np.uint8) for _ in range(8)]
        result = NavigationVLMHarness(
            PortalAuditForbiddenBackend(),
            instruction_completion_prompt_version=(
                "v24_multireference_threshold_calibration"
            )).judge_edge_instruction_completion(
                sub_instruction=SubInstruction.from_mapping(stage),
                previous_node_id="node_0000",
                current_node_id="node_0001",
                previous_position_xyz=[0, 0, 0],
                current_position_xyz=[0, 0, -2],
                previous_base_yaw_rad=0.0,
                current_base_yaw_rad=0.0,
                previous_six_views=views,
                current_six_views=views,
                previous_environment_semantics={"views": []},
                current_environment_semantics={"views": []},
                edge_action_history=[{
                    "action": "forward", "turn_deg": 0.0,
                    "moved_m": 2.0}],
                edge_keyframes=[views[0], views[1]])
        self.assertNotIn("portal_endpoint_side_audit", result)
        self.assertNotIn("portal_endpoint_side_veto_applied", result)

    def test_explicit_exit_semantics_normalize_legacy_other_form(self):
        views = [np.zeros((24, 32, 3), np.uint8) for _ in range(8)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 0,
            "navigation_instruction": (
                "head towards the door that leads to the outside"),
            "landmark": "door to outside", "form": "OTHER",
            "semantic_spatial_target": "free floor just beyond the doorway",
            "completion_cue": "reach the doorway leading outside",
            "visual_arrival_evidence": (
                "camera crosses the door frame and sees outside area"),
        })
        harness = NavigationVLMHarness(
            StructuredPartialExitCrossingBackend(),
            instruction_completion_prompt_version=(
                "v21_stage_endpoint_recovery_structured"))
        result = harness.judge_edge_instruction_completion(
            sub_instruction=sub_instruction,
            previous_node_id="node_0000", current_node_id="node_0001",
            previous_position_xyz=[0, 0, 0], current_position_xyz=[0, 0, -2],
            previous_base_yaw_rad=0.0, current_base_yaw_rad=0.0,
            previous_six_views=views, current_six_views=views,
            previous_environment_semantics={"views": []},
            current_environment_semantics={"views": []},
            edge_action_history=[
                {"action": "forward", "turn_deg": 0.0, "moved_m": 2.0}],
            edge_keyframes=[views[0], views[1]])
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["exit_temporal_crossing_override"])

    def test_generic_other_doorway_approach_is_not_exit_recovery(self):
        views = [np.zeros((24, 32, 3), np.uint8) for _ in range(8)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 0,
            "navigation_instruction": "approach the doorway",
            "landmark": "doorway", "form": "OTHER",
            "semantic_spatial_target": "free floor near the doorway",
            "completion_cue": "doorway is ahead",
        })
        harness = NavigationVLMHarness(
            StructuredPartialExitCrossingBackend(),
            instruction_completion_prompt_version=(
                "v21_stage_endpoint_recovery_structured"))
        result = harness.judge_edge_instruction_completion(
            sub_instruction=sub_instruction,
            previous_node_id="node_0000", current_node_id="node_0001",
            previous_position_xyz=[0, 0, 0], current_position_xyz=[0, 0, -2],
            previous_base_yaw_rad=0.0, current_base_yaw_rad=0.0,
            previous_six_views=views, current_six_views=views,
            previous_environment_semantics={"views": []},
            current_environment_semantics={"views": []},
            edge_action_history=[
                {"action": "forward", "turn_deg": 0.0, "moved_m": 2.0}],
            edge_keyframes=[views[0], views[1]])
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(result["exit_temporal_crossing_override"])

    def test_partial_enter_region_is_not_promoted_by_endpoint_recovery(self):
        views = [np.zeros((24, 32, 3), np.uint8) for _ in range(8)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 2,
            "navigation_instruction": "walk into the entrance under the balcony",
            "landmark": "balcony", "form": "ENTER_REGION",
            "semantic_spatial_target": "entrance floor under the balcony",
            "spatial_relation": "under the balcony, inside the entrance",
            "completion_cue": "camera is under the balcony",
        })
        semantics = {"views": [{
            "view_index": 0,
            "detections": [{"label": "balcony", "score": 0.45}],
        }]}
        harness = NavigationVLMHarness(
            StructuredPartialEnterRegionBackend(),
            instruction_completion_prompt_version=(
                "v21_stage_endpoint_recovery_structured"))
        result = harness.judge_edge_instruction_completion(
            sub_instruction=sub_instruction,
            previous_node_id="node_0000", current_node_id="node_0001",
            previous_position_xyz=[0, 0, 0], current_position_xyz=[0, 0, -2],
            previous_base_yaw_rad=0.0, current_base_yaw_rad=0.0,
            previous_six_views=views, current_six_views=views,
            previous_environment_semantics={"views": []},
            current_environment_semantics=semantics,
            edge_action_history=[{"action": "forward", "moved_m": 2.0}],
            edge_keyframes=[views[0], views[1]])
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(result["endpoint_recovery"])
        self.assertFalse(result["endpoint_recovery_evidence"]
                               ["enter_full_endpoint_required"])

    def test_qualified_enter_region_requires_qualifier_detection(self):
        views = [np.zeros((24, 32, 3), np.uint8) for _ in range(8)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 2,
            "navigation_instruction": "enter the room with cardboard boxes",
            "landmark": "room with cardboard boxes", "form": "ENTER_REGION",
            "semantic_spatial_target": "floor inside the room with cardboard boxes",
            "spatial_relation": "inside the room with cardboard boxes",
            "completion_cue": "camera is inside the room with cardboard boxes",
        })
        harness = NavigationVLMHarness(
            StructuredSatisfiedEnterRegionBackend(),
            instruction_completion_prompt_version=(
                "v21_stage_endpoint_recovery_structured"))
        common = dict(
            sub_instruction=sub_instruction,
            previous_node_id="node_0000", current_node_id="node_0001",
            previous_position_xyz=[0, 0, 0], current_position_xyz=[0, 0, -2],
            previous_base_yaw_rad=0.0, current_base_yaw_rad=0.0,
            previous_six_views=views, current_six_views=views,
            previous_environment_semantics={"views": []},
            edge_action_history=[{"action": "forward", "moved_m": 2.0}],
            edge_keyframes=[views[0], views[1]],
            carryover_evidence={"stage_progress": None,
                                "current_edge": None})
        missing = harness.judge_edge_instruction_completion(
            current_environment_semantics={"views": [{
                "view_index": 0,
                "detections": [{"label": "doorway room", "score": 0.65}],
            }]}, **common)
        present = harness.judge_edge_instruction_completion(
            current_environment_semantics={"views": [{
                "view_index": 0,
                "detections": [{"label": "cardboard boxes", "score": 0.65}],
            }]}, **common)
        self.assertEqual(missing["status"], "unknown")
        self.assertFalse(missing["decision_gates"]
                                ["qualified_region_identity_supported"])
        self.assertEqual(present["status"], "completed")
        self.assertTrue(present["decision_gates"]
                               ["qualified_region_identity_supported"])

    def test_compound_enter_along_extent_rejects_landmark_still_ahead(self):
        views = [np.zeros((24, 32, 3), np.uint8) for _ in range(8)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 1,
            "navigation_instruction": (
                "Walk into the kitchen and walk along the barstools and "
                "countertop"),
            "landmark": "barstools and countertop", "form": "ENTER_REGION",
            "semantic_spatial_target": (
                "free floor along the barstools and countertop"),
            "spatial_relation": "alongside the barstools and countertop",
            "completion_cue": "Reach the end of the countertop area",
            "visual_arrival_evidence": (
                "You have passed the length of the countertop and barstools"),
        })
        harness = NavigationVLMHarness(
            StructuredSatisfiedEnterRegionBackend(),
            instruction_completion_prompt_version=(
                "v21_stage_endpoint_recovery_structured"))
        result = harness.judge_edge_instruction_completion(
            sub_instruction=sub_instruction,
            previous_node_id="node_0000", current_node_id="node_0001",
            previous_position_xyz=[0, 0, 0], current_position_xyz=[0, 0, -2],
            previous_base_yaw_rad=0.0, current_base_yaw_rad=0.0,
            previous_six_views=views, current_six_views=views,
            previous_environment_semantics={"views": []},
            current_environment_semantics={"views": [{
                "view_index": 0,
                "detections": [{
                    "label": "counter countertop", "score": 0.55}],
            }]},
            edge_action_history=[{"action": "forward", "moved_m": 2.0}],
            edge_keyframes=[views[0], views[1]])
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(result["decision_gates"]
                               ["explicit_terminal_extent_front_clear"])

    def test_forward_stair_detection_vetoes_false_top_landing(self):
        views = [np.zeros((24, 32, 3), np.uint8) for _ in range(8)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 0,
            "navigation_instruction": "walk up the stairs to the top",
            "landmark": "stairs", "form": "VERTICAL_UP",
            "semantic_spatial_target": "top landing of the stairs",
            "completion_cue": "reach the top of the stairs",
        })
        current_semantics = {"views": [
            {"view_index": index, "detections": (
                [{"label": "stair flight", "score": 0.34}]
                if index == 0 else [])}
            for index in range(6)]}
        for prompt_version in (
                "v19_vertical_guard_consensus",
                "v21_stage_endpoint_recovery_structured"):
            with self.subTest(prompt_version=prompt_version):
                harness = NavigationVLMHarness(
                    StructuredFalseTopVerticalBackend(),
                    instruction_completion_prompt_version=prompt_version)
                result = harness.judge_edge_instruction_completion(
                    sub_instruction=sub_instruction,
                    previous_node_id="node_0000", current_node_id="node_0001",
                    previous_position_xyz=[0, 0, 0],
                    current_position_xyz=[0, 1, -1],
                    previous_base_yaw_rad=0.0, current_base_yaw_rad=0.0,
                    previous_six_views=views, current_six_views=views,
                    previous_environment_semantics={"views": []},
                    current_environment_semantics=current_semantics,
                    edge_action_history=[
                        {"action": "forward", "moved_m": 1.4}],
                    edge_keyframes=[views[0], views[1]])
                self.assertEqual(result["status"], "unknown")
                self.assertFalse(
                    result["decision_gates"]["no_remaining_stair_flight"])
                self.assertTrue(
                    result["vertical_endpoint_evidence"]
                          ["detector_remaining_stair_front"])

    def test_side_completed_flight_and_landing_label_do_not_veto_top(self):
        views = [np.zeros((24, 32, 3), np.uint8) for _ in range(8)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 0,
            "navigation_instruction": "walk up the stairs to the top",
            "landmark": "stairs", "form": "VERTICAL_UP",
            "semantic_spatial_target": "top landing of the stairs",
            "completion_cue": "reach the top of the stairs",
        })
        # The backend supplies all clean-RGB/keyframe endpoint gates as true.
        # A landing proposal straight ahead and the just-completed flight in a
        # side-forward camera are compatible with standing on the upper floor.
        current_semantics = {"views": [
            {"view_index": index, "detections": (
                [{"label": "stairs stair landing", "score": 0.36}]
                if index == 0 else
                [{"label": "stair flight", "score": 0.40}]
                if index == 5 else [])}
            for index in range(6)]}
        harness = NavigationVLMHarness(
            StructuredFalseTopVerticalBackend(),
            instruction_completion_prompt_version=(
                "v21_stage_endpoint_recovery_structured"))
        result = harness.judge_edge_instruction_completion(
            sub_instruction=sub_instruction,
            previous_node_id="node_0000", current_node_id="node_0001",
            previous_position_xyz=[0, 0, 0],
            current_position_xyz=[0, 1.4, -1],
            previous_base_yaw_rad=0.0, current_base_yaw_rad=0.0,
            previous_six_views=views, current_six_views=views,
            previous_environment_semantics={"views": []},
            current_environment_semantics=current_semantics,
            edge_action_history=[{"action": "forward", "moved_m": 1.4}],
            edge_keyframes=[views[0], views[1]])
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["vertical_endpoint_evidence"]
                               ["detector_remaining_stair_front"])

    def test_ordered_rear_sector_pass_allows_long_landmark_front_overlap(self):
        views = [np.zeros((24, 32, 3), np.uint8) for _ in range(8)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 1,
            "navigation_instruction": "walk past the stairs and bathroom",
            "landmark": "stairs and bathroom", "form": "PASS_LANDMARK",
            "semantic_spatial_target": "hall beyond the stairs and bathroom",
            "completion_cue": "stairs and bathroom are behind",
        })
        harness = NavigationVLMHarness(
            StructuredRearAndFrontPassBackend(),
            instruction_completion_prompt_version=(
                "v21_stage_endpoint_recovery_structured"))
        result = harness.judge_edge_instruction_completion(
            sub_instruction=sub_instruction,
            previous_node_id="node_0000", current_node_id="node_0001",
            previous_position_xyz=[0, 0, 0], current_position_xyz=[0, 0, -1],
            previous_base_yaw_rad=0.0, current_base_yaw_rad=0.0,
            previous_six_views=views, current_six_views=views,
            previous_environment_semantics={"views": []},
            current_environment_semantics={"views": []},
            edge_action_history=[
                {"action": "forward", "turn_deg": 0.0, "moved_m": 1.0}],
            edge_keyframes=[views[0], views[1]])
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["pass_directional_override"])

    def test_runtime_adapter_keeps_ordered_rear_pass_with_front_overlap(self):
        six_views = [np.zeros((24, 32, 3), np.uint8) for _ in range(6)]
        eight_views = [np.zeros((24, 32, 3), np.uint8) for _ in range(8)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 1,
            "navigation_instruction": "walk past the stairs and bathroom",
            "landmark": "stairs and bathroom", "form": "PASS_LANDMARK",
            "semantic_spatial_target": "hall beyond the landmarks",
            "completion_cue": "landmarks are behind",
        })
        harness = NavigationVLMHarness(
            StructuredRearAndFrontPassBackend(),
            instruction_completion_prompt_version=(
                "v21_stage_endpoint_recovery_structured"))
        with tempfile.TemporaryDirectory() as temporary:
            memory = NavigationGraphMemory(temporary, FakeSemanticExtractor())
            memory.add_origin_node(
                [0, 0, 0], 0.0, 0, six_views,
                completion_views=eight_views)
            node, _ = memory.add_navigation_stop_node(
                [0, 0, -1], 0.0, 1, six_views, sub_instruction,
                [{"action": "forward", "turn_deg": 0.0, "moved_m": 1.0}],
                arrival_signal="point_navigation_arrived",
                completion_views=eight_views)
            result = NodeTransitionInstructionCompletionJudge(
                harness, memory, [sub_instruction]).judge(
                    node, sub_instruction.sub_instruction_id,
                    eight_views, [eight_views[0], eight_views[1]])
        self.assertEqual(result.status, "completed")
        self.assertTrue(result.validation_overrides[
            "pass_directional_override"])

    def test_local_occlusion_boundary_cannot_complete_pass(self):
        six_views = [np.zeros((24, 32, 3), np.uint8) for _ in range(6)]
        eight_views = [np.zeros((24, 32, 3), np.uint8) for _ in range(8)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 1,
            "navigation_instruction": "walk past the stairs and bathroom",
            "landmark": "stairs and bathroom", "form": "PASS_LANDMARK",
            "semantic_spatial_target": "hall beyond both landmarks",
            "completion_cue": "stairs and bathroom are behind",
        })
        harness = NavigationVLMHarness(
            StructuredRearAndFrontPassBackend(),
            instruction_completion_prompt_version=(
                "v21_stage_endpoint_recovery_structured"))
        with tempfile.TemporaryDirectory() as temporary:
            memory = NavigationGraphMemory(temporary, FakeSemanticExtractor())
            memory.add_origin_node(
                [0, 0, 0], 0.0, 0, six_views,
                completion_views=eight_views)
            node, _ = memory.add_navigation_stop_node(
                [0, 0, -1], 0.0, 1, six_views, sub_instruction,
                [{"action": "forward", "turn_deg": 0.0, "moved_m": 1.0}],
                arrival_signal="point_navigation_arrived",
                completion_views=eight_views,
                edge_metadata={"point_selection_review": {
                    "postselection_repair": {
                        "status": "repaired_to_local_occlusion_boundary"}}})
            result = NodeTransitionInstructionCompletionJudge(
                harness, memory, [sub_instruction]).judge(
                    node, sub_instruction.sub_instruction_id,
                    eight_views, [eight_views[0], eight_views[1]])
        self.assertEqual(result.status, "unknown")
        self.assertFalse(result.instruction_completed)
        self.assertTrue(result.validation_overrides[
            "local_occlusion_boundary_intermediate_veto"])

    def test_partial_multi_landmark_pass_cannot_use_rear_override(self):
        class PartialBackend(StructuredRearAndFrontPassBackend):
            def generate_json(self, prompt, images, schema):
                result = super().generate_json(prompt, images, schema)
                result["endpoint_evidence"]["current_target_state"] = "partial"
                result["endpoint_evidence"]["current_completion_cue"] = "partial"
                result["temporal_evidence"][
                    "same_reference_instance"] = "ambiguous"
                return result

        views = [np.zeros((24, 32, 3), np.uint8) for _ in range(8)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 1,
            "navigation_instruction": "walk past the stairs and bathroom",
            "landmark": "stairs and bathroom", "form": "PASS_LANDMARK",
            "semantic_spatial_target": "hall beyond both landmarks",
            "completion_cue": "stairs and bathroom are behind",
        })
        harness = NavigationVLMHarness(
            PartialBackend(), instruction_completion_prompt_version=(
                "v21_stage_endpoint_recovery_structured"))
        result = harness.judge_edge_instruction_completion(
            sub_instruction=sub_instruction,
            previous_node_id="node_0000", current_node_id="node_0001",
            previous_position_xyz=[0, 0, 0], current_position_xyz=[0, 0, -1],
            previous_base_yaw_rad=0.0, current_base_yaw_rad=0.0,
            previous_six_views=views, current_six_views=views,
            previous_environment_semantics={"views": []},
            current_environment_semantics={"views": []},
            edge_action_history=[{"action": "forward", "moved_m": 1.0}],
            edge_keyframes=[views[0], views[1]])
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(result["pass_directional_override"])

    def test_runtime_adapter_preserves_endpoint_override_audit_flags(self):
        six_views = [np.zeros((24, 32, 3), np.uint8) for _ in range(6)]
        eight_views = [np.zeros((24, 32, 3), np.uint8) for _ in range(8)]
        cases = (
            ({
                "sub_instruction_id": 0,
                "navigation_instruction": "walk around the backside of the couches",
                "landmark": "couches", "form": "CIRCUMNAVIGATE",
                "semantic_spatial_target": "free floor behind the couches",
                "completion_cue": "couches are behind",
             }, StructuredRearOnlyCircumnavigationBackend(),
             "circumnavigation_rear_endpoint_override"),
            ({
                "sub_instruction_id": 0,
                "navigation_instruction": "head through the door to outside",
                "landmark": "door to outside", "form": "EXIT_REGION",
                "semantic_spatial_target": "free floor beyond the doorway",
                "completion_cue": "cross the door frame",
             }, StructuredPartialExitCrossingBackend(),
             "exit_temporal_crossing_override"),
        )
        for mapping, backend, audit_key in cases:
            with self.subTest(audit_key=audit_key), tempfile.TemporaryDirectory() as temporary:
                sub_instruction = SubInstruction.from_mapping(mapping)
                harness = NavigationVLMHarness(
                    backend,
                    instruction_completion_prompt_version=(
                        "v19_vertical_guard_consensus"))
                memory = NavigationGraphMemory(temporary, FakeSemanticExtractor())
                memory.add_origin_node(
                    [0, 0, 0], 0.0, 0, six_views,
                    completion_views=eight_views)
                node, _ = memory.add_navigation_stop_node(
                    [0, 0, -2], 0.5, 1, six_views, sub_instruction,
                    [{"action": "forward", "turn_deg": 0.0, "moved_m": 2.0}],
                    arrival_signal="point_navigation_arrived",
                    completion_views=eight_views)
                result = NodeTransitionInstructionCompletionJudge(
                    harness, memory, [sub_instruction]).judge(
                        node, sub_instruction.sub_instruction_id,
                        eight_views, [eight_views[0], eight_views[1]])
                self.assertEqual(result.status, "completed")
                self.assertTrue(result.validation_overrides[audit_key])

    def test_runtime_adapter_collapses_latent_on_route_to_unknown(self):
        six_views = [np.zeros((24, 32, 3), np.uint8) for _ in range(6)]
        eight_views = [np.zeros((24, 32, 3), np.uint8) for _ in range(8)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 6,
            "navigation_instruction": "walk through the doorway into the hall",
            "landmark": "doorway and hall", "form": "ENTER_REGION",
            "semantic_spatial_target": "inside the hall",
        })
        harness = NavigationVLMHarness(
            ThreeWayOnRouteBackend(),
            instruction_completion_prompt_version="v9_three_way_edge_progress")
        with tempfile.TemporaryDirectory() as temporary:
            memory = NavigationGraphMemory(temporary, FakeSemanticExtractor())
            memory.add_origin_node(
                [0, 0, 0], 0.0, 0, six_views,
                completion_views=eight_views)
            node, _ = memory.add_navigation_stop_node(
                [0, 0, -1], 0.0, 1, six_views, sub_instruction,
                [{"action": "forward", "moved_m": 1.0}],
                arrival_signal="point_navigation_arrived",
                completion_views=eight_views)
            result = NodeTransitionInstructionCompletionJudge(
                harness, memory, [sub_instruction]).judge(
                    node, sub_instruction.sub_instruction_id,
                    eight_views, [eight_views[0], eight_views[1]])
        self.assertEqual(result.status, "unknown")
        self.assertFalse(result.instruction_completed)
        self.assertFalse(result.instruction_on_route)

    def test_eight_view_pass_uses_detector_to_reject_front_dominance(self):
        views = [np.zeros((24, 32, 3), np.uint8) for _ in range(8)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 1,
            "navigation_instruction": "pass the gray couch",
            "landmark": "gray couch", "form": "PASS_LANDMARK",
        })
        semantics = {
            "views": [
                {"view_index": 0, "detections": [
                    {"label": "sofa gray couch", "score": 0.70,
                     "median_depth_m": 1.8}]},
                {"view_index": 3, "detections": [
                    {"label": "sofa gray couch", "score": 0.35,
                     "median_depth_m": 3.2}]},
            ]}
        harness = NavigationVLMHarness(
            RearPassClaimBackend(),
            instruction_completion_prompt_version=(
                "v8_eight_view_spatial_relations"))
        result = harness.judge_edge_instruction_completion(
            sub_instruction=sub_instruction,
            previous_node_id="node_0000", current_node_id="node_0001",
            previous_position_xyz=[0, 0, 0], current_position_xyz=[0, 0, -1],
            previous_six_views=views, current_six_views=views,
            previous_environment_semantics={"views": []},
            current_environment_semantics=semantics,
            edge_action_history=[{"action": "forward", "moved_m": 1.0}],
            edge_keyframes=[views[0], views[1]])
        self.assertEqual(result["status"], "unknown")
        self.assertTrue(result["detector_directional_override"])

    def test_partial_stair_harness_fuses_node_and_edge_runtime_evidence(self):
        views = [np.zeros((24, 32, 3), np.uint8) for _ in range(6)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 2,
            "navigation_instruction": "go about halfway up the stairs",
            "form": "VERTICAL_UP",
            "semantic_spatial_target": "a step approximately halfway up",
            "completion_cue": "several steps below and above",
            "visual_arrival_evidence": "stairs remain below and above",
        })
        harness = NavigationVLMHarness(
            UnknownCompletionEvidenceBackend(),
            instruction_completion_prompt_version=(
                "v7_structured_partial_extent"))
        result = harness.judge_edge_instruction_completion(
            sub_instruction=sub_instruction,
            previous_node_id="node_0000", current_node_id="node_0001",
            previous_position_xyz=[0, 0, 0],
            current_position_xyz=[0, 1.0, -1.0],
            previous_six_views=views, current_six_views=views,
            previous_environment_semantics=self.stair_semantics(1),
            current_environment_semantics=self.stair_semantics(3),
            edge_action_history=[{"action": "forward", "moved_m": 1.1}],
            edge_keyframes=[views[0], views[1]])
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["node_edge_operational_override"])

    def test_ancestor_route_and_six_view_backtrack_selection(self):
        target_views = [np.full((40, 60, 3), 20 + index, np.uint8)
                        for index in range(6)]
        current_views = [np.full((40, 60, 3), 80 + index, np.uint8)
                         for index in range(6)]
        depths = [np.ones((40, 60), np.float32) for _ in range(6)]
        sub_instruction = InstructionDecomposer(FakeHarness()).decompose("exit")[0]
        with tempfile.TemporaryDirectory() as temporary:
            memory = NavigationGraphMemory(temporary, FakeSemanticExtractor())
            memory.add_origin_node([0, 0, 0], 0.0, 0, target_views)
            node, edge = memory.add_navigation_stop_node(
                [0, 0, -2], 0.0, 1, current_views, sub_instruction,
                [{"step": 0, "action": "forward", "moved_m": 2.0}],
                arrival_signal="point_navigation_arrived")
            self.assertEqual(memory.ancestor_path(node.node_id, "node_0000"),
                             ["node_0001", "node_0000"])
            self.assertEqual(memory.get_edge("node_0000", node.node_id).edge_id,
                             edge.edge_id)

            selector = NodeBacktrackingPointSelector(
                memory, FakeFloorSegmenter(), mode="hybrid")
            selection = selector.select(
                current_node_id=node.node_id, target_node_id="node_0000",
                current_rgbs=current_views, current_depths=depths,
                current_position=node.position_xyz, current_yaw=0.0,
                forward_action_history=edge.action_history)
            # From z=-2 to z=0 is the reverse-facing 180-degree panorama view.
            self.assertEqual(selection.view_index, 3)
            x, y = np.round(selection.point_xy).astype(int)
            self.assertTrue(selection.target_mask[y, x])
            self.assertEqual(
                selection.record["forward_action_history_reversed_for_context"][0]
                ["action"], "forward")

    def test_reverse_breadcrumb_follows_executed_corner_not_endpoint_chord(self):
        class Edge:
            action_history = [
                {"position_xyz": [0, 0, -2]},
                {"position_xyz": [-3, 0, -2]},
                {"position_xyz": [-3, 0, 0]},
            ]
            metadata = {}

        result = reverse_route_breadcrumb(
            Edge(), target_position=[0, 0, 0], source_position=[-3, 0, 0],
            current_position=[-3, 0, 0], lookahead_m=1.5)
        # The reverse trace first returns down the final corridor; a direct
        # endpoint chord would incorrectly point east through the corner.
        self.assertAlmostEqual(result["position_xyz"][0], -3.0, places=4)
        self.assertAlmostEqual(result["position_xyz"][2], -1.5, places=4)
        self.assertEqual(result["trace_source"], "forward_action_pose_trace")
        self.assertAlmostEqual(result["trace_length_m"], 7.0, places=4)

    def test_backtrack_budget_contains_one_discrete_endpoint_step(self):
        breadcrumb = {
            "current_route_progress_m": 1.0,
            "target_route_progress_m": 1.0,
            "off_trace_distance_m": 0.795,
        }
        budget = backtrack_segment_travel_budget(
            breadcrumb, controller_step_m=0.22,
            settling_margin_m=0.10)
        self.assertAlmostEqual(budget, 1.115, places=6)
        self.assertGreaterEqual(budget - 0.795, 0.22)

    def test_v4_short_range_recovery_guides_executor_to_stored_node(self):
        selected = np.array([4.0, 0.0, 2.0], np.float32)
        stored = np.array([1.0, 0.0, 0.0], np.float32)
        endpoint, geodesic, source = recovery_executor_endpoint(
            "breadcrumb_endpoint_v4", {}, selected, 3.0, stored, 1.1)
        np.testing.assert_allclose(endpoint, stored)
        self.assertAlmostEqual(geodesic, 1.1)
        self.assertEqual(source, "stored_node_short_range")
        far_endpoint, far_geodesic, far_source = recovery_executor_endpoint(
            "breadcrumb_endpoint_v4", {}, selected, 3.0, stored, 1.6)
        np.testing.assert_allclose(far_endpoint, selected)
        self.assertAlmostEqual(far_geodesic, 3.0)
        self.assertEqual(far_source, "selected_floor_anchor")

    def test_v5_extends_stored_node_endpoint_guidance_to_three_metres(self):
        selected = np.array([4.0, 0.0, 2.0], np.float32)
        stored = np.array([1.0, 0.0, 0.0], np.float32)
        endpoint, geodesic, source = recovery_executor_endpoint(
            "breadcrumb_endpoint_v5", {}, selected, 4.0, stored, 2.8)
        np.testing.assert_allclose(endpoint, stored)
        self.assertAlmostEqual(geodesic, 2.8)
        self.assertEqual(source, "stored_node_short_range")
        far_endpoint, far_geodesic, far_source = recovery_executor_endpoint(
            "breadcrumb_endpoint_v5", {}, selected, 4.0, stored, 3.1)
        np.testing.assert_allclose(far_endpoint, selected)
        self.assertAlmostEqual(far_geodesic, 4.0)
        self.assertEqual(far_source, "selected_floor_anchor")

    def test_compound_turn_requires_destination_not_only_heading(self):
        self.assertTrue(compound_turn_requires_semantic_endpoint(
            "TURN_LEFT", "Turn to the left and walk towards the far side"))
        self.assertFalse(compound_turn_requires_semantic_endpoint(
            "TURN_LEFT", "Turn to the left and walk towards the far side",
            directional_towards_is_commitment=True))
        self.assertTrue(compound_turn_requires_semantic_endpoint(
            "TURN_RIGHT", "Turn right into the hallway"))
        self.assertTrue(compound_turn_requires_semantic_endpoint(
            "TURN_RIGHT", "Turn right and walk across the room to the doorway",
            directional_towards_is_commitment=True))
        self.assertFalse(compound_turn_requires_semantic_endpoint(
            "TURN_LEFT", "Turn left."))

    def test_compound_turn_portal_threshold_requires_ordered_side_straddle(self):
        common = dict(
            form="TURN_RIGHT",
            instruction_text=(
                "Turn right and walk across the room to the doorway"),
            current_reference_sectors={"front_right", "rear_right"},
            semantic_order="instructed", boundary_event="observed",
            keyframe_support=True,
            same_reference_instance="yes", motion_fit="supports",
            stationary=False, reverse_observed=False,
            deterministic_turn_direction=True,
            horizontal_displacement_m=2.0, traveled_distance_m=2.4)
        self.assertTrue(compound_turn_portal_threshold_supported(**common))
        # The terminal camera can be corrected past the doorway normal.  The
        # action-history guard, not the terminal camera side, proves the turn.
        self.assertTrue(compound_turn_portal_threshold_supported(
            **dict(common, current_reference_sectors={
                "front_left", "rear_left"})))
        self.assertFalse(compound_turn_portal_threshold_supported(
            **dict(common, current_reference_sectors={"front_right"})))
        self.assertFalse(compound_turn_portal_threshold_supported(
            **dict(common, current_reference_sectors={
                "front_left", "rear_right"})))
        self.assertFalse(compound_turn_portal_threshold_supported(
            **dict(common, semantic_order="ambiguous")))
        self.assertFalse(compound_turn_portal_threshold_supported(
            **dict(common, boundary_event="not_observed")))
        self.assertFalse(compound_turn_portal_threshold_supported(
            **dict(common, deterministic_turn_direction=False)))

    def test_compound_turn_partial_consumes_direction_for_one_continuation(self):
        stage = {
            "form": "TURN_RIGHT",
            "navigation_instruction": (
                "Turn right and walk across the room to the doorway"),
        }
        progress = {
            "active": True,
            "prior_selected_yaw_rad": -0.8,
            "prior_endpoint_evidence": {
                "current_target_state": "partial"},
            "prior_motion_evidence": {
                "instruction_motion_fit": "supports",
                "stationary_or_blocked": False},
        }
        self.assertTrue(compound_turn_continuation_required(stage, progress))
        self.assertFalse(compound_turn_continuation_required(
            {"form": "TURN_RIGHT", "navigation_instruction": "Turn right"},
            progress))
        self.assertFalse(compound_turn_continuation_required(
            stage, {**progress, "prior_motion_evidence": {
                "instruction_motion_fit": "contradicts",
                "stationary_or_blocked": False}}))

    def test_between_gap_endpoint_requires_opposing_lateral_references(self):
        common = dict(
            form="BETWEEN_OBJECTS",
            instruction_text="Walk between the bar and chairs to reach the gap",
            landmark="bar and chairs")
        self.assertTrue(between_gap_lateral_bracket_supported(
            **common,
            detections=[
                {"score": 0.61, "direction": "front_left",
                 "matched_tokens": ["bar"]},
                {"score": 0.58, "direction": "right",
                 "matched_tokens": ["chairs"]},
            ]))
        self.assertFalse(between_gap_lateral_bracket_supported(
            **common,
            detections=[
                {"score": 0.61, "direction": "front",
                 "matched_tokens": ["bar"]},
                {"score": 0.58, "direction": "rear",
                 "matched_tokens": ["chairs"]},
            ]))
        # Explicit pass-through wording defines an endpoint beyond the pair;
        # it is governed by the existing boundary/extent gates instead.
        self.assertTrue(between_gap_lateral_bracket_supported(
            form="BETWEEN_OBJECTS",
            instruction_text="Walk through between the bar and chairs",
            landmark="bar and chairs",
            detections=[]))

    def test_terminal_facing_landmark_needs_explicit_heading_language(self):
        self.assertEqual(terminal_facing_landmark({
            "navigation_instruction": (
                "reach the next doorway, facing a large bed, and stop"),
            "completion_cue": "Facing the large bed through the doorway",
        }), "large bed")
        self.assertIsNone(terminal_facing_landmark({
            "navigation_instruction": "stop in front of the toilet",
            "completion_cue": "at a safe offset",
        }))
        self.assertIsNone(terminal_facing_landmark({
            "navigation_instruction": (
                "Turn left and walk towards the far side of the loft"),
            "completion_cue": "facing the far side and moving towards it",
        }))

    def test_explicit_turn_selection_preserves_panorama_rotation_evidence(self):
        left = {
            "direction_gate": {
                "active": True, "three_view_gate": True, "sector": "left"},
            "view_index": 1,
            "refined_relative_yaw_deg": 45,
        }
        self.assertTrue(explicit_turn_selection_supported("TURN_LEFT", left))
        self.assertTrue(explicit_turn_selection_supported(
            "TURN_LEFT", {**left, "refinement_accepted": True,
                          "refined_relative_yaw_deg": 67.5}))
        self.assertFalse(explicit_turn_selection_supported("TURN_RIGHT", left))
        self.assertFalse(explicit_turn_selection_supported(
            "TURN_LEFT", {**left, "direction_gate": {
                **left["direction_gate"], "three_view_gate": False}}))
        right = {
            "direction_gate": {
                "active": True, "three_view_gate": True, "sector": "right"},
            "view_index": 7,
            "refined_relative_yaw_deg": -45,
        }
        self.assertTrue(explicit_turn_selection_supported("TURN_RIGHT", right))

    def test_towards_route_requires_ordered_following_landmark_witness(self):
        review = {
            "confidence": 0.9,
            "first_following_landmark": "opening to the stairs",
            "first_following_landmark_view_index": 1,
            "sequence_alignment": "same_view",
        }
        self.assertTrue(following_landmark_route_commitment_supported(review))
        self.assertTrue(following_landmark_route_commitment_supported(
            {**review, "sequence_alignment": "adjacent_view"}))
        self.assertFalse(following_landmark_route_commitment_supported(
            {**review, "first_following_landmark_view_index": -1}))
        self.assertFalse(following_landmark_route_commitment_supported(
            {**review, "confidence": 0.6}))

    def test_landmark_center_refinement_uses_image_side_not_later_route(self):
        self.assertIsNone(landmark_center_refined_yaw_deg(
            -90.0, "centered", "center"))
        self.assertAlmostEqual(landmark_center_refined_yaw_deg(
            -90.0, "off_center", "left"), -67.5)
        self.assertAlmostEqual(landmark_center_refined_yaw_deg(
            -90.0, "off_center", "right"), -112.5)
        with self.assertRaises(ValueError):
            landmark_center_refined_yaw_deg(-90.0, "absent", "absent")

    def test_current_rgb_landmark_alignment_requires_exact_real_witness(self):
        review = {"ambiguous_landmark_route_adjudication": {"active": True}}
        alignment = {
            "status": "aligned_to_current_rgb_landmark",
            "committed_selection_yaw_rad": math.pi,
            "final_yaw_rad": -math.pi,
            "current_landmark_alignment": {
                "view_index": 2,
                "relative_yaw_deg": 90.0,
                "identity_evidence": "same carved wooden fireplace",
                "input_policy": "current_node_rgb_only_depth_prohibited",
            },
        }
        common = dict(
            form="TURN_TO_LANDMARK", status="unknown", point_arrived=True,
            point_selection_review=review,
            post_arrival_alignment=alignment,
            endpoint_evidence={"current_target_state": "partial"},
            reverse_transition_observed=False, stationary_or_blocked=False)
        self.assertTrue(current_rgb_turn_landmark_alignment_supported(**common))
        self.assertFalse(current_rgb_turn_landmark_alignment_supported(
            **dict(common, point_arrived=False)))
        self.assertFalse(current_rgb_turn_landmark_alignment_supported(
            **dict(common, endpoint_evidence={
                "current_target_state": "unsatisfied"})))
        self.assertFalse(current_rgb_turn_landmark_alignment_supported(
            **dict(common, post_arrival_alignment={
                **alignment, "final_yaw_rad": math.pi / 2})))

    def test_v24_coordinated_pass_uses_independent_reference_identity(self):
        class CoordinatedPassBackend:
            def generate_json(self, prompt, images, schema):
                self.prompt = prompt
                return {
                    "status": "completed", "confidence": 0.8,
                    "endpoint_evidence": {
                        "previous_target_state": "unsatisfied",
                        "current_target_state": "satisfied",
                        "current_completion_cue": "satisfied",
                        "reference_previous_sectors": ["front"],
                        "reference_current_sectors": ["rear_left"],
                    },
                    "temporal_evidence": {
                        "semantic_order": "instructed",
                        "boundary_event": "endpoint_inferred",
                        "keyframe_support": True,
                        "same_reference_instance": "not_applicable",
                        "reverse_transition_observed": False,
                    },
                    "motion_evidence": {
                        "instruction_motion_fit": "supports",
                        "stationary_or_blocked": False,
                    },
                    "reason": "both references were checked independently",
                    "visual_evidence": "one vanished and one is rear-left",
                }

        backend = CoordinatedPassBackend()
        views = [np.zeros((24, 32, 3), np.uint8) for _ in range(8)]
        result = NavigationVLMHarness(
            backend, instruction_completion_prompt_version=(
                "v24_multireference_threshold_calibration")
        ).judge_edge_instruction_completion(
            sub_instruction=SubInstruction.from_mapping({
                "sub_instruction_id": 0,
                "navigation_instruction": "Walk past the stairs and bathroom",
                "landmark": "stairs and bathroom",
                "form": "PASS_LANDMARK",
                "semantic_spatial_target": (
                    "floor beyond the stairs and bathroom"),
                "completion_cue": "stairs and bathroom are behind",
            }),
            previous_node_id="node_0000", current_node_id="node_0001",
            previous_position_xyz=[0, 0, 0],
            current_position_xyz=[0, 0, -2],
            previous_base_yaw_rad=0.0, current_base_yaw_rad=0.0,
            previous_six_views=views, current_six_views=views,
            previous_environment_semantics={"views": []},
            current_environment_semantics={"views": []},
            edge_action_history=[{"action": "forward", "moved_m": 2.0}],
            edge_keyframes=[views[0], views[1]])
        self.assertEqual(result["status"], "completed")
        self.assertIn("one noun phrase at a time", backend.prompt)
        self.assertIn("topological context transition", backend.prompt)

    def test_focused_pass_region_adjudication_uses_only_clean_rgb(self):
        class PassRegionBackend:
            def generate_json(self, prompt, images, schema):
                self.prompt = prompt
                self.image_count = len(images)
                return {
                    "pass_boundary_satisfied": True,
                    "current_distinct_downstream_context": True,
                    "old_region_still_encloses_front": False,
                    "keyframe_transition_support": True,
                    "confidence": 0.8,
                    "reason": "entered a distinct foyer",
                    "visual_evidence": "room recedes through a side opening",
                }

        backend = PassRegionBackend()
        views = [np.zeros((24, 32, 3), np.uint8) for _ in range(8)]
        result = NavigationVLMHarness(backend).adjudicate_pass_region_transition(
            SubInstruction.from_mapping({
                "sub_instruction_id": 0,
                "navigation_instruction": "Walk straight past the living room",
                "landmark": "living room", "form": "PASS_LANDMARK",
            }), views, views[:2], views)
        self.assertTrue(result["pass_boundary_satisfied"])
        self.assertEqual(result["input_policy"],
                         "clean_rgb_only_depth_prohibited")
        self.assertEqual(backend.image_count, 3)
        self.assertIn("Do not use detector labels, depth", backend.prompt)

    def test_pass_region_adjudication_does_not_depend_on_primary_flags(self):
        class AmbiguousPrimaryButClearRegionBackend:
            def generate_json(self, prompt, images, schema):
                if "pass_boundary_satisfied" in schema.get(
                        "properties", {}):
                    return {
                        "pass_boundary_satisfied": True,
                        "current_distinct_downstream_context": True,
                        "old_region_still_encloses_front": False,
                        "keyframe_transition_support": True,
                        "confidence": 0.85,
                        "reason": "clean RGB shows the hallway was passed",
                        "visual_evidence": "old region recedes behind",
                    }
                return {
                    "status": "unknown", "confidence": 0.6,
                    "endpoint_evidence": {
                        "previous_target_state": "unsatisfied",
                        "current_target_state": "partial",
                        "current_completion_cue": "partial",
                        "reference_previous_sectors": ["front"],
                        "reference_current_sectors": ["rear", "front"],
                    },
                    "temporal_evidence": {
                        "semantic_order": "ambiguous",
                        "boundary_event": "not_observed",
                        "keyframe_support": False,
                        "same_reference_instance": "ambiguous",
                        "reverse_transition_observed": False,
                    },
                    "motion_evidence": {
                        "instruction_motion_fit": "supports",
                        "stationary_or_blocked": False,
                    },
                    "reason": "primary view is ambiguous",
                    "visual_evidence": "noisy broad detector labels",
                }

        six_views = [np.zeros((24, 32, 3), np.uint8) for _ in range(6)]
        eight_views = [np.zeros((24, 32, 3), np.uint8) for _ in range(8)]
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 0,
            "navigation_instruction": "walk past the bathroom",
            "landmark": "bathroom", "form": "PASS_LANDMARK",
            "completion_cue": "bathroom is behind",
        })
        harness = NavigationVLMHarness(
            AmbiguousPrimaryButClearRegionBackend(),
            instruction_completion_prompt_version=(
                "v24_multireference_threshold_calibration"))
        with tempfile.TemporaryDirectory() as temporary:
            memory = NavigationGraphMemory(temporary, FakeSemanticExtractor())
            memory.add_origin_node(
                [0, 0, 0], 0.0, 0, six_views,
                completion_views=eight_views)
            node, _ = memory.add_navigation_stop_node(
                [0, 0, -1], 0.0, 1, six_views, sub_instruction,
                [{"action": "forward", "turn_deg": 0.0, "moved_m": 1.0}],
                arrival_signal="point_navigation_arrived",
                completion_views=eight_views)
            result = NodeTransitionInstructionCompletionJudge(
                harness, memory, [sub_instruction]).judge(
                    node, 0, eight_views,
                    [eight_views[0], eight_views[1]])
        self.assertEqual(result.status, "completed")
        self.assertTrue(result.validation_overrides[
            "pass_region_rgb_override"])
        self.assertEqual(result.temporal_evidence["semantic_order"],
                         "instructed")

    def test_compound_turn_partial_endpoint_is_not_heading_recovered(self):
        class StrongPartialTurnBackend(StructuredPartialTurnLandmarkBackend):
            def generate_json(self, prompt, images, schema):
                result = super().generate_json(prompt, images, schema)
                result["status"] = "completed"
                result["temporal_evidence"].update({
                    "semantic_order": "instructed",
                    "boundary_event": "not_observed",
                    "same_reference_instance": "yes",
                })
                result["motion_evidence"]["instruction_motion_fit"] = "supports"
                return result

        views = [np.zeros((24, 32, 3), np.uint8) for _ in range(8)]
        common = dict(
            previous_node_id="node_0000", current_node_id="node_0001",
            previous_position_xyz=[0, 0, 0], current_position_xyz=[-1, 0, 0],
            previous_base_yaw_rad=0.0, current_base_yaw_rad=np.pi / 2,
            previous_six_views=views, current_six_views=views,
            previous_environment_semantics={"views": []},
            current_environment_semantics={"views": []},
            edge_action_history=[
                {"action": "turn_left_forward", "turn_deg": 90.0,
                 "moved_m": 1.0}],
            edge_keyframes=[views[0], views[1]])
        compound = SubInstruction.from_mapping({
            "sub_instruction_id": 0,
            "navigation_instruction": (
                "Turn to the left and walk towards the far side of the loft"),
            "landmark": "far side of the loft", "form": "TURN_LEFT",
            "semantic_spatial_target": "far side of the loft",
            "completion_cue": "reach the far side",
        })
        harness = NavigationVLMHarness(
            StrongPartialTurnBackend(),
            instruction_completion_prompt_version="v23_relation_geometry_guard")
        result = harness.judge_edge_instruction_completion(
            sub_instruction=compound, **common)
        self.assertEqual(result["status"], "unknown")
        self.assertTrue(result["endpoint_recovery_evidence"]
                              ["compound_turn_destination"])

        bare = SubInstruction.from_mapping({
            "sub_instruction_id": 0,
            "navigation_instruction": "Turn left.",
            "landmark": "new heading", "form": "TURN_LEFT",
        })
        bare_result = harness.judge_edge_instruction_completion(
            sub_instruction=bare, **common)
        self.assertEqual(bare_result["status"], "completed")

        directional = SubInstruction.from_mapping({
            "sub_instruction_id": 0,
            "navigation_instruction": (
                "Turn to the left and walk towards the far side of the loft"),
            "landmark": "far side of the loft", "form": "TURN_LEFT",
            "semantic_spatial_target": "free floor along the path to the far side",
            "completion_cue": "facing the far side and moving towards it",
        })
        v24_common = {
            **common,
            "current_position_xyz": [-2, 0, 0],
            "edge_action_history": [{
                "action": "turn_left_forward", "turn_deg": 90.0,
                "moved_m": 2.0}],
        }
        v24_result = NavigationVLMHarness(
            StrongPartialTurnBackend(),
            instruction_completion_prompt_version=(
                "v24_multireference_threshold_calibration")
        ).judge_edge_instruction_completion(
            sub_instruction=directional, **v24_common)
        self.assertEqual(v24_result["status"], "unknown")
        self.assertTrue(v24_result["endpoint_recovery_evidence"]
                                  ["directional_towards_commitment"])

    def test_breadcrumb_selector_applies_reverse_route_direction_gate(self):
        views = [np.full((40, 60, 3), 30 + index, np.uint8)
                 for index in range(6)]
        depths = [np.ones((40, 60), np.float32) for _ in range(6)]
        sub_instruction = InstructionDecomposer(FakeHarness()).decompose("exit")[0]
        with tempfile.TemporaryDirectory() as temporary:
            memory = NavigationGraphMemory(temporary, FakeSemanticExtractor())
            memory.add_origin_node([0, 0, 0], 0.0, 0, views)
            node, edge = memory.add_navigation_stop_node(
                [-3, 0, 0], 0.0, 1, views, sub_instruction,
                [{"action": "forward", "moved_m": 2.0,
                  "position_xyz": [0, 0, -2]},
                 {"action": "forward", "moved_m": 3.0,
                  "position_xyz": [-3, 0, -2]},
                 {"action": "forward", "moved_m": 2.0,
                  "position_xyz": [-3, 0, 0]}],
                arrival_signal="point_navigation_arrived")
            selector = NodeBacktrackingPointSelector(
                memory, FakeFloorSegmenter(), mode="hybrid",
                planner_profile="breadcrumb_guard_v1")
            selection = selector.select(
                current_node_id=node.node_id, target_node_id="node_0000",
                current_rgbs=views, current_depths=depths,
                current_position=node.position_xyz, current_yaw=0.0,
                forward_action_history=edge.action_history,
                forward_edge=edge, reference_source_node_id=node.node_id)
            self.assertTrue(selection.record["selected_direction_allowed"])
            self.assertLessEqual(
                abs(np.degrees(selection.candidates[selection.view_index]
                               ["yaw_error_rad"])), 100.0)
            self.assertEqual(
                selection.record["breadcrumb"]["trace_source"],
                "forward_action_pose_trace")

    def test_node_revisit_requires_position_and_visual_match(self):
        views = [np.full((24, 32, 3), index * 15, np.uint8) for index in range(6)]
        sub_instruction = InstructionDecomposer(FakeHarness()).decompose("exit")[0]
        with tempfile.TemporaryDirectory() as temporary:
            memory = NavigationGraphMemory(temporary, FakeSemanticExtractor())
            target = memory.add_origin_node([1, 0, 2], 0.0, 0, views)
            current, _ = memory.add_navigation_stop_node(
                [1.2, 0, 2.1], 0.0, 1, views, sub_instruction, [],
                arrival_signal="point_navigation_arrived",
                edge_kind="node_backtrack_attempt")
            result = NodeRevisitMatcher(
                reach_radius_m=0.75, minimum_visual_similarity=0.75).match(
                    current, target)
            self.assertTrue(result.reached)
            self.assertLess(result.planar_distance_m, 0.75)
            self.assertGreater(result.visual_similarity, 0.99)

    def test_sequence_recovery_allows_one_extra_hop_then_blocks_branch(self):
        sub_instructions = [SubInstruction.from_mapping({
            "sub_instruction_id": index,
            "navigation_instruction": f"instruction {index}",
            "form": "ADVANCE_STRAIGHT",
        }) for index in range(2)]
        state = InstructionSequenceStateMachine(
            sub_instructions, initial_node_id="node_0000")
        wrong = {
            "belongs_to_sequence": False,
            "matched_sub_instruction_id": -1,
            "confidence": 0.8,
        }
        first = state.observe("node_0001", wrong, selected_yaw=0.75)
        self.assertEqual(first.action, "explore_once_more")
        self.assertEqual(state.blocked_yaws(), [])

        second = state.observe("node_0002", wrong, selected_yaw=1.2)
        self.assertEqual(second.action, "backtrack_and_block")
        self.assertEqual(second.backtrack_target_node_id, "node_0001")
        self.assertAlmostEqual(second.direction_to_block_yaw_rad, 1.2)
        # The direction is committed only after the physical backtrack works.
        self.assertEqual(state.blocked_yaws(), [])
        state.on_backtrack(success=True)
        self.assertEqual(len(state.blocked_yaws()), 1)
        self.assertAlmostEqual(state.blocked_yaws()[0], 1.2)

        correct_zero = {
            "belongs_to_sequence": True,
            "matched_sub_instruction_id": 0,
            "confidence": 0.9,
        }
        advance = state.observe("node_0003", correct_zero, selected_yaw=-0.2)
        self.assertEqual(advance.action, "advance_sequence")
        self.assertEqual(state.expected_sub_instruction_id, 1)
        correct_one = dict(correct_zero, matched_sub_instruction_id=1)
        complete = state.observe("node_0004", correct_one, selected_yaw=0.1)
        self.assertEqual(complete.action, "complete")
        self.assertTrue(state.complete)

    def test_sequence_one_hop_lookahead_can_recover_without_backtracking(self):
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 0,
            "navigation_instruction": "exit the room",
            "form": "EXIT_REGION",
        })
        state = InstructionSequenceStateMachine(
            [sub_instruction], initial_node_id="node_0000")
        state.observe("node_0001", {
            "belongs_to_sequence": False,
            "matched_sub_instruction_id": -1,
            "confidence": 0.9,
        }, selected_yaw=1.0)
        directive = state.observe("node_0002", {
            "belongs_to_sequence": True,
            "matched_sub_instruction_id": 0,
            "confidence": 0.9,
        }, selected_yaw=1.1)
        self.assertEqual(directive.action, "complete")
        self.assertTrue(state.complete)
        self.assertEqual(state.recovery_backtracks, 0)
        self.assertEqual(state.blocked_yaws_by_verified_node, {})

    def test_portal_threshold_recovery_preserves_route_and_deepens_retry(self):
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 0,
            "navigation_instruction": "go forward through the doorway",
            "form": "TRAVERSE_PORTAL_REGION",
        })
        state = InstructionSequenceStateMachine(
            [sub_instruction], initial_node_id="node_0000")
        partial = {
            "belongs_to_sequence": False,
            "matched_sub_instruction_id": -1,
            "confidence": 0.8,
            "unknown_disposition": "on_route",
            "active_form": "TRAVERSE_PORTAL_REGION",
            "portal_endpoint_side_veto_applied": True,
        }
        first = state.observe(
            "node_0001", partial, selected_yaw=1.1)
        self.assertEqual(first.action, "continue_current_instruction")
        second = state.observe(
            "node_0002", partial, selected_yaw=1.2)
        self.assertEqual(second.action, "backtrack_and_block")
        state.on_backtrack(success=True, current_node_id="node_0003")
        self.assertEqual(state.blocked_yaws(), [])
        self.assertTrue(state.portal_threshold_retry_after_backtrack)
        self.assertEqual(state.portal_threshold_retry_count, 1)
        self.assertAlmostEqual(state.branch_route_yaw_rad, 1.1)

        # One further endpoint-vetoed retry is allowed.  The next identical
        # UNKNOWN must use ordinary blocking so a portal cannot churn through
        # the global hop budget while repeatedly returning to the same node.
        third = state.observe(
            "node_0004", partial, selected_yaw=1.15)
        self.assertEqual(third.action, "backtrack_and_block")
        state.on_backtrack(success=True, current_node_id="node_0005")
        self.assertEqual(state.portal_threshold_retry_count,
                         MAX_PORTAL_THRESHOLD_ROUTE_RETRIES)
        self.assertEqual(state.blocked_yaws(), [])
        fourth = state.observe(
            "node_0006", partial, selected_yaw=1.18)
        self.assertEqual(fourth.action, "backtrack_and_block")
        state.on_backtrack(success=True, current_node_id="node_0007")
        self.assertFalse(state.portal_threshold_retry_after_backtrack)
        self.assertEqual(len(state.blocked_yaws()), 1)

    def test_vertical_instruction_allows_multiple_supported_partial_hops(self):
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 0,
            "navigation_instruction": "head up the stairs to the top",
            "form": "VERTICAL_UP",
        })
        state = InstructionSequenceStateMachine(
            [sub_instruction], initial_node_id="node_0000")
        partial = {
            "belongs_to_sequence": False,
            "matched_sub_instruction_id": -1,
            "confidence": 0.8,
            "unknown_disposition": "on_route",
            "active_form": "VERTICAL_UP",
        }
        for index in range(1, 5):
            directive = state.observe(
                f"node_{index:04d}", partial, selected_yaw=0.2 * index)
            self.assertEqual(directive.action, "continue_current_instruction")
        self.assertEqual(state.consecutive_on_route_unknowns, 4)
        self.assertEqual(len(state.off_sequence_nodes), 4)

    def test_exit_allows_three_audited_same_portal_progress_hops(self):
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 0,
            "navigation_instruction": "exit through the door on the right",
            "form": "EXIT_REGION",
        })
        state = InstructionSequenceStateMachine(
            [sub_instruction], initial_node_id="node_0000")
        partial = {
            "belongs_to_sequence": False,
            "matched_sub_instruction_id": -1,
            "confidence": 0.6,
            "unknown_disposition": "on_route",
            "active_form": "EXIT_REGION",
        }
        for index in range(1, 4):
            directive = state.observe(
                f"node_{index:04d}", partial, selected_yaw=-1.2)
            self.assertEqual(directive.action,
                             "continue_current_instruction")
        fourth = state.observe(
            "node_0004", partial, selected_yaw=-1.2)
        self.assertEqual(fourth.action, "backtrack_and_block")

    def test_sequence_rejects_correct_id_with_low_vlm_confidence(self):
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 0,
            "navigation_instruction": "exit the room",
            "form": "EXIT_REGION",
        })
        state = InstructionSequenceStateMachine(
            [sub_instruction], "node_0000",
            minimum_classification_confidence=0.5)
        directive = state.observe("node_0001", {
            "belongs_to_sequence": True,
            "matched_sub_instruction_id": 0,
            "confidence": 0.0,
        }, selected_yaw=0.4)
        self.assertEqual(directive.action, "explore_once_more")
        self.assertFalse(state.complete)

    def test_sequence_blocked_direction_is_never_reintroduced_by_vlm_fallback(self):
        views = [np.zeros((32, 48, 3), np.uint8) for _ in range(6)]
        candidates = []
        for index, rgb in enumerate(views):
            mask = np.zeros((32, 48), bool)
            mask[14:27, 8:40] = True
            candidates.append({
                "rgb": rgb, "target_mask": mask,
                "point": np.array([24, 22], np.float32),
                "excluded": index == 0,
                "hard_excluded": index == 0,
                "semantic_detections": [], "small_seg_object_mask": None,
                "ground_detection_records": [],
            })
        stage = SubInstruction.from_mapping({
            "sub_instruction_id": 0,
            "navigation_instruction": "exit the room",
            "landmark": "doorway", "completion_cue": "outside",
            "semantic_spatial_target": "floor beyond the doorway",
            "spatial_relation": "beyond", "visual_arrival_evidence": "outside",
            "forbidden_target": "inside room", "form": "EXIT_REGION",
        }).to_stage_dict()
        harness = NavigationVLMHarness(HeuristicBackend())
        selected_view, _, record = harness.select_ground_target(
            stage, candidates, action_history=[])
        self.assertNotEqual(selected_view, 0)
        self.assertNotIn(0, record["allowed_views"])

    def test_loop_closure_rebases_future_recovery_ancestor_path(self):
        views = [np.full((16, 24, 3), index, np.uint8) for index in range(6)]
        sub_instruction = InstructionDecomposer(FakeHarness()).decompose("exit")[0]
        with tempfile.TemporaryDirectory() as temporary:
            memory = NavigationGraphMemory(temporary, FakeSemanticExtractor())
            origin = memory.add_origin_node([0, 0, 0], 0, 0, views)
            wrong_one, _ = memory.add_navigation_stop_node(
                [0, 0, -1], 0, 1, views, sub_instruction, [], None)
            wrong_two, _ = memory.add_navigation_stop_node(
                [0, 0, -2], 0, 2, views, sub_instruction, [], None)
            revisit, _ = memory.add_navigation_stop_node(
                [0, 0, 0.1], 0, 3, views, sub_instruction, [], None,
                edge_kind="node_backtrack_attempt")
            closure = memory.add_loop_closure_edge(origin.node_id, revisit.node_id)
            self.assertEqual(closure.edge_kind, "node_revisit_loop_closure")
            self.assertEqual(memory.ancestor_path(revisit.node_id, origin.node_id),
                             [revisit.node_id, origin.node_id])
            self.assertNotEqual(wrong_one.node_id, wrong_two.node_id)


if __name__ == "__main__":
    unittest.main()
