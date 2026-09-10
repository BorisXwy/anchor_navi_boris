#!/usr/bin/env python3
"""Small deterministic contract tests for every point-navigation subsystem."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_point_navigation import (  # noqa: E402
    episode_indices, provider_fatal_output, termination_category,
)
from evaluate_r2r_point_selection import (  # noqa: E402
    angular_error, distance_to_polyline_xz, heading_between, pixel_to_world,
    wilson_interval,
)
from habitat_point_navigation import (  # noqa: E402
    ExplorationVideoComposer, VideoFrameSink, visited_area_estimate,
)
from image_goal_policy import predict, transform_images  # noqa: E402
from instruction_decomposer import SubInstruction  # noqa: E402
from instruction_completion_evidence import (  # noqa: E402
    canonical_label_tokens, instruction_tokens, summarize_semantic_transition,
)
from instruction_taxonomy import decompose_by_definition  # noqa: E402
from navigation_graph_memory import CompactVisualEmbedder  # noqa: E402
from point_navigation_executor import (  # noqa: E402
    POINT_NAVIGATION_ARRIVED, STOP_ARRIVAL_REASON, PointNavigationExecutor,
    PointNavigationRequest, crop_goal, dual_crop_tracking_clusters,
    point_cluster, policy_heading, snap_points_to_mask,
    TRACKING_CLUSTER_PROFILES,
)
from point_selection_strategies import STRATEGIES  # noqa: E402
from point_selectors import (  # noqa: E402
    _route_corridor_anchor_options,
    RandomExplorationPointSelector, distance_to_position_history,
    exploration_ground_points, extend_lower_ground_mask, frontier_key,
    local_ray_navmesh_waypoint,
    route_backtrack_exclusion,
    path_novelty_statistics,
    pixel_ground_to_world, pixel_ray_within_absolute_corridor,
    repair_selected_ground_point, select_ground_point,
    targetable_ground_mask,
)
from semantic_detector import (  # noqa: E402
    DinoBoxDetection, DinoSamDetector, DinoSamFloorSegmenter,
    SemanticDetection, extract_detection_queries,
)
from semantic_point_strategy import (  # noqa: E402
    apply_cross_view_semantic_policy, constrain_floor_candidates,
)
from summarize_r2r_point_selection import metrics  # noqa: E402
from track_cluster import make_cluster, read_video, render  # noqa: E402
from vlm_harness import (  # noqa: E402
    NavigationVLMHarness, VLMProviderFatalError,
    between_panorama_endpoint_recovery_supported,
    between_recovery_source_is_forward,
    between_terminal_extent_recovery_allowed,
    circumnavigation_rear_endpoint_supported,
)


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections
        self.calls = []

    def detect(self, rgb, queries, depth=None):
        self.calls.append(list(queries))
        return list(self.detections)


class FatalProviderBackend:
    def __init__(self):
        self.calls = 0

    def generate_json(self, prompt, images, schema):
        self.calls += 1
        raise VLMProviderFatalError("DeepSeek HTTP 402: Insufficient Balance")


class ProviderFailureContractTests(unittest.TestCase):
    def test_provider_fatal_is_logged_once_and_never_retried(self):
        backend = FatalProviderBackend()
        with tempfile.TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / "vlm_calls.json"
            harness = NavigationVLMHarness(
                backend, retries=3, log_path=log_path)
            with self.assertRaises(VLMProviderFatalError):
                harness._call(
                    "contract_probe", "prompt", [],
                    {"type": "object", "properties": {}}, lambda value: value)
            self.assertEqual(backend.calls, 1)
            attempts = json.loads(
                log_path.with_name("vlm_calls_attempts.json").read_text())
            self.assertEqual(len(attempts), 1)
            self.assertEqual(attempts[0]["status"], "provider_fatal")

    def test_batch_provider_fatal_detection_is_specific(self):
        self.assertTrue(provider_fatal_output(
            "VLMProviderFatalError: DeepSeek HTTP 402: Insufficient Balance"))
        self.assertTrue(provider_fatal_output(
            "DeepSeek API key is missing; set DEEPSEEK_API_KEY"))
        self.assertTrue(provider_fatal_output(
            "VLM harness exhausted retries for review: "
            "DeepSeek request failed: <urlopen error [Errno 111] "
            "Connection refused>"))
        self.assertFalse(provider_fatal_output(
            "RuntimeError: VLM harness exhausted retries for point_selection"))


class FakeDinoBoxDetector:
    def __init__(self, proposals):
        self.proposals = proposals
        self.calls = []

    def detect_boxes(self, rgb, queries):
        self.calls.append({"shape": rgb.shape, "queries": list(queries)})
        return list(self.proposals)


class FakeSamBoxSegmenter:
    def __init__(self, masks):
        self.masks = masks
        self.calls = []

    def segment_boxes(self, rgb, boxes):
        self.calls.append({"shape": rgb.shape, "boxes": list(boxes)})
        return list(self.masks)


class CapturePointSelectionBackend:
    def __init__(self):
        self.prompts = []

    def generate_json(self, prompt, images, schema):
        self.prompts.append(prompt)
        return {"view_index": 0, "anchor_index": 0,
                "reason": "forward floor"}


class FirstAllowedPointSelectionBackend:
    def __init__(self):
        self.prompts = []
        self.image_counts = []

    def generate_json(self, prompt, images, schema):
        self.prompts.append(prompt)
        self.image_counts.append(len(images))
        return {
            "view_index": schema["properties"]["view_index"]["enum"][0],
            "anchor_index": 0,
            "reason": "first explicitly allowed sector anchor",
        }


class EightViewRefinementBackend:
    def __init__(self, decision="request_refined", refined_yaw_deg=20.0,
                 target_centering="centered"):
        self.decision = decision
        self.refined_yaw_deg = refined_yaw_deg
        self.target_centering = target_centering
        self.prompts = []
        self.image_counts = []

    def generate_json(self, prompt, images, schema):
        self.prompts.append(prompt)
        self.image_counts.append(len(images))
        if "decision" in schema["properties"]:
            result = {
                "decision": self.decision,
                "view_index": schema["properties"]["view_index"]["enum"][0],
                "refined_relative_yaw_deg": self.refined_yaw_deg,
                "reason": "target route lies between two compass sectors",
            }
            if "alternative_view_index" in schema["properties"]:
                allowed = schema["properties"]["view_index"]["enum"]
                result.update({
                    "alternative_view_index": allowed[min(1, len(allowed) - 1)],
                    "confidence": 0.62,
                })
            if "target_centering" in schema["properties"]:
                allowed = schema["properties"]["view_index"]["enum"]
                result.update({
                    "target_centering": self.target_centering,
                    "confidence": 0.81,
                    "strongest_competing_view_index": allowed[
                        min(1, len(allowed) - 1)],
                    "visible_route_evidence": (
                        "the named portal and outgoing floor are visible"),
                    "competitor_rejection": (
                        "the competing opening lacks the named landmark"),
                    "first_following_landmark": "doorway",
                    "first_following_landmark_view_index": allowed[0],
                    "sequence_alignment": "same_view",
                })
            return result
        return {
            "view_index": schema["properties"]["view_index"]["enum"][0],
            "anchor_index": 0,
            "reason": "central route-consistent ground anchor",
            **({
                "current_landmark_evidence": "named landmark is visible",
                "following_route_evidence": "later route is compatible",
            } if "current_landmark_evidence" in schema["properties"] else {}),
        }


class CircumnavigateBoundaryBackend:
    def generate_json(self, prompt, images, schema):
        if "decision" in schema["properties"]:
            return {
                "decision": "use_existing",
                "view_index": 2,
                "refined_relative_yaw_deg": 90.0,
                "reason": "view 2 shows the obstacle edge and route",
                "target_centering": "centered",
                "confidence": 0.8,
                "strongest_competing_view_index": 1,
                "visible_route_evidence": "adjacent views share the route",
                "competitor_rejection": "view 1 shows less continuation",
                "first_following_landmark": "not visible",
                "first_following_landmark_view_index": -1,
                "sequence_alignment": "not_visible",
            }
        return {
            "view_index": schema["properties"]["view_index"]["enum"][0],
            "anchor_index": 0,
            "reason": "midpoint floor anchor",
        }


class DualCandidateAdjudicationBackend(EightViewRefinementBackend):
    def generate_json(self, prompt, images, schema):
        if "decision" in schema["properties"]:
            return super().generate_json(prompt, images, schema)
        self.prompts.append(prompt)
        self.image_counts.append(len(images))
        allowed = schema["properties"]["view_index"]["enum"]
        return {
            "view_index": allowed[min(1, len(allowed) - 1)],
            "anchor_index": 0,
            "reason": "the competing view has stronger raw RGB identity",
        }


class DisallowedObjectViewBackend(EightViewRefinementBackend):
    """Mimic a provider ignoring the enum for a no-ground object view."""

    def generate_json(self, prompt, images, schema):
        if "decision" not in schema["properties"]:
            return super().generate_json(prompt, images, schema)
        self.prompts.append(prompt)
        self.image_counts.append(len(images))
        return {
            "decision": "use_existing",
            "view_index": 3,
            "refined_relative_yaw_deg": 135.0,
            "reason": "the object itself is strongest in view 3",
            "target_centering": "centered",
            "confidence": 0.9,
            "strongest_competing_view_index": 4,
            "visible_route_evidence": "object is visible across views 3 and 4",
            "competitor_rejection": "view 4 has weaker object evidence",
            "first_following_landmark": "not applicable",
            "first_following_landmark_view_index": -1,
            "sequence_alignment": "not_applicable",
        }


class FakePathfinder:
    def try_step(self, old_position, proposed):
        return np.asarray(proposed, np.float32)

    def find_path(self, shortest):
        start = np.asarray(shortest.requested_start, np.float32)
        end = np.asarray(shortest.requested_end, np.float32)
        shortest.geodesic_distance = float(np.linalg.norm(end - start))
        shortest.points = [start, end]
        return bool(np.isfinite(end).all())

    def get_topdown_view(self, resolution, height):
        return np.ones((80, 80), bool)

    def get_bounds(self):
        return (np.array([-2.0, 0.0, -2.0], np.float32),
                np.array([2.0, 2.0, 2.0], np.float32))


class FakeAgent:
    def __init__(self, position):
        self.state = SimpleNamespace(
            position=np.asarray(position, np.float32), rotation=None)

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state


class FakeSim:
    def __init__(self, rgb, position=(0.0, 0.0, 0.0)):
        self.rgb = np.asarray(rgb)
        self.agent = FakeAgent(position)
        self.pathfinder = FakePathfinder()

    def get_agent(self, index):
        self.assert_agent_index(index)
        return self.agent

    @staticmethod
    def assert_agent_index(index):
        if index != 0:
            raise AssertionError(index)

    def get_sensor_observations(self):
        return {"rgb": self.rgb}


class ArrivingTracker:
    def reset(self, rgb, points):
        self.points = np.asarray(points, np.float32)
        return self.points.copy(), np.ones(len(points), bool)

    def step(self, rgb):
        visible = np.ones(len(self.points), bool)
        visible[len(self.points) // 2:] = False
        return self.points.copy(), visible


class LostNavigationTracker(ArrivingTracker):
    def reset(self, rgb, points):
        self.points = np.asarray(points, np.float32)
        visible = np.ones(len(points), bool)
        visible[:len(points) // 2] = False
        return self.points.copy(), visible


class AllClustersLostAfterStepTracker(ArrivingTracker):
    def step(self, rgb):
        return self.points.copy(), np.zeros(len(self.points), bool)


def fake_predict(model, config, context, goal, name, device, samples, seed):
    del model, config, context, goal, name, device, samples, seed
    return 1.25, np.array([[[1.0, 0.0], [2.0, 0.0]]], np.float32)


class PerceptionContractTest(unittest.TestCase):
    def test_detection_queries_reduce_action_clauses_to_noun_phrases(self):
        fixtures = [
            ({"form": "APPROACH_LANDMARK",
              "navigation_instruction": "Walk towards the sink",
              "source_clause": "Walk towards the sink",
              "landmark": "Walk towards the sink"}, "sink"),
            ({"form": "ENTER_REGION",
              "navigation_instruction": "Go straight into the atrium",
              "source_clause": "Go straight into the atrium",
              "landmark": "Go straight into the atrium"}, "atrium"),
            ({"form": "CIRCUMNAVIGATE",
              "navigation_instruction": "Go around the table and to the right",
              "source_clause": "Go around the table and to the right",
              "landmark": "Go around the table and to the right"}, "table"),
            ({"form": "PASS_LANDMARK",
              "secondary_forms": ["TRAVERSE_PORTAL_REGION"],
              "navigation_instruction": "Walk past the sink through the doorway",
              "source_clause": "Walk past the sink through the doorway",
              "landmark": "sink"}, "doorway"),
        ]
        for stage, expected in fixtures:
            queries = extract_detection_queries(stage)
            self.assertIn(expected, queries)
            self.assertFalse(any(
                token in query for query in queries
                for token in ("walk ", "go ", "turn ")))

    def test_cross_view_semantic_gate_is_versioned_hard_or_soft(self):
        def candidates():
            return [
                {"view_index": 0, "point": np.array([4.0, 5.0]),
                 "strategy_application": {"mode": "floor_only"}},
                {"view_index": 1, "point": np.array([6.0, 7.0]),
                 "strategy_application": {
                     "mode": "floor_through_detected_portal"}},
            ]
        stage = {"form": "EXIT_REGION"}
        hard = candidates()
        apply_cross_view_semantic_policy(
            stage, hard, "hard_detection_gate")
        self.assertIsNone(hard[0]["point"])
        self.assertEqual(
            hard[0]["strategy_application"]["cross_view_candidate_status"],
            "hard_rejected")

        soft = candidates()
        apply_cross_view_semantic_policy(
            stage, soft, "soft_detection_evidence")
        self.assertIsNotNone(soft[0]["point"])
        self.assertEqual(
            soft[0]["strategy_application"]["cross_view_candidate_status"],
            "retained_as_ground_with_missing_detection_evidence")

    def test_dino_sam_floor_union_and_prompt_record(self):
        shape = (60, 80)
        first = np.zeros(shape, bool); first[35:55, 5:35] = True
        second = np.zeros(shape, bool); second[40:58, 45:75] = True
        detections = [
            SemanticDetection("floor", 0.91, [5, 35, 35, 55], first, 2.25),
            SemanticDetection("rug", 0.82, [45, 40, 75, 58], second, 1.75),
        ]
        detector = FakeDetector(detections)
        segmenter = DinoSamFloorSegmenter(detector, close_kernel=1)
        mask, returned = segmenter(np.zeros((*shape, 3), np.uint8),
                                   np.full(shape, 2.0, np.float32))
        self.assertTrue(np.array_equal(mask, first | second))
        self.assertEqual(returned, detections)
        self.assertEqual(detections[0].prompt_record()["median_depth_m"], 2.25)
        self.assertGreater(detections[0].prompt_record()["mask_area_fraction"], 0)

    def test_dino_boxes_are_frozen_before_sam_masking(self):
        shape = (60, 80)
        floor = np.zeros(shape, bool); floor[31:58, 4:76] = True
        rug = np.zeros(shape, bool); rug[40:55, 20:50] = True
        proposals = [
            DinoBoxDetection("floor", 0.91, [4.0, 30.0, 76.0, 59.0]),
            DinoBoxDetection("rug", 0.83, [20.0, 40.0, 50.0, 55.0]),
        ]
        dino = FakeDinoBoxDetector(proposals)
        sam = FakeSamBoxSegmenter([floor, rug])
        detector = DinoSamDetector(
            device="cpu", box_detector=dino, mask_segmenter=sam)
        rgb = np.zeros((*shape, 3), np.uint8)
        detections = detector.detect(rgb, ["floor", "rug"])
        self.assertEqual(dino.calls[0]["queries"], ["floor", "rug"])
        self.assertEqual(
            sam.calls[0]["boxes"], [proposal.box_xyxy for proposal in proposals])
        self.assertEqual([item.label for item in detections], ["floor", "rug"])
        self.assertTrue(np.array_equal(detections[0].mask, floor))
        self.assertTrue(np.array_equal(detections[1].mask, rug))

    def test_detection_queries_preserve_portal_and_landmark(self):
        queries = extract_detection_queries({
            "form": "EXIT_REGION", "source_clause": "Exit the bedroom",
            "landmark": "bedroom doorway",
        })
        self.assertIn("doorway", queries)
        self.assertIn("bedroom doorway", queries)
        stairs = extract_detection_queries({
            "form": "VERTICAL_UP", "source_clause": "Go upstairs",
            "landmark": "stairs",
        })
        self.assertIn("staircase", stairs)

    def test_detection_queries_keep_ambiguous_manel_visual_heads(self):
        queries = extract_detection_queries({
            "form": "TURN_TO_LANDMARK",
            "source_clause": "Turn towards a large wooden manel",
            "landmark": "large wooden manel",
        })
        self.assertIn("mantel", queries)
        self.assertIn("fireplace mantel", queries)
        self.assertIn("large wooden mantel", queries)
        self.assertIn("panel", queries)
        self.assertIn("large wooden panel", queries)
        self.assertNotIn("large wooden manel", queries)

    def test_portal_relation_constrains_candidates_to_floor(self):
        shape = (100, 120)
        floor = np.zeros(shape, bool); floor[45:95, 5:115] = True
        instance = np.zeros(shape, bool); instance[10:60, 40:80] = True
        detection = SemanticDetection(
            "doorway", 0.9, [40, 10, 80, 60], instance, 3.0)
        candidate, audit = constrain_floor_candidates(
            {"form": "EXIT_REGION"}, floor,
            np.full(shape, 5.0, np.float32), [detection])
        self.assertEqual(audit["mode"], "floor_through_detected_portal")
        self.assertGreater(candidate.sum(), 24)
        self.assertFalse(np.any(candidate & ~floor))
        self.assertFalse(np.any(candidate & instance))

    def test_portal_relation_never_uses_destination_object_as_portal(self):
        shape = (100, 120)
        floor = np.zeros(shape, bool); floor[45:95, 5:115] = True
        bed_mask = np.zeros(shape, bool); bed_mask[20:75, 10:55] = True
        door_mask = np.zeros(shape, bool); door_mask[10:65, 75:105] = True
        bed = SemanticDetection("large bed", 0.95, [10, 20, 55, 75],
                                bed_mask, None)
        door = SemanticDetection("next doorway", 0.65, [75, 10, 105, 65],
                                 door_mask, None)
        candidate, audit = constrain_floor_candidates(
            {"form": "SELECT_PORTAL"}, floor, None, [bed, door])
        self.assertEqual(audit["selected_detections"][0]["label"],
                         "next doorway")
        self.assertTrue(candidate[:, 70:].any())

    def test_detection_failure_falls_back_to_safe_floor(self):
        floor = np.ones((40, 50), bool)
        objects = np.zeros_like(floor); objects[20:25, 20:25] = True
        candidate, audit = constrain_floor_candidates(
            {"form": "PASS_LANDMARK"}, floor,
            np.ones_like(floor, np.float32), [], objects)
        self.assertEqual(audit["fallback"], "no_matching_detection")
        self.assertFalse(candidate[22, 22])

    def test_near_full_frame_region_detection_cannot_erase_pass_floor(self):
        floor = np.zeros((60, 80), bool)
        floor[32:58, 8:72] = True
        region = np.ones_like(floor)
        bathroom = SemanticDetection(
            "bathroom", 0.46, [0, 0, 80, 60], region, None)
        candidate, audit = constrain_floor_candidates(
            {"form": "PASS_LANDMARK",
             "navigation_instruction": "walk past the bathroom"},
            floor, None, [bathroom], object_mask=region)
        self.assertGreaterEqual(int(candidate.sum()), int(floor.sum()) * 0.9)
        self.assertFalse(audit["object_mask_sanity"][
            "accepted_for_floor_subtraction"])
        self.assertEqual(audit["ignored_region_masks"][0]["label"],
                         "bathroom")


class PointGeometryContractTest(unittest.TestCase):

    def test_local_ray_waypoint_respects_frozen_ray_and_cap(self):
        class DirectPathfinder(FakePathfinder):
            @staticmethod
            def snap_point(point):
                return np.asarray(point, np.float32)

        sim = SimpleNamespace(pathfinder=DirectPathfinder())
        waypoint = local_ray_navmesh_waypoint(
            sim, np.zeros(3, np.float32), 0.0,
            minimum_distance_m=0.35, maximum_distance_m=2.0,
            step_m=0.20)
        self.assertIsNotNone(waypoint)
        self.assertLessEqual(waypoint["geodesic_m"], 2.0 + 1e-6)
        self.assertGreaterEqual(waypoint["geodesic_m"], 1.8)
        self.assertAlmostEqual(
            float(waypoint["position_xyz"][0]), 0.0, places=5)
        self.assertLess(float(waypoint["position_xyz"][2]), -1.8)

    def test_far_same_ray_is_locally_truncated_only_for_local_caps(self):
        class DirectPathfinder(FakePathfinder):
            @staticmethod
            def snap_point(point):
                return np.asarray(point, np.float32)

        sim = SimpleNamespace(pathfinder=DirectPathfinder())
        mask = np.zeros((64, 64), bool)
        mask[36:56, 24:40] = True
        candidate = {
            "target_mask": mask,
            "depth": np.full(mask.shape, 4.0, np.float32),
            "yaw": 0.0,
            "relative_yaw_rad": 0.0,
            "point": np.array([31.5, 48.0], np.float32),
            "excluded": False,
        }
        local = repair_selected_ground_point(
            sim, [candidate], 0, candidate["point"],
            np.zeros(3, np.float32), selection={"allowed_views": [0]},
            max_geodesic_m=2.0, minimum_geodesic_m=0.35)
        self.assertEqual(local[2]["status"],
                         "repaired_to_local_geodesic_cap_boundary")
        self.assertLessEqual(local[2]["selected_geodesic_m"], 2.0 + 1e-6)
        self.assertIn("physical_waypoint_override_xyz", local[2])

        ordinary = repair_selected_ground_point(
            sim, [candidate], 0, candidate["point"],
            np.zeros(3, np.float32), selection={"allowed_views": [0]},
            max_geodesic_m=8.0, minimum_geodesic_m=0.35)
        self.assertEqual(ordinary[2]["status"],
                         "selected_point_reachable")
        self.assertNotIn("physical_waypoint_override_xyz", ordinary[2])

    def test_over_cap_bent_path_is_still_path_ray_inconsistent(self):
        class BentPathfinder(FakePathfinder):
            @staticmethod
            def snap_point(point):
                return np.asarray(point, np.float32)

            @staticmethod
            def find_path(shortest):
                start = np.asarray(shortest.requested_start, np.float32)
                end = np.asarray(shortest.requested_end, np.float32)
                # Every reachable endpoint first requires a 90-degree detour
                # from the selected forward image ray.
                bend = start + np.array([-1.0, 0.0, 0.0], np.float32)
                shortest.geodesic_distance = (
                    1.0 + float(np.linalg.norm(end - bend)))
                shortest.points = [start, bend, end]
                return True

        sim = SimpleNamespace(pathfinder=BentPathfinder())
        mask = np.zeros((64, 64), bool)
        mask[36:56, 24:40] = True
        point = np.array([31.5, 48.0], np.float32)
        candidate = {
            "target_mask": mask,
            "depth": np.full(mask.shape, 10.0, np.float32),
            "yaw": 0.0,
            "relative_yaw_rad": 0.0,
            "point": point,
            "excluded": False,
        }
        repaired = repair_selected_ground_point(
            sim, [candidate], 0, point, np.zeros(3, np.float32),
            selection={"allowed_views": [0]}, max_geodesic_m=6.0,
            minimum_geodesic_m=0.75)
        self.assertEqual(repaired[2]["status"], "path_ray_inconsistent")
        self.assertGreater(
            repaired[2]["repair_search_audit"][0][
                "minimum_reachable_geodesic_m"], 6.0)

    def test_route_corridor_sampler_covers_thin_angular_bypass_edges(self):
        mask = np.zeros((240, 320), bool)
        mask[174:230, 68:84] = True
        mask[158:215, 150:178] = True
        camera_yaw = 3.0874
        route_yaw = 3.0718
        anchors = _route_corridor_anchor_options(
            mask, camera_yaw, route_yaw, math.radians(30.0),
            vertical_samples=16)
        self.assertTrue(any(
            68 <= float(point[0]) <= 84 and 174 <= float(point[1]) < 230
            for point in anchors))
        self.assertTrue(all(pixel_ray_within_absolute_corridor(
            camera_yaw, float(point[0]), 320, route_yaw,
            math.radians(30.0)) for point in anchors))

    def test_postselection_pixel_ray_stays_in_absolute_route_corridor(self):
        camera_yaw = -1.500154659
        route_yaw = -2.146975296
        half_width = math.radians(20.0)
        self.assertTrue(pixel_ray_within_absolute_corridor(
            camera_yaw, 212, 320, route_yaw, half_width))
        self.assertFalse(pixel_ray_within_absolute_corridor(
            camera_yaw, 199, 320, route_yaw, half_width))
        self.assertTrue(pixel_ray_within_absolute_corridor(
            camera_yaw, 199, 320, None, None))

    def test_lower_ground_extension_does_not_invent_upper_mask_pixels(self):
        mask = np.zeros((20, 30), bool)
        mask[14:, 12:18] = True
        extended = extend_lower_ground_mask(mask, pixels=3,
                                            lower_fraction=0.65)
        self.assertTrue(np.array_equal(extended[:13], mask[:13]))
        self.assertTrue(extended[16, 9])
        self.assertTrue(extended[16, 20])
        self.assertFalse(extended[8].any())
    def test_ground_selection_projection_and_frontier_statistics(self):
        mask = np.ones((100, 120), bool)
        targetable = targetable_ground_mask(mask)
        self.assertFalse(targetable[5, 60])
        self.assertFalse(targetable[95, 60])
        point, score = select_ground_point(mask)
        self.assertTrue(targetable[round(point[1]), round(point[0])])
        self.assertGreater(score, 0)

        depth = np.full((100, 120), 2.0, np.float32)
        world, measured = pixel_ground_to_world(
            np.array([59.5, 49.5]), depth, [1.0, 0.0, 2.0], 0.0)
        self.assertAlmostEqual(measured, 2.0)
        self.assertAlmostEqual(float(world[0]), 1.0, places=4)
        self.assertAlmostEqual(float(world[2]), 0.0, places=4)
        depth[50, 60] = np.nan
        invalid, _ = pixel_ground_to_world([60, 50], depth, [0, 0, 0], 0)
        self.assertIsNone(invalid)

        candidates = exploration_ground_points(mask, np.full(mask.shape, 3.0), count=4)
        self.assertGreaterEqual(len(candidates), 2)
        stats = path_novelty_statistics(
            [[0, 0, 0], [2, 0, 0]], [[0, 0, 0]], radius=0.5)
        self.assertGreater(stats["novel_fraction"], 0)
        self.assertEqual(stats["minimum_history_distance_m"], 0)
        self.assertEqual(frontier_key([0.49, 0, -0.49], 0.25), (2, -2))
        self.assertAlmostEqual(distance_to_position_history(
            [1, 0, 0], [[0, 0, 0], [3, 0, 0]]), 1.0)

    def test_crop_is_horizontal_symmetric_and_has_policy_ratio(self):
        rgb = np.zeros((120, 160, 3), np.uint8)
        crop, geometry = crop_goal(
            rgb, np.array([[80.0, 90.0]], np.float32), np.array([True]),
            output_size=(85, 64))
        quad = np.asarray(geometry["quad_xy"])
        self.assertEqual(crop.size, (85, 64))
        self.assertAlmostEqual(quad[0, 1], quad[1, 1], places=5)
        self.assertAlmostEqual(quad[2, 1], quad[3, 1], places=5)
        self.assertAlmostEqual(geometry["top_midpoint_xy"][0],
                               geometry["bottom_midpoint_xy"][0])
        self.assertGreaterEqual(geometry["axis_length_px"], 120 / 3)
        self.assertAlmostEqual(geometry["aspect_ratio"], 85 / 64)

    def test_point_and_dual_clusters_obey_distinct_regions(self):
        mask = np.zeros((120, 160), bool); mask[25:110, 15:145] = True
        points, requested, distances = point_cluster([80, 90], mask)
        self.assertEqual(points.shape, (9, 2))
        self.assertEqual(requested.shape, (9, 2))
        self.assertEqual(len({tuple(point) for point in points}), 9)
        self.assertEqual(len(distances), 9)
        crop, geometry = crop_goal(
            np.zeros((120, 160, 3), np.uint8), points,
            np.ones(9, bool), output_size=(85, 64))
        self.assertIsNotNone(crop)
        navigation, stopping, metadata = dual_crop_tracking_clusters(
            geometry, mask)
        self.assertEqual(navigation.shape, (9, 2))
        self.assertEqual(stopping.shape, (9, 2))
        self.assertTrue(all(mask[round(y), round(x)] for x, y in stopping))
        self.assertLess(float(np.median(navigation[:, 1])),
                        float(np.median(stopping[:, 1])))
        self.assertFalse(metadata["stop_region_ground_fallback"])

        dense_navigation, dense_stopping, dense_metadata = (
            dual_crop_tracking_clusters(
                geometry, mask, profile="dense_stop_v2"))
        self.assertEqual(dense_navigation.shape, (9, 2))
        self.assertEqual(dense_stopping.shape, (45, 2))
        self.assertTrue(all(
            mask[round(y), round(x)] for x, y in dense_stopping))
        self.assertEqual(
            dense_metadata["cluster_profile"]["arrival_visible_fraction"],
            0.04)

    def test_dual_cluster_fallback_never_leaves_ground_mask(self):
        mask = np.zeros((120, 160), bool)
        mask[82:108, 75:86] = True
        anchors, _, _ = point_cluster([80, 96], mask)
        _, geometry = crop_goal(
            np.zeros((120, 160, 3), np.uint8), anchors,
            np.ones(len(anchors), bool), output_size=(85, 64))
        navigation, stopping, metadata = dual_crop_tracking_clusters(
            geometry, mask, profile="dense_stop_geodesic_guard_v9")
        self.assertTrue(all(mask[round(y), round(x)]
                            for x, y in navigation))
        self.assertTrue(all(mask[round(y), round(x)]
                            for x, y in stopping))
        self.assertTrue(metadata["navigation_source_ground_snapped"])
        self.assertTrue(metadata["stop_source_ground_snapped"])

    def test_dual_cluster_preserves_ground_on_high_target_crop_boundary(self):
        # A stair/landing target above the horizontal image axis makes the
        # crop extend downward from the selected point. If the valid ground
        # terminates exactly at that point, it lies only on the crop edge and
        # OpenCV may omit it during inverse rasterisation.
        mask = np.zeros((120, 160), bool)
        mask[10, 10] = True
        # The real stair failure projected the only selectable boundary just
        # below output row 63. This minimal transform reproduces that exact
        # raster condition without depending on OpenCV platform rounding.
        geometry = {
            "output_size_wh": [85, 64],
            "source_to_crop_transform": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 54.0],
                [0.0, 0.0, 1.0],
            ],
        }
        navigation, stopping, metadata = dual_crop_tracking_clusters(
            geometry, mask, profile="dense_stop_motion_recovery_v13")
        self.assertTrue(all(mask[round(y), round(x)]
                            for x, y in navigation))
        self.assertTrue(all(mask[round(y), round(x)]
                            for x, y in stopping))
        self.assertTrue(metadata["boundary_projection_used"])

    def test_snap_rejects_empty_mask_and_policy_heading_uses_waypoint(self):
        with self.assertRaisesRegex(ValueError, "allowed mask is empty"):
            snap_points_to_mask([[1, 1]], np.zeros((4, 4), bool))
        heading = policy_heading(np.array([[[1, 0], [1, 1]]], np.float32))
        self.assertAlmostEqual(heading, math.pi / 4, places=5)


class ExecutorContractTest(unittest.TestCase):
    @staticmethod
    def request(rgb):
        mask = np.zeros(rgb.shape[:2], bool); mask[25:110, 15:145] = True
        return PointNavigationRequest(
            rgb=rgb, selected_point_xy=np.array([80, 90], np.float32),
            selectable_mask=mask, yaw=0.0,
            position_history=[np.zeros(3, np.float32)],
            instruction="exit room", semantic_target="floor beyond doorway")

    def test_executor_returns_arrival_only_after_stop_cluster_disappears(self):
        rgb = np.zeros((120, 160, 3), np.uint8)
        executor = PointNavigationExecutor(
            FakeSim(rgb), ArrivingTracker(), object(),
            {"image_size": [85, 64], "context_size": 1}, "gnm",
            fake_predict, device="cpu", max_steps=3)
        result = executor.execute(self.request(rgb))
        self.assertTrue(result.arrived)
        self.assertEqual(result.signal, POINT_NAVIGATION_ARRIVED)
        self.assertEqual(result.end_reason, STOP_ARRIVAL_REASON)
        self.assertEqual(len(result.action_history), 1)
        self.assertEqual(len(result.action_history[0]["position_xyz"]), 3)
        self.assertIn("yaw_rad", result.action_history[0])
        self.assertEqual(result.record["terminal_stop_visible_fraction"], 0.0)
        self.assertGreaterEqual(len(result.edge_keyframes), 2)

    def test_navigation_cluster_loss_is_not_arrival(self):
        rgb = np.zeros((120, 160, 3), np.uint8)
        executor = PointNavigationExecutor(
            FakeSim(rgb), LostNavigationTracker(), object(),
            {"image_size": [85, 64], "context_size": 1}, "gnm",
            fake_predict, device="cpu", max_steps=2)
        result = executor.execute(self.request(rgb))
        self.assertFalse(result.arrived)
        self.assertIsNone(result.signal)
        self.assertEqual(result.end_reason, "navigation_cluster_lost")

    def test_unreachable_selected_point_is_rejected_before_tracking(self):
        rgb = np.zeros((120, 160, 3), np.uint8)
        tracker = ArrivingTracker()
        executor = PointNavigationExecutor(
            FakeSim(rgb), tracker, object(),
            {"image_size": [85, 64], "context_size": 1}, "gnm",
            fake_predict, device="cpu", max_steps=3,
            tracking_cluster_profile="dense_stop_confirm_v5")
        request = self.request(rgb)
        request.selected_point_reachable = False
        result = executor.execute(request)
        self.assertFalse(result.arrived)
        self.assertEqual(result.end_reason, "selected_point_unreachable")
        self.assertEqual(result.action_history, [])
        self.assertFalse(hasattr(tracker, "points"))

    def test_terminal_loss_of_all_three_clusters_is_consensus_arrival(self):
        rgb = np.zeros((120, 160, 3), np.uint8)
        executor = PointNavigationExecutor(
            FakeSim(rgb), AllClustersLostAfterStepTracker(), object(),
            {"image_size": [85, 64], "context_size": 1}, "gnm",
            fake_predict, device="cpu", max_steps=3,
            tracking_cluster_profile="dense_stop_consensus_v6")
        result = executor.execute(self.request(rgb))
        self.assertTrue(result.arrived)
        self.assertEqual(result.end_reason, STOP_ARRIVAL_REASON)
        self.assertTrue(result.record["terminal_cluster_consensus"])

    def test_motion_guard_rejects_early_cluster_occlusion(self):
        rgb = np.zeros((120, 160, 3), np.uint8)
        executor = PointNavigationExecutor(
            FakeSim(rgb), AllClustersLostAfterStepTracker(), object(),
            {"image_size": [85, 64], "context_size": 1}, "gnm",
            fake_predict, device="cpu", max_steps=3,
            tracking_cluster_profile="dense_stop_motion_guard_v7")
        request = self.request(rgb)
        request.selected_point_depth_m = 10.0
        result = executor.execute(request)
        self.assertFalse(result.arrived)
        self.assertEqual(result.end_reason, "navigation_cluster_lost")

    def test_motion_guard_allows_consensus_after_sufficient_travel(self):
        rgb = np.zeros((120, 160, 3), np.uint8)
        executor = PointNavigationExecutor(
            FakeSim(rgb), AllClustersLostAfterStepTracker(), object(),
            {"image_size": [85, 64], "context_size": 1}, "gnm",
            fake_predict, device="cpu", max_steps=3,
            tracking_cluster_profile="dense_stop_motion_guard_v7")
        request = self.request(rgb)
        request.selected_point_depth_m = 0.3
        result = executor.execute(request)
        self.assertTrue(result.arrived)
        self.assertTrue(result.record["terminal_motion_guard_satisfied"])

    def test_v9_motion_guard_uses_longer_initial_geodesic(self):
        rgb = np.zeros((120, 160, 3), np.uint8)
        executor = PointNavigationExecutor(
            FakeSim(rgb), AllClustersLostAfterStepTracker(), object(),
            {"image_size": [85, 64], "context_size": 1}, "gnm",
            fake_predict, device="cpu", max_steps=3,
            tracking_cluster_profile="dense_stop_geodesic_guard_v9")
        request = self.request(rgb)
        request.selected_point_depth_m = 0.3
        request.selected_point_initial_geodesic_m = 10.0
        result = executor.execute(request)
        self.assertFalse(result.arrived)
        self.assertEqual(result.end_reason, "navigation_cluster_lost")

    def test_v9_recovers_near_geodesic_cluster_loss(self):
        rgb = np.zeros((120, 160, 3), np.uint8)
        executor = PointNavigationExecutor(
            FakeSim(rgb), AllClustersLostAfterStepTracker(), object(),
            {"image_size": [85, 64], "context_size": 1}, "gnm",
            fake_predict, device="cpu", max_steps=3,
            tracking_cluster_profile="dense_stop_geodesic_guard_v9")
        request = self.request(rgb)
        request.selected_point_depth_m = 10.0
        request.selected_point_initial_geodesic_m = 0.3
        result = executor.execute(request)
        self.assertTrue(result.arrived)
        self.assertEqual(
            result.end_reason, "near_geodesic_cluster_loss_arrival")

    def test_near_field_shortcut_can_be_disabled_for_reverse_segment(self):
        rgb = np.zeros((120, 160, 3), np.uint8)
        executor = PointNavigationExecutor(
            FakeSim(rgb), AllClustersLostAfterStepTracker(), object(),
            {"image_size": [85, 64], "context_size": 1}, "gnm",
            fake_predict, device="cpu", max_steps=3,
            tracking_cluster_profile="dense_stop_motion_guard_v7")
        request = self.request(rgb)
        request.selected_point_depth_m = 0.3
        request.allow_initial_near_field_arrival = False
        result = executor.execute(request)
        self.assertTrue(result.arrived)
        self.assertGreater(len(result.action_history), 0)
        self.assertNotEqual(
            result.end_reason, "initial_selected_point_in_near_field")

    def test_segment_travel_budget_stops_without_false_arrival(self):
        rgb = np.zeros((120, 160, 3), np.uint8)
        executor = PointNavigationExecutor(
            FakeSim(rgb), ArrivingTracker(), object(),
            {"image_size": [85, 64], "context_size": 1}, "gnm",
            fake_predict, device="cpu", max_steps=3,
            tracking_cluster_profile="dense_stop_motion_guard_v7")
        request = self.request(rgb)
        request.max_travel_distance_m = 0.1
        result = executor.execute(request)
        self.assertFalse(result.arrived)
        self.assertIsNone(result.signal)
        self.assertEqual(result.end_reason, "travel_budget_reached")
        self.assertGreaterEqual(
            result.record["terminal_cumulative_moved_m"], 0.1)

    def test_endpoint_guard_does_not_treat_far_total_cluster_loss_as_arrival(self):
        rgb = np.zeros((120, 160, 3), np.uint8)
        executor = PointNavigationExecutor(
            FakeSim(rgb), AllClustersLostAfterStepTracker(), object(),
            {"image_size": [85, 64], "context_size": 1}, "gnm",
            fake_predict, device="cpu", max_steps=3,
            tracking_cluster_profile=(
                "dense_stop_motion_recovery_v28_endpoint_loss_guard"))
        request = self.request(rgb)
        request.selected_point_navmesh_xyz = np.array(
            [0.0, 0.0, -2.0], np.float32)
        request.selected_point_initial_geodesic_m = 2.0
        result = executor.execute(request)
        self.assertFalse(result.arrived)
        self.assertEqual(result.end_reason, "max_steps")
        self.assertEqual(len(result.action_history), 3)
        self.assertTrue(result.record["endpoint_loss_guard_blocked_arrival"])
        self.assertTrue(any(
            action["endpoint_guidance"]
            for action in result.action_history[1:]))

    def test_endpoint_guard_blocks_zero_step_nearfield_outside_radius(self):
        rgb = np.zeros((120, 160, 3), np.uint8)
        executor = PointNavigationExecutor(
            FakeSim(rgb), AllClustersLostAfterStepTracker(), object(),
            {"image_size": [85, 64], "context_size": 1}, "gnm",
            fake_predict, device="cpu", max_steps=3,
            tracking_cluster_profile=(
                "dense_stop_motion_recovery_v28_endpoint_loss_guard"))
        request = self.request(rgb)
        request.selected_point_depth_m = 0.3
        request.selected_point_navmesh_xyz = np.array(
            [0.0, 0.0, -0.85], np.float32)
        request.selected_point_initial_geodesic_m = 0.85
        request.allow_initial_near_field_arrival = True
        result = executor.execute(request)
        self.assertTrue(result.arrived)
        self.assertGreater(len(result.action_history), 0)
        self.assertNotEqual(
            result.end_reason, "initial_selected_point_in_near_field")
        self.assertLessEqual(
            result.record["online_endpoint_geodesic_distance_m"], 0.75)

    def test_endpoint_guard_recovers_budget_edge_arrival_with_half_loss(self):
        rgb = np.zeros((120, 160, 3), np.uint8)
        executor = PointNavigationExecutor(
            FakeSim(rgb), ArrivingTracker(), object(),
            {"image_size": [85, 64], "context_size": 1}, "gnm",
            fake_predict, device="cpu", max_steps=3,
            tracking_cluster_profile=(
                "dense_stop_motion_recovery_v28_endpoint_loss_guard"))
        request = self.request(rgb)
        request.selected_point_navmesh_xyz = np.array(
            [0.0, 0.0, -1.4], np.float32)
        request.selected_point_initial_geodesic_m = 1.4
        result = executor.execute(request)
        self.assertTrue(result.arrived)
        self.assertEqual(
            result.end_reason, "selected_point_endpoint_geodesic_arrival")
        self.assertLessEqual(result.record["endpoint_geodesic_distance_m"], 0.75)


class ModelAndMemoryContractTest(unittest.TestCase):
    def test_image_goal_tensor_and_gnm_contract(self):
        images = [Image.new("RGB", (32, 24), color) for color in ("red", "blue")]
        tensor = transform_images(images, (85, 64))
        self.assertEqual(tuple(tensor.shape), (1, 6, 64, 85))

        class FakeGNM:
            def __call__(self, obs, goal):
                self.obs_shape = tuple(obs.shape)
                self.goal_shape = tuple(goal.shape)
                return (torch.tensor([0.75]),
                        torch.tensor([[[1.0, 0.0], [2.0, 0.5]]]))

        model = FakeGNM()
        distance, trajectories = predict(
            model, {"context_size": 1, "image_size": [85, 64]},
            [images[0]], images[1], "gnm", torch.device("cpu"))
        self.assertAlmostEqual(distance, 0.75)
        self.assertEqual(trajectories.shape, (1, 2, 2))
        self.assertEqual(model.obs_shape, (1, 6, 64, 85))
        self.assertEqual(model.goal_shape, (1, 3, 64, 85))

    def test_visual_embedding_is_deterministic_normalized_and_256d(self):
        views = [np.full((32, 48, 3), index * 30, np.uint8) for index in range(6)]
        embedder = CompactVisualEmbedder()
        first = embedder.embed(views)
        second = embedder.embed(views)
        self.assertEqual(first.shape, (256,))
        self.assertTrue(np.allclose(first, second))
        self.assertAlmostEqual(float(np.linalg.norm(first)), 1.0, places=5)
        with self.assertRaisesRegex(ValueError, "requires six views"):
            embedder.embed(views[:5])

    def test_streaming_tracker_cluster_and_video_io(self):
        cluster = make_cluster(20, 15, 3, 3, 4)
        self.assertEqual(cluster.shape, (9, 2))
        self.assertTrue(np.allclose(cluster[4], [20, 15]))
        frames = np.zeros((3, 30, 40, 3), np.uint8)
        frames[:, :, :, 1] = 120
        tracks = np.repeat(cluster[None], 3, axis=0)
        visible = np.ones((3, 9), bool)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "tracks.mp4"
            render(frames, tracks, visible, cluster, output, fps=5)
            decoded, fps = read_video(output, max_frames=2)
            self.assertEqual(decoded.shape, (2, 30, 40, 3))
            self.assertGreater(fps, 0)


class PlanningAndUtilityContractTest(unittest.TestCase):
    def test_leading_turn_follow_keeps_side_tangent_visible(self):
        base = math.radians(50.0)
        self.assertAlmostEqual(route_backtrack_exclusion({
            "form": "FOLLOW_PATH_BOUNDARY",
            "navigation_instruction": "Turn and follow the railing",
        }, base), base)
        self.assertAlmostEqual(route_backtrack_exclusion({
            "form": "FOLLOW_PATH_BOUNDARY",
            "navigation_instruction": "Follow the railing ahead",
        }, base), math.radians(90.0))
        self.assertAlmostEqual(route_backtrack_exclusion({
            "form": "PASS_LANDMARK",
            "navigation_instruction": "Walk past the couch",
        }, base), math.radians(90.0))
        self.assertAlmostEqual(route_backtrack_exclusion({
            "form": "PASS_LANDMARK",
            "navigation_instruction": "Walk past the couch",
            "metadata": {"bare_turn_route_setup": {"active": True}},
        }, base), 0.0)

    def test_taxonomy_defines_typed_spatial_targets_and_strategies(self):
        stages = decompose_by_definition(
            "Exit the bedroom. Turn left. Pass the couch. Stop near the rug.")
        forms = [stage["form"] for stage in stages]
        self.assertEqual(forms,
                         ["EXIT_REGION", "TURN_LEFT", "PASS_LANDMARK", "STOP_WAIT"])
        for stage in stages:
            self.assertTrue(stage["semantic_spatial_target"])
            self.assertTrue(stage["forbidden_target"])
            self.assertIn(stage["form"], STRATEGIES)
            self.assertIn("rank", stage["point_selection_strategy"])

    def test_random_selector_blacklists_failed_frontier(self):
        selector = RandomExplorationPointSelector(
            segmenter=None, video_composer=None, novelty_radius=1.0)
        selection = SimpleNamespace(chosen={
            "selection": {}, "world_point": [2.0, 0.0, 1.0]})
        record = {}
        selector.on_navigation_result(
            selection, SimpleNamespace(arrived=False), [[0, 0, 0]], record)
        self.assertEqual(len(selector.blocked_frontiers), 1)
        self.assertTrue(record["frontier_blacklisted_after_failure"])
        snapshot = selector.snapshot([[0, 0, 0]])
        self.assertEqual(len(snapshot["blocked_frontiers_xyz"]), 1)

    def test_evaluation_geometry_and_aggregation(self):
        self.assertAlmostEqual(heading_between([0, 0, 0], [0, 0, -1]), 0)
        self.assertAlmostEqual(angular_error(math.pi - 0.1, -math.pi + 0.1),
                               math.degrees(0.2), places=5)
        self.assertAlmostEqual(distance_to_polyline_xz(
            [1, 0, 1], [[0, 0, 0], [2, 0, 0]]), 1.0)
        depth = np.full((20, 30), 2.0, np.float32)
        world, measured = pixel_to_world(
            [14.5, 9.5], depth, [0, 0, 0], 0, 30, 20)
        self.assertAlmostEqual(measured, 2.0)
        self.assertAlmostEqual(float(world[2]), -2.0, places=4)
        low, high = wilson_interval(8, 10)
        self.assertLess(low, 0.8)
        self.assertGreater(high, 0.8)
        summary = metrics([
            {"error": None, "distance_to_future_demo_path_m": 0.4,
             "heading_error_to_demo_deg": 20, "backtracking_selection": False},
            {"error": "failed", "distance_to_future_demo_path_m": None,
             "heading_error_to_demo_deg": None, "backtracking_selection": False},
        ])
        self.assertEqual(summary["valid"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["within_0_5m"], 1.0)

    def test_cli_episode_and_failure_categories(self):
        self.assertEqual(episode_indices("2,5", 0, 10), [2, 5])
        self.assertEqual(episode_indices(None, 3, 2), [3, 4])
        with self.assertRaises(ValueError):
            episode_indices(None, 0, 0)
        self.assertEqual(termination_category(
            "vlm_selection_failed: VLM did not return JSON"), "vlm_invalid_json")

    def test_three_panel_video_layout_and_sink(self):
        sim = FakeSim(np.zeros((60, 80, 3), np.uint8))
        composer = ExplorationVideoComposer(
            sim, [0, 0, 0], obs_width=80, obs_height=60)
        frame = composer.compose(
            np.zeros((60, 80, 3), np.uint8),
            [[0, 0, 0], [0.1, 0, 0]], [0.1, 0, 0], 0,
            "Exit the room", 0, "navigation")
        self.assertEqual(frame.shape, (120, 240, 3))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "layout.mp4"
            sink = VideoFrameSink(output, composer.frame_size, fps=5)
            sink.append(frame)
            self.assertEqual(len(sink), 1)
            sink.close()
            self.assertGreater(output.stat().st_size, 0)

    def test_visited_area_increases_with_a_distant_point(self):
        one = visited_area_estimate([[0, 0, 0]], radius=0.5, resolution=0.25)
        two = visited_area_estimate(
            [[0, 0, 0], [2, 0, 0]], radius=0.5, resolution=0.25)
        self.assertGreater(one, 0)
        self.assertGreater(two, one)


class VLMPointPromptVersionContractTest(unittest.TestCase):
    def test_completion_detector_tokens_are_plural_normalized_and_focused(self):
        self.assertEqual(instruction_tokens("bar and chairs"), ["bar", "chair"])
        self.assertEqual(canonical_label_tokens("wooden chairs"), {
            "wooden", "chair"})
        semantics = {"views": [{
            "view_index": 0,
            "detections": [
                {"label": "hallway corridor", "score": 0.9},
                {"label": "chair", "score": 0.6},
            ],
        }]}
        summary = summarize_semantic_transition(
            semantics, semantics, "walk between the bar and chairs")
        labels = [item["label"] for item in summary["current_node"]
                  ["instruction_relevant_detections"]]
        self.assertIn("chair", labels)
        self.assertNotIn("hallway corridor", labels)

    def test_between_recovery_requires_pair_ahead_at_source(self):
        self.assertTrue(between_recovery_source_is_forward(
            ["front", "front_right"]))
        self.assertTrue(between_recovery_source_is_forward(
            ["left", "rear_left"]))
        self.assertFalse(between_recovery_source_is_forward(
            ["rear_left", "rear", "rear_right"]))

    def test_between_end_extent_rejects_partial_corridor_recovery(self):
        self.assertTrue(between_terminal_extent_recovery_allowed(
            "walk between the chairs", "partial", "partial",
            "not_observed"))
        self.assertFalse(between_terminal_extent_recovery_allowed(
            "continue to the end of the carpet between ropes",
            "partial", "partial", "not_observed"))
        self.assertTrue(between_terminal_extent_recovery_allowed(
            "continue to the end of the carpet between ropes",
            "satisfied", "satisfied", "observed"))

    def test_between_panorama_can_recover_neutral_long_edge(self):
        common = dict(
            pair_tokens_seen=2, side_bracket=False, panorama_bracket=True,
            source_pair_ahead=True, terminal_extent_ok=True,
            semantic_order="ambiguous", reverse_observed=False,
            stationary=False, motion_fit="neutral", keyframe_support=False,
            traveled_distance_m=2.0)
        self.assertTrue(between_panorama_endpoint_recovery_supported(**common))
        self.assertFalse(between_panorama_endpoint_recovery_supported(
            **dict(common, reverse_observed=True)))
        self.assertFalse(between_panorama_endpoint_recovery_supported(
            **dict(common, pair_tokens_seen=1)))

    def test_circumnavigation_rear_side_accepts_overlap_but_not_exact_front(self):
        common = dict(
            reference_sectors=["rear", "front_right"],
            detector_sectors=["rear_right"], current_state="partial",
            reverse_observed=False, stationary=False, motion_fit="neutral",
            horizontal_displacement_m=2.1, traveled_distance_m=2.4)
        self.assertTrue(circumnavigation_rear_endpoint_supported(**common))
        self.assertFalse(circumnavigation_rear_endpoint_supported(
            **dict(common, reference_sectors=["rear", "front"])))
        self.assertFalse(circumnavigation_rear_endpoint_supported(
            **dict(common, traveled_distance_m=0.9)))

    @staticmethod
    def _candidates():
        mask = np.zeros((24, 32), bool)
        mask[4:22, 4:28] = True
        offsets = [0, 60, 120, 180, 240, 300]
        return [{
            "view_index": index,
            "relative_yaw_rad": math.radians(offset),
            "rgb": np.zeros((24, 32, 3), np.uint8),
            "target_mask": mask.copy(),
            "point": np.array([16.0, 16.0], np.float32),
            "excluded": False,
            "detection_records": [],
            "ground_detection_records": [],
            "semantic_detections": [],
            "small_seg_object_mask": np.zeros_like(mask),
        } for index, offset in enumerate(offsets)]

    @staticmethod
    def _stage():
        return {
            "navigation_instruction": "walk forward through the doorway",
            "landmark": "doorway", "completion_cue": "past the doorway",
            "semantic_spatial_target": "floor beyond the doorway",
            "spatial_relation": "through", "visual_arrival_evidence": "hallway",
            "forbidden_target": "wall", "form": "TRAVERSE_PORTAL_REGION",
            "point_selection_strategy": {},
        }

    @staticmethod
    def _eight_candidates():
        mask = np.zeros((24, 32), bool)
        mask[4:22, 4:28] = True
        offsets = [0, 45, 90, 135, 180, 225, 270, 315]
        return [{
            "view_index": index,
            "relative_yaw_rad": math.radians(offset),
            "yaw": math.radians(offset),
            "rgb": np.zeros((24, 32, 3), np.uint8),
            "mask": mask.copy(),
            "target_mask": mask.copy(),
            "point": np.array([16.0, 16.0], np.float32),
            "excluded": False,
            "hard_excluded": False,
            "detection_records": [],
            "ground_detection_records": [],
            "semantic_detections": [],
            "small_seg_object_mask": np.zeros_like(mask),
            "strategy_application": {},
        } for index, offset in enumerate(offsets)]

    def test_v1_remains_available_and_v2_adds_orientation_coordinates(self):
        baseline_backend = CapturePointSelectionBackend()
        baseline = NavigationVLMHarness(
            baseline_backend, point_selection_prompt_version="v1_baseline")
        _, _, baseline_record = baseline.select_ground_target(
            self._stage(), self._candidates(), [])
        self.assertNotIn(
            "ORIENTATION_AND_CONTINUITY_GUIDANCE", baseline_backend.prompts[0])
        self.assertEqual(
            baseline_record["point_selection_prompt_version"], "v1_baseline")

        improved_backend = CapturePointSelectionBackend()
        improved = NavigationVLMHarness(
            improved_backend,
            point_selection_prompt_version="v2_orientation_continuity")
        _, _, improved_record = improved.select_ground_target(
            self._stage(), self._candidates(), [])
        prompt = improved_backend.prompts[0]
        self.assertIn("ORIENTATION_AND_CONTINUITY_GUIDANCE", prompt)
        self.assertIn('"0": {"relative_yaw_deg": 0.0, "sector": "forward"', prompt)
        self.assertIn('"1": {"relative_yaw_deg": 60.0, "sector": "front-left"', prompt)
        self.assertIn('"5": {"relative_yaw_deg": -60.0, "sector": "front-right"', prompt)
        self.assertEqual(
            improved_record["point_selection_prompt_version"],
            "v2_orientation_continuity")

        soft_backend = CapturePointSelectionBackend()
        soft = NavigationVLMHarness(
            soft_backend,
            point_selection_prompt_version="v3_orientation_soft_semantic")
        soft.select_ground_target(self._stage(), self._candidates(), [])
        self.assertIn(
            "SOFT_CROSS_VIEW_SEMANTIC_EVIDENCE", soft_backend.prompts[0])
        self.assertEqual(
            soft.point_selection_candidate_policy, "soft_detection_evidence")

    def test_v3_is_the_accepted_default_and_v1_is_reversible(self):
        default = NavigationVLMHarness(CapturePointSelectionBackend())
        self.assertEqual(
            default.point_selection_prompt_version,
            "v3_orientation_soft_semantic")
        self.assertEqual(
            default.point_selection_candidate_policy,
            "soft_detection_evidence")

        rollback = NavigationVLMHarness(
            CapturePointSelectionBackend(),
            point_selection_prompt_version="v1_baseline")
        self.assertEqual(
            rollback.point_selection_candidate_policy,
            "hard_detection_gate")

    def test_v4_hard_gates_explicit_turn_sector_and_adds_identity_checks(self):
        backend = FirstAllowedPointSelectionBackend()
        harness = NavigationVLMHarness(
            backend, point_selection_prompt_version="v4_sector_identity_gates")
        stage = dict(self._stage())
        stage.update({
            "navigation_instruction": "Turn right into the hallway",
            "form": "ENTER_REGION",
        })
        chosen, _, record = harness.select_ground_target(
            stage, self._candidates(), [])
        self.assertIn(chosen, {4, 5})
        self.assertEqual(record["allowed_views"], [4, 5])
        self.assertEqual(record["direction_gate"]["sector"], "right")
        self.assertIn("SECTOR_THEN_IDENTITY_GATES", backend.prompts[0])
        self.assertIn("IMAGE 1 is clean RGB", backend.prompts[0])
        self.assertEqual(backend.image_counts, [2])
        self.assertEqual(
            harness.point_selection_candidate_policy,
            "soft_detection_evidence")

    def test_v35_grounds_qualified_region_to_localized_detector_sector(self):
        candidates = self._eight_candidates()
        candidates[4]["detection_records"] = [{
            "label": "cardboard boxes", "score": 0.61,
            "box_xyxy": [8, 7, 21, 19], "mask_area_fraction": 0.12,
        }]
        stage = dict(self._stage())
        stage.update({
            "navigation_instruction": "Enter the room with cardboard boxes",
            "landmark": "room with cardboard boxes",
            "semantic_spatial_target": (
                "free floor inside the room with cardboard boxes"),
            "form": "ENTER_REGION",
        })
        harness = NavigationVLMHarness(
            EightViewRefinementBackend(decision="use_existing"),
            point_selection_prompt_version=(
                "v35_qualified_region_detector_grounding"))
        _, _, record = harness.select_ground_target(stage, candidates, [])
        self.assertEqual(record["allowed_views"], [3])
        self.assertTrue(record["relation_gate"]["active"])
        self.assertEqual(record["relation_gate"]["allowed_after_gate"],
                         [3, 4, 5])
        self.assertEqual(record["relation_gate"]
                               ["qualified_detection_views"], [4])
        self.assertEqual(record["relation_gate"]
                               ["qualified_region_tokens"], [
                                   "box", "cardboard"])

    def test_v5_separates_clean_sector_selection_from_anchor_selection(self):
        backend = FirstAllowedPointSelectionBackend()
        harness = NavigationVLMHarness(
            backend,
            point_selection_prompt_version="v5_two_stage_sector_anchor")
        chosen, _, record = harness.select_ground_target(
            self._stage(), self._candidates(), [])
        self.assertEqual(chosen, 0)
        self.assertEqual(record["allowed_views"], [0])
        self.assertEqual(record["sector_selection"]["view_index"], 0)
        self.assertEqual(backend.image_counts, [1, 2])
        self.assertIn("GROUND_TARGET_SECTOR_SELECTION", backend.prompts[0])
        self.assertIn("TWO_STAGE_SECTOR_RESULT", backend.prompts[1])

    def test_v6_gates_portal_forms_on_validated_walkable_relation(self):
        backend = FirstAllowedPointSelectionBackend()
        harness = NavigationVLMHarness(
            backend,
            point_selection_prompt_version="v6_validated_relation_two_stage")
        candidates = self._candidates()
        candidates[3]["strategy_application"] = {
            "mode": "floor_through_detected_portal",
            "candidate_pixels": 120,
            "relation_detection_evidence_in_view": True,
        }
        chosen, _, record = harness.select_ground_target(
            self._stage(), candidates, [])
        self.assertEqual(chosen, 3)
        self.assertTrue(record["relation_gate"]["active"])
        self.assertEqual(record["relation_gate"]["allowed_after_gate"], [3])
        self.assertEqual(
            record["sector_selection"]["allowed_views_before_sector_selection"],
            [3])
        self.assertIn("Validated portal-floor relation gate", backend.prompts[0])

    def test_v6_does_not_promote_tiny_or_unvalidated_portal_proposal(self):
        backend = FirstAllowedPointSelectionBackend()
        harness = NavigationVLMHarness(
            backend,
            point_selection_prompt_version="v6_validated_relation_two_stage")
        candidates = self._candidates()
        candidates[3]["strategy_application"] = {
            "mode": "floor_through_detected_portal",
            "candidate_pixels": 12,
            "relation_detection_evidence_in_view": True,
        }
        chosen, _, record = harness.select_ground_target(
            self._stage(), candidates, [])
        self.assertEqual(chosen, 0)
        self.assertFalse(record["relation_gate"]["active"])

    def test_v7_keeps_single_v3_call_but_applies_validated_relation_gate(self):
        backend = FirstAllowedPointSelectionBackend()
        harness = NavigationVLMHarness(
            backend,
            point_selection_prompt_version="v7_single_stage_relation_gates")
        candidates = self._candidates()
        candidates[3]["strategy_application"] = {
            "mode": "floor_through_detected_portal",
            "candidate_pixels": 120,
            "relation_detection_evidence_in_view": True,
        }
        chosen, _, record = harness.select_ground_target(
            self._stage(), candidates, [])
        self.assertEqual(chosen, 3)
        self.assertEqual(backend.image_counts, [1])
        self.assertEqual(len(backend.prompts), 1)
        self.assertIn(
            "Point-selection prompt version: v3_orientation_soft_semantic",
            backend.prompts[0])
        self.assertNotIn("GROUND_TARGET_SECTOR_SELECTION", backend.prompts[0])
        self.assertEqual(
            record["point_selection_prompt_version"],
            "v7_single_stage_relation_gates")

    def test_v7_hard_gates_explicit_turn_without_second_call(self):
        backend = FirstAllowedPointSelectionBackend()
        harness = NavigationVLMHarness(
            backend,
            point_selection_prompt_version="v7_single_stage_relation_gates")
        stage = dict(self._stage())
        stage.update({
            "navigation_instruction": "Turn right into the hallway",
            "form": "ENTER_REGION",
        })
        chosen, _, record = harness.select_ground_target(
            stage, self._candidates(), [])
        self.assertIn(chosen, {4, 5})
        self.assertEqual(record["allowed_views"], [4, 5])
        self.assertEqual(backend.image_counts, [1])

    def test_v8_routes_only_object_relation_forms_through_two_stage(self):
        route_backend = FirstAllowedPointSelectionBackend()
        route_harness = NavigationVLMHarness(
            route_backend,
            point_selection_prompt_version="v8_object_relation_router")
        route_harness.select_ground_target(self._stage(), self._candidates(), [])
        self.assertEqual(route_backend.image_counts, [1])

        object_backend = FirstAllowedPointSelectionBackend()
        object_harness = NavigationVLMHarness(
            object_backend,
            point_selection_prompt_version="v8_object_relation_router")
        object_stage = dict(self._stage())
        object_stage["form"] = "APPROACH_LANDMARK"
        object_harness.select_ground_target(
            object_stage, self._candidates(), [])
        self.assertEqual(object_backend.image_counts, [1, 2])
        self.assertIn("GROUND_TARGET_SECTOR_SELECTION", object_backend.prompts[0])

    def test_v9_uses_one_identity_call_with_validated_relation_gate(self):
        backend = FirstAllowedPointSelectionBackend()
        harness = NavigationVLMHarness(
            backend,
            point_selection_prompt_version=(
                "v9_single_stage_identity_relation_gates"))
        candidates = self._candidates()
        candidates[3]["strategy_application"] = {
            "mode": "floor_through_detected_portal",
            "candidate_pixels": 120,
            "relation_detection_evidence_in_view": True,
        }
        chosen, _, record = harness.select_ground_target(
            self._stage(), candidates, [])
        self.assertEqual(chosen, 3)
        self.assertEqual(backend.image_counts, [2])
        self.assertEqual(len(backend.prompts), 1)
        self.assertTrue(record["relation_gate"]["active"])
        self.assertIn("SECTOR_THEN_IDENTITY_GATES", backend.prompts[0])
        self.assertIn("IMAGE 1 is clean RGB", backend.prompts[0])
        self.assertNotIn("GROUND_TARGET_SECTOR_SELECTION", backend.prompts[0])

    def test_v10_two_stage_is_limited_to_single_landmark_approach(self):
        approach_backend = FirstAllowedPointSelectionBackend()
        approach = NavigationVLMHarness(
            approach_backend,
            point_selection_prompt_version="v10_approach_relation_router")
        approach_stage = dict(self._stage())
        approach_stage["form"] = "APPROACH_LANDMARK"
        approach.select_ground_target(
            approach_stage, self._candidates(), [])
        self.assertEqual(approach_backend.image_counts, [1, 2])

        for form in ("BETWEEN_OBJECTS", "CIRCUMNAVIGATE"):
            backend = FirstAllowedPointSelectionBackend()
            harness = NavigationVLMHarness(
                backend,
                point_selection_prompt_version=(
                    "v10_approach_relation_router"))
            stage = dict(self._stage())
            stage["form"] = form
            harness.select_ground_target(stage, self._candidates(), [])
            self.assertEqual(backend.image_counts, [1])
            self.assertNotIn(
                "GROUND_TARGET_SECTOR_SELECTION", backend.prompts[0])

    def test_v20_first_step_route_guard_reuses_rgb_review_and_is_audited(self):
        backend = EightViewRefinementBackend(decision="use_existing")
        harness = NavigationVLMHarness(
            backend,
            point_selection_prompt_version="v20_first_step_route_guard")
        chosen, _, record = harness.select_ground_target(
            self._stage(), self._eight_candidates(), [])
        self.assertEqual(chosen, 0)
        self.assertEqual(
            record["point_selection_prompt_version"],
            "v20_first_step_route_guard")
        self.assertIn(
            "FIRST-STEP ROUTE GUARD (V20)", "\n".join(backend.prompts))
        self.assertEqual(harness.requested_point_selection_prompt_version,
                         "v20_first_step_route_guard")
        self.assertEqual(harness.point_selection_candidate_policy,
                         "soft_detection_evidence")

    def test_v11_can_request_one_local_refined_rgbd_view(self):
        backend = EightViewRefinementBackend()
        harness = NavigationVLMHarness(
            backend,
            point_selection_prompt_version="v11_eight_view_refinement")
        candidates = self._eight_candidates()
        candidates[0]["detection_records"] = [{
            "label": "doorway", "median_depth_m": 2.75,
            "box_xyxy": [4, 4, 20, 20],
        }]
        stage = dict(self._stage())
        stage["point_selection_strategy"] = {
            "perception": ["RGB", "depth"],
            "rank": "depth then route identity",
        }

        def provider(relative_yaw_rad):
            candidate = dict(candidates[0])
            candidate["relative_yaw_rad"] = relative_yaw_rad
            candidate["yaw"] = relative_yaw_rad
            candidate["is_refined_view"] = True
            return candidate

        chosen, _, record = harness.select_ground_target(
            stage, candidates, [], refinement_provider=provider)
        self.assertEqual(chosen, 8)
        self.assertEqual(len(candidates), 9)
        self.assertEqual(backend.image_counts, [2, 2])
        self.assertIn("EIGHT_VIEW_GROUND_TARGET_REVIEW", backend.prompts[0])
        self.assertIn('"7": {"relative_yaw_deg": -45.0', backend.prompts[0])
        self.assertIn("STRICT_45_DEGREE_GROUND_RAY", backend.prompts[1])
        self.assertNotIn("median_depth_m", "\n".join(backend.prompts))
        self.assertNotIn('"perception": ["RGB", "depth"]', backend.prompts[1])
        self.assertTrue(record["view_refinement"]["refinement_accepted"])
        self.assertEqual(record["allowed_views"], [8])
        self.assertEqual(
            record["selection_input_policy"], "rgb_only_depth_prohibited")

    def test_v11_rejects_nonlocal_refinement_and_uses_fallback(self):
        backend = EightViewRefinementBackend(refined_yaw_deg=80.0)
        harness = NavigationVLMHarness(
            backend,
            point_selection_prompt_version="v11_eight_view_refinement")
        candidates = self._eight_candidates()
        provider_calls = []
        chosen, _, record = harness.select_ground_target(
            self._stage(), candidates, [],
            refinement_provider=lambda yaw: provider_calls.append(yaw))
        self.assertEqual(chosen, 0)
        self.assertEqual(provider_calls, [])
        self.assertFalse(record["view_refinement"]["refinement_accepted"])
        self.assertIn("not a local refinement", record[
            "view_refinement"]["fallback_reason"])

    def test_v11_requires_eight_initial_views(self):
        harness = NavigationVLMHarness(
            EightViewRefinementBackend(),
            point_selection_prompt_version="v11_eight_view_refinement")
        with self.assertRaisesRegex(ValueError, "exactly eight initial views"):
            harness.select_ground_target(
                self._stage(), self._candidates(), [])

    def test_v12_second_pass_can_overrule_primary_with_competitor(self):
        backend = DualCandidateAdjudicationBackend(
            decision="use_existing", refined_yaw_deg=0.0)
        harness = NavigationVLMHarness(
            backend,
            point_selection_prompt_version="v12_eight_view_dual_candidate")
        candidates = self._eight_candidates()
        chosen, _, record = harness.select_ground_target(
            self._stage(), candidates, [])
        self.assertEqual(chosen, 1)
        self.assertEqual(record["allowed_views"], [0, 1])
        self.assertEqual(
            record["view_refinement"]["alternative_view_index"], 1)
        self.assertEqual(backend.image_counts, [2, 2])
        self.assertIn(
            "SECOND_ADJUDICATION", backend.prompts[1])

    def test_v13_uses_native_eight_rgb_inputs_then_one_confirmed_view(self):
        backend = EightViewRefinementBackend(
            decision="use_existing", refined_yaw_deg=0.0,
            target_centering="centered")
        harness = NavigationVLMHarness(
            backend,
            point_selection_prompt_version="v13_eight_view_native_rgb")
        candidates = self._eight_candidates()
        candidates[0]["detection_records"] = [{
            "label": "doorway", "median_depth_m": 1.25,
            "box_xyxy": [4, 4, 20, 20],
        }]
        chosen, _, record = harness.select_ground_target(
            self._stage(), candidates, [])
        self.assertEqual(chosen, 0)
        self.assertEqual(backend.image_counts, [10, 2])
        self.assertIn(
            "IMAGES 1..8 are the eight separate", backend.prompts[0])
        self.assertIn(
            "single confirmed direction", backend.prompts[1])
        self.assertNotIn("median_depth_m", "\n".join(backend.prompts))
        self.assertEqual(
            record["selection_input_policy"], "rgb_only_depth_prohibited")

    def test_v17_maps_no_ground_object_view_to_adjacent_allowed_ground(self):
        backend = DisallowedObjectViewBackend()
        harness = NavigationVLMHarness(
            backend,
            point_selection_prompt_version=(
                "v17_turn_three_view_commit_prior"))
        candidates = self._eight_candidates()
        candidates[3]["point"] = None
        candidates[3]["target_mask"][:] = False
        candidates[3]["detection_records"] = [{
            "label": "plant", "score": 0.9,
            "box_xyxy": [10.0, 5.0, 22.0, 23.0],
            "mask_area_fraction": 0.02,
        }]
        stage = dict(self._stage())
        stage.update({
            "form": "APPROACH_LANDMARK",
            "navigation_instruction": "Go to the plant",
            "landmark": "plant",
            "semantic_spatial_target": "floor near the plant",
        })
        chosen, _, record = harness.select_ground_target(
            stage, candidates, [])
        self.assertEqual(chosen, 4)
        refinement = record["view_refinement"]
        self.assertEqual(refinement["model_view_index"], 3)
        self.assertIn("ground_fallback_adjustment", refinement)
        self.assertIn("never return it", backend.prompts[0])

    def test_v31_future_landmark_without_ground_is_only_tie_break_context(self):
        class FutureLandmarkWithoutGroundBackend:
            def generate_json(self, prompt, images, schema):
                if "decision" in schema["properties"]:
                    allowed = schema["properties"]["view_index"]["enum"]
                    return {
                        "decision": "use_existing",
                        "view_index": allowed[0],
                        "refined_relative_yaw_deg": 0.0,
                        "reason": "the current doorway and its floor are ahead",
                        "target_centering": "centered",
                        "confidence": 0.9,
                        "strongest_competing_view_index": allowed[0],
                        "visible_route_evidence": (
                            "current-stage doorway and floor are aligned"),
                        "competitor_rejection": "no other legal ground view",
                        "first_following_landmark": "lamp",
                        "first_following_landmark_view_index": 4,
                        "sequence_alignment": "not_applicable",
                    }
                return {
                    "view_index": schema["properties"]["view_index"]["enum"][0],
                    "anchor_index": 0,
                    "reason": "current-stage connected floor anchor",
                }

        harness = NavigationVLMHarness(
            FutureLandmarkWithoutGroundBackend(),
            point_selection_prompt_version=(
                "v31_circumnavigate_forward_competitor"))
        candidates = self._eight_candidates()
        for index, candidate in enumerate(candidates):
            if index != 0:
                candidate["point"] = None
                candidate["target_mask"][:] = False
        stage = dict(self._stage())
        stage.update({
            "form": "ENTER_REGION",
            "navigation_instruction": (
                "Enter the room and then walk toward the lamp"),
            "next_sub_instruction_context": {
                "navigation_instruction": "walk toward the lamp",
                "landmark": "lamp",
            },
        })
        chosen, _, record = harness.select_ground_target(
            stage, candidates, [])
        self.assertEqual(chosen, 0)
        self.assertIn(
            "following_landmark_ground_unavailable",
            record["view_refinement"])

    def test_v31_future_clause_cannot_replace_current_object_bearing(self):
        class CurrentObjectBeforeFutureDoorBackend:
            def generate_json(self, prompt, images, schema):
                if "decision" in schema["properties"]:
                    return {
                        "decision": "use_existing", "view_index": 6,
                        "refined_relative_yaw_deg": -90.0,
                        "reason": "the mantel is visible in view 6",
                        "target_centering": "centered", "confidence": 0.9,
                        "strongest_competing_view_index": 5,
                        "visible_route_evidence": (
                            "the current mantel and its floor are in view 6"),
                        "competitor_rejection": "view 5 clips the mantel",
                        "first_following_landmark": "doorway",
                        "first_following_landmark_view_index": 0,
                        "sequence_alignment": "adjacent_view",
                    }
                return {
                    "view_index": schema["properties"]["view_index"]["enum"][0],
                    "anchor_index": 0, "reason": "current object floor",
                }

        harness = NavigationVLMHarness(
            CurrentObjectBeforeFutureDoorBackend(),
            point_selection_prompt_version=(
                "v31_circumnavigate_forward_competitor"))
        stage = dict(self._stage())
        stage.update({
            "form": "TURN_TO_LANDMARK",
            "navigation_instruction": (
                "Turn towards a wooden manel and then go through a doorway"),
            "landmark": "wooden manel",
            "next_sub_instruction_context": {
                "navigation_instruction": "go through a doorway",
                "landmark": "doorway",
            },
        })
        chosen, _, record = harness.select_ground_target(
            stage, self._eight_candidates(), [])
        self.assertEqual(chosen, 6)
        self.assertIn(
            "following_landmark_tiebreak_rejected",
            record["view_refinement"])

    def test_v31_future_landmark_never_replaces_current_portal_direction(self):
        class CurrentPortalBeforeFutureStairsBackend:
            def generate_json(self, prompt, images, schema):
                if "decision" in schema["properties"]:
                    return {
                        "decision": "use_existing", "view_index": 0,
                        "refined_relative_yaw_deg": 0.0,
                        "reason": "the current exit doorway is in view 0",
                        "target_centering": "centered", "confidence": 0.9,
                        "strongest_competing_view_index": 1,
                        "visible_route_evidence": "door frame and floor beyond",
                        "competitor_rejection": "view 1 is an interior wall",
                        "first_following_landmark": "stairs",
                        "first_following_landmark_view_index": 4,
                        "sequence_alignment": "adjacent_view",
                    }
                return {
                    "view_index": schema["properties"]["view_index"]["enum"][0],
                    "anchor_index": 0, "reason": "current portal floor",
                }

        harness = NavigationVLMHarness(
            CurrentPortalBeforeFutureStairsBackend(),
            point_selection_prompt_version=(
                "v31_circumnavigate_forward_competitor"))
        stage = dict(self._stage())
        stage.update({
            "form": "EXIT_REGION",
            "navigation_instruction": (
                "head through the door to outside, then climb the stairs"),
            "landmark": "door to outside",
            "next_sub_instruction_context": {
                "navigation_instruction": "turn left and climb the stairs",
                "landmark": "stairs",
            },
        })
        chosen, _, record = harness.select_ground_target(
            stage, self._eight_candidates(), [])
        self.assertEqual(chosen, 0)
        self.assertEqual(
            record["postselection_repair_max_view_delta_deg"], 30.0)
        self.assertIn(
            "following_landmark_tiebreak_rejected",
            record["view_refinement"])

    def test_v31_bare_turn_uses_nominal_quarter_turn_when_available(self):
        class OvercommittedTurnBackend:
            def generate_json(self, prompt, images, schema):
                if "decision" in schema["properties"]:
                    return {
                        "decision": "use_existing", "view_index": 3,
                        "refined_relative_yaw_deg": 135.0,
                        "reason": "a hallway is visible at 135 degrees",
                        "target_centering": "centered", "confidence": 0.9,
                        "strongest_competing_view_index": 2,
                        "visible_route_evidence": "clear side hallway",
                        "competitor_rejection": "90-degree view is narrow",
                        "first_following_landmark": "not visible",
                        "first_following_landmark_view_index": -1,
                        "sequence_alignment": "not_visible",
                    }
                return {
                    "view_index": schema["properties"]["view_index"]["enum"][0],
                    "anchor_index": 0, "reason": "nominal turn floor",
                }

        harness = NavigationVLMHarness(
            OvercommittedTurnBackend(),
            point_selection_prompt_version=(
                "v31_circumnavigate_forward_competitor"))
        stage = dict(self._stage())
        stage.update({
            "form": "TURN_LEFT", "navigation_instruction": "Turn left",
            "landmark": "",
        })
        chosen, _, record = harness.select_ground_target(
            stage, self._eight_candidates(), [])
        self.assertEqual(chosen, 2)
        self.assertIn(
            "bare_turn_nominal_ray_adjustment",
            record["view_refinement"])
        self.assertEqual(
            record["postselection_repair_max_view_delta_deg"], 0.0)

    def test_v31_bare_turn_route_setup_repairs_across_full_commanded_side(self):
        backend = EightViewRefinementBackend(
            decision="request_refined", refined_yaw_deg=112.5,
            target_centering="uncertain")
        harness = NavigationVLMHarness(
            backend, point_selection_prompt_version=(
                "v31_circumnavigate_forward_competitor"))
        stage = dict(self._stage())
        stage.update({
            "form": "PASS_LANDMARK",
            "navigation_instruction": "pass the couch",
            "landmark": "couch",
            "metadata": {"bare_turn_route_setup": {
                "active": True,
                "orientation_stage": {
                    "form": "TURN_LEFT",
                    "navigation_instruction": "Turn left",
                },
            }},
        })
        _, _, record = harness.select_ground_target(
            stage, self._eight_candidates(), [])
        self.assertEqual(record["direction_gate"]["allowed_after_gate"],
                         [1, 2, 3])
        self.assertEqual(record["postselection_repair_max_view_delta_deg"],
                         90.0)

    def test_v17_never_reopens_non_right_views_when_right_gate_is_empty(self):
        harness = NavigationVLMHarness(
            EightViewRefinementBackend(),
            point_selection_prompt_version=(
                "v17_turn_three_view_commit_prior"))
        candidates = self._eight_candidates()
        for index in (5, 6, 7):
            candidates[index]["point"] = None
            candidates[index]["target_mask"][:] = False
        stage = dict(self._stage())
        stage.update({
            "form": "TURN_RIGHT",
            "navigation_instruction": "turn right",
        })
        with self.assertRaisesRegex(
                RuntimeError, "explicit right direction gate"):
            harness.select_ground_target(stage, candidates, [])

    def test_v18_rejects_refined_view_that_reenters_incoming_direction(self):
        backend = EightViewRefinementBackend(
            decision="request_refined", refined_yaw_deg=20.0)
        harness = NavigationVLMHarness(
            backend,
            point_selection_prompt_version="v18_history_safe_refinement")
        candidates = self._eight_candidates()

        def provider(relative_yaw_rad):
            candidate = dict(candidates[0])
            candidate["relative_yaw_rad"] = relative_yaw_rad
            candidate["yaw"] = relative_yaw_rad
            candidate["is_refined_view"] = True
            candidate["excluded"] = True
            return candidate

        chosen, _, record = harness.select_ground_target(
            self._stage(), candidates, [], refinement_provider=provider)
        self.assertEqual(chosen, 0)
        refinement = record["view_refinement"]
        self.assertTrue(refinement["refinement_attempted"])
        self.assertFalse(refinement["refinement_accepted"])
        self.assertIn("incoming/blocked/direction-gated", refinement[
            "fallback_reason"])
        self.assertIn("GROUND-MASK SANITY", "\n".join(backend.prompts))

    def test_v18_filters_final_pixel_rays_that_cross_incoming_boundary(self):
        backend = EightViewRefinementBackend(
            decision="use_existing", refined_yaw_deg=0.0)
        harness = NavigationVLMHarness(
            backend,
            point_selection_prompt_version="v18_history_safe_refinement")
        candidates = self._eight_candidates()
        for candidate in candidates:
            candidate["incoming_back_relative_yaw_rad"] = math.radians(135)
            candidate["backtrack_exclusion_rad"] = math.radians(100)
            candidate["blocked_relative_yaws_rad"] = []
        _, _, record = harness.select_ground_target(
            self._stage(), candidates, [])
        # View 2 is centered at +90 degrees. Although its camera center can be
        # offered by an upstream caller, every sampled anchor ray falls inside
        # the 70-degree incoming exclusion and must disappear before the VLM.
        self.assertNotIn(
            2, record["view_refinement"]["initial_allowed_views"])

    def test_v31_inherits_final_pixel_history_guard(self):
        backend = EightViewRefinementBackend(
            decision="use_existing", refined_yaw_deg=0.0)
        harness = NavigationVLMHarness(
            backend,
            point_selection_prompt_version=(
                "v31_circumnavigate_forward_competitor"))
        self.assertTrue(harness._history_safe_refinement)
        candidates = self._eight_candidates()
        for candidate in candidates:
            candidate["incoming_back_relative_yaw_rad"] = math.radians(135)
            candidate["backtrack_exclusion_rad"] = math.radians(100)
            candidate["blocked_relative_yaws_rad"] = []
        _, _, record = harness.select_ground_target(
            self._stage(), candidates, [])
        self.assertNotIn(
            2, record["view_refinement"]["initial_allowed_views"])

    def test_v31_resamples_full_strict_mask_after_compact_anchors_are_blocked(self):
        harness = NavigationVLMHarness(
            EightViewRefinementBackend(
                decision="use_existing", refined_yaw_deg=90.0),
            point_selection_prompt_version=(
                "v31_circumnavigate_forward_competitor"))
        candidates = self._eight_candidates()
        for index, candidate in enumerate(candidates):
            candidate["excluded"] = index != 2
            candidate["hard_excluded"] = index != 2
            candidate["incoming_back_relative_yaw_rad"] = math.radians(135)
            candidate["backtrack_exclusion_rad"] = math.radians(70)
            candidate["blocked_relative_yaws_rad"] = []
        chosen, _, _ = harness.select_ground_target(
            self._stage(), candidates, [])
        self.assertEqual(chosen, 2)
        self.assertEqual(
            candidates[2]["history_safe_anchor_recovery"],
            "full_strict_ground_mask_resample")
        for x, y in candidates[2]["ground_anchors"]:
            self.assertTrue(candidates[2]["target_mask"][round(y), round(x)])

    def test_v31_reopens_only_tangent_ground_when_soft_incoming_exhausts_views(self):
        harness = NavigationVLMHarness(
            EightViewRefinementBackend(
                decision="use_existing", refined_yaw_deg=0.0),
            point_selection_prompt_version=(
                "v31_circumnavigate_forward_competitor"))
        candidates = self._eight_candidates()
        for index, candidate in enumerate(candidates):
            candidate["excluded"] = True
            candidate["hard_excluded"] = index != 0
            candidate["incoming_back_relative_yaw_rad"] = 0.0
            candidate["backtrack_exclusion_rad"] = math.radians(50)
            candidate["blocked_relative_yaws_rad"] = []
        chosen, _, record = harness.select_ground_target(
            self._stage(), candidates, [])
        self.assertEqual(chosen, 0)
        self.assertTrue(candidates[0]["soft_incoming_tangent_reopened"])
        self.assertEqual(record["view_refinement"]["initial_allowed_views"], [0])

    def test_v31_reopens_blocked_view_edges_but_not_failed_ray_core(self):
        harness = NavigationVLMHarness(
            EightViewRefinementBackend(decision="use_existing"),
            point_selection_prompt_version=(
                "v31_circumnavigate_forward_competitor"))
        candidates = self._eight_candidates()
        for candidate in candidates:
            candidate["excluded"] = True
            candidate["hard_excluded"] = True
            candidate["incoming_back_relative_yaw_rad"] = None
            candidate["blocked_relative_yaws_rad"] = [0.0]
            candidate["blocked_direction_exclusion_rad"] = math.radians(50)
        stage = dict(self._stage())
        stage["metadata"] = {"recovery_route_corridor": {
            "active": True, "absolute_yaw_rad": math.radians(45),
            "half_width_deg": 75}}
        chosen, _, _ = harness.select_ground_target(stage, candidates, [])
        self.assertTrue(candidates[chosen]["soft_blocked_tangent_reopened"])
        width = candidates[chosen]["target_mask"].shape[1]
        center_x = (width - 1.0) / 2.0
        half_width = width / 2.0
        for x, _ in candidates[chosen]["ground_anchors"]:
            ray = float(candidates[chosen]["relative_yaw_rad"]) - math.atan(
                (float(x) - center_x) / half_width)
            self.assertGreaterEqual(abs(math.degrees(
                (ray + math.pi) % (2 * math.pi) - math.pi)), 20.0 - 1e-4)

    def test_v31_does_not_reopen_blocked_tangents_without_route_evidence(self):
        harness = NavigationVLMHarness(
            EightViewRefinementBackend(decision="use_existing"),
            point_selection_prompt_version=(
                "v31_circumnavigate_forward_competitor"))
        candidates = self._eight_candidates()
        for candidate in candidates:
            candidate["excluded"] = True
            candidate["hard_excluded"] = True
            candidate["incoming_back_relative_yaw_rad"] = None
            candidate["blocked_relative_yaws_rad"] = [0.0]
        with self.assertRaisesRegex(RuntimeError, "No floor-bearing"):
            harness.select_ground_target(self._stage(), candidates, [])

    def test_supported_partial_lookahead_is_forward_route_committed(self):
        harness = NavigationVLMHarness(
            EightViewRefinementBackend(decision="use_existing"),
            point_selection_prompt_version=(
                "v31_circumnavigate_forward_competitor"))
        stage = dict(self._stage())
        stage.update({
            "form": "ADVANCE_STRAIGHT",
            "navigation_instruction": "continue straight along the corridor",
            "metadata": {
                "supported_partial_route_continuation": {"active": True}},
        })
        _, _, record = harness.select_ground_target(
            stage, self._eight_candidates(), action_history=[{
                "action": "forward", "moved_m": 1.0}])
        self.assertEqual(
            record["direction_gate"]["sector"],
            "supported_partial_continuation")
        self.assertEqual(
            record["direction_gate"]["allowed_after_gate"], [0])
        self.assertIsNotNone(record["supported_partial_endpoint_repair"])

    def test_exit_continuation_exposes_clean_same_portal_rgb_reference(self):
        backend = EightViewRefinementBackend(decision="use_existing")
        harness = NavigationVLMHarness(
            backend, point_selection_prompt_version=(
                "v31_circumnavigate_forward_competitor"))
        reference = np.full((24, 32, 3), 137, np.uint8)
        _, _, record = harness.select_ground_target(
            self._stage(), self._eight_candidates(),
            action_history=[{"action": "forward", "moved_m": 1.0}],
            semantic_reference_rgb=reference)
        self.assertIn("SAME_PORTAL_IDENTITY_REFERENCE", backend.prompts[0])
        self.assertEqual(backend.image_counts[0], 11)
        self.assertTrue(record["semantic_reference_image_used"])

    def test_supported_partial_corridor_acquires_local_rgb_ground_seam(self):
        harness = NavigationVLMHarness(
            EightViewRefinementBackend(decision="use_existing"),
            point_selection_prompt_version=(
                "v31_circumnavigate_forward_competitor"))
        candidates = self._eight_candidates()
        # Leave one ordinary floor view at +90 degrees, outside a narrow
        # persisted route corridor centred on zero.  The local provider
        # supplies a strict-ground image on that already-committed bearing.
        for index, candidate in enumerate(candidates):
            if index != 2:
                candidate["point"] = None
                candidate["target_mask"][:] = False
        stage = dict(self._stage())
        stage["metadata"] = {
            "supported_partial_route_continuation": {"active": True},
            "recovery_route_corridor": {
                "active": True, "absolute_yaw_rad": 0.0,
                "half_width_deg": 20.0,
            },
        }
        requested = []

        def provider(relative_yaw_rad):
            requested.append(math.degrees(relative_yaw_rad))
            candidate = self._eight_candidates()[0]
            candidate["relative_yaw_rad"] = float(relative_yaw_rad)
            candidate["yaw"] = float(relative_yaw_rad)
            candidate["is_refined_view"] = True
            return candidate

        chosen, _, record = harness.select_ground_target(
            stage, candidates, [{"action": "forward", "moved_m": 1.0}],
            refinement_provider=provider)
        self.assertEqual(chosen, 8)
        self.assertAlmostEqual(requested[0], 0.0, places=5)
        self.assertTrue(candidates[8]["route_corridor_refinement"]["active"])
        self.assertIn(8, record["view_refinement"]["initial_allowed_views"])

    def test_supported_partial_corridor_allows_one_adjacent_rgb_ground_bend(self):
        harness = NavigationVLMHarness(
            EightViewRefinementBackend(decision="use_existing"),
            point_selection_prompt_version=(
                "v31_circumnavigate_forward_competitor"))
        candidates = self._eight_candidates()
        for candidate in candidates:
            candidate["point"] = None
            candidate["target_mask"][:] = False
        # Only the +90-degree adjacent bend has visible connected floor. A
        # supported partial may reacquire this single local corner, while the
        # exact incoming and blocked cores remain forbidden.
        candidates[2]["target_mask"][8:22, 16:24] = True
        candidates[2]["point"] = np.array([20.0, 16.0], np.float32)
        stage = dict(self._stage())
        stage["metadata"] = {
            "supported_partial_route_continuation": {"active": True},
            "recovery_route_corridor": {
                "active": True, "absolute_yaw_rad": 0.0,
                "half_width_deg": 45.0,
            },
        }
        chosen, _, record = harness.select_ground_target(
            stage, candidates,
            [{"action": "forward", "moved_m": 1.0}])
        self.assertEqual(chosen, 2)
        self.assertTrue(record["route_corridor_bend_fallback"]["active"])

    def test_circumnavigation_corridor_can_disable_wider_bend_fallback(self):
        harness = NavigationVLMHarness(
            EightViewRefinementBackend(decision="use_existing"),
            point_selection_prompt_version=(
                "v31_circumnavigate_forward_competitor"))
        candidates = self._eight_candidates()
        for candidate in candidates:
            candidate["point"] = None
            candidate["target_mask"][:] = False
        candidates[2]["target_mask"][8:22, 16:24] = True
        candidates[2]["point"] = np.array([20.0, 16.0], np.float32)
        stage = dict(self._stage())
        stage.update({
            "form": "CIRCUMNAVIGATE",
            "metadata": {
                "supported_partial_route_continuation": {"active": True},
                "instruction_committed_route_corridor": {
                    "active": True, "absolute_yaw_rad": 0.0,
                    "half_width_deg": 45.0,
                    "allow_bend_fallback": False,
                },
            },
        })
        with self.assertRaisesRegex(RuntimeError, "No strict-ground ray"):
            harness.select_ground_target(
                stage, candidates,
                [{"action": "forward", "moved_m": 1.0}])

    def test_route_corridor_without_supported_partial_stays_hard(self):
        harness = NavigationVLMHarness(
            EightViewRefinementBackend(decision="use_existing"),
            point_selection_prompt_version=(
                "v31_circumnavigate_forward_competitor"))
        candidates = self._eight_candidates()
        for candidate in candidates:
            candidate["point"] = None
            candidate["target_mask"][:] = False
        candidates[2]["target_mask"][8:22, 16:24] = True
        candidates[2]["point"] = np.array([20.0, 16.0], np.float32)
        stage = dict(self._stage())
        stage["metadata"] = {"recovery_route_corridor": {
            "active": True, "absolute_yaw_rad": 0.0,
            "half_width_deg": 45.0,
        }}
        with self.assertRaisesRegex(RuntimeError, r"recovery corridor"):
            harness.select_ground_target(
                stage, candidates,
                [{"action": "forward", "moved_m": 1.0}])

    def test_failed_corridor_probes_do_not_expand_nine_view_schema(self):
        harness = NavigationVLMHarness(
            EightViewRefinementBackend(decision="use_existing"),
            point_selection_prompt_version=(
                "v31_circumnavigate_forward_competitor"))
        candidates = self._eight_candidates()
        for index, candidate in enumerate(candidates):
            if index != 2:
                candidate["point"] = None
                candidate["target_mask"][:] = False
        stage = dict(self._stage())
        stage["metadata"] = {"recovery_route_corridor": {
            "active": True, "absolute_yaw_rad": 0.0,
            "half_width_deg": 45.0,
        }}
        calls = []

        def provider(relative_yaw_rad):
            calls.append(relative_yaw_rad)
            candidate = self._eight_candidates()[0]
            candidate["relative_yaw_rad"] = float(relative_yaw_rad)
            candidate["yaw"] = float(relative_yaw_rad)
            if len(calls) < 3:
                candidate["point"] = None
                candidate["target_mask"][:] = False
            return candidate

        chosen, _, _ = harness.select_ground_target(
            stage, candidates, [{"action": "forward", "moved_m": 1.0}],
            refinement_provider=provider)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(candidates), 9)
        self.assertEqual(chosen, 8)

    def test_instruction_committed_turn_corridor_excludes_opposite_route(self):
        harness = NavigationVLMHarness(
            EightViewRefinementBackend(decision="use_existing"),
            point_selection_prompt_version=(
                "v31_circumnavigate_forward_competitor"))
        candidates = self._eight_candidates()
        stage = dict(self._stage())
        stage["metadata"] = {"instruction_committed_route_corridor": {
            "active": True,
            "absolute_yaw_rad": 0.0,
            "half_width_deg": 45.0,
        }}
        chosen, _, record = harness.select_ground_target(
            stage, candidates, [{"action": "forward", "moved_m": 1.0}])
        self.assertNotEqual(chosen, 4)
        self.assertNotIn(
            4, record["view_refinement"]["initial_allowed_views"])

    def test_review_prompt_carries_ordered_route_disambiguation_context(self):
        backend = EightViewRefinementBackend(decision="use_existing")
        harness = NavigationVLMHarness(
            backend, point_selection_prompt_version=(
                "v31_circumnavigate_forward_competitor"))
        stage = dict(self._stage())
        stage["following_sub_instruction_contexts"] = [
            {"navigation_instruction": "go through the doorway",
             "landmark": "doorway"},
            {"navigation_instruction": "continue down the red carpet",
             "landmark": "red carpet"},
        ]
        harness.select_ground_target(stage, self._eight_candidates(), [])
        review_prompt = backend.prompts[0]
        self.assertIn("Remaining decomposed route context", review_prompt)
        self.assertIn("red carpet", review_prompt)

    def test_panorama_wide_turn_alias_gets_ordered_rgb_adjudication(self):
        backend = EightViewRefinementBackend(decision="use_existing")
        harness = NavigationVLMHarness(
            backend, point_selection_prompt_version=(
                "v31_circumnavigate_forward_competitor"))
        candidates = self._eight_candidates()
        for candidate in candidates[:5]:
            candidate["detection_records"] = [{
                "label": "wooden panel mantel", "score": 0.4,
                "box_xyxy": [0, 0, 10, 10],
                "mask_area_fraction": 0.1,
            }]
        candidates[6]["detection_records"] = [{
            "label": "wooden feature", "score": 0.9,
            "box_xyxy": [8, 2, 24, 22], "mask_area_fraction": 0.2,
        }]
        stage = dict(self._stage())
        stage.update({
            "form": "TURN_TO_LANDMARK",
            "navigation_instruction": "turn toward the wooden feature",
            "landmark": "wooden feature",
            "following_sub_instruction_contexts": [
                {"navigation_instruction": "go through the doorway"},
                {"navigation_instruction": "follow the carpet"},
            ],
        })
        chosen, _, record = harness.select_ground_target(stage, candidates, [])
        adjudication = record["view_refinement"][
            "ambiguous_landmark_route_adjudication"]
        self.assertTrue(adjudication["active"])
        self.assertEqual(adjudication[
            "detector_supported_view_indices"], [0, 1, 2, 3, 4, 6])
        self.assertEqual(chosen, 1)
        self.assertNotIn(
            0, adjudication["explicit_turn_allowed_view_indices"])
        self.assertIn("AMBIGUOUS_LANDMARK_ROUTE_ADJUDICATION",
                      backend.prompts[1])
        self.assertIn("ordered continuation", backend.prompts[1])
        self.assertEqual(
            adjudication["policy"],
            "panorama-wide detector alias is non-discriminative; the current "
            "landmark must be visible and the full ordered clean-RGB route "
            "resolves its instance and bearing")
        self.assertEqual(record["view_refinement"]["decision"], "use_existing")
        self.assertNotIn("executable_reacquisition", adjudication)

    def test_turn_landmark_endpoint_alignment_is_current_rgb_only(self):
        class AlignmentBackend:
            def generate_json(self, prompt, images, schema):
                self.prompt = prompt
                self.image_count = len(images)
                return {
                    "view_index": 7,
                    "reason": "same wooden landmark is front-right",
                    "identity_evidence": "matching panels and fireplace",
                }

        backend = AlignmentBackend()
        harness = NavigationVLMHarness(
            backend, point_selection_prompt_version=(
                "v31_circumnavigate_forward_competitor"))
        result = harness.select_turn_landmark_alignment(
            {
                "navigation_instruction": "turn toward the wooden feature",
                "landmark": "wooden feature",
            },
            [np.zeros((24, 32, 3), np.uint8) for _ in range(8)],
            {"ambiguous_landmark_route_adjudication": {
                "current_landmark_evidence": "wooden object at route edge",
                "following_route_evidence": "doorway then carpet",
            }})
        self.assertEqual(result["view_index"], 7)
        self.assertEqual(result["relative_yaw_deg"], -45.0)
        self.assertEqual(result["input_policy"],
                         "current_node_rgb_only_depth_prohibited")
        self.assertEqual(backend.image_count, 9)
        self.assertIn("orientation-only decision", backend.prompt)
        self.assertIn("must not determine", backend.prompt)
        self.assertNotIn("median_depth", backend.prompt)

    def test_supported_corridor_reacquires_after_final_anchor_exhaustion(self):
        harness = NavigationVLMHarness(
            EightViewRefinementBackend(decision="use_existing"),
            point_selection_prompt_version=(
                "v31_circumnavigate_forward_competitor"))
        candidates = self._eight_candidates()
        # Only view 0 survives the coarse corridor test. Its original floor
        # pixels all lie in the incoming core, so exhaustion happens at the
        # final pixel-ray validation rather than at the earlier seam gate.
        for index, candidate in enumerate(candidates):
            if index != 0:
                candidate["point"] = None
                candidate["target_mask"][:] = False
            candidate["incoming_back_relative_yaw_rad"] = 0.0
            candidate["backtrack_exclusion_rad"] = np.deg2rad(50.0)
            candidate["blocked_relative_yaws_rad"] = []
        candidates[0]["target_mask"][:] = False
        candidates[0]["target_mask"][8:20, 13:19] = True
        stage = dict(self._stage())
        stage["metadata"] = {"recovery_route_corridor": {
            "active": True, "absolute_yaw_rad": 0.0,
            "half_width_deg": 30.0,
        }}

        def provider(relative_yaw_rad):
            candidate = self._eight_candidates()[0]
            candidate["relative_yaw_rad"] = float(relative_yaw_rad)
            candidate["yaw"] = float(relative_yaw_rad)
            candidate["incoming_back_relative_yaw_rad"] = None
            candidate["blocked_relative_yaws_rad"] = []
            candidate["is_refined_view"] = True
            return candidate

        chosen, _, _ = harness.select_ground_target(
            stage, candidates, [{"action": "forward", "moved_m": 1.0}],
            refinement_provider=provider)
        self.assertEqual(chosen, 8)
        self.assertEqual(
            candidates[8]["route_corridor_refinement"]["phase"],
            "final_history_safe_anchor_validation")

    def test_unqualified_stop_relation_excludes_rear_hemisphere(self):
        harness = NavigationVLMHarness(
            EightViewRefinementBackend(decision="use_existing"),
            point_selection_prompt_version=(
                "v31_circumnavigate_forward_competitor"))
        stage = dict(self._stage())
        stage.update({
            "form": "STOP_WAIT",
            "navigation_instruction": "stop near the rug",
            "landmark": "rug",
            "spatial_relation": "near the rug",
        })
        _, _, record = harness.select_ground_target(
            stage, self._eight_candidates(), [])
        self.assertEqual(
            record["direction_gate"]["sector"],
            "forward_terminal_relation")
        self.assertEqual(
            record["direction_gate"]["allowed_after_gate"], [0, 1, 7])

    def test_vertical_partial_does_not_repeat_initial_left_turn(self):
        harness = NavigationVLMHarness(
            EightViewRefinementBackend(decision="use_existing"),
            point_selection_prompt_version=(
                "v31_circumnavigate_forward_competitor"))
        stage = dict(self._stage())
        stage.update({
            "form": "VERTICAL_UP",
            "navigation_instruction": "Turn left and head up the stairs",
            "metadata": {"vertical_continuation": {"active": True}},
        })
        _, _, record = harness.select_ground_target(
            stage, self._eight_candidates(), action_history=[{
                "action": "forward", "moved_m": 1.0}])
        allowed = record["direction_gate"]["allowed_after_gate"]
        self.assertEqual(record["direction_gate"]["sector"], "front_vertical")
        self.assertIn(0, allowed)
        self.assertNotIn(3, allowed)

    def test_v31_explicit_left_turn_reopens_only_soft_incoming_side(self):
        harness = NavigationVLMHarness(
            EightViewRefinementBackend(
                decision="use_existing", refined_yaw_deg=90.0),
            point_selection_prompt_version=(
                "v31_circumnavigate_forward_competitor"))
        candidates = self._eight_candidates()
        for index, candidate in enumerate(candidates):
            candidate["excluded"] = index in {1, 2, 3}
            candidate["incoming_back_relative_yaw_rad"] = math.pi / 2
            candidate["backtrack_exclusion_rad"] = math.radians(70)
            candidate["blocked_relative_yaws_rad"] = []
        stage = dict(self._stage())
        stage.update({
            "form": "TURN_LEFT",
            "navigation_instruction": "Turn left.",
        })
        chosen, _, record = harness.select_ground_target(
            stage, candidates, [])
        self.assertIn(chosen, {1, 2, 3})
        self.assertEqual(record["direction_gate"]["sector"], "left")

    def test_v31_explicit_left_turn_never_reopens_blocked_side(self):
        harness = NavigationVLMHarness(
            EightViewRefinementBackend(
                decision="use_existing", refined_yaw_deg=90.0),
            point_selection_prompt_version=(
                "v31_circumnavigate_forward_competitor"))
        candidates = self._eight_candidates()
        for candidate in candidates:
            candidate["incoming_back_relative_yaw_rad"] = math.pi / 2
            candidate["backtrack_exclusion_rad"] = math.radians(70)
            candidate["blocked_relative_yaws_rad"] = [
                math.radians(value) for value in range(0, 360, 45)]
            candidate["blocked_direction_exclusion_rad"] = math.radians(30)
        stage = dict(self._stage())
        stage.update({
            "form": "TURN_LEFT",
            "navigation_instruction": "Turn left.",
        })
        with self.assertRaisesRegex(
                RuntimeError, "No history-safe ground anchor"):
            harness.select_ground_target(stage, candidates, [])

    def test_v18_explicit_turn_around_overrides_soft_incoming_exclusion(self):
        backend = EightViewRefinementBackend(
            decision="use_existing", refined_yaw_deg=135.0)
        harness = NavigationVLMHarness(
            backend,
            point_selection_prompt_version="v18_history_safe_refinement")
        candidates = self._eight_candidates()
        for index, candidate in enumerate(candidates):
            candidate["excluded"] = index in {3, 4, 5}
            candidate["incoming_back_relative_yaw_rad"] = math.pi
            candidate["backtrack_exclusion_rad"] = math.radians(50)
            candidate["blocked_relative_yaws_rad"] = []
        stage = dict(self._stage())
        stage.update({
            "form": "TURN_AROUND",
            "navigation_instruction": "turn around",
        })
        chosen, _, record = harness.select_ground_target(
            stage, candidates, [])
        self.assertIn(chosen, {3, 4, 5})
        self.assertEqual(record["direction_gate"]["sector"], "rear")

    def test_v13_noncentered_target_forces_local_rgb_refinement(self):
        backend = EightViewRefinementBackend(
            decision="use_existing", refined_yaw_deg=20.0,
            target_centering="left_edge")
        harness = NavigationVLMHarness(
            backend,
            point_selection_prompt_version="v13_eight_view_native_rgb")
        candidates = self._eight_candidates()

        def provider(relative_yaw_rad):
            candidate = dict(candidates[0])
            candidate["relative_yaw_rad"] = relative_yaw_rad
            candidate["yaw"] = relative_yaw_rad
            candidate["is_refined_view"] = True
            return candidate

        chosen, _, record = harness.select_ground_target(
            self._stage(), candidates, [], refinement_provider=provider)
        self.assertEqual(chosen, 8)
        self.assertEqual(backend.image_counts, [10, 2])
        refinement = record["view_refinement"]
        self.assertEqual(refinement["model_decision"], "use_existing")
        self.assertEqual(refinement["decision"], "request_refined")
        self.assertTrue(refinement["refinement_accepted"])

    def test_v14_compact_rgb_evidence_corrects_distant_semantic_sector(self):
        backend = EightViewRefinementBackend(
            decision="use_existing", refined_yaw_deg=0.0,
            target_centering="centered")
        harness = NavigationVLMHarness(
            backend,
            point_selection_prompt_version="v14_rgb_evidence_refinement")
        candidates = self._eight_candidates()
        candidates[4]["detection_records"] = [{
            "label": "atrium", "score": 0.91,
            "box_xyxy": [11, 5, 21, 18],
            "mask_area_fraction": 0.12,
        }]
        stage = dict(self._stage())
        stage.update({
            "form": "ENTER_REGION",
            "navigation_instruction": "enter the atrium",
            "landmark": "atrium",
        })
        chosen, _, record = harness.select_ground_target(
            stage, candidates, [])
        self.assertEqual(chosen, 4)
        refinement = record["view_refinement"]
        self.assertEqual(refinement["model_view_index"], 0)
        self.assertIn("compact RGB detection", refinement[
            "rgb_evidence_adjustment"])

    def test_v31_semantic_route_commitment_beats_distant_compact_detection(self):
        backend = EightViewRefinementBackend(
            decision="use_existing", refined_yaw_deg=0.0,
            target_centering="centered")
        harness = NavigationVLMHarness(
            backend,
            point_selection_prompt_version=(
                "v31_circumnavigate_forward_competitor"))
        candidates = self._eight_candidates()
        candidates[3]["detection_records"] = [{
            "label": "stairs", "score": 0.91,
            "box_xyxy": [11, 5, 21, 18],
            "mask_area_fraction": 0.12,
        }]
        stage = dict(self._stage())
        stage.update({
            "form": "VERTICAL_UP",
            "navigation_instruction": "head up the stairs",
            "landmark": "stairs",
        })
        chosen, _, record = harness.select_ground_target(
            stage, candidates, [])
        self.assertEqual(chosen, 0)
        refinement = record["view_refinement"]
        self.assertNotIn("rgb_evidence_adjustment", refinement)
        self.assertIn(
            "high-confidence centered eight-view route",
            refinement["rgb_evidence_adjustment_suppressed"])

    def test_v14_requests_rgb_refinement_when_landmark_has_no_nearby_ground(self):
        backend = EightViewRefinementBackend(
            decision="use_existing", refined_yaw_deg=-45.0,
            target_centering="centered")
        harness = NavigationVLMHarness(
            backend,
            point_selection_prompt_version="v14_rgb_evidence_refinement")
        candidates = self._eight_candidates()
        for index, candidate in enumerate(candidates):
            if index != 7:
                candidate["point"] = None
        candidates[1]["detection_records"] = [{
            "label": "sink", "score": 0.93,
            "box_xyxy": [18, 9, 24, 16],
            "mask_area_fraction": 0.04,
        }]
        stage = dict(self._stage())
        stage.update({
            "form": "APPROACH_LANDMARK",
            "navigation_instruction": "walk towards the sink",
            "landmark": "sink",
        })

        def provider(relative_yaw_rad):
            candidate = dict(candidates[7])
            candidate["relative_yaw_rad"] = relative_yaw_rad
            candidate["yaw"] = relative_yaw_rad
            candidate["is_refined_view"] = True
            return candidate

        chosen, _, record = harness.select_ground_target(
            stage, candidates, [], refinement_provider=provider)
        self.assertEqual(chosen, 8)
        refinement = record["view_refinement"]
        self.assertTrue(refinement["refinement_attempted"])
        self.assertTrue(refinement["refinement_accepted"])
        self.assertEqual(refinement["refinement_base_view_index"], 1)
        self.assertIn("no ground-bearing", refinement[
            "rgb_evidence_adjustment"])

    def test_v15_preserves_legal_edge_ground_when_center_has_no_support(self):
        backend = EightViewRefinementBackend(
            decision="use_existing", refined_yaw_deg=0.0,
            target_centering="centered")
        harness = NavigationVLMHarness(
            backend,
            point_selection_prompt_version="v15_rgb_center_preferred")
        candidates = self._eight_candidates()
        edge_mask = np.zeros((24, 32), bool)
        edge_mask[5:22, 1:7] = True
        for candidate in candidates:
            candidate["mask"] = edge_mask.copy()
            candidate["target_mask"] = edge_mask.copy()
            candidate["point"] = np.array([4.0, 16.0], np.float32)

        def provider(relative_yaw_rad):
            candidate = dict(candidates[0])
            candidate["relative_yaw_rad"] = relative_yaw_rad
            candidate["yaw"] = relative_yaw_rad
            candidate["is_refined_view"] = True
            return candidate

        chosen, point, record = harness.select_ground_target(
            self._stage(), candidates, [], refinement_provider=provider)
        self.assertEqual(chosen, 8)
        self.assertLess(float(point[0]), 0.30 * edge_mask.shape[1])
        refinement = record["view_refinement"]
        self.assertTrue(refinement["refinement_accepted"])
        self.assertIn("same selected ground mask", refinement[
            "central_anchor_fallback"])

    def test_v15_accepts_exact_thirty_degree_refinement_with_float_roundoff(self):
        backend = EightViewRefinementBackend(
            decision="request_refined", refined_yaw_deg=-105.0,
            target_centering="uncertain")
        harness = NavigationVLMHarness(
            backend,
            point_selection_prompt_version="v15_rgb_center_preferred")
        candidates = self._eight_candidates()
        for index, candidate in enumerate(candidates):
            if index != 5:
                candidate["point"] = None
        stage = dict(self._stage())
        stage.update({
            "form": "PASS_LANDMARK",
            "navigation_instruction": "walk past the stairs",
            "landmark": "stairs",
        })

        def provider(relative_yaw_rad):
            candidate = dict(candidates[5])
            candidate["relative_yaw_rad"] = relative_yaw_rad
            candidate["yaw"] = relative_yaw_rad
            candidate["is_refined_view"] = True
            return candidate

        chosen, _, record = harness.select_ground_target(
            stage, candidates, [], refinement_provider=provider)
        self.assertEqual(chosen, 8)
        refinement = record["view_refinement"]
        self.assertGreater(refinement[
            "refinement_delta_from_fallback_deg"], 30.0)
        self.assertTrue(refinement["refinement_accepted"])

    def test_v16_hard_limits_left_and_right_turns_to_three_side_views(self):
        expectations = {
            "TURN_LEFT": ("turn left", [1, 2, 3]),
            "TURN_RIGHT": ("turn right", [5, 6, 7]),
        }
        for form, (instruction, expected_views) in expectations.items():
            with self.subTest(form=form):
                backend = EightViewRefinementBackend(
                    decision="use_existing", refined_yaw_deg=0.0,
                    target_centering="centered")
                harness = NavigationVLMHarness(
                    backend,
                    point_selection_prompt_version=(
                        "v16_turn_three_view_gate"))
                stage = dict(self._stage())
                stage.update({
                    "form": form,
                    "navigation_instruction": instruction,
                    "landmark": "",
                })
                chosen, _, record = harness.select_ground_target(
                    stage, self._eight_candidates(), [])
                self.assertIn(chosen, expected_views)
                gate = record["direction_gate"]
                self.assertTrue(gate["active"])
                self.assertTrue(gate["three_view_gate"])
                self.assertEqual(gate["allowed_after_gate"], expected_views)
                self.assertIn("three corresponding", gate["reason"])

    def test_v17_bare_turn_prefers_committed_side_view_over_shallow_veer(self):
        backend = EightViewRefinementBackend(
            decision="use_existing", refined_yaw_deg=45.0,
            target_centering="centered")
        harness = NavigationVLMHarness(
            backend,
            point_selection_prompt_version=(
                "v17_turn_three_view_commit_prior"))
        candidates = self._eight_candidates()
        # Reproduce a valid +45/+135 pair when the +90 DINO+SAM view has
        # no ground.  The VLM backend selects the first allowed (+45) view.
        candidates[2]["point"] = None
        stage = dict(self._stage())
        stage.update({
            "form": "TURN_LEFT",
            "navigation_instruction": "Turn left",
            "landmark": "Turn left",
        })
        chosen, _, record = harness.select_ground_target(
            stage, candidates, [])
        self.assertEqual(record["direction_gate"][
            "allowed_after_gate"], [1, 3])
        self.assertEqual(chosen, 3)
        refinement = record["view_refinement"]
        self.assertEqual(refinement["model_view_index"], 1)
        self.assertIn("committed side view", refinement[
            "turn_commit_adjustment"])

    def test_v31_first_circumnavigation_refines_adjacent_side_pair_midpoint(self):
        harness = NavigationVLMHarness(
            CircumnavigateBoundaryBackend(),
            point_selection_prompt_version=(
                "v31_circumnavigate_forward_competitor"))
        candidates = self._eight_candidates()
        stage = dict(self._stage())
        stage.update({
            "form": "CIRCUMNAVIGATE",
            "navigation_instruction": "Walk around the backside of the couches",
            "landmark": "couches",
            "semantic_spatial_target": "free floor behind the couches",
            "spatial_relation": "behind the couches",
        })
        requested = []

        def provider(relative_yaw_rad):
            requested.append(math.degrees(relative_yaw_rad))
            candidate = dict(candidates[1])
            candidate["relative_yaw_rad"] = relative_yaw_rad
            candidate["yaw"] = relative_yaw_rad
            candidate["is_refined_view"] = True
            return candidate

        chosen, _, record = harness.select_ground_target(
            stage, candidates, [], refinement_provider=provider)
        self.assertEqual(chosen, 8)
        self.assertAlmostEqual(requested[0], 67.5)
        self.assertEqual(
            record["view_refinement"]
                  ["circumnavigation_boundary_refinement"]["policy"],
            "same-side adjacent RGB route midpoint")

    def test_all_vlm_point_selection_versions_reject_depth_candidates(self):
        harness = NavigationVLMHarness(
            CapturePointSelectionBackend(),
            point_selection_prompt_version="v3_orientation_soft_semantic")
        candidates = self._candidates()
        candidates[0]["depth"] = np.ones((24, 32), np.float32)
        with self.assertRaisesRegex(ValueError, "must not contain depth"):
            harness.select_ground_target(self._stage(), candidates, [])

    def test_legacy_prompt_versions_scrub_detector_depth_metadata(self):
        backend = CapturePointSelectionBackend()
        harness = NavigationVLMHarness(
            backend,
            point_selection_prompt_version="v3_orientation_soft_semantic")
        candidates = self._candidates()
        candidates[0]["detection_records"] = [{
            "label": "doorway", "median_depth_m": 2.0,
            "box_xyxy": [4, 4, 20, 20],
        }]
        harness.select_ground_target(self._stage(), candidates, [])
        self.assertNotIn("median_depth_m", backend.prompts[0])

    def test_frozen_other_preamble_is_retyped_to_actionable_exit(self):
        sub = SubInstruction.from_mapping({
            "navigation_instruction": (
                "Start in the middle of the room and head towards the door "
                "that leads to the outside."),
            "form": "OTHER",
            "semantic_spatial_target": "free floor just beyond the doorway",
            "spatial_relation": "beyond the door frame",
        })
        self.assertEqual(sub.form, "EXIT_REGION")
        self.assertEqual(
            sub.semantic_spatial_target, "free floor just beyond the doorway")
        self.assertEqual(
            sub.metadata["form_normalization"]["policy"],
            "first_actionable_taxonomy_clause")

    def test_compound_turn_and_stair_uses_vertical_terminal_form(self):
        sub = SubInstruction.from_mapping({
            "navigation_instruction": "Turn left and head up the stairs.",
            "form": "TURN_LEFT",
            "completion_cue": "Reach the top of the stairs",
            "semantic_spatial_target": "free floor at the top of the stairs",
        })
        self.assertEqual(sub.form, "VERTICAL_UP")
        self.assertIn("TURN_LEFT", sub.secondary_forms)
        self.assertEqual(
            sub.metadata["terminal_form_normalization"]["policy"],
            "compound_vertical_completion_cue_has_endpoint_priority")

    def test_next_clause_detector_metadata_is_rgb_only_and_prompted_separately(self):
        backend = CapturePointSelectionBackend()
        harness = NavigationVLMHarness(
            backend,
            point_selection_prompt_version="v3_orientation_soft_semantic")
        candidates = self._candidates()
        candidates[0]["next_context_detection_records"] = [{
            "label": "barstools", "median_depth_m": 3.0,
            "box_xyxy": [4, 4, 20, 20],
        }]
        harness.select_ground_target(self._stage(), candidates, [])
        prompt = backend.prompts[0]
        self.assertIn("Next-clause detector evidence", prompt)
        self.assertIn("barstools", prompt)
        self.assertNotIn("median_depth_m", prompt)

    def test_post_vertical_stage_prompt_reopens_landing_not_down_stairs(self):
        backend = EightViewRefinementBackend(decision="use_existing")
        harness = NavigationVLMHarness(
            backend,
            point_selection_prompt_version=(
                "v31_circumnavigate_forward_competitor"))
        stage = dict(self._stage())
        stage["form"] = "FOLLOW_PATH_BOUNDARY"
        stage["navigation_instruction"] = (
            "Turn and follow the railing into the bedroom area")
        stage["metadata"] = {"post_vertical_stage_transition": {
            "active": True, "source_form": "VERTICAL_UP"}}
        harness.select_ground_target(stage, self._eight_candidates(), [])
        joined = "\n".join(backend.prompts)
        self.assertIn("POST-VERTICAL LANDING ROUTE RULE", joined)
        self.assertIn("Reject any ray whose visible route goes", joined)
        self.assertIn("first choose the level-floor tangent", joined)
        self.assertIn("most salient immediate room view", joined)

    def test_pass_landmark_preserves_reviewed_lateral_route(self):
        class LateralPassBackend(EightViewRefinementBackend):
            def generate_json(self, prompt, images, schema):
                if "decision" in schema["properties"]:
                    allowed = schema["properties"]["view_index"]["enum"]
                    lateral = 6 if 6 in allowed else allowed[-1]
                    return {
                        "decision": "use_existing",
                        "view_index": lateral,
                        "refined_relative_yaw_deg": -90.0,
                        "reason": "lateral view is the continuing corridor",
                        "target_centering": "centered",
                        "confidence": 0.9,
                        "strongest_competing_view_index": allowed[0],
                        "visible_route_evidence": (
                            "the hallway continues beyond both landmarks"),
                        "competitor_rejection": (
                            "the forward floor enters a side room"),
                        "first_following_landmark": "room",
                        "first_following_landmark_view_index": lateral,
                        "sequence_alignment": "same_view",
                    }
                return super().generate_json(prompt, images, schema)

        harness = NavigationVLMHarness(
            LateralPassBackend(),
            point_selection_prompt_version=(
                "v31_circumnavigate_forward_competitor"))
        stage = dict(self._stage())
        stage.update({
            "form": "PASS_LANDMARK",
            "navigation_instruction": "Walk past the stairs and bathroom",
            "landmark": "stairs and bathroom",
        })
        chosen, _, record = harness.select_ground_target(
            stage, self._eight_candidates(), [])
        self.assertEqual(chosen, 6)
        refinement = record["view_refinement"]
        self.assertNotIn("straight_route_geometry_adjustment", refinement)
        self.assertEqual(
            refinement["stage2_final_ray_geometry"]
                      ["requested_relative_yaw_deg"], -90.0)

    def test_endpoint_coast_profile_is_bounded_extension_of_v13(self):
        parent = TRACKING_CLUSTER_PROFILES[
            "dense_stop_motion_recovery_v13"]
        profile = TRACKING_CLUSTER_PROFILES[
            "dense_stop_motion_recovery_v24_endpoint_coast"]
        self.assertEqual(parent["navigation_loss_coast_frames"], 8)
        self.assertEqual(profile["navigation_loss_coast_frames"], 12)
        self.assertEqual(profile["arrival_visible_fraction"],
                         parent["arrival_visible_fraction"])

    def test_unknown_prompt_version_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown point-selection"):
            NavigationVLMHarness(
                CapturePointSelectionBackend(),
                point_selection_prompt_version="unversioned-change")


if __name__ == "__main__":
    unittest.main()
