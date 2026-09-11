import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from verify_round_stage_completions import (
    _geometry_gate,
    _motion_summary,
    _node_view_paths,
    _real_edge,
    _stage_target_edge_keyframes,
    _verify_one,
    system_all_judged_edges,
    system_stage_completion_candidates,
    trajectory_episode_index,
)


def _rgb_only_target(index, status, stage_id=0):
    return {
        "target_index": index,
        "policy_input_contract": "rgb_only_v1",
        "point_target_arrived": True,
        "node_created_after_point_arrival": True,
        "navigation_graph_node_id": f"node_{index + 1:04d}",
        "navigation_graph_edge_id": f"edge_{index:04d}",
        "sub_instruction": {"sub_instruction_id": stage_id},
        "instruction_completion": {
            "status": status,
            "instruction_completed": status == "completed",
            "expected_sub_instruction_id": stage_id,
            "confidence": 0.7 if status == "completed" else 0.6,
        },
    }


class RoundStageVerifierTest(unittest.TestCase):
    def test_episode_index_comes_from_trajectory_not_id_named_directory(self):
        # episode_0007 is episode_id 7 == dataset index 6 in val_unseen.
        self.assertEqual(trajectory_episode_index(
            {"config": {"episode_index": 6}}, Path("/round/shard_0/episode_0007")), 6)
        self.assertEqual(trajectory_episode_index(
            {}, Path("/legacy_round/episode_0006")), 6)

    def test_completion_verifier_prefers_saved_eight_view_panorama(self):
        node = {
            "six_views": [{"image_path": "six/old.jpg"}],
            "metadata": {"instruction_completion_panorama": {
                "views": [
                    {"image_path": f"eight/view_{index}.jpg"}
                    for index in range(8)]}},
        }
        paths = _node_view_paths(Path("/tmp/graph"), node)
        self.assertEqual(len(paths), 8)
        self.assertEqual(paths[0], Path("/tmp/graph/eight/view_0.jpg"))

    def test_extracts_primary_and_chained_real_edge_completions(self):
        primary = {
            "status": "completed", "instruction_completed": True,
            "expected_sub_instruction_id": 0,
        }
        chained = dict(primary, expected_sub_instruction_id=1)
        trajectory = {"targets": [{
            "point_target_arrived": True,
            "navigation_physical_arrival": True,
            "node_created_after_point_arrival": True,
            "navigation_graph_node_id": "node_0001",
            "navigation_graph_edge_id": "edge_0000",
            "instruction_completion": primary,
            "chained_stop_wait_completion": chained,
        }]}
        result = system_stage_completion_candidates(trajectory)
        self.assertEqual(sorted(result), [0, 1])
        self.assertEqual(result[1]["field"], "chained_stop_wait_completion")

    def test_rejects_completion_without_a_real_arrival_edge(self):
        trajectory = {"targets": [{
            "point_target_arrived": True,
            "navigation_physical_arrival": False,
            "node_created_after_point_arrival": True,
            "navigation_graph_node_id": "node_0001",
            "navigation_graph_edge_id": "edge_0000",
            "instruction_completion": {
                "status": "completed", "instruction_completed": True,
                "expected_sub_instruction_id": 0,
            },
        }]}
        self.assertEqual(system_stage_completion_candidates(trajectory), {})

    def test_multihop_stage_preserves_all_arrived_edges_for_verification(self):
        def target(index, status):
            return {
                "target_index": index,
                "point_target_arrived": True,
                "navigation_physical_arrival": True,
                "node_created_after_point_arrival": True,
                "navigation_graph_node_id": f"node_{index + 1:04d}",
                "navigation_graph_edge_id": f"edge_{index:04d}",
                "sub_instruction": {"sub_instruction_id": 2},
                "instruction_completion": {
                    "status": status,
                    "instruction_completed": status == "completed",
                    "expected_sub_instruction_id": 2,
                },
            }
        trajectory = {"targets": [
            target(4, "unknown"), target(5, "unknown"),
            target(6, "completed"),
        ]}
        result = system_stage_completion_candidates(trajectory)
        self.assertEqual(
            [item["target_index"] for item in result[2]["stage_targets"]],
            [4, 5, 6])

    def test_rgb_only_arrival_without_physical_flag_is_a_real_edge(self):
        target = _rgb_only_target(0, "completed")
        self.assertNotIn("navigation_physical_arrival", target)
        self.assertTrue(_real_edge(target))
        result = system_stage_completion_candidates({"targets": [target]})
        self.assertEqual(sorted(result), [0])

    def test_untagged_target_without_physical_flag_is_not_a_real_edge(self):
        target = _rgb_only_target(0, "completed")
        del target["policy_input_contract"]
        self.assertFalse(_real_edge(target))
        self.assertEqual(
            system_stage_completion_candidates({"targets": [target]}), {})

    def test_explicit_physical_flag_stays_authoritative_for_rgb_only(self):
        target = dict(_rgb_only_target(0, "completed"),
                      navigation_physical_arrival=False)
        self.assertFalse(_real_edge(target))

    def test_all_judged_edges_keeps_unknown_and_one_record_per_target(self):
        trajectory = {"targets": [
            _rgb_only_target(0, "unknown"),
            _rgb_only_target(1, "completed"),
            dict(_rgb_only_target(2, "unknown"), point_target_arrived=False),
        ]}
        records = system_all_judged_edges(trajectory)
        self.assertEqual(
            [(item["target"]["target_index"], item["online_status"])
             for item in records],
            [(0, "unknown"), (1, "completed")])
        self.assertEqual(records[0]["stage_targets"], [trajectory["targets"][0]])
        self.assertEqual(records[1]["online_confidence"], 0.7)
        # The stage candidate view still collapses to the completed edge.
        self.assertEqual(
            system_stage_completion_candidates(trajectory)[0]["target"][
                "target_index"], 1)

    def test_edge_keyframes_fall_back_to_graph_edge_metadata(self):
        edges_by_id = {"edge_0000": {
            "edge_id": "edge_0000",
            "metadata": {"edge_keyframes": [
                {"image_path": "edge_keyframes/target_00_turn_start.jpg",
                 "phase": "turn_start"},
                {"image_path": "edge_keyframes/target_00_keyframe_00.jpg"},
            ]},
        }}
        from_graph = _stage_target_edge_keyframes(
            {"navigation_graph_edge_id": "edge_0000"}, edges_by_id)
        self.assertEqual(
            [item["image_path"] for item in from_graph],
            ["edge_keyframes/target_00_turn_start.jpg",
             "edge_keyframes/target_00_keyframe_00.jpg"])
        own = _stage_target_edge_keyframes(
            {"navigation_graph_edge_id": "edge_0000",
             "edge_keyframes": [{"image_path": "own.jpg"}]}, edges_by_id)
        self.assertEqual([item["image_path"] for item in own], ["own.jpg"])
        self.assertEqual(_stage_target_edge_keyframes(
            {"navigation_graph_edge_id": "edge_9999"}, edges_by_id), [])

    def test_motion_summary_counts_commanded_actions_without_pose(self):
        summary = _motion_summary({"action_history": [
            {"action": "turn_right", "commanded_turn_deg": -15.0},
            {"action": "turn_right", "commanded_turn_deg": -15.0},
            {"action": "move_forward", "commanded_turn_deg": 0.0},
            {"action": "turn_left", "commanded_turn_deg": 15.0},
        ]})
        self.assertEqual(summary["action_count"], 4)
        self.assertEqual(summary["forward_command_count"], 1)
        self.assertEqual(summary["right_turn_command_count"], 2)
        self.assertEqual(summary["left_turn_command_count"], 1)
        self.assertEqual(summary["commanded_turn_deg_total"], -15.0)
        self.assertEqual(summary["signed_turn_deg"], 0.0)
        self.assertEqual(summary["traveled_distance_m"], 0.0)

    def test_verdict_still_ands_geometry_gate_when_vlm_is_forced(self):
        class Backend:
            calls = 0

            def generate_json(self, prompt, images, schema):
                Backend.calls += 1
                return {
                    "semantic_completion_verified": True,
                    "ordered_stage_boundary_verified": True,
                    "confidence": 0.9, "reason": "r", "visual_evidence": "v",
                }

        scored = {
            "selection_within_30deg": False,
            "executed_edge_within_30deg_and_forward": True,
            "selection_heading_error_to_gt_deg": 80.0,
            "executed_heading_error_to_gt_deg": 10.0,
            "gt_path_progress_delta_m": 0.5,
        }
        image = np.zeros((24, 32, 3), np.uint8)
        result = _verify_one(
            Backend(), {"navigation_instruction": "Turn right",
                        "form": "TURN_RIGHT"},
            None, None, {"action_history": []}, scored, [image])
        self.assertEqual(Backend.calls, 1)
        self.assertTrue(result["model_semantic_completion"])
        self.assertFalse(result["semantic_completion_verified"])
        self.assertFalse(result["independent_vlm_skipped"])
        self.assertFalse(result["postrun_gt_geometry_gate"]["passed"])

    def test_geometry_gate_requires_selection_edge_and_forward_progress(self):
        valid = {
            "selection_within_30deg": True,
            "executed_edge_within_30deg_and_forward": True,
        }
        self.assertTrue(_geometry_gate(valid))
        for field in valid:
            invalid = dict(valid)
            invalid[field] = False
            self.assertFalse(_geometry_gate(invalid))

    def test_independent_prompt_treats_pass_region_as_topological_rgb_event(self):
        class Backend:
            def generate_json(self, prompt, images, schema):
                self.prompt = prompt
                return {
                    "semantic_completion_verified": True,
                    "ordered_stage_boundary_verified": True,
                    "confidence": 0.8, "reason": "distinct foyer",
                    "visual_evidence": "old room is behind",
                }

        backend = Backend()
        scored = {
            "selection_within_30deg": True,
            "executed_edge_within_30deg_and_forward": True,
            "selection_heading_error_to_gt_deg": 3.0,
            "executed_heading_error_to_gt_deg": 4.0,
            "gt_path_progress_delta_m": 1.0,
        }
        image = np.zeros((24, 32, 3), np.uint8)
        result = _verify_one(
            backend,
            {"navigation_instruction": "Walk past the living room",
             "form": "PASS_LANDMARK", "landmark": "living room"},
            None, None, {"action_history": []}, scored,
            [image, image, image])
        self.assertTrue(result["semantic_completion_verified"])
        self.assertIn("topological RGB transition", backend.prompt)
        self.assertIn("clean RGB is authoritative", backend.prompt)
        self.assertIn("obstacle need not remain visible", backend.prompt)


if __name__ == "__main__":
    unittest.main()
