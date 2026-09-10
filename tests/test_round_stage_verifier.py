import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from verify_round_stage_completions import (
    _geometry_gate,
    _node_view_paths,
    _verify_one,
    system_stage_completion_candidates,
    trajectory_episode_index,
)


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
