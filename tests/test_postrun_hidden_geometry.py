import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_active_stop_round import audit_episode
from postrun_hidden_geometry import load_hidden_geometry


def _write_geometry(episode_dir, events):
    evaluation_dir = Path(episode_dir) / "evaluation_only"
    evaluation_dir.mkdir(parents=True)
    (evaluation_dir / "evaluation_geometry.json").write_text(json.dumps({
        "scope": "postrun_test_validation_only",
        "point_events": events,
    }))


class PostrunHiddenGeometryTest(unittest.TestCase):
    def test_pairs_selected_and_stopped_events_by_target_index(self):
        with tempfile.TemporaryDirectory() as root:
            _write_geometry(root, [
                {"event": "point_selected",
                 "payload": {"target_index": 0},
                 "position_xyz": [0.0, 0.1, 0.0], "yaw_rad": 1.0,
                 "selected_point_navmesh_xyz": [3.0, 0.1, 0.0]},
                {"event": "point_navigation_stopped",
                 "payload": {"target_index": 0,
                             "policy_declared_arrival": True,
                             "end_reason": "rgb_only_dense_stop_cluster_arrival"},
                 "position_xyz": [2.0, 0.1, 0.0], "yaw_rad": 1.0,
                 "final_target_geodesic_distance_m": 1.0},
                {"event": "point_selected",
                 "payload": {"target_index": 1},
                 "position_xyz": [2.0, 0.1, 0.0]},
            ])
            geometry = load_hidden_geometry(root)
        self.assertEqual(sorted(geometry), [0])
        self.assertEqual(geometry[0]["start_xyz"], [0.0, 0.1, 0.0])
        self.assertEqual(geometry[0]["end_xyz"], [2.0, 0.1, 0.0])
        self.assertEqual(geometry[0]["selected_navmesh_xyz"], [3.0, 0.1, 0.0])
        self.assertTrue(geometry[0]["policy_declared_arrival"])
        self.assertEqual(geometry[0]["final_target_geodesic_distance_m"], 1.0)

    def test_missing_file_returns_empty_dict(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(load_hidden_geometry(root), {})

    def test_audit_episode_uses_override_without_mutating_trajectory(self):
        trajectory = {
            "reference_state": {"position_xyz": [0.0, 0.0, 0.0]},
            "targets": [{
                "target_index": 0,
                "policy_input_contract": "rgb_only_v1",
                "point_target_arrived": True,
                "sub_instruction": {"sub_instruction_id": 0,
                                    "form": "ADVANCE_STRAIGHT"},
                "instruction_completion": {"status": "unknown"},
            }],
            "instruction_stages": [{"sub_instruction_id": 0}],
        }
        dataset_episode = {"reference_path": [
            [0.0, 0.0, 0.0], [0.0, 0.0, 5.0]]}
        without = audit_episode(0, trajectory, dataset_episode)["targets"][0]
        self.assertIsNone(without["selection_heading_error_to_gt_deg"])
        self.assertEqual(without["gt_path_progress_delta_m"], 0.0)
        hidden = {0: {"end_xyz": [0.0, 0.0, 2.0],
                      "selected_navmesh_xyz": [0.0, 0.0, 3.0]}}
        scored = audit_episode(
            0, trajectory, dataset_episode, hidden_geometry=hidden)["targets"][0]
        self.assertAlmostEqual(scored["gt_path_progress_delta_m"], 2.0)
        self.assertAlmostEqual(scored["selection_heading_error_to_gt_deg"], 0.0)
        self.assertTrue(scored["selection_within_30deg"])
        self.assertTrue(scored["executed_edge_within_30deg_and_forward"])
        self.assertNotIn("steps", trajectory["targets"][0])
        self.assertNotIn("selected_navmesh_target_xyz", trajectory["targets"][0])


if __name__ == "__main__":
    unittest.main()
