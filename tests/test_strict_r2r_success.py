import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_point_navigation import (
    instruction_validated_goal_metrics,
    instruction_validated_spl,
    simulator_stop_goal_metrics,
    strict_stage_completion_metrics,
)


class StrictR2RSuccessTest(unittest.TestCase):
    def trajectory(self, all_count=4, evaluated_count=4, path_index=None):
        return {
            "reference_state": {"path_index": path_index},
            "all_decomposed_sub_instructions": [{}] * all_count,
            "sub_instructions": [{}] * evaluated_count,
            "task_stop": {"issued": True},
        }

    def test_accidental_goal_hit_is_invalid(self):
        result = instruction_validated_goal_metrics(
            self.trajectory(), {"success": True}, {"success": False}, True)
        self.assertFalse(result["instruction_validated_r2r_success"])
        self.assertTrue(result["invalid_goal_radius_hit"])

    def test_truncated_prefix_goal_hit_is_invalid(self):
        result = instruction_validated_goal_metrics(
            self.trajectory(evaluated_count=2),
            {"goal_radius_hit_diagnostic": True}, {"success": True}, True)
        self.assertFalse(result["instruction_validated_r2r_success"])
        self.assertTrue(result["invalid_goal_radius_hit"])

    def test_non_start_state_goal_hit_is_invalid(self):
        result = instruction_validated_goal_metrics(
            self.trajectory(path_index=5),
            {"goal_radius_hit_diagnostic": True}, {"success": True}, True)
        self.assertFalse(result["instruction_validated_r2r_success"])
        self.assertTrue(result["invalid_goal_radius_hit"])

    def test_complete_ordered_execution_and_goal_hit_succeeds(self):
        result = instruction_validated_goal_metrics(
            self.trajectory(),
            {"goal_radius_hit_diagnostic": True}, {"success": True}, True)
        self.assertTrue(result["instruction_validated_r2r_success"])
        self.assertFalse(result["invalid_goal_radius_hit"])

    def test_unverified_complete_sequence_is_not_success(self):
        result = instruction_validated_goal_metrics(
            self.trajectory(),
            {"goal_radius_hit_diagnostic": True}, {"success": True}, False)
        self.assertFalse(result["instruction_validated_r2r_success"])
        self.assertTrue(result["invalid_goal_radius_hit"])

    def test_spl_is_recomputed_after_post_run_verification(self):
        metrics = {
            "initial_geodesic_distance_m": 8.0,
            "path_length_m": 32.0,
            "spl": 0.0,
        }
        self.assertEqual(instruction_validated_spl(metrics, True), 0.25)
        self.assertEqual(instruction_validated_spl(metrics, False), 0.0)

    def test_simulator_success_requires_active_stop_and_goal_radius(self):
        result = simulator_stop_goal_metrics(
            {"task_stop": {"issued": True}},
            {"goal_radius_hit_diagnostic": True})
        self.assertTrue(result["stop_action_issued"])
        self.assertTrue(result["simulator_reported_success"])

    def test_goal_radius_without_stop_is_not_simulator_success(self):
        result = simulator_stop_goal_metrics(
            {"task_stop": {"issued": False}},
            {"goal_radius_hit_diagnostic": True})
        self.assertFalse(result["simulator_reported_success"])

    def test_stop_outside_goal_radius_is_not_simulator_success(self):
        result = simulator_stop_goal_metrics(
            {"task_stop": {"issued": True}},
            {"goal_radius_hit_diagnostic": False})
        self.assertFalse(result["simulator_reported_success"])

    def test_strict_success_requires_stop(self):
        trajectory = self.trajectory()
        trajectory["task_stop"]["issued"] = False
        result = instruction_validated_goal_metrics(
            trajectory, {"goal_radius_hit_diagnostic": True},
            {"success": True}, True)
        self.assertFalse(result["instruction_validated_r2r_success"])

    def test_stage_requires_node_judgment_and_independent_verification(self):
        import json
        import tempfile

        trajectory = {
            "sub_instructions": [{"stage_id": 0}, {"stage_id": 1}],
        }
        targets = [{
            "point_target_arrived": True,
            "node_created_after_point_arrival": True,
            "navigation_graph_node_id": "node_0001",
            "navigation_graph_edge_id": "edge_0000",
            "instruction_completion": {
                "status": "completed",
                "instruction_completed": True,
                "expected_sub_instruction_id": 0,
                "current_node_id": "node_0001",
                "incoming_edge_id": "edge_0000",
            },
        }]
        with tempfile.TemporaryDirectory() as directory:
            audit_path = Path(directory) / "stage_completion_verification.json"
            audit_path.write_text(json.dumps({"stages": [{
                "sub_instruction_id": 0,
                "semantic_completion_verified": True,
                "ordered_stage_boundary_verified": True,
                "verification_source": "frozen_manual_edge_audit",
                "evidence_artifacts": ["node_views", "edge_keyframes"],
            }]}))
            result = strict_stage_completion_metrics(
                trajectory, targets, audit_path)
        self.assertEqual(result["system_node_completed_stage_ids"], [0])
        self.assertEqual(result["independently_verified_stage_ids"], [0])
        self.assertEqual(result["verified_ordered_prefix_count"], 1)
        self.assertFalse(result["all_evaluated_stages_jointly_passed"])

    def test_stage_that_overruns_next_boundary_fails_strict_audit(self):
        import json
        import tempfile

        trajectory = {"sub_instructions": [{"stage_id": 0}]}
        targets = [{
            "point_target_arrived": True,
            "node_created_after_point_arrival": True,
            "navigation_graph_node_id": "node_0001",
            "navigation_graph_edge_id": "edge_0000",
            "instruction_completion": {
                "status": "completed",
                "instruction_completed": True,
                "expected_sub_instruction_id": 0,
                "current_node_id": "node_0001",
                "incoming_edge_id": "edge_0000",
            },
        }]
        with tempfile.TemporaryDirectory() as directory:
            audit_path = Path(directory) / "stage_completion_verification.json"
            audit_path.write_text(json.dumps({"stages": [{
                "sub_instruction_id": 0,
                "semantic_completion_verified": True,
                "ordered_stage_boundary_verified": False,
                "verification_source": "independent_rgb_action_trajectory_audit",
                "evidence_artifacts": ["edge_keyframes"],
            }]}))
            result = strict_stage_completion_metrics(
                trajectory, targets, audit_path)
        self.assertEqual(result["independently_verified_stage_ids"], [])
        self.assertEqual(result["verified_ordered_prefix_count"], 0)

    def test_chained_stop_wait_completion_uses_same_real_node_and_edge(self):
        import json
        import tempfile

        trajectory = {
            "sub_instructions": [{"stage_id": 0}, {"stage_id": 1}],
        }
        primary = {
            "status": "completed", "instruction_completed": True,
            "expected_sub_instruction_id": 0,
            "current_node_id": "node_0001", "incoming_edge_id": "edge_0000",
        }
        chained = dict(primary, expected_sub_instruction_id=1)
        targets = [{
            "point_target_arrived": True,
            "node_created_after_point_arrival": True,
            "navigation_graph_node_id": "node_0001",
            "navigation_graph_edge_id": "edge_0000",
            "instruction_completion": primary,
            "chained_stop_wait_completion": chained,
        }]
        with tempfile.TemporaryDirectory() as directory:
            audit_path = Path(directory) / "stage_completion_verification.json"
            audit_path.write_text(json.dumps({"stages": [{
                "sub_instruction_id": stage_id,
                "semantic_completion_verified": True,
                "ordered_stage_boundary_verified": True,
                "verification_source": "independent_rgb_action_trajectory_audit",
                "evidence_artifacts": ["same_arrival_edge"],
            } for stage_id in (0, 1)]}))
            result = strict_stage_completion_metrics(
                trajectory, targets, audit_path)
        self.assertEqual(result["system_node_completed_stage_ids"], [0, 1])
        self.assertTrue(result["all_evaluated_stages_jointly_passed"])

    def test_missing_audit_fails_closed(self):
        result = strict_stage_completion_metrics(
            {"sub_instructions": [{"stage_id": 0}]}, [],
            Path("/definitely/missing/stage_completion_verification.json"))
        self.assertEqual(result["verified_ordered_prefix_count"], 0)
        self.assertEqual(result["audit_status"], "missing")


if __name__ == "__main__":
    unittest.main()
