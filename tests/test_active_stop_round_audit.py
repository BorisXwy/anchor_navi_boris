import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_active_stop_round", ROOT / "scripts" / "audit_active_stop_round.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class ActiveStopRoundAuditTest(unittest.TestCase):
    def test_polyline_progress_and_heading_are_monotonic(self):
        geometry = AUDIT._polyline([
            [0.0, 0.0, 0.0], [0.0, 0.0, -2.0], [2.0, 0.0, -2.0]])
        first = AUDIT._project_to_path([0.1, 0.0, -0.5], geometry)
        second = AUDIT._project_to_path([0.1, 0.0, -1.5], geometry)
        self.assertGreater(second["progress_m"], first["progress_m"])
        tangent = AUDIT._forward_tangent([0.0, 0.0, -0.5], geometry)
        self.assertAlmostEqual(
            AUDIT._heading_error([0.0, 0.0, -1.0], tangent), 0.0)
        self.assertAlmostEqual(
            AUDIT._heading_error([1.0, 0.0, 0.0], tangent), 90.0)

    def test_vertical_path_projection_retains_three_dimensions(self):
        geometry = AUDIT._polyline([
            [0.0, 0.0, 0.0], [0.0, 2.0, -2.0], [0.0, 2.0, -4.0]])
        low = AUDIT._project_to_path([0.0, 0.1, -0.1], geometry)
        high = AUDIT._project_to_path([0.0, 1.9, -1.9], geometry)
        self.assertGreater(high["progress_m"], low["progress_m"])

    def test_edge_horizon_heading_scores_a_turn_by_the_path_chord(self):
        geometry = AUDIT._polyline([
            [0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [2.0, 0.0, -1.0]])
        heading = AUDIT._forward_reference_heading(
            [0.0, 0.0, -0.8], geometry, 2.0)
        reference = np.array([1.8, -0.2])
        reference /= np.linalg.norm(reference)
        self.assertTrue(np.allclose(heading, reference))
        self.assertAlmostEqual(
            AUDIT._heading_error([1.8, 0.0, -0.2], heading), 0.0)

    def test_failure_taxonomy_keeps_geometric_only_hit_invalid(self):
        item = {
            "simulator_reported_success": False,
            "stop_action_issued": False,
            "goal_radius_hit": True,
            "sub_instructions_completed_online": 1,
            "sub_instruction_count": 4,
            "termination_reason": "budget_exhausted",
            "target_count": 3,
            "selected_targets_within_30deg": 3,
            "executed_edges_within_30deg_and_forward": 2,
        }
        labels = AUDIT._failure_taxonomy(item)
        self.assertIn("goal_radius_hit_without_active_stop", labels)
        self.assertIn("online_subinstruction_prefix_incomplete", labels)

    def test_failure_taxonomy_keeps_stop_outside_invalid(self):
        item = {
            "simulator_reported_success": False,
            "stop_action_issued": True,
            "goal_radius_hit": False,
            "sub_instructions_completed_online": 3,
            "sub_instruction_count": 3,
            "termination_reason": "instruction_sequence_complete",
            "target_count": 2,
            "selected_targets_within_30deg": 2,
            "executed_edges_within_30deg_and_forward": 2,
        }
        labels = AUDIT._failure_taxonomy(item)
        self.assertIn("active_stop_outside_goal_radius", labels)
        self.assertNotIn("online_subinstruction_prefix_incomplete", labels)

    def test_simulator_success_without_stage_audit_or_alignment_is_not_final(self):
        item = {
            "simulator_reported_success": True,
            "instruction_validated_aligned_success": False,
            "stop_action_issued": True,
            "goal_radius_hit": True,
            "sub_instructions_completed_online": 4,
            "sub_instruction_count": 4,
            "independent_stage_verification_valid": False,
            "completed_stage_edges_gt_aligned": False,
            "termination_reason": "instruction_sequence_complete",
            "target_count": 4,
            "selected_targets_within_30deg": 2,
            "executed_edges_within_30deg_and_forward": 2,
        }
        labels = AUDIT._failure_taxonomy(item)
        self.assertIn(
            "missing_or_failed_independent_stage_verification", labels)
        self.assertIn("completed_stage_edge_not_gt_aligned", labels)


if __name__ == "__main__":
    unittest.main()
