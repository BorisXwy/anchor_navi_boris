import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from r2r_stop_success import build_stop_action_record, simulator_stop_success


class R2RStopSuccessTest(unittest.TestCase):
    def test_final_completed_stage_emits_stop_at_current_pose(self):
        record = build_stop_action_record(
            sequence_completed_in_order=True,
            complete_instruction_was_evaluated=True,
            global_step=42,
            position_xyz=[1.0, 2.0, 3.0])
        self.assertTrue(record["issued"])
        self.assertEqual(record["action"], "STOP")
        self.assertEqual(record["global_step"], 42)
        self.assertEqual(record["position_xyz"], [1.0, 2.0, 3.0])

    def test_partial_sequence_cannot_emit_stop(self):
        record = build_stop_action_record(
            sequence_completed_in_order=True,
            complete_instruction_was_evaluated=False,
            global_step=42,
            position_xyz=[1.0, 2.0, 3.0])
        self.assertFalse(record["issued"])
        self.assertIsNone(record["action"])

    def test_stop_and_radius_are_jointly_required(self):
        self.assertTrue(simulator_stop_success(
            stop_action_issued=True, goal_radius_hit=True))
        self.assertFalse(simulator_stop_success(
            stop_action_issued=False, goal_radius_hit=True))
        self.assertFalse(simulator_stop_success(
            stop_action_issued=True, goal_radius_hit=False))


if __name__ == "__main__":
    unittest.main()
