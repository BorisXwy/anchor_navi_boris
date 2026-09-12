#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from direction_gate import (  # noqa: E402
    IN_PLACE_TURN_PHASE, TURN_GATE_CENTER_DEG, DirectionGateEmptyError,
    in_place_turn_step_count, plan_in_place_turn,
)


class DirectionGatePlanTests(unittest.TestCase):
    def test_sector_centres_are_the_commanded_turn_angles(self):
        self.assertEqual(
            TURN_GATE_CENTER_DEG,
            {"left": 90.0, "right": -90.0, "rear": 180.0})

    def test_left_turn_plan_uses_six_left_steps_at_fifteen_degrees(self):
        plan = plan_in_place_turn("left", 15.0)
        self.assertEqual(len(plan), 6)
        self.assertEqual({item["action"] for item in plan}, {"turn_left"})
        self.assertEqual({item["commanded_turn_deg"] for item in plan}, {15.0})
        self.assertEqual([item["step"] for item in plan], list(range(6)))
        for item in plan:
            self.assertTrue(item["orientation_only"])
            self.assertFalse(item["forward_commanded"])
            self.assertEqual(item["phase"], IN_PLACE_TURN_PHASE)
            self.assertEqual(item["policy_input_contract"], "rgb_only_v1")

    def test_right_turn_plan_uses_right_steps_with_negative_delta(self):
        plan = plan_in_place_turn("right", 15.0)
        self.assertEqual(len(plan), 6)
        self.assertEqual({item["action"] for item in plan}, {"turn_right"})
        self.assertEqual({item["commanded_turn_deg"] for item in plan}, {-15.0})

    def test_rear_plan_is_a_half_rotation_to_the_left(self):
        plan = plan_in_place_turn("rear", 15.0)
        self.assertEqual(len(plan), 12)
        self.assertEqual({item["action"] for item in plan}, {"turn_left"})
        self.assertEqual(in_place_turn_step_count("rear", 30.0), 6)

    def test_turn_step_must_divide_the_sector_centre(self):
        with self.assertRaises(ValueError):
            plan_in_place_turn("left", 40.0)
        with self.assertRaises(ValueError):
            in_place_turn_step_count("forward", 15.0)


class DirectionGateEmptyErrorTests(unittest.TestCase):
    def test_default_message_matches_the_historical_runtime_error(self):
        error = DirectionGateEmptyError("right")
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(error.sector, "right")
        self.assertEqual(
            str(error),
            "No floor-bearing candidate remains inside the explicit right "
            "direction gate")

    def test_explicit_message_is_kept(self):
        error = DirectionGateEmptyError("forward", "custom text")
        self.assertEqual(str(error), "custom text")
        self.assertEqual(error.sector, "forward")


if __name__ == "__main__":
    unittest.main()
