#!/usr/bin/env python3
"""The RGB-only edge judge must accept any keyframe count and a rule stub."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from instruction_decomposer import SubInstruction  # noqa: E402
from vlm_harness import HeuristicBackend, NavigationVLMHarness  # noqa: E402


class RecordingHeuristicBackend(HeuristicBackend):
    def __init__(self):
        self.calls = []

    def generate_json(self, prompt, images, schema):
        self.calls.append({"prompt": prompt, "images": images, "schema": schema})
        return super().generate_json(prompt, images, schema)


class RGBOnlyEdgeJudgeTest(unittest.TestCase):
    @staticmethod
    def views(count, shade):
        return [np.full((48, 64, 3), shade, np.uint8) for _ in range(count)]

    def judge(self, keyframe_count, views=8, edge_action_history=None):
        backend = RecordingHeuristicBackend()
        harness = NavigationVLMHarness(backend, retries=0)
        sub_instruction = SubInstruction.from_mapping({
            "sub_instruction_id": 0,
            "navigation_instruction": "walk into the kitchen",
            "form": "ENTER_REGION",
        })
        result = harness.judge_edge_instruction_completion_rgb_only(
            sub_instruction=sub_instruction, following_sub_instruction=None,
            previous_node_id="node_0000", current_node_id="node_0001",
            previous_views=self.views(views, 40),
            current_views=self.views(views, 90),
            edge_action_history=(
                edge_action_history if edge_action_history is not None
                else [{"action": "move_forward", "step": 0}]),
            edge_keyframes=self.views(keyframe_count, 120))
        return backend, result

    def test_turn_to_view_actions_are_counted_and_kept_in_prompt(self):
        turn_actions = [{
            "step": index - 6, "action": "turn_right",
            "commanded_turn_deg": -15.0, "forward_commanded": False,
            "orientation_only": True, "phase": "turn_to_selected_target",
            "policy_input_contract": "rgb_only_v1",
        } for index in range(6)]
        forward_actions = [{
            "step": step, "action": "move_forward", "commanded_turn_deg": 0.0,
            "forward_commanded": True, "rgb_motion_score": 12.0,
            "rgb_motion_threshold": 2.0, "policy_input_contract": "rgb_only_v1",
        } for step in range(2)]
        backend, _ = self.judge(
            6, edge_action_history=turn_actions + forward_actions)
        prompt = backend.calls[-1]["prompt"]
        self.assertIn('"right_turn_command_count": 6', prompt)
        self.assertIn('"left_turn_command_count": 0', prompt)
        self.assertIn('"forward_command_count": 2', prompt)
        self.assertIn('"control_steps": 8', prompt)
        self.assertIn('"phase": "turn_to_selected_target"', prompt)
        self.assertIn('"orientation_only": true', prompt)
        self.assertNotIn("policy_input_contract", prompt.split(
            "RGB-only commanded action history:")[1].split(
            "RGB detector evidence")[0])

    def test_executor_default_five_keyframes_do_not_crash(self):
        for keyframe_count in (1, 5, 6, 7):
            backend, result = self.judge(keyframe_count)
            self.assertEqual(result["status"], "unknown", keyframe_count)
            self.assertEqual(len(backend.calls[-1]["images"]), 3)

    def test_heuristic_stub_never_claims_completion(self):
        _, result = self.judge(5)
        self.assertEqual(result["status"], "unknown")
        self.assertLessEqual(result["confidence"], 0.5)
        self.assertEqual(result["policy_input_contract"], "rgb_only_v1")
        self.assertEqual(result["privileged_inputs_used"], [])
        self.assertEqual(result["temporal_evidence"]["rgb_chronology"],
                         "no chronological evidence evaluated")

    def test_six_view_panoramas_are_accepted_with_five_keyframes(self):
        _, result = self.judge(5, views=6)
        self.assertEqual(result["status"], "unknown")


if __name__ == "__main__":
    unittest.main()
