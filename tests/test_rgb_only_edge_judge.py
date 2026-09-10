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

    def judge(self, keyframe_count, views=8):
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
            edge_action_history=[{"action": "move_forward", "step": 0}],
            edge_keyframes=self.views(keyframe_count, 120))
        return backend, result

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
