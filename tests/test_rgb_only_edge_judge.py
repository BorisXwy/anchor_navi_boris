#!/usr/bin/env python3
"""The RGB-only edge judge must accept any keyframe count and a rule stub."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_judge_round import parse_judge_prompt  # noqa: E402
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

    def judge(self, keyframe_count, views=8, edge_action_history=None,
              prompt_version="v1_baseline", sub_instruction=None):
        backend = RecordingHeuristicBackend()
        harness = NavigationVLMHarness(
            backend, retries=0,
            rgb_only_completion_prompt_version=prompt_version)
        sub_instruction = sub_instruction or SubInstruction.from_mapping({
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

    @staticmethod
    def stop_wait_stage(**overrides):
        payload = {
            "sub_instruction_id": 2,
            "navigation_instruction": "Stop at refrigerator",
            "landmark": "refrigerator",
            "completion_cue": "camera is near the refrigerator and has stopped",
            "semantic_spatial_target": "floor in front of the refrigerator",
            "spatial_relation": "directly in front of the refrigerator",
            "visual_arrival_evidence": "the refrigerator is prominently visible",
            "forbidden_target": "the refrigerator surface",
            "form": "STOP_WAIT",
            "point_selection_strategy": {
                "arrival": "relation and safe distance hold; then emit stop"},
            "metadata": {"vlm_parent_raw_index": 1},
        }
        payload.update(overrides)
        return SubInstruction.from_mapping(payload)

    @staticmethod
    def active_block(prompt):
        return prompt.split("ACTIVE SUB-INSTRUCTION:")[1].split(
            "FOLLOWING SUB-INSTRUCTION")[0]

    def test_v1_prompt_keeps_full_sub_instruction_and_no_form_rules(self):
        backend, _ = self.judge(5, sub_instruction=self.stop_wait_stage())
        prompt = backend.calls[-1]["prompt"]
        self.assertIn("blocked/stationary", prompt)
        self.assertIn("then emit stop", self.active_block(prompt))
        self.assertNotIn("MOTION-STATE RULE", prompt)
        self.assertNotIn("STOP_WAIT RULE", prompt)

    def test_v2_compacts_sub_instruction_and_adds_stop_wait_rule(self):
        backend, _ = self.judge(
            5, prompt_version="v2_form_aware_stop_relation",
            sub_instruction=self.stop_wait_stage())
        prompt = backend.calls[-1]["prompt"]
        block = self.active_block(prompt)
        for key in ("point_selection_strategy", "metadata", "source_clause"):
            self.assertNotIn(key, block)
        for key in ("completion_cue", "visual_arrival_evidence", "form",
                    "spatial_relation", "definition"):
            self.assertIn(f'"{key}"', block)
        self.assertNotIn("stationary, ambiguous", prompt)
        self.assertIn("MOTION-STATE RULE", prompt)
        self.assertIn("STOP_WAIT RULE", prompt)
        self.assertIn("only AFTER you return completed", prompt)
        self.assertLess(prompt.index("STOP_WAIT RULE"),
                        prompt.index("Image 0 is the previous panorama"))

    def test_v2_stop_wait_rule_follows_the_form_not_the_wording(self):
        backend, _ = self.judge(
            5, prompt_version="v2_form_aware_stop_relation")
        prompt = backend.calls[-1]["prompt"]
        self.assertIn("MOTION-STATE RULE", prompt)
        self.assertNotIn("STOP_WAIT RULE", prompt)
        backend, _ = self.judge(
            5, prompt_version="v2_form_aware_stop_relation",
            sub_instruction=self.stop_wait_stage(
                form="EXIT_REGION", secondary_forms=["STOP_WAIT"]))
        self.assertIn("STOP_WAIT RULE", backend.calls[-1]["prompt"])

    def test_both_versions_stay_parseable_by_round_analysis(self):
        actions = [{"step": index, "action": "move_forward",
                    "commanded_turn_deg": 0.0, "forward_commanded": True,
                    "rgb_motion_score": 9.0} for index in range(3)]
        for version in sorted(
                NavigationVLMHarness.RGB_ONLY_COMPLETION_PROMPT_VERSIONS):
            backend, _ = self.judge(
                5, prompt_version=version, edge_action_history=actions,
                sub_instruction=self.stop_wait_stage())
            parsed = parse_judge_prompt(backend.calls[-1]["prompt"])
            self.assertEqual(parsed["sub_instruction"]["form"], "STOP_WAIT",
                             version)
            self.assertEqual(parsed["sub_instruction"]["sub_instruction_id"], 2)
            self.assertEqual(parsed["action_summary"]["forward_command_count"],
                             3, version)
            self.assertEqual(parsed["action_summary"]["control_steps"], 3)

    def test_unknown_prompt_version_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "rgb-only-completion"):
            NavigationVLMHarness(
                HeuristicBackend(),
                rgb_only_completion_prompt_version="v3_missing")
        with self.assertRaisesRegex(ValueError, "rgb-only-completion"):
            NavigationVLMHarness.render_rgb_only_completion_prompt(
                "bogus", {}, {}, {}, {}, {})
        with self.assertRaisesRegex(ValueError, "rgb-only-completion"):
            NavigationVLMHarness.rgb_only_completion_schema("bogus")

    V3 = "v3_sector_relation_evidence"

    def test_v3_keeps_v2_rules_and_adds_evidence_protocol(self):
        backend, result = self.judge(
            5, prompt_version=self.V3, sub_instruction=self.stop_wait_stage())
        prompt = backend.calls[-1]["prompt"]
        for marker in ("MOTION-STATE RULE", "STOP_WAIT RULE",
                       "EVIDENCE PROTOCOL", "REASONS THAT NEVER JUSTIFY UNKNOWN",
                       "RELATION RULE", "STOP_WAIT: apply the STOP_WAIT RULE",
                       "FALSE-POSITIVE GUARD", "CONFIDENCE:"):
            self.assertIn(marker, prompt)
        # Everything new sits in the form_rules slot after the JSON blocks.
        self.assertLess(prompt.index("RGB detector evidence at CURRENT node"),
                        prompt.index("EVIDENCE PROTOCOL"))
        self.assertLess(prompt.index("FALSE-POSITIVE GUARD"),
                        prompt.index("Image 0 is the previous panorama"))
        schema = backend.calls[-1]["schema"]
        for field in ("landmark_visible_current", "landmark_sector_current",
                      "relation_satisfied", "transition_observed"):
            self.assertIn(field, schema["properties"])
            self.assertIn(field, schema["required"])
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["prompt_version"], self.V3)
        self.assertEqual(result["relation_evidence"]["landmark_sector_current"],
                         "not_visible")

    def test_v3_relation_rule_follows_active_and_secondary_forms(self):
        backend, _ = self.judge(5, prompt_version=self.V3)
        prompt = backend.calls[-1]["prompt"]
        self.assertIn("ENTER_REGION: completed only if", prompt)
        self.assertNotIn("EXIT_REGION: completed only if", prompt)
        self.assertNotIn("GENERIC: completed only when", prompt)
        backend, _ = self.judge(
            5, prompt_version=self.V3,
            sub_instruction=self.stop_wait_stage(
                form="TURN_LEFT", secondary_forms=["PASS_LANDMARK"]))
        prompt = backend.calls[-1]["prompt"]
        self.assertIn("TURN_LEFT: completed when", prompt)
        self.assertIn("PASS_LANDMARK: completed only if", prompt)
        self.assertNotIn("STOP_WAIT RULE", prompt)
        prompt = NavigationVLMHarness.render_rgb_only_completion_prompt(
            self.V3, {"sub_instruction_id": 0, "form": "OTHER",
                      "navigation_instruction": "go over there"},
            {}, {"chronological_actions": []}, {}, {})
        self.assertIn("GENERIC: completed only when", prompt)
        self.assertNotIn(": completed only if", prompt)

    def test_v3b_shares_protocol_and_relaxes_named_rules(self):
        v3b = "v3b_relation_rules_relaxed"
        self.assertEqual(NavigationVLMHarness.rgb_only_completion_schema(v3b),
                         NavigationVLMHarness.rgb_only_completion_schema(self.V3))
        for form, marker in (("PASS_LANDMARK", "extended landmark"),
                             ("TURN_LEFT", "Do not require a corridor"),
                             ("STOP_WAIT", "may lie in ANY sector"),
                             ("EXIT_REGION", "THROUGH a doorway")):
            backend, _ = self.judge(
                5, prompt_version=v3b,
                sub_instruction=self.stop_wait_stage(form=form))
            prompt = backend.calls[-1]["prompt"]
            self.assertIn("EVIDENCE PROTOCOL", prompt)
            self.assertIn(marker, prompt, form)
            backend, _ = self.judge(
                5, prompt_version=self.V3,
                sub_instruction=self.stop_wait_stage(form=form))
            self.assertNotIn(marker, backend.calls[-1]["prompt"], form)

    def test_v2_schema_has_no_evidence_fields(self):
        schema = NavigationVLMHarness.rgb_only_completion_schema(
            "v2_form_aware_stop_relation")
        self.assertNotIn("relation_satisfied", schema["properties"])
        self.assertEqual(sorted(schema["required"]), sorted(
            ["status", "confidence", "reason", "visual_evidence",
             "temporal_evidence"]))

    def test_v3_normalization_overrides_and_coercions(self):
        normalize = NavigationVLMHarness.normalize_rgb_only_completion_result
        actions = [{"action": "move_forward"}]
        base = {"status": "Completed", "confidence": 0.9, "reason": "r",
                "visual_evidence": "v", "temporal_evidence": "t",
                "landmark_visible_current": "true",
                "landmark_sector_current": "Behind",
                "relation_satisfied": True, "transition_observed": "yes"}
        out = normalize(self.V3, base, actions)
        self.assertEqual(out["status"], "completed")
        self.assertEqual(out["overrides"], [])
        self.assertEqual(out["evidence"], {
            "landmark_visible_current": True,
            "landmark_sector_current": "rear",
            "relation_satisfied": True, "transition_observed": True})
        # A completed verdict whose own evidence denies the relation is
        # demoted deterministically.
        out = normalize(self.V3, {**base, "relation_satisfied": False}, actions)
        self.assertEqual(out["status"], "unknown")
        self.assertLessEqual(out["confidence"], 0.49)
        self.assertEqual(out["overrides"], ["relation_not_satisfied"])
        self.assertEqual(out["model_status"], "completed")
        # The empty-edge override applies to every version.
        out = normalize("v2_form_aware_stop_relation",
                        {k: base[k] for k in ("status", "confidence")}, [])
        self.assertEqual(out["overrides"], ["empty_action_edge"])
        self.assertEqual(out["evidence"], {})
        with self.assertRaisesRegex(ValueError, "landmark_sector_current"):
            normalize(self.V3, {**base, "landmark_sector_current": "up"},
                      actions)
        with self.assertRaisesRegex(ValueError, "boolean"):
            normalize(self.V3, {**base, "relation_satisfied": "maybe"},
                      actions)
        with self.assertRaises(KeyError):
            normalize(self.V3, {k: base[k] for k in ("status", "confidence")},
                      actions)


if __name__ == "__main__":
    unittest.main()
