#!/usr/bin/env python3
"""Decomposition prompt versions: v1 stays frozen, v2 bans motion-state cues."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from instruction_decomposer import InstructionDecomposer  # noqa: E402
from vlm_harness import (  # noqa: E402
    HeuristicBackend, NavigationVLMHarness, VLMBackend)


V1_PROMPT_HEAD = """STAGE_DECOMPOSITION
You are the instruction planner for an R2R indoor navigation agent. Split the
instruction into ordered, visually grounded stages. Every stage MUST terminate
at a semantic 3D spatial region that can be represented by a walkable floor
pixel, not merely at an action such as "turn" or "go out".

Grounding rules:
- "go/exit out of a room" targets free walkable floor just BEYOND the current
  room's doorway, far enough that the camera has crossed the door frame.
- "enter room/hall" targets free floor just INSIDE that named space.
- "pass object" targets floor beyond the object along the instructed side/path.
- "turn left/right" targets visible walkable floor along the new corridor or
  opening after the turn; never target a wall merely to cause rotation.
- "stop near X" targets free floor near X at a safe offset, not pixels on X.
- "walk straight" targets distant visible floor along the corridor/open space.

For every stage specify the reference landmark, precise semantic spatial target,
its spatial relation, visual evidence that the camera has arrived, and forbidden
regions. Preserve turns, motion, landmarks and stopping conditions. Do not
invent objects not implied by the instruction or generic targets like
"somewhere ahead".
R2R instruction: Walk to the fridge. Stop at the fridge.
Return only the requested JSON."""


def stage(**overrides):
    base = {
        "stage_id": 0, "navigation_instruction": "Stop at the fridge",
        "landmark": "refrigerator",
        "completion_cue": "the refrigerator is directly ahead at close range",
        "semantic_spatial_target": "floor in front of the refrigerator",
        "spatial_relation": "in front of the refrigerator",
        "visual_arrival_evidence": "refrigerator door occupies the front sector",
        "forbidden_target": "the refrigerator surface",
    }
    base.update(overrides)
    return base


class ScriptedBackend(VLMBackend):
    """Returns one canned decomposition per call, in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate_json(self, prompt, images, schema):
        self.prompts.append(prompt)
        response = self.responses[min(len(self.prompts) - 1,
                                      len(self.responses) - 1)]
        return {"stages": [dict(item) for item in response]}


class DecompositionPromptVersionTest(unittest.TestCase):
    def test_v1_baseline_prompt_is_frozen(self):
        backend = ScriptedBackend([[stage()]])
        harness = NavigationVLMHarness(backend, retries=0)
        harness.decompose_instruction("Walk to the fridge. Stop at the fridge.")
        self.assertEqual(backend.prompts[0], V1_PROMPT_HEAD)

    def test_v2_adds_wording_rule_and_keeps_sentinel(self):
        backend = ScriptedBackend([[stage()]])
        harness = NavigationVLMHarness(
            backend, retries=0,
            decomposition_prompt_version="v2_relation_only_completion")
        harness.decompose_instruction("Walk to the fridge. Stop at the fridge.")
        prompt = backend.prompts[0]
        self.assertTrue(prompt.startswith("STAGE_DECOMPOSITION\n"))
        self.assertIn("Completion wording rule", prompt)
        self.assertIn('"stationary"', prompt)
        self.assertIn('"immediately adjacent"', prompt)
        self.assertTrue(prompt.endswith("Return only the requested JSON."))
        self.assertIn(
            "R2R instruction: Walk to the fridge. Stop at the fridge.", prompt)

    def test_unknown_version_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "decomposition prompt version"):
            NavigationVLMHarness(HeuristicBackend(),
                                 decomposition_prompt_version="v9_missing")
        with self.assertRaisesRegex(ValueError, "decomposition prompt version"):
            NavigationVLMHarness.render_decomposition_prompt("bogus", "x")

    def test_heuristic_backend_dispatches_both_versions(self):
        for version in sorted(NavigationVLMHarness.DECOMPOSITION_PROMPT_VERSIONS):
            harness = NavigationVLMHarness(
                HeuristicBackend(), retries=0,
                decomposition_prompt_version=version)
            stages = harness.decompose_instruction(
                "Leave the closet. Stop once you exit the bedroom door.")
            self.assertEqual(len(stages), 2, version)
            self.assertEqual(stages[1]["navigation_instruction"],
                             "Stop once you exit the bedroom door.")

    def test_v2_motion_state_cue_is_retried_with_feedback(self):
        backend = ScriptedBackend([
            [stage(completion_cue="camera is near the refrigerator and has stopped")],
            [stage()],
        ])
        harness = NavigationVLMHarness(
            backend, retries=1,
            decomposition_prompt_version="v2_relation_only_completion")
        stages = harness.decompose_instruction("Stop at the fridge.")
        self.assertEqual(len(backend.prompts), 2)
        self.assertIn("has stopped", backend.prompts[1])
        self.assertIn("Previous response was invalid", backend.prompts[1])
        self.assertEqual(stages[0]["completion_cue"],
                         "the refrigerator is directly ahead at close range")
        self.assertNotIn("completion_wording_repair", stages[0])

    def test_v2_extreme_proximity_evidence_is_rejected(self):
        backend = ScriptedBackend([
            [stage(visual_arrival_evidence=(
                "the open white door fills a significant portion of the "
                "forward view and is close enough to touch"))],
            [stage()],
        ])
        harness = NavigationVLMHarness(
            backend, retries=1,
            decomposition_prompt_version="v2_relation_only_completion")
        harness.decompose_instruction("Wait at the open white door.")
        self.assertEqual(len(backend.prompts), 2)
        self.assertIn("visual_arrival_evidence", backend.prompts[1])

    def test_v2_exhausted_retries_repair_instead_of_crashing(self):
        backend = ScriptedBackend([[stage(
            completion_cue="the camera is stationary near the refrigerator",
            visual_arrival_evidence="the refrigerator is immediately adjacent")]])
        harness = NavigationVLMHarness(
            backend, retries=2,
            decomposition_prompt_version="v2_relation_only_completion")
        stages = harness.decompose_instruction("Stop at the fridge.")
        self.assertEqual(len(backend.prompts), 3)
        repair = stages[0]["completion_wording_repair"]
        self.assertEqual(repair["fields"],
                         ["completion_cue", "visual_arrival_evidence"])
        self.assertEqual(repair["matched"], ["stationary", "immediately adjacent"])
        self.assertEqual(
            stages[0]["completion_cue"],
            "refrigerator is visible at the stated relation "
            "(in front of the refrigerator) within a safe stopping offset")
        self.assertEqual(stages[0]["visual_arrival_evidence"],
                         stages[0]["completion_cue"])
        self.assertEqual(repair["original"]["completion_cue"],
                         "the camera is stationary near the refrigerator")
        # The repair note survives into the SubInstruction metadata.
        sub_instructions = InstructionDecomposer(harness).decompose(
            "Stop at the fridge.")
        self.assertIn("completion_wording_repair",
                      sub_instructions[0].metadata)

    def test_v1_never_checks_cue_wording(self):
        backend = ScriptedBackend([[stage(
            completion_cue="the camera has stopped near the refrigerator")]])
        harness = NavigationVLMHarness(backend, retries=1)
        stages = harness.decompose_instruction("Stop at the fridge.")
        self.assertEqual(len(backend.prompts), 1)
        self.assertEqual(stages[0]["completion_cue"],
                         "the camera has stopped near the refrigerator")

    def test_wording_check_ignores_instruction_and_landmark_fields(self):
        violations = NavigationVLMHarness.completion_wording_violations(stage(
            navigation_instruction="Stop and wait at the bus stop sign",
            landmark="stop sign"))
        self.assertEqual(violations, [])
        self.assertEqual(
            NavigationVLMHarness.completion_wording_violations(stage(
                completion_cue="the stop sign is still visible on the left")),
            [])
        self.assertEqual(
            NavigationVLMHarness.completion_wording_violations(stage(
                completion_cue="the camera stops beside the stop sign")),
            [("completion_cue", "stops")])
        self.assertEqual(
            NavigationVLMHarness.completion_wording_violations(stage(
                visual_arrival_evidence=(
                    "the refrigerator is visible directly ahead, occupying "
                    "a significant portion of the forward view"))),
            [("visual_arrival_evidence",
              "occupying a significant portion of the forward view")])


if __name__ == "__main__":
    unittest.main()
