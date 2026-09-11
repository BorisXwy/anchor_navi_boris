#!/usr/bin/env python3

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_point_navigation import termination_category  # noqa: E402
from instruction_decomposer import SubInstruction  # noqa: E402
from navigation_graph_memory import NavigationGraphMemory  # noqa: E402
from rgb_only_instruction_sequence import (  # noqa: E402
    RGBOnlyInstructionSequenceExplorationStrategy,
)
from rgb_only_runtime import RGBOnlyPolicySimulator  # noqa: E402
from vlm_harness import VLMProviderFatalError  # noqa: E402


class FakeRawSimulator:
    def __init__(self):
        self.pathfinder = object()
        self.actions = []

    def get_sensor_observations(self):
        views = {"rgb": np.zeros((12, 16, 4), np.uint8)}
        for index in range(1, 6):
            views[f"pano_rgb_{index}"] = np.zeros((12, 16, 4), np.uint8)
        for index in range(1, 8):
            views[f"completion_rgb_{index}"] = np.zeros((12, 16, 4), np.uint8)
        return views

    def step(self, action):
        self.actions.append(action)
        return self.get_sensor_observations()


class EmptySemanticExtractor:
    name = "rgb_only_sequence_test"

    def extract(self, six_views, six_depths=None, sub_instruction=None):
        return {"labels": [], "views": []}


class RaisingPointSelector:
    def __init__(self, error):
        self.error = error
        self.requests = []

    def select(self, request):
        self.requests.append(request)
        raise self.error


class UntouchableDependency:
    """Stand-in that fails loudly if the strategy reaches it."""

    def __getattr__(self, name):
        raise AssertionError(f"dependency accessed before selection: {name}")


def build_strategy(temporary_directory, point_selector, sub_instruction_count=2):
    raw = FakeRawSimulator()
    sim = RGBOnlyPolicySimulator(raw)
    graph = NavigationGraphMemory(
        Path(temporary_directory), semantic_extractor=EmptySemanticExtractor())
    graph.add_origin_node(
        position_xyz=None, base_yaw_rad=None, global_step=0,
        six_views=[np.zeros((12, 16, 3), np.uint8) for _ in range(6)],
        six_depths=None)
    sub_instructions = [
        SubInstruction.from_mapping({
            "sub_instruction_id": index,
            "navigation_instruction": text,
            "form": form,
        })
        for index, (text, form) in enumerate([
            ("turn right", "TURN_RIGHT"),
            ("wait by the glass panes", "STOP_WAIT"),
        ][:sub_instruction_count])
    ]
    strategy = RGBOnlyInstructionSequenceExplorationStrategy(
        sim=sim, sub_instructions=sub_instructions,
        point_selector=point_selector,
        point_navigation_executor=UntouchableDependency(),
        graph_memory=graph, vlm_harness=UntouchableDependency(),
        segmenter=UntouchableDependency(),
        output_dir=temporary_directory, max_exploration_hops=5)
    return strategy, raw


class RGBOnlySequenceSelectionFailureTests(unittest.TestCase):
    def test_no_candidate_runtime_error_ends_episode_gracefully(self):
        message = "No floor-bearing candidate can be sent to the VLM"
        selector = RaisingPointSelector(RuntimeError(message))
        with tempfile.TemporaryDirectory() as temporary_directory:
            strategy, raw = build_strategy(temporary_directory, selector)
            result = strategy.run(
                initial_action_heading=0.25, initial_global_step=7)
        self.assertFalse(result.success)
        self.assertEqual(result.end_reason, f"vlm_selection_failed: {message}")
        self.assertEqual(result.records, [])
        self.assertEqual(result.recovery_records, [])
        self.assertEqual(result.exploration_hops, 0)
        self.assertEqual(result.completed_sub_instructions, 0)
        self.assertEqual(result.expected_sub_instruction_id, 0)
        self.assertEqual(result.final_yaw, 0.25)
        self.assertEqual(result.next_global_step, 7)
        self.assertEqual(result.selected_yaws_rad, [])
        self.assertEqual(result.state["cursor"], 0)
        self.assertEqual(len(selector.requests), 1)
        self.assertEqual(raw.actions, [])
        self.assertEqual(termination_category(result.end_reason),
                         "no_floor_bearing_candidate")

    def test_direction_gate_and_retry_exhaustion_map_to_evaluator_categories(self):
        cases = {
            "No floor-bearing candidate remains inside the explicit right "
            "direction gate": "no_floor_bearing_candidate",
            "VLM harness exhausted retries for select_ground_target: "
            "[\"'view_index'\"]": "vlm_selection_failed_other",
        }
        for message, expected_category in cases.items():
            with self.subTest(message=message):
                selector = RaisingPointSelector(RuntimeError(message))
                with tempfile.TemporaryDirectory() as temporary_directory:
                    strategy, _ = build_strategy(temporary_directory, selector)
                    result = strategy.run()
                self.assertFalse(result.success)
                self.assertEqual(
                    result.end_reason, f"vlm_selection_failed: {message}")
                self.assertEqual(termination_category(result.end_reason),
                                 expected_category)

    def test_provider_fatal_error_still_propagates(self):
        selector = RaisingPointSelector(
            VLMProviderFatalError("DeepSeek HTTP 401: unauthorized"))
        with tempfile.TemporaryDirectory() as temporary_directory:
            strategy, _ = build_strategy(temporary_directory, selector)
            with self.assertRaises(VLMProviderFatalError):
                strategy.run()

    def test_value_error_from_selector_contract_still_propagates(self):
        selector = RaisingPointSelector(
            ValueError("rgb_only_v1 request carried a depth field"))
        with tempfile.TemporaryDirectory() as temporary_directory:
            strategy, _ = build_strategy(temporary_directory, selector)
            with self.assertRaises(ValueError):
                strategy.run()


if __name__ == "__main__":
    unittest.main()
