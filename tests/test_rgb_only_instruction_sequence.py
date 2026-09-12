#!/usr/bin/env python3

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_rgb_only_contract import _violations  # noqa: E402
from direction_gate import DirectionGateEmptyError  # noqa: E402
from evaluate_point_navigation import termination_category  # noqa: E402
from instruction_decomposer import SubInstruction  # noqa: E402
from navigation_graph_memory import NavigationGraphMemory  # noqa: E402
from point_navigation_executor import PointNavigationResult  # noqa: E402
import point_selectors  # noqa: E402
from point_selectors import (  # noqa: E402
    InstructionVLMPointSelector, PointSelectionRequest, PointSelectionResult,
)
from rgb_only_instruction_sequence import (  # noqa: E402
    RGBOnlyBacktrackResult, RGBOnlyInstructionSequenceExplorationStrategy,
)
from rgb_only_runtime import RGBOnlyPolicySimulator  # noqa: E402
from vlm_harness import VLMProviderFatalError  # noqa: E402


class FakeRawSimulator:
    def __init__(self):
        self.pathfinder = object()
        self.actions = []

    def get_sensor_observations(self):
        # The forward frame encodes how many turns were issued so tests can
        # tell the pre-turn frame from the post-turn frame.
        shade = min(255, 10 + 20 * len(self.actions))
        views = {"rgb": np.full((12, 16, 4), shade, np.uint8)}
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


TURN_STEPS = 6
EXECUTOR_STEPS = 2


class TurningPointSelector:
    """Mimics choose_view: probes, turns toward the chosen view, returns it."""

    def __init__(self, turn_steps=TURN_STEPS):
        self.turn_steps = int(turn_steps)

    def select(self, request):
        if self.turn_steps:
            # Eight-view refinement probes turn out and back; they must not
            # become edge actions.
            for phase in ("rgb_refinement_probe", "rgb_refinement_probe_return"):
                request.motion_log.append({
                    "phase": phase, "target_index": request.target_index,
                    "turn_frame": 1, "action": "turn_left",
                    "commanded_turn_deg": 15.0,
                    "policy_input_contract": "rgb_only_v1"})
        for step in range(1, self.turn_steps + 1):
            request.sim.step("turn_right")
            request.motion_log.append({
                "phase": "turn_to_selected_target",
                "target_index": request.target_index, "turn_frame": step,
                "action": "turn_right", "commanded_turn_deg": -15.0,
                "policy_input_contract": "rgb_only_v1"})
        rgb = request.sim.get_sensor_observations()["rgb"]
        mask = np.ones(rgb.shape[:2], bool)
        return PointSelectionResult(chosen={
            "rgb": rgb, "yaw": request.yaw - math.radians(15 * self.turn_steps),
            "relative_yaw_rad": -math.radians(15 * self.turn_steps),
            "point": np.array([8.0, 9.0], np.float32),
            "target_mask": mask, "mask": mask, "vlm_selection": {},
        }, candidates=[])


class ForwardExecutor:
    def __init__(self, output_dir):
        self.keyframes_dir = Path(output_dir) / "edge_keyframes"
        self.keyframes_dir.mkdir(parents=True, exist_ok=True)
        self.requests = []

    def execute(self, request):
        self.requests.append(request)
        actions = [{
            "step": step, "action": "move_forward", "commanded_turn_deg": 0.0,
            "forward_commanded": True, "rgb_motion_score": 12.0,
            "rgb_motion_threshold": 2.0, "policy_input_contract": "rgb_only_v1",
        } for step in range(EXECUTOR_STEPS)]
        frames = [np.asarray(request.rgb, np.uint8)[..., :3]] + [
            np.full((12, 16, 3), 200 + step, np.uint8)
            for step in range(EXECUTOR_STEPS - 1)]
        keyframe_records = []
        for order, frame in enumerate(frames):
            path = self.keyframes_dir / (
                f"target_{request.target_index:02d}_keyframe_{order:02d}.jpg")
            Image.fromarray(frame).save(path)
            keyframe_records.append({
                "step": order - 1,
                "phase": "edge_start" if order == 0 else "edge_motion",
                "action": None if order == 0 else "move_forward",
                "policy_input_contract": "rgb_only_v1",
                "keyframe_index": order, "source_frame_index": order,
                "image_path": str(path.relative_to(self.keyframes_dir.parent)),
            })
        return PointNavigationResult(
            arrived=True, signal="rgb_only_dense_stop_cluster_arrival",
            end_reason="rgb_only_dense_stop_cluster_arrival",
            final_rgb=frames[-1], final_yaw=request.yaw,
            next_global_step=request.global_step + EXECUTOR_STEPS,
            action_history=actions,
            record={"action_history": actions, "steps": [],
                    "edge_keyframes": keyframe_records,
                    "arrival_signal": "rgb_only_dense_stop_cluster_arrival"},
            edge_keyframes=frames)


class StallingExecutor(ForwardExecutor):
    """First hop ends as an RGB forward stall, later hops arrive normally."""

    def execute(self, request):
        result = super().execute(request)
        if len(self.requests) > 1:
            return result
        stalled_actions = [dict(item, rgb_motion_score=0.0,
                                stall_forward_streak=index + 1)
                           for index, item in enumerate(result.action_history)]
        record = dict(result.record, action_history=stalled_actions,
                      arrival_signal=None, end_reason="rgb_forward_stall",
                      terminal_stall_forward_streak=len(stalled_actions))
        return PointNavigationResult(
            arrived=False, signal=None, end_reason="rgb_forward_stall",
            final_rgb=result.final_rgb, final_yaw=result.final_yaw,
            next_global_step=result.next_global_step,
            action_history=stalled_actions, record=record,
            edge_keyframes=result.edge_keyframes)


class RecordingBacktracker:
    def __init__(self):
        self.calls = []

    def recover(self, target_node_id, heading, global_step, target_index,
                failed_action_history):
        self.calls.append({
            "target_node_id": target_node_id,
            "failed_end_reasons": [
                item.get("stall_forward_streak") for item in failed_action_history],
        })
        return RGBOnlyBacktrackResult(
            success=True, target_node_id=target_node_id,
            recovered_node_id=target_node_id,
            final_action_heading_rad=float(heading),
            next_global_step=int(global_step), attempts=[],
            end_reason="rgb_only_backtrack_already_at_node")


class RecordingVLMHarness:
    def __init__(self):
        self.judge_calls = []

    def judge_edge_instruction_completion_rgb_only(self, **kwargs):
        self.judge_calls.append(kwargs)
        return {"status": "completed", "confidence": 0.9,
                "reason": "turned and moved as instructed",
                "visual_evidence": "view rotated right then advanced",
                "temporal_evidence": {}}


class ScriptedGateSelector:
    """Raise the scripted errors in order; otherwise act like TurningPointSelector."""

    def __init__(self, errors, turn_steps=0):
        self.errors = list(errors)
        self.requests = []
        self.turning = TurningPointSelector(turn_steps)

    def select(self, request):
        self.requests.append(request)
        if self.errors:
            error = self.errors.pop(0)
            if error is not None:
                raise error
        return self.turning.select(request)


class ScriptedVLMHarness(RecordingVLMHarness):
    def __init__(self, statuses):
        super().__init__()
        self.statuses = list(statuses)

    def judge_edge_instruction_completion_rgb_only(self, **kwargs):
        result = super().judge_edge_instruction_completion_rgb_only(**kwargs)
        status = self.statuses.pop(0) if self.statuses else "completed"
        return dict(result, status=status,
                    confidence=0.9 if status == "completed" else 0.4)


def build_strategy(temporary_directory, point_selector, sub_instruction_count=2,
                   executor=None, vlm_harness=None,
                   backtrack_method="action-reversal", **strategy_kwargs):
    raw = FakeRawSimulator()
    sim = RGBOnlyPolicySimulator(raw)
    graph = NavigationGraphMemory(
        Path(temporary_directory), semantic_extractor=EmptySemanticExtractor())
    graph.add_origin_node(
        position_xyz=None, base_yaw_rad=None, global_step=0,
        six_views=[np.zeros((12, 16, 3), np.uint8) for _ in range(6)],
        six_depths=None,
        completion_views=[np.zeros((12, 16, 3), np.uint8) for _ in range(8)])
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
        point_navigation_executor=(
            executor if executor is not None else UntouchableDependency()),
        graph_memory=graph,
        vlm_harness=(
            vlm_harness if vlm_harness is not None else UntouchableDependency()),
        segmenter=UntouchableDependency(),
        output_dir=temporary_directory, max_exploration_hops=5,
        backtrack_method=backtrack_method, **strategy_kwargs)
    return strategy, raw


class RGBOnlySequenceTurnHistoryTests(unittest.TestCase):
    def run_one_hop(self, temporary_directory, turn_steps):
        executor = ForwardExecutor(temporary_directory)
        harness = RecordingVLMHarness()
        strategy, raw = build_strategy(
            temporary_directory, TurningPointSelector(turn_steps),
            sub_instruction_count=1, executor=executor, vlm_harness=harness)
        result = strategy.run(initial_action_heading=0.0, initial_global_step=3)
        return strategy, raw, executor, harness, result

    def test_turn_to_view_commands_are_prepended_to_edge_history(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            strategy, raw, executor, harness, result = self.run_one_hop(
                temporary_directory, TURN_STEPS)
            self.assertTrue(result.success)
            self.assertEqual(result.end_reason, "instruction_sequence_complete")
            self.assertEqual(raw.actions, ["turn_right"] * TURN_STEPS)

            edge = strategy.graph_memory.edges[-1]
            history = edge.action_history
            self.assertEqual(len(history), TURN_STEPS + EXECUTOR_STEPS)
            turn_part, executor_part = (
                history[:TURN_STEPS], history[TURN_STEPS:])
            self.assertEqual([item["step"] for item in turn_part],
                             list(range(-TURN_STEPS, 0)))
            for item in turn_part:
                self.assertEqual(item["action"], "turn_right")
                self.assertEqual(item["commanded_turn_deg"], -15.0)
                self.assertFalse(item["forward_commanded"])
                self.assertTrue(item["orientation_only"])
                self.assertEqual(item["phase"], "turn_to_selected_target")
                self.assertNotIn("rgb_motion_score", item)
            self.assertEqual([item["action"] for item in executor_part],
                             ["move_forward"] * EXECUTOR_STEPS)
            self.assertFalse(any(
                "refinement" in str(item.get("phase")) for item in history))
            self.assertEqual(edge.control_step_count, TURN_STEPS + EXECUTOR_STEPS)
            self.assertEqual(edge.traveled_distance_m, 0.0)

            # The hop record, the persisted edge and the judge input must be
            # the same list (post-run edge integrity check compares them).
            self.assertEqual(result.records[0]["action_history"], history)
            judge_kwargs = harness.judge_calls[-1]
            self.assertEqual(judge_kwargs["edge_action_history"], history)

            # Judge storyboard: pre-turn frame first, then executor frames.
            keyframes = judge_kwargs["edge_keyframes"]
            self.assertEqual(len(keyframes), EXECUTOR_STEPS + 1)
            self.assertEqual(int(keyframes[0][0, 0, 0]), 10)
            self.assertEqual(int(keyframes[1][0, 0, 0]), 10 + 20 * TURN_STEPS)

            records = edge.metadata["edge_keyframes"]
            self.assertEqual(records[0]["phase"], "turn_start")
            self.assertEqual(records[0]["keyframe_index"], 0)
            self.assertEqual(records[0]["source_frame_index"], -1)
            self.assertEqual(records[0]["step"], -(TURN_STEPS + 1))
            self.assertTrue(
                (Path(temporary_directory) / records[0]["image_path"]).exists())
            self.assertEqual([item["keyframe_index"] for item in records[1:]],
                             list(range(1, EXECUTOR_STEPS + 1)))
            self.assertEqual(records[1]["phase"], "edge_start")

            graph_payload = json.loads(
                strategy.graph_memory.graph_path.read_text())
            self.assertEqual(_violations(graph_payload["edges"]), [])

            # The executor's own record is left untouched.
            self.assertEqual(len(result.records[0]["executor"]["action_history"]),
                             EXECUTOR_STEPS)

    def test_zero_turn_hop_keeps_executor_history_and_keyframes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            strategy, raw, executor, harness, result = self.run_one_hop(
                temporary_directory, 0)
            self.assertTrue(result.success)
            self.assertEqual(raw.actions, [])
            edge = strategy.graph_memory.edges[-1]
            self.assertEqual([item["action"] for item in edge.action_history],
                             ["move_forward"] * EXECUTOR_STEPS)
            self.assertEqual(len(harness.judge_calls[-1]["edge_keyframes"]),
                             EXECUTOR_STEPS)
            records = edge.metadata["edge_keyframes"]
            self.assertEqual(records[0]["phase"], "edge_start")
            self.assertEqual([item["keyframe_index"] for item in records],
                             list(range(EXECUTOR_STEPS)))
            self.assertFalse(
                (Path(temporary_directory) / "edge_keyframes" /
                 "target_00_turn_start.jpg").exists())


class RGBOnlySequenceForwardStallTests(unittest.TestCase):
    def test_stalled_hop_backtracks_blocks_direction_and_continues(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            executor = StallingExecutor(temporary_directory)
            harness = RecordingVLMHarness()
            strategy, raw = build_strategy(
                temporary_directory, TurningPointSelector(0),
                sub_instruction_count=1, executor=executor,
                vlm_harness=harness, backtrack_method="visual")
            backtracker = RecordingBacktracker()
            strategy.backtracker = backtracker
            origin_node_id = strategy.state.last_verified_node_id
            result = strategy.run(initial_action_heading=0.0,
                                  initial_global_step=0)

            self.assertTrue(result.success)
            self.assertEqual(result.end_reason, "instruction_sequence_complete")
            self.assertEqual(len(executor.requests), 2)

            stalled, arrived = result.records[0], result.records[1]
            self.assertEqual(stalled["end_reason"], "rgb_forward_stall")
            self.assertFalse(stalled["arrived"])
            self.assertIn("physical_failure_recovery", stalled)
            self.assertEqual(len(backtracker.calls), 1)
            self.assertEqual(backtracker.calls[0]["failed_end_reasons"],
                             list(range(1, EXECUTOR_STEPS + 1)))
            self.assertEqual(
                strategy.state.blocked_yaws_by_verified_node[origin_node_id],
                [0.0])
            self.assertTrue(arrived["arrived"])
            self.assertNotIn("physical_failure_recovery", arrived)
            # A stalled hop never becomes a graph node; only the arrival did.
            self.assertEqual(len(strategy.graph_memory.edges), 1)
            self.assertEqual(len(harness.judge_calls), 1)


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


class RGBOnlySequenceEmptyTurnGateTests(unittest.TestCase):
    def test_empty_turn_gate_rotates_in_place_and_hands_edge_to_judge(self):
        selector = ScriptedGateSelector([DirectionGateEmptyError("right")])
        harness = RecordingVLMHarness()
        with tempfile.TemporaryDirectory() as temporary_directory:
            strategy, raw = build_strategy(
                temporary_directory, selector, sub_instruction_count=1,
                vlm_harness=harness)
            result = strategy.run(initial_action_heading=0.0,
                                  initial_global_step=3)
            graph = strategy.graph_memory
            keyframe_files = sorted(
                (Path(temporary_directory) / "edge_keyframes").glob(
                    "target_00_in_place_turn_*.jpg"))
            self.assertEqual(len(keyframe_files), 5)

        self.assertTrue(result.success)
        self.assertEqual(result.end_reason, "instruction_sequence_complete")
        self.assertEqual(raw.actions, ["turn_right"] * 6)
        self.assertAlmostEqual(result.final_yaw, -math.pi / 2)
        self.assertEqual(result.next_global_step, 9)
        self.assertEqual(len(selector.requests), 1)

        self.assertEqual(len(graph.nodes), 2)
        edge = graph.edges[-1]
        self.assertEqual(edge.edge_kind, "instruction_sequence_rgb_only_in_place_turn")
        self.assertEqual(len(edge.action_history), 6)
        self.assertEqual({item["action"] for item in edge.action_history},
                         {"turn_right"})
        self.assertEqual(edge.metadata["in_place_turn"]["sector"], "right")
        self.assertEqual(graph.nodes[-1].arrival_signal,
                         "direction_gate_in_place_turn")

        judge_call = harness.judge_calls[0]
        self.assertEqual(len(judge_call["edge_action_history"]), 6)
        self.assertEqual(len(judge_call["edge_keyframes"]), 5)

        record = result.records[0]
        self.assertEqual(record["in_place_turn"]["sector"], "right")
        self.assertEqual(record["in_place_turn"]["turn_command_count"], 6)
        self.assertFalse(record["point_target_arrived"])
        self.assertTrue(record["sub_instruction_satisfied"])
        self.assertEqual(
            result.state["in_place_turn_records_by_sub_instruction_id"][0][
                "sector"], "right")
        self.assertEqual(_violations(record), [])

    def test_rear_gate_turns_half_a_rotation_left(self):
        selector = ScriptedGateSelector([DirectionGateEmptyError("rear")])
        with tempfile.TemporaryDirectory() as temporary_directory:
            strategy, raw = build_strategy(
                temporary_directory, selector, sub_instruction_count=1,
                vlm_harness=RecordingVLMHarness())
            result = strategy.run()
        self.assertEqual(raw.actions, ["turn_left"] * 12)
        self.assertAlmostEqual(abs(result.final_yaw), math.pi)

    def test_in_place_turn_is_used_once_per_sub_instruction(self):
        selector = ScriptedGateSelector([
            DirectionGateEmptyError("right"), DirectionGateEmptyError("forward")])
        with tempfile.TemporaryDirectory() as temporary_directory:
            strategy, raw = build_strategy(
                temporary_directory, selector, sub_instruction_count=1,
                vlm_harness=ScriptedVLMHarness(["unknown"]))
            result = strategy.run()
        self.assertFalse(result.success)
        self.assertTrue(result.end_reason.startswith("vlm_selection_failed:"))
        self.assertIn("explicit forward direction gate", result.end_reason)
        self.assertEqual(termination_category(result.end_reason),
                         "no_floor_bearing_candidate")
        self.assertEqual(raw.actions, ["turn_right"] * 6)
        self.assertEqual(len(selector.requests), 2)
        retry_stage = selector.requests[1].stage
        self.assertTrue(
            retry_stage["metadata"]["in_place_turn_executed"]["active"])
        self.assertEqual(
            retry_stage["metadata"]["in_place_turn_executed"]["sector"], "right")
        self.assertEqual(len(result.records), 1)
        self.assertFalse(result.records[0]["sub_instruction_satisfied"])

    def test_second_side_gate_error_for_same_clause_ends_episode(self):
        # Even if the harness reported a side sector again, the strategy
        # itself refuses a second in-place turn for the same clause.
        selector = ScriptedGateSelector([
            DirectionGateEmptyError("left"), DirectionGateEmptyError("left")])
        with tempfile.TemporaryDirectory() as temporary_directory:
            strategy, raw = build_strategy(
                temporary_directory, selector, sub_instruction_count=1,
                vlm_harness=ScriptedVLMHarness(["unknown"]))
            result = strategy.run()
        self.assertEqual(raw.actions, ["turn_left"] * 6)
        self.assertEqual(termination_category(result.end_reason),
                         "no_floor_bearing_candidate")

    def test_in_place_turn_keeps_the_true_incoming_direction(self):
        selector = ScriptedGateSelector(
            [None, DirectionGateEmptyError("left"),
             DirectionGateEmptyError("forward")], turn_steps=TURN_STEPS)
        with tempfile.TemporaryDirectory() as temporary_directory:
            executor = ForwardExecutor(temporary_directory)
            strategy, raw = build_strategy(
                temporary_directory, selector, sub_instruction_count=2,
                executor=executor,
                vlm_harness=ScriptedVLMHarness(["completed", "unknown"]))
            result = strategy.run(initial_action_heading=0.0)
        self.assertEqual(len(selector.requests), 3)
        arrival_heading = -math.radians(15 * TURN_STEPS)
        # Second request: after the forward hop, behind = arrival + pi.
        self.assertAlmostEqual(
            selector.requests[1].back_yaw,
            (arrival_heading + math.pi + math.pi) % (2 * math.pi) - math.pi)
        # Third request: the in-place left turn rotated the heading by +90
        # degrees, but the agent still came from the same direction.
        self.assertAlmostEqual(selector.requests[2].back_yaw,
                               selector.requests[1].back_yaw)
        self.assertAlmostEqual(selector.requests[2].yaw, 0.0)
        self.assertEqual(result.completed_sub_instructions, 1)

    def test_switch_off_restores_the_old_ending(self):
        selector = ScriptedGateSelector([DirectionGateEmptyError("right")])
        with tempfile.TemporaryDirectory() as temporary_directory:
            strategy, raw = build_strategy(
                temporary_directory, selector, sub_instruction_count=1,
                in_place_turn_on_empty_gate=False)
            result = strategy.run()
        self.assertEqual(raw.actions, [])
        self.assertEqual(result.records, [])
        self.assertEqual(termination_category(result.end_reason),
                         "no_floor_bearing_candidate")

    def test_turn_step_that_does_not_divide_the_gate_is_rejected_early(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(ValueError):
                build_strategy(
                    temporary_directory, ScriptedGateSelector([]),
                    sub_instruction_count=1, scan_step_deg=40.0)


class InstructionVLMPointSelectorConeTests(unittest.TestCase):
    def _captured_exclusion(self, **selector_kwargs):
        captured = {}

        def fake_choose_view(*args, **kwargs):
            captured.update(kwargs)
            return None, []

        selector = InstructionVLMPointSelector(
            segmenter=None, semantic_detector=None, vlm_harness=None,
            video_composer=None, views=8, policy_input_contract="rgb_only_v1",
            **selector_kwargs)
        request = PointSelectionRequest(
            sim=None, position=None, yaw=0.0, stage={"form": "TURN_LEFT"},
            target_index=0, rendered=[], motion_log=[], position_history=[],
            blocked_yaws=[0.5], policy_input_contract="rgb_only_v1")
        original = point_selectors.choose_view
        point_selectors.choose_view = fake_choose_view
        try:
            selector.select(request)
        finally:
            point_selectors.choose_view = original
        return captured["blocked_direction_exclusion"]

    def test_default_blocked_cone_is_half_the_eight_view_spacing(self):
        self.assertAlmostEqual(
            self._captured_exclusion(), math.radians(22.5))

    def test_legacy_fifty_degree_cone_remains_selectable(self):
        self.assertAlmostEqual(
            self._captured_exclusion(blocked_direction_exclusion_deg=50.0),
            math.radians(50.0))


if __name__ == "__main__":
    unittest.main()
