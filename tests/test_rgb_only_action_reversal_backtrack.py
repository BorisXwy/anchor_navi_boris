#!/usr/bin/env python3
"""Action-reversal backtracking: replay a hop backwards, let the VLM confirm."""

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
from evaluate_point_navigation import termination_category  # noqa: E402
from instruction_decomposer import SubInstruction  # noqa: E402
from navigation_graph_memory import NavigationGraphMemory  # noqa: E402
from point_navigation_executor import (  # noqa: E402
    REVERSAL_SKIP_KEY, PointNavigationResult,
)
from point_selectors import PointSelectionResult  # noqa: E402
from rgb_only_instruction_sequence import (  # noqa: E402
    ACTION_REVERSAL_REPLAY_PHASE, ACTION_REVERSAL_RESTORE_HEADING_PHASE,
    ACTION_REVERSAL_TURN_AROUND_PHASE, RGBOnlyActionReversalBacktracker,
    RGBOnlyBacktrackResult, RGBOnlyInstructionSequenceExplorationStrategy,
    count_reversal_skips, plan_action_reversal, turn_around_action_count,
)
from rgb_only_runtime import RGBOnlyPolicySimulator  # noqa: E402
from vlm_harness import (  # noqa: E402
    HeuristicBackend, NavigationVLMHarness, VLMProviderFatalError,
)

TURN_STEP_DEG = 15.0
HALF_TURN = 12


def action_records(letters):
    mapping = {"F": ("move_forward", 0.0), "L": ("turn_left", TURN_STEP_DEG),
               "R": ("turn_right", -TURN_STEP_DEG)}
    return [{"step": index, "action": mapping[letter][0],
             "commanded_turn_deg": mapping[letter][1],
             "forward_commanded": letter == "F",
             "policy_input_contract": "rgb_only_v1"}
            for index, letter in enumerate(letters)]


class GridPoseFakeRawSimulator:
    """Fake Habitat handle that tracks an integer pose for test assertions only.

    The pose lives purely inside the fake; the strategy only ever calls
    ``step``/``get_sensor_observations`` through the RGB-only facade.
    """

    def __init__(self):
        self.pathfinder = object()
        self.actions = []
        self.grid_xy = [0, 0]
        self.heading_units = 0  # multiples of the 15-degree turn step
        self.events = []

    def get_sensor_observations(self):
        shade = min(255, 10 + 5 * len(self.actions))
        views = {"rgb": np.full((12, 16, 4), shade, np.uint8)}
        for index in range(1, 6):
            views[f"pano_rgb_{index}"] = np.zeros((12, 16, 4), np.uint8)
        for index in range(1, 8):
            views[f"completion_rgb_{index}"] = np.zeros((12, 16, 4), np.uint8)
        return views

    def step(self, action):
        self.actions.append(action)
        if action == "turn_left":
            self.heading_units = (self.heading_units + 1) % (2 * HALF_TURN)
        elif action == "turn_right":
            self.heading_units = (self.heading_units - 1) % (2 * HALF_TURN)
        elif action == "move_forward":
            angle = math.radians(TURN_STEP_DEG * self.heading_units)
            # Scale so every heading maps to a distinct integer displacement.
            self.grid_xy[0] += int(round(1000 * math.cos(angle)))
            self.grid_xy[1] += int(round(1000 * math.sin(angle)))
        return self.get_sensor_observations()


class EmptySemanticExtractor:
    name = "rgb_only_action_reversal_test"

    def extract(self, six_views, six_depths=None, sub_instruction=None):
        return {"labels": [], "views": []}


class UntouchableDependency:
    def __getattr__(self, name):
        raise AssertionError(f"dependency accessed unexpectedly: {name}")


class ScriptedPointSelector:
    """Turns ``turn_letters`` toward the target and returns the front view."""

    def __init__(self, turn_letters=""):
        self.turn_letters = turn_letters
        self.requests = []

    def select(self, request):
        self.requests.append(request)
        yaw = request.yaw
        for step, letter in enumerate(self.turn_letters, start=1):
            action = "turn_left" if letter == "L" else "turn_right"
            commanded = TURN_STEP_DEG if letter == "L" else -TURN_STEP_DEG
            request.sim.step(action)
            yaw += math.radians(commanded)
            request.motion_log.append({
                "phase": "turn_to_selected_target",
                "target_index": request.target_index, "turn_frame": step,
                "action": action, "commanded_turn_deg": commanded,
                "policy_input_contract": "rgb_only_v1"})
        rgb = request.sim.get_sensor_observations()["rgb"]
        mask = np.ones(rgb.shape[:2], bool)
        return PointSelectionResult(chosen={
            "rgb": rgb, "yaw": yaw,
            "relative_yaw_rad": yaw - request.yaw,
            "point": np.array([8.0, 9.0], np.float32),
            "target_mask": mask, "mask": mask, "vlm_selection": {},
        }, candidates=[])


class ScriptedExecutor:
    """Drives the fake simulator with fixed letters; ``arrivals`` per hop."""

    def __init__(self, output_dir, letters_per_hop, arrivals):
        self.keyframes_dir = Path(output_dir) / "edge_keyframes"
        self.keyframes_dir.mkdir(parents=True, exist_ok=True)
        self.letters_per_hop = list(letters_per_hop)
        self.arrivals = list(arrivals)
        self.requests = []

    def execute(self, request):
        hop = len(self.requests)
        self.requests.append(request)
        letters = self.letters_per_hop[hop]
        actions = action_records(letters)
        yaw = float(request.yaw)
        frames = [np.asarray(request.rgb, np.uint8)[..., :3]]
        for record in actions:
            observations = self.sim.step(record["action"])
            yaw += math.radians(record["commanded_turn_deg"])
            frames.append(np.asarray(observations["rgb"], np.uint8)[..., :3])
        keyframe_records = []
        for order, frame in enumerate(frames):
            path = self.keyframes_dir / (
                f"target_{request.target_index:02d}_keyframe_{order:02d}.jpg")
            Image.fromarray(frame).save(path)
            keyframe_records.append({
                "step": order - 1,
                "phase": "edge_start" if order == 0 else "edge_motion",
                "action": None if order == 0 else actions[order - 1]["action"],
                "policy_input_contract": "rgb_only_v1",
                "keyframe_index": order, "source_frame_index": order,
                "image_path": str(path.relative_to(self.keyframes_dir.parent)),
            })
        arrived = bool(self.arrivals[hop])
        signal = "rgb_only_dense_stop_cluster_arrival" if arrived else None
        end_reason = signal if arrived else "max_steps"
        return PointNavigationResult(
            arrived=arrived, signal=signal, end_reason=end_reason,
            final_rgb=frames[-1], final_yaw=yaw,
            next_global_step=request.global_step + len(actions),
            action_history=actions,
            record={"action_history": actions, "steps": [],
                    "edge_keyframes": keyframe_records,
                    "arrival_signal": signal, "end_reason": end_reason},
            edge_keyframes=frames)


class ScriptedVLMHarness:
    def __init__(self, completion_statuses=("completed",), same_place=True):
        self.completion_statuses = list(completion_statuses)
        self.same_place = same_place
        self.judge_calls = []
        self.revisit_calls = []

    def judge_edge_instruction_completion_rgb_only(self, **kwargs):
        index = min(len(self.judge_calls), len(self.completion_statuses) - 1)
        self.judge_calls.append(kwargs)
        return {"status": self.completion_statuses[index], "confidence": 0.9,
                "reason": "scripted", "visual_evidence": "scripted",
                "temporal_evidence": {}}

    def judge_node_revisit_rgb_only(self, **kwargs):
        self.revisit_calls.append(kwargs)
        if isinstance(self.same_place, Exception):
            raise self.same_place
        return {"same_place": bool(self.same_place), "confidence": 0.8,
                "reason": "scripted", "visual_evidence": "scripted"}


def build_strategy(temporary_directory, point_selector, executor, harness,
                   sub_instruction_count=1, backtrack_method="action-reversal",
                   max_exploration_hops=6):
    raw = GridPoseFakeRawSimulator()
    sim = RGBOnlyPolicySimulator(
        raw, _evaluation_event_hook=(
            lambda event, payload, _sim: raw.events.append((event, payload))))
    executor.sim = sim
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
            "navigation_instruction": text, "form": form})
        for index, (text, form) in enumerate([
            ("turn right", "TURN_RIGHT"),
            ("wait by the glass panes", "STOP_WAIT"),
        ][:sub_instruction_count])]
    strategy = RGBOnlyInstructionSequenceExplorationStrategy(
        sim=sim, sub_instructions=sub_instructions,
        point_selector=point_selector, point_navigation_executor=executor,
        graph_memory=graph, vlm_harness=harness,
        segmenter=UntouchableDependency(), output_dir=temporary_directory,
        max_exploration_hops=max_exploration_hops,
        backtrack_method=backtrack_method, turn_step_deg=TURN_STEP_DEG)
    return strategy, raw


class ActionReversalPlanTests(unittest.TestCase):
    def test_user_example_is_reversed_with_turns_swapped(self):
        plan = plan_action_reversal(action_records("FFRFLF"), TURN_STEP_DEG)
        actions = [item["action"] for item in plan]
        self.assertEqual(actions, (
            ["turn_left"] * HALF_TURN +
            ["move_forward", "turn_right", "move_forward", "turn_left",
             "move_forward", "move_forward"] +
            ["turn_left"] * HALF_TURN))
        self.assertEqual([item["phase"] for item in plan], (
            [ACTION_REVERSAL_TURN_AROUND_PHASE] * HALF_TURN +
            [ACTION_REVERSAL_REPLAY_PHASE] * 6 +
            [ACTION_REVERSAL_RESTORE_HEADING_PHASE] * HALF_TURN))
        self.assertEqual([item["step"] for item in plan],
                         list(range(len(plan))))
        for item in plan:
            expected = {"turn_left": TURN_STEP_DEG,
                        "turn_right": -TURN_STEP_DEG,
                        "move_forward": 0.0}[item["action"]]
            self.assertEqual(item["commanded_turn_deg"], expected)
            self.assertEqual(item["forward_commanded"],
                             item["action"] == "move_forward")
            self.assertEqual(item["policy_input_contract"], "rgb_only_v1")
        self.assertAlmostEqual(
            sum(item["commanded_turn_deg"] for item in plan), 360.0)

    def test_empty_history_only_turns_around_twice(self):
        plan = plan_action_reversal([], TURN_STEP_DEG)
        self.assertEqual([item["action"] for item in plan],
                         ["turn_left"] * (2 * HALF_TURN))

    def test_turn_step_must_divide_half_rotation(self):
        self.assertEqual(turn_around_action_count(15.0), 12)
        self.assertEqual(turn_around_action_count(30.0), 6)
        with self.assertRaises(ValueError):
            plan_action_reversal(action_records("F"), 25.0)
        with self.assertRaises(ValueError):
            plan_action_reversal([{"action": "stop"}], TURN_STEP_DEG)

    def test_replay_returns_fake_pose_to_origin(self):
        raw = GridPoseFakeRawSimulator()
        for letters in ("FFRFLF", "LLLFFFRRFLFFFF", "RRRRRRFFFLLLF", "F"):
            raw.grid_xy = [0, 0]
            raw.heading_units = 3
            history = action_records(letters)
            for record in history:
                raw.step(record["action"])
            self.assertNotEqual(raw.grid_xy, [0, 0])
            for record in plan_action_reversal(history, TURN_STEP_DEG):
                raw.step(record["action"])
            self.assertEqual(raw.grid_xy, [0, 0], letters)
            self.assertEqual(raw.heading_units, 3, letters)

    def test_tagged_no_motion_forwards_are_not_replayed(self):
        # Lower-case letters are forwards the executor tagged as stalled:
        # the fake wall swallows them on the way out, so replaying them on
        # the way back would overshoot the origin.
        raw = GridPoseFakeRawSimulator()
        for letters in ("FFfffRFLF", "LLFfffRRfff", "fff", "FRfLf"):
            raw.grid_xy = [0, 0]
            raw.heading_units = 3
            history = action_records(letters.upper())
            for record, letter in zip(history, letters):
                if letter == "f":
                    record["stall_forward_streak"] = 1
                    record[REVERSAL_SKIP_KEY] = "forward_stall"
                else:
                    raw.step(record["action"])
            plan = plan_action_reversal(history, TURN_STEP_DEG)
            replay = [item["action"] for item in plan
                      if item["phase"] == ACTION_REVERSAL_REPLAY_PHASE]
            self.assertEqual(len(replay), len(letters) - letters.count("f"))
            for record in plan:
                raw.step(record["action"])
            self.assertEqual(raw.grid_xy, [0, 0], letters)
            self.assertEqual(raw.heading_units, 3, letters)
        self.assertEqual(count_reversal_skips(history), 2)


class ActionReversalStrategyTests(unittest.TestCase):
    def test_failed_hop_replays_back_blocks_direction_and_continues(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            executor = ScriptedExecutor(
                temporary_directory, ["FFRFLF", "FF"], [False, True])
            harness = ScriptedVLMHarness(same_place=True)
            selector = ScriptedPointSelector("RR")
            strategy, raw = build_strategy(
                temporary_directory, selector, executor, harness)
            origin_node_id = strategy.state.last_verified_node_id
            result = strategy.run(initial_action_heading=0.3,
                                  initial_global_step=5)

            self.assertTrue(result.success)
            self.assertEqual(result.end_reason, "instruction_sequence_complete")
            self.assertEqual(len(executor.requests), 2)
            failed, arrived = result.records[0], result.records[1]
            self.assertFalse(failed["arrived"])
            self.assertEqual(failed["backtrack_method"], "action-reversal")
            recovery = failed["physical_failure_recovery"]
            self.assertTrue(recovery["success"])
            self.assertEqual(recovery["target_node_id"], origin_node_id)
            self.assertEqual(recovery["recovered_node_id"], origin_node_id)
            self.assertEqual(recovery["end_reason"],
                             "action_reversal_return_confirmed")
            attempt = recovery["attempts"][0]
            self.assertEqual(attempt["backtrack_method"], "action-reversal")
            self.assertEqual(attempt["turn_around_actions"], HALF_TURN)
            self.assertEqual(attempt["forward_step_count"], 8)
            self.assertTrue(attempt["vlm_node_revisit"]["same_place"])
            self.assertIn("target_panorama_similarity_after", attempt)

            # Selector turned RR, executor walked FFRFLF; the replay undoes
            # both and the fake pose is back at the origin before hop two.
            forward = ["turn_right", "turn_right"] + [
                item["action"] for item in action_records("FFRFLF")]
            reversal = [item["action"] for item in
                        plan_action_reversal(action_records("RRFFRFLF"),
                                             TURN_STEP_DEG)]
            self.assertEqual(raw.actions[:len(forward) + len(reversal)],
                             forward + reversal)
            self.assertEqual(attempt["action_history"][HALF_TURN]["action"],
                             "move_forward")
            self.assertEqual(
                [item["action"] for item in attempt["action_history"]],
                reversal)

            # The second hop starts from the restored heading and the
            # blocked ray is the one the first selection chose.
            self.assertAlmostEqual(selector.requests[1].yaw, 0.3)
            self.assertEqual(
                strategy.state.blocked_yaws_by_verified_node[origin_node_id],
                [failed["selected_action_heading_rad"]])
            self.assertEqual(selector.requests[1].back_yaw, None)
            self.assertEqual(selector.requests[1].previous_action_history, [])
            self.assertEqual(len(strategy.graph_memory.edges), 1)
            self.assertEqual(len(strategy.graph_memory.nodes), 2)
            self.assertEqual(len(harness.revisit_calls), 1)
            self.assertEqual(len(harness.revisit_calls[0]["target_node_views"]),
                             6)
            self.assertEqual(len(harness.judge_calls), 1)
            self.assertEqual(result.recovery_records[0]["trigger"],
                             "physical_failure")
            events = [event for event, _ in raw.events]
            self.assertIn("action_reversal_backtrack_started", events)
            self.assertIn("action_reversal_backtrack_finished", events)
            finished = [payload for event, payload in raw.events
                        if event == "action_reversal_backtrack_finished"][0]
            self.assertTrue(finished["same_place"])
            self.assertEqual(termination_category(result.end_reason),
                             "instruction_sequence_complete")

    def test_vlm_rejection_ends_episode(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            executor = ScriptedExecutor(temporary_directory, ["FFF"], [False])
            harness = ScriptedVLMHarness(same_place=False)
            strategy, raw = build_strategy(
                temporary_directory, ScriptedPointSelector(), executor, harness)
            result = strategy.run()
            self.assertFalse(result.success)
            self.assertEqual(result.end_reason,
                             "rgb_only_physical_failure_recovery_failed")
            recovery = result.records[0]["physical_failure_recovery"]
            self.assertFalse(recovery["success"])
            self.assertEqual(recovery["end_reason"],
                             "action_reversal_return_rejected_by_vlm")
            self.assertIsNone(recovery["recovered_node_id"])
            self.assertEqual(len(strategy.graph_memory.nodes), 1)
            self.assertEqual(raw.grid_xy, [0, 0])
            self.assertEqual(termination_category(result.end_reason),
                             "rgb_only_physical_failure_recovery_failed")

    def test_vlm_runtime_error_is_recorded_and_fatal_error_propagates(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            executor = ScriptedExecutor(temporary_directory, ["FFF"], [False])
            harness = ScriptedVLMHarness(
                same_place=RuntimeError("VLM harness exhausted retries"))
            strategy, _ = build_strategy(
                temporary_directory, ScriptedPointSelector(), executor, harness)
            result = strategy.run()
            self.assertFalse(result.success)
            recovery = result.records[0]["physical_failure_recovery"]
            self.assertEqual(recovery["end_reason"],
                             "action_reversal_vlm_error")
            self.assertIn("exhausted", recovery["attempts"][0]["selection_error"])
        with tempfile.TemporaryDirectory() as temporary_directory:
            executor = ScriptedExecutor(temporary_directory, ["FFF"], [False])
            harness = ScriptedVLMHarness(
                same_place=VLMProviderFatalError("HTTP 401"))
            strategy, _ = build_strategy(
                temporary_directory, ScriptedPointSelector(), executor, harness)
            with self.assertRaises(VLMProviderFatalError):
                strategy.run()

    def test_backtrack_and_block_records_revisit_node_and_loop_closure(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            executor = ScriptedExecutor(
                temporary_directory, ["FF", "FRF", "FF"], [True, True, True])
            harness = ScriptedVLMHarness(
                completion_statuses=("unknown", "unknown", "completed"),
                same_place=True)
            strategy, raw = build_strategy(
                temporary_directory, ScriptedPointSelector(), executor, harness)
            result = strategy.run()

            self.assertTrue(result.success)
            self.assertEqual(result.end_reason, "instruction_sequence_complete")
            second = result.records[1]
            self.assertEqual(second["sequence_directive"]["action"],
                             "backtrack_and_block")
            recovery = second["sequence_recovery_backtrack"]
            self.assertTrue(recovery["success"])
            self.assertEqual(recovery["end_reason"],
                             "action_reversal_return_confirmed_node_recorded")
            node_b = result.records[0]["navigation_graph_node_id"]
            node_c = second["navigation_graph_node_id"]
            self.assertEqual(recovery["target_node_id"], node_b)
            revisit_id = recovery["recovered_node_id"]
            self.assertNotEqual(revisit_id, node_b)
            # on_backtrack re-anchored the state machine on the revisit node
            # and carried the blocked B->C ray onto it before hop three.
            self.assertEqual(
                strategy.state.blocked_yaws_by_verified_node[revisit_id],
                [second["selected_action_heading_rad"]])
            self.assertEqual(strategy.state.recovery_backtracks, 1)
            self.assertEqual(
                result.records[2]["navigation_graph_edge_id"],
                [edge.edge_id for edge in strategy.graph_memory.edges
                 if edge.source_node_id == revisit_id][0])

            graph = strategy.graph_memory
            kinds = [(edge.source_node_id, edge.target_node_id, edge.edge_kind)
                     for edge in graph.edges]
            self.assertIn((node_c, revisit_id,
                           "rgb_only_action_reversal_backtrack"), kinds)
            self.assertIn((node_b, revisit_id, "node_revisit_loop_closure"),
                          kinds)
            revisit_edge = [edge for edge in graph.edges
                            if edge.edge_kind ==
                            "rgb_only_action_reversal_backtrack"][0]
            self.assertEqual(
                [item["action"] for item in revisit_edge.action_history],
                [item["action"] for item in plan_action_reversal(
                    action_records("FRF"), TURN_STEP_DEG)])
            revisit_node = graph.get_node(revisit_id)
            self.assertIsNone(revisit_node.position_xyz)
            self.assertEqual(revisit_node.metadata["backtrack_method"],
                             "action-reversal")
            self.assertEqual(revisit_node.metadata["rgb_revisit_target_node_id"],
                             node_b)
            payload = json.loads(graph.graph_path.read_text())
            self.assertEqual(_violations(payload["edges"]), [])
            self.assertEqual(_violations(payload["nodes"]), [])
            # The replay walked back exactly along B->C.
            started = [payload for event, payload in raw.events
                       if event == "action_reversal_backtrack_started"]
            self.assertEqual(len(started), 1)
            self.assertEqual(started[0]["target_node_id"], node_b)
            self.assertEqual(started[0]["forward_step_count"], 3)
            self.assertEqual(result.recovery_records[0]["trigger"],
                             "backtrack_and_block")
            self.assertEqual(len(executor.requests), 3)

    def test_visual_method_still_uses_graph_backtracker(self):
        class RecordingBacktracker:
            def __init__(self):
                self.calls = []

            def recover(self, target_node_id, heading, global_step,
                        target_index, failed_action_history):
                self.calls.append(target_node_id)
                return RGBOnlyBacktrackResult(
                    success=True, target_node_id=target_node_id,
                    recovered_node_id=target_node_id,
                    final_action_heading_rad=float(heading),
                    next_global_step=int(global_step), attempts=[],
                    end_reason="rgb_only_backtrack_already_at_node")

        with tempfile.TemporaryDirectory() as temporary_directory:
            executor = ScriptedExecutor(
                temporary_directory, ["FF", "FF"], [False, True])
            harness = ScriptedVLMHarness(same_place=True)
            strategy, raw = build_strategy(
                temporary_directory, ScriptedPointSelector(), executor,
                harness, backtrack_method="visual")
            backtracker = RecordingBacktracker()
            strategy.backtracker = backtracker
            origin_node_id = strategy.state.last_verified_node_id
            result = strategy.run()
            self.assertTrue(result.success)
            self.assertEqual(backtracker.calls, [origin_node_id])
            self.assertEqual(harness.revisit_calls, [])
            self.assertEqual(result.records[0]["backtrack_method"], "visual")
            self.assertNotIn("turn_left", raw.actions)

    def test_unknown_backtrack_method_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(ValueError):
                build_strategy(
                    temporary_directory, ScriptedPointSelector(),
                    ScriptedExecutor(temporary_directory, [], []),
                    ScriptedVLMHarness(), backtrack_method="teleport")


class NodeRevisitHarnessTests(unittest.TestCase):
    @staticmethod
    def views(shade):
        return [np.full((48, 64, 3), shade, np.uint8) for _ in range(6)]

    def test_heuristic_backend_confirms_revisit_through_harness(self):
        class RecordingHeuristicBackend(HeuristicBackend):
            def __init__(self):
                self.calls = []

            def generate_json(self, prompt, images, schema):
                self.calls.append({"prompt": prompt, "images": images,
                                   "schema": schema})
                return super().generate_json(prompt, images, schema)

        backend = RecordingHeuristicBackend()
        harness = NavigationVLMHarness(backend, retries=0)
        plan = plan_action_reversal(action_records("FRF"), TURN_STEP_DEG)
        verdict = harness.judge_node_revisit_rgb_only(
            target_node_id="node_0003", target_node_views=self.views(40),
            current_views=self.views(90), reversal_action_history=plan)
        self.assertTrue(verdict["same_place"])
        self.assertEqual(verdict["target_node_id"], "node_0003")
        self.assertEqual(verdict["prompt_version"],
                         "v1_two_panorama_same_place")
        self.assertEqual(verdict["privileged_inputs_used"], [])
        call = backend.calls[0]
        self.assertIn("RGB_ONLY_NODE_REVISIT_CONFIRMATION", call["prompt"])
        self.assertIn("node_0003", call["prompt"])
        self.assertEqual(len(call["images"]), 1)
        self.assertEqual(call["schema"]["required"],
                         ["same_place", "confidence", "reason",
                          "visual_evidence"])
        self.assertNotIn("position", call["prompt"].lower().replace(
            "pose, depth", ""))
        self.assertEqual(harness.calls[-1]["task"],
                         "judge_node_revisit_rgb_only")

    def test_harness_rejects_incomplete_panoramas_and_bad_answers(self):
        harness = NavigationVLMHarness(HeuristicBackend(), retries=0)
        with self.assertRaises(ValueError):
            harness.judge_node_revisit_rgb_only(
                target_node_id="node_0001", target_node_views=self.views(1)[:5],
                current_views=self.views(2))

        class WrongTypeBackend(HeuristicBackend):
            def generate_json(self, prompt, images, schema):
                return {"same_place": "yes", "confidence": 0.5,
                        "reason": "", "visual_evidence": ""}

        harness = NavigationVLMHarness(WrongTypeBackend(), retries=0)
        with self.assertRaises(RuntimeError):
            harness.judge_node_revisit_rgb_only(
                target_node_id="node_0001", target_node_views=self.views(1),
                current_views=self.views(2))


if __name__ == "__main__":
    unittest.main()
