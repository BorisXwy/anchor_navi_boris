#!/usr/bin/env python3
"""Unit tests of the M4 walking-layer rules in ``_execute_rgb_only``.

Three rules are covered with an RGB-only fake simulator (a wall is a frame
that stops changing after ``move_forward``):

* arrival coast -- portal-crossing forms keep walking a few commands after
  the dense stop cluster leaves the frame, stopping on the first no-motion
  frame;
* stall recovery probe -- a forward stall first turns in place (alternating
  left/right) and resumes servoing; the hop only ends as a stall after every
  probe stalled again;
* reversal tagging -- forwards that produced no RGB motion carry a marker so
  action reversal does not replay them.
"""

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from point_navigation_executor import (  # noqa: E402
    ARRIVAL_COAST_FORMS, ARRIVAL_COAST_PHASE, REVERSAL_SKIP_KEY,
    STALL_RECOVERY_PROBE_PHASE, PointNavigationExecutor,
    PointNavigationRequest, TRACKING_CLUSTER_PROFILES,
)
from rgb_only_instruction_sequence import plan_action_reversal  # noqa: E402
from rgb_only_runtime import RGBOnlyPolicySimulator  # noqa: E402
from vlm_harness import HeuristicBackend, NavigationVLMHarness  # noqa: E402


HEIGHT, WIDTH = 48, 64
FRAMES = int(TRACKING_CLUSTER_PROFILES["rgb_only_dense_stop_v1"]
             ["stall_forward_frames"])
PROBE_TURNS = 2  # 30 degrees at the 15-degree turn step


def textured_frame(shift):
    rng = np.random.RandomState(0)
    base = rng.randint(0, 255, size=(HEIGHT, WIDTH, 3)).astype(np.uint8)
    return np.roll(base, int(shift), axis=0)


class WallFakeSimulator:
    """RGB-only fake.  Forwards beyond ``blocked_after`` stop changing the
    view until ``unblock_after_turns`` turns have been issued."""

    def __init__(self, blocked_after=10_000, unblock_after_turns=None):
        self.blocked_after = int(blocked_after)
        self.unblock_after_turns = unblock_after_turns
        self.forward_count = 0
        self.moved_count = 0
        self.turn_count = 0
        self.actions = []
        self.pathfinder = object()

    def _frame(self):
        rgb = textured_frame(3 * self.moved_count + 5 * self.turn_count)
        rgba = np.concatenate(
            [rgb, np.full((HEIGHT, WIDTH, 1), 255, np.uint8)], axis=2)
        return {"rgb": rgba}

    def get_sensor_observations(self):
        return self._frame()

    def step(self, action):
        self.actions.append(action)
        if action == "move_forward":
            self.forward_count += 1
            unblocked = (self.unblock_after_turns is not None and
                         self.turn_count >= self.unblock_after_turns)
            if self.forward_count <= self.blocked_after or unblocked:
                self.moved_count += 1
        else:
            self.turn_count += 1
        return self._frame()


class StaticVisibleTracker:
    def reset(self, rgb, points):
        self.points = np.asarray(points, np.float32).copy()
        return self.points.copy(), np.ones(len(self.points), bool)

    def step(self, rgb):
        return self.points.copy(), np.ones(len(self.points), bool)


class LosingTracker(StaticVisibleTracker):
    """Every track disappears after ``lose_after`` tracked frames, which the
    RGB-only arrival rule reads as the dense stop cluster leaving the frame."""

    def __init__(self, lose_after):
        self.lose_after = int(lose_after)
        self.frames = 0

    def step(self, rgb):
        self.frames += 1
        visible = np.full(len(self.points), self.frames < self.lose_after)
        return self.points.copy(), visible


def straight_predict(model, config, context, goal, name, device, samples, seed):
    trajectories = np.tile(
        np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], np.float32),
        (samples, 1, 1))
    return 3.0, trajectories


def build_executor(raw_sim, max_steps=40, tracker=None, arrival_tracker=None,
                   **kwargs):
    return PointNavigationExecutor(
        sim=RGBOnlyPolicySimulator(raw_sim),
        tracker=tracker or StaticVisibleTracker(),
        arrival_tracker=arrival_tracker or StaticVisibleTracker(),
        policy=None,
        policy_config={"image_size": [85, 64], "context_size": 5},
        policy_name="fake", predict_fn=straight_predict, device="cpu",
        output_dir=None, max_steps=max_steps,
        tracking_cluster_profile="rgb_only_dense_stop_v1", **kwargs)


def build_request(raw_sim, instruction_form=None):
    rgb = np.asarray(raw_sim.get_sensor_observations()["rgb"])[..., :3]
    ground = np.zeros((HEIGHT, WIDTH), bool)
    ground[HEIGHT // 2:, :] = True
    return PointNavigationRequest(
        rgb=rgb, selected_point_xy=np.array([WIDTH / 2, HEIGHT - 6], np.float32),
        selectable_mask=ground, ground_mask=ground, yaw=0.0,
        position_history=[], instruction="walk to the point",
        instruction_form=instruction_form,
        target_index=0, global_step=0, policy_input_contract="rgb_only_v1")


class ProfileTests(unittest.TestCase):
    def test_profile_declares_recovery_rules(self):
        profile = TRACKING_CLUSTER_PROFILES["rgb_only_dense_stop_v1"]
        self.assertEqual(profile["stall_recovery_max_probes"], 2)
        self.assertEqual(profile["stall_recovery_turn_deg"], 30.0)
        self.assertEqual(profile["arrival_coast_forward_steps"], 3)
        self.assertEqual(ARRIVAL_COAST_FORMS, {
            "EXIT_REGION", "ENTER_REGION", "TRAVERSE_PORTAL_REGION",
            "SELECT_PORTAL"})

    def test_constructor_overrides_profile_values(self):
        executor = build_executor(
            WallFakeSimulator(), arrival_coast_forward_steps=0,
            stall_recovery_max_probes=5)
        self.assertEqual(executor.cluster_config["arrival_coast_forward_steps"], 0)
        self.assertEqual(executor.cluster_config["stall_recovery_max_probes"], 5)
        untouched = build_executor(WallFakeSimulator())
        self.assertEqual(untouched.cluster_config["stall_recovery_max_probes"], 2)


class StallRecoveryProbeTests(unittest.TestCase):
    def test_wall_probes_left_then_right_then_ends_as_stall(self):
        raw = WallFakeSimulator(blocked_after=0)
        executor = build_executor(raw)
        result = executor.execute(build_request(raw))

        forwards = ["move_forward"] * FRAMES
        self.assertEqual(raw.actions, (
            forwards + ["turn_left"] * PROBE_TURNS +
            forwards + ["turn_right"] * PROBE_TURNS + forwards))
        self.assertFalse(result.arrived)
        self.assertEqual(result.end_reason, "rgb_forward_stall")
        self.assertEqual(result.record["terminal_stall_forward_streak"], FRAMES)
        self.assertEqual(result.record["extra_physical_actions"],
                         2 * PROBE_TURNS)
        self.assertEqual(result.next_global_step, len(raw.actions))

        probes = result.record["stall_recovery_probes"]
        self.assertEqual([probe["action"] for probe in probes],
                         ["turn_left", "turn_right"])
        self.assertEqual([probe["stalled_forwards_marked"] for probe in probes],
                         [FRAMES, FRAMES])
        # The policy loop itself only ran the forward commands.
        self.assertEqual(len(result.record["steps"]), 3 * FRAMES)

        history = result.action_history
        self.assertEqual(len(history), len(raw.actions))
        probe_records = [item for item in history
                         if item.get("phase") == STALL_RECOVERY_PROBE_PHASE]
        self.assertEqual(len(probe_records), 2 * PROBE_TURNS)
        np.testing.assert_allclose(
            [item["commanded_turn_deg"] for item in probe_records],
            [15.0, 15.0, -15.0, -15.0])
        # Every stalled forward is tagged, including the terminal streak.
        tagged = [item for item in history if item.get(REVERSAL_SKIP_KEY)]
        self.assertEqual(len(tagged), 3 * FRAMES)
        self.assertTrue(all(item["action"] == "move_forward"
                            and item[REVERSAL_SKIP_KEY] == "forward_stall"
                            for item in tagged))
        self.assertTrue(all(REVERSAL_SKIP_KEY not in item
                            for item in probe_records))
        # The action-frame heading returns to where it started.
        self.assertAlmostEqual(result.final_yaw, 0.0, places=9)

    def test_probe_that_frees_the_agent_resumes_servoing(self):
        raw = WallFakeSimulator(blocked_after=0, unblock_after_turns=PROBE_TURNS)
        executor = build_executor(raw, max_steps=10)
        result = executor.execute(build_request(raw))

        self.assertEqual(result.end_reason, "max_steps")
        self.assertEqual(raw.actions[:FRAMES + PROBE_TURNS],
                         ["move_forward"] * FRAMES + ["turn_left"] * PROBE_TURNS)
        self.assertEqual(len(result.record["stall_recovery_probes"]), 1)
        later_forwards = [item for item in result.action_history[FRAMES + PROBE_TURNS:]
                          if item["action"] == "move_forward"]
        self.assertGreater(len(later_forwards), 0)
        self.assertTrue(all(item["stall_forward_streak"] == 0
                            and REVERSAL_SKIP_KEY not in item
                            for item in later_forwards))
        tagged = [item for item in result.action_history
                  if item.get(REVERSAL_SKIP_KEY)]
        self.assertEqual(len(tagged), FRAMES)

    def test_single_probe_budget(self):
        raw = WallFakeSimulator(blocked_after=0)
        executor = build_executor(raw, stall_recovery_max_probes=1)
        result = executor.execute(build_request(raw))
        self.assertEqual(raw.actions, (
            ["move_forward"] * FRAMES + ["turn_left"] * PROBE_TURNS +
            ["move_forward"] * FRAMES))
        self.assertEqual(result.end_reason, "rgb_forward_stall")

    def test_zero_probes_restores_immediate_stall_stop(self):
        raw = WallFakeSimulator(blocked_after=0)
        executor = build_executor(raw, stall_recovery_max_probes=0)
        result = executor.execute(build_request(raw))
        self.assertEqual(raw.actions, ["move_forward"] * FRAMES)
        self.assertEqual(result.end_reason, "rgb_forward_stall")
        self.assertEqual(result.record["stall_recovery_probes"], [])
        self.assertEqual(
            [item.get(REVERSAL_SKIP_KEY) for item in result.action_history],
            ["forward_stall"] * FRAMES)

    def test_reversal_plan_skips_tagged_forwards_but_keeps_probe_turns(self):
        raw = WallFakeSimulator(blocked_after=0)
        result = build_executor(raw).execute(build_request(raw))
        plan = plan_action_reversal(result.action_history, 15.0)
        replay = [item["action"] for item in plan
                  if item["phase"] == "action_reversal_replay"]
        # Only the probe turns are replayed, mirrored and in reverse order.
        self.assertEqual(replay, ["turn_left"] * PROBE_TURNS +
                         ["turn_right"] * PROBE_TURNS)


class ArrivalCoastTests(unittest.TestCase):
    LOSE_AFTER = 5

    def run_hop(self, raw, instruction_form, **kwargs):
        executor = build_executor(
            raw, tracker=LosingTracker(self.LOSE_AFTER),
            arrival_tracker=LosingTracker(self.LOSE_AFTER), **kwargs)
        return executor, executor.execute(
            build_request(raw, instruction_form=instruction_form))

    def test_portal_form_coasts_after_cluster_loss(self):
        raw = WallFakeSimulator()
        executor, result = self.run_hop(raw, "TRAVERSE_PORTAL_REGION")
        coast_steps = int(executor.cluster_config["arrival_coast_forward_steps"])

        self.assertTrue(result.arrived)
        self.assertEqual(result.end_reason, "rgb_only_dense_stop_cluster_arrival")
        self.assertEqual(raw.actions,
                         ["move_forward"] * (self.LOSE_AFTER + coast_steps))
        coast = result.record["arrival_coast"]
        self.assertEqual(coast["instruction_form"], "TRAVERSE_PORTAL_REGION")
        self.assertEqual(coast["requested_steps"], coast_steps)
        self.assertEqual(coast["executed_steps"], coast_steps)
        self.assertFalse(coast["stopped_by_stall"])
        self.assertEqual(result.record["extra_physical_actions"], coast_steps)
        coast_records = result.action_history[-coast_steps:]
        self.assertTrue(all(item["phase"] == ARRIVAL_COAST_PHASE and
                            item["action"] == "move_forward" and
                            REVERSAL_SKIP_KEY not in item
                            for item in coast_records))
        self.assertEqual(len(result.action_history),
                         self.LOSE_AFTER + coast_steps)
        # Coast frames carry no tracks and therefore never enter ``steps``.
        self.assertEqual(len(result.record["steps"]), self.LOSE_AFTER)
        self.assertEqual(result.next_global_step, len(raw.actions))
        self.assertEqual(len(result.edge_keyframes), 5)

    def test_coast_stops_on_first_no_motion_frame_and_tags_it(self):
        raw = WallFakeSimulator(blocked_after=self.LOSE_AFTER)
        executor, result = self.run_hop(raw, "EXIT_REGION")

        self.assertTrue(result.arrived)
        coast = result.record["arrival_coast"]
        self.assertEqual(coast["executed_steps"], 1)
        self.assertTrue(coast["stopped_by_stall"])
        self.assertEqual(result.action_history[-1][REVERSAL_SKIP_KEY],
                         "arrival_coast_stall")
        self.assertEqual(len(raw.actions), self.LOSE_AFTER + 1)

    def test_non_portal_forms_do_not_coast(self):
        for form in ("PASS_LANDMARK", "STOP_WAIT", None):
            raw = WallFakeSimulator()
            executor, result = self.run_hop(raw, form)
            self.assertTrue(result.arrived, form)
            self.assertEqual(len(raw.actions), self.LOSE_AFTER, form)
            self.assertEqual(result.record["arrival_coast"]["requested_steps"],
                             0, form)
            self.assertEqual(result.record["arrival_coast"]["executed_steps"],
                             0, form)

    def test_zero_coast_steps_disables_rule(self):
        raw = WallFakeSimulator()
        executor, result = self.run_hop(
            raw, "ENTER_REGION", arrival_coast_forward_steps=0)
        self.assertTrue(result.arrived)
        self.assertEqual(len(raw.actions), self.LOSE_AFTER)

    def test_instruction_form_is_not_a_privileged_field(self):
        raw = WallFakeSimulator()
        request = build_request(raw, instruction_form="ENTER_REGION")
        request.selected_point_depth_m = 1.0
        with self.assertRaises(ValueError):
            build_executor(raw).execute(request)


class JudgeActionSummaryTests(unittest.TestCase):
    def test_probe_turns_do_not_count_as_instruction_turns(self):
        harness = NavigationVLMHarness(HeuristicBackend(), None)
        history = [
            {"step": 0, "action": "move_forward", "commanded_turn_deg": 0.0},
            {"step": 1, "action": "turn_left", "commanded_turn_deg": 15.0},
            {"step": 2, "action": "turn_left", "commanded_turn_deg": 15.0,
             "phase": STALL_RECOVERY_PROBE_PHASE},
            {"step": 2, "action": "turn_left", "commanded_turn_deg": 15.0,
             "phase": STALL_RECOVERY_PROBE_PHASE},
            {"step": 3, "action": "turn_right", "commanded_turn_deg": -15.0},
        ]
        views = [np.zeros((12, 16, 3), np.uint8)] * 6
        result = harness.judge_edge_instruction_completion_rgb_only(
            {"sub_instruction_id": 0, "navigation_instruction": "turn left",
             "form": "TURN_LEFT"}, None, "node_0000", "node_0001",
            views, views, history, [np.zeros((12, 16, 3), np.uint8)])
        summary = result["motion_evidence"]
        self.assertEqual(summary["control_steps"], 5)
        self.assertEqual(summary["left_turn_command_count"], 1)
        self.assertEqual(summary["right_turn_command_count"], 1)
        self.assertEqual(summary["stall_recovery_turn_command_count"], 2)


if __name__ == "__main__":
    unittest.main()
