#!/usr/bin/env python3
"""End-to-end unit tests of ``PointNavigationExecutor._execute_rgb_only``.

The fake simulator below only renders RGB and accepts the three discrete
actions, exactly like the production ``RGBOnlyPolicySimulator`` facade.  A
"wall" is modelled purely by returning the same frame after ``move_forward``:
the executor must infer the stall from consecutive low RGB motion, never from
a collision flag.
"""

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from point_navigation_executor import (  # noqa: E402
    PointNavigationExecutor, PointNavigationRequest,
    TRACKING_CLUSTER_PROFILES,
)
from rgb_only_runtime import RGBOnlyPolicySimulator  # noqa: E402


HEIGHT, WIDTH = 48, 64


def textured_frame(shift):
    """Deterministic textured RGB frame; ``shift`` rolls it vertically."""
    rng = np.random.RandomState(0)
    base = rng.randint(0, 255, size=(HEIGHT, WIDTH, 3)).astype(np.uint8)
    return np.roll(base, int(shift), axis=0)


class RGBOnlyFakeSimulator:
    """RGB-only fake: ``blocked_after`` forward commands stop changing the view."""

    def __init__(self, blocked_after=0):
        self.blocked_after = int(blocked_after)
        self.forward_count = 0
        self.turn_count = 0
        self.actions = []
        self.pathfinder = object()

    def _frame(self):
        moved = min(self.forward_count, self.blocked_after)
        rgb = textured_frame(3 * moved + 5 * self.turn_count)
        rgba = np.concatenate(
            [rgb, np.full((HEIGHT, WIDTH, 1), 255, np.uint8)], axis=2)
        return {"rgb": rgba}

    def get_sensor_observations(self):
        return self._frame()

    def step(self, action):
        self.actions.append(action)
        if action == "move_forward":
            self.forward_count += 1
        else:
            self.turn_count += 1
        return self._frame()


class StaticVisibleTracker:
    """Keeps every track at its seed pixel and always visible."""

    def reset(self, rgb, points):
        self.points = np.asarray(points, np.float32).copy()
        return self.points.copy(), np.ones(len(self.points), bool)

    def step(self, rgb):
        return self.points.copy(), np.ones(len(self.points), bool)


def straight_predict(model, config, context, goal, name, device, samples, seed):
    trajectories = np.tile(
        np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], np.float32),
        (samples, 1, 1))
    return 3.0, trajectories


def alternating_predict(state):
    """Return a predict_fn that alternates a sharp-left and a straight heading."""

    def predict(model, config, context, goal, name, device, samples, seed):
        state["calls"] = state.get("calls", 0) + 1
        if state["calls"] % 2:
            waypoints = [[0.0, 0.0], [0.2, 0.6], [0.4, 1.2]]
        else:
            waypoints = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
        trajectories = np.tile(
            np.array(waypoints, np.float32), (samples, 1, 1))
        return 3.0, trajectories

    return predict


def build_executor(raw_sim, max_steps, predict_fn=straight_predict,
                   profile_overrides=None):
    executor = PointNavigationExecutor(
        sim=RGBOnlyPolicySimulator(raw_sim),
        tracker=StaticVisibleTracker(),
        arrival_tracker=StaticVisibleTracker(),
        policy=None,
        policy_config={"image_size": [85, 64], "context_size": 5},
        policy_name="fake", predict_fn=predict_fn, device="cpu",
        output_dir=None, max_steps=max_steps,
        tracking_cluster_profile="rgb_only_dense_stop_v1")
    if profile_overrides:
        executor.cluster_config.update(profile_overrides)
    return executor


def build_request(raw_sim):
    rgb = np.asarray(raw_sim.get_sensor_observations()["rgb"])[..., :3]
    ground = np.zeros((HEIGHT, WIDTH), bool)
    ground[HEIGHT // 2:, :] = True
    return PointNavigationRequest(
        rgb=rgb, selected_point_xy=np.array([WIDTH / 2, HEIGHT - 6], np.float32),
        selectable_mask=ground, ground_mask=ground, yaw=0.0,
        position_history=[], instruction="walk to the point",
        target_index=0, global_step=0, policy_input_contract="rgb_only_v1")


class RGBOnlyExecutorStallTests(unittest.TestCase):
    def test_profile_declares_stall_rule(self):
        profile = TRACKING_CLUSTER_PROFILES["rgb_only_dense_stop_v1"]
        self.assertGreater(profile["stall_motion_threshold"], 0.0)
        self.assertGreater(profile["stall_forward_frames"], 0)

    def test_identical_frames_after_forward_end_hop_as_stall(self):
        raw = RGBOnlyFakeSimulator(blocked_after=0)
        executor = build_executor(raw, max_steps=40)
        result = executor.execute(build_request(raw))
        frames = int(executor.cluster_config["stall_forward_frames"])

        self.assertFalse(result.arrived)
        self.assertIsNone(result.signal)
        self.assertEqual(result.end_reason, "rgb_forward_stall")
        self.assertEqual(raw.actions, ["move_forward"] * frames)
        self.assertEqual(len(result.action_history), frames)
        self.assertEqual(result.record["terminal_stall_forward_streak"], frames)
        self.assertEqual(len(result.record["terminal_stall_motion_scores"]),
                         frames)
        self.assertEqual(
            [item["stall_forward_streak"] for item in result.action_history],
            list(range(1, frames + 1)))
        self.assertEqual(len(result.record["steps"]), frames)
        self.assertNotIn("position_xyz", result.action_history[0])

    def test_moving_frames_do_not_stall_and_exhaust_budget(self):
        raw = RGBOnlyFakeSimulator(blocked_after=10_000)
        executor = build_executor(raw, max_steps=6)
        result = executor.execute(build_request(raw))

        self.assertFalse(result.arrived)
        self.assertEqual(result.end_reason, "max_steps")
        self.assertEqual(len(raw.actions), 6)
        self.assertTrue(all(
            item["stall_forward_streak"] == 0
            for item in result.action_history))

    def test_stall_only_after_real_travel_stops(self):
        raw = RGBOnlyFakeSimulator(blocked_after=4)
        executor = build_executor(raw, max_steps=40)
        result = executor.execute(build_request(raw))
        frames = int(executor.cluster_config["stall_forward_frames"])

        self.assertEqual(result.end_reason, "rgb_forward_stall")
        self.assertEqual(len(raw.actions), 4 + frames)
        streaks = [item["stall_forward_streak"] for item in result.action_history]
        self.assertEqual(streaks[:4], [0, 0, 0, 0])
        self.assertEqual(streaks[4:], list(range(1, frames + 1)))

    def test_zero_stall_frames_disables_rule(self):
        raw = RGBOnlyFakeSimulator(blocked_after=0)
        executor = build_executor(
            raw, max_steps=7, profile_overrides={"stall_forward_frames": 0})
        result = executor.execute(build_request(raw))

        self.assertEqual(result.end_reason, "max_steps")
        self.assertEqual(len(raw.actions), 7)

    def test_turns_between_blocked_forwards_do_not_reset_streak(self):
        raw = RGBOnlyFakeSimulator(blocked_after=0)
        state = {}
        # The selected point sits on the image centre line, so the pixel term
        # is zero and the policy heading alone decides: the clipped sharp-left
        # heading (0.2 * 0.6 rad ~ 6.9 deg) exceeds a 5 deg deadband, the
        # straight heading does not.
        executor = build_executor(
            raw, max_steps=40, predict_fn=alternating_predict(state),
            profile_overrides={"turn_deadband_deg": 5.0})
        result = executor.execute(build_request(raw))
        frames = int(executor.cluster_config["stall_forward_frames"])

        self.assertEqual(result.end_reason, "rgb_forward_stall")
        forwards = [a for a in raw.actions if a == "move_forward"]
        turns = [a for a in raw.actions if a != "move_forward"]
        self.assertEqual(len(forwards), frames)
        self.assertGreater(len(turns), 0)
        turn_records = [item for item in result.action_history
                        if item["action"] != "move_forward"]
        # A turn leaves the streak untouched instead of clearing it.
        self.assertTrue(any(item["stall_forward_streak"] > 0
                            for item in turn_records))


if __name__ == "__main__":
    unittest.main()
