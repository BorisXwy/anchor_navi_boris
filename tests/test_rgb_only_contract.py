#!/usr/bin/env python3

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_rgb_only_contract import audit_run  # noqa: E402
from navigation_graph_memory import NavigationGraphMemory  # noqa: E402
from point_navigation_executor import (  # noqa: E402
    PointNavigationExecutor, PointNavigationRequest,
)
from rgb_only_runtime import RGBOnlyPolicySimulator  # noqa: E402


class FakeRawSimulator:
    def __init__(self):
        self.pathfinder = object()
        self.actions = []

    def get_sensor_observations(self):
        return {
            "rgb": np.zeros((12, 16, 4), np.uint8),
            "pano_rgb_1": np.ones((12, 16, 4), np.uint8),
            "depth": np.ones((12, 16), np.float32),
            "semantic": np.ones((12, 16), np.int32),
        }

    def step(self, action):
        self.actions.append(action)
        return self.get_sensor_observations()


class EmptySemanticExtractor:
    name = "rgb_only_test"

    def extract(self, six_views, six_depths=None, sub_instruction=None):
        if six_depths is not None:
            raise AssertionError("depth reached graph semantic extractor")
        return {"labels": [], "views": []}


class RGBOnlyContractTests(unittest.TestCase):
    def test_facade_exposes_rgb_and_discrete_actions_only(self):
        raw = FakeRawSimulator()
        events = []
        sim = RGBOnlyPolicySimulator(
            raw, _evaluation_hook=lambda action, _: events.append(action))
        observations = sim.get_sensor_observations()
        self.assertEqual(set(observations), {"rgb", "pano_rgb_1"})
        self.assertEqual(observations["rgb"].shape[-1], 3)
        sim.step("turn_left")
        self.assertEqual(raw.actions, ["turn_left"])
        self.assertEqual(events, ["turn_left"])
        for name in ("pathfinder", "get_agent", "depth", "position",
                     "collision", "geodesic_distance"):
            with self.assertRaises(RuntimeError):
                getattr(sim, name)
        with self.assertRaises(ValueError):
            sim.step("teleport")

    def test_strict_executor_rejects_any_privileged_request_field(self):
        executor = object.__new__(PointNavigationExecutor)
        executor.sim = RGBOnlyPolicySimulator(FakeRawSimulator())
        executor.tracking_cluster_profile = "rgb_only_dense_stop_v1"
        request = PointNavigationRequest(
            rgb=np.zeros((24, 32, 3), np.uint8),
            selected_point_xy=np.array([16, 18], np.float32),
            selectable_mask=np.ones((24, 32), bool), yaw=0.0,
            position_history=[], policy_input_contract="rgb_only_v1",
            selected_point_depth_m=2.0)
        with self.assertRaisesRegex(ValueError, "selected_point_depth_m"):
            executor.execute(request)

    def test_rgb_graph_node_has_no_pose_depth_or_absolute_heading(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            graph = NavigationGraphMemory(
                Path(temporary_directory),
                semantic_extractor=EmptySemanticExtractor())
            views = [np.zeros((12, 16, 3), np.uint8) for _ in range(6)]
            node = graph.add_origin_node(
                position_xyz=None, base_yaw_rad=None, global_step=0,
                six_views=views, six_depths=None)
            self.assertIsNone(node.position_xyz)
            self.assertIsNone(node.base_yaw_rad)
            self.assertTrue(all("absolute_yaw_rad" not in item
                                for item in node.six_views))

    def test_active_rgb_strategy_has_no_privileged_simulator_access(self):
        source = (ROOT / "scripts" / "rgb_only_instruction_sequence.py").read_text()
        tree = ast.parse(source)
        forbidden_attributes = {
            "pathfinder", "get_agent", "get_state", "depth", "position",
            "rotation", "navmesh", "geodesic_distance", "collision",
        }
        accessed = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        self.assertFalse(accessed & forbidden_attributes)
        imported = {
            alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertNotIn("habitat_sim", imported)

    def test_artifact_auditor_separates_hidden_evaluation_geometry(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            (output / "navigation_graph").mkdir()
            (output / "trajectory.json").write_text(json.dumps({
                "policy_input_contract": {"name": "rgb_only_v1"},
                "motion_log": [{"action": "move_forward"}],
                "targets": [{"selected_point_xy": [10, 20]}],
            }))
            (output / "navigation_graph" / "navigation_graph.json").write_text(json.dumps({
                "policy_input_contract": "rgb_only_v1",
                "nodes": [{"position_xyz": None, "base_yaw_rad": None}],
                "edges": [{"traveled_distance_m": 0.0}],
            }))
            (output / "evaluation_geometry.json").write_text(json.dumps({
                "position_xyz": [1, 2, 3], "depth_m": 2.0,
            }))
            self.assertTrue(audit_run(output)["passed"])


if __name__ == "__main__":
    unittest.main()
