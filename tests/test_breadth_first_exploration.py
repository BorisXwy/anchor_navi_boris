#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from breadth_first_exploration import BreadthFirstFrontierMemory  # noqa: E402
from node_backtracking import (  # noqa: E402
    NodeBacktrackingController, _select_reachable_ground_point,
    choose_recovery_candidate,
    forward_route_breadcrumb,
    navmesh_first_route_waypoint,
)


class BreadthFirstExplorationTest(unittest.TestCase):
    def test_frontiers_pop_fifo_within_depth_and_retries_precede_deeper_work(self):
        memory = BreadthFirstFrontierMemory(dedup_radius_m=0.2)
        first = memory.enqueue("node_0", 0, [0, 0, -1])
        second = memory.enqueue("node_0", 0, [1, 0, -1])
        deeper = memory.enqueue("node_1", 1, [0, 0, -2])

        self.assertIs(memory.pop(), first)
        memory.requeue(first)
        self.assertIs(memory.pop(), second)
        self.assertIs(memory.pop(), first)
        self.assertIs(memory.pop(), deeper)
        self.assertEqual(
            [event["source_depth"] for event in memory.pop_events],
            [0, 0, 0, 1])
        self.assertTrue(all(
            event["was_queue_head"] for event in memory.pop_events))

    def test_frontier_memory_spatially_deduplicates_discoveries(self):
        memory = BreadthFirstFrontierMemory(dedup_radius_m=0.8)
        self.assertIsNotNone(memory.enqueue("node_0", 0, [0, 0, 0]))
        self.assertIsNone(memory.enqueue("node_1", 1, [0.4, 0, 0.4]))
        self.assertIsNotNone(memory.enqueue("node_1", 1, [1.0, 0, 0]))

    def test_forward_breadcrumb_replays_corner_in_forward_direction(self):
        edge = SimpleNamespace(
            action_history=[
                {"position_xyz": [0, 0, -2]},
                {"position_xyz": [-3, 0, -2]},
                {"position_xyz": [-3, 0, 0]},
            ], metadata={})
        result = forward_route_breadcrumb(
            edge, source_position=[0, 0, 0],
            target_position=[-3, 0, 0], current_position=[0, 0, 0],
            lookahead_m=1.5)
        self.assertAlmostEqual(result["position_xyz"][0], 0.0, places=4)
        self.assertAlmostEqual(result["position_xyz"][2], -1.5, places=4)
        self.assertEqual(result["route_direction"], "forward")

    def test_graph_route_uses_loop_closure_and_supports_forward_edges(self):
        nodes = [SimpleNamespace(node_id=f"node_{index}") for index in range(4)]
        edges = [
            SimpleNamespace(
                edge_id=f"edge_{index}", source_node_id=f"node_{index}",
                target_node_id=f"node_{index + 1}",
                traveled_distance_m=5.0, edge_kind="forward_navigation")
            for index in range(3)
        ]
        edges.append(SimpleNamespace(
            edge_id="closure", source_node_id="node_0",
            target_node_id="node_3", traveled_distance_m=0.0,
            edge_kind="node_revisit_loop_closure"))
        controller = NodeBacktrackingController.__new__(
            NodeBacktrackingController)
        controller.graph_memory = SimpleNamespace(nodes=nodes, edges=edges)
        controller.max_hops = 10
        route, edge_ids = controller._shortest_graph_route(
            "node_3", "node_1")
        self.assertEqual(route, ["node_3", "node_0", "node_1"])
        self.assertEqual(edge_ids, ["closure", "edge_0"])

    def test_backtrack_ground_filter_rejects_disconnected_navmesh_pixels(self):
        class DisconnectedPathfinder:
            @staticmethod
            def snap_point(point):
                return np.asarray(point, np.float32)

            @staticmethod
            def find_path(shortest):
                return False

        sim = SimpleNamespace(pathfinder=DisconnectedPathfinder())
        mask = np.zeros((120, 160), bool)
        mask[55:105, 15:145] = True
        depth = np.full(mask.shape, 2.0, np.float32)
        point, score, record = _select_reachable_ground_point(
            sim, mask, depth, [0, 0, 0], 0.0)
        self.assertIsNone(point)
        self.assertEqual(score, 0.0)
        self.assertFalse(record["selected_point_reachable"])

    def test_backtrack_ground_anchor_prefers_stored_node_vicinity(self):
        class ReachablePathfinder:
            @staticmethod
            def snap_point(point):
                return np.asarray(point, np.float32)

            @staticmethod
            def find_path(shortest):
                shortest.geodesic_distance = float(np.linalg.norm(
                    np.asarray(shortest.requested_end)[[0, 2]] -
                    np.asarray(shortest.requested_start)[[0, 2]]))
                return True

        sim = SimpleNamespace(pathfinder=ReachablePathfinder())
        mask = np.ones((40, 60), bool)
        depth = np.full(mask.shape, 2.0, np.float32)
        far = np.array([10.0, 20.0], np.float32)
        near = np.array([30.0, 20.0], np.float32)

        def project(point, unused_depth, unused_position, unused_yaw):
            if np.allclose(point, far):
                return np.array([0.0, 0.0, -5.0], np.float32), 5.0
            return np.array([0.0, 0.0, -1.0], np.float32), 1.0

        with patch("point_selectors.exploration_ground_points",
                   return_value=[(far, 100.0), (near, 10.0)]), patch(
                       "point_selectors.pixel_ground_to_world",
                       side_effect=project):
            point, score, record = _select_reachable_ground_point(
                sim, mask, depth, [0, 0, 0], 0.0,
                preferred_world_position=[0, 0, -1.1])
        np.testing.assert_allclose(point, near)
        self.assertEqual(score, 10.0)
        self.assertAlmostEqual(
            record["selected_point_target_planar_distance_m"], 0.1,
            places=5)

    def test_live_recovery_bearing_uses_first_navmesh_corner(self):
        class CornerPathfinder:
            @staticmethod
            def find_path(shortest):
                shortest.points = [
                    np.array([0.0, 0.0, 0.0], np.float32),
                    np.array([1.0, 0.0, 0.0], np.float32),
                    np.array([1.0, 0.0, -1.0], np.float32),
                ]
                shortest.geodesic_distance = 2.0
                return True

        sim = SimpleNamespace(pathfinder=CornerPathfinder())
        waypoint = navmesh_first_route_waypoint(
            sim, [0, 0, 0], [0, 0, -1])
        np.testing.assert_allclose(waypoint, [1, 0, 0])

    def test_short_range_recovery_prioritizes_node_vicinity_across_views(self):
        candidates = [
            {
                "view_index": 0, "hybrid_score": 0.92,
                "yaw_error_rad": 0.05,
                "selected_point_reachable": True,
                "selected_point_target_planar_distance_m": 1.70,
            },
            {
                "view_index": 4, "hybrid_score": 0.56,
                "yaw_error_rad": 0.65,
                "selected_point_reachable": True,
                "selected_point_target_planar_distance_m": 0.63,
            },
        ]
        selected, proximity_priority = choose_recovery_candidate(
            candidates, target_route_geodesic_m=0.88)
        self.assertEqual(selected["view_index"], 4)
        self.assertTrue(proximity_priority)

    def test_long_range_recovery_retains_directional_hybrid_ranking(self):
        candidates = [
            {
                "view_index": 0, "hybrid_score": 0.92,
                "yaw_error_rad": 0.05,
                "selected_point_reachable": True,
                "selected_point_target_planar_distance_m": 1.70,
            },
            {
                "view_index": 4, "hybrid_score": 0.56,
                "yaw_error_rad": 0.65,
                "selected_point_reachable": True,
                "selected_point_target_planar_distance_m": 0.63,
            },
        ]
        selected, proximity_priority = choose_recovery_candidate(
            candidates, target_route_geodesic_m=5.6)
        self.assertEqual(selected["view_index"], 0)
        self.assertFalse(proximity_priority)


if __name__ == "__main__":
    unittest.main()
