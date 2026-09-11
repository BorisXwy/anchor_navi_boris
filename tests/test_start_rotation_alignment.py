#!/usr/bin/env python3
"""Unit tests for the opening-turn / GT-path start_rotation alignment rule."""

import copy
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import start_rotation_alignment as alignment  # noqa: E402


def episode(instruction, reference_path, start_rotation=(0.0, 0.0, 0.0, 1.0)):
    return {
        "episode_id": 1,
        "trajectory_id": 2,
        "scene_id": "mp3d/x/x.glb",
        "start_position": list(reference_path[0]),
        "start_rotation": list(start_rotation),
        "info": {"geodesic_distance": 3.0},
        "goals": [{"position": list(reference_path[-1]), "radius": 3.0}],
        "instruction": {"instruction_text": instruction, "instruction_tokens": [1, 2]},
        "reference_path": [list(point) for point in reference_path],
    }


class QuaternionConventionTest(unittest.TestCase):
    def test_yaw_round_trip(self):
        for degrees in (-179.0, -90.0, -30.0, 0.0, 45.0, 90.0, 179.0):
            yaw = math.radians(degrees)
            coeffs = alignment.quaternion_coeffs_from_yaw(yaw)
            self.assertEqual(coeffs[0], 0.0)
            self.assertEqual(coeffs[2], 0.0)
            self.assertAlmostEqual(
                alignment.yaw_from_quaternion_coeffs(coeffs), yaw, places=9)

    def test_official_examples_decode_to_expected_yaw(self):
        # val_unseen episode 1: [0, 0.5, 0, 0.866] is a 60 degree left turn.
        self.assertAlmostEqual(
            math.degrees(alignment.yaw_from_quaternion_coeffs(
                [-0.0, 0.5, 0.0, 0.8660254037844387])), 60.0, places=6)
        self.assertEqual(alignment.yaw_from_quaternion_coeffs([0, 0, 0, 1]), 0.0)

    def test_bearing_follows_habitat_axes(self):
        self.assertAlmostEqual(alignment.bearing_between([0, 0, 0], [0, 0, -2]), 0.0)
        self.assertAlmostEqual(
            alignment.bearing_between([0, 0, 0], [-2, 0, 0]), math.pi / 2)
        self.assertAlmostEqual(
            alignment.bearing_between([0, 0, 0], [2, 0, 0]), -math.pi / 2)
        self.assertAlmostEqual(
            abs(alignment.bearing_between([0, 0, 0], [0, 0, 2])), math.pi)


class GTInitialBearingTest(unittest.TestCase):
    def test_first_waypoint_when_it_is_far_enough(self):
        bearing, index, distance = alignment.gt_initial_bearing(
            [[0, 0, 0], [0, 0, -1.5], [3, 0, -1.5]])
        self.assertAlmostEqual(bearing, 0.0)
        self.assertEqual(index, 1)
        self.assertAlmostEqual(distance, 1.5)

    def test_short_first_segment_skips_to_next_waypoint(self):
        bearing, index, distance = alignment.gt_initial_bearing(
            [[0, 0, 0], [0, 0, -0.6], [-1.0, 0, -0.6]])
        self.assertEqual(index, 2)
        self.assertAlmostEqual(distance, 1.6)
        self.assertAlmostEqual(bearing, math.atan2(1.0, 0.6))

    def test_short_path_falls_back_to_last_waypoint(self):
        bearing, index, distance = alignment.gt_initial_bearing(
            [[0, 0, 0], [0, 0, -0.3], [0, 0, -0.5]])
        self.assertEqual(index, 2)
        self.assertAlmostEqual(distance, 0.5)
        self.assertAlmostEqual(bearing, 0.0)

    def test_y_component_is_ignored(self):
        bearing, _, _ = alignment.gt_initial_bearing(
            [[0, 0, 0], [0, 5.0, -2.0]])
        self.assertAlmostEqual(bearing, 0.0)

    def test_rejects_degenerate_paths(self):
        with self.assertRaises(ValueError):
            alignment.gt_initial_bearing([[0, 0, 0]])
        with self.assertRaises(ValueError):
            alignment.gt_initial_bearing([[0, 0, 0], [0, 0, 0]])


class AlignEpisodeTest(unittest.TestCase):
    # GT path heads toward -x, i.e. bearing +90 degrees (left of -z).
    PATH = [[0, 0, 0], [-2.0, 0, 0], [-4.0, 0, 0]]

    def aligned_yaw_deg(self, instruction, start_rotation=(0.0, 0.0, 0.0, 1.0)):
        aligned, row = alignment.align_episode(
            episode(instruction, self.PATH, start_rotation))
        return math.degrees(alignment.yaw_from_quaternion_coeffs(
            aligned["start_rotation"])), row

    def test_turn_left_starts_ninety_degrees_right_of_gt(self):
        yaw, row = self.aligned_yaw_deg("Turn left and walk down the hall.")
        self.assertAlmostEqual(yaw, 0.0, places=6)
        self.assertEqual(row["opening_form"], "TURN_LEFT")
        self.assertEqual(row["turn_offset_deg"], 90.0)
        self.assertEqual(row["status"], alignment.STATUS_REWRITTEN)

    def test_turn_right_starts_ninety_degrees_left_of_gt(self):
        yaw, row = self.aligned_yaw_deg("Turn right and exit the kitchen.")
        self.assertAlmostEqual(abs(yaw), 180.0, places=6)
        self.assertEqual(row["turn_offset_deg"], -90.0)

    def test_turn_around_starts_facing_away_from_gt(self):
        yaw, row = self.aligned_yaw_deg("Turn around and walk to the bed.")
        self.assertAlmostEqual(yaw, -90.0, places=6)
        self.assertEqual(row["turn_offset_deg"], 180.0)

    def test_non_turn_opening_faces_gt_directly(self):
        for instruction in ("Walk straight until the door.",
                            "Exit the bedroom and turn left.",
                            "Walk past the couch into the kitchen."):
            yaw, row = self.aligned_yaw_deg(instruction)
            self.assertAlmostEqual(yaw, 90.0, places=6, msg=instruction)
            self.assertEqual(row["turn_offset_deg"], 0.0)
            self.assertNotIn(row["opening_form"], alignment.UNDEFINED_OPENING_FORMS)

    def test_undefined_opening_forms_keep_official_rotation(self):
        official = alignment.quaternion_coeffs_from_yaw(math.radians(-150.0))
        for instruction in ("Turn to face the black door. Walk forward.",
                            "Face the indoor grill and turn right."):
            source = episode(instruction, self.PATH, official)
            aligned, row = alignment.align_episode(source)
            self.assertEqual(aligned["start_rotation"], official, msg=instruction)
            self.assertEqual(row["status"], alignment.STATUS_KEPT_OFFICIAL)
            self.assertIsNone(row["turn_offset_deg"])
            self.assertEqual(row["delta_deg"], 0.0)
            self.assertIn(row["opening_form"], alignment.UNDEFINED_OPENING_FORMS)

    def test_only_start_rotation_changes_and_source_is_untouched(self):
        official = alignment.quaternion_coeffs_from_yaw(math.radians(30.0))
        source = episode("Turn left and walk.", self.PATH, official)
        snapshot = copy.deepcopy(source)
        aligned, row = alignment.align_episode(source)
        self.assertEqual(source, snapshot)
        self.assertNotEqual(aligned["start_rotation"], source["start_rotation"])
        for key in source:
            if key != "start_rotation":
                self.assertEqual(aligned[key], source[key], key)
        self.assertAlmostEqual(row["gt_bearing_deg"], 90.0)
        self.assertAlmostEqual(row["official_yaw_deg"], 30.0)
        self.assertAlmostEqual(row["aligned_yaw_deg"], 0.0)
        self.assertAlmostEqual(row["delta_deg"], -30.0)
        self.assertEqual(row["gt_anchor_index"], 1)


if __name__ == "__main__":
    unittest.main()
