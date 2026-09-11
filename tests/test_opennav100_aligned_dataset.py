#!/usr/bin/env python3
"""Freeze checks for the committed OpenNav100 start-aligned dataset."""

import gzip
import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import start_rotation_alignment as alignment  # noqa: E402
from evaluate_point_navigation import DEFAULT_R2R_DATA  # noqa: E402
from run_end_to_end_eval import ALIGNED_OPENNAV100_DATA, OPENNAV100_IDS_FILE  # noqa: E402

BUILD_MANIFEST = ALIGNED_OPENNAV100_DATA.with_name("build_manifest.json")
AUDIT_JSONL = ALIGNED_OPENNAV100_DATA.with_name("start_rotation_alignment_audit.jsonl")


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AlignedDatasetArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with gzip.open(ALIGNED_OPENNAV100_DATA, "rt") as handle:
            cls.payload = json.load(handle)
        cls.manifest = json.loads(BUILD_MANIFEST.read_text())
        cls.ids = [str(v) for v in json.loads(
            OPENNAV100_IDS_FILE.read_text())["episode_ids"]]
        cls.audit = [json.loads(line) for line in
                     AUDIT_JSONL.read_text().splitlines() if line.strip()]

    def test_episode_ids_match_the_frozen_manifest_in_order(self):
        episodes = self.payload["episodes"]
        self.assertEqual(len(episodes), 100)
        self.assertEqual([str(e["episode_id"]) for e in episodes], self.ids)
        self.assertEqual(self.manifest["n_episodes"], 100)
        self.assertEqual(self.manifest["source_ids_sha256"],
                         sha256_of(OPENNAV100_IDS_FILE))

    def test_build_manifest_hashes_match_the_files(self):
        self.assertEqual(self.manifest["dataset_sha256"],
                         sha256_of(ALIGNED_OPENNAV100_DATA))
        self.assertEqual(self.manifest["audit_jsonl_sha256"], sha256_of(AUDIT_JSONL))
        self.assertNotIn("bertidx", str(ALIGNED_OPENNAV100_DATA))

    def test_rule_parameters_are_the_frozen_ones(self):
        rule = self.manifest["rule"]
        self.assertEqual(rule["opening_turn_offset_deg"],
                         alignment.OPENING_TURN_OFFSET_DEG)
        self.assertEqual(rule["undefined_opening_forms_keep_official"],
                         sorted(alignment.UNDEFINED_OPENING_FORMS))
        self.assertEqual(rule["gt_bearing_min_distance_m"],
                         alignment.GT_BEARING_MIN_DISTANCE_M)
        statuses = {row["status"] for row in self.audit}
        self.assertEqual(statuses, {alignment.STATUS_REWRITTEN,
                                    alignment.STATUS_KEPT_OFFICIAL})
        self.assertEqual(len(self.audit), 100)

    def test_rotations_are_unit_yaw_only_quaternions(self):
        for episode in self.payload["episodes"]:
            x, y, z, w = episode["start_rotation"]
            self.assertEqual((x, z), (0.0, 0.0), episode["episode_id"])
            self.assertAlmostEqual(y * y + w * w, 1.0, places=9)

    def test_only_start_rotation_differs_from_official_and_rule_is_deterministic(self):
        if not Path(DEFAULT_R2R_DATA).exists():
            self.skipTest(f"official val_unseen not available: {DEFAULT_R2R_DATA}")
        with gzip.open(DEFAULT_R2R_DATA, "rt") as handle:
            official = json.load(handle)
        self.assertEqual(self.manifest["source_val_unseen_sha256"],
                         sha256_of(DEFAULT_R2R_DATA))
        by_id = {str(e["episode_id"]): e for e in official["episodes"]}
        self.assertEqual(self.payload["instruction_vocab"],
                         official.get("instruction_vocab"))
        audit_by_id = {str(row["episode_id"]): row for row in self.audit}
        for aligned in self.payload["episodes"]:
            source = by_id[str(aligned["episode_id"])]
            self.assertEqual(set(source), set(aligned))
            for key in source:
                if key != "start_rotation":
                    self.assertEqual(aligned[key], source[key],
                                     (aligned["episode_id"], key))
            rebuilt, row = alignment.align_episode(
                source, self.manifest["rule"]["gt_bearing_min_distance_m"])
            self.assertEqual(rebuilt["start_rotation"], aligned["start_rotation"],
                             aligned["episode_id"])
            self.assertEqual(row, audit_by_id[str(aligned["episode_id"])])


if __name__ == "__main__":
    unittest.main()
