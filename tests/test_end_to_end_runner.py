#!/usr/bin/env python3
"""Contract tests for the end-to-end evaluation orchestrator (no GPU/Habitat)."""

import contextlib
import gzip
import io
import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_end_to_end_eval as runner  # noqa: E402
from evaluate_point_navigation import failed_episode_result  # noqa: E402


HIDDEN_GEOMETRY_KEYS = {
    "start_position", "start_rotation", "reference_path", "goals",
    "position_xyz", "depth", "geodesic_distance", "navmesh",
}

STUB_SHARD_RUNNER = textwrap.dedent("""
    import json, sys
    from pathlib import Path
    from evaluate_point_navigation import failed_episode_result, summarize_results
    argv = sys.argv[1:]
    episodes = [int(v) for v in argv[argv.index("--episodes") + 1].split(",")]
    root = Path(argv[argv.index("--output-root") + 1])
    results = []
    for index in episodes:
        episode_dir = root / f"episode_{index:04d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        (episode_dir / "trajectory.json").write_text("{}")
        row = failed_episode_result(index, 0, episode_dir)
        row["simulator_reported_success"] = index % 2 == 0
        row["termination_category"] = "instruction_sequence_complete"
        results.append(row)
    (root / "run_manifest.json").write_text(json.dumps({"argv": argv}))
    (root / "summary.json").write_text(json.dumps(summarize_results(
        results, episodes, "semantic", {"device": "cpu", "stub": True})))
""")


def fake_episode(episode_id, scene):
    return {
        "episode_id": episode_id,
        "trajectory_id": 1000 + episode_id,
        "scene_id": f"mp3d/{scene}/{scene}.glb",
        "start_position": [0.0, 0.0, 0.0],
        "start_rotation": [0.0, 0.0, 0.0, 1.0],
        "reference_path": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "goals": [{"position": [1.0, 0.0, 0.0], "radius": 3.0}],
        "instruction": {"instruction_text": f"walk to goal {episode_id}"},
    }


def json_keys(payload):
    if isinstance(payload, dict):
        for key, value in payload.items():
            yield key
            yield from json_keys(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from json_keys(value)


class EndToEndRunnerFixture(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.dataset = self.root / "val_unseen.json.gz"
        episodes = [fake_episode(7, "sceneA"), fake_episode(11, "sceneB"),
                    fake_episode(13, "sceneA"), fake_episode(40, "sceneC")]
        with gzip.open(self.dataset, "wt") as handle:
            json.dump({"episodes": episodes}, handle)
        self.mp3d_root = self.root / "mp3d"
        for scene in ("sceneA", "sceneB", "sceneC"):
            (self.mp3d_root / scene).mkdir(parents=True)
            (self.mp3d_root / scene / f"{scene}.glb").write_bytes(b"glb")
        self.assets = self.root / "assets"
        for relative in (
                "models/tapnet/tapnet/checkpoints/causal_bootstapir_checkpoint.pt",
                "models/visualnav-transformer/deployment/model_weights/gnm.pth",
                "weights/huggingface/.keep"):
            path = self.assets / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"weights")
        self.stub_runner = self.root / "stub_shard_runner.py"
        self.stub_runner.write_text(STUB_SHARD_RUNNER)
        self.output_root = self.root / "outputs"

    def tearDown(self):
        self._temporary.cleanup()

    def base_argv(self, *extra):
        return [
            "--r2r-data", str(self.dataset), "--mp3d-root", str(self.mp3d_root),
            "--assets-root", str(self.assets), "--shard-runner",
            str(self.stub_runner), "--output-root", str(self.output_root),
            "--device", "cpu", "--allow-cpu", "--vlm-backend", "heuristic",
            "--poll-interval-s", "0.05", *extra,
        ]


class SelectionAndShardingTest(EndToEndRunnerFixture):
    def test_id_list_parsing_rejects_non_numeric_and_duplicates(self):
        self.assertEqual(runner.parse_id_list("7, 11,13", "ids"), ["7", "11", "13"])
        with self.assertRaises(runner.PreflightError):
            runner.parse_id_list("7,x", "ids")
        with self.assertRaises(runner.PreflightError):
            runner.parse_id_list("7,7", "ids")

    def test_dataset_loader_refuses_opennav_orientation_file(self):
        bad = self.root / "OpenNav_R2R-CE_100_bertidx.json.gz"
        bad.write_bytes(self.dataset.read_bytes())
        with self.assertRaises(runner.PreflightError):
            runner.load_dataset(bad)

    def test_ids_and_indices_resolve_to_sorted_dataset_rows(self):
        episodes, id_to_index = runner.load_dataset(self.dataset)
        by_id = runner.resolve_selection(
            episodes, id_to_index, episode_ids=["40", "7"])
        self.assertEqual([item["episode_index"] for item in by_id], [0, 3])
        self.assertEqual([item["episode_id"] for item in by_id], ["7", "40"])
        by_index = runner.resolve_selection(
            episodes, id_to_index, episode_indices=["2", "1"])
        self.assertEqual([item["episode_id"] for item in by_index], ["11", "13"])
        with self.assertRaises(runner.PreflightError):
            runner.resolve_selection(episodes, id_to_index, episode_ids=["999"])
        with self.assertRaises(runner.PreflightError):
            runner.resolve_selection(
                episodes, id_to_index, episode_indices=["4"])

    def test_opennav100_ids_are_frozen_and_exclude_orientation_source(self):
        ids = runner.load_opennav100_ids()
        self.assertEqual(len(ids), 100)
        self.assertEqual(len(set(ids)), 100)
        payload = json.loads(runner.OPENNAV100_IDS_FILE.read_text())
        self.assertIn("bertidx", payload["note"])
        self.assertNotIn("start_rotation", set(json_keys(payload["episode_ids"])))

    def test_round_robin_sharding_matches_reference_launcher(self):
        self.assertEqual(runner.shard_assignment(7, 3),
                         [[0, 3, 6], [1, 4], [2, 5]])
        self.assertEqual(runner.shard_assignment(2, 5), [[0], [1]])
        with self.assertRaises(runner.PreflightError):
            runner.shard_assignment(2, 0)

    def test_shard_command_pins_rgb_only_production_flags(self):
        command = runner.shard_command(
            Path("/x/evaluate_point_navigation.py"), [6, 10], Path("/out/shard_0"),
            "cuda:1", {"policy": "gnm", "targets": None, "seed": 3},
            ["--views", "6"])
        joined = " ".join(command)
        self.assertIn("--mode semantic", joined)
        self.assertIn("--exploration-strategy instruction-sequence-recovery", joined)
        self.assertIn("--tracking-cluster-profile rgb_only_dense_stop_v1", joined)
        self.assertIn("--episodes 6,10", joined)
        self.assertIn("--output-root /out/shard_0", joined)
        self.assertIn("--device cuda:1", joined)
        self.assertIn("--policy gnm", joined)
        self.assertIn("--seed 3", joined)
        self.assertNotIn("--targets", joined)
        self.assertTrue(joined.endswith("--views 6"))


class MergeTest(EndToEndRunnerFixture):
    def test_merge_keeps_unscored_and_not_run_episodes_explicit(self):
        round_dir = self.root / "round"
        shard_0 = round_dir / "shard_0"
        shard_1 = round_dir / "shard_1"
        shard_0.mkdir(parents=True)
        shard_1.mkdir(parents=True)
        rows = []
        for index in (0, 2):
            row = failed_episode_result(index, 0, shard_0 / f"episode_{index:04d}")
            row["simulator_reported_success"] = index == 2
            (shard_0 / f"episode_{index:04d}").mkdir()
            if index == 2:
                row["termination_category"] = "instruction_sequence_complete"
                (shard_0 / f"episode_{index:04d}" / "trajectory.json").write_text("{}")
            rows.append(row)
        (shard_0 / "summary.json").write_text(json.dumps({
            "configuration": {"device": "cuda:0"}, "results": rows}))
        (shard_1 / "episode_0001").mkdir()
        (shard_1 / "episode_0001" / "trajectory.json").write_text("{}")
        (shard_1 / "provider_interruption_episode_0003.json").write_text(
            json.dumps({"episode_index": 3, "status": "invalid"}))
        shards = [
            {"directory": "shard_0", "episode_indices": [0, 2], "returncode": 0},
            {"directory": "shard_1", "episode_indices": [1, 3], "returncode": 86},
        ]
        summary = runner.merge_shards(round_dir, shards)
        self.assertEqual(summary["episode_count"], 2)
        self.assertEqual(summary["scored_episode_indices"], [0, 2])
        self.assertEqual(summary["simulator_reported_success_count"], 1)
        self.assertEqual(summary["process_failed_episode_indices"], [0])
        self.assertEqual(summary["unscored_episode_indices"], [1])
        self.assertEqual(summary["not_run_episode_indices"], [3])
        self.assertEqual(summary["provider_interruptions"][0]["episode_index"], 3)
        self.assertEqual(summary["configuration"], {"device": "cuda:0"})
        self.assertTrue(summary["shards"][0]["summary_present"])
        self.assertFalse(summary["shards"][1]["summary_present"])

    def test_merge_without_any_scored_episode_does_not_divide_by_zero(self):
        round_dir = self.root / "empty_round"
        (round_dir / "shard_0").mkdir(parents=True)
        summary = runner.merge_shards(round_dir, [
            {"directory": "shard_0", "episode_indices": [5], "returncode": 1}])
        self.assertEqual(summary["episode_count"], 0)
        self.assertEqual(summary["not_run_episode_indices"], [5])


class OrchestratorTest(EndToEndRunnerFixture):
    @staticmethod
    def run_main(argv):
        with contextlib.redirect_stdout(io.StringIO()):
            return runner.main(argv)

    def round_dirs(self):
        return sorted(path for path in self.output_root.glob("*") if path.is_dir())

    def test_dry_run_passes_preflight_without_creating_anything(self):
        code = self.run_main(self.base_argv("--dry-run", "7,40", "--workers", "2"))
        self.assertEqual(code, 0)
        self.assertFalse(self.output_root.exists())

    def test_preflight_fails_closed_on_missing_scene_and_cpu_without_override(self):
        (self.mp3d_root / "sceneC" / "sceneC.glb").unlink()
        self.assertEqual(self.run_main(self.base_argv("--dry-run", "40")), 2)
        argv = [item for item in self.base_argv("--dry-run", "7")
                if item != "--allow-cpu"]
        self.assertEqual(runner.main(argv), 2)

    def test_selection_flags_are_mutually_exclusive(self):
        self.assertEqual(self.run_main(self.base_argv("--dry-run", "7", "--all")), 2)
        self.assertEqual(self.run_main(self.base_argv("--dry-run")), 2)

    def test_parallel_shards_produce_round_manifest_log_and_merged_summary(self):
        code = self.run_main(self.base_argv(
            "7,11,13", "--workers", "2", "--run-tag", "unit", "--views", "6"))
        self.assertEqual(code, 0)
        (round_dir,) = self.round_dirs()
        self.assertTrue(round_dir.name.endswith("_unit"))
        manifest = json.loads((round_dir / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["workers"], 2)
        self.assertEqual(manifest["episode_selection"]["source"], "ids")
        self.assertEqual(
            [item["episode_index"] for item in
             manifest["episode_selection"]["episodes"]], [0, 1, 2])
        self.assertEqual(manifest["shards"][0]["episode_indices"], [0, 2])
        self.assertEqual(manifest["shards"][1]["episode_indices"], [1])
        self.assertEqual([shard["returncode"] for shard in manifest["shards"]],
                         [0, 0])
        self.assertEqual(manifest["forwarded_args"], ["--views", "6"])
        self.assertEqual(manifest["not_run_episode_indices"], [])
        self.assertFalse(HIDDEN_GEOMETRY_KEYS & set(json_keys(manifest)))
        self.assertTrue((round_dir / "process.log").read_text().strip())
        for shard in ("shard_0", "shard_1"):
            self.assertTrue((round_dir / shard / "shard.log").exists())
            self.assertTrue((round_dir / shard / "summary.json").exists())
        self.assertTrue(
            (round_dir / "shard_0" / "episode_0002" / "trajectory.json").exists())
        summary = json.loads((round_dir / "summary.json").read_text())
        self.assertEqual(summary["episode_count"], 3)
        self.assertEqual(summary["scored_episode_indices"], [0, 1, 2])
        self.assertEqual(summary["simulator_reported_success_count"], 2)
        self.assertEqual(summary["termination_category_counts"],
                         {"instruction_sequence_complete": 3})
        shard_argv = json.loads(
            (round_dir / "shard_0" / "run_manifest.json").read_text())["argv"]
        self.assertIn("--views", shard_argv)
        self.assertIn("rgb_only_dense_stop_v1", shard_argv)

    def test_resume_reuses_manifest_selection_and_config(self):
        self.assertEqual(self.run_main(self.base_argv("--all")), 2)  # ids absent
        self.assertEqual(self.run_main(self.base_argv("--episode-indices", "1,3")), 0)
        (round_dir,) = self.round_dirs()
        original = json.loads((round_dir / "manifest.json").read_text())
        code = self.run_main(self.base_argv("--resume", str(round_dir)))
        self.assertEqual(code, 0)
        self.assertEqual(self.round_dirs(), [round_dir])
        resumed = json.loads((round_dir / "manifest.json").read_text())
        self.assertEqual(resumed["created_utc"], original["created_utc"])
        self.assertIsNotNone(resumed["resumed_utc"])
        self.assertEqual(resumed["episode_selection"], original["episode_selection"])
        self.assertEqual(self.run_main(self.base_argv(
            "--resume", str(round_dir), "7")), 2)

    def test_failed_shard_is_reported_and_exit_code_is_nonzero(self):
        self.stub_runner.write_text("import sys; sys.exit(86)")
        code = self.run_main(self.base_argv("7,11", "--workers", "2"))
        self.assertEqual(code, 1)
        (round_dir,) = self.round_dirs()
        manifest = json.loads((round_dir / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete_with_failures")
        self.assertEqual([shard["returncode"] for shard in manifest["shards"]],
                         [86, 86])
        summary = json.loads((round_dir / "summary.json").read_text())
        self.assertEqual(summary["episode_count"], 0)
        self.assertEqual(summary["not_run_episode_indices"], [0, 1])


if __name__ == "__main__":
    unittest.main()
