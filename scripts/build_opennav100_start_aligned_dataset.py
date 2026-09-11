#!/usr/bin/env python3
"""Build the OpenNav100 subset of R2R val_unseen with ``start_rotation`` aligned
to the instruction's opening turn and the ground-truth path.

Every field except ``start_rotation`` is copied verbatim from the official
``val_unseen.json.gz``; the rule lives in ``start_rotation_alignment.py``.
Outputs (default ``data/datasets/opennav100_start_aligned/``):

  val_unseen_opennav100ids_start_aligned.json.gz   the dataset (VLN-CE schema)
  build_manifest.json                               provenance + rule parameters
  start_rotation_alignment_audit.jsonl / .md        per-episode review table
"""

import argparse
import gzip
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import start_rotation_alignment as alignment  # noqa: E402
from evaluate_point_navigation import DEFAULT_R2R_DATA  # noqa: E402

DEFAULT_IDS_FILE = ROOT / "data/opennav100_episode_ids.json"
DEFAULT_OUTPUT_DIR = ROOT / "data/datasets/opennav100_start_aligned"
DATASET_NAME = "val_unseen_opennav100ids_start_aligned.json.gz"
AUDIT_NAME = "start_rotation_alignment_audit"
MANIFEST_NAME = "build_manifest.json"
RULE_VERSION = "opennav100_start_aligned_v1"


def sha256_of(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def git_revision():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_aligned_episodes(val_unseen, episode_ids, min_distance_m):
    by_id = {}
    for episode in val_unseen["episodes"]:
        by_id.setdefault(str(episode["episode_id"]), episode)
    missing = [value for value in episode_ids if value not in by_id]
    if missing:
        raise SystemExit(f"episode ids absent from val_unseen: {missing}")
    aligned_episodes, audit_rows = [], []
    for value in episode_ids:
        aligned, row = alignment.align_episode(by_id[value], min_distance_m)
        verify_alignment(by_id[value], aligned, row)
        aligned_episodes.append(aligned)
        audit_rows.append(row)
    return aligned_episodes, audit_rows


def verify_alignment(official, aligned, row):
    for key in set(official) | set(aligned):
        if key == "start_rotation":
            continue
        if official.get(key) != aligned.get(key):
            raise SystemExit(f"episode {row['episode_id']}: field {key!r} changed")
    if row["status"] == alignment.STATUS_KEPT_OFFICIAL:
        if aligned["start_rotation"] != official["start_rotation"]:
            raise SystemExit(f"episode {row['episode_id']}: kept_official but rotated")
        return
    yaw = alignment.yaw_from_quaternion_coeffs(aligned["start_rotation"])
    expected = math.radians(row["gt_bearing_deg"] - row["turn_offset_deg"])
    if abs(alignment.wrap_angle(yaw - expected)) > 1e-3:
        raise SystemExit(f"episode {row['episode_id']}: aligned yaw mismatch")


def audit_markdown(rows):
    lines = [
        "| episode_id | opening_form | status | offset° | GT bearing° | official yaw° "
        "| aligned yaw° | Δ° | anchor (idx, m) | instruction |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        offset = "" if row["turn_offset_deg"] is None else f"{row['turn_offset_deg']:+.0f}"
        instruction = row["instruction"].replace("|", "/").strip()
        lines.append(
            f"| {row['episode_id']} | {row['opening_form']} | {row['status']} | {offset} "
            f"| {row['gt_bearing_deg']:.1f} | {row['official_yaw_deg']:.1f} "
            f"| {row['aligned_yaw_deg']:.1f} | {row['delta_deg']:+.1f} "
            f"| ({row['gt_anchor_index']}, {row['gt_anchor_distance_m']:.2f}) "
            f"| {instruction} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--val-unseen", type=Path, default=DEFAULT_R2R_DATA)
    parser.add_argument("--ids", type=Path, default=DEFAULT_IDS_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-distance-m", type=float,
                        default=alignment.GT_BEARING_MIN_DISTANCE_M)
    args = parser.parse_args(argv)

    ids_payload = load_json(args.ids)
    episode_ids = [str(value) for value in ids_payload["episode_ids"]]
    val_unseen = load_json(args.val_unseen)
    aligned_episodes, audit_rows = build_aligned_episodes(
        val_unseen, episode_ids, args.min_distance_m)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output_dir / DATASET_NAME
    payload = json.dumps({"episodes": aligned_episodes,
                          "instruction_vocab": val_unseen.get("instruction_vocab")})
    # mtime=0 keeps the gzip bytes (and hence dataset_sha256) reproducible.
    with gzip.GzipFile(dataset_path, mode="wb", mtime=0) as handle:
        handle.write(payload.encode("utf-8"))
    audit_jsonl = args.output_dir / f"{AUDIT_NAME}.jsonl"
    audit_jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                   for row in audit_rows), encoding="utf-8")
    (args.output_dir / f"{AUDIT_NAME}.md").write_text(
        audit_markdown(audit_rows), encoding="utf-8")

    status_counts = Counter(row["status"] for row in audit_rows)
    manifest = {
        "rule_version": RULE_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "source_val_unseen": str(Path(args.val_unseen).resolve()),
        "source_val_unseen_sha256": sha256_of(args.val_unseen),
        "source_ids_file": str(Path(args.ids).resolve()),
        "source_ids_sha256": sha256_of(args.ids),
        "n_episodes": len(aligned_episodes),
        "rule": {
            "aligned_yaw": "gt_initial_bearing - opening_turn_offset",
            "opening_turn_offset_deg": alignment.OPENING_TURN_OFFSET_DEG,
            "other_forms_offset_deg": 0.0,
            "undefined_opening_forms_keep_official": sorted(
                alignment.UNDEFINED_OPENING_FORMS),
            "gt_bearing_min_distance_m": args.min_distance_m,
            "opening_form_classifier": "instruction_taxonomy.decompose_by_definition[0].form",
            "quaternion_order": "[x, y, z, w], rotation about +y, yaw = 2*atan2(y, w)",
        },
        "opening_form_counts": dict(sorted(Counter(
            row["opening_form"] for row in audit_rows).items())),
        "status_counts": dict(status_counts),
        "dataset_file": DATASET_NAME,
        "dataset_sha256": sha256_of(dataset_path),
        "audit_jsonl_sha256": sha256_of(audit_jsonl),
    }
    (args.output_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {dataset_path} ({len(aligned_episodes)} episodes, "
          f"{dict(status_counts)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
