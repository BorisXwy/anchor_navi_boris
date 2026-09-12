#!/usr/bin/env python3
"""Freeze per-hop labels for calibrating the RGB-only completion judge.

Every judged edge of one finished round already carries two post-run labels
that the online judge never saw: the independent re-verification verdict
(``verify_round_stage_completions.py``) and hidden reference-path geometry
(``analyze_judge_round.py`` -> ``judge_audit/judge_audit.csv``).  This script
turns them into a deterministic three-way label per hop and a dev/holdout
split by episode, so a new judge prompt version can be scored offline with
``replay_rgb_only_judge_round.py`` before it touches any online run.

Label rule (project_rulle.md 16.3: labels are frozen before the model call
and never enter a prompt):

* POS  -- last stage inside the 3 m goal radius (the task metric itself), or
          a non-last hop whose executed edge is on the reference route
          (heading error <= 30 deg, progress > 0.5 m, distance to route
          < 1.5 m) and which the independent verifier called complete.
* NEG  -- last stage more than 3 m from the goal (a STOP there fails the
          episode), or a non-last hop that left the route (heading error
          > 60 deg, or no progress, or > 2.5 m from the route) and which the
          independent verifier called not complete.
* AMB  -- everything else; replayed but never scored.

The route thresholds are the ones used for 病根 D / E in the 95-failure
analysis.  The split is the parity of the episode's rank among sorted ids.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROUND = ROOT / "outputs/e2e_eval/20260911_235744_opennav100_aligned_actrev"
DEFAULT_OUTPUT = ROOT / "data/judge_calibration_labels_20260911_235744_v1.json"

ON_ROUTE = {"max_heading_error_deg": 30.0, "min_progress_m": 0.5,
            "max_distance_to_route_m": 1.5}
OFF_ROUTE = {"min_heading_error_deg": 60.0, "max_progress_m": 0.0,
             "min_distance_to_route_m": 2.5}
GOAL_RADIUS_M = 3.0
LABELS = ("POS", "NEG", "AMB")
SPLITS = ("dev", "holdout")


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool(value):
    return str(value).strip().lower() == "true"


def on_route(row):
    heading = _float(row["executed_heading_error_to_gt_deg"])
    progress = _float(row["gt_path_progress_delta_m"])
    distance = _float(row["distance_to_reference_path_m"])
    return (heading is not None and progress is not None and
            distance is not None and
            heading <= ON_ROUTE["max_heading_error_deg"] and
            progress > ON_ROUTE["min_progress_m"] and
            distance < ON_ROUTE["max_distance_to_route_m"])


def off_route(row):
    heading = _float(row["executed_heading_error_to_gt_deg"])
    progress = _float(row["gt_path_progress_delta_m"])
    distance = _float(row["distance_to_reference_path_m"])
    return ((heading is not None and
             heading > OFF_ROUTE["min_heading_error_deg"]) or
            (progress is not None and
             progress <= OFF_ROUTE["max_progress_m"]) or
            (distance is not None and
             distance > OFF_ROUTE["min_distance_to_route_m"]))


def label_row(row):
    last_stage = _bool(row["is_last_stage"])
    verified = _bool(row["independent_model_semantic_completion"])
    goal_distance = _float(row["distance_to_goal_xz_m"])
    if last_stage:
        if _bool(row["within_goal_radius_xz"]):
            return "POS", "last_stage_inside_goal_radius"
        if goal_distance is not None and goal_distance > GOAL_RADIUS_M:
            return "NEG", "last_stage_outside_goal_radius"
        return "AMB", "last_stage_goal_distance_unknown"
    if on_route(row) and verified:
        return "POS", "on_route_and_independently_verified"
    if off_route(row) and not verified:
        return "NEG", "off_route_and_independently_rejected"
    return "AMB", "geometry_and_verifier_disagree_or_middling"


def split_by_episode(episode_ids):
    ordered = sorted(set(int(value) for value in episode_ids))
    return {episode: SPLITS[rank % 2] for rank, episode in enumerate(ordered)}


def build_labels(csv_path):
    rows = list(csv.DictReader(Path(csv_path).open()))
    if not rows:
        raise SystemExit(f"no rows in {csv_path}")
    split = split_by_episode(row["episode_id"] for row in rows)
    labels = []
    for row in rows:
        label, rule = label_row(row)
        labels.append({
            "episode_id": int(row["episode_id"]),
            "target_index": int(row["target_index"]),
            "sub_instruction_id": int(row["sub_instruction_id"]),
            "form": row["form"],
            "is_last_stage": _bool(row["is_last_stage"]),
            "label": label,
            "rule": rule,
            "split": split[int(row["episode_id"])],
            "online_status": row["online_status"],
            "online_confidence": _float(row["online_confidence"]),
            "verdict_class": row["verdict_class"],
            "evidence": {
                "independent_model_semantic_completion": _bool(
                    row["independent_model_semantic_completion"]),
                "executed_heading_error_to_gt_deg": _float(
                    row["executed_heading_error_to_gt_deg"]),
                "gt_path_progress_delta_m": _float(
                    row["gt_path_progress_delta_m"]),
                "distance_to_reference_path_m": _float(
                    row["distance_to_reference_path_m"]),
                "distance_to_goal_xz_m": _float(row["distance_to_goal_xz_m"]),
                "within_goal_radius_xz": _bool(row["within_goal_radius_xz"]),
            },
        })
    return labels, split


def summarize(labels):
    counts = {
        "total": len(labels),
        "by_label": dict(Counter(item["label"] for item in labels)),
        "by_split": {
            name: dict(Counter(item["label"] for item in labels
                               if item["split"] == name))
            for name in SPLITS},
        "episodes_by_split": {
            name: len({item["episode_id"] for item in labels
                       if item["split"] == name})
            for name in SPLITS},
        "by_form": {},
        "online_confusion": {},
    }
    forms = sorted({item["form"] for item in labels})
    for form in forms:
        counts["by_form"][form] = dict(Counter(
            item["label"] for item in labels if item["form"] == form))
    confusion = Counter((item["label"], item["online_status"])
                        for item in labels)
    counts["online_confusion"] = {
        f"{label}/{status}": count
        for (label, status), count in sorted(confusion.items())}
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--judge-audit-csv", type=Path,
                        default=DEFAULT_ROUND / "judge_audit/judge_audit.csv")
    parser.add_argument("--round-dir", type=Path, default=DEFAULT_ROUND)
    parser.add_argument("--online-judge-version", default="v1_baseline",
                        help="prompt version the round's online judge ran")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    labels, split = build_labels(args.judge_audit_csv)
    counts = summarize(labels)
    payload = {
        "schema_version": 1,
        "task": "rgb_only_edge_completion_judge_calibration",
        "source": {
            "round_dir": str(args.round_dir),
            "judge_audit_csv": str(args.judge_audit_csv),
            "online_judge_version": args.online_judge_version,
        },
        "policy": {
            "labels": list(LABELS),
            "goal_radius_m": GOAL_RADIUS_M,
            "on_route": ON_ROUTE,
            "off_route": OFF_ROUTE,
            "positive": ("last stage within goal radius, or non-last hop "
                         "on route and independently verified complete"),
            "negative": ("last stage beyond goal radius, or non-last hop "
                         "off route and independently rejected"),
            "ambiguous": "replayed, never scored",
            "split": "parity of the episode rank among sorted episode ids",
            "usage": ("frozen scoring labels; never an input to any prompt, "
                      "candidate ranking, controller or judge"),
        },
        "counts": counts,
        "labels": labels,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(counts, indent=2))
    print(f"wrote {len(labels)} labels to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
