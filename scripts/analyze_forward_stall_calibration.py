#!/usr/bin/env python3
"""Post-run calibration of the RGB-only forward-stall rule.

Pairs every executor ``move_forward`` command with the hidden per-action
displacement stored under ``evaluation_only/evaluation_geometry.json`` and
reports how well the online ``rgb_motion_score`` separates blocked forwards
from free motion.  Hidden geometry is read only here, after the run; nothing
in this module is imported by the online navigation stack.

Example::

    python scripts/analyze_forward_stall_calibration.py \
        outputs/e2e_eval/20260911_001436_e2e_opennav --output stall.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


ARRIVAL_END_REASON = "rgb_only_dense_stop_cluster_arrival"
STALL_END_REASON = "rgb_forward_stall"
DISPLACEMENT_BUCKETS = (
    ("stuck_lt_2cm", 0.0, 0.02),
    ("micro_2_10cm", 0.02, 0.10),
    ("slide_10_20cm", 0.10, 0.20),
    ("free_ge_20cm", 0.20, float("inf")),
)
DEFAULT_THRESHOLDS = (2.0, 5.0, 8.0, 10.0, 12.0)
DEFAULT_FRAMES = (2, 3, 4)
# A hop that later reports arrival and still travels this far after the rule
# would have fired is counted as a false stop.
FALSE_STOP_TRAVEL_M = 0.3


def _episode_dirs(round_root: Path):
    for trajectory in sorted(round_root.glob("**/trajectory.json")):
        episode_dir = trajectory.parent
        geometry = episode_dir / "evaluation_only" / "evaluation_geometry.json"
        if geometry.exists():
            yield episode_dir


def _hop_steps(trajectory: dict, action_trace: list[dict]):
    """Align executor actions with the hidden trace and attach displacement.

    The trace holds every simulator action in order: for each target first
    the ``turn_to_selected_target`` frames from ``motion_log``, then the
    executor's own ``action_history``.  Returns ``None`` when the counts do
    not line up (for example a run interrupted mid-hop).
    """
    turns_by_target = defaultdict(list)
    for item in trajectory.get("motion_log", []):
        turns_by_target[item["target_index"]].append(item)
    positions = [np.asarray(item["position_xyz"], np.float64)
                 for item in action_trace]
    hops = []
    index = 0
    for target in trajectory.get("targets", []):
        index += len(turns_by_target[target["target_index"]])
        steps = []
        for action in target.get("action_history", []):
            if index >= len(positions):
                return None
            previous = positions[index - 1] if index > 0 else positions[index]
            moved = float(np.linalg.norm(positions[index] - previous))
            steps.append({
                "action": action["action"],
                "rgb_motion_score": float(action.get("rgb_motion_score", 0.0)),
                "moved_m": moved,
            })
            index += 1
        hops.append({
            "target_index": target["target_index"],
            "end_reason": target.get("end_reason"),
            "steps": steps,
        })
    if index != len(positions):
        return None
    return hops


def _bucket(moved: float):
    for name, low, high in DISPLACEMENT_BUCKETS:
        if low <= moved < high:
            return name
    return DISPLACEMENT_BUCKETS[-1][0]


def simulate_rule(hops, threshold: float, frames: int):
    """Replay the online stall rule offline on recorded hops."""
    fired_ok = fired_bad = steps_saved = 0
    fired_at = []
    false_stops = []
    for hop in hops:
        streak = 0
        fired = None
        for position, step in enumerate(hop["steps"]):
            if step["action"] != "move_forward":
                continue
            streak = streak + 1 if step["rgb_motion_score"] < threshold else 0
            if streak >= frames:
                fired = position
                break
        if fired is None:
            continue
        remaining = hop["steps"][fired + 1:]
        travelled_after = sum(step["moved_m"] for step in remaining)
        if (hop["end_reason"] == ARRIVAL_END_REASON and
                travelled_after > FALSE_STOP_TRAVEL_M):
            fired_bad += 1
            false_stops.append({
                "episode": hop["episode"], "target_index": hop["target_index"],
                "fired_at_step": fired + 1,
                "travelled_after_m": round(travelled_after, 3)})
        else:
            fired_ok += 1
            steps_saved += len(remaining)
            fired_at.append(fired + 1)
    return {
        "threshold": threshold, "frames": frames,
        "fired_ok": fired_ok, "fired_bad": fired_bad,
        "steps_saved": steps_saved,
        "median_fired_at_step": float(np.median(fired_at)) if fired_at else None,
        "false_stops": false_stops,
    }


def summarize_hop(hop):
    forwards = [step["moved_m"] for step in hop["steps"]
                if step["action"] == "move_forward"]
    return {
        "episode": hop["episode"], "target_index": hop["target_index"],
        "end_reason": hop["end_reason"], "steps": len(hop["steps"]),
        "forward_commands": len(forwards),
        "stuck_forwards": sum(1 for moved in forwards if moved < 0.02),
        "micro_forwards": sum(1 for moved in forwards if 0.02 <= moved < 0.10),
        "travelled_m": round(sum(step["moved_m"] for step in hop["steps"]), 3),
    }


def analyze(round_root: Path, thresholds, frames_options):
    hops = []
    skipped = []
    for episode_dir in _episode_dirs(round_root):
        trajectory = json.loads((episode_dir / "trajectory.json").read_text())
        geometry = json.loads(
            (episode_dir / "evaluation_only" /
             "evaluation_geometry.json").read_text())
        aligned = _hop_steps(trajectory, geometry.get("action_trace", []))
        if aligned is None:
            skipped.append(episode_dir.name)
            continue
        for hop in aligned:
            hop["episode"] = episode_dir.name
            hops.append(hop)

    forwards = [step for hop in hops for step in hop["steps"]
                if step["action"] == "move_forward"]
    buckets = {}
    for name, _, _ in DISPLACEMENT_BUCKETS:
        scores = [step["rgb_motion_score"] for step in forwards
                  if _bucket(step["moved_m"]) == name]
        buckets[name] = {
            "count": len(scores),
            "score_percentiles_5_25_50_75_95": (
                np.percentile(scores, [5, 25, 50, 75, 95]).round(2).tolist()
                if scores else None),
        }
    sweep = [simulate_rule(hops, threshold, frames)
             for threshold in thresholds for frames in frames_options]
    unreached = [summarize_hop(hop) for hop in hops
                 if hop["end_reason"] in ("max_steps", STALL_END_REASON)]
    end_reasons = defaultdict(int)
    for hop in hops:
        end_reasons[str(hop["end_reason"])] += 1
    return {
        "round_root": str(round_root),
        "episodes_with_hidden_geometry": len(
            {hop["episode"] for hop in hops}),
        "episodes_skipped_misaligned": skipped,
        "hops": len(hops), "forward_commands": len(forwards),
        "hop_end_reasons": dict(end_reasons),
        "motion_score_by_displacement": buckets,
        "rule_sweep": sweep,
        "unreached_hops": unreached,
    }


def _print_report(report):
    print(f"round: {report['round_root']}")
    print(f"episodes with hidden geometry: "
          f"{report['episodes_with_hidden_geometry']}  hops: {report['hops']}  "
          f"forward commands: {report['forward_commands']}")
    if report["episodes_skipped_misaligned"]:
        print(f"skipped (trace misaligned): "
              f"{', '.join(report['episodes_skipped_misaligned'])}")
    print(f"hop end reasons: {report['hop_end_reasons']}")
    print("\nrgb_motion_score percentiles (5/25/50/75/95) by hidden displacement:")
    for name, item in report["motion_score_by_displacement"].items():
        print(f"  {name:14s} n={item['count']:5d}  "
              f"{item['score_percentiles_5_25_50_75_95']}")
    print("\nstall rule sweep (consecutive forwards with score < T):")
    for item in report["rule_sweep"]:
        print(f"  T={item['threshold']:>4}  K={item['frames']}  "
              f"fired_ok={item['fired_ok']:3d}  fired_bad={item['fired_bad']:2d}  "
              f"steps_saved={item['steps_saved']:4d}  "
              f"median_fired_at={item['median_fired_at_step']}")
        for stop in item["false_stops"][:3]:
            print(f"      false stop: {stop}")
    print("\nunreached hops (max_steps / rgb_forward_stall):")
    for item in report["unreached_hops"]:
        print(f"  {item['episode']} t{item['target_index']:<3d} "
              f"{item['end_reason']:22s} steps={item['steps']:3d} "
              f"fwd={item['forward_commands']:3d} stuck={item['stuck_forwards']:3d} "
              f"micro={item['micro_forwards']:3d} travelled={item['travelled_m']:.2f}m")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("round_root", type=Path)
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--frames", type=int, nargs="+",
                        default=list(DEFAULT_FRAMES))
    parser.add_argument("--output", type=Path,
                        help="optional JSON report path")
    args = parser.parse_args()
    report = analyze(args.round_root, args.thresholds, args.frames)
    _print_report(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
