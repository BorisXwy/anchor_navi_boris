#!/usr/bin/env python3
"""Replay stored RGB-only completion judgments with another prompt version.

Every online judge call leaves its full prompt in ``vlm_calls.json`` and the
three contact sheets it was shown in ``vlm_artifacts/``.  This script rebuilds
the judge inputs from that stored prompt (so nothing but the prompt version
changes), optionally swaps in completion cues from a re-run of the instruction
decomposer, sends the same three images to the VLM again and writes the old
and new verdicts side by side.  No Habitat, GPU or navigation code is
involved.

The hidden ``dist_goal_at_hop_end_m`` label is read from a post-run analysis
file only to annotate the CSV; it never reaches a prompt.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from analyze_judge_round import JUDGE_TASK, _json_block_after, iter_episode_dirs
from instruction_decomposer import InstructionDecomposer
from vlm_harness import NavigationVLMHarness, build_vlm_backend


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOP_LABELS = (
    ROOT / "docs/e2e_eval_reports/20260911_235744_opennav100_aligned_actrev/"
    "analysis_95_failures.json")
PROMPT_MARKERS = {
    "item": "ACTIVE SUB-INSTRUCTION:\n",
    "following": (
        "FOLLOWING SUB-INSTRUCTION (context only; do not complete it early):\n"),
    "action_summary": "RGB-only commanded action history:\n",
    "previous_semantics": "RGB detector evidence at PREVIOUS node:\n",
    "current_semantics": "RGB detector evidence at CURRENT node:\n",
}
# Fields a re-run decomposer may replace on the judged stage.  Identity and
# form fields stay as recorded so the judge call remains the same test case.
REDECOMPOSED_CUE_FIELDS = (
    "landmark", "completion_cue", "semantic_spatial_target",
    "spatial_relation", "visual_arrival_evidence", "forbidden_target",
)
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["completed", "unknown"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
        "visual_evidence": {"type": "string"},
        "temporal_evidence": {"type": "string"},
    },
    "required": ["status", "confidence", "reason",
                 "visual_evidence", "temporal_evidence"],
}
CSV_COLUMNS = [
    "episode_id", "group", "judge_call_ordinal", "target_index",
    "sub_instruction_id", "is_last_stage", "form", "secondary_forms",
    "landmark", "old_status", "old_confidence", "new_status",
    "new_confidence", "status_changed", "dist_goal_at_hop_end_m",
    "in_3m_circle", "completion_cue_old", "completion_cue_new",
    "visual_arrival_evidence_old", "visual_arrival_evidence_new",
    "decomposition_replaced", "old_reason", "new_reason",
]


def parse_stored_prompt(prompt):
    blocks = {}
    for key, marker in PROMPT_MARKERS.items():
        value = _json_block_after(prompt, marker)
        if value is None:
            raise ValueError(f"stored prompt lacks the {marker.strip()!r} block")
        blocks[key] = value
    return blocks


def load_hop_labels(path):
    """Map episode id -> (group, [hop rows that received a judge verdict])."""
    if path is None or not Path(path).is_file():
        return {}
    labels = {}
    for row in json.loads(Path(path).read_text()):
        judged = [hop for hop in row.get("hops", []) if hop.get("judge")]
        labels[int(row["id"])] = (row.get("group"), judged)
    return labels


def episode_id_of(episode_dir):
    trajectory = episode_dir / "trajectory.json"
    if trajectory.is_file():
        try:
            return int(json.loads(trajectory.read_text())["episode_id"])
        except (KeyError, TypeError, ValueError):
            pass
    return int(str(episode_dir.name).split("_")[-1])


def load_images(episode_dir, image_paths):
    images = []
    for relative in image_paths:
        path = episode_dir / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        images.append(np.asarray(Image.open(path).convert("RGB")))
    return images


def redecompose_last_stage(harness, instruction, recorded_last_stage):
    """Return replacement cue fields for the recorded last stage, or None when
    the new decomposition does not end in a stage of the same form."""
    stages = InstructionDecomposer(harness).decompose(instruction)
    new_last = stages[-1].to_dict()
    recorded_form = str(recorded_last_stage.get("form", "")).upper()
    if str(new_last.get("form", "")).upper() != recorded_form:
        return None, {"new_stage_count": len(stages),
                      "new_last_form": new_last.get("form")}
    return ({key: new_last.get(key) for key in REDECOMPOSED_CUE_FIELDS},
            {"new_stage_count": len(stages),
             "new_last_form": new_last.get("form")})


def replay_episode(episode_dir, args, backend, hop_labels, writer,
                   calls_log):
    episode_id = episode_id_of(episode_dir)
    decomposition = json.loads(
        (episode_dir / "instruction_decomposition.json").read_text())
    recorded_stages = decomposition["selected_sub_instructions"]
    last_stage = recorded_stages[-1]
    last_stage_id = int(last_stage["sub_instruction_id"])
    group, judged_hops = hop_labels.get(episode_id, (None, []))

    replacement, redecomposition_note = None, None
    if args.decomposer_version:
        decomposer_harness = NavigationVLMHarness(
            backend, args.output_dir / f"decompose_{episode_id:04d}.json",
            retries=args.vlm_retries,
            decomposition_prompt_version=args.decomposer_version)
        replacement, redecomposition_note = redecompose_last_stage(
            decomposer_harness, decomposition["instruction"], last_stage)

    calls = [call for call in json.loads(
        (episode_dir / "vlm_calls.json").read_text())
        if call.get("task") == JUDGE_TASK]
    if judged_hops and len(judged_hops) != len(calls):
        print(f"[{episode_id}] {len(calls)} judge calls but {len(judged_hops)} "
              "labelled hops; distance labels left empty")
        judged_hops = []

    rows = 0
    for ordinal, call in enumerate(calls):
        blocks = parse_stored_prompt(call["prompt"])
        item = dict(blocks["item"])
        stage_id = int(item.get("sub_instruction_id", -1))
        is_last = stage_id == last_stage_id
        if not args.all_stages and not is_last:
            continue
        cue_old = {key: item.get(key) for key in REDECOMPOSED_CUE_FIELDS}
        replaced = False
        if replacement is not None and is_last:
            item.update(replacement)
            replaced = True
        prompt = NavigationVLMHarness.render_rgb_only_completion_prompt(
            args.judge_version, item, blocks["following"],
            blocks["action_summary"], blocks["previous_semantics"],
            blocks["current_semantics"])
        images = load_images(episode_dir, call["image_paths"])
        raw = backend.generate_json(prompt, images, JUDGE_SCHEMA)
        status = str(raw["status"]).strip().lower()
        if status not in {"completed", "unknown"}:
            raise ValueError(f"replayed status {status!r} is not binary")
        confidence = float(raw["confidence"])
        old = call.get("result") or {}
        hop = judged_hops[ordinal] if judged_hops else {}
        distance = hop.get("dist_goal_at_hop_end_m")
        row = {
            "episode_id": episode_id,
            "group": group,
            "judge_call_ordinal": ordinal,
            "target_index": hop.get("target_index"),
            "sub_instruction_id": stage_id,
            "is_last_stage": is_last,
            "form": item.get("form"),
            "secondary_forms": "|".join(item.get("secondary_forms") or []),
            "landmark": item.get("landmark"),
            "old_status": old.get("status"),
            "old_confidence": old.get("confidence"),
            "new_status": status,
            "new_confidence": confidence,
            "status_changed": old.get("status") != status,
            "dist_goal_at_hop_end_m": distance,
            "in_3m_circle": (distance is not None and distance <= 3.0),
            "completion_cue_old": cue_old["completion_cue"],
            "completion_cue_new": item.get("completion_cue"),
            "visual_arrival_evidence_old": cue_old["visual_arrival_evidence"],
            "visual_arrival_evidence_new": item.get("visual_arrival_evidence"),
            "decomposition_replaced": replaced,
            "old_reason": old.get("reason"),
            "new_reason": raw.get("reason"),
        }
        writer.writerow(row)
        calls_log.write(json.dumps({
            **row, "prompt": prompt, "image_paths": call["image_paths"],
            "response": raw, "redecomposition": redecomposition_note,
            "usage": getattr(backend, "last_usage", None),
        }, ensure_ascii=False) + "\n")
        calls_log.flush()
        rows += 1
        print(f"[{episode_id}] stage {stage_id} hop {ordinal}: "
              f"{old.get('status')} -> {status} ({confidence:.2f})"
              + (f", goal {distance:.2f} m" if distance is not None else ""))
    return rows


def summarize(csv_path):
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return {"rows": 0}

    def rate(subset, key):
        return (sum(row[key] == "completed" for row in subset), len(subset))

    inside = [row for row in rows if row["in_3m_circle"] == "True"]
    outside = [row for row in rows
               if row["dist_goal_at_hop_end_m"] not in ("", "None")
               and row["in_3m_circle"] == "False"]
    return {
        "rows": len(rows),
        "old_completed": rate(rows, "old_status"),
        "new_completed": rate(rows, "new_status"),
        "inside_3m_old_completed": rate(inside, "old_status"),
        "inside_3m_new_completed": rate(inside, "new_status"),
        "outside_3m_old_completed": rate(outside, "old_status"),
        "outside_3m_new_completed": rate(outside, "new_status"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--round-dir", type=Path, required=True)
    parser.add_argument("--episodes", required=True,
                        help="comma-separated episode ids")
    parser.add_argument(
        "--judge-version", default="v2_form_aware_stop_relation",
        choices=sorted(NavigationVLMHarness.RGB_ONLY_COMPLETION_PROMPT_VERSIONS))
    parser.add_argument(
        "--decomposer-version", default=None,
        choices=sorted(NavigationVLMHarness.DECOMPOSITION_PROMPT_VERSIONS),
        help="re-run the decomposer with this version and replace the last "
             "stage's cue fields before judging")
    parser.add_argument("--all-stages", action="store_true",
                        help="replay every judge call, not only the last stage")
    parser.add_argument("--vlm-backend",
                        choices=["deepseek", "ollama", "heuristic"],
                        default="deepseek")
    parser.add_argument("--vlm-model", default=None)
    parser.add_argument("--vlm-timeout", type=int, default=180)
    parser.add_argument("--vlm-retries", type=int, default=2)
    parser.add_argument("--deepseek-env", type=Path,
                        default=ROOT / ".env.deepseek")
    parser.add_argument("--hop-labels-json", type=Path,
                        default=DEFAULT_HOP_LABELS,
                        help="post-run analysis with per-hop goal distances "
                             "(CSV annotation only)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    wanted = {int(value) for value in args.episodes.split(",") if value.strip()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    backend = build_vlm_backend(
        args.vlm_backend, args.vlm_model, timeout=args.vlm_timeout,
        deepseek_env_file=args.deepseek_env)
    hop_labels = load_hop_labels(args.hop_labels_json)
    csv_path = args.output_dir / "summary.csv"
    total = 0
    with csv_path.open("w", newline="") as csv_handle, \
            (args.output_dir / "replay_calls.jsonl").open("w") as calls_log:
        writer = csv.DictWriter(csv_handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        seen = set()
        for episode_dir in iter_episode_dirs(args.round_dir):
            episode_id = episode_id_of(episode_dir)
            if episode_id not in wanted:
                continue
            seen.add(episode_id)
            total += replay_episode(
                episode_dir, args, backend, hop_labels, writer, calls_log)
    missing = sorted(wanted - seen)
    summary = {
        "round_dir": str(args.round_dir),
        "judge_version": args.judge_version,
        "decomposer_version": args.decomposer_version,
        "vlm_backend": args.vlm_backend,
        "episodes": sorted(seen),
        "missing_episodes": missing,
        **summarize(csv_path),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
