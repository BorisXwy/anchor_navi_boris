#!/usr/bin/env python3
"""Replay stored RGB-only completion judgments with another prompt version.

Every online judge call leaves its full prompt in ``vlm_calls.json`` and the
three contact sheets it was shown in ``vlm_artifacts/``.  This script rebuilds
the judge inputs from that stored prompt (so nothing but the prompt version
changes), optionally swaps in completion cues from a re-run of the instruction
decomposer, sends the same three images to the VLM again and writes the old
and new verdicts side by side.  No Habitat, GPU or navigation code is
involved.

Scoring uses the frozen per-hop labels written by
``build_judge_calibration_labels.py`` (POS / NEG / AMB plus a dev/holdout
split).  They are joined on ``(episode_id, target_index,
sub_instruction_id)``; the target index comes from ``trajectory.json``, whose
judged targets appear in the same order as the judge calls.  Labels only
annotate and score; they never reach a prompt.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from analyze_judge_round import JUDGE_TASK, _json_block_after, iter_episode_dirs
from instruction_decomposer import InstructionDecomposer
from vlm_harness import NavigationVLMHarness, build_vlm_backend


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ROOT / "data/judge_calibration_labels_20260911_235744_v1.json"
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
CSV_COLUMNS = [
    "episode_id", "split", "label", "label_rule", "judge_call_ordinal",
    "target_index", "sub_instruction_id", "is_last_stage", "form",
    "secondary_forms", "landmark", "old_status", "old_confidence",
    "new_status", "new_confidence", "new_model_status", "new_overrides",
    "status_changed", "landmark_visible_current", "landmark_sector_current",
    "relation_satisfied", "transition_observed", "dist_goal_at_hop_end_m",
    "in_3m_circle", "completion_cue_old", "completion_cue_new",
    "visual_arrival_evidence_old", "visual_arrival_evidence_new",
    "decomposition_replaced", "old_reason", "new_reason",
]
SCORED_LABELS = ("POS", "NEG")


def parse_stored_prompt(prompt):
    blocks = {}
    for key, marker in PROMPT_MARKERS.items():
        value = _json_block_after(prompt, marker)
        if value is None:
            raise ValueError(f"stored prompt lacks the {marker.strip()!r} block")
        blocks[key] = value
    return blocks


def load_labels(path):
    """Frozen labels keyed by (episode_id, target_index, sub_instruction_id)."""
    if path is None or not Path(path).is_file():
        return {}, {}
    payload = json.loads(Path(path).read_text())
    labels, splits = {}, {}
    for item in payload["labels"]:
        key = (int(item["episode_id"]), int(item["target_index"]),
               int(item["sub_instruction_id"]))
        labels[key] = item
        splits[int(item["episode_id"])] = item["split"]
    return labels, splits


def episode_id_of(episode_dir):
    trajectory = episode_dir / "trajectory.json"
    if trajectory.is_file():
        try:
            return int(json.loads(trajectory.read_text())["episode_id"])
        except (KeyError, TypeError, ValueError):
            pass
    return int(str(episode_dir.name).split("_")[-1])


def judged_targets(episode_dir):
    """(target_index, expected_sub_instruction_id) per judged hop, in the
    order the judge was called; None when the episode has no trajectory."""
    trajectory = episode_dir / "trajectory.json"
    if not trajectory.is_file():
        return None
    payload = json.loads(trajectory.read_text())
    judged = []
    for target in payload.get("targets", []):
        completion = target.get("instruction_completion")
        if not completion:
            continue
        judged.append((int(target["target_index"]),
                       int(completion.get("expected_sub_instruction_id", -1))))
    return judged


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


def make_backend(args):
    return build_vlm_backend(
        args.vlm_backend, args.vlm_model, timeout=args.vlm_timeout,
        deepseek_env_file=args.deepseek_env)


def replay_episode(episode_dir, args, labels, splits):
    """Replay one episode; returns (rows, log entries).  Runs in a worker
    thread, so it owns its backend and writes nothing shared."""
    backend = make_backend(args)
    episode_id = episode_id_of(episode_dir)
    decomposition = json.loads(
        (episode_dir / "instruction_decomposition.json").read_text())
    recorded_stages = decomposition["selected_sub_instructions"]
    last_stage = recorded_stages[-1]
    last_stage_id = int(last_stage["sub_instruction_id"])

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
    targets = judged_targets(episode_dir)
    if targets is not None and len(targets) != len(calls):
        raise RuntimeError(
            f"[{episode_id}] {len(calls)} judge calls but {len(targets)} "
            "judged targets in trajectory.json; cannot join labels")

    schema = NavigationVLMHarness.rgb_only_completion_schema(args.judge_version)
    rows, log_entries = [], []
    for ordinal, call in enumerate(calls):
        blocks = parse_stored_prompt(call["prompt"])
        item = dict(blocks["item"])
        stage_id = int(item.get("sub_instruction_id", -1))
        target_index = None
        if targets is not None:
            target_index, expected_id = targets[ordinal]
            if expected_id != stage_id:
                raise RuntimeError(
                    f"[{episode_id}] judge call {ordinal} is for stage "
                    f"{stage_id} but trajectory target {target_index} "
                    f"expected stage {expected_id}")
        is_last = stage_id == last_stage_id
        if not args.all_stages and not is_last:
            continue
        label = labels.get((episode_id, target_index, stage_id), {})
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
        actions = (blocks["action_summary"] or {}).get(
            "chronological_actions", [])
        raw, normalized, errors = None, None, []
        for attempt in range(args.vlm_retries + 1):
            try:
                raw = backend.generate_json(prompt, images, schema)
                normalized = NavigationVLMHarness.normalize_rgb_only_completion_result(
                    args.judge_version, raw, actions)
                break
            except (ValueError, KeyError, TypeError) as exc:
                errors.append(str(exc))
                prompt += (f"\nPrevious response was invalid: {exc}. "
                           "Return corrected JSON only.")
        if normalized is None:
            raise RuntimeError(
                f"[{episode_id}] judge replay exhausted retries: {errors}")
        old = call.get("result") or {}
        evidence = normalized["evidence"] or {}
        distance = (label.get("evidence") or {}).get("distance_to_goal_xz_m")
        row = {
            "episode_id": episode_id,
            "split": label.get("split") or splits.get(episode_id),
            "label": label.get("label"),
            "label_rule": label.get("rule"),
            "judge_call_ordinal": ordinal,
            "target_index": target_index,
            "sub_instruction_id": stage_id,
            "is_last_stage": is_last,
            "form": item.get("form"),
            "secondary_forms": "|".join(item.get("secondary_forms") or []),
            "landmark": item.get("landmark"),
            "old_status": old.get("status"),
            "old_confidence": old.get("confidence"),
            "new_status": normalized["status"],
            "new_confidence": normalized["confidence"],
            "new_model_status": normalized["model_status"],
            "new_overrides": "|".join(normalized["overrides"]),
            "status_changed": old.get("status") != normalized["status"],
            "landmark_visible_current": evidence.get("landmark_visible_current"),
            "landmark_sector_current": evidence.get("landmark_sector_current"),
            "relation_satisfied": evidence.get("relation_satisfied"),
            "transition_observed": evidence.get("transition_observed"),
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
        rows.append(row)
        log_entries.append({
            **row, "prompt": prompt, "image_paths": call["image_paths"],
            "response": raw, "validation_errors": errors,
            "redecomposition": redecomposition_note,
            "usage": getattr(backend, "last_usage", None),
        })
        print(f"[{episode_id}] stage {stage_id} hop {ordinal}: "
              f"{old.get('status')} -> {normalized['status']} "
              f"({normalized['confidence']:.2f}) label={label.get('label')}"
              + (f", goal {distance:.2f} m" if distance is not None else ""),
              flush=True)
    return rows, log_entries


def confusion(rows, status_key):
    counts = Counter()
    for row in rows:
        if row["label"] not in SCORED_LABELS:
            continue
        completed = str(row[status_key]) == "completed"
        if row["label"] == "POS":
            counts["tp" if completed else "fn"] += 1
        else:
            counts["fp" if completed else "tn"] += 1
    tp, fn, fp, tn = (counts[key] for key in ("tp", "fn", "fp", "tn"))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall else (0.0 if tp + fp + fn else None))
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "scored": tp + fn + fp + tn,
            "precision": precision, "recall": recall, "f1": f1}


def score(rows):
    """Confusion matrices of the online (old) and replayed (new) verdicts
    against the frozen labels: overall, per split, per form, per episode,
    plus the last-stage NEG false positives that would trigger wrong STOPs."""
    def block(subset):
        ambiguous = [row for row in subset if row["label"] == "AMB"]
        last_neg = [row for row in subset
                    if row["label"] == "NEG" and row["is_last_stage"]]
        return {
            "rows": len(subset),
            "old": confusion(subset, "old_status"),
            "new": confusion(subset, "new_status"),
            "ambiguous_completed": {
                "old": sum(row["old_status"] == "completed"
                           for row in ambiguous),
                "new": sum(row["new_status"] == "completed"
                           for row in ambiguous),
                "total": len(ambiguous)},
            "last_stage_neg_false_positives": {
                "old": sum(row["old_status"] == "completed"
                           for row in last_neg),
                "new": sum(row["new_status"] == "completed"
                           for row in last_neg),
                "total": len(last_neg)},
        }

    def grouped(key):
        groups = defaultdict(list)
        for row in rows:
            groups[row[key]].append(row)
        return {str(name): block(subset)
                for name, subset in sorted(groups.items(), key=lambda kv: str(kv[0]))}

    return {
        "overall": block(rows),
        "by_split": grouped("split"),
        "by_form": grouped("form"),
        "by_episode": grouped("episode_id"),
    }


def _fmt(value):
    return "-" if value is None else f"{value:.3f}"


def scoring_markdown(args, scoring, rows):
    lines = [f"# Judge replay scoring: {args.judge_version}", "",
             f"round: `{args.round_dir}`  split: `{args.split}`  "
             f"rows: {len(rows)}  labels: `{args.labels_json}`", "",
             "old = online verdict recorded in the round; new = replayed "
             f"`{args.judge_version}`.", ""]

    def table(title, blocks):
        lines.extend([f"## {title}", "",
                      "| group | rows | old TP/FN/FP/TN | old P / R / F1 | "
                      "new TP/FN/FP/TN | new P / R / F1 | AMB completed "
                      "old/new/n | last-stage NEG FP old/new/n |",
                      "|---|---:|---|---|---|---|---|---|"])
        for name, block in blocks.items():
            old, new = block["old"], block["new"]
            amb, last = (block["ambiguous_completed"],
                         block["last_stage_neg_false_positives"])
            lines.append(
                f"| {name} | {block['rows']} | "
                f"{old['tp']}/{old['fn']}/{old['fp']}/{old['tn']} | "
                f"{_fmt(old['precision'])} / {_fmt(old['recall'])} / "
                f"{_fmt(old['f1'])} | "
                f"{new['tp']}/{new['fn']}/{new['fp']}/{new['tn']} | "
                f"{_fmt(new['precision'])} / {_fmt(new['recall'])} / "
                f"{_fmt(new['f1'])} | "
                f"{amb['old']}/{amb['new']}/{amb['total']} | "
                f"{last['old']}/{last['new']}/{last['total']} |")
        lines.append("")

    table("Overall", {"all": scoring["overall"]})
    table("By split", scoring["by_split"])
    table("By form", scoring["by_form"])
    table("By episode", scoring["by_episode"])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--round-dir", type=Path, required=True)
    parser.add_argument("--episodes", default=None,
                        help="comma-separated episode ids; default = every "
                             "episode of --split")
    parser.add_argument("--split", choices=["dev", "holdout", "all"],
                        default="all",
                        help="episode subset from the labels file")
    parser.add_argument("--labels-json", type=Path, default=DEFAULT_LABELS,
                        help="frozen POS/NEG/AMB labels (scoring only)")
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
    parser.add_argument("--workers", type=int, default=1,
                        help="episodes replayed concurrently (one backend each)")
    parser.add_argument("--vlm-backend",
                        choices=["deepseek", "ollama", "heuristic"],
                        default="deepseek")
    parser.add_argument("--vlm-model", default=None)
    parser.add_argument("--vlm-timeout", type=int, default=180)
    parser.add_argument("--vlm-retries", type=int, default=2)
    parser.add_argument("--deepseek-env", type=Path,
                        default=ROOT / ".env.deepseek")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    labels, splits = load_labels(args.labels_json)
    if args.episodes:
        wanted = {int(value) for value in args.episodes.split(",")
                  if value.strip()}
    elif args.split != "all":
        wanted = {episode for episode, split in splits.items()
                  if split == args.split}
    else:
        wanted = None
    if args.split != "all" and wanted is not None:
        wanted = {episode for episode in wanted
                  if splits.get(episode) == args.split}
    if wanted is not None and not wanted:
        parser.error("no episodes selected; check --episodes/--split/--labels-json")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode_dirs = []
    for episode_dir in iter_episode_dirs(args.round_dir):
        episode_id = episode_id_of(episode_dir)
        if wanted is None or episode_id in wanted:
            episode_dirs.append(episode_dir)
    seen = {episode_id_of(path) for path in episode_dirs}
    missing = sorted((wanted or set()) - seen)

    all_rows, all_logs = [], []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(replay_episode, path, args, labels, splits)
                   for path in episode_dirs]
        for future in futures:
            rows, logs = future.result()
            all_rows.extend(rows)
            all_logs.extend(logs)
    all_rows.sort(key=lambda row: (row["episode_id"], row["judge_call_ordinal"]))
    all_logs.sort(key=lambda row: (row["episode_id"], row["judge_call_ordinal"]))

    csv_path = args.output_dir / "summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)
    with (args.output_dir / "replay_calls.jsonl").open("w") as handle:
        for entry in all_logs:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    scoring = score(all_rows)
    summary = {
        "round_dir": str(args.round_dir),
        "judge_version": args.judge_version,
        "decomposer_version": args.decomposer_version,
        "vlm_backend": args.vlm_backend,
        "split": args.split,
        "labels_json": str(args.labels_json),
        "episodes": sorted(seen),
        "missing_episodes": missing,
        "rows": len(all_rows),
        "scoring": scoring,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "scoring.md").write_text(
        scoring_markdown(args, scoring, all_rows))
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "scoring"}, indent=2))
    print(json.dumps(scoring["overall"], indent=2))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
