#!/usr/bin/env python3
"""Deterministic post-run statistics for the RGB-only completion judge.

Two data sources are kept apart on purpose:

* ``raw_judge_call_rows``: every ``judge_edge_instruction_completion_rgb_only``
  call found in an episode's ``vlm_calls.json``.  Form and commanded-turn
  counts are recovered from the saved prompt text, so crashed episodes with no
  ``trajectory.json`` are still covered.
* ``full_fidelity_rows``: one row per real judged edge of the episodes that do
  have ``trajectory.json``, enriched with post-run hidden geometry and, when
  ``stage_completion_verification.json`` carries ``judgments``, with the
  independent verifier's verdict.

Nothing here calls a VLM or feeds anything back into online navigation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from audit_active_stop_round import (
    _load_dataset, _polyline, _project_to_path, audit_episode)
from postrun_hidden_geometry import load_hidden_geometry
from verify_round_stage_completions import (
    system_all_judged_edges, trajectory_episode_index)


JUDGE_TASK = "judge_edge_instruction_completion_rgb_only"
TURN_FORMS = {"TURN_LEFT", "TURN_RIGHT", "TURN_AROUND"}
GOAL_RADIUS_M = 3.0

# Ordered heuristics over the judge's free-text reason; first match wins.
# They label why the judge withheld "completed", not whether it was right.
REASON_CATEGORIES = [
    ("missing_turn_evidence", re.compile(
        r"\b(?:0|zero|no)\b[^.;]{0,40}?turn(?:ing|s)?\b[^.;]{0,30}?"
        r"(?:command|action)|only (?:forward|move_forward)|"
        r"consecutive forward|exclusively forward|all forward|"
        r"no (?:left|right)[- ]turn|forward (?:moves?|movements?|commands?) "
        r"only|without (?:any )?(?:left|right|turn)", re.I)),
    ("wrong_turn_direction", re.compile(
        r"(?:shows?|executed|commanded|made) (?:a |only )?(?:right|left) turn"
        r"[^.;]{0,60}(?:instead|rather than|but the (?:active )?sub-instruction "
        r"(?:is|requires) (?:to )?(?:turn(?:ing)? |take a |a )?(?:left|right))|"
        r"opposite (?:direction|turn)|turned (?:right|left) instead", re.I)),
    ("crossed_into_next_stage", re.compile(
        r"next (?:sub-)?instruction|later (?:stage|clause)|"
        r"already (?:entered|passed|executed) the next|overshot", re.I)),
    ("reversed_or_stationary", re.compile(
        r"revers|backtrack|stationary|no (?:net )?(?:progress|movement)|"
        r"did not move|blocked|barely moved", re.I)),
    ("not_there_yet", re.compile(
        r"still (?:shows?|detects?|visible|prominent|inside|within|ahead|"
        r"in front|partway|on the way|approaching|surround|remain|"
        r"present)|remains? (?:visible|inside|in view|ahead|in front|"
        r"prominent)|not yet|has not (?:yet |clearly |fully |been )?"
        r"(?:reached|entered|passed|crossed|cleared|exited|left|arrived)|"
        r"no clear (?:crossing|traversal|exit|entry)|"
        r"not (?:clearly |fully )?(?:crossed|cleared|reached|exited|"
        r"passed|behind)|continues? toward|moderate distance|"
        r"(?:ahead|in front) of the (?:agent|camera)|"
        r"(?:remain|is|are) (?:still )?(?:ahead|in front|to the side)", re.I)),
    ("landmark_absent_or_wrong_place", re.compile(
        r"rather than|instead|wrong (?:room|door|side|way)|"
        r"different (?:room|doorway|corridor|hallway)|does not match|"
        r"not the (?:same|named|instructed)|away from|"
        r"no \w+(?: \w+){0,3} (?:is|are)? ?(?:visible|detected|seen|present)|"
        r"(?:is|are) not (?:clearly )?(?:visible|detected|seen|present)|"
        r"not visible|not detected|does not (?:show|contain|depict|"
        r"include) (?:a|an|any|the) \w+|no (?:sign|evidence) of (?:a|the)|"
        r"cannot (?:be )?(?:identif|locat|see)|with no (?:visible|clear)|"
        r"no (?:visible|ascending|clear) \w+|still in an? \w+|"
        r"not (?:on|in|at|standing|positioned) (?:the|a|an) ", re.I)),
    ("insufficient_evidence", re.compile(
        r"cannot (?:be )?(?:determine|confirm|verif)|unclear|insufficient|"
        r"ambiguous|hard to (?:tell|confirm)|no (?:clear|visible) evidence|"
        r"does not (?:clearly |conclusively )?(?:show|confirm|demonstrate|"
        r"establish|indicate)|not (?:clearly |conclusively )?"
        r"(?:confirmed|satisfied|evident|established|demonstrated|shown)",
        re.I)),
]

TURN_COMMAND_FIELD = {
    "TURN_LEFT": ("left_turn_command_count",),
    "TURN_RIGHT": ("right_turn_command_count",),
    "TURN_AROUND": ("left_turn_command_count", "right_turn_command_count"),
}


def turn_form_without_turn_commands(form, action_counts):
    """TURN_* judged on an edge whose commanded history has no turn in the
    instructed direction (either direction for TURN_AROUND)."""
    fields = TURN_COMMAND_FIELD.get(str(form))
    if not fields:
        return False
    return sum(int((action_counts or {}).get(field, 0) or 0)
               for field in fields) == 0


def categorize_reason(text):
    text = str(text or "")
    for name, pattern in REASON_CATEGORIES:
        if pattern.search(text):
            return name
    return "uncategorized"


_DECODER = json.JSONDecoder()


def _json_block_after(prompt, marker):
    start = prompt.find(marker)
    if start < 0:
        return None
    start += len(marker)
    brace = prompt.find("{", start)
    if brace < 0:
        return None
    try:
        value, _ = _DECODER.raw_decode(prompt, brace)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def parse_judge_prompt(prompt):
    """Recover the sub-instruction and commanded-action summary the online
    judge was shown; both are embedded verbatim as JSON in its prompt."""
    return {
        "sub_instruction": _json_block_after(
            prompt, "ACTIVE SUB-INSTRUCTION:\n") or {},
        "action_summary": _json_block_after(
            prompt, "RGB-only commanded action history:\n") or {},
    }


def iter_episode_dirs(round_root):
    round_root = Path(round_root)
    for pattern in ("shard_*/episode_*", "episode_*"):
        for path in sorted(round_root.glob(pattern)):
            if path.is_dir():
                yield path


def _episode_id_from_dir(episode_dir):
    try:
        return int(str(episode_dir.name).split("_")[-1])
    except ValueError:
        return None


def _load_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def raw_judge_call_rows(round_root):
    rows = []
    for episode_dir in iter_episode_dirs(round_root):
        calls = _load_json(episode_dir / "vlm_calls.json") or []
        trajectory = _load_json(episode_dir / "trajectory.json")
        crashed = trajectory is None
        episode_id = (trajectory or {}).get(
            "episode_id", _episode_id_from_dir(episode_dir))
        sequence = 0
        for call in calls:
            if call.get("task") != JUDGE_TASK:
                continue
            result = call.get("result") or {}
            parsed = parse_judge_prompt(call.get("prompt") or "")
            sub = parsed["sub_instruction"]
            actions = parsed["action_summary"]
            form = str(sub.get("form", ""))
            left = int(actions.get("left_turn_command_count", 0) or 0)
            right = int(actions.get("right_turn_command_count", 0) or 0)
            usage = call.get("usage") or {}
            meta = call.get("call_meta") or {}
            rows.append({
                "episode_id": episode_id,
                "episode_dir": str(episode_dir),
                "crashed_episode": crashed,
                "judge_call_ordinal": sequence,
                "sub_instruction_id": sub.get("sub_instruction_id"),
                "form": form,
                "is_turn_form": form in TURN_FORMS,
                "navigation_instruction": sub.get("navigation_instruction"),
                "status": result.get("status"),
                "confidence": result.get("confidence"),
                "reason": result.get("reason"),
                "reason_category": categorize_reason(result.get("reason")),
                "control_steps": actions.get("control_steps"),
                "forward_command_count": actions.get("forward_command_count"),
                "left_turn_command_count": left,
                "right_turn_command_count": right,
                "turn_form_without_turn_commands": (
                    turn_form_without_turn_commands(form, actions)),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "elapsed_ms": meta.get("elapsed_ms"),
            })
            sequence += 1
    return rows


def _xz_distance(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.linalg.norm((a - b)[[0, 2]]))


def _independent_lookup(episode_dir):
    payload = _load_json(episode_dir / "stage_completion_verification.json")
    lookup = {}
    for item in (payload or {}).get("judgments", []) or []:
        lookup[(int(item.get("target_index", -1)),
                str(item.get("completion_field")),
                int(item.get("sub_instruction_id", -1)))] = item
    return lookup


def _verdict_class(online_status, model_semantic):
    if model_semantic is None:
        return "not_verified"
    if online_status == "completed":
        return ("completed_confirmed" if model_semantic
                else "completed_refuted")
    return "unknown_but_verified" if model_semantic else "unknown_confirmed"


def full_fidelity_rows(round_root, dataset, episode_ids=None):
    rows = []
    for episode_dir in iter_episode_dirs(round_root):
        trajectory = _load_json(episode_dir / "trajectory.json")
        if trajectory is None:
            continue
        episode_id = trajectory.get("episode_id")
        if episode_ids is not None and episode_id not in episode_ids:
            continue
        episode_index = trajectory_episode_index(trajectory, episode_dir)
        dataset_episode = dataset[episode_index]
        hidden = load_hidden_geometry(episode_dir)
        scored = {item["target_index"]: item for item in audit_episode(
            episode_index, trajectory, dataset_episode,
            episode_dir / "trajectory.json",
            hidden_geometry=hidden)["targets"]}
        reference = _polyline(dataset_episode.get("reference_path") or [])
        goal = (trajectory.get("r2r_goal_positions") or [None])[-1]
        stages = trajectory.get("instruction_stages") or []
        stage_ids = [int(stage.get("stage_id", stage.get(
            "sub_instruction_id", ordinal)))
            for ordinal, stage in enumerate(stages)]
        last_stage_id = max(stage_ids) if stage_ids else None
        independent = _independent_lookup(episode_dir)
        for item in system_all_judged_edges(trajectory):
            target = item["target"]
            completion = item["completion"]
            target_index = int(target["target_index"])
            geometry = hidden.get(target_index, {})
            score = scored.get(target_index, {})
            motion = completion.get("motion_evidence") or {}
            end_xyz = geometry.get("end_xyz")
            start_xyz = geometry.get("start_xyz")
            distance_to_goal = (_xz_distance(end_xyz, goal)
                                if end_xyz is not None and goal is not None
                                else None)
            start_distance_to_goal = (
                _xz_distance(start_xyz, goal)
                if start_xyz is not None and goal is not None else None)
            projection = (_project_to_path(end_xyz, reference)
                          if end_xyz is not None and len(reference[0])
                          else None)
            verified = independent.get(
                (target_index, item["field"], item["stage_id"]))
            model_semantic = (verified or {}).get("model_semantic_completion")
            form = str((target.get("sub_instruction") or {}).get("form", ""))
            is_last_stage = (last_stage_id is not None and
                             item["stage_id"] == last_stage_id)
            rows.append({
                "episode_id": episode_id,
                "episode_index": episode_index,
                "episode_dir": str(episode_dir),
                "target_index": target_index,
                "sub_instruction_id": item["stage_id"],
                "completion_field": item["field"],
                "form": form,
                "is_turn_form": form in TURN_FORMS,
                "navigation_instruction": (target.get("sub_instruction") or {}).get(
                    "navigation_instruction"),
                "is_last_stage": is_last_stage,
                "online_status": item["online_status"],
                "online_confidence": item["online_confidence"],
                "online_reason": completion.get("reason"),
                "online_reason_category": categorize_reason(
                    completion.get("reason")),
                "control_steps": motion.get("control_steps"),
                "forward_command_count": motion.get("forward_command_count"),
                "left_turn_command_count": motion.get("left_turn_command_count"),
                "right_turn_command_count": motion.get(
                    "right_turn_command_count"),
                "turn_form_without_turn_commands": (
                    turn_form_without_turn_commands(form, motion)),
                "end_reason": target.get("end_reason"),
                "hop_final_target_geodesic_distance_m": geometry.get(
                    "final_target_geodesic_distance_m"),
                "start_distance_to_goal_xz_m": start_distance_to_goal,
                "distance_to_goal_xz_m": distance_to_goal,
                "within_goal_radius_xz": (
                    distance_to_goal is not None and
                    distance_to_goal <= GOAL_RADIUS_M),
                "reference_path_progress_m": (
                    projection["progress_m"] if projection else None),
                "distance_to_reference_path_m": (
                    projection["distance_m"] if projection else None),
                "gt_path_progress_delta_m": score.get("gt_path_progress_delta_m"),
                "selection_heading_error_to_gt_deg": score.get(
                    "selection_heading_error_to_gt_deg"),
                "executed_heading_error_to_gt_deg": score.get(
                    "executed_heading_error_to_gt_deg"),
                "geometry_gate_passed": bool(
                    score.get("selection_within_30deg") and
                    score.get("executed_edge_within_30deg_and_forward")),
                "stop_blocking_unknown": bool(
                    item["online_status"] == "unknown" and is_last_stage and
                    distance_to_goal is not None and
                    distance_to_goal <= GOAL_RADIUS_M),
                "independent_available": verified is not None,
                "independent_model_semantic_completion": model_semantic,
                "independent_model_ordered_boundary": (verified or {}).get(
                    "model_ordered_boundary"),
                "independent_semantic_verified": (verified or {}).get(
                    "semantic_completion_verified"),
                "independent_ordered_verified": (verified or {}).get(
                    "ordered_stage_boundary_verified"),
                "independent_confidence": (verified or {}).get("confidence"),
                "independent_reason": (verified or {}).get("reason"),
                "independent_visual_evidence": (verified or {}).get(
                    "visual_evidence"),
                "independent_vlm_skipped": (verified or {}).get(
                    "independent_vlm_skipped"),
                "verdict_class": _verdict_class(
                    item["online_status"], model_semantic),
            })
    return rows


def episode_summaries(round_root, episode_ids=None):
    rows = []
    for episode_dir in iter_episode_dirs(round_root):
        trajectory = _load_json(episode_dir / "trajectory.json")
        if trajectory is None:
            continue
        episode_id = trajectory.get("episode_id")
        if episode_ids is not None and episode_id not in episode_ids:
            continue
        metrics = trajectory.get("r2r_metrics") or {}
        stop = trajectory.get("task_stop") or {}
        sequence = trajectory.get("instruction_sequence_exploration") or {}
        statuses = []
        for target in trajectory.get("targets", []):
            completion = target.get("instruction_completion") or {}
            form = (target.get("sub_instruction") or {}).get("form", "")
            statuses.append(
                f"{(target.get('sub_instruction') or {}).get('sub_instruction_id')}"
                f":{form}:{completion.get('status') or target.get('end_reason')}")
        rows.append({
            "episode_id": episode_id,
            "scene": trajectory.get("scene"),
            "instruction": trajectory.get("instruction"),
            "stage_count": len(trajectory.get("instruction_stages") or []),
            "target_count": len(trajectory.get("targets") or []),
            "completed_sub_instructions_online": sequence.get(
                "completed_sub_instructions"),
            "end_reason": sequence.get("end_reason"),
            "stop_issued": bool(stop.get("issued")),
            "simulator_reported_success": bool(
                metrics.get("simulator_reported_success")),
            "initial_geodesic_distance_m": metrics.get(
                "initial_geodesic_distance_m"),
            "final_geodesic_distance_m": metrics.get(
                "final_geodesic_distance_m"),
            "hop_sequence": " | ".join(statuses),
        })
    return rows


def _mean(values):
    values = [float(v) for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def _median(values):
    values = sorted(float(v) for v in values if v is not None)
    if not values:
        return None
    middle = len(values) // 2
    if len(values) % 2:
        return round(values[middle], 3)
    return round((values[middle - 1] + values[middle]) / 2, 3)


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(
            "" if value is None else str(value) for value in row) + " |")
    return "\n".join(lines)


def build_counts(raw_rows, full_rows):
    counts = {
        "raw_judge_calls": len(raw_rows),
        "raw_status": dict(Counter(r["status"] for r in raw_rows)),
        "raw_episodes": len({r["episode_dir"] for r in raw_rows}),
        "raw_crashed_episodes": len({
            r["episode_dir"] for r in raw_rows if r["crashed_episode"]}),
        "full_fidelity_judgments": len(full_rows),
        "full_fidelity_episodes": len({r["episode_dir"] for r in full_rows}),
        "full_status": dict(Counter(r["online_status"] for r in full_rows)),
        "independent_available": sum(
            1 for r in full_rows if r["independent_available"]),
        "verdict_class": dict(Counter(r["verdict_class"] for r in full_rows)),
        "stop_blocking_unknown": sum(
            1 for r in full_rows if r["stop_blocking_unknown"]),
    }
    return counts


def summary_markdown(round_root, raw_rows, full_rows, episode_rows, counts):
    out = [f"# Judge audit summary: `{round_root}`", ""]
    out.append(
        f"- raw judge calls: {counts['raw_judge_calls']} over "
        f"{counts['raw_episodes']} episode dirs "
        f"({counts['raw_crashed_episodes']} crashed, no trajectory.json); "
        f"status {counts['raw_status']}")
    out.append(
        f"- full-fidelity judged edges: {counts['full_fidelity_judgments']} "
        f"over {counts['full_fidelity_episodes']} episodes; status "
        f"{counts['full_status']}; independent verdict available for "
        f"{counts['independent_available']}")
    out.append("")

    out.append("## Raw calls by form (all episodes)")
    by_form = defaultdict(list)
    for row in raw_rows:
        by_form[row["form"] or "(unparsed)"].append(row)
    table_rows = []
    for form, rows in sorted(by_form.items(), key=lambda kv: -len(kv[1])):
        completed = [r for r in rows if r["status"] == "completed"]
        unknown = [r for r in rows if r["status"] == "unknown"]
        table_rows.append([
            form, len(rows), len(completed), len(unknown),
            f"{100 * len(unknown) / len(rows):.0f}%" if rows else "",
            _mean(r["confidence"] for r in completed),
            _mean(r["confidence"] for r in unknown),
            sum(1 for r in rows if r["turn_form_without_turn_commands"]),
        ])
    out.append(_table(
        ["form", "n", "completed", "unknown", "unknown %",
         "conf completed", "conf unknown", "turn form w/o turn cmds"],
        table_rows))
    out.append("")

    out.append("## Raw unknown reasons by category")
    cat = Counter((r["reason_category"], r["is_turn_form"])
                  for r in raw_rows if r["status"] == "unknown")
    categories = sorted({c for c, _ in cat})
    out.append(_table(
        ["category", "turn forms", "other forms", "total"],
        [[c, cat[(c, True)], cat[(c, False)],
          cat[(c, True)] + cat[(c, False)]] for c in categories]))
    out.append("")

    turn_rows = [r for r in raw_rows if r["is_turn_form"]]
    if turn_rows:
        out.append("## Turn forms: structural turn-command check (raw)")
        without = [r for r in turn_rows if r["turn_form_without_turn_commands"]]
        with_turns = [r for r in turn_rows
                      if not r["turn_form_without_turn_commands"]]
        out.append(_table(
            ["subset", "n", "completed", "unknown"],
            [["no turn command in edge", len(without),
              sum(r["status"] == "completed" for r in without),
              sum(r["status"] == "unknown" for r in without)],
             ["at least one turn command", len(with_turns),
              sum(r["status"] == "completed" for r in with_turns),
              sum(r["status"] == "unknown" for r in with_turns)]]))
        out.append("")

    if full_rows:
        out.append("## Full fidelity: online status x independent verdict")
        matrix = Counter((r["online_status"],
                          r["independent_model_semantic_completion"],
                          r["geometry_gate_passed"]) for r in full_rows)
        out.append(_table(
            ["online status", "independent says completed",
             "geometry gate", "n"],
            [[k[0], k[1], k[2], v] for k, v in sorted(
                matrix.items(), key=lambda kv: str(kv[0]))]))
        out.append("")
        out.append("verdict classes: " + json.dumps(counts["verdict_class"]))
        out.append("")

        out.append("## Full fidelity by form")
        by_form = defaultdict(list)
        for row in full_rows:
            by_form[row["form"]].append(row)
        table_rows = []
        for form, rows in sorted(by_form.items(), key=lambda kv: -len(kv[1])):
            classes = Counter(r["verdict_class"] for r in rows)
            table_rows.append([
                form, len(rows),
                sum(r["online_status"] == "completed" for r in rows),
                sum(r["online_status"] == "unknown" for r in rows),
                classes.get("completed_confirmed", 0),
                classes.get("completed_refuted", 0),
                classes.get("unknown_but_verified", 0),
                classes.get("unknown_confirmed", 0),
                sum(r["geometry_gate_passed"] for r in rows),
                _median(r["hop_final_target_geodesic_distance_m"]
                        for r in rows),
            ])
        out.append(_table(
            ["form", "n", "online completed", "online unknown",
             "completed confirmed", "completed refuted",
             "unknown but verified", "unknown confirmed",
             "geometry gate passed", "median hop dist to point (m)"],
            table_rows))
        out.append("")

        out.append("## Last-stage judgments and STOP-blocking unknowns")
        last = [r for r in full_rows if r["is_last_stage"]]
        out.append(
            f"- last-stage judgments: {len(last)}; unknown: "
            f"{sum(r['online_status'] == 'unknown' for r in last)}; "
            f"unknown within {GOAL_RADIUS_M} m of goal (xz): "
            f"{sum(r['stop_blocking_unknown'] for r in last)}")
        out.append(_table(
            ["episode", "target", "form", "online", "dist to goal xz (m)",
             "independent completed", "geometry gate", "online reason"],
            [[r["episode_id"], r["target_index"], r["form"],
              r["online_status"],
              None if r["distance_to_goal_xz_m"] is None
              else round(r["distance_to_goal_xz_m"], 2),
              r["independent_model_semantic_completion"],
              r["geometry_gate_passed"],
              (r["online_reason"] or "")[:160]]
             for r in sorted(last, key=lambda r: (
                 r["online_status"], r["distance_to_goal_xz_m"] or 1e9))]))
        out.append("")

    if episode_rows:
        out.append("## Episodes with trajectory.json")
        out.append(_table(
            ["episode", "stages", "targets", "completed online", "end reason",
             "STOP", "success", "start->final geodesic (m)", "hops"],
            [[r["episode_id"], r["stage_count"], r["target_count"],
              r["completed_sub_instructions_online"], r["end_reason"],
              r["stop_issued"], r["simulator_reported_success"],
              f"{r['initial_geodesic_distance_m']:.2f} -> "
              f"{r['final_geodesic_distance_m']:.2f}"
              if r["initial_geodesic_distance_m"] is not None and
              r["final_geodesic_distance_m"] is not None else "",
              r["hop_sequence"]] for r in episode_rows]))
        out.append("")
    return "\n".join(out)


def _write_csv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if value is None else value)
                             for key, value in row.items()})


def write_outputs(round_root, output_dir, raw_rows, full_rows, episode_rows):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = build_counts(raw_rows, full_rows)
    (output_dir / "judge_audit.json").write_text(json.dumps({
        "round_root": str(round_root),
        "counts": counts,
        "raw_calls": raw_rows,
        "full_fidelity_judgments": full_rows,
        "episodes": episode_rows,
    }, ensure_ascii=False, indent=2) + "\n")
    _write_csv(output_dir / "judge_audit_raw_calls.csv", raw_rows)
    _write_csv(output_dir / "judge_audit.csv", full_rows)
    _write_csv(output_dir / "judge_audit_episodes.csv", episode_rows)
    (output_dir / "judge_audit_summary.md").write_text(summary_markdown(
        round_root, raw_rows, full_rows, episode_rows, counts) + "\n")
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("round_root", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default <round_root>/judge_audit")
    parser.add_argument("--episode-ids", default=None,
                        help="comma-separated episode_id filter for the "
                             "full-fidelity and episode tables")
    args = parser.parse_args()
    episode_ids = None
    if args.episode_ids:
        episode_ids = {int(item) for item in args.episode_ids.split(",")
                       if item.strip()}
    dataset = _load_dataset(args.dataset)
    raw_rows = raw_judge_call_rows(args.round_root)
    full_rows = full_fidelity_rows(args.round_root, dataset, episode_ids)
    episode_rows = episode_summaries(args.round_root, episode_ids)
    output_dir = args.output_dir or (args.round_root / "judge_audit")
    counts = write_outputs(
        args.round_root, output_dir, raw_rows, full_rows, episode_rows)
    print(json.dumps({"output_dir": str(output_dir), "counts": counts},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
