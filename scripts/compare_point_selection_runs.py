#!/usr/bin/env python3
"""Paired comparison of two real-state VLM point-selection evaluations."""

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def load_rows(directory):
    return [json.loads(line) for line in
            (Path(directory) / "results.jsonl").read_text().splitlines()
            if line]


def identity(row):
    return (row["split"], str(row["episode_id"]), row["episode_index"],
            row["scene_id"], row["path_index"], row["stage_index"],
            row["stage"]["form"])


def exact_mcnemar_p(fixed, regressed):
    discordant = fixed + regressed
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k)
               for k in range(min(fixed, regressed) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("variant", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    baseline = load_rows(args.baseline)
    variant = load_rows(args.variant)
    if [identity(row) for row in baseline] != [identity(row) for row in variant]:
        raise RuntimeError("baseline and variant do not use identical cases")

    transitions = collections.Counter()
    by_form = collections.defaultdict(collections.Counter)
    fixed_cases, regressed_cases = [], []
    error_deltas = []
    for case_index, (old, new) in enumerate(zip(baseline, variant)):
        old_correct = (old["error"] is None and
                       old["heading_error_to_demo_deg"] < 90)
        new_correct = (new["error"] is None and
                       new["heading_error_to_demo_deg"] < 90)
        transition = ("correct" if old_correct else "wrong") + "->" + (
            "correct" if new_correct else "wrong")
        transitions[transition] += 1
        by_form[old["stage"]["form"]][transition] += 1
        delta = (float(new["heading_error_to_demo_deg"]) -
                 float(old["heading_error_to_demo_deg"]))
        error_deltas.append(delta)
        record = {
            "case_index": case_index,
            "episode_id": old["episode_id"],
            "form": old["stage"]["form"],
            "baseline_heading_error_deg": old["heading_error_to_demo_deg"],
            "variant_heading_error_deg": new["heading_error_to_demo_deg"],
            "baseline_view": old["chosen_view"],
            "variant_view": new["chosen_view"],
            "allowed_views": old.get("selection", {}).get("allowed_views"),
        }
        if transition == "wrong->correct":
            fixed_cases.append(record)
        elif transition == "correct->wrong":
            regressed_cases.append(record)

    def count_under(rows, threshold):
        return sum(row["error"] is None and
                   row["heading_error_to_demo_deg"] < threshold for row in rows)

    fixed = transitions["wrong->correct"]
    regressed = transitions["correct->wrong"]
    report = {
        "baseline": str(args.baseline.resolve()),
        "variant": str(args.variant.resolve()),
        "identical_case_count": len(baseline),
        "case_identity_equal": True,
        "baseline_counts": {str(value): count_under(baseline, value)
                            for value in (30, 45, 60, 90)},
        "variant_counts": {str(value): count_under(variant, value)
                           for value in (30, 45, 60, 90)},
        "primary_accuracy_delta": (
            count_under(variant, 90) - count_under(baseline, 90)) /
            max(len(baseline), 1),
        "transitions": dict(transitions),
        "paired_mcnemar_exact_two_sided_p": exact_mcnemar_p(fixed, regressed),
        "mean_heading_error_delta_deg": float(np.mean(error_deltas)),
        "median_heading_error_delta_deg": float(np.median(error_deltas)),
        "heading_error_improved_cases": sum(delta < 0 for delta in error_deltas),
        "heading_error_regressed_cases": sum(delta > 0 for delta in error_deltas),
        "unchanged_cases": sum(delta == 0 for delta in error_deltas),
        "rear_view_selection": {
            "baseline": sum(row.get("chosen_view") in {2, 3, 4}
                            for row in baseline),
            "variant": sum(row.get("chosen_view") in {2, 3, 4}
                           for row in variant),
        },
        "wrong_rear_view_selection": {
            "baseline": sum(
                row.get("chosen_view") in {2, 3, 4} and
                row["heading_error_to_demo_deg"] >= 90 for row in baseline),
            "variant": sum(
                row.get("chosen_view") in {2, 3, 4} and
                row["heading_error_to_demo_deg"] >= 90 for row in variant),
        },
        "fixed_cases": fixed_cases,
        "regressed_cases": regressed_cases,
        "by_form_transitions": {
            form: dict(values) for form, values in sorted(by_form.items())},
    }
    output = args.output or args.variant / "comparison_to_baseline.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    manifest_path = args.variant / "manifest.json"
    if manifest_path.exists() and output.parent.resolve() == args.variant.resolve():
        manifest = json.loads(manifest_path.read_text())
        comparison_summary = {
            key: report[key] for key in (
                "identical_case_count", "case_identity_equal",
                "baseline_counts", "variant_counts", "primary_accuracy_delta",
                "transitions", "paired_mcnemar_exact_two_sided_p",
                "mean_heading_error_delta_deg", "rear_view_selection",
                "wrong_rear_view_selection")}
        # A variant may be compared with several retained versions. Keep every
        # comparison addressable instead of silently overwriting the previous
        # baseline entry when an additional diagnostic comparison is created.
        baseline_key = args.baseline.name
        manifest.setdefault("paired_comparisons", {})[
            baseline_key] = comparison_summary
        if output.name == "comparison_to_baseline.json":
            manifest["paired_comparison"] = comparison_summary
        manifest.setdefault("artifact_sha256", {})[
            output.name] = hashlib.sha256(output.read_bytes()).hexdigest()
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
