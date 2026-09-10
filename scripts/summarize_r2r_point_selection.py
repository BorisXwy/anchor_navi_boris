#!/usr/bin/env python3
"""Create per-form metrics from an R2R point-selection results JSONL."""

import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np


def metrics(rows):
    valid = [row for row in rows if row["error"] is None]
    distances = [row["distance_to_future_demo_path_m"] for row in valid]
    headings = [row["heading_error_to_demo_deg"] for row in valid]
    return {
        "scenarios": len(rows), "valid": len(valid), "failed": len(rows) - len(valid),
        "within_0_5m": sum(x <= 0.5 for x in distances) / max(len(valid), 1),
        "within_1m": sum(x <= 1 for x in distances) / max(len(valid), 1),
        "within_2m": sum(x <= 2 for x in distances) / max(len(valid), 1),
        "median_distance_m": float(np.median(distances)) if distances else math.inf,
        "mean_distance_m": float(np.mean(distances)) if distances else math.inf,
        "heading_within_30deg": sum(x <= 30 for x in headings) / max(len(valid), 1),
        "heading_within_45deg": sum(x <= 45 for x in headings) / max(len(valid), 1),
        "heading_within_60deg": sum(x <= 60 for x in headings) / max(len(valid), 1),
        "heading_in_forward_hemisphere": sum(x < 90 for x in headings) / max(len(valid), 1),
        "median_heading_error_deg": float(np.median(headings)) if headings else math.inf,
        "backtracking_rate": sum(row["backtracking_selection"] for row in valid) / max(len(valid), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.results.read_text().splitlines() if line]
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["stage"]["form"]].append(row)
    errors = collections.Counter(row["error"] for row in rows if row["error"])
    summary = {
        "overall": metrics(rows),
        "by_form": {form: metrics(values) for form, values in sorted(groups.items())},
        "failure_reasons": dict(errors),
    }
    output = args.output or args.results.with_name("form_summary.json")
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
