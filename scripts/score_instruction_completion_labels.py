#!/usr/bin/env python3
"""Score already-recorded completion decisions against audited edge labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-upstream-physical-failures", action="store_true",
        help="normally semantic scoring requires outcome=true_arrival")
    args = parser.parse_args(argv)

    label_document = json.loads(args.labels.read_text())
    labels = {int(item["case_index"]): item
              for item in label_document["labels"]}
    rows = []
    for case_dir in sorted((args.source_root / args.profile).glob("case_*")):
        path = case_dir / "result.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        case_index = int(result["case_index"])
        if case_index not in labels:
            continue
        if not result.get("instruction_completion_evaluated"):
            continue
        if (not args.include_upstream_physical_failures and
                result.get("outcome") != "true_arrival"):
            continue
        reference = str(labels[case_index]["status"])
        prediction = str(result["instruction_completion"]["status"])
        rows.append({
            "case_index": case_index,
            "prediction": prediction,
            "reference": reference,
            "correct": prediction == reference,
            "rationale": labels[case_index].get("rationale", ""),
            "source_result": str(path),
        })

    true_positive = sum(
        row["prediction"] == "completed" and row["reference"] == "completed"
        for row in rows)
    false_positive = sum(
        row["prediction"] == "completed" and row["reference"] == "unknown"
        for row in rows)
    false_negative = sum(
        row["prediction"] == "unknown" and row["reference"] == "completed"
        for row in rows)
    true_negative = sum(
        row["prediction"] == "unknown" and row["reference"] == "unknown"
        for row in rows)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    summary = {
        "count": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "accuracy": sum(row["correct"] for row in rows) / max(len(rows), 1),
        "confusion": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
        },
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "reference_source": "manual_edge_semantic_audit",
        "requires_true_point_arrival": not args.include_upstream_physical_failures,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
