#!/usr/bin/env python3
"""Rule-transparent corpus analysis for all local R2R instruction splits."""

import argparse
import collections
import gzip
import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT.parent / "3d_wm_vln/StreamVLN/data/datasets/r2r"
ACTION = (r"go|walk|turn|take|head|continue|proceed|stop|wait|exit|enter|pass|"
          r"cross|veer|bear|make|keep|follow|move|travel|climb|descend|leave|get")
SPLIT_RE = re.compile(
    rf"\s*(?:[.;!?]+|,\s*(?=(?:then\s+)?(?:{ACTION})\b)|"
    rf"\b(?:and\s+then|then)\b|\band\s+(?=(?:{ACTION})\b))\s*", re.I)

CATEGORY_PATTERNS = collections.OrderedDict([
    ("STOP_WAIT", r"\b(stop|wait|stand|remain)\b"),
    ("VERTICAL_DOWN", r"\b(descend|downstairs)\b|\bdown\s+(?:the\s+|a\s+)?(?:stairs?|steps?|staircase)\b|\b(?:stairs?|steps?|staircase)\s+down\b"),
    ("VERTICAL_UP", r"\b(climb|upstairs)\b|\bup\s+(?:the\s+|a\s+)?(?:stairs?|steps?|staircase)\b|\b(?:stairs?|steps?|staircase)\s+up\b"),
    ("EXIT_REGION", r"\b(exit|exiting|leave|leaving|outside)\b|\b(go|walk|head|get|move)\s+(?:straight\s+)?out\b"),
    ("ENTER_REGION", r"\b(enter|entering)\b|\b(go|walk|head|move|get)\s+(?:straight\s+)?into\b"),
    ("TURN_AROUND", r"\b(turn|veer|bear|make|take)\b.{0,20}\b(around|u[- ]?turn|180)\b"),
    ("TURN_LEFT", r"\b(turn|veer|bear|make|take|go|head)\b.{0,18}\bleft\b"),
    ("TURN_RIGHT", r"\b(turn|veer|bear|make|take|go|head)\b.{0,18}\bright\b"),
    ("PASS_LANDMARK", r"\b(pass|past|passed|passing)\b"),
    ("CIRCUMNAVIGATE", r"\b(around|circle)\b|\b(right|left)\s+side\s+of\b"),
    ("CROSS_SPACE", r"\b(across|cross|crossing)\b"),
    ("BETWEEN_OBJECTS", r"\bbetween\b"),
    ("SELECT_PORTAL", r"\b(first|second|third|last|next|nearest)\s+(open\s+)?(door|doorway|opening)\b|\b(door|doorway)\s+on\s+(the|your)\s+(left|right)\b"),
    ("TRAVERSE_PORTAL_REGION", r"\bthrough\b"),
    ("FOLLOW_PATH_BOUNDARY", r"\b(follow|along)\b|\bkeep\b.{0,20}\b(wall|hall|path|corridor)\b"),
    ("ADVANCE_STRAIGHT", r"\b(straight|forward|ahead)\b|\bkeep (walking|going)\b|\b(?:go|walk|head|continue)\s+down\s+(?:the\s+)?(?:hall|hallway|corridor)\b"),
    ("APPROACH_LANDMARK", r"\b(towards?|until|near|beside)\b|\bnext to\b|\b(go|walk|head|move)\s+to\b"),
    ("OBSERVATION_CUE", r"\b(you('ll| will)?|you should)\s+(see|notice|find)\b|\bwhen you see\b"),
    ("DISTANCE_PROGRESS", r"\b(half ?way|all the way|part ?way|end of|as far as)\b"),
])
COMPILED = [(name, re.compile(pattern, re.I)) for name, pattern in CATEGORY_PATTERNS.items()]

SPATIAL_TARGET_TEMPLATES = {
    "STOP_WAIT": "free floor at/near the stated landmark with a safe offset",
    "VERTICAL_DOWN": "walkable landing or stair region below the current level",
    "VERTICAL_UP": "walkable landing or stair region above the current level",
    "EXIT_REGION": "free floor immediately beyond the exit portal",
    "ENTER_REGION": "free floor immediately inside the destination region",
    "TURN_AROUND": "visible floor/opening aligned with the reverse heading",
    "TURN_LEFT": "visible floor/opening along the new left heading",
    "TURN_RIGHT": "visible floor/opening along the new right heading",
    "PASS_LANDMARK": "free floor beyond the referenced landmark",
    "CIRCUMNAVIGATE": "free floor on the specified side and beyond the obstacle",
    "CROSS_SPACE": "free floor on the far side of the crossed region",
    "BETWEEN_OBJECTS": "walkable gap between the referenced objects",
    "SELECT_PORTAL": "free floor through the ordinal/side-selected portal",
    "TRAVERSE_PORTAL_REGION": "free floor beyond the referenced portal or region",
    "FOLLOW_PATH_BOUNDARY": "distant floor along the hallway/path/boundary",
    "ADVANCE_STRAIGHT": "distant visible floor along the current travel axis",
    "APPROACH_LANDMARK": "free floor near the landmark at a safe offset",
    "OBSERVATION_CUE": "not a standalone target; attach as evidence to another segment",
    "DISTANCE_PROGRESS": "floor at the stated fractional/end progress along the route",
    "OTHER": "requires manual/VLM interpretation",
}


def split_instruction(text):
    return [piece.strip(" ,") for piece in SPLIT_RE.split(text) if piece.strip(" ,")]


def classify(segment):
    labels = [name for name, pattern in COMPILED if pattern.search(segment)]
    return labels or ["OTHER"]


def length_summary(values):
    values = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(values.mean()), "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)), "p95": float(np.percentile(values, 95)),
        "max": int(values.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/r2r_instruction_analysis")
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, split_stats = [], {}
    for split in ("train", "val_seen", "val_unseen", "test"):
        path = args.data_root / split / f"{split}.json.gz"
        with gzip.open(path, "rt") as handle:
            episodes = json.load(handle)["episodes"]
        instructions = [episode["instruction"]["instruction_text"].strip()
                        for episode in episodes]
        segments = [(text, segment) for text in instructions
                    for segment in split_instruction(text)]
        split_stats[split] = {
            "instructions": len(instructions), "unique_instructions": len(set(instructions)),
            "segments": len(segments), "segments_per_instruction": len(segments) / len(instructions),
            "instruction_words": length_summary([len(re.findall(r"[a-z']+", x.lower()))
                                                  for x in instructions]),
            "segment_words": length_summary([len(re.findall(r"[a-z']+", x[1].lower()))
                                              for x in segments]),
        }
        for instruction, segment in segments:
            rows.append({"split": split, "instruction": instruction,
                         "segment": segment, "labels": classify(segment)})

    label_counts = collections.Counter(label for row in rows for label in row["labels"])
    primary_counts = collections.Counter(row["labels"][0] for row in rows)
    cooccurrence = collections.Counter(
        tuple(sorted(row["labels"])) for row in rows if len(row["labels"]) > 1)
    examples = {}
    for label in list(CATEGORY_PATTERNS) + ["OTHER"]:
        seen = set(); selected = []
        for row in rows:
            if label in row["labels"] and row["segment"].lower() not in seen:
                selected.append({"split": row["split"], "segment": row["segment"]})
                seen.add(row["segment"].lower())
                if len(selected) == args.examples:
                    break
        examples[label] = selected

    total = len(rows)
    summary = {
        "data_root": str(args.data_root), "total_instructions": sum(
            value["instructions"] for value in split_stats.values()),
        "total_segments": total,
        "split_stats": split_stats,
        "multi_label_segments": sum(len(row["labels"]) > 1 for row in rows),
        "unmatched_segments": primary_counts["OTHER"],
        "label_counts": dict(label_counts.most_common()),
        "spatial_target_templates": SPATIAL_TARGET_TEMPLATES,
        "primary_counts": dict(primary_counts.most_common()),
        "top_label_cooccurrences": [
            {"labels": list(labels), "count": count}
            for labels, count in cooccurrence.most_common(30)],
        "examples": examples,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output_dir / "segments.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    lines = ["# R2R instruction-form analysis", "", "## Corpus", ""]
    lines += ["| split | instructions | segments | segments/instruction |",
              "|---|---:|---:|---:|"]
    for split, value in split_stats.items():
        lines.append(f"| {split} | {value['instructions']} | {value['segments']} | "
                     f"{value['segments_per_instruction']:.2f} |")
    lines += ["", f"Total: {summary['total_instructions']} instructions, {total} segments.",
              "", "## Segment forms (multi-label)", "",
              "| form | count | percent | spatial-target interpretation |",
              "|---|---:|---:|---|"]
    for label, count in label_counts.most_common():
        lines.append(f"| {label} | {count} | {100 * count / total:.2f}% | "
                     f"{SPATIAL_TARGET_TEMPLATES[label]} |")
    lines += ["", f"Multi-label segments: {summary['multi_label_segments']} "
              f"({100 * summary['multi_label_segments'] / total:.2f}%).",
              f"Unmatched segments: {summary['unmatched_segments']} "
              f"({100 * summary['unmatched_segments'] / total:.2f}%).",
              "", "## Examples", ""]
    for label in label_counts:
        lines.append(f"### {label}")
        lines.append("")
        lines.extend(f"- [{item['split']}] {item['segment']}" for item in examples[label])
        lines.append("")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "total_instructions", "total_segments", "multi_label_segments",
        "unmatched_segments", "label_counts")}, indent=2))
    print(f"report: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
