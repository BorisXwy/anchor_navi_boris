#!/usr/bin/env python3
"""Export the executable point-selection taxonomy as readable Markdown."""

import argparse
from pathlib import Path

from point_selection_strategies import STRATEGIES


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/r2r_instruction_analysis/point_selection_strategies.md")
    args = parser.parse_args()
    lines = [
        "# R2R form-specific point-selection strategies", "",
        "Every strategy follows detect -> candidate generation -> geometric constraints -> ranking -> arrival.", "",
    ]
    for form, strategy in STRATEGIES.items():
        lines += [f"## {form}", "", f"- Perception: {', '.join(strategy['perception'])}",
                  f"- Detect: {strategy['detect']}", f"- Candidate generation: {strategy['candidates']}",
                  "- Constraints:"]
        lines += [f"  - {constraint}" for constraint in strategy["constraints"]]
        lines += [f"- Rank: {strategy['rank']}", f"- Arrival: {strategy['arrival']}",
                  f"- Fallback: {strategy['fallback']}", f"- Waypoints: {strategy['waypoints']}", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
