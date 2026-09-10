#!/usr/bin/env python3
"""Run a fixed R2R episode subset and aggregate per-step-VLM SR/SPL."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default="0,1,2,3,4")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "outputs/r2r_vlm_eval")
    parser.add_argument("--max-steps-per-target", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vlm-backend", choices=["deepseek", "ollama", "heuristic"],
                        default="deepseek")
    parser.add_argument("--vlm-model", default=None)
    parser.add_argument("--deepseek-env", type=Path, default=ROOT / ".env.deepseek")
    parser.add_argument("--policy", default="gnm", choices=["gnm", "vint", "nomad"])
    args = parser.parse_args()
    indices = [int(value) for value in args.episodes.split(",") if value.strip()]
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for index in indices:
        output = args.output_root / f"episode_{index:04d}"
        trajectory = output / "trajectory.json"
        if not trajectory.exists():
            command = [
                sys.executable, str(ROOT / "scripts/random_exploration.py"),
                "--episode-index", str(index), "--policy", args.policy,
                "--device", args.device, "--vlm-backend", args.vlm_backend,
                "--deepseek-env", str(args.deepseek_env),
                "--max-steps-per-target", str(args.max_steps_per_target),
                "--output-dir", str(output),
            ]
            if args.vlm_model is not None:
                command.extend(["--vlm-model", args.vlm_model])
            subprocess.run(command, cwd=ROOT, check=True)
        data = json.loads(trajectory.read_text())
        metric = data["r2r_metrics"]
        results.append({
            "episode_index": index, "episode_id": data["episode_id"],
            "success": metric["success"], "spl": metric["spl"],
            "initial_geodesic_distance_m": metric["initial_geodesic_distance_m"],
            "final_geodesic_distance_m": metric["final_geodesic_distance_m"],
            "path_length_m": metric["path_length_m"],
            "stages_completed": data["targets_completed"],
            "stages_requested": data["targets_requested"],
            "control_steps": data["total_control_steps"],
        })
    count = len(results)
    summary = {
        "episodes": count,
        "success_rate": sum(item["success"] for item in results) / count,
        "mean_spl": sum(item["spl"] for item in results) / count,
        "mean_final_geodesic_distance_m": sum(
            item["final_geodesic_distance_m"] for item in results) / count,
        "results": results,
    }
    path = args.output_root / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"summary: {path}")


if __name__ == "__main__":
    main()
