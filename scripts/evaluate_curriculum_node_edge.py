#!/usr/bin/env python3
"""Audit the R0-N node/edge invariant on the frozen ten-EP artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


EPISODES = (0, 3, 6, 9, 18, 27, 45, 126, 204, 219)
REQUIRED_NODE_FIELDS = {
    "node_id", "node_kind", "position_xyz", "base_yaw_rad",
    "created_global_step", "departure_purpose_sub_instruction", "six_views",
    "environment_semantics", "visual_embedding", "visual_embedding_model",
    "arrival_signal", "sub_instruction_match", "metadata",
}
REQUIRED_EDGE_FIELDS = {
    "edge_id", "edge_kind", "source_node_id", "target_node_id",
    "departure_purpose_sub_instruction", "action_history",
    "control_step_count", "traveled_distance_m", "metadata",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(input_root: Path, output_root: Path) -> dict:
    records = []
    for episode in EPISODES:
        case = input_root / f"episode_{episode:04d}"
        graph_path = case / "navigation_graph" / "navigation_graph.json"
        graph = json.loads(graph_path.read_text())
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        errors = []
        if len(nodes) < 2:
            errors.append(f"expected at least 2 nodes, got {len(nodes)}")
        if not edges:
            errors.append("no edge created")
        node_ids = {node.get("node_id") for node in nodes}
        for node in nodes:
            missing = sorted(REQUIRED_NODE_FIELDS - set(node))
            if missing:
                errors.append(f"{node.get('node_id')}: missing node fields {missing}")
            if len(node.get("six_views", [])) != 6:
                errors.append(f"{node.get('node_id')}: six_views != 6")
            if len(node.get("visual_embedding", [])) != 256:
                errors.append(f"{node.get('node_id')}: visual_embedding != 256")
            pano = node.get("metadata", {}).get("instruction_completion_panorama", {})
            if len(pano.get("views", [])) != 8:
                errors.append(f"{node.get('node_id')}: completion panorama != 8 views")
        for edge in edges:
            missing = sorted(REQUIRED_EDGE_FIELDS - set(edge))
            if missing:
                errors.append(f"{edge.get('edge_id')}: missing edge fields {missing}")
            if edge.get("source_node_id") not in node_ids:
                errors.append(f"{edge.get('edge_id')}: source node missing")
            if edge.get("target_node_id") not in node_ids:
                errors.append(f"{edge.get('edge_id')}: target node missing")
            if not edge.get("action_history"):
                errors.append(f"{edge.get('edge_id')}: empty action_history")
            if not edge.get("metadata", {}).get("edge_keyframes"):
                errors.append(f"{edge.get('edge_id')}: empty edge_keyframes")
        records.append({
            "episode_index": episode,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "passed": not errors,
            "errors": errors,
            "graph_sha256": sha256(graph_path),
        })
    passed = sum(record["passed"] for record in records)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root.resolve()),
        "episodes": list(EPISODES),
        "records": records,
        "metrics": {"count": len(records), "passed": passed,
                    "failed": len(records) - passed,
                    "invariant_pass_rate": passed / len(records)},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    audit(args.input_root, args.output_root)


if __name__ == "__main__":
    main()
