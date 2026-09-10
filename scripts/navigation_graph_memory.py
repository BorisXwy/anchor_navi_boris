#!/usr/bin/env python3
"""Persistent node/edge memory for sub-instruction point navigation."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

import cv2
import numpy as np
from PIL import Image


PANORAMA_YAW_OFFSETS_DEG = (0, 60, 120, 180, 240, 300)
COMPLETION_PANORAMA_YAW_OFFSETS_DEG = (0, 45, 90, 135, 180, 225, 270, 315)
DEFAULT_ENVIRONMENT_QUERIES = (
    "doorway", "open door", "hallway", "corridor", "stairs", "stair landing",
    "table", "chair", "sofa", "bed", "cabinet", "counter", "rug", "carpet",
)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class VisualEmbedder(Protocol):
    name: str

    def embed(self, six_views: list[np.ndarray]) -> np.ndarray:
        ...


class EnvironmentSemanticExtractor(Protocol):
    name: str

    def extract(self, six_views, six_depths=None, sub_instruction=None) -> dict:
        ...


class CompactVisualEmbedder:
    """Deterministic 256-D panorama descriptor with no extra model weights.

    It combines an 8x8 RGB layout, HSV histograms and an edge-orientation
    histogram per view, then averages and L2-normalizes all six views. The
    protocol allows replacing it with DINO/CLIP/Qwen-VL embeddings later.
    """

    name = "compact_rgb_hsv_edge_v1"
    dimension = 256

    @staticmethod
    def embed_view(rgb):
        rgb = np.asarray(rgb, np.uint8)[..., :3]
        layout = cv2.resize(rgb, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32)
        layout = (layout.reshape(-1) / 255.0)  # 192 dimensions
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        histograms = []
        for channel, value_range in enumerate(((0, 180), (0, 256), (0, 256))):
            hist = cv2.calcHist([hsv], [channel], None, [16], list(value_range)).reshape(-1)
            histograms.append(hist / max(float(hist.sum()), 1.0))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude, angle = cv2.cartToPolar(dx, dy, angleInDegrees=True)
        edge_hist, _ = np.histogram(
            angle, bins=16, range=(0.0, 360.0), weights=magnitude)
        edge_hist = edge_hist.astype(np.float32)
        edge_hist /= max(float(edge_hist.sum()), 1.0)
        result = np.concatenate([layout, *histograms, edge_hist]).astype(np.float32)
        result /= max(float(np.linalg.norm(result)), 1e-8)
        return result

    def embed(self, six_views):
        if len(six_views) != 6:
            raise ValueError(f"visual embedding requires six views, got {len(six_views)}")
        embedding = np.mean([self.embed_view(view) for view in six_views], axis=0)
        embedding /= max(float(np.linalg.norm(embedding)), 1e-8)
        return embedding.astype(np.float32)


class DinoSamEnvironmentSemanticExtractor:
    """Use an injected DINO+SAM detector to describe a six-view node."""

    name = "dino_sam_rgb_only_environment_v2"

    def __init__(self, detector, queries=None, max_detections_per_view=12):
        self.detector = detector
        self.queries = list(queries or DEFAULT_ENVIRONMENT_QUERIES)
        self.max_detections_per_view = int(max_detections_per_view)

    @staticmethod
    def _instruction_queries(sub_instruction):
        if sub_instruction is None:
            return []
        value = (sub_instruction.to_dict() if hasattr(sub_instruction, "to_dict")
                 else dict(sub_instruction))
        landmark = str(value.get("landmark", "")).strip(" .")
        ignored = {"", "none", "unknown", "unspecified"}
        return ([landmark] if landmark.lower() not in ignored and
                len(landmark.split()) <= 8 else [])

    def extract(self, six_views, six_depths=None, sub_instruction=None):
        if len(six_views) != 6:
            raise ValueError(f"environment semantics require six views, got {len(six_views)}")
        if six_depths is not None:
            raise ValueError(
                "online node semantics are RGB-only; depth is evaluator-only")
        queries = list(dict.fromkeys(
            self.queries + self._instruction_queries(sub_instruction)))
        views = []
        label_scores = {}
        for index, rgb in enumerate(six_views):
            detections = sorted(
                self.detector.detect(rgb, queries),
                key=lambda detection: detection.score, reverse=True,
            )[:self.max_detections_per_view]
            records = [detection.rgb_prompt_record() for detection in detections]
            views.append({"view_index": index, "detections": records})
            for record in records:
                label = record["label"].lower()
                label_scores[label] = max(label_scores.get(label, 0.0), record["score"])
        return {
            "extractor": self.name,
            "queries": queries,
            "labels": sorted(label_scores),
            "label_scores": label_scores,
            "views": views,
        }


# Compatibility for readers importing the historical public class name.
GroundedSamEnvironmentSemanticExtractor = DinoSamEnvironmentSemanticExtractor


@dataclass
class NavigationNode:
    node_id: str
    node_kind: str
    # ``None`` in production RGB-only runs.  Historical/evaluation artifacts
    # may contain a simulator position, but online graph logic must not depend
    # on it under the rgb_only_v1 contract.
    position_xyz: Optional[list[float]]
    base_yaw_rad: Optional[float]
    created_global_step: int
    departure_purpose_sub_instruction: Optional[dict]
    six_views: list[dict]
    environment_semantics: dict
    visual_embedding: list[float]
    visual_embedding_model: str
    arrival_signal: Optional[str] = None
    sub_instruction_match: Optional[dict] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return _jsonable(asdict(self))


@dataclass
class NavigationEdge:
    edge_id: str
    source_node_id: str
    target_node_id: str
    departure_purpose_sub_instruction: dict
    action_history: list[dict]
    control_step_count: int
    traveled_distance_m: float
    edge_kind: str = "forward_navigation"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return _jsonable(asdict(self))


class NavigationGraphMemory:
    """Append-only episode graph persisted after every node or match update."""

    schema_version = 2

    def __init__(self, output_dir, semantic_extractor: EnvironmentSemanticExtractor,
                 visual_embedder: Optional[VisualEmbedder] = None,
                 policy_input_contract: str = "legacy_rgbd_geometry"):
        self.output_dir = Path(output_dir)
        self.nodes_dir = self.output_dir / "nodes"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        self.graph_path = self.output_dir / "navigation_graph.json"
        self.semantic_extractor = semantic_extractor
        self.visual_embedder = visual_embedder or CompactVisualEmbedder()
        self.policy_input_contract = str(policy_input_contract)
        self.nodes: list[NavigationNode] = []
        self.edges: list[NavigationEdge] = []

    def import_graph(self, graph_path):
        """Restore a previously persisted graph for a real continuation.

        A continuation run must keep the original node identities, incoming
        edge history, and keyframe/view paths.  Reconstructing only the pose
        in a fresh graph silently turns the resumed node into a new origin and
        lets the sequence judge classify an unexecuted hop.  This loader is
        deliberately model-free: it only deserializes records and copies the
        persisted visual artifacts into the new run directory.
        """
        if self.nodes or self.edges:
            raise RuntimeError("cannot import into a non-empty navigation graph")
        source_graph = Path(graph_path)
        payload = json.loads(source_graph.read_text())
        source_root = source_graph.parent
        raw_nodes = list(payload.get("nodes", []))
        raw_edges = list(payload.get("edges", []))
        if not raw_nodes:
            raise ValueError(f"navigation graph has no nodes: {source_graph}")

        def copy_artifact(relative_path):
            if not relative_path:
                return relative_path
            src = source_root / str(relative_path)
            dst = self.output_dir / str(relative_path)
            if src.exists() and src.resolve() != dst.resolve():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            return str(relative_path)

        for record in raw_nodes:
            node_record = dict(record)
            node_record["six_views"] = [
                dict(view, image_path=copy_artifact(view.get("image_path")))
                for view in node_record.get("six_views", [])]
            metadata = dict(node_record.get("metadata") or {})
            panorama = metadata.get("instruction_completion_panorama")
            if isinstance(panorama, dict):
                panorama = dict(panorama)
                panorama["views"] = [
                    dict(view, image_path=copy_artifact(view.get("image_path")))
                    for view in panorama.get("views", [])]
                metadata["instruction_completion_panorama"] = panorama
            node_record["metadata"] = metadata
            self.nodes.append(NavigationNode(**node_record))
        self.edges = [NavigationEdge(**dict(edge)) for edge in raw_edges]
        self.save()
        return self.nodes[-1]

    @staticmethod
    def _sub_instruction_dict(sub_instruction):
        if sub_instruction is None:
            return None
        if hasattr(sub_instruction, "to_dict"):
            return _jsonable(sub_instruction.to_dict())
        return _jsonable(dict(sub_instruction))

    def _save_views(self, node_id, six_views, base_yaw_rad):
        if len(six_views) != 6:
            raise ValueError(f"node requires six views, got {len(six_views)}")
        node_dir = self.nodes_dir / node_id
        node_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for index, (rgb, offset) in enumerate(zip(six_views, PANORAMA_YAW_OFFSETS_DEG)):
            path = node_dir / f"view_{index}_{offset:03d}deg.jpg"
            Image.fromarray(np.asarray(rgb, np.uint8)[..., :3]).save(path, quality=92)
            record = {
                "view_index": index,
                "relative_yaw_deg": offset,
                "image_path": str(path.relative_to(self.output_dir)),
            }
            if base_yaw_rad is not None:
                record["absolute_yaw_rad"] = float(
                    (base_yaw_rad + math.radians(offset) + math.pi) %
                    (2 * math.pi) - math.pi)
            records.append(record)
        return records

    def _save_completion_views(self, node_id, eight_views, base_yaw_rad):
        """Persist an optional eight-compass panorama for completion judgments."""
        if len(eight_views) != 8:
            raise ValueError(
                f"instruction completion requires eight views, got {len(eight_views)}")
        node_dir = self.nodes_dir / node_id
        node_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for index, (rgb, offset) in enumerate(zip(
                eight_views, COMPLETION_PANORAMA_YAW_OFFSETS_DEG)):
            path = node_dir / f"completion_view_{index}_{offset:03d}deg.jpg"
            Image.fromarray(np.asarray(rgb, np.uint8)[..., :3]).save(path, quality=92)
            record = {
                "view_index": index,
                "relative_yaw_deg": offset,
                "image_path": str(path.relative_to(self.output_dir)),
            }
            if base_yaw_rad is not None:
                record["absolute_yaw_rad"] = float(
                    (base_yaw_rad + math.radians(offset) + math.pi) %
                    (2 * math.pi) - math.pi)
            records.append(record)
        return {
            "profile": "eight_compass_45deg_v1",
            "direction_labels": [
                "front", "front_left", "left", "rear_left", "rear",
                "rear_right", "right", "front_right",
            ],
            "views": records,
        }

    def _add_node(self, node_kind, position_xyz, base_yaw_rad, global_step,
                  six_views, six_depths, sub_instruction, arrival_signal,
                  metadata=None, completion_views=None):
        node_id = f"node_{len(self.nodes):04d}"
        sub_instruction_dict = self._sub_instruction_dict(sub_instruction)
        node_metadata = _jsonable(metadata or {})
        if completion_views is not None:
            node_metadata["instruction_completion_panorama"] = (
                self._save_completion_views(
                    node_id, completion_views, base_yaw_rad))
        node = NavigationNode(
            node_id=node_id,
            node_kind=str(node_kind),
            position_xyz=(
                None if position_xyz is None else
                np.asarray(position_xyz, np.float32).tolist()),
            base_yaw_rad=(None if base_yaw_rad is None else float(base_yaw_rad)),
            created_global_step=int(global_step),
            departure_purpose_sub_instruction=sub_instruction_dict,
            six_views=self._save_views(node_id, six_views, base_yaw_rad),
            environment_semantics=_jsonable(self.semantic_extractor.extract(
                six_views, six_depths, sub_instruction)),
            visual_embedding=self.visual_embedder.embed(six_views).tolist(),
            visual_embedding_model=str(self.visual_embedder.name),
            arrival_signal=arrival_signal,
            metadata=node_metadata,
        )
        self.nodes.append(node)
        return node

    def add_origin_node(self, position_xyz, base_yaw_rad, global_step,
                        six_views, six_depths=None, metadata=None,
                        completion_views=None):
        if self.nodes:
            raise RuntimeError("origin node can only be added to an empty graph")
        node = self._add_node(
            "episode_origin", position_xyz, base_yaw_rad, global_step,
            six_views, six_depths, None, None, metadata, completion_views)
        self.save()
        return node

    def add_arrival_node(self, position_xyz, base_yaw_rad, global_step,
                         six_views, sub_instruction, action_history,
                         arrival_signal, six_depths=None, metadata=None):
        """Compatibility helper for a successful point-navigation stop."""
        if not arrival_signal:
            raise ValueError("arrival_signal must be present for an arrival node")
        return self.add_navigation_stop_node(
            position_xyz=position_xyz, base_yaw_rad=base_yaw_rad,
            global_step=global_step, six_views=six_views,
            sub_instruction=sub_instruction, action_history=action_history,
            arrival_signal=arrival_signal, six_depths=six_depths,
            metadata=metadata)

    def add_navigation_stop_node(self, position_xyz, base_yaw_rad, global_step,
                                 six_views, sub_instruction, action_history,
                                 arrival_signal=None, six_depths=None,
                                 metadata=None, edge_kind="forward_navigation",
                                 edge_metadata=None, completion_views=None):
        """Record every executor stop; the signal distinguishes true arrival."""
        if not self.nodes:
            raise RuntimeError("add an origin node before the first navigation-stop node")
        source = self.nodes[-1]
        node = self._add_node(
            ("point_navigation_arrival" if arrival_signal
             else "point_navigation_stop"),
            position_xyz, base_yaw_rad, global_step,
            six_views, six_depths, sub_instruction, arrival_signal, metadata,
            completion_views)
        history = _jsonable(action_history)
        edge = NavigationEdge(
            edge_id=f"edge_{len(self.edges):04d}",
            source_node_id=source.node_id,
            target_node_id=node.node_id,
            departure_purpose_sub_instruction=(
                self._sub_instruction_dict(sub_instruction) or {}),
            action_history=history,
            control_step_count=len(history),
            traveled_distance_m=float(sum(
                float(action.get("moved_m", 0.0)) for action in history)),
            edge_kind=str(edge_kind),
            metadata=_jsonable(edge_metadata or {}),
        )
        self.edges.append(edge)
        self.save()
        return node, edge

    def set_sub_instruction_match(self, node_id, result):
        matches = [node for node in self.nodes if node.node_id == node_id]
        if not matches:
            raise KeyError(node_id)
        matches[0].sub_instruction_match = _jsonable(result)
        self.save()

    def set_node_metadata(self, node_id, values):
        node = self.get_node(node_id)
        node.metadata.update(_jsonable(values))
        self.save()

    def add_loop_closure_edge(self, reference_node_id, revisit_node_id,
                              metadata=None):
        """Link a newly observed revisit node to its older logical identity."""
        reference = self.get_node(reference_node_id)
        revisit = self.get_node(revisit_node_id)
        edge = NavigationEdge(
            edge_id=f"edge_{len(self.edges):04d}",
            source_node_id=reference.node_id,
            target_node_id=revisit.node_id,
            departure_purpose_sub_instruction={},
            action_history=[], control_step_count=0,
            traveled_distance_m=0.0,
            edge_kind="node_revisit_loop_closure",
            metadata=_jsonable(metadata or {}),
        )
        self.edges.append(edge)
        self.save()
        return edge

    def get_node(self, node_id):
        matches = [node for node in self.nodes if node.node_id == str(node_id)]
        if not matches:
            raise KeyError(f"unknown navigation node {node_id!r}")
        return matches[0]

    def load_node_views(self, node_id):
        """Load the six persisted RGB views for a node in panorama order."""
        node = self.get_node(node_id)
        records = sorted(node.six_views, key=lambda item: item["view_index"])
        if len(records) != 6:
            raise ValueError(
                f"node {node_id} has {len(records)} views; expected six")
        return [np.asarray(Image.open(
            self.output_dir / record["image_path"]).convert("RGB"))
            for record in records]

    def get_edge(self, source_node_id, target_node_id):
        matches = [edge for edge in self.edges
                   if edge.source_node_id == str(source_node_id) and
                   edge.target_node_id == str(target_node_id)]
        if not matches:
            raise KeyError(f"no edge {source_node_id!r} -> {target_node_id!r}")
        return matches[-1]

    def get_edge_by_id(self, edge_id):
        matches = [edge for edge in self.edges
                   if edge.edge_id == str(edge_id)]
        if not matches:
            raise KeyError(f"unknown navigation edge {edge_id!r}")
        return matches[-1]

    def load_edge_keyframes(self, edge_or_id):
        """Load the persisted RGB storyboard frames for one real edge."""
        edge = (self.get_edge_by_id(edge_or_id)
                if isinstance(edge_or_id, str) else edge_or_id)
        records = sorted(
            list((edge.metadata or {}).get("edge_keyframes", [])),
            key=lambda item: int(item.get("keyframe_index", 0)))
        if not records:
            raise RuntimeError(f"stored edge {edge.edge_id} has no keyframes")
        images = []
        for record in records:
            path = self.output_dir / str(record.get("image_path", ""))
            if not path.exists():
                # Forward-edge keyframes are owned by the episode renderer
                # and historically stored beside ``navigation_graph/``;
                # imported/unit graphs may instead keep them inside it.
                # Accept both persisted layouts without rewriting evidence.
                alternate = self.output_dir.parent / str(
                    record.get("image_path", ""))
                if alternate.exists():
                    path = alternate
            if not path.exists():
                raise FileNotFoundError(
                    f"missing stored edge keyframe: {path}")
            images.append(np.asarray(Image.open(path).convert("RGB")))
        return images

    def predecessor(self, node_id):
        incoming = [edge for edge in self.edges if edge.target_node_id == str(node_id)]
        if not incoming:
            return None, None
        edge = incoming[-1]
        return self.get_node(edge.source_node_id), edge

    def ancestor_path(self, source_node_id, target_node_id, max_hops=None):
        """Return ``[source, ..., target]`` following stored incoming edges."""
        source_node_id = str(source_node_id)
        target_node_id = str(target_node_id)
        self.get_node(source_node_id)
        self.get_node(target_node_id)
        path = [source_node_id]
        seen = {source_node_id}
        current = source_node_id
        while current != target_node_id:
            predecessor, _ = self.predecessor(current)
            if predecessor is None:
                raise ValueError(
                    f"{target_node_id} is not an ancestor of {source_node_id}")
            current = predecessor.node_id
            if current in seen:
                raise RuntimeError("cycle detected while resolving ancestor path")
            path.append(current)
            seen.add(current)
            if max_hops is not None and len(path) - 1 > int(max_hops):
                raise ValueError(
                    f"backtrack path exceeds max_hops={max_hops}: {path}")
        return path

    def load_node_views(self, node_or_id):
        node = (self.get_node(node_or_id) if isinstance(node_or_id, str)
                else node_or_id)
        views = []
        for record in sorted(node.six_views, key=lambda item: item["view_index"]):
            path = self.output_dir / record["image_path"]
            if not path.exists():
                raise FileNotFoundError(f"missing stored node view: {path}")
            views.append(np.asarray(Image.open(path).convert("RGB")))
        if len(views) != 6:
            raise RuntimeError(f"stored node {node.node_id} does not have six views")
        return views

    def load_node_completion_views(self, node_or_id):
        """Load a node's optional completion-only eight-compass panorama."""
        node = (self.get_node(node_or_id) if isinstance(node_or_id, str)
                else node_or_id)
        panorama = node.metadata.get("instruction_completion_panorama")
        if not panorama:
            raise RuntimeError(
                f"stored node {node.node_id} has no eight-view completion panorama")
        records = sorted(
            panorama.get("views", []), key=lambda item: item["view_index"])
        if len(records) != 8:
            raise RuntimeError(
                f"stored node {node.node_id} has {len(records)} completion views; "
                "expected eight")
        views = []
        for record in records:
            path = self.output_dir / record["image_path"]
            if not path.exists():
                raise FileNotFoundError(f"missing completion panorama view: {path}")
            views.append(np.asarray(Image.open(path).convert("RGB")))
        return views

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "policy_input_contract": self.policy_input_contract,
            "semantic_extractor": str(self.semantic_extractor.name),
            "visual_embedder": str(self.visual_embedder.name),
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    def summary(self):
        return {
            "graph_path": str(self.graph_path),
            "policy_input_contract": self.policy_input_contract,
            "node_count": len(self.nodes),
            "navigation_stop_node_count": sum(
                node.node_kind in {"point_navigation_arrival", "point_navigation_stop"}
                for node in self.nodes),
            "arrival_node_count": sum(node.node_kind == "point_navigation_arrival"
                                      for node in self.nodes),
            "edge_count": len(self.edges),
            "latest_node_id": self.nodes[-1].node_id if self.nodes else None,
            "semantic_extractor": str(self.semantic_extractor.name),
            "visual_embedder": str(self.visual_embedder.name),
        }

    def save(self):
        temporary = self.graph_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        temporary.replace(self.graph_path)
