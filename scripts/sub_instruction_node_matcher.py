#!/usr/bin/env python3
"""Judge whether a stored navigation node belongs to a sub-instruction."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import numpy as np


STOPWORDS = {
    "a", "an", "and", "at", "by", "for", "from", "in", "into", "of", "on",
    "or", "the", "then", "to", "toward", "towards", "walkable", "floor", "free",
    "region", "target", "space", "current", "after", "before", "through", "along",
}


def _tokens(value):
    return {token for token in re.findall(r"[a-z0-9]+", str(value).lower())
            if len(token) > 1 and token not in STOPWORDS}


def _flatten_semantic_text(value):
    if isinstance(value, dict):
        return " ".join(_flatten_semantic_text(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten_semantic_text(item) for item in value)
    return "" if isinstance(value, (int, float, bool)) or value is None else str(value)


@dataclass
class SubInstructionMatchResult:
    belongs: bool
    score: float
    threshold: float
    purpose_score: float
    environment_semantic_score: float
    visual_evidence_score: float
    evidence: list[str]

    def to_dict(self):
        return asdict(self)


class SubInstructionNodeMatcher:
    """Auditable default matcher with an optional learned visual scorer.

    Association with the node's stored departure purpose is primary. Current
    six-view semantics and the validity of the stored panorama embedding supply
    independent arrival evidence. ``visual_scorer`` can later be a CLIP/DINO or
    VLM adapter accepting ``(visual_embedding, sub_instruction_dict)``.
    """

    def __init__(self, threshold=0.60,
                 visual_scorer: Optional[Callable[[np.ndarray, dict], float]] = None):
        self.threshold = float(threshold)
        self.visual_scorer = visual_scorer

    @staticmethod
    def _mapping(value):
        if value is None:
            return {}
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if hasattr(value, "__dict__"):
            return dict(value.__dict__)
        return dict(value)

    def match(self, node, sub_instruction):
        node_value = self._mapping(node)
        wanted = self._mapping(sub_instruction)
        stored = self._mapping(
            node_value.get("departure_purpose_sub_instruction"))
        evidence = []

        wanted_id = wanted.get("sub_instruction_id", wanted.get("stage_id"))
        stored_id = stored.get("sub_instruction_id", stored.get("stage_id"))
        id_match = wanted_id is not None and stored_id is not None and wanted_id == stored_id
        wanted_text = _tokens(wanted.get("navigation_instruction", ""))
        stored_text = _tokens(stored.get("navigation_instruction", ""))
        text_union = wanted_text | stored_text
        text_similarity = (len(wanted_text & stored_text) / len(text_union)
                           if text_union else 0.0)
        form_match = bool(wanted.get("form") and
                          wanted.get("form") == stored.get("form"))
        purpose_score = max(1.0 if id_match else 0.0, text_similarity,
                            0.45 if form_match else 0.0)
        if id_match:
            evidence.append(f"stored departure purpose has sub_instruction_id={wanted_id}")
        if text_similarity > 0:
            evidence.append(f"purpose text token similarity={text_similarity:.3f}")

        environment_semantics = node_value.get("environment_semantics", {})
        # Only observed labels/detections count as evidence. Prompt queries are
        # deliberately excluded: asking DINO+SAM for "doorway" must not be
        # mistaken for actually observing one.
        if isinstance(environment_semantics, dict):
            observed_semantics = {
                "labels": environment_semantics.get("labels", []),
                "views": [{"detections": view.get("detections", [])}
                          for view in environment_semantics.get("views", [])],
            }
        else:
            observed_semantics = environment_semantics
        semantic_text = _flatten_semantic_text(observed_semantics)
        semantic_tokens = _tokens(semantic_text)
        expected_tokens = _tokens(" ".join(str(wanted.get(key, "")) for key in (
            "landmark", "semantic_spatial_target", "spatial_relation",
            "visual_arrival_evidence", "form")))
        semantic_score = (len(expected_tokens & semantic_tokens) / len(expected_tokens)
                          if expected_tokens else 0.0)
        if expected_tokens & semantic_tokens:
            evidence.append("environment semantics overlap: " + ", ".join(
                sorted(expected_tokens & semantic_tokens)))

        embedding = np.asarray(node_value.get("visual_embedding", []), np.float32)
        if self.visual_scorer is not None and embedding.size:
            visual_score = float(np.clip(
                self.visual_scorer(embedding, wanted), 0.0, 1.0))
            evidence.append(f"replaceable visual scorer={visual_score:.3f}")
        else:
            six_views = node_value.get("six_views", [])
            norm = float(np.linalg.norm(embedding)) if embedding.size else 0.0
            visual_score = float(len(six_views) == 6 and np.isfinite(norm) and norm > 0.5)
            if visual_score:
                evidence.append("six-view visual embedding is present and valid")

        score = (0.55 * purpose_score + 0.35 * semantic_score +
                 0.10 * visual_score)
        belongs = bool(score >= self.threshold)
        if not belongs:
            evidence.append("combined stored-node evidence is below threshold")
        return SubInstructionMatchResult(
            belongs=belongs, score=round(score, 6), threshold=self.threshold,
            purpose_score=round(purpose_score, 6),
            environment_semantic_score=round(semantic_score, 6),
            visual_evidence_score=round(visual_score, 6), evidence=evidence)


def node_belongs_to_sub_instruction(node, sub_instruction, threshold=0.60):
    return SubInstructionNodeMatcher(threshold).match(node, sub_instruction)
