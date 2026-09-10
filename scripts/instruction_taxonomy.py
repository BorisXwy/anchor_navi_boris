#!/usr/bin/env python3
"""Defined R2R segment forms and deterministic definition-driven decomposition."""

import re

from point_selection_strategies import STRATEGIES


ACTION = (r"go|walk|turn|take|head|continue|proceed|stop|wait|exit|enter|pass|"
          r"cross|veer|bear|make|keep|follow|move|travel|climb|descend|leave|get")
SPLIT_RE = re.compile(
    rf"\s*(?:[.;!?]+|,\s*(?=(?:then\s+)?(?:{ACTION})\b)|"
    rf"\b(?:and\s+then|then)\b|\band\s+(?=(?:{ACTION})\b))\s*", re.I)


FORM_DEFINITIONS = {
    "EXIT_REGION": {
        "pattern": r"\b(exit|exiting|leave|leaving|outside)\b|\b(go|walk|head|get|move)\s+(?:straight\s+)?out\b",
        "definition": "cross from the current region through its exit portal",
        "spatial_target": "free walkable floor immediately beyond the current region's exit portal",
        "arrival": "camera center has crossed the portal plane",
        "forbidden": "floor still inside the source region, door leaf, wall, or threshold edge",
    },
    "ENTER_REGION": {
        "pattern": r"\b(enter|entering)\b|\b(go|walk|head|move|get)\s+(?:straight\s+)?into\b",
        "definition": "cross a portal into the named destination region",
        "spatial_target": "free walkable floor immediately inside the destination region",
        "arrival": "camera is inside the destination beyond its portal",
        "forbidden": "source-side floor, wall, door leaf, and objects inside the destination",
    },
    "TURN_LEFT": {
        "pattern": r"\b(turn|veer|bear|make|take|go|head)\b.{0,18}\bleft\b",
        "definition": "change heading left and commit to the resulting navigable opening/path",
        "spatial_target": "free floor in the left-side corridor, doorway, or open route after the turn",
        "arrival": "agent is aligned with and has entered the left-side route",
        "forbidden": "left wall or an object selected merely to induce rotation",
    },
    "TURN_RIGHT": {
        "pattern": r"\b(turn|veer|bear|make|take|go|head)\b.{0,18}\bright\b",
        "definition": "change heading right and commit to the resulting navigable opening/path",
        "spatial_target": "free floor in the right-side corridor, doorway, or open route after the turn",
        "arrival": "agent is aligned with and has entered the right-side route",
        "forbidden": "right wall or an object selected merely to induce rotation",
    },
    "TURN_AROUND": {
        "pattern": r"\b(turn|veer|bear|make|take)\b.{0,20}\b(around|u[- ]?turn|180)\b",
        "definition": "reverse heading and commit to the route behind",
        "spatial_target": "free floor or an opening aligned with the reverse heading",
        "arrival": "agent faces and begins entering the reverse route",
        "forbidden": "walls and non-navigable pixels behind",
    },
    "TURN_TO_LANDMARK": {
        # A turn toward a named landmark is semantically different from a
        # generic approach: the immediate target is the landmark's bearing,
        # which may be lateral or behind the arrival heading.  Exclude the
        # explicit left/right/around commands so those retain their stronger
        # directional forms below/above in the priority list.
        "pattern": r"\bturn\s+(?:towards?|to)\s+(?!(?:the\s+)?(?:left|right|around|back|rear)\b)[a-z][^,.;!?]*",
        "definition": "rotate toward the explicitly named landmark and align with its bearing",
        "spatial_target": "free floor on the landmark bearing, at a safe offset and with a connected route",
        "arrival": "heading is aligned to the named landmark and a walkable floor ray points toward it",
        "forbidden": "unrelated open floor, the landmark surface itself, walls, and a ray that points away from the named landmark",
    },
    "VERTICAL_UP": {
        "pattern": r"\b(climb(?:ing|ed)?|ascend(?:ing|ed)?|upstairs)\b|\bup\s+(?:the\s+|a\s+)?(?:(?:flight\s+of\s+)?(?:stairs?|steps?)|staircase)\b",
        "definition": "move to the next higher navigable level using stairs",
        "spatial_target": "walkable stair/landing region above the current level",
        "arrival": "agent reaches the upper landing",
        "forbidden": "railings, stair walls, and void beyond steps",
    },
    "VERTICAL_DOWN": {
        "pattern": r"\b(descend(?:ing|ed)?|downstairs)\b|\bdown\s+(?:the\s+|a\s+)?(?:(?:flight\s+of\s+)?(?:stairs?|steps?)|staircase)\b",
        "definition": "move to the next lower navigable level using stairs",
        "spatial_target": "walkable stair/landing region below the current level",
        "arrival": "agent reaches the lower landing",
        "forbidden": "railings, stair walls, and void beyond steps",
    },
    "PASS_LANDMARK": {
        "pattern": r"\b(pass|past|passed|passing)\b",
        "definition": "continue until the referenced landmark lies behind the agent",
        "spatial_target": "free floor beyond the referenced landmark along the route",
        "arrival": "agent has passed the landmark rather than merely approached it",
        "forbidden": "floor before/beside the landmark and pixels on the landmark",
    },
    "CIRCUMNAVIGATE": {
        "pattern": r"\b(around|circle)\b|\b(right|left)\s+side\s+of\b",
        "definition": "go around an obstacle on the stated side",
        "spatial_target": "free floor on the instructed side and beyond the obstacle",
        "arrival": "agent clears the obstacle while preserving the requested side",
        "forbidden": "the obstacle surface and the wrong side",
    },
    "CROSS_SPACE": {
        "pattern": r"\b(across|cross|crossing)\b",
        "definition": "traverse a bounded open region to its far side",
        "spatial_target": "free floor on the far side of the referenced region",
        "arrival": "agent reaches the opposite boundary/side",
        "forbidden": "near-side floor and non-walkable interior objects",
    },
    "BETWEEN_OBJECTS": {
        "pattern": r"\bbetween\b",
        "definition": "move through the navigable gap separating referenced objects",
        "spatial_target": "free floor centered in or just beyond the stated gap",
        "arrival": "agent enters or clears the gap",
        "forbidden": "object surfaces and gaps narrower than navigable clearance",
    },
    "SELECT_PORTAL": {
        "pattern": r"\b(first|second|third|last|next|nearest)\s+(open\s+)?(door|doorway|opening)\b|\b(door|doorway)\s+on\s+(the|your)\s+(left|right)\b",
        "definition": "select the portal identified by order or side",
        "spatial_target": "free floor just through the specified portal",
        "arrival": "agent crosses the selected, not an adjacent, portal",
        "forbidden": "other portals, door surface, frame, and wall",
    },
    "TRAVERSE_PORTAL_REGION": {
        "pattern": r"\bthrough\b",
        "definition": "traverse the referenced portal or intermediate region",
        "spatial_target": "free floor beyond the referenced portal/region",
        "arrival": "agent clears the portal or intermediate region",
        "forbidden": "near-side floor, wall, door leaf, and obstacle pixels",
    },
    "FOLLOW_PATH_BOUNDARY": {
        "pattern": r"\b(follow|along)\b|\bkeep\b.{0,20}\b(wall|hall|path|corridor)\b",
        "definition": "continue along a path or boundary while preserving adjacency",
        "spatial_target": "distant free floor along the stated path/boundary",
        "arrival": "agent makes progress while retaining the referenced boundary relation",
        "forbidden": "crossing the boundary or selecting wall pixels",
    },
    "ADVANCE_STRAIGHT": {
        "pattern": r"\b(straight|forward|ahead)\b|\bkeep (walking|going)\b|\b(?:go|walk|head|continue)\s+down\s+(?:the\s+)?(?:hall|hallway|corridor)\b",
        "definition": "advance along the current route without choosing a new branch",
        "spatial_target": "distant visible free floor on the current route axis",
        "arrival": "agent makes forward progress to the next decision area",
        "forbidden": "side branches, near-camera floor, walls, and objects",
    },
    "APPROACH_LANDMARK": {
        "pattern": r"\b(towards?|until|near|beside)\b|\bnext to\b|\b(go|walk|head|move)\s+to\b",
        "definition": "approach the referenced landmark without colliding with it",
        "spatial_target": "free floor near the landmark at a safe navigable offset",
        "arrival": "landmark is near and clearly visible at the requested relation",
        "forbidden": "pixels on the landmark and floor beyond it when not requested",
    },
    "STOP_WAIT": {
        "pattern": r"\b(stop|wait|stand|remain)\b",
        "definition": "terminate at the stated spatial relation to a landmark/region",
        "spatial_target": "free floor satisfying the stated final landmark relation",
        "arrival": "agent occupies that free floor and the relation is visually satisfied",
        "forbidden": "landmark surfaces and unrelated nearby floor",
    },
}

COMPILED = [(name, re.compile(value["pattern"], re.I))
            for name, value in FORM_DEFINITIONS.items()]
PRIMARY_PRIORITY = [
    "STOP_WAIT", "VERTICAL_UP", "VERTICAL_DOWN", "EXIT_REGION", "ENTER_REGION",
    "TURN_AROUND", "TURN_LEFT", "TURN_RIGHT", "TURN_TO_LANDMARK", "SELECT_PORTAL", "PASS_LANDMARK",
    "CIRCUMNAVIGATE", "BETWEEN_OBJECTS", "CROSS_SPACE", "TRAVERSE_PORTAL_REGION",
    "FOLLOW_PATH_BOUNDARY", "ADVANCE_STRAIGHT", "APPROACH_LANDMARK",
]


def split_instruction(text):
    return [piece.strip(" ,") for piece in SPLIT_RE.split(text) if piece.strip(" ,")]


def decompose_by_definition(text):
    stages = []
    for clause in split_instruction(text):
        labels = [name for name, pattern in COMPILED if pattern.search(clause)]
        labels.sort(key=lambda name: PRIMARY_PRIORITY.index(name))
        form = labels[0] if labels else "OTHER"
        definition = FORM_DEFINITIONS.get(form, {
            "definition": "unclassified navigation or observation clause",
            "spatial_target": "free floor inferred from the full clause and surrounding views",
            "arrival": "clause-specific relation is satisfied",
            "forbidden": "walls, objects, and non-walkable pixels",
        })
        stages.append({
            "stage_id": len(stages), "source_clause": clause, "form": form,
            "secondary_forms": labels[1:], "definition": definition["definition"],
            "semantic_spatial_target": definition["spatial_target"],
            "visual_arrival_evidence": definition["arrival"],
            "forbidden_target": definition["forbidden"],
            # Compatibility with NavigationVLMHarness.
            "navigation_instruction": clause, "landmark": clause,
            "completion_cue": definition["arrival"], "spatial_relation": form.lower(),
            "point_selection_strategy": STRATEGIES.get(form, STRATEGIES["OTHER"]),
        })
    return stages
