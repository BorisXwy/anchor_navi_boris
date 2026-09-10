#!/usr/bin/env python3
"""Instruction decomposition with a stable ``SubInstruction`` contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Optional

from instruction_taxonomy import decompose_by_definition


@dataclass
class SubInstruction:
    """One ordered, visually grounded unit of an R2R instruction.

    ``stage_id`` is retained in :meth:`to_stage_dict` only as a compatibility
    alias for the existing point selector. New code should use
    ``sub_instruction_id`` and call these objects sub-instructions.
    """

    sub_instruction_id: int
    navigation_instruction: str
    landmark: str
    completion_cue: str
    semantic_spatial_target: str
    spatial_relation: str
    visual_arrival_evidence: str
    forbidden_target: str
    source_clause: str = ""
    form: str = "UNCLASSIFIED"
    secondary_forms: list[str] = field(default_factory=list)
    definition: str = ""
    point_selection_strategy: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], index: Optional[int] = None):
        value = dict(value)
        # Frozen decomposition artifacts can predate taxonomy fixes.  In
        # particular, a location preamble ("start in ...") used to make an
        # otherwise explicit exit/enter clause ``OTHER``.  Re-type only
        # unclassified mappings from their own instruction text; preserve all
        # VLM-authored spatial targets and completion evidence.  This is a
        # grammar-level migration and never consults episode IDs, images,
        # demonstrations, or execution outcomes.
        original_form = str(value.get("form", "UNCLASSIFIED")).upper()
        if original_form in {"OTHER", "UNCLASSIFIED", ""}:
            typed = decompose_by_definition(str(
                value.get("navigation_instruction", "")))
            actionable = [item for item in typed if str(
                item.get("form", "OTHER")).upper() != "OTHER"]
            if actionable:
                typed_base = actionable[0]
                for key in ("source_clause", "form", "secondary_forms",
                            "definition", "point_selection_strategy"):
                    if key in typed_base:
                        value[key] = typed_base[key]
                metadata = dict(value.get("metadata", {}))
                metadata["form_normalization"] = {
                    "from": original_form or "UNCLASSIFIED",
                    "to": str(typed_base.get("form", "OTHER")),
                    "policy": "first_actionable_taxonomy_clause",
                }
                value["metadata"] = metadata
        # A frozen/VLM stage can deliberately keep a compound maneuver such
        # as "turn left and head up the stairs" as one semantic unit.  Its
        # terminal completion cue (reach the top/bottom landing) governs the
        # physical endpoint, so a leading TURN form must not let the judge
        # accept an intermediate stair point.  Promote the vertical endpoint
        # to the primary form while retaining the turn as a secondary routing
        # constraint.  This is grammar/cue driven and never consults an EP,
        # image, demonstration, or outcome.
        navigation_text = str(value.get("navigation_instruction", ""))
        endpoint_text = " ".join(str(value.get(key, "")) for key in (
            "completion_cue", "semantic_spatial_target",
            "visual_arrival_evidence"))
        typed_parts = [item for item in decompose_by_definition(navigation_text)
                       if str(item.get("form", "OTHER")).upper() != "OTHER"]
        typed_forms = [str(item.get("form", "OTHER")).upper()
                       for item in typed_parts]
        vertical_terminal = None
        if ("VERTICAL_UP" in typed_forms and
                any(token in endpoint_text.lower() for token in (
                    "top", "upper landing", "higher level"))):
            vertical_terminal = "VERTICAL_UP"
        elif ("VERTICAL_DOWN" in typed_forms and
              any(token in endpoint_text.lower() for token in (
                  "bottom", "lower landing", "lower level"))):
            vertical_terminal = "VERTICAL_DOWN"
        if vertical_terminal is not None and original_form != vertical_terminal:
            vertical_part = typed_parts[typed_forms.index(vertical_terminal)]
            old_primary = str(value.get("form", original_form)).upper()
            secondary = [old_primary, *value.get("secondary_forms", [])]
            value["form"] = vertical_terminal
            value["secondary_forms"] = list(dict.fromkeys(
                item for item in secondary
                if item and item not in {"OTHER", "UNCLASSIFIED", vertical_terminal}))
            for key in ("definition", "point_selection_strategy"):
                if key in vertical_part:
                    value[key] = vertical_part[key]
            metadata = dict(value.get("metadata", {}))
            metadata["terminal_form_normalization"] = {
                "from": old_primary,
                "to": vertical_terminal,
                "policy": "compound_vertical_completion_cue_has_endpoint_priority",
            }
            value["metadata"] = metadata
        sub_id = int(value.get(
            "sub_instruction_id", value.get("stage_id", 0 if index is None else index)))
        known = {
            "sub_instruction_id", "stage_id", "navigation_instruction", "landmark",
            "completion_cue", "semantic_spatial_target", "spatial_relation",
            "visual_arrival_evidence", "forbidden_target", "source_clause", "form",
            "secondary_forms", "definition", "point_selection_strategy", "metadata",
        }
        metadata = dict(value.get("metadata", {}))
        metadata.update({key: item for key, item in value.items() if key not in known})
        return cls(
            sub_instruction_id=sub_id,
            navigation_instruction=str(value.get("navigation_instruction", "")).strip(),
            landmark=str(value.get("landmark", "unspecified")).strip(),
            completion_cue=str(value.get("completion_cue", "")).strip(),
            semantic_spatial_target=str(
                value.get("semantic_spatial_target", "walkable floor at the target")).strip(),
            spatial_relation=str(value.get("spatial_relation", "at the target")).strip(),
            visual_arrival_evidence=str(
                value.get("visual_arrival_evidence", "target transition is complete")).strip(),
            forbidden_target=str(
                value.get("forbidden_target", "non-walkable regions")).strip(),
            source_clause=str(value.get("source_clause", "")).strip(),
            form=str(value.get("form", "UNCLASSIFIED")).strip() or "UNCLASSIFIED",
            secondary_forms=list(value.get("secondary_forms", [])),
            definition=str(value.get("definition", "")).strip(),
            point_selection_strategy=dict(value.get("point_selection_strategy", {})),
            metadata=metadata,
        )

    def to_dict(self):
        return asdict(self)

    def to_stage_dict(self):
        """Return the selector-compatible representation."""
        result = self.to_dict()
        result["stage_id"] = self.sub_instruction_id
        return result


class InstructionDecomposer:
    """Turn one route instruction into typed :class:`SubInstruction` objects.

    The VLM harness is injected so DeepSeek, Ollama/LLaVA, a heuristic backend,
    or a future planner can be exchanged without changing graph/navigation code.
    """

    def __init__(
            self, vlm_harness=None,
            definition_decomposer: Callable[[str], list[dict]] = decompose_by_definition):
        self.vlm_harness = vlm_harness
        self.definition_decomposer = definition_decomposer

    def decompose(self, instruction: str, limit: Optional[int] = None):
        instruction = str(instruction).strip()
        if not instruction:
            raise ValueError("instruction must be non-empty")
        if self.vlm_harness is not None:
            raw = self.vlm_harness.decompose_instruction(instruction)
        else:
            raw = self.definition_decomposer(instruction)
        if not raw:
            raise RuntimeError("Instruction decomposition produced no sub-instructions")

        sub_instructions = []
        for raw_index, value in enumerate(raw):
            enriched = dict(value)
            navigation_instruction = str(
                enriched.get("navigation_instruction", instruction)).strip()
            typed = self.definition_decomposer(navigation_instruction)
            # VLM decomposers sometimes return a fluent sentence containing
            # multiple independently verifiable actions as one item (for
            # example, "enter the kitchen and walk along the counter").  A
            # single primary form then discards the endpoint of the latter
            # action.  The deterministic taxonomy already splits only at an
            # explicit connector followed by another navigation verb, so use
            # those typed clauses as separate sub-instructions.  This is a
            # grammar-level rule and does not depend on episode IDs, landmark
            # names, demonstrations, or hidden path geometry.
            actionable_parts = [
                item for item in typed
                if str(item.get("form", "OTHER")).upper() != "OTHER"
            ]
            if self.vlm_harness is not None and len(actionable_parts) > 1:
                parent_metadata = dict(enriched.get("metadata", {}))
                parent_metadata.update({
                    "vlm_parent_raw_index": raw_index,
                    "vlm_parent_navigation_instruction": navigation_instruction,
                    "compound_action_count": len(actionable_parts),
                })
                for part_index, typed_part in enumerate(actionable_parts):
                    part = dict(typed_part)
                    clause = str(typed_part.get("source_clause", "")).strip()
                    part.update({
                        "navigation_instruction": clause,
                        "landmark": clause,
                        "completion_cue": typed_part.get(
                            "visual_arrival_evidence", ""),
                        "metadata": dict(
                            parent_metadata,
                            compound_part_index=part_index,
                        ),
                    })
                    part["sub_instruction_id"] = len(sub_instructions)
                    sub_instructions.append(SubInstruction.from_mapping(
                        part, len(sub_instructions)))
                continue
            if typed:
                # A VLM may merge an initial-location preamble with the first
                # actionable clause (for example, ``Start in the room and
                # head towards the outside door``).  Taking typed[0] would
                # classify that whole sub-instruction as OTHER and discard
                # the actionable EXIT/ENTER/APPROACH form.  Select the first
                # non-OTHER form in the taxonomy's priority order while
                # retaining the VLM's original natural-language text.
                typed_actionable = [item for item in typed if str(
                    item.get("form", "OTHER")).upper() != "OTHER"]
                typed_base = (typed_actionable[0] if typed_actionable else
                              typed[0])
                enriched.update({key: typed_base[key] for key in (
                    "source_clause", "form", "secondary_forms", "definition",
                    "point_selection_strategy") if key in typed_base})
                # The VLM spatial grounding is preferred. Definition-based
                # fields fill gaps but never erase a harness decision.
                enriched.setdefault(
                    "semantic_spatial_target", typed_base.get("semantic_spatial_target", ""))
                enriched.setdefault(
                    "visual_arrival_evidence", typed_base.get("visual_arrival_evidence", ""))
                enriched.setdefault("forbidden_target", typed_base.get("forbidden_target", ""))
            enriched["sub_instruction_id"] = len(sub_instructions)
            sub_instructions.append(SubInstruction.from_mapping(
                enriched, len(sub_instructions)))
        return sub_instructions[:limit] if limit is not None else sub_instructions


def decompose_instruction(instruction, vlm_harness=None, limit=None):
    """Functional decomposition API used by small scripts/tests."""
    return InstructionDecomposer(vlm_harness).decompose(instruction, limit)
