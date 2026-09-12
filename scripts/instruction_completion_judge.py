#!/usr/bin/env python3
"""Judge an edge as arrived, on-route, or unknown for its sub-instruction.

The public/default contract is ``completed``/``unknown``.  Three-way v9/v10
prompt evidence is collapsed at the adapter boundary unless the explicitly
diagnostic ``expose_progress_status`` option is enabled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Optional


COMPLETED = "completed"
ARRIVED = "arrived"
ON_ROUTE = "on_route"
UNKNOWN = "unknown"


def generic_near_stop_override_eligible(navigation_instruction,
                                        semantic_spatial_target="",
                                        spatial_relation=""):
    """Limit the operational STOP fallback to an unqualified near relation.

    A relation such as ``near the corner/end/front of X`` contains a precise
    semantic endpoint which object-size growth cannot establish.  Those
    clauses must remain under the normal multiview/temporal VLM judge.  The
    fallback is retained only for plain near/beside/adjacent relations.
    """
    text = " ".join(str(value or "") for value in (
        navigation_instruction, semantic_spatial_target, spatial_relation
    )).lower()
    plain_near = bool(re.search(
        r"\b(?:near|beside|next to|adjacent(?: to)?)\b", text))
    precise_endpoint = bool(re.search(
        r"\b(?:corner|end|edge|entrance|threshold|front|behind|between|"
        r"under|beneath|across from|opposite|left side|right side)\b", text))
    return bool(plain_near and not precise_endpoint)


def landmark_detection_aliases(landmark):
    """Return conservative detector aliases for one named landmark.

    Open-vocabulary detectors legitimately alternate between ``rug`` and
    ``carpet`` for the same flat floor covering across adjacent views. Treat
    those labels as one visual class when estimating endpoint area; unrelated
    object nouns are left unchanged.
    """
    tokens = {
        token for token in re.findall(r"[a-z]+", str(landmark).lower())
        if len(token) >= 3
    }
    if tokens.intersection({"rug", "carpet", "mat"}):
        tokens.update({"rug", "carpet", "mat"})
    return tokens


def current_rgb_turn_landmark_alignment_supported(
        form, status, point_arrived, point_selection_review,
        post_arrival_alignment, endpoint_evidence=None,
        reverse_transition_observed=False, stationary_or_blocked=False):
    """Validate a focused current-node RGB landmark-centering decision.

    The dedicated alignment VLM sees all eight clean current-node RGB views
    before an orientation-only action.  Its chosen camera is then rotated to
    FRONT and persisted in the node panorama.  This witness may resolve a
    conservative generic completion result when broad detector aliases scatter
    across sectors, but only after a real point arrival and exact yaw execution.
    """
    if (str(form or "").upper() != "TURN_TO_LANDMARK" or
            str(status or "").lower() != UNKNOWN or not point_arrived or
            reverse_transition_observed or stationary_or_blocked):
        return False
    review = point_selection_review or {}
    if not (review.get(
            "ambiguous_landmark_route_adjudication", {}) or {}).get("active"):
        return False
    alignment = post_arrival_alignment or {}
    focused = alignment.get("current_landmark_alignment", {}) or {}
    endpoint_state = str((endpoint_evidence or {}).get(
        "current_target_state", ""))
    try:
        target = float(alignment.get("committed_selection_yaw_rad"))
        final = float(alignment.get("final_yaw_rad"))
        view_index = int(focused.get("view_index"))
        expected_offset = (0.0, 45.0, 90.0, 135.0,
                           180.0, -135.0, -90.0, -45.0)[view_index]
        reported_offset = float(focused.get("relative_yaw_deg"))
    except (TypeError, ValueError, IndexError):
        return False
    yaw_error = abs(math.degrees(
        (final - target + math.pi) % (2.0 * math.pi) - math.pi))
    return bool(
        alignment.get("status") == "aligned_to_current_rgb_landmark" and
        yaw_error <= 2.0 and
        abs(reported_offset - expected_offset) <= 1e-6 and
        str(focused.get("identity_evidence", "")).strip() and
        focused.get("input_policy") == (
            "current_node_rgb_only_depth_prohibited") and
        endpoint_state in {"partial", "satisfied"})


@dataclass
class InstructionCompletionResult:
    status: str
    instruction_completed: bool
    instruction_on_route: bool
    expected_sub_instruction_id: int
    confidence: float
    reason: str
    visual_evidence: str
    previous_node_id: str
    current_node_id: str
    incoming_edge_id: str
    directional_evidence: Optional[dict] = None
    detector_directional_guard: Optional[dict] = None
    validation_overrides: Optional[dict] = None
    # Structured evidence is retained for the outer exploration policy.  It
    # does not alter the public completed/unknown decision; it only lets the
    # policy distinguish a supported partial transition (continue the active
    # sub-instruction) from a contradictory/ambiguous transition (recover).
    endpoint_evidence: Optional[dict] = None
    temporal_evidence: Optional[dict] = None
    motion_evidence: Optional[dict] = None
    # Preserve deterministic harness gates so the outer policy and audit can
    # distinguish a deliberate route-integrity veto from an unstructured VLM
    # unknown. Adapter recovery rules must never erase a false hard gate.
    decision_gates: Optional[dict] = None
    model_status: Optional[str] = None
    exit_endpoint_side_audit: Optional[dict] = None
    portal_endpoint_side_audit: Optional[dict] = None

    def to_dict(self):
        return asdict(self)


class NodeTransitionInstructionCompletionJudge:
    """Replaceable VLM adapter over previous node, edge, and current node."""

    def __init__(self, vlm_harness, graph_memory, sub_instructions,
                 minimum_confidence=0.5, expose_progress_status=False):
        self.vlm_harness = vlm_harness
        self.graph_memory = graph_memory
        self.sub_instructions = {
            int(item.sub_instruction_id): item for item in sub_instructions
        }
        self.minimum_confidence = float(minimum_confidence)
        self.expose_progress_status = bool(expose_progress_status)

    def judge(self, current_node, expected_sub_instruction_id,
              current_six_views, edge_keyframes):
        expected_id = int(expected_sub_instruction_id)
        if expected_id not in self.sub_instructions:
            raise KeyError(f"unknown sub-instruction {expected_id}")
        previous_node, incoming_edge = self.graph_memory.predecessor(
            current_node.node_id)
        if previous_node is None or incoming_edge is None:
            raise ValueError(
                "instruction completion requires a previous node and incoming edge")
        prompt_version = self.vlm_harness.instruction_completion_prompt_version
        three_way_progress = bool(prompt_version in {
            "v9_three_way_edge_progress", "v10_bidirectional_three_way"})
        eight_view_completion = bool(
            prompt_version in
            self.vlm_harness.EIGHT_VIEW_COMPLETION_PROMPT_VERSIONS)
        previous_six_views = (
            self.graph_memory.load_node_completion_views(previous_node.node_id)
            if eight_view_completion else
            self.graph_memory.load_node_views(previous_node.node_id))
        turn_carryover = incoming_edge.metadata.get("turn_carryover")
        stage_progress_carryover = incoming_edge.metadata.get(
            "stage_progress_carryover")
        point_selection_review = incoming_edge.metadata.get(
            "point_selection_review", {}) or {}
        # Persist the current frozen semantic ray beside any prior-stage
        # carryover.  Stage-level completion rules can then distinguish a
        # coherent bend around an obstacle from an unrelated/reverse retry,
        # using only online graph metadata and action history.
        current_edge_evidence = {
            "selected_yaw_rad": incoming_edge.metadata.get(
                "selected_yaw_rad"),
            "blocked_yaws_rad": list(incoming_edge.metadata.get(
                "blocked_yaws_rad", []) or []),
            "post_arrival_orientation_alignment": (
                incoming_edge.metadata.get(
                    "post_arrival_orientation_alignment")),
            "preselection_orientation_alignment": (
                incoming_edge.metadata.get(
                    "preselection_orientation_alignment")),
            "point_selection_review": point_selection_review,
        }
        carryover_evidence = {"current_edge": current_edge_evidence}
        carryover_previous_views = None
        carryover_edge_keyframes = None
        if turn_carryover or stage_progress_carryover:
            # Keep the two causal sources separate. A prior turn explains the
            # current heading, while a supported-partial edge supplies the
            # beginning of a longer semantic transition. Both are online
            # graph evidence; neither is a completion label.
            carryover_evidence.update({
                "turn": turn_carryover,
                "stage_progress": stage_progress_carryover,
            })
        if isinstance(stage_progress_carryover, dict):
            source_node_id = stage_progress_carryover.get("source_node_id")
            prior_edge_id = stage_progress_carryover.get("prior_edge_id")
            if source_node_id and prior_edge_id:
                # JSON summaries cannot reliably communicate a doorway/gap/
                # landmark crossing.  Supply the actual stage-start panorama
                # and preceding real-edge storyboard so the same generic VLM
                # can inspect the full A->B->C visual transition.
                carryover_previous_views = (
                    self.graph_memory.load_node_completion_views(
                        source_node_id)
                    if eight_view_completion else
                    self.graph_memory.load_node_views(source_node_id))
                carryover_edge_keyframes = (
                    self.graph_memory.load_edge_keyframes(prior_edge_id))
        raw = self.vlm_harness.judge_edge_instruction_completion(
            sub_instruction=self.sub_instructions[expected_id],
            following_sub_instruction=(
                self.sub_instructions.get(expected_id + 1)),
            previous_node_id=previous_node.node_id,
            current_node_id=current_node.node_id,
            previous_position_xyz=previous_node.position_xyz,
            current_position_xyz=current_node.position_xyz,
            previous_six_views=previous_six_views,
            current_six_views=current_six_views,
            previous_environment_semantics=previous_node.environment_semantics,
            current_environment_semantics=current_node.environment_semantics,
            edge_action_history=incoming_edge.action_history,
            edge_keyframes=edge_keyframes,
            previous_base_yaw_rad=previous_node.base_yaw_rad,
            current_base_yaw_rad=current_node.base_yaw_rad,
            previous_visual_embedding=previous_node.visual_embedding,
            current_visual_embedding=current_node.visual_embedding,
            edge_keyframe_records=incoming_edge.metadata.get(
                "edge_keyframes", []),
            carryover_evidence=carryover_evidence,
            carryover_previous_six_views=carryover_previous_views,
            carryover_edge_keyframes=carryover_edge_keyframes,
        )
        form = str(self.sub_instructions[expected_id].form).upper()
        point_arrived = bool(
            getattr(current_node, "arrival_signal", None) or
            (current_node.metadata or {}).get("point_target_arrived"))
        temporal_now = raw.get("temporal_evidence", {}) or {}
        motion_now = raw.get("motion_evidence", {}) or {}
        current_rgb_alignment_witness = (
            current_rgb_turn_landmark_alignment_supported(
                form=form, status=raw.get("status"),
                point_arrived=point_arrived,
                point_selection_review=point_selection_review,
                post_arrival_alignment=incoming_edge.metadata.get(
                    "post_arrival_orientation_alignment"),
                endpoint_evidence=raw.get("endpoint_evidence"),
                reverse_transition_observed=bool(
                    temporal_now.get("reverse_transition_observed")),
                stationary_or_blocked=bool(
                    motion_now.get("stationary_or_blocked"))))
        if current_rgb_alignment_witness:
            raw = dict(raw)
            raw["status"] = COMPLETED
            raw["confidence"] = max(
                float(raw.get("confidence", 0.0)), self.minimum_confidence)
            raw["turn_landmark_current_rgb_alignment_override"] = True
        # An explicit front-sector witness means a PASS/ADVANCE landmark has
        # not yet left the forward hemisphere.  Enforce this at the adapter
        # boundary even when the backend returned ``completed`` directly;
        # otherwise a backend-specific optimistic override can advance the
        # instruction cursor on an ambiguous front+rear panorama.  This uses
        # only structured node evidence and is independent of scene/path.
        if form in {"PASS_LANDMARK", "ADVANCE_STRAIGHT",
                    "FOLLOW_PATH_BOUNDARY"}:
            endpoint_now = raw.get("endpoint_evidence", {}) or {}
            temporal_now = raw.get("temporal_evidence", {}) or {}
            motion_now = raw.get("motion_evidence", {}) or {}
            current_sector_set = {
                str(value).strip().lower() for value in
                endpoint_now.get("reference_current_sectors", [])}
            ordered_rear_pass = bool(
                form == "PASS_LANDMARK" and
                current_sector_set.intersection({
                    "rear_left", "rear", "rear_right"}) and
                temporal_now.get("semantic_order") == "instructed" and
                bool(temporal_now.get("keyframe_support")) and
                motion_now.get("instruction_motion_fit") == "supports" and
                not bool(motion_now.get("stationary_or_blocked")) and
                not bool(temporal_now.get("reverse_transition_observed")))
            if ("front" in current_sector_set and
                    str(raw.get("status", "")) in {COMPLETED, ARRIVED} and
                    not ordered_rear_pass):
                raw = dict(raw)
                raw["status"] = UNKNOWN
                raw["confidence"] = min(float(raw.get("confidence", 0.0)),
                                         self.minimum_confidence - 1e-3)
                raw["pass_front_persistence_guard"] = True
                raw["directional_evidence"] = {
                    **(raw.get("directional_evidence") or {}),
                    "rule": "front-sector landmark witness blocks pass completion",
                    "current_sectors": sorted(current_sector_set),
                }
        # A PASS/ADVANCE completion must not be accepted when the new node is
        # geometrically behind the source relative to the source's incoming
        # lane.  The VLM can call a visually plausible rear room "past the
        # landmark"; the persisted graph edge provides a model-independent
        # contradiction signal.  Use only node positions already stored in
        # this graph, never the reference trajectory or hidden labels.  Keep
        # the node itself (physical arrival is still valid), but downgrade
        # the instruction judgment to UNKNOWN so the outer recovery policy
        # can decide whether this is wrong or merely on-route.
        reverse_guard = None
        if form in {"PASS_LANDMARK", "ADVANCE_STRAIGHT",
                    "FOLLOW_PATH_BOUNDARY"}:
            predecessor_of_source, _ = self.graph_memory.predecessor(
                previous_node.node_id)
            if predecessor_of_source is not None:
                import numpy as np
                prior_vec = np.asarray(previous_node.position_xyz,
                                        dtype=np.float64)[[0, 2]] - np.asarray(
                                            predecessor_of_source.position_xyz,
                                            dtype=np.float64)[[0, 2]]
                edge_vec = np.asarray(current_node.position_xyz,
                                      dtype=np.float64)[[0, 2]] - np.asarray(
                                          previous_node.position_xyz,
                                          dtype=np.float64)[[0, 2]]
                prior_norm = float(np.linalg.norm(prior_vec))
                edge_norm = float(np.linalg.norm(edge_vec))
                if prior_norm > 0.35 and edge_norm > 0.35:
                    cosine = float(np.dot(prior_vec, edge_vec) /
                                   (prior_norm * edge_norm))
                    reverse_guard = {
                        "rule": "forward-progress edge must not reverse source incoming lane",
                        "cosine_with_source_incoming_lane": cosine,
                        "prior_displacement_m": prior_norm,
                        "edge_displacement_m": edge_norm,
                        "threshold": -0.20,
                        "triggered": bool(cosine < -0.20),
                    }
                    if cosine < -0.20:
                        raw = dict(raw)
                        raw["status"] = UNKNOWN
                        raw["confidence"] = min(float(raw.get("confidence", 0.0)),
                                                  self.minimum_confidence - 1e-3)
                        raw["pass_reverse_guard"] = True
                        raw["directional_evidence"] = {
                            **(raw.get("directional_evidence") or {}),
                            "forward_progress_reverse_guard": reverse_guard,
                        }
        # Broad room-category detector proposals are frequently duplicated
        # across every panorama sector and can dominate the general judge even
        # when clean RGB shows that the agent entered a distinct downstream
        # foyer.  For a PASS whose landmark is a region, ask a separate RGB-
        # only adjudicator about that topological transition.  This is a
        # generic form/noun rule and remains gated by real point arrival,
        # ordered motion, keyframes, and the reverse guard.
        landmark_text = str(
            getattr(self.sub_instructions[expected_id], "landmark", ""))
        pass_region_reference = bool(re.search(
            r"\b(?:room|hall(?:way)?|corridor|lobby|foyer|kitchen|bedroom|"
            r"bathroom|entrance|area|region)\b", landmark_text.lower()))
        if (form == "PASS_LANDMARK" and pass_region_reference and
                str(raw.get("status", "")) == UNKNOWN and point_arrived and
                not bool(raw.get("pass_reverse_guard"))):
            traveled = sum(float(action.get("moved_m", 0.0))
                           for action in incoming_edge.action_history)
            # This focused clean-RGB adjudicator exists specifically because
            # the broad primary judge can make unstable claims about semantic
            # order/keyframe support when detector text is noisy.  Gating the
            # adjudicator on those same model-produced booleans makes it run
            # on one identical edge and disappear on another.  Eligibility
            # therefore uses only real edge facts; the adjudicator itself is
            # responsible for deciding the visual chronology and boundary.
            eligible = bool(
                traveled >= 0.75 and
                bool(edge_keyframes) and
                not bool(raw.get("pass_reverse_guard")))
            if eligible:
                adjudication = (
                    self.vlm_harness.adjudicate_pass_region_transition(
                        self.sub_instructions[expected_id],
                        previous_six_views, edge_keyframes,
                        current_six_views))
                adjudicated_complete = bool(
                    adjudication.get("pass_boundary_satisfied") and
                    adjudication.get(
                        "current_distinct_downstream_context") and
                    not adjudication.get("old_region_still_encloses_front") and
                    adjudication.get("keyframe_transition_support") and
                    float(adjudication.get("confidence", 0.0)) >= 0.65)
                raw = dict(raw)
                raw["pass_region_rgb_adjudication"] = adjudication
                if adjudicated_complete:
                    raw["status"] = COMPLETED
                    raw["confidence"] = max(
                        float(raw.get("confidence", 0.0)),
                        float(adjudication.get("confidence", 0.0)))
                    raw["pass_region_rgb_override"] = True
                    raw["endpoint_evidence"] = {
                        **(raw.get("endpoint_evidence") or {}),
                        "current_target_state": "satisfied",
                        "current_completion_cue": "satisfied",
                    }
                    raw["temporal_evidence"] = {
                        **(raw.get("temporal_evidence") or {}),
                        "semantic_order": "instructed",
                        "keyframe_support": True,
                        "boundary_event": "endpoint_inferred"}
                    raw["decision_gates"] = {
                        **(raw.get("decision_gates") or {}),
                        "current_full_target_satisfied": True,
                        "current_completion_cue_satisfied": True,
                        "completion_boundary_supported": True,
                    }
        # Adapter-level safety net for the project-wide eight-view PASS rule.
        # Some VLM responses are conservatively normalized to ``unknown`` (and
        # even assign confidence below the sequence threshold) when a noisy
        # front-right detector co-occurs with a clear rear/rear-side landmark.
        # The public judge already receives the structured temporal evidence;
        # apply the same generic forward-motion + rear-side boundary rule here
        # so the sequence state machine cannot branch solely on that noisy
        # textual confidence.  No demonstration path or hidden outcome is
        # consulted.
        if (str(self.sub_instructions[expected_id].form).upper() ==
                "PASS_LANDMARK" and str(raw.get("status", "")) == UNKNOWN):
            endpoint = raw.get("endpoint_evidence", {}) or {}
            temporal = raw.get("temporal_evidence", {}) or {}
            motion = raw.get("motion_evidence", {}) or {}
            current_sectors = {
                str(value).strip().lower() for value in
                endpoint.get("reference_current_sectors", [])}
            try:
                traveled = float(
                    (raw.get("structured_motion_summary", {}) or {}).get(
                        "traveled_distance_m", 0.0))
            except (TypeError, ValueError):
                traveled = 0.0
            if traveled <= 0.0:
                traveled = sum(float(action.get("moved_m", 0.0))
                               for action in incoming_edge.action_history)
            pass_override = bool(
                endpoint.get("previous_target_state") != "satisfied" and
                endpoint.get("current_target_state") == "satisfied" and
                endpoint.get("current_completion_cue") == "satisfied" and
                temporal.get("same_reference_instance") == "yes" and
                current_sectors.intersection({
                    "rear_left", "rear", "rear_right"}) and
                temporal.get("semantic_order") == "instructed" and
                bool(temporal.get("keyframe_support")) and
                motion.get("instruction_motion_fit") == "supports" and
                not bool(motion.get("stationary_or_blocked")) and
                not bool(temporal.get("reverse_transition_observed")) and
                not bool(raw.get("pass_reverse_guard")) and
                traveled >= 0.50)
            if pass_override:
                raw = dict(raw)
                raw["status"] = COMPLETED
                raw["confidence"] = max(
                    float(raw.get("confidence", 0.0)), self.minimum_confidence)
                raw["pass_directional_override"] = True
                raw["pass_directional_evidence"] = {
                    "rule": "motion-supported rear-side eight-view pass boundary",
                    "current_sectors": sorted(current_sectors),
                    "traveled_distance_m": traveled,
                }
        # Finite corridors/gaps are often named by their route surface while
        # the visual boundary is supplied by side objects (rope barriers,
        # chairs, rails). The surface itself can continue into the next room.
        # A newly visible, instruction-provided next landmark after an
        # ordered real edge is therefore positive evidence that the current
        # BETWEEN passage has been cleared; it does not complete the next
        # instruction. Require novelty across the two stored nodes so a
        # distant landmark already visible before motion cannot trigger it.
        between_route_integrity = bool(
            (raw.get("decision_gates") or {}).get(
                "between_stage_route_integrity", True))
        if (form == "BETWEEN_OBJECTS" and between_route_integrity and
                str(raw.get("status", "")) == UNKNOWN and
                (expected_id + 1) in self.sub_instructions):
            following = self.sub_instructions[expected_id + 1]
            next_tokens = landmark_detection_aliases(
                getattr(following, "landmark", ""))
            next_tokens -= {
                "next", "large", "small", "left", "right", "front",
                "near", "room", "area", "floor", "region"}

            def observed_token_scores(semantics):
                scores = {token: 0.0 for token in next_tokens}
                for view in (semantics or {}).get("views", []):
                    for detection in view.get("detections", []):
                        label = str(detection.get("label", "")).lower()
                        score = float(detection.get("score", 0.0) or 0.0)
                        for token in next_tokens:
                            if token in label:
                                scores[token] = max(scores[token], score)
                return scores

            previous_scores = observed_token_scores(
                previous_node.environment_semantics)
            current_scores = observed_token_scores(
                current_node.environment_semantics)
            novel_tokens = [
                token for token in next_tokens
                if current_scores.get(token, 0.0) >= 0.28 and
                previous_scores.get(token, 0.0) < 0.22]
            temporal = raw.get("temporal_evidence", {}) or {}
            motion = raw.get("motion_evidence", {}) or {}
            traveled = sum(float(action.get("moved_m", 0.0))
                           for action in incoming_edge.action_history)
            point_arrived = bool(
                getattr(current_node, "arrival_signal", None) or
                (current_node.metadata or {}).get("point_target_arrived"))
            following_view_index = point_selection_review.get(
                "first_following_landmark_view_index", -1)
            try:
                following_visible = bool(
                    following_view_index is not None and
                    int(following_view_index) >= 0)
            except (TypeError, ValueError):
                following_visible = False
            selection_boundary_witness = bool(
                following_visible and
                str(point_selection_review.get(
                    "sequence_alignment", "")).lower() in {
                        "same_view", "adjacent_view"} and
                float(point_selection_review.get(
                    "confidence", 0.0) or 0.0) >= 0.80 and
                point_arrived and traveled >= 2.00 and
                (raw.get("endpoint_evidence", {}) or {}).get(
                    "current_target_state") in {"partial", "satisfied"} and
                temporal.get("semantic_order") == "instructed" and
                bool(temporal.get("keyframe_support")) and
                not bool(temporal.get("reverse_transition_observed")) and
                motion.get("instruction_motion_fit") == "supports" and
                not bool(motion.get("stationary_or_blocked")))
            if selection_boundary_witness:
                raw = dict(raw)
                raw["status"] = COMPLETED
                raw["confidence"] = max(
                    float(raw.get("confidence", 0.0)),
                    self.minimum_confidence)
                raw["between_visible_following_boundary_override"] = True
                raw["between_visible_following_boundary_evidence"] = {
                    "rule": (
                        "real BETWEEN edge reaches a point selected with a "
                        "high-confidence same-route following landmark"),
                    "following_landmark": point_selection_review.get(
                        "first_following_landmark"),
                    "following_view_index": int(following_view_index),
                    "sequence_alignment": point_selection_review.get(
                        "sequence_alignment"),
                    "selection_confidence": float(
                        point_selection_review.get("confidence", 0.0)),
                    "traveled_distance_m": traveled,
                }
            following_boundary_witness = bool(
                not selection_boundary_witness and
                novel_tokens and point_arrived and traveled >= 0.50 and
                temporal.get("semantic_order") == "instructed" and
                bool(temporal.get("keyframe_support")) and
                not bool(temporal.get("reverse_transition_observed")) and
                motion.get("instruction_motion_fit") == "supports" and
                not bool(motion.get("stationary_or_blocked")))
            if following_boundary_witness:
                raw = dict(raw)
                raw["status"] = COMPLETED
                raw["confidence"] = max(
                    float(raw.get("confidence", 0.0)),
                    self.minimum_confidence)
                raw["between_following_landmark_override"] = True
                raw["between_following_landmark_evidence"] = {
                    "rule": (
                        "new following-instruction landmark after ordered "
                        "BETWEEN passage"),
                    "following_sub_instruction_id": expected_id + 1,
                    "novel_landmark_tokens": sorted(novel_tokens),
                    "previous_token_scores": previous_scores,
                    "current_token_scores": current_scores,
                    "traveled_distance_m": traveled,
                }
            # A high-confidence selection review on the preceding real edge
            # can already establish that the following landmark lies on the
            # same outgoing route even when open-vocabulary detection misses
            # it at the next node.  The one permitted continuation is capped
            # to 1.25 m by the exploration policy above.  Accept that short
            # clearance edge only with ordered keyframes, supporting motion,
            # no reversal/stall, and a true point arrival.  This is a generic
            # A->B->C boundary handoff and never reads the demonstration path.
            stage_progress = (stage_progress_carryover
                              if isinstance(stage_progress_carryover, dict)
                              else {})
            endpoint = raw.get("endpoint_evidence", {}) or {}
            carryover_boundary_witness = bool(
                not following_boundary_witness and
                stage_progress.get("prior_following_landmark_visible") and
                point_arrived and 0.30 <= traveled <= 1.40 and
                endpoint.get("current_target_state") in {
                    "partial", "satisfied"} and
                temporal.get("semantic_order") == "instructed" and
                bool(temporal.get("keyframe_support")) and
                not bool(temporal.get("reverse_transition_observed")) and
                motion.get("instruction_motion_fit") == "supports" and
                not bool(motion.get("stationary_or_blocked")))
            if carryover_boundary_witness:
                raw = dict(raw)
                raw["status"] = COMPLETED
                raw["confidence"] = max(
                    float(raw.get("confidence", 0.0)),
                    self.minimum_confidence)
                raw["between_route_boundary_lookahead_override"] = True
                raw["between_route_boundary_lookahead_evidence"] = {
                    "rule": (
                        "bounded second edge after high-confidence same-route "
                        "following-landmark review"),
                    "following_landmark": stage_progress.get(
                        "prior_following_landmark"),
                    "sequence_alignment": stage_progress.get(
                        "prior_sequence_alignment"),
                    "traveled_distance_m": traveled,
                    "maximum_policy_distance_m": 1.40,
                }
        # Final adapter invariant: no semantic recovery (present or future)
        # may override the V24 physical route-integrity veto. Keep the model's
        # extracted endpoint fields for audit and bounded continuation, but do
        # not claim completion on this edge.
        if form == "BETWEEN_OBJECTS" and not between_route_integrity:
            raw = dict(raw)
            raw["status"] = UNKNOWN
            raw["confidence"] = min(
                float(raw.get("confidence", 0.0)),
                self.minimum_confidence - 1e-3)
            raw["between_stage_route_integrity_veto"] = True
        # Portal completion occasionally arrives with every structured field
        # satisfied while the backend's top-level label remains ``unknown``
        # (typically its prose even says the threshold was crossed).  Promote
        # only the full source->boundary->destination witness: real point
        # arrival, positive translation, target/cue satisfied, observed
        # boundary, ordered keyframes, same instance, supporting motion, and
        # no reversal/stall.  This is stricter than trusting prose and is
        # reusable across ENTER/EXIT/TRAVERSE/SELECT forms.
        if (form in {"EXIT_REGION", "ENTER_REGION", "SELECT_PORTAL",
                     "TRAVERSE_PORTAL_REGION"} and
                str(raw.get("status", "")) == UNKNOWN and
                not bool(raw.get("exit_endpoint_side_veto_applied"))):
            endpoint = raw.get("endpoint_evidence", {}) or {}
            temporal = raw.get("temporal_evidence", {}) or {}
            motion = raw.get("motion_evidence", {}) or {}
            traveled = sum(float(action.get("moved_m", 0.0))
                           for action in incoming_edge.action_history)
            point_arrived = bool(
                getattr(current_node, "arrival_signal", None) or
                (current_node.metadata or {}).get("point_target_arrived"))
            portal_transition_witness = bool(
                point_arrived and traveled >= 0.50 and
                endpoint.get("previous_target_state") != "satisfied" and
                endpoint.get("current_target_state") == "satisfied" and
                endpoint.get("current_completion_cue") == "satisfied" and
                temporal.get("boundary_event") == "observed" and
                temporal.get("semantic_order") == "instructed" and
                bool(temporal.get("keyframe_support")) and
                temporal.get("same_reference_instance") == "yes" and
                not bool(temporal.get("reverse_transition_observed")) and
                motion.get("instruction_motion_fit") == "supports" and
                not bool(motion.get("stationary_or_blocked")))
            if portal_transition_witness:
                raw = dict(raw)
                raw["status"] = COMPLETED
                raw["confidence"] = max(
                    float(raw.get("confidence", 0.0)),
                    self.minimum_confidence)
                raw["portal_structured_transition_override"] = True
                raw["portal_structured_transition_evidence"] = {
                    "rule": (
                        "full structured source-boundary-destination witness"),
                    "traveled_distance_m": traveled,
                    "boundary_event": temporal.get("boundary_event"),
                    "semantic_order": temporal.get("semantic_order"),
                }
        # Portal endpoint-side evidence is a hard RGB occupancy gate.  It is
        # evaluated without chronology or detector text in the harness and
        # must survive every adapter recovery rule.  A destination-side audit
        # may permit the ordinary structured portal witness above; a source or
        # threshold verdict can never be promoted by those same fields.
        if (form in {"EXIT_REGION", "ENTER_REGION", "SELECT_PORTAL",
                     "TRAVERSE_PORTAL_REGION"} and
                bool(raw.get("exit_endpoint_side_veto_applied"))):
            raw = dict(raw)
            raw["status"] = UNKNOWN
            raw["confidence"] = min(
                float(raw.get("confidence", 0.0)),
                self.minimum_confidence - 1e-3)
        # STOP_WAIT is satisfied by being stably adjacent to the named
        # landmark; the robot's final camera heading is not part of the
        # relation unless the instruction explicitly says "approach/front".
        # DeepSeek is often over-conservative when the landmark is in a rear
        # sector after a legitimate 180-degree turn.  Use only information
        # already persisted at the two graph nodes and incoming edge: a real
        # point-arrival signal, non-trivial motion, and a generic detector
        # area-growth/near-size witness for the same landmark.  This does not
        # use the R2R reference path, hidden labels, or a scene-specific rule.
        if (str(self.sub_instructions[expected_id].form).upper() ==
                "STOP_WAIT" and str(raw.get("status", "")) == UNKNOWN):
            item = self.sub_instructions[expected_id]
            instruction_text = " ".join(str(getattr(item, key, ""))
                                         for key in (
                                             "navigation_instruction",
                                             "semantic_spatial_target",
                                             "spatial_relation",
                                             "completion_cue")).lower()
            # ``completion_cue`` is often generated as “visible in front and
            # close” even for the ordinary semantic relation “stop near X”.
            # Treat explicit front/approach wording as a hard requirement only
            # when it occurs in the navigation clause or spatial relation,
            # not in that descriptive cue.
            directional_text = " ".join(str(getattr(item, key, ""))
                                         for key in (
                                             "navigation_instruction",
                                             "semantic_spatial_target",
                                             "spatial_relation")).lower()
            explicit_front_approach = any(token in directional_text for token in (
                "approach", "in front", "visible ahead", "front of"))
            near_relation = generic_near_stop_override_eligible(
                getattr(item, "navigation_instruction", ""),
                getattr(item, "semantic_spatial_target", ""),
                getattr(item, "spatial_relation", ""))
            point_arrived = bool(
                getattr(current_node, "arrival_signal", None) or
                (current_node.metadata or {}).get("point_target_arrived"))
            traveled = sum(float(action.get("moved_m", 0.0))
                           for action in incoming_edge.action_history)

            def landmark_area_max(semantics):
                best = 0.0
                landmark_tokens = landmark_detection_aliases(
                    getattr(item, "landmark", ""))
                for view in (semantics or {}).get("views", []):
                    for detection in view.get("detections", []):
                        label = str(detection.get("label", "")).lower()
                        if any(token in label for token in landmark_tokens):
                            best = max(best, float(
                                detection.get(
                                    "area_fraction",
                                    detection.get("mask_area_fraction", 0.0)) or
                                0.0))
                return best

            previous_area = landmark_area_max(previous_node.environment_semantics)
            current_area = landmark_area_max(current_node.environment_semantics)
            area_growth = (current_area / max(previous_area, 1e-4)
                           if previous_area > 0.0 else current_area)
            stop_relation_witness = bool(
                near_relation and not explicit_front_approach and point_arrived and
                traveled >= 0.50 and current_area >= 0.06 and
                (area_growth >= 1.20 or previous_area <= 0.0) and
                str((raw.get("endpoint_evidence", {}) or {}).get(
                    "current_target_state", "")) == "satisfied" and
                str((raw.get("endpoint_evidence", {}) or {}).get(
                    "current_completion_cue", "")) == "satisfied" and
                not bool((raw.get("temporal_evidence", {}) or {}).get(
                    "reverse_transition_observed")))
            if stop_relation_witness:
                raw = dict(raw)
                raw["status"] = COMPLETED
                raw["confidence"] = max(
                    float(raw.get("confidence", 0.0)), self.minimum_confidence)
                raw["stop_relation_operational_override"] = True
                raw["stop_relation_evidence"] = {
                    "rule": "physical point arrival + landmark area growth",
                    "previous_area_fraction": previous_area,
                    "current_area_fraction": current_area,
                    "area_growth_ratio": area_growth,
                    "traveled_distance_m": traveled,
                    "front_only_required": explicit_front_approach,
                }
        # A ray-local endpoint constructed immediately before a physical
        # obstruction is explicitly an intermediate observation node.  It
        # can preserve a correct semantic direction and provide new views,
        # but by construction it has not reached the VLM-selected far-side
        # relation.  This veto must run after every semantic override so an
        # optimistic completion narrative cannot advance the instruction at
        # the obstruction boundary.
        postselection_repair = point_selection_review.get(
            "postselection_repair", {}) or {}
        if (postselection_repair.get("status") ==
                "repaired_to_local_occlusion_boundary"):
            raw = dict(raw)
            raw["status"] = UNKNOWN
            raw["confidence"] = min(
                float(raw.get("confidence", 0.0)),
                self.minimum_confidence - 1e-3)
            raw["local_occlusion_boundary_intermediate_veto"] = True
        status = str(raw["status"])
        confidence = float(raw["confidence"])
        accepted = confidence >= self.minimum_confidence
        completed = bool(
            status in {COMPLETED, ARRIVED} and accepted)
        on_route = bool(status == ON_ROUTE and accepted)
        if three_way_progress and self.expose_progress_status:
            final_status = (
                ARRIVED if completed else ON_ROUTE if on_route else UNKNOWN)
        else:
            final_status = COMPLETED if completed else UNKNOWN
        return InstructionCompletionResult(
            status=final_status,
            instruction_completed=completed,
            instruction_on_route=bool(
                on_route and self.expose_progress_status),
            expected_sub_instruction_id=expected_id,
            confidence=confidence,
            reason=str(raw.get("reason", "")),
            visual_evidence=str(raw.get("visual_evidence", "")),
            previous_node_id=previous_node.node_id,
            current_node_id=current_node.node_id,
            incoming_edge_id=incoming_edge.edge_id,
            directional_evidence=raw.get("directional_evidence"),
            detector_directional_guard=raw.get("detector_directional_guard"),
            validation_overrides={
                key: bool(raw[key]) for key in (
                    "identity_ambiguity_override", "pass_relation_override",
                    "detector_directional_override",
                    "pass_directional_override",
                    "circumnavigation_rear_endpoint_override",
                    "exit_temporal_crossing_override",
                    "exit_endpoint_side_veto_applied",
                    "portal_endpoint_side_veto_applied",
                    "between_following_landmark_override",
                    "between_visible_following_boundary_override",
                    "between_route_boundary_lookahead_override",
                    "between_stage_route_integrity_veto",
                    "portal_structured_transition_override",
                    "stop_relation_operational_override",
                    "turn_landmark_current_rgb_alignment_override",
                    "pass_region_rgb_override",
                    "local_occlusion_boundary_intermediate_veto")
                if key in raw
            } or None,
            endpoint_evidence=raw.get("endpoint_evidence"),
            temporal_evidence=raw.get("temporal_evidence"),
            motion_evidence=raw.get("motion_evidence"),
            decision_gates=raw.get("decision_gates"),
            model_status=raw.get("model_status"),
            exit_endpoint_side_audit=raw.get("exit_endpoint_side_audit"),
            portal_endpoint_side_audit=raw.get(
                "portal_endpoint_side_audit"),
        )


class RGBOnlyNodeTransitionInstructionCompletionJudge:
    """Completion adapter whose complete online input is RGB + action tokens."""

    def __init__(self, vlm_harness, graph_memory, sub_instructions,
                 minimum_confidence=0.5):
        self.vlm_harness = vlm_harness
        self.graph_memory = graph_memory
        self.sub_instructions = {
            int(item.sub_instruction_id): item for item in sub_instructions
        }
        self.minimum_confidence = float(minimum_confidence)

    def judge(self, current_node, expected_sub_instruction_id,
              current_views, edge_keyframes):
        expected_id = int(expected_sub_instruction_id)
        if expected_id not in self.sub_instructions:
            raise KeyError(f"unknown sub-instruction {expected_id}")
        previous_node, incoming_edge = self.graph_memory.predecessor(
            current_node.node_id)
        if previous_node is None or incoming_edge is None:
            raise ValueError(
                "instruction completion requires a previous node and incoming edge")
        use_eight = len(current_views) == 8
        previous_views = (
            self.graph_memory.load_node_completion_views(previous_node.node_id)
            if use_eight else
            self.graph_memory.load_node_views(previous_node.node_id))
        raw = self.vlm_harness.judge_edge_instruction_completion_rgb_only(
            sub_instruction=self.sub_instructions[expected_id],
            following_sub_instruction=self.sub_instructions.get(expected_id + 1),
            previous_node_id=previous_node.node_id,
            current_node_id=current_node.node_id,
            previous_views=previous_views,
            current_views=current_views,
            edge_action_history=incoming_edge.action_history,
            edge_keyframes=edge_keyframes,
            previous_environment_semantics=previous_node.environment_semantics,
            current_environment_semantics=current_node.environment_semantics)
        accepted = bool(
            raw["status"] == COMPLETED and
            float(raw["confidence"]) >= self.minimum_confidence and
            current_node.arrival_signal)
        status = COMPLETED if accepted else UNKNOWN
        return InstructionCompletionResult(
            status=status,
            instruction_completed=accepted,
            instruction_on_route=False,
            expected_sub_instruction_id=expected_id,
            confidence=float(raw["confidence"]),
            reason=str(raw["reason"]),
            visual_evidence=str(raw["visual_evidence"]),
            previous_node_id=previous_node.node_id,
            current_node_id=current_node.node_id,
            incoming_edge_id=incoming_edge.edge_id,
            validation_overrides={
                "policy_input_contract": "rgb_only_v1",
                "privileged_inputs_used": [],
                "prompt_version": raw.get("prompt_version"),
                "harness_overrides": raw.get("validation_overrides", []),
                "relation_evidence": raw.get("relation_evidence", {}),
            },
            temporal_evidence=raw.get("temporal_evidence"),
            motion_evidence=raw.get("motion_evidence"),
            model_status=raw.get("status"))
