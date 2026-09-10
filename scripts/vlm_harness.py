#!/usr/bin/env python3
"""Replaceable VLM backends plus validated R2R instruction/ground-point harness."""

import base64
import hashlib
import io
import json
import math
import os
import random
import re
import socket
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from instruction_completion_evidence import (
    instruction_tokens, summarize_motion, summarize_semantic_transition,
    summarize_visual_transition)


DEFAULT_OLLAMA_MODEL = "llama3.2-vision:latest"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash-vision-exp"
# The cloud VLM is reached through the DMXAPI OpenAI-compatible relay.  The
# historical OpenRouter variable names below are shared with Navi-Agent's
# Final Method so one local_env.sh block configures both projects; the older
# DEEPSEEK_* names stay as compatibility fallbacks.
DEFAULT_DEEPSEEK_BASE_URL = "https://www.dmxapi.cn/v1"
RELAY_API_KEY_ENV = ("OPENROUTER_API_KEY", "DEEPSEEK_API_KEY")
RELAY_API_URL_ENV = ("OPENROUTER_API_URL", "DEEPSEEK_BASE_URL")
RELAY_MODEL_ENV = ("NAVI_OPENROUTER_MODEL", "DEEPSEEK_VLM_MODEL")
RELAY_PLACEHOLDER_KEYS = {"your_openrouter_key_here", "your_dmxapi_key_here"}
RELAY_RETRYABLE_HTTP = {408, 429, 500, 502, 503, 504}
LOCAL_ENV_SH_PATH = Path(__file__).resolve().parents[1] / "local_env.sh"


def compound_terminal_extent_portal_stage(item):
    """Whether a portal-labelled stage actually ends at a landmark extent.

    Decomposition sometimes represents ``enter the kitchen and walk along
    the counter to its end`` as ENTER_REGION.  Its semantic endpoint is the
    end of the counter, not the kitchen threshold, so a binary portal-side
    audit is the wrong independent question.  This lexical test uses only the
    frozen sub-instruction fields and deliberately mirrors the executor's
    compound-terminal-extent definition without importing it (which would
    introduce a circular dependency).
    """
    item = dict(item or {})
    if str(item.get("form", "")).upper() not in {
            "ENTER_REGION", "FOLLOW_PATH_BOUNDARY"}:
        return False
    text = " ".join(str(item.get(key, "")) for key in (
        "navigation_instruction", "semantic_spatial_target",
        "spatial_relation", "completion_cue",
        "visual_arrival_evidence")).lower()
    return bool(
        re.search(r"\b(?:along|follow|length)\b", text) and
        re.search(r"\b(?:far end|the end|until|entire length|"
                  r"passed? the length|reach the end)\b", text))


class VLMProviderFatalError(Exception):
    """Non-retryable provider/configuration failure that must abort a run.

    This intentionally does not inherit from ``RuntimeError``.  Navigation
    policy code treats RuntimeError as a recoverable semantic decision
    failure; authentication and account-balance failures are evaluation
    infrastructure failures and must never be converted into navigation
    outcomes.
    """


def default_vlm_model(backend_name):
    """Return the project default model for a configured backend."""
    if backend_name == "deepseek":
        for key in RELAY_MODEL_ENV:
            if os.environ.get(key, "").strip():
                return os.environ[key].strip()
        return DEFAULT_DEEPSEEK_MODEL
    return DEFAULT_OLLAMA_MODEL


def between_recovery_source_is_forward(previous_sectors):
    """Whether named gap references were still ahead at the source node.

    This is a conservative prerequisite only for deterministic recovery of a
    VLM-unknown BETWEEN_OBJECTS edge. If both named references were already
    behind before the edge, their later front/rear co-occurrence cannot prove
    that this edge entered the instructed gap; it may be a repeated instance
    or a route reversal.
    """
    sectors = {str(value).strip().lower()
               for value in (previous_sectors or [])}
    return bool(sectors.intersection(
        {"front", "front_left", "front_right", "left", "right"}))


def between_terminal_extent_recovery_allowed(
        instruction_text, current_state, completion_cue, boundary_event):
    """Protect an explicit corridor/gap end from partial BETWEEN recovery."""
    terminal_extent = bool(re.search(
        r"\b(?:far end|end|until|all the way)\b",
        str(instruction_text or "").lower()))
    if not terminal_extent:
        return True
    return bool(
        str(current_state).lower() == "satisfied" and
        str(completion_cue).lower() == "satisfied" and
        str(boundary_event).lower() in {"observed", "endpoint_inferred"})


def between_panorama_endpoint_recovery_supported(
        *, pair_tokens_seen, side_bracket, panorama_bracket,
        source_pair_ahead, terminal_extent_ok, semantic_order,
        reverse_observed, stationary, motion_fit, keyframe_support,
        traveled_distance_m):
    """Recover a visible BETWEEN relation without trusting VLM prose alone.

    Two independently named objects bracketing the current panorama after a
    meaningful, non-reversed translation is direct endpoint evidence.  A
    missing keyframe flag or neutral motion prose must not erase that RGB
    geometry; either the ordinary strong temporal judgment or a longer real
    edge can support the boundary.  This is form-level and uses no depth,
    navmesh, demonstration path, scene identity, or expected result.
    """
    temporal_support = bool(
        (keyframe_support and motion_fit == "supports") or
        (float(traveled_distance_m) >= 1.5 and
         str(motion_fit) != "contradicts"))
    return bool(
        int(pair_tokens_seen) >= 2 and
        (side_bracket or panorama_bracket) and
        source_pair_ahead and terminal_extent_ok and
        str(semantic_order) != "reversed" and
        not reverse_observed and not stationary and temporal_support)


def circumnavigation_rear_endpoint_supported(
        *, reference_sectors, detector_sectors, current_state,
        reverse_observed, stationary, motion_fit,
        horizontal_displacement_m, traveled_distance_m):
    """Recognize a completed behind/around relation at an actual edge end.

    Eight overlapping views make a landmark at roughly 45 degrees appear in a
    nominal ``front_right``/``front_left`` sector even after it has moved to
    the robot's side.  For a backside/around clause, only an exact FRONT
    witness contradicts the endpoint.  Require agreement between the VLM
    panorama sectors and focused detector evidence in a rear sector, plus a
    substantial non-reversed physical edge.  This consumes no depth, navmesh,
    demonstration path, scene identity, or episode label.
    """
    reference_sectors = {str(value).strip().lower()
                         for value in (reference_sectors or [])}
    detector_sectors = {str(value).strip().lower()
                        for value in (detector_sectors or [])}
    return bool(
        str(current_state).lower() in {"partial", "satisfied"} and
        reference_sectors.intersection({"rear_left", "rear", "rear_right"}) and
        detector_sectors.intersection({"rear_left", "rear", "rear_right"}) and
        "front" not in reference_sectors and
        "front" not in detector_sectors and
        not reverse_observed and not stationary and
        str(motion_fit).lower() != "contradicts" and
        float(horizontal_displacement_m) >= 1.0 and
        float(traveled_distance_m) >= 1.5)


def circumnavigation_stage_route_integrity_supported(
        instruction_text, *, current_edge_traveled_m,
        stage_progress_context=None, current_selected_yaw_rad=None,
        maximum_interedge_bend_deg=90.0):
    """Require a coherent physical route before accepting a far-side result.

    A single short edge can make a repeated couch/table proposal appear in
    rear sectors even after recovery took the agent back toward its start.
    Explicit around/backside/far-side relations therefore need either one
    substantial edge or the persisted preceding partial edge plus a
    direction-consistent current edge.  All inputs are online graph/action
    history; no depth, goal position, reference trajectory, or episode ID is
    consumed.
    """
    text = str(instruction_text or "").lower()
    far_side_relation = bool(re.search(
        r"\b(?:around|backside|back side|behind|far side|beyond)\b", text))
    if not far_side_relation:
        return True
    current_distance = float(current_edge_traveled_m or 0.0)
    if current_distance >= 4.0:
        return True
    context = (stage_progress_context
               if isinstance(stage_progress_context, dict) else {})
    if not context.get("active"):
        return False
    prior_distance = float(context.get(
        "prior_edge_traveled_distance_m", 0.0) or 0.0)
    prior_motion = context.get("prior_motion_evidence", {}) or {}
    if (prior_distance + current_distance < 4.0 or
            str(prior_motion.get(
                "instruction_motion_fit", "neutral")).lower() ==
            "contradicts" or
            bool(prior_motion.get("stationary_or_blocked"))):
        return False
    prior_yaw = context.get("prior_selected_yaw_rad")
    if prior_yaw is None or current_selected_yaw_rad is None:
        return False
    bend = abs(math.degrees((float(current_selected_yaw_rad) -
                             float(prior_yaw) + math.pi) %
                            (2.0 * math.pi) - math.pi))
    return bool(bend <= float(maximum_interedge_bend_deg) + 1e-6)


def between_stage_route_integrity_supported(
        instruction_text="", *, current_edge_traveled_m,
        stage_progress_context=None,
        current_selected_yaw_rad=None, minimum_stage_travel_m=2.25,
        maximum_interedge_bend_deg=60.0):
    """Reject a shallow beside-object edge as a completed BETWEEN stage.

    Seeing the two named objects in opposite side cameras proves a candidate
    gap but not that the agent has physically entered it. Accept either one
    substantive edge or two direction-consistent real edges whose cumulative
    motion reaches the same form-level threshold. Inputs are persisted online
    node/edge data only; no depth, reference route, goal, or episode identity.
    """
    current_distance = float(current_edge_traveled_m or 0.0)
    text = str(instruction_text or "").lower()
    # Reaching/entering the bracketed gap is a nearer semantic endpoint than
    # clearing/passing through the full gap. Preserve the stronger default for
    # full traversal, but do not veto clean RGB evidence merely because a
    # gap-center point is less than 2.25 m from its source node.
    gap_center_endpoint = bool(
        re.search(r"\b(?:reach|enter|arrive at|get to)\b[^.;]{0,60}"
                  r"\b(?:gap|between)\b", text) and
        not re.search(
            r"\b(?:clear(?:\s+the)?\s+gap|(?:walk|pass|go)\s+through|"
            r"beyond|past|exit)\b", text))
    threshold = min(float(minimum_stage_travel_m), 1.25) if (
        gap_center_endpoint) else float(minimum_stage_travel_m)
    if current_distance >= threshold:
        return True
    context = (stage_progress_context
               if isinstance(stage_progress_context, dict) else {})
    if not context.get("active"):
        return False
    prior_distance = float(context.get(
        "prior_edge_traveled_distance_m", 0.0) or 0.0)
    prior_motion = context.get("prior_motion_evidence", {}) or {}
    if (prior_distance + current_distance < threshold or
            str(prior_motion.get(
                "instruction_motion_fit", "neutral")).lower() ==
            "contradicts" or
            bool(prior_motion.get("stationary_or_blocked"))):
        return False
    prior_yaw = context.get("prior_selected_yaw_rad")
    if prior_yaw is None or current_selected_yaw_rad is None:
        return False
    bend = abs(math.degrees((float(current_selected_yaw_rad) -
                             float(prior_yaw) + math.pi) %
                            (2.0 * math.pi) - math.pi))
    return bool(bend <= float(maximum_interedge_bend_deg) + 1e-6)


def terminal_extent_front_landmark_clear(instruction_text, detections):
    """Reject a claimed route end while its named extent continues ahead.

    This complements form-specific endpoint recovery.  A compound clause such
    as "enter the kitchen and walk along the counter to its end" is not
    complete merely on entering the room.  When the instruction explicitly
    asks for a terminal extent and an instruction-relevant landmark remains a
    strong exact-front RGB-aligned detection, the endpoint is still visually
    open.  Side/rear detections are allowed at a genuine corner or endpoint.
    """
    text = str(instruction_text or "").lower()
    terminal_extent = bool(re.search(
        r"\b(?:far end|the end|until|all the way|entire length|"
        r"passed? the length|reach the end)\b", text))
    route_extent = bool(re.search(
        r"\b(?:along|follow|continue|length|until|all the way)\b", text))
    if not (terminal_extent and route_extent):
        return True
    return not any(
        str(hit.get("direction", "")).lower() == "front" and
        float(hit.get("score", 0.0) or 0.0) >= 0.30
        for hit in (detections or []))


def terminal_extent_rgb_rear_clear_supported(
        *, current_target_state, current_completion_cue, same_instance,
        semantic_order, boundary_event, keyframe_support, motion_fit,
        stationary_or_blocked, reverse_transition_observed,
        current_reference_sectors):
    """Let complete clean-RGB chronology resolve one detector front outlier.

    Terminal ``along ... to the end`` clauses are unfinished when the named
    extent remains ahead. A broad open-vocabulary detector can nevertheless
    attach the landmark query to unrelated forward furniture. Override that
    single detector veto only when the structured clean-RGB result itself is
    fully complete, tracks the same instance, and confines it to the side/rear
    hemisphere after an ordered, motion-supported transition.
    """
    sectors = {str(value).lower() for value in current_reference_sectors or []}
    return bool(
        current_target_state == "satisfied" and
        current_completion_cue == "satisfied" and
        same_instance == "yes" and
        semantic_order == "instructed" and
        boundary_event in {"observed", "endpoint_inferred"} and
        keyframe_support and motion_fit == "supports" and
        not stationary_or_blocked and not reverse_transition_observed and
        sectors and
        not sectors.intersection({"front", "front_left", "front_right"}) and
        sectors.intersection({"left", "rear_left", "rear", "rear_right", "right"}))


def terminal_extent_rgb_completion_supported(
        instruction_text, *, current_target_state, current_completion_cue,
        same_instance, semantic_order, keyframe_support, motion_fit,
        stationary_or_blocked, reverse_transition_observed,
        current_reference_sectors, cumulative_stage_travel_m):
    """Recognize a completed finite landmark extent from clean RGB chronology.

    This is the positive counterpart to the front detector veto. It is usable
    only for explicit along/follow-to-the-end clauses, after at least four
    metres of coherent current-stage translation (possibly across two point
    edges), with the same landmark confined to side/rear bearings. It prevents
    one forward open-vocabulary false positive from turning a proven endpoint
    into an off-route recovery.
    """
    text = str(instruction_text or "").lower()
    terminal_extent = bool(re.search(
        r"\b(?:far end|the end|until|all the way|entire length|"
        r"passed? the length|reach the end)\b", text))
    route_extent = bool(re.search(
        r"\b(?:along|follow|continue|length|until|all the way)\b", text))
    sectors = {str(value).lower() for value in current_reference_sectors or []}
    return bool(
        terminal_extent and route_extent and
        current_target_state in {"partial", "satisfied"} and
        current_completion_cue in {"partial", "satisfied"} and
        same_instance == "yes" and semantic_order == "instructed" and
        keyframe_support and motion_fit == "supports" and
        not stationary_or_blocked and not reverse_transition_observed and
        float(cumulative_stage_travel_m or 0.0) >= 4.0 and sectors and
        not sectors.intersection({"front", "front_left", "front_right"}) and
        sectors.intersection({"left", "rear_left", "rear", "rear_right", "right"}))


def terminal_relation_detection_supported(form, detections):
    """Require observed endpoint identity for landmark terminal relations."""
    form = str(form or "").upper()
    if form not in {"APPROACH_LANDMARK", "STOP_WAIT"}:
        return True
    return any(
        float(hit.get("score", 0.0) or 0.0) >= 0.30 and
        float(hit.get("area_fraction", 0.0) or 0.0) >= 0.005 and
        bool(hit.get("matched_tokens"))
        for hit in (detections or []))


def terminal_relation_rgb_consensus_supported(
        *, same_instance, current_target_state, current_completion_cue,
        current_reference_sectors):
    """Allow strong clean-RGB endpoint consensus to survive a detector miss.

    This is only a detector-veto fallback; it cannot complete a relation by
    itself.  The structured RGB decision must agree on the same instance, the
    full target and cue, and at least one explicit panorama sector.
    """
    return bool(
        same_instance == "yes" and
        current_target_state == "satisfied" and
        current_completion_cue == "satisfied" and
        current_reference_sectors)


def under_relation_multiview_supported(instruction_text, detections):
    """Reject a one-view detector hit as proof of an under/beneath relation.

    An overhead structure at the camera's location has panorama support beyond
    one narrow bearing.  Requiring two RGB-aligned sectors suppresses common
    open-vocabulary false positives on ceiling trim and cabinets while keeping
    the decision independent of depth, navmesh, demonstration paths and
    episode identity.
    """
    text = str(instruction_text or "").lower()
    matches = list(re.finditer(
        r"\b(?:under|beneath)\s+([a-z][a-z\s-]{1,80})", text))
    if not matches:
        return True
    qualifier_tokens = set()
    for match in matches:
        phrase = re.split(
            r"[.;,]|\b(?:and|after|before|until|then)\b",
            match.group(1), maxsplit=1)[0]
        qualifier_tokens.update(re.findall(r"[a-z]+", phrase))
    supported_views = {
        int(hit.get("view_index", -1))
        for hit in (detections or [])
        if float(hit.get("score", 0.0) or 0.0) >= 0.30 and
        qualifier_tokens.intersection(
            str(token).lower() for token in
            (hit.get("matched_tokens", []) or []))
    }
    return len(supported_views) >= 2


def between_gap_lateral_bracket_supported(
        form, instruction_text, landmark, detections):
    """Require a gap endpoint to be laterally bracketed by both references.

    ``reach/walk between A and B`` ends in the gap, so a front object plus a
    rear object is not enough even if a VLM describes both as passed. Explicit
    ``through/past/beyond/clear`` wording instead permits an endpoint beyond
    the pair. This check uses only current-node RGB-aligned detections.
    """
    if str(form or "").upper() != "BETWEEN_OBJECTS":
        return True
    text = " ".join(str(instruction_text or "").lower().split())
    if re.search(r"\b(?:through|past|beyond|clear(?:ed)?)\b", text):
        return True
    if not re.search(r"\b(?:between|gap)\b", text):
        return True
    pair_words = list(dict.fromkeys(instruction_tokens(landmark)))[:4]
    if len(pair_words) < 2:
        return False
    lateral = {
        "left": {"left", "front_left", "rear_left"},
        "right": {"right", "front_right", "rear_right"},
    }
    directions = {word: set() for word in pair_words}
    for hit in detections or []:
        if float(hit.get("score", 0.0) or 0.0) < 0.30:
            continue
        direction = str(hit.get("direction", "")).lower()
        for token in hit.get("matched_tokens", []) or []:
            normalized_tokens = instruction_tokens(str(token).lower())
            for normalized_token in normalized_tokens:
                if normalized_token in directions:
                    directions[normalized_token].add(direction)
    for first_index, first in enumerate(pair_words):
        for second in pair_words[first_index + 1:]:
            if ((directions[first] & lateral["left"] and
                 directions[second] & lateral["right"]) or
                    (directions[first] & lateral["right"] and
                     directions[second] & lateral["left"])):
                return True
    return False


def turn_direction_motion_supported(form, structured_motion,
                                    minimum_turn_deg=45.0):
    """Validate turn direction from persisted online node/edge geometry."""
    form = str(form or "").upper()
    if form not in {"TURN_LEFT", "TURN_RIGHT", "TURN_AROUND"}:
        return True
    heading = float((structured_motion or {}).get(
        "endpoint_heading_delta_deg", 0.0) or 0.0)
    cumulative = float((structured_motion or {}).get(
        "signed_cumulative_turn_deg", 0.0) or 0.0)
    threshold = float(minimum_turn_deg)
    if form == "TURN_AROUND":
        return max(abs(heading), abs(cumulative)) >= 120.0
    expected_sign = 1.0 if form == "TURN_LEFT" else -1.0
    strong = [value for value in (heading, cumulative)
              if abs(value) >= threshold]
    # One strong observed sign is required; two strong signals must agree.
    # This vetoes VLM prose that calls an actual right turn "left".
    return bool(strong and all(value * expected_sign > 0.0 for value in strong))


def explicit_turn_selection_supported(form, point_selection_review):
    """Validate a committed left/right turn from its frozen panorama sector.

    Panorama selection physically rotates the camera onto the selected ray
    before point execution, so that initial turn is not repeated in the
    executor action log. It is valid edge evidence only when the generic
    three-view direction gate was active and the frozen choice remains inside
    the requested 45/90/135-degree side sector.
    """
    form = str(form or "").upper()
    expected_sector = {"TURN_LEFT": "left", "TURN_RIGHT": "right"}.get(form)
    if expected_sector is None:
        return False
    review = point_selection_review or {}
    gate = review.get("direction_gate", {}) or {}
    if (not gate.get("active") or not gate.get("three_view_gate") or
            str(gate.get("sector", "")).lower() != expected_sector):
        return False
    try:
        view_index = int(review.get("view_index"))
        if review.get("refinement_accepted"):
            relative_yaw_deg = float(review.get("refined_relative_yaw_deg"))
        else:
            relative_yaw_deg = (45.0, 90.0, 135.0, 180.0,
                                -135.0, -90.0, -45.0, 0.0)[view_index - 1]
    except (TypeError, ValueError):
        return False
    relative_yaw_deg = (relative_yaw_deg + 180.0) % 360.0 - 180.0
    if form == "TURN_LEFT":
        return bool(view_index in {1, 2, 3} and
                    30.0 <= relative_yaw_deg <= 150.0)
    return bool(view_index in {5, 6, 7} and
                -150.0 <= relative_yaw_deg <= -30.0)


def following_landmark_route_commitment_supported(point_selection_review):
    """Require the next clause's landmark to corroborate a towards-route.

    The following landmark is only an ordered route witness; it cannot prove
    that the later clause is complete.  Requiring a confident same/adjacent
    view relation prevents generic turn motion from completing an arbitrary
    branch while allowing ``turn ... and walk towards X`` to end at a real
    committed-route node rather than at X itself.
    """
    review = point_selection_review or {}
    try:
        following_view_index = int(review.get(
            "first_following_landmark_view_index", -1))
    except (TypeError, ValueError):
        return False
    return bool(
        float(review.get("confidence", 0.0) or 0.0) >= 0.75 and
        str(review.get("first_following_landmark", "")).strip() and
        following_view_index >= 0 and
        str(review.get("sequence_alignment", "")) in {
            "same_view", "adjacent_view"})


def compound_turn_requires_semantic_endpoint(
        form, instruction_text, directional_towards_is_commitment=False):
    """Whether a turn clause also names a destination that must be reached."""
    if str(form or "").upper() not in {
            "TURN_LEFT", "TURN_RIGHT", "TURN_AROUND"}:
        return False
    text = " ".join(str(instruction_text or "").lower().split())
    continuation = re.search(
        r"\b(?:and|then)\s+(?:walk|go|head|move|proceed|continue|cross|"
        r"enter|exit|pass|follow|approach)\b", text)
    hard_destination = re.search(
        r"\bturn\b.{0,80}\b(?:into|through|across|until|past|to (?:the )?"
        r"(?:door|doorway|opening|entrance|exit|end|far side))\b", text)
    if hard_destination:
        return True
    if not continuation:
        return False
    # ``walk towards X`` specifies a committed route direction, not that X
    # itself must already have been reached.  R2R often follows it with a
    # separate endpoint clause.  Preserve an explicit endpoint when the same
    # clause also says to/into/through/across/until/past a destination.
    if (directional_towards_is_commitment and
            re.search(r"\b(?:walk|go|head|move|proceed)\s+towards?\b", text)):
        return bool(re.search(
            r"\b(?:into|through|across|until|past|to (?:the )?"
            r"(?:door|doorway|opening|entrance|exit|end|far side))\b", text))
    return True


def compound_turn_portal_threshold_supported(
        *, form, instruction_text, current_reference_sectors,
        semantic_order, boundary_event, keyframe_support,
        same_reference_instance,
        motion_fit, stationary, reverse_observed,
        deterministic_turn_direction, horizontal_displacement_m,
        traveled_distance_m):
    """Recognize a side doorway straddling the camera at a turn endpoint.

    At a doorway threshold, overlapping 45-degree cameras can place the same
    frame in FRONT_SIDE and REAR_SIDE simultaneously.  The terminal camera
    side is deliberately independent of the commanded turn side: the point
    executor can correct its heading while crossing the room, whereas the
    commanded turn is already checked by ``deterministic_turn_direction``.
    Requiring both to have the same sign therefore rejects real thresholds
    after a harmless final heading correction.
    """
    form = str(form or "").upper()
    if form not in {"TURN_LEFT", "TURN_RIGHT"}:
        return False
    text = " ".join(str(instruction_text or "").lower().split())
    if not re.search(r"\b(?:door|doorway|opening|entrance|threshold)\b", text):
        return False
    sectors = {str(value).lower() for value in current_reference_sectors}
    threshold_straddles_side = bool(
        ({"front_left", "rear_left"} <= sectors) or
        ({"front_right", "rear_right"} <= sectors))
    return bool(
        threshold_straddles_side and
        semantic_order == "instructed" and
        boundary_event in {"observed", "endpoint_inferred"} and
        keyframe_support and
        same_reference_instance == "yes" and motion_fit == "supports" and
        not stationary and not reverse_observed and
        deterministic_turn_direction and
        float(horizontal_displacement_m) >= 1.0 and
        float(traveled_distance_m) >= 1.5)


def landmark_center_refined_yaw_deg(view_yaw_deg, landmark_centering,
                                    landmark_horizontal_side,
                                    offset_deg=22.5):
    """Return a camera yaw that centers an off-axis landmark in clean RGB.

    Habitat positive yaw is physical left.  This is a fixed local acquisition
    step, not a range or route estimate: it uses only the VLM's image-side
    observation and the selected panorama camera yaw.
    """
    centering = str(landmark_centering or "").lower()
    side = str(landmark_horizontal_side or "").lower()
    if centering == "centered" and side == "center":
        return None
    if centering != "off_center" or side not in {"left", "right"}:
        raise ValueError(
            "landmark centering/side must be centered+center or "
            "off_center+(left|right)")
    value = float(view_yaw_deg) + (float(offset_deg) if side == "left" else
                                   -float(offset_deg))
    value = (value + 180.0) % 360.0 - 180.0
    return 180.0 if abs(value + 180.0) < 1e-6 else value


def ambiguous_landmark_spelling_aliases(landmark_text, candidate_phrases,
                                        maximum_edit_distance=1):
    """Find unknown landmark tokens with multiple equally near aliases.

    Detector query expansion can legitimately map a misspelling to more than
    one common visual noun. The VLM must not silently select whichever object
    is most salient. This lexical diagnostic is derived only from instruction
    text and online query phrases.
    """
    ignored = {
        "the", "and", "with", "toward", "towards", "large", "small",
        "wooden", "white", "black", "brown", "gray", "grey", "red",
        "blue", "green", "left", "right", "front", "near",
    }
    source = [word for word in re.findall(
        r"[a-z]+", str(landmark_text or "").lower())
              if len(word) >= 4 and word not in ignored]
    vocabulary = {
        word for phrase in (candidate_phrases or [])
        for word in re.findall(r"[a-z]+", str(phrase).lower())
        if len(word) >= 4 and word not in ignored
    }

    def edit_distance(first, second):
        previous = list(range(len(second) + 1))
        for row, first_char in enumerate(first, 1):
            current = [row]
            for column, second_char in enumerate(second, 1):
                current.append(min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (first_char != second_char)))
            previous = current
        return previous[-1]

    result = {}
    for token in source:
        aliases = sorted({
            candidate for candidate in vocabulary if candidate != token and
            edit_distance(token, candidate) <= int(maximum_edit_distance)})
        if len(aliases) >= 2:
            result[token] = aliases
    return result


def ambiguous_landmark_executable_view(
        selected_index, allowed_detection_indices, candidates,
        minimum_ground_gain_ratio=1.5):
    """Prefer a substantially more executable view of an ambiguous noun.

    This applies only after lexical ambiguity and RGB landmark support are
    established. It does not infer object identity: it prevents a salient,
    object-dominated crop from beating another legal landmark view with much
    more strict connected ground for the point executor.
    """
    selected_index = int(selected_index)
    legal = [int(index) for index in allowed_detection_indices
             if 0 <= int(index) < len(candidates)]
    if selected_index not in legal or not legal:
        return selected_index
    ground = lambda index: float(candidates[index].get(
        "ground_fraction", 0.0) or 0.0)
    best = max(legal, key=lambda index: (ground(index), -index))
    selected_ground = ground(selected_index)
    required = max(selected_ground * float(minimum_ground_gain_ratio),
                   selected_ground + 0.03)
    return best if ground(best) >= required else selected_index


_ENV_ASSIGNMENT = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def _read_env_file(path, strict=True):
    """Read KEY=VALUE / export KEY=VALUE lines used for local API configuration.

    With ``strict=False`` any line that is not a plain assignment is skipped so
    a real shell script such as ``local_env.sh`` (conda activation, ``cd``,
    ``eval`` lines) can be scanned for exported credentials without sourcing it.
    """
    if path is None:
        return {}
    path = Path(path).expanduser()
    if not path.exists():
        return {}
    values = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _ENV_ASSIGNMENT.match(line)
        if match is None:
            if strict:
                raise RuntimeError(
                    f"Invalid environment entry at {path}:{line_number}; expected KEY=VALUE")
            continue
        key, value = match.group(1), match.group(2).strip()
        if value[:1] == value[-1:] and value[:1] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def _valid_api_key(value):
    normalized = str(value or "").strip()
    return bool(
        normalized and normalized not in RELAY_PLACEHOLDER_KEYS
        and not normalized.startswith("${"))


def _normalize_base_url(url):
    """Accept either a bare API root or a full .../chat/completions endpoint."""
    normalized = str(url or "").strip().rstrip("/")
    suffix = "/chat/completions"
    if normalized.endswith(suffix):
        normalized = normalized[:-len(suffix)]
    return normalized or DEFAULT_DEEPSEEK_BASE_URL


def _truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_json_object(content):
    """Parse an API model response while tolerating a fenced JSON object."""
    if isinstance(content, list):
        content = "".join(
            item.get("text", "") for item in content if isinstance(item, dict))
    if not isinstance(content, str):
        raise RuntimeError(f"VLM returned unsupported content type: {type(content).__name__}")
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise RuntimeError(f"VLM did not return JSON: {content[:500]}") from exc


class VLMBackend(ABC):
    """Minimal interface required by the navigation harness."""

    @abstractmethod
    def generate_json(self, prompt, images, schema):
        raise NotImplementedError


class OllamaBackend(VLMBackend):
    def __init__(self, model=DEFAULT_OLLAMA_MODEL, host="http://127.0.0.1:11434",
                 timeout=180):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    @staticmethod
    def _encode(image):
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG", quality=90)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def generate_json(self, prompt, images, schema):
        message = {"role": "user", "content": prompt}
        if images:
            message["images"] = [self._encode(image) for image in images]
        is_stage_decomposition = "stages" in schema.get("properties", {})
        is_node_sequence_classification = (
            "visual_evidence" in schema.get("properties", {}) and
            "matched_sub_instruction_id" in schema.get("properties", {}))
        payload = {
            "model": self.model, "messages": [message], "stream": False,
            "format": schema, "options": {
                "temperature": 0, "seed": 17,
                "num_predict": (
                    1024 if is_stage_decomposition else
                    256 if is_node_sequence_classification else 96),
            },
        }
        request = urllib.request.Request(
            f"{self.host}/api/chat", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"Ollama HTTP {exc.code} at {self.host}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Ollama request failed at {self.host}: {exc}") from exc
        content = result.get("message", {}).get("content", "")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            partial = {}
            for key in ("view_index", "x_norm", "y_norm"):
                value = re.search(rf'"{key}"\s*:\s*(-?[0-9]+(?:\.[0-9]+)?)', content)
                if value:
                    partial[key] = float(value.group(1))
            if "view_index" in partial:
                partial["view_index"] = int(partial["view_index"])
            if {"x_norm", "y_norm"}.issubset(partial):
                partial["reason"] = "salvaged_from_truncated_json"
                return partial
            raise RuntimeError(f"VLM did not return JSON: {content[:500]}") from exc


class DeepSeekBackend(VLMBackend):
    """DeepSeek vision model behind an OpenAI-compatible relay (DMXAPI).

    Configuration precedence for every setting is: explicit argument >
    environment > ``.env.deepseek`` (``env_file``) > repository ``local_env.sh``
    > built-in default.  Environment/file lookups try the Navi-Agent relay
    names first (``OPENROUTER_API_KEY``, ``OPENROUTER_API_URL``,
    ``NAVI_OPENROUTER_MODEL``, ``NAVI_LLM_DISABLE_THINKING``) and then the
    older ``DEEPSEEK_*`` names.
    """

    def __init__(self, model=None, base_url=None, api_key=None, env_file=None,
                 timeout=180, image_detail="high", thinking=None,
                 local_env_path=LOCAL_ENV_SH_PATH, disable_proxy=None,
                 max_attempts=3, retry_base_s=None, retry_cap_s=None):
        self.env_file = Path(env_file).expanduser() if env_file is not None else None
        self.local_env_path = (
            Path(local_env_path).expanduser() if local_env_path is not None else None)
        # (source label, mapping) in precedence order after explicit arguments.
        self._sources = [("environment", os.environ)]
        if self.env_file is not None:
            self._sources.append((str(self.env_file), _read_env_file(self.env_file)))
        if self.local_env_path is not None:
            self._sources.append((
                str(self.local_env_path),
                _read_env_file(self.local_env_path, strict=False)))

        if api_key:
            self.api_key, self.credential_source = api_key, "argument"
        else:
            self.api_key, self.credential_source = self._lookup(RELAY_API_KEY_ENV)
        if not _valid_api_key(self.api_key):
            self.api_key = ""
        self.base_url = _normalize_base_url(
            base_url or self._lookup(RELAY_API_URL_ENV)[0] or DEFAULT_DEEPSEEK_BASE_URL)
        self.model = (
            model or self._lookup(RELAY_MODEL_ENV)[0] or DEFAULT_DEEPSEEK_MODEL)
        self.thinking = thinking or self._resolve_thinking()
        if self.thinking not in {"enabled", "disabled"}:
            raise RuntimeError("DEEPSEEK_THINKING must be enabled or disabled")
        if disable_proxy is None:
            configured = self._lookup(("NAVI_OPENROUTER_DISABLE_PROXY",))[0]
            disable_proxy = True if configured == "" else _truthy(configured)
        self.disable_proxy = bool(disable_proxy)
        # An empty ProxyHandler drops any inherited http(s)_proxy variables so
        # relay traffic never depends on the manual mihomo proxy being up.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.timeout = timeout
        self.image_detail = image_detail
        self.max_attempts = max(1, int(max_attempts))
        self.retry_base_s = float(
            retry_base_s if retry_base_s is not None
            else self._lookup(("LLM_HTTP_RETRY_BASE_S",))[0] or 2.0)
        self.retry_cap_s = float(
            retry_cap_s if retry_cap_s is not None
            else self._lookup(("LLM_HTTP_RETRY_CAP_S",))[0] or 40.0)
        self.last_usage = None
        self.last_finish_reason = None
        self.last_call_meta = None

    def _lookup(self, keys):
        """Return (value, source) for the first non-empty key across sources."""
        for label, mapping in self._sources:
            for key in keys:
                value = str(mapping.get(key, "") or "").strip()
                if value:
                    return value, f"{label}:{key}"
        return "", None

    def _resolve_thinking(self):
        disable_flag = self._lookup(("NAVI_LLM_DISABLE_THINKING",))[0]
        if disable_flag:
            return "disabled" if _truthy(disable_flag) else "enabled"
        return self._lookup(("DEEPSEEK_THINKING",))[0] or "disabled"

    @property
    def endpoint(self):
        return f"{self.base_url}/chat/completions"

    @staticmethod
    def _encode_data_url(image):
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG", quality=90)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
    def _token_limit(schema):
        properties = schema.get("properties", {})
        if "stages" in properties:
            return 4096
        if "visual_evidence" in properties:
            return 1024
        return 512

    def _open(self, request):
        if self.disable_proxy:
            return self._opener.open(request, timeout=self.timeout)
        return urllib.request.urlopen(request, timeout=self.timeout)

    def _post_once(self, request):
        """One HTTP round trip; returns (parsed JSON body, http status)."""
        with self._open(request) as response:
            return json.load(response), getattr(response, "status", 200)

    def generate_json(self, prompt, images, schema):
        if not self.api_key:
            locations = " / ".join(
                str(path) for path in (self.local_env_path, self.env_file) if path)
            raise VLMProviderFatalError(
                "DeepSeek relay API key is missing or a placeholder; export "
                "OPENROUTER_API_KEY (or DEEPSEEK_API_KEY) in "
                f"{locations or 'the environment'} before running semantic navigation")

        schema_text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        content = [{
            "type": "text",
            "text": (
                f"{prompt}\nThe response must be one JSON object matching this JSON "
                f"Schema exactly:\n{schema_text}\nReturn JSON only."),
        }]
        content.extend({
            "type": "image_url",
            "image_url": {
                "url": self._encode_data_url(image),
                "detail": self.image_detail,
            },
        } for image in images)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "stream": False,
            "response_format": {"type": "json_object"},
            "thinking": {"type": self.thinking},
            "max_tokens": self._token_limit(schema),
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        self.last_call_meta = {
            "endpoint": self.endpoint,
            "model": self.model,
            "credential_source": self.credential_source,
            "thinking": self.thinking,
            "proxy_disabled": self.disable_proxy,
            "attempts": 0,
            "http_status": None,
            "elapsed_ms": None,
        }
        started = time.monotonic()
        result = None
        last_error = None
        for attempt in range(self.max_attempts):
            self.last_call_meta["attempts"] = attempt + 1
            try:
                result, status = self._post_once(request)
                self.last_call_meta["http_status"] = status
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
                self.last_call_meta["http_status"] = exc.code
                message = f"DeepSeek HTTP {exc.code} at {self.base_url}: {detail}"
                if exc.code in {401, 402, 403}:
                    raise VLMProviderFatalError(message) from exc
                if exc.code not in RELAY_RETRYABLE_HTTP:
                    raise RuntimeError(message) from exc
                last_error = RuntimeError(message)
            except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                last_error = RuntimeError(
                    f"DeepSeek request failed at {self.base_url}: {exc}")
            except json.JSONDecodeError as exc:
                raise RuntimeError("DeepSeek returned a non-JSON HTTP response") from exc
            if attempt + 1 < self.max_attempts:
                time.sleep(min(self.retry_cap_s,
                               self.retry_base_s ** attempt + random.random()))
        self.last_call_meta["elapsed_ms"] = int((time.monotonic() - started) * 1000)
        if result is None:
            raise RuntimeError(
                f"{last_error} (after {self.last_call_meta['attempts']} attempts)"
            ) from last_error

        choices = result.get("choices") or []
        if not choices:
            raise RuntimeError(f"DeepSeek response has no choices: {str(result)[:500]}")
        self.last_usage = result.get("usage")
        self.last_finish_reason = choices[0].get("finish_reason")
        content = choices[0].get("message", {}).get("content", "")
        if self.last_finish_reason == "length":
            raise RuntimeError(
                "DeepSeek JSON was truncated at max_tokens; reduce output or raise the limit")
        return _parse_json_object(content)


class HeuristicBackend(VLMBackend):
    """Explicit test backend; never selected as an implicit Ollama fallback."""

    def generate_json(self, prompt, images, schema):
        if "STAGE_DECOMPOSITION" in prompt:
            instruction = prompt.split("R2R instruction:", 1)[-1].split("\n", 1)[0].strip()
            parts = [x.strip() for x in re.split(r"(?<=[.!?])\s+|\bthen\b", instruction)
                     if x.strip()]
            return {"stages": [{
                "stage_id": i, "navigation_instruction": part,
                "landmark": "unspecified", "completion_cue": "sub-instruction completed",
                "semantic_spatial_target": "walkable floor that completes this sub-instruction",
                "spatial_relation": "in the instructed direction",
                "visual_arrival_evidence": "the instructed spatial transition is complete",
                "forbidden_target": "walls, objects, and non-walkable regions",
            } for i, part in enumerate(parts or [instruction])]}
        if "NODE_SUB_INSTRUCTION_SEQUENCE_CLASSIFICATION" in prompt:
            expected = re.search(r"Expected sub_instruction_id:\s*([0-9]+)", prompt)
            matched = int(expected.group(1)) if expected else -1
            return {
                "belongs_to_sequence": matched >= 0,
                "matched_sub_instruction_id": matched,
                "confidence": 0.75,
                "reason": "heuristic sequence-classifier test backend",
                "visual_evidence": "test backend assumes the expected node",
            }
        if "UNORDERED_ENDPOINT_INSTRUCTION_ROLE" in prompt:
            return {
                "node_x_role": "source_context",
                "node_y_role": "completion_target",
                "confidence": 0.75,
                "reason": "heuristic endpoint-role test backend",
                "visual_evidence": "node X precedes node Y",
            }
        if "You are the RGB-only edge completion judge" in prompt:
            # A rule stub cannot see semantic change, so it never claims
            # completion; this keeps key-free smoke runs alive past arrival.
            return {
                "status": "unknown",
                "confidence": 0.5,
                "reason": "heuristic rgb-only edge judge test backend",
                "visual_evidence": "test backend does not inspect RGB",
                "temporal_evidence": "no chronological evidence evaluated",
            }
        if "EDGE_INSTRUCTION_COMPLETION_JUDGMENT" in prompt:
            if "endpoint_evidence" in schema.get("properties", {}):
                return {
                    "status": "completed", "confidence": 0.75,
                    "endpoint_evidence": {
                        "previous_target_state": "unsatisfied",
                        "current_target_state": "satisfied",
                        "current_completion_cue": "satisfied",
                        "reference_previous_sectors": ["front"],
                        "reference_current_sectors": ["rear"],
                    },
                    "temporal_evidence": {
                        "semantic_order": "instructed",
                        "boundary_event": "observed",
                        "keyframe_support": True,
                        "same_reference_instance": "yes",
                        "reverse_transition_observed": False,
                    },
                    "motion_evidence": {
                        "instruction_motion_fit": "supports",
                        "stationary_or_blocked": False,
                    },
                    "reason": "heuristic structured edge-completion test backend",
                    "visual_evidence": "test backend assumes a supported transition",
                }
            if "observed_status" in schema.get("properties", {}):
                return {
                    "observed_status": "arrived",
                    "reversed_status": "unknown",
                    "direction_fit": "observed",
                    "confidence": 0.75,
                    "reason": "heuristic bidirectional edge test backend",
                    "visual_evidence": "observed order matches the instruction",
                }
            status_values = (
                schema.get("properties", {}).get("status", {}).get("enum", []))
            three_way = "arrived" in status_values
            result = {
                "status": "arrived" if three_way else "completed",
                "confidence": 0.75,
                "reason": "heuristic edge-completion test backend",
                "visual_evidence": "test backend assumes completion",
            }
            if "progress_evidence" in schema.get("properties", {}):
                result["progress_evidence"] = {
                    "completion_boundary_observed": True,
                    "instruction_consistent_progress": True,
                    "contradiction_observed": False,
                }
            if "completion_evidence" in schema.get("properties", {}):
                result["completion_evidence"] = {
                    "completion_boundary_observed": True,
                    "temporal_relation_change_observed": True,
                    "contradiction_observed": False,
                }
            if "directional_evidence" in schema.get("properties", {}):
                result["directional_evidence"] = {
                    "reference_previous_sectors": ["front"],
                    "reference_current_sectors": ["rear"],
                    "reference_previous_view_indices": [0],
                    "reference_current_view_indices": [4],
                    "same_instance_confident": True,
                    "multiple_similar_instances": False,
                    "rear_sector_evidence": True,
                    "front_sector_contradiction": False,
                    "temporal_relation_change": True,
                }
            if "partial_extent_evidence" in schema.get("properties", {}):
                result["partial_extent_evidence"] = {
                    "starts_at_stair_base": True,
                    "continuous_ascent": True,
                    "current_steps_below": True,
                    "current_steps_above": True,
                }
            return result
        if "BACKTRACK_GROUND_TARGET_SELECTION" in prompt:
            recommended = re.search(
                r"Recommended current view from graph geometry:\s*([0-9]+)", prompt)
            view_index = int(recommended.group(1)) if recommended else 0
            return {"view_index": view_index, "anchor_index": 0,
                    "reason": "graph-geometric backtrack fallback"}
        if "STEP_GROUND_TARGET_SELECTION" in prompt:
            return {"x_norm": 0.5, "y_norm": 0.68}
        allowed = re.search(r"allowed view from (\[[0-9, ]+\])", prompt)
        view_index = json.loads(allowed.group(1))[0] if allowed else 0
        return {"view_index": view_index, "anchor_index": 0, "reason": "first safe ground anchor"}


def build_vlm_backend(name, model=None, host="http://127.0.0.1:11434", timeout=180,
                      *, deepseek_env_file=None, deepseek_base_url=None,
                      deepseek_api_key=None, max_attempts=3):
    factories = {
        "ollama": lambda: OllamaBackend(model or DEFAULT_OLLAMA_MODEL, host, timeout),
        "deepseek": lambda: DeepSeekBackend(
            model=model, base_url=deepseek_base_url, api_key=deepseek_api_key,
            env_file=deepseek_env_file, timeout=timeout, max_attempts=max_attempts),
        "heuristic": HeuristicBackend,
    }
    if name not in factories:
        raise ValueError(f"Unknown VLM backend {name!r}; available: {sorted(factories)}")
    return factories[name]()


class NavigationVLMHarness:
    STAGE_SCHEMA = {
        "type": "object", "required": ["stages"], "properties": {
            "stages": {"type": "array", "minItems": 1, "items": {
                "type": "object",
                "required": [
                    "stage_id", "navigation_instruction", "landmark", "completion_cue",
                    "semantic_spatial_target", "spatial_relation",
                    "visual_arrival_evidence", "forbidden_target",
                ],
                "properties": {
                    "stage_id": {"type": "integer"},
                    "navigation_instruction": {"type": "string"},
                    "landmark": {"type": "string"},
                    "completion_cue": {"type": "string"},
                    "semantic_spatial_target": {"type": "string"},
                    "spatial_relation": {"type": "string"},
                    "visual_arrival_evidence": {"type": "string"},
                    "forbidden_target": {"type": "string"},
                },
            }}
        },
    }
    POINT_SCHEMA = {
        "type": "object", "required": ["view_index", "anchor_index", "reason"],
        "properties": {
            "view_index": {"type": "integer"},
            "anchor_index": {"type": "integer", "minimum": 0, "maximum": 5},
            "reason": {"type": "string"},
        },
    }
    STEP_SCHEMA = {
        "type": "object", "required": ["x_norm", "y_norm", "reason"],
        "properties": {
            "x_norm": {"type": "number", "minimum": 0, "maximum": 1},
            "y_norm": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
        },
    }
    NODE_SEQUENCE_SCHEMA = {
        "type": "object",
        "required": [
            "belongs_to_sequence", "matched_sub_instruction_id",
            "confidence", "reason", "visual_evidence",
        ],
        "properties": {
            "belongs_to_sequence": {"type": "boolean"},
            "matched_sub_instruction_id": {"type": "integer", "minimum": -1},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string", "maxLength": 180},
            "visual_evidence": {"type": "string", "maxLength": 180},
        },
    }
    INSTRUCTION_COMPLETION_SCHEMA = {
        "type": "object",
        "required": ["status", "confidence", "reason", "visual_evidence"],
        "properties": {
            "status": {"type": "string", "enum": ["completed", "unknown"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string", "maxLength": 240},
            "visual_evidence": {"type": "string", "maxLength": 240},
        },
    }
    PARTIAL_EXTENT_COMPLETION_SCHEMA = {
        "type": "object",
        "required": [
            "status", "confidence", "reason", "visual_evidence",
            "partial_extent_evidence",
        ],
        "properties": {
            **INSTRUCTION_COMPLETION_SCHEMA["properties"],
            "partial_extent_evidence": {
                "type": "object",
                "required": [
                    "starts_at_stair_base", "continuous_ascent",
                    "current_steps_below", "current_steps_above",
                ],
                "properties": {
                    "starts_at_stair_base": {"type": "boolean"},
                    "continuous_ascent": {"type": "boolean"},
                    "current_steps_below": {"type": "boolean"},
                    "current_steps_above": {"type": "boolean"},
                },
            },
        },
    }
    DIRECTIONAL_EVIDENCE_SCHEMA = {
        "type": "object",
        "required": [
            "reference_previous_sectors", "reference_current_sectors",
            "reference_previous_view_indices", "reference_current_view_indices",
            "same_instance_confident", "multiple_similar_instances",
            "rear_sector_evidence", "front_sector_contradiction",
            "temporal_relation_change",
        ],
        "properties": {
            "reference_previous_sectors": {
                "type": "array", "items": {"type": "string"}},
            "reference_current_sectors": {
                "type": "array", "items": {"type": "string"}},
            "reference_previous_view_indices": {
                "type": "array", "items": {
                    "type": "integer", "minimum": 0, "maximum": 7}},
            "reference_current_view_indices": {
                "type": "array", "items": {
                    "type": "integer", "minimum": 0, "maximum": 7}},
            "same_instance_confident": {"type": "boolean"},
            "multiple_similar_instances": {"type": "boolean"},
            "rear_sector_evidence": {"type": "boolean"},
            "front_sector_contradiction": {"type": "boolean"},
            "temporal_relation_change": {"type": "boolean"},
        },
    }
    EIGHT_VIEW_COMPLETION_SCHEMA = {
        "type": "object",
        "required": [
            "status", "confidence", "reason", "visual_evidence",
            "directional_evidence",
        ],
        "properties": {
            **INSTRUCTION_COMPLETION_SCHEMA["properties"],
            "directional_evidence": DIRECTIONAL_EVIDENCE_SCHEMA,
        },
    }
    EIGHT_VIEW_PARTIAL_EXTENT_COMPLETION_SCHEMA = {
        "type": "object",
        "required": [
            "status", "confidence", "reason", "visual_evidence",
            "directional_evidence", "partial_extent_evidence",
        ],
        "properties": {
            **EIGHT_VIEW_COMPLETION_SCHEMA["properties"],
            "partial_extent_evidence": (
                PARTIAL_EXTENT_COMPLETION_SCHEMA["properties"]
                ["partial_extent_evidence"]),
        },
    }
    THREE_WAY_PROGRESS_SCHEMA = {
        "type": "object",
        "required": [
            "status", "confidence", "reason", "visual_evidence",
            "progress_evidence",
        ],
        "properties": {
            "status": {
                "type": "string",
                "enum": ["arrived", "on_route", "unknown"],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string", "maxLength": 320},
            "visual_evidence": {"type": "string", "maxLength": 320},
            "progress_evidence": {
                "type": "object",
                "required": [
                    "completion_boundary_observed",
                    "instruction_consistent_progress",
                    "contradiction_observed",
                ],
                "properties": {
                    "completion_boundary_observed": {"type": "boolean"},
                    "instruction_consistent_progress": {"type": "boolean"},
                    "contradiction_observed": {"type": "boolean"},
                },
            },
        },
    }
    BINARY_PROGRESS_SCHEMA = {
        "type": "object",
        "required": [
            "status", "confidence", "reason", "visual_evidence",
            "completion_evidence",
        ],
        "properties": {
            "status": {
                "type": "string", "enum": ["completed", "unknown"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string", "maxLength": 320},
            "visual_evidence": {"type": "string", "maxLength": 320},
            "completion_evidence": {
                "type": "object",
                "required": [
                    "completion_boundary_observed",
                    "temporal_relation_change_observed",
                    "contradiction_observed",
                ],
                "properties": {
                    "completion_boundary_observed": {"type": "boolean"},
                    "temporal_relation_change_observed": {"type": "boolean"},
                    "contradiction_observed": {"type": "boolean"},
                },
            },
        },
    }
    STRUCTURED_BINARY_COMPLETION_SCHEMA = {
        "type": "object",
        "required": [
            "status", "confidence", "endpoint_evidence", "temporal_evidence",
            "motion_evidence", "reason", "visual_evidence",
        ],
        "properties": {
            "status": {"type": "string", "enum": ["completed", "unknown"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "endpoint_evidence": {
                "type": "object",
                "required": [
                    "previous_target_state", "current_target_state",
                    "current_completion_cue", "reference_previous_sectors",
                    "reference_current_sectors",
                ],
                "properties": {
                    "previous_target_state": {
                        "type": "string",
                        "enum": ["satisfied", "unsatisfied", "ambiguous"],
                    },
                    "current_target_state": {
                        "type": "string",
                        "enum": ["satisfied", "partial", "unsatisfied", "ambiguous"],
                    },
                    "current_completion_cue": {
                        "type": "string",
                        "enum": ["satisfied", "partial", "absent", "ambiguous"],
                    },
                    "reference_previous_sectors": {
                        "type": "array", "items": {"type": "string"}},
                    "reference_current_sectors": {
                        "type": "array", "items": {"type": "string"}},
                },
            },
            "temporal_evidence": {
                "type": "object",
                "required": [
                    "semantic_order", "boundary_event", "keyframe_support",
                    "same_reference_instance", "reverse_transition_observed",
                ],
                "properties": {
                    "semantic_order": {
                        "type": "string",
                        "enum": ["instructed", "reversed", "no_change", "ambiguous"],
                    },
                    "boundary_event": {
                        "type": "string",
                        "enum": ["observed", "endpoint_inferred", "not_observed", "ambiguous"],
                    },
                    "keyframe_support": {"type": "boolean"},
                    "same_reference_instance": {
                        "type": "string", "enum": ["yes", "no", "not_applicable", "ambiguous"]},
                    "reverse_transition_observed": {"type": "boolean"},
                },
            },
            "motion_evidence": {
                "type": "object",
                "required": ["instruction_motion_fit", "stationary_or_blocked"],
                "properties": {
                    "instruction_motion_fit": {
                        "type": "string", "enum": ["supports", "contradicts", "neutral"]},
                    "stationary_or_blocked": {"type": "boolean"},
                },
            },
            "reason": {"type": "string", "maxLength": 420},
            "visual_evidence": {"type": "string", "maxLength": 420},
        },
    }
    # Stage-C revision: require explicit landing evidence for full stair
    # instructions, so a mid-flight endpoint cannot be mistaken for completion.
    # Non-vertical forms keep the same stable schema and set these flags false.
    STRUCTURED_VERTICAL_BINARY_COMPLETION_SCHEMA = {
        **STRUCTURED_BINARY_COMPLETION_SCHEMA,
        "required": [
            *STRUCTURED_BINARY_COMPLETION_SCHEMA["required"],
            "vertical_endpoint_evidence",
        ],
        "properties": {
            **STRUCTURED_BINARY_COMPLETION_SCHEMA["properties"],
            "vertical_endpoint_evidence": {
                "type": "object",
                "required": [
                    "current_level_landing",
                    "remaining_stair_flight_visible",
                    "endpoint_evidence_stable",
                ],
                "properties": {
                    "current_level_landing": {"type": "boolean"},
                    "remaining_stair_flight_visible": {"type": "boolean"},
                    "endpoint_evidence_stable": {"type": "boolean"},
                },
            },
        },
    }
    BIDIRECTIONAL_PROGRESS_SCHEMA = {
        "type": "object",
        "required": [
            "observed_status", "reversed_status", "direction_fit",
            "confidence", "reason", "visual_evidence",
        ],
        "properties": {
            "observed_status": {
                "type": "string",
                "enum": ["arrived", "on_route", "unknown"],
            },
            "reversed_status": {
                "type": "string",
                "enum": ["arrived", "on_route", "unknown"],
            },
            "direction_fit": {
                "type": "string",
                "enum": ["observed", "reversed", "neither"],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string", "maxLength": 320},
            "visual_evidence": {"type": "string", "maxLength": 320},
        },
    }
    UNORDERED_ENDPOINT_ROLE_SCHEMA = {
        "type": "object",
        "required": [
            "node_x_role", "node_y_role", "confidence", "reason",
            "visual_evidence",
        ],
        "properties": {
            "node_x_role": {
                "type": "string",
                "enum": [
                    "source_context", "intermediate_progress",
                    "completion_target", "no_clear_relation"],
            },
            "node_y_role": {
                "type": "string",
                "enum": [
                    "source_context", "intermediate_progress",
                    "completion_target", "no_clear_relation"],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string", "maxLength": 320},
            "visual_evidence": {"type": "string", "maxLength": 320},
        },
    }

    POINT_SELECTION_PROMPT_VERSIONS = {
        "v1_baseline", "v2_orientation_continuity",
        "v3_orientation_soft_semantic",
        "v4_sector_identity_gates",
        "v5_two_stage_sector_anchor",
        "v6_validated_relation_two_stage",
        "v7_single_stage_relation_gates",
        "v8_object_relation_router",
        "v9_single_stage_identity_relation_gates",
        "v10_approach_relation_router",
        "v11_eight_view_refinement",
        "v12_eight_view_dual_candidate",
        "v13_eight_view_native_rgb",
        "v14_rgb_evidence_refinement",
        "v15_rgb_center_preferred",
        "v16_turn_three_view_gate",
        "v17_turn_three_view_commit_prior",
        "v18_history_safe_refinement",
        "v19_pixel_ray_history_and_reverse_override",
        # First-step route guard: keeps the V17 RGB review contract while
        # adding a first-stage-only route/ground sanity instruction.  The
        # requested version is retained in audit records; internal behavior
        # is selected below so later stages are not accidentally rewritten.
        "v20_first_step_route_guard",
        # Relation-aware first-step review.  This keeps the same RGB-only
        # eight-view contract, but removes the blanket "first step is forward"
        # assumption and adds generic behind/diagonal/hallway/stair rules.
        "v21_relation_aware_route_review",
        # Task-condition tightening for the stricter 30-degree acceptance
        # gate.  V22 keeps V21's relation rules but adds a ray-centered anchor
        # requirement and protects a high-confidence landmark/route decision
        # from a later floor-area-only corridor flip.
        "v22_task30_route_anchor",
        # Stage-00 selection revision: distinguish an explicit named-landmark
        # turn from an ordinary approach and use landmark-bearing rays for
        # stairs/side views without changing the executor or completion model.
        "v23_landmark_turn_stair_ray",
        # Stage-1 route refinement: compound left/right clauses may be nearly
        # forward in the current node, so retain a forward candidate while
        # preserving the strict side gate for bare turn commands.
        "v24_compound_turn_ray_center",
        # Stage-2 route continuity review: preserve the frozen history/turn
        # contract while comparing forward/portal/gap rays generically.
        "v25_stage2_route_continuity",
        # Consensus variant for later-stage route comparison; a side switch is
        # committed only when two independent RGB adjudicators agree.
        "v26_stage2_route_consensus",
        # Stage-2 final-ray geometry: apply the route-angle constraint after
        # the optional local RGB refinement as well as before it.  This keeps
        # the semantic review and the executable floor ray on the same sector.
        "v27_stage2_final_ray_geometry",
        # Stage-2 side-aware stop fallback: when Grounded-SAM has no legal
        # front anchor, acquire one RGB ray between the landmark-bearing side
        # sector and the front sector, using only the independent textual side
        # evidence from the RGB review.
        "v28_stage2_side_aware_stop_ray",
        "v29_stage3_stop_relation_ray",
        # Stage-3 relation/portal ray refinement: use lexical relation cues
        # only when the RGB review actually grounds the landmark, and retain
        # a high-confidence portal bearing instead of a blanket 75-degree cap.
        "v30_stage3_relation_portal_ray",
        # Side-qualified STOP/WAIT relation: acquire an oblique near-side ray
        # instead of aiming through the named doorway/landmark into the
        # destination region.  This is an RGB/2-D form rule, not an episode
        # correction and does not inspect depth, navmesh, or reference paths.
        "v33_stop_relation_near_side",
        # Strict start-stage semantic route gate: an unlabeled
        # circum-navigation clause may require a short forward approach before
        # the side clearance becomes visible.  Keep forward as a legal RGB
        # competitor unless the instruction explicitly names a side.
        "v31_circumnavigate_forward_competitor",
        # Strict first-stage shallow-route adjudication.  For unqualified
        # enter/exit/pass/straight clauses, compare a side proposal against a
        # legal near-forward floor lane before committing to the side ray.
        "v32_first_stage_shallow_route",
        # Cumulative online route policy: keep V31 circum-navigation forward
        # competition, V32 shallow/continuity adjudication and V33 near-side
        # STOP/portal geometry in one version.  Earlier research versions stay
        # selectable for frozen-round reproducibility.
        "v34_unified_history_route_guard",
        # V31-compatible selector that grounds identity-qualified destination
        # regions ("room with X") with localized DINO proposals before VLM
        # floor ranking.
        "v35_qualified_region_detector_grounding",
    }
    INSTRUCTION_COMPLETION_PROMPT_VERSIONS = {
        "v1_edge_evidence", "v2_transition_gates",
        "v3_temporal_completion_event", "v4_partial_extent_and_turns",
        "v5_source_destination_dominance",
        "v6_operational_partial_extent",
        "v7_structured_partial_extent",
        "v8_eight_view_spatial_relations",
        "v9_three_way_edge_progress",
        "v10_bidirectional_three_way",
        "v12_binary_completion",
        "v13_structured_node_edge_binary",
        "v14_structured_with_unordered_reverse_veto",
        "v15_structured_with_swap_consistent_reverse_veto",
        "v16_temporal_clause_aggregation_swap_veto",
        "v17_primary_consensus_swap_veto",
        "v18_vertical_endpoint_guard",
        "v19_vertical_guard_consensus",
        # Stage-1 endpoint recovery: keeps the structured RGB/action-history
        # contract while allowing generic endpoint-inferred completion when a
        # VLM is conservatively unable to mark the boundary explicitly.
        "v20_stage_endpoint_recovery",
        "v21_stage_endpoint_recovery_structured",
        # Stage-2 completion revision: recover a generic ENTER_REGION crossing
        # from a source-side portal plus destination-token front/rear coverage
        # when the VLM is conservative about identity/order.
        "v22_stage2_enter_transition",
        # Cumulative V21 endpoint recovery with detector-backed terminal
        # identity and multi-view under/beneath relation guards.
        "v23_relation_geometry_guard",
        # Calibrates coordinated PASS evidence, directional ``towards`` turns,
        # and overlapping side views at a doorway threshold.
        "v24_multireference_threshold_calibration",
    }
    EIGHT_VIEW_COMPLETION_PROMPT_VERSIONS = {
        "v8_eight_view_spatial_relations",
        "v9_three_way_edge_progress",
        "v10_bidirectional_three_way",
        "v12_binary_completion",
        "v13_structured_node_edge_binary",
        "v14_structured_with_unordered_reverse_veto",
        "v15_structured_with_swap_consistent_reverse_veto",
        "v16_temporal_clause_aggregation_swap_veto",
        "v17_primary_consensus_swap_veto",
        "v18_vertical_endpoint_guard",
        "v19_vertical_guard_consensus",
        "v20_stage_endpoint_recovery",
        "v21_stage_endpoint_recovery_structured",
        "v22_stage2_enter_transition",
        "v23_relation_geometry_guard",
        "v24_multireference_threshold_calibration",
    }

    EIGHT_VIEW_FORM_RULES = {
        "PASS_LANDMARK": (
            "First identify the referenced landmark instance by appearance and "
            "surrounding context; another object of the same class does not count. "
            "If multiple similar landmarks make identity uncertain, return unknown. "
            "If that same landmark is now "
            "visible in REAR_LEFT, REAR, or REAR_RIGHT after forward progress, "
            "treat the pass boundary as satisfied: it need not disappear from "
            "every view. Side-only visibility means beside/partial progress. "
            "Rear evidence is insufficient only if identity is ambiguous, the "
            "agent reversed, or the landmark still clearly dominates FRONT."),
        "CIRCUMNAVIGATE": (
            "Require the obstacle to move from front to the instructed side and "
            "then into a rear sector while the route clears beyond it. Reject "
            "motion around the wrong side; side-only evidence is still partial."),
        "EXIT_REGION": (
            "Require a source-side -> threshold -> destination-side transition. "
            "At completion the source room or its exit frame should normally be "
            "behind (REAR_LEFT/REAR/REAR_RIGHT), while destination space occupies "
            "front/side sectors. A doorway merely visible ahead is not completion."),
        "ENTER_REGION": (
            "Require crossing into the named destination. The entry frame/source "
            "should move to a rear sector and the destination should surround the "
            "camera across front and side sectors. Looking into the room is partial."),
        "SELECT_PORTAL": (
            "First verify portal identity/order/side, then require the selected "
            "portal frame to move behind and the intended beyond-portal floor to "
            "occupy front sectors. Crossing a neighboring portal is unknown."),
        "TRAVERSE_PORTAL_REGION": (
            "Require the referenced portal or bounded intermediate region to be "
            "cleared. Its near boundary should transition front -> side -> rear, "
            "with beyond-region space in front. Seeing the portal ahead is partial."),
        "CROSS_SPACE": (
            "Require motion from the near side to the far boundary. The near-side "
            "boundary/entry context should be behind and the far-side context should "
            "surround the current node; merely moving inside the open region is partial."),
        "BETWEEN_OBJECTS": (
            "Require the named pair to bracket opposite side sectors while entering "
            "the gap; if the instruction says clear/pass between them, the pair must "
            "then shift into side-rear or rear sectors. A bar/counter merely visible "
            "along one side while its long edge still extends ahead means BESIDE the "
            "object, not yet inside the named gap. Require chronological motion into "
            "the central lane with both inner faces genuinely on opposing sides. One "
            "object alone or broad panorama-wide detector aliases are ambiguous."),
        "TURN_LEFT": (
            "Require a leftward heading change: content previously in LEFT should "
            "correspond to current FRONT, supported by left-turn actions. Translation "
            "without this panorama correspondence is not completion."),
        "TURN_RIGHT": (
            "Require a rightward heading change: content previously in RIGHT should "
            "correspond to current FRONT, supported by right-turn actions. Translation "
            "without this panorama correspondence is not completion."),
        "TURN_AROUND": (
            "Require approximately reversed orientation: previous REAR content should "
            "become current FRONT and previous FRONT should become current REAR, with "
            "turn actions supporting the reversal. If the original reference is "
            "not confidently re-identified after entering a different corridor, a "
            "stored endpoint heading reversal of at least about 150 degrees plus "
            "real translated motion and chronological turn actions is sufficient; "
            "do not reject solely because a noisy detector fails to match the old "
            "door/object instance."),
        "ADVANCE_STRAIGHT": (
            "Require forward scene progression with limited net heading change: old "
            "front content expands and moves toward side/rear while a farther part of "
            "the same route becomes FRONT. A large branch turn contradicts completion."),
        "FOLLOW_PATH_BOUNDARY": (
            "Require forward progress while the referenced wall/path boundary remains "
            "on the instructed side across keyframes. Completion is the requested next "
            "decision area/end cue, not merely one frame with the boundary visible."),
        "APPROACH_LANDMARK": (
            "Require the landmark to become nearer/larger in FRONT, FRONT_LEFT, or "
            "FRONT_RIGHT at a safe offset. Rear-only visibility normally means it was "
            "overshot and contradicts an approach/near relation."),
        "TURN_TO_LANDMARK": (
            "Require an explicit heading change toward the named landmark. The "
            "landmark may be lateral or behind the arrival heading; use its raw "
            "RGB identity and bearing rather than an ordinary forward prior. "
            "Completion is a stable landmark-aligned heading with a connected "
            "floor ray, not merely a salient object in an unrelated sector."),
        "STOP_WAIT": (
            "Require the stated final spatial relation at the current node. For a "
            "near/beside stop, front is preferred but is not mandatory: a stable "
            "same-instance landmark at close/medium range in a side or rear sector "
            "can satisfy the relation when the edge ends adjacent to it and the "
            "chronological motion approaches/stops there. Treat rear-only as a "
            "contradiction only when the instruction explicitly requires an "
            "approach/front relation or the edge visibly passes the landmark. "
            "Use RGB scale, temporal stability, and motion history together."),
        "VERTICAL_UP": (
            "Use vertical delta and chronological stair traversal. For a full ascent, "
            "lower stairs/entry shift to rear sectors and the upper landing surrounds "
            "front/side. For 'halfway', require stairs both below/rear and above/front."),
        "VERTICAL_DOWN": (
            "Use negative vertical delta and chronological stair traversal. For a full "
            "descent, upper stairs/entry shift to rear sectors and the lower landing "
            "surrounds front/side. For 'halfway', require stairs above/rear and below/front."),
        "OTHER": (
            "Infer the explicit spatial relation from the clause, then require a "
            "temporal before/after change. Use rear sectors only when the relation "
            "semantically means passed, crossed, cleared, exited, or left behind."),
    }

    def __init__(self, backend, log_path=None, retries=2,
                 point_selection_prompt_version="v3_orientation_soft_semantic",
                 instruction_completion_prompt_version="v1_edge_evidence"):
        self.backend = backend
        self.log_path = Path(log_path) if log_path is not None else None
        self.retries = retries
        if point_selection_prompt_version not in self.POINT_SELECTION_PROMPT_VERSIONS:
            raise ValueError(
                "unknown point-selection prompt version "
                f"{point_selection_prompt_version!r}; expected one of "
                f"{sorted(self.POINT_SELECTION_PROMPT_VERSIONS)}")
        self.requested_point_selection_prompt_version = (
            point_selection_prompt_version)
        # V31 is the stable semantic selector used by the strict-start
        # baseline, but its v17 compatibility mapping previously bypassed the
        # final pixel-ray incoming/blocked guard introduced in V18/V19.  Keep
        # the prompt behavior while activating the geometry contract on the
        # requested version.  This is local graph history only, never GT.
        self._history_safe_refinement = point_selection_prompt_version in {
            "v18_history_safe_refinement",
            "v19_pixel_ray_history_and_reverse_override",
            "v31_circumnavigate_forward_competitor",
            "v35_qualified_region_detector_grounding",
        }
        self._first_step_route_guard = point_selection_prompt_version in {
            "v20_first_step_route_guard", "v32_first_stage_shallow_route",
            "v34_unified_history_route_guard"}
        self._relation_route_guard = point_selection_prompt_version in {
            "v21_relation_aware_route_review",
            "v22_task30_route_anchor",
            "v23_landmark_turn_stair_ray",
            "v24_compound_turn_ray_center",
            "v25_stage2_route_continuity",
            "v26_stage2_route_consensus",
            "v27_stage2_final_ray_geometry",
            "v28_stage2_side_aware_stop_ray",
            "v29_stage3_stop_relation_ray",
            "v30_stage3_relation_portal_ray",
            "v31_circumnavigate_forward_competitor",
            "v32_first_stage_shallow_route",
            "v33_stop_relation_near_side",
            "v34_unified_history_route_guard",
            "v35_qualified_region_detector_grounding",
        }
        self._task30_route_anchor = (
            point_selection_prompt_version in {
                "v22_task30_route_anchor",
                "v23_landmark_turn_stair_ray",
                "v24_compound_turn_ray_center",
                "v25_stage2_route_continuity",
                "v26_stage2_route_consensus",
                "v27_stage2_final_ray_geometry",
                "v28_stage2_side_aware_stop_ray",
                "v29_stage3_stop_relation_ray",
                "v30_stage3_relation_portal_ray",
                "v31_circumnavigate_forward_competitor",
                "v32_first_stage_shallow_route",
                "v33_stop_relation_near_side",
                "v34_unified_history_route_guard",
                "v35_qualified_region_detector_grounding",
            })
        self._stage1_turn_soft = (
            point_selection_prompt_version == "v24_compound_turn_ray_center")
        self._qualified_region_detector_grounding = (
            point_selection_prompt_version in {
                "v31_circumnavigate_forward_competitor",
                "v35_qualified_region_detector_grounding",
            })
        self._stage2_route_guard = point_selection_prompt_version in {
            "v25_stage2_route_continuity", "v26_stage2_route_consensus",
            "v27_stage2_final_ray_geometry", "v28_stage2_side_aware_stop_ray",
            "v29_stage3_stop_relation_ray", "v30_stage3_relation_portal_ray",
            # V31 is the active strict-start selector. Its geometry and
            # history guards were enabled, but the independent later-stage
            # RGB route comparison was accidentally omitted. That lets one
            # confident yet factually wrong portal description override all
            # competing legal views. Enable the same form-generic review;
            # it sees RGB/ground overlays only, never depth or GT geometry.
            "v31_circumnavigate_forward_competitor",
            # V32 is also used on later hops of the first instruction.  The
            # same independent route-continuity check is needed there; the
            # first-hop shallow rule is disabled once action history exists.
            "v32_first_stage_shallow_route", "v33_stop_relation_near_side",
            "v34_unified_history_route_guard"}
        self._stage2_side_stop_v28 = (
            point_selection_prompt_version == "v28_stage2_side_aware_stop_ray")
        self._stage3_stop_relation_v29 = (
            point_selection_prompt_version in {
                "v29_stage3_stop_relation_ray",
                "v33_stop_relation_near_side",
                "v34_unified_history_route_guard",
            })
        self._stage3_relation_portal_v30 = (
            point_selection_prompt_version in {
                "v30_stage3_relation_portal_ray",
                # V33 keeps the V30 portal evidence/semantic-ray policy while
                # adding near-side STOP anchor handling.
                "v33_stop_relation_near_side",
                "v34_unified_history_route_guard",
            })
        self._stage2_geometry_v27 = (
            point_selection_prompt_version in {
                "v27_stage2_final_ray_geometry",
                "v28_stage2_side_aware_stop_ray",
                "v29_stage3_stop_relation_ray",
                "v30_stage3_relation_portal_ray",
                "v31_circumnavigate_forward_competitor",
                "v35_qualified_region_detector_grounding",
                "v32_first_stage_shallow_route",
                "v33_stop_relation_near_side",
                "v34_unified_history_route_guard",
            })
        # V20 intentionally reuses the already-tested V17 eight-view RGB
        # review/anchor machinery.  Its new behavior is injected through a
        # first-stage-only prompt guard below, while the requested version is
        # preserved for reproducibility in call/selection records.
        self.point_selection_prompt_version = (
            "v17_turn_three_view_commit_prior"
            if (self._first_step_route_guard or self._relation_route_guard)
            else point_selection_prompt_version)
        if (instruction_completion_prompt_version not in
                self.INSTRUCTION_COMPLETION_PROMPT_VERSIONS):
            raise ValueError(
                "unknown instruction-completion prompt version "
                f"{instruction_completion_prompt_version!r}; expected one of "
                f"{sorted(self.INSTRUCTION_COMPLETION_PROMPT_VERSIONS)}")
        self.instruction_completion_prompt_version = str(
            instruction_completion_prompt_version)
        self.point_selection_candidate_policy = (
            "soft_detection_evidence"
            if point_selection_prompt_version in {
                "v3_orientation_soft_semantic",
                "v4_sector_identity_gates",
                "v5_two_stage_sector_anchor",
                "v6_validated_relation_two_stage",
                "v7_single_stage_relation_gates",
                "v8_object_relation_router",
                "v9_single_stage_identity_relation_gates",
                "v10_approach_relation_router",
                "v11_eight_view_refinement",
                "v12_eight_view_dual_candidate",
                "v13_eight_view_native_rgb",
                "v14_rgb_evidence_refinement",
                "v15_rgb_center_preferred",
                "v16_turn_three_view_gate",
                "v17_turn_three_view_commit_prior",
                "v18_history_safe_refinement",
                "v19_pixel_ray_history_and_reverse_override",
                # V20 deliberately keeps valid floor-bearing sectors when a
                # detector hit has no relation-mask support (e.g. a clipped
                # landmark).  The RGB route guard adjudicates among them;
                # hard-gating them would leave only false-positive wall masks.
                "v20_first_step_route_guard",
                "v21_relation_aware_route_review",
                "v22_task30_route_anchor",
                "v23_landmark_turn_stair_ray",
                "v24_compound_turn_ray_center",
                "v25_stage2_route_continuity",
                "v26_stage2_route_consensus",
                "v27_stage2_final_ray_geometry",
                "v28_stage2_side_aware_stop_ray",
                "v29_stage3_stop_relation_ray",
                "v30_stage3_relation_portal_ray",
                "v31_circumnavigate_forward_competitor",
                "v35_qualified_region_detector_grounding",
                "v32_first_stage_shallow_route",
                "v33_stop_relation_near_side",
                "v34_unified_history_route_guard",
            } else
            "hard_detection_gate")
        self.calls = []
        self.attempts = []

    def _save_attempt_inputs(self, task, prompt, images, schema, attempt):
        """Persist the exact VLM request inputs for reproducible real-state tests."""
        if self.log_path is None:
            return []
        artifact_dir = self.log_path.parent / "vlm_artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{len(self.attempts):04d}_{task}_attempt_{attempt}"
        prompt_path = artifact_dir / f"{prefix}_prompt.txt"
        schema_path = artifact_dir / f"{prefix}_schema.json"
        prompt_path.write_text(str(prompt) + "\n")
        schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n")
        image_paths = []
        for index, value in enumerate(images):
            image = np.asarray(value)
            path = artifact_dir / f"{prefix}_image_{index:02d}.jpg"
            Image.fromarray(image.astype(np.uint8)).save(path, quality=95)
            image_paths.append(str(path.relative_to(self.log_path.parent)))
        return image_paths

    def _call(self, task, prompt, images, schema, validator):
        errors = []
        for attempt in range(self.retries + 1):
            attempt_prompt = str(prompt)
            image_paths = self._save_attempt_inputs(
                task, attempt_prompt, images, schema, attempt)
            try:
                result = self.backend.generate_json(attempt_prompt, images, schema)
                validated = validator(result)
                call = {
                    "task": task, "attempt": attempt, "result": result,
                    "prompt": attempt_prompt, "image_paths": image_paths,
                    "schema": schema,
                }
                if getattr(self.backend, "last_usage", None) is not None:
                    call["usage"] = self.backend.last_usage
                if getattr(self.backend, "last_finish_reason", None) is not None:
                    call["finish_reason"] = self.backend.last_finish_reason
                if getattr(self.backend, "last_call_meta", None) is not None:
                    call["call_meta"] = dict(self.backend.last_call_meta)
                self.attempts.append({**call, "status": "success"})
                self.calls.append(call)
                self.flush()
                return validated
            except VLMProviderFatalError as exc:
                self.attempts.append({
                    "task": task, "attempt": attempt,
                    "status": "provider_fatal", "error": str(exc),
                    "prompt": attempt_prompt, "image_paths": image_paths,
                    "schema": schema, **self._backend_call_meta(),
                })
                self.flush()
                raise
            except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                errors.append(str(exc))
                self.attempts.append({
                    "task": task, "attempt": attempt, "status": "error",
                    "error": str(exc), "prompt": attempt_prompt,
                    "image_paths": image_paths, "schema": schema,
                    **self._backend_call_meta(),
                })
                self.flush()
                prompt += f"\nPrevious response was invalid: {exc}. Return corrected JSON only."
        raise RuntimeError(f"VLM harness exhausted retries for {task}: {errors}")

    def _backend_call_meta(self):
        meta = getattr(self.backend, "last_call_meta", None)
        return {"call_meta": dict(meta)} if meta else {}

    def flush(self):
        if self.log_path:
            self.log_path.write_text(json.dumps(self.calls, indent=2) + "\n")
            attempts_path = self.log_path.with_name(
                f"{self.log_path.stem}_attempts.json")
            attempts_path.write_text(
                json.dumps(self.attempts, indent=2) + "\n")

    def decompose_instruction(self, instruction):
        prompt = f"""STAGE_DECOMPOSITION
You are the instruction planner for an R2R indoor navigation agent. Split the
instruction into ordered, visually grounded stages. Every stage MUST terminate
at a semantic 3D spatial region that can be represented by a walkable floor
pixel, not merely at an action such as "turn" or "go out".

Grounding rules:
- "go/exit out of a room" targets free walkable floor just BEYOND the current
  room's doorway, far enough that the camera has crossed the door frame.
- "enter room/hall" targets free floor just INSIDE that named space.
- "pass object" targets floor beyond the object along the instructed side/path.
- "turn left/right" targets visible walkable floor along the new corridor or
  opening after the turn; never target a wall merely to cause rotation.
- "stop near X" targets free floor near X at a safe offset, not pixels on X.
- "walk straight" targets distant visible floor along the corridor/open space.

For every stage specify the reference landmark, precise semantic spatial target,
its spatial relation, visual evidence that the camera has arrived, and forbidden
regions. Preserve turns, motion, landmarks and stopping conditions. Do not
invent objects not implied by the instruction or generic targets like
"somewhere ahead".
R2R instruction: {instruction}
Return only the requested JSON."""

        def validate(result):
            stages = result["stages"]
            if not isinstance(stages, list) or not stages:
                raise ValueError("stages must be a non-empty list")
            cleaned = []
            for index, stage in enumerate(stages):
                text = str(stage["navigation_instruction"]).strip()
                if not text:
                    raise ValueError("empty navigation_instruction")
                cleaned.append({
                    "stage_id": index, "navigation_instruction": text,
                    "landmark": str(stage.get("landmark", "")).strip(),
                    "completion_cue": str(stage.get("completion_cue", "")).strip(),
                    "semantic_spatial_target": str(stage["semantic_spatial_target"]).strip(),
                    "spatial_relation": str(stage["spatial_relation"]).strip(),
                    "visual_arrival_evidence": str(stage["visual_arrival_evidence"]).strip(),
                    "forbidden_target": str(stage["forbidden_target"]).strip(),
                })
                if not all(cleaned[-1][key] for key in (
                        "semantic_spatial_target", "spatial_relation",
                        "visual_arrival_evidence", "forbidden_target")):
                    raise ValueError("each stage needs a non-empty semantic spatial grounding")
            return cleaned

        return self._call("decompose_instruction", prompt, [], self.STAGE_SCHEMA, validate)

    @staticmethod
    def _overlay(rgb, mask, index, excluded, detections=None, object_mask=None,
                 ground_anchors=None):
        image = rgb.copy()
        if object_mask is not None:
            blue = np.zeros_like(image); blue[..., 2] = 255
            image[object_mask] = (0.82 * image[object_mask] +
                                  0.18 * blue[object_mask]).astype(np.uint8)
        green = np.zeros_like(image); green[..., 1] = 255
        image[mask] = (0.55 * image[mask] + 0.45 * green[mask]).astype(np.uint8)
        colors = [(255, 80, 40), (255, 200, 30), (180, 80, 255), (40, 210, 255)]
        for det_index, detection in enumerate(detections or []):
            color = colors[det_index % len(colors)]
            det_mask = detection.mask
            image[det_mask] = (0.72 * image[det_mask] +
                               0.28 * np.asarray(color)).astype(np.uint8)
            x0, y0, x1, y1 = np.round(detection.box_xyxy).astype(int)
            cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
            cv2.putText(image, f"{detection.label} {detection.score:.2f}",
                        (max(2, x0), max(38, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, color, 1)
        for anchor_index, point in enumerate(ground_anchors or []):
            xy = tuple(np.round(point).astype(int))
            cv2.circle(image, xy, 7, (255, 255, 255), -1)
            cv2.circle(image, xy, 7, (20, 20, 20), 1)
            cv2.putText(image, str(anchor_index), (xy[0] - 4, xy[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
        cv2.putText(image, f"VIEW {index}" + (" EXCLUDED" if excluded else ""), (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 50, 50), 2)
        return image

    @staticmethod
    def _contact_sheet(images):
        if len(images) != 6:
            raise ValueError("ground selection harness expects exactly six views")
        return np.concatenate([
            np.concatenate(images[:3], axis=1),
            np.concatenate(images[3:], axis=1),
        ], axis=0)

    @staticmethod
    def _point_selection_contact_sheet(images):
        """Render six legacy or eight-plus-refined point-selection views."""
        if len(images) == 6:
            columns = 3
        elif len(images) in {8, 9}:
            columns = 4
        else:
            raise ValueError(
                "point selection expects six, eight, or eight plus one refined view")
        rows = int(math.ceil(len(images) / columns))
        height, width = images[0].shape[:2]
        padded = [np.asarray(image, np.uint8) for image in images]
        padded.extend(
            np.zeros((height, width, 3), np.uint8)
            for _ in range(rows * columns - len(padded)))
        return np.concatenate([
            np.concatenate(
                padded[row * columns:(row + 1) * columns], axis=1)
            for row in range(rows)
        ], axis=0)

    @staticmethod
    def _completion_contact_sheet(images):
        """Render eight explicitly labeled compass sectors as a 4x2 sheet."""
        if len(images) != 8:
            raise ValueError(
                "eight-view completion harness expects exactly eight views")
        labels = (
            "VIEW 0 FRONT 0deg", "VIEW 1 FRONT_LEFT +45deg",
            "VIEW 2 LEFT +90deg", "VIEW 3 REAR_LEFT +135deg",
            "VIEW 4 REAR 180deg", "VIEW 5 REAR_RIGHT -135deg",
            "VIEW 6 RIGHT -90deg", "VIEW 7 FRONT_RIGHT -45deg",
        )
        rendered = []
        for image, label in zip(images, labels):
            value = np.asarray(image, np.uint8)[..., :3].copy()
            cv2.rectangle(value, (0, 0), (value.shape[1], 27), (0, 0, 0), -1)
            cv2.putText(value, label, (5, 19), cv2.FONT_HERSHEY_SIMPLEX,
                        0.43, (255, 255, 255), 1, cv2.LINE_AA)
            rendered.append(value)
        return np.concatenate([
            np.concatenate(rendered[:4], axis=1),
            np.concatenate(rendered[4:], axis=1),
        ], axis=0)

    @staticmethod
    def _keyframe_strip(images):
        if not images:
            raise ValueError("instruction completion requires edge keyframes")
        height = 160
        rendered = []
        for index, value in enumerate(images):
            image = np.asarray(value, np.uint8)[..., :3]
            width = max(1, round(image.shape[1] * height / image.shape[0]))
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            cv2.putText(image, f"KF {index}", (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
            rendered.append(image)
        return np.concatenate(rendered, axis=1)

    @staticmethod
    def _keyframe_storyboard(images):
        """Render temporal keyframes large enough for landmark transitions."""
        if not images:
            raise ValueError("instruction completion requires edge keyframes")
        height, width = 240, 320
        rendered = []
        for index, value in enumerate(images):
            image = cv2.resize(
                np.asarray(value, np.uint8)[..., :3], (width, height),
                interpolation=cv2.INTER_AREA)
            cv2.rectangle(image, (0, 0), (width, 30), (0, 0, 0), -1)
            cv2.putText(image, f"KF {index} (chronological)", (7, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 1,
                        cv2.LINE_AA)
            rendered.append(image)
        columns = min(3, len(rendered))
        while len(rendered) % columns:
            rendered.append(np.zeros((height, width, 3), np.uint8))
        rows = [np.concatenate(rendered[index:index + columns], axis=1)
                for index in range(0, len(rendered), columns)]
        return np.concatenate(rows, axis=0)

    @staticmethod
    def _nearest_ground(mask, requested):
        ys, xs = np.nonzero(mask)
        if not len(xs):
            raise ValueError("selected view has no floor pixels")
        distances = (xs - requested[0]) ** 2 + (ys - requested[1]) ** 2
        index = int(np.argmin(distances))
        return np.array([float(xs[index]), float(ys[index])], np.float32)

    @staticmethod
    def _ground_anchors(mask, preferred=None, count=6):
        """Diverse, well-interior discrete anchors guaranteed to lie on mask."""
        valid = mask.astype(np.uint8)
        ys, xs = np.nonzero(valid)
        if not len(xs):
            return []
        distance = cv2.distanceTransform(valid, cv2.DIST_L2, 5)
        coords = np.stack([xs, ys], axis=1).astype(np.float32)
        if preferred is not None:
            first = int(np.argmin(((coords - preferred) ** 2).sum(1)))
        else:
            first = int(np.argmax(distance[ys, xs]))
        chosen = [coords[first]]
        while len(chosen) < min(count, len(coords)):
            separation = np.min(np.stack([
                ((coords - point) ** 2).sum(1) for point in chosen]), axis=0)
            interior = distance[ys, xs] ** 2
            score = separation * (0.25 + interior)
            index = int(np.argmax(score))
            if any(np.array_equal(coords[index], point) for point in chosen):
                break
            chosen.append(coords[index])
        return chosen

    @classmethod
    def _task30_ground_anchors(cls, mask, preferred=None, count=6):
        """Sample route-bearing floor anchors across the usable RGB mask.

        The ordinary sampler intentionally concentrates on the deepest central
        component.  That is a good safety prior, but it can hide a connected
        exit that enters from one side of a wide view.  V22 keeps the central
        preference while exposing a small set of horizontal quantiles so the
        RGB anchor pass can select the visible outgoing gap rather than an
        arbitrary mask centroid.  This uses only the 2-D floor mask.
        """
        valid = np.asarray(mask, dtype=np.uint8)
        ys, xs = np.nonzero(valid)
        if not len(xs):
            return []
        height, width = valid.shape[:2]
        distance = cv2.distanceTransform(valid, cv2.DIST_L2, 5)
        coords = np.stack([xs, ys], axis=1).astype(np.float32)
        # Keep points on a plausible support band and away from one-pixel mask
        # fringes, but fall back to all pixels for thin hallway masks.
        support = (ys >= int(0.52 * height)) & (distance[ys, xs] >= 1.0)
        if int(np.count_nonzero(support)) < max(8, count):
            support = distance[ys, xs] > 0.0
        sx, sy = xs[support], ys[support]
        sd = distance[sy, sx]
        if preferred is not None:
            preferred = np.asarray(preferred, np.float32)
            support_coords = np.stack([sx, sy], axis=1).astype(np.float32)
            first = int(np.argmin(((support_coords - preferred) ** 2).sum(1)))
            chosen = [support_coords[first]]
        else:
            first = int(np.argmax(sd))
            chosen = [np.array([float(sx[first]), float(sy[first])],
                                np.float32)]
        # Quantiles include both sides when the mask has a side entrance, while
        # retaining central anchors for the usual forward corridor case.
        for quantile in (0.15, 0.35, 0.55, 0.75, 0.90):
            if len(chosen) >= count:
                break
            target_x = float(np.quantile(sx, quantile))
            # Prefer a well-interior pixel near the requested horizontal ray.
            score = ((sx.astype(np.float32) - target_x) ** 2 /
                     max(width * width, 1.0) -
                     0.08 * (sd / max(float(sd.max()), 1.0)))
            order = np.argsort(score)
            for index in order:
                point = np.array([float(sx[index]), float(sy[index])],
                                 np.float32)
                if not any(np.linalg.norm(point - old) < 2.0
                           for old in chosen):
                    chosen.append(point)
                    break
        return chosen[:count]

    @staticmethod
    def _vertical_ground_anchors(mask, direction="up", count=6):
        """Sample ordered anchors along a visible stair/landing route.

        A generic stair target is not satisfied by an arbitrary point on the
        nearest tread.  When the landing is not segmented, the only honest
        RGB-only fallback is the furthest visible connected tread in the
        requested vertical direction.  We therefore expose anchors ordered by
        image-space progression (top-to-bottom for an ascent, bottom-to-top
        for a descent), while retaining a small centrality/interior preference
        to avoid railing and mask fringes.  This uses no depth or navmesh.
        """
        valid = np.asarray(mask, dtype=np.uint8)
        ys, xs = np.nonzero(valid)
        if not len(xs):
            return []
        height, width = valid.shape[:2]
        distance = cv2.distanceTransform(valid, cv2.DIST_L2, 5)
        order = np.argsort(ys if str(direction).lower() == "up" else -ys)
        # Keep samples spread over the connected visible route rather than
        # returning six almost-identical pixels from one tread.
        ordered_y = ys[order].astype(np.float32)
        quantiles = np.linspace(0.05, 0.95, max(2, int(count)))
        if str(direction).lower() != "up":
            quantiles = quantiles[::-1]
        chosen = []
        for quantile in quantiles:
            target_y = float(np.quantile(ordered_y, quantile))
            band = np.abs(ys.astype(np.float32) - target_y) <= max(3.0, 0.04 * height)
            indices = np.flatnonzero(band)
            if not len(indices):
                indices = np.arange(len(xs))
            # Prefer central, well-interior pixels at the requested route
            # depth.  The y term is dominant, so the first anchor is always
            # the furthest visible endpoint candidate.
            centrality = 1.0 - np.abs(xs[indices].astype(np.float32) -
                                     (width - 1) / 2.0) / max(width / 2.0, 1.0)
            score = (0.70 * distance[ys[indices], xs[indices]] +
                     0.30 * centrality)
            index = int(indices[np.argmax(score)])
            point = np.array([float(xs[index]), float(ys[index])], np.float32)
            if not any(np.linalg.norm(point - old) < 3.0 for old in chosen):
                chosen.append(point)
        if not chosen:
            chosen = [np.array([float(xs[order[0]]), float(ys[order[0]])],
                               np.float32)]
        return chosen[:count]

    def select_ground_target(
            self, stage, candidates, action_history=None,
            refinement_provider=None, semantic_reference_rgb=None):
        eight_view_versions = {
            "v11_eight_view_refinement",
            "v12_eight_view_dual_candidate",
            "v13_eight_view_native_rgb",
            "v14_rgb_evidence_refinement",
            "v15_rgb_center_preferred",
            "v16_turn_three_view_gate",
            "v17_turn_three_view_commit_prior",
            "v18_history_safe_refinement",
            "v19_pixel_ray_history_and_reverse_override",
        }
        if any("depth" in candidate for candidate in candidates):
            raise ValueError(
                "VLM point-selection candidates must not contain depth; retain "
                "it only behind the post-selection hidden projection boundary")
        # Scrub numeric depth fields from detector records for every prompt
        # version, including custom callers that did not use rgb_prompt_record.
        for candidate in candidates:
            for field in ("detection_records", "next_context_detection_records",
                          "ground_detection_records"):
                candidate[field] = [
                    {key: value for key, value in dict(record).items()
                     if "depth" not in str(key).lower()}
                    for record in candidate.get(field, [])
                ]
        use_rgb_evidence_refinement = (
            self.point_selection_prompt_version in {
                "v14_rgb_evidence_refinement",
                "v15_rgb_center_preferred",
                "v16_turn_three_view_gate",
                "v17_turn_three_view_commit_prior",
                "v18_history_safe_refinement",
                "v19_pixel_ray_history_and_reverse_override",
                "v31_circumnavigate_forward_competitor",
            })
        use_center_preferred_fallback = (
            self.point_selection_prompt_version in {
                "v15_rgb_center_preferred",
                "v16_turn_three_view_gate",
                "v17_turn_three_view_commit_prior",
                "v18_history_safe_refinement",
                "v19_pixel_ray_history_and_reverse_override",
            })
        use_turn_three_view_gate = (
            self.point_selection_prompt_version in {
                "v16_turn_three_view_gate",
                "v17_turn_three_view_commit_prior",
                "v18_history_safe_refinement",
                "v19_pixel_ray_history_and_reverse_override",
            })
        use_turn_commit_prior = (
            self.point_selection_prompt_version in {
                "v17_turn_three_view_commit_prior",
                "v18_history_safe_refinement",
                "v19_pixel_ray_history_and_reverse_override",
                "v31_circumnavigate_forward_competitor",
            })
        first_step_route_guard_active = bool(
            self._first_step_route_guard and not (action_history or []))
        instruction_lower = str(stage.get(
            "navigation_instruction", "")).lower()
        bare_turn_setup = ((stage.get("metadata", {}) or {}).get(
            "bare_turn_route_setup", {}) or {})
        orientation_stage = (bare_turn_setup.get("orientation_stage", {})
                             if bare_turn_setup.get("active") else {}) or {}
        direction_form = str(orientation_stage.get(
            "form", stage.get("form", "")))
        direction_instruction_lower = str(orientation_stage.get(
            "navigation_instruction", instruction_lower)).lower()
        explicit_reverse_command = bool(
            direction_form == "TURN_AROUND" or
            re.search(
                r"\bturn\s+(?:all\s+the\s+way\s+)?around\b",
                direction_instruction_lower))
        explicit_turn_command = bool(
            direction_form in {
                "TURN_LEFT", "TURN_RIGHT", "TURN_AROUND"} or
            re.fullmatch(
                r"\s*(?:then\s+)?(?:make\s+)?(?:a\s+)?turn\s+"
                r"(?:to\s+the\s+)?(?:left|right|around)"
                r"(?:\s+\d+\s*degrees?)?\s*[.!]?\s*",
                direction_instruction_lower))
        stage_metadata = stage.get("metadata", {}) or {}
        route_corridor = (
            (stage_metadata.get("instruction_committed_route_corridor", {})
             or {}) or
            (stage_metadata.get("recovery_route_corridor", {}) or {}))
        route_corridor_active = bool(route_corridor.get("active"))
        route_corridor_absolute_yaw = (
            float(route_corridor["absolute_yaw_rad"])
            if route_corridor_active else None)
        route_corridor_half_width = math.radians(float(
            route_corridor.get("half_width_deg", 45.0)))
        route_continuation = bool((stage.get("metadata", {}) or {}).get(
            "supported_partial_route_continuation", {}).get("active"))

        def route_ray_allowed(candidate, pixel_x):
            if not route_corridor_active:
                return True
            width = candidate["target_mask"].shape[1]
            center_x = (width - 1.0) / 2.0
            half_width = max(width / 2.0, 1.0)
            pixel_bearing = math.atan(
                (float(pixel_x) - center_x) / half_width)
            absolute_ray = float(candidate.get("yaw", 0.0)) - pixel_bearing
            bend_fallback = candidate.get(
                "route_corridor_bend_fallback", {}) or {}
            allowed_half_width = route_corridor_half_width
            if bend_fallback.get("active"):
                allowed_half_width = math.radians(float(
                    bend_fallback.get("maximum_delta_deg", 90.0)))
            return abs((absolute_ray - route_corridor_absolute_yaw +
                        math.pi) % (2 * math.pi) - math.pi) <= (
                            allowed_half_width)

        def history_safe_anchors(candidate, anchors):
            """Remove final RGB rays that cross an incoming/blocked boundary."""
            if not self._history_safe_refinement:
                return list(anchors)
            height, width = candidate["target_mask"].shape
            del height
            center_x = (width - 1.0) / 2.0
            half_width = max(width / 2.0, 1.0)
            candidate_yaw = float(candidate.get("relative_yaw_rad", 0.0))
            back_yaw = candidate.get("incoming_back_relative_yaw_rad")
            back_exclusion = float(candidate.get(
                "backtrack_exclusion_rad", math.radians(50.0)))
            if candidate.get("soft_incoming_tangent_reopened"):
                # The ordinary incoming cone is deliberately conservative.
                # When every other strict-ground view is empty, retain only
                # rays outside its narrow reverse core.  Sequence-blocked
                # directions remain unchanged and hard.
                back_exclusion = min(back_exclusion, math.radians(25.0))
            blocked_yaws = list(candidate.get(
                "blocked_relative_yaws_rad", []))
            blocked_exclusion = float(candidate.get(
                "blocked_direction_exclusion_rad", math.radians(50.0)))
            if candidate.get("soft_blocked_tangent_reopened"):
                # A blocked branch centre remains forbidden, but a wide 50°
                # camera-centre cone must not erase a distinct strict-ground
                # ray at the edge of the same image.  This fallback is only
                # enabled after ordinary rays are exhausted below.
                blocked_exclusion = min(
                    blocked_exclusion, math.radians(20.0))
            safe = []
            for anchor in anchors:
                pixel_bearing = math.atan(
                    (float(anchor[0]) - center_x) / half_width)
                ray_yaw = (candidate_yaw - pixel_bearing + math.pi) % (
                    2 * math.pi) - math.pi
                enters_back = bool(
                    back_yaw is not None and not explicit_turn_command and
                    abs((ray_yaw - float(back_yaw) + math.pi) %
                        (2 * math.pi) - math.pi) < back_exclusion)
                enters_blocked = any(
                    abs((ray_yaw - float(blocked) + math.pi) %
                        (2 * math.pi) - math.pi) < blocked_exclusion
                    for blocked in blocked_yaws)
                if (not enters_back and not enters_blocked and
                        route_ray_allowed(candidate, anchor[0])):
                    safe.append(anchor)
            if not safe and np.asarray(candidate["target_mask"], bool).any():
                # The normal anchor sampler is intentionally compact and can
                # place all six proposals in an excluded image column even
                # though the same strict ground component contains a legal
                # side ray.  Re-sample the complete mask after intersecting it
                # with the exact same ray guard; never add pixels outside the
                # original target mask.
                columns_safe = np.zeros(width, dtype=bool)
                for x in range(width):
                    pixel_bearing = math.atan(
                        (float(x) - center_x) / half_width)
                    ray_yaw = (candidate_yaw - pixel_bearing + math.pi) % (
                        2 * math.pi) - math.pi
                    enters_back = bool(
                        back_yaw is not None and not explicit_turn_command and
                        abs((ray_yaw - float(back_yaw) + math.pi) %
                            (2 * math.pi) - math.pi) < back_exclusion)
                    enters_blocked = any(
                        abs((ray_yaw - float(blocked) + math.pi) %
                            (2 * math.pi) - math.pi) < blocked_exclusion
                        for blocked in blocked_yaws)
                    columns_safe[x] = bool(
                        not enters_back and not enters_blocked and
                        route_ray_allowed(candidate, x))
                safe_mask = (np.asarray(candidate["target_mask"], bool) &
                             columns_safe[None, :])
                if safe_mask.any():
                    safe = self._ground_anchors(
                        safe_mask, candidate.get("point"), count=12)
                    safe = [anchor for anchor in safe
                            if safe_mask[int(round(float(anchor[1]))),
                                         int(round(float(anchor[0])))]]
                    if safe:
                        candidate["history_safe_anchor_recovery"] = (
                            "full_strict_ground_mask_resample")
            return safe
        if use_rgb_evidence_refinement:
            def usable_rgb_detection(record, shape):
                label = str(record.get("label", "")).lower()
                words = set(re.findall(r"[a-z]+", label))
                if words & {
                        "walk", "go", "turn", "cross", "your", "toward",
                        "towards"}:
                    return False
                mask_fraction = float(record.get(
                    "mask_area_fraction", 0.0) or 0.0)
                if mask_fraction > 0.45:
                    return False
                box = record.get("box_xyxy")
                if box is not None and len(box) == 4:
                    height, width = shape[:2]
                    box_fraction = (
                        max(0.0, float(box[2]) - float(box[0])) *
                        max(0.0, float(box[3]) - float(box[1])) /
                        max(1.0, float(width * height)))
                    if box_fraction > 0.55:
                        return False
                    # A STOP/WAIT landmark whose proposal is clipped by the
                    # image boundary is not reliable instance evidence: the
                    # visible sliver can belong to a different rug/sofa or
                    # can point the anchor toward a side wall.  Keep the
                    # underlying RGB floor candidate, but remove the clipped
                    # detector from semantic ranking so all views can be
                    # compared.  This is a form-level image-quality rule,
                    # not an episode/scene or fixed-pixel patch.
                    if (str(stage.get("form", "")).upper() == "STOP_WAIT" and
                            (float(box[0]) <= 1.0 or
                             float(box[1]) <= 1.0 or
                             float(box[2]) >= float(width - 2) or
                             float(box[3]) >= float(height - 2))):
                        return False
                return bool(label)

            for candidate in candidates:
                records = candidate.get("detection_records", [])
                keep = [usable_rgb_detection(
                    record, candidate["rgb"].shape) for record in records]
                candidate["detection_records"] = [
                    record for record, accepted in zip(records, keep)
                    if accepted]
                detections = candidate.get("semantic_detections", [])
                if len(detections) == len(keep):
                    candidate["semantic_detections"] = [
                        detection for detection, accepted in
                        zip(detections, keep) if accepted]
        if (self.point_selection_prompt_version in eight_view_versions and
                len(candidates) != 8):
            raise ValueError(
                "v11 eight-view refinement requires exactly eight initial views")
        object_relation_two_stage_forms = {
            "APPROACH_LANDMARK", "BETWEEN_OBJECTS", "CIRCUMNAVIGATE",
        }
        use_two_stage = (
            self.point_selection_prompt_version in {
                "v5_two_stage_sector_anchor",
                "v6_validated_relation_two_stage",
            } or (
                self.point_selection_prompt_version ==
                "v8_object_relation_router" and
                str(stage.get("form", "")) in
                object_relation_two_stage_forms) or (
                self.point_selection_prompt_version ==
                "v10_approach_relation_router" and
                str(stage.get("form", "")) == "APPROACH_LANDMARK"))
        allowed = [i for i, candidate in enumerate(candidates)
                   if candidate["point"] is not None and not candidate["excluded"]]
        if (not allowed and self._history_safe_refinement and
                not explicit_turn_command):
            # A 50-degree camera-centre incoming exclusion can remove both
            # views bordering a narrow corridor even when their outer floor
            # rays are tangential rather than reversing.  Reopen only
            # non-hard-excluded views with strict ground outside a 25-degree
            # reverse core.  Blocked branch sectors are never reopened.
            tangent_allowed = []
            for index, candidate in enumerate(candidates):
                if (candidate.get("point") is None or
                        candidate.get("hard_excluded", False)):
                    continue
                back_yaw = candidate.get("incoming_back_relative_yaw_rad")
                if back_yaw is None:
                    continue
                mask = np.asarray(candidate["target_mask"], bool)
                if not mask.any():
                    continue
                width = mask.shape[1]
                xs = np.flatnonzero(mask.any(axis=0))
                center_x = (width - 1.0) / 2.0
                half_width = max(width / 2.0, 1.0)
                candidate_yaw = float(candidate.get("relative_yaw_rad", 0.0))
                has_tangent_ray = any(
                    abs((((candidate_yaw - math.atan(
                        (float(x) - center_x) / half_width)) -
                        float(back_yaw) + math.pi) % (2 * math.pi)) -
                        math.pi) >= math.radians(25.0)
                    for x in xs)
                if has_tangent_ray:
                    candidate["soft_incoming_tangent_reopened"] = True
                    tangent_allowed.append(index)
            allowed = tangent_allowed
        if (not allowed and self._history_safe_refinement and
                route_corridor_active):
            # Sequence recovery may accumulate several 45°-spaced failed
            # camera centres.  Their ordinary 50° cones can cover the whole
            # panorama even though a strict-ground edge ray is at least 20°
            # from every failed centre.  Reopen only those tangent pixels;
            # the exact failed ray and its 20° core remain hard-blocked.
            blocked_tangent_allowed = []
            for index, candidate in enumerate(candidates):
                if candidate.get("point") is None:
                    continue
                blocked_yaws = list(candidate.get(
                    "blocked_relative_yaws_rad", []))
                if not blocked_yaws or len(blocked_yaws) >= 5:
                    continue
                mask = np.asarray(candidate["target_mask"], bool)
                if not mask.any():
                    continue
                width = mask.shape[1]
                center_x = (width - 1.0) / 2.0
                half_width = max(width / 2.0, 1.0)
                candidate_yaw = float(candidate.get(
                    "relative_yaw_rad", 0.0))
                back_yaw = candidate.get(
                    "incoming_back_relative_yaw_rad")
                xs = np.flatnonzero(mask.any(axis=0))
                has_safe_tangent = False
                for x in xs:
                    pixel_bearing = math.atan(
                        (float(x) - center_x) / half_width)
                    ray_yaw = (candidate_yaw - pixel_bearing + math.pi) % (
                        2 * math.pi) - math.pi
                    outside_blocks = all(
                        abs((ray_yaw - float(blocked) + math.pi) %
                            (2 * math.pi) - math.pi) >= math.radians(20.0)
                        for blocked in blocked_yaws)
                    outside_incoming = bool(
                        back_yaw is None or explicit_turn_command or
                        abs((ray_yaw - float(back_yaw) + math.pi) %
                            (2 * math.pi) - math.pi) >= math.radians(25.0))
                    if (outside_blocks and outside_incoming and
                            route_ray_allowed(candidate, x)):
                        has_safe_tangent = True
                        break
                if has_safe_tangent:
                    candidate["soft_blocked_tangent_reopened"] = True
                    blocked_tangent_allowed.append(index)
            allowed = blocked_tangent_allowed
        if not allowed and not self._history_safe_refinement:
            # Incoming-direction exclusion is soft and may be relaxed. A
            # sequence-recovery blocked direction is hard and must never be
            # silently reintroduced.
            allowed = [i for i, candidate in enumerate(candidates)
                       if candidate["point"] is not None and
                       not candidate.get("hard_excluded", False)]
        if not allowed:
            raise RuntimeError("No floor-bearing candidate can be sent to the VLM")
        if route_corridor_active:
            corridor_allowed = []
            for index in allowed:
                mask = np.asarray(candidates[index]["target_mask"], bool)
                xs = np.flatnonzero(mask.any(axis=0))
                if any(route_ray_allowed(candidates[index], x) for x in xs):
                    corridor_allowed.append(index)
            if not corridor_allowed and refinement_provider is not None:
                # A correct corridor can fall exactly on the seam between two
                # 45-degree panorama cameras.  Dense-majority Grounded-SAM may
                # then retain floor only at the bottom/outer edge of each
                # image, leaving no legal ray although the route itself is
                # visible. Acquire up to three local RGB views centred on the
                # already-persisted online corridor (0, -15, +15 degrees).
                # This is not a new semantic direction decision and does not
                # expose depth/navmesh/GT to the VLM.
                reference_yaw = float(candidates[0].get("yaw", 0.0)) - float(
                    candidates[0].get("relative_yaw_rad", 0.0))
                corridor_relative = (
                    route_corridor_absolute_yaw - reference_yaw + math.pi
                ) % (2.0 * math.pi) - math.pi
                for offset_deg in (0.0, -15.0, 15.0):
                    requested_relative = (
                        corridor_relative + math.radians(offset_deg) + math.pi
                    ) % (2.0 * math.pi) - math.pi
                    refined = dict(refinement_provider(requested_relative))
                    if "depth" in refined:
                        raise ValueError(
                            "corridor refinement candidate must not contain depth")
                    refined["view_index"] = len(candidates)
                    refined["route_corridor_refinement"] = {
                        "active": True,
                        "corridor_center_relative_yaw_deg": round(
                            math.degrees(corridor_relative), 3),
                        "camera_offset_from_corridor_deg": float(offset_deg),
                        "policy": (
                            "local RGB acquisition around persisted supported-"
                            "partial route; no new semantic branch choice"),
                    }
                    mask = np.asarray(refined.get("target_mask"), bool)
                    xs = np.flatnonzero(mask.any(axis=0)) if mask.ndim == 2 else []
                    if str(stage.get("form", "")) in {
                            "VERTICAL_UP", "VERTICAL_DOWN"}:
                        refined_anchors = self._vertical_ground_anchors(
                            refined["target_mask"],
                            ("up" if str(stage.get("form", "")) ==
                             "VERTICAL_UP" else "down"),
                            count=6)
                    else:
                        refined_anchors = self._ground_anchors(
                            refined["target_mask"], refined.get("point"))
                    refined["ground_anchors"] = history_safe_anchors(
                        refined, refined_anchors)
                    if (refined.get("point") is not None and
                            not refined.get("hard_excluded", False) and
                            any(route_ray_allowed(refined, x) for x in xs) and
                            refined["ground_anchors"]):
                        # Failed seam probes are diagnostics only. Appending
                        # them would grow the audited eight-plus-one contact
                        # sheet past its schema and make the third probe a
                        # technical failure rather than a navigation result.
                        candidates.append(refined)
                        index = len(candidates) - 1
                        corridor_allowed.append(index)
                        break
            if not corridor_allowed:
                # A persisted absolute bearing is reliable within an open
                # corridor, but becomes stale at a real corner: after the
                # first partial edge the connected route can turn into the
                # adjacent panorama sector and contain no floor on the old
                # compass ray.  Permit one RGB-only local bend reacquisition
                # for supported same-stage continuation.  The expanded ray
                # remains within 90 degrees of the committed bearing and is
                # still filtered by the incoming core, blocked directions,
                # and the original strict ground mask.  The VLM must choose
                # among these visible candidates; this is not a geometric or
                # demonstration-derived branch choice.
                if (route_continuation and
                        bool(route_corridor.get(
                            "allow_bend_fallback", True))):
                    bend_allowed = []
                    for index in allowed:
                        candidate = candidates[index]
                        mask = np.asarray(candidate["target_mask"], bool)
                        if not mask.any() or candidate.get(
                                "hard_excluded", False):
                            continue
                        candidate["route_corridor_bend_fallback"] = {
                            "active": True,
                            "maximum_delta_deg": 90.0,
                            "source": "exhausted_strict_primary_corridor",
                            "policy": (
                                "one adjacent RGB-ground route bend after a "
                                "supported partial edge; incoming and blocked "
                                "cores remain forbidden"),
                        }
                        anchors = self._ground_anchors(
                            mask, candidate.get("point"), count=12)
                        safe = history_safe_anchors(candidate, anchors)
                        if safe:
                            candidate["ground_anchors"] = safe
                            bend_allowed.append(index)
                        else:
                            candidate.pop("route_corridor_bend_fallback", None)
                    corridor_allowed = bend_allowed
            if not corridor_allowed:
                raise RuntimeError(
                    "No strict-ground ray remains inside the supported-partial "
                    "+/-45-degree recovery corridor or its single adjacent "
                    "RGB bend")
            allowed = corridor_allowed
        direction_gate = {
            "active": False, "sector": None,
            "reason": "no explicit turn command",
            "allowed_before_gate": list(allowed),
            "allowed_after_gate": list(allowed),
        }
        if self.point_selection_prompt_version in {
                "v4_sector_identity_gates",
                "v5_two_stage_sector_anchor",
                "v6_validated_relation_two_stage",
                "v7_single_stage_relation_gates",
                "v8_object_relation_router",
                "v9_single_stage_identity_relation_gates",
                "v10_approach_relation_router",
                "v11_eight_view_refinement",
                "v12_eight_view_dual_candidate",
                "v13_eight_view_native_rgb",
                "v14_rgb_evidence_refinement",
                "v15_rgb_center_preferred",
                "v16_turn_three_view_gate",
                "v17_turn_three_view_commit_prior",
                "v18_history_safe_refinement",
                "v19_pixel_ray_history_and_reverse_override"}:
            instruction = direction_instruction_lower
            form = direction_form
            vertical_continuation_active = bool(
                form in {"VERTICAL_UP", "VERTICAL_DOWN"} and
                (stage.get("metadata", {}) or {}).get(
                    "vertical_continuation", {}).get("active"))
            sector = None
            compound_turn_soft = bool(
                self._task30_route_anchor and
                form not in {"TURN_LEFT", "TURN_RIGHT", "TURN_AROUND"} and
                re.search(r"\b(?:exit|enter|walk|go|head|continue|pass|through)\b"
                          r"[^.]{0,100}\bturn(?:\s+\w+){0,3}\s+(?:left|right)\b",
                          instruction))
            if (form == "TURN_AROUND" or
                    re.search(r"\bturn\s+(?:all\s+the\s+way\s+)?around\b", instruction)):
                sector = "rear"
            elif (form == "TURN_LEFT" or
                  re.search(r"\bturn(?:\s+\w+){0,3}\s+left\b", instruction)):
                sector = "left"
            elif (form == "TURN_RIGHT" or
                  re.search(r"\bturn(?:\s+\w+){0,3}\s+right\b", instruction)):
                sector = "right"
            if vertical_continuation_active:
                # A compound stair clause's left/right phrase locates the
                # first tread.  Once a real partial ascent/descent node has
                # been reached, repeating that side turn can leave the stair
                # flight.  Continue relative to the arrived stair heading.
                sector = None
                direction_gate["reason"] = (
                    "vertical continuation suppresses the already executed "
                    "initial turn phrase")
            if compound_turn_soft:
                # The first physical transition is the portal/region verb;
                # the later turn is a follow-on maneuver.  Do not force a
                # 90-degree sector before the agent has exited/entered it.
                sector = None
                direction_gate["reason"] = (
                    "compound portal/region transition keeps turn as a soft "
                    "follow-on prior")
            sector_allowed = []
            strict_turn_commit = use_rgb_evidence_refinement
            direction_source = list(allowed)
            if (sector in {"left", "right", "rear"} and
                    self._history_safe_refinement and
                    explicit_turn_command):
                # A literal turn authorizes its commanded side even when that
                # side overlaps the soft incoming-direction cone.  Sequence-
                # blocked directions remain hard and are never reopened.
                direction_source = [
                    index for index, candidate in enumerate(candidates)
                    if candidate.get("point") is not None and
                    (not candidate.get("hard_excluded", False) or
                     candidate.get("soft_blocked_tangent_reopened", False))]
            for index in direction_source:
                relative_deg = math.degrees(float(candidates[index].get(
                    "relative_yaw_rad", 0.0)))
                relative_deg = (relative_deg + 180.0) % 360.0 - 180.0
                # In the Habitat yaw convention used by the executor,
                # physical left-turn actions increase yaw.  Therefore the
                # semantic left sector is positive relative yaw; right is
                # negative.  Keep this sign convention explicit at the hard
                # gate so VLM direction cannot send a valid left clause into
                # the opposite branch.
                if (sector == "left" and
                        (45.0 if use_turn_three_view_gate else
                         75.0 if strict_turn_commit else 30.0) <=
                        relative_deg <=
                        (135.0 if use_turn_three_view_gate else 150.0)):
                    sector_allowed.append(index)
                elif (sector == "right" and
                      -(135.0 if use_turn_three_view_gate else 150.0) <=
                      relative_deg <=
                      -(45.0 if use_turn_three_view_gate else
                        75.0 if strict_turn_commit else 30.0)):
                    sector_allowed.append(index)
                elif (sector == "rear" and abs(relative_deg) >=
                      (135.0 if strict_turn_commit else 110.0)):
                    sector_allowed.append(index)
            if sector is not None:
                if (self._stage1_turn_soft and sector in {"left", "right"} and
                        not re.fullmatch(
                            r"\s*turn\s+(?:to\s+the\s+)?(?:left|right)\s*[.!]?\s*",
                            instruction, flags=re.IGNORECASE)):
                    # Compound turn clauses can ask to cross a room or reach a
                    # doorway whose demonstrated bearing is only slightly off
                    # the current heading.  Keep a connected forward/45° ray
                    # as a legal competitor; the VLM still has to ground the
                    # named opening and the anchor pass enforces the final ray.
                    forward_candidates = [
                        index for index in direction_source
                        if abs((math.degrees(float(
                            candidates[index].get("relative_yaw_rad", 0.0))) +
                                180.0) % 360.0 - 180.0) <= 30.0001]
                    sector_allowed = list(dict.fromkeys(
                        sector_allowed + forward_candidates))
                if (not sector_allowed and sector in {"left", "right"} and
                        self._relation_route_guard):
                    # A previous portal hop can already align the camera with
                    # the instructed branch.  If every strict +/-45/90/135
                    # side sector lacks a targetable RGB floor anchor, retain
                    # a forward connected-floor fallback rather than failing
                    # before the VLM can inspect it.  This is an explicit,
                    # audited exception for missing 2-D support—not a scene
                    # or path-specific direction override.
                    forward_fallback = [
                        index for index in direction_source
                        if abs((math.degrees(float(candidates[index].get(
                            "relative_yaw_rad", 0.0))) + 180.0) % 360.0 -
                               180.0) <= 45.0001 and
                        candidates[index].get("point") is not None]
                    if forward_fallback:
                        sector_allowed = forward_fallback
                        direction_gate = {
                            **direction_gate,
                            "fallback": "forward_connected_floor_when_side_ground_missing",
                        }
                if not sector_allowed:
                    raise RuntimeError(
                        "No floor-bearing candidate remains inside the "
                        f"explicit {sector} direction gate")
                allowed = sector_allowed
                direction_gate = {
                    "active": True, "sector": sector,
                    "reason": (
                        "compound turn keeps a forward/45-degree route ray as "
                        "a legal competitor while preserving turn-side views"
                        if self._stage1_turn_soft and sector in {"left", "right"}
                        else
                        "explicit left/right turn is hard-limited to the three "
                        "corresponding 45/90/135-degree side views before RGB "
                        "semantic ranking" if use_turn_three_view_gate and
                        sector in {"left", "right"} else
                        "explicit turn language/form; target identity may rank "
                        "within the sector but may not reopen the opposite sector"),
                    "three_view_gate": bool(
                        use_turn_three_view_gate and
                        sector in {"left", "right"}),
                    "allowed_before_gate": direction_gate[
                        "allowed_before_gate"],
                    "allowed_after_gate": list(allowed),
                }
            # Descending/ascending is a forward physical transition unless
            # the instruction explicitly says to reverse or turn around.
            # Without this soft front gate a salient rear corridor can win
            # over the stair/landing view, even though it contradicts the
            # first move.  The gate is V22-only and remains purely RGB/2-D.
            vertical_form = form in {"VERTICAL_DOWN", "VERTICAL_UP"}
            vertical_continuation = vertical_continuation_active
            explicit_vertical_reverse = bool(re.search(
                r"\b(?:behind|backward|back|reverse|turn\s+(?:around|back))\b",
                instruction))
            if (self._task30_route_anchor and vertical_form and
                    not explicit_vertical_reverse):
                vertical_front = []
                for index in allowed:
                    rel = math.degrees(float(candidates[index].get(
                        "relative_yaw_rad", 0.0)))
                    rel = (rel + 180.0) % 360.0 - 180.0
                    if abs(rel) <= 90.0:
                        vertical_front.append(index)
                if vertical_front:
                    direction_gate = {
                        "active": True,
                        "sector": "front_vertical",
                        "reason": (
                            "vertical stair transition defaults to the forward "
                            "hemisphere; rear is allowed only with explicit "
                            "reverse language"),
                        "allowed_before_gate": list(allowed),
                        "allowed_after_gate": list(vertical_front),
                    }
                    allowed = vertical_front
            if (route_continuation and form in {
                    "ADVANCE_STRAIGHT", "PASS_LANDMARK", "CROSS_SPACE",
                    "BETWEEN_OBJECTS"}):
                forward_continuation = []
                continuation_limit_deg = (
                    45.0 if vertical_form else 30.0)
                for index in allowed:
                    rel = math.degrees(float(candidates[index].get(
                        "relative_yaw_rad", 0.0)))
                    rel = (rel + 180.0) % 360.0 - 180.0
                    if abs(rel) <= continuation_limit_deg + 0.0001:
                        forward_continuation.append(index)
                if forward_continuation:
                    before = list(allowed)
                    allowed = forward_continuation
                    direction_gate = {
                        "active": True,
                        "sector": ("vertical_continuation" if
                                   vertical_continuation else
                                   "supported_partial_continuation"),
                        "reason": (
                            "one supported partial edge commits the single "
                            "permitted lookahead to the current forward route"),
                        "allowed_before_gate": before,
                        "allowed_after_gate": list(allowed),
                    }
            terminal_form = str(stage.get("form", "")).upper()
            terminal_text = " ".join(str(stage.get(key, "")) for key in (
                "navigation_instruction", "semantic_spatial_target",
                "spatial_relation")).lower()
            explicit_terminal_side = bool(re.search(
                r"\b(?:behind|back|rear|left|right|around|backside)\b",
                terminal_text))
            if (self._task30_route_anchor and
                    terminal_form in {"APPROACH_LANDMARK", "STOP_WAIT"} and
                    not explicit_terminal_side and
                    not bare_turn_setup.get("active")):
                forward_terminal = []
                for index in allowed:
                    rel = math.degrees(float(candidates[index].get(
                        "relative_yaw_rad", 0.0)))
                    rel = (rel + 180.0) % 360.0 - 180.0
                    if abs(rel) <= 75.0001:
                        forward_terminal.append(index)
                if forward_terminal:
                    before = list(allowed)
                    allowed = forward_terminal
                    direction_gate = {
                        "active": True,
                        "sector": "forward_terminal_relation",
                        "reason": (
                            "an unqualified approach/stop relation preserves "
                            "route continuity and cannot select the rear "
                            "hemisphere"),
                        "allowed_before_gate": before,
                        "allowed_after_gate": list(allowed),
                    }
        relation_gate = {
            "active": False,
            "form": str(stage.get("form", "")),
            "reason": "no validated portal-floor relation gate",
            "allowed_before_gate": list(allowed),
            "relation_evidence_views": [],
            "allowed_after_gate": list(allowed),
        }
        relation_direction_gate = {
            "active": False,
            "form": str(stage.get("form", "")),
            "reason": "no V21 side-route requirement",
            "allowed_before_gate": list(allowed),
            "allowed_after_gate": list(allowed),
        }
        if self._relation_route_guard and not explicit_reverse_command:
            # A behind/around/diagonal first move is a side transition around
            # the reference, not a 180-degree jump to the rear.  Constrain
            # only these semantic forms to the three adjacent side sectors
            # (±45/±90); the VLM still chooses the left/right side from RGB.
            form = str(stage.get("form", ""))
            instruction = str(stage.get(
                "navigation_instruction", "")).lower()
            relation_text = " ".join(str(stage.get(key, ""))
                                     for key in (
                                         "semantic_spatial_target",
                                         "spatial_relation", "landmark"))
            side_route = (
                form == "CIRCUMNAVIGATE" or
                (form == "APPROACH_LANDMARK" and bool(re.search(
                    r"\b(behind|backside|around|beyond)\b", 
                    instruction + " " + relation_text))) or
                (form == "TRAVERSE_PORTAL_REGION" and bool(re.search(
                    r"\b(diagonal(?:ly)?|angled?)\b",
                    instruction + " " + relation_text))))
            if side_route:
                side_allowed = []
                for index in allowed:
                    yaw_deg = math.degrees(float(
                        candidates[index].get("relative_yaw_rad", 0.0)))
                    yaw_deg = (yaw_deg + 180.0) % 360.0 - 180.0
                    if 30.0 <= abs(yaw_deg) <= 100.0:
                        side_allowed.append(index)
                # An unqualified "around the obstacle" does not imply an
                # immediate lateral turn.  Indoor demonstrations often first
                # approach the obstacle on the current route, then expose the
                # side clearance.  Keep the forward sectors as legal RGB
                # competitors for this new strict-start variant; explicit
                # left/right wording remains side-gated.
                # An unqualified around/backside clause has two physical
                # phases: expose a side-clearance ray, then continue along
                # that lane.  Once this hop has action history, suppressing
                # the forward sectors would force every retry to turn
                # sideways again and can send the agent behind the intended
                # obstacle.  Keep forward as a legal RGB competitor after
                # progress, while explicit left/right relations remain
                # side-gated.  The v31 name is retained for reproducibility
                # of older runs; the history condition is the generic rule.
                allow_forward_approach = bool(
                    form == "CIRCUMNAVIGATE" and not re.search(
                        r"\b(?:left|right)\b", instruction + " " + relation_text)
                    and (bool(action_history) or
                         self.requested_point_selection_prompt_version in {
                             "v31_circumnavigate_forward_competitor",
                             "v34_unified_history_route_guard",
                             "v35_qualified_region_detector_grounding"}))
                if allow_forward_approach:
                    forward = [
                        index for index in allowed
                        if abs((math.degrees(float(candidates[index].get(
                            "relative_yaw_rad", 0.0))) + 180.0) % 360.0 -
                               180.0) < 30.0001]
                    # Keep both a short forward approach and visible
                    # side-clearance rays as legal competitors.  The VLM's
                    # RGB route review decides whether the obstacle already
                    # blocks the front ray; the post-selection anchor check
                    # below prevents a side sector from being rotated to the
                    # opposite branch.  This is a generic relation rule.
                    side_allowed = side_allowed + forward
                if side_allowed:
                    before = list(allowed)
                    allowed = side_allowed
                    relation_direction_gate = {
                        "active": True,
                        "form": form,
                        "reason": (
                            "unqualified circum-navigation keeps forward approach "
                            "as a legal competitor; explicit side/diagonal relations "
                            "use a local ±45/±90 transition"
                            if allow_forward_approach else
                            "behind/around/diagonal relation uses a local side "
                            "transition; forward and rear sectors are not first "
                            "anchors when a ±45/±90 floor sector exists"),
                        "allowed_before_gate": before,
                        "allowed_after_gate": list(allowed),
                    }
        if first_step_route_guard_active and not explicit_reverse_command:
            # A first stage without an explicit reverse/U-turn must begin from
            # the forward-facing 180° half of the compass (0, ±45, ±90).
            # Rear sectors are common doorway/counter false positives.  A
            # compound PASS→PORTAL instruction is allowed the ±135° side view
            # because the portal can appear after the landmark has been
            # passed; simpler first stages stay within the forward 180° half.
            # Explicit turns/U-turns are exempt.
            before = list(allowed)
            stage_form = str(stage.get("form", ""))
            stage_secondary = set(stage.get("secondary_forms", []) or [])
            stage_text = " ".join(str(stage.get(key, "")) for key in (
                "navigation_instruction", "semantic_spatial_target",
                "spatial_relation", "completion_cue")).lower()
            # A literal side-qualified portal (for example, "the door on
            # the right") is a bearing instruction, even when it occurs in
            # an EXIT/ENTER form rather than a TURN form.  Keep the normal
            # forward-half preference only if a forward candidate actually
            # contains a connected portal-floor relation.  If it does not,
            # admit the nearest side sector (45/90/135 degrees), never the
            # exact rear sector.  This is a form/2-D relation rule and does
            # not depend on an episode, path, depth, or execution result.
            explicit_side_portal = bool(re.search(
                r"\b(?:door|doorway|opening|entrance|exit)\b[^.]{0,45}"
                r"\b(?:on|to|at|from)?\s*(?:the\s+)?(?:left|right)\b|"
                r"\b(?:left|right)\b[^.]{0,45}\b(?:door|doorway|opening|"
                r"entrance|exit)\b", stage_text))
            compound_route = (
                stage_form == "PASS_LANDMARK" and bool(
                    {"TRAVERSE_PORTAL_REGION", "ENTER_REGION", "EXIT_REGION"} &
                    stage_secondary))
            max_first_step_yaw = 135.0001 if compound_route else 90.0001
            forward_route = [
                index for index in allowed
                if abs(((math.degrees(float(candidates[index].get(
                    "relative_yaw_rad", 0.0))) + 180.0) % 360.0) - 180.0)
                <= max_first_step_yaw]
            if forward_route:
                allowed = forward_route
                direction_gate = {
                    **direction_gate,
                    "active": True,
                    "sector": "forward_route",
                    "reason": (
                        ("compound PASS→PORTAL stage permits a ±135-degree "
                         "side portal after the landmark"
                         if compound_route else
                         "first non-reverse stage is limited to the forward "
                         "180-degree half (0, ±45, ±90)")),
                    "allowed_before_gate": before,
                    "allowed_after_gate": list(allowed),
                }
            if (explicit_side_portal and stage_form in {
                    "EXIT_REGION", "ENTER_REGION", "TRAVERSE_PORTAL_REGION",
                    "SELECT_PORTAL"}):
                def _signed_rel(index):
                    value = math.degrees(float(candidates[index].get(
                        "relative_yaw_rad", 0.0)))
                    return (value + 180.0) % 360.0 - 180.0

                def _portal_floor_evidence(index):
                    strategy = candidates[index].get(
                        "strategy_application", {}) or {}
                    return bool(
                        candidates[index].get("point") is not None and
                        not candidates[index].get("hard_excluded", False) and
                        strategy.get("mode") == "floor_through_detected_portal" and
                        int(strategy.get("candidate_pixels", 0) or 0) >= 24 and
                        bool(strategy.get("relation_detection_evidence_in_view", False)))

                forward_portal = [index for index in allowed
                                  if abs(_signed_rel(index)) <= 90.0001 and
                                  _portal_floor_evidence(index)]
                side_portal = [index for index in before
                               if 45.0 <= abs(_signed_rel(index)) <= 135.0001 and
                               _portal_floor_evidence(index)]
                if side_portal and not forward_portal:
                    allowed = list(dict.fromkeys(side_portal))
                    direction_gate = {
                        **direction_gate,
                        "active": True,
                        "sector": "side_qualified_portal",
                        "reason": (
                            "explicit left/right portal relation reopened the "
                            "nearest side 45/90/135-degree portal-floor view "
                            "because no forward portal-floor candidate was valid"),
                        "allowed_before_gate": before,
                        "allowed_after_gate": list(allowed),
                    }
            if compound_route:
                # Compound PASS→PORTAL stages should use an opening that is
                # visibly tied to the named landmark. A portal detected only
                # in a different room (bathroom/side corridor) is not enough
                # evidence for the first move. Keep same-view co-occurrence
                # whenever RGB detections provide it; otherwise retain the
                # broader floor set as a conservative fallback.
                landmark_words = {
                    word for word in re.findall(
                        r"[a-z]+", str(stage.get("landmark", "")).lower())
                    if len(word) > 2 and word not in {
                        "the", "and", "into", "room", "area",
                    }}
                portal_words = {
                    "door", "doorway", "opening", "hallway", "portal",
                }
                coherent_views = []
                for index in allowed:
                    records = candidates[index].get("detection_records", [])
                    has_landmark = any(
                        landmark_words & set(re.findall(
                            r"[a-z]+", str(record.get("label", "")).lower()))
                        for record in records)
                    has_portal = any(
                        portal_words & set(re.findall(
                            r"[a-z]+", str(record.get("label", "")).lower()))
                        for record in records)
                    if has_landmark and has_portal:
                        coherent_views.append(index)
                if coherent_views:
                    before = list(allowed)
                    allowed = coherent_views
                    direction_gate = {
                        **direction_gate,
                        "active": True,
                        "sector": "landmark_portal_context",
                        "reason": (
                            "compound PASS→PORTAL keeps views where the named "
                            "landmark and outgoing portal co-occur in RGB"),
                        "allowed_before_gate": before,
                        "allowed_after_gate": list(allowed),
                    }
        if self.point_selection_prompt_version in {
                "v6_validated_relation_two_stage",
                "v7_single_stage_relation_gates",
                "v8_object_relation_router",
                "v9_single_stage_identity_relation_gates",
                "v10_approach_relation_router",
                "v11_eight_view_refinement",
                "v12_eight_view_dual_candidate",
                "v13_eight_view_native_rgb",
                "v14_rgb_evidence_refinement",
                "v15_rgb_center_preferred",
                "v16_turn_three_view_gate",
                "v17_turn_three_view_commit_prior",
                "v18_history_safe_refinement",
                "v19_pixel_ray_history_and_reverse_override"}:
            # For portal-bound relations, a detector proposal is promoted to a
            # gate only after the downstream strategy has produced a non-trivial
            # intersection with DINO+SAM walkable ground.  This is stronger
            # than a text label alone and remains generic across rooms/scenes.
            portal_forms = {
                "EXIT_REGION", "ENTER_REGION", "SELECT_PORTAL",
                "TRAVERSE_PORTAL_REGION",
            }
            form = str(stage.get("form", ""))
            secondary_forms = set(stage.get("secondary_forms", []) or [])
            compound_portal = (
                form == "PASS_LANDMARK" and bool(
                    {"TRAVERSE_PORTAL_REGION", "ENTER_REGION", "EXIT_REGION"} &
                    secondary_forms))
            relation_views = []
            if form in portal_forms or compound_portal:
                for index in allowed:
                    strategy = candidates[index].get(
                        "strategy_application", {}) or {}
                    if (strategy.get("mode") == "floor_through_detected_portal" and
                            int(strategy.get("candidate_pixels", 0) or 0) >= 24 and
                            bool(strategy.get(
                                "relation_detection_evidence_in_view", False))):
                        relation_views.append(index)
            if relation_views:
                before = list(allowed)
                advisory_only = self.point_selection_prompt_version in {
                    "v13_eight_view_native_rgb",
                    "v14_rgb_evidence_refinement",
                    "v15_rgb_center_preferred",
                    "v16_turn_three_view_gate",
                    "v17_turn_three_view_commit_prior",
                    "v18_history_safe_refinement",
                    "v19_pixel_ray_history_and_reverse_override",
                } and (not self._first_step_route_guard or
                       self.requested_point_selection_prompt_version in {
                           "v32_first_stage_shallow_route",
                           "v34_unified_history_route_guard"})
                # V33 is the current strict RGB route policy.  Once a portal
                # proposal and connected floor mask co-occur, keep that
                # evidence as a hard candidate gate; otherwise the VLM can
                # reopen unrelated side/rear doors even when a valid exit
                # sector is available.  The gate is still entirely form/2-D
                # and does not expose depth, a reference path, or episode id.
                if self.requested_point_selection_prompt_version in {
                        "v33_stop_relation_near_side",
                        "v34_unified_history_route_guard"}:
                    advisory_only = False
                # In a compound PASS→PORTAL stage, a portal mask can be
                # clipped or absent even though a neighboring view contains
                # the actual connected floor.  Keep all floor-bearing views as
                # soft evidence and let the RGB route review choose among
                # them; hard narrowing to one accidental portal box is unsafe.
                if compound_portal:
                    advisory_only = True
                if not advisory_only:
                    allowed = relation_views
                relation_gate = {
                    "active": not advisory_only,
                    "advisory_only": advisory_only,
                    "form": form,
                    "reason": (
                        "RGB-only portal detector geometry intersects at least "
                        "24 pixels of DINO+SAM walkable ground; V13 treats "
                        "this as fallible ranking evidence rather than a hard "
                        "direction gate" if advisory_only else
                        "portal detector geometry intersects at least 24 pixels "
                        "of DINO+SAM walkable ground beyond the portal"),
                    "allowed_before_gate": before,
                    "relation_evidence_views": list(relation_views),
                    "allowed_after_gate": list(allowed),
                }
        if (self._qualified_region_detector_grounding and
                str(stage.get("form", "")).upper() == "ENTER_REGION"):
            qualifier_pattern = re.compile(
                r"\b(?:with|containing|under|beneath)\s+"
                r"(?:the\s+|a\s+|an\s+)?"
                r"([a-z][a-z0-9]*(?:\s+[a-z][a-z0-9]*){0,3})")
            qualifier_match = None
            for key in ("landmark", "navigation_instruction",
                        "semantic_spatial_target", "completion_cue"):
                qualifier_match = qualifier_pattern.search(
                    str(stage.get(key, "")).lower())
                if qualifier_match:
                    break

            def qualifier_token(word):
                word = str(word).lower()
                if len(word) > 4 and word.endswith("ies"):
                    return word[:-3] + "y"
                if (len(word) > 4 and
                        word.endswith(("ches", "shes", "sses", "xes", "zes"))):
                    return word[:-2]
                if (len(word) > 3 and word.endswith("s") and
                        not word.endswith("ss")):
                    return word[:-1]
                return word
            qualifier_tokens = set()
            if qualifier_match:
                qualifier_phrase = re.split(
                    r"\b(?:and|while|before|after|that|where)\b|[,.;!?]",
                    qualifier_match.group(1), maxsplit=1)[0]
                qualifier_tokens = {
                    qualifier_token(word)
                    for word in re.findall(r"[a-z]+", qualifier_phrase)
                    if word not in {
                        "the", "with", "room", "area", "floor", "ground",
                        "inside", "just", "free", "walkable"}}
            qualified_detection_views = []
            if qualifier_tokens:
                for index, candidate in enumerate(candidates):
                    for record in candidate.get("detection_records", []):
                        score = float(record.get("score", 0.0) or 0.0)
                        area = float(record.get(
                            "mask_area_fraction", 0.0) or 0.0)
                        label_tokens = {
                            qualifier_token(word)
                            for word in re.findall(
                                r"[a-z]+", str(record.get(
                                    "label", "")).lower())}
                        if (qualifier_tokens.intersection(label_tokens) and
                                score >= 0.28 and 0.002 <= area <= 0.50):
                            qualified_detection_views.append(index)
                            break
            qualified_detection_views = list(dict.fromkeys(
                qualified_detection_views))
            if qualified_detection_views:
                def qualified_view_neighborhood(candidate_index):
                    candidate_deg = math.degrees(float(candidates[
                        candidate_index].get("relative_yaw_rad", 0.0)))
                    return any(abs((candidate_deg - math.degrees(float(
                        candidates[evidence_index].get(
                            "relative_yaw_rad", 0.0))) + 180.0) % 360.0 -
                                       180.0) <= 45.0001
                               for evidence_index in qualified_detection_views)

                exact_qualified_allowed = [
                    index for index in allowed
                    if index in qualified_detection_views]
                qualified_neighborhood_allowed = [
                    index for index in allowed
                    if qualified_view_neighborhood(index)]
                v31_joint_gate = (
                    self.requested_point_selection_prompt_version ==
                    "v31_circumnavigate_forward_competitor")
                grounded_allowed = (
                    list(exact_qualified_allowed) or
                    qualified_neighborhood_allowed
                    if v31_joint_gate else
                    qualified_neighborhood_allowed)
                # A destination-qualified ENTER relation has two independent
                # pieces: the named qualifier must be localized and the same
                # view must contain a portal-ground intersection.  When both
                # proposal sets exist, use their exact intersection before
                # falling back to a one-sector qualifier seam.  This removes
                # a common consistent-hallucination failure where two VLM
                # calls describe an ungrounded side view as the named region.
                # Both proposal sets are RGB/2-D detector products; no depth,
                # navmesh, GT route, or goal coordinate participates.
                if relation_views and v31_joint_gate:
                    joint_allowed = [
                        index for index in grounded_allowed
                        if index in relation_views]
                    if joint_allowed:
                        grounded_allowed = joint_allowed
                if grounded_allowed:
                    before = list(allowed)
                    allowed = grounded_allowed
                    relation_gate = {
                        **relation_gate,
                        "active": True,
                        "form": "ENTER_REGION",
                        "reason": (
                            "identity-qualified destination uses only floor "
                            "rays in the same or adjacent 45-degree sector as "
                            "a localized DINO qualifier proposal"),
                        "qualified_region_tokens": sorted(qualifier_tokens),
                        "qualified_detection_views": (
                            qualified_detection_views),
                        "qualified_exact_allowed_views": (
                            exact_qualified_allowed),
                        "qualified_portal_joint_gate": bool(
                            relation_views and set(grounded_allowed) <=
                            set(relation_views)),
                        "allowed_before_gate": before,
                        "allowed_after_gate": list(allowed),
                    }
        rgb_evidence_plan = None
        if use_rgb_evidence_refinement:
            evidence_forms = {
                "EXIT_REGION", "ENTER_REGION", "TRAVERSE_PORTAL_REGION",
                "VERTICAL_UP", "VERTICAL_DOWN", "PASS_LANDMARK",
                "FOLLOW_PATH_BOUNDARY", "APPROACH_LANDMARK",
                "TURN_TO_LANDMARK",
            }
            form = str(stage.get("form", ""))
            records = []
            rgb_evidence_context = {}
            generic_portal_words = {
                "door", "doorway", "open", "opening", "hallway", "portal",
            }

            def evidence_view_delta(first, second):
                first_deg = math.degrees(float(candidates[first].get(
                    "relative_yaw_rad", 0.0)))
                second_deg = math.degrees(float(candidates[second].get(
                    "relative_yaw_rad", 0.0)))
                return abs(math.degrees((math.radians(
                    first_deg - second_deg) + math.pi) %
                    (2 * math.pi) - math.pi))

            for index, candidate in enumerate(candidates):
                # A detector hit in an incoming/blocked direction may provide
                # context, but it must never become the base of a new camera
                # request.  Keep only evidence that has legal ground in the
                # same or an adjacent 45-degree sector.
                if (self._history_safe_refinement and not any(
                            evidence_view_delta(index, allowed_index) <= 45.0001
                            for allowed_index in allowed)):
                    continue
                height, width = candidate["rgb"].shape[:2]
                for record in candidate.get("detection_records", []):
                    box = record.get("box_xyxy")
                    if box is None or len(box) != 4:
                        continue
                    center_x = 0.5 * (float(box[0]) + float(box[2]))
                    center_x_norm = center_x / max(1.0, width - 1.0)
                    label_words = set(re.findall(
                        r"[a-z]+", str(record.get("label", "")).lower()))
                    score = float(record.get("score", 0.0) or 0.0)
                    centrality = max(
                        0.15, 1.0 - 1.25 * abs(center_x_norm - 0.5))
                    records.append({
                        "view_index": index,
                        "label": str(record.get("label", "")),
                        "label_words": label_words,
                        "score": score,
                        "center_x_norm": center_x_norm,
                        "central_score": score * centrality,
                    })
            if form in evidence_forms and records:
                usable = list(records)
                # A PASS_LANDMARK stage may explicitly carry a secondary
                # portal/traversal transition.  In that compound case the
                # first physical route is the opening, not the most confident
                # object box (which is often a counter/sink seen from a side
                # view).  Prefer portal evidence for the RGB route prior while
                # retaining the landmark as contextual evidence in the prompt.
                secondary_forms = set(stage.get("secondary_forms", []) or [])
                portal_transition = (
                    "TRAVERSE_PORTAL_REGION" in secondary_forms or
                    "ENTER_REGION" in secondary_forms or
                    "EXIT_REGION" in secondary_forms)
                if form == "PASS_LANDMARK" and portal_transition:
                    portal_records = [record for record in usable if
                                      record["label_words"] & generic_portal_words]
                    if portal_records:
                        # Keep the portal tied to the named landmark's local
                        # room context.  A high-scoring doorway elsewhere in
                        # the panorama may be a bathroom/side opening rather
                        # than the exit reached after passing the landmark.
                        landmark_records = [record for record in usable if not
                                            (record["label_words"] &
                                             generic_portal_words)]
                        if landmark_records:
                            landmark_anchor = max(
                                landmark_records,
                                key=lambda record: record["score"])
                            # The portal may be one compass sector beyond the
                            # landmark after the agent passes it.  Use a 90°
                            # context neighborhood for compound routes; a
                            # farther (135°) portal is still rejected as a
                            # likely unrelated room/opening.
                            context_gate_deg = 90.0
                            nearby_portals = [record for record in portal_records
                                              if evidence_view_delta(
                                                  record["view_index"],
                                                  landmark_anchor["view_index"])
                                              <= context_gate_deg + 1e-4]
                            usable = nearby_portals or portal_records
                            rgb_evidence_context = {
                                "landmark_anchor_view_index": int(
                                    landmark_anchor["view_index"]),
                                "landmark_anchor_label": landmark_anchor[
                                    "label"],
                                "portal_context_gate_deg": context_gate_deg,
                                "portal_context_gate_active": bool(
                                    nearby_portals),
                            }
                        else:
                            usable = portal_records
                if form == "EXIT_REGION" and stage.get("secondary_forms"):
                    specific = [record for record in usable if not
                                record["label_words"].issubset(
                                    generic_portal_words)]
                    if specific:
                        usable = specific
                if (form == "PASS_LANDMARK" and portal_transition):
                    # For the outgoing portal, image-center alignment is more
                    # reliable than raw detector confidence: edge-clipped
                    # doorway boxes otherwise steer the selected floor ray
                    # toward a wall or side opening.
                    preferred = max(
                        usable, key=lambda record: record["central_score"])
                    evidence_score_name = "central_score"
                elif form == "FOLLOW_PATH_BOUNDARY":
                    preferred = max(
                        usable, key=lambda record: record["central_score"])
                    evidence_score_name = "central_score"
                else:
                    preferred = max(usable, key=lambda record: record["score"])
                    evidence_score_name = "score"

                preferred_index = int(preferred["view_index"])
                neighborhood = [
                    index for index in allowed
                    if evidence_view_delta(index, preferred_index) <= 45.0001]
                image_width = candidates[preferred_index]["rgb"].shape[1]
                pixel_bearing_deg = math.degrees(math.atan(
                    ((preferred["center_x_norm"] * (image_width - 1)) -
                     (image_width - 1) / 2) / (image_width / 2)))
                preferred_yaw_deg = math.degrees(float(candidates[
                    preferred_index].get("relative_yaw_rad", 0.0)))
                ray_yaw_deg = (
                    preferred_yaw_deg - pixel_bearing_deg + 180.0) % 360.0 - 180.0
                rgb_evidence_plan = {
                    "active": True,
                    "form": form,
                    "preferred_view_index": preferred_index,
                    "preferred_label": preferred["label"],
                    "preferred_score": round(float(preferred[
                        evidence_score_name]), 4),
                    "preferred_center_x_norm": round(float(
                        preferred["center_x_norm"]), 4),
                    "preferred_image_ray_yaw_deg": round(ray_yaw_deg, 3),
                    "ground_bearing_neighborhood": neighborhood,
                    "input_policy": "RGB detections and 2D image geometry only",
                    **rgb_evidence_context,
                }
        vertical_form = str(stage.get("form", "")) in {
            "VERTICAL_UP", "VERTICAL_DOWN"}
        vertical_direction = (
            "up" if str(stage.get("form", "")) == "VERTICAL_UP" else "down")
        for candidate in candidates:
            if vertical_form:
                candidate["ground_anchors"] = self._vertical_ground_anchors(
                    candidate["target_mask"], vertical_direction, count=6)
            else:
                candidate["ground_anchors"] = self._ground_anchors(
                    candidate["target_mask"], candidate.get("point"))
            candidate["ground_anchors"] = history_safe_anchors(
                candidate, candidate["ground_anchors"])
        allowed_before_anchor_validation = list(allowed)
        allowed = [
            index for index in allowed
            if candidates[index].get("ground_anchors")]
        if not allowed and self._history_safe_refinement:
            # The semantic/direction gates already selected these views.  If
            # only the broad blocked-ray cones removed their strict-ground
            # pixels, retry the same gated views with the narrow 20° blocked
            # core.  This cannot introduce an opposite or ungated view.
            for index in allowed_before_anchor_validation:
                candidate = candidates[index]
                candidate_blocked = candidate.get(
                    "blocked_relative_yaws_rad", [])
                if not candidate_blocked or len(candidate_blocked) >= 5:
                    continue
                candidate["soft_blocked_tangent_reopened"] = True
                if vertical_form:
                    anchors = self._vertical_ground_anchors(
                        candidate["target_mask"], vertical_direction, count=6)
                else:
                    anchors = self._ground_anchors(
                        candidate["target_mask"], candidate.get("point"))
                candidate["ground_anchors"] = history_safe_anchors(
                    candidate, anchors)
            allowed = [
                index for index in allowed_before_anchor_validation
                if candidates[index].get("ground_anchors")]
        if (not allowed and route_corridor_active and
                refinement_provider is not None and len(candidates) < 9):
            # A candidate may survive the coarse corridor mask and only lose
            # every point during the final per-pixel history/block check.  In
            # that case the earlier seam recovery is never entered.  Acquire
            # one real RGB view centred on the already committed route and
            # run the exact same strict-ground/history validation.  This does
            # not reopen a semantic direction, an incoming ray, or a blocked
            # branch, and keeps the contact sheet at its audited 9-view cap.
            reference_yaw = float(candidates[0].get("yaw", 0.0)) - float(
                candidates[0].get("relative_yaw_rad", 0.0))
            corridor_relative = (
                route_corridor_absolute_yaw - reference_yaw + math.pi
            ) % (2.0 * math.pi) - math.pi
            refined = dict(refinement_provider(corridor_relative))
            if "depth" in refined:
                raise ValueError(
                    "final corridor refinement candidate must not contain depth")
            refined["view_index"] = len(candidates)
            refined["route_corridor_refinement"] = {
                "active": True,
                "phase": "final_history_safe_anchor_validation",
                "corridor_center_relative_yaw_deg": round(
                    math.degrees(corridor_relative), 3),
                "policy": (
                    "single local RGB reacquisition on the persisted route "
                    "after final per-pixel history validation"),
            }
            if vertical_form:
                anchors = self._vertical_ground_anchors(
                    refined["target_mask"], vertical_direction, count=6)
            else:
                anchors = self._ground_anchors(
                    refined["target_mask"], refined.get("point"))
            refined["ground_anchors"] = history_safe_anchors(refined, anchors)
            if (refined.get("point") is not None and
                    not refined.get("hard_excluded", False) and
                    refined["ground_anchors"]):
                candidates.append(refined)
                allowed = [len(candidates) - 1]
        if not allowed:
            raise RuntimeError(
                "No history-safe ground anchor remains after final RGB-ray "
                "validation")
        views = [self._overlay(c["rgb"], c["target_mask"], i, i not in allowed,
                               c.get("semantic_detections"),
                               c.get("small_seg_object_mask"), c["ground_anchors"])
                 for i, c in enumerate(candidates)]
        annotated_sheet = self._point_selection_contact_sheet(views)
        first_step_route_guidance = ""
        relation_route_guidance = ""
        if (self.point_selection_prompt_version in {
                "v4_sector_identity_gates",
                "v9_single_stage_identity_relation_gates",
                "v11_eight_view_refinement",
                "v12_eight_view_dual_candidate",
                "v13_eight_view_native_rgb",
                "v14_rgb_evidence_refinement",
                "v15_rgb_center_preferred",
                "v16_turn_three_view_gate",
                "v17_turn_three_view_commit_prior",
                "v18_history_safe_refinement",
                "v19_pixel_ray_history_and_reverse_override"} or
                use_two_stage):
            raw_views = []
            for index, candidate in enumerate(candidates):
                raw = np.asarray(candidate["rgb"]).copy()
                cv2.rectangle(raw, (0, 0), (112, 24), (0, 0, 0), -1)
                cv2.putText(
                    raw, f"RAW VIEW {index}", (5, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1,
                    cv2.LINE_AA)
                raw_views.append(raw)
            images = [
                self._point_selection_contact_sheet(raw_views),
                annotated_sheet]
        else:
            images = [annotated_sheet]
        view_refinement = None
        if self.point_selection_prompt_version in eight_view_versions:
            dual_candidate_review = (
                self.point_selection_prompt_version ==
                "v12_eight_view_dual_candidate")
            native_rgb_review = (
                self.point_selection_prompt_version in {
                    "v13_eight_view_native_rgb",
                    "v14_rgb_evidence_refinement",
                    "v15_rgb_center_preferred",
                    "v16_turn_three_view_gate",
                    "v17_turn_three_view_commit_prior",
                    "v18_history_safe_refinement",
                    "v19_pixel_ray_history_and_reverse_override",
                })
            initial_allowed = list(allowed)
            def signed_relative_yaw_deg(candidate):
                value = math.degrees(float(candidate.get(
                    "relative_yaw_rad", 0.0)))
                value = (value + 180.0) % 360.0 - 180.0
                return 180.0 if abs(value + 180.0) < 1e-6 else value

            initial_orientation = {
                str(index): {
                    "relative_yaw_deg": round(signed_relative_yaw_deg(
                        candidates[index]), 1),
                    "allowed": index in initial_allowed,
                    "has_ground_anchor": bool(
                        candidates[index].get("ground_anchors")),
                } for index in range(len(candidates))
            }
            route_sequence_active = bool(
                stage.get("secondary_forms") or re.search(
                    r"\b(?:past|then|after|before|towards?|until|and)\b",
                    str(stage.get("navigation_instruction", "")).lower()))
            review_schema = {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "decision", "view_index", "refined_relative_yaw_deg",
                    "reason"],
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": ["use_existing", "request_refined"],
                    },
                    "view_index": {
                        "type": "integer", "enum": initial_allowed,
                    },
                    "refined_relative_yaw_deg": {
                        "type": "number", "minimum": -360.0,
                        "maximum": 360.0,
                    },
                    "reason": {"type": "string", "maxLength": 420},
                },
            }
            if dual_candidate_review:
                review_schema["required"].extend([
                    "alternative_view_index", "confidence"])
                review_schema["properties"].update({
                    "alternative_view_index": {
                        "type": "integer", "enum": initial_allowed,
                    },
                    "confidence": {
                        "type": "number", "minimum": 0.0,
                        "maximum": 1.0,
                    },
                })
            if native_rgb_review:
                review_schema["required"].extend([
                    "target_centering", "confidence",
                    "strongest_competing_view_index",
                    "visible_route_evidence", "competitor_rejection",
                    "first_following_landmark",
                    "first_following_landmark_view_index",
                    "sequence_alignment"])
                review_schema["properties"].update({
                    "target_centering": {
                        "type": "string",
                        "enum": [
                            "centered", "left_edge", "right_edge",
                            "uncertain",
                        ],
                    },
                    "confidence": {
                        "type": "number", "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "strongest_competing_view_index": {
                        "type": "integer", "enum": initial_allowed,
                    },
                    "visible_route_evidence": {
                        "type": "string", "maxLength": 320,
                    },
                    "competitor_rejection": {
                        "type": "string", "maxLength": 320,
                    },
                    "first_following_landmark": {
                        "type": "string", "maxLength": 120,
                    },
                    "first_following_landmark_view_index": {
                        "type": "integer",
                        "enum": [-1, *range(len(candidates))],
                    },
                    "sequence_alignment": {
                        "type": "string",
                        "enum": [
                            "same_view", "adjacent_view", "not_visible",
                            "not_applicable",
                        ],
                    },
                })
            dual_review_guidance = (
                "Also return the strongest competing existing view, which must "
                "differ from view_index whenever at least two views are allowed, "
                "and calibrated confidence in the primary proposal. The next "
                "VLM pass will independently compare both candidates and may "
                "overrule the primary proposal. Do not choose an adjacent view "
                "automatically: the strongest semantic competitor may be across "
                "the panorama."
                if dual_candidate_review else "")
            native_review_guidance = (
                "IMAGES 1..8 are the eight separate, full-resolution clean RGB "
                "views, in view-index order 0..7. IMAGE 9 is a clean panoramic "
                "overview preserving adjacency around the robot. IMAGE 10 is an "
                "annotated eight-view overview containing only RGB-derived DINO+SAM "
                "ground, detector proposals, and numbered anchors. Inspect the "
                "separate RGB images for identity and route geometry; use the "
                "clean overview to compare continuous exits and the annotated "
                "overview only to verify ground availability. Also classify "
                "target_centering. Use centered only when both the semantic "
                "reference and its intended outgoing ground route are usable in "
                "the central 50% of the fallback view. A clipped reference, a "
                "route entering through the left/right image boundary, or "
                "uncertain identity requires request_refined with left_edge, "
                "right_edge, or uncertain respectively."
                if native_rgb_review else
                "IMAGE 1 is a clean RGB contact sheet; IMAGE 2 contains "
                "DINO+SAM ground, detections, and numbered anchors.")
            ground_surface_sanity_guidance = (
                "GROUND-MASK SANITY: Green is a DINO+SAM proposal, not proof "
                "of walkable floor. Inspect the aligned clean RGB before "
                "accepting it. Reject green pixels that visibly lie on a "
                "countertop, table, bed, sofa/seat, shelf, wall, or raised "
                "stair face. Valid floor/rug/carpet should form a plausible "
                "support surface connected toward the camera or the base of "
                "the intended opening. If the strongest proposal is on an "
                "object, choose another allowed ground-bearing sector; only "
                "request a local view when it can expose real floor without "
                "leaving the allowed direction."
                if self.point_selection_prompt_version in {
                    "v18_history_safe_refinement",
                    "v19_pixel_ray_history_and_reverse_override",
                    "v33_stop_relation_near_side",
                    "v34_unified_history_route_guard"} else "")
            if (self.requested_point_selection_prompt_version in {
                    "v33_stop_relation_near_side",
                    "v34_unified_history_route_guard"} and
                    str(stage.get("form", "")) in {
                        "TURN_LEFT", "TURN_RIGHT", "TURN_AROUND"}):
                ground_surface_sanity_guidance += """
TURN-RAY RGB SANITY (V33): For an explicit turn, a detector box or green
proposal on a broad wall/door panel, or an opening visible only as a thin edge,
is not a corridor.  The confirmed view must show a connected floor lane with
substantial visible width continuing after the turn; if the clean RGB is mostly
wall, reject that view even when numbered anchors exist.  Prefer the allowed
side view with the clearest corridor floor, not the view with the largest
detector mask.  A point on a wall or on a sliver at the image edge is invalid.
"""
            floor_proposal_evidence = {
                str(index): [
                    {
                        "label": str(record.get("label", "")),
                        "score": round(float(record.get("score", 0.0) or 0.0), 3),
                        "mask_area_fraction": round(float(
                            record.get("mask_area_fraction", 0.0) or 0.0), 4),
                        "box_xyxy": record.get("box_xyxy"),
                    }
                    for record in candidates[index].get(
                        "ground_detection_records", [])
                    if "depth" not in str(record).lower()
                ]
                for index in range(len(candidates))
            }
            first_step_route_guidance = (
                "FIRST-STEP ROUTE GUARD (V20): This is the first physical move "
                "for the decomposed instruction, so establish the route from "
                "the requested spatial relation before choosing a visually "
                "salient object. For PASS_LANDMARK, choose a connected floor "
                "ray that goes beyond the named landmark, not a side floor or "
                "the landmark itself. For TRAVERSE_PORTAL_REGION/ENTER/EXIT, "
                "choose the first clearly connected floor just through the "
                "named portal; reject a generic opening or a floor patch that "
                "does not continue through that opening. For APPROACH_LANDMARK "
                "or STOP_WAIT, choose the near-side floor on the same bearing "
                "as the identified landmark while keeping a safe offset. Compare "
                "all legal sectors and state why the strongest competitor is "
                "not the instructed route. This is a generic RGB/2-D rule: do "
                "not use demonstration-path, depth, navmesh, or hidden state "
                "information."
                if first_step_route_guard_active else "")
            if (first_step_route_guard_active and
                    self.requested_point_selection_prompt_version in {
                        "v32_first_stage_shallow_route",
                        "v34_unified_history_route_guard"}):
                first_step_route_guidance += """
SHALLOW-FIRST-ROUTE ADJUDICATION (V32):
- For an unqualified first-stage ENTER/EXIT/PASS/ADVANCE/CROSS clause, a
  legal connected floor ray within +/-45 degrees of the current heading is
  preferred when it shows the same named destination or continuous route as a
  side view.  Do not choose +/-90 merely because its detector box is larger.
- A side ray may win only when the near-forward views are visibly blocked,
  terminate at a wall/object, or do not show the named destination.  Explain
  that obstruction from clean RGB.  This is a comparison rule, not a blanket
  forward override; explicit left/right/around/turn and vertical clauses keep
  their form-specific directions.
"""
            if self._relation_route_guard:
                # These are image-grounded route tests, not scene-specific
                # overrides.  In particular, they prevent the common failure
                # where a salient object straight ahead is mistaken for the
                # floor ray that satisfies a behind/around relation.
                relation_route_guidance = """
RELATION-AWARE ROUTE REVIEW (V21):
- Do not assume that the first move is forward.  All floor-bearing sectors,
  including rear and rear-side sectors, are legal unless an explicit turn or
  history gate says otherwise.  Use the instruction and RGB route geometry.
- CIRCUMNAVIGATE or words such as around/backside/behind: the selected ray must
  pass the obstacle, not terminate at its front face.  A front view showing a
  couch/table/wall with no visible connected floor beyond it is only an
  approach/blocked view.  Prefer the nearest side sector where the obstacle's
  side edge and a continuous floor lane around it are both visible.  If the
  route is visible on the far side, a rear-side sector is valid; choose the side
  whose floor continuation is actually visible, not the side with the largest
  empty mask.
- APPROACH_LANDMARK with behind/beyond/towards: keep the landmark as the
  reference, but choose floor on the same outgoing bearing beyond/behind it;
  do not place the point on the landmark or on the near floor that would stop
  before it.  When the landmark blocks the central ray, use its visible side
  and the connected floor behind it rather than treating a frontal view as
  automatically correct.
- TRAVERSE/THROUGH plus diagonal/angled: a generic straight opening is not
  sufficient.  Select the sector whose portal/room aperture and connected floor
  continue along the stated diagonal; compare both side sectors and reject a
  view that only looks into the room without a traversable continuation.
- An explicit side-qualified portal such as "the door on the right/left" is
  a bearing constraint, not an invitation to force the forward view.  Prefer
  the forward half when it contains a connected floor-through-that-door ray;
  otherwise use the nearest matching side sector among +/-45, +/-90, and
  +/-135 degrees.  Never use the exact rear sector, and do not choose a side
  view solely because its detector box is larger than the visible doorway.
- FOLLOW_PATH_BOUNDARY or hallway-to-end: follow the continuous corridor to the
  stated endpoint/archway and keep the named wall/boundary on the instructed
  side.  The endpoint may be behind the current camera; do not discard a rear
  corridor merely because no word "turn" appears.  Reject a nearer side opening
  or a hallway that terminates at a wall before the named endpoint.
- VERTICAL_DOWN/UP or stairs: select only a view with repeated stair treads (or
  a clearly connected landing immediately beyond them) and a floor ray that
  continues in the requested vertical direction.  A visually salient hallway,
  statue, or landing without the correct stair continuation is not evidence.
- TURN_TO_LANDMARK (an explicit "turn towards/to <named landmark>") is a
  bearing instruction, not an ordinary approach or forward continuation. First
  identify the named landmark in clean RGB, then choose the compass sector and
  connected floor ray that point toward its visible bearing. Lateral and rear
  sectors are legal when the landmark is there; do not suppress them with a
  first-step forward prior. If the landmark is near an image edge, keep its
  bearing through a same-side floor anchor or request one local RGB refinement.
- For every choice, explicitly compare the best front, side, and rear legal
  sectors.  Ground segmentation only establishes where an anchor can be placed;
  it never decides the semantic direction.  Use RGB appearance and 2-D layout
  only; never use depth, navmesh, demonstration paths, or hidden state.
- When a next decomposed clause is provided and two lanes both satisfy the
  current relation, use its named room/landmark only as an outgoing-lane
  tie-breaker. Prefer the current-stage far-side lane whose visible continuation
  is compatible with that next context. Never skip the present boundary or call
  the next clause completed.
- For SELECT_PORTAL/ENTER/EXIT, the portal threshold is the primary current
  target. A destination object visible in the same room is not evidence for a
  portal unless the clean RGB shows that object framed beyond the opening and
  a connected floor ray crosses the threshold. For ordinal wording such as
  "next doorway", compare the full panorama; do not assume that "next" means
  straight or that it inherits the incoming bearing.
"""
                if (self.requested_point_selection_prompt_version in {
                        "v31_circumnavigate_forward_competitor",
                        "v34_unified_history_route_guard",
                        "v35_qualified_region_detector_grounding"} or
                        (form == "CIRCUMNAVIGATE" and bool(action_history))):
                    relation_route_guidance += """
- UNQUALIFIED CIRCUMNAVIGATION: “around/behind the couches” can begin with a
  short forward approach when the front view shows the same connected route.
  Compare that approach with any visible side-clearance lane and choose the
  route whose connected floor actually continues around the obstacle; do not
  force a lateral view when the obstacle has not blocked the front. Explicit
  left/right clauses still require the corresponding side sector. If prior
  action history is present, the side-clearance phase has already begun: a
  forward continuation along the same visible lane is preferred over repeating
  a lateral turn, unless clean RGB shows that the front lane is blocked.
- EXPLICIT FAR-SIDE CUE: when the clause or completion cue explicitly says
  “backside”, “behind”, “pass”, “far side”, or “beyond”, a near-side lane is
  only an intermediate approach and must not be selected as the completed
  route ray. Prefer the sector whose RGB shows the obstacle's far/rear boundary
  together with a connected floor lane continuing beyond it. If the front and
  side views are ambiguous, compare the two side sectors and select the one
  with visible far-edge/continuation evidence; do not infer the side from mask
  area, furniture salience, or a single cropped lane. This is a form-level RGB
  rule and does not use depth, navmesh, demonstration paths, or episode IDs.
"""
                if (self.requested_point_selection_prompt_version in {
                        "v33_stop_relation_near_side",
                        "v34_unified_history_route_guard"} and
                        form == "STOP_WAIT"):
                    relation_route_guidance += """
- SIDE-QUALIFIED STOP/WAIT NEAR-SIDE RULE (V33): for “stop/wait by the
  doorway/landmark on the left/right”, keep the named reference visible in a
  front-side oblique view and place the floor anchor on the near side of the
  threshold. Do not aim a straight ray through the opening into the named
  destination room: that is an overshoot and violates the requested near
  relation. Compare the +/-45-degree oblique ray with the +/-90-degree side
  view; prefer the oblique ray when it preserves the same doorway identity and
  has connected floor. This is an RGB/2-D relation rule and never uses depth,
  navmesh, reference paths, or execution outcomes.
"""
            stage2_route_guidance = ("""
STAGE-2 ROUTE-CONTINUITY REVIEW (V25):
This is a later sub-instruction reached from a real prior node. Preserve the
incoming-direction exclusion for genuine backtracking, but select the
instruction's next connected floor ray rather than the largest mask.
- STOP/WAIT at a corner or end: if no side word is present, prefer a
  forward/near-forward connected floor lane that reaches the named boundary;
  a pure +/-90 degree view is only valid when the corner is visibly lateral and
  no shallower lane reaches it.
- ADVANCE_STRAIGHT: prefer the longest visually continuous corridor/arch ray in
  the forward hemisphere; reject a wall-facing ray and do not use a rear ray
  merely because it has more green pixels.
- ENTER/TRAVERSE: choose the smallest legal angular offset that still shows the
  named portal and floor beyond it. If +/-90 and +/-135 show the same portal,
  prefer +/-90 unless the shallower view is visibly blocked.
- BETWEEN_OBJECTS: choose the floor lane through the named gap and along its
  outgoing corridor, not a point that only faces one object. Compare both sides
  and keep the lane connected beyond the pair.
- A selected pixel is one ray: center it on the connected route while keeping
  the semantic reference visible. These are generic RGB/2-D rules only; never
  use depth, navmesh, reference paths, or hidden outcomes.
""" if self._stage2_route_guard else "")
            post_vertical_transition = bool(
                ((stage.get("metadata", {}) or {}).get(
                    "post_vertical_stage_transition", {}) or {}).get(
                        "active"))
            post_vertical_guidance = ("""
POST-VERTICAL LANDING ROUTE RULE:
The previous verified stage ended on a different elevation. The next correct
level-floor route may overlap the incoming stair's horizontal camera bearing,
especially on a switchback, so that bearing is not automatically backtracking.
Inspect the full RGB panorama and choose the named level-floor continuation
(railing, room, corridor, or portal). Reject any ray whose visible route goes
down the stair flight just used. Do not prefer a generic forward opening over
the named continuation merely because the arrival camera faces away from it.
When the new clause says FOLLOW a railing/wall/boundary INTO a region, route
order matters: first choose the level-floor tangent that visibly continues
alongside that boundary. A nearby side room is not the route merely because it
already contains the requested room category. It may win only when the same
view shows the named boundary continuously leading through that room threshold;
otherwise keep following the boundary before entering. Compare the long
boundary-tangent view against the most salient immediate room view explicitly.
""" if post_vertical_transition else "")
            review_angle_requirement = (
                "within 30 degrees" if self._task30_route_anchor else
                "within 45 degrees")
            task30_anchor_guidance = ("""
TASK-30 RAY-CENTERING REQUIREMENT (V22):
- The final point ray, including its horizontal pixel offset inside the chosen
  camera, must stay within 30 degrees of the instruction-consistent route.
- Prefer the candidate whose connected route is centered in its RGB view.  A
  side sector with an edge-only floor ray is not acceptable when an adjacent
  sector shows the same route with a central floor lane.
- In the anchor pass choose the most central numbered floor anchor that still
  lies on the visible connected route; never use an extreme left/right anchor
  merely because its mask is larger.
- A high-confidence route review with a matching following-landmark view is a
  semantic commitment.  Do not replace it solely with a larger lower-floor
  mask from an opposite corridor.
- For VERTICAL_DOWN/UP, a level corridor is not a stair route.  Require raw RGB
  evidence of repeated treads, a stair rail with a descending landing, or a
  landing immediately connected to visible stairs.  The structured floor
  proposals below are supporting 2-D evidence only and must agree with RGB;
  never promote a corridor solely because its green mask is larger.
- For VERTICAL_UP/DOWN, when the stair flight enters from the left or right of
  a camera sector, keep the floor anchor on that same visible stair ray instead
  of blindly recentering it. The intended ray is the connected stair/landing
  continuation, not the largest side corridor.
- For TURN_TO_LANDMARK, center the final ray on the named landmark's image
  bearing. Centrality is only a tie-breaker: an off-center landmark requires a
  same-side floor anchor whose pixel offset preserves the landmark bearing.
  Never replace that bearing with generic central floor merely because it has
  a larger mask.
- For BETWEEN_OBJECTS, place the anchor on the visible outgoing gap/doorway
  side of the named pair.  If the gap is lateral in the image, a central mask
  centroid on the opposite side is not route-consistent; use the nearest
  connected floor anchor that points through the gap while remaining safely
  inside the mask.
- For CIRCUMNAVIGATE, treat the camera sector and pixel as one ray: a left
  sector uses a connected floor anchor on the image-left side, and a right
  sector uses an image-right anchor. An opposite-side anchor rotates a valid
  side view back toward the wrong branch and is not acceptable.
- For a complete CIRCUMNAVIGATE/around/backside clause, the first target is
  the far end of the visible connected lane, not the nearest floor beside the
  landmark. Prefer a numbered anchor with smaller image y (farther along the
  visible lane) and enough lateral offset to keep the compound ray on that
  lane. A near-side point that leaves the landmark in front or beside the
  agent is only partial progress and must not be presented as the endpoint.
- Camera sector and pixel offset are one navigation ray: when a selected
  +/-45-degree sector has a clearly lateral outgoing gap, use the same-side
  anchor (right side for a right sector, left side for a left sector) if it is
  connected floor.  Do not call an anchor centered merely because it is away
  from the image border.
- PASS/PAST AND LONG CONTINUATIONS: "straight" describes the connected route,
  not an obligation to keep the current camera yaw.  If the forward view is
  dominated by the landmark's near/interior area while an adjacent +/-45
  view shows the far opening or a continuous floor lane beyond that landmark,
  prefer the adjacent route-bearing view and its connected floor anchor.  Do
  not reject that view merely because it is a side sector or because the
  previous action history was mostly forward; the far-side opening is the
  semantic evidence for having passed the landmark.
""" if self._task30_route_anchor else "")
            circumnavigation_continuation_guidance = ("""
SUPPORTED-PARTIAL CIRCUMNAVIGATION RULE:
This is the single continuation of an already executed, real side-clearance
edge. Complete the SAME obstacle pass: keep the same couch/object instance at
the same side or rear-side edge of RGB and follow the floor around its far
extent. Do not select a direct view of the following room/landmark merely
because that later clause is salient; that is a stage skip when it cuts away
from the current obstacle. Prefer the legal sector whose clean RGB retains the
current obstacle boundary and continuous floor around it. The following clause
may break a tie only after those current-stage conditions are equal.
""" if (str(stage.get("form", "")).upper() == "CIRCUMNAVIGATE" and
           route_continuation) else "")
            review_prompt = f"""EIGHT_VIEW_GROUND_TARGET_REVIEW
You control an indoor R2R agent and must choose a direction {review_angle_requirement}
of the instruction-consistent route.  Eight simultaneous camera sectors are
centered every 45 degrees around the robot. In the Habitat camera/yaw
  convention used by this executor, physical TURN_LEFT actions increase yaw
  (positive relative-yaw gate) and TURN_RIGHT actions decrease yaw.  The
  rendered image's left/right pixel side is not itself a yaw sign; follow the
  explicit physical direction gate and the connected RGB floor lane.
Current stage: {stage['navigation_instruction']}
Landmark: {stage['landmark']}.
Semantic spatial target: {stage['semantic_spatial_target']}.
Required spatial relation: {stage['spatial_relation']}.
Visual arrival evidence: {stage['visual_arrival_evidence']}.
Forbidden target: {stage['forbidden_target']}.
Next decomposed clause (outgoing-route tie-breaker only; never the completion
target for this stage): {json.dumps(stage.get('next_sub_instruction_context'), ensure_ascii=False)}.
Remaining decomposed route context (ordered disambiguation evidence only;
never skip the current stage): {json.dumps(stage.get('following_sub_instruction_contexts', []), ensure_ascii=False)}.
Allowed existing views: {initial_allowed}.
Orientation/ground availability: {json.dumps(initial_orientation, ensure_ascii=False)}.
Explicit direction gate: {json.dumps(direction_gate, ensure_ascii=False)}.
V21 relation-direction gate: {json.dumps(relation_direction_gate, ensure_ascii=False)}.
Validated portal-floor relation gate: {json.dumps(relation_gate, ensure_ascii=False)}.
Detector proposals: {json.dumps({str(i): candidates[i].get('detection_records', []) for i in range(len(candidates))}, ensure_ascii=False)}.
Next-clause detector proposals (outgoing-lane tie-breaker only): {json.dumps({str(i): candidates[i].get('next_context_detection_records', []) for i in range(len(candidates))}, ensure_ascii=False)}.
Floor/stair proposal evidence (2-D RGB-aligned, fallible): {json.dumps(floor_proposal_evidence, ensure_ascii=False)}.
Compound route sequence active: {route_sequence_active}.
RGB evidence/refinement plan: {json.dumps(rgb_evidence_plan, ensure_ascii=False)}.

The RGB evidence plan's preferred_view_index identifies the strongest OBJECT
evidence and is not automatically a navigable answer. If that index is absent
from Allowed existing views, never return it: choose an allowed member of its
ground_bearing_neighborhood instead. The selected view must always contain a
usable ground anchor; object evidence may come from an adjacent sector.
If approximately the same open-vocabulary label is proposed across most of
the panorama, it is non-discriminative (often a wall, trim, panel, or typo
alias). Do not choose its largest box. Resolve the exact RGB identity together
with the ordered remaining route: prefer the current-stage bearing whose
visible continuation can execute the next portal/path/landmark sequence, while
still requiring current-stage evidence. This rule is lexical/visual only.

{"For a bare TURN_LEFT/TURN_RIGHT command with no named destination, all three side views remain legal, but a shallow +/-45-degree view is only a fallback. If a ground-bearing +/-90 or +/-135-degree view exists, prefer that committed turn; +/-45 means veering and must not win merely because it looks more open." if use_turn_commit_prior else ""}

{native_review_guidance}
{ground_surface_sanity_guidance}
{first_step_route_guidance}
{relation_route_guidance}
{stage2_route_guidance}
{post_vertical_guidance}
{circumnavigation_continuation_guidance}
First identify the actual landmark/portal/room and the route
relation.  Return use_existing when one allowed sector centers the intended
route clearly.  Return request_refined only when the intended route lies near
a boundary between sectors, the landmark/portal is clipped or occluded, or a
slightly shifted camera center is needed to verify identity and expose usable
ground.  For request_refined, view_index is the best existing fallback and
refined_relative_yaw_deg is the exact desired camera-center yaw, not a guess at
a pixel.  Never use a refined request to reopen a direction forbidden by an
explicit direction or validated relation gate.  Do not use floor area alone.
ROUTE-SEQUENCE GROUNDING: when the stage contains a sequence such as exit A,
pass B, then approach C, select the direction for the earliest not-yet-executed
physical transition. Later landmarks disambiguate that route but are not a
license to jump directly to a different obvious doorway. Cite only landmarks
actually visible in RGB. Compare the chosen view against the strongest allowed
competitor and state the visible evidence that rejects it; detector text without
matching RGB appearance is not evidence. If a first following landmark is
visible, return its view index independently and require the current transition
to share that view or an immediately adjacent 45-degree view when such a legal
ground-bearing view exists. The following landmark is a tie-breaker only: if its
sector has no legal current ground ray, retain the best current-stage transition
instead of declaring that no route exists. Use -1/not_visible only when the
landmark truly cannot be identified in any clean RGB view.
No depth, distance-map, 3D position, navmesh, or demonstration-path information
is available or permitted in this decision.
{dual_review_guidance}
{task30_anchor_guidance}
Return only requested JSON."""

            def validate_review(result):
                decision = str(result["decision"])
                model_view_index = int(result["view_index"])
                view_index = model_view_index
                ground_fallback_adjustment = None
                if view_index not in initial_allowed:
                    if not 0 <= view_index < len(candidates):
                        raise ValueError(
                            f"view_index must be one of {initial_allowed}")
                    model_yaw = signed_relative_yaw_deg(
                        candidates[view_index])

                    def model_delta(candidate_index):
                        return abs(math.degrees((math.radians(
                            signed_relative_yaw_deg(
                                candidates[candidate_index]) - model_yaw) +
                            math.pi) % (2 * math.pi) - math.pi))

                    adjacent_allowed = [
                        index for index in initial_allowed
                        if model_delta(index) <= 45.0001]
                    if not adjacent_allowed:
                        raise ValueError(
                            f"view_index must be one of {initial_allowed}")
                    competitor = int(result.get(
                        "strongest_competing_view_index", -1))
                    if competitor in adjacent_allowed:
                        view_index = competitor
                    else:
                        view_index = min(adjacent_allowed, key=model_delta)
                    decision = "use_existing"
                    ground_fallback_adjustment = (
                        "VLM selected a sector without a permitted ground "
                        "anchor; used its allowed <=45-degree ground-bearing "
                        "neighbor without relaxing direction/relation gates")
                refined_deg = float(result["refined_relative_yaw_deg"])
                if not math.isfinite(refined_deg) or abs(refined_deg) > 360:
                    raise ValueError(
                        "refined_relative_yaw_deg must be finite in [-360,360]")
                refined_deg = (refined_deg + 180.0) % 360.0 - 180.0
                if abs(refined_deg + 180.0) < 1e-6:
                    refined_deg = 180.0
                cleaned = {
                    "decision": decision, "view_index": view_index,
                    "refined_relative_yaw_deg": refined_deg,
                    "reason": str(result.get("reason", "")),
                }
                if ground_fallback_adjustment is not None:
                    cleaned.update({
                        "model_view_index": model_view_index,
                        "ground_fallback_adjustment":
                            ground_fallback_adjustment,
                    })
                if dual_candidate_review:
                    alternative = int(result["alternative_view_index"])
                    if alternative not in initial_allowed:
                        raise ValueError(
                            "alternative_view_index must be an allowed view")
                    if (len(initial_allowed) > 1 and
                            alternative == view_index):
                        raise ValueError(
                            "primary and alternative views must differ")
                    confidence = float(result["confidence"])
                    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                        raise ValueError("confidence must be finite in [0,1]")
                    cleaned.update({
                        "alternative_view_index": alternative,
                        "confidence": confidence,
                    })
                if native_rgb_review:
                    target_centering = str(result["target_centering"])
                    if target_centering not in {
                            "centered", "left_edge", "right_edge",
                            "uncertain"}:
                        raise ValueError("invalid target_centering")
                    confidence = float(result["confidence"])
                    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                        raise ValueError("confidence must be finite in [0,1]")
                    cleaned.update({
                        "target_centering": target_centering,
                        "confidence": confidence,
                        "strongest_competing_view_index": int(
                            result["strongest_competing_view_index"]),
                        "visible_route_evidence": str(
                            result.get("visible_route_evidence", "")),
                        "competitor_rejection": str(
                            result.get("competitor_rejection", "")),
                        "first_following_landmark": str(
                            result.get("first_following_landmark", "")),
                        "first_following_landmark_view_index": int(
                            result["first_following_landmark_view_index"]),
                        "sequence_alignment": str(
                            result["sequence_alignment"]),
                    })
                    competitor = cleaned["strongest_competing_view_index"]
                    if (ground_fallback_adjustment is not None and
                            competitor == view_index):
                        replacement_competitors = [
                            index for index in initial_allowed
                            if index != view_index]
                        if replacement_competitors:
                            cleaned[
                                "model_strongest_competing_view_index"] = (
                                    competitor)
                            competitor = min(
                                replacement_competitors,
                                key=lambda index: abs(math.degrees((
                                    math.radians(signed_relative_yaw_deg(
                                        candidates[index]) -
                                        signed_relative_yaw_deg(
                                            candidates[view_index])) +
                                    math.pi) % (2 * math.pi) - math.pi)))
                            cleaned[
                                "strongest_competing_view_index"] = competitor
                    if competitor not in initial_allowed:
                        # Vision models occasionally name a visually salient
                        # but mask-free panorama sector as the competitor.  A
                        # competitor is audit metadata, not a navigation
                        # decision; for V21 normalize it to the nearest legal
                        # sector instead of wasting all retries on a harmless
                        # bookkeeping error.  Other prompt versions retain
                        # their strict historical validator.
                        if self._relation_route_guard:
                            replacement = [
                                index for index in initial_allowed
                                if index != view_index]
                            if not replacement:
                                raise ValueError(
                                    "strongest competitor must be an allowed view")
                            cleaned["model_strongest_competing_view_index"] = (
                                competitor)
                            competitor = min(
                                replacement,
                                key=lambda index: abs(math.degrees((
                                    math.radians(signed_relative_yaw_deg(
                                        candidates[index]) -
                                        signed_relative_yaw_deg(
                                            candidates[view_index])) +
                                    math.pi) % (2 * math.pi) - math.pi)))
                            cleaned["strongest_competing_view_index"] = competitor
                        else:
                            raise ValueError(
                                "strongest competitor must be an allowed view")
                    if len(initial_allowed) > 1 and competitor == view_index:
                        raise ValueError(
                            "strongest competitor must differ from selected view")
                    landmark_view = cleaned[
                        "first_following_landmark_view_index"]
                    sequence_alignment = cleaned["sequence_alignment"]
                    if landmark_view != -1 and not 0 <= landmark_view < len(candidates):
                        raise ValueError(
                            "following landmark view must be a panorama view or -1")
                    if landmark_view == -1 and sequence_alignment not in {
                            "not_visible", "not_applicable"}:
                        raise ValueError(
                            "missing following landmark requires not_visible or "
                            "not_applicable")
                    if (route_sequence_active and landmark_view != -1 and
                            abs(math.degrees((math.radians(
                                signed_relative_yaw_deg(candidates[view_index]) -
                                signed_relative_yaw_deg(candidates[landmark_view])) +
                                math.pi) % (2 * math.pi) - math.pi)) > 45.0):
                        sequence_consistent = sorted(
                            initial_allowed,
                            key=lambda index: abs(math.degrees((math.radians(
                                signed_relative_yaw_deg(candidates[index]) -
                                signed_relative_yaw_deg(candidates[landmark_view])) +
                                math.pi) % (2 * math.pi) - math.pi)))
                        corrected_view = sequence_consistent[0]
                        corrected_error = abs(math.degrees((math.radians(
                            signed_relative_yaw_deg(candidates[corrected_view]) -
                            signed_relative_yaw_deg(candidates[landmark_view])) +
                            math.pi) % (2 * math.pi) - math.pi))
                        current_form = str(stage.get("form", "")).upper()
                        object_grounded_form = current_form in {
                            "TURN_TO_LANDMARK", "PASS_LANDMARK",
                            "APPROACH_LANDMARK", "BETWEEN_OBJECTS",
                            "CIRCUMNAVIGATE",
                        }
                        current_landmark_tokens = {
                            word for word in re.findall(
                                r"[a-z]+", str(stage.get(
                                    "landmark", "")).lower())
                            if len(word) > 2 and word not in {
                                "the", "and", "near", "between", "object",
                                "large", "small", "wooden", "white", "black",
                                "brown", "red", "blue", "green", "gray",
                                "grey", "long", "short",
                            }
                        }

                        def current_landmark_visible_near(candidate_index):
                            nearby = [
                                index for index in initial_allowed
                                if abs(math.degrees((math.radians(
                                    signed_relative_yaw_deg(candidates[index]) -
                                    signed_relative_yaw_deg(
                                        candidates[candidate_index])) +
                                    math.pi) % (2 * math.pi) - math.pi)) <= 45.0
                            ]
                            for index in nearby:
                                for record in candidates[index].get(
                                        "detection_records", []):
                                    record_tokens = set(re.findall(
                                        r"[a-z]+", str(record.get(
                                            "label", "")).lower()))
                                    record_tokens.update(
                                        str(token).lower() for token in
                                        (record.get("matched_tokens", []) or []))
                                    if current_landmark_tokens.intersection(
                                            record_tokens):
                                        return True
                            return False

                        current_transition_retained = bool(
                            not object_grounded_form or
                            not current_landmark_tokens or
                            current_landmark_visible_near(corrected_view))
                        if corrected_error <= 45.0:
                            # Future landmarks are useful evidence inside a
                            # current portal/route view, but they are not an
                            # executable target for this stage.  Earlier code
                            # deterministically replaced the current semantic
                            # decision with the nearest view to a claimed
                            # future landmark.  A hallucinated or independently
                            # visible later landmark could therefore jump to a
                            # different door. Keep the VLM's current-stage view
                            # and record the mismatch for audit/prompt context.
                            cleaned["following_landmark_tiebreak_rejected"] = {
                                "landmark_view_index": landmark_view,
                                "candidate_view_index": corrected_view,
                                "current_transition_rgb_supported": bool(
                                    current_transition_retained),
                                "policy": (
                                    "later-clause bearing never overrides the "
                                    "current-stage semantic direction; it is "
                                    "advisory evidence only"),
                            }
                        else:
                            # A later clause is route context, never the
                            # current execution target.  When its observed
                            # sector has no adjacent legal ground ray, retain
                            # the independently grounded current transition.
                            # Failing/retrying here previously made a valid
                            # current-stage portal or stair route impossible.
                            cleaned[
                                "following_landmark_ground_unavailable"] = {
                                    "landmark_view_index": landmark_view,
                                    "nearest_legal_view_index": corrected_view,
                                    "nearest_angular_error_deg": round(
                                        float(corrected_error), 3),
                                    "policy": (
                                        "future clause is tie-break context; "
                                        "retain current-stage route"),
                                }
                    if decision == "use_existing" and target_centering != "centered":
                        cleaned["model_decision"] = decision
                        cleaned["decision"] = "request_refined"
                        cleaned["decision_adjustment"] = (
                            "non-centered or uncertain target requires one "
                            "local RGB refinement")
                return cleaned

            review_images = images
            if native_rgb_review:
                review_images = [*raw_views, images[0], annotated_sheet]
            if semantic_reference_rgb is not None:
                semantic_reference_rgb = np.asarray(
                    semantic_reference_rgb, np.uint8)
                if (semantic_reference_rgb.ndim != 3 or
                        semantic_reference_rgb.shape[2] != 3):
                    raise ValueError(
                        "semantic reference must be one clean RGB image")
                review_images = [*review_images, semantic_reference_rgb]
                review_prompt += """

SAME_PORTAL_IDENTITY_REFERENCE:
The FINAL input image is a clean RGB view of the doorway/opening selected on
the first real hop of this same active portal-crossing instruction. It is identity
memory, not a desired camera bearing and not proof of completion. Choose the
current panorama view that continues through that SAME architectural portal,
using its frame, adjacent walls/trim, connected room appearance, and ordered
following-landmark context. Do not select a different generic doorway merely
because it is currently frontal or has a larger floor mask. The camera may
have moved and rotated, so the same portal may now be lateral or rear-side.
No depth, navmesh, demonstration trajectory, goal geometry, or episode label
is contained in this reference.
"""
            review = self._call(
                "review_eight_view_ground_target", review_prompt, review_images,
                review_schema, validate_review)
            # When two portal views both satisfy the current EXIT/ENTER
            # relation, the immediately following vertical landmark is useful
            # solely to identify which *current portal* begins the ordered
            # route.  DINO+SAM stair proposals live in the generic ground
            # evidence channel because stairs are walkable surfaces, so the
            # ordinary next-landmark proposal table can be empty.  Resolve a
            # primary/competitor ambiguity only when the competitor itself has
            # current portal evidence and is materially closer on the circular
            # panorama to the strongest stair proposal.  This never executes
            # or completes the later stair clause.
            next_context = stage.get("next_sub_instruction_context") or {}
            current_form = str(stage.get("form", "")).upper()
            next_form = str(next_context.get("form", "")).upper()
            if (route_sequence_active and
                    current_form in {"EXIT_REGION", "ENTER_REGION",
                                     "SELECT_PORTAL",
                                     "TRAVERSE_PORTAL_REGION"} and
                    next_form in {"VERTICAL_UP", "VERTICAL_DOWN"} and
                    len(initial_allowed) > 1):
                primary = int(review["view_index"])
                competitor = int(review.get(
                    "strongest_competing_view_index", -1))

                def circular_view_delta(a, b):
                    return abs((signed_relative_yaw_deg(candidates[a]) -
                                signed_relative_yaw_deg(candidates[b]) +
                                180.0) % 360.0 - 180.0)

                def has_portal_evidence(index):
                    portal_words = {
                        "door", "doorway", "opening", "hallway", "portal"}
                    for record in candidates[index].get(
                            "detection_records", []):
                        tokens = set(re.findall(
                            r"[a-z]+", str(record.get("label", "")).lower()))
                        if portal_words.intersection(tokens):
                            return True
                    return False

                stair_scores = []
                for index in initial_allowed:
                    best = max((
                        float(record.get("score", 0.0) or 0.0)
                        for record in candidates[index].get(
                            "ground_detection_records", [])
                        if any(token.startswith("stair") or token == "steps"
                               for token in re.findall(
                                   r"[a-z]+", str(record.get(
                                       "label", "")).lower()))
                    ), default=0.0)
                    if best >= 0.50:
                        stair_scores.append((best, index))
                if (competitor in initial_allowed and competitor != primary and
                        has_portal_evidence(competitor) and stair_scores):
                    _, stair_view = max(stair_scores)
                    primary_delta = circular_view_delta(primary, stair_view)
                    competitor_delta = circular_view_delta(
                        competitor, stair_view)
                    if (competitor_delta <= 90.0001 and
                            primary_delta - competitor_delta >= 45.0):
                        review["model_view_index"] = primary
                        review["view_index"] = competitor
                        review["decision"] = "use_existing"
                        review["target_centering"] = "centered"
                        review["route_sequence_portal_adjudication"] = {
                            "active": True,
                            "original_view_index": primary,
                            "adjudicated_view_index": competitor,
                            "following_vertical_view_index": stair_view,
                            "original_to_following_deg": round(
                                float(primary_delta), 3),
                            "adjudicated_to_following_deg": round(
                                float(competitor_delta), 3),
                            "policy": (
                                "later stair evidence disambiguates two current "
                                "portal candidates without skipping the portal"),
                        }
            following_contexts = list(stage.get(
                "following_sub_instruction_contexts", []) or [])
            current_detection_views = [
                index for index in initial_allowed
                if any(float(record.get("score", 0.0) or 0.0) >= 0.28
                       for record in candidates[index].get(
                           "detection_records", []))
            ]
            ambiguous_turn_landmark = bool(
                str(stage.get("form", "")).upper() == "TURN_TO_LANDMARK" and
                len(following_contexts) >= 2 and
                len(current_detection_views) >= max(
                    5, int(math.ceil(0.625 * len(candidates)))))
            if ambiguous_turn_landmark:
                # When an open-vocabulary alias fires in most compass sectors,
                # its largest box cannot identify the intended instance. Ask
                # a separate RGB-only pass to resolve the current bearing by
                # the ordered visible route (e.g. portal then path boundary),
                # without exposing depth, navmesh, a reference path, or the
                # first model's prose as ground truth.
                ambiguity_allowed = [
                    index for index in initial_allowed
                    if abs(signed_relative_yaw_deg(candidates[index])) > 30.0]
                if not ambiguity_allowed:
                    ambiguity_allowed = list(initial_allowed)
                spelling_aliases = ambiguous_landmark_spelling_aliases(
                    stage.get("landmark", ""), {
                        phrase for candidate in candidates
                        for phrase in candidate.get(
                            "detection_queries", [])})
                ambiguity_schema = {
                    "type": "object", "additionalProperties": False,
                    "required": [
                        "view_index", "reason", "current_landmark_evidence",
                        "following_route_evidence"],
                    "properties": {
                        "view_index": {"type": "integer",
                                       "enum": ambiguity_allowed},
                        "reason": {"type": "string", "maxLength": 420},
                        "current_landmark_evidence": {
                            "type": "string", "maxLength": 320},
                        "following_route_evidence": {
                            "type": "string", "maxLength": 320},
                    },
                }
                ambiguity_prompt = f"""AMBIGUOUS_LANDMARK_ROUTE_ADJUDICATION
Choose the compass view for the first executable route of this ordered R2R
instruction fragment.
Current stage: {stage.get('navigation_instruction', '')}
Current landmark: {stage.get('landmark', '')}
Remaining ordered route: {json.dumps(following_contexts, ensure_ascii=False)}
Legal ground-bearing views after the explicit-turn action gate:
{ambiguity_allowed}. A view within 30 degrees of the current forward heading
cannot satisfy the verb "turn" and has been removed.

The current open-vocabulary label appeared in {current_detection_views}, most
of the panorama, so its boxes are explicitly non-discriminative. Inspect the
clean RGB views. Identify the intended current landmark/bearing together with
the directly connected doorway/path/landmark sequence that follows. Do not
choose a closer or larger lookalike when its bearing cannot execute that
ordered continuation. The current stage remains the completion target; later
clauses only disambiguate its bearing. Use no depth, navmesh, demonstration
path, episode identity, or hidden outcome. Return JSON only."""
                if spelling_aliases:
                    ambiguity_prompt += """
The landmark token has no unique exact visual-noun match. Detector query
expansions are proposal-recall aids only and must not silently redefine that
token as a familiar salient object. Ground the intended bearing jointly from
the unknown token's visible candidate and the complete ordered route.
"""

                def validate_ambiguity(result):
                    index = int(result["view_index"])
                    if index not in ambiguity_allowed:
                        raise ValueError(
                            f"view_index must be one of {ambiguity_allowed}")
                    return {
                        "view_index": index,
                        "reason": str(result.get("reason", "")),
                        "current_landmark_evidence": str(result.get(
                            "current_landmark_evidence", "")),
                        "following_route_evidence": str(result.get(
                            "following_route_evidence", "")),
                    }

                ambiguity = self._call(
                    "adjudicate_ambiguous_landmark_route", ambiguity_prompt,
                    review_images, ambiguity_schema, validate_ambiguity)
                original_index = int(review["view_index"])
                review["model_view_index"] = original_index
                review["view_index"] = int(ambiguity["view_index"])
                # For panorama-wide aliases, this yaw represents the complete
                # executable route ray.  Re-centering the ambiguous noun, or
                # replacing the VLM result with a larger floor crop, can turn
                # away from that route and is therefore deliberately disabled.
                review["decision"] = "use_existing"
                review["target_centering"] = "centered"
                review["reason"] = ambiguity["reason"]
                review["visible_route_evidence"] = ambiguity[
                    "following_route_evidence"]
                if int(review.get(
                        "strongest_competing_view_index", -1)) == int(
                            review["view_index"]):
                    review["strongest_competing_view_index"] = original_index
                review["ambiguous_landmark_route_adjudication"] = {
                    "active": True,
                    "detector_supported_view_indices": (
                        current_detection_views),
                    "explicit_turn_allowed_view_indices": ambiguity_allowed,
                    "original_view_index": original_index,
                    "lexical_ambiguity_diagnostic": spelling_aliases,
                    **ambiguity,
                    "policy": (
                        "panorama-wide detector alias is non-discriminative; "
                        "the current landmark must be visible and the full "
                        "ordered clean-RGB route resolves its instance and "
                        "bearing"),
                }
            if (self._relation_route_guard and
                    str(stage.get("form", "")) == "CIRCUMNAVIGATE" and
                    not (action_history or []) and
                    len(initial_allowed) > 1):
                # Around/backside lanes often straddle the 45/90-degree
                # panorama boundary: one RGB view shows the obstacle edge and
                # its adjacent same-side competitor shows the continuation.
                # When the VLM itself ranks that adjacent pair first and
                # second, acquire their midpoint as a real RGB view. This uses
                # no depth, navmesh, demonstration, scene ID, or outcome.
                primary = int(review["view_index"])
                competitor = int(review.get(
                    "strongest_competing_view_index", -1))
                if competitor in initial_allowed and competitor != primary:
                    primary_yaw = signed_relative_yaw_deg(candidates[primary])
                    competitor_yaw = signed_relative_yaw_deg(
                        candidates[competitor])
                    delta = ((competitor_yaw - primary_yaw + 180.0) %
                             360.0 - 180.0)
                    same_side = bool(
                        primary_yaw * competitor_yaw > 0.0 and
                        30.0 <= abs(primary_yaw) <= 100.0 and
                        30.0 <= abs(competitor_yaw) <= 100.0)
                    if same_side and abs(delta) <= 45.0001:
                        midpoint_yaw = primary_yaw + 0.5 * delta
                        review["decision"] = "request_refined"
                        review["refined_relative_yaw_deg"] = midpoint_yaw
                        review["target_centering"] = "uncertain"
                        review["circumnavigation_boundary_refinement"] = {
                            "primary_view_index": primary,
                            "competitor_view_index": competitor,
                            "refined_relative_yaw_deg": round(
                                float(midpoint_yaw), 3),
                            "policy": "same-side adjacent RGB route midpoint",
                        }
                if not review.get("circumnavigation_boundary_refinement"):
                    # The model can select a +/-90 view yet name FRONT as the
                    # formal competitor even while its prose and RGB show the
                    # usable lane spanning the adjacent +/-45 sector. For an
                    # unqualified around/backside clause, acquire that 67.5°
                    # midpoint deterministically. Explicit left/right routes
                    # retain their exact commanded sector.
                    primary_yaw = signed_relative_yaw_deg(
                        candidates[primary])
                    stage_text = " ".join(str(stage.get(key, "")) for key in (
                        "navigation_instruction", "spatial_relation",
                        "semantic_spatial_target")).lower()
                    explicit_side = bool(re.search(
                        r"\b(?:left|right)\b", stage_text))
                    if (not explicit_side and
                            80.0 <= abs(primary_yaw) <= 100.0):
                        desired_shallow = 45.0 if primary_yaw > 0.0 else -45.0
                        shallow = [index for index in initial_allowed
                                   if abs(signed_relative_yaw_deg(
                                       candidates[index]) -
                                          desired_shallow) <= 1e-6]
                        if shallow:
                            midpoint_yaw = 0.5 * (
                                primary_yaw + desired_shallow)
                            review["decision"] = "request_refined"
                            review["refined_relative_yaw_deg"] = midpoint_yaw
                            review["target_centering"] = "uncertain"
                            review["circumnavigation_boundary_refinement"] = {
                                "primary_view_index": primary,
                                "competitor_view_index": int(shallow[0]),
                                "refined_relative_yaw_deg": round(
                                    float(midpoint_yaw), 3),
                                "policy": (
                                    "unqualified right-angle around lane uses "
                                    "its same-side 45/90 RGB midpoint"),
                            }
            if (self._relation_route_guard and route_continuation and
                    str(stage.get("form", "")) == "CIRCUMNAVIGATE" and
                    0 in initial_allowed):
                # On the second around-obstacle hop, a selected +/-45 sector
                # and the current forward sector often see the same curved
                # far-edge lane. Acquire their +/-22.5 midpoint so an
                # eight-view boundary does not turn a valid continuation into
                # a >30-degree execution. This remains inside the persisted
                # 60-degree route-integrity cone and uses only two already
                # legal RGB-ground views.
                primary = int(review["view_index"])
                primary_yaw = signed_relative_yaw_deg(candidates[primary])
                if 40.0 <= abs(primary_yaw) <= 50.0:
                    midpoint_yaw = 0.5 * primary_yaw
                    review["decision"] = "request_refined"
                    review["refined_relative_yaw_deg"] = midpoint_yaw
                    review["target_centering"] = "uncertain"
                    review[
                        "circumnavigation_continuation_midpoint_refinement"] = {
                            "primary_view_index": primary,
                            "forward_view_index": 0,
                            "refined_relative_yaw_deg": round(
                                float(midpoint_yaw), 3),
                            "policy": (
                                "same-stage curved far-edge lane uses the "
                                "forward/45-degree RGB midpoint"),
                        }
            if (self._stage1_turn_soft and
                    str(stage.get("form", "")) == "PASS_LANDMARK" and
                    len(initial_allowed) > 1 and
                    abs(signed_relative_yaw_deg(
                        candidates[int(review["view_index"])]) ) <= 30.0):
                # A pass/beyond clause is especially sensitive to a visually
                # plausible forward ray that stops inside the landmark.  Use a
                # small, independent RGB adjudication over the already legal
                # sectors.  It is deliberately semantic (no depth, navmesh,
                # demo path, or post-run outcome) and applies to every
                # PASS_LANDMARK stage rather than to a particular scene.
                pass_schema = {
                    "type": "object", "additionalProperties": False,
                    "required": ["view_index", "reason"],
                    "properties": {
                        "view_index": {"type": "integer",
                                       "enum": initial_allowed},
                        "reason": {"type": "string", "maxLength": 320},
                    },
                }
                pass_prompt = f"""PASS_LANDMARK_ROUTE_ADJUDICATION
Choose the legal RGB view whose connected floor ray actually goes beyond the
named landmark, not merely toward its near/interior area.
Stage: {stage.get('navigation_instruction', '')}
Landmark: {stage.get('landmark', '')}
Required relation: {stage.get('spatial_relation', '')}
Legal views: {initial_allowed}
Initial semantic proposal: {int(review.get('view_index'))}
Initial strongest competitor: {int(review.get('strongest_competing_view_index', -1))}
Orientation/ground availability: {json.dumps(initial_orientation, ensure_ascii=False)}
Compare the clean RGB views and their aligned 2-D ground overlays.  If the
forward view is dominated by the landmark or its near side, an adjacent 45
degree view with a visible far opening/continuous floor is the better route,
even when the instruction says straight.  If the forward view clearly shows
the floor beyond the landmark and side views are only dead ends, retain it.
Use only RGB appearance, 2-D floor masks and anchor geometry.  Do not use
depth, navmesh, demonstration paths, or hidden execution outcomes. Return only
the selected legal view and a concise reason."""

                def validate_pass(result):
                    index = int(result["view_index"])
                    if index not in initial_allowed:
                        raise ValueError(
                            f"view_index must be one of {initial_allowed}")
                    return index, str(result.get("reason", ""))

                adjudicated_index, adjudicated_reason = self._call(
                    "adjudicate_pass_landmark_route", pass_prompt,
                    review_images, pass_schema, validate_pass)
                original_index = int(review["view_index"])
                review["pass_adjudication"] = {
                    "original_view_index": original_index,
                    "adjudicated_view_index": int(adjudicated_index),
                    "reason": adjudicated_reason,
                }
                if adjudicated_index != original_index:
                    review["model_view_index"] = original_index
                    review["view_index"] = int(adjudicated_index)
                    # The semantic side-sector decision can sit on a
                    # 45-degree panorama boundary while its available floor
                    # mask is image-edge biased.  Ask for one local RGB view
                    # between the original and adjudicated sectors; this
                    # preserves the route choice while giving the anchor pass
                    # a centered, connected floor ray.  The midpoint rule is
                    # generic and does not use depth, navmesh, or a demo path.
                    original_yaw = signed_relative_yaw_deg(
                        candidates[original_index])
                    adjudicated_yaw = signed_relative_yaw_deg(
                        candidates[adjudicated_index])
                    delta = (adjudicated_yaw - original_yaw + 180.0) % 360.0 - 180.0
                    review["decision"] = "request_refined"
                    review["refined_relative_yaw_deg"] = (
                        original_yaw + float(delta) * 0.5)
                    review["target_centering"] = "uncertain"
                    review["visible_route_evidence"] = adjudicated_reason
                    review["competitor_rejection"] = (
                        "independent pass/beyond adjudication preferred the "
                        "route-bearing view")
                    review["strongest_competing_view_index"] = original_index
            if (first_step_route_guard_active and
                    self.requested_point_selection_prompt_version in {
                        "v32_first_stage_shallow_route",
                        "v34_unified_history_route_guard"} and
                    len(initial_allowed) > 1 and
                    not explicit_reverse_command and
                    str(stage.get("form", "")) in {
                        "ENTER_REGION", "EXIT_REGION", "PASS_LANDMARK",
                        "ADVANCE_STRAIGHT", "CROSS_SPACE",
                        "TRAVERSE_PORTAL_REGION", "SELECT_PORTAL"}):
                # A side-facing semantic proposal can be valid but still be a
                # poor first ray when the same destination is visibly reachable
                # from a near-forward sector.  Ask a second RGB-only
                # adjudicator to compare the two route hypotheses.  This is a
                # form-level safeguard; it does not assume that forward is
                # correct when the front view is genuinely blocked.
                current_index = int(review["view_index"])
                current_rel = signed_relative_yaw_deg(candidates[current_index])
                shallow = [index for index in initial_allowed
                           if abs(signed_relative_yaw_deg(candidates[index])) <= 45.0]
                shallow = [index for index in shallow
                           if candidates[index].get("ground_anchors")]
                if shallow and abs(current_rel) > 45.0:
                    shallow_schema = {
                        "type": "object", "additionalProperties": False,
                        "required": ["view_index", "reason"],
                        "properties": {
                            "view_index": {"type": "integer",
                                           "enum": initial_allowed},
                            "reason": {"type": "string", "maxLength": 320},
                        },
                    }
                    shallow_prompt = f"""FIRST_STAGE_SHALLOW_ROUTE_ADJUDICATION
Choose the legal RGB view whose connected floor ray executes the first
sub-instruction.  The current proposal is side-facing, so explicitly compare
it with the near-forward candidates.  Prefer a +/-45-degree (or smaller) ray
when it shows the same named room/portal/landmark and a continuous route.  Keep
the side proposal only if every near-forward view is visibly blocked, ends at a
wall/object, or lacks the named destination.
Instruction: {stage.get('navigation_instruction', '')}
Form: {stage.get('form', '')}
Semantic target: {stage.get('semantic_spatial_target', '')}
Current proposal view: {current_index}
Near-forward legal candidates: {shallow}
All legal views: {initial_allowed}
Use clean RGB and aligned 2-D ground masks only; no depth, navmesh,
demonstration path, or execution outcome. Return one legal view and a concise
reason."""

                    def validate_shallow(result):
                        index = int(result["view_index"])
                        if index not in initial_allowed:
                            raise ValueError(
                                f"view_index must be one of {initial_allowed}")
                        return index, str(result.get("reason", ""))

                    shallow_index, shallow_reason = self._call(
                        "adjudicate_first_stage_shallow_route", shallow_prompt,
                        review_images, shallow_schema, validate_shallow)
                    review["first_stage_shallow_adjudication"] = {
                        "original_view_index": current_index,
                        "adjudicated_view_index": int(shallow_index),
                        "near_forward_candidates": shallow,
                        "reason": shallow_reason,
                    }
                    if shallow_index != current_index:
                        review["model_view_index"] = current_index
                        review["view_index"] = int(shallow_index)
                        review["decision"] = "use_existing"
                        review["refined_relative_yaw_deg"] = (
                            signed_relative_yaw_deg(candidates[shallow_index]))
                        review["target_centering"] = "centered"
                        review["visible_route_evidence"] = shallow_reason
                        review["competitor_rejection"] = (
                            "independent first-stage shallow-route adjudication "
                            "selected the connected near-forward ray")
            side_stop_near_side = bool(
                self.requested_point_selection_prompt_version in {
                    "v33_stop_relation_near_side",
                    "v34_unified_history_route_guard"} and
                str(stage.get("form", "")) == "STOP_WAIT" and
                re.search(r"\b(left|right)\b", str(stage.get(
                    "navigation_instruction", "")).lower()))
            if (self._stage2_route_guard and len(initial_allowed) > 1 and
                    str(stage.get("form", "")) in {
                        "STOP_WAIT", "ADVANCE_STRAIGHT", "ENTER_REGION",
                        "EXIT_REGION", "TRAVERSE_PORTAL_REGION",
                        "SELECT_PORTAL", "BETWEEN_OBJECTS", "CIRCUMNAVIGATE",
                        "PASS_LANDMARK", "FOLLOW_PATH_BOUNDARY",
                        "APPROACH_LANDMARK"} and
                    (str(stage.get("form", "")) != "CIRCUMNAVIGATE" or
                     bool(action_history)) and
                    not side_stop_near_side and
                    (self.requested_point_selection_prompt_version !=
                     "v31_circumnavigate_forward_competitor" or
                     str(stage.get("form", "")) in {
                         "ENTER_REGION", "EXIT_REGION",
                         "TRAVERSE_PORTAL_REGION", "SELECT_PORTAL"})):
                # Later-stage clauses are easily satisfied by a salient side
                # object while the connected route is elsewhere.  A second
                # independent RGB comparison is deliberately restricted to
                # the legal views and has no access to depth/navmesh/demo
                # data.  It is a form-level route continuity check.
                route_schema = {
                    "type": "object", "additionalProperties": False,
                    "required": ["view_index", "reason"],
                    "properties": {
                        "view_index": {"type": "integer", "enum": initial_allowed},
                        "reason": {"type": "string", "maxLength": 320},
                    },
                }
                route_prompt = f"""STAGE2_ROUTE_CONTINUITY_ADJUDICATION
Choose the legal RGB view whose connected walkable floor ray best executes the
active sub-instruction from the current node.  This is a semantic route check,
not a detector-area contest.
Instruction: {stage.get('navigation_instruction', '')}
Form: {stage.get('form', '')}
Landmark/target: {stage.get('landmark', '')} / {stage.get('semantic_spatial_target', '')}
Legal views: {initial_allowed}
Initial proposal: {int(review.get('view_index'))}
Rules: for STOP/WAIT corner or end without a side word, prefer the connected
forward or near-forward lane over a pure side-facing object; for
ADVANCE_STRAIGHT prefer a long forward corridor/arch and reject a wall-facing
ray; for ENTER/TRAVERSE choose the smallest legal offset that still shows the
named portal and floor beyond; for BETWEEN_OBJECTS choose the lane through the
gap and along its outgoing corridor; for an unqualified CIRCUMNAVIGATE or
PASS/FOLLOW continuation with action history, prefer the connected lane that
continues the already-begun side transition rather than repeating a lateral
turn.  Never reopen a blocked/incoming direction
and never use depth, navmesh, reference paths, or execution outcomes.
Compare clean RGB and aligned 2-D ground overlays, then return one legal view
and a concise reason."""

                def validate_route(result):
                    index = int(result["view_index"])
                    if index not in initial_allowed:
                        raise ValueError(f"view_index must be one of {initial_allowed}")
                    return index, str(result.get("reason", ""))

                route_index, route_reason = self._call(
                    "adjudicate_stage2_route_continuity", route_prompt,
                    review_images, route_schema, validate_route)
                route_original = int(review["view_index"])
                route_consensus = None
                if (self.point_selection_prompt_version ==
                        "v17_turn_three_view_commit_prior" and
                        self.requested_point_selection_prompt_version ==
                        "v26_stage2_route_consensus" and
                        int(route_index) != route_original):
                    # A single side-sector VLM call can over-weight a salient
                    # object.  Require an independent second call to agree
                    # before changing the original proposal; disagreement
                    # keeps the first semantic proposal.
                    route_consensus, route_consensus_reason = self._call(
                        "adjudicate_stage2_route_continuity_consensus",
                        route_prompt, review_images, route_schema,
                        validate_route)
                    if int(route_consensus) != int(route_index):
                        route_reason = (
                            f"first adjudicator: {route_reason}; second "
                            f"adjudicator disagreed ({int(route_consensus)}): "
                            f"{route_consensus_reason}; retained original")
                        route_index = route_original
                review["stage2_route_adjudication"] = {
                    "original_view_index": route_original,
                    "adjudicated_view_index": int(route_index),
                    "reason": route_reason,
                }
                if route_consensus is not None:
                    review["stage2_route_adjudication"][
                        "consensus_view_index"] = int(route_consensus)
                if route_index != route_original:
                    review["model_view_index"] = route_original
                    review["view_index"] = int(route_index)
                    review["decision"] = "use_existing"
                    review["refined_relative_yaw_deg"] = signed_relative_yaw_deg(
                        candidates[route_index])
                    review["target_centering"] = "centered"
                    review["visible_route_evidence"] = route_reason
                    review["competitor_rejection"] = (
                        "independent stage-2 route-continuity adjudication "
                        "preferred the connected instruction ray")
            if self._stage2_route_guard:
                # Form-level geometry guard: avoid a deep side ray when a
                # legal shallower connected floor ray can execute the same
                # later-stage relation.  This is independent of episode IDs
                # and never uses reference-path data.
                form = str(stage.get("form", ""))
                instruction_lower = str(stage.get(
                    "navigation_instruction", "")).lower()
                current_index = int(review["view_index"])
                current_rel = signed_relative_yaw_deg(candidates[current_index])
                legal = [index for index in initial_allowed
                         if candidates[index].get("point") is not None]
                def rel(index):
                    return signed_relative_yaw_deg(candidates[index])
                route_landmark_words = [
                    word for word in re.findall(
                        r"[a-z]+", str(stage.get("landmark", "")).lower())
                    if len(word) > 2 and word not in {
                        "the", "and", "room", "area", "entrance",
                    }]
                route_review_text = " ".join(str(review.get(key, ""))
                                             for key in (
                                                 "visible_route_evidence",
                                                 "reason")).lower()
                relation_target_seen = bool(
                    route_landmark_words and any(
                        re.search(r"\b" + re.escape(word) + r"\b",
                                  route_review_text)
                        for word in route_landmark_words))
                # The review schema may place the landmark mention in a
                # structured field (for example first_following_landmark)
                # rather than in reason/visible_route_evidence.  Treat a
                # matching structured mention as equivalent evidence for the
                # form-level stop-ray guard; this remains model-local and
                # does not use a reference path or scene identity.
                review_landmark_text = json.dumps(review, ensure_ascii=False).lower()
                relation_target_seen = bool(
                    relation_target_seen or
                    (route_landmark_words and any(
                        re.search(r"\b" + re.escape(word) + r"\b",
                                  review_landmark_text)
                        for word in route_landmark_words)))
                adjustment_index = None
                adjustment_reason = None
                if (form == "STOP_WAIT" and
                        not re.search(r"\b(left|right|side|beside|along)\b",
                                      instruction_lower) and
                        abs(current_rel) > 45.0):
                    # For the strict RGB route-review prompt, a high-
                    # confidence landmark-bearing ray can be an intentional
                    # diagonal/rear-side endpoint (for example after a turn).
                    # Do not silently replace that reviewed view with an
                    # unrelated +/-45-degree floor patch.  The wider ray is
                    # still limited to a single 180-degree panorama and must have a ground
                    # anchor; low-confidence or unreviewed stops retain the
                    # conservative near-forward cap.
                    preserve_reviewed_stop = bool(
                        self.requested_point_selection_prompt_version in {
                            "v33_stop_relation_near_side",
                            "v34_unified_history_route_guard"} and
                        relation_target_seen and
                        float(review.get("confidence", 0.0) or 0.0) >= 0.70 and
                        abs(current_rel) <= 180.0 and
                        candidates[current_index].get("point") is not None)
                    if not preserve_reviewed_stop:
                        shallower = [index for index in legal
                                     if abs(rel(index)) <= 45.0]
                        if shallower:
                            adjustment_index = max(
                                shallower,
                                key=lambda index: float(candidates[index].get(
                                    "ground_fraction", 0.0)))
                            adjustment_reason = (
                                "STOP/WAIT without a side word capped to the "
                                "nearest connected forward ray")
                    else:
                        review["wide_stop_route_preserved"] = {
                            "relative_yaw_deg": float(current_rel),
                            "confidence": float(review.get("confidence", 0.0)),
                            "policy": "high-confidence RGB landmark-bearing stop ray",
                        }
                elif (form == "ADVANCE_STRAIGHT" and abs(current_rel) > 45.0):
                    shallower = [index for index in legal if abs(rel(index)) <= 45.0]
                    if shallower:
                        adjustment_index = max(
                            shallower,
                            key=lambda index: (
                                float(candidates[index].get("ground_fraction", 0.0)),
                                -abs(rel(index))))
                        adjustment_reason = (
                            "ADVANCE_STRAIGHT retained a forward connected "
                            "floor ray instead of a deep side ray")
                elif (form in {"ENTER_REGION", "EXIT_REGION",
                               "TRAVERSE_PORTAL_REGION", "SELECT_PORTAL"} and
                      abs(current_rel) > 90.0 and not (
                          self._stage3_relation_portal_v30 and
                          form in {"ENTER_REGION", "EXIT_REGION",
                                   "TRAVERSE_PORTAL_REGION", "SELECT_PORTAL"} and
                          relation_target_seen and
                          float(review.get("confidence", 0.0) or 0.0) >= 0.80)):
                    shallower = [index for index in legal
                                 if abs(rel(index)) <= 90.0 and
                                 float(candidates[index].get(
                                     "ground_fraction", 0.0)) > 0.02]
                    if shallower:
                        adjustment_index = min(
                            shallower, key=lambda index: abs(abs(rel(index)) - 90.0))
                        adjustment_reason = (
                            "portal/region route preferred the smallest legal "
                            "offset that still exposes connected floor")
                elif (form == "BETWEEN_OBJECTS" and abs(current_rel) > 90.0):
                    shallower = [index for index in legal
                                 if abs(rel(index)) <= 90.0 and
                                 float(candidates[index].get(
                                     "ground_fraction", 0.0)) > 0.02]
                    if shallower:
                        adjustment_index = max(
                            shallower, key=lambda index: abs(rel(index)))
                        adjustment_reason = (
                            "between-object route kept the widest non-rear "
                            "gap-bearing sector")
                if adjustment_index is not None and adjustment_index != current_index:
                    review["stage2_geometry_adjustment"] = {
                        "original_view_index": current_index,
                        "adjusted_view_index": int(adjustment_index),
                        "reason": adjustment_reason,
                    }
                    review["model_view_index"] = current_index
                    review["view_index"] = int(adjustment_index)
                    review["decision"] = "request_refined"
                    review["refined_relative_yaw_deg"] = rel(adjustment_index)
                    review["target_centering"] = "uncertain"
                    review["competitor_rejection"] = adjustment_reason
            if self._stage2_geometry_v27:
                # V27 applies the same form-level ray policy once more after
                # semantic adjudication, because the optional local RGB view
                # can otherwise reintroduce a deep side ray.  The policy is
                # expressed only in instruction form and local RGB geometry:
                # it never reads a reference path, navmesh, depth, or result.
                form = str(stage.get("form", ""))
                text_lower = str(stage.get(
                    "navigation_instruction", "")).lower()
                current_index = int(review["view_index"])
                current_rel = signed_relative_yaw_deg(candidates[current_index])
                geometry_lock = False
                desired_rel = None
                relation_landmark_words = [
                    word for word in re.findall(
                        r"[a-z]+", str(stage.get("landmark", "")).lower())
                    if len(word) > 2 and word not in {
                        "the", "and", "room", "area", "entrance",
                    }]
                relation_review_text = " ".join(str(review.get(key, ""))
                                                 for key in (
                                                     "visible_route_evidence",
                                                     "reason")).lower()
                relation_target_seen = bool(
                    relation_landmark_words and any(
                        re.search(r"\b" + re.escape(word) + r"\b",
                                  relation_review_text)
                        for word in relation_landmark_words))
                review_landmark_text = json.dumps(review, ensure_ascii=False).lower()
                relation_target_seen = bool(
                    relation_target_seen or
                    (relation_landmark_words and any(
                        re.search(r"\b" + re.escape(word) + r"\b",
                                  review_landmark_text)
                        for word in relation_landmark_words)))
                if form == "STOP_WAIT":
                    side_match = re.search(
                        r"\b(left|right)\b", text_lower)
                    if ((self._stage3_stop_relation_v29 or
                         self._stage3_relation_portal_v30) and side_match):
                        # For a relation such as "doorway on the left", keep
                        # the semantic review's side-bearing camera ray.  A
                        # fixed +/-45-degree recentering can turn a valid
                        # lateral stop into a direct approach through the
                        # doorway (the landmark then appears in FRONT rather
                        # than on the requested side).  V33 therefore lets
                        # the RGB review select among the explicit side
                        # sectors and uses near-side anchor geometry below to
                        # control the stopping distance.
                        if (self.requested_point_selection_prompt_version in {
                                "v33_stop_relation_near_side",
                                "v34_unified_history_route_guard"}):
                            desired_rel = float(current_rel)
                        else:
                            desired_rel = (
                                45.0 if side_match.group(1) == "left"
                                else -45.0)
                        geometry_lock = True
                    elif (self._stage3_stop_relation_v29 or
                          self._stage3_relation_portal_v30):
                        # A final stop is not automatically a straight-ahead
                        # stop.  Preserve a landmark-bearing RGB route when
                        # the independent route adjudicator actually mentions
                        # the named landmark and the ray is not rear-facing;
                        # otherwise retain the conservative central ray.
                        relation_words = {
                            "near", "corner", "beside", "along", "past",
                            "beyond", "front", "doorway", "opening",
                        }
                        relation_text = " ".join(str(stage.get(key, ""))
                                                  for key in (
                                                      "navigation_instruction",
                                                      "spatial_relation",
                                                      "semantic_spatial_target",
                                                  )).lower()
                        relation_target_stop = bool(
                            relation_target_seen and
                            relation_words.intersection(
                                set(re.findall(r"[a-z]+", relation_text))))
                        wide_stop_allowed = bool(
                            self.requested_point_selection_prompt_version in {
                                "v33_stop_relation_near_side",
                                "v34_unified_history_route_guard"} and
                            relation_target_stop and
                            float(review.get("confidence", 0.0) or 0.0) >= 0.70 and
                            abs(current_rel) <= 180.0)
                        if ((relation_target_stop and abs(current_rel) <= 90.0) or
                                wide_stop_allowed):
                            desired_rel = float(current_rel)
                            review["stage3_landmark_bearing_preserved"] = True
                        else:
                            desired_rel = 0.0
                        geometry_lock = True
                    elif not re.search(r"\b(left|right|side|beside|along)\b",
                                       text_lower):
                        # V27/V28 behavior: an unqualified corner/end stop is
                        # a front-lane stop; ask the local RGB provider for a
                        # truly central ray when the eight-view set has no
                        # front floor.
                        desired_rel = 0.0
                        geometry_lock = True
                elif form == "ADVANCE_STRAIGHT":
                    # Preserve a connected forward ray and suppress a later
                    # semantic refinement that veers into a side branch.
                    desired_rel = float(np.clip(current_rel, -45.0, 45.0))
                    geometry_lock = True
                elif form in {"PASS_LANDMARK", "FOLLOW_PATH_BOUNDARY",
                              "APPROACH_LANDMARK"}:
                    # A straight/pass continuation may show the named object
                    # from a side sector while the executable route remains
                    # in front.  When an allowed near-forward ground ray is
                    # present, keep that route instead of turning toward the
                    # salient landmark itself.  This is a form-level RGB/2-D
                    # geometry guard; the VLM remains responsible for naming
                    # the object and no demonstration/path state is used.
                    rgb_scene_stds = [
                        float(np.asarray(candidates[index].get(
                            "rgb", np.zeros((1, 1, 3), np.uint8)),
                            np.float32).std())
                        for index in initial_allowed]
                    scene_std_floor = max(
                        5.0, 0.80 * float(np.median(rgb_scene_stds))
                        if rgb_scene_stds else 5.0)
                    near_forward = [
                        index for index in initial_allowed
                        if candidates[index].get("point") is not None and
                        candidates[index].get("ground_anchors") and
                        float(np.asarray(candidates[index].get(
                            "rgb", np.zeros((1, 1, 3), np.uint8)),
                            np.float32).std()) >= scene_std_floor and
                        abs(signed_relative_yaw_deg(
                            candidates[index])) <= 45.0001]
                    low_quality_forward = [
                        index for index in initial_allowed
                        if candidates[index].get("point") is not None and
                        candidates[index].get("ground_anchors") and
                        float(np.asarray(candidates[index].get(
                            "rgb", np.zeros((1, 1, 3), np.uint8)),
                            np.float32).std()) < scene_std_floor and
                        abs(signed_relative_yaw_deg(
                            candidates[index])) <= 45.0001]
                    # PASS/FOLLOW are route-relative operations, not
                    # camera-forward operations.  A prior version replaced a
                    # reviewed +/-90-degree corridor with any visually usable
                    # +/-45-degree floor patch.  This is wrong after a doorway
                    # or landing, where the continuing hallway can genuinely
                    # be lateral to the arrival camera.  Preserve the VLM's
                    # RGB-reviewed ray for PASS/FOLLOW; the incoming-direction
                    # and blocked-yaw gates have already removed true reversal.
                    # Keep the near-forward repair only for APPROACH, where a
                    # side detector can otherwise pull the point off the floor
                    # lane leading toward the landmark.
                    if (form == "APPROACH_LANDMARK" and near_forward and
                            abs(current_rel) > 45.0001):
                        def forward_floor_score(index):
                            mask = np.asarray(
                                candidates[index]["target_mask"], bool)
                            h, w = mask.shape
                            lower = mask[int(0.52 * h):int(0.86 * h),
                                         int(0.15 * w):int(0.85 * w)]
                            central = mask[:, int(0.30 * w):int(0.70 * w)]
                            return float(lower.sum()) + 0.35 * float(central.sum())
                        best_forward = max(near_forward,
                                            key=forward_floor_score)
                        review["straight_route_geometry_adjustment"] = {
                            "from_view_index": int(current_index),
                            "to_view_index": int(best_forward),
                            "from_relative_yaw_deg": round(float(current_rel), 3),
                            "to_relative_yaw_deg": round(float(
                                signed_relative_yaw_deg(
                                    candidates[best_forward])), 3),
                            "policy": "retain executable near-forward pass lane",
                        }
                        review["model_view_index"] = current_index
                        current_index = int(best_forward)
                        review["view_index"] = current_index
                        current_rel = signed_relative_yaw_deg(
                            candidates[current_index])
                        # If the exact forward sector is a low-information
                        # frame (e.g. a clipped wall) but a neighboring
                        # +/-45-degree sector is visible, request one local
                        # intermediate RGB view rather than accepting the
                        # full side-sector heading.  The midpoint stays
                        # within the strict angular budget and is validated
                        # for ground anchors by the normal refinement path.
                        if (low_quality_forward and
                                abs(current_rel) > 1e-3):
                            poor = min(
                                low_quality_forward,
                                key=lambda index: abs(
                                    signed_relative_yaw_deg(
                                        candidates[index])))
                            poor_rel = signed_relative_yaw_deg(
                                candidates[poor])
                            midpoint_rel = poor_rel + 0.5 * (
                                current_rel - poor_rel)
                            review["decision"] = "request_refined"
                            review["refined_relative_yaw_deg"] = float(
                                midpoint_rel)
                            review["target_centering"] = "uncertain"
                            review["rgb_quality_refinement"] = {
                                "poor_view_index": int(poor),
                                "visible_neighbor_view_index": int(
                                    current_index),
                                "requested_relative_yaw_deg": float(
                                    midpoint_rel),
                                "policy": "midpoint between low-quality forward and visible adjacent ground ray",
                            }
                        else:
                            review["decision"] = "use_existing"
                    desired_rel = (
                        float(np.clip(current_rel, -45.0, 45.0))
                        if form == "APPROACH_LANDMARK" else
                        float(current_rel))
                    geometry_lock = True
                elif form in {"ENTER_REGION", "EXIT_REGION",
                              "TRAVERSE_PORTAL_REGION", "SELECT_PORTAL",
                              "BETWEEN_OBJECTS"}:
                    # Portal/gap crossings need a modest lateral offset to
                    # expose the opening, but not a 90/135-degree side ray.
                    stage_relation_text = " ".join(str(stage.get(key, ""))
                                                    for key in (
                                                        "navigation_instruction",
                                                        "semantic_spatial_target",
                                                        "spatial_relation",
                                                        "completion_cue")).lower()
                    explicit_side_portal = bool(re.search(
                        r"\b(?:door|doorway|opening|entrance|exit)\b[^.]{0,45}"
                        r"\b(?:on|to|at|from)?\s*(?:the\s+)?(?:left|right)\b|"
                        r"\b(?:left|right)\b[^.]{0,45}\b(?:door|doorway|opening|"
                        r"entrance|exit)\b", stage_relation_text))
                    current_strategy = candidates[current_index].get(
                        "strategy_application", {}) or {}
                    side_portal_evidence = bool(
                        explicit_side_portal and
                        45.0 <= abs(current_rel) <= 135.0001 and
                        candidates[current_index].get("point") is not None and
                        current_strategy.get("mode") ==
                        "floor_through_detected_portal" and
                        int(current_strategy.get("candidate_pixels", 0) or 0) >= 24 and
                        bool(current_strategy.get(
                            "relation_detection_evidence_in_view", False)))
                    if side_portal_evidence:
                        # Preserve an explicit side-qualified portal bearing;
                        # clipping it to +/-75 would turn a real 135-degree
                        # door view into a different corridor.  The candidate
                        # has already passed the RGB/ground relation gate.
                        desired_rel = float(current_rel)
                    elif (self._stage3_relation_portal_v30 and
                            form == "SELECT_PORTAL" and
                            relation_target_seen and
                            float(review.get("confidence", 0.0) or 0.0) >= 0.80):
                        # A high-confidence ordered portal view is already a
                        # semantic bearing.  Keeping it (up to ±135°) avoids
                        # turning a doorway/bed target into an unrelated
                        # central floor patch.
                        desired_rel = float(np.clip(current_rel, -135.0, 135.0))
                    elif abs(current_rel) > 75.0:
                        desired_rel = math.copysign(75.0, current_rel)
                    else:
                        desired_rel = current_rel
                    geometry_lock = True
                if geometry_lock:
                    if (form in {"PASS_LANDMARK", "FOLLOW_PATH_BOUNDARY",
                                 "APPROACH_LANDMARK"} and
                            isinstance(review.get("rgb_quality_refinement"),
                                       dict)):
                        desired_rel = float(review[
                            "rgb_quality_refinement"].get(
                                "requested_relative_yaw_deg", desired_rel))
                    if (self._stage2_side_stop_v28 and form == "STOP_WAIT" and
                            not re.search(
                                r"\b(left|right|side|beside|along)\b",
                                text_lower)):
                        # If no legal initial sector has a near-forward floor
                        # anchor, use an RGB-only side cue to acquire a view
                        # between the landmark-bearing side and the front.
                        # This recovers Grounded-SAM gaps without admitting a
                        # rear/backtracking ray.  The cue is aggregated over
                        # the VLM's independent review text, not an episode ID.
                        near_forward = [
                            index for index in initial_allowed
                            if candidates[index].get("point") is not None and
                            abs(rel(index)) <= 30.0]
                        if not near_forward:
                            review_text = " ".join(str(review.get(key, ""))
                                                    for key in (
                                                        "reason",
                                                        "visible_route_evidence",
                                                        "competitor_rejection"))
                            right_hits = len(re.findall(r"\bright\b",
                                                         review_text.lower()))
                            left_hits = len(re.findall(r"\bleft\b",
                                                       review_text.lower()))
                            if right_hits > left_hits:
                                desired_rel = -30.0
                            elif left_hits > right_hits:
                                desired_rel = 30.0
                            else:
                                desired_rel = 0.0
                            base_candidates = [
                                index for index, candidate in
                                enumerate(candidates)
                                if not candidate.get("hard_excluded", False)]
                            if base_candidates:
                                current_index = min(
                                    base_candidates,
                                    key=lambda index: abs(
                                        rel(index) - desired_rel))
                                current_rel = rel(current_index)
                                review["view_index"] = current_index
                                refinement_base_index = current_index
                            review["stage2_side_aware_fallback"] = {
                                "right_text_hits": int(right_hits),
                                "left_text_hits": int(left_hits),
                                "requested_relative_yaw_deg": float(
                                    desired_rel),
                                "policy": (
                                    "no near-forward Grounded-SAM anchor: "
                                    "request a non-rear side/front RGB ray"),
                            }
                    review["stage2_final_ray_geometry"] = {
                        "current_relative_yaw_deg": round(float(current_rel), 3),
                        "requested_relative_yaw_deg": round(float(
                            desired_rel), 3),
                        "policy": (
                            "STOP_WAIT central front ray; ADVANCE_STRAIGHT "
                            "connected forward ray; portal/gap smallest "
                            "legal +/-75-degree ray"),
                    }
                    review["stage2_geometry_lock"] = True
                    # A prior RGB-centrality pass may already have populated a
                    # different requested yaw.  The final-ray contract owns
                    # that value for V27 even when the candidate sector itself
                    # is retained, so do not let the stale refinement request
                    # bypass the form-level cap below.
                    review["refined_relative_yaw_deg"] = float(desired_rel)
                    if abs(float(desired_rel) - float(current_rel)) > 1e-3:
                        review["model_view_index"] = current_index
                        review["decision"] = "request_refined"
                        review["refined_relative_yaw_deg"] = float(desired_rel)
                        review["target_centering"] = "uncertain"
                        review["competitor_rejection"] = (
                            "V27 final-ray geometry kept the instruction-form "
                            "route in a central/near-side connected sector")
                        refinement_base_index = current_index
            if (bare_turn_setup.get("active") and
                    direction_form in {"TURN_LEFT", "TURN_RIGHT"}):
                # The original eight-view side gate is the semantic contract
                # for a bare turn.  When the RGB reviewer ranks two adjacent
                # legal side sectors first and second, their shared corridor
                # commonly lies on the 45-degree image seam.  Acquire one
                # midpoint RGB view instead of forcing the following
                # STOP/route form back to relative yaw zero.  Both endpoints
                # remain inside the commanded 45/90/135-degree side set.
                primary = int(review["view_index"])
                competitor = int(review.get(
                    "strongest_competing_view_index", -1))
                if (competitor in initial_allowed and competitor != primary):
                    primary_rel = signed_relative_yaw_deg(candidates[primary])
                    competitor_rel = signed_relative_yaw_deg(
                        candidates[competitor])
                    delta = ((competitor_rel - primary_rel + 180.0) %
                             360.0 - 180.0)
                    correct_side = bool(
                        (direction_form == "TURN_LEFT" and
                         primary_rel > 0.0 and competitor_rel > 0.0) or
                        (direction_form == "TURN_RIGHT" and
                         primary_rel < 0.0 and competitor_rel < 0.0))
                    if correct_side and abs(delta) <= 45.0001:
                        midpoint_rel = primary_rel + 0.5 * delta
                        review["decision"] = "request_refined"
                        review["refined_relative_yaw_deg"] = float(
                            midpoint_rel)
                        review["target_centering"] = "uncertain"
                        refinement_base_index = primary
                        review["bare_turn_side_boundary_refinement"] = {
                            "primary_view_index": primary,
                            "competitor_view_index": competitor,
                            "requested_relative_yaw_deg": round(
                                float(midpoint_rel), 3),
                            "policy": (
                                "midpoint RGB view between the two strongest "
                                "adjacent legal turn-side sectors"),
                        }
                        review["stage2_final_ray_geometry"] = {
                            "current_relative_yaw_deg": round(
                                float(primary_rel), 3),
                            "requested_relative_yaw_deg": round(
                                float(midpoint_rel), 3),
                            "policy": (
                                "bare-turn three-view sector owns the final "
                                "ray over the following route form"),
                        }
            if (use_turn_commit_prior and
                    str(stage.get("form", "")) in {
                        "TURN_LEFT", "TURN_RIGHT"} and
                    (re.fullmatch(
                        r"\s*turn\s+(?:to\s+the\s+)?(?:left|right)\s*[.!]?\s*",
                        str(stage.get("navigation_instruction", "")),
                        flags=re.IGNORECASE) or
                     re.search(r"\bturn\s+(?:left|right)\b",
                               str(stage.get("navigation_instruction", "")),
                               flags=re.IGNORECASE))):
                selected_turn_index = int(review["view_index"])
                selected_turn_yaw = signed_relative_yaw_deg(
                    candidates[selected_turn_index])
                form = str(stage.get("form", ""))
                shallow_turn = (
                    0.0 < selected_turn_yaw <= 45.001
                    if form == "TURN_LEFT" else
                    -45.001 <= selected_turn_yaw < 0.0)
                committed_views = [
                    index for index in initial_allowed
                    if ((form == "TURN_LEFT" and
                         89.999 <= signed_relative_yaw_deg(
                             candidates[index]) <= 135.001) or
                        (form == "TURN_RIGHT" and
                         -135.001 <= signed_relative_yaw_deg(
                             candidates[index]) <= -89.999))]
                review_text = " ".join(str(review.get(key, ""))
                                        for key in (
                                            "reason", "visible_route_evidence"))
                strong_connected_route = bool(re.search(
                    r"\b(?:clear|wide|connected|continuous|extending|"
                    r"visible)\b[^.]{0,80}\b(?:corridor|floor|lane|"
                    r"opening|route)\b",
                    review_text, flags=re.IGNORECASE))
                review_confidence = float(review.get("confidence", 0.0) or 0.0)
                # A committed +/-90-degree ray is a fallback for an
                # ambiguous shallow proposal, not a blanket override.  When
                # the RGB adjudicator is confident and explicitly describes
                # a connected corridor in the +/-45-degree sector, preserve
                # that semantic ray; replacing it with a wall-heavy 90-degree
                # view can send the executor into a different branch.
                if (shallow_turn and committed_views and
                        # A bare turn has no destination semantics that can
                        # justify a shallow veer. Unless the review gives
                        # explicit connected-route evidence, commit to the
                        # full side sector even when its confidence number is
                        # high; this keeps the direction gate stable across
                        # stochastic VLM confidence calibration.
                        (not strong_connected_route or
                         not re.fullmatch(
                             r"\s*turn\s+(?:to\s+the\s+)?(?:left|right)\s*[.!]?\s*",
                             str(stage.get("navigation_instruction", "")),
                             flags=re.IGNORECASE) or
                         review_confidence < 0.65)):
                    desired_yaw = 90.0 if form == "TURN_LEFT" else -90.0
                    corrected_index = min(
                        committed_views,
                        key=lambda index: abs(
                            signed_relative_yaw_deg(candidates[index]) -
                            desired_yaw))
                    review["model_view_index"] = selected_turn_index
                    review["view_index"] = int(corrected_index)
                    review["decision"] = "use_existing"
                    review["refined_relative_yaw_deg"] = (
                        signed_relative_yaw_deg(candidates[corrected_index]))
                    review["target_centering"] = "centered"
                    review["turn_commit_adjustment"] = (
                        "bare turn command preferred a ground-bearing fully "
                        "committed side view over a shallow 45-degree veer")
                # A direction-only turn has a defined nominal quarter-turn,
                # and the executor later aligns the arrived camera to that
                # exact +/-90-degree heading.  If a legal +/-90 panorama ray
                # exists, do not first translate toward a +/-135 ray and then
                # rotate back: that lateral displacement is unrelated to any
                # named destination and can enter the neighboring branch.
                bare_turn_exact = bool(re.fullmatch(
                    r"\s*turn\s+(?:to\s+the\s+)?(?:left|right)\s*[.!]?\s*",
                    str(stage.get("navigation_instruction", "")),
                    flags=re.IGNORECASE))
                selected_turn_index = int(review["view_index"])
                desired_yaw = 90.0 if form == "TURN_LEFT" else -90.0
                nominal_views = [
                    index for index in initial_allowed
                    if abs(signed_relative_yaw_deg(candidates[index]) -
                           desired_yaw) <= 10.0]
                if (bare_turn_exact and nominal_views and
                        abs(signed_relative_yaw_deg(
                            candidates[selected_turn_index]) -
                            desired_yaw) > 30.0):
                    corrected_index = min(
                        nominal_views,
                        key=lambda index: abs(
                            signed_relative_yaw_deg(candidates[index]) -
                            desired_yaw))
                    review["model_view_index"] = selected_turn_index
                    review["view_index"] = int(corrected_index)
                    review["decision"] = "use_existing"
                    review["refined_relative_yaw_deg"] = (
                        signed_relative_yaw_deg(candidates[corrected_index]))
                    review["target_centering"] = "centered"
                    review["bare_turn_nominal_ray_adjustment"] = (
                        "direction-only turn used the available nominal "
                        "+/-90-degree ground ray before exact post-arrival "
                        "orientation alignment")
                    review["stage2_geometry_lock"] = True
            refinement_base_index = int(review["view_index"])
            if (use_rgb_evidence_refinement and rgb_evidence_plan is not None
                    and not review.get("stage2_geometry_lock", False)):
                preferred_index = int(rgb_evidence_plan[
                    "preferred_view_index"])
                neighborhood = list(rgb_evidence_plan[
                    "ground_bearing_neighborhood"])

                def evidence_view_delta(first, second):
                    return abs(math.degrees((math.radians(
                        signed_relative_yaw_deg(candidates[first]) -
                        signed_relative_yaw_deg(candidates[second])) +
                        math.pi) % (2 * math.pi) - math.pi))

                model_index = int(review["view_index"])
                semantic_route_committed = bool(
                    self._task30_route_anchor and
                    str(review.get("decision", "")) == "use_existing" and
                    str(review.get("target_centering", "")) == "centered" and
                    float(review.get("confidence", 0.0) or 0.0) >= 0.8 and
                    bool(str(review.get(
                        "visible_route_evidence", "")).strip()) and
                    bool(str(review.get(
                        "competitor_rejection", "")).strip()) and
                    int(review.get(
                        "first_following_landmark_view_index", -1)) ==
                    model_index and
                    str(review.get("sequence_alignment", "")) == "same_view")
                sequence_route_committed = bool(
                    self._task30_route_anchor and
                    ((isinstance(review.get("sequence_adjustment"), str) and
                      bool(review.get("sequence_adjustment")) and
                      int(review.get(
                          "first_following_landmark_view_index", -1)) >= 0) or
                     bool((review.get(
                         "ambiguous_landmark_route_adjudication", {}) or {})
                          .get("active"))))
                # For a first-stage PASS_LANDMARK explicitly described as
                # straight/go-straight, preserve a VLM-confirmed near-forward
                # route when that view already has a non-empty beyond-landmark
                # RGB ground mask.  The compact detector evidence used by the
                # generic neighborhood heuristic can be one panorama sector
                # off (e.g. the landmark box straddles two views); replacing a
                # confirmed forward route with that sector creates a near-field
                # side ray and corrupts the next action history.  This is a
                # form-level RGB/2-D rule and does not inspect depth, navmesh,
                # demonstration paths, or execution outcomes.
                pass_forward_commit = bool(
                    self.requested_point_selection_prompt_version in {
                        "v32_first_stage_shallow_route",
                        "v34_unified_history_route_guard"} and
                    self._first_step_route_guard and
                    str(stage.get("form", "")) == "PASS_LANDMARK" and
                    re.search(r"\b(?:straight|go|walk|continue)\b",
                              str(stage.get("navigation_instruction", "")),
                              flags=re.IGNORECASE) and
                    abs(signed_relative_yaw_deg(
                        candidates[model_index])) <= 30.0 and
                    int(np.asarray(candidates[model_index][
                        "target_mask"], bool).sum()) >= 24)
                if (neighborhood and
                        "ground_fallback_adjustment" not in review and
                        not pass_forward_commit and
                        not semantic_route_committed and
                        not sequence_route_committed and
                        evidence_view_delta(model_index, preferred_index) >=
                        44.999):
                    corrected_index = min(
                        neighborhood,
                        key=lambda index: evidence_view_delta(
                            index, preferred_index))
                    review["model_view_index"] = model_index
                    review["view_index"] = int(corrected_index)
                    review["decision"] = "use_existing"
                    review["refined_relative_yaw_deg"] = (
                        signed_relative_yaw_deg(candidates[corrected_index]))
                    review["target_centering"] = "centered"
                    review["rgb_evidence_adjustment"] = (
                        "moved to the nearest ground-bearing sector around a "
                        "compact RGB detection")
                elif ((semantic_route_committed or sequence_route_committed) and
                        neighborhood and
                        evidence_view_delta(model_index, preferred_index) >=
                        44.999):
                    review["rgb_evidence_adjustment_suppressed"] = (
                        "retained the high-confidence centered eight-view "
                        "route already grounded by current/next-clause "
                        "sequence consistency; compact detector evidence is "
                        "supporting only")
                elif not neighborhood:
                    review["model_view_index"] = model_index
                    review["decision"] = "request_refined"
                    review["refined_relative_yaw_deg"] = float(
                        rgb_evidence_plan[
                            "preferred_image_ray_yaw_deg"])
                    review["target_centering"] = "uncertain"
                    review["rgb_evidence_adjustment"] = (
                        "compact RGB landmark has no ground-bearing same or "
                        "adjacent sector; request a centered landmark-ray view")
                    refinement_base_index = preferred_index
            selected_index = int(review["view_index"])
            shallow_compound_turn = bool(
                str(stage.get("form", "")).upper() in {
                    "TURN_LEFT", "TURN_RIGHT"} and
                re.search(
                    r"\b(?:walk|go|head|move|proceed)\s+towards?\b",
                    str(stage.get("navigation_instruction", "")).lower()) and
                abs(abs(signed_relative_yaw_deg(
                    candidates[selected_index])) - 45.0) <= 1e-6)
            if shallow_compound_turn:
                competitor_index = int(review.get(
                    "strongest_competing_view_index", -1))
                if competitor_index in initial_allowed:
                    selected_deg = signed_relative_yaw_deg(
                        candidates[selected_index])
                    competitor_deg = signed_relative_yaw_deg(
                        candidates[competitor_index])
                    competitor_delta = (
                        (competitor_deg - selected_deg + 180.0) % 360.0 -
                        180.0)
                    same_side = selected_deg * competitor_deg > 0.0
                    if same_side and abs(competitor_delta) <= 45.0001:
                        requested_deg = selected_deg + 0.5 * competitor_delta
                        review["decision"] = "request_refined"
                        review["refined_relative_yaw_deg"] = requested_deg
                        review["target_centering"] = "uncertain"
                        review["compound_turn_boundary_refinement"] = {
                            "base_view_index": int(selected_index),
                            "competitor_view_index": int(competitor_index),
                            "base_relative_yaw_deg": float(selected_deg),
                            "competitor_relative_yaw_deg": float(
                                competitor_deg),
                            "requested_relative_yaw_deg": float(
                                requested_deg),
                            "policy": (
                                "mandatory midpoint RGB comparison between "
                                "the VLM primary and strongest adjacent "
                                "same-side route views"),
                        }
                        refinement_base_index = selected_index
            if (use_rgb_evidence_refinement and
                    review["decision"] == "use_existing" and
                    not review.get("stage2_geometry_lock", False)):
                mask = np.asarray(candidates[selected_index]["target_mask"], bool)
                width = mask.shape[1]
                central_floor = bool(mask[:, int(math.floor(0.30 * width)):
                                          int(math.ceil(0.70 * width))].any())
                pass_route_committed = bool(
                    self._stage1_turn_soft and
                    str(stage.get("form", "")) == "PASS_LANDMARK" and
                    isinstance(review.get("pass_adjudication"), dict) and
                    int(review["pass_adjudication"].get(
                        "adjudicated_view_index", selected_index)) !=
                    int(review["pass_adjudication"].get(
                        "original_view_index", selected_index)))
                if not central_floor and not pass_route_committed:
                    base_deg = signed_relative_yaw_deg(
                        candidates[selected_index])
                    if (rgb_evidence_plan is not None and
                            int(rgb_evidence_plan[
                                "preferred_view_index"]) == selected_index):
                        requested_deg = float(rgb_evidence_plan[
                            "preferred_image_ray_yaw_deg"])
                    else:
                        point = np.asarray(candidates[selected_index].get(
                            "point", [width / 2, 0]), np.float32)
                        pixel_bearing = math.degrees(math.atan(
                            (float(point[0]) - (width - 1) / 2) /
                            (width / 2)))
                        requested_deg = base_deg - pixel_bearing
                    delta = (requested_deg - base_deg + 180.0) % 360.0 - 180.0
                    requested_deg = base_deg + float(np.clip(delta, -30.0, 30.0))
                    requested_deg = (requested_deg + 180.0) % 360.0 - 180.0
                    review["decision"] = "request_refined"
                    review["refined_relative_yaw_deg"] = requested_deg
                    review["target_centering"] = "uncertain"
                    review["rgb_central_floor_adjustment"] = (
                        "confirmed sector has no target-mask pixels in the "
                        "central 40%; request one local RGB view")
                    refinement_base_index = selected_index
                elif pass_route_committed and not central_floor:
                    # The independent PASS_LANDMARK adjudication has already
                    # committed to a deliberate side-sector far opening.  A
                    # central-floor repair would pull that semantic route back
                    # toward the original forward sector and can destroy the
                    # intended bearing.  Keep the committed 2-D route ray.
                    review["pass_route_center_override"] = (
                        "kept adjudicated side-sector route despite sparse "
                        "central floor; no local recentering")
            if self._relation_route_guard and str(stage.get("form", "")) == (
                    "FOLLOW_PATH_BOUNDARY"):
                # A hallway/end-boundary instruction is vulnerable to two
                # visually similar corridors in opposite panorama sectors.
                # If the VLM selected one but an opposite legal sector has
                # substantially stronger connected lower/central floor, use
                # that RGB 2-D continuity cue as a generic tie-break.  This
                # does not inspect depth, navmesh, or the reference path.
                def corridor_floor_score(index):
                    mask = np.asarray(candidates[index]["target_mask"], bool)
                    height, width = mask.shape
                    central = mask[:, int(0.20 * width):int(0.80 * width)]
                    lower = mask[int(0.52 * height):int(0.86 * height),
                                 int(0.15 * width):int(0.85 * width)]
                    if not mask.any():
                        return -1.0
                    return (float(lower.sum()) + 0.35 * float(central.sum())) / \
                        max(float(height * width), 1.0)

                route_allowed = [
                    index for index in initial_allowed
                    if candidates[index].get("ground_anchors")]
                if len(route_allowed) > 1:
                    selected_score = corridor_floor_score(selected_index)
                    best_route = max(route_allowed, key=corridor_floor_score)
                    best_score = corridor_floor_score(best_route)
                    selected_yaw = signed_relative_yaw_deg(
                        candidates[selected_index])
                    best_yaw = signed_relative_yaw_deg(candidates[best_route])
                    yaw_delta = abs(math.degrees((math.radians(
                        selected_yaw - best_yaw) + math.pi) %
                        (2 * math.pi) - math.pi))
                    route_review_committed = bool(
                        self._task30_route_anchor and
                        float(review.get("confidence", 0.0) or 0.0) >= 0.8 and
                        int(review.get(
                            "first_following_landmark_view_index", -1)) ==
                        int(selected_index))
                    if (best_route != selected_index and yaw_delta > 45.0 and
                            best_score >= selected_score * 1.20 and
                            not route_review_committed and
                            re.search(r"\b(?:hall|hallway|corridor|end|archway)\b",
                                      str(stage.get(
                                          "navigation_instruction", "")).lower())):
                        review["model_view_index"] = selected_index
                        review["view_index"] = int(best_route)
                        review["decision"] = "use_existing"
                        review["refined_relative_yaw_deg"] = best_yaw
                        review["target_centering"] = "centered"
                        review["corridor_floor_adjustment"] = {
                            "from_view_index": int(selected_index),
                            "to_view_index": int(best_route),
                            "yaw_delta_deg": round(float(yaw_delta), 3),
                            "selected_score": round(float(selected_score), 5),
                            "replacement_score": round(float(best_score), 5),
                            "policy": "RGB 2-D lower/central floor continuity",
                        }
                        selected_index = int(best_route)
            primary_index = selected_index
            alternative_index = int(review.get(
                "alternative_view_index", selected_index))
            view_refinement = {
                **review,
                "initial_allowed_views": initial_allowed,
                "refinement_attempted": False,
                "refinement_accepted": False,
                "refined_view_index": None,
                "fallback_reason": None,
            }
            if review["decision"] == "request_refined":
                if side_stop_near_side:
                    # The semantic reviewer can select a +/-90-degree view
                    # because the doorway occupies its image center.  The
                    # V33 contract deliberately acquires the corresponding
                    # +/-45-degree near-side ray.  Use the closest existing
                    # sector as the refinement base so the local RGB request
                    # is measured from the intended side, not from a stale
                    # rear/side candidate (which would appear >45 degrees
                    # away and be rejected as non-local).
                    instruction_lower = str(stage.get(
                        "navigation_instruction", "")).lower()
                    requested_side_deg = float(review.get(
                        "refined_relative_yaw_deg", 0.0))
                    if self.requested_point_selection_prompt_version not in {
                            "v33_stop_relation_near_side",
                            "v34_unified_history_route_guard"}:
                        requested_side_deg = (
                            45.0 if re.search(r"\bleft\b", instruction_lower)
                            else -45.0)
                    def side_ray_delta(index):
                        delta = (signed_relative_yaw_deg(candidates[index]) -
                                 requested_side_deg + 180.0) % 360.0 - 180.0
                        return abs(float(delta))
                    side_bases = [
                        index for index in initial_allowed
                        if not candidates[index].get("hard_excluded", False)]
                    if side_bases:
                        refinement_base_index = min(side_bases,
                                                   key=side_ray_delta)
                view_refinement["refinement_attempted"] = True
                fallback_deg = math.degrees(float(candidates[
                    refinement_base_index].get("relative_yaw_rad", 0.0)))
                view_refinement["refinement_base_view_index"] = (
                    refinement_base_index)
                refinement_delta_deg = abs(math.degrees(
                    (math.radians(review["refined_relative_yaw_deg"] -
                                  fallback_deg) + math.pi) %
                    (2 * math.pi) - math.pi))
                view_refinement["refinement_delta_from_fallback_deg"] = (
                    refinement_delta_deg)
                # Trigonometric round trips can represent an exact 30-degree
                # request as 30.000000000000018.  Keep the documented closed
                # interval and reject only a materially non-local request.
                max_local_delta = 30.0
                if (self._stage2_geometry_v27 and
                        str(stage.get("form", "")) == "STOP_WAIT" and
                        not re.search(r"\b(left|right|side|beside|along)\b",
                                      str(stage.get(
                                          "navigation_instruction", "")).lower())):
                    # If no initial sector has a front floor anchor, a
                    # central STOP/WAIT ray can legitimately be a 45-degree
                    # local RGB acquisition from the nearest floor-bearing
                    # sector.  This is a form-level fallback, not an EP case.
                    max_local_delta = 45.0
                if ((self._stage3_stop_relation_v29 or
                     self._stage3_relation_portal_v30) and
                        str(stage.get("form", "")) == "STOP_WAIT"):
                    # v29 may request a calibrated side ray for a final
                    # relation; permit that one local 45-degree acquisition
                    # even when the instruction contains an explicit side
                    # word.  The ray remains in the forward hemisphere.
                    max_local_delta = 45.0
                if refinement_delta_deg > max_local_delta + 1e-6:
                    view_refinement["fallback_reason"] = (
                        f"requested yaw is more than {max_local_delta:g} degrees from the best "
                        "existing fallback and is not a local refinement")
                elif refinement_provider is None:
                    view_refinement["fallback_reason"] = (
                        "refinement provider unavailable")
                else:
                    refined = refinement_provider(math.radians(
                        review["refined_relative_yaw_deg"]))
                    refined = dict(refined)
                    if "depth" in refined:
                        raise ValueError(
                            "refined VLM candidate must not contain depth")
                    for field in (
                            "detection_records", "ground_detection_records"):
                        refined[field] = [
                            {key: value for key, value in dict(record).items()
                             if "depth" not in str(key).lower()}
                            for record in refined.get(field, [])
                        ]
                    refined["view_index"] = len(candidates)
                    refined["ground_anchors"] = self._ground_anchors(
                        refined["target_mask"], refined.get("point"))
                    refined["ground_anchors"] = history_safe_anchors(
                        refined, refined["ground_anchors"])
                    refined_yaw_deg = signed_relative_yaw_deg(refined)
                    refined_outside_direction_gate = False
                    if ((self._history_safe_refinement or
                         self.requested_point_selection_prompt_version in {
                             "v33_stop_relation_near_side",
                             "v34_unified_history_route_guard"}) and
                            direction_gate.get("active")):
                        sector = direction_gate.get("sector")
                        side_min_deg = (20.0 if review.get(
                            "compound_turn_boundary_refinement") else 45.0)
                        refined_outside_direction_gate = bool(
                            (sector == "left" and not
                             side_min_deg <= refined_yaw_deg <= 135.0) or
                            (sector == "right" and not
                             -135.0 <= refined_yaw_deg <= -side_min_deg) or
                            (sector == "rear" and
                            abs(refined_yaw_deg) < 135.0))
                    if (self._stage1_turn_soft and
                            direction_gate.get("active")):
                        sector = direction_gate.get("sector")
                        side_min_deg = (20.0 if review.get(
                            "compound_turn_boundary_refinement") else 45.0)
                        refined_outside_direction_gate = bool(
                            (sector == "left" and not
                             side_min_deg <= refined_yaw_deg <= 135.0) or
                            (sector == "right" and not
                             -135.0 <= refined_yaw_deg <= -side_min_deg))
                    if (self._relation_route_guard and
                            relation_direction_gate.get("active")):
                        # V21's local side-route gate applies to requested
                        # refined camera centers as well as existing sectors.
                        # A refinement must not silently reopen the frontal or
                        # rear sector that the first semantic pass rejected.
                        refined_outside_direction_gate = bool(
                            not (30.0 <= abs(refined_yaw_deg) <= 100.0))
                    if (refined.get("point") is None or
                            (refined.get("hard_excluded", False) and not
                             refined.get(
                                 "soft_blocked_tangent_reopened", False)) or
                            (self._history_safe_refinement and
                             refined.get("excluded", False) and
                             not explicit_turn_command) or
                            refined_outside_direction_gate or
                            not refined["ground_anchors"]):
                        view_refinement["fallback_reason"] = (
                            "requested view lacks allowed navigable ground or "
                            "re-enters an incoming/blocked/direction-gated sector")
                    else:
                        candidates.append(refined)
                        selected_index = len(candidates) - 1
                        view_refinement["refinement_accepted"] = True
                        view_refinement["refined_view_index"] = selected_index
            if use_rgb_evidence_refinement:
                candidate = candidates[selected_index]
                height, width = candidate["target_mask"].shape
                if vertical_form:
                    anchors = self._vertical_ground_anchors(
                        candidate["target_mask"], vertical_direction, count=6)
                else:
                    anchor_sampler = (self._task30_ground_anchors
                                      if self._task30_route_anchor else
                                      self._ground_anchors)
                    anchors = anchor_sampler(
                        candidate["target_mask"],
                        preferred=np.array([width / 2, 0.66 * height], np.float32),
                        count=12 if not self._task30_route_anchor else 6)
                anchors = history_safe_anchors(candidate, anchors)
                central_anchors = anchors if self._task30_route_anchor else [
                    anchor for anchor in anchors
                    if 0.30 * width <= float(anchor[0]) <= 0.70 * width]
                if not central_anchors:
                    if not use_center_preferred_fallback:
                        raise RuntimeError(
                            "V14 has no central-40% RGB ground anchor after "
                            "its single allowed refinement")
                    # V15 keeps the central band as a strong RGB preference,
                    # but it must not turn a valid connected DINO+SAM mask
                    # into a missing selection.  The anchor sampler is already
                    # seeded on the horizontal image axis; retain its nearest
                    # legal mask points when the central band has no support.
                    candidate["ground_anchors"] = anchors[:6]
                    view_refinement["central_anchor_fallback"] = (
                        "central 40% has no RGB ground; retained the nearest "
                        "legal anchors from the same selected ground mask")
                else:
                    candidate["ground_anchors"] = central_anchors[:6]
                if side_stop_near_side:
                    # A side-qualified STOP relation is about the landmark's
                    # lateral position at the *arrival node*, not about
                    # driving directly at the landmark.  In the confirmed
                    # left (-45) view, use a legal floor anchor on the image
                    # right; for right (+45), use image left.  The resulting
                    # camera-plus-pixel ray approaches the threshold while
                    # keeping the doorway on the requested side.  This is
                    # purely 2-D mask geometry and remains depth/navmesh/demo
                    # independent.  Put the relation-preserving anchor first
                    # so an anchor-0 VLM choice cannot undo the constraint.
                    height, width = candidate["target_mask"].shape
                    side_left = bool(re.search(
                        r"\bleft\b", str(stage.get(
                            "navigation_instruction", "")).lower()))
                    relation_anchors = [
                        point for point in anchors
                        if ((float(point[0]) >= 0.62 * width) if side_left
                            else (float(point[0]) <= 0.38 * width))]
                    # If the mask is confined to the landmark side (common in
                    # a narrow hallway), there may be no literal opposite-
                    # side pixel.  Fall back to the most lateral legal anchor
                    # and bias it toward the lower/near part of the support
                    # mask.  This still prevents selecting the far/top ray
                    # that would cross a doorway, while remaining valid on a
                    # one-sided floor mask.
                    relation_pool = relation_anchors or list(anchors)
                    if relation_pool:
                        # Couple the near-side floor ray to the detected
                        # portal's image boundary.  A fixed 0.74/0.26 image
                        # prior can place the point beside an unrelated
                        # doorway when several openings are visible.  Use
                        # only compact 2-D detector boxes and keep a small
                        # offset on the camera side of the named portal;
                        # depth/navmesh/demo information is not consulted.
                        portal_terms = {
                            "door", "doorway", "opening", "entrance",
                            "portal", "archway",
                        }
                        portal_boxes = []
                        for record in candidate.get("detection_records", []):
                            label = str(record.get("label", "")).lower()
                            if not any(term in label for term in portal_terms):
                                continue
                            box = record.get("box_xyxy")
                            if not isinstance(box, (list, tuple)) or len(box) != 4:
                                continue
                            x1, _, x2, _ = [float(value) for value in box]
                            center = 0.5 * (x1 + x2)
                            side_match = ((center <= 0.55 * width)
                                          if side_left else
                                          (center >= 0.45 * width))
                            if side_match:
                                portal_boxes.append((
                                    float(record.get("score", 0.0)),
                                    x1, x2, box))
                        selected_portal = max(
                            portal_boxes, key=lambda item: item[0],
                            default=None)
                        if selected_portal is not None:
                            _, box_x1, box_x2, selected_box = selected_portal
                            target_x = (
                                box_x2 + 0.05 * width if side_left else
                                box_x1 - 0.05 * width)
                            target_x = float(np.clip(target_x, 0, width - 1))
                        else:
                            # Without a compact portal box, an extreme
                            # opposite-side prior is unsafe: it can select a
                            # different opening or stairs simply because the
                            # mask is wide.  Keep the ray near the horizontal
                            # centre until a portal is visually grounded;
                            # the VLM still supplies the semantic view choice.
                            target_x = 0.50 * width
                        # A STOP/WAIT endpoint must remain on the camera side
                        # of the landmark.  The ordinary anchor sampler is
                        # intentionally centred around y≈0.66H for general
                        # navigation; for a side-qualified near relation,
                        # prefer legal mask pixels in the lower support band
                        # (y>=0.82H) so the selected ray terminates near the
                        # landmark rather than at a far doorway/interior ray.
                        # This is image-mask geometry only: no depth,
                        # navmesh, demo path, or execution result is used.
                        lower_relation_pool = [
                            point for point in relation_pool
                            if float(point[1]) >= 0.82 * height]
                        if not lower_relation_pool:
                            # The interior sampler can contain no anchor in
                            # the lower band when a doorway mask is thin.
                            # Synthesize one legal pixel from that same
                            # 2-D mask (prefer an interior pixel when
                            # available) instead of falling back to the far
                            # y≈0.66H anchor.  This remains RGB/mask-only.
                            valid = np.asarray(candidate["target_mask"],
                                               dtype=np.uint8)
                            lower = np.zeros_like(valid)
                            lower[int(0.82 * height):] = valid[
                                int(0.82 * height):]
                            distance = cv2.distanceTransform(
                                lower, cv2.DIST_L2, 5)
                            ys, xs = np.nonzero(lower)
                            if len(xs):
                                keep = distance[ys, xs] >= 1.0
                                if not keep.any():
                                    keep = np.ones_like(xs, dtype=bool)
                                ys, xs = ys[keep], xs[keep]
                                dist = distance[ys, xs]
                                order = np.lexsort(
                                    (-dist, -ys,
                                     np.abs(xs - target_x)))
                                lower_relation_pool = [
                                    np.asarray([float(xs[order[0]]),
                                                float(ys[order[0]])],
                                               np.float32)]
                                relation_pool = list(relation_pool) + \
                                    lower_relation_pool
                        if lower_relation_pool:
                            relation_pool = lower_relation_pool
                        preferred_relation = min(
                            relation_pool,
                            key=lambda point: (
                                abs(float(point[0]) - target_x) / max(width, 1),
                                -float(point[1]) / max(height, 1)))
                        if lower_relation_pool:
                            # For a near-side STOP, do not leave farther
                            # sampler anchors in the choice set: the VLM
                            # could otherwise select one of them after the
                            # preferred anchor is prepended.  Every exposed
                            # anchor must satisfy the same lower-band rule.
                            candidate["ground_anchors"] = [
                                preferred_relation] + [
                                    point for point in relation_pool
                                    if not np.array_equal(
                                        point, preferred_relation)][:5]
                        else:
                            candidate["ground_anchors"] = [preferred_relation] + [
                                point for point in candidate["ground_anchors"]
                                if not np.array_equal(point, preferred_relation)]
                        view_refinement["stop_relation_anchor_adjustment"] = {
                            "side": "left" if side_left else "right",
                            "anchor_xy": [float(preferred_relation[0]),
                                          float(preferred_relation[1])],
                            "opposite_side_mask_available": bool(
                                relation_anchors),
                            "policy": (
                                "opposite-side floor anchor when available; "
                                "otherwise lateral lower/near floor anchor "
                                "keeps the named doorway from being crossed"),
                            "minimum_image_y_fraction": 0.82,
                            "lower_support_candidates": len(
                                lower_relation_pool),
                            "portal_box_xyxy": (
                                list(selected_portal[3])
                                if selected_portal is not None else None),
                            "portal_anchor_offset_policy": (
                                "camera_side_of_best_side_matched_portal"
                                if selected_portal is not None else
                                "lateral_fallback_without_compact_portal_box"),
                        }
            if dual_candidate_review:
                allowed = []
                for index in (
                        selected_index, primary_index, alternative_index,
                        view_refinement.get("refined_view_index")):
                    if index is not None and index not in allowed:
                        allowed.append(index)
            else:
                allowed = [selected_index]
            view_refinement["final_candidate_views"] = list(allowed)
            # Re-render the final anchor decision after the semantic review and
            # optional RGB acquisition. V12 deliberately leaves the primary and
            # strongest competitor open for an independent second adjudication.
            views = [
                self._overlay(
                    candidate["rgb"], candidate["target_mask"], index,
                    index not in allowed, candidate.get("semantic_detections"),
                    candidate.get("small_seg_object_mask"),
                    candidate.get("ground_anchors"))
                for index, candidate in enumerate(candidates)
            ]
            annotated_sheet = self._point_selection_contact_sheet(views)
            raw_views = []
            for index, candidate in enumerate(candidates):
                raw = np.asarray(candidate["rgb"]).copy()
                cv2.rectangle(raw, (0, 0), (132, 24), (0, 0, 0), -1)
                cv2.putText(
                    raw,
                    (f"REFINED VIEW {index}" if candidate.get(
                        "is_refined_view") else f"RAW VIEW {index}"),
                    (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (255, 255, 255), 1, cv2.LINE_AA)
                raw_views.append(raw)
            images = [
                self._point_selection_contact_sheet(raw_views),
                annotated_sheet]
            if native_rgb_review:
                # Preserve native per-view detail in the anchor pass as well.
                # Only the semantically confirmed view is exposed, so this pass
                # cannot silently reopen another compass sector.
                images = [raw_views[selected_index], views[selected_index]]
        history = action_history or []
        history_text = json.dumps(history[-20:], ensure_ascii=False) if history else "none (first stage)"
        detection_evidence = {
            str(index): candidate.get("detection_records", [])
            for index, candidate in enumerate(candidates)
        }
        next_context_detection_evidence = {
            str(index): candidate.get("next_context_detection_records", [])
            for index, candidate in enumerate(candidates)
        }
        ground_seg_evidence = {
            str(index): {
                "source": candidate.get("ground_mask_source", "grounded_sam"),
                "detections": candidate.get("ground_detection_records", []),
            }
            for index, candidate in enumerate(candidates)
        }
        ground_anchor_evidence = {
            str(index): [{"anchor_index": anchor_index,
                          "xy_norm": [round(float(point[0] / (candidate['rgb'].shape[1] - 1)), 3),
                                      round(float(point[1] / (candidate['rgb'].shape[0] - 1)), 3)],
                          "pixel_right_deg": round(float(math.degrees(math.atan(
                              (float(point[0]) - (candidate['rgb'].shape[1] - 1) / 2.0) /
                              max(candidate['rgb'].shape[1] / 2.0, 1.0)))), 1)}
                         for anchor_index, point in enumerate(candidate["ground_anchors"])]
            for index, candidate in enumerate(candidates)
        }
        orientation_guidance = ""
        if self.point_selection_prompt_version in {
                "v2_orientation_continuity",
                "v3_orientation_soft_semantic",
                "v4_sector_identity_gates",
                "v5_two_stage_sector_anchor",
                "v6_validated_relation_two_stage",
                "v7_single_stage_relation_gates",
                "v8_object_relation_router",
                "v9_single_stage_identity_relation_gates",
                "v10_approach_relation_router",
                "v11_eight_view_refinement",
                "v12_eight_view_dual_candidate",
                "v13_eight_view_native_rgb",
                "v14_rgb_evidence_refinement",
                "v15_rgb_center_preferred",
                "v16_turn_three_view_gate",
                "v17_turn_three_view_commit_prior",
                "v18_history_safe_refinement",
                "v19_pixel_ray_history_and_reverse_override"}:
            orientation_evidence = {}
            for index, candidate in enumerate(candidates):
                relative_deg = math.degrees(float(candidate.get(
                    "relative_yaw_rad", 0.0)))
                relative_deg = (relative_deg + 180.0) % 360.0 - 180.0
                if abs(relative_deg) < 30:
                    sector = "forward"
                elif 30 <= relative_deg <= 90:
                    sector = "front-left"
                elif -90 <= relative_deg <= -30:
                    sector = "front-right"
                elif relative_deg > 90:
                    sector = "rear-left"
                elif relative_deg < -90:
                    sector = "rear-right"
                else:
                    sector = "rear"
                orientation_evidence[str(index)] = {
                    "relative_yaw_deg": round(relative_deg, 1),
                    "sector": sector,
                    "allowed": index in allowed,
                }
            forward_allowed = [index for index in allowed if abs(
                orientation_evidence[str(index)]["relative_yaw_deg"]) <= 90]
            landmark_turn_guidance = ""
            if str(stage.get("form", "")) == "TURN_TO_LANDMARK":
                landmark_turn_guidance = """
- NAMED-LANDMARK TURN: This clause explicitly says turn toward/to a named
  landmark. Ignore the ordinary forward-hemisphere preference above when it
  conflicts with the landmark's visible bearing. A lateral or rear sector is
  correct if and only if the named landmark is actually identified there and a
  connected floor ray points toward it. The pixel anchor must preserve that
  bearing; central floor in an unrelated sector is not a valid substitute.
"""
            orientation_guidance = f"""
ORIENTATION_AND_CONTINUITY_GUIDANCE:
- All {len(candidates)} views are simultaneous except any explicitly labeled
  REFINED VIEW, and use the robot's arrival heading as 0 degrees.
- In this Habitat camera convention physical left-turn actions use positive
  relative yaw and right-turn actions use negative relative yaw; obey the
  explicit direction gate rather than inferring signs from image pixels.
- View orientation metadata: {json.dumps(orientation_evidence, ensure_ascii=False)}.
- Allowed front-hemisphere views for this call: {forward_allowed}.
- First decide the instruction-consistent direction sector, then select an
  anchor inside that sector. Do not choose a visually attractive landmark in a
  direction that contradicts the instruction.
- For ordinary route continuation (forward, exit, enter, cross, pass, through,
  between, approach, follow, or stop after progress), prefer an allowed
  front-hemisphere view. A rear view needs explicit reversal language or clear,
  unique visual evidence that the instructed route is behind the agent.
- Interpret explicit direction words in this coordinate system: left favors
  positive-yaw views, right favors negative-yaw views, forward favors near 0
  degrees, and turn-around favors near 180 degrees. These are soft navigation
  priors: semantic evidence and walkable-floor validity remain mandatory.
{landmark_turn_guidance}
"""
        semantic_fusion_guidance = ""
        if self.point_selection_prompt_version in {
                "v3_orientation_soft_semantic",
                "v4_sector_identity_gates",
                "v5_two_stage_sector_anchor",
                "v6_validated_relation_two_stage",
                "v7_single_stage_relation_gates",
                "v8_object_relation_router",
                "v9_single_stage_identity_relation_gates",
                "v10_approach_relation_router",
                "v11_eight_view_refinement",
                "v12_eight_view_dual_candidate",
                "v13_eight_view_native_rgb",
                "v14_rgb_evidence_refinement",
                "v15_rgb_center_preferred",
                "v16_turn_three_view_gate",
                "v17_turn_three_view_commit_prior",
                "v18_history_safe_refinement",
                "v19_pixel_ray_history_and_reverse_override"}:
            semantic_policy_evidence = {
                str(index): {
                    "relation_mode": candidate.get(
                        "strategy_application", {}).get("mode"),
                    "cross_view_status": candidate.get(
                        "strategy_application", {}).get(
                            "cross_view_candidate_status"),
                    "detector_evidence_count": len(candidate.get(
                        "detection_records", [])),
                } for index, candidate in enumerate(candidates)
            }
            semantic_fusion_guidance = f"""
SOFT_CROSS_VIEW_SEMANTIC_EVIDENCE (prompt version v3):
- Per-view evidence status: {json.dumps(semantic_policy_evidence, ensure_ascii=False)}.
- Object/portal detections are useful but fallible and may be missing in a
  correct view or spuriously present in a wrong view. Missing detector evidence
  does not invalidate otherwise valid walkable ground.
- Prefer detector-supported views when their direction and visible scene agree
  with the instruction. Do not reverse route progress solely because a rear
  view has a detection while a forward view lacks one.
- Resolve conflicts by jointly checking instruction direction, route
  continuity, raw visual evidence, detection label/geometry, and ground access.
"""
        sector_identity_guidance = ""
        if (self.point_selection_prompt_version in {
                "v4_sector_identity_gates",
                "v9_single_stage_identity_relation_gates",
                "v11_eight_view_refinement",
                "v12_eight_view_dual_candidate",
                "v13_eight_view_native_rgb",
                "v14_rgb_evidence_refinement",
                "v15_rgb_center_preferred",
                "v16_turn_three_view_gate",
                "v17_turn_three_view_commit_prior",
                "v18_history_safe_refinement",
                "v19_pixel_ray_history_and_reverse_override"} or
                use_two_stage):
            sector_identity_guidance = f"""
SECTOR_THEN_IDENTITY_GATES (prompt version v4):
- Explicit direction gate: {json.dumps(direction_gate, ensure_ascii=False)}.
- Validated portal-floor relation gate: {json.dumps(relation_gate, ensure_ascii=False)}.
- Make the decision in this order: (1) obey any explicit turn sector gate,
  (2) identify the actual named landmark/room/portal from raw RGB and surrounding
  context, (3) verify that the required spatial relation is possible in that
  view, and only then (4) choose a numbered ground anchor.
- A detector label is a proposal, not proof. Downweight a box/mask that covers
  most of the image, visibly lies on the wrong object class, repeats the whole
  verb phrase as its label, or conflicts with raw visual appearance. Never claim
  that a sink, stair, atrium, hallway, couch, table, door, or other landmark is
  visible unless its shape and scene context are actually visible in the RGB.
- Compare all allowed views for landmark identity. A generic opening is not the
  named destination when another view has stronger room/landmark context.
- Do not default to view 0 merely because the instruction says walk/go: the
  target may initially be beside or behind the camera. Conversely, do not chase
  a rear detector hit when the raw identity is weak.
- In the reason, name the raw visual cue and spatial relation that distinguish
  the selected view from the strongest competing allowed view.
"""
        strict_heading_guidance = ""
        if self.point_selection_prompt_version in eight_view_versions:
            if (self.point_selection_prompt_version ==
                    "v12_eight_view_dual_candidate"):
                strict_heading_guidance = """
STRICT_45_DEGREE_GROUND_RAY_AND_SECOND_ADJUDICATION:
- The first review supplied a primary proposal and its strongest semantic
  competitor. Independently compare their raw RGB identity, instruction
  relation, route continuity, and visible connected ground. You MUST overrule
  the primary proposal when the competitor has stronger evidence; confidence
  or detector text from the first pass is not proof.
- After choosing the better semantic view, prefer a route-consistent ground
  anchor in the central 40% of that image. Horizontal pixel offset changes the
  final 3D heading, so an edge anchor can violate the 45-degree requirement.
- Use only RGB appearance, DINO+SAM image masks and 2D geometry.
"""
            else:
                strict_heading_guidance = """
STRICT_45_DEGREE_GROUND_RAY:
- The semantic direction has already been confirmed by the eight-view review.
  Do not reconsider another view in this anchor pass.
- The final navigation-ray heading includes horizontal pixel offset inside the
  camera. Prefer a route-consistent ground anchor in the central 40% of the
  confirmed image. An edge anchor can turn a correct camera sector into a
  greater-than-45-degree navigation error.
- Among central valid anchors, prefer visually farther connected ground along the route;
  do not trade direction accuracy for a larger mask or nearer floor.
"""
            if vertical_form:
                strict_heading_guidance += f"""
VERTICAL_ENDPOINT_ANCHOR_RULE:
- This stage is a {str(stage.get('form', 'VERTICAL'))} transition.  The semantic
  endpoint is the top/bottom landing, not the nearest visible stair tread.
- Compare the numbered anchors in the confirmed stair view along the visible
  connected flight.  For VERTICAL_UP prefer the highest (smallest image-y)
  connected floor anchor that still lies on a tread/landing; for VERTICAL_DOWN
  prefer the lowest (largest image-y) one.  A midpoint is allowed only when the
  instruction explicitly says halfway/middle/partway.
- If the landing is not segmented, select the furthest visible connected tread
  as an explicit intermediate waypoint; the arrival judge must then require a
  later level-landing observation before marking the vertical instruction
  complete.  Do not stop at a convenient central tread merely because it is
  well centered.  This is RGB/2-D mask geometry only; never use depth, navmesh,
  demonstration paths, or hidden execution outcomes.
"""
            if (self._task30_route_anchor and
                    str(stage.get("form", "")) == "TURN_TO_LANDMARK"):
                strict_heading_guidance += """
NAMED-LANDMARK BEARING OVERRIDE:
- For an explicit turn toward/to a named landmark, the landmark bearing is the
  route. Do not force the anchor into the central 40% when that would rotate the
  final camera-plus-pixel ray away from the identified landmark. Select the
  same-bearing connected floor anchor, including a justified lateral/rear view,
  and keep the landmark visible in the clean RGB evidence.
- This is an orientation stage, not an approach stage. Among same-bearing
  connected anchors prefer a safe lower-image/local floor anchor; do not choose
  a distant point merely to get closer to the landmark or execute the following
  clause. The arrived node must re-observe the landmark before completion.
"""
        sector_selection = None
        if use_two_stage:
            sector_schema = {
                "type": "object",
                "additionalProperties": False,
                "required": ["view_index", "reason"],
                "properties": {
                    "view_index": {"type": "integer", "enum": list(allowed)},
                    "reason": {"type": "string", "maxLength": 320},
                },
            }
            sector_prompt = f"""GROUND_TARGET_SECTOR_SELECTION
Point-selection prompt version: {self.point_selection_prompt_version}.
This first pass selects only a semantic direction; it does not select a pixel.
Current stage: {stage['navigation_instruction']}
Landmark: {stage['landmark']}.
Semantic spatial target: {stage['semantic_spatial_target']}.
Required spatial relation: {stage['spatial_relation']}.
Visual arrival evidence: {stage['visual_arrival_evidence']}.
Forbidden target: {stage['forbidden_target']}.
Allowed views: {allowed}.
Orientation metadata: {json.dumps(orientation_evidence, ensure_ascii=False)}.
Explicit direction gate: {json.dumps(direction_gate, ensure_ascii=False)}.
V21 relation-direction gate: {json.dumps(relation_direction_gate, ensure_ascii=False)}.
Validated portal-floor relation gate: {json.dumps(relation_gate, ensure_ascii=False)}.
Detector proposals by view: {json.dumps(detection_evidence, ensure_ascii=False)}.
You receive one CLEAN RGB six-view contact sheet. First identify the actual named
landmark, room, portal, stairs, or route from raw appearance and scene context.
Choose the allowed view that contains or most directly leads to the required
semantic spatial target. Detector text is only a proposal: ignore whole-image,
wrong-class, verb-phrase, or visually contradicted detections. A generic opening
is not a named destination when another view has stronger identity evidence.
Do not select based on floor area or anchor placement; those are handled later.
Return only requested JSON."""

            def validate_sector(result):
                view_index = int(result["view_index"])
                if view_index not in allowed:
                    raise ValueError(f"view_index must be one of {allowed}")
                return view_index, str(result.get("reason", ""))

            sector_view, sector_reason = self._call(
                "select_ground_sector", sector_prompt, [images[0]],
                sector_schema, validate_sector)
            sector_selection = {
                "view_index": int(sector_view),
                "reason": sector_reason,
                "allowed_views_before_sector_selection": list(allowed),
                "relation_gate": relation_gate,
            }
            allowed = [int(sector_view)]
            sector_identity_guidance += f"""
TWO_STAGE_SECTOR_RESULT:
- The clean-RGB semantic sector pass selected view {sector_view} because:
  {sector_reason}
- This second pass must only select the best numbered ground anchor in that
  fixed view. Do not reopen or reconsider another direction.
"""
        # V7 deliberately preserves the exact V3 model-facing prompt for all
        # ungated cases.  Its improvement is deterministic candidate validation,
        # not a second stochastic model decision or prompt retuning.
        model_prompt_version = (
            "v3_orientation_soft_semantic"
            if self.point_selection_prompt_version in {
                "v7_single_stage_relation_gates",
                "v8_object_relation_router",
                "v10_approach_relation_router",
            } and not use_two_stage
            else self.point_selection_prompt_version)
        # The taxonomy still retains historical strategy prose mentioning
        # depth, but no point-selection prompt version may receive or request
        # it after the project-wide RGB-only rule.
        selection_recipe = {
            "input_policy": "RGB only; depth is forbidden",
            "ground_candidates": (
                "DINO+SAM floor/carpet/rug masks in image space"),
            "semantic_evidence": (
                "clean RGB appearance plus DINO+SAM detector boxes/masks"),
            "ranking": (
                "instruction relation, verified identity, camera-center "
                "alignment, connected visible ground"),
        }
        final_ground_surface_sanity = (
            "GROUND-MASK SANITY: The green mask is a fallible proposal. "
            "Cross-check IMAGE 1 clean RGB and reject every anchor visibly "
            "on a countertop, table, bed, sofa/seat, shelf, wall, raised "
            "platform face, or other object surface. A valid anchor lies on "
            "floor/rug/carpet that plausibly connects toward the camera or "
            "the base of the intended opening. If this confirmed view has no "
            "such anchor, do not reinterpret an object top as floor."
            if self.point_selection_prompt_version in {
                "v18_history_safe_refinement",
                "v19_pixel_ray_history_and_reverse_override"} else "")
        if first_step_route_guard_active:
            final_ground_surface_sanity = (
                f"{final_ground_surface_sanity}\n{first_step_route_guidance}")
        prompt = f"""GROUND_TARGET_SELECTION
Point-selection prompt version: {model_prompt_version}.
You control an indoor R2R agent. Current stage: {stage['navigation_instruction']}
Landmark: {stage['landmark']}. Completion cue: {stage['completion_cue']}.
Semantic spatial target: {stage['semantic_spatial_target']}.
Required spatial relation: {stage['spatial_relation']}.
Visual arrival evidence: {stage['visual_arrival_evidence']}.
Forbidden target regions: {stage['forbidden_target']}.
Perception and point-selection recipe: {json.dumps(selection_recipe, ensure_ascii=False)}.
Open-vocabulary detection + instance-mask evidence by view: {json.dumps(detection_evidence, ensure_ascii=False)}.
Next-clause detector evidence by view (tie-breaker only, never the current target): {json.dumps(next_context_detection_evidence, ensure_ascii=False)}.
DINO+SAM floor/carpet/rug segmentation evidence by view: {json.dumps(ground_seg_evidence, ensure_ascii=False)}.
Numbered valid ground anchors by view: {json.dumps(ground_anchor_evidence, ensure_ascii=False)}.
Actions used to reach the previous stage target: {history_text}.
{orientation_guidance}
{semantic_fusion_guidance}
{sector_identity_guidance}
{strict_heading_guidance}
{final_ground_surface_sanity}
{relation_route_guidance}
EIGHT_VIEW_REVIEW_RESULT: {json.dumps(view_refinement, ensure_ascii=False) if view_refinement is not None else 'not used'}.
You receive one 3-column x 2-row contact sheet containing forward views 0..5.
Green overlay is the DINO+SAM floor/carpet/rug candidate region;
colored boxes/masks are instruction-related instances. No generic object
segmentation is used in this experiment.
Choose one allowed view from {allowed} that
contains or leads most directly to the semantic spatial target, then choose a
navigable point ON the green floor that satisfies the stated spatial relation.
HARD REQUIREMENT: select one of the numbered white anchors visibly inside the
green region. Return its view_index and anchor_index. Do not invent coordinates.
Colored masks and boxes are detected semantic references; use their label and
image-space geometry as evidence, but place the navigation point on green floor,
not inside an object mask. If the instruction says pass/beyond/through, choose
floor beyond the matched reference rather than merely approaching its box.
x_norm/y_norm are normalized image
coordinates from top-left. Prefer a point well inside the mask, not an object,
wall, image edge, or the near-camera bottom. Return only requested JSON."""
        if (self.point_selection_prompt_version in {
                "v4_sector_identity_gates",
                "v9_single_stage_identity_relation_gates",
                "v11_eight_view_refinement",
                "v12_eight_view_dual_candidate",
                "v13_eight_view_native_rgb",
                "v14_rgb_evidence_refinement",
                "v15_rgb_center_preferred",
                "v16_turn_three_view_gate",
                "v17_turn_three_view_commit_prior",
                "v18_history_safe_refinement",
                "v19_pixel_ray_history_and_reverse_override"} or
                use_two_stage):
            if self.point_selection_prompt_version in {
                    "v13_eight_view_native_rgb",
                    "v14_rgb_evidence_refinement",
                    "v15_rgb_center_preferred",
                    "v16_turn_three_view_gate",
                    "v17_turn_three_view_commit_prior",
                    "v18_history_safe_refinement",
                    "v19_pixel_ray_history_and_reverse_override"}:
                image_description = (
                    "You receive two native-resolution images of the single "
                    "confirmed direction. IMAGE 1 is its clean RGB view. IMAGE 2 "
                    "is the aligned DINO+SAM/detector/anchor overlay. Do not "
                    "reconsider another compass direction; establish identity "
                    "from IMAGE 1 and select a valid ground anchor in IMAGE 2.")
            else:
                image_description = (
                    f"You receive two aligned grid contact sheets for views "
                    f"0..{len(candidates) - 1}. IMAGE 1 is clean RGB for "
                    "landmark/room identity. IMAGE 2 is the annotated "
                    "DINO+SAM/detector/anchor sheet. Establish identity "
                    "from IMAGE 1, then use IMAGE 2 to select a valid ground "
                    "anchor.")
            prompt = prompt.replace(
                "You receive one 3-column x 2-row contact sheet containing forward views 0..5.",
                image_description)
        prompt += (
            "\nDEPTH-PROHIBITION: Make this anchor decision from RGB, "
            "DINO+SAM image masks, and 2D anchor positions only. No "
            "depth, 3D distance, navmesh, or demonstration path is exposed."
        )

        def validate(result):
            view_index = int(result["view_index"])
            if view_index not in allowed:
                raise ValueError(f"view_index must be one of {allowed}")
            candidate = candidates[view_index]
            anchor_index = int(result["anchor_index"])
            anchors = candidate["ground_anchors"]
            if not 0 <= anchor_index < len(anchors):
                raise ValueError(f"anchor_index must be in [0,{len(anchors)-1}]")
            point = np.asarray(anchors[anchor_index], np.float32)
            supported_partial_endpoint_repair = None
            if (route_continuation and len(anchors) > 1 and
                    str(stage.get("form", "")).upper() not in {
                        "APPROACH_LANDMARK", "STOP_WAIT",
                        "VERTICAL_UP", "VERTICAL_DOWN"}):
                # This is the only lookahead allowed after a supported partial
                # edge.  Use the farthest visible point on the same 2-D lane
                # so it has a fair chance to reach the semantic boundary.
                # Restrict x displacement to preserve the VLM-selected lane;
                # only RGB mask geometry is consulted.
                height, width = candidate["target_mask"].shape
                same_lane = [
                    (index, np.asarray(anchor, np.float32))
                    for index, anchor in enumerate(anchors)
                    if abs(float(anchor[0]) - float(point[0])) <= 0.25 * width]
                endpoint_index, endpoint_point = min(
                    same_lane, key=lambda item: float(item[1][1]))
                if (endpoint_index != anchor_index and
                        float(point[1]) - float(endpoint_point[1]) >=
                        0.08 * height):
                    original_anchor_index = anchor_index
                    anchor_index = int(endpoint_index)
                    point = endpoint_point
                    supported_partial_endpoint_repair = {
                        "status": "far_visible_same_lane_anchor",
                        "original_anchor_index": original_anchor_index,
                        "repaired_anchor_index": anchor_index,
                        "policy": (
                            "the single supported-partial lookahead targets "
                            "the farthest visible anchor on its selected lane"),
                    }
            vertical_endpoint_repair = None
            if vertical_form:
                # For an endpoint clause ("to the top/bottom"), a middle
                # tread is only an intermediate waypoint.  Prefer the most
                # advanced visible connected floor anchor in the requested
                # image-space direction.  Midway/halfway clauses intentionally
                # keep the VLM-selected interior anchor.
                stage_text = " ".join(str(stage.get(key, "")) for key in (
                    "navigation_instruction", "semantic_spatial_target",
                    "spatial_relation", "completion_cue")).lower()
                partial_vertical = bool(re.search(
                    r"\b(?:half(?:way)?|middle|midway|part(?:way)?|some of the way)\b",
                    stage_text))
                if not partial_vertical and len(anchors) > 1:
                    ys = np.asarray([float(anchor[1]) for anchor in anchors])
                    endpoint_index = int(np.argmin(ys) if vertical_direction ==
                                         "up" else np.argmax(ys))
                    endpoint_y = float(ys[endpoint_index])
                    selected_y = float(point[1])
                    interior_limit = float(np.quantile(ys, 0.50))
                    is_interior = (selected_y > interior_limit + 1e-3
                                   if vertical_direction == "up" else
                                   selected_y < interior_limit - 1e-3)
                    if is_interior and endpoint_index != anchor_index:
                        original_anchor_index = anchor_index
                        anchor_index = endpoint_index
                        point = np.asarray(anchors[anchor_index], np.float32)
                        vertical_endpoint_repair = {
                            "status": "furthest_visible_vertical_anchor",
                            "direction": vertical_direction,
                            "original_anchor_index": original_anchor_index,
                            "repaired_anchor_index": anchor_index,
                            "original_y": selected_y,
                            "repaired_y": endpoint_y,
                            "policy": (
                                "endpoint stages use the furthest visible "
                                "connected stair/landing anchor; partial "
                                "vertical stages retain the model anchor"),
                        }
            portal_endpoint_repair = None
            if (str(stage.get("form", "")) in {
                    "EXIT_REGION", "ENTER_REGION", "SELECT_PORTAL",
                    "TRAVERSE_PORTAL_REGION"} and len(anchors) > 1):
                # A portal/region endpoint should be beyond the threshold,
                # not the nearest camera-side floor pixel.  The anchor set is
                # already restricted to connected RGB ground; select its
                # visually farther (smaller-y) member when the VLM picked a
                # clearly near/interior anchor.  This remains a generic 2-D
                # relation safeguard and uses no depth, navmesh, demo path,
                # scene, or episode identity.
                ys = np.asarray([float(anchor[1]) for anchor in anchors])
                endpoint_index = int(np.argmin(ys))
                endpoint_y = float(ys[endpoint_index])
                selected_y = float(point[1])
                if (endpoint_index != anchor_index and
                        selected_y > float(np.median(ys)) + 1e-3):
                    original_anchor_index = anchor_index
                    anchor_index = endpoint_index
                    point = np.asarray(anchors[anchor_index], np.float32)
                    portal_endpoint_repair = {
                        "status": "furthest_visible_portal_floor_anchor",
                        "original_anchor_index": original_anchor_index,
                        "repaired_anchor_index": anchor_index,
                        "original_y": selected_y,
                        "repaired_y": endpoint_y,
                        "policy": (
                            "portal endpoint uses the farthest visible "
                            "connected RGB floor anchor"),
                    }
            turn_endpoint_repair = None
            if (str(stage.get("form", "")) in {
                    "TURN_LEFT", "TURN_RIGHT", "TURN_AROUND"} and
                    len(anchors) > 1):
                # A turn target should enter the branch, not stop at the
                # nearest side pixel that merely induces a yaw change.  Use
                # the farthest visible connected floor anchor in the chosen
                # sector when the model selected a clearly near anchor.  The
                # test is image-space only and keeps the explicit directional
                # gate; no depth/navmesh/demo information is exposed here.
                ys = np.asarray([float(anchor[1]) for anchor in anchors])
                endpoint_index = int(np.argmin(ys))
                selected_y = float(point[1])
                if (endpoint_index != anchor_index and
                        selected_y > float(np.median(ys)) + 1e-3):
                    original_anchor_index = anchor_index
                    anchor_index = endpoint_index
                    point = np.asarray(anchors[anchor_index], np.float32)
                    turn_endpoint_repair = {
                        "status": "branch_entry_far_floor_anchor",
                        "original_anchor_index": original_anchor_index,
                        "repaired_anchor_index": anchor_index,
                        "original_y": selected_y,
                        "repaired_y": float(ys[endpoint_index]),
                        "policy": "turn enters the far visible branch floor",
                    }
            # A +/-45-degree semantic sector and an opposite-side pixel
            # anchor form a compound ray that can leave the accepted sector by
            # more than the task-30 tolerance.  For object-between/through
            # relations, deterministically move to the nearest same-side
            # connected floor anchor when the VLM selected the wrong side of
            # the confirmed view.  This is a 2-D consistency safeguard only;
            # it does not use depth, navmesh, or demonstration geometry.
            same_side_repair = None
            if self._task30_route_anchor and str(stage.get("form", "")) == "BETWEEN_OBJECTS":
                center_x = (candidate["rgb"].shape[1] - 1.0) / 2.0
                relative_yaw = math.degrees(float(
                    candidate.get("relative_yaw_rad", 0.0)))
                relative_yaw = (relative_yaw + 180.0) % 360.0 - 180.0
                if abs(relative_yaw) >= 30.0 and abs(relative_yaw) <= 75.0:
                    side_sign = 1.0 if relative_yaw < 0.0 else -1.0
                    same_side = [
                        (index, np.asarray(anchor, np.float32))
                        for index, anchor in enumerate(anchors)
                        if side_sign * (float(anchor[0]) - center_x) >
                        0.03 * candidate["rgb"].shape[1]
                    ]
                    selected_side = side_sign * (float(point[0]) - center_x)
                    if same_side and selected_side < 0.0:
                        repaired_index, repaired_point = min(
                            same_side,
                            key=lambda item: abs(float(item[1][0]) - center_x))
                        same_side_repair = {
                            "status": "same_side_anchor",
                            "original_anchor_index": anchor_index,
                            "repaired_anchor_index": int(repaired_index),
                            "relative_yaw_deg": round(relative_yaw, 2),
                        }
                        anchor_index, point = int(repaired_index), repaired_point
            relation_side_repair = None
            if (self._task30_route_anchor and
                    str(stage.get("form", "")) == "CIRCUMNAVIGATE"):
                # A side camera sector and a pixel anchor form one compound
                # navigation ray.  For a left-side sector, an anchor on the
                # image's right half rotates the ray back toward the front (or
                # the opposite branch); the converse holds for a right-side
                # sector.  Keep the selected semantic sector but repair only
                # this 2-D sign inconsistency to the nearest connected anchor
                # on the same side.  No depth, path, or episode state is used.
                width = candidate["rgb"].shape[1]
                center_x = (width - 1.0) / 2.0
                relative_yaw = math.degrees(float(
                    candidate.get("relative_yaw_rad", 0.0)))
                relative_yaw = (relative_yaw + 180.0) % 360.0 - 180.0
                if 30.0 <= abs(relative_yaw) <= 135.0:
                    desired_sign = -1.0 if relative_yaw > 0.0 else 1.0
                    same_side = [
                        (index, np.asarray(anchor, np.float32))
                        for index, anchor in enumerate(anchors)
                        if desired_sign * (float(anchor[0]) - center_x) >
                        0.03 * width
                    ]
                    selected_side = desired_sign * (float(point[0]) - center_x)
                    if same_side and selected_side < 0.0:
                        repaired_index, repaired_point = min(
                            same_side,
                            key=lambda item: abs(float(item[1][0]) - center_x))
                        relation_side_repair = {
                            "status": "same_side_circumnavigation_anchor",
                            "original_anchor_index": anchor_index,
                            "repaired_anchor_index": int(repaired_index),
                            "relative_yaw_deg": round(relative_yaw, 2),
                        }
                        anchor_index, point = int(repaired_index), repaired_point
            circum_endpoint_repair = None
            if (self._task30_route_anchor and
                    str(stage.get("form", "")) == "CIRCUMNAVIGATE"):
                # ``around/backside`` is an endpoint relation, not merely a
                # lateral approach.  A valid side sector can still contain a
                # near-field floor patch alongside the landmark.  Prefer the
                # visually far end of that same connected 2-D lane, while
                # preserving the selected sector.  The preferred image side
                # compensates the compound camera-ray geometry: at a shallow
                # 45-degree sector the inner-side pixel continues the turn;
                # at a 90-degree sector the outer-side pixel brings the ray
                # back toward the forward route.  This uses only anchor x/y,
                # image dimensions, and the instruction form—never depth,
                # navmesh, demonstration path, or episode identity.
                stage_text = " ".join(str(stage.get(key, "")) for key in (
                    "navigation_instruction", "semantic_spatial_target",
                    "spatial_relation", "completion_cue")).lower()
                partial_circum = bool(re.search(
                    r"\b(?:half(?:way)?|middle|midway|part(?:way)?|alongside)\b",
                    stage_text))
                if not partial_circum and len(anchors) > 1:
                    width, height = candidate["rgb"].shape[1:3]
                    center_x = (width - 1.0) / 2.0
                    center_y = (height - 1.0) / 2.0
                    relative_yaw = math.degrees(float(
                        candidate.get("relative_yaw_rad", 0.0)))
                    relative_yaw = (relative_yaw + 180.0) % 360.0 - 180.0
                    if abs(relative_yaw) >= 15.0:
                        sector_sign = 1.0 if relative_yaw > 0.0 else -1.0
                        image_sign = (
                            -sector_sign if abs(relative_yaw) < 67.5
                            else sector_sign)
                        side_anchors = [
                            (index, np.asarray(anchor, np.float32))
                            for index, anchor in enumerate(anchors)
                            if image_sign * (float(anchor[0]) - center_x) >
                            0.02 * width]
                        if not side_anchors:
                            side_anchors = [
                                (index, np.asarray(anchor, np.float32))
                                for index, anchor in enumerate(anchors)]
                        def endpoint_score(item):
                            _, anchor = item
                            side_extent = image_sign * (
                                float(anchor[0]) - center_x) / max(width, 1)
                            far_extent = (center_y - float(anchor[1])) / max(
                                height, 1)
                            return 0.8 * side_extent + 1.2 * far_extent
                        endpoint_index, endpoint_point = max(
                            side_anchors, key=endpoint_score)
                        current_score = endpoint_score((anchor_index, point))
                        best_score = endpoint_score((endpoint_index,
                                                     endpoint_point))
                        if (endpoint_index != anchor_index and
                                best_score > current_score + 0.03):
                            original_anchor_index = anchor_index
                            anchor_index = int(endpoint_index)
                            point = endpoint_point
                            circum_endpoint_repair = {
                                "status": "far_end_circumnavigation_anchor",
                                "original_anchor_index": original_anchor_index,
                                "repaired_anchor_index": anchor_index,
                                "relative_yaw_deg": round(relative_yaw, 2),
                                "image_side": (
                                    "left" if image_sign < 0 else "right"),
                                "policy": (
                                    "same-sector far visual lane endpoint; "
                                    "shallow sectors use inner-side pixels "
                                    "and right-angle sectors outer-side pixels"),
                            }
            turn_center_repair = None
            if (self._stage1_turn_soft and str(stage.get("form", "")) in {
                    "TURN_LEFT", "TURN_RIGHT"}):
                # The camera sector carries the semantic turn; an extreme
                # horizontal anchor can rotate the resulting ground ray back
                # toward a wall or a neighboring branch.  Prefer the nearest
                # connected central anchor when one exists, without using
                # depth, navmesh, or demonstration information.
                width = candidate["rgb"].shape[1]
                central = [
                    (index, np.asarray(anchor, np.float32))
                    for index, anchor in enumerate(anchors)
                    if 0.40 * width <= float(anchor[0]) <= 0.60 * width]
                if central and not (0.40 * width <= float(point[0]) <=
                                    0.60 * width):
                    repaired_index, repaired_point = min(
                        central,
                        key=lambda item: abs(float(item[1][0]) - width / 2.0))
                    turn_center_repair = {
                        "status": "central_turn_anchor",
                        "original_anchor_index": anchor_index,
                        "repaired_anchor_index": int(repaired_index),
                        "policy": "nearest connected central anchor",
                    }
                    anchor_index, point = int(repaired_index), repaired_point
            return view_index, point, {
                "anchor_index": anchor_index, "requested_xy": point.tolist(),
                "snapped_xy": point.tolist(), "requested_on_ground": True,
                "reason": str(result.get("reason", "")), "allowed_views": allowed,
                "direction_gate": direction_gate,
                "relation_direction_gate": relation_direction_gate,
                "relation_gate": relation_gate,
                "sector_selection": sector_selection,
                "view_refinement": view_refinement,
                "selection_input_policy": (
                    "rgb_only_depth_prohibited"),
                "point_selection_prompt_version": (
                    self.requested_point_selection_prompt_version),
                "route_corridor_bend_fallback": candidate.get(
                    "route_corridor_bend_fallback"),
                # A V21 point may be checked against a connected local floor
                # neighbor after the RGB decision.  This is an execution
                # safety fallback only; it is never exposed to the VLM.
                "postselection_repair_max_view_delta_deg": (
                    0.0 if bool((view_refinement or {}).get(
                        "bare_turn_nominal_ray_adjustment")) else
                    # A seam refinement between two legal bare-turn sectors
                    # may land at +/-112.5 degrees while the only reachable
                    # floor is in the +/-45-degree endpoint sector.  All
                    # three audited views are still explicitly on the
                    # commanded side, so allow the safety repair to search
                    # that complete side set instead of terminating the hop.
                    90.0 if bool((stage.get("metadata", {}) or {}).get(
                        "bare_turn_route_setup", {}).get("active")) else
                    30.0 if self._task30_route_anchor else 45.0),
                "task30_same_side_anchor_repair": same_side_repair,
                "circumnavigation_side_anchor_repair": relation_side_repair,
                "circumnavigation_endpoint_repair": circum_endpoint_repair,
                "stage1_turn_center_repair": turn_center_repair,
                "vertical_endpoint_anchor_repair": vertical_endpoint_repair,
                "portal_endpoint_anchor_repair": portal_endpoint_repair,
                "turn_endpoint_anchor_repair": turn_endpoint_repair,
                "supported_partial_endpoint_repair": (
                    supported_partial_endpoint_repair),
                "semantic_reference_image_used": bool(
                    semantic_reference_rgb is not None),
            }

        point_schema = json.loads(json.dumps(self.POINT_SCHEMA))
        point_schema["properties"]["view_index"]["enum"] = allowed
        return self._call("select_ground_target", prompt, images, point_schema, validate)

    def select_backtrack_ground_target(
            self, target_node_id, target_node_views, current_node_id,
            candidates, forward_action_history=None, recommended_view=None):
        """Choose a current-view ground anchor leading to a stored prior node."""
        if len(target_node_views) != 6 or len(candidates) != 6:
            raise ValueError("node backtracking requires two complete six-view panoramas")
        allowed = [index for index, candidate in enumerate(candidates)
                   if candidate.get("point") is not None and
                   candidate.get("backtrack_allowed", True)]
        direction_gate_relaxed = False
        if not allowed:
            # When the path trace points at a wall and only a side floor view
            # exists, retain exactly the least-wrong floor-bearing view.  This
            # is explicit in the call record instead of silently reopening all
            # six directions.
            usable = [index for index, candidate in enumerate(candidates)
                      if candidate.get("point") is not None]
            if not usable:
                raise RuntimeError(
                    "No floor-bearing current view for node backtracking")
            allowed = [min(usable, key=lambda index: float(
                candidates[index].get("yaw_error_rad", math.inf)))]
            direction_gate_relaxed = True
        for candidate in candidates:
            candidate["ground_anchors"] = self._ground_anchors(
                candidate["target_mask"], candidate.get("point"))
        target_sheet = self._contact_sheet([
            self._overlay(rgb, np.zeros(rgb.shape[:2], bool), index, False)
            for index, rgb in enumerate(target_node_views)
        ])
        current_sheet = self._contact_sheet([
            self._overlay(candidate["rgb"], candidate["target_mask"], index,
                          index not in allowed,
                          ground_anchors=candidate["ground_anchors"])
            for index, candidate in enumerate(candidates)
        ])
        two_node_sheet = np.concatenate([target_sheet, current_sheet], axis=0)
        evidence = {
            str(index): {
                "absolute_yaw_rad": round(float(candidate["yaw"]), 4),
                "visual_similarity_to_target_panorama": round(
                    float(candidate.get("visual_similarity", 0.0)), 4),
                "graph_direction_score": round(
                    float(candidate.get("direction_score", 0.0)), 4),
                "breadcrumb_direction_error_deg": round(math.degrees(
                    float(candidate.get("yaw_error_rad", math.pi))), 2),
                "passes_reverse_route_gate": bool(candidate.get(
                    "backtrack_allowed", True)),
                "ground_fraction": round(float(candidate.get("ground_fraction", 0.0)), 4),
            } for index, candidate in enumerate(candidates)
        }
        anchor_evidence = {
            str(index): [{"anchor_index": anchor_index,
                          "xy": np.asarray(point).round(1).tolist()}
                         for anchor_index, point in enumerate(
                             candidate["ground_anchors"])]
            for index, candidate in enumerate(candidates)
        }
        history = json.dumps(
            (forward_action_history or [])[-20:], ensure_ascii=False)
        recommended = (int(recommended_view) if recommended_view is not None
                       else allowed[0])
        prompt = f"""BACKTRACK_GROUND_TARGET_SELECTION
You must navigate from current graph node {current_node_id} back to stored node
{target_node_id}. You receive ONE combined image. Its TOP half is the TARGET
node's stored six-view panorama in a 3-column x 2-row layout (views 0..5). Its
BOTTOM half is the CURRENT node panorama in the same layout. Green regions and
numbered white anchors in the BOTTOM/current half are valid DINO+SAM
walkable-floor targets.
Original forward action history from target node to the later node:
{history}
Current-view evidence: {json.dumps(evidence, ensure_ascii=False)}
Current-view anchors: {json.dumps(anchor_evidence, ensure_ascii=False)}
Recommended current view from graph geometry: {recommended}
Compare both panoramas and reverse the original route. Select one allowed
CURRENT view from {allowed} which leads toward the target node, then select a
numbered floor anchor in that view. The selected point must be in the BOTTOM
half, never the TOP reference half. The allowed-view list is a HARD online
reverse-route gate computed from the robot's own forward action poses. Use
visual correspondence only to disambiguate allowed views and anchors; never
override that gate with panorama appearance. Return JSON only."""

        def validate(result):
            view_index = int(result["view_index"])
            if view_index not in allowed:
                raise ValueError(f"view_index must be one of {allowed}")
            anchors = candidates[view_index]["ground_anchors"]
            anchor_index = int(result["anchor_index"])
            if not 0 <= anchor_index < len(anchors):
                raise ValueError(f"anchor_index must be in [0,{len(anchors)-1}]")
            point = np.asarray(anchors[anchor_index], np.float32)
            return view_index, point, {
                "selector": "vlm_two_node_six_view",
                "target_node_id": str(target_node_id),
                "current_node_id": str(current_node_id),
                "anchor_index": anchor_index,
                "requested_xy": point.tolist(),
                "snapped_xy": point.tolist(),
                "reason": str(result.get("reason", "")),
                "allowed_views": allowed,
                "recommended_view": recommended,
                "direction_gate_relaxed": direction_gate_relaxed,
            }

        schema = json.loads(json.dumps(self.POINT_SCHEMA))
        schema["properties"]["view_index"]["enum"] = allowed
        return self._call(
            "select_backtrack_ground_target", prompt,
            [two_node_sheet], schema, validate)

    def select_backtrack_ground_target_rgb_only(
            self, target_node_id, target_node_views, current_node_id,
            candidates, forward_action_history=None):
        """Choose a visual return ray without graph geometry or metric state."""
        if len(target_node_views) != 6 or len(candidates) != 6:
            raise ValueError(
                "RGB-only backtracking requires two complete six-view panoramas")
        allowed = [index for index, candidate in enumerate(candidates)
                   if candidate.get("point") is not None]
        if not allowed:
            raise RuntimeError(
                "No segmented RGB ground anchor for node backtracking")
        for candidate in candidates:
            candidate["ground_anchors"] = self._ground_anchors(
                candidate["target_mask"], candidate.get("point"))
        target_sheet = self._contact_sheet([
            self._overlay(rgb, np.zeros(rgb.shape[:2], bool), index, False)
            for index, rgb in enumerate(target_node_views)])
        current_sheet = self._contact_sheet([
            self._overlay(candidate["rgb"], candidate["target_mask"], index,
                          index not in allowed,
                          ground_anchors=candidate["ground_anchors"])
            for index, candidate in enumerate(candidates)])
        two_node_sheet = np.concatenate([target_sheet, current_sheet], axis=0)
        actions = [{
            key: value for key, value in dict(record).items()
            if key in {"step", "action", "commanded_turn_deg",
                       "forward_commanded", "rgb_motion_score"}
        } for record in (forward_action_history or [])[-30:]]
        anchors = {
            str(index): [{"anchor_index": anchor_index,
                          "xy": np.asarray(point).round(1).tolist()}
                         for anchor_index, point in enumerate(
                             candidate["ground_anchors"])]
            for index, candidate in enumerate(candidates)}
        prompt = f"""RGB_ONLY_NODE_BACKTRACK
Return from current graph node {current_node_id} to the visually stored prior
node {target_node_id}. The TOP half is the prior node's six-view panorama; the
BOTTOM half is the current six-view panorama. Green/numbered pixels in the
BOTTOM half are the only selectable floor anchors.

Forward commanded actions, supplied only as causal history:
{json.dumps(actions, ensure_ascii=False)}
Current allowed views: {allowed}
Current floor anchors: {json.dumps(anchors, ensure_ascii=False)}

Infer the reverse route from panorama correspondence and the action order.
Do not infer or request pose, depth, metric distance, collision, navmesh,
shortest path, compass, or world coordinates. Select one allowed current view
and one numbered anchor. Return JSON only."""

        def validate(result):
            view_index = int(result["view_index"])
            if view_index not in allowed:
                raise ValueError(f"view_index must be one of {allowed}")
            options = candidates[view_index]["ground_anchors"]
            anchor_index = int(result["anchor_index"])
            if not 0 <= anchor_index < len(options):
                raise ValueError(
                    f"anchor_index must be in [0,{len(options)-1}]")
            point = np.asarray(options[anchor_index], np.float32)
            return view_index, point, {
                "selector": "rgb_only_two_node_six_view",
                "target_node_id": str(target_node_id),
                "current_node_id": str(current_node_id),
                "anchor_index": anchor_index,
                "requested_xy": point.tolist(),
                "snapped_xy": point.tolist(),
                "reason": str(result.get("reason", "")),
                "allowed_views": allowed,
                "policy_input_contract": "rgb_only_v1",
                "privileged_inputs_used": [],
            }

        schema = json.loads(json.dumps(self.POINT_SCHEMA))
        schema["properties"]["view_index"]["enum"] = allowed
        return self._call(
            "select_backtrack_ground_target_rgb_only", prompt,
            [two_node_sheet], schema, validate)

    def select_turn_landmark_alignment(self, sub_instruction,
                                       current_eight_views,
                                       point_selection_review=None,
                                       alignment_kind="turn"):
        """Select the current-node RGB view that centers a turn landmark.

        Point selection and semantic endpoint orientation are intentionally
        separate.  The first decision chooses an executable route ray; after
        real point arrival this pass chooses only the view that best centers
        the same intended landmark at the new position.  No floor/depth or
        reference-path information is supplied.
        """
        if len(current_eight_views) != 8:
            raise ValueError(
                "turn landmark alignment requires exactly eight RGB views")
        stage = (sub_instruction.to_stage_dict()
                 if hasattr(sub_instruction, "to_stage_dict") else
                 dict(sub_instruction))
        review = point_selection_review or {}
        ambiguity = review.get(
            "ambiguous_landmark_route_adjudication", {}) or {}
        compact_prior = {
            "current_landmark_evidence": ambiguity.get(
                "current_landmark_evidence"),
            "following_route_evidence": ambiguity.get(
                "following_route_evidence"),
            "lexical_ambiguity_diagnostic": ambiguity.get(
                "lexical_ambiguity_diagnostic"),
        }
        schema = {
            "type": "object", "additionalProperties": False,
            "required": ["view_index", "reason", "identity_evidence"],
            "properties": {
                "view_index": {"type": "integer", "enum": list(range(8))},
                "reason": {"type": "string", "maxLength": 360},
                "identity_evidence": {"type": "string", "maxLength": 320},
            },
        }
        terminal_facing = str(alignment_kind) == "terminal_facing"
        prompt = f"""TURN_LANDMARK_CURRENT_NODE_ALIGNMENT
A real point-navigation edge has arrived. Choose the CURRENT-node compass view
that most directly centers the named landmark for this {'explicit final-facing clause' if terminal_facing else 'direction-only stage'}.
This is an orientation-only decision: do not choose the easiest floor, doorway,
or outgoing route, and do not claim instruction completion.
Stage: {stage.get('navigation_instruction', '')}
Named landmark exactly as written: {stage.get('landmark', '')}
Prior clean-RGB identity/route evidence used to select the executed route ray:
{json.dumps(compact_prior, ensure_ascii=False)}

Images 1..8 are clean current-node RGB views labeled by Image 0 as FRONT,
FRONT_LEFT, LEFT, REAR_LEFT, REAR, REAR_RIGHT, RIGHT, FRONT_RIGHT. Match the
same intended landmark using appearance and the prior identity evidence. The
prior following-route evidence is identity context only; it must not determine
the new heading. Select the view in which the landmark's visual center is
closest to that camera's horizontal center. If the instruction token is
misspelled or ambiguous, do not silently reinterpret it as the most familiar
object. Use no depth, floor mask, navmesh, demonstration path, episode identity,
or hidden outcome. Return JSON only."""

        def validate(result):
            view_index = int(result["view_index"])
            if view_index not in range(8):
                raise ValueError("view_index must be in [0,7]")
            evidence = str(result.get("identity_evidence", "")).strip()
            if not evidence:
                raise ValueError("identity_evidence must be non-empty")
            offsets = (0.0, 45.0, 90.0, 135.0,
                       180.0, -135.0, -90.0, -45.0)
            return {
                "view_index": view_index,
                "relative_yaw_deg": offsets[view_index],
                "reason": str(result.get("reason", "")),
                "identity_evidence": evidence,
                "input_policy": "current_node_rgb_only_depth_prohibited",
            }

        sheet = self._completion_contact_sheet(current_eight_views)
        return self._call(
            "select_turn_landmark_current_node_alignment", prompt,
            [sheet, *current_eight_views], schema, validate)

    def classify_node_sub_instruction_sequence(
            self, sub_instructions, expected_sub_instruction_id,
            node_id, six_views, environment_semantics,
            classification_history=None):
        """Visually classify a node against the ordered sub-instruction list."""
        if len(six_views) != 6:
            raise ValueError("node sequence classification requires six views")
        sequence = [
            (item.to_dict() if hasattr(item, "to_dict") else dict(item))
            for item in sub_instructions]
        valid_ids = {int(item["sub_instruction_id"]) for item in sequence}
        expected_id = int(expected_sub_instruction_id)
        if expected_id not in valid_ids:
            raise ValueError(f"expected sub-instruction {expected_id} is absent")
        compact_sequence = [{
            key: item.get(key) for key in (
                "sub_instruction_id", "navigation_instruction", "landmark",
                "form", "semantic_spatial_target", "spatial_relation",
                "visual_arrival_evidence", "forbidden_target")
        } for item in sequence]
        history = list(classification_history or [])[-8:]
        prompt = f"""NODE_SUB_INSTRUCTION_SEQUENCE_CLASSIFICATION
Classify the CURRENT R2R graph node using only its six-view panorama and
observed environment semantics. Determine whether it visually realizes one of
the ordered sub-instructions below. A sub-instruction is realized only when its
semantic spatial target and visual arrival evidence are satisfied, not merely
because the agent intended to go there.

Ordered sub-instructions: {json.dumps(compact_sequence, ensure_ascii=False)}
Expected sub_instruction_id: {expected_id}
Current node_id: {node_id}
Observed DINO+SAM environment semantics:
{json.dumps(environment_semantics, ensure_ascii=False)}
Recent node classifications: {json.dumps(history, ensure_ascii=False)}

The image is a 3-column x 2-row six-view panorama, views 0..5. Do NOT infer the
answer from an intended/departure-purpose field; it is deliberately not
provided. If no completion region is visually supported, return
belongs_to_sequence=false and matched_sub_instruction_id=-1. If a different
sub-instruction is visible, return its ID even when it is not the expected ID.
Sequence correctness is checked outside the VLM. Return JSON only."""

        def validate(result):
            matched = int(result["matched_sub_instruction_id"])
            belongs = bool(result["belongs_to_sequence"])
            if matched != -1 and matched not in valid_ids:
                raise ValueError(
                    f"matched_sub_instruction_id must be -1 or one of {sorted(valid_ids)}")
            if not belongs:
                matched = -1
            confidence = float(result["confidence"])
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be in [0,1]")
            return {
                "node_id": str(node_id),
                "belongs_to_sequence": belongs and matched >= 0,
                "matched_sub_instruction_id": matched,
                "expected_sub_instruction_id": expected_id,
                "expected_sequence_position": bool(
                    belongs and matched == expected_id),
                "confidence": confidence,
                "reason": str(result.get("reason", "")),
                "visual_evidence": str(result.get("visual_evidence", "")),
            }

        return self._call(
            "classify_node_sub_instruction_sequence", prompt,
            [self._contact_sheet(six_views)], self.NODE_SEQUENCE_SCHEMA, validate)

    def adjudicate_pass_region_transition(
            self, sub_instruction, previous_eight_views, edge_keyframes,
            current_eight_views):
        """Focused clean-RGB check for passing out of a named region."""
        if (len(previous_eight_views) != 8 or
                len(current_eight_views) != 8 or not edge_keyframes):
            raise ValueError(
                "PASS region adjudication requires two eight-view nodes and "
                "chronological keyframes")
        item = (sub_instruction.to_dict()
                if hasattr(sub_instruction, "to_dict") else
                dict(sub_instruction))
        schema = {
            "type": "object", "additionalProperties": False,
            "required": [
                "pass_boundary_satisfied",
                "current_distinct_downstream_context",
                "old_region_still_encloses_front",
                "keyframe_transition_support", "confidence", "reason",
                "visual_evidence"],
            "properties": {
                "pass_boundary_satisfied": {"type": "boolean"},
                "current_distinct_downstream_context": {"type": "boolean"},
                "old_region_still_encloses_front": {"type": "boolean"},
                "keyframe_transition_support": {"type": "boolean"},
                "confidence": {"type": "number", "minimum": 0,
                               "maximum": 1},
                "reason": {"type": "string", "maxLength": 400},
                "visual_evidence": {"type": "string", "maxLength": 400},
            },
        }
        prompt = f"""PASS_REGION_RGB_ADJUDICATION
Determine whether the chronological edge passed out of/through the named room
or region into a distinct downstream spatial context.
Instruction: {item.get('navigation_instruction', '')}
Named region: {item.get('landmark', '')}
Required relation: {item.get('spatial_relation', '')}
Completion cue: {item.get('completion_cue', '')}

Image 0 is the PREVIOUS eight-view compass panorama. Image 1 is the ordered
edge-keyframe storyboard. Image 2 is the CURRENT eight-view compass panorama.
Use clean RGB only. Do not use detector labels, depth, navmesh, demonstration
paths, coordinates, or another model's prior verdict.

A room/region pass is topological: it is satisfied when the ordered keyframes
leave the old enclosing interior and CURRENT is a visually distinct downstream
foyer, hall, junction, entrance, or adjacent region. The old room may remain
visible through a rear/side opening. Set old_region_still_encloses_front=true
only if clean CURRENT front RGB is still enclosed by the same named interior;
do not equate a doorway view back into that room with remaining inside it.
Require a real visible transition in the keyframes; mere viewpoint appearance
change is insufficient. Return JSON only."""

        def validate(result):
            confidence = float(result["confidence"])
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be in [0,1]")
            return {
                "pass_boundary_satisfied": bool(
                    result["pass_boundary_satisfied"]),
                "current_distinct_downstream_context": bool(
                    result["current_distinct_downstream_context"]),
                "old_region_still_encloses_front": bool(
                    result["old_region_still_encloses_front"]),
                "keyframe_transition_support": bool(
                    result["keyframe_transition_support"]),
                "confidence": confidence,
                "reason": str(result.get("reason", "")),
                "visual_evidence": str(result.get("visual_evidence", "")),
                "input_policy": "clean_rgb_only_depth_prohibited",
            }

        return self._call(
            "adjudicate_pass_region_transition", prompt,
            [self._completion_contact_sheet(previous_eight_views),
             self._keyframe_storyboard(edge_keyframes),
             self._completion_contact_sheet(current_eight_views)],
            schema, validate)

    @staticmethod
    def _strip_privileged_semantics(value):
        """Remove geometry fields before an RGB-only completion request."""
        forbidden = {
            "median_depth_m", "depth", "depth_m", "position_xyz",
            "pose", "yaw_rad", "absolute_yaw_rad", "geodesic_distance_m",
            "navmesh", "world_xyz",
        }
        if isinstance(value, dict):
            return {
                key: NavigationVLMHarness._strip_privileged_semantics(item)
                for key, item in value.items() if str(key) not in forbidden
            }
        if isinstance(value, list):
            return [NavigationVLMHarness._strip_privileged_semantics(item)
                    for item in value]
        return value

    def judge_edge_instruction_completion_rgb_only(
            self, sub_instruction, following_sub_instruction,
            previous_node_id, current_node_id,
            previous_views, current_views, edge_action_history,
            edge_keyframes, previous_environment_semantics=None,
            current_environment_semantics=None):
        """Binary node/edge judgment with no pose, depth or navmesh input."""
        item = (sub_instruction.to_dict()
                if hasattr(sub_instruction, "to_dict") else
                dict(sub_instruction))
        following = (
            following_sub_instruction.to_dict()
            if hasattr(following_sub_instruction, "to_dict") else
            dict(following_sub_instruction or {}))
        if len(previous_views) not in {6, 8} or len(current_views) != len(
                previous_views):
            raise ValueError(
                "RGB-only completion requires matching 6- or 8-view panoramas")
        if not edge_keyframes:
            raise ValueError(
                "RGB-only completion requires chronological RGB keyframes")
        sanitized_actions = []
        for record in edge_action_history or []:
            sanitized_actions.append({
                key: value for key, value in dict(record).items()
                if key in {
                    "step", "action", "commanded_turn_deg",
                    "forward_commanded", "rgb_motion_score",
                    "orientation_only", "phase",
                }
            })
        action_summary = {
            "control_steps": len(sanitized_actions),
            "forward_command_count": sum(
                str(item.get("action")) == "move_forward"
                for item in sanitized_actions),
            "left_turn_command_count": sum(
                str(item.get("action")) == "turn_left"
                for item in sanitized_actions),
            "right_turn_command_count": sum(
                str(item.get("action")) == "turn_right"
                for item in sanitized_actions),
            "chronological_actions": sanitized_actions,
            "measurement_note": (
                "Commands and RGB-change scores only; no pose, metric "
                "distance, depth, collision or navmesh feedback."),
        }
        previous_semantics = self._strip_privileged_semantics(
            previous_environment_semantics or {})
        current_semantics = self._strip_privileged_semantics(
            current_environment_semantics or {})
        panorama_sheet = (self._completion_contact_sheet
                          if len(previous_views) == 8 else self._contact_sheet)
        previous_sheet = panorama_sheet(previous_views)
        current_sheet = panorama_sheet(current_views)
        keyframe_sheet = self._keyframe_storyboard(edge_keyframes)
        prompt = f"""You are the RGB-only edge completion judge for an R2R robot.

Decide whether the REAL observed transition from the previous node, through
the chronological RGB keyframes, to the current node COMPLETED the active
sub-instruction.  The alternative UNKNOWN includes still on the way, wrong
direction, blocked/stationary, ambiguous identity, or insufficient evidence.

ACTIVE SUB-INSTRUCTION:
{json.dumps(item, ensure_ascii=False, indent=2)}

FOLLOWING SUB-INSTRUCTION (context only; do not complete it early):
{json.dumps(following, ensure_ascii=False, indent=2)}

RGB-only commanded action history:
{json.dumps(action_summary, ensure_ascii=False, indent=2)}

RGB detector evidence at PREVIOUS node:
{json.dumps(previous_semantics, ensure_ascii=False, indent=2)}

RGB detector evidence at CURRENT node:
{json.dumps(current_semantics, ensure_ascii=False, indent=2)}

Image 0 is the previous panorama, image 1 is the chronological edge
storyboard, and image 2 is the current panorama.  Use visible semantic and
temporal change.  Do not infer metric distance, elevation, collision, global
heading, pose or map structure.  Seeing a landmark without reaching the full
semantic spatial relation is UNKNOWN. Return JSON only."""
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["completed", "unknown"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string"},
                "visual_evidence": {"type": "string"},
                "temporal_evidence": {"type": "string"},
            },
            "required": ["status", "confidence", "reason",
                         "visual_evidence", "temporal_evidence"],
        }

        def validate(result):
            status = str(result["status"]).strip().lower()
            if status not in {"completed", "unknown"}:
                raise ValueError("status must be completed or unknown")
            confidence = float(result["confidence"])
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be in [0,1]")
            # An empty action edge cannot establish a new spatial boundary.
            if status == "completed" and not sanitized_actions:
                status = "unknown"
                confidence = min(confidence, 0.49)
            return {
                "status": status,
                "confidence": confidence,
                "reason": str(result["reason"]),
                "visual_evidence": str(result["visual_evidence"]),
                "temporal_evidence": {
                    "rgb_chronology": str(result["temporal_evidence"]),
                },
                "motion_evidence": action_summary,
                "policy_input_contract": "rgb_only_v1",
                "privileged_inputs_used": [],
                "previous_node_id": str(previous_node_id),
                "current_node_id": str(current_node_id),
            }

        return self._call(
            "judge_edge_instruction_completion_rgb_only", prompt,
            [previous_sheet, keyframe_sheet, current_sheet], schema, validate)

    def judge_edge_instruction_completion(
            self, sub_instruction, previous_node_id, current_node_id,
            previous_position_xyz, current_position_xyz,
            previous_six_views, current_six_views,
            previous_environment_semantics, current_environment_semantics,
            edge_action_history, edge_keyframes, *,
            previous_base_yaw_rad=None, current_base_yaw_rad=None,
            previous_visual_embedding=None, current_visual_embedding=None,
            edge_keyframe_records=None, carryover_evidence=None,
            carryover_previous_six_views=None,
            carryover_edge_keyframes=None, following_sub_instruction=None):
        """Judge completion from the complete previous-node -> current-node edge."""
        three_way_progress = bool(
            self.instruction_completion_prompt_version ==
            "v9_three_way_edge_progress")
        bidirectional_progress = bool(
            self.instruction_completion_prompt_version ==
            "v10_bidirectional_three_way")
        strict_binary_completion = bool(
            self.instruction_completion_prompt_version ==
            "v12_binary_completion")
        structured_binary_completion = bool(
            self.instruction_completion_prompt_version in {
                "v13_structured_node_edge_binary",
                "v14_structured_with_unordered_reverse_veto",
                "v15_structured_with_swap_consistent_reverse_veto",
                "v16_temporal_clause_aggregation_swap_veto",
                "v17_primary_consensus_swap_veto",
                "v18_vertical_endpoint_guard",
                "v19_vertical_guard_consensus",
                "v20_stage_endpoint_recovery",
                "v21_stage_endpoint_recovery_structured",
                "v22_stage2_enter_transition",
                "v23_relation_geometry_guard",
                "v24_multireference_threshold_calibration",
            })
        unordered_reverse_veto = bool(
            self.instruction_completion_prompt_version in {
                "v14_structured_with_unordered_reverse_veto",
                "v15_structured_with_swap_consistent_reverse_veto",
                "v16_temporal_clause_aggregation_swap_veto",
                "v17_primary_consensus_swap_veto",
            })
        swap_consistent_reverse_veto = bool(
            self.instruction_completion_prompt_version in {
                "v15_structured_with_swap_consistent_reverse_veto",
                "v16_temporal_clause_aggregation_swap_veto",
                "v17_primary_consensus_swap_veto",
            })
        temporal_clause_aggregation = bool(
            self.instruction_completion_prompt_version ==
            "v16_temporal_clause_aggregation_swap_veto")
        primary_consensus_completion = bool(
            self.instruction_completion_prompt_version in {
                "v17_primary_consensus_swap_veto",
                "v19_vertical_guard_consensus",
                "v20_stage_endpoint_recovery",
                "v21_stage_endpoint_recovery_structured",
                "v22_stage2_enter_transition",
                "v23_relation_geometry_guard",
                "v24_multireference_threshold_calibration",
            })
        vertical_endpoint_guard = bool(
            self.instruction_completion_prompt_version in {
                "v18_vertical_endpoint_guard",
                "v19_vertical_guard_consensus",
                # Endpoint-recovery prompt revisions are cumulative.  They
                # must retain the full-top/full-bottom stair guard instead of
                # trading vertical safety for turn/portal recovery.
                "v20_stage_endpoint_recovery",
                "v21_stage_endpoint_recovery_structured",
                "v22_stage2_enter_transition",
                "v23_relation_geometry_guard",
                "v24_multireference_threshold_calibration",
            })
        completion_model_prompt_version = (
            ("v16_temporal_clause_aggregation_swap_veto"
             if temporal_clause_aggregation else
             "v13_structured_node_edge_binary")
            if unordered_reverse_veto else
            self.instruction_completion_prompt_version)
        binary_completion = bool(
            strict_binary_completion or structured_binary_completion)
        eight_view_completion = bool(
            self.instruction_completion_prompt_version in
            self.EIGHT_VIEW_COMPLETION_PROMPT_VERSIONS)
        required_view_count = 8 if eight_view_completion else 6
        if (len(previous_six_views) != required_view_count or
                len(current_six_views) != required_view_count):
            raise ValueError(
                "instruction completion requires two "
                f"{required_view_count}-view nodes for prompt version "
                f"{self.instruction_completion_prompt_version}")
        if not edge_keyframes:
            raise ValueError("instruction completion requires edge keyframes")
        visual_carryover = bool(
            carryover_previous_six_views is not None or
            carryover_edge_keyframes is not None)
        if visual_carryover:
            if (carryover_previous_six_views is None or
                    carryover_edge_keyframes is None):
                raise ValueError(
                    "visual carryover requires both prior panorama and "
                    "prior edge keyframes")
            if len(carryover_previous_six_views) != required_view_count:
                raise ValueError(
                    "carryover stage-start panorama has wrong view count")
            if not carryover_edge_keyframes:
                raise ValueError("carryover prior edge has no keyframes")
        item = (sub_instruction.to_dict() if hasattr(sub_instruction, "to_dict")
                else dict(sub_instruction))
        actions = list(edge_action_history or [])
        previous_position = np.asarray(previous_position_xyz, np.float32)
        current_position = np.asarray(current_position_xyz, np.float32)
        if previous_position.shape != (3,) or current_position.shape != (3,):
            raise ValueError("instruction completion requires two XYZ node positions")
        vertical_delta_m = float(current_position[1] - previous_position[1])
        action_counts = {}
        for action in actions:
            name = str(action.get("action", "unknown"))
            action_counts[name] = action_counts.get(name, 0) + 1
        action_summary = {
            "control_steps": len(actions),
            "traveled_distance_m": round(sum(
                float(action.get("moved_m", 0.0)) for action in actions), 4),
            "net_turn_deg": round(sum(
                float(action.get("turn_deg", 0.0)) for action in actions), 3),
            "action_counts": action_counts,
            "chronological_actions": actions,
        }
        observed_compact_motion = {
            key: action_summary[key] for key in (
                "control_steps", "traveled_distance_m", "net_turn_deg",
                "action_counts")
        }
        reversed_action_counts = dict(action_counts)
        reversed_action_counts["turn_left"] = action_counts.get("turn_right", 0)
        reversed_action_counts["turn_right"] = action_counts.get("turn_left", 0)
        reversed_compact_motion = {
            "control_steps": action_summary["control_steps"],
            "traveled_distance_m": action_summary["traveled_distance_m"],
            "net_turn_deg": -action_summary["net_turn_deg"],
            "action_counts": reversed_action_counts,
        }
        compact_instruction = {key: item.get(key) for key in (
            "sub_instruction_id", "navigation_instruction", "landmark", "form",
            "semantic_spatial_target", "spatial_relation", "completion_cue",
            "visual_arrival_evidence", "forbidden_target")}
        if following_sub_instruction is not None:
            following_item = (
                following_sub_instruction.to_dict()
                if hasattr(following_sub_instruction, "to_dict") else
                dict(following_sub_instruction))
            compact_instruction["following_sub_instruction_context"] = {
                key: following_item.get(key) for key in (
                    "navigation_instruction", "landmark", "form",
                    "semantic_spatial_target")
            }
        carryover_text = ""
        if isinstance(carryover_evidence, dict):
            carryover_text = f"""
CARRYOVER_TRANSITION_CONTEXT (generic action-history evidence):
The active transition may have begun on the preceding real graph edge.  The
graph already recorded this context online:
{json.dumps(carryover_evidence, ensure_ascii=False)}
Evaluate PRECEDING EDGE -> CURRENT EDGE as one chronological attempt at the
same sub-instruction.  The prior endpoint/temporal fields are observations, not
labels: compare them with CURRENT RGB and the current keyframes.  Add the two
traveled distances and check that their route headings are consistent.  A
prior partial endpoint followed by a current satisfied endpoint is the normal
signature of a long relation crossing.  RGB endpoint and boundary evidence
remain mandatory, and carry-over alone can never establish completion.
When five images are supplied, inspect the complete chronology explicitly:
stage-start panorama -> preceding-edge keyframes -> partial-node panorama ->
current-edge keyframes -> current panorama. Do not treat the partial node as a
new instruction start or reduce either storyboard to its numeric distance.
"""
        active_form = str(item.get("form", "OTHER")).upper()
        vertical_guard_active = bool(
            vertical_endpoint_guard and active_form in {"VERTICAL_UP", "VERTICAL_DOWN"})
        structured_motion = structured_semantics = structured_visual = None
        structured_image_order = (
            "- Image 0: STAGE-START eight-view panorama.\n"
            "- Image 1: preceding real-edge keyframe storyboard.\n"
            "- Image 2: PARTIAL-NODE eight-view panorama.\n"
            "- Image 3: current real-edge keyframe storyboard.\n"
            "- Image 4: CURRENT eight-view panorama."
            if visual_carryover else
            "- Image 0: PREVIOUS node eight-view compass panorama.\n"
            "- Image 1: chronological edge keyframe storyboard, KF0 to KFn.\n"
            "- Image 2: CURRENT node eight-view compass panorama.")
        if structured_binary_completion:
            structured_motion = summarize_motion(
                actions, previous_position, current_position,
                previous_base_yaw_rad, current_base_yaw_rad,
                edge_keyframe_records)
            # Detector matching should be grounded by the actual clause and
            # landmark noun phrase.  Generated completion prose contains
            # generic words such as ``corridor``, ``clear``, and ``reach``;
            # including those words floods the compact evidence with unrelated
            # detections and can evict the named pair (for example bar/chair).
            instruction_grounding_text = " ".join(str(
                compact_instruction.get(key, "")) for key in (
                    "navigation_instruction", "landmark"))
            structured_semantics = summarize_semantic_transition(
                previous_environment_semantics, current_environment_semantics,
                instruction_grounding_text)
            structured_visual = summarize_visual_transition(
                previous_six_views, current_six_views,
                previous_visual_embedding, current_visual_embedding,
                previous_base_yaw_rad, current_base_yaw_rad)
        transition_gate_text = ""
        if structured_binary_completion:
            transition_gate_text = """
Use this evidence-first binary protocol. Work through the returned fields in
their schema order before choosing status.

1. Endpoint test: compare the FULL semantic spatial target at PREVIOUS and
   CURRENT. `current_target_state=satisfied` requires the entire target, not an
   intermediate clause, a plausible room, or merely seeing the landmark.
2. Temporal-order test: use PREVIOUS -> KF0..KFn -> CURRENT. `instructed` means
   the named reference/region relation changes in the requested order;
   `reversed` means the opposite order. Never infer order from distance alone.
3. Boundary test: use `observed` when a threshold/pass/turn/arrival event is
   visible in the keyframes. Use `endpoint_inferred` only when distinct start
   and end contexts plus the chronological keyframes jointly force the crossing
   even though the exact threshold frame is occluded. This is stronger than
   merely ending somewhere plausible.
4. Motion-history test: the structured trajectory is supporting evidence for
   direction, turn sign/order, vertical change, stationary motion, and whether
   keyframes correspond to movement phases. It cannot name a room or landmark.
5. Detector and embedding summaries are fallible supporting evidence. Verify
   them against RGB. A noisy label alone never completes an instruction.
6. Return COMPLETED only when the previous target was not already satisfied,
   the current full target and completion cue are satisfied, semantic order is
   instructed, a boundary is observed or strongly endpoint-inferred, and no
   reverse/motion contradiction exists. Otherwise return UNKNOWN. UNKNOWN also
   includes all partial/on-route states; do not separately classify them.
"""
            if temporal_clause_aggregation:
                transition_gate_text += """
7. Compound-clause localization: require every named transition in its proper
   chronological place across PREVIOUS -> keyframes -> CURRENT. Intermediate
   portals/objects need not remain visible at CURRENT after being passed; they
   must be supported in the ordered keyframes or behind-side context. The final
   destination/spatial relation, however, must still hold at CURRENT. Do not
   collapse a missing intermediate object in the final panorama into partial.
8. Route-heading setup: an initial in-place rotation before the first forward
   phase may align the camera/agent to the route and is not itself a branch
   deviation. Evaluate turn consistency after translation begins, together
   with the scene transition. A later large turn can still contradict a
   straight/through instruction.
9. Endpoint wording is semantic rather than a literal view-index template.
   For reach/toward/near targets, a landmark may be side or side-rear after the
   final approach when the current node is at its transition and farther route
   context has opened. This is not enough by itself: scale, surrounding RGB,
   ordered keyframes, and the full current relation must agree. A landmark
   merely glimpsed behind after overshoot remains UNKNOWN.
10. Clean RGB is the identity authority. Missing or repeated detector labels
    do not force ambiguity when appearance plus local spatial context uniquely
    establish the same reference; conversely, detector text alone never
    establishes identity. Apply this identically to all forms and scenes.
"""
            if vertical_guard_active:
                transition_gate_text += """
11. Full vertical endpoint guard (generic, form-driven): for VERTICAL_UP or
    VERTICAL_DOWN whose target says top, bottom, landing, or full ascent/descent,
    a positive/negative elevation change and stair keyframes are not sufficient.
    `current_level_landing` is true only when CURRENT shows a level walkable
    landing/end floor in the arrival direction; set
    `remaining_stair_flight_visible=true` only when clean RGB shows repeated
    treads that continue farther UP/DOWN away from the camera along the active
    arrival route. A stairwell opening, railing, or the already traversed flight
    descending below/beside/behind the agent is positive landing context, not an
    unfinished flight. Do not copy noisy detector sectors into this flag:
    inspect tread direction in RGB. A genuinely continuing flight ahead means
    the target is partial and the final status must be UNKNOWN.
    `endpoint_evidence_stable` requires the level endpoint to be
    supported by CURRENT plus the last chronological keyframe, not a single
    ambiguous crop. For non-vertical forms set all three flags false. Never
    use metric height, action count, or a single stair detector label as a
    substitute for the landing evidence.
"""
            if self.instruction_completion_prompt_version in {
                    "v20_stage_endpoint_recovery",
                    "v21_stage_endpoint_recovery_structured",
                    "v23_relation_geometry_guard",
                    "v24_multireference_threshold_calibration"}:
                transition_gate_text += """
12. GENERIC ENDPOINT-INFERENCE RECOVERY (stage-1 research version): a
    conservative partial/not-observed response may be upgraded only when the
    ordered edge, current endpoint semantics and motion jointly force the full
    spatial relation. For PASS/CIRCUMNAVIGATE, a matched reference in a rear
    sector after instructed motion is the pass boundary. For ENTER/EXIT/TRAVERSE,
    a matched destination/portal transition with ordered keyframes and
    supported motion is sufficient. For pure turns, a stored node-heading change
    plus a new corridor is sufficient, except when the instruction explicitly
    continues up/down stairs and unfinished treads remain visible. Never infer
    completion from distance or executor arrival alone; retain UNKNOWN for
    identity ambiguity, reverse/blocked motion, or a target still clearly ahead.
"""
                if self.instruction_completion_prompt_version in {
                        "v21_stage_endpoint_recovery_structured",
                        "v22_stage2_enter_transition",
                        "v23_relation_geometry_guard",
                        "v24_multireference_threshold_calibration"}:
                    transition_gate_text += """
13. Structured relation refinements: for BETWEEN_OBJECTS whose cue is to reach
the gap, the endpoint may be inside the gap, so the two named objects can
occupy opposite side/rear sectors; do not require both objects to remain in
FRONT. For a generic TRAVERSE/ENTER doorway, one crossed doorway can be
identified by a strong rear doorway after ordered motion even when another
opening is visible in FRONT; use this only when the instruction names no
ordinal or unique competing portal. For a turn clause that also says up/down
stairs or landing, an unfinished stair flight visible in FRONT, FRONT_LEFT, or
FRONT_RIGHT vetoes completion even if the heading changed.
For a CIRCUMNAVIGATE/around/backside endpoint, the eight views overlap: a
landmark in FRONT_LEFT or FRONT_RIGHT is lateral, not necessarily still ahead.
If the named landmark is supported in rear/rear-side sectors, exact FRONT no
longer contains it, a new open route is ahead, and the chronological real edge
shows substantial non-reversed translation, treat the behind/around relation
as satisfied. Do not demand that the landmark vanish from every side view.
14. Compound terminal-extent clauses remain atomic. If a clause says to enter
a region and then walk ALONG/FOLLOW a named landmark to its END/FAR END/entire
length, entering the region is only partial. The landmark continuing strongly
in the exact FRONT view means the terminal extent remains ahead, so return
UNKNOWN. A genuine endpoint may retain the landmark at the side or rear.
"""
            if (self.instruction_completion_prompt_version in {
                    "v23_relation_geometry_guard",
                    "v24_multireference_threshold_calibration"}):
                transition_gate_text += """
15. Detector-backed terminal relation guard: APPROACH_LANDMARK and STOP_WAIT
require an RGB-aligned current-node detection of the named landmark. A textual
claim without that observed identity remains UNKNOWN. For `under/beneath X`,
the overhead qualifier must be supported in at least two panorama sectors;
one isolated open-vocabulary hit is insufficient. These detector checks are
vetoes only and cannot by themselves produce COMPLETED.
16. BETWEEN/FOLLOW corridor boundary: distinguish the route surface from the
side objects that define its finite passage. A carpet/path may continue into
the next room and can remain visible in FRONT after the instructed rope/chair/
barrier passage has ended. If the side boundaries move from front/side to
rear or disappear after ordered keyframes and the following instructed
landmark/region becomes visible ahead, that is positive completion evidence.
Do not reject solely because the carpet/path texture continues. Conversely,
surface texture alone cannot prove completion while the defining side objects
still extend ahead. `following_sub_instruction_context` is instruction-only
context for recognizing that boundary; it is never permission to execute or
complete the following clause early.
"""
            if (self.instruction_completion_prompt_version ==
                    "v24_multireference_threshold_calibration"):
                transition_gate_text += """
17. Coordinated PASS landmarks must be checked one noun phrase at a time. For
"pass the stairs and the bathroom", do not pretend the two nouns are one
trackable instance. A reference visible at the source and absent from clean RGB
after continuous forward motion may already be behind; another reference may
remain genuinely visible only in rear/rear-side views. In this coordinated
case set same_reference_instance=not_applicable after independently checking
all named references. Near-full-frame detector proposals repeated across
nonadjacent sectors are image-level query matches, not localized proof that a
landmark occupies every bearing; cross-check them against RGB and do not copy
their sectors when the object/region is not actually visible there.
18. A compound turn ending "at/to the doorway/opening" may finish with the
same door frame straddling FRONT_SIDE and REAR_SIDE because adjacent panorama
cameras overlap at the threshold. That side-straddling relation, ordered
keyframes, the named room context moving behind, and real motion can establish
the doorway endpoint; exact FRONT is not mandatory after the final heading
correction. By contrast, a turn followed by "walk towards X" whose completion
cue is explicitly "facing/moving towards X" is a directional commitment, not
an instruction to reach X. It completes after the requested turn plus sustained
motion on that visible route; do not invent a reach-the-far-side boundary.
19. PASS of a room/region is a topological context transition, not a demand
that a broad room-category query disappear from all eight overlapping views.
Compare clean RGB at PREVIOUS, ordered keyframes, and CURRENT. If continuous
instructed motion leaves the named room's enclosing interior and CURRENT is a
distinct downstream foyer, hall, junction, entrance, or adjacent region, the
pass boundary is satisfied even when the old room remains visible through a
rear/side opening. Conversely, if CURRENT is still enclosed by the same room
with no distinct downstream spatial context, return UNKNOWN. Near-full-frame
room detections duplicated in many sectors are not bearing evidence: localize
`reference_current_sectors` from clean RGB, and do not copy those detector
sectors into the endpoint record.
"""
        elif strict_binary_completion:
            transition_gate_text = """
Apply this strict binary completion protocol:
1. COMPLETED requires a new, fully supported completion event on this edge.
   The previous panorama, chronological keyframes, and current panorama must
   establish the requested semantic relation or boundary in temporal order.
2. UNKNOWN includes every non-completed case: partial/on-route progress, a
   plausible direction without the final boundary, reversed or wrong motion,
   stationary motion, insufficient evidence, and ambiguity.
3. Do not infer completion from path length, action count, displacement, point
   executor arrival, or merely seeing the named landmark/room. These can only
   support a visible semantic completion event.
4. If the completion target already holds at the previous node and no new
   instructed completion event occurs, return UNKNOWN.
5. Detect reversal from semantic order: destination -> source, near-target ->
   farther-away, passed-landmark -> landmark-ahead, or an opposite turn cannot
   be COMPLETED even if low-level translations are called `forward`.
6. The completion cue is semantic, not an exact camera-heading constraint. If
   the destination surrounds the current node or a passed landmark is clearly
   behind/side-behind, do not reject completion only because it is not centered
   in VIEW 0.

Set completion_boundary_observed=true only for the full instruction target,
not an intermediate clause. temporal_relation_change_observed must reflect a
supported Image 0 -> keyframes -> Image 2 change. contradiction_observed is
true for a supported reverse/wrong relation. The harness derives the final
binary status from these fields.
"""
        elif three_way_progress or bidirectional_progress:
            transition_gate_text = """
Apply this mutually exclusive three-way decision protocol:
1. ARRIVED requires a new completion event on this edge. The chronological
   images must show the requested semantic boundary/relation being satisfied,
   and the CURRENT panorama must support the instruction's completion cue.
2. ON_ROUTE requires clear positive but incomplete progress: the referenced
   landmark, portal, region, path boundary, turn, or destination must change in
   the instructed direction across Image 0 -> keyframes -> Image 2. Merely
   moving, turning, seeing a plausible object, or ending in a plausible room is
   insufficient.
3. UNKNOWN is mandatory when the edge is reversed, moves away from the target,
   takes the wrong turn/portal/region, lacks a supported temporal relation
   change, remains effectively stationary, or is visually ambiguous. UNKNOWN
   does not assert a supervised off-route class.
4. If the completion cue was already true at the previous node and no new
   instructed completion event occurs, do not return ARRIVED. If the edge then
   moves consistently toward a later unsatisfied part of the same instruction,
   ON_ROUTE is allowed; otherwise return UNKNOWN.
5. A long trajectory, net displacement, action count, or point-executor arrival
   signal can only support visual evidence. It can never independently produce
   ARRIVED or ON_ROUTE.
6. Detect direction reversal from semantic order, not just action names. For
   example destination -> source, near-target -> farther-away, passed-landmark
   -> landmark-ahead, or right-turn goal -> reverse leftward relation is
   contradiction evidence even if every low-level translation action is named
   `forward`.
"""
            if three_way_progress:
                transition_gate_text += """

Evidence fields are observations, not independent votes. Set
completion_boundary_observed=true only for full completion;
instruction_consistent_progress=true for both ARRIVED and ON_ROUTE;
contradiction_observed=true for a supported reverse/wrong semantic transition.
The harness derives the final status from these three fields so they must match
the visual-temporal evidence rather than the proposed status.
"""
            else:
                transition_gate_text += """

Compare both temporal orders without assuming the observed order is correct.
The observed order is PREVIOUS -> chronological keyframes -> CURRENT. The
reversed candidate is CURRENT -> reverse-chronological keyframes -> PREVIOUS.
Choose direction_fit=observed only when the observed order has clearly stronger
instruction-consistent semantic progress. Choose reversed when the reversed
order is clearly stronger. Choose neither for ties, symmetric/ambiguous scenes,
or when neither order proves instruction-consistent progress.
"""
        elif self.instruction_completion_prompt_version in {
                "v2_transition_gates", "v3_temporal_completion_event",
                "v4_partial_extent_and_turns",
                "v5_source_destination_dominance",
                "v6_operational_partial_extent",
                "v7_structured_partial_extent",
                "v8_eight_view_spatial_relations"}:
            transition_gate_text = """
Apply these transition gates before returning completed:
1. ENTER/EXIT/TRAVERSE_PORTAL requires a before-to-after region or door-frame
   crossing visible across the keyframes; merely seeing a doorway is unknown.
2. PASS/CIRCUMNAVIGATE requires the landmark relation to change across time;
   merely seeing the landmark in the current panorama is unknown.
3. VERTICAL_UP/DOWN requires keyframe evidence of stair traversal and a current
   landing/end state; merely seeing stairs is unknown.
4. APPROACH/BETWEEN/STOP requires the current spatial relation itself, not just
   motion in a plausible direction.
5. Turns, distance traveled, and the executor arrival signal are supporting
   evidence only and can never establish semantic completion by themselves.
If the evidence supports progress but not the completion boundary, return
unknown. If evidence suggests a wrong move, also return unknown.
"""
        if self.instruction_completion_prompt_version in {
                "v3_temporal_completion_event",
                "v4_partial_extent_and_turns",
                "v5_source_destination_dominance",
                "v6_operational_partial_extent",
                "v7_structured_partial_extent",
                "v8_eight_view_spatial_relations"}:
            transition_gate_text += """
6. This is an EDGE completion judgment, not a current-node scene classifier.
   First decide whether the completion condition was already true at the
   previous node. If it was already true and the keyframes do not show a new
   instructed completion event, return unknown: this edge did not complete it.
7. Compare KF 0 against the later keyframes explicitly. A final room or
   landmark alone is insufficient when the same room or landmark was already
   present at KF 0. Require a visible relation change or boundary-crossing event
   during this edge.
"""
        if self.instruction_completion_prompt_version in {
                "v4_partial_extent_and_turns",
                "v5_source_destination_dominance",
                "v6_operational_partial_extent",
                "v7_structured_partial_extent",
                "v8_eight_view_spatial_relations"}:
            transition_gate_text += """
8. Interpret the requested extent literally. For a partial vertical instruction
   such as "halfway up the stairs", do NOT require a landing: completed may be
   supported when KF 0 is at the stair base, keyframes show continuous ascent,
   and the current panorama clearly shows substantial stairs both below and
   still above. For "top", "bottom", or "landing", keep requiring that boundary.
9. For a pure TURN_LEFT/TURN_RIGHT/TURN_AROUND instruction, completion is an
   orientation-change event rather than a destination region. Require the
   action turns and panorama correspondence to support the requested rotation;
   forward translation without that rotation remains unknown.
"""
        if self.instruction_completion_prompt_version in {
                "v5_source_destination_dominance",
                "v6_operational_partial_extent",
                "v7_structured_partial_extent",
                "v8_eight_view_spatial_relations"}:
            transition_gate_text += """
10. For ENTER/EXIT/TRAVERSE_PORTAL, establish which side contains the camera at
    the previous node. Seeing the source room through one reverse doorway does
    not mean the camera is still inside it. If the destination region already
    surrounds/dominates the previous six-view panorama and only the source is
    glimpsed through a doorway, the crossing happened before this edge: return
    unknown unless this edge crosses a different requested boundary.
11. Conversely, a valid portal completion should show the source region
    surrounding most previous views or KF 0 being on its side, the frame being
    traversed in chronological keyframes, and the destination surrounding the
    current node. State this source-side -> frame -> destination-side evidence
   explicitly before returning completed.
"""
        if self.instruction_completion_prompt_version in {
                "v6_operational_partial_extent",
                "v7_structured_partial_extent",
                "v8_eight_view_spatial_relations"}:
            transition_gate_text += """
12. RGB evidence cannot prove an exact metric fraction such as 50%. For an
    explicitly approximate partial extent ("about halfway", "part way"), use
    the sub-instruction's visual_arrival_evidence as the operational completion
    definition. When the edge starts at the base, continuously ascends, and the
    current panorama visibly contains several steps below AND several steps
    still above, return completed. Do not reject that evidence merely because
    exact mathematical halfway is unavailable. This exception does not apply
    to a top, bottom, or landing instruction.
"""
        instruction_text = " ".join(str(item.get(key, "")) for key in (
            "navigation_instruction", "semantic_spatial_target",
            "completion_cue", "visual_arrival_evidence")).lower()
        structured_partial_extent = bool(
            self.instruction_completion_prompt_version in {
                "v7_structured_partial_extent",
                "v8_eight_view_spatial_relations"} and
            str(item.get("form", "")).upper() in {"VERTICAL_UP", "VERTICAL_DOWN"} and
            any(token in instruction_text for token in (
                "halfway", "half way", "partway", "part way")))
        if structured_partial_extent:
            transition_gate_text += """
13. For this approximate partial-stair instruction, do not estimate a numeric
    percentage and do not let status substitute for evidence extraction. Fill
    partial_extent_evidence from the images: starts_at_stair_base from Image 0/
    KF 0, continuous_ascent from the chronological strip, and current_steps_
    below/current_steps_above from Image 2. The harness will apply the stated
                operational definition to those four observations.
"""
        directional_prompt = ""
        if structured_binary_completion:
            form = str(item.get("form", "OTHER")).upper()
            form_rule = self.EIGHT_VIEW_FORM_RULES.get(
                form, self.EIGHT_VIEW_FORM_RULES["OTHER"])
            directional_prompt = f"""
EIGHT-VIEW STRUCTURED ENDPOINT ANALYSIS (v13):
Each endpoint panorama is a 4x2 sheet: FRONT, FRONT_LEFT, LEFT, REAR_LEFT,
REAR, REAR_RIGHT, RIGHT, FRONT_RIGHT. Directions are relative to that node's
stored heading, so use the online heading delta and absolute-direction visual
alignment when comparing the two nodes.

Active-form completion rule for {form}:
{form_rule}

STOP-WAIT NEAR-RELATION CLARIFICATION:
When the active form is STOP_WAIT with a "near", "beside", or "stop near"
relation, do not reject completion solely because the landmark is in REAR or a
rear-side sector. If it is the same visible instance, remains close/stable at
the current node, and the edge motion ends adjacent to it without a clear
overshoot, the near relation is satisfied even if the robot's final heading is
not pointed at the landmark. Likewise, "in front of X" normally describes the
robot's object-relative stopping position, not its camera orientation: do not
reject a close, temporally approached near-side endpoint merely because X is
in a side sector after the final control turn. Require X in the current FRONT
only when the instruction explicitly says to face/look toward X or keep X
ahead. A landmark that moves past the robot into rear sectors during continued
translation is still an overshoot and remains UNKNOWN.

For a discrete reference, list only sectors where the same reference instance
is actually supported; otherwise use empty lists and set identity appropriately.
For region-level targets, judge which environment surrounds the camera rather
than demanding that a room label be centered in FRONT. A reference remaining
visible behind the agent can support PASS/EXIT/ENTER completion; it does not by
itself imply failure. {"For compound instructions, every clause named in the semantic target/completion cue must be satisfied over the ordered edge; the final destination/relation must hold at CURRENT, while already-passed intermediate clauses may be supported by keyframes and behind-side context." if temporal_clause_aggregation else "For compound instructions, every clause named in the semantic target/completion cue must be satisfied at CURRENT."}
"""
        elif strict_binary_completion:
            form = str(item.get("form", "OTHER")).upper()
            form_rule = self.EIGHT_VIEW_FORM_RULES.get(
                form, self.EIGHT_VIEW_FORM_RULES["OTHER"])
            directional_prompt = f"""
EIGHT-VIEW BINARY COMPLETION ANALYSIS (v12):
Each panorama is ordered as VIEW 0 FRONT, VIEW 1 FRONT_LEFT, VIEW 2 LEFT,
VIEW 3 REAR_LEFT, VIEW 4 REAR, VIEW 5 REAR_RIGHT, VIEW 6 RIGHT, and
VIEW 7 FRONT_RIGHT, relative to that node's heading.

Active-form spatial rule for {form}:
{form_rule}

Apply the rule to the actual temporal order. Any state described as partial,
approaching, beside, looking into, moving inside, or otherwise before the full
completion cue must be UNKNOWN. Do not classify whether that partial state is
on the correct route.
"""
        elif bidirectional_progress:
            form = str(item.get("form", "OTHER")).upper()
            form_rule = self.EIGHT_VIEW_FORM_RULES.get(
                form, self.EIGHT_VIEW_FORM_RULES["OTHER"])
            directional_prompt = f"""
EIGHT-VIEW BIDIRECTIONAL ANALYSIS (v10):
Each panorama is ordered as VIEW 0 FRONT, VIEW 1 FRONT_LEFT, VIEW 2 LEFT,
VIEW 3 REAR_LEFT, VIEW 4 REAR, VIEW 5 REAR_RIGHT, VIEW 6 RIGHT, and
VIEW 7 FRONT_RIGHT, relative to that node's agent heading.

Active-form spatial rule for {form}:
{form_rule}

Evaluate Candidate A and Candidate B independently using the same rule. For
each candidate, distinguish full completion from clear partial progress. A
partial-looking endpoint is on_route only when the candidate's temporal order
shows the requested relation changing toward it. Do not prefer Candidate A just
because it is called observed, and do not use action magnitude as a tiebreaker.
"""
        elif three_way_progress:
            form = str(item.get("form", "OTHER")).upper()
            form_rule = self.EIGHT_VIEW_FORM_RULES.get(
                form, self.EIGHT_VIEW_FORM_RULES["OTHER"])
            directional_prompt = f"""
EIGHT-VIEW TEMPORAL PROGRESS ANALYSIS (v9):
Each panorama is a 4-column x 2-row sheet in this exact order:
top row: VIEW 0 FRONT, VIEW 1 FRONT_LEFT, VIEW 2 LEFT, VIEW 3 REAR_LEFT;
bottom row: VIEW 4 REAR, VIEW 5 REAR_RIGHT, VIEW 6 RIGHT,
VIEW 7 FRONT_RIGHT. Directions are relative to that node's agent heading.

Active-form spatial rule for {form}:
{form_rule}

Apply the rule in temporal order. First state the reference relation at the
previous node, then the relation changes visible in the chronological strip,
then the current relation. A clear partial state described by the rule maps to
ON_ROUTE only when the transition into that state follows the requested order.
The same partial-looking state reached in reverse or without temporal support is
UNKNOWN. When the full semantic target and completion cue are established, use
ARRIVED even if the named landmark is in a side rather than the exact FRONT view;
the instruction's semantic relation has priority over a narrow heading literal.
"""
        elif eight_view_completion:
            form = str(item.get("form", "OTHER")).upper()
            form_rule = self.EIGHT_VIEW_FORM_RULES.get(
                form, self.EIGHT_VIEW_FORM_RULES["OTHER"])
            directional_prompt = f"""
EIGHT-VIEW DIRECTIONAL ANALYSIS (v8):
Each panorama is a 4-column x 2-row sheet in this exact order:
top row: VIEW 0 FRONT, VIEW 1 FRONT_LEFT, VIEW 2 LEFT, VIEW 3 REAR_LEFT;
bottom row: VIEW 4 REAR, VIEW 5 REAR_RIGHT, VIEW 6 RIGHT,
VIEW 7 FRONT_RIGHT. Directions are relative to that node's agent heading.

Active-form rule for {form}:
{form_rule}

Before choosing status, fill directional_evidence. List only sectors where the
instruction's same reference landmark/portal/region boundary is actually
visible, and give the corresponding exact view indices; use empty lists when
no discrete reference exists or it is not visible. Because adjacent 90-degree
cameras overlap, count a sector only when the reference center lies inside that
view, not when a few edge pixels leak into it. For repeated object classes,
compare appearance and surrounding context across the previous panorama,
keyframes, and current panorama. Set multiple_similar_instances=true whenever
two plausible instances make the identity ambiguous; such ambiguity cannot
complete PASS/CIRCUMNAVIGATE/APPROACH/STOP.
rear_sector_evidence means the relevant reference is visible in VIEW 3, 4, or
5 at the CURRENT node and same_instance_confident=true. front_sector_
contradiction means current front sectors show a relation that specifically
conflicts with completion; it is not simply edge overlap. temporal_relation_
change requires a supported relation change from previous panorama through
keyframes to current.
"""
        if structured_binary_completion:
            prompt = f"""EDGE_INSTRUCTION_COMPLETION_JUDGMENT
Decide whether this one real graph edge completed the active R2R
sub-instruction. The public result is binary: `completed` or `unknown`.
Prompt version: {completion_model_prompt_version}
{transition_gate_text}
{directional_prompt}

ACTIVE_SUB_INSTRUCTION:
{json.dumps(compact_instruction, ensure_ascii=False)}
{carryover_text}

GRAPH_ENDPOINTS:
{json.dumps({
    "previous_node_id": previous_node_id,
    "current_node_id": current_node_id,
    "previous_position_xyz": previous_position.tolist(),
    "current_position_xyz": current_position.tolist(),
}, ensure_ascii=False)}

STRUCTURED_MOVEMENT_HISTORY:
{json.dumps(structured_motion, ensure_ascii=False)}

STRUCTURED_NODE_SEMANTIC_TRANSITION:
{json.dumps(structured_semantics, ensure_ascii=False)}

STRUCTURED_NODE_VISUAL_TRANSITION:
{json.dumps(structured_visual, ensure_ascii=False)}

IMAGE ORDER:
{structured_image_order}

Cross-check all structured evidence against these RGB images. The structured
data are observations, not labels. No case category, expected result,
demonstration waypoint/path, future state, or earlier model result is provided.
Return JSON only."""
        elif strict_binary_completion:
            prompt = f"""EDGE_INSTRUCTION_COMPLETION_JUDGMENT
Decide only whether the real PREVIOUS -> CURRENT edge completed the active R2R
sub-instruction. Return exactly `completed` or `unknown`; partial/on-route
progress is always `unknown` and is not otherwise classified.
Prompt version: {self.instruction_completion_prompt_version}
{transition_gate_text}
{directional_prompt}

Active sub-instruction:
{json.dumps(compact_instruction, ensure_ascii=False)}
{carryover_text}
Previous node: {previous_node_id}
Current node: {current_node_id}
Online node geometry (not demonstration truth):
{json.dumps({
    "previous_position_xyz": previous_position.tolist(),
    "current_position_xyz": current_position.tolist(),
    "vertical_delta_m": round(vertical_delta_m, 4),
}, ensure_ascii=False)}
Actual edge action history:
{json.dumps(action_summary, ensure_ascii=False)}
Previous observed semantics:
{json.dumps(previous_environment_semantics, ensure_ascii=False)}
Current observed semantics:
{json.dumps(current_environment_semantics, ensure_ascii=False)}

Image 0 is the previous eight-view panorama. Image 1 is the chronological edge
keyframe strip. Image 2 is the current eight-view panorama.
No case category, expected result, demonstration waypoint, future path, path
index, or earlier model result is provided. Return JSON only."""
        elif bidirectional_progress:
            prompt = f"""EDGE_INSTRUCTION_COMPLETION_JUDGMENT
Compare both temporal orientations of one observed indoor edge against the
active R2R sub-instruction. This is a symmetric direction test: Candidate A is
not presumed correct.

For each candidate return arrived, on_route, or unknown using these meanings:
- arrived: this candidate completes the semantic spatial target;
- on_route: this candidate makes clear instruction-consistent progress but
  remains before the completion boundary;
- unknown: progress is reversed/wrong, unsupported, stationary, or ambiguous.
Then choose direction_fit=observed only if Candidate A is clearly more
instruction-consistent, reversed only if Candidate B is clearly more
instruction-consistent, otherwise neither.
Prompt version: {self.instruction_completion_prompt_version}
{transition_gate_text}
{directional_prompt}

Active sub-instruction:
{json.dumps(compact_instruction, ensure_ascii=False)}

Candidate A (observed):
- temporal order: Image 0 -> Image 1 -> Image 2
- geometry: {json.dumps({
    "start_xyz": previous_position.tolist(),
    "end_xyz": current_position.tolist(),
    "vertical_delta_m": round(vertical_delta_m, 4),
}, ensure_ascii=False)}
- compact motion: {json.dumps(observed_compact_motion, ensure_ascii=False)}
- start semantics: {json.dumps(previous_environment_semantics, ensure_ascii=False)}
- end semantics: {json.dumps(current_environment_semantics, ensure_ascii=False)}

Candidate B (reversed):
- temporal order: Image 2 -> Image 3 -> Image 0
- geometry: {json.dumps({
    "start_xyz": current_position.tolist(),
    "end_xyz": previous_position.tolist(),
    "vertical_delta_m": round(-vertical_delta_m, 4),
}, ensure_ascii=False)}
- compact inverse motion: {json.dumps(reversed_compact_motion, ensure_ascii=False)}
- start semantics: {json.dumps(current_environment_semantics, ensure_ascii=False)}
- end semantics: {json.dumps(previous_environment_semantics, ensure_ascii=False)}

Image 0 is the previous eight-view panorama. Image 1 is the observed
chronological keyframe strip. Image 2 is the current eight-view panorama.
Image 3 is the same real keyframes in reverse chronological order.
No demonstration waypoint, future path, path index, case category, expected
label, or earlier model result is provided. Return JSON only."""
        elif three_way_progress:
            prompt = f"""EDGE_INSTRUCTION_COMPLETION_JUDGMENT
Classify only the real transition from the PREVIOUS graph node to the CURRENT
graph node for the active R2R sub-instruction.

Return exactly one status:
- arrived: this edge completed the semantic spatial target;
- on_route: this edge made clear instruction-consistent progress but did not
  complete the target;
- unknown: neither claim is reliably established, including reverse/wrong or
  ambiguous motion.
Prompt version: {self.instruction_completion_prompt_version}
{transition_gate_text}
{directional_prompt}

Active sub-instruction:
{json.dumps(compact_instruction, ensure_ascii=False)}
Previous node: {previous_node_id}
Current node: {current_node_id}
Node geometry available at runtime (not demonstration truth):
{json.dumps({
    "previous_position_xyz": previous_position.tolist(),
    "current_position_xyz": current_position.tolist(),
    "vertical_delta_m": round(vertical_delta_m, 4),
}, ensure_ascii=False)}
Edge action history:
{json.dumps(action_summary, ensure_ascii=False)}
Previous observed semantics:
{json.dumps(previous_environment_semantics, ensure_ascii=False)}
Current observed semantics:
{json.dumps(current_environment_semantics, ensure_ascii=False)}

Image 0 is the previous {required_view_count}-view panorama. Image 1 is the
chronological edge keyframe strip from start to stop. Image 2 is the current
{required_view_count}-view panorama.
No demonstration waypoint, future path, path index, case category, or expected
label is provided. Never infer status from the point executor's arrival signal.
Return JSON only."""
        else:
            prompt = f"""EDGE_INSTRUCTION_COMPLETION_JUDGMENT
Decide only whether the transition from the PREVIOUS graph node to the CURRENT
graph node COMPLETED the active R2R sub-instruction.

Return status=completed only when the before/after panoramas, chronological edge
keyframes, and actual action history jointly support the semantic spatial target
and completion evidence. Return status=unknown whenever completion is not
established. Unknown includes partial progress, an ambiguous transition,
insufficient visual evidence, and a possibly wrong move. Do not decide whether
an unknown node is on-route or off-route; that belongs to an outer exploration
policy.
Prompt version: {self.instruction_completion_prompt_version}
{transition_gate_text}
{directional_prompt}

Active sub-instruction:
{json.dumps(compact_instruction, ensure_ascii=False)}
Previous node: {previous_node_id}
Current node: {current_node_id}
Node geometry available at runtime (not demonstration truth):
{json.dumps({
    "previous_position_xyz": previous_position.tolist(),
    "current_position_xyz": current_position.tolist(),
    "vertical_delta_m": round(vertical_delta_m, 4),
}, ensure_ascii=False)}
Edge action history:
{json.dumps(action_summary, ensure_ascii=False)}
Previous observed semantics:
{json.dumps(previous_environment_semantics, ensure_ascii=False)}
Current observed semantics:
{json.dumps(current_environment_semantics, ensure_ascii=False)}

Image 0 is the previous {required_view_count}-view panorama. Image 1 is the
chronological edge keyframe strip from start to stop. Image 2 is the current
{required_view_count}-view panorama.
Never infer completion merely from the intended destination or from the point
executor's arrival signal. Return JSON only."""

        def validate(result):
            if structured_binary_completion:
                model_status = str(result["status"]).strip().lower()
                if model_status not in {"completed", "unknown"}:
                    raise ValueError("status must be completed or unknown")
                confidence = float(result["confidence"])
                if not 0.0 <= confidence <= 1.0:
                    raise ValueError("confidence must be in [0,1]")
                endpoint = result["endpoint_evidence"]
                temporal = result["temporal_evidence"]
                motion = result["motion_evidence"]
                known_sectors = {
                    "front", "front_left", "left", "rear_left", "rear",
                    "rear_right", "right", "front_right"}
                previous_sectors = [str(value).strip().lower() for value in
                                    endpoint["reference_previous_sectors"]]
                current_sectors = [str(value).strip().lower() for value in
                                   endpoint["reference_current_sectors"]]
                previous_sector_set = set(previous_sectors)
                current_sector_set = set(current_sectors)
                invalid_sectors = (
                    set(previous_sectors) | set(current_sectors)) - known_sectors
                if invalid_sectors:
                    raise ValueError(
                        f"unknown structured endpoint sectors: {sorted(invalid_sectors)}")
                previous_state = str(endpoint["previous_target_state"])
                current_state = str(endpoint["current_target_state"])
                completion_cue = str(endpoint["current_completion_cue"])
                semantic_order = str(temporal["semantic_order"])
                boundary_event = str(temporal["boundary_event"])
                same_instance = str(temporal["same_reference_instance"])
                keyframe_support = bool(temporal["keyframe_support"])
                reverse_observed = bool(temporal["reverse_transition_observed"])
                motion_fit = str(motion["instruction_motion_fit"])
                stationary = bool(motion["stationary_or_blocked"])
                form = str(item.get("form", "OTHER")).upper()
                current_node_relevant_for_gate = list(
                    ((structured_semantics or {}).get("current_node", {}) or
                     {}).get("instruction_relevant_detections", []))
                instruction_text_for_region_gate = " ".join(
                    str(compact_instruction.get(key, "")) for key in (
                        "navigation_instruction", "landmark",
                        "semantic_spatial_target", "spatial_relation",
                        "completion_cue", "visual_arrival_evidence"))
                stage_progress_context = (
                    (carryover_evidence or {}).get("stage_progress", {})
                    if isinstance(carryover_evidence, dict) else {}) or {}
                current_edge_context = (
                    (carryover_evidence or {}).get("current_edge", {})
                    if isinstance(carryover_evidence, dict) else {}) or {}
                circumnavigation_route_integrity = (
                    circumnavigation_stage_route_integrity_supported(
                        instruction_text_for_region_gate,
                        current_edge_traveled_m=float(
                            structured_motion.get(
                                "traveled_distance_m", 0.0) or 0.0),
                        stage_progress_context=stage_progress_context,
                        current_selected_yaw_rad=current_edge_context.get(
                            "selected_yaw_rad"))
                    if (form == "CIRCUMNAVIGATE" and
                        self.instruction_completion_prompt_version ==
                        "v24_multireference_threshold_calibration") else True)
                between_route_integrity = (
                    between_stage_route_integrity_supported(
                        instruction_text_for_region_gate,
                        current_edge_traveled_m=float(
                            structured_motion.get(
                                "traveled_distance_m", 0.0) or 0.0),
                        stage_progress_context=stage_progress_context,
                        current_selected_yaw_rad=current_edge_context.get(
                            "selected_yaw_rad"))
                    if (form == "BETWEEN_OBJECTS" and
                        self.instruction_completion_prompt_version ==
                        "v24_multireference_threshold_calibration") else True)
                # ENTER_REGION targets such as "room with cardboard boxes" or
                # "entrance under the balcony" contain an identity-bearing
                # qualifier.  A generic doorway/room transition is not enough:
                # at least one non-generic qualifier token must be supported by
                # the current node's RGB-aligned open-vocabulary detections.
                # This is a form-level observation gate; it does not inspect
                # depth, navmesh geometry, demonstration paths, or episode IDs.
                qualified_region_tokens = set()
                if form == "ENTER_REGION":
                    for match in re.finditer(
                            r"\b(?:with|under|beneath|containing)\s+"
                            r"([a-z][a-z\s-]{1,80})",
                            instruction_text_for_region_gate.lower()):
                        phrase = re.split(
                            r"[.;,]|\b(?:and|after|before|until|then)\b",
                            match.group(1), maxsplit=1)[0]
                        qualified_region_tokens.update(
                            token for token in re.findall(r"[a-z]+", phrase)
                            if token not in {
                                "a", "an", "the", "free", "walkable",
                                "floor", "ground", "area", "region", "room",
                                "entrance", "door", "doorway", "opening",
                                "inside", "outside", "just", "camera", "is",
                                "visible", "structure", "located", "reach",
                                "reached", "sees", "has",
                            })
                current_region_evidence_tokens = {
                    str(token).lower()
                    for hit in current_node_relevant_for_gate
                    if float(hit.get("score", 0.0) or 0.0) >= 0.30
                    for token in (hit.get("matched_tokens", []) or [])
                }
                qualified_region_identity_supported = bool(
                    not qualified_region_tokens or
                    qualified_region_tokens.intersection(
                        current_region_evidence_tokens))
                terminal_extent_front_clear = (
                    terminal_extent_front_landmark_clear(
                        instruction_text_for_region_gate,
                        current_node_relevant_for_gate))
                cumulative_stage_travel_m = float(structured_motion.get(
                    "traveled_distance_m", 0.0) or 0.0)
                if stage_progress_context.get("active"):
                    cumulative_stage_travel_m += float(
                        stage_progress_context.get(
                            "prior_edge_traveled_distance_m", 0.0) or 0.0)
                terminal_extent_rgb_completion_override = bool(
                    terminal_extent_rgb_completion_supported(
                        instruction_text_for_region_gate,
                        current_target_state=current_state,
                        current_completion_cue=completion_cue,
                        same_instance=same_instance,
                        semantic_order=semantic_order,
                        keyframe_support=keyframe_support,
                        motion_fit=motion_fit,
                        stationary_or_blocked=stationary,
                        reverse_transition_observed=reverse_observed,
                        current_reference_sectors=current_sector_set,
                        cumulative_stage_travel_m=(
                            cumulative_stage_travel_m)))
                terminal_extent_rgb_override = bool(
                    not terminal_extent_front_clear and
                    terminal_extent_rgb_rear_clear_supported(
                        current_target_state=current_state,
                        current_completion_cue=completion_cue,
                        same_instance=same_instance,
                        semantic_order=semantic_order,
                        boundary_event=boundary_event,
                        keyframe_support=keyframe_support,
                        motion_fit=motion_fit,
                        stationary_or_blocked=stationary,
                        reverse_transition_observed=reverse_observed,
                        current_reference_sectors=current_sector_set))
                terminal_extent_front_clear = bool(
                    terminal_extent_front_clear or
                    terminal_extent_rgb_override or
                    terminal_extent_rgb_completion_override)
                relation_geometry_guard_active = bool(
                    self.instruction_completion_prompt_version in {
                        "v23_relation_geometry_guard",
                        "v24_multireference_threshold_calibration"})
                terminal_relation_identity_supported = bool(
                    not relation_geometry_guard_active or
                    terminal_relation_detection_supported(
                        form, current_node_relevant_for_gate) or
                    terminal_relation_rgb_consensus_supported(
                        same_instance=same_instance,
                        current_target_state=current_state,
                        current_completion_cue=completion_cue,
                        current_reference_sectors=current_sector_set))
                under_relation_supported = bool(
                    not relation_geometry_guard_active or
                    under_relation_multiview_supported(
                        instruction_text_for_region_gate,
                        current_node_relevant_for_gate))
                between_lateral_bracket_supported = (
                    between_gap_lateral_bracket_supported(
                        form, instruction_text_for_region_gate,
                        compact_instruction.get("landmark", ""),
                        current_node_relevant_for_gate))
                coordinated_pass = bool(
                    form == "PASS_LANDMARK" and
                    re.search(r"\b(?:and|along with|as well as)\b",
                              str(compact_instruction.get(
                                  "landmark", "")).lower()))
                identity_required = bool(form in {
                    "PASS_LANDMARK", "CIRCUMNAVIGATE", "APPROACH_LANDMARK",
                    "BETWEEN_OBJECTS"} and not coordinated_pass)
                turn_motion_required = form in {
                    "TURN_LEFT", "TURN_RIGHT", "TURN_AROUND"}
                deterministic_turn_direction = turn_direction_motion_supported(
                    form, structured_motion,
                    minimum_turn_deg=(
                        40.0 if self.instruction_completion_prompt_version ==
                        "v24_multireference_threshold_calibration" else 45.0))
                deterministic_turn_direction = bool(
                    deterministic_turn_direction or
                    explicit_turn_selection_supported(
                        form, current_edge_context.get(
                            "point_selection_review", {})))
                gates = {
                    "previous_target_not_already_satisfied": (
                        previous_state != "satisfied"),
                    "current_full_target_satisfied": current_state == "satisfied",
                    "current_completion_cue_satisfied": completion_cue == "satisfied",
                    "semantic_order_instructed": semantic_order == "instructed",
                    "completion_boundary_supported": boundary_event in {
                        "observed", "endpoint_inferred"},
                    "chronological_keyframes_support_transition": keyframe_support,
                    "reference_identity_supported": (
                        same_instance == "yes" if identity_required else
                        same_instance in {"yes", "not_applicable"}),
                    "no_reverse_transition": not reverse_observed,
                    "motion_not_contradictory": motion_fit != "contradicts",
                    "required_turn_motion_supported": (
                        motion_fit == "supports" and
                        deterministic_turn_direction
                        if turn_motion_required else True),
                    "not_stationary_or_blocked": not stationary,
                    "qualified_region_identity_supported": (
                        qualified_region_identity_supported),
                    "explicit_terminal_extent_front_clear": (
                        terminal_extent_front_clear),
                    "terminal_relation_identity_supported": (
                        terminal_relation_identity_supported),
                    "under_relation_multiview_supported": (
                        under_relation_supported),
                    "circumnavigation_stage_route_integrity": (
                        circumnavigation_route_integrity),
                    "between_stage_route_integrity": (
                        between_route_integrity),
                    "between_gap_lateral_bracket_supported": (
                        between_lateral_bracket_supported),
                }
                vertical_evidence = None
                if vertical_guard_active:
                    raw_vertical = result["vertical_endpoint_evidence"]
                    vertical_evidence = {
                        "current_level_landing": bool(
                            raw_vertical["current_level_landing"]),
                        "remaining_stair_flight_visible": bool(
                            raw_vertical["remaining_stair_flight_visible"]),
                        "endpoint_evidence_stable": bool(
                            raw_vertical["endpoint_evidence_stable"]),
                    }
                    # This gate is only active for complete up/down targets;
                    # it is deliberately not a metric-height heuristic and is
                    # applied uniformly to every episode with this form.
                    vertical_form = form in {"VERTICAL_UP", "VERTICAL_DOWN"}
                    vertical_text = " ".join(str(compact_instruction.get(key, ""))
                                              for key in (
                                                  "navigation_instruction",
                                                  "semantic_spatial_target",
                                                  "completion_cue",
                                                  "visual_arrival_evidence"))
                    full_vertical_target = bool(
                        vertical_form and not any(token in vertical_text.lower()
                                                  for token in (
                                                      "halfway", "half way",
                                                      "partway", "part way")))
                    if full_vertical_target:
                        semantic_views = list(
                            (current_environment_semantics or {}).get(
                                "views", []))
                        # Only the exact forward camera can hard-veto a VLM
                        # top/bottom-landing judgement. At a genuine landing,
                        # the completed (descending) flight is commonly still
                        # visible in a +/-45 or +/-60 degree side-forward
                        # camera. Treating those side rays as "stairs ahead"
                        # caused a verified upper landing to remain unknown.
                        # Likewise, a proposal explicitly labelled as a stair
                        # *landing* is not evidence of an unfinished flight.
                        # The clean-RGB VLM and chronological keyframes still
                        # have to assert a stable level endpoint.
                        forward_view_indices = {0}
                        detector_forward_stair_hits = []
                        for view in semantic_views:
                            view_index = int(view.get("view_index", -1))
                            if view_index not in forward_view_indices:
                                continue
                            threshold = 0.30 if view_index == 0 else 0.33
                            for detection in view.get("detections", []):
                                label = str(detection.get("label", "")).lower()
                                score = float(detection.get("score", 0.0) or 0.0)
                                if ("stair" in label and
                                        "landing" not in label and
                                        score >= threshold):
                                    detector_forward_stair_hits.append({
                                        "view_index": view_index,
                                        "label": label,
                                        "score": score,
                                    })
                        detector_remaining_stair_front = bool(
                            detector_forward_stair_hits)
                        vertical_evidence.update({
                            "detector_remaining_stair_front": (
                                detector_remaining_stair_front),
                            "detector_forward_stair_hits": (
                                detector_forward_stair_hits),
                        })
                        gates.update({
                            "vertical_level_landing_supported": (
                                vertical_evidence["current_level_landing"]),
                            "no_remaining_stair_flight": bool(
                                not vertical_evidence[
                                    "remaining_stair_flight_visible"] and
                                not detector_remaining_stair_front),
                            "vertical_endpoint_stable": vertical_evidence[
                                "endpoint_evidence_stable"],
                        })
                # Pure orientation commands can be executed by an in-place
                # target-alignment phase that is represented by the stored
                # node-heading delta, while the edge action list contains only
                # subsequent translation.  Combine that delta with the VLM's
                # cross-panorama sectors as a generic fallback.
                orientation_override = False
                orientation_evidence = None
                if structured_binary_completion and structured_motion:
                    heading_delta = abs(float(structured_motion.get(
                        "endpoint_heading_delta_deg", 0.0)))
                    previous_sector_set = set(
                        previous_sectors)
                    current_sector_set = set(current_sectors)
                    if form == "TURN_AROUND":
                        orientation_flip = bool(
                            previous_sector_set.intersection(
                                {"rear", "rear_left", "rear_right"}) and
                            current_sector_set.intersection(
                                {"front", "front_left", "front_right"}))
                        # A U-turn may rotate into a different corridor, so
                        # the original landmark/door need not remain a
                        # confidently matched instance in the current
                        # panorama.  When stored node headings and the
                        # chronological edge both show a large reversal plus
                        # real translation, treat that as generic orientation
                        # evidence; retain the visual flip path when it is
                        # available, but do not make a noisy detector identity
                        # a necessary condition.
                        geometry_orientation_support = bool(
                            # Real actuator trajectories can undershoot a
                            # nominal 180-degree command.  A >=120-degree
                            # reversal is still a strong, form-specific
                            # orientation completion signal when it is backed
                            # by translation, control steps, and turn actions.
                            heading_delta >= 120.0 and
                            float(structured_motion.get(
                                "horizontal_displacement_m", 0.0)) >= 0.20 and
                            float(structured_motion.get(
                                "traveled_distance_m", 0.0)) >= 0.50 and
                            int(structured_motion.get(
                                "control_steps", 0)) >= 2 and
                            not stationary and
                            any(name in structured_motion.get(
                                "action_counts", {}) for name in (
                                    "turn_left", "turn_right")))
                        # A reverse transition is the requested completion for
                        # TURN_AROUND itself, so it is positive evidence here;
                        # the generic reverse veto remains active for all
                        # other instruction forms.
                        orientation_override = bool(
                            ((heading_delta >= 135.0 and orientation_flip) or
                             geometry_orientation_support) and
                            not stationary)
                        if orientation_override:
                            orientation_evidence = {
                                "endpoint_heading_delta_deg": heading_delta,
                                "previous_rear_to_current_front": orientation_flip,
                                "geometry_orientation_support": (
                                    geometry_orientation_support),
                                "source": "node_heading_and_eight_view_relation",
                            }
                    elif form == "TURN_TO_LANDMARK":
                        landmark_front = bool(current_sector_set.intersection(
                            {"front", "front_left", "front_right"}))
                        landmark_rear = bool(current_sector_set.intersection(
                            {"rear", "rear_left", "rear_right"}))
                        orientation_override = bool(
                            heading_delta >= 45.0 and landmark_front and
                            not landmark_rear and
                            current_state == "satisfied" and
                            completion_cue == "satisfied" and
                            semantic_order == "instructed" and
                            boundary_event in {"observed", "endpoint_inferred"} and
                            keyframe_support and
                            same_instance in {"yes", "not_applicable"} and
                            motion_fit == "supports" and
                            not stationary and not reverse_observed)
                        if orientation_override:
                            orientation_evidence = {
                                "endpoint_heading_delta_deg": heading_delta,
                                "landmark_in_current_forward_hemisphere": landmark_front,
                                "source": "node_heading_and_named_landmark_sectors",
                            }
                # For a PASS_LANDMARK edge, the project-level eight-view rule
                # treats a same-reference bearing moving from the forward
                # hemisphere at PREVIOUS to any rear-side sector at CURRENT
                # as the pass boundary.  DeepSeek sometimes marks the target
                # ``partial`` or ``same_reference_instance=ambiguous`` when a
                # noisy detector also fires in a forward sector.  Keep the rule
                # tied to the full temporal evidence (rear endpoint,
                # supported forward motion, keyframes, and no reversal) and
                # apply it uniformly instead of letting that single noisy
                # label force an unnecessary branch/reselection.
                pass_directional_override = False
                pass_directional_evidence = None
                if (structured_binary_completion and
                        form == "PASS_LANDMARK" and
                        previous_state != "satisfied" and
                        current_state == "satisfied" and
                        completion_cue == "satisfied" and
                        same_instance == "yes" and
                        current_sector_set.intersection(
                            {"rear_left", "rear", "rear_right"}) and
                        # In an eight-view panorama a long landmark (stairs,
                        # counter, railing) can remain visible in both rear and
                        # forward sectors after its pass boundary.  The rear
                        # sector is therefore sufficient only together with
                        # instructed temporal order, supporting keyframes,
                        # real translation, and no observed reverse transition.
                        semantic_order == "instructed" and keyframe_support and
                        motion_fit == "supports" and not stationary and
                        not reverse_observed and
                        float(structured_motion.get(
                            "traveled_distance_m", 0.0)) >= 0.50):
                    pass_directional_override = True
                    pass_directional_evidence = {
                        "rule": "motion-supported rear-side eight-view pass boundary",
                        "previous_sectors": sorted(previous_sector_set),
                        "current_sectors": sorted(current_sector_set),
                        "semantic_order": semantic_order,
                        "keyframe_support": bool(keyframe_support),
                        "motion_fit": motion_fit,
                        "traveled_distance_m": float(structured_motion.get(
                            "traveled_distance_m", 0.0)),
                        "reverse_transition_observed": bool(reverse_observed),
                    }
                circumnavigation_rear_endpoint_override = False
                circumnavigation_rear_endpoint_evidence = None
                rear_relation_sectors = {"rear_left", "rear", "rear_right"}
                if (structured_binary_completion and
                        form == "CIRCUMNAVIGATE" and
                        previous_state != "satisfied" and
                        current_state == "satisfied" and
                        completion_cue == "satisfied" and
                        same_instance == "yes" and
                        semantic_order == "instructed" and
                        bool(current_sector_set.intersection(
                            rear_relation_sectors)) and
                        not stationary and not reverse_observed and
                        motion_fit != "contradicts" and
                        keyframe_support and
                        float(structured_motion.get(
                            "horizontal_displacement_m", 0.0)) >= 1.0 and
                        float(structured_motion.get(
                            "traveled_distance_m", 0.0)) >= 1.5):
                    # Project rule: in an eight-view endpoint panorama, the
                    # tracked landmark appearing in rear/rear-side sectors is
                    # sufficient evidence that a backside/around
                    # waypoint has been passed. This also handles episodes
                    # that begin with the object already behind the camera,
                    # where requiring a front->rear transition is impossible.
                    circumnavigation_rear_endpoint_override = True
                    circumnavigation_rear_endpoint_evidence = {
                        "rule": "rear-present eight-view circumnavigation endpoint",
                        "previous_sectors": sorted(previous_sector_set),
                        "current_sectors": sorted(current_sector_set),
                        "horizontal_displacement_m": float(
                            structured_motion.get(
                                "horizontal_displacement_m", 0.0)),
                        "traveled_distance_m": float(structured_motion.get(
                            "traveled_distance_m", 0.0)),
                    }
                if orientation_override:
                    gates.update({
                        "current_full_target_satisfied": True,
                        "current_completion_cue_satisfied": True,
                        "semantic_order_instructed": True,
                        "completion_boundary_supported": True,
                        "chronological_keyframes_support_transition": True,
                        "reference_identity_supported": True,
                        "no_reverse_transition": True,
                        "motion_not_contradictory": True,
                        "required_turn_motion_supported": True,
                        "not_stationary_or_blocked": True,
                    })
                if pass_directional_override:
                    gates.update({
                        "current_full_target_satisfied": True,
                        "current_completion_cue_satisfied": True,
                        "semantic_order_instructed": True,
                        "completion_boundary_supported": True,
                        "chronological_keyframes_support_transition": True,
                        "reference_identity_supported": True,
                        "no_reverse_transition": True,
                        "motion_not_contradictory": True,
                        "required_turn_motion_supported": True,
                        "not_stationary_or_blocked": True,
                    })
                if circumnavigation_rear_endpoint_override:
                    gates.update({
                        "current_full_target_satisfied": True,
                        "current_completion_cue_satisfied": True,
                        "semantic_order_instructed": True,
                        "completion_boundary_supported": True,
                        "chronological_keyframes_support_transition": True,
                        "reference_identity_supported": True,
                        "no_reverse_transition": True,
                        "motion_not_contradictory": True,
                        "required_turn_motion_supported": True,
                        "not_stationary_or_blocked": True,
                    })
                if terminal_extent_rgb_completion_override:
                    gates.update({
                        "current_full_target_satisfied": True,
                        "current_completion_cue_satisfied": True,
                        "semantic_order_instructed": True,
                        "completion_boundary_supported": True,
                        "chronological_keyframes_support_transition": True,
                        "reference_identity_supported": True,
                        "no_reverse_transition": True,
                        "motion_not_contradictory": True,
                        "not_stationary_or_blocked": True,
                        "explicit_terminal_extent_front_clear": True,
                    })
                # EXIT_REGION must end on the destination side of its portal.
                # If the current endpoint still reports the same portal in
                # both forward and rear hemispheres, the crossing is visually
                # ambiguous (typically a glass/doorway false positive). Keep
                # the binary result UNKNOWN unless the endpoint panorama has
                # a one-sided rear portal relation, as required by the form.
                exit_semantic_text = " ".join(
                    str(compact_instruction.get(key, "")) for key in (
                        "navigation_instruction", "semantic_spatial_target",
                        "completion_cue", "visual_arrival_evidence")
                ).lower()
                # Frozen decompositions created before the current taxonomy
                # sometimes label an explicit exit-to-outside clause OTHER.
                # Keep the conservative ambiguity guard broad (a doorway is
                # enough to require portal evidence), but only grant positive
                # EXIT_REGION recovery to OTHER when the text explicitly
                # names leaving/exiting or the outside destination.  This
                # normalizes semantic aliases without promoting a generic
                # "approach the doorway" OTHER clause.
                exit_guard_like = bool(
                    form == "EXIT_REGION" or (
                        form == "OTHER" and any(
                            token in exit_semantic_text for token in (
                                "outside", "outdoors", "outdoor", "doorway",
                                "door to outside", "exit"))))
                exit_positive_like = bool(
                    form == "EXIT_REGION" or (
                        form == "OTHER" and any(
                            token in exit_semantic_text for token in (
                                "outside", "outdoors", "outdoor",
                                "door to outside", "exit", "leave the room",
                                "leave room"))))
                exit_front_rear_ambiguity_veto = bool(
                    # EXIT_REGION is the decomposer's canonical form, but
                    # older/less specific decompositions may classify a
                    # doorway-to-outside clause as OTHER.  Detect that
                    # semantic portal target generically from the frozen
                    # sub-instruction text so the same visual ambiguity rule
                    # applies without episode-specific overrides.
                    exit_guard_like and
                    current_sector_set.intersection(
                        {"front", "front_left", "front_right"}) and
                    current_sector_set.intersection(
                        {"rear", "rear_left", "rear_right"}))
                if exit_front_rear_ambiguity_veto:
                    gates.update({
                        "current_full_target_satisfied": False,
                        "current_completion_cue_satisfied": False,
                        "completion_boundary_supported": False,
                    })
                # A doorway can remain visible in a side/front sector after a
                # genuine crossing because the eight-view panorama overlaps
                # adjacent bearings.  For an exit-like clause, allow the
                # rear-side relation to resolve that ambiguity only when it
                # strictly dominates the forward-side sectors and the edge
                # independently proves an ordered crossing.  A balanced
                # front/rear report (e.g. a doorway viewed in place) remains
                # UNKNOWN under the veto above.
                exit_rear_dominance_override = False
                if exit_guard_like:
                    rear_count = len(current_sector_set.intersection(
                        {"rear", "rear_left", "rear_right"}))
                    front_count = len(current_sector_set.intersection(
                        {"front", "front_left", "front_right"}))
                    exit_rear_dominance_override = bool(
                        previous_state != "satisfied" and
                        current_state == "satisfied" and
                        completion_cue == "satisfied" and
                        semantic_order == "instructed" and
                        boundary_event in {"observed", "endpoint_inferred"} and
                        keyframe_support and motion_fit == "supports" and
                        not stationary and not reverse_observed and
                        rear_count >= front_count + 1 and
                        float(structured_motion.get(
                            "traveled_distance_m", 0.0)) >= 0.50)
                if exit_rear_dominance_override:
                    gates.update({
                        "current_full_target_satisfied": True,
                        "current_completion_cue_satisfied": True,
                        "semantic_order_instructed": True,
                        "completion_boundary_supported": True,
                        "chronological_keyframes_support_transition": True,
                        "reference_identity_supported": True,
                        "no_reverse_transition": True,
                        "motion_not_contradictory": True,
                        "required_turn_motion_supported": True,
                        "not_stationary_or_blocked": True,
                    })
                exit_temporal_crossing_override = False
                if exit_positive_like:
                    exit_temporal_crossing_override = bool(
                        previous_state != "satisfied" and
                        current_state in {"satisfied", "partial"} and
                        completion_cue in {"satisfied", "partial"} and
                        bool(current_sector_set.intersection(
                            {"rear", "rear_left", "rear_right"})) and
                        semantic_order == "instructed" and keyframe_support and
                        same_instance == "yes" and
                        motion_fit == "supports" and not stationary and
                        not reverse_observed and
                        float(structured_motion.get(
                            "horizontal_displacement_m", 0.0)) >= 1.0 and
                        float(structured_motion.get(
                            "traveled_distance_m", 0.0)) >= 1.5)
                if exit_temporal_crossing_override:
                    # Door masks overlap adjacent front/rear views near a
                    # threshold. Ordered keyframes, same-instance evidence,
                    # rear portal visibility and real translation jointly
                    # establish the crossing even if the VLM calls the
                    # endpoint merely partial.
                    gates.update({
                        "current_full_target_satisfied": True,
                        "current_completion_cue_satisfied": True,
                        "semantic_order_instructed": True,
                        "completion_boundary_supported": True,
                        "chronological_keyframes_support_transition": True,
                        "reference_identity_supported": True,
                        "no_reverse_transition": True,
                        "motion_not_contradictory": True,
                        "required_turn_motion_supported": True,
                        "not_stationary_or_blocked": True,
                    })
                endpoint_recovery = False
                endpoint_recovery_evidence = None
                if (self.instruction_completion_prompt_version in {
                        "v20_stage_endpoint_recovery",
                        "v21_stage_endpoint_recovery_structured",
                        "v22_stage2_enter_transition",
                        "v23_relation_geometry_guard",
                        "v24_multireference_threshold_calibration"}):
                    current_node_sem = (structured_semantics or {}).get(
                        "current_node", {})
                    current_relevant = list(current_node_sem.get(
                        "instruction_relevant_detections", []))
                    current_scores = [float(hit.get("score", 0.0))
                                      for hit in current_relevant]
                    current_match = bool(current_relevant and
                                         max(current_scores, default=0.0) >= 0.30)
                    moved = bool(float(structured_motion.get(
                        "traveled_distance_m", 0.0)) >= 0.20)
                    supports = bool(motion_fit == "supports" and moved and
                                    not stationary and not reverse_observed)
                    noncontradictory = bool(
                        motion_fit != "contradicts" and moved and
                        not stationary and not reverse_observed)
                    rear = bool(current_sector_set.intersection(
                        {"rear", "rear_left", "rear_right"}))
                    front = bool(current_sector_set.intersection(
                        {"front", "front_left", "front_right"}))
                    previous_node_relevant = list((structured_semantics or {}).get(
                        "previous_node", {}).get(
                            "instruction_relevant_detections", []))
                    instruction_text_lower = " ".join(
                        str(compact_instruction.get(key, "")) for key in (
                            "navigation_instruction", "semantic_spatial_target",
                            "completion_cue", "visual_arrival_evidence")).lower()
                    has_stairs = bool(re.search(
                        r"\bstairs?\b|\bstair\b", instruction_text_lower))
                    if form == "PASS_LANDMARK":
                        # If the VLM itself calls the endpoint partial, a
                        # named object still strongly visible in FRONT is a
                        # generic veto for a pass/behind relation.  This does
                        # not veto a room that the VLM judged satisfied merely
                        # because a large room remains visible in side views.
                        pass_front_conflict = bool(
                            current_state != "satisfied" and any(
                                str(hit.get("direction", "")).lower() in {
                                    "front", "front_left", "front_right"} and
                                float(hit.get("score", 0.0)) >= 0.30 and
                                str(hit.get("matched_tokens", [])) not in {
                                    "['corridor']", "['hallway']"}
                                for hit in current_relevant))
                        endpoint_recovery = bool(
                            semantic_order == "instructed" and keyframe_support and
                            same_instance == "yes" and rear and supports and
                            not pass_front_conflict)
                        endpoint_recovery_evidence = {
                            "rule": "matched reference in rear after instructed motion",
                            "rear_sector": rear, "keyframe_support": keyframe_support,
                            "same_reference_instance": same_instance,
                            "motion_support": supports,
                            "front_conflict": pass_front_conflict,
                        }
                    elif form == "CIRCUMNAVIGATE":
                        detector_sectors = {
                            str(hit.get("direction", "")).lower()
                            for hit in current_relevant
                            if float(hit.get("score", 0.0) or 0.0) >= 0.28}
                        endpoint_recovery = (
                            circumnavigation_rear_endpoint_supported(
                                reference_sectors=current_sector_set,
                                detector_sectors=detector_sectors,
                                current_state=current_state,
                                reverse_observed=reverse_observed,
                                stationary=stationary,
                                motion_fit=motion_fit,
                                horizontal_displacement_m=float(
                                    structured_motion.get(
                                        "horizontal_displacement_m", 0.0)),
                                traveled_distance_m=float(
                                    structured_motion.get(
                                        "traveled_distance_m", 0.0))))
                        stagewide_partial_endpoint = False
                        if self.instruction_completion_prompt_version in {
                                "v23_relation_geometry_guard",
                                "v24_multireference_threshold_calibration"}:
                            # Rear-side detector overlap corroborates a
                            # satisfied endpoint; it cannot overrule two VLM
                            # calls that still report partial/ambiguous.
                            stagewide_partial_endpoint = bool(
                                endpoint_recovery and
                                circumnavigation_route_integrity and
                                isinstance(stage_progress_context, dict) and
                                stage_progress_context.get("active") and
                                current_state in {"partial", "satisfied"} and
                                completion_cue in {"partial", "satisfied"})
                            endpoint_recovery = bool(
                                endpoint_recovery and
                                circumnavigation_route_integrity and (
                                    (current_state == "satisfied" and
                                     completion_cue == "satisfied" and
                                     semantic_order == "instructed" and
                                     keyframe_support) or
                                    stagewide_partial_endpoint))
                        endpoint_recovery_evidence = {
                            "rule": (
                                "focused detector plus eight-view rear-side "
                                "circumnavigation endpoint"),
                            "reference_sectors": sorted(current_sector_set),
                            "detector_sectors": sorted(detector_sectors),
                            "exact_front_conflict": bool(
                                "front" in current_sector_set or
                                "front" in detector_sectors),
                            "horizontal_displacement_m": float(
                                structured_motion.get(
                                    "horizontal_displacement_m", 0.0)),
                            "traveled_distance_m": float(
                                structured_motion.get(
                                    "traveled_distance_m", 0.0)),
                            "stage_route_integrity": bool(
                                circumnavigation_route_integrity),
                            "stagewide_partial_endpoint": bool(
                                stagewide_partial_endpoint),
                        }
                    elif form in {"ENTER_REGION", "EXIT_REGION",
                                  "TRAVERSE_PORTAL_REGION", "SELECT_PORTAL"}:
                        portal_identity_ok = same_instance == "yes"
                        enter_transition = False
                        previous_opening_front = False
                        if (self.instruction_completion_prompt_version in {
                                "v22_stage2_enter_transition",
                                "v23_relation_geometry_guard",
                                "v24_multireference_threshold_calibration"} and
                                form == "ENTER_REGION"):
                            # A destination structure can legitimately occupy
                            # both front and rear sectors after crossing it;
                            # detector identity/order is often marked
                            # ambiguous in that configuration.  Recover only
                            # when the current named token spans the camera,
                            # the edge has chronological keyframe support and
                            # supported motion, and the source view shows an
                            # opening (or the edge has enough continuous
                            # travel to cross a portal).  This is a generic
                            # RGB/action-history rule, not a case override.
                            previous_views = list((structured_semantics or {})
                                                 .get("previous_node", {})
                                                 .get("views", []))
                            previous_opening_front = any(
                                str(view.get("direction", "")).lower() in {
                                    "front", "front_left", "front_right"} and
                                any(
                                    float(det.get("score", 0.0)) >= 0.28 and
                                    re.search(r"\b(door|doorway|opening|portal|hallway|corridor)\b",
                                              str(det.get("label", "")).lower())
                                    for det in view.get("top_detections", []))
                                for view in previous_views)
                            # The VLM's temporal fields are useful evidence,
                            # but are often marked ``ambiguous`` after a
                            # portal crossing because the destination spans
                            # several panorama sectors.  For ENTER_REGION,
                            # use the observation geometry and stored motion
                            # as the primary, model-independent transition
                            # test: the named target must persist in both the
                            # forward and rear hemispheres after a meaningful
                            # continuous advance, with an opening at the
                            # source (or enough travel to cross one).  This is
                            # a form-level rule and intentionally does not
                            # depend on an episode's object names.
                            travel_m = float(structured_motion.get(
                                "traveled_distance_m", 0.0))
                            enter_transition = bool(
                                current_match and front and rear and
                                travel_m >= 1.0 and
                                not stationary and
                                motion_fit != "contradicts" and
                                not reverse_observed and
                                (previous_opening_front or travel_m >= 1.5))
                            # The v19 judge may mark a destination region
                            # partial even after the named structure spans
                            # both front and rear sectors.  Once continuous
                            # motion has carried the agent through at least a
                            # portal-length distance, this is generic inside
                            # evidence even when the source opening was not
                            # detected in the compressed panorama.
                            if (not enter_transition and current_match and
                                    front and rear and travel_m >= 1.5 and
                                    not stationary and
                                    motion_fit != "contradicts" and
                                    not reverse_observed):
                                enter_transition = True
                                portal_identity_ok = True
                            if enter_transition:
                                portal_identity_ok = True
                        # A generic, non-ordinal doorway can remain ambiguous
                        # to the VLM when a second opening is visible ahead.
                        # Preserve identity conservatively using a strong
                        # rear doorway plus ordered keyframes; this is a form
                        # rule, not an episode-specific exception.
                        if (self.instruction_completion_prompt_version in {
                                "v21_stage_endpoint_recovery_structured",
                                "v23_relation_geometry_guard",
                                "v24_multireference_threshold_calibration"} and
                                form == "TRAVERSE_PORTAL_REGION" and
                                not re.search(r"\b(first|second|third|next|left|right|ordinal)\b",
                                              instruction_text_lower)):
                            rear_portal_score = max((float(hit.get("score", 0.0))
                                                     for hit in current_relevant
                                                     if str(hit.get("direction", "")) in {
                                                         "rear", "rear_left", "rear_right"}
                                                     and re.search(r"\b(door|doorway|portal|opening)\b",
                                                                   str(hit.get("label", "")).lower())),
                                                    default=0.0)
                            previous_front_portal_score = max((
                                float(hit.get("score", 0.0)) for hit in
                                previous_node_relevant if str(hit.get(
                                    "direction", "")) in {
                                        "front", "front_left", "front_right"}
                                and re.search(r"\b(door|doorway|portal|opening)\b",
                                              str(hit.get("label", "")).lower())),
                                default=0.0)
                            source_to_destination_transition = bool(
                                previous_front_portal_score >= 0.30 and
                                rear_portal_score >= 0.30 and
                                keyframe_support and noncontradictory)
                            portal_identity_ok = bool(
                                portal_identity_ok or
                                (rear_portal_score >= 0.38 and keyframe_support and
                                 semantic_order == "instructed") or
                                source_to_destination_transition)
                        else:
                            rear_portal_score = 0.0
                            previous_front_portal_score = 0.0
                            source_to_destination_transition = False
                        portal_order_ok = semantic_order == "instructed"
                        if (self.instruction_completion_prompt_version in {
                                "v22_stage2_enter_transition",
                                "v23_relation_geometry_guard",
                                "v24_multireference_threshold_calibration"} and
                                form == "ENTER_REGION" and enter_transition):
                            portal_order_ok = True
                        if (self.instruction_completion_prompt_version in {
                                "v21_stage_endpoint_recovery_structured",
                                "v23_relation_geometry_guard",
                                "v24_multireference_threshold_calibration"} and
                                form == "TRAVERSE_PORTAL_REGION"):
                            portal_order_ok = bool(
                                portal_order_ok or source_to_destination_transition)
                        enter_full_endpoint_required = bool(
                            form != "ENTER_REGION" or
                            enter_transition or
                            (current_state == "satisfied" and
                             completion_cue == "satisfied" and
                             boundary_event in {"observed", "endpoint_inferred"}))
                        qualified_enter_endpoint = bool(
                            form == "ENTER_REGION" and re.search(
                                r"\b(?:under|beneath|between|corner|far end|"
                                r"end of)\b", instruction_text_lower))
                        qualified_enter_endpoint_satisfied = bool(
                            not qualified_enter_endpoint or
                            (current_state == "satisfied" and
                             completion_cue == "satisfied" and
                             boundary_event in {
                                 "observed", "endpoint_inferred"}))
                        endpoint_recovery = bool(
                            portal_order_ok and keyframe_support and
                            portal_identity_ok and supports and current_match and
                            (rear or front) and enter_full_endpoint_required and
                            qualified_region_identity_supported and
                            qualified_enter_endpoint_satisfied)
                        endpoint_recovery_evidence = {
                            "rule": "matched destination/portal with ordered edge",
                            "keyframe_support": keyframe_support,
                            "same_reference_instance": same_instance,
                            "portal_identity_supported": portal_identity_ok,
                            "source_to_destination_transition": source_to_destination_transition,
                            "enter_transition": enter_transition,
                            "previous_opening_front": previous_opening_front,
                            "motion_support": supports,
                            "current_relevant_detection": current_match,
                            "enter_full_endpoint_required": (
                                enter_full_endpoint_required),
                            "qualified_enter_endpoint": (
                                qualified_enter_endpoint),
                            "qualified_enter_endpoint_satisfied": (
                                qualified_enter_endpoint_satisfied),
                            "qualified_region_tokens": sorted(
                                qualified_region_tokens),
                            "current_region_evidence_tokens": sorted(
                                current_region_evidence_tokens),
                            "qualified_region_identity_supported": (
                                qualified_region_identity_supported),
                        }
                    elif form in {"TURN_LEFT", "TURN_RIGHT", "TURN_AROUND"}:
                        heading_delta = abs(float(structured_motion.get(
                            "endpoint_heading_delta_deg", 0.0)))
                        turn_threshold_deg = (
                            40.0 if self.instruction_completion_prompt_version ==
                            "v24_multireference_threshold_calibration" else 45.0)
                        stair_guard = bool(has_stairs and current_match)
                        compound_turn_destination = (
                            compound_turn_requires_semantic_endpoint(
                                form, compact_instruction.get(
                                    "navigation_instruction", ""),
                                directional_towards_is_commitment=(
                                    self.instruction_completion_prompt_version ==
                                    "v24_multireference_threshold_calibration")))
                        # A directional turn is also observable from the
                        # referenced landmark crossing the image: an object
                        # that was on the agent's right and is now on its left
                        # is evidence of a right turn (and vice versa).  The
                        # language model's signed ``motion_fit`` occasionally
                        # flips this convention, so use the temporal sector
                        # transition as a geometry-level, form-generic
                        # recovery when identity, motion, and heading change
                        # agree.  This does not rely on an episode or demo
                        # label and keeps the stair continuation veto.
                        previous_sector_set = {
                            str(item).lower() for item in previous_sectors}
                        current_sector_set_for_turn = {
                            str(item).lower() for item in current_sectors}
                        right_sectors = {"right", "front_right", "rear_right"}
                        left_sectors = {"left", "front_left", "rear_left"}
                        right_to_left = bool(
                            previous_sector_set.intersection(right_sectors) and
                            current_sector_set_for_turn.intersection(left_sectors))
                        left_to_right = bool(
                            previous_sector_set.intersection(left_sectors) and
                            current_sector_set_for_turn.intersection(right_sectors))
                        directional_sector_support = bool(
                            same_instance == "yes" and moved and
                            not stationary and heading_delta >= turn_threshold_deg and
                            ((form == "TURN_RIGHT" and right_to_left) or
                             (form == "TURN_LEFT" and left_to_right)))
                        strong_temporal_endpoint = bool(
                            semantic_order == "instructed" and
                            keyframe_support and same_instance == "yes" and
                            motion_fit == "supports")
                        endpoint_recovery = bool(
                            heading_delta >= turn_threshold_deg and moved and
                            not stationary and not reverse_observed and
                            not stair_guard and deterministic_turn_direction and
                            (front or rear) and
                            (directional_sector_support or
                             strong_temporal_endpoint))
                        if directional_sector_support and not stair_guard:
                            endpoint_recovery = True
                        if compound_turn_destination:
                            # "Turn left" can finish at a new heading, while
                            # "turn left and walk to the far side" also names
                            # a semantic endpoint.  Do not let turn geometry
                            # overwrite the model's explicit partial endpoint
                            # evidence for such compound clauses.
                            endpoint_recovery = bool(
                                endpoint_recovery and
                                current_state == "satisfied" and
                                completion_cue == "satisfied" and
                                semantic_order == "instructed" and
                                boundary_event in {
                                    "observed", "endpoint_inferred"} and
                                keyframe_support and
                                same_instance in {"yes", "not_applicable"})
                        portal_threshold_endpoint = False
                        directional_towards_commitment = False
                        ordered_following_route = False
                        prior_turn_direction_supported = False
                        if self.instruction_completion_prompt_version == (
                                "v24_multireference_threshold_calibration"):
                            stage_progress_context = (
                                (carryover_evidence or {}).get(
                                    "stage_progress", {})
                                if isinstance(carryover_evidence, dict) else {})
                            prior_heading_delta = (
                                stage_progress_context.get(
                                    "prior_node_heading_delta_deg")
                                if isinstance(stage_progress_context, dict)
                                else None)
                            prior_turn_direction_supported = bool(
                                prior_heading_delta is not None and
                                turn_direction_motion_supported(
                                    form, {
                                        "endpoint_heading_delta_deg": (
                                            prior_heading_delta),
                                        "signed_cumulative_turn_deg": (
                                            prior_heading_delta),
                                    }, minimum_turn_deg=turn_threshold_deg) and
                                float(stage_progress_context.get(
                                    "prior_edge_traveled_distance_m", 0.0)
                                      or 0.0) >= 1.5 and
                                ((stage_progress_context.get(
                                    "prior_motion_evidence") or {}).get(
                                        "instruction_motion_fit") ==
                                 "supports") and
                                not bool((stage_progress_context.get(
                                    "prior_motion_evidence") or {}).get(
                                        "stationary_or_blocked")))
                            cumulative_turn_direction_supported = bool(
                                deterministic_turn_direction or
                                prior_turn_direction_supported)
                            portal_threshold_endpoint = (
                                compound_turn_portal_threshold_supported(
                                    form=form,
                                    instruction_text=instruction_text_lower,
                                    current_reference_sectors=(
                                        current_sector_set_for_turn),
                                    semantic_order=semantic_order,
                                    boundary_event=boundary_event,
                                    keyframe_support=keyframe_support,
                                    same_reference_instance=same_instance,
                                    motion_fit=motion_fit,
                                    stationary=stationary,
                                    reverse_observed=reverse_observed,
                                    deterministic_turn_direction=(
                                        cumulative_turn_direction_supported),
                                    horizontal_displacement_m=float(
                                        structured_motion.get(
                                            "horizontal_displacement_m", 0.0)),
                                    traveled_distance_m=float(
                                        structured_motion.get(
                                            "traveled_distance_m", 0.0))))
                            towards_clause = bool(re.search(
                                r"\b(?:walk|go|head|move|proceed)\s+towards?\b",
                                instruction_text_lower))
                            directional_towards_commitment = bool(
                                towards_clause and
                                not compound_turn_destination and
                                deterministic_turn_direction and
                                heading_delta >= turn_threshold_deg and
                                moved and not stationary and
                                not reverse_observed and
                                motion_fit == "supports" and keyframe_support and
                                front and float(structured_motion.get(
                                    "horizontal_displacement_m", 0.0)) >= 1.0 and
                                float(structured_motion.get(
                                    "traveled_distance_m", 0.0)) >= 1.5)
                            ordered_following_route = (
                                following_landmark_route_commitment_supported(
                                    current_edge_context.get(
                                        "point_selection_review", {})))
                            if (towards_clause and
                                    not compound_turn_destination and
                                    not (current_state == "satisfied" and
                                         completion_cue == "satisfied" and
                                         semantic_order == "instructed")):
                                endpoint_recovery = False
                            # A towards-clause ends when its route is committed,
                            # not when the named far landmark is reached.  The
                            # VLM's separately structured following-landmark
                            # alignment supplies the ordered visual witness;
                            # motion and the hard side-sector gate supply the
                            # physical one.  This cannot complete the later
                            # landmark clause itself.
                            if (directional_towards_commitment and
                                    ordered_following_route):
                                endpoint_recovery = True
                            if portal_threshold_endpoint:
                                endpoint_recovery = True
                        endpoint_recovery_evidence = {
                            "rule": ("node heading change plus new corridor"
                                      if not directional_sector_support else
                                      "landmark sector transition confirms turn direction"),
                            "heading_delta_deg": heading_delta,
                            "front_or_rear_context": bool(front or rear),
                            "stair_continuation_guard": stair_guard,
                            "directional_sector_support": directional_sector_support,
                            "strong_temporal_endpoint": strong_temporal_endpoint,
                            "compound_turn_destination": (
                                compound_turn_destination),
                            "compound_destination_satisfied": bool(
                                not compound_turn_destination or (
                                    current_state == "satisfied" and
                                    completion_cue == "satisfied" and
                                    boundary_event in {
                                        "observed", "endpoint_inferred"})),
                            "portal_threshold_endpoint": bool(
                                portal_threshold_endpoint),
                            "directional_towards_commitment": bool(
                                directional_towards_commitment),
                            "ordered_following_route": bool(
                                ordered_following_route),
                            "turn_threshold_deg": turn_threshold_deg,
                            "prior_turn_direction_supported": bool(
                                prior_turn_direction_supported),
                            "previous_sectors": sorted(previous_sector_set),
                            "current_sectors": sorted(current_sector_set_for_turn),
                        }
                    elif form == "BETWEEN_OBJECTS":
                        # The relation is defined by the landmark noun phrase,
                        # not by generic generated prose (``clear corridor``,
                        # ``reach gap``).  Match in the same singularized token
                        # space as the detector summary so plural instructions
                        # such as ``chairs`` can match a ``chair`` proposal.
                        pair_words = instruction_tokens(
                            compact_instruction.get("landmark", ""))
                        if len(pair_words) < 2:
                            pair_words = instruction_tokens(
                                compact_instruction.get(
                                    "navigation_instruction", ""))
                        pair_words = list(dict.fromkeys(pair_words))[:4]
                        left_side = {"left", "front_left", "rear_left"}
                        right_side = {"right", "front_right", "rear_right"}
                        pair_directions = {}
                        compound_hits = []
                        for hit in current_relevant:
                            direction = str(hit.get("direction", "")).lower()
                            matched = [str(token).lower() for token in
                                       (hit.get("matched_tokens", []) or [])
                                       if str(token).lower() in pair_words]
                            # Prefer proposals that identify one member of the
                            # pair independently.  DINO query ensembles also
                            # contain compound labels such as ``chair bar``;
                            # those cannot alone prove that two instances
                            # bracket the route, but can fill a genuinely
                            # missing member after singleton evidence is used.
                            if len(matched) > 1:
                                compound_hits.append((matched, direction))
                                continue
                            for token in matched:
                                pair_directions.setdefault(
                                    token, set()).add(direction)
                        for matched, direction in compound_hits:
                            for token in matched:
                                if token not in pair_directions:
                                    pair_directions.setdefault(
                                        token, set()).add(direction)
                        pair_seen = sum(bool(pair_directions.get(token))
                                        for token in pair_words)
                        side_bracket = bool(
                            any(any(direction in left_side for direction in directions)
                                for token, directions in pair_directions.items()
                                if token in pair_words) and
                            any(any(direction in right_side for direction in directions)
                                for token, directions in pair_directions.items()
                                if token in pair_words))
                        sector_degrees = {
                            "front": 0.0, "front_left": 45.0,
                            "left": 90.0, "rear_left": 135.0,
                            "rear": 180.0, "rear_right": -135.0,
                            "right": -90.0, "front_right": -45.0,
                        }
                        named_direction_sets = [
                            directions for token, directions in
                            pair_directions.items()
                            if token in pair_words and directions]
                        panorama_bracket = False
                        for first_index, first_directions in enumerate(
                                named_direction_sets):
                            for second_directions in named_direction_sets[
                                    first_index + 1:]:
                                if any(
                                        abs((sector_degrees[first] -
                                             sector_degrees[second] + 180.0) %
                                            360.0 - 180.0) >= 90.0
                                        for first in first_directions
                                        for second in second_directions
                                        if first in sector_degrees and
                                        second in sector_degrees):
                                    panorama_bracket = True
                                    break
                            if panorama_bracket:
                                break
                        gap_cue = bool(re.search(r"\b(gap|between)\b",
                                                 instruction_text_lower))
                        source_pair_ahead = between_recovery_source_is_forward(
                            previous_sectors)
                        terminal_extent_ok = (
                            between_terminal_extent_recovery_allowed(
                                instruction_text_lower, current_state,
                                completion_cue, boundary_event))
                        between_recovery = (
                            between_panorama_endpoint_recovery_supported(
                                pair_tokens_seen=pair_seen,
                                side_bracket=side_bracket,
                                panorama_bracket=panorama_bracket,
                                source_pair_ahead=source_pair_ahead,
                                terminal_extent_ok=terminal_extent_ok,
                                semantic_order=semantic_order,
                                reverse_observed=reverse_observed,
                                stationary=stationary,
                                motion_fit=motion_fit,
                                keyframe_support=keyframe_support,
                                traveled_distance_m=float(
                                    structured_motion.get(
                                        "traveled_distance_m", 0.0))))
                        endpoint_recovery = bool(
                            current_match and gap_cue and between_recovery and
                            between_route_integrity)
                        if self.instruction_completion_prompt_version in {
                                "v23_relation_geometry_guard",
                                "v24_multireference_threshold_calibration"}:
                            # Pair detections can corroborate an explicitly
                            # satisfied gap endpoint, but broad counter/chair
                            # boxes spanning the panorama cannot promote the
                            # model's partial/ambiguous result by themselves.
                            endpoint_recovery = bool(
                                endpoint_recovery and
                                current_state == "satisfied" and
                                completion_cue == "satisfied" and
                                semantic_order == "instructed" and
                                keyframe_support)
                        endpoint_recovery_evidence = {
                            "rule": "both named objects remain as bracketing evidence",
                            "pair_tokens_seen": pair_seen,
                            "pair_landmark_tokens": pair_words,
                            "pair_token_directions": {
                                token: sorted(directions) for token, directions
                                in pair_directions.items()},
                            "side_bracket": side_bracket,
                            "panorama_bracket": panorama_bracket,
                            "gap_cue": gap_cue,
                            "source_pair_ahead": source_pair_ahead,
                            "terminal_extent_recovery_allowed": (
                                terminal_extent_ok),
                            "semantic_order": semantic_order,
                            "keyframe_support": bool(keyframe_support),
                            "motion_support": supports,
                            "panorama_motion_recovery": between_recovery,
                            "stage_route_integrity": between_route_integrity,
                        }
                    elif form == "OTHER":
                        endpoint_recovery = bool(
                            current_match and (supports or (
                                self.instruction_completion_prompt_version in {
                                    "v21_stage_endpoint_recovery_structured",
                                    "v23_relation_geometry_guard",
                                    "v24_multireference_threshold_calibration"} and
                                noncontradictory)) and
                            float(structured_motion.get(
                                "traveled_distance_m", 0.0)) >= 0.8 and
                            not (has_stairs and current_state == "partial"))
                        endpoint_recovery_evidence = {
                            "rule": "strong current instruction-token endpoint evidence",
                            "current_relevant_detection": current_match,
                            "motion_support": supports,
                            "traveled_distance_m": structured_motion.get(
                                "traveled_distance_m", 0.0),
                        }
                    stair_turn_veto = bool(
                        self.instruction_completion_prompt_version in {
                            "v21_stage_endpoint_recovery_structured",
                            "v23_relation_geometry_guard",
                            "v24_multireference_threshold_calibration"} and
                        form in {"TURN_LEFT", "TURN_RIGHT", "TURN_AROUND"} and
                        has_stairs and current_match and
                        any(str(hit.get("direction", "")).lower() in {
                            "front", "front_left", "front_right"} and
                            re.search(r"\bstairs?\b|\bstair\b",
                                      str(hit.get("label", "")).lower()) and
                            float(hit.get("score", 0.0)) >= 0.28
                            for hit in current_relevant))
                    if stair_turn_veto:
                        endpoint_recovery = False
                        endpoint_recovery_evidence = {
                            "rule": "unfinished stair flight in forward sectors vetoes turn-only completion",
                            "stair_turn_veto": True,
                        }
                        gates.update({
                            "current_full_target_satisfied": False,
                            "current_completion_cue_satisfied": False,
                            "completion_boundary_supported": False,
                            "required_turn_motion_supported": False,
                        })
                    if endpoint_recovery:
                        gates.update({
                            "current_full_target_satisfied": True,
                            "current_completion_cue_satisfied": True,
                            "semantic_order_instructed": True,
                            "completion_boundary_supported": True,
                            "chronological_keyframes_support_transition": (
                                bool(keyframe_support) or form in {
                                    "TURN_LEFT", "TURN_RIGHT", "TURN_AROUND",
                                    "BETWEEN_OBJECTS", "CIRCUMNAVIGATE", "OTHER"}),
                            "reference_identity_supported": True,
                            "no_reverse_transition": True,
                            "motion_not_contradictory": True,
                            "required_turn_motion_supported": True,
                            "not_stationary_or_blocked": True,
                        })
                derived_status = (
                    "completed" if all(gates.values()) else "unknown")
                normalized = derived_status != model_status
                normalized_confidence = (
                    min(confidence, 0.65) if normalized else confidence)
                if endpoint_recovery and derived_status == "completed":
                    # The model's confidence refers to its original UNKNOWN
                    # prose.  Once the deterministic evidence gates establish
                    # a completion, retaining a sub-threshold value makes the
                    # adapter silently discard that verified recovery.  Keep
                    # it at the public acceptance boundary, never above the
                    # conservative normalized cap.
                    normalized_confidence = max(
                        normalized_confidence, 0.50)
                return {
                    "status": derived_status,
                    "model_status": model_status,
                    "status_normalized": normalized,
                    "confidence": normalized_confidence,
                    "reason": str(result.get("reason", "")),
                    "visual_evidence": str(result.get("visual_evidence", "")),
                    "endpoint_evidence": {
                        "previous_target_state": previous_state,
                        "current_target_state": current_state,
                        "current_completion_cue": completion_cue,
                        "reference_previous_sectors": previous_sectors,
                        "reference_current_sectors": current_sectors,
                    },
                    "temporal_evidence": {
                        "semantic_order": semantic_order,
                        "boundary_event": boundary_event,
                        "keyframe_support": keyframe_support,
                        "same_reference_instance": same_instance,
                        "reverse_transition_observed": reverse_observed,
                    },
                    "motion_evidence": {
                        "instruction_motion_fit": motion_fit,
                        "stationary_or_blocked": stationary,
                    },
                    "decision_gates": gates,
                    "structured_motion_summary": structured_motion,
                    "structured_semantic_summary": structured_semantics,
                    "structured_visual_summary": structured_visual,
                    "vertical_endpoint_evidence": vertical_evidence,
                    "orientation_heading_override": orientation_override,
                    "orientation_evidence": orientation_evidence,
                    "pass_directional_override": pass_directional_override,
                    "pass_directional_evidence": pass_directional_evidence,
                    "circumnavigation_rear_endpoint_override": (
                        circumnavigation_rear_endpoint_override),
                    "circumnavigation_rear_endpoint_evidence": (
                        circumnavigation_rear_endpoint_evidence),
                    "terminal_extent_rgb_rear_clear_override": (
                        terminal_extent_rgb_override),
                    "terminal_extent_rgb_completion_override": (
                        terminal_extent_rgb_completion_override),
                    "terminal_extent_cumulative_stage_travel_m": (
                        cumulative_stage_travel_m),
                    "exit_front_rear_ambiguity_veto": (
                        exit_front_rear_ambiguity_veto),
                    "exit_rear_dominance_override": (
                        exit_rear_dominance_override),
                    "exit_temporal_crossing_override": (
                        exit_temporal_crossing_override),
                    "endpoint_recovery": endpoint_recovery,
                    "endpoint_recovery_evidence": endpoint_recovery_evidence,
                }
            if strict_binary_completion:
                model_status = str(result["status"]).strip().lower()
                if model_status not in {"completed", "unknown"}:
                    raise ValueError("status must be completed or unknown")
                confidence = float(result["confidence"])
                if not 0.0 <= confidence <= 1.0:
                    raise ValueError("confidence must be in [0,1]")
                evidence = result["completion_evidence"]
                boundary = bool(evidence["completion_boundary_observed"])
                temporal_change = bool(
                    evidence["temporal_relation_change_observed"])
                contradiction = bool(evidence["contradiction_observed"])
                derived_status = (
                    "completed" if boundary and temporal_change and
                    not contradiction else "unknown")
                normalized = derived_status != model_status
                return {
                    "status": derived_status,
                    "model_status": model_status,
                    "status_normalized": normalized,
                    "confidence": min(confidence, 0.65) if normalized else confidence,
                    "reason": str(result.get("reason", "")),
                    "visual_evidence": str(result.get("visual_evidence", "")),
                    "completion_evidence": {
                        "completion_boundary_observed": boundary,
                        "temporal_relation_change_observed": temporal_change,
                        "contradiction_observed": contradiction,
                    },
                }
            if bidirectional_progress:
                observed_status = str(result["observed_status"]).strip().lower()
                reversed_status = str(result["reversed_status"]).strip().lower()
                direction_fit = str(result["direction_fit"]).strip().lower()
                known_statuses = {"arrived", "on_route", "unknown"}
                if observed_status not in known_statuses:
                    raise ValueError("observed_status has an unknown value")
                if reversed_status not in known_statuses:
                    raise ValueError("reversed_status has an unknown value")
                if direction_fit not in {"observed", "reversed", "neither"}:
                    raise ValueError("direction_fit has an unknown value")
                confidence = float(result["confidence"])
                if not 0.0 <= confidence <= 1.0:
                    raise ValueError("confidence must be in [0,1]")
                normalized = False
                if direction_fit == "observed" and observed_status == "unknown":
                    direction_fit = "neither"
                    normalized = True
                elif direction_fit == "reversed" and reversed_status == "unknown":
                    direction_fit = "neither"
                    normalized = True
                final_status = (
                    observed_status if direction_fit == "observed" else "unknown")
                return {
                    "status": final_status,
                    "observed_status": observed_status,
                    "reversed_status": reversed_status,
                    "direction_fit": direction_fit,
                    "status_normalized": normalized,
                    "confidence": min(confidence, 0.65) if normalized else confidence,
                    "reason": str(result.get("reason", "")),
                    "visual_evidence": str(result.get("visual_evidence", "")),
                    "bidirectional_unknown_gate": direction_fit != "observed",
                }
            status = str(result["status"]).strip().lower()
            if three_way_progress:
                if status not in {"arrived", "on_route", "unknown"}:
                    raise ValueError(
                        "status must be arrived, on_route, or unknown")
                confidence = float(result["confidence"])
                if not 0.0 <= confidence <= 1.0:
                    raise ValueError("confidence must be in [0,1]")
                evidence = result["progress_evidence"]
                completion = bool(evidence["completion_boundary_observed"])
                progress = bool(evidence["instruction_consistent_progress"])
                contradiction = bool(evidence["contradiction_observed"])
                if contradiction or not progress:
                    derived_status = "unknown"
                elif completion:
                    derived_status = "arrived"
                else:
                    derived_status = "on_route"
                normalized = derived_status != status
                return {
                    "status": derived_status,
                    "model_status": status,
                    "status_normalized": normalized,
                    "confidence": min(confidence, 0.65) if normalized else confidence,
                    "reason": str(result.get("reason", "")),
                    "visual_evidence": str(result.get("visual_evidence", "")),
                    "progress_evidence": {
                        "completion_boundary_observed": completion,
                        "instruction_consistent_progress": progress,
                        "contradiction_observed": contradiction,
                    },
                }
            if status not in {"completed", "unknown"}:
                raise ValueError("status must be completed or unknown")
            confidence = float(result["confidence"])
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be in [0,1]")
            validated = {
                "status": status,
                "confidence": confidence,
                "reason": str(result.get("reason", "")),
                "visual_evidence": str(result.get("visual_evidence", "")),
            }
            if eight_view_completion:
                evidence = result["directional_evidence"]
                known_sectors = {
                    "front", "front_left", "left", "rear_left", "rear",
                    "rear_right", "right", "front_right",
                }
                previous_sectors = [
                    str(value).strip().lower()
                    for value in evidence["reference_previous_sectors"]]
                current_sectors = [
                    str(value).strip().lower()
                    for value in evidence["reference_current_sectors"]]
                unknown = ((set(previous_sectors) | set(current_sectors)) -
                           known_sectors)
                if unknown:
                    raise ValueError(
                        f"unknown eight-view direction sectors: {sorted(unknown)}")
                previous_indices = [
                    int(value) for value in
                    evidence["reference_previous_view_indices"]]
                current_indices = [
                    int(value) for value in
                    evidence["reference_current_view_indices"]]
                if any(not 0 <= value <= 7
                       for value in previous_indices + current_indices):
                    raise ValueError("eight-view indices must lie in [0,7]")
                index_to_sector = (
                    "front", "front_left", "left", "rear_left", "rear",
                    "rear_right", "right", "front_right")
                if set(previous_sectors) != {
                        index_to_sector[value] for value in previous_indices}:
                    raise ValueError(
                        "previous sectors must exactly match previous view indices")
                if set(current_sectors) != {
                        index_to_sector[value] for value in current_indices}:
                    raise ValueError(
                        "current sectors must exactly match current view indices")
                same_instance_confident = bool(
                    evidence["same_instance_confident"])
                multiple_similar_instances = bool(
                    evidence["multiple_similar_instances"])
                calculated_rear = bool(
                    same_instance_confident and
                    any(value in {3, 4, 5} for value in current_indices))
                if bool(evidence["rear_sector_evidence"]) != calculated_rear:
                    raise ValueError(
                        "rear_sector_evidence must agree with current view indices "
                        "and same_instance_confident")
                validated["directional_evidence"] = {
                    "reference_previous_sectors": previous_sectors,
                    "reference_current_sectors": current_sectors,
                    "reference_previous_view_indices": previous_indices,
                    "reference_current_view_indices": current_indices,
                    "same_instance_confident": same_instance_confident,
                    "multiple_similar_instances": multiple_similar_instances,
                    "rear_sector_evidence": calculated_rear,
                    "front_sector_contradiction": bool(
                        evidence["front_sector_contradiction"]),
                    "temporal_relation_change": bool(
                        evidence["temporal_relation_change"]),
                }
                identity_gated_forms = {
                    "PASS_LANDMARK", "CIRCUMNAVIGATE",
                    "APPROACH_LANDMARK", "STOP_WAIT",
                }
                if (str(item.get("form", "")).upper() in identity_gated_forms and
                        (not same_instance_confident or
                         multiple_similar_instances)):
                    validated["status"] = "unknown"
                    validated["identity_ambiguity_override"] = True
                if str(item.get("form", "")).upper() == "PASS_LANDMARK":
                    pass_relation_supported = bool(
                        calculated_rear and
                        validated["directional_evidence"]
                        ["temporal_relation_change"] and
                        not validated["directional_evidence"]
                        ["front_sector_contradiction"])
                    if (validated["status"] == "completed" and
                            not pass_relation_supported):
                        validated["status"] = "unknown"
                        validated["pass_relation_override"] = True
                    # DINO+SAM semantics use the stable six-view panorama:
                    # 0/1/5 are front/front-side and 2/3/4 are rear/rear-side.
                    # A strong closer front detection of the target class makes
                    # a VLM claim about a rear instance ambiguous (common in
                    # rooms containing more than one sofa/chair/table).
                    ignored_tokens = {
                        "the", "this", "that", "gray", "grey", "black",
                        "white", "brown", "red", "blue", "green", "small",
                        "large", "left", "right", "front", "behind",
                    }
                    landmark_tokens = [
                        token for token in re.findall(
                            r"[a-z]+", str(item.get("landmark", "")).lower())
                        if len(token) >= 3 and token not in ignored_tokens]

                    def detector_score(view_indices):
                        best_score = 0.0
                        best_depth = None
                        for view in current_environment_semantics.get("views", []):
                            if int(view.get("view_index", -1)) not in view_indices:
                                continue
                            for detection in view.get("detections", []):
                                label = str(detection.get("label", "")).lower()
                                if (landmark_tokens and not any(
                                        token in label for token in landmark_tokens)):
                                    continue
                                score = float(detection.get("score", 0.0))
                                if score > best_score:
                                    best_score = score
                                    depth = detection.get("median_depth_m")
                                    best_depth = (float(depth)
                                                  if depth is not None else None)
                        return best_score, best_depth

                    front_score, front_depth = detector_score({0, 1, 5})
                    rear_score, rear_depth = detector_score({2, 3, 4})
                    detector_guard = {
                        "source": "dino_sam_six_view_semantics",
                        "landmark_tokens": landmark_tokens,
                        "front_score": front_score,
                        "rear_score": rear_score,
                        "front_depth_m": front_depth,
                        "rear_depth_m": rear_depth,
                    }
                    front_dominates = bool(
                        landmark_tokens and front_score >= 0.5 and
                        front_score >= rear_score + 0.12)
                    detector_guard["front_dominates"] = front_dominates
                    validated["detector_directional_guard"] = detector_guard
                    if validated["status"] == "completed" and front_dominates:
                        validated["status"] = "unknown"
                        validated["detector_directional_override"] = True
                        validated["reason"] += (
                            " DINO+SAM still finds stronger target-class "
                            "evidence in front than rear, so same-instance pass "
                            "completion is ambiguous.")
            if structured_partial_extent:
                evidence = result["partial_extent_evidence"]
                flags = {
                    key: bool(evidence[key]) for key in (
                        "starts_at_stair_base", "continuous_ascent",
                        "current_steps_below", "current_steps_above")
                }
                validated["partial_extent_evidence"] = flags
                validated["status"] = (
                    "completed" if all(flags.values()) else "unknown")
                validated["operational_status_override"] = True
                def stair_view_count(semantics):
                    return sum(any(
                        "stair" in str(detection.get("label", "")).lower()
                        for detection in view.get("detections", []))
                        for view in semantics.get("views", []))
                previous_stair_views = stair_view_count(
                    previous_environment_semantics)
                current_stair_views = stair_view_count(
                    current_environment_semantics)
                direction_satisfied = (
                    vertical_delta_m >= 0.5
                    if str(item.get("form", "")).upper() == "VERTICAL_UP"
                    else vertical_delta_m <= -0.5)
                node_edge_evidence = {
                    "vertical_direction_and_delta_satisfied": bool(
                        direction_satisfied),
                    "vertical_delta_m": vertical_delta_m,
                    "previous_stair_view_count": previous_stair_views,
                    "current_stair_view_count": current_stair_views,
                    "traveled_distance_m": action_summary["traveled_distance_m"],
                }
                node_edge_operational_complete = bool(
                    direction_satisfied and previous_stair_views >= 1 and
                    current_stair_views >= 2 and
                    float(action_summary["traveled_distance_m"]) >= 0.8)
                validated["node_edge_partial_extent_evidence"] = (
                    node_edge_evidence)
                if node_edge_operational_complete:
                    validated["status"] = "completed"
                    validated["node_edge_operational_override"] = True
            return validated

        panorama_sheet = (self._completion_contact_sheet
                          if eight_view_completion else self._contact_sheet)
        current_storyboard = (
            self._keyframe_storyboard(edge_keyframes)
            if structured_binary_completion else
            self._keyframe_strip(edge_keyframes))
        if visual_carryover:
            prior_storyboard = (
                self._keyframe_storyboard(carryover_edge_keyframes)
                if structured_binary_completion else
                self._keyframe_strip(carryover_edge_keyframes))
            images = [
                panorama_sheet(carryover_previous_six_views),
                prior_storyboard,
                panorama_sheet(previous_six_views),
                current_storyboard,
                panorama_sheet(current_six_views),
            ]
        else:
            images = [
                panorama_sheet(previous_six_views),
                current_storyboard,
                panorama_sheet(current_six_views),
            ]
        if structured_binary_completion:
            schema = (self.STRUCTURED_VERTICAL_BINARY_COMPLETION_SCHEMA
                      if vertical_guard_active else
                      self.STRUCTURED_BINARY_COMPLETION_SCHEMA)
        elif strict_binary_completion:
            schema = self.BINARY_PROGRESS_SCHEMA
        elif bidirectional_progress:
            images.append(self._keyframe_strip(list(reversed(edge_keyframes))))
            schema = self.BIDIRECTIONAL_PROGRESS_SCHEMA
        elif three_way_progress:
            schema = self.THREE_WAY_PROGRESS_SCHEMA
        elif eight_view_completion:
            schema = (self.EIGHT_VIEW_PARTIAL_EXTENT_COMPLETION_SCHEMA
                      if structured_partial_extent
                      else self.EIGHT_VIEW_COMPLETION_SCHEMA)
        else:
            schema = (self.PARTIAL_EXTENT_COMPLETION_SCHEMA
                      if structured_partial_extent
                      else self.INSTRUCTION_COMPLETION_SCHEMA)
        primary = self._call(
            "judge_edge_instruction_completion", prompt, images,
            schema, validate)
        if primary_consensus_completion:
            second_primary = self._call(
                "judge_edge_instruction_completion_confirmation",
                prompt, images, schema, validate)
            primary_candidates = [primary, second_primary]
            if primary["status"] != second_primary["status"]:
                primary_candidates.append(self._call(
                    "judge_edge_instruction_completion_tiebreak",
                    prompt, images, schema, validate))
            completed_votes = sum(
                item["status"] == "completed"
                for item in primary_candidates)
            consensus_status = (
                "completed" if completed_votes >
                len(primary_candidates) / 2 else "unknown")
            matching = [
                item for item in primary_candidates
                if item["status"] == consensus_status]
            primary = dict(max(
                matching, key=lambda item: float(item.get(
                    "confidence", 0.0))))
            primary["primary_completion_consensus"] = {
                "policy": (
                    "two independent structured calls; a third is used only "
                    "to break a status disagreement"),
                "call_count": len(primary_candidates),
                "statuses": [
                    item["status"] for item in primary_candidates],
                "completed_votes": completed_votes,
                "unknown_votes": len(primary_candidates) - completed_votes,
                "consensus_status": consensus_status,
                "results": primary_candidates,
            }
        # A generic portal failure mode is a near-threshold node that still
        # occupies the source room.  The chronology-rich completion calls can
        # over-read forward motion and describe the portal as behind even when
        # the raw endpoint panorama still shows the source room around the
        # camera.  Before accepting any portal-crossing form, run two
        # independent endpoint-side audits that see only the before/after RGB
        # panoramas and instruction.  All four forms below have a crossed-
        # portal arrival definition in instruction_taxonomy.py; this gate is
        # therefore form-generic rather than an episode-specific heuristic.
        # It has no action history, detector labels, depth, navmesh, GT path,
        # episode identity, or prior completion result.
        portal_crossing_forms = {
            "EXIT_REGION", "ENTER_REGION", "SELECT_PORTAL",
            "TRAVERSE_PORTAL_REGION",
        }
        portal_form = str(item.get("form", "")).upper()
        if (portal_form in portal_crossing_forms and
                not compound_terminal_extent_portal_stage(item) and
                self.instruction_completion_prompt_version ==
                "v24_multireference_threshold_calibration"):
            exit_side_schema = {
                "type": "object", "additionalProperties": False,
                "required": ["endpoint_side", "confidence", "reason"],
                "properties": {
                    "endpoint_side": {
                        "type": "string",
                        "enum": ["destination_side", "source_or_threshold"],
                    },
                    "confidence": {"type": "number", "minimum": 0,
                                   "maximum": 1},
                    "reason": {"type": "string", "maxLength": 420},
                },
            }
            audit_prompt_name = (
                "EXIT_ENDPOINT_SIDE_AUDIT" if portal_form == "EXIT_REGION"
                else "PORTAL_ENDPOINT_SIDE_AUDIT")
            exit_side_prompt = f"""{audit_prompt_name}
Decide whether the CURRENT camera center has actually crossed the instructed
portal.  Image 0 is the source node panorama and Image 1 is the current
node panorama.  Do not infer crossing from apparent forward motion or a door
being visible.  Return destination_side only when the current camera is in the
space beyond the named portal: the source room/threshold is behind or confined
to rear-side views and the destination space surrounds the current camera.
If the current panorama is still dominated by the source room, the door frame
is beside/ahead of the camera, or the camera is merely at the threshold,
return source_or_threshold.  Overlapping panorama views can show a crossed
door at adjacent bearings, so judge physical room occupancy rather than
counting sectors.  Use raw RGB only; no depth, detector text, action history,
navmesh, demonstration path, goal geometry, prior model decision, or episode
metadata is supplied.
Instruction: {item.get('navigation_instruction', '')}
Semantic target: {item.get('semantic_spatial_target', '')}
Completion cue: {item.get('completion_cue', '')}
Return JSON only."""

            def validate_exit_side(result):
                endpoint_side = str(result["endpoint_side"]).strip().lower()
                if endpoint_side not in {
                        "destination_side", "source_or_threshold"}:
                    raise ValueError("invalid portal endpoint side")
                confidence = float(result["confidence"])
                if not 0.0 <= confidence <= 1.0:
                    raise ValueError("confidence must be in [0,1]")
                return {
                    "endpoint_side": endpoint_side,
                    "confidence": confidence,
                    "reason": str(result.get("reason", "")),
                }

            exit_side_first = self._call(
                "audit_exit_endpoint_side", exit_side_prompt,
                [panorama_sheet(previous_six_views),
                 panorama_sheet(current_six_views)],
                exit_side_schema, validate_exit_side)
            exit_side_second = self._call(
                "audit_exit_endpoint_side_confirmation", exit_side_prompt,
                [panorama_sheet(previous_six_views),
                 panorama_sheet(current_six_views)],
                exit_side_schema, validate_exit_side)
            exit_side_results = [exit_side_first, exit_side_second]
            destination_unanimous = all(
                result["endpoint_side"] == "destination_side" and
                float(result["confidence"]) >= 0.70
                for result in exit_side_results)
            if destination_unanimous:
                exit_side = dict(min(
                    exit_side_results,
                    key=lambda result: float(result["confidence"])))
            else:
                source_votes = [
                    result for result in exit_side_results
                    if result["endpoint_side"] == "source_or_threshold"]
                exit_side = dict(max(
                    source_votes or exit_side_results,
                    key=lambda result: float(result["confidence"])))
                exit_side["endpoint_side"] = "source_or_threshold"
            exit_side["endpoint_side_consensus"] = {
                "policy": (
                    "two chronology-blind raw-RGB audits must unanimously "
                    "verify destination_side at confidence >= 0.70"),
                "results": exit_side_results,
                "destination_unanimous": destination_unanimous,
            }
            exit_side["input_policy"] = (
                "two raw RGB panoramas only; chronology and prior result hidden")
            primary["exit_endpoint_side_audit"] = exit_side
            # Generic name for new consumers.  Keep the legacy EXIT field so
            # archived rounds and downstream audit readers remain compatible.
            primary["portal_endpoint_side_audit"] = exit_side
            if not destination_unanimous:
                primary["status"] = "unknown"
                primary["confidence"] = min(
                    float(primary.get("confidence", 0.0)), 0.65)
                primary["exit_endpoint_side_veto_applied"] = True
                primary["portal_endpoint_side_veto_applied"] = True
                primary["reason"] += (
                    " Independent raw-RGB endpoint-side audit did not verify "
                    "that the camera center had crossed the instructed portal.")
        if not unordered_reverse_veto or primary["status"] != "completed":
            return primary

        # The second pass sees no chronology or previous model result.  Its
        # only authority is a conservative reverse-order veto: if the source
        # endpoint independently looks like the completion target and the
        # destination does not, this edge cannot be the completion event.
        shuffle_key = "|".join((
            str(previous_node_id), str(current_node_id),
            str(item.get("navigation_instruction", ""))))
        swap = bool(hashlib.sha256(shuffle_key.encode("utf-8")).digest()[0] & 1)
        def run_endpoint_roles(call_swapped):
            if call_swapped:
                result = self.classify_unordered_instruction_endpoint_roles(
                    item, current_six_views, previous_six_views,
                    current_environment_semantics,
                    previous_environment_semantics)
                previous = result["node_y_role"]
                current = result["node_x_role"]
            else:
                result = self.classify_unordered_instruction_endpoint_roles(
                    item, previous_six_views, current_six_views,
                    previous_environment_semantics,
                    current_environment_semantics)
                previous = result["node_x_role"]
                current = result["node_y_role"]
            reverse = bool(
                float(result["confidence"]) >= 0.70 and
                previous == "completion_target" and
                current != "completion_target")
            return result, previous, current, reverse

        role_result, previous_role, current_role, first_reverse = (
            run_endpoint_roles(swap))
        second_result = None
        second_previous_role = None
        second_current_role = None
        second_reverse = None
        permutation_consistent = None
        if swap_consistent_reverse_veto:
            (second_result, second_previous_role, second_current_role,
             second_reverse) = run_endpoint_roles(not swap)
            exact_role_consistency = bool(
                previous_role == second_previous_role and
                current_role == second_current_role)
            permutation_consistent = bool(
                (previous_role == "completion_target") ==
                (second_previous_role == "completion_target") and
                (current_role == "completion_target") ==
                (second_current_role == "completion_target"))
            reverse_veto = bool(
                first_reverse and second_reverse and permutation_consistent)
        else:
            exact_role_consistency = None
            reverse_veto = first_reverse
        primary["unordered_endpoint_role_guard"] = {
            "deterministically_shuffled": True,
            "swap": swap,
            "previous_endpoint_role": previous_role,
            "current_endpoint_role": current_role,
            "confidence": float(role_result["confidence"]),
            "preferred_order_in_shuffled_coordinates": role_result[
                "preferred_order"],
            "swap_consistency_required": swap_consistent_reverse_veto,
            "opposite_swap": (not swap if swap_consistent_reverse_veto
                              else None),
            "opposite_previous_endpoint_role": second_previous_role,
            "opposite_current_endpoint_role": second_current_role,
            "opposite_confidence": (
                float(second_result["confidence"])
                if second_result is not None else None),
            "exact_role_consistency": exact_role_consistency,
            "permutation_consistent": permutation_consistent,
            "reverse_completion_veto": reverse_veto,
            "result": role_result,
            "opposite_result": second_result,
        }
        if reverse_veto:
            primary["status"] = "unknown"
            primary["confidence"] = min(float(primary["confidence"]), 0.65)
            primary["unordered_reverse_veto_applied"] = True
            primary["reason"] += (
                " Two chronology-hidden endpoint-role checks with opposite "
                "X/Y assignments consistently find the previous node, but "
                "not the current node, already matches the completion target; "
                "the observed edge is therefore conservatively treated as "
                "unknown."
                if swap_consistent_reverse_veto else
                " An independent chronology-hidden endpoint-role check finds "
                "the previous node, but not the current node, already matches "
                "the completion target; the observed edge is therefore "
                "conservatively treated as unknown.")
        return primary

    def classify_unordered_instruction_endpoint_roles(
            self, sub_instruction, node_x_views, node_y_views,
            node_x_environment_semantics, node_y_environment_semantics):
        """Classify two deliberately unordered nodes by instruction role."""
        if len(node_x_views) != 8 or len(node_y_views) != 8:
            raise ValueError("unordered endpoint roles require two eight-view nodes")
        item = (sub_instruction.to_dict() if hasattr(sub_instruction, "to_dict")
                else dict(sub_instruction))
        compact_instruction = {key: item.get(key) for key in (
            "navigation_instruction", "landmark", "form",
            "semantic_spatial_target", "spatial_relation", "completion_cue",
            "visual_arrival_evidence")}
        form = str(item.get("form", "OTHER")).upper()
        form_rule = self.EIGHT_VIEW_FORM_RULES.get(
            form, self.EIGHT_VIEW_FORM_RULES["OTHER"])
        prompt = f"""UNORDERED_ENDPOINT_INSTRUCTION_ROLE
Two real indoor nodes X and Y have been deliberately shuffled by a deterministic
hash. Their names and image order do NOT reveal chronology. Classify each node's
semantic role relative to the active instruction independently.

Allowed roles:
- source_context: visually fits the instruction's source/before-side context;
- intermediate_progress: clearly beyond the source in the requested semantic
  direction, but still before the full completion target;
- completion_target: satisfies the semantic spatial target and completion cue;
- no_clear_relation: none of the above is established or the role is ambiguous.

Do not assume X precedes Y. Do not infer a role from absolute room plausibility
alone: use the named landmark/portal/region and required spatial relation. A
completion_target must satisfy the whole instruction, not merely an early
clause. For repeated objects or indistinguishable rooms, use no_clear_relation.

Active instruction:
{json.dumps(compact_instruction, ensure_ascii=False)}

Active-form spatial rule for {form}:
{form_rule}

Node X observed semantics:
{json.dumps(node_x_environment_semantics, ensure_ascii=False)}
Node Y observed semantics:
{json.dumps(node_y_environment_semantics, ensure_ascii=False)}

Image 0 is Node X's eight-view panorama. Image 1 is Node Y's eight-view
panorama. No chronology, action history, case category, expected label,
demonstration waypoint/path index, or earlier model prediction is provided.
Return JSON only."""

        def validate(result):
            roles = {
                "source_context": 0,
                "intermediate_progress": 1,
                "completion_target": 2,
                "no_clear_relation": None,
            }
            node_x_role = str(result["node_x_role"]).strip().lower()
            node_y_role = str(result["node_y_role"]).strip().lower()
            if node_x_role not in roles or node_y_role not in roles:
                raise ValueError("endpoint role has an unknown value")
            confidence = float(result["confidence"])
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be in [0,1]")
            x_rank, y_rank = roles[node_x_role], roles[node_y_role]
            if x_rank is None or y_rank is None or x_rank == y_rank:
                preferred_order = "neither"
            elif x_rank < y_rank:
                preferred_order = "x_to_y"
            else:
                preferred_order = "y_to_x"
            return {
                "node_x_role": node_x_role,
                "node_y_role": node_y_role,
                "preferred_order": preferred_order,
                "confidence": confidence,
                "reason": str(result.get("reason", "")),
                "visual_evidence": str(result.get("visual_evidence", "")),
            }

        return self._call(
            "classify_unordered_instruction_endpoint_roles", prompt,
            [self._completion_contact_sheet(node_x_views),
             self._completion_contact_sheet(node_y_views)],
            self.UNORDERED_ENDPOINT_ROLE_SCHEMA, validate)

    def select_step_target(self, stage, rgb, target_mask, action_history):
        """Select a fresh floor waypoint and coarse action for one control step."""
        if not target_mask.any():
            raise RuntimeError("Current observation contains no targetable floor")
        overlay = self._overlay(rgb, target_mask, 0, False)
        history_text = ", ".join(action_history[-6:]) if action_history else "none"
        prompt = f"""STEP_GROUND_TARGET_SELECTION
You control one step of an indoor R2R agent. Current stage:
{stage['navigation_instruction']}
Landmark: {stage['landmark']}. Completion cue: {stage['completion_cue']}.
Recent actions (oldest to newest): {history_text}.
Green pixels are valid floor in the CURRENT camera view. Choose the floor point
that best advances the stage; the geometric controller derives steering from
the point, so do not emit a discrete action. Always choose a waypoint ON green floor. x_norm and
y_norm are coordinates within this single image from top-left. Avoid image
edges and the near-camera bottom. Return only requested JSON."""

        def validate(result):
            h, w = target_mask.shape
            requested = np.array([
                float(result["x_norm"]) * (w - 1),
                float(result["y_norm"]) * (h - 1),
            ], np.float32)
            point = self._nearest_ground(target_mask, requested)
            return point, {
                "action": "point_steer", "requested_xy": requested.tolist(),
                "snapped_xy": point.tolist(), "reason": str(result.get("reason", "")),
            }

        return self._call("select_step_target", prompt, [overlay], self.STEP_SCHEMA, validate)
