#!/usr/bin/env python3
"""VLM-per-hop instruction exploration with one-hop recovery and backtracking."""

from __future__ import annotations

import math
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from instruction_completion_judge import (
    NodeTransitionInstructionCompletionJudge, UNKNOWN,
    generic_near_stop_override_eligible, landmark_detection_aliases,
)
from node_backtracking import NodeBacktrackingController
from path_projection import draw_reference_path_overlay, project_reference_path


PROGRESS_REQUIRED_FORMS = frozenset({
    "TURN_LEFT", "TURN_RIGHT", "TURN_AROUND",
    "VERTICAL_UP", "VERTICAL_DOWN", "ADVANCE_STRAIGHT",
    "PASS_LANDMARK", "EXIT_REGION", "ENTER_REGION",
    "TRAVERSE_PORTAL_REGION", "SELECT_PORTAL", "BETWEEN_OBJECTS",
    "FOLLOW_PATH_BOUNDARY", "CIRCUMNAVIGATE",
})
# A pure U-turn establishes the reverse route at its first local floor node.
# Three metres can consume an immediately following doorway/region clause
# before that clause gets its own edge and node.  Two and a half metres leaves
# enough visible floor beyond the horizontal-camera blind strip while
# preserving the next semantic boundary.
TURN_AROUND_LOCAL_WAYPOINT_MAX_M = 2.50
BARE_SIDE_TURN_ROUTE_SETUP_MAX_M = 4.00
FORWARD_CONTINUATION_WAYPOINT_MAX_M = 6.00
# PASS is anchored by a named landmark but may require a short detour around
# furniture occupying the direct image ray.  Keep this separate from generic
# straight/follow forms: the VLM decision remains frozen, while the physical
# post-selection layer may accept at most two extra metres of navmesh detour.
# This stays below the form's 40 x 0.22 m execution envelope.
PASS_LANDMARK_WAYPOINT_MAX_M = 8.00
COMPOUND_DIRECTIONAL_ROUTE_SETUP_MAX_M = 3.00
# A 1.5 m cap can reject every floor-projected ray when the named object is
# one oblique camera sector away, even though the visible near-side patch is
# still local. Two metres remains a bounded object approach and leaves the
# completion judge responsible for the exact relation before STOP.
LOCAL_TERMINAL_RELATION_WAYPOINT_MAX_M = 2.00
# The project acceptance gate is 30 degrees against the demonstrated route.
# A continuation cone is relative to the preceding online-selected ray, whose
# own residual error is unknown at runtime.  Reserving half of that budget for
# the second edge prevents ordinary per-hop errors from adding to >30 degrees.
LINEAR_SAME_STAGE_CONTINUATION_HALF_WIDTH_DEG = 15.0
# A portal endpoint veto may mean the first floor point stopped at the
# threshold.  Preserve/deepen the same online-supported ray twice; beyond
# that, identical UNKNOWN evidence is no longer new route information and
# must enter ordinary branch blocking instead of creating an unbounded
# backtrack loop.
MAX_PORTAL_THRESHOLD_ROUTE_RETRIES = 2


_BARE_TURN_PATTERNS = {
    "TURN_LEFT": (
        r"(?:then )?(?:turn|face)(?: to your)? left(?: (?:90|ninety)(?: degrees?)?)?",
        r"(?:then )?make(?: a)? left turn",
        r"(?:then )?take(?: a)?(?: hard)? left(?: turn)?",
    ),
    "TURN_RIGHT": (
        r"(?:then )?(?:turn|face)(?: to your)? right(?: (?:90|ninety)(?: degrees?)?)?",
        r"(?:then )?make(?: a)? right turn",
        r"(?:then )?take(?: a)?(?: hard)? right(?: turn)?",
    ),
    "TURN_AROUND": (
        r"(?:then )?turn(?: all the way)? around(?: (?:180|one hundred eighty)(?: degrees?)?)?",
        r"(?:then )?make(?: a)? u turn",
        r"(?:then )?reverse direction",
    ),
}
_BARE_TURN_DELTAS = {
    "TURN_LEFT": math.pi / 2.0,
    "TURN_RIGHT": -math.pi / 2.0,
    "TURN_AROUND": math.pi,
}


def form_requires_physical_progress(form):
    """Whether an instruction relation needs a distinct arrival node."""
    return str(form or "").upper() in PROGRESS_REQUIRED_FORMS


def fixed_recovery_corridor_eligible(form):
    """Whether partial recovery may preserve one absolute horizontal ray.

    Only explicitly linear route relations retain a fixed compass bearing.
    Boundary following, circumnavigation, region traversal and compound turn
    clauses can legitimately bend or hairpin while the same instruction is
    active; freezing their first edge's absolute yaw converts a correct first
    hop into a wrong second hop. Vertical routes similarly use visible treads
    rather than a previous flight bearing.
    """
    return str(form or "").upper() in {
        "ADVANCE_STRAIGHT", "PASS_LANDMARK", "CROSS_SPACE",
        "BETWEEN_OBJECTS",
    }


def portal_recovery_hemisphere_eligible(form):
    """Keep a supported portal retry from traversing the portal backwards.

    Portal paths may bend, so they must not use the narrow fixed corridor used
    by linear clauses.  After an on-route edge and its one permitted lookahead
    are both endpoint-vetoed, however, recovery returns to the partial node.
    At that point a new 360-degree search can select the same doorway in the
    reverse direction and falsely complete by crossing back into the source
    room.  Preserve only the online supported ray's forward hemisphere; RGB
    still chooses freely within it and no demonstration geometry is used.
    """
    return str(form or "").upper() in {
        "EXIT_REGION", "ENTER_REGION", "SELECT_PORTAL",
        "TRAVERSE_PORTAL_REGION",
    }


def portal_recovery_corridor_active(form, retry_after_backtrack):
    """Apply the same-portal bearing only after a physical backtrack.

    The first endpoint-vetoed portal edge is followed from its newly reached
    node and may legitimately bend immediately beyond the threshold.  It
    already carries raw-RGB portal identity, so constraining that ordinary
    lookahead to an absolute hemisphere can remove the only visible floor
    continuation.  The absolute bearing is needed only after recovery has
    physically returned to the partial node, where an unrestricted panorama
    could otherwise traverse the same doorway backwards.
    """
    return bool(
        retry_after_backtrack and
        portal_recovery_hemisphere_eligible(form))


def exit_threshold_portal_continuation_required(stage, progress_context):
    """Continue an endpoint-vetoed portal form through the same portal.

    Region traversal may generally bend, so portal forms are deliberately
    absent from ``fixed_recovery_corridor_eligible``.  The narrower case here
    is different:
    the primary RGB transition judge saw the instructed crossing while the
    independent endpoint-side audit vetoed completion.  The initial relative
    direction phrase has therefore served its purpose (portal identity), but
    the route may legitimately bend at the doorway.  Subsequent selection
    must continue through that portal without reapplying ``left``/``right``
    from the controller's new camera heading.
    """
    return bool(
        str((stage or {}).get("form", "")).upper() in {
            "EXIT_REGION", "ENTER_REGION", "SELECT_PORTAL",
            "TRAVERSE_PORTAL_REGION",
        } and
        isinstance(progress_context, dict) and
        progress_context.get("active") and
        progress_context.get("prior_exit_endpoint_side_veto_applied") and
        progress_context.get("prior_selected_yaw_rad") is not None)


def compound_turn_continuation_required(stage, progress_context):
    """Whether a supported partial turn edge consumed its direction phrase.

    A clause such as ``turn right and walk across the room to the doorway``
    has two pieces: establish the right-hand route, then continue to a named
    endpoint. Once a real edge executes that turn and the node judge reports
    supported partial progress, applying ``turn right`` again leaves the
    established route. Bare turns remain excluded because their endpoint is
    the new orientation itself.
    """
    form = str((stage or {}).get("form", "")).upper()
    if form not in {"TURN_LEFT", "TURN_RIGHT", "TURN_AROUND"}:
        return False
    if not isinstance(progress_context, dict) or not progress_context.get(
            "active"):
        return False
    if bare_turn_delta_rad(stage) is not None:
        return False
    endpoint = progress_context.get("prior_endpoint_evidence", {}) or {}
    motion = progress_context.get("prior_motion_evidence", {}) or {}
    return bool(
        endpoint.get("current_target_state") == "partial" and
        motion.get("instruction_motion_fit") == "supports" and
        not motion.get("stationary_or_blocked") and
        progress_context.get("prior_selected_yaw_rad") is not None)


def local_landmark_stop_relation(stage):
    """Whether STOP_WAIT names a local object-relative stopping pose.

    Object relations such as ``in front of the toilet`` and ``at the corner
    of the bar`` need short near-side hops; ``inside the workout room`` is a
    region-entry target and may legitimately require a longer point edge.
    This classifier uses only the decomposed instruction text.
    """
    if str(stage.get("form", "")).upper() != "STOP_WAIT":
        return False
    landmark = str(stage.get("landmark", "")).lower()
    if re.search(
            r"\b(?:room|hall(?:way)?|corridor|area|region|lobby|foyer|"
            r"kitchen|bathroom|bedroom|office|closet|garage)\b", landmark):
        return False
    relation_text = " ".join(str(stage.get(key, "")) for key in (
        "navigation_instruction", "spatial_relation",
        "semantic_spatial_target", "completion_cue")).lower()
    return bool(re.search(
        r"\b(?:in front of|next to|beside|by|near|at|corner of|end of)\b",
        relation_text))


def selection_geodesic_cap_m(form, *, turn_carryover=False,
                             retry_after_unknown=False, bare_turn=False):
    """Return the online execution cap for an RGB-selected ground ray.

    The cap is evaluated only after the VLM freezes its view and pixel.  It is
    therefore an execution-safety boundary, not semantic or demonstration
    evidence.  Pure U-turns need a larger local rear window because a
    horizontal camera cannot see the floor immediately under the agent;
    forward continuation forms can use the distance already covered by their
    32--40 step execution budgets.
    """
    form = str(form or "").upper()
    # A direction-only turn is an orientation/branch-entry boundary, not a
    # room traversal.  Keep its physical point local even on an UNKNOWN retry;
    # otherwise the selector can translate 5--8m into a side room and only
    # then rotate the camera to the requested heading.  Compound clauses such
    # as "turn right and cross the room" retain their relation-scale budget.
    if bare_turn and form in {"TURN_LEFT", "TURN_RIGHT"}:
        # The first visible connected floor after an exact side turn can lie
        # just beyond three metres at a bedroom/doorway junction.  Four metres
        # remains a local branch-entry node while avoiding a false no-anchor
        # termination; U-turns keep a tighter 2.5-metre rear window so they do
        # not consume an immediately following portal clause.
        return BARE_SIDE_TURN_ROUTE_SETUP_MAX_M
    if bare_turn and form == "TURN_AROUND":
        return TURN_AROUND_LOCAL_WAYPOINT_MAX_M
    if form == "EXIT_REGION":
        # EXIT is a boundary crossing, not a hallway traversal.  Stop at the
        # first local node beyond the portal so an immediately following turn
        # is still executable from the junction.  The 3 m cap is evaluated
        # only against the chosen pixel's physical path after RGB selection.
        return 3.0
    if form == "SELECT_PORTAL":
        # Portal selection ends at/just through the chosen threshold.  A
        # bounded three-metre ray prevents the selector from consuming a full
        # destination room before the node judge can verify the crossing.
        return 3.0
    if form == "STOP_WAIT":
        # A terminal relation should establish a nearby stopping node, not
        # traverse an entire room in one crop. Short bounded hops also make a
        # failed first relation judgment recoverable without overshooting the
        # Habitat goal and reversing on the next attempt.
        return 3.0
    if turn_carryover or (retry_after_unknown and form != "TURN_TO_LANDMARK"):
        return 8.0
    if form == "TURN_AROUND":
        return TURN_AROUND_LOCAL_WAYPOINT_MAX_M
    if form == "TURN_TO_LANDMARK":
        # A horizontal camera often cannot see connected floor inside 2 m;
        # 3 m remains a bounded bearing-alignment edge. The committed-view
        # post-arrival orientation prevents this translation from switching
        # to a different repeated landmark instance.
        return 3.0
    if form in {"VERTICAL_UP", "VERTICAL_DOWN"}:
        # A full stair instruction is explicitly multi-point and its visible
        # upper/lower landing can be farther than the flat-ground default.
        return 8.0
    if form == "PASS_LANDMARK":
        return PASS_LANDMARK_WAYPOINT_MAX_M
    if form in {"ADVANCE_STRAIGHT", "FOLLOW_PATH_BOUNDARY"}:
        return FORWARD_CONTINUATION_WAYPOINT_MAX_M
    return None


def local_portal_endpoint_cap_m(stage, current_cap_m=None):
    """Bound explicitly local portal endpoints without shortening corridors.

    ``TRAVERSE_PORTAL_REGION`` can legitimately span a long intermediate
    region, so its form-level budget remains broad.  When the decomposed
    spatial target explicitly says *just/immediately* inside or beyond a
    doorway, however, the node belongs at the first floor patch across that
    boundary.  A long ray would silently consume the following instruction's
    reference state.  This rule reads only the frozen instruction fields and
    is applied after the RGB/VLM decision; no demo, depth, or goal geometry is
    exposed to the selector.
    """
    stage = dict(stage or {})
    # EXIT/SELECT_PORTAL already have a form-level 3 m boundary above.
    # ENTER_REGION may name a destination whose entrance is several metres
    # from the current node, so clipping it here can eliminate every valid
    # RGB ground anchor before the agent even reaches the doorway.  The
    # additional lexical cap is only needed for otherwise-unbounded
    # TRAVERSE_PORTAL_REGION.
    if str(stage.get("form", "")).upper() != "TRAVERSE_PORTAL_REGION":
        return current_cap_m
    text = " ".join(str(stage.get(key, "")) for key in (
        "semantic_spatial_target", "completion_cue", "spatial_relation",
    )).lower()
    explicitly_local = bool(re.search(
        r"\b(?:just|immediately)\s+(?:inside|outside|beyond|past|after)\b|"
        r"\bfirst\s+(?:safe\s+)?(?:floor|ground|space|area)\s+"
        r"(?:inside|outside|beyond|past|after)\b|"
        r"\b(?:at|across)\s+the\s+threshold\b", text))
    if not explicitly_local:
        return current_cap_m
    # A doorway ray can be visually local while the connected navmesh path
    # bends around furniture/rope barriers.  Four metres keeps the first
    # post-threshold node locally observable, while avoiding the 3 m cutoff
    # that rejects every anchor in otherwise clear RGB portal views.
    return min(float(current_cap_m or 4.0), 4.0)


def directional_towards_route_setup_cap_m(stage, current_cap_m=None):
    """Keep a turn-and-move-towards clause at its route-entry boundary.

    ``turn left and walk towards X`` commits to a route; it does not authorize
    traversing all the way to X or consuming a following STOP/PASS clause.
    Hard endpoint wording remains governed by its semantic destination.
    """
    stage = dict(stage or {})
    if str(stage.get("form", "")).upper() not in {"TURN_LEFT", "TURN_RIGHT"}:
        return current_cap_m
    text = " ".join(str(stage.get(key, "")) for key in (
        "navigation_instruction", "source_clause")).lower()
    if not re.search(
            r"\b(?:walk|go|head|move|proceed)\s+towards?\b", text):
        return current_cap_m
    if re.search(
            r"\b(?:into|through|across|until|past|to\s+the\s+(?:door|"
            r"doorway|opening|entrance|exit|end|far side))\b", text):
        return current_cap_m
    return min(float(current_cap_m or
                     COMPOUND_DIRECTIONAL_ROUTE_SETUP_MAX_M),
               COMPOUND_DIRECTIONAL_ROUTE_SETUP_MAX_M)


def bare_turn_delta_rad(stage):
    """Return the exact orientation delta for a direction-only turn clause.

    A bare turn ends at an orientation boundary, whereas a compound clause
    such as ``turn right and walk across the room to the doorway`` ends at a
    semantic place.  Full-match patterns deliberately keep the former small
    and prevent orientation-only execution from short-circuiting the latter.
    """
    form = str((stage or {}).get("form", "")).upper()
    patterns = _BARE_TURN_PATTERNS.get(form, ())
    text = str((stage or {}).get("navigation_instruction", "")).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    if any(re.fullmatch(pattern, text) for pattern in patterns):
        return float(_BARE_TURN_DELTAS[form])
    return None


def terminal_facing_landmark(stage):
    """Return the explicitly named final-facing landmark, if present.

    ``in front of X`` is a spatial relation and does not imply camera
    heading. Only explicit face/facing/look language activates this boundary.
    """
    # Only literal user-facing instruction text may trigger an extra camera
    # rotation. Generated completion cues often paraphrase ordinary route
    # progress as "facing X and moving towards it"; treating that paraphrase
    # as a terminal-facing clause can rotate away from a correct route.
    text = " ".join(str((stage or {}).get(key, "")) for key in (
        "navigation_instruction", "source_clause")).lower()
    match = re.search(
        r"\b(?:face|facing|look(?:ing)?\s+(?:at|towards?))\s+"
        r"(?:(?:a|an|the)\s+)?([a-z][a-z0-9 -]{1,60})", text)
    if not match:
        return None
    phrase = re.split(
        r"[,.;]|\b(?:and|through|after|before|while|until|then)\b",
        match.group(1), maxsplit=1)[0].strip()
    return phrase or None


def bare_turn_preview_yaw_delta_rad(preselection_alignment,
                                    selected_yaw_rad):
    """Return the real selector preview turn omitted by the executor log.

    The eight-view selector physically rotates the Habitat agent from the
    frozen incoming-route frame to its selected side view before the point
    executor starts. For a bare turn this rotation belongs to the current
    instruction edge and must be recorded for semantic/audit chronology.
    """
    alignment = preselection_alignment or {}
    if alignment.get("status") != "deferred_until_physical_arrival":
        return None
    try:
        source_yaw = float(alignment["source_node_yaw_rad"])
        selected_yaw = float(selected_yaw_rad)
    except (KeyError, TypeError, ValueError):
        return None
    return _wrap_angle(selected_yaw - source_yaw)


def named_landmark_route_setup_stage(stage, next_sub_instruction,
                                     progress_lookahead_active=False):
    """Keep TURN_TO translation on the landmark bearing and reorient after.

    The next decomposed clause is legal online tie-break context, but its
    spatial words are normally relative to the heading *after* this turn.
    Executing that next clause from the pre-turn panorama can therefore choose
    the opposite doorway.  The current named landmark supplies the only safe
    pre-turn bearing; after the local point arrives it is reacquired again.
    """
    stage = dict(stage or {})
    if (str(stage.get("form", "")).upper() != "TURN_TO_LANDMARK" or
            next_sub_instruction is None or progress_lookahead_active):
        return stage, None
    route_setup = dict(stage)
    route_setup["metadata"] = dict(
        route_setup.get("metadata", {}) or {},
        turn_to_landmark_route_setup={
            "active": True,
            "orientation_stage": {
                key: stage.get(key) for key in (
                    "sub_instruction_id", "navigation_instruction",
                    "landmark", "semantic_spatial_target")
            },
            "policy": (
                "translate locally on the current named-landmark bearing; "
                "the following clause is advisory only until this turn is "
                "completed, then reacquire the landmark at the arrived state"),
        })
    return route_setup, stage


def named_landmark_route_setup_cap_m(base_cap_m, active):
    """Bound a TURN_TO route-setup edge even when the next form has no cap."""
    if not active:
        return base_cap_m
    if base_cap_m is None:
        return 4.00
    return min(float(base_cap_m), 4.00)


def bare_turn_route_setup_stage(stage, next_sub_instruction,
                                progress_lookahead_active=False):
    """Use the next stated route for the translation part of a bare turn.

    A horizontal camera can have no reachable floor at exactly +/-90 degrees
    even though the corridor entered *after* the turn is plainly visible.
    Every stage still needs a real translated edge, so select a local point on
    the immediately following user-provided route and then apply the exact
    turn orientation at the arrived state.  The original stage remains the
    completion target; this never consumes the following sub-instruction.
    """
    stage = dict(stage or {})
    if (bare_turn_delta_rad(stage) is None or next_sub_instruction is None or
            progress_lookahead_active):
        return stage, False
    route_setup = next_sub_instruction.to_stage_dict()
    route_setup["metadata"] = dict(
        route_setup.get("metadata", {}) or {},
        bare_turn_route_setup={
            "active": True,
            "orientation_stage": {
                key: stage.get(key) for key in (
                    "sub_instruction_id", "navigation_instruction", "form")
            },
            "policy": (
                "translate on the immediately following stated route, then "
                "align exactly to the current bare-turn orientation"),
        })
    return route_setup, True


def _wrap_angle(value):
    return (value + math.pi) % (2 * math.pi) - math.pi


def incoming_route_yaw_rad(previous_position_xyz, current_position_xyz,
                           minimum_displacement_m=0.35):
    """Infer the arrival-facing route frame from one real graph displacement.

    Habitat camera forward projects to ``[-sin(yaw), -cos(yaw)]`` in XZ.  A
    point tracker may stop with the camera rotated away from its translation
    direction, so explicit left/right language must be framed by this stored
    incoming edge rather than the incidental terminal camera yaw.
    """
    previous = np.asarray(previous_position_xyz, np.float64)
    current = np.asarray(current_position_xyz, np.float64)
    if previous.shape != (3,) or current.shape != (3,):
        raise ValueError("incoming route frame requires two XYZ positions")
    delta = current - previous
    if float(np.linalg.norm(delta[[0, 2]])) < float(minimum_displacement_m):
        return None
    return _wrap_angle(math.atan2(-float(delta[0]), -float(delta[2])))


def selected_point_ray_yaw_rad(candidate):
    """Absolute horizontal ray of the selected pixel, not the view centre."""
    mask = np.asarray(candidate["target_mask"])
    width = int(mask.shape[1])
    center_x = (width - 1.0) / 2.0
    half_width = max(width / 2.0, 1.0)
    point_x = float(np.asarray(candidate["point"])[0])
    pixel_bearing = math.atan((point_x - center_x) / half_width)
    return _wrap_angle(float(candidate["yaw"]) - pixel_bearing)


def selection_contact_sheet(images):
    """Render legacy six-view or current eight/refined-view selections."""
    if len(images) == 6:
        columns = 3
    elif len(images) in {8, 9}:
        columns = 4
    else:
        raise ValueError(
            "selection visualization expects six, eight, or nine views")
    height, width = np.asarray(images[0]).shape[:2]
    rows = int(math.ceil(len(images) / columns))
    padded = [np.asarray(image, np.uint8) for image in images]
    padded.extend(
        np.zeros((height, width, 3), np.uint8)
        for _ in range(rows * columns - len(padded)))
    return np.concatenate([
        np.concatenate(
            padded[row * columns:(row + 1) * columns], axis=1)
        for row in range(rows)
    ], axis=0)


def infer_unknown_disposition(completion, action_history):
    """Classify completion-negative evidence for outer recovery policy.

    The completion judge remains binary (completed/unknown).  This helper is
    intentionally conservative and consumes only the judge's structured node/
    edge evidence plus the observed action history. ``on_route`` means the
    current endpoint is explicitly partial/unfinished and the edge contains
    real translation without a reversal or stall.  Weak evidence is allowed
    here because gradual, visually repetitive instructions can legitimately
    need several point-navigation hops.  The state machine applies a bounded
    consecutive-continuation budget, so this helper does not need to turn
    missing keyframes or ambiguous instance identity into an immediate error.
    """
    if str(completion.get("status", "")).lower() != UNKNOWN:
        return None
    endpoint = completion.get("endpoint_evidence") or {}
    temporal = completion.get("temporal_evidence") or {}
    motion = completion.get("motion_evidence") or {}
    current_state = str(endpoint.get("current_target_state", "")).lower()
    cue_state = str(endpoint.get("current_completion_cue", "")).lower()
    reversed_transition = bool(temporal.get("reverse_transition_observed"))
    stationary = bool(motion.get("stationary_or_blocked"))
    moved = sum(float(action.get("moved_m", 0.0))
                for action in (action_history or []))
    supported_partial = (
        current_state in {"partial", "unsatisfied"} and
        cue_state in {"partial", "unsatisfied"} and
        not reversed_transition and not stationary and moved >= 0.5)
    # BETWEEN and far-side CIRCUMNAVIGATE endpoints can look semantically
    # complete before their minimum coherent route span is accumulated. The
    # V24 integrity gates deliberately convert those shallow claims to
    # UNKNOWN. Let the outer policy spend its existing single continuation on
    # a second coherent edge; its accumulated distance and bend are checked by
    # the completion judge before it may claim success.
    gates = completion.get("decision_gates") or {}
    integrity_limited_partial = bool(
        (gates.get("between_stage_route_integrity") is False or
         gates.get("circumnavigation_stage_route_integrity") is False) and
        str(completion.get("model_status", "")).lower() == "completed" and
        not reversed_transition and not stationary and moved >= 0.5)
    # EXIT has a second, chronology-blind RGB endpoint-side audit.  When the
    # primary transition judge sees an instructed portal crossing but that
    # independent audit can only place the endpoint on the source/threshold
    # side, completion must remain UNKNOWN.  This is nevertheless useful
    # route evidence: spend the existing bounded continuation on the same
    # frozen RGB-selected ray instead of reopening a fresh 360-degree branch.
    # The audit is not weakened here and the next real node is judged anew.
    exit_audit = completion.get("exit_endpoint_side_audit") or {}
    exit_endpoint_limited_partial = bool(
        (completion.get("validation_overrides") or {}).get(
            "exit_endpoint_side_veto_applied") and
        str(exit_audit.get("endpoint_side", "")).lower() in {
            "source_or_threshold", "destination_side"} and
        str(completion.get("model_status", "")).lower() == "completed" and
        not reversed_transition and not stationary and moved >= 0.5)
    local_occlusion_boundary_partial = bool(
        (completion.get("validation_overrides") or {}).get(
            "local_occlusion_boundary_intermediate_veto") and
        not reversed_transition and not stationary and moved >= 0.5)
    return "on_route" if (
        supported_partial or integrity_limited_partial or
        exit_endpoint_limited_partial or
        local_occlusion_boundary_partial) else "wrong"


def chained_stop_wait_eligible(directive, next_sub_instruction,
                               is_final_sub_instruction=False):
    """Whether the just-arrived edge may also close the next pure STOP."""
    del is_final_sub_instruction
    return bool(
        directive is not None and
        directive.action == "advance_sequence" and
        next_sub_instruction is not None and
        str(next_sub_instruction.form).upper() == "STOP_WAIT")


def chained_near_stop_evidence_supported(
        sub_instruction, completion, previous_semantics, current_semantics,
        action_history):
    """Resolve a conservative near-STOP on the edge that reached its landmark.

    A preceding pass/approach edge may already finish an immediately following
    ``stop near X`` clause.  Requiring a second point then walks away from X.
    This witness is limited to unqualified near/beside relations (never
    corner/end/front/entrance), an arrived translated edge with no reversal or
    stall, and a sufficiently large same-class landmark detection corroborated
    by either clean-RGB bearing evidence or detector area growth. A chained
    STOP is a state at the endpoint, so the VLM's ``no_change`` label is not a
    rejection: the preceding clause owns the motion that established it. It
    uses neither the demonstration trajectory nor goal/depth geometry.
    """
    if str(getattr(sub_instruction, "form", "")).upper() != "STOP_WAIT":
        return False
    if not generic_near_stop_override_eligible(
            getattr(sub_instruction, "navigation_instruction", ""),
            getattr(sub_instruction, "semantic_spatial_target", ""),
            getattr(sub_instruction, "spatial_relation", "")):
        return False
    if str((completion or {}).get("status", "")).lower() != UNKNOWN:
        return False
    endpoint = (completion or {}).get("endpoint_evidence", {}) or {}
    temporal = (completion or {}).get("temporal_evidence", {}) or {}
    motion = (completion or {}).get("motion_evidence", {}) or {}
    if (str(endpoint.get("current_target_state", "")).lower() not in {
            "partial", "satisfied"} or
            str(endpoint.get("current_completion_cue", "")).lower() not in {
                "partial", "satisfied"} or
            bool(temporal.get("reverse_transition_observed")) or
            str(motion.get("instruction_motion_fit", "")).lower() not in {
                "supports", "neutral"} or
            bool(motion.get("stationary_or_blocked"))):
        return False
    moved = sum(float(item.get("moved_m", 0.0) or 0.0)
                for item in (action_history or []))
    if moved < 0.50:
        return False
    aliases = landmark_detection_aliases(
        getattr(sub_instruction, "landmark", ""))
    if not aliases:
        return False

    def max_area(semantics):
        best = 0.0
        for view in (semantics or {}).get("views", []):
            for detection in view.get("detections", []):
                label = str(detection.get("label", "")).lower()
                if any(alias in label for alias in aliases):
                    best = max(best, float(detection.get(
                        "mask_area_fraction",
                        detection.get("area_fraction", 0.0)) or 0.0))
        return best

    previous_area = max_area(previous_semantics)
    current_area = max_area(current_semantics)
    rgb_reference_visible = bool(endpoint.get("reference_current_sectors"))
    return bool(
        current_area >= 0.06 and
        (previous_area <= 0.0 or
         current_area / max(previous_area, 1e-4) >= 1.20 or
         rgb_reference_visible))


def inherits_incoming_route_corridor(stage):
    """Whether a clause explicitly continues the incoming route.

    An ordinal such as ``next doorway`` identifies a portal but says nothing
    about its bearing. Indoor routes often turn sharply at the end of a hall,
    so treating every unqualified portal as straight silently removes the
    instructed candidate from the panorama. Only lexical continuation cues
    commit the incoming route; otherwise RGB identity and route context choose
    the direction.
    """
    form = str((stage or {}).get("form", "")).upper()
    text = " ".join(str((stage or {}).get(key, "")) for key in (
        "navigation_instruction", "spatial_relation", "source_clause")).lower()
    has_explicit_turn = bool(re.search(
        r"\b(?:turn|left|right|behind|back|rear|around|u[- ]?turn)\b", text))
    has_continuation = bool(re.search(
        r"\b(?:continue|forward|straight|ahead|until|same corridor|same hallway)\b",
        text))
    if form in {
            "EXIT_REGION", "ENTER_REGION", "SELECT_PORTAL",
            "TRAVERSE_PORTAL_REGION"}:
        return bool(has_continuation and not has_explicit_turn)
    # Linear movement clauses inherit only when the user explicitly says the
    # motion remains straight/forward.  PASS by itself may require going
    # around an obstacle and FOLLOW may bend with a boundary, so neither is
    # frozen without this lexical commitment.
    if form in {
            "ADVANCE_STRAIGHT", "PASS_LANDMARK", "CROSS_SPACE",
            "FOLLOW_PATH_BOUNDARY", "BETWEEN_OBJECTS"}:
        explicit_linear = bool(re.search(
            r"\b(?:straight|forward|ahead|keep walking|continue walking|"
            r"same corridor|same hallway)\b", text))
        return bool(explicit_linear and not has_explicit_turn)
    return False


def rejected_ray_retry_reopens_incoming(stage):
    """Whether a safety retry may deliberately reconsider the incoming ray.

    Rejecting a through-wall/around-wall RGB endpoint is negative evidence
    about that selected ray only.  It must not erase action-history protection
    and turn an ordinary semantic retry into a return through the portal just
    traversed.  Only an explicit TURN_AROUND instruction authorizes that
    reversal; other forms keep the normal incoming-sector exclusion while the
    rejected direction itself is additionally blocked.
    """
    return str((stage or {}).get("form", "")).upper() == "TURN_AROUND"


def compound_terminal_extent_route(stage):
    """Whether one stage continues along a finite landmark to its endpoint."""
    form = str((stage or {}).get("form", "")).upper()
    if form not in {"ENTER_REGION", "FOLLOW_PATH_BOUNDARY"}:
        return False
    text = " ".join(str((stage or {}).get(key, "")) for key in (
        "navigation_instruction", "semantic_spatial_target", "spatial_relation",
        "completion_cue", "visual_arrival_evidence")).lower()
    return bool(
        re.search(r"\b(?:along|follow|length)\b", text) and
        re.search(r"\b(?:far end|the end|until|entire length|passed? the length|"
                  r"reach the end)\b", text))


def explicit_linear_segment_cap_m(stage, current_cap_m=None,
                                  corridor_active=False):
    """Keep one explicitly straight semantic segment locally observable.

    A carried incoming bearing says *which way* to move, but it does not say
    that the whole visible corridor belongs to the current clause. Long floor
    rays can pass the named landmark or the next turn before a node is built.
    Limit only lexically straight linear clauses whose incoming-route corridor
    is actually active. Portal forms and ordinary curved clauses are unchanged.
    """
    form = str((stage or {}).get("form", "")).upper()
    if not corridor_active or form not in {
            "ADVANCE_STRAIGHT", "PASS_LANDMARK", "CROSS_SPACE",
            "FOLLOW_PATH_BOUNDARY", "BETWEEN_OBJECTS"}:
        return current_cap_m
    if not inherits_incoming_route_corridor(stage):
        return current_cap_m
    return min(float(current_cap_m or 2.5), 2.5)


def circumnavigation_continuation_cap_m(current_cap_m=None,
                                        stage_progress_active=False):
    """Bound the second around-obstacle point before the following clause."""
    if not stage_progress_active:
        return current_cap_m
    return min(float(current_cap_m or 3.5), 3.5)


def infer_turn_carryover(stage, action_history, minimum_turn_deg=20.0,
                         minimum_motion_m=0.50):
    """Detect a turn already executed on the incoming edge.

    Natural-language clauses and physical stopping nodes do not always share
    a boundary: an EXIT edge can cross a doorway and perform the following
    small turn before the next node is created.  Repeating a full side turn at
    that node corrupts the route.  This form-level helper uses only persisted
    action names/turn deltas and translation; it never reads a scene, path,
    depth, or episode identity.
    """
    form = str(stage.get("form", "")).upper()
    actions = list(action_history or [])
    moved = sum(float(item.get("moved_m", 0.0) or 0.0) for item in actions)
    signed_turn = sum(float(item.get("turn_deg", 0.0) or 0.0)
                      for item in actions)
    if moved < float(minimum_motion_m):
        return None
    # A forward/pass clause immediately following a completed turn must use
    # the resulting corridor, even when the turn was persisted on the prior
    # edge.  In that case the geometrically correct outgoing lane can lie in
    # the old incoming/rear hemisphere, so the normal reverse-sector guard is
    # intentionally relaxed by the selector (see point_selectors.py).
    forward_after_turn = form in {"PASS_LANDMARK", "ADVANCE_STRAIGHT",
                                  "FOLLOW_PATH_BOUNDARY"}
    if form not in {"TURN_LEFT", "TURN_RIGHT", "TURN_AROUND"} and not forward_after_turn:
        return None
    if (form == "TURN_LEFT" or forward_after_turn) and signed_turn >= float(minimum_turn_deg):
        direction = "left"
    elif form == "TURN_RIGHT" and signed_turn <= -float(minimum_turn_deg):
        direction = "right"
    elif form == "TURN_AROUND" and abs(signed_turn) >= 120.0:
        direction = "around"
    else:
        return None
    return {
        "active": True,
        "stage_id": int(stage.get("sub_instruction_id",
                               stage.get("stage_id", -1))),
        "direction": direction,
        "signed_turn_deg": float(signed_turn),
        "traveled_distance_m": float(moved),
        "policy": "incoming edge already contains the requested turn; continue along its resulting corridor",
    }


@dataclass
class SequenceDirective:
    action: str
    expected_sub_instruction_id: int | None
    matched_sub_instruction_id: int
    node_id: str
    backtrack_target_node_id: str | None = None
    direction_to_block_yaw_rad: float | None = None
    reason: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class InstructionSequenceExplorationResult:
    success: bool
    end_reason: str
    completed_sub_instructions: int
    expected_sub_instruction_id: int | None
    exploration_hops: int
    recovery_backtracks: int
    final_yaw: float
    next_global_step: int
    traveled_distance_m: float
    selected_yaws_rad: list[float]
    records: list[dict]
    recovery_records: list[dict]
    state: dict

    def to_dict(self):
        return asdict(self)


class InstructionSequenceStateMachine:
    """Strict ordered-sub-instruction state with exactly one lookahead hop."""

    def __init__(self, sub_instructions, initial_node_id,
                 max_blocked_directions_per_node=5,
                 minimum_classification_confidence=0.5,
                 max_consecutive_on_route_unknowns=1):
        self.sub_instructions = list(sub_instructions)
        self.ids = [int(item.sub_instruction_id) for item in self.sub_instructions]
        if not self.ids:
            raise ValueError("instruction sequence must be non-empty")
        self.cursor = 0
        self.last_verified_node_id = str(initial_node_id)
        self.branch_origin_node_id = None
        self.branch_direction_yaw_rad = None
        self.branch_route_yaw_rad = None
        self.portal_threshold_retry_after_backtrack = False
        self.portal_threshold_retry_count = 0
        self.off_sequence_nodes = []
        self.pending_block = None
        self.blocked_yaws_by_verified_node: dict[str, list[float]] = {}
        # Subset of the blocks above that were confirmed by the completion
        # judge; only these count toward ``max_blocked_directions_per_node``.
        self.judge_blocked_yaws_by_verified_node: dict[str, list[float]] = {}
        self.in_place_turn_records_by_sub_instruction_id: dict[int, dict] = {}
        self.classification_history = []
        self.recovery_backtracks = 0
        self.consecutive_on_route_unknowns = 0
        # Project contract: after one UNKNOWN arrival, allow exactly one
        # outgoing lookahead.  A second UNKNOWN must backtrack to that first
        # node and block B->C; it must not receive a third speculative hop.
        self.max_consecutive_on_route_unknowns = min(
            1, max(0, int(max_consecutive_on_route_unknowns)))
        self.max_blocked_directions_per_node = int(
            max_blocked_directions_per_node)
        self.minimum_classification_confidence = float(
            minimum_classification_confidence)
        self.terminated_reason = None

    @property
    def complete(self):
        return self.cursor >= len(self.ids)

    @property
    def expected_sub_instruction_id(self):
        return None if self.complete else self.ids[self.cursor]

    @property
    def active_sub_instruction(self):
        return None if self.complete else self.sub_instructions[self.cursor]

    def blocked_yaws(self):
        selection_base = (self.branch_origin_node_id
                          if self.off_sequence_nodes else
                          self.last_verified_node_id)
        return list(self.blocked_yaws_by_verified_node.get(selection_base, []))

    @staticmethod
    def _append_unique_yaw(values, yaw):
        if not any(abs(_wrap_angle(yaw - value)) < math.radians(20)
                   for value in values):
            values.append(float(yaw))

    def _add_block(self, node_id, yaw, *, counts_toward_cap=True):
        node_key = str(node_id)
        self._append_unique_yaw(
            self.blocked_yaws_by_verified_node.setdefault(node_key, []), yaw)
        if not counts_toward_cap:
            return
        cap_values = self.judge_blocked_yaws_by_verified_node.setdefault(
            node_key, [])
        self._append_unique_yaw(cap_values, yaw)
        if len(cap_values) >= self.max_blocked_directions_per_node:
            self.terminated_reason = "all_candidate_directions_blocked_at_verified_node"

    def block_failed_physical_direction(self, yaw):
        """Remember an executor-failed ray at the current logical origin.

        A failed point attempt never creates a graph node, so it cannot flow
        through :meth:`observe`.  Without an explicit block the next loop sees
        the identical verified state and may choose the same unreachable ray
        until the global hop limit.  This records only the online attempted
        direction; it does not infer semantic correctness or use a reference
        trajectory.  A physical failure says the chosen point was unreachable,
        not that the semantic direction is wrong, so it is hard-excluded but
        does not consume the judge-driven block cap.
        """
        selection_base = (self.branch_origin_node_id
                          if self.off_sequence_nodes else
                          self.last_verified_node_id)
        self._add_block(selection_base, float(yaw), counts_toward_cap=False)
        return str(selection_base)

    def in_place_turn_record(self, sub_instruction_id):
        return self.in_place_turn_records_by_sub_instruction_id.get(
            int(sub_instruction_id))

    def record_in_place_turn(self, sub_instruction_id, record):
        self.in_place_turn_records_by_sub_instruction_id[
            int(sub_instruction_id)] = dict(record)

    def observe(self, node_id, classification, selected_yaw):
        node_id = str(node_id)
        matched = int(classification.get("matched_sub_instruction_id", -1))
        expected = self.expected_sub_instruction_id
        confidence = float(classification.get("confidence", 0.0))
        correct = bool(
            classification.get("belongs_to_sequence", False) and
            matched == expected and
            confidence >= self.minimum_classification_confidence)
        self.classification_history.append({
            "node_id": node_id, "expected_sub_instruction_id": expected,
            "matched_sub_instruction_id": matched,
            "correct_sequence_position": correct,
            "confidence": confidence,
            "minimum_confidence": self.minimum_classification_confidence,
            "unknown_disposition": classification.get(
                "unknown_disposition"),
        })

        if correct:
            self.last_verified_node_id = node_id
            self.cursor += 1
            self.branch_origin_node_id = None
            self.branch_direction_yaw_rad = None
            self.branch_route_yaw_rad = None
            self.portal_threshold_retry_after_backtrack = False
            self.portal_threshold_retry_count = 0
            self.off_sequence_nodes = []
            self.pending_block = None
            self.consecutive_on_route_unknowns = 0
            return SequenceDirective(
                action="complete" if self.complete else "advance_sequence",
                expected_sub_instruction_id=self.expected_sub_instruction_id,
                matched_sub_instruction_id=matched, node_id=node_id,
                reason="node matches the expected ordered sub-instruction")

        # ``unknown`` is deliberately the only public completion-negative
        # status.  A structured judge may nevertheless provide positive,
        # non-contradictory *partial* evidence.  Treat that evidence as an
        # in-route continuation rather than manufacturing an off-sequence
        # branch; the active sub-instruction remains unsatisfied and must be
        # selected again from the newly created real node.
        active_form = str(classification.get("active_form", "")).upper()
        if active_form in {"VERTICAL_UP", "VERTICAL_DOWN"}:
            partial_hop_limit = max(
                4, self.max_consecutive_on_route_unknowns)
        elif active_form == "EXIT_REGION":
            # An EXIT point is capped at three metres so a large source room
            # can require several real nodes before the named threshold.  A
            # chronology-blind endpoint audit still gates every node; permit
            # up to three same-portal progress hops before branch recovery.
            partial_hop_limit = max(
                3, self.max_consecutive_on_route_unknowns)
        else:
            partial_hop_limit = self.max_consecutive_on_route_unknowns
        if (str(classification.get("unknown_disposition", "")).lower() ==
                "on_route" and
                self.consecutive_on_route_unknowns < partial_hop_limit):
            self.last_verified_node_id = node_id
            # This physical node is the origin B of the one permitted trial
            # B->C.  Keeping it in off_sequence_nodes makes a second UNKNOWN
            # at C enter the existing backtrack path immediately.
            if self.branch_origin_node_id is None:
                self.branch_origin_node_id = node_id
            self.branch_direction_yaw_rad = None
            # Persist the absolute camera ray that produced supported partial
            # progress.  If the one permitted lookahead is still UNKNOWN and
            # recovery returns to this node, later alternatives must remain
            # in this semantic corridor rather than scanning the full circle.
            self.branch_route_yaw_rad = float(selected_yaw)
            if node_id not in self.off_sequence_nodes:
                self.off_sequence_nodes.append(node_id)
            self.pending_block = None
            self.consecutive_on_route_unknowns += 1
            return SequenceDirective(
                action="continue_current_instruction",
                expected_sub_instruction_id=expected,
                matched_sub_instruction_id=matched, node_id=node_id,
                reason=("completion is unknown but structured endpoint/motion "
                        "evidence supports continuing the active instruction"))

        if not self.off_sequence_nodes:
            # B is the first off-sequence node. Allow one outgoing trial B->C.
            # If C is also wrong, return to B and block B->C.
            self.branch_origin_node_id = node_id
            self.branch_direction_yaw_rad = None
            self.branch_route_yaw_rad = None
            self.off_sequence_nodes = [node_id]
            return SequenceDirective(
                action="explore_once_more",
                expected_sub_instruction_id=expected,
                matched_sub_instruction_id=matched, node_id=node_id,
                reason=("first off-sequence node; exactly one additional VLM "
                        "exploration hop is allowed"))

        self.off_sequence_nodes.append(node_id)
        self.branch_direction_yaw_rad = float(selected_yaw)
        preserve_supported_portal_route = bool(
            active_form in {
                "EXIT_REGION", "ENTER_REGION", "SELECT_PORTAL",
                "TRAVERSE_PORTAL_REGION"} and
            str(classification.get("unknown_disposition", "")).lower() ==
                "on_route" and
            classification.get("portal_endpoint_side_veto_applied") and
            self.portal_threshold_retry_count <
                MAX_PORTAL_THRESHOLD_ROUTE_RETRIES)
        self.pending_block = {
            "node_id": self.branch_origin_node_id,
            "yaw_rad": float(selected_yaw),
            "preserve_supported_portal_route": (
                preserve_supported_portal_route),
        }
        return SequenceDirective(
            action="backtrack_and_block",
            expected_sub_instruction_id=expected,
            matched_sub_instruction_id=matched, node_id=node_id,
            backtrack_target_node_id=self.branch_origin_node_id,
            direction_to_block_yaw_rad=self.branch_direction_yaw_rad,
            reason=("two consecutive nodes are outside the expected sequence; "
                    "return to the previous node and block its failed direction"))

    def on_backtrack(self, success, current_node_id=None):
        if not success:
            self.terminated_reason = "sequence_recovery_backtrack_failed"
            return
        if self.pending_block is None:
            raise RuntimeError("successful recovery has no pending direction block")
        failed_origin = str(self.pending_block["node_id"])
        failed_yaw = float(self.pending_block["yaw_rad"])
        preserve_supported_portal_route = bool(
            self.pending_block.get("preserve_supported_portal_route"))
        if not preserve_supported_portal_route:
            self._add_block(failed_origin, failed_yaw)
        self.portal_threshold_retry_after_backtrack = (
            preserve_supported_portal_route)
        if preserve_supported_portal_route:
            self.portal_threshold_retry_count += 1
        self.recovery_backtracks += 1
        # A backtrack attempt creates a real node at the physically revisited
        # pose.  It is often within the target node's tolerance but has a new
        # graph id; future selection must use that concrete node as the search
        # anchor, otherwise the next recovery route is planned from a stale
        # canonical id and can repeatedly traverse the wrong edge.  Mirror the
        # failed absolute yaw on the recovered node so the bad branch remains
        # blocked after this identity hand-off.  The optional argument keeps
        # the pure state-machine API backward compatible for unit tests.
        recovered = (str(current_node_id)
                     if current_node_id is not None else failed_origin)
        if recovered != failed_origin:
            # A physical revisit has a fresh graph id but represents the same
            # logical branch origin.  Carry the complete exclusion set across
            # that identity hand-off.  Copying only the latest failed yaw
            # forgets all earlier trials and can cycle through them until the
            # global hop budget is exhausted.
            judge_yaws = self.judge_blocked_yaws_by_verified_node.get(
                failed_origin, [])
            for blocked_yaw in list(
                    self.blocked_yaws_by_verified_node.get(
                        failed_origin, [])):
                self._add_block(
                    recovered, blocked_yaw,
                    counts_toward_cap=any(
                        abs(_wrap_angle(blocked_yaw - value)) < 1e-9
                        for value in judge_yaws))
        self.last_verified_node_id = recovered
        # B already consumed the one promised lookahead when B->C returned a
        # second UNKNOWN.  A physical revisit creates a new graph id for B,
        # but must not reset that semantic fact: each later alternative B->D
        # is itself the one-hop trial and, if UNKNOWN, returns directly to B.
        # Resetting to an empty branch here previously allowed B->D->E, moved
        # the branch origin forward, forgot exclusions, and could churn until
        # the global hop budget expired.
        self.branch_origin_node_id = recovered
        self.off_sequence_nodes = [recovered]
        self.branch_direction_yaw_rad = None
        self.pending_block = None
        self.consecutive_on_route_unknowns = 1

    def snapshot(self):
        return {
            "sub_instruction_ids": self.ids,
            "cursor": self.cursor,
            "expected_sub_instruction_id": self.expected_sub_instruction_id,
            "completed": self.complete,
            "last_verified_node_id": self.last_verified_node_id,
            "branch_origin_node_id": self.branch_origin_node_id,
            "branch_route_yaw_rad": self.branch_route_yaw_rad,
            "portal_threshold_retry_after_backtrack": (
                self.portal_threshold_retry_after_backtrack),
            "portal_threshold_retry_count": self.portal_threshold_retry_count,
            "off_sequence_nodes": list(self.off_sequence_nodes),
            "blocked_yaws_by_verified_node": {
                node_id: list(values) for node_id, values in
                self.blocked_yaws_by_verified_node.items()},
            "judge_blocked_yaws_by_verified_node": {
                node_id: list(values) for node_id, values in
                self.judge_blocked_yaws_by_verified_node.items()},
            "in_place_turn_records_by_sub_instruction_id": {
                int(key): dict(value) for key, value in
                self.in_place_turn_records_by_sub_instruction_id.items()},
            "classification_history": list(self.classification_history),
            "recovery_backtracks": self.recovery_backtracks,
            "consecutive_on_route_unknowns": (
                self.consecutive_on_route_unknowns),
            "max_consecutive_on_route_unknowns": (
                self.max_consecutive_on_route_unknowns),
            "minimum_classification_confidence": (
                self.minimum_classification_confidence),
            "terminated_reason": self.terminated_reason,
        }


class VLMNodeSequenceClassifier:
    """Replaceable adapter around the navigation VLM harness."""

    def __init__(self, vlm_harness, graph_memory, sub_instructions):
        self.vlm_harness = vlm_harness
        self.graph_memory = graph_memory
        self.sub_instructions = list(sub_instructions)

    def classify(self, node, expected_sub_instruction_id, six_views,
                 classification_history=None):
        return self.vlm_harness.classify_node_sub_instruction_sequence(
            sub_instructions=self.sub_instructions,
            expected_sub_instruction_id=expected_sub_instruction_id,
            node_id=node.node_id, six_views=six_views,
            environment_semantics=node.environment_semantics,
            classification_history=classification_history)


class InstructionSequenceExplorationStrategy:
    """Complete VLM selection/execution/node-validation/recovery strategy."""

    mode = "instruction-sequence-recovery"

    def __init__(
            self, sim, sub_instructions, point_selector,
            point_navigation_executor, graph_memory, vlm_harness,
            segmenter, position_history, rendered, video_composer, motion_log,
            output_dir, scan_step_deg=10.0, max_exploration_hops=30,
            max_blocked_directions_per_node=5,
            minimum_classification_confidence=0.5,
            recovery_backtrack_attempts_per_hop=4,
            recovery_reach_radius_m=0.75,
            recovery_minimum_visual_similarity=0.75,
            initial_incoming_origin=None,
            initial_previous_action_history=None,
            backtrack_planner_profile="legacy_direct",
            reference_path=None, reference_path_index=0,
            full_instruction=None, instruction_context_sub_instructions=None):
        self.sim = sim
        self.sub_instructions = list(sub_instructions)
        self.instruction_context_sub_instructions = list(
            instruction_context_sub_instructions or sub_instructions)
        # Immutable complete instruction for the dedicated video tile.  The
        # active sub-instruction remains a separate field on each request.
        self.full_instruction = full_instruction
        self.point_selector = point_selector
        self.point_navigation_executor = point_navigation_executor
        self.graph_memory = graph_memory
        self.vlm_harness = vlm_harness
        self.position_history = position_history
        self.rendered = rendered
        self.video_composer = video_composer
        self.motion_log = motion_log
        # Hidden diagnostic context used only by point-selector overlays.
        # Never forward the reference path to the VLM harness.
        self.reference_path = reference_path
        self.reference_path_index = int(reference_path_index or 0)
        self.output_dir = Path(output_dir)
        self.initial_incoming_origin = (
            None if initial_incoming_origin is None else
            np.asarray(initial_incoming_origin, np.float32))
        self.initial_previous_action_history = list(
            initial_previous_action_history or [])
        self.selection_dir = self.output_dir / "sequence_exploration"
        self.selection_dir.mkdir(parents=True, exist_ok=True)
        self.max_exploration_hops = int(max_exploration_hops)
        self.completion_judge = NodeTransitionInstructionCompletionJudge(
            vlm_harness, graph_memory, self.sub_instructions,
            minimum_confidence=minimum_classification_confidence)
        self.state = InstructionSequenceStateMachine(
            self.sub_instructions, graph_memory.nodes[-1].node_id,
            max_blocked_directions_per_node,
            minimum_classification_confidence)
        self.backtracker = NodeBacktrackingController(
            sim=sim, graph_memory=graph_memory, segmenter=segmenter,
            point_navigation_executor=point_navigation_executor,
            position_history=position_history, rendered=rendered,
            video_composer=video_composer, motion_log=motion_log,
            vlm_harness=vlm_harness, selector_mode="vlm",
            scan_step_deg=scan_step_deg,
            max_attempts_per_hop=recovery_backtrack_attempts_per_hop,
            reach_radius_m=recovery_reach_radius_m,
            minimum_visual_similarity=recovery_minimum_visual_similarity,
            max_hops=8, output_dir=output_dir,
            planner_profile=backtrack_planner_profile,
            reference_path=self.reference_path,
            reference_path_index=self.reference_path_index)

    @staticmethod
    def _candidate_records(candidates):
        return [{
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in candidate.items()
            if key not in {
                "rgb", "depth", "mask", "target_mask", "point",
                "semantic_detections", "small_seg_object_mask", "ground_anchors",
            }
        } for candidate in candidates]

    def _save_selection(self, candidates, chosen, hop_index):
        views = []
        ground_dir = self.output_dir / "ground_masks"
        target_dir = self.output_dir / "target_masks"
        depth_dir = self.output_dir / "selection_depths"
        candidate_dir = self.output_dir / "selection_six_views"
        ground_dir.mkdir(exist_ok=True)
        target_dir.mkdir(exist_ok=True)
        depth_dir.mkdir(exist_ok=True)
        candidate_dir.mkdir(exist_ok=True)
        for candidate in candidates:
            view_index = int(candidate.get("view_index", len(views)))
            Image.fromarray(candidate["rgb"]).save(
                candidate_dir / f"hop_{hop_index:03d}_view_{view_index}.jpg")
            Image.fromarray(
                (candidate["mask"].astype(np.uint8) * 255)).save(
                ground_dir / f"hop_{hop_index:03d}_view_{view_index}.png")
            Image.fromarray(
                (candidate["target_mask"].astype(np.uint8) * 255)).save(
                target_dir / f"hop_{hop_index:03d}_view_{view_index}.png")
            np.save(
                depth_dir / f"hop_{hop_index:03d}_view_{view_index}.npy",
                np.asarray(candidate["depth"], np.float32))
            rgb = candidate["rgb"].copy()
            green = np.zeros_like(rgb); green[..., 1] = 255
            mask = candidate["target_mask"]
            rgb[mask] = (0.55 * rgb[mask] + 0.45 * green[mask]).astype(np.uint8)
            projection = candidate.get("reference_path_projection")
            if projection is None and self.reference_path:
                projection = project_reference_path(
                    self.reference_path,
                    range(self.reference_path_index, len(self.reference_path)),
                    np.asarray(self.sim.get_agent(0).get_state().position,
                               np.float32) + np.array([0.0, 1.25, 0.0], np.float32),
                    float(candidate["yaw"]), rgb.shape[1], rgb.shape[0])
            if projection is not None:
                rgb = draw_reference_path_overlay(
                    rgb, projection,
                    selected_point=(chosen.get("point") if candidate is chosen else None),
                    selected=bool(candidate is chosen),
                    next_path_index=self.reference_path_index + 1)
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            status = ("BLOCKED" if candidate.get("hard_excluded") else
                      "SOFT EXCLUDED" if candidate.get("excluded") else "ALLOWED")
            cv2.putText(frame, f"VIEW {candidate.get('view_index', len(views))} {status}",
                        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0, 0, 255) if "BLOCKED" in status else (255, 255, 255), 1)
            if candidate is chosen:
                cv2.drawMarker(frame, tuple(np.round(chosen["point"]).astype(int)),
                               (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
            views.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        sheet = selection_contact_sheet(views)
        Image.fromarray(sheet).save(
            self.selection_dir / f"hop_{hop_index:03d}_vlm_selection.jpg")

    def run(self, initial_yaw=0.0, initial_global_step=0):
        # Lazy imports preserve unit-test access to the pure state machine.
        from point_navigation_executor import PointNavigationRequest, execute_point_navigation
        from point_selectors import (
            PointSelectionRequest, continuous_turn, observe_eight_rgb,
            pixel_ground_to_world, set_pose,
        )

        yaw = float(initial_yaw)
        global_step = int(initial_global_step)
        records, recovery_records, selected_yaws = [], [], []
        selection_failures = []
        previous_action_history = list(self.initial_previous_action_history)
        previous_hop_origin = self.initial_incoming_origin
        traveled_distance = 0.0
        end_reason = "max_sequence_exploration_hops"
        require_progress_after_unknown = False
        turn_carryover_active = None
        stage_progress_carryover_active = None
        exit_portal_reference_rgb = None
        exit_portal_reference_stage_id = None

        for hop_index in range(self.max_exploration_hops):
            if self.state.complete:
                end_reason = "instruction_sequence_complete"
                break
            if self.state.terminated_reason is not None:
                end_reason = self.state.terminated_reason
                break
            sub_instruction = self.state.active_sub_instruction
            stage = sub_instruction.to_stage_dict()
            current_stage_id = int(stage.get(
                "sub_instruction_id", stage.get("stage_id", -1)))
            if (exit_portal_reference_stage_id is not None and
                    int(exit_portal_reference_stage_id) != current_stage_id):
                exit_portal_reference_rgb = None
                exit_portal_reference_stage_id = None
            if (stage_progress_carryover_active is not None and
                    int(stage_progress_carryover_active.get(
                        "stage_id", -2)) != current_stage_id):
                stage_progress_carryover_active = None
            if (turn_carryover_active is not None and
                    int(stage.get("sub_instruction_id",
                                stage.get("stage_id", -1))) !=
                    int(turn_carryover_active.get("stage_id", -2))):
                turn_carryover_active = None
            # Never reinterpret an active TURN instruction as already
            # completed merely because the preceding EXIT/portal edge
            # contains incidental turn-to-view control actions.  Carryover is
            # only meaningful for the following forward/pass clause.
            if (turn_carryover_active is None and
                    str(stage.get("form", "")).upper() in {
                        "PASS_LANDMARK", "ADVANCE_STRAIGHT",
                        "FOLLOW_PATH_BOUNDARY"}):
                turn_carryover_active = infer_turn_carryover(
                    stage, previous_action_history)
            # A turn-to-view occurs before edge action history is recorded, so
            # its yaw change may be absent from ``turn_deg``.  Recover the
            # completed-turn context from the verified source node's
            # departure form and source/predecessor headings.  This graph-local
            # rule keeps a following PASS/ADVANCE clause from reapplying the
            # old incoming-direction exclusion; no reference path or scene
            # identity is consulted.
            if (turn_carryover_active is None and
                    str(stage.get("form", "")).upper() in {
                        "PASS_LANDMARK", "ADVANCE_STRAIGHT",
                        "FOLLOW_PATH_BOUNDARY"} and self.graph_memory.nodes):
                source_node = self.graph_memory.nodes[-1]
                predecessor_node, _ = self.graph_memory.predecessor(
                    source_node.node_id)
                purpose = source_node.departure_purpose_sub_instruction or {}
                prior_form = str(purpose.get("form", "")).upper()
                if predecessor_node is not None and prior_form in {
                        "TURN_LEFT", "TURN_RIGHT", "TURN_AROUND"}:
                    heading_delta = _wrap_angle(
                        float(source_node.base_yaw_rad) -
                        float(predecessor_node.base_yaw_rad))
                    if abs(math.degrees(heading_delta)) >= 20.0:
                        turn_carryover_active = {
                            "active": True,
                            "stage_id": int(stage.get("sub_instruction_id",
                                                   stage.get("stage_id", -1))),
                            "direction": ("left" if heading_delta > 0.0
                                          else "right"),
                            "signed_turn_deg": float(math.degrees(heading_delta)),
                            "traveled_distance_m": float(sum(
                                float(item.get("moved_m", 0.0) or 0.0)
                                for item in previous_action_history)),
                            "policy": "verified source node heading records the completed prior turn",
                        }
            selection_stage = dict(stage)
            preselection_orientation_alignment = None
            recovery_form = stage.get("form")
            linear_recovery_corridor = fixed_recovery_corridor_eligible(
                recovery_form)
            portal_recovery_hemisphere = portal_recovery_corridor_active(
                recovery_form,
                self.state.portal_threshold_retry_after_backtrack)
            if (self.state.branch_route_yaw_rad is not None and
                    self.state.off_sequence_nodes and
                    (linear_recovery_corridor or
                     portal_recovery_hemisphere)):
                recovery_half_width_deg = (
                    45.0 if linear_recovery_corridor else 20.0)
                selection_stage["metadata"] = dict(
                    stage.get("metadata", {}) or {},
                    recovery_route_corridor={
                        "active": True,
                        "absolute_yaw_rad": float(
                            self.state.branch_route_yaw_rad),
                        "half_width_deg": recovery_half_width_deg,
                        "route_kind": (
                            "linear_corridor" if linear_recovery_corridor
                            else "portal_threshold_same_route_retry"),
                        "policy": (
                            "recovery alternatives remain in the online "
                            "supported-partial route; portal paths may bend "
                            "within the forward hemisphere but cannot "
                            "traverse the same doorway backwards"),
                    })
            source_node = (self.graph_memory.nodes[-1]
                           if self.graph_memory.nodes else None)
            if (source_node is not None and
                    inherits_incoming_route_corridor(selection_stage) and
                    not (selection_stage.get("metadata", {}) or {}).get(
                        "instruction_committed_route_corridor")):
                _, incoming_source_edge = self.graph_memory.predecessor(
                    source_node.node_id)
                incoming_route_yaw = (
                    (incoming_source_edge.metadata or {}).get(
                        "selected_yaw_rad")
                    if incoming_source_edge is not None else None)
                if incoming_route_yaw is not None:
                    selection_stage["metadata"] = dict(
                        selection_stage.get("metadata", {}) or {},
                        instruction_committed_route_corridor={
                            "active": True,
                            "absolute_yaw_rad": float(incoming_route_yaw),
                            "half_width_deg": 20.0,
                            "source": "incoming_verified_edge_route",
                            "policy": (
                                "unqualified next portal continues the "
                                "incoming semantic route; side/reverse "
                                "portals require explicit instruction"),
                        })
            source_purpose = (
                source_node.departure_purpose_sub_instruction
                if source_node is not None else {}) or {}
            source_stage_id = source_purpose.get(
                "sub_instruction_id", source_purpose.get("stage_id"))
            source_form = str(source_purpose.get("form", "")).upper()
            post_vertical_stage_transition = bool(
                source_form in {"VERTICAL_UP", "VERTICAL_DOWN"} and
                source_stage_id is not None and
                int(source_stage_id) != current_stage_id)
            if post_vertical_stage_transition:
                selection_stage["metadata"] = dict(
                    stage.get("metadata", {}) or {},
                    post_vertical_stage_transition={
                        "active": True,
                        "source_form": source_form,
                        "policy": (
                            "reopen landing panorama because horizontal "
                            "incoming overlap does not imply vertical "
                            "backtracking; reject descending-stair rays "
                            "semantically"),
                    })
            # The following decomposed clause is user-provided route context,
            # not demonstration leakage.  It may break a tie between two
            # current-stage outgoing lanes, but is never the current
            # completion target.
            context_index = self.state.cursor + 1
            next_context = None
            if context_index < len(self.instruction_context_sub_instructions):
                next_context = self.instruction_context_sub_instructions[
                    context_index]
                selection_stage["next_sub_instruction_context"] = {
                    "navigation_instruction": next_context.navigation_instruction,
                    "landmark": next_context.landmark,
                    "form": next_context.form,
                    "semantic_spatial_target": (
                        next_context.semantic_spatial_target),
                }
                selection_stage["following_sub_instruction_contexts"] = [{
                    "navigation_instruction": context.navigation_instruction,
                    "landmark": context.landmark,
                    "form": context.form,
                    "semantic_spatial_target": context.semantic_spatial_target,
                } for context in self.instruction_context_sub_instructions[
                    context_index:context_index + 3]]
            turn_to_landmark_alignment_stage = None
            bare_turn_route_setup_active = False
            if (str(stage.get("form", "")).upper() == "TURN_TO_LANDMARK" and
                    next_context is not None and
                    not require_progress_after_unknown):
                # A named-landmark turn is an orientation boundary, whereas
                # this project requires every graph edge to make real route
                # progress. Use the immediately following clause to choose a
                # short outgoing-route floor point, then reacquire and face
                # the current landmark from that arrived state below. This is
                # instruction-only lookahead, never demonstration leakage.
                route_setup, turn_to_landmark_alignment_stage = (
                    named_landmark_route_setup_stage(
                        stage, next_context, require_progress_after_unknown))
                route_setup["next_sub_instruction_context"] = (
                    selection_stage.get("next_sub_instruction_context"))
                route_setup["following_sub_instruction_contexts"] = (
                    selection_stage.get("following_sub_instruction_contexts"))
                selection_stage = route_setup
            elif (bare_turn_delta_rad(stage) is not None and
                  next_context is not None and
                  not require_progress_after_unknown):
                # Like a named-landmark turn, a bare turn defines the arrived
                # heading but does not itself provide a useful floor endpoint.
                # Translate on the next stated route and preserve the exact
                # current-stage +/-90/180 alignment below.
                route_setup, bare_turn_route_setup_active = (
                    bare_turn_route_setup_stage(
                        stage, next_context, require_progress_after_unknown))
                route_setup["next_sub_instruction_context"] = (
                    selection_stage.get("next_sub_instruction_context"))
                route_setup["following_sub_instruction_contexts"] = (
                    selection_stage.get("following_sub_instruction_contexts"))
                selection_stage = route_setup
            if turn_carryover_active is not None:
                # Keep the original stage for node construction and
                # completion judging, but ask the point selector for the
                # resulting forward corridor rather than applying the same
                # turn a second time.
                selection_stage["form"] = "ADVANCE_STRAIGHT"
                selection_stage["navigation_instruction"] = (
                    "Continue forward along the corridor after the "
                    f"completed {turn_carryover_active['direction']} turn")
                selection_stage["spatial_relation"] = (
                    "along the connected corridor after the incoming turn")
                # Preserve route constraints already attached above.  Using
                # the pristine decomposition metadata here used to erase the
                # incoming committed corridor exactly when a preceding turn
                # was carried into a straight/pass clause.
                selection_stage["metadata"] = dict(
                    selection_stage.get("metadata", {}) or {},
                    turn_carryover=turn_carryover_active)
            if require_progress_after_unknown:
                # One real partial edge earns exactly one lookahead.  Keep
                # that lookahead on the route currently in front of the
                # arrived camera instead of allowing a fresh 360° branch
                # decision.  The VLM still selects among RGB ground anchors
                # inside the cone, and no demonstration information is used.
                continuation = {
                    "active": True,
                    "policy": (
                        "single supported-partial lookahead stays within "
                        "the arrived route's forward +/-45 degree cone"),
                }
                selection_stage["metadata"] = dict(
                    selection_stage.get("metadata", {}) or {},
                    supported_partial_route_continuation=continuation)
                active_form = str(stage.get("form", "")).upper()
                if exit_threshold_portal_continuation_required(
                        stage, stage_progress_carryover_active):
                    selection_stage["form"] = "TRAVERSE_PORTAL_REGION"
                    selection_stage["navigation_instruction"] = (
                        "Continue through the same already-selected doorway "
                        "into its destination space. Do not reapply the "
                        "initial left/right direction from the new camera "
                        "heading.")
                    selection_stage["spatial_relation"] = (
                        "through and beyond the already approached doorway")
                    selection_stage["metadata"][
                        "exit_initial_direction_consumed_continuation"] = {
                            "active": True,
                            "original_form": active_form,
                            "policy": (
                                "the first portal hop selected the named "
                                "portal; the audited-threshold hop follows "
                                "its visible opening and may bend"),
                        }
                if compound_turn_continuation_required(
                        stage, stage_progress_carryover_active):
                    # The first real edge established the signed route. Keep
                    # moving toward the compound endpoint without applying
                    # the direction phrase a second time.
                    prior_route_yaw = stage_progress_carryover_active.get(
                        "prior_selected_yaw_rad")
                    selection_stage["form"] = "ADVANCE_STRAIGHT"
                    selection_stage["navigation_instruction"] = (
                        "Continue on the already established route toward "
                        "the active instruction's named endpoint")
                    selection_stage["metadata"][
                        "turn_direction_consumed_continuation"] = {
                            "active": True,
                            "original_form": active_form,
                            "policy": (
                                "a supported partial edge consumed the signed "
                                "turn; continue on its frozen RGB-selected ray"),
                        }
                    selection_stage["metadata"][
                        "instruction_committed_route_corridor"] = {
                            "active": True,
                            "absolute_yaw_rad": float(prior_route_yaw),
                            "half_width_deg": 30.0,
                            "allow_bend_fallback": True,
                            "source": "compound_turn_partial_prior_vlm_ray",
                            "policy": (
                                "same-stage endpoint continuation cannot "
                                "repeat the turn or reverse the established "
                                "route; if that ray has no RGB ground, one "
                                "history-safe adjacent bend may be reacquired"),
                        }
                elif active_form in {
                        "VERTICAL_UP", "VERTICAL_DOWN"}:
                    selection_stage["metadata"]["vertical_continuation"] = {
                        "active": True,
                        "policy": (
                            "the initial turn-to-stairs phrase was consumed; "
                            "continue along the current stair flight"),
                    }
                elif (active_form == "CIRCUMNAVIGATE" and
                      isinstance(stage_progress_carryover_active, dict)):
                    # Around/backside paths may curve, but the V24 integrity
                    # contract permits at most a 60-degree inter-edge bend.
                    # Enforce that same bound during selection so a visually
                    # tempting 90-degree branch cannot be executed and only
                    # rejected after the fact.
                    prior_route_yaw = stage_progress_carryover_active.get(
                        "prior_selected_yaw_rad")
                    if prior_route_yaw is not None:
                        selection_stage["metadata"][
                            "instruction_committed_route_corridor"] = {
                                "active": True,
                                "absolute_yaw_rad": float(prior_route_yaw),
                                "half_width_deg": 60.0,
                                "allow_bend_fallback": False,
                                "source": (
                                    "circumnavigation_partial_prior_vlm_ray"),
                                "policy": (
                                    "same-stage around continuation may curve "
                                    "but cannot exceed its audited 60-degree "
                                    "inter-edge integrity bound"),
                            }
                elif (isinstance(stage_progress_carryover_active, dict) and
                      compound_terminal_extent_route(stage)):
                    # A compound "enter X and follow Y to its end" stage can
                    # bend around the finite landmark. Preserve the first
                    # VLM-selected route with one 60-degree bend, instead of
                    # reinterpreting the controller's terminal camera yaw as
                    # a fresh 360-degree branch and commonly reversing.
                    prior_route_yaw = stage_progress_carryover_active.get(
                        "prior_selected_yaw_rad")
                    if prior_route_yaw is not None:
                        selection_stage["metadata"][
                            "instruction_committed_route_corridor"] = {
                                "active": True,
                                "absolute_yaw_rad": float(prior_route_yaw),
                                "half_width_deg": 60.0,
                                "allow_bend_fallback": False,
                                "source": (
                                    "compound_terminal_extent_prior_vlm_ray"),
                                "policy": (
                                    "same-stage finite-landmark continuation "
                                    "may bend once but cannot reverse or "
                                    "select an unrelated branch"),
                            }
                elif (isinstance(stage_progress_carryover_active, dict) and
                      fixed_recovery_corridor_eligible(active_form)):
                    # The controller may finish a correct point hop facing
                    # 30--60 degrees away from its frozen semantic ray. A
                    # same-stage continuation must follow that VLM-committed
                    # absolute route, not reinterpret controller yaw as a new
                    # branch. Reserve half of the 30-degree acceptance budget
                    # for the preceding ray's unknown residual error rather
                    # than allowing two full per-edge errors to accumulate.
                    prior_route_yaw = stage_progress_carryover_active.get(
                        "prior_selected_yaw_rad")
                    if prior_route_yaw is not None:
                        selection_stage["metadata"][
                            "instruction_committed_route_corridor"] = {
                                "active": True,
                                "absolute_yaw_rad": float(prior_route_yaw),
                                "half_width_deg": (
                                    LINEAR_SAME_STAGE_CONTINUATION_HALF_WIDTH_DEG),
                                "source": "supported_partial_prior_vlm_ray",
                                "policy": (
                                    "same-stage continuation preserves the "
                                    "previous frozen semantic route rather "
                                    "than controller terminal yaw"),
                            }
            position = np.asarray(
                self.sim.get_agent(0).get_state().position, np.float32)
            explicit_turn_route_frame_yaw = None
            if (source_node is not None and
                    source_stage_id is not None and
                    int(source_stage_id) != current_stage_id and
                    previous_hop_origin is not None and
                    not require_progress_after_unknown and
                    str(stage.get("form", "")).upper() in {
                        "TURN_LEFT", "TURN_RIGHT"}):
                # A point hop can finish with the camera looking away from
                # its actual direction of travel.  Explicit left/right is
                # relative to the incoming route, not that controller yaw.
                # Recover the route frame only from the preceding real graph
                # displacement (never depth or the demonstration path), and
                # normalize the panorama before applying the selector's
                # fixed three-view turn contract.
                explicit_turn_route_frame_yaw = incoming_route_yaw_rad(
                    previous_hop_origin, position)
                if explicit_turn_route_frame_yaw is not None:
                    alignment_delta = _wrap_angle(
                        float(explicit_turn_route_frame_yaw) - float(yaw))
                    if (math.radians(15.0) < abs(alignment_delta) <=
                            math.radians(120.0)):
                        alignment_start_yaw = float(yaw)
                        target_yaw = float(explicit_turn_route_frame_yaw)
                        alignment_steps = max(
                            1, int(math.ceil(abs(alignment_delta) /
                                             math.radians(10.0))))
                        yaw, _ = continuous_turn(
                            self.sim, position, alignment_start_yaw,
                            target_yaw, math.radians(10.0), self.rendered,
                            self.motion_log, hop_index,
                            "preselection_incoming_route_frame_alignment",
                            self.video_composer, self.position_history,
                            sub_instruction.navigation_instruction,
                            reference_path=self.reference_path,
                            reference_path_index=self.reference_path_index)
                        set_pose(self.sim, position, yaw)
                        global_step += alignment_steps
                        preselection_orientation_alignment = {
                            "status": (
                                "aligned_to_incoming_route_frame_for_"
                                "explicit_turn"),
                            "form": str(stage.get("form", "")).upper(),
                            "yaw_delta_deg": float(math.degrees(
                                alignment_delta)),
                            "source_node_yaw_rad": alignment_start_yaw,
                            "target_yaw_rad": target_yaw,
                            "final_yaw_rad": float(yaw),
                            "rendered_turn_frames": alignment_steps,
                            "policy": (
                                "normalize explicit turn to the preceding "
                                "real translation direction"),
                        }
            if (not bare_turn_route_setup_active and source_node is not None and
                    source_stage_id is not None and
                    int(source_stage_id) != current_stage_id and
                    str(selection_stage.get("form", "")).upper() in {
                        "EXIT_REGION", "ENTER_REGION", "SELECT_PORTAL",
                        "TRAVERSE_PORTAL_REGION"}):
                transition_text = " ".join(str(selection_stage.get(key, ""))
                                             for key in (
                                                 "navigation_instruction",
                                                 "spatial_relation",
                                                 "source_clause")).lower()
                has_explicit_direction = bool(re.search(
                    r"\b(?:turn|left|right|behind|back|rear|around|"
                    r"u[- ]?turn)\b", transition_text))
                _, incoming_edge_for_alignment = self.graph_memory.predecessor(
                    source_node.node_id)
                prior_semantic_yaw = (
                    (incoming_edge_for_alignment.metadata or {}).get(
                        "selected_yaw_rad")
                    if incoming_edge_for_alignment is not None else None)
                if prior_semantic_yaw is not None and not has_explicit_direction:
                    # Point tracking can stop with the camera 30--60 degrees
                    # away from the VLM-frozen route even though translation
                    # was correct. Recenter the next portal panorama on that
                    # persisted semantic ray, without restricting its 360°
                    # candidate set. A genuinely turning doorway remains free
                    # for RGB selection; this is only a camera-frame reset.
                    alignment_delta = _wrap_angle(
                        float(prior_semantic_yaw) - float(yaw))
                    if (math.radians(15.0) < abs(alignment_delta) <=
                            math.radians(75.0)):
                        alignment_start_yaw = float(yaw)
                        target_yaw = float(prior_semantic_yaw)
                        alignment_steps = max(
                            1, int(math.ceil(abs(alignment_delta) /
                                             math.radians(10.0))))
                        yaw, _ = continuous_turn(
                            self.sim, position, alignment_start_yaw, target_yaw,
                            math.radians(10.0), self.rendered, self.motion_log,
                            hop_index, "preselection_route_frame_alignment",
                            self.video_composer, self.position_history,
                            sub_instruction.navigation_instruction,
                            reference_path=self.reference_path,
                            reference_path_index=self.reference_path_index)
                        set_pose(self.sim, position, yaw)
                        global_step += alignment_steps
                        preselection_orientation_alignment = {
                            "status": "aligned_to_incoming_semantic_route",
                            "form": str(selection_stage.get(
                                "form", "")).upper(),
                            "yaw_delta_deg": float(math.degrees(
                                alignment_delta)),
                            "source_node_yaw_rad": alignment_start_yaw,
                            "target_yaw_rad": target_yaw,
                            "final_yaw_rad": float(yaw),
                            "rendered_turn_frames": alignment_steps,
                            "policy": (
                                "normalize the panorama to the prior VLM ray; "
                                "do not constrain the new portal direction"),
                        }
            if bare_turn_route_setup_active:
                # Keep selection in the verified source-node frame.  The
                # point selector consumes ``bare_turn_route_setup`` and limits
                # candidates to the commanded 45/90/135-degree side views.
                # Rotating first turns the following STOP/route clause into a
                # misleading generic forward target and bypasses that explicit
                # three-view contract.  Exact orientation is applied once,
                # after physical arrival, by the audited block below.
                preselection_orientation_alignment = {
                    "status": "deferred_until_physical_arrival",
                    "form": str(stage.get("form", "")).upper(),
                    "source_node_yaw_rad": float(
                        explicit_turn_route_frame_yaw
                        if explicit_turn_route_frame_yaw is not None else
                        (source_node.base_yaw_rad if source_node is not None
                         else yaw)),
                    "incoming_route_frame_normalized": bool(
                        explicit_turn_route_frame_yaw is not None),
                    "commanded_delta_deg": float(math.degrees(
                        bare_turn_delta_rad(stage))),
                    "policy": (
                        "select translated floor in the original three-view "
                        "turn sector, then perform one exact arrival alignment"),
                }
            back_yaw = None
            if previous_hop_origin is not None:
                vector = previous_hop_origin - position
                if np.linalg.norm(vector[[0, 2]]) > 0.05:
                    # The executor projects camera-forward motion as
                    # [-sin(yaw), -cos(yaw)].  Invert that mapping for the
                    # current->previous world ray so the true incoming
                    # direction is excluded from a new selection.
                    back_yaw = _wrap_angle(math.atan2(
                        -float(vector[0]), -float(vector[2])))
            if post_vertical_stage_transition:
                # On switchbacks the correct upper/lower-level continuation
                # can overlap the incoming stair's horizontal bearing.  A
                # planar back-yaw exclusion would remove it even though it is
                # on a different elevation.  Reopen the panorama only for
                # the first edge of the next sub-instruction; the VLM still
                # sees the explicit instruction and the persisted metadata
                # forbids choosing the descending stair itself.
                back_yaw = None
            blocked_yaws = self.state.blocked_yaws()
            # A selected RGB ray whose online endpoint is far beyond the
            # bounded execution budget is not a valid target.  Re-query the
            # same real state with that view blocked once, rather than
            # sending a known unsafe point to the executor.  This is a
            # generic selection/execution boundary; no scene or trajectory
            # identity is used.
            rejected_selection_yaws = []
            allow_historical_direction_fallback = False
            selection_result = None
            final_selection_rejected = False
            # A directional/transition sub-instruction must produce a real
            # branch-entry or region-crossing point, not a near-zero point at
            # the current pose.  STOP_WAIT is intentionally excluded because
            # a valid final relation can be camera-side and already nearby.
            # These relations require a different physical node. A point at
            # the source pose cannot establish an object gap, advance along a
            # boundary, or move around an obstacle; accepting it can let a
            # later loop plus semantic co-visibility create false completion.
            progress_required_form = form_requires_physical_progress(
                selection_stage.get("form", ""))
            # A portal/region transition is not complete at an arbitrary
            # near-side floor pixel: the camera must have enough physical
            # progress to cross the portal boundary.  Use a form-level lower
            # bound only on the first selection from a real node; once the
            # node judge reports partial/on-route, the ordinary 0.75 m
            # reacquisition bound applies.  This boundary uses no RGB model
            # input, depth as semantic evidence, demo path, or episode id.
            initial_transition_minimums = {
                "EXIT_REGION": 1.25,
                "ENTER_REGION": 1.00,
                "SELECT_PORTAL": 1.00,
                "TRAVERSE_PORTAL_REGION": 1.00,
            }
            # Pure directional turns are allowed to use a short physical
            # waypoint: the heading change itself is the semantic progress,
            # so insisting on a 0.75m geodesic can reject a valid branch-entry
            # point immediately beside the node.  Forward/portal relations
            # retain the stronger 0.75m anti-stall bound.
            current_form_upper = str(selection_stage.get("form", "")).upper()
            turn_progress_floor = 0.30 if current_form_upper in {
                "TURN_LEFT", "TURN_RIGHT", "TURN_AROUND",
            } else 0.75
            # A supported partial has already contributed a real >=0.5 m
            # translated edge.  Its one permitted lookahead can legitimately
            # expose only a short residual floor strip at a doorway, gap, or
            # endpoint.  Require a still-auditable two-control-step residual
            # move (0.35 m), while retaining the stronger first-hop 0.75 m
            # anti-stall bound for ordinary progress-required relations.
            if require_progress_after_unknown:
                minimum_selection_progress_m = 0.35
            elif progress_required_form:
                minimum_selection_progress_m = turn_progress_floor
            else:
                minimum_selection_progress_m = 0.0
            # A carried-over turn is a continuation from a real arrival node,
            # not a justification for accepting an unchanged floor pixel.  A
            # small positive bound plus the executor's forced motion attempt
            # ensures the next edge contains real history for the judge.
            if turn_carryover_active is not None:
                minimum_selection_progress_m = max(
                    minimum_selection_progress_m, 0.50)
            if bare_turn_route_setup_active:
                minimum_selection_progress_m = max(
                    minimum_selection_progress_m, 0.30)
            if (not require_progress_after_unknown and
                    not previous_action_history):
                minimum_selection_progress_m = max(
                    minimum_selection_progress_m,
                    float(initial_transition_minimums.get(
                        str(selection_stage.get("form", "")), 0.0)))
            cap_form = (str(stage.get("form", "")).upper()
                        if bare_turn_route_setup_active else
                        current_form_upper)
            maximum_selection_geodesic_m = selection_geodesic_cap_m(
                cap_form,
                turn_carryover=(turn_carryover_active is not None),
                retry_after_unknown=require_progress_after_unknown,
                bare_turn=bare_turn_route_setup_active or
                (bare_turn_delta_rad(selection_stage) is not None))
            maximum_selection_geodesic_m = local_portal_endpoint_cap_m(
                selection_stage, maximum_selection_geodesic_m)
            if (self.state.portal_threshold_retry_after_backtrack and
                    portal_recovery_hemisphere_eligible(
                        selection_stage.get("form"))):
                # The one normal lookahead was too shallow to put the camera
                # on the destination side.  Recovery returns to the partial
                # node and retries the same online-supported portal ray with
                # a gradually deeper but still local endpoint.  Do not block
                # the semantically correct ray merely because the physical
                # endpoint was insufficient; each retry is capped at 3--4 m.
                retry_cap_m = min(
                    4.0, 2.0 + float(
                        self.state.portal_threshold_retry_count))
                maximum_selection_geodesic_m = min(
                    float(maximum_selection_geodesic_m or retry_cap_m),
                    retry_cap_m)
                selection_stage["metadata"] = dict(
                    selection_stage.get("metadata", {}) or {},
                    portal_threshold_same_route_retry={
                        "active": True,
                        "retry_index": int(
                            self.state.portal_threshold_retry_count),
                        "maximum_geodesic_m": retry_cap_m,
                        "policy": (
                            "deepen the same supported portal ray after an "
                            "endpoint-side veto; never reverse through it"),
                    })
            if bool((selection_stage.get("metadata", {}) or {}).get(
                    "exit_initial_direction_consumed_continuation", {}).get(
                        "active")):
                # After the first portal edge has already selected the named
                # portal, the second edge is only a threshold clearance.  A
                # 3--4 m far-floor point commonly consumes the immediately
                # following landmark instruction before the node judge runs.
                # Keep this continuation local; post-selection can truncate
                # the same frozen RGB ray to a directly reachable endpoint.
                maximum_selection_geodesic_m = min(
                    float(maximum_selection_geodesic_m or 2.0), 2.0)
                selection_stage["metadata"][
                    "exit_threshold_clearance_cap"] = {
                        "active": True,
                        "maximum_geodesic_m": 2.0,
                        "policy": (
                            "same selected portal clears only its immediate "
                            "threshold before the next sub-instruction"),
                    }
            maximum_selection_geodesic_m = (
                directional_towards_route_setup_cap_m(
                    selection_stage, maximum_selection_geodesic_m))
            committed_route = (selection_stage.get("metadata", {}) or {}).get(
                "instruction_committed_route_corridor") or {}
            linear_segment_cap = explicit_linear_segment_cap_m(
                stage, maximum_selection_geodesic_m,
                corridor_active=bool(committed_route.get("active")))
            if linear_segment_cap != maximum_selection_geodesic_m:
                maximum_selection_geodesic_m = linear_segment_cap
                selection_stage["metadata"] = dict(
                    selection_stage.get("metadata", {}) or {},
                    explicit_linear_segment_cap={
                        "active": True,
                        "maximum_geodesic_m": float(
                            maximum_selection_geodesic_m),
                        "policy": (
                            "an explicit straight clause builds a local node "
                            "before it can consume the next semantic turn"),
                    })
            if (current_form_upper == "CIRCUMNAVIGATE" and
                    isinstance(stage_progress_carryover_active, dict)):
                maximum_selection_geodesic_m = (
                    circumnavigation_continuation_cap_m(
                        maximum_selection_geodesic_m,
                        stage_progress_active=True))
                selection_stage["metadata"] = dict(
                    selection_stage.get("metadata", {}) or {},
                    circumnavigation_continuation_cap={
                        "active": True,
                        "maximum_geodesic_m": float(
                            maximum_selection_geodesic_m),
                        "policy": (
                            "the second around-obstacle node stops near the "
                            "far edge before executing the following clause"),
                    })
            if (maximum_selection_geodesic_m is not None and
                    local_portal_endpoint_cap_m(selection_stage, None) is not
                    None):
                selection_stage["metadata"] = dict(
                    selection_stage.get("metadata", {}) or {},
                    local_portal_endpoint_cap={
                        "active": True,
                        "maximum_geodesic_m": float(
                            maximum_selection_geodesic_m),
                        "policy": (
                            "explicit just/immediately portal endpoint uses "
                            "the first safe floor patch across the boundary"),
                    })
            if (not bare_turn_route_setup_active and
                    local_landmark_stop_relation(stage)):
                # A local object relation must be approached in short,
                # re-observable increments.  A full three-metre STOP ray can
                # cross the object's near-side floor and leave the landmark
                # behind before the node judge runs.
                maximum_selection_geodesic_m = min(
                    float(maximum_selection_geodesic_m or
                          LOCAL_TERMINAL_RELATION_WAYPOINT_MAX_M),
                    LOCAL_TERMINAL_RELATION_WAYPOINT_MAX_M)
                selection_stage["metadata"] = dict(
                    selection_stage.get("metadata", {}) or {},
                    local_terminal_relation_cap={
                        "active": True,
                        "maximum_geodesic_m": (
                            LOCAL_TERMINAL_RELATION_WAYPOINT_MAX_M),
                        "policy": (
                            "object-relative STOP uses short near-side hops; "
                            "region-entry STOP keeps the ordinary cap"),
                    })
            if (require_progress_after_unknown and
                    isinstance(stage_progress_carryover_active, dict) and
                    stage_progress_carryover_active.get(
                        "prior_following_landmark_visible") and
                    current_form_upper in {
                        "BETWEEN_OBJECTS", "FOLLOW_PATH_BOUNDARY",
                        "PASS_LANDMARK", "ADVANCE_STRAIGHT", "CROSS_SPACE"}):
                # The previous semantic review already saw the next clause's
                # landmark on the same/adjacent outgoing ray.  A partial
                # verdict therefore means the current boundary needs one
                # short physical clearance step, not another full-room hop
                # that consumes the next sub-instruction.  This cap is online
                # route evidence only; completion still requires a real point
                # arrival and the node/edge judge below.
                maximum_selection_geodesic_m = min(
                    float(maximum_selection_geodesic_m or 2.0), 2.0)
                selection_stage["metadata"] = dict(
                    selection_stage.get("metadata", {}) or {},
                    following_landmark_boundary_lookahead={
                        "active": True,
                        "maximum_geodesic_m": 2.0,
                        "source": "prior_high_confidence_same_route_review",
                        "policy": (
                            "clear the current endpoint without consuming the "
                            "next decomposed transition"),
                    })
            terminal_portal = bool(
                current_form_upper == "SELECT_PORTAL" and
                next_context is None and
                re.search(r"\b(?:stop|wait|end|finish)\b", " ".join(
                    str(stage.get(key, "")) for key in (
                        "navigation_instruction", "completion_cue",
                        "source_clause")).lower()))
            if terminal_portal:
                # A final doorway/threshold relation still needs its own real
                # point edge, but the residual strip can be very close when
                # the preceding clause ended at the same opening.  Keep it
                # auditable (>=0.30m) with a bounded three-metre threshold ray;
                # this is form/text structure only, never goal distance.
                minimum_selection_progress_m = min(
                    minimum_selection_progress_m, 0.30)
                maximum_selection_geodesic_m = 3.00
                selection_stage["metadata"] = dict(
                    selection_stage.get("metadata", {}) or {},
                    terminal_portal_residual_edge={
                        "active": True,
                        "minimum_geodesic_m": 0.30,
                        "maximum_geodesic_m": 3.00,
                        "policy": (
                            "final portal relation requires a distinct short "
                            "arrival edge before active STOP"),
                    })
            if turn_to_landmark_alignment_stage is not None:
                # Route setup for an orientation clause is deliberately
                # local: it should enter the outgoing lane, not consume the
                # following portal/room transition before that stage is active.
                maximum_selection_geodesic_m = (
                    named_landmark_route_setup_cap_m(
                        maximum_selection_geodesic_m, True))
            elif bare_turn_route_setup_active:
                maximum_selection_geodesic_m = (
                    named_landmark_route_setup_cap_m(
                        maximum_selection_geodesic_m, True))
            for selection_attempt in range(2):
                try:
                    selection_result = self.point_selector.select(
                        PointSelectionRequest(
                            sim=self.sim, position=position, yaw=yaw,
                            stage=selection_stage,
                            target_index=hop_index, rendered=self.rendered,
                            motion_log=self.motion_log,
                            position_history=self.position_history,
                            previous_action_history=previous_action_history,
                            back_yaw=back_yaw,
                            blocked_yaws=blocked_yaws + rejected_selection_yaws,
                            reference_path=self.reference_path,
                            reference_path_index=self.reference_path_index,
                            minimum_progress_distance_m=(
                                minimum_selection_progress_m),
                            maximum_initial_geodesic_m=(
                                maximum_selection_geodesic_m),
                            semantic_reference_rgb=(
                                exit_portal_reference_rgb
                                if exit_threshold_portal_continuation_required(
                                    stage, stage_progress_carryover_active)
                                else None),
                            allow_historical_direction_fallback=(
                                allow_historical_direction_fallback)))
                except RuntimeError as exc:
                    end_reason = f"vlm_selection_failed: {exc}"
                    selection_result = None
                    break
                candidate = (selection_result.chosen
                             if selection_result is not None else None)
                repair = ((candidate or {}).get("vlm_selection", {}) or {}).get(
                    "postselection_repair", {}) or {}
                selection_meta = ((candidate or {}).get("vlm_selection", {})
                                  or {})
                rejected_for_safety = (
                    str(repair.get("status", "")) in {
                        "no_reachable_ground_anchor",
                        "path_ray_inconsistent",
                    } or
                    str((selection_meta.get("minimum_progress_reject", {})
                         or {}).get("status", "")) ==
                    "too_near_to_progress")
                if (rejected_for_safety and candidate is not None):
                    # choose_view performs a preview turn before returning.
                    # A rejected point must not leave the simulator facing
                    # that failed ray: otherwise the retry panorama and its
                    # action-history context are silently rotated into a new
                    # branch.  Restore the exact real-node pose before any
                    # second semantic decision.
                    set_pose(self.sim, position, yaw)
                    if (str(repair.get("status", "")) ==
                            "path_ray_inconsistent" and
                            selection_attempt == 0):
                        # A visible floor patch whose shortest physical path
                        # begins in a substantially different direction is a
                        # through-wall/around-wall endpoint, not an executable
                        # point-tracking ray. Block that RGB sector and make
                        # one fresh review, but retain incoming/action-history
                        # exclusion unless the instruction explicitly says to
                        # turn around. The VLM still sees RGB/masks only;
                        # navmesh is used solely to reject the unsafe choice.
                        rejected_selection_yaws.append(float(
                            candidate.get("yaw", yaw)))
                        allow_historical_direction_fallback = (
                            rejected_ray_retry_reopens_incoming(stage))
                        continue
                    # A rejected pixel is a physical safety failure, not
                    # evidence that another semantic panorama sector is the
                    # route.  Never launch a second stochastic VLM direction
                    # decision here: that can replace a correct corridor view
                    # with an unrelated side wall and contaminate history.
                    # The caller records the failure and can retry the same
                    # node with a new explicit selection policy.
                    final_selection_rejected = True
                    break
                break
            if selection_result is None:
                break
            if final_selection_rejected:
                # Preserve the candidate set for the caller's audit, but do
                # not execute a point that failed the generic online safety
                # boundary twice.  No node/edge is created for a rejected
                # selection, so action history remains uncontaminated.
                rejected_candidate = selection_result.chosen
                selection_failures.append({
                    "hop_index": int(hop_index),
                    "stage": stage,
                    "selection": ((rejected_candidate or {}).get(
                        "vlm_selection", {}) or {}),
                    "candidate_views": self._candidate_records(
                        selection_result.candidates),
                    "reason": "postselection_safety_boundary_rejected_twice",
                })
                (self.output_dir / "selection_failures.json").write_text(
                    json.dumps(selection_failures, ensure_ascii=False, indent=2)
                    + "\n")
                # Persist the exact rejected RGB, raw ensemble mask, final
                # semantic target mask, and depth used only by the physical
                # safety boundary.  This makes failures reproducible without
                # accidentally diagnosing the previous successful hop.
                self._save_selection(
                    selection_result.candidates,
                    rejected_candidate or {}, hop_index)
                selection_result.chosen = None
                end_reason = "vlm_selection_no_safe_point_after_retry"
            chosen, candidates = selection_result.chosen, selection_result.candidates
            if chosen is None:
                end_reason = "vlm_selection_returned_no_safe_point"
                break
            self._save_selection(candidates, chosen, hop_index)
            selected_route_yaw = selected_point_ray_yaw_rad(chosen)
            selected_yaws.append(float(selected_route_yaw))
            selected_pixel_world, selected_depth = pixel_ground_to_world(
                chosen["point"], chosen["depth"], position, chosen["yaw"])
            repair_metadata = ((chosen.get("vlm_selection", {}) or {}).get(
                "postselection_repair", {}) or {})
            physical_waypoint_override = repair_metadata.get(
                "physical_waypoint_override_xyz")
            selected_world = (
                np.asarray(physical_waypoint_override, np.float32)
                if physical_waypoint_override is not None else
                selected_pixel_world)
            selected_navmesh = None
            initial_target_geodesic = math.inf
            if selected_world is not None:
                snapped = np.asarray(
                    self.sim.pathfinder.snap_point(selected_world), np.float32)
                if np.isfinite(snapped).all():
                    selected_navmesh = snapped
                    shortest = __import__("habitat_sim").ShortestPath()
                    shortest.requested_start = position
                    shortest.requested_end = selected_navmesh
                    if self.sim.pathfinder.find_path(shortest):
                        initial_target_geodesic = float(
                            shortest.geodesic_distance)
            navigation_request = PointNavigationRequest(
                    rgb=chosen["rgb"], selected_point_xy=chosen["point"],
                    selectable_mask=chosen["target_mask"],
                    ground_mask=chosen["mask"], yaw=chosen["yaw"],
                    position_history=self.position_history,
                    instruction=sub_instruction.navigation_instruction,
                    full_instruction=self.full_instruction,
                    sub_instruction=sub_instruction.navigation_instruction,
                    semantic_target=sub_instruction.semantic_spatial_target,
                    target_index=hop_index,
                    stage_count=len(self.sub_instructions),
                    global_step=global_step,
                    selected_point_depth_m=float(selected_depth),
                    selected_point_reachable=math.isfinite(
                        initial_target_geodesic),
                    selected_point_initial_geodesic_m=(
                        initial_target_geodesic
                        if math.isfinite(initial_target_geodesic) else None),
                    selected_point_navmesh_xyz=selected_navmesh,
                    # A progress-qualified target must be physically
                    # attempted even when its RGB pixel lies in the camera's
                    # near band; otherwise the executor can emit an empty
                    # edge and the semantic judge sees fabricated motion.
                    allow_initial_near_field_arrival=bool(
                        minimum_selection_progress_m <= 0.0))
            # Form-level physical budgets are generic execution policy.  A
            # stair transition and a long obstacle-side clearance need more
            # control steps than an ordinary point hop; otherwise a valid
            # upper landing/behind-side target is rejected merely because the
            # global budget was tuned for flat, short segments.  Preserve the
            # configured budget as the floor and restore it after each hop.
            base_max_steps = int(self.point_navigation_executor.max_steps)
            form_step_budget = {
                "VERTICAL_UP": 32,
                "VERTICAL_DOWN": 32,
                "CIRCUMNAVIGATE": 28,
                "TURN_TO_LANDMARK": 32,
                "TURN_LEFT": 32,
                "TURN_RIGHT": 32,
                "BETWEEN_OBJECTS": 32,
                "OTHER": 32,
                "ENTER_REGION": 32,
                "TRAVERSE_PORTAL_REGION": 32,
                # Passing a landmark is a forward continuation, not a
                # bounded doorway hop; long rooms/corridors may need several
                # crop/tracking cycles before the landmark is truly behind.
                "PASS_LANDMARK": 40,
                "ADVANCE_STRAIGHT": 32,
            }.get(str(selection_stage.get("form", "")).upper(),
                  base_max_steps)
            self.point_navigation_executor.max_steps = max(
                base_max_steps, form_step_budget)
            try:
                navigation_result = execute_point_navigation(
                    self.point_navigation_executor, navigation_request)
            finally:
                self.point_navigation_executor.max_steps = base_max_steps
            yaw = navigation_result.final_yaw
            global_step = navigation_result.next_global_step
            action_history = navigation_result.action_history
            bare_turn_preview_delta = bare_turn_preview_yaw_delta_rad(
                preselection_orientation_alignment, chosen["yaw"])
            if bare_turn_preview_delta is not None:
                # choose_view already executed and rendered this rotation in
                # Habitat before the point controller began. Persist that real
                # action at the head of the edge history so completion and the
                # independent audit observe the full commanded turn.
                action_history.insert(0, {
                    "step": -1,
                    "action": (
                        "preselection_bare_turn_left_preview"
                        if bare_turn_preview_delta > 0.0 else
                        "preselection_bare_turn_right_preview"),
                    "turn_deg": float(math.degrees(
                        bare_turn_preview_delta)),
                    "moved_m": 0.0,
                    "position_xyz": position.tolist(),
                    "yaw_rad": float(chosen["yaw"]),
                    "orientation_only": True,
                    "source_node_yaw_rad": float(
                        preselection_orientation_alignment[
                            "source_node_yaw_rad"]),
                    "target_yaw_rad": float(chosen["yaw"]),
                    "already_executed_by": "point_selector_preview_turn",
                })
            if (preselection_orientation_alignment is not None and
                    preselection_orientation_alignment.get("status") !=
                    "deferred_until_physical_arrival"):
                action_history.insert(0, {
                    "step": -1,
                    "action": (
                        "preselection_turn_left_alignment"
                        if preselection_orientation_alignment[
                            "yaw_delta_deg"] > 0.0 else
                        "preselection_turn_right_alignment"),
                    "turn_deg": preselection_orientation_alignment[
                        "yaw_delta_deg"],
                    "moved_m": 0.0,
                    "position_xyz": position.tolist(),
                    "yaw_rad": preselection_orientation_alignment[
                        "final_yaw_rad"],
                    "orientation_only": True,
                    "source_node_yaw_rad": preselection_orientation_alignment[
                        "source_node_yaw_rad"],
                    "target_yaw_rad": preselection_orientation_alignment[
                        "target_yaw_rad"],
                })
            segment_distance = sum(
                float(action.get("moved_m", 0.0)) for action in action_history)
            traveled_distance += segment_distance
            stop_position = np.asarray(
                self.sim.get_agent(0).get_state().position, np.float32)
            observations = self.sim.get_sensor_observations()
            stop_rgbs = [observations["rgb"][..., :3]] + [
                observations[f"pano_rgb_{index}"][..., :3] for index in range(1, 6)]
            stop_depths = [observations["depth"]] + [
                observations[f"pano_depth_{index}"] for index in range(1, 6)]
            stop_completion_views = (
                observe_eight_rgb(self.sim)
                if self.vlm_harness.instruction_completion_prompt_version in
                self.vlm_harness.EIGHT_VIEW_COMPLETION_PROMPT_VERSIONS else None)
            final_target_geodesic = math.inf
            if selected_navmesh is not None:
                shortest = __import__("habitat_sim").ShortestPath()
                shortest.requested_start = stop_position
                shortest.requested_end = selected_navmesh
                if self.sim.pathfinder.find_path(shortest):
                    final_target_geodesic = float(shortest.geodesic_distance)
            point_target_reference_reached = bool(
                math.isfinite(final_target_geodesic) and
                final_target_geodesic <= 0.75)
            physical_arrival = bool(
                navigation_result.arrived and point_target_reference_reached)
            physical_failure_type = None
            if navigation_result.arrived and not physical_arrival:
                physical_failure_type = "premature_offscreen_stop"
            elif (not navigation_result.arrived and
                  point_target_reference_reached):
                physical_failure_type = "missed_point_arrival_signal"
            elif not navigation_result.arrived:
                physical_failure_type = str(navigation_result.end_reason)
            post_arrival_orientation_alignment = None
            bare_turn_delta = bare_turn_delta_rad(stage)
            if physical_arrival and bare_turn_delta is not None:
                # A direction-only turn has an exact orientation boundary,
                # not a second semantic destination.  First retain the point
                # executor's normal physical-arrival contract, then align
                # relative to the verified source-node yaw.  The continuous
                # frames and the aggregate action are persisted before node
                # capture, so the judge consumes real, auditable history.
                alignment_start_yaw = float(yaw)
                source_yaw = float(
                    explicit_turn_route_frame_yaw
                    if explicit_turn_route_frame_yaw is not None else
                    (source_node.base_yaw_rad if source_node is not None
                     else alignment_start_yaw))
                target_yaw = _wrap_angle(source_yaw + bare_turn_delta)
                alignment_delta = _wrap_angle(target_yaw - alignment_start_yaw)
                alignment_steps = max(
                    1, int(math.ceil(abs(alignment_delta) /
                                     math.radians(10.0))))
                yaw, _ = continuous_turn(
                    self.sim, stop_position, alignment_start_yaw, target_yaw,
                    math.radians(10.0), self.rendered, self.motion_log,
                    hop_index, "post_arrival_turn_alignment",
                    self.video_composer, self.position_history,
                    sub_instruction.navigation_instruction,
                    reference_path=self.reference_path,
                    reference_path_index=self.reference_path_index)
                action_history.append({
                    "step": len(action_history),
                    "action": (
                        "post_arrival_turn_left_alignment"
                        if alignment_delta > 0.0
                        else "post_arrival_turn_right_alignment"),
                    "turn_deg": float(math.degrees(alignment_delta)),
                    "moved_m": 0.0,
                    "position_xyz": stop_position.tolist(),
                    "yaw_rad": float(yaw),
                    "orientation_only": True,
                    "source_node_yaw_rad": source_yaw,
                    "target_yaw_rad": target_yaw,
                })
                global_step += alignment_steps
                post_arrival_orientation_alignment = {
                    "status": "aligned_bare_turn_from_arrived_state",
                    "form": current_form_upper,
                    "start_yaw_rad": alignment_start_yaw,
                    "source_node_yaw_rad": source_yaw,
                    "target_yaw_rad": target_yaw,
                    "final_yaw_rad": float(yaw),
                    "yaw_delta_deg": float(math.degrees(alignment_delta)),
                    "absolute_source_to_final_delta_deg": float(math.degrees(
                        _wrap_angle(float(yaw) - source_yaw))),
                    "rendered_turn_frames": alignment_steps,
                    "policy": (
                        "direction-only clause: physically reach selected "
                        "ground point, align exactly relative to verified "
                        "source yaw, then capture node evidence"),
                }
                set_pose(self.sim, stop_position, yaw)
                observations = self.sim.get_sensor_observations()
                stop_rgbs = [observations["rgb"][..., :3]] + [
                    observations[f"pano_rgb_{index}"][..., :3]
                    for index in range(1, 6)]
                stop_depths = [observations["depth"]] + [
                    observations[f"pano_depth_{index}"]
                    for index in range(1, 6)]
                stop_completion_views = (
                    observe_eight_rgb(self.sim)
                    if self.vlm_harness.instruction_completion_prompt_version in
                    self.vlm_harness.EIGHT_VIEW_COMPLETION_PROMPT_VERSIONS
                    else None)
            elif (physical_arrival and
                  str(stage.get("form", "")).upper() == "TURN_TO_LANDMARK"):
                # Route selection and semantic endpoint orientation are two
                # different decisions.  For panorama-wide aliases, the first
                # VLM pass chooses the executable ordered-route ray; after the
                # real arrival, a second clean-RGB pass centers that same
                # landmark at the new position.  This prevents route motion
                # from being corrupted by object centering while still making
                # the node visually satisfy “turn toward X”.
                alignment_start_yaw = float(yaw)
                selection_review = dict(
                    ((chosen.get("vlm_selection", {}) or {}).get(
                        "view_refinement", {}) or {}))
                ambiguity_active = bool((selection_review.get(
                    "ambiguous_landmark_route_adjudication", {}) or {}).get(
                        "active"))
                current_landmark_alignment = None
                if ambiguity_active and stop_completion_views is not None:
                    current_landmark_alignment = (
                        self.vlm_harness.select_turn_landmark_alignment(
                            sub_instruction, stop_completion_views,
                            selection_review))
                    target_yaw = _wrap_angle(
                        alignment_start_yaw + math.radians(float(
                            current_landmark_alignment[
                                "relative_yaw_deg"])))
                else:
                    target_yaw = float(chosen["yaw"])
                alignment_delta = _wrap_angle(target_yaw - alignment_start_yaw)
                alignment_steps = max(
                    1, int(math.ceil(abs(alignment_delta) /
                                     math.radians(10.0))))
                yaw, _ = continuous_turn(
                    self.sim, stop_position, alignment_start_yaw, target_yaw,
                    math.radians(10.0), self.rendered, self.motion_log,
                    hop_index, "post_arrival_named_landmark_alignment",
                    self.video_composer, self.position_history,
                    sub_instruction.navigation_instruction,
                    reference_path=self.reference_path,
                    reference_path_index=self.reference_path_index)
                action_history.append({
                    "step": len(action_history),
                    "action": "post_arrival_named_landmark_alignment",
                    "turn_deg": float(math.degrees(alignment_delta)),
                    "moved_m": 0.0,
                    "position_xyz": stop_position.tolist(),
                    "yaw_rad": float(yaw),
                    "orientation_only": True,
                    "committed_selection_yaw_rad": target_yaw,
                    "current_landmark_alignment": (
                        current_landmark_alignment),
                })
                global_step += alignment_steps
                post_arrival_orientation_alignment = {
                    "status": (
                        "aligned_to_current_rgb_landmark"
                        if current_landmark_alignment is not None else
                        "aligned_to_committed_selection_bearing"),
                    "start_yaw_rad": alignment_start_yaw,
                    "final_yaw_rad": float(yaw),
                    "yaw_delta_deg": float(math.degrees(alignment_delta)),
                    "committed_selection_yaw_rad": target_yaw,
                    "route_selection_yaw_rad": float(chosen["yaw"]),
                    "current_landmark_alignment": (
                        current_landmark_alignment),
                    "policy": (
                        "after real route-point arrival, center the same "
                        "landmark from current-node clean RGB; otherwise "
                        "reuse the frozen semantic bearing"),
                }
                # Capture all node evidence after the orientation-only phase.
                # Position is restored explicitly as a safety invariant.
                set_pose(self.sim, stop_position, yaw)
                observations = self.sim.get_sensor_observations()
                stop_rgbs = [observations["rgb"][..., :3]] + [
                    observations[f"pano_rgb_{index}"][..., :3]
                    for index in range(1, 6)]
                stop_depths = [observations["depth"]] + [
                    observations[f"pano_depth_{index}"]
                    for index in range(1, 6)]
                stop_completion_views = (
                    observe_eight_rgb(self.sim)
                    if self.vlm_harness.instruction_completion_prompt_version in
                    self.vlm_harness.EIGHT_VIEW_COMPLETION_PROMPT_VERSIONS
                    else None)
            elif (physical_arrival and stop_completion_views is not None and
                  terminal_facing_landmark(stage) is not None):
                # The point edge establishes the requested place. For an
                # explicit final-facing clause, acquire the named landmark
                # again from the arrived node before persisting/judging it.
                # This orientation-only call receives clean RGB, never depth,
                # navmesh state, reference paths, or outcome labels.
                facing_landmark = terminal_facing_landmark(stage)
                alignment_stage = dict(stage)
                alignment_stage["landmark"] = facing_landmark
                alignment_start_yaw = float(yaw)
                current_landmark_alignment = (
                    self.vlm_harness.select_turn_landmark_alignment(
                        alignment_stage, stop_completion_views,
                        ((chosen.get("vlm_selection", {}) or {}).get(
                            "view_refinement", {}) or {}),
                        alignment_kind="terminal_facing"))
                target_yaw = _wrap_angle(
                    alignment_start_yaw + math.radians(float(
                        current_landmark_alignment["relative_yaw_deg"])))
                alignment_delta = _wrap_angle(target_yaw - alignment_start_yaw)
                alignment_steps = max(
                    1, int(math.ceil(abs(alignment_delta) /
                                     math.radians(10.0))))
                yaw, _ = continuous_turn(
                    self.sim, stop_position, alignment_start_yaw, target_yaw,
                    math.radians(10.0), self.rendered, self.motion_log,
                    hop_index, "post_arrival_terminal_facing_alignment",
                    self.video_composer, self.position_history,
                    sub_instruction.navigation_instruction,
                    reference_path=self.reference_path,
                    reference_path_index=self.reference_path_index)
                action_history.append({
                    "step": len(action_history),
                    "action": "post_arrival_terminal_facing_alignment",
                    "turn_deg": float(math.degrees(alignment_delta)),
                    "moved_m": 0.0,
                    "position_xyz": stop_position.tolist(),
                    "yaw_rad": float(yaw),
                    "orientation_only": True,
                    "target_yaw_rad": target_yaw,
                    "facing_landmark": facing_landmark,
                    "current_landmark_alignment": current_landmark_alignment,
                })
                global_step += alignment_steps
                post_arrival_orientation_alignment = {
                    "status": "aligned_terminal_facing_landmark",
                    "start_yaw_rad": alignment_start_yaw,
                    "final_yaw_rad": float(yaw),
                    "yaw_delta_deg": float(math.degrees(alignment_delta)),
                    "committed_selection_yaw_rad": target_yaw,
                    "facing_landmark": facing_landmark,
                    "current_landmark_alignment": current_landmark_alignment,
                    "policy": (
                        "explicit final-facing clause uses arrived-node clean "
                        "RGB only; spatial point selection is unchanged"),
                }
                set_pose(self.sim, stop_position, yaw)
                observations = self.sim.get_sensor_observations()
                stop_rgbs = [observations["rgb"][..., :3]] + [
                    observations[f"pano_rgb_{index}"][..., :3]
                    for index in range(1, 6)]
                stop_depths = [observations["depth"]] + [
                    observations[f"pano_depth_{index}"]
                    for index in range(1, 6)]
                stop_completion_views = observe_eight_rgb(self.sim)
            arrival_dir = self.output_dir / "arrival_six_views"
            arrival_depth_dir = self.output_dir / "arrival_depths"
            arrival_dir.mkdir(exist_ok=True)
            arrival_depth_dir.mkdir(exist_ok=True)
            for view_index, (rgb, depth) in enumerate(
                    zip(stop_rgbs, stop_depths)):
                Image.fromarray(rgb).save(
                    arrival_dir /
                    f"hop_{hop_index:03d}_view_{view_index}.jpg")
                np.save(
                    arrival_depth_dir /
                    f"hop_{hop_index:03d}_view_{view_index}.npy",
                    np.asarray(depth, np.float32))
            # A node represents a verified point-navigation arrival.  A
            # failed executor attempt may still leave the agent displaced, but
            # it is not a valid instruction boundary and must never pollute
            # the graph or completion history.  Keep the failed attempt in the
            # round record and retry/reselect from the last real node.
            stop_node = None
            stop_edge = None
            stage_progress_carryover_used = stage_progress_carryover_active
            # The executor's cluster/off-screen signal is only a proposal for
            # arrival.  A graph node and semantic boundary require the
            # independent physical endpoint check as well.  In particular,
            # never turn a premature off-screen signal into a node: doing so
            # would contaminate the incoming action history and make the next
            # sub-instruction appear to start from a fabricated location.
            if physical_arrival:
                stop_node, stop_edge = self.graph_memory.add_navigation_stop_node(
                    position_xyz=stop_position, base_yaw_rad=yaw,
                    global_step=global_step, six_views=stop_rgbs,
                    six_depths=stop_depths, sub_instruction=sub_instruction,
                    action_history=action_history,
                    arrival_signal=navigation_result.signal,
                    completion_views=stop_completion_views,
                    metadata={
                        "strategy": self.mode, "exploration_hop": hop_index,
                        "point_target_arrived": True,
                        "point_target_reference_reached": (
                            point_target_reference_reached),
                        "expected_sub_instruction_id": (
                            self.state.expected_sub_instruction_id),
                    },
                    edge_kind="instruction_sequence_exploration",
                    edge_metadata={
                        "selected_yaw_rad": float(chosen["yaw"]),
                        "selected_point_ray_yaw_rad": float(
                            selected_route_yaw),
                        "point_selection_review": dict(
                            ((chosen.get("vlm_selection", {}) or {}).get(
                                "view_refinement", {}) or {}),
                            direction_gate=dict(
                                ((chosen.get("vlm_selection", {}) or {}).get(
                                    "direction_gate", {}) or {})),
                            selected_view=(
                                (chosen.get("vlm_selection", {}) or {}).get(
                                    "selected_view")),
                            postselection_repair=dict(
                                ((chosen.get("vlm_selection", {}) or {}).get(
                                    "postselection_repair", {}) or {})),
                        ),
                        "blocked_yaws_rad": blocked_yaws,
                        "turn_carryover": turn_carryover_active,
                        "stage_progress_carryover": (
                            stage_progress_carryover_used),
                        "post_arrival_orientation_alignment": (
                            post_arrival_orientation_alignment),
                        "preselection_orientation_alignment": (
                            preselection_orientation_alignment),
                        "edge_keyframes": navigation_result.record.get(
                            "edge_keyframes", []),
                    })
            if physical_arrival:
                completion = self.completion_judge.judge(
                    stop_node, self.state.expected_sub_instruction_id,
                    (stop_completion_views
                     if stop_completion_views is not None else stop_rgbs),
                    navigation_result.edge_keyframes).to_dict()
            else:
                completion = {
                    "status": UNKNOWN,
                    "instruction_completed": False,
                    "expected_sub_instruction_id": int(
                        self.state.expected_sub_instruction_id),
                    "confidence": 1.0,
                    "reason": (
                        "point executor did not declare arrival at its selected "
                        "point; instruction completion was not queried"),
                    "visual_evidence": "",
                    "previous_node_id": (
                        self.graph_memory.nodes[-1].node_id
                        if self.graph_memory.nodes else None),
                    "current_node_id": None,
                    "incoming_edge_id": None,
                }
            # Compatibility boundary for the existing outer exploration state
            # machine.  Only an arrived node can be classified or advance the
            # sequence; failed point attempts remain retryable diagnostics.
            chained_stop_wait_completion = None
            chained_stop_wait_classification = None
            chained_stop_wait_directive = None
            if physical_arrival:
                classification = {
                    "node_id": stop_node.node_id,
                    "belongs_to_sequence": bool(
                        completion["instruction_completed"]),
                    "matched_sub_instruction_id": (
                        int(self.state.expected_sub_instruction_id)
                        if completion["instruction_completed"] else -1),
                    "expected_sub_instruction_id": int(
                        self.state.expected_sub_instruction_id),
                    "expected_sequence_position": bool(
                        completion["instruction_completed"]),
                    "confidence": float(completion["confidence"]),
                    "reason": completion["reason"],
                    "visual_evidence": completion["visual_evidence"],
                    "compatibility_source": "instruction_completion",
                    "active_form": str(stage.get("form", "")).upper(),
                    "portal_endpoint_side_veto_applied": bool(
                        (completion.get("validation_overrides") or {}).get(
                            "exit_endpoint_side_veto_applied")),
                    "unknown_disposition": infer_unknown_disposition(
                        completion, action_history),
                }
                progress_lookahead_was_active = bool(
                    require_progress_after_unknown)
                directive = self.state.observe(
                    stop_node.node_id, classification,
                    float(selected_route_yaw))
                initial_sequence_directive = directive
                next_requires_progress = bool(
                    classification.get("unknown_disposition") == "on_route")
                if (next_requires_progress and
                        not progress_lookahead_was_active and
                        directive.action == "continue_current_instruction"):
                    moved_m = float(sum(
                        float(item.get("moved_m", 0.0) or 0.0)
                        for item in action_history))
                    net_turn_deg = float(sum(
                        float(item.get("turn_deg", 0.0) or 0.0)
                        for item in action_history))
                    source_position = np.asarray(position, np.float64)
                    endpoint_position = np.asarray(stop_position, np.float64)
                    displacement = endpoint_position - source_position
                    source_graph_node = self.graph_memory.get_node(
                        completion.get("previous_node_id"))
                    prior_node_heading_delta_deg = None
                    if source_graph_node is not None and stop_node is not None:
                        heading_delta = (
                            float(stop_node.base_yaw_rad) -
                            float(source_graph_node.base_yaw_rad) + math.pi
                        ) % (2.0 * math.pi) - math.pi
                        prior_node_heading_delta_deg = round(
                            math.degrees(heading_delta), 3)
                    chosen_review = (((chosen.get("vlm_selection", {}) or {}).get(
                        "view_refinement", {}) or {}))
                    following_view_index = chosen_review.get(
                        "first_following_landmark_view_index", -1)
                    try:
                        following_view_visible = bool(
                            following_view_index is not None and
                            int(following_view_index) >= 0)
                    except (TypeError, ValueError):
                        following_view_visible = False
                    stage_progress_carryover_active = {
                        "active": True,
                        "stage_id": int(self.state.expected_sub_instruction_id),
                        "source_node_id": completion.get("previous_node_id"),
                        "partial_node_id": completion.get("current_node_id"),
                        "prior_edge_id": completion.get("incoming_edge_id"),
                        "prior_selected_yaw_rad": float(selected_route_yaw),
                        "prior_edge_traveled_distance_m": round(moved_m, 4),
                        "prior_edge_net_turn_deg": round(net_turn_deg, 3),
                        "prior_node_heading_delta_deg": (
                            prior_node_heading_delta_deg),
                        "prior_edge_displacement_xz_m": [
                            round(float(displacement[0]), 4),
                            round(float(displacement[2]), 4),
                        ],
                        "prior_endpoint_evidence": completion.get(
                            "endpoint_evidence"),
                        "prior_temporal_evidence": completion.get(
                            "temporal_evidence"),
                        "prior_motion_evidence": completion.get(
                            "motion_evidence"),
                        "prior_exit_endpoint_side_veto_applied": bool(
                            (completion.get("validation_overrides") or {}).get(
                                "exit_endpoint_side_veto_applied")),
                        "prior_exit_endpoint_side_audit": completion.get(
                            "exit_endpoint_side_audit"),
                        "prior_following_landmark_visible": bool(
                            following_view_visible and
                            str(chosen_review.get(
                                "sequence_alignment", "")).lower() in {
                                        "same_view", "adjacent_view"} and
                            float(chosen_review.get(
                                "confidence", 0.0) or 0.0) >= 0.80),
                        "prior_following_landmark": str(
                            chosen_review.get("first_following_landmark", "")),
                        "prior_sequence_alignment": str(
                            chosen_review.get("sequence_alignment", "")),
                        "policy": (
                            "one supported-partial real edge is carried into "
                            "the single permitted lookahead edge"),
                    }
                    if (stage_progress_carryover_active.get(
                            "prior_exit_endpoint_side_veto_applied") and
                            str(stage.get("form", "")).upper() in {
                                "EXIT_REGION", "ENTER_REGION",
                                "SELECT_PORTAL", "TRAVERSE_PORTAL_REGION"} and
                            exit_portal_reference_rgb is None):
                        exit_portal_reference_rgb = np.asarray(
                            chosen["rgb"], np.uint8).copy()
                        exit_portal_reference_stage_id = current_stage_id
                elif (completion.get("instruction_completed") or
                      directive.action in {"backtrack_and_block", "complete",
                                           "advance_sequence"}):
                    stage_progress_carryover_active = None
                    if directive.action == "backtrack_and_block":
                        exit_portal_reference_rgb = None
                        exit_portal_reference_stage_id = None
                require_progress_after_unknown = next_requires_progress
            else:
                source_node_id = (self.graph_memory.nodes[-1].node_id
                                  if self.graph_memory.nodes else None)
                classification = {
                    "node_id": None,
                    "belongs_to_sequence": False,
                    "matched_sub_instruction_id": -1,
                    "expected_sub_instruction_id": int(
                        self.state.expected_sub_instruction_id),
                    "expected_sequence_position": False,
                    "confidence": 1.0,
                    "reason": completion["reason"],
                    "visual_evidence": "",
                    "compatibility_source": "point_navigation_failure",
                    "unknown_disposition": "wrong",
                }
                directive = SequenceDirective(
                    action="retry_current_instruction",
                    expected_sub_instruction_id=(
                        self.state.expected_sub_instruction_id),
                    matched_sub_instruction_id=-1,
                    node_id=str(source_node_id or ""),
                    reason=("point navigation did not arrive; no node or edge "
                            "created, retry selection from last verified node"),
                )
                initial_sequence_directive = directive
            if (physical_arrival and chained_stop_wait_eligible(
                    directive, self.state.active_sub_instruction,
                    is_final_sub_instruction=(
                        self.state.cursor ==
                        len(self.state.sub_instructions) - 1))):
                # A clause such as "walk between the chairs; stop at the bar
                # corner" can have one physical endpoint.  Reuse the same
                # arrived node and incoming edge to judge the immediately
                # following STOP/WAIT relation.  No synthetic node, motion,
                # or automatic completion is created: the ordinary semantic
                # judge must independently return completed.
                chained_expected = int(
                    self.state.expected_sub_instruction_id)
                chained_stop_wait_completion = self.completion_judge.judge(
                    stop_node, chained_expected,
                    (stop_completion_views
                     if stop_completion_views is not None else stop_rgbs),
                    navigation_result.edge_keyframes).to_dict()
                chained_previous_node = self.graph_memory.get_node(
                    chained_stop_wait_completion.get("previous_node_id"))
                chained_near_supported = chained_near_stop_evidence_supported(
                        self.state.active_sub_instruction,
                        chained_stop_wait_completion,
                        ((chained_previous_node.environment_semantics
                          if chained_previous_node is not None else {})),
                        stop_node.environment_semantics,
                        action_history)
                if chained_near_supported:
                    chained_stop_wait_completion = dict(
                        chained_stop_wait_completion,
                        status="completed",
                        instruction_completed=True,
                        confidence=max(
                            0.80, float(chained_stop_wait_completion.get(
                                "confidence", 0.0) or 0.0)),
                        chained_near_stop_structured_override=True,
                        chained_near_stop_structured_rule=(
                            "plain near/beside relation with ordered real "
                            "arrival edge and same-class landmark area growth"),
                    )
                if (chained_stop_wait_completion.get(
                        "instruction_completed") and
                        float(chained_stop_wait_completion.get(
                            "confidence", 0.0)) >=
                        self.state.minimum_classification_confidence):
                    chained_stop_wait_classification = {
                        "node_id": stop_node.node_id,
                        "belongs_to_sequence": True,
                        "matched_sub_instruction_id": chained_expected,
                        "expected_sub_instruction_id": chained_expected,
                        "expected_sequence_position": True,
                        "confidence": float(
                            chained_stop_wait_completion["confidence"]),
                        "reason": chained_stop_wait_completion["reason"],
                        "visual_evidence": chained_stop_wait_completion[
                            "visual_evidence"],
                        "compatibility_source": (
                            "same_arrival_edge_chained_stop_wait_completion"),
                        "unknown_disposition": None,
                    }
                    chained_stop_wait_directive = self.state.observe(
                        stop_node.node_id,
                        chained_stop_wait_classification,
                        float(selected_route_yaw))
                    directive = chained_stop_wait_directive
                    require_progress_after_unknown = False
            # A failed physical point attempt did not create a valid new
            # instruction-progress boundary.  Preserve the prior on-route
            # requirement so the next selection cannot silently fall back to
            # a near-zero duplicate point at the same node.
            if stop_node is not None:
                self.graph_memory.set_node_metadata(stop_node.node_id, {
                    "instruction_completion": completion,
                    "sub_instruction_sequence_classification": classification,
                    "sequence_directive": directive.to_dict(),
                    "chained_stop_wait_completion": (
                        chained_stop_wait_completion),
                    "chained_stop_wait_classification": (
                        chained_stop_wait_classification),
                })
            record = {
                "target_index": hop_index,
                "strategy": self.mode,
                "sub_instruction": sub_instruction.to_dict(),
                "instruction_stage": stage,
                "selection_stage": selection_stage,
                "turn_carryover": turn_carryover_active,
                "stage_progress_carryover_used": (
                    stage_progress_carryover_used),
                "stage_progress_carryover_next": (
                    stage_progress_carryover_active),
                "post_arrival_orientation_alignment": (
                    post_arrival_orientation_alignment),
                "preselection_orientation_alignment": (
                    preselection_orientation_alignment),
                "blocked_yaws_before_selection": blocked_yaws,
                "selected_yaw_rad": float(chosen["yaw"]),
                "selected_point_ray_yaw_rad": float(selected_route_yaw),
                "selected_point_xy": np.asarray(chosen["point"]).tolist(),
                "selected_point_depth_m": float(selected_depth),
                "selected_point_world_xyz": (
                    np.asarray(selected_world).tolist()
                    if selected_world is not None else None),
                "selected_pixel_projected_world_xyz": (
                    np.asarray(selected_pixel_world).tolist()
                    if selected_pixel_world is not None else None),
                "physical_waypoint_override_used": bool(
                    physical_waypoint_override is not None),
                "selected_navmesh_target_xyz": (
                    selected_navmesh.tolist()
                    if selected_navmesh is not None else None),
                "initial_target_geodesic_distance_m": (
                    initial_target_geodesic
                    if math.isfinite(initial_target_geodesic) else None),
                "final_target_geodesic_distance_m": (
                    final_target_geodesic
                    if math.isfinite(final_target_geodesic) else None),
                "navigation_physical_arrival": physical_arrival,
                "navigation_physical_arrival_threshold_m": 0.75,
                "navigation_physical_failure_type": physical_failure_type,
                "point_target_arrived": bool(navigation_result.arrived),
                "point_target_reference_reached": (
                    point_target_reference_reached),
                "point_arrival_judgment": {
                    "owner": "PointNavigationExecutor",
                    "signal": navigation_result.signal,
                    "cluster_profile": navigation_result.record.get(
                        "tracking_cluster_profile"),
                    "executor_declared_arrival": bool(
                        navigation_result.arrived),
                    "reference_final_geodesic_within_threshold": (
                        point_target_reference_reached),
                    "reference_threshold_m": 0.75,
                },
                "selection": chosen["vlm_selection"],
                "candidate_views": self._candidate_records(candidates),
                "executor": "PointNavigationExecutor",
                "arrived": navigation_result.arrived,
                **navigation_result.record,
                "action_history": action_history,
                "navigation_graph_node_id": (
                    stop_node.node_id if stop_node is not None else None),
                "navigation_graph_edge_id": (
                    stop_edge.edge_id if stop_edge is not None else None),
                "node_created_after_point_arrival": bool(
                    physical_arrival and stop_node is not None),
                "sub_instruction_sequence_classification": classification,
                "instruction_completion": completion,
                "initial_sequence_directive": (
                    initial_sequence_directive.to_dict()),
                "chained_stop_wait_completion": (
                    chained_stop_wait_completion),
                "chained_stop_wait_classification": (
                    chained_stop_wait_classification),
                "chained_stop_wait_directive": (
                    chained_stop_wait_directive.to_dict()
                    if chained_stop_wait_directive is not None else None),
                "sequence_directive": directive.to_dict(),
                "sub_instruction_satisfied": bool(
                    completion["instruction_completed"]),
            }
            records.append(record)
            previous_action_history = action_history
            previous_hop_origin = position

            # A failed physical attempt can still displace the simulator
            # agent.  Never reselect from that unverified pose: return to the
            # latest verified node through the real backtracking selector and
            # point executor, without persisting a transient node.  This keeps
            # the next selection's incoming direction/action history causal.
            if not physical_arrival:
                verified_node_id = self.state.last_verified_node_id
                recovery = self.backtracker.recover_live_to_verified_node(
                    target_node_id=verified_node_id, current_yaw=yaw,
                    global_step=global_step,
                    target_index_offset=len(self.sub_instructions) + hop_index,
                    max_attempts=self.backtracker.max_attempts_per_hop)
                recovery_payload = recovery.to_dict()
                record["physical_failure_recovery"] = recovery_payload
                recovery_records.append(recovery_payload)
                if not recovery.success:
                    end_reason = "physical_failure_recovery_failed"
                    break
                blocked_at_node = self.state.block_failed_physical_direction(
                    float(selected_route_yaw))
                record["physical_failure_direction_block"] = {
                    "node_id": blocked_at_node,
                    "yaw_rad": float(selected_route_yaw),
                    "reason": (
                        "point navigation failed and the live pose was "
                        "physically recovered; do not retry the same ray"),
                }
                yaw = recovery.final_yaw
                global_step = recovery.next_global_step
                traveled_distance += recovery.traveled_distance_m
                # Re-establish the verified node's original incoming context,
                # not the diagnostic recovery edge.  This keeps the previous
                # direction blocked on the next semantic selection and
                # preserves the action history that actually brought the
                # agent to the verified node.
                predecessor, incoming_edge = self.graph_memory.predecessor(
                    verified_node_id)
                previous_hop_origin = (
                    np.asarray(predecessor.position_xyz, np.float32)
                    if predecessor is not None else None)
                previous_action_history = (
                    list(incoming_edge.action_history)
                    if incoming_edge is not None else [])
                require_progress_after_unknown = False
                # Retry the same sub-instruction from the verified node.  The
                # failed point attempt remains diagnostic-only and cannot
                # advance the sequence or invoke the completion judge.
                continue

            if directive.action == "backtrack_and_block":
                recovery = self.backtracker.backtrack(
                    target_node_id=directive.backtrack_target_node_id,
                    source_node_id=stop_node.node_id,
                    current_yaw=yaw, global_step=global_step,
                    target_index_offset=len(self.sub_instructions) + hop_index)
                recovery_records.append(recovery.to_dict())
                record["sequence_recovery_backtrack"] = recovery.to_dict()
                yaw = recovery.final_yaw
                global_step = recovery.next_global_step
                traveled_distance += recovery.traveled_distance_m
                recovered_node_id = (
                    self.graph_memory.nodes[-1].node_id
                    if recovery.success and self.graph_memory.nodes else None)
                self.state.on_backtrack(
                    recovery.success, current_node_id=recovered_node_id)
                previous_hop_origin = None
                previous_action_history = (
                    recovery.attempts[-1]["action_history"]
                    if recovery.attempts else [])
                require_progress_after_unknown = False
                if not recovery.success:
                    end_reason = "sequence_recovery_backtrack_failed"
                    break
            elif directive.action == "complete":
                end_reason = "instruction_sequence_complete"
                break

        success = self.state.complete
        return InstructionSequenceExplorationResult(
            success=success, end_reason=end_reason,
            completed_sub_instructions=self.state.cursor,
            expected_sub_instruction_id=self.state.expected_sub_instruction_id,
            exploration_hops=len(records),
            recovery_backtracks=self.state.recovery_backtracks,
            final_yaw=yaw, next_global_step=global_step,
            traveled_distance_m=traveled_distance,
            selected_yaws_rad=selected_yaws, records=records,
            recovery_records=recovery_records, state=self.state.snapshot())
