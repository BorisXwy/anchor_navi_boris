#!/usr/bin/env python3
"""Independently verify completed stage boundaries after a navigation run.

Online completion outputs are used only to identify which real node/edge needs
auditing; their status, confidence, and prose are never shown to this verifier.
The reference trajectory is loaded post-run and supplies a separate <=30 degree
forward-progress gate.  Nothing written here is available to online navigation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from audit_active_stop_round import _load_dataset, audit_episode
from postrun_hidden_geometry import load_hidden_geometry
from vlm_harness import DeepSeekBackend


VERIFY_SCHEMA = {
    "type": "object",
    "required": [
        "semantic_completion_verified", "ordered_stage_boundary_verified",
        "confidence", "reason", "visual_evidence",
    ],
    "properties": {
        "semantic_completion_verified": {"type": "boolean"},
        "ordered_stage_boundary_verified": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string", "maxLength": 500},
        "visual_evidence": {"type": "string", "maxLength": 500},
    },
}


def _completion_candidates(target):
    values = []
    for field in ("instruction_completion", "chained_stop_wait_completion"):
        completion = target.get(field)
        if not isinstance(completion, dict):
            continue
        if not (completion.get("status") == "completed" and
                completion.get("instruction_completed")):
            continue
        try:
            stage_id = int(completion["expected_sub_instruction_id"])
        except (KeyError, TypeError, ValueError):
            continue
        values.append((stage_id, field, completion))
    return values


def _real_edge(target):
    """Whether a target produced a genuine navigation-graph node and edge.

    Legacy strategies stamp ``navigation_physical_arrival`` from privileged
    online geometry and it stays authoritative when present.  ``rgb_only_v1``
    strategies cannot write it by contract, so for them a policy-declared
    arrival with a persisted node/edge is the real-edge signal and geometric
    correctness is checked post-run by the hidden-GT gate instead.
    """
    if not (target.get("point_target_arrived", target.get("arrived", False))
            and target.get("node_created_after_point_arrival")
            and target.get("navigation_graph_node_id")
            and target.get("navigation_graph_edge_id")):
        return False
    physical = target.get("navigation_physical_arrival")
    if physical is not None:
        return bool(physical)
    return target.get("policy_input_contract") == "rgb_only_v1"


def system_all_judged_edges(trajectory):
    """Every real-edge judgment (completed or unknown), one record per target.

    ``online_status``/``online_confidence`` are for output labelling only and
    must never reach the verifier prompt.
    """
    records = []
    for target in trajectory.get("targets", []):
        if not _real_edge(target):
            continue
        for field in ("instruction_completion", "chained_stop_wait_completion"):
            completion = target.get(field)
            if not isinstance(completion, dict):
                continue
            if completion.get("status") not in ("completed", "unknown"):
                continue
            try:
                stage_id = int(completion["expected_sub_instruction_id"])
            except (KeyError, TypeError, ValueError):
                continue
            records.append({
                "target": target, "field": field, "completion": completion,
                "stage_targets": [target], "stage_id": stage_id,
                "online_status": completion.get("status"),
                "online_confidence": completion.get("confidence"),
            })
    return records


def system_stage_completion_candidates(trajectory):
    """Return the first ordered real-edge completion candidate per stage."""
    result = {}
    real_attempts = {}
    for target in trajectory.get("targets", []):
        if not _real_edge(target):
            continue
        active = target.get("sub_instruction") or target.get(
            "instruction_stage") or {}
        try:
            active_stage_id = int(active.get(
                "sub_instruction_id", active.get("stage_id")))
        except (TypeError, ValueError):
            active_stage_id = None
        if active_stage_id is not None:
            real_attempts.setdefault(active_stage_id, []).append(target)
        for stage_id, field, completion in _completion_candidates(target):
            result.setdefault(stage_id, {
                "target": target, "field": field, "completion": completion,
                # A stage may deliberately consume several arrived point
                # edges before its terminal relation becomes visible.
                "stage_targets": list(real_attempts.get(stage_id, [target])),
            })
    return result


def _read_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _contact_sheet(paths, columns=3, tile_wh=(256, 192), labels=None):
    images = []
    for index, path in enumerate(paths):
        if Path(path).is_file():
            image = _read_rgb(path)
            tile = cv2.resize(image, tile_wh, interpolation=cv2.INTER_AREA)
            if labels is not None and index < len(labels):
                label = str(labels[index])
                cv2.rectangle(tile, (0, 0),
                              (min(tile_wh[0] - 1, 12 + 9 * len(label)), 24),
                              (0, 0, 0), -1)
                cv2.putText(tile, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, (255, 255, 255), 1, cv2.LINE_AA)
            images.append(tile)
    if not images:
        return np.zeros((tile_wh[1], tile_wh[0], 3), np.uint8)
    rows = int(math.ceil(len(images) / columns))
    images.extend(np.zeros_like(images[0])
                  for _ in range(rows * columns - len(images)))
    return np.concatenate([
        np.concatenate(images[row * columns:(row + 1) * columns], axis=1)
        for row in range(rows)
    ], axis=0)


def _node_view_paths(graph_root, node):
    node = node or {}
    completion_panorama = (node.get("metadata", {}) or {}).get(
        "instruction_completion_panorama", {}) or {}
    completion_views = completion_panorama.get("views", []) or []
    records = completion_views or node.get("six_views", [])
    return [graph_root / item["image_path"] for item in records]


def _panorama_labels(prefix, count):
    bearings = [
        "FRONT", "FRONT-LEFT", "REAR-LEFT", "REAR",
        "REAR-RIGHT", "FRONT-RIGHT",
    ]
    if count == 8:
        bearings = [
            "FRONT", "FRONT-LEFT", "LEFT", "REAR-LEFT", "REAR",
            "REAR-RIGHT", "RIGHT", "FRONT-RIGHT",
        ]
    return [f"{prefix} {bearings[index]}"
            if index < len(bearings) else f"{prefix} VIEW {index + 1}"
            for index in range(count)]


def _target_current_node_id(stage_target):
    for field in ("instruction_completion", "chained_stop_wait_completion"):
        completion = stage_target.get(field)
        if isinstance(completion, dict) and completion.get("current_node_id"):
            return completion["current_node_id"]
    return stage_target.get("navigation_graph_node_id")


def _stage_target_edge_keyframes(stage_target, edges_by_id):
    """rgb_only_v1 targets keep keyframes on the graph edge, not the target."""
    records = stage_target.get("edge_keyframes")
    if records:
        return list(records)
    edge = edges_by_id.get(stage_target.get("navigation_graph_edge_id"))
    if edge is None:
        return []
    return list((edge.get("metadata") or {}).get("edge_keyframes") or [])


def _evidence_images(episode_dir, trajectory, candidate):
    target = candidate["target"]
    completion = candidate["completion"]
    stage_targets = list(candidate.get("stage_targets") or [target])
    graph_path = Path(trajectory["navigation_graph_memory"]["graph_path"])
    graph = json.loads(graph_path.read_text())
    nodes = {item["node_id"]: item for item in graph.get("nodes", [])}
    edges_by_id = {item["edge_id"]: item for item in graph.get("edges", [])}
    graph_root = graph_path.parent
    first_completion = stage_targets[0].get("instruction_completion") or {}
    previous = nodes.get(
        first_completion.get("previous_node_id") or
        completion.get("previous_node_id"))
    current = nodes.get(completion.get("current_node_id"))
    previous_paths = _node_view_paths(graph_root, previous)
    current_paths = _node_view_paths(graph_root, current)
    images = [_contact_sheet(
        previous_paths,
        labels=_panorama_labels("START", len(previous_paths)))]
    evidence_sequence = [
        "IMAGE 1: stage-start node panorama, with bearing labels"]
    ordered_paths = list(previous_paths)
    intermediate_semantics = []
    for edge_ordinal, stage_target in enumerate(stage_targets, start=1):
        edge_keyframe_records = [
            item for item in _stage_target_edge_keyframes(
                stage_target, edges_by_id)
            if item.get("image_path")]
        edge_keyframe_paths = [
            episode_dir / item["image_path"] for item in edge_keyframe_records]
        images.append(_contact_sheet(
            edge_keyframe_paths, columns=3, labels=[
                f"EDGE {edge_ordinal} TIME {index + 1}/{len(edge_keyframe_paths)}"
                + (f" {item['phase']}" if item.get("phase") == "turn_start"
                   else "")
                for index, item in enumerate(edge_keyframe_records)]))
        ordered_paths.extend(edge_keyframe_paths)
        evidence_sequence.append(
            f"IMAGE {len(images)}: chronological keyframes for point edge "
            f"{edge_ordinal} of {len(stage_targets)}")
        if edge_ordinal < len(stage_targets):
            intermediate = nodes.get(_target_current_node_id(stage_target))
            intermediate_paths = _node_view_paths(graph_root, intermediate)
            images.append(_contact_sheet(
                intermediate_paths,
                labels=_panorama_labels(
                    f"MID {edge_ordinal}", len(intermediate_paths))))
            ordered_paths.extend(intermediate_paths)
            evidence_sequence.append(
                f"IMAGE {len(images)}: intermediate arrived-node panorama "
                f"after point edge {edge_ordinal}, with bearing labels")
            intermediate_semantics.append({
                "after_point_edge": edge_ordinal,
                "environment_semantics": (
                    (intermediate or {}).get("environment_semantics", {})),
            })
    images.append(_contact_sheet(
        current_paths,
        labels=_panorama_labels("CURRENT", len(current_paths))))
    ordered_paths.extend(current_paths)
    evidence_sequence.append(
        f"IMAGE {len(images)}: final current-node panorama, with bearing labels")
    evidence_paths = [str(path) for path in ordered_paths if path.is_file()]
    node_observations = {
        "previous_node_environment_semantics": (
            (previous or {}).get("environment_semantics", {})),
        "current_node_environment_semantics": (
            (current or {}).get("environment_semantics", {})),
        "reliability_note": (
            "DINO+SAM labels, mask areas and median ranges are fallible "
            "saved observations; confirm identity and relation against RGB"),
        "stage_span_target_indices": [
            int(item.get("target_index", -1)) for item in stage_targets],
        "intermediate_node_environment_semantics": intermediate_semantics,
        "evidence_image_sequence": evidence_sequence,
    }
    return images, evidence_paths, node_observations


def _motion_summary(target, stage_targets=None):
    targets = list(stage_targets or [target])
    actions = [action for item in targets
               for action in item.get("action_history", [])]
    positions = [item.get("position_xyz") for item in actions
                 if isinstance(item.get("position_xyz"), (list, tuple)) and
                 len(item.get("position_xyz")) >= 3]
    vertical_delta = 0.0
    horizontal_displacement = 0.0
    if len(positions) >= 2:
        start = np.asarray(positions[0], np.float64)
        end = np.asarray(positions[-1], np.float64)
        vertical_delta = float(end[1] - start[1])
        horizontal_displacement = float(np.linalg.norm((end - start)[[0, 2]]))
    action_names = [str(item.get("action", "")) for item in actions]
    return {
        "action_count": len(actions),
        "traveled_distance_m": round(sum(
            float(item.get("moved_m", 0.0) or 0.0) for item in actions), 3),
        "signed_turn_deg": round(sum(
            float(item.get("turn_deg", 0.0) or 0.0) for item in actions), 2),
        "commanded_turn_deg_total": round(sum(
            float(item.get("commanded_turn_deg", 0.0) or 0.0)
            for item in actions), 2),
        "forward_command_count": action_names.count("move_forward"),
        "left_turn_command_count": action_names.count("turn_left"),
        "right_turn_command_count": action_names.count("turn_right"),
        "vertical_displacement_m": round(vertical_delta, 3),
        "horizontal_endpoint_displacement_m": round(
            horizontal_displacement, 3),
        "blocked_action_count": sum(
            float(item.get("moved_m", 0.0) or 0.0) < 0.01 and
            "forward" in str(item.get("action", "")) for item in actions),
        "point_edge_count": len(targets),
        "point_target_indices": [
            int(item.get("target_index", -1)) for item in targets],
    }


def _geometry_gate(scored):
    return bool(
        scored["selection_within_30deg"] and
        scored["executed_edge_within_30deg_and_forward"])


def _geometry_rejection(scored):
    return {
        "semantic_completion_verified": False,
        "ordered_stage_boundary_verified": False,
        "confidence": 1.0,
        "reason": (
            "post-run hidden GT geometry gate rejected the completed edge; "
            "independent VLM call was intentionally skipped"),
        "visual_evidence": "",
        "model_semantic_completion": None,
        "model_ordered_boundary": None,
        "independent_vlm_skipped": True,
        "postrun_gt_geometry_gate": {
            "selection_heading_error_deg": scored[
                "selection_heading_error_to_gt_deg"],
            "executed_heading_error_deg": scored[
                "executed_heading_error_to_gt_deg"],
            "gt_progress_delta_m": scored["gt_path_progress_delta_m"],
            "selection_within_30deg": scored["selection_within_30deg"],
            "executed_edge_within_30deg_and_forward": scored[
                "executed_edge_within_30deg_and_forward"],
            "passed": False,
        },
    }


def _verify_one(backend, stage, previous_stage, next_stage, target, scored,
                images, node_observations=None, stage_targets=None):
    previous_text = ((previous_stage or {}).get("navigation_instruction") or
                     "NONE (this is the first stage)")
    next_text = ((next_stage or {}).get("navigation_instruction") or
                 "NONE (this is the final stage)")
    prompt = f"""POST_RUN_INDEPENDENT_STAGE_BOUNDARY_AUDIT
You are auditing an executed R2R navigation stage span after the run. You are not
given the online model's decision. Decide from chronological RGB and executed
actions whether this exact one-or-more-edge span actually completed the stated
sub-instruction.

Sub-instruction: {stage.get('navigation_instruction', '')}
Immediately previous sub-instruction: {previous_text}
Immediately next sub-instruction: {next_text}
Form: {stage.get('form', '')}
Landmark: {stage.get('landmark', '')}
Required spatial target: {stage.get('semantic_spatial_target', '')}
Completion cue: {stage.get('completion_cue', '')}
Arrival evidence: {stage.get('visual_arrival_evidence', '')}
Forbidden endpoint: {stage.get('forbidden_target', '')}
Executed motion summary: {json.dumps(_motion_summary(target, stage_targets))}
Saved node semantic observations: {json.dumps(
    node_observations or {}, ensure_ascii=False)}

The attached images are chronological. Their exact sequence is:
{json.dumps((node_observations or {}).get('evidence_image_sequence', []),
            ensure_ascii=False)}

Return semantic_completion_verified=true only if the named relation and its
terminal cue is visibly achieved at the final current-node image through the
chronological edge and intermediate-node evidence. Merely moving toward it,
seeing it, being partway, reaching the wrong similar instance, or crossing
into the next clause is false. Return
ordered_stage_boundary_verified=true only if this edge ends at this stage's
boundary without visibly executing a later stage. Be conservative.

For a full stair ascent/descent, distinguish CURRENT FRONT from the rear and
side views using the burned-in labels. A completed flight may remain visible
behind or beside the agent. Reject it as unfinished only when ascending or
descending steps visibly continue in the instructed travel direction from the
current endpoint; accept a level landing ahead after chronological stair
traversal and a consistent signed vertical displacement.

For PASS of a named room/region, audit the topological RGB transition rather
than demanding that a room-category detector disappear from every overlapping
view. If the chronological evidence leaves the old enclosing interior and the
final image is a distinct
downstream foyer, hall, junction, entrance, or adjacent region, the pass is
complete even when the old room remains visible through a rear/side opening.
Near-full-frame room detections repeated across many sectors are fallible saved
observations, not bearing or containment proof; clean RGB is authoritative.
Reject only when CURRENT front is still enclosed by the same room or there is
no chronological evidence of entering a distinct downstream context.

For TURN_TO_LANDMARK/facing instructions, judge the final orientation from the
burned-in CURRENT FRONT panel first: a named landmark centered there is facing
the landmark even if a second similar surface also appears in a left/right
panel. Resolve an obvious misspelling by the complete visual/route context
(for example a fireplace surround versus a generic wall panel), and do not
claim that an object is only to the side without explicitly checking CURRENT
FRONT. Conversely, visibility only in a side/rear panel is not facing it.

For TURN_LEFT/TURN_RIGHT followed by ``walk/go/head/move towards X``, the
sub-instruction is a directional route commitment unless it separately says
to enter, cross, pass, traverse, or reach an endpoint. Verify completion when
the chronological action span establishes the requested side turn and makes
substantial continuous forward progress along the visually selected route.
Do not require arrival at X or reject merely because X is not centered in the
final FRONT panel; use the full current panorama and the immediately following
clause's landmark only as ordered route corroboration. Still reject an
opposite turn, reversal, stationary edge, or a route with no relevant visible
continuation.

For BETWEEN_OBJECTS with a ``reach/walk between A and B`` endpoint, being in
the traversable corridor with A and B on opposing lateral panorama sides is
positive endpoint evidence. The gap need not remain ahead once the camera is
inside it. Require both distinct named references and chronological movement
into their bracket; a front/rear pair or one repeated detector label is not
enough. Explicit ``through/past/beyond`` wording instead requires clearing the
pair according to its terminal cue.

For CIRCUMNAVIGATE/around/backside, the named obstacle need not remain visible
at CURRENT after a multi-edge pass. Verify every edge and intermediate-node
panorama in the complete chronology: an obstacle that moves from front to
side/rear and then disappears while the agent follows continuous floor is
valid evidence of clearing its far extent. Open
floor in the immediately adjacent next region (for example a kitchen beyond
living-room couches) does not by itself mean the next stage was executed; it
may be exactly the free floor behind the obstacle. Reject when the keyframes do
not show the named obstacle-side transition, when motion shortcuts through an
unrelated portal, or when CURRENT has already traversed a later-stage landmark.
Apply this generic endpoint rule literally: obstacle visible at the start,
then at a side/rear bearing at an intermediate node, followed by a final node
where it is confined to side/rear bearings or has just disappeared and
continuous open floor lies forward, is positive completion evidence. Do not
demand a separately recognizable surface called "behind". Seeing the region
named by the next stage beyond that floor is context, not proof that its later
action (such as following its landmarks or traversing it) has already occurred.
"""
    response = backend.generate_json(prompt, images, VERIFY_SCHEMA)
    for key in (
            "semantic_completion_verified",
            "ordered_stage_boundary_verified"):
        if not isinstance(response.get(key), bool):
            raise RuntimeError(f"independent verifier omitted boolean {key}")
    confidence = float(response.get("confidence", 0.0))
    model_semantic = bool(response["semantic_completion_verified"])
    model_ordered = bool(response["ordered_stage_boundary_verified"])
    geometry_aligned = _geometry_gate(scored)
    return {
        "semantic_completion_verified": bool(
            model_semantic and geometry_aligned),
        "ordered_stage_boundary_verified": bool(
            model_ordered and geometry_aligned),
        "confidence": confidence,
        "reason": str(response.get("reason", "")),
        "visual_evidence": str(response.get("visual_evidence", "")),
        "model_semantic_completion": model_semantic,
        "model_ordered_boundary": model_ordered,
        "independent_vlm_skipped": False,
        "postrun_gt_geometry_gate": {
            "selection_heading_error_deg": scored[
                "selection_heading_error_to_gt_deg"],
            "executed_heading_error_deg": scored[
                "executed_heading_error_to_gt_deg"],
            "gt_progress_delta_m": scored["gt_path_progress_delta_m"],
            "selection_within_30deg": scored["selection_within_30deg"],
            "executed_edge_within_30deg_and_forward": scored[
                "executed_edge_within_30deg_and_forward"],
            "passed": geometry_aligned,
        },
    }


def trajectory_episode_index(trajectory, episode_dir):
    """Episode directories are named by episode_id since 2026-09-10; the
    dataset row index is authoritative from the trajectory config and the
    directory suffix is only a fallback for older index-named runs."""
    recorded = (trajectory.get("config") or {}).get(
        "episode_index", trajectory.get("episode_index"))
    if recorded is not None:
        return int(recorded)
    return int(Path(episode_dir).name.split("_")[-1])


def _stage_lookup(stages):
    lookup = {}
    for ordinal, stage in enumerate(stages):
        stage_id = int(stage.get(
            "stage_id", stage.get("sub_instruction_id", ordinal)))
        lookup.setdefault(stage_id, (ordinal, stage))
    return lookup


def _run_one(backend, episode_dir, trajectory, stages, stage_lookup, scored,
             candidate, stage_id, always_call_vlm=False):
    target = candidate["target"]
    score = scored[int(target["target_index"])]
    images, evidence_paths, node_observations = _evidence_images(
        episode_dir, trajectory, candidate)
    if not _geometry_gate(score) and not always_call_vlm:
        result = _geometry_rejection(score)
    else:
        ordinal, stage = stage_lookup[stage_id]
        previous_stage = stages[ordinal - 1] if ordinal else None
        next_stage = (stages[ordinal + 1]
                      if ordinal + 1 < len(stages) else None)
        result = _verify_one(
            backend, stage, previous_stage, next_stage, target, score,
            images, node_observations,
            stage_targets=candidate.get("stage_targets"))
    return {
        "target_index": int(target["target_index"]),
        "completion_field": candidate["field"],
        "verification_source": "independent_rgb_action_trajectory_audit",
        "online_completion_hidden_from_verifier": True,
        "reference_path_exposed_to_online_navigation": False,
        "evidence_artifacts": evidence_paths,
        **result,
    }


def plan_episode(trajectory_path, dataset_episode, episode_index=None,
                 include_unknown=False, always_call_vlm=False):
    """Count the VLM calls a verify_episode run would make, without a backend."""
    trajectory_path = Path(trajectory_path)
    episode_dir = trajectory_path.parent
    trajectory = json.loads(trajectory_path.read_text())
    if episode_index is None:
        episode_index = trajectory_episode_index(trajectory, episode_dir)
    scored_episode = audit_episode(
        episode_index, trajectory, dataset_episode, trajectory_path,
        hidden_geometry=load_hidden_geometry(episode_dir))
    scored = {item["target_index"]: item
              for item in scored_episode["targets"]}
    stage_lookup = _stage_lookup(trajectory.get("instruction_stages") or [])
    candidates = [
        (stage_id, candidate) for stage_id, candidate
        in system_stage_completion_candidates(trajectory).items()
        if stage_id in stage_lookup]
    judged = [item for item in system_all_judged_edges(trajectory)
              if item["stage_id"] in stage_lookup] if include_unknown else []
    judged_keys = {(int(item["target"]["target_index"]), item["field"],
                    item["stage_id"]) for item in judged}
    planned = 0
    skipped = 0
    reused = 0
    for stage_id, candidate in candidates:
        key = (int(candidate["target"]["target_index"]), candidate["field"],
               stage_id)
        if len(candidate.get("stage_targets") or []) <= 1 and key in judged_keys:
            reused += 1
            continue
        score = scored[int(candidate["target"]["target_index"])]
        if always_call_vlm or _geometry_gate(score):
            planned += 1
        else:
            skipped += 1
    for item in judged:
        score = scored[int(item["target"]["target_index"])]
        if always_call_vlm or _geometry_gate(score):
            planned += 1
        else:
            skipped += 1
    return {
        "episode_id": trajectory.get("episode_id"),
        "episode_index": episode_index,
        "stage_candidates": len(candidates),
        "judged_edges": len(judged),
        "stage_candidates_reusing_edge_verdict": reused,
        "planned_vlm_calls": planned,
        "geometry_skipped_calls": skipped,
    }


def verify_episode(trajectory_path, dataset_episode, backend,
                   episode_index=None, include_unknown=False,
                   always_call_vlm=False):
    trajectory_path = Path(trajectory_path)
    episode_dir = trajectory_path.parent
    trajectory = json.loads(trajectory_path.read_text())
    if episode_index is None:
        episode_index = trajectory_episode_index(trajectory, episode_dir)
    scored_episode = audit_episode(
        episode_index, trajectory, dataset_episode, trajectory_path,
        hidden_geometry=load_hidden_geometry(episode_dir))
    scored = {item["target_index"]: item
              for item in scored_episode["targets"]}
    candidates = system_stage_completion_candidates(trajectory)
    stages = trajectory.get("instruction_stages") or []
    stage_lookup = _stage_lookup(stages)
    judgments = []
    # A single-edge stage candidate and its completed judgment are the same
    # edge with the same evidence; audit it once and reuse the verdict.
    single_edge_results = {}
    if include_unknown:
        for candidate in system_all_judged_edges(trajectory):
            if candidate["stage_id"] not in stage_lookup:
                continue
            result = _run_one(backend, episode_dir, trajectory, stages,
                              stage_lookup, scored, candidate,
                              candidate["stage_id"], always_call_vlm)
            single_edge_results[(
                result["target_index"], candidate["field"],
                candidate["stage_id"])] = result
            judgments.append({
                "sub_instruction_id": candidate["stage_id"],
                "online_status_output_only": candidate["online_status"],
                "online_confidence_output_only": candidate[
                    "online_confidence"],
                **result,
            })
    records = []
    for ordinal, stage in enumerate(stages):
        stage_id = int(stage.get(
            "stage_id", stage.get("sub_instruction_id", ordinal)))
        candidate = candidates.get(stage_id)
        if candidate is None:
            records.append({
                "sub_instruction_id": stage_id,
                "semantic_completion_verified": False,
                "ordered_stage_boundary_verified": False,
                "verification_source": (
                    "independent_rgb_action_trajectory_audit"),
                "reason": "no online-completed real node/edge candidate",
                "evidence_artifacts": [str(trajectory_path)],
            })
            continue
        reuse_key = (int(candidate["target"]["target_index"]),
                     candidate["field"], stage_id)
        if (len(candidate.get("stage_targets") or []) <= 1 and
                reuse_key in single_edge_results):
            result = single_edge_results[reuse_key]
        else:
            result = _run_one(backend, episode_dir, trajectory, stages,
                              stage_lookup, scored, candidate, stage_id,
                              always_call_vlm)
        records.append({"sub_instruction_id": stage_id, **result})
    payload = {
        "schema_version": 1,
        "episode_index": episode_index,
        "policy": (
            "independent post-run RGB/action audit intersected with hidden "
            "GT <=30-degree positive-progress geometry gate"),
        "online_completion_result_exposed_to_verifier": False,
        "reference_path_usage": "post_run_verification_only",
        "stages": records,
    }
    if include_unknown:
        payload["judgments_policy"] = (
            "every real judged edge (completed and unknown) audited "
            "independently; online status recorded for labelling only"
            + (", VLM called even when the geometry gate fails"
               if always_call_vlm else ""))
        payload["judgments"] = judgments
    output = episode_dir / "stage_completion_verification.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return output, payload


def _iter_round_episodes(round_root, dataset, episode_ids=None):
    for trajectory_path in sorted(round_root.rglob("trajectory.json")):
        episode_dir = trajectory_path.parent
        direct_serial_episode = episode_dir.parent.resolve() == (
            round_root.resolve())
        sharded_episode = episode_dir.parent.name.startswith("shard_")
        focused_probe = episode_dir.resolve() == round_root.resolve()
        if not ((episode_dir.name.startswith("episode_") and
                 (direct_serial_episode or sharded_episode)) or focused_probe):
            continue
        trajectory = json.loads(trajectory_path.read_text())
        if episode_ids is not None:
            try:
                if int(trajectory.get("episode_id")) not in episode_ids:
                    continue
            except (TypeError, ValueError):
                continue
        if focused_probe:
            episode_id = str(trajectory.get("episode_id"))
            trajectory_id = str(trajectory.get("trajectory_id"))
            matches = [index for index, item in enumerate(dataset)
                       if str(item.get("episode_id")) == episode_id and
                       str(item.get("trajectory_id")) == trajectory_id]
            if len(matches) != 1:
                raise RuntimeError(
                    "focused probe trajectory does not uniquely identify "
                    "one dataset episode")
            episode_index = matches[0]
        else:
            episode_index = trajectory_episode_index(trajectory, episode_dir)
        yield trajectory_path, episode_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("round_root", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--deepseek-env", type=Path, default=None,
                        help="optional KEY=VALUE file; relay credentials "
                             "otherwise come from the environment / local_env.sh")
    parser.add_argument("--model", default=None)
    parser.add_argument("--include-unknown", action="store_true",
                        help="also audit every unknown judgment, one record "
                             "per judged edge, under payload['judgments']")
    parser.add_argument("--always-call-vlm", action="store_true",
                        help="call the verifier even when the hidden geometry "
                             "gate fails (the gate is still AND-ed into the "
                             "final verdict)")
    parser.add_argument("--episode-ids", default=None,
                        help="comma-separated episode_id filter (the R2R "
                             "episode_id, not the dataset row index)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print planned VLM call counts; no backend, no "
                             "files written")
    args = parser.parse_args()
    dataset = _load_dataset(args.dataset)
    episode_ids = None
    if args.episode_ids:
        episode_ids = {int(item) for item in args.episode_ids.split(",")
                       if item.strip()}
    episodes = list(_iter_round_episodes(args.round_root, dataset, episode_ids))
    if args.dry_run:
        total = {"episodes": 0, "planned_vlm_calls": 0,
                 "geometry_skipped_calls": 0}
        for trajectory_path, episode_index in episodes:
            plan = plan_episode(
                trajectory_path, dataset[episode_index],
                episode_index=episode_index,
                include_unknown=args.include_unknown,
                always_call_vlm=args.always_call_vlm)
            print(json.dumps(plan, ensure_ascii=False), flush=True)
            total["episodes"] += 1
            total["planned_vlm_calls"] += plan["planned_vlm_calls"]
            total["geometry_skipped_calls"] += plan["geometry_skipped_calls"]
        print(json.dumps({"dry_run_total": total}, ensure_ascii=False))
        return
    backend = DeepSeekBackend(
        model=args.model, env_file=args.deepseek_env, timeout=180,
        image_detail="high")
    outputs = []
    for trajectory_path, episode_index in episodes:
        output, payload = verify_episode(
            trajectory_path, dataset[episode_index], backend,
            episode_index=episode_index,
            include_unknown=args.include_unknown,
            always_call_vlm=args.always_call_vlm)
        summary = {
            "episode_index": episode_index,
            "output": str(output),
            "verified_stages": sum(
                item["semantic_completion_verified"] and
                item["ordered_stage_boundary_verified"]
                for item in payload["stages"]),
            "stage_count": len(payload["stages"]),
        }
        if args.include_unknown:
            summary["judged_edges"] = len(payload["judgments"])
            summary["independently_verified_edges"] = sum(
                item["semantic_completion_verified"]
                for item in payload["judgments"])
        outputs.append(summary)
        print(json.dumps(outputs[-1], ensure_ascii=False), flush=True)
    (args.round_root / "stage_verification_summary.json").write_text(
        json.dumps({"episodes": outputs}, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
