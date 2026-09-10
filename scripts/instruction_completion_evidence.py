#!/usr/bin/env python3
"""Runtime-only structured evidence for node-transition instruction completion.

The functions in this module deliberately consume only information that is
available in the navigation graph at decision time.  They do not use an R2R
demonstration path, episode/category identity, or a reference label.
"""

from __future__ import annotations

import math
import re

import numpy as np


SIX_VIEW_DIRECTIONS = (
    "front", "front_left", "rear_left", "rear", "rear_right", "front_right")
EIGHT_VIEW_DIRECTIONS = (
    "front", "front_left", "left", "rear_left",
    "rear", "rear_right", "right", "front_right")

_IGNORED_INSTRUCTION_TOKENS = {
    "about", "after", "ahead", "along", "around", "before", "behind",
    "between", "continue", "down", "enter", "exit", "forward", "from",
    "front", "into", "large", "left", "move", "near", "next", "past",
    "right", "room", "small", "straight", "through", "toward", "towards",
    "turn", "until", "wait", "walk", "with", "your", "agent", "floor",
    "inside", "outside", "area", "space", "path", "there", "here", "then",
    "this", "that", "the", "and", "you", "are", "will", "would", "set",
}


def _round(value, digits=3):
    return round(float(value), digits)


def wrap_angle_rad(value):
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _position(record, fallback):
    value = np.asarray(record.get("position_xyz", fallback), np.float64)
    return value if value.shape == (3,) else np.asarray(fallback, np.float64)


def _phase_name(action):
    name = str(action.get("action", "unknown")).lower()
    if "left" in name and "turn" in name:
        return "turn_left"
    if "right" in name and "turn" in name:
        return "turn_right"
    if "forward" in name and "blocked" not in name:
        return "forward"
    if "blocked" in name or "collision" in name:
        return "blocked"
    if "stop" in name:
        return "stop"
    return name


def _trajectory_phases(actions, previous_position, previous_yaw):
    phases = []
    for index, action in enumerate(actions):
        name = _phase_name(action)
        if not phases or phases[-1]["action"] != name:
            start_position = (
                previous_position if index == 0 else
                _position(actions[index - 1], previous_position))
            start_yaw = (
                previous_yaw if index == 0 else
                float(actions[index - 1].get("yaw_rad", previous_yaw)))
            phases.append({
                "action": name,
                "start_step": index,
                "end_step": index,
                "count": 0,
                "distance_m": 0.0,
                "signed_turn_deg": 0.0,
                "start_position_xyz": start_position.tolist(),
                "start_yaw_deg": math.degrees(start_yaw),
            })
        phase = phases[-1]
        phase["end_step"] = index
        phase["count"] += 1
        phase["distance_m"] += float(action.get("moved_m", 0.0))
        phase["signed_turn_deg"] += float(action.get("turn_deg", 0.0))
        phase["end_position_xyz"] = _position(action, previous_position).tolist()
        phase["end_yaw_deg"] = math.degrees(float(
            action.get("yaw_rad", previous_yaw)))
    return [{
        **phase,
        "distance_m": _round(phase["distance_m"]),
        "signed_turn_deg": _round(phase["signed_turn_deg"], 1),
        "start_position_xyz": [_round(value) for value in phase["start_position_xyz"]],
        "end_position_xyz": [_round(value) for value in phase["end_position_xyz"]],
        "start_yaw_deg": _round(phase["start_yaw_deg"], 1),
        "end_yaw_deg": _round(phase["end_yaw_deg"], 1),
    } for phase in phases]


def _pose_at_source_frame(actions, frame_index, previous_position, previous_yaw):
    frame_index = int(frame_index)
    if frame_index <= 0 or not actions:
        return previous_position, previous_yaw
    action = actions[min(frame_index - 1, len(actions) - 1)]
    return (_position(action, previous_position),
            float(action.get("yaw_rad", previous_yaw)))


def summarize_motion(actions, previous_position_xyz, current_position_xyz,
                     previous_yaw_rad=None, current_yaw_rad=None,
                     keyframe_records=None):
    """Compress a full executor history without discarding maneuver order."""
    actions = list(actions or [])
    previous = np.asarray(previous_position_xyz, np.float64)
    current = np.asarray(current_position_xyz, np.float64)
    if previous.shape != (3,) or current.shape != (3,):
        raise ValueError("motion summary requires two XYZ positions")
    previous_yaw = float(previous_yaw_rad or 0.0)
    if current_yaw_rad is None:
        current_yaw = float(actions[-1].get("yaw_rad", previous_yaw)) if actions else previous_yaw
    else:
        current_yaw = float(current_yaw_rad)

    displacement = current - previous
    horizontal_displacement = float(np.linalg.norm(displacement[[0, 2]]))
    net_displacement = float(np.linalg.norm(displacement))
    traveled = sum(max(0.0, float(action.get("moved_m", 0.0))) for action in actions)
    left_turn = sum(max(0.0, float(action.get("turn_deg", 0.0))) for action in actions)
    right_turn = sum(max(0.0, -float(action.get("turn_deg", 0.0))) for action in actions)
    signed_turn = left_turn - right_turn
    phases = _trajectory_phases(actions, previous, previous_yaw)
    counts = {}
    for action in actions:
        name = _phase_name(action)
        counts[name] = counts.get(name, 0) + 1

    records = list(keyframe_records or [])
    if not records:
        if len(actions) <= 1:
            indices = [0, len(actions)]
        else:
            indices = np.linspace(0, len(actions), 5).round().astype(int).tolist()
        records = [{"keyframe_index": index, "source_frame_index": value}
                   for index, value in enumerate(indices)]
    keyframe_poses = []
    for index, record in enumerate(records):
        source_index = int(record.get("source_frame_index", index))
        position, yaw = _pose_at_source_frame(
            actions, source_index, previous, previous_yaw)
        keyframe_poses.append({
            "keyframe_index": int(record.get("keyframe_index", index)),
            "source_frame_index": source_index,
            "position_xyz": [_round(value) for value in position],
            "yaw_deg": _round(math.degrees(yaw), 1),
            "distance_from_start_m": _round(float(np.linalg.norm(position - previous))),
        })

    return {
        "coordinate_note": (
            "XYZ and yaw are online Habitat coordinates; use deltas and action order, "
            "not global compass assumptions"),
        "control_steps": len(actions),
        "action_counts": counts,
        "traveled_distance_m": _round(traveled),
        "net_displacement_m": _round(net_displacement),
        "horizontal_displacement_m": _round(horizontal_displacement),
        "vertical_delta_m": _round(displacement[1]),
        "path_efficiency": _round(net_displacement / max(traveled, 1e-6)),
        "start_yaw_deg": _round(math.degrees(previous_yaw), 1),
        "end_yaw_deg": _round(math.degrees(current_yaw), 1),
        "endpoint_heading_delta_deg": _round(
            math.degrees(wrap_angle_rad(current_yaw - previous_yaw)), 1),
        "cumulative_left_turn_deg": _round(left_turn, 1),
        "cumulative_right_turn_deg": _round(right_turn, 1),
        "signed_cumulative_turn_deg": _round(signed_turn, 1),
        "trajectory_phases": phases,
        "keyframe_pose_timeline": keyframe_poses,
    }


def instruction_tokens(instruction):
    words = re.findall(r"[a-z][a-z0-9'-]*", str(instruction).lower())
    return sorted({_canonical_token(word) for word in words
                   if len(word) >= 3 and word not in _IGNORED_INSTRUCTION_TOKENS})


def _canonical_token(word):
    """Return a conservative detector/text token normal form.

    Open-vocabulary detectors commonly return a singular class name even when
    an R2R instruction names a plural (``chairs`` -> ``chair``).  Keep this
    intentionally morphological rather than semantic: it must not invent a
    synonym or merge distinct landmark classes.
    """
    word = str(word).strip().lower()
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("ches", "shes", "sses", "xes", "zes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def canonical_label_tokens(label):
    """Tokenize a detector label in the same space as instruction tokens."""
    return {_canonical_token(word) for word in re.findall(
        r"[a-z][a-z0-9'-]*", str(label).lower())}


def _compact_node_semantics(semantics, relevant_tokens, max_per_view=5):
    direction_records = []
    relevant = []
    for raw_view in semantics.get("views", []):
        index = int(raw_view.get("view_index", -1))
        direction = SIX_VIEW_DIRECTIONS[index] if 0 <= index < 6 else "unknown"
        detections = sorted(
            raw_view.get("detections", []),
            key=lambda value: float(value.get("score", 0.0)), reverse=True)
        compact = []
        for detection in detections[:max_per_view]:
            record = {
                "label": str(detection.get("label", "")),
                "score": _round(detection.get("score", 0.0)),
                "depth_m": (None if detection.get("median_depth_m") is None
                            else _round(detection["median_depth_m"])),
                "area_fraction": _round(detection.get("mask_area_fraction", 0.0), 4),
            }
            compact.append(record)
        for detection in detections:
            label = str(detection.get("label", "")).lower()
            label_tokens = canonical_label_tokens(label)
            matched = sorted(token for token in relevant_tokens
                             if token in label_tokens)
            if not matched:
                continue
            relevant.append({
                "view_index": index, "direction": direction,
                "label": str(detection.get("label", "")),
                "matched_tokens": matched,
                "score": _round(detection.get("score", 0.0)),
                "depth_m": (None if detection.get("median_depth_m") is None
                            else _round(detection["median_depth_m"])),
                "area_fraction": _round(detection.get("mask_area_fraction", 0.0), 4),
                "box_xyxy": detection.get("box_xyxy"),
            })
        direction_records.append({
            "view_index": index, "direction": direction,
            "top_detections": compact})
    relevant.sort(key=lambda value: value["score"], reverse=True)
    label_scores = sorted(
        ((str(label), float(score))
         for label, score in semantics.get("label_scores", {}).items()),
        key=lambda value: value[1], reverse=True)[:12]
    return {
        "top_label_scores": [
            {"label": label, "score": _round(score)} for label, score in label_scores],
        "instruction_relevant_detections": relevant[:18],
        "views": direction_records,
    }


def summarize_semantic_transition(previous_semantics, current_semantics, instruction):
    """Make detector evidence directional, compact, and explicitly fallible."""
    tokens = instruction_tokens(instruction)
    previous = _compact_node_semantics(previous_semantics or {}, tokens)
    current = _compact_node_semantics(current_semantics or {}, tokens)
    previous_scores = {
        item["label"]: item["score"] for item in previous["top_label_scores"]}
    current_scores = {
        item["label"]: item["score"] for item in current["top_label_scores"]}
    labels = set(previous_scores) | set(current_scores)
    deltas = sorted(({
        "label": label,
        "previous_score": previous_scores.get(label, 0.0),
        "current_score": current_scores.get(label, 0.0),
        "delta": _round(current_scores.get(label, 0.0) -
                        previous_scores.get(label, 0.0)),
    } for label in labels), key=lambda value: abs(value["delta"]), reverse=True)[:12]
    return {
        "reliability_note": (
            "DINO+SAM detections are supporting evidence only; labels can be noisy "
            "or missing and must be checked against RGB and temporal identity"),
        "six_view_direction_order": list(SIX_VIEW_DIRECTIONS),
        "instruction_relevant_tokens": tokens,
        "previous_node": previous,
        "current_node": current,
        "largest_label_score_changes": deltas,
    }


def _image_descriptor(image):
    import cv2
    value = np.asarray(image, np.uint8)[..., :3]
    small = cv2.resize(value, (8, 8), interpolation=cv2.INTER_AREA)
    descriptor = small.astype(np.float32).reshape(-1) / 255.0
    descriptor -= float(descriptor.mean())
    norm = float(np.linalg.norm(descriptor))
    return descriptor / max(norm, 1e-8)


def summarize_visual_transition(previous_views, current_views,
                                previous_embedding=None, current_embedding=None,
                                previous_yaw_rad=None, current_yaw_rad=None):
    """Summarize graph-node visual change and absolute-yaw sector alignment."""
    if len(previous_views) != len(current_views):
        raise ValueError("visual transition requires equally sized panoramas")
    count = len(previous_views)
    previous_descriptors = [_image_descriptor(value) for value in previous_views]
    current_descriptors = [_image_descriptor(value) for value in current_views]
    matrix = np.asarray([
        [float(np.dot(left, right)) for right in current_descriptors]
        for left in previous_descriptors], np.float32)
    best_matches = [{
        "previous_view": index,
        "previous_direction": (EIGHT_VIEW_DIRECTIONS[index]
                               if count == 8 else SIX_VIEW_DIRECTIONS[index]),
        "best_current_view": int(np.argmax(matrix[index])),
        "similarity": _round(float(np.max(matrix[index]))),
    } for index in range(count)]
    result = {
        "descriptor_note": (
            "similarities are low-level appearance support, never semantic proof"),
        "panorama_view_count": count,
        "mean_best_view_similarity": _round(np.max(matrix, axis=1).mean()),
        "best_appearance_correspondences": best_matches,
    }
    if previous_embedding is not None and current_embedding is not None:
        left = np.asarray(previous_embedding, np.float32)
        right = np.asarray(current_embedding, np.float32)
        if left.shape == right.shape and left.size:
            cosine = float(np.dot(left, right) /
                           max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-8))
            result["stored_node_embedding_cosine"] = _round(cosine)
            result["stored_node_embedding_dimension"] = int(left.size)
    if previous_yaw_rad is not None and current_yaw_rad is not None:
        delta = wrap_angle_rad(float(current_yaw_rad) - float(previous_yaw_rad))
        sector_degrees = 360.0 / count
        sector_shift = int(round(math.degrees(delta) / sector_degrees))
        aligned = []
        for previous_index in range(count):
            current_index = (previous_index - sector_shift) % count
            aligned.append({
                "previous_view": previous_index,
                "same_absolute_direction_current_view": current_index,
                "similarity": _round(matrix[previous_index, current_index]),
            })
        result.update({
            "endpoint_heading_delta_deg": _round(math.degrees(delta), 1),
            "quantized_sector_shift": sector_shift,
            "same_absolute_direction_alignment": aligned,
            "mean_same_absolute_direction_similarity": _round(
                np.mean([item["similarity"] for item in aligned])),
        })
    return result
