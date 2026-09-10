#!/usr/bin/env python3
"""Turn typed R2R relations and detections into executable floor candidates."""

import re

import cv2
import numpy as np


DETECTION_GROUNDED_FORMS = {
    "EXIT_REGION", "ENTER_REGION", "SELECT_PORTAL", "TRAVERSE_PORTAL_REGION",
    "VERTICAL_UP", "VERTICAL_DOWN", "PASS_LANDMARK", "CIRCUMNAVIGATE",
    "BETWEEN_OBJECTS", "APPROACH_LANDMARK", "TURN_TO_LANDMARK", "STOP_WAIT",
}

CROSS_VIEW_SEMANTIC_POLICIES = {
    "hard_detection_gate", "soft_detection_evidence",
}


def apply_cross_view_semantic_policy(stage, candidates,
                                     policy="hard_detection_gate"):
    """Fuse per-view detection evidence without changing any ground mask.

    The baseline policy preserves the historical hard gate: once one panorama
    view has a relation-constrained detection mask, views with only floor are
    removed from VLM selection. The soft policy keeps those valid ground
    candidates and records detection presence as evidence for the VLM instead.
    """
    if policy not in CROSS_VIEW_SEMANTIC_POLICIES:
        raise ValueError(
            f"unknown cross-view semantic policy {policy!r}; expected one of "
            f"{sorted(CROSS_VIEW_SEMANTIC_POLICIES)}")
    form = stage.get("form")
    grounded = []
    if form in DETECTION_GROUNDED_FORMS:
        grounded = [candidate for candidate in candidates
                    if candidate["strategy_application"]["mode"] not in
                    {"floor_only", "floor_only_fallback"}]
    grounded_ids = {id(candidate) for candidate in grounded}
    grounded_views = [int(candidate.get("view_index", index))
                      for index, candidate in enumerate(candidates)
                      if id(candidate) in grounded_ids]
    for index, candidate in enumerate(candidates):
        application = candidate["strategy_application"]
        has_relation_evidence = id(candidate) in grounded_ids
        application.update({
            "cross_view_semantic_policy": policy,
            "relation_detection_evidence_in_view": has_relation_evidence,
            "relation_detection_evidence_views": grounded_views,
        })
        if not grounded or has_relation_evidence:
            application["cross_view_candidate_status"] = (
                "relation_evidence" if has_relation_evidence else
                "no_competing_relation_evidence")
            continue
        if policy == "hard_detection_gate":
            candidate["point"] = None
            application["view_rejected"] = (
                "semantic_evidence_available_in_another_view")
            application["cross_view_candidate_status"] = "hard_rejected"
        else:
            application["cross_view_candidate_status"] = (
                "retained_as_ground_with_missing_detection_evidence")
            application["soft_evidence_warning"] = (
                "another view has detector evidence; compare its reliability, "
                "instruction direction, and route continuity")
    return {
        "policy": policy,
        "form": form,
        "relation_evidence_views": grounded_views,
        "retained_candidate_views": [
            int(candidate.get("view_index", index))
            for index, candidate in enumerate(candidates)
            if candidate.get("point") is not None],
    }


def _depth_valid(depth):
    return np.isfinite(depth) & (depth > 0)


def _rect(shape, x0, y0, x1, y1):
    h, w = shape
    result = np.zeros(shape, bool)
    x0, x1 = np.clip(np.round([x0, x1]).astype(int), 0, w)
    y0, y1 = np.clip(np.round([y0, y1]).astype(int), 0, h)
    if x1 > x0 and y1 > y0:
        result[y0:y1, x0:x1] = True
    return result


def _best(detections):
    return max(detections, key=lambda item: item.score) if detections else None


def constrain_floor_candidates(stage, floor_mask, depth, detections,
                               object_mask=None, min_pixels=24):
    """Return a relation mask; ``depth=None`` enforces RGB-only geometry."""
    base = floor_mask.astype(bool)
    form = stage.get("form", "OTHER")
    record = {
        "form": form, "mode": "floor_only", "detection_required": False,
        "detection_count": len(detections), "fallback": None,
    }
    if object_mask is not None:
        # A conservative image-space safety margin prevents snapping to floor
        # immediately under chairs/tables or onto uncertain object boundaries.
        object_margin = cv2.dilate(
            object_mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        safe_base = base & ~object_margin
        record["small_segmenter_object_pixels"] = int(object_mask.sum())
        record["safe_floor_pixels"] = int(safe_base.sum())
        object_fraction = float(np.asarray(object_mask, bool).mean())
        base_pixels = int(base.sum())
        safe_retention = (
            float(safe_base.sum()) / max(base_pixels, 1))
        object_mask_sane = bool(
            object_fraction <= 0.60 and safe_retention >= 0.20)
        record["object_mask_sanity"] = {
            "accepted_for_floor_subtraction": object_mask_sane,
            "image_fraction": round(object_fraction, 5),
            "ground_retention_fraction": round(safe_retention, 5),
            "policy": (
                "near-full-frame or floor-erasing open-vocabulary masks are "
                "semantic evidence only, not collision geometry"),
        }
        if object_mask_sane and int(safe_base.sum()) >= min_pixels:
            base = safe_base
    if form not in DETECTION_GROUNDED_FORMS:
        record["mode"] = "small_seg_safe_floor"
        return base, record
    record["detection_required"] = True
    if not detections:
        record["fallback"] = "no_matching_detection"
        return base, record

    h, w = base.shape
    candidate = np.zeros_like(base)
    relation = ""
    selected = []

    secondary_forms = set(stage.get("secondary_forms", []) or [])
    compound_portal = (
        form == "PASS_LANDMARK" and bool(
            {"TRAVERSE_PORTAL_REGION", "ENTER_REGION", "EXIT_REGION"} &
            secondary_forms))
    if form == "BETWEEN_OBJECTS" and len(detections) >= 2:
        ranked = sorted(detections, key=lambda item: item.score, reverse=True)
        first = ranked[0]
        second = max(ranked[1:], key=lambda item:
                     abs((item.box_xyxy[0] + item.box_xyxy[2]) -
                         (first.box_xyxy[0] + first.box_xyxy[2])))
        left, right = sorted([first, second], key=lambda item: item.box_xyxy[0])
        gap0, gap1 = left.box_xyxy[2], right.box_xyxy[0]
        if gap1 <= gap0:
            centers = [(item.box_xyxy[0] + item.box_xyxy[2]) / 2 for item in (left, right)]
            middle = sum(centers) / 2
            gap0, gap1 = middle - 0.10 * w, middle + 0.10 * w
        candidate = _rect(base.shape, gap0, min(left.box_xyxy[1], right.box_xyxy[1]),
                          gap1, h)
        relation, selected = "gap_between_instances", [left, right]
    else:
        detection = _best(detections)
        selected = [detection]
        x0, y0, x1, y1 = detection.box_xyxy
        box_width = max(x1 - x0, 12)
        object_depth = detection.median_depth_m
        if (form in {"EXIT_REGION", "ENTER_REGION", "SELECT_PORTAL",
                     "TRAVERSE_PORTAL_REGION"} or compound_portal):
            # Portal forms must derive their floor wedge from a portal, never
            # from a higher-scoring destination object (bed, couch, table).
            # Destination detections remain available to the VLM as semantic
            # evidence but cannot geometrically masquerade as a doorway.
            portal_detections = [item for item in detections if
                                 set(re.findall(
                                     r"[a-z]+", str(item.label).lower())) &
                                 {"door", "doorway", "opening", "hallway",
                                  "portal"}]
            if portal_detections:
                detection = _best(portal_detections)
                selected = [detection]
                x0, y0, x1, y1 = detection.box_xyxy
                box_width = max(x1 - x0, 12)
                # Edge-clipped portal boxes do not provide a reliable pixel
                # bearing. Broaden toward the image center while retaining
                # the same detected portal as the geometric source.
                if x0 <= 0.05 * w or x1 >= 0.95 * w:
                    x0 = min(x0, 0.15 * w)
                    x1 = max(x1, 0.85 * w)
                    record["clipped_portal_expansion"] = True
            else:
                record["fallback"] = "no_portal_detection"
                record["mode"] = "floor_only_fallback"
                return base, record
            candidate = _rect(base.shape, x0 + 0.12 * box_width, (y0 + y1) / 2,
                              x1 - 0.12 * box_width, min(h, y1 + 0.16 * h))
            relation = "floor_through_detected_portal"
            # Grounded-SAM often marks only the near lower strip of a doorway
            # floor.  Dilate that connected 2-D support upward inside the
            # portal wedge so the RGB anchor pass can choose a point just
            # beyond the threshold rather than stopping on the near side.
            # The wedge is still constrained by the detected portal's image
            # width; no depth, navmesh, demonstration path, or scene identity
            # participates in this morphology-only candidate expansion.
            portal_support = cv2.dilate(
                base.astype(np.uint8), np.ones((31, 9), np.uint8)).astype(bool)
            portal_wedge = _rect(
                base.shape, x0 - 0.20 * box_width, 0.48 * h,
                x1 + 0.20 * box_width, min(h, y1 + 0.16 * h))
            base |= portal_support & portal_wedge
            candidate |= portal_support & portal_wedge
            if object_depth is not None:
                candidate &= (~_depth_valid(depth) | (depth >= 0.80 * object_depth))
        elif form in {"VERTICAL_UP", "VERTICAL_DOWN"}:
            expanded = cv2.dilate(detection.mask.astype(np.uint8),
                                  np.ones((17, 17), np.uint8)).astype(bool)
            candidate = expanded | _rect(base.shape, x0, y0, x1, min(h, y1 + 0.12 * h))
            relation = "stair_or_landing_centerline"
        elif form == "PASS_LANDMARK":
            candidate = _rect(base.shape, x0 - 0.45 * box_width, y0,
                              x1 + 0.45 * box_width, h)
            relation = "floor_beyond_landmark"
            if object_depth is not None:
                candidate &= _depth_valid(depth) & (depth >= object_depth + 0.25)
        elif form == "CIRCUMNAVIGATE":
            clause = stage.get("source_clause", stage.get("navigation_instruction", "")).lower()
            side = "left" if re.search(r"\bleft\b", clause) else "right" if re.search(r"\bright\b", clause) else "either"
            clearance = 0.12 * w
            if side == "left":
                candidate = _rect(base.shape, x0 - clearance - box_width, y0, x0 - clearance, h)
            elif side == "right":
                candidate = _rect(base.shape, x1 + clearance, y0, x1 + clearance + box_width, h)
            else:
                candidate = (_rect(base.shape, x0 - clearance - box_width, y0, x0 - clearance, h) |
                             _rect(base.shape, x1 + clearance, y0, x1 + clearance + box_width, h))
            relation = f"{side}_side_clearance_corridor"
        elif form == "TURN_TO_LANDMARK":
            # A named turn is a bearing relation rather than an approach
            # endpoint.  Keep the whole connected floor band around the
            # landmark's image bearing so the RGB VLM can place the final ray
            # toward the landmark even when it is lateral or rearward.
            candidate = _rect(base.shape, x0 - 0.55 * box_width, y0 - 0.05 * h,
                              x1 + 0.55 * box_width, h)
            relation = "floor_on_landmark_bearing"
        else:  # APPROACH_LANDMARK / STOP_WAIT
            candidate = _rect(base.shape, x0 - 0.65 * box_width, y1 - 0.08 * h,
                              x1 + 0.65 * box_width, h)
            relation = "safe_near_side_floor"
            if object_depth is not None:
                candidate &= (_depth_valid(depth) & (depth <= object_depth) &
                              (depth >= max(0.25, object_depth - 1.75)))

    candidate &= base
    # Never permit points on detected instances, including dilated collision margin.
    obstacle = np.zeros_like(base)
    ignored_region_masks = []
    for detection in selected:
        detection_fraction = float(np.asarray(detection.mask, bool).mean())
        if detection_fraction > 0.60:
            ignored_region_masks.append({
                "label": str(detection.label),
                "mask_area_fraction": round(detection_fraction, 5),
                "reason": "near_full_frame_region_mask_not_obstacle",
            })
            continue
        obstacle |= detection.mask
    obstacle = cv2.dilate(obstacle.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    candidate &= ~obstacle
    record.update({
        "mode": relation,
        "selected_detections": [item.prompt_record() for item in selected],
        "candidate_pixels": int(candidate.sum()),
        "candidate_fraction": round(float(candidate.mean()), 5),
        "ignored_region_masks": ignored_region_masks,
    })
    if int(candidate.sum()) < min_pixels:
        record["fallback"] = "relation_mask_too_small"
        record["mode"] = "floor_only_fallback"
        return base, record
    return candidate, record
