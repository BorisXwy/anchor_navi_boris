#!/usr/bin/env python3
"""Reusable point-navigation executor with an external point-selection API."""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
import math
import time

import cv2
import numpy as np
import torch
from habitat_sim.utils.common import quat_from_angle_axis
from PIL import Image
from path_projection import draw_reference_path_overlay, project_reference_path


POINT_NAVIGATION_ARRIVED = "point_navigation_arrived"
STOP_ARRIVAL_REASON = "all_stop_cluster_points_disappeared"


TRACKING_CLUSTER_PROFILES = {
    # Kept for reproducibility of the first 100-case run.
    "legacy_3x3": {
        "navigation_rows": 3,
        "navigation_cols": 3,
        "stop_rows": 3,
        "stop_cols": 3,
        "navigation_x_range": (0.40, 0.60),
        "navigation_y_range": (0.40, 0.60),
        "stop_x_range": (0.40, 0.60),
        "stop_y_range": (0.90, 0.99),
        "arrival_visible_fraction": 0.0,
        "arrival_confirmation_frames": 1,
        "maximum_goal_loss_frames": 1,
        "initial_near_field_depth_m": None,
        "initial_near_field_min_y_fraction": None,
    },
    # Round-1 dense profile: cover the whole grounded bottom band instead of
    # making the decision from only nine, tightly packed points.
    "dense_bottom_v1": {
        "navigation_rows": 5,
        "navigation_cols": 5,
        "stop_rows": 5,
        "stop_cols": 9,
        "navigation_x_range": (0.36, 0.64),
        "navigation_y_range": (0.36, 0.64),
        "stop_x_range": (0.32, 0.68),
        "stop_y_range": (0.84, 0.99),
        "arrival_visible_fraction": 0.04,
        "arrival_confirmation_frames": 1,
        "maximum_goal_loss_frames": 2,
        "initial_near_field_depth_m": None,
        "initial_near_field_min_y_fraction": None,
    },
    # Round-2 ablation: only the physical stopping evidence is densified. The
    # 3x3 navigation cluster stays identical to the legacy controller so the
    # arrival experiment does not silently change steering behavior.
    "dense_stop_v2": {
        "navigation_rows": 3,
        "navigation_cols": 3,
        "stop_rows": 5,
        "stop_cols": 9,
        "navigation_x_range": (0.40, 0.60),
        "navigation_y_range": (0.40, 0.60),
        "stop_x_range": (0.32, 0.68),
        "stop_y_range": (0.84, 0.99),
        "arrival_visible_fraction": 0.04,
        "arrival_confirmation_frames": 1,
        "maximum_goal_loss_frames": 2,
        "initial_near_field_depth_m": None,
        "initial_near_field_min_y_fraction": None,
    },
    # Round-3 candidate: the same controller plus one temporal confirmation
    # frame. This tests correlated one-frame visibility loss independently of
    # point density.
    "dense_stop_confirm_v3": {
        "navigation_rows": 3,
        "navigation_cols": 3,
        "stop_rows": 5,
        "stop_cols": 9,
        "navigation_x_range": (0.40, 0.60),
        "navigation_y_range": (0.40, 0.60),
        "stop_x_range": (0.32, 0.68),
        "stop_y_range": (0.84, 0.99),
        "arrival_visible_fraction": 0.04,
        "arrival_confirmation_frames": 2,
        "maximum_goal_loss_frames": 2,
        "initial_near_field_depth_m": None,
        "initial_near_field_min_y_fraction": None,
    },
    # Round-2 candidate. Two consecutive low-visibility frames suppress a
    # one-frame TAPIR visibility collapse. While confirming, the farther
    # grounded navigation cluster remains the image-goal reference.
    "dense_bottom_v2": {
        "navigation_rows": 5,
        "navigation_cols": 7,
        "stop_rows": 7,
        "stop_cols": 11,
        "navigation_x_range": (0.34, 0.66),
        "navigation_y_range": (0.34, 0.66),
        "stop_x_range": (0.30, 0.70),
        "stop_y_range": (0.82, 0.99),
        "arrival_visible_fraction": 0.04,
        "arrival_confirmation_frames": 2,
        "maximum_goal_loss_frames": 2,
        "initial_near_field_depth_m": None,
        "initial_near_field_min_y_fraction": None,
    },
    # Round-4 calibration candidate: require two post-loss control frames. It
    # is evaluated against v3 rather than promoted blindly.
    "dense_stop_confirm_v4": {
        "navigation_rows": 3,
        "navigation_cols": 3,
        "stop_rows": 5,
        "stop_cols": 9,
        "navigation_x_range": (0.40, 0.60),
        "navigation_y_range": (0.40, 0.60),
        "stop_x_range": (0.32, 0.68),
        "stop_y_range": (0.84, 0.99),
        "arrival_visible_fraction": 0.04,
        "arrival_confirmation_frames": 3,
        "maximum_goal_loss_frames": 3,
        "initial_near_field_depth_m": None,
        "initial_near_field_min_y_fraction": None,
    },
    # Round-5 candidate: three post-loss frames plus a conservative initial
    # near-field gate. The latter requires both short depth and a point already
    # in the bottom fifth of the observation; depth alone is not sufficient.
    "dense_stop_confirm_v5": {
        "navigation_rows": 3,
        "navigation_cols": 3,
        "stop_rows": 5,
        "stop_cols": 9,
        "navigation_x_range": (0.40, 0.60),
        "navigation_y_range": (0.40, 0.60),
        "stop_x_range": (0.32, 0.68),
        "stop_y_range": (0.84, 0.99),
        "arrival_visible_fraction": 0.04,
        "arrival_confirmation_frames": 4,
        "maximum_goal_loss_frames": 4,
        "initial_near_field_depth_m": 1.0,
        "initial_near_field_min_y_fraction": 0.79,
    },
    # Round-6 candidate: preserve v5's conservative steady-state rule, but do
    # not demand four extra frames after the controller itself loses its
    # navigation cluster. Simultaneous loss of the independently tracked dense
    # bottom, legacy goal, and navigation clusters is a terminal consensus.
    "dense_stop_consensus_v6": {
        "navigation_rows": 3,
        "navigation_cols": 3,
        "stop_rows": 5,
        "stop_cols": 9,
        "navigation_x_range": (0.40, 0.60),
        "navigation_y_range": (0.40, 0.60),
        "stop_x_range": (0.32, 0.68),
        "stop_y_range": (0.84, 0.99),
        "arrival_visible_fraction": 0.04,
        "arrival_confirmation_frames": 4,
        "maximum_goal_loss_frames": 4,
        "initial_near_field_depth_m": 1.0,
        "initial_near_field_min_y_fraction": 0.79,
        "terminal_all_cluster_loss_is_arrival": True,
    },
    # Round-7 candidate: doorway occlusion can make all clusters vanish before
    # the selected floor point. Require some physical travel relative to the
    # point's initial monocular/RGB-D depth, while retaining the explicit v5
    # near-field exception. This is runtime evidence, not post-hoc target range.
    "dense_stop_motion_guard_v7": {
        "navigation_rows": 3,
        "navigation_cols": 3,
        "stop_rows": 5,
        "stop_cols": 9,
        "navigation_x_range": (0.40, 0.60),
        "navigation_y_range": (0.40, 0.60),
        "stop_x_range": (0.32, 0.68),
        "stop_y_range": (0.84, 0.99),
        "arrival_visible_fraction": 0.04,
        "arrival_confirmation_frames": 4,
        "maximum_goal_loss_frames": 4,
        "initial_near_field_depth_m": 1.0,
        "initial_near_field_min_y_fraction": 0.79,
        "terminal_all_cluster_loss_is_arrival": True,
        "minimum_travel_to_initial_depth_fraction": 0.70,
    },
    # Round-9 candidate: RGB-D depth can be much shorter than the actual
    # navigable route when the selected floor lies beyond a railing, doorway,
    # or obstacle.  Guard tracker-loss arrival against the larger of direct
    # depth and the start-time navmesh path length.  Two terminal rules recover
    # low-visibility points at the control budget and points already inside the
    # physical arrival radius; neither consumes post-hoc final target distance.
    "dense_stop_geodesic_guard_v9": {
        "navigation_rows": 3,
        "navigation_cols": 3,
        "stop_rows": 5,
        "stop_cols": 9,
        "navigation_x_range": (0.40, 0.60),
        "navigation_y_range": (0.40, 0.60),
        "stop_x_range": (0.32, 0.68),
        "stop_y_range": (0.84, 0.99),
        "arrival_visible_fraction": 0.04,
        "arrival_confirmation_frames": 4,
        "maximum_goal_loss_frames": 4,
        "initial_near_field_depth_m": 1.0,
        "initial_near_field_min_y_fraction": 0.79,
        "terminal_all_cluster_loss_is_arrival": True,
        "minimum_travel_to_initial_depth_fraction": 0.70,
        "use_initial_geodesic_for_motion_guard": True,
        "budget_terminal_goal_lost": True,
        "budget_terminal_stop_fraction": 0.15,
        "budget_terminal_travel_fraction": 0.90,
        "near_geodesic_cluster_loss_arrival": True,
        "near_geodesic_threshold_m": 0.75,
        "near_geodesic_minimum_travel_fraction": 0.30,
    },
    # Stage-00 navigation revision.  Keep v9's dense stop cluster and
    # geodesic/motion safeguards, but require the physically interpretable
    # half-visible stop evidence and probe a bounded side turn after repeated
    # blocked-forward controls.  The probe is an executor-only recovery: it
    # never changes the selected RGB point or consults hidden reference data.
    "dense_stop_motion_recovery_v10": {
        "navigation_rows": 3,
        "navigation_cols": 3,
        "stop_rows": 5,
        "stop_cols": 9,
        "navigation_x_range": (0.40, 0.60),
        "navigation_y_range": (0.40, 0.60),
        "stop_x_range": (0.32, 0.68),
        "stop_y_range": (0.84, 0.99),
        # At least half of the dense bottom/stop cluster must be lost before
        # arrival can be considered.  The four-frame confirmation remains.
        "arrival_visible_fraction": 0.50,
        "arrival_confirmation_frames": 4,
        "maximum_goal_loss_frames": 4,
        "initial_near_field_depth_m": 1.0,
        "initial_near_field_min_y_fraction": 0.79,
        "terminal_all_cluster_loss_is_arrival": True,
        "minimum_travel_to_initial_depth_fraction": 0.90,
        "use_initial_geodesic_for_motion_guard": True,
        "budget_terminal_goal_lost": True,
        "budget_terminal_stop_fraction": 0.50,
        "budget_terminal_travel_fraction": 0.95,
        "near_geodesic_cluster_loss_arrival": True,
        "near_geodesic_threshold_m": 2.0,
        "near_geodesic_minimum_travel_fraction": 0.50,
        "near_loss_requires_goal_visibility": False,
        "near_loss_long_target_minimum_travel_fraction": 0.90,
        "stall_recovery_trigger_frames": 3,
        "stall_recovery_turn_deg": 30.0,
        "stall_recovery_max_probes": 4,
        "navigation_loss_grace_frames": 4,
    },
}

# Navigation-only candidate for the next stage-00 executor round.  Keeping a
# separate profile preserves the v10 run as an immutable parent artifact while
# changing only the generic terminal-loss motion gate.
TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v11"] = {
    **TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v10"],
    "near_loss_long_target_minimum_travel_fraction": 0.90,
}

# Navigation-only candidate: react to the first blocked-forward control with
# a small alternating side probe.  This is a generic doorway/obstacle
# recovery and does not alter the selected point, use depth, or consult the
# reference trajectory.  v11 remains the frozen parent for comparison.
TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v12"] = {
    **TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v11"],
    "stall_recovery_trigger_frames": 1,
    "stall_recovery_max_probes": 6,
    "navigation_loss_grace_frames": 4,
}

# Navigation-only candidate: at a doorway the TAPIR cluster can disappear
# while the robot still has a valid, bounded heading command.  Retain the
# last *online* cluster for a short coast window so the crop/policy can
# re-observe the target; after that window the executor still fails closed.
# No depth, reference path, or hidden target is used.
TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v13"] = {
    **TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v11"],
    "navigation_loss_coast_frames": 8,
}

# Near a selected endpoint, the dense bottom cluster can disappear first and
# the retained online center cluster then needs several frames merely to turn
# back toward that same endpoint.  Give this fail-closed coast four additional
# frames so the controller can finish the final bounded approach.  This does
# not change the dense-cluster arrival rule, point target, semantic direction,
# or selection inputs, and the stale online witness still expires.
TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v24_endpoint_coast"] = {
    **TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v13"],
    "navigation_loss_coast_frames": 12,
}

# Navigation-only candidate: combine the first bounded side probe with a
# small forward component (under the executor turn limit), so an obstacle
# corner can be rounded instead of spending the recovery frame rotating in
# place.  The probe alternates sides and remains entirely collision/local
# observation based.
TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v14"] = {
    **TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v11"],
    "stall_recovery_trigger_frames": 1,
    "stall_recovery_turn_deg": 18.0,
    "stall_recovery_max_probes": 6,
    "navigation_loss_coast_frames": 4,
}

# Navigation-only candidate: a larger alternating diagonal probe is allowed
# to advance one bounded step in the probed heading.  This explicitly rounds
# a collision corner; it still relies only on `try_step` collision outcome and
# online point visibility, and is capped by the probe count.
TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v15"] = {
    **TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v11"],
    "stall_recovery_trigger_frames": 1,
    "stall_recovery_turn_deg": 45.0,
    "stall_recovery_max_probes": 6,
    "stall_recovery_forward": True,
    "navigation_loss_coast_frames": 4,
}

# Navigation-only candidate: once a collision probe starts, keep the bounded
# alternating sweep active until one probe actually moves.  This avoids the
# visual steering loop immediately cancelling a probe while the target is
# occluded behind the doorway.
TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v16"] = {
    **TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v15"],
    "stall_recovery_max_probes": 8,
    "navigation_loss_coast_frames": 8,
    "persist_stall_recovery": True,
}

# Navigation-only candidate: hold one side of the collision sweep long enough
# to reach a perpendicular corridor, then let visual reacquisition resume.
# `stall_recovery_probe_sign` is a generic policy parameter; the opposite sign
# is tested as a separate candidate rather than mixing directions in one run.
TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v17"] = {
    **TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v16"],
    "stall_recovery_probe_sign": -1.0,
}

TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v18"] = {
    **TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v16"],
    "stall_recovery_probe_sign": 1.0,
}

# Navigation-only arrival candidate: when the navigation cluster itself is
# lost, require the dense stop cluster to be effectively empty before
# declaring arrival.  This prevents a low-visibility false positive while
# retaining the normal half-visible stop rule whenever navigation remains
# trackable.
TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v19"] = {
    **TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v17"],
    "navigation_loss_arrival_stop_fraction": 0.05,
}

# Frozen-parent candidate for the stage gate: retain v11's alternating local
# collision probes (the v17 fixed-side sweep is useful diagnostically but is
# not promoted) and apply the stricter navigation-loss arrival gate.
TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v20"] = {
    **TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v11"],
    "navigation_loss_arrival_stop_fraction": 0.05,
}

# Stage-00 vertical-safety candidate.  A stair/landing target is often
# occluded while the robot is still on the flight.  The generic near-loss rule
# therefore requires a full start-time route traversal before it can emit an
# arrival signal; it cannot turn a halfway stair observation into a completed
# node.  Other forms retain v20's thresholds for comparison.
TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v21_vertical_safe"] = {
    **TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v20"],
    "near_geodesic_minimum_travel_fraction": 0.90,
    "near_loss_long_target_minimum_travel_fraction": 0.90,
}

# v22 keeps the v21 strict arrival gate and adds a bounded online coast when
# TAPIR briefly loses the central navigation cluster at a stair/doorway.  The
# retained tracks expire after eight frames and are never extrapolated beyond
# the last observed image, so failure still returns a non-arrival signal.
TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v22_vertical_coast"] = {
    **TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v21_vertical_safe"],
    "navigation_loss_coast_frames": 8,
}

# Navigation-only candidate: a terminal cluster loss can occur after the
# visual ray has already crossed a doorway or object, especially when the
# selected pixel is on a side opening.  Bound the near-geodesic loss fallback
# by a generic upper travel ratio so an executor cannot continue well beyond
# its start-time point estimate and then report arrival.  The ordinary
# half-visible stop consensus remains unchanged; this bound applies only to
# the loss fallback and uses no final/reference trajectory information.
TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v23_no_overshoot"] = {
    **TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v22_vertical_coast"],
    "near_loss_maximum_travel_fraction": 1.35,
    "cluster_loss_maximum_travel_fraction": 1.35,
    "max_travel_to_initial_geodesic_fraction": 1.50,
}

# Dual-cluster arrival candidate: the dense bottom cluster is the physical
# stop witness, while the center cluster is the steering witness.  A few
# stale goal-confirmation tracks must not suppress arrival once the dense
# bottom points have stayed below threshold, the steering cluster is almost
# gone, and the start-time motion guard is satisfied.
TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v24_dual_cluster"] = {
    **TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v23_no_overshoot"],
    "dense_stop_only_arrival": True,
    "dense_stop_navigation_visible_fraction": 0.15,
}

# Slightly less brittle dual-cluster candidate: the center steering cluster
# may retain a sparse third of tracks on a doorway edge even after the dense
# bottom stop cluster has crossed its half-loss threshold.  Keep the stop
# witness and motion confirmation unchanged, but allow that bounded sparse
# steering remainder to confirm arrival; the overshoot cap still fails closed.
TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v25_dual_cluster_sparse_nav"] = {
    **TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v24_dual_cluster"],
    "dense_stop_navigation_visible_fraction": 0.35,
}

# Near-field candidate: a navmesh projection can already be inside the
# physical arrival radius even when its image pixel is not on the bottom-most
# row (for example a floor point selected near the center of a short crop).
# Use the start-time geodesic projection as an additional generic witness;
# this is controller geometry, not VLM/depth input, and remains gated by the
# caller's explicit ``allow_initial_near_field_arrival`` flag.
TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v26_nearfield_dual_cluster"] = {
    **TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v25_dual_cluster_sparse_nav"],
    "initial_near_field_geodesic_m": 0.35,
    # If the tracker reaches the selected point but the finite-step budget
    # expires before a final loss frame, use the selected point's own online
    # navmesh distance (never the hidden R2R path) together with the dense
    # bottom-cluster witness to emit the physical arrival signal.
    "endpoint_geodesic_arrival": True,
    "endpoint_geodesic_radius_m": 0.75,
    "endpoint_geodesic_stop_fraction": 0.50,
}

# Online overshoot guard: combine the selected point's own Habitat/navmesh
# endpoint with the already-required dense bottom-cluster loss.  This is not a
# semantic cue and never enters VLM input.  It stops the controller on the
# first control frame in which the agent is physically within 0.35 m of its
# selected point and at least half of the dense stop cluster has disappeared,
# rather than discovering the same fact only after it has walked past it.
TRACKING_CLUSTER_PROFILES[
    "dense_stop_motion_recovery_v27_online_endpoint"] = {
        **TRACKING_CLUSTER_PROFILES[
            "dense_stop_motion_recovery_v26_nearfield_dual_cluster"],
        "online_endpoint_geodesic_arrival": True,
        "online_endpoint_geodesic_radius_m": 0.35,
        "online_endpoint_stop_fraction": 0.50,
}

# Loss-validated endpoint continuation.  A tracker may drop every image point
# at a doorway even though the frozen, selected navmesh endpoint is still well
# ahead of the robot.  In that case cluster loss is explicitly *not* arrival:
# keep the bounded executor running toward the same selected endpoint.  The
# endpoint cannot change semantic/view selection and is never taken from the
# demonstration path.  Conversely, a finite control budget may expire just
# inside the same physical radius; the existing dense half-loss witness then
# allows the executor to emit the otherwise missed arrival signal.
TRACKING_CLUSTER_PROFILES[
    "dense_stop_motion_recovery_v28_endpoint_loss_guard"] = {
        **TRACKING_CLUSTER_PROFILES["dense_stop_motion_recovery_v13"],
        # Retain the last online steering pixels for the complete form-level
        # budget so the loop can keep producing observations after TAPIR loss.
        # Steering is replaced by shortest-path endpoint guidance while all
        # live navigation points are absent; stale pixels only define the
        # diagnostic crop shown in the video.
        "navigation_loss_coast_frames": 64,
        "endpoint_geodesic_loss_guard": True,
        "endpoint_guidance_after_cluster_loss": True,
        "endpoint_geodesic_arrival": True,
        "endpoint_geodesic_radius_m": 0.75,
        "endpoint_geodesic_stop_fraction": 0.50,
}

# Production RGB-only arrival profile.  It intentionally contains no metric
# distance, depth, pose, collision or navmesh thresholds.  Physical arrival is
# proposed only from the independent dense bottom cluster after repeated
# forward commands produced observable RGB change.  Habitat geometry is used
# later by the evaluator to label that proposal, never to change it.
TRACKING_CLUSTER_PROFILES["rgb_only_dense_stop_v1"] = {
    "navigation_rows": 3,
    "navigation_cols": 3,
    "stop_rows": 5,
    "stop_cols": 9,
    "navigation_x_range": (0.40, 0.60),
    "navigation_y_range": (0.40, 0.60),
    "stop_x_range": (0.32, 0.68),
    "stop_y_range": (0.84, 0.99),
    "arrival_visible_fraction": 0.50,
    "arrival_confirmation_frames": 3,
    "maximum_goal_loss_frames": 4,
    "terminal_all_cluster_loss_is_arrival": True,
    "minimum_forward_commands": 3,
    "minimum_rgb_motion_frames": 2,
    "rgb_motion_threshold": 2.0,
    "forward_commands_after_turn": 1,
    "turn_deadband_deg": 7.0,
    "navigation_loss_grace_frames": 3,
    # Forward-stall early stop.  A blocked agent (sliding against a wall)
    # renders near-identical consecutive frames, so repeated forward commands
    # with a tiny grayscale change are treated as no physical progress and the
    # hop ends as a non-arrival instead of burning the whole step budget.
    # Calibrated on hidden geometry of 25 OpenNav episodes (2004 forward
    # commands): displacement < 2 cm gives scores 0-8, free motion >= 20 cm
    # has a 5th percentile of 12.3; T=8/K=3 fired 16 times with no false stop.
    "stall_motion_threshold": 8.0,
    "stall_forward_frames": 3,
}


def snap_points_to_mask(points, mask):
    """Snap XY points to distinct allowed pixels."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("cannot snap point cluster: allowed mask is empty")
    allowed = np.stack([xs, ys], axis=1).astype(np.float32)
    snapped = []
    distances = []
    used = set()
    for point in np.asarray(points, np.float32):
        squared = ((allowed - point) ** 2).sum(axis=1)
        for nearest in np.argsort(squared):
            nearest = int(nearest)
            pixel = tuple(allowed[nearest].astype(int))
            if pixel not in used:
                used.add(pixel)
                break
        snapped.append(allowed[nearest])
        distances.append(float(np.sqrt(squared[nearest])))
    return np.asarray(snapped, np.float32), distances


def point_cluster(center, mask, rows=3, cols=3, spacing=6.0):
    """Create a 3x3 selection-anchor cluster on allowed pixels."""
    xs = center[0] + (np.arange(cols) - (cols - 1) / 2) * spacing
    ys = center[1] + (np.arange(rows) - (rows - 1) / 2) * spacing
    xx, yy = np.meshgrid(xs, ys)
    requested = np.stack([xx.ravel(), yy.ravel()], 1).astype(np.float32)
    snapped, distances = snap_points_to_mask(requested, mask)
    return snapped, requested, distances


def crop_goal(rgb, tracks, visible, output_size=(85, 64)):
    """Create the fixed horizontal-axis-symmetric point crop."""
    points = tracks[visible]
    if len(points) == 0:
        return None, None
    bottom = np.median(points, axis=0).astype(np.float32)
    height, width = rgb.shape[:2]
    frame_center = np.array(
        [(width - 1) / 2, (height - 1) / 2], np.float32)
    top = np.array(
        [bottom[0], 2 * frame_center[1] - bottom[1]], np.float32)
    axis = bottom - top
    axis_length = float(np.linalg.norm(axis))
    minimum_height = height / 3
    if axis_length < minimum_height:
        vertical_sign = 1.0 if bottom[1] >= frame_center[1] else -1.0
        direction = np.array([0.0, vertical_sign], np.float32)
        top = bottom - direction * minimum_height
        axis = bottom - top
        axis_length = minimum_height
    direction = axis / max(axis_length, 1e-6)
    perpendicular = np.array([-direction[1], direction[0]], np.float32)
    out_width, out_height = map(int, output_size)
    crop_width = axis_length * out_width / out_height
    half_width = perpendicular * (crop_width / 2)
    quad = np.stack([
        top - half_width, top + half_width,
        bottom + half_width, bottom - half_width,
    ]).astype(np.float32)
    destination = np.array([
        [0, 0], [out_width - 1, 0],
        [out_width - 1, out_height - 1], [0, out_height - 1],
    ], np.float32)
    transform = cv2.getPerspectiveTransform(quad, destination)
    crop = cv2.warpPerspective(
        rgb, transform, (out_width, out_height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE)
    geometry = {
        "quad_xy": quad.tolist(),
        "top_midpoint_xy": top.tolist(),
        "bottom_midpoint_xy": bottom.tolist(),
        "axis_length_px": axis_length,
        "minimum_height_px": minimum_height,
        "output_size_wh": [out_width, out_height],
        "aspect_ratio": out_width / out_height,
        "source_to_crop_transform": transform.tolist(),
    }
    return Image.fromarray(crop), geometry


def dual_crop_tracking_clusters(
        crop_geometry, allowed_mask, rows=3, cols=3, *, profile=None):
    """Build independent navigation-center and stopping-bottom clusters."""
    config = dict(TRACKING_CLUSTER_PROFILES["legacy_3x3"])
    if profile is not None:
        if isinstance(profile, str):
            if profile not in TRACKING_CLUSTER_PROFILES:
                raise ValueError(
                    f"unknown tracking-cluster profile {profile!r}; expected "
                    f"one of {sorted(TRACKING_CLUSTER_PROFILES)}")
            config.update(TRACKING_CLUSTER_PROFILES[profile])
        else:
            config.update(dict(profile))
    elif rows != 3 or cols != 3:
        config.update({
            "navigation_rows": int(rows), "navigation_cols": int(cols),
            "stop_rows": int(rows), "stop_cols": int(cols),
        })
    out_width, out_height = map(int, crop_geometry["output_size_wh"])
    transform = np.asarray(
        crop_geometry["source_to_crop_transform"], np.float32)
    crop_allowed = cv2.warpPerspective(
        allowed_mask.astype(np.uint8), transform, (out_width, out_height),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
        borderValue=0).astype(bool)

    # ``crop_goal`` deliberately puts the selected point cluster on the
    # crop's bottom edge.  With a high-image target (stairs/upper landing),
    # all segmented pixels can lie on or just beyond that edge. OpenCV's
    # inverse rasterisation may then omit the boundary pixel and produce an
    # empty mask even though the source selection is valid ground. Preserve
    # those real source pixels by forward-projecting them and splatting only
    # points within a one-pixel rasterisation tolerance. Downstream points
    # are still snapped back to the original ``allowed_mask``, so this does
    # not invent selectable ground.
    boundary_projection_used = False
    if not crop_allowed.any():
        source_y, source_x = np.nonzero(allowed_mask)
        if len(source_x):
            source_points = np.stack(
                [source_x, source_y], axis=1).astype(np.float32)
            projected = cv2.perspectiveTransform(
                source_points[None], transform)[0]
            tolerance = 1.0
            inside = (
                (projected[:, 0] >= -tolerance) &
                (projected[:, 0] <= (out_width - 1) + tolerance) &
                (projected[:, 1] >= -tolerance) &
                (projected[:, 1] <= (out_height - 1) + tolerance))
            if inside.any():
                raster = np.rint(projected[inside]).astype(int)
                raster[:, 0] = np.clip(raster[:, 0], 0, out_width - 1)
                raster[:, 1] = np.clip(raster[:, 1], 0, out_height - 1)
                crop_allowed[raster[:, 1], raster[:, 0]] = True
                boundary_projection_used = bool(crop_allowed.any())

    navigation_region = np.zeros((out_height, out_width), bool)
    x0 = int(math.floor(out_width / 3))
    x1 = int(math.ceil(2 * out_width / 3))
    y0 = int(math.floor(out_height / 3))
    y1 = int(math.ceil(2 * out_height / 3))
    navigation_region[y0:y1, x0:x1] = True
    stop_region = np.zeros((out_height, out_width), bool)
    stop_y0 = max(0, int(math.floor(0.84 * out_height)))
    stop_region[stop_y0:out_height, x0:x1] = True

    navigation_allowed = crop_allowed & navigation_region
    stop_allowed = crop_allowed & stop_region
    navigation_fallback = not navigation_allowed.any()
    stop_fallback = not stop_allowed.any()
    if not crop_allowed.any():
        raise ValueError(
            "selected crop contains no pixels from the selectable ground mask")
    if navigation_fallback:
        # Preserve the physical ground contract. If the preferred center
        # ninth contains no ground, use another ground pixel inside the crop;
        # never manufacture an unsegmented rectangular navigation cluster.
        navigation_allowed = crop_allowed
    if stop_fallback:
        # Likewise, a missing bottom-band surface changes the cluster's
        # placement but not its semantic membership: stopping points remain
        # on segmented ground instead of falling back to arbitrary pixels.
        stop_allowed = crop_allowed

    navigation_x = np.linspace(
        config["navigation_x_range"][0] * (out_width - 1),
        config["navigation_x_range"][1] * (out_width - 1),
        int(config["navigation_cols"]))
    navigation_y = np.linspace(
        config["navigation_y_range"][0] * (out_height - 1),
        config["navigation_y_range"][1] * (out_height - 1),
        int(config["navigation_rows"]))
    xx, yy = np.meshgrid(navigation_x, navigation_y)
    navigation_requested = np.stack(
        [xx.ravel(), yy.ravel()], 1).astype(np.float32)
    navigation_crop, navigation_crop_snap = snap_points_to_mask(
        navigation_requested, navigation_allowed)

    stop_x = np.linspace(
        config["stop_x_range"][0] * (out_width - 1),
        config["stop_x_range"][1] * (out_width - 1),
        int(config["stop_cols"]))
    stop_y = np.linspace(
        config["stop_y_range"][0] * (out_height - 1),
        config["stop_y_range"][1] * (out_height - 1),
        int(config["stop_rows"]))
    xx, yy = np.meshgrid(stop_x, stop_y)
    stop_requested = np.stack([xx.ravel(), yy.ravel()], 1).astype(np.float32)
    stop_crop, stop_crop_snap = snap_points_to_mask(
        stop_requested, stop_allowed)

    crop_to_source = np.linalg.inv(transform)

    def to_source(points, enforce_allowed):
        mapped = cv2.perspectiveTransform(points[None], crop_to_source)[0]
        if enforce_allowed:
            snapped, source_snap = snap_points_to_mask(mapped, allowed_mask)
        else:
            snapped = mapped.copy()
            snapped[:, 0] = np.clip(
                snapped[:, 0], 0, allowed_mask.shape[1] - 1)
            snapped[:, 1] = np.clip(
                snapped[:, 1], 0, allowed_mask.shape[0] - 1)
            source_snap = np.linalg.norm(snapped - mapped, axis=1).tolist()
        return snapped, mapped, source_snap

    navigation_source, navigation_mapped, navigation_source_snap = to_source(
        navigation_crop, enforce_allowed=True)
    stop_source, stop_mapped, stop_source_snap = to_source(
        stop_crop, enforce_allowed=True)
    metadata = {
        "cluster_profile": config,
        "navigation_region_crop_xyxy": [x0, y0, x1, y1],
        "stop_region_crop_xyxy": [x0, stop_y0, x1, out_height],
        "navigation_requested_crop_xy": navigation_requested.tolist(),
        "navigation_ground_snapped_crop_xy": navigation_crop.tolist(),
        "navigation_crop_snap_distance_px": navigation_crop_snap,
        "navigation_mapped_source_xy": navigation_mapped.tolist(),
        "navigation_source_snap_distance_px": navigation_source_snap,
        "navigation_region_ground_fallback": navigation_fallback,
        "navigation_source_ground_snapped": True,
        "stop_requested_crop_xy": stop_requested.tolist(),
        "stop_ground_snapped_crop_xy": stop_crop.tolist(),
        "stop_crop_snap_distance_px": stop_crop_snap,
        "stop_mapped_source_xy": stop_mapped.tolist(),
        "stop_source_snap_distance_px": stop_source_snap,
        "stop_region_ground_fallback": stop_fallback,
        "stop_source_ground_snapped": True,
        "boundary_projection_used": boundary_projection_used,
    }
    return navigation_source, stop_source, metadata


def policy_heading(trajectories):
    trajectory = np.asarray(trajectories)
    waypoint = trajectory[0, min(1, trajectory.shape[1] - 1)]
    return math.atan2(float(waypoint[1]), max(float(waypoint[0]), 1e-4))


def _wrap_angle(value):
    return (value + math.pi) % (2 * math.pi) - math.pi


def _set_pose(sim, position, yaw):
    state = sim.get_agent(0).get_state()
    state.position = np.asarray(position, np.float32)
    state.rotation = quat_from_angle_axis(
        yaw, np.array([0.0, 1.0, 0.0]))
    sim.get_agent(0).set_state(state)


def _observe(sim):
    return sim.get_sensor_observations()["rgb"][..., :3]


@dataclass
class PointNavigationRequest:
    """External selection result and execution context for one point target."""

    rgb: np.ndarray
    selected_point_xy: np.ndarray
    selectable_mask: np.ndarray
    yaw: float
    position_history: list
    instruction: str = "point navigation"
    # Optional full/stage instruction pair used only by the video composer.
    # ``instruction`` remains the backwards-compatible fallback for callers
    # that do not provide a decomposition.
    full_instruction: Optional[str] = None
    sub_instruction: Optional[str] = None
    semantic_target: str = "selected point"
    target_index: int = 0
    stage_count: int = 1
    global_step: int = 0
    # Optional immutable raw segmentation mask. When supplied, the executor
    # independently intersects the caller's semantic/selectable mask with it
    # before creating any TAPIR cluster.
    ground_mask: Optional[np.ndarray] = None
    selected_point_depth_m: Optional[float] = None
    selected_point_reachable: Optional[bool] = None
    # Start-time path length to the selected navmesh projection.  This is
    # available from the selected point itself and is not a post-hoc label.
    selected_point_initial_geodesic_m: Optional[float] = None
    # Optional online navmesh projection of the selected RGB floor point.  It
    # is supplied by the Habitat adapter, never by the VLM.  The executor may
    # use the final online geodesic to the *selected point itself* as a
    # physical-arrival witness when dense stop tracks disappear at the control
    # budget; this does not expose a demonstration path or semantic target.
    selected_point_navmesh_xyz: Optional[np.ndarray] = None
    # Optional controller-side segment cap.  Reaching it is a neutral stop,
    # not a point-arrival signal; node backtracking uses this to traverse one
    # reverse route breadcrumb without overshooting a stored node.
    max_travel_distance_m: Optional[float] = None
    allow_initial_near_field_arrival: bool = True
    # Online capability contract.  Production R2R runs must use
    # ``rgb_only_v1``.  The legacy value exists only so historical geometry
    # ablations and their frozen unit tests remain readable.
    policy_input_contract: str = "legacy_rgbd_geometry"
    # Hidden diagnostic context.  The reference path is only projected into
    # rendered frames after control decisions; it is never used by the policy,
    # tracker, or arrival rule.
    reference_path: Optional[list] = None
    reference_path_index: int = 0


@dataclass
class PointNavigationResult:
    """Executor result. Callers should branch on ``arrived`` or ``signal``."""

    arrived: bool
    signal: Optional[str]
    end_reason: str
    final_rgb: np.ndarray
    final_yaw: float
    next_global_step: int
    action_history: list
    record: dict
    edge_keyframes: list[np.ndarray]


class PointNavigationExecutor:
    """Execute navigation to a caller-selected image point.

    Point selection is intentionally outside this class. Internally it creates
    the goal crop, initializes the navigation/stopping TAPIR clusters, runs the
    low-level image-goal policy, and emits an explicit arrival signal only when
    the dense crop-bottom arrival cluster satisfies the configured temporal
    disappearance rule. A caller may reject an initially unreachable projected
    target through ``selected_point_reachable=False``; this is an input-validity
    check, never a hidden post-navigation distance label.
    """

    def __init__(
            self, sim, tracker, policy, policy_config, policy_name,
            predict_fn: Callable[..., Any], device="cuda:0", output_dir=None,
            video_sink=None, video_composer=None, max_steps=14,
            forward_step=0.22, turn_step_deg=15.0, seed=17,
            tracking_cluster_profile="legacy_3x3", edge_keyframe_count=5,
            arrival_tracker=None):
        self.sim = sim
        self.tracker = tracker
        self.arrival_tracker = arrival_tracker
        self.policy = policy
        self.policy_config = policy_config
        self.policy_name = str(policy_name)
        self.predict_fn = predict_fn
        self.device = torch.device(device)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.video_sink = video_sink
        self.video_composer = video_composer
        self.max_steps = int(max_steps)
        self.forward_step = float(forward_step)
        self.turn_limit = math.radians(float(turn_step_deg))
        self.seed = int(seed)
        if tracking_cluster_profile not in TRACKING_CLUSTER_PROFILES:
            raise ValueError(
                f"unknown tracking-cluster profile "
                f"{tracking_cluster_profile!r}")
        self.tracking_cluster_profile = str(tracking_cluster_profile)
        self.cluster_config = dict(
            TRACKING_CLUSTER_PROFILES[self.tracking_cluster_profile])
        self.edge_keyframe_count = max(2, int(edge_keyframe_count))
        if self.output_dir is not None:
            self.crops_dir = self.output_dir / "goal_crops"
            self.crops_dir.mkdir(parents=True, exist_ok=True)
            self.keyframes_dir = self.output_dir / "edge_keyframes"
            self.keyframes_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.crops_dir = None
            self.keyframes_dir = None

    def _append_video(self, frame, request, position, yaw, phase):
        if (request.policy_input_contract == "rgb_only_v1" and
                self.video_composer is not None and
                hasattr(self.video_composer, "emit")):
            self.video_composer.emit(
                frame, request.instruction, request.target_index, phase,
                full_instruction=(request.full_instruction or request.instruction),
                sub_instruction=(request.sub_instruction or request.instruction))
            return
        if self.video_sink is None:
            return
        if request.reference_path:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            camera_position = (np.asarray(position, np.float32) +
                               np.array([0.0, 1.25, 0.0], np.float32))
            projection = project_reference_path(
                request.reference_path,
                range(int(request.reference_path_index),
                      len(request.reference_path)),
                camera_position, float(yaw), rgb.shape[1], rgb.shape[0])
            rgb = draw_reference_path_overlay(rgb, projection)
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if self.video_composer is not None:
            frame = self.video_composer.compose(
                frame, request.position_history, position, yaw,
                request.instruction, request.target_index, phase,
                full_instruction=(request.full_instruction
                                  or request.instruction),
                sub_instruction=(request.sub_instruction
                                 or request.instruction))
        self.video_sink.append(frame)

    def _save_crop(self, crop, name):
        if self.crops_dir is None:
            return None
        path = self.crops_dir / name
        crop.save(path)
        return str(path.relative_to(self.output_dir))

    def _select_and_save_keyframes(self, frames, frame_metadata, target_index):
        count = min(self.edge_keyframe_count, len(frames))
        indices = np.unique(np.round(
            np.linspace(0, len(frames) - 1, count)).astype(int)).tolist()
        selected = [np.asarray(frames[index], np.uint8) for index in indices]
        records = []
        for order, (index, frame) in enumerate(zip(indices, selected)):
            item = dict(frame_metadata[index])
            item.update({"keyframe_index": order, "source_frame_index": index})
            if self.keyframes_dir is not None:
                path = self.keyframes_dir / (
                    f"target_{target_index:02d}_keyframe_{order:02d}.jpg")
                Image.fromarray(frame).save(path, quality=92)
                item["image_path"] = str(path.relative_to(self.output_dir))
            records.append(item)
        return selected, records

    @staticmethod
    def _rgb_motion_score(before, after):
        """Robust image-only ego-motion witness (mean grayscale L1)."""
        before_gray = cv2.cvtColor(
            np.asarray(before, np.uint8)[..., :3], cv2.COLOR_RGB2GRAY)
        after_gray = cv2.cvtColor(
            np.asarray(after, np.uint8)[..., :3], cv2.COLOR_RGB2GRAY)
        return float(np.mean(np.abs(
            after_gray.astype(np.float32) - before_gray.astype(np.float32))))

    def _execute_rgb_only(self, request: PointNavigationRequest):
        """Execute a selected image point with RGB and action history only.

        The simulator argument is a capability-limited facade: this method can
        render RGB and issue discrete actions, but cannot query pose, depth,
        collision, pathfinder or navmesh.  Consequently every action and every
        arrival decision below is reproducible from RGB frames plus commanded
        action tokens.
        """
        from rgb_only_runtime import require_rgb_only_policy_sim

        sim = require_rgb_only_policy_sim(self.sim)
        forbidden = {
            "selected_point_depth_m": request.selected_point_depth_m,
            "selected_point_reachable": request.selected_point_reachable,
            "selected_point_initial_geodesic_m": (
                request.selected_point_initial_geodesic_m),
            "selected_point_navmesh_xyz": request.selected_point_navmesh_xyz,
            "max_travel_distance_m": request.max_travel_distance_m,
            "reference_path": request.reference_path,
        }
        leaked = [name for name, value in forbidden.items()
                  if value is not None]
        if leaked:
            raise ValueError(
                "rgb_only_v1 request contains privileged policy fields: " +
                ", ".join(leaked))
        if self.tracking_cluster_profile != "rgb_only_dense_stop_v1":
            raise ValueError(
                "rgb_only_v1 requires tracking profile rgb_only_dense_stop_v1")

        rgb = np.asarray(request.rgb, np.uint8)[..., :3]
        requested_selectable_mask = np.asarray(request.selectable_mask, bool)
        selectable_mask = requested_selectable_mask.copy()
        if request.ground_mask is not None:
            ground_mask = np.asarray(request.ground_mask, bool)
            if ground_mask.shape != selectable_mask.shape:
                raise ValueError(
                    "ground_mask and selectable_mask must have identical shapes")
            selectable_mask &= ground_mask
        if not selectable_mask.any():
            raise ValueError(
                "point-navigation selectable mask has no segmented RGB ground")

        anchor_points, requested_points, anchor_snap_distances = point_cluster(
            np.asarray(request.selected_point_xy, np.float32), selectable_mask)
        initial_crop, initial_geometry = crop_goal(
            rgb, anchor_points, np.ones(len(anchor_points), dtype=bool),
            tuple(self.policy_config["image_size"]))
        initial_crop_path = self._save_crop(
            initial_crop, f"target_{request.target_index:02d}_initial.jpg")
        navigation_points, stop_points, dual_geometry = (
            dual_crop_tracking_clusters(
                initial_geometry, selectable_mask,
                profile=self.tracking_cluster_profile))
        navigation_count = len(navigation_points)
        stop_count = len(stop_points)
        _, goal_points, _ = dual_crop_tracking_clusters(
            initial_geometry, selectable_mask, profile="legacy_3x3")
        goal_count = len(goal_points)
        separate_arrival_tracker = self.arrival_tracker is not None
        tracker_points = np.concatenate(
            [navigation_points, goal_points], axis=0).astype(np.float32)
        tracks, visible = self.tracker.reset(rgb, tracker_points)
        if separate_arrival_tracker:
            stop_tracks, stop_visible = self.arrival_tracker.reset(
                rgb, stop_points)
        else:
            # This fallback remains RGB-only but the production constructor is
            # expected to provide the independent tracker state.
            combined = np.concatenate(
                [navigation_points, goal_points, stop_points], axis=0)
            tracks, visible = self.tracker.reset(rgb, combined)

        context = deque(maxlen=self.policy_config["context_size"] + 1)
        for _ in range(self.policy_config["context_size"] + 1):
            context.append(Image.fromarray(rgb))
        action_heading_rad = float(request.yaw)
        global_step = int(request.global_step)
        action_history = []
        edge_frames = [rgb.copy()]
        edge_frame_metadata = [{
            "step": -1, "phase": "edge_start", "action": None,
            "policy_input_contract": "rgb_only_v1",
        }]
        dense_stop_low_streak = 0
        goal_loss_streak = 0
        navigation_loss_streak = 0
        forward_commands = 0
        forward_since_turn = 0
        rgb_motion_frames = 0
        stall_forward_streak = 0
        stall_motion_scores = []
        stall_motion_threshold = float(
            self.cluster_config.get("stall_motion_threshold", 0.0))
        stall_forward_frames = int(
            self.cluster_config.get("stall_forward_frames", 0))
        last_navigation_tracks = None
        last_navigation_visible = None
        record = {
            "policy_input_contract": "rgb_only_v1",
            "policy_observations": ["rgb", "commanded_action_history"],
            "privileged_inputs_used": [],
            "initial_point_xy": np.asarray(
                request.selected_point_xy, np.float32).tolist(),
            "raw_ground_mask_provided": request.ground_mask is not None,
            "selectable_pixels_before_ground_intersection": int(
                requested_selectable_mask.sum()),
            "selectable_pixels_after_ground_intersection": int(
                selectable_mask.sum()),
            "initial_cluster_requested_xy": requested_points.tolist(),
            "initial_cluster_ground_snapped_xy": anchor_points.tolist(),
            "initial_cluster_snap_distance_px": anchor_snap_distances,
            "all_initial_cluster_points_on_ground": all(
                selectable_mask[int(point[1]), int(point[0])]
                for point in anchor_points),
            "initial_crop": initial_crop_path,
            "initial_crop_geometry": initial_geometry,
            "dual_cluster_geometry": dual_geometry,
            "navigation_cluster_initial_xy": navigation_points.tolist(),
            "navigation_cluster_size": navigation_count,
            "stop_cluster_initial_xy": stop_points.tolist(),
            "stop_cluster_size": stop_count,
            "goal_cluster_initial_xy": goal_points.tolist(),
            "goal_cluster_size": goal_count,
            "separate_arrival_tracker": separate_arrival_tracker,
            "tracking_cluster_profile": self.tracking_cluster_profile,
            "arrival_visible_fraction_threshold": float(
                self.cluster_config["arrival_visible_fraction"]),
            "arrival_confirmation_frames": int(
                self.cluster_config["arrival_confirmation_frames"]),
            "arrival_rule": (
                "dense RGB stop-cluster loss + forward action history + "
                "RGB ego-motion confirmation"),
            "stall_motion_threshold": stall_motion_threshold,
            "stall_forward_frames": stall_forward_frames,
            "steps": [], "end_reason": None, "arrival_signal": None,
            "terminal_navigation_visible_fraction": None,
            "terminal_stop_visible_fraction": None,
        }

        for local_step in range(self.max_steps):
            navigation_tracks = tracks[:navigation_count]
            navigation_visible = visible[:navigation_count]
            goal_start = navigation_count
            goal_tracks = tracks[goal_start:goal_start + goal_count]
            goal_visible = visible[goal_start:goal_start + goal_count]
            if not separate_arrival_tracker:
                stop_start = goal_start + goal_count
                stop_tracks = tracks[stop_start:stop_start + stop_count]
                stop_visible = visible[stop_start:stop_start + stop_count]

            navigation_fraction = float(navigation_visible.mean())
            goal_fraction = float(goal_visible.mean())
            stop_fraction = float(stop_visible.mean())
            if navigation_visible.any():
                navigation_loss_streak = 0
                last_navigation_tracks = navigation_tracks.copy()
                last_navigation_visible = navigation_visible.copy()
            else:
                navigation_loss_streak += 1

            control_tracks = navigation_tracks
            control_visible = navigation_visible
            control_source = "navigation"
            grace = int(self.cluster_config.get(
                "navigation_loss_grace_frames", 0))
            if not control_visible.any() and navigation_loss_streak <= grace:
                if goal_visible.any():
                    control_tracks, control_visible = goal_tracks, goal_visible
                    control_source = "goal_loss_grace"
                elif stop_visible.any():
                    control_tracks, control_visible = stop_tracks, stop_visible
                    control_source = "stop_loss_grace"
                elif last_navigation_tracks is not None:
                    control_tracks = last_navigation_tracks
                    control_visible = last_navigation_visible
                    control_source = "last_rgb_track_grace"

            dense_stop_low = bool(
                stop_fraction <= float(
                    self.cluster_config["arrival_visible_fraction"]))
            goal_lost = bool(not goal_visible.any())
            dense_stop_low_streak = (
                dense_stop_low_streak + 1 if dense_stop_low else 0)
            goal_loss_streak = goal_loss_streak + 1 if goal_lost else 0
            minimum_forward = int(
                self.cluster_config["minimum_forward_commands"])
            minimum_motion = int(
                self.cluster_config["minimum_rgb_motion_frames"])
            forward_after_turn = int(
                self.cluster_config["forward_commands_after_turn"])
            action_progress = bool(
                forward_commands >= minimum_forward and
                forward_since_turn >= forward_after_turn and
                rgb_motion_frames >= minimum_motion)
            ordinary_arrival = bool(
                local_step > 0 and goal_lost and
                dense_stop_low_streak >= int(
                    self.cluster_config["arrival_confirmation_frames"]) and
                action_progress)
            terminal_consensus = bool(
                local_step > 0 and
                self.cluster_config.get(
                    "terminal_all_cluster_loss_is_arrival", False) and
                not navigation_visible.any() and goal_lost and
                dense_stop_low and action_progress)
            if ordinary_arrival or terminal_consensus:
                record["arrival_signal"] = POINT_NAVIGATION_ARRIVED
                record["end_reason"] = "rgb_only_dense_stop_cluster_arrival"
                record["terminal_cluster_consensus"] = terminal_consensus
                record["terminal_navigation_visible_fraction"] = (
                    navigation_fraction)
                record["terminal_stop_visible_fraction"] = stop_fraction
                record["terminal_forward_commands"] = forward_commands
                record["terminal_rgb_motion_frames"] = rgb_motion_frames
                terminal_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    terminal_frame,
                    f"RGB ARRIVED N {int(navigation_visible.sum())}/{navigation_count} "
                    f"G {int(goal_visible.sum())}/{goal_count} "
                    f"A {int(stop_visible.sum())}/{stop_count}",
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 0), 2)
                self._append_video(
                    terminal_frame, request, None, action_heading_rad,
                    "rgb_only_stop_cluster_arrived")
                break
            if not control_visible.any():
                record["end_reason"] = "rgb_navigation_cluster_lost"
                record["terminal_navigation_visible_fraction"] = 0.0
                record["terminal_stop_visible_fraction"] = stop_fraction
                break

            crop_tracks = goal_tracks if goal_visible.any() else control_tracks
            crop_visible = goal_visible if goal_visible.any() else control_visible
            goal, crop_geometry = crop_goal(
                rgb, crop_tracks, crop_visible,
                tuple(self.policy_config["image_size"]))
            if goal is None:
                record["end_reason"] = "no_rgb_crop"
                break
            goal_path = self._save_crop(
                goal,
                f"target_{request.target_index:02d}_step_{local_step:03d}.jpg")
            start = time.perf_counter()
            distance, trajectories = self.predict_fn(
                self.policy, self.policy_config, list(context), goal,
                self.policy_name, self.device, samples=4,
                seed=self.seed + global_step)
            inference_seconds = time.perf_counter() - start
            centroid = np.median(control_tracks[control_visible], axis=0)
            width = rgb.shape[1]
            pixel_angle = math.atan(
                (centroid[0] - width / 2) / (0.8 * width))
            navigation_angle = policy_heading(trajectories)
            desired_turn = (
                0.8 * pixel_angle +
                0.2 * np.clip(navigation_angle, -0.6, 0.6))
            deadband = math.radians(float(
                self.cluster_config["turn_deadband_deg"]))
            if desired_turn > deadband:
                action = "turn_right"
                commanded_turn_deg = -math.degrees(self.turn_limit)
            elif desired_turn < -deadband:
                action = "turn_left"
                commanded_turn_deg = math.degrees(self.turn_limit)
            else:
                action = "move_forward"
                commanded_turn_deg = 0.0

            observations = sim.step(action)
            next_rgb = np.asarray(observations["rgb"], np.uint8)[..., :3]
            motion_score = self._rgb_motion_score(rgb, next_rgb)
            if action == "move_forward":
                forward_commands += 1
                forward_since_turn += 1
                if motion_score >= float(
                        self.cluster_config["rgb_motion_threshold"]):
                    rgb_motion_frames += 1
                # Turns are deliberately left out: a turn that frees the agent
                # shows up as a high-motion forward on the next step anyway.
                if motion_score < stall_motion_threshold:
                    stall_forward_streak += 1
                    stall_motion_scores.append(motion_score)
                else:
                    stall_forward_streak = 0
                    stall_motion_scores = []
            else:
                forward_since_turn = 0
                action_heading_rad = _wrap_angle(
                    action_heading_rad + math.radians(commanded_turn_deg))
            forward_stalled = bool(
                stall_forward_frames > 0 and
                stall_forward_streak >= stall_forward_frames)

            action_record = {
                "step": local_step,
                "action": action,
                "commanded_turn_deg": commanded_turn_deg,
                "forward_commanded": action == "move_forward",
                "rgb_motion_score": motion_score,
                "rgb_motion_threshold": float(
                    self.cluster_config["rgb_motion_threshold"]),
                "stall_forward_streak": stall_forward_streak,
                "policy_input_contract": "rgb_only_v1",
            }
            action_history.append(action_record)
            edge_frames.append(next_rgb.copy())
            edge_frame_metadata.append({
                "step": local_step, "phase": "edge_motion",
                "action": action, "commanded_turn_deg": commanded_turn_deg,
                "rgb_motion_score": motion_score,
                "policy_input_contract": "rgb_only_v1",
            })

            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for point, ok in zip(navigation_tracks, navigation_visible):
                if ok:
                    cv2.circle(frame, tuple(np.round(point).astype(int)), 3,
                               (0, 255, 255), -1)
            for point, ok in zip(goal_tracks, goal_visible):
                if ok:
                    cv2.drawMarker(frame, tuple(np.round(point).astype(int)),
                                   (255, 255, 0), cv2.MARKER_CROSS, 7, 1)
            for point, ok in zip(stop_tracks, stop_visible):
                if ok:
                    cv2.drawMarker(frame, tuple(np.round(point).astype(int)),
                                   (255, 0, 255), cv2.MARKER_DIAMOND, 8, 1)
            cv2.putText(
                frame, f"RGB-ONLY {action} motion {motion_score:.1f}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                (255, 255, 255), 1)
            self._append_video(
                frame, request, None, action_heading_rad,
                "rgb_only_point_navigation")

            record["steps"].append({
                "global_step": global_step,
                "navigation_visible_fraction": navigation_fraction,
                "goal_visible_fraction": goal_fraction,
                "stop_visible_fraction": stop_fraction,
                "navigation_tracks_xy": navigation_tracks.tolist(),
                "navigation_visible": navigation_visible.tolist(),
                "goal_tracks_xy": goal_tracks.tolist(),
                "goal_visible": goal_visible.tolist(),
                "stop_tracks_xy": stop_tracks.tolist(),
                "stop_visible": stop_visible.tolist(),
                "navigation_control_source": control_source,
                "crop_geometry": crop_geometry,
                "goal_crop": goal_path,
                "policy_distance": distance,
                "policy_first_trajectory": np.asarray(
                    trajectories)[0].tolist(),
                "pixel_heading_rad": pixel_angle,
                "policy_heading_rad": navigation_angle,
                "action": action,
                "commanded_turn_deg": commanded_turn_deg,
                "rgb_motion_score": motion_score,
                "stall_forward_streak": stall_forward_streak,
                "policy_inference_sec": inference_seconds,
            })
            global_step += 1
            rgb = next_rgb
            context.append(Image.fromarray(rgb))
            if forward_stalled:
                record["end_reason"] = "rgb_forward_stall"
                record["terminal_stall_forward_streak"] = stall_forward_streak
                record["terminal_stall_motion_scores"] = list(
                    stall_motion_scores)
                record["terminal_navigation_visible_fraction"] = (
                    navigation_fraction)
                record["terminal_stop_visible_fraction"] = stop_fraction
                terminal_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    terminal_frame,
                    f"RGB STALL {stall_forward_streak} fwd < {stall_motion_threshold:.0f}",
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 255), 2)
                self._append_video(
                    terminal_frame, request, None, action_heading_rad,
                    "rgb_only_forward_stall")
                break
            tracks, visible = self.tracker.step(rgb)
            if separate_arrival_tracker:
                stop_tracks, stop_visible = self.arrival_tracker.step(rgb)
        else:
            record["end_reason"] = "max_steps"
            record["terminal_navigation_visible_fraction"] = float(
                visible[:navigation_count].mean())
            record["terminal_stop_visible_fraction"] = float(
                stop_visible.mean())

        edge_keyframes, keyframe_records = self._select_and_save_keyframes(
            edge_frames, edge_frame_metadata, request.target_index)
        record["action_history"] = action_history
        record["edge_keyframes"] = keyframe_records
        arrived = record["arrival_signal"] == POINT_NAVIGATION_ARRIVED
        return PointNavigationResult(
            arrived=arrived, signal=record["arrival_signal"],
            end_reason=record["end_reason"], final_rgb=rgb,
            final_yaw=action_heading_rad, next_global_step=global_step,
            action_history=action_history, record=record,
            edge_keyframes=edge_keyframes)

    def execute(self, request: PointNavigationRequest) -> PointNavigationResult:
        """Run one point target and return an explicit arrival result."""
        if request.policy_input_contract == "rgb_only_v1":
            return self._execute_rgb_only(request)
        if request.policy_input_contract != "legacy_rgbd_geometry":
            raise ValueError(
                f"unknown policy_input_contract {request.policy_input_contract!r}")
        rgb = np.asarray(request.rgb)
        if request.selected_point_reachable is False:
            record = {
                "initial_point_xy": np.asarray(
                    request.selected_point_xy, np.float32).tolist(),
                "selected_point_reachable": False,
                "tracking_cluster_profile": self.tracking_cluster_profile,
                "steps": [],
                "action_history": [],
                "edge_keyframes": [],
                "end_reason": "selected_point_unreachable",
                "arrival_signal": None,
                "terminal_navigation_visible_fraction": None,
                "terminal_stop_visible_fraction": None,
            }
            return PointNavigationResult(
                arrived=False,
                signal=None,
                end_reason=record["end_reason"],
                final_rgb=rgb,
                final_yaw=float(request.yaw),
                next_global_step=int(request.global_step),
                action_history=[],
                record=record,
                edge_keyframes=[],
            )
        requested_selectable_mask = np.asarray(request.selectable_mask, bool)
        selectable_mask = requested_selectable_mask.copy()
        if request.ground_mask is not None:
            ground_mask = np.asarray(request.ground_mask, bool)
            if ground_mask.shape != selectable_mask.shape:
                raise ValueError(
                    "ground_mask and selectable_mask must have identical shapes")
            selectable_mask &= ground_mask
        if not selectable_mask.any():
            raise ValueError(
                "point-navigation selectable mask has no segmented ground")
        anchor_points, requested_points, anchor_snap_distances = point_cluster(
            np.asarray(request.selected_point_xy, np.float32), selectable_mask)
        initial_crop, initial_geometry = crop_goal(
            rgb, anchor_points, np.ones(len(anchor_points), dtype=bool),
            tuple(self.policy_config["image_size"]))
        initial_crop_path = self._save_crop(
            initial_crop, f"target_{request.target_index:02d}_initial.jpg")
        navigation_points, stop_points, dual_geometry = (
            dual_crop_tracking_clusters(
                initial_geometry, selectable_mask,
                profile=self.tracking_cluster_profile))
        navigation_count = len(navigation_points)
        stop_count = len(stop_points)
        shared_goal_and_arrival_cluster = bool(
            self.tracking_cluster_profile == "legacy_3x3")
        if shared_goal_and_arrival_cluster:
            goal_points = stop_points
        else:
            _, goal_points, _ = dual_crop_tracking_clusters(
                initial_geometry, selectable_mask, profile="legacy_3x3")
        goal_count = len(goal_points)
        separate_arrival_tracker = bool(
            not shared_goal_and_arrival_cluster and
            self.arrival_tracker is not None)
        tracker_groups = [navigation_points]
        if not shared_goal_and_arrival_cluster:
            tracker_groups.append(goal_points)
        if not separate_arrival_tracker:
            tracker_groups.append(stop_points)
        tracker_points = np.concatenate(tracker_groups, axis=0).astype(np.float32)
        tracks, visible = self.tracker.reset(rgb, tracker_points)
        if separate_arrival_tracker:
            stop_tracks, stop_visible = self.arrival_tracker.reset(
                rgb, stop_points)

        context = deque(maxlen=self.policy_config["context_size"] + 1)
        for _ in range(self.policy_config["context_size"] + 1):
            context.append(Image.fromarray(rgb))
        yaw = float(request.yaw)
        global_step = int(request.global_step)
        action_history = []
        edge_frames = [rgb.copy()]
        edge_frame_metadata = [{
            "step": -1, "phase": "edge_start", "action": None,
            "moved_m": 0.0, "cumulative_moved_m": 0.0,
        }]
        low_visibility_streak = 0
        dense_stop_low_streak = 0
        goal_loss_streak = 0
        blocked_forward_streak = 0
        stall_probe_count = 0
        navigation_loss_streak = 0
        last_navigation_tracks = None
        last_navigation_visible = None
        record = {
            "initial_point_xy": np.asarray(
                request.selected_point_xy, np.float32).tolist(),
            "raw_ground_mask_provided": request.ground_mask is not None,
            "selectable_pixels_before_ground_intersection": int(
                requested_selectable_mask.sum()),
            "selectable_pixels_after_ground_intersection": int(
                selectable_mask.sum()),
            "selected_point_reachable": request.selected_point_reachable,
            "selected_point_depth_m": request.selected_point_depth_m,
            "selected_point_initial_geodesic_m": (
                request.selected_point_initial_geodesic_m),
            "max_travel_distance_m": request.max_travel_distance_m,
            "allow_initial_near_field_arrival": bool(
                request.allow_initial_near_field_arrival),
            "initial_cluster_requested_xy": requested_points.tolist(),
            "initial_cluster_ground_snapped_xy": anchor_points.tolist(),
            "initial_cluster_snap_distance_px": anchor_snap_distances,
            "all_initial_cluster_points_on_ground": all(
                selectable_mask[int(point[1]), int(point[0])]
                for point in anchor_points),
            "initial_crop": initial_crop_path,
            "initial_crop_geometry": initial_geometry,
            "dual_cluster_geometry": dual_geometry,
            "navigation_cluster_initial_xy": navigation_points.tolist(),
            "navigation_cluster_size": navigation_count,
            "all_navigation_cluster_points_on_ground": all(
                selectable_mask[int(point[1]), int(point[0])]
                for point in navigation_points),
            "stop_cluster_initial_xy": stop_points.tolist(),
            "stop_cluster_size": stop_count,
            "all_stop_cluster_points_on_ground": all(
                selectable_mask[int(point[1]), int(point[0])]
                for point in stop_points),
            "goal_cluster_initial_xy": goal_points.tolist(),
            "goal_cluster_size": goal_count,
            "goal_cluster_shared_with_arrival_cluster": (
                shared_goal_and_arrival_cluster),
            "separate_arrival_tracker": separate_arrival_tracker,
            "steps": [],
            "end_reason": None,
            "arrival_signal": None,
            "terminal_navigation_visible_fraction": None,
            "terminal_stop_visible_fraction": None,
            "tracking_cluster_profile": self.tracking_cluster_profile,
            "arrival_visible_fraction_threshold": float(
                self.cluster_config["arrival_visible_fraction"]),
            "arrival_confirmation_frames": int(
                self.cluster_config["arrival_confirmation_frames"]),
            "maximum_goal_loss_frames": int(
                self.cluster_config["maximum_goal_loss_frames"]),
            "initial_near_field_depth_m": (
                self.cluster_config["initial_near_field_depth_m"]),
            "initial_near_field_min_y_fraction": (
                self.cluster_config["initial_near_field_min_y_fraction"]),
            "terminal_all_cluster_loss_is_arrival": bool(
                self.cluster_config.get(
                    "terminal_all_cluster_loss_is_arrival", False)),
            "minimum_travel_to_initial_depth_fraction": (
                self.cluster_config.get(
                    "minimum_travel_to_initial_depth_fraction")),
            "use_initial_geodesic_for_motion_guard": bool(
                self.cluster_config.get(
                    "use_initial_geodesic_for_motion_guard", False)),
            "stall_recovery_trigger_frames": int(
                self.cluster_config.get("stall_recovery_trigger_frames", 0)),
            "stall_recovery_turn_deg": float(
                self.cluster_config.get("stall_recovery_turn_deg", 0.0)),
            "stall_recovery_max_probes": int(
                self.cluster_config.get("stall_recovery_max_probes", 0)),
            "navigation_loss_grace_frames": int(
                self.cluster_config.get("navigation_loss_grace_frames", 0)),
        }

        for local_step in range(self.max_steps):
            navigation_tracks = tracks[:navigation_count]
            navigation_visible = visible[:navigation_count]
            if shared_goal_and_arrival_cluster:
                goal_tracks = tracks[
                    navigation_count:navigation_count + goal_count]
                goal_visible = visible[
                    navigation_count:navigation_count + goal_count]
                stop_tracks = goal_tracks
                stop_visible = goal_visible
            else:
                goal_start = navigation_count
                stop_start = goal_start + goal_count
                goal_tracks = tracks[goal_start:stop_start]
                goal_visible = visible[goal_start:stop_start]
                if not separate_arrival_tracker:
                    stop_tracks = tracks[stop_start:stop_start + stop_count]
                    stop_visible = visible[stop_start:stop_start + stop_count]
            navigation_fraction = float(navigation_visible.mean())
            goal_fraction = float(goal_visible.mean())
            stop_fraction = float(stop_visible.mean())

            # TAPIR can drop the central navigation cluster for one or two
            # frames at a doorway/turn while the independently tracked goal or
            # dense stop cluster is still visible.  Keep the original
            # navigation fractions for scoring, but use a bounded visible
            # fallback for control instead of terminating immediately.  This
            # is intentionally limited to currently visible tracks; it never
            # extrapolates from a stale point or hidden reference path.
            if navigation_visible.any():
                navigation_loss_streak = 0
                last_navigation_tracks = navigation_tracks.copy()
                last_navigation_visible = navigation_visible.copy()
            else:
                navigation_loss_streak += 1
            control_navigation_tracks = navigation_tracks
            control_navigation_visible = navigation_visible
            control_navigation_source = "navigation"
            grace_frames = int(self.cluster_config.get(
                "navigation_loss_grace_frames", 0))
            if (not navigation_visible.any() and
                    navigation_loss_streak <= grace_frames):
                if goal_visible.any():
                    control_navigation_tracks = goal_tracks
                    control_navigation_visible = goal_visible
                    control_navigation_source = "goal_loss_grace"
                elif stop_visible.any():
                    control_navigation_tracks = stop_tracks
                    control_navigation_visible = stop_visible
                    control_navigation_source = "stop_loss_grace"

            # If every currently tracked point is temporarily absent, keep a
            # bounded online-only coast window.  The retained pixels are from
            # the immediately preceding observation and are never used after
            # the configured window; this allows the robot to cross a short
            # doorway/occlusion and gives TAPIR a chance to reacquire.
            coast_frames = int(self.cluster_config.get(
                "navigation_loss_coast_frames", 0))
            if (not control_navigation_visible.any() and
                    coast_frames > 0 and last_navigation_tracks is not None and
                    navigation_loss_streak <= grace_frames + coast_frames):
                control_navigation_tracks = last_navigation_tracks
                control_navigation_visible = last_navigation_visible
                control_navigation_source = "navigation_loss_coast"

            goal_visibility_low = bool(not goal_visible.any())
            dense_stop_visibility_low = bool(
                stop_fraction <=
                float(self.cluster_config["arrival_visible_fraction"]))
            stop_visibility_low = bool(
                goal_visibility_low and dense_stop_visibility_low)
            goal_loss_streak = (
                goal_loss_streak + 1 if goal_visibility_low else 0)
            low_visibility_streak = (
                low_visibility_streak + 1 if stop_visibility_low else 0)
            dense_stop_low_streak = (
                dense_stop_low_streak + 1
                if dense_stop_visibility_low else 0)
            arrival_confirmed = bool(
                local_step > 0 and goal_visibility_low and (
                    low_visibility_streak >= int(
                        self.cluster_config["arrival_confirmation_frames"])
                    or goal_loss_streak >= int(
                        self.cluster_config["maximum_goal_loss_frames"])))
            navigation_loss_stop_limit = self.cluster_config.get(
                "navigation_loss_arrival_stop_fraction")
            if (arrival_confirmed and navigation_loss_stop_limit is not None and
                    not navigation_visible.any() and
                    stop_fraction > float(navigation_loss_stop_limit)):
                arrival_confirmed = False
            terminal_cluster_consensus = bool(
                local_step > 0 and
                self.cluster_config.get(
                    "terminal_all_cluster_loss_is_arrival", False) and
                not navigation_visible.any() and goal_visibility_low and
                dense_stop_visibility_low and
                (navigation_loss_stop_limit is None or
                 stop_fraction <= float(navigation_loss_stop_limit)))
            near_depth = self.cluster_config["initial_near_field_depth_m"]
            near_y = self.cluster_config["initial_near_field_min_y_fraction"]
            near_geodesic = self.cluster_config.get(
                "initial_near_field_geodesic_m")
            near_image_witness = bool(
                near_depth is not None and near_y is not None and
                request.selected_point_depth_m is not None and
                float(request.selected_point_depth_m) <= float(near_depth) and
                float(request.selected_point_xy[1]) >=
                float(near_y) * rgb.shape[0])
            near_geodesic_witness = bool(
                near_geodesic is not None and
                request.selected_point_initial_geodesic_m is not None and
                math.isfinite(float(request.selected_point_initial_geodesic_m)) and
                float(request.selected_point_initial_geodesic_m) <=
                float(near_geodesic))
            initial_near_field_arrival = bool(
                request.allow_initial_near_field_arrival and
                local_step == 0 and near_depth is not None and near_y is not None and
                (near_image_witness or near_geodesic_witness))
            minimum_travel_fraction = self.cluster_config.get(
                "minimum_travel_to_initial_depth_fraction")
            cumulative_moved = float(sum(
                item["moved_m"] for item in action_history))
            depth_reference_m = (
                float(request.selected_point_depth_m)
                if request.selected_point_depth_m is not None and
                math.isfinite(float(request.selected_point_depth_m)) else None)
            geodesic_reference_m = (
                float(request.selected_point_initial_geodesic_m)
                if request.selected_point_initial_geodesic_m is not None and
                math.isfinite(float(request.selected_point_initial_geodesic_m))
                else None)
            # On stairs/slopes, camera-ray depth can be substantially longer
            # than the reachable path to the selected navmesh endpoint. When
            # the caller supplied that endpoint distance, it is the physical
            # quantity the arrival guard should use. Taking max(depth,
            # geodesic) caused the agent to reach its point, lose every dense
            # cluster, and then spin until timeout because an oblique RGB ray
            # could never satisfy the inflated travel threshold.
            if (self.cluster_config.get(
                    "use_initial_geodesic_for_motion_guard", False) and
                    geodesic_reference_m is not None):
                travel_reference_m = geodesic_reference_m
            else:
                travel_reference_m = depth_reference_m
            motion_guard_satisfied = bool(
                minimum_travel_fraction is None or
                travel_reference_m is None or
                cumulative_moved >= (
                    float(minimum_travel_fraction) *
                    travel_reference_m))
            dense_stop_only_arrival = bool(
                self.cluster_config.get("dense_stop_only_arrival", False) and
                local_step > 0 and
                dense_stop_low_streak >= int(
                    self.cluster_config["arrival_confirmation_frames"]) and
                navigation_fraction <= float(self.cluster_config.get(
                    "dense_stop_navigation_visible_fraction", 0.0)) and
                motion_guard_satisfied)
            online_endpoint_distance = None
            online_endpoint_waypoint = None
            online_endpoint_arrival = False
            if ((self.cluster_config.get(
                    "online_endpoint_geodesic_arrival", False) or
                 self.cluster_config.get(
                    "endpoint_geodesic_loss_guard", False)) and
                    request.selected_point_navmesh_xyz is not None):
                try:
                    endpoint = np.asarray(
                        request.selected_point_navmesh_xyz, np.float32)
                    current = np.asarray(
                        self.sim.get_agent(0).get_state().position,
                        np.float32)
                    endpoint_path = __import__("habitat_sim").ShortestPath()
                    endpoint_path.requested_start = current
                    endpoint_path.requested_end = endpoint
                    if (np.isfinite(endpoint).all() and
                            self.sim.pathfinder.find_path(endpoint_path)):
                        online_endpoint_distance = float(
                            endpoint_path.geodesic_distance)
                        path_points = list(getattr(
                            endpoint_path, "points", []) or [])
                        if path_points:
                            # Habitat returns the start as the first point.
                            # Use the first meaningfully displaced point so
                            # guidance follows bends in the navmesh route
                            # instead of cutting directly through obstacles.
                            for path_point in path_points[1:]:
                                path_point = np.asarray(path_point, np.float32)
                                if np.linalg.norm(
                                        path_point[[0, 2]] -
                                        current[[0, 2]]) > 0.05:
                                    online_endpoint_waypoint = path_point
                                    break
                        if online_endpoint_waypoint is None:
                            online_endpoint_waypoint = endpoint
                        online_endpoint_arrival = bool(
                            self.cluster_config.get(
                                "online_endpoint_geodesic_arrival", False) and
                            local_step > 0 and
                            online_endpoint_distance <= float(
                                self.cluster_config.get(
                                    "online_endpoint_geodesic_radius_m",
                                    0.35)) and
                            stop_fraction <= float(
                                self.cluster_config.get(
                                    "online_endpoint_stop_fraction", 0.50)))
                except Exception:
                    online_endpoint_distance = None
                    online_endpoint_arrival = False
            arrival_confirmed = bool(
                initial_near_field_arrival or motion_guard_satisfied and (
                    arrival_confirmed or terminal_cluster_consensus or
                    dense_stop_only_arrival or online_endpoint_arrival))
            endpoint_loss_guard_active = bool(
                self.cluster_config.get(
                    "endpoint_geodesic_loss_guard", False) and
                request.selected_point_navmesh_xyz is not None and
                online_endpoint_distance is not None and
                (not navigation_visible.any() or arrival_confirmed))
            endpoint_arrival_radius = float(self.cluster_config.get(
                "endpoint_geodesic_radius_m", 0.75))
            endpoint_loss_guard_blocked_arrival = bool(
                endpoint_loss_guard_active and
                online_endpoint_distance > endpoint_arrival_radius)
            if endpoint_loss_guard_blocked_arrival:
                # No image-space arrival proposal is physical arrival while
                # the robot is still outside the frozen selected endpoint's
                # radius. This includes dense stop-cluster loss while a few
                # navigation tracks remain and the zero-step near-field
                # shortcut; both produced false positives in real R2R nodes.
                arrival_confirmed = False
            record["endpoint_loss_guard_active"] = (
                endpoint_loss_guard_active)
            record["endpoint_loss_guard_blocked_arrival"] = (
                endpoint_loss_guard_blocked_arrival)
            record["online_endpoint_geodesic_distance_m"] = (
                online_endpoint_distance)
            # A simultaneous loss of all clusters is still an observation
            # event, not proof that the point was reached.  Bound every loss
            # based arrival (including terminal consensus) by the same
            # start-time travel estimate used by the near-loss fallback; this
            # prevents a tracker that has coasted past a doorway or side ray
            # from declaring success after a large overshoot.
            cluster_loss_max_fraction = self.cluster_config.get(
                "cluster_loss_maximum_travel_fraction")
            cluster_loss_travel_ok = bool(
                cluster_loss_max_fraction is None or
                travel_reference_m is None or
                cumulative_moved <= float(cluster_loss_max_fraction) *
                float(travel_reference_m))
            if not cluster_loss_travel_ok and not initial_near_field_arrival:
                arrival_confirmed = False
                terminal_cluster_consensus = False
                if (goal_visibility_low and not control_navigation_visible.any() and
                        record["end_reason"] is None):
                    # Fail closed as soon as the online target has vanished
                    # beyond the bounded start-time travel estimate.  Waiting
                    # for the global step budget would continue generating
                    # action history after the selected point was already
                    # missed, which is precisely the state we must not pass to
                    # the next semantic node.
                    record["end_reason"] = "cluster_loss_overshoot_guard"
                    record["terminal_cluster_consensus"] = False
                    record["terminal_motion_guard_satisfied"] = False
                    record["terminal_cumulative_moved_m"] = cumulative_moved
                    record["terminal_navigation_visible_fraction"] = (
                        navigation_fraction)
                    record["terminal_stop_visible_fraction"] = stop_fraction
                    break
            if arrival_confirmed:
                near_geodesic_loss = bool(
                    not initial_near_field_arrival and
                    self.cluster_config.get(
                        "near_geodesic_cluster_loss_arrival", False) and
                    geodesic_reference_m is not None and
                    geodesic_reference_m <= float(self.cluster_config.get(
                        "near_geodesic_threshold_m", 0.0)) and
                    terminal_cluster_consensus)
                record["end_reason"] = (
                    "initial_selected_point_in_near_field"
                    if initial_near_field_arrival else
                    "selected_point_online_geodesic_cluster_arrival"
                    if online_endpoint_arrival else STOP_ARRIVAL_REASON)
                if near_geodesic_loss:
                    record["end_reason"] = (
                        "near_geodesic_cluster_loss_arrival")
                record["arrival_signal"] = POINT_NAVIGATION_ARRIVED
                record["online_endpoint_geodesic_arrival"] = bool(
                    online_endpoint_arrival)
                record["online_endpoint_geodesic_distance_m"] = (
                    online_endpoint_distance)
                record["online_endpoint_stop_fraction"] = stop_fraction
                record["terminal_cluster_consensus"] = bool(
                    terminal_cluster_consensus)
                record["terminal_motion_guard_satisfied"] = bool(
                    motion_guard_satisfied)
                record["terminal_cumulative_moved_m"] = cumulative_moved
                record["terminal_navigation_visible_fraction"] = navigation_fraction
                record["terminal_stop_visible_fraction"] = stop_fraction
                terminal_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                for point, ok in zip(navigation_tracks, navigation_visible):
                    if ok:
                        cv2.circle(terminal_frame,
                                   tuple(np.round(point).astype(int)), 3,
                                   (0, 255, 255), -1)
                for point, ok in zip(goal_tracks, goal_visible):
                    if ok:
                        cv2.drawMarker(
                            terminal_frame,
                            tuple(np.round(point).astype(int)),
                            (255, 255, 0), cv2.MARKER_CROSS, 7, 1)
                for point, ok in zip(stop_tracks, stop_visible):
                    if ok:
                        cv2.drawMarker(
                            terminal_frame,
                            tuple(np.round(point).astype(int)),
                            (255, 0, 255), cv2.MARKER_DIAMOND, 8, 1)
                cv2.putText(
                    terminal_frame,
                    f"ARRIVED N {int(navigation_visible.sum())}/{navigation_count} "
                    f"G {int(goal_visible.sum())}/{goal_count} "
                    f"A {int(stop_visible.sum())}/{stop_count}", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2)
                self._append_video(
                    terminal_frame, request, request.position_history[-1], yaw,
                    "stop_cluster_arrived")
                break
            if not control_navigation_visible.any():
                record["end_reason"] = "navigation_cluster_lost"
                record["terminal_navigation_visible_fraction"] = 0.0
                record["terminal_stop_visible_fraction"] = stop_fraction
                break

            crop_goal_tracks = goal_tracks
            crop_goal_visible = goal_visible
            goal_cluster_source = "goal"
            if not goal_visible.any() and control_navigation_visible.any():
                crop_goal_tracks = control_navigation_tracks
                crop_goal_visible = control_navigation_visible
                goal_cluster_source = (
                    "navigation_confirmation" if
                    control_navigation_source == "navigation" else
                    control_navigation_source)
            goal, crop_geometry = crop_goal(
                rgb, crop_goal_tracks, crop_goal_visible,
                tuple(self.policy_config["image_size"]))
            if goal is None:
                record["end_reason"] = "no_crop"
                record["terminal_navigation_visible_fraction"] = navigation_fraction
                record["terminal_stop_visible_fraction"] = stop_fraction
                break
            goal_path = self._save_crop(
                goal,
                f"target_{request.target_index:02d}_step_{local_step:03d}.jpg")
            start = time.perf_counter()
            distance, trajectories = self.predict_fn(
                self.policy, self.policy_config, list(context), goal,
                self.policy_name, self.device, samples=4,
                seed=self.seed + global_step)
            inference_seconds = time.perf_counter() - start
            centroid = np.median(
                control_navigation_tracks[control_navigation_visible], axis=0)
            width = rgb.shape[1]
            pixel_angle = math.atan(
                (centroid[0] - width / 2) / (0.8 * width))
            navigation_angle = policy_heading(trajectories)
            desired_turn = (
                0.8 * pixel_angle +
                0.2 * np.clip(navigation_angle, -0.6, 0.6))
            endpoint_guidance = False
            endpoint_guidance_distance = None
            # Final-approach guard: once the agent is close to the selected
            # online navmesh projection, the learned policy can lose the
            # crop's point cluster and spin in place even though the target
            # is straight ahead.  Use the selected point's own geometry only
            # to keep the last short approach directed at that same target;
            # semantic view choice and the dense stop-cluster arrival rule are
            # unchanged.  No demonstration/reference path or VLM input is
            # involved.
            if (local_step > 0 and
                    request.selected_point_navmesh_xyz is not None):
                try:
                    endpoint = np.asarray(
                        request.selected_point_navmesh_xyz, np.float32)
                    current_position = np.asarray(
                        self.sim.get_agent(0).get_state().position,
                        np.float32)
                    endpoint_delta = endpoint - current_position
                    endpoint_guidance_distance = (
                        online_endpoint_distance
                        if online_endpoint_distance is not None else
                        float(np.linalg.norm(endpoint_delta[[0, 2]])))
                    final_approach_radius = float(
                        self.cluster_config.get(
                            "final_approach_guidance_radius_m", 0.80))
                    loss_guidance = bool(
                        self.cluster_config.get(
                            "endpoint_guidance_after_cluster_loss", False) and
                        not navigation_visible.any() and
                        online_endpoint_waypoint is not None)
                    if (np.isfinite(endpoint).all() and
                            (endpoint_guidance_distance <= final_approach_radius or
                             loss_guidance)):
                        guidance_delta = (
                            np.asarray(online_endpoint_waypoint, np.float32) -
                            current_position if loss_guidance else
                            endpoint_delta)
                        endpoint_yaw = math.atan2(
                            -float(guidance_delta[0]),
                            -float(guidance_delta[2]))
                        desired_turn = _wrap_angle(yaw - endpoint_yaw)
                        endpoint_guidance = True
                except Exception:
                    endpoint_guidance = False
            applied_turn = float(np.clip(
                desired_turn, -self.turn_limit, self.turn_limit))
            stall_probe = False
            stall_trigger = int(self.cluster_config.get(
                "stall_recovery_trigger_frames", 0))
            stall_turn_deg = float(self.cluster_config.get(
                "stall_recovery_turn_deg", 0.0))
            stall_max_probes = int(self.cluster_config.get(
                "stall_recovery_max_probes", 0))
            if (stall_trigger > 0 and stall_turn_deg > 0.0 and
                    blocked_forward_streak >= stall_trigger and
                    stall_probe_count < stall_max_probes):
                # Alternate a bounded 30-degree probe to expose a side
                # opening after a genuine blocked-forward streak.  The
                # controller then re-observes and resumes normal point
                # servoing; no target, depth, or reference path is changed.
                fixed_probe_sign = self.cluster_config.get(
                    "stall_recovery_probe_sign")
                if fixed_probe_sign in (-1, -1.0, 1, 1.0):
                    probe_sign = float(fixed_probe_sign)
                else:
                    probe_sign = -1.0 if stall_probe_count % 2 == 0 else 1.0
                applied_turn = math.radians(stall_turn_deg) * probe_sign
                desired_turn = applied_turn
                stall_probe = True
                stall_probe_count += 1
                blocked_forward_streak = 0
            yaw = _wrap_angle(yaw - applied_turn)
            old_position = np.asarray(
                self.sim.get_agent(0).get_state().position, np.float32)
            moved = 0.0
            if (abs(desired_turn) <= self.turn_limit or
                    (stall_probe and self.cluster_config.get(
                        "stall_recovery_forward", False))):
                forward = np.array(
                    [-math.sin(yaw), 0.0, -math.cos(yaw)], np.float32)
                proposed = old_position + self.forward_step * forward
                new_position = np.asarray(
                    self.sim.pathfinder.try_step(old_position, proposed),
                    np.float32)
                moved = float(np.linalg.norm(new_position - old_position))
            else:
                new_position = old_position
            _set_pose(self.sim, new_position, yaw)
            request.position_history.append(new_position.copy())
            next_rgb = _observe(self.sim)

            if moved > 0.01 and abs(applied_turn) < math.radians(3):
                action = "forward"
            elif moved > 0.01:
                action = ("turn_left_forward" if applied_turn < 0
                          else "turn_right_forward")
            elif abs(applied_turn) < math.radians(3):
                action = "blocked_forward"
            else:
                action = "turn_left" if applied_turn < 0 else "turn_right"
            if action == "blocked_forward":
                blocked_forward_streak += 1
            elif moved > 0.01:
                blocked_forward_streak = 0
                stall_probe_count = 0
            elif (self.cluster_config.get("persist_stall_recovery", False) and
                  (stall_probe or stall_probe_count > 0)):
                # Keep collision recovery armed across in-place turns until
                # a side probe produces actual motion.  This is local
                # collision feedback, not a semantic or reference-path cue.
                blocked_forward_streak = max(
                    1, int(self.cluster_config.get(
                        "stall_recovery_trigger_frames", 1)))
            elif not stall_probe:
                blocked_forward_streak = 0
            action_history.append({
                "step": local_step, "action": action,
                "turn_deg": math.degrees(-applied_turn), "moved_m": moved,
                # Persist the online robot trace on the graph edge.  Node
                # backtracking consumes these poses in reverse order instead
                # of assuming that the two stored nodes have line of sight.
                "position_xyz": new_position.tolist(), "yaw_rad": yaw,
                "stall_recovery_probe": stall_probe,
                "blocked_forward_streak": blocked_forward_streak,
                "stall_probe_count": stall_probe_count,
                "endpoint_guidance": endpoint_guidance,
                "endpoint_guidance_distance_m": endpoint_guidance_distance,
            })
            edge_frames.append(next_rgb.copy())
            edge_frame_metadata.append({
                "step": local_step, "phase": "edge_motion",
                "action": action, "moved_m": moved,
                "cumulative_moved_m": float(sum(
                    item["moved_m"] for item in action_history)),
                "position_xyz": new_position.tolist(), "yaw_rad": yaw,
            })

            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for point, ok in zip(navigation_tracks, navigation_visible):
                if ok:
                    cv2.circle(frame, tuple(np.round(point).astype(int)), 3,
                               (0, 255, 255), -1)
            for point, ok in zip(goal_tracks, goal_visible):
                if ok:
                    cv2.drawMarker(
                        frame, tuple(np.round(point).astype(int)),
                        (255, 255, 0), cv2.MARKER_CROSS, 7, 1)
            for point, ok in zip(stop_tracks, stop_visible):
                if ok:
                    cv2.drawMarker(
                        frame, tuple(np.round(point).astype(int)),
                        (255, 0, 255), cv2.MARKER_DIAMOND, 8, 1)
            crop_quad = np.round(crop_geometry["quad_xy"]).astype(np.int32)
            cv2.polylines(frame, [crop_quad], True, (0, 255, 0), 1)
            cv2.putText(
                frame,
                f"stage {request.target_index + 1}/{request.stage_count} "
                f"step {local_step}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(
                frame,
                f"N {int(navigation_visible.sum())}/{navigation_count} "
                f"G {int(goal_visible.sum())}/{goal_count} "
                f"A {int(stop_visible.sum())}/{stop_count} {action}",
                (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (0, 255, 255), 1)
            cv2.putText(
                frame,
                f"turn {math.degrees(-applied_turn):+.1f} deg "
                f"move {moved:.2f} m", (8, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
            cv2.putText(
                frame, request.instruction[:48],
                (8, rgb.shape[0] - 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (255, 255, 255), 1)
            cv2.putText(
                frame, f"target: {request.semantic_target[:43]}",
                (8, rgb.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.36, (0, 255, 255), 1)
            if local_step == 0:
                selected_xy = tuple(np.round(
                    np.asarray(request.selected_point_xy)).astype(int))
                cv2.drawMarker(frame, selected_xy, (0, 0, 255),
                               cv2.MARKER_CROSS, 22, 3)
                cv2.putText(frame, "VLM SELECTED POINT",
                            (8, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (0, 0, 255), 1)
            self._append_video(
                frame, request, new_position, yaw,
                "point_navigation_executor")

            record["steps"].append({
                "global_step": global_step,
                "navigation_visible_fraction": navigation_fraction,
                "stop_visible_fraction": stop_fraction,
                "goal_visible_fraction": goal_fraction,
                "navigation_centroid_xy": centroid.tolist(),
                "navigation_control_source": control_navigation_source,
                "navigation_loss_streak": navigation_loss_streak,
                "navigation_tracks_xy": navigation_tracks.tolist(),
                "navigation_visible": navigation_visible.tolist(),
                "stop_tracks_xy": stop_tracks.tolist(),
                "stop_visible": stop_visible.tolist(),
                "goal_tracks_xy": goal_tracks.tolist(),
                "goal_visible": goal_visible.tolist(),
                "crop_goal_tracks_xy": crop_goal_tracks.tolist(),
                "crop_goal_visible": crop_goal_visible.tolist(),
                "stop_visibility_low": stop_visibility_low,
                "dense_stop_visibility_low": dense_stop_visibility_low,
                "goal_visibility_low": goal_visibility_low,
                "goal_loss_streak": goal_loss_streak,
                "low_visibility_streak": low_visibility_streak,
                "dense_stop_low_streak": dense_stop_low_streak,
                "goal_cluster_source": goal_cluster_source,
                "crop_geometry": crop_geometry,
                "action": action,
                "goal_crop": goal_path,
                "policy_distance": distance,
                "policy_first_trajectory": np.asarray(
                    trajectories)[0].tolist(),
                "pixel_heading_rad": pixel_angle,
                "policy_heading_rad": navigation_angle,
                "applied_turn_rad": applied_turn,
                "moved_m": moved,
                "position_xyz": new_position.tolist(),
                "yaw_rad": yaw,
                "policy_inference_sec": inference_seconds,
                "stall_recovery_probe": stall_probe,
                "blocked_forward_streak": blocked_forward_streak,
                "stall_probe_count": stall_probe_count,
            })
            global_step += 1
            rgb = next_rgb
            initial_geodesic_cap = self.cluster_config.get(
                "max_travel_to_initial_geodesic_fraction")
            initial_geodesic = request.selected_point_initial_geodesic_m
            cumulative_after_action = float(sum(
                item["moved_m"] for item in action_history))
            if (initial_geodesic_cap is not None and
                    initial_geodesic is not None and
                    math.isfinite(float(initial_geodesic)) and
                    cumulative_after_action >= float(
                        initial_geodesic_cap) * float(initial_geodesic)):
                # The agent has exceeded a bounded shortest-path estimate
                # without satisfying the visual arrival gate.  Stop with an
                # explicit non-arrival signal rather than allowing the policy
                # to keep accumulating a misleading action history.
                record["end_reason"] = "initial_geodesic_travel_cap_exceeded"
                record["terminal_cumulative_moved_m"] = (
                    cumulative_after_action)
                record["terminal_navigation_visible_fraction"] = (
                    navigation_fraction)
                record["terminal_stop_visible_fraction"] = stop_fraction
                record["terminal_motion_guard_satisfied"] = False
                break
            travel_budget_reached = bool(
                request.max_travel_distance_m is not None and
                sum(item["moved_m"] for item in action_history) >=
                float(request.max_travel_distance_m))
            if travel_budget_reached:
                record["end_reason"] = "travel_budget_reached"
                record["terminal_cumulative_moved_m"] = float(sum(
                    item["moved_m"] for item in action_history))
                record["terminal_navigation_visible_fraction"] = (
                    navigation_fraction)
                record["terminal_stop_visible_fraction"] = stop_fraction
                break
            context.append(Image.fromarray(rgb))
            tracks, visible = self.tracker.step(rgb)
            if separate_arrival_tracker:
                stop_tracks, stop_visible = self.arrival_tracker.step(rgb)
        else:
            record["end_reason"] = "max_steps"
            record["terminal_navigation_visible_fraction"] = float(
                visible[:navigation_count].mean())
            if separate_arrival_tracker:
                record["terminal_stop_visible_fraction"] = float(
                    stop_visible.mean())
            else:
                stop_start = (navigation_count if shared_goal_and_arrival_cluster
                              else navigation_count + goal_count)
                record["terminal_stop_visible_fraction"] = float(
                    visible[stop_start:stop_start + stop_count].mean())

        if (record["arrival_signal"] is None and (
                self.tracking_cluster_profile ==
                "dense_stop_geodesic_guard_v9" or
                self.cluster_config.get(
                    "near_geodesic_cluster_loss_arrival", False))):
            cumulative_moved = float(sum(
                item["moved_m"] for item in action_history))
            depth = request.selected_point_depth_m
            initial_geodesic = request.selected_point_initial_geodesic_m
            reference_values = [
                float(value) for value in (depth, initial_geodesic)
                if value is not None and math.isfinite(float(value))]
            travel_reference_m = max(reference_values) if reference_values else None
            terminal_goal_lost = bool(not goal_visible.any())
            terminal_stop_fraction = float(stop_visible.mean())
            terminal_navigation_fraction = record.get(
                "terminal_navigation_visible_fraction")
            budget_arrival = bool(
                record["end_reason"] == "max_steps" and
                self.cluster_config.get("budget_terminal_goal_lost", False) and
                terminal_goal_lost and
                terminal_stop_fraction <= float(self.cluster_config[
                    "budget_terminal_stop_fraction"]) and
                (self.cluster_config.get(
                    "navigation_loss_arrival_stop_fraction") is None or
                 terminal_navigation_fraction is not None and
                 (float(terminal_navigation_fraction) > 0.0 or
                  terminal_stop_fraction <= float(self.cluster_config[
                      "navigation_loss_arrival_stop_fraction"]))) and
                travel_reference_m is not None and
                (self.cluster_config.get(
                    "cluster_loss_maximum_travel_fraction") is None or
                 cumulative_moved <= float(self.cluster_config[
                     "cluster_loss_maximum_travel_fraction"]) *
                 float(travel_reference_m)) and
                cumulative_moved >= float(self.cluster_config[
                    "budget_terminal_travel_fraction"]) * travel_reference_m)
            near_loss_arrival = bool(
                record["end_reason"] == "navigation_cluster_lost" and
                self.cluster_config.get(
                    "near_geodesic_cluster_loss_arrival", False) and
                initial_geodesic is not None and
                math.isfinite(float(initial_geodesic)) and
                (terminal_goal_lost or not self.cluster_config.get(
                    "near_loss_requires_goal_visibility", True)) and
                terminal_stop_fraction <= float(
                    self.cluster_config["arrival_visible_fraction"]) and
                (self.cluster_config.get(
                    "navigation_loss_arrival_stop_fraction") is None or
                 terminal_navigation_fraction is not None and
                  (float(terminal_navigation_fraction) > 0.0 or
                  terminal_stop_fraction <= float(self.cluster_config[
                      "navigation_loss_arrival_stop_fraction"]))) and
                (self.cluster_config.get(
                    "near_loss_maximum_travel_fraction") is None or
                 travel_reference_m is None or
                 cumulative_moved <= float(self.cluster_config[
                     "near_loss_maximum_travel_fraction"]) *
                 float(initial_geodesic)) and
                cumulative_moved >= float(self.cluster_config[
                    "near_geodesic_minimum_travel_fraction"] if
                    float(initial_geodesic) <= float(self.cluster_config[
                        "near_geodesic_threshold_m"]) else
                    self.cluster_config.get(
                        "near_loss_long_target_minimum_travel_fraction",
                        self.cluster_config[
                            "near_geodesic_minimum_travel_fraction"])) *
                float(initial_geodesic))
            if budget_arrival or near_loss_arrival:
                record["arrival_signal"] = POINT_NAVIGATION_ARRIVED
                record["end_reason"] = (
                    "budget_terminal_low_visibility_arrival"
                    if budget_arrival else
                    "near_geodesic_cluster_loss_arrival")
                record["terminal_cumulative_moved_m"] = cumulative_moved
                record["terminal_goal_visibility_low"] = terminal_goal_lost
                record["terminal_stop_visible_fraction"] = (
                    terminal_stop_fraction)
                record["terminal_travel_reference_m"] = travel_reference_m

        # A long selected ray can reach the RGB target while the tracker still
        # reports ``max_steps`` (the navigation cluster remains visible, but
        # the independent bottom/goal clusters have disappeared).  In that
        # case the selected point's own online navmesh distance is a direct
        # physical witness and is safer than relaxing the generic travel
        # fraction.  This is deliberately opt-in per profile and uses no
        # reference trajectory, semantic label, or hidden future state.
        if (record["arrival_signal"] is None and
                self.cluster_config.get(
                    "endpoint_geodesic_arrival", False) and
                request.selected_point_navmesh_xyz is not None and
                record.get("end_reason") in {
                    "max_steps", "navigation_cluster_lost",
                    "all_stop_cluster_points_disappeared"}):
            try:
                endpoint = np.asarray(
                    request.selected_point_navmesh_xyz, np.float32)
                current = np.asarray(
                    self.sim.get_agent(0).get_state().position, np.float32)
                shortest = __import__("habitat_sim").ShortestPath()
                shortest.requested_start = current
                shortest.requested_end = endpoint
                endpoint_reached = bool(
                    np.isfinite(endpoint).all() and
                    self.sim.pathfinder.find_path(shortest) and
                    float(shortest.geodesic_distance) <= float(
                        self.cluster_config.get(
                            "endpoint_geodesic_radius_m", 0.75)))
            except Exception:
                endpoint_reached = False
            stop_fraction = float(record.get(
                "terminal_stop_visible_fraction", 1.0))
            stop_threshold = float(self.cluster_config.get(
                "endpoint_geodesic_stop_fraction", 0.50))
            if endpoint_reached and stop_fraction <= stop_threshold:
                record["arrival_signal"] = POINT_NAVIGATION_ARRIVED
                record["end_reason"] = "selected_point_endpoint_geodesic_arrival"
                record["endpoint_geodesic_arrival"] = True
                record["endpoint_geodesic_distance_m"] = float(
                    shortest.geodesic_distance)
                record["endpoint_geodesic_radius_m"] = float(
                    self.cluster_config.get(
                        "endpoint_geodesic_radius_m", 0.75))
                record["endpoint_geodesic_stop_fraction"] = stop_fraction
            else:
                record["endpoint_geodesic_arrival"] = False

        edge_keyframes, keyframe_records = self._select_and_save_keyframes(
            edge_frames, edge_frame_metadata, request.target_index)
        record["action_history"] = action_history
        record["edge_keyframes"] = keyframe_records
        arrived = record["arrival_signal"] == POINT_NAVIGATION_ARRIVED
        return PointNavigationResult(
            arrived=arrived,
            signal=record["arrival_signal"],
            end_reason=record["end_reason"],
            final_rgb=rgb,
            final_yaw=yaw,
            next_global_step=global_step,
            action_history=action_history,
            record=record,
            edge_keyframes=edge_keyframes,
        )


def execute_point_navigation(
        executor: PointNavigationExecutor,
        request: PointNavigationRequest) -> PointNavigationResult:
    """Functional API for one externally selected point-navigation target."""
    return executor.execute(request)
