"""Camera projection and path/point overlays for reproducible evaluations.

The overlays are diagnostic only: the reference path is never passed to a
model or candidate ranker.  They are rendered after selection so a reviewer
can distinguish the VLM-selected pixel from the hidden reference direction.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import cv2
import numpy as np


def project_world_point(point, camera_position, camera_yaw, width, height,
                        hfov_deg=90.0, near_clip=0.05):
    """Project an MP3D/Habitat world point into one RGB camera.

    Habitat's camera convention is used by :func:`pixel_to_world`: camera
    forward is negative local Z, local X is image-right, and local Y is
    image-up.  ``None`` means the point is behind the camera or too close to
    project stably.  The returned coordinates are not clipped, allowing the
    caller to retain off-image path evidence in its audit record.
    """
    delta = np.asarray(point, np.float32) - np.asarray(camera_position, np.float32)
    cosine, sine = math.cos(float(camera_yaw)), math.sin(float(camera_yaw))
    local_x = cosine * float(delta[0]) - sine * float(delta[2])
    local_z = sine * float(delta[0]) + cosine * float(delta[2])
    if not math.isfinite(local_z) or local_z >= -float(near_clip):
        return None
    focal = float(width) / (2.0 * math.tan(math.radians(float(hfov_deg)) / 2.0))
    x = (float(width) - 1.0) / 2.0 + local_x * focal / (-local_z)
    local_y = float(delta[1])
    y = (float(height) - 1.0) / 2.0 - local_y * focal / (-local_z)
    return [float(x), float(y)]


def project_reference_path(path: Sequence[Sequence[float]], path_indices: Iterable[int],
                           camera_position, camera_yaw, width, height,
                           hfov_deg=90.0):
    """Return a JSON-friendly projection audit for selected path vertices."""
    vertices = []
    for index in path_indices:
        if not 0 <= int(index) < len(path):
            continue
        pixel = project_world_point(
            path[int(index)], camera_position, camera_yaw, width, height, hfov_deg)
        vertices.append({
            "path_index": int(index),
            "pixel_xy": pixel,
            "visible_in_camera": bool(
                pixel is not None and 0 <= pixel[0] < width and 0 <= pixel[1] < height),
        })
    segments = []
    for first, second in zip(vertices[:-1], vertices[1:]):
        if first["pixel_xy"] is None or second["pixel_xy"] is None:
            continue
        segments.append({
            "from_path_index": first["path_index"],
            "to_path_index": second["path_index"],
            "from_pixel_xy": first["pixel_xy"],
            "to_pixel_xy": second["pixel_xy"],
        })
    return {"vertices": vertices, "segments": segments}


def draw_reference_path_overlay(image, projection, selected_point=None,
                                selected=False, next_path_index=None,
                                selected_label="VLM SELECTED",
                                legend_point_label="VLM POINT"):
    """Draw the hidden reference path (cyan) and selected VLM point (red)."""
    rendered = np.asarray(image).copy()
    # RGB colors: cyan path, yellow next waypoint, red selected point.
    for segment in projection.get("segments", []):
        p0 = tuple(int(round(v)) for v in segment["from_pixel_xy"])
        p1 = tuple(int(round(v)) for v in segment["to_pixel_xy"])
        cv2.line(rendered, p0, p1, (0, 230, 255), 3, cv2.LINE_AA)
    for vertex in projection.get("vertices", []):
        pixel = vertex.get("pixel_xy")
        if pixel is None:
            continue
        point = tuple(int(round(v)) for v in pixel)
        inside = vertex.get("visible_in_camera", False)
        color = (0, 255, 255) if vertex.get("path_index") == next_path_index \
            else (0, 180, 220)
        cv2.circle(rendered, point, 4 if inside else 3, color, -1)
    if selected and selected_point is not None:
        point = tuple(int(round(v)) for v in selected_point)
        cv2.drawMarker(rendered, point, (255, 30, 30), cv2.MARKER_CROSS, 22, 3)
        cv2.putText(rendered, selected_label, (max(2, point[0] - 45),
                    max(18, point[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (255, 30, 30), 1, cv2.LINE_AA)
    # The legend is deliberately present on every view in the contact sheet.
    cv2.putText(rendered, "GT PATH", (6, rendered.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 230, 255), 1, cv2.LINE_AA)
    cv2.putText(rendered, legend_point_label, (6, rendered.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 30, 30), 1, cv2.LINE_AA)
    return rendered
