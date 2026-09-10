#!/usr/bin/env python3
"""Create a traversable-ground mask with a replaceable production backend."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ground_segmentation_backends import (
    build_ground_segmenter, ground_segmenter_display_name,
)


class GroundSegmenter:
    """Small CLI adapter around the navigation ground-mask factory."""

    def __init__(self, device="cuda:0", backend="dense-majority"):
        self.backend = backend
        self.MODEL = ground_segmenter_display_name(backend)
        self.segmenter, self.detector = build_ground_segmenter(
            backend, device)
        self.queries = getattr(self.segmenter, "queries", None)

    def __call__(self, rgb: np.ndarray):
        return self.segmenter(rgb)

    def batch(self, rgbs):
        return self.segmenter.batch(rgbs)


def save_result(rgb, mask, detections, output: Path, segmenter: GroundSegmenter):
    overlay = rgb.copy()
    green = np.zeros_like(rgb); green[..., 1] = 255
    overlay[mask] = (0.45 * overlay[mask] + 0.55 * green[mask]).astype(np.uint8)
    for detection in detections:
        x0, y0, x1, y1 = np.round(detection.box_xyxy).astype(int)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 220, 255), 1)
        cv2.putText(overlay, f"{detection.label} {detection.score:.2f}",
                    (max(2, x0), max(14, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (0, 220, 255), 1)
    mask_rgb = np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, 2)
    panel = np.concatenate([rgb, overlay, mask_rgb], 1).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(panel).save(output)
    Image.fromarray((mask * 255).astype(np.uint8)).save(
        output.with_name(output.stem + "_mask.png"))
    stats = {
        "model": segmenter.MODEL,
        "input_size": [rgb.shape[1], rgb.shape[0]],
        "ground_fraction": float(mask.mean()),
        "ground_queries": segmenter.queries,
        "ground_detections": [item.prompt_record() for item in detections],
        "mask_policy": (
            "pixelwise >=2/3 dense ADE20K ground agreement"
            if segmenter.backend == "dense-majority" else
            "union of text-grounded boxes segmented by SAM"),
        "strict_ground_mask": bool(
            getattr(segmenter.segmenter, "strict_ground_mask", False)),
    }
    output.with_suffix(".json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/ground_segmentation.png"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--backend", choices=["dense-majority", "grounded-sam", "dino-sam"],
        default="dense-majority")
    args = parser.parse_args()
    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise FileNotFoundError(args.image)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    segmenter = GroundSegmenter(args.device, args.backend)
    mask, detections = segmenter(rgb)
    print(json.dumps(save_result(rgb, mask, detections, args.output, segmenter),
                     indent=2))


if __name__ == "__main__":
    main()
