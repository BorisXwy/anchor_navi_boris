#!/usr/bin/env python3
"""Replaceable RGB-only traversable-surface segmentation backends.

The navigation stack consumes a boolean mask in the native observation
resolution.  Backends in this module deliberately expose the same contract so
that a benchmark can compare promptable and dense semantic models without
changing point sampling or coordinate handling.
"""

from __future__ import annotations

import gc
import re
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image

from semantic_detector import (
    GROUND_QUERIES,
    DinoSamFloorSegmenter,
    GroundedSamDetector,
    SemanticDetection,
)


EXPANDED_GROUND_QUERIES = [
    "floor",
    "wooden floor",
    "tile floor",
    "carpet",
    "rug",
    "indoor ground",
    "walkable surface",
    "stairs",
    "staircase",
    "steps",
    "stair tread",
    "stair landing",
]
SAFE_STAIRS_GROUND_QUERIES = list(GROUND_QUERIES) + [
    "stairs", "staircase", "steps",
]


@dataclass
class GroundSegmentationResult:
    mask: np.ndarray
    class_map: np.ndarray | None
    labels: list[str]
    records: list[dict]


class GroundSegmentationBackend:
    name: str
    model_id: str

    def predict(self, rgb: np.ndarray) -> GroundSegmentationResult:
        raise NotImplementedError

    def predict_batch(self, rgbs: list[np.ndarray]) -> list[GroundSegmentationResult]:
        return [self.predict(rgb) for rgb in rgbs]

    def close(self) -> None:
        """Release accelerator memory before the next large model is loaded."""
        for value in list(vars(self).values()):
            if isinstance(value, torch.nn.Module):
                value.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class GroundedSamGroundBackend(GroundSegmentationBackend):
    """Original public Grounding-DINO Swin-T + SAM ViT-H pipeline."""

    def __init__(self, device="cuda:0", queries=None, detector=None,
                 name="grounded_sam_current"):
        if not str(device).startswith("cuda"):
            raise ValueError("segmentation benchmark model inference must use CUDA")
        self.name = name
        self.model_id = "IDEA-Research Grounding-DINO Swin-T + Meta SAM ViT-H"
        self.queries = list(queries or GROUND_QUERIES)
        self.detector = detector or GroundedSamDetector(device=device)
        self.segmenter = DinoSamFloorSegmenter(
            self.detector, queries=self.queries, close_kernel=9)

    def predict(self, rgb):
        mask, detections = self.segmenter(np.asarray(rgb, np.uint8), None)
        return GroundSegmentationResult(
            mask=np.asarray(mask, bool),
            class_map=None,
            labels=sorted({str(item.label) for item in detections}),
            records=[item.rgb_prompt_record() for item in detections],
        )

    def close(self):
        # The benchmark can share one Grounded-SAM detector between different
        # prompt vocabularies.  Its owner releases it after both passes.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _is_ground_label(label: str) -> bool:
    """ADE20K labels accepted as physical support surfaces.

    Word boundaries prevent accidental matches such as ``background``.  The
    list includes ordinary floor materials, vertical transition surfaces, and
    the occasional exterior support surface visible in Matterport panoramas.
    """
    words = set(re.findall(r"[a-z]+", str(label).lower()))
    return bool(words & {
        "floor", "rug", "carpet", "ground", "earth", "land", "soil",
        "stair", "stairs", "stairway", "staircase", "step", "steps",
        "landing", "sidewalk", "path", "road",
    })


class _TransformersDenseBackend(GroundSegmentationBackend):
    """Shared CUDA-only adapter for ADE20K dense semantic checkpoints."""

    architecture: str

    def __init__(self, model_id: str, name: str, architecture: str,
                 device="cuda:0"):
        if not str(device).startswith("cuda") or not torch.cuda.is_available():
            raise ValueError("segmentation benchmark model inference must use CUDA")
        from transformers import (
            AutoImageProcessor,
            Mask2FormerForUniversalSegmentation,
            OneFormerForUniversalSegmentation,
            OneFormerProcessor,
            SegformerForSemanticSegmentation,
        )

        self.name = name
        self.model_id = model_id
        self.architecture = architecture
        self.device = torch.device(device)
        if architecture == "oneformer":
            self.processor = OneFormerProcessor.from_pretrained(
                model_id, local_files_only=True)
            self.model = OneFormerForUniversalSegmentation.from_pretrained(
                model_id, local_files_only=True, use_safetensors=False)
        elif architecture == "mask2former":
            self.processor = AutoImageProcessor.from_pretrained(
                model_id, local_files_only=True)
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                model_id, local_files_only=True, use_safetensors=True)
        elif architecture == "segformer":
            self.processor = AutoImageProcessor.from_pretrained(
                model_id, local_files_only=True)
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_id, local_files_only=True, use_safetensors=False)
        else:
            raise ValueError(f"unsupported dense architecture: {architecture}")
        self.model.to(self.device).eval()
        self.id2label = {
            int(index): str(label)
            for index, label in self.model.config.id2label.items()
        }
        self.ground_ids = sorted(
            index for index, label in self.id2label.items()
            if _is_ground_label(label))
        if not self.ground_ids:
            raise RuntimeError(f"{model_id} exposes no recognized ground labels")

    @torch.inference_mode()
    def predict_batch(self, rgbs):
        rgbs = [np.asarray(rgb, np.uint8) for rgb in rgbs]
        if not rgbs:
            return []
        sizes = [rgb.shape[:2] for rgb in rgbs]
        images = [Image.fromarray(rgb, mode="RGB") for rgb in rgbs]
        if self.architecture == "oneformer":
            inputs = self.processor(
                images=images, task_inputs=["semantic"] * len(images),
                return_tensors="pt")
        else:
            inputs = self.processor(images=images, return_tensors="pt")
        inputs = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        outputs = self.model(**inputs)
        if self.architecture in {"oneformer", "mask2former"}:
            segmentations = self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=sizes)
        else:
            # SegFormer accepts mixed source sizes but emits one common logit
            # grid. Resize each batch element back to its own observation.
            segmentations = [torch.nn.functional.interpolate(
                outputs.logits[index:index + 1], size=size, mode="bilinear",
                align_corners=False).argmax(dim=1)[0]
                for index, size in enumerate(sizes)]
        results = []
        for segmentation, (height, width) in zip(segmentations, sizes):
            class_map = segmentation.detach().cpu().numpy().astype(np.int16)
            mask = np.isin(class_map, self.ground_ids)
            present_ids, counts = np.unique(class_map[mask], return_counts=True)
            records = [{
                "class_id": int(index),
                "label": self.id2label[int(index)],
                "pixel_count": int(count),
                "area_fraction": float(count / max(height * width, 1)),
            } for index, count in zip(present_ids, counts)]
            results.append(GroundSegmentationResult(
                mask=mask.astype(bool), class_map=class_map,
                labels=[item["label"] for item in records], records=records))
        return results

    def predict(self, rgb):
        return self.predict_batch([rgb])[0]

    def close(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


def build_dense_ground_backend(name: str, device="cuda:0"):
    specs = {
        "oneformer_ade20k": (
            "shi-labs/oneformer_ade20k_swin_tiny", "oneformer"),
        "mask2former_ade20k": (
            "facebook/mask2former-swin-small-ade-semantic", "mask2former"),
        "segformer_ade20k": (
            "nvidia/segformer-b5-finetuned-ade-640-640", "segformer"),
    }
    if name not in specs:
        raise ValueError(f"unknown dense ground backend: {name}")
    model_id, architecture = specs[name]
    return _TransformersDenseBackend(
        model_id=model_id, name=name, architecture=architecture,
        device=device)


def _mask_box(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def _is_stair_label(label):
    words = set(re.findall(r"[a-z]+", str(label).lower()))
    return bool(words & {
        "stair", "stairs", "stairway", "staircase", "step", "steps",
        "landing",
    })


class DenseMajorityGroundSegmenter:
    """Production RGB ground mask: >=2/3 dense ADE20K model agreement.

    The returned union is the immutable physical selection contract.  Callers
    may remove pixels for semantic relations, but must never add pixels that
    are absent from this mask.  ``strict_ground_mask`` lets legacy call sites
    disable their historical unsegmented lower-image rectangle fallback.
    """

    MODEL = (
        "Dense 2/3 majority (OneFormer Swin-T + Mask2Former Swin-S + "
        "SegFormer-B5, ADE20K)")
    name = "dense_majority"
    strict_ground_mask = True

    def __init__(self, device="cuda:0"):
        if not str(device).startswith("cuda") or not torch.cuda.is_available():
            raise ValueError("dense-majority ground segmentation requires CUDA")
        self.device = device
        self.backends = [build_dense_ground_backend(name, device=device) for name in (
            "oneformer_ade20k", "mask2former_ade20k", "segformer_ade20k")]

    def batch(self, rgbs, depths=None):
        del depths  # RGB-only by design; depth is never segmentation evidence.
        rgbs = [np.asarray(rgb, np.uint8) for rgb in rgbs]
        per_model = [backend.predict_batch(rgbs) for backend in self.backends]
        outputs = []
        for image_index, rgb in enumerate(rgbs):
            results = [model_results[image_index] for model_results in per_model]
            votes = np.stack([result.mask for result in results], axis=0).sum(axis=0)
            majority = votes >= 2
            stair_votes = np.zeros_like(votes, dtype=np.uint8)
            for backend, result in zip(self.backends, results):
                stair_ids = [class_id for class_id, label in backend.id2label.items()
                             if _is_stair_label(label)]
                stair_votes += np.isin(result.class_map, stair_ids)
            stair_mask = (stair_votes >= 2) & majority
            detections = []
            floor_mask = majority & ~stair_mask
            for label, mask in (("floor", floor_mask), ("stairs", stair_mask)):
                if mask.any():
                    detections.append(SemanticDetection(
                        label=label, score=float(votes[mask].mean() / 3.0),
                        box_xyxy=_mask_box(mask), mask=mask))
            outputs.append((majority.astype(bool), detections))
        return outputs

    def __call__(self, rgb, depth=None):
        return self.batch([rgb], None if depth is None else [depth])[0]

    def close(self):
        for backend in self.backends:
            backend.close()


def build_ground_segmenter(name, device="cuda:0", box_threshold=0.28,
                           text_threshold=0.22):
    """Build a navigation-facing ground segmenter by stable CLI name."""
    if name == "dense-majority":
        return DenseMajorityGroundSegmenter(device=device), None
    detector_class = (GroundedSamDetector if name == "grounded-sam"
                      else None)
    if name == "dino-sam":
        from semantic_detector import DinoSamDetector
        detector_class = DinoSamDetector
    if detector_class is None:
        raise ValueError(f"unknown ground segmenter: {name}")
    detector = detector_class(
        device, box_threshold=box_threshold, text_threshold=text_threshold)
    return DinoSamFloorSegmenter(detector), detector


def ground_segmenter_display_name(name):
    return {
        "dense-majority": DenseMajorityGroundSegmenter.MODEL,
        "grounded-sam": (
            "Grounded-SAM (Grounding-DINO Swin-T + SAM ViT-H)"),
        "dino-sam": "Grounding DINO Tiny + SAM ViT-B",
    }[name]
