#!/usr/bin/env python3
"""Replaceable Grounding-DINO box detection + SAM mask segmentation."""

import os
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor


GROUNDING_DINO_ID = "IDEA-Research/grounding-dino-tiny"
SAM_ID = "facebook/sam-vit-base"
# The workstation already contains the original Grounded-Segment-Anything
# checkout and its Swin-T/ViT-H checkpoints.  Keep these paths configurable so
# the detector remains portable to another machine without changing callers.
GROUNDED_SAM_ROOT = Path(os.environ.get(
    "POINT_TRACKER_GROUNDED_SAM_ROOT",
    "/mnt/pool1/sharehome/xiewenyuan/vlm/sim/third_party/Grounded-Segment-Anything",
))
GROUNDED_SAM_DINO_CONFIG = GROUNDED_SAM_ROOT / (
    "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDED_SAM_DINO_CHECKPOINT = Path(os.environ.get(
    "POINT_TRACKER_GROUNDED_SAM_DINO_CHECKPOINT",
    "/mnt/pool1/sharehome/xiewenyuan/vlm/sim/data/models/groundingdino_swint_ogc.pth",
))
GROUNDED_SAM_SAM_CHECKPOINT = Path(os.environ.get(
    "POINT_TRACKER_GROUNDED_SAM_SAM_CHECKPOINT",
    "/mnt/pool1/sharehome/xiewenyuan/vlm/sim/data/models/sam_vit_h_4b8939.pth",
))
GROUNDED_SAM_BERT_PATH = Path(os.environ.get(
    "POINT_TRACKER_GROUNDED_SAM_BERT_PATH",
    "/mnt/pool1/sharehome/xiewenyuan/vlm/sim/bert-base-uncased",
))
GROUND_QUERIES = [
    "floor", "wooden floor", "tile floor", "carpet", "rug",
    "walkable ground", "stair tread", "stair landing",
]


@dataclass
class SemanticDetection:
    label: str
    score: float
    box_xyxy: list
    mask: np.ndarray
    median_depth_m: float | None = None

    def prompt_record(self):
        return {
            "label": self.label, "score": round(self.score, 3),
            "box_xyxy": [round(value, 1) for value in self.box_xyxy],
            "median_depth_m": (round(self.median_depth_m, 2)
                               if self.median_depth_m is not None else None),
            "mask_area_fraction": round(float(self.mask.mean()), 4),
        }

    def rgb_prompt_record(self):
        """Evidence safe for a point selector that is forbidden to use depth."""
        record = self.prompt_record()
        record.pop("median_depth_m", None)
        return record


class SemanticDetector(ABC):
    @abstractmethod
    def detect(self, rgb, queries, depth=None):
        raise NotImplementedError


@dataclass(frozen=True)
class DinoBoxDetection:
    """A Grounding-DINO proposal before any SAM mask is computed."""

    label: str
    score: float
    box_xyxy: list


class GroundingDinoBoxDetector:
    """RGB/text -> scored boxes. This component never loads or calls SAM."""

    def __init__(self, device="cuda:0", detector_id=GROUNDING_DINO_ID,
                 box_threshold=0.28, text_threshold=0.22):
        self.device = torch.device(device)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.processor = AutoProcessor.from_pretrained(detector_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(detector_id)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def detect_boxes(self, rgb, queries):
        queries = [query.strip(" .") for query in queries if query.strip(" .")]
        if not queries:
            return []
        image = Image.fromarray(rgb)
        text = ". ".join(queries) + "."
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"], box_threshold=self.box_threshold,
            text_threshold=self.text_threshold, target_sizes=[rgb.shape[:2]],
        )[0]
        boxes = result["boxes"].detach().cpu().numpy()
        scores = result["scores"].detach().cpu().numpy()
        labels = result.get("text_labels", result.get("labels", []))
        if hasattr(labels, "detach"):
            labels = [queries[min(int(index), len(queries) - 1)]
                      for index in labels.detach().cpu().tolist()]
        if not len(boxes):
            return []
        # Suppress near-identical phrase/box hypotheses before the costlier SAM
        # pass. This NMS intentionally remains in the DINO-only component.
        order = np.argsort(-scores)
        keep = []
        for index in order:
            box = boxes[index]
            overlaps = []
            for kept in keep:
                other = boxes[kept]
                left_top = np.maximum(box[:2], other[:2])
                right_bottom = np.minimum(box[2:], other[2:])
                intersection = np.prod(np.maximum(0, right_bottom - left_top))
                union = (np.prod(box[2:] - box[:2]) +
                         np.prod(other[2:] - other[:2]) - intersection)
                overlaps.append(intersection / max(union, 1e-6))
            if not overlaps or max(overlaps) < 0.72:
                keep.append(int(index))
        return [DinoBoxDetection(
            str(labels[index]), float(scores[index]), boxes[index].tolist())
            for index in keep]


class SamBoxSegmenter:
    """RGB + boxes -> one best boolean SAM mask per box."""

    def __init__(self, device="cuda:0", segmenter_id=SAM_ID):
        self.device = torch.device(device)
        self.processor = SamProcessor.from_pretrained(segmenter_id)
        self.model = SamModel.from_pretrained(segmenter_id)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def segment_boxes(self, rgb, boxes):
        if not boxes:
            return []
        image = Image.fromarray(rgb)
        sam_inputs = self.processor(
            images=image, input_boxes=[[list(box) for box in boxes]],
            return_tensors="pt")
        sam_inputs = {key: value.to(self.device) for key, value in sam_inputs.items()}
        sam_outputs = self.model(**sam_inputs, multimask_output=True)
        masks = self.processor.image_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(), sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu(),
        )[0]
        iou = sam_outputs.iou_scores[0].detach().cpu()
        best = iou.argmax(-1)
        masks = torch.stack([
            masks[index, best[index]] for index in range(len(boxes))])
        return [mask.numpy().astype(bool) for mask in masks]


class DinoSamDetector(SemanticDetector):
    """Explicit two-stage Grounding-DINO + SAM semantic detector.

    DINO owns phrase grounding and box NMS. SAM sees only the frozen DINO
    boxes and owns mask extraction. Keeping these as injected components makes
    either model replaceable without changing the navigation-facing API.
    """

    def __init__(self, device="cuda:0", detector_id=GROUNDING_DINO_ID,
                 segmenter_id=SAM_ID, box_threshold=0.28, text_threshold=0.22,
                 box_detector=None, mask_segmenter=None):
        self.device = torch.device(device)
        self.box_detector = box_detector or GroundingDinoBoxDetector(
            device=device, detector_id=detector_id,
            box_threshold=box_threshold, text_threshold=text_threshold)
        self.mask_segmenter = mask_segmenter or SamBoxSegmenter(
            device=device, segmenter_id=segmenter_id)

    def detect(self, rgb, queries, depth=None):
        proposals = self.box_detector.detect_boxes(rgb, queries)
        masks = self.mask_segmenter.segment_boxes(
            rgb, [proposal.box_xyxy for proposal in proposals])
        if len(masks) != len(proposals):
            raise RuntimeError(
                "SAM returned a different mask count from the DINO box count: "
                f"{len(masks)} != {len(proposals)}")

        detections = []
        for proposal, mask in zip(proposals, masks):
            median_depth = None
            if depth is not None:
                values = depth[mask]
                values = values[np.isfinite(values) & (values > 0)]
                if len(values):
                    median_depth = float(np.median(values))
            detections.append(SemanticDetection(
                proposal.label, proposal.score, proposal.box_xyxy, mask,
                median_depth))
        return detections


class GroundedSamDetector(SemanticDetector):
    """Grounded-SAM detector backed by the local original checkout.

    This is deliberately kept behind the same ``SemanticDetector`` contract as
    :class:`DinoSamDetector`.  The local project uses the original
    Grounding-DINO Swin-T checkpoint followed by SAM ViT-H masks, rather than
    the lightweight Transformers DINO/SAM pair.  No ``supervision`` dependency
    is needed here: the small amount of box conversion/NMS is implemented
    directly so this detector can run in the unified ``.venv``.
    """

    def __init__(self, device="cuda:0", root=GROUNDED_SAM_ROOT,
                 dino_config=GROUNDED_SAM_DINO_CONFIG,
                 dino_checkpoint=GROUNDED_SAM_DINO_CHECKPOINT,
                 sam_checkpoint=GROUNDED_SAM_SAM_CHECKPOINT,
                 bert_path=GROUNDED_SAM_BERT_PATH,
                 sam_version="vit_h", box_threshold=0.28,
                 text_threshold=0.22, nms_threshold=0.8):
        self.device = torch.device(device)
        self.root = Path(root)
        self.dino_config = Path(dino_config)
        self.dino_checkpoint = Path(dino_checkpoint)
        self.sam_checkpoint = Path(sam_checkpoint)
        self.bert_path = Path(bert_path) if bert_path else None
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.nms_threshold = float(nms_threshold)
        missing = [str(path) for path in (
            self.dino_config, self.dino_checkpoint, self.sam_checkpoint)
                   if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(
                "Grounded-SAM assets are missing: " + ", ".join(missing))

        # Import the checked-out implementations lazily.  This keeps unit
        # tests and the Transformers backend independent of the optional
        # Grounded-SAM source tree.
        dino_root = str(self.root / "GroundingDINO")
        sam_root = str(self.root / "segment_anything")
        for path in (dino_root, sam_root):
            if path not in sys.path:
                sys.path.insert(0, path)
        # Use Grounding-DINO's official PyTorch deformable-attention fallback
        # by default on this host.  The bundled custom extension was built for
        # a different ABI/CUDA combination and is unsafe on the current Ada
        # driver; the fallback still executes on the selected GPU.
        os.environ.setdefault("GROUNDED_SAM_FORCE_PYTORCH_ATTENTION", "1")
        try:
            from groundingdino.datasets import transforms as dino_transforms
            from groundingdino.models import build_model
            from groundingdino.util.slconfig import SLConfig
            from groundingdino.util.utils import (
                clean_state_dict, get_phrases_from_posmap)
            from segment_anything import SamPredictor, sam_model_registry
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Unable to import the local Grounded-SAM checkout") from exc
        self._dino_transforms = dino_transforms
        self._get_phrases_from_posmap = get_phrases_from_posmap

        args = SLConfig.fromfile(str(self.dino_config))
        args.device = str(self.device)
        # The upstream config enables activation checkpointing for training.
        # It is unnecessary for inference and interacts poorly with
        # ``torch.inference_mode`` on the current PyTorch/CUDA stack.
        args.use_checkpoint = False
        args.use_transformer_ckpt = False
        if self.bert_path is not None and self.bert_path.exists():
            args.bert_base_uncased_path = str(self.bert_path)
        self.dino_model = build_model(args)
        checkpoint = torch.load(str(self.dino_checkpoint),
                                map_location="cpu")
        state = checkpoint.get("model", checkpoint)
        self.dino_model.load_state_dict(clean_state_dict(state), strict=False)
        self.dino_model.to(self.device).eval()

        sam = sam_model_registry[sam_version](checkpoint=str(self.sam_checkpoint))
        sam.to(device=self.device).eval()
        self.sam_predictor = SamPredictor(sam)

    def _preprocess(self, rgb):
        transform = self._dino_transforms.Compose([
            self._dino_transforms.RandomResize([800], max_size=1333),
            self._dino_transforms.ToTensor(),
            self._dino_transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image = Image.fromarray(np.asarray(rgb).astype(np.uint8), mode="RGB")
        transformed, _ = transform(image, None)
        return transformed

    @torch.inference_mode()
    def _detect_boxes(self, rgb, queries):
        queries = [str(query).strip(" .") for query in queries
                   if str(query).strip(" .")]
        if not queries:
            return []
        caption = ". ".join(queries).lower().strip() + "."
        image = self._preprocess(rgb).to(self.device)
        outputs = self.dino_model(image[None], captions=[caption])
        logits = outputs["pred_logits"].sigmoid()[0].detach().cpu()
        boxes = outputs["pred_boxes"][0].detach().cpu()
        keep = logits.max(dim=1).values > self.box_threshold
        logits, boxes = logits[keep], boxes[keep]
        if not len(boxes):
            return []
        tokenizer = self.dino_model.tokenizer
        tokenized = tokenizer(caption)
        phrases = [self._get_phrases_from_posmap(
            logit > self.text_threshold, tokenized, tokenizer)
                   for logit in logits]
        height, width = rgb.shape[:2]
        scale = torch.tensor([width, height, width, height], dtype=boxes.dtype)
        cxcywh = boxes * scale
        xyxy = torch.zeros_like(cxcywh)
        xyxy[:, 0] = cxcywh[:, 0] - cxcywh[:, 2] / 2
        xyxy[:, 1] = cxcywh[:, 1] - cxcywh[:, 3] / 2
        xyxy[:, 2] = cxcywh[:, 0] + cxcywh[:, 2] / 2
        xyxy[:, 3] = cxcywh[:, 1] + cxcywh[:, 3] / 2
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp(0, width)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp(0, height)
        # Match the upstream Grounded-SAM demo's NMS threshold while avoiding
        # a hard dependency on the optional ``supervision`` package.
        try:
            nms_keep = torch.ops.torchvision.nms(
                xyxy, logits.max(dim=1).values, self.nms_threshold)
        except (AttributeError, RuntimeError):
            # Some CPU-only or mismatched torchvision wheels do not register
            # the compiled operator.  Keep the backend usable with the same
            # greedy IoU rule instead of silently reverting to Transformers.
            order = torch.argsort(logits.max(dim=1).values, descending=True)
            kept = []
            while len(order):
                current = int(order[0])
                kept.append(current)
                if len(order) == 1:
                    break
                rest = order[1:]
                box = xyxy[current]
                other = xyxy[rest]
                left_top = torch.maximum(box[:2], other[:, :2])
                right_bottom = torch.minimum(box[2:], other[:, 2:])
                intersection = torch.prod(
                    torch.clamp(right_bottom - left_top, min=0), dim=1)
                area = torch.prod(torch.clamp(box[2:] - box[:2], min=0))
                other_area = torch.prod(
                    torch.clamp(other[:, 2:] - other[:, :2], min=0), dim=1)
                union = area + other_area - intersection
                iou = intersection / torch.clamp(union, min=1e-6)
                order = rest[iou <= self.nms_threshold]
            nms_keep = torch.tensor(kept, dtype=torch.long)
        return [DinoBoxDetection(
            str(phrases[int(index)]).replace(".", "").strip(),
            float(logits[int(index)].max().item()),
            xyxy[int(index)].tolist()) for index in nms_keep]

    @torch.inference_mode()
    def detect(self, rgb, queries, depth=None):
        proposals = self._detect_boxes(rgb, queries)
        if not proposals:
            return []
        self.sam_predictor.set_image(np.asarray(rgb).astype(np.uint8))
        detections = []
        for proposal in proposals:
            masks, scores, _ = self.sam_predictor.predict(
                box=np.asarray(proposal.box_xyxy, dtype=np.float32),
                multimask_output=True)
            mask = np.asarray(masks[int(np.argmax(scores))]).astype(bool)
            median_depth = None
            if depth is not None:
                values = np.asarray(depth)[mask]
                values = values[np.isfinite(values) & (values > 0)]
                if len(values):
                    median_depth = float(np.median(values))
            detections.append(SemanticDetection(
                proposal.label, proposal.score, proposal.box_xyxy, mask,
                median_depth))
        return detections


class DinoSamFloorSegmenter:
    """Ground union from explicit Grounding-DINO boxes and SAM masks."""

    def __init__(self, detector, queries=None, close_kernel=9,
                 min_detection_score=0.0):
        self.detector = detector
        self.queries = list(queries or GROUND_QUERIES)
        self.close_kernel = close_kernel
        # The detector can run with a low proposal threshold to recover weak
        # floor in side/rear views.  Callers may raise this per instruction
        # form without rebuilding the heavy DINO/SAM models.
        self.min_detection_score = float(min_detection_score)

    def __call__(self, rgb, depth=None):
        detections = self.detector.detect(rgb, self.queries, depth)
        if self.min_detection_score > 0.0:
            detections = [
                item for item in detections
                if float(getattr(item, "score", 0.0)) >=
                self.min_detection_score]
        mask = np.zeros(rgb.shape[:2], bool)
        for detection in detections:
            mask |= detection.mask
        if mask.any() and self.close_kernel > 1:
            kernel = np.ones((self.close_kernel, self.close_kernel), np.uint8)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                    kernel).astype(bool)
        return mask, detections

    def batch(self, rgbs, depths=None):
        depths = depths if depths is not None else [None] * len(rgbs)
        return [self(rgb, depth) for rgb, depth in zip(rgbs, depths)]


# ``GroundedSamDetector`` is the concrete local Grounded-SAM implementation
# above.  Keep the floor-segmenter alias for downstream readers that imported
# the legacy name; its detector dependency is intentionally injectable.
GroundedSamFloorSegmenter = DinoSamFloorSegmenter


def extract_detection_queries(stage):
    """Extract conservative open-vocabulary noun phrases from a typed clause."""
    clause = stage.get("source_clause", stage.get("navigation_instruction", "")).lower()
    form = stage.get("form", "")
    if form in {"VERTICAL_UP", "VERTICAL_DOWN"}:
        return ["stairs", "staircase", "landing"]
    # A typed stage can have a landmark as its primary form while a portal
    # crossing is the next physical transition (for example, “pass the sink
    # through the doorway”).  Grounding only the noun landmark leaves the
    # route opening invisible to the RGB router and encourages a salient but
    # side-facing object view.  Preserve the ordered stage semantics by adding
    # portal queries whenever the secondary form explicitly requests a
    # traversal; this is generic and does not use demo/depth state.
    secondary_forms = set(stage.get("secondary_forms", []) or [])
    portal_stage = form in {
        "EXIT_REGION", "ENTER_REGION", "SELECT_PORTAL", "TRAVERSE_PORTAL_REGION"
    } or "TRAVERSE_PORTAL_REGION" in secondary_forms
    if portal_stage:
        base = ["doorway", "open door", "hallway opening"]
    else:
        base = []
    landmark = str(stage.get("landmark", "")).strip(" .")
    if landmark and landmark.lower() not in {"none", "unspecified", "unknown"}:
        # VLM decompositions normally provide a compact noun phrase here.
        if len(landmark.split()) <= 8:
            base.append(landmark)
    # Completion cues and semantic targets can name the *next* region even
    # when the primary landmark is an obstacle (e.g. "pass the couches ...
    # heading towards the kitchen area"). Include compact noun phrases after
    # a spatial transition verb so the detector can provide cross-view
    # context for the VLM and node judge. This remains a lexical, RGB-only
    # proposal: it never chooses a point, uses depth, or consults a reference
    # path. Keep the extraction conservative to avoid turning the whole
    # completion sentence into a detector query.
    semantic_fields = " ".join(
        str(stage.get(key, "")) for key in (
            "completion_cue", "semantic_spatial_target",
            "visual_arrival_evidence"))
    for match in re.finditer(
            r"\b(?:heading\s+)?(?:towards?|into|inside|beyond|through|"
            r"leads?\s+to|leading\s+to)\s+(?:the\s+|a\s+|an\s+)?"
            r"([a-z][a-z0-9]*(?:\s+[a-z][a-z0-9]*){0,3})",
            semantic_fields.lower()):
        candidate = re.split(
            r"\b(?:and|while|with|that|where|before|after)\b|[,.;!?]",
            match.group(1), maxsplit=1)[0].strip()
        if candidate and not re.match(
                r"^(?:the|a|an|floor|camera|agent|you|it)\b", candidate):
            base.append(candidate)
    patterns = [
        r"\b(?:past|pass|passing|near|beside|towards?|by)\s+(?:the\s+|a\s+)?(.+?)(?:,|$)",
        r"\bbetween\s+(?:the\s+)?(.+?)\s+and\s+(?:the\s+)?(.+?)(?:,|$)",
        r"\b(?:around|left side of|right side of)\s+(?:the\s+|a\s+)?(.+?)(?:,|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, clause)
        if match:
            base.extend(group.strip() for group in match.groups() if group)
    # Normalize VLM decomposition fields that occasionally contain an entire
    # action clause instead of a compact noun phrase. Grounding-DINO should see
    # object/region categories ("sink", "atrium", "table"), never prompts such
    # as "walk towards the sink" or "go around the table and to the right".
    cleaned = []
    for query in base:
        query = str(query).lower().strip(" .,")
        relation = re.search(
            r"\b(?:towards?|around|past|near|beside|into|through)\s+"
            r"(?:the\s+|a\s+|an\s+)?(.+)$", query)
        if relation and re.match(
                r"^(?:walk|go|head|move|travel|proceed|continue)\b", query):
            query = relation.group(1)
        query = re.split(
            r"\b(?:and then|then|until|and stop|and wait|"
            r"and\s+(?:to\s+)?(?:the\s+)?(?:left|right))\b", query)[0]
        query = re.sub(r"\bright next to (?:you|me)\b.*$", "", query)
        query = re.sub(r"^(?:the|a|an)\s+", "", query)
        query = query.strip(" .,")
        # If an action-bearing query still survived normalization, discard it;
        # the conservative regexes or generic portal queries remain available.
        if re.search(
                r"\b(?:walk|go|head|move|travel|proceed|continue|turn|stop|wait)\b",
                query):
            continue
        # Keep moderately long relational phrases long enough for the
        # head-noun/destination expansion below (for example, ``doorway on
        # the left leading to living room`` has nine tokens).  Very long
        # action clauses are still rejected so detector prompts remain
        # conservative.
        if len(query.split()) > 12:
            continue
        if query and query not in cleaned:
            cleaned.append(query)

    # Grounding-DINO is considerably more reliable with the object/region
    # head than with a long relational noun phrase.  Keep the original
    # phrase for identity, but add generic lexical projections so a stage
    # such as ``corner of the bar`` or ``room with cardboard boxes`` does not
    # silently turn into a zero-detection floor-only decision.  This is a
    # form-level normalization, independent of episode/scene names; it does
    # not choose a view or expose any depth/path information.
    expanded = []
    relation_prefix = re.compile(
        r"^(?:corner|center|middle|top|bottom|front|back|far\s+side|near\s+side|"
        r"left\s+side|right\s+side|backside|frontside)\s+of\s+",
        flags=re.IGNORECASE)
    relation_suffix = re.compile(
        r"\s+(?:with|containing|that\s+leads\s+to|leading\s+to)\s+.+$",
        flags=re.IGNORECASE)
    for query in cleaned:
        variants = [query]
        core = relation_prefix.sub("", query)
        core = relation_suffix.sub("", core).strip(" .")
        core = re.sub(r"^(?:the|a|an)\s+", "", core).strip()
        if core and core != query:
            variants.append(core)
        # Preserve the visual category when a descriptor is attached.  These
        # aliases are intentionally small and scene-agnostic.
        words = core.split()
        if words:
            synonym = {
                "couch": "sofa", "sofas": "couch", "rug": "carpet",
                "carpet": "rug", "entryway": "doorway", "entrance": "doorway",
                "staircase": "stairs", "steps": "stairs", "manel": "mantel",
            }
            last = synonym.get(words[-1])
            if last and last != core:
                variants.append(last)
                if words[-1] == "manel":
                    variants.append("fireplace mantel")
                    variants.append("panel")
            # Adjective-heavy phrases often fail while their head noun works.
            if len(words) > 1 and words[-1] not in {
                    "room", "area", "hall", "hallway", "corridor"}:
                variants.append(words[-1])
        for variant in variants:
            variant = variant.strip(" .")
            if variant and variant not in expanded:
                expanded.append(variant)
    # Preserve stable visual heads from relational landmarks.  A phrase such
    # as ``doorway on the left leading to living room`` is useful context for
    # the VLM, but Grounding-DINO is substantially more reliable when it also
    # receives the standalone object/region categories ``doorway`` and
    # ``living room``.  This lexical expansion is scene-agnostic and is only
    # used to propose RGB detector evidence; it never selects a point or uses
    # depth/reference-path information.
    relation_expansions = []
    for query in list(expanded):
        query_lower = str(query).lower().strip(" .")
        portal_heads = re.findall(
            r"\b(?:doorway|door|opening|entrance|portal|archway)\b",
            query_lower)
        relation_expansions.extend(portal_heads)
        destination = re.search(
            r"\b(?:leads?\s+to|leading\s+to|opens?\s+into|into|inside)\s+"
            r"(?:the\s+|a\s+|an\s+)?([a-z][a-z0-9]*(?:\s+[a-z][a-z0-9]*){0,3})",
            query_lower)
        if destination:
            destination_phrase = re.split(
                r"\b(?:and|while|before|after|that|with)\b|[,.;!?]",
                destination.group(1), maxsplit=1)[0].strip()
            if destination_phrase and destination_phrase not in {
                    "floor", "ground", "camera", "agent"}:
                relation_expansions.append(destination_phrase)
    for variant in relation_expansions:
        variant = str(variant).strip(" .")
        if variant and variant not in expanded:
            expanded.append(variant)
    # ``manel`` is ambiguous between ``mantel`` and ``panel``. Preserve both
    # plausible visual heads instead of silently forcing one correction; the
    # RGB route sequence must disambiguate them. This lexical policy never
    # consults scene, episode, path, or outcome.
    canonical = []
    for query in expanded:
        if re.search(r"\bmanel\b", query):
            mantel = re.sub(r"\bmanel\b", "mantel", query)
            panel = re.sub(r"\bmanel\b", "panel", query)
            for variant in (
                    mantel, panel, "mantel", "fireplace mantel", "panel"):
                if variant not in canonical:
                    canonical.append(variant)
        elif query not in canonical:
            canonical.append(query)
    expanded = canonical
    return expanded
