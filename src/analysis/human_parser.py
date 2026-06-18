"""
Human garment parsing (SCHP-equivalent).

Given a cropped person image, produces a per-pixel garment label map and the
dominant color of each garment region. This plays the same role as SCHP
(Self-Correcting Human Parsing) but uses a SegFormer model fine-tuned on the
ATR human-parsing dataset (``mattmdjaga/segformer_b2_clothes``), which runs out
of the box on CPU / Apple-Silicon MPS without the inplace_abn / CUDA build that
the original SCHP requires.

The model is downloaded automatically from the Hugging Face hub on first use
and cached under ~/.cache/huggingface.
"""

import os
import logging

# Force `transformers` to use the PyTorch backend only. By default it probes
# for TensorFlow/Flax at import time; on this machine the TensorFlow build is
# broken (native DLL load fails + a protobuf DType registration clash), which
# made the SegFormer clothing parser fail to load and fall back to heuristics.
# We use torch exclusively, so disable the TF/Flax probes. Must be set before
# transformers is imported (done lazily in _ensure_loaded below).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TORCH", "1")

import cv2
import numpy as np

MODEL_NAME = "mattmdjaga/segformer_b2_clothes"

# ATR / segformer_b2_clothes label set (id -> name)
ATR_LABELS = {
    0: "background",
    1: "hat",
    2: "hair",
    3: "sunglasses",
    4: "upper-clothes",
    5: "skirt",
    6: "pants",
    7: "dress",
    8: "belt",
    9: "left-shoe",
    10: "right-shoe",
    11: "face",
    12: "left-leg",
    13: "right-leg",
    14: "left-arm",
    15: "right-arm",
    16: "bag",
    17: "scarf",
}

# Labels we actually report as clothing/accessories (skip background, hair,
# face, bare limbs — those are not garments).
GARMENT_LABELS = {1, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17}

# Map ATR garment names to the search engine's item vocabulary so queries like
# "hat", "pants", "bag", "shoe" keep matching detected_items.
ITEM_NORMALIZE = {
    "upper-clothes": "shirt",
    "pants": "pants",
    "skirt": "skirt",
    "dress": "dress",
    "hat": "hat",
    "belt": "belt",
    "bag": "bag",
    "scarf": "scarf",
    "sunglasses": "sunglasses",
    "shoe": "shoe",
}


def _merge_shoes(garments):
    """Collapse left-shoe / right-shoe into a single 'shoe' entry."""
    shoes = [g for g in garments if g["type"] in ("left-shoe", "right-shoe")]
    if not shoes:
        return garments
    others = [g for g in garments if g["type"] not in ("left-shoe", "right-shoe")]
    best = max(shoes, key=lambda g: g["area_ratio"])
    merged = {
        "type": "shoe",
        "color": best["color"],
        "rgb": best["rgb"],
        "dom_bgr": best.get("dom_bgr"),
        "hsv": best.get("hsv"),
        "secondary_colors": best.get("secondary_colors", []),
        "area_ratio": round(sum(g["area_ratio"] for g in shoes), 4),
    }
    # Preserve the extcolors second opinion from the larger shoe, if present.
    if "color_extcolors" in best:
        merged["color_extcolors"] = best["color_extcolors"]
        merged["rgb_extcolors"] = best["rgb_extcolors"]
    others.append(merged)
    return others


def _build_metadata(garments):
    """
    Turn a flat list of per-garment {type,color,rgb,area_ratio} dicts into the
    stored metadata shape. `garments` is the single source of truth; `colors`
    and `detected_items` are flat aggregates kept for fast query matching.
    """
    colors = []
    detected_items = []
    for g in garments:
        if g["color"] and g["color"] not in colors:
            colors.append(g["color"])
        item = ITEM_NORMALIZE.get(g["type"])
        if item and item not in detected_items:
            detected_items.append(item)

    return {
        "garments": garments,
        "colors": colors,
        "detected_items": detected_items,
    }


class HumanParser:
    """Lazy-loaded SegFormer garment parser (SCHP-equivalent)."""

    def __init__(self):
        self._model = None
        self._processor = None
        self._torch = None
        self._device = None
        self._load_failed = False

    def _ensure_loaded(self):
        """Load model + processor on first use. Returns True if usable."""
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        try:
            import torch
            from transformers import (
                AutoModelForSemanticSegmentation,
                SegformerImageProcessor,
            )

            self._torch = torch
            self._processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
            self._model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_NAME)
            self._device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._model.to(self._device)
            self._model.eval()
            logging.info(f"Human parser (SegFormer clothes) loaded on {self._device}")
            return True
        except Exception as e:
            logging.error(f"Failed to load human parser: {e}")
            self._load_failed = True
            return False

    def preload(self):
        """Eagerly load the model (call from a background thread at startup)."""
        return self._ensure_loaded()

    def parse(self, crop_bgr):
        """
        Run garment parsing on a cropped person image (BGR ndarray).
        Returns an (H, W) int32 label map of ATR class ids, or None on failure.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        if not self._ensure_loaded():
            return None
        try:
            from PIL import Image

            torch = self._torch
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            inputs = self._processor(images=pil, return_tensors="pt").to(self._device)
            with torch.no_grad():
                logits = self._model(**inputs).logits  # (1, C, h, w)
            # Upsample class logits back to the crop's resolution.
            upsampled = torch.nn.functional.interpolate(
                logits,
                size=crop_bgr.shape[:2],  # (H, W)
                mode="bilinear",
                align_corners=False,
            )
            label_map = upsampled.argmax(dim=1)[0].to("cpu").numpy().astype(np.int32)
            return label_map
        except Exception as e:
            logging.error(f"Human parse error: {e}")
            return None

    def parse_garments(self, frame_bgr, bbox, analyzer):
        """
        Full garment pipeline for one detected human.

        Crops the person out of ``frame_bgr`` using ``bbox`` (the YOLO person
        bounding box), parses the crop into garment regions with SegFormer, and
        extracts the dominant color of each garment via
        ``analyzer.dominant_color_from_pixels``.

        SegFormer's per-pixel label map is the sole authority for which pixels
        belong to which garment — it already separates background, hair, face,
        and bare limbs from clothing, so no extra YOLO segmentation mask or skin
        removal is needed.

        Args:
            frame_bgr: Full frame (BGR)
            bbox: (x, y, w, h) of the person in full-frame coordinates
            analyzer: HumanImageAnalyzer (used for color classification)

        Returns: metadata dict (see _build_metadata) or None if parsing failed
            / no garments were found.
        """
        try:
            x, y, w, h = (int(v) for v in bbox)
            H, W = frame_bgr.shape[:2]
            x = max(0, x); y = max(0, y)
            w = max(1, min(w, W - x)); h = max(1, min(h, H - y))
            crop = frame_bgr[y:y + h, x:x + w]
            if crop.size == 0:
                return None

            label_map = self.parse(crop)
            if label_map is None:
                return None

            total_px = crop.shape[0] * crop.shape[1]
            min_px = max(40, int(total_px * 0.005))  # ignore tiny/noisy regions

            # White-balance gains from the WHOLE frame (shared by all garments).
            wb_gains = analyzer.estimate_wb_gains(frame_bgr)

            garments = []
            for lid in sorted(GARMENT_LABELS):
                region = label_map == lid
                cnt = int(region.sum())
                if cnt < min_px:
                    continue
                region_px = crop[region]
                # Primary method: dominant color + its 2 secondary colors.
                res = analyzer.dominant_color_from_pixels(
                    region_px, wb_gains, n_secondary=2
                )
                if res is None:
                    continue
                color_name, rgb, secondary = res
                # Dominant color in BGR + its HSV, for inspection/debugging the
                # naming decision (Hue 0-180, Sat/Val 0-255 on OpenCV's scale).
                dom_bgr = [int(rgb[2]), int(rgb[1]), int(rgb[0])]
                hsv = cv2.cvtColor(
                    np.uint8([[dom_bgr]]), cv2.COLOR_BGR2HSV
                )[0, 0]
                garment = {
                    "type": ATR_LABELS[lid],
                    "color": color_name,
                    "rgb": [int(c) for c in rgb],
                    "dom_bgr": dom_bgr,
                    "hsv": [int(hsv[0]), int(hsv[1]), int(hsv[2])],
                    "secondary_colors": secondary,
                    "area_ratio": round(cnt / total_px, 4),
                }
                # Second, independent estimate via the extcolors library
                # (most-frequent color group). Stored for cross-checking.
                res2 = analyzer.dominant_color_extcolors(region_px, wb_gains)
                if res2 is not None:
                    garment["color_extcolors"] = res2[0]
                    garment["rgb_extcolors"] = [int(c) for c in res2[1]]
                garments.append(garment)

            if not garments:
                return None

            garments = _merge_shoes(garments)
            return _build_metadata(garments)
        except Exception as e:
            logging.error(f"parse_garments error: {e}")
            return None
