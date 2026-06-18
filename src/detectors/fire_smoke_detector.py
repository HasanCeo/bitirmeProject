"""
YOLO-based fire & smoke detection.

Replaces the color/brightness heuristic with a trained YOLOv8 model that learns
flame/smoke *texture and shape* (so it ignores warm clothing, skin and lamps,
and can also detect smoke). A K-of-N temporal-confirmation stage suppresses
single-frame false positives: a region must be detected in at least K of the
last N processed frames, in roughly the same place, before it is reported.

The model weights are a normal Ultralytics ``.pt`` file loaded from a configured
path (see settings.FIRE_MODEL_PATH). They are NOT downloaded automatically — a
``.pt`` is a pickle and loading an untrusted one runs arbitrary code, so the
user supplies a trusted weights file. When no weights are present, available()
returns False and the app falls back to the basic detector.
"""

import logging
from collections import deque
from pathlib import Path

import numpy as np

from .base import BaseDetector


def _iou(a, b):
    """IoU of two (x, y, w, h) boxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class FireSmokeDetector(BaseDetector):
    """Trained YOLO fire/smoke detector with temporal confirmation."""

    def __init__(self, model_path, conf=0.4, persist_k=3, persist_n=5,
                 iou_link=0.2):
        """
        Args:
            model_path: path to a YOLO fire/smoke .pt weights file
            conf: minimum detection confidence
            persist_k: a region must appear in >= K of the last N frames
            persist_n: temporal window length
            iou_link: IoU above which boxes in different frames are "the same"
        """
        self.model_path = str(model_path)
        self.conf = conf
        self.k = persist_k
        self.n = persist_n
        self.iou_link = iou_link

        self._model = None
        self._load_failed = False
        self._history = deque(maxlen=persist_n)  # past frames' raw boxes
        self.class_names = {}

    def _ensure_loaded(self):
        """Load the model on first use. Returns True if usable."""
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        if not Path(self.model_path).exists():
            self._load_failed = True
            logging.warning(
                f"Fire model not found at {self.model_path}; "
                "YOLO fire/smoke detection disabled (add the weights file to enable)."
            )
            return False
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            self.class_names = dict(getattr(self._model, "names", {}) or {})
            logging.info(
                f"Fire/smoke YOLO model loaded from {self.model_path} "
                f"(classes: {list(self.class_names.values()) or 'unknown'})"
            )
            return True
        except Exception as e:
            logging.error(f"Failed to load fire model: {e}")
            self._load_failed = True
            return False

    def available(self):
        """Whether a usable model is loaded (used to decide the active detector)."""
        return self._ensure_loaded()

    def preload(self):
        """Eagerly load the model from a background thread."""
        return self._ensure_loaded()

    def detect(self, frame):
        return self.detect_fire(frame)

    def detect_fire(self, frame):
        """
        Run detection + temporal confirmation.

        Returns: numpy array of confirmed (x, y, w, h) boxes (possibly empty),
        matching the interface of the classic FireDetector so the GUI is
        unchanged.
        """
        if not self._ensure_loaded():
            return np.array([], dtype=np.int32)
        try:
            results = self._model(frame, verbose=False, conf=self.conf)
            raw = []
            for r in results:
                if r.boxes is None:
                    continue
                for b in r.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                    raw.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))

            confirmed = self._confirm(raw)
            if confirmed:
                return np.array(confirmed, dtype=np.int32)
            return np.array([], dtype=np.int32)
        except Exception as e:
            logging.error(f"Fire detection error: {e}")
            return np.array([], dtype=np.int32)

    def _confirm(self, boxes):
        """
        K-of-N temporal confirmation. Record this frame's boxes, then keep only
        boxes that overlap a detection in at least K of the last N frames (this
        frame included). Until the window is full we report nothing, to avoid
        firing on the very first noisy frame.
        """
        self._history.append(boxes)
        if len(self._history) < self.n:
            return []
        confirmed = []
        for b in boxes:
            hits = sum(
                1 for past in self._history
                if any(_iou(b, pb) > self.iou_link for pb in past)
            )
            if hits >= self.k:
                confirmed.append(b)
        return confirmed
