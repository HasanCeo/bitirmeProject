"""
Object detection using YOLOv8n-seg.
Single-pass detection for humans, vehicles, and pets with SORT tracking.
"""

import logging
from typing import List, Tuple, Optional

import numpy as np
from ultralytics import YOLO

from src.config.settings import (
    SORT_MAX_AGE,
    SORT_MIN_HITS,
    SORT_IOU_THRESHOLD,
    YOLO_MODEL_PATH,
)
from src.config.constants import (
    COCO_CLASS_PERSON,
    COCO_CLASS_CAR,
    COCO_CLASS_MOTORCYCLE,
    COCO_CLASS_BUS,
    COCO_CLASS_TRUCK,
    COCO_CLASS_CAT,
    COCO_CLASS_DOG,
    VEHICLE_CLASS_NAMES,
    PET_CLASS_NAMES,
)
from src.tracking.sort import Sort

# Detection formats: human (x,y,w,h,track_id,mask), vehicle/pet (x,y,w,h,class_name,track_id,mask)
HumanDet = Tuple[int, int, int, int, int, Optional[np.ndarray]]
VehicleDet = Tuple[int, int, int, int, str, int, Optional[np.ndarray]]
PetDet = Tuple[int, int, int, int, str, int, Optional[np.ndarray]]


def _iou_box(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _match_tracks_to_detections(
    track_boxes: np.ndarray,
    det_boxes: np.ndarray,
) -> List[Optional[int]]:
    """For each track box, return the best matching detection index by IoU, or None."""
    if track_boxes.size == 0 or len(det_boxes) == 0:
        return []
    out = []
    used = set()
    for t in track_boxes:
        tb = np.array(t[:4], dtype=float)
        best_iou, best_idx = 0.0, -1
        for j, db in enumerate(det_boxes):
            if j in used:
                continue
            iou = _iou_box(tb, db[:4])
            if iou > best_iou:
                best_iou, best_idx = iou, j
        if best_idx >= 0 and best_iou > 0.1:
            out.append(best_idx)
            used.add(best_idx)
        else:
            out.append(None)
    return out


class YoloMultiObjectDetector:
    """YOLOv8n-seg detector for humans, vehicles, and pets with SORT tracking."""

    # COCO class IDs we care about
    PERSON_IDS = (COCO_CLASS_PERSON,)
    VEHICLE_IDS = (COCO_CLASS_CAR, COCO_CLASS_MOTORCYCLE, COCO_CLASS_BUS, COCO_CLASS_TRUCK)
    PET_IDS = (COCO_CLASS_CAT, COCO_CLASS_DOG)

    def __init__(self) -> None:
        self.model = YOLO(YOLO_MODEL_PATH)
        logging.info(f"YOLOv8-seg loaded from {YOLO_MODEL_PATH}")

        self._human_tracker = Sort(
            max_age=SORT_MAX_AGE,
            min_hits=SORT_MIN_HITS,
            iou_threshold=SORT_IOU_THRESHOLD,
        )
        self._vehicle_tracker = Sort(
            max_age=SORT_MAX_AGE,
            min_hits=SORT_MIN_HITS,
            iou_threshold=SORT_IOU_THRESHOLD,
        )
        self._pet_tracker = Sort(
            max_age=SORT_MAX_AGE,
            min_hits=SORT_MIN_HITS,
            iou_threshold=SORT_IOU_THRESHOLD,
        )

        self._all_classes = list(self.PERSON_IDS) + list(self.VEHICLE_IDS) + list(self.PET_IDS)

    def detect_all(
        self, frame
    ) -> Tuple[List[HumanDet], List[VehicleDet], List[PetDet]]:
        """
        Run one YOLOv8-seg pass and return (humans, vehicles, pets).
        humans: [(x, y, w, h, track_id, mask)]
        vehicles: [(x, y, w, h, class_name, track_id, mask)]
        pets: [(x, y, w, h, class_name, track_id, mask)]
        """
        try:
            results = self.model(frame, verbose=False, classes=self._all_classes)

            hum_dets, hum_masks = [], []
            veh_dets, veh_cls, veh_masks = [], [], []
            pet_dets, pet_cls, pet_masks = [], [], []

            for r in results:
                if r.masks is None:
                    continue
                boxes = r.boxes
                for i, box in enumerate(boxes):
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cid = int(box.cls[0].cpu().numpy())
                    mask = r.masks.data[i].cpu().numpy()
                    row = [*xyxy, conf]

                    if cid in self.PERSON_IDS:
                        hum_dets.append(row)
                        hum_masks.append(mask)
                    elif cid in self.VEHICLE_IDS:
                        veh_dets.append(row)
                        veh_cls.append(cid)
                        veh_masks.append(mask)
                    elif cid in self.PET_IDS:
                        pet_dets.append(row)
                        pet_cls.append(cid)
                        pet_masks.append(mask)

            humans = self._run_tracker(
                hum_dets, hum_masks,
                self._human_tracker,
                {}, "person",
                format_human=True,
            )
            vehicles = self._run_tracker(
                veh_dets, veh_masks,
                self._vehicle_tracker,
                VEHICLE_CLASS_NAMES, "vehicle",
                format_human=False,
                class_ids=veh_cls,
            )
            pets = self._run_tracker(
                pet_dets, pet_masks,
                self._pet_tracker,
                PET_CLASS_NAMES, "pet",
                format_human=False,
                class_ids=pet_cls,
            )

            return humans, vehicles, pets

        except Exception as e:
            logging.error(f"YOLO detection error: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return [], [], []

    def _run_tracker(
        self,
        dets: List,
        masks: List,
        tracker: Sort,
        class_name_map: dict,
        default_class: str,
        format_human: bool,
        class_ids: Optional[List[int]] = None,
    ) -> List:
        if not dets:
            tracker.update(np.empty((0, 5)))
            return []

        arr = np.array(dets)
        tracks = tracker.update(arr)

        if tracks.size == 0:
            return []
        if tracks.ndim == 1:
            tracks = tracks.reshape(1, -1)

        track_boxes = tracks[:, :4]
        det_boxes = arr[:, :4]
        matched_det_idx = _match_tracks_to_detections(track_boxes, det_boxes)

        out = []
        for k, t in enumerate(tracks):
            x1, y1, x2, y2 = t[0], t[1], t[2], t[3]
            tid = int(t[4])
            x, y = int(x1), int(y1)
            w, h = int(x2 - x1), int(y2 - y1)

            det_idx = matched_det_idx[k] if k < len(matched_det_idx) else None
            mask = masks[det_idx] if det_idx is not None and det_idx < len(masks) else None

            if format_human:
                out.append((x, y, w, h, tid, mask))
            else:
                cname = default_class
                if class_ids and det_idx is not None and det_idx < len(class_ids):
                    cname = class_name_map.get(class_ids[det_idx], default_class)
                out.append((x, y, w, h, cname, tid, mask))

        return out
