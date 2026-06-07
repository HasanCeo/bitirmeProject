"""Image processing utilities"""

import cv2
import numpy as np


def resize_frame(frame, target_width=640, target_height=480):
    """Resize frame to target dimensions"""
    if frame.shape[:2] != (target_height, target_width):
        return cv2.resize(frame, (target_width, target_height))
    return frame


def calculate_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes
    bbox format: (x, y, w, h)
    Returns IoU value between 0 and 1
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Calculate intersection
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area
