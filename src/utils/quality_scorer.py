"""Frame quality scoring utilities"""

import cv2
import numpy as np

from src.config.constants import (
    QUALITY_SIZE_WEIGHT, QUALITY_BLUR_WEIGHT, 
    QUALITY_CENTER_WEIGHT, QUALITY_ASPECT_WEIGHT
)


def calculate_frame_quality_score(frame, bbox, frame_shape):
    """
    Calculate quality score for a detected object frame
    Higher score = better quality (clearer, larger, more centered)
    
    Args:
        frame: Full frame (BGR)
        bbox: Bounding box (x, y, w, h)
        frame_shape: Shape of the frame (height, width, channels)
    
    Returns: float score (0-100)
    """
    x, y, w, h = bbox
    frame_height, frame_width = frame_shape[:2]
    
    # 1. Size score (30 points) - larger bbox = closer/clearer
    bbox_area = w * h
    frame_area = frame_width * frame_height
    size_ratio = bbox_area / frame_area
    size_score = min(size_ratio * 100, QUALITY_SIZE_WEIGHT)
    
    # 2. Blur score (40 points) - Laplacian variance for sharpness
    try:
        roi = frame[y:y+h, x:x+w]
        if roi.size > 0:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
            # Normalize: typical sharp images have variance > 100
            blur_score = min(laplacian_var / 5, QUALITY_BLUR_WEIGHT)
        else:
            blur_score = 0
    except:
        blur_score = 0
    
    # 3. Center proximity (20 points) - objects near center are usually better framed
    center_x = x + w/2
    center_y = y + h/2
    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2
    
    # Calculate distance from center (normalized)
    dist_x = abs(center_x - frame_center_x) / frame_center_x
    dist_y = abs(center_y - frame_center_y) / frame_center_y
    center_dist = (dist_x + dist_y) / 2
    center_score = max(0, QUALITY_CENTER_WEIGHT * (1 - center_dist))
    
    # 4. Aspect ratio score (10 points) - check if bbox is not too distorted
    aspect_ratio = w / h if h > 0 else 0
    # For humans: ideal ratio ~0.4-0.6, for cars: ~1.5-2.5
    # Give points for reasonable aspect ratios
    if 0.3 < aspect_ratio < 3.0:
        aspect_score = QUALITY_ASPECT_WEIGHT
    else:
        aspect_score = QUALITY_ASPECT_WEIGHT / 2
    
    total_score = size_score + blur_score + center_score + aspect_score
    return total_score
