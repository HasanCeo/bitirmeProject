"""Fire detection module"""

import logging
import cv2
import numpy as np

from src.config.settings import DEFAULT_FIRE_MIN_AREA
from .base import BaseDetector


class FireDetector(BaseDetector):
    """Detects fire using color-based detection with temporal analysis"""
    
    def __init__(self, min_area=100, history_size=5):
        """
        Initialize fire detector
        
        Args:
            min_area: Minimum area for fire detection
            history_size: Number of frames to keep for flicker detection
        """
        self.min_area = min_area
        self.history_size = history_size
        self.frame_history = []
        self.mask_history = []

    def detect(self, frame):
        """Detect fire in the frame"""
        return self.detect_fire(frame)
    
    def detect_fire(self, frame, detect_flicker=True):
        """
        Detect fire in the frame using color-based detection with flicker analysis
        
        Args:
            frame: Input frame (BGR format)
            detect_flicker: Whether to use flicker detection to filter static objects
        
        Returns:
            numpy array: Array of bounding boxes (x, y, w, h)
        """
        try:
            # Convert BGR to HSV for better color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Define color ranges for fire (red, orange, yellow)
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            lower_orange = np.array([10, 100, 100])
            upper_orange = np.array([25, 255, 255])
            lower_yellow = np.array([25, 100, 100])
            upper_yellow = np.array([35, 255, 255])

            # Create masks for each color range
            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

            # Combine all masks
            fire_mask = cv2.bitwise_or(mask_red1, mask_red2)
            fire_mask = cv2.bitwise_or(fire_mask, mask_orange)
            fire_mask = cv2.bitwise_or(fire_mask, mask_yellow)

            # Apply morphological operations to reduce noise
            kernel = np.ones((5, 5), np.uint8)
            fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)
            fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)

            # Additional filtering: fire should be bright
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, brightness_mask = cv2.threshold(gray, 100, 220, cv2.THRESH_BINARY)
            fire_mask = cv2.bitwise_and(fire_mask, brightness_mask)

            # Kare boyutu değiştiyse (örn. webcam'den videoya geçiş) eski
            # maskeler farklı boyutta olur; absdiff hata verir. Geçmişi temizle.
            if self.mask_history and self.mask_history[-1].shape != fire_mask.shape:
                self.mask_history = []

            # Flicker detection: fire changes over time, lamps are static
            if detect_flicker and len(self.mask_history) >= 3:
                flicker_mask = np.zeros_like(fire_mask)
                for prev_mask in self.mask_history[-3:]:
                    diff = cv2.absdiff(fire_mask, prev_mask)
                    flicker_mask = cv2.bitwise_or(flicker_mask, diff)
                fire_mask = cv2.bitwise_and(fire_mask, flicker_mask)
                fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)

            # Update history
            self.mask_history.append(fire_mask.copy())
            if len(self.mask_history) > self.history_size:
                self.mask_history.pop(0)

            # Find contours
            contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Extract bounding boxes with additional filtering
            rects = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    if (w > 20 and h > 20 and 
                        w < frame.shape[1] * 0.8 and 
                        h < frame.shape[0] * 0.8):
                        extent = area / (w * h)
                        aspect_ratio = float(w) / h if h > 0 else 0
                        if extent < 0.75 and 0.3 < aspect_ratio < 3.0:
                            rects.append((x, y, w, h))

            return np.array(rects, dtype=np.int32) if len(rects) > 0 else np.array([], dtype=np.int32)
        except Exception as e:
            logging.error(f"Error during fire detection: {e}")
            return np.array([], dtype=np.int32)
