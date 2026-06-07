"""Motion detection module"""

import cv2
import logging

from .base import BaseDetector


class MotionDetector(BaseDetector):
    """Detects motion in video frames using background subtraction"""
    
    def __init__(self, threshold=30, min_area=500):
        """
        Initialize motion detector
        
        Args:
            threshold: Threshold for motion detection
            min_area: Minimum area for motion to be considered valid
        """
        self.threshold = threshold
        self.min_area = min_area
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    def detect(self, frame):
        """Detect motion in the frame"""
        return self.detect_motion(frame)
    
    def detect_motion(self, frame):
        """
        Detect motion in the frame
        
        Args:
            frame: Input frame (BGR format)
        
        Returns:
            tuple: (motion_detected: bool, motion_mask: numpy array)
        """
        fg_mask = self.bg_subtractor.apply(frame)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                motion_detected = True
                break

        return motion_detected, thresh
