"""Base detector class"""

from abc import ABC, abstractmethod


class BaseDetector(ABC):
    """Base class for all detectors"""
    
    @abstractmethod
    def detect(self, frame):
        """
        Detect objects in a frame
        
        Args:
            frame: Input frame (BGR format)
        
        Returns:
            List of detections (format depends on detector type)
        """
        pass
