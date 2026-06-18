"""Detection modules"""

from .motion_detector import MotionDetector
from .fire_detector import FireDetector
from .fire_smoke_detector import FireSmokeDetector

__all__ = [
    "MotionDetector",
    "FireDetector",
    "FireSmokeDetector",
]
