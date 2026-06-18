"""
Main entry point for Security Camera Monitoring System
"""

import os
import sys
from pathlib import Path

# Force FFmpeg to decode video files single-threaded. OpenCV's default frame-
# multithreaded H.264 decoder races with the Torch / YOLO inference threads on
# macOS, corrupting the decoder ("Invalid NAL unit size" / "Assertion
# fctx->async_lock failed" abort). Single-threaded decode is plenty fast for
# webcam-resolution clips and removes the crash. Read by OpenCV when a capture
# is opened, so it must be set before any cv2.VideoCapture call.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;1")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging
from src.gui.main_window import WebcamMonitor

# Setup logging
setup_logging()
print("Logging setup complete")

if __name__ == "__main__":
    import tkinter as tk
    
    root = tk.Tk()
    app = WebcamMonitor(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
