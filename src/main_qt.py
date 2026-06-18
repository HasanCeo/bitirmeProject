"""
Entry point for the modern PySide6 (Qt) UI.

Run with:  python src/main_qt.py
The classic Tkinter UI (src/main.py) is left intact as a fallback.
"""

import os
import sys
from pathlib import Path

# Single-threaded FFmpeg decode fallback (see src/main.py for rationale)
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;1")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging
setup_logging()

from PySide6.QtWidgets import QApplication
from src.gui.qt_main_window import MainWindow, STYLE


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
