"""
Main entry point for Security Camera Monitoring System
"""

import sys
from pathlib import Path

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
