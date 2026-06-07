"""Logging configuration"""

import logging
import os
from pathlib import Path

from src.config.settings import LOGS_DIR


def setup_logging():
    """Configure logging for the application"""
    log_file = LOGS_DIR / "detection.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)
