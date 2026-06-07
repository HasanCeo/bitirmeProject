"""Application settings and configuration"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
DETECTED_HUMANS_DIR = DATA_DIR / "detected_humans"
DETECTED_CARS_DIR = DATA_DIR / "detected_cars"
DETECTED_PETS_DIR = DATA_DIR / "detected_pets"
LOGS_DIR = DATA_DIR / "logs"
MODELS_DIR = DATA_DIR / "models"

# Configuration files
BLACKLIST_FILE = BASE_DIR / "blacklist.json"

# Create data directories if they don't exist
for directory in [DATA_DIR, DETECTED_HUMANS_DIR, DETECTED_CARS_DIR, 
                  DETECTED_PETS_DIR, LOGS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model paths - use local model (data/models or project root) if present, else download
_candidates = [MODELS_DIR / "yolov8n-seg.pt", BASE_DIR / "yolov8n-seg.pt"]
YOLO_MODEL_PATH = next((str(p) for p in _candidates if p.exists()), "yolov8n-seg.pt")

# Detection settings
DEFAULT_MOTION_THRESHOLD = 30
DEFAULT_MOTION_MIN_AREA = 500
DEFAULT_FIRE_MIN_AREA = 100

# SORT tracker settings
SORT_MAX_AGE = 30
SORT_MIN_HITS = 3
SORT_IOU_THRESHOLD = 0.3

# Video settings
DEFAULT_VIDEO_WIDTH = 640
DEFAULT_VIDEO_HEIGHT = 480
DEFAULT_VIDEO_FPS = 30

# ESP32-CAM Settings
ESP32_CAM_IP = "192.168.1.100"                 # ESP32-CAM IP adresi (Seri monitörden kontrol edin)
ESP32_CAM_STREAM_PORT = 81                     # MJPEG stream portu
ESP32_CAM_HTTP_PORT = 80                       # HTTP API portu (capture, status, led)
ESP32_CAM_STREAM_URL = f"http://{ESP32_CAM_IP}:{ESP32_CAM_STREAM_PORT}/stream"
ESP32_CAM_CAPTURE_URL = f"http://{ESP32_CAM_IP}:{ESP32_CAM_HTTP_PORT}/capture"
ESP32_CAM_STATUS_URL = f"http://{ESP32_CAM_IP}:{ESP32_CAM_HTTP_PORT}/status"
ESP32_CAM_RECONNECT_DELAY = 3                  # Bağlantı koparsa yeniden deneme süresi (saniye)
ESP32_CAM_TIMEOUT = 10                         # HTTP bağlantı zaman aşımı (saniye)
