# Security Camera Monitoring System

A comprehensive security camera monitoring system with real-time detection of humans, vehicles, pets, and fire using computer vision and machine learning.

## Features

- **Human Detection**: Real-time human detection with tracking and metadata extraction
- **Vehicle Detection**: Detection of cars, trucks, buses, and motorcycles
- **Pet Detection**: Detection of cats and dogs
- **Fire Detection**: Color-based fire detection with flicker analysis
- **Search Functionality**: Search detected objects by color, clothing, and time range
- **Blacklist System**: Security alerts for blacklisted objects
- **Quality-Based Photo Saving**: Automatically saves best quality frames
- **Metadata Management**: JSON-based metadata storage for all detections

## Project Structure

```
bitirmeyeni/
├── src/                          # Main source code
│   ├── main.py                   # Entry point
│   ├── gui/                      # GUI components
│   │   └── main_window.py        # Main application window
│   ├── detectors/                # Detection modules
│   │   ├── motion_detector.py
│   │   ├── human_detector.py
│   │   ├── car_detector.py
│   │   ├── pet_detector.py
│   │   └── fire_detector.py
│   ├── tracking/                 # Object tracking
│   │   └── sort.py               # SORT tracker
│   ├── analysis/                 # Image analysis
│   │   └── image_analyzer.py     # Color and metadata extraction
│   ├── core/                     # Core business logic
│   │   ├── metadata_manager.py
│   │   ├── blacklist_manager.py
│   │   ├── photo_manager.py
│   │   └── search_engine.py
│   ├── utils/                    # Utility functions
│   │   ├── logger.py
│   │   ├── image_utils.py
│   │   ├── quality_scorer.py
│   │   └── file_utils.py
│   └── config/                   # Configuration
│       ├── settings.py
│       └── constants.py
├── data/                         # Data directories
│   ├── detected_humans/
│   ├── detected_cars/
│   ├── detected_pets/
│   └── logs/
└── requirements.txt
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python src/main.py
```

Or from the project root:
```bash
python -m src.main
```

## Configuration

- Detection hours can be set in the GUI
- Blacklist entries can be added through the GUI
- All settings are in `src/config/settings.py`

## Dependencies

- OpenCV (cv2)
- NumPy
- Ultralytics (YOLO)
- scikit-learn
- PIL/Pillow
- tkinter
- filterpy (for SORT tracker)

## License

See individual file headers for license information.
