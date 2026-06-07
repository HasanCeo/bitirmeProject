"""Metadata management for detected objects"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from src.utils.file_utils import load_json, save_json
from src.config.settings import DETECTED_HUMANS_DIR, DETECTED_CARS_DIR


class MetadataManager:
    """Manages metadata for detected objects (humans, vehicles, pets)"""
    
    def __init__(self):
        """Initialize metadata manager"""
        self.human_metadata_file = DETECTED_HUMANS_DIR / "metadata.json"
        self.car_metadata_file = DETECTED_CARS_DIR / "metadata.json"
    
    def load_metadata(self, metadata_file: Path) -> Dict[str, Any]:
        """
        Load metadata from JSON file
        
        Args:
            metadata_file: Path to metadata JSON file
        
        Returns:
            dict: Metadata dictionary with 'images' list
        """
        data = load_json(metadata_file)
        if not data:
            return {'metadata_version': '1.0', 'images': []}
        return data
    
    def save_metadata(self, metadata_file: Path, metadata: Dict[str, Any]):
        """
        Save metadata to JSON file
        
        Args:
            metadata_file: Path to metadata JSON file
            metadata: Metadata dictionary to save
        """
        save_json(metadata_file, metadata)
    
    def add_image_metadata(self, metadata_file: Path, image_metadata: Dict[str, Any]):
        """
        Add a new image metadata entry to the JSON file
        
        Args:
            metadata_file: Path to metadata JSON file
            image_metadata: Metadata for a single image
        """
        metadata = self.load_metadata(metadata_file)
        if 'images' not in metadata:
            metadata['images'] = []
        metadata['images'].append(image_metadata)
        self.save_metadata(metadata_file, metadata)
