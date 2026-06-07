"""File operation utilities"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


def load_json(filepath: Path) -> Dict[str, Any]:
    """Load JSON file, return empty dict if file doesn't exist"""
    try:
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Convert timestamp strings back to datetime objects
                if 'images' in data:
                    for entry in data['images']:
                        if 'timestamp' in entry and isinstance(entry['timestamp'], str):
                            entry['timestamp'] = datetime.fromisoformat(entry['timestamp'])
                return data
        return {}
    except Exception as e:
        import logging
        logging.error(f"Error loading JSON from {filepath}: {e}")
        return {}


def save_json(filepath: Path, data: Dict[str, Any]):
    """Save data to JSON file, converting datetime objects to ISO strings"""
    try:
        # Create a copy to avoid modifying original
        data_copy = data.copy()
        
        # Convert datetime objects to ISO format strings
        if 'images' in data_copy:
            data_copy['images'] = []
            for entry in data.get('images', []):
                entry_copy = entry.copy()
                if 'timestamp' in entry_copy:
                    if isinstance(entry_copy['timestamp'], datetime):
                        entry_copy['timestamp'] = entry_copy['timestamp'].isoformat()
                    elif not isinstance(entry_copy['timestamp'], str):
                        entry_copy['timestamp'] = str(entry_copy['timestamp'])
                
                # Convert numpy types to Python native types
                if 'bbox' in entry_copy:
                    if isinstance(entry_copy['bbox'], (list, tuple)):
                        entry_copy['bbox'] = [int(x) for x in entry_copy['bbox']]
                    else:
                        entry_copy['bbox'] = []
                
                if 'hour' in entry_copy:
                    entry_copy['hour'] = int(entry_copy['hour'])
                
                if 'confidence' in entry_copy:
                    entry_copy['confidence'] = float(entry_copy['confidence'])
                
                data_copy['images'].append(entry_copy)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_copy, f, indent=2, ensure_ascii=False)
    except Exception as e:
        import logging
        logging.error(f"Error saving JSON to {filepath}: {e}")
        import traceback
        logging.error(traceback.format_exc())
