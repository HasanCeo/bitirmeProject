"""Photo management and saving for detected objects"""

import logging
import cv2
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from src.config.settings import DETECTED_HUMANS_DIR, DETECTED_CARS_DIR, DETECTED_PETS_DIR
from src.config.constants import (
    COLOR_HUMAN, COLOR_VEHICLE, COLOR_PET,
    MAX_CANDIDATES_PER_TRACK, CANDIDATE_COLLECTION_FRAMES, MIN_CANDIDATES_FOR_SAVE
)
from src.utils.quality_scorer import calculate_frame_quality_score
from src.core.metadata_manager import MetadataManager
from src.core.blacklist_manager import BlacklistManager
from src.analysis.image_analyzer import HumanImageAnalyzer


class PhotoManager:
    """Manages photo saving, candidate frame collection, and quality-based selection"""
    
    def __init__(self, image_analyzer: HumanImageAnalyzer, 
                 metadata_manager: MetadataManager,
                 blacklist_manager: BlacklistManager):
        """
        Initialize photo manager
        
        Args:
            image_analyzer: Image analyzer for metadata extraction
            metadata_manager: Metadata manager for saving metadata
            blacklist_manager: Blacklist manager for security alerts
        """
        self.image_analyzer = image_analyzer
        self.metadata_manager = metadata_manager
        self.blacklist_manager = blacklist_manager
        
        # Photo directories
        self.human_photos_dir = DETECTED_HUMANS_DIR
        self.car_photos_dir = DETECTED_CARS_DIR
        self.pet_photos_dir = DETECTED_PETS_DIR
        
        # Track saved objects to avoid duplicates
        self.saved_humans: List[Dict[str, Any]] = []
        self.saved_cars: List[Dict[str, Any]] = []
        self.saved_pets: List[Dict[str, Any]] = []
        
        # Track candidate frames for quality-based selection
        self.human_track_candidates: Dict[int, List[Dict[str, Any]]] = {}
        self.car_track_candidates: Dict[int, List[Dict[str, Any]]] = {}
        self.pet_track_candidates: Dict[int, List[Dict[str, Any]]] = {}
    
    def add_candidate_frame(self, track_id: int, frame, detection_tuple: Tuple, 
                           timestamp: datetime, is_human: bool = True):
        """
        Add a candidate frame for a track ID
        Keeps only the best frames based on quality score
        
        Args:
            track_id: Track ID
            frame: Full frame (BGR)
            detection_tuple: Detection tuple (format depends on object type)
            timestamp: Detection timestamp
            is_human: True for humans, False for vehicles/pets
        """
        x, y, w, h = detection_tuple[:4]
        bbox = (x, y, w, h)
        
        # Calculate quality score
        quality_score = calculate_frame_quality_score(frame, bbox, frame.shape)
        
        # Create candidate entry
        candidate = {
            'frame': frame.copy(),
            'detection_tuple': detection_tuple,
            'bbox': bbox,
            'timestamp': timestamp,
            'quality_score': quality_score
        }
        
        # Add to appropriate dictionary
        if is_human:
            candidates_dict = self.human_track_candidates
            max_candidates = MAX_CANDIDATES_PER_TRACK
        else:
            # Check class_name at index 4
            class_name = detection_tuple[4] if len(detection_tuple) > 4 else None
            if class_name in ['cat', 'dog']:
                candidates_dict = self.pet_track_candidates
                max_candidates = MAX_CANDIDATES_PER_TRACK
            else:  # Vehicle
                candidates_dict = self.car_track_candidates
                max_candidates = MAX_CANDIDATES_PER_TRACK
        
        if track_id not in candidates_dict:
            candidates_dict[track_id] = []
        
        candidates_dict[track_id].append(candidate)
        
        # Keep only top candidates (sorted by quality score)
        candidates_dict[track_id].sort(key=lambda x: x['quality_score'], reverse=True)
        candidates_dict[track_id] = candidates_dict[track_id][:max_candidates]
    
    def get_best_candidate_and_save(self, track_id: int, is_human: bool = True) -> Optional[str]:
        """
        Get the best candidate frame for a track and save it
        Called when track is lost or after collection period
        
        Args:
            track_id: Track ID
            is_human: True for humans, False for vehicles
        
        Returns:
            Optional[str]: Filepath if saved successfully, None otherwise
        """
        candidates_dict = self.human_track_candidates if is_human else self.car_track_candidates
        
        if track_id not in candidates_dict or len(candidates_dict[track_id]) == 0:
            return None
        
        # Require minimum frames to ensure quality selection
        if len(candidates_dict[track_id]) < MIN_CANDIDATES_FOR_SAVE:
            del candidates_dict[track_id]
            return None
        
        # Get best candidate (highest quality score)
        best_candidate = candidates_dict[track_id][0]
        
        # Save the best frame
        if is_human:
            filepath = self.save_human_photo(
                best_candidate['frame'],
                best_candidate['detection_tuple'],
                best_candidate['timestamp']
            )
        else:
            filepath = self.save_car_photo(
                best_candidate['frame'],
                best_candidate['detection_tuple'],
                best_candidate['timestamp']
            )
        
        # Clean up candidates for this track
        del candidates_dict[track_id]
        
        return filepath
    
    def should_save_human(self, detection_tuple: Tuple, current_time: datetime) -> bool:
        """
        Check if this human should be saved (not a duplicate) using track_id
        
        Args:
            detection_tuple: (x, y, w, h, track_id, mask)
            current_time: Current timestamp
        
        Returns:
            bool: True if should save, False if duplicate
        """
        x, y, w, h, track_id, mask = detection_tuple
        
        # Check if this track_id has been saved before
        for saved_human in self.saved_humans:
            if saved_human.get('track_id') == track_id:
                return False
        
        return True
    
    def should_save_car(self, detection_tuple: Tuple, current_time: datetime) -> bool:
        """Check if this vehicle should be saved (not a duplicate)"""
        x, y, w, h, class_name, track_id, mask = detection_tuple
        
        for saved_car in self.saved_cars:
            if saved_car.get('track_id') == track_id:
                return False
        
        return True
    
    def should_save_pet(self, detection_tuple: Tuple, current_time: datetime) -> bool:
        """Check if this pet should be saved (not a duplicate)"""
        x, y, w, h, class_name, track_id, mask = detection_tuple
        
        for saved_pet in self.saved_pets:
            if saved_pet.get('track_id') == track_id:
                return False
        
        return True
    
    def save_human_photo(self, frame, detection_tuple: Tuple, timestamp: datetime) -> Optional[str]:
        """
        Save human photo with metadata
        
        Args:
            frame: Full frame (BGR)
            detection_tuple: (x, y, w, h, track_id, mask)
            timestamp: Detection timestamp
        
        Returns:
            Optional[str]: Filepath if saved successfully, None otherwise
        """
        try:
            x, y, w, h, track_id, mask = detection_tuple
            bbox = (x, y, w, h)
            
            # Create a copy of the frame to draw on
            frame_copy = frame.copy()
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), COLOR_HUMAN, 2)
            cv2.putText(frame_copy, f"Human ID:{track_id}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HUMAN, 2)
            
            # Generate filename
            filename = f"human_{timestamp.strftime('%Y%m%d_%H%M%S')}_id{track_id}.jpg"
            filepath = self.human_photos_dir / filename
            
            # Save frame
            cv2.imwrite(str(filepath), frame_copy)
            
            # Extract metadata
            metadata_info = self.image_analyzer.extract_metadata(
                frame, object_type='human', bbox=bbox, mask=mask
            )
            
            # Calculate quality score
            quality_score = calculate_frame_quality_score(frame, bbox, frame.shape)
            
            # Create metadata entry
            colors = []
            upper_clothing = metadata_info.get('upper_clothing', {'type': '', 'color': ''})
            lower_clothing = metadata_info.get('lower_clothing', {'type': '', 'color': ''})
            if upper_clothing.get('color'):
                colors.append(upper_clothing['color'])
            if lower_clothing.get('color') and lower_clothing['color'] not in colors:
                colors.append(lower_clothing['color'])
            
            image_metadata = {
                'filename': filename,
                'filepath': str(filepath),
                'object_type': 'human',
                'track_id': int(track_id),
                'upper_clothing': upper_clothing,
                'lower_clothing': lower_clothing,
                'colors': colors,
                'detected_items': metadata_info.get('detected_items', []),
                'timestamp': timestamp.isoformat(),
                'hour': int(timestamp.hour),
                'bbox': [int(x), int(y), int(w), int(h)],
                'confidence': 0.8,
                'quality_score': float(quality_score)
            }
            
            # Save metadata
            self.metadata_manager.add_image_metadata(
                self.metadata_manager.human_metadata_file, image_metadata
            )
            
            # Check blacklist
            is_match, matched_entry = self.blacklist_manager.check_match(image_metadata, 'human')
            if is_match and self.blacklist_manager.should_trigger_alert(matched_entry):
                self.blacklist_manager.record_alert(matched_entry, str(filepath))
                # Return filepath for alert handling
                return str(filepath)
            
            # Add to saved humans list
            self.saved_humans.append({
                'timestamp': timestamp,
                'bbox': bbox,
                'filepath': str(filepath),
                'track_id': track_id
            })
            
            return str(filepath)
        except Exception as e:
            logging.error(f"Error saving human photo: {e}")
            return None
    
    def save_car_photo(self, frame, detection_tuple: Tuple, timestamp: datetime) -> Optional[str]:
        """Save vehicle photo with metadata"""
        try:
            x, y, w, h, class_name, track_id, mask = detection_tuple
            bbox = (x, y, w, h)
            
            # Create a copy of the frame to draw on
            frame_copy = frame.copy()
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), COLOR_VEHICLE, 2)
            cv2.putText(frame_copy, f"{class_name.capitalize()} ID:{track_id}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_VEHICLE, 2)
            
            # Generate filename
            filename = f"{class_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}_id{track_id}.jpg"
            filepath = self.car_photos_dir / filename
            
            # Save frame
            cv2.imwrite(str(filepath), frame_copy)
            
            # Extract metadata
            metadata_info = self.image_analyzer.extract_metadata(
                frame, object_type=class_name, bbox=bbox, mask=mask
            )
            
            # Calculate quality score
            quality_score = calculate_frame_quality_score(frame, bbox, frame.shape)
            
            # Create metadata entry
            image_metadata = {
                'filename': filename,
                'filepath': str(filepath),
                'object_type': class_name,
                'track_id': int(track_id),
                'colors': metadata_info['colors'],
                'detected_items': metadata_info['detected_items'],
                'timestamp': timestamp.isoformat(),
                'hour': int(timestamp.hour),
                'bbox': [int(x), int(y), int(w), int(h)],
                'confidence': 0.85,
                'quality_score': float(quality_score)
            }
            
            # Save metadata
            self.metadata_manager.add_image_metadata(
                self.metadata_manager.car_metadata_file, image_metadata
            )
            
            # Check blacklist
            is_match, matched_entry = self.blacklist_manager.check_match(image_metadata, 'vehicle')
            if is_match and self.blacklist_manager.should_trigger_alert(matched_entry):
                self.blacklist_manager.record_alert(matched_entry, str(filepath))
                return str(filepath)
            
            # Add to saved cars list
            self.saved_cars.append({
                'timestamp': timestamp,
                'bbox': bbox,
                'class_name': class_name,
                'filepath': str(filepath),
                'track_id': track_id
            })
            
            return str(filepath)
        except Exception as e:
            logging.error(f"Error saving car photo: {e}")
            return None
    
    def save_pet_photo(self, frame, detection_tuple: Tuple, timestamp: datetime) -> Optional[str]:
        """Save pet photo (no metadata JSON)"""
        try:
            x, y, w, h, class_name, track_id, mask = detection_tuple
            bbox = (x, y, w, h)
            
            # Create a copy of the frame to draw on
            frame_copy = frame.copy()
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), COLOR_PET, 2)
            emoji = "🐱" if class_name == "cat" else "🐶"
            cv2.putText(frame_copy, f"{emoji} {class_name.capitalize()} ID:{track_id}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PET, 2)
            
            # Generate filename
            filename = f"{class_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}_id{track_id}.jpg"
            filepath = self.pet_photos_dir / filename
            
            # Save frame
            cv2.imwrite(str(filepath), frame_copy)
            
            # Check blacklist (simple metadata)
            simple_metadata = {
                'object_type': class_name,
                'track_id': track_id
            }
            is_match, matched_entry = self.blacklist_manager.check_match(simple_metadata, class_name)
            if is_match and self.blacklist_manager.should_trigger_alert(matched_entry):
                self.blacklist_manager.record_alert(matched_entry, str(filepath))
                return str(filepath)
            
            # Add to saved pets list
            self.saved_pets.append({
                'timestamp': timestamp,
                'bbox': bbox,
                'class_name': class_name,
                'filepath': str(filepath),
                'track_id': track_id
            })
            
            return str(filepath)
        except Exception as e:
            logging.error(f"Error saving pet photo: {e}")
            return None
