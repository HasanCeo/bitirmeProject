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
                 blacklist_manager: BlacklistManager,
                 human_parser=None):
        """
        Initialize photo manager

        Args:
            image_analyzer: Image analyzer for metadata extraction
            metadata_manager: Metadata manager for saving metadata
            blacklist_manager: Blacklist manager for security alerts
            human_parser: Optional HumanParser (SCHP-equivalent garment parser).
                When provided, the best saved human frame is parsed into garment
                regions and each garment's color is recorded. Falls back to the
                heuristic color estimate when None or when parsing fails.
        """
        self.image_analyzer = image_analyzer
        self.metadata_manager = metadata_manager
        self.blacklist_manager = blacklist_manager
        self.human_parser = human_parser
        
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
            # Preferred path: run the SCHP-equivalent garment parser on the best
            # frame to get real per-garment regions and their colors.
            parse_result = None
            if self.human_parser is not None:
                try:
                    parse_result = self.human_parser.parse_garments(
                        best_candidate['frame'],
                        best_candidate['bbox'],
                        self.image_analyzer,
                    )
                except Exception as e:
                    logging.error(f"Garment parsing failed: {e}")
                    parse_result = None

            if parse_result:
                filepath = self.save_human_photo(
                    best_candidate['frame'],
                    best_candidate['detection_tuple'],
                    best_candidate['timestamp'],
                    parse_result=parse_result
                )
            else:
                # Fallback: temporal voting across candidate frames (robust to
                # single-frame motion blur / lighting) using the heuristic
                # torso/legs color estimate.
                color_override = self.image_analyzer.vote_clothing_colors(
                    candidates_dict[track_id]
                )
                filepath = self.save_human_photo(
                    best_candidate['frame'],
                    best_candidate['detection_tuple'],
                    best_candidate['timestamp'],
                    color_override=color_override
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
    
    @staticmethod
    def _heuristic_to_garments(metadata_info: Dict[str, Any]):
        """
        Convert the heuristic torso/legs estimate (upper_clothing /
        lower_clothing) into the unified `garments` shape, so every human
        record has a single source of truth regardless of which path produced
        it. RGB / area_ratio are unknown for the heuristic and left as None.

        Returns: (garments, colors, detected_items)
        """
        garments: List[Dict[str, Any]] = []
        colors: List[str] = []
        detected_items: List[str] = []

        upper = metadata_info.get('upper_clothing', {}) or {}
        lower = metadata_info.get('lower_clothing', {}) or {}
        up_color = (upper.get('color') or '').strip()
        low_color = (lower.get('color') or '').strip()

        if up_color:
            garments.append({'type': 'upper-clothes', 'color': up_color,
                             'rgb': None, 'area_ratio': None})
            colors.append(up_color)
            detected_items.append('shirt')
        if low_color:
            garments.append({'type': 'pants', 'color': low_color,
                             'rgb': None, 'area_ratio': None})
            if low_color not in colors:
                colors.append(low_color)
            detected_items.append('pants')

        return garments, colors, detected_items

    def save_human_photo(self, frame, detection_tuple: Tuple, timestamp: datetime,
                         color_override: Optional[Dict[str, str]] = None,
                         parse_result: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Save human photo with metadata

        Args:
            frame: Full frame (BGR)
            detection_tuple: (x, y, w, h, track_id, mask)
            timestamp: Detection timestamp
            color_override: Optional {'upper': color, 'lower': color} from temporal
                voting across candidate frames (overrides single-frame estimate)
            parse_result: Optional output of HumanParser.parse_garments — a dict
                with 'garments', 'colors' and 'detected_items'. When present it
                takes priority over the heuristic estimate and color_override.

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
            
            # Build clothing metadata. `garments` is the SINGLE source of truth
            # (a list of {type, color, rgb, area_ratio}); `colors` and
            # `detected_items` are flat aggregates for fast query matching.
            # Preferred path: SCHP-equivalent garment parse (real per-garment
            # regions + colors). Otherwise fall back to the heuristic torso/legs
            # estimate, converted into the same garments shape.
            if parse_result:
                garments = parse_result.get('garments', [])
                colors = list(parse_result.get('colors', []))
                detected_items = list(parse_result.get('detected_items', []))
            else:
                metadata_info = self.image_analyzer.extract_metadata(
                    frame, object_type='human', bbox=bbox, mask=mask
                )

                # Apply temporal-voting colors if provided (more robust than 1 frame)
                if color_override:
                    if color_override.get('upper'):
                        metadata_info.setdefault('upper_clothing', {'type': 'shirt', 'color': ''})
                        metadata_info['upper_clothing']['color'] = color_override['upper']
                    if color_override.get('lower'):
                        metadata_info.setdefault('lower_clothing', {'type': 'pants', 'color': ''})
                        metadata_info['lower_clothing']['color'] = color_override['lower']

                garments, colors, detected_items = self._heuristic_to_garments(metadata_info)

            # Calculate quality score
            quality_score = calculate_frame_quality_score(frame, bbox, frame.shape)

            # Create metadata entry
            image_metadata = {
                'filename': filename,
                'filepath': str(filepath),
                'object_type': 'human',
                'track_id': int(track_id),
                'garments': garments,
                'colors': colors,
                'detected_items': detected_items,
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
            
            # Record this save FIRST, so the track_id dedup (should_save_human)
            # always sees it — even on a blacklist match, which returns early.
            # (Previously the early return skipped this, letting a watch-listed
            # person be saved repeatedly under the same track_id.)
            self.saved_humans.append({
                'timestamp': timestamp,
                'bbox': bbox,
                'filepath': str(filepath),
                'track_id': track_id
            })

            # Check blacklist
            is_match, matched_entry = self.blacklist_manager.check_match(image_metadata, 'human')
            if is_match and self.blacklist_manager.should_trigger_alert(matched_entry):
                self.blacklist_manager.record_alert(matched_entry, str(filepath))

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
