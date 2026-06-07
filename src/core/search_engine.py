"""Search engine for finding detected objects by query"""

import logging
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

from src.core.metadata_manager import MetadataManager
from src.analysis.image_analyzer import HumanImageAnalyzer


class SearchEngine:
    """Search engine for finding detected objects matching queries"""
    
    def __init__(self, image_analyzer: HumanImageAnalyzer, 
                 metadata_manager: MetadataManager):
        """
        Initialize search engine
        
        Args:
            image_analyzer: Image analyzer for query parsing
            metadata_manager: Metadata manager for loading metadata
        """
        self.image_analyzer = image_analyzer
        self.metadata_manager = metadata_manager
    
    def search(self, query: str, start_hour: int, end_hour: int) -> List[Tuple[str, str, float]]:
        """
        Search for objects matching the query in the specified time range
        
        Args:
            query: Search query (e.g., "red hat", "blue car")
            start_hour: Start hour (0-23)
            end_hour: End hour (0-23)
        
        Returns:
            List of tuples: (filepath, filename, confidence)
        """
        try:
            query_lower = query.lower()
            color, item, region = self.image_analyzer.parse_query(query)
            vehicle_types = ['car', 'truck', 'bus', 'motorcycle']
            is_vehicle_search = any(vehicle in query_lower for vehicle in vehicle_types)
            
            matching_images = []
            
            # Search in human photos
            if not is_vehicle_search:
                metadata = self.metadata_manager.load_metadata(
                    self.metadata_manager.human_metadata_file
                )
                
                for entry in metadata.get('images', []):
                    # Check time range
                    entry_hour = entry.get('hour', -1)
                    if start_hour <= end_hour:
                        in_range = start_hour <= entry_hour <= end_hour
                    else:  # Overnight range
                        in_range = entry_hour >= start_hour or entry_hour <= end_hour
                    
                    if not in_range:
                        continue
                    
                    # Check if file exists
                    filepath = entry.get('filepath')
                    if not filepath or not os.path.exists(filepath):
                        continue
                    
                    # Match query against metadata
                    confidence = self._calculate_match_confidence(entry, color, item, is_human=True)
                    
                    if confidence > 0.5 or (not color and not item):
                        matching_images.append((
                            filepath, 
                            entry.get('filename', ''), 
                            confidence
                        ))
            
            # Search in vehicle photos
            metadata = self.metadata_manager.load_metadata(
                self.metadata_manager.car_metadata_file
            )
            
            for entry in metadata.get('images', []):
                # Check time range
                entry_hour = entry.get('hour', -1)
                if start_hour <= end_hour:
                    in_range = start_hour <= entry_hour <= end_hour
                else:  # Overnight range
                    in_range = entry_hour >= start_hour or entry_hour <= end_hour
                
                if not in_range:
                    continue
                
                # Check if file exists
                filepath = entry.get('filepath')
                if not filepath or not os.path.exists(filepath):
                    continue
                
                # Check vehicle type match
                object_type = entry.get('object_type', '')
                query_matches_type = False
                
                if item and item in vehicle_types:
                    if item == object_type:
                        query_matches_type = True
                elif is_vehicle_search:
                    for vehicle in vehicle_types:
                        if vehicle in query_lower and vehicle == object_type:
                            query_matches_type = True
                            break
                else:
                    query_matches_type = True
                
                if not query_matches_type:
                    continue
                
                # Match query against metadata
                confidence = self._calculate_match_confidence(
                    entry, color, item, is_human=False, object_type=object_type
                )
                
                if confidence > 0.5 or (not color and query_matches_type):
                    matching_images.append((
                        filepath,
                        entry.get('filename', ''),
                        confidence
                    ))
            
            # Sort by confidence (highest first)
            matching_images.sort(key=lambda x: x[2], reverse=True)
            
            return matching_images
            
        except Exception as e:
            logging.error(f"Error in search: {e}")
            return []
    
    def _calculate_match_confidence(self, entry: Dict[str, Any], color: str, item: str,
                                   is_human: bool = True, object_type: str = None) -> float:
        """
        Calculate match confidence for an entry
        
        Args:
            entry: Metadata entry
            color: Color from query
            item: Item from query
            is_human: True if human, False if vehicle
            object_type: Object type (for vehicles)
        
        Returns:
            float: Confidence score (0.0-1.0)
        """
        match_score = 0.0
        confidence = 0.0
        
        if is_human:
            # Check color match in multiple places
            entry_colors = entry.get('colors', [])
            upper_clothing = entry.get('upper_clothing', {})
            lower_clothing = entry.get('lower_clothing', {})
            upper_color = upper_clothing.get('color', '').lower()
            lower_color = lower_clothing.get('color', '').lower()
            
            # Combine all color sources
            all_colors = [c.lower() for c in entry_colors]
            if upper_color:
                all_colors.append(upper_color)
            if lower_color:
                all_colors.append(lower_color)
            
            if color:
                if color in all_colors:
                    match_score += 0.7
                    confidence += 0.7
                elif not all_colors:
                    return 0.0  # Query requires color but image has none
            
            # Check item match
            entry_items = entry.get('detected_items', [])
            if item:
                if item in entry_items:
                    match_score += 0.3
                    confidence += 0.3
                elif item in ['pants', 'shirt', 'tshirt', 'jacket']:
                    if (item in ['pants'] and lower_clothing.get('type') == 'pants') or \
                       (item in ['shirt', 'tshirt', 'jacket'] and 
                        upper_clothing.get('type') in ['shirt', 'tshirt', 'jacket']):
                        match_score += 0.3
                        confidence += 0.3
                elif color and color in all_colors:
                    match_score += 0.2
                    confidence += 0.2
        else:
            # Vehicle matching
            entry_colors = entry.get('colors', [])
            if color:
                if color in entry_colors:
                    match_score += 0.7
                    confidence += 0.7
                elif not entry_colors:
                    return 0.0
            
            # Vehicle type match
            if item and item == object_type:
                match_score += 0.3
                confidence += 0.3
            elif object_type:
                confidence += 0.3
        
        return confidence
