"""Search engine for finding detected objects by query"""

import logging
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

from src.core.metadata_manager import MetadataManager
from src.analysis.image_analyzer import HumanImageAnalyzer

# Query item vocabulary -> normalized garment item (the vocab stored in
# garments' detected_items). Lets "jacket"/"t-shirt" match an "upper-clothes"
# region and "cap" match a "hat".
_QUERY_ITEM_NORMALIZE = {
    'shirt': 'shirt', 'tshirt': 'shirt', 'jacket': 'shirt',
    'hat': 'hat', 'cap': 'hat',
    'pants': 'pants', 'skirt': 'skirt', 'dress': 'dress',
    'bag': 'bag', 'shoe': 'shoe', 'shoes': 'shoe',
    'scarf': 'scarf', 'sunglasses': 'sunglasses', 'belt': 'belt',
}

# Garment 'type' (ATR names stored in `garments`) -> normalized item.
_GARMENT_TYPE_TO_ITEM = {
    'upper-clothes': 'shirt', 'pants': 'pants', 'skirt': 'skirt',
    'dress': 'dress', 'hat': 'hat', 'belt': 'belt', 'shoe': 'shoe',
    'bag': 'bag', 'scarf': 'scarf', 'sunglasses': 'sunglasses',
}


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
            return self._match_human(entry, color, item)
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

    def _match_human(self, entry: Dict[str, Any], color: str, item: str) -> float:
        """
        Score a human entry against a (color, item) query using the `garments`
        field as the source of truth.

        Strongest signal is a SINGLE garment matching both the requested item
        and color (e.g. a *red hat*, not "wears red somewhere and a hat
        somewhere"). Falls back to the flat `colors` / `detected_items`
        aggregates for older records that predate `garments`.
        """
        # Per-garment (item, color) pairs from the rich field.
        pairs = []
        for g in entry.get('garments', []):
            gtype = _GARMENT_TYPE_TO_ITEM.get((g.get('type') or '').lower())
            gcolor = (g.get('color') or '').lower()
            if gtype or gcolor:
                pairs.append((gtype, gcolor))

        # Aggregates: prefer garment-derived, else fall back to stored flat lists.
        flat_colors = [c.lower() for c in entry.get('colors', [])]
        flat_items = [i.lower() for i in entry.get('detected_items', [])]
        all_colors = [c for _, c in pairs if c] or flat_colors
        all_items = [t for t, _ in pairs if t] or flat_items

        norm_item = _QUERY_ITEM_NORMALIZE.get(item) if item else None

        # Color + item query -> demand precision.
        if color and norm_item:
            if any(t == norm_item and c == color for t, c in pairs):
                return 1.0  # one garment is exactly that color AND item

            # If the requested item was actually parsed as a garment, we know its
            # real color -> a color mismatch means "wrong color" (e.g. red hat vs
            # query 'blue hat'), so reject rather than reward coincidental blue
            # elsewhere on the person.
            item_colors = [c for t, c in pairs if t == norm_item]
            if item_colors:
                return 0.3   # right item, wrong color -> below match threshold

            # Item not among parsed garments; fall back to flat aggregates
            # (covers older records without per-garment data).
            item_ok = norm_item in all_items
            color_ok = color in all_colors
            if item_ok and color_ok:
                return 0.8   # both present, but not provably the same garment
            if item_ok:
                return 0.45  # item present, color absent -> below match threshold
            if color_ok:
                return 0.5   # color present, item absent -> below match threshold
            return 0.0

        # Color-only query.
        if color:
            return 0.7 if color in all_colors else 0.0

        # Item-only query.
        if norm_item:
            return 0.6 if norm_item in all_items else 0.0

        return 0.0
