"""Blacklist management for security alerts"""

import logging
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

from src.utils.file_utils import load_json, save_json
from src.config.settings import BLACKLIST_FILE
from src.config.constants import ALERT_COOLDOWN_SECONDS, MAX_ALERT_HISTORY
from src.core.notifier import TelegramNotifier


class BlacklistManager:
    """Manages security blacklist and alert system"""
    
    def __init__(self):
        """Initialize blacklist manager"""
        self.blacklist_file = Path(BLACKLIST_FILE)
        self.blacklist: List[Dict[str, Any]] = []
        self.blacklist_alerts: List[Dict[str, Any]] = []
        self.alert_cooldown = ALERT_COOLDOWN_SECONDS
        self.notifier = TelegramNotifier()
        self.load_blacklist()
    
    def load_blacklist(self):
        """Load blacklist from JSON file"""
        try:
            data = load_json(self.blacklist_file)
            self.blacklist = data.get('entries', []) if data else []
            logging.info(f"Blacklist loaded: {len(self.blacklist)} entries")
        except Exception as e:
            logging.error(f"Error loading blacklist: {e}")
            self.blacklist = []
    
    def save_blacklist(self):
        """Save blacklist to JSON file"""
        try:
            data = {'entries': self.blacklist}
            save_json(self.blacklist_file, data)
            logging.info(f"Blacklist saved: {len(self.blacklist)} entries")
        except Exception as e:
            logging.error(f"Error saving blacklist: {e}")
    
    def add_entry(self, query: str, object_type: str, description: str = None):
        """
        Add a new entry to blacklist
        
        Args:
            query: Search query/description
            object_type: Type of object ('human', 'vehicle', 'cat', 'dog', 'any')
            description: Optional description
        """
        entry = {
            'query': query,
            'object_type': object_type,
            'description': description or query,
            'added_time': datetime.now().isoformat()
        }
        self.blacklist.append(entry)
        self.save_blacklist()
    
    def remove_entry(self, index: int):
        """
        Remove entry from blacklist by index
        
        Args:
            index: Index of entry to remove
        """
        if 0 <= index < len(self.blacklist):
            self.blacklist.pop(index)
            self.save_blacklist()
    
    def check_match(self, metadata: Dict[str, Any], object_type: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if detected object matches any blacklist entry
        
        Args:
            metadata: Metadata of detected object
            object_type: Type of detected object
        
        Returns:
            tuple: (is_match: bool, matched_entry: Optional[Dict])
        """
        try:
            for entry in self.blacklist:
                # Check if object types match
                if entry['object_type'] == object_type or entry['object_type'] == 'any':
                    query = entry['query'].lower()
                    
                    # For humans: check upper and lower clothing
                    if object_type == 'human':
                        upper_color = metadata.get('upper_clothing', {}).get('color', '').lower()
                        lower_color = metadata.get('lower_clothing', {}).get('color', '').lower()
                        
                        if (upper_color and upper_color in query) or \
                           (lower_color and lower_color in query) or \
                           (upper_color in query.split() or lower_color in query.split()):
                            return True, entry
                    
                    # For vehicles: check colors and vehicle type
                    elif object_type in ['vehicle', 'car', 'truck', 'bus', 'motorcycle']:
                        colors = metadata.get('colors', [])
                        vehicle_type = metadata.get('object_type', '').lower()
                        
                        for color in colors:
                            color_lower = color.lower()
                            if (color_lower and color_lower in query) or \
                               (vehicle_type and vehicle_type in query):
                                return True, entry
                    
                    # For pets: simple type match
                    elif object_type in ['cat', 'dog']:
                        pet_type = metadata.get('object_type', '').lower()
                        if pet_type and pet_type in query:
                            return True, entry
            
            return False, None
        except Exception as e:
            logging.error(f"Error checking blacklist: {e}")
            return False, None
    
    def should_trigger_alert(self, entry: Dict[str, Any]) -> bool:
        """
        Check if alert should be triggered (cooldown check)
        
        Args:
            entry: Blacklist entry that matched
        
        Returns:
            bool: True if alert should be triggered
        """
        current_time = datetime.now()
        
        # Check cooldown to avoid alert spam
        for alert in self.blacklist_alerts:
            if alert['entry'] == entry and \
               (current_time - alert['time']).seconds < self.alert_cooldown:
                return False
        
        return True
    
    def record_alert(self, entry: Dict[str, Any], filepath: str):
        """
        Record that an alert was triggered
        
        Args:
            entry: Blacklist entry that matched
            filepath: Path to saved image
        """
        self.blacklist_alerts.append({
            'entry': entry,
            'time': datetime.now(),
            'filepath': filepath
        })

        # Keep only last N alerts
        if len(self.blacklist_alerts) > MAX_ALERT_HISTORY:
            self.blacklist_alerts = self.blacklist_alerts[-MAX_ALERT_HISTORY:]

        # Telefona anlık bildirim gönder (Telegram)
        self._send_notification(entry, filepath)

    def _send_notification(self, entry: Dict[str, Any], filepath: str):
        """Eşleşen blacklist kaydı için telefona bildirim gönderir."""
        object_type = entry.get('object_type', 'nesne')
        description = entry.get('description') or entry.get('query', '')
        timestamp = datetime.now().strftime('%d.%m.%Y %H:%M:%S')
        message = (
            "🚨 GÜVENLİK UYARISI\n"
            f"Blacklist eşleşmesi tespit edildi!\n\n"
            f"Tür: {object_type}\n"
            f"Tanım: {description}\n"
            f"Zaman: {timestamp}"
        )
        self.notifier.send_alert(message, filepath)
