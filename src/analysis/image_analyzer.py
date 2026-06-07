import logging
from collections import Counter

import cv2
import numpy as np
from sklearn.cluster import KMeans


class HumanImageAnalyzer:
    def __init__(self):
        """Initialize image analyzer for human appearance detection"""
        # Color names in Turkish and English
        self.color_keywords = {
            "kırmızı": "red",
            "red": "red",
            "mavi": "blue",
            "blue": "blue",
            "yeşil": "green",
            "green": "green",
            "sarı": "yellow",
            "yellow": "yellow",
            "siyah": "black",
            "black": "black",
            "beyaz": "white",
            "white": "white",
            "gri": "gray",
            "gray": "gray",
            "grey": "gray",
            "turuncu": "orange",
            "orange": "orange",
            "mor": "purple",
            "purple": "purple",
            "pembe": "pink",
            "pink": "pink",
            "kahverengi": "brown",
            "brown": "brown",
        }

        # Item keywords in Turkish and English
        self.item_keywords = {
            "şapka": "hat",
            "hat": "hat",
            "kep": "cap",
            "cap": "cap",
            "gömlek": "shirt",
            "shirt": "shirt",
            "pantolon": "pants",
            "pants": "pants",
            "trouser": "pants",
            "ceket": "jacket",
            "jacket": "jacket",
            "çanta": "bag",
            "bag": "bag",
            "ayakkabı": "shoe",
            "shoe": "shoe",
            "shoes": "shoe",
            "tişört": "tshirt",
            "tshirt": "tshirt",
            "t-shirt": "tshirt",
            # Vehicle keywords
            "car": "car",
            "araba": "car",
            "otomobil": "car",
            "truck": "truck",
            "kamyon": "truck",
            "tır": "truck",
            "bus": "bus",
            "otobüs": "bus",
            "motorcycle": "motorcycle",
            "motor": "motorcycle",
            "motosiklet": "motorcycle",
        }

        # HSV color ranges (kept for analyze_image function)
        self.hsv_colors = {
            "red": [(0, 50, 50), (10, 255, 255), (170, 50, 50), (180, 255, 255)],
            "blue": [(100, 50, 50), (130, 255, 255)],
            "green": [(40, 50, 50), (80, 255, 255)],
            "yellow": [(20, 50, 50), (30, 255, 255)],
            "orange": [(10, 50, 50), (20, 255, 255)],
            "white": [(0, 0, 200), (180, 30, 255)],
            "black": [(0, 0, 0), (180, 255, 50)],
            "gray": [(0, 0, 50), (180, 30, 200)],
            "purple": [(130, 50, 50), (160, 255, 255)],
            "pink": [(160, 50, 50), (170, 255, 255)],
            "brown": [(10, 100, 50), (20, 255, 150)],
        }

        # Color name to RGB mapping (representative colors)
        self.color_rgb = {
            "red": (200, 0, 0),
            "blue": (0, 100, 200),
            "navy": (0, 0, 128),
            "green": (0, 150, 0),
            "yellow": (255, 220, 0),
            "orange": (255, 140, 0),
            "white": (250, 250, 250),
            "black": (20, 20, 20),
            "gray": (128, 128, 128),
            "silver": (192, 192, 192),
            "dark_gray": (64, 64, 64),
            "beige": (245, 245, 220),
            "brown": (139, 69, 19),
            "purple": (128, 0, 128),
            "pink": (255, 105, 180),  # Hot pink - more distinctive
        }

    def parse_query(self, query):
        """
        Parse user query to extract color and item information
        Returns: (color, item, region) tuple
        """
        query_lower = query.lower()
        color = None
        item = None
        region = "upper"  # Default to upper body (for hat, shirt, etc.)

        # Extract color
        for keyword, color_name in self.color_keywords.items():
            if keyword in query_lower:
                color = color_name
                break

        # Extract item
        for keyword, item_name in self.item_keywords.items():
            if keyword in query_lower:
                item = item_name
                # Determine region
                if item_name in ["hat", "cap"]:
                    region = "upper"
                elif item_name in ["shoe", "shoes"]:
                    region = "lower"
                else:
                    region = "upper"  # Default
                break

        return color, item, region

    def detect_color_in_region(self, image, color_name, region="upper"):
        """
        Detect if specified color exists in a region of the image
        region: 'upper', 'lower', or 'all'
        Returns: True if color detected significantly
        """
        if color_name not in self.hsv_colors:
            return False

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]

        # Define region
        if region == "upper":
            roi = hsv[0 : h // 2, :]  # Top half
        elif region == "lower":
            roi = hsv[h // 2 : h, :]  # Bottom half
        else:
            roi = hsv  # Full image

        color_ranges = self.hsv_colors[color_name]

        # Create mask for color
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)

        if len(color_ranges) == 2:  # Single range
            lower = np.array(color_ranges[0])
            upper = np.array(color_ranges[1])
            mask = cv2.inRange(roi, lower, upper)
        elif len(color_ranges) == 4:  # Two ranges (for red)
            lower1 = np.array(color_ranges[0])
            upper1 = np.array(color_ranges[1])
            lower2 = np.array(color_ranges[2])
            upper2 = np.array(color_ranges[3])
            mask1 = cv2.inRange(roi, lower1, upper1)
            mask2 = cv2.inRange(roi, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)

        # Check if color covers significant area (at least 5% of region)
        color_ratio = np.sum(mask > 0) / (roi.shape[0] * roi.shape[1])
        return color_ratio > 0.05

    def analyze_image(self, image_path, query):
        """
        Analyze an image to see if it matches the query
        Returns: (match, confidence) tuple
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return False, 0.0

            color, item, region = self.parse_query(query)

            # If no color or item specified, return False
            if not color and not item:
                return False, 0.0

            match_score = 0.0
            confidence = 0.0

            # Check if item is a vehicle
            is_vehicle = item in ["car", "truck", "bus", "motorcycle"] if item else False

            # Check color
            if color:
                # For vehicles, check color in entire image (region='all')
                check_region = "all" if is_vehicle else region
                color_detected = self.detect_color_in_region(image, color, check_region)
                if color_detected:
                    match_score += 0.7
                    confidence += 0.7

            # Check item (basic heuristic)
            if item:
                if is_vehicle:
                    # Vehicle type is matched (already verified in search function via filename)
                    match_score += 0.3
                    confidence += 0.3
                elif item in ["hat", "cap"]:
                    if region == "upper":
                        match_score += 0.3
                        confidence += 0.3
                elif item in ["shoe", "shoes"]:
                    if region == "lower":
                        match_score += 0.3
                        confidence += 0.3
                else:
                    # For other items, if color is detected, assume item might be present
                    match_score += 0.3
                    confidence += 0.3

            # Match if confidence > 0.5
            match = confidence > 0.5
            return match, confidence

        except Exception as e:
            logging.error(f"Error analyzing image {image_path}: {e}")
            return False, 0.0

    def rgb_to_color_name(self, rgb):
        """
        Convert RGB color to closest color name using simple Euclidean distance
        Returns: color name (string)
        """
        r, g, b = rgb

        # Simple approach: find closest color by Euclidean distance
        min_distance = float("inf")
        closest_color = "gray"

        for color_name, color_rgb in self.color_rgb.items():
            # Calculate Euclidean distance in RGB space
            distance = ((r - color_rgb[0]) ** 2 + (g - color_rgb[1]) ** 2 + (b - color_rgb[2]) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_color = color_name

        return closest_color

    def _get_most_vibrant_color(self, palette):
        """
        Get the most vibrant (saturated) color from a palette
        Avoids gray/black colors by preferring saturated colors
        Returns RGB tuple of the most colorful color
        """
        best_color = palette[0]  # Default to first
        best_score = 0

        for rgb in palette:
            r, g, b = rgb

            # Calculate saturation and brightness
            max_rgb = max(r, g, b)
            min_rgb = min(r, g, b)
            brightness = (r + g + b) / 3

            # Saturation (0.0 to 1.0)
            if max_rgb == 0:
                saturation = 0
            else:
                saturation = (max_rgb - min_rgb) / max_rgb

            # Score: prefer high saturation + reasonable brightness
            score = saturation * 100

            # Bonus for colors that are not too dark or too bright
            if 40 < brightness < 220:
                score *= 1.5  # 50% bonus for good brightness

            # Penalty for very dark or very bright (shadows/reflections)
            if brightness < 30 or brightness > 240:
                score *= 0.3

            if score > best_score:
                best_score = score
                best_color = rgb

        return best_color

    def extract_dominant_color_accurate(self, image, bbox, mask):
        """
        Most accurate color detection using K-means clustering and HSV analysis
        Args:
            image: Input frame (BGR)
            bbox: Bounding box (x, y, w, h)
            mask: Segmentation mask from YOLOv8-seg
        Returns: (color_name, rgb_tuple)
        """
        try:
            x, y, w, h = bbox
            roi = image[y : y + h, x : x + w]

            # 1. APPLY MASK (NO background pixels!)
            if mask is not None:
                mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                mask_roi = mask_resized[y : y + h, x : x + w]
                mask_binary = (mask_roi > 0.5).astype(np.uint8)

                # Get ONLY masked pixels (no background!)
                masked_pixels = roi[mask_binary > 0]

                # If not enough pixels, use fallback
                if len(masked_pixels) < 100:
                    return self._fallback_color_detection(roi)

                pixels = masked_pixels
            else:
                # No mask - use all pixels
                pixels = roi.reshape(-1, 3)

            # 2. FILTER OUT VERY DARK AND VERY BRIGHT PIXELS
            # Convert to HSV to get brightness (V channel)
            hsv_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)
            hsv_pixels = hsv_pixels.reshape(-1, 3)

            # Filter: V > 30 (not too dark) and V < 240 (not too bright)
            brightness = hsv_pixels[:, 2]
            valid_mask = (brightness > 30) & (brightness < 240)

            if np.sum(valid_mask) < 50:
                # Too few valid pixels - use all
                filtered_pixels = pixels
            else:
                filtered_pixels = pixels[valid_mask]

            # 3. K-MEANS CLUSTERING (find k=5 dominant colors)
            filtered_pixels_float = filtered_pixels.astype(float)

            # Apply K-means
            n_clusters = min(5, len(filtered_pixels_float))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(filtered_pixels_float)

            # Get cluster labels
            labels = kmeans.labels_

            # Count pixels in each cluster
            label_counts = Counter(labels)

            # Find the most common cluster (dominant color)
            dominant_label = label_counts.most_common(1)[0][0]

            # Get the center color of dominant cluster
            dominant_color_bgr = kmeans.cluster_centers_[dominant_label].astype(int)

            # 4. CONVERT TO RGB
            dominant_color_rgb = cv2.cvtColor(
                np.uint8([[dominant_color_bgr]]), cv2.COLOR_BGR2RGB
            )[0][0]

            # 5. CONVERT TO HSV for classification
            dominant_hsv = cv2.cvtColor(
                np.uint8([[dominant_color_bgr]]), cv2.COLOR_BGR2HSV
            )[0][0]

            h, s, v = dominant_hsv

            # 6. HSV-BASED COLOR CLASSIFICATION (most accurate)
            color_name = self._classify_color_from_hsv(h, s, v, tuple(dominant_color_rgb))

            return color_name, tuple(dominant_color_rgb)

        except Exception as e:
            logging.error(f"K-means color detection error: {e}")
            import traceback

            logging.error(traceback.format_exc())
            return "gray", (128, 128, 128)

    def _classify_color_from_hsv(self, h, s, v, rgb):
        """
        Classify color name from HSV values (most accurate method)
        Args:
            h: Hue (0-180 in OpenCV)
            s: Saturation (0-255)
            v: Value/Brightness (0-255)
            rgb: RGB tuple for fallback
        Returns: color name (string)
        """
        # First check neutral colors using saturation and value

        # Very dark → Black
        if v < 50:
            return "black"

        # Very bright + low saturation → White
        if v > 200 and s < 30:
            return "white"

        # Low saturation → Gray tones
        if s < 30:
            if v > 170:
                return "white"
            elif v > 120:
                return "silver"
            elif v > 70:
                return "gray"
            else:
                return "dark_gray"

        # Medium saturation + bright → Beige/Cream
        if s < 60 and v > 150:
            r, g, b = rgb
            if r > g and r > b:
                return "beige"

        # High saturation → Colorful (classify by Hue)
        # HSV Hue in OpenCV: 0-180 scale
        if h < 10 or h > 170:  # Red (0-10 and 170-180)
            return "red"
        elif h < 25:  # Orange (10-25)
            return "orange"
        elif h < 35:  # Yellow (25-35)
            return "yellow"
        elif h < 85:  # Green (35-85)
            return "green"
        elif h < 130:  # Blue (85-130)
            return "blue"
        elif h < 155:  # Purple (130-155)
            return "purple"
        else:  # Pink/Reddish (155-170)
            return "pink"

    def _fallback_color_detection(self, roi):
        """
        Fallback color detection if mask fails
        Returns: (color_name, rgb_tuple)
        """
        try:
            # Get average color
            avg_color_bgr = np.mean(roi, axis=(0, 1)).astype(int)
            avg_color_rgb = tuple(avg_color_bgr[::-1])  # BGR to RGB
            color_name = self.rgb_to_color_name(avg_color_rgb)
            return color_name, avg_color_rgb
        except Exception:
            return "gray", (128, 128, 128)

    def extract_metadata(self, image, object_type="human", bbox=None, mask=None):
        """
        Extract metadata from an image using segmentation mask
        For humans: analyzes upper and lower body separately
        For vehicles: analyzes entire image
        Args:
            image: Input image
            object_type: 'human' or 'vehicle'
            bbox: Bounding box (x, y, w, h)
            mask: Segmentation mask from YOLOv8-seg (optional but highly recommended)
        Returns: dict with appropriate metadata based on object_type
        """
        try:
            # If bbox is provided, crop image to bbox region
            if bbox is not None:
                x, y, w, h = bbox
                # Crop to bounding box
                roi = image[y : y + h, x : x + w]

                # If mask is provided, resize it to match ROI and apply it
                if mask is not None:
                    # Resize mask to match frame size
                    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                    # Crop mask to bbox
                    mask_roi = mask_resized[y : y + h, x : x + w]
                    # Convert to binary mask
                    mask_binary = (mask_roi > 0.5).astype(np.uint8) * 255

                    # Apply mask to ROI - set background to white for better color detection
                    roi_masked = roi.copy()
                    # Create 3-channel mask
                    mask_3ch = cv2.merge([mask_binary, mask_binary, mask_binary])
                    # Set background pixels to white
                    roi_masked[mask_3ch == 0] = 255
                    analysis_image = roi_masked
                else:
                    analysis_image = roi
            else:
                analysis_image = image

            # For humans, analyze upper and lower body separately
            if object_type == "human":
                if bbox is None:
                    # No bbox provided, use entire image
                    return {
                        "upper_clothing": {"type": "shirt", "color": ""},
                        "lower_clothing": {"type": "pants", "color": ""},
                    }

                x, y, w, h = bbox

                # Create separate dicts for upper and lower regions
                upper_clothing = {"type": "shirt", "color": ""}
                lower_clothing = {"type": "pants", "color": ""}

                # Analyze upper body (0-60% of height)
                try:
                    upper_bbox = (x, y, w, int(h * 0.6))

                    # Create upper mask if mask is provided
                    upper_mask = None
                    if mask is not None:
                        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                        upper_mask_roi = mask_resized[y : y + int(h * 0.6), x : x + w]
                        upper_mask = upper_mask_roi

                    # Use K-means + HSV for accurate color detection
                    color_name, rgb = self.extract_dominant_color_accurate(image, upper_bbox, upper_mask)
                    upper_clothing["color"] = color_name
                    logging.debug(f"Upper clothing: {color_name} RGB: {rgb}")
                except Exception as e:
                    logging.debug(f"Error analyzing upper region: {e}")

                # Analyze lower body (40-100% of height)
                try:
                    lower_y = y + int(h * 0.4)
                    lower_h = h - int(h * 0.4)
                    lower_bbox = (x, lower_y, w, lower_h)

                    # Create lower mask if mask is provided
                    lower_mask = None
                    if mask is not None:
                        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                        lower_mask_roi = mask_resized[lower_y : y + h, x : x + w]
                        lower_mask = lower_mask_roi

                    # Use K-means + HSV for accurate color detection
                    color_name, rgb = self.extract_dominant_color_accurate(image, lower_bbox, lower_mask)
                    lower_clothing["color"] = color_name
                    logging.debug(f"Lower clothing: {color_name} RGB: {rgb}")
                except Exception as e:
                    logging.debug(f"Error analyzing lower region: {e}")

                return {
                    "upper_clothing": upper_clothing,
                    "lower_clothing": lower_clothing,
                }

            # For vehicles/pets, analyze entire image with K-means + HSV
            else:
                colors = []
                detected_items = []

                if bbox is None:
                    # No bbox provided, use fallback
                    return {"colors": ["gray"], "detected_items": []}

                try:
                    # Use K-means + HSV for accurate color detection
                    color_name, rgb = self.extract_dominant_color_accurate(image, bbox, mask)
                    colors.append(color_name)
                    logging.debug(f"Vehicle/Pet color detected: {color_name} RGB: {rgb}")
                except Exception as e:
                    logging.debug(f"Error getting color: {e}")
                    colors.append("gray")

                return {
                    "colors": colors,
                    "detected_items": detected_items,
                }

        except Exception as e:
            logging.error(f"Error extracting metadata: {e}")
            import traceback

            logging.error(traceback.format_exc())
            # Return appropriate empty structure based on object type
            if object_type == "human":
                return {
                    "upper_clothing": {"type": "", "color": ""},
                    "lower_clothing": {"type": "", "color": ""},
                }
            else:
                return {"colors": [], "detected_items": []}

