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

    def _skin_mask(self, roi_bgr):
        """
        Detect skin pixels in a BGR ROI using a lighting-robust YCrCb range.
        Returns a uint8 mask (255 = skin). Used to exclude face/arms/legs so the
        dominant remaining color is the clothing, not the person's skin.
        """
        ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        return cv2.inRange(ycrcb, lower, upper)

    def extract_dominant_color_accurate(self, image, bbox, full_mask, remove_skin=True):
        """
        Accurate dominant-color detection using K-means + HSV, with optional
        skin removal (used for human clothing, disabled for vehicles/pets so a
        red car is not partially filtered as skin).

        Args:
            image: Full input frame (BGR)
            bbox: Sub-region (x, y, w, h) in full-frame coordinates (e.g. torso)
            full_mask: RAW YOLOv8-seg object mask (or None)
            remove_skin: If True, exclude skin pixels before clustering
        Returns: (color_name, rgb_tuple)
        """
        try:
            x, y, w, h = bbox
            # Clamp the sub-region to the frame
            x = max(0, int(x)); y = max(0, int(y))
            w = max(1, min(int(w), image.shape[1] - x))
            h = max(1, min(int(h), image.shape[0] - y))
            roi = image[y : y + h, x : x + w]
            if roi.size == 0:
                return "gray", (128, 128, 128)

            # 1. PERSON MASK for this sub-region (no background pixels)
            if full_mask is not None:
                m = cv2.resize(full_mask, (image.shape[1], image.shape[0]))
                person = (m[y : y + h, x : x + w] > 0.5).astype(np.uint8)
            else:
                person = np.ones(roi.shape[:2], dtype=np.uint8)

            # 2. REMOVE SKIN so clothing dominates the clustering (humans only)
            if remove_skin:
                skin = self._skin_mask(roi)
                clothing = person.copy()
                clothing[skin > 0] = 0
                pixels = roi[clothing > 0]
            else:
                pixels = roi[person > 0]

            # If skin removal left too little (bare skin / shorts / tiny crop),
            # fall back to all person pixels rather than guessing on noise.
            if len(pixels) < 80:
                pixels = roi[person > 0]
            if len(pixels) < 30:
                return self._fallback_color_detection(roi)

            # 3. Drop ONLY blown-out specular highlights (near-255). We must KEEP
            # dark pixels: black clothing's signal lives at low brightness, so the
            # old "remove very dark" filter turned black garments white (the few
            # bright folds/reflections were all that survived). Letting K-means
            # vote over all remaining pixels makes the true majority color win.
            hsv_pixels = cv2.cvtColor(
                pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV
            ).reshape(-1, 3)
            brightness = hsv_pixels[:, 2]
            valid = brightness < 250
            filtered = pixels[valid] if np.sum(valid) >= 50 else pixels

            # 4. K-MEANS — dominant remaining (clothing) color.
            # Cap clusters by the number of DISTINCT colors to avoid sklearn's
            # ConvergenceWarning (and degenerate fits) on near-uniform regions.
            data = filtered.astype(np.float32)
            n_unique = len(np.unique(data, axis=0))
            n_clusters = max(1, min(4, n_unique))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data)
            dominant_label = Counter(labels).most_common(1)[0][0]
            dominant_bgr = kmeans.cluster_centers_[dominant_label].astype(int)

            # 5. Convert + classify
            dominant_rgb = cv2.cvtColor(
                np.uint8([[dominant_bgr]]), cv2.COLOR_BGR2RGB
            )[0][0]
            hsv = cv2.cvtColor(np.uint8([[dominant_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
            rgb_tuple = tuple(int(c) for c in dominant_rgb)
            color_name = self._classify_color_from_hsv(hsv[0], hsv[1], hsv[2], rgb_tuple)

            return color_name, rgb_tuple

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
        # --- Neutral colors first (saturation/value driven) ---

        # Very dark → Black
        if v < 45:
            return "black"

        # Low saturation → grayscale ramp
        if s < 35:
            if v > 195:
                return "white"
            elif v > 140:
                return "silver"
            elif v > 80:
                return "gray"
            else:
                return "dark_gray"

        # Warm but desaturated & bright → Beige/Cream
        if s < 60 and v > 160:
            r, g, b = rgb
            if r >= g >= b:
                return "beige"

        # Brown = warm hue (red/orange) that is dark → must be checked
        # BEFORE the orange/red hue buckets, otherwise brown reads as orange.
        if h < 25 and v < 130 and s > 60:
            return "brown"

        # --- Saturated colors classified by Hue (OpenCV 0-180 scale) ---
        if h < 10 or h > 170:  # Red
            return "red"
        elif h < 25:  # Orange
            return "orange"
        elif h < 35:  # Yellow
            return "yellow"
        elif h < 85:  # Green
            return "green"
        elif h < 130:  # Blue
            return "blue"
        elif h < 150:  # Purple
            return "purple"
        else:  # Pink/Reddish
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

                # Central horizontal band: avoids arms (skin) and the background
                # that leaks in at the left/right edges of the bbox.
                band_x = x + int(w * 0.22)
                band_w = max(1, int(w * 0.56))

                # NOTE: the RAW person mask is passed straight through;
                # extract_dominant_color_accurate crops it to the sub-region itself.

                # Torso: skip the head (top ~15%) and stop at the waist (~50%)
                try:
                    torso_bbox = (band_x, y + int(h * 0.15), band_w, int(h * 0.35))
                    color_name, rgb = self.extract_dominant_color_accurate(image, torso_bbox, mask)
                    upper_clothing["color"] = color_name
                    logging.debug(f"Upper clothing: {color_name} RGB: {rgb}")
                except Exception as e:
                    logging.debug(f"Error analyzing upper region: {e}")

                # Legs: start below the waist (~55%) and skip the feet/shoes (~10%)
                try:
                    legs_bbox = (band_x, y + int(h * 0.55), band_w, int(h * 0.33))
                    color_name, rgb = self.extract_dominant_color_accurate(image, legs_bbox, mask)
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
                    # Vehicles/pets: keep all object pixels (no skin removal)
                    color_name, rgb = self.extract_dominant_color_accurate(
                        image, bbox, mask, remove_skin=False
                    )
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

    def vote_clothing_colors(self, candidates, max_frames=5):
        """
        Temporal voting: estimate clothing colors across several candidate frames
        of the SAME person and return the majority color per region. This is far
        more robust than a single frame (motion blur / lighting can flip one frame).

        Args:
            candidates: list of dicts with keys 'frame' and 'detection_tuple'
                        where detection_tuple = (x, y, w, h, track_id, mask)
            max_frames: how many candidate frames to sample
        Returns: dict {'upper': color_or_None, 'lower': color_or_None}
        """
        uppers, lowers = [], []
        for cand in candidates[:max_frames]:
            try:
                frame = cand["frame"]
                x, y, w, h, _track_id, mask = cand["detection_tuple"]
                md = self.extract_metadata(
                    frame, object_type="human", bbox=(x, y, w, h), mask=mask
                )
                u = md.get("upper_clothing", {}).get("color")
                l = md.get("lower_clothing", {}).get("color")
                if u:
                    uppers.append(u)
                if l:
                    lowers.append(l)
            except Exception as e:
                logging.debug(f"vote_clothing_colors frame skipped: {e}")
                continue

        upper = Counter(uppers).most_common(1)[0][0] if uppers else None
        lower = Counter(lowers).most_common(1)[0][0] if lowers else None
        return {"upper": upper, "lower": lower}

