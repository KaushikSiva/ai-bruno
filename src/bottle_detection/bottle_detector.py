#!/usr/bin/env python3
"""
Enhanced Bottle Detection System for MasterPi Robot (Bruno)
Detects plastic bottles using multiple computer vision techniques
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

class BottleDetector:
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.setup_logging()
        
    def _default_config(self) -> Dict:
        """Default detection configuration"""
        return {
            'color_detection': {
                'clear_plastic': {
                    'lower_hsv': [0, 0, 150],
                    'upper_hsv': [180, 50, 255]
                },
                'blue_plastic': {
                    'lower_hsv': [100, 150, 50],
                    'upper_hsv': [130, 255, 255]
                },
                'green_plastic': {
                    'lower_hsv': [40, 100, 50],
                    'upper_hsv': [80, 255, 255]
                }
            },
            'size_filter': {
                'min_area': 1000,
                'max_area': 50000,
                'min_aspect_ratio': 1.2,
                'max_aspect_ratio': 4.5
            },
            'morphology': {
                'kernel_size': 3,
                'iterations': 2
            },
            'confidence_threshold': 0.6
        }
    
    def setup_logging(self):
        """Setup logging for the detector"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for better detection"""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        return hsv
    
    def create_color_mask(self, hsv_frame: np.ndarray) -> np.ndarray:
        """Create combined mask for different bottle colors"""
        combined_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
        
        for color_name, color_range in self.config['color_detection'].items():
            lower = np.array(color_range['lower_hsv'])
            upper = np.array(color_range['upper_hsv'])
            
            mask = cv2.inRange(hsv_frame, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        return combined_mask
    
    def apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean up the mask"""
        kernel_size = self.config['morphology']['kernel_size']
        iterations = self.config['morphology']['iterations']
        
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
        
        # Fill gaps
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        
        return mask
    
    def filter_contours(self, contours: List) -> List[Dict]:
        """Filter and analyze contours to find bottle-like shapes"""
        bottles = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if (area < self.config['size_filter']['min_area'] or 
                area > self.config['size_filter']['max_area']):
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / w if w > 0 else 0
            
            # Filter by aspect ratio (bottles are typically tall)
            if (aspect_ratio < self.config['size_filter']['min_aspect_ratio'] or 
                aspect_ratio > self.config['size_filter']['max_aspect_ratio']):
                continue
            
            # Calculate additional features
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Calculate confidence score based on features
            confidence = self.calculate_confidence(area, aspect_ratio, solidity)
            
            if confidence >= self.config['confidence_threshold']:
                bottles.append({
                    'contour': contour,
                    'center': (x + w // 2, y + h // 2),
                    'bbox': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'solidity': solidity,
                    'confidence': confidence
                })
        
        return bottles
    
    def calculate_confidence(self, area: float, aspect_ratio: float, solidity: float) -> float:
        """Calculate confidence score for bottle detection"""
        # Normalize factors (0-1 scale)
        area_score = min(area / 10000.0, 1.0)  # Larger bottles get higher score
        
        # Ideal aspect ratio for bottles is around 2.5
        aspect_score = 1.0 - abs(aspect_ratio - 2.5) / 2.5
        aspect_score = max(0, aspect_score)
        
        # Bottles should have moderate solidity
        solidity_score = solidity if 0.6 <= solidity <= 0.9 else 0.5
        
        # Weighted average
        confidence = (area_score * 0.3 + aspect_score * 0.5 + solidity_score * 0.2)
        
        return min(confidence, 1.0)
    
    def detect_bottles(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """
        Main bottle detection function
        Returns: (list of detected bottles, annotated frame)
        """
        try:
            # Preprocess frame
            hsv_frame = self.preprocess_frame(frame)
            
            # Create color mask
            mask = self.create_color_mask(hsv_frame)
            
            # Apply morphological operations
            mask = self.apply_morphology(mask)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and analyze contours
            bottles = self.filter_contours(contours)
            
            # Sort bottles by confidence
            bottles.sort(key=lambda b: b['confidence'], reverse=True)
            
            # Annotate frame
            annotated_frame = self.annotate_frame(frame.copy(), bottles, mask)
            
            self.logger.debug(f"Detected {len(bottles)} bottles")
            
            return bottles, annotated_frame
            
        except Exception as e:
            self.logger.error(f"Error in bottle detection: {e}")
            return [], frame
    
    def annotate_frame(self, frame: np.ndarray, bottles: List[Dict], mask: np.ndarray) -> np.ndarray:
        """Annotate frame with detection results"""
        # Show mask in corner
        mask_resized = cv2.resize(mask, (160, 120))
        mask_colored = cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)
        frame[10:130, 10:170] = mask_colored
        
        cv2.putText(frame, "Detection Mask", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw detections
        for i, bottle in enumerate(bottles):
            x, y, w, h = bottle['bbox']
            center = bottle['center']
            confidence = bottle['confidence']
            
            # Color based on confidence
            if confidence > 0.8:
                color = (0, 255, 0)  # Green for high confidence
            elif confidence > 0.6:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 165, 255)  # Orange for low confidence
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw center point
            cv2.circle(frame, center, 5, (255, 0, 0), -1)
            
            # Add label
            label = f"Bottle {i+1} ({confidence:.2f})"
            cv2.putText(frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add additional info
            info = f"AR: {bottle['aspect_ratio']:.1f}"
            cv2.putText(frame, info, (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame
    
    def get_best_bottle(self, bottles: List[Dict]) -> Optional[Dict]:
        """Get the best bottle candidate for pickup"""
        if not bottles:
            return None
        
        # Return the bottle with highest confidence
        return bottles[0]
    
    def update_config(self, new_config: Dict):
        """Update detection configuration"""
        self.config.update(new_config)
        self.logger.info("Detection configuration updated")