#!/usr/bin/env python3
"""
Enhanced Bottle Detection System
More aggressive detection with multiple methods and better filtering
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

class EnhancedBottleDetector:
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.setup_logging()
        
        # Detection methods
        self.use_color_detection = True
        self.use_edge_detection = True
        self.use_contour_analysis = True
        self.use_template_matching = False  # Can be enabled with templates
        
        # Adaptive parameters
        self.frame_count = 0
        self.detection_history = []
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        
    def _default_config(self) -> Dict:
        """Enhanced detection configuration with more aggressive settings"""
        return {
            'color_detection': {
                'plastic_colors': {
                    # More inclusive ranges for better detection
                    'clear_white': {
                        'lower_hsv': [0, 0, 180],      # Higher brightness threshold
                        'upper_hsv': [180, 30, 255]    # Lower saturation for clear plastic
                    },
                    'light_blue': {
                        'lower_hsv': [90, 50, 50],
                        'upper_hsv': [130, 255, 255]
                    },
                    'light_green': {
                        'lower_hsv': [35, 40, 40],
                        'upper_hsv': [85, 255, 255]
                    },
                    'gray_plastic': {
                        'lower_hsv': [0, 0, 100],
                        'upper_hsv': [180, 40, 200]
                    }
                }
            },
            'size_filter': {
                'min_area': 500,        # Reduced from 1000 for smaller bottles
                'max_area': 80000,      # Increased for larger bottles
                'min_aspect_ratio': 1.0,  # Reduced from 1.2 for wider bottles
                'max_aspect_ratio': 5.0   # Increased from 4.5 for tall bottles
            },
            'edge_detection': {
                'canny_low': 50,
                'canny_high': 150,
                'dilate_kernel': 3,
                'dilate_iterations': 1
            },
            'morphology': {
                'kernel_size': 5,      # Increased for better noise removal
                'open_iterations': 2,
                'close_iterations': 3  # More aggressive gap filling
            },
            'confidence_weights': {
                'area': 0.3,
                'aspect_ratio': 0.25,
                'solidity': 0.15,
                'edge_density': 0.15,
                'color_match': 0.15
            },
            'detection_threshold': 0.4,  # Lowered from 0.6 for more detections
            'motion_detection': True,
            'adaptive_threshold': True
        }
    
    def setup_logging(self):
        """Setup logging for the detector"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def preprocess_frame(self, frame: np.ndarray) -> Dict:
        """Enhanced preprocessing with multiple outputs"""
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)  # Increased blur
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        
        # Motion detection (helps with moving bottles)
        motion_mask = None
        if self.config['motion_detection']:
            motion_mask = self.background_subtractor.apply(frame)
        
        return {
            'original': frame,
            'blurred': blurred,
            'hsv': hsv,
            'lab': lab,
            'gray': gray,
            'motion_mask': motion_mask
        }
    
    def create_enhanced_color_mask(self, frames: Dict) -> np.ndarray:
        """Create enhanced color mask using multiple color spaces"""
        hsv = frames['hsv']
        lab = frames['lab']
        
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        # HSV color detection
        for color_name, color_range in self.config['color_detection']['plastic_colors'].items():
            lower = np.array(color_range['lower_hsv'])
            upper = np.array(color_range['upper_hsv'])
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Add LAB color space detection for better plastic detection
        # Focus on L channel for brightness-based detection
        l_channel = lab[:, :, 0]
        bright_mask = cv2.inRange(l_channel, 140, 255)  # Bright objects
        combined_mask = cv2.bitwise_or(combined_mask, bright_mask)
        
        # Add motion-based detection if available
        if frames['motion_mask'] is not None:
            # Dilate motion mask to catch bottle edges
            motion_dilated = cv2.dilate(frames['motion_mask'], 
                                      np.ones((5, 5), np.uint8), iterations=2)
            combined_mask = cv2.bitwise_or(combined_mask, motion_dilated)
        
        return combined_mask
    
    def create_edge_mask(self, gray: np.ndarray) -> np.ndarray:
        """Create mask based on edge detection"""
        if not self.use_edge_detection:
            return np.zeros(gray.shape, dtype=np.uint8)
        
        # Canny edge detection
        edges = cv2.Canny(gray, 
                         self.config['edge_detection']['canny_low'],
                         self.config['edge_detection']['canny_high'])
        
        # Dilate edges to create regions
        kernel = np.ones((self.config['edge_detection']['dilate_kernel'],
                         self.config['edge_detection']['dilate_kernel']), np.uint8)
        
        edge_mask = cv2.dilate(edges, kernel, 
                              iterations=self.config['edge_detection']['dilate_iterations'])
        
        return edge_mask
    
    def apply_enhanced_morphology(self, mask: np.ndarray) -> np.ndarray:
        """Apply enhanced morphological operations"""
        kernel_size = self.config['morphology']['kernel_size']
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Opening to remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                               iterations=self.config['morphology']['open_iterations'])
        
        # Closing to fill gaps
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,
                               iterations=self.config['morphology']['close_iterations'])
        
        return mask
    
    def analyze_contour_features(self, contour: np.ndarray, mask: np.ndarray, 
                                edges: np.ndarray) -> Dict:
        """Enhanced contour feature analysis"""
        # Basic measurements
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / w if w > 0 else 0
        
        # Advanced measurements
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Perimeter and circularity
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Edge density within bounding box
        roi_edges = edges[y:y+h, x:x+w]
        edge_density = np.sum(roi_edges > 0) / (w * h) if w * h > 0 else 0
        
        # Color uniformity
        roi_mask = mask[y:y+h, x:x+w]
        color_density = np.sum(roi_mask > 0) / (w * h) if w * h > 0 else 0
        
        return {
            'area': area,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'circularity': circularity,
            'edge_density': edge_density,
            'color_density': color_density,
            'bbox': (x, y, w, h),
            'center': (x + w // 2, y + h // 2)
        }
    
    def calculate_enhanced_confidence(self, features: Dict) -> float:
        """Calculate confidence score using multiple features"""
        weights = self.config['confidence_weights']
        
        # Normalize and score each feature
        scores = {}
        
        # Area score (prefer medium-large objects)
        area_optimal = 5000  # Optimal bottle area
        area_score = 1.0 - abs(features['area'] - area_optimal) / area_optimal
        area_score = max(0, min(1, area_score))
        scores['area'] = area_score
        
        # Aspect ratio score (bottles are typically tall)
        aspect_optimal = 2.5  # Optimal aspect ratio
        aspect_score = 1.0 - abs(features['aspect_ratio'] - aspect_optimal) / aspect_optimal
        aspect_score = max(0, min(1, aspect_score))
        scores['aspect_ratio'] = aspect_score
        
        # Solidity score (bottles have moderate solidity)
        solidity_score = features['solidity'] if 0.6 <= features['solidity'] <= 0.9 else 0.3
        scores['solidity'] = solidity_score
        
        # Edge density score (bottles have moderate edges)
        edge_score = features['edge_density'] if 0.1 <= features['edge_density'] <= 0.4 else 0.5
        scores['edge_density'] = edge_score
        
        # Color match score
        color_score = features['color_density']
        scores['color_match'] = color_score
        
        # Calculate weighted confidence
        confidence = sum(scores[feature] * weights[feature] for feature in scores)
        
        return min(confidence, 1.0)
    
    def filter_and_rank_detections(self, bottles: List[Dict]) -> List[Dict]:
        """Filter and rank bottles by confidence"""
        # Filter by minimum confidence
        filtered_bottles = [b for b in bottles 
                           if b['confidence'] >= self.config['detection_threshold']]
        
        # Sort by confidence (highest first)
        filtered_bottles.sort(key=lambda b: b['confidence'], reverse=True)
        
        # Apply non-maximum suppression to remove overlapping detections
        filtered_bottles = self.apply_nms(filtered_bottles)
        
        return filtered_bottles
    
    def apply_nms(self, bottles: List[Dict], overlap_threshold: float = 0.3) -> List[Dict]:
        """Apply non-maximum suppression to remove overlapping detections"""
        if len(bottles) <= 1:
            return bottles
        
        # Calculate overlaps
        filtered = []
        used = [False] * len(bottles)
        
        for i, bottle_a in enumerate(bottles):
            if used[i]:
                continue
            
            filtered.append(bottle_a)
            used[i] = True
            
            # Mark overlapping bottles as used
            for j, bottle_b in enumerate(bottles[i+1:], i+1):
                if used[j]:
                    continue
                
                # Calculate intersection over union
                overlap = self.calculate_bbox_overlap(bottle_a['bbox'], bottle_b['bbox'])
                if overlap > overlap_threshold:
                    used[j] = True
        
        return filtered
    
    def calculate_bbox_overlap(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate overlap between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        intersection = x_overlap * y_overlap
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def detect_bottles(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """Enhanced bottle detection with multiple methods"""
        self.frame_count += 1
        
        try:
            # Preprocess frame
            frames = self.preprocess_frame(frame)
            
            # Create enhanced masks
            color_mask = self.create_enhanced_color_mask(frames)
            edge_mask = self.create_edge_mask(frames['gray'])
            
            # Combine masks
            combined_mask = cv2.bitwise_or(color_mask, edge_mask)
            
            # Apply morphological operations
            final_mask = self.apply_enhanced_morphology(combined_mask)
            
            # Find contours
            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze each contour
            bottles = []
            for contour in contours:
                # Skip tiny contours
                if cv2.contourArea(contour) < self.config['size_filter']['min_area']:
                    continue
                
                # Extract features
                features = self.analyze_contour_features(contour, final_mask, edge_mask)
                
                # Filter by size
                if (features['area'] < self.config['size_filter']['min_area'] or
                    features['area'] > self.config['size_filter']['max_area']):
                    continue
                
                # Filter by aspect ratio
                if (features['aspect_ratio'] < self.config['size_filter']['min_aspect_ratio'] or
                    features['aspect_ratio'] > self.config['size_filter']['max_aspect_ratio']):
                    continue
                
                # Calculate confidence
                confidence = self.calculate_enhanced_confidence(features)
                
                bottles.append({
                    'contour': contour,
                    'center': features['center'],
                    'bbox': features['bbox'],
                    'area': features['area'],
                    'aspect_ratio': features['aspect_ratio'],
                    'solidity': features['solidity'],
                    'confidence': confidence,
                    'features': features
                })
            
            # Filter and rank detections
            bottles = self.filter_and_rank_detections(bottles)
            
            # Update detection history
            self.detection_history.append(len(bottles))
            if len(self.detection_history) > 10:
                self.detection_history.pop(0)
            
            # Annotate frame
            annotated_frame = self.annotate_enhanced_frame(frame.copy(), bottles, final_mask)
            
            self.logger.debug(f"Enhanced detection: {len(bottles)} bottles found")
            
            return bottles, annotated_frame
            
        except Exception as e:
            self.logger.error(f"Error in enhanced bottle detection: {e}")
            return [], frame
    
    def annotate_enhanced_frame(self, frame: np.ndarray, bottles: List[Dict], mask: np.ndarray) -> np.ndarray:
        """Enhanced frame annotation with more information"""
        # Show mask in corner (larger)
        mask_resized = cv2.resize(mask, (200, 150))
        mask_colored = cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)
        frame[10:160, 10:210] = mask_colored
        
        cv2.putText(frame, "Enhanced Detection", (10, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw detections with enhanced information
        for i, bottle in enumerate(bottles):
            x, y, w, h = bottle['bbox']
            center = bottle['center']
            confidence = bottle['confidence']
            features = bottle.get('features', {})
            
            # Color based on confidence (enhanced scale)
            if confidence > 0.7:
                color = (0, 255, 0)      # Green for high confidence
            elif confidence > 0.5:
                color = (0, 255, 255)    # Yellow for medium confidence
            elif confidence > 0.3:
                color = (0, 165, 255)    # Orange for low confidence
            else:
                color = (0, 0, 255)      # Red for very low confidence
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw center point
            cv2.circle(frame, center, 7, (255, 0, 0), -1)
            
            # Enhanced labels with more information
            label = f"Bottle {i+1}"
            conf_label = f"Conf: {confidence:.2f}"
            size_label = f"Area: {int(features.get('area', 0))}"
            
            cv2.putText(frame, label, (x, y - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, conf_label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, size_label, (x, y + h + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add detection statistics
        avg_detections = sum(self.detection_history) / len(self.detection_history) if self.detection_history else 0
        stats_text = f"Avg: {avg_detections:.1f} bottles/frame"
        cv2.putText(frame, stats_text, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def get_best_bottle(self, bottles: List[Dict]) -> Optional[Dict]:
        """Get the best bottle candidate"""
        if not bottles:
            return None
        
        # Return highest confidence bottle (already sorted)
        return bottles[0]
    
    def update_config(self, new_config: Dict):
        """Update detection configuration"""
        self.config.update(new_config)
        self.logger.info("Enhanced detection configuration updated")