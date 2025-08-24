#!/usr/bin/env python3
"""
Distance Estimation for Bottle Detection
Estimates distance to bottles based on size and position
"""

import cv2
import numpy as np
import math
from typing import Dict, Tuple, Optional

class DistanceEstimator:
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        
        # Calibration parameters (need to be adjusted for your specific setup)
        self.focal_length = self.config.get('focal_length', 500)  # pixels
        self.real_bottle_height = self.config.get('real_bottle_height', 20)  # cm
        self.camera_height = self.config.get('camera_height', 25)  # cm from ground
        self.camera_tilt = self.config.get('camera_tilt', 15)  # degrees downward
        
        # Distance zones (in cm)
        self.stop_distance = self.config.get('stop_distance', 30)  # 1 foot = ~30cm
        self.approach_distance = self.config.get('approach_distance', 100)  # Start approaching
        self.max_detection_distance = self.config.get('max_detection_distance', 200)
        
    def _default_config(self) -> Dict:
        """Default distance estimation configuration"""
        return {
            'focal_length': 500,  # Estimated focal length in pixels
            'real_bottle_height': 20,  # Average bottle height in cm
            'camera_height': 25,  # Camera height from ground in cm
            'camera_tilt': 15,  # Camera tilt angle in degrees
            'stop_distance': 30,  # Stop distance in cm (1 foot)
            'approach_distance': 100,  # Start approaching distance in cm
            'max_detection_distance': 200  # Maximum detection distance in cm
        }
    
    def estimate_distance_by_size(self, bottle_height_pixels: int) -> float:
        """
        Estimate distance based on bottle size in pixels
        Uses similar triangles principle
        """
        if bottle_height_pixels <= 0:
            return float('inf')
        
        # Distance = (Real Height * Focal Length) / Pixel Height
        distance_cm = (self.real_bottle_height * self.focal_length) / bottle_height_pixels
        
        return max(0, distance_cm)
    
    def estimate_distance_by_ground_plane(self, bottle_center_y: int, frame_height: int) -> float:
        """
        Estimate distance based on ground plane geometry
        Assumes bottles are on the ground and camera has known height/tilt
        """
        # Convert pixel y to angle from camera center
        center_y = frame_height // 2
        pixel_offset = bottle_center_y - center_y
        
        # Convert pixels to angle (rough approximation)
        vertical_fov = 45  # degrees (typical camera FOV)
        angle_per_pixel = vertical_fov / frame_height
        angle_offset = pixel_offset * angle_per_pixel
        
        # Ground plane calculation
        camera_angle = self.camera_tilt + angle_offset
        
        if camera_angle <= 0:
            return float('inf')  # Looking above horizon
        
        # Distance = camera_height / tan(angle)
        distance_cm = self.camera_height / math.tan(math.radians(camera_angle))
        
        return max(0, distance_cm)
    
    def estimate_distance_combined(self, bottle: Dict, frame_height: int) -> Dict:
        """
        Combine multiple distance estimation methods for better accuracy
        """
        bbox = bottle['bbox']
        x, y, w, h = bbox
        center_y = y + h // 2
        
        # Method 1: Size-based estimation
        distance_size = self.estimate_distance_by_size(h)
        
        # Method 2: Ground plane estimation
        distance_ground = self.estimate_distance_by_ground_plane(center_y, frame_height)
        
        # Weighted average (prefer size-based for closer objects)
        if distance_size < 100:  # Close objects - trust size more
            weight_size = 0.7
            weight_ground = 0.3
        else:  # Far objects - trust ground plane more
            weight_size = 0.3
            weight_ground = 0.7
        
        # Calculate weighted distance
        if distance_size == float('inf'):
            final_distance = distance_ground
        elif distance_ground == float('inf'):
            final_distance = distance_size
        else:
            final_distance = (distance_size * weight_size + distance_ground * weight_ground)
        
        # Clamp to reasonable range
        final_distance = max(10, min(final_distance, self.max_detection_distance))
        
        return {
            'distance_cm': final_distance,
            'distance_size': distance_size,
            'distance_ground': distance_ground,
            'distance_zone': self.get_distance_zone(final_distance)
        }
    
    def get_distance_zone(self, distance_cm: float) -> str:
        """Get distance zone classification"""
        if distance_cm <= self.stop_distance:
            return 'STOP'
        elif distance_cm <= self.approach_distance:
            return 'CLOSE'
        else:
            return 'FAR'
    
    def get_approach_direction(self, bottle: Dict, frame_width: int) -> Dict:
        """
        Calculate approach direction and movement commands
        """
        bbox = bottle['bbox']
        x, y, w, h = bbox
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Calculate horizontal offset from center
        frame_center_x = frame_width // 2
        horizontal_offset = center_x - frame_center_x
        
        # Convert to normalized coordinates (-1 to 1)
        horizontal_offset_norm = horizontal_offset / (frame_width // 2)
        
        # Determine movement direction
        if abs(horizontal_offset_norm) < 0.1:  # Dead zone
            turn_direction = 'STRAIGHT'
            turn_speed = 0
        elif horizontal_offset_norm > 0:
            turn_direction = 'RIGHT'
            turn_speed = min(abs(horizontal_offset_norm) * 100, 50)  # Max 50% speed
        else:
            turn_direction = 'LEFT'  
            turn_speed = min(abs(horizontal_offset_norm) * 100, 50)
        
        return {
            'horizontal_offset': horizontal_offset,
            'horizontal_offset_norm': horizontal_offset_norm,
            'turn_direction': turn_direction,
            'turn_speed': turn_speed,
            'center_x': center_x,
            'center_y': center_y
        }
    
    def get_movement_command(self, bottle: Dict, frame_width: int, frame_height: int) -> Dict:
        """
        Generate complete movement command for approaching bottle
        """
        # Get distance information
        distance_info = self.estimate_distance_combined(bottle, frame_height)
        
        # Get direction information
        direction_info = self.get_approach_direction(bottle, frame_width)
        
        # Determine forward speed based on distance
        distance_cm = distance_info['distance_cm']
        distance_zone = distance_info['distance_zone']
        
        if distance_zone == 'STOP':
            forward_speed = 0
            action = 'STOP'
        elif distance_zone == 'CLOSE':
            # Slow approach
            forward_speed = 20  # 20% speed
            action = 'APPROACH_SLOW'
        else:  # FAR
            # Normal approach
            forward_speed = 40  # 40% speed
            action = 'APPROACH_NORMAL'
        
        # Combine all information
        movement_command = {
            'action': action,
            'forward_speed': forward_speed,
            'turn_direction': direction_info['turn_direction'],
            'turn_speed': direction_info['turn_speed'],
            'distance_cm': distance_cm,
            'distance_zone': distance_zone,
            'bottle_center': (direction_info['center_x'], direction_info['center_y']),
            'confidence': bottle.get('confidence', 0)
        }
        
        return movement_command
    
    def calibrate_focal_length(self, known_distance_cm: float, bottle_height_pixels: int):
        """
        Calibrate focal length using a bottle at known distance
        Place a bottle at known distance and call this function
        """
        if bottle_height_pixels > 0:
            self.focal_length = (known_distance_cm * bottle_height_pixels) / self.real_bottle_height
            print(f"Calibrated focal length: {self.focal_length:.1f} pixels")
            
            # Update config
            self.config['focal_length'] = self.focal_length
    
    def draw_distance_info(self, frame: np.ndarray, bottle: Dict, movement_cmd: Dict):
        """Draw distance and movement information on frame"""
        bbox = bottle['bbox']
        x, y, w, h = bbox
        
        distance_cm = movement_cmd['distance_cm']
        distance_zone = movement_cmd['distance_zone']
        action = movement_cmd['action']
        
        # Choose color based on distance zone
        if distance_zone == 'STOP':
            color = (0, 0, 255)  # Red
        elif distance_zone == 'CLOSE':
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 255, 0)  # Green
        
        # Draw distance text
        distance_text = f"{distance_cm:.1f}cm ({distance_zone})"
        cv2.putText(frame, distance_text, (x, y - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw action text
        action_text = f"{action}"
        cv2.putText(frame, action_text, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw movement arrows
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Forward arrow
        if movement_cmd['forward_speed'] > 0:
            cv2.arrowedLine(frame, (center_x, center_y), 
                          (center_x, center_y - 30), (255, 0, 0), 3)
        
        # Turn arrows
        if movement_cmd['turn_direction'] == 'LEFT':
            cv2.arrowedLine(frame, (center_x, center_y), 
                          (center_x - 30, center_y), (0, 255, 0), 2)
        elif movement_cmd['turn_direction'] == 'RIGHT':
            cv2.arrowedLine(frame, (center_x, center_y), 
                          (center_x + 30, center_y), (0, 255, 0), 2)
        
        return frame