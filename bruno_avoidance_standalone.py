#!/usr/bin/env python3
"""
Standalone Bruno Obstacle Avoidance System
Works without hardware dependencies for testing
"""

import cv2
import time
import signal
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

class StandaloneObstacleAvoidance:
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.setup_logging()
        
        # Obstacle detection parameters
        self.obstacle_threshold = self.config.get('obstacle_threshold', 80)  # pixels
        self.danger_threshold = self.config.get('danger_threshold', 120)    # pixels
        self.frame_width = 640
        self.frame_height = 480
        
        # Distance smoothing
        self.distance_history = []
        self.max_history = 5
        
        # State management
        self.is_running = False
        self.current_action = "IDLE"
        self.last_obstacle_time = 0
        
        # Camera setup
        self.camera_url = self.config.get('camera_url', 'http://127.0.0.1:8080?action=stream')
        self.cap = None
        
        self.logger.info("🤖 Standalone Obstacle Avoidance System initialized")
    
    def _default_config(self) -> Dict:
        """Default obstacle avoidance configuration"""
        return {
            'obstacle_threshold': 80,      # Distance in pixels to start avoiding
            'danger_threshold': 120,       # Distance to emergency stop
            'camera_url': 'http://127.0.0.1:8080?action=stream',
            'detection_area': {            # Area of frame to check for obstacles
                'top': 0.3,               # 30% from top
                'bottom': 0.9,            # 90% from top  
                'left': 0.1,              # 10% from left
                'right': 0.9              # 90% from left
            },
            'avoidance_speed': 25,         # Speed when avoiding obstacles
            'normal_speed': 35,            # Normal forward speed
            'turn_time': 1.0,              # Time to turn when avoiding
            'backup_time': 0.8,            # Time to backup when obstacle detected
        }
    
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def initialize_camera(self) -> bool:
        """Initialize camera connection"""
        try:
            if self.camera_url:
                self.cap = cv2.VideoCapture(self.camera_url)
                if not self.cap.isOpened():
                    self.logger.warning("⚠️  Failed to open camera - using test mode")
                    self.cap = None
                    return True  # Continue in test mode
                self.logger.info("📹 Camera initialized successfully")
            return True
        except Exception as e:
            self.logger.warning(f"⚠️  Camera initialization error: {e} - using test mode")
            self.cap = None
            return True
    
    def detect_obstacles_vision(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect obstacles using computer vision
        Returns list of detected obstacles with distance and position info
        """
        obstacles = []
        
        try:
            # Define detection area
            h, w = frame.shape[:2]
            area = self.config['detection_area']
            
            y1 = int(h * area['top'])
            y2 = int(h * area['bottom'])
            x1 = int(w * area['left'])
            x2 = int(w * area['right'])
            
            # Extract region of interest
            roi = frame[y1:y2, x1:x2]
            
            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area_pixels = cv2.contourArea(contour)
                
                # Filter small noise
                if area_pixels < 800:
                    continue
                
                # Get bounding box
                x, y, w_box, h_box = cv2.boundingRect(contour)
                
                # Convert back to full frame coordinates
                full_x = x + x1
                full_y = y + y1
                
                # Estimate distance based on object size and position
                distance = self.estimate_obstacle_distance(w_box, h_box, full_y, h)
                
                # Calculate angle from center
                center_x = full_x + w_box/2
                angle = np.arctan2(center_x - w/2, w/2) * 180 / np.pi
                
                # Only consider obstacles in front (within 60 degrees)
                if abs(angle) < 60:
                    obstacle = {
                        'distance': distance,
                        'angle': angle,
                        'bbox': (full_x, full_y, w_box, h_box),
                        'area': area_pixels,
                        'confidence': min(area_pixels / 5000, 1.0),
                        'type': 'obstacle'
                    }
                    obstacles.append(obstacle)
            
        except Exception as e:
            self.logger.error(f"Obstacle detection error: {e}")
        
        return obstacles
    
    def estimate_obstacle_distance(self, width: int, height: int, y_pos: int, frame_height: int) -> float:
        """
        Estimate obstacle distance in pixels
        Larger objects closer to bottom of frame are considered closer
        """
        # Size factor (larger objects are closer)
        size_factor = (width * height) / (100 * 100)  # Normalize to 100x100 baseline
        
        # Position factor (lower in frame = closer)
        position_factor = (frame_height - y_pos) / frame_height
        
        # Combine factors (higher value = closer obstacle)
        proximity_score = size_factor * 0.6 + position_factor * 0.4
        
        # Convert to distance (higher proximity = lower distance)
        # Scale so that typical obstacles are in 50-200 pixel range
        distance = max(20, 200 - (proximity_score * 100))
        
        return distance
    
    def smooth_distance(self, current_distance: float) -> float:
        """Smooth distance measurements to reduce noise"""
        self.distance_history.append(current_distance)
        
        if len(self.distance_history) > self.max_history:
            self.distance_history.pop(0)
        
        # Simple average smoothing
        if len(self.distance_history) > 0:
            return np.mean(self.distance_history)
        
        return current_distance
    
    def analyze_obstacles(self, obstacles: List[Dict]) -> Dict:
        """
        Analyze detected obstacles and determine avoidance strategy
        """
        if not obstacles:
            return {
                'action': 'CONTINUE',
                'direction': 'FORWARD',
                'speed': self.config['normal_speed'],
                'emergency': False,
                'message': 'No obstacles detected'
            }
        
        # Find closest obstacle
        closest_obstacle = min(obstacles, key=lambda x: x['distance'])
        distance = closest_obstacle['distance']
        angle = closest_obstacle['angle']
        
        # Smooth the distance measurement
        smooth_distance = self.smooth_distance(distance)
        
        # Determine action based on distance
        if smooth_distance >= self.danger_threshold:
            # DANGER ZONE - Emergency stop
            self.last_obstacle_time = time.time()
            return {
                'action': 'EMERGENCY_STOP',
                'direction': 'STOP',
                'speed': 0,
                'emergency': True,
                'distance': smooth_distance,
                'message': f'DANGER: Obstacle at {smooth_distance:.1f}px'
            }
        
        elif smooth_distance >= self.obstacle_threshold:
            # AVOIDANCE ZONE - Navigate around obstacle
            return self.plan_avoidance_maneuver(closest_obstacle, smooth_distance)
        
        else:
            # SAFE ZONE - Continue forward
            return {
                'action': 'CONTINUE',
                'direction': 'FORWARD', 
                'speed': self.config['normal_speed'],
                'emergency': False,
                'distance': smooth_distance,
                'message': f'Safe: Obstacle at {smooth_distance:.1f}px'
            }
    
    def plan_avoidance_maneuver(self, obstacle: Dict, distance: float) -> Dict:
        """Plan obstacle avoidance maneuver"""
        angle = obstacle['angle']
        
        # Choose avoidance strategy based on obstacle position
        if abs(angle) < 15:  # Obstacle directly ahead
            # Back up then turn to the side with more space
            return {
                'action': 'BACKUP_AND_TURN',
                'direction': 'LEFT' if angle >= 0 else 'RIGHT',
                'speed': self.config['avoidance_speed'],
                'emergency': False,
                'distance': distance,
                'message': f'AVOID: Obstacle ahead, backing up and turning'
            }
        
        elif angle > 15:  # Obstacle to the right
            return {
                'action': 'TURN_LEFT',
                'direction': 'LEFT',
                'speed': self.config['avoidance_speed'],
                'emergency': False,
                'distance': distance,
                'message': f'AVOID: Obstacle to right, turning left'
            }
        
        else:  # Obstacle to the left
            return {
                'action': 'TURN_RIGHT',
                'direction': 'RIGHT',
                'speed': self.config['avoidance_speed'],
                'emergency': False,
                'distance': distance,
                'message': f'AVOID: Obstacle to left, turning right'
            }
    
    def execute_avoidance_action(self, action_plan: Dict):
        """Simulate executing the planned avoidance action"""
        action = action_plan['action']
        direction = action_plan['direction']
        speed = action_plan['speed']
        
        self.current_action = action
        
        # Log what would happen with real hardware
        if action == 'EMERGENCY_STOP':
            self.logger.warning("🚨 [SIM] EMERGENCY STOP - All motors stopped")
        elif action == 'CONTINUE':
            self.logger.info(f"🤖 [SIM] Moving forward at {speed}% speed")
        elif action == 'BACKUP_AND_TURN':
            self.logger.info(f"🤖 [SIM] Backing up then turning {direction}")
        elif action == 'TURN_LEFT':
            self.logger.info(f"🤖 [SIM] Turning LEFT at {speed}% speed")
        elif action == 'TURN_RIGHT':
            self.logger.info(f"🤖 [SIM] Turning RIGHT at {speed}% speed")
    
    def draw_obstacle_info(self, frame: np.ndarray, obstacles: List[Dict], action_plan: Dict) -> np.ndarray:
        """Draw obstacle detection and avoidance info on frame"""
        # Draw detection area
        h, w = frame.shape[:2]
        area = self.config['detection_area']
        
        y1 = int(h * area['top'])
        y2 = int(h * area['bottom'])
        x1 = int(w * area['left'])
        x2 = int(w * area['right'])
        
        # Detection area outline
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, "Detection Area", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw detected obstacles
        for i, obstacle in enumerate(obstacles):
            x, y, w_box, h_box = obstacle['bbox']
            distance = obstacle['distance']
            angle = obstacle['angle']
            
            # Color based on distance
            if distance >= self.danger_threshold:
                color = (0, 0, 255)  # Red - danger
                zone_text = "DANGER"
            elif distance >= self.obstacle_threshold:
                color = (0, 255, 255)  # Yellow - caution
                zone_text = "CAUTION"
            else:
                color = (0, 255, 0)  # Green - safe
                zone_text = "SAFE"
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 3)
            
            # Draw obstacle info
            cv2.putText(frame, f"Obs {i+1}: {distance:.1f}px", 
                       (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Angle: {angle:.1f}°", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, zone_text, 
                       (x, y + h_box + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw action info at top
        action_text = f"Action: {action_plan['action']}"
        cv2.putText(frame, action_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        direction_text = f"Direction: {action_plan['direction']}"
        cv2.putText(frame, direction_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        message_text = action_plan['message'][:50] + "..." if len(action_plan['message']) > 50 else action_plan['message']
        cv2.putText(frame, message_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw speed info
        speed_text = f"Speed: {action_plan['speed']}%"
        cv2.putText(frame, speed_text, (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw thresholds legend
        cv2.putText(frame, f"Danger: >{self.danger_threshold}px", 
                   (w-200, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, f"Caution: >{self.obstacle_threshold}px", 
                   (w-200, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(frame, "Safe: Otherwise", 
                   (w-200, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    
    def start(self):
        """Start obstacle avoidance system"""
        self.logger.info("🚀 Starting Standalone Obstacle Avoidance System")
        
        if not self.initialize_camera():
            return False
        
        self.is_running = True
        return True
    
    def stop(self):
        """Stop obstacle avoidance system"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.logger.info("🛑 Standalone Obstacle Avoidance System stopped")
    
    def run_with_camera(self):
        """Run with real camera feed"""
        if not self.start():
            return
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame")
                    time.sleep(0.1)
                    continue
                
                # Detect obstacles
                obstacles = self.detect_obstacles_vision(frame)
                
                # Analyze and plan avoidance
                action_plan = self.analyze_obstacles(obstacles)
                
                # Execute avoidance action (simulated)
                self.execute_avoidance_action(action_plan)
                
                # Draw visualization
                display_frame = self.draw_obstacle_info(frame, obstacles, action_plan)
                
                # Resize for display
                display_frame = cv2.resize(display_frame, (640, 480))
                cv2.imshow('Bruno Obstacle Avoidance - Live', display_frame)
                
                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    break
                
                time.sleep(0.1)  # Control loop frequency
                
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Runtime error: {e}")
        finally:
            self.stop()


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    global avoidance_system
    print("\n🛑 Shutting down...")
    if avoidance_system:
        avoidance_system.stop()


if __name__ == '__main__':
    # Global variable for signal handler
    avoidance_system = None
    
    try:
        # Setup signal handling
        signal.signal(signal.SIGINT, signal_handler)
        
        # Create obstacle avoidance system
        config = {
            'obstacle_threshold': 80,
            'danger_threshold': 120,
            'normal_speed': 35,
            'avoidance_speed': 25,
            'camera_url': 'http://127.0.0.1:8080?action=stream'  # Use None to test without camera
        }
        
        avoidance_system = StandaloneObstacleAvoidance(config)
        
        print("🤖 Bruno Standalone Obstacle Avoidance System")
        print("=" * 60)
        print("This system demonstrates obstacle detection and avoidance")
        print("without requiring the full hardware setup.")
        print("Controls:")
        print("  ESC - Exit")
        print("  CTRL+C - Stop")
        print("=" * 60)
        
        # Try to run with camera
        avoidance_system.run_with_camera()
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if avoidance_system:
            avoidance_system.stop()
        print("✅ Cleanup complete")