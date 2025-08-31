#!/usr/bin/env python3
"""
Bruno Obstacle Avoidance System
Integrates camera-based obstacle detection with Bruno's movement system
"""

import cv2
import time
import signal
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

# Optional pandas import for data smoothing
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Bruno's existing components
from src.robot_control.movement_controller import MovementController
from src.robot_control.roomba_navigator import RoombaNavigator, NavigationState

class BrunoObstacleAvoidance:
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.setup_logging()
        
        # Initialize Bruno's movement systems
        self.movement = MovementController(self.config.get('movement', {}))
        self.navigator = RoombaNavigator(self.config.get('navigation', {}))
        
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
        
        self.logger.info("🤖 Bruno Obstacle Avoidance System initialized")
    
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
            'movement': {                  # Movement controller config
                'max_speed': 50,
                'min_speed': 15
            },
            'navigation': {                # Navigation config
                'forward_speed': 35,
                'turn_speed': 30
            }
        }
    
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def initialize_camera(self) -> bool:
        """Initialize camera connection"""
        try:
            self.cap = cv2.VideoCapture(self.camera_url)
            if not self.cap.isOpened():
                self.logger.error("❌ Failed to open camera")
                return False
            self.logger.info("📹 Camera initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Camera initialization error: {e}")
            return False
    
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
        
        # Use pandas for statistical smoothing if available, otherwise simple average
        if PANDAS_AVAILABLE and len(self.distance_history) > 2:
            data = pd.DataFrame(self.distance_history)
            data_copy = data.copy()
            mean_val = data_copy.mean()[0]
            std_val = data_copy.std()[0]
            
            # Filter outliers
            if std_val > 0:
                filtered_data = data[np.abs(data - mean_val) <= std_val]
                if len(filtered_data) > 0:
                    return filtered_data.mean()[0]
        
        # Fallback to simple average
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
        """Execute the planned avoidance action"""
        action = action_plan['action']
        direction = action_plan['direction']
        speed = action_plan['speed']
        
        self.current_action = action
        
        if action == 'EMERGENCY_STOP':
            self.movement.emergency_stop()
            self.logger.warning("🚨 EMERGENCY STOP")
        
        elif action == 'CONTINUE':
            self.movement.move_forward(speed)
            
        elif action == 'BACKUP_AND_TURN':
            # First backup
            self.movement.move_backward(speed)
            time.sleep(self.config['backup_time'])
            
            # Then turn
            if direction == 'LEFT':
                self.movement.turn('LEFT', speed)
            else:
                self.movement.turn('RIGHT', speed)
            time.sleep(self.config['turn_time'])
            
        elif action == 'TURN_LEFT':
            self.movement.turn('LEFT', speed)
            
        elif action == 'TURN_RIGHT':
            self.movement.turn('RIGHT', speed)
        
        self.logger.info(f"🤖 Action: {action} - {action_plan['message']}")
    
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
        
        # Draw detected obstacles
        for obstacle in obstacles:
            x, y, w_box, h_box = obstacle['bbox']
            distance = obstacle['distance']
            
            # Color based on distance
            if distance >= self.danger_threshold:
                color = (0, 0, 255)  # Red - danger
            elif distance >= self.obstacle_threshold:
                color = (0, 255, 255)  # Yellow - caution
            else:
                color = (0, 255, 0)  # Green - safe
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
            
            # Draw distance text
            cv2.putText(frame, f"{distance:.1f}px", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw action info
        action_text = f"Action: {action_plan['action']}"
        cv2.putText(frame, action_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        message_text = action_plan['message']
        cv2.putText(frame, message_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def start(self):
        """Start obstacle avoidance system"""
        self.logger.info("🚀 Starting Bruno Obstacle Avoidance System")
        
        if not self.initialize_camera():
            return False
        
        self.is_running = True
        return True
    
    def stop(self):
        """Stop obstacle avoidance system"""
        self.is_running = False
        self.movement.stop()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.logger.info("🛑 Bruno Obstacle Avoidance System stopped")
    
    def run(self):
        """Main obstacle avoidance loop"""
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
                
                # Execute avoidance action
                self.execute_avoidance_action(action_plan)
                
                # Draw visualization
                display_frame = self.draw_obstacle_info(frame, obstacles, action_plan)
                
                # Resize for display
                display_frame = cv2.resize(display_frame, (640, 480))
                cv2.imshow('Bruno Obstacle Avoidance', display_frame)
                
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
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop()
        self.movement.cleanup()
        self.navigator.cleanup()


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    global bruno_avoidance
    print("\n🛑 Shutting down Bruno Obstacle Avoidance...")
    if bruno_avoidance:
        bruno_avoidance.stop()


if __name__ == '__main__':
    # Global variable for signal handler
    bruno_avoidance = None
    
    try:
        # Setup signal handling
        signal.signal(signal.SIGINT, signal_handler)
        
        # Create and run obstacle avoidance system
        config = {
            'obstacle_threshold': 80,
            'danger_threshold': 120,
            'normal_speed': 35,
            'avoidance_speed': 25,
            'camera_url': 'http://127.0.0.1:8080?action=stream'
        }
        
        bruno_avoidance = BrunoObstacleAvoidance(config)
        
        print("🤖 Bruno Obstacle Avoidance System")
        print("=" * 50)
        print("Controls:")
        print("  ESC - Exit")
        print("  CTRL+C - Stop")
        print("=" * 50)
        
        bruno_avoidance.run()
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if bruno_avoidance:
            bruno_avoidance.cleanup()
        print("✅ Cleanup complete")