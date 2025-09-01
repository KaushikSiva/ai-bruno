#!/usr/bin/env python3
"""
Bruno Obstacle Avoidance System - Headless Version
Runs obstacle avoidance without GUI display for headless systems
"""

import cv2
import time
import signal
import numpy as np
import logging
import json
import os
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

class BrunoObstacleAvoidanceHeadless:
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
        self.frame_count = 0
        self.start_time = time.time()
        
        # Camera setup
        self.camera_url = self.config.get('camera_url', 'http://127.0.0.1:8080?action=stream')
        self.cap = None
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'obstacles_detected': 0,
            'emergency_stops': 0,
            'avoidance_maneuvers': 0
        }
        
        self.logger.info("🤖 Bruno Headless Obstacle Avoidance System initialized")
    
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
            'save_debug_images': False,    # Save debug images to disk
            'debug_image_interval': 30,    # Save debug image every N frames
            'status_report_interval': 100, # Print status every N frames
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
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_camera_config(self) -> Dict:
        """Load camera configuration from config file"""
        config_file = "config/bruno_config.json"
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    return config.get('camera', {})
            except Exception as e:
                self.logger.warning(f"Could not load camera config: {e}")
        return {}
    
    def initialize_camera(self) -> bool:
        """Initialize camera connection with support for USB cameras and streams"""
        try:
            # Load camera config
            camera_config = self.load_camera_config()
            device_id = camera_config.get('device_id', self.camera_url)
            fallback_device_id = camera_config.get('fallback_device_id', 0)
            
            # Try primary camera source
            self.logger.info(f"Trying primary camera: {device_id}")
            
            if isinstance(device_id, int):
                # USB camera with DirectShow backend for Windows
                self.cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
            else:
                # Stream URL
                self.cap = cv2.VideoCapture(device_id)
            
            if self.cap.isOpened():
                # Set camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                ret, frame = self.cap.read()
                if ret:
                    self.logger.info(f"📹 Primary camera initialized successfully: {device_id}")
                    return True
                else:
                    self.cap.release()
            
            # Try fallback camera
            self.logger.info(f"Trying fallback camera: {fallback_device_id}")
            if isinstance(fallback_device_id, int):
                self.cap = cv2.VideoCapture(fallback_device_id, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(fallback_device_id)
            
            if self.cap.isOpened():
                # Set camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                ret, frame = self.cap.read()
                if ret:
                    self.logger.info(f"📹 Fallback camera initialized successfully: {fallback_device_id}")
                    return True
                else:
                    self.cap.release()
            
            self.logger.error("❌ Failed to open any camera")
            return False
            
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
        
        # Update statistics
        self.stats['obstacles_detected'] += 1
        
        # Determine action based on distance
        # NOTE: Lower distance values mean closer obstacles
        if smooth_distance <= 50:  # Very close - emergency stop
            # IMMEDIATE DANGER - Emergency stop
            self.last_obstacle_time = time.time()
            self.stats['emergency_stops'] += 1
            return {
                'action': 'EMERGENCY_STOP',
                'direction': 'STOP',
                'speed': 0,
                'emergency': True,
                'distance': smooth_distance,
                'message': f'EMERGENCY: Very close obstacle at {smooth_distance:.1f}px'
            }
        
        elif smooth_distance <= self.obstacle_threshold:  # Close - avoid
            # AVOIDANCE ZONE - Navigate around obstacle
            self.stats['avoidance_maneuvers'] += 1
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
            
            # If we've been emergency stopping for a while, try to back away
            if time.time() - self.last_obstacle_time > 2.0:
                self.logger.info("🔄 Emergency stop timeout - attempting recovery backup")
                self.movement.move_backward(20)
                time.sleep(1.5)  # Back up for 1.5 seconds
                self.movement.stop()
                self.last_obstacle_time = time.time()  # Reset timer
            
            self.logger.warning("🚨 EMERGENCY STOP")
        
        elif action == 'CONTINUE':
            self.movement.move_forward(speed)
            
        elif action == 'BACKUP_AND_TURN':
            # First backup
            self.logger.info("⬅️ Backing up to avoid obstacle")
            self.movement.move_backward(speed)
            time.sleep(self.config['backup_time'])
            
            # Then turn
            turn_direction = 'LEFT' if direction == 'LEFT' else 'RIGHT'
            self.logger.info(f"🔄 Turning {turn_direction} to avoid obstacle")
            if direction == 'LEFT':
                self.movement.turn('LEFT', speed)
            else:
                self.movement.turn('RIGHT', speed)
            time.sleep(self.config['turn_time'])
            
            # Stop after maneuver
            self.movement.stop()
            
        elif action == 'TURN_LEFT':
            self.logger.info("🔄 Turning LEFT to avoid obstacle")
            self.movement.turn('LEFT', speed)
            time.sleep(0.8)  # Turn for a bit
            self.movement.stop()
            
        elif action == 'TURN_RIGHT':
            self.logger.info("🔄 Turning RIGHT to avoid obstacle")
            self.movement.turn('RIGHT', speed)
            time.sleep(0.8)  # Turn for a bit
            self.movement.stop()
        
        self.logger.info(f"🤖 Action: {action} - {action_plan['message']}")
    
    def save_debug_image(self, frame: np.ndarray, obstacles: List[Dict], action_plan: Dict):
        """Save debug image to disk"""
        if not self.config.get('save_debug_images', False):
            return
        
        if self.frame_count % self.config.get('debug_image_interval', 30) != 0:
            return
        
        try:
            # Create debug directory
            debug_dir = "debug_images"
            os.makedirs(debug_dir, exist_ok=True)
            
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
            
            # Save image
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{debug_dir}/debug_{timestamp}_{self.frame_count:06d}.jpg"
            cv2.imwrite(filename, frame)
            self.logger.info(f"Debug image saved: {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving debug image: {e}")
    
    def print_status_report(self):
        """Print periodic status report"""
        if self.frame_count % self.config.get('status_report_interval', 100) != 0:
            return
        
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        self.logger.info("=" * 60)
        self.logger.info("📊 BRUNO OBSTACLE AVOIDANCE STATUS REPORT")
        self.logger.info("=" * 60)
        self.logger.info(f"Runtime: {elapsed_time:.1f}s")
        self.logger.info(f"Frames processed: {self.frame_count}")
        self.logger.info(f"Average FPS: {fps:.1f}")
        self.logger.info(f"Current action: {self.current_action}")
        self.logger.info(f"Obstacles detected: {self.stats['obstacles_detected']}")
        self.logger.info(f"Emergency stops: {self.stats['emergency_stops']}")
        self.logger.info(f"Avoidance maneuvers: {self.stats['avoidance_maneuvers']}")
        self.logger.info("=" * 60)
    
    def start(self):
        """Start obstacle avoidance system"""
        self.logger.info("🚀 Starting Bruno Headless Obstacle Avoidance System")
        
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
        self.logger.info("🛑 Bruno Headless Obstacle Avoidance System stopped")
    
    def run(self):
        """Main obstacle avoidance loop - no GUI display"""
        if not self.start():
            return
        
        self.logger.info("🤖 Headless obstacle avoidance running...")
        self.logger.info("Press Ctrl+C to stop")
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame")
                    time.sleep(0.1)
                    continue
                
                self.frame_count += 1
                self.stats['frames_processed'] = self.frame_count
                
                # Detect obstacles
                obstacles = self.detect_obstacles_vision(frame)
                
                # Analyze and plan avoidance
                action_plan = self.analyze_obstacles(obstacles)
                
                # Execute avoidance action
                self.execute_avoidance_action(action_plan)
                
                # Save debug image if enabled
                self.save_debug_image(frame, obstacles, action_plan)
                
                # Print status report periodically
                self.print_status_report()
                
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
    print("\n🛑 Shutting down Bruno Headless Obstacle Avoidance...")
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
            'obstacle_threshold': 100,     # Start avoiding at 100px distance
            'danger_threshold': 50,        # Emergency stop at 50px (very close)
            'normal_speed': 30,            # Slower normal speed for safety
            'avoidance_speed': 20,         # Slower avoidance speed
            'camera_url': 'http://127.0.0.1:8080?action=stream',
            'detection_area': {            # Area of frame to check for obstacles
                'top': 0.4,               # 40% from top (focus on closer area)
                'bottom': 0.9,            # 90% from top  
                'left': 0.2,              # 20% from left (narrower focus)
                'right': 0.8              # 80% from left
            },
            'turn_time': 1.2,              # Longer turn time for better avoidance
            'backup_time': 1.0,            # Longer backup time
            'save_debug_images': True,     # Save debug images for analysis
            'debug_image_interval': 20,    # Save every 20 frames (more frequent)
            'status_report_interval': 50,  # More frequent status reports
            'movement': {                  # Movement controller config
                'max_speed': 40,
                'min_speed': 10
            },
            'navigation': {                # Navigation config
                'forward_speed': 30,
                'turn_speed': 25
            }
        }
        
        bruno_avoidance = BrunoObstacleAvoidanceHeadless(config)
        
        print("🤖 Bruno Headless Obstacle Avoidance System")
        print("=" * 50)
        print("Running without GUI display")
        print("Debug images saved to: debug_images/")
        print("Press CTRL+C to stop")
        print("=" * 50)
        
        bruno_avoidance.run()
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if bruno_avoidance:
            bruno_avoidance.cleanup()
        print("✅ Cleanup complete")