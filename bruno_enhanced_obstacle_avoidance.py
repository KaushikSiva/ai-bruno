#!/usr/bin/env python3
"""
Bruno Enhanced Obstacle Avoidance System
Combines camera vision, ultrasonic sensors, and RGB LED feedback
Uses MasterPi's built-in ultrasonic sensor and RGB LEDs
"""

import cv2
import time
import signal
import numpy as np
import logging
import json
import os
from typing import Dict, List, Optional, Tuple
from threading import Thread, Event
import threading

# Optional pandas import for data smoothing
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Bruno's existing components
from src.robot_control.movement_controller import MovementController
from src.robot_control.roomba_navigator import RoombaNavigator, NavigationState
from common.ros_robot_controller_sdk import Board

class UltrasonicSensor:
    """Interface to MasterPi's built-in ultrasonic sensor"""
    
    def __init__(self, board: Board, trig_pin: int = 21, echo_pin: int = 20):
        """
        Initialize ultrasonic sensor
        Default pins based on common MasterPi configurations
        """
        self.board = board
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        self.logger = logging.getLogger(__name__ + '.UltrasonicSensor')
        
        # Setup GPIO pins
        try:
            # Note: Actual GPIO setup may vary based on MasterPi SDK
            # This is a conceptual implementation
            self.logger.info(f"Ultrasonic sensor initialized on pins {trig_pin}/{echo_pin}")
        except Exception as e:
            self.logger.error(f"Failed to initialize ultrasonic sensor: {e}")
    
    def get_distance(self) -> Optional[float]:
        """
        Get distance measurement from ultrasonic sensor
        Returns distance in centimeters, or None if measurement fails
        """
        try:
            # Trigger pulse (10 microseconds)
            # Note: This is conceptual - actual implementation depends on MasterPi SDK
            # You may need to use board-specific GPIO functions
            
            # Send trigger pulse
            start_time = time.time()
            
            # Wait for echo response (with timeout)
            timeout = start_time + 0.1  # 100ms timeout
            
            # Simulate ultrasonic measurement for now
            # Replace with actual Board() GPIO calls when available
            distance_cm = self._simulate_ultrasonic_reading()
            
            if distance_cm is not None and 2.0 <= distance_cm <= 400.0:  # Valid range
                return distance_cm
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Ultrasonic sensor error: {e}")
            return None
    
    def _simulate_ultrasonic_reading(self) -> Optional[float]:
        """
        Temporary simulation of ultrasonic reading
        Replace with actual Board() calls when MasterPi GPIO interface is available
        """
        # This is a placeholder - replace with actual sensor reading
        # Example of what the real implementation might look like:
        # pulse_duration = self.board.get_gpio_pulse_duration(self.echo_pin)
        # distance = (pulse_duration * 34300) / 2  # Speed of sound calculation
        
        # For now, return a simulated reading that indicates no immediate obstacles
        return 150.0  # Simulate 150cm (safe distance)

class RGBLEDController:
    """Interface to MasterPi's RGB LEDs"""
    
    def __init__(self, board: Board):
        self.board = board
        self.logger = logging.getLogger(__name__ + '.RGBLEDController')
        self.current_color = (0, 0, 0)
        
        # LED control may use PWM pins or specific LED control functions
        # This depends on MasterPi's actual hardware implementation
        try:
            self.logger.info("RGB LED controller initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize RGB LEDs: {e}")
    
    def set_color(self, red: int, green: int, blue: int):
        """
        Set RGB LED color
        Values: 0-255 for each color component
        """
        try:
            # Note: Replace with actual MasterPi LED control function
            # This might be something like:
            # self.board.set_rgb_led(red, green, blue)
            # or PWM control for individual color channels
            
            self.current_color = (red, green, blue)
            self.logger.debug(f"LED color set to RGB({red}, {green}, {blue})")
            
        except Exception as e:
            self.logger.error(f"Error setting LED color: {e}")
    
    def set_status_color(self, status: str):
        """Set LED color based on obstacle avoidance status"""
        colors = {
            'safe': (0, 255, 0),       # Green - safe to proceed
            'caution': (255, 255, 0),  # Yellow - obstacle detected
            'danger': (255, 0, 0),     # Red - emergency stop
            'avoiding': (0, 0, 255),   # Blue - actively avoiding
            'idle': (128, 128, 128),   # Gray - idle/stopped
            'off': (0, 0, 0)           # Off
        }
        
        if status in colors:
            r, g, b = colors[status]
            self.set_color(r, g, b)
        else:
            self.logger.warning(f"Unknown status color: {status}")
    
    def pulse_color(self, color: Tuple[int, int, int], duration: float = 1.0):
        """Pulse LED with specified color"""
        try:
            original_color = self.current_color
            
            # Fade in
            steps = 20
            for i in range(steps):
                intensity = i / steps
                r = int(color[0] * intensity)
                g = int(color[1] * intensity)
                b = int(color[2] * intensity)
                self.set_color(r, g, b)
                time.sleep(duration / (steps * 2))
            
            # Fade out
            for i in range(steps, 0, -1):
                intensity = i / steps
                r = int(color[0] * intensity)
                g = int(color[1] * intensity)
                b = int(color[2] * intensity)
                self.set_color(r, g, b)
                time.sleep(duration / (steps * 2))
            
            # Restore original color
            self.set_color(*original_color)
            
        except Exception as e:
            self.logger.error(f"Error pulsing LED: {e}")

class EnhancedObstacleAvoidance:
    """
    Enhanced obstacle avoidance combining camera vision, ultrasonic sensor, and RGB feedback
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.setup_logging()
        
        # Initialize MasterPi board
        self.board = Board()
        
        # Initialize Bruno's movement systems
        self.movement = MovementController(self.config.get('movement', {}))
        self.navigator = RoombaNavigator(self.config.get('navigation', {}))
        
        # Initialize sensors and LEDs
        self.ultrasonic = UltrasonicSensor(self.board, 
                                         self.config.get('ultrasonic_trig_pin', 21),
                                         self.config.get('ultrasonic_echo_pin', 20))
        self.rgb_led = RGBLEDController(self.board)
        
        # Obstacle detection parameters
        self.obstacle_threshold = self.config.get('obstacle_threshold', 80)  # pixels (vision)
        self.ultrasonic_threshold = self.config.get('ultrasonic_threshold', 30)  # cm
        self.danger_threshold = self.config.get('danger_threshold', 15)    # cm (ultrasonic)
        
        # Data fusion and smoothing
        self.vision_history = []
        self.ultrasonic_history = []
        self.max_history = 5
        
        # State management
        self.is_running = False
        self.current_action = "IDLE"
        self.last_obstacle_time = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Camera setup
        self.cap = None
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'vision_obstacles': 0,
            'ultrasonic_obstacles': 0,
            'emergency_stops': 0,
            'avoidance_maneuvers': 0
        }
        
        # Background sensor reading
        self.sensor_thread = None
        self.stop_sensors = Event()
        self.latest_ultrasonic = None
        self.sensor_lock = threading.Lock()
        
        self.rgb_led.set_status_color('idle')
        self.logger.info("🤖 Enhanced Obstacle Avoidance System initialized")
    
    def _default_config(self) -> Dict:
        """Default enhanced obstacle avoidance configuration"""
        return {
            'obstacle_threshold': 80,      # Vision-based threshold (pixels)
            'ultrasonic_threshold': 30,    # Ultrasonic threshold (cm)
            'danger_threshold': 15,        # Emergency stop threshold (cm)
            'camera_url': 'http://127.0.0.1:8080?action=stream',
            'detection_area': {
                'top': 0.4,               # 40% from top
                'bottom': 0.9,            # 90% from top  
                'left': 0.2,              # 20% from left
                'right': 0.8              # 80% from left
            },
            'ultrasonic_trig_pin': 21,     # GPIO pin for ultrasonic trigger
            'ultrasonic_echo_pin': 20,     # GPIO pin for ultrasonic echo
            'avoidance_speed': 20,         # Speed when avoiding
            'normal_speed': 30,            # Normal forward speed
            'turn_time': 1.2,              # Time to turn when avoiding
            'backup_time': 1.0,            # Time to backup
            'sensor_fusion_weight': 0.7,   # Weight for ultrasonic vs vision (0.0-1.0)
            'sensor_reading_interval': 0.1, # Ultrasonic reading interval
            'save_debug_images': True,
            'debug_image_interval': 30,
            'status_report_interval': 50,
            'movement': {
                'max_speed': 40,
                'min_speed': 10
            },
            'navigation': {
                'forward_speed': 30,
                'turn_speed': 25
            }
        }
    
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def start_sensor_thread(self):
        """Start background thread for continuous ultrasonic readings"""
        if self.sensor_thread is None or not self.sensor_thread.is_alive():
            self.stop_sensors.clear()
            self.sensor_thread = Thread(target=self._sensor_reading_loop, daemon=True)
            self.sensor_thread.start()
            self.logger.info("Sensor reading thread started")
    
    def _sensor_reading_loop(self):
        """Background loop for reading ultrasonic sensor"""
        while not self.stop_sensors.is_set():
            try:
                distance = self.ultrasonic.get_distance()
                
                with self.sensor_lock:
                    self.latest_ultrasonic = distance
                
                time.sleep(self.config.get('sensor_reading_interval', 0.1))
                
            except Exception as e:
                self.logger.error(f"Sensor reading error: {e}")
                time.sleep(0.5)
    
    def get_ultrasonic_distance(self) -> Optional[float]:
        """Get latest ultrasonic reading"""
        with self.sensor_lock:
            return self.latest_ultrasonic
    
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
        """Initialize camera connection"""
        try:
            camera_config = self.load_camera_config()
            device_id = camera_config.get('device_id', self.config['camera_url'])
            fallback_device_id = camera_config.get('fallback_device_id', 0)
            
            self.logger.info(f"Trying primary camera: {device_id}")
            
            if isinstance(device_id, int):
                self.cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(device_id)
            
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                ret, frame = self.cap.read()
                if ret:
                    self.logger.info(f"📹 Primary camera initialized: {device_id}")
                    return True
                else:
                    self.cap.release()
            
            # Try fallback
            self.logger.info(f"Trying fallback camera: {fallback_device_id}")
            if isinstance(fallback_device_id, int):
                self.cap = cv2.VideoCapture(fallback_device_id, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(fallback_device_id)
            
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                ret, frame = self.cap.read()
                if ret:
                    self.logger.info(f"📹 Fallback camera initialized: {fallback_device_id}")
                    return True
                else:
                    self.cap.release()
            
            self.logger.error("❌ Failed to open any camera")
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Camera initialization error: {e}")
            return False
    
    def detect_obstacles_vision(self, frame: np.ndarray) -> List[Dict]:
        """Computer vision obstacle detection"""
        obstacles = []
        
        try:
            h, w = frame.shape[:2]
            area = self.config['detection_area']
            
            y1 = int(h * area['top'])
            y2 = int(h * area['bottom'])
            x1 = int(w * area['left'])
            x2 = int(w * area['right'])
            
            roi = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area_pixels = cv2.contourArea(contour)
                
                if area_pixels < 800:
                    continue
                
                x, y, w_box, h_box = cv2.boundingRect(contour)
                full_x = x + x1
                full_y = y + y1
                
                distance = self.estimate_obstacle_distance_vision(w_box, h_box, full_y, h)
                center_x = full_x + w_box/2
                angle = np.arctan2(center_x - w/2, w/2) * 180 / np.pi
                
                if abs(angle) < 60:
                    obstacle = {
                        'distance': distance,
                        'angle': angle,
                        'bbox': (full_x, full_y, w_box, h_box),
                        'area': area_pixels,
                        'confidence': min(area_pixels / 5000, 1.0),
                        'type': 'vision',
                        'source': 'camera'
                    }
                    obstacles.append(obstacle)
            
        except Exception as e:
            self.logger.error(f"Vision obstacle detection error: {e}")
        
        return obstacles
    
    def estimate_obstacle_distance_vision(self, width: int, height: int, y_pos: int, frame_height: int) -> float:
        """Estimate distance from vision (in pixels, converted to rough cm equivalent)"""
        size_factor = (width * height) / (100 * 100)
        position_factor = (frame_height - y_pos) / frame_height
        proximity_score = size_factor * 0.6 + position_factor * 0.4
        
        # Convert to rough distance estimate (pixels -> cm approximation)
        distance_pixels = max(20, 200 - (proximity_score * 100))
        # Rough conversion: assume 1 pixel ≈ 0.5 cm at typical distances
        distance_cm = distance_pixels * 0.5
        
        return distance_cm
    
    def fuse_sensor_data(self, vision_obstacles: List[Dict], ultrasonic_distance: Optional[float]) -> Dict:
        """
        Fuse vision and ultrasonic sensor data to determine obstacle status
        Returns combined assessment of obstacle situation
        """
        assessment = {
            'closest_distance': float('inf'),
            'obstacle_detected': False,
            'emergency_stop': False,
            'primary_source': None,
            'confidence': 0.0,
            'action_required': 'CONTINUE'
        }
        
        # Process vision data
        vision_min_distance = float('inf')
        if vision_obstacles:
            vision_min_distance = min(obs['distance'] for obs in vision_obstacles)
            self.stats['vision_obstacles'] += 1
        
        # Process ultrasonic data
        ultrasonic_distance_cm = ultrasonic_distance if ultrasonic_distance else float('inf')
        if ultrasonic_distance_cm < 200:  # Valid ultrasonic reading
            self.stats['ultrasonic_obstacles'] += 1
        
        # Sensor fusion with configurable weighting
        fusion_weight = self.config.get('sensor_fusion_weight', 0.7)
        
        if ultrasonic_distance_cm < float('inf') and vision_min_distance < float('inf'):
            # Both sensors have data - fuse them
            fused_distance = (fusion_weight * ultrasonic_distance_cm + 
                            (1 - fusion_weight) * vision_min_distance)
            assessment['closest_distance'] = fused_distance
            assessment['primary_source'] = 'fused'
            assessment['confidence'] = 0.9
        elif ultrasonic_distance_cm < float('inf'):
            # Only ultrasonic data
            assessment['closest_distance'] = ultrasonic_distance_cm
            assessment['primary_source'] = 'ultrasonic'
            assessment['confidence'] = 0.8
        elif vision_min_distance < float('inf'):
            # Only vision data
            assessment['closest_distance'] = vision_min_distance
            assessment['primary_source'] = 'vision'
            assessment['confidence'] = 0.6
        else:
            # No obstacle data
            assessment['primary_source'] = 'none'
            assessment['confidence'] = 0.3
        
        # Determine action based on fused distance
        if assessment['closest_distance'] <= self.danger_threshold:
            assessment['emergency_stop'] = True
            assessment['action_required'] = 'EMERGENCY_STOP'
            assessment['obstacle_detected'] = True
        elif assessment['closest_distance'] <= self.ultrasonic_threshold:
            assessment['obstacle_detected'] = True
            assessment['action_required'] = 'AVOID'
        else:
            assessment['action_required'] = 'CONTINUE'
        
        return assessment
    
    def execute_avoidance_action(self, assessment: Dict):
        """Execute avoidance action based on sensor fusion assessment"""
        action = assessment['action_required']
        distance = assessment['closest_distance']
        source = assessment['primary_source']
        
        self.current_action = action
        
        # Update LED status
        if action == 'EMERGENCY_STOP':
            self.rgb_led.set_status_color('danger')
            self.movement.emergency_stop()
            
            # Recovery mechanism
            if time.time() - self.last_obstacle_time > 3.0:
                self.logger.info("🔄 Emergency timeout - attempting recovery backup")
                self.rgb_led.set_status_color('avoiding')
                self.movement.move_backward(15)
                time.sleep(1.5)
                self.movement.stop()
                self.last_obstacle_time = time.time()
                
            self.stats['emergency_stops'] += 1
            self.logger.warning(f"🚨 EMERGENCY STOP - {source} sensor: {distance:.1f}cm")
            
        elif action == 'AVOID':
            self.rgb_led.set_status_color('caution')
            self.stats['avoidance_maneuvers'] += 1
            
            # Simple avoidance: backup and turn
            self.logger.info(f"🔄 AVOIDING - {source} sensor: {distance:.1f}cm")
            self.rgb_led.set_status_color('avoiding')
            
            # Backup
            self.movement.move_backward(self.config['avoidance_speed'])
            time.sleep(self.config['backup_time'])
            
            # Turn (alternate left/right for variety)
            turn_direction = 'LEFT' if (time.time() % 2) < 1 else 'RIGHT'
            self.movement.turn(turn_direction, self.config['avoidance_speed'])
            time.sleep(self.config['turn_time'])
            
            self.movement.stop()
            
        else:  # CONTINUE
            self.rgb_led.set_status_color('safe')
            self.movement.move_forward(self.config['normal_speed'])
        
        self.logger.info(f"🤖 Action: {action} - Distance: {distance:.1f}cm ({source})")
    
    def print_status_report(self):
        """Print periodic status report"""
        if self.frame_count % self.config.get('status_report_interval', 50) != 0:
            return
        
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        ultrasonic_dist = self.get_ultrasonic_distance()
        
        self.logger.info("=" * 70)
        self.logger.info("📊 BRUNO ENHANCED OBSTACLE AVOIDANCE STATUS")
        self.logger.info("=" * 70)
        self.logger.info(f"Runtime: {elapsed_time:.1f}s | FPS: {fps:.1f} | Action: {self.current_action}")
        self.logger.info(f"Ultrasonic: {ultrasonic_dist:.1f}cm" if ultrasonic_dist else "Ultrasonic: No reading")
        self.logger.info(f"Vision obstacles: {self.stats['vision_obstacles']} | Ultrasonic obstacles: {self.stats['ultrasonic_obstacles']}")
        self.logger.info(f"Emergency stops: {self.stats['emergency_stops']} | Avoidance maneuvers: {self.stats['avoidance_maneuvers']}")
        self.logger.info("=" * 70)
    
    def save_debug_image(self, frame: np.ndarray, obstacles: List[Dict], assessment: Dict):
        """Save debug image with obstacle detection visualization"""
        if not self.config.get('save_debug_images', False):
            return
        
        if self.frame_count % self.config.get('debug_image_interval', 30) != 0:
            return
        
        try:
            debug_dir = "debug_images"
            os.makedirs(debug_dir, exist_ok=True)
            
            # Draw detection area
            h, w = frame.shape[:2]
            area = self.config['detection_area']
            
            y1 = int(h * area['top'])
            y2 = int(h * area['bottom'])
            x1 = int(w * area['left'])
            x2 = int(w * area['right'])
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            
            # Draw vision obstacles
            for obstacle in obstacles:
                if obstacle['type'] == 'vision':
                    x, y, w_box, h_box = obstacle['bbox']
                    distance = obstacle['distance']
                    
                    color = (0, 0, 255) if distance <= self.danger_threshold else (0, 255, 255)
                    cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
                    cv2.putText(frame, f"V:{distance:.1f}cm", 
                               (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add status overlay
            status_color = (0, 255, 0) if assessment['action_required'] == 'CONTINUE' else (0, 0, 255)
            cv2.putText(frame, f"Action: {assessment['action_required']}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            cv2.putText(frame, f"Distance: {assessment['closest_distance']:.1f}cm ({assessment['primary_source']})", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            ultrasonic_dist = self.get_ultrasonic_distance()
            ultrasonic_text = f"Ultrasonic: {ultrasonic_dist:.1f}cm" if ultrasonic_dist else "Ultrasonic: N/A"
            cv2.putText(frame, ultrasonic_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Save image
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{debug_dir}/enhanced_debug_{timestamp}_{self.frame_count:06d}.jpg"
            cv2.imwrite(filename, frame)
            self.logger.info(f"Debug image saved: {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving debug image: {e}")
    
    def start(self):
        """Start enhanced obstacle avoidance system"""
        self.logger.info("🚀 Starting Bruno Enhanced Obstacle Avoidance System")
        
        if not self.initialize_camera():
            return False
        
        # Start background sensor reading
        self.start_sensor_thread()
        
        self.is_running = True
        self.rgb_led.set_status_color('safe')
        return True
    
    def stop(self):
        """Stop enhanced obstacle avoidance system"""
        self.is_running = False
        self.stop_sensors.set()
        
        if self.sensor_thread and self.sensor_thread.is_alive():
            self.sensor_thread.join(timeout=2.0)
        
        self.movement.stop()
        if self.cap:
            self.cap.release()
        
        self.rgb_led.set_status_color('off')
        self.logger.info("🛑 Enhanced Obstacle Avoidance System stopped")
    
    def run(self):
        """Main enhanced obstacle avoidance loop"""
        if not self.start():
            return
        
        self.logger.info("🤖 Enhanced obstacle avoidance running...")
        self.logger.info("Features: Camera vision + Ultrasonic sensor + RGB LEDs")
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
                
                # Get sensor data
                vision_obstacles = self.detect_obstacles_vision(frame)
                ultrasonic_distance = self.get_ultrasonic_distance()
                
                # Fuse sensor data
                assessment = self.fuse_sensor_data(vision_obstacles, ultrasonic_distance)
                
                # Execute action
                self.execute_avoidance_action(assessment)
                
                # Save debug image
                self.save_debug_image(frame, vision_obstacles, assessment)
                
                # Print status
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
    global bruno_enhanced
    print("\n🛑 Shutting down Bruno Enhanced Obstacle Avoidance...")
    if bruno_enhanced:
        bruno_enhanced.stop()


if __name__ == '__main__':
    bruno_enhanced = None
    
    try:
        signal.signal(signal.SIGINT, signal_handler)
        
        # Configuration with both vision and ultrasonic settings
        config = {
            'obstacle_threshold': 80,       # Vision threshold (pixels->cm)
            'ultrasonic_threshold': 30,     # Ultrasonic threshold (cm)
            'danger_threshold': 15,         # Emergency stop threshold (cm)
            'normal_speed': 25,             # Reduced for safety
            'avoidance_speed': 15,          # Slower avoidance
            'camera_url': 'http://127.0.0.1:8080?action=stream',
            'detection_area': {
                'top': 0.4,
                'bottom': 0.9,
                'left': 0.2,
                'right': 0.8
            },
            'ultrasonic_trig_pin': 21,      # Adjust based on actual MasterPi config
            'ultrasonic_echo_pin': 20,      # Adjust based on actual MasterPi config
            'sensor_fusion_weight': 0.7,    # Favor ultrasonic (more reliable)
            'turn_time': 1.2,
            'backup_time': 1.0,
            'save_debug_images': True,
            'debug_image_interval': 25,
            'status_report_interval': 40,
            'movement': {
                'max_speed': 35,
                'min_speed': 10
            },
            'navigation': {
                'forward_speed': 25,
                'turn_speed': 20
            }
        }
        
        bruno_enhanced = EnhancedObstacleAvoidance(config)
        
        print("🤖 Bruno Enhanced Obstacle Avoidance System")
        print("=" * 60)
        print("Features:")
        print("  📹 Camera vision obstacle detection")
        print("  📡 Ultrasonic distance sensor")
        print("  💡 RGB LED status feedback")
        print("  🧠 Sensor fusion for better accuracy")
        print("")
        print("LED Status Colors:")
        print("  🟢 Green: Safe to proceed")
        print("  🟡 Yellow: Obstacle detected")
        print("  🔴 Red: Emergency stop")
        print("  🔵 Blue: Actively avoiding")
        print("  ⚫ Gray: Idle/stopped")
        print("")
        print("Press CTRL+C to stop")
        print("=" * 60)
        
        bruno_enhanced.run()
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if bruno_enhanced:
            bruno_enhanced.cleanup()
        print("✅ Enhanced cleanup complete")