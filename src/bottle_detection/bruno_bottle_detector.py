#!/usr/bin/env python3
"""
Bruno (MasterPi) - Enhanced Bottle Detection with Movement Control
Detects plastic bottles, estimates distance, and approaches them stopping at 1 foot
"""

import cv2
import numpy as np
import time
import threading
import logging
import json
from typing import Optional, Dict, List
from common.ros_robot_controller_sdk import Board
from masterpi_sdk.kinematics_sdk.kinematics.arm_move_ik import *

# Import our custom modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from bottle_detection.bottle_detector import BottleDetector
from bottle_detection.distance_estimator import DistanceEstimator
from robot_control.head_controller import HeadController
from robot_control.movement_controller import MovementController

class BrunoBottleDetector:
    def __init__(self, config_file: Optional[str] = None):
        self.setup_logging()
        self.load_config(config_file)
        
        # Initialize components
        self.bottle_detector = BottleDetector(self.config.get('detection', {}))
        self.distance_estimator = DistanceEstimator(self.config.get('distance_estimation', {}))
        self.head_controller = HeadController(self.config.get('head_control', {}))
        self.movement_controller = MovementController(self.config.get('movement_control', {}))
        
        # Initialize hardware
        self.board = Board()
        
        # Initialize arm if available and enabled
        arm_enabled = self.config.get('arm_control', {}).get('enabled', True)
        if arm_enabled:
            try:
                self.arm = ArmIK()
            except Exception as e:
                self.logger.warning(f"Could not initialize arm: {e}")
                self.arm = None
        else:
            self.arm = None
        
        # Camera setup
        self.setup_camera()
        
        # State variables
        self.running = False
        self.last_detection_time = 0
        self.detection_count = 0
        self.bottles_found_total = 0
        self.current_target_bottle = None
        self.last_bottles = []
        self.approach_mode = False
        
        # Movement state
        self.last_movement_command = None
        self.bottles_reached = 0
        
        # Initialize arm to safe position
        if self.arm:
            self.move_to_home()
    
    def setup_logging(self):
        """Setup logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_config(self, config_file: Optional[str]):
        """Load configuration from file or use defaults"""
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self._default_config()
    
    def _default_config(self) -> Dict:
        """Default configuration for Bruno"""
        return {
            'camera': {
                'device_id': "http://127.0.0.1:8080?action=stream",
                'fallback_device_id': 0,
                'width': 640,
                'height': 480,
                'fps': 30,
                'flip_horizontal': True
            },
            'detection': {
                'confidence_threshold': 0.6,
                'detection_cooldown': 1.0,  # seconds between detections
                'auto_approach': True  # Enable automatic approach
            },
            'distance_estimation': {
                'focal_length': 500,
                'real_bottle_height': 20,
                'stop_distance': 30,  # 1 foot in cm
                'approach_distance': 100
            },
            'head_control': {
                'enabled': True,
                'response_to_detection': True,
                'look_at_bottles': True
            },
            'movement_control': {
                'enabled': True,
                'max_speed': 40,
                'approach_enabled': True
            },
            'arm_control': {
                'enabled': True,
                'pickup_height': 5,
                'home_position': [15, 0, 20],
                'drop_position': [20, -15, 10]
            },
            'behavior': {
                'celebration_on_detection': True,
                'approach_bottles_automatically': True,
                'stop_on_reach': True
            }
        }
    
    def setup_camera(self):
        """Initialize camera with configuration and fallbacks"""
        camera_config = self.config['camera']
        
        # Try multiple camera initialization methods - MasterPi specific order
        primary_device = camera_config['device_id']
        fallback_device = camera_config.get('fallback_device_id', 0)
        
        camera_methods = [
            # Primary configured camera (could be URL or device ID)
            lambda: cv2.VideoCapture(primary_device),
            # MasterPi default camera streams (from Hiwonder docs)
            lambda: cv2.VideoCapture('http://127.0.0.1:8080?action=stream'),
            lambda: cv2.VideoCapture('http://localhost:8080?action=stream'),
            lambda: cv2.VideoCapture('http://127.0.0.1:8080/stream'),
            # Standard USB camera methods
            lambda: cv2.VideoCapture(fallback_device),
            lambda: cv2.VideoCapture(0),
            lambda: cv2.VideoCapture(1),
            # Try with V4L2 backend
            lambda: cv2.VideoCapture(fallback_device, cv2.CAP_V4L2) if isinstance(fallback_device, int) else cv2.VideoCapture(0, cv2.CAP_V4L2),
            # Try Raspberry Pi camera
            lambda: cv2.VideoCapture(-1),
        ]
        
        self.camera = None
        
        for i, method in enumerate(camera_methods):
            try:
                self.logger.info(f"Trying camera initialization method {i+1}...")
                test_camera = method()
                
                if test_camera.isOpened():
                    # Test if we can actually read a frame
                    ret, frame = test_camera.read()
                    if ret and frame is not None:
                        self.camera = test_camera
                        self.logger.info(f"✓ Camera initialized successfully with method {i+1}")
                        break
                    else:
                        self.logger.warning(f"Camera opens but cannot read frame (method {i+1})")
                        test_camera.release()
                else:
                    self.logger.warning(f"Cannot open camera with method {i+1}")
                    test_camera.release()
                    
            except Exception as e:
                self.logger.warning(f"Camera method {i+1} failed: {e}")
        
        if self.camera is None:
            self.logger.error("All camera initialization methods failed!")
            self.logger.error("Run: python src/calibration/camera_test.py to diagnose")
            raise Exception("Cannot open camera")
        
        # Set camera properties
        try:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config['width'])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config['height'])
            self.camera.set(cv2.CAP_PROP_FPS, camera_config['fps'])
            
            # Verify settings
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"Camera resolution: {actual_width}x{actual_height} @ {actual_fps}fps")
            
        except Exception as e:
            self.logger.warning(f"Could not set camera properties: {e}")
        
        self.logger.info("Camera initialized successfully")
    
    def move_to_home(self):
        """Move arm to home position"""
        if not self.arm:
            return
        
        try:
            home_pos = self.config['arm_control']['home_position']
            self.arm.setPitchRangeMoving(tuple(home_pos), -90, -95, -65, 1000)
            time.sleep(1.5)
            self.logger.info("Arm moved to home position")
        except Exception as e:
            self.logger.error(f"Error moving arm to home: {e}")
    
    def handle_bottle_detection(self, bottles: List[Dict], frame: np.ndarray):
        """Handle detected bottles with movement control"""
        current_time = time.time()
        
        # Store bottles for external access
        self.last_bottles = bottles
        
        # Check detection cooldown
        if current_time - self.last_detection_time < self.config['detection']['detection_cooldown']:
            return
        
        if not bottles:
            # No bottles detected - stop movement
            if self.approach_mode:
                self.movement_controller.stop()
                self.approach_mode = False
                self.logger.info("No bottles detected - stopping approach")
            return
        
        # Bottles detected!
        self.detection_count += 1
        self.bottles_found_total += len(bottles)
        self.last_detection_time = current_time
        
        best_bottle = self.bottle_detector.get_best_bottle(bottles)
        self.current_target_bottle = best_bottle
        
        self.logger.info(f"Detected {len(bottles)} bottles, best confidence: {best_bottle['confidence']:.2f}")
        
        # Head movement response
        if self.config['head_control']['response_to_detection']:
            if self.config['head_control']['look_at_bottles']:
                # Look at the best bottle
                center = best_bottle['center']
                self.head_controller.look_at_position(center[0], center[1])
                time.sleep(0.3)
            
            # Respond based on confidence
            self.head_controller.bottle_detected_response(len(bottles), best_bottle['confidence'])
        
        # Movement control for approaching bottle
        if (self.config['behavior']['approach_bottles_automatically'] and 
            self.config['movement_control']['enabled']):
            
            frame_height, frame_width = frame.shape[:2]
            movement_command = self.distance_estimator.get_movement_command(
                best_bottle, frame_width, frame_height
            )
            
            self.execute_movement_command(movement_command)
    
    def execute_movement_command(self, command: Dict):
        """Execute movement command with logging and state tracking"""
        if not command:
            return
        
        action = command.get('action', 'STOP')
        distance_cm = command.get('distance_cm', 0)
        
        # Store command for external access
        self.last_movement_command = command
        
        if action == 'STOP':
            if self.approach_mode:
                self.bottles_reached += 1
                self.approach_mode = False
                self.logger.info(f"🎯 BOTTLE REACHED! Distance: {distance_cm:.1f}cm - Total reached: {self.bottles_reached}")
                
                # Celebration nod
                if self.config['behavior']['celebration_on_detection']:
                    self.head_controller.nod_yes(3)
            
            self.movement_controller.execute_movement_command(command)
            
        elif action in ['APPROACH_SLOW', 'APPROACH_NORMAL']:
            if not self.approach_mode:
                self.approach_mode = True
                self.logger.info(f"🚀 Starting bottle approach - Distance: {distance_cm:.1f}cm")
            
            self.movement_controller.execute_movement_command(command)
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame for bottle detection and movement"""
        # Flip frame if configured
        if self.config['camera']['flip_horizontal']:
            frame = cv2.flip(frame, 1)
        
        # Detect bottles
        bottles, annotated_frame = self.bottle_detector.detect_bottles(frame)
        
        # Add distance information and movement commands
        if bottles and self.current_target_bottle:
            frame_height, frame_width = frame.shape[:2]
            movement_command = self.distance_estimator.get_movement_command(
                self.current_target_bottle, frame_width, frame_height
            )
            
            # Draw distance and movement info
            annotated_frame = self.distance_estimator.draw_distance_info(
                annotated_frame, self.current_target_bottle, movement_command
            )
        
        # Handle detections
        self.handle_bottle_detection(bottles, frame)
        
        # Add status information
        self.add_status_overlay(annotated_frame, bottles)
        
        # Check for movement timeout (safety)
        self.movement_controller.check_movement_timeout()
        
        return annotated_frame
    
    def add_status_overlay(self, frame: np.ndarray, bottles: List[Dict]):
        """Add status information overlay to the frame"""
        height, width = frame.shape[:2]
        
        # Background for status text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, height - 150), (500, height - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Status text
        status_lines = [
            f"Bottles detected: {len(bottles)}",
            f"Total detections: {self.detection_count}",
            f"Bottles reached: {self.bottles_reached}",
            f"Approach mode: {'ON' if self.approach_mode else 'OFF'}",
            f"Movement: {'ACTIVE' if self.movement_controller.is_moving else 'STOPPED'}"
        ]
        
        # Add distance info if available
        if self.last_movement_command:
            cmd = self.last_movement_command
            status_lines.append(f"Distance: {cmd.get('distance_cm', 0):.1f}cm ({cmd.get('distance_zone', 'N/A')})")
        
        y_offset = height - 130
        for line in status_lines:
            cv2.putText(frame, line, (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
        
        # Instructions
        cv2.putText(frame, "SPACE: Manual mode | Q: Quit | E: Emergency Stop", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    def run(self):
        """Main execution loop"""
        self.logger.info("Starting Bruno Bottle Detection & Approach System")
        self.logger.info("Bruno will detect bottles and approach them, stopping at 1 foot distance")
        
        self.running = True
        
        try:
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    self.logger.error("Failed to capture frame")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow("Bruno - Bottle Detection & Approach", processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    self.logger.info("Quit requested by user")
                    break
                elif key == ord('e'):
                    self.emergency_stop()
                elif key == ord(' '):
                    self.toggle_approach_mode()
                elif key == ord('r'):
                    self.reset_stats()
                elif key == ord('c'):
                    self.head_controller.calibrate_servo()
        
        except KeyboardInterrupt:
            self.logger.info("Program interrupted by user")
        
        finally:
            self.cleanup()
    
    def emergency_stop(self):
        """Emergency stop all movement"""
        self.logger.warning("🚨 EMERGENCY STOP ACTIVATED")
        self.movement_controller.emergency_stop()
        self.approach_mode = False
    
    def toggle_approach_mode(self):
        """Toggle automatic approach mode"""
        auto_approach = not self.config['behavior']['approach_bottles_automatically']
        self.config['behavior']['approach_bottles_automatically'] = auto_approach
        
        mode = "ENABLED" if auto_approach else "DISABLED"
        self.logger.info(f"Automatic approach mode: {mode}")
        
        if not auto_approach:
            self.movement_controller.stop()
            self.approach_mode = False
    
    def reset_stats(self):
        """Reset detection and movement statistics"""
        self.detection_count = 0
        self.bottles_found_total = 0
        self.bottles_reached = 0
        self.approach_mode = False
        self.movement_controller.stop()
        self.logger.info("Statistics reset")
    
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        
        # Stop all movement
        if hasattr(self, 'movement_controller'):
            self.movement_controller.cleanup()
        
        if hasattr(self, 'camera'):
            self.camera.release()
        
        cv2.destroyAllWindows()
        
        if hasattr(self, 'head_controller'):
            self.head_controller.cleanup()
        
        if self.arm:
            self.move_to_home()
        
        self.logger.info("Bruno Bottle Detection System stopped")

def main():
    """Main function"""
    try:
        config_file = "config/bruno_config.json" if os.path.exists("config/bruno_config.json") else None
        bruno = BrunoBottleDetector(config_file)
        bruno.run()
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()