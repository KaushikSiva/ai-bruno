#!/usr/bin/env python3
"""
Bruno Roomba Simple - Autonomous navigation with basic bottle detection
Fixed version using regular bottle detector
"""

import os
import sys
import json
import time
import logging
from datetime import datetime

# Set environment to prevent Qt issues
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import OpenCV after setting environment
import cv2
import numpy as np

from bottle_detection.bottle_detector import BottleDetector
from bottle_detection.distance_estimator import DistanceEstimator
from robot_control.roomba_navigator import RoombaNavigator

class BrunoRoombaSimple:
    def __init__(self, config_file: str = None):
        self.setup_logging()
        self.load_config(config_file)
        
        # Initialize components
        self.bottle_detector = BottleDetector(self.config.get('detection', {}))
        self.distance_estimator = DistanceEstimator(self.config.get('distance_estimation', {}))
        self.navigator = RoombaNavigator(self.config.get('navigation', {}))
        
        # Initialize head controller with error handling
        try:
            from robot_control.head_controller import HeadController
            self.head_controller = HeadController(self.config.get('head_control', {}))
            self.logger.info("✅ Head controller initialized")
        except Exception as e:
            self.logger.warning(f"⚠️  Head controller failed: {e}")
            self.head_controller = None
        
        # Setup camera
        self.setup_camera()
        
        # State variables
        self.running = False
        self.total_detections = 0
        self.bottles_approached = 0
        self.start_time = time.time()
        
        # Create output directory
        self.detections_dir = "detections"
        os.makedirs(self.detections_dir, exist_ok=True)
        
        self.logger.info("🤖 Bruno Roomba Simple initialized")
    
    def setup_logging(self):
        """Setup logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('bruno_roomba.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_config(self, config_file: str = None):
        """Load configuration with working defaults"""
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        else:
            # Complete working configuration
            self.config = {
                'camera': {
                    'device_id': "http://127.0.0.1:8080?action=stream",
                    'width': 640,
                    'height': 480,
                    'flip_horizontal': True
                },
                'detection': {
                    'color_detection': {
                        'clear_plastic': {
                            'lower_hsv': [0, 0, 120],      # More aggressive detection
                            'upper_hsv': [180, 60, 255]
                        },
                        'blue_plastic': {
                            'lower_hsv': [90, 50, 50],
                            'upper_hsv': [130, 255, 255]
                        },
                        'green_plastic': {
                            'lower_hsv': [35, 40, 40],
                            'upper_hsv': [85, 255, 255]
                        }
                    },
                    'size_filter': {
                        'min_area': 800,               # Reduced for smaller bottles
                        'max_area': 60000,
                        'min_aspect_ratio': 1.0,       # More permissive
                        'max_aspect_ratio': 5.0
                    },
                    'morphology': {
                        'kernel_size': 4,
                        'iterations': 2
                    },
                    'confidence_threshold': 0.4        # Lower threshold
                },
                'distance_estimation': {
                    'focal_length': 500,
                    'real_bottle_height': 20,
                    'stop_distance': 25,
                    'approach_distance': 120,
                    'max_detection_distance': 250
                },
                'navigation': {
                    'forward_speed': 30,
                    'turn_speed': 25,
                    'backup_speed': 20,
                    'forward_time_min': 2.0,
                    'forward_time_max': 8.0,
                    'turn_time_min': 1.0,
                    'turn_time_max': 3.0,
                    'backup_time': 1.5,
                    'spiral_time': 10.0,
                    'stuck_detection': True
                },
                'head_control': {
                    'enabled': True,
                    'response_to_detection': True,
                    'head_servo': {
                        'id': 2,
                        'positions': {
                            'center': 1500,
                            'left': 1200,
                            'right': 1800
                        },
                        'timings': {
                            'normal_movement': 0.5,
                            'fast_movement': 0.3
                        }
                    }
                },
                'behavior': {
                    'save_detections': True
                }
            }
    
    def setup_camera(self):
        """Initialize camera with multiple fallbacks"""
        camera_sources = [
            'http://127.0.0.1:8080?action=stream',
            'http://localhost:8080?action=stream',
            0, 1
        ]
        
        self.camera = None
        
        for source in camera_sources:
            self.logger.info(f"🔍 Trying camera: {source}")
            try:
                cap = cv2.VideoCapture(source)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.camera = cap
                        self.logger.info(f"✅ Camera connected: {source}")
                        break
                    else:
                        cap.release()
            except Exception as e:
                self.logger.warning(f"Camera {source} failed: {e}")
        
        if not self.camera:
            raise Exception("❌ No working camera found")
    
    def process_frame(self, frame):
        """Process frame for detection and navigation"""
        # Flip frame if configured
        if self.config['camera']['flip_horizontal']:
            frame = cv2.flip(frame, 1)
        
        # Detect bottles
        bottles, annotated_frame = self.bottle_detector.detect_bottles(frame)
        
        # Handle detections
        best_bottle = None
        bottle_position = None
        
        if bottles:
            self.total_detections += 1
            best_bottle = self.bottle_detector.get_best_bottle(bottles)
            bottle_position = best_bottle['center']
            
            # Get distance information
            frame_height, frame_width = frame.shape[:2]
            movement_command = self.distance_estimator.get_movement_command(
                best_bottle, frame_width, frame_height
            )
            
            # Draw distance info
            annotated_frame = self.distance_estimator.draw_distance_info(
                annotated_frame, best_bottle, movement_command
            )
            
            # Handle bottle approach
            self.handle_bottle_detection(best_bottle, movement_command)
            
            # Save detection
            if self.config['behavior']['save_detections']:
                self.save_detection_image(annotated_frame, best_bottle)
        
        # Update navigation
        should_approach = bottles and best_bottle['confidence'] > 0.5 if bottles else False
        navigation_command = self.navigator.update_navigation(
            bottle_detected=should_approach,
            bottle_position=bottle_position
        )
        
        # Handle head movement
        if bottles and self.head_controller:
            self.handle_head_movement(best_bottle)
        
        # Add status overlay
        annotated_frame = self.add_status_overlay(annotated_frame, bottles, navigation_command)
        
        return annotated_frame
    
    def handle_bottle_detection(self, bottle, movement_command):
        """Handle bottle detection"""
        distance_cm = movement_command.get('distance_cm', 0)
        action = movement_command.get('action', 'UNKNOWN')
        confidence = bottle['confidence']
        
        if action == 'STOP':
            self.bottles_approached += 1
            self.logger.info(f"🎯 BOTTLE REACHED! #{self.bottles_approached} - Distance: {distance_cm:.1f}cm, Confidence: {confidence:.2f}")
            
            # Brief celebration
            if self.head_controller:
                try:
                    self.head_controller.nod_yes(2)
                except Exception as e:
                    self.logger.warning(f"Celebration failed: {e}")
            
        elif action in ['APPROACH_SLOW', 'APPROACH_NORMAL']:
            self.logger.info(f"🚀 Approaching bottle - Distance: {distance_cm:.1f}cm, Confidence: {confidence:.2f}")
    
    def handle_head_movement(self, bottle):
        """Handle head movement"""
        if not self.head_controller:
            return
        
        try:
            # Look at bottle
            center = bottle['center']
            self.head_controller.look_at_position(center[0], center[1])
            
            # Respond based on confidence
            if bottle['confidence'] > 0.7:
                self.head_controller.execute_pattern('excited')
            elif bottle['confidence'] > 0.5:
                self.head_controller.execute_pattern('acknowledgment')
        except Exception as e:
            self.logger.warning(f"Head movement failed: {e}")
    
    def add_status_overlay(self, frame, bottles, navigation_command):
        """Add status information to frame"""
        height, width = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, height - 120), (450, height - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        # Status information
        nav_state = navigation_command.get('state', 'unknown')
        nav_action = navigation_command.get('action', 'none')
        runtime = time.time() - self.start_time
        runtime_str = f"{int(runtime//60):02d}:{int(runtime%60):02d}"
        
        status_lines = [
            f"🤖 Bruno Roomba - Runtime: {runtime_str}",
            f"🧭 Navigation: {nav_state.upper()} ({nav_action})",
            f"🍼 Bottles: {len(bottles)} current, {self.total_detections} total",
            f"🎯 Approached: {self.bottles_approached} bottles",
            f"🔧 Hardware: {'✅' if self.navigator.hardware_available else '🎮 SIM'}"
        ]
        
        # Draw status text
        y_offset = height - 100
        for line in status_lines:
            cv2.putText(frame, line, (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 16
        
        # Controls
        cv2.putText(frame, "Q=Quit, E=Emergency Stop", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return frame
    
    def save_detection_image(self, frame, bottle):
        """Save detection image"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            confidence = bottle['confidence']
            filename = f"{self.detections_dir}/roomba_simple_{timestamp}_conf{confidence:.2f}.jpg"
            cv2.imwrite(filename, frame)
            self.logger.debug(f"💾 Saved: {filename}")
        except Exception as e:
            self.logger.warning(f"Save failed: {e}")
    
    def run(self, max_runtime=20):
        """Main execution loop"""
        self.logger.info("🚀 Starting Bruno Roomba Simple Mode")
        self.logger.info("🏠 Autonomous navigation with bottle detection")
        self.logger.info(f"⏱️  Will stop automatically after {max_runtime} seconds")
        
        self.running = True
        frame_count = 0
        last_status_log = 0
        start_time = time.time()
        
        try:
            while self.running and (time.time() - start_time) < max_runtime:
                ret, frame = self.camera.read()
                if not ret:
                    self.logger.error("Failed to capture frame")
                    time.sleep(1)
                    continue
                
                frame_count += 1
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Periodic status logging
                current_time = time.time()
                if current_time - last_status_log >= 10:  # Every 10 seconds
                    last_status_log = current_time
                    nav_status = self.navigator.get_status()
                    self.logger.info(f"📊 Status: {nav_status['state']} | Bottles: {self.total_detections} detected, {self.bottles_approached} approached")
                
                time.sleep(0.05)  # ~20 FPS
        
        except KeyboardInterrupt:
            self.logger.info("🛑 Stopped by user")
        except Exception as e:
            self.logger.error(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            elapsed_time = time.time() - start_time
            if elapsed_time >= max_runtime:
                self.logger.info(f"⏱️  Auto-stopped after {max_runtime} seconds")
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        
        if self.navigator:
            self.navigator.cleanup()
        
        if self.head_controller:
            self.head_controller.cleanup()
        
        if self.camera:
            self.camera.release()
        
        runtime = time.time() - self.start_time
        self.logger.info("📊 Final Stats:")
        self.logger.info(f"   ⏱️  Runtime: {runtime/60:.1f} minutes")
        self.logger.info(f"   🍼 Detections: {self.total_detections}")
        self.logger.info(f"   🎯 Approached: {self.bottles_approached}")
        
        self.logger.info("✅ Bruno Roomba cleanup complete")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Bruno Roomba Simple')
    parser.add_argument('--config', '-c', help='Configuration file')
    
    args = parser.parse_args()
    
    try:
        bruno = BrunoRoombaSimple(args.config)
        bruno.run()
        
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()