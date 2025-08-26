#!/usr/bin/env python3
"""
Bruno Simple Mode - Basic bottle detection and approach
Simplified version without complex threading or signal handling
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

class BrunoSimple:
    def __init__(self, config_file: str = None):
        self.setup_logging()
        self.load_config(config_file)
        
        # Initialize components
        self.bottle_detector = BottleDetector(self.config.get('detection', {}))
        self.distance_estimator = DistanceEstimator(self.config.get('distance_estimation', {}))
        
        # Initialize movement and head controllers with error handling
        try:
            from robot_control.head_controller import HeadController
            self.head_controller = HeadController(self.config.get('head_control', {}))
            self.logger.info("✅ Head controller initialized")
        except Exception as e:
            self.logger.warning(f"⚠️  Head controller failed: {e}")
            self.head_controller = None
        
        try:
            from robot_control.movement_controller import MovementController
            self.movement_controller = MovementController(self.config.get('movement_control', {}))
            self.logger.info("✅ Movement controller initialized")
        except Exception as e:
            self.logger.warning(f"⚠️  Movement controller failed: {e}")
            self.movement_controller = None
        
        # Setup camera
        self.setup_camera()
        
        # State variables
        self.running = False
        self.detection_count = 0
        self.bottles_reached = 0
        self.approach_mode = False
        self.last_bottles = []
        self.last_movement_command = None
        
        # Create detections directory
        self.detections_dir = "detections"
        os.makedirs(self.detections_dir, exist_ok=True)
    
    def setup_logging(self):
        """Setup logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('bruno_simple.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_config(self, config_file: str = None):
        """Load configuration from file or use defaults"""
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                'camera': {
                    'device_id': "http://127.0.0.1:8080?action=stream",
                    'fallback_device_id': 0,
                    'width': 640,
                    'height': 480,
                    'flip_horizontal': True
                },
                'detection': {
                    'confidence_threshold': 0.6
                },
                'distance_estimation': {
                    'stop_distance': 30,
                    'approach_distance': 100
                },
                'behavior': {
                    'approach_bottles_automatically': True,
                    'save_detections': True
                }
            }
    
    def setup_camera(self):
        """Initialize camera with fallbacks"""
        camera_config = self.config['camera']
        
        # Try different camera sources
        camera_sources = [
            camera_config['device_id'],
            'http://127.0.0.1:8080?action=stream',
            'http://localhost:8080?action=stream',
            0, 1
        ]
        
        self.camera = None
        
        for i, source in enumerate(camera_sources):
            self.logger.info(f"Trying camera source {i+1}: {source}")
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
                self.logger.warning(f"Camera source {source} failed: {e}")
        
        if not self.camera:
            raise Exception("No working camera found")
        
        # Set camera properties
        try:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config['width'])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config['height'])
        except:
            pass
    
    def process_frame(self, frame):
        """Process a single frame"""
        # Flip frame if configured
        if self.config['camera']['flip_horizontal']:
            frame = cv2.flip(frame, 1)
        
        # Detect bottles
        bottles, annotated_frame = self.bottle_detector.detect_bottles(frame)
        self.last_bottles = bottles
        
        if bottles:
            self.detection_count += 1
            best_bottle = self.bottle_detector.get_best_bottle(bottles)
            
            # Get distance and movement info
            frame_height, frame_width = frame.shape[:2]
            movement_command = self.distance_estimator.get_movement_command(
                best_bottle, frame_width, frame_height
            )
            self.last_movement_command = movement_command
            
            # Draw distance info
            annotated_frame = self.distance_estimator.draw_distance_info(
                annotated_frame, best_bottle, movement_command
            )
            
            # Handle movement
            self.handle_movement(movement_command)
            
            # Handle head movement
            self.handle_head_movement(best_bottle)
            
            # Save detection if enabled
            if self.config['behavior'].get('save_detections', True):
                self.save_detection_image(annotated_frame, movement_command)
        
        # Add status overlay
        self.add_status_overlay(annotated_frame, bottles)
        
        return annotated_frame
    
    def handle_movement(self, movement_command):
        """Handle movement commands"""
        if not self.movement_controller or not movement_command:
            return
        
        action = movement_command.get('action', 'STOP')
        distance_cm = movement_command.get('distance_cm', 0)
        
        if action == 'STOP':
            if self.approach_mode:
                self.bottles_reached += 1
                self.approach_mode = False
                self.logger.info(f"🎯 BOTTLE REACHED! Distance: {distance_cm:.1f}cm")
        elif action in ['APPROACH_SLOW', 'APPROACH_NORMAL']:
            if not self.approach_mode:
                self.approach_mode = True
                self.logger.info(f"🚀 Approaching bottle - Distance: {distance_cm:.1f}cm")
        
        # Execute movement command
        self.movement_controller.execute_movement_command(movement_command)
    
    def handle_head_movement(self, best_bottle):
        """Handle head movement responses"""
        if not self.head_controller:
            return
        
        try:
            # Look at bottle
            center = best_bottle['center']
            self.head_controller.look_at_position(center[0], center[1])
            
            # Respond based on confidence
            confidence = best_bottle['confidence']
            if confidence > 0.8:
                self.head_controller.execute_pattern('excited')
            elif confidence > 0.6:
                self.head_controller.execute_pattern('acknowledgment')
        except Exception as e:
            self.logger.warning(f"Head movement error: {e}")
    
    def save_detection_image(self, frame, movement_command):
        """Save detection images"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            distance_cm = movement_command.get('distance_cm', 0)
            action = movement_command.get('action', 'UNKNOWN')
            
            filename = f"{self.detections_dir}/bottle_{timestamp}_{distance_cm:.0f}cm_{action}.jpg"
            cv2.imwrite(filename, frame)
        except Exception as e:
            self.logger.warning(f"Failed to save detection: {e}")
    
    def add_status_overlay(self, frame, bottles):
        """Add status information to frame"""
        height, width = frame.shape[:2]
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, height - 120), (400, height - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Status text
        status_lines = [
            f"Bottles: {len(bottles)}",
            f"Detections: {self.detection_count}",
            f"Reached: {self.bottles_reached}",
            f"Mode: {'APPROACHING' if self.approach_mode else 'SCANNING'}"
        ]
        
        if self.last_movement_command:
            cmd = self.last_movement_command
            distance_cm = cmd.get('distance_cm', 0)
            action = cmd.get('action', 'IDLE')
            status_lines.append(f"Distance: {distance_cm:.1f}cm - {action}")
        
        y_offset = height - 100
        for line in status_lines:
            cv2.putText(frame, line, (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 15
    
    def run(self):
        """Main execution loop"""
        self.logger.info("🤖 Starting Bruno Simple Mode")
        self.logger.info("📱 Headless operation - no display windows")
        self.logger.info("🎯 Press Ctrl+C to stop")
        
        self.running = True
        frame_count = 0
        last_log_time = 0
        
        try:
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    self.logger.error("Failed to capture frame")
                    time.sleep(1)
                    continue
                
                frame_count += 1
                
                # Process frame
                self.process_frame(frame)
                
                # Log status every 5 seconds
                current_time = time.time()
                if current_time - last_log_time >= 5:
                    last_log_time = current_time
                    bottles = len(self.last_bottles)
                    self.logger.info(f"📊 Status: {bottles} bottles, {self.detection_count} detections, {self.bottles_reached} reached")
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.05)
        
        except KeyboardInterrupt:
            self.logger.info("🛑 Stopped by user (Ctrl+C)")
        except Exception as e:
            self.logger.error(f"❌ Error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        
        if self.camera:
            self.camera.release()
        
        if self.movement_controller:
            self.movement_controller.cleanup()
        
        if self.head_controller:
            self.head_controller.cleanup()
        
        self.logger.info("✅ Bruno Simple cleanup complete")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Bruno Simple - Bottle Detection Robot')
    parser.add_argument('--config', '-c', 
                       default='config/bruno_config.json',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    try:
        bruno = BrunoSimple(args.config)
        bruno.run()
        
    except Exception as e:
        print(f"❌ Failed to start Bruno: {e}")
        print("🔧 Try running: python test_imports.py")

if __name__ == "__main__":
    main()