#!/usr/bin/env python3
"""
Bruno Roomba Mode - Autonomous navigation with bottle detection
Moves like a Roomba while detecting and approaching plastic bottles
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

from bottle_detection.enhanced_bottle_detector import EnhancedBottleDetector
from bottle_detection.distance_estimator import DistanceEstimator
from robot_control.roomba_navigator import RoombaNavigator

class BrunoRoomba:
    def __init__(self, config_file: str = None):
        self.setup_logging()
        self.load_config(config_file)
        
        # Initialize components
        self.bottle_detector = EnhancedBottleDetector(self.config.get('detection', {}))
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
        self.navigation_mode = "autonomous"  # autonomous, bottle_approach, manual
        
        # Statistics
        self.start_time = time.time()
        self.distance_traveled = 0  # Simulated
        self.areas_covered = set()  # Simulated coverage
        
        # Detection history for better decision making
        self.recent_detections = []
        self.max_detection_history = 5
        
        # Create output directories
        self.detections_dir = "detections"
        self.navigation_log_dir = "navigation_logs"
        os.makedirs(self.detections_dir, exist_ok=True)
        os.makedirs(self.navigation_log_dir, exist_ok=True)
        
        self.logger.info("🤖 Bruno Roomba Mode initialized")
    
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
        """Load configuration with enhanced defaults"""
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
                    'detection_threshold': 0.4,  # More aggressive detection
                    'motion_detection': True,
                    'adaptive_threshold': True
                },
                'distance_estimation': {
                    'focal_length': 500,
                    'real_bottle_height': 20,
                    'stop_distance': 25,      # Closer approach
                    'approach_distance': 150,  # Wider detection range
                    'max_detection_distance': 300
                },
                'navigation': {
                    'forward_speed': 35,      # Faster movement
                    'turn_speed': 30,
                    'backup_speed': 25,
                    'forward_time_min': 3.0,  # Longer straight segments
                    'forward_time_max': 10.0,
                    'patterns': ['forward', 'spiral', 'random_turn'],
                    'pattern_change_interval': 45.0,  # Change patterns more frequently
                    'stuck_detection': True
                },
                'head_control': {
                    'enabled': True,
                    'response_to_detection': True,
                    'look_at_bottles': True
                },
                'behavior': {
                    'autonomous_navigation': True,
                    'bottle_approach_mode': True,
                    'save_detections': True,
                    'save_navigation_log': True,
                    'approach_timeout': 10.0,     # Max time to approach bottle
                    'search_mode_after_approach': True
                }
            }
    
    def setup_camera(self):
        """Initialize camera with multiple fallbacks"""
        camera_config = self.config['camera']
        
        camera_sources = [
            camera_config['device_id'],
            'http://127.0.0.1:8080?action=stream',
            'http://localhost:8080?action=stream',
            0, 1, -1
        ]
        
        self.camera = None
        
        for i, source in enumerate(camera_sources):
            self.logger.info(f"🔍 Trying camera source {i+1}: {source}")
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
            raise Exception("❌ No working camera found")
        
        # Set camera properties
        try:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config['width'])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config['height'])
            self.camera.set(cv2.CAP_PROP_FPS, 20)  # Lower FPS for better processing
        except:
            pass
    
    def process_frame(self, frame):
        """Process frame for both detection and navigation"""
        # Flip frame if configured
        if self.config['camera']['flip_horizontal']:
            frame = cv2.flip(frame, 1)
        
        # Detect bottles using enhanced detector
        bottles, annotated_frame = self.bottle_detector.detect_bottles(frame)
        
        # Update detection history
        self.recent_detections.append(len(bottles))
        if len(self.recent_detections) > self.max_detection_history:
            self.recent_detections.pop(0)
        
        # Determine navigation mode
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
            
            # Handle bottle approach logic
            self.handle_bottle_detection(best_bottle, movement_command)
        
        # Update navigation system
        should_approach_bottle = (bottles and 
                                 self.config['behavior']['bottle_approach_mode'] and
                                 best_bottle['confidence'] > 0.5)
        
        navigation_command = self.navigator.update_navigation(
            bottle_detected=should_approach_bottle,
            bottle_position=bottle_position
        )
        
        # Handle head movement
        if bottles and self.head_controller:
            self.handle_head_movement(best_bottle, navigation_command)
        
        # Add comprehensive status overlay
        annotated_frame = self.add_status_overlay(annotated_frame, bottles, navigation_command)
        
        # Save detection images
        if bottles and self.config['behavior']['save_detections']:
            self.save_detection_image(annotated_frame, best_bottle, navigation_command)
        
        return annotated_frame
    
    def handle_bottle_detection(self, bottle, movement_command):
        """Handle bottle detection and approach logic"""
        distance_cm = movement_command.get('distance_cm', 0)
        action = movement_command.get('action', 'UNKNOWN')
        
        if action == 'STOP':
            self.bottles_approached += 1
            self.logger.info(f"🎯 BOTTLE REACHED! #{self.bottles_approached} - Distance: {distance_cm:.1f}cm")
            
            # Brief celebration pause
            if self.head_controller:
                try:
                    self.head_controller.nod_yes(2)
                except Exception as e:
                    self.logger.warning(f"Celebration nod failed: {e}")
            
            # Pause briefly then resume navigation
            time.sleep(2)
            
        elif action in ['APPROACH_SLOW', 'APPROACH_NORMAL']:
            confidence = bottle['confidence']
            self.logger.info(f"🚀 Approaching bottle (conf: {confidence:.2f}) - Distance: {distance_cm:.1f}cm")
    
    def handle_head_movement(self, bottle, navigation_command):
        """Handle head movement based on detection and navigation"""
        if not self.head_controller:
            return
        
        try:
            nav_state = navigation_command.get('state', 'unknown')
            
            if nav_state == 'bottle_approach':
                # Look at bottle during approach
                center = bottle['center']
                self.head_controller.look_at_position(center[0], center[1])
                
            elif nav_state in ['turning', 'backing_up']:
                # Look in movement direction
                if 'left' in str(navigation_command.get('action', '')):
                    self.head_controller.move_to_position('left', 0.3)
                elif 'right' in str(navigation_command.get('action', '')):
                    self.head_controller.move_to_position('right', 0.3)
                
            elif nav_state == 'forward' and bottle['confidence'] > 0.6:
                # Acknowledge detected bottle while moving
                self.head_controller.execute_pattern('acknowledgment')
                
        except Exception as e:
            self.logger.warning(f"Head movement error: {e}")
    
    def add_status_overlay(self, frame, bottles, navigation_command):
        """Add comprehensive status overlay"""
        height, width = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, height - 160), (500, height - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        # Navigation status
        nav_state = navigation_command.get('state', 'unknown')
        nav_action = navigation_command.get('action', 'none')
        
        # Calculate runtime
        runtime = time.time() - self.start_time
        runtime_str = f"{int(runtime//60):02d}:{int(runtime%60):02d}"
        
        # Status information
        status_lines = [
            f"🤖 Bruno Roomba Mode - Runtime: {runtime_str}",
            f"🧭 Navigation: {nav_state.upper()} ({nav_action})",
            f"🍼 Bottles: {len(bottles)} detected, {self.total_detections} total",
            f"🎯 Approached: {self.bottles_approached} bottles",
            f"📊 Detection avg: {sum(self.recent_detections)/len(self.recent_detections):.1f}/frame",
            f"🔧 Hardware: {'✅' if self.navigator.hardware_available else '🎮 SIM'}"
        ]
        
        # Add navigation-specific info
        if nav_state == 'bottle_approach':
            status_lines.append("🚀 APPROACHING BOTTLE!")
        elif nav_state == 'backing_up':
            status_lines.append("⬅️ AVOIDING OBSTACLE")
        elif nav_state == 'spiral':
            status_lines.append("🌀 SPIRAL SEARCH PATTERN")
        
        # Draw status text
        y_offset = height - 145
        for line in status_lines:
            cv2.putText(frame, line, (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 18
        
        # Add controls help
        cv2.putText(frame, "Controls: Q=Quit, E=Emergency Stop, S=Save Image", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return frame
    
    def save_detection_image(self, frame, bottle, navigation_command):
        """Save detection images with comprehensive info"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            confidence = bottle['confidence']
            distance = navigation_command.get('distance_cm', 0)
            nav_state = navigation_command.get('state', 'unknown')
            
            filename = f"{self.detections_dir}/roomba_{timestamp}_conf{confidence:.2f}_dist{distance:.0f}cm_{nav_state}.jpg"
            cv2.imwrite(filename, frame)
            self.logger.debug(f"💾 Saved: {filename}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save detection: {e}")
    
    def save_navigation_log(self):
        """Save navigation statistics"""
        if not self.config['behavior']['save_navigation_log']:
            return
        
        try:
            runtime = time.time() - self.start_time
            nav_status = self.navigator.get_status()
            
            log_data = {
                'timestamp': datetime.now().isoformat(),
                'runtime_seconds': runtime,
                'total_detections': self.total_detections,
                'bottles_approached': self.bottles_approached,
                'navigation_state': nav_status['state'],
                'hardware_available': nav_status['hardware_available'],
                'detection_rate': self.total_detections / (runtime / 60) if runtime > 0 else 0  # per minute
            }
            
            filename = f"{self.navigation_log_dir}/bruno_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            self.logger.info(f"📝 Navigation log saved: {filename}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save navigation log: {e}")
    
    def run(self):
        """Main execution loop with Roomba-style navigation"""
        self.logger.info("🚀 Starting Bruno Roomba Mode")
        self.logger.info("🏠 Autonomous navigation with bottle detection")
        self.logger.info("🔄 Moving like a Roomba while hunting bottles!")
        self.logger.info("⏹️  Press Ctrl+C to stop")
        
        self.running = True
        frame_count = 0
        last_status_log = 0
        
        try:
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    self.logger.error("Failed to capture frame")
                    time.sleep(1)
                    continue
                
                frame_count += 1
                
                # Process frame (detection + navigation)
                processed_frame = self.process_frame(frame)
                
                # Periodic status logging
                current_time = time.time()
                if current_time - last_status_log >= 15:  # Every 15 seconds
                    last_status_log = current_time
                    nav_status = self.navigator.get_status()
                    self.logger.info(f"📊 Status: {nav_status['state']} | Bottles: {self.total_detections} detected, {self.bottles_approached} approached")
                
                # Small delay for processing
                time.sleep(0.05)  # ~20 FPS
        
        except KeyboardInterrupt:
            self.logger.info("🛑 Stopped by user (Ctrl+C)")
        except Exception as e:
            self.logger.error(f"❌ Runtime error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def emergency_stop(self):
        """Emergency stop all operations"""
        self.logger.warning("🚨 EMERGENCY STOP ACTIVATED")
        self.navigator.emergency_stop()
        self.running = False
    
    def cleanup(self):
        """Cleanup all resources"""
        self.running = False
        
        # Save final navigation log
        self.save_navigation_log()
        
        # Cleanup components
        if self.navigator:
            self.navigator.cleanup()
        
        if self.head_controller:
            self.head_controller.cleanup()
        
        if self.camera:
            self.camera.release()
        
        # Final statistics
        runtime = time.time() - self.start_time
        self.logger.info("📊 Final Statistics:")
        self.logger.info(f"   ⏱️  Runtime: {runtime/60:.1f} minutes")
        self.logger.info(f"   🍼 Total detections: {self.total_detections}")
        self.logger.info(f"   🎯 Bottles approached: {self.bottles_approached}")
        self.logger.info(f"   📈 Detection rate: {self.total_detections/(runtime/60):.1f} per minute")
        
        self.logger.info("✅ Bruno Roomba cleanup complete")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Bruno Roomba - Autonomous bottle hunting robot')
    parser.add_argument('--config', '-c', 
                       default='config/bruno_config.json',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    try:
        bruno = BrunoRoomba(args.config)
        bruno.run()
        
    except Exception as e:
        print(f"❌ Failed to start Bruno Roomba: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()