#!/usr/bin/env python3
# coding: utf-8
"""
Bruno Face Follow Test - Enhanced Face Detection and Tracking System

Features:
- Systematic arm scanning to find faces (based on HiWonder section 4.4.3)
- MediaPipe face detection and recognition (based on section 5.7)
- Camera servo control for smooth face following
- State machine: SCANNING → FACE_DETECTED → TRACKING → (FACE_LOST → SCANNING)
- Safety systems and audio feedback integration

Usage:
  python3 face_follow_test.py --mode external --audio --voice Dominus
  python3 face_follow_test.py --mode builtin --scan-speed 1.5 --debug

Dependencies:
- mediapipe, opencv-python
- MasterPi SDK for servo control and arm kinematics
- bruno_surveillance audio and camera systems
"""

import os
import sys
import time
import math
import argparse
from enum import Enum
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

# Add MasterPi SDK to path
sys.path.append('/home/pi/MasterPi')

import cv2
import numpy as np
import mediapipe as mp

# Bruno surveillance imports
from utils import LOG
from camera_shared import make_camera, read_or_reconnect
from audio_tts import TTSSpeaker

# Hardware imports with fallback
try:
    from common.ros_robot_controller_sdk import Board
    from kinematics.arm_move_ik import ArmIK
    HW_AVAILABLE = True
    LOG.info("Hardware SDK loaded successfully")
except ImportError as e:
    LOG.warning(f"Hardware SDK not available: {e}")
    Board = None
    ArmIK = None
    HW_AVAILABLE = False


class FaceFollowState(Enum):
    """State machine states for face following behavior."""
    INITIALIZING = "initializing"
    SCANNING = "scanning" 
    FACE_DETECTED = "face_detected"
    TRACKING = "tracking"
    FACE_LOST = "face_lost"
    ERROR = "error"


@dataclass
class FaceInfo:
    """Information about a detected face."""
    x: int
    y: int
    width: int
    height: int
    center_x: int
    center_y: int
    area: int
    confidence: float
    timestamp: float


class ArmScanner:
    """
    Handles systematic arm movements for face scanning.
    Based on HiWonder section 4.4.3 movement patterns.
    """
    
    def __init__(self, board: Optional[Board] = None, arm_ik: Optional[ArmIK] = None):
        self.board = board
        self.arm_ik = arm_ik
        self.current_position = 0
        self.scan_positions = []
        self.scan_speed = 1200  # milliseconds for movement - faster scanning
        self.is_scanning = False
        
        # Define scanning pattern based on HiWonder docs
        self._init_scan_positions()
    
    def _init_scan_positions(self):
        """Initialize the scanning positions for comprehensive face scanning."""
        # Enhanced scanning pattern with wider coverage and better arm extension
        base_positions = [
            (0, 10, 25),    # Center/initial - extended reach
            (15, 10, 25),   # Right - wider angle
            (15, 18, 15),   # Right-high - full extension
            (15, 25, 8),    # Right-far - maximum reach
            (0, 25, 8),     # Center-far - maximum forward reach
            (-15, 25, 8),   # Left-far - maximum reach
            (-15, 18, 15),  # Left-high - full extension  
            (-15, 10, 25),  # Left - wider angle
            (-8, 15, 20),   # Left-mid - intermediate position
            (8, 15, 20),    # Right-mid - intermediate position
            (0, 10, 25),    # Back to center
        ]
        
        # Add intermediate positions for smoother scanning
        self.scan_positions = []
        for pos in base_positions:
            self.scan_positions.append(pos)
            # Add intermediate position for smoother movement
            if pos != base_positions[-1]:  # Don't add after last position
                next_pos = base_positions[base_positions.index(pos) + 1]
                intermediate = (
                    (pos[0] + next_pos[0]) / 2,
                    (pos[1] + next_pos[1]) / 2,
                    (pos[2] + next_pos[2]) / 2
                )
                self.scan_positions.append(intermediate)
    
    def start_scanning(self):
        """Start the scanning sequence."""
        self.is_scanning = True
        self.current_position = 0
        LOG.info("🔍 Starting arm scanning for faces")
    
    def stop_scanning(self):
        """Stop scanning and return to center position."""
        self.is_scanning = False
        if self.arm_ik and HW_AVAILABLE:
            try:
                # Return to center position  
                self.arm_ik.setPitchRangeMoving((0, 10, 25), 0, -90, 90, self.scan_speed)
                LOG.info("🎯 Stopped scanning, returned to center")
            except Exception as e:
                LOG.warning(f"Failed to return arm to center: {e}")
    
    def scan_step(self) -> bool:
        """
        Execute one step of the scanning sequence.
        Returns True if scanning is complete, False if continuing.
        """
        if not self.is_scanning or not HW_AVAILABLE:
            return True
            
        if self.current_position >= len(self.scan_positions):
            # Completed one full scan cycle
            self.current_position = 0
            return True
        
        try:
            pos = self.scan_positions[self.current_position]
            if self.arm_ik:
                self.arm_ik.setPitchRangeMoving(pos, 0, -90, 90, self.scan_speed)
                LOG.debug(f"Arm moving to position {self.current_position}: {pos}")
            
            self.current_position += 1
            return False
            
        except Exception as e:
            LOG.error(f"Arm movement failed: {e}")
            return True
    
    def set_scan_speed(self, speed_ms: int):
        """Set scanning speed in milliseconds."""
        self.scan_speed = max(500, min(3000, speed_ms))  # Clamp between 0.5-3 seconds


class CameraMountController:
    """
    Controls camera servo positioning for face tracking.
    Based on patterns from face_recognition.py servo control.
    """
    
    def __init__(self, board: Optional[Board] = None):
        self.board = board
        self.camera_servo_id = 6  # Horizontal camera servo from face_recognition.py
        self.current_pulse = 1500  # Center position
        self.min_pulse = 1100
        self.max_pulse = 1900
        self.step_size = 20
        
        # Tracking parameters - improved responsiveness
        self.center_dead_zone = 15  # pixels - smaller for better tracking
        self.max_speed = 80  # max pulse change per update - faster response
        
    def center_camera(self):
        """Move camera to center position."""
        if self.board and HW_AVAILABLE:
            try:
                self.current_pulse = 1500
                self.board.pwm_servo_set_position(0.1, [[self.camera_servo_id, self.current_pulse]])
                LOG.debug("Camera centered")
            except Exception as e:
                LOG.warning(f"Failed to center camera: {e}")
    
    def track_face(self, face_center_x: int, frame_width: int) -> bool:
        """
        Move camera to track face horizontally.
        Returns True if tracking movement was made.
        """
        if not self.board or not HW_AVAILABLE:
            return False
            
        frame_center = frame_width // 2
        error = face_center_x - frame_center
        
        # Dead zone check
        if abs(error) < self.center_dead_zone:
            return False
        
        # Calculate pulse adjustment with improved responsiveness
        # Negative error = face is left of center, servo should move left (decrease pulse)
        # Positive error = face is right of center, servo should move right (increase pulse)
        pulse_adjustment = int(error * 0.8)  # Increased scale factor for better responsiveness
        pulse_adjustment = max(-self.max_speed, min(self.max_speed, pulse_adjustment))
        
        new_pulse = self.current_pulse + pulse_adjustment
        new_pulse = max(self.min_pulse, min(self.max_pulse, new_pulse))
        
        if new_pulse != self.current_pulse:
            try:
                self.current_pulse = new_pulse
                self.board.pwm_servo_set_position(0.02, [[self.camera_servo_id, self.current_pulse]])  # Faster update rate
                LOG.debug(f"Camera tracking: pulse={self.current_pulse}, error={error}")
                return True
            except Exception as e:
                LOG.warning(f"Camera tracking failed: {e}")
        
        return False


class FaceTracker:
    """
    Face detection and tracking using MediaPipe.
    Based on HiWonder section 5.7 face recognition implementation.
    """
    
    def __init__(self, detection_confidence: float = 0.8):
        # Initialize MediaPipe face detection
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=detection_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Tracking state
        self.last_face: Optional[FaceInfo] = None
        self.face_lost_time: Optional[float] = None
        self.face_lost_threshold = 0.8  # seconds - shorter for fast movement tracking
        
        # Face area thresholds for distance estimation
        self.min_face_area = 1000    # Too far away
        self.max_face_area = 50000   # Too close
        self.target_face_area = 15000 # Optimal distance
    
    def detect_faces(self, frame: np.ndarray) -> Optional[FaceInfo]:
        """
        Detect faces in frame and return the best face info.
        Based on face_recognition.py MediaPipe integration.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(frame_rgb)
        
        if not results.detections:
            return None
        
        # Find the largest/most confident face
        best_face = None
        best_score = 0
        frame_h, frame_w = frame.shape[:2]
        
        for detection in results.detections:
            confidence = detection.score[0]
            
            if confidence > best_score and confidence > 0.7:  # Minimum confidence
                # Convert normalized coordinates to pixel coordinates
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * frame_w)
                y = int(bbox.ymin * frame_h)
                width = int(bbox.width * frame_w)
                height = int(bbox.height * frame_h)
                
                # Ensure valid bounding box
                if width > 0 and height > 0:
                    face_info = FaceInfo(
                        x=x, y=y, width=width, height=height,
                        center_x=x + width // 2,
                        center_y=y + height // 2,
                        area=width * height,
                        confidence=confidence,
                        timestamp=time.time()
                    )
                    
                    if face_info.area > best_score:  # Use area as tie-breaker
                        best_face = face_info
                        best_score = face_info.area
        
        return best_face
    
    def draw_face_info(self, frame: np.ndarray, face: FaceInfo) -> np.ndarray:
        """Draw face detection information on frame."""
        # Draw bounding box
        cv2.rectangle(frame, (face.x, face.y), 
                     (face.x + face.width, face.y + face.height), 
                     (0, 255, 0), 2)
        
        # Draw center point
        cv2.circle(frame, (face.center_x, face.center_y), 5, (0, 255, 0), -1)
        
        # Draw face info text
        info_text = f"Area: {face.area}, Conf: {face.confidence:.2f}"
        cv2.putText(frame, info_text, (face.x, face.y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def update_tracking_state(self, face: Optional[FaceInfo]) -> bool:
        """
        Update face tracking state.
        Returns True if face is being tracked, False if lost.
        """
        current_time = time.time()
        
        if face:
            self.last_face = face
            self.face_lost_time = None
            return True
        else:
            if self.face_lost_time is None:
                self.face_lost_time = current_time
            
            # Check if face has been lost for too long
            time_lost = current_time - self.face_lost_time
            return time_lost < self.face_lost_threshold
    
    def get_distance_category(self, face: FaceInfo) -> str:
        """Categorize face distance based on area."""
        if face.area < self.min_face_area:
            return "too_far"
        elif face.area > self.max_face_area:
            return "too_close"
        else:
            return "good"


class FaceFollowTest:
    """
    Main face following system integrating all components.
    """
    
    def __init__(self, camera_mode: str = "external", audio_enabled: bool = False, 
                 voice: str = "Dominus", scan_speed: float = 1.5, debug: bool = False,
                 headless: bool = False):
        self.debug = debug
        self.headless = headless
        self.state = FaceFollowState.INITIALIZING
        
        # Initialize hardware
        self.board = Board() if HW_AVAILABLE else None
        self.arm_ik = ArmIK() if HW_AVAILABLE else None
        if self.arm_ik and self.board:
            self.arm_ik.board = self.board
        
        # Initialize camera
        self.camera = make_camera(camera_mode, retry_attempts=3, retry_delay=2.0)
        
        # Initialize audio
        self.speaker = TTSSpeaker(enabled=audio_enabled, voice=voice) if audio_enabled else None
        if self.speaker:
            self.speaker.start()
        
        # Initialize subsystems
        self.arm_scanner = ArmScanner(self.board, self.arm_ik)
        self.camera_controller = CameraMountController(self.board)
        self.face_tracker = FaceTracker()
        
        # Configuration
        self.arm_scanner.set_scan_speed(int(scan_speed * 1000))  # Convert to milliseconds
        
        # State tracking
        self.greeted_this_session = False
        self.last_state_change = time.time()
        self.state_timeout = 10.0  # seconds
        
        # Performance tracking
        self.loop_count = 0
        self.fps_counter = time.time()
        
        LOG.info(f"🤖 Face Follow Test initialized (HW: {'✓' if HW_AVAILABLE else '✗'}, Display: {'Headless' if headless else 'GUI'})")
    
    def _change_state(self, new_state: FaceFollowState, reason: str = ""):
        """Change state and log the transition."""
        if new_state != self.state:
            old_state = self.state
            self.state = new_state
            self.last_state_change = time.time()
            LOG.info(f"State: {old_state.value} → {new_state.value} ({reason})")
    
    def _greet_person(self):
        """Greet detected person with nod and audio."""
        if self.greeted_this_session:
            return
            
        self.greeted_this_session = True
        LOG.info("👋 Greeting detected person")
        
        # Nod gesture (servo 3 - head movement)
        if self.board and HW_AVAILABLE:
            try:
                nod_sequence = [
                    (500, 0.15),   # Look up
                    (900, 0.15),   # Look down  
                    (500, 0.15),   # Look up
                    (700, 0.15)    # Return to neutral
                ]
                for pulse, duration in nod_sequence:
                    self.board.pwm_servo_set_position(0.1, [[3, pulse]])
                    time.sleep(duration)
            except Exception as e:
                LOG.warning(f"Greeting gesture failed: {e}")
        
        # Audio greeting
        if self.speaker:
            try:
                greetings = [
                    "Hello! I can see you there.",
                    "Hi! Nice to meet you.",
                    "Hey there! I'm Bruno.",
                    "Hello! I'll follow you around."
                ]
                import random
                greeting = random.choice(greetings)
                self.speaker.speak_sync(greeting)
            except Exception as e:
                LOG.warning(f"Audio greeting failed: {e}")
    
    def _handle_state_scanning(self, frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Handle SCANNING state logic."""
        if frame is None:
            return None
        
        # Look for faces while scanning
        face = self.face_tracker.detect_faces(frame)
        if face:
            self._change_state(FaceFollowState.FACE_DETECTED, f"Found face (area: {face.area}, conf: {face.confidence:.2f})")
            self.arm_scanner.stop_scanning()
            return self.face_tracker.draw_face_info(frame, face)
        
        # Continue scanning
        scan_complete = self.arm_scanner.scan_step()
        if scan_complete:
            # Start new scan cycle
            self.arm_scanner.start_scanning()
        
        # Add scanning indicator to frame with position info
        scan_pos = f"{self.arm_scanner.current_position}/{len(self.arm_scanner.scan_positions)}"
        cv2.putText(frame, f"SCANNING FOR FACES ({scan_pos})", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return frame
    
    def _handle_state_face_detected(self, frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Handle FACE_DETECTED state logic.""" 
        if frame is None:
            self._change_state(FaceFollowState.SCANNING, "Lost camera feed")
            return None
        
        # Verify face is still there
        face = self.face_tracker.detect_faces(frame)
        if not face:
            self._change_state(FaceFollowState.FACE_LOST, "Face disappeared")
            return frame
        
        # Center camera on face and greet
        self.camera_controller.track_face(face.center_x, frame.shape[1])
        self._greet_person()
        
        # Transition to tracking
        self._change_state(FaceFollowState.TRACKING, "Face confirmed, starting tracking")
        
        return self.face_tracker.draw_face_info(frame, face)
    
    def _handle_state_tracking(self, frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Handle TRACKING state logic."""
        if frame is None:
            self._change_state(FaceFollowState.FACE_LOST, "Lost camera feed")
            return None
        
        face = self.face_tracker.detect_faces(frame)
        
        if self.face_tracker.update_tracking_state(face):
            if face:
                # Active tracking
                self.camera_controller.track_face(face.center_x, frame.shape[1])
                
                # Distance feedback
                distance_cat = self.face_tracker.get_distance_category(face)
                distance_color = {
                    "too_far": (0, 0, 255),    # Red
                    "good": (0, 255, 0),       # Green
                    "too_close": (255, 0, 0)   # Blue
                }.get(distance_cat, (128, 128, 128))
                
                cv2.putText(frame, f"TRACKING - Distance: {distance_cat}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, distance_color, 2)
                
                return self.face_tracker.draw_face_info(frame, face)
            else:
                # Face temporarily lost but still in grace period
                cv2.putText(frame, "TRACKING - Face temporarily lost", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                return frame
        else:
            # Face lost for too long
            self._change_state(FaceFollowState.FACE_LOST, "Face lost for too long")
            return frame
    
    def _handle_state_face_lost(self, frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Handle FACE_LOST state logic."""
        if frame is None:
            self._change_state(FaceFollowState.SCANNING, "No camera feed")
            return None
        
        # Quick check for face before starting full scan
        face = self.face_tracker.detect_faces(frame)
        if face:
            self._change_state(FaceFollowState.FACE_DETECTED, "Face reappeared quickly")
            return self.face_tracker.draw_face_info(frame, face)
        
        # No face found, return to comprehensive scanning
        self._change_state(FaceFollowState.SCANNING, "Starting comprehensive scan cycle")
        self.arm_scanner.start_scanning()
        
        cv2.putText(frame, "FACE LOST - Returning to scan", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def _draw_debug_info(self, frame: np.ndarray) -> np.ndarray:
        """Draw debug information on frame."""
        if not self.debug:
            return frame
        
        # Performance info
        self.loop_count += 1
        if time.time() - self.fps_counter >= 1.0:
            fps = self.loop_count
            self.loop_count = 0
            self.fps_counter = time.time()
            
            # State info
            cv2.putText(frame, f"State: {self.state.value}", 
                       (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"FPS: {fps}", 
                       (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"HW: {'Available' if HW_AVAILABLE else 'Simulated'}", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Main execution loop."""
        LOG.info("🚀 Starting Face Follow Test")
        if self.headless:
            LOG.info("🖥️  Running in headless mode - no GUI display")
        
        if not self.camera.open():
            LOG.error("❌ Cannot open camera")
            return
        
        # Initialize system
        self.camera_controller.center_camera()
        self._change_state(FaceFollowState.SCANNING, "System initialized")
        self.arm_scanner.start_scanning()
        
        last_frame = None
        
        try:
            while True:
                # Read camera frame
                last_frame = read_or_reconnect(self.camera, last_frame)
                
                # State machine processing
                if self.state == FaceFollowState.SCANNING:
                    processed_frame = self._handle_state_scanning(last_frame)
                elif self.state == FaceFollowState.FACE_DETECTED:
                    processed_frame = self._handle_state_face_detected(last_frame)
                elif self.state == FaceFollowState.TRACKING:
                    processed_frame = self._handle_state_tracking(last_frame)
                elif self.state == FaceFollowState.FACE_LOST:
                    processed_frame = self._handle_state_face_lost(last_frame)
                else:
                    processed_frame = last_frame
                
                # Display frame with debug info (only in GUI mode)
                if processed_frame is not None and not self.headless:
                    display_frame = self._draw_debug_info(processed_frame)
                    
                    # Resize for display
                    display_frame = cv2.resize(display_frame, (640, 480))
                    cv2.imshow('Bruno Face Follow Test', display_frame)
                
                # Handle keyboard input (GUI mode only)
                if not self.headless:
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC key
                        break
                    elif key == ord('r'):  # Reset
                        LOG.info("Manual reset requested")
                        self.greeted_this_session = False
                        self._change_state(FaceFollowState.SCANNING, "Manual reset")
                        self.arm_scanner.start_scanning()
                
                # Small delay to prevent excessive CPU usage - reduced for better responsiveness
                time.sleep(0.02)
                
        except KeyboardInterrupt:
            LOG.info("🛑 Interrupted by user")
        except Exception as e:
            LOG.error(f"❌ Unexpected error: {e}")
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up resources."""
        LOG.info("🧹 Cleaning up resources")
        
        try:
            # Stop scanning and center arm
            self.arm_scanner.stop_scanning()
            self.camera_controller.center_camera()
            
            # Close camera
            if hasattr(self.camera, 'close'):
                self.camera.close()
            
            # Stop audio
            if self.speaker:
                self.speaker.stop()
            
            # Close OpenCV windows (GUI mode only)
            if not self.headless:
                cv2.destroyAllWindows()
            
        except Exception as e:
            LOG.warning(f"Cleanup error: {e}")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description='Bruno Face Follow Test System')
    parser.add_argument('--mode', choices=['builtin', 'external'], 
                       default=os.environ.get('CAM_MODE', 'external'),
                       help='Camera mode (default: external)')
    parser.add_argument('--audio', action='store_true', 
                       help='Enable audio greetings')
    parser.add_argument('--voice', default=os.environ.get('BRUNO_AUDIO_VOICE', 'Dominus'),
                       help='TTS voice (default: Dominus)')
    parser.add_argument('--scan-speed', type=float, default=1.5,
                       help='Arm scanning speed in seconds (default: 1.5)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug information display')
    parser.add_argument('--headless', action='store_true',
                       help='Run without GUI display (for headless systems)')
    
    args = parser.parse_args()
    
    # Create and run face follow system
    face_follow = FaceFollowTest(
        camera_mode=args.mode,
        audio_enabled=args.audio,
        voice=args.voice,
        scan_speed=args.scan_speed,
        debug=args.debug,
        headless=args.headless
    )
    
    face_follow.run()


if __name__ == '__main__':
    main()