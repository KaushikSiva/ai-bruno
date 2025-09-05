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
        self.last_move_time = 0  # Track when last movement started
        self.position_hold_time = 1.5  # seconds to hold each position for face detection
        
        # Define scanning pattern based on HiWonder docs
        self._init_scan_positions()
    
    def _init_scan_positions(self):
        """Initialize comprehensive scanning positions for full room and height coverage."""
        # COMPREHENSIVE SCANNING PATTERN - Floor to Ceiling, Sitting to Standing
        base_positions = [
            # Layer 1: Sitting persons and low positions (Z: 30-35cm)
            (0, 8, 35),     # Center-sitting level
            (20, 8, 35),    # Right-sitting
            (20, 15, 30),   # Right-sitting extended
            (0, 20, 30),    # Center-sitting far
            (-20, 15, 30),  # Left-sitting extended
            (-20, 8, 35),   # Left-sitting
            
            # Layer 2: Standing persons - torso level (Z: 15-25cm)
            (0, 10, 25),    # Center-standing torso
            (25, 10, 25),   # Right-standing wide
            (25, 20, 15),   # Right-standing extended
            (25, 30, 8),    # Right-maximum reach
            (0, 30, 8),     # Center-maximum forward
            (-25, 30, 8),   # Left-maximum reach
            (-25, 20, 15),  # Left-standing extended
            (-25, 10, 25),  # Left-standing wide
            
            # Layer 3: Head level for standing persons (Z: 5-15cm)
            (0, 12, 15),    # Center-head level
            (20, 12, 15),   # Right-head level
            (20, 25, 8),    # Right-head extended
            (-20, 25, 8),   # Left-head extended
            (-20, 12, 15),  # Left-head level
            
            # Layer 4: Intermediate sweep positions for coverage gaps
            (15, 18, 20),   # Right-mid sweep
            (-15, 18, 20),  # Left-mid sweep
            (10, 22, 12),   # Right-diagonal
            (-10, 22, 12),  # Left-diagonal
            
            # Return to optimal center position
            (0, 12, 20),    # Center-optimal
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
        self.last_move_time = 0  # Reset timing
        LOG.info("🔍 Starting comprehensive arm scanning for faces")
    
    def stop_scanning(self):
        """Stop scanning and return to center position."""
        self.is_scanning = False
        if self.arm_ik and HW_AVAILABLE:
            try:
                # Return to optimal center position  
                self.arm_ik.setPitchRangeMoving((0, 12, 20), 0, -90, 90, self.scan_speed)
                LOG.info("🎯 Stopped scanning, returned to center")
            except Exception as e:
                LOG.warning(f"Failed to return arm to center: {e}")
    
    def scan_step(self) -> bool:
        """
        Execute one step of the scanning sequence with proper timing.
        Returns True if a position was executed, False if waiting.
        """
        if not self.is_scanning:
            return False
        
        current_time = time.time()
        
        # Check if enough time has passed since last movement
        if current_time - self.last_move_time < self.position_hold_time:
            return False  # Still holding current position
            
        if self.current_position >= len(self.scan_positions):
            # Completed one full scan cycle, restart from beginning
            self.current_position = 0
            LOG.info("🔄 Completed scan cycle, restarting...")
        
        try:
            pos = self.scan_positions[self.current_position]
            if self.arm_ik and HW_AVAILABLE:
                self.arm_ik.setPitchRangeMoving(pos, 0, -90, 90, self.scan_speed)
                LOG.info(f"🔍 Arm scanning position {self.current_position + 1}/{len(self.scan_positions)}: {pos}")
            elif not HW_AVAILABLE:
                LOG.debug(f"[SIMULATED] Arm moving to position {self.current_position + 1}: {pos}")
            
            self.current_position += 1
            self.last_move_time = current_time
            return True
            
        except Exception as e:
            LOG.error(f"Arm movement failed: {e}")
            self.current_position += 1  # Skip this position and continue
            self.last_move_time = current_time
            return False
    
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
        
        # Servo configuration based on color_tracking.py pattern
        self.horizontal_servo_id = 6  # Left/right camera movement
        self.vertical_servo_id = 3    # Up/down camera movement
        
        # Current positions
        self.horizontal_pulse = 1500  # Center position
        self.vertical_pulse = 1500    # Center position
        
        # Servo limits
        self.min_pulse = 1100
        self.max_pulse = 1900
        self.step_size = 20
        
        # Tracking parameters - improved responsiveness
        self.center_dead_zone = 10  # pixels - smaller for tighter tracking
        self.max_speed = 100  # max pulse change per update - faster response
        self.invert_horizontal = True   # Set to True if horizontal servo direction is backwards
        self.invert_vertical = False    # Set to True if vertical servo direction is backwards
        
        # Predictive tracking system
        self.last_face_pos = None       # (x, y, timestamp)
        self.face_velocity = (0, 0)     # (vx, vy) pixels per second
        self.prediction_factor = 0.3    # How much to predict ahead
        self.velocity_smoothing = 0.7   # Velocity smoothing factor
        self.lock_mode = False          # Enhanced lock mode when face is stable
        
    def center_camera(self):
        """Move camera to center position (both horizontal and vertical)."""
        if self.board and HW_AVAILABLE:
            try:
                self.horizontal_pulse = 1500
                self.vertical_pulse = 1500
                # Set both servos simultaneously like in color_tracking.py
                self.board.pwm_servo_set_position(0.1, [
                    [self.vertical_servo_id, self.vertical_pulse],    # Servo 3 (up/down)
                    [self.horizontal_servo_id, self.horizontal_pulse]  # Servo 6 (left/right)
                ])
                LOG.info("📹 Camera centered (both axes)")
            except Exception as e:
                LOG.warning(f"Failed to center camera: {e}")
    
    def _update_face_velocity(self, face_x: int, face_y: int) -> None:
        """Update face velocity estimation for predictive tracking."""
        current_time = time.time()
        
        if self.last_face_pos is not None:
            last_x, last_y, last_time = self.last_face_pos
            dt = current_time - last_time
            
            if dt > 0:  # Avoid division by zero
                # Calculate instantaneous velocity
                vx = (face_x - last_x) / dt
                vy = (face_y - last_y) / dt
                
                # Apply velocity smoothing
                self.face_velocity = (
                    self.velocity_smoothing * self.face_velocity[0] + (1 - self.velocity_smoothing) * vx,
                    self.velocity_smoothing * self.face_velocity[1] + (1 - self.velocity_smoothing) * vy
                )
        
        self.last_face_pos = (face_x, face_y, current_time)
    
    def _get_predicted_position(self, face_x: int, face_y: int) -> tuple:
        """Get predicted face position based on current velocity."""
        vx, vy = self.face_velocity
        
        # Predict future position based on velocity
        predicted_x = face_x + (vx * self.prediction_factor)
        predicted_y = face_y + (vy * self.prediction_factor)
        
        return int(predicted_x), int(predicted_y)
    
    def _update_lock_mode(self, movement_magnitude: float) -> None:
        """Update enhanced lock mode based on face movement."""
        # Enter lock mode if face is relatively stable
        if movement_magnitude < 20:  # pixels per second
            if not self.lock_mode:
                self.lock_mode = True
                LOG.info("🔒 Enhanced lock mode ENGAGED - face stable")
        else:
            if self.lock_mode:
                self.lock_mode = False
                LOG.info("🔓 Enhanced lock mode RELEASED - face moving fast")
    
    def track_face(self, face_center_x: int, face_center_y: int, frame_width: int, frame_height: int) -> bool:
        """
        Move camera to track face both horizontally and vertically with predictive tracking.
        Returns True if tracking movement was made.
        """
        if not self.board or not HW_AVAILABLE:
            return False
        
        # Update velocity estimation for predictive tracking
        self._update_face_velocity(face_center_x, face_center_y)
        
        # Use predicted position for tracking if face is moving
        vx, vy = self.face_velocity
        movement_magnitude = (vx**2 + vy**2)**0.5
        self._update_lock_mode(movement_magnitude)
        
        # Choose target position based on lock mode and movement
        if self.lock_mode or movement_magnitude < 10:
            # Use current position for stable faces
            target_x, target_y = face_center_x, face_center_y
        else:
            # Use predicted position for moving faces
            target_x, target_y = self._get_predicted_position(face_center_x, face_center_y)
            
        # Calculate errors for both axes using target position
        frame_center_x = frame_width // 2
        frame_center_y = frame_height // 2
        error_x = target_x - frame_center_x
        error_y = target_y - frame_center_y
        
        # Check if face is within dead zone for both axes
        x_in_deadzone = abs(error_x) < self.center_dead_zone
        y_in_deadzone = abs(error_y) < self.center_dead_zone
        
        if x_in_deadzone and y_in_deadzone:
            return False  # No movement needed
        
        movement_made = False
        
        # Horizontal tracking
        if not x_in_deadzone:
            # Calculate horizontal pulse adjustment
            h_pulse_adjustment = int(error_x * 0.8)  # Scale factor for responsiveness
            
            # Apply direction inversion if needed
            if self.invert_horizontal:
                h_pulse_adjustment = -h_pulse_adjustment
                
            h_pulse_adjustment = max(-self.max_speed, min(self.max_speed, h_pulse_adjustment))
            new_h_pulse = self.horizontal_pulse + h_pulse_adjustment
            new_h_pulse = max(self.min_pulse, min(self.max_pulse, new_h_pulse))
            
            if new_h_pulse != self.horizontal_pulse:
                self.horizontal_pulse = new_h_pulse
                movement_made = True
        
        # Vertical tracking  
        if not y_in_deadzone:
            # Calculate vertical pulse adjustment
            v_pulse_adjustment = int(error_y * 0.8)  # Scale factor for responsiveness
            
            # Apply direction inversion if needed
            if self.invert_vertical:
                v_pulse_adjustment = -v_pulse_adjustment
                
            v_pulse_adjustment = max(-self.max_speed, min(self.max_speed, v_pulse_adjustment))
            new_v_pulse = self.vertical_pulse + v_pulse_adjustment
            new_v_pulse = max(self.min_pulse, min(self.max_pulse, new_v_pulse))
            
            if new_v_pulse != self.vertical_pulse:
                self.vertical_pulse = new_v_pulse
                movement_made = True
        
        # Apply servo movements if any adjustments were made
        if movement_made:
            try:
                # Set both servos simultaneously like in color_tracking.py
                self.board.pwm_servo_set_position(0.02, [
                    [self.vertical_servo_id, self.vertical_pulse],      # Servo 3 (up/down)
                    [self.horizontal_servo_id, self.horizontal_pulse]   # Servo 6 (left/right)
                ])
                
                h_dir = "RIGHT" if error_x > 0 else "LEFT"
                v_dir = "UP" if error_y < 0 else "DOWN"  # Y coordinates are inverted in images
                lock_status = "🔒LOCKED" if self.lock_mode else "🎯TRACKING"
                velocity_info = f"V:{movement_magnitude:.1f}px/s"
                LOG.info(f"📹 {lock_status} {h_dir}/{v_dir}: H={self.horizontal_pulse}, V={self.vertical_pulse} {velocity_info}")
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
        
        # Person memory and identification system
        self.tracked_person = None      # Current person being tracked
        self.person_features = None     # Feature vector of current person
        self.person_confidence = 0.0    # Confidence in current person match
        self.person_history = []        # History of person positions
        self.max_history_length = 10   # Keep last 10 positions
        self.identification_threshold = 0.7  # Minimum confidence for person match
        self.feature_memory_time = 30.0      # Remember person for 30 seconds after loss
    
    def detect_faces(self, frame: np.ndarray) -> Optional[FaceInfo]:
        """
        Detect faces in frame and return the best face info with person tracking.
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
        
        # Update person tracking with the best face found
        if best_face is not None:
            self._update_person_tracking(best_face, frame)
        
        return best_face
    
    def _extract_face_features(self, frame: np.ndarray, face: FaceInfo) -> Optional[np.ndarray]:
        """Extract simple features from face for identification (basic implementation)."""
        try:
            # Extract face region
            face_region = frame[face.y:face.y+face.height, face.x:face.x+face.width]
            if face_region.size == 0:
                return None
            
            # Simple feature extraction - normalized face area, aspect ratio, center position
            aspect_ratio = face.width / max(face.height, 1)
            center_x_norm = face.center_x / max(frame.shape[1], 1)
            center_y_norm = face.center_y / max(frame.shape[0], 1)
            area_norm = face.area / max(frame.shape[0] * frame.shape[1], 1)
            
            # Create feature vector
            features = np.array([
                aspect_ratio,
                center_x_norm,
                center_y_norm,
                area_norm,
                face.confidence
            ], dtype=np.float32)
            
            return features
        except Exception:
            return None
    
    def _calculate_person_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate similarity between two feature vectors."""
        try:
            # Simple cosine similarity
            dot_product = np.dot(features1, features2)
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return max(0.0, similarity)  # Clamp to [0, 1]
        except Exception:
            return 0.0
    
    def _update_person_tracking(self, face: Optional[FaceInfo], frame: np.ndarray) -> bool:
        """Update person identification and tracking."""
        if face is None:
            return False
        
        # Extract features from current face
        current_features = self._extract_face_features(frame, face)
        if current_features is None:
            return False
        
        # If we have a tracked person, check if this is the same person
        if self.person_features is not None:
            similarity = self._calculate_person_similarity(self.person_features, current_features)
            
            if similarity >= self.identification_threshold:
                # Same person - update confidence and features
                self.person_confidence = min(1.0, self.person_confidence + 0.1)
                # Blend features for stability
                self.person_features = 0.7 * self.person_features + 0.3 * current_features
                
                # Update position history
                self.person_history.append((face.center_x, face.center_y, time.time()))
                if len(self.person_history) > self.max_history_length:
                    self.person_history.pop(0)
                
                LOG.debug(f"👤 Same person confirmed: confidence={self.person_confidence:.2f}, similarity={similarity:.2f}")
                return True
            else:
                # Different person or lost track
                LOG.info(f"👤 Person changed: similarity={similarity:.2f} < {self.identification_threshold}")
        
        # New person or first detection
        self.person_features = current_features
        self.person_confidence = face.confidence
        self.person_history = [(face.center_x, face.center_y, time.time())]
        self.tracked_person = f"person_{int(time.time())}"  # Simple ID
        
        LOG.info(f"👤 New person detected: ID={self.tracked_person}, confidence={self.person_confidence:.2f}")
        return True
    
    def get_predicted_person_location(self) -> Optional[tuple]:
        """Predict where the person might be based on movement history."""
        if len(self.person_history) < 2:
            return None
        
        # Use last two positions to predict movement
        (x1, y1, t1) = self.person_history[-2]
        (x2, y2, t2) = self.person_history[-1]
        
        dt = t2 - t1
        if dt <= 0:
            return None
        
        # Calculate velocity
        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt
        
        # Predict position 0.5 seconds ahead
        prediction_time = 0.5
        predicted_x = int(x2 + vx * prediction_time)
        predicted_y = int(y2 + vy * prediction_time)
        
        return (predicted_x, predicted_y)
    
    def draw_face_info(self, frame: np.ndarray, face: FaceInfo) -> np.ndarray:
        """Draw face detection information on frame."""
        # Draw bounding box
        cv2.rectangle(frame, (face.x, face.y), 
                     (face.x + face.width, face.y + face.height), 
                     (0, 255, 0), 2)
        
        # Draw center point
        cv2.circle(frame, (face.center_x, face.center_y), 5, (0, 255, 0), -1)
        
        # Draw face info text with person identification
        person_id = self.tracked_person if self.tracked_person else "Unknown"
        info_text = f"ID: {person_id}, Conf: {self.person_confidence:.2f}"
        cv2.putText(frame, info_text, (face.x, face.y - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        area_text = f"Area: {face.area}"
        cv2.putText(frame, area_text, (face.x, face.y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
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
                 headless: bool = False, invert_camera: bool = True, invert_vertical: bool = False):
        self.debug = debug
        self.headless = headless
        self.invert_camera = invert_camera
        self.invert_vertical = invert_vertical
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
        self.camera_controller.invert_horizontal = self.invert_camera  # Apply camera direction settings
        self.camera_controller.invert_vertical = self.invert_vertical
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
        
        # Continue scanning - execute one position at a time
        self.arm_scanner.scan_step()
        
        # No need to restart scanning as scan_step now loops continuously
        
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
        self.camera_controller.track_face(face.center_x, face.center_y, frame.shape[1], frame.shape[0])
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
                # Active face tracking with enhanced fixation (both X and Y axes)
                tracking_moved = self.camera_controller.track_face(face.center_x, face.center_y, frame.shape[1], frame.shape[0])
                
                # Distance feedback
                distance_cat = self.face_tracker.get_distance_category(face)
                distance_color = {
                    "too_far": (0, 0, 255),    # Red
                    "good": (0, 255, 0),       # Green
                    "too_close": (255, 0, 0)   # Blue
                }.get(distance_cat, (128, 128, 128))
                
                # Show tracking status
                tracking_status = "FIXATED" if not tracking_moved else "ADJUSTING"
                cv2.putText(frame, f"TRACKING {tracking_status} - Distance: {distance_cat}", 
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
    parser.add_argument('--invert-camera', action='store_true', default=True,
                       help='Invert horizontal camera servo direction (default: True)')
    parser.add_argument('--invert-vertical', action='store_true', default=False,
                       help='Invert vertical camera servo direction (default: False)')
    
    args = parser.parse_args()
    
    # Create and run face follow system
    face_follow = FaceFollowTest(
        camera_mode=args.mode,
        audio_enabled=args.audio,
        voice=args.voice,
        scan_speed=args.scan_speed,
        debug=args.debug,
        headless=args.headless,
        invert_camera=args.invert_camera,
        invert_vertical=args.invert_vertical
    )
    
    face_follow.run()


if __name__ == '__main__':
    main()