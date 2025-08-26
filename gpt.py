#!/usr/bin/env python3
"""
Bruno GPT Vision Bottle Detection and Pickup System

Enhanced bottle detection using OpenAI's GPT Vision API integrated with the Bruno robot platform.
This module provides AI-powered bottle detection and automated pickup behavior with collision avoidance.

Key Features:
- Uses OpenAI GPT Vision for intelligent bottle and bin detection
- Integrates with existing Bruno robot control systems
- Supports both local camera and network camera streams
- Configurable detection parameters and robot behavior
- Fallback to local OpenCV detection if API is unavailable
- Advanced collision avoidance and safety systems
- Obstacle detection and avoidance
- Emergency stop functionality
- Path planning and navigation improvements

Usage:
    python3 gpt.py --config config/bruno_config.json --dry-run
    python3 gpt.py --camera-url http://127.0.0.1:8080?action=stream --save-debug
"""

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import time
import logging
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from pathlib import Path
from enum import Enum

import numpy as np
from PIL import Image

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from openai import OpenAI, APIError
    OPENAI_AVAILABLE = True
except ImportError:
    print("Warning: OpenAI library not available. Install with: pip install openai")
    OPENAI_AVAILABLE = False

# Import Bruno project modules
try:
    from src.robot_control.movement_controller import MovementController
    from src.robot_control.head_controller import HeadController
    from src.bottle_detection.bottle_detector import BottleDetector
    BRUNO_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Bruno modules not available: {e}")
    BRUNO_MODULES_AVAILABLE = False

# ------------------------- Safety and Collision Avoidance -------------------------

class SafetyLevel(Enum):
    """Safety levels for robot operation"""
    SAFE = "safe"
    CAUTION = "caution"
    DANGER = "danger"
    EMERGENCY = "emergency"

@dataclass
class ObstacleInfo:
    """Information about detected obstacles"""
    distance: float
    angle: float
    confidence: float
    type: str  # "wall", "object", "person", "unknown"
    bbox: Optional[Tuple[int, int, int, int]] = None

class CollisionAvoidance:
    """
    Advanced collision avoidance system using computer vision and safety zones.
    """
    def __init__(self, config: Dict):
        self.config = config.get('collision_avoidance', self._default_config())
        self.setup_logging()
        
        # Safety zones (in pixels, assuming 640x480 camera)
        self.safe_distance = self.config.get('safe_distance', 100)  # pixels
        self.caution_distance = self.config.get('caution_distance', 150)  # pixels
        self.danger_distance = self.config.get('danger_distance', 80)  # pixels
        
        # Movement restrictions
        self.min_safe_speed = self.config.get('min_safe_speed', 0.1)
        self.max_safe_speed = self.config.get('max_safe_speed', 0.4)
        
        # Obstacle tracking
        self.obstacles: List[ObstacleInfo] = []
        self.last_obstacle_check = 0.0
        self.obstacle_check_interval = self.config.get('obstacle_check_interval', 0.5)
        
        # Emergency stop
        self.emergency_stop_active = False
        self.emergency_stop_timeout = self.config.get('emergency_stop_timeout', 5.0)
        self.last_emergency_stop = 0.0
        
    def _default_config(self) -> Dict:
        """Default collision avoidance configuration"""
        return {
            'safe_distance': 100,      # pixels
            'caution_distance': 150,   # pixels
            'danger_distance': 80,     # pixels
            'min_safe_speed': 0.1,
            'max_safe_speed': 0.4,
            'obstacle_check_interval': 0.5,
            'emergency_stop_timeout': 5.0,
            'enable_obstacle_detection': True,
            'enable_emergency_stop': True,
            'enable_speed_limiting': True
        }
    
    def setup_logging(self):
        """Setup logging for collision avoidance"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def detect_obstacles(self, frame: np.ndarray) -> List[ObstacleInfo]:
        """
        Detect obstacles in the frame using edge detection and depth estimation.
        This is a simplified version - in a real implementation, you might use:
        - Depth sensors (LiDAR, stereo cameras)
        - Machine learning models for obstacle detection
        - Ultrasonic sensors
        """
        obstacles = []
        H, W = frame.shape[:2]
        
        try:
            import cv2
            
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            self.logger.debug(f"Found {len(contours)} contours in frame")
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area < 500:  # Filter small noise
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate distance estimate (simplified - assumes larger objects are closer)
                # In reality, you'd use proper depth estimation
                distance = self._estimate_distance(w, h, W, H)
                
                # Calculate angle from center
                center_x = x + w/2
                angle = np.arctan2(center_x - W/2, W/2) * 180 / np.pi
                
                # Determine obstacle type based on size and position
                obstacle_type = self._classify_obstacle(w, h, distance, angle)
                
                obstacle = ObstacleInfo(
                    distance=distance,
                    angle=angle,
                    confidence=min(area / 10000, 1.0),  # Normalize confidence
                    type=obstacle_type,
                    bbox=(x, y, w, h)
                )
                obstacles.append(obstacle)
                
                self.logger.debug(f"Obstacle {i+1}: {obstacle_type} at {distance:.1f}px, angle {angle:.1f}°, size {w}x{h}, conf {obstacle.confidence:.2f}")
                
        except ImportError:
            self.logger.warning("OpenCV not available for obstacle detection")
        except Exception as e:
            self.logger.error(f"Obstacle detection failed: {e}")
        
        self.logger.debug(f"Total obstacles detected: {len(obstacles)}")
        return obstacles
    
    def _estimate_distance(self, width: int, height: int, frame_width: int, frame_height: int) -> float:
        """
        Estimate distance to object based on its apparent size.
        This is a simplified heuristic - real systems use proper depth sensors.
        """
        # Assume objects closer to camera appear larger
        # This is a rough approximation
        apparent_size = width * height
        max_size = frame_width * frame_height * 0.1  # 10% of frame
        
        # Normalize distance (0 = very close, 1 = far away)
        distance_ratio = 1.0 - (apparent_size / max_size)
        distance_ratio = max(0.0, min(1.0, distance_ratio))
        
        # Convert to pixel distance (assuming 640x480 frame)
        return distance_ratio * 300  # Max distance of 300 pixels
    
    def _classify_obstacle(self, width: int, height: int, distance: float, angle: float) -> str:
        """Classify obstacle type based on its properties"""
        aspect_ratio = width / height if height > 0 else 1.0
        
        if distance < 50:
            return "wall"  # Very close, likely a wall
        elif aspect_ratio > 2.0:
            return "wall"  # Wide and low, likely a wall
        elif width > 100 and height > 100:
            return "object"  # Large object
        elif abs(angle) < 30:
            return "person"  # In front, could be a person
        else:
            return "unknown"
    
    def check_safety(self, obstacles: List[ObstacleInfo]) -> SafetyLevel:
        """
        Check safety level based on detected obstacles.
        Returns the appropriate safety level for current conditions.
        """
        if not obstacles:
            return SafetyLevel.SAFE
        
        # Find the closest obstacle
        closest_obstacle = min(obstacles, key=lambda o: o.distance)
        
        if closest_obstacle.distance < self.danger_distance:
            self.logger.warning(f"DANGER: Obstacle at {closest_obstacle.distance:.1f} pixels")
            return SafetyLevel.DANGER
        elif closest_obstacle.distance < self.caution_distance:
            self.logger.info(f"CAUTION: Obstacle at {closest_obstacle.distance:.1f} pixels")
            return SafetyLevel.CAUTION
        else:
            return SafetyLevel.SAFE
    
    def should_emergency_stop(self, obstacles: List[ObstacleInfo]) -> bool:
        """Determine if emergency stop is needed"""
        if not self.config.get('enable_emergency_stop', True):
            return False
        
        # Check for very close obstacles
        for obstacle in obstacles:
            if obstacle.distance < self.danger_distance * 0.5:  # Very close
                return True
        
        return False
    
    def get_safe_speed(self, base_speed: float, safety_level: SafetyLevel) -> float:
        """Get speed adjusted for safety level"""
        if not self.config.get('enable_speed_limiting', True):
            return base_speed
        
        if safety_level == SafetyLevel.DANGER:
            return 0.0  # Stop
        elif safety_level == SafetyLevel.CAUTION:
            return base_speed * 0.3  # Slow down
        else:
            return base_speed
    
    def get_safe_direction(self, target_angle: float, obstacles: List[ObstacleInfo]) -> float:
        """
        Calculate safe direction to avoid obstacles.
        Returns adjusted angle that avoids obstacles.
        """
        if not obstacles:
            return target_angle
        
        # Find obstacles in the target direction
        target_obstacles = [o for o in obstacles if abs(o.angle - target_angle) < 30]
        
        if not target_obstacles:
            return target_angle
        
        # Find the best alternative direction
        best_angle = target_angle
        best_score = float('inf')
        
        # Check angles in 10-degree increments
        for angle_offset in range(-90, 91, 10):
            test_angle = target_angle + angle_offset
            
            # Check if this direction is clear
            blocking_obstacles = [o for o in obstacles if abs(o.angle - test_angle) < 20]
            
            if not blocking_obstacles:
                # Calculate score based on how far from target direction
                score = abs(angle_offset)
                if score < best_score:
                    best_score = score
                    best_angle = test_angle
        
        return best_angle

# ------------------------- Enhanced Camera Integration -------------------------

class BrunoCamera:
    """
    Enhanced camera interface with safety features and error recovery.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.device_id = config.get('device_id', 0)
        self.width = config.get('width', 640)
        self.height = config.get('height', 480)
        self.fps = config.get('fps', 30)
        self.flip_horizontal = config.get('flip_horizontal', False)
        
        self.cap = None
        self.ffmpeg_proc = None
        self.frame_size = self.width * self.height * 3
        
        # Error recovery
        self.consecutive_failures = 0
        self.max_failures = config.get('max_camera_failures', 10)
        self.recovery_attempts = 0
        self.max_recovery_attempts = config.get('max_recovery_attempts', 3)
        
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for camera operations"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start camera capture with error recovery"""
        for attempt in range(self.max_recovery_attempts):
            try:
                if isinstance(self.device_id, str) and self.device_id.startswith('http'):
                    success = self._start_ffmpeg_capture()
                else:
                    success = self._start_opencv_capture()
                
                if success:
                    self.consecutive_failures = 0
                    self.recovery_attempts = 0
                    return True
                    
            except Exception as e:
                self.logger.error(f"Camera start attempt {attempt + 1} failed: {e}")
                self.recovery_attempts += 1
                time.sleep(1.0)
        
        self.logger.error("All camera start attempts failed")
        return False
    
    def _start_opencv_capture(self):
        """Start OpenCV camera capture"""
        try:
            import cv2
            self.cap = cv2.VideoCapture(self.device_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            if not self.cap.isOpened():
                raise Exception(f"Could not open camera device {self.device_id}")
            
            self.logger.info(f"Started OpenCV camera capture: {self.device_id}")
            return True
            
        except ImportError:
            self.logger.error("OpenCV not available for local camera capture")
            return False
    
    def _start_ffmpeg_capture(self):
        """Start FFmpeg capture for network cameras"""
        try:
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-f", "mjpeg",
                "-i", self.device_id,
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-vf", f"scale={self.width}:{self.height}",
                "-"
            ]
            
            self.ffmpeg_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, bufsize=self.frame_size*2
            )
            
            self.logger.info(f"Started FFmpeg capture from: {self.device_id}")
            return True
            
        except FileNotFoundError:
            self.logger.error("FFmpeg not found. Install with: sudo apt-get install -y ffmpeg")
            return False
    
    def read(self) -> Optional[np.ndarray]:
        """Read a frame from the camera with error handling"""
        try:
            if self.cap is not None:
                frame = self._read_opencv_frame()
            elif self.ffmpeg_proc is not None:
                frame = self._read_ffmpeg_frame()
            else:
                return None
            
            if frame is not None:
                self.consecutive_failures = 0
                return frame
            else:
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.max_failures:
                    self.logger.error("Too many consecutive camera failures")
                    return None
                return None
                
        except Exception as e:
            self.logger.error(f"Camera read error: {e}")
            self.consecutive_failures += 1
            return None
    
    def _read_opencv_frame(self) -> Optional[np.ndarray]:
        """Read frame from OpenCV camera"""
        import cv2
        ret, frame = self.cap.read()
        if ret:
            if self.flip_horizontal:
                frame = cv2.flip(frame, 1)
            return frame
        return None
    
    def _read_ffmpeg_frame(self) -> Optional[np.ndarray]:
        """Read frame from FFmpeg pipe"""
        if self.ffmpeg_proc is None or self.ffmpeg_proc.stdout is None:
            return None
            
        raw = self.ffmpeg_proc.stdout.read(self.frame_size)
        if len(raw) != self.frame_size:
            return None
            
        frame = np.frombuffer(raw, np.uint8).reshape((self.height, self.width, 3))
        
        if self.flip_horizontal:
            import cv2
            frame = cv2.flip(frame, 1)
            
        return frame
    
    def stop(self):
        """Stop camera capture"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        if self.ffmpeg_proc is not None:
            self.ffmpeg_proc.terminate()
            try:
                self.ffmpeg_proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.ffmpeg_proc.kill()
            self.ffmpeg_proc = None

# ------------------------- Enhanced Robot Control Integration -------------------------

class BrunoRobotController:
    """
    Enhanced robot controller with safety features and collision avoidance.
    """
    def __init__(self, config: Dict, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.setup_logging()
        
        # Initialize robot controllers
        self.movement_controller = None
        self.head_controller = None
        
        if BRUNO_MODULES_AVAILABLE and not dry_run:
            try:
                self.movement_controller = MovementController(config.get('movement_control', {}))
                self.head_controller = HeadController(config.get('head_control', {}))
                self.logger.info("Bruno robot controllers initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Bruno controllers: {e}")
        
        # Movement parameters
        self.forward_speed = config.get('movement_control', {}).get('max_speed', 40) / 100.0
        self.turn_speed = self.forward_speed * 0.8
        self.approach_speed = self.forward_speed * 0.6
        
        # Safety and collision avoidance
        self.collision_avoidance = CollisionAvoidance(config)
        self.emergency_stop_active = False
        self.last_safety_check = 0.0
        self.safety_check_interval = 0.1  # seconds
        
        # Movement state tracking
        self.current_speed = 0.0
        self.current_direction = 0.0
        self.last_movement_time = 0.0
        
    def setup_logging(self):
        """Setup logging for robot control"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def emergency_stop(self):
        """Emergency stop all robot movement"""
        self.logger.warning("EMERGENCY STOP ACTIVATED")
        self.emergency_stop_active = True
        self.current_speed = 0.0
        self.current_direction = 0.0
        
        if self.dry_run:
            self.logger.info("[EMERGENCY_STOP]")
            return
            
        if self.movement_controller:
            self.movement_controller.stop()
    
    def clear_emergency_stop(self):
        """Clear emergency stop condition"""
        self.emergency_stop_active = False
        self.logger.info("Emergency stop cleared")
    
    def drive(self, left: float, right: float, duration: float = None):
        """Drive robot with differential steering and safety checks"""
        if self.emergency_stop_active:
            self.logger.warning("Movement blocked by emergency stop")
            return
        
        left = float(np.clip(left, -1.0, 1.0))
        right = float(np.clip(right, -1.0, 1.0))
        
        # Update movement state
        self.current_speed = (abs(left) + abs(right)) / 2
        self.current_direction = np.arctan2(right - left, left + right) * 180 / np.pi
        
        self.logger.debug(f"DRIVE: left={left:.2f} right={right:.2f} speed={self.current_speed:.2f} direction={self.current_direction:.1f}°")
        
        if self.dry_run:
            self.logger.info(f"[DRIVE] left={left:.2f} right={right:.2f} duration={duration}")
            if duration:
                time.sleep(duration)
            return
        
        if self.movement_controller:
            # Convert differential to mecanum commands
            if abs(left - right) < 0.1:
                # Forward/backward
                speed = (left + right) / 2 * self.forward_speed * 100
                if speed > 0:
                    self.movement_controller.move_forward(speed)
                else:
                    self.movement_controller.move_backward(abs(speed))
            else:
                # Turning
                turn_speed = (right - left) * self.turn_speed * 100
                if turn_speed > 0:
                    self.movement_controller.turn('RIGHT', abs(turn_speed))
                else:
                    self.movement_controller.turn('LEFT', abs(turn_speed))
            
            self.last_movement_time = time.time()
            
            if duration:
                time.sleep(duration)
                self.stop()
    
    def stop(self):
        """Stop robot movement"""
        self.current_speed = 0.0
        self.current_direction = 0.0
        
        if self.dry_run:
            self.logger.info("[STOP]")
            return
            
        if self.movement_controller:
            self.movement_controller.stop()
    
    def rotate(self, speed: float, duration: float = None):
        """Rotate robot in place with safety checks"""
        if self.emergency_stop_active:
            return
        
        self.drive(-speed, speed, duration)
    
    def approach_target(self, target_center_x: float, frame_width: int, duration: float = 0.1, obstacles: List[ObstacleInfo] = None):
        """Approach target with collision avoidance"""
        if self.emergency_stop_active:
            return
        
        center_x = frame_width / 2
        error = target_center_x - center_x
        turn_gain = 0.003
        base_turn = np.clip(error * turn_gain, -0.4, 0.4)
        
        self.logger.debug(f"APPROACH: target={target_center_x:.1f}px center={center_x:.1f}px error={error:.1f}px base_turn={base_turn:.3f}")
        
        # Apply collision avoidance
        if obstacles:
            safety_level = self.collision_avoidance.check_safety(obstacles)
            self.logger.debug(f"Safety level: {safety_level.value}")
            
            if safety_level == SafetyLevel.DANGER:
                self.logger.warning("DANGER level detected during approach - emergency stop")
                self.emergency_stop()
                return
            
            # Adjust direction to avoid obstacles
            safe_turn = self.collision_avoidance.get_safe_direction(base_turn, obstacles)
            
            # Adjust speed based on safety level
            safe_speed = self.collision_avoidance.get_safe_speed(self.approach_speed, safety_level)
            
            self.logger.debug(f"Collision avoidance: base_turn={base_turn:.3f} safe_turn={safe_turn:.3f} safe_speed={safe_speed:.3f}")
            
            if safe_speed > 0:
                self.drive(safe_speed, safe_speed, duration)
                self.rotate(safe_turn, duration=0.05)
            else:
                self.logger.warning("Safe speed is 0 - stopping")
                self.stop()
        else:
            # No obstacles detected, proceed normally
            self.logger.debug("No obstacles detected - proceeding normally")
            self.drive(self.approach_speed, self.approach_speed, duration)
            self.rotate(base_turn, duration=0.05)
    
    def head_nod(self, pattern: str = "acknowledgment"):
        """Make head nod gesture"""
        if self.dry_run:
            self.logger.info(f"[HEAD] nod {pattern}")
            return
            
        if self.head_controller:
            if pattern == "excited":
                self.head_controller.nod_excited()
            elif pattern == "acknowledgment":
                self.head_controller.nod_yes()
            elif pattern == "scanning":
                self.head_controller.scan_horizontally()

# ------------------------- Enhanced Detection via OpenAI Vision -------------------------

@dataclass
class Detection:
    bottle_present: bool = False
    bottle_bbox: Optional[Tuple[int,int,int,int]] = None  # x,y,w,h
    bottle_conf: float = 0.0
    bin_present: bool = False
    bin_bbox: Optional[Tuple[int,int,int,int]] = None
    bin_conf: float = 0.0
    obstacles: List[ObstacleInfo] = None

    def __post_init__(self):
        if self.obstacles is None:
            self.obstacles = []


class GPTDetector:
    """
    Enhanced OpenAI GPT Vision detector with obstacle detection and safety features.
    """
    def __init__(self, config: Dict, model: str = "gpt-4o-mini", temperature: float = 0.0):
        self.config = config
        self.model = model
        self.temperature = temperature
        self.setup_logging()
        
        # Initialize OpenAI client if available
        if OPENAI_AVAILABLE:
            try:
                self.client = OpenAI()
                self.api_available = True
                self.logger.info("OpenAI API client initialized")
            except Exception as e:
                self.logger.warning(f"OpenAI API not available: {e}")
                self.api_available = False
        else:
            self.api_available = False
        
        # Fallback to local detector
        if not self.api_available and BRUNO_MODULES_AVAILABLE:
            try:
                self.local_detector = BottleDetector(config.get('detection', {}))
                self.logger.info("Local bottle detector initialized as fallback")
            except Exception as e:
                self.logger.error(f"Failed to initialize local detector: {e}")
                self.local_detector = None
        
        # Collision avoidance
        self.collision_avoidance = CollisionAvoidance(config)
        
        self.prompt = (
            "You are a detection model for a robot. Given an image from the robot's front camera, "
            "find: (1) any **plastic water bottles** (various colors and sizes), "
            "(2) any **garbage bins or trash containers** (any color), "
            "(3) any **obstacles or dangerous objects** that the robot should avoid. "
            "Return STRICT JSON ONLY, no extra text, matching this schema:\n\n"
            "{\n"
            '  "bottle": {"present": true|false, "bbox": [x,y,w,h] or null, "confidence": 0..1},\n'
            '  "bin":    {"present": true|false, "bbox": [x,y,w,h] or null, "confidence": 0..1},\n'
            '  "obstacles": [{"type": "wall|object|person|unknown", "bbox": [x,y,w,h], "confidence": 0..1}],\n'
            '  "frame":  {"width": Wpx, "height": Hpx}\n'
            "}\n\n"
            "Rules: If unsure, set present=false and bbox=null. "
            "If multiple candidates, choose the best one. Coordinates must be integer pixel values. "
            "Include all obstacles that could block the robot's path."
        )
    
    def setup_logging(self):
        """Setup logging for detector"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def encode_jpeg_from_bgr(frame_bgr: np.ndarray, max_width: int = 480, quality: int = 70) -> bytes:
        """Convert BGR frame to JPEG bytes"""
        H, W = frame_bgr.shape[:2]
        
        # Downscale to reduce bandwidth
        if W > max_width:
            scale = max_width / W
            new_w = max_width
            new_h = int(H * scale)
            # Convert BGR -> RGB for Pillow
            rgb = frame_bgr[:, :, ::-1]
            img = Image.fromarray(rgb).resize((new_w, new_h))
        else:
            rgb = frame_bgr[:, :, ::-1]
            img = Image.fromarray(rgb)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()
    
    def detect(self, frame_bgr: np.ndarray) -> Optional[Detection]:
        """Detect bottles, bins, and obstacles in frame"""
        self.logger.debug("Starting detection process...")
        
        # Always detect obstacles locally for safety
        obstacles = self.collision_avoidance.detect_obstacles(frame_bgr)
        self.logger.debug(f"Local obstacle detection: {len(obstacles)} obstacles found")
        
        if self.api_available:
            self.logger.debug("Using GPT Vision API for detection")
            detection = self._detect_with_gpt(frame_bgr)
        elif hasattr(self, 'local_detector') and self.local_detector:
            self.logger.debug("Using local OpenCV detector (GPT API unavailable)")
            detection = self._detect_with_local(frame_bgr)
        else:
            self.logger.warning("No detection method available")
            detection = Detection()
        
        # Add obstacles to detection result
        if detection:
            detection.obstacles = obstacles
            self.logger.debug(f"Detection complete - Bottle: {detection.bottle_present}, Bin: {detection.bin_present}, Obstacles: {len(obstacles)}")
        else:
            self.logger.warning("Detection failed - no results")
        
        return detection
    
    def _detect_with_gpt(self, frame_bgr: np.ndarray) -> Optional[Detection]:
        """Detect using OpenAI GPT Vision"""
        jpg_bytes = self.encode_jpeg_from_bgr(frame_bgr)
        b64 = base64.b64encode(jpg_bytes).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{b64}"

        body = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": self.prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ]

        try:
            resp = self.client.responses.create(
                model=self.model,
                input=body,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            
            text = getattr(resp, "output_text", None)
            if not text:
                if hasattr(resp, "output") and isinstance(resp.output, list) and resp.output:
                    text = str(resp.output[0])
            if not text:
                return None

            payload = json.loads(text)
            return self._payload_to_detection(payload)

        except (APIError, json.JSONDecodeError) as e:
            self.logger.warning(f"GPT detection failed: {e}")
            return None
    
    def _detect_with_local(self, frame_bgr: np.ndarray) -> Optional[Detection]:
        """Detect using local OpenCV-based detector"""
        try:
            # Use the existing Bruno bottle detector
            bottles = self.local_detector.detect_bottles(frame_bgr)
            
            det = Detection()
            if bottles:
                # Get the best bottle detection
                best_bottle = max(bottles, key=lambda x: x.get('confidence', 0))
                det.bottle_present = True
                det.bottle_bbox = best_bottle.get('bbox')
                det.bottle_conf = best_bottle.get('confidence', 0.0)
            
            # Note: Local detector doesn't detect bins, so bin detection will be False
            return det
            
        except Exception as e:
            self.logger.error(f"Local detection failed: {e}")
            return None
    
    @staticmethod
    def _payload_to_detection(payload: dict) -> Detection:
        """Convert API payload to Detection object"""
        det = Detection()
        
        if "bottle" in payload:
            b = payload["bottle"]
            det.bottle_present = bool(b.get("present", False))
            bb = b.get("bbox")
            if det.bottle_present and isinstance(bb, (list, tuple)) and len(bb) == 4:
                det.bottle_bbox = tuple(int(v) for v in bb)
            det.bottle_conf = float(b.get("confidence", 0.0))

        if "bin" in payload:
            g = payload["bin"]
            det.bin_present = bool(g.get("present", False))
            bb = g.get("bbox")
            if det.bin_present and isinstance(bb, (list, tuple)) and len(bb) == 4:
                det.bin_bbox = tuple(int(v) for v in bb)
            det.bin_conf = float(g.get("confidence", 0.0))

        # Parse obstacles from GPT response
        if "obstacles" in payload:
            obstacles = []
            for obs in payload["obstacles"]:
                if isinstance(obs, dict) and "bbox" in obs:
                    bb = obs.get("bbox")
                    if isinstance(bb, (list, tuple)) and len(bb) == 4:
                        obstacle = ObstacleInfo(
                            distance=300,  # Default distance
                            angle=0,       # Will be calculated
                            confidence=float(obs.get("confidence", 0.5)),
                            type=obs.get("type", "unknown"),
                            bbox=tuple(int(v) for v in bb)
                        )
                        obstacles.append(obstacle)
            det.obstacles = obstacles

        return det

# ------------------------- Enhanced Main Behavior Loop -------------------------

class States:
    SEARCH_BOTTLE = "search_bottle"
    APPROACH_BOTTLE = "approach_bottle"
    GRAB_BOTTLE = "grab_bottle"
    SEARCH_BIN = "search_bin"
    APPROACH_BIN = "approach_bin"
    DROP_BOTTLE = "drop_bottle"
    DONE = "done"
    EMERGENCY_STOP = "emergency_stop"


def main():
    # Setup comprehensive logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('bruno_gpt.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description="Bruno GPT Vision Bottle Detection System with Collision Avoidance")
    parser.add_argument("--config", default="config/bruno_config.json", 
                       help="Path to Bruno configuration file")
    parser.add_argument("--camera-url", help="Override camera URL from config")
    parser.add_argument("--fps", type=int, default=5, help="Camera FPS")
    parser.add_argument("--gpt-interval", type=float, default=1.2, 
                       help="Seconds between API calls")
    parser.add_argument("--model", default="gpt-4o-mini", 
                       help="OpenAI model for vision")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Print actions instead of actuating")
    parser.add_argument("--save-debug", action="store_true", 
                       help="Save debug images and results")
    parser.add_argument("--enable-safety", action="store_true", default=True,
                       help="Enable collision avoidance and safety features")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Set logging level")
    args = parser.parse_args()
    
    # Set log level from command line
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    logger.info("=" * 60)
    logger.info("BRUNO GPT VISION SYSTEM STARTING")
    logger.info("=" * 60)
    logger.info(f"Arguments: {vars(args)}")

    # Load configuration
    logger.info("Loading configuration...")
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from: {args.config}")
        logger.debug(f"Camera config: {config.get('camera', {})}")
        logger.debug(f"Movement config: {config.get('movement_control', {})}")
        logger.debug(f"Safety config: {config.get('collision_avoidance', {})}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid configuration file: {e}")
        sys.exit(1)

    # Override camera URL if specified
    if args.camera_url:
        config['camera']['device_id'] = args.camera_url
        logger.info(f"Camera URL overridden: {args.camera_url}")

    # Check OpenAI API key
    if not os.environ.get("OPENAI_API_KEY") and not args.dry_run:
        logger.warning("OPENAI_API_KEY not set. Will use local detection only.")
        logger.warning("Set with: export OPENAI_API_KEY=sk-...")
    else:
        logger.info("OpenAI API key found")

    # Initialize components
    logger.info("Initializing system components...")
    camera = BrunoCamera(config['camera'])
    robot = BrunoRobotController(config, dry_run=args.dry_run)
    detector = GPTDetector(config, model=args.model)
    logger.info("All components initialized successfully")

    # Start camera
    logger.info("Starting camera...")
    if not camera.start():
        logger.error("Failed to start camera")
        sys.exit(1)
    logger.info("Camera started successfully")

    # Initialize robot
    logger.info("Initializing robot...")
    robot.stop()
    if hasattr(robot, 'head_controller') and robot.head_controller:
        robot.head_controller.center_head()
        logger.info("Robot head centered")
    logger.info("Robot initialization complete")

    # Behavior parameters
    bottle_close_h = 220  # "close" if bbox height >= this
    bin_close_h = 200
    timeout_each = 180  # seconds
    last_api_time = 0.0
    emergency_stop_timeout = 10.0  # seconds to wait before clearing emergency stop

    state = States.SEARCH_BOTTLE
    phase_start = time.time()
    emergency_stop_start = 0.0

    logger.info("=" * 60)
    logger.info("STARTING BEHAVIOR LOOP")
    logger.info("=" * 60)
    logger.info(f"Initial state: {state}")
    logger.info(f"Bottle close height: {bottle_close_h} pixels")
    logger.info(f"Bin close height: {bin_close_h} pixels")
    logger.info(f"Detection interval: {args.gpt_interval} seconds")
    logger.info(f"Safety enabled: {args.enable_safety}")
    logger.info(f"Dry run mode: {args.dry_run}")
    logger.info("Press Ctrl+C to stop.")
    logger.info("=" * 60)
    
    try:
        while True:
            frame = camera.read()
            if frame is None:
                logger.error("No frame from camera. Exiting.")
                break

            now = time.time()
            
            # Handle emergency stop state
            if state == States.EMERGENCY_STOP:
                time_in_emergency = now - emergency_stop_start
                logger.debug(f"Emergency stop active for {time_in_emergency:.1f}s")
                
                if time_in_emergency > emergency_stop_timeout:
                    robot.clear_emergency_stop()
                    state = States.SEARCH_BOTTLE
                    phase_start = time.time()
                    logger.info("EMERGENCY STOP CLEARED - Resuming normal operation")
                else:
                    robot.stop()
                    continue

            if now - last_api_time < args.gpt_interval:
                # Keep moving while waiting for next detection
                if state == States.SEARCH_BOTTLE:
                    robot.rotate(0.18, duration=0.05)
                    logger.debug("Searching for bottles - rotating right")
                elif state == States.SEARCH_BIN:
                    robot.rotate(-0.18, duration=0.05)
                    logger.debug("Searching for bins - rotating left")
                continue

            # Run detection
            logger.debug("Running detection...")
            det = detector.detect(frame)
            last_api_time = time.time()
            
            if det:
                logger.debug(f"Detection results - Bottle: {det.bottle_present} (conf: {det.bottle_conf:.2f}), "
                           f"Bin: {det.bin_present} (conf: {det.bin_conf:.2f}), "
                           f"Obstacles: {len(det.obstacles)}")
            else:
                logger.debug("No detection results")

            # Check for emergency stop conditions
            if det and det.obstacles and args.enable_safety:
                if robot.collision_avoidance.should_emergency_stop(det.obstacles):
                    robot.emergency_stop()
                    state = States.EMERGENCY_STOP
                    emergency_stop_start = time.time()
                    logger.warning("EMERGENCY STOP TRIGGERED - Obstacle detected!")
                    logger.warning(f"Obstacles detected: {len(det.obstacles)}")
                    for i, obs in enumerate(det.obstacles):
                        logger.warning(f"  Obstacle {i+1}: {obs.type} at {obs.distance:.1f}px, angle {obs.angle:.1f}°")
                    continue

            # Save debug info
            if args.save_debug and det is not None:
                logger.debug("Saving debug information...")
                jpg = detector.encode_jpeg_from_bgr(frame)
                with open("last_sent.jpg", "wb") as f:
                    f.write(jpg)
                with open("last_result.json", "w") as f:
                    payload = {
                        "bottle_present": det.bottle_present,
                        "bottle_bbox": det.bottle_bbox,
                        "bottle_conf": det.bottle_conf,
                        "bin_present": det.bin_present,
                        "bin_bbox": det.bin_bbox,
                        "bin_conf": det.bin_conf,
                        "obstacles_count": len(det.obstacles) if det.obstacles else 0,
                    }
                    json.dump(payload, f, indent=2)
                logger.debug("Debug information saved")

            H, W = frame.shape[:2]
            cx_img = W / 2

            # State machine with enhanced safety
            if state == States.SEARCH_BOTTLE:
                if det and det.bottle_present and det.bottle_bbox:
                    x, y, w, h = det.bottle_bbox
                    target_cx = x + w/2
                    
                    logger.info(f"BOTTLE DETECTED - Position: ({x},{y}) Size: {w}x{h} Confidence: {det.bottle_conf:.2f}")
                    logger.info(f"Target center: {target_cx:.1f}px, Frame center: {W/2:.1f}px")
                    
                    # Approach bottle with collision avoidance
                    logger.info("Approaching bottle with collision avoidance...")
                    robot.approach_target(target_cx, W, duration=0.1, obstacles=det.obstacles)
                    
                    if h >= bottle_close_h:
                        robot.stop()
                        robot.head_nod("excited")
                        state = States.GRAB_BOTTLE
                        phase_start = time.time()
                        logger.info("STATE CHANGE: SEARCH_BOTTLE -> GRAB_BOTTLE")
                        logger.info(f"Bottle is close enough (height: {h} >= {bottle_close_h})")
                    else:
                        logger.debug(f"Bottle not close enough yet (height: {h} < {bottle_close_h})")
                else:
                    # Keep scanning
                    robot.rotate(0.2, duration=0.12)
                    logger.debug("No bottle detected - continuing scan")

                if time.time() - phase_start > timeout_each:
                    logger.warning("Timed out searching bottle; continuing to scan.")
                    phase_start = time.time()

            elif state == States.GRAB_BOTTLE:
                logger.info("GRAB_BOTTLE state - executing pickup sequence")
                
                # Check for obstacles before moving forward
                if det and det.obstacles:
                    safety_level = robot.collision_avoidance.check_safety(det.obstacles)
                    logger.info(f"Safety check before grab: {safety_level.value}")
                    if safety_level == SafetyLevel.DANGER:
                        logger.warning("Obstacles detected during grab - stopping and returning to search")
                        robot.stop()
                        state = States.SEARCH_BOTTLE
                        continue
                
                # Simulate bottle pickup (actual implementation would use arm controller)
                logger.info("Moving forward to grab bottle...")
                robot.drive(0.18, 0.18, duration=0.8)
                robot.stop()
                logger.info("ACTION: Picking up bottle")
                time.sleep(1.0)  # Simulate pickup time
                state = States.SEARCH_BIN
                phase_start = time.time()
                logger.info("STATE CHANGE: GRAB_BOTTLE -> SEARCH_BIN")
                logger.info("Bottle pickup complete - now searching for bin")

            elif state == States.SEARCH_BIN:
                if det and det.bin_present and det.bin_bbox:
                    x, y, w, h = det.bin_bbox
                    target_cx = x + w/2
                    
                    logger.info(f"BIN DETECTED - Position: ({x},{y}) Size: {w}x{h} Confidence: {det.bin_conf:.2f}")
                    logger.info(f"Target center: {target_cx:.1f}px, Frame center: {W/2:.1f}px")
                    
                    # Approach bin with collision avoidance
                    logger.info("Approaching bin with collision avoidance...")
                    robot.approach_target(target_cx, W, duration=0.1, obstacles=det.obstacles)
                    
                    if h >= bin_close_h:
                        robot.stop()
                        state = States.DROP_BOTTLE
                        phase_start = time.time()
                        logger.info("STATE CHANGE: SEARCH_BIN -> DROP_BOTTLE")
                        logger.info(f"Bin is close enough (height: {h} >= {bin_close_h})")
                    else:
                        logger.debug(f"Bin not close enough yet (height: {h} < {bin_close_h})")
                else:
                    robot.rotate(-0.18, duration=0.12)
                    logger.debug("No bin detected - continuing scan")

                if time.time() - phase_start > timeout_each:
                    logger.warning("Timed out searching bin; continuing to scan.")
                    phase_start = time.time()

            elif state == States.DROP_BOTTLE:
                logger.info("DROP_BOTTLE state - executing drop sequence")
                
                # Check for obstacles before moving forward
                if det and det.obstacles:
                    safety_level = robot.collision_avoidance.check_safety(det.obstacles)
                    logger.info(f"Safety check before drop: {safety_level.value}")
                    if safety_level == SafetyLevel.DANGER:
                        logger.warning("Obstacles detected during drop - stopping and returning to search")
                        robot.stop()
                        state = States.SEARCH_BIN
                        continue
                
                # Simulate bottle drop
                logger.info("Moving forward to drop bottle...")
                robot.drive(0.15, 0.15, duration=0.8)
                robot.stop()
                logger.info("ACTION: Dropping bottle in bin")
                time.sleep(1.0)  # Simulate drop time
                logger.info("Backing away from bin...")
                robot.drive(-0.25, -0.25, duration=0.9)
                robot.stop()
                robot.head_nod("acknowledgment")
                state = States.DONE
                logger.info("STATE CHANGE: DROP_BOTTLE -> DONE")
                logger.info("Bottle drop complete - task finished")

            elif state == States.DONE:
                robot.stop()
                logger.info("=" * 60)
                logger.info("TASK COMPLETE!")
                logger.info("=" * 60)
                logger.info("Bruno successfully completed the bottle pickup and disposal task")
                break

    except KeyboardInterrupt:
        logger.info("\n" + "=" * 60)
        logger.info("INTERRUPTED BY USER")
        logger.info("=" * 60)
        logger.info("Stopping Bruno GPT Vision system...")
    finally:
        logger.info("Cleaning up...")
        robot.stop()
        camera.stop()
        logger.info("System shutdown complete")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
