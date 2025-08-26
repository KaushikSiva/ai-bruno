#!/usr/bin/env python3
"""
Bruno GPT Vision Bottle Detection and Pickup System

Enhanced bottle detection using OpenAI's GPT Vision API integrated with the Bruno robot platform.
This module provides AI-powered bottle detection and automated pickup behavior.

Key Features:
- Uses OpenAI GPT Vision for intelligent bottle and bin detection
- Integrates with existing Bruno robot control systems
- Supports both local camera and network camera streams
- Configurable detection parameters and robot behavior
- Fallback to local OpenCV detection if API is unavailable

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
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from pathlib import Path

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

# ------------------------- Camera Integration -------------------------

class BrunoCamera:
    """
    Camera interface that works with Bruno's camera configuration.
    Supports both local V4L2 devices and network camera streams.
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
        
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for camera operations"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start camera capture"""
        try:
            if isinstance(self.device_id, str) and self.device_id.startswith('http'):
                # Network camera - use FFmpeg
                return self._start_ffmpeg_capture()
            else:
                # Local camera - use OpenCV
                return self._start_opencv_capture()
        except Exception as e:
            self.logger.error(f"Failed to start camera: {e}")
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
        """Read a frame from the camera"""
        if self.cap is not None:
            return self._read_opencv_frame()
        elif self.ffmpeg_proc is not None:
            return self._read_ffmpeg_frame()
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

# ------------------------- Robot Control Integration -------------------------

class BrunoRobotController:
    """
    Integrated robot controller using Bruno's existing control modules.
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
        
    def setup_logging(self):
        """Setup logging for robot control"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def drive(self, left: float, right: float, duration: float = None):
        """Drive robot with differential steering"""
        left = float(np.clip(left, -1.0, 1.0))
        right = float(np.clip(right, -1.0, 1.0))
        
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
            
            if duration:
                time.sleep(duration)
                self.stop()
    
    def stop(self):
        """Stop robot movement"""
        if self.dry_run:
            self.logger.info("[STOP]")
            return
            
        if self.movement_controller:
            self.movement_controller.stop()
    
    def rotate(self, speed: float, duration: float = None):
        """Rotate robot in place"""
        self.drive(-speed, speed, duration)
    
    def approach_target(self, target_center_x: float, frame_width: int, duration: float = 0.1):
        """Approach target with proportional steering"""
        center_x = frame_width / 2
        error = target_center_x - center_x
        turn_gain = 0.003
        turn = np.clip(error * turn_gain, -0.4, 0.4)
        
        self.drive(self.approach_speed, self.approach_speed, duration)
        self.rotate(turn, duration=0.05)
    
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

# ------------------------- Detection via OpenAI Vision -------------------------

@dataclass
class Detection:
    bottle_present: bool = False
    bottle_bbox: Optional[Tuple[int,int,int,int]] = None  # x,y,w,h
    bottle_conf: float = 0.0
    bin_present: bool = False
    bin_bbox: Optional[Tuple[int,int,int,int]] = None
    bin_conf: float = 0.0


class GPTDetector:
    """
    OpenAI GPT Vision detector for bottles and bins.
    Falls back to local OpenCV detection if API is unavailable.
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
        
        self.prompt = (
            "You are a detection model for a robot. Given an image from the robot's front camera, "
            "find: (1) any **plastic water bottles** (various colors and sizes), "
            "(2) any **garbage bins or trash containers** (any color). "
            "Return STRICT JSON ONLY, no extra text, matching this schema:\n\n"
            "{\n"
            '  "bottle": {"present": true|false, "bbox": [x,y,w,h] or null, "confidence": 0..1},\n'
            '  "bin":    {"present": true|false, "bbox": [x,y,w,h] or null, "confidence": 0..1},\n'
            '  "frame":  {"width": Wpx, "height": Hpx}\n'
            "}\n\n"
            "Rules: If unsure, set present=false and bbox=null. "
            "If multiple candidates, choose the best one. Coordinates must be integer pixel values."
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
        """Detect bottles and bins in frame"""
        if self.api_available:
            return self._detect_with_gpt(frame_bgr)
        elif hasattr(self, 'local_detector') and self.local_detector:
            return self._detect_with_local(frame_bgr)
        else:
            self.logger.warning("No detection method available")
            return None
    
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

        return det

# ------------------------- Main Behavior Loop -------------------------

class States:
    SEARCH_BOTTLE = "search_bottle"
    APPROACH_BOTTLE = "approach_bottle"
    GRAB_BOTTLE = "grab_bottle"
    SEARCH_BIN = "search_bin"
    APPROACH_BIN = "approach_bin"
    DROP_BOTTLE = "drop_bottle"
    DONE = "done"


def main():
    parser = argparse.ArgumentParser(description="Bruno GPT Vision Bottle Detection System")
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
    args = parser.parse_args()

    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Configuration file not found: {args.config}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Invalid configuration file: {e}")
        sys.exit(1)

    # Override camera URL if specified
    if args.camera_url:
        config['camera']['device_id'] = args.camera_url

    # Check OpenAI API key
    if not os.environ.get("OPENAI_API_KEY") and not args.dry_run:
        print("Warning: OPENAI_API_KEY not set. Will use local detection only.")
        print("Set with: export OPENAI_API_KEY=sk-...")

    # Initialize components
    camera = BrunoCamera(config['camera'])
    robot = BrunoRobotController(config, dry_run=args.dry_run)
    detector = GPTDetector(config, model=args.model)

    # Start camera
    if not camera.start():
        print("Failed to start camera")
        sys.exit(1)

    # Initialize robot
    robot.stop()
    if hasattr(robot, 'head_controller') and robot.head_controller:
        robot.head_controller.center_head()

    # Behavior parameters
    bottle_close_h = 220  # "close" if bbox height >= this
    bin_close_h = 200
    timeout_each = 180  # seconds
    last_api_time = 0.0

    state = States.SEARCH_BOTTLE
    phase_start = time.time()

    print("[INFO] Starting Bruno GPT Vision behavior loop. Press Ctrl+C to stop.")
    
    try:
        while True:
            frame = camera.read()
            if frame is None:
                print("No frame from camera. Exiting.")
                break

            now = time.time()
            if now - last_api_time < args.gpt_interval:
                # Keep moving while waiting for next detection
                if state == States.SEARCH_BOTTLE:
                    robot.rotate(0.18, duration=0.05)
                elif state == States.SEARCH_BIN:
                    robot.rotate(-0.18, duration=0.05)
                continue

            # Run detection
            det = detector.detect(frame)
            last_api_time = time.time()

            # Save debug info
            if args.save_debug and det is not None:
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
                    }
                    json.dump(payload, f, indent=2)

            H, W = frame.shape[:2]
            cx_img = W / 2

            # State machine
            if state == States.SEARCH_BOTTLE:
                if det and det.bottle_present and det.bottle_bbox:
                    x, y, w, h = det.bottle_bbox
                    target_cx = x + w/2
                    
                    # Approach bottle
                    robot.approach_target(target_cx, W, duration=0.1)
                    
                    if h >= bottle_close_h:
                        robot.stop()
                        robot.head_nod("excited")
                        state = States.GRAB_BOTTLE
                        phase_start = time.time()
                        print("[STATE] GRAB_BOTTLE")
                else:
                    # Keep scanning
                    robot.rotate(0.2, duration=0.12)

                if time.time() - phase_start > timeout_each:
                    print("[WARN] Timed out searching bottle; continuing to scan.")
                    phase_start = time.time()

            elif state == States.GRAB_BOTTLE:
                # Simulate bottle pickup (actual implementation would use arm controller)
                robot.drive(0.18, 0.18, duration=0.8)
                robot.stop()
                print("[ACTION] Picking up bottle")
                time.sleep(1.0)  # Simulate pickup time
                state = States.SEARCH_BIN
                phase_start = time.time()
                print("[STATE] SEARCH_BIN")

            elif state == States.SEARCH_BIN:
                if det and det.bin_present and det.bin_bbox:
                    x, y, w, h = det.bin_bbox
                    target_cx = x + w/2
                    
                    # Approach bin
                    robot.approach_target(target_cx, W, duration=0.1)
                    
                    if h >= bin_close_h:
                        robot.stop()
                        state = States.DROP_BOTTLE
                        phase_start = time.time()
                        print("[STATE] DROP_BOTTLE")
                else:
                    robot.rotate(-0.18, duration=0.12)

                if time.time() - phase_start > timeout_each:
                    print("[WARN] Timed out searching bin; continuing to scan.")
                    phase_start = time.time()

            elif state == States.DROP_BOTTLE:
                # Simulate bottle drop
                robot.drive(0.15, 0.15, duration=0.8)
                robot.stop()
                print("[ACTION] Dropping bottle in bin")
                time.sleep(1.0)  # Simulate drop time
                robot.drive(-0.25, -0.25, duration=0.9)
                robot.stop()
                robot.head_nod("acknowledgment")
                state = States.DONE
                print("[STATE] DONE")

            elif state == States.DONE:
                robot.stop()
                print("[INFO] Task complete!")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Stopping.")
    finally:
        robot.stop()
        camera.stop()


if __name__ == "__main__":
    main()
