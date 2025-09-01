#!/usr/bin/env python3
# coding: utf-8
"""
Bruno Surveillance - External Camera Version
- Specifically designed for external USB cameras
- Handles V4L2 issues and camera access problems
- Robust camera detection and connection
"""

import os, sys, time, json, signal, logging, threading, urllib.request
import base64, io
from urllib.parse import urlparse
from typing import Optional, List, Dict
import subprocess

# Load environment variables from .env file
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value
    print(f"✓ Loaded .env file")

# --- Hiwonder SDK path ---
sys.path.append('/home/pi/MasterPi')

import cv2
import numpy as np
from PIL import Image

# Optional smoothing with pandas
try:
    import pandas as pd
    PANDAS_OK = True
except Exception:
    pd = None
    PANDAS_OK = False

# OpenAI for GPT Vision
try:
    from openai import OpenAI, APIError
    OPENAI_AVAILABLE = True
    print("✓ OpenAI library available")
except ImportError:
    print("✗ OpenAI library not available. Install with: pip install openai")
    OPENAI_AVAILABLE = False

# Hiwonder modules
import common.sonar as Sonar
import common.mecanum as mecanum
from common.ros_robot_controller_sdk import Board
from kinematics.transform import *
from kinematics.arm_move_ik import ArmIK

# =========================
# External Camera Helper Functions
# =========================
def get_available_cameras():
    """Get all available camera devices with detailed info"""
    cameras = []
    
    # Check /dev/video* devices
    for i in range(10):
        device_path = f"/dev/video{i}"
        if os.path.exists(device_path):
            # Get device info if possible
            try:
                # Check if device is accessible
                if os.access(device_path, os.R_OK):
                    cameras.append({
                        'path': device_path,
                        'index': i,
                        'accessible': True
                    })
                else:
                    cameras.append({
                        'path': device_path,
                        'index': i,
                        'accessible': False
                    })
            except Exception:
                pass
    
    return cameras

def fix_camera_permissions():
    """Attempt to fix camera permission issues"""
    print("🔧 Checking and fixing camera permissions...")
    
    try:
        # Add current user to video group (requires sudo)
        result = subprocess.run(['groups'], capture_output=True, text=True)
        if 'video' not in result.stdout:
            print("⚠️  User not in video group. Run: sudo usermod -a -G video $USER")
            print("   Then logout and login again")
        
        # Try to fix device permissions
        for i in range(5):
            device_path = f"/dev/video{i}"
            if os.path.exists(device_path) and not os.access(device_path, os.R_OK):
                try:
                    subprocess.run(['sudo', 'chmod', '666', device_path], timeout=5)
                    print(f"✅ Fixed permissions for {device_path}")
                except Exception:
                    print(f"⚠️  Could not fix permissions for {device_path}")
    except Exception as e:
        print(f"⚠️  Permission check failed: {e}")

def open_external_camera_robust(device_info):
    """Robustly open external camera with multiple fallback methods"""
    device_path = device_info['path']
    device_index = device_info['index']
    
    print(f"🎥 Attempting to open {device_path} (index {device_index})...")
    
    # Method list in order of preference for external USB cameras
    methods = [
        # Method 1: Direct index with V4L2 (most reliable for USB cameras)
        {
            'name': 'V4L2 with index',
            'func': lambda: cv2.VideoCapture(device_index, cv2.CAP_V4L2)
        },
        # Method 2: Direct path with V4L2
        {
            'name': 'V4L2 with path', 
            'func': lambda: cv2.VideoCapture(device_path, cv2.CAP_V4L2)
        },
        # Method 3: Index only (default backend)
        {
            'name': 'Default with index',
            'func': lambda: cv2.VideoCapture(device_index)
        },
        # Method 4: Path only (default backend)
        {
            'name': 'Default with path',
            'func': lambda: cv2.VideoCapture(device_path)
        },
        # Method 5: GStreamer pipeline (alternative for USB cameras)
        {
            'name': 'GStreamer pipeline',
            'func': lambda: cv2.VideoCapture(f'v4l2src device={device_path} ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
        },
    ]
    
    for method in methods:
        try:
            print(f"   Trying {method['name']}...")
            cap = method['func']()
            
            if cap and cap.isOpened():
                # Test frame capture
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    print(f"   ✅ {method['name']} SUCCESS! Resolution: {width}x{height}")
                    
                    # Set optimal properties for external cameras
                    try:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to avoid stale frames
                        print(f"   📐 Camera properties set")
                    except Exception as e:
                        print(f"   ⚠️  Could not set properties: {e}")
                    
                    return cap, method['name']
                else:
                    print(f"   ❌ {method['name']} opened but can't read frames")
                    cap.release()
            else:
                print(f"   ❌ {method['name']} failed to open")
                if cap:
                    cap.release()
                    
        except Exception as e:
            print(f"   ❌ {method['name']} error: {e}")
    
    return None, None

def find_working_external_camera():
    """Find the first working external camera"""
    print("🔍 Scanning for external cameras...")
    
    # First, try to fix permissions
    fix_camera_permissions()
    
    # Get available cameras
    cameras = get_available_cameras()
    
    if not cameras:
        print("❌ No camera devices found at /dev/video*")
        return None, None
    
    print(f"📹 Found {len(cameras)} camera device(s)")
    
    # Try each camera
    for camera in cameras:
        if not camera['accessible']:
            print(f"⚠️  {camera['path']} not accessible (permissions issue)")
            continue
            
        cap, method = open_external_camera_robust(camera)
        if cap:
            print(f"✅ Successfully opened {camera['path']} using {method}")
            return cap, camera
    
    print("❌ No working external cameras found")
    return None, None

# =========================
# Config
# =========================
CONFIG = {
    # External camera specific settings
    "external_camera_only": True,
    "camera_retry_attempts": 3,
    "camera_retry_delay": 2.0,
    
    # Standard surveillance settings
    "ultra_caution_cm": float(os.environ.get("ULTRA_CAUTION_CM", "50")),
    "ultra_danger_cm":  float(os.environ.get("ULTRA_DANGER_CM", "25")),
    "forward_speed": int(os.environ.get("BRUNO_SPEED", "40")),
    "turn_speed":    int(os.environ.get("BRUNO_TURN_SPEED", "40")),
    "turn_time":     float(os.environ.get("BRUNO_TURN_TIME", "0.5")),
    "backup_time":   float(os.environ.get("BRUNO_BACKUP_TIME", "0.0")),
    
    # Vision settings
    "use_vision": True,
    "roi": {"top": 0.5, "bottom": 0.95, "left": 0.15, "right": 0.85},
    "edge_min_area": 800,
    "danger_px": 70,
    "avoid_px": 110,
    
    # GPT Vision settings
    "gpt_photo_interval": 15,
    "gpt_vision_prompt": "Describe what you see in this image. Focus on: objects, furniture, rooms, people, pets, bottles, bins/containers, obstacles, walls, and the general environment. Be concise but detailed.",
    "save_gpt_images": True,
    
    "save_debug": True,
    "debug_interval": 20,
}

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOG = logging.getLogger("bruno.external_camera")

# =========================
# Ultrasonic with LED
# =========================
class UltrasonicRGB:
    def __init__(self, avg_samples: int = 5, sample_delay: float = 0.02):
        self.sonar = Sonar.Sonar()
        self.avg_samples = avg_samples
        self.sample_delay = sample_delay
        self.set_rgb(0, 0, 255)

    def get_distance_cm(self) -> Optional[float]:
        vals: List[float] = []
        for _ in range(self.avg_samples):
            try:
                d_cm = self.sonar.getDistance() / 10.0
                if 2.0 <= d_cm <= 400.0:
                    vals.append(float(d_cm))
            except Exception:
                pass
            time.sleep(self.sample_delay)
        if not vals:
            return None
        if len(vals) >= 4:
            vals.remove(max(vals)); vals.remove(min(vals))
        if PANDAS_OK and len(vals) >= 3:
            s = pd.Series(vals)
            m, std = float(s.mean()), float(s.std())
            if std > 0:
                s = s[np.abs(s - m) <= std]
            if len(s) > 0:
                return float(s.mean())
        return float(np.mean(vals))

    def set_rgb(self, r: int, g: int, b: int):
        try:
            self.sonar.setPixelColor(0, (r, g, b))
            self.sonar.setPixelColor(1, (r, g, b))
        except Exception:
            pass

# =========================
# Vision obstacles detection
# =========================
def _estimate_px_distance(w: int, h: int, y: int, H: int) -> float:
    size = (w * h) / (100 * 100)
    pos  = (H - y) / max(H, 1)
    prox = size * 0.6 + pos * 0.4
    return max(20, 200 - prox * 100)

def _vision_obstacles(frame: np.ndarray, cfg: Dict) -> List[Dict]:
    try:
        H, W = frame.shape[:2]
        r = cfg["roi"]
        y1, y2 = int(H*r["top"]), int(H*r["bottom"])
        x1, x2 = int(W*r["left"]), int(W*r["right"])
        roi = frame[y1:y2, x1:x2]
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (5, 5), 0)
        edges = cv2.Canny(g, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        obs = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < cfg["edge_min_area"]:
                continue
            x, y, w, h = cv2.boundingRect(c)
            fy = y + y1
            dpx = _estimate_px_distance(w, h, fy, H)
            cx = x + x1 + w / 2.0
            ang = np.degrees(np.arctan2(cx - W/2.0, W/2.0))
            if abs(ang) < 60:
                obs.append({"bbox": (x + x1, y + y1, w, h), "px": dpx, "angle": ang})
        return obs
    except Exception as e:
        LOG.warning(f"Vision detect failed: {e}")
        return []

# =========================
# GPT Vision (simplified for external camera)
# =========================
class GPTVision:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.enabled = False
        self.client = None
        
        LOG.info("🔍 Initializing GPT Vision...")
        
        # Check API key and initialize
        api_key = os.environ.get("OPENAI_API_KEY")
        LOG.info(f"📋 OPENAI_API_KEY status: {'Found' if api_key else 'NOT FOUND'}")
        LOG.info(f"📋 OpenAI library available: {OPENAI_AVAILABLE}")
        
        if api_key and OPENAI_AVAILABLE:
            try:
                self.client = OpenAI()
                self.enabled = True
                LOG.info("✅ GPT Vision ENABLED - photos will be taken")
                LOG.info(f"📸 Photo interval: {cfg['gpt_photo_interval']} seconds")
            except Exception as e:
                LOG.error(f"❌ Failed to initialize OpenAI client: {e}")
                self.enabled = False
        else:
            LOG.error("❌ GPT Vision DISABLED - missing API key or library")
            LOG.error("   Add OPENAI_API_KEY=your_key_here to .env file")
        
        self.last_photo_time = 0
        self.photo_count = 0
        
    def should_take_photo(self) -> bool:
        if not self.enabled:
            LOG.warning("🔍 GPT Vision disabled - not taking photos")
            return False
        
        current_time = time.time()
        elapsed = current_time - self.last_photo_time
        should_take = elapsed >= self.cfg["gpt_photo_interval"] or self.photo_count == 0
        
        # Enhanced debugging
        if self.photo_count == 0:
            LOG.info("📸 First photo ready - will take immediately")
            return True
        
        # Log every 3 seconds to show timing progress
        if int(elapsed) % 3 == 0 and elapsed < self.cfg["gpt_photo_interval"]:
            remaining = self.cfg["gpt_photo_interval"] - elapsed
            LOG.info(f"⏰ Next photo in {remaining:.0f} seconds... (elapsed: {elapsed:.1f}s)")
        
        if should_take:
            LOG.info(f"📸 Time to take photo #{self.photo_count + 1}!")
        
        return should_take
    
    def capture_and_describe(self, frame: np.ndarray, current_action: str = "UNKNOWN") -> Optional[str]:
        if not self.enabled:
            return None
        
        try:
            current_time = time.time()
            self.photo_count += 1
            
            LOG.info(f"📸 EXTERNAL CAMERA PHOTO #{self.photo_count}")
            
            # Convert and save
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            
            # Always save images for debugging (create gpt_images folder like other scripts)
            os.makedirs("gpt_images", exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            microseconds = int((current_time * 1000000) % 1000000)
            filename = f"gpt_images/external_camera_photo_{self.photo_count:03d}_{timestamp}_{microseconds:06d}.jpg"
            
            try:
                image.save(filename)
                LOG.info(f"💾 Image saved successfully: {filename}")
                LOG.info(f"📏 Image size: {image.size[0]}x{image.size[1]} pixels")
                
                # Verify file was actually created
                if os.path.exists(filename):
                    file_size = os.path.getsize(filename)
                    LOG.info(f"✅ File verified: {file_size} bytes")
                else:
                    LOG.error(f"❌ File not found after saving: {filename}")
                    
            except Exception as e:
                LOG.error(f"❌ Failed to save image: {e}")
            
            # Send to GPT
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            jpeg_data = buffer.getvalue()
            base64_data = base64.b64encode(jpeg_data).decode('utf-8')
            image_data = f"data:image/jpeg;base64,{base64_data}"
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [{
                        "type": "text", 
                        "text": self.cfg["gpt_vision_prompt"]
                    }, {
                        "type": "image_url",
                        "image_url": {"url": image_data, "detail": "low"}
                    }]
                }],
                max_tokens=300,
                temperature=0.1
            )
            
            description = response.choices[0].message.content
            self.last_photo_time = current_time
            
            LOG.info(f"🔍 EXTERNAL CAMERA ANALYSIS #{self.photo_count}:")
            LOG.info(f"   {description}")
            
            return description
            
        except Exception as e:
            LOG.error(f"❌ External camera GPT Vision error: {e}")
            self.last_photo_time = time.time()
            return None

# =========================
# Main External Camera Surveillance
# =========================
class BrunoExternalCameraSurveillance:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.car = mecanum.MecanumChassis()
        self.board = Board()
        self.AK = ArmIK(); self.AK.board = self.board
        
        self.ultra = UltrasonicRGB()
        self.gpt_vision = GPTVision(cfg)
        self.cap = None
        self.camera_info = None
        self.running = False
        self.frame_idx = 0

    def _open_camera(self) -> bool:
        LOG.info("🎥 Opening external camera...")
        
        for attempt in range(self.cfg["camera_retry_attempts"]):
            LOG.info(f"📹 Camera connection attempt {attempt + 1}...")
            
            self.cap, self.camera_info = find_working_external_camera()
            
            if self.cap:
                LOG.info(f"✅ External camera connected: {self.camera_info['path']}")
                return True
            
            if attempt < self.cfg["camera_retry_attempts"] - 1:
                LOG.info(f"⏳ Retrying in {self.cfg['camera_retry_delay']} seconds...")
                time.sleep(self.cfg["camera_retry_delay"])
        
        LOG.error("❌ Failed to connect external camera")
        return False

    def stop_all(self):
        try:
            self.car.set_velocity(0, 0, 0)
        except Exception:
            pass

    def run(self):
        LOG.info("🤖 Bruno External Camera Surveillance")
        LOG.info("📹 Designed specifically for external USB cameras")
        LOG.info("🛡️  Full obstacle avoidance with external camera vision")
        
        if not self._open_camera():
            LOG.error("❌ Cannot start without external camera")
            return
        
        self.running = True
        
        try:
            while self.running:
                self.frame_idx += 1
                frame = None
                
                # Get camera frame with error handling
                if self.cap and self.cap.isOpened():
                    ok, frame = self.cap.read()
                    if not ok or frame is None:
                        LOG.warning("📹 Camera disconnected, attempting reconnection...")
                        self.cap.release()
                        time.sleep(1)
                        self._open_camera()
                        continue

                # ============ ULTRASONIC OBSTACLE AVOIDANCE ============
                d_cm = self.ultra.get_distance_cm()
                current_action = "UNKNOWN"
                
                if d_cm is not None and d_cm <= self.cfg["ultra_danger_cm"]:
                    self.ultra.set_rgb(255, 0, 0)
                    self.stop_all()
                    current_action = f"EMERGENCY STOP ({d_cm:.1f} cm)"
                    LOG.warning(current_action)
                    time.sleep(0.05)
                    continue
                    
                elif d_cm is not None and d_cm <= self.cfg["ultra_caution_cm"]:
                    self.ultra.set_rgb(255, 180, 0)
                    
                    if self.cfg["backup_time"] > 0:
                        self.car.set_velocity(self.cfg["turn_speed"], 90, 0)
                        time.sleep(self.cfg["backup_time"])
                        self.stop_all()
                    
                    left = ((self.frame_idx // 60) % 2 == 0)
                    self.car.set_velocity(0, 90, -0.5 if left else 0.5)
                    time.sleep(self.cfg["turn_time"])
                    self.stop_all()
                    current_action = f"ULTRA AVOID ({d_cm:.1f} cm) {'LEFT' if left else 'RIGHT'}"
                    LOG.info(current_action)
                    continue
                else:
                    self.ultra.set_rgb(0, 255, 0)

                # ============ VISION OBSTACLE AVOIDANCE ============
                current_action = "FORWARD (ultra safe)"
                obs = []
                if frame is not None and self.cfg["use_vision"]:
                    obs = _vision_obstacles(frame, self.cfg)
                    if obs:
                        closest = min(obs, key=lambda o: o["px"])
                        px = closest["px"]
                        if px <= self.cfg["danger_px"]:
                            self.stop_all()
                            current_action = f"VISION STOP ({px:.1f}px)"
                            LOG.warning(current_action)
                        elif px <= self.cfg["avoid_px"]:
                            left = closest["angle"] >= 0
                            self.car.set_velocity(0, 90, -0.5 if left else 0.5)
                            time.sleep(self.cfg["turn_time"])
                            self.stop_all()
                            current_action = f"VISION AVOID ({px:.1f}px) {'LEFT' if left else 'RIGHT'}"
                            LOG.info(current_action)
                        else:
                            self.car.set_velocity(self.cfg["forward_speed"], 90, 0)
                            current_action = "FORWARD (vision safe)"
                    else:
                        self.car.set_velocity(self.cfg["forward_speed"], 90, 0)
                else:
                    self.car.set_velocity(self.cfg["forward_speed"], 90, 0)

                # ============ EXTERNAL CAMERA GPT VISION ============
                if frame is not None:
                    # Debug: Always check if we should take photo
                    should_take = self.gpt_vision.should_take_photo()
                    
                    if should_take:
                        LOG.info("🛑 STOPPING SURVEILLANCE for external camera photo session...")
                        self.stop_all()
                        time.sleep(0.3)
                        
                        # Flush camera buffer and get fresh frame
                        LOG.info("🔄 Flushing camera buffer...")
                        for i in range(3):
                            self.cap.read()  # Discard old frames
                            time.sleep(0.05)
                        
                        # Capture fresh frame
                        fresh_frame = None
                        for attempt in range(3):
                            ok, fresh_frame = self.cap.read()
                            if ok and fresh_frame is not None:
                                LOG.info(f"✅ Got fresh frame on attempt {attempt + 1}")
                                break
                            time.sleep(0.1)
                        
                        if fresh_frame is not None:
                            LOG.info(f"📸 Processing external camera photo #{self.gpt_vision.photo_count + 1}...")
                            description = self.gpt_vision.capture_and_describe(fresh_frame, current_action)
                            
                            if description:
                                LOG.info("✅ Photo processing completed successfully")
                            else:
                                LOG.error("❌ Photo processing failed")
                        else:
                            LOG.error("❌ Failed to capture fresh frame from external camera")
                            # Still update timer to prevent getting stuck
                            self.gpt_vision.last_photo_time = time.time()
                        
                        time.sleep(0.5)
                        LOG.info(f"▶️  RESUMING SURVEILLANCE for next {self.cfg['gpt_photo_interval']} seconds...")
                        continue
                else:
                    # Debug: Log when no frame available
                    if self.frame_idx % 100 == 0:  # Every ~3 seconds
                        LOG.warning("⚠️  No camera frame available for GPT Vision")

                time.sleep(0.03)

        except KeyboardInterrupt:
            LOG.info("Interrupted by user")
        finally:
            self.shutdown()

    def shutdown(self):
        LOG.info("Shutting down...")
        self.running = False
        self.stop_all()
        
        if self.cap:
            self.cap.release()
        
        self.ultra.set_rgb(0, 0, 0)

# =========================
# Entrypoint
# =========================
RUNNER: Optional[BrunoExternalCameraSurveillance] = None

def _sig_handler(signum, frame):
    global RUNNER
    print("\n🛑 Ctrl-C received; stopping...")
    if RUNNER:
        RUNNER.shutdown()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, _sig_handler)
    LOG.info("🤖 Bruno External Camera Surveillance System")
    LOG.info("📹 Optimized for external USB cameras")
    LOG.info("Press Ctrl+C to stop")

    RUNNER = BrunoExternalCameraSurveillance(CONFIG)
    RUNNER.run()