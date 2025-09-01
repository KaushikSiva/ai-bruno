#!/usr/bin/env python3
# coding: utf-8
"""
Bruno Local Surveillance System
- Integrates working hft.py local BLIP model
- Uses local GPT-OSS server on port 1234 for enhanced context
- Two-stage analysis: BLIP caption → GPT-OSS full context
- Auto-detects cameras (built-in or external)
"""

import os, sys, time, json, signal, logging, threading, urllib.request
import requests
from urllib.parse import urlparse
from typing import Optional, List, Dict

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

# Import local BLIP captioning from hft.py
try:
    from hft import get_caption
    print("✓ Local BLIP captioning from hft.py loaded")
    HFT_AVAILABLE = True
except ImportError:
    print("✗ Could not import hft.py - please ensure it's in the same directory")
    HFT_AVAILABLE = False

# Optional smoothing with pandas
try:
    import pandas as pd
    PANDAS_OK = True
except Exception:
    pd = None
    PANDAS_OK = False

# Hiwonder modules
import common.sonar as Sonar
import common.mecanum as mecanum
from common.ros_robot_controller_sdk import Board
from kinematics.transform import *
from kinematics.arm_move_ik import ArmIK

# =========================
# Config
# =========================
def _get_camera_sources():
    """Get list of camera sources, checking which devices exist first"""
    sources = []
    
    # 1. External camera from environment (highest priority)
    stream_source = os.environ.get("STREAM_SOURCE")
    if stream_source:
        sources.append(stream_source)
    
    # 2. Built-in camera stream
    sources.append("http://127.0.0.1:8080?action=stream")
    
    # 3. Only add USB device indices that actually exist
    for i in range(5):
        if _check_device_exists(i):
            sources.append(i)
    
    # 4. Only add device paths that exist
    for i in range(5):
        device_path = f"/dev/video{i}"
        if os.path.exists(device_path):
            sources.append(device_path)
    
    return sources

CONFIG = {
    # Camera settings - AUTO-DETECT built-in vs external
    "camera_sources": [],  # Will be populated by _get_camera_sources()
    "autostart_local_stream": True,
    "stream_port": 8080,
    "stream_width": int(os.environ.get("STREAM_WIDTH", "1280")),
    "stream_height": int(os.environ.get("STREAM_HEIGHT", "720")),
    "stream_fps": int(os.environ.get("STREAM_FPS", "30")),

    # Ultrasonic thresholds (cm) - SAME AS WORKING SCRIPT
    "ultra_caution_cm": float(os.environ.get("ULTRA_CAUTION_CM", "50")),
    "ultra_danger_cm":  float(os.environ.get("ULTRA_DANGER_CM", "25")),

    # Drive parameters - SAME AS WORKING SCRIPT
    "forward_speed": int(os.environ.get("BRUNO_SPEED", "40")),
    "turn_speed":    int(os.environ.get("BRUNO_TURN_SPEED", "40")),
    "turn_time":     float(os.environ.get("BRUNO_TURN_TIME", "0.5")),
    "backup_time":   float(os.environ.get("BRUNO_BACKUP_TIME", "0.0")),

    # Vision - SAME AS WORKING SCRIPT
    "use_vision": True,
    "roi": {"top": 0.5, "bottom": 0.95, "left": 0.15, "right": 0.85},
    "edge_min_area": 800,
    "danger_px": 70,
    "avoid_px": 110,

    # Local Vision settings
    "vision_photo_interval": 15,  # Every 15 seconds
    "save_vision_images": True,
    
    # Local GPT-OSS server settings
    "gpt_oss_host": os.environ.get("GPT_OSS_HOST", "localhost"),
    "gpt_oss_port": int(os.environ.get("GPT_OSS_PORT", "1234")),
    "gpt_oss_timeout": 30,
    
    # Debug
    "save_debug": True,
    "debug_interval": 20,
}

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOG = logging.getLogger("bruno.local_surveillance")

# =========================
# Camera Helper Functions
# =========================
def _check_device_exists(device_id: int) -> bool:
    """Check if a camera device exists without opening it fully"""
    try:
        device_path = f"/dev/video{device_id}"
        return os.path.exists(device_path)
    except Exception:
        return False

def _parse_source(val):
    """Parse camera source - int for device ID, string for path/URL"""
    if isinstance(val, int):
        return val
    try:
        return int(val)
    except Exception:
        return val

def _open_camera_source(source) -> Optional[cv2.VideoCapture]:
    """Open a camera source with multiple backend attempts"""
    LOG.info(f"Trying camera source: {source}")
    
    cap = None
    source = _parse_source(source)
    
    try:
        if isinstance(source, int):
            # Check if device exists first to avoid "Camera index out of range" error
            if not _check_device_exists(source):
                LOG.warning(f"Camera device {source} does not exist (/dev/video{source} not found)")
                return None
            
            # USB camera device - try V4L2 first with error suppression
            try:
                cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(source)
            except Exception as e:
                LOG.warning(f"Failed to open camera {source}: {e}")
                return None
                
        elif isinstance(source, str):
            if source.startswith(('http://', 'https://', 'rtsp://')):
                # Network stream
                cap = cv2.VideoCapture(source)
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            else:
                # File or device path - check if path exists first
                if source.startswith('/dev/video'):
                    if not os.path.exists(source):
                        LOG.warning(f"Device path {source} does not exist")
                        return None
                
                cap = cv2.VideoCapture(source)
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        
        if cap and cap.isOpened():
            # Set camera properties with error handling
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["stream_width"])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["stream_height"])
                cap.set(cv2.CAP_PROP_FPS, CONFIG["stream_fps"])
            except Exception as e:
                LOG.warning(f"Could not set camera properties: {e}")
            
            # Test read with timeout
            try:
                ok, frame = cap.read()
                if ok and frame is not None:
                    LOG.info(f"✓ Camera source {source} working ({frame.shape[1]}x{frame.shape[0]})")
                    return cap
                else:
                    LOG.warning(f"Camera {source} opened but failed to read frame")
            except Exception as e:
                LOG.warning(f"Camera {source} test read failed: {e}")
        
        if cap:
            cap.release()
        return None
        
    except Exception as e:
        LOG.warning(f"Camera source {source} failed with exception: {e}")
        if cap:
            try:
                cap.release()
            except Exception:
                pass
        return None

def _find_working_camera() -> Optional[cv2.VideoCapture]:
    """Try all camera sources and return the first working one"""
    LOG.info("🔍 Auto-detecting camera...")
    
    # Initialize camera sources if not done yet
    if not CONFIG["camera_sources"]:
        CONFIG["camera_sources"] = _get_camera_sources()
        LOG.info(f"Available camera sources: {CONFIG['camera_sources']}")
    
    for source in CONFIG["camera_sources"]:
        cap = _open_camera_source(source)
        if cap:
            LOG.info(f"✓ Using camera source: {source}")
            return cap
    
    LOG.error("❌ No working camera found")
    return None

# =========================
# Stream Helper Functions
# =========================
def _is_stream_up(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False

def _looks_like_local(url: str, default_port: int = 8080) -> bool:
    try:
        u = urlparse(url)
        if u.scheme not in ("http", "https"):
            return False
        host = (u.hostname or "").lower()
        port = u.port or (443 if u.scheme == "https" else 80)
        return host in ("127.0.0.1", "localhost") and port == default_port
    except Exception:
        return False

def _start_stream_background(host="0.0.0.0", port=8080):
    try:
        import live_stream_test
    except Exception as e:
        LOG.error(f"Could not import live_stream_test for autostart: {e}")
        return None
    t = threading.Thread(target=live_stream_test.run_stream, kwargs={"host": host, "port": port}, daemon=True)
    t.start()
    return t

def _wait_until(pred, timeout_s: float, interval_s: float = 0.25) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if pred():
            return True
        time.sleep(interval_s)
    return False

# =========================
# Ultrasonic with LED (EXACT COPY from working script)
# =========================
class UltrasonicRGB:
    def __init__(self, avg_samples: int = 5, sample_delay: float = 0.02):
        self.sonar = Sonar.Sonar()
        self.avg_samples = avg_samples
        self.sample_delay = sample_delay
        # heartbeat (blue) so you see it's alive at start
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
        except Exception:
            pass
        try:
            self.sonar.setPixelColor(1, (r, g, b))
        except Exception:
            pass

# =========================
# Vision (EXACT COPY from working script)
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
# Local Vision System (BLIP + GPT-OSS)
# =========================
class LocalVisionSystem:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.enabled = False
        
        LOG.info("🔍 Initializing Local Vision System...")
        
        # Check if HFT (local BLIP) is available
        if not HFT_AVAILABLE:
            LOG.error("✗ hft.py not available - Local Vision DISABLED")
            return
        
        # Test local BLIP model
        try:
            # Test with a dummy call to ensure model is loaded
            LOG.info("🧠 Testing local BLIP model...")
            test_result = get_caption("https://via.placeholder.com/150")
            if "error" in test_result.lower():
                LOG.warning(f"BLIP model test warning: {test_result}")
            else:
                LOG.info("✓ Local BLIP model is working")
        except Exception as e:
            LOG.error(f"✗ BLIP model test failed: {e}")
        
        # Test GPT-OSS server connection
        gpt_url = f"http://{cfg['gpt_oss_host']}:{cfg['gpt_oss_port']}"
        try:
            LOG.info(f"🌐 Testing GPT-OSS server at {gpt_url}...")
            # Test if server is running
            response = requests.get(f"{gpt_url}/health", timeout=5)
            if response.status_code == 200:
                LOG.info("✓ GPT-OSS server is responding")
            else:
                LOG.warning(f"GPT-OSS server responded with status {response.status_code}")
        except requests.exceptions.ConnectionError:
            LOG.warning(f"⚠️  Could not connect to GPT-OSS at {gpt_url}")
            LOG.warning("   Make sure your local GPT-OSS server is running on port 1234")
        except Exception as e:
            LOG.warning(f"GPT-OSS test failed: {e}")
        
        # Enable if BLIP is working (GPT-OSS is optional enhancement)
        self.enabled = HFT_AVAILABLE
        
        # Timing
        self.last_photo_time = 0  # Force immediate first photo
        self.photo_count = 0
        self.descriptions = []
        self.surveillance_start_time = time.time()
        
        if self.enabled:
            LOG.info("✓ Local Vision System ready")
            LOG.info(f"✓ Photo interval: {cfg['vision_photo_interval']} seconds")
        
    def should_take_photo(self) -> bool:
        if not self.enabled:
            return False
        
        current_time = time.time()
        elapsed = current_time - self.last_photo_time
        should_take = elapsed >= self.cfg["vision_photo_interval"]
        
        # Special case: first photo immediately when surveillance starts
        if self.photo_count == 0:
            should_take = True
            LOG.info("📸 Taking first surveillance photo immediately...")
        
        # Log countdown every 5 seconds (but not spam)
        elif int(current_time) % 5 == 0 and elapsed > 5:
            remaining = self.cfg["vision_photo_interval"] - elapsed
            if remaining > 0:
                LOG.info(f"⏰ Next photo in {remaining:.0f} seconds...")
        
        return should_take
    
    def _query_gpt_oss(self, blip_caption: str, current_action: str) -> str:
        """Query local GPT-OSS server for enhanced context analysis"""
        gpt_url = f"http://{self.cfg['gpt_oss_host']}:{self.cfg['gpt_oss_port']}"
        
        # Create enhanced prompt for GPT-OSS
        enhanced_prompt = f"""
You are analyzing a surveillance image from Bruno, an autonomous robot.

BLIP Caption: {blip_caption}
Current Robot Action: {current_action}

Please provide a detailed surveillance analysis including:
1. Environment description and context
2. Any objects, people, or activities detected
3. Potential security concerns or points of interest
4. Recommendations for the surveillance system

Keep the response concise but informative for security monitoring purposes.
"""
        
        try:
            payload = {
                "model": "gpt-3.5-turbo",  # Adjust based on your GPT-OSS setup
                "messages": [
                    {
                        "role": "user",
                        "content": enhanced_prompt
                    }
                ],
                "max_tokens": 300,
                "temperature": 0.3
            }
            
            LOG.info("🧠 Querying local GPT-OSS for enhanced analysis...")
            response = requests.post(
                f"{gpt_url}/v1/chat/completions", 
                json=payload, 
                timeout=self.cfg["gpt_oss_timeout"]
            )
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    enhanced_context = result["choices"][0]["message"]["content"]
                    LOG.info("✅ GPT-OSS enhanced analysis received")
                    return enhanced_context
                else:
                    LOG.warning("GPT-OSS returned unexpected format")
                    return blip_caption  # Fallback to BLIP caption
            else:
                LOG.warning(f"GPT-OSS returned status {response.status_code}: {response.text}")
                return blip_caption  # Fallback to BLIP caption
                
        except requests.exceptions.ConnectionError:
            LOG.warning("Could not connect to GPT-OSS server - using BLIP caption only")
            return blip_caption
        except Exception as e:
            LOG.warning(f"GPT-OSS query failed: {e}")
            return blip_caption
    
    def capture_and_analyze(self, frame: np.ndarray, current_action: str = "UNKNOWN") -> Optional[str]:
        if not self.enabled:
            LOG.warning("Local Vision not enabled - skipping photo")
            return None
        
        try:
            # Immediately update timing to ensure fresh 15-second cycle
            current_time = time.time()
            self.photo_count += 1
            
            LOG.info(f"📸 CAPTURING FRESH PHOTO #{self.photo_count} - Action: {current_action}")
            
            # Convert current frame to RGB and save
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            
            # Save image with clear numbering
            temp_image_path = None
            if self.cfg.get("save_vision_images", True):
                os.makedirs("vision_images", exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                temp_image_path = f"vision_images/surveillance_photo_{self.photo_count:03d}_{timestamp}.jpg"
                image.save(temp_image_path)
                LOG.info(f"💾 Image #{self.photo_count} saved: {temp_image_path}")
            
            # Stage 1: Get BLIP caption using hft.py
            LOG.info(f"🧠 Stage 1: Getting BLIP caption...")
            if temp_image_path:
                blip_caption = get_caption(temp_image_path)
            else:
                # Save temporarily for BLIP analysis
                temp_image_path = f"temp_surveillance_{self.photo_count}.jpg"
                image.save(temp_image_path)
                blip_caption = get_caption(temp_image_path)
                try:
                    os.remove(temp_image_path)  # Clean up
                except:
                    pass
            
            LOG.info(f"✅ BLIP Caption: {blip_caption}")
            
            # Stage 2: Enhance with GPT-OSS
            LOG.info(f"🧠 Stage 2: Enhancing with GPT-OSS...")
            enhanced_context = self._query_gpt_oss(blip_caption, current_action)
            
            # Reset timer AFTER successful analysis
            self.last_photo_time = current_time
            
            # Store analysis with surveillance info
            analysis_entry = {
                'timestamp': time.strftime("%H:%M:%S"),
                'photo_count': self.photo_count,
                'blip_caption': blip_caption,
                'enhanced_context': enhanced_context,
                'current_action': current_action,
                'surveillance_time': current_time - self.surveillance_start_time
            }
            self.descriptions.append(analysis_entry)
            
            # Display analysis
            LOG.info("\n" + "=" * 80)
            LOG.info(f"🔍 SURVEILLANCE PHOTO #{self.photo_count} ANALYSIS")
            LOG.info("=" * 80)
            LOG.info(f"Time: {analysis_entry['timestamp']} | Action: {current_action}")
            LOG.info(f"Surveillance Duration: {analysis_entry['surveillance_time']:.1f}s")
            LOG.info("")
            LOG.info(f"📷 BLIP Caption: {blip_caption}")
            LOG.info("")
            LOG.info(f"🧠 GPT-OSS Enhanced Context:")
            LOG.info(enhanced_context)
            LOG.info("")
            LOG.info("=" * 80)
            LOG.info(f"✅ Analysis complete. Resuming surveillance for {self.cfg['vision_photo_interval']} seconds...")
            LOG.info("=" * 80)
            
            return enhanced_context
            
        except Exception as e:
            LOG.error(f"❌ Local Vision error on photo #{self.photo_count}: {e}")
            # Reset timer even on error to continue surveillance cycle
            self.last_photo_time = time.time()
            return None

# =========================
# Main Bruno Local Surveillance Controller
# =========================
class BrunoLocalSurveillance:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.car = mecanum.MecanumChassis()
        self.board = Board()
        self.AK = ArmIK(); self.AK.board = self.board

        # Arm init (same as working script)
        try:
            servo1 = 1500
            self.board.pwm_servo_set_position(0.3, [[1, servo1]])
            self.AK.setPitchRangeMoving((0, 6, 18), 0, -90, 90, 1500)
        except Exception as e:
            LOG.warning(f"Arm init skipped: {e}")

        self.ultra = UltrasonicRGB()
        self.vision_system = LocalVisionSystem(cfg)
        self.cap = None
        self.running = False
        self.frame_idx = 0

    def _autostart_stream_if_needed(self):
        """Start local stream if needed for built-in camera"""
        cam_url = "http://127.0.0.1:8080?action=stream"
        if self.cfg["autostart_local_stream"] and _looks_like_local(cam_url, self.cfg["stream_port"]):
            if not _is_stream_up(cam_url):
                LOG.info(f"Local stream {cam_url} not running. Launching live_stream_test...")
                _start_stream_background(host="0.0.0.0", port=self.cfg["stream_port"])
                ok = _wait_until(lambda: _is_stream_up(cam_url), timeout_s=8.0, interval_s=0.5)
                if not ok:
                    LOG.warning("Local stream did not come up in time")

    def _open_camera(self) -> bool:
        """Open camera with auto-detection and fallbacks"""
        LOG.info("🎥 Setting up camera...")
        
        # Try to start local stream first (for built-in camera users)
        self._autostart_stream_if_needed()
        
        # Find working camera
        self.cap = _find_working_camera()
        if self.cap:
            LOG.info("✓ Camera ready")
            return True
        
        LOG.error("❌ No camera available")
        return False

    def stop_all(self):
        try:
            self.car.set_velocity(0, 0, 0)
        except Exception:
            pass

    def run(self):
        LOG.info("🤖 Starting Bruno Local Surveillance System")
        LOG.info("🎥 Auto-detecting camera (built-in or external)")
        LOG.info(f"📸 Local Vision Analysis every {self.cfg['vision_photo_interval']} seconds")
        LOG.info("🧠 Two-stage analysis: BLIP → GPT-OSS")
        LOG.info("🛡️  Obstacle avoidance: Ultrasonic primary + Vision secondary")
        
        if not self._open_camera():
            LOG.warning("⚠️  Continuing without camera (ultrasonic-only mode)")
        
        self.running = True
        
        try:
            while self.running:
                self.frame_idx += 1
                frame = None
                
                # Get camera frame
                if self.cap and self.cap.isOpened():
                    ok, frame = self.cap.read()
                    if not ok: 
                        frame = None
                        LOG.warning("Camera read failed, attempting reconnection...")
                        self.cap.release()
                        time.sleep(1)
                        self.cap = _find_working_camera()

                # ============ ULTRASONIC OBSTACLE AVOIDANCE ============
                d_cm = self.ultra.get_distance_cm()
                current_action = "UNKNOWN"
                
                if d_cm is not None and d_cm <= self.cfg["ultra_danger_cm"]:
                    # EMERGENCY STOP
                    self.ultra.set_rgb(255, 0, 0)  # red
                    self.stop_all()
                    current_action = f"EMERGENCY STOP ({d_cm:.1f} cm)"
                    LOG.warning(current_action)
                    time.sleep(0.05)
                    continue
                    
                elif d_cm is not None and d_cm <= self.cfg["ultra_caution_cm"]:
                    # AVOIDANCE MANEUVER
                    self.ultra.set_rgb(255, 180, 0)  # amber
                    
                    # Optional backup
                    if self.cfg["backup_time"] > 0:
                        self.car.set_velocity(self.cfg["turn_speed"], 90, 0)
                        time.sleep(self.cfg["backup_time"])
                        self.stop_all()
                    
                    # Alternate turn direction
                    left = ((self.frame_idx // 60) % 2 == 0)
                    self.car.set_velocity(0, 90, -0.5 if left else 0.5)
                    time.sleep(self.cfg["turn_time"])
                    self.stop_all()
                    current_action = f"ULTRA AVOID ({d_cm:.1f} cm) {'LEFT' if left else 'RIGHT'}"
                    LOG.info(current_action)
                    continue
                else:
                    # SAFE TO MOVE FORWARD
                    self.ultra.set_rgb(0, 255, 0)  # green

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
                    # No vision - just go forward if ultrasonic is safe
                    self.car.set_velocity(self.cfg["forward_speed"], 90, 0)

                # ============ LOCAL VISION ANALYSIS (BLIP + GPT-OSS) ============
                if frame is not None and self.vision_system.should_take_photo():
                    # Stop Bruno for analysis session
                    LOG.info("🛑 STOPPING SURVEILLANCE for vision analysis...")
                    self.stop_all()
                    time.sleep(0.3)  # Ensure complete stop
                    
                    # Two-stage analysis: BLIP → GPT-OSS
                    enhanced_context = self.vision_system.capture_and_analyze(frame, current_action)
                    
                    # Brief pause to show analysis completed, then resume
                    time.sleep(0.5)
                    LOG.info(f"▶️  RESUMING SURVEILLANCE for next {self.cfg['vision_photo_interval']} seconds...")
                    continue  # Skip to next iteration to resume movement

                # Same loop timing as working script
                time.sleep(0.03)

        except KeyboardInterrupt:
            LOG.info("Interrupted by user")
        finally:
            self.shutdown()

    def shutdown(self):
        LOG.info("Shutting down...")
        self.running = False
        self.stop_all()
        
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        
        try:
            self.ultra.set_rgb(0, 0, 0)
        except Exception:
            pass

# =========================
# Entrypoint
# =========================
RUNNER: Optional[BrunoLocalSurveillance] = None

def _sig_handler(signum, frame):
    global RUNNER
    print("\n🛑 Ctrl-C received; stopping...")
    if RUNNER:
        RUNNER.shutdown()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, _sig_handler)
    LOG.info("🤖 Bruno Local Surveillance System")
    LOG.info(f"📸 Vision analysis every {CONFIG['vision_photo_interval']} seconds")
    LOG.info("🎥 Auto-detecting camera (built-in or external)")
    LOG.info("🧠 Local BLIP + GPT-OSS enhanced analysis")
    LOG.info(f"🌐 GPT-OSS server: http://{CONFIG['gpt_oss_host']}:{CONFIG['gpt_oss_port']}")
    LOG.info("Press Ctrl+C to stop")

    RUNNER = BrunoLocalSurveillance(CONFIG)
    RUNNER.run()