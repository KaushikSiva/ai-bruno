#!/usr/bin/env python3
# coding: utf-8
"""
Bruno Intelligent Exploration (Fixed)
- Based on working headless_chicken.py obstacle avoidance
- Adds GPT Vision photos every 20 seconds with proper debugging
- Fixed timing and environment variable loading
"""

import os, sys, time, json, signal, logging, threading, urllib.request
import base64, io
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
# Config
# =========================
CONFIG = {
    # Camera stream
    "camera_url": os.environ.get("BRUNO_CAMERA_URL", "http://127.0.0.1:8080?action=stream"),
    "autostart_local_stream": True,
    "stream_port": 8080,

    # Ultrasonic thresholds (cm) - SAME AS HEADLESS_CHICKEN.PY
    "ultra_caution_cm": float(os.environ.get("ULTRA_CAUTION_CM", "50")),
    "ultra_danger_cm":  float(os.environ.get("ULTRA_DANGER_CM", "25")),

    # Drive parameters - SAME AS HEADLESS_CHICKEN.PY
    "forward_speed": int(os.environ.get("BRUNO_SPEED", "40")),
    "turn_speed":    int(os.environ.get("BRUNO_TURN_SPEED", "40")),
    "turn_time":     float(os.environ.get("BRUNO_TURN_TIME", "0.5")),
    "backup_time":   float(os.environ.get("BRUNO_BACKUP_TIME", "0.0")),

    # Vision - SAME AS HEADLESS_CHICKEN.PY
    "use_vision": True,
    "roi": {"top": 0.5, "bottom": 0.95, "left": 0.15, "right": 0.85},
    "edge_min_area": 800,
    "danger_px": 70,
    "avoid_px": 110,

    # GPT Vision settings - REDUCED INTERVAL FOR TESTING
    "gpt_photo_interval": 10,  # Every 10 seconds for faster testing
    "gpt_vision_prompt": "Describe what you see in this image. Focus on: objects, furniture, rooms, people, pets, bottles, bins/containers, obstacles, walls, and the general environment. Be concise but detailed.",
    "save_gpt_images": True,
    
    # Debug
    "save_debug": True,
    "debug_interval": 20,
}

# =========================
# Logging with more verbosity
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOG = logging.getLogger("bruno.intelligent")

# Check API key immediately
api_key = os.environ.get("OPENAI_API_KEY")
if api_key:
    LOG.info(f"✓ OPENAI_API_KEY found (length: {len(api_key)})")
else:
    LOG.warning("✗ OPENAI_API_KEY not found!")

# =========================
# Helpers (from headless_chicken.py)
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
# Ultrasonic with LED (EXACT COPY from headless_chicken.py)
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
# Vision (EXACT COPY from headless_chicken.py)
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
# GPT Vision with DETAILED DEBUGGING
# =========================
class GPTVision:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.enabled = False
        self.client = None
        
        LOG.info("🔍 Initializing GPT Vision...")
        
        # Check API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            LOG.warning("✗ OPENAI_API_KEY not found - GPT Vision DISABLED")
            return
        
        # Check OpenAI library
        if not OPENAI_AVAILABLE:
            LOG.warning("✗ OpenAI library not available - GPT Vision DISABLED")
            return
        
        # Initialize client
        try:
            self.client = OpenAI()
            self.enabled = True
            LOG.info("✓ GPT Vision ENABLED")
            LOG.info(f"✓ Photo interval: {cfg['gpt_photo_interval']} seconds")
        except Exception as e:
            LOG.error(f"✗ Failed to initialize OpenAI client: {e}")
            return
        
        # Timing
        self.last_photo_time = time.time()  # Initialize to current time
        self.photo_count = 0
        self.descriptions = []
        
        LOG.info(f"✓ GPT Vision ready - first photo in {cfg['gpt_photo_interval']} seconds")
    
    def should_take_photo(self) -> bool:
        """Check if it's time for a GPT photo with detailed logging"""
        if not self.enabled:
            return False
        
        current_time = time.time()
        elapsed = current_time - self.last_photo_time
        should_take = elapsed >= self.cfg["gpt_photo_interval"]
        
        # Log timing info every 50 calls (to avoid spam)
        if int(current_time) % 5 == 0:  # Every 5 seconds
            remaining = self.cfg["gpt_photo_interval"] - elapsed
            LOG.info(f"⏰ GPT Photo timing: {elapsed:.1f}s elapsed, {remaining:.1f}s remaining, should_take={should_take}")
        
        return should_take
    
    def capture_and_describe(self, frame: np.ndarray, current_action: str = "UNKNOWN") -> Optional[str]:
        """Capture frame and send to GPT Vision"""
        if not self.enabled:
            LOG.warning("GPT Vision not enabled - skipping photo")
            return None
        
        try:
            LOG.info("📸 TAKING GPT PHOTO NOW!")
            
            # Convert frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            
            # Save image
            if self.cfg.get("save_gpt_images", True):
                os.makedirs("gpt_images", exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"gpt_images/bruno_explore_{timestamp}_{self.photo_count:04d}.jpg"
                image.save(filename)
                LOG.info(f"💾 Image saved: {filename}")
            
            # Encode for GPT
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            jpeg_data = buffer.getvalue()
            base64_data = base64.b64encode(jpeg_data).decode('utf-8')
            image_data = f"data:image/jpeg;base64,{base64_data}"
            
            # Send to GPT
            LOG.info("🧠 Sending to GPT Vision API...")
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": self.cfg["gpt_vision_prompt"]
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data,
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            description = response.choices[0].message.content
            
            # Update counters
            self.photo_count += 1
            self.last_photo_time = time.time()  # IMPORTANT: Reset timer
            
            # Store description
            description_entry = {
                'timestamp': time.strftime("%H:%M:%S"),
                'photo_count': self.photo_count,
                'description': description,
                'current_action': current_action
            }
            self.descriptions.append(description_entry)
            
            # Display description
            LOG.info("\n" + "=" * 70)
            LOG.info("🔍 GPT VISION DESCRIPTION")
            LOG.info("=" * 70)
            LOG.info(f"Time: {description_entry['timestamp']} | Photo: #{self.photo_count} | Action: {current_action}")
            LOG.info(description)
            LOG.info("=" * 70)
            LOG.info(f"✓ Next photo in {self.cfg['gpt_photo_interval']} seconds")
            
            return description
            
        except Exception as e:
            LOG.error(f"❌ GPT Vision error: {e}")
            # Still reset timer to avoid getting stuck
            self.last_photo_time = time.time()
            return None

# =========================
# Main controller (PRESERVE headless_chicken.py logic)
# =========================
class BrunoIntelligentExplorer:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.car = mecanum.MecanumChassis()
        self.board = Board()
        self.AK = ArmIK(); self.AK.board = self.board

        # Arm init (same as headless_chicken.py)
        try:
            servo1 = 1500
            self.board.pwm_servo_set_position(0.3, [[1, servo1]])
            self.AK.setPitchRangeMoving((0, 6, 18), 0, -90, 90, 1500)
        except Exception as e:
            LOG.warning(f"Arm init skipped: {e}")

        self.ultra = UltrasonicRGB()
        self.gpt_vision = GPTVision(cfg)
        self.cap = None
        self.running = False
        self.frame_idx = 0

    def _autostart_stream_if_needed(self):
        cam_url = self.cfg["camera_url"]
        if self.cfg["autostart_local_stream"] and _looks_like_local(cam_url, self.cfg["stream_port"]):
            if not _is_stream_up(cam_url):
                LOG.info(f"Local stream {cam_url} not running. Launching live_stream_test...")
                _start_stream_background(host="0.0.0.0", port=self.cfg["stream_port"])
                ok = _wait_until(lambda: _is_stream_up(cam_url), timeout_s=8.0, interval_s=0.5)
                if not ok:
                    LOG.warning("Local stream did not come up in time")

    def _open_camera(self) -> bool:
        self._autostart_stream_if_needed()
        src = self.cfg["camera_url"]
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            try:
                self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            except Exception:
                pass
        if self.cap and self.cap.isOpened():
            return True
        LOG.error(f"Failed to open camera source: {src}")
        return False

    def stop_all(self):
        try:
            self.car.set_velocity(0, 0, 0)
        except Exception:
            pass

    def run(self):
        LOG.info("🤖 Starting Bruno Intelligent Explorer")
        LOG.info(f"📸 GPT Photos every {self.cfg['gpt_photo_interval']} seconds")
        LOG.info("🛡️  Obstacle avoidance: Ultrasonic primary + Vision secondary")
        
        if not self._open_camera():
            LOG.warning("Continuing without camera (ultrasonic-only mode)")
        
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

                # ============ ULTRASONIC OBSTACLE AVOIDANCE (EXACT COPY from headless_chicken.py) ============
                d_cm = self.ultra.get_distance_cm()
                current_action = "UNKNOWN"
                
                if d_cm is not None:
                    LOG.info(f"[ULTRA] {d_cm:.1f} cm")
                    
                    if d_cm <= self.cfg["ultra_danger_cm"]:
                        # EMERGENCY STOP (same as headless_chicken.py)
                        self.ultra.set_rgb(255, 0, 0)  # red
                        self.stop_all()
                        current_action = f"EMERGENCY STOP ({d_cm:.1f} cm)"
                        LOG.warning(current_action)
                        time.sleep(0.05)
                        continue
                        
                    elif d_cm <= self.cfg["ultra_caution_cm"]:
                        # AVOIDANCE (same as headless_chicken.py)
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
                        # SAFE
                        self.ultra.set_rgb(0, 255, 0)  # green

                # ============ VISION (same as headless_chicken.py) ============
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

                # ============ GPT VISION (NEW) ============
                if frame is not None and self.gpt_vision.should_take_photo():
                    # Stop Bruno before taking picture
                    LOG.info("🛑 Stopping Bruno for GPT photo...")
                    self.stop_all()
                    time.sleep(0.2)  # Brief pause to ensure full stop
                    
                    # Take photo and get GPT response
                    description = self.gpt_vision.capture_and_describe(frame, current_action)
                    
                    # Resume movement after GPT response
                    LOG.info("▶️  Resuming Bruno movement after GPT photo...")
                    continue  # Skip to next iteration to resume normal movement logic

                # Same loop timing as headless_chicken.py
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
RUNNER: Optional[BrunoIntelligentExplorer] = None

def _sig_handler(signum, frame):
    global RUNNER
    print("\n🛑 Ctrl-C received; stopping...")
    if RUNNER:
        RUNNER.shutdown()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, _sig_handler)
    LOG.info("🤖 Bruno Intelligent Explorer (Fixed)")
    LOG.info(f"📸 Photos every {CONFIG['gpt_photo_interval']} seconds")
    LOG.info("Press Ctrl+C to stop")

    RUNNER = BrunoIntelligentExplorer(CONFIG)
    RUNNER.run()